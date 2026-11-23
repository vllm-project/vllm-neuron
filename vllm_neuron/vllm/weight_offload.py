# SPDX-License-Identifier: Apache-2.0
"""Neuron model weight offload for the vLLM ``sleep``/``wake_up`` API.

The upstream GPU worker implements ``sleep``/``wake_up`` on top of a
``DeviceAllocator`` (CUDA ``cumem``/``xpumem``): sleeping offloads the whole
device pool to host memory and frees the device side, and waking reallocates
the pool and copies the tensors back. Neuron has no equivalent pool allocator,
so this module provides the Neuron-specific weight offload directly.

The one rule the Neuron backend must respect is the same one the weight-transfer
backend (A2) enforces: the model is a compiled graph (``torch.compile`` +
neuronx-cc), and the compiled NEFF reads the model's parameter tensors *by
address*. Offloading therefore never reassigns a parameter's storage. It copies
each tensor's values to CPU and leaves the device tensor untouched, so the
addresses the compiled graph reads stay valid. Restoring copies the saved values
back into those same tensors, in place.

This offload moves the weight *values* to CPU. It does not reclaim their HBM
allocation: freeing the HBM while a compiled graph still binds to the address is
a Neuron-runtime concern (unloading the NEFF), not a tensor-level operation, and
is a follow-up. The device tensor is deliberately left intact — an in-place
``zero_()`` is unsupported on Neuron shared-storage tensors (it raises
``Can't call ReserveSpace on shared storage``) and would not reclaim HBM anyway.
``sleep`` here is therefore the "params to CPU during training, reload during
rollout" half of the colocation story; the memory-reclamation half needs runtime
support.
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Tag the offload covers. ``kv_cache`` is intentionally absent: vLLM Neuron
# marks KV offloading unsupported (see ``model_runner_v2_policy.py``), so sleep
# on Neuron only ever moves weights.
_WEIGHTS_TAG = "weights"


def _is_offloadable(tensor: torch.Tensor) -> bool:
    """Return whether a tensor lives on a device whose memory can be offloaded.

    CPU tensors are already on host memory. ``meta`` tensors (CPU-compile mode)
    have no storage to offload.
    """
    return tensor.device.type not in ("cpu", "meta")


class NeuronWeightOffloader:
    """Offload and restore a model's weights and buffers across ``sleep``/``wake_up``.

    Holds the CPU copies of every offloaded tensor so the worker can restore the
    resident model in place without touching the compiled graph's addresses.
    """

    def __init__(self) -> None:
        self._saved_parameters: dict[str, torch.Tensor] = {}
        self._saved_buffers: dict[str, torch.Tensor] = {}

    @property
    def is_sleeping(self) -> bool:
        """Return whether an offload is currently held."""
        return bool(self._saved_parameters)

    @staticmethod
    def _inner(model: nn.Module) -> nn.Module:
        """Return the uncompiled module, unwrapping a ``torch.compile`` wrapper."""
        return getattr(model, "_orig_mod", model)

    @staticmethod
    def _to_host(tensor: torch.Tensor) -> torch.Tensor:
        """Copy a device tensor's values into an independent host tensor."""
        return tensor.detach().to("cpu")

    @staticmethod
    def _from_host(
        value: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Move a host tensor back onto ``device`` at ``dtype``."""
        return value.to(device=device, dtype=dtype)

    def _offload_tensors(
        self, tensors: Iterable[tuple[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Copy each offloadable tensor to CPU.

        The device tensor is left intact (no in-place zero), both to preserve the
        addresses a compiled graph reads and because in-place zeroing is
        unsupported on Neuron shared-storage tensors and does not reclaim HBM.

        Args:
            tensors: ``(name, tensor)`` pairs from a module's parameters or buffers.

        Returns:
            A ``name -> CPU tensor`` map of the saved values.
        """
        saved: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, tensor in tensors:
                if not _is_offloadable(tensor):
                    continue
                saved[name] = self._to_host(tensor)
        return saved

    def offload(self, model: nn.Module, level: int = 1) -> None:
        """Offload a model's weights (and, at level 2, buffers) to CPU.

        Saves every device parameter's values to CPU. Level 2 additionally does
        the same for buffers. Device tensors are left untouched (see
        :meth:`_offload_tensors`).

        Args:
            model: The resident model (a ``torch.compile`` wrapper is accepted).
            level: Sleep level. 1 offloads parameters only; 2 also offloads buffers.

        Raises:
            RuntimeError: If an offload is already held; call :meth:`restore` or
                :meth:`discard` first so the saved weights are not lost.
        """
        if self.is_sleeping:
            raise RuntimeError(
                "Model is already offloaded. Call restore() or discard() first "
                "so the saved weights are not overwritten."
            )

        inner = self._inner(model)
        self._saved_parameters = self._offload_tensors(inner.named_parameters())
        if level >= 2:
            self._saved_buffers = self._offload_tensors(inner.named_buffers())

        logger.info(
            "Offloaded %d parameter tensor(s) and %d buffer(s) to CPU (level %d).",
            len(self._saved_parameters),
            len(self._saved_buffers),
            level,
        )

    def restore(self, model: nn.Module, tags: list[str] | None = None) -> None:
        """Restore offloaded tensors in place.

        Copies each saved value back into the matching tensor's existing storage,
        preserving the address the compiled graph reads. Only the ``weights`` tag
        is restored; the ``kv_cache`` tag is a no-op on Neuron (KV offloading is
        unsupported).

        Args:
            model: The resident model to restore into.
            tags: Tags selecting what to restore, or ``None`` to restore all.
        """
        wake_weights = tags is None or _WEIGHTS_TAG in tags
        if not wake_weights:
            return

        inner = self._inner(model)
        parameters = dict(inner.named_parameters())
        buffers = dict(inner.named_buffers())

        with torch.no_grad():
            for name, value in self._saved_parameters.items():
                param = parameters.get(name)
                if param is None:
                    logger.warning(
                        "No parameter '%s' in the resident model; skipping restore.",
                        name,
                    )
                    continue
                param.data.copy_(self._from_host(value, param.device, param.dtype))
            for name, value in self._saved_buffers.items():
                buffer = buffers.get(name)
                if buffer is None:
                    logger.warning(
                        "No buffer '%s' in the resident model; skipping restore.",
                        name,
                    )
                    continue
                buffer.data.copy_(self._from_host(value, buffer.device, buffer.dtype))

        logger.info(
            "Restored %d parameter tensor(s) and %d buffer(s) from CPU.",
            len(self._saved_parameters),
            len(self._saved_buffers),
        )
        self._saved_parameters = {}
        self._saved_buffers = {}

    def discard(self, tags: tuple[str, ...] | None = None) -> None:
        """Drop saved CPU tensors without restoring them.

        Frees the host memory held by the offload. Only the ``weights`` tag is
        discarded; the ``kv_cache`` tag is a no-op on Neuron.

        Args:
            tags: Tags selecting what to discard, or ``None`` to discard all.
        """
        if tags is not None and _WEIGHTS_TAG not in tags:
            return
        self._saved_parameters = {}
        self._saved_buffers = {}
        logger.info("Discarded offloaded weights from CPU.")
