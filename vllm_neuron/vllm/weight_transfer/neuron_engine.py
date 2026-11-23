# SPDX-License-Identifier: Apache-2.0
"""Neuron weight-transfer engine (inference-worker side).

Implements vLLM's :class:`~vllm.distributed.weight_transfer.base.WeightTransferEngine`
contract for AWS Trainium/Neuron.

The one Neuron-specific rule this backend enforces is *in-place* weight
application. On GPU the reference engines hand received tensors to
``model.load_weights``, which may swap parameter storage. On Neuron the model is
a compiled graph (``torch.compile`` + neuronx-cc); the compiled NEFF reads the
model's parameter tensors by address. Swapping a parameter therefore leaves the
compiled graph reading stale weights, while ``param.data.copy_`` mutates the
same storage in place and keeps the graph valid. Every received tensor is
therefore written with ``param.data.copy_``.

Transport is a store-based rendezvous between the trainer process (rank 0) and
the inference workers (ranks 1..N), built on vLLM's ``StatelessProcessGroup``.
This is the correctness-first data plane: it needs no NCCL and no Neuron
runtime, so it initializes on a compile host too, and it matches the "pinned
CPU staging buffer" pattern — weights arrive on host memory and are copied into
the resident device tensors. A faster device/gloo data plane is a follow-up;
the per-name broadcast loop below is the seam where it plugs in.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import torch

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Maps a trainer-side parameter name to the resident model's parameter name.
# Defaults to identity: the trainer is expected to ship weights keyed by the
# neuron model's own ``named_parameters()`` names (the verl fork owns any
# HF -> neuron fused-name translation, e.g. ``q_proj``+``k_proj``+``v_proj`` ->
# ``qkv_proj_weight``). A callable override can be injected for checkpoint-format
# trainers.
NameMap = Callable[[str], str]


def _identity_name_map(name: str) -> str:
    return name


@dataclass
class NeuronWeightTransferInitInfo(WeightTransferInitInfo):
    """Worker-side init info for the Neuron weight-transfer backend.

    Mirrors the NCCL backend's rendezvous fields. The trainer is always rank 0
    in the model-update group; workers sit at ``rank_offset`` (1) onwards.
    """

    master_address: str
    master_port: int
    rank_offset: int
    world_size: int


@dataclass
class NeuronWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Per-round update info for the Neuron weight-transfer backend.

    Carries only parameter metadata, like the NCCL backend. The weight bytes
    move over the store broadcast in :meth:`receive_weights`, keyed to this
    list's order.
    """

    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]

    def __post_init__(self) -> None:
        num_params = len(self.names)
        if len(self.dtype_names) != num_params:
            raise ValueError(
                f"`dtype_names` should be of the same size as `names`: "
                f"got {len(self.dtype_names)} and {num_params}"
            )
        if len(self.shapes) != num_params:
            raise ValueError(
                f"`shapes` should be of the same size as `names`: "
                f"got {len(self.shapes)} and {num_params}"
            )


class NeuronWeightTransferEngine(
    WeightTransferEngine[NeuronWeightTransferInitInfo, NeuronWeightTransferUpdateInfo]
):
    """Weight transfer engine that writes received weights into the resident
    Neuron model's existing parameter tensors, in place."""

    init_info_cls = NeuronWeightTransferInitInfo
    update_info_cls = NeuronWeightTransferUpdateInfo

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        # Lazily created at init_transfer_engine(); see also shutdown().
        self._group: Any = None
        # Trainer-side name -> neuron parameter name mapping. Identity by
        # default; inject a callable to support checkpoint-format trainers.
        self.name_map: NameMap = _identity_name_map
        # Lazily built lookup over the resident model's parameters. Keyed on
        # the *inner* (uncompiled) module so ``_orig_mod`` wrappers are bypassed.
        self._params: dict[str, torch.Tensor] | None = None

    # -- lifecycle ----------------------------------------------------------

    def init_transfer_engine(self, init_info: NeuronWeightTransferInitInfo) -> None:
        """Join the trainer<->worker model-update group.

        Computes this worker's unique rank across all data-parallel replicas
        (trainer = 0, workers = 1..N) and opens the stateless rendezvous store.
        """
        from vllm.distributed.utils import StatelessProcessGroup

        parallel_config = self.parallel_config
        dp_rank = parallel_config.data_parallel_index
        world_size_per_dp = parallel_config.world_size  # TP * PP
        rank_within_dp = parallel_config.rank
        worker_rank = dp_rank * world_size_per_dp + rank_within_dp
        rank = worker_rank + init_info.rank_offset

        self._group = StatelessProcessGroup.create(
            host=init_info.master_address,
            port=init_info.master_port,
            rank=rank,
            world_size=init_info.world_size,
        )
        logger.info(
            "Neuron weight transfer initialized: rank=%s world_size=%s",
            rank,
            init_info.world_size,
        )

    def start_weight_update(self) -> None:
        """Prepare for a new weight update.

        In-place application needs no layerwise-reload setup, so this is a no-op
        (the engine applies weights directly into the resident tensors).
        """

    def finish_weight_update(self) -> None:
        """Finalize the current weight update.

        In-place application is already committed per tensor; nothing to drain.
        """

    def receive_weights(self, update_info: NeuronWeightTransferUpdateInfo) -> None:
        """Receive one round of weights from the trainer and copy them in place.

        Each parameter is broadcast from the trainer (rank 0) over the store in
        ``update_info.names`` order, then copied into the matching resident
        parameter's storage without changing that parameter's identity.
        """
        if self._group is None:
            raise RuntimeError(
                "Neuron weight transfer not initialized. "
                "Call init_transfer_engine() first."
            )

        for name in update_info.names:
            # Receiver path: the source rank (trainer) writes the tensor to the
            # store; the returned value is a fresh host tensor.
            received = self._group.broadcast(None, src=0)
            self._copy_into_param(name, received)

    def update_weights(self, update_info: dict[str, Any]) -> None:
        """Receive one weight update chunk and load it into the model.

        Overrides the base to skip ``torch.accelerator.synchronize()``: the
        Neuron native backend does not register ``torch.accelerator``, and every
        weight write in :meth:`receive_weights` is a synchronous in-place
        ``copy_`` (blocking host->device), so the extra sync is both redundant
        and unsafe here.
        """
        typed_update_info = self.parse_update_info(update_info)
        self.receive_weights(typed_update_info)

    def shutdown(self) -> None:
        """Release the model-update group. No other resources are held."""
        self._group = None
        self._params = None

    # -- helpers ------------------------------------------------------------

    def _copy_into_param(self, name: str, tensor: torch.Tensor) -> None:
        """Copy ``tensor`` into the resident model's parameter named ``name``.

        Uses ``param.data.copy_`` (not ``load_state_dict`` with ``assign=True``)
        so the parameter object — and therefore the address the compiled graph
        reads — is preserved. This is the whole correctness contract of the
        backend.
        """
        params = self._param_lookup()
        target_name = self.name_map(name)
        param = params.get(target_name)
        if param is None:
            # Missing parameters are reported, not raised: a trainer that ships a
            # superset of the rollout's weights (e.g. with draft-model tensors)
            # should not abort the update of the tensors that do exist.
            logger.warning(
                "Neuron weight transfer: no parameter '%s' (from '%s') in the "
                "resident model; skipping.",
                target_name,
                name,
            )
            return

        dst = param.data
        src = tensor
        if src.device != dst.device or src.dtype != dst.dtype:
            src = src.to(device=dst.device, dtype=dst.dtype)
        with torch.no_grad():
            dst.copy_(src)

    def _param_lookup(self) -> dict[str, torch.Tensor]:
        """Build (and cache) a name -> tensor map over the inner model."""
        if self._params is None:
            inner = getattr(self.model, "_orig_mod", self.model)
            self._params = dict(inner.named_parameters())
        return self._params
