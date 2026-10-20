# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

import torch

from vllm_neuron import envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def can_run_kernel(device: torch.Tensor | str = "") -> bool:
    """Check if NKI kernels can run on the given device."""
    if envs.VLLM_NEURON_DISABLE_NKI_KERNELS:
        return False
    if envs.VLLM_NEURON_CPU_MODE:
        return os.environ.get("NKI_SIMULATOR") == "1"
    device_str = str(device.device) if isinstance(device, torch.Tensor) else device
    return device_str != "cpu"


def model_forward_context(
    vllm_config: VllmConfig,
) -> contextlib.AbstractContextManager[None]:
    """Context manager for model forward: skips fail_on_recompile in CPU mode."""
    if envs.VLLM_NEURON_CPU_MODE and vllm_config.model_config.enforce_eager:
        return contextlib.nullcontext()
    return torch.compiler.set_stance("fail_on_recompile")


def lane_grad_context() -> contextlib.AbstractContextManager[None]:
    """Enter the grad mode the parallel-trace lane compiled in.

    Wrap every invocation of a callable the lane compiled. Dynamo records the
    grad mode of whoever compiled a frame as a GLOBAL_STATE guard and checks it
    before any tensor guard, so warmup and serving have to agree. Warmup is what
    compiles; a served forward runs inside ``model_forward_context`` above, where
    the fail_on_recompile stance turns a disagreement into EngineDeadError rather
    than a quiet retrace.

    This matches the lane rather than imposing a mode. Lite's ``parallel_trace``
    sets ``grad_scope`` to ``torch.no_grad()`` only for a native compile; an XLA
    capture lane leaves it as ``nullcontext`` and traces on meta tensors. So this
    is no_grad on native and a no-op on XLA, because forcing no_grad on both
    would create on the XLA route the mismatch it removes on the native one.

    no_grad and not inference_mode. Both disable grad, but inference_mode drops
    ADInplaceOrView from a new tensor's dispatch key set while no_grad keeps it,
    so they leave different guard state.

    Lives here rather than on a runner because four call sites need it and they
    share no base class: the V1 runner's compiled model, the V2 runner's compiled
    adapter, and the Eagle and DFlash proposers' compiled draft models.

    Returns:
        A context manager. ``torch.no_grad()`` on the native route, otherwise a
        null context.
    """
    if not envs.is_native_backend():
        return contextlib.nullcontext()
    return torch.no_grad()


def in_lane_grad_mode(method):
    """Run a worker entry point in the grad mode the trace lane compiled in.

    Decorate the OUTER boundary, not the individual forward calls. Dynamo keys a
    cache entry on the frame it compiles, and on this route those frames live in
    wrappers rather than at the call sites: ``external_utils.wrap_inline``'s
    ``inner`` for a compiled nn.Module, and the ``forward`` that
    ``spec_decode/decorator.py`` substitutes for a target model's own. Neither is
    reachable by listing the places that call a model, so covering call sites one
    at a time cannot be shown to be complete, and three successive attempts to do
    that each missed a different frame.

    Setting the mode once at the entry point makes every frame below it -- known
    or not -- run in one mode, so completeness stops depending on enumeration.

    Both sides have to move together. Warmup is what compiles, serving reuses, and
    a served forward runs inside ``model_forward_context``'s fail_on_recompile
    stance where a disagreement is EngineDeadError rather than a quiet retrace.
    Decorating warmup alone is what broke the route in 464f6b2142.

    Returns:
        The wrapped method. A no-op on the XLA route, whose capture lane traces on
        ambient grad; see ``lane_grad_context``.
    """
    import functools

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        with lane_grad_context():
            return method(*args, **kwargs)

    # An explicit marker so a test can assert the boundary is covered without
    # inspecting closures, which would break the moment another decorator is
    # stacked on one of these methods.
    wrapper._sets_lane_grad_mode = True
    return wrapper
