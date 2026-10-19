# SPDX-License-Identifier: Apache-2.0
"""Fused pre-norm residual add + RMSNorm for prefill (token-major [T, H] or [B, S, H]).

Wraps the nkilib ``rmsnorm_residual_prefill`` kernel, which adds the residual inside the
DMA engines during the load and RMSNorms the sum in one pass. It returns both the normed
output and the pre-norm sum, so the caller reuses the sum as the next residual instead of
recomputing ``residual + hidden`` on the Vector engine.

The variance sums every column and divides by ``unpadded_hidden_size``. Do not narrow the
sum to the first ``unpadded_hidden_size`` columns: the mxfp4 H order interleaves the zero
pads instead of keeping them in a tail.

When the kernel cannot run (CPU / simulation / kernels disabled) this falls back to a
torch RMSNorm over the same math.
"""

from typing import Optional, Tuple

import torch

from nkilib.experimental.subkernels.rmsnorm_residual_prefill import (
    rmsnorm_residual_prefill as _rmsnorm_residual_prefill_kernel,
)

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel


def _can_use_kernel(hidden_states: torch.Tensor) -> bool:
    """Report if the kernel can run. It tiles H and shards T itself, so only the
    global kernel switch and the supported rank gate it."""
    if not can_run_kernel(hidden_states):
        return False

    # The kernel accepts [T, H] and [B, S, H]. Fall back for any other rank.
    return hidden_states.dim() in (2, 3)


def rmsnorm_residual_prefill(
    hidden_states: torch.Tensor,
    gamma: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    unpadded_hidden_size: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fused residual add + RMSNorm.

    Args:
        hidden_states: [T, H] or [B, S, H] hidden states.
        gamma: ``[H]`` or ``[1, H]`` normalization weight. It must use the same H order
            as ``hidden_states``.
        residual: Same shape as ``hidden_states``, added before the norm. Omit for a
            plain RMSNorm.
        eps: Epsilon for numerical stability.
        unpadded_hidden_size: True hidden width when H is padded, used as the mean
            divisor. Defaults to H.

    Returns:
        ``(normed, pre_norm_sum)``, both shaped like ``hidden_states``.
        ``pre_norm_sum`` is ``hidden + residual``, or None when ``residual`` is None.

    Examples:
        >>> import torch
        >>> hidden = torch.randn(256, 3072, dtype=torch.bfloat16)
        >>> gamma = torch.ones(3072, dtype=torch.bfloat16)
        >>> residual = torch.randn(256, 3072, dtype=torch.bfloat16)
        >>> normed, pre_norm_sum = rmsnorm_residual_prefill(
        ...     hidden, gamma, residual, unpadded_hidden_size=2880
        ... )  # doctest: +SKIP
    """
    # The kernel asserts gamma is [1, H]. A norm module holds it as [H].
    if gamma.ndim == 1:
        gamma = gamma.unsqueeze(0)

    # Check the kernel contract on both paths, so a bad input fails the same way on a
    # CPU fallback as on the device.
    hidden_size = hidden_states.shape[-1]
    if residual is not None:
        assert residual.shape == hidden_states.shape, (
            f"residual {tuple(residual.shape)} must match "
            f"hidden {tuple(hidden_states.shape)}"
        )
    assert tuple(gamma.shape) == (1, hidden_size), (
        f"gamma {tuple(gamma.shape)} must be [{hidden_size}] or [1, {hidden_size}]"
    )
    assert unpadded_hidden_size is None or 0 < unpadded_hidden_size <= hidden_size, (
        f"unpadded_hidden_size {unpadded_hidden_size} must be in (0, {hidden_size}]"
    )

    if not _can_use_kernel(hidden_states):
        return _torch_impl(hidden_states, gamma, residual, eps, unpadded_hidden_size)

    # wrap_nki(...)[N] selects the LNC degree (grid size), NOT the return-value count.
    # The kernel supports LNC 1 or 2 and shards tokens across cores itself. The grid must
    # match the configured LNC, because NEURON_ENFORCE_NKI_LNC_CONSISTENCY rejects a kernel
    # whose launch grid differs from the LNC the graph compiles at.
    outs = wrap_nki(_rmsnorm_residual_prefill_kernel)[get_logical_nc_config()](
        hidden=hidden_states,
        gamma=gamma,
        residual=residual,
        eps=eps,
        hidden_actual=unpadded_hidden_size,
    )

    if residual is not None:
        out, output_residual = outs
        return out.view_as(hidden_states), output_residual.view_as(hidden_states)
    out = outs[0] if isinstance(outs, (list, tuple)) else outs
    return out.view_as(hidden_states), None


def _torch_impl(
    hidden_states: torch.Tensor,
    gamma: torch.Tensor,
    residual: Optional[torch.Tensor],
    eps: float,
    unpadded_hidden_size: Optional[int],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Torch reference mirroring the kernel: add, then norm over the full padded width.

    The variance sums every column and divides by the unpadded width, which is what the
    kernel does. This is correct for a natural H order and for the shuffled H order that
    the mxfp4 model uses, because a zero pad adds nothing to the sum wherever it sits.
    """
    x = hidden_states if residual is None else hidden_states + residual
    pre_norm_sum = None if residual is None else x

    n = unpadded_hidden_size or x.shape[-1]
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True) * (x.shape[-1] / n)
    out = x_fp32 * torch.rsqrt(variance + eps) * gamma.to(torch.float32)

    return out.to(hidden_states.dtype), pre_norm_sum
