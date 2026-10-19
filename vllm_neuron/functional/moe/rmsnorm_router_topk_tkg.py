# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import nki
import nki.language as nl

from nkilib.core.moe_block.moe_block_tkg_utils import _pmax
from nkilib.core.router_topk.router_topk import XSBLayout_tp102__0, XSBLayout_tp201__2
from nkilib.core.router_topk.router_topk import router_topk as _router_topk
from nkilib.core.subkernels.rmsnorm_mx_quantize_tkg import (
    rmsnorm_mx_quantize_tkg as _rmsnorm_mx_quantize_tkg,
)
from nkilib.core.subkernels.rmsnorm_tkg import _rmsnorm_tkg_dloc
from nkilib.core.utils.common_types import QuantizationType, RouterActFnType
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.mx_torch_common import quantize_mx_golden

import torch
import numpy as np

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel


# Map torch dtypes to the NKI language dtypes the kernel expects.
_TORCH_TO_NKI_DTYPE = {
    torch.bfloat16: nl.bfloat16,
    torch.float16: nl.float16,
    torch.float32: nl.float32,
}


def rmsnorm_router_topk_tkg(
    hidden_states: torch.Tensor,
    gamma: torch.Tensor,
    router_weights: torch.Tensor,
    router_bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    top_k: int = 1,
    hidden_actual: Optional[int] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    router_mm_dtype: torch.dtype = torch.bfloat16,
    router_act_fn: RouterActFnType = RouterActFnType.SIGMOID,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused RMSNorm (+ optional MX quantize) + Router TopK.

    With `quantization_type=MX`, RMSNorm is fused with MX quantization and the quantized
    tensor and scales are returned as packed [T, H + H/4] FP8 output, representing [quant | scales].

    Args:
        hidden_states (torch.Tensor): [B, S, H] input tensor.
        gamma (torch.Tensor): [1, H] or [H] RMSNorm weights.
        router_weights (torch.Tensor): [H, E] router weights.
        router_bias (Optional[torch.Tensor]): [1, E] optional router bias.
        eps (float): RMSNorm epsilon. Default 1e-6.
        top_k (int): Number of top experts per token. Default 1.
        hidden_actual (Optional[int]): Actual hidden dim for padded inputs.
        quantization_type (QuantizationType): NONE for [T, H] unquantized output,
            MX for fused MX quantization with packed [T, H + H/4] FP8 output.
        router_mm_dtype (torch.dtype): Dtype for router matmul; also the dtype of
            the unquantized norm output.
        router_act_fn (RouterActFnType): Router activation (SOFTMAX or SIGMOID).

    Returns:
        Tuple of (norm_output, expert_index, expert_affinities):
            norm_output: [T, H] in router_mm_dtype when quantization_type=NONE.
                [T, H + H/4] FP8 packed (quant | MX scales) when MX.
            expert_index: [T, K] int32 top-K expert indices.
            expert_affinities: [T, E] bf16 masked expert affinities (zero
                outside top-K positions).

    Example:
        >>> norm, expert_idx, affinities = rmsnorm_router_topk_tkg(
        ...     hidden_states=hidden, gamma=gamma, router_weights=W,
        ...     top_k=2, router_act_fn=RouterActFnType.SOFTMAX,
        ... )
    """
    _validate_inputs(
        hidden_states,
        gamma,
        router_weights,
        router_bias,
        top_k,
        hidden_actual,
        quantization_type,
        router_act_fn,
    )

    # Legalize gamma to 2D [1, H] for the kernel.
    if gamma.ndim == 1:
        gamma = gamma.unsqueeze(0)

    if _can_use_kernel(
        hidden_states, router_weights, router_mm_dtype, quantization_type
    ):
        wrapped = wrap_nki(_rmsnorm_router_topk_tkg_nki)
        return wrapped[2](
            hidden_states=hidden_states,
            gamma=gamma,
            router_weights=router_weights,
            router_bias=router_bias,
            eps=eps,
            top_k=top_k,
            hidden_actual=hidden_actual,
            quantization_type=quantization_type,
            router_mm_dtype=_TORCH_TO_NKI_DTYPE[router_mm_dtype],
            router_act_fn=router_act_fn,
        )

    return _torch_impl(
        hidden_states,
        gamma,
        router_weights,
        router_bias,
        eps,
        top_k,
        hidden_actual,
        quantization_type,
        router_mm_dtype,
        router_act_fn,
    )


def _validate_inputs(
    hidden_states: torch.Tensor,
    gamma: torch.Tensor,
    router_weights: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    top_k: int,
    hidden_actual: Optional[int],
    quantization_type: QuantizationType,
    router_act_fn: RouterActFnType,
) -> None:
    """Validate inputs for rmsnorm_router_topk_tkg."""
    assert router_act_fn in (RouterActFnType.SIGMOID, RouterActFnType.SOFTMAX), (
        f"router_act_fn must be SIGMOID or SOFTMAX, got {router_act_fn}"
    )
    assert hidden_states.ndim == 3, (
        f"hidden_states must be [B, S, H], got shape {hidden_states.shape}"
    )
    B, S, H = hidden_states.shape
    T = B * S

    assert gamma.shape in ((1, H), (H,)), (
        f"gamma must be [1, {H}] or [{H}], got shape {gamma.shape}"
    )
    assert router_weights.ndim == 2 and router_weights.shape[0] == H, (
        f"router_weights must be [H={H}, E], got shape {router_weights.shape}"
    )
    E = router_weights.shape[1]

    if router_bias is not None:
        assert router_bias.shape == (1, E), (
            f"router_bias must be [1, {E}], got shape {router_bias.shape}"
        )

    assert top_k >= 1 and top_k <= E, f"top_k must be in [1, E={E}], got top_k={top_k}"
    assert H % 256 == 0, f"H must be divisible by 256, got H={H}"
    if hidden_actual is not None:
        assert 0 < hidden_actual <= H, (
            f"hidden_actual must be in (0, H={H}], got hidden_actual={hidden_actual}"
        )
    assert quantization_type in (QuantizationType.NONE, QuantizationType.MX), (
        f"quantization_type must be NONE or MX, got {quantization_type}"
    )
    if quantization_type == QuantizationType.MX:
        assert H % 512 == 0, (
            f"H must be divisible by 512 for MX quantization (4-wide quant tiles "
            f"on partition dim 128), got H={H}"
        )


def _can_use_kernel(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    router_mm_dtype: torch.dtype,
    quantization_type: QuantizationType,
) -> bool:
    """Return True if the NKI kernel can run for these inputs, else False.

    When any constraint is unmet, returns False so the caller falls back to the
    torch reference (used in CPU mode and for shapes outside the kernel's bounds).

    Constraints (all enforced by the underlying router_topk kernel):
        - Device supports NKI kernels (can_run_kernel)
        - router_mm_dtype is one of the NKI-supported dtypes
        - E <= F_MAX = 512 (gemm moving free-dim cap)
        - Without MX quantization, T must be divisible by 256: the unquantized
          norm path tiles T as T//2 across the two PE partitions and requires
          T//2 % 128 == 0 (NCC_INKI016). MX quantization tiles differently and
          is not subject to this.
    """

    if not can_run_kernel(hidden_states):
        return False

    if router_mm_dtype not in _TORCH_TO_NKI_DTYPE:
        return False

    _F_MAX = 512
    B, S, _ = hidden_states.shape
    T = B * S
    E = router_weights.shape[1]

    if E > _F_MAX:
        return False

    if quantization_type != QuantizationType.MX and T % 256 != 0:
        return False

    return True


def _torch_impl(
    hidden_states: torch.Tensor,
    gamma: torch.Tensor,
    router_weights: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    eps: float,
    top_k: int,
    hidden_actual: Optional[int],
    quantization_type: QuantizationType,
    router_mm_dtype: torch.dtype,
    router_act_fn: RouterActFnType,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference for rmsnorm_router_topk_tkg.

    Supports both QuantizationType.NONE and QuantizationType.MX. For MX, the
    norm_output is the packed [T, H + H/4] FP8 (quant | scales) tensor, while
    the router still consumes the unquantized norm (in router_mm_dtype) so
    the router topk path is dtype-independent of the downstream MoE input dtype.
    """
    B, S, H = hidden_states.shape
    T = B * S

    # RMSNorm in fp32, with optional hidden_actual for padded inputs.
    hidden_f32 = hidden_states.to(torch.float32).reshape(T, H)
    gamma_f32 = gamma.to(torch.float32)
    if hidden_actual is not None:
        sum_squares = torch.sum(hidden_f32**2, dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(sum_squares / hidden_actual + eps)
    else:
        inv_rms = torch.rsqrt(torch.mean(hidden_f32**2, dim=-1, keepdim=True) + eps)
    # Unquantized norm in router_mm_dtype, used for router matmul.
    norm = (hidden_f32 * inv_rms * gamma_f32).to(router_mm_dtype)

    # Router matmul (+ optional bias). Cast operands to router_mm_dtype then back
    # to fp32 so the matmul applies the cast's precision loss but accumulates in
    # fp32 — matching the tensor engine (bf16 inputs, fp32 accumulation). A true
    # bf16-accumulation matmul over H loses far more precision and flips near-tie
    # expert selections.
    norm_router = norm.to(router_mm_dtype).float()
    weights_router = router_weights.to(router_mm_dtype).float()
    logits = norm_router @ weights_router
    if router_bias is not None:
        logits = logits + router_bias.to(router_mm_dtype).float()

    # Top-K on raw logits → activation on only the selected logits → scatter.
    # Matches the kernel's router_topk with router_pre_norm=False,
    # norm_topk_prob=False. Activating over all E logits before top-K would give
    # different softmax affinities.
    topk_logits, topk_idx = torch.topk(logits, k=top_k, dim=-1)
    if router_act_fn == RouterActFnType.SOFTMAX:
        topk_probs = torch.softmax(topk_logits, dim=-1)
    elif router_act_fn == RouterActFnType.SIGMOID:
        topk_probs = torch.sigmoid(topk_logits)
    else:
        raise ValueError(f"Unsupported router_act_fn: {router_act_fn}")

    expert_index = topk_idx.to(torch.int32)

    expert_affinities = torch.zeros(
        T, router_weights.shape[1], dtype=torch.bfloat16, device=logits.device
    )
    expert_affinities.scatter_(
        dim=-1, index=topk_idx, src=topk_probs.to(torch.bfloat16)
    )

    if quantization_type == QuantizationType.NONE:
        return norm, expert_index, expert_affinities

    # MX path: produce the packed [T, H + H/4] FP8 (quant | scales) output.
    norm_output = _mx_quantize_packed(norm, T=T, H=H)
    return norm_output, expert_index, expert_affinities


def _mx_quantize_packed(norm: torch.Tensor, T: int, H: int) -> torch.Tensor:
    """MX-quantize a [T, H] norm output and pack into [T, H + H/4] FP8.

    Mirrors the `output_quant_packed=True` path of nkilib's
    rmsnorm_mx_quantize_tkg_torch_ref: tile along H into [H0=128, H/512, T, 4]
    quadrants, run quantize_mx_golden over the [H0, T*H/512*4] flat layout, then
    transpose back and concatenate the FP8 quant data and (uint8 → FP8 bitcast)
    scales along the H dim.
    """

    # Shapes
    H0 = 128
    H1 = H // H0
    _q_width = 4
    num_H512_tiles = H1 // _q_width
    out_quant_dtype = nl.float8_e4m3fn

    # Reshape norm -> [H0, T, H1] (matches the kernel's intermediate layout).
    norm_f16 = norm.to(torch.float16).reshape(T, H1, H0).permute(2, 0, 1)
    qmx_input_2D = (
        norm_f16.float()
        .reshape(H0, T, _q_width, num_H512_tiles)
        .permute(0, 3, 1, 2)
        .reshape(H0, -1)
    )

    out_data_dummy = np.empty(
        (H0, qmx_input_2D.shape[1] // _q_width), dtype=nl.float8_e4m3fn_x4
    )
    out_scale_dummy = np.empty((H0, qmx_input_2D.shape[1] // _q_width), dtype=np.uint8)
    mx_result = quantize_mx_golden(
        qmx_input_2D.numpy(), out_data_dummy, out_scale_dummy
    )

    # Pack: transpose to [T, H/512, 128] for both quant and scale, view as fp8,
    # flatten, and concat → [T, H + H/4] fp8.
    mx_data = mx_result["out_data_hbm"].reshape(H0, num_H512_tiles, T)
    out_scale = mx_result["out_scale_hbm"].reshape(H0, num_H512_tiles, T)

    mx_data = (
        np.ascontiguousarray(np.transpose(mx_data, (2, 1, 0)))
        # out_data_hbm is float8_e4m3fn_x4 (4-wide packed, itemsize 4). View as
        # the 1-byte fp8 so each x4 cell expands to its 4 constituent fp8 values
        # along H, yielding [T, H] — viewing as x4 would be a no-op and leave
        # H/4 columns (the reshape-size mismatch).
        .view(out_quant_dtype)
        .reshape(T, H)
    )
    out_scale = (
        np.ascontiguousarray(np.transpose(out_scale, (2, 1, 0)))
        .view(out_quant_dtype)
        .reshape(T, H // 4)
    )
    packed = np.concatenate((mx_data, out_scale), axis=1)
    return torch.from_numpy(packed.view(np.uint8)).view(torch.float8_e4m3fn)


@nki.jit
def _rmsnorm_router_topk_tkg_nki(
    hidden_states: nl.NkiTensor,
    gamma: nl.NkiTensor,
    router_weights: nl.NkiTensor,
    router_bias: Optional[nl.NkiTensor] = None,
    eps: float = 1e-6,
    top_k: int = 1,
    hidden_actual: Optional[int] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    router_mm_dtype=nl.bfloat16,
    router_act_fn: RouterActFnType = RouterActFnType.SIGMOID,
):
    """Fused RMSNorm (+ optional MX quantize) + Router TopK.

    Args:
        hidden_states (nl.NkiTensor): [B, S, H], Input tensor on HBM.
        gamma (nl.NkiTensor): [1, H], RMSNorm weights on HBM.
        router_weights (nl.NkiTensor): [H, E], Router weights on HBM.
        router_bias (Optional[nl.NkiTensor]): [1, E], Optional router bias on HBM.
        eps (float): Epsilon for RMSNorm. Default 1e-6.
        top_k (int): Number of top experts per token. Default 1.
        hidden_actual (Optional[int]): Actual hidden dim for padded inputs.
        quantization_type (QuantizationType): NONE or MX. Default NONE.
        router_mm_dtype: Dtype for router matmul. Default nl.bfloat16.
        router_act_fn (RouterActFnType): SOFTMAX or SIGMOID. Default SIGMOID.

    Returns:
        norm_output: [T, H] (NONE) or [T, H + H/4] FP8 packed quant|scales (MX).
        expert_index: [T, K] int32 top-K indices.
        expert_affinities: [T, E] bfloat16 masked top-K affinities (zero elsewhere).

    Notes:
        - Requires LNC=2 sharding.
        - QuantizationType.NONE: H must be divisible by 128; T must be a multiple of 256 (DLoC tiling).
        - QuantizationType.MX: H must be divisible by 512 (MX block size).
    """
    # Shapes, allocations
    B, S, H = hidden_states.shape
    T = B * S
    _, E = router_weights.shape
    H_free = H // _pmax
    expert_index = nl.ndarray((T, top_k), dtype=nl.int32, buffer=nl.shared_hbm)
    expert_affinities = nl.ndarray((T, E), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    rmsnorm_out_sb = nl.ndarray(
        (_pmax, T, H_free), dtype=router_mm_dtype, buffer=nl.sbuf
    )

    # RMSNorm + optional fused MX quantization
    if quantization_type == QuantizationType.MX:
        # MX path produces SBUF in XSBLayout_tp201__2 layout for the router.
        norm_output = nl.ndarray(
            (T, H + H // 4), dtype=nl.float8_e4m3fn, buffer=nl.shared_hbm
        )
        _rmsnorm_mx_quantize_tkg(
            input=hidden_states,
            gamma=gamma,
            output=rmsnorm_out_sb,
            output_quant=norm_output,
            output_scale=None,
            eps=eps,
            hidden_actual=hidden_actual,
            hidden_dim_tp=True,
        )
        x_sb_layout = XSBLayout_tp201__2
    elif quantization_type == QuantizationType.NONE:
        # DLoC RMSNorm produces SBUF in XSBLayout_tp102__0 layout.
        norm_output = nl.ndarray((T, H), dtype=router_mm_dtype, buffer=nl.shared_hbm)
        _rmsnorm_tkg_dloc(
            input_hbm=hidden_states,
            gamma=gamma,
            output_hbm=norm_output,
            output_sb=rmsnorm_out_sb,
            eps=eps,
            hidden_actual=hidden_actual,
            sync_output=True,
        )
        x_sb_layout = XSBLayout_tp102__0
    else:
        kernel_assert(
            False,
            f"rmsnorm_router_topk_tkg only supports QuantizationType.NONE or MX, got {quantization_type}",
        )

    # Router, TopK
    _router_topk(
        x=rmsnorm_out_sb,
        w=router_weights,
        w_bias=router_bias,
        router_logits=None,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        act_fn=router_act_fn,
        k=top_k,
        x_hbm_layout=0,
        x_sb_layout=x_sb_layout,
        router_pre_norm=False,
        norm_topk_prob=False,
        use_column_tiling=True,
        use_indirect_dma_scatter=True,
        use_PE_broadcast_w_bias=True,
        shard_on_tokens=T > 1,
        skip_store_expert_index=False,
        skip_store_router_logits=True,
    )

    return norm_output, expert_index, expert_affinities
