# SPDX-License-Identifier: Apache-2.0
from nkilib.experimental.transformer.attention_block_tkg import (
    attention_block_tkg as _nki_attention_block_tkg,
)
from nkilib.experimental.transformer.attention_block_tkg_sharding import (
    CPCollectiveMode,
)

# GQA (kv_heads > 1): the kernel infers kv_heads from the K/V-cache shape (no
# kv_heads argument) — see _infer_kv_heads for the cache-layout convention. The
# caller's only GQA obligation is a 4D cache + a per-head 3D block table
# [B, kv_heads, num_blocks]; a 2D-table GQA request routes to the torch fallback
# (see _can_use_attention_block_kernel).

from typing import Optional, Tuple

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed import ProcessGroup

import nki
import nki.collectives as ncc
from torch import Tensor

from nkilib.core.utils.allocator import SbufManager
from nkilib.core.utils.common_types import QuantizationType

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron import envs
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel


def _pv_swapped_layout_admit_d128(atp, cfg, sbm, fa_ctx, TC):
    # Mirror of nkilib _use_pv_swapped_layout, but admit d_head 128. The nkilib
    # PV-swap body is d_head-generic; only the gate limits it to d_head 64.
    # Keep this in sync with nkilib until the gate change lands in the wheel.
    col_tile_width = TC.sbuf_quadrant_size
    if col_tile_width % atp.s_active_qh != 0:
        return False
    fold_pack = col_tile_width // atp.s_active_qh
    batch_tile = TC.p_max // col_tile_width
    if fold_pack * cfg.d_head > TC.psum_f_max:
        return False
    if atp.qk_swapped and atp.s_active_bqh_remainder != 0:
        return False
    return (
        sbm.is_auto_alloc()
        and atp.is_block_kv
        and cfg.out_in_sb
        and cfg.curr_sprior >= 8192
        and atp.sprior_n_prgs == 1
        and cfg.d_head in (64, 128)
        and atp.d_tile_size == cfg.d_head
        and atp.n_d_tiles == 1
        and atp.block_len == 16
        and atp.bs % batch_tile == 0
        and fa_ctx.tile_n_sprior % fold_pack == 0
    )


def _maybe_enable_pv_swap_mm2():
    # Override the nkilib gate to admit d_head 128 when the flag is on.
    if not envs.VLLM_NEURON_ATTN_PV_SWAP:
        return
    import nkilib.core.attention.attention_tkg as _tkg

    _tkg._use_pv_swapped_layout = _pv_swapped_layout_admit_d128


_maybe_enable_pv_swap_mm2()

from vllm_neuron.functional.attention.attention_decode_mask import (
    _resize_block_len as _mask_resize_block_len,
    P_MAX as _MASK_P_MAX,
    maybe_transpose_mask_for_qk_swap,
    relayout_block_kv_mask,
)

_PMAX = 128


def _resolve_dcp_collective_mode(dcp_collective_mode: str, dcp_size: int):
    """Resolve the DCP collective mode before entering the NKI trace.

    ALL_TO_ALL needs at least four ranks, so smaller groups use
    REDUCE_SCATTER regardless of the requested mode.
    """
    if dcp_collective_mode not in ("auto", "reduce_scatter", "all_to_all"):
        raise ValueError(
            "dcp_collective_mode must be one of 'auto'/'reduce_scatter'/"
            f"'all_to_all', got {dcp_collective_mode!r}"
        )
    if dcp_size < 4 or dcp_collective_mode == "reduce_scatter":
        return CPCollectiveMode.REDUCE_SCATTER
    return CPCollectiveMode.ALL_TO_ALL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _maybe_transpose_mask_for_qk_swap(
    attention_mask: Optional[Tensor],
    V_cache: Tensor,
    active_blocks_table: Optional[Tensor],
    k_scale: Optional[Tensor],
    fp8_packed: bool,
) -> Optional[Tensor]:
    """Thin wrapper over the shared ``maybe_transpose_mask_for_qk_swap`` in
    ``attention_decode_mask`` (which owns the nkilib QK-swap layout contract).
    Kept as a wrapper so the layout decision has a single implementation.
    Only applies on the full pre-generated mask path (``pos_ids is None``); the
    fused mask-gen path uses an active-only mask the kernel never transposes.
    """
    return maybe_transpose_mask_for_qk_swap(
        attention_mask,
        d_head=int(V_cache.shape[-1]),
        is_fp8_kv=k_scale is not None,
        fp8_packed=fp8_packed,
    )


def _build_default_active_mask(
    B: int,
    S_tkg: int,
    q_heads: int,
    device: torch.device,
    dcp_active_owner: Optional[Tensor] = None,
) -> Tensor:
    """Build the active-only causal overlay.

    DCP stores the active token on every rank; the owner gate prevents double
    counting in the CP LSE combine.
    """
    triu = torch.triu(torch.ones(S_tkg, S_tkg, dtype=torch.float32, device=device))
    mask = triu[:, None, None, :].expand(S_tkg, B, q_heads, S_tkg).contiguous()
    if dcp_active_owner is not None:
        # Attention-DP is excluded because it redistributes B after mask construction.
        assert dcp_active_owner.shape[0] == B, (
            f"dcp_active_owner batch {dcp_active_owner.shape[0]} != mask batch {B}"
        )
        owner = dcp_active_owner.to(mask.dtype).reshape(B, -1)  # [B, 1] or [B, S_tkg]
        owner = owner[None, :, None, :]  # [1, B, 1, S_tkg or 1]
        mask = mask * owner
    return mask


def _maybe_build_sbm(
    sbm_lower_bound: Optional[int],
    sbm_upper_bound: Optional[int],
    sbm_use_auto_alloc: bool,
    sbm_default_stack_alloc: bool,
) -> Optional[SbufManager]:
    if sbm_lower_bound is None and sbm_upper_bound is None:
        return None

    return SbufManager(
        sb_lower_bound=sbm_lower_bound,
        sb_upper_bound=sbm_upper_bound,
        use_auto_alloc=sbm_use_auto_alloc,
        default_stack_alloc=sbm_default_stack_alloc,
    )


# ---------------------------------------------------------------------------
# NKI entry-point (runs on NeuronCore)
# ---------------------------------------------------------------------------


@nki.jit
def _torch_compatible_attention_block_tkg_kernel(
    # -- input
    X: Tensor,
    X_hidden_dim_actual: int = None,
    # -- rmsnorm X
    rmsnorm_X_enabled: bool = False,
    rmsnorm_X_eps: Optional[float] = None,
    rmsnorm_X_gamma: Tensor = None,
    # -- qkv projections
    W_qkv: Tensor = None,
    bias_qkv: Tensor = None,
    quantization_type_qkv: QuantizationType = QuantizationType.NONE,
    weight_dequant_scale_qkv: torch.Tensor = None,
    input_dequant_scale_qkv: torch.Tensor = None,
    # -- Q/K processing: pre-RoPE RMSNorm
    rmsnorm_QK_pre_rope_enabled: bool = False,
    rmsnorm_QK_pre_rope_eps: float = 1e-6,
    rmsnorm_QK_pre_rope_W_Q: Tensor = None,
    rmsnorm_QK_pre_rope_W_K: Tensor = None,
    # -- Q/K processing: RoPE
    cos: Optional[Tensor] = None,
    sin: Optional[Tensor] = None,
    rope_contiguous_layout: bool = True,
    # -- Q/K processing: post-RoPE RMSNorm
    rmsnorm_QK_post_rope_enabled: bool = False,
    rmsnorm_QK_post_rope_eps: float = 1e-6,
    rmsnorm_QK_post_rope_W_Q: Tensor = None,
    rmsnorm_QK_post_rope_W_K: Tensor = None,
    # -- attention
    K_cache_transposed: bool = False,
    active_blocks_table: Tensor = None,
    K_cache: Tensor = None,
    V_cache: Tensor = None,
    attention_mask: Tensor = None,
    sink: Tensor = None,
    softmax_scale: float = None,
    # -- in-kernel mask generation (when pos_ids is provided, the kernel
    #    generates the prior causal/SWA mask on-chip from pos_ids and
    #    attention_mask carries only the active-only portion)
    pos_ids: Tensor = None,
    swa_start_pos_ids: Tensor = None,
    # -- KV cache update
    update_cache: bool = False,
    kv_cache_update_idx: Tensor = None,
    k_scale: Tensor = None,
    v_scale: Tensor = None,
    # -- packed FP8 K cache layout
    fp8_packed: bool = False,
    # -- output projection
    W_out: Tensor = None,
    bias_out: Tensor = None,
    quantization_type_out: QuantizationType = QuantizationType.NONE,
    weight_dequant_scale_out: Tensor = None,
    input_dequant_scale_out: Tensor = None,
    transposed_out: bool = False,
    # -- output control
    out_in_sb: bool = False,
    skip_attention: bool = False,
    # -- STATIC_MX layout flag
    is_h_transposed_by_4: bool = False,
    # -- sbm
    sbm_lower_bound: int = None,
    sbm_upper_bound: int = None,
    sbm_use_auto_alloc: bool = True,
    sbm_default_stack_alloc: bool = True,
):
    """
    Torch-friendly @nki.jit wrapper that reconstructs NKI-specific objects
    (SbufManager, QuantizationType) from plain scalar / tensor args and
    delegates to the library attention_block_tkg kernel.
    """

    sbm = _maybe_build_sbm(
        sbm_lower_bound, sbm_upper_bound, sbm_use_auto_alloc, sbm_default_stack_alloc
    )

    # Forward to the library kernel (no kv_heads argument — see the module header).
    return _nki_attention_block_tkg(
        X=X,
        X_hidden_dim_actual=X_hidden_dim_actual,
        rmsnorm_X_enabled=rmsnorm_X_enabled,
        rmsnorm_X_eps=rmsnorm_X_eps,
        rmsnorm_X_gamma=rmsnorm_X_gamma,
        W_qkv=W_qkv,
        bias_qkv=bias_qkv,
        quantization_type_qkv=quantization_type_qkv,
        weight_dequant_scale_qkv=weight_dequant_scale_qkv,
        input_dequant_scale_qkv=input_dequant_scale_qkv,
        rmsnorm_QK_pre_rope_enabled=rmsnorm_QK_pre_rope_enabled,
        rmsnorm_QK_pre_rope_eps=rmsnorm_QK_pre_rope_eps,
        rmsnorm_QK_pre_rope_W_Q=rmsnorm_QK_pre_rope_W_Q,
        rmsnorm_QK_pre_rope_W_K=rmsnorm_QK_pre_rope_W_K,
        cos=cos,
        sin=sin,
        rope_contiguous_layout=rope_contiguous_layout,
        rmsnorm_QK_post_rope_enabled=rmsnorm_QK_post_rope_enabled,
        rmsnorm_QK_post_rope_eps=rmsnorm_QK_post_rope_eps,
        rmsnorm_QK_post_rope_W_Q=rmsnorm_QK_post_rope_W_Q,
        rmsnorm_QK_post_rope_W_K=rmsnorm_QK_post_rope_W_K,
        K_cache_transposed=K_cache_transposed,
        active_blocks_table=active_blocks_table,
        K_cache=K_cache,
        V_cache=V_cache,
        attention_mask=attention_mask,
        sink=sink,
        softmax_scale=softmax_scale,
        update_cache=update_cache,
        kv_cache_update_idx=kv_cache_update_idx,
        k_scale=k_scale,
        v_scale=v_scale,
        fp8_packed=fp8_packed,
        W_out=W_out,
        bias_out=bias_out,
        quantization_type_out=quantization_type_out,
        weight_dequant_scale_out=weight_dequant_scale_out,
        input_dequant_scale_out=input_dequant_scale_out,
        transposed_out=transposed_out,
        out_in_sb=out_in_sb,
        sbm=sbm,
        skip_attention=skip_attention,
        is_h_transposed_by_4=is_h_transposed_by_4,
        KVDP=1,
        KVDP_replica_group=None,
        pos_ids=pos_ids,
        swa_start_pos_ids=swa_start_pos_ids,
        S_ctx=None,
    )


@nki.jit
def _torch_compatible_attention_block_tkg_kernel_dcp(
    # -- input
    X: Tensor,
    X_hidden_dim_actual: int = None,
    # -- rmsnorm X
    rmsnorm_X_enabled: bool = False,
    rmsnorm_X_eps: Optional[float] = None,
    rmsnorm_X_gamma: Tensor = None,
    # -- qkv projections
    W_qkv: Tensor = None,
    bias_qkv: Tensor = None,
    quantization_type_qkv: QuantizationType = QuantizationType.NONE,
    weight_dequant_scale_qkv: torch.Tensor = None,
    input_dequant_scale_qkv: torch.Tensor = None,
    # -- Q/K processing: pre-RoPE RMSNorm
    rmsnorm_QK_pre_rope_enabled: bool = False,
    rmsnorm_QK_pre_rope_eps: float = 1e-6,
    rmsnorm_QK_pre_rope_W_Q: Tensor = None,
    rmsnorm_QK_pre_rope_W_K: Tensor = None,
    # -- Q/K processing: RoPE
    cos: Optional[Tensor] = None,
    sin: Optional[Tensor] = None,
    rope_contiguous_layout: bool = True,
    # -- Q/K processing: post-RoPE RMSNorm
    rmsnorm_QK_post_rope_enabled: bool = False,
    rmsnorm_QK_post_rope_eps: float = 1e-6,
    rmsnorm_QK_post_rope_W_Q: Tensor = None,
    rmsnorm_QK_post_rope_W_K: Tensor = None,
    # -- attention
    K_cache_transposed: bool = False,
    active_blocks_table: Tensor = None,
    K_cache: Tensor = None,
    V_cache: Tensor = None,
    attention_mask: Tensor = None,
    sink: Tensor = None,
    softmax_scale: float = None,
    # -- in-kernel mask generation (when pos_ids is provided, the kernel
    #    generates the prior causal/SWA mask on-chip from pos_ids and
    #    attention_mask carries only the active-only portion)
    pos_ids: Tensor = None,
    swa_start_pos_ids: Tensor = None,
    # -- KV cache update
    update_cache: bool = False,
    kv_cache_update_idx: Tensor = None,
    k_scale: Tensor = None,
    v_scale: Tensor = None,
    # -- packed FP8 K cache layout
    fp8_packed: bool = False,
    # -- output projection
    W_out: Tensor = None,
    bias_out: Tensor = None,
    quantization_type_out: QuantizationType = QuantizationType.NONE,
    weight_dequant_scale_out: Tensor = None,
    input_dequant_scale_out: Tensor = None,
    transposed_out: bool = False,
    # -- output control
    out_in_sb: bool = False,
    skip_attention: bool = False,
    # -- STATIC_MX layout flag
    is_h_transposed_by_4: bool = False,
    # -- sbm
    sbm_lower_bound: int = None,
    sbm_upper_bound: int = None,
    sbm_use_auto_alloc: bool = True,
    sbm_default_stack_alloc: bool = True,
    # -- decode context parallelism
    dcp_size: int = 1,
    dcp_group_ranks=None,
    dcp_collective_mode_resolved=None,
):
    """Torch-friendly @nki.jit wrapper that reconstructs NKI objects (SbufManager,
    QuantizationType, ReplicaGroup) from plain args and delegates to
    attention_block_tkg.
    """

    sbm = _maybe_build_sbm(
        sbm_lower_bound, sbm_upper_bound, sbm_use_auto_alloc, sbm_default_stack_alloc
    )

    # ReplicaGroup cannot cross the HOP boundary. Pass the ranks from vLLM's
    # DCP group and rebuild the single group inside the trace.
    dcp_replica_group = (
        ncc.ReplicaGroup([list(dcp_group_ranks)])
        if dcp_group_ranks is not None
        else None
    )
    return _nki_attention_block_tkg(
        X=X,
        X_hidden_dim_actual=X_hidden_dim_actual,
        rmsnorm_X_enabled=rmsnorm_X_enabled,
        rmsnorm_X_eps=rmsnorm_X_eps,
        rmsnorm_X_gamma=rmsnorm_X_gamma,
        W_qkv=W_qkv,
        bias_qkv=bias_qkv,
        quantization_type_qkv=quantization_type_qkv,
        weight_dequant_scale_qkv=weight_dequant_scale_qkv,
        input_dequant_scale_qkv=input_dequant_scale_qkv,
        rmsnorm_QK_pre_rope_enabled=rmsnorm_QK_pre_rope_enabled,
        rmsnorm_QK_pre_rope_eps=rmsnorm_QK_pre_rope_eps,
        rmsnorm_QK_pre_rope_W_Q=rmsnorm_QK_pre_rope_W_Q,
        rmsnorm_QK_pre_rope_W_K=rmsnorm_QK_pre_rope_W_K,
        cos=cos,
        sin=sin,
        rope_contiguous_layout=rope_contiguous_layout,
        rmsnorm_QK_post_rope_enabled=rmsnorm_QK_post_rope_enabled,
        rmsnorm_QK_post_rope_eps=rmsnorm_QK_post_rope_eps,
        rmsnorm_QK_post_rope_W_Q=rmsnorm_QK_post_rope_W_Q,
        rmsnorm_QK_post_rope_W_K=rmsnorm_QK_post_rope_W_K,
        K_cache_transposed=K_cache_transposed,
        active_blocks_table=active_blocks_table,
        K_cache=K_cache,
        V_cache=V_cache,
        attention_mask=attention_mask,
        sink=sink,
        softmax_scale=softmax_scale,
        update_cache=update_cache,
        kv_cache_update_idx=kv_cache_update_idx,
        k_scale=k_scale,
        v_scale=v_scale,
        fp8_packed=fp8_packed,
        W_out=W_out,
        bias_out=bias_out,
        quantization_type_out=quantization_type_out,
        weight_dequant_scale_out=weight_dequant_scale_out,
        input_dequant_scale_out=input_dequant_scale_out,
        transposed_out=transposed_out,
        out_in_sb=out_in_sb,
        sbm=sbm,
        skip_attention=skip_attention,
        is_h_transposed_by_4=is_h_transposed_by_4,
        KVDP=1,
        KVDP_replica_group=None,
        CP=dcp_size,
        CP_replica_group=dcp_replica_group,
        CP_collective_mode=dcp_collective_mode_resolved,
        pos_ids=pos_ids,
        swa_start_pos_ids=swa_start_pos_ids,
        S_ctx=None,
    )


# ---------------------------------------------------------------------------
# Torch helpers for the fallback implementation
# ---------------------------------------------------------------------------


def _torch_rms_norm(
    x: Tensor, eps: float, weight: Optional[Tensor], dim_actual: Optional[int] = None
) -> Tensor:
    """
    RMS normalization: x / sqrt(mean(x^2) + eps), optionally scaled by weight.

    Args:
        x: Input tensor, normalised along the last dimension.
        eps: Epsilon for numerical stability.
        weight: Optional per-element scale (broadcastable to x).
        dim_actual: If the last dimension is zero-padded, supply the real
                    (unpadded) size so the mean is computed correctly.

    Returns:
        Normalised tensor in the same dtype as *x*.
    """
    input_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    d = dim_actual if dim_actual is not None else x.shape[-1]
    variance = x_fp32.pow(2).sum(-1, keepdim=True) / d
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    if weight is not None:
        x_normed = x_normed * weight.to(torch.float32)
    return x_normed.to(input_dtype)


def _torch_rms_norm_heads(x: Tensor, eps: float, weight: Optional[Tensor]) -> Tensor:
    """
    Per-head RMS normalization over the last (d_head) dimension.

    Args:
        x: [..., d_head]
        eps: Epsilon.
        weight: [1, d_head] or [d_head] scale. Broadcast across batch/head dims.

    Returns:
        Normalised tensor, same shape and dtype as *x*.
    """
    input_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    if weight is not None:
        w = weight.to(torch.float32).view(-1)  # flatten to [d_head]
        x_normed = x_normed * w
    return x_normed.to(input_dtype)


def _torch_rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input (contiguous layout)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _torch_rotate_half_interleaved(x: Tensor) -> Tensor:
    """Rotates hidden dims using interleaved layout (pairs of adjacent elements)."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def _torch_apply_rope(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    rope_contiguous_layout: bool,
) -> Tuple[Tensor, Tensor]:
    """
    Apply Rotary Position Embedding to Q and K.

    Args:
        q: [B, q_heads, S_tkg, d_head]
        k: [B, kv_heads, S_tkg, d_head]
        cos: [d_head//2, B, S_tkg]  RoPE cosine values.
        sin: [d_head//2, B, S_tkg]  RoPE sine values.
        rope_contiguous_layout: True = first-half/second-half split,
                                False = interleaved pairs.

    Returns:
        (q_rotated, k_rotated) with the same shapes and dtype as inputs.
    """
    # cos, sin: [d_head//2, B, S_tkg] -> [B, 1, S_tkg, d_head//2]
    cos_r = cos.permute(1, 2, 0).unsqueeze(1)  # [B, 1, S_tkg, d_head//2]
    sin_r = sin.permute(1, 2, 0).unsqueeze(1)  # [B, 1, S_tkg, d_head//2]

    if rope_contiguous_layout:
        cos_full = torch.cat([cos_r, cos_r], dim=-1)  # [B, 1, S_tkg, d_head]
        sin_full = torch.cat([sin_r, sin_r], dim=-1)
        rotate_fn = _torch_rotate_half
    else:
        cos_full = torch.stack([cos_r, cos_r], dim=-1).flatten(-2)
        sin_full = torch.stack([sin_r, sin_r], dim=-1).flatten(-2)
        rotate_fn = _torch_rotate_half_interleaved

    cos_full = cos_full.to(q.dtype)
    sin_full = sin_full.to(q.dtype)

    q_rotated = (q * cos_full) + (rotate_fn(q) * sin_full)
    k_rotated = (k * cos_full) + (rotate_fn(k) * sin_full)

    return q_rotated, k_rotated


# ---------------------------------------------------------------------------
# Packed FP8 K-cache swizzle helpers (torch fallback)
# ---------------------------------------------------------------------------
#
# The packed layout interleaves two consecutive sequence positions into the
# trailing size-2 dim so the kernel can bf16-reinterpret + DMA-transpose:
#
#     unpacked: [..., block_len, d_head]
#     packed:   [..., block_len // 2, d_head, 2]   (dim -1: 0=even pos, 1=odd pos)
#
# These mirror the kernel's integration-test swizzle
#   K.reshape(nb, bl // 2, 2, d).transpose(0, 1, 3, 2)
# but operate on whatever leading dims the cache carries (3D/4D/5D).


def _unswizzle_packed_k(k_packed: Tensor) -> Tensor:
    """[..., block_len // 2, d_head, 2] fp8 → [..., block_len, d_head].

    Inverse of :func:`_swizzle_packed_k`. Leaves dtype unchanged.
    """
    *lead, half, d_head, two = k_packed.shape
    assert two == 2, f"packed K last dim must be 2, got {two}"
    # [..., half, d_head, 2] → [..., half, 2, d_head] → [..., half*2, d_head]
    return k_packed.transpose(-1, -2).reshape(*lead, half * 2, d_head).contiguous()


def _swizzle_packed_k(k_unpacked: Tensor) -> Tensor:
    """[..., block_len, d_head] → [..., block_len // 2, d_head, 2].

    Inverse of :func:`_unswizzle_packed_k`. Leaves dtype unchanged.
    """
    *lead, block_len, d_head = k_unpacked.shape
    assert block_len % 2 == 0, f"block_len must be even to pack, got {block_len}"
    # [..., bl, d_head] → [..., bl//2, 2, d_head] → [..., bl//2, d_head, 2]
    return (
        k_unpacked.reshape(*lead, block_len // 2, 2, d_head)
        .transpose(-1, -2)
        .contiguous()
    )


def scatter_packed_k(
    k_cache: Tensor,
    k_flat: Tensor,
    block_indices: Tensor,
    head_indices: Tensor,
    position_indices: Tensor,
) -> None:
    """Scatter new K token *pairs* directly into the swizzled packed K cache.

    Avoids the whole-cache unswizzle → scatter → reswizzle round-trip on the
    prefill KV-cache write. The packed slot ``[d_head, 2]`` interleaves two
    adjacent token positions ``(2k, 2k+1)`` into the trailing size-2 dim (lane 0
    = position 2k, lane 1 = position 2k+1; see :func:`_swizzle_packed_k`).

    Rather than a per-token single-lane write — ``k_cache[.., :, p % 2] = ...`` —
    which needs a tensor-valued inner index and a stride-2 partial store that
    does NOT lower on the Neuron backend, we write whole PAIRS: reshape the new
    K rows into ``[num_pairs, d_head, 2]`` and ``index_put_`` on the three
    leading dims ``(block, head, outer=position // 2)``, leaving the full
    ``[d_head, 2]`` slot as the contiguous payload. This is the same advanced-
    indexing structure as the legacy unpacked scatter (3 leading index dims + a
    dense payload), so it lowers, and it writes only the new tokens' bytes. All
    pairs go in one ``index_put_``, so the trace holds a single indirect DMA
    regardless of token count.

    A row targeting physical block 0 is padding and does not replace its lane.
    Mixed pairs use the valid lane's coordinate and preserve the stored bytes
    for the null lane. All-null pairs self-write a shared block-0 scratch slot.

    Precondition: rows are pair-aligned, so consecutive rows ``(2k, 2k+1)``
    represent lanes 0 and 1 of one logical pair, and the row count is even.
    Valid lanes in the same pair share their block, head, and packed outer.

    Args:
        k_cache: packed K cache ``[blocks, Nkh, block_size // 2, d_head, 2]``.
        k_flat: new K rows ``[num_indices, d_head]`` (already scaled/cast),
            pair-aligned as described above.
        block_indices: ``[num_indices]`` block coordinate per row.
        head_indices: ``[num_indices]`` KV-head coordinate per row.
        position_indices: ``[num_indices]`` position-within-block per row.
    """
    num_rows, d_head = k_flat.shape

    # Tensor-valued Python assertions are eager-only: Dynamo folds this guard
    # before handing the FX graph to any backend. Literal FakeTensor execution
    # also skips value assertions; real eager tensors enforce every contract.
    validate_eager_contract = (
        not torch.compiler.is_dynamo_compiling() and not isinstance(k_flat, FakeTensor)
    )
    if validate_eager_contract:
        assert num_rows % 2 == 0, (
            f"scatter_packed_k requires an even, pair-aligned row count (even "
            f"tokens per head x num_heads), got {num_rows}"
        )

    k_pairs = (
        k_flat.view(-1, 2, d_head).transpose(1, 2).contiguous()
    )  # [num_pairs, d_head, 2]
    block_lanes = block_indices.view(-1, 2)
    head_lanes = head_indices.view(-1, 2)
    position_lanes = position_indices.view(-1, 2)
    valid_lanes = block_lanes > 0

    if validate_eager_contract:
        assert (
            block_indices.shape[0]
            == head_indices.shape[0]
            == position_indices.shape[0]
            == num_rows
        ), "scatter_packed_k index lengths must match k_flat rows"
        assert torch.all(head_lanes[:, 0] == head_lanes[:, 1]), (
            "scatter_packed_k pairs cannot cross KV-head boundaries"
        )
        assert torch.all(
            (~valid_lanes[:, 0]) | (position_lanes[:, 0] % 2 == 0)
        ) and torch.all((~valid_lanes[:, 1]) | (position_lanes[:, 1] % 2 == 1)), (
            "scatter_packed_k valid rows must match their packed lane parity"
        )
        both_valid = valid_lanes[:, 0] & valid_lanes[:, 1]
        same_pair = (
            (block_lanes[:, 0] == block_lanes[:, 1])
            & (head_lanes[:, 0] == head_lanes[:, 1])
            & (position_lanes[:, 0] // 2 == position_lanes[:, 1] // 2)
        )
        assert torch.all(~both_valid | same_pair), (
            "scatter_packed_k valid row pairs must share one packed coordinate"
        )

    packed_outer = k_cache.shape[-3]

    pair_valid = valid_lanes.any(dim=1)
    use_lane0 = valid_lanes[:, 0]

    block_pair = torch.where(use_lane0, block_lanes[:, 0], block_lanes[:, 1])
    head_pair = torch.where(use_lane0, head_lanes[:, 0], head_lanes[:, 1])
    outer_pair = torch.where(
        use_lane0, position_lanes[:, 0] // 2, position_lanes[:, 1] // 2
    )

    if validate_eager_contract:
        assert packed_outer > 0, "scatter_packed_k requires a non-empty cache block"
        assert torch.all(~pair_valid | (outer_pair < packed_outer)), (
            f"scatter_packed_k position exceeds the packed block extent "
            f"{packed_outer} (needs an even block_size)"
        )

    # All-null pairs are redirected to block-0 scratch, which makes them
    # byte-identical self-writes: the merge below re-selects the loaded bytes for
    # both lanes, so writers sharing a scratch slot all store the same value.
    block_pair = torch.where(pair_valid, block_pair, torch.zeros_like(block_pair))
    outer_pair = torch.where(pair_valid, outer_pair, torch.zeros_like(outer_pair))

    index = (block_pair, head_pair, outer_pair)
    merged_pairs = torch.where(
        valid_lanes.unsqueeze(1), k_pairs, k_cache[index]
    ).contiguous()
    k_cache.index_put_(index, merged_pairs)


def packed_fp8_viable_for_bucket(
    block_len: int, bs: int, q_head: int, s_active: int, s_prior: int
) -> bool:
    """Whether the packed FP8 decode kernel is usable for this bucket geometry.

    The decode kernel resizes block_len so blocks_per_batch is a multiple of
    (lnc * p_max). The packed layout pairs two consecutive tokens per BF16 slot,
    so it needs the resized block_len to stay >= 2; some bucket geometries
    (small SWA windows, or batch=1 which forces s_prior sharding) resize it down
    to 1. Mirror the kernel's resize math (shared replica) to decide statically,
    per compiled decode NEFF, whether to read the packed cache directly or
    un-swizzle it to the standard layout first.
    """
    if block_len <= 0 or s_prior <= 0:
        return False
    return _mask_resize_block_len(block_len, bs, q_head, s_active, s_prior) >= 2


# ---------------------------------------------------------------------------
# PyTorch fallback implementation
# ---------------------------------------------------------------------------


def _torch_attention_decode_impl(
    # -- input
    X: Tensor,
    X_hidden_dim_actual: Optional[int] = None,
    # -- rmsnorm X
    rmsnorm_X_enabled: bool = False,
    rmsnorm_X_eps: Optional[float] = None,
    rmsnorm_X_gamma: Optional[Tensor] = None,
    # -- qkv projections
    W_qkv: Tensor = None,
    bias_qkv: Optional[Tensor] = None,
    # -- Q/K processing: pre-RoPE RMSNorm
    rmsnorm_QK_pre_rope_enabled: bool = False,
    rmsnorm_QK_pre_rope_eps: float = 1e-6,
    rmsnorm_QK_pre_rope_W_Q: Optional[Tensor] = None,
    rmsnorm_QK_pre_rope_W_K: Optional[Tensor] = None,
    # -- Q/K processing: RoPE
    cos: Optional[Tensor] = None,
    sin: Optional[Tensor] = None,
    rope_contiguous_layout: bool = True,
    # -- Q/K processing: post-RoPE RMSNorm
    rmsnorm_QK_post_rope_enabled: bool = False,
    rmsnorm_QK_post_rope_eps: float = 1e-6,
    rmsnorm_QK_post_rope_W_Q: Optional[Tensor] = None,
    rmsnorm_QK_post_rope_W_K: Optional[Tensor] = None,
    # -- attention
    active_blocks_table: Optional[Tensor] = None,
    K_cache: Tensor = None,
    V_cache: Tensor = None,
    attention_mask: Tensor = None,
    sink: Tensor = None,
    softmax_scale: Optional[float] = None,
    # -- KV cache update
    update_cache: bool = False,
    kv_cache_update_idx: Optional[Tensor] = None,
    # -- FP8 KV cache quantization
    k_scale: Optional[Tensor] = None,
    v_scale: Optional[Tensor] = None,
    # -- packed FP8 K cache layout
    fp8_packed: bool = False,
    # -- output projection
    W_out: Optional[Tensor] = None,
    bias_out: Optional[Tensor] = None,
    # -- Attention Dependent DP
    attention_dp: int = 1,
    attention_dp_group: Optional[ProcessGroup] = None,
    attention_dp_rank: int = 0,
    attention_dp_rank_tensor: Optional[Tensor] = None,
    kv_needs_a2a: bool = False,
    # -- DCP Decode (Gather Q + LSE correction)
    dcp_size: int = 1,
    dcp_group=None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    PyTorch fallback implementation of the fused attention block for TKG.

    Supports **block KV cache** layout only, with single or multiple KV heads.

    Cache layout convention (matching the model code):
        - 4D: [num_blocks, kv_heads, block_len, d_head]
        - 3D: [num_blocks, block_len, d_head]  (kv_heads=1, pre-squeezed by caller)

    The implementation follows the same algorithmic stages as the NKI kernel:

        RMSNorm(X) → QKV projection → split Q/K/V → optional pre-RoPE
        RMSNorm → RoPE → optional post-RoPE RMSNorm → GQA attention with
        block KV cache → optional KV cache update → optional output projection.

    The attention mask is expected to come from ``gen_mask`` and therefore
    already contains the active-token causal pattern in its last ``S_tkg``
    rows.

    The KV cache update uses ``index_put_`` with (block, head, position)
    indices, matching the pattern used in the model's ``forward_decode``
    and ``forward_prefill`` methods.

    Returns:
        - update_cache=True: ``output`` only (the caches are written in place,
          so the new K/V tokens are redundant).
        - update_cache=False: ``(output, K_out, V_out)`` where K_out/V_out are
          the new K/V tokens for the caller to write into its cache.

        output:
            - With W_out: [B*S_tkg, H]
            - Without W_out: [B, q_heads, d_head, S_tkg]
        K_out: [d_head, B, S_tkg] new K tokens (update_cache=False only).
        V_out: [B, 1, S_tkg, d_head] new V tokens (update_cache=False only).
    """
    assert active_blocks_table is not None, (
        "PyTorch attention_block fallback only supports block KV cache "
        "(active_blocks_table must be provided)"
    )

    B, S_tkg, H = X.shape

    # ── Determine kv_heads and cache geometry ─────────────────────────────
    # Model convention:
    #   4D cache: [num_blocks, kv_heads_cache, block_len, d_head]
    #   3D cache: [num_blocks, block_len, d_head]  (kv_heads=1, squeezed)
    # Geometry is derived from V_cache because it is never packed; the packed
    # K_cache (fp8_packed=True) stores block_len//2 in a swizzled layout and so
    # cannot be read for block_len/d_head directly. kv_heads_cache comes from the
    # shared _infer_kv_heads (same convention as the kernel router).
    kv_heads_cache = _infer_kv_heads(V_cache)
    if V_cache.dim() == 4:
        num_blocks_total, _, block_len, d_head = V_cache.shape
    else:
        num_blocks_total, block_len, d_head = V_cache.shape

    if fp8_packed:
        # Packed K: [num_blocks, (kv_heads,) block_len // 2, d_head, 2] fp8.
        expected = (
            (num_blocks_total, kv_heads_cache, block_len // 2, d_head, 2)
            if V_cache.dim() == 4
            else (num_blocks_total, block_len // 2, d_head, 2)
        )
        assert tuple(K_cache.shape) == expected, (
            f"fp8_packed K_cache shape mismatch: expected {expected}, "
            f"got {tuple(K_cache.shape)}"
        )

    # kv_heads for QKV split: derived from weight. With kv_needs_a2a the weight
    # has fewer KV heads than the cache (weight is dependent-DP-sharded, cache stores
    # the gathered result). Derive from cache + attention_dp to get the projected count.
    kv_heads = kv_heads_cache // attention_dp if kv_needs_a2a else kv_heads_cache

    I = W_qkv.shape[1]
    q_heads = I // d_head - 2 * kv_heads
    num_kv_groups = q_heads // kv_heads  # GQA group size

    # num_blocks is the last axis for both [B, num_blocks] (kv_heads=1) and
    # [B, kv_heads, num_blocks] (GQA); shape[1] would wrongly pick kv_heads for a
    # 3D GQA table.
    num_blocks_per_seq = active_blocks_table.shape[-1]
    S_ctx = num_blocks_per_seq * block_len

    # The gather below flattens the table as [B, num_blocks] and lacks the
    # per-head-inner squeeze a 3D GQA table needs, so reject GQA here instead of
    # silently gathering wrong blocks (the kernel path handles kv_heads>1).
    if active_blocks_table.dim() > 2 and kv_heads_cache > 1:
        raise NotImplementedError(
            "Torch fallback for block-KV attention does not support GQA "
            f"(kv_heads={kv_heads_cache}) with a per-head block table; use the "
            "NKI kernel path or a single-KV-head (2D) table."
        )

    # ================================================================
    # Stage 1: Optional RMSNorm on input
    # ================================================================
    hidden = X
    if rmsnorm_X_enabled:
        eps = rmsnorm_X_eps if rmsnorm_X_eps is not None else 1e-3
        hidden = _torch_rms_norm(hidden, eps, rmsnorm_X_gamma, X_hidden_dim_actual)

    # ================================================================
    # Stage 2: QKV projection
    # ================================================================
    qkv = hidden @ W_qkv  # [B, S_tkg, I]
    if bias_qkv is not None:
        qkv = qkv + bias_qkv

    # ================================================================
    # Stage 3: Split Q, K, V and reshape to head layout
    # ================================================================
    q_end = q_heads * d_head
    k_end = q_end + kv_heads * d_head

    q = qkv[..., :q_end]  # [B, S_tkg, q_heads * d_head]
    k = qkv[..., q_end:k_end]  # [B, S_tkg, kv_heads * d_head]
    v = qkv[..., k_end:]  # [B, S_tkg, kv_heads * d_head]

    q = q.view(B, S_tkg, q_heads, d_head).transpose(1, 2)  # [B, q_heads, S_tkg, d_head]
    k = k.view(B, S_tkg, kv_heads, d_head).transpose(
        1, 2
    )  # [B, kv_heads, S_tkg, d_head]
    v = v.view(B, S_tkg, kv_heads, d_head).transpose(
        1, 2
    )  # [B, kv_heads, S_tkg, d_head]

    # ================================================================
    # Stage 4: Optional pre-RoPE RMSNorm on Q/K
    # ================================================================
    if rmsnorm_QK_pre_rope_enabled:
        q = _torch_rms_norm_heads(q, rmsnorm_QK_pre_rope_eps, rmsnorm_QK_pre_rope_W_Q)
        k = _torch_rms_norm_heads(k, rmsnorm_QK_pre_rope_eps, rmsnorm_QK_pre_rope_W_K)

    # ================================================================
    # Stage 5: Dependent DP all-to-all Q + select local K/V
    # Done before RoPE so RoPE only needs local batch's cos/sin.
    # ================================================================
    if attention_dp > 1:
        from vllm_neuron.functional import all_to_all

        # q: [DDP*B_local, q_heads_small, S_tkg, d_head]
        # Contiguous needed: transpose from QKV split leaves q non-contiguous
        # all-to-all: swap batch↔heads
        # → [B_local, DDP*q_heads_small, S_tkg, d_head]
        q = all_to_all(
            q.contiguous(),
            split_dim=0,
            concat_dim=1,
            group=attention_dp_group,
            rank=attention_dp_rank_tensor,
        )

        B_local = q.shape[0]

        if kv_needs_a2a:
            # KV also sharded across attention DP — a2a to gather all KV heads
            k = all_to_all(
                k.contiguous(),
                split_dim=0,
                concat_dim=1,
                group=attention_dp_group,
                rank=attention_dp_rank_tensor,
            )
            v = all_to_all(
                v.contiguous(),
                split_dim=0,
                concat_dim=1,
                group=attention_dp_group,
                rank=attention_dp_rank_tensor,
            )
        else:
            # KV fits in TP — just select local batch.
            # SPMD: when a runtime rank tensor is supplied, express the slice as
            # `index_select(arange(B_local) + rank * B_local)` so the graph is
            # rank-invariant. Otherwise fall back to the baked-int slice (MPMD).
            if attention_dp_rank_tensor is not None:
                # Keep the rank tensor 1-D (shape (1,)) — do NOT ``.reshape(())``
                # to a 0-D scalar. Under chunked prefill
                # (max_num_batched_tokens < max_model_len) torch.compile
                # constant-folds a 0-D reshape into a Python scalar and bakes a
                # per-rank slice into the FX graph, splintering the compile
                # cache (see fe1252e42 / shard_select). The 1-D form broadcasts
                # through arange + add without collapsing.
                offset = attention_dp_rank_tensor.to(torch.int64) * B_local
                idx = torch.arange(B_local, device=k.device, dtype=torch.int64) + offset
                k = k.index_select(0, idx)
                v = v.index_select(0, idx)
            else:
                k = k[attention_dp_rank * B_local : (attention_dp_rank + 1) * B_local]
                v = v[attention_dp_rank * B_local : (attention_dp_rank + 1) * B_local]

        # Update head counts for post-all-to-all state
        q_heads = q.shape[1]
        kv_heads = k.shape[1]
        num_kv_groups = q_heads // kv_heads
        B = B_local

    # ================================================================
    # Stage 6: RoPE (on local batch only when attention DP enabled)
    # ================================================================
    if cos is not None and sin is not None:
        q, k = _torch_apply_rope(q, k, cos, sin, rope_contiguous_layout)

    # ================================================================
    # Stage 6.5: Optional post-RoPE RMSNorm on Q/K
    # ================================================================
    if rmsnorm_QK_post_rope_enabled:
        q = _torch_rms_norm_heads(q, rmsnorm_QK_post_rope_eps, rmsnorm_QK_post_rope_W_Q)
        k = _torch_rms_norm_heads(k, rmsnorm_QK_post_rope_eps, rmsnorm_QK_post_rope_W_K)

    # ================================================================
    # Stage 6.8: DCP AllGather Q across DCP group
    # Each rank has Q for its local heads. After gather, every rank has
    # Q for all heads in the DCP group (the KV replica set).
    # ================================================================
    if dcp_size > 1:
        q = dcp_group.all_gather(q.contiguous(), dim=1)
        q_heads = q.shape[1]
        num_kv_groups = q_heads // kv_heads

    # ================================================================
    # Stage 7: Attention with block KV cache
    # ================================================================

    # 7a. Gather K/V from block cache → [B, kv_heads_cache, S_ctx, d_head]
    # Use kv_heads_cache (not kv_heads) since the cache stores the full
    # per-TP heads, which may differ from the projected count with kv_needs_a2a.
    #
    # torch fallback only: clamp -1 sentinels back to 0 for KV block indexing
    safe_idx = torch.where(
        active_blocks_table < 0,
        torch.zeros_like(active_blocks_table),
        active_blocks_table,
    )
    flat_idx = safe_idx.long().reshape(-1)  # [B * num_blocks_per_seq]

    # When the cache is FP8, cast to compute dtype before indexing since
    # PyTorch CPU does not support fancy indexing on float8 dtypes.
    # Dequantization scales are already fused into softmax_scale (for K)
    # and W_out (for V) by the caller.
    #
    # For the packed FP8 K cache, un-swizzle to the standard
    # [num_blocks, (kv_heads,) block_len, d_head] layout first so the gather
    # logic below is identical to the unpacked path.
    K_cache_read = _unswizzle_packed_k(K_cache) if fp8_packed else K_cache
    K_blocks = K_cache_read.to(X.dtype)[flat_idx]
    V_blocks = V_cache.to(X.dtype)[flat_idx]

    if V_cache.dim() == 4:
        # 4D: [num_blocks, kv_heads_cache, block_len, d_head]
        K_blocks = K_blocks.view(
            B, num_blocks_per_seq, kv_heads_cache, block_len, d_head
        )
        V_blocks = V_blocks.view(
            B, num_blocks_per_seq, kv_heads_cache, block_len, d_head
        )
        K_gathered = K_blocks.permute(0, 2, 1, 3, 4).reshape(
            B, kv_heads_cache, S_ctx, d_head
        )
        V_gathered = V_blocks.permute(0, 2, 1, 3, 4).reshape(
            B, kv_heads_cache, S_ctx, d_head
        )
    else:
        # 3D: [num_blocks, block_len, d_head] (kv_heads=1)
        K_gathered = K_blocks.reshape(B, S_ctx, d_head).unsqueeze(
            1
        )  # [B, 1, S_ctx, d_head]
        V_gathered = V_blocks.reshape(B, S_ctx, d_head).unsqueeze(
            1
        )  # [B, 1, S_ctx, d_head]

    # 7b. Place active K/V into the last S_tkg positions.
    #     The mask's last S_tkg rows (from gen_mask's active_mask) encode
    #     the causal pattern among active tokens; stale cache data in
    #     those positions is masked out.
    #
    #     When FP8 KV cache is enabled, round-trip active k/v through FP8
    #     quantization so the quantization noise matches the kernel path.
    #     k_scale/v_scale are reciprocal scales: quantize via (tensor * scale).

    # k, v: [B, kv_heads, S_tkg, d_head]
    if k_scale is not None:
        from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX

        k_fp8 = (
            (k * k_scale).clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(torch.float8_e4m3fn)
        )
        v_fp8 = (
            (v * v_scale).clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(torch.float8_e4m3fn)
        )
        K_gathered[:, :, -S_tkg:, :] = k_fp8.to(X.dtype)
        V_gathered[:, :, -S_tkg:, :] = v_fp8.to(X.dtype)
        k = k_fp8
        v = v_fp8
    else:
        K_gathered[:, :, -S_tkg:, :] = k
        V_gathered[:, :, -S_tkg:, :] = v

    # 7c. GQA expansion: [B, kv_heads, S_ctx, d_head] → [B, q_heads, S_ctx, d_head]
    K_full = K_gathered.repeat_interleave(
        num_kv_groups, dim=1
    )  # [B, q_heads, S_ctx, d_head]
    V_full = V_gathered.repeat_interleave(
        num_kv_groups, dim=1
    )  # [B, q_heads, S_ctx, d_head]

    # 7d. Compute attention scores
    scale = softmax_scale if softmax_scale is not None else d_head**-0.5
    # q: [B, q_heads, S_tkg, d_head] × K^T: [B, q_heads, d_head, S_ctx]
    scores = (
        torch.matmul(q, K_full.transpose(-2, -1)) * scale
    )  # [B, q_heads, S_tkg, S_ctx]

    # 7e. Apply attention mask
    # attention_mask: [S_ctx, B, q_heads, S_tkg] → [B, q_heads, S_tkg, S_ctx]
    resized_bl = _mask_resize_block_len(block_len, B, q_heads, S_tkg, S_ctx)
    if resized_bl > 0:
        # gen_mask_tkg stores block-KV masks in its partition-major HBM layout.
        # Convert that layout back to the block/offset order used by K_gathered.
        # DCP uses the same mask layout; only its validity threshold differs.
        mask = (
            attention_mask.permute(1, 2, 3, 0)
            .reshape(-1, resized_bl, _MASK_P_MAX)
            .swapaxes(-1, -2)
            .reshape(B, q_heads, S_tkg, S_ctx)
        )
    else:
        mask = attention_mask.permute(1, 2, 3, 0)

    scores = scores.to(torch.float32)
    scores = scores.masked_fill(mask == 0, float("-inf"))

    # 7f. Attention sink
    if sink is not None:
        # sink shape: [q_heads, 1] (pre-gathered across attention DP at load time).
        sink_score = (
            sink.to(torch.float32).view(1, q_heads, 1, 1).expand(B, -1, S_tkg, -1)
        )
        scores = torch.cat([scores, sink_score], dim=-1)  # [B, q_heads, S_tkg, S_ctx+1]

    # 7g. Softmax → weighted sum
    # With DCP, extract partial LSE before softmax for cross-rank correction.
    if dcp_size > 1:
        partial_lse = torch.logsumexp(scores, dim=-1)  # [B, q_heads, S_tkg]

    attn_weights = torch.softmax(scores, dim=-1).to(q.dtype)

    # Strip the sink column so the matmul only touches real V positions
    if sink is not None:
        attn_weights = attn_weights[..., :-1]  # [B, q_heads, S_tkg, S_ctx]

    attn_out = torch.matmul(attn_weights, V_full)  # [B, q_heads, S_tkg, d_head]

    # ================================================================
    # Stage 7.5a: DCP LSE correction + ReduceScatter
    # Each rank computed attention over its local KV shard. Combine partials
    # across DCP ranks using LSE-weighted correction, then reduce-scatter
    # to distribute heads back to their owning ranks.
    # ================================================================
    if dcp_size > 1:
        all_lse = dcp_group.all_gather(
            partial_lse.contiguous(), dim=0
        )  # [dcp_size * B, q_heads, S_tkg]
        all_lse = all_lse.view(
            dcp_size, B, q_heads, S_tkg
        )  # [dcp_size, B, q_heads, S_tkg]
        global_lse = torch.logsumexp(all_lse, dim=0)  # [B, q_heads, S_tkg]

        correction = torch.exp(
            partial_lse - global_lse
        )  # [B, q_heads, S_tkg] (float32)
        attn_out = (attn_out.float() * correction.unsqueeze(-1)).to(attn_out.dtype)

        attn_out = dcp_group.reduce_scatter(
            attn_out.contiguous(), dim=1
        )  # [B, q_heads_local, S_tkg, d_head]
        q_heads = attn_out.shape[1]

    # ================================================================
    # Stage 7.5b: Dependent DP reverse all-to-all
    # ================================================================
    if attention_dp > 1:
        # attn_out: [B_local, DDP*q_heads_small, S_tkg, d_head]
        # → [DDP*B_local, q_heads_small, S_tkg, d_head]
        attn_out = all_to_all(
            attn_out,
            split_dim=1,
            concat_dim=0,
            group=attention_dp_group,
            rank=attention_dp_rank_tensor,
        )

        # Restore B to DDP*B_local for O projection and return
        B = attn_out.shape[0]
        q_heads = attn_out.shape[1]

    # ================================================================
    # Stage 8: KV cache update
    # ================================================================
    if update_cache:
        # In-place KV cache update via index_put_.
        # kv_cache_update_idx: [B_attn, S_tkg] (caller-sliced when attention_dp > 1
        # — the NKI kernel expects the same B_attn = B/KVDP layout).
        # slot = block_idx * block_len + position_in_block.
        assert kv_cache_update_idx is not None, (
            "kv_cache_update_idx must be provided when update_cache=True"
        )
        # DCP non-owning ranks receive -1 sentinels in slot_mapping. The NKI
        # kernel handles this via oob_mode.skip in scatter DMA. The caller
        # casts to uint32 for the kernel (so -1 becomes 0xFFFFFFFF). Convert
        # back to signed and detect sentinels, then redirect them to the LAST
        # cache slot (block num_blocks_total-1, position block_len-1) rather
        # than slot 0. The cache is over-allocated, so the last slot is never a
        # real token's write target — writing there is a harmless no-op. This
        # matches the pre-#2306 model-level behavior, where a raw -1 index
        # negative-indexed to the last block. (Redirecting to slot 0 instead
        # corrupts the first token's KV.)
        max_slot = num_blocks_total * block_len
        slot_mapping = kv_cache_update_idx.to(torch.int64).reshape(-1)  # [B*S_tkg]
        is_sentinel = (slot_mapping < 0) | (slot_mapping >= max_slot)
        slot_mapping = torch.where(
            is_sentinel,
            torch.full_like(slot_mapping, max_slot - 1),
            slot_mapping,
        )

        block_indices = (slot_mapping // block_len).repeat(kv_heads)
        position_indices = (slot_mapping % block_len).repeat(kv_heads)
        head_indices = torch.arange(
            kv_heads, dtype=torch.long, device=K_cache.device
        ).repeat_interleave(slot_mapping.shape[0])

        # k, v: [B, kv_heads, S_tkg, d_head] → [kv_heads, B*S_tkg, d_head]
        k_flat = k.transpose(0, 1).reshape(-1, d_head).to(K_cache.dtype)
        v_flat = v.transpose(0, 1).reshape(-1, d_head).to(V_cache.dtype)

        # Sentinel slots were redirected to the harmless last cache slot above
        # (not slot 0), so their real K/V values write to a never-read target —
        # no need to zero them out (zeroing slot 0 corrupted the first token's
        # KV).
        if fp8_packed:
            # K_cache is the swizzled packed FP8 layout. Un-swizzle into a
            # standard [num_blocks, (kv_heads,) block_len, d_head] buffer,
            # scatter the new tokens with the same (block, head, position)
            # indices as the unpacked path, then re-swizzle and write back in
            # place. The scatter runs in the compute dtype to avoid fp8 fancy
            # indexing (unsupported on CPU); fp8→compute→fp8 is lossless since
            # k is already fp8-quantized.
            K_unpacked = _unswizzle_packed_k(K_cache).to(X.dtype)
            k_flat_c = k_flat.to(X.dtype)
            if V_cache.dim() == 4:
                K_unpacked.index_put_(
                    (block_indices, head_indices, position_indices), k_flat_c
                )
                V_cache.index_put_(
                    (block_indices, head_indices, position_indices), v_flat
                )
            else:
                K_unpacked.index_put_((block_indices, position_indices), k_flat_c)
                V_cache.index_put_((block_indices, position_indices), v_flat)
            K_cache.copy_(_swizzle_packed_k(K_unpacked.to(K_cache.dtype)))
        elif V_cache.dim() == 4:
            K_cache.index_put_((block_indices, head_indices, position_indices), k_flat)
            V_cache.index_put_((block_indices, head_indices, position_indices), v_flat)
        else:
            # 3D: [num_blocks, block_len, d_head] (kv_heads=1, pre-squeezed)
            K_cache.index_put_((block_indices, position_indices), k_flat)
            V_cache.index_put_((block_indices, position_indices), v_flat)
        K_out = V_out = None
    else:
        # No cache update: return new K/V tokens.
        # Use k.shape[0] (the local batch), NOT B: with attention DP, Stage 5
        # slices k/v to B_local, while Stage 7.5b restores B to DDP*B_local for
        # the output projection. K_new: [d_head, B_local*kv_heads, S_tkg]
        # (matches NKI kernel output).
        B_kv = k.shape[0]
        k_for_return = k.reshape(B_kv * kv_heads, S_tkg, d_head)
        K_out = k_for_return.permute(2, 0, 1).contiguous()
        # V_out: [B_kv, kv_heads, S_tkg, d_head]
        V_out = v

    # ================================================================
    # Stage 9: Output projection
    # ================================================================
    if W_out is not None:
        # attn_out: [B, q_heads, S_tkg, d_head] → [B*S_tkg, q_heads*d_head]
        attn_flat = attn_out.transpose(1, 2).reshape(B * S_tkg, q_heads * d_head)
        output = attn_flat @ W_out  # [B*S_tkg, H]
        if bias_out is not None:
            output = output + bias_out
    else:
        # Without projection: return [B, q_heads, d_head, S_tkg] (matches NKI kernel)
        output = attn_out.permute(0, 1, 3, 2).contiguous()

    # When update_cache=True the caches are written in place, so the new K/V
    # tokens are redundant — return only the attention output. The
    # update_cache=False path still returns the new tokens for the caller to
    # write into its cache.
    if update_cache:
        return output
    return output, K_out, V_out


def _fold_gqa_block_kv_mask(
    attention_mask: Optional[Tensor],
    X: Tensor,
    W_qkv: Tensor,
    V_cache: Optional[Tensor],
    active_blocks_table: Optional[Tensor],
) -> Optional[Tensor]:
    """Give a GQA block-KV mask the layout of the kernel's folded batch.

    The caller builds the mask for ``(B, q_heads)``. With ``kv_heads > 1`` the
    kernel folds KV heads into the batch (``B * kv_heads`` batches of
    ``q_heads // kv_heads`` heads) and resizes block_len for that folded shape.
    The resize can differ from the one the mask was built with, for example at
    an odd batch with ``B * q_heads * S >= 128``. The kernel then reads prior
    and active rows from the wrong positions. The output is wrong, and nothing
    fails. Move the mask to the folded layout here, so every caller gets it.
    """
    if (
        attention_mask is None
        or attention_mask.dim() != 4
        or V_cache is None
        or V_cache.dim() != 4
        or active_blocks_table is None
    ):
        return attention_mask
    kv_heads = V_cache.shape[1]
    if kv_heads <= 1:
        return attention_mask
    block_len = V_cache.shape[2]
    B, S_tkg, _ = X.shape
    d_head = V_cache.shape[-1]
    q_heads = W_qkv.shape[1] // d_head - 2 * kv_heads
    s_ctx = active_blocks_table.shape[-1] * block_len
    if attention_mask.shape != (s_ctx, B, q_heads, S_tkg):
        return attention_mask
    from_bl = _mask_resize_block_len(block_len, B, q_heads, S_tkg, s_ctx)
    to_bl = _mask_resize_block_len(
        block_len, B * kv_heads, q_heads // kv_heads, S_tkg, s_ctx
    )
    return relayout_block_kv_mask(attention_mask, from_bl, to_bl)


# ---------------------------------------------------------------------------
# Kernel eligibility check
# ---------------------------------------------------------------------------


def _infer_kv_heads(V_cache: Tensor) -> int:
    """Derive kv_heads from the KV-cache layout — the single source of truth.

    Model convention (V_cache is never packed, so geometry is read from it):
      4D ``[num_blocks, kv_heads, block_len, d_head]`` → ``kv_heads`` at dim 1,
      3D ``[num_blocks, block_len, d_head]``            → ``kv_heads == 1``.
    The NKI kernel infers kv_heads from the cache shape the same way; the kernel
    router and the torch fallback both call this so the cache-layout convention
    lives in exactly one place.
    """
    assert V_cache is not None, (
        "attention_decode requires a 3D/4D V_cache to infer kv_heads; got None."
    )
    assert V_cache.dim() in (3, 4), (
        f"V_cache must be 3D [num_blocks, block_len, d_head] or 4D "
        f"[num_blocks, kv_heads, block_len, d_head]; got {V_cache.dim()}D "
        f"{tuple(V_cache.shape)}."
    )
    return 1 if V_cache.dim() == 3 else V_cache.shape[1]


def _can_use_attention_block_kernel(
    X: Tensor,
    V_cache: Optional[Tensor],
    active_blocks_table: Optional[Tensor] = None,
    *,
    attention_dp: int = 1,
) -> bool:
    """
    Check whether the NKI attention_block_tkg kernel can be used.

    Returns ``True`` when every kernel constraint is satisfied and the tensors
    reside on a NeuronCore device or CPU with the NKI simulator; ``False`` otherwise (→ PyTorch fallback).
    """
    if not can_run_kernel(X):
        return False

    # TODO: Enable once the attention-DP + DCP kernel combination is covered
    # end-to-end. Until then, preserve the established torch attention-DP path.
    if attention_dp > 1:
        return False

    _B, _S_tkg, H = X.shape

    # H must be a multiple of 128
    if H % _PMAX != 0:
        return False

    # GQA (kv_heads > 1) routes to the kernel only when the caller passes a per-head
    # (3D [B, kv_heads, num_blocks]) block table. The kernel infers kv_heads from the
    # 4D cache but still folds kv_heads into the batch dim and reshapes the table to
    # [B*kv_heads, num_blocks] (attention_block_tkg.py), which a 2D [B, num_blocks]
    # pool table cannot satisfy.
    #   - The BF16 GQA decoders (qwen3_vl model_bf16) pass a 2D table +
    #     4D cache, so they stay on the torch fallback (its 4D-cache gather handles it).
    #   - The native-MX decoder (Qwen3VLTextAttentionMX) builds a per-head 3D table
    #     (build_per_head_block_table) and DOES reach the kernel — the only production
    #     GQA caller that does.
    kv_heads = _infer_kv_heads(V_cache)
    if kv_heads > 1:
        if active_blocks_table is None or active_blocks_table.dim() < 3:
            return False

    d_head = V_cache.shape[-1]

    if d_head % 2 != 0:
        return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def attention_decode(
    # -- input
    X: Tensor,
    X_hidden_dim_actual: Optional[int] = None,
    # -- rmsnorm X
    rmsnorm_X_enabled: bool = False,
    rmsnorm_X_eps: Optional[float] = None,
    rmsnorm_X_gamma: Optional[Tensor] = None,
    # -- qkv projections
    W_qkv: Tensor = None,
    bias_qkv: Optional[Tensor] = None,
    quantization_type_qkv: QuantizationType = QuantizationType.NONE,
    weight_dequant_scale_qkv: Optional[Tensor] = None,
    input_dequant_scale_qkv: Optional[Tensor] = None,
    # -- Q/K processing: pre-RoPE RMSNorm
    rmsnorm_QK_pre_rope_enabled: bool = False,
    rmsnorm_QK_pre_rope_eps: float = 1e-6,
    rmsnorm_QK_pre_rope_W_Q: Optional[Tensor] = None,
    rmsnorm_QK_pre_rope_W_K: Optional[Tensor] = None,
    # -- Q/K processing: RoPE
    cos: Optional[Tensor] = None,
    sin: Optional[Tensor] = None,
    rope_contiguous_layout: bool = True,
    # -- Q/K processing: post-RoPE RMSNorm
    rmsnorm_QK_post_rope_enabled: bool = False,
    rmsnorm_QK_post_rope_eps: float = 1e-6,
    rmsnorm_QK_post_rope_W_Q: Optional[Tensor] = None,
    rmsnorm_QK_post_rope_W_K: Optional[Tensor] = None,
    # -- attention
    K_cache_transposed: bool = False,
    active_blocks_table: Optional[Tensor] = None,
    K_cache: Tensor = None,
    V_cache: Tensor = None,
    attention_mask: Optional[Tensor] = None,
    sink: Optional[Tensor] = None,
    softmax_scale: Optional[float] = None,
    # -- in-kernel mask generation (fused path)
    pos_ids: Optional[Tensor] = None,
    swa_start_pos_ids: Optional[Tensor] = None,
    # -- KV cache update
    update_cache: bool = False,
    # With update_cache=True, also return the UPDATED K/V CACHE tensors.
    # REQUIRED when K_cache/V_cache is a computed temporary (the packed-FP8 fallback
    # bucket un-swizzles the cache first): the kernel's aliased in-place write is
    # silently dropped for a non-input tensor, so the caller must consume the alias
    # OUTPUT rather than re-reading the tensor it passed in.
    return_updated_kv: bool = False,
    kv_cache_update_idx: Optional[Tensor] = None,
    k_scale: Optional[Tensor] = None,
    v_scale: Optional[Tensor] = None,
    # -- packed FP8 K cache layout
    fp8_packed: bool = False,
    # -- output projection
    W_out: Optional[Tensor] = None,
    bias_out: Optional[Tensor] = None,
    quantization_type_out: QuantizationType = QuantizationType.NONE,
    weight_dequant_scale_out: Optional[Tensor] = None,
    input_dequant_scale_out: Optional[Tensor] = None,
    transposed_out: bool = False,
    # -- output control
    out_in_sb: bool = False,
    skip_attention: bool = False,
    # -- STATIC_MX layout flag
    is_h_transposed_by_4: bool = False,
    # -- sbm control (optional)
    sbm_lower_bound: Optional[int] = None,
    sbm_upper_bound: Optional[int] = None,
    sbm_use_auto_alloc: bool = True,
    sbm_default_stack_alloc: bool = True,
    # -- Attention Dependent DP (decode-only Q/O sharding across DP)
    attention_dp: int = 1,
    attention_dp_group: Optional[ProcessGroup] = None,
    attention_dp_rank: int = 0,
    attention_dp_rank_tensor: Optional[Tensor] = None,
    kv_needs_a2a: bool = False,
    dcp_size: int = 1,
    dcp_group=None,
    dcp_collective_mode: str = "auto",
    # -- fused-mask DCP active-token ownership
    dcp_active_owner: Optional[Tensor] = None,
    # -- explicit FP8 fused-mask selection
    fp8_fused_mask_allowed: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Fused Attention Block for Token Generation (TKG).

    Automatically selects between NKI kernel and PyTorch fallback.

    Dispatches to:
    - NKI kernel: When all constraints are satisfied and running on Neuron
    - PyTorch implementation: When constraints are violated or not on Neuron
      (block KV cache only; raises for unsupported features)

    Performs end-to-end attention: optional RMSNorm → QKV projection →
    optional pre-RoPE RMSNorm → optional RoPE → optional post-RoPE RMSNorm →
    attention → KV-cache update → optional output projection.

    Dimensions:
        B:       Batch size (≤ 16 recommended)
        S_tkg:   Number of new tokens (≤ 8)
        S_ctx:   Current KV-cache sequence length
        S_max:   Maximum KV-cache capacity
        H:       Hidden dimension (multiple of 128)
        d_head:  Head dimension (must be even, ≤ 128)
        q_heads: Number of query heads
        kv_heads: Number of key/value heads (≥1). Not a parameter — the NKI kernel
                 infers it from the K/V-cache shape (4D cache → dim 1; 3D → 1). GQA
                 (kv_heads > 1) reaches the kernel only with a per-head 3D block
                 table; otherwise it routes to the torch fallback.

    Args:
        X:                  [B, S_tkg, H]  Input hidden states.
        X_hidden_dim_actual: Actual H if X is zero-padded (None = H).
        rmsnorm_X_enabled:  Apply RMSNorm to X before QKV.
        rmsnorm_X_eps:      RMSNorm epsilon (default 1e-3).
        rmsnorm_X_gamma:    [1, H]  RMSNorm gamma weights.
        W_qkv:              [H, d_head*(q_heads+2*kv_heads)]  QKV projection weights.
        bias_qkv:           [1, d_head*(q_heads+2*kv_heads)]  Optional QKV bias.
        quantization_type_qkv: NONE or STATIC.
        weight_dequant_scale_qkv: [PMAX, 1] weight scale (STATIC only).
        input_dequant_scale_qkv:  [PMAX, 1] input scale (STATIC only).
        rmsnorm_QK_pre_rope_enabled:  Pre-RoPE RMSNorm on Q/K.
        rmsnorm_QK_pre_rope_eps:      Epsilon.
        rmsnorm_QK_pre_rope_W_Q:      [1, d_head]  Q gamma.
        rmsnorm_QK_pre_rope_W_K:      [1, d_head]  K gamma.
        cos:                [d_head//2, B, S_tkg]  RoPE cos (None = skip).
        sin:                [d_head//2, B, S_tkg]  RoPE sin (None = skip).
        rope_contiguous_layout: True=contiguous halves, False=interleaved.
        rmsnorm_QK_post_rope_enabled: Post-RoPE RMSNorm on Q/K.
        rmsnorm_QK_post_rope_eps:     Epsilon.
        rmsnorm_QK_post_rope_W_Q:     [1, d_head]  Q gamma.
        rmsnorm_QK_post_rope_W_K:     [1, d_head]  K gamma.
        K_cache_transposed: K-cache layout flag (NKI kernel only).
        active_blocks_table: int32 block indices (block KV). [B, num_blocks] for
                            a single shared KV-head table; the GQA per-head
                            variant [B, kv_heads, num_blocks] is also accepted
                            (required to route GQA to the NKI kernel).
        K_cache:            Key cache in HBM.
            - 3D: [num_blocks, block_len, d_head]
            - 4D: [num_blocks, kv_heads, block_len, d_head]
            - packed (fp8_packed=True): the swizzled FP8 layout
              [num_blocks, block_len//2, d_head, 2] (4D) or
              [num_blocks, kv_heads, block_len//2, d_head, 2] (5D). Two
              consecutive sequence positions are packed into the trailing
              size-2 dim so the kernel can bf16-reinterpret + DMA-transpose
              instead of using the slower PE-transpose FP8 path.
        V_cache:            Value cache in HBM. Same layout as the *unpacked*
            K_cache (3D/4D [num_blocks, (kv_heads,) block_len, d_head]) even
            when fp8_packed=True — only K is swizzled, V is never packed.
        attention_mask:     Attention mask. Shape depends on whether ``pos_ids``
                            is provided:
                              - ``pos_ids is None``: [S_ctx, B, q_heads, S_tkg]
                                full pre-generated mask (legacy path).
                              - ``pos_ids is not None``: [S_tkg, B, q_heads, S_tkg]
                                active-only causal mask. If ``None``, the
                                wrapper builds a default triu causal mask.
        sink:               [q_heads, 1]  Attention-sink scores (NKI kernel only).
        softmax_scale:      Custom scale (None → 1/√d_head).
        pos_ids:            [B, S_tkg] @ HBM, float32 prior-token count for each
                            active token: absolute sequence position normally,
                            rank-local filled count under DCP. When provided, the
                            kernel generates the prior causal mask on-chip via
                            ``iota < pos_id`` and ``attention_mask`` carries only
                            the active-only portion. Block-KV only.
        swa_start_pos_ids:  [B, S_tkg] @ HBM, float32 per-query SWA window start
                            (inclusive). Requires ``pos_ids``. When provided,
                            the kernel generates a banded mask
                            ``start_pos <= kv_pos < pos_id``.
        update_cache:       Write new K/V tokens into the cache.
        return_updated_kv:  With update_cache=True, return
                            ``(output, K_cache, V_cache)`` — the UPDATED caches —
                            instead of ``output`` alone. Required when the caches
                            passed in are computed temporaries rather than buffers,
                            because the aliased in-place write cannot be threaded
                            back to a non-input tensor.
        kv_cache_update_idx: [B, 1] uint32  Cache write positions.
        k_scale:            [PMAX, 1] or [1, 1]  FP8 K quant scale (NKI kernel only).
        v_scale:            [PMAX, 1] or [1, 1]  FP8 V quant scale (NKI kernel only).
        fp8_packed:         When True, K_cache uses the swizzled packed FP8
                            layout (see K_cache above). Requires a float8_e4m3
                            K_cache and an even block_len. V_cache stays
                            unpacked. Typically paired with k_scale/v_scale.
        W_out:              [q_heads*d_head, H]  Output projection weights.
        bias_out:           [1, H]  Output projection bias.
        quantization_type_out: NONE or STATIC.
        weight_dequant_scale_out: [PMAX, 1]  (STATIC only).
        input_dequant_scale_out:  [PMAX, 1]  (STATIC only).
        transposed_out:     Transpose output layout (requires W_out).
        out_in_sb:          Return output in SBUF (NKI kernel only).
        skip_attention:     Skip attention (NKI kernel only).
        sbm_lower_bound:    Optional SBUF lower bound.
        sbm_upper_bound:    Optional SBUF upper bound.
        sbm_use_auto_alloc: Use auto-allocation for SBUF.
        sbm_default_stack_alloc: Default stack allocation.
        attention_dp:       Attention-DP degree. Values greater than one use the
                            torch path until attention DP and DCP are verified
                            together on the kernel.
        attention_dp_group: Process group used by the attention-DP torch path.
        attention_dp_rank: Eager fallback rank within ``attention_dp_group``.
        attention_dp_rank_tensor: Runtime rank for an SPMD graph.
        kv_needs_a2a:       Whether attention DP must exchange K/V heads.
        dcp_size:           Decode-context-parallel degree. DCP shards the KV
                            sequence across ranks. Kernel-eligible calls use the
                            NKI DCP path; the torch path implements the same
                            all-gather/LSE-correction/reduce-scatter algorithm.
        dcp_group:          vLLM DCP process group. Required when
                            ``dcp_size > 1`` and used by both implementations.
        dcp_collective_mode: Kernel DCP output collective: ``auto``,
                            ``reduce_scatter``, or ``all_to_all``.
        dcp_active_owner:   [B] or [B, S_tkg] DCP active-token owner gate for
                            the fused ``pos_ids`` path.
        fp8_fused_mask_allowed: Keep ``pos_ids`` on the FP8 fused-mask path
                            instead of rerouting to the external mask.

    Returns:
        - update_cache=True: ``output`` only — the K/V caches are written in
          place (in-kernel scatter DMA, or index_put_ on the torch fallback),
          so no cache tensors are returned. With return_updated_kv=True this
          returns ``(output, K_cache, V_cache)`` instead; see that argument.
        - update_cache=False: ``(output, K_out, V_out)`` where K_out/V_out are
          the new K/V tokens for the caller to write into its cache.

        output: Attention output (shape depends on W_out / transposed_out).

    Torch Fallback Constraints (raises NotImplementedError if violated):
        - Requires block KV cache (active_blocks_table must be provided)
        - Quantization not supported
        - FP8 KV cache quantization not supported
        - out_in_sb not supported
        - skip_attention not supported
        - sink not supported
    """

    if dcp_size > 1 and dcp_group is None:
        raise ValueError("dcp_group is required when dcp_size > 1")

    # The NKI kernel relies on int32 with -1 sentinels for inactive blocks
    if active_blocks_table is not None:
        assert active_blocks_table.dtype == torch.int32, (
            f"active_blocks_table must be int32 with -1 padding, got "
            f"{active_blocks_table.dtype}"
        )

    # kv_heads (GQA) is derived from the KV cache layout — never a caller argument
    # (shared with the kernel router + torch fallback via _infer_kv_heads).
    kv_heads = _infer_kv_heads(V_cache)

    # Validate the GQA head split early (clear error here vs an opaque NKI failure
    # downstream). Skip when kv_needs_a2a / attention_dp > 1: there W_qkv is sized by
    # the smaller pre-a2a KV count while V_cache.shape[1] is the larger gathered
    # count, so q_heads can't be derived this way — and that path uses the torch
    # fallback (which re-derives kv_heads = kv_heads_cache // attention_dp) anyway.
    # NOTE: assumes v_head_dim == qk_head_dim (so V_cache.shape[-1] is the per-head
    # width of every Q/K/V head in the fused W_qkv out-dim) — holds for all current
    # GQA decoders; a future v_head_dim != qk_head_dim model would need W_qkv split
    # by qk_head_dim here instead.
    if kv_heads > 1 and not kv_needs_a2a and attention_dp == 1:
        q_heads_total = W_qkv.shape[1] // V_cache.shape[-1] - 2 * kv_heads
        if q_heads_total % kv_heads != 0:
            raise ValueError(
                f"GQA requires q_heads ({q_heads_total}) divisible by kv_heads "
                f"({kv_heads}); check W_qkv ({tuple(W_qkv.shape)}) vs V_cache "
                f"({tuple(V_cache.shape)})."
            )

    # pos_ids selects on-chip prior-mask generation; attention_mask then holds
    # only the active-token overlay.
    if swa_start_pos_ids is not None and pos_ids is None:
        raise ValueError("swa_start_pos_ids requires pos_ids to also be provided")
    if pos_ids is not None:
        if active_blocks_table is None:
            raise ValueError(
                "pos_ids (fused mask-gen) requires block KV cache "
                "(active_blocks_table must be provided)"
            )
        B_x, S_tkg_x, _ = X.shape
        d_head_local = V_cache.shape[-1]
        q_heads_local = W_qkv.shape[1] // d_head_local - 2 * kv_heads
        # Derive block_len from V_cache: it is never packed, so its block_len
        # dim is the true logical block length (the packed K_cache stores
        # block_len//2 in its swizzled layout, so K_cache is unreliable here).
        block_len_local = V_cache.shape[1] if V_cache.dim() == 3 else V_cache.shape[2]
        # num_blocks is the last axis for both layouts: [B, num_blocks] (kv_heads=1)
        # and [B, kv_heads, num_blocks] (GQA). shape[1] would wrongly pick kv_heads.
        S_ctx_local = active_blocks_table.shape[-1] * block_len_local
        # Fused FP8 changes softmax/PV accumulation order, so only callers with
        # bucket-specific validation may bypass the external mask.
        if k_scale is not None and not fp8_fused_mask_allowed:
            from vllm_neuron.functional.attention.attention_decode_mask import (
                gen_attention_decode_mask,
            )

            attention_mask = gen_attention_decode_mask(
                pos_ids=pos_ids.reshape(1, B_x * S_tkg_x).to(torch.float32),
                bs=B_x,
                q_head=q_heads_local,
                s_active=S_tkg_x,
                s_prior=S_ctx_local,
                start_pos=swa_start_pos_ids.reshape(1, B_x * S_tkg_x).to(torch.float32)
                if swa_start_pos_ids is not None
                else None,
                block_len=block_len_local,
            )
            pos_ids = None
            swa_start_pos_ids = None
        elif attention_mask is None:
            # DCP gathers Q heads; the owner gate prevents active-token duplication
            # in the CP LSE combine.
            q_heads_active = q_heads_local * dcp_size
            attention_mask = _build_default_active_mask(
                B=B_x,
                S_tkg=S_tkg_x,
                q_heads=q_heads_active,
                device=X.device,
                dcp_active_owner=dcp_active_owner,
            )
        else:
            assert attention_mask.shape[0] == S_tkg_x, (
                f"With pos_ids, attention_mask dim 0 must equal S_tkg "
                f"({S_tkg_x}); got {tuple(attention_mask.shape)}"
            )
            assert attention_mask.shape[3] == S_tkg_x, (
                f"With pos_ids, attention_mask dim 3 must equal S_tkg "
                f"({S_tkg_x}); got {tuple(attention_mask.shape)}"
            )

    can_use_kernel = _can_use_attention_block_kernel(
        X=X,
        V_cache=V_cache,
        active_blocks_table=active_blocks_table,
        attention_dp=attention_dp,
    )

    # 2D-table GQA is already routed to the torch fallback by
    # _can_use_attention_block_kernel.

    if can_use_kernel:
        if pos_ids is None and dcp_size == 1:
            attention_mask = _fold_gqa_block_kv_mask(
                attention_mask, X, W_qkv, V_cache, active_blocks_table
            )
        # Bumped nkilib may select the QK-swap path for this shape and then
        # expect the transposed mask layout. Only the full pre-generated mask
        # (pos_ids is None) is transposed; the fused mask-gen mask is not.
        if pos_ids is None:
            attention_mask = _maybe_transpose_mask_for_qk_swap(
                attention_mask=attention_mask,
                V_cache=V_cache,
                active_blocks_table=active_blocks_table,
                k_scale=k_scale,
                fp8_packed=fp8_packed,
            )
        # The decode kernel shards over the process LNC degree. DCP and non-DCP
        # use separate wrappers so the common dcp_size=1 trace retains the
        # original kernel signature.
        lnc_degree = get_logical_nc_config()
        if dcp_size > 1:
            # Resolve to the enum eagerly (NKI forbids 'import'/'raise' in-trace).
            dcp_collective_mode_resolved = _resolve_dcp_collective_mode(
                dcp_collective_mode, dcp_size
            )
            dcp_group_ranks = tuple(dcp_group.ranks)
            wrapped = wrap_nki(_torch_compatible_attention_block_tkg_kernel_dcp)
            kernel_out = wrapped[lnc_degree](
                X,
                X_hidden_dim_actual=X_hidden_dim_actual,
                rmsnorm_X_enabled=rmsnorm_X_enabled,
                rmsnorm_X_eps=rmsnorm_X_eps,
                rmsnorm_X_gamma=rmsnorm_X_gamma,
                W_qkv=W_qkv,
                bias_qkv=bias_qkv,
                quantization_type_qkv=quantization_type_qkv,
                weight_dequant_scale_qkv=weight_dequant_scale_qkv,
                input_dequant_scale_qkv=input_dequant_scale_qkv,
                rmsnorm_QK_pre_rope_enabled=rmsnorm_QK_pre_rope_enabled,
                rmsnorm_QK_pre_rope_eps=rmsnorm_QK_pre_rope_eps,
                rmsnorm_QK_pre_rope_W_Q=rmsnorm_QK_pre_rope_W_Q,
                rmsnorm_QK_pre_rope_W_K=rmsnorm_QK_pre_rope_W_K,
                cos=cos,
                sin=sin,
                rope_contiguous_layout=rope_contiguous_layout,
                rmsnorm_QK_post_rope_enabled=rmsnorm_QK_post_rope_enabled,
                rmsnorm_QK_post_rope_eps=rmsnorm_QK_post_rope_eps,
                rmsnorm_QK_post_rope_W_Q=rmsnorm_QK_post_rope_W_Q,
                rmsnorm_QK_post_rope_W_K=rmsnorm_QK_post_rope_W_K,
                K_cache_transposed=K_cache_transposed,
                active_blocks_table=active_blocks_table,
                K_cache=K_cache,
                V_cache=V_cache,
                attention_mask=attention_mask,
                sink=sink,
                softmax_scale=softmax_scale,
                update_cache=update_cache,
                kv_cache_update_idx=kv_cache_update_idx,
                k_scale=k_scale,
                v_scale=v_scale,
                fp8_packed=fp8_packed,
                W_out=W_out,
                bias_out=bias_out,
                quantization_type_out=quantization_type_out,
                weight_dequant_scale_out=weight_dequant_scale_out,
                input_dequant_scale_out=input_dequant_scale_out,
                transposed_out=transposed_out,
                out_in_sb=out_in_sb,
                skip_attention=skip_attention,
                is_h_transposed_by_4=is_h_transposed_by_4,
                sbm_lower_bound=sbm_lower_bound,
                sbm_upper_bound=sbm_upper_bound,
                sbm_use_auto_alloc=sbm_use_auto_alloc,
                sbm_default_stack_alloc=sbm_default_stack_alloc,
                dcp_size=dcp_size,
                dcp_group_ranks=dcp_group_ranks,
                dcp_collective_mode_resolved=dcp_collective_mode_resolved,
                pos_ids=pos_ids,
                swa_start_pos_ids=swa_start_pos_ids,
            )
        else:
            wrapped = wrap_nki(_torch_compatible_attention_block_tkg_kernel)
            kernel_out = wrapped[lnc_degree](
                X,
                X_hidden_dim_actual=X_hidden_dim_actual,
                rmsnorm_X_enabled=rmsnorm_X_enabled,
                rmsnorm_X_eps=rmsnorm_X_eps,
                rmsnorm_X_gamma=rmsnorm_X_gamma,
                W_qkv=W_qkv,
                bias_qkv=bias_qkv,
                quantization_type_qkv=quantization_type_qkv,
                weight_dequant_scale_qkv=weight_dequant_scale_qkv,
                input_dequant_scale_qkv=input_dequant_scale_qkv,
                rmsnorm_QK_pre_rope_enabled=rmsnorm_QK_pre_rope_enabled,
                rmsnorm_QK_pre_rope_eps=rmsnorm_QK_pre_rope_eps,
                rmsnorm_QK_pre_rope_W_Q=rmsnorm_QK_pre_rope_W_Q,
                rmsnorm_QK_pre_rope_W_K=rmsnorm_QK_pre_rope_W_K,
                cos=cos,
                sin=sin,
                rope_contiguous_layout=rope_contiguous_layout,
                rmsnorm_QK_post_rope_enabled=rmsnorm_QK_post_rope_enabled,
                rmsnorm_QK_post_rope_eps=rmsnorm_QK_post_rope_eps,
                rmsnorm_QK_post_rope_W_Q=rmsnorm_QK_post_rope_W_Q,
                rmsnorm_QK_post_rope_W_K=rmsnorm_QK_post_rope_W_K,
                K_cache_transposed=K_cache_transposed,
                active_blocks_table=active_blocks_table,
                K_cache=K_cache,
                V_cache=V_cache,
                attention_mask=attention_mask,
                sink=sink,
                softmax_scale=softmax_scale,
                update_cache=update_cache,
                kv_cache_update_idx=kv_cache_update_idx,
                k_scale=k_scale,
                v_scale=v_scale,
                fp8_packed=fp8_packed,
                W_out=W_out,
                bias_out=bias_out,
                quantization_type_out=quantization_type_out,
                weight_dequant_scale_out=weight_dequant_scale_out,
                input_dequant_scale_out=input_dequant_scale_out,
                transposed_out=transposed_out,
                out_in_sb=out_in_sb,
                skip_attention=skip_attention,
                is_h_transposed_by_4=is_h_transposed_by_4,
                sbm_lower_bound=sbm_lower_bound,
                sbm_upper_bound=sbm_upper_bound,
                sbm_use_auto_alloc=sbm_use_auto_alloc,
                sbm_default_stack_alloc=sbm_default_stack_alloc,
                KVDP=attention_dp,
                pos_ids=pos_ids,
                swa_start_pos_ids=swa_start_pos_ids,
            )
        # The NKI kernel always returns (output, K, V). When update_cache=True
        # the K/V outputs are the in-place-written caches (kept live so the FX
        # aliasing pass threads the write back to the caller's cache); they're
        # redundant to return, so expose only the attention output to match the
        # torch fallback's update_cache=True contract.
        if update_cache:
            # kernel_out[1]/[2] are the alias outputs = the updated K/V caches.
            return kernel_out if return_updated_kv else kernel_out[0]
        return kernel_out
    else:
        # Validate torch-unsupported features
        if quantization_type_qkv != QuantizationType.NONE:
            raise NotImplementedError(
                "Attention block torch fallback does not support QKV quantization"
            )
        if quantization_type_out != QuantizationType.NONE:
            raise NotImplementedError(
                "Attention block torch fallback does not support output quantization"
            )

        if out_in_sb:
            raise NotImplementedError(
                "Attention block torch fallback does not support out_in_sb=True"
            )
        if skip_attention:
            raise NotImplementedError(
                "Attention block torch fallback does not support skip_attention=True"
            )

        # Torch fallback consumes a full [S_ctx, B, q_heads, S_tkg] mask. When
        # pos_ids is provided (fused mask-gen path), build the equivalent full
        # mask via gen_attention_decode_mask so kernel and torch paths produce
        # identical numerics.
        if pos_ids is not None:
            from vllm_neuron.functional.attention.attention_decode_mask import (
                gen_attention_decode_mask,
            )

            attention_mask = gen_attention_decode_mask(
                pos_ids=pos_ids.reshape(1, B_x * S_tkg_x).to(torch.float32),
                bs=B_x,
                q_head=q_heads_local,
                s_active=S_tkg_x,
                s_prior=S_ctx_local,
                start_pos=swa_start_pos_ids.reshape(1, B_x * S_tkg_x).to(torch.float32)
                if swa_start_pos_ids is not None
                else None,
                block_len=block_len_local,
            )

        _torch_result = _torch_attention_decode_impl(
            X=X,
            X_hidden_dim_actual=X_hidden_dim_actual,
            rmsnorm_X_enabled=rmsnorm_X_enabled,
            rmsnorm_X_eps=rmsnorm_X_eps,
            rmsnorm_X_gamma=rmsnorm_X_gamma,
            W_qkv=W_qkv,
            bias_qkv=bias_qkv,
            rmsnorm_QK_pre_rope_enabled=rmsnorm_QK_pre_rope_enabled,
            rmsnorm_QK_pre_rope_eps=rmsnorm_QK_pre_rope_eps,
            rmsnorm_QK_pre_rope_W_Q=rmsnorm_QK_pre_rope_W_Q,
            rmsnorm_QK_pre_rope_W_K=rmsnorm_QK_pre_rope_W_K,
            cos=cos,
            sin=sin,
            rope_contiguous_layout=rope_contiguous_layout,
            rmsnorm_QK_post_rope_enabled=rmsnorm_QK_post_rope_enabled,
            rmsnorm_QK_post_rope_eps=rmsnorm_QK_post_rope_eps,
            rmsnorm_QK_post_rope_W_Q=rmsnorm_QK_post_rope_W_Q,
            rmsnorm_QK_post_rope_W_K=rmsnorm_QK_post_rope_W_K,
            active_blocks_table=active_blocks_table,
            K_cache=K_cache,
            V_cache=V_cache,
            attention_mask=attention_mask,
            sink=sink,
            softmax_scale=softmax_scale,
            update_cache=update_cache,
            kv_cache_update_idx=kv_cache_update_idx,
            k_scale=k_scale,
            v_scale=v_scale,
            fp8_packed=fp8_packed,
            W_out=W_out,
            bias_out=bias_out,
            attention_dp=attention_dp,
            attention_dp_group=attention_dp_group,
            attention_dp_rank=attention_dp_rank,
            attention_dp_rank_tensor=attention_dp_rank_tensor,
            kv_needs_a2a=kv_needs_a2a,
            dcp_size=dcp_size,
            dcp_group=dcp_group,
        )
        if update_cache and return_updated_kv:
            # Mirror the NKI contract. The torch path mutates K_cache/V_cache with
            # plain aten ops, which ARE visible on a temporary, so the tensors passed
            # in already hold the update; hand them back for a uniform API.
            return _torch_result, K_cache, V_cache
        return _torch_result
