# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL MXFP8 CPU-dequant weight-loader factories (NMI-260).

The LLM-Compressor MXFP8 text checkpoint stores each quantized text-decoder
``nn.Linear`` as a native FP8 (``e4m3``) ``.weight`` of HF shape ``[out, in]``
plus a companion ``.weight_scale`` of shape ``[out, in/32]`` in ``ue8m0`` (one
U8 exponent per 32 INPUT elements per output row). Per-block dequant is EXACT
because the scale is a power of two:

    bf16[o, i] = fp8[o, i].float() * 2^(scale_u8[o, i // 32] - 127)

The 32-element block runs along the HF INPUT dim (storage dim 1) for ALL seven
projections; the output dim (storage dim 0) is per-row (block size 1).

Each loader is dual-mode and detects FP8 by the slice count (``+1`` scale slice
per weight). The dequant runs on the loader thread (CPU), so the device only ever
sees bf16 and the model graph is identical to ``model_bf16``. Only the rank's
weight slice and the scale blocks covering it are read from disk -- same I/O
footprint as the bf16 path. The within-block column offset is handled generically,
so an arbitrary TP that splits a 32-block still dequants correctly.

PRE-FLIGHT (see ``e8m0_to_scale``): this assumes safetensors surfaces
``.weight_scale`` as ``torch.uint8``. The checkpoint header confirms every
``weight_scale`` is stored as ``U8`` (-> ``torch.uint8``), so ``2^(u8-127)`` is
correct. If a future checkpoint surfaces the scale as ``float8_e8m0fnu`` / any
float, ``s.to(float32)`` already yields the decoded power-of-two and the ``-127``
must be dropped.
"""

from __future__ import annotations

import math

import torch

from vllm_neuron.functional.mlp import _PMAX, _Q_WIDTH, _TILE_SIZE
from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader, pad_to_shape

# ``build_per_head_block_table`` is the shared GQA decode helper; it now lives in
# ``utils.decode_kv`` (used by both the bf16 and native-MX decode paths). Re-exported
# here so this module's existing consumers + the MX attention tests keep importing it
# from ``weight_loaders_mxfp8`` unchanged.
from vllm_neuron.model.qwen3_vl.utils.decode_kv import (  # noqa: F401
    build_per_head_block_table,
)

__all__ = [
    "e8m0_to_scale",
    "mx_dequant",
    "dequant_mxfp8_weight",
    "mxfp8_fused_qkv_loader",
    "mxfp8_sharded_transposed_loader",
    "MX_GROUP_SIZE",
    # Native 3D-unpacked-fp8 path (shared by CTE prefill and TKG decode mega-kernel).
    "pre_shuffle_h",
    "qkv_weight_pack_3d",
    "qkv_shard_offsets",
    "fused_qkv_weight_loader_3d",
    "fused_qkv_scale_loader_3d",
    "o_proj_weight_pack_mx",
    "o_proj_weight_pack_3d",
    "o_proj_weight_loader",
    "o_proj_scale_loader",
    "build_per_head_block_table",
    "build_mx_attention_mappings",
    # Native-MX MLP CTE path (gate/up + down, H_X4_INNERMOST swizzle).
    "mlp_gate_up_weight_loader",
    "mlp_gate_up_scale_loader",
    "mlp_down_weight_loader",
    "mlp_down_scale_loader",
    # Native-MX MLP TKG (decode) path — 3D uint32/u8 layout.
    "mlp_gate_up_weight_loader_tkg",
    "mlp_gate_up_scale_loader_tkg",
    "mlp_down_weight_loader_tkg",
    "mlp_down_scale_loader_tkg",
    "_n_i512_tiles_tkg",
    "build_mx_mlp_mappings",
]

MX_GROUP_SIZE = 32  # E8M0 scale block size, along the HF input dim
_MX_BLOCK = MX_GROUP_SIZE  # internal alias kept for the existing slice helpers
_FP8_DTYPE = torch.float8_e4m3fn
# _PMAX / _Q_WIDTH / _TILE_SIZE are the shared MX tile geometry, defined in
# vllm_neuron.functional.mlp and imported above (re-exported here so model_mxfp8 and
# the loader tests keep importing them from this module).
# E8M0 micro-scales per 512-tile along the contraction dim: one per MX_GROUP_SIZE (32)
# elements -> 512 / 32 = 16. (This is the leading "16" of the MX CTE scale layout, e.g.
# gen_mlp_mxfp_weights' [16, H/512, I] base.)
_SCALES_PER_TILE = _TILE_SIZE // MX_GROUP_SIZE  # 16


def x4_pack_fp8(w: torch.Tensor, *, contraction_axis: int) -> torch.Tensor:
    """Pack 4 consecutive fp8 bytes along ``contraction_axis`` into one uint32.

    Output shape has ``shape[contraction_axis] // 4``; output dtype is
    ``torch.uint32``. The pack is a bit-level reinterpret (no float math), so
    the inverse round-trip with ``unpack_float8_e4m3fn_x4`` is bit-exact.
    Copied from the former shared STATIC_MX weight-pack module (removed) so
    this loader is self-contained; the qwen3 (dense) copy must stay in sync.
    """
    assert w.dtype == torch.float8_e4m3fn, (
        f"x4_pack_fp8 expects torch.float8_e4m3fn, got {w.dtype}"
    )
    contraction_axis = contraction_axis % w.ndim
    n = w.shape[contraction_axis]
    assert n % 4 == 0, (
        f"x4_pack_fp8 requires shape[{contraction_axis}]={n} divisible by 4"
    )
    # Move contraction axis to the last position, make contiguous, then view as
    # uint8 -> uint32. The little-endian view is what nl.float8_e4m3fn_x4
    # expects (byte 0 -> low 8 bits, byte 3 -> high 8 bits); see
    # nkilib/core/utils/mx_torch_common.py::unpack_float8_e4m3fn_x4 for the
    # canonical reverse transform.
    moved = w.movedim(contraction_axis, -1).contiguous()
    bytes_view = moved.view(torch.uint8)
    new_last = n // 4
    packed = bytes_view.reshape(*moved.shape[:-1], new_last, 4)
    # int64 intermediate: older torch lacks a uint32 lshift kernel; fits in 32 bits.
    packed_i64 = (
        packed[..., 0].to(torch.int64)
        | (packed[..., 1].to(torch.int64) << 8)
        | (packed[..., 2].to(torch.int64) << 16)
        | (packed[..., 3].to(torch.int64) << 24)
    )
    return packed_i64.to(torch.uint32).movedim(-1, contraction_axis).contiguous()


def e8m0_to_scale(scale_u8: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 (bias 127) -> fp32 power-of-two multiplier.

    The canonical MXFP8 scale decode, shared by the production load path
    (``_read_dequant_slice``), the full-tensor ``mx_dequant`` / ``dequant_mxfp8_weight``,
    and the test golden + reference builders. One definition so the decode cannot
    drift between the device-load path and any CPU reference.

    Cast to float32 BEFORE subtracting 127 (subtracting from a ``uint8``
    underflows/wraps). Correct iff ``scale_u8`` is ``torch.uint8``.
    """
    assert scale_u8.dtype == torch.uint8, (
        f"E8M0 scale expected torch.uint8, got {scale_u8.dtype}; if it is a "
        "float8/float dtype use s.to(float32) directly with NO -127."
    )
    return torch.exp2(scale_u8.to(torch.float32) - 127.0)


def mx_dequant(
    data_fp8: torch.Tensor,
    scale_u8: torch.Tensor,
    *,
    group_size: int = MX_GROUP_SIZE,
    dim: int = -1,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize a whole (unsliced) MX-quantized tensor: ``data_fp8`` (float8)
    with one E8M0 ``scale_u8`` exponent per ``group_size`` elements along ``dim``.

    Generic over leading dims, scale axis, and output dtype, so it is the single
    decode for BOTH MX weights (``dim=1``, ``out_dtype=bfloat16`` -- see
    ``dequant_mxfp8_weight``) and MX activations / CPU references
    (``dim=-1``, ``out_dtype=float32``, the default). The block scale is decoded
    via ``e8m0_to_scale`` and broadcast with ``repeat_interleave``; the caller
    guarantees the scaled axis is a multiple of ``group_size`` (no within-block
    trim -- the sharded loaders use ``_read_dequant_slice`` for that).
    """
    scale = e8m0_to_scale(scale_u8).repeat_interleave(group_size, dim=dim)
    return (data_fp8.to(torch.float32) * scale).to(out_dtype)


def dequant_mxfp8_weight(
    weight_fp8: torch.Tensor, scale_u8: torch.Tensor
) -> torch.Tensor:
    """Full-tensor MXFP8 weight dequant: HF ``weight`` ``[out, in]`` (float8_e4m3fn)
    + ``scale`` ``[out, in/32]`` (uint8 E8M0) -> bf16 ``[out, in]``.

    The whole-tensor (no shard, no transpose) entry point used by the test golden
    builder; a thin ``mx_dequant`` call with the weight axis convention (blocks
    along the HF input dim = storage dim 1, bf16 output). Requires
    ``in % MX_GROUP_SIZE == 0`` -- a valid MXFP8 checkpoint always stores ``in``
    as a multiple of 32, so a non-multiple signals a malformed/mismatched scale
    tensor.
    """
    assert weight_fp8.shape[1] % MX_GROUP_SIZE == 0, (
        f"input dim {weight_fp8.shape[1]} not a multiple of {MX_GROUP_SIZE}"
    )
    return mx_dequant(
        weight_fp8, scale_u8, group_size=MX_GROUP_SIZE, dim=1, out_dtype=torch.bfloat16
    )


def _scale_col_window(k_start: int, k_size: int) -> tuple[int, int, int]:
    """Block range + within-block offset covering ``[k_start:k_start+k_size]`` cols.

    Returns ``(block_start, block_end, within_block_offset)``. Use
    ``scale[:, block_start:block_end]`` to pick the 32-blocks touching the
    slice; ``within_block_offset`` trims the expanded scale back to the slice.
    """
    elem_end = k_start + k_size
    block_start = k_start // _MX_BLOCK
    block_end = (elem_end + _MX_BLOCK - 1) // _MX_BLOCK
    return block_start, block_end, k_start % _MX_BLOCK


def _read_dequant_slice(
    w_slice,
    s_slice,
    n_start: int,
    n_size: int,
    k_start: int,
    k_size: int,
) -> torch.Tensor:
    """Read ``[n_start:n_start+n_size, k_start:k_start+k_size]`` of an FP8 HF
    weight + its ``weight_scale``, returning the dequantized bf16 slice.

    Only the needed weight rows/cols and the scale blocks (along the input dim)
    touching them are pulled from disk. The output dim (rows) is per-row scaled,
    so scale rows map 1:1 to weight rows.
    """
    kb0, kb1, k_off = _scale_col_window(k_start, k_size)
    w = w_slice[n_start : n_start + n_size, k_start : k_start + k_size]  # fp8
    s = s_slice[n_start : n_start + n_size, kb0:kb1]  # u8, covering blocks
    scale = e8m0_to_scale(s).repeat_interleave(_MX_BLOCK, dim=1)
    scale = scale[:, k_off : k_off + k_size]
    return (w.to(torch.float32) * scale).to(torch.bfloat16)


def mxfp8_sharded_transposed_loader(
    shard_dim: int, shard_size: int, num_shards: int
) -> SafetensorsWeightLoader:
    """FP8-dequant equivalent of ``sharding_weight_loader(is_storage_transposed=True)``.

    Param is ``[in, out]``; HF weight ``[out, in]``; scale ``[out, in/32]``.
      ``shard_dim=1`` -> shard the param output -> slice HF dim 0 (OUT rows;
                         != block axis -> all scale cols).
      ``shard_dim=0`` -> shard the param input  -> slice HF dim 1 (IN cols;
                         == block axis -> /32 scale-col window).

    Dual-mode: ``len(slices)==1`` is the bf16 fast path (matches the standard
    loader); ``len(slices)==2`` is ``[weight, weight_scale]`` and dequants.
    """
    assert shard_dim in (0, 1), f"shard_dim must be 0 or 1, got {shard_dim}"
    storage_shard_dim = 1 - shard_dim  # transposed storage swaps 0/1

    def transform(slices, rank):
        local_rank = rank % num_shards
        start = local_rank * shard_size

        if len(slices) == 1:
            w_slice = slices[0]
            sl = [slice(None), slice(None)]
            sl[storage_shard_dim] = slice(start, start + shard_size)
            return w_slice[tuple(sl)].T

        assert len(slices) == 2, (
            f"mxfp8_sharded_transposed_loader expected 1 (bf16) or 2 (fp8 + "
            f"scale) slices, got {len(slices)}"
        )
        w_slice, s_slice = slices
        N, K = w_slice.get_shape()
        if storage_shard_dim == 0:
            # OUT shard: weight+scale rows [start:start+shard_size], all cols.
            return _read_dequant_slice(w_slice, s_slice, start, shard_size, 0, K).T
        # IN shard == block axis: all rows, cols [start:start+shard_size].
        return _read_dequant_slice(w_slice, s_slice, 0, N, start, shard_size).T

    return SafetensorsWeightLoader(transform=transform)


# ===========================================================================
# Native 3D-unpacked-fp8 loaders for the fused attention-decode (TKG) kernel
# ===========================================================================
# The QKV-TKG and O-Proj-TKG MX paths inside the fused decode mega-kernel consume
# weights as **3D unpacked fp8** ``[dim//4, other, 4]`` ``float8_e4m3fn`` (NOT the
# 2D ``uint32`` x4-packed container the CTE O-Proj kernel uses), with the per-group
# MX microscales kept as ``uint8`` ``[*//32, *]``. These build that layout directly.


def pre_shuffle_h(x: torch.Tensor) -> torch.Tensor:
    """Pre-shuffle the H dimension for ``is_h_transposed_by_4=True``.

    Matches the nkilib QKV TKG MX contract (qkv_tkg_mx_impl.py:146-147):
        [B, S, H//512, 128_H, 4_H] -> [B, S, 4_H, H//512, 128_H] -> flatten [B, S, H]
    i.e. the 4_H axis (innermost) is moved to the FRONT of the H sub-block.
    """
    shape = x.shape
    H = shape[-1]
    lead = shape[:-1]
    reshaped = x.reshape(*lead, H // (_PMAX * _Q_WIDTH), _PMAX, _Q_WIDTH)
    permuted = reshaped.permute(*range(len(lead)), -1, -3, -2).contiguous()
    return permuted.reshape(shape)


def qkv_weight_pack_3d(w_HI: torch.Tensor) -> torch.Tensor:
    """Pack fp8 weight [H, I] -> [H//4, I, 4] float8_e4m3fn (MX_CONTIGUOUS, unpacked).

    This is the 2D-uint32 packing's intermediate layout without the final
    flatten + x4 byte-pack: the TKG MX kernel consumes the 3D fp8 view directly.
    """
    assert w_HI.dtype == _FP8_DTYPE
    H, I = w_HI.shape
    assert H % 4 == 0
    return w_HI.reshape(H // 4, 4, I).transpose(1, 2).contiguous()


def qkv_shard_offsets(
    q_total: int,
    kv_total: int,
    num_shards: int,
    num_kv_replicas: int,
    rank: int,
) -> tuple[slice, slice]:
    """TP-shard offsets for fused QKV, shared by all loaders and the CPU reference.

    Q is sharded one head-group per rank; K/V use GQA replication (num_kv_replicas
    ranks share a KV group). Returns ``(q_slice, kv_slice)`` row ranges to apply to
    the Q and K/V tensors respectively. Single source of truth for the GQA sharding
    so the loaders and the reference cannot diverge.
    """
    assert num_shards % num_kv_replicas == 0, (
        f"num_shards ({num_shards}) must be divisible by num_kv_replicas "
        f"({num_kv_replicas})"
    )
    assert q_total % num_shards == 0, (
        f"q_total ({q_total}) must be divisible by num_shards ({num_shards})"
    )
    kv_groups = num_shards // num_kv_replicas
    assert kv_total % kv_groups == 0, (
        f"kv_total ({kv_total}) must be divisible by num KV groups ({kv_groups} = "
        f"num_shards // num_kv_replicas)"
    )
    # Index by ``rank % num_shards`` so a global rank >= num_shards (pipeline
    # parallelism, or a TP group smaller than the world) indexes the checkpoint
    # correctly instead of running off the output dim.
    local_rank = rank % num_shards
    q_per_rank = q_total // num_shards
    q_start = local_rank * q_per_rank
    kv_per_group = kv_total // kv_groups
    kv_start = (local_rank // num_kv_replicas) * kv_per_group
    return (
        slice(q_start, q_start + q_per_rank),
        slice(kv_start, kv_start + kv_per_group),
    )


def fused_qkv_weight_loader_3d(
    num_shards: int,
    num_kv_replicas: int,
) -> SafetensorsWeightLoader:
    """Load Q/K/V fp8 weights -> fuse -> TP shard -> transpose -> 3D fp8 pack.

    Returns [H//4, fused_I, 4] float8_e4m3fn. Per-rank Q/KV row counts are derived
    from the slice shapes inside the transform (via qkv_shard_offsets), so no
    explicit q_size/kv_size are needed.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        q_slice, k_slice, v_slice = slices
        q_sl, kv_sl = qkv_shard_offsets(
            q_slice.get_shape()[0],
            k_slice.get_shape()[0],
            num_shards,
            num_kv_replicas,
            rank,
        )
        q, k, v = q_slice[q_sl, :], k_slice[kv_sl, :], v_slice[kv_sl, :]

        fused_OF = torch.cat([q, k, v], dim=0)  # [fused_I, H] fp8
        fused_HI = (
            fused_OF.contiguous().view(torch.uint8).t().contiguous().view(_FP8_DTYPE)
        )  # [H, fused_I] fp8
        return qkv_weight_pack_3d(fused_HI)

    return SafetensorsWeightLoader(transform=transform)


def fused_qkv_scale_loader_3d(
    num_shards: int,
    num_kv_replicas: int,
) -> SafetensorsWeightLoader:
    """Load Q/K/V MX weight scales -> fuse -> TP shard -> transpose to [H//32, fused_I]."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        q_slice, k_slice, v_slice = slices
        q_sl, kv_sl = qkv_shard_offsets(
            q_slice.get_shape()[0],
            k_slice.get_shape()[0],
            num_shards,
            num_kv_replicas,
            rank,
        )
        q, k, v = q_slice[q_sl, :], k_slice[kv_sl, :], v_slice[kv_sl, :]

        fused = torch.cat([q, k, v], dim=0)  # [fused_I, H//32] uint8
        return fused.t().contiguous()  # [H//32, fused_I]

    return SafetensorsWeightLoader(transform=transform)


def o_proj_weight_pack_mx(w_NDH: torch.Tensor) -> torch.Tensor:
    """Pack fp8 weight [N*D, H] -> [N*D//4, H] uint32 for the CTE MX kernel."""
    assert w_NDH.dtype == _FP8_DTYPE
    ND, H = w_NDH.shape
    assert ND % 4 == 0
    rearranged = w_NDH.reshape(ND // 4, 4, H).transpose(1, 2).reshape(ND // 4, H * 4)
    return x4_pack_fp8(rearranged, contraction_axis=-1)


def o_proj_weight_pack_3d(w_NDH: torch.Tensor) -> torch.Tensor:
    """Pack fp8 weight [N*D, H] -> [N*D//4, H, 4] float8_e4m3fn (3D unpacked).

    The TKG O-Proj MX kernel consumes this 3D fp8 view directly -- the same
    rearrange as ``o_proj_weight_pack_mx`` but WITHOUT the final flatten + x4
    byte-pack into the uint32 container the CTE kernel uses.
    """
    assert w_NDH.dtype == _FP8_DTYPE
    ND, H = w_NDH.shape
    assert ND % 4 == 0
    return w_NDH.reshape(ND // 4, 4, H).transpose(1, 2).contiguous()


def o_proj_scale_loader(
    shard_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Load o_proj MX weight scales for both CTE and TKG kernels.

    Checkpoint: o_proj.weight_scale [H, N*D//32] uint8.
    Both CTE and TKG kernels expect [N*D_per_rank//32, H] uint8.
    TP shard on dim 1 (N*D//32 axis), then transpose.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1
        scale = slices[0]
        scale_per_rank = shard_size // MX_GROUP_SIZE
        # Symmetric to o_proj_weight_loader's nd_total == shard_size*num_shards
        # check: fail loud if the checkpoint scale's N*D//32 axis isn't exactly
        # (shard_size//32)*num_shards. Without this a TP that doesn't divide the
        # scale axis evenly (or padded MX scale rows) would silently return a
        # short/empty slice on the last rank -> wrong-shaped scale vs the weight.
        nd_scale_total = scale.get_shape()[1]
        assert nd_scale_total == scale_per_rank * num_shards, (
            f"o_proj weight_scale N*D//32 axis ({nd_scale_total}) must equal "
            f"(shard_size//{MX_GROUP_SIZE})*num_shards "
            f"({scale_per_rank}*{num_shards}={scale_per_rank * num_shards})."
        )
        start = (rank % num_shards) * scale_per_rank  # wrap to the attention-TP group
        shard_scale = scale[:, start : start + scale_per_rank]  # [H, shard//32]
        return shard_scale.t().contiguous()  # [shard//32, H]

    return SafetensorsWeightLoader(transform=transform)


def o_proj_weight_loader(
    shard_size: int,
    num_shards: int,
    pack_3d: bool = False,
) -> SafetensorsWeightLoader:
    """Load o_proj FP8 weight: TP shard, transpose, then pack for the target kernel.

    Checkpoint: [H, N*D] fp8. The shard + transpose to [N*D_per_rank, H] is shared;
    only the final pack differs by kernel:
      - ``pack_3d=False`` (default, CTE): [N*D_per_rank//4, H] uint32 (x4-packed).
      - ``pack_3d=True`` (TKG): [N*D_per_rank//4, H, 4] float8_e4m3fn (3D unpacked).
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1
        weight = slices[0]
        full_shape = weight.get_shape()
        nd_total = full_shape[1]
        assert nd_total == shard_size * num_shards
        start = (rank % num_shards) * shard_size  # wrap to the attention-TP group
        sliced_HnD = weight[:, start : start + shard_size]
        bytes_view = sliced_HnD.contiguous().view(torch.uint8)
        bytes_NDH = bytes_view.transpose(0, 1).contiguous()
        w_NDH = bytes_NDH.view(_FP8_DTYPE)  # [N*D_per_rank, H] fp8
        return o_proj_weight_pack_3d(w_NDH) if pack_3d else o_proj_weight_pack_mx(w_NDH)

    return SafetensorsWeightLoader(transform=transform)


def mxfp8_fused_qkv_loader(
    q_size: int, kv_size: int, num_shards: int, num_kv_replicas: int
) -> SafetensorsWeightLoader:
    """FP8-dequant equivalent of ``fused_qkv_weight_loader(shard_dim=1,
    is_storage_transposed=True)``: ``storage_shard_dim=0``, so the block axis
    (HF dim 1 = input) is never the shard axis -> each scale follows the same
    OUT rows, all cols, no /32 interaction.

    Dequant is applied per q/k/v BEFORE the cat (each has its own scale); the three
    transposed ``[in, size]`` tensors are concatenated along dim 1 (the param output
    dim) in q,k,v order, matching ``qkv_split_indices`` in ``model_bf16.forward``.

    Dual-mode: ``len(slices)==3`` is the bf16 fast path ([qW, kW, vW]);
    ``len(slices)==6`` is the interleaved [qW, qS, kW, kS, vW, vS] FP8 path.
    """

    def transform(slices, rank):
        local_rank = rank % num_shards
        q_rank = local_rank
        kv_rank = local_rank // num_kv_replicas

        if len(slices) == 3:
            out = []
            for w_slice, shard_size, shard_rank in zip(
                slices, [q_size, kv_size, kv_size], [q_rank, kv_rank, kv_rank]
            ):
                s0 = shard_rank * shard_size
                out.append(w_slice[s0 : s0 + shard_size, :].T)
            return torch.cat(out, dim=1)

        assert len(slices) == 6, (
            f"mxfp8_fused_qkv_loader expected 3 (bf16) or 6 (fp8: "
            f"[qW,qS,kW,kS,vW,vS]) slices, got {len(slices)}"
        )
        out = []
        for (w_slice, s_slice), shard_size, shard_rank in (
            ((slices[0], slices[1]), q_size, q_rank),
            ((slices[2], slices[3]), kv_size, kv_rank),
            ((slices[4], slices[5]), kv_size, kv_rank),
        ):
            s0 = shard_rank * shard_size
            _, K = w_slice.get_shape()
            out.append(_read_dequant_slice(w_slice, s_slice, s0, shard_size, 0, K).T)
        return torch.cat(out, dim=1)  # concat along param output dim (q|k|v)

    return SafetensorsWeightLoader(transform=transform)


def build_mx_attention_mappings(hf_prefix: str, layer_prefix: str = "") -> dict:
    """Param -> checkpoint-key mappings for the native-MX attention weights.

    Single source of truth for the native-MX QKV + O-Proj weight/scale mappings,
    shared by the served model's ``load_weights`` and the CTE three-way module test.
    The module's native loaders (``Qwen3VLTextAttentionMX._setup_weight_loaders``)
    do the pack/shard; this only names the source tensors. Covers ONLY the
    MX-quantized projections — the plain-bf16 norms (q/k/input_layernorm) are owned
    by each caller (the served loader sets them in its shared norm block; the test
    adds its own), so they are deliberately not emitted here.

    Args:
        hf_prefix: the HF self_attn prefix, e.g.
            ``model.language_model.layers.{i}.self_attn``.
        layer_prefix: prefixes the destination param names (empty when loading
            into a bare attention wrapper; e.g. ``language_model.layers.{i}`` for
            the served model).

    Returns:
        A ``{dest_param_name: checkpoint_key(s)}`` dict. Both o-proj buffers map to
        the SAME checkpoint key; their loaders pack the kernel-specific layout
        (3D-fp8 for decode, 2D-uint32 for prefill).
    """
    assert ".self_attn" in hf_prefix, (
        f"build_mx_attention_mappings expects an HF self_attn prefix ending in "
        f"'.self_attn'; got {hf_prefix!r}."
    )
    p = f"{layer_prefix}." if layer_prefix else ""
    return {
        f"{p}self_attn.qkv_proj_weight": [
            f"{hf_prefix}.q_proj.weight",
            f"{hf_prefix}.k_proj.weight",
            f"{hf_prefix}.v_proj.weight",
        ],
        f"{p}self_attn.qkv_weight_scale": [
            f"{hf_prefix}.q_proj.weight_scale",
            f"{hf_prefix}.k_proj.weight_scale",
            f"{hf_prefix}.v_proj.weight_scale",
        ],
        # Both o-proj buffers load from the SAME checkpoint key; their loaders pack
        # the kernel-specific layout (3D-fp8 for decode, 2D-uint32 for prefill).
        # TODO(NMI-191): unify these two buffers once the CTE/TKG kernels accept a
        # single canonical O-Proj layout.
        f"{p}self_attn.o_proj_weight_decode": [f"{hf_prefix}.o_proj.weight"],
        f"{p}self_attn.o_proj_weight_prefill": [f"{hf_prefix}.o_proj.weight"],
        f"{p}self_attn.o_proj_weight_scale": [f"{hf_prefix}.o_proj.weight_scale"],
    }


def _n_i512_tiles(intermediate_size_per_rank: int) -> int:
    """Number of 512-tiles along the per-rank I dim, rounded UP to an EVEN count.

    The MLP CTE MX kernel shards the intermediate dim across the 2 LNC cores
    whenever ``intermediate_size > 4096`` (mlp_cte_sharding.calculate_sharding sets
    ``shard_on_inter=True`` unconditionally there), splitting ``intermediate_size //
    2``. ``intermediate_size`` is derived from the weight's tile count * 512, so that
    tile count must be EVEN for the half-split to land on a 512 boundary; an odd
    count makes the second core's offset run off the end of the weight (KLIR
    out-of-bound). Equivalent to requiring the padded I to be a multiple of 1024.
    """
    return 2 * math.ceil(intermediate_size_per_rank / (2 * _TILE_SIZE))


def mlp_gate_up_weight_loader(
    intermediate_size_per_rank: int,
    hidden_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Load an MLP gate/up FP8 weight -> TP shard (output/I dim) -> H_X4_INNERMOST pack.

    HF weight is ``[I, H]`` fp8 (output-major). Shard the I/output dim, transpose to
    ``[H, I_per_rank]``, pad I to an even number of 512-tiles, then reshape/permute to
    the 6D ``[128, H/512, n_i512, 4, 128, 4]`` the kernel reads. The reshape+permute
    pack body matches llama3's ``mlp_gate_up_weight_loader_mxfp8_cte`` (the
    H_X4_INNERMOST swizzle is shared with STATIC_MX); ported here so qwen3_vl does not
    import from llama3. NOTE: the I-tile count differs from llama3 -- this uses
    ``_n_i512_tiles`` (forced EVEN) where llama3 uses ``math.ceil(I/512)`` (any count),
    because the MX CTE kernel half-splits I across the 2 LNC cores (an odd tile count
    runs the second core off the end of the weight). Do NOT revert to plain ceil.
    """
    assert hidden_size % _TILE_SIZE == 0, (
        f"hidden_size ({hidden_size}) must be a multiple of {_TILE_SIZE} for the "
        f"H_X4_INNERMOST tiling."
    )
    n_h512 = hidden_size // _TILE_SIZE
    n_i512 = _n_i512_tiles(intermediate_size_per_rank)
    padded_i = n_i512 * _TILE_SIZE

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1
        w = slices[0]  # HF [I, H] fp8
        # Fail loud if the checkpoint I axis isn't exactly i_pr*num_shards (a TP that
        # doesn't divide I evenly would otherwise let the pad step silently zero-pad
        # a short/empty high-rank slice). Mirrors o_proj_weight_loader's guard.
        i_total = w.get_shape()[0]
        assert i_total == intermediate_size_per_rank * num_shards, (
            f"gate/up weight I axis ({i_total}) must equal i_per_rank*num_shards "
            f"({intermediate_size_per_rank}*{num_shards})."
        )
        start = (rank % num_shards) * intermediate_size_per_rank
        # Slice the I/output rows for this rank, transpose to [H, I_per_rank].
        w_HI = w[start : start + intermediate_size_per_rank, :].T  # [H, I_pr] fp8
        if w_HI.shape[1] < padded_i:  # pad I up to the even-tile count (fp8 F.pad OK)
            w_HI = pad_to_shape(w_HI, (hidden_size, padded_i))
        # H_X4_INNERMOST: [H, I] -> [H/512, 128_H, 4_H, I/512, 128_I, 4_I]
        # -> permute(1,0,3,5,4,2) -> [128_H, H/512, I/512, 4_I, 128_I, 4_H].
        w6 = w_HI.reshape(n_h512, _PMAX, _Q_WIDTH, n_i512, _PMAX, _Q_WIDTH)
        return w6.permute(1, 0, 3, 5, 4, 2).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def mlp_gate_up_scale_loader(
    intermediate_size_per_rank: int,
    hidden_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Load an MLP gate/up MX weight scale -> TP shard -> swizzle to match the 6D weight.

    HF scale is ``[I, H/32]`` uint8 (one E8M0 per 32 H-elements per output row). Shard
    the I/output dim (same axis the weight shards), transpose to logical ``[H/32, I_pr]``,
    then build the kernel's ``[16, H/512, n_i512, 4, 128]`` layout. The "16" is the
    per-512-tile H-scale count (= 128/4: 8 partition-rows x 4 x4-lanes span 32 H), so the
    logical scale row maps as ``h_block = h512 * 16 + p``. The trailing I split (128, 4)
    is transposed to physical (4, 128) to match the weight's (4_I, 128_I) ordering
    (mirrors the integration test's reshape(16, H/512, I/512, 128, 4).transpose(...,4,3)).
    """
    assert hidden_size % _TILE_SIZE == 0, (
        f"hidden_size ({hidden_size}) must be a multiple of {_TILE_SIZE} for the "
        f"H_X4_INNERMOST scale tiling."
    )
    n_h512 = hidden_size // _TILE_SIZE
    n_i512 = _n_i512_tiles(intermediate_size_per_rank)
    padded_i = n_i512 * _TILE_SIZE

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1
        s = slices[0]  # HF [I, H/32] uint8
        i_total = s.get_shape()[0]
        assert i_total == intermediate_size_per_rank * num_shards, (
            f"gate/up scale I axis ({i_total}) must equal i_per_rank*num_shards "
            f"({intermediate_size_per_rank}*{num_shards})."
        )
        start = (rank % num_shards) * intermediate_size_per_rank
        # [I_pr, H/32] -> logical [H/32, I_pr].
        s_log = s[start : start + intermediate_size_per_rank, :].T.contiguous()
        # [H/32, I_pr] -> base [16, H/512, I_pr]; H/32 row = h512 * 16 + p.
        base = s_log.reshape(n_h512, _SCALES_PER_TILE, -1).permute(1, 0, 2).contiguous()
        # Pad the I axis to padded_i (n_i512 even-tile count * 512).
        if base.shape[2] < padded_i:
            base = torch.nn.functional.pad(base, (0, padded_i - base.shape[2]))
        # Split I into 512-tiles, transpose (128, 4) -> (4, 128) to match the weight.
        sc = base.reshape(_SCALES_PER_TILE, n_h512, n_i512, _PMAX, _Q_WIDTH)
        return sc.permute(0, 1, 2, 4, 3).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def mlp_down_weight_loader(
    intermediate_size_per_rank: int,
    hidden_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Load an MLP down FP8 weight -> TP shard (input/I dim) -> down swizzle.

    HF weight is ``[H, I]`` fp8. Shard the I/input (contraction) dim, transpose to
    ``[I_per_rank, H]``, pad I to an even number of 512-tiles, then reshape/permute to
    the 4D ``[128, n_i512, H, 4]`` the kernel reads. The reshape+permute pack body
    matches llama3's ``mlp_down_weight_loader_mxfp8_cte``, but the I-tile count differs
    (``_n_i512_tiles`` forced EVEN vs llama3's ``math.ceil(I/512)``) for the same LNC-2
    half-split reason as the gate/up loader. Do NOT revert to plain ceil.
    """
    n_i512 = _n_i512_tiles(intermediate_size_per_rank)
    padded_i = n_i512 * _TILE_SIZE

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1
        w = slices[0]  # HF [H, I] fp8
        i_total = w.get_shape()[1]
        assert i_total == intermediate_size_per_rank * num_shards, (
            f"down weight I axis ({i_total}) must equal i_per_rank*num_shards "
            f"({intermediate_size_per_rank}*{num_shards})."
        )
        start = (rank % num_shards) * intermediate_size_per_rank
        # Slice the I/input cols for this rank, transpose to [I_per_rank, H].
        w_IH = w[:, start : start + intermediate_size_per_rank].T  # [I_pr, H] fp8
        if w_IH.shape[0] < padded_i:  # pad I up to the even-tile count (fp8 F.pad OK)
            w_IH = pad_to_shape(w_IH, (padded_i, hidden_size))
        # [I, H] -> [I/512, 128_I, 4_I, H] -> permute(1,0,3,2) -> [128_I, I/512, H, 4_I].
        w4 = w_IH.reshape(n_i512, _PMAX, _Q_WIDTH, hidden_size)
        return w4.permute(1, 0, 3, 2).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def mlp_down_scale_loader(
    intermediate_size_per_rank: int,
    hidden_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Load an MLP down MX weight scale -> TP shard -> ``[16, n_i512, H]``.

    HF scale is ``[H, I/32]`` uint8 (one E8M0 per 32 I-elements per H row -- the 32-block
    runs along the HF input dim = I for down). Shard the I/32 axis (same I the weight
    shards), transpose to logical ``[I_pr/32, H]``, pad the I/32 rows up to ``n_i512*16``
    (16 = 512/32; required so the reshape into 512-tiles is exact, since I_pr/32 need not
    be a multiple of 16), then reshape to ``[n_i512, 16, H]`` and permute to
    ``[16, n_i512, H]`` (matches the kernel's down-scale layout, inverted by
    ``_undo_mx_down_sc_reshape``). ``n_i512`` is the EVEN tile count from
    ``_n_i512_tiles`` (see the weight loaders), not ``ceil(I/512)``.
    """
    assert intermediate_size_per_rank % MX_GROUP_SIZE == 0, (
        f"i_per_rank ({intermediate_size_per_rank}) must be a multiple of "
        f"{MX_GROUP_SIZE} so the per-rank E8M0 scale row count is exact (no floor)."
    )
    n_i512 = _n_i512_tiles(intermediate_size_per_rank)
    scale_per_rank = intermediate_size_per_rank // MX_GROUP_SIZE  # I_pr/32
    padded_scale_rows = n_i512 * _SCALES_PER_TILE  # n_i512 * 16

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1
        s = slices[0]  # HF [H, I/32] uint8
        scale_total = s.get_shape()[1]
        assert scale_total == scale_per_rank * num_shards, (
            f"down scale I/32 axis ({scale_total}) must equal (i_per_rank//"
            f"{MX_GROUP_SIZE})*num_shards ({scale_per_rank}*{num_shards})."
        )
        start = (rank % num_shards) * scale_per_rank
        # [H, I_pr/32] -> logical [I_pr/32, H].
        s_log = s[:, start : start + scale_per_rank].T.contiguous()
        # Pad the I/32 rows to n_i512*16 so reshape into 512-tiles is exact.
        if s_log.shape[0] < padded_scale_rows:
            s_log = torch.nn.functional.pad(
                s_log, (0, 0, 0, padded_scale_rows - s_log.shape[0])
            )
        # [16 * n_i512, H] -> [n_i512, 16, H] -> [16, n_i512, H].
        sc = s_log.reshape(n_i512, _SCALES_PER_TILE, hidden_size)
        return sc.permute(1, 0, 2).contiguous()

    return SafetensorsWeightLoader(transform=transform)


# ---------------------------------------------------------------------------
# Native-MX MLP TKG (decode) loaders — 3D uint32/u8, distinct from the CTE 6D/4D.
# ---------------------------------------------------------------------------
# The TKG MX kernel (mlp_tkg/gate_up_projection_mx_shard_H.py, down_projection_mx_
# shard_H.py) consumes a DIFFERENT weight layout than the CTE kernel:
#   * gate/up: 3D [128, H/512, I] uint32 (x4 along H, CONTIGUOUS-4: slot (p,h512,q)
#     -> 4 consecutive logical-H 512*h512 + 4*p + q); kernel views uint32 -> fp8_e4m3fn_x4.
#     Scale: 3D [16, H/512, I] u8.
#   * down:    3D [128, ceil(I/512), H] uint32 (x4 along I, STRIDE-128: value byte
#     (p,i512,h) lane q -> logical-I i512*512 + q*128 + p) + 3D [16, ceil(I/512), H] u8
#     scale. UNLIKE the CTE down (4p+q), the TKG down kernel contracts the intermediate
#     stride-128, so its per-(8p x 4q) hardware MX scale block spans 4 disjoint I-runs
#     and CANNOT take the checkpoint's per-32-contiguous scale -> the down decode loaders
#     DEQUANT + RE-QUANTIZE in stride-128 blocks (see _down_requant_stride128_tkg).
#     DEVICE-CONFIRMED (TP1 NF.mlp sweep): stride-128 re-quant cos 0.998; 4p+q cos 0.028.
# The gate/up CONTIGUOUS-4 H-map pairs with pre_shuffle_h on the served decode activation:
# the plain-MX TKG kernel runs the activation through _layout_adapter_hbm
# (mlp_tkg_utils.py:167) INTERNALLY, which reinterprets its input H as [4_H,H/512,16_H,8_H].
# Feeding pre_shuffle_h([T,H]) (the SAME shuffle nkilib's own MX-TKG test applies at
# test_mlp_common.py:276) makes the adapter present logical-H = 512*h512 + 4*p + q
# (contiguous-4) at each (p,h512,q) contraction slot. nc_matmul_mx contracts weight and
# hidden over the SAME slot, so the gate/up weight uses this contiguous-4 map. Verified by
# pure-index trace (no device): pre_shuffle+adapter vs contiguous-4 weight = 0/1024 slot
# mismatches (vs 1020/1024 for stride-128, 768/1024 for plain-input). The down weight's
# I-contiguous 4p+q pack is the golden _fp8_to_down_x4 (test_mlp_common.py:930) verbatim.
# Unlike the CTE loaders, TKG does NOT force an EVEN 512-tile count (it shards H across
# the 2 LNC cores, not I), so it uses plain ceil(I/512).


def _n_i512_tiles_tkg(intermediate_size_per_rank: int) -> int:
    """Number of 512-tiles along the per-rank I dim for TKG (plain ceil).

    TKG shards the HIDDEN dim across the 2 LNC cores (not the intermediate dim), so
    the CTE even-tile constraint (``_n_i512_tiles``) does not apply -- the down
    projection just needs I padded up to a whole number of 512-tiles.
    """
    return math.ceil(intermediate_size_per_rank / _TILE_SIZE)


def mlp_gate_up_weight_loader_tkg(
    intermediate_size_per_rank: int,
    hidden_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Load an MLP gate/up FP8 weight -> TP shard -> TKG 3D ``[128, H/512, I]`` uint32.

    HF weight is ``[I, H]`` fp8 (output-major). Shard the I/output dim, transpose to
    ``[H, I_per_rank]``, then pack H with the CONTIGUOUS-4 convention: the 4 x4-lane
    values at slot ``(p, h512, i)`` are 4 CONSECUTIVE logical-H ``{512*h512 + 4*p + q :
    q=0..3}``. Concretely ``[H, I] -> reshape [h512, 128_p, 4_q, I] -> permute(1,0,3,2)
    [128, h512, I, 4] -> x4-pack the trailing q -> [128, h512, I]`` uint32. This is the
    nkilib golden ``_fp8_to_gate_up_x4(contiguous_x4=True)`` (test_mlp_common.py:921-924).

    Why CONTIGUOUS-4, paired with ``pre_shuffle_h`` on the served activation: the
    plain-MX TKG kernel runs the activation through ``_layout_adapter_hbm``
    (mlp_tkg_utils.py:167) INTERNALLY. That adapter reinterprets its input H as
    ``[4_H, H/512, 16_H, 8_H]`` and, after the load+transpose, lands hidden slot
    ``(p, h512, q)`` at whatever logical-H sits at buffer position
    ``q*(H/512)*128 + h512*128 + p``. Feeding PLAIN ``[T, H]`` there lands
    ``(p,h512,q)`` -> logical-H ``q*(H/512)*128 + h512*128 + p`` which matches NEITHER
    weight pack. Feeding ``pre_shuffle_h([T,H])`` first (reshape ``[h512,128,4]`` ->
    permute ``[4,h512,128]``, same as nkilib's own test at test_mlp_common.py:276)
    makes the adapter present logical-H ``512*h512 + 4*p + q`` = CONTIGUOUS-4 at every
    ``(p,h512,q)`` contraction slot. Verified by a pure-index trace: pre_shuffle+adapter
    vs contiguous-4 weight = 0/1024 slot mismatches (vs 1020/1024 for stride-128, and
    768/1024 for plain-input+stride-128). ``nc_matmul_mx`` contracts weight and hidden
    over the SAME ``(p,h512,q)`` slot, so the weight MUST use this contiguous-4 map. The
    MX block ``(8p x 4q)=32`` then spans 32 CONSECUTIVE logical-H ``[512*h512+32*(p//8)
    : +32)``, matching the checkpoint's per-32-H E8M0 scale delivered via the
    ``[16, H/512, I]`` scale layout.

    The FREE-I (output) dim is padded to a whole 512-tile (``ceil(I/512)*512``): the
    MX TKG kernel derives ``intermediate_size`` from ``gate_proj.shape[-1]`` and tiles
    by ``ceil(I/512)``, so an unpadded ``I`` (e.g. 6400 -> 12.5 tiles) would be read
    out-of-bounds and miscompute. Padded columns are fp8-0 (inert). Matches the down
    path and the model's ``gate_proj_weight_decode`` buffer shape.
    """
    assert hidden_size % _TILE_SIZE == 0, (
        f"hidden_size ({hidden_size}) must be a multiple of {_TILE_SIZE} for the "
        f"TKG H-x4 tiling."
    )
    n_h512 = hidden_size // _TILE_SIZE
    i_pad = _n_i512_tiles_tkg(intermediate_size_per_rank) * _TILE_SIZE

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1
        w = slices[0]  # HF [I, H] fp8
        i_total = w.get_shape()[0]
        assert i_total == intermediate_size_per_rank * num_shards, (
            f"gate/up weight I axis ({i_total}) must equal i_per_rank*num_shards "
            f"({intermediate_size_per_rank}*{num_shards})."
        )
        i_pr = intermediate_size_per_rank
        start = (rank % num_shards) * i_pr
        # Slice the I/output rows for this rank, transpose to [H, I_per_rank].
        w_HI = w[start : start + i_pr, :].T.contiguous()  # [H, I_pr]
        if i_pad != i_pr:  # pad the free-I up to a whole 512-tile (fp8 0)
            w_HI = pad_to_shape(w_HI, (hidden_size, i_pad))
        # Contiguous-4 x4 pack (_fp8_to_gate_up_x4 contiguous_x4=True): slot (p,h512,q)
        # packs 4 CONSECUTIVE logical-H = 512*h512 + 4*p + q, matching pre_shuffle_h +
        # _layout_adapter_hbm on the served activation.
        w4 = w_HI.reshape(n_h512, _PMAX, _Q_WIDTH, i_pad)  # [h512, 128_p, 4_q, I_pad]
        w4 = w4.permute(1, 0, 3, 2).contiguous()  # [128_p, h512, I_pad, 4_q]
        return x4_pack_fp8(w4, contraction_axis=3).reshape(_PMAX, n_h512, i_pad)

    return SafetensorsWeightLoader(transform=transform)


def mlp_gate_up_scale_loader_tkg(
    intermediate_size_per_rank: int,
    hidden_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Load an MLP gate/up MX weight scale -> TP shard -> TKG 3D ``[16, H/512, I]`` u8.

    HF scale is ``[I, H/32]`` uint8 (one E8M0 per 32 H-elements per output row). Shard
    the I/output dim, transpose to logical ``[H/32, I_pr]``, then reshape/permute to
    ``[16, H/512, I_pad]`` (the kernel undoes this via ``permute(1,0,2).reshape(H/32,
    I)`` — mlp_proj_mx_torch.py:92). "16" = per-512-tile H-scale count (512/32). This
    is the CTE loader's ``base`` layout WITHOUT the trailing 5D (4,128) split — the TKG
    kernel keeps the scale 3D. The free-I is padded to a whole 512-tile to match the
    gate/up weight buffer (the kernel tiles by ceil(I/512)).
    """
    # Symmetric with the paired weight loader (which asserts the same): the H/32 axis
    # must reshape exactly into [H/512, 16], i.e. H must be a whole number of 512-tiles.
    # The MX kernel does NOT validate scale shapes, so a non-multiple H would silently
    # swizzle the scale WRONG here while the weight loader would raise — assert both.
    assert hidden_size % _TILE_SIZE == 0, (
        f"hidden_size ({hidden_size}) must be a multiple of {_TILE_SIZE} for the "
        f"TKG H-x4 scale tiling."
    )
    n_h512 = hidden_size // _TILE_SIZE
    i_pad = _n_i512_tiles_tkg(intermediate_size_per_rank) * _TILE_SIZE

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1
        s = slices[0]  # HF [I, H/32] uint8
        i_total = s.get_shape()[0]
        assert i_total == intermediate_size_per_rank * num_shards, (
            f"gate/up scale I axis ({i_total}) must equal i_per_rank*num_shards "
            f"({intermediate_size_per_rank}*{num_shards})."
        )
        start = (rank % num_shards) * intermediate_size_per_rank
        # [I_pr, H/32] -> logical [H/32, I_pr] -> [16, H/512, I_pr] (H/32 row = h512*16 + p).
        s_log = s[start : start + intermediate_size_per_rank, :].T.contiguous()
        sc = s_log.reshape(n_h512, _SCALES_PER_TILE, -1).permute(1, 0, 2)
        if i_pad != intermediate_size_per_rank:  # pad free-I to a whole 512-tile
            # pad() returns a fresh contiguous tensor, so no extra .contiguous() needed.
            return torch.nn.functional.pad(sc, (0, i_pad - intermediate_size_per_rank))
        return sc.contiguous()  # materialize the permute (no pad path)

    return SafetensorsWeightLoader(transform=transform)


def _down_requant_stride128_tkg(
    w_HI_fp8: torch.Tensor,
    s_HI32_u8: torch.Tensor,
    intermediate_size_per_rank: int,
    hidden_size: int,
    num_shards: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shard + RE-QUANTIZE an MLP down weight into the TKG kernel's STRIDE-128 MX
    blocks, returning ``(value [128, n_i512, H] uint32, scale [16, n_i512, H] u8)``.

    DEVICE-CONFIRMED CONTRACT (2026-07-05, TP1 ``NF.mlp`` sweep — the authoritative
    oracle): the plain-MX TKG down kernel (``down_projection_mx_shard_H``) contracts
    the intermediate in **stride-128** I-order — its ``nc_matmul_mx`` reads value byte
    ``(p, i512, h)`` lane ``q`` as logical-I ``i512*512 + q*128 + p`` (NOT the CTE
    ``4p+q``: device cos 0.998 for stride-128 vs 0.028 for 4p+q). But a stride-128
    value's per-``(8p x 4q)=32`` hardware MX scale block then spans **4 disjoint I-runs**
    ``{i512*512 + q*128 + (8m..8m+7) : q=0..3}``, which the checkpoint's
    per-32-CONTIGUOUS-I scale CANNOT represent in one E8M0 slot (device: any
    scale-row reorder of the pre-quantized fp8 caps at cos 0.60-0.69). So we DEQUANT
    the checkpoint down weight and **re-quantize** it in the kernel's own stride-128
    blocks (value + matching E8M0 scale computed TOGETHER via nkilib
    ``quantize_to_mx``) → device cos 0.998. This is why the down decode loaders take
    BOTH ``[weight, weight_scale]`` (unlike gate/up, whose contiguous-4 pack keeps a
    contiguous-32-H hardware block and so consumes the checkpoint scale directly).

    Per 512-I tile ``t``: build ``A[128_p, H*4]`` with column index ``h*4 + q`` and
    value ``W_deq[t*512 + q*128 + p, h]``; ``quantize_to_mx(A, float8_e4m3fn_x4)``
    blocks exactly ``8p x 4q`` (= the hardware block) → packed ``[128, H]`` uint32 +
    scale ``[16, H]`` u8, stored at ``[:, t, :]``. ``I`` is padded to a whole
    512-tile (inert fp8 0). The CTE down path is unaffected (different kernel, 4p+q).
    """
    import numpy as np  # noqa: PLC0415 — loader-thread (CPU) only
    import nki.language as nl  # noqa: PLC0415
    from nkilib.core.utils.mx_torch_common import quantize_to_mx  # noqa: PLC0415

    n_i512 = _n_i512_tiles_tkg(intermediate_size_per_rank)
    padded_i = n_i512 * _TILE_SIZE
    i_pr = intermediate_size_per_rank
    scale_per_rank = i_pr // MX_GROUP_SIZE

    w_total = w_HI_fp8.get_shape()[1]
    assert w_total == i_pr * num_shards, (
        f"down weight I axis ({w_total}) must equal i_per_rank*num_shards "
        f"({i_pr}*{num_shards})."
    )
    # Guard the scale I/32 axis too (the weight above and the CTE sibling both assert
    # their I axis): a checkpoint whose down.weight and down.weight_scale I axes
    # disagree would slice the wrong scale window for ranks>0 -> a wrong dequant /
    # obscure broadcast crash instead of this loud, specific message.
    s_total = s_HI32_u8.get_shape()[1]
    assert s_total == scale_per_rank * num_shards, (
        f"down scale I/32 axis ({s_total}) must equal (i_per_rank//{MX_GROUP_SIZE})"
        f"*num_shards ({scale_per_rank}*{num_shards}); scale and weight I axes disagree."
    )
    w_start = (rank % num_shards) * i_pr
    s_start = (rank % num_shards) * scale_per_rank

    # Dequant the rank's [H, I_pr] fp8 slice with its per-32 scale (exact, pow-2).
    w_fp8 = w_HI_fp8[:, w_start : w_start + i_pr]  # [H, I_pr] fp8
    s_u8 = s_HI32_u8[:, s_start : s_start + scale_per_rank]  # [H, I_pr/32] u8
    scale = e8m0_to_scale(s_u8).repeat_interleave(MX_GROUP_SIZE, dim=1)  # [H, I_pr]
    w_deq_IH = (w_fp8.to(torch.float32) * scale).T.contiguous()  # [I_pr, H] fp32
    if w_deq_IH.shape[0] < padded_i:  # pad I up to a whole 512-tile (inert 0)
        w_deq_IH = pad_to_shape(w_deq_IH, (padded_i, hidden_size))

    val = torch.empty(_PMAX, n_i512, hidden_size, dtype=torch.uint32)
    sc = torch.empty(_SCALES_PER_TILE, n_i512, hidden_size, dtype=torch.uint8)
    for t in range(n_i512):
        # [512_I, H] -> stride-128 block A[128_p, H*4], col = h*4 + q,
        # value = W_deq[t*512 + q*128 + p, h].
        blk = w_deq_IH[t * _TILE_SIZE : (t + 1) * _TILE_SIZE].reshape(
            _Q_WIDTH, _PMAX, hidden_size
        )  # [q, p, h]
        A = (
            blk.permute(1, 2, 0).reshape(_PMAX, hidden_size * _Q_WIDTH).numpy()
        )  # [128, H*4] (col h*4+q)
        packed_np, scale_np = quantize_to_mx(
            A, nl.float8_e4m3fn_x4
        )  # [128, H], [16, H]
        val[:, t, :] = torch.from_numpy(packed_np.view(np.uint32).astype(np.int64)).to(
            torch.uint32
        )
        sc[:, t, :] = torch.from_numpy(scale_np.astype(np.uint8))
    return val, sc


def mlp_down_weight_loader_tkg(
    intermediate_size_per_rank: int,
    hidden_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Load MLP down weight -> TP shard -> RE-QUANTIZE -> TKG 3D ``[128, ceil(I/512), H]``
    uint32 (stride-128 blocks).

    Takes BOTH ``[down.weight (fp8 [H,I]), down.weight_scale (u8 [H,I/32])]`` and
    delegates to ``_down_requant_stride128_tkg`` (dequant → re-quant in the kernel's
    stride-128 MX blocks), returning the packed uint32 value tensor. See that helper
    for the device-confirmed rationale (the TKG down kernel contracts stride-128, and
    the checkpoint's per-32-contiguous scale can't map to it → must re-quantize).
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 2, (
            f"mlp_down_weight_loader_tkg expects [weight, weight_scale] (re-quant "
            f"needs both); got {len(slices)} slice(s)."
        )
        w, s = slices  # HF [H, I] fp8, [H, I/32] u8
        val, _ = _down_requant_stride128_tkg(
            w, s, intermediate_size_per_rank, hidden_size, num_shards, rank
        )
        return val

    return SafetensorsWeightLoader(transform=transform)


def mlp_down_scale_loader_tkg(
    intermediate_size_per_rank: int,
    hidden_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Load MLP down scale -> TP shard -> RE-QUANTIZE -> TKG 3D ``[16, ceil(I/512), H]``
    u8 (stride-128 blocks).

    Takes BOTH ``[down.weight, down.weight_scale]`` and delegates to
    ``_down_requant_stride128_tkg`` (same computation as the weight loader), returning
    the re-quantized E8M0 scale that matches the stride-128 value pack. See that helper
    for the device-confirmed rationale.
    """
    assert intermediate_size_per_rank % MX_GROUP_SIZE == 0, (
        f"i_per_rank ({intermediate_size_per_rank}) must be a multiple of "
        f"{MX_GROUP_SIZE} so the per-rank E8M0 scale row count is exact (no floor)."
    )

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 2, (
            f"mlp_down_scale_loader_tkg expects [weight, weight_scale] (re-quant "
            f"needs both); got {len(slices)} slice(s)."
        )
        w, s = slices  # HF [H, I] fp8, [H, I/32] u8
        _, sc = _down_requant_stride128_tkg(
            w, s, intermediate_size_per_rank, hidden_size, num_shards, rank
        )
        return sc

    return SafetensorsWeightLoader(transform=transform)


def build_mx_mlp_mappings(
    hf_prefix: str, layer_prefix: str = "", decode: bool = False
) -> dict:
    """Param -> checkpoint-key mappings for the native-MX MLP weights.

    The MLP counterpart of ``build_mx_attention_mappings``: single source of truth
    for the native-MX gate/up/down weight+scale mappings, shared by the CTE MLP
    three-way module test (and a future served MLP loader). The module's native
    loaders (``Qwen3VLTextMLPMX._setup_weight_loaders``) do the pack/shard; this only
    names the source tensors. Covers ONLY the MX-quantized projections -- the plain
    bf16 ``post_attention_layernorm`` is owned by each caller (like the attention
    builder omits the norms), so it is deliberately not emitted here.

    The CTE (prefill) and TKG (decode) MLP MX kernels take INCOMPATIBLE weight
    layouts (CTE gate/up 6D scalar-fp8 + 5D scale vs TKG 3D uint32 + 3D scale; CTE
    consumes a pre-quantized ROW ``[T, H+4]`` activation while TKG quantizes bf16
    online), so the module carries BOTH a ``_prefill`` and a ``_decode`` buffer set
    (mirrors attention's ``o_proj_weight_prefill`` / ``o_proj_weight_decode`` and
    llama3's ``gate_proj_weight`` vs ``..._tkg``). Both sets load from the SAME
    checkpoint key; their loaders pack the kernel-specific layout. When
    ``decode=False`` only the ``_prefill`` keys are emitted (the CTE-only module test).

    Args:
        hf_prefix: the HF mlp prefix, e.g.
            ``model.language_model.layers.{i}.mlp``.
        layer_prefix: prefixes the destination param names (empty when loading into a
            bare MLP wrapper; e.g. ``language_model.layers.{i}`` for the served model).
        decode: also emit the ``_decode`` (TKG) buffer keys (same checkpoint source).

    Returns:
        A ``{dest_param_name: [checkpoint_key]}`` dict for gate/up/down weight+scale.
    """
    assert hf_prefix.endswith(".mlp"), (
        f"build_mx_mlp_mappings expects an HF mlp prefix ending in '.mlp'; "
        f"got {hf_prefix!r}."
    )
    p = f"{layer_prefix}." if layer_prefix else ""
    mappings = {
        f"{p}mlp.gate_proj_weight_prefill": [f"{hf_prefix}.gate_proj.weight"],
        f"{p}mlp.gate_proj_weight_scale_prefill": [
            f"{hf_prefix}.gate_proj.weight_scale"
        ],
        f"{p}mlp.up_proj_weight_prefill": [f"{hf_prefix}.up_proj.weight"],
        f"{p}mlp.up_proj_weight_scale_prefill": [f"{hf_prefix}.up_proj.weight_scale"],
        f"{p}mlp.down_proj_weight_prefill": [f"{hf_prefix}.down_proj.weight"],
        f"{p}mlp.down_proj_weight_scale_prefill": [
            f"{hf_prefix}.down_proj.weight_scale"
        ],
    }
    if decode:
        # Same checkpoint keys; the _decode loaders pack the 3D TKG layout. The DOWN
        # decode buffers are the exception: the TKG down kernel contracts the
        # intermediate in STRIDE-128 I-order, whose per-(8p x 4q) hardware MX scale
        # block spans 4 disjoint I-runs and so CANNOT be fed the checkpoint's
        # per-32-contiguous-I scale directly (device-confirmed: any repack of the
        # pre-quantized fp8 caps at cos ~0.6-0.7). Both down loaders therefore
        # RE-QUANTIZE: they need the checkpoint WEIGHT *and* SCALE together
        # ([weight, weight_scale] -> dequant -> re-quant in stride-128 blocks), so map
        # BOTH source keys to each down decode param (the loader keeps only its piece).
        mappings.update(
            {
                f"{p}mlp.gate_proj_weight_decode": [f"{hf_prefix}.gate_proj.weight"],
                f"{p}mlp.gate_proj_weight_scale_decode": [
                    f"{hf_prefix}.gate_proj.weight_scale"
                ],
                f"{p}mlp.up_proj_weight_decode": [f"{hf_prefix}.up_proj.weight"],
                f"{p}mlp.up_proj_weight_scale_decode": [
                    f"{hf_prefix}.up_proj.weight_scale"
                ],
                f"{p}mlp.down_proj_weight_decode": [
                    f"{hf_prefix}.down_proj.weight",
                    f"{hf_prefix}.down_proj.weight_scale",
                ],
                f"{p}mlp.down_proj_weight_scale_decode": [
                    f"{hf_prefix}.down_proj.weight",
                    f"{hf_prefix}.down_proj.weight_scale",
                ],
            }
        )
    return mappings
