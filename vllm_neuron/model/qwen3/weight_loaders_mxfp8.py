# SPDX-License-Identifier: Apache-2.0
"""Dense Qwen3 native-MX weight-loader helpers.

This module keeps the dense Qwen3 MXFP8 path self-contained. It owns the native-MX
packing and checkpoint-mapping helpers used by
``vllm_neuron.model.qwen3.model_mxfp8`` instead of reaching into the Qwen3-VL
package for the same logic.
"""

from __future__ import annotations

import math

import torch

from vllm_neuron.functional.mlp import _PMAX, _Q_WIDTH, _TILE_SIZE
from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader, pad_to_shape

# Single-sourced from the checkpoint-contract parser so the group size can't drift.
from .quantization import MXFP8_GROUP_SIZE as MX_GROUP_SIZE

_FP8_DTYPE = torch.float8_e4m3fn
_SCALES_PER_TILE = _TILE_SIZE // MX_GROUP_SIZE
_SBUF_QUADRANT = 32  # partitions per SBUF quadrant (scale scatter step)
_N_QUADRANTS = _PMAX // _SBUF_QUADRANT  # 4
_ROWS_PER_QUADRANT = _SCALES_PER_TILE // _N_QUADRANTS  # 4
# Slots the MX scale start offset can take. The ISA rule ``scale_pidx % 4 == 0 and
# scale_pidx < 16`` gives {0, 4, 8, 12}: exactly 4, so exactly a factor of 4 may
# move from the free axis to the partition axis. See NKI "Packing scale values".
_SCALE_FOLD_SLOTS = 4
# Rows of a folded (pack64) scale: 4 quadrants x 4 slots x 4 rows.
_SCALE_FOLD_ROWS = _N_QUADRANTS * _SCALE_FOLD_SLOTS * _ROWS_PER_QUADRANT  # 64


def _scatter_scale_tkg(scale16: torch.Tensor) -> torch.Tensor:
    """Scatter the 16 E8M0 rows into the 128-partition layout the kernel loads.

    Place row ``j`` at partition ``32 * (j // 4) + (j % 4)``. Zero the other
    partitions. The in-tree TKG MLP kernel reads the scale from these scattered
    partitions and loads all 128 partitions with one DMA. So the host builds the
    layout once and the kernel avoids the per-quadrant scatter DMA.

    Args:
        scale16: uint8 ``[16, ...]`` E8M0 scale.

    Returns:
        uint8 ``[128, ...]`` scale with the quadrant scatter.

    Example:
        >>> import torch
        >>> _scatter_scale_tkg(torch.zeros(16, 2, dtype=torch.uint8)).shape
        torch.Size([128, 2])
    """
    assert scale16.shape[0] == _SCALES_PER_TILE, (
        f"scale16 must have {_SCALES_PER_TILE} rows for the quadrant scatter; "
        f"got {scale16.shape[0]}."
    )
    n_quadrants = _PMAX // _SBUF_QUADRANT
    rows_per_quad = scale16.shape[0] // n_quadrants
    out = torch.zeros((_PMAX, *scale16.shape[1:]), dtype=torch.uint8)
    for q in range(n_quadrants):
        dst0 = q * _SBUF_QUADRANT
        out[dst0 : dst0 + rows_per_quad] = scale16[
            q * rows_per_quad : (q + 1) * rows_per_quad
        ]
    return out


def fold_scale_gateup_tkg(
    scale16: torch.Tensor, i_pad: int, n_i512: int
) -> torch.Tensor:
    """Fold the gate/up sub-lane factor of 4 into the scale partition axis.

    The gate and up projections issue 4 matmuls for each 512-wide contraction
    tile, one for each 128-lane sub-block of I. The 4 matmuls differ only in the
    free slice of the scale, so that factor of 4 can move to the partition axis
    and leave the free axis 4x shorter. The MX scale start offset takes one of
    {0, 4, 8, 12}, which is exactly 4 slots, so the fold is exact.

    Row ``16 * q + 4 * m + r`` holds logical scale row ``4 * q + r`` for the I
    columns of 128-lane sub-block ``m``. The kernel then loads quadrant ``q``
    with one 16-partition DMA, and reads sub-block ``m`` at partition offset
    ``4 * m``.

    The last 512-tile can be ragged, because I pads to a multiple of 128 and not
    of 512. Its unused sub-block slots stay zero. The kernel never reads them:
    ``_i_lanes_in_tile`` skips the lanes that hold no weight.

    Args:
        scale16: uint8 ``[16, n_tile, i_pad]`` E8M0 scale.
        i_pad: Padded I extent (a multiple of 128).
        n_i512: Number of 512-wide contraction tiles.

    Returns:
        uint8 ``[64, n_tile, n_i512 * 128]`` folded scale.

    Example:
        >>> import torch
        >>> fold_scale_gateup_tkg(torch.zeros(16, 2, 256, dtype=torch.uint8), 256, 1).shape
        torch.Size([64, 2, 128])
    """
    assert scale16.shape[0] == _SCALES_PER_TILE, (
        f"scale16 must have {_SCALES_PER_TILE} rows for the sub-lane fold; got "
        f"{scale16.shape[0]}."
    )
    assert i_pad % _PMAX == 0, f"i_pad ({i_pad}) must be a multiple of {_PMAX}."
    n_tile = scale16.shape[1]
    out = torch.zeros((_SCALE_FOLD_ROWS, n_tile, n_i512 * _PMAX), dtype=torch.uint8)
    for q in range(_N_QUADRANTS):
        src_lo = q * _ROWS_PER_QUADRANT
        for m in range(_SCALE_FOLD_SLOTS):
            dst_lo = q * _SCALE_FOLD_SLOTS * _ROWS_PER_QUADRANT + m * _ROWS_PER_QUADRANT
            for t in range(n_i512):
                lo = t * _TILE_SIZE + m * _PMAX
                if lo >= i_pad:
                    continue  # a ragged tile has no lane here
                hi = min(lo + _PMAX, i_pad)
                out[
                    dst_lo : dst_lo + _ROWS_PER_QUADRANT,
                    :,
                    t * _PMAX : t * _PMAX + (hi - lo),
                ] = scale16[src_lo : src_lo + _ROWS_PER_QUADRANT, :, lo:hi]
    return out


def fold_scale_down_tkg(scale16: torch.Tensor) -> torch.Tensor:
    """Fold 4 down-projection contraction tiles into one scale buffer.

    The down projection issues ONE matmul for each (512-tile, H chunk) pair and
    it covers all 4 packed contraction lanes, so it has no sub-lane factor to
    fold. Fold the TILE index instead: 4 contraction tiles share one
    128-partition buffer, so ``ceil(n_i512 / 4)`` buffers replace ``n_i512``.

    Row ``16 * q + 4 * (k % 4) + r`` of buffer ``k // 4`` holds logical scale row
    ``4 * q + r`` of contraction tile ``k``.

    Args:
        scale16: uint8 ``[16, n_i512, H]`` E8M0 scale.

    Returns:
        uint8 ``[64, ceil(n_i512 / 4), H]`` folded scale.

    Example:
        >>> import torch
        >>> fold_scale_down_tkg(torch.zeros(16, 7, 64, dtype=torch.uint8)).shape
        torch.Size([64, 2, 64])
    """
    assert scale16.shape[0] == _SCALES_PER_TILE, (
        f"scale16 must have {_SCALES_PER_TILE} rows for the tile fold; got "
        f"{scale16.shape[0]}."
    )
    n_i512 = scale16.shape[1]
    width = scale16.shape[2]
    n_buf = math.ceil(n_i512 / _SCALE_FOLD_SLOTS)
    out = torch.zeros((_SCALE_FOLD_ROWS, n_buf, width), dtype=torch.uint8)
    for q in range(_N_QUADRANTS):
        src_lo = q * _ROWS_PER_QUADRANT
        for k in range(n_i512):
            dst_lo = (
                q * _SCALE_FOLD_SLOTS * _ROWS_PER_QUADRANT
                + (k % _SCALE_FOLD_SLOTS) * _ROWS_PER_QUADRANT
            )
            out[dst_lo : dst_lo + _ROWS_PER_QUADRANT, k // _SCALE_FOLD_SLOTS, :] = (
                scale16[src_lo : src_lo + _ROWS_PER_QUADRANT, k, :]
            )
    return out


def x4_pack_fp8(w: torch.Tensor, *, contraction_axis: int) -> torch.Tensor:
    """Pack 4 consecutive fp8 bytes along ``contraction_axis`` into one uint32.

    Output shape has ``shape[contraction_axis] // 4``; output dtype is
    ``torch.uint32``. The pack is a bit-level reinterpret (no float math), so
    the inverse round-trip with ``unpack_float8_e4m3fn_x4`` is bit-exact.
    Copied from the former shared STATIC_MX weight-pack module (removed) so
    this loader is self-contained; the qwen3_vl copy must stay in sync.
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


def _get_storage_dtype(slice_or_tensor) -> torch.dtype:
    """Return the storage dtype for a PySafeSlice-like object or tensor."""

    if isinstance(slice_or_tensor, torch.Tensor):
        return slice_or_tensor.dtype
    shape = slice_or_tensor.get_shape()
    sample = slice_or_tensor[tuple(0 for _ in shape)]
    return sample.dtype


def _assert_uint8_scale_storage(scale_slice, tensor_name: str) -> None:
    """Reject non-E8M0 scale storage before any silent cast to uint8 can happen."""

    dtype = _get_storage_dtype(scale_slice)
    assert dtype == torch.uint8, (
        f"{tensor_name} expected torch.uint8 E8M0 scale storage, got {dtype}."
    )


def pre_shuffle_h(x: torch.Tensor) -> torch.Tensor:
    """Shuffle the hidden axis for ``is_h_transposed_by_4=True`` decode kernels."""

    shape = x.shape
    hidden_size = shape[-1]
    lead = shape[:-1]
    reshaped = x.reshape(*lead, hidden_size // (_PMAX * _Q_WIDTH), _PMAX, _Q_WIDTH)
    permuted = reshaped.permute(*range(len(lead)), -1, -3, -2).contiguous()
    return permuted.reshape(shape)


def qkv_weight_pack_3d(weight_hi: torch.Tensor) -> torch.Tensor:
    """Pack ``[H, I]`` fp8 weights into the decode kernel's ``[H//4, I, 4]`` view."""

    assert weight_hi.dtype == _FP8_DTYPE
    hidden_size, input_size = weight_hi.shape
    assert hidden_size % _Q_WIDTH == 0
    return (
        weight_hi.reshape(hidden_size // _Q_WIDTH, _Q_WIDTH, input_size)
        .transpose(1, 2)
        .contiguous()
    )


def qkv_shard_offsets(
    q_total: int,
    kv_total: int,
    num_shards: int,
    num_kv_replicas: int,
    rank: int,
) -> tuple[slice, slice]:
    """Return the per-rank Q and KV row ranges for a fused-QKV checkpoint."""

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
    """Fuse Q/K/V fp8 weights and pack them for the decode MX kernel."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        q_slice, k_slice, v_slice = slices
        q_rows, kv_rows = qkv_shard_offsets(
            q_slice.get_shape()[0],
            k_slice.get_shape()[0],
            num_shards,
            num_kv_replicas,
            rank,
        )
        q = q_slice[q_rows, :]
        k = k_slice[kv_rows, :]
        v = v_slice[kv_rows, :]

        fused_of = torch.cat([q, k, v], dim=0)
        fused_hi = (
            fused_of.contiguous().view(torch.uint8).t().contiguous().view(_FP8_DTYPE)
        )
        return qkv_weight_pack_3d(fused_hi)

    return SafetensorsWeightLoader(transform=transform)


def fused_qkv_scale_loader_3d(
    num_shards: int,
    num_kv_replicas: int,
) -> SafetensorsWeightLoader:
    """Fuse Q/K/V MX scales and transpose them to ``[H//32, fused_I]``."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        q_slice, k_slice, v_slice = slices
        _assert_uint8_scale_storage(q_slice, "q_proj.weight_scale")
        _assert_uint8_scale_storage(k_slice, "k_proj.weight_scale")
        _assert_uint8_scale_storage(v_slice, "v_proj.weight_scale")
        q_rows, kv_rows = qkv_shard_offsets(
            q_slice.get_shape()[0],
            k_slice.get_shape()[0],
            num_shards,
            num_kv_replicas,
            rank,
        )
        q = q_slice[q_rows, :]
        k = k_slice[kv_rows, :]
        v = v_slice[kv_rows, :]

        fused = torch.cat([q, k, v], dim=0)
        return fused.t().contiguous()

    return SafetensorsWeightLoader(transform=transform)


def o_proj_weight_pack_mx(weight_ndh: torch.Tensor) -> torch.Tensor:
    """Pack ``[N*D, H]`` fp8 weights into the CTE kernel's uint32 container."""

    assert weight_ndh.dtype == _FP8_DTYPE
    nd_total, hidden_size = weight_ndh.shape
    assert nd_total % _Q_WIDTH == 0
    rearranged = (
        weight_ndh.reshape(nd_total // _Q_WIDTH, _Q_WIDTH, hidden_size)
        .transpose(1, 2)
        .reshape(nd_total // _Q_WIDTH, hidden_size * _Q_WIDTH)
    )
    return x4_pack_fp8(rearranged, contraction_axis=-1)


def o_proj_weight_pack_3d(weight_ndh: torch.Tensor) -> torch.Tensor:
    """Pack ``[N*D, H]`` fp8 weights into the decode kernel's 3D fp8 layout."""

    assert weight_ndh.dtype == _FP8_DTYPE
    nd_total, hidden_size = weight_ndh.shape
    assert nd_total % _Q_WIDTH == 0
    return (
        weight_ndh.reshape(nd_total // _Q_WIDTH, _Q_WIDTH, hidden_size)
        .transpose(1, 2)
        .contiguous()
    )


def o_proj_scale_loader(
    shard_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Shard and transpose ``o_proj.weight_scale`` for both decode and prefill."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1
        scale = slices[0]
        _assert_uint8_scale_storage(scale, "o_proj.weight_scale")
        scale_per_rank = shard_size // MX_GROUP_SIZE
        nd_scale_total = scale.get_shape()[1]
        assert nd_scale_total == scale_per_rank * num_shards, (
            f"o_proj weight_scale N*D//32 axis ({nd_scale_total}) must equal "
            f"(shard_size//{MX_GROUP_SIZE})*num_shards "
            f"({scale_per_rank}*{num_shards}={scale_per_rank * num_shards})."
        )
        start = (rank % num_shards) * scale_per_rank
        shard_scale = scale[:, start : start + scale_per_rank]
        return shard_scale.t().contiguous()

    return SafetensorsWeightLoader(transform=transform)


def o_proj_weight_loader(
    shard_size: int,
    num_shards: int,
    pack_3d: bool = False,
) -> SafetensorsWeightLoader:
    """Shard, transpose, and pack ``o_proj.weight`` for the target kernel."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1
        weight = slices[0]
        nd_total = weight.get_shape()[1]
        assert nd_total == shard_size * num_shards
        start = (rank % num_shards) * shard_size
        sliced_hnd = weight[:, start : start + shard_size]
        bytes_view = sliced_hnd.contiguous().view(torch.uint8)
        bytes_ndh = bytes_view.transpose(0, 1).contiguous()
        weight_ndh = bytes_ndh.view(_FP8_DTYPE)
        if pack_3d:
            return o_proj_weight_pack_3d(weight_ndh)
        return o_proj_weight_pack_mx(weight_ndh)

    return SafetensorsWeightLoader(transform=transform)


def build_mx_attention_mappings(
    hf_prefix: str, layer_prefix: str = "", prefill: bool = True
) -> dict:
    """Return checkpoint mappings for native-MX QKV and O-proj tensors.

    Set ``prefill=False`` to omit the prefill-layout O-proj key. A decode-only
    pool does not create that parameter, so a mapping for it has no target.
    """

    assert ".self_attn" in hf_prefix, (
        f"build_mx_attention_mappings expects an HF self_attn prefix ending in "
        f"'.self_attn'; got {hf_prefix!r}."
    )
    prefix = f"{layer_prefix}." if layer_prefix else ""
    mappings = {
        f"{prefix}self_attn.qkv_proj_weight": [
            f"{hf_prefix}.q_proj.weight",
            f"{hf_prefix}.k_proj.weight",
            f"{hf_prefix}.v_proj.weight",
        ],
        f"{prefix}self_attn.qkv_weight_scale": [
            f"{hf_prefix}.q_proj.weight_scale",
            f"{hf_prefix}.k_proj.weight_scale",
            f"{hf_prefix}.v_proj.weight_scale",
        ],
        f"{prefix}self_attn.o_proj_weight_decode": [f"{hf_prefix}.o_proj.weight"],
        f"{prefix}self_attn.o_proj_weight_scale": [f"{hf_prefix}.o_proj.weight_scale"],
    }
    if prefill:
        mappings[f"{prefix}self_attn.o_proj_weight_prefill"] = [
            f"{hf_prefix}.o_proj.weight"
        ]
    return mappings


def _n_i512_tiles(intermediate_size_per_rank: int) -> int:
    """Return the even 512-tile count required by the CTE MLP kernels."""

    return 2 * math.ceil(intermediate_size_per_rank / (2 * _TILE_SIZE))


def mlp_gate_up_weight_loader(
    intermediate_size_per_rank: int,
    hidden_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Pack gate/up prefill weights into the CTE kernel's 6D layout."""

    assert hidden_size % _TILE_SIZE == 0, (
        f"hidden_size ({hidden_size}) must be a multiple of {_TILE_SIZE} for the "
        f"H_X4_INNERMOST tiling."
    )
    n_h512 = hidden_size // _TILE_SIZE
    n_i512 = _n_i512_tiles(intermediate_size_per_rank)
    padded_i = n_i512 * _TILE_SIZE

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1
        weight = slices[0]
        i_total = weight.get_shape()[0]
        assert i_total == intermediate_size_per_rank * num_shards, (
            f"gate/up weight I axis ({i_total}) must equal i_per_rank*num_shards "
            f"({intermediate_size_per_rank}*{num_shards})."
        )
        start = (rank % num_shards) * intermediate_size_per_rank
        weight_hi = weight[start : start + intermediate_size_per_rank, :].T
        if weight_hi.shape[1] < padded_i:
            weight_hi = pad_to_shape(weight_hi, (hidden_size, padded_i))
        weight_6d = weight_hi.reshape(n_h512, _PMAX, _Q_WIDTH, n_i512, _PMAX, _Q_WIDTH)
        return weight_6d.permute(1, 0, 3, 5, 4, 2).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def mlp_gate_up_scale_loader(
    intermediate_size_per_rank: int,
    hidden_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Pack gate/up prefill scales into the CTE kernel's 5D layout."""

    assert hidden_size % _TILE_SIZE == 0, (
        f"hidden_size ({hidden_size}) must be a multiple of {_TILE_SIZE} for the "
        f"H_X4_INNERMOST scale tiling."
    )
    n_h512 = hidden_size // _TILE_SIZE
    n_i512 = _n_i512_tiles(intermediate_size_per_rank)
    padded_i = n_i512 * _TILE_SIZE

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1
        scale = slices[0]
        _assert_uint8_scale_storage(scale, "gate_proj/up_proj.weight_scale")
        i_total = scale.get_shape()[0]
        assert i_total == intermediate_size_per_rank * num_shards, (
            f"gate/up scale I axis ({i_total}) must equal i_per_rank*num_shards "
            f"({intermediate_size_per_rank}*{num_shards})."
        )
        start = (rank % num_shards) * intermediate_size_per_rank
        scale_logical = scale[
            start : start + intermediate_size_per_rank, :
        ].T.contiguous()
        base = (
            scale_logical.reshape(n_h512, _SCALES_PER_TILE, -1)
            .permute(1, 0, 2)
            .contiguous()
        )
        if base.shape[2] < padded_i:
            base = torch.nn.functional.pad(base, (0, padded_i - base.shape[2]))
        scale_5d = base.reshape(_SCALES_PER_TILE, n_h512, n_i512, _PMAX, _Q_WIDTH)
        return scale_5d.permute(0, 1, 2, 4, 3).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def mlp_down_weight_loader(
    intermediate_size_per_rank: int,
    hidden_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Pack down-proj prefill weights into the CTE kernel's 4D layout."""

    n_i512 = _n_i512_tiles(intermediate_size_per_rank)
    padded_i = n_i512 * _TILE_SIZE

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1
        weight = slices[0]
        i_total = weight.get_shape()[1]
        assert i_total == intermediate_size_per_rank * num_shards, (
            f"down weight I axis ({i_total}) must equal i_per_rank*num_shards "
            f"({intermediate_size_per_rank}*{num_shards})."
        )
        start = (rank % num_shards) * intermediate_size_per_rank
        weight_ih = weight[:, start : start + intermediate_size_per_rank].T
        if weight_ih.shape[0] < padded_i:
            weight_ih = pad_to_shape(weight_ih, (padded_i, hidden_size))
        weight_4d = weight_ih.reshape(n_i512, _PMAX, _Q_WIDTH, hidden_size)
        return weight_4d.permute(1, 0, 3, 2).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def mlp_down_scale_loader(
    intermediate_size_per_rank: int,
    hidden_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Pack down-proj prefill scales into the CTE kernel's ``[16, I//512, H]`` layout."""

    assert intermediate_size_per_rank % MX_GROUP_SIZE == 0, (
        f"i_per_rank ({intermediate_size_per_rank}) must be a multiple of "
        f"{MX_GROUP_SIZE} so the per-rank E8M0 scale row count is exact (no floor)."
    )
    n_i512 = _n_i512_tiles(intermediate_size_per_rank)
    scale_per_rank = intermediate_size_per_rank // MX_GROUP_SIZE
    padded_scale_rows = n_i512 * _SCALES_PER_TILE

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1
        scale = slices[0]
        _assert_uint8_scale_storage(scale, "down_proj.weight_scale")
        scale_total = scale.get_shape()[1]
        assert scale_total == scale_per_rank * num_shards, (
            f"down scale I/32 axis ({scale_total}) must equal (i_per_rank//"
            f"{MX_GROUP_SIZE})*num_shards ({scale_per_rank}*{num_shards})."
        )
        start = (rank % num_shards) * scale_per_rank
        scale_logical = scale[:, start : start + scale_per_rank].T.contiguous()
        if scale_logical.shape[0] < padded_scale_rows:
            scale_logical = torch.nn.functional.pad(
                scale_logical, (0, 0, 0, padded_scale_rows - scale_logical.shape[0])
            )
        scale_3d = scale_logical.reshape(n_i512, _SCALES_PER_TILE, hidden_size)
        return scale_3d.permute(1, 0, 2).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _n_i512_tiles_tkg(intermediate_size_per_rank: int) -> int:
    """Return the TKG MLP tile count, which is a plain ceil over 512-wide tiles."""

    return math.ceil(intermediate_size_per_rank / _TILE_SIZE)


def gate_up_i_pad_tkg(intermediate_size_per_rank: int, intree: bool) -> int:
    """Return the padded decode gate/up I extent for the selected MLP kernel.

    I is the free axis of the gate/up decode weight, and the in-tree kernel
    consumes it one 128-wide lane at a time, so that kernel needs only a multiple
    of 128. The shared nkilib kernel needs a whole 512-tile.

    Qwen3-32B pays 12% on the I axis at TP8 (3200 -> 3584) and 28% at TP16 and
    TP32 under the 512 rule. The 128 rule gives 3200, 1664 and 896.

    Args:
        intermediate_size_per_rank: Per-rank intermediate size.
        intree: True for the in-tree kernel, False for the nkilib kernel.

    Returns:
        Padded I extent in elements.

    Example:
        >>> gate_up_i_pad_tkg(3200, intree=True), gate_up_i_pad_tkg(3200, False)
        (3200, 3584)
    """
    if intree:
        return math.ceil(intermediate_size_per_rank / _PMAX) * _PMAX
    return _n_i512_tiles_tkg(intermediate_size_per_rank) * _TILE_SIZE


def mlp_gate_up_weight_loader_tkg(
    intermediate_size_per_rank: int,
    hidden_size: int,
    num_shards: int,
    i_pad: int | None = None,
) -> SafetensorsWeightLoader:
    """Pack gate/up decode weights into the TKG kernel's uint32 layout.

    ``i_pad`` gives the padded I extent. The caller passes the same value it used
    to size the parameter, so the layout and the buffer cannot disagree. The
    default pads to a whole 512-tile, which the nkilib kernel needs.
    """

    assert hidden_size % _TILE_SIZE == 0, (
        f"hidden_size ({hidden_size}) must be a multiple of {_TILE_SIZE} for the "
        f"TKG H-x4 tiling."
    )
    n_h512 = hidden_size // _TILE_SIZE
    if i_pad is None:
        i_pad = _n_i512_tiles_tkg(intermediate_size_per_rank) * _TILE_SIZE
    assert i_pad >= intermediate_size_per_rank and i_pad % _PMAX == 0, (
        f"i_pad ({i_pad}) must hold i_per_rank ({intermediate_size_per_rank}) and "
        f"be a multiple of {_PMAX}."
    )

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1
        weight = slices[0]
        i_total = weight.get_shape()[0]
        assert i_total == intermediate_size_per_rank * num_shards, (
            f"gate/up weight I axis ({i_total}) must equal i_per_rank*num_shards "
            f"({intermediate_size_per_rank}*{num_shards})."
        )
        start = (rank % num_shards) * intermediate_size_per_rank
        weight_hi = weight[start : start + intermediate_size_per_rank, :].T.contiguous()
        if i_pad != intermediate_size_per_rank:
            weight_hi = pad_to_shape(weight_hi, (hidden_size, i_pad))
        weight_4d = weight_hi.reshape(n_h512, _PMAX, _Q_WIDTH, i_pad)
        weight_4d = weight_4d.permute(1, 0, 3, 2).contiguous()
        return x4_pack_fp8(weight_4d, contraction_axis=3).reshape(_PMAX, n_h512, i_pad)

    return SafetensorsWeightLoader(transform=transform)


def mlp_gate_up_scale_loader_tkg(
    intermediate_size_per_rank: int,
    hidden_size: int,
    num_shards: int,
    scatter: bool = False,
    i_pad: int | None = None,
    fold: bool = False,
) -> SafetensorsWeightLoader:
    """Pack gate/up decode scales into the TKG kernel's E8M0 layout.

    Returns the ``[16, H//512, I]`` scale by default. The shared nkilib kernel
    wants that layout, so it takes the default.

    With ``fold=True`` the sub-lane factor of 4 moves to the partition axis and
    the result is the ``[64, H//512, n_i512*128]`` pack64 layout (see
    ``fold_scale_gateup_tkg``). This is the preferred in-tree form: it reads 3.6x
    fewer scale bytes from HBM than ``scatter`` and needs 3.6x less SBUF.
    ``fold`` takes precedence over ``scatter``.

    With ``scatter=True`` the 16 E8M0 rows are scattered into the
    ``[128, H//512, I]`` layout (the first 4 partitions of each 32-partition
    quadrant), so the kernel loads all partitions with one DMA. 112 of every 128
    partitions are then host-written zero, which is why ``fold`` supersedes it.
    """

    assert hidden_size % _TILE_SIZE == 0, (
        f"hidden_size ({hidden_size}) must be a multiple of {_TILE_SIZE} for the "
        f"TKG H-x4 scale tiling."
    )
    n_h512 = hidden_size // _TILE_SIZE
    if i_pad is None:
        i_pad = _n_i512_tiles_tkg(intermediate_size_per_rank) * _TILE_SIZE
    assert i_pad >= intermediate_size_per_rank and i_pad % _PMAX == 0, (
        f"i_pad ({i_pad}) must hold i_per_rank ({intermediate_size_per_rank}) and "
        f"be a multiple of {_PMAX}."
    )

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1
        scale = slices[0]
        _assert_uint8_scale_storage(scale, "gate_proj/up_proj.weight_scale")
        i_total = scale.get_shape()[0]
        assert i_total == intermediate_size_per_rank * num_shards, (
            f"gate/up scale I axis ({i_total}) must equal i_per_rank*num_shards "
            f"({intermediate_size_per_rank}*{num_shards})."
        )
        start = (rank % num_shards) * intermediate_size_per_rank
        scale_logical = scale[
            start : start + intermediate_size_per_rank, :
        ].T.contiguous()
        scale_3d = scale_logical.reshape(n_h512, _SCALES_PER_TILE, -1).permute(1, 0, 2)
        if i_pad != intermediate_size_per_rank:
            scale_3d = torch.nn.functional.pad(
                scale_3d, (0, i_pad - intermediate_size_per_rank)
            )
        scale_3d = scale_3d.contiguous()
        # The in-tree kernel reads a folded or a scattered layout. nkilib reads
        # the 16-row layout, so return it untouched for the fallback.
        if fold:
            return fold_scale_gateup_tkg(
                scale_3d, i_pad, _n_i512_tiles_tkg(intermediate_size_per_rank)
            )
        if scatter:
            return _scatter_scale_tkg(scale_3d)
        return scale_3d

    return SafetensorsWeightLoader(transform=transform)


def e8m0_to_scale(scale_u8: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 exponents into fp32 powers of two."""

    assert scale_u8.dtype == torch.uint8, (
        f"E8M0 scale expected torch.uint8, got {scale_u8.dtype}; if it is a "
        "float8/float dtype use s.to(float32) directly with NO -127."
    )
    return torch.exp2(scale_u8.to(torch.float32) - 127.0)


def _down_requant_stride128_tkg(
    weight_hi_fp8,
    scale_hi32_u8,
    intermediate_size_per_rank: int,
    hidden_size: int,
    num_shards: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-quantize a down-proj shard into the TKG kernel's stride-128 MX layout."""

    n_i512 = _n_i512_tiles_tkg(intermediate_size_per_rank)
    padded_i = n_i512 * _TILE_SIZE
    i_per_rank = intermediate_size_per_rank
    scale_per_rank = i_per_rank // MX_GROUP_SIZE
    _assert_uint8_scale_storage(scale_hi32_u8, "down_proj.weight_scale")

    weight_total = weight_hi_fp8.get_shape()[1]
    assert weight_total == i_per_rank * num_shards, (
        f"down weight I axis ({weight_total}) must equal i_per_rank*num_shards "
        f"({i_per_rank}*{num_shards})."
    )
    scale_total = scale_hi32_u8.get_shape()[1]
    assert scale_total == scale_per_rank * num_shards, (
        f"down scale I/32 axis ({scale_total}) must equal (i_per_rank//{MX_GROUP_SIZE})"
        f"*num_shards ({scale_per_rank}*{num_shards}); scale and weight I axes disagree."
    )

    weight_start = (rank % num_shards) * i_per_rank
    scale_start = (rank % num_shards) * scale_per_rank

    weight_fp8 = weight_hi_fp8[:, weight_start : weight_start + i_per_rank]
    scale_u8 = scale_hi32_u8[:, scale_start : scale_start + scale_per_rank]
    scale = e8m0_to_scale(scale_u8).repeat_interleave(MX_GROUP_SIZE, dim=1)
    weight_deq_ih = (weight_fp8.to(torch.float32) * scale).T.contiguous()
    if weight_deq_ih.shape[0] < padded_i:
        weight_deq_ih = pad_to_shape(weight_deq_ih, (padded_i, hidden_size))

    import numpy as np  # noqa: PLC0415
    import nki.language as nl  # noqa: PLC0415
    from nkilib.core.utils.mx_torch_common import quantize_to_mx  # noqa: PLC0415

    value = torch.empty(_PMAX, n_i512, hidden_size, dtype=torch.uint32)
    scale_out = torch.empty(_SCALES_PER_TILE, n_i512, hidden_size, dtype=torch.uint8)
    for tile_idx in range(n_i512):
        block = weight_deq_ih[
            tile_idx * _TILE_SIZE : (tile_idx + 1) * _TILE_SIZE
        ].reshape(_Q_WIDTH, _PMAX, hidden_size)
        packed_input = (
            block.permute(1, 2, 0).reshape(_PMAX, hidden_size * _Q_WIDTH).numpy()
        )
        packed_np, scale_np = quantize_to_mx(packed_input, nl.float8_e4m3fn_x4)
        value[:, tile_idx, :] = torch.from_numpy(
            packed_np.view(np.uint32).astype(np.int64)
        ).to(torch.uint32)
        scale_out[:, tile_idx, :] = torch.from_numpy(scale_np.astype(np.uint8))
    return value, scale_out


def _down_requant_stride128_tkg_cached(
    cache: dict | None,
    slices: list,
    intermediate_size_per_rank: int,
    hidden_size: int,
    num_shards: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``_down_requant_stride128_tkg`` at most once per (layer, rank).

    The TKG decode down-proj packs weight and scale as two separate parameters,
    both mapped to the same ``[weight, weight_scale]`` checkpoint pair, so each
    would otherwise re-run the expensive dequant->re-quant (numpy + nki
    quantize_mx over ``n_i512`` tiles) for the same shard. ``_setup_weight_loaders``
    hands both the weight and scale TKG loaders one shared ``cache`` dict per MLP
    layer; the first loader computes ``(value, scale_out)`` and caches it, the
    second reads it back. ``rank`` is fixed per worker, so this is one slot.
    """
    assert len(slices) == 2, (
        "TKG down-proj re-quant expects [weight, weight_scale] slices; got "
        f"{len(slices)}."
    )
    if cache is not None and rank in cache:
        return cache[rank]
    weight, scale = slices
    result = _down_requant_stride128_tkg(
        weight, scale, intermediate_size_per_rank, hidden_size, num_shards, rank
    )
    if cache is not None:
        cache[rank] = result
    return result


def mlp_down_weight_loader_tkg(
    intermediate_size_per_rank: int,
    hidden_size: int,
    num_shards: int,
    cache: dict | None = None,
) -> SafetensorsWeightLoader:
    """Pack down-proj decode weights into the TKG kernel's stride-128 layout."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        value, _ = _down_requant_stride128_tkg_cached(
            cache, slices, intermediate_size_per_rank, hidden_size, num_shards, rank
        )
        return value

    return SafetensorsWeightLoader(transform=transform)


def mlp_down_scale_loader_tkg(
    intermediate_size_per_rank: int,
    hidden_size: int,
    num_shards: int,
    cache: dict | None = None,
    scatter: bool = False,
    fold: bool = False,
) -> SafetensorsWeightLoader:
    """Pack down-proj decode scales into the TKG kernel's E8M0 layout.

    Returns the ``[16, I//512, H]`` scale by default. The shared nkilib kernel
    wants that layout, so it takes the default.

    With ``fold=True`` four contraction tiles share one 128-partition buffer and
    the result is the ``[64, ceil(n_i512/4), H]`` layout (see
    ``fold_scale_down_tkg``). The down projection has no sub-lane factor, so it
    folds the tile index instead. ``fold`` takes precedence over ``scatter``.

    With ``scatter=True`` the 16 E8M0 rows are scattered into the
    ``[128, I//512, H]`` layout (the first 4 partitions of each 32-partition
    quadrant), so the kernel loads all partitions with one DMA.
    """

    assert intermediate_size_per_rank % MX_GROUP_SIZE == 0, (
        f"i_per_rank ({intermediate_size_per_rank}) must be a multiple of "
        f"{MX_GROUP_SIZE} so the per-rank E8M0 scale row count is exact (no floor)."
    )

    def transform(slices: list, rank: int) -> torch.Tensor:
        _, scale_out = _down_requant_stride128_tkg_cached(
            cache, slices, intermediate_size_per_rank, hidden_size, num_shards, rank
        )
        # Fold or scatter for the in-tree kernel; 16-row layout for nkilib.
        if fold:
            return fold_scale_down_tkg(scale_out)
        if scatter:
            return _scatter_scale_tkg(scale_out)
        return scale_out

    return SafetensorsWeightLoader(transform=transform)


def build_mx_mlp_mappings(
    hf_prefix: str,
    layer_prefix: str = "",
    decode: bool = False,
    prefill: bool = True,
) -> dict:
    """Return checkpoint mappings for native-MX MLP tensors.

    Set ``prefill=False`` to omit the six prefill-layout keys. A decode-only pool
    does not create those parameters, so a mapping for them has no target.
    """

    assert hf_prefix.endswith(".mlp"), (
        f"build_mx_mlp_mappings expects an HF mlp prefix ending in '.mlp'; "
        f"got {hf_prefix!r}."
    )
    assert decode or prefill, (
        "build_mx_mlp_mappings must emit at least one layout. Set decode=True, "
        "or prefill=True, or both."
    )
    prefix = f"{layer_prefix}." if layer_prefix else ""
    mappings = {}
    if prefill:
        mappings.update(
            {
                f"{prefix}mlp.gate_proj_weight_prefill": [
                    f"{hf_prefix}.gate_proj.weight"
                ],
                f"{prefix}mlp.gate_proj_weight_scale_prefill": [
                    f"{hf_prefix}.gate_proj.weight_scale"
                ],
                f"{prefix}mlp.up_proj_weight_prefill": [f"{hf_prefix}.up_proj.weight"],
                f"{prefix}mlp.up_proj_weight_scale_prefill": [
                    f"{hf_prefix}.up_proj.weight_scale"
                ],
                f"{prefix}mlp.down_proj_weight_prefill": [
                    f"{hf_prefix}.down_proj.weight"
                ],
                f"{prefix}mlp.down_proj_weight_scale_prefill": [
                    f"{hf_prefix}.down_proj.weight_scale"
                ],
            }
        )
    if decode:
        mappings.update(
            {
                f"{prefix}mlp.gate_proj_weight_decode": [
                    f"{hf_prefix}.gate_proj.weight"
                ],
                f"{prefix}mlp.gate_proj_weight_scale_decode": [
                    f"{hf_prefix}.gate_proj.weight_scale"
                ],
                f"{prefix}mlp.up_proj_weight_decode": [f"{hf_prefix}.up_proj.weight"],
                f"{prefix}mlp.up_proj_weight_scale_decode": [
                    f"{hf_prefix}.up_proj.weight_scale"
                ],
                f"{prefix}mlp.down_proj_weight_decode": [
                    f"{hf_prefix}.down_proj.weight",
                    f"{hf_prefix}.down_proj.weight_scale",
                ],
                f"{prefix}mlp.down_proj_weight_scale_decode": [
                    f"{hf_prefix}.down_proj.weight",
                    f"{hf_prefix}.down_proj.weight_scale",
                ],
            }
        )
    return mappings
