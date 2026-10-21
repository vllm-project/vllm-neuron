# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2.5 weight loaders (FP8-blockwise checkpoint -> BF16 runtime).

The released base checkpoint stores nearly every matmul weight as
``float8_e4m3`` with a companion ``*.weight_scale_inv`` tensor holding one
fp32 scale per ``128 x 128`` tile (``quantization_config.weight_block_size``).
``config.dtype`` is ``bfloat16`` — that is the COMPUTE dtype, not the storage
dtype. This port runs BF16 end to end, so every loader here dequantizes
host-side inside the ``SafetensorsWeightLoader`` transform and returns BF16.

Two distinct on-disk scale-grid layouts exist and they are NOT interchangeable:

1. **Plain / flat grid** — used by every weight except the fused QKV. A weight
   of shape ``[R, C]`` pairs with a scale of shape
   ``[ceil(R/128), ceil(C/128)]``; tile ``(i, j)`` scales rows
   ``[128i, 128i+128)`` x cols ``[128j, 128j+128)``. Verified against the
   checkpoint: ``mlp.gate_proj [16384, 4096] <-> [128, 32]``,
   ``mlp.down_proj [4096, 16384] <-> [32, 128]``,
   ``experts.N.gate_proj [2048, 4096] <-> [16, 32]``,
   ``experts.N.down_proj [4096, 2048] <-> [32, 16]``.

2. **Pre-sharded fused QKV** — ``self_attn.qkv_proj.weight`` is stored as
   ``disk_tp`` CONCATENATED tensor-parallel shards, each shard laid out as
   ``[Q rows | K rows | V rows]``, and each of those three sections gets its
   OWN ceil-padded 128-row scale grid. That padding is why the scale row count
   does not match a naive ``ceil(total_rows / 128)``:

       full-attn layer (nh=64, nkv=4, hd=192, vhd=128), disk_tp=4
         per-shard rows   = 16*192 + 1*192 + 1*128 = 3392  (x4 = 13568 ✓)
         per-shard blocks = 24     + 2     + 1     = 27    (x4 = 108   ✓)
       SWA layer (nkv=8)
         per-shard rows   = 16*192 + 2*192 + 2*128 = 3712  (x4 = 14848 ✓)
         per-shard blocks = 24     + 3     + 2     = 29    (x4 = 116   ✓)

   A naive flat grid would give 106 / 112 rows respectively, so treating this
   tensor as layout (1) silently mis-scales most of Q. ``disk_tp`` is recovered
   from the scale row count (the weight shape alone is invariant to it), so a
   re-sharded or unsharded re-export loads correctly too.

The one exception to "returns BF16" is the routed experts in fp8 mode
(``MIMO_V2_5_EXPERT_DTYPE``, auto-selected for V2.5-Pro, whose BF16 experts do
not fit): :func:`expert_gate_up_requant_loader` / :func:`expert_down_requant_loader`
dequantize the 128x128 blocks to fp32 and requantize them to one scale per
OUTPUT channel inside Trn2's e4m3 range (max 240, not e4m3fn's 448), so the
published checkpoint loads as is with no offline conversion.

All loaders read only the tiles they need: the fused-QKV loader touches just
the 128-row blocks covering this rank's heads (~2 blocks of Q instead of the
whole 3072-row Q section), and the TP-sharded MLP loaders slice before
dequantizing.
"""

import math
import threading

import torch

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

FP8_BLOCK = 128

# Largest magnitude Trn2's fp8 e4m3 represents. Values at or below it have the
# same bit pattern in torch's float8_e4m3fn (whose max is 448), so bytes
# requantized into this range are stored as float8_e4m3fn and read by Trn2 as
# its own e4m3 unchanged.
TRN2_FP8_MAX = 240.0
FP8_DTYPE = torch.float8_e4m3fn


# =============================================================================
# Blockwise FP8 dequantization primitives (host-side, load time only)
# =============================================================================


def _scale_row_to_width(
    scale_row: torch.Tensor, width: int, block_k: int
) -> torch.Tensor:
    """Expand one scale-grid row to a per-column scale vector of length ``width``.

    ``repeat_interleave`` here runs on a ~32-element CPU tensor at load time
    (never traced), and the trailing ``[:width]`` handles a ceil-padded final
    column block.
    """
    return scale_row.reshape(-1).to(torch.float32).repeat_interleave(block_k)[:width]


def dequant_fp8_blockwise(
    w_slice,
    s_slice,
    row_range: tuple[int, int] | None = None,
    col_range: tuple[int, int] | None = None,
    block: tuple[int, int] = (FP8_BLOCK, FP8_BLOCK),
) -> torch.Tensor:
    """Dequantize a flat-grid blockwise-FP8 2-D tensor to fp32.

    Args:
        w_slice: checkpoint slice of the ``[R, C]`` fp8 weight.
        s_slice: checkpoint slice of the ``[ceil(R/bm), ceil(C/bk)]`` fp32 scale.
        row_range: optional ``[lo, hi)`` row window (e.g. a TP shard). Only the
            128-row blocks intersecting it are read; the result is trimmed to
            exactly ``hi - lo`` rows, so an unaligned window is still correct.
        col_range: optional ``[lo, hi)`` column window.
        block: ``(block_m, block_k)`` tile shape.

    Returns:
        fp32 tensor of shape ``[hi_r - lo_r, hi_c - lo_c]``.
    """
    R, C = tuple(w_slice.get_shape())
    r0, r1 = row_range if row_range is not None else (0, R)
    c0, c1 = col_range if col_range is not None else (0, C)
    block_m, block_k = block

    first_blk = r0 // block_m
    last_blk = (r1 - 1) // block_m

    parts = []
    for bi in range(first_blk, last_blk + 1):
        rr0 = bi * block_m
        rr1 = min(rr0 + block_m, R)
        w = w_slice[rr0:rr1, c0:c1].to(torch.float32)
        scale_full = _scale_row_to_width(s_slice[bi : bi + 1], C, block_k)
        parts.append(w * scale_full[c0:c1].view(1, -1))

    out = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
    offset = first_blk * block_m
    return out[r0 - offset : r1 - offset]


def _dequant_section_rows(
    w_slice,
    s_slice,
    row_lo: int,
    row_hi: int,
    sec_row_start: int,
    sec_row_end: int,
    sec_blk_start: int,
    block_m: int = FP8_BLOCK,
    block_k: int = FP8_BLOCK,
) -> torch.Tensor:
    """Dequantize absolute rows ``[row_lo, row_hi)`` inside ONE fused-QKV section.

    The section spans absolute rows ``[sec_row_start, sec_row_end)`` and its
    scale grid starts at scale row ``sec_blk_start``. Block indices are relative
    to the section (that is exactly what "each section gets its own ceil-padded
    grid" means), so they must not be computed from the absolute row number.
    """
    first_blk = (row_lo - sec_row_start) // block_m
    last_blk = (row_hi - 1 - sec_row_start) // block_m

    C = int(w_slice.get_shape()[1])
    parts = []
    for bi in range(first_blk, last_blk + 1):
        rr0 = sec_row_start + bi * block_m
        rr1 = min(rr0 + block_m, sec_row_end)
        w = w_slice[rr0:rr1].to(torch.float32)
        scale_full = _scale_row_to_width(
            s_slice[sec_blk_start + bi : sec_blk_start + bi + 1], C, block_k
        )
        parts.append(w * scale_full.view(1, -1))

    out = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
    offset = sec_row_start + first_blk * block_m
    return out[row_lo - offset : row_hi - offset]


# =============================================================================
# Fused QKV (pre-sharded on disk, per-section ceil-padded scale grid)
# =============================================================================


def _qkv_section_geometry(disk_tp: int, nh: int, nkv: int, hd: int, vhd: int):
    """Per-disk-shard row / scale-block geometry of one fused-QKV tensor."""
    q_rows = (nh // disk_tp) * hd
    k_rows = (nkv // disk_tp) * hd
    v_rows = (nkv // disk_tp) * vhd
    q_blks = math.ceil(q_rows / FP8_BLOCK)
    k_blks = math.ceil(k_rows / FP8_BLOCK)
    v_blks = math.ceil(v_rows / FP8_BLOCK)
    return (
        q_rows,
        k_rows,
        v_rows,
        q_rows + k_rows + v_rows,
        q_blks,
        k_blks,
        v_blks,
        q_blks + k_blks + v_blks,
    )


def qkv_disk_tp_candidates(
    scale_rows: int, nh: int, nkv: int, hd: int, vhd: int
) -> list[int]:
    """Shard counts whose per-section scale grid totals ``scale_rows`` rows."""
    candidates = [d for d in (1, 2, 4, 8, 16, 32, 64) if nh % d == 0 and nkv % d == 0]
    return [
        d
        for d in candidates
        if d * _qkv_section_geometry(d, nh, nkv, hd, vhd)[7] == scale_rows
    ]


def infer_qkv_disk_tp(scale_rows: int, nh: int, nkv: int, hd: int, vhd: int) -> int:
    """Recover the checkpoint's fused-QKV shard count from its scale row count.

    The weight row count (``nh*hd + nkv*hd + nkv*vhd``) is independent of
    ``disk_tp``, but the ceil-padded per-section scale grid usually is: 4 shards
    of a MiMo full-attn layer (nkv=4) give 108 scale rows where 1 shard would
    give 106, so that layer pins the value.

    NOT every layer does. When every section is already 128-divisible the
    padding vanishes and the total is shard-count invariant — MiMo's SWA layers
    (nkv=8) give 116 rows for disk_tp in {1, 2, 4} alike, even though the row
    ORDER differs ([Q0 K0 V0][Q1 K1 V1]... vs one [Q|K|V]). Such a layer cannot
    be resolved from its own shapes; pass ``disk_tp`` explicitly (the checkpoint
    index's ``metadata.tp_size``, which :class:`MiMoV2Config` picks up as
    ``qkv_disk_tp``).
    """
    matches = qkv_disk_tp_candidates(scale_rows, nh, nkv, hd, vhd)
    if len(matches) != 1:
        raise ValueError(
            f"cannot infer fused-QKV disk shard count from scale_rows={scale_rows} "
            f"(nh={nh}, nkv={nkv}, head_dim={hd}, v_head_dim={vhd}); "
            f"candidates matching: {matches}. Pass disk_tp explicitly (see "
            f"MiMoV2Config.qkv_disk_tp / the checkpoint index's metadata.tp_size)."
        )
    return matches[0]


def fused_qkv_fp8_loader(
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    v_head_dim: int,
    num_shards: int,
    num_kv_replicas: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    disk_tp: int | None = None,
) -> SafetensorsWeightLoader:
    """Load one rank's slice of a pre-sharded blockwise-FP8 fused QKV weight.

    The framework's :func:`fused_qkv_weight_loader` cannot serve this tensor: it
    asserts three separate q/k/v checkpoint slices, while MiMo ships ONE fused
    tensor that is additionally pre-sharded across ``disk_tp`` ranks with a
    per-section scale grid.

    Args:
        num_attention_heads: global Q head count.
        num_key_value_heads: global KV head count.
        head_dim: Q/K head width (192).
        v_head_dim: V head width (128).
        num_shards: runtime TP degree (may differ from the checkpoint's).
        num_kv_replicas: how many consecutive ranks share one KV head. Set when
            ``num_shards >= num_key_value_heads`` so every rank holds exactly one
            KV head instead of a zero-width slice.
        dtype: runtime dtype of the returned parameter.
        disk_tp: the checkpoint's own shard count. REQUIRED whenever the scale
            grid is shard-count invariant (MiMo's SWA layers, nkv=8: 116 rows
            for disk_tp 1, 2 and 4 alike) — inference cannot distinguish those,
            and guessing wrong permutes the head rows. Read it from the
            checkpoint index (``metadata.tp_size``); ``None`` falls back to
            :func:`infer_qkv_disk_tp`, which raises on an ambiguous grid rather
            than picking a candidate.

    Returns:
        Loader producing ``[hidden, q_local*hd + kv_local*hd + kv_local*vhd]``
        (transposed relative to the checkpoint's row-major Linear layout, which
        is what :func:`vllm_neuron.functional.qkv_proj` consumes).
    """
    nh, nkv, hd, vhd = num_attention_heads, num_key_value_heads, head_dim, v_head_dim
    declared_disk_tp = disk_tp

    def transform(slices: list, rank: int) -> torch.Tensor:
        if len(slices) == 1:
            # BF16 (or otherwise unquantized) re-export: no scale grid exists,
            # so nothing can be inferred. Trust the declared value; default to
            # the canonical HF single unsharded [Q|K|V] tensor.
            w_slice, s_slice = slices[0], None
            disk_tp = declared_disk_tp if declared_disk_tp is not None else 1
        elif len(slices) == 2:
            w_slice, s_slice = slices
            scale_rows = int(s_slice.get_shape()[0])
            if declared_disk_tp is not None:
                # Cross-check rather than trust blindly: a declared value that
                # contradicts the grid means the checkpoint was re-sharded and
                # the index metadata went stale.
                allowed = qkv_disk_tp_candidates(scale_rows, nh, nkv, hd, vhd)
                if declared_disk_tp not in allowed:
                    raise ValueError(
                        f"declared disk_tp={declared_disk_tp} is inconsistent with "
                        f"the fused-QKV scale grid ({scale_rows} rows for nh={nh}, "
                        f"nkv={nkv}); grid admits {allowed}"
                    )
                disk_tp = declared_disk_tp
            else:
                disk_tp = infer_qkv_disk_tp(scale_rows, nh, nkv, hd, vhd)
        else:
            raise ValueError(
                f"fused_qkv_fp8_loader expects 1 (bf16) or 2 (fp8 + scale) "
                f"slices, got {len(slices)}"
            )

        (q_rows, k_rows, _v_rows, shard_rows, q_blks, k_blks, _v_blks, shard_blks) = (
            _qkv_section_geometry(disk_tp, nh, nkv, hd, vhd)
        )

        tp_rank = rank % num_shards
        q_per_rank = nh // num_shards
        kv_per_rank = max(nkv // num_shards, 1)

        # Which global heads does this rank own?
        q_head_lo = tp_rank * q_per_rank
        if num_kv_replicas > 1:
            # Every rank holds ONE replicated KV head; consecutive groups of
            # num_kv_replicas ranks share it (matches the runtime geometry in
            # MiMoV2Attention.__init__).
            kv_head_lo = tp_rank // num_kv_replicas
            kv_per_rank = 1
        else:
            kv_head_lo = tp_rank * kv_per_rank

        q_per_disk_shard = nh // disk_tp
        kv_per_disk_shard = nkv // disk_tp

        def _read(section: str, head_idx: int) -> torch.Tensor:
            """Dequantize one logical head's rows from whichever disk shard holds it."""
            if section == "q":
                ds = head_idx // q_per_disk_shard
                local = head_idx % q_per_disk_shard
                sec_start = ds * shard_rows
                width, sec_blk = hd, ds * shard_blks
            elif section == "k":
                ds = head_idx // kv_per_disk_shard
                local = head_idx % kv_per_disk_shard
                sec_start = ds * shard_rows + q_rows
                width, sec_blk = hd, ds * shard_blks + q_blks
            else:  # "v"
                ds = head_idx // kv_per_disk_shard
                local = head_idx % kv_per_disk_shard
                sec_start = ds * shard_rows + q_rows + k_rows
                width, sec_blk = vhd, ds * shard_blks + q_blks + k_blks

            sec_len = (
                q_per_disk_shard * hd
                if section == "q"
                else kv_per_disk_shard * (hd if section == "k" else vhd)
            )
            lo = sec_start + local * width
            hi = lo + width
            if s_slice is None:
                return w_slice[lo:hi].to(torch.float32)
            return _dequant_section_rows(
                w_slice,
                s_slice,
                row_lo=lo,
                row_hi=hi,
                sec_row_start=sec_start,
                sec_row_end=sec_start + sec_len,
                sec_blk_start=sec_blk,
            )

        rows = []
        for h in range(q_head_lo, q_head_lo + q_per_rank):
            rows.append(_read("q", h))
        for j in range(kv_head_lo, kv_head_lo + kv_per_rank):
            rows.append(_read("k", j))
        for j in range(kv_head_lo, kv_head_lo + kv_per_rank):
            rows.append(_read("v", j))

        fused = torch.cat(rows, dim=0)  # [fused_qkv_dim, hidden]
        return fused.T.contiguous().to(dtype)  # [hidden, fused_qkv_dim]

    return SafetensorsWeightLoader(transform=transform)


# =============================================================================
# Dense MLP (layer 0) — flat scale grid, TP-sharded on the intermediate dim
# =============================================================================


def dense_gate_up_fp8_loader(
    shard_size: int, num_shards: int, dtype: torch.dtype = torch.bfloat16
) -> SafetensorsWeightLoader:
    """HF ``[I, H]`` fp8 -> our ``[H, I/TP]`` bf16 (gate_proj / up_proj)."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        tp_rank = rank % num_shards
        lo = tp_rank * shard_size
        rows = (lo, lo + shard_size)
        if len(slices) == 1:
            w = slices[0][rows[0] : rows[1], :]
        else:
            w = dequant_fp8_blockwise(slices[0], slices[1], row_range=rows)
        return w.T.contiguous().to(dtype)

    return SafetensorsWeightLoader(transform=transform)


def dense_down_fp8_loader(
    shard_size: int, num_shards: int, dtype: torch.dtype = torch.bfloat16
) -> SafetensorsWeightLoader:
    """HF ``[H, I]`` fp8 -> our ``[I/TP, H]`` bf16 (down_proj, row-parallel)."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        tp_rank = rank % num_shards
        lo = tp_rank * shard_size
        cols = (lo, lo + shard_size)
        if len(slices) == 1:
            w = slices[0][:, cols[0] : cols[1]]
        else:
            w = dequant_fp8_blockwise(slices[0], slices[1], col_range=cols)
        return w.T.contiguous().to(dtype)

    return SafetensorsWeightLoader(transform=transform)


# =============================================================================
# MoE experts — per-expert keys, flat scale grid
# =============================================================================
# MiMo stores experts as SEPARATE keys per expert
# MXFP4 (MiMo-V2.6, ``store_dtype: mxfp4``): OCP microscaling FP4. Each byte
# packs two e2m1 values, the even column in the low nibble; one e8m0 scale
# (``2 ** (s - 127)``) covers 32 consecutive columns of a row. Weight
# ``[R, C // 2]`` uint8, scale ``[R, C // 32]`` uint8 -- vLLM's Mxfp4MoEMethod
# layout.
MXFP4_BLOCK = 32
# The e2m1 values by nibble. Kept as a list and turned into a tensor per call:
# this module can be imported while the default device is ``meta``, and a
# module-level tensor would then be a meta tensor.
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def dequant_mxfp4(
    w_slice,
    s_slice,
    row_range: tuple[int, int] | None = None,
    col_range: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Dequantize an MXFP4 ``[R, C]`` weight (packed ``[R, C/2]``) to fp32."""
    R, half_c = tuple(w_slice.get_shape())
    C = half_c * 2
    r0, r1 = row_range if row_range is not None else (0, R)
    c0, c1 = col_range if col_range is not None else (0, C)
    if c0 % MXFP4_BLOCK or c1 % MXFP4_BLOCK:
        raise ValueError(f"MXFP4 column window [{c0}, {c1}) must be {MXFP4_BLOCK}-aligned")
    # Explicitly on the host: the loaders run while the model's default device
    # is ``meta``, and any tensor a factory makes here would land there.
    with torch.device("cpu"):
        packed = w_slice[r0:r1, c0 // 2 : c1 // 2].to("cpu", torch.uint8)
        table = torch.tensor(_E2M1, dtype=torch.float32, device="cpu")
        lo = table[(packed & 0x0F).long()]
        hi = table[(packed >> 4).long()]
        vals = torch.stack((lo, hi), dim=-1).reshape(r1 - r0, c1 - c0)
        exp = s_slice[r0:r1, c0 // MXFP4_BLOCK : c1 // MXFP4_BLOCK].to("cpu", torch.int32)
        scale = torch.exp2((exp - 127).to(torch.float32))
        return vals * scale.repeat_interleave(MXFP4_BLOCK, dim=1)


def _slice_dtype(w_slice) -> str:
    get = getattr(w_slice, "get_dtype", None)
    if callable(get):
        return str(get())
    return str(w_slice[0:1, 0:1].dtype)


def dequant_expert_weight(
    w_slice,
    s_slice,
    row_range: tuple[int, int] | None = None,
    col_range: tuple[int, int] | None = None,
) -> torch.Tensor:
    """One routed-expert matrix to fp32, from either on-disk format.

    MXFP4 (packed ``uint8``, V2.6) or 128x128 blockwise FP8 (V2.5, V2.5-Pro);
    the weight's dtype tells them apart.
    """
    if _slice_dtype(w_slice) in ("U8", "torch.uint8", "uint8"):
        return dequant_mxfp4(w_slice, s_slice, row_range=row_range, col_range=col_range)
    return dequant_fp8_blockwise(w_slice, s_slice, row_range=row_range, col_range=col_range)


# (``mlp.experts.{e}.{gate,up,down}_proj.weight`` + ``.weight_scale_inv``), so
# the parameter's checkpoint mapping is a flat INTERLEAVED list and the EP
# wrapper is :func:`expert_parallel_interleaved_loader`. That wrapper slices the
# list down to this rank's experts before calling the transform, so ``E_local``
# is derived from the slice count here — never from ``n_routed_experts``.
#
# ``quantized`` is passed in from ``MiMoV2Config.is_fp8_checkpoint`` rather than
# sniffed from slice shapes: it decides the items-per-expert stride, and reading
# that wrong silently mis-groups every expert's tensors.


def _local_expert_count(n_slices: int, items_per_expert: int, kind: str) -> int:
    if n_slices % items_per_expert != 0:
        raise ValueError(
            f"{kind} loader expects len(slices) divisible by {items_per_expert}, "
            f"got {n_slices}"
        )
    return n_slices // items_per_expert


def expert_gate_up_fp8_loader(
    shard_size: int,
    num_shards: int,
    quantized: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> SafetensorsWeightLoader:
    """Fuse + dequantize per-expert gate/up into ``[E_L, H, 2*I/TP]``.

    Args:
        shard_size: ``2 * intermediate_size_per_rank``. The kernel reshapes the
            result to ``[E_L, H, 2, I/TP]``, so gate must occupy the first half
            of the last dim.
        num_shards: MoE tensor-parallel degree.
        quantized: when True ``slices`` is ``[gate_w, gate_scale, up_w,
            up_scale, ...]`` (4 items/expert); when False ``[gate_w, up_w, ...]``.
        dtype: runtime dtype of the returned parameter.
    """
    items = 4 if quantized else 2

    def transform(slices: list, rank: int) -> torch.Tensor:
        n_experts = _local_expert_count(len(slices), items, "expert_gate_up")

        tp_rank = rank % num_shards
        half = shard_size // 2  # intermediate_size_per_rank
        rows = (tp_rank * half, tp_rank * half + half)

        per_expert = []
        for e in range(n_experts):
            base = e * items
            if quantized:
                gate = dequant_expert_weight(
                    slices[base], slices[base + 1], row_range=rows
                )
                up = dequant_expert_weight(
                    slices[base + 2], slices[base + 3], row_range=rows
                )
            else:
                gate = slices[base][rows[0] : rows[1], :]
                up = slices[base + 1][rows[0] : rows[1], :]
            fused = torch.cat([gate, up], dim=0)  # [2*I/TP, H]
            per_expert.append(fused.T.contiguous())  # [H, 2*I/TP]

        return torch.stack(per_expert, dim=0).to(dtype)  # [E_L, H, 2*I/TP]

    return SafetensorsWeightLoader(transform=transform)


def expert_down_fp8_loader(
    shard_size: int,
    num_shards: int,
    quantized: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> SafetensorsWeightLoader:
    """Dequantize per-expert down_proj into ``[E_L, I/TP, H]``.

    Args:
        shard_size: ``intermediate_size_per_rank``.
        num_shards: MoE tensor-parallel degree.
        quantized: when True ``slices`` is ``[down_w, down_scale, ...]``
            (2 items/expert); when False ``[down_w, ...]``.
        dtype: runtime dtype of the returned parameter.
    """
    items = 2 if quantized else 1

    def transform(slices: list, rank: int) -> torch.Tensor:
        n_experts = _local_expert_count(len(slices), items, "expert_down")

        tp_rank = rank % num_shards
        cols = (tp_rank * shard_size, tp_rank * shard_size + shard_size)

        per_expert = []
        for e in range(n_experts):
            base = e * items
            if quantized:
                w = dequant_expert_weight(
                    slices[base], slices[base + 1], col_range=cols
                )  # [H, I/TP]
            else:
                w = slices[base][:, cols[0] : cols[1]]
            per_expert.append(w.T.contiguous())  # [I/TP, H]

        return torch.stack(per_expert, dim=0).to(dtype)  # [E_L, I/TP, H]

    return SafetensorsWeightLoader(transform=transform)


# =============================================================================
# MoE experts, fp8-resident mode — per-output-channel requantization
# =============================================================================
# The 128x128 block scales cannot be used as they are: a block scale varies
# along BOTH matmul dims, so it cannot be applied after the matmul, and the
# checkpoint's e4m3fn bytes go up to 448, past Trn2's 240. One fp32 scale per
# output channel CAN be pulled out of the contraction
# (``x @ (q * s[None, :]) == (x @ q) * s``), so the forward converts q to bf16,
# runs the matmul, and multiplies by ``s`` afterwards.
#
# Each expert is dequantized and requantized on its own, so the fp32 peak is
# one expert's gate+up (~100 MB for Pro), not the layer's.


def requant_rows_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``[out, in]`` -> (fp8 ``[out, in]``, fp32 scale ``[out]``).

    ``scale = rowmax|w| / 240`` per output row, then ``w / scale`` is clamped to
    ``[-240, 240]`` and cast to ``float8_e4m3fn``. An all-zero row gets a tiny
    floor scale (and zero bytes) instead of a division by zero.
    """
    w = w.to(torch.float32)
    amax = w.abs().amax(dim=-1).clamp(min=1e-12)
    scale = amax / TRN2_FP8_MAX
    q = (w / scale.unsqueeze(-1)).clamp(-TRN2_FP8_MAX, TRN2_FP8_MAX)
    return q.to(FP8_DTYPE), scale


def _fp8_transpose(q: torch.Tensor) -> torch.Tensor:
    """Transpose an fp8 matrix through a uint8 view (no fp8 CPU kernels)."""
    return q.view(torch.uint8).T.contiguous()


class ExpertRequantMemo:
    """Compute-once cache shared by an fp8 expert weight and its scale.

    ``gate_up_proj_weight`` and ``gate_up_proj_scale`` map to the same
    checkpoint keys and both come out of one requantization. Their loaders run
    on the checkpoint loader's thread pool, possibly at the same time, so the
    first ``take`` for a key computes both parts under that key's lock and each
    part is handed out once (the slot is dropped when both are taken, so a
    second ``load_weights`` recomputes rather than reusing stale tensors).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._slots: dict = {}

    def take(self, key, part: str, compute):
        with self._lock:
            slot = self._slots.setdefault(key, {"lock": threading.Lock(), "parts": {}})
        with slot["lock"]:
            if part not in slot["parts"]:
                slot["parts"] = compute()
            value = slot["parts"].pop(part)
            if not slot["parts"]:
                with self._lock:
                    self._slots.pop(key, None)
        return value


def _requant_gate_up(
    slices: list, rank: int, shard_size: int, num_shards: int, quantized: bool
) -> dict:
    items = 4 if quantized else 2
    n_experts = _local_expert_count(len(slices), items, "expert_gate_up_requant")
    tp_rank = rank % num_shards
    half = shard_size // 2
    rows = (tp_rank * half, tp_rank * half + half)

    weights, scales = [], []
    for e in range(n_experts):
        base = e * items
        if quantized:
            gate = dequant_expert_weight(slices[base], slices[base + 1], row_range=rows)
            up = dequant_expert_weight(
                slices[base + 2], slices[base + 3], row_range=rows
            )
        else:
            gate = slices[base][rows[0] : rows[1], :].to(torch.float32)
            up = slices[base + 1][rows[0] : rows[1], :].to(torch.float32)
        # Same order as the bf16 loader: gate is the first half of the 2I dim.
        q, s = requant_rows_fp8(torch.cat([gate, up], dim=0))  # [2I/TP, H]
        del gate, up
        weights.append(_fp8_transpose(q))  # [H, 2I/TP] uint8
        scales.append(s)
    return {
        "weight": torch.stack(weights, dim=0).view(FP8_DTYPE),  # [E_L, H, 2I/TP]
        "scale": torch.stack(scales, dim=0).unsqueeze(1),  # [E_L, 1, 2I/TP]
    }


def _requant_down(
    slices: list, rank: int, shard_size: int, num_shards: int, quantized: bool
) -> dict:
    items = 2 if quantized else 1
    n_experts = _local_expert_count(len(slices), items, "expert_down_requant")
    tp_rank = rank % num_shards
    cols = (tp_rank * shard_size, tp_rank * shard_size + shard_size)

    weights, scales = [], []
    for e in range(n_experts):
        base = e * items
        if quantized:
            w = dequant_expert_weight(slices[base], slices[base + 1], col_range=cols)
        else:
            w = slices[base][:, cols[0] : cols[1]].to(torch.float32)
        # [H, I/TP]: output channel = H. Under moe TP the scale is this shard's
        # rowmax, which is exact: every rank scales its own partial sum.
        q, s = requant_rows_fp8(w)
        del w
        weights.append(_fp8_transpose(q))  # [I/TP, H] uint8
        scales.append(s)
    return {
        "weight": torch.stack(weights, dim=0).view(FP8_DTYPE),  # [E_L, I/TP, H]
        "scale": torch.stack(scales, dim=0).unsqueeze(1),  # [E_L, 1, H]
    }


def expert_gate_up_requant_loader(
    shard_size: int,
    num_shards: int,
    part: str,
    memo: ExpertRequantMemo,
    quantized: bool = True,
) -> SafetensorsWeightLoader:
    """Per-expert gate/up -> fp8 ``[E_L, H, 2I/TP]`` or fp32 scale ``[E_L, 1, 2I/TP]``.

    ``part`` is ``"weight"`` or ``"scale"``; the two loaders of one layer share
    ``memo``, so the requantization runs once. Slice conventions (interleaved
    per-expert list, ``quantized`` stride) are those of
    :func:`expert_gate_up_fp8_loader`; a BF16 re-export is requantized too.
    """
    if part not in ("weight", "scale"):
        raise ValueError(f"part must be 'weight' or 'scale', got {part!r}")

    def transform(slices: list, rank: int) -> torch.Tensor:
        return memo.take(
            ("gate_up", rank),
            part,
            lambda: _requant_gate_up(slices, rank, shard_size, num_shards, quantized),
        )

    return SafetensorsWeightLoader(transform=transform)


def expert_down_requant_loader(
    shard_size: int,
    num_shards: int,
    part: str,
    memo: ExpertRequantMemo,
    quantized: bool = True,
) -> SafetensorsWeightLoader:
    """Per-expert down -> fp8 ``[E_L, I/TP, H]`` or fp32 scale ``[E_L, 1, H]``."""
    if part not in ("weight", "scale"):
        raise ValueError(f"part must be 'weight' or 'scale', got {part!r}")

    def transform(slices: list, rank: int) -> torch.Tensor:
        return memo.take(
            ("down", rank),
            part,
            lambda: _requant_down(slices, rank, shard_size, num_shards, quantized),
        )

    return SafetensorsWeightLoader(transform=transform)
