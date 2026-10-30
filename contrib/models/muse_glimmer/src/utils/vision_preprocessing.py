# SPDX-License-Identifier: Apache-2.0
"""CPU preprocessing for the MuseGlimmer perception encoder.

Everything here is pure CPU tensor math on ``grid_thw``; the compiled vision
graph only sees fixed-shape packed tensors. The functions are vendored (not
imported from ``transformers.vision_utils``) because that module changed
between transformers 5.15.0 (DLAMI) and 5.20 (the position-table helper only
gained ``padding="zeros"`` after 5.15.0). The math below reproduces both HF
versions:

* position table: bilinear, ``align_corners=False``, zero padding
  (``F.grid_sample`` semantics; 5.15.0 ``get_vision_bilinear_indices_and_weights``
  == 5.20 ``get_vision_interpolation_indices_and_weights(..., "bilinear", False,
  1, "zeros")``), fp32 weights.
* 2D axial RoPE: positions ``(w + 1, h + 1)`` per un-merged patch, 24 inverse
  frequencies ``theta ** -(2i / 48)``, angle layout ``[w, h, w, h]`` over the
  96-dim head.
* windows: 32x32-patch windows in un-merged patch space, row-major, partial
  windows at the right / bottom edges, never crossing frames.

Pack order
----------
Inside its block, each frame of an item is laid out
``window (row-major) -> 2x2 merge block (row-major) -> in-block (row-major)``.
Window edges are multiples of 32 (even), so every 2x2 block lies inside one
window. Consequences (verified against the HF helpers in
the module tests):

1. every HF attention window is a contiguous run -> window attention is a
   plain ``[start, end)`` bound per token;
2. the HF pixel shuffle becomes a reshape:
   ``view(nb, bs/4, 4, D).transpose(-1, -2).reshape(nb, bs/4, 4D)``;
3. the merged rows come out window-major; ``merged_order`` (a per-block gather
   index) restores the HF order (2x2-block raster per frame) before the cache
   write.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .vision_block_packing import BlockAssignment

WINDOW_PATCHES = 32  # pos_emb_height * patch_size / patch_size (448 px)


# ---------------------------------------------------------------------------
# Per-frame helpers (one (h, w) patch grid)
# ---------------------------------------------------------------------------


def frame_pack_order(h: int, w: int, win: int = WINDOW_PATCHES, merge: int = 2) -> torch.Tensor:
    """Raster indices of an ``h x w`` frame in pack order (window -> 2x2 block -> in-block)."""
    if h % merge or w % merge:
        raise ValueError(f"patch grid {h}x{w} must be divisible by merge size {merge}")
    if win % merge:
        raise ValueError(f"window {win} must be divisible by merge size {merge}")
    r = torch.arange(h).view(h, 1).expand(h, w)
    c = torch.arange(w).view(1, w).expand(h, w)
    nwx = math.ceil(w / win)
    bh, bw = h // merge, w // merge
    key = (r // win) * nwx + (c // win)                 # window id (row-major)
    key = key * bh + (r // merge)                        # block row
    key = key * bw + (c // merge)                        # block col
    key = key * merge + (r % merge)                      # in-block row
    key = key * merge + (c % merge)                      # in-block col
    return torch.argsort(key.reshape(-1))


def frame_window_bounds(h: int, w: int, win: int = WINDOW_PATCHES) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token ``[start, end)`` of its window, in pack-order coordinates of one frame."""
    starts, ends = [], []
    off = 0
    for wy in range(0, h, win):
        for wx in range(0, w, win):
            n = min(win, h - wy) * min(win, w - wx)
            starts.append(torch.full((n,), off, dtype=torch.int32))
            ends.append(torch.full((n,), off + n, dtype=torch.int32))
            off += n
    return torch.cat(starts), torch.cat(ends)


def frame_merged_order(h: int, w: int, order: torch.Tensor, merge: int = 2) -> torch.Tensor:
    """Gather index: ``hf_merged[j] = packed_merged[merged_order[j]]`` for one frame.

    HF merged order is 2x2-block raster: rank ``(r // m) * (w // m) + (c // m)``.
    """
    top_left = order.view(-1, merge * merge)[:, 0]
    r, c = top_left // w, top_left % w
    hf_rank = (r // merge) * (w // merge) + (c // merge)
    inv = torch.empty_like(hf_rank)
    inv[hf_rank] = torch.arange(hf_rank.numel())
    return inv


def frame_rope_cos_sin(
    h: int, w: int, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Axial 2D RoPE cos/sin ``[h*w, head_dim]`` (fp32) in raster order.

    HF: ``position_ids = (h, w).flip(-1) + 1``; ``inv_freq`` over ``head_dim // 2``;
    ``freq = cat([w*f, h*f, w*f, h*f])``.
    """
    spatial_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
    hh = torch.arange(h).view(h, 1).expand(h, w).reshape(-1)
    ww = torch.arange(w).view(1, w).expand(h, w).reshape(-1)
    fw = (ww + 1).float()[:, None] * inv_freq[None, :]
    fh = (hh + 1).float()[:, None] * inv_freq[None, :]
    freq = torch.cat([fw, fh, fw, fh], dim=-1)
    return freq.cos(), freq.sin()


def frame_bilinear_taps(h: int, w: int, side: int) -> tuple[torch.Tensor, torch.Tensor]:
    """4-tap bilinear indices / fp32 weights ``[4, h*w]`` into the ``side x side`` table.

    ``align_corners=False`` and zero padding: out-of-range taps are clamped and
    get weight 0. Tap order (floor,floor), (floor,ceil), (ceil,floor), (ceil,ceil),
    as HF 5.15.0 ``get_vision_bilinear_indices_and_weights``.
    """

    def axis(n: int):
        g = (torch.arange(n).float() + 0.5) * (side / n) - 0.5
        lo = torch.floor(g).long()
        hi = lo + 1
        frac = g - lo.float()
        lo_ok = (lo >= 0) & (lo <= side - 1)
        hi_ok = (hi >= 0) & (hi <= side - 1)
        return lo.clamp(0, side - 1), hi.clamp(0, side - 1), frac, lo_ok, hi_ok

    hl, hc, hf, hl_ok, hc_ok = axis(h)
    wl, wc, wf, wl_ok, wc_ok = axis(w)
    idx = [
        (hl[:, None] * side + wl[None, :]),
        (hl[:, None] * side + wc[None, :]),
        (hc[:, None] * side + wl[None, :]),
        (hc[:, None] * side + wc[None, :]),
    ]
    wt = [
        (1 - hf)[:, None] * (1 - wf)[None, :] * (hl_ok[:, None] & wl_ok[None, :]),
        (1 - hf)[:, None] * wf[None, :] * (hl_ok[:, None] & wc_ok[None, :]),
        hf[:, None] * (1 - wf)[None, :] * (hc_ok[:, None] & wl_ok[None, :]),
        hf[:, None] * wf[None, :] * (hc_ok[:, None] & wc_ok[None, :]),
    ]
    return (
        torch.stack([i.reshape(-1) for i in idx]).to(torch.int32),
        torch.stack([x.reshape(-1) for x in wt]).float(),
    )


# ---------------------------------------------------------------------------
# Block packing of a batch of items
# ---------------------------------------------------------------------------


@dataclass
class PackedVisionInputs:
    pixel_values: torch.Tensor  # [nb, bs, patch_dim]
    pos_emb_idx: torch.Tensor  # [4, nb, bs] int32
    pos_emb_weight: torch.Tensor  # [4, nb, bs] fp32
    cos: torch.Tensor  # [nb, bs, head_dim] fp32
    sin: torch.Tensor  # [nb, bs, head_dim] fp32
    bound_min: torch.Tensor  # [nb, bs, 1] int32 (full attention: per temporal group)
    bound_max: torch.Tensor
    window_bound_min: torch.Tensor  # [nb, bs, 1] int32 (window attention)
    window_bound_max: torch.Tensor
    merged_order: torch.Tensor  # [nb, bs // merge**2] int64, block-local gather index
    # window-chunk layout (only with ``win_chunks``): window layers attend in
    # ``K = win_chunks`` sub-blocks of ``C = win_chunk`` rows per block.
    win_gather: torch.Tensor | None = None  # [nb, K*C] int64: chunk slot -> block-local row
    win_scatter: torch.Tensor | None = None  # [nb, bs] int64: block-local row -> chunk slot
    win_bound_min: torch.Tensor | None = None  # [nb*K, C, 1] int32, chunk-local
    win_bound_max: torch.Tensor | None = None

    def as_kwargs(self) -> dict[str, torch.Tensor]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ---------------------------------------------------------------------------
# window-chunk planning
# ---------------------------------------------------------------------------


class WindowChunkOverflow(ValueError):
    """A block's windows need more ``win_chunk``-row chunks than the graph has."""


def frame_window_sizes(h: int, w: int, win: int = WINDOW_PATCHES) -> list[int]:
    """Token count of each window of one frame, in pack order (row-major windows)."""
    return [min(win, h - wy) * min(win, w - wx) for wy in range(0, h, win) for wx in range(0, w, win)]


def ffd_chunks(sizes: list[int], chunk: int) -> list[list[int]]:
    """First-fit-decreasing of window indices into ``chunk``-row bins (each size <= chunk)."""
    order = sorted(range(len(sizes)), key=lambda i: (-sizes[i], i))
    bins: list[list[int]] = []
    room: list[int] = []
    for i in order:
        for b in range(len(bins)):
            if room[b] >= sizes[i]:
                bins[b].append(i)
                room[b] -= sizes[i]
                break
        else:
            bins.append([i])
            room.append(chunk - sizes[i])
    return bins


def window_chunks_needed(grid_rows: list[tuple[int, int, int]], chunk: int, win: int = WINDOW_PATCHES) -> int:
    """Chunks one block needs for these items (all frames of all rows in the block)."""
    sizes: list[int] = []
    for t, h, w in grid_rows:
        sizes += frame_window_sizes(h, w, win) * t
    return len(ffd_chunks(sizes, chunk))


_MAX_CHUNKS: dict[tuple[int, int, int, int], int] = {}


def max_window_chunks(rows: int, chunk: int, win: int = WINDOW_PATCHES, merge: int = 2) -> int:
    """Worst-case chunks for any content one block of ``rows`` can hold.

    A block holds one image, or ``k`` same-size temporal groups of one video
    (``one_item_per_block``), so this enumerates every even ``h x w`` grid with
    ``h*w <= rows`` and every ``k <= rows // (h*w)``. Cached per geometry.
    """
    key = (rows, chunk, win, merge)
    if key in _MAX_CHUNKS:
        return _MAX_CHUNKS[key]
    worst = 1
    for h in range(merge, rows + 1, merge):
        for w in range(merge, rows // h + 1, merge):
            sizes = frame_window_sizes(h, w, win)
            kmax = rows // (h * w)
            if len(sizes) == 1:  # one window per frame: bins = ceil(k / per-chunk)
                need = -(-kmax // (chunk // sizes[0]))
            else:
                need = max(len(ffd_chunks(sizes * k, chunk)) for k in range(1, kmax + 1))
            worst = max(worst, need)
    _MAX_CHUNKS[key] = worst
    return worst


def pack_vision_inputs(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    assignment: BlockAssignment,
    *,
    head_dim: int,
    rope_theta: float,
    num_grid_per_side: int,
    merge_size: int = 2,
    window: int = WINDOW_PATCHES,
    rope_dtype: torch.dtype | None = None,
    win_chunks: int = 0,
    win_chunk: int = 1024,
) -> PackedVisionInputs:
    """Pack items (rows of ``grid_thw``, raster-ordered ``pixel_values``) into blocks.

    Items go into their assigned blocks in item-index order (as
    ``scatter_to_blocks``). Pad rows: zero pixels / taps / cos-sin, bounds
    ``(0, 0)`` (attend to nothing), identity ``merged_order``.

    ``rope_dtype``: round cos/sin through this dtype (HF casts them to the
    hidden-state dtype before the fp32 rotation), then keep fp32.

    ``win_chunks`` (``MUSE_GLIMMER_VIS_WIN_SPLIT``): also build the
    window-chunk layout. Each block's windows are FFD-packed into ``win_chunks``
    chunks of ``win_chunk`` rows; ``win_gather`` lists, per chunk slot, the
    block-local row it holds (pad slots point at a pad row or row 0 and get
    bounds ``(0, 0)``), and ``win_scatter`` maps each block row back to its slot.
    The block layout itself (and so every non-window op) is unchanged.
    """
    nb, bs = assignment.num_blocks, assignment.block_size
    mu = merge_size * merge_size
    if bs % mu:
        raise ValueError(f"block_size {bs} must be divisible by {mu}")
    patch_dim = pixel_values.shape[-1]

    pix = pixel_values.new_zeros(nb, bs, patch_dim)
    tap_idx = torch.zeros(4, nb, bs, dtype=torch.int32)
    tap_w = torch.zeros(4, nb, bs, dtype=torch.float32)
    cos = torch.zeros(nb, bs, head_dim, dtype=torch.float32)
    sin = torch.zeros(nb, bs, head_dim, dtype=torch.float32)
    fmin = torch.zeros(nb, bs, 1, dtype=torch.int32)
    fmax = torch.zeros(nb, bs, 1, dtype=torch.int32)
    wmin = torch.zeros(nb, bs, 1, dtype=torch.int32)
    wmax = torch.zeros(nb, bs, 1, dtype=torch.int32)
    morder = torch.arange(bs // mu, dtype=torch.int64).repeat(nb, 1)

    rows = [tuple(int(v) for v in r) for r in grid_thw.tolist()]
    tokens = [t * h * w for t, h, w in rows]
    src_off = [0]
    for n in tokens:
        src_off.append(src_off[-1] + n)

    cache: dict[tuple[int, int], tuple] = {}

    def frame_tables(h: int, w: int):
        if (h, w) not in cache:
            order = frame_pack_order(h, w, window, merge_size)
            ws, we = frame_window_bounds(h, w, window)
            mo = frame_merged_order(h, w, order, merge_size)
            c, s = frame_rope_cos_sin(h, w, head_dim, rope_theta)
            if rope_dtype is not None:
                c, s = c.to(rope_dtype).float(), s.to(rope_dtype).float()
            ti, tw = frame_bilinear_taps(h, w, num_grid_per_side)
            cache[(h, w)] = (order, ws, we, mo, c[order], s[order], ti[:, order], tw[:, order])
        return cache[(h, w)]

    for b, items in enumerate(assignment.bins):
        off = 0
        for i in sorted(items):
            t, h, w = rows[i]
            order, ws, we, mo, c, s, ti, tw = frame_tables(h, w)
            n = h * w
            for f in range(t):
                src = pixel_values[src_off[i] + f * n : src_off[i] + (f + 1) * n]
                sl = slice(off, off + n)
                pix[b, sl] = src[order]
                tap_idx[:, b, sl] = ti
                tap_w[:, b, sl] = tw
                cos[b, sl] = c
                sin[b, sl] = s
                fmin[b, sl, 0] = off
                fmax[b, sl, 0] = off + n
                wmin[b, sl, 0] = ws + off
                wmax[b, sl, 0] = we + off
                morder[b, off // mu : (off + n) // mu] = mo + off // mu
                off += n
        if off > bs:
            raise ValueError(f"block {b} overflows: {off} > {bs}")

    win = {}
    if win_chunks:
        win = _window_chunk_layout(assignment, rows, wmin, wmax, win_chunks, win_chunk, bs, window)

    return PackedVisionInputs(
        pixel_values=pix,
        pos_emb_idx=tap_idx,
        pos_emb_weight=tap_w,
        cos=cos,
        sin=sin,
        bound_min=fmin,
        bound_max=fmax,
        window_bound_min=wmin,
        window_bound_max=wmax,
        merged_order=morder,
        **win,
    )


def _window_chunk_layout(assignment, rows, wmin, wmax, K, C, bs, window) -> dict[str, torch.Tensor]:
    """per block, FFD the windows into ``K`` chunks of ``C`` rows (see pack_vision_inputs)."""
    if C > bs or bs > K * C:
        raise ValueError(f"window chunks: need C <= bs <= K*C, got C={C} bs={bs} K={K}")
    nb = assignment.num_blocks
    gather = torch.zeros(nb, K * C, dtype=torch.int64)
    scatter = torch.zeros(nb, bs, dtype=torch.int64)
    bmin = torch.zeros(nb * K, C, 1, dtype=torch.int32)
    bmax = torch.zeros(nb * K, C, 1, dtype=torch.int32)
    for b, items in enumerate(assignment.bins):
        # windows of this block as (start row, size), in pack order
        wins: list[tuple[int, int]] = []
        off = 0
        for i in sorted(items):
            t, h, w = rows[i]
            for _ in range(t):
                for n in frame_window_sizes(h, w, window):
                    wins.append((off, n))
                    off += n
        bins = ffd_chunks([n for _, n in wins], C)
        if len(bins) > K:
            raise WindowChunkOverflow(f"block {b}: windows need {len(bins)} chunks of {C} rows > {K}")
        used = torch.zeros(K * C, dtype=torch.bool)
        covered = torch.zeros(bs, dtype=torch.bool)
        for k, members in enumerate(bins):
            o = 0
            for wi in sorted(members):
                start, n = wins[wi]
                slot = k * C + o
                gather[b, slot : slot + n] = torch.arange(start, start + n)
                scatter[b, start : start + n] = torch.arange(slot, slot + n)
                bmin[b * K + k, o : o + n, 0] = o
                bmax[b * K + k, o : o + n, 0] = o + n
                used[slot : slot + n] = True
                o += n
        for start, n in wins:
            covered[start : start + n] = True
        # block pad rows (no window): park them in free slots, bounds (0, 0)
        pad_rows = (~covered).nonzero().flatten()
        free = (~used).nonzero().flatten()[: pad_rows.numel()]
        gather[b, free] = pad_rows
        scatter[b, pad_rows] = free
        # sanity: same window bounds as the block layout, chunk-shifted
        assert torch.equal((wmax[b, :off, 0] - wmin[b, :off, 0]).long(),
                           (bmax.view(nb, K * C)[b, scatter[b, :off]] - bmin.view(nb, K * C)[b, scatter[b, :off]]).long())
    return dict(win_gather=gather, win_scatter=scatter, win_bound_min=bmin, win_bound_max=bmax)


# ---------------------------------------------------------------------------
# Video: one encoder item per temporal group
# ---------------------------------------------------------------------------


def plan_video_items(
    grid_thw: torch.Tensor, block_size: int, merge_size: int = 2
) -> tuple[torch.Tensor, list[int], list[list[int]]]:
    """Split videos into per-temporal-group encoder items.

    HF runs the ViT on a video with ``cu_seqlens`` = ``h*w`` repeated ``t`` times
    (``merge_temporal=False``) and windows that never cross a group, so a
    ``[t, h, w]`` video is exactly ``t`` independent ``[1, h, w]`` items with the
    same position taps / RoPE / pack order, concatenated in time order.

    Returns ``(grid_items [sum t, 3], group_ids, tokens_per_block)``:
    ``group_ids`` keeps a video's groups in their own block run (``ffd_pack_images``
    with ``one_item_per_block=True``), and ``tokens_per_block[v]`` lists the real
    merged tokens per cache block of video ``v`` (whole groups per block; a non-last
    block may end in a pad tail when groups do not tile the block).
    """
    grid_thw = grid_thw.to(torch.int64).reshape(-1, 3)
    mu = merge_size * merge_size
    tpb_all: list[list[int]] = []
    for t, h, w in grid_thw.tolist():
        per = h * w
        if per > block_size:
            raise ValueError(
                f"encoder_attention_block_size={block_size} < per-group patches {per} "
                f"(video grid {t}x{h}x{w}); a temporal group must fit in one block. "
                "Increase the block size or lower mm_processor_kwargs max_video_frame_tokens."
            )
        fpb = block_size // per
        tpb, left = [], t
        while left > 0:
            k = min(fpb, left)
            tpb.append(k * per // mu)
            left -= k
        tpb_all.append(tpb)
    frames = grid_thw[:, 0]
    grid_items = grid_thw.repeat_interleave(frames, dim=0)
    grid_items[:, 0] = 1
    group_ids = torch.repeat_interleave(torch.arange(grid_thw.shape[0]), frames).tolist()
    return grid_items, group_ids, tpb_all
