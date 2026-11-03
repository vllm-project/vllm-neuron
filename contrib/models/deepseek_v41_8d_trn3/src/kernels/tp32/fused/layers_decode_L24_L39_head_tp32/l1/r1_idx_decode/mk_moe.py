# SPDX-License-Identifier: Apache-2.0
"""Token-grouped routed experts for the megakernels ("real" MoE).

The router leaves `wexp[t, e]`, the routing weight of local expert `e` for row
`t` (0 where the row did not pick it). Instead of running every expert densely
over every row, each expert runs only over the rows that picked it:

  1. `build_expert_lists`: the row ids of each expert, compacted in ascending
     order, by one `nonzero_with_count` over the mask rows (GpSimd, up to 8
     experts per instruction), written to a small HBM list with the count.
  2. per expert, a `fori_loop` over `ceil(count / 128)` tiles on both cores (a runtime bound:
     an expert nobody picked costs nothing, not even its weight reads): gather
     the tile's rows (indirect DMA), stage them for the MX matmul, w1 / w3,
     SwiGLU x routing weight, stage, w2, and scatter-add the tile into the fp32
     accumulator with `dma_compute` (read-modify-write through the same row ids).

A tile's padding slots (beyond the expert's last row) point at distinct junk
rows past the real ones -- the gathered inputs there are zero rows, and
`dma_compute` wants unique indices -- so the accumulator is allocated with one
extra 128-row tile that is never read.
"""

import nki.isa as nisa
import nki.language as nl

from . import mk_mx as X
from .mk_mx import PT

#: `run_expert` keeps an expert's weight slices in SBUF across its row tiles (loaded
#: once per expert instead of re-streamed by every tile).
EXPERT_W_RESIDENT = False
#: (OFF: 112 KB of resident weights a partition slowed the whole kernel, attention
#: 3.57 -> 5.10 ms at in8192_bs1: SBUF allocation.)
#: `run_expert` scatter-adds each 512-column block of the w2 output into the
#: accumulator as soon as it is copied out of PSUM (overlapping the remaining w2
#: matmuls) instead of the whole row tile at the end. (OFF: slower, in8192_bs1
#: 10.19 -> 10.73 ms: ten serialized RMW DMAs of ~1.9 us GpSimd each per tile.)
SCATTER_PER_NTILE = False
#: `run_expert` runs an expert's row tiles two per loop iteration (`body_n`): each
#: weight group is streamed once for both tiles (a hot expert's tiles were
#: weight-DMA-bound, re-streaming ~9 MB of weights per tile).
EXPERT_TILE_PAIRS = True

#: With EXPERT_TILE_PAIRS: no pair loop when an expert has at most one row tile
#: (Tp = 128: T = 6 and T = 60; its trip count is always 0 there, but each dynamic loop
#: costs a branch on every engine, ~6 us an expert even when nobody picked it).
PAIRS_ONLY_IF_FIT = True
#: Every local expert's row count loaded by one DMA before the experts (each expert's
#: loop bound a `scalar_load` from SBUF) instead of one count DMA per expert on the
#: chain between the experts' loops.
COUNTS_ONCE = True

#: `body_n`'s scatter-add skips a tile's padding slots (distinct out-of-bounds row ids
#: past the accumulator, `oob_mode.skip`) instead of read-modify-writing 128 junk rows
#: (2.6 MB of RMW traffic a tile at dim 5120 for an expert with typically 1-3 rows).
SCATTER_SKIP_PAD = True

#: Mask rows an instruction compacts (one per GpSimd core: partitions 0, 16, .., 112).
LANES = 8

#: (dense, Tp = 128) ONE dynamic loop over the local experts some row picked (their ids
#: compacted ascending by `nonzero_with_count`, the expert's weights addressed by a device
#: index) instead of one loop per local expert: each dynamic loop costs ~8-9 us of
#: all-engine control even when nobody picked the expert (6 of them a block).
DENSE_ONE_LOOP = True
#: (DENSE_ONE_LOOP) each 512-column tile's SwiGLU and staging interleaved with the w3 GEMM
#: (the tail after an expert's last weight bytes was w3 -> the whole SwiGLU -> staging ->
#: w2, ~14 us of serial ACT / DVE / PE latency).
DENSE_TILE_SWIGLU = True
#: (fused launch, DENSE_TILE_SWIGLU) the dense experts' weight tiles allocated (their scale
#: tiles zeroed) once and every expert loaded into them in turn (`mk_mx.alloc_w_wide`): no
#: per-expert memsets ahead of the scale DMAs; a weight's next load waits only for the
#: matmuls reading it.
DENSE_W_HOIST = True
#: (fused launch) At Tp = 128 with at least this many rows, the dense routed experts run as
#: static code over EVERY local expert (`run_experts_dense_static`) instead of the dynamic
#: loop over the active ones: with T rows a local expert is active with probability
#: 1 - (1 - k / E)^T (~60% at T = 60: the slowest rank has nearly all active), and static
#: code lets each expert's weight stream overlap the previous expert's matmuls / SwiGLU
#: (no per-iteration loop control). An inactive expert's rows all add +-0 (`wexp` = 0),
#: which leaves the accumulator unchanged. 0: never.
DENSE_STATIC_MIN_T = 24


def alloc_lists(n_local, Tp):
    """Private HBM for the per-expert row lists: `[n_local, (ntmax + 1) * 128]` int32,
    row `e` = ids (ascending), -1 padding, and the count at column `Tp`."""
    ntmax = Tp // PT
    return nl.ndarray((n_local, (ntmax + 1) * PT), dtype=nl.int32, buffer=nl.private_hbm)


def build_expert_lists(wexp_hbm, Tp, n_local, glist):
    """Compact `wexp[0:Tp, e] > 0` into `glist` (see `alloc_lists`)."""
    nt = Tp // PT
    for e0 in range(0, n_local, LANES):
        ne = min(LANES, n_local - e0)
        maskT = nl.ndarray((PT, Tp), dtype=nl.float32, buffer=nl.sbuf)
        for i in range(nt):
            w = nl.ndarray((PT, n_local), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w, src=wexp_hbm[i * PT : (i + 1) * PT, 0:n_local])
            wide = nl.ndarray((PT, PT), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(wide, 0.0)
            for q in range(ne):
                nisa.tensor_scalar(dst=wide[:, 16 * q : 16 * q + 1], data=w[:, e0 + q : e0 + q + 1],
                                   op0=nl.greater, operand0=0.0)
            tp = nl.ndarray((PT, PT), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=wide)
            nisa.tensor_copy(dst=maskT[:, i * PT : (i + 1) * PT], src=tp)
        nz = nl.ndarray((PT, Tp + 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.nonzero_with_count(dst=nz, src=maskT, index_offset=0, padding_val=-1)
        for q in range(ne):
            nisa.dma_copy(dst=glist[e0 + q : e0 + q + 1, 0 : Tp + 1], src=nz[16 * q : 16 * q + 1, :])


def expert_tiles(glist, e, Tp):
    """`ceil(count / 128)` of expert `e` as a device scalar (a loop bound)."""
    c = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c, src=glist[e : e + 1, Tp : Tp + 1])
    cs = nisa.scalar_load(c)
    return (cs + (PT - 1)) // PT


def partition_ramp():
    """`(128, 1)` fp32 with partition `p` holding `p`."""
    ramp = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=ramp, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    rampf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rampf, src=ramp)
    return rampf


def zero_rows(hbm, rows, n, dtype=nl.float32):
    """`hbm[0:rows, 0:n] = 0`."""
    if rows <= 0:
        return
    z = nl.ndarray((PT, n), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(z, 0.0)
    for r in range(0, rows, PT):
        nr = min(PT, rows - r)
        nisa.dma_copy(dst=hbm[r : r + nr, 0:n], src=z[0:nr, :])


def _swiglu_chunk(inter):
    for cand in (1152, 768, 576, 512, 384, 256, 128, 64, 32):
        if inter % cand == 0:
            return cand
    return inter


def swiglu_tile(g16, u16, wcol, y16, inter, limit):
    """`y16 = bf16(silu(min(g, limit)) * clamp(u, -limit, limit) * w)` for one tile
    (`g16`/`u16` bf16 SBUF `[128, inter]`, `wcol` fp32 `(128, 1)` or None)."""
    chunk = _swiglu_chunk(inter)
    for j in range(inter // chunk):
        c0 = j * chunk
        g = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=g, data=g16[:, c0 : c0 + chunk], op0=nl.minimum, operand0=limit)
        u = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=u, data=u16[:, c0 : c0 + chunk], op0=nl.maximum, operand0=-limit,
                           op1=nl.minimum, operand1=limit)
        sg = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sg, op=nl.silu, data=g)
        # the last product writes bf16 directly (its fp32 result cast on output)
        if wcol is not None:
            y = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=y, data1=sg, data2=u, op=nl.multiply)
            nisa.tensor_scalar(dst=y16[:, c0 : c0 + chunk], data=y, op0=nl.multiply,
                               operand0=wcol[0:PT, 0:1])
        else:
            nisa.tensor_tensor(dst=y16[:, c0 : c0 + chunk], data1=sg, data2=u, op=nl.multiply)


def run_expert(ntiles, e, glist, Tp, x_hbm, wexp_hbm, n_local, w1, s1, w3, s3, w2, s2,
               dim, inter, lo, hi, limit, eacc_hbm, rampf):
    """Expert `e` over its `ntiles` (a device scalar) row tiles; see the module doc.

    Both cores run every expert with the same trip count (an LNC=2 dynamic loop must
    iterate the same count on both cores) and split each tile between them along
    the intermediate axis: this core computes `inter` columns `[lo, hi)` of w1 / w3,
    the SwiGLU on them, and w2 over that contraction slice, an fp32 partial that
    its own accumulator sums; the two accumulators are added afterwards.

    `x_hbm` is the bf16 `[Tp + 128, dim]` input with zero rows past `Tp`, `wexp_hbm`
    the `[Tp + 128, n_local]` fp32 routing weights (zero past `Tp`), `w*`/`s*` the
    expert's MX weights (`[K/4, N]` x4 views and `[K/32, N]` scales), `eacc_hbm` the
    `[Tp + 128, dim]` fp32 accumulator.
    """
    ntmax = Tp // PT
    g3 = glist.reshape((n_local, ntmax + 1, PT))
    span = hi - lo
    if EXPERT_W_RESIDENT:
        # the expert's weight slices stay in SBUF across its tiles: loaded by the first
        # tile (a loop of min(ntiles, 1) trips: an expert nobody picked loads nothing),
        # read by every tile
        w1t, w1s = X.alloc_w_full(dim, span, fp4=True)
        w3t, w3s = X.alloc_w_full(dim, span, fp4=True)
        w2t, w2s = X.alloc_w_full(span, dim, fp4=True)

    def body(i, t0=0, first=False):
        gi = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gi, src=g3.ap(pattern=[[1, PT], [1, 1]], offset=(e * (ntmax + 1) + t0) * PT,
                                       scalar_offset=i, indirect_dim=1))
        # padding (-1) -> Tp + 1 + p - 1 = Tp + p: a distinct zero row per slot
        gf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gf, src=gi)
        neg = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=neg, data=gf, op0=nl.less, operand0=0.0)
        fix = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=fix, data=rampf, op0=nl.add, operand0=float(Tp + 1))
        nisa.tensor_tensor(dst=fix, data1=fix, data2=neg, op=nl.multiply)
        nisa.tensor_tensor(dst=gf, data1=gf, data2=fix, op=nl.add)
        nisa.tensor_copy(dst=gi, src=gf)
        if EXPERT_W_RESIDENT and first:
            X.load_w_full(w1, s1, dim, span, w1t, w1s, n0=lo)
            X.load_w_full(w3, s3, dim, span, w3t, w3s, n0=lo)
            X.load_w_full(w2, s2, span, dim, w2t, w2s, krow0=lo // 4)
        # the tile's rows, quantized
        xq, xs = X.alloc_x(1, dim)
        X.stage_x_gathered(x_hbm, gi, dim, xq, xs, 0)
        g16 = nl.ndarray((PT, span), dtype=nl.bfloat16, buffer=nl.sbuf)
        u16 = nl.ndarray((PT, span), dtype=nl.bfloat16, buffer=nl.sbuf)
        if EXPERT_W_RESIDENT:
            X.gemm_sb_pre(xq, xs, dim, w1t, w1s, g16, span)
            X.gemm_sb_pre(xq, xs, dim, w3t, w3s, u16, span)
        else:
            X.gemm_sb(xq, xs, dim, w1, s1, g16, span, fp4=True, n0=lo)
            X.gemm_sb(xq, xs, dim, w3, s3, u16, span, fp4=True, n0=lo)
        wcol = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=wcol, src=wexp_hbm.ap(pattern=[[n_local, PT], [1, 1]], offset=e,
                                               vector_offset=gi[0:PT, 0:1], indirect_dim=0))
        y16 = nl.ndarray((PT, span), dtype=nl.bfloat16, buffer=nl.sbuf)
        swiglu_tile(g16, u16, wcol, y16, span, limit)
        yq, ys = X.alloc_x(1, span)
        X.stage_rows(y16, span, yq, ys, 0)
        o32 = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)

        def scatter_add(c0, w):
            tgt = eacc_hbm.ap(pattern=[[dim, PT], [1, w]], offset=c0, vector_offset=gi[0:PT, 0:1],
                              indirect_dim=0)
            nisa.dma_compute(dst=tgt, srcs=[tgt, o32[:, c0 : c0 + w]], reduce_op=nl.add)

        if EXPERT_W_RESIDENT:
            X.gemm_sb_pre(yq, ys, span, w2t, w2s, o32, dim)
        else:
            X.gemm_sb(yq, ys, span, w2, s2, o32, dim, fp4=True, krow0=lo // 4,
                      epilogue=scatter_add if SCATTER_PER_NTILE else None)
        if EXPERT_W_RESIDENT or not SCATTER_PER_NTILE:
            scatter_add(0, dim)

    def tile_ids(i, t, skip=False):
        # tile i + t's row ids; padding (-1) -> Tp + p: a distinct zero row per slot
        # (with `skip` also the scatter's ids: padding -> rows + p, out of bounds)
        gi = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gi, src=g3.ap(pattern=[[1, PT], [1, 1]], offset=(e * (ntmax + 1) + t) * PT,
                                       scalar_offset=i, indirect_dim=1))
        gf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gf, src=gi)
        neg = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=neg, data=gf, op0=nl.less, operand0=0.0)
        fix = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=fix, data=rampf, op0=nl.add, operand0=float(Tp + 1))
        nisa.tensor_tensor(dst=fix, data1=fix, data2=neg, op=nl.multiply)
        nisa.tensor_tensor(dst=gf, data1=gf, data2=fix, op=nl.add)
        nisa.tensor_copy(dst=gi, src=gf)
        if not skip:
            return gi
        # padding Tp + p -> rows + p (rows = the accumulator's Tp + 128)
        gs = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gs, data=neg, op0=nl.multiply, operand0=float(eacc_hbm.shape[0] - Tp),
                           op1=nl.add, operand1=gf[0:PT, 0:1])
        return gi, gs

    def body_n(i, nmt):
        # the `nmt` tiles i .. i + nmt - 1 (each tile's values exactly `body`'s): their
        # rows staged together, every weight group loaded once for all of them
        if SCATTER_SKIP_PAD:
            ids = [tile_ids(i, t, skip=True) for t in range(nmt)]
            gis = [a for a, _ in ids]
            gss = [b for _, b in ids]
        else:
            gis = [tile_ids(i, t) for t in range(nmt)]
        xq, xs = X.alloc_x(nmt, dim)
        for t in range(nmt):
            X.stage_x_gathered(x_hbm, gis[t], dim, xq, xs, t)
        g16 = [nl.ndarray((PT, span), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(nmt)]
        u16 = [nl.ndarray((PT, span), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(nmt)]

        def to_g(t, c0, w, acc):
            nisa.tensor_copy(dst=g16[t][:, c0 : c0 + w], src=acc)

        def to_u(t, c0, w, acc):
            nisa.tensor_copy(dst=u16[t][:, c0 : c0 + w], src=acc)

        X.gemm_sb_multi(xq, xs, nmt, dim, w1, s1, span, to_g, fp4=True, n0=lo)
        X.gemm_sb_multi(xq, xs, nmt, dim, w3, s3, span, to_u, fp4=True, n0=lo)
        yq, ys = X.alloc_x(nmt, span)
        for t in range(nmt):
            wcol = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=wcol, src=wexp_hbm.ap(pattern=[[n_local, PT], [1, 1]], offset=e,
                                                   vector_offset=gis[t][0:PT, 0:1], indirect_dim=0))
            y16 = nl.ndarray((PT, span), dtype=nl.bfloat16, buffer=nl.sbuf)
            swiglu_tile(g16[t], u16[t], wcol, y16, span, limit)
            X.stage_rows(y16, span, yq, ys, t)

        o32 = [nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf) for _ in range(nmt)]

        def to_o(t, c0, w, acc):
            nisa.tensor_copy(dst=o32[t][:, c0 : c0 + w], src=acc)

        X.gemm_sb_multi(yq, ys, nmt, span, w2, s2, dim, to_o, fp4=True, krow0=lo // 4)
        for t in range(nmt):
            if SCATTER_SKIP_PAD:
                tgt = eacc_hbm.ap(pattern=[[dim, PT], [1, dim]], vector_offset=gss[t][0:PT, 0:1],
                                  indirect_dim=0)
                nisa.dma_compute(dst=tgt, srcs=[tgt, o32[t]], reduce_op=nl.add,
                                 oob_mode=nisa.oob_mode.skip)
                continue
            tgt = eacc_hbm.ap(pattern=[[dim, PT], [1, dim]], vector_offset=gis[t][0:PT, 0:1],
                              indirect_dim=0)
            nisa.dma_compute(dst=tgt, srcs=[tgt, o32[t]], reduce_op=nl.add)

    if EXPERT_W_RESIDENT:
        # min(ntiles, 1) for ntiles in [0, ntmax]
        n1 = (ntiles + ntmax) // (ntmax + 1)
        nl.fori_loop(0, n1, lambda i: body(i, 0, True))
        nl.fori_loop(0, ntiles - n1, lambda i: body(i, 1))
    elif EXPERT_TILE_PAIRS and PAIRS_ONLY_IF_FIT and ntmax < 2:
        # at most one tile: the pair loop would never iterate
        nl.fori_loop(0, ntiles, lambda i: body_n(i, 1))
    elif EXPERT_TILE_PAIRS:
        # tiles (i, i + 1) per iteration sharing every weight load, then the odd last one
        npair = ntiles // 2
        nl.fori_loop(0, npair * 2, lambda i: body_n(i, 2), step=2)
        nl.fori_loop(npair * 2, ntiles, lambda i: body_n(i, 1))
    else:
        nl.fori_loop(0, ntiles, body)


def run_expert_dense(ntiles, e, xq, xs, wexp_sb, w1, s1, w3, s3, w2, s2, dim, inter, lo, hi,
                     limit, acc_sb):
    """Expert `e` over ALL the 128 rows of the single row tile (`Tp` = 128) when it has any
    (`ntiles` = 0 or 1, a device scalar): the stationary is the shared expert's staged
    tile of every row (`xq`, `xs`), the SwiGLU scales row `t` by `wexp[t, e]` (0 where the
    row did not pick the expert), and the w2 output is added to the fp32 SBUF accumulator
    `acc_sb` (`[128, dim]`). A picking row's values are `body_n`'s (rows are independent in
    every op; the same quantization, products and fp32 adds, experts in the same order);
    the other rows receive +-0, which leaves the sums unchanged. No row gather, no
    re-staging, no scatter-add."""
    span = hi - lo

    def body(i):
        g16 = nl.ndarray((PT, span), dtype=nl.bfloat16, buffer=nl.sbuf)
        u16 = nl.ndarray((PT, span), dtype=nl.bfloat16, buffer=nl.sbuf)

        def to_g(t, c0, w, acc):
            nisa.tensor_copy(dst=g16[:, c0 : c0 + w], src=acc)

        def to_u(t, c0, w, acc):
            nisa.tensor_copy(dst=u16[:, c0 : c0 + w], src=acc)

        X.gemm_sb_multi(xq, xs, 1, dim, w1, s1, span, to_g, fp4=True, n0=lo)
        X.gemm_sb_multi(xq, xs, 1, dim, w3, s3, span, to_u, fp4=True, n0=lo)
        yq, ys = X.alloc_x(1, span)
        y16 = nl.ndarray((PT, span), dtype=nl.bfloat16, buffer=nl.sbuf)
        swiglu_tile(g16, u16, wexp_sb[0:PT, e : e + 1], y16, span, limit)
        X.stage_rows(y16, span, yq, ys, 0)

        def to_acc(t, c0, w, acc):
            nisa.tensor_tensor(dst=acc_sb[:, c0 : c0 + w], data1=acc_sb[:, c0 : c0 + w], data2=acc,
                               op=nl.add)

        X.gemm_sb_multi(yq, ys, 1, span, w2, s2, dim, to_acc, fp4=True, krow0=lo // 4)

    nl.fori_loop(0, ntiles, body)


def _dense_wbufs(dim, span):
    """`alloc_w_wide`'s tiles for one dense expert's w1, w3 and w2 slices."""
    return (X.alloc_w_wide(dim, span, True), X.alloc_w_wide(dim, span, True),
            X.alloc_w_wide(span, dim, True))


def _dense_expert_tiled(xq, xs, wcol, w1, s1, w3, s3, w2, s2, dim, span, lo, limit, acc_sb,
                        eid=None, wbufs=None, acts=None):
    """One dense expert (`run_expert_dense`'s values) when w1 / w3's output tiles are w2's
    contraction chunks: each tile's SwiGLU runs as soon as its u is out of PSUM and its
    staging after the next tile's w3 matmuls (on ACT / DVE while the PE streams), instead
    of the whole SwiGLU and staging after the w3 GEMM (elementwise / per chunk: the same
    values). `eid`: the expert's device index into the stacked `w*` / `s*` (else they are
    the expert's own); `wbufs`: `_dense_wbufs`' tiles to load into; `acts`:
    `_dense_acts`' activation tiles (else new ones)."""
    tiles = X.ntiles_of(span)
    nt = len(tiles)
    col = {c0: j for j, (c0, _) in enumerate(tiles)}
    if acts is None:
        acts = _dense_acts(span)
    g16, u16, y16, yq, ys = acts

    def stage(j):
        pk = tiles[j][1] // 4
        xT = nl.ndarray((PT, PT, 4), dtype=nl.bfloat16, buffer=nl.sbuf)
        X.interleave_rows(y16[j], 0, tiles[j][1], xT)
        nisa.quantize_mx(dst=yq[0:pk, 0, j, :], src=xT[0:pk, :, :].reshape((pk, PT * 4)),
                         dst_scale=ys[0:pk, 0, j, :])

    def to_g1(t, c0, w, acc):
        nisa.tensor_copy(dst=g16[col[c0]], src=acc)

    def to_u1(t, c0, w, acc):
        j = col[c0]
        nisa.tensor_copy(dst=u16[j], src=acc)
        swiglu_tile(g16[j], u16[j], wcol, y16[j], w, limit)
        if j > 0:
            stage(j - 1)

    def to_acc1(t, c0, w, acc):
        nisa.tensor_tensor(dst=acc_sb[:, c0 : c0 + w], data1=acc_sb[:, c0 : c0 + w],
                           data2=acc, op=nl.add)

    wb = wbufs if wbufs is not None else (None, None, None)
    # every weight DMA of the expert issued first (the issuing queues also run
    # SwiGLU work, which would hold w2's loads behind the w3 matmuls)
    gw1 = X.load_w_wide(w1, s1, dim, span, True, n0=lo, eid=eid, into=wb[0])
    gw3 = X.load_w_wide(w3, s3, dim, span, True, n0=lo, eid=eid, into=wb[1])
    gw2 = X.load_w_wide(w2, s2, span, dim, True, krow0=lo // 4, eid=eid, into=wb[2])
    X.gemm_sb_multi_pre(xq, xs, 1, dim, gw1, to_g1)
    X.gemm_sb_multi_pre(xq, xs, 1, dim, gw3, to_u1)
    stage(nt - 1)
    X.gemm_sb_multi_pre(yq, ys, 1, span, gw2, to_acc1)


def _dense_acts(span):
    """`_dense_expert_tiled`'s per-tile g / u / y tiles and the staged y."""
    tiles = X.ntiles_of(span)
    g16 = [nl.ndarray((PT, w), dtype=nl.bfloat16, buffer=nl.sbuf) for _, w in tiles]
    u16 = [nl.ndarray((PT, w), dtype=nl.bfloat16, buffer=nl.sbuf) for _, w in tiles]
    y16 = [nl.ndarray((PT, w), dtype=nl.bfloat16, buffer=nl.sbuf) for _, w in tiles]
    yq, ys = X.alloc_x(1, span)
    return g16, u16, y16, yq, ys


def run_experts_dense_static(n_local, xq, xs, wexp_sb, w1, s1, w3, s3, w2, s2, dim, inter, lo,
                             hi, limit, acc_sb):
    """`run_expert_dense` of EVERY local expert, ascending, as static code (no dynamic loop):
    the experts nobody picked add +-0 (their `wexp` column is 0), the others exactly
    `run_experts_dense_active`'s values (each expert's ops are `_dense_expert_tiled`'s and
    the w2 outputs are added to `acc_sb` in expert order).

    Software-pipelined over the experts: the PE runs w1(0), w3(0), then for each e
    w1(e+1), w2(e), w3(e+1), ... -- expert e+1's w1 matmuls fill the PE while the last
    tile's SwiGLU / staging of expert e (ACT / DVE) completes. The weight tiles are reused
    (w1's double-buffered), each reload issued right after the matmuls that release its
    tiles, so every load has about two GEMM phases to stream; the activation tiles
    alternate between experts."""
    span = hi - lo
    tiles = X.ntiles_of(span)
    nt = len(tiles)
    col = {c0: j for j, (c0, _) in enumerate(tiles)}
    w1b = [X.alloc_w_wide(dim, span, True) for _ in range(2)]
    w3b = X.alloc_w_wide(dim, span, True)
    w2b = X.alloc_w_wide(span, dim, True)
    acts = [_dense_acts(span) for _ in range(2)]
    g1, g3, g2 = {}, {}, {}

    def L1(e):
        g1[e] = X.load_w_wide(w1[e], s1[e], dim, span, True, n0=lo, into=w1b[e % 2])

    def L3(e):
        g3[e] = X.load_w_wide(w3[e], s3[e], dim, span, True, n0=lo, into=w3b)

    def L2(e):
        g2[e] = X.load_w_wide(w2[e], s2[e], span, dim, True, krow0=lo // 4, into=w2b)

    def stage(e, j):
        _, _, y16, yq, ys = acts[e % 2]
        pk = tiles[j][1] // 4
        xT = nl.ndarray((PT, PT, 4), dtype=nl.bfloat16, buffer=nl.sbuf)
        X.interleave_rows(y16[j], 0, tiles[j][1], xT)
        nisa.quantize_mx(dst=yq[0:pk, 0, j, :], src=xT[0:pk, :, :].reshape((pk, PT * 4)),
                         dst_scale=ys[0:pk, 0, j, :])

    def P1(e):
        g16 = acts[e % 2][0]

        def to_g1(t, c0, w, acc):
            nisa.tensor_copy(dst=g16[col[c0]], src=acc)

        X.gemm_sb_multi_pre(xq, xs, 1, dim, g1.pop(e), to_g1)

    def P3(e):
        g16, u16, y16, _, _ = acts[e % 2]
        wcol = wexp_sb[0:PT, e : e + 1]

        def to_u1(t, c0, w, acc):
            j = col[c0]
            nisa.tensor_copy(dst=u16[j], src=acc)
            swiglu_tile(g16[j], u16[j], wcol, y16[j], w, limit)
            if j > 0:
                stage(e, j - 1)

        X.gemm_sb_multi_pre(xq, xs, 1, dim, g3.pop(e), to_u1)

    def P2(e):
        _, _, _, yq, ys = acts[e % 2]

        def to_acc1(t, c0, w, acc):
            nisa.tensor_tensor(dst=acc_sb[:, c0 : c0 + w], data1=acc_sb[:, c0 : c0 + w],
                               data2=acc, op=nl.add)

        X.gemm_sb_multi_pre(yq, ys, 1, span, g2.pop(e), to_acc1)

    n = n_local
    # the loads in the order they are needed
    L1(0)
    L3(0)
    if n > 1:
        L1(1)
    L2(0)
    P1(0)
    if n > 2:
        L1(2)
    for e in range(n):
        P3(e)
        if e + 1 < n:
            L3(e + 1)
            P1(e + 1)
            if e + 3 < n:
                L1(e + 3)
        stage(e, nt - 1)
        P2(e)
        if e + 1 < n:
            L2(e + 1)


def run_experts_dense_active(n_local, xq, xs, wexp_sb, w1, s1, w3, s3, w2, s2, dim, inter, lo, hi,
                             limit, acc_sb):
    """`run_expert_dense` of every local expert some row picked, in ascending expert order
    (the inactive ones added nothing), by one dynamic loop over their compacted ids:
    iteration `i` reads id `i` and addresses the expert's slice of the stacked weights
    (`w*` / `s*` `[E, ...]`) and its `wexp_sb` column with it. Every value and add is
    `run_expert_dense`'s."""
    span = hi - lo
    # each expert's number of picking rows (wexp > 0; the tile's rows past T are zero):
    # the pick mask summed over the partitions by a ones matmul, one row on partition 0
    pick = nl.ndarray((PT, n_local), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=pick, data=wexp_sb, op0=nl.greater, operand0=0.0)
    ones = nl.ndarray((PT, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)
    cnt = nl.ndarray((1, n_local), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=cnt, stationary=ones, moving=pick)
    mask = nl.ndarray((PT, n_local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(mask, 0.0)
    nisa.tensor_scalar(dst=mask[0:1, :], data=cnt, op0=nl.greater, operand0=0.0)
    # row 0: the active ids ascending (-1 padding), their number at column n_local
    nz = nl.ndarray((PT, n_local + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nonzero_with_count(dst=nz, src=mask, index_offset=0, padding_val=-1)
    nact = nisa.scalar_load(nz[0:1, n_local : n_local + 1])
    tiled = DENSE_TILE_SWIGLU and X.ntiles_of(span) == X.chunks_of(span)
    wbufs = _dense_wbufs(dim, span) if tiled and DENSE_W_HOIST else None

    def body(i):
        eid = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=eid, src=nz.ap(pattern=[[n_local + 1, 1], [1, 1]], offset=0,
                                            scalar_offset=i, indirect_dim=1))
        # the wexp column (an on-chip dynamic copy takes a uint32 index)
        eiu = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=eiu, src=eid)
        wcol = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=wcol, src=wexp_sb.ap(pattern=[[n_local, PT], [1, 1]], offset=0,
                                                  scalar_offset=eiu, indirect_dim=1))
        # the expert id as a register (the offset of its weights)
        eid = nisa.scalar_load(eid)
        if tiled:
            _dense_expert_tiled(xq, xs, wcol, w1, s1, w3, s3, w2, s2, dim, span, lo, limit, acc_sb,
                                eid=eid, wbufs=wbufs)
            return
        g16 = nl.ndarray((PT, span), dtype=nl.bfloat16, buffer=nl.sbuf)
        u16 = nl.ndarray((PT, span), dtype=nl.bfloat16, buffer=nl.sbuf)

        def to_g(t, c0, w, acc):
            nisa.tensor_copy(dst=g16[:, c0 : c0 + w], src=acc)

        def to_u(t, c0, w, acc):
            nisa.tensor_copy(dst=u16[:, c0 : c0 + w], src=acc)

        X.gemm_sb_multi(xq, xs, 1, dim, w1, s1, span, to_g, fp4=True, n0=lo, eid=eid)
        X.gemm_sb_multi(xq, xs, 1, dim, w3, s3, span, to_u, fp4=True, n0=lo, eid=eid)
        yq, ys = X.alloc_x(1, span)
        y16 = nl.ndarray((PT, span), dtype=nl.bfloat16, buffer=nl.sbuf)
        swiglu_tile(g16, u16, wcol, y16, span, limit)
        X.stage_rows(y16, span, yq, ys, 0)

        def to_acc(t, c0, w, acc):
            nisa.tensor_tensor(dst=acc_sb[:, c0 : c0 + w], data1=acc_sb[:, c0 : c0 + w], data2=acc,
                               op=nl.add)

        X.gemm_sb_multi(yq, ys, 1, span, w2, s2, dim, to_acc, fp4=True, krow0=lo // 4, eid=eid)

    nl.fori_loop(0, nact, body)


def routed_experts(pid, n_local, Tp, x_hbm, wexp_hbm, w1, s1, w3, s3, w2, s2, dim, inter, lo, hi,
                   limit, eacc_hbm, zero_acc=True, dense=None, nrows=None):
    """Every routed expert, this core taking `inter` columns `[lo, hi)` of each
    (see `run_expert`), into `eacc_hbm` (zeroed here unless `zero_acc` is False: the
    caller zeroed it). `w1..s2` are the stacked `[E, K/4, N]` x4 views / scales."""
    one_loop = (DENSE_ONE_LOOP and X.WIDE_W_BYTES_SB > 0 and len(X.chunks_of(dim)) <= X.WG
                and len(X.chunks_of(hi - lo)) <= X.WG)
    if dense is not None and one_loop:
        # (the row lists are not needed: the active experts come from `wexp_sb`)
        xq, xs, wexp_sb, acc_sb = dense
        span = hi - lo
        if (DENSE_STATIC_MIN_T and nrows is not None and nrows >= DENSE_STATIC_MIN_T
                and DENSE_TILE_SWIGLU and X.ntiles_of(span) == X.chunks_of(span)):
            run_experts_dense_static(n_local, xq, xs, wexp_sb, w1, s1, w3, s3, w2, s2, dim, inter,
                                     lo, hi, limit, acc_sb)
            return
        run_experts_dense_active(n_local, xq, xs, wexp_sb, w1, s1, w3, s3, w2, s2, dim, inter, lo,
                                 hi, limit, acc_sb)
        return
    if zero_acc:
        zero_rows(eacc_hbm, Tp + PT, dim)
    glist = alloc_lists(n_local, Tp)
    build_expert_lists(wexp_hbm, Tp, n_local, glist)
    rampf = partition_ramp()
    counts = None
    if COUNTS_ONCE and n_local <= PT:
        # every expert's count (column Tp of its list row), expert e on partition e
        counts = nl.ndarray((n_local, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=counts, src=glist[0:n_local, Tp : Tp + 1])
    for e in range(n_local):
        if counts is not None:
            nt = (nisa.scalar_load(counts[e : e + 1, 0:1]) + (PT - 1)) // PT
        else:
            nt = expert_tiles(glist, e, Tp)
        if dense is not None:
            # `dense` = (xq, xs, wexp_sb, acc_sb): see `run_expert_dense` (Tp = 128)
            xq, xs, wexp_sb, acc_sb = dense
            run_expert_dense(nt, e, xq, xs, wexp_sb, w1[e], s1[e], w3[e], s3[e], w2[e], s2[e],
                             dim, inter, lo, hi, limit, acc_sb)
            continue
        run_expert(nt, e, glist, Tp, x_hbm, wexp_hbm, n_local, w1[e], s1[e], w3[e], s3[e], w2[e], s2[e],
                   dim, inter, lo, hi, limit, eacc_hbm, rampf)
