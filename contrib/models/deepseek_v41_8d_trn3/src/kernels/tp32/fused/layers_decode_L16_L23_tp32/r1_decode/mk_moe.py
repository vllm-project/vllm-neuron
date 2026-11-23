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

#: At one row tile (Tp = 128: T = 6 and T = 60) each picked expert runs over the whole
#: tile in row order instead of its gathered rows (`run_expert_dense`): the rows are the
#: caller's one staging of the tile (the shared expert's), the routing weight column is
#: read in place (0 on the rows that did not pick the expert, so their SwiGLU output and
#: w2 rows are exactly 0), and the w2 output tiles add straight from PSUM into an SBUF
#: fp32 accumulator (no gather, no per-expert staging of x, no RMW scatter through HBM,
#: no zeroed HBM accumulator). Every row's values and adds are the gathered path's.
#: w1 / w3 / SwiGLU / staging of y run per 512-column tile so each tile's SwiGLU and
#: quantization overlap the next tile's matmuls.
DENSE_ONE_TILE = True
#: `run_expert_dense`'s weight-scale tiles are zeroed by GpSimd (idle there) instead of
#: the compiler's pick (DVE, busy with the SwiGLU / staging / accumulation): the hardware
#: DGE's scale DMAs wait on that memset, so a DVE memset queued behind the SwiGLU held
#: the w2 weight stream back until the w1 / w3 phase was done.
DENSE_WS_GPSIMD = True

#: Mask rows an instruction compacts (one per GpSimd core: partitions 0, 16, .., 112).
LANES = 8


#: `DENSE_ONE_TILE` with at least this many rows: every local expert's body runs
#: unconditionally (static code, no `fori_loop`). An expert no row picked adds exact zeros
#: (its routing column is 0 on every row), and each dynamic loop costs ~8-10 us of
#: all-engine semaphore rounds, a register load and the activation-table load even when
#: empty; with many rows most local experts are picked (P = 1 - (1 - 6/384)^T: 0.61 at
#: T = 60, every expert on the busiest rank), the PE runs the experts back to back and the
#: next expert's weight DMAs overlap this one's matmuls. Fewer rows keep the loops.
#: OFF (None): at in1024_bs10 (T = 60) 8.55 ms vs 8.12 with the loops -- the world
#: all-reduce's skew wait went away (~55 us) but every rank then ran 6 experts at ~37 us
#: each (~220 us a block), more than the busiest rank's ~4 active ones with the loops.
DENSE_STATIC_MIN_ROWS = None
#: (fewer rows) every expert's trip count is loaded before the first expert's loop
#: instead of in front of each loop.
DENSE_COUNTS_FIRST = True
#: `run_expert_dense`'s weight-scale tiles (one per output tile of w1 / w3 / w2) are
#: allocated and zeroed once, before the first expert's loop, and shared by every expert
#: (each body's DMAs only write their quadrant rows, the other partitions stay 0):
#: inside the loop the 16 GpSimd memsets a body (~11 us) gated the scale DMAs, and the
#: weight data DMAs queued behind those on the same (sync) queue. (Measured neutral at
#: in1024: an active body stays ~42 us, ~6.5 us of it before the first weight tile lands.)
DENSE_WS_HOISTED = True
#: `run_expert_dense` issues tile c+1's w1 / w3 matmuls before tile c's y staging
#: (transposes, quantize): the PE runs its queue in order and waited ~1.5-2.5 us a tile
#: for the SwiGLU on the DVE before the transposes (same instructions, same values).
DENSE_STAGE_SKEW = True


def dense_fits(Tp, n_local):
    """Whether `routed_experts` takes the one-tile dense path (`DENSE_ONE_TILE`)."""
    return DENSE_ONE_TILE and Tp == PT and n_local <= PT


def alloc_lists(n_local, Tp):
    """Private HBM for the per-expert row lists: `[n_local, (ntmax + 1) * 128]` int32,
    row `e` = ids (ascending), -1 padding, and the count at column `Tp`."""
    ntmax = Tp // PT
    return nl.ndarray((n_local, (ntmax + 1) * PT), dtype=nl.int32, buffer=nl.private_hbm)


def build_expert_lists(wexp_hbm, Tp, n_local, glist):
    """Compact `wexp[0:Tp, e] > 0` into `glist` (see `alloc_lists`); with `glist` None,
    return each expert's count as a (1, 1) int32 SBUF view instead."""
    nt = Tp // PT
    cnts = []
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
        if glist is None:
            cnts += [nz[16 * q : 16 * q + 1, Tp : Tp + 1] for q in range(ne)]
            continue
        for q in range(ne):
            nisa.dma_copy(dst=glist[e0 + q : e0 + q + 1, 0 : Tp + 1], src=nz[16 * q : 16 * q + 1, :])
    return cnts


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
    """`hbm[0:rows, 0:n] = 0` (rows a multiple of 128)."""
    z = nl.ndarray((PT, n), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(z, 0.0)
    for i in range(rows // PT):
        nisa.dma_copy(dst=hbm[i * PT : (i + 1) * PT, 0:n], src=z)


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


def _mm_tile(xq, xs, chunks, w_mx, s_mx, krow0, n0, w, ws_pre=None):
    """One fp32 PSUM output tile (columns `n0 .. n0 + w`) of slot 0 of `(xq, xs)` against the
    fp4 weight: `gemm_sb`'s weight loads and matmuls for that tile, in its order.
    `ws_pre`: a zeroed `(PT, nkc, w)` scale tile to load into (`DENSE_WS_HOISTED`)."""
    ps = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.psum)
    nkc = len(chunks)
    for g0 in range(0, nkc, X.WG):
        ng = min(X.WG, nkc - g0)
        if ws_pre is not None:
            assert ng == nkc
            wt = nl.ndarray((PT, ng, w), dtype=X.FP4_X4, buffer=nl.sbuf)
            ws = ws_pre
            X._load_w_into(wt, ws, 0, w_mx, s_mx, chunks, g0, ng, krow0, n0, w, X.GEMM_SB_HWDGE)
        else:
            wt, ws = X.load_w_group(w_mx, s_mx, chunks, g0, ng, krow0, n0, w, True,
                                    hw=X.GEMM_SB_HWDGE,
                                    ms_engine=nisa.engine.gpsimd if DENSE_WS_GPSIMD else None)
        for j in range(ng):
            kc = g0 + j
            pk = chunks[kc][1] // 4
            nisa.nc_matmul_mx(
                dst=ps, stationary=xq[0:pk, 0, kc, :], moving=wt[0:pk, j, :],
                stationary_scale=xs[0:pk, 0, kc, :], moving_scale=ws[0:pk, j, :],
                accumulate=(kc > 0),
            )
    return ps


def _swiglu_cols(g16, u16, wcol, y16, c0, w, limit):
    """`swiglu_tile`'s arithmetic on the columns `c0 .. c0 + w`."""
    g = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=g, data=g16[:, c0 : c0 + w], op0=nl.minimum, operand0=limit)
    u = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=u, data=u16[:, c0 : c0 + w], op0=nl.maximum, operand0=-limit,
                       op1=nl.minimum, operand1=limit)
    sg = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sg, op=nl.silu, data=g)
    y = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=y, data1=sg, data2=u, op=nl.multiply)
    nisa.tensor_scalar(dst=y16[:, c0 : c0 + w], data=y, op0=nl.multiply, operand0=wcol)


def alloc_ws_dense(dim, span):
    """`DENSE_WS_HOISTED`: the zeroed scale tiles of one `run_expert_dense` body:
    `{("w1" | "w3", kc) | ("w2", i): (PT, nkc, w) uint8}`."""
    nk1 = len(X.chunks_of(dim))
    nk2 = len(X.chunks_of(span))
    wss = {}
    for kc, (_, w) in enumerate(X.chunks_of(span)):
        for nm in ("w1", "w3"):
            wss[(nm, kc)] = nl.ndarray((PT, nk1, w), dtype=nl.uint8, buffer=nl.sbuf)
    for i, (_, w) in enumerate(X.ntiles_of(dim)):
        wss[("w2", i)] = nl.ndarray((PT, nk2, w), dtype=nl.uint8, buffer=nl.sbuf)
    for t in wss.values():
        nisa.memset(t, 0, engine=nisa.engine.gpsimd)
    return wss


def run_expert_dense(ntiles, e, xq, xs, wall, w1, s1, w3, s3, w2, s2, dim, lo, hi, limit, acc,
                     wss=None):
    """Expert `e` over the whole (single) row tile, `ntiles` (0 or 1, a device scalar) times;
    see `DENSE_ONE_TILE`. `(xq, xs)` slot 0 is the staged tile, `wall` the `(128, n_local)`
    fp32 routing weights, `acc` the `(128, dim)` fp32 SBUF accumulator, `wss` the
    `alloc_ws_dense` scale tiles (or None)."""
    span = hi - lo
    kch = X.chunks_of(dim)
    ych = X.chunks_of(span)
    wsp = (lambda k: wss[k]) if wss is not None else (lambda k: None)

    def body(i):
        wcol = wall[0:PT, e : e + 1]
        g16 = nl.ndarray((PT, span), dtype=nl.bfloat16, buffer=nl.sbuf)
        u16 = nl.ndarray((PT, span), dtype=nl.bfloat16, buffer=nl.sbuf)
        y16 = nl.ndarray((PT, span), dtype=nl.bfloat16, buffer=nl.sbuf)
        yq, ys = X.alloc_x(1, span)

        # w1 / w3 output tile c = the w2 contraction chunk c (both 512-wide)
        def up(kc):
            c0, w = ych[kc]
            ps = _mm_tile(xq, xs, kch, w1, s1, 0, lo + c0, w, wsp(("w1", kc)))
            nisa.tensor_copy(dst=g16[:, c0 : c0 + w], src=ps)
            ps = _mm_tile(xq, xs, kch, w3, s3, 0, lo + c0, w, wsp(("w3", kc)))
            nisa.tensor_copy(dst=u16[:, c0 : c0 + w], src=ps)
            _swiglu_cols(g16, u16, wcol, y16, c0, w, limit)

        def stage(kc):
            c0, w = ych[kc]
            pk = w // 4
            xT = nl.ndarray((PT, PT, 4), dtype=nl.bfloat16, buffer=nl.sbuf)
            X.interleave_rows(y16, c0, w, xT)
            nisa.quantize_mx(dst=yq[0:pk, 0, kc, :], src=xT[0:pk, :, :].reshape((pk, PT * 4)),
                             dst_scale=ys[0:pk, 0, kc, :])

        if DENSE_STAGE_SKEW:
            up(0)
            for kc in range(len(ych)):
                if kc + 1 < len(ych):
                    up(kc + 1)
                stage(kc)
        else:
            for kc in range(len(ych)):
                up(kc)
                stage(kc)
        for i, (c0, w) in enumerate(X.ntiles_of(dim)):
            ps = _mm_tile(yq, ys, ych, w2, s2, lo // 4, c0, w, wsp(("w2", i)))
            nisa.tensor_tensor(dst=acc[:, c0 : c0 + w], data1=acc[:, c0 : c0 + w], data2=ps,
                               op=nl.add)

    if ntiles is None:
        # `DENSE_STATIC_MIN_ROWS`: unconditionally, once
        body(0)
        return
    nl.fori_loop(0, ntiles, body)


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


def routed_experts(pid, n_local, Tp, x_hbm, wexp_hbm, w1, s1, w3, s3, w2, s2, dim, inter, lo, hi,
                   limit, eacc_hbm, zero_acc=True, staged=None, rows=None):
    """Every routed expert, this core taking `inter` columns `[lo, hi)` of each
    (see `run_expert`), into `eacc_hbm` (zeroed here unless `zero_acc` is False: the
    caller zeroed it). `w1..s2` are the stacked `[E, K/4, N]` x4 views / scales.

    With `dense_fits(Tp, n_local)` (one row tile) the experts run `run_expert_dense` and
    the accumulator is returned as an SBUF `(128, dim)` fp32 tile instead (`eacc_hbm` is
    not used); `staged` is the caller's `(xq, xs)` of `x_hbm`'s rows 0..127 (else staged
    here; `rows` the real row count, see `DENSE_STATIC_MIN_ROWS`). Otherwise returns None."""
    if dense_fits(Tp, n_local):
        static = (DENSE_STATIC_MIN_ROWS is not None and rows is not None
                  and rows >= DENSE_STATIC_MIN_ROWS)
        acc = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(acc, 0.0)
        cnts = None if static else build_expert_lists(wexp_hbm, Tp, n_local, None)
        wall = nl.ndarray((PT, n_local), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=wall, src=wexp_hbm[0:PT, 0:n_local])
        if staged is None:
            staged = X.alloc_x(1, dim)
            X.stage_x(x_hbm, 0, dim, staged[0], staged[1], 0)
        xq, xs = staged
        wss = alloc_ws_dense(dim, hi - lo) if DENSE_WS_HOISTED else None
        nts = [None] * n_local
        if not static and DENSE_COUNTS_FIRST:
            nts = [(nisa.scalar_load(cnts[e]) + (PT - 1)) // PT for e in range(n_local)]
        for e in range(n_local):
            nt = nts[e]
            if not static and not DENSE_COUNTS_FIRST:
                nt = (nisa.scalar_load(cnts[e]) + (PT - 1)) // PT
            run_expert_dense(nt, e, xq, xs, wall, w1[e], s1[e], w3[e], s3[e], w2[e], s2[e],
                             dim, lo, hi, limit, acc, wss)
        return acc
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
        run_expert(nt, e, glist, Tp, x_hbm, wexp_hbm, n_local, w1[e], s1[e], w3[e], s3[e], w2[e], s2[e],
                   dim, inter, lo, hi, limit, eacc_hbm, rampf)
    return None
