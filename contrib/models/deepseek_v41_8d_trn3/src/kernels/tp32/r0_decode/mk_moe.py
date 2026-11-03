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

#: With T < 128 rows, an expert tile's gather and scatter-add cover only its first T
#: slots (where the compacted real rows are).
EXPERT_REAL_ROWS = True

#: With a single row tile (T <= 128): ONE dynamic loop over the active experts (those
#: some row picked; their ids compacted on chip) instead of a loop per expert (every
#: dynamic loop costs its block-switch overhead even at zero trips), the expert id a
#: device scalar that offsets the weight DMAs. The T rows are staged once and every
#: active expert runs over all of them, a row that did not pick it weighted 0 (its
#: SwiGLU output, w2 output and accumulator update are exact zeros); a row's values
#: are independent of the other rows (per-row quantization, per-row matmul outputs),
#: so every picked row's result and its accumulation order (ascending expert) are
#: unchanged. The accumulator update is a static RMW over the T rows.
EXPERT_MERGED = True
#: From this many rows (most local experts active), the merged loop runs two experts
#: per iteration.
EXPERT_PAIR_MIN_T = 16

#: Mask rows an instruction compacts (one per GpSimd core: partitions 0, 16, .., 112).
LANES = 8


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
    z = nl.ndarray((PT, n), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(z, 0.0)
    for r0 in range(0, rows, PT):
        nr = min(PT, rows - r0)
        nisa.dma_copy(dst=hbm[r0 : r0 + nr, 0:n], src=z[0:nr, :])


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
               dim, inter, lo, hi, limit, eacc_hbm, rampf, nreal=PT):
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

    def tile_ids(i, t):
        # tile i + t's row ids; padding (-1) -> Tp + p: a distinct zero row per slot
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
        return gi

    def body_n(i, nmt):
        # the `nmt` tiles i .. i + nmt - 1 (each tile's values exactly `body`'s): their
        # rows staged together, every weight group loaded once for all of them
        gis = [tile_ids(i, t) for t in range(nmt)]
        xq, xs = X.alloc_x(nmt, dim)
        for t in range(nmt):
            X.stage_x_gathered(x_hbm, gis[t], dim, xq, xs, t, nrows=nreal)
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
            tgt = eacc_hbm.ap(pattern=[[dim, nreal], [1, dim]], vector_offset=gis[t][0:nreal, 0:1],
                              indirect_dim=0)
            nisa.dma_compute(dst=tgt, srcs=[tgt, o32[t][0:nreal, :]], reduce_op=nl.add)

    if EXPERT_W_RESIDENT:
        # min(ntiles, 1) for ntiles in [0, ntmax]
        n1 = (ntiles + ntmax) // (ntmax + 1)
        nl.fori_loop(0, n1, lambda i: body(i, 0, True))
        nl.fori_loop(0, ntiles - n1, lambda i: body(i, 1))
    elif EXPERT_TILE_PAIRS and ntmax < 2:
        # a single row tile (decode): no pair loop (it would never iterate, but every
        # dynamic loop costs its block-switch overhead even at zero trips)
        nl.fori_loop(0, ntiles, lambda i: body_n(i, 1))
    elif EXPERT_TILE_PAIRS:
        # tiles (i, i + 1) per iteration sharing every weight load, then the odd last one
        npair = ntiles // 2
        nl.fori_loop(0, npair * 2, lambda i: body_n(i, 2), step=2)
        nl.fori_loop(npair * 2, ntiles, lambda i: body_n(i, 1))
    else:
        nl.fori_loop(0, ntiles, body)


def routed_experts_merged(n_local, T, x_hbm, wexp_hbm, w1, s1, w3, s3, w2, s2, dim, lo, hi, limit,
                          eacc_hbm, xstage=None):
    """`routed_experts` for T <= 128 rows (see `EXPERT_MERGED`). `xstage`: the rows
    already staged (`X.stage_x` of `x_hbm` rows 0..128, rows past T zero)."""
    span = hi - lo
    nreal = min(PT, T)
    # routing weights [rows, experts] and the active experts' ids (ascending) + count
    wsb = nl.ndarray((PT, n_local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=wsb, src=wexp_hbm[0:PT, 0:n_local])
    m16 = nl.ndarray((PT, n_local), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=m16, data=wsb, op0=nl.greater, operand0=0.0)
    ones = nl.ndarray((PT, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)
    cnt = nl.ndarray((1, n_local), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=cnt, stationary=ones, moving=m16)
    flag = nl.ndarray((PT, n_local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(flag, 0.0)
    nisa.tensor_scalar(dst=flag[0:1, :], data=cnt, op0=nl.greater, operand0=0.0)
    nz = nl.ndarray((PT, n_local + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nonzero_with_count(dst=nz, src=flag, index_offset=0, padding_val=-1)
    alist = nl.ndarray((1, n_local + 1), dtype=nl.int32, buffer=nl.private_hbm)
    nisa.dma_copy(dst=alist, src=nz[0:1, :])
    c = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c, src=alist[0:1, n_local : n_local + 1])
    nact = nisa.scalar_load(c)
    # the rows, quantized once (rows past T zero)
    if xstage is not None:
        xq, xs = xstage
    else:
        xq, xs = X.alloc_x(1, dim)
        rows = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        if nreal < PT:
            nisa.memset(rows, 0.0, engine=nisa.engine.gpsimd)
        nisa.dma_copy(dst=rows[0:nreal, :], src=x_hbm[0:nreal, 0:dim])
        X.stage_rows(rows, dim, xq, xs, 0)
    rid = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=rid, pattern=[[1, 1]], offset=0, channel_multiplier=1)

    def body(i, t=0):
        # active expert i + t
        ec = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ec, src=alist.ap(pattern=[[n_local + 1, 1], [1, 1]], offset=t, scalar_offset=i,
                                           indirect_dim=1))
        e = nisa.scalar_load(ec)
        g16 = nl.ndarray((PT, span), dtype=nl.bfloat16, buffer=nl.sbuf)
        u16 = nl.ndarray((PT, span), dtype=nl.bfloat16, buffer=nl.sbuf)

        def to_g(t, c0, w, acc):
            nisa.tensor_copy(dst=g16[:, c0 : c0 + w], src=acc)

        def to_u(t, c0, w, acc):
            nisa.tensor_copy(dst=u16[:, c0 : c0 + w], src=acc)

        X.gemm_sb_multi(xq, xs, 1, dim, w1, s1, span, to_g, fp4=True, n0=lo, eoff=e)
        X.gemm_sb_multi(xq, xs, 1, dim, w3, s3, span, to_u, fp4=True, n0=lo, eoff=e)
        wcol = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=wcol, src=wexp_hbm.ap(pattern=[[n_local, PT], [1, 1]], offset=0, scalar_offset=e,
                                               indirect_dim=1))
        y16 = nl.ndarray((PT, span), dtype=nl.bfloat16, buffer=nl.sbuf)
        swiglu_tile(g16, u16, wcol, y16, span, limit)
        yq, ys = X.alloc_x(1, span)
        X.stage_rows(y16, span, yq, ys, 0)
        o32 = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)

        def to_o(t, c0, w, acc):
            nisa.tensor_copy(dst=o32[:, c0 : c0 + w], src=acc)

        X.gemm_sb_multi(yq, ys, 1, span, w2, s2, dim, to_o, fp4=True, krow0=lo // 4, eoff=e)
        # (through identity row ids: the indirect form `run_expert` uses)
        tgt = eacc_hbm.ap(pattern=[[dim, nreal], [1, dim]], vector_offset=rid[0:nreal, 0:1], indirect_dim=0)
        nisa.dma_compute(dst=tgt, srcs=[tgt, o32[0:nreal, :]], reduce_op=nl.add)

    if T >= EXPERT_PAIR_MIN_T:
        # two active experts per iteration (the second's weight streams overlap the first's
        # matmuls), then the odd last one
        def body2(i):
            body(i, 0)
            body(i, 1)

        n2 = (nact // 2) * 2
        nl.fori_loop(0, n2, body2, step=2)
        nl.fori_loop(n2, nact, body)
    else:
        nl.fori_loop(0, nact, body)


def routed_experts(pid, n_local, Tp, x_hbm, wexp_hbm, w1, s1, w3, s3, w2, s2, dim, inter, lo, hi,
                   limit, eacc_hbm, zero_acc=True, T=None, xstage=None):
    """Every routed expert, this core taking `inter` columns `[lo, hi)` of each
    (see `run_expert`), into `eacc_hbm` (zeroed here unless `zero_acc` is False: the
    caller zeroed it). `w1..s2` are the stacked `[E, K/4, N]` x4 views / scales."""
    if zero_acc:
        zero_rows(eacc_hbm, Tp + PT, dim)
    # (not for a single row: the merged loop's indirect scatter-add `dma_compute` needs a
    # partition size > 1 -- a neuronx-cc limit -- so T = 1 takes the per-expert loops)
    if EXPERT_MERGED and Tp == PT and T is not None and T > 1:
        routed_experts_merged(n_local, T, x_hbm, wexp_hbm, w1, s1, w3, s3, w2, s2, dim, lo, hi, limit,
                              eacc_hbm, xstage=xstage)
        return
    glist = alloc_lists(n_local, Tp)
    build_expert_lists(wexp_hbm, Tp, n_local, glist)
    rampf = partition_ramp()
    for e in range(n_local):
        nt = expert_tiles(glist, e, Tp)
        # an expert's row list is compacted ascending, so a tile holds at most min(T, 128)
        # real rows, in its first slots: the gather and the scatter-add skip the rest
        # (padding slots: zero inputs, discarded outputs). At least 2: the indirect
        # scatter-add needs a partition size > 1, and a padding slot is harmless (it
        # gathers a zero row and scatter-adds into the extra zero tile).
        nreal = PT if (T is None or not EXPERT_REAL_ROWS) else max(2, min(PT, T))
        run_expert(nt, e, glist, Tp, x_hbm, wexp_hbm, n_local, w1[e], s1[e], w3[e], s3[e], w2[e], s2[e],
                   dim, inter, lo, hi, limit, eacc_hbm, rampf, nreal=nreal)
