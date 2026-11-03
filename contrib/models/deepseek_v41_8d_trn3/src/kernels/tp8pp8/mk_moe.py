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
        y = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y, data1=sg, data2=u, op=nl.multiply)
        if wcol is not None:
            nisa.tensor_scalar(dst=y, data=y, op0=nl.multiply, operand0=wcol[0:PT, 0:1])
        nisa.tensor_copy(dst=y16[:, c0 : c0 + chunk], src=y)


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

    def body(i):
        gi = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gi, src=g3.ap(pattern=[[1, PT], [1, 1]], offset=e * (ntmax + 1) * PT,
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
        # the tile's rows, quantized
        xq, xs = X.alloc_x(1, dim)
        X.stage_x_gathered(x_hbm, gi, dim, xq, xs, 0)
        g16 = nl.ndarray((PT, span), dtype=nl.bfloat16, buffer=nl.sbuf)
        X.gemm_sb(xq, xs, dim, w1, s1, g16, span, fp4=True, n0=lo)
        u16 = nl.ndarray((PT, span), dtype=nl.bfloat16, buffer=nl.sbuf)
        X.gemm_sb(xq, xs, dim, w3, s3, u16, span, fp4=True, n0=lo)
        wcol = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=wcol, src=wexp_hbm.ap(pattern=[[n_local, PT], [1, 1]], offset=e,
                                               vector_offset=gi[0:PT, 0:1], indirect_dim=0))
        y16 = nl.ndarray((PT, span), dtype=nl.bfloat16, buffer=nl.sbuf)
        swiglu_tile(g16, u16, wcol, y16, span, limit)
        yq, ys = X.alloc_x(1, span)
        X.stage_rows(y16, span, yq, ys, 0)
        o32 = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
        X.gemm_sb(yq, ys, span, w2, s2, o32, dim, fp4=True, krow0=lo // 4)
        tgt = eacc_hbm.ap(pattern=[[dim, PT], [1, dim]], vector_offset=gi[0:PT, 0:1], indirect_dim=0)
        nisa.dma_compute(dst=tgt, srcs=[tgt, o32], reduce_op=nl.add)

    nl.fori_loop(0, ntiles, body)


def routed_experts(pid, n_local, Tp, x_hbm, wexp_hbm, w1, s1, w3, s3, w2, s2, dim, inter, lo, hi,
                   limit, eacc_hbm):
    """Every routed expert, this core taking `inter` columns `[lo, hi)` of each
    (see `run_expert`), into `eacc_hbm` (zeroed here). `w1..s2` are the stacked
    `[E, K/4, N]` x4 views / scales."""
    zero_rows(eacc_hbm, Tp + PT, dim)
    glist = alloc_lists(n_local, Tp)
    build_expert_lists(wexp_hbm, Tp, n_local, glist)
    rampf = partition_ramp()
    for e in range(n_local):
        nt = expert_tiles(glist, e, Tp)
        run_expert(nt, e, glist, Tp, x_hbm, wexp_hbm, n_local, w1[e], s1[e], w3[e], s3[e], w2[e], s2[e],
                   dim, inter, lo, hi, limit, eacc_hbm, rampf)
