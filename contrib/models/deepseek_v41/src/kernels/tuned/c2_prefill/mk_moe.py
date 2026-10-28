# SPDX-License-Identifier: Apache-2.0
"""Token-grouped routed experts for the Trn2 megakernels.

The router leaves `wexp[t, e]`, the routing weight of local expert `e` for row
`t` (0 where the row did not pick it). Each expert runs only over the rows that
picked it, as the trn3 megakernels do it:

  1. `build_expert_lists`: the row ids of each expert, compacted in ascending
     order by one `nonzero_with_count` over the mask rows (GpSimd, 8 experts an
     instruction), written to a small HBM list with the count.
  2. per expert, a `fori_loop` over `ceil(count / 128)` row tiles (a bound read on
     the device: an expert nobody picked costs nothing, not even its weight
     reads): gather the tile's rows, the whole SwiGLU FFN, and scatter-add the
     fp32 result into the accumulator with `dma_compute`.

What differs from trn3 is the FFN itself, which here is `kernels/moe_expert`'s
fp4-in-SBUF pipeline (`_gemm_fp4`: the k-major fp4 weight DMA'd straight into
SBUF, four nibbles a lane op decoded on the vector engine, bf16 matmuls), the
kernel the per-op path runs per expert -- with its two-core split: core `p`
computes intermediate columns `[p*1152, (p+1)*1152)` of w1 / w3 and its own
SwiGLU, the halves are exchanged through shared HBM, and core `p` then computes
output columns `[p*2560, (p+1)*2560)` of w2 over the whole intermediate axis. So
the two cores' results are disjoint *columns* of the accumulator, and each core
scatter-adds only its own.

A tile's padding slots (past the expert's last row) point at distinct junk rows
past the real ones -- the gathered inputs there are zero rows and `dma_compute`
wants unique indices -- so the input, the weights and the accumulator all carry
one extra 128-row tile.
"""

import nki.isa as nisa
import nki.language as nl

from ... import moe_expert as ME

PT = 128
#: Mask rows an instruction compacts (one per GpSimd core: partitions 0, 16, .., 112).
LANES = 8
DIM = ME.DIM
INTER = ME.INTER
DIM_H = ME.DIM_H
INTER_H = ME.INTER_H


def alloc_lists(n_local, Tp):
    """Private HBM for the per-expert row lists: `[n_local, (Tp/128 + 1) * 128]`
    int32, row `e` = ids (ascending), -1 padding, and the count at column `Tp`."""
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


def zero_cols(hbm, rows, c0, c1, dtype=nl.float32):
    """`hbm[0:rows, c0:c1] = 0` (rows a multiple of 128)."""
    z = nl.ndarray((PT, c1 - c0), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(z, 0.0)
    for i in range(rows // PT):
        nisa.dma_copy(dst=hbm[i * PT : (i + 1) * PT, c0:c1], src=z)


def _expert_tile(hs, wcols, w1, s1, w3, s3, w2, s2, ytile, tx, pid):
    """`moe_expert._expert_group` for `len(hs)` gathered 128-row tiles held in SBUF
    (`hs[m]` bf16 `[128, DIM]`, `wcols[m]` fp32 `(128, 1)`), each staged weight tile
    decoded once for all of them: this core's output columns `[pid*DIM_H, ...)` of
    the expert's fp32 result into `ytile[m*128 : (m+1)*128, ...]`. Every m-tile's
    PSUM accumulation is the one tile's (same widths, same k order)."""
    mg = len(hs)
    nw1, nw2 = ME._widths(mg)
    assert (nw1, nw2) == ME._widths(1), "a group must accumulate as one tile does"
    icol0 = pid * INTER_H
    ocol0 = pid * DIM_H
    alay1 = [ME._quant_layout(h, PT, DIM, ME.CHK1) for h in hs]
    gs = [nl.ndarray((PT, INTER_H), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in hs]
    us = [nl.ndarray((PT, INTER_H), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in hs]
    ME._gemm_fp4([w1, w3], [s1, s3], INTER, icol0, INTER_H, nw1, ME.NBG1, ME.SEG1, ME.SSC1,
                 alay1, [gs, us], None, ME.SCENG1, ME.PRENG1, ME.KSP1, ME.PDEP1)
    ts = []
    for m in range(mg):
        t = nl.ndarray((PT, INTER_H), dtype=nl.bfloat16, buffer=nl.sbuf)
        ME._swiglu(gs[m], us[m], wcols[m], t)
        o0 = (pid * mg + m) * PT
        nisa.dma_copy(dst=tx[o0 : o0 + PT, 0:INTER_H], src=t)
        ts.append(t)
    nisa.core_barrier(data=tx, cores=(0, 1))
    alay2 = []
    oth = (1 - pid) * INTER_H
    for m in range(mg):
        full = nl.ndarray((PT, INTER), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=full[:, icol0 : icol0 + INTER_H], src=ts[m])
        o0 = ((1 - pid) * mg + m) * PT
        nisa.dma_copy(dst=full[:, oth : oth + INTER_H], src=tx[o0 : o0 + PT, 0:INTER_H])
        alay2.append(ME._quant_layout(full, PT, INTER, ME.CHK2))
    nisa.core_barrier(data=tx, cores=(0, 1))
    ME._gemm_fp4([w2], [s2], DIM, ocol0, DIM_H, nw2, ME.NBG2, ME.SEG2, ME.SSC2,
                 alay2, None, (ytile, [m * PT for m in range(mg)]), ME.SCENG2, ME.PRENG2,
                 ME.KSP2, ME.PDEP2)


#: Row tiles of one expert decoded together (`_expert_tile`): an expert with `n`
#: tiles runs `n // PAIR` groups of `PAIR`, then the rest one at a time.
PAIR = 2


def run_expert(ntiles, e, glist, Tp, x_hbm, wexp_hbm, n_local, w, eacc_hbm, rampf, ytile, tx, pid,
               ytile2=None, tx2=None):
    """Expert `e` over its `ntiles` (a device scalar) row tiles.

    Both cores run every expert with the same trip count (an LNC=2 dynamic loop
    must) and split each tile as `_expert_tile` says. `x_hbm` is the bf16
    `[Tp + 128, DIM]` input with zero rows past `Tp`, `wexp_hbm` the
    `[Tp + 128, n_local]` fp32 routing weights (zero past `Tp`), `w` the expert's
    `(w1, s1, w3, s3, w2, s2)` k-major fp4 / e8m0 tensors, `eacc_hbm` the
    `[Tp + 128, DIM]` fp32 accumulator; this core adds into its own column half.
    With `ytile2`/`tx2` (sized for `PAIR` tiles) whole pairs of tiles share one
    weight decode.
    """
    ntmax = Tp // PT
    g3 = glist.reshape((n_local, ntmax + 1, PT))
    w1, s1, w3, s3, w2, s2 = w
    ocol0 = pid * DIM_H

    def load(i, m):
        """Tile `i + m`'s row ids, rows and routing weights."""
        gi = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gi, src=g3.ap(pattern=[[1, PT], [1, 1]],
                                       offset=e * (ntmax + 1) * PT + m * PT,
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
        h = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=h, src=x_hbm.ap(pattern=[[DIM, PT], [1, DIM]], vector_offset=gi[0:PT, 0:1],
                                          indirect_dim=0))
        wcol = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=wcol, src=wexp_hbm.ap(pattern=[[n_local, PT], [1, 1]], offset=e,
                                               vector_offset=gi[0:PT, 0:1], indirect_dim=0))
        return gi, h, wcol

    def group(i, mg, yt, txb):
        lds = [load(i, m) for m in range(mg)]
        _expert_tile([h for _, h, _ in lds], [w for _, _, w in lds], w1, s1, w3, s3, w2, s2, yt,
                     txb, pid)
        for m, (gi, _, _) in enumerate(lds):
            y = nl.ndarray((PT, DIM_H), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=y, src=yt[m * PT : (m + 1) * PT, ocol0 : ocol0 + DIM_H])
            tgt = eacc_hbm.ap(pattern=[[DIM, PT], [1, DIM_H]], offset=ocol0,
                              vector_offset=gi[0:PT, 0:1], indirect_dim=0)
            nisa.dma_compute(dst=tgt, srcs=[tgt, y], reduce_op=nl.add)

    if ytile2 is not None and ntmax >= PAIR:
        # whole groups of PAIR tiles, then the rest one at a time
        npair = (ntiles // PAIR) * PAIR
        nl.fori_loop(0, npair, lambda i: group(i, PAIR, ytile2, tx2), step=PAIR)
        nl.fori_loop(npair, ntiles, lambda i: group(i, 1, ytile, tx))
    else:
        nl.fori_loop(0, ntiles, lambda i: group(i, 1, ytile, tx))


def routed_experts(pid, n_local, Tp, x_hbm, wexp_hbm, experts, eacc_hbm, tag=""):
    """Every routed expert into `eacc_hbm` (this core's column half, zeroed here).
    `experts[e]` is expert `e`'s `(w1, s1, w3, s3, w2, s2)`; `tag` keeps the named
    exchange buffer unique when one kernel calls this more than once."""
    zero_cols(eacc_hbm, Tp + PT, pid * DIM_H, (pid + 1) * DIM_H)
    glist = alloc_lists(n_local, Tp)
    build_expert_lists(wexp_hbm, Tp, n_local, glist)
    rampf = partition_ramp()
    ytile = nl.ndarray((PT, DIM), dtype=nl.float32, buffer=nl.private_hbm)
    tx = nl.ndarray((2 * PT, INTER_H), dtype=nl.bfloat16, buffer=nl.shared_hbm, name=f"moe_tx{tag}")
    ytile2 = tx2 = None
    if Tp // PT >= PAIR:
        ytile2 = nl.ndarray((PAIR * PT, DIM), dtype=nl.float32, buffer=nl.private_hbm)
        tx2 = nl.ndarray((2 * PAIR * PT, INTER_H), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                         name=f"moe_tx2{tag}")
    for e in range(n_local):
        nt = expert_tiles(glist, e, Tp)
        run_expert(nt, e, glist, Tp, x_hbm, wexp_hbm, n_local, experts[e], eacc_hbm, rampf,
                   ytile, tx, pid, ytile2=ytile2, tx2=tx2)
