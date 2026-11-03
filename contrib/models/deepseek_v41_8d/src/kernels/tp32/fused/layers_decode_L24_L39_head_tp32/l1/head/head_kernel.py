# SPDX-License-Identifier: Apache-2.0
"""The two ends of the flat scheme's decode step as fused-launch members (`mk_model.run_fused`):
`embed_expand_body` opens the step's first launch (the token embedding and the hyper-
connection expand, `mk_kernels.embed_expand`), `head_greedy_body` closes its last one.

The greedy head is `model.forward`'s tail after the last block -- collapse, final norm,
vocab-parallel head, argmax -- for a decode step whose rows are all greedy, so the head
costs no launch of its own (`mk_kernels.head_greedy`):

    x      = sum_c pre_mix[t, c] * stream[t, c, :]      fp32, bf16 out (`hc_collapse`)
    xn     = RMSNorm(x) * norm_w                        bf16 (`rmsnorm`)
    logits = xn @ head_w.T                              this rank's columns, bf16 -> fp32
    ids[t] = argmax over the full vocabulary

The argmax never gathers the logits (the separate path all-gathered `[T, vocab]` fp32
twice a step: once for the runner, once for the sampler's greedy override): each rank
takes the (max, index) of its own shard -- columns past the real vocabulary pushed to
-1e30 -- and an all_gather of those pairs decides, ties to the lowest rank, i.e. the
lowest vocabulary index, as `argmax` does. The drafter's head
(`draft_kernel`, "head") is the same computation for its draft rows.

`flags` = (vocab_size, kver). Output: `ids [T, 1]` int32.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

from .mk_common import (NORM_EPS, PT, bcast_row, engram_gather, norm_rows, pad_to, rsqrt_newton,
                        sum_squares, tiles_of)
from .mk_compress import bf16_gemm_rows

NEG = -1.0e30
#: The collapse and the final norm on BOTH cores (the same ops on the same rows) straight
#: from the collapse's SBUF tile, each core's normed rows in its own HBM copy for its half of
#: the head GEMM: no xh / xhn round trips through shared HBM and no core barriers on them
#: (core 1 waited for core 0's collapse + norm, ~50 us at T = 6).
HEAD_BOTH_CORES = True


def _half(n, pid):
    tiles = n // PT
    a = (tiles + 1) // 2
    return (0, a * PT) if pid == 0 else (a * PT, n)


def head_greedy_body(stream, pre_mix, norm_w, head_w, vocab_start, groups_world, flags):
    """`ids [T, 1]` int32: the greedy token of every row (see the module docstring).
    `stream [T, hc, dim]` bf16 and `pre_mix [T, hc]` fp32 are the last block's outputs,
    `norm_w [1, dim]` the final norm, `head_w [part, dim]` bf16 this rank's head rows,
    `vocab_start [1, 1]` int32 this rank's first column."""
    vocab_size, _kver = flags
    T, hc, dim = stream.shape
    part = head_w.shape[0]
    assert head_w.shape[1] == dim and part % PT == 0 and T <= PT, (T, part, dim)
    world = len(groups_world[0])
    W8 = max(world, 8)
    Tp = pad_to(T, PT)
    pid = nl.program_id(0)
    rg_world = ncc.ReplicaGroup([list(g) for g in groups_world])
    st = stream.reshape((T, hc * dim))
    rows = tiles_of(T, 0)             # the step's rows: one tile, core 0's
    mine = tiles_of(T, pid)

    ids = nl.ndarray((T, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    xh = nl.ndarray((Tp, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="hd_xh")     # collapsed
    xhn = nl.ndarray((Tp, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="hd_xhn")   # normed
    lg = nl.ndarray((Tp, part), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="hd_lg")
    gv_src = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="hd_gv_src")
    gi_src = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="hd_gi_src")
    gv_dst = nl.ndarray((world * T, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="hd_gv_dst")
    gi_dst = nl.ndarray((world * T, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="hd_gi_dst")

    if HEAD_BOTH_CORES and Tp == PT:
        xhn_c = nl.ndarray((Tp, dim), dtype=nl.bfloat16, buffer=nl.private_hbm)
        ch = 1024 if dim % 1024 == 0 else dim
        for r0, nr in rows:
            pm = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=pm[0:nr, :], src=pre_mix[r0 : r0 + nr, 0:hc])
            acc = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
            for c in range(hc):
                xc = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=xc[0:nr, :], src=st[r0 : r0 + nr, c * dim : (c + 1) * dim])
                if c == 0:
                    nisa.tensor_scalar(dst=acc[0:nr, :], data=xc[0:nr, :], op0=nl.multiply, operand0=pm[0:nr, 0:1],
                                       engine=nisa.engine.vector)
                else:
                    nisa.scalar_tensor_tensor(dst=acc[0:nr, :], data=xc[0:nr, :], op0=nl.multiply,
                                              operand0=pm[0:nr, c : c + 1], op1=nl.add, operand1=acc[0:nr, :])
            t = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=t[0:nr, :], src=acc[0:nr, :])
            # `norm_rows` (chunk 1024) on the bf16 rows
            ss = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            sum_squares(ss, t, nr, dim, ch)
            rstd = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            rstd[0:nr, :] = rsqrt_newton(nl.add(nl.multiply(ss[0:nr, 0:1], 1.0 / dim), NORM_EPS))
            y16 = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(y16, 0.0)
            for c in range(dim // ch):
                lo_, hi_ = c * ch, (c + 1) * ch
                wj = bcast_row(norm_w[0:1, lo_:hi_], ch, nr)
                yv = nl.ndarray((PT, ch), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=yv[0:nr, :], op=nl.copy, data=t[0:nr, lo_:hi_], scale=rstd[0:nr, 0:1])
                nisa.tensor_tensor(dst=yv[0:nr, :], data1=yv[0:nr, :], data2=wj, op=nl.multiply)
                nisa.tensor_copy(dst=y16[0:nr, lo_:hi_], src=yv[0:nr, :])
            # (rows past T zero: the pad rows the GEMM reads)
            nisa.dma_copy(dst=xhn_c[r0 : r0 + PT, :], src=y16)
        lo, hi = _half(part, pid)
        bf16_gemm_rows(xhn_c, head_w, lg, Tp, dim, part, lo, hi)
    # collapse with the last pre_mix (fp32 sum, bf16 out); the pad rows zero
    elif pid == 0:
        for r0, nr in rows:
            pm = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=pm[0:nr, :], src=pre_mix[r0 : r0 + nr, 0:hc])
            acc = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
            for c in range(hc):
                xc = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=xc[0:nr, :], src=st[r0 : r0 + nr, c * dim : (c + 1) * dim])
                if c == 0:
                    nisa.tensor_scalar(dst=acc[0:nr, :], data=xc[0:nr, :], op0=nl.multiply, operand0=pm[0:nr, 0:1],
                                       engine=nisa.engine.vector)
                else:
                    nisa.scalar_tensor_tensor(dst=acc[0:nr, :], data=xc[0:nr, :], op0=nl.multiply,
                                              operand0=pm[0:nr, c : c + 1], op1=nl.add, operand1=acc[0:nr, :])
            y16 = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(y16, 0.0)
            nisa.tensor_copy(dst=y16[0:nr, :], src=acc[0:nr, :])
            nisa.dma_copy(dst=xh[r0 : r0 + PT, :], src=y16)
    if not (HEAD_BOTH_CORES and Tp == PT):
        nisa.core_barrier(data=xh, cores=(0, 1))
        norm_rows(xh, norm_w, xhn, mine, dim, qdq=False, chunk=1024)
        if pid == 0 and T < Tp:
            padh = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(padh, 0.0)
            nisa.dma_copy(dst=xhn[T:Tp, :], src=padh[0 : Tp - T, :])
        nisa.core_barrier(data=xhn, cores=(0, 1))
        lo, hi = _half(part, pid)
        bf16_gemm_rows(xhn, head_w, lg, Tp, dim, part, lo, hi)
    nisa.core_barrier(data=lg, cores=(0, 1))

    # this rank's (max, global index) per row, the vocabulary pad pushed to -1e30
    if pid == 0:
        vs = bcast_row(vocab_start[0:1, 0:1], 1, PT)
        for r0, nr in rows:
            l16 = nl.ndarray((PT, part), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=l16[0:nr, :], src=lg[r0 : r0 + nr, 0:part])
            l32 = nl.ndarray((PT, part), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=l32[0:nr, :], src=l16[0:nr, :])
            ci = nl.ndarray((PT, part), dtype=nl.int32, buffer=nl.sbuf)
            nisa.iota(dst=ci[0:nr, :], pattern=[[1, part]], offset=0, channel_multiplier=0)
            cf = nl.ndarray((PT, part), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=cf[0:nr, :], src=ci[0:nr, :])
            live = nl.ndarray((PT, part), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=live[0:nr, :], data=cf[0:nr, :], op0=nl.add, operand0=vs[0:nr, 0:1],
                               op1=nl.less, operand1=float(vocab_size))
            dead = nl.ndarray((PT, part), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=dead[0:nr, :], data=live[0:nr, :], op0=nl.add, operand0=-1.0,
                               op1=nl.multiply, operand1=-NEG)
            nisa.tensor_tensor(dst=l32[0:nr, :], data1=l32[0:nr, :], data2=dead[0:nr, :], op=nl.add)
            best = nl.ndarray((PT, 8), dtype=nl.float32, buffer=nl.sbuf)
            nisa.max8(dst=best[0:nr, :], src=l32[0:nr, :])
            bidx = nl.ndarray((PT, 8), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.nc_find_index8(dst=bidx[0:nr, :], data=l32[0:nr, :], vals=best[0:nr, :])
            bf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=bf[0:nr, :], src=bidx[0:nr, 0:1])
            gf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=gf[0:nr, :], data1=bf[0:nr, :], data2=vs[0:nr, 0:1], op=nl.add)
            nisa.dma_copy(dst=gv_src[r0 : r0 + nr, 0:1], src=best[0:nr, 0:1])
            nisa.dma_copy(dst=gi_src[r0 : r0 + nr, 0:1], src=gf[0:nr, :])
    nisa.core_barrier(data=gv_src, cores=(0, 1))
    nisa.core_barrier(data=gi_src, cores=(0, 1))
    ncc.all_gather(srcs=[gv_src], dsts=[gv_dst], replica_group=rg_world, collective_dim=0)
    ncc.all_gather(srcs=[gi_src], dsts=[gi_dst], replica_group=rg_world, collective_dim=0)

    # the global argmax: the highest shard max, ties to the lowest rank (lowest index).
    # Rank r's row t is gathered row r * T + t: a row t reads its world values at stride T.
    if pid == 0:
        vrow = nl.ndarray((PT, W8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(vrow, NEG)
        nisa.dma_copy(dst=vrow[0:T, 0:world], src=gv_dst.ap(pattern=[[1, T], [T, world]], offset=0))
        trow = nl.ndarray((PT, W8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(trow, 0.0)
        nisa.dma_copy(dst=trow[0:T, 0:world], src=gi_dst.ap(pattern=[[1, T], [T, world]], offset=0))
        gbest = nl.ndarray((PT, 8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.max8(dst=gbest[0:T, :], src=vrow[0:T, :])
        gidx = nl.ndarray((PT, 8), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.nc_find_index8(dst=gidx[0:T, :], data=vrow[0:T, :], vals=gbest[0:T, :])
        tok = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.nc_n_gather(dst=tok[0:T, :], data=trow[0:T, :], indices=gidx[0:T, 0:1])
        ti = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ti[0:T, :], src=tok[0:T, :])
        nisa.dma_copy(dst=ids[0:T, 0:1], src=ti[0:T, :])
    nisa.core_barrier(data=ids, cores=(0, 1))
    return (ids,)


def embed_expand_body(ids, embed_w, vocab_start, groups_world, flags):
    """`(stream [T, hc, dim] bf16, pre_mix [T, hc] fp32)`: `model.forward`'s head of the
    stack for a decode step -- the step's tokens embedded (vocab-parallel: this rank's
    rows, zero for the others' tokens, all-reduced over the world; one rank holds each
    row, so the bf16 sum is exact) and expanded to the `hc` copies with the one-hot
    `pre_mix` that makes the first block's `hc_pre` return copy 0 (`hc_expand`).
    `ids [T, 1]` int32, `embed_w [part, dim]` bf16 this rank's rows, `vocab_start [1, 1]`
    int32 its first token. `flags` = (hc, kver)."""
    hc, _kver = flags
    T = ids.shape[0]
    part, dim = embed_w.shape
    assert T <= PT, T
    pid = nl.program_id(0)
    rg_world = ncc.ReplicaGroup([list(g) for g in groups_world])
    rows = tiles_of(T, 0)

    stream = nl.ndarray((T, hc, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    pre_mix = nl.ndarray((T, hc), dtype=nl.float32, buffer=nl.shared_hbm)
    e_src = nl.ndarray((T, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="em_src")
    e_dst = nl.ndarray((T, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="em_dst")
    st = stream.reshape((T, hc * dim))

    if pid == 0:
        engram_gather(ids, embed_w, vocab_start, e_src, rows, 1, dim, part)
    nisa.core_barrier(data=e_src, cores=(0, 1))
    ncc.all_reduce(srcs=[e_src], dsts=[e_dst], replica_group=rg_world, op=nl.add)
    if pid == 0:
        for r0, nr in rows:
            x = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=x[0:nr, :], src=e_dst[r0 : r0 + nr, 0:dim])
            for c in range(hc):
                nisa.dma_copy(dst=st[r0 : r0 + nr, c * dim : (c + 1) * dim], src=x[0:nr, :])
            pm = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(pm, 0.0)
            nisa.memset(pm[:, 0:1], 1.0)
            nisa.dma_copy(dst=pre_mix[r0 : r0 + nr, 0:hc], src=pm[0:nr, :])
    nisa.core_barrier(data=stream, cores=(0, 1))
    nisa.core_barrier(data=pre_mix, cores=(0, 1))
    return stream, pre_mix


#: Standalone launches (the unit tests').
head_greedy = nki.jit(head_greedy_body)
embed_expand = nki.jit(embed_expand_body)
