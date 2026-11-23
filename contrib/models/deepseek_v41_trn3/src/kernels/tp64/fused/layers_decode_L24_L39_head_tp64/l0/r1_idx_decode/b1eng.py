# SPDX-License-Identifier: Apache-2.0
"""The engram read-out of a block for one 6-row span, the gate on chip (round 5): the gated stream left
in the b1 block's packed SBUF layout (`b1.block(S_in=...)`) instead of an HBM stream.

    the n-gram rows of this rank's table shard (one 2-D gather, core 0) -> AR-64 (the 6 rows)
    -> the fp8 projection (this rank's 1024-column blocks, each core half) -> AG-8 over the attention group
    -> key / value / gate weight in the packed layout -> per (row, copy): hsq, ksq, sum(h w k) (PE row sums)
    -> gate = sigmoid(sign * sqrt(|dot|)) -> stream' = bf16(stream + gate * value)

Same arithmetic as `mk_common.engram_readout` / `engram_gate_tile` (the masked table rows, the MX
projection of the summed rows, the fp32 gate); the gate's fp32 sums run in a different order."""

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

from . import b1 as B1
from . import b1attn as A
from .mk_common import NORM_EPS, PT

F32 = nl.float32
BF = nl.bfloat16
T6 = A.T6
KC = A.KC
H0 = 64
#: the projection's column blocks (`moe_tp.ENGRAM_BLK`) dealt to the attention group's 8 ranks
EBLK = 1024
EGROUP = 8


def _hw(eng):
    return dict(dge_mode=nisa.dge_mode.hwdge, engine=eng)


def usable(hash_ids, etable, e_wkv4, e_qk, hc, dim):
    """This path's shapes: 6 rows, the sharded projection (`moe_tp.engram_sharded`), whole 1024 blocks."""
    e_full = (hc + 1) * dim
    wloc = e_wkv4.shape[1]
    nk = hash_ids.shape[0] * hash_ids.shape[1]
    return (hash_ids.shape[0] == T6 and PT < nk <= 2 * PT and nk % 2 == 0 and nk // 2 <= PT and wloc < e_full and wloc % EBLK == 0 and wloc % (2 * KC) == 0
            and (hash_ids.shape[1] * etable.shape[1]) % KC == 0 and dim % EBLK == 0
            and e_qk.shape[1] == hc * dim and EGROUP * wloc >= e_full)


def _msum_h(L):
    """`[P, Q] = (row(P) == row(Q)) * (half(P) == half(Q))` fp32 (the 128-partition layout's row sums
    within one half), once a launch."""
    if "msum_h" in L:
        return L["msum_h"]
    m = nl.ndarray((PT, PT), dtype=F32, buffer=nl.sbuf)
    nisa.memset(m, 0.0)
    nisa.tensor_copy(dst=m[0:H0, 0:H0], src=L["msum2"][0:H0, 0:H0])
    nisa.tensor_copy(dst=m[H0:PT, H0:PT], src=L["msum2"][H0:PT, H0:PT])
    L["msum_h"] = m
    return m


def readout(pid, L, S, hash_ids, etable, erow_start, e_wkv4, e_wkv_s, e_qk, rg_world, rg_attn, C, barrier):
    """The gated stream `[128, 2, 512]` bf16 (packed) from the packed input stream `S` (see the module
    docstring). `L`: the launch's b1 dict (constants)."""
    T = T6
    hc = 4
    dim = C * KC
    NP = T * C
    ncol = hash_ids.shape[1]
    ehd = etable.shape[1]
    e_in = ncol * ehd
    wloc = e_wkv4.shape[1]
    sync, act = nisa.engine.sync, nisa.engine.scalar
    # ------------------------------------------------------------ 1. the n-gram rows (core 0)
    emb = nl.ndarray((T, e_in), dtype=BF, buffer=nl.shared_hbm, name="e_emb")
    emb_sum = nl.ndarray((T, e_in), dtype=BF, buffer=nl.shared_hbm, name="e_emb_sum")
    if pid == 0:
        # the 6 x 24 ids as flat k = 24 t + j: the gather's index tile [a, b] = id k = 128 b + a (the 2-D
        # vector_offset read is column-major), its destination [p, g] = row k = 2 p + g (72 partitions x 2)
        nk = T * ncol
        G = 2
        NPd = nk // G
        rs = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=rs, src=erow_start.ap(pattern=[[0, PT], [1, 1]], offset=0), dge_mode=nisa.dge_mode.none)
        rsf = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=rsf, src=rs)
        nrows = float(etable.shape[0])
        ii = nl.ndarray((PT, G), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(ii, 0)
        nisa.dma_copy(dst=ii[:, 0:1], src=hash_ids.ap(pattern=[[1, PT], [1, 1]], offset=0), dge_mode=nisa.dge_mode.none)
        nisa.dma_copy(dst=ii[0 : nk - PT, 1:2], src=hash_ids.ap(pattern=[[1, nk - PT], [1, 1]], offset=PT),
                      dge_mode=nisa.dge_mode.none)
        lf = nl.ndarray((PT, G), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=lf, src=ii)
        nisa.tensor_scalar(dst=lf, data=lf, op0=nl.subtract, operand0=rsf[:, 0:1], op1=nl.maximum, operand1=0.0)
        nisa.tensor_scalar(dst=lf, data=lf, op0=nl.minimum, operand0=nrows - 1.0)
        ji = nl.ndarray((PT, G), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ji, src=lf)
        g = nl.ndarray((PT, G, ehd), dtype=BF, buffer=nl.sbuf)
        nisa.dma_copy(dst=g, src=etable.ap(pattern=[[ehd, PT * G], [1, ehd]], offset=0,
                                           vector_offset=ji[0:PT, 0:G], indirect_dim=0))
        # the mask in the destination layout: (0 <= id - row_start < rows)
        di = nl.ndarray((PT, G), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=di[0:NPd], src=hash_ids.ap(pattern=[[G, NPd], [1, G]], offset=0), dge_mode=nisa.dge_mode.none)
        dl = nl.ndarray((PT, G), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=dl[0:NPd], src=di[0:NPd])
        nisa.tensor_scalar(dst=dl[0:NPd], data=dl[0:NPd], op0=nl.subtract, operand0=rsf[0:NPd, 0:1])
        m0 = nl.ndarray((PT, G), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=m0[0:NPd], data=dl[0:NPd], op0=nl.greater_equal, operand0=0.0)
        mask = nl.ndarray((PT, G), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mask[0:NPd], data=dl[0:NPd], op0=nl.less, operand0=nrows)
        nisa.tensor_tensor(dst=mask[0:NPd], data1=mask[0:NPd], data2=m0[0:NPd], op=nl.multiply)
        e16 = nl.ndarray((PT, G, ehd), dtype=BF, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=e16[0:NPd], data1=g[0:NPd],
                           data2=mask.ap(pattern=[[G, NPd], [1, G], [0, ehd]], offset=0), op=nl.multiply)
        nisa.dma_copy(dst=emb.ap(pattern=[[G * ehd, NPd], [1, G * ehd]], offset=0), src=e16[0:NPd], **_hw(sync))
    barrier(emb)
    ncc.all_reduce(srcs=[emb], dsts=[emb_sum], replica_group=rg_world, op=nl.add)
    # ------------------------------------------------------------ 2. the projection, AG-8
    x16 = nl.ndarray((T, e_in), dtype=BF, buffer=nl.sbuf)
    nisa.dma_copy(dst=x16, src=emb_sum[0:T, :], **_hw(sync))
    xq, xs = A.stage_rows(x16, T, e_in, T)
    parts = A.chunk_parts(e_in)
    half = wloc // 2
    part = nl.ndarray((T, wloc), dtype=BF, buffer=nl.shared_hbm, name="e_part")
    for n0 in range(pid * half, (pid + 1) * half, KC):
        wt, ws = A.load_w(e_wkv4, e_wkv_s, e_in, n0, KC, act if (n0 // KC) % 2 else sync)
        o16 = nl.ndarray((T, KC), dtype=BF, buffer=nl.sbuf)
        A.evac(o16, A.mx_gemm(xq, xs, T, T, parts, wt, ws, 0, KC))
        nisa.dma_copy(dst=part[0:T, n0 : n0 + KC], src=o16, **_hw(sync))
    barrier(part)
    gath = nl.ndarray((EGROUP * T, wloc), dtype=BF, buffer=nl.shared_hbm, name="e_gath")
    ncc.all_gather(srcs=[part], dsts=[gath], replica_group=rg_attn, collective_dim=0)
    # ------------------------------------------------------------ 3. key / value copy-major [5, 6, dim] (HBM)
    # (global block b = slot b % 8, local block b // 8; copy i = blocks 5 i .. 5 i + 4)
    nb_copy = dim // EBLK
    ekv = nl.ndarray(((hc + 1) * T, dim), dtype=BF, buffer=nl.shared_hbm, name="e_kv")
    for i in range(hc + 1):
        if (i % 2) != pid:
            continue
        for m in range(nb_copy):
            b = i * nb_copy + m
            nisa.dma_copy(dst=ekv.ap(pattern=[[dim, T], [1, EBLK]], offset=i * T * dim + m * EBLK),
                          src=gath.ap(pattern=[[wloc, T], [1, EBLK]], offset=(b % EGROUP) * T * wloc + (b // EGROUP) * EBLK),
                          **_hw(sync if m % 2 else act))
    barrier(ekv)
    # the packed key (copy 2 s + h on half h) and value (both halves), the gate weight on 10 partitions
    K = nl.ndarray((PT, 2, KC), dtype=BF, buffer=nl.sbuf)
    V = nl.ndarray((PT, KC), dtype=BF, buffer=nl.sbuf)
    nisa.memset(K, 0.0)
    nisa.memset(V, 0.0)
    for h in range(2):
        for s_ in range(2):
            i = 2 * s_ + h
            nisa.dma_copy(dst=K[h * H0 : h * H0 + NP, s_, :], src=ekv.ap(pattern=[[KC, NP], [1, KC]], offset=i * T * dim),
                          **_hw(sync if s_ else act))
        nisa.dma_copy(dst=V[h * H0 : h * H0 + NP, :], src=ekv.ap(pattern=[[KC, NP], [1, KC]], offset=hc * T * dim),
                      **_hw(sync if h else act))
    w10 = nl.ndarray((PT, hc, KC), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w10[0:C], src=e_qk.ap(pattern=[[KC, C], [dim, hc], [1, KC]], offset=0), dge_mode=nisa.dge_mode.none)
    # replicated onto the 128 partitions (fp32 0/1 matmul, exact): wr[P, i, f] = w[i dim + 512 c(P) + f]
    if "e_rep10" not in L:
        # rep[c, P] = (chunk(P) == c) (the packed layout's chunk ids, -1 on the pad partitions)
        crow = B1._row([[0, 2], [0, T6], [1, C]], NP, -1)
        pc = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=pc, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        pcf = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pcf, src=pc)
        rep = nl.ndarray((PT, PT), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rep[0:C], data=crow[0:C], op0=nl.equal, operand0=pcf[0:C, 0:1])
        L["e_rep10"] = rep
    wr = nl.ndarray((PT, hc, KC), dtype=F32, buffer=nl.sbuf)
    for i in range(hc):
        wp = nl.ndarray((PT, KC), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(dst=wp, stationary=L["e_rep10"][0:C, :], moving=w10[0:C, i, :])
        nisa.tensor_copy(dst=wr[:, i, :], src=wp, engine=act if i % 2 else nisa.engine.vector)
    # ------------------------------------------------------------ 4. the gate statistics, row sums
    st = nl.ndarray((PT, 3, 2), dtype=F32, buffer=nl.sbuf)
    nisa.memset(st, 0.0)
    junk = nl.ndarray((PT, KC), dtype=F32, buffer=nl.sbuf)
    for s_ in range(2):
        nisa.activation(dst=junk, op=nl.square, data=S[:, s_, :], reduce_op=nl.add,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce, reduce_res=st[:, 0, s_ : s_ + 1])
        nisa.activation(dst=junk, op=nl.square, data=K[:, s_, :], reduce_op=nl.add,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce, reduce_res=st[:, 1, s_ : s_ + 1])
    for h in range(2):
        for s_ in range(2):
            i = 2 * s_ + h
            ph = slice(h * H0, h * H0 + H0)
            p1 = nl.ndarray((PT, KC), dtype=F32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=p1[ph], data1=S[ph, s_, :], data2=wr[ph, i, :], op=nl.multiply)
            nisa.tensor_tensor(dst=p1[ph], data1=p1[ph], data2=K[ph, s_, :], op=nl.multiply)
            nisa.tensor_reduce(dst=st[ph, 2, s_ : s_ + 1], op=nl.add, data=p1[ph], axis=(1,), keepdims=True)
    tot = nl.ndarray((PT, 6), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(dst=tot, stationary=_msum_h(L), moving=st.reshape((PT, 6)))
    # ------------------------------------------------------------ 5. the gate, the gated stream
    sums = nl.ndarray((PT, 3, 2), dtype=F32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sums.reshape((PT, 6)), src=tot)
    out = nl.ndarray((PT, 2, KC), dtype=BF, buffer=nl.sbuf)
    nisa.memset(out, 0.0)
    for s_ in range(2):
        hs = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
        ks = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=hs, src=sums[:, 0, s_ : s_ + 1])
        nisa.tensor_copy(dst=ks, src=sums[:, 1, s_ : s_ + 1])
        rstd = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=rstd, data1=A.rstd_of(hs, PT, dim), data2=A.rstd_of(ks, PT, dim), op=nl.multiply)
        dot = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=dot, data1=sums[:, 2, s_ : s_ + 1], data2=rstd, op=nl.multiply)
        nisa.tensor_scalar(dst=dot, data=dot, op0=nl.multiply, operand0=float(dim) ** -0.5)
        mag = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
        nisa.activation(dst=mag, op=nl.abs, data=dot)
        nisa.tensor_scalar(dst=mag, data=mag, op0=nl.maximum, operand0=1e-6)
        nisa.activation(dst=mag, op=nl.sqrt, data=mag)
        sgn = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sgn, data=dot, op0=nl.greater_equal, operand0=0.0)
        nisa.tensor_scalar(dst=sgn, data=sgn, op0=nl.multiply, operand0=2.0, op1=nl.subtract, operand1=1.0)
        gate = nl.ndarray((PT, 1), dtype=F32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gate, data1=mag, data2=sgn, op=nl.multiply)
        nisa.activation(dst=gate, op=nl.sigmoid, data=gate)
        # out = bf16(v * gate + h)
        yv = nl.ndarray((PT, KC), dtype=F32, buffer=nl.sbuf)
        nisa.activation(dst=yv, op=nl.copy, data=V, scale=gate[:, 0:1])
        nisa.tensor_tensor(dst=out[:, s_, :], data1=yv, data2=S[:, s_, :], op=nl.add)
    return out
