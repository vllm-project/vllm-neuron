# SPDX-License-Identifier: Apache-2.0
"""Building blocks shared by the layer and drafter megakernels.

Plain Python helpers traced inside a `@nki.jit` body: none of them is a kernel
on its own. Every helper is written for the two physical cores of a logical
NeuronCore (LNC=2): a helper either takes the row/column range the calling core
owns, or says in its docstring that it runs on both cores unchanged.

Numerics mirror the shipped per-op kernels (`blockquant`, `hc_in`, `hc_attn_out`,
`sparse_attn`), which were validated token for token against the checkpoint's
reference: the values written to the KV caches round-trip through `_qdq_fp8`
(FP8_PEAK = 224), the quantized GEMMs run on the tensor engine's MX matmul
(`mk_mx`, which quantizes their inputs on chip), fp32 PSUM accumulation, bf16
roundings at the same points as the torch path.
"""

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

from . import mk_mx as X
from .blockquant import (  # noqa: F401 - re-exported for the megakernels
    BLK,
    PT,
    _gemm,
    _qdq_fp8,
    _stage_rows,
)

#: `rms_norm_eps`, `hc_sinkhorn_iters`, `hc_eps` -- the config's.
NORM_EPS = 1e-20
SINKHORN_ITERS = 20
HC_EPS = 1e-06
#: RoPE: adjacent element pairs over the last `ROPE_DIM` columns of a head.
ROPE_DIM = 64
ROPE_HALF = ROPE_DIM // 2
#: `sparse_attn`'s empty-slot bias and the slot-tile width it gathers in.
NEG_BOUND = -1.0e30
KTILE = 128
#: A psum bank: 512 fp32 per partition.
PSUM_FP32 = 512


def cdiv(n, d):
    return (n + d - 1) // d


def pad_to(n, m):
    return cdiv(n, m) * m


def tiles_of(rows, pid, cores=2):
    """`(r0, nr)` for the 128-row tiles of `rows` that core `pid` owns (tile `i`
    to core `i % cores`). The partial last tile goes to its owner like any other.
    """
    out = []
    for i in range(cdiv(rows, PT)):
        if i % cores == pid:
            r0 = i * PT
            out.append((r0, min(PT, rows - r0)))
    return out


def rsqrt_newton(m):
    """`1 / sqrt(m)` with one Newton step, as `kernels.rmsnorm` does it."""
    y = nl.rsqrt(m)
    corr = nl.add(nl.multiply(nl.multiply(m, nl.multiply(y, y)), -0.5), 1.5)
    return nl.multiply(y, corr)


def sum_squares(dst, src, nr, width, chunk, junk=None):
    """`dst[0:nr] = sum(src[0:nr, 0:width]^2)` in fp32 on the scalar engine (`junk`: an
    optional `(PT, >= chunk)` fp32 scratch for the discarded squares)."""
    if junk is None:
        junk = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
    nchunks = width // chunk
    for c in range(nchunks):
        last = c == nchunks - 1
        nisa.activation(
            dst=junk[0:nr, 0:chunk],
            op=nl.square,
            data=src[0:nr, c * chunk : (c + 1) * chunk],
            reduce_op=nl.add,
            reduce_cmd=nisa.reduce_cmd.reset_reduce if c == 0 else nisa.reduce_cmd.reduce,
            reduce_res=dst[0:nr, 0:1] if last else None,
        )


def bcast_row(hbm_row, n, nr, dtype=nl.float32):
    """`hbm_row[0:1, 0:n]` broadcast to `nr` partitions as an SBUF tile."""
    return nl.broadcast_to(nl.load(hbm_row[0:1, 0:n], dtype=dtype), (nr, n))


# --------------------------------------------------------------------------- hyper-connections


def sinkhorn(t2, nr, hc):
    """`softmax(rows) + eps`, then the 20 alternating normalizations of `kernels.hc_in`."""

    def rowsum_div(src, dst):
        for i in range(hc):
            lo, hi = i * hc, (i + 1) * hc
            s = nl.sum(src[0:nr, lo:hi], axis=1, keepdims=True)
            nisa.tensor_scalar(
                dst=dst[0:nr, lo:hi], data=src[0:nr, lo:hi],
                op0=nl.multiply, operand0=nl.reciprocal(nl.add(s, HC_EPS)),
            )

    def colsum_div(src, dst):
        s = nl.add(src[0:nr, 0:hc], src[0:nr, hc : 2 * hc])
        for i in range(2, hc):
            s = nl.add(s, src[0:nr, i * hc : (i + 1) * hc])
        d = nl.reciprocal(nl.add(s, HC_EPS))
        for i in range(hc):
            lo, hi = i * hc, (i + 1) * hc
            nisa.tensor_tensor(
                dst=dst[0:nr, lo:hi], data1=src[0:nr, lo:hi], data2=d, op=nl.multiply
            )

    a = nl.ndarray((PT, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
    b = nl.ndarray((PT, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
    for i in range(hc):
        lo, hi = i * hc, (i + 1) * hc
        mx = nl.max(t2[0:nr, lo:hi], axis=1, keepdims=True)
        e = nl.exp(nl.subtract(t2[0:nr, lo:hi], nl.broadcast_to(mx, (nr, hc))))
        s = nl.sum(e, axis=1, keepdims=True)
        nisa.tensor_scalar(
            dst=a[0:nr, lo:hi], data=e,
            op0=nl.multiply, operand0=nl.reciprocal(s), op1=nl.add, operand1=HC_EPS,
        )
    colsum_div(a, b)
    for _ in range(SINKHORN_ITERS - 1):
        rowsum_div(b, a)
        colsum_div(a, b)
    return b


def sinkhorn_batched(t2all, nt, hc, after=None):
    """`sinkhorn` of `nt` row tiles at once: `t2all` is `(PT, nt, hc*hc)` fp32 (tile `t`'s
    `t2` at `[:, t, :]`), the result likewise. Each element goes through the
    instructions `sinkhorn`'s lowering issues for it (DVE reduce over the 4 columns,
    ACT copy with the eps bias, ACT reciprocal, fp32 DVE multiplies / adds), only with
    all tiles and rows in one instruction instead of one per tile (and row)."""
    n = nt * hc
    f32 = nl.float32
    x3 = t2all.reshape((PT, n, hc))

    def recip_rows(src3):
        # 1 / (sum over the 4 columns + eps), per (tile, row): (PT, n, 1)
        sm = nl.ndarray((PT, n, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sm, op=nl.add, data=src3, axis=(2,), keepdims=True)
        r = nl.ndarray((PT, n, 1), dtype=f32, buffer=nl.sbuf)
        if SINKHORN_FUSED_RECIP:
            nisa.activation(dst=r, op=nl.reciprocal, data=sm, bias=HC_EPS, scale=1.0)
            return r
        se = nl.ndarray((PT, n, 1), dtype=f32, buffer=nl.sbuf)
        nisa.activation(dst=se, op=nl.copy, data=sm, bias=HC_EPS, scale=1.0)
        nisa.activation(dst=r, op=nl.reciprocal, data=se)
        return r

    def rowsum_div(src, dst):
        src3 = src.reshape((PT, n, hc))
        r = recip_rows(src3)
        nisa.tensor_tensor(dst=dst.reshape((PT, n, hc)), data1=src3,
                           data2=nl.broadcast_to(r, (PT, n, hc)), op=nl.multiply)

    def colsum_div(src, dst):
        s4 = src.reshape((PT, nt, hc, hc))
        d4 = dst.reshape((PT, nt, hc, hc))
        cs = nl.ndarray((PT, nt, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=cs, data1=s4[:, :, 0, :], data2=s4[:, :, 1, :], op=nl.add)
        for i in range(2, hc):
            nisa.tensor_tensor(dst=cs, data1=cs, data2=s4[:, :, i, :], op=nl.add)
        d = nl.ndarray((PT, nt, hc), dtype=f32, buffer=nl.sbuf)
        if SINKHORN_FUSED_RECIP:
            nisa.activation(dst=d, op=nl.reciprocal, data=cs, bias=HC_EPS, scale=1.0)
        else:
            ce = nl.ndarray((PT, nt, hc), dtype=f32, buffer=nl.sbuf)
            nisa.activation(dst=ce, op=nl.copy, data=cs, bias=HC_EPS, scale=1.0)
            nisa.activation(dst=d, op=nl.reciprocal, data=ce)
        if SINKHORN_ONE_MUL:
            # the hc products in one instruction (d repeated along i by a stride-0 view)
            nisa.tensor_tensor(dst=d4, data1=s4, data2=d.ap(pattern=[[nt * hc, PT], [hc, nt], [0, hc], [1, hc]]),
                               op=nl.multiply)
            return
        for i in range(hc):
            nisa.tensor_tensor(dst=d4[:, :, i, :], data1=s4[:, :, i, :], data2=d, op=nl.multiply)

    a = nl.ndarray((PT, nt, hc * hc), dtype=f32, buffer=nl.sbuf)
    b = nl.ndarray((PT, nt, hc * hc), dtype=f32, buffer=nl.sbuf)
    # softmax over each row's 4 columns, + eps
    mx = nl.ndarray((PT, n, 1), dtype=f32, buffer=nl.sbuf)
    # (`after`: schedule-only anchor of the chain's first instruction; the rest follow it)
    _sched_after(nisa.tensor_reduce(dst=mx, op=nl.max, data=x3, axis=(2,), keepdims=True), after)
    dif = nl.ndarray((PT, n, hc), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=dif, data1=x3, data2=nl.broadcast_to(mx, (PT, n, hc)), op=nl.subtract)
    e = nl.ndarray((PT, n, hc), dtype=f32, buffer=nl.sbuf)
    nisa.activation(dst=e, op=nl.exp, data=dif)
    se = nl.ndarray((PT, n, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=se, op=nl.add, data=e, axis=(2,), keepdims=True)
    re = nl.ndarray((PT, n, 1), dtype=f32, buffer=nl.sbuf)
    nisa.activation(dst=re, op=nl.reciprocal, data=se)
    a3 = a.reshape((PT, n, hc))
    nisa.tensor_tensor(dst=a3, data1=e, data2=nl.broadcast_to(re, (PT, n, hc)), op=nl.multiply)
    nisa.tensor_scalar(dst=a3, data=a3, op0=nl.add, operand0=HC_EPS)
    colsum_div(a, b)
    for _ in range(SINKHORN_ITERS - 1):
        rowsum_div(b, a)
        colsum_div(a, b)
    return b


#: `sinkhorn_batched`'s column normalization multiplies all hc row blocks in one DVE
#: instruction (the column reciprocals broadcast along the block axis) instead of hc.
SINKHORN_ONE_MUL = True
#: `sinkhorn_batched`'s `1 / (sum + eps)`: one ACT reciprocal with the eps as its bias
#: instead of an ACT copy with the bias and then the reciprocal (the 20-iteration chain is
#: latency-bound: one instruction fewer per normalization).
SINKHORN_FUSED_RECIP = True


#: `hc_in` calls run the sinkhorn once over all of the call's row tiles (`sinkhorn_batched`).
HC_SINKHORN_BATCHED = True


#: `hc_in_consts` broadcasts its rows to the partitions by DMA (else DVE copies).
HC_CONSTS_DMA = True


def hc_in_consts(base, scale, norm_w, hc, dim, out_chunk=1024, with_w=True):
    """`hc_in_tile`'s row-invariant operands (`scale`, `base`, `norm_w` rows broadcast
    to 128 partitions, fp32), materialized once for all of a call's tiles."""

    def mat(row, c0, n):
        t = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.sbuf)
        if HC_CONSTS_DMA:
            # a broadcast DMA (partition stride 0, the bf16 -> fp32 widening exact) instead
            # of a DVE copy of a partition-broadcast load
            nisa.dma_copy(dst=t, src=row.ap(pattern=[[0, PT], [1, n]], offset=c0))
        else:
            nisa.tensor_copy(dst=t, src=bcast_row(row[0:1, c0 : c0 + n], n, PT))
        return t

    d = dict(
        s0=mat(scale, 0, 1), s1=mat(scale, 1, 1), s2=mat(scale, 2, 1),
        b0=mat(base, 0, hc), b1=mat(base, hc, hc),
        b2=mat(base, 2 * hc, hc * hc),
        w=[mat(norm_w, c * out_chunk, out_chunk) for c in range(dim // out_chunk)] if with_w else None,
    )
    return d


#: `hc_in_tile` (DMA-transposed mix operand, <= 16 rows) writes its sums of squares'
#: discarded output to a PSUM scratch (one per call, in `consts`): as a write-only SBUF
#: scratch it was given the pre-mix accumulator's address and the ACT chain waited for the
#: DVE pre-mix (a false WAR dependency: in1024_bs1 hc_in(attn) sum of squares started at
#: 38 us instead of 20 us). (With PE transposes -- more rows -- PSUM is busy: in1024_bs10
#: work +10 us, so not there.)
HC_SQ_JUNK = True


#: PE transposes per PSUM -> SBUF copy in `hc_in_tile`'s mix GEMM.
HC_T_BATCH = 4
#: The fp32 GEMMs whose stationary is bf16 data widened to fp32 (`hc_in_tile`'s mix,
#: `gemm_fp32`) skip the stationary's LOW fp32 pass (`is_stationary_onezero`): the
#: tensor engine splits an fp32 stationary into HIGH and LOW halves and runs one
#: weight load + matmul per half; a bf16 value's LOW half is zero, so that pass only
#: accumulates exact zeros.
FP32_BF16_STATIONARY = True


#: `hc_in_tile`'s mix GEMM operand transposed by DMA (`dma_transpose`) instead of by PE
#: transposes (the tensor engine then runs only the matmuls).
HC_MIX_DMA_T = True
#: ... for tiles of at most this many rows (one 16-row transpose per 16 rows and group)
HC_MIX_DMA_T_ROWS = 16
#: The DMA-transposed operand's bf16 -> fp32 casts all on DVE (not alternating with ACT).
HC_MIX_CAST_DVE = True


#: `hc_in_tile` interleaves its pre-mix (DVE) and sum of squares (ACT) with the mix
#: GEMM's transpose copies (all on the other engine meanwhile) instead of issuing them
#: before it (same instructions, program order only).
HC_IN_INTERLEAVE = True


def hc_in_tile(r0, nr, x2, pre_mix, fnT, base, scale, norm_w, hc, dim, mix,
               normed, pre, post, comb, sq_chunk=1024, out_chunk=1024, consts=None, xin=None,
               t2_dst=None, x_after=None, coef_only=False):
    """One 128-row tile of the hyper-connection sublayer-in (`kernels.hc_in`).

    Reads `x2[r0:r0+nr]` (`[T, hc*dim]` bf16 HBM) and `pre_mix`, writes
    `normed[r0:r0+nr]` (bf16) and the fp32 coefficients `pre`/`post`/`comb`.
    `fnT` is the resident `(PT, nk, mix)` fp32 SBUF copy of `fn.T`; `consts` the
    call's `hc_in_consts` (built here when not given). `xin`: the tile's rows already
    in SBUF (`hc` `(PT, dim)` bf16 tiles, e.g. `hc_out_tile(keep=True)`'s), else they
    are loaded from `x2`. `t2_dst`: a `(PT, hc*hc)` SBUF view that receives the
    pre-sinkhorn comb logits instead of running the sinkhorn and storing `comb` here
    (the caller runs `sinkhorn_batched` over its tiles). `coef_only`: only the
    coefficients (`pre`/`post`/`comb` or `t2_dst`) -- no pre-mix or RMS norm, `normed`
    is not written (`hc_in_norm_spread` makes it, on the other core).
    """
    if consts is None:
        consts = hc_in_consts(base, scale, norm_w, hc, dim, out_chunk)
    width = hc * dim
    kpg = dim // PT
    inv_width = 1.0 / width
    inv_dim = 1.0 / dim

    if xin is not None:
        xg = xin
    else:
        xg = []
        for g in range(hc):
            xg.append(nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf))
        for g in range(hc):
            # (`x_after`: schedule-only; a load waiting on rows still to be produced would
            # hold up the DMA queue it sits in)
            _sched_after(nisa.dma_copy(dst=xg[g][0:nr, :], src=x2[r0 : r0 + nr, g * dim : (g + 1) * dim]),
                         x_after)
    pm = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pm[0:nr, :], src=pre_mix[r0 : r0 + nr, 0:hc])

    acc = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
    y16 = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)

    def premix():
        # the first product on ACT (as hc_out's), the accumulating ones on DVE; the last
        # accumulation writes the bf16 rows directly (its fp32 result cast on output, as
        # the separate cast copy did)
        nisa.tensor_scalar(
            dst=acc[0:nr, :], data=xg[0][0:nr, :], op0=nl.multiply,
            operand0=pm[0:nr, 0:1], engine=nisa.engine.scalar,
        )
        for g in range(1, hc):
            nisa.scalar_tensor_tensor(
                dst=y16[0:nr, :] if g == hc - 1 else acc[0:nr, :], data=xg[g][0:nr, :], op0=nl.multiply,
                operand0=pm[0:nr, g : g + 1], op1=nl.add, operand1=acc[0:nr, :],
            )

    ss = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    cj = None
    if HC_SQ_JUNK and HC_MIX_DMA_T and nr <= HC_MIX_DMA_T_ROWS:
        cj = consts.get("junk")
        if cj is None or cj.shape[1] < sq_chunk:
            cj = nl.ndarray((PT, sq_chunk), dtype=nl.float32, buffer=nl.psum)
            consts["junk"] = cj
        junk = cj
    else:
        junk = nl.ndarray((PT, sq_chunk), dtype=nl.float32, buffer=nl.sbuf)
    nsq = dim // sq_chunk

    def sumsq():
        # one uninterrupted ACT accumulation chain
        for g in range(hc):
            for c in range(nsq):
                first = g == 0 and c == 0
                last = g == hc - 1 and c == nsq - 1
                nisa.activation(
                    dst=junk[0:nr, 0:sq_chunk], op=nl.square,
                    data=xg[g][0:nr, c * sq_chunk : (c + 1) * sq_chunk],
                    reduce_op=nl.add,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce if first else nisa.reduce_cmd.reduce,
                    reduce_res=ss[0:nr, 0:1] if last else None,
                )

    rstd = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)

    def norm_ss():
        ss2 = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        sum_squares(ss2, y16, nr, dim, sq_chunk, junk=cj)
        rstd[0:nr, :] = rsqrt_newton(nl.add(nl.multiply(ss2[0:nr, 0:1], inv_dim), NORM_EPS))

    def norm_chunk(c):
        lo, hi = c * out_chunk, (c + 1) * out_chunk
        wj = consts["w"][c][0:nr, 0:out_chunk]
        yv = nl.ndarray((PT, out_chunk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=yv[0:nr, :], op=nl.copy, data=y16[0:nr, lo:hi], scale=rstd[0:nr, 0:1])
        o16 = nl.ndarray((PT, out_chunk), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=o16[0:nr, :], data1=yv[0:nr, :], data2=wj, op=nl.multiply)
        nisa.dma_copy(dst=normed[r0 : r0 + nr, lo:hi], src=o16[0:nr, :])

    nbatch = hc * cdiv(kpg, HC_T_BATCH)
    nout = dim // out_chunk
    split = HC_IN_INTERLEAVE and nbatch >= 4 * (nout + 3)
    # interleaved: the pre-mix, the sum of squares and the whole RMS norm of the mixed
    # rows (none needs the mix GEMM) are issued between the GEMM's transpose batches,
    # so they overlap it instead of forming a DVE / ACT stretch between tiles that
    # stalls the next tile's transpose copies (only the coefficients stay after it)
    work = {}
    if coef_only:
        split = True
        work[2] = [sumsq]
    elif split:
        work[0] = [premix]
        work[2] = [sumsq]
        work[8] = [norm_ss]
        for c in range(nout):
            work.setdefault(12 + 4 * c, []).append(lambda c=c: norm_chunk(c))
    else:
        premix()
        sumsq()

    mixp = nl.ndarray((PT, mix), dtype=nl.float32, buffer=nl.psum)
    # few rows: each group's rows transposed by one DMA transpose (`[a, kpg, 128]` ->
    # `[128, kpg, 16]`; the columns past nr are never read) instead of kpg PE transposes: the tensor engine runs only the matmuls
    # (blocks of 16 rows: the transpose's output steps must be 32-byte aligned)
    a_t = 16
    nq = cdiv(nr, a_t)
    dma_t = HC_MIX_DMA_T and nr <= HC_MIX_DMA_T_ROWS
    xTs = []
    if dma_t:
        for g in range(hc):
            t16 = nl.ndarray((PT, kpg, nq * a_t), dtype=nl.bfloat16, buffer=nl.sbuf)
            for q in range(nq):
                nisa.dma_transpose(dst=t16[0:PT, :, q * a_t : (q + 1) * a_t],
                                   src=xg[g].ap(pattern=[[dim, a_t], [PT, kpg], [1, PT]], offset=q * a_t * dim),
                                   axes=(2, 1, 0))
            t32 = nl.ndarray((PT, kpg, nq * a_t), dtype=nl.float32, buffer=nl.sbuf)
            # (`HC_MIX_CAST_DVE`: all on DVE -- on ACT, a cast waiting for its DMA transpose
            # held up the sum-of-squares chain queued behind it, ~6 us at in1024_bs1)
            nisa.tensor_copy(dst=t32[0:PT, :, 0:nr], src=t16[0:PT, :, 0:nr],
                             engine=nisa.vector_engine if (HC_MIX_CAST_DVE or g % 2 == 0)
                             else nisa.scalar_engine)
            xTs.append(t32)
    nbt = 0
    for g in range(hc):
        for cb in range(0, kpg, HC_T_BATCH):
            nb_ = min(HC_T_BATCH, kpg - cb)
            if not dma_t:
                tps = nl.ndarray((PT, HC_T_BATCH, PT), dtype=nl.bfloat16, buffer=nl.psum)
                for j in range(nb_):
                    c = cb + j
                    nisa.nc_transpose(dst=tps[0:PT, j, 0:nr], data=xg[g][0:nr, c * PT : (c + 1) * PT])
                xT = nl.ndarray((PT, HC_T_BATCH, PT), dtype=nl.float32, buffer=nl.sbuf)
                eng = nisa.scalar_engine if nbt % 2 else nisa.vector_engine
                nisa.tensor_copy(dst=xT[0:PT, 0:nb_, 0:nr], src=tps[0:PT, 0:nb_, 0:nr], engine=eng)
            for f in work.get(nbt, []):
                f()
            nbt += 1
            for j in range(nb_):
                kb = g * kpg + cb + j
                st_ = xTs[g][0:PT, cb + j, 0:nr] if dma_t else xT[0:PT, j, 0:nr]
                nisa.nc_matmul(
                    dst=mixp[0:nr, 0:mix], stationary=st_,
                    moving=fnT[0:PT, kb, 0:mix], accumulate=(kb > 0),
                    is_stationary_onezero=FP32_BF16_STATIONARY,
                )

    rs = rsqrt_newton(nl.add(nl.multiply(ss[0:nr, 0:1], inv_width), NORM_EPS))
    mixv = nl.ndarray((PT, mix), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=mixv[0:nr, :], data=mixp[0:nr, :], op0=nl.multiply, operand0=rs)

    s0 = consts["s0"][0:nr, 0:1]
    s1 = consts["s1"][0:nr, 0:1]
    s2 = consts["s2"][0:nr, 0:1]
    b0 = consts["b0"][0:nr, 0:hc]
    b1 = consts["b1"][0:nr, 0:hc]
    b2 = consts["b2"][0:nr, 0 : hc * hc]

    t0 = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=t0[0:nr, :], data=mixv[0:nr, 0:hc], op0=nl.multiply, operand0=s0,
        op1=nl.add, operand1=b0,
    )
    pre_t = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
    pre_t[0:nr, :] = nl.add(nl.sigmoid(t0[0:nr, :]), HC_EPS)
    nisa.dma_copy(dst=pre[r0 : r0 + nr, 0:hc], src=pre_t[0:nr, :])

    t1 = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=t1[0:nr, :], data=mixv[0:nr, hc : 2 * hc], op0=nl.multiply, operand0=s1,
        op1=nl.add, operand1=b1,
    )
    post_t = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
    post_t[0:nr, :] = nl.multiply(nl.sigmoid(t1[0:nr, :]), 2.0)
    nisa.dma_copy(dst=post[r0 : r0 + nr, 0:hc], src=post_t[0:nr, :])

    t2 = nl.ndarray((PT, hc * hc), dtype=nl.float32, buffer=nl.sbuf) if t2_dst is None else t2_dst
    nisa.scalar_tensor_tensor(
        dst=t2[0:nr, :], data=mixv[0:nr, 2 * hc : 2 * hc + hc * hc],
        op0=nl.multiply, operand0=s2, op1=nl.add, operand1=b2,
    )
    if t2_dst is None:
        comb_t = sinkhorn(t2, nr, hc)
        nisa.dma_copy(dst=comb[r0 : r0 + nr, 0 : hc * hc], src=comb_t[0:nr, :])

    if not split:
        norm_ss()
        for c in range(nout):
            norm_chunk(c)


#: Decode (one row tile of few rows): `hc_in` split between the two cores -- core 0 the
#: pre-mix and the RMS norm (the normed rows the projections / MoE wait for) in the
#: spread layout of `hc_out_spread_*` (nr * S partitions instead of nr), core 1 the
#: coefficients (`hc_in_tile(coef_only=True)`: the whole-width sum of squares, the mix
#: GEMM, pre / post / comb). Same per-element instructions; the row sums (sum of squares
#: of the mixed rows) stay ACT accumulation chains over each row's columns in order.
HC_IN_SPLIT = True
#: With `HC_IN_SPLIT`: the decode hc_outs (spread layout) split by columns between the
#: cores (each core its half of every hc group's columns; elementwise, so exact).
HC_OUT_SPLIT = True
#: Decode (one row tile): the q and o ropes split by whole heads between the cores (each
#: core its GEMM's half of the q columns, then one barrier; o's halves after its barrier).
ROPE_COL_SPLIT = True


def hc_in_norm_spread(nr, x2, pre_mix, norm_w, hc, dim, normed, sq_chunk=1024, x_after=None):
    """`hc_in_tile`'s pre-mix and RMS norm for rows `0 .. nr` (one tile) in the spread
    layout (`spread_of`): `y = sum_g pre_mix[:, g] * x[:, g, :]` (fp32, bf16 on its last
    add), `normed = (y * rstd) * norm_w` with `rstd` from the sum of squares of the bf16
    `y` rows -- computed in the row layout (the rows written to a private HBM scratch
    and read back) so each row's accumulation is the same ACT chain."""
    S = spread_of(nr, dim)
    W = dim // S
    cw = _spread_chunk(W)
    P = nr * S
    width = hc * dim
    f32, bf16 = nl.float32, nl.bfloat16
    mode, groups = _sp_parts(nr, S)
    gn = S if mode == "t" else nr
    # the norm weight in the spread layout (partition (t, s): columns s*W .. +W), fp32
    # (the bf16 -> fp32 widening in the DMA, as `hc_in_consts`)
    ws = nl.ndarray((PT, W), dtype=f32, buffer=nl.sbuf)
    for p0, q in groups:
        if mode == "t":
            src = norm_w.ap(pattern=[[W, S], [1, W]], offset=0)
        else:
            src = norm_w.ap(pattern=[[0, nr], [1, W]], offset=q * W)
        nisa.dma_copy(dst=ws[p0 : p0 + gn, :], src=src)
    pm = nl.ndarray((PT, hc), dtype=f32, buffer=nl.sbuf)
    for p0, q in groups:
        nisa.dma_copy(dst=pm[p0 : p0 + gn, :], src=_sp_bcast(mode, q, nr, S, pre_mix, hc))
    nch = W // cw
    y16 = nl.ndarray((PT, W), dtype=bf16, buffer=nl.sbuf)
    for c in range(nch):
        xs = nl.ndarray((PT, hc, cw), dtype=bf16, buffer=nl.sbuf)
        for k, (p0, q) in enumerate(groups):
            _sched_after(_sp_dma(mode, q, nr, S, W, x2, width, c * cw, cw, dst=xs[p0 : p0 + gn, 0:hc, 0:cw],
                                 mid=(dim, hc), eng=_sp_eng(k)), x_after)
        acc = nl.ndarray((PT, cw), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=acc[0:P, :], data=xs[0:P, 0, :], op0=nl.multiply,
                           operand0=pm[0:P, 0:1], engine=nisa.engine.scalar)
        for g in range(1, hc):
            nisa.scalar_tensor_tensor(
                dst=y16[0:P, c * cw : (c + 1) * cw] if g == hc - 1 else acc[0:P, :], data=xs[0:P, g, :],
                op0=nl.multiply, operand0=pm[0:P, g : g + 1], op1=nl.add, operand1=acc[0:P, :],
            )
    # the mixed rows' sum of squares in the row layout
    yh = nl.ndarray((PT, dim), dtype=bf16, buffer=nl.private_hbm)
    for k, (p0, q) in enumerate(groups):
        _sp_dma(mode, q, nr, S, W, yh, dim, 0, W, src=y16[p0 : p0 + gn, 0:W], eng=_sp_eng(k))
    yr = nl.ndarray((PT, dim), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=yr[0:nr, :], src=yh[0:nr, :])
    junk = nl.ndarray((PT, sq_chunk), dtype=f32, buffer=nl.psum)
    ss2 = nl.ndarray((PT, 1), dtype=f32, buffer=nl.sbuf)
    sum_squares(ss2, yr, nr, dim, sq_chunk, junk=junk)
    rstd = nl.ndarray((PT, 1), dtype=f32, buffer=nl.sbuf)
    rstd[0:nr, :] = rsqrt_newton(nl.add(nl.multiply(ss2[0:nr, 0:1], 1.0 / dim), NORM_EPS))
    rh = nl.ndarray((PT, 1), dtype=f32, buffer=nl.private_hbm)
    nisa.dma_copy(dst=rh[0:nr, :], src=rstd[0:nr, :])
    rs = nl.ndarray((PT, 1), dtype=f32, buffer=nl.sbuf)
    for k, (p0, q) in enumerate(groups):
        e = _sp_eng(k)
        kw = {} if e is None else dict(dge_mode=nisa.dge_mode.hwdge, engine=e)
        nisa.dma_copy(dst=rs[p0 : p0 + gn, :], src=_sp_bcast(mode, q, nr, S, rh, 1), **kw)
    last = None
    for c in range(nch):
        cols = slice(c * cw, (c + 1) * cw)
        yv = nl.ndarray((PT, cw), dtype=f32, buffer=nl.sbuf)
        nisa.activation(dst=yv[0:P, :], op=nl.copy, data=y16[0:P, cols], scale=rs[0:P, 0:1])
        o16 = nl.ndarray((PT, cw), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=o16[0:P, :], data1=yv[0:P, :], data2=ws[0:P, cols], op=nl.multiply)
        for k, (p0, q) in enumerate(groups):
            last = _sp_dma(mode, q, nr, S, W, normed, dim, c * cw, cw, src=o16[p0 : p0 + gn, 0:cw], eng=_sp_eng(k))
    return last


FN_T_HWDGE = True


def load_fn_t(fn_t, width, mix):
    """`fn.T` (`[hc*dim, mix]` fp32 HBM) resident as `(PT, nk, mix)` fp32 SBUF."""
    nk = width // PT
    fnT = nl.ndarray((PT, nk, mix), dtype=nl.float32, buffer=nl.sbuf)
    if FN_T_HWDGE:
        # (on the sync engine's hardware DGE queue: as a GpSimd DMA the 2 MB load sat in
        # GpSimd's queue between the projections' / attention's DMAs)
        nisa.dma_copy(dst=fnT, src=fn_t.ap(pattern=[[mix, PT], [PT * mix, nk], [1, mix]]),
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    else:
        nisa.dma_copy(dst=fnT, src=fn_t.ap(pattern=[[mix, PT], [PT * mix, nk], [1, mix]]))
    return fnT


#: `hc_out_tile`'s `comb * res` terms `i` (of 1..hc-1) computed as an ACT product plus a
#: GpSimd fp32 add instead of one DVE scalar_tensor_tensor (the same two fp32 roundings).
HC_OUT_OFFLOAD_I = ()


def hc_out_tile(r0, nr, x_hbm, res_hbm, post_hbm, comb_hbm, y_hbm, hc, dim, chunk=1024,
                x_dtype=nl.float32, keep=False, acc_first=False, acc_only=False, accs_in=None,
                after=None):
    """One 128-row tile of `hc_post` (`kernels.hc_attn_out`):
    `y[t, j, :] = post[t, j] * x[t, :] + sum_i comb[t, i, j] * res[t, i, :]`.

    `x_hbm` is the sublayer output `[T, dim]` (fp32 or bf16 -- it is widened to fp32
    for the products either way, matching `out.to(x.dtype)` followed by the fp32
    kernel math when `x_dtype` is bf16 and the all-reduced fp32 partial when fp32),
    `res_hbm` the `[T, hc*dim]` bf16 residual stream, `y_hbm` the `[T, hc*dim]` bf16 output.
    `keep`: the output rows also stay in SBUF and are returned (`hc` `(PT, dim)` bf16
    tiles, `hc_in_tile`'s `xin`). `acc_first`: every chunk's `sum_i comb * res` is
    issued before the first `x` load (they do not depend on `x`, e.g. an all-reduce
    result still in flight), then the `post * x` terms. `acc_only`: only those
    accumulators, returned (SBUF); `accs_in`: a previous `acc_only` call's, so only the
    `post * x` terms and the stores run here (a caller pipelines tiles with them).
    """
    # `after`: an instruction (the all-reduce whose result `x_hbm` is) every op here is
    # scheduled after (schedule-only): the x-independent prework then fills the
    # all-reduce wait instead of delaying the all-reduce's trigger
    post = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
    _sched_after(nisa.dma_copy(dst=post[0:nr, :], src=post_hbm[r0 : r0 + nr, 0:hc]), after)
    comb = nl.ndarray((PT, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
    _sched_after(nisa.dma_copy(dst=comb[0:nr, :], src=comb_hbm[r0 : r0 + nr, 0 : hc * hc]), after)
    ys = [nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(hc)] if keep else None
    nch = dim // chunk

    def acc_of(c):
        col0 = c * chunk
        rg = nl.ndarray((PT, hc, chunk), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i in range(hc):
            _sched_after(nisa.dma_copy(
                dst=rg[0:nr, i, :],
                src=res_hbm[r0 : r0 + nr, i * dim + col0 : i * dim + col0 + chunk],
            ), after)
        acc = nl.ndarray((PT, hc, chunk), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(hc):
            _sched_after(nisa.tensor_scalar(
                dst=acc[0:nr, j, :], data=rg[0:nr, 0, :], op0=nl.multiply,
                operand0=comb[0:nr, j : j + 1], engine=nisa.engine.scalar,
            ), after)
            for i in range(1, hc):
                col = i * hc + j
                if i in HC_OUT_OFFLOAD_I:
                    # the product on ACT, the fp32 add on GpSimd (DVE is the busy engine)
                    t = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=t[0:nr, :], data=rg[0:nr, i, :], op0=nl.multiply,
                                       operand0=comb[0:nr, col : col + 1], engine=nisa.engine.scalar)
                    nisa.tensor_tensor(dst=acc[0:nr, j, :], data1=t[0:nr, :], data2=acc[0:nr, j, :],
                                       op=nl.add, engine=nisa.engine.gpsimd)
                    continue
                _sched_after(nisa.scalar_tensor_tensor(
                    dst=acc[0:nr, j, :], data=rg[0:nr, i, :], op0=nl.multiply,
                    operand0=comb[0:nr, col : col + 1], op1=nl.add, operand1=acc[0:nr, j, :],
                ), after)
        return acc

    def out_of(c, acc):
        col0 = c * chunk
        xg = nl.ndarray((PT, chunk), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=xg[0:nr, :], src=x_hbm[r0 : r0 + nr, col0 : col0 + chunk])
        og = None if keep else nl.ndarray((PT, hc, chunk), dtype=nl.bfloat16, buffer=nl.sbuf)

        def og_j(j):
            return ys[j][0:nr, col0 : col0 + chunk] if keep else og[0:nr, j, :]

        for j in range(hc):
            nisa.scalar_tensor_tensor(
                dst=og_j(j), data=xg[0:nr, :], op0=nl.multiply,
                operand0=post[0:nr, j : j + 1], op1=nl.add, operand1=acc[0:nr, j, :],
            )
        for j in range(hc):
            nisa.dma_copy(
                dst=y_hbm[r0 : r0 + nr, j * dim + col0 : j * dim + col0 + chunk],
                src=og_j(j),
            )

    if acc_only:
        return [acc_of(c) for c in range(nch)]
    if accs_in is not None:
        for c in range(nch):
            out_of(c, accs_in[c])
    elif acc_first:
        accs = [acc_of(c) for c in range(nch)]
        for c in range(nch):
            out_of(c, accs[c])
    else:
        for c in range(nch):
            out_of(c, acc_of(c))
    return ys


#: hc_out of a single row tile (decode): each row's `dim` columns spread over S
#: partitions, so the elementwise products run on nr * S partitions instead of nr (the
#: same per-element instructions; only the layout differs). Partition of (row t, column
#: slice s): t * S + s when S >= nr (one DMA per row), else s * nr + t (one per slice).
HC_OUT_SPREAD = True
HC_SPREAD_MIN = 2


def spread_of(nr, dim):
    """Partitions per row for `hc_out_spread_*` (1: not worth it)."""
    if nr <= 0 or PT // nr < HC_SPREAD_MIN:
        return 1
    for s in range(PT // nr, 0, -1):
        if dim % s == 0 and (dim // s) % 32 == 0:
            return s
    return 1


def _spread_chunk(w):
    for cand in (1024, 768, 640, 512, 384, 320, 256, 128):
        if cand <= w and w % cand == 0:
            return cand
    return w


def _sp_parts(nr, S):
    """`[(p0, q, nrows_or_S)]`: the DMA groups of the spread layout -- per row t
    (partitions t*S .. +S, the row's S slices) or per slice s (partitions s*nr .. +nr)."""
    if S >= nr:
        return "t", [(t * S, t) for t in range(nr)]
    return "s", [(s * nr, s) for s in range(S)]


def _sp_dma(mode, q, nr, S, W, hbm, row_stride, col0, n, dst=None, src=None, mid=None, eng=None):
    """One DMA group between the spread SBUF partitions and the HBM rows: element
    (row, slice, k) of the group at `hbm[row * row_stride + col0 + slice * W + k]`
    (`mid = (stride, count)`: a middle dimension, e.g. the hc column groups). `eng`: on
    that engine's hardware DGE queue."""
    m = [list(mid)] if mid is not None else []
    if mode == "t":
        ap = hbm.ap(pattern=[[W, S]] + m + [[1, n]], offset=q * row_stride + col0)
    else:
        ap = hbm.ap(pattern=[[row_stride, nr]] + m + [[1, n]], offset=q * W + col0)
    kw = {} if eng is None else dict(dge_mode=nisa.dge_mode.hwdge, engine=eng)
    if dst is not None:
        return nisa.dma_copy(dst=dst, src=ap, **kw)
    return nisa.dma_copy(dst=ap, src=src, **kw)


#: `hc_in_norm_spread`'s per-group DMAs (one per row) alternate between the SP and ACT
#: hardware DGE queues (one queue issued them one after another: ~1.5 us each).
HC_SP_DMA_ALT = True
#: ... and the spread hc_outs' (`hc_out_spread_*`).
HC_OUT_DMA_ALT = True


def _sp_eng(k):
    if not HC_SP_DMA_ALT:
        return None
    return nisa.engine.sync if k % 2 == 0 else nisa.engine.scalar


def _sp_bcast(mode, q, nr, S, hbm, n):
    """Source AP of the per-row coefficient rows (`hbm` `[T, n]`) for a DMA group."""
    if mode == "t":
        return hbm.ap(pattern=[[0, S], [1, n]], offset=q * n)
    return hbm.ap(pattern=[[n, nr], [1, n]], offset=0)


def hc_out_spread_acc(nr, res_hbm, post_hbm, comb_hbm, hc, dim, after=None, cols=None):
    """`hc_out_tile`'s x-independent half (`post` and the `sum_i comb * res`
    accumulators) for rows `0 .. nr` in the spread layout (`spread_of`); every op is
    scheduled after `after`. Returns the state `hc_out_spread_out` takes. `cols = (c0,
    n)`: only columns `c0 .. c0+n` of each of the hc groups (e.g. one core's half)."""
    c0, dimc = cols if cols is not None else (0, dim)
    S = spread_of(nr, dimc)
    W = dimc // S
    cw = _spread_chunk(W)
    P = nr * S
    width = hc * dim
    mode, groups = _sp_parts(nr, S)
    gn = S if mode == "t" else nr
    post = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
    comb = nl.ndarray((PT, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
    for p0, q in groups:
        _sched_after(nisa.dma_copy(dst=post[p0 : p0 + gn, :], src=_sp_bcast(mode, q, nr, S, post_hbm, hc)),
                     after)
        _sched_after(nisa.dma_copy(dst=comb[p0 : p0 + gn, :],
                                   src=_sp_bcast(mode, q, nr, S, comb_hbm, hc * hc)), after)
    accs = []
    last = None
    for c in range(W // cw):
        rg = nl.ndarray((PT, hc, cw), dtype=nl.bfloat16, buffer=nl.sbuf)
        for k, (p0, q) in enumerate(groups):
            last = _sp_dma(mode, q, nr, S, W, res_hbm, width, c0 + c * cw, cw, dst=rg[p0 : p0 + gn, 0:hc, 0:cw],
                           mid=(dim, hc), eng=_sp_eng(k) if HC_OUT_DMA_ALT else None)
            _sched_after(last, after)
        acc = nl.ndarray((PT, hc, cw), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(hc):
            _sched_after(nisa.tensor_scalar(
                dst=acc[0:P, j, :], data=rg[0:P, 0, :], op0=nl.multiply,
                operand0=comb[0:P, j : j + 1], engine=nisa.engine.scalar,
            ), after)
            for i in range(1, hc):
                col = i * hc + j
                _sched_after(nisa.scalar_tensor_tensor(
                    dst=acc[0:P, j, :], data=rg[0:P, i, :], op0=nl.multiply,
                    operand0=comb[0:P, col : col + 1], op1=nl.add, operand1=acc[0:P, j, :],
                ), after)
        accs.append(acc)
    return dict(S=S, W=W, cw=cw, P=P, post=post, accs=accs, last=last, c0=c0, sub=dimc != dim)


def hc_out_spread_out(st, nr, x_hbm, y_hbm, hc, dim):
    """`hc_out_tile`'s x-dependent half for `hc_out_spread_acc`'s state: `y = post * x +
    acc` (x: the `[nr, dim]` fp32 sublayer output, read into bf16 by the DMA as
    `hc_out_tile` does), stored to the `[T, hc*dim]` bf16 `y_hbm`. The x loads are
    scheduled after the accumulators' loads (an x load waiting on an all-reduce would
    hold up the DMA queue it sits in)."""
    S, W, cw, P, post = st["S"], st["W"], st["cw"], st["P"], st["post"]
    c0 = st.get("c0", 0)
    width = hc * dim
    mode, groups = _sp_parts(nr, S)
    gn = S if mode == "t" else nr
    last = None
    for c, acc in enumerate(st["accs"]):
        xg = nl.ndarray((PT, cw), dtype=nl.bfloat16, buffer=nl.sbuf)
        if mode == "t" and not st.get("sub"):
            # (row stride dim = S * W: one uniform-stride DMA)
            _sched_after(nisa.dma_copy(dst=xg[0:P, :], src=x_hbm.ap(pattern=[[W, P], [1, cw]], offset=c * cw)),
                         st["last"])
        else:
            for k, (p0, q) in enumerate(groups):
                # (no HWDGE here: the fp32 -> bf16 cast is a software-DGE DMA)
                _sched_after(_sp_dma(mode, q, nr, S, W, x_hbm, dim, c0 + c * cw, cw, dst=xg[p0 : p0 + gn, :]),
                             st["last"])
        og = nl.ndarray((PT, hc, cw), dtype=nl.bfloat16, buffer=nl.sbuf)
        for j in range(hc):
            nisa.scalar_tensor_tensor(
                dst=og[0:P, j, :], data=xg[0:P, :], op0=nl.multiply,
                operand0=post[0:P, j : j + 1], op1=nl.add, operand1=acc[0:P, j, :],
            )
        for k, (p0, q) in enumerate(groups):
            last = _sp_dma(mode, q, nr, S, W, y_hbm, width, c0 + c * cw, cw, src=og[p0 : p0 + gn, 0:hc, 0:cw],
                           mid=(dim, hc), eng=_sp_eng(k) if HC_OUT_DMA_ALT else None)
    return last


# --------------------------------------------------------------------------- norms, rope, quant


def norm_rows(src_hbm, w_hbm, dst_hbm, tiles, n, qdq, chunk=None):
    """RMSNorm of `src_hbm[r0:r0+nr, 0:n]` (bf16) into `dst_hbm` (bf16) for the given
    row tiles, optionally followed by the fp8 activation round trip (`qdq`).
    """
    chunk = chunk or n
    inv_n = 1.0 / n
    # the weight rows broadcast once, not per row tile
    wjs = [bcast_row(w_hbm[0:1, c * chunk : (c + 1) * chunk], chunk, PT) for c in range(n // chunk)]
    for r0, nr in tiles:
        t = nl.ndarray((PT, n), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=t[0:nr, :], src=src_hbm[r0 : r0 + nr, 0:n])
        ss = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        sum_squares(ss, t, nr, n, chunk)
        rstd = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        rstd[0:nr, :] = rsqrt_newton(nl.add(nl.multiply(ss[0:nr, 0:1], inv_n), NORM_EPS))
        y16 = nl.ndarray((PT, n), dtype=nl.bfloat16, buffer=nl.sbuf)
        if qdq and nr < PT:
            nisa.memset(y16, 0.0)
        for c in range(n // chunk):
            lo, hi = c * chunk, (c + 1) * chunk
            wj = wjs[c][0:nr, :]
            yv = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=yv[0:nr, :], op=nl.copy, data=t[0:nr, lo:hi], scale=rstd[0:nr, 0:1])
            nisa.tensor_tensor(dst=y16[0:nr, lo:hi], data1=yv[0:nr, :], data2=wj, op=nl.multiply)
        if qdq:
            out = _qdq_fp8(y16, PT, n)
            nisa.dma_copy(dst=dst_hbm[r0 : r0 + nr, 0:n], src=out[0:nr, :])
        else:
            nisa.dma_copy(dst=dst_hbm[r0 : r0 + nr, 0:n], src=y16[0:nr, :])


def load_freqs(freqs_hbm, row_pos_hbm, r0, nr):
    """`(cos, sin)` fp32 `(PT, 32)` tiles for rows `r0..r0+nr`, gathered from the
    `[max_pos, 64]` (cos | sin) table at the rows' absolute positions.
    """
    pos = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pos[0:nr, :], src=row_pos_hbm[r0 : r0 + nr, 0:1])
    cs = nl.ndarray((PT, 2 * ROPE_HALF), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=cs[0:nr, :],
        src=freqs_hbm.ap(
            pattern=[[2 * ROPE_HALF, nr], [1, 2 * ROPE_HALF]],
            vector_offset=pos[0:nr, 0:1], indirect_dim=0,
        ),
    )
    return cs[0:nr, 0:ROPE_HALF], cs[0:nr, ROPE_HALF : 2 * ROPE_HALF]


def rope_inplace(tile, nr, col0, cos, sin, inverse):
    """Rotate the `ROPE_DIM` columns at `col0` of the bf16 SBUF `tile` in place,
    adjacent pairs as complex numbers (`model.apply_rotary_emb`). `cos`/`sin` are
    `(nr, 32)` fp32. All four products are formed before either half is written
    back, because `a`/`b` are stride-2 views of the columns being rewritten.
    """
    seg = tile[0:nr, col0 : col0 + ROPE_DIM].reshape((nr, ROPE_HALF, 2))
    a = seg[0:nr, 0:ROPE_HALF, 0]
    b = seg[0:nr, 0:ROPE_HALF, 1]

    def mulf(x, z):
        r = nl.ndarray((PT, ROPE_HALF), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=r[0:nr, :], data1=x, data2=z, op=nl.multiply)
        return r

    ac = mulf(a, cos)
    bs = mulf(b, sin)
    as_ = mulf(a, sin)
    bc = mulf(b, cos)
    o = nl.ndarray((PT, ROPE_HALF, 2), dtype=nl.bfloat16, buffer=nl.sbuf)
    if inverse:
        nisa.tensor_tensor(dst=o[0:nr, :, 0], data1=ac[0:nr, :], data2=bs[0:nr, :], op=nl.add)
        nisa.tensor_tensor(dst=o[0:nr, :, 1], data1=bc[0:nr, :], data2=as_[0:nr, :], op=nl.subtract)
    else:
        nisa.tensor_tensor(dst=o[0:nr, :, 0], data1=ac[0:nr, :], data2=bs[0:nr, :], op=nl.subtract)
        nisa.tensor_tensor(dst=o[0:nr, :, 1], data1=as_[0:nr, :], data2=bc[0:nr, :], op=nl.add)
    nisa.tensor_copy(dst=tile[0:nr, col0 : col0 + ROPE_DIM], src=o[0:nr].reshape((nr, ROPE_DIM)))


def rope_rows(src_hbm, dst_hbm, tiles, width, head, nheads, freqs_hbm, row_pos_hbm,
              inverse, qdq=False, freqs_cache=None, col0=0):
    """Rope the last `ROPE_DIM` columns of each of `nheads` heads of width `head`
    in `src_hbm[rows, width]` (bf16), optionally fp8-round-trip the rows, and write
    `dst_hbm`. `src_hbm` and `dst_hbm` may be the same tensor. `freqs_cache`: `{r0:
    (cos, sin)}` already gathered (`load_freqs`) for the tiles. `col0`: the rows' columns
    `col0 .. col0+width` only (whole heads, e.g. one core's half)."""
    for r0, nr in tiles:
        t = nl.ndarray((PT, width), dtype=nl.bfloat16, buffer=nl.sbuf)
        if qdq and nr < PT:
            nisa.memset(t, 0.0)
        nisa.dma_copy(dst=t[0:nr, :], src=src_hbm[r0 : r0 + nr, col0 : col0 + width])
        if freqs_cache is not None and r0 in freqs_cache:
            cos, sin = freqs_cache[r0]
        else:
            cos, sin = load_freqs(freqs_hbm, row_pos_hbm, r0, nr)
        for h in range(nheads):
            rope_inplace(t, nr, h * head + head - ROPE_DIM, cos, sin, inverse)
        if qdq:
            t = _qdq_fp8(t, PT, width)
        nisa.dma_copy(dst=dst_hbm[r0 : r0 + nr, col0 : col0 + width], src=t[0:nr, :])


# --------------------------------------------------------------------------- GEMMs (fp32 router)


#: `gemm_fp32` (the router) with at most 16 rows: operand transposed by DMA, not the PE.
GEMM_FP32_DMA_T = True


def gemm_fp32(x_hbm, w_t_hbm, out_hbm, tiles, k, n, n0=0):
    """`out[rows, n0:n0+n] = x[rows, k] (bf16) @ w_t[k, n0:n0+n] (fp32)` with **fp32
    operands** on the tensor engine (the router's `matmul(x.float(), w.float().t())`).
    The `w_t` columns are staged resident as `(PT, k//PT, n)`; `n <= 512`.
    """
    nk = k // PT
    ncol = w_t_hbm.shape[1]
    wT = nl.ndarray((PT, nk, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=wT, src=w_t_hbm.ap(pattern=[[ncol, PT], [PT * ncol, nk], [1, n]], offset=n0))
    for r0, nr in tiles:
        x = nl.ndarray((PT, k), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=x[0:nr, :], src=x_hbm[r0 : r0 + nr, 0:k])
        acc = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.psum)
        if GEMM_FP32_DMA_T and nr <= 16:
            # few rows: the operand transposed by one DMA transpose (as `hc_in_tile`'s mix
            # GEMM; columns past nr never read), the tensor engine runs only the matmuls
            t16 = nl.ndarray((PT, nk, 16), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_transpose(dst=t16, src=x.ap(pattern=[[k, 16], [PT, nk], [1, PT]], offset=0),
                               axes=(2, 1, 0))
            t32 = nl.ndarray((PT, nk, 16), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=t32[0:PT, :, 0:nr], src=t16[0:PT, :, 0:nr])
            for kb in range(nk):
                nisa.nc_matmul(dst=acc[0:nr, 0:n], stationary=t32[0:PT, kb, 0:nr],
                               moving=wT[0:PT, kb, 0:n], accumulate=(kb > 0),
                               is_stationary_onezero=FP32_BF16_STATIONARY)
        for kb in range(nk if not (GEMM_FP32_DMA_T and nr <= 16) else 0):
            # four transposes per PSUM tile, one copy (bf16 -> fp32, exact) for the four
            if kb % 4 == 0:
                nb_ = min(4, nk - kb)
                tps = nl.ndarray((PT, 4, PT), dtype=nl.bfloat16, buffer=nl.psum)
                for j in range(nb_):
                    nisa.nc_transpose(dst=tps[0:PT, j, 0:nr],
                                      data=x[0:nr, (kb + j) * PT : (kb + j + 1) * PT])
                xT4 = nl.ndarray((PT, 4, PT), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=xT4[0:PT, 0:nb_, 0:nr], src=tps[0:PT, 0:nb_, 0:nr])
            xT = xT4[0:PT, kb % 4, :]
            nisa.nc_matmul(dst=acc[0:nr, 0:n], stationary=xT[0:PT, 0:nr],
                           moving=wT[0:PT, kb, 0:n], accumulate=(kb > 0),
                           is_stationary_onezero=FP32_BF16_STATIONARY)
        o = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o[0:nr, :], src=acc[0:nr, :])
        nisa.dma_copy(dst=out_hbm[r0 : r0 + nr, n0 : n0 + n], src=o[0:nr, :])


# --------------------------------------------------------------------------- gathers / scatters


def scatter_rows(src_hbm, slots_hbm, cache_hbm, tiles, n):
    """`cache[slots[r]] = src[r]` for the rows of `tiles` (bf16, `n` columns)."""
    for r0, nr in tiles:
        t = nl.ndarray((PT, n), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=t[0:nr, :], src=src_hbm[r0 : r0 + nr, 0:n])
        ix = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ix[0:nr, :], src=slots_hbm[r0 : r0 + nr, 0:1])
        nisa.dma_copy(
            dst=cache_hbm.ap(pattern=[[n, nr], [1, n]], vector_offset=ix[0:nr, 0:1], indirect_dim=0),
            src=t[0:nr, :],
        )


def gather_rows(cache_hbm, slots_hbm, dst_hbm, tiles, n, dtype=nl.bfloat16, pair=False):
    """`dst[r] = cache[slots[r]]` for the rows of `tiles` (`n` columns); `pair`: stored
    in the pair-interleaved column order (`kv_pair_layout`)."""
    for r0, nr in tiles:
        ix = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ix[0:nr, :], src=slots_hbm[r0 : r0 + nr, 0:1])
        t = nl.ndarray((PT, n), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=t[0:nr, :],
            src=cache_hbm.ap(pattern=[[n, nr], [1, n]], vector_offset=ix[0:nr, 0:1], indirect_dim=0),
        )
        if pair:
            npair = n // (2 * PT)
            tp = nl.ndarray((PT, npair, PT, 2), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=tp[0:nr, :, :, :],
                             src=t.ap(pattern=[[n, nr], [2 * PT, npair], [1, PT], [PT, 2]]))
            t = tp.reshape((PT, n))
        nisa.dma_copy(dst=dst_hbm[r0 : r0 + nr, 0:n], src=t[0:nr, :])


def gather_rows_cores(cache_hbm, slots_hbm, s0, dst_hbm, d0, rows, pid, n, pair=False, off=0):
    """`dst[d0 + r] = cache[slots[s0 + r]]` for the rows `r < rows` of core `pid`'s 128-row
    tiles (tile `i` to core `(i + off) % 2`, as `tiles_of` for `off = 0`), `gather_rows`'
    per-tile gathers with the full tiles' slot indices loaded by ONE hardware-DGE DMA
    (`ix[p, j]` = slot of row `(i0 + 2j) * 128 + p`) instead of one GpSimd DMA per tile.
    Same rows, same values (`off` only moves tiles between the cores)."""
    full = [i for i in range(rows // PT) if (i + off) % 2 == pid]
    if full:
        k = len(full)
        i0 = full[0]
        ix = nl.ndarray((PT, k), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ix, src=slots_hbm.ap(pattern=[[1, PT], [2 * PT, k]], offset=s0 + i0 * PT),
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        for j, i in enumerate(full):
            t = nl.ndarray((PT, n), dtype=dst_hbm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=t,
                src=cache_hbm.ap(pattern=[[n, PT], [1, n]], vector_offset=ix[0:PT, j : j + 1], indirect_dim=0),
            )
            if pair:
                npair = n // (2 * PT)
                tp = nl.ndarray((PT, npair, PT, 2), dtype=dst_hbm.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=tp[0:PT, :, :, :],
                                 src=t.ap(pattern=[[n, PT], [2 * PT, npair], [1, PT], [PT, 2]]))
                t = tp.reshape((PT, n))
            r0 = d0 + i * PT
            nisa.dma_copy(dst=dst_hbm[r0 : r0 + PT, 0:n], src=t)
    if rows % PT and (rows // PT + off) % 2 == pid:
        r0 = (rows // PT) * PT
        gather_rows(cache_hbm, slots_hbm[s0 : s0 + rows], dst_hbm[d0 : d0 + rows], [(r0, rows - r0)], n,
                    pair=pair)


#: The attention slot table stores each row's 128-column chunks interleaved in pairs,
#: `row'[m*256 + 2i + s] = row[(2m + s)*128 + i]`, so a key tile's transpose is one fp32
#: PE transpose per chunk pair (a bit-exact pass-through costing about what a bf16 one
#: does) instead of one bf16 transpose per chunk; the transposed halves are split back
#: into the chunks by the PSUM -> SBUF copy, and the PV result (over the interleaved
#: columns) is read back in the natural column order by its scaling.
ATTN_PAIR_T = True


def kv_pair_layout(d):
    return ATTN_PAIR_T and (d // PT) % 2 == 0 and d % PT == 0


def key_tile_t(gn, kb, gt, d, eng):
    """`gt[:, c, kb*128 : (kb+1)*128] = gn[:, kb, c*128 : (c+1)*128].T` for every chunk
    `c` (gathered rows `gn` `(PT, ktiles, d)` in the slot table's column order)."""
    dchunks = d // PT
    if kv_pair_layout(d):
        npair = dchunks // 2
        gnf = gn.view(nl.float32)
        tps = nl.ndarray((PT, npair, PT), dtype=nl.float32, buffer=nl.psum)
        for m in range(npair):
            nisa.nc_transpose(dst=tps[0:PT, m, 0:PT], data=gnf[0:PT, kb, m * PT : (m + 1) * PT])
        kpad = gt.shape[2]
        gt4 = gt.reshape((PT, npair, 2, kpad))
        nisa.tensor_copy(dst=gt4[0:PT, 0:npair, 0:2, kb * PT : (kb + 1) * PT],
                         src=tps.ap(pattern=[[npair * PT * 2, PT], [2 * PT, npair], [1, 2], [2, PT]],
                                    dtype=nl.bfloat16),
                         engine=eng)
        return
    tps = nl.ndarray((PT, dchunks, PT), dtype=nl.bfloat16, buffer=nl.psum)
    for c in range(dchunks):
        nisa.nc_transpose(dst=tps[0:PT, c, 0:PT], data=gn[0:PT, kb, c * PT : (c + 1) * PT])
    nisa.tensor_copy(dst=gt[0:PT, 0:dchunks, kb * PT : (kb + 1) * PT], src=tps, engine=eng)


def key_tile_packed(gn, kb, gtp, d, eng):
    """`key_tile_t` keeping the pair layout: `gtp[:, m, kb*128 + k, s]` = chunk `2m + s`
    of key `k` transposed (`(PT, d // 256, kpad, 2)` bf16; one contiguous PSUM copy, the
    score matmuls read a chunk with stride 2)."""
    npair = d // (2 * PT)
    gnf = gn.view(nl.float32)
    tps = nl.ndarray((PT, npair, PT), dtype=nl.float32, buffer=nl.psum)
    first = None
    for m in range(npair):
        ins = nisa.nc_transpose(dst=tps[0:PT, m, 0:PT], data=gnf[0:PT, kb, m * PT : (m + 1) * PT])
        if first is None:
            first = ins
    cp = nisa.tensor_copy(dst=gtp[0:PT, 0:npair, kb * PT : (kb + 1) * PT, 0:2],
                          src=tps.ap(pattern=[[npair * PT * 2, PT], [2 * PT, npair], [2, PT], [1, 2]],
                                     dtype=nl.bfloat16),
                          engine=eng)
    return first, cp


def pv_out(o16, acc_o, rows, rcp, d):
    """`o16[0:rows] = acc_o * rcp` (bf16) with the PV result's columns (the slot table's
    column order) back in the natural order."""
    if kv_pair_layout(d):
        npair = d // (2 * PT)
        o4 = o16.reshape((o16.shape[0], npair, 2, PT))
        nisa.tensor_scalar(dst=o4[0:rows, :, :, :],
                           data=acc_o.ap(pattern=[[d, rows], [2 * PT, npair], [1, 2], [2, PT]]),
                           op0=nl.multiply, operand0=rcp)
        return
    nisa.tensor_scalar(dst=o16[0:rows, :], data=acc_o[0:rows, :], op0=nl.multiply, operand0=rcp)


# --------------------------------------------------------------------------- sparse attention


def _load_col_f32(col_hbm, r0, nr):
    """An int32 `[T, 1]` host column as an fp32 `(PT, 1)` SBUF tile (rows r0..r0+nr)."""
    ci = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ci[0:nr, :], src=col_hbm[r0 : r0 + nr, 0:1])
    cf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cf[0:nr, :], src=ci[0:nr, :])
    return cf


def attn_index_table(idx_hbm, tiles, twin, wv, win, cbase, topk_hbm, topk, kpad, idxp_hbm=None,
                     zrow0=None):
    """Build the per-row slot index table `idx_hbm[T, kpad]` (int32, -1 = empty):

        idx[r, j]         = twin[r] + j         if wv[r] + j >= 0 else -1   (j < win)
        idx[r, win + j]   = cbase[r] + topk[r, j]  if topk[r, j] >= 0 else -1
        idx[r, win+topk:] = -1

    `twin`, `wv`, `cbase` are `[T, 1]` int32 host columns (table row of the first
    window slot, window validity base, table row of the first compressed slot).

    `idxp_hbm` (optional) gets the same rows permuted for the batched gather
    (`gather_index_p`): `idxp[r, p * kt + m] = idx[r, m * 128 + p]`, `kt = kpad / 128`.
    With `zrow0`, an empty slot of `idxp` is `zrow0 + p` instead of -1: one of 128 zero
    rows of the slot table, so the masked slots of a gather do not all read one row.
    `idxp_hbm` is stored transposed per row (`[r, a * kt + q] = idxp[r, q * 128 + a]`,
    see below), so `attn_step` loads a token's gather index contiguously.
    """
    ktiles = kpad // PT
    for r0, nr in tiles:
        row = nl.ndarray((PT, kpad), dtype=nl.float32, buffer=nl.sbuf)
        j = nl.ndarray((PT, win), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=j[0:nr, :], pattern=[[1, win]], offset=0, channel_multiplier=0)
        jf = nl.ndarray((PT, win), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=jf[0:nr, :], src=j[0:nr, :])
        tw = _load_col_f32(twin, r0, nr)
        wvc = _load_col_f32(wv, r0, nr)
        # valid = (j + wv >= 0) as 0/1
        valid = nl.ndarray((PT, win), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=valid[0:nr, :], data=jf[0:nr, :], op0=nl.add, operand0=wvc[0:nr, 0:1],
                           op1=nl.greater_equal, operand1=0.0)
        # idx = (j + tw) * valid + (valid - 1)
        base = nl.ndarray((PT, win), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=base[0:nr, :], data=jf[0:nr, :], op0=nl.add, operand0=tw[0:nr, 0:1])
        nisa.tensor_tensor(dst=base[0:nr, :], data1=base[0:nr, :], data2=valid[0:nr, :], op=nl.multiply)
        vm1 = nl.ndarray((PT, win), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=vm1[0:nr, :], data=valid[0:nr, :], op0=nl.add, operand0=-1.0)
        nisa.tensor_tensor(dst=row[0:nr, 0:win], data1=base[0:nr, :], data2=vm1[0:nr, :], op=nl.add)
        if topk:
            tki = nl.ndarray((PT, topk), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=tki[0:nr, :], src=topk_hbm[r0 : r0 + nr, 0:topk])
            tk = nl.ndarray((PT, topk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=tk[0:nr, :], src=tki[0:nr, :])
            cb = _load_col_f32(cbase, r0, nr)
            tv = nl.ndarray((PT, topk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=tv[0:nr, :], data=tk[0:nr, :], op0=nl.greater_equal, operand0=0.0)
            tb = nl.ndarray((PT, topk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=tb[0:nr, :], data=tk[0:nr, :], op0=nl.add, operand0=cb[0:nr, 0:1])
            nisa.tensor_tensor(dst=tb[0:nr, :], data1=tb[0:nr, :], data2=tv[0:nr, :], op=nl.multiply)
            tm1 = nl.ndarray((PT, topk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=tm1[0:nr, :], data=tv[0:nr, :], op0=nl.add, operand0=-1.0)
            nisa.tensor_tensor(dst=row[0:nr, win : win + topk], data1=tb[0:nr, :], data2=tm1[0:nr, :], op=nl.add)
        if kpad > win + topk:
            nisa.memset(row[0:nr, win + topk : kpad], -1.0)
        irow = nl.ndarray((PT, kpad), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=irow[0:nr, :], src=row[0:nr, :])
        nisa.dma_copy(dst=idx_hbm[r0 : r0 + nr, 0:kpad], src=irow[0:nr, :])
        if idxp_hbm is not None:
            prow = nl.ndarray((PT, PT, ktiles), dtype=nl.int32, buffer=nl.sbuf)
            if zrow0 is None:
                for m in range(ktiles):
                    nisa.tensor_copy(dst=prow[0:nr, :, m], src=irow[0:nr, m * PT : (m + 1) * PT])
            else:
                # empty (-1) -> zrow0 + p: row + (row < 0) * (zrow0 + 1 + p), exact in fp32
                zp = nl.ndarray((PT, PT), dtype=nl.int32, buffer=nl.sbuf)
                nisa.iota(dst=zp[0:nr, :], pattern=[[1, PT]], offset=zrow0 + 1, channel_multiplier=0)
                zpf = nl.ndarray((PT, PT), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=zpf[0:nr, :], src=zp[0:nr, :])
                for m in range(ktiles):
                    c0, c1 = m * PT, (m + 1) * PT
                    neg = nl.ndarray((PT, PT), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=neg[0:nr, :], data=row[0:nr, c0:c1], op0=nl.less, operand0=0.0)
                    nisa.tensor_tensor(dst=neg[0:nr, :], data1=neg[0:nr, :], data2=zpf[0:nr, :], op=nl.multiply)
                    nisa.tensor_tensor(dst=neg[0:nr, :], data1=neg[0:nr, :], data2=row[0:nr, c0:c1], op=nl.add)
                    nisa.tensor_copy(dst=prow[0:nr, :, m], src=neg[0:nr, :])
            # stored transposed as `(128, kt)` per row: the gather's `(128, kt)` index tile
            # (read column-major) is then one contiguous run per partition,
            # `idxp_t[r, a * kt + q] = idxp[r, q * 128 + a]`
            pq = prow.reshape((PT, ktiles, PT))
            xrow = nl.ndarray((PT, PT, ktiles), dtype=nl.int32, buffer=nl.sbuf)
            for q in range(ktiles):
                nisa.tensor_copy(dst=xrow[0:nr, :, q], src=pq[0:nr, q, :])
            nisa.dma_copy(dst=idxp_hbm.reshape((idxp_hbm.shape[0], PT, ktiles))[r0 : r0 + nr, :, :],
                          src=xrow[0:nr, :, :])


def _ktile(k):
    if k <= PSUM_FP32:
        return k
    for cand in (PSUM_FP32, 256, PT):
        if k % cand == 0:
            return cand
    return PT


def _align16(n):
    return -(-n // 16) * 16


def attn_token_body(i, tok0, q_hbm, kvtab_hbm, idx_hbm, o_hbm, sink_t, heads, d, kpad, scale):
    """Sparse attention for token `tok0 + i` (`i` a loop register): the algorithm of
    `kernels.sparse_attn.gathered_attn_idx`, with the empty-slot bias built here.

    `q_hbm`/`o_hbm` are `[Tp, heads*d]` bf16, `kvtab_hbm` the `[rows, d]` bf16 slot
    table, `idx_hbm` `[Tp, kpad]` int32 (-1 = empty), `sink_t` `(heads, 1)` fp32 SBUF.
    """
    dchunks = d // PT
    ktiles = kpad // PT
    kt = _ktile(kpad)
    ntiles = kpad // kt
    hp = _align16(heads)
    hw = heads * d

    # This token's slot indices, twice: (PT, ktiles) for the gather, (1, kpad) for the bias.
    ids = nl.ndarray((PT, ktiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=ids,
        src=idx_hbm.ap(pattern=[[1, PT], [PT, ktiles]], offset=tok0 * kpad,
                       scalar_offset=i, indirect_dim=0),
    )
    gidx = nl.ndarray((PT, ktiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=gidx, data=ids, op0=nl.maximum, operand0=0.0)
    idrow = nl.ndarray((1, kpad), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=idrow,
        src=idx_hbm.ap(pattern=[[kpad, 1], [1, kpad]], offset=tok0 * kpad,
                       scalar_offset=i, indirect_dim=0),
    )
    # bias = (idx < 0) * NEG_BOUND, broadcast over the heads.
    brow = nl.ndarray((1, kpad), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=brow, data=idrow, op0=nl.less, operand0=0.0,
                       op1=nl.multiply, operand1=NEG_BOUND)
    bias_b = nl.broadcast_to(brow, (heads, kpad))

    # q rows of this token: (heads, d) -> transposed (PT, dchunks, heads) on the PE.
    qrow = nl.ndarray((hp, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=qrow[0:heads, :],
        src=q_hbm.ap(pattern=[[d, heads], [1, d]], offset=tok0 * hw, scalar_offset=i, indirect_dim=0),
    )
    qt = nl.ndarray((PT, dchunks, hp), dtype=nl.bfloat16, buffer=nl.sbuf)
    for c in range(dchunks):
        qps = nl.ndarray((PT, hp), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=qps[0:PT, 0:heads], data=qrow[0:heads, c * PT : (c + 1) * PT])
        nisa.tensor_copy(dst=qt[0:PT, c, 0:heads], src=qps[0:PT, 0:heads])

    gt = nl.ndarray((PT, dchunks, kpad), dtype=nl.bfloat16, buffer=nl.sbuf)
    gn = nl.ndarray((PT, ktiles, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kb in range(ktiles):
        nisa.dma_copy(
            dst=gn[0:PT, kb, 0:d],
            src=kvtab_hbm.ap(pattern=[[d, PT], [1, d]], offset=0,
                             vector_offset=gidx[0:PT, kb : kb + 1], indirect_dim=0),
        )
        key_tile_t(gn, kb, gt, d, nisa.engine.unknown)

    sc = nl.ndarray((heads, kpad), dtype=nl.float32, buffer=nl.sbuf)
    for nb in range(ntiles):
        acc = nl.ndarray((heads, kt), dtype=nl.float32, buffer=nl.psum)
        for c in range(dchunks):
            nisa.nc_matmul(
                dst=acc, stationary=qt[0:PT, c, 0:heads],
                moving=gt[0:PT, c, nb * kt : (nb + 1) * kt], accumulate=(c > 0),
            )
        nisa.scalar_tensor_tensor(
            dst=sc[0:heads, nb * kt : (nb + 1) * kt], data=acc, op0=nl.multiply, operand0=scale,
            op1=nl.add, operand1=bias_b[0:heads, nb * kt : (nb + 1) * kt],
        )

    neg_max = nl.negative(nl.max(sc[0:heads, 0:kpad], axis=1, keepdims=True))
    p32 = nl.ndarray((heads, kpad), dtype=nl.float32, buffer=nl.sbuf)
    den = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=p32, op=nl.exp, data=sc[0:heads, 0:kpad], bias=neg_max, scale=1.0,
                    reduce_op=nl.add, reduce_res=den)
    sk = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sk, op=nl.exp, data=sink_t, bias=neg_max)
    rcp = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rcp, data=nl.add(den, sk))
    p16 = nl.copy(p32, dtype=nl.bfloat16)

    pt = nl.ndarray((PT, ktiles, hp), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kb in range(ktiles):
        ptps = nl.ndarray((PT, hp), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=ptps[0:PT, 0:heads], data=p16[0:heads, kb * PT : (kb + 1) * PT])
        nisa.tensor_copy(dst=pt[0:PT, kb, 0:heads], src=ptps[0:PT, 0:heads])
    acc_o = nl.ndarray((heads, d), dtype=nl.float32, buffer=nl.psum)
    for kb in range(ktiles):
        nisa.nc_matmul(dst=acc_o, stationary=pt[0:PT, kb, 0:heads], moving=gn[0:PT, kb, 0:d],
                       accumulate=(kb > 0))
    o16 = nl.ndarray((hp, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    pv_out(o16, acc_o, heads, rcp, d)
    nisa.dma_copy(
        dst=o_hbm.ap(pattern=[[d, heads], [1, d]], offset=tok0 * hw, scalar_offset=i, indirect_dim=0),
        src=o16[0:heads, :],
    )


#: SBUF bytes a partition may hold of the resident transposed key table
#: (`build_kT`): 4 chunks x 2 bytes x rows, so 96 KiB covers 12288 slot-table rows.
KT_SBUF_BYTES = 96 * 1024
#: The widest tensor a tensor-indirect matmul operand may read (NCC_IBIR039).
TI_MAX_COLS = 4096


def kt_resident(ntab, hd):
    """Whether a slot table of `ntab` rows fits the resident transposed layout
    (the per-partition cost: `hd // 128` chunks of `pad_to(ntab, 128)` bf16)."""
    ntab_p = pad_to(ntab, PT)
    return (hd // PT) * ntab_p * 2 <= KT_SBUF_BYTES and ntab_p <= TI_MAX_COLS


def _ti_block(kpad):
    """Score-block width of the resident path: the widest multiple of 16 that
    divides `kpad` and fits one PSUM bank."""
    for cand in range(PSUM_FP32, 0, -16):
        if kpad % cand == 0:
            return cand
    return 16


def build_kT(kvtab_hbm, ntab, hd):
    """The slot table transposed and resident: one SBUF tensor per 128-wide head-dim
    chunk, `kT[c][d, r] = kvtab[r, c*128 + d]` (bf16), built with DMA transposes.
    `kvtab_hbm` has at least `pad_to(ntab, 128)` rows."""
    ntab_p = pad_to(ntab, PT)
    kTs = []
    for c in range(hd // PT):
        kT = nl.ndarray((PT, ntab_p), dtype=nl.bfloat16, buffer=nl.sbuf)
        for r0 in range(0, ntab_p, PT):
            nisa.dma_transpose(dst=kT[0:PT, r0 : r0 + PT], src=kvtab_hbm[r0 : r0 + PT, c * PT : (c + 1) * PT])
        kTs.append(kT)
    return kTs


def ti_score_psum(heads, kpad):
    """The resident path's score accumulators, one PSUM bank-sized block each. A
    tensor-indirect matmul's destination is loop-carried for the compiler, so these
    are allocated outside the token loop."""
    kb = _ti_block(kpad)
    return [nl.ndarray((heads, kb), dtype=nl.float32, buffer=nl.psum) for _ in range(kpad // kb)]


def attn_token_body_ti(i, tok0, q_hbm, kvtab_hbm, kTs, accs, idx_hbm, o_hbm, sink_t, heads, d, kpad,
                       scale):
    """`attn_token_body` with the keys selected on chip: the score matmuls read the
    token's `kpad` columns of the resident `kTs` through tensor indirection (the
    slot indices in the 16-partition snake the tensor engine reads, replicated into
    each quadrant), so no per-token transposes; the values are still gathered from
    `kvtab_hbm` by indirect DMA for the PV matmul."""
    dchunks = d // PT
    ktiles = kpad // PT
    bw = _ti_block(kpad)
    nblk = kpad // bw
    ncol = bw // 16
    hp = _align16(heads)
    hw = heads * d

    ids = nl.ndarray((PT, ktiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=ids,
        src=idx_hbm.ap(pattern=[[1, PT], [PT, ktiles]], offset=tok0 * kpad,
                       scalar_offset=i, indirect_dim=0),
    )
    gidx = nl.ndarray((PT, ktiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=gidx, data=ids, op0=nl.maximum, operand0=0.0)
    idrow = nl.ndarray((1, kpad), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=idrow,
        src=idx_hbm.ap(pattern=[[kpad, 1], [1, kpad]], offset=tok0 * kpad,
                       scalar_offset=i, indirect_dim=0),
    )
    brow = nl.ndarray((1, kpad), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=brow, data=idrow, op0=nl.less, operand0=0.0,
                       op1=nl.multiply, operand1=NEG_BOUND)
    bias_b = nl.broadcast_to(brow, (heads, kpad))
    # the snake, per score block of `bw` columns: block offset j at [j % 16, j // 16]
    # of each quadrant
    ix32 = nl.ndarray((PT, nblk, ncol), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(ix32, 0)
    for qd in range(4):
        nisa.dma_copy(
            dst=ix32[32 * qd : 32 * qd + 16, 0:nblk, 0:ncol],
            src=idx_hbm.ap(pattern=[[1, 16], [bw, nblk], [16, ncol]], offset=tok0 * kpad,
                           scalar_offset=i, indirect_dim=0),
        )
    nisa.tensor_scalar(dst=ix32, data=ix32, op0=nl.maximum, operand0=0)
    ixs = []
    for nb in range(nblk):
        ix = nl.ndarray((PT, ncol), dtype=nl.uint16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ix, src=ix32[0:PT, nb, 0:ncol])
        ixs.append(ix)

    qrow = nl.ndarray((hp, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=qrow[0:heads, :],
        src=q_hbm.ap(pattern=[[d, heads], [1, d]], offset=tok0 * hw, scalar_offset=i, indirect_dim=0),
    )
    qt = nl.ndarray((PT, dchunks, hp), dtype=nl.bfloat16, buffer=nl.sbuf)
    for c in range(dchunks):
        qps = nl.ndarray((PT, hp), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=qps[0:PT, 0:heads], data=qrow[0:heads, c * PT : (c + 1) * PT])
        nisa.tensor_copy(dst=qt[0:PT, c, 0:heads], src=qps[0:PT, 0:heads])

    gn = nl.ndarray((PT, ktiles, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kb in range(ktiles):
        nisa.dma_copy(
            dst=gn[0:PT, kb, 0:d],
            src=kvtab_hbm.ap(pattern=[[d, PT], [1, d]], offset=0,
                             vector_offset=gidx[0:PT, kb : kb + 1], indirect_dim=0),
        )

    sc = nl.ndarray((heads, kpad), dtype=nl.float32, buffer=nl.sbuf)
    for nb in range(nblk):
        acc = accs[nb]
        for c in range(dchunks):
            nisa.nc_matmul(
                dst=acc, stationary=qt[0:PT, c, 0:heads],
                moving=kTs[c].indirect(ixs[nb], num_elem=bw), accumulate=(c > 0),
            )
        nisa.scalar_tensor_tensor(
            dst=sc[0:heads, nb * bw : (nb + 1) * bw], data=acc, op0=nl.multiply, operand0=scale,
            op1=nl.add, operand1=bias_b[0:heads, nb * bw : (nb + 1) * bw],
        )

    neg_max = nl.negative(nl.max(sc[0:heads, 0:kpad], axis=1, keepdims=True))
    p32 = nl.ndarray((heads, kpad), dtype=nl.float32, buffer=nl.sbuf)
    den = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=p32, op=nl.exp, data=sc[0:heads, 0:kpad], bias=neg_max, scale=1.0,
                    reduce_op=nl.add, reduce_res=den)
    sk = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sk, op=nl.exp, data=sink_t, bias=neg_max)
    rcp = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rcp, data=nl.add(den, sk))
    p16 = nl.copy(p32, dtype=nl.bfloat16)

    pt = nl.ndarray((PT, ktiles, hp), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kb in range(ktiles):
        ptps = nl.ndarray((PT, hp), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=ptps[0:PT, 0:heads], data=p16[0:heads, kb * PT : (kb + 1) * PT])
        nisa.tensor_copy(dst=pt[0:PT, kb, 0:heads], src=ptps[0:PT, 0:heads])
    acc_o = nl.ndarray((heads, d), dtype=nl.float32, buffer=nl.psum)
    for kb in range(ktiles):
        nisa.nc_matmul(dst=acc_o, stationary=pt[0:PT, kb, 0:heads], moving=gn[0:PT, kb, 0:d],
                       accumulate=(kb > 0))
    o16 = nl.ndarray((hp, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o16[0:heads, :], data=acc_o, op0=nl.multiply, operand0=rcp)
    nisa.dma_copy(
        dst=o_hbm.ap(pattern=[[d, heads], [1, d]], offset=tok0 * hw, scalar_offset=i, indirect_dim=0),
        src=o16[0:heads, :],
    )


def attn_tokens(i, toks, q_hbm, kvtab_hbm, kTs, accs, idx_hbm, o_hbm, sink_t, heads, d, kpad, scale,
                idxp_hbm=None):
    """`attn_token_body_ti` (`kTs` given) or `attn_token_body` (`kTs` None: the keys
    transposed from the gathered rows on the PE) for `len(toks)` tokens per loop step,
    emitted stage by stage
    (indices and gathers of every token first, then the matmuls, softmaxes, PV) so the
    tokens' DMA and compute chains overlap. Token `u` is row `toks[u] + i * U` of the
    `[Tp, ...]` tables: `q_hbm` / `idx_hbm` / `o_hbm` come reshaped to `U` rows per
    step (`[Tp / U, U * width]`), so the loop register `i` steps `U` rows. `accs` are
    the resident path's score accumulator sets (`ti_score_psum`), token `u` uses set
    `u % len(accs)`. Same arithmetic per token as the single-token bodies.

    With `idxp_hbm` (the permuted index table of `attn_index_table`, reshaped like
    `idx_hbm`) a token's values are gathered by ONE batched indirect DMA (2-D
    `vector_offset`: the index is read column-major and the rows land row-major, so
    the permutation puts key `m * 128 + p` at `gn[p, m]`, the same layout)."""
    U = len(toks)
    ti = kTs is not None
    dchunks = d // PT
    ktiles = kpad // PT
    kt = _ktile(kpad)
    bw = _ti_block(kpad)
    nblk = kpad // bw
    ncol = bw // 16
    hp = _align16(heads)
    hw = heads * d

    # indices (gather column, bias row, snake) and the value gathers
    gns, biases, ixss, qrows = [], [], [], []
    for u in range(U):
        t0 = toks[u]
        ids = nl.ndarray((PT, ktiles), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=ids,
            src=idx_hbm.ap(pattern=[[1, PT], [PT, ktiles]], offset=t0 * kpad,
                           scalar_offset=i, indirect_dim=0),
        )
        gidx = nl.ndarray((PT, ktiles), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gidx, data=ids, op0=nl.maximum, operand0=0.0)
        gn = nl.ndarray((PT, ktiles, d), dtype=nl.bfloat16, buffer=nl.sbuf)
        if idxp_hbm is not None:
            pids = nl.ndarray((PT, ktiles), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=pids,
                src=idxp_hbm.ap(pattern=[[1, PT], [PT, ktiles]], offset=t0 * kpad,
                                scalar_offset=i, indirect_dim=0),
            )
            pidx = nl.ndarray((PT, ktiles), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=pidx, data=pids, op0=nl.maximum, operand0=0.0)
            nisa.dma_copy(
                dst=gn,
                src=kvtab_hbm.ap(pattern=[[d, PT * ktiles], [1, d]], offset=0,
                                 vector_offset=pidx, indirect_dim=0),
            )
        else:
            for kb in range(ktiles):
                nisa.dma_copy(
                    dst=gn[0:PT, kb, 0:d],
                    src=kvtab_hbm.ap(pattern=[[d, PT], [1, d]], offset=0,
                                     vector_offset=gidx[0:PT, kb : kb + 1], indirect_dim=0),
                )
        gns.append(gn)
    for u in range(U):
        t0 = toks[u]
        if ti:
            ix32 = nl.ndarray((PT, nblk, ncol), dtype=nl.int32, buffer=nl.sbuf)
            nisa.memset(ix32, 0)
            for qd in range(4):
                nisa.dma_copy(
                    dst=ix32[32 * qd : 32 * qd + 16, 0:nblk, 0:ncol],
                    src=idx_hbm.ap(pattern=[[1, 16], [bw, nblk], [16, ncol]], offset=t0 * kpad,
                                   scalar_offset=i, indirect_dim=0),
                )
            nisa.tensor_scalar(dst=ix32, data=ix32, op0=nl.maximum, operand0=0)
            ixs = []
            for nb in range(nblk):
                ix = nl.ndarray((PT, ncol), dtype=nl.uint16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ix, src=ix32[0:PT, nb, 0:ncol])
                ixs.append(ix)
            ixss.append(ixs)
        qrow = nl.ndarray((hp, d), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=qrow[0:heads, :],
            src=q_hbm.ap(pattern=[[d, heads], [1, d]], offset=t0 * hw, scalar_offset=i, indirect_dim=0),
        )
        qrows.append(qrow)
        idrow = nl.ndarray((1, kpad), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=idrow,
            src=idx_hbm.ap(pattern=[[kpad, 1], [1, kpad]], offset=t0 * kpad,
                           scalar_offset=i, indirect_dim=0),
        )
        brow = nl.ndarray((1, kpad), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=brow, data=idrow, op0=nl.less, operand0=0.0,
                           op1=nl.multiply, operand1=NEG_BOUND)
        biases.append(nl.broadcast_to(brow, (heads, kpad)))

    # q transposes and scores
    scs = []
    for u in range(U):
        qt = nl.ndarray((PT, dchunks, hp), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c in range(dchunks):
            qps = nl.ndarray((PT, hp), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=qps[0:PT, 0:heads], data=qrows[u][0:heads, c * PT : (c + 1) * PT])
            nisa.tensor_copy(dst=qt[0:PT, c, 0:heads], src=qps[0:PT, 0:heads])
        sc = nl.ndarray((heads, kpad), dtype=nl.float32, buffer=nl.sbuf)
        if ti:
            for nb in range(nblk):
                acc = accs[u % len(accs)][nb]
                for c in range(dchunks):
                    nisa.nc_matmul(
                        dst=acc, stationary=qt[0:PT, c, 0:heads],
                        moving=kTs[c].indirect(ixss[u][nb], num_elem=bw), accumulate=(c > 0),
                    )
                nisa.scalar_tensor_tensor(
                    dst=sc[0:heads, nb * bw : (nb + 1) * bw], data=acc, op0=nl.multiply,
                    operand0=scale, op1=nl.add, operand1=biases[u][0:heads, nb * bw : (nb + 1) * bw],
                )
        else:
            gt = nl.ndarray((PT, dchunks, kpad), dtype=nl.bfloat16, buffer=nl.sbuf)
            for kb in range(ktiles):
                tps = nl.ndarray((PT, dchunks, PT), dtype=nl.bfloat16, buffer=nl.psum)
                for c in range(dchunks):
                    nisa.nc_transpose(dst=tps[0:PT, c, 0:PT], data=gns[u][0:PT, kb, c * PT : (c + 1) * PT])
                nisa.tensor_copy(dst=gt[0:PT, 0:dchunks, kb * PT : (kb + 1) * PT], src=tps)
            for nb in range(kpad // kt):
                acc = nl.ndarray((heads, kt), dtype=nl.float32, buffer=nl.psum)
                for c in range(dchunks):
                    nisa.nc_matmul(
                        dst=acc, stationary=qt[0:PT, c, 0:heads],
                        moving=gt[0:PT, c, nb * kt : (nb + 1) * kt], accumulate=(c > 0),
                    )
                nisa.scalar_tensor_tensor(
                    dst=sc[0:heads, nb * kt : (nb + 1) * kt], data=acc, op0=nl.multiply,
                    operand0=scale, op1=nl.add, operand1=biases[u][0:heads, nb * kt : (nb + 1) * kt],
                )
        scs.append(sc)

    # softmax, p transposes, PV, store
    for u in range(U):
        sc = scs[u]
        neg_max = nl.negative(nl.max(sc[0:heads, 0:kpad], axis=1, keepdims=True))
        p32 = nl.ndarray((heads, kpad), dtype=nl.float32, buffer=nl.sbuf)
        den = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=p32, op=nl.exp, data=sc[0:heads, 0:kpad], bias=neg_max, scale=1.0,
                        reduce_op=nl.add, reduce_res=den)
        sk = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sk, op=nl.exp, data=sink_t, bias=neg_max)
        rcp = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rcp, data=nl.add(den, sk))
        p16 = nl.copy(p32, dtype=nl.bfloat16)

        pt = nl.ndarray((PT, ktiles, hp), dtype=nl.bfloat16, buffer=nl.sbuf)
        for kb in range(ktiles):
            ptps = nl.ndarray((PT, hp), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=ptps[0:PT, 0:heads], data=p16[0:heads, kb * PT : (kb + 1) * PT])
            nisa.tensor_copy(dst=pt[0:PT, kb, 0:heads], src=ptps[0:PT, 0:heads])
        acc_o = nl.ndarray((heads, d), dtype=nl.float32, buffer=nl.psum)
        for kb in range(ktiles):
            nisa.nc_matmul(dst=acc_o, stationary=pt[0:PT, kb, 0:heads], moving=gns[u][0:PT, kb, 0:d],
                           accumulate=(kb > 0))
        o16 = nl.ndarray((hp, d), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=o16[0:heads, :], data=acc_o, op0=nl.multiply, operand0=rcp)
        nisa.dma_copy(
            dst=o_hbm.ap(pattern=[[d, heads], [1, d]], offset=toks[u] * hw, scalar_offset=i,
                         indirect_dim=0),
            src=o16[0:heads, :],
        )


#: Of a token's `kpad // 128` PSUM -> SBUF copies of its transposed keys, the tiles
#: copied on the scalar engine (the rest on the vector engine).
GT_COPY_ACT = (1, 3)
#: With the pair layout, the transposed keys stay packed (`key_tile_packed`): a
#: contiguous PSUM copy, and the score matmuls read each chunk with stride 2.
GT_PACKED = True


#: `attn_scores`: token u's key transposes are scheduled after token u - LAG's score
#: matmuls (a schedule-only dependency; 0 = none). Without it the compiler's list
#: schedule of the step varies between builds: some builds issue all of a step's key
#: transposes first and the score matmuls in one block after them (attention 3.57 ->
#: 4.2 ms at in8192_bs1).
ATTN_KT_LAG = 2
#: The pipelined loop schedules each step's DVE / ACT work (q / key copies, bias) after
#: the previous step's p-stationary copy (schedule-only): else a next-step key copy,
#: waiting on its gather, can sit ahead of it in the in-order DVE queue and stall the
#: step's PV (~8 us a step in some builds).
ATTN_STEP_ORDER = True
#: The step buffers carry the bf16 output tile the o store reads (allocated with them,
#: outside the loop): a loop-local one shared its SBUF address with the next step's
#: softmax temporaries, which then waited (WAR) for the o store DMA, queued on GpSimd
#: behind the next step's gathers (~12 us stalls).
ATTN_O16_BUF = True

#: Rows (tokens x heads) per softmax / PV group of `attn_step`.
ATTN_GROUP_ROWS = 128
#: `attn_softmax`: the row max negated by the reduce itself and the exponentials written
#: as bf16 by the exp activation (no fp32 copy + cast on the softmax -> PV path).
ATTN_SOFTMAX_FUSED = True
#: `attn_compute` fills a group's per-token masked q / p stationaries with ONE strided
#: copy each (token u's columns on its diagonal block) instead of one copy per token.
ATTN_DIAG_COPY = True


def _attn_groups(U, heads):
    """`(Ug, Mg)`: tokens and rows per `attn_step` group."""
    ug = max(1, min(U, ATTN_GROUP_ROWS // heads))
    while U % ug:
        ug -= 1
    return ug, ug * heads


def attn_step_masks(U, heads, d, kpad):
    """`attn_step`'s per-token masked stationaries, zeroed once outside the loop (the
    step writes only each token's own columns): `(qtm, ptm, sel)`; `qtm[:, u, c, :]` /
    `ptm[:, u, kb, :]` hold token `u`'s q / p columns (at its rows of its group) and
    zeros elsewhere; `sel[u, g, :]` is the 0/1 token-to-row selection of group `g`."""
    Ug, Mg = _attn_groups(U, heads)
    G = U // Ug
    qtm = nl.ndarray((PT, U, d // PT, Mg), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(qtm, 0.0)
    ptm = nl.ndarray((PT, U, kpad // PT, Mg), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(ptm, 0.0)
    # sel[u, g, c] = 1 if u == g * Ug + c // heads, i.e. heads*u - heads*Ug*g - c in (-heads, 0]
    rel = nl.ndarray((U, G, Mg), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=rel, pattern=[[-heads * Ug, G], [-1, Mg]], offset=0, channel_multiplier=heads)
    relf = nl.ndarray((U, G, Mg), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=relf, src=rel)
    le = nl.ndarray((U, G, Mg), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=le, data=relf, op0=nl.less_equal, operand0=0.0)
    gt_ = nl.ndarray((U, G, Mg), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=gt_, data=relf, op0=nl.greater, operand0=float(-heads))
    sel = nl.ndarray((U, G, Mg), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sel, data1=le, data2=gt_, op=nl.multiply)
    return qtm, ptm, sel


#: `attn_fetch` loads its U tokens' gather indices with one DMA (else one per token).
ATTN_PIDS_MERGED = True


def _dyn_ap(t, pattern, offset, i):
    """`t.ap(pattern, offset)`, stepped by the loop register `i` (rows of `t`) unless
    `i` is None (a static access)."""
    if i is None:
        return t.ap(pattern=pattern, offset=offset)
    return t.ap(pattern=pattern, offset=offset, scalar_offset=i, indirect_dim=0)


def attn_step_bufs(U, heads, d, kpad):
    """The SBUF buffers one `attn_step` fetches into (`attn_fetch`) and computes from
    (`attn_compute`): the U tokens' gathered values, their q rows and index rows."""
    ktiles = kpad // PT
    b = dict(
        gns=[nl.ndarray((PT, ktiles, d), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(U)],
        qrow=nl.ndarray((U * heads, d), dtype=nl.bfloat16, buffer=nl.sbuf),
        idrow=nl.ndarray((U, kpad), dtype=nl.float32, buffer=nl.sbuf),
    )
    if ATTN_O16_BUF:
        Ug, Mg = _attn_groups(U, heads)
        b["o16"] = [nl.ndarray((Mg, d), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(U // Ug)]
    return b


def attn_fetch(i, tok0, U, q_hbm, kvtab_hbm, idx_hbm, heads, d, kpad, bufs, idxp_hbm=None):
    """`attn_step`'s loads for the `U` tokens `tok0 + R * i + u` (`R` = the rows per step
    of the reshaped `q_hbm` / `idx_hbm` / `idxp_hbm` views; `i` None = static tokens
    `tok0 + u`): the value gathers, the q rows and the index rows, into `bufs`."""
    M = U * heads
    ktiles = kpad // PT
    hw = heads * d
    gns = bufs["gns"]
    if idxp_hbm is not None and ATTN_PIDS_MERGED:
        # all U tokens' gather indices in one DMA: pall[a, u, q] = idxp[tok0 + u][q * 128 + a]
        pall = nl.ndarray((PT, U, ktiles), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=pall, src=_dyn_ap(idxp_hbm, [[ktiles, PT], [kpad, U], [1, ktiles]], tok0 * kpad, i))
    for u in range(U):
        t0 = tok0 + u
        gn = gns[u]
        if idxp_hbm is not None:
            # `attn_index_table`'s transposed rows: pids[a, q] = idxp[q * 128 + a]; empty
            # slots already point at zero rows (`zrow0`)
            if ATTN_PIDS_MERGED:
                pids = pall[0:PT, u, 0:ktiles]
            else:
                pids = nl.ndarray((PT, ktiles), dtype=nl.int32, buffer=nl.sbuf)
                nisa.dma_copy(dst=pids, src=_dyn_ap(idxp_hbm, [[ktiles, PT], [1, ktiles]], t0 * kpad, i))
            nisa.dma_copy(
                dst=gn,
                src=kvtab_hbm.ap(pattern=[[d, PT * ktiles], [1, d]], offset=0,
                                 vector_offset=pids, indirect_dim=0),
            )
        else:
            ids = nl.ndarray((PT, ktiles), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ids, src=_dyn_ap(idx_hbm, [[1, PT], [PT, ktiles]], t0 * kpad, i))
            gidx = nl.ndarray((PT, ktiles), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=gidx, data=ids, op0=nl.maximum, operand0=0.0)
            for kb in range(ktiles):
                nisa.dma_copy(
                    dst=gn[0:PT, kb, 0:d],
                    src=kvtab_hbm.ap(pattern=[[d, PT], [1, d]], offset=0,
                                     vector_offset=gidx[0:PT, kb : kb + 1], indirect_dim=0),
                )
    # the step's q rows (one DMA)
    nisa.dma_copy(dst=bufs["qrow"], src=_dyn_ap(q_hbm, [[d, M], [1, d]], tok0 * hw, i))
    # the step's U index rows in one DMA
    nisa.dma_copy(dst=bufs["idrow"], src=_dyn_ap(idx_hbm, [[kpad, U], [1, kpad]], tok0 * kpad, i))


def attn_step(i, tok0, U, q_hbm, kvtab_hbm, idx_hbm, o_hbm, sink_u, masks, heads, d, kpad, scale,
              idxp_hbm=None):
    """`attn_token_body` for the `U` tokens `tok0 + U * i + u` of one loop step:
    `attn_fetch` then `attn_compute` (local buffers)."""
    bufs = attn_step_bufs(U, heads, d, kpad)
    attn_fetch(i, tok0, U, q_hbm, kvtab_hbm, idx_hbm, heads, d, kpad, bufs, idxp_hbm=idxp_hbm)
    attn_compute(i, tok0, U, o_hbm, bufs, sink_u, masks, heads, d, kpad, scale)


def _sched_after(ins, after):
    """Schedule-only dependency of `ins` on `after` (no hardware sync), if both exist."""
    if after is not None and ins is not None and hasattr(ins, "depends_on"):
        ins.depends_on(after, kind=nisa.dependency.schedule)


def attn_compute(i, tok0, U, o_hbm, bufs, sink_u, masks, heads, d, kpad, scale, after=None):
    """The tokens of one `attn_fetch`ed step (`bufs`), stored to `o_hbm` (stepped like
    the fetch's views by `i`).

    The tokens are processed in groups of `Ug` (`_attn_groups`) with the group's
    tokens stacked on partitions: token `u`, head `h` is row `(u % Ug) * heads + h` of
    the group's score, softmax and output tiles, so the p transposes, the softmax,
    the scaling and the o store run once per group.

    The score and PV matmuls stay per token (each token has its own keys / values):
    token `u`'s use a stationary holding only its own columns (zeros elsewhere,
    `masks`), all accumulating into the group's PSUM tile. A row then receives its own
    token's partial products in the single-token order plus exact zeros from the other
    tokens' matmuls, so every value is the single-token body's.
    `sink_u` is a `(>= Mg, 1)` fp32 sink column (the heads' sinks repeated per token).

    The step runs as four phases (`attn_scores`, `attn_softmax`, `attn_pv`,
    `attn_store`); the 3-buffer pipelined loop interleaves consecutive steps' phases."""
    sts = attn_scores(U, bufs, masks, heads, d, kpad, after=after)
    sm = attn_softmax(U, sts, sink_u, heads, kpad, scale, bufs)
    pcs = []
    for g in range(len(sts)):
        acc_o = attn_pv(U, bufs, masks, heads, d, kpad, g, sm[g][0], pcs=pcs)
        attn_store(i, tok0, U, o_hbm, acc_o, sm[g][1], heads, d, g,
                   o16=bufs["o16"][g] if "o16" in bufs else None)
    # the last p-stationary copy: the next step's DVE / ACT work is scheduled after it
    # (`ATTN_STEP_ORDER`)
    return pcs[-1] if (pcs and ATTN_STEP_ORDER) else None


def attn_scores(U, bufs, masks, heads, d, kpad, after=None):
    """`attn_compute`'s q stationaries, empty-slot bias, key transposes and score
    matmuls: per group `(accs, bias)` (the score PSUM tiles and the fp32 bias).
    `after`: an instruction every DVE / ACT op here is scheduled after (schedule-only)."""
    M = U * heads
    Ug, Mg = _attn_groups(U, heads)
    G = U // Ug
    qtm, ptm, sel = masks
    dchunks = d // PT
    ktiles = kpad // PT
    kt = _ti_block(kpad)
    nkb = kpad // kt
    gns = bufs["gns"]
    qrow = bufs["qrow"]
    idrow = bufs["idrow"]

    # the q rows transposed; each token's columns into its stationary
    qps = nl.ndarray((PT, dchunks, M), dtype=nl.bfloat16, buffer=nl.psum)
    for c in range(dchunks):
        nisa.nc_transpose(dst=qps[0:PT, c, 0:M], data=qrow[0:M, c * PT : (c + 1) * PT])
    if ATTN_DIAG_COPY:
        # one copy per group: token u's heads columns land on its block of the diagonal
        for g in range(G):
            u0 = g * Ug
            qc = nisa.tensor_copy(
                dst=qtm.ap(pattern=[[U * dchunks * Mg, PT], [dchunks * Mg + heads, Ug], [Mg, dchunks],
                                    [1, heads]], offset=u0 * dchunks * Mg),
                src=qps.ap(pattern=[[dchunks * M, PT], [heads, Ug], [M, dchunks], [1, heads]],
                           offset=u0 * heads))
            _sched_after(qc, after)
    else:
        for u in range(U):
            c0 = (u % Ug) * heads
            nisa.tensor_copy(dst=qtm[0:PT, u, 0:dchunks, c0 : c0 + heads],
                             src=qps[0:PT, 0:dchunks, u * heads : (u + 1) * heads])

    # the index rows' empty-slot masks (0/1, exact in bf16) repeated over each token's
    # heads by the 0/1 selection matmul `sel`; bias = mask * NEG_BOUND in fp32, the
    # single-token body's value
    emask = nl.ndarray((U, kpad), dtype=nl.bfloat16, buffer=nl.sbuf)
    _sched_after(nisa.tensor_scalar(dst=emask, data=idrow, op0=nl.less, operand0=0.0), after)

    out = []
    for g in range(G):
        us = range(g * Ug, (g + 1) * Ug)
        bias = nl.ndarray((Mg, kpad), dtype=nl.float32, buffer=nl.sbuf)
        for nb in range(nkb):
            bps = nl.ndarray((Mg, kt), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=bps, stationary=sel[0:U, g, 0:Mg], moving=emask[0:U, nb * kt : (nb + 1) * kt])
            _sched_after(nisa.tensor_scalar(dst=bias[0:Mg, nb * kt : (nb + 1) * kt], data=bps,
                                            op0=nl.multiply, operand0=NEG_BOUND), after)

        # key transposes and scores, token by token into the group's score tiles
        accs = [nl.ndarray((Mg, kt), dtype=nl.float32, buffer=nl.psum) for _ in range(nkb)]
        last_mm = {}
        for u in us:
            packed = kv_pair_layout(d) and GT_PACKED
            if packed:
                gtp = nl.ndarray((PT, dchunks // 2, kpad, 2), dtype=nl.bfloat16, buffer=nl.sbuf)
            else:
                gt = nl.ndarray((PT, dchunks, kpad), dtype=nl.bfloat16, buffer=nl.sbuf)
            for kb in range(ktiles):
                eng = nisa.scalar_engine if kb in GT_COPY_ACT else nisa.vector_engine
                if packed:
                    t0_ins, cp_ins = key_tile_packed(gns[u], kb, gtp, d, eng)
                    _sched_after(cp_ins, after)
                    if (ATTN_KT_LAG and kb == 0 and u - ATTN_KT_LAG in last_mm
                            and t0_ins is not None and last_mm[u - ATTN_KT_LAG] is not None):
                        # schedule-only: token u's key transposes after token u - LAG's
                        # scores (the transposes do not run ahead of the score matmuls,
                        # filling PSUM while the DVE / ACT copies drain it)
                        t0_ins.depends_on(last_mm[u - ATTN_KT_LAG], kind=nisa.dependency.schedule)
                else:
                    key_tile_t(gns[u], kb, gt, d, eng)
            # chunk-outer: consecutive matmuls share the stationary; each score tile still
            # accumulates its chunks in order
            for c in range(dchunks):
                for nb in range(nkb):
                    mv = (gtp[0:PT, c // 2, nb * kt : (nb + 1) * kt, c % 2] if packed
                          else gt[0:PT, c, nb * kt : (nb + 1) * kt])
                    last_mm[u] = nisa.nc_matmul(
                        dst=accs[nb], stationary=qtm[0:PT, u, c, 0:Mg],
                        moving=mv, accumulate=(u > us[0] or c > 0),
                    )
        out.append((accs, bias))
    return out


def attn_softmax(U, sts, sink_u, heads, kpad, scale, bufs):
    """`attn_compute`'s scaled, biased scores and their softmax, per group `(p16, rcp)`
    (the bf16 probabilities and the fp32 1 / (sum + sink)); into `bufs["p16"]` /
    `bufs["rcp"]` when the step buffers carry them (`attn_bufs3`)."""
    Ug, Mg = _attn_groups(U, heads)
    kt = _ti_block(kpad)
    nkb = kpad // kt
    out = []
    for g, (accs, bias) in enumerate(sts):
        sc = nl.ndarray((Mg, kpad), dtype=nl.float32, buffer=nl.sbuf)
        for nb in range(nkb):
            nisa.scalar_tensor_tensor(
                dst=sc[0:Mg, nb * kt : (nb + 1) * kt], data=accs[nb], op0=nl.multiply,
                operand0=scale, op1=nl.add, operand1=bias[0:Mg, nb * kt : (nb + 1) * kt],
            )

        # softmax of the group's rows
        if ATTN_SOFTMAX_FUSED:
            neg_max = nl.ndarray((Mg, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=neg_max, op=nl.maximum, data=sc[0:Mg, 0:kpad], axis=1, negate=True,
                               keepdims=True)
        else:
            neg_max = nl.negative(nl.max(sc[0:Mg, 0:kpad], axis=1, keepdims=True))
        den = nl.ndarray((Mg, 1), dtype=nl.float32, buffer=nl.sbuf)
        if ATTN_SOFTMAX_FUSED and "rcp" not in bufs:
            # the exponentials written as bf16 by the activation itself (its fp32 sum is
            # the reduce, as before)
            p16 = nl.ndarray((Mg, kpad), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=p16, op=nl.exp, data=sc[0:Mg, 0:kpad], bias=neg_max, scale=1.0,
                            reduce_op=nl.add, reduce_res=den)
            sk = nl.ndarray((Mg, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=sk, op=nl.exp, data=sink_u[0:Mg, :], bias=neg_max)
            rcp = nl.ndarray((Mg, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=rcp, data=nl.add(den, sk))
            out.append((p16, rcp))
            continue
        p32 = nl.ndarray((Mg, kpad), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=p32, op=nl.exp, data=sc[0:Mg, 0:kpad], bias=neg_max, scale=1.0,
                        reduce_op=nl.add, reduce_res=den)
        sk = nl.ndarray((Mg, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sk, op=nl.exp, data=sink_u[0:Mg, :], bias=neg_max)
        if "rcp" in bufs:
            rcp = bufs["rcp"][g]
            p16 = bufs["p16"][g]
            nisa.reciprocal(dst=rcp[0:Mg, :], data=nl.add(den, sk))
            nisa.tensor_copy(dst=p16[0:Mg, :], src=p32, engine=nisa.vector_engine)
        else:
            rcp = nl.ndarray((Mg, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=rcp, data=nl.add(den, sk))
            p16 = nl.copy(p32, dtype=nl.bfloat16)
        out.append((p16, rcp))
    return out


def attn_pv(U, bufs, masks, heads, d, kpad, g, p16, pcs=None):
    """`attn_compute`'s p transposes into the masked stationaries and the PV matmuls of
    group `g`: the `(Mg, d)` fp32 PSUM output."""
    Ug, Mg = _attn_groups(U, heads)
    qtm, ptm, sel = masks
    ktiles = kpad // PT
    gns = bufs["gns"]
    us = range(g * Ug, (g + 1) * Ug)
    ptps = nl.ndarray((PT, ktiles, Mg), dtype=nl.bfloat16, buffer=nl.psum)
    for kb in range(ktiles):
        nisa.nc_transpose(dst=ptps[0:PT, kb, 0:Mg], data=p16[0:Mg, kb * PT : (kb + 1) * PT])
    if ATTN_DIAG_COPY:
        pc = nisa.tensor_copy(
            dst=ptm.ap(pattern=[[U * ktiles * Mg, PT], [ktiles * Mg + heads, Ug], [Mg, ktiles],
                                [1, heads]], offset=us[0] * ktiles * Mg),
            src=ptps.ap(pattern=[[ktiles * Mg, PT], [heads, Ug], [Mg, ktiles], [1, heads]], offset=0))
        if pcs is not None:
            pcs.append(pc)
    else:
        for u in us:
            c0 = (u % Ug) * heads
            nisa.tensor_copy(dst=ptm[0:PT, u, 0:ktiles, c0 : c0 + heads],
                             src=ptps[0:PT, 0:ktiles, c0 : c0 + heads])

    # PV, token by token into the group's output tile; one scaling and one store
    acc_o = nl.ndarray((Mg, d), dtype=nl.float32, buffer=nl.psum)
    for u in us:
        for kb in range(ktiles):
            nisa.nc_matmul(dst=acc_o, stationary=ptm[0:PT, u, kb, 0:Mg], moving=gns[u][0:PT, kb, 0:d],
                           accumulate=(u > us[0] or kb > 0))
    return acc_o


def attn_store(i, tok0, U, o_hbm, acc_o, rcp, heads, d, g, o16=None):
    """`attn_compute`'s output scaling (`pv_out`) and store of group `g` (through the
    step buffer's `o16` when given)."""
    Ug, Mg = _attn_groups(U, heads)
    hw = heads * d
    if o16 is None:
        o16 = nl.ndarray((Mg, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    pv_out(o16, acc_o, Mg, rcp[0:Mg, :], d)
    nisa.dma_copy(
        dst=o_hbm.ap(pattern=[[d, Mg], [1, d]], offset=(tok0 + g * Ug) * hw, scalar_offset=i,
                     indirect_dim=0),
        src=o16,
    )


#: Pipelined attention loop with THREE step buffers (`attn_bufs3`): step s + 1's key
#: transposes and score matmuls are issued before step s's p transposes and PV, so the
#: tensor engine works on them while step s's softmax runs on DVE / ACT (instead of
#: idling); step s + 3 is fetched into step s's buffer once its PV has read it.
#: Every value is computed by the same instructions as `attn_compute`'s.
#: OFF: bit-exact but slower at in8192_bs1 (attention 3.60 -> 4.7 ms with 3 buffers,
#: ~4.5 with 4): the tensor engine still stalls ~8 us a step on the gathers (iter-5 notes).
ATTN_PIPE3 = False
#: Step buffers of that loop (step s + NBUF is fetched into step s's buffer after its
#: PV: NBUF - 2 steps of lead for the gathers) and steps per iteration (a multiple of
#: NBUF, so a step's buffer is the same in every iteration).
ATTN_PIPE3_NBUF = 4
ATTN_PIPE3_STEPS = 4


def attn_bufs3(U, heads, d, kpad):
    """`attn_step_bufs` plus the step's softmax results (written by `attn_softmax`,
    read by the next phases one pipeline stage later)."""
    b = attn_step_bufs(U, heads, d, kpad)
    Ug, Mg = _attn_groups(U, heads)
    G = U // Ug
    b["p16"] = [nl.ndarray((Mg, kpad), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(G)]
    b["rcp"] = [nl.ndarray((Mg, 1), dtype=nl.float32, buffer=nl.sbuf) for _ in range(G)]
    return b


# --------------------------------------------------------------------------- MoE routing


def route_topk(scores_hbm, bias_hbm, lo_hbm, tiles, n_experts, topk, n_local, route_scale,
               wexp_hbm):
    """The router selection: `sqrtsoftplus` of the fp32 logits in `scores_hbm[T, n_experts]`,
    top-`topk` of `score + bias`, raw-score weights normalized to sum 1 and scaled, and
    per-local-expert dense weights `wexp_hbm[T, n_local]` (fp32): the routing weight of
    expert `lo + e` for the row, or 0 when the row did not pick it.
    """
    for r0, nr in tiles:
        lg = nl.ndarray((PT, n_experts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=lg[0:nr, :], src=scores_hbm[r0 : r0 + nr, 0:n_experts])
        sp = nl.ndarray((PT, n_experts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sp[0:nr, :], op=nl.softplus, data=lg[0:nr, :])
        scores = nl.ndarray((PT, n_experts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=scores[0:nr, :], op=nl.sqrt, data=sp[0:nr, :])
        bb = bcast_row(bias_hbm[0:1, 0:n_experts], n_experts, nr)
        sel = nl.ndarray((PT, n_experts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sel[0:nr, :], data1=scores[0:nr, :], data2=bb, op=nl.add)
        best = nl.ndarray((PT, 8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.max8(dst=best[0:nr, :], src=sel[0:nr, :])
        idx8 = nl.ndarray((PT, 8), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.nc_find_index8(dst=idx8[0:nr, :], data=sel[0:nr, :], vals=best[0:nr, :])
        picked = nl.ndarray((PT, topk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.nc_n_gather(dst=picked[0:nr, :], data=scores[0:nr, :], indices=idx8[0:nr, 0:topk])
        tot = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tot[0:nr, :], op=nl.add, data=picked[0:nr, :], axis=(1,), keepdims=True)
        inv = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        inv[0:nr, :] = nl.multiply(nl.reciprocal(nl.add(tot[0:nr, :], 1e-20)), route_scale)
        wts = nl.ndarray((PT, topk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=wts[0:nr, :], data=picked[0:nr, :], op0=nl.multiply, operand0=inv[0:nr, 0:1])
        idf = nl.ndarray((PT, topk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idf[0:nr, :], src=idx8[0:nr, 0:topk])
        lo = bcast_row(lo_hbm[0:1, 0:1], 1, nr)  # fp32 [1, 1] from the host
        wexp = nl.ndarray((PT, n_local), dtype=nl.float32, buffer=nl.sbuf)
        for e in range(n_local):
            target = nl.add(lo, float(e))
            m = nl.ndarray((PT, topk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=m[0:nr, :], data=idf[0:nr, :], op0=nl.equal, operand0=target[0:nr, 0:1])
            nisa.tensor_tensor(dst=m[0:nr, :], data1=m[0:nr, :], data2=wts[0:nr, :], op=nl.multiply)
            nisa.tensor_reduce(dst=wexp[0:nr, e : e + 1], op=nl.add, data=m[0:nr, :], axis=(1,), keepdims=True)
        nisa.dma_copy(dst=wexp_hbm[r0 : r0 + nr, 0:n_local], src=wexp[0:nr, :])


def _swiglu_chunk(inter):
    for cand in (1152, 768, 576, 512, 384, 256, 128, 64, 32):
        if inter % cand == 0:
            return cand
    return inter


def swiglu_rows(gate_hbm, up_hbm, w_hbm, wcol, dst_hbm, tiles, inter, limit, chunk=None):
    """`dst = bf16( silu(min(gate, limit)) * clamp(up, -limit, limit) * w )` for the row
    tiles; `w_hbm[T, n_local]` column `wcol` is the per-row routing weight (or None for
    the shared expert's unit weight). `kernels.swiglu`'s arithmetic.
    """
    chunk = chunk or _swiglu_chunk(inter)
    for r0, nr in tiles:
        if w_hbm is not None:
            w = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w[0:nr, :], src=w_hbm[r0 : r0 + nr, wcol : wcol + 1])
        for j in range(inter // chunk):
            c0 = j * chunk
            g = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=g[0:nr, :], src=gate_hbm[r0 : r0 + nr, c0 : c0 + chunk])
            nisa.tensor_scalar(dst=g[0:nr, :], data=g[0:nr, :], op0=nl.minimum, operand0=limit)
            u = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=u[0:nr, :], src=up_hbm[r0 : r0 + nr, c0 : c0 + chunk])
            nisa.tensor_scalar(dst=u[0:nr, :], data=u[0:nr, :], op0=nl.maximum, operand0=-limit,
                               op1=nl.minimum, operand1=limit)
            sg = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=sg[0:nr, :], op=nl.silu, data=g[0:nr, :])
            # the last product writes bf16 directly (its fp32 result cast on output)
            y16 = nl.ndarray((PT, chunk), dtype=nl.bfloat16, buffer=nl.sbuf)
            if w_hbm is not None:
                y = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=y[0:nr, :], data1=sg[0:nr, :], data2=u[0:nr, :], op=nl.multiply)
                nisa.tensor_scalar(dst=y16[0:nr, :], data=y[0:nr, :], op0=nl.multiply, operand0=w[0:nr, 0:1])
            else:
                nisa.tensor_tensor(dst=y16[0:nr, :], data1=sg[0:nr, :], data2=u[0:nr, :], op=nl.multiply)
            nisa.dma_copy(dst=dst_hbm[r0 : r0 + nr, c0 : c0 + chunk], src=y16[0:nr, :])


def add_rows(acc_hbm, src_hbm, tiles, n, first, chunk=1024):
    """`acc[rows] (+)= src[rows]` in fp32 (`src` bf16 or fp32); `first` initializes."""
    chunk = min(chunk, n)
    for r0, nr in tiles:
        for c in range(n // chunk):
            lo, hi = c * chunk, (c + 1) * chunk
            s = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=s[0:nr, :], src=src_hbm[r0 : r0 + nr, lo:hi])
            if first:
                nisa.dma_copy(dst=acc_hbm[r0 : r0 + nr, lo:hi], src=s[0:nr, :])
            else:
                a = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=a[0:nr, :], src=acc_hbm[r0 : r0 + nr, lo:hi])
                nisa.tensor_tensor(dst=a[0:nr, :], data1=a[0:nr, :], data2=s[0:nr, :], op=nl.add)
                nisa.dma_copy(dst=acc_hbm[r0 : r0 + nr, lo:hi], src=a[0:nr, :])


def cast_rows(src_hbm, dst_hbm, tiles, n, dtype, chunk=1024):
    """`dst[rows] = dtype(src[rows])`."""
    chunk = min(chunk, n)
    for r0, nr in tiles:
        for c in range(n // chunk):
            lo, hi = c * chunk, (c + 1) * chunk
            s = nl.ndarray((PT, chunk), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=s[0:nr, :], src=src_hbm[r0 : r0 + nr, lo:hi])
            nisa.dma_copy(dst=dst_hbm[r0 : r0 + nr, lo:hi], src=s[0:nr, :])


# --------------------------------------------------------------------------- engram


def engram_gather(ids_hbm, table_hbm, row_start_hbm, emb_hbm, tiles, ncol, hd, row_width):
    """`emb[t, j*hd:(j+1)*hd] = table[ids[t, j] - row_start]` where the row is this rank's,
    else 0 -- this rank's contribution to the n-gram rows (summed across ranks after)."""
    for r0, nr in tiles:
        ids = nl.ndarray((PT, ncol), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ids[0:nr, :], src=ids_hbm[r0 : r0 + nr, 0:ncol])
        idf = nl.ndarray((PT, ncol), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idf[0:nr, :], src=ids[0:nr, :])
        rs = bcast_row(row_start_hbm[0:1, 0:1], 1, nr)
        loc = nl.ndarray((PT, ncol), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=loc[0:nr, :], data=idf[0:nr, :], op0=nl.subtract, operand0=rs[0:nr, 0:1])
        # mask = (loc >= 0) * (loc < row_width); clamped = min(max(loc, 0), row_width - 1)
        m0 = nl.ndarray((PT, ncol), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=m0[0:nr, :], data=loc[0:nr, :], op0=nl.greater_equal, operand0=0.0)
        m1 = nl.ndarray((PT, ncol), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=m1[0:nr, :], data=loc[0:nr, :], op0=nl.less, operand0=float(row_width))
        mask = nl.ndarray((PT, ncol), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=mask[0:nr, :], data1=m0[0:nr, :], data2=m1[0:nr, :], op=nl.multiply)
        cl = nl.ndarray((PT, ncol), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=cl[0:nr, :], data=loc[0:nr, :], op0=nl.maximum, operand0=0.0,
                           op1=nl.minimum, operand1=float(row_width - 1))
        cli = nl.ndarray((PT, ncol), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cli[0:nr, :], src=cl[0:nr, :])
        for j in range(ncol):
            g = nl.ndarray((PT, hd), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=g[0:nr, :], src=table_hbm.ap(pattern=[[hd, nr], [1, hd]],
                                                            vector_offset=cli[0:nr, j : j + 1], indirect_dim=0))
            gf = nl.ndarray((PT, hd), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=gf[0:nr, :], data=g[0:nr, :], op0=nl.multiply, operand0=mask[0:nr, j : j + 1])
            g16 = nl.ndarray((PT, hd), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=g16[0:nr, :], src=gf[0:nr, :])
            nisa.dma_copy(dst=emb_hbm[r0 : r0 + nr, j * hd : (j + 1) * hd], src=g16[0:nr, :])


def engram_readout(pid, mine, T, Tp, hc, dim, hash_ids, etable, erow_start, e_wkv4, e_wkv_s, e_qk,
                   strm, stream0, rg_table):
    """The engram read-out of a block, both cores: gather this rank's shard rows of the
    n-gram table, sum the shards over `rg_table` (one rank holds each row, so the
    bf16 sum is exact), the fp8 `wkv` projection (this core's output columns), and
    the gate into `stream0[T, hc*dim]` from `strm`. Allocates its own scratch."""
    ncol = hash_ids.shape[1]
    ehd = etable.shape[1]
    e_in = ncol * ehd
    e_out = e_wkv4.shape[1]
    emb = nl.ndarray((Tp, e_in), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="emb")
    emb_sum = nl.ndarray((Tp, e_in), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="emb_sum")
    ekv = nl.ndarray((Tp, e_out), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="ekv")
    engram_gather(hash_ids, etable, erow_start, emb, mine, ncol, ehd, etable.shape[0])
    if pid == 0 and T < Tp:
        padz = nl.ndarray((PT, e_in), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(padz, 0.0)
        nisa.dma_copy(dst=emb[T:Tp, :], src=padz[0 : Tp - T, :])
    nisa.core_barrier(data=emb, cores=(0, 1))
    ncc.all_reduce(srcs=[emb], dsts=[emb_sum], replica_group=rg_table, op=nl.add)
    tiles = e_out // PT
    a = (tiles + 1) // 2
    lo, hi = (0, a * PT) if pid == 0 else (a * PT, e_out)
    X.gemm_rows(emb_sum, Tp, e_in, e_wkv4, e_wkv_s, ekv, lo, hi)
    nisa.core_barrier(data=ekv, cores=(0, 1))
    if ENGRAM_GATE_SPLIT and Tp == PT and T <= PT and hc % 2 == 0:
        # one row tile: the hc groups (each its own gate) split between the cores
        engram_gate_tile(0, T, strm, ekv, e_qk, stream0, hc, dim,
                         groups=range(pid * (hc // 2), (pid + 1) * (hc // 2)))
    else:
        for r0, nr in mine:
            engram_gate_tile(r0, nr, strm, ekv, e_qk, stream0, hc, dim)
    nisa.core_barrier(data=stream0, cores=(0, 1))


#: Decode (one row tile): `engram_gate_tile`'s hc groups split between the two cores
#: (each group's gate and output depend on its own columns only; the same instructions).
ENGRAM_GATE_SPLIT = True


def engram_gate_tile(r0, nr, h_hbm, kv_hbm, w_hbm, out_hbm, hc, dim, chunk=1024, groups=None):
    """`kernels.engram_gate.gated_add` for one row tile: `h_hbm[T, hc*dim]` bf16 stream,
    `kv_hbm[T, (hc+1)*dim]` bf16 (key | value), `w_hbm[1, hc*dim]` fp32 -> `out_hbm`.
    `groups`: only these of the hc column groups (default all)."""
    width = hc * dim
    nchunks = dim // chunk
    inv_dim = 1.0 / dim
    dim_rsqrt = float(dim) ** -0.5
    vf = []
    for j in range(nchunks):
        v = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=v[0:nr, :], src=kv_hbm[r0 : r0 + nr, width + j * chunk : width + (j + 1) * chunk])
        vf.append(v)
    for c in (range(hc) if groups is None else groups):
        base = c * dim
        hf = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
        hsq = ksq = dotp = None
        for j in range(nchunks):
            lo = base + j * chunk
            hj = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=hj[0:nr, :], src=h_hbm[r0 : r0 + nr, lo : lo + chunk])
            nisa.tensor_copy(dst=hf[0:nr, j * chunk : (j + 1) * chunk], src=hj[0:nr, :])
            kj = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=kj[0:nr, :], src=kv_hbm[r0 : r0 + nr, lo : lo + chunk])
            wj = bcast_row(w_hbm[0:1, lo : lo + chunk], chunk, nr)
            prod = nl.multiply(nl.multiply(hj[0:nr, :], wj), kj[0:nr, :])
            sh = nl.sum(nl.square(hj[0:nr, :]), axis=1, keepdims=True)
            sk = nl.sum(nl.square(kj[0:nr, :]), axis=1, keepdims=True)
            sd = nl.sum(prod, axis=1, keepdims=True)
            hsq = sh if hsq is None else nl.add(hsq, sh)
            ksq = sk if ksq is None else nl.add(ksq, sk)
            dotp = sd if dotp is None else nl.add(dotp, sd)
        rstd = nl.multiply(
            rsqrt_newton(nl.add(nl.multiply(hsq, inv_dim), NORM_EPS)),
            rsqrt_newton(nl.add(nl.multiply(ksq, inv_dim), NORM_EPS)),
        )
        dot = nl.multiply(nl.multiply(dotp, rstd), dim_rsqrt)
        mag = nl.sqrt(nl.maximum(nl.abs(dot), 1e-6))
        sgn = nl.subtract(nl.multiply(nl.copy(nl.greater_equal(dot, 0.0), dtype=nl.float32), 2.0), 1.0)
        gate = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        gate[0:nr, :] = nl.sigmoid(nl.multiply(mag, sgn))
        for j in range(nchunks):
            yv = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=yv[0:nr, :], op=nl.copy, data=vf[j][0:nr, :], scale=gate[0:nr, 0:1])
            nisa.tensor_tensor(dst=yv[0:nr, :], data1=yv[0:nr, :], op=nl.add,
                               data2=hf[0:nr, j * chunk : (j + 1) * chunk])
            y16 = nl.ndarray((PT, chunk), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=y16[0:nr, :], src=yv[0:nr, :])
            nisa.dma_copy(dst=out_hbm[r0 : r0 + nr, base + j * chunk : base + (j + 1) * chunk], src=y16[0:nr, :])


def tap_tile(r0, nr, stream_hbm, out_hbm, hc, dim, chunk=1024):
    """`out[rows] = mean over the hc copies of stream[rows]` in fp32 (the DSpark tap)."""
    for j in range(dim // chunk):
        acc = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
        for c in range(hc):
            t = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=t[0:nr, :], src=stream_hbm[r0 : r0 + nr, c * dim + j * chunk : c * dim + (j + 1) * chunk])
            if c == 0:
                nisa.tensor_copy(dst=acc[0:nr, :], src=t[0:nr, :])
            else:
                nisa.tensor_tensor(dst=acc[0:nr, :], data1=acc[0:nr, :], data2=t[0:nr, :], op=nl.add)
        nisa.tensor_scalar(dst=acc[0:nr, :], data=acc[0:nr, :], op0=nl.multiply, operand0=1.0 / hc)
        nisa.dma_copy(dst=out_hbm[r0 : r0 + nr, j * chunk : (j + 1) * chunk], src=acc[0:nr, :])
