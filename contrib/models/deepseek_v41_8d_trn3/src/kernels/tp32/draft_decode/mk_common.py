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
#: Sinkhorn column sums as one reduce over the transposed view (else hc - 1 adds)
SINKHORN_COLSUM_REDUCE = False
HC_EPS = 1e-06
#: The fp32 matmuls whose stationary holds bf16 values (the hc_in mix GEMM, the router's
#: `gemm_fp32`) skip the stationary's LOW fp32 pass (`is_stationary_onezero`): the tensor
#: engine splits an fp32 stationary into HIGH and LOW halves and runs one weight load +
#: matmul per half; a bf16 value's LOW half is zero, so that pass only accumulates exact zeros.
FP32_BF16_STATIONARY = True
#: hc_in_mix's mix-GEMM operand (and gemm_fp32's) transposed by DMA (`dma_transpose`)
#: instead of by PE transposes, for tiles of at most this many rows. Off: measured slower
#: here (in1024_bs1: each [16, 40, 128] DMA transpose ~11 us, serialized on the queue)
HC_MIX_DMA_T = False
HC_MIX_DMA_T_ROWS = 16
#: hc_in_mix for at most this many rows with no transposes: the contraction in the order
#: k = p * nk + c (partition p holds a row's columns p * nk .. p * nk + nk - 1: one DMA of the
#: rows, cast to fp32 into the K-tile layout `moe_tp.mix_bd` reads, fn.T loaded in the same
#: order by `load_fn_t(spread=True)`), then the block-diagonal mix GEMM (J K-tiles a matmul:
#: 8 matmuls for 5 rows instead of 160 PE transposes + 160 matmuls on core 1) and its sum of
#: squares. 0: off (the fp32 sums then run in the order before).
HC_MIX_SPREAD_ROWS = 0
#: hc_in_mix for at most this many rows (`mix_spread` off for them) with the PE transposes of
#: 128-row stacks of the rows' K-tiles (`PT // nr` K-tiles a transpose: 7 for 5 rows instead of
#: 160), the matmuls and their order as before (exact)
HC_MIX_STACK_ROWS = 16


def mix_spread(nr):
    """Whether `hc_in_mix` of `nr`-row tiles runs the transpose-free path (its `fnT` then
    loaded with `load_fn_t(spread=True)`)."""
    return 0 < nr <= HC_MIX_SPREAD_ROWS
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
    """`1 / sqrt(m)` (round 5: the hardware rsqrt is within 1 fp32 ulp -- no Newton step, which
    `kernels.rmsnorm` did)."""
    return nl.rsqrt(m)


def sum_squares(dst, src, nr, width, chunk):
    """`dst[0:nr] = sum(src[0:nr, 0:width]^2)` in fp32 on the scalar engine."""
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


#: `sinkhorn` with each matrix stored `(hc + 1) x (hc + 1)` (an eps column after every row, an
#: eps row after the last: a reduce over hc + 1 entries is `sum + eps`): a reduce, a DVE
#: reciprocal and a multiply a normalization -- 6 dependent DVE instructions an iteration
#: instead of ~10 (DVE adds, ACT reciprocals). (A fused `(1 / sum) * x` is not an ISA op: DVE
#: has no divide in `tensor_tensor` / `scalar_tensor_tensor`.)
SINKHORN_FUSED = True


def _sinkhorn_fused(t2, nr, hc, after=None):
    f32 = nl.float32
    H = hc + 1
    G = H * H
    n2 = hc * hc

    def v3(t):
        return t.ap(pattern=[[n2, nr], [hc, hc], [1, hc]])

    mx = nl.ndarray((PT, hc), dtype=f32, buffer=nl.sbuf)
    sched_after(nisa.tensor_reduce(dst=mx[0:nr, :], op=nl.maximum, data=v3(t2), axis=(2,), keepdims=False), after)
    dlt = nl.ndarray((PT, n2), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=v3(dlt), data1=v3(t2), data2=mx.ap(pattern=[[hc, nr], [1, hc], [0, hc]]),
                       op=nl.subtract)
    es = nl.ndarray((PT, n2), dtype=f32, buffer=nl.sbuf)
    nisa.activation(dst=es[0:nr, :], op=nl.exp, data=dlt[0:nr, :])
    ssum = nl.ndarray((PT, hc), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ssum[0:nr, :], op=nl.add, data=v3(es), axis=(2,), keepdims=False)
    rs = nl.ndarray((PT, hc), dtype=f32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rs[0:nr, :], data=ssum[0:nr, :])
    a = nl.ndarray((PT, H, H), dtype=f32, buffer=nl.sbuf)
    b = nl.ndarray((PT, H, H), dtype=f32, buffer=nl.sbuf)
    nisa.memset(a, HC_EPS)
    nisa.memset(b, HC_EPS)

    def mat(x):
        return x.ap(pattern=[[G, nr], [H, hc], [1, hc]], offset=0)

    def rows_b(v):
        return v.ap(pattern=[[hc, nr], [1, hc], [0, hc]], offset=0)

    def cols_b(v):
        return v.ap(pattern=[[hc, nr], [0, hc], [1, hc]], offset=0)

    # softmax rows + eps
    nisa.tensor_tensor(dst=mat(a), data1=v3(es), data2=rows_b(rs), op=nl.multiply)
    nisa.tensor_scalar(dst=mat(a), data=mat(a), op0=nl.add, operand0=HC_EPS)

    def rowdiv(src, dst):
        sm = nl.ndarray((PT, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sm[0:nr, :], op=nl.add, data=src.ap(pattern=[[G, nr], [H, hc], [1, H]], offset=0),
                           axis=(2,))
        r = nl.ndarray((PT, hc), dtype=f32, buffer=nl.sbuf)
        nisa.reciprocal(dst=r[0:nr, :], data=sm[0:nr, :])
        nisa.tensor_tensor(dst=mat(dst), data1=mat(src), data2=rows_b(r), op=nl.multiply)

    def coldiv(src, dst):
        cs = nl.ndarray((PT, hc), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=cs[0:nr, :], op=nl.add, data=src.ap(pattern=[[G, nr], [1, hc], [H, H]], offset=0),
                           axis=(2,))
        d = nl.ndarray((PT, hc), dtype=f32, buffer=nl.sbuf)
        nisa.reciprocal(dst=d[0:nr, :], data=cs[0:nr, :])
        nisa.tensor_tensor(dst=dst, data1=mat(src), data2=cols_b(d), op=nl.multiply)

    out = nl.ndarray((PT, n2), dtype=f32, buffer=nl.sbuf)
    if SINKHORN_ITERS == 1:
        coldiv(a, v3(out))
        return out
    coldiv(a, mat(b))
    for it in range(SINKHORN_ITERS - 1):
        rowdiv(b, a)
        coldiv(a, v3(out) if it == SINKHORN_ITERS - 2 else mat(b))
    return out


def sinkhorn(t2, nr, hc, after=None):
    """`softmax(rows) + eps`, then the 20 alternating normalizations of `kernels.hc_in`.
    `after`: schedule the chain (its first instruction; the rest depend on it) after it."""
    if SINKHORN_FUSED:
        return _sinkhorn_fused(t2, nr, hc, after)

    # Each normalization is a handful of instructions over all hc x hc entries at once
    # (the [nr, hc, hc] view, the sums' reciprocals broadcast by a zero-stride pattern):
    # the same per-element operations as one row / column at a time, ~2.5x fewer
    # dependent instructions on the few live rows.
    n2 = hc * hc

    def v3(t):
        return t.ap(pattern=[[n2, nr], [hc, hc], [1, hc]])

    def rowsum_div(src, dst):
        s = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=s[0:nr, :], op=nl.add, data=v3(src), axis=(2,), keepdims=False)
        r = nl.reciprocal(nl.add(s[0:nr, :], HC_EPS))
        nisa.tensor_tensor(dst=v3(dst), data1=v3(src), data2=r.ap(pattern=[[hc, nr], [1, hc], [0, hc]]),
                           op=nl.multiply)

    def colsum_div(src, dst):
        if SINKHORN_COLSUM_REDUCE:
            s = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=s[0:nr, :], op=nl.add, data=src.ap(pattern=[[n2, nr], [1, hc], [hc, hc]]),
                               axis=(2,), keepdims=False)
            s = s[0:nr, :]
        else:
            s = nl.add(src[0:nr, 0:hc], src[0:nr, hc : 2 * hc])
            for i in range(2, hc):
                s = nl.add(s, src[0:nr, i * hc : (i + 1) * hc])
        d = nl.reciprocal(nl.add(s, HC_EPS))
        nisa.tensor_tensor(dst=v3(dst), data1=v3(src), data2=d.ap(pattern=[[hc, nr], [0, hc], [1, hc]]),
                           op=nl.multiply)

    a = nl.ndarray((PT, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
    b = nl.ndarray((PT, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
    # the row softmaxes of all hc rows at once (max / exp / sum over the [nr, hc, hc] view)
    mx = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
    ins = nisa.tensor_reduce(dst=mx[0:nr, :], op=nl.maximum, data=v3(t2), axis=(2,), keepdims=False)
    sched_after(ins, after)
    dlt = nl.ndarray((PT, n2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=v3(dlt), data1=v3(t2), data2=mx.ap(pattern=[[hc, nr], [1, hc], [0, hc]]),
                       op=nl.subtract)
    es = nl.ndarray((PT, n2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=es[0:nr, :], op=nl.exp, data=dlt[0:nr, :])
    ssum = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ssum[0:nr, :], op=nl.add, data=v3(es), axis=(2,), keepdims=False)
    rs = nl.reciprocal(ssum[0:nr, :])
    for i in range(hc):
        lo, hi = i * hc, (i + 1) * hc
        nisa.tensor_scalar(
            dst=a[0:nr, lo:hi], data=es[0:nr, lo:hi],
            op0=nl.multiply, operand0=rs[0:nr, i : i + 1], op1=nl.add, operand1=HC_EPS,
        )
    colsum_div(a, b)
    for _ in range(SINKHORN_ITERS - 1):
        rowsum_div(b, a)
        colsum_div(a, b)
    return b


def hc_in_tile(r0, nr, x2, pre_mix, fnT, base, scale, norm_w, hc, dim, mix,
               normed, pre, post, comb, sq_chunk=1024, out_chunk=1024):
    """One 128-row tile of the hyper-connection sublayer-in (`kernels.hc_in`).

    Reads `x2[r0:r0+nr]` (`[T, hc*dim]` bf16 HBM) and `pre_mix`, writes
    `normed[r0:r0+nr]` (bf16) and the fp32 coefficients `pre`/`post`/`comb`.
    `fnT` is the resident `(PT, nk, mix)` fp32 SBUF copy of `fn.T`.
    The two halves are independent (`hc_in_norm`, `hc_in_mix`): a caller may run
    them on different cores or at different times.
    """
    hc_in_norm(r0, nr, x2, pre_mix, norm_w, hc, dim, normed, sq_chunk, out_chunk)
    hc_in_mix(r0, nr, x2, fnT, base, scale, hc, dim, mix, pre, post, comb, sq_chunk)


def _load_streams(r0, nr, x2, hc, dim):
    xg = []
    for g in range(hc):
        xg.append(nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf))
    for g in range(hc):
        nisa.dma_copy(dst=xg[g][0:nr, :], src=x2[r0 : r0 + nr, g * dim : (g + 1) * dim])
    return xg


def hc_in_norm(r0, nr, x2, pre_mix, norm_w, hc, dim, normed, sq_chunk=1024, out_chunk=1024):
    """`hc_in`'s sublayer input: `normed = rmsnorm(sum_g pre_mix[g] * x2[g]) * norm_w`."""
    inv_dim = 1.0 / dim
    xg = _load_streams(r0, nr, x2, hc, dim)
    pm = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pm[0:nr, :], src=pre_mix[r0 : r0 + nr, 0:hc])

    acc = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=acc[0:nr, :], data=xg[0][0:nr, :], op0=nl.multiply,
        operand0=pm[0:nr, 0:1], engine=nisa.engine.vector,
    )
    for g in range(1, hc):
        nisa.scalar_tensor_tensor(
            dst=acc[0:nr, :], data=xg[g][0:nr, :], op0=nl.multiply,
            operand0=pm[0:nr, g : g + 1], op1=nl.add, operand1=acc[0:nr, :],
        )
    y16 = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=y16[0:nr, :], src=acc[0:nr, :])

    ss2 = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    sum_squares(ss2, y16, nr, dim, sq_chunk)
    rstd = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    rstd[0:nr, :] = rsqrt_newton(nl.add(nl.multiply(ss2[0:nr, 0:1], inv_dim), NORM_EPS))
    for c in range(dim // out_chunk):
        lo, hi = c * out_chunk, (c + 1) * out_chunk
        wj = bcast_row(norm_w[0:1, lo:hi], out_chunk, nr)
        yv = nl.ndarray((PT, out_chunk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=yv[0:nr, :], op=nl.copy, data=y16[0:nr, lo:hi], scale=rstd[0:nr, 0:1])
        nisa.tensor_tensor(dst=yv[0:nr, :], data1=yv[0:nr, :], data2=wj, op=nl.multiply)
        o16 = nl.ndarray((PT, out_chunk), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o16[0:nr, :], src=yv[0:nr, :])
        nisa.dma_copy(dst=normed[r0 : r0 + nr, lo:hi], src=o16[0:nr, :])


def hc_in_norm_split(r0, nr, x2, pre_mix, norm_w, hc, dim, normed, ysc, pid, sq_chunk=1024,
                     out_chunk=1024, cores=2):
    """`hc_in_norm` on both cores (each calls it for the same rows): a column half each
    of the pre-mix sum and of the output, exchanged through `ysc` (`[T, dim]` bf16
    shared HBM) so each core runs the whole sum of squares (the same chain, in the
    same order) for the rms."""
    inv_dim = 1.0 / dim
    nch = dim // out_chunk
    mine = hc_out_chunks(dim, pid, out_chunk, cores)
    c_lo, c_hi = mine[0] * out_chunk, (mine[-1] + 1) * out_chunk
    w = c_hi - c_lo
    xg = []
    for g in range(hc):
        xg.append(nl.ndarray((PT, w), dtype=nl.bfloat16, buffer=nl.sbuf))
    for g in range(hc):
        nisa.dma_copy(dst=xg[g][0:nr, :], src=x2[r0 : r0 + nr, g * dim + c_lo : g * dim + c_hi])
    pm = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pm[0:nr, :], src=pre_mix[r0 : r0 + nr, 0:hc])

    acc = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=acc[0:nr, :], data=xg[0][0:nr, :], op0=nl.multiply,
        operand0=pm[0:nr, 0:1], engine=nisa.engine.vector,
    )
    for g in range(1, hc):
        nisa.scalar_tensor_tensor(
            dst=acc[0:nr, :], data=xg[g][0:nr, :], op0=nl.multiply,
            operand0=pm[0:nr, g : g + 1], op1=nl.add, operand1=acc[0:nr, :],
        )
    y16 = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=y16[0:nr, c_lo:c_hi], src=acc[0:nr, :])
    nisa.dma_copy(dst=ysc[r0 : r0 + nr, c_lo:c_hi], src=y16[0:nr, c_lo:c_hi])
    nisa.core_barrier(data=ysc, cores=(0, 1))
    if c_lo > 0:
        nisa.dma_copy(dst=y16[0:nr, 0:c_lo], src=ysc[r0 : r0 + nr, 0:c_lo])
    if c_hi < dim:
        nisa.dma_copy(dst=y16[0:nr, c_hi:dim], src=ysc[r0 : r0 + nr, c_hi:dim])

    ss2 = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    sum_squares(ss2, y16, nr, dim, sq_chunk)
    rstd = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    rstd[0:nr, :] = rsqrt_newton(nl.add(nl.multiply(ss2[0:nr, 0:1], inv_dim), NORM_EPS))
    for c in mine:
        lo, hi = c * out_chunk, (c + 1) * out_chunk
        wj = bcast_row(norm_w[0:1, lo:hi], out_chunk, nr)
        yv = nl.ndarray((PT, out_chunk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=yv[0:nr, :], op=nl.copy, data=y16[0:nr, lo:hi], scale=rstd[0:nr, 0:1])
        nisa.tensor_tensor(dst=yv[0:nr, :], data1=yv[0:nr, :], data2=wj, op=nl.multiply)
        o16 = nl.ndarray((PT, out_chunk), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o16[0:nr, :], src=yv[0:nr, :])
        nisa.dma_copy(dst=normed[r0 : r0 + nr, lo:hi], src=o16[0:nr, :])


#: hc_in_norm_spread's rms on chip in the spread layout (`moe_tp.spread_rstd`: each partition's
#: sum of squares, the rows' sums and the rstd broadcast back by selector matmuls) instead of the
#: `ysc` / `rsc` HBM round trips and the sum of squares on the rows' few partitions (the fp32 sum
#: of squares then runs in another order)
NORM_SS_SPREAD = False


def hc_in_norm_spread(nr, x2, pre_mix, norm_w, hc, dim, normed, ysc, rsc, sq_chunk=1024, r0=0, store_y=True):
    """`hc_in_norm` of rows `0 .. nr` on one core, the elementwise parts with the rows'
    column blocks spread over the partitions (as `hc_out_spread`); the sum of squares
    runs on the rows' layout (`ysc` `[T, dim]` bf16 HBM round trip: the same chain in
    the same order), its rms back to the spread rows through `rsc` (`[T, 1]` fp32 HBM).
    `r0`: the rows `r0 .. r0 + nr` (a core's share; the per-row ops do not depend on it)."""
    inv_dim = 1.0 / dim
    width = hc * dim
    nblk, w = _spread_blocks(nr, dim)
    P = nr * nblk
    rmajor = nr <= nblk
    pm = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
    rg = nl.ndarray((PT, hc, w), dtype=nl.bfloat16, buffer=nl.sbuf)
    wn = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
    if rmajor:
        for r in range(nr):
            p0 = r * nblk
            nisa.dma_copy(dst=pm[p0 : p0 + nblk, :], src=pre_mix.ap(pattern=[[0, nblk], [1, hc]], offset=(r0 + r) * hc))
            nisa.dma_copy(dst=rg[p0 : p0 + nblk, 0:hc, :],
                          src=x2.ap(pattern=[[w, nblk], [dim, hc], [1, w]], offset=(r0 + r) * width))
            nisa.dma_copy(dst=wn[p0 : p0 + nblk, :], src=norm_w.ap(pattern=[[w, nblk], [1, w]], offset=0))
    else:
        for b in range(nblk):
            p0 = b * nr
            nisa.dma_copy(dst=pm[p0 : p0 + nr, :], src=pre_mix[r0 : r0 + nr, 0:hc])
            nisa.dma_copy(dst=rg[p0 : p0 + nr, 0:hc, :],
                          src=x2.ap(pattern=[[width, nr], [dim, hc], [1, w]], offset=r0 * width + b * w))
            nisa.dma_copy(dst=wn[p0 : p0 + nr, :], src=norm_w.ap(pattern=[[0, nr], [1, w]], offset=b * w))

    acc = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=acc[0:P, :], data=rg[0:P, 0, :], op0=nl.multiply,
        operand0=pm[0:P, 0:1], engine=nisa.engine.vector,
    )
    for g in range(1, hc):
        nisa.scalar_tensor_tensor(
            dst=acc[0:P, :], data=rg[0:P, g, :], op0=nl.multiply,
            operand0=pm[0:P, g : g + 1], op1=nl.add, operand1=acc[0:P, :],
        )
    y16 = nl.ndarray((PT, w), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=y16[0:P, :], src=acc[0:P, :])
    if rmajor and NORM_SS_SPREAD:
        from . import moe_tp as MT

        rs = MT.spread_rstd(y16, nr, nblk, "t", w, NORM_EPS, None)
        if store_y:
            # (`ysc` the mixed rows when the caller reads them: the head's confidence input)
            nisa.dma_copy(dst=ysc.ap(pattern=[[w, P], [1, w]], offset=r0 * dim), src=y16[0:P, :])
    elif rmajor:
        # partition r * nblk + b holds row r's columns b * w ..: the rows' `nblk * w = dim`
        # columns are contiguous rows of `ysc`, so one DMA writes them all
        nisa.dma_copy(dst=ysc.ap(pattern=[[w, P], [1, w]], offset=r0 * dim), src=y16[0:P, :])
    else:
        for b in range(nblk):
            p0 = b * nr
            nisa.dma_copy(dst=ysc[r0 : r0 + nr, b * w : (b + 1) * w], src=y16[p0 : p0 + nr, :])
    if not (rmajor and NORM_SS_SPREAD):
        yr = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=yr[0:nr, :], src=ysc[r0 : r0 + nr, 0:dim])
        ss2 = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        sum_squares(ss2, yr, nr, dim, sq_chunk)
        rstd = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        rstd[0:nr, :] = rsqrt_newton(nl.add(nl.multiply(ss2[0:nr, 0:1], inv_dim), NORM_EPS))
        nisa.dma_copy(dst=rsc[r0 : r0 + nr, 0:1], src=rstd[0:nr, 0:1])
        rs = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        if rmajor:
            for r in range(nr):
                nisa.dma_copy(dst=rs[r * nblk : (r + 1) * nblk, 0:1],
                              src=rsc.ap(pattern=[[0, nblk], [1, 1]], offset=r0 + r))
        else:
            for b in range(nblk):
                nisa.dma_copy(dst=rs[b * nr : (b + 1) * nr, 0:1], src=rsc[r0 : r0 + nr, 0:1])
    yv = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=yv[0:P, :], op=nl.copy, data=y16[0:P, :], scale=rs[0:P, 0:1])
    nisa.tensor_tensor(dst=yv[0:P, :], data1=yv[0:P, :], data2=wn[0:P, :], op=nl.multiply)
    o16 = nl.ndarray((PT, w), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o16[0:P, :], src=yv[0:P, :])
    if rmajor:
        nisa.dma_copy(dst=normed.ap(pattern=[[w, P], [1, w]], offset=r0 * dim), src=o16[0:P, :])
    else:
        for b in range(nblk):
            p0 = b * nr
            nisa.dma_copy(dst=normed[r0 : r0 + nr, b * w : (b + 1) * w], src=o16[p0 : p0 + nr, :])


def _mix_spread(r0, nr, x2, fnT, width, mix):
    """`HC_MIX_SPREAD_ROWS`' mix GEMM: `(mix PSUM [nr, mix], ss SBUF [nr, 1])` of rows
    `r0 .. r0 + nr` of `x2` (`[T, width]` bf16 HBM), `fnT` from `load_fn_t(spread=True)`."""
    from . import moe_tp as MT

    nk = width // PT
    xs = nl.ndarray((PT, nr, nk), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=x2.ap(pattern=[[nk, PT], [width, nr], [1, nk]], offset=r0 * width))
    # t32c[p, c * nr + r] = x[r, p * nk + c]: K-tile c's rows at columns c * nr ..
    t32c = nl.ndarray((PT, nk * nr), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=t32c.ap(pattern=[[nk * nr, PT], [1, nr], [nr, nk]], offset=0), src=xs)
    key = ("bd", nr, mix)
    if key not in MT.CACHE:
        MT.CACHE[key] = MT.bd_consts(nr, mix)
    ss = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    return MT.mix_bd(t32c, fnT, nr, nk, mix, MT.CACHE[key], ss_out=ss), ss


def hc_in_mix(r0, nr, x2, fnT, base, scale, hc, dim, mix, pre, post, comb, sq_chunk=1024, sink=True,
              spread=False):
    """`hc_in`'s coefficients: the mix GEMM of the rms-scaled streams, then `pre` / `post`
    (sigmoids) and `comb` (Sinkhorn), fp32 into HBM. `sink=False`: stop before the
    Sinkhorn and return its input (an SBUF tile) for `hc_in_mix_sink`. `spread`: `fnT` is
    `load_fn_t(spread=True)`'s (the transpose-free path, `mix_spread(nr)`)."""
    width = hc * dim
    kpg = dim // PT
    inv_width = 1.0 / width
    if spread:
        mixp, ss = _mix_spread(r0, nr, x2, fnT, width, mix)
        return _mix_coefs(r0, nr, mixp, ss, inv_width, base, scale, hc, mix, pre, post, comb, sink)
    xg = _load_streams(r0, nr, x2, hc, dim)

    ss = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    junk = nl.ndarray((PT, sq_chunk), dtype=nl.float32, buffer=nl.sbuf)
    nsq = dim // sq_chunk
    for g in range(hc):
        for c in range(nsq):
            first = g == 0 and c == 0
            last = g == hc - 1 and c == nsq - 1
            nisa.activation(
                dst=junk[0:nr, :], op=nl.square,
                data=xg[g][0:nr, c * sq_chunk : (c + 1) * sq_chunk],
                reduce_op=nl.add,
                reduce_cmd=nisa.reduce_cmd.reset_reduce if first else nisa.reduce_cmd.reduce,
                reduce_res=ss[0:nr, 0:1] if last else None,
            )

    nkb = hc * kpg
    mixp = nl.ndarray((PT, mix), dtype=nl.float32, buffer=nl.psum)
    if 0 < nr <= HC_MIX_STACK_ROWS:
        # the rows' K-tiles stacked on the partitions (partition r * js + j = row r's 128-column
        # chunk k0 + j: one DMA a row and group) and transposed by ONE PE transpose a group of
        # js = PT // nr chunks into x^T's K-tile columns (kb * nr + r), then the same matmuls in
        # the same order (the same products and fp32 accumulation: exact)
        J = PT // nr
        nk = nkb
        t32c = nl.ndarray((PT, nk * nr), dtype=nl.float32, buffer=nl.sbuf)
        x2c = x2.reshape((x2.shape[0] * nk, PT))
        for k0 in range(0, nk, J):
            js = min(J, nk - k0)
            n = js * nr
            xs = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
            for r in range(nr):
                k = (r0 + r) * nk + k0
                nisa.dma_copy(dst=xs[r * js : (r + 1) * js, 0:PT], src=x2c[k : k + js, 0:PT],
                              dge_mode=nisa.dge_mode.hwdge,
                              engine=nisa.engine.sync if r % 2 == 0 else nisa.engine.scalar)
            tps = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps[0:PT, 0:n], data=xs[0:n, 0:PT])
            # t32c[:, (k0 + j) * nr + r] = tps[:, r * js + j]
            nisa.tensor_copy(dst=t32c.ap(pattern=[[nk * nr, PT], [nr, js], [1, nr]], offset=k0 * nr),
                             src=tps.ap(pattern=[[PT, PT], [1, js], [js, nr]]))
        for kb in range(nk):
            nisa.nc_matmul(
                dst=mixp[0:nr, 0:mix], stationary=t32c[0:PT, kb * nr : (kb + 1) * nr],
                moving=fnT[0:PT, kb, 0:mix], accumulate=(kb > 0),
                is_stationary_onezero=FP32_BF16_STATIONARY,
            )
    elif HC_MIX_DMA_T and nr <= HC_MIX_DMA_T_ROWS:
        # few rows: each stream's rows transposed by DMA (`[16, kpg, 128]` ->
        # `[128, kpg, 16]`; the columns past nr are never read) instead of kpg PE
        # transposes: the tensor engine runs only the matmuls
        a_t = 16
        nq = cdiv(nr, a_t)
        for g in range(hc):
            t16 = nl.ndarray((PT, kpg, nq * a_t), dtype=nl.bfloat16, buffer=nl.sbuf)
            for q in range(nq):
                nisa.dma_transpose(dst=t16[0:PT, :, q * a_t : (q + 1) * a_t],
                                   src=xg[g].ap(pattern=[[dim, a_t], [PT, kpg], [1, PT]], offset=q * a_t * dim),
                                   axes=(2, 1, 0))
            t32 = nl.ndarray((PT, kpg, nq * a_t), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=t32[0:PT, :, 0:nr], src=t16[0:PT, :, 0:nr], engine=nisa.vector_engine)
            for c in range(kpg):
                kb = g * kpg + c
                nisa.nc_matmul(
                    dst=mixp[0:nr, 0:mix], stationary=t32[0:PT, c, 0:nr],
                    moving=fnT[0:PT, kb, 0:mix], accumulate=(kb > 0),
                    is_stationary_onezero=FP32_BF16_STATIONARY,
                )
    else:
        # the streams transposed on the PE, as many 128-column blocks per PSUM bank as fit
        # (one copy to fp32 SBUF a bank instead of one a block), then the fp32 mix GEMM
        nrp = -(-nr // 8) * 8              # 16-byte aligned blocks
        nb = max(1, min(kpg, 1024 // nrp))
        while kpg % nb:
            nb -= 1
        for b0 in range(0, nkb, nb):
            tps = nl.ndarray((PT, nb, nrp), dtype=nl.bfloat16, buffer=nl.psum)
            for j in range(nb):
                kb = b0 + j
                g, c = kb // kpg, kb % kpg
                nisa.nc_transpose(dst=tps[0:PT, j, 0:nr], data=xg[g][0:nr, c * PT : (c + 1) * PT])
            xT = nl.ndarray((PT, nb, nrp), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=xT[0:PT, 0:nb, 0:nr], src=tps[0:PT, 0:nb, 0:nr])
            for j in range(nb):
                kb = b0 + j
                nisa.nc_matmul(
                    dst=mixp[0:nr, 0:mix], stationary=xT[0:PT, j, 0:nr],
                    moving=fnT[0:PT, kb, 0:mix], accumulate=(kb > 0),
                    is_stationary_onezero=FP32_BF16_STATIONARY,
                )

    return _mix_coefs(r0, nr, mixp, ss, inv_width, base, scale, hc, mix, pre, post, comb, sink)


def _mix_coefs(r0, nr, mixp, ss, inv_width, base, scale, hc, mix, pre, post, comb, sink):
    """`hc_in_mix` from its mix GEMM (`mixp`, PSUM) and sum of squares (`ss`) on."""
    rs = rsqrt_newton(nl.add(nl.multiply(ss[0:nr, 0:1], inv_width), NORM_EPS))
    mixv = nl.ndarray((PT, mix), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=mixv[0:nr, :], data=mixp[0:nr, :], op0=nl.multiply, operand0=rs)

    s0 = bcast_row(scale[0:1, 0:1], 1, nr)
    s1 = bcast_row(scale[0:1, 1:2], 1, nr)
    s2 = bcast_row(scale[0:1, 2:3], 1, nr)
    b0 = bcast_row(base[0:1, 0:hc], hc, nr)
    b1 = bcast_row(base[0:1, hc : 2 * hc], hc, nr)
    b2 = bcast_row(base[0:1, 2 * hc : 2 * hc + hc * hc], hc * hc, nr)

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

    t2 = nl.ndarray((PT, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=t2[0:nr, :], data=mixv[0:nr, 2 * hc : 2 * hc + hc * hc],
        op0=nl.multiply, operand0=s2, op1=nl.add, operand1=b2,
    )
    if not sink:
        return t2
    hc_in_mix_sink(r0, nr, t2, hc, comb)


def hc_in_mix_sink(r0, nr, t2, hc, comb, after=None):
    """`hc_in_mix`'s Sinkhorn of its returned tile `t2` -> `comb` (fp32 HBM), schedule-pinned
    after `after` (else the list scheduler hoists it into the work before the all_reduce)."""
    comb_t = sinkhorn(t2, nr, hc, after=after)
    nisa.dma_copy(dst=comb[r0 : r0 + nr, 0 : hc * hc], src=comb_t[0:nr, :])


def hc_out_chunks(dim, pid, chunk=1024, cores=2):
    """The `hc_out_tile` column blocks core `pid` takes when both cores split a tile."""
    n = dim // chunk
    a = (n + cores - 1) // cores
    return list(range(0, a)) if pid == 0 else list(range(a, n))


def load_fn_t(fn_t, width, mix, spread=False):
    """`fn.T` (`[hc*dim, mix]` fp32 HBM) resident as `(PT, nk, mix)` fp32 SBUF: K-tile c at
    `[:, c]`, row k = c * PT + p at partition p (`spread`: k = p * nk + c, one contiguous run a
    partition -- `hc_in_mix(spread=True)`'s order)."""
    nk = width // PT
    fnT = nl.ndarray((PT, nk, mix), dtype=nl.float32, buffer=nl.sbuf)
    pat = [[nk * mix, PT], [mix, nk], [1, mix]] if spread else [[mix, PT], [PT * mix, nk], [1, mix]]
    nisa.dma_copy(dst=fnT, src=fn_t.ap(pattern=pat), dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    return fnT


def hc_out_tile(r0, nr, x_hbm, res_hbm, post_hbm, comb_hbm, y_hbm, hc, dim, chunk=1024,
                x_dtype=nl.float32, chunks=None):
    """One 128-row tile of `hc_post` (`kernels.hc_attn_out`):
    `y[t, j, :] = post[t, j] * x[t, :] + sum_i comb[t, i, j] * res[t, i, :]`.

    `x_hbm` is the sublayer output `[T, dim]` (fp32 or bf16 -- it is widened to fp32
    for the products either way, matching `out.to(x.dtype)` followed by the fp32
    kernel math when `x_dtype` is bf16 and the all-reduced fp32 partial when fp32),
    `res_hbm` the `[T, hc*dim]` bf16 residual stream, `y_hbm` the `[T, hc*dim]` bf16 output.
    `chunks` (if given) the `chunk`-column blocks to do (the columns are independent).
    """
    post = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=post[0:nr, :], src=post_hbm[r0 : r0 + nr, 0:hc])
    comb = nl.ndarray((PT, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=comb[0:nr, :], src=comb_hbm[r0 : r0 + nr, 0 : hc * hc])
    for c in (range(dim // chunk) if chunks is None else chunks):
        col0 = c * chunk
        xg = nl.ndarray((PT, chunk), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=xg[0:nr, :], src=x_hbm[r0 : r0 + nr, col0 : col0 + chunk])
        rg = nl.ndarray((PT, hc, chunk), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i in range(hc):
            nisa.dma_copy(
                dst=rg[0:nr, i, :],
                src=res_hbm[r0 : r0 + nr, i * dim + col0 : i * dim + col0 + chunk],
            )
        og = nl.ndarray((PT, hc, chunk), dtype=nl.bfloat16, buffer=nl.sbuf)
        acc = nl.ndarray((PT, hc, chunk), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(hc):
            nisa.tensor_scalar(
                dst=acc[0:nr, j, :], data=rg[0:nr, 0, :], op0=nl.multiply,
                operand0=comb[0:nr, j : j + 1], engine=nisa.engine.scalar,
            )
            for i in range(1, hc):
                col = i * hc + j
                nisa.scalar_tensor_tensor(
                    dst=acc[0:nr, j, :], data=rg[0:nr, i, :], op0=nl.multiply,
                    operand0=comb[0:nr, col : col + 1], op1=nl.add, operand1=acc[0:nr, j, :],
                )
            nisa.scalar_tensor_tensor(
                dst=og[0:nr, j, :], data=xg[0:nr, :], op0=nl.multiply,
                operand0=post[0:nr, j : j + 1], op1=nl.add, operand1=acc[0:nr, j, :],
            )
        for j in range(hc):
            nisa.dma_copy(
                dst=y_hbm[r0 : r0 + nr, j * dim + col0 : j * dim + col0 + chunk],
                src=og[0:nr, j, :],
            )


def _spread_blocks(nr, ncols):
    """Column blocks of a `[nr, ncols]` row set spread over the partitions: the most
    blocks `nblk` (dividing `ncols`) with `nr * nblk <= 128`."""
    nblk = max(1, PT // nr)
    while ncols % nblk:
        nblk -= 1
    return nblk, ncols // nblk


def hc_out_spread(nr, x_hbm, res_hbm, post_hbm, comb_hbm, y_hbm, hc, dim, pid, cores=2):
    """`hc_out_tile` of rows `0 .. nr` (`nr <= 128`), core `pid` doing its `1 / cores`
    of the columns, with the rows' column blocks spread over the partitions (`nr * nblk`
    partitions of `w` columns instead of `nr` partitions of the whole width): the same
    per-element operations (the per-row `post` / `comb` scalars replicated to each of
    the row's partitions), a fraction of the instructions' length."""
    width = hc * dim
    part = dim // cores
    assert part * cores == dim
    c0 = pid * part
    nblk, w = _spread_blocks(nr, part)
    P = nr * nblk
    rmajor = nr <= nblk          # partition r * nblk + b (else b * nr + r): fewer DMAs
    # on a hardware DGE queue: the loads wait for the all_reduce, and on GpSimd's queue
    # they held up the next stage's KV write there
    hw = dict(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    post = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
    comb = nl.ndarray((PT, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
    xg = nl.ndarray((PT, w), dtype=nl.bfloat16, buffer=nl.sbuf)
    rg = nl.ndarray((PT, hc, w), dtype=nl.bfloat16, buffer=nl.sbuf)
    if rmajor:
        for r in range(nr):
            p0 = r * nblk
            nisa.dma_copy(dst=post[p0 : p0 + nblk, :], src=post_hbm.ap(pattern=[[0, nblk], [1, hc]], offset=r * hc), **hw)
            nisa.dma_copy(dst=comb[p0 : p0 + nblk, :],
                          src=comb_hbm.ap(pattern=[[0, nblk], [1, hc * hc]], offset=r * hc * hc), **hw)
            nisa.dma_copy(dst=xg[p0 : p0 + nblk, :], src=x_hbm.ap(pattern=[[w, nblk], [1, w]], offset=r * dim + c0))  # (a casting DMA: not on HWDGE)
            nisa.dma_copy(dst=rg[p0 : p0 + nblk, 0:hc, :],
                          src=res_hbm.ap(pattern=[[w, nblk], [dim, hc], [1, w]], offset=r * width + c0), **hw)
    else:
        for b in range(nblk):
            p0 = b * nr
            col = c0 + b * w
            nisa.dma_copy(dst=post[p0 : p0 + nr, :], src=post_hbm[0:nr, 0:hc], **hw)
            nisa.dma_copy(dst=comb[p0 : p0 + nr, :], src=comb_hbm[0:nr, 0 : hc * hc], **hw)
            nisa.dma_copy(dst=xg[p0 : p0 + nr, :], src=x_hbm[0:nr, col : col + w])
            nisa.dma_copy(dst=rg[p0 : p0 + nr, 0:hc, :],
                          src=res_hbm.ap(pattern=[[width, nr], [dim, hc], [1, w]], offset=col), **hw)
    og = nl.ndarray((PT, hc, w), dtype=nl.bfloat16, buffer=nl.sbuf)
    acc = nl.ndarray((PT, hc, w), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(hc):
        nisa.tensor_scalar(
            dst=acc[0:P, j, :], data=rg[0:P, 0, :], op0=nl.multiply,
            operand0=comb[0:P, j : j + 1], engine=nisa.engine.scalar,
        )
        for i in range(1, hc):
            col = i * hc + j
            nisa.scalar_tensor_tensor(
                dst=acc[0:P, j, :], data=rg[0:P, i, :], op0=nl.multiply,
                operand0=comb[0:P, col : col + 1], op1=nl.add, operand1=acc[0:P, j, :],
            )
        nisa.scalar_tensor_tensor(
            dst=og[0:P, j, :], data=xg[0:P, :], op0=nl.multiply,
            operand0=post[0:P, j : j + 1], op1=nl.add, operand1=acc[0:P, j, :],
        )
    if rmajor:
        for r in range(nr):
            p0 = r * nblk
            nisa.dma_copy(dst=y_hbm.ap(pattern=[[w, nblk], [dim, hc], [1, w]], offset=r * width + c0),
                          src=og[p0 : p0 + nblk, 0:hc, :], **hw)
    else:
        for b in range(nblk):
            p0 = b * nr
            nisa.dma_copy(dst=y_hbm.ap(pattern=[[width, nr], [dim, hc], [1, w]], offset=c0 + b * w),
                          src=og[p0 : p0 + nr, 0:hc, :], **hw)


# --------------------------------------------------------------------------- norms, rope, quant


def norm_rows(src_hbm, w_hbm, dst_hbm, tiles, n, qdq, chunk=None):
    """RMSNorm of `src_hbm[r0:r0+nr, 0:n]` (bf16) into `dst_hbm` (bf16) for the given
    row tiles, optionally followed by the fp8 activation round trip (`qdq`).
    """
    chunk = chunk or n
    inv_n = 1.0 / n
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
            wj = bcast_row(w_hbm[0:1, lo:hi], chunk, nr)
            yv = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=yv[0:nr, :], op=nl.copy, data=t[0:nr, lo:hi], scale=rstd[0:nr, 0:1])
            nisa.tensor_tensor(dst=yv[0:nr, :], data1=yv[0:nr, :], data2=wj, op=nl.multiply)
            nisa.tensor_copy(dst=y16[0:nr, lo:hi], src=yv[0:nr, :])
        if qdq:
            out = _qdq_fp8(y16, PT, n)
            nisa.dma_copy(dst=dst_hbm[r0 : r0 + nr, 0:n], src=out[0:nr, :])
        else:
            nisa.dma_copy(dst=dst_hbm[r0 : r0 + nr, 0:n], src=y16[0:nr, :])


def norm_rows_sb(src_hbm, w_hbm, nr, n, mr, chunk=None):
    """`norm_rows` (no qdq) of rows `0 .. nr` kept on chip: a bf16 SBUF `[128, n]` tile with
    rows `nr .. mr` zero (the same per-row ops; for a GEMM staging the first `mr` rows)."""
    chunk = chunk or n
    inv_n = 1.0 / n
    t = nl.ndarray((PT, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=t[0:nr, :], src=src_hbm[0:nr, 0:n])
    ss = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    sum_squares(ss, t, nr, n, chunk)
    rstd = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    rstd[0:nr, :] = rsqrt_newton(nl.add(nl.multiply(ss[0:nr, 0:1], inv_n), NORM_EPS))
    y16 = nl.ndarray((PT, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    if nr < mr:
        nisa.memset(y16[0:mr, :], 0.0)
    for c in range(n // chunk):
        lo, hi = c * chunk, (c + 1) * chunk
        wj = bcast_row(w_hbm[0:1, lo:hi], chunk, nr)
        yv = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=yv[0:nr, :], op=nl.copy, data=t[0:nr, lo:hi], scale=rstd[0:nr, 0:1])
        nisa.tensor_tensor(dst=yv[0:nr, :], data1=yv[0:nr, :], data2=wj, op=nl.multiply)
        nisa.tensor_copy(dst=y16[0:nr, lo:hi], src=yv[0:nr, :])
    return y16


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
              inverse, qdq=False):
    """Rope the last `ROPE_DIM` columns of each of `nheads` heads of width `head`
    in `src_hbm[rows, width]` (bf16), optionally fp8-round-trip the rows, and write
    `dst_hbm`. `src_hbm` and `dst_hbm` may be the same tensor.
    """
    for r0, nr in tiles:
        t = nl.ndarray((PT, width), dtype=nl.bfloat16, buffer=nl.sbuf)
        if qdq and nr < PT:
            nisa.memset(t, 0.0)
        nisa.dma_copy(dst=t[0:nr, :], src=src_hbm[r0 : r0 + nr, 0:width])
        cos, sin = load_freqs(freqs_hbm, row_pos_hbm, r0, nr)
        for h in range(nheads):
            rope_inplace(t, nr, h * head + head - ROPE_DIM, cos, sin, inverse)
        if qdq:
            t = _qdq_fp8(t, PT, width)
        nisa.dma_copy(dst=dst_hbm[r0 : r0 + nr, 0:width], src=t[0:nr, :])


# --------------------------------------------------------------------------- GEMMs (fp32 router)


def gemm_fp32(x_hbm, w_t_hbm, out_hbm, tiles, k, n, c0=0):
    """`out[rows, c0:c0+n] = x[rows, k] (bf16) @ w_t[k, c0:c0+n] (fp32)` with **fp32
    operands** on the tensor engine (the router's `matmul(x.float(), w.float().t())`).
    `w_t`'s columns are staged resident as `(PT, k//PT, n)`; `n <= 512`. Each output
    column is its own accumulation, so a caller may split the columns between cores. A bf16
    `w_t` (exact bf16 values) runs bf16 operands.
    """
    nk = k // PT
    N = w_t_hbm.shape[1]
    # (a bf16 weight -- the layout's exact copy of a bf16-valued fp32 router -- runs one bf16
    # pass: the same exact products, accumulated in fp32 PSUM)
    bf = w_t_hbm.dtype == nl.bfloat16
    wdt = nl.bfloat16 if bf else nl.float32
    wT = nl.ndarray((PT, nk, n), dtype=wdt, buffer=nl.sbuf)
    nisa.dma_copy(dst=wT, src=w_t_hbm.ap(pattern=[[N, PT], [PT * N, nk], [1, n]], offset=c0))
    for r0, nr in tiles:
        x = nl.ndarray((PT, k), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=x[0:nr, :], src=x_hbm[r0 : r0 + nr, 0:k])
        acc = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.psum)
        if HC_MIX_DMA_T and nr <= HC_MIX_DMA_T_ROWS:
            # few rows: the operand transposed by one DMA transpose (as hc_in_mix's mix
            # GEMM; columns past nr never read), the tensor engine runs only the matmuls
            t16 = nl.ndarray((PT, nk, 16), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_transpose(dst=t16, src=x.ap(pattern=[[k, 16], [PT, nk], [1, PT]], offset=0),
                               axes=(2, 1, 0))
            if bf:
                t32 = t16
            else:
                t32 = nl.ndarray((PT, nk, 16), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=t32[0:PT, :, 0:nr], src=t16[0:PT, :, 0:nr])
            for kb in range(nk):
                nisa.nc_matmul(dst=acc[0:nr, 0:n], stationary=t32[0:PT, kb, 0:nr],
                               moving=wT[0:PT, kb, 0:n], accumulate=(kb > 0),
                               **({} if bf else dict(is_stationary_onezero=FP32_BF16_STATIONARY)))
        # the PE transposes of as many blocks as a PSUM bank holds, one copy a bank
        nrp = -(-nr // 8) * 8
        nb = max(1, min(nk, 1024 // nrp))
        while nk % nb:
            nb -= 1
        for b0 in (range(0, nk, nb) if not (HC_MIX_DMA_T and nr <= HC_MIX_DMA_T_ROWS) else ()):
            tps = nl.ndarray((PT, nb, nrp), dtype=nl.bfloat16, buffer=nl.psum)
            for j in range(nb):
                kb = b0 + j
                nisa.nc_transpose(dst=tps[0:PT, j, 0:nr], data=x[0:nr, kb * PT : (kb + 1) * PT])
            xT = nl.ndarray((PT, nb, nrp), dtype=wdt, buffer=nl.sbuf)
            nisa.tensor_copy(dst=xT[0:PT, 0:nb, 0:nr], src=tps[0:PT, 0:nb, 0:nr])
            for j in range(nb):
                kb = b0 + j
                nisa.nc_matmul(dst=acc[0:nr, 0:n], stationary=xT[0:PT, j, 0:nr],
                               moving=wT[0:PT, kb, 0:n], accumulate=(kb > 0),
                               **({} if bf else dict(is_stationary_onezero=FP32_BF16_STATIONARY)))
        o = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o[0:nr, :], src=acc[0:nr, :])
        nisa.dma_copy(dst=out_hbm[r0 : r0 + nr, c0 : c0 + n], src=o[0:nr, :])


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


#: `copy_rows_hw`: rows (of 512 elements) per DMA (~1 MB pieces) and the widest view row.
COPY_ROWS = 1024
COPY_ELEMS = 8192
#: The hardware DGE queues the copies alternate over.
COPY_ENGINES = (nisa.engine.scalar, nisa.engine.sync)
#: The copies' DMA QoS priority (0-3, lower is higher): the lowest, below the weight and
#: activation DMAs they share the queues / DMA engines with (None: the default).
COPY_PRIORITY = 3


def sched_after(ins, after):
    """Schedule-only dependency of `ins` on `after` (no hardware sync), if both exist."""
    if after is not None and ins is not None and hasattr(ins, "depends_on"):
        ins.depends_on(after, kind=nisa.dependency.schedule)


def copy_rows_hw(src, dst, pid, after=None, cores=2, part=(0, 1)):
    """`dst = src` (HBM -> HBM, `[rows, n]`), core `pid` copying its `1 / cores` of the rows
    in DMAs of ~`COPY_ROWS` x 512 elements on the hardware DGE queues `COPY_ENGINES`,
    schedule-pinned after `after`. The tensors are viewed as `[rows / g, n * g]` (up to
    `COPY_ELEMS` contiguous elements a view row): fewer, larger descriptors. `part = (k, m)`:
    only the k-th of m equal row ranges (still split over the cores)."""
    rows, n = src.shape
    g = 1
    for cand in (64, 32, 16, 8, 4, 2):
        if rows % cand == 0 and n * cand <= COPY_ELEMS:
            g = cand
            break
    src = src.reshape((rows // g, n * g))
    dst = dst.reshape((rows // g, n * g))
    rows = rows // g
    step = max(1, COPY_ROWS * 512 // (n * g))
    k, m = part
    p0, p1 = rows * k // m, rows * (k + 1) // m
    per = cdiv(p1 - p0, cores)
    lo, hi = p0 + pid * per, min(p1, p0 + (pid + 1) * per)
    for j, r0 in enumerate(range(lo, hi, step)):
        r1 = min(r0 + step, hi)
        ins = nisa.dma_copy(dst=dst[r0:r1, :], src=src[r0:r1, :], dge_mode=nisa.dge_mode.hwdge,
                            engine=COPY_ENGINES[j % len(COPY_ENGINES)], priority=COPY_PRIORITY)
        sched_after(ins, after)


def transpose_pe(w_hbm, n, k, r0=0):
    """`w_hbm[r0:r0+n, 0:k]` (bf16 HBM) transposed onto the partitions, `[PT, k / PT, n]`
    (`t[p, kb, j] = w[r0 + j, kb * PT + p]`): row-block loads and PE transposes (exact) instead
    of DMA transposes (in the prologue those waited for the embedding all_reduce on the
    Sync queue, and everything queued behind them with it)."""
    rows = nl.ndarray((PT, n // PT, k), dtype=nl.bfloat16, buffer=nl.sbuf)
    for nb in range(n // PT):
        nisa.dma_copy(dst=rows[0:PT, nb, 0:k], src=w_hbm[r0 + nb * PT : r0 + (nb + 1) * PT, 0:k])
    out = nl.ndarray((PT, k // PT, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kb in range(k // PT):
        for nb in range(n // PT):
            tps = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=rows[0:PT, nb, kb * PT : (kb + 1) * PT])
            nisa.tensor_copy(dst=out[0:PT, kb, nb * PT : (nb + 1) * PT], src=tps)
    return out


def gather_rows(cache_hbm, slots_hbm, dst_hbm, tiles, n, dtype=nl.bfloat16):
    """`dst[r] = cache[slots[r]]` for the rows of `tiles` (`n` columns)."""
    for r0, nr in tiles:
        ix = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ix[0:nr, :], src=slots_hbm[r0 : r0 + nr, 0:1])
        t = nl.ndarray((PT, n), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=t[0:nr, :],
            src=cache_hbm.ap(pattern=[[n, nr], [1, n]], vector_offset=ix[0:nr, 0:1], indirect_dim=0),
        )
        nisa.dma_copy(dst=dst_hbm[r0 : r0 + nr, 0:n], src=t[0:nr, :])


# --------------------------------------------------------------------------- sparse attention


def _load_col_f32(col_hbm, r0, nr):
    """An int32 `[T, 1]` host column as an fp32 `(PT, 1)` SBUF tile (rows r0..r0+nr)."""
    ci = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ci[0:nr, :], src=col_hbm[r0 : r0 + nr, 0:1])
    cf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cf[0:nr, :], src=ci[0:nr, :])
    return cf


def attn_index_table(idx_hbm, tiles, twin, wv, win, cbase, topk_hbm, topk, kpad):
    """Build the per-row slot index table `idx_hbm[T, kpad]` (int32, -1 = empty):

        idx[r, j]         = twin[r] + j         if wv[r] + j >= 0 else -1   (j < win)
        idx[r, win + j]   = cbase[r] + topk[r, j]  if topk[r, j] >= 0 else -1
        idx[r, win+topk:] = -1

    `twin`, `wv`, `cbase` are `[T, 1]` int32 host columns (table row of the first
    window slot, window validity base, table row of the first compressed slot).
    """
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
        tps = nl.ndarray((PT, dchunks, PT), dtype=nl.bfloat16, buffer=nl.psum)
        for c in range(dchunks):
            nisa.nc_transpose(dst=tps[0:PT, c, 0:PT], data=gn[0:PT, kb, c * PT : (c + 1) * PT])
        nisa.tensor_copy(dst=gt[0:PT, 0:dchunks, kb * PT : (kb + 1) * PT], src=tps)

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
    nisa.tensor_scalar(dst=o16[0:heads, :], data=acc_o, op0=nl.multiply, operand0=rcp)
    nisa.dma_copy(
        dst=o_hbm.ap(pattern=[[d, heads], [1, d]], offset=tok0 * hw, scalar_offset=i, indirect_dim=0),
        src=o16[0:heads, :],
    )


#: SBUF bytes a partition may hold of the resident transposed key table
#: (`build_kT`): 4 chunks x 2 bytes x rows, so 96 KiB covers 12288 slot-table rows.
KT_SBUF_BYTES = 96 * 1024


def kt_resident(ntab, hd):
    """Whether a slot table of `ntab` rows fits the resident transposed layout."""
    return pad_to(ntab, PT) * hd * 2 <= KT_SBUF_BYTES


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


def attn_token_body_ti(i, tok0, q_hbm, kvtab_hbm, kTs, idx_hbm, o_hbm, sink_t, heads, d, kpad, scale):
    """`attn_token_body` with the keys selected on chip: the score matmuls read the
    token's `kpad` columns of the resident `kTs` through tensor indirection (the
    slot indices in the 16-partition snake the tensor engine reads, replicated into
    each quadrant), so no per-token transposes; the values are still gathered from
    `kvtab_hbm` by indirect DMA for the PV matmul."""
    dchunks = d // PT
    ktiles = kpad // PT
    ncol = kpad // 16
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
    # the snake: offset j at [j % 16, j // 16] of each quadrant
    ix32 = nl.ndarray((PT, ncol), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(ix32, 0)
    for qd in range(4):
        nisa.dma_copy(
            dst=ix32[32 * qd : 32 * qd + 16, 0:ncol],
            src=idx_hbm.ap(pattern=[[1, 16], [16, ncol]], offset=tok0 * kpad,
                           scalar_offset=i, indirect_dim=0),
        )
    nisa.tensor_scalar(dst=ix32, data=ix32, op0=nl.maximum, operand0=0)
    ix = nl.ndarray((PT, ncol), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ix, src=ix32)

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

    acc = nl.ndarray((heads, kpad), dtype=nl.float32, buffer=nl.psum)
    for c in range(dchunks):
        nisa.nc_matmul(
            dst=acc, stationary=qt[0:PT, c, 0:heads],
            moving=kTs[c].indirect(ix, num_elem=kpad), accumulate=(c > 0),
        )
    sc = nl.ndarray((heads, kpad), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=sc, data=acc, op0=nl.multiply, operand0=scale, op1=nl.add, operand1=bias_b,
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


def attn_units(B, K, heads, pid, cores=2):
    """The dense attention's work units of core `pid`: `(b, t0, nt)` = draft rows
    `b * K + t0 .. + nt` of sequence `b`, at most `PT // heads` tokens a unit (one row
    tile of (token, head) pairs). Sequences are split over the cores; a lone unit is
    split by tokens."""
    tmax = max(1, PT // heads)
    units = [(b, t0, min(tmax, K - t0)) for b in range(B) for t0 in range(0, K, tmax)]
    if len(units) == 1 and units[0][2] > 1:
        b, t0, nt = units[0]
        h = cdiv(nt, 2)
        units = [(b, t0, h), (b, t0 + h, nt - h)]
    n0 = cdiv(len(units), cores)
    return units[0:n0] if pid == 0 else units[n0:]


def attn_sink_rows(attn_sink_s, heads, ntok):
    """`(ntok * heads, 1)` fp32 SBUF: row t * heads + h = the sink of head h."""
    sk = nl.ndarray((ntok * heads, 1), dtype=nl.float32, buffer=nl.sbuf)
    for t in range(ntok):
        nisa.dma_copy(dst=sk[t * heads : (t + 1) * heads, 0:1], src=attn_sink_s[0:heads, 0:1])
    return sk


def attn_rows_dense(q_hbm, kvtab_hbm, idx_hbm, o_hbm, sinkr, heads, d, kpad, scale, row0, nt, tb):
    """Attention of the `nt` draft rows `row0 ..` of one sequence whose slot table is
    `kvtab_hbm[tb : tb + kpad]`: the rows' index rows are all `idx[row0]` and map column
    j to table row `tb + j` (or -1, empty), so the keys / values are the table rows
    themselves -- the same per-element arithmetic as `attn_token_body_ti` (score
    column j = q . k[tb + j] accumulated over the head-dim chunks, the same bias, an
    empty column's probability exactly 0), with the (token, head) pairs of the rows as
    one tile: `q_hbm` / `o_hbm` viewed as `[rows * heads, d]`."""
    dchunks = d // PT
    ktiles = kpad // PT
    n = nt * heads
    assert n <= PT

    idrow = nl.ndarray((1, kpad), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=idrow, src=idx_hbm[row0 : row0 + 1, 0:kpad])
    brow = nl.ndarray((1, kpad), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=brow, data=idrow, op0=nl.less, operand0=0.0,
                       op1=nl.multiply, operand1=NEG_BOUND)
    bias_b = nl.broadcast_to(brow, (n, kpad))

    # the keys, transposed: kT[c][d, j] = kvtab[tb + j, c * 128 + d]; the values as is
    kTs = []
    for c in range(dchunks):
        kT = nl.ndarray((PT, kpad), dtype=nl.bfloat16, buffer=nl.sbuf)
        for kb in range(ktiles):
            nisa.dma_transpose(dst=kT[0:PT, kb * PT : (kb + 1) * PT],
                               src=kvtab_hbm[tb + kb * PT : tb + (kb + 1) * PT, c * PT : (c + 1) * PT])
        kTs.append(kT)
    vt = nl.ndarray((PT, ktiles, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kb in range(ktiles):
        nisa.dma_copy(dst=vt[0:PT, kb, 0:d], src=kvtab_hbm[tb + kb * PT : tb + (kb + 1) * PT, 0:d])

    qrow = nl.ndarray((n, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=qrow, src=q_hbm.ap(pattern=[[d, n], [1, d]], offset=row0 * heads * d))
    qps = nl.ndarray((PT, dchunks, n), dtype=nl.bfloat16, buffer=nl.psum)
    for c in range(dchunks):
        nisa.nc_transpose(dst=qps[0:PT, c, 0:n], data=qrow[0:n, c * PT : (c + 1) * PT])
    qt = nl.ndarray((PT, dchunks, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qt, src=qps)

    acc = nl.ndarray((n, kpad), dtype=nl.float32, buffer=nl.psum)
    for c in range(dchunks):
        nisa.nc_matmul(dst=acc, stationary=qt[0:PT, c, 0:n], moving=kTs[c][0:PT, 0:kpad],
                       accumulate=(c > 0))
    sc = nl.ndarray((n, kpad), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=sc, data=acc, op0=nl.multiply, operand0=scale, op1=nl.add, operand1=bias_b,
    )

    neg_max = nl.negative(nl.max(sc[0:n, 0:kpad], axis=1, keepdims=True))
    p32 = nl.ndarray((n, kpad), dtype=nl.float32, buffer=nl.sbuf)
    den = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=p32, op=nl.exp, data=sc[0:n, 0:kpad], bias=neg_max, scale=1.0,
                    reduce_op=nl.add, reduce_res=den)
    sk = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sk, op=nl.exp, data=sinkr[0:n, 0:1], bias=neg_max)
    rcp = nl.ndarray((n, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rcp, data=nl.add(den, sk))
    p16 = nl.copy(p32, dtype=nl.bfloat16)

    ptps = nl.ndarray((PT, ktiles, n), dtype=nl.bfloat16, buffer=nl.psum)
    for kb in range(ktiles):
        nisa.nc_transpose(dst=ptps[0:PT, kb, 0:n], data=p16[0:n, kb * PT : (kb + 1) * PT])
    pt = nl.ndarray((PT, ktiles, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pt, src=ptps)
    acc_o = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.psum)
    for kb in range(ktiles):
        nisa.nc_matmul(dst=acc_o, stationary=pt[0:PT, kb, 0:n], moving=vt[0:PT, kb, 0:d],
                       accumulate=(kb > 0))
    o16 = nl.ndarray((n, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o16, data=acc_o, op0=nl.multiply, operand0=rcp)
    nisa.dma_copy(dst=o_hbm.ap(pattern=[[d, n], [1, d]], offset=row0 * heads * d), src=o16)


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
            y = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=y[0:nr, :], data1=sg[0:nr, :], data2=u[0:nr, :], op=nl.multiply)
            if w_hbm is not None:
                nisa.tensor_scalar(dst=y[0:nr, :], data=y[0:nr, :], op0=nl.multiply, operand0=w[0:nr, 0:1])
            y16 = nl.ndarray((PT, chunk), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=y16[0:nr, :], src=y[0:nr, :])
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



def engram_gather_fp8(ids_hbm, table_hbm, row_start_hbm, emb_hbm, tiles, ncol, hd, row_width):
    """`engram_gather` over an fp8 table (`engram_hd`): each tile's `ncol` row gathers into one
    `[PT, ncol, hd + hd / 32]` uint8 tile, dequantized and masked together (`engram_dequant`)."""
    ec = table_hbm.shape[1]
    for r0, nr in tiles:
        ids = nl.ndarray((PT, ncol), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ids[0:nr, :], src=ids_hbm[r0 : r0 + nr, 0:ncol])
        idf = nl.ndarray((PT, ncol), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idf[0:nr, :], src=ids[0:nr, :])
        rs = bcast_row(row_start_hbm[0:1, 0:1], 1, nr)
        loc = nl.ndarray((PT, ncol), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=loc[0:nr, :], data=idf[0:nr, :], op0=nl.subtract, operand0=rs[0:nr, 0:1])
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
        g8 = nl.ndarray((PT, ncol, ec), dtype=nl.uint8, buffer=nl.sbuf)
        for j in range(ncol):
            nisa.dma_copy(dst=g8[0:nr, j, :], src=table_hbm.ap(pattern=[[ec, nr], [1, ec]],
                                                                vector_offset=cli[0:nr, j : j + 1], indirect_dim=0))
        e16 = engram_dequant(g8, nr, ncol, hd, mask)
        nisa.dma_copy(dst=emb_hbm[r0 : r0 + nr, 0 : ncol * hd], src=e16.reshape((PT, ncol * hd))[0:nr, :])


def engram_hd(etable):
    """The engram row width in values: a bf16 table `[rows, hd]`, or the fp8 one `[rows, hd + hd / 32]`
    uint8 (`weight_loader`: hd e4m3fn codes, then hd / 32 e8m0 scales)."""
    ec = etable.shape[1]
    if etable.dtype == nl.uint8:
        assert ec % 33 == 0, f"fp8 engram row of {ec} bytes"
        return ec // 33 * 32
    return ec


def engram_dequant(g8, nr, G, ehd, mask=None):
    """Gathered fp8 engram rows `g8` `[PT, G, ehd + ehd / 32]` uint8 -> `[PT, G, ehd]` bf16 (x `mask`
    `[PT, G]` fp32): code * 2^(e - 127), exact in bf16 (`weight_loader.dequant_engram_rows`)."""
    nb = ehd // 32
    # the scales' fp32 bit patterns e << 23 = 2^(e - 127): e * 2^23 is an exact integer in fp32
    sf = nl.ndarray((PT, G, nb), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sf[0:nr], src=g8[0:nr, :, ehd : ehd + nb])
    nisa.tensor_scalar(dst=sf[0:nr], data=sf[0:nr], op0=nl.multiply, operand0=8388608.0)
    si = nl.ndarray((PT, G, nb), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=si[0:nr], src=sf[0:nr])
    sc = si.view(nl.float32)
    if mask is not None:
        scm = nl.ndarray((PT, G, nb), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=scm[0:nr], data1=sc[0:nr],
                           data2=mask.ap(pattern=[[G, nr], [1, G], [0, nb]], offset=0), op=nl.multiply)
        sc = scm
    vals = nl.ndarray((PT, G, ehd), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vals[0:nr], src=g8.view(nl.float8_e4m3fn)[0:nr, :, 0:ehd])
    e16 = nl.ndarray((PT, G, ehd), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=e16.reshape((PT, G * nb, 32))[0:nr], data1=vals.reshape((PT, G * nb, 32))[0:nr],
                       data2=sc.ap(pattern=[[G * nb, nr], [1, G * nb], [0, 32]], offset=0), op=nl.multiply)
    return e16

def engram_readout(pid, mine, T, Tp, hc, dim, hash_ids, etable, erow_start, e_wkv4, e_wkv_s, e_qk,
                   strm, stream0, rg_table):
    """The engram read-out of a block, both cores: gather this rank's shard rows of the
    n-gram table, sum the shards over `rg_table` (one rank holds each row, so the
    bf16 sum is exact), the fp8 `wkv` projection (this core's output columns), and
    the gate into `stream0[T, hc*dim]` from `strm`. Allocates its own scratch."""
    ncol = hash_ids.shape[1]
    ehd = engram_hd(etable)
    e_in = ncol * ehd
    e_out = e_wkv4.shape[1]
    emb = nl.ndarray((Tp, e_in), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="emb")
    emb_sum = nl.ndarray((Tp, e_in), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="emb_sum")
    ekv = nl.ndarray((Tp, e_out), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="ekv")
    (engram_gather_fp8 if etable.dtype == nl.uint8 else engram_gather)(
        hash_ids, etable, erow_start, emb, mine, ncol, ehd, etable.shape[0])
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
    for r0, nr in mine:
        engram_gate_tile(r0, nr, strm, ekv, e_qk, stream0, hc, dim)
    nisa.core_barrier(data=stream0, cores=(0, 1))


def engram_gate_tile(r0, nr, h_hbm, kv_hbm, w_hbm, out_hbm, hc, dim, chunk=1024):
    """`kernels.engram_gate.gated_add` for one row tile: `h_hbm[T, hc*dim]` bf16 stream,
    `kv_hbm[T, (hc+1)*dim]` bf16 (key | value), `w_hbm[1, hc*dim]` fp32 -> `out_hbm`."""
    width = hc * dim
    nchunks = dim // chunk
    inv_dim = 1.0 / dim
    dim_rsqrt = float(dim) ** -0.5
    vf = []
    for j in range(nchunks):
        v = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=v[0:nr, :], src=kv_hbm[r0 : r0 + nr, width + j * chunk : width + (j + 1) * chunk])
        vf.append(v)
    for c in range(hc):
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


def moe_partial(pid, nr, dim, eacc, eo32, part1, dst):
    """The rank's MoE partial of rows `0 .. nr`: each core adds its routed accumulator and
    shared-expert output (`eacc + eo32`), core 1 hands its sum through `part1` to core 0,
    which adds the two (core 0's + core 1's) into `dst`; returns `dst`'s core barrier.
    The rows' contiguous `[nr, dim]` block is viewed as `[nr * nblk, w]` (each row's column
    blocks on their own partitions): the same elementwise fp32 adds on up to 128 partitions
    instead of `nr` (5 rows x 5120 took ~4.5 us an add on 5 partitions)."""
    nblk, w = _spread_blocks(nr, dim)
    P = nr * nblk

    def v(t):
        return t.ap(pattern=[[w, P], [1, w]], offset=0)

    hw = dict(dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    pa = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pa[0:P, :], src=v(eacc), **hw)
    pe = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pe[0:P, :], src=v(eo32), **hw)
    nisa.tensor_tensor(dst=pa[0:P, :], data1=pa[0:P, :], data2=pe[0:P, :], op=nl.add)
    if pid == 1:
        nisa.dma_copy(dst=v(part1), src=pa[0:P, :])
    nisa.core_barrier(data=part1, cores=(0, 1))
    if pid == 0:
        pb = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=pb[0:P, :], src=v(part1), **hw)
        nisa.tensor_tensor(dst=pa[0:P, :], data1=pa[0:P, :], data2=pb[0:P, :], op=nl.add)
        nisa.dma_copy(dst=v(dst), src=pa[0:P, :])
    return nisa.core_barrier(data=dst, cores=(0, 1))
