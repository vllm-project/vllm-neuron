# SPDX-License-Identifier: Apache-2.0
"""Building blocks shared by the layer and drafter megakernels.

Plain Python helpers traced inside a `@nki.jit` body: none of them is a kernel
on its own. Every helper is written for the two physical cores of a logical
NeuronCore (LNC=2): a helper either takes the row/column range the calling core
owns, or says in its docstring that it runs on both cores unchanged.

Numerics mirror the shipped per-op kernels (`blockquant`, `hc_in`, `hc_attn_out`,
`sparse_attn`), which were validated token for token against the checkpoint's
reference: the values written to the KV caches round-trip through `_qdq_fp8`
(FP8_PEAK = 224), the quantized GEMMs stage their inputs through the same round
trip and multiply bf16 weights on the tensor engine (`mk_gemm`), fp32 PSUM
accumulation, bf16 roundings at the same points as the torch path.

Ported from the Trainium3 megakernels; the one helper that needed the trn3
tensor engine -- `attn_token_body_ti`, which selects keys through tensor
indirection (`.indirect`, NeuronCore-v4 only) -- is not here, so every attention
runs `attn_token_body`, the per-token gather + PE-transpose path.
"""

import nki.isa as nisa
import nki.language as nl

from ...blockquant import (  # noqa: F401 - re-exported for the megakernels
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


def sinkhorn_multi(t2s, nrs, hc):
    """`sinkhorn` of several independent tiles, their instructions interleaved step
    by step (each tile's chain is latency-bound; the same ops, in the same order a tile)."""
    n = len(t2s)
    A = [nl.ndarray((PT, hc * hc), dtype=nl.float32, buffer=nl.sbuf) for _ in range(n)]
    B = [nl.ndarray((PT, hc * hc), dtype=nl.float32, buffer=nl.sbuf) for _ in range(n)]

    def rowsum_div(srcs, dsts):
        for i in range(hc):
            lo, hi = i * hc, (i + 1) * hc
            for src, dst, nr in zip(srcs, dsts, nrs):
                s = nl.sum(src[0:nr, lo:hi], axis=1, keepdims=True)
                nisa.tensor_scalar(
                    dst=dst[0:nr, lo:hi], data=src[0:nr, lo:hi],
                    op0=nl.multiply, operand0=nl.reciprocal(nl.add(s, HC_EPS)),
                )

    def colsum_div(srcs, dsts):
        ss = [nl.add(src[0:nr, 0:hc], src[0:nr, hc : 2 * hc]) for src, nr in zip(srcs, nrs)]
        for i in range(2, hc):
            ss = [nl.add(s, src[0:nr, i * hc : (i + 1) * hc]) for s, src, nr in zip(ss, srcs, nrs)]
        ds = [nl.reciprocal(nl.add(s, HC_EPS)) for s in ss]
        for i in range(hc):
            lo, hi = i * hc, (i + 1) * hc
            for src, dst, d, nr in zip(srcs, dsts, ds, nrs):
                nisa.tensor_tensor(
                    dst=dst[0:nr, lo:hi], data1=src[0:nr, lo:hi], data2=d, op=nl.multiply
                )

    for i in range(hc):
        lo, hi = i * hc, (i + 1) * hc
        for t2, a, nr in zip(t2s, A, nrs):
            mx = nl.max(t2[0:nr, lo:hi], axis=1, keepdims=True)
            e = nl.exp(nl.subtract(t2[0:nr, lo:hi], nl.broadcast_to(mx, (nr, hc))))
            s = nl.sum(e, axis=1, keepdims=True)
            nisa.tensor_scalar(
                dst=a[0:nr, lo:hi], data=e,
                op0=nl.multiply, operand0=nl.reciprocal(s), op1=nl.add, operand1=HC_EPS,
            )
    colsum_div(A, B)
    for _ in range(SINKHORN_ITERS - 1):
        rowsum_div(B, A)
        colsum_div(A, B)
    return B


#: hc_in's x.T psum copies all on the scalar engine (the vector engine has the
#: hc_out of the next tile to do meanwhile).
HC_XT_ACT = True


def hc_in_tile(*args, **kw):
    """`_hc_in_gen` run through."""
    g = _hc_in_gen(*args, **kw)
    next(g)
    return _gen_finish(g)


def _gen_finish(g):
    try:
        next(g)
    except StopIteration as e:
        return e.value
    raise AssertionError("generator did not finish")


def hc_in_consts(base, scale, norm_w, hc, dim, out_chunk=1024):
    """The row constants of `hc_in_tile` broadcast to all partitions, once for every tile."""
    return dict(
        s=[bcast_row(scale[0:1, k : k + 1], 1, PT) for k in range(3)],
        b=[bcast_row(base[0:1, 0:hc], hc, PT), bcast_row(base[0:1, hc : 2 * hc], hc, PT),
           bcast_row(base[0:1, 2 * hc : 2 * hc + hc * hc], hc * hc, PT)],
        w=[bcast_row(norm_w[0:1, c * out_chunk : (c + 1) * out_chunk], out_chunk, PT)
           for c in range(dim // out_chunk)],
    )


def _hc_in_gen(r0, nr, x2, pre_mix, fnT, base, scale, norm_w, hc, dim, mix,
               normed, pre, post, comb, sq_chunk=1024, out_chunk=1024, defer_comb=False, xs=None,
               consts=None):
    """One 128-row tile of the hyper-connection sublayer-in (`kernels.hc_in`).

    Reads `x2[r0:r0+nr]` (`[T, hc*dim]` bf16 HBM) and `pre_mix`, writes
    `normed[r0:r0+nr]` (bf16) and the fp32 coefficients `pre`/`post`/`comb`.
    `fnT` is the resident `(PT, nk, mix)` fp32 SBUF copy of `fn.T`.
    With `defer_comb` the sinkhorn is left to the caller (`hc_in_tiles`): returns
    its input `t2`. `xs` (the `hc` `(PT, dim)` bf16 SBUF tiles of `x2[r0:r0+nr]`, as
    `hc_out_tile(keep=True)` leaves them) skips the load.

    A generator: the first `next` issues everything up to the mix GEMM, the second
    the rest (so a caller can put another tile's work between the two).
    """
    width = hc * dim
    kpg = dim // PT
    inv_width = 1.0 / width
    inv_dim = 1.0 / dim

    if xs is not None:
        xg = xs
    else:
        xg = []
        for g in range(hc):
            xg.append(nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf))
        for g in range(hc):
            nisa.dma_copy(dst=xg[g][0:nr, :], src=x2[r0 : r0 + nr, g * dim : (g + 1) * dim])
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

    # x.T in groups of TGRP 128-column blocks: the group's transposes fill one psum
    # bank and leave it in one copy (alternating engines).
    mixp = nl.ndarray((PT, mix), dtype=nl.float32, buffer=nl.psum)
    nkb = hc * kpg
    TGRP = 8
    for gi, kb0 in enumerate(range(0, nkb, TGRP)):
        nkg = min(TGRP, nkb - kb0)
        tps = nl.ndarray((PT, TGRP, PT), dtype=nl.bfloat16, buffer=nl.psum)
        for j in range(nkg):
            g, c = divmod(kb0 + j, kpg)
            nisa.nc_transpose(dst=tps[0:PT, j, 0:nr], data=xg[g][0:nr, c * PT : (c + 1) * PT])
        xT = nl.ndarray((PT, TGRP, PT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xT[0:PT, 0:nkg, 0:nr], src=tps[0:PT, 0:nkg, 0:nr],
                         engine=nisa.engine.vector if gi % 2 == 0 and not HC_XT_ACT
                         else nisa.engine.scalar)
        for j in range(nkg):
            kb = kb0 + j
            nisa.nc_matmul(
                dst=mixp[0:nr, 0:mix], stationary=xT[0:PT, j, 0:nr],
                moving=fnT[0:PT, kb, 0:mix], accumulate=(kb > 0),
            )

    yield
    rs = rsqrt_newton(nl.add(nl.multiply(ss[0:nr, 0:1], inv_width), NORM_EPS))
    mixv = nl.ndarray((PT, mix), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=mixv[0:nr, :], data=mixp[0:nr, :], op0=nl.multiply, operand0=rs)

    if consts is None:
        consts = hc_in_consts(base, scale, norm_w, hc, dim, out_chunk)
    s0, s1, s2 = [t[0:nr, :] for t in consts["s"]]
    b0, b1, b2 = [t[0:nr, :] for t in consts["b"]]

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
    if not defer_comb:
        comb_t = sinkhorn(t2, nr, hc)
        nisa.dma_copy(dst=comb[r0 : r0 + nr, 0 : hc * hc], src=comb_t[0:nr, :])

    ss2 = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    sum_squares(ss2, y16, nr, dim, sq_chunk)
    rstd = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    rstd[0:nr, :] = rsqrt_newton(nl.add(nl.multiply(ss2[0:nr, 0:1], inv_dim), NORM_EPS))
    for c in range(dim // out_chunk):
        lo, hi = c * out_chunk, (c + 1) * out_chunk
        wj = consts["w"][c][0:nr, :]
        yv = nl.ndarray((PT, out_chunk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=yv[0:nr, :], op=nl.copy, data=y16[0:nr, lo:hi], scale=rstd[0:nr, 0:1])
        nisa.tensor_tensor(dst=yv[0:nr, :], data1=yv[0:nr, :], data2=wj, op=nl.multiply)
        o16 = nl.ndarray((PT, out_chunk), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o16[0:nr, :], src=yv[0:nr, :])
        nisa.dma_copy(dst=normed[r0 : r0 + nr, lo:hi], src=o16[0:nr, :])
    return t2


def hc_in_tiles(tiles, x2, pre_mix, fnT, base, scale, norm_w, hc, dim, mix, normed, pre, post,
                comb, out_args=None):
    """`hc_in_tile` over `tiles`, the tiles' sinkhorns run together at the end.

    With `out_args = (x_hbm, res_hbm, post_hbm, comb_hbm)` each tile first runs
    `hc_out_tile` into `x2` and the hc_in reads the rows it left in SBUF (no reload)."""
    # software-pipelined: tile k's first half (and its hc_out) before tile k-1's second
    t2s = []
    prev = None
    consts = hc_in_consts(base, scale, norm_w, hc, dim) if tiles else None
    for r0, nr in tiles:
        xs = None
        if out_args is not None:
            xs = hc_out_tile(r0, nr, *out_args, x2, hc, dim, keep=True)
        g = _hc_in_gen(r0, nr, x2, pre_mix, fnT, base, scale, norm_w, hc, dim, mix, normed, pre,
                       post, comb, defer_comb=True, xs=xs, consts=consts)
        next(g)
        if prev is not None:
            t2s.append(_gen_finish(prev))
        prev = g
    if prev is not None:
        t2s.append(_gen_finish(prev))
    if not tiles:
        return
    nrs = [nr for _, nr in tiles]
    cts = sinkhorn_multi(t2s, nrs, hc)
    for (r0, nr), ct in zip(tiles, cts):
        nisa.dma_copy(dst=comb[r0 : r0 + nr, 0 : hc * hc], src=ct[0:nr, :])


def load_fn_t(fn_t, width, mix):
    """`fn.T` (`[hc*dim, mix]` fp32 HBM) resident as `(PT, nk, mix)` fp32 SBUF."""
    nk = width // PT
    fnT = nl.ndarray((PT, nk, mix), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=fnT, src=fn_t.ap(pattern=[[mix, PT], [PT * mix, nk], [1, mix]]))
    return fnT


def cc_chunks(Tp, n):
    """`(r0, r1)` row chunks of `[0, Tp)` for a chunked all_reduce: up to `n`, each a
    whole number of 2-tile groups (so both cores own rows of every chunk), the last one
    taking the remainder."""
    nt = cdiv(Tp, PT)
    npair = nt // 2
    n = max(1, min(n, npair))
    out = []
    for k in range(n):
        r0 = (k * npair // n) * 2 * PT
        r1 = Tp if k == n - 1 else ((k + 1) * npair // n) * 2 * PT
        out.append((r0, r1))
    return out


def pick_chunk(x_hbm, r0):
    """`(tensor, row0)` holding row `r0`: `x_hbm` itself (row0 0), or the entry of a
    `[(r0, r1, tensor)]` chunk list (chunk-local rows)."""
    if not isinstance(x_hbm, (list, tuple)):
        return x_hbm, 0
    for c0, c1, t in x_hbm:
        if c0 <= r0 < c1:
            return t, c0
    raise AssertionError("row %d in no chunk" % r0)


def hc_out_res(r0, nr, res_hbm, comb_hbm, hc, dim, chunk=1024, nchunks=None):
    """The residual half of `hc_out_tile`, which needs no sublayer output:
    `acc[t, j, :] = sum_i comb[t, i, j] * res[t, i, :]` of the first `nchunks` column
    chunks, one fp32 `(PT, hc, chunk)` SBUF tile each, for `hc_out_tile(acc_pre=)` (the
    same ops, so they can run during the all_reduce of the sublayer output)."""
    comb = nl.ndarray((PT, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=comb[0:nr, :], src=comb_hbm[r0 : r0 + nr, 0 : hc * hc])
    accs = []
    for c in range(dim // chunk if nchunks is None else nchunks):
        col0 = c * chunk
        acc = nl.ndarray((PT, hc, chunk), dtype=nl.float32, buffer=nl.sbuf)
        accs.append(acc)
        rg = nl.ndarray((PT, hc, chunk), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i in range(hc):
            nisa.dma_copy(
                dst=rg[0:nr, i, :],
                src=res_hbm[r0 : r0 + nr, i * dim + col0 : i * dim + col0 + chunk],
            )
        for j in range(hc):
            a = acc[0:nr, j, :]
            nisa.tensor_scalar(dst=a, data=rg[0:nr, 0, :], op0=nl.multiply,
                               operand0=comb[0:nr, j : j + 1], engine=nisa.engine.scalar)
            for i in range(1, hc):
                col = i * hc + j
                nisa.scalar_tensor_tensor(dst=a, data=rg[0:nr, i, :], op0=nl.multiply,
                                          operand0=comb[0:nr, col : col + 1], op1=nl.add,
                                          operand1=a)
    return accs


def hc_out_tile(r0, nr, x_hbm, res_hbm, post_hbm, comb_hbm, y_hbm, hc, dim, chunk=1024,
                x_dtype=nl.float32, keep=False, acc_pre=None):
    """One 128-row tile of `hc_post` (`kernels.hc_attn_out`):
    `y[t, j, :] = post[t, j] * x[t, :] + sum_i comb[t, i, j] * res[t, i, :]`.

    `x_hbm` is the sublayer output `[T, dim]` (fp32 or bf16 -- it is widened to fp32
    for the products either way, matching `out.to(x.dtype)` followed by the fp32
    kernel math when `x_dtype` is bf16 and the all-reduced fp32 partial when fp32),
    `res_hbm` the `[T, hc*dim]` bf16 residual stream, `y_hbm` the `[T, hc*dim]` bf16 output.
    `keep` also returns the output rows as `hc` `(PT, dim)` bf16 SBUF tiles.
    `x_hbm` may be a chunk list (`pick_chunk`): one chunked all_reduce's destinations.
    `acc_pre`: the tile's `hc_out_res` (its first column chunks), already computed.
    """
    x_hbm, xr0 = pick_chunk(x_hbm, r0)
    ys = [nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(hc)] if keep else None
    post = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=post[0:nr, :], src=post_hbm[r0 : r0 + nr, 0:hc])
    acc_pre = acc_pre or []
    if len(acc_pre) < dim // chunk:
        comb = nl.ndarray((PT, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=comb[0:nr, :], src=comb_hbm[r0 : r0 + nr, 0 : hc * hc])
    for c in range(dim // chunk):
        col0 = c * chunk
        xg = nl.ndarray((PT, chunk), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=xg[0:nr, :], src=x_hbm[r0 - xr0 : r0 - xr0 + nr, col0 : col0 + chunk])
        og = nl.ndarray((PT, hc, chunk), dtype=nl.bfloat16, buffer=nl.sbuf)
        if c < len(acc_pre):
            for j in range(hc):
                nisa.scalar_tensor_tensor(
                    dst=ys[j][0:nr, col0 : col0 + chunk] if keep else og[0:nr, j, :],
                    data=xg[0:nr, :], op0=nl.multiply, operand0=post[0:nr, j : j + 1],
                    op1=nl.add, operand1=acc_pre[c][0:nr, j, :],
                )
            for j in range(hc):
                nisa.dma_copy(
                    dst=y_hbm[r0 : r0 + nr, j * dim + col0 : j * dim + col0 + chunk],
                    src=ys[j][0:nr, col0 : col0 + chunk] if keep else og[0:nr, j, :],
                )
            continue
        rg = nl.ndarray((PT, hc, chunk), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i in range(hc):
            nisa.dma_copy(
                dst=rg[0:nr, i, :],
                src=res_hbm[r0 : r0 + nr, i * dim + col0 : i * dim + col0 + chunk],
            )
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
                dst=ys[j][0:nr, col0 : col0 + chunk] if keep else og[0:nr, j, :], data=xg[0:nr, :],
                op0=nl.multiply, operand0=post[0:nr, j : j + 1], op1=nl.add,
                operand1=acc[0:nr, j, :],
            )
        for j in range(hc):
            nisa.dma_copy(
                dst=y_hbm[r0 : r0 + nr, j * dim + col0 : j * dim + col0 + chunk],
                src=ys[j][0:nr, col0 : col0 + chunk] if keep else og[0:nr, j, :],
            )
    return ys


# --------------------------------------------------------------------------- norms, rope, quant


def norm_rows(src_hbm, w_hbm, dst_hbm, tiles, n, qdq, chunk=None):
    """RMSNorm of `src_hbm[r0:r0+nr, 0:n]` (bf16) into `dst_hbm` (bf16) for the given
    row tiles, optionally followed by the fp8 activation round trip (`qdq`).
    """
    chunk = chunk or n
    inv_n = 1.0 / n
    wjs = [bcast_row(w_hbm[0:1, c * chunk : (c + 1) * chunk], chunk, PT)
           for c in range(n // chunk)] if tiles else []
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
            nisa.tensor_tensor(dst=yv[0:nr, :], data1=yv[0:nr, :], data2=wj, op=nl.multiply)
            nisa.tensor_copy(dst=y16[0:nr, lo:hi], src=yv[0:nr, :])
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
              inverse, qdq=False):
    """Rope the last `ROPE_DIM` columns of each of `nheads` heads of width `head`
    in `src_hbm[rows, width]` (bf16), optionally fp8-round-trip the rows, and write
    `dst_hbm`. `src_hbm` and `dst_hbm` may be the same tensor.

    `qdq` is False, True (`_qdq_fp8`, the GEMM-input round trip) or a function
    `(tile, rows, width) -> tile` (the reference-exact `mk_compress.fp8_qdq_exact_tile`
    the KV-cache writes use).
    """
    for r0, nr in tiles:
        t = nl.ndarray((PT, width), dtype=nl.bfloat16, buffer=nl.sbuf)
        if qdq and nr < PT:
            nisa.memset(t, 0.0)
        nisa.dma_copy(dst=t[0:nr, :], src=src_hbm[r0 : r0 + nr, 0:width])
        cos, sin = load_freqs(freqs_hbm, row_pos_hbm, r0, nr)
        for h in range(nheads):
            rope_inplace(t, nr, h * head + head - ROPE_DIM, cos, sin, inverse)
        if qdq is True:
            t = _qdq_fp8(t, PT, width)
        elif qdq:
            t = qdq(t, nr, width)
        nisa.dma_copy(dst=dst_hbm[r0 : r0 + nr, 0:width], src=t[0:nr, :])


# --------------------------------------------------------------------------- GEMMs (fp32 router)


def gemm_fp32(x_hbm, w_t_hbm, out_hbm, tiles, k, n):
    """`out[rows, n] = x[rows, k] (bf16) @ w_t[k, n] (fp32)` with **fp32 operands** on the
    tensor engine (the router's `matmul(x.float(), w.float().t())`). `w_t` is staged
    resident as `(PT, k//PT, n)`; `n <= 512`.
    """
    nk = k // PT
    wT = nl.ndarray((PT, nk, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=wT, src=w_t_hbm.ap(pattern=[[n, PT], [PT * n, nk], [1, n]]))
    for r0, nr in tiles:
        x = nl.ndarray((PT, k), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=x[0:nr, :], src=x_hbm[r0 : r0 + nr, 0:k])
        acc = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.psum)
        for kb in range(nk):
            tps = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps[0:PT, 0:nr], data=x[0:nr, kb * PT : (kb + 1) * PT])
            xT = nl.ndarray((PT, PT), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=xT[0:PT, 0:nr], src=tps[0:PT, 0:nr])
            nisa.nc_matmul(dst=acc[0:nr, 0:n], stationary=xT[0:PT, 0:nr],
                           moving=wT[0:PT, kb, 0:n], accumulate=(kb > 0))
        o = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o[0:nr, :], src=acc[0:nr, :])
        nisa.dma_copy(dst=out_hbm[r0 : r0 + nr, 0:n], src=o[0:nr, :])


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


#: Gather a token's non-window tiles with ONE batched indirect DMA (2D vector offset)
#: instead of one per tile (`_gbatch`).
ATTN_GATHER_BATCH = True


#: The gathered (non-window) K tiles of a token arrive transposed (`kt`, a gather-
#: transpose DMA) instead of being transposed on the PE (`attn_token_a`). Off: bit-exact
#: but slower (1k 9.25 -> 10.45 ms; the software-DGE transpose is expensive on gpsimd).
ATTN_KT_DMA = False


def attn_gtab_width(ktiles, nwt):
    """`gtab`'s ids a partition: the `ktiles` (gathered columns permuted for the batched
    gather), then, with `ATTN_KT_DMA`, the gathered tiles' ids again in slot order (the
    gather-transpose reads them that way)."""
    return ktiles + (ktiles - nwt if ATTN_KT_DMA and nwt and _gbatch(ktiles, nwt) else 0)


def _gbatch(ktiles, nwt):
    """Whether the gathered tiles `nwt..ktiles` of a token go as one batched DMA: its
    `(PT, M)` offsets are read column-major, so `gtab` holds them permuted
    (`attn_index_table`), which needs `M | PT`."""
    m = ktiles - nwt
    return ATTN_GATHER_BATCH and m > 1 and PT % m == 0


def attn_index_table(idx_hbm, tiles, twin, wv, win, cbase, topk_hbm, topk, kpad, gtab_hbm=None,
                     zrow=0, shift=0, slab_max=None, btab_hbm=None, nwt=None):
    """Build the per-row slot index table `idx_hbm[T, kpad]` (int32, -1 = empty):

        idx[r, j]         = twin[r] + j         if wv[r] + j >= 0 else -1   (j < win)
        idx[r, win + j]   = cbase[r] + topk[r, j]  if topk[r, j] >= 0 else -1
        idx[r, win+topk:] = -1

    `twin`, `wv`, `cbase` are `[T, 1]` int32 host columns (table row of the first
    window slot, window validity base, table row of the first compressed slot).

    `gtab_hbm` (`[T, PT, kpad/PT]` int32), when given, gets the gather rows of each row
    partition-major: `gtab[r, p, kb] = idx[r, kb*PT + p]`, an empty slot pointing at the
    zero row `zrow + p` instead (128 distinct zero rows, so the empty slots of a gather
    do not all hit one row; their probability is exactly 0 either way), so a token's
    gather offsets are one contiguous 20-byte run a partition. Its rows are those of a
    table `shift` rows down (`kvtab` behind a zero front pad).

    `slab_max` (slab mode, `win % PT == 0`): the window tiles are not gathered but
    loaded as one contiguous slab of `win` table rows (`attn_token_slab`), so
    `gtab[r, 0, 0]` holds the slab's first row instead, `twin[r] + shift` clamped to
    `[0, slab_max]`. The valid window slots are a suffix (`wv + j >= 0`), so whenever
    one is valid its row `twin + j` is in the table and the slab is not clamped
    (the front pad covers `twin < 0`); the rows of the invalid slots are masked by
    the bias either way.

    `nwt` (given and `_gbatch(kpad/PT, nwt)`): the gathered columns `nwt..` of `gtab`
    are permuted for `attn_token_load`'s batched gather: with `M = kpad/PT - nwt`,
    `A = PT/M`, `gtab[r, M*a + m, nwt + c]` holds the id of tile `nwt + m`, row `A*c + a`
    (the DMA reads the offsets column-major and fills its destination row-major).
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
        if btab_hbm is not None:
            # the empty-slot bias `attn_token_a` would compute from the index row
            brow = nl.ndarray((PT, kpad), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=brow[0:nr, :], data=irow[0:nr, :], op0=nl.less, operand0=0.0,
                               op1=nl.multiply, operand1=NEG_BOUND)
            nisa.dma_copy(dst=btab_hbm[r0 : r0 + nr, 0:kpad], src=brow[0:nr, :])
        if gtab_hbm is not None:
            ktiles = kpad // PT
            # empty (-1) -> zrow + p: -1 + (zrow + p + 1)
            zi = nl.ndarray((PT, PT), dtype=nl.int32, buffer=nl.sbuf)
            nisa.iota(dst=zi[0:nr, :], pattern=[[1, PT]], offset=zrow + 1 - shift,
                      channel_multiplier=0)
            zf = nl.ndarray((PT, PT), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=zf[0:nr, :], src=zi[0:nr, :])
            g3 = nl.ndarray((PT, PT, ktiles), dtype=nl.int32, buffer=nl.sbuf)
            for kb in range(ktiles):
                adj = nl.ndarray((PT, PT), dtype=nl.float32, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(dst=adj[0:nr, :], data=row[0:nr, kb * PT : (kb + 1) * PT],
                                          op0=nl.less, operand0=0.0, op1=nl.multiply,
                                          operand1=zf[0:nr, :])
                if shift:
                    sh = nl.ndarray((PT, PT), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=sh[0:nr, :], data=row[0:nr, kb * PT : (kb + 1) * PT],
                                       op0=nl.add, operand0=float(shift))
                    nisa.tensor_tensor(dst=g3[0:nr, :, kb], data1=sh[0:nr, :],
                                       data2=adj[0:nr, :], op=nl.add)
                else:
                    nisa.tensor_tensor(dst=g3[0:nr, :, kb], data1=row[0:nr, kb * PT : (kb + 1) * PT],
                                       data2=adj[0:nr, :], op=nl.add)
            if slab_max is not None:
                st = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=st[0:nr, :], data=tw[0:nr, 0:1], op0=nl.add,
                                   operand0=float(shift), op1=nl.maximum, operand1=0.0)
                nisa.tensor_scalar(dst=g3[0:nr, 0, 0:1], data=st[0:nr, :], op0=nl.minimum,
                                   operand0=float(slab_max))
            if nwt is not None and _gbatch(ktiles, nwt):
                M = ktiles - nwt
                A = PT // M
                g5 = nl.ndarray((PT, PT, gtab_hbm.shape[2]), dtype=nl.int32, buffer=nl.sbuf)
                if nwt:
                    nisa.tensor_copy(dst=g5[0:nr, :, 0:nwt], src=g3[0:nr, :, 0:nwt])
                src4 = g3[0:nr, :, :].reshape((nr, M, A, ktiles))   # [c, a, kb]
                dst4 = g5[0:nr, :, :].reshape((nr, A, M, gtab_hbm.shape[2]))   # [a, m, col]
                for c in range(M):
                    nisa.tensor_copy(dst=dst4[0:nr, :, :, nwt + c], src=src4[0:nr, c, :, nwt:ktiles])
                gw = gtab_hbm.shape[2]
                if gw > ktiles:
                    # the slot-order copy for the K gather-transpose
                    nisa.tensor_copy(dst=g5[0:nr, :, ktiles:gw], src=g3[0:nr, :, nwt:ktiles])
                g3 = g5
            nisa.dma_copy(dst=gtab_hbm[r0 : r0 + nr, :, :], src=g3[0:nr, :, :])


#: Sets of stage B's small buffers consecutive tokens rotate through (`attn_rot_bufs`).
ATTN_ROT = 4
#: K-transpose PSUM copies of the tiles `kb % 2 == ATTN_KCOPY_ACT` run on ACT (the
#: rest on vector, the loop's busiest engine).
ATTN_KCOPY_ACT = 1
#: The softmax exp writes bf16 p directly (its sum still accumulates in fp32).
ATTN_EXP_BF16 = True
#: The queue of a token's output store (ACT, which produced the row; the sync queue
#: issues the loads). None: software DGE (gpsimd, which the batched gather left idle).
ATTN_OSTORE_ENGINE = nisa.engine.scalar
#: Tokens whose stage A runs ahead of a token's stage B (`attn_block_compute`).
ATTN_DEPTH = 3


def _ktile(k):
    if k <= PSUM_FP32:
        return k
    for cand in (PSUM_FP32, 256, PT):
        if k % cand == 0:
            return cand
    return PT


def _align16(n):
    return -(-n // 16) * 16


def _score_tiles(kpad):
    """`(c0, n)` column tiles of the score row: as wide as a psum bank (512 fp32), all
    multiples of `PT` (`kpad` is)."""
    out = []
    c0 = 0
    while c0 < kpad:
        n = min(PSUM_FP32, kpad - c0)
        out.append((c0, n))
        c0 += n
    return out


def _tok_ap(hbm, pattern, offset, i):
    """`hbm.ap` at `offset` (+ the loop register `i` rows when given)."""
    if i is None:
        return hbm.ap(pattern=pattern, offset=offset)
    return hbm.ap(pattern=pattern, offset=offset, scalar_offset=i, indirect_dim=0)


def attn_bufs(U, heads, d, kpad, bias=False, kt=False):
    """Per-token SBUF buffers of `attn_token_load(2)` for `U` tokens, allocated once
    (outside the token loop) so a block's loads can be issued during the previous
    block's compute."""
    ktiles = kpad // PT
    hp = _align16(heads)
    out = []
    for _ in range(U):
        out.append(dict(
            ids=nl.ndarray((PT, ktiles), dtype=nl.int32, buffer=nl.sbuf),
            gn=nl.ndarray((PT, ktiles, d), dtype=nl.bfloat16, buffer=nl.sbuf),
            idrow=nl.ndarray((heads, kpad), dtype=nl.float32 if bias else nl.int32,
                             buffer=nl.sbuf),
            qrow=nl.ndarray((hp, d), dtype=nl.bfloat16, buffer=nl.sbuf),
        ))
        if kt:
            out[-1]["kt"] = nl.ndarray((PT, d // PT, kpad), dtype=nl.bfloat16, buffer=nl.sbuf)
    return out


#: Tokens a slab load (`attn_token_slab`) trails its token's slot-id load by on the
#: in-order sync queue, so the register read of the slab's start does not stall it.
ATTN_SLAB_LAG = 4


def attn_token_slab(ids, gn, kvtab_hbm, nwt, d):
    """The window tiles of a token (slab mode): `nwt * PT` consecutive table rows
    from `ids[0, 0]` into `gn[:, 0:nwt]`, one hardware-DGE DMA (no gather
    descriptors on gpsimd)."""
    s = nisa.scalar_load(ids[0:1, 0:1])
    nisa.dma_copy(dst=gn[0:PT, 0:nwt, 0:d],
                  src=kvtab_hbm.ap(pattern=[[d, PT], [PT * d, nwt], [1, d]], offset=0,
                                   scalar_offset=s, indirect_dim=0),
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)


def attn_token_load(i, tok0, q_hbm, kvtab_hbm, idx_hbm, heads, d, kpad, gtab_hbm=None, buf=None,
                    nwt=0, slab_now=True, ids_view=None):
    """The loads of the sparse attention of token `tok0 + i` (`i` a loop register):
    slot ids, the empty-slot bias, q, the gathered KV rows. `attn_token_a` (K
    transposes, masked scaled scores) and `attn_token_b` (softmax, PV, output) finish
    it; together they are the algorithm of
    `kernels.sparse_attn.gathered_attn_idx`, with the empty-slot bias built here.

    `q_hbm`/`o_hbm` are `[Tp, heads*d]` bf16, `kvtab_hbm` the `[rows, d]` bf16 slot
    table, `idx_hbm` `[Tp, kpad]` int32 (-1 = empty), `sink_t` `(heads, 1)` fp32 SBUF.

    The small per-token transfers use hardware DGE (the sync queue) so gpsimd only
    generates the gathers' descriptors; PSUM tiles are packed (one bank for the q
    transposes, one per two K tiles, 512-wide score tiles, one for the p transposes)
    so several tokens can be in flight. Every element is computed exactly as before.

    `nwt` (slab mode, needs `gtab_hbm`): the first `nwt` tiles are the window's
    consecutive rows, loaded by `attn_token_slab` (now, or by the caller later when
    not `slab_now`) instead of gathered.
    """
    dchunks = d // PT
    ktiles = kpad // PT
    hp = _align16(heads)
    hw = heads * d
    HW = nisa.dge_mode.hwdge

    # This token's slot indices, twice: (PT, ktiles) for the gather, (heads, kpad) for the bias.
    ids = buf["ids"] if buf is not None else nl.ndarray((PT, ktiles), dtype=nl.int32, buffer=nl.sbuf)
    if ids_view is not None:
        # loaded with its block (`attn_block_ids`)
        ids = gidx = ids_view
    elif gtab_hbm is not None:
        # already clamped, partition-major
        gw = gtab_hbm.shape[2]
        ids = nl.ndarray((PT, gw), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ids, src=_tok_ap(gtab_hbm, [[gw, PT], [1, gw]], tok0 * PT * gw, i),
                      dge_mode=HW, engine=nisa.engine.sync)
        gidx = ids
    else:
        nisa.dma_copy(dst=ids, src=_tok_ap(idx_hbm, [[1, PT], [PT, ktiles]], tok0 * kpad, i),
                      dge_mode=HW, engine=nisa.engine.sync)
        gidx = nl.ndarray((PT, ktiles), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gidx, data=ids, op0=nl.maximum, operand0=0.0)
    if buf is not None:
        gn = buf["gn"]
    else:
        gn = nl.ndarray((PT, ktiles, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    batched = gtab_hbm is not None and _gbatch(ktiles, nwt)
    if batched:
        # one batched gather (`attn_index_table` permuted the offsets)
        M = ktiles - nwt
        nisa.dma_copy(
            dst=gn[0:PT, nwt:ktiles, 0:d],
            src=kvtab_hbm.ap(pattern=[[d, PT * M], [1, d]], offset=0,
                             vector_offset=gidx[0:PT, nwt:ktiles], indirect_dim=0),
        )
    if batched and buf is not None and "kt" in buf:
        # the gathered K tiles transposed, `kt[p, c, nwt*PT + n] = K[slot n][c*PT + p]`
        M = ktiles - nwt
        nisa.dma_transpose(
            dst=buf["kt"][0:PT, 0:dchunks, nwt * PT : kpad],
            src=kvtab_hbm.ap(pattern=[[d, M * PT], [PT, dchunks], [1, PT]], offset=0,
                             vector_offset=gidx[0:PT, ktiles : ktiles + M].view(nl.uint32),
                             indirect_dim=0),
            axes=(2, 1, 0),
        )
    for kb in range(nwt, ktiles) if not batched else ():
        nisa.dma_copy(
            dst=gn[0:PT, kb, 0:d],
            src=kvtab_hbm.ap(pattern=[[d, PT], [1, d]], offset=0,
                             vector_offset=gidx[0:PT, kb : kb + 1], indirect_dim=0),
        )
    if nwt and slab_now:
        attn_token_slab(ids, gn, kvtab_hbm, nwt, d)
    return (i, tok0, gn)


def attn_token_load2(ld, q_hbm, idx_hbm, heads, d, kpad, buf=None, btab_hbm=None):
    """The rest of `attn_token_load`'s token: its index row on every head's partition
    (for the empty-slot bias; with `btab_hbm` the fp32 bias row itself) and its q rows."""
    i, tok0, gn = ld
    hp = _align16(heads)
    hw = heads * d
    HW = nisa.dge_mode.hwdge
    rdt = nl.float32 if btab_hbm is not None else nl.int32
    if buf is not None:
        idrow, qrow = buf["idrow"], buf["qrow"]
    else:
        idrow = nl.ndarray((heads, kpad), dtype=rdt, buffer=nl.sbuf)
        qrow = nl.ndarray((hp, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    src_tab = btab_hbm if btab_hbm is not None else idx_hbm
    nisa.dma_copy(dst=idrow, src=_tok_ap(src_tab, [[0, heads], [1, kpad]], tok0 * kpad, i),
                  dge_mode=HW, engine=nisa.engine.sync)
    # q rows of this token: (heads, d) -> transposed (PT, dchunks, heads) on the PE.
    nisa.dma_copy(dst=qrow[0:heads, :], src=_tok_ap(q_hbm, [[d, heads], [1, d]], tok0 * hw, i),
                  dge_mode=HW, engine=nisa.engine.sync)
    return (i, tok0, idrow, qrow, gn)


def attn_token_a(ld, heads, d, kpad, scale):
    """Stage A of `attn_token_load`'s token: q and K transposes, masked scaled scores."""
    return attn_token_a2(attn_token_a1(ld, heads, d, kpad), heads, d, kpad, scale)


def attn_token_a1(ld, heads, d, kpad):
    """The first half of `attn_token_a`: the q and K transposes."""
    i, tok0, idrow, qrow, gn = ld[:5]
    kt, ntr = (ld[5], ld[6]) if len(ld) > 5 else (None, kpad // PT)
    dchunks = d // PT
    ktiles = kpad // PT
    hp = _align16(heads)
    # bias = (idx < 0) * NEG_BOUND, on every head's row.
    if idrow.dtype == nl.float32:
        bias_b = idrow   # already the bias (`attn_index_table`'s `btab_hbm`)
    else:
        bias_b = nl.ndarray((heads, kpad), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=bias_b, data=idrow, op0=nl.less, operand0=0.0,
                           op1=nl.multiply, operand1=NEG_BOUND)
    qt = nl.ndarray((PT, dchunks, hp), dtype=nl.bfloat16, buffer=nl.sbuf)
    qps = nl.ndarray((PT, dchunks, hp), dtype=nl.bfloat16, buffer=nl.psum)
    for c in range(dchunks):
        nisa.nc_transpose(dst=qps[0:PT, c, 0:heads], data=qrow[0:heads, c * PT : (c + 1) * PT])
    nisa.tensor_copy(dst=qt[0:PT, 0:dchunks, 0:heads], src=qps[0:PT, 0:dchunks, 0:heads],
                     engine=nisa.engine.scalar)

    # with `kt` (tiles `ntr..` already transposed by DMA) only the first `ntr` here
    gt = kt if kt is not None else nl.ndarray((PT, dchunks, kpad), dtype=nl.bfloat16,
                                              buffer=nl.sbuf)
    # K transposes two tiles a psum bank, copied out on vector.
    for kb0 in range(0, ntr, 2):
        nk = min(2, ntr - kb0)
        tps = nl.ndarray((PT, 2, dchunks, PT), dtype=nl.bfloat16, buffer=nl.psum)
        for j in range(nk):
            for c in range(dchunks):
                nisa.nc_transpose(dst=tps[0:PT, j, c, 0:PT],
                                  data=gn[0:PT, kb0 + j, c * PT : (c + 1) * PT])
        for j in range(nk):
            eng = nisa.engine.scalar if (kb0 + j) % 2 == ATTN_KCOPY_ACT else nisa.engine.vector
            nisa.tensor_copy(dst=gt[0:PT, 0:dchunks, (kb0 + j) * PT : (kb0 + j + 1) * PT],
                             src=tps[0:PT, j, 0:dchunks, 0:PT], engine=eng)
    return (i, tok0, gn, bias_b, qt, gt)


def attn_token_a2(st, heads, d, kpad, scale):
    """The second half of `attn_token_a`: the masked scaled scores."""
    i, tok0, gn, bias_b, qt, gt = st
    dchunks = d // PT
    sc = nl.ndarray((heads, kpad), dtype=nl.float32, buffer=nl.sbuf)
    for c0, n in _score_tiles(kpad):
        acc = nl.ndarray((heads, PSUM_FP32), dtype=nl.float32, buffer=nl.psum)
        for c in range(dchunks):
            nisa.nc_matmul(
                dst=acc[0:heads, 0:n], stationary=qt[0:PT, c, 0:heads],
                moving=gt[0:PT, c, c0 : c0 + n], accumulate=(c > 0),
            )
        nisa.scalar_tensor_tensor(
            dst=sc[0:heads, c0 : c0 + n], data=acc[0:heads, 0:n], op0=nl.multiply, operand0=scale,
            op1=nl.add, operand1=bias_b[0:heads, c0 : c0 + n],
        )

    return (i, tok0, gn, sc)


def attn_rot_bufs(n, heads, d):
    """`n` sets of stage B's small per-token buffers, used in rotation by consecutive
    tokens: allocated fresh, every token gets the same addresses, so a token's softmax
    would wait (WAR) for the previous token's output store."""
    hp = _align16(heads)
    out = []
    for _ in range(n):
        out.append(dict(
            neg_max=nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf),
            den=nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf),
            sk=nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf),
            dsum=nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf),
            rcp=nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf),
            o16=nl.ndarray((hp, d), dtype=nl.bfloat16, buffer=nl.sbuf),
        ))
    return out


def attn_token_b(st, o_hbm, sink_t, heads, d, kpad, rb=None):
    """Stage B of `attn_token_a`'s token: softmax with the sink, PV, the output row.
    `rb`: one set of `attn_rot_bufs` (default: fresh buffers)."""
    i, tok0, gn, sc = st
    ktiles = kpad // PT
    hp = _align16(heads)
    hw = heads * d
    HW = nisa.dge_mode.hwdge
    if rb is None:
        rb = attn_rot_bufs(1, heads, d)[0]
    neg_max = rb["neg_max"]
    nisa.tensor_reduce(dst=neg_max, op=nl.max, data=sc[0:heads, 0:kpad], axis=1, negate=True,
                       keepdims=True)
    den = rb["den"]
    if ATTN_EXP_BF16:
        p16 = nl.ndarray((heads, kpad), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(dst=p16, op=nl.exp, data=sc[0:heads, 0:kpad], bias=neg_max, scale=1.0,
                        reduce_op=nl.add, reduce_res=den)
    else:
        p32 = nl.ndarray((heads, kpad), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=p32, op=nl.exp, data=sc[0:heads, 0:kpad], bias=neg_max, scale=1.0,
                        reduce_op=nl.add, reduce_res=den)
    sk = rb["sk"]
    nisa.activation(dst=sk, op=nl.exp, data=sink_t, bias=neg_max)
    rcp = rb["rcp"]
    nisa.tensor_tensor(dst=rb["dsum"], data1=den, data2=sk, op=nl.add)
    nisa.reciprocal(dst=rcp, data=rb["dsum"])
    if not ATTN_EXP_BF16:
        p16 = nl.ndarray((heads, kpad), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=p16, src=p32, engine=nisa.engine.scalar)

    pt = nl.ndarray((PT, ktiles, hp), dtype=nl.bfloat16, buffer=nl.sbuf)
    ptps = nl.ndarray((PT, ktiles, hp), dtype=nl.bfloat16, buffer=nl.psum)
    for kb in range(ktiles):
        nisa.nc_transpose(dst=ptps[0:PT, kb, 0:heads], data=p16[0:heads, kb * PT : (kb + 1) * PT])
    nisa.tensor_copy(dst=pt[0:PT, 0:ktiles, 0:heads], src=ptps[0:PT, 0:ktiles, 0:heads],
                     engine=nisa.engine.scalar)
    acc_o = nl.ndarray((heads, d), dtype=nl.float32, buffer=nl.psum)
    for kb in range(ktiles):
        nisa.nc_matmul(dst=acc_o, stationary=pt[0:PT, kb, 0:heads], moving=gn[0:PT, kb, 0:d],
                       accumulate=(kb > 0))
    o16 = rb["o16"]
    nisa.activation(dst=o16[0:heads, :], op=nl.copy, data=acc_o, scale=rcp)
    if ATTN_OSTORE_ENGINE is None:
        nisa.dma_copy(dst=_tok_ap(o_hbm, [[d, heads], [1, d]], tok0 * hw, i), src=o16[0:heads, :])
    else:
        nisa.dma_copy(dst=_tok_ap(o_hbm, [[d, heads], [1, d]], tok0 * hw, i), src=o16[0:heads, :],
                      dge_mode=HW, engine=ATTN_OSTORE_ENGINE)


def attn_token_body(i, tok0, q_hbm, kvtab_hbm, idx_hbm, o_hbm, sink_t, heads, d, kpad, scale,
                    gtab_hbm=None, nwt=0, btab_hbm=None):
    """Sparse attention for token `tok0 + i` (`i` a loop register)."""
    ld = attn_token_load(i, tok0, q_hbm, kvtab_hbm, idx_hbm, heads, d, kpad, gtab_hbm=gtab_hbm,
                         nwt=nwt)
    ld = attn_token_load2(ld, q_hbm, idx_hbm, heads, d, kpad, btab_hbm=btab_hbm)
    attn_token_b(attn_token_a(ld, heads, d, kpad, scale), o_hbm, sink_t, heads, d, kpad)


def attn_block_body(i, tok0, U, q_hbm, kvtab_hbm, idx_hbm, o_hbm, sink_t, heads, d, kpad, scale):
    """Tokens `tok0 + i + u` (u < U). The engine queues run in program order, so every
    token's slot ids and gathers are issued first, then its index row and q (nothing
    on those queues waits on compute), then the compute is software-pipelined: stage A of token u+1 before
    stage B of token u."""
    lds = [attn_token_load(i, tok0 + u, q_hbm, kvtab_hbm, idx_hbm, heads, d, kpad)
           for u in range(U)]
    lds = [attn_token_load2(ld, q_hbm, idx_hbm, heads, d, kpad) for ld in lds]
    prev = attn_token_a(lds[0], heads, d, kpad, scale)
    for u in range(1, U + 1):
        nxt = attn_token_a(lds[u], heads, d, kpad, scale) if u < U else None
        attn_token_b(prev, o_hbm, sink_t, heads, d, kpad)
        prev = nxt


def attn_slab_flush(pend, kvtab_hbm, nwt, d, keep=0):
    """Issue the pending slab loads of `pend` (`(ids, gn)`) but the last `keep`."""
    while len(pend) > keep:
        ids, gn = pend.pop(0)
        attn_token_slab(ids, gn, kvtab_hbm, nwt, d)


def attn_ids_bufs(U, kpad, gw=None):
    """Two `(PT, U, kpad/PT)` slot-id tiles: the blocks of `attn_block_compute`
    alternate them (block parity), so a block's ids load while the previous block
    still reads its own."""
    gw = gw or kpad // PT
    return [nl.ndarray((PT, U, gw), dtype=nl.int32, buffer=nl.sbuf) for _ in range(2)]


def attn_block_ids(i, tok0, U, gtab_hbm, idsb, kpad):
    """The partition-major gather ids of tokens `tok0 (+ i) + u` (u < U) into `idsb`,
    one DMA."""
    gw = gtab_hbm.shape[2]
    nisa.dma_copy(dst=idsb[0:PT, 0:U, 0:gw],
                  src=_tok_ap(gtab_hbm, [[gw, PT], [PT * gw, U], [1, gw]], tok0 * PT * gw, i),
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)


def attn_block_load(i, tok0, U, q_hbm, kvtab_hbm, idx_hbm, gtab_hbm, heads, d, kpad, bufs, nwt=0,
                    btab_hbm=None, idsb=None):
    """The loads of tokens `tok0 (+ i) + u` (u < U) into `bufs` (`attn_bufs`): every
    token's gathers first, then the window slabs, then the index rows and q. With
    `idsb` (an `attn_ids_bufs` tile) the block's ids load as one DMA."""
    ktiles = kpad // PT
    if idsb is not None:
        attn_block_ids(i, tok0, U, gtab_hbm, idsb, kpad)
    views = [idsb[0:PT, u, :] if idsb is not None else None for u in range(U)]
    lds = [attn_token_load(i, tok0 + u, q_hbm, kvtab_hbm, idx_hbm, heads, d, kpad,
                           gtab_hbm=gtab_hbm, buf=bufs[u], nwt=nwt, slab_now=False,
                           ids_view=views[u])
           for u in range(U)]
    if nwt:
        attn_slab_flush([(views[u] if idsb is not None else bufs[u]["ids"], bufs[u]["gn"])
                         for u in range(U)], kvtab_hbm, nwt, d)
    return [attn_token_load2(ld, q_hbm, idx_hbm, heads, d, kpad, buf=bufs[u], btab_hbm=btab_hbm)
            for u, ld in enumerate(lds)]


def attn_block_compute(i, tok0, U, q_hbm, kvtab_hbm, idx_hbm, gtab_hbm, o_hbm, sink_t, heads, d,
                       kpad, scale, bufs, nxt_tok0=None, nwt=0, pend=None, btab_hbm=None,
                       idsb=None, carry=None, close=True):
    """Tokens `tok0 (+ i) + u` whose loads are already in `bufs`, software-pipelined
    (stage A of token u+1 before stage B of token u). With `nxt_tok0`, token u's
    buffers are refilled with token `nxt_tok0 (+ i) + u` right after its stage B (the
    next block's loads overlap this block's compute).

    Slab mode (`nwt`): a refilled token's window slab is issued `ATTN_SLAB_LAG`
    refills later; `pend` (in/out) carries the ones still pending into the next
    block (the caller flushes it at the end of a loop trip).

    The pipeline (stage A of token u + ATTN_DEPTH before stage B of token u) runs
    across blocks: `carry` (the previous block's, when not closed) holds its last
    tokens' pending stage B (+ refill) calls, run after this block's first stage As.
    Returns `(pend, carry)`; with `close` every stage B runs here (`carry` None)."""
    pend = [] if pend is None else pend
    ktiles = kpad // PT
    if nxt_tok0 is not None and idsb is not None:
        # the next block's ids, into the other parity's tile (`idsb`: that one)
        attn_block_ids(i, nxt_tok0, U, gtab_hbm, idsb, kpad)
    lds = [(i, tok0 + u, bufs[u]["idrow"], bufs[u]["qrow"], bufs[u]["gn"])
           + ((bufs[u]["kt"], nwt) if "kt" in bufs[u] else ()) for u in range(U)]
    rot = attn_rot_bufs(ATTN_ROT, heads, d)

    def finish(st, u):
        attn_token_b(st, o_hbm, sink_t, heads, d, kpad, rb=rot[u % ATTN_ROT])
        if nxt_tok0 is not None:
            b = bufs[u]
            view = idsb[0:PT, u, :] if idsb is not None else None
            ld = attn_token_load(i, nxt_tok0 + u, q_hbm, kvtab_hbm, idx_hbm, heads, d, kpad,
                                 gtab_hbm=gtab_hbm, buf=b, nwt=nwt, slab_now=False, ids_view=view)
            attn_token_load2(ld, q_hbm, idx_hbm, heads, d, kpad, buf=b, btab_hbm=btab_hbm)
            if nwt:
                pend.append((view if view is not None else b["ids"], b["gn"]))
        if nwt:
            attn_slab_flush(pend, kvtab_hbm, nwt, d, keep=ATTN_SLAB_LAG if nxt_tok0 is not None else 0)

    pb = list(carry) if carry else []
    for u in range(U):
        st = attn_token_a(lds[u], heads, d, kpad, scale)
        pb.append(lambda st=st, u=u: finish(st, u))
        while len(pb) > ATTN_DEPTH:
            pb.pop(0)()
    if close:
        while pb:
            pb.pop(0)()
    return pend, (pb or None)


# --------------------------------------------------------------------------- MoE routing


def route_topk(scores_hbm, bias_hbm, lo_hbm, tiles, n_experts, topk, n_local, route_scale,
               wexp_hbm, live_hbm=None):
    """The router selection: `sqrtsoftplus` of the fp32 logits in `scores_hbm[T, n_experts]`,
    top-`topk` of `score + bias`, raw-score weights normalized to sum 1 and scaled, and
    per-local-expert dense weights `wexp_hbm[T, n_local]` (fp32): the routing weight of
    expert `lo + e` for the row, or 0 when the row did not pick it.

    `live_hbm`, an fp32 `[1, 1]` row count, zeroes the weights of every row at or past
    it: the padding rows of a bucketed step pick no expert, as `model._MOE_LIVE_ROWS`
    keeps them out of the per-op routing.
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
        if live_hbm is not None:
            ramp = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.iota(dst=ramp[0:nr, :], pattern=[[1, 1]], offset=r0, channel_multiplier=1)
            rf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=rf[0:nr, :], src=ramp[0:nr, :])
            lv = bcast_row(live_hbm[0:1, 0:1], 1, nr)
            keep = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=keep[0:nr, :], data1=rf[0:nr, :], data2=lv, op=nl.less)
            nisa.tensor_scalar(dst=wexp[0:nr, :], data=wexp[0:nr, :], op0=nl.multiply, operand0=keep[0:nr, 0:1])
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


def swiglu_sb(g16, u16, y16, nr, inter, limit, chunk=None):
    """`swiglu_rows` (unit weight) on SBUF: the bf16 `(PT, inter)` gate/up tiles into
    the bf16 `y16`, the same fp32 ops."""
    chunk = chunk or _swiglu_chunk(inter)
    for j in range(inter // chunk):
        c0 = j * chunk
        g = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=g[0:nr, :], data=g16[0:nr, c0 : c0 + chunk], op0=nl.minimum,
                           operand0=limit)
        u = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=u[0:nr, :], data=u16[0:nr, c0 : c0 + chunk], op0=nl.maximum,
                           operand0=-limit, op1=nl.minimum, operand1=limit)
        sg = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sg[0:nr, :], op=nl.silu, data=g[0:nr, :])
        y = nl.ndarray((PT, chunk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y[0:nr, :], data1=sg[0:nr, :], data2=u[0:nr, :], op=nl.multiply)
        nisa.tensor_copy(dst=y16[0:nr, c0 : c0 + chunk], src=y[0:nr, :])


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
