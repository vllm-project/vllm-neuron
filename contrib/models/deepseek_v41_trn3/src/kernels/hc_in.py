# SPDX-License-Identifier: Apache-2.0
"""The hyper-connection **sublayer-in** as one NKI kernel.

What one call computes, for a `[T, hc, dim]` bf16 residual stream `x` and the
previous sublayer's `pre_mix` `[T, hc]` fp32 (the reference's
`HyperConnectSublayerIn` + `hc_split_sinkhorn` followed by the sublayer's
RMSNorm):

    flat   = x.flatten(1).float()                          # [T, hc*dim]
    rsq    = rsqrt(mean(flat^2, -1) + norm_eps)            # one scalar a token
    mixes  = (flat @ fn.T) * rsq                           # [T, 24], fp32 GEMM
    pre    = sigmoid(mixes[:, 0:hc]  * scale[0] + base[0:hc])  + hc_eps
    post   = 2 * sigmoid(mixes[:, hc:2hc] * scale[1] + base[hc:2hc])
    comb   = softmax((mixes[:, 2hc:] * scale[2] + base[2hc:]).view(hc, hc), -1) + hc_eps
    comb   = comb / (comb.sum(-2) + eps); 19 x { comb /= comb.sum(-1) + eps; comb /= comb.sum(-2) + eps }
    y      = bf16( sum_j pre_mix[:, j] * x[:, j, :] )      # the collapse, fp32 accumulate
    normed = bf16( y.float() * rsqrt(mean(y^2) + norm_eps) * norm_w )

and returns `(normed [T, dim] bf16, pre, post [T, hc] fp32, comb [T, hc*hc] fp32)`.
`pre` is the *next* sublayer's collapse coefficient; `post`/`comb` feed this
sublayer's `hc_out`. This replaces `torch.matmul(rows.float(), fn.t())` +
`kernels.hc_mixes` + `kernels.hc_collapse` + `kernels.rmsnorm` -- four launches
and one HLO-side fp32 GEMM -- with one kernel, twice per layer.

Numerics, chosen to match what the port validated against the checkpoint's own
reference: the mixes GEMM runs with **fp32 operands** on the tensor engine (the
reference computes `F.linear(x.float(), fn)` in fp32; a bf16-operand GEMM moves
the coefficients by ~0.4%), both RMS statistics are fp32 sums of fp32 squares,
the collapse accumulates in fp32 and is rounded to bf16 exactly once before the
norm (as `HyperConnections.pre` returned bf16), and the Sinkhorn runs exactly
`SINKHORN_ITERS` passes.

Layout. The contraction axis of the GEMM is the flattened stream, so each 128-
wide chunk of `x` is transposed on the tensor engine (bf16, bit-exact), widened
to fp32 and used as the stationary operand against the matching 128 rows of
`fn_t = fn.T` (`[hc*dim, 24]` fp32, staged once in SBUF). Tokens sit on the
partition axis everywhere else; 128 a tile with a partial last tile.
"""

import nki
import nki.isa as nisa
import nki.language as nl

#: Partition width: tokens a tile.
PT = 128
#: `hc_sinkhorn_iters`, `hc_eps`, `rms_norm_eps` -- the config's; `model.py`
#: checks them against the config at construction.
SINKHORN_ITERS = 20
HC_EPS = 1e-06
NORM_EPS = 1e-20
#: Free-axis chunk of the sum-of-squares passes (fp32 scratch of this width).
SQ_CHUNK = 1024
#: Free-axis chunk of the normalized output stores.
OUT_CHUNK = 1024


def _rsqrt(m):
    """`1 / sqrt(m)` with one Newton step, as `kernels.rmsnorm` does it."""
    y = nl.rsqrt(m)
    corr = nl.add(nl.multiply(nl.multiply(m, nl.multiply(y, y)), -0.5), 1.5)
    return nl.multiply(y, corr)


def _sum_squares(dst, src, nr, width, chunk):
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


def _sinkhorn(t2, nr, hc):
    """`softmax(rows) + eps`, then 20 alternating column/row normalizations.
    `t2` is `[nr, hc*hc]` fp32 in SBUF (row `i` of the matrix at `[i*hc, (i+1)*hc)`).
    Returns the SBUF tile holding `comb`.
    """

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


@nki.jit
def sublayer_in(x, pre_mix, fn_t, base, scale, norm_w):
    """See the module docstring. `fn_t` is `fn.T`, `[hc*dim, 24]` fp32."""
    T, hc, dim = x.shape
    width = hc * dim
    kw, mix = fn_t.shape
    assert kw == width, f"fn_t {fn_t.shape} wants ({width}, mix)"
    assert mix == (2 + hc) * hc, f"mix width {mix} is not (2 + hc) * hc for hc={hc}"
    assert width % PT == 0, f"stream width {width} must be a multiple of {PT}"
    assert dim % SQ_CHUNK == 0 and dim % OUT_CHUNK == 0, f"dim={dim}"
    assert pre_mix.shape == (T, hc), f"pre_mix {pre_mix.shape} wants ({T}, {hc})"
    assert base.shape == (1, mix), f"base {base.shape} wants (1, {mix})"
    assert scale.shape == (1, 3), f"scale {scale.shape} wants (1, 3)"
    assert norm_w.shape == (1, dim), f"norm_w {norm_w.shape} wants (1, {dim})"

    nk = width // PT
    kpg = dim // PT  # contraction chunks a hc copy

    normed = nl.ndarray((T, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    pre = nl.ndarray((T, hc), dtype=nl.float32, buffer=nl.shared_hbm)
    post = nl.ndarray((T, hc), dtype=nl.float32, buffer=nl.shared_hbm)
    comb = nl.ndarray((T, hc * hc), dtype=nl.float32, buffer=nl.shared_hbm)

    x2 = x.reshape((T, width))

    # fn^T resident for the whole kernel: chunk kb of the contraction axis at
    # fnT[:, kb, :], 128 contraction rows on partitions, the 24 mix columns free.
    fnT = nl.ndarray((PT, nk, mix), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=fnT, src=fn_t.ap(pattern=[[mix, PT], [PT * mix, nk], [1, mix]]))

    inv_width = 1.0 / width
    inv_dim = 1.0 / dim

    def body(r0, nr):
        # One tile per hc copy, every read issued before any compute.
        xg = []
        for g in range(hc):
            xg.append(nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf))
        for g in range(hc):
            nisa.dma_copy(dst=xg[g][0:nr, :], src=x2[r0 : r0 + nr, g * dim : (g + 1) * dim])
        pm = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=pm[0:nr, :], src=pre_mix[r0 : r0 + nr, 0:hc])

        # --- the collapse: acc = sum_j pm[:, j] * x[:, j, :] in fp32, then bf16 ---
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

        # --- the stream statistic over all hc copies ---
        ss = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        junk = nl.ndarray((PT, SQ_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
        nsq = dim // SQ_CHUNK
        for g in range(hc):
            for c in range(nsq):
                first = g == 0 and c == 0
                last = g == hc - 1 and c == nsq - 1
                nisa.activation(
                    dst=junk[0:nr, :], op=nl.square,
                    data=xg[g][0:nr, c * SQ_CHUNK : (c + 1) * SQ_CHUNK],
                    reduce_op=nl.add,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce if first else nisa.reduce_cmd.reduce,
                    reduce_res=ss[0:nr, 0:1] if last else None,
                )

        # --- the mixes GEMM, fp32 operands, contraction on partitions ---
        mixp = nl.ndarray((PT, mix), dtype=nl.float32, buffer=nl.psum)
        for g in range(hc):
            for c in range(kpg):
                kb = g * kpg + c
                tps = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(
                    dst=tps[0:PT, 0:nr], data=xg[g][0:nr, c * PT : (c + 1) * PT]
                )
                xT = nl.ndarray((PT, PT), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=xT[0:PT, 0:nr], src=tps[0:PT, 0:nr])
                nisa.nc_matmul(
                    dst=mixp[0:nr, 0:mix],
                    stationary=xT[0:PT, 0:nr],
                    moving=fnT[0:PT, kb, 0:mix],
                    accumulate=(kb > 0),
                )

        # --- the coefficients ---
        rs = _rsqrt(nl.add(nl.multiply(ss[0:nr, 0:1], inv_width), NORM_EPS))
        mixv = nl.ndarray((PT, mix), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mixv[0:nr, :], data=mixp[0:nr, :], op0=nl.multiply, operand0=rs)

        s0 = nl.broadcast_to(nl.load(scale[0:1, 0:1]), (nr, 1))
        s1 = nl.broadcast_to(nl.load(scale[0:1, 1:2]), (nr, 1))
        s2 = nl.broadcast_to(nl.load(scale[0:1, 2:3]), (nr, 1))
        b0 = nl.broadcast_to(nl.load(base[0:1, 0:hc]), (nr, hc))
        b1 = nl.broadcast_to(nl.load(base[0:1, hc : 2 * hc]), (nr, hc))
        b2 = nl.broadcast_to(nl.load(base[0:1, 2 * hc : 2 * hc + hc * hc]), (nr, hc * hc))

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
        comb_t = _sinkhorn(t2, nr, hc)
        nisa.dma_copy(dst=comb[r0 : r0 + nr, 0 : hc * hc], src=comb_t[0:nr, :])

        # --- the sublayer RMSNorm of the collapsed stream ---
        ss2 = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        _sum_squares(ss2, y16, nr, dim, SQ_CHUNK)
        rstd = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        rstd[0:nr, :] = _rsqrt(nl.add(nl.multiply(ss2[0:nr, 0:1], inv_dim), NORM_EPS))
        for c in range(dim // OUT_CHUNK):
            lo, hi = c * OUT_CHUNK, (c + 1) * OUT_CHUNK
            wj = nl.broadcast_to(nl.load(norm_w[0:1, lo:hi], dtype=nl.float32), (nr, OUT_CHUNK))
            yv = nl.ndarray((PT, OUT_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=yv[0:nr, :], op=nl.copy, data=y16[0:nr, lo:hi], scale=rstd[0:nr, 0:1])
            nisa.tensor_tensor(dst=yv[0:nr, :], data1=yv[0:nr, :], data2=wj, op=nl.multiply)
            o16 = nl.ndarray((PT, OUT_CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=o16[0:nr, :], src=yv[0:nr, :])
            nisa.dma_copy(dst=normed[r0 : r0 + nr, lo:hi], src=o16[0:nr, :])

    full, tail = T // PT, T % PT
    for t in range(full):
        body(t * PT, PT)
    if tail:
        body(full * PT, tail)
    return normed, pre, post, comb
