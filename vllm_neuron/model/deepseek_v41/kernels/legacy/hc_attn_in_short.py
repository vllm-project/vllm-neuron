"""The NKI kernel for layers.0.hc_attn_in.

    flat  = x.flatten(2).float()                        # (tok, hc*dim) = (128, 20480)
    rsqrt = 1 / sqrt(mean(flat**2, -1) + norm_eps)       # one statistic per token
    mixes = (flat @ fn.T) * rsqrt                       # (128, 24), fp32
    pre   = sigmoid(mixes[:4]  * scale[0] + base[:4]) + eps
    post  = 2 * sigmoid(mixes[4:8] * scale[1] + base[4:8])
    comb  = mixes[8:24] * scale[2] + base[8:24]  reshaped (4, 4)
    comb  = softmax(comb, -1) + eps
    comb  = comb / (comb.sum(-2) + eps)                  # one column pass, then
    repeat 19x:  comb = comb / (comb.sum(-1) + eps);  comb = comb / (comb.sum(-2) + eps)
    y     = sum_j pre_mix[..., j] * x[..., j, :]         # fp32, then back to bfloat16
    return y, post, comb, pre
"""

import nki
import nki.language as nl
import nki.isa as nisa

#: SBUF partitions in one tile, and so tokens handled per outer iteration.
PMAX = 128

#: Elements of the flattened `hc*dim` axis contracted per PE step. The PE contracts along
#: partitions, so this is exactly the partition count.
KCHUNK = 128

#: Elements of that axis brought in per DMA.
KGROUP = 1024

#: Columns of `y` written per store. The accumulator is built for the whole row, so this is
#: only about how the float32 -> bfloat16 rounding and the DMA out are chunked; 2560 beat
#: both 1024 and one store of all 5120.
YCHUNK = 2560

#: `hc_split_sinkhorn`'s own constants, from `config.json`: hc_mult, hc_sinkhorn_iters,
#: hc_eps, norm_eps. They are the module's configuration, not anything read off a tensor.
HC = 4
SINKHORN_ITERS = 20
HC_EPS = 1e-06
NORM_EPS = 1e-20


def _col(shape0):
    """One per-token float32 column, (P, 1)."""
    return nl.ndarray((shape0, 1), dtype=nl.float32, buffer=nl.sbuf)


def _inv_eps(dst, data):
    """`dst = 1 / (data + eps)`, one instruction on the Scalar Engine."""
    nisa.activation(dst=dst, op=nl.reciprocal, data=data, bias=HC_EPS)


def _row_softmax(comb):
    """`comb.softmax(-1) + eps`, row by row, in place."""
    p = comb.shape[0]
    for j in range(HC):
        row = comb[:, j, :]
        neg_max = _col(p)
        nisa.tensor_reduce(dst=neg_max, op=nl.max, data=row, axis=(1,), negate=True)
        row_sum = _col(p)
        shifted = nl.ndarray((p, HC), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=shifted, op=nl.exp, data=row, bias=neg_max,
                        reduce_op=nl.add, reduce_res=row_sum,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        inv = _col(p)
        nisa.reciprocal(dst=inv, data=row_sum)
        nisa.tensor_scalar(dst=row, data=shifted, op0=nl.multiply, operand0=inv,
                           op1=nl.add, operand1=HC_EPS, engine=nisa.vector_engine)


def _row_normalize(comb):
    """`comb = comb / (comb.sum(-1) + eps)`, in place."""
    p = comb.shape[0]
    row_sum = nl.ndarray((p, HC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=row_sum, op=nl.add, data=comb, axis=(2,))
    inv = nl.ndarray((p, HC, 1), dtype=nl.float32, buffer=nl.sbuf)
    _inv_eps(inv.reshape((p, HC)), row_sum)
    invrep = nl.broadcast_to(inv, (p, HC, HC))
    nisa.tensor_tensor(dst=comb, data1=comb, data2=invrep, op=nl.multiply,
                       engine=nisa.vector_engine)


def _col_normalize(comb):
    """`comb = comb / (comb.sum(-2) + eps)`, in place."""
    p = comb.shape[0]
    assert HC >= 2 and HC & (HC - 1) == 0, "hc must be a power of two and at least 2"
    col_sum = comb.reshape((p, HC * HC))
    n = HC
    while n > 1:
        n //= 2
        nxt = nl.ndarray((p, n * HC), dtype=nl.float32, buffer=nl.sbuf)
        if n == 1:
            nisa.scalar_tensor_tensor(dst=nxt, data=col_sum[:, 0:HC], op0=nl.add,
                                      operand0=HC_EPS, op1=nl.add,
                                      operand1=col_sum[:, HC:2 * HC])
        else:
            nisa.tensor_tensor(dst=nxt, data1=col_sum[:, 0:n * HC],
                               data2=col_sum[:, n * HC:2 * n * HC], op=nl.add,
                               engine=nisa.vector_engine)
        col_sum = nxt
    inv = nl.ndarray((p, 1, HC), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv.reshape((p, HC)), data=col_sum)
    invrep = nl.broadcast_to(inv, (p, HC, HC))
    nisa.tensor_tensor(dst=comb, data1=comb, data2=invrep, op=nl.multiply,
                       engine=nisa.vector_engine)


@nki.jit
def kernel(input_0, input_1, layers_0_hc_attn_in_base, layers_0_hc_attn_in_fn,
           layers_0_hc_attn_in_scale):
    """Compute layers.0.hc_attn_in (hc_attn_in) in NKI."""
    n_tok = input_0.shape[1]
    hc = input_0.shape[2]
    dim = input_0.shape[3]
    flat_dim = hc * dim
    mix_hc = layers_0_hc_attn_in_fn.shape[0]

    assert flat_dim % KGROUP == 0 and dim % KGROUP == 0 and KGROUP % KCHUNK == 0
    assert dim % YCHUNK == 0 and n_tok % PMAX == 0

    y_out = nl.ndarray((1, n_tok, dim), dtype=input_0.dtype, buffer=nl.shared_hbm)
    post_out = nl.ndarray((1, n_tok, hc), dtype=nl.float32, buffer=nl.shared_hbm)
    comb_out = nl.ndarray((1, n_tok, hc, hc), dtype=nl.float32, buffer=nl.shared_hbm)
    pre_out = nl.ndarray((1, n_tok, hc), dtype=nl.float32, buffer=nl.shared_hbm)

    x2 = input_0.reshape((n_tok, flat_dim))
    pm2 = input_1.reshape((n_tok, hc))
    y2 = y_out.reshape((n_tok, dim))
    post2 = post_out.reshape((n_tok, hc))
    comb2 = comb_out.reshape((n_tok, hc, hc))
    pre2 = pre_out.reshape((n_tok, hc))
    base2 = layers_0_hc_attn_in_base.reshape((1, mix_hc))
    scale2 = layers_0_hc_attn_in_scale.reshape((1, 3))

    for t in nl.affine_range(n_tok // PMAX):
        t0 = t * PMAX

        mix = nl.ndarray((PMAX, mix_hc), dtype=nl.float32, buffer=nl.psum)
        ss = nl.ndarray((PMAX, flat_dim // KGROUP), dtype=nl.float32, buffer=nl.sbuf)

        pm = nl.load(pm2[t0:t0 + PMAX, :])
        acc = nl.ndarray((PMAX, dim), dtype=nl.float32, buffer=nl.sbuf)

        for g in range(flat_dim // KGROUP):
            g0 = g * KGROUP

            xg = nl.load(x2[t0:t0 + PMAX, g0:g0 + KGROUP])
            wg = nl.load(layers_0_hc_attn_in_fn[0:mix_hc, g0:g0 + KGROUP],
                         dtype=nl.bfloat16)

            j = g0 // dim
            f0 = g0 - j * dim
            if j == 0:
                nisa.tensor_scalar(dst=acc[:, f0:f0 + KGROUP], data=xg, op0=nl.multiply,
                                   operand0=pm[:, j:j + 1])
            else:
                nisa.scalar_tensor_tensor(dst=acc[:, f0:f0 + KGROUP], data=xg,
                                          op0=nl.multiply, operand0=pm[:, j:j + 1],
                                          op1=nl.add, operand1=acc[:, f0:f0 + KGROUP])

            sink = nl.ndarray((PMAX, KGROUP), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=sink, op=nl.square, data=xg,
                            reduce_op=nl.add, reduce_res=ss[:, g:g + 1],
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)


            n_sub = KGROUP // KCHUNK
            xt_ps = nl.ndarray((PMAX, KGROUP), dtype=nl.bfloat16, buffer=nl.psum)
            wt_ps = nl.ndarray((KCHUNK, n_sub * mix_hc), dtype=nl.bfloat16, buffer=nl.psum)
            for i in range(n_sub):
                nisa.nc_transpose(dst=xt_ps[:, i * KCHUNK:(i + 1) * KCHUNK],
                                  data=xg[:, i * KCHUNK:(i + 1) * KCHUNK])
                nisa.nc_transpose(dst=wt_ps[:, i * mix_hc:(i + 1) * mix_hc],
                                  data=wg[:, i * KCHUNK:(i + 1) * KCHUNK])
            xt = nl.ndarray((PMAX, KGROUP), dtype=nl.bfloat16, buffer=nl.sbuf)
            wt = nl.ndarray((KCHUNK, n_sub * mix_hc), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=xt, src=xt_ps)
            nisa.tensor_copy(dst=wt, src=wt_ps, engine=nisa.scalar_engine)

            for i in range(n_sub):
                nisa.nc_matmul(dst=mix,
                               stationary=xt[:, i * KCHUNK:(i + 1) * KCHUNK],
                               moving=wt[:, i * mix_hc:(i + 1) * mix_hc],
                               accumulate=(g > 0 or i > 0))

        rsqrt = nl.rsqrt(nl.add(nl.multiply(nl.sum(ss, axis=1, keepdims=True),
                                           1.0 / flat_dim), NORM_EPS))
        mixes = nl.ndarray((PMAX, mix_hc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mixes, data=mix, op0=nl.multiply, operand0=rsqrt)

        base_b = nl.broadcast_to(nl.load(base2), (PMAX, mix_hc))
        scale_row = nl.load(scale2)

        scaled = nl.ndarray((PMAX, mix_hc), dtype=nl.float32, buffer=nl.sbuf)
        for i in range(3):
            width = hc if i < 2 else hc * hc
            lo = i * hc
            nisa.tensor_scalar(dst=scaled[:, lo:lo + width], data=mixes[:, lo:lo + width],
                               op0=nl.multiply,
                               operand0=nl.broadcast_to(scale_row[0:1, i:i + 1], (PMAX, 1)))
        nisa.tensor_tensor(dst=scaled, data1=scaled, data2=base_b, op=nl.add)

        pre = nl.ndarray((PMAX, hc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=pre, op=nl.sigmoid, data=scaled[:, 0:hc])
        nisa.tensor_scalar(dst=pre, data=pre, op0=nl.add, operand0=HC_EPS)

        post = nl.ndarray((PMAX, hc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=post, op=nl.sigmoid, data=scaled[:, hc:2 * hc])
        nisa.tensor_scalar(dst=post, data=post, op0=nl.multiply, operand0=2.0)

        comb = nl.ndarray((PMAX, hc, hc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=comb.reshape((PMAX, hc * hc)),
                         src=scaled[:, 2 * hc:mix_hc])

        for f in range(dim // YCHUNK):
            f0 = f * YCHUNK
            nl.store(y2[t0:t0 + PMAX, f0:f0 + YCHUNK],
                     value=nl.copy(acc[:, f0:f0 + YCHUNK], dtype=input_0.dtype))

        _row_softmax(comb)
        _col_normalize(comb)
        for _ in range(SINKHORN_ITERS - 1):
            _row_normalize(comb)
            _col_normalize(comb)

        nl.store(pre2[t0:t0 + PMAX, :], value=pre)
        nl.store(post2[t0:t0 + PMAX, :], value=post)
        nl.store(comb2[t0:t0 + PMAX, :, :], value=comb)

    return y_out, post_out, comb_out, pre_out
