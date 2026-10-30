# SPDX-License-Identifier: Apache-2.0
"""The compressed-KV source machinery of the layer megakernel: the compressor,
the indexer's keys/queries/scores and the top-`index_topk` selection.

All helpers run on both physical cores and take the row tiles the calling core
owns, except where the docstring says otherwise. Numerics mirror `model.py`:

  * `fp4_qdq`: `fp4_quant_dequant(x, block, scale_e4m3)` -- e2m1 round trip with
    either a power-of-two block scale (`scale = 2^ceil(log2(amax/6))`) or an e4m3
    block scale (`round_to_e4m3(amax/6)`, the quotient refined by one Newton step
    before the half-to-even e2m1 rounding, exactly as the model does on the host).
  * compressor (ratio 2): fp32 projections written to the per-slot pooling caches,
    softmax pooling over each group's members in fp32, RMSNorm to bf16;
    (ratio 1): the bf16 `wkv` GEMM and RMSNorm.
  * indexer keys: `k_norm(wk(latent))` (bf16 GEMM), rope, fp4 (pow2) round trip.
  * index scores: `sum_h relu(q_h . k) * w_h` per token with fp32 PSUM accumulation,
    the sink-free formulation of `kernels.index_score`.
"""

import nki.isa as nisa
import nki.language as nl

from .blockquant import _pow2_ceil_log2
from .mk_common import (
    PT,
    NEG_BOUND,
    _gemm,
    _stage_rows,
    bcast_row,
    cdiv,
    load_freqs,
    norm_rows,
    rope_inplace,
    rsqrt_newton,
    sum_squares,
)

FP4_MAX = 6.0
#: 1.5 * 2^23: `y + RNE_MAGIC` lands in [2^23, 2^24) for |y| < 2^22, where the fp32
#: ulp is exactly 1, for either sign (at 2^23 itself a negative `y` would sit just
#: below the binade with an ulp of 0.5 and come back unrounded).
RNE_MAGIC = 12582912.0
F32_BIAS = 127


def _floor_log2(mag):
    """`floor(log2(mag))` as int32, for positive normal fp32 `mag` (bit arithmetic)."""
    bits = nl.copy(mag, dtype=nl.float32).view(nl.int32)
    return nl.subtract(nl.bitwise_and(nl.right_shift(bits, 23), 0xFF), F32_BIAS)


def _pow2_of(e):
    """`2 ** e` as fp32 from an int32 exponent tile."""
    return nl.copy(nl.left_shift(nl.add(e, F32_BIAS), 23)).view(nl.float32)


def _rne(y, shape):
    """Round-half-to-even of the fp32 SBUF tile `y` (|y| < 2^22) via the 2^23 magic
    constant. Two separate instructions on purpose: fused into one two-op
    `tensor_scalar`, the add and the subtract cancel in a wider intermediate and
    nothing is rounded (measured: negative inputs came back unchanged).
    """
    t = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t, data=y, op0=nl.add, operand0=RNE_MAGIC, engine=nisa.engine.vector)
    r = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=r, data=t, op0=nl.subtract, operand0=RNE_MAGIC, engine=nisa.engine.vector)
    return r


def _abs(x, shape):
    a = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=a, op=nl.abs, data=x)
    return a


def _round_e2m1(n, shape):
    """`_round_to_bits(n, mantissa=1, min_exp=0)`: nearest e2m1 value of `n` in [-6, 6]."""
    mag = nl.maximum(_abs(n, shape), 1.0)
    e = _floor_log2(mag)
    step = _pow2_of(nl.subtract(e, 1))
    inv_step = _pow2_of(nl.subtract(1, e))
    y = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=y, data1=n, data2=inv_step, op=nl.multiply)
    return nl.multiply(_rne(y, shape), step)


def _round_e4m3(r, shape):
    """`_round_to_bits(r, mantissa=3, min_exp=-6)` for positive `r`."""
    mag = nl.maximum(r, 2.0 ** -6)
    e = _floor_log2(mag)
    step = _pow2_of(nl.subtract(e, 3))
    inv_step = _pow2_of(nl.subtract(3, e))
    y = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=y, data1=r, data2=inv_step, op=nl.multiply)
    return nl.multiply(_rne(y, shape), step)


def fp4_qdq_tile(x, nr, n, blk, e4m3):
    """`fp4_quant_dequant` of the fp32 SBUF tile `x[0:nr, 0:n]`; returns a bf16 SBUF tile."""
    nb = n // blk
    xb = x[0:nr, 0:n].reshape((nr, nb, blk))
    ab = nl.ndarray((PT, nb, blk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=ab[0:nr], op=nl.abs, data=xb)
    amax = nl.ndarray((PT, nb), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=amax[0:nr], op=nl.maximum, data=ab[0:nr], axis=(2,))
    if e4m3:
        am = nl.maximum(amax[0:nr], 6.0 * 2.0 ** -9)
        scale = _round_e4m3(nl.multiply(am, 1.0 / FP4_MAX), (nr, nb))
        # 1 / scale with one Newton step, then the quotient refined once more against
        # the exact product, as the host does: n = n - (n * scale - x) / scale.
        rcp = nl.ndarray((PT, nb), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rcp[0:nr], data=scale)
        rcp2 = nl.multiply(rcp[0:nr], nl.subtract(2.0, nl.multiply(scale, rcp[0:nr])))
        sb = nl.broadcast_to(scale.reshape((nr, nb, 1)), (nr, nb, blk))
        rb = nl.broadcast_to(rcp2.reshape((nr, nb, 1)), (nr, nb, blk))
        nrm = nl.multiply(xb, rb)
        resid = nl.subtract(nl.multiply(nrm, sb), xb)
        nrm = nl.subtract(nrm, nl.multiply(resid, rb))
    else:
        am = nl.maximum(amax[0:nr], 6.0 * 2.0 ** -126)
        scale, inv = _pow2_ceil_log2(nl.multiply(am, 1.0 / FP4_MAX))
        sb = nl.broadcast_to(scale.reshape((nr, nb, 1)), (nr, nb, blk))
        ib = nl.broadcast_to(inv.reshape((nr, nb, 1)), (nr, nb, blk))
        nrm = nl.multiply(xb, ib)
    nrm = nl.minimum(nl.maximum(nrm, -FP4_MAX), FP4_MAX)
    rounded = _round_e2m1(nrm, (nr, nb, blk))
    out = nl.ndarray((PT, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out[0:nr, :].reshape((nr, nb, blk)), src=nl.multiply(rounded, sb))
    return out


def rope_fp4_put(src_hbm, tiles, width, head, nheads, freqs_hbm, pos_hbm, blk, e4m3,
                 cache_hbm=None, slots_hbm=None, dst_hbm=None):
    """For the row tiles: load bf16 rows of `src_hbm[rows, width]`, rope the last 64
    columns of each of `nheads` heads of width `head` at positions `pos_hbm`, fp4
    round-trip (`blk`, `e4m3`) and scatter into `cache_hbm[slots_hbm[row]]`
    (and, optionally, store to `dst_hbm[rows]`).
    """
    for r0, nr in tiles:
        t = nl.ndarray((PT, width), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=t[0:nr, :], src=src_hbm[r0 : r0 + nr, 0:width])
        cos, sin = load_freqs(freqs_hbm, pos_hbm, r0, nr)
        for h in range(nheads):
            rope_inplace(t, nr, h * head + head - 64, cos, sin, False)
        tf = nl.ndarray((PT, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tf[0:nr, :], src=t[0:nr, :])
        q = fp4_qdq_tile(tf, nr, width, blk, e4m3)
        if cache_hbm is not None:
            ix = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ix[0:nr, :], src=slots_hbm[r0 : r0 + nr, 0:1])
            nisa.dma_copy(
                dst=cache_hbm.ap(pattern=[[width, nr], [1, width]], vector_offset=ix[0:nr, 0:1], indirect_dim=0),
                src=q[0:nr, :],
            )
        if dst_hbm is not None:
            nisa.dma_copy(dst=dst_hbm[r0 : r0 + nr, 0:width], src=q[0:nr, :])


# --------------------------------------------------------------------------- compressor


def fp32_proj_rows(x_hbm, w_t_hbm, out_hbm, tiles, k, n):
    """`out[rows, n] = x[rows, k] (bf16) @ w_t[k, n] (fp32)` with fp32 operands on the
    tensor engine (the compressor's `matmul(xf, weight.float().t())`); `n <= 512`.
    `w_t` is staged once per call as `(PT, k//PT, n)` fp32."""
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


def scatter_rows_f32(src_hbm, slots_hbm, cache_hbm, tiles, n):
    for r0, nr in tiles:
        t = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t[0:nr, :], src=src_hbm[r0 : r0 + nr, 0:n])
        ix = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ix[0:nr, :], src=slots_hbm[r0 : r0 + nr, 0:1])
        nisa.dma_copy(
            dst=cache_hbm.ap(pattern=[[n, nr], [1, n]], vector_offset=ix[0:nr, 0:1], indirect_dim=0),
            src=t[0:nr, :],
        )


def pool_groups(pool_kv, pool_score, mslots_hbm, tiles, ratio, hd, out_hbm):
    """`out[g] = sum_m kv[g, m] * softmax_m(score[g, m])` over each group's `ratio`
    member slots (`mslots_hbm[g*ratio + m]`), fp32 in, bf16 out (the latent before
    its norm). `tiles` are group tiles."""
    for g0, ng in tiles:
        kvm, scm = [], []
        for m in range(ratio):
            ix = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ix[0:ng, :],
                          src=mslots_hbm.ap(pattern=[[ratio, ng], [1, 1]], offset=g0 * ratio + m))
            kv = nl.ndarray((PT, hd), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=kv[0:ng, :], src=pool_kv.ap(pattern=[[hd, ng], [1, hd]],
                                                            vector_offset=ix[0:ng, 0:1], indirect_dim=0))
            sc = nl.ndarray((PT, hd), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=sc[0:ng, :], src=pool_score.ap(pattern=[[hd, ng], [1, hd]],
                                                               vector_offset=ix[0:ng, 0:1], indirect_dim=0))
            kvm.append(kv)
            scm.append(sc)
        mx = scm[0][0:ng, :]
        for m in range(1, ratio):
            mx = nl.maximum(mx, scm[m][0:ng, :])
        es = [nl.exp(nl.subtract(scm[m][0:ng, :], mx)) for m in range(ratio)]
        den = es[0]
        for m in range(1, ratio):
            den = nl.add(den, es[m])
        rz = nl.ndarray((PT, hd), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rz[0:ng, :], data=den)
        rz2 = nl.multiply(rz[0:ng, :], nl.subtract(2.0, nl.multiply(den, rz[0:ng, :])))
        acc = None
        for m in range(ratio):
            term = nl.multiply(kvm[m][0:ng, :], nl.multiply(es[m], rz2))
            acc = term if acc is None else nl.add(acc, term)
        o16 = nl.ndarray((PT, hd), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o16[0:ng, :], src=acc)
        nisa.dma_copy(dst=out_hbm[g0 : g0 + ng, 0:hd], src=o16[0:ng, :])


def bf16_gemm_rows(x_hbm, w_hbm, out_hbm, m, k, n, lo, hi):
    """`out[:, lo:hi] = x[m, k] @ w[lo:hi, :].T`, bf16 operands (no quantization), bf16 out;
    `m` a multiple of 128 (`x_hbm` is bf16 HBM of that height)."""
    _gemm(x_hbm, w_hbm[lo:hi, 0:k], out_hbm[0:m, lo:hi], m, k, hi - lo)


# --------------------------------------------------------------------------- indexer


#: `index_scores_span`'s keys per score matmul (one fp32 PSUM bank), the last block partial:
#: each score is the same column of the same matmul whatever the block's width (128-key
#: blocks when read_len is not a multiple of 512 cost 4x the instructions, e.g. 66 a token
#: at read_len 8448).
IDX_SCORE_BLOCK = 512


def index_scores_span(qi_hbm, wi_hbm, ktab_hbm, sc_hbm, tok0, ntok, nh, read_len, kd=128, k0=0,
                      k1=None):
    """Index scores for `ntok` tokens from row `tok0` of one span (a dynamic loop):
    `sc[t, j] = sum_h relu(q[t, h] . k[j]) * w[t, h]`, fp32, for keys `k0 <= j < k1`
    (default all; multiples of 128). `ktab_hbm[read_len, kd]`
    (bf16) is the span's index-key table; `qi_hbm[Tp, nh*kd]` the fp4-rounded, roped
    queries; `wi_hbm[Tp, nh]` the per-head weights (already scaled)."""
    k1 = read_len if k1 is None else k1
    nk = k1 - k0
    nt = IDX_SCORE_BLOCK
    blocks = [(b0, min(nt, nk - b0)) for b0 in range(0, nk, nt)]
    kT = nl.ndarray((PT, nk), dtype=nl.bfloat16, buffer=nl.sbuf)
    for jb in range(nk // PT):
        nisa.dma_transpose(dst=kT[0:PT, jb * PT : (jb + 1) * PT],
                           src=ktab_hbm[k0 + jb * PT : k0 + (jb + 1) * PT, 0:kd])
    hp = -(-nh // 16) * 16

    def body(i):
        qrow = nl.ndarray((hp, kd), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=qrow[0:nh, :],
                      src=qi_hbm.ap(pattern=[[kd, nh], [1, kd]], offset=tok0 * nh * kd,
                                    scalar_offset=i, indirect_dim=0))
        qps = nl.ndarray((PT, hp), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=qps[0:PT, 0:nh], data=qrow[0:nh, 0:kd])
        qt = nl.ndarray((PT, hp), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qt[0:PT, 0:nh], src=qps[0:PT, 0:nh])
        wt = nl.ndarray((hp, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt[0:nh, :],
                      src=wi_hbm.ap(pattern=[[1, nh], [1, 1]], offset=tok0 * nh, scalar_offset=i, indirect_dim=0))
        row = nl.ndarray((1, nk), dtype=nl.float32, buffer=nl.sbuf)
        for b0, w in blocks:
            acc = nl.ndarray((hp, nt), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=acc[0:nh, 0:w], stationary=qt[0:PT, 0:nh], moving=kT[0:PT, b0 : b0 + w])
            rect = nl.ndarray((hp, nt), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rect[0:nh, 0:w], data=acc[0:nh, 0:w], op0=nl.maximum, operand0=0.0)
            wsum = nl.ndarray((1, nt), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=wsum[0:1, 0:w], stationary=wt[0:nh, 0:1], moving=rect[0:nh, 0:w])
            nisa.tensor_copy(dst=row[0:1, b0 : b0 + w], src=wsum[0:1, 0:w])
        nisa.dma_copy(dst=sc_hbm.ap(pattern=[[read_len, 1], [1, nk]], offset=tok0 * read_len + k0,
                                    scalar_offset=i, indirect_dim=0), src=row)

    nl.fori_loop(0, ntok, body)


#: `index_scores_span` as a static loop: a span's tokens' queries transposed together (one
#: load, one PE transpose for 128 // nh tokens), their weights loaded by one DMA, the
#: per-token score chains unrolled (no per-token dynamic DMAs, which GpSimd generates;
#: the scheduler overlaps the tokens' chains). Each score's operations are unchanged.
IDX_SCORES_STATIC = True


#: `index_scores_span_static` scores up to `PT // S` tokens per key block in one head-score
#: matmul + one ReLU (token u's heads on PSUM partitions `S*u ..`, S = the heads rounded up
#: to 32) instead of one each per token; each token's head-weighted sum stays its own
#: matmul over exactly its heads' partitions (the same operands, at partition offset S*u).
IDX_TOK_GROUP = True

#: With IDX_TOK_GROUP: the odd tokens' head-sum PSUM -> SBUF row copies on the Scalar
#: engine (an fp32 -> fp32 copy, idle during the scoring) instead of all on the DVE, which
#: also runs the ReLUs (the DVE bound the scoring: ~0.55 us a [1, 512] copy).
IDX_COPY_ACT = True


def index_scores_span_static(qi_hbm, wi_hbm, ktab_hbm, sc_hbm, tok0, ntok, nh, read_len, kd=128,
                             k0=0, k1=None):
    """`index_scores_span`'s values, statically unrolled over the `ntok` tokens."""
    k1 = read_len if k1 is None else k1
    nk = k1 - k0
    nt = IDX_SCORE_BLOCK
    blocks = [(b0, min(nt, nk - b0)) for b0 in range(0, nk, nt)]
    kT = nl.ndarray((PT, nk), dtype=nl.bfloat16, buffer=nl.sbuf)
    for jb in range(nk // PT):
        nisa.dma_transpose(dst=kT[0:PT, jb * PT : (jb + 1) * PT],
                           src=ktab_hbm[k0 + jb * PT : k0 + (jb + 1) * PT, 0:kd])
    hp = -(-nh // 16) * 16
    UT = max(1, PT // nh)
    qv = qi_hbm.reshape((qi_hbm.shape[0] * nh, kd))
    S = -(-nh // 32) * 32
    G = PT // S if S <= PT else 1
    if IDX_TOK_GROUP and G > 1:
        for u0 in range(0, ntok, UT):
            un = min(UT, ntok - u0)
            t0 = tok0 + u0
            qrows = nl.ndarray((PT, kd), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=qrows[0 : un * nh, :], src=qv[t0 * nh : (t0 + un) * nh, 0:kd])
            qps = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=qps[0:kd, 0 : un * nh], data=qrows[0 : un * nh, 0:kd])
            for g0 in range(0, un, G):
                gn = min(G, un - g0)
                np_ = S * (gn - 1) + nh
                # qg[:, S*u + h] = token (t0 + g0 + u)'s head h query, 0 between
                qg = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.memset(qg[0:kd, 0:np_], 0.0)
                wg = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
                for u in range(gn):
                    c = (g0 + u) * nh
                    nisa.tensor_copy(dst=qg[0:kd, S * u : S * u + nh], src=qps[0:kd, c : c + nh])
                    nisa.dma_copy(dst=wg[S * u : S * u + nh, 0:1],
                                  src=wi_hbm.ap(pattern=[[1, nh], [1, 1]], offset=(t0 + g0 + u) * nh))
                # token u's scores on partition S*u
                rows = nl.ndarray((PT, nk), dtype=nl.float32, buffer=nl.sbuf)
                for b0, w in blocks:
                    acc = nl.ndarray((PT, nt), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=acc[0:np_, 0:w], stationary=qg[0:kd, 0:np_],
                                   moving=kT[0:PT, b0 : b0 + w])
                    rect = nl.ndarray((PT, nt), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=rect[0:np_, 0:w], data=acc[0:np_, 0:w], op0=nl.maximum,
                                       operand0=0.0)
                    # the per-token weighted sums in the PE's row x column tiles (S-row,
                    # 32-column tiles at (S*u, S*u)), run side by side
                    wsum = nl.ndarray((PT, nt), dtype=nl.float32, buffer=nl.psum)
                    for u in range(gn):
                        p0 = S * u
                        nisa.nc_matmul(dst=wsum[p0 : p0 + 1, 0:w], stationary=wg[p0 : p0 + nh, 0:1],
                                       moving=rect[p0 : p0 + nh, 0:w], tile_position=(p0, p0),
                                       tile_size=(S, 32))
                    for u in range(gn):
                        p0 = S * u
                        eng = (nisa.engine.scalar if (IDX_COPY_ACT and u % 2 == 1)
                               else nisa.engine.unknown)
                        nisa.tensor_copy(dst=rows[p0 : p0 + 1, b0 : b0 + w], src=wsum[p0 : p0 + 1, 0:w],
                                         engine=eng)
                for u in range(gn):
                    t = t0 + g0 + u
                    nisa.dma_copy(dst=sc_hbm[t : t + 1, k0:k1], src=rows[S * u : S * u + 1, 0:nk])
        return
    for u0 in range(0, ntok, UT):
        un = min(UT, ntok - u0)
        t0 = tok0 + u0
        # row u * nh + h = token t0 + u's head h query
        qrows = nl.ndarray((PT, kd), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=qrows[0 : un * nh, :], src=qv[t0 * nh : (t0 + un) * nh, 0:kd])
        qps = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=qps[0:kd, 0 : un * nh], data=qrows[0 : un * nh, 0:kd])
        qt = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qt[0:kd, 0 : un * nh], src=qps[0:kd, 0 : un * nh])
        # wt[h, u] = w[t0 + u, h]
        wt = nl.ndarray((hp, UT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt[0:nh, 0:un], src=wi_hbm.ap(pattern=[[1, nh], [nh, un]], offset=t0 * nh))
        for u in range(un):
            row = nl.ndarray((1, nk), dtype=nl.float32, buffer=nl.sbuf)
            for b0, w in blocks:
                acc = nl.ndarray((hp, nt), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=acc[0:nh, 0:w], stationary=qt[0:kd, u * nh : (u + 1) * nh],
                               moving=kT[0:PT, b0 : b0 + w])
                rect = nl.ndarray((hp, nt), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=rect[0:nh, 0:w], data=acc[0:nh, 0:w], op0=nl.maximum, operand0=0.0)
                wsum = nl.ndarray((1, nt), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=wsum[0:1, 0:w], stationary=wt[0:nh, u : u + 1], moving=rect[0:nh, 0:w])
                nisa.tensor_copy(dst=row[0:1, b0 : b0 + w], src=wsum[0:1, 0:w])
            nisa.dma_copy(dst=sc_hbm[t0 + u : t0 + u + 1, k0:k1], src=row[0:1, 0:nk])


#: `index_topk` builds each 8-token group's 16-partition snake tile from 32-byte row runs
#: and a PE transpose instead of a stride-16 element DMA a token (one 2-byte descriptor an
#: element: ~450 us of sync-queue time a top-k at T = 60, read_len = 8448).
IDX_SNAKE_PE = True

#: `index_topk`'s per-token output scatter (`[[1, 16], [16, k16]]`: one 4-byte descriptor an
#: index, 512 a token) replaced by one PE transpose of the group's indices (fp32, exact below
#: 2^24) and one DMA of 64-byte runs a group.
IDX_TOPK_OUT_PE = True

#: At one row tile (T <= 128) `index_topk`'s mask / bf16 pass (step 1) and the clen pass
#: (step 3) split by columns between the cores instead of core 0 alone (elementwise).
IDX_TOPK_COLSPLIT = True


def index_topk(sc_hbm, clen_hbm, sc16_hbm, topk_hbm, T, Tp, read_len, topk, pid):
    """Mask `sc[t, j]` for `j >= clen[t]` to NEG_BOUND, round to bf16, select the top
    `topk` positions per token on the GpSimd engine (16-partition snake layout) and
    write `topk_hbm[t, 0:topk]` (int32, -1 where the pick lies beyond `clen[t]`).
    Rows beyond `T` are left alone. Token groups of 8 alternate between the cores."""
    n16 = read_len // 16
    k16 = cdiv(topk, 16)
    # 1. mask + bf16, by row tile (this core's tiles; one tile: this core's column half)
    tiles = [(i * PT, min(PT, T - i * PT)) for i in range(cdiv(T, PT)) if i % 2 == pid]
    colsplit = IDX_TOPK_COLSPLIT and cdiv(T, PT) == 1
    if colsplit:
        h1 = (read_len // 2 // 16) * 16
        work = [(0, T, 0, h1) if pid == 0 else (0, T, h1, read_len)]
        hk = (topk // 2 // 16) * 16
        work3 = [(0, T, 0, hk) if pid == 0 else (0, T, hk, topk)]
    else:
        work = [(r0, nr, 0, read_len) for r0, nr in tiles]
        work3 = [(r0, nr, 0, topk) for r0, nr in tiles]
    for r0, nr, c0, c1 in work:
        w = c1 - c0
        if w <= 0:
            continue
        s = nl.ndarray((PT, read_len), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=s[0:nr, 0:w], src=sc_hbm[r0 : r0 + nr, c0:c1])
        j = nl.ndarray((PT, read_len), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=j[0:nr, 0:w], pattern=[[1, w]], offset=c0, channel_multiplier=0)
        jf = nl.ndarray((PT, read_len), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=jf[0:nr, 0:w], src=j[0:nr, 0:w])
        ci = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ci[0:nr, :], src=clen_hbm[r0 : r0 + nr, 0:1])
        cf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cf[0:nr, :], src=ci[0:nr, :])
        # live = (j < clen) ; s = s * live + (live - 1) * (-NEG_BOUND)  -> NEG_BOUND where dead
        live = nl.ndarray((PT, read_len), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=live[0:nr, 0:w], data=jf[0:nr, 0:w], op0=nl.less, operand0=cf[0:nr, 0:1])
        nisa.tensor_tensor(dst=s[0:nr, 0:w], data1=s[0:nr, 0:w], data2=live[0:nr, 0:w], op=nl.multiply)
        dead = nl.ndarray((PT, read_len), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=dead[0:nr, 0:w], data=live[0:nr, 0:w], op0=nl.add, operand0=-1.0,
                           op1=nl.multiply, operand1=-NEG_BOUND)
        nisa.tensor_tensor(dst=s[0:nr, 0:w], data1=s[0:nr, 0:w], data2=dead[0:nr, 0:w], op=nl.add)
        s16 = nl.ndarray((PT, read_len), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=s16[0:nr, 0:w], src=s[0:nr, 0:w])
        nisa.dma_copy(dst=sc16_hbm[r0 : r0 + nr, c0:c1], src=s16[0:nr, 0:w])
    nisa.core_barrier(data=sc16_hbm, cores=(0, 1))
    # 2. top-k, 8 tokens per instruction
    ngroups = cdiv(T, 8)
    for g in range(ngroups):
        if g % 2 != pid:
            continue
        rows = min(8, T - g * 8)
        snake = nl.ndarray((PT, n16), dtype=nl.bfloat16, buffer=nl.sbuf)
        if rows < 8:
            nisa.memset(snake, NEG_BOUND)
        if IDX_SNAKE_PE:
            # snake[16q + r, c] = sc16[g*8 + q, 16c + r]: per 128 columns c, the group's rows
            # as [c, (q, r)] (32-byte runs) by one DMA, then one PE transpose (exact for bf16)
            for c0 in range(0, n16, PT):
                pc = min(PT, n16 - c0)
                m = nl.ndarray((PT, 8 * 16), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=m.ap(pattern=[[8 * 16, pc], [16, rows], [1, 16]], offset=0),
                              src=sc16_hbm.ap(pattern=[[16, pc], [read_len, rows], [1, 16]],
                                              offset=g * 8 * read_len + c0 * 16))
                tp = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tp[0 : rows * 16, 0:pc], data=m[0:pc, 0 : rows * 16])
                nisa.tensor_copy(dst=snake[0 : rows * 16, c0 : c0 + pc], src=tp[0 : rows * 16, 0:pc])
        else:
            for q in range(rows):
                nisa.dma_copy(dst=snake[16 * q : 16 * q + 16, 0:n16],
                              src=sc16_hbm.ap(pattern=[[1, 16], [16, n16]],
                                              offset=(g * 8 + q) * read_len))
        vals = nl.ndarray((PT, topk), dtype=nl.bfloat16, buffer=nl.sbuf)
        idx = nl.ndarray((PT, topk), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.topk(val_dst=vals, idx_dst=idx, src=snake, n=read_len, sorted=False)
        if IDX_TOPK_OUT_PE and topk % 16 == 0 and k16 <= PT:
            # oi[c, 16q + r] = idx[16q + r, c] = token q's pick 16c + r: 64-byte runs
            idf = nl.ndarray((PT, k16), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=idf[0 : rows * 16, :], src=idx[0 : rows * 16, 0:k16])
            tpo = nl.ndarray((PT, PT), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=tpo[0:k16, 0 : rows * 16], data=idf[0 : rows * 16, 0:k16])
            oi = nl.ndarray((PT, PT), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=oi[0:k16, 0 : rows * 16], src=tpo[0:k16, 0 : rows * 16])
            nisa.dma_copy(dst=topk_hbm.ap(pattern=[[16, k16], [topk, rows], [1, 16]], offset=g * 8 * topk),
                          src=oi.ap(pattern=[[PT, k16], [16, rows], [1, 16]], offset=0))
        else:
            for q in range(rows):
                nisa.dma_copy(dst=topk_hbm.ap(pattern=[[1, 16], [16, k16]], offset=(g * 8 + q) * topk),
                              src=idx.view(nl.int32)[16 * q : 16 * q + 16, 0:k16])
    nisa.core_barrier(data=topk_hbm, cores=(0, 1))
    # 3. picks beyond the row's compress_len are empty
    for r0, nr, c0, c1 in work3:
        w = c1 - c0
        if w <= 0:
            continue
        ti = nl.ndarray((PT, topk), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ti[0:nr, 0:w], src=topk_hbm[r0 : r0 + nr, c0:c1])
        tf = nl.ndarray((PT, topk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tf[0:nr, 0:w], src=ti[0:nr, 0:w])
        ci = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ci[0:nr, :], src=clen_hbm[r0 : r0 + nr, 0:1])
        cf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cf[0:nr, :], src=ci[0:nr, :])
        live = nl.ndarray((PT, topk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=live[0:nr, 0:w], data=tf[0:nr, 0:w], op0=nl.less, operand0=cf[0:nr, 0:1])
        nisa.tensor_tensor(dst=tf[0:nr, 0:w], data1=tf[0:nr, 0:w], data2=live[0:nr, 0:w], op=nl.multiply)
        lm1 = nl.ndarray((PT, topk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=lm1[0:nr, 0:w], data=live[0:nr, 0:w], op0=nl.add, operand0=-1.0)
        nisa.tensor_tensor(dst=tf[0:nr, 0:w], data1=tf[0:nr, 0:w], data2=lm1[0:nr, 0:w], op=nl.add)
        nisa.tensor_copy(dst=ti[0:nr, 0:w], src=tf[0:nr, 0:w])
        nisa.dma_copy(dst=topk_hbm[r0 : r0 + nr, c0:c1], src=ti[0:nr, 0:w])
    nisa.core_barrier(data=topk_hbm, cores=(0, 1))
