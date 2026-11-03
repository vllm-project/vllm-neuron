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


#: `index_scores_span` scores `G = 128 // nh` tokens a pass (a static loop over the token
#: groups) instead of one token a dynamic iteration: the first matmul's stationary holds G
#: tokens' `nh` head queries side by side (every output element the same dot product), the
#: ReLU runs over the G * nh partitions, and the head-weighted sum is ONE fp32 matmul
#: against a block-diagonal weight `[G * nh, G]` (token t's `nh` weights in column t, +0.0
#: elsewhere: each token's sum is its own terms, in the same partition order, plus exact
#: zeros). With the nh = 4 index heads a rank the per-token loop ran each key block on 4 of
#: the PE's 128 columns. (The fused decode launches' indexer, `tp64/fused/*/r1_idx_decode`;
#: here the block-diagonal weight is one DMA of the group's weights times a 0/1 mask.)
IDX_BATCH = True
#: The batched pass's key blocks: IDX_NT score columns (a shorter last one) whatever
#: read_len % 512 (each column independent).
IDX_NT = 512


def index_scores_span(qi_hbm, wi_hbm, ktab_hbm, sc_hbm, tok0, ntok, nh, read_len, kd=128):
    """Index scores for `ntok` tokens from row `tok0` of one span (a dynamic loop, or
    `IDX_BATCH`'s static one over groups of tokens):
    `sc[t, j] = sum_h relu(q[t, h] . k[j]) * w[t, h]`, fp32. `ktab_hbm[read_len, kd]`
    (bf16) is the span's index-key table; `qi_hbm[Tp, nh*kd]` the fp4-rounded, roped
    queries; `wi_hbm[Tp, nh]` the per-head weights (already scaled)."""
    nb = read_len // 512 if read_len % 512 == 0 else None
    nt = 512 if nb else 128
    nb = read_len // nt
    kT = nl.ndarray((PT, read_len), dtype=nl.bfloat16, buffer=nl.sbuf)
    for jb in range(read_len // PT):
        nisa.dma_transpose(dst=kT[0:PT, jb * PT : (jb + 1) * PT], src=ktab_hbm[jb * PT : (jb + 1) * PT, 0:kd])
    hp = -(-nh // 16) * 16
    G = min(ntok, PT // nh) if IDX_BATCH and 2 * nh <= PT else 1
    if G > 1:
        # mask[p, t] = 1.0 where partition p holds one of token t's heads (p - nh t in [0, nh))
        rel = nl.ndarray((PT, G), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=rel, pattern=[[-nh, G]], offset=0, channel_multiplier=1)
        relf = nl.ndarray((PT, G), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=relf, src=rel)
        ge = nl.ndarray((PT, G), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ge, data=relf, op0=nl.greater_equal, operand0=0.0)
        lt = nl.ndarray((PT, G), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=lt, data=relf, op0=nl.less, operand0=float(nh))
        mask = nl.ndarray((PT, G), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=mask, data1=ge, data2=lt, op=nl.multiply)
        for t0 in range(0, ntok, G):
            g = min(G, ntok - t0)
            m = g * nh
            r0 = tok0 + t0
            qrow = nl.ndarray((PT, kd), dtype=nl.bfloat16, buffer=nl.sbuf)
            # partition t * nh + h = token r0 + t's head h (contiguous rows of qi)
            nisa.dma_copy(dst=qrow[0:m, :], src=qi_hbm.ap(pattern=[[kd, m], [1, kd]], offset=r0 * nh * kd))
            qps = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=qps[0:PT, 0:m], data=qrow[0:m, 0:kd])
            qt = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=qt[0:PT, 0:m], src=qps[0:PT, 0:m])
            # wb[t * nh + h, t] = w[r0 + t, h], +0.0 elsewhere (the mask times the weight
            # column, plus 0.0: an off-diagonal -0.0 becomes +0.0)
            wcol = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=wcol[0:m, :], src=wi_hbm.ap(pattern=[[1, m], [1, 1]], offset=r0 * nh))
            wb = nl.ndarray((PT, G), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=wb[0:m, 0:g], data=mask[0:m, 0:g], op0=nl.multiply, operand0=wcol[0:m, 0:1],
                               op1=nl.add, operand1=0.0)
            row = nl.ndarray((G, read_len), dtype=nl.float32, buffer=nl.sbuf)
            for c0 in range(0, read_len, IDX_NT):
                w_ = min(IDX_NT, read_len - c0)
                acc = nl.ndarray((PT, IDX_NT), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=acc[0:m, 0:w_], stationary=qt[0:PT, 0:m], moving=kT[0:PT, c0 : c0 + w_])
                rect = nl.ndarray((PT, IDX_NT), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=rect[0:m, 0:w_], data=acc[0:m, 0:w_], op0=nl.maximum, operand0=0.0)
                wsum = nl.ndarray((G, IDX_NT), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=wsum[0:g, 0:w_], stationary=wb[0:m, 0:g], moving=rect[0:m, 0:w_])
                nisa.tensor_copy(dst=row[0:g, c0 : c0 + w_], src=wsum[0:g, 0:w_])
            nisa.dma_copy(dst=sc_hbm[r0 : r0 + g, 0:read_len], src=row[0:g, :])
        return

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
        row = nl.ndarray((1, read_len), dtype=nl.float32, buffer=nl.sbuf)
        for b in range(nb):
            acc = nl.ndarray((hp, nt), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=acc[0:nh, :], stationary=qt[0:PT, 0:nh], moving=kT[0:PT, b * nt : (b + 1) * nt])
            rect = nl.ndarray((hp, nt), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rect[0:nh, :], data=acc[0:nh, :], op0=nl.maximum, operand0=0.0)
            wsum = nl.ndarray((1, nt), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=wsum, stationary=wt[0:nh, 0:1], moving=rect[0:nh, 0:nt])
            nisa.tensor_copy(dst=row[0:1, b * nt : (b + 1) * nt], src=wsum)
        nisa.dma_copy(dst=sc_hbm.ap(pattern=[[read_len, 1], [1, read_len]], offset=tok0 * read_len,
                                    scalar_offset=i, indirect_dim=0), src=row)

    nl.fori_loop(0, ntok, body)


#: `index_topk` builds each 8-token group's snake tile (`snake[16 q + i, k] =
#: sc16[t_q, 16 k + i]`) from contiguous row loads and PE transposes instead of 2-byte
#: element-strided DMAs (~14 us a token at read_len 8192): `A[p, q, 16 j + i] =
#: sc16[t_q, 16 (J p + j) + i]` (J = read_len / 2048; 32 J contiguous bytes a partition),
#: then for each j the [128 p, 8 q x 16 i] slice transposed lands in snake columns
#: J p + j. The same bf16 values (pure data movement); needs read_len % 2048 == 0.
TOPK_SNAKE_PE = True
#: The same for any read_len % 16 == 0 (e.g. 8448, 1280): the snake's last n16 % 128 columns
#: from one more transpose (`index_topk`).
TOPK_SNAKE_PE_ANY = True
#: `index_topk`'s picks stored as 64-byte runs (the uint32 index tile converted to fp32 and
#: PE-transposed) instead of 4-byte element-strided DMAs.
TOPK_STORE_T = True


#: `index_topk`'s mask + bf16 step at T <= 128 (one row tile, else core 0 alone on T
#: partitions): the rows halved between the cores and each row spread over S partitions
#: (`_topk_mask_spread`; the same per-element ops).
TOPK_MASK_SPREAD = True


def _topk_mask_spread(sc_hbm, clen_hbm, sc16_hbm, T, Tp, read_len, pid):
    """`index_topk` step 1 for `T <= 128` rows: core `pid` takes rows `[t0, t1)` (`T`
    halved), row `t`'s columns `q * w + [0, w)` on partition `(t - t0) * S + q`
    (`w = read_len / S`: the row-major HBM rows read as `[T * S, w]`). The mask compares
    the local column with `clen[t] - q * w` (integers, exact in fp32: the same
    `j < clen[t]`), that column built on `nr` partitions and spread by an HBM round trip."""
    half = (T + 1) // 2
    t0, t1 = (0, half) if pid == 0 else (half, T)
    nr = t1 - t0
    if nr <= 0:
        return
    S = 1
    while nr * S * 2 <= PT and read_len % (S * 2) == 0:
        S *= 2
    w = read_len // S
    np_ = nr * S
    s16v = sc16_hbm.reshape((sc16_hbm.shape[0] * S, w))
    s = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
    if isinstance(sc_hbm, (tuple, list)):
        # the scores as two column halves (`[T, read_len / 2]` each): row t's partitions
        # [t S, t S + S / 2) from the first, the rest from the second (the same columns q w)
        assert S % 2 == 0
        H = S // 2
        for tl in range(nr):
            for i in range(2):
                hv = sc_hbm[i].reshape((sc_hbm[i].shape[0] * H, w))
                p0 = tl * S + i * H
                nisa.dma_copy(dst=s[p0 : p0 + H, :], src=hv[(t0 + tl) * H : (t0 + tl + 1) * H, 0:w])
    else:
        scv = sc_hbm.reshape((sc_hbm.shape[0] * S, w))
        nisa.dma_copy(dst=s[0:np_, :], src=scv[t0 * S : t1 * S, 0:w])
    ci = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ci[0:nr, :], src=clen_hbm[t0:t1, 0:1])
    cf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cf[0:nr, :], src=ci[0:nr, :])
    # cfs[t, q] = clen[t] - q * w
    qo = nl.ndarray((PT, S), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=qo[0:nr, :], pattern=[[w, S]], offset=0, channel_multiplier=0)
    qof = nl.ndarray((PT, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qof[0:nr, :], src=qo[0:nr, :])
    cfs = nl.ndarray((PT, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=cfs[0:nr, :], data=qof[0:nr, :], op0=nl.subtract, operand0=cf[0:nr, 0:1],
                       op1=nl.multiply, operand1=-1.0)
    cfh = nl.ndarray((PT * S, 1), dtype=nl.float32, buffer=nl.private_hbm)
    nisa.dma_copy(dst=cfh.reshape((PT, S))[0:nr, 0:S], src=cfs[0:nr, 0:S])
    cfp = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cfp[0:np_, :], src=cfh[0:np_, 0:1])
    j = nl.ndarray((PT, w), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=j[0:np_, :], pattern=[[1, w]], offset=0, channel_multiplier=0)
    jf = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=jf[0:np_, :], src=j[0:np_, :])
    live = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=live[0:np_, :], data=jf[0:np_, :], op0=nl.less, operand0=cfp[0:np_, 0:1])
    nisa.tensor_tensor(dst=s[0:np_, :], data1=s[0:np_, :], data2=live[0:np_, :], op=nl.multiply)
    dead = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=dead[0:np_, :], data=live[0:np_, :], op0=nl.add, operand0=-1.0,
                       op1=nl.multiply, operand1=-NEG_BOUND)
    nisa.tensor_tensor(dst=s[0:np_, :], data1=s[0:np_, :], data2=dead[0:np_, :], op=nl.add)
    s16 = nl.ndarray((PT, w), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=s16[0:np_, :], src=s[0:np_, :])
    nisa.dma_copy(dst=s16v[t0 * S : t1 * S, 0:w], src=s16[0:np_, :])


def index_topk(sc_hbm, clen_hbm, sc16_hbm, topk_hbm, T, Tp, read_len, topk, pid):
    """Mask `sc[t, j]` for `j >= clen[t]` to NEG_BOUND, round to bf16, select the top
    `topk` positions per token on the GpSimd engine (16-partition snake layout) and
    write `topk_hbm[t, 0:topk]` (int32, -1 where the pick lies beyond `clen[t]`).
    Rows beyond `T` are left alone. Token groups of 8 alternate between the cores."""
    n16 = read_len // 16
    k16 = cdiv(topk, 16)
    # 1. mask + bf16, by row tile (this core's tiles)
    tiles = [(i * PT, min(PT, T - i * PT)) for i in range(cdiv(T, PT)) if i % 2 == pid]
    spread = TOPK_MASK_SPREAD and T <= PT
    if spread:
        _topk_mask_spread(sc_hbm, clen_hbm, sc16_hbm, T, Tp, read_len, pid)
    for r0, nr in ([] if spread else tiles):
        s = nl.ndarray((PT, read_len), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=s[0:nr, :], src=sc_hbm[r0 : r0 + nr, 0:read_len])
        j = nl.ndarray((PT, read_len), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=j[0:nr, :], pattern=[[1, read_len]], offset=0, channel_multiplier=0)
        jf = nl.ndarray((PT, read_len), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=jf[0:nr, :], src=j[0:nr, :])
        ci = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ci[0:nr, :], src=clen_hbm[r0 : r0 + nr, 0:1])
        cf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cf[0:nr, :], src=ci[0:nr, :])
        # live = (j < clen) ; s = s * live + (live - 1) * (-NEG_BOUND)  -> NEG_BOUND where dead
        live = nl.ndarray((PT, read_len), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=live[0:nr, :], data=jf[0:nr, :], op0=nl.less, operand0=cf[0:nr, 0:1])
        nisa.tensor_tensor(dst=s[0:nr, :], data1=s[0:nr, :], data2=live[0:nr, :], op=nl.multiply)
        dead = nl.ndarray((PT, read_len), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=dead[0:nr, :], data=live[0:nr, :], op0=nl.add, operand0=-1.0,
                           op1=nl.multiply, operand1=-NEG_BOUND)
        nisa.tensor_tensor(dst=s[0:nr, :], data1=s[0:nr, :], data2=dead[0:nr, :], op=nl.add)
        s16 = nl.ndarray((PT, read_len), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=s16[0:nr, :], src=s[0:nr, :])
        nisa.dma_copy(dst=sc16_hbm[r0 : r0 + nr, 0:read_len], src=s16[0:nr, :])
    nisa.core_barrier(data=sc16_hbm, cores=(0, 1))
    # 2. top-k, 8 tokens per instruction
    ngroups = cdiv(T, 8)
    for g in range(ngroups):
        if g % 2 != pid:
            continue
        rows = min(8, T - g * 8)
        if TOPK_SNAKE_PE and n16 % PT == 0:
            J = n16 // PT
            A = nl.ndarray((PT, 8, 16 * J), dtype=nl.bfloat16, buffer=nl.sbuf)
            if rows < 8:
                nisa.memset(A, NEG_BOUND)
            for q in range(rows):
                nisa.dma_copy(dst=A[0:PT, q, :],
                              src=sc16_hbm.ap(pattern=[[16 * J, PT], [1, 16 * J]],
                                              offset=(g * 8 + q) * read_len))
            snake3 = nl.ndarray((PT, PT, J), dtype=nl.bfloat16, buffer=nl.sbuf)
            for j in range(J):
                X = nl.ndarray((PT, 8, 16), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=X, src=A[0:PT, :, 16 * j : 16 * (j + 1)])
                tps = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tps[0:PT, 0:PT], data=X.reshape((PT, PT)))
                nisa.tensor_copy(dst=snake3[0:PT, :, j], src=tps[0:PT, 0:PT])
            snake = snake3.reshape((PT, n16))
        elif TOPK_SNAKE_PE_ANY and read_len % 16 == 0:
            # n16 = 128 J + R: the snake's first 128 J columns as above (column J p + j of
            # transpose j), its last R from one more transpose of B[k, 16 q + i] =
            # sc16[t_q, 16 (128 J + k) + i] (R partitions, 32 contiguous bytes each)
            J = n16 // PT
            R = n16 - J * PT
            snake = nl.ndarray((PT, n16), dtype=nl.bfloat16, buffer=nl.sbuf)
            if J:
                A = nl.ndarray((PT, 8, 16 * J), dtype=nl.bfloat16, buffer=nl.sbuf)
                if rows < 8:
                    nisa.memset(A, NEG_BOUND)
                for q in range(rows):
                    nisa.dma_copy(dst=A[0:PT, q, :],
                                  src=sc16_hbm.ap(pattern=[[16 * J, PT], [1, 16 * J]],
                                                  offset=(g * 8 + q) * read_len))
                for j in range(J):
                    X = nl.ndarray((PT, 8, 16), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=X, src=A[0:PT, :, 16 * j : 16 * (j + 1)])
                    tps = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tps[0:PT, 0:PT], data=X.reshape((PT, PT)))
                    nisa.tensor_copy(dst=snake.ap(pattern=[[n16, PT], [J, PT]], offset=j),
                                     src=tps[0:PT, 0:PT])
            if R:
                B = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
                if rows < 8:
                    nisa.memset(B, NEG_BOUND)
                for q in range(rows):
                    nisa.dma_copy(dst=B[0:R, 16 * q : 16 * (q + 1)],
                                  src=sc16_hbm.ap(pattern=[[16, R], [1, 16]],
                                                  offset=(g * 8 + q) * read_len + 16 * PT * J))
                tpr = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tpr[0:PT, 0:R], data=B[0:R, 0:PT])
                nisa.tensor_copy(dst=snake[0:PT, PT * J : PT * J + R], src=tpr[0:PT, 0:R])
        else:
            snake = nl.ndarray((PT, n16), dtype=nl.bfloat16, buffer=nl.sbuf)
            if rows < 8:
                nisa.memset(snake, NEG_BOUND)
            for q in range(rows):
                nisa.dma_copy(dst=snake[16 * q : 16 * q + 16, 0:n16],
                              src=sc16_hbm.ap(pattern=[[1, 16], [16, n16]], offset=(g * 8 + q) * read_len))
        vals = nl.ndarray((PT, topk), dtype=nl.bfloat16, buffer=nl.sbuf)
        idx = nl.ndarray((PT, topk), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.topk(val_dst=vals, idx_dst=idx, src=snake, n=read_len, sorted=False)
        if TOPK_STORE_T and topk == 16 * k16 and k16 <= PT:
            # idx[16 q + i, k] -> fp32 (exact, < 2^24) -> PE transpose -> int32 ti[k, 16 q + i]:
            # row t_q's picks 16 k + i as k16 partitions of 16 contiguous values, one DMA
            idf = nl.ndarray((PT, k16), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=idf, src=idx[0:PT, 0:k16])
            tpi = nl.ndarray((PT, PT), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=tpi[0:k16, 0:PT], data=idf[0:PT, 0:k16])
            ti2 = nl.ndarray((PT, PT), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ti2[0:k16, 0:PT], src=tpi[0:k16, 0:PT])
            nisa.dma_copy(dst=topk_hbm.ap(pattern=[[16, k16], [topk, rows], [1, 16]], offset=g * 8 * topk),
                          src=ti2[0:k16, 0 : 16 * rows])
        else:
            for q in range(rows):
                nisa.dma_copy(dst=topk_hbm.ap(pattern=[[1, 16], [16, k16]], offset=(g * 8 + q) * topk),
                              src=idx.view(nl.int32)[16 * q : 16 * q + 16, 0:k16])
    nisa.core_barrier(data=topk_hbm, cores=(0, 1))
    # 3. picks beyond the row's compress_len are empty
    for r0, nr in tiles:
        ti = nl.ndarray((PT, topk), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ti[0:nr, :], src=topk_hbm[r0 : r0 + nr, 0:topk])
        tf = nl.ndarray((PT, topk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tf[0:nr, :], src=ti[0:nr, :])
        ci = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ci[0:nr, :], src=clen_hbm[r0 : r0 + nr, 0:1])
        cf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cf[0:nr, :], src=ci[0:nr, :])
        live = nl.ndarray((PT, topk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=live[0:nr, :], data=tf[0:nr, :], op0=nl.less, operand0=cf[0:nr, 0:1])
        nisa.tensor_tensor(dst=tf[0:nr, :], data1=tf[0:nr, :], data2=live[0:nr, :], op=nl.multiply)
        lm1 = nl.ndarray((PT, topk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=lm1[0:nr, :], data=live[0:nr, :], op0=nl.add, operand0=-1.0)
        nisa.tensor_tensor(dst=tf[0:nr, :], data1=tf[0:nr, :], data2=lm1[0:nr, :], op=nl.add)
        nisa.tensor_copy(dst=ti[0:nr, :], src=tf[0:nr, :])
        nisa.dma_copy(dst=topk_hbm[r0 : r0 + nr, 0:topk], src=ti[0:nr, :])
    nisa.core_barrier(data=topk_hbm, cores=(0, 1))
