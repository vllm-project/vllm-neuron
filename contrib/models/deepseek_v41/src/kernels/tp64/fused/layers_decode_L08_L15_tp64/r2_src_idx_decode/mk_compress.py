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


#: `index_scores_span` scores up to 512 keys per matmul pair (else 128 unless 512 divides
#: the span's keys).
INDEX_SCORE_WIDE = True
#: `index_topk` stores the masked bf16 scores snake-major (see there): the top-k's snake
#: tile loads with one contiguous-run DMA per 8 tokens instead of 8 two-byte-strided ones.
TOPK_SNAKE_ROWS = True


def index_scores_span(qi_hbm, wi_hbm, ktab_hbm, sc_hbm, tok0, ntok, nh, read_len, kd=128, ktab_full=None,
                      krow0=0):
    """Index scores for `ntok` tokens from row `tok0` of one span (a dynamic loop):
    `sc[t, j] = sum_h relu(q[t, h] . k[j]) * w[t, h]`, fp32. `ktab_hbm[read_len, kd]`
    (bf16) is the span's index-key table; `qi_hbm[Tp, nh*kd]` the fp4-rounded, roped
    queries; `wi_hbm[Tp, nh]` the per-head weights (already scaled)."""
    if INDEX_SCORE_WIDE:
        # blocks of up to 512 score columns (one PSUM bank), the last one shorter: each
        # column's two contractions do not depend on its block (4224 keys: 9 blocks, not 33)
        n128 = (read_len // PT) * PT
        blocks = [(b0, min(512, n128 - b0)) for b0 in range(0, n128, 512)]
    else:
        nb = read_len // 512 if read_len % 512 == 0 else None
        nt = 512 if nb else 128
        nb = read_len // nt
        blocks = [(b * nt, nt) for b in range(nb)]
    ntw = max(w for _, w in blocks)
    kT = nl.ndarray((PT, read_len), dtype=nl.bfloat16, buffer=nl.sbuf)
    nkb = read_len // PT
    if INDEX_KT_PE and ktab_full is not None and kd == PT:
        # the key tiles by plain row DMAs (4 tiles a DMA) and PE transposes (exact data
        # movement) into one PSUM bank, one copy out per 4 tiles, instead of one DMA
        # transpose per tile (2-byte-element DMA transposes, slow and contending with the
        # slot table's gathers for the DMA rings)
        for g0 in range(0, nkb, 4):
            ng = min(4, nkb - g0)
            kr = nl.ndarray((PT, 4, kd), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=kr[0:PT, 0:ng, 0:kd],
                          src=ktab_full.ap(pattern=[[kd, PT], [PT * kd, ng], [1, kd]], offset=(krow0 + g0 * PT) * kd))
            tp = nl.ndarray((PT, 4 * PT), dtype=nl.bfloat16, buffer=nl.psum)
            for m in range(ng):
                nisa.nc_transpose(dst=tp[0:PT, m * PT : (m + 1) * PT], data=kr[0:PT, m, 0:kd])
            nisa.tensor_copy(dst=kT[0:PT, g0 * PT : (g0 + ng) * PT], src=tp[0:PT, 0 : ng * PT],
                             engine=nisa.engine.scalar)
    else:
        for jb in range(nkb):
            nisa.dma_transpose(dst=kT[0:PT, jb * PT : (jb + 1) * PT], src=ktab_hbm[jb * PT : (jb + 1) * PT, 0:kd])
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
        row = nl.ndarray((1, read_len), dtype=nl.float32, buffer=nl.sbuf)
        for b0, nt in blocks:
            acc = nl.ndarray((hp, ntw), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=acc[0:nh, 0:nt], stationary=qt[0:PT, 0:nh], moving=kT[0:PT, b0 : b0 + nt])
            rect = nl.ndarray((hp, ntw), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rect[0:nh, 0:nt], data=acc[0:nh, 0:nt], op0=nl.maximum, operand0=0.0)
            wsum = nl.ndarray((1, ntw), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=wsum[0:1, 0:nt], stationary=wt[0:nh, 0:1], moving=rect[0:nh, 0:nt])
            nisa.tensor_copy(dst=row[0:1, b0 : b0 + nt], src=wsum[0:1, 0:nt])
        nisa.dma_copy(dst=sc_hbm.ap(pattern=[[read_len, 1], [1, read_len]], offset=tok0 * read_len,
                                    scalar_offset=i, indirect_dim=0), src=row)

    if ntok > INDEX_SCORE_STATIC_MAX:
        nl.fori_loop(0, ntok, body)
        return
    # few tokens (decode): a static loop, the tokens' chains interleaved block by block (each
    # token's instructions unchanged) instead of one token's serial PE -> DVE -> PE chain at
    # a time in the dynamic loop
    if INDEX_SCORE_PAIR and 2 * nh <= PT and kd == PT and ntok > 1:
        _index_scores_pairs(qi_hbm, wi_hbm, kT, sc_hbm, tok0, ntok, nh, read_len, kd, blocks, ntw)
        return
    qts, wts, rows = [], [], []
    for i in range(ntok):
        t = tok0 + i
        qrow = nl.ndarray((hp, kd), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=qrow[0:nh, :], src=qi_hbm.ap(pattern=[[kd, nh], [1, kd]], offset=t * nh * kd))
        qps = nl.ndarray((PT, hp), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=qps[0:PT, 0:nh], data=qrow[0:nh, 0:kd])
        qt = nl.ndarray((PT, hp), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qt[0:PT, 0:nh], src=qps[0:PT, 0:nh])
        wt = nl.ndarray((hp, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt[0:nh, :], src=wi_hbm.ap(pattern=[[1, nh], [1, 1]], offset=t * nh))
        qts.append(qt)
        wts.append(wt)
        rows.append(nl.ndarray((1, read_len), dtype=nl.float32, buffer=nl.sbuf))
    for b0, nt in blocks:
        accs = []
        for i in range(ntok):
            acc = nl.ndarray((hp, ntw), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=acc[0:nh, 0:nt], stationary=qts[i][0:PT, 0:nh], moving=kT[0:PT, b0 : b0 + nt])
            accs.append(acc)
        rects = []
        for i in range(ntok):
            rect = nl.ndarray((hp, ntw), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rect[0:nh, 0:nt], data=accs[i][0:nh, 0:nt], op0=nl.maximum, operand0=0.0)
            rects.append(rect)
        for i in range(ntok):
            wsum = nl.ndarray((1, ntw), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=wsum[0:1, 0:nt], stationary=wts[i][0:nh, 0:1], moving=rects[i][0:nh, 0:nt])
            nisa.tensor_copy(dst=rows[i][0:1, b0 : b0 + nt], src=wsum[0:1, 0:nt])
    for i in range(ntok):
        nisa.dma_copy(dst=sc_hbm.ap(pattern=[[read_len, 1], [1, read_len]], offset=(tok0 + i) * read_len),
                      src=rows[i])


#: `index_scores_span` with at most this many tokens runs them in a static loop, interleaved.
INDEX_SCORE_STATIC_MAX = 4
#: `index_scores_span` builds the transposed key table by row DMAs + PE transposes (off:
#: no gain measured, in8192_bs10 exec - cc 6.81 -> 6.83 ms; the scores then waited elsewhere).
INDEX_KT_PE = False
#: The static loop's tokens several at a time (`_index_scores_pairs`: as many as fit
#: 128 / heads) when 2 * heads <= 128.
INDEX_SCORE_PAIR = True


def _index_scores_pairs(qi_hbm, wi_hbm, kT, sc_hbm, tok0, ntok, nh, read_len, kd, blocks, ntw):
    """`index_scores_span`'s static loop with the tokens taken G = 128 / heads at a time:
    the group's queries side by side in one stationary (the score matmul's output rows are
    the tokens' heads: each row's contraction over the same 128 key-dim rows, unchanged),
    one relu over all of them, and the head-weight matmul against a block-diagonal
    stationary (each token's column sums its own heads' rows; the other tokens' rows add
    exact zeros), so each PE / DVE instruction serves G tokens."""
    G = max(1, min(ntok, PT // nh))
    groups = [(i0, min(G, ntok - i0)) for i0 in range(0, ntok, G)]
    qts, wts, rows = [], [], []
    for i0, g in groups:
        t = tok0 + i0
        qrow = nl.ndarray((PT, kd), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=qrow[0 : g * nh, :], src=qi_hbm.ap(pattern=[[kd, g * nh], [1, kd]], offset=t * nh * kd))
        qps = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=qps[0:PT, 0 : g * nh], data=qrow[0 : g * nh, 0:kd])
        qt = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qt[0:PT, 0 : g * nh], src=qps[0:PT, 0 : g * nh])
        wt = nl.ndarray((PT, G), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(wt, 0.0)
        for k in range(g):
            nisa.dma_copy(dst=wt[k * nh : (k + 1) * nh, k : k + 1],
                          src=wi_hbm.ap(pattern=[[1, nh], [1, 1]], offset=(t + k) * nh))
        qts.append(qt)
        wts.append(wt)
        rows.append(nl.ndarray((G, read_len), dtype=nl.float32, buffer=nl.sbuf))
    for b0, nt in blocks:
        accs = []
        for (i0, g), qt in zip(groups, qts):
            acc = nl.ndarray((PT, ntw), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=acc[0 : g * nh, 0:nt], stationary=qt[0:PT, 0 : g * nh], moving=kT[0:PT, b0 : b0 + nt])
            accs.append(acc)
        rects = []
        for (i0, g), acc in zip(groups, accs):
            rect = nl.ndarray((PT, ntw), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rect[0 : g * nh, 0:nt], data=acc[0 : g * nh, 0:nt], op0=nl.maximum, operand0=0.0)
            rects.append(rect)
        for (i0, g), wt, rect, row in zip(groups, wts, rects, rows):
            wsum = nl.ndarray((G, ntw), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=wsum[0:g, 0:nt], stationary=wt[0 : g * nh, 0:g], moving=rect[0 : g * nh, 0:nt])
            nisa.tensor_copy(dst=row[0:g, b0 : b0 + nt], src=wsum[0:g, 0:nt])
    for (i0, g), row in zip(groups, rows):
        nisa.dma_copy(dst=sc_hbm.ap(pattern=[[read_len, g], [1, read_len]], offset=(tok0 + i0) * read_len),
                      src=row[0:g, :])


#: The index scores split by KEYS across the cores (`index_scores_keys`): each core gathers
#: its half of a span's key tiles straight from the index-key cache into SBUF and scores
#: every token of the span on them (no `ktab` HBM table, store, cross-core barrier or DMA
#: transposes; the gathers no longer wait behind the compressor's stores on the HWDGE queues).
INDEX_KEY_SPLIT = True


def index_scores_keys(qi_hbm, wi_hbm, kcache, kslots, ks0, sc_hbm, tok0, ntok, nh, read_len, pid, kd=128):
    """`index_scores_span`'s scores for ALL `ntok` tokens of a span from row `tok0`, over this
    core's contiguous half of the span's 128-key tiles (core 0 the first ceil(n/2)): the
    keys `kcache[kslots[ks0 + j]]` gathered by tile (the per-tile indirect DMA of
    `gather_rows`), PE-transposed (exact data movement) into `kT`, then the token groups of
    `_index_scores_pairs` (each score column's contractions unchanged) over 512-column blocks;
    each token row's slice `sc[t, lo : hi]` stored once."""
    nfull = read_len // PT
    a = (nfull + 1) // 2
    t_lo, t_hi = (0, a) if pid == 0 else (a, nfull)
    nloc = t_hi - t_lo
    if nloc <= 0:
        return
    ncol = nloc * PT
    c0 = t_lo * PT
    ix = nl.ndarray((PT, nloc), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ix, src=kslots.ap(pattern=[[1, PT], [PT, nloc]], offset=ks0 + c0),
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    kT = nl.ndarray((PT, ncol), dtype=nl.bfloat16, buffer=nl.sbuf)
    for g0 in range(0, nloc, 4):
        ng = min(4, nloc - g0)
        tp = nl.ndarray((PT, 4 * PT), dtype=nl.bfloat16, buffer=nl.psum)
        for m in range(ng):
            kr = nl.ndarray((PT, kd), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=kr, src=kcache.ap(pattern=[[kd, PT], [1, kd]], vector_offset=ix[0:PT, g0 + m : g0 + m + 1],
                                                indirect_dim=0))
            nisa.nc_transpose(dst=tp[0:PT, m * PT : (m + 1) * PT], data=kr[0:PT, 0:kd])
        nisa.tensor_copy(dst=kT[0:PT, g0 * PT : (g0 + ng) * PT], src=tp[0:PT, 0 : ng * PT],
                         engine=nisa.engine.scalar)
    blocks = [(b0, min(512, ncol - b0)) for b0 in range(0, ncol, 512)]
    ntw = max(w for _, w in blocks)
    G = max(1, min(ntok, PT // nh))
    groups = [(i0, min(G, ntok - i0)) for i0 in range(0, ntok, G)]
    qts, wts, rows = [], [], []
    for i0, g in groups:
        t = tok0 + i0
        qrow = nl.ndarray((PT, kd), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=qrow[0 : g * nh, :], src=qi_hbm.ap(pattern=[[kd, g * nh], [1, kd]], offset=t * nh * kd))
        qps = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=qps[0:PT, 0 : g * nh], data=qrow[0 : g * nh, 0:kd])
        qt = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qt[0:PT, 0 : g * nh], src=qps[0:PT, 0 : g * nh])
        wt = nl.ndarray((PT, G), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(wt, 0.0)
        for k in range(g):
            nisa.dma_copy(dst=wt[k * nh : (k + 1) * nh, k : k + 1],
                          src=wi_hbm.ap(pattern=[[1, nh], [1, 1]], offset=(t + k) * nh))
        qts.append(qt)
        wts.append(wt)
        rows.append(nl.ndarray((G, ncol), dtype=nl.float32, buffer=nl.sbuf))
    for b0, nt in blocks:
        accs = []
        for (i0, g), qt in zip(groups, qts):
            acc = nl.ndarray((PT, ntw), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=acc[0 : g * nh, 0:nt], stationary=qt[0:PT, 0 : g * nh], moving=kT[0:PT, b0 : b0 + nt])
            accs.append(acc)
        rects = []
        for (i0, g), acc in zip(groups, accs):
            rect = nl.ndarray((PT, ntw), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rect[0 : g * nh, 0:nt], data=acc[0 : g * nh, 0:nt], op0=nl.maximum, operand0=0.0)
            rects.append(rect)
        for (i0, g), wt, rect, row in zip(groups, wts, rects, rows):
            wsum = nl.ndarray((G, ntw), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=wsum[0:g, 0:nt], stationary=wt[0 : g * nh, 0:g], moving=rect[0 : g * nh, 0:nt])
            nisa.tensor_copy(dst=row[0:g, b0 : b0 + nt], src=wsum[0:g, 0:nt])
    for (i0, g), row in zip(groups, rows):
        nisa.dma_copy(dst=sc_hbm.ap(pattern=[[read_len, g], [1, ncol]], offset=(tok0 + i0) * read_len + c0),
                      src=row[0:g, 0:ncol])


def index_topk(sc_hbm, clen_hbm, sc16_hbm, topk_hbm, T, Tp, read_len, topk, pid):
    """Mask `sc[t, j]` for `j >= clen[t]` to NEG_BOUND, round to bf16, select the top
    `topk` positions per token on the GpSimd engine (16-partition snake layout) and
    write `topk_hbm[t, 0:topk]` (int32, -1 where the pick lies beyond `clen[t]`).
    Rows beyond `T` are left alone. Token groups of 8 alternate between the cores."""
    n16 = read_len // 16
    k16 = cdiv(topk, 16)
    # 1. mask + bf16, by row tile (this core's tiles)
    tiles = [(i * PT, min(PT, T - i * PT)) for i in range(cdiv(T, PT)) if i % 2 == pid]
    for r0, nr in tiles:
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
        if TOPK_SNAKE_ROWS:
            # stored snake-major, `sc16[t, p * n16 + c] = s16[t, 16 c + p]` (the bf16 cast
            # copy reads `s` in that order), so a token's 16 snake partitions are 16
            # contiguous runs and a group's 8 tokens load with one DMA
            s16 = nl.ndarray((PT, 16, n16), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=s16[0:nr, :, :], src=s.ap(pattern=[[read_len, nr], [1, 16], [16, n16]], offset=0))
            nisa.dma_copy(dst=sc16_hbm[r0 : r0 + nr, 0:read_len], src=s16.reshape((PT, read_len))[0:nr, :])
        else:
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
        snake = nl.ndarray((PT, n16), dtype=nl.bfloat16, buffer=nl.sbuf)
        if rows < 8:
            nisa.memset(snake, NEG_BOUND)
        if TOPK_SNAKE_ROWS:
            # snake[16 q + p, c] = sc16[g*8 + q, p * n16 + c]: a uniform partition stride n16
            nisa.dma_copy(dst=snake[0 : 16 * rows, 0:n16],
                          src=sc16_hbm.ap(pattern=[[n16, 16 * rows], [1, n16]], offset=g * 8 * read_len))
        else:
            for q in range(rows):
                nisa.dma_copy(dst=snake[16 * q : 16 * q + 16, 0:n16],
                              src=sc16_hbm.ap(pattern=[[1, 16], [16, n16]], offset=(g * 8 + q) * read_len))
        vals = nl.ndarray((PT, topk), dtype=nl.bfloat16, buffer=nl.sbuf)
        idx = nl.ndarray((PT, topk), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.topk(val_dst=vals, idx_dst=idx, src=snake, n=read_len, sorted=False)
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
