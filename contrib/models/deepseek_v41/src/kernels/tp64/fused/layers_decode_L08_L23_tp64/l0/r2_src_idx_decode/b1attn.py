# SPDX-License-Identifier: Apache-2.0
"""The attention side of a bs1 decode block (one span of T = 6 rows), SBUF resident (round 5).

    xn (HBM rows) -> q / kv projections -> sparse attention -> wo_a / wo_b -> cc_src (HBM)

Replaces the HBM round trips of the generic body (xn -> qa -> qr -> q -> kvtab / idx -> o -> oq -> z
-> cc_src, a core barrier at each) with on-chip hand-offs: every intermediate stays in SBUF and the
two cores exchange the few values they need by `sendrecv`:

  core 0: qa = xn @ wq_a, q_norm, qr staged (MX)    core 1: kv = xn @ wkv, kv_norm, rope, fp8 round
                                                            trip, the cache scatter
          -- swap: qr (staged) to core 1, the step's kv rows to core 0 --
  both:   q = qr @ wq_b for its 4 heads (6 rows), rope, transposed per head
          -- swap: the other core's rows of its heads --
  both:   attention of its 3 rows (all 8 heads): the window (127 cached keys + the 6 new ones) and
          its rows' top-k compressed keys, one score tile [24 = (row, head), keys], one softmax,
          the PV matmuls accumulated into one [24, 512] tile (the compressed part per row with the
          other rows' columns of P zeroed)
          z = o @ wo_a for its 3 rows (inverse rope as o is staged)
          -- swap: z (staged) --
  both:   its half of wo_b's output columns for all 6 rows -> cc_src

Same arithmetic as the generic body's (the MX activation quantization of each GEMM input, bf16
GEMM outputs, the fp32 rope / norms / softmax, the fp8 round trip of kv); the softmax probabilities
are scaled by the reciprocal of their sum before the bf16 cast of the PV operand (the body scales
the fp32 PV result), and the fp32 sums run in a different order.
"""

import nki.isa as nisa
import nki.language as nl

from .blockquant import _qdq_fp8
from .mk_common import NORM_EPS, PT, ROPE_DIM, ROPE_HALF

FP8_X4 = nl.float8_e4m3fn_x4
KC = 512
NEG = -1.0e30
#: rows of a span (1 committed + 5 drafted tokens) and their split over the two cores
T6 = 6
RH = 3
HEADS = 8
HL = 4
HD = 512
DC = HD // PT


#: debug: the first `attn_side` call of a trace stores its intermediates (fp32) into shared-HBM
#: tensors `DBG` (name, tensor [2 (core), P, F]) the launch returns after its outputs
DEBUG = __import__("os").environ.get("DS41_B1_DEBUG", "0") == "1"
DBG = []
_dbg_on = [False]


DBG_SHAPES = [("xn", 6, 5120), ("ixc", 128, 12), ("wbias", 24, 133), ("cb24", 24, 512), ("qa16", 6, 1280),
              ("qr16", 6, 1280), ("kv16", 6, 512), ("kvq", 6, 512), ("q16pre", 6, 2048), ("q16", 6, 2048),
              ("kvrecv", 6, 512), ("s32", 24, 645), ("o16pre", 24, 512), ("o16", 24, 512), ("z16", 4, 1024),
              ("ccsrc", 6, 2560), ("qall", 128, 96)]


def dbg_alloc():
    if not (DEBUG and _dbg_on[0]):
        return
    for n, p, f in DBG_SHAPES:
        DBG.append((f"dbg_{n}", nl.ndarray((2, p, f), dtype=nl.float32, buffer=nl.shared_hbm, name=f"dbg_{n}")))


def dbg(name, tile, p, f, pid):
    if not (DEBUG and _dbg_on[0]):
        return
    h = [t for n, t in DBG if n == f"dbg_{name}"][0]
    t32 = nl.ndarray((p, f), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=t32, src=tile)
    nisa.dma_copy(dst=h[pid], src=t32)


def _hw(eng):
    return dict(dge_mode=nisa.dge_mode.hwdge, engine=eng)


def chunk_parts(k):
    """`[(k0, pk)]` of a contraction of `k` (512-wide chunks; `pk` = partitions used)."""
    out = []
    k0 = 0
    while k0 < k:
        kk = min(KC, k - k0)
        out.append((k0, kk // 4))
        k0 += kk
    return out


#: instructions of the attention side others pin after: "side" = this core's q / kv path's last
#: (its exchange buffer written)
ANCH = {}
#: the last weight DMA `load_w` issued (`prefetch_next` returns it: the MoE pins its down GEMM after it)
_LAST_DMA = [None]
#: every DMA `load_w` issued since the list was last cleared (`prefetch_next` returns them all: the
#: weights' DMA too, which the scheduler otherwise sinks into the next block)
_ALL_DMA = []
#: zero a loaded weight's scale tile first (its partitions past the quadrants' first 4 are not read)
LOADW_MEMSET = False


def _pin_after(ins, after):
    """`ins` scheduled after `after` (a schedule-only dependency), if given."""
    if after is not None and hasattr(ins, "depends_on") and hasattr(after, "depends_on"):
        ins.depends_on(after, kind=nisa.dependency.schedule)
    return ins


def load_w(w4, s, k, n0, n, eng, prio=None, dst=None):
    """Weight columns `[n0, n0 + n)` of an MX weight (`w4` `[k/4, N]` x4 view, `s` `[k/32, N]`
    scales) into SBUF: `(wt [128, nkc, n], ws [128, nkc, n])`, the scales in the quadrant layout
    (into `dst`'s tiles, if given)."""
    parts = chunk_parts(k)
    nkc = len(parts)
    if dst is not None:
        wt, ws = dst
    else:
        wt = nl.ndarray((PT, nkc, n), dtype=w4.dtype, buffer=nl.sbuf)
        ws = nl.ndarray((PT, nkc, n), dtype=nl.uint8, buffer=nl.sbuf)
    if LOADW_MEMSET:
        nisa.memset(ws, 0, engine=nisa.engine.gpsimd)
    kw = dict(_hw(eng))
    if prio is not None:
        kw["priority"] = prio
    nfull = sum(1 for _, pk in parts if pk == PT)
    ncol = w4.shape[1]
    scol = s.shape[1]
    if nfull:
        _LAST_DMA[0] = nisa.dma_copy(dst=wt[0:PT, 0:nfull, :], src=w4.ap(pattern=[[ncol, PT], [PT * ncol, nfull], [1, n]], offset=n0),
                      **kw)
        _ALL_DMA.append(_LAST_DMA[0])
        for q in range(4):
            _LAST_DMA[0] = nisa.dma_copy(dst=ws[32 * q : 32 * q + 4, 0:nfull, :],
                          src=s.ap(pattern=[[scol, 4], [16 * scol, nfull], [1, n]], offset=4 * q * scol + n0), **kw)
    for kc in range(nfull, nkc):
        k0, pk = parts[kc]
        r0 = k0 // 4
        _LAST_DMA[0] = nisa.dma_copy(dst=wt[0:pk, kc, :], src=w4[r0 : r0 + pk, n0 : n0 + n], **kw)
        _ALL_DMA.append(_LAST_DMA[0])
        for q in range(pk // 32):
            g0 = r0 // 8 + 4 * q
            _LAST_DMA[0] = nisa.dma_copy(dst=ws[32 * q : 32 * q + 4, kc, :], src=s[g0 : g0 + 4, n0 : n0 + n], **kw)
    _ALL_DMA.append(_LAST_DMA[0])
    return wt, ws


def stage_rows(rows, nr, k, nc):
    """bf16 SBUF rows `rows[0:nr, 0:k]` (k a multiple of 512, columns past the data zero) -> the MX
    stationary `(xq [128, nkc * nc], xs [128, nkc * nc])`: chunk kc's operand is columns
    `kc*nc .. kc*nc + nr` (nc >= nr even; columns past nr zero)."""
    nkc = k // KC
    tp = nl.ndarray((PT, nkc, 4, nc), dtype=nl.bfloat16, buffer=nl.psum)
    for kc in range(nkc):
        for j in range(4):
            nisa.nc_transpose(dst=tp[:, kc, j, 0:nr], data=rows.ap(pattern=[[rows.shape[1], nr], [4, PT]],
                                                                    offset=kc * KC + j))
    xT = nl.ndarray((PT, nkc, nc, 4), dtype=nl.bfloat16, buffer=nl.sbuf)
    if nc > nr:
        nisa.memset(xT, 0.0)
    nisa.tensor_copy(dst=xT[:, :, 0:nr, :], src=tp.ap(pattern=[[nkc * 4 * nc, PT], [4 * nc, nkc], [1, nr], [nc, 4]]))
    xq = nl.ndarray((PT, nkc * nc), dtype=FP8_X4, buffer=nl.sbuf)
    xs = nl.ndarray((PT, nkc * nc), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(dst=xq, src=xT.reshape((PT, nkc * nc * 4)), dst_scale=xs)
    return xq, xs


def norm_w_t(w_hbm, n, kp, eng):
    """`norm_stage`'s weight transposed: wT[p, kc, j] = w[512 kc + 4 p + j] (zero past n), bf16."""
    nkc = kp // KC
    wT = nl.ndarray((PT, nkc, 4), dtype=nl.bfloat16, buffer=nl.sbuf)
    nfull = n // KC
    if nfull < nkc:
        nisa.memset(wT, 0.0)
    nisa.dma_copy(dst=wT[:, 0:nfull, :], src=w_hbm.ap(pattern=[[4, PT], [KC, nfull], [1, 4]], offset=0), **_hw(eng))
    for kc in range(nfull, nkc):
        r = n - kc * KC
        if r > 0:
            nisa.dma_copy(dst=wT[0 : r // 4, kc, :], src=w_hbm.ap(pattern=[[4, r // 4], [1, 4]], offset=kc * KC),
                          **_hw(eng))
    return wT


def norm_stage(rows16, nr, n, kp, w_hbm, wT=None):
    """`stage_rows(rmsnorm(rows16) * w, nr, kp, nr)` with the norm in the transposed domain (the
    staging's transposes first: 128 partitions x a few columns instead of `nr` partitions x `n`):
    the sum of squares of the bf16 values per row (fp32: the per-partition sums of the transposes, the
    partitions' by a ones matmul onto every partition), rstd, x * rstd * w (fp32, one bf16 rounding),
    quantized. `rows16[0:nr, 0:kp]` bf16 (columns n .. kp zero), `w_hbm` `[1, n]`."""
    nkc = kp // KC
    tp = nl.ndarray((PT, nkc, 4, nr), dtype=nl.bfloat16, buffer=nl.psum)
    for kc in range(nkc):
        for j in range(4):
            nisa.nc_transpose(dst=tp[:, kc, j, 0:nr], data=rows16.ap(pattern=[[rows16.shape[1], nr], [4, PT]],
                                                                      offset=kc * KC + j))
    # tv[p, t, kc, j] = x[t, 512 kc + 4 p + j]
    tv = tp.ap(pattern=[[nkc * 4 * nr, PT], [1, nr], [4 * nr, nkc], [nr, 4]], offset=0)
    sq = nl.ndarray((PT, nr, nkc, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=tv)
    ssp = nl.ndarray((PT, nr), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ssp, op=nl.add, data=sq.reshape((PT, nr, nkc * 4)), axis=(2,))
    ones = nl.ndarray((PT, PT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)
    ps = nl.ndarray((PT, nr), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ps, stationary=ones, moving=ssp)
    # rstd = 1 / sqrt(ss / n + eps), one Newton step (`rstd_of` per (partition, row))
    m = nl.ndarray((PT, nr), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=m, data=ps, op0=nl.multiply, operand0=1.0 / n, op1=nl.add, operand1=NORM_EPS)
    y = nl.ndarray((PT, nr), dtype=nl.float32, buffer=nl.sbuf)
    y[...] = nl.rsqrt(m)
    yy = nl.ndarray((PT, nr), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=yy, data1=y, data2=y, op=nl.multiply)
    c = nl.ndarray((PT, nr), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=c, data1=yy, data2=m, op=nl.multiply)
    nisa.tensor_scalar(dst=c, data=c, op0=nl.multiply, operand0=-0.5, op1=nl.add, operand1=1.5)
    rs = nl.ndarray((PT, nr), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=rs, data1=y, data2=c, op=nl.multiply)
    # the weight transposed: wT[p, kc, j] = w[512 kc + 4 p + j] (zero past n)
    if wT is None:
        wT = norm_w_t(w_hbm, n, kp, nisa.engine.scalar)
    yv = nl.ndarray((PT, nkc, nr, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=yv, data1=tp.ap(pattern=[[nkc * 4 * nr, PT], [4 * nr, nkc], [1, nr], [nr, 4]], offset=0),
                       data2=rs.ap(pattern=[[nr, PT], [0, nkc], [1, nr], [0, 4]], offset=0), op=nl.multiply)
    xT = nl.ndarray((PT, nkc, nr, 4), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=xT, data1=yv, data2=wT.ap(pattern=[[nkc * 4, PT], [4, nkc], [0, nr], [1, 4]], offset=0),
                       op=nl.multiply)
    xq = nl.ndarray((PT, nkc * nr), dtype=FP8_X4, buffer=nl.sbuf)
    xs = nl.ndarray((PT, nkc * nr), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(dst=xq, src=xT.reshape((PT, nkc * nr * 4)), dst_scale=xs)
    return xq, xs


def qdq_fp8(tile, rows, n):
    """`blockquant._qdq_fp8` (blockwise fp8 round trip, scale 2^ceil(log2(amax / peak))) in fewer
    instructions: the exponent arithmetic fused two ops an instruction, the power-of-two scale and
    its reciprocal used in fp32 (exact powers of two: the same products as their bf16 copies)."""
    from .blockquant import BLK, FP8_AMAX_FLOOR, FP8_PEAK
    nb = n // BLK
    blocked = tile.reshape((rows, nb, BLK))
    ab = nl.ndarray((PT, nb, BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=ab[0:rows], op=nl.abs, data=blocked, bias=0.0, scale=1.0)
    am = nl.ndarray((PT, nb, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=am[0:rows], op=nl.max, data=ab[0:rows], axis=(2,), keepdims=True)
    m = nl.ndarray((PT, nb, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=m[0:rows], data=am[0:rows], op0=nl.maximum, operand0=FP8_AMAX_FLOOR, op1=nl.multiply,
                       operand1=1.0 / FP8_PEAK)
    bits = m.view(nl.int32)
    e = nl.ndarray((PT, nb, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=e[0:rows], data=bits[0:rows], op0=nl.right_shift, operand0=23, op1=nl.bitwise_and,
                       operand1=0xFF)
    bump = nl.ndarray((PT, nb, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=bump[0:rows], data=bits[0:rows], op0=nl.bitwise_and, operand0=0x7FFFFF)
    nisa.tensor_scalar(dst=bump[0:rows], data=bump[0:rows], op0=nl.greater, operand0=0)
    nisa.tensor_tensor(dst=e[0:rows], data1=e[0:rows], data2=bump[0:rows], op=nl.add)
    sc = nl.ndarray((PT, nb, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sc[0:rows], data=e[0:rows], op0=nl.left_shift, operand0=23)
    ie = nl.ndarray((PT, nb, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ie[0:rows], data=e[0:rows], op0=nl.multiply, operand0=-1, op1=nl.add, operand1=254)
    nisa.tensor_scalar(dst=ie[0:rows], data=ie[0:rows], op0=nl.left_shift, operand0=23)
    q8 = nl.ndarray((PT, nb, BLK), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=q8[0:rows], data1=blocked, data2=ie.view(nl.float32).ap(
        pattern=[[nb, rows], [1, nb], [0, BLK]], offset=0), op=nl.multiply)
    deq = nl.ndarray((PT, nb, BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=deq[0:rows], data1=q8[0:rows], data2=sc.view(nl.float32).ap(
        pattern=[[nb, rows], [1, nb], [0, BLK]], offset=0), op=nl.multiply)
    return deq.reshape((PT, n))


#: kv's fp8 round trip by `qdq_fp8` (else `blockquant._qdq_fp8`)
QDQ_FAST = True
#: q_norm and the wq_b staging in the transposed domain (`norm_stage`) instead of `norm_rows_sb` on 6
#: partitions + `stage_rows`
Q_NORM_T = True


def stage_xn(xn_hbm, k, packed=None, want_t=False):
    """`xn_hbm[0:6, 0:k]` (bf16 HBM rows) -> the MX stationary `(xq, xs)` (6 columns a chunk): one
    contiguous DMA into the packed tile (partition t * nkc + c = row t's chunk c), four PE transposes.
    `packed`: that tile already in SBUF (`xn_hbm` unused)."""
    nkc = k // KC
    q = T6 * nkc
    if packed is not None:
        xp = packed
    else:
        xp = nl.ndarray((PT, KC), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=xp[0:q, :], src=xn_hbm.ap(pattern=[[KC, q], [1, KC]], offset=0), **_hw(nisa.engine.sync))
    tp = nl.ndarray((PT, 4, q), dtype=nl.bfloat16, buffer=nl.psum)
    for j in range(4):
        nisa.nc_transpose(dst=tp[:, j, :], data=xp.ap(pattern=[[KC, q], [4, PT]], offset=j))
    # xT[p, c, t, j] = tp[p, j, t * nkc + c]
    xT = nl.ndarray((PT, nkc, T6, 4), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=xT, src=tp.ap(pattern=[[4 * q, PT], [1, nkc], [nkc, T6], [q, 4]]))
    xq = nl.ndarray((PT, nkc * T6), dtype=FP8_X4, buffer=nl.sbuf)
    xs = nl.ndarray((PT, nkc * T6), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(dst=xq, src=xT.reshape((PT, nkc * T6 * 4)), dst_scale=xs)
    if want_t:
        return xq, xs, xT
    return xq, xs


def mx_gemm(xq, xs, nr, nc, parts, wt, ws, c0, n, nt=512):
    """`x @ w[:, c0 : c0 + n]` for the staged stationary (`nc` columns a chunk, `nr` used): one PSUM
    tile per `nt`-column output tile (an accumulation group does not span PSUM banks: matmuls whose
    moving operand covers several banks accumulate only into the first), the contraction chunks in
    the outer loop. Returns `[(col, w, psum [nr, w])]`."""
    tiles = [(o, min(nt, n - o)) for o in range(0, n, nt)]
    accs = [nl.ndarray((nr, w), dtype=nl.float32, buffer=nl.psum) for _, w in tiles]
    for i, (_, pk) in enumerate(parts):
        for (o, w), acc in zip(tiles, accs):
            nisa.nc_matmul_mx(dst=acc, stationary=xq[0:pk, i * nc : i * nc + nr],
                              moving=wt[0:pk, i, c0 + o : c0 + o + w],
                              stationary_scale=xs[0:pk, i * nc : i * nc + nr],
                              moving_scale=ws[0:pk, i, c0 + o : c0 + o + w], accumulate=(i > 0))
    return [(o, w, acc) for (o, w), acc in zip(tiles, accs)]


#: the engine evacuating the GEMMs' PSUM tiles (the vector engine runs a Sinkhorn chain)
EVAC_ENG = nisa.engine.scalar


def evac(dst, tiles, eng=EVAC_ENG):
    """Copy `mx_gemm`'s PSUM tiles into the SBUF tile `dst[:, col ..]` (casting to its dtype)."""
    for o, w, acc in tiles:
        if eng is None:
            nisa.tensor_copy(dst=dst[:, o : o + w], src=acc)
        else:
            nisa.tensor_copy(dst=dst[:, o : o + w], src=acc, engine=eng)


def bcast_rows(row_hbm, n, nr, dtype, eng=None):
    """`row_hbm[0:1, 0:n]` on `nr` partitions (a broadcast DMA)."""
    t = nl.ndarray((PT, n), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t[0:nr, :], src=row_hbm.ap(pattern=[[0, nr], [1, n]], offset=0),
                  **(_hw(eng) if eng is not None else {}))
    return t


def rstd_of(ss, nr, n):
    """`1 / sqrt(ss / n + eps)` with one Newton step (`mk_common.rsqrt_newton`), `(nr, 1)` fp32."""
    m = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=m[0:nr, :], data=ss[0:nr, :], op0=nl.multiply, operand0=1.0 / n, op1=nl.add,
                       operand1=NORM_EPS)
    y = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    y[0:nr, :] = nl.rsqrt(m[0:nr, :])
    yy = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=yy[0:nr, :], data1=y[0:nr, :], data2=y[0:nr, :], op=nl.multiply)
    c = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=c[0:nr, :], data1=yy[0:nr, :], data2=m[0:nr, :], op=nl.multiply)
    nisa.tensor_scalar(dst=c[0:nr, :], data=c[0:nr, :], op0=nl.multiply, operand0=-0.5, op1=nl.add, operand1=1.5)
    r = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=r[0:nr, :], data1=y[0:nr, :], data2=c[0:nr, :], op=nl.multiply)
    return r


def norm_rows_sb(x16, nr, n, w_b, out16):
    """RMSNorm of the bf16 SBUF rows `x16[0:nr, 0:n]` with the broadcast weight `w_b` into `out16`
    (bf16): `mk_common.norm_rows`' arithmetic (sum of squares of the bf16 values, rstd, x * rstd,
    * w, one bf16 rounding)."""
    junk = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.sbuf)
    ss = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=junk[0:nr, :], op=nl.square, data=x16[0:nr, 0:n], reduce_op=nl.add,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce, reduce_res=ss[0:nr, 0:1])
    rs = rstd_of(ss, nr, n)
    yv = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=yv[0:nr, :], op=nl.copy, data=x16[0:nr, 0:n], scale=rs[0:nr, 0:1])
    nisa.tensor_tensor(dst=out16[0:nr, 0:n], data1=yv[0:nr, :], data2=w_b[0:nr, 0:n], op=nl.multiply)


def rope_cols(t, nr, base, stride, nh, cs, inverse):
    """Rope (`mk_common.rope_inplace`'s arithmetic) on the bf16 SBUF tile `t`: the `ROPE_DIM` columns
    at `base + h * stride` of rows `0..nr`, for `h < nh`; `cs` `(PT, 64)` fp32 (cos | sin) per row."""
    w = t.shape[1]
    a = t.ap(pattern=[[w, nr], [stride, nh], [2, ROPE_HALF]], offset=base)
    b = t.ap(pattern=[[w, nr], [stride, nh], [2, ROPE_HALF]], offset=base + 1)
    cos = cs.ap(pattern=[[2 * ROPE_HALF, nr], [0, nh], [1, ROPE_HALF]], offset=0)
    sin = cs.ap(pattern=[[2 * ROPE_HALF, nr], [0, nh], [1, ROPE_HALF]], offset=ROPE_HALF)

    def mul(x, y):
        r = nl.ndarray((PT, nh, ROPE_HALF), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=r[0:nr], data1=x, data2=y, op=nl.multiply)
        return r

    ac, bs, as_, bc = mul(a, cos), mul(b, sin), mul(a, sin), mul(b, cos)
    if inverse:
        nisa.tensor_tensor(dst=a, data1=ac[0:nr], data2=bs[0:nr], op=nl.add)
        nisa.tensor_tensor(dst=b, data1=bc[0:nr], data2=as_[0:nr], op=nl.subtract)
    else:
        nisa.tensor_tensor(dst=a, data1=ac[0:nr], data2=bs[0:nr], op=nl.subtract)
        nisa.tensor_tensor(dst=b, data1=as_[0:nr], data2=bc[0:nr], op=nl.add)


def freqs_rows(freqs, row_pos, rows, nrep=1):
    """`(PT, 64)` fp32 (cos | sin) of the given rows' positions, row `rows[i]` on partitions
    `i * nrep .. i * nrep + nrep` (an indirect gather by position)."""
    n = len(rows) * nrep
    pos = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    for i, r in enumerate(rows):
        nisa.dma_copy(dst=pos[i * nrep : (i + 1) * nrep, :], src=row_pos.ap(pattern=[[0, nrep], [1, 1]], offset=r),
                      **_hw(nisa.engine.sync))
    cs = nl.ndarray((PT, 2 * ROPE_HALF), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cs[0:n, :], src=freqs.ap(pattern=[[2 * ROPE_HALF, n], [1, 2 * ROPE_HALF]],
                                             vector_offset=pos[0:n, 0:1], indirect_dim=0))
    return cs


#: the q / kv exchange as two small sendrecvs (core 0's staged qr rows `[128, 16]` words, core 1's kv rows
#: `[8, 256]` words) instead of one `[128, 256]` tile (128 KB each way for ~8 KB of data)
SWAP21_SPLIT = False
#: the attention exchanges' receive tiles launch-persistent (`swap(bufs=)`): a fresh tile could share
#: SBUF with a transient tile read later by the background coefficient steps, the exchange then
#: waiting for them (a false write-after-read dependency)
SWAP_PERSIST = True


def swap(pid, src, pipe, bufs=None):
    """`sendrecv` with the other core (its `src` into the returned tile of the same shape). `bufs`: a
    launch-level dict for a persistent receive tile (`SWAP_PERSIST`)."""
    if SWAP_PERSIST and bufs is not None:
        key = ("swap", pipe, tuple(src.shape), str(src.dtype), pid)
        dst = bufs.get(key)
        if dst is None:
            dst = bufs[key] = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.sbuf, name=f"swr{pipe}_{pid}")
    else:
        dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.sbuf)
    nisa.sendrecv(src=src, dst=dst, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=pipe,
                  dma_engine=nisa.dma_engine.gpsimd_dma)
    return dst


def col_f32(col_hbm, rows, nrep=1):
    """int32 host column values of `rows` as fp32 on partitions (row i on i*nrep..)."""
    n = len(rows) * nrep
    ci = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    for i, r in enumerate(rows):
        nisa.dma_copy(dst=ci[i * nrep : (i + 1) * nrep, :], src=col_hbm.ap(pattern=[[0, nrep], [1, 1]], offset=r))
    cf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cf[0:n, :], src=ci[0:n, :])
    return cf


def _abuf(bufs, pid, w4, k, n):
    """The launch's persistent SBUF tiles of the first GEMM's weights (`A_PERSIST`), else None."""
    if not A_PERSIST or bufs is None or (A_PERSIST_CORES is not None and pid not in A_PERSIST_CORES):
        return None
    key = ("abuf", pid, k, n, str(w4.dtype))
    if key not in bufs:
        nkc = len(chunk_parts(k))
        bufs[key] = (nl.ndarray((PT, nkc, n), dtype=w4.dtype, buffer=nl.sbuf, name=f"abuf_w{pid}_{k}_{n}"),
                     nl.ndarray((PT, nkc, n), dtype=nl.uint8, buffer=nl.sbuf, name=f"abuf_s{pid}_{k}_{n}"))
    return bufs[key]


def attn_weights(pid, wq_a, wq_a_s, wq_b, wq_b_s, wkv, wkv_s, wo_a, wo_a_s, wo_b, wo_b_s, prio=None, eng=None,
                 parts=("a", "qb", "oa", "ob"), W=None, bufs=None):
    """The attention side's weights of this core in SBUF (`load_w`): q_a (core 0) or kv (core 1), its
    half of wq_b's heads, all of wo_a, its half of wo_b's columns. The x4 views of the int32 / int16
    HBM words are taken here. `parts`: which ("a": q_a / kv), into `W` (a dict, else a new one)."""
    v = lambda w: w if w.dtype == FP8_X4 else w.view(FP8_X4)
    wq_a, wq_b, wkv, wo_a, wo_b = v(wq_a), v(wq_b), v(wkv), v(wo_a), v(wo_b)
    dim = wq_a.shape[0] * 4
    q_lora = wq_a.shape[1]
    g_in = wo_a.shape[0] * 4
    o_lora = wo_a.shape[1]
    half = dim // 2
    hw0, hw1 = (nisa.engine.sync, nisa.engine.scalar) if eng is None else (eng, eng)
    W = {} if W is None else W
    if "a" in parts:
        if pid == 0:
            W["qa"] = load_w(wq_a, wq_a_s, dim, 0, q_lora, hw0, prio, dst=_abuf(bufs, pid, wq_a, dim, q_lora))
        else:
            W["kv"] = load_w(wkv, wkv_s, dim, 0, HD, hw0, prio, dst=_abuf(bufs, pid, wkv, dim, HD))
    if "qb" in parts:
        W["qb"] = load_w(wq_b, wq_b_s, q_lora, pid * HL * HD, HL * HD, hw1, prio,
                         dst=_abuf(bufs, pid, wq_b, q_lora, HL * HD) if "qb" in PERSIST_PARTS else None)
    if "oa" in parts:
        W["oa"] = load_w(wo_a, wo_a_s, g_in, 0, o_lora, hw1, prio,
                         dst=_abuf(bufs, pid, wo_a, g_in, o_lora) if "oa" in PERSIST_PARTS else None)
    if "ob" in parts:
        W["ob"] = load_w(wo_b, wo_b_s, o_lora, pid * half, half, hw0, prio)
    return W


#: DMA priority (0 highest .. 3) of the next member's attention-weight prefetch
PREFETCH_PRIO = None
#: its DMA queue (the ACT queue: the sync queue carries the block-start loads / the all-reduce stores)
PREFETCH_ENG = nisa.engine.scalar
#: the weights prefetched (the first GEMM's: q_a / kv; the others loaded at the block's start, the
#: DMA queues idle then, needed tens of us later) -- the prefetch window and the SBUF next to the
#: MoE's buffers hold less
PREFETCH_PARTS = ("a",)
#: the MoE's down GEMM pinned after every prefetch DMA (else the last one only, a scales load: the
#: weights' DMA was then issued by the scheduler only at the next block's x_norm)
PIN_ALL = True
#: the first GEMM's weights (q_a on core 0, kv on core 1) in one launch-level SBUF tile pair that every
#: member loads / prefetches into (the scheduler otherwise delayed the 6.5 MB q_a prefetch into the
#: next block: its fresh tile's space was taken by the next block's hoisted loads)
A_PERSIST = False
#: the cores with the persistent tile (None: both)
A_PERSIST_CORES = None
#: the other parts in launch-level tiles too (those prefetched: `PREFETCH_PARTS`)
PERSIST_PARTS = ()
#: hc_in tail steps (`b1.Ticker`) at the attention side's waits: before the q / kv swap, behind the
#: wq_b GEMM, the score matmuls, the wo_a GEMM
TICKS = dict(q=3, p=4, o=3)
#: the q exchange issued before the background steps at its wait (else after them)
SWAP_FIRST = False
#: the q / kv paths' norm weights loaded first, on the sync queue
SMALL_FIRST = True
#: the window keys' transposes and the index head weights behind the q / kv exchange (else before
#: the q / kv paths, where the window gather's wait held the DVE queue)
LATE_SIDE_WORK = True


def prefetch_next(pid, wq_a_s, share):
    """Issue the next member of this kind's attention weights (`share["b1"]["attn_next"]`, keyed by
    this member's `wq_a_s`, the launch's table) into SBUF now; that member picks them up."""
    B = share.setdefault("b1", {}) if share is not None else {}
    nxt = B.get("attn_next", {}).get(id(wq_a_s))
    if nxt is None:
        return
    _ALL_DMA.clear()
    B.setdefault("pref", {})[id(nxt[1])] = attn_weights(pid, *nxt, prio=PREFETCH_PRIO, eng=PREFETCH_ENG,
                                                        parts=PREFETCH_PARTS, bufs=B)
    return list(_ALL_DMA) if PIN_ALL else _LAST_DMA[0]


def launch_tables(pid, row_pos, wv, freqs, nwin, my_rows):
    """The step's position tables for this core's rows (the same for every member of a launch):
    `(wbias [24, nwin] fp32, cs6 [6, 64], cs24 [24, 64])`: the window validity bias on the
    (row, head) partitions (window table row i valid for global row g iff 0 <= i - g <= 127 and
    wv[g] + i - g >= 0: rows g .. g + 127 of the table are g's window) and the rope (cos | sin)."""
    nq = RH * HEADS
    eng = nisa.engine.sync
    gi = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    p0 = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    wvi = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    for r, g in enumerate(my_rows):
        sl8 = slice(r * HEADS, (r + 1) * HEADS)
        nisa.dma_copy(dst=gi[sl8, :], src=row_pos.ap(pattern=[[0, HEADS], [1, 1]], offset=g), **_hw(eng))
        nisa.dma_copy(dst=wvi[sl8, :], src=wv.ap(pattern=[[0, HEADS], [1, 1]], offset=g), **_hw(eng))
    nisa.dma_copy(dst=p0[0:nq, :], src=row_pos.ap(pattern=[[0, nq], [1, 1]], offset=0), **_hw(eng))
    gf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gf[0:nq], src=gi[0:nq])
    p0f = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=p0f[0:nq], src=p0[0:nq])
    nisa.tensor_tensor(dst=gf[0:nq], data1=gf[0:nq], data2=p0f[0:nq], op=nl.subtract)
    wvf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=wvf[0:nq], src=wvi[0:nq])
    ii = nl.ndarray((PT, nwin), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=ii, pattern=[[1, nwin]], offset=0, channel_multiplier=0)
    rel = nl.ndarray((PT, nwin), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rel, src=ii)
    nisa.tensor_scalar(dst=rel[0:nq], data=rel[0:nq], op0=nl.subtract, operand0=gf[0:nq, 0:1])
    v1 = nl.ndarray((PT, nwin), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=v1[0:nq], data=rel[0:nq], op0=nl.greater_equal, operand0=0.0)
    v2 = nl.ndarray((PT, nwin), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=v2[0:nq], data=rel[0:nq], op0=nl.less_equal, operand0=float(PT - 1))
    nisa.tensor_tensor(dst=v1[0:nq], data1=v1[0:nq], data2=v2[0:nq], op=nl.multiply)
    nisa.tensor_scalar(dst=v2[0:nq], data=rel[0:nq], op0=nl.add, operand0=wvf[0:nq, 0:1], op1=nl.greater_equal,
                       operand1=0.0)
    nisa.tensor_tensor(dst=v1[0:nq], data1=v1[0:nq], data2=v2[0:nq], op=nl.multiply)
    wbias = nl.ndarray((PT, nwin), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=wbias[0:nq], data=v1[0:nq], op0=nl.subtract, operand0=1.0, op1=nl.multiply,
                       operand1=-NEG)
    cs6 = freqs_rows(freqs, row_pos, list(range(T6)))
    cs24 = freqs_rows(freqs, row_pos, my_rows, HEADS)
    return wbias, cs6, cs24


def group_keys(pid, topk_in, cmp_slots, compress_kv_cache, my_rows):
    """This core's rows' top-k compressed keys (the same for every member reading this top-k and
    cache): `(vc [128, 12, 512] bf16: key c*128 + p of local row r at [p, r*4 + c], kTc [128, 4, 3, topk]
    (K^T), cb24 [24, topk] fp32: the validity bias (topk < 0) on the (row, head) partitions)`.
    slot = cmp_slots[max(topk, 0)]: tile i = (row r, 128-run c) of the ids on partition i of a
    [12, 128] tile, the slot lookup on chip, the gather column one PE transpose."""
    topk = topk_in.shape[1]
    ktc = topk // PT
    nrc = RH * ktc
    nq = RH * HEADS
    eng = nisa.engine.sync
    read_len = cmp_slots.shape[0]
    tki = nl.ndarray((PT, PT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tki[0:nrc, :], src=topk_in.ap(pattern=[[PT, nrc], [1, PT]], offset=pid * RH * topk), **_hw(eng))
    tkf = nl.ndarray((PT, PT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tkf[0:nrc, :], src=tki[0:nrc, :])
    nisa.tensor_scalar(dst=tkf[0:nrc, :], data=tkf[0:nrc, :], op0=nl.maximum, operand0=0.0)
    tku = nl.ndarray((PT, PT), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tku[0:nrc, :], src=tkf[0:nrc, :])
    stab = nl.ndarray((PT, read_len), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=stab[0:nrc, :], src=cmp_slots.ap(pattern=[[0, nrc], [1, read_len]], offset=0), **_hw(eng))
    sl = nl.ndarray((PT, PT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nc_n_gather(dst=sl[0:nrc, :], data=stab[0:nrc, :], indices=tku[0:nrc, :])
    slf = nl.ndarray((PT, PT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=slf[0:nrc, :], src=sl[0:nrc, :])
    ixp = nl.ndarray((PT, PT), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=ixp[0:PT, 0:nrc], data=slf[0:nrc, :])
    ixc = nl.ndarray((PT, nrc), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ixc, src=ixp[0:PT, 0:nrc])
    vc = nl.ndarray((PT, nrc, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i in range(nrc):
        nisa.dma_copy(dst=vc[:, i, :], src=compress_kv_cache.ap(pattern=[[HD, PT], [1, HD]],
                                                                vector_offset=ixc[0:PT, i : i + 1], indirect_dim=0))
    kTc = nl.ndarray((PT, DC, RH, topk), dtype=nl.bfloat16, buffer=nl.sbuf)
    for r in range(RH):
        for c in range(ktc):
            tps = nl.ndarray((PT, DC, PT), dtype=nl.bfloat16, buffer=nl.psum)
            for dc in range(DC):
                nisa.nc_transpose(dst=tps[:, dc, :], data=vc[:, r * ktc + c, dc * PT : (dc + 1) * PT])
            nisa.tensor_copy(dst=kTc[:, :, r, c * PT : (c + 1) * PT], src=tps)
    tk24 = nl.ndarray((PT, topk), dtype=nl.int32, buffer=nl.sbuf)
    for r, g in enumerate(my_rows):
        nisa.dma_copy(dst=tk24[r * HEADS : (r + 1) * HEADS, :], src=topk_in.ap(pattern=[[0, HEADS], [1, topk]],
                                                                                 offset=g * topk), **_hw(eng))
    tkf24 = nl.ndarray((PT, topk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tkf24[0:nq], src=tk24[0:nq])
    cb24 = nl.ndarray((PT, topk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=cb24[0:nq], data=tkf24[0:nq], op0=nl.less, operand0=0.0, op1=nl.multiply, operand1=NEG)
    return vc, kTc, cb24


def attn_side(pid, xn, wq_a, wq_a_s, q_norm_w, wq_b, wq_b_s, wkv, wkv_s, kv_norm_w, wo_a, wo_a_s,
              wo_b, wo_b_s, attn_sink, freqs, row_pos, write_slots, win_slots, cmp_slots, wv, topk_in,
              window_kv_cache, compress_kv_cache, cc_src, after_gemm=None, share=None, xn_packed=False,
              after_side=None, tick=None, ix=None, window_only=False, src=None):
    """The block's attention side for one 6-row span (see the module docstring); writes this
    core's columns of `cc_src[0:6]` (fp32). `after_gemm()` runs once the q / kv GEMMs are issued,
    `after_side()` once this core's q / kv path is (before the swap); `tick(n)` steps a background
    chain at the waits (`TICKS`). `ix`: an index source's indexer (`b1idx`; its top-k into `topk_in`,
    the attended keys from it) instead of reading `topk_in`. `window_only`: no compressed keys (the r0
    block: `topk_in` / `cmp_slots` / `compress_kv_cache` unread)."""
    tk = tick if tick is not None else (lambda n=1, after=None: None)
    _dbg_on[0] = DEBUG and not DBG
    dbg_alloc()
    dim = wq_a.shape[0] * 4
    q_lora = wq_a.shape[1]
    nkd = dim // KC
    dparts = chunk_parts(dim)
    qparts = chunk_parts(q_lora)
    q_lora_p = len(qparts) * KC
    o_lora = wo_a.shape[1]
    g_in = wo_a.shape[0] * 4
    assert g_in == HEADS * HD and wkv.shape[1] == HD and wq_b.shape[1] == HEADS * HD
    nwin = win_slots.shape[0]
    nold = nwin - T6
    topk = 0 if window_only else topk_in.shape[1]
    ktc = topk // PT
    scale = float(HD) ** -0.5
    hw0, hw1 = nisa.engine.sync, nisa.engine.scalar
    my_rows = list(range(pid * RH, pid * RH + RH))

    # ---------------------------------------------------------------- small vectors first
    # (the norm weights on the sync queue ahead of the big weight loads: the q / kv path waited for them)
    pre_w = {}
    if SMALL_FIRST:
        if pid == 0 and Q_NORM_T:
            pre_w["q"] = norm_w_t(q_norm_w, q_lora, q_lora_p, hw0)
        if pid == 1:
            pre_w["kv"] = bcast_rows(kv_norm_w, HD, T6, nl.bfloat16, eng=hw0)

    if src is not None:
        # (a kv source: this step's compressed rows / index keys written before the indexer reads them;
        # its projection weight first, on the sync queue)
        from . import b1src
        srcs = b1src.prep(pid, src, dim, hw0)

    # ---------------------------------------------------------------- weights
    # (prefetched by the previous member of this kind, `prefetch_next`, else loaded here)
    B0 = share.setdefault("b1", {}) if share is not None else {}
    W = B0.setdefault("pref", {}).pop(id(wq_a_s), None)
    if W is None:
        W = attn_weights(pid, wq_a, wq_a_s, wq_b, wq_b_s, wkv, wkv_s, wo_a, wo_a_s, wo_b, wo_b_s, bufs=B0)
    else:
        # (the parts not prefetched, now: ACT queue)
        miss = tuple(k for k in ("a", "qb", "oa", "ob") if (k if k != "a" else ("qa" if pid == 0 else "kv")) not in W)
        attn_weights(pid, wq_a, wq_a_s, wq_b, wq_b_s, wkv, wkv_s, wo_a, wo_a_s, wo_b, wo_b_s, eng=nisa.engine.scalar,
                     parts=miss, W=W)
    if pid == 0:
        wqa_t = W["qa"]
    else:
        wkv_t = W["kv"]
    wqb_t, woa_t, wob_t = W["qb"], W["oa"], W["ob"]
    half = dim // 2

    # ---------------------------------------------------------------- tables
    # launch-level (the step's positions: the same for every member), the index group's compressed
    # keys (the top-k and the compressed cache: the same for every member reading them), this block's
    # cached window keys
    nq = RH * HEADS
    B = share.setdefault("b1", {}) if share is not None else {}
    lkey = ("lt", id(row_pos), id(wv), id(freqs), pid)
    if lkey not in B:
        B[lkey] = launch_tables(pid, row_pos, wv, freqs, nwin, my_rows)
    wbias, cs6, cs24 = B[lkey]
    gkey = ("gk", id(topk_in), id(compress_kv_cache), id(cmp_slots), pid)
    if window_only:
        pass
    elif ix is None:
        if gkey not in B:
            B[gkey] = group_keys(pid, topk_in, cmp_slots, compress_kv_cache, my_rows)
        vc, kTc, cb24 = B[gkey]
    else:
        from . import b1idx
        ixs = b1idx.prep(pid, B, ix, cmp_slots, q_lora)
    # old window rows (table rows 0 .. nold - 1): V rows on partitions, K^T by PE transposes
    wslot = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=wslot[0:nold, :], src=win_slots[0:nold, 0:1], **_hw(hw0))
    v_old = nl.ndarray((PT, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=v_old[0:nold, :], src=window_kv_cache.ap(pattern=[[HD, nold], [1, HD]], vector_offset=wslot[0:nold, 0:1],
                                                             indirect_dim=0))
    sink24 = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    for r in range(RH):
        nisa.dma_copy(dst=sink24[r * HEADS : (r + 1) * HEADS, :], src=attn_sink[0:HEADS, 0:1], **_hw(hw1))

    # ---------------------------------------------------------------- q / kv projections
    if ix is None:
        xq, xs = stage_xn(xn, dim, packed=xn if xn_packed else None)
    else:
        xq, xs, xT = stage_xn(xn, dim, packed=xn if xn_packed else None, want_t=True)
    xnd = nl.ndarray((T6, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
    if DEBUG and _dbg_on[0] and not xn_packed:
        nisa.dma_copy(dst=xnd, src=xn[0:T6, :])
    dbg("xn", xnd, T6, dim, pid)
    dbg("wbias", wbias[0:nq, :], nq, nwin, pid)
    if ix is None and not window_only:
        dbg("cb24", cb24[0:nq, :], nq, topk, pid)
    if pid == 0:
        qa_p = mx_gemm(xq, xs, T6, T6, dparts, *wqa_t, 0, q_lora)
    else:
        kv_p = mx_gemm(xq, xs, T6, T6, dparts, *wkv_t, 0, HD)
    if after_gemm is not None:
        after_gemm()
    if src is not None:
        b1src.compress(pid, srcs, xT, ix["barrier"])
    kTw = nl.ndarray((PT, DC, nwin), dtype=nl.bfloat16, buffer=nl.sbuf)

    def window_keys(after=None):
        # (PSUM accesses 4-byte aligned: each chunk's 127 bf16 columns in a 128-column slot)
        tpw = nl.ndarray((PT, DC, PT), dtype=nl.bfloat16, buffer=nl.psum)
        for dc in range(DC):
            _pin_after(nisa.nc_transpose(dst=tpw[:, dc, 0:nold], data=v_old[0:nold, dc * PT : (dc + 1) * PT]), after)
        _pin_after(nisa.tensor_copy(dst=kTw[:, :, 0:nold], src=tpw[:, :, 0:nold]), after)

    if not LATE_SIDE_WORK:
        if ix is not None:
            b1idx.weights(ixs, xT)
        window_keys()
    # exchange buffer: [128, 256] fp32 words; core 0 sends the staged qr, core 1 the kv rows
    nqc = len(qparts)
    QW = nqc * T6
    QSW = (QW + 3) // 4
    if SWAP21_SPLIT:
        xbq = nl.ndarray((PT, QW + QSW + (QW + QSW) % 2), dtype=nl.int32, buffer=nl.sbuf)
        xbk = nl.ndarray((8, 256), dtype=nl.int32, buffer=nl.sbuf)
        QS = QW
    else:
        xb = nl.ndarray((PT, 256), dtype=nl.int32, buffer=nl.sbuf)
        xbq = xbk = xb
        QS = 64
    if pid == 0:
        qa16 = nl.ndarray((T6, q_lora_p), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(qa16[:, q_lora:q_lora_p], 0.0)
        evac(qa16, qa_p)
        if Q_NORM_T:
            qrq, qrs = norm_stage(qa16, T6, q_lora, q_lora_p, q_norm_w, wT=pre_w.get("q"))
        else:
            qnw = bcast_rows(q_norm_w, q_lora, T6, nl.bfloat16)
            qr16 = nl.ndarray((T6, q_lora_p), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(qr16[:, q_lora:q_lora_p], 0.0)
            norm_rows_sb(qa16, T6, q_lora, qnw, qr16)
            dbg("qa16", qa16[:, 0:q_lora], T6, q_lora, pid)
            dbg("qr16", qr16[:, 0:q_lora], T6, q_lora, pid)
            qrq, qrs = stage_rows(qr16, T6, q_lora_p, T6)
        nisa.tensor_copy(dst=xbq[:, 0:QW], src=qrq.view(nl.int32))
        ANCH["side"] = nisa.tensor_copy(dst=xbq[:, QS : QS + QSW].view(nl.uint8)[:, 0:QW], src=qrs)
    else:
        kv16 = nl.ndarray((T6, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
        evac(kv16, kv_p)
        kvw = pre_w["kv"] if "kv" in pre_w else bcast_rows(kv_norm_w, HD, T6, nl.bfloat16)
        kvn = nl.ndarray((T6, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
        norm_rows_sb(kv16, T6, HD, kvw, kvn)
        rope_cols(kvn, T6, HD - ROPE_DIM, HD, 1, cs6, inverse=False)
        kvq = qdq_fp8(kvn, T6, HD) if QDQ_FAST else _qdq_fp8(kvn, T6, HD)
        dbg("kv16", kv16, T6, HD, pid)
        dbg("kvq", kvq, T6, HD, pid)
        # this step's rows into the cache (the input, mutated in place)
        wsl = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=wsl[0:T6, :], src=write_slots[0:T6, 0:1])
        nisa.dma_copy(dst=window_kv_cache.ap(pattern=[[HD, T6], [1, HD]], vector_offset=wsl[0:T6, 0:1], indirect_dim=0),
                      src=kvq[0:T6, :])
        ANCH["side"] = nisa.tensor_copy(dst=xbk[0:T6, :].view(nl.bfloat16), src=kvq[0:T6, :])
    if after_side is not None:
        after_side()
    if SWAP21_SPLIT:
        xrq = swap(pid, xbq, 21, bufs=B0)
        xrk = swap(pid, xbk, 24, bufs=B0)
    else:
        xrq = xrk = swap(pid, xb, 21, bufs=B0)
    if LATE_SIDE_WORK:
        # (behind the q / kv path, in the exchange's wait: ahead of it they held the DVE / PE queues;
        # scheduled after the path's last instruction -- the window gather lands late)
        if ix is not None:
            b1idx.weights(ixs, xT)
        window_keys(after=ANCH.get("side"))
    if pid == 0:
        kvq = nl.ndarray((T6, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kvq, src=xrk[0:T6, :].view(nl.bfloat16))
        qrq_u, qrs_u = qrq, qrs
    else:
        qrq_i = nl.ndarray((PT, nqc * T6), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qrq_i, src=xrq[:, 0:QW])
        qrq_u = qrq_i.view(FP8_X4)
        qrs_u = nl.ndarray((PT, nqc * T6), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qrs_u, src=xrq[:, QS : QS + QSW].view(nl.uint8)[:, 0:QW])
    if ix is not None:
        # the index scores (both cores: half of the keys each) into their all-reduce
        b1idx.scores(pid, ixs, qrq_u, qrs_u, qparts, cs6, ix["rg"], ix["barrier"], freqs=freqs, row_pos=row_pos)
    # the new keys: V rows on partitions 0..5, K^T into the window table's last 6 columns
    v_new = kvq
    tpn = nl.ndarray((PT, DC, T6), dtype=nl.bfloat16, buffer=nl.psum)
    for dc in range(DC):
        nisa.nc_transpose(dst=tpn[:, dc, :], data=v_new[0:T6, dc * PT : (dc + 1) * PT])
    nisa.tensor_copy(dst=kTw[:, :, nold:nwin], src=tpn)

    # ---------------------------------------------------------------- q = qr @ wq_b (my 4 heads)
    q_p = mx_gemm(qrq_u, qrs_u, T6, T6, qparts, *wqb_t, 0, HL * HD)
    q16 = nl.ndarray((T6, HL * HD), dtype=nl.bfloat16, buffer=nl.sbuf)
    evac(q16, q_p)
    cs6q = cs6
    dbg("q16pre", q16, T6, HL * HD, pid)
    rope_cols(q16, T6, HD - ROPE_DIM, HD, HL, cs6q, inverse=False)
    dbg("q16", q16, T6, HL * HD, pid)
    dbg("kvrecv", v_new, T6, HD, pid)
    # q^T: qT[d, dc, h, t] (my heads, all rows)
    tq = nl.ndarray((PT, DC, HL, T6), dtype=nl.bfloat16, buffer=nl.psum)
    for h in range(HL):
        for dc in range(DC):
            nisa.nc_transpose(dst=tq[:, dc, h, :], data=q16[0:T6, h * HD + dc * PT : h * HD + (dc + 1) * PT])
    qT = nl.ndarray((PT, DC, HL, T6), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qT, src=tq)
    # the other core's rows of my heads -> it; its heads of my rows <- it
    oth = 1 - pid
    qs = nl.ndarray((PT, DC, HL, RH), dtype=nl.bfloat16, buffer=nl.sbuf)
    a1 = nisa.tensor_copy(dst=qs, src=qT[:, :, :, oth * RH : oth * RH + RH])
    # (background steps after this core's q path: the q exchange's wait; the exchange issued first --
    # GpSimd runs it and the steps' GpSimd work, in order)
    if SWAP_FIRST:
        qo = swap(pid, qs, 22, bufs=B0)
        tk(TICKS["q"], after=a1)
    else:
        tk(TICKS["q"], after=a1)
        qo = swap(pid, qs, 22, bufs=B0)
    # qall[d, dc, (r, h)] for my rows, all 8 heads (head h of the other core = its local h - 4*oth)
    qall = nl.ndarray((PT, DC, RH, HEADS), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qall[:, :, :, pid * HL : pid * HL + HL],
                     src=qT.ap(pattern=[[DC * HL * T6, PT], [HL * T6, DC], [1, RH], [T6, HL]], offset=pid * RH))
    nisa.tensor_copy(dst=qall[:, :, :, oth * HL : oth * HL + HL],
                     src=qo.ap(pattern=[[DC * HL * RH, PT], [HL * RH, DC], [1, RH], [RH, HL]], offset=0))
    nq = RH * HEADS
    qall2 = qall.reshape((PT, DC, nq))
    # per-row masked copies (row r's 8 columns, the others zero) for the compressed keys
    if not window_only:
        qm = nl.ndarray((PT, RH, DC, nq), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(qm, 0.0)
        for r in range(RH):
            nisa.tensor_copy(dst=qm[:, r, :, r * HEADS : (r + 1) * HEADS], src=qall2[:, :, r * HEADS : (r + 1) * HEADS])

    if ix is not None:
        # this core's rows' top-k (-> topk_in) and the attended keys from it (for the members after)
        vc, kTc, cb24 = B[gkey] = b1idx.topk_tables(pid, ixs, compress_kv_cache)

    # ---------------------------------------------------------------- scores
    sw = nl.ndarray((nq, nwin), dtype=nl.float32, buffer=nl.psum)
    for dc in range(DC):
        nisa.nc_matmul(dst=sw, stationary=qall2[:, dc, :], moving=kTw[:, dc, :], accumulate=(dc > 0))
    if topk:
        sc = nl.ndarray((nq, topk), dtype=nl.float32, buffer=nl.psum)
        i = 0
        for r in range(RH):
            for dc in range(DC):
                nisa.nc_matmul(dst=sc, stationary=qm[:, r, dc, :], moving=kTc[:, dc, r, :], accumulate=(i > 0))
                i += 1
    nk = nwin + topk
    s32 = nl.ndarray((nq, nk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=s32[:, 0:nwin], data=sw, op0=nl.multiply, operand0=scale, op1=nl.add,
                              operand1=wbias[0:nq, :])
    if topk:
        nisa.scalar_tensor_tensor(dst=s32[:, nwin:nk], data=sc, op0=nl.multiply, operand0=scale, op1=nl.add,
                                  operand1=cb24[0:nq, :])
    mx = nl.ndarray((nq, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mx, op=nl.max, data=s32, axis=(1,), keepdims=True)
    nmx = nl.ndarray((nq, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nmx, data=mx, op0=nl.multiply, operand0=-1.0)
    p32 = nl.ndarray((nq, nk), dtype=nl.float32, buffer=nl.sbuf)
    den = nl.ndarray((nq, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=p32, op=nl.exp, data=s32, bias=nmx, scale=1.0, reduce_op=nl.add, reduce_res=den)
    sk = nl.ndarray((nq, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sk, op=nl.exp, data=sink24[0:nq, :], bias=nmx)
    dsum = nl.ndarray((nq, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=dsum, data1=den, data2=sk, op=nl.add)
    rcp = nl.ndarray((nq, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rcp, data=dsum)
    p16 = nl.ndarray((nq, nk), dtype=nl.bfloat16, buffer=nl.sbuf)
    a2 = nisa.tensor_scalar(dst=p16, data=p32, op0=nl.multiply, operand0=rcp)
    tk(TICKS["p"], after=a2)
    dbg("s32", s32, nq, nk, pid)

    # ---------------------------------------------------------------- P^T, PV
    # window: keys 0..nold-1 (cached) and nold..nwin-1 (new); compressed: 4 tiles of 128
    pTw = nl.ndarray((PT, 2, nq), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(dst=pTw[0:nold, 0, :], data=p16[:, 0:nold])
    nisa.nc_transpose(dst=pTw[0:T6, 1, :], data=p16[:, nold:nwin])
    pTws = nl.ndarray((PT, 2, nq), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pTws[0:nold, 0, :], src=pTw[0:nold, 0, :])
    nisa.tensor_copy(dst=pTws[0:T6, 1, :], src=pTw[0:T6, 1, :])
    if ktc:
        pTc = nl.ndarray((PT, ktc, nq), dtype=nl.bfloat16, buffer=nl.psum)
        for c in range(ktc):
            nisa.nc_transpose(dst=pTc[:, c, :], data=p16[:, nwin + c * PT : nwin + (c + 1) * PT])
        pm = nl.ndarray((PT, RH, ktc, nq), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(pm, 0.0)
        for r in range(RH):
            nisa.tensor_copy(dst=pm[:, r, :, r * HEADS : (r + 1) * HEADS], src=pTc[:, :, r * HEADS : (r + 1) * HEADS])
    o_p = nl.ndarray((nq, HD), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=o_p, stationary=pTws[0:nold, 0, :], moving=v_old[0:nold, :], accumulate=False)
    nisa.nc_matmul(dst=o_p, stationary=pTws[0:T6, 1, :], moving=v_new[0:T6, :], accumulate=True)
    for r in range(RH):
        for c in range(ktc):
            nisa.nc_matmul(dst=o_p, stationary=pm[:, r, c, :], moving=vc[:, r * ktc + c, :], accumulate=True)
    o16 = nl.ndarray((nq, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o16, src=o_p)
    # inverse rope (per partition = (row, head): the row's position)
    dbg("o16pre", o16, nq, HD, pid)
    rope_cols(o16, nq, HD - ROPE_DIM, HD, 1, cs24, inverse=True)
    dbg("o16", o16, nq, HD, pid)

    # ---------------------------------------------------------------- z = o @ wo_a (my 3 rows)
    # MX operand: chunk h = head h's 512 values; xT[p, h, r, j] = o[(r, h), 4p + j]
    tpo = nl.ndarray((PT, 4, nq), dtype=nl.bfloat16, buffer=nl.psum)
    for j in range(4):
        nisa.nc_transpose(dst=tpo[:, j, :], data=o16.ap(pattern=[[HD, nq], [4, PT]], offset=j))
    RC = RH + 1
    oT = nl.ndarray((PT, HEADS, RC, 4), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(oT[:, :, RH, :], 0.0)
    nisa.tensor_copy(dst=oT[:, :, 0:RH, :], src=tpo.ap(pattern=[[4 * nq, PT], [1, HEADS], [HEADS, RH], [nq, 4]]))
    oq = nl.ndarray((PT, HEADS * RC), dtype=FP8_X4, buffer=nl.sbuf)
    os_ = nl.ndarray((PT, HEADS * RC), dtype=nl.uint8, buffer=nl.sbuf)
    a3 = nisa.quantize_mx(dst=oq, src=oT.reshape((PT, HEADS * RC * 4)), dst_scale=os_)
    tk(TICKS["o"], after=a3)
    z_p = mx_gemm(oq, os_, RC, RC, chunk_parts(g_in), *woa_t, 0, o_lora)
    z16 = nl.ndarray((RC, o_lora), dtype=nl.bfloat16, buffer=nl.sbuf)
    evac(z16, z_p)
    dbg("z16", z16, RC, o_lora, pid)
    zq, zs = stage_rows(z16, RC, o_lora, RC)
    # exchange the staged z rows: zall[p, kc, t] for the 6 rows
    nzc = o_lora // KC
    zb = nl.ndarray((PT, 2 * nzc * RC), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=zb[:, 0 : nzc * RC], src=zq.view(nl.int32))
    nisa.tensor_copy(dst=zb[:, nzc * RC : nzc * RC + nzc].view(nl.uint8)[:, 0 : nzc * RC], src=zs)
    zr = swap(pid, zb, 23, bufs=B0)
    zall = nl.ndarray((PT, nzc, T6), dtype=nl.int32, buffer=nl.sbuf)
    zalls = nl.ndarray((PT, nzc, T6), dtype=nl.uint8, buffer=nl.sbuf)
    zmine = zq.view(nl.int32).reshape((PT, nzc, RC))
    zoth = zr[:, 0 : nzc * RC].reshape((PT, nzc, RC))
    zs_mine = zs.reshape((PT, nzc, RC))
    zs_oth = zr[:, nzc * RC : nzc * RC + nzc].view(nl.uint8)[:, 0 : nzc * RC].reshape((PT, nzc, RC))
    nisa.tensor_copy(dst=zall[:, :, pid * RH : pid * RH + RH], src=zmine[:, :, 0:RH])
    nisa.tensor_copy(dst=zall[:, :, oth * RH : oth * RH + RH], src=zoth[:, :, 0:RH])
    nisa.tensor_copy(dst=zalls[:, :, pid * RH : pid * RH + RH], src=zs_mine[:, :, 0:RH])
    nisa.tensor_copy(dst=zalls[:, :, oth * RH : oth * RH + RH], src=zs_oth[:, :, 0:RH])

    # ---------------------------------------------------------------- wo_b: my half of the columns
    zq2 = zall.reshape((PT, nzc * T6)).view(FP8_X4)
    zs2 = zalls.reshape((PT, nzc * T6))
    out = nl.ndarray((T6, half), dtype=nl.float32, buffer=nl.sbuf)
    evac(out, mx_gemm(zq2, zs2, T6, T6, chunk_parts(o_lora), *wob_t, 0, half))
    nisa.dma_copy(dst=cc_src[0:T6, pid * half : pid * half + half], src=out, **_hw(hw0))
    dbg("ccsrc", out, T6, half, pid)
    dbg("qall", qall.reshape((PT, DC * nq)), PT, DC * nq, pid)
    return out
