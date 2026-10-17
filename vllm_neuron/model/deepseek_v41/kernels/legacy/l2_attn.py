"""`layers.2.attention` on 4 ranks, sequence-parallel: a rank owns a quarter of the **sequence**
and the whole of every weight, rather than a quarter of the heads. `wo_b` is therefore not
row-parallel — a rank writes *finished* output rows — so the ranks rejoin by `all_gather` rather
than `all_reduce`. A rank's query tiles are interleaved (global tile `WORLD_SIZE * j + rank` is
its local tile `j`), which is what lets one NEFF schedule all four ranks identically.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

SEQ = 8192
DIM = 5120
HEAD_DIM = 512
N_HEADS = 64
ROPE_DIM = 64
Q_LORA = 1280
O_LORA = 1024
O_GROUPS = 8
WINDOW = 128
RATIO = 2
CSEQ = SEQ // RATIO
INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_TOPK = 512
EPS = 1e-20
INDEX_SCORE_SCALE = 1.0 / 64.0
NEG_INF = -1.0e30
THR_FLOOR = -1.0e29

EXACT_SOFTMAX_SHIFT = False
SM_SCALE = HEAD_DIM ** -0.5
NEG_BOUND = -1.0e30
HEADS_PER_GROUP = N_HEADS // O_GROUPS

PT = 128
NT = 512
WIN_SPAN = 2 * PT

FP8_BLK = 32
FP8_MAX = 448.0
FP8_AMAX_FLOOR = 1e-4
QDQ_CHUNK = 1280

CORES = 2
MT = SEQ // PT
NG = CSEQ // NT

WORLD_SIZE = 4
SEQ_LOC = SEQ // WORLD_SIZE
MT_LOC = SEQ_LOC // PT
CSEQ_LOC = CSEQ // WORLD_SIZE
OUT_DTYPE = nl.bfloat16
CC_DTYPE = nl.bfloat16
CC_HALF = DIM // 2


def _core_runs(pid, cores=CORES):
    per = MT_LOC // (2 * cores)
    a = (pid * per, pid * per + per)
    b = (MT_LOC - (pid + 1) * per, MT_LOC - pid * per)
    gr = [a, b]
    return (tuple((t0 * PT, (t1 - t0) * PT) for t0, t1 in gr),
            tuple(t for t0, t1 in gr for t in range(t0, t1)))


def _rank_id():
    rid = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.register_store(rid, ncc.rank_id())
    return rid


def _rtile(t, cols, rid, tile, extra=0, rows=PT):
    return t.reshape((t.shape[0] // PT, PT, cols)).ap(
        pattern=[[cols, rows], [1, cols]],
        offset=(WORLD_SIZE * tile + extra) * PT * cols, scalar_offset=rid, indirect_dim=0)


def _rank_col(rid, rows, scale=1.0):
    r1 = nl.copy(rid.reshape((1, 1)), dtype=nl.float32)
    return nl.multiply(nl.broadcast_to(r1, (rows, 1)), scale)


def _shard0(t, rank_rows, cols, rid, row, rows, col=0, ncols=None):
    return t.reshape((WORLD_SIZE, rank_rows, cols)).ap(
        pattern=[[cols, rows], [1, cols if ncols is None else ncols]],
        offset=row * cols + col, scalar_offset=rid, indirect_dim=0)


def _shard1(t, rows_all, rank_cols, rid, row, rows, col=0, ncols=None):
    return t.reshape((rows_all * WORLD_SIZE, rank_cols)).ap(
        pattern=[[WORLD_SIZE * rank_cols, rows], [1, rank_cols if ncols is None else ncols]],
        offset=row * WORLD_SIZE * rank_cols + col, scalar_offset=rid, indirect_dim=0)


def _shard_src(t, shard, n, k, rid, row, rows, col=0, ncols=None):
    if shard == 0:
        return _shard0(t, n, k, rid, row, rows, col, ncols)
    return _shard1(t, n, k, rid, row, rows, col, ncols)


def _load_bcast(vec_hbm, n, rows):
    return nl.broadcast_to(nl.load(vec_hbm.reshape((1, n)), dtype=nl.float32), (rows, n))


def _load_bcast_shard(vec_hbm, n, rows, rid):
    v = nl.ndarray((1, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=v, src=vec_hbm.reshape((WORLD_SIZE, n)).ap(
        pattern=[[n, 1], [1, n]], offset=0, scalar_offset=rid, indirect_dim=0))
    return nl.broadcast_to(v, (rows, n))


def _rmsnorm(dst, tile, wb, n):
    sq = nl.ndarray(tile.shape, dtype=nl.float32, buffer=nl.sbuf)
    ss = nl.ndarray((tile.shape[0], 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=tile, reduce_op=nl.add, reduce_res=ss,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    ms = nl.multiply(ss, 1.0 / n)
    ms = nl.add(ms, EPS)
    y = nl.rsqrt(ms)
    y = nl.multiply(y, nl.subtract(1.5, nl.multiply(nl.multiply(ms, 0.5), nl.multiply(y, y))))
    nisa.scalar_tensor_tensor(dst=dst, data=tile, op0=nl.multiply, operand0=y,
                              op1=nl.multiply, operand1=wb)


def _pow2_ceil_log2(x):
    bits = nl.copy(x, dtype=nl.float32).view(nl.int32)
    exp = nl.bitwise_and(nl.right_shift(bits, 23), 0xFF)
    man = nl.bitwise_and(bits, 0x7FFFFF)
    bump = nl.copy(nl.greater(man, 0), dtype=nl.int32)
    e = nl.add(exp, bump)
    scale = nl.copy(nl.left_shift(e, 23)).view(nl.float32)
    inv = nl.copy(nl.left_shift(nl.subtract(254, e), 23)).view(nl.float32)
    return scale, inv, e


E4M3_TOP_BINADE = 248.0


def _cast_e4m3(v):
    safe = nl.minimum(nl.maximum(v, -240.0), 240.0)
    direct = nl.copy(nl.copy(safe, dtype=nl.float8_e4m3fn), dtype=nl.float32)
    halved = nl.copy(nl.copy(nl.multiply(v, 0.5), dtype=nl.float8_e4m3fn), dtype=nl.float32)
    return nl.where(nl.greater_equal(nl.abs(v), E4M3_TOP_BINADE),
                    nl.multiply(halved, 2.0), direct)


def _u8(shape):
    return nl.ndarray(shape, dtype=nl.uint8, buffer=nl.sbuf)


E4M3_TOP_M7 = 15 * 8


def _decode_e4m3(b):
    shape = b.shape
    bv = b.view(nl.float8_e4m3fn)
    m7 = _u8(shape)
    nisa.tensor_scalar(dst=m7, data=b, op0=nl.bitwise_and, operand0=0x7F)
    b2 = _u8(shape)
    nisa.tensor_scalar(dst=b2, data=b, op0=nl.subtract, operand0=8)
    top = _u8(shape)
    nisa.tensor_scalar(dst=top, data=m7, op0=nl.greater_equal, operand0=E4M3_TOP_M7)
    v = nl.ndarray(shape, dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=v, op=nl.copy, data=bv)
    v2 = nl.ndarray(shape, dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=v2, op=nl.copy, data=b2.view(nl.float8_e4m3fn), scale=2.0)
    nisa.tensor_copy_predicated(dst=v, src=v2, predicate=top)
    return v


def _block_amax(blocked, floor, k):
    ab = nl.ndarray(blocked.shape, dtype=blocked.dtype, buffer=nl.sbuf)
    nisa.activation(dst=ab, op=nl.abs, data=blocked)
    amax = nl.max(ab, axis=2, keepdims=True, dtype=nl.float32)
    out = nl.ndarray(amax.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=out, data=amax, op0=nl.maximum, operand0=floor,
                       op1=nl.multiply, operand1=k)
    return out


def _qdq_fp8(tile, rows, n):
    nb = n // FP8_BLK
    blocked = tile.reshape((rows, nb, FP8_BLK))
    scale, rcp, _ = _pow2_ceil_log2(_block_amax(blocked, FP8_AMAX_FLOOR, 2.0 / FP8_MAX))
    inv = rcp.broadcast(2, FP8_BLK)
    wide = scale.broadcast(2, FP8_BLK)
    q8 = nl.ndarray((rows, nb, FP8_BLK), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=q8, data1=blocked, data2=inv, op=nl.multiply)
    out = nl.ndarray((rows, nb, FP8_BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out, data1=q8, data2=wide, op=nl.multiply)
    return out.reshape((rows, n))


def _act_chunk(k):
    for c in (2560, QDQ_CHUNK, 1024, 512, 256, 128):
        if k % c == 0:
            return c
    return k


def _quantize_act(src_hbm, dst_hbm, runs, k):
    chunk = _act_chunk(k)
    for r0, rn in runs:
        for mb in nl.affine_range(rn // PT):
            for cb in nl.affine_range(k // chunk):
                t = nl.load(src_hbm[r0 + mb * PT:r0 + (mb + 1) * PT,
                                    cb * chunk:(cb + 1) * chunk], dtype=nl.float32)
                nl.store(dst_hbm[r0 + mb * PT:r0 + (mb + 1) * PT,
                                 cb * chunk:(cb + 1) * chunk],
                         value=_qdq_fp8(t, PT, chunk))


def _scale_selector():
    sper = PT // FP8_BLK
    pcol = nl.ndarray((sper, PT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=pcol, pattern=[[1, PT]], offset=0, channel_multiplier=0)
    srow = nl.ndarray((sper, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=srow, pattern=[[1, 1]], offset=0, channel_multiplier=FP8_BLK)
    d = nl.ndarray((sper, PT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=d, data=nl.copy(pcol, dtype=nl.float32), op0=nl.subtract,
                       operand0=nl.copy(srow, dtype=nl.float32))
    hit = nl.multiply(nl.copy(nl.greater_equal(d, 0.0), dtype=nl.float32),
                      nl.copy(nl.less(d, float(FP8_BLK)), dtype=nl.float32))
    return nl.copy(hit, dtype=nl.bfloat16)


def _dq_kchunk(k):
    for c in (2560, 2048, 1280, 1024, 512, FP8_BLK):
        if k % c == 0:
            return c
    return k


def _dequant_weight(w_u8, s_u8, dst_hbm, n, k, shard=None, rid=None):
    sper = PT // FP8_BLK
    kc = _dq_kchunk(k)
    cb = kc // FP8_BLK
    sel = _scale_selector()
    for nb in nl.affine_range(n // PT):
        r0 = nb * PT
        for c0 in nl.affine_range(k // kc):
            if shard is None:
                wraw = nl.load(w_u8[r0:r0 + PT, c0 * kc:(c0 + 1) * kc])
                sb = nl.load(s_u8[r0 // FP8_BLK:r0 // FP8_BLK + sper,
                                  c0 * cb:(c0 + 1) * cb], dtype=nl.int32)
            else:
                wraw = _u8((PT, kc))
                nisa.dma_copy(dst=wraw,
                              src=_shard_src(w_u8, shard, n, k, rid, r0, PT,
                                             col=c0 * kc, ncols=kc))
                sb = nl.ndarray((sper, cb), dtype=nl.int32, buffer=nl.sbuf)
                nisa.dma_copy(dst=sb,
                              src=_shard_src(s_u8, shard, n // FP8_BLK, k // FP8_BLK, rid,
                                             r0 // FP8_BLK, sper, col=c0 * cb, ncols=cb))
            wq = _decode_e4m3(wraw)
            ex = nl.ndarray((PT, cb), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=ex, stationary=sel,
                           moving=nl.copy(sb, dtype=nl.bfloat16), accumulate=False)
            sfw = nl.copy(nl.left_shift(nl.copy(ex, dtype=nl.int32), 23)).view(nl.float32)
            sf = nl.ndarray((PT, cb), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=sf, src=sfw)
            rep = sf.reshape((PT, cb, 1)).broadcast(2, FP8_BLK)
            out = nl.ndarray((PT, cb, FP8_BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=out, data1=wq.reshape((PT, cb, FP8_BLK)), data2=rep,
                               op=nl.multiply)
            nl.store(dst_hbm[r0:r0 + PT, c0 * kc:(c0 + 1) * kc],
                     value=out.reshape((PT, kc)))


FP4_MAX = 6.0


RNE_MAGIC = 12582912.0


def _rne_e2m1(v):
    bits = nl.copy(v, dtype=nl.float32).view(nl.int32)
    ef = nl.bitwise_and(nl.right_shift(bits, 23), 0xFF)
    sf = nl.maximum(nl.subtract(ef, 1), 126)
    step = nl.copy(nl.left_shift(sf, 23)).view(nl.float32)
    mag = nl.multiply(step, RNE_MAGIC)
    return nl.subtract(nl.add(v, mag), mag)


_E2M1_MID = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E2M1_STEP = (0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 2.0)
_E2M1_TIE_UP = (0, 1, 0, 1, 0, 1, 0)


def _quant_e2m1_by_threshold(av, s_narrow, shape):
    acc = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    for i in range(len(_E2M1_MID)):
        thr = nl.broadcast_to(nl.multiply(s_narrow, _E2M1_MID[i]), shape)
        if _E2M1_TIE_UP[i]:
            hit = nl.copy(nl.greater_equal(av, thr), dtype=nl.float32)
        else:
            hit = nl.copy(nl.greater(av, thr), dtype=nl.float32)
        if i == 0:
            nisa.tensor_scalar(dst=acc, data=hit, op0=nl.multiply, operand0=_E2M1_STEP[0])
        else:
            nisa.scalar_tensor_tensor(dst=acc, data=hit, op0=nl.multiply,
                                      operand0=_E2M1_STEP[i], op1=nl.add, operand1=acc)
    return acc


def _qdq_fp4(tile, rows, n, blk, e4m3_scale):
    nb = n // blk
    blocked = tile.reshape((rows, nb, blk))
    floor_v = 6.0 * (2.0 ** -9) if e4m3_scale else 6.0 * (2.0 ** -126)
    ab = nl.ndarray((rows, nb, blk), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=ab, op=nl.abs, data=blocked)
    amax = nl.maximum(nl.max(ab, axis=2, keepdims=True), floor_v)
    sixth = nl.multiply(amax, 1.0 / FP4_MAX)
    if not e4m3_scale:
        _, _, es = _pow2_ceil_log2(sixth)
    if e4m3_scale:
        scale = _cast_e4m3(sixth)
        wide = nl.broadcast_to(scale, (rows, nb, blk))
        mag = _quant_e2m1_by_threshold(ab, scale, (rows, nb, blk))
        deq = nl.multiply(nl.multiply(mag, wide), nl.sign(blocked))
        return nl.copy(deq.reshape((rows, n)), dtype=nl.bfloat16)
    else:
        es22 = nl.ndarray((rows, nb, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=es22, data=es, op0=nl.add, operand0=22)
        ef = nl.ndarray((rows, nb, blk), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ef, data=ab.view(nl.int32), op0=nl.right_shift, operand0=23)
        sfe = nl.ndarray((rows, nb, blk), dtype=nl.int32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=sfe, data=ef, op0=nl.add, operand0=22,
                                  op1=nl.maximum, operand1=es22.broadcast(2, blk))
        magb = nl.ndarray((rows, nb, blk), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=magb, data=sfe, op0=nl.left_shift, operand0=23,
                           op1=nl.bitwise_or, operand1=0x400000)
        mag = magb.view(nl.float32)
        up = nl.ndarray((rows, nb, blk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=up, data1=blocked, data2=mag, op=nl.add)
        out = nl.ndarray((rows, nb, blk), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=out, data1=up, data2=mag, op=nl.subtract)
        return out.reshape((rows, n))


def _qdq_fp4_rows(src_hbm, dst_hbm, runs, n, blk, e4m3_scale, heads=1, mstride=SEQ_LOC, hg=1):
    hgn = hg if heads % hg == 0 else 1
    for hb in nl.affine_range(heads // hgn):
        for r0, rn in runs:
            for mb in nl.affine_range(rn // PT):
                if hgn == 1:
                    row = hb * mstride + r0 + mb * PT
                    t = nl.load(src_hbm[row:row + PT, 0:n], dtype=nl.float32)
                    nl.store(dst_hbm[row:row + PT, 0:n],
                             value=_qdq_fp4(t, PT, n, blk, e4m3_scale))
                else:
                    t = nl.ndarray((PT, hgn, n), dtype=nl.float32, buffer=nl.sbuf)
                    for hi in nl.static_range(hgn):
                        row = (hb * hgn + hi) * mstride + r0 + mb * PT
                        nisa.tensor_copy(dst=t[0:PT, hi, 0:n],
                                         src=nl.load(src_hbm[row:row + PT, 0:n]))
                    q = _qdq_fp4(t.reshape((PT, hgn * n)), PT, hgn * n, blk, e4m3_scale)
                    q3 = q.reshape((PT, hgn, n))
                    for hi in nl.static_range(hgn):
                        row = (hb * hgn + hi) * mstride + r0 + mb * PT
                        nl.store(dst_hbm[row:row + PT, 0:n], value=q3[0:PT, hi, 0:n])


def _rope_qdq_fp4_rows(src_hbm, dst_hbm, freqs_hbm, runs, n, pos_stride, blk, e4m3_scale,
                       heads=1, mstride=SEQ_LOC, hg=1):
    half = ROPE_DIM // 2
    lo = n - ROPE_DIM
    rg = n // ROPE_DIM - 1
    hgn = hg if heads % hg == 0 else 1
    for hb in nl.affine_range(heads // hgn):
        for r0, rn in runs:
            for mb in nl.affine_range(rn // PT):
                p0 = (r0 + mb * PT) * pos_stride
                p1 = p0 + PT * pos_stride
                cosb = nl.ndarray((PT, 1, half), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=cosb[0:PT, 0, 0:half],
                                 src=nl.load(freqs_hbm[0][p0:p1:pos_stride, 0:half],
                                             dtype=nl.float32))
                sinb = nl.ndarray((PT, 1, half), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=sinb[0:PT, 0, 0:half],
                                 src=nl.load(freqs_hbm[1][p0:p1:pos_stride, 0:half],
                                             dtype=nl.float32))
                cos = cosb.broadcast(1, hgn)
                sin = sinb.broadcast(1, hgn)
                t = nl.ndarray((PT, hgn, n), dtype=nl.float32, buffer=nl.sbuf)
                for hi in nl.static_range(hgn):
                    row = (hb * hgn + hi) * mstride + r0 + mb * PT
                    nisa.tensor_copy(dst=t[0:PT, hi, 0:n],
                                     src=nl.load(src_hbm[row:row + PT, 0:n]))
                src = nl.ndarray((PT, hgn, half, 2), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=src.reshape((PT, hgn, ROPE_DIM)),
                                 src=t[0:PT, 0:hgn, lo:n])
                a = src[0:PT, 0:hgn, 0:half, 0]
                b = src[0:PT, 0:hgn, 0:half, 1]
                o = nl.ndarray((PT, hgn, half, 2), dtype=nl.float32, buffer=nl.sbuf)
                ac = nl.multiply(a, cos)
                bs = nl.multiply(b, sin)
                as_ = nl.multiply(a, sin)
                bc = nl.multiply(b, cos)
                nisa.tensor_copy(dst=o[0:PT, 0:hgn, 0:half, 0], src=nl.subtract(ac, bs))
                nisa.tensor_copy(dst=o[0:PT, 0:hgn, 0:half, 1], src=nl.add(as_, bc))
                nisa.tensor_copy(dst=t[0:PT, 0:hgn, lo:n],
                                 src=nl.copy(o.reshape((PT, hgn, ROPE_DIM)),
                                             dtype=nl.bfloat16))
                q = _qdq_fp4(t.reshape((PT, hgn * n)), PT, hgn * n, blk, e4m3_scale)
                q3 = q.reshape((PT, hgn, n))
                for hi in nl.static_range(hgn):
                    row = (hb * hgn + hi) * mstride + r0 + mb * PT
                    nl.store(dst_hbm[row:row + PT, 0:n], value=q3[0:PT, hi, 0:n])


def _rope_inplace(t_hbm, freqs_hbm, runs, width, pos_stride, inverse, heads=1, mstride=SEQ_LOC,
                  hg=1):
    half = ROPE_DIM // 2
    lo = width - ROPE_DIM
    hgn = hg if heads % hg == 0 else 1
    for r0, rn in runs:
        for mb in nl.affine_range(rn // PT):
            p0 = (r0 + mb * PT) * pos_stride
            p1 = p0 + PT * pos_stride
            cosb = nl.ndarray((PT, 1, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=cosb[0:PT, 0, 0:half],
                             src=nl.load(freqs_hbm[0][p0:p1:pos_stride, 0:half],
                                         dtype=nl.float32))
            sinb = nl.ndarray((PT, 1, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=sinb[0:PT, 0, 0:half],
                             src=nl.load(freqs_hbm[1][p0:p1:pos_stride, 0:half],
                                         dtype=nl.float32))
            cos = cosb.broadcast(1, hgn)
            sin = sinb.broadcast(1, hgn)
            for hb in nl.affine_range(heads // hgn):
                src = nl.ndarray((PT, hgn, ROPE_DIM), dtype=nl.float32, buffer=nl.sbuf)
                for hi in nl.static_range(hgn):
                    row = (hb * hgn + hi) * mstride + r0 + mb * PT
                    nisa.tensor_copy(dst=src[0:PT, hi, 0:ROPE_DIM],
                                     src=nl.load(t_hbm[row:row + PT, lo:width]))
                v = src.reshape((PT, hgn, half, 2))
                a = v[0:PT, 0:hgn, 0:half, 0]
                b = v[0:PT, 0:hgn, 0:half, 1]
                o = nl.ndarray((PT, hgn, half, 2), dtype=nl.float32, buffer=nl.sbuf)
                ac = nl.multiply(a, cos)
                bs = nl.multiply(b, sin)
                as_ = nl.multiply(a, sin)
                bc = nl.multiply(b, cos)
                if inverse:
                    nisa.tensor_copy(dst=o[0:PT, 0:hgn, 0:half, 0], src=nl.add(ac, bs))
                    nisa.tensor_copy(dst=o[0:PT, 0:hgn, 0:half, 1], src=nl.subtract(bc, as_))
                else:
                    nisa.tensor_copy(dst=o[0:PT, 0:hgn, 0:half, 0], src=nl.subtract(ac, bs))
                    nisa.tensor_copy(dst=o[0:PT, 0:hgn, 0:half, 1], src=nl.add(as_, bc))
                ob = nl.copy(o.reshape((PT, hgn, ROPE_DIM)), dtype=nl.bfloat16)
                for hi in nl.static_range(hgn):
                    row = (hb * hgn + hi) * mstride + r0 + mb * PT
                    nl.store(t_hbm[row:row + PT, lo:width], value=ob[0:PT, hi, 0:ROPE_DIM])


def _qdq_rows(src_hbm, dst_hbm, runs, n):
    for r0, rn in runs:
        for mb in nl.affine_range(rn // PT):
            t = nl.load(src_hbm[r0 + mb * PT:r0 + (mb + 1) * PT, 0:n], dtype=nl.float32)
            nl.store(dst_hbm[r0 + mb * PT:r0 + (mb + 1) * PT, 0:n], value=_qdq_fp8(t, PT, n))


GEMM_STAGE_BYTES = 120 * 1024
MGROUP_BYTES = 96 * 1024


def _mg_cap(runs):
    g = 0
    for _, rn in runs:
        g |= rn // PT
    return g & (-g)


def _col_blocks(n):
    blocks = []
    o = 0
    while n - o >= NT:
        blocks.append((o, NT))
        o += NT
    if n > o:
        blocks.append((o, n - o))
    return blocks


def _gemm_nest(k, nt, nbs, mmax, wb_total=None):
    xb = k * 2
    wb_one = (k // PT) * nt * 2
    if nbs == 1:
        return True, 1
    if (wb_total if wb_total is not None else wb_one * nbs) + xb <= GEMM_STAGE_BYTES:
        return True, 1
    mg = 1
    while mg * 2 <= mmax and (mg * 2) * xb <= MGROUP_BYTES:
        mg *= 2
    return False, mg


def _gemm_store(dst_hbm, row, col, w, acc, out_dtype, fp8):
    bt = nl.copy(acc, dtype=out_dtype)
    if fp8:
        nl.store(dst_hbm[row:row + PT, col:col + w],
                 value=_qdq_fp8(nl.copy(bt, dtype=nl.float32), PT, w))
    else:
        nl.store(dst_hbm[row:row + PT, col:col + w], value=bt)


def _gemm(x_hbm, w_hbm, dst_hbm, runs, k, n, out_dtype=nl.bfloat16,
          x_row=0, w_row=0, dst_col=0, fp8=False):
    ktiles = k // PT
    blocks = _col_blocks(n)
    w3 = w_hbm.reshape((w_hbm.shape[0], ktiles, PT))
    x3 = x_hbm.reshape((x_hbm.shape[0], ktiles, PT))
    wres, mg = _gemm_nest(k, max(w for _, w in blocks), len(blocks), _mg_cap(runs),
                          wb_total=ktiles * n * 2)

    if wres:
        wts = [nl.ndarray((PT, ktiles, w), dtype=nl.bfloat16, buffer=nl.sbuf)
               for _, w in blocks]
        for (o, w), wt in zip(blocks, wts):
            nisa.dma_transpose(dst=wt[0:PT, 0:ktiles, 0:w],
                               src=w3[w_row + o:w_row + o + w, 0:ktiles, 0:PT])
        for r0, rn in runs:
            for mb in nl.affine_range(rn // PT):
                xt = nl.ndarray((PT, ktiles, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_transpose(dst=xt[0:PT, 0:ktiles, 0:PT],
                                   src=x3[x_row + r0 + mb * PT:x_row + r0 + (mb + 1) * PT,
                                          0:ktiles, 0:PT])
                for (o, w), wt in zip(blocks, wts):
                    acc = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.psum)
                    for kb in nl.static_range(ktiles):
                        nisa.nc_matmul(dst=acc, stationary=xt[0:PT, kb, 0:PT],
                                       moving=wt[0:PT, kb, 0:w], accumulate=(kb > 0))
                    _gemm_store(dst_hbm, r0 + mb * PT, dst_col + o, w, acc, out_dtype, fp8)
        return

    for r0, rn in runs:
        for rg in nl.affine_range((rn // PT) // mg):
            xt = nl.ndarray((PT, mg, ktiles, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
            for i in nl.static_range(mg):
                nisa.dma_transpose(
                    dst=xt[0:PT, i, 0:ktiles, 0:PT],
                    src=x3[x_row + r0 + (rg * mg + i) * PT:
                           x_row + r0 + (rg * mg + i + 1) * PT, 0:ktiles, 0:PT])
            for o, w in blocks:
                wt = nl.ndarray((PT, ktiles, w), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_transpose(dst=wt[0:PT, 0:ktiles, 0:w],
                                   src=w3[w_row + o:w_row + o + w, 0:ktiles, 0:PT])
                for i in nl.static_range(mg):
                    acc = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.psum)
                    for kb in nl.static_range(ktiles):
                        nisa.nc_matmul(dst=acc, stationary=xt[0:PT, i, kb, 0:PT],
                                       moving=wt[0:PT, kb, 0:w], accumulate=(kb > 0))
                    mr = r0 + (rg * mg + i) * PT
                    _gemm_store(dst_hbm, mr, dst_col + o, w, acc, out_dtype, fp8)


def _gemm_multi(x_hbm, specs, runs, k):
    ktiles = k // PT
    x3 = x_hbm.reshape((x_hbm.shape[0], ktiles, PT))
    plan = []
    for w_hbm, dst_hbm, n, out_dtype in specs:
        w3 = w_hbm.reshape((w_hbm.shape[0], ktiles, PT))
        wts = []
        for o, w in _col_blocks(n):
            wt = nl.ndarray((PT, ktiles, w), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_transpose(dst=wt[0:PT, 0:ktiles, 0:w],
                               src=w3[o:o + w, 0:ktiles, 0:PT])
            wts.append((o, w, wt))
        plan.append((dst_hbm, out_dtype, wts))
    for r0, rn in runs:
        for mb in nl.affine_range(rn // PT):
            xt = nl.ndarray((PT, ktiles, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_transpose(dst=xt[0:PT, 0:ktiles, 0:PT],
                               src=x3[r0 + mb * PT:r0 + (mb + 1) * PT, 0:ktiles, 0:PT])
            for dst_hbm, out_dtype, wts in plan:
                for o, w, wt in wts:
                    acc = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.psum)
                    for kb in nl.static_range(ktiles):
                        nisa.nc_matmul(dst=acc, stationary=xt[0:PT, kb, 0:PT],
                                       moving=wt[0:PT, kb, 0:w], accumulate=(kb > 0))
                    nl.store(dst_hbm[r0 + mb * PT:r0 + (mb + 1) * PT, o:o + w],
                             value=nl.copy(acc, dtype=out_dtype))


def _load_rope_pair(freqs_hbm, row, hpg):
    half = ROPE_DIM // 2
    cosb = nl.ndarray((PT, 1, half), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cosb[0:PT, 0, 0:half],
                     src=nl.load(freqs_hbm[0][row:row + PT, 0:half], dtype=nl.float32))
    sinb = nl.ndarray((PT, 1, half), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sinb[0:PT, 0, 0:half],
                     src=nl.load(freqs_hbm[1][row:row + PT, 0:half], dtype=nl.float32))
    return cosb.broadcast(1, hpg), sinb.broadcast(1, hpg)


def _head_post(acc, nt, hdim, cos, sin, fp4):
    hpg = nt // hdim
    half = ROPE_DIM // 2
    lo = hdim - ROPE_DIM
    sb = nl.copy(acc, dtype=nl.bfloat16)
    if cos is None:
        return sb
    sb3 = sb.reshape((PT, hpg, hdim))
    wide = nl.ndarray((PT, hpg, hdim), dtype=nl.float32, buffer=nl.sbuf) if fp4 else None
    if fp4:
        nisa.tensor_copy(dst=wide.reshape((PT, nt)), src=sb)
    src = nl.ndarray((PT, hpg, half, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=src.reshape((PT, hpg, ROPE_DIM)), src=sb3[0:PT, 0:hpg, lo:hdim])
    a = src[0:PT, 0:hpg, 0:half, 0]
    b = src[0:PT, 0:hpg, 0:half, 1]
    o = nl.ndarray((PT, hpg, half, 2), dtype=nl.float32, buffer=nl.sbuf)
    ac = nl.multiply(a, cos)
    bs = nl.multiply(b, sin)
    as_ = nl.multiply(a, sin)
    bc = nl.multiply(b, cos)
    nisa.tensor_copy(dst=o[0:PT, 0:hpg, 0:half, 0], src=nl.subtract(ac, bs))
    nisa.tensor_copy(dst=o[0:PT, 0:hpg, 0:half, 1], src=nl.add(as_, bc))
    ob = nl.copy(o.reshape((PT, hpg, ROPE_DIM)), dtype=nl.bfloat16)
    if not fp4:
        nisa.tensor_copy(dst=sb3[0:PT, 0:hpg, lo:hdim], src=ob)
        return sb
    nisa.tensor_copy(dst=wide[0:PT, 0:hpg, lo:hdim], src=ob)
    return _qdq_fp4(wide.reshape((PT, nt)), PT, nt, fp4[0], fp4[1])


def _gemm_head_major(x_hbm, w_hbm, dst_hbm, runs, k, n, hdim, mstride=SEQ_LOC,
                     freqs_hbm=None, fp4=None):
    ktiles = k // PT
    hpg = max(1, NT // hdim)
    nt = hdim * hpg
    nbs = n // nt
    w3 = w_hbm.reshape((w_hbm.shape[0], ktiles, PT))
    x3 = x_hbm.reshape((x_hbm.shape[0], ktiles, PT))
    wres, mg = _gemm_nest(k, nt, nbs, _mg_cap(runs))

    if wres:
        wt = nl.ndarray((PT, nbs, ktiles, nt), dtype=nl.bfloat16, buffer=nl.sbuf)
        for nb in nl.static_range(nbs):
            nisa.dma_transpose(dst=wt[0:PT, nb, 0:ktiles, 0:nt],
                               src=w3[nb * nt:(nb + 1) * nt, 0:ktiles, 0:PT])
        for r0, rn in runs:
            for mb in nl.affine_range(rn // PT):
                xt = nl.ndarray((PT, ktiles, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_transpose(dst=xt[0:PT, 0:ktiles, 0:PT],
                                   src=x3[r0 + mb * PT:r0 + (mb + 1) * PT, 0:ktiles, 0:PT])
                cos = sin = None
                if freqs_hbm is not None:
                    cos, sin = _load_rope_pair(freqs_hbm, r0 + mb * PT, hpg)
                for nb in nl.affine_range(nbs):
                    acc = nl.ndarray((PT, nt), dtype=nl.float32, buffer=nl.psum)
                    for kb in nl.static_range(ktiles):
                        nisa.nc_matmul(dst=acc, stationary=xt[0:PT, kb, 0:PT],
                                       moving=wt[0:PT, nb, kb, 0:nt], accumulate=(kb > 0))
                    sb = _head_post(acc, nt, hdim, cos, sin, fp4)
                    for j in nl.static_range(hpg):
                        row = (nb * hpg + j) * mstride + r0 + mb * PT
                        nl.store(dst_hbm[row:row + PT, 0:hdim],
                                 value=sb[0:PT, j * hdim:(j + 1) * hdim])
        return

    for r0, rn in runs:
        for rg in nl.affine_range((rn // PT) // mg):
            xt = nl.ndarray((PT, mg, ktiles, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
            for i in nl.static_range(mg):
                nisa.dma_transpose(
                    dst=xt[0:PT, i, 0:ktiles, 0:PT],
                    src=x3[r0 + (rg * mg + i) * PT:r0 + (rg * mg + i + 1) * PT,
                           0:ktiles, 0:PT])
            cs = [(None, None)] * mg
            if freqs_hbm is not None:
                cs = [_load_rope_pair(freqs_hbm, r0 + (rg * mg + i) * PT, hpg)
                      for i in range(mg)]
            for nb in nl.affine_range(nbs):
                wt = nl.ndarray((PT, ktiles, nt), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_transpose(dst=wt[0:PT, 0:ktiles, 0:nt],
                                   src=w3[nb * nt:(nb + 1) * nt, 0:ktiles, 0:PT])
                for i in range(mg):
                    acc = nl.ndarray((PT, nt), dtype=nl.float32, buffer=nl.psum)
                    for kb in nl.static_range(ktiles):
                        nisa.nc_matmul(dst=acc, stationary=xt[0:PT, i, kb, 0:PT],
                                       moving=wt[0:PT, kb, 0:nt], accumulate=(kb > 0))
                    sb = _head_post(acc, nt, hdim, cs[i][0], cs[i][1], fp4)
                    for j in nl.static_range(hpg):
                        row = (nb * hpg + j) * mstride + r0 + (rg * mg + i) * PT
                        nl.store(dst_hbm[row:row + PT, 0:hdim],
                                 value=sb[0:PT, j * hdim:(j + 1) * hdim])


def _norm_rows(src_hbm, w_hbm, dst_hbm, runs, n):
    wb = _load_bcast(w_hbm, n, PT)
    for r0, rn in runs:
        for mb in nl.affine_range(rn // PT):
            t = nl.load(src_hbm[r0 + mb * PT:r0 + (mb + 1) * PT, 0:n], dtype=nl.float32)
            o = nl.ndarray((PT, n), dtype=nl.bfloat16, buffer=nl.sbuf)
            _rmsnorm(o, t, wb, n)
            nl.store(dst_hbm[r0 + mb * PT:r0 + (mb + 1) * PT, 0:n], value=o)


def _index_mask(index_q, ik_hbm, iw_hbm, msk_hbm, scr_hbm, thr_hbm, tiles, rid):
    kmax = max(j // 2 for j in tiles) + 1
    ikt_all = nl.ndarray((INDEX_DIM, kmax, NT), dtype=nl.bfloat16, buffer=nl.sbuf)
    for gb in nl.affine_range(kmax):
        nisa.tensor_copy(dst=ikt_all[0:INDEX_DIM, gb, 0:NT],
                         src=nl.load_transpose2d(ik_hbm[gb * NT:(gb + 1) * NT, 0:INDEX_DIM]))

    bpos_i = nl.ndarray((PT, NT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=bpos_i, pattern=[[1, NT]], offset=0, channel_multiplier=0)
    bpos = nl.copy(bpos_i, dtype=nl.float32)
    prow_i = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=prow_i, pattern=[[1, 1]], offset=1, channel_multiplier=1)
    halfr = nl.floor(nl.multiply(nl.copy(prow_i, dtype=nl.float32), 1.0 / RATIO))
    cbase = nl.add(halfr, _rank_col(rid, PT, float(PT // RATIO)))
    clim = [cbase, nl.add(cbase, float(NT // RATIO))]
    neg = nl.full((PT, NT), NEG_INF, dtype=nl.bfloat16)

    for j in tiles:
        gbn = j // 2 + 1
        q0 = j * PT
        wrow = nl.load(iw_hbm[q0:q0 + PT, 0:INDEX_HEADS], dtype=nl.float32)
        iqt = nl.ndarray((INDEX_DIM, INDEX_HEADS, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
        for hb in nl.affine_range(INDEX_HEADS):
            nisa.tensor_copy(dst=iqt[0:INDEX_DIM, hb, 0:PT],
                             src=nl.load_transpose2d(
                                 index_q[hb * SEQ_LOC + q0:hb * SEQ_LOC + q0 + PT,
                                         0:INDEX_DIM]))
        rows = nl.ndarray((PT, gbn, NT), dtype=nl.bfloat16, buffer=nl.sbuf)
        for gb in nl.static_range(gbn):
            tot = nl.ndarray((PT, NT), dtype=nl.float32, buffer=nl.sbuf)
            for hb in nl.static_range(INDEX_HEADS):
                ps = nl.ndarray((PT, NT), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=ps, stationary=iqt[0:INDEX_DIM, hb, 0:PT],
                               moving=ikt_all[0:INDEX_DIM, gb, 0:NT], accumulate=False)
                rect = nl.ndarray((PT, NT), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=rect, op=nl.relu, data=ps)
                if hb == 0:
                    nisa.tensor_scalar(dst=tot, data=rect, op0=nl.multiply,
                                       operand0=wrow[0:PT, hb:hb + 1])
                else:
                    nisa.scalar_tensor_tensor(dst=tot, data=rect, op0=nl.multiply,
                                              operand0=wrow[0:PT, hb:hb + 1], op1=nl.add,
                                              operand1=tot)
            tgt = rows[0:PT, gb, 0:NT]
            if gb == gbn - 1:
                reach = nl.ndarray((PT, NT), dtype=nl.uint8, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=reach, data=bpos, op0=nl.less, operand0=clim[j % 2])
                nisa.tensor_copy(dst=tgt,
                                 src=nl.where(reach, nl.copy(tot, dtype=nl.bfloat16), neg))
            else:
                nisa.tensor_copy(dst=tgt, src=tot)

        src_x = (gbn * NT) // 16
        nl.store(scr_hbm[q0:q0 + PT, 0:16, 0:src_x], value=rows.reshape((PT, 16, src_x)))
        for gi in nl.static_range(PT // 8):
            a0 = q0 + gi * 8
            tsrc = nl.ndarray((PT, src_x), dtype=nl.bfloat16, buffer=nl.sbuf)
            vdst = nl.ndarray((PT, INDEX_TOPK), dtype=nl.bfloat16, buffer=nl.sbuf)
            vk = vdst[0:PT, 0:INDEX_TOPK // 16]
            nisa.memset(dst=vk, value=-NEG_INF)
            idst = nl.ndarray((PT, INDEX_TOPK), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=tsrc, src=scr_hbm[a0:a0 + 8, 0:16, 0:src_x])
            nisa.topk(val_dst=vdst, idx_dst=idst, src=tsrc, n=gbn * NT)
            vmin = nl.min(vk, axis=1, keepdims=True)
            nisa.dma_copy(dst=thr_hbm[a0:a0 + 8, 0:16], src=vmin)
        thr = nl.maximum(nl.min(nl.load(thr_hbm[q0:q0 + PT, 0:16], dtype=nl.float32),
                                axis=1, keepdims=True), THR_FLOOR)
        for gb in nl.affine_range(gbn):
            mt_ = nl.ndarray((PT, NT), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mt_, data=rows[0:PT, gb, 0:NT],
                               op0=nl.greater_equal, operand0=thr)
            nl.store(msk_hbm[q0:q0 + PT, gb * NT:(gb + 1) * NT], value=mt_)


def _sparse_attention(q_hbm, kvp_hbm, ckv_hbm, msk_hbm, sink_hbm, freqs_hbm,
                      og_hbm, winl, tiles, rid):
    dt = HEAD_DIM // PT
    st = NT // PT
    cmax = (max(j // 2 for j in tiles) + 1) * NT

    jio_i = nl.ndarray((PT, WIN_SPAN), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=jio_i, pattern=[[1, WIN_SPAN]], offset=0, channel_multiplier=0)
    jio = nl.copy(jio_i, dtype=nl.float32)
    pio_i = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=pio_i, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    pio = nl.copy(pio_i, dtype=nl.float32)
    dio = nl.ndarray((PT, WIN_SPAN), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=dio, data=jio, op0=nl.subtract, operand0=pio)
    inwin = nl.multiply(nl.copy(nl.greater_equal(dio, 1.0), dtype=nl.float32),
                        nl.copy(nl.less_equal(dio, float(PT)), dtype=nl.float32))
    nonneg = nl.copy(nl.greater_equal(jio, float(PT)), dtype=nl.float32)
    rflag = nl.copy(nl.greater(_rank_col(rid, PT), 0.5), dtype=nl.float32)
    one = nl.full((PT, 1), 1.0, dtype=nl.float32)
    sinkb = _load_bcast(sink_hbm, N_HEADS, PT)

    ckv3 = ckv_hbm.reshape((CSEQ, dt, PT))
    q3 = q_hbm.reshape((N_HEADS * SEQ_LOC, dt, PT))
    winl3 = winl.reshape((MT_LOC * 2 * PT, dt, PT))
    ckv_t = nl.ndarray((PT, dt, cmax), dtype=nl.bfloat16, buffer=nl.sbuf)
    ckv_n = nl.ndarray((PT, cmax // PT, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    for gp in nl.affine_range(cmax // PT):
        nisa.tensor_copy(dst=ckv_n[0:PT, gp, 0:HEAD_DIM],
                         src=nl.load(ckv_hbm[gp * PT:(gp + 1) * PT, 0:HEAD_DIM]))
        nisa.dma_transpose(dst=ckv_t[0:PT, 0:dt, gp * PT:(gp + 1) * PT],
                           src=ckv3[gp * PT:(gp + 1) * PT, 0:dt, 0:PT])

    for j in tiles:
        for wt in nl.static_range(2):
            wtile = nl.ndarray((PT, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=wtile, src=_rtile(kvp_hbm, HEAD_DIM, rid, j, extra=wt))
            nl.store(winl[(j * 2 + wt) * PT:(j * 2 + wt + 1) * PT, 0:HEAD_DIM], value=wtile)

    for j in tiles:
        gbn = j // 2 + 1
        q0 = j * PT
        flag = rflag if j == 0 else one
        live = nl.ndarray((PT, WIN_SPAN), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=live, data=nonneg, op0=nl.maximum, operand0=flag)
        wadd = nl.multiply(nl.subtract(nl.multiply(live, inwin), 1.0), -NEG_BOUND)
        win_t = nl.ndarray((PT, dt, WIN_SPAN), dtype=nl.bfloat16, buffer=nl.sbuf)
        win_n = nl.ndarray((PT, 2, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        for wt in nl.static_range(2):
            w0 = (j * 2 + wt) * PT
            nisa.tensor_copy(dst=win_n[0:PT, wt, 0:HEAD_DIM],
                             src=nl.load(winl[w0:w0 + PT, 0:HEAD_DIM]))
            nisa.dma_transpose(dst=win_t[0:PT, 0:dt, wt * PT:(wt + 1) * PT],
                               src=winl3[w0:w0 + PT, 0:dt, 0:PT])
        cadd = nl.ndarray((PT, gbn * NT), dtype=nl.float32, buffer=nl.sbuf)
        for gb in nl.affine_range(gbn):
            mf = nl.copy(nl.load(msk_hbm[q0:q0 + PT, gb * NT:(gb + 1) * NT]),
                         dtype=nl.float32)
            nisa.tensor_scalar(dst=cadd[0:PT, gb * NT:(gb + 1) * NT], data=mf,
                               op0=nl.subtract, operand0=1.0,
                               op1=nl.multiply, operand1=-NEG_BOUND)
        rcos = nl.load(freqs_hbm[0][q0:q0 + PT, 0:ROPE_DIM // 2], dtype=nl.float32)
        rsin = nl.load(freqs_hbm[1][q0:q0 + PT, 0:ROPE_DIM // 2], dtype=nl.float32)

        for g in nl.affine_range(O_GROUPS):
            for hh in nl.affine_range(HEADS_PER_GROUP):
                h = g * HEADS_PER_GROUP + hh
                qt = nl.ndarray((PT, dt, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_transpose(dst=qt[0:PT, 0:dt, 0:PT],
                                   src=q3[h * SEQ_LOC + q0:h * SEQ_LOC + q0 + PT, 0:dt, 0:PT])
                mcol = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
                lwp = nl.ndarray((PT, WIN_SPAN), dtype=nl.float32, buffer=nl.psum)
                for dc in nl.static_range(dt):
                    nisa.nc_matmul(dst=lwp, stationary=qt[0:PT, dc, 0:PT],
                                   moving=win_t[0:PT, dc, 0:WIN_SPAN],
                                   accumulate=(dc > 0))
                lw = nl.ndarray((PT, WIN_SPAN), dtype=nl.float32, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(dst=lw, data=lwp, op0=nl.multiply,
                                          operand0=SM_SCALE, op1=nl.add, operand1=wadd)
                nisa.tensor_reduce(dst=mcol[0:PT, 0:1], op=nl.max, data=lw, axis=[1])
                lcs = nl.ndarray((PT, gbn, NT), dtype=nl.float32, buffer=nl.sbuf)
                for gb in nl.affine_range(gbn):
                    lcp = nl.ndarray((PT, NT), dtype=nl.float32, buffer=nl.psum)
                    for dc in nl.static_range(dt):
                        nisa.nc_matmul(dst=lcp, stationary=qt[0:PT, dc, 0:PT],
                                       moving=ckv_t[0:PT, dc, gb * NT:(gb + 1) * NT],
                                       accumulate=(dc > 0))
                    nisa.scalar_tensor_tensor(dst=lcs[0:PT, gb, 0:NT], data=lcp,
                                              op0=nl.multiply, operand0=SM_SCALE,
                                              op1=nl.add,
                                              operand1=cadd[0:PT, gb * NT:(gb + 1) * NT])
                m = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=m, data=mcol[0:PT, 0:1], op0=nl.maximum,
                                   operand0=0.0)
                negm = nl.multiply(m, -1.0)
                acc = nl.ndarray((PT, HEAD_DIM), dtype=nl.float32, buffer=nl.psum)
                pw = nl.ndarray((PT, WIN_SPAN), dtype=nl.bfloat16, buffer=nl.sbuf)
                dcol = nl.ndarray((PT, gbn + 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=pw, op=nl.exp, data=lw, bias=negm, scale=1.0,
                                reduce_op=nl.add, reduce_res=dcol[0:PT, 0:1],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
                pwt = nl.ndarray((PT, 2, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_transpose(dst=pwt[0:PT, 0:2, 0:PT],
                                   src=pw.reshape((PT, 2, PT))[0:PT, 0:2, 0:PT])
                for wt in nl.static_range(2):
                    nisa.nc_matmul(dst=acc, stationary=pwt[0:PT, wt, 0:PT],
                                   moving=win_n[0:PT, wt, 0:HEAD_DIM],
                                   accumulate=(wt > 0))
                for gb in nl.affine_range(gbn):
                    pc = nl.ndarray((PT, NT), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.activation(dst=pc, op=nl.exp, data=lcs[0:PT, gb, 0:NT],
                                    bias=negm, scale=1.0,
                                    reduce_op=nl.add,
                                    reduce_res=dcol[0:PT, gb + 1:gb + 2],
                                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
                    pct = nl.ndarray((PT, st, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.dma_transpose(dst=pct[0:PT, 0:st, 0:PT],
                                       src=pc.reshape((PT, st, PT))[0:PT, 0:st, 0:PT])
                    for sb in nl.static_range(st):
                        nisa.nc_matmul(dst=acc, stationary=pct[0:PT, sb, 0:PT],
                                       moving=ckv_n[0:PT, gb * st + sb, 0:HEAD_DIM],
                                       accumulate=True)
                esink = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=esink, op=nl.exp, data=sinkb[0:PT, h:h + 1],
                                bias=negm, scale=1.0)
                den = nl.add(nl.sum(dcol, axis=1, keepdims=True), esink)
                rden = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.reciprocal(dst=rden, data=den)
                ot = nl.ndarray((PT, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=ot, data=acc, op0=nl.multiply, operand0=rden)
                rv = nl.copy(ot[0:PT, HEAD_DIM - ROPE_DIM:HEAD_DIM],
                             dtype=nl.float32).reshape((PT, ROPE_DIM // 2, 2))
                ra = rv[0:PT, 0:ROPE_DIM // 2, 0]
                rb = rv[0:PT, 0:ROPE_DIM // 2, 1]
                rot = nl.ndarray((PT, ROPE_DIM // 2, 2), dtype=nl.float32,
                                 buffer=nl.sbuf)
                nisa.tensor_tensor(dst=rot[0:PT, 0:ROPE_DIM // 2, 0],
                                   data1=nl.multiply(ra, rcos),
                                   data2=nl.multiply(rb, rsin), op=nl.add)
                nisa.tensor_tensor(dst=rot[0:PT, 0:ROPE_DIM // 2, 1],
                                   data1=nl.multiply(rb, rcos),
                                   data2=nl.multiply(ra, rsin), op=nl.subtract)
                nisa.tensor_copy(dst=ot[0:PT, HEAD_DIM - ROPE_DIM:HEAD_DIM],
                                 src=rot.reshape((PT, ROPE_DIM)))
                nl.store(og_hbm[g * SEQ_LOC + q0:g * SEQ_LOC + q0 + PT,
                                hh * HEAD_DIM:(hh + 1) * HEAD_DIM], value=ot)


def _gather_index(rows):
    pio = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=pio, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    pf = nl.copy(pio, dtype=nl.float32)
    hi = nl.copy(nl.greater_equal(pf, float(CSEQ_LOC // 16)), dtype=nl.float32)
    base = nl.add(pf, nl.multiply(hi, float(CSEQ_LOC - CSEQ_LOC // 16)))
    gf = nl.ndarray((PT, rows // PT), dtype=nl.float32, buffer=nl.sbuf)
    for b in range(rows // PT):
        off = 2 * CSEQ_LOC * (b % 2) + (CSEQ_LOC // 16) * (b // 2)
        nisa.tensor_scalar(dst=gf[0:PT, b:b + 1], data=base, op0=nl.add, operand0=float(off))
    return nl.copy(gf, dtype=nl.uint32)


@nki.jit
def _impl(hidden_states, layers_2_attn_attn_sink, layers_2_attn_compress_kv_cache, layers_2_attn_compressor_kv_state, layers_2_attn_compressor_norm_weight, layers_2_attn_compressor_score_state, layers_2_attn_compressor_wgate_weight, layers_2_attn_compressor_wkv_weight, layers_2_attn_freqs_cis, layers_2_attn_indexer_k_cache, layers_2_attn_indexer_k_norm_weight, layers_2_attn_indexer_weights_proj_weight, layers_2_attn_indexer_wk_weight, layers_2_attn_indexer_wq_b_scale, layers_2_attn_indexer_wq_b_weight, layers_2_attn_kv_norm_weight, layers_2_attn_q_norm_weight, layers_2_attn_window_kv_cache, layers_2_attn_wkv_scale, layers_2_attn_wkv_weight, layers_2_attn_wo_a_scale, layers_2_attn_wo_a_weight, layers_2_attn_wo_b_scale, layers_2_attn_wo_b_weight, layers_2_attn_wq_a_scale, layers_2_attn_wq_a_weight, layers_2_attn_wq_b_scale, layers_2_attn_wq_b_weight, layers_2_attn_norm_weight, replica_group):
    pid = nl.program_id(0)
    QR, TILES = _core_runs(pid)
    CR = tuple((r0 // RATIO, rn // RATIO) for r0, rn in QR)
    ATILES = sorted(TILES, reverse=True)

    x_hbm = hidden_states.reshape((SEQ, DIM))
    out = nl.ndarray((1, SEQ, DIM), dtype=OUT_DTYPE, buffer=nl.shared_hbm)
    out2 = out.reshape((SEQ, DIM))
    cc_src_a = nl.ndarray((SEQ_LOC, CC_HALF), dtype=CC_DTYPE, buffer=nl.shared_hbm,
                          name="cc_src_a")
    cc_dst_a = nl.ndarray((SEQ, CC_HALF), dtype=CC_DTYPE, buffer=nl.shared_hbm, name="cc_dst_a")
    cc_src_b = nl.ndarray((SEQ_LOC, CC_HALF), dtype=CC_DTYPE, buffer=nl.shared_hbm,
                          name="cc_src_b")
    cc_dst_b = nl.ndarray((SEQ, CC_HALF), dtype=CC_DTYPE, buffer=nl.shared_hbm, name="cc_dst_b")
    kv_src = nl.ndarray((SEQ_LOC, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                        name="kv_src")
    kv_dst = nl.ndarray((SEQ, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kv_dst")
    ck_src = nl.ndarray((CSEQ_LOC, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                        name="ck_src")
    ck_dst = nl.ndarray((CSEQ, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                        name="ck_dst")
    ik_src = nl.ndarray((CSEQ_LOC, INDEX_DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                        name="ik_src")
    ik_dst = nl.ndarray((CSEQ, INDEX_DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                        name="ik_dst")

    rid = _rank_id()

    wq_a = nl.ndarray((Q_LORA, DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="wq_a")
    wkv = nl.ndarray((HEAD_DIM, DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="wkv")
    iwqb = nl.ndarray((INDEX_HEADS * INDEX_DIM, Q_LORA), dtype=nl.bfloat16,
                      buffer=nl.shared_hbm, name="iwqb")
    wqb = nl.ndarray((N_HEADS * HEAD_DIM, Q_LORA), dtype=nl.bfloat16,
                     buffer=nl.shared_hbm, name="wqb")
    wo_a = nl.ndarray((O_GROUPS * O_LORA, HEADS_PER_GROUP * HEAD_DIM), dtype=nl.bfloat16,
                      buffer=nl.shared_hbm, name="wo_a")
    wo_b = nl.ndarray((DIM, O_GROUPS * O_LORA), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                      name="wo_b")
    WQB_N = N_HEADS * HEAD_DIM // WORLD_SIZE
    WOA_N = O_GROUPS * O_LORA // WORLD_SIZE
    WOB_N = DIM // WORLD_SIZE
    wqb_src = nl.ndarray((WQB_N, Q_LORA), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                         name="wqb_src")
    woa_src = nl.ndarray((WOA_N, HEADS_PER_GROUP * HEAD_DIM), dtype=nl.bfloat16,
                         buffer=nl.shared_hbm, name="woa_src")
    wob_src = nl.ndarray((WOB_N, O_GROUPS * O_LORA), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                         name="wob_src")
    if pid == 0:
        _dequant_weight(layers_2_attn_wq_b_weight, layers_2_attn_wq_b_scale, wqb_src,
                        WQB_N, Q_LORA, shard=0, rid=rid)
        _dequant_weight(layers_2_attn_wo_a_weight, layers_2_attn_wo_a_scale, woa_src,
                        WOA_N, HEADS_PER_GROUP * HEAD_DIM, shard=0, rid=rid)
        _dequant_weight(layers_2_attn_wkv_weight, layers_2_attn_wkv_scale, wkv, HEAD_DIM, DIM)
    else:
        _dequant_weight(layers_2_attn_wo_b_weight, layers_2_attn_wo_b_scale, wob_src,
                        WOB_N, O_GROUPS * O_LORA, shard=0, rid=rid)
        _dequant_weight(layers_2_attn_wq_a_weight, layers_2_attn_wq_a_scale, wq_a, Q_LORA, DIM)
        _dequant_weight(layers_2_attn_indexer_wq_b_weight, layers_2_attn_indexer_wq_b_scale,
                        iwqb, INDEX_HEADS * INDEX_DIM, Q_LORA)
    if CORES > 1:
        for wbuf in (wq_a, wkv, iwqb, wqb_src, woa_src, wob_src):
            nisa.core_barrier(data=wbuf, cores=tuple(range(CORES)))

    xloc = nl.ndarray((SEQ_LOC, DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    half = ROPE_DIM // 2
    fcos = nl.ndarray((SEQ_LOC, half), dtype=nl.float32, buffer=nl.private_hbm)
    fsin = nl.ndarray((SEQ_LOC, half), dtype=nl.float32, buffer=nl.private_hbm)
    floc = (fcos, fsin)
    fg = layers_2_attn_freqs_cis.reshape((layers_2_attn_freqs_cis.shape[0], ROPE_DIM))
    for j in TILES:
        xt = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt, src=_rtile(x_hbm, DIM, rid, j))
        nl.store(xloc[j * PT:(j + 1) * PT, 0:DIM], value=xt)
        ft = nl.ndarray((PT, half, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ft.reshape((PT, ROPE_DIM)), src=_rtile(fg, ROPE_DIM, rid, j))
        fs = nl.ndarray((PT, 2, half), dtype=nl.float32, buffer=nl.sbuf)
        for ci in range(2):
            nisa.tensor_copy(dst=fs[0:PT, ci, 0:half], src=ft[0:PT, 0:half, ci])
        nl.store(fcos[j * PT:(j + 1) * PT, 0:half], value=fs[0:PT, 0, 0:half])
        nl.store(fsin[j * PT:(j + 1) * PT, 0:half], value=fs[0:PT, 1, 0:half])

    xn = nl.ndarray((SEQ_LOC, DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    _norm_rows(xloc, layers_2_attn_norm_weight, xn, QR, DIM)
    xn_q = nl.ndarray((SEQ_LOC, DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    _quantize_act(xn, xn_q, QR, DIM)

    qa = nl.ndarray((SEQ_LOC, Q_LORA), dtype=nl.bfloat16, buffer=nl.private_hbm)
    _gemm(xn_q, wq_a, qa, QR, DIM, Q_LORA)
    qr = nl.ndarray((SEQ_LOC, Q_LORA), dtype=nl.bfloat16, buffer=nl.private_hbm)
    _norm_rows(qa, layers_2_attn_q_norm_weight, qr, QR, Q_LORA)

    kva = nl.ndarray((SEQ_LOC, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    _gemm(xn_q, wkv, kva, QR, DIM, HEAD_DIM)
    kvr = nl.ndarray((SEQ_LOC, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    _norm_rows(kva, layers_2_attn_kv_norm_weight, kvr, QR, HEAD_DIM)
    _rope_inplace(kvr, floc, QR, HEAD_DIM, 1, False)
    _qdq_rows(kvr, kv_src, QR, HEAD_DIM)

    ckv = nl.ndarray((SEQ_LOC, HEAD_DIM), dtype=nl.float32, buffer=nl.private_hbm)
    cgt = nl.ndarray((SEQ_LOC, HEAD_DIM), dtype=nl.float32, buffer=nl.private_hbm)
    iwa = nl.ndarray((SEQ_LOC, INDEX_HEADS), dtype=nl.bfloat16, buffer=nl.private_hbm)
    _gemm_multi(xn,
                ((layers_2_attn_compressor_wkv_weight, ckv, HEAD_DIM, nl.float32),
                 (layers_2_attn_compressor_wgate_weight, cgt, HEAD_DIM, nl.float32),
                 (layers_2_attn_indexer_weights_proj_weight, iwa, INDEX_HEADS, nl.bfloat16)),
                QR, DIM)
    pooled = nl.ndarray((CSEQ_LOC, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    for c0, cn in CR:
        for gb in nl.affine_range(cn // PT):
            r0 = (c0 + gb * PT) * RATIO
            r1 = r0 + PT * RATIO
            g0 = nl.load(cgt[r0:r1:RATIO, 0:HEAD_DIM], dtype=nl.float32)
            g1 = nl.load(cgt[r0 + 1:r1:RATIO, 0:HEAD_DIM], dtype=nl.float32)
            v0 = nl.load(ckv[r0:r1:RATIO, 0:HEAD_DIM], dtype=nl.float32)
            v1 = nl.load(ckv[r0 + 1:r1:RATIO, 0:HEAD_DIM], dtype=nl.float32)
            mx = nl.maximum(g0, g1)
            e0 = nl.exp(nl.subtract(g0, mx))
            e1 = nl.exp(nl.subtract(g1, mx))
            rz = nl.ndarray((PT, HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=rz, data=nl.add(e0, e1))
            p0 = nl.multiply(e0, rz)
            p1 = nl.multiply(e1, rz)
            acc = nl.add(nl.multiply(v0, p0), nl.multiply(v1, p1))
            nl.store(pooled[c0 + gb * PT:c0 + (gb + 1) * PT, 0:HEAD_DIM],
                     value=nl.copy(acc, dtype=nl.bfloat16))
    latent = nl.ndarray((CSEQ_LOC, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    _norm_rows(pooled, layers_2_attn_compressor_norm_weight, latent, CR, HEAD_DIM)

    ika = nl.ndarray((CSEQ_LOC, INDEX_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    _gemm(latent, layers_2_attn_indexer_wk_weight, ika, CR, HEAD_DIM, INDEX_DIM)
    ikn = nl.ndarray((CSEQ_LOC, INDEX_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    _norm_rows(ika, layers_2_attn_indexer_k_norm_weight, ikn, CR, INDEX_DIM)
    _rope_qdq_fp4_rows(ikn, ik_src, floc, CR, INDEX_DIM, RATIO, 32, False)
    _rope_qdq_fp4_rows(latent, ck_src, floc, CR, HEAD_DIM, RATIO, 16, True)

    qr_q = nl.ndarray((SEQ_LOC, Q_LORA), dtype=nl.bfloat16, buffer=nl.private_hbm)
    _quantize_act(qr, qr_q, QR, Q_LORA)
    index_q = nl.ndarray((INDEX_HEADS * SEQ_LOC, INDEX_DIM), dtype=nl.bfloat16,
                         buffer=nl.private_hbm)
    _gemm_head_major(qr_q, iwqb, index_q, QR, Q_LORA, INDEX_HEADS * INDEX_DIM, INDEX_DIM,
                     freqs_hbm=floc, fp4=(32, False))

    iw = nl.ndarray((SEQ_LOC, INDEX_HEADS), dtype=nl.bfloat16, buffer=nl.private_hbm)
    for r0, rn in QR:
        for mb in nl.affine_range(rn // PT):
            t = nl.load(iwa[r0 + mb * PT:r0 + (mb + 1) * PT, 0:INDEX_HEADS], dtype=nl.float32)
            nl.store(iw[r0 + mb * PT:r0 + (mb + 1) * PT, 0:INDEX_HEADS],
                     value=nl.copy(nl.multiply(t, INDEX_SCORE_SCALE), dtype=nl.bfloat16))

    if CORES > 1:
        for cbuf in (kv_src, ck_src, ik_src):
            nisa.core_barrier(data=cbuf, cores=tuple(range(CORES)))
    ncc.all_gather(srcs=[kv_src, ck_src, ik_src, wqb_src, woa_src, wob_src],
                   dsts=[kv_dst, ck_dst, ik_dst, wqb, wo_a, wo_b],
                   replica_group=replica_group, collective_dim=0)

    kvp = nl.ndarray((SEQ + PT, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    nl.store(kvp[0:PT, 0:HEAD_DIM], value=nl.zeros((PT, HEAD_DIM), dtype=nl.bfloat16))
    for j in range(MT_LOC):
        for s in range(WORLD_SIZE):
            g = WORLD_SIZE * j + s
            nisa.dma_copy(dst=kvp[(g + 1) * PT:(g + 2) * PT, 0:HEAD_DIM],
                          src=kv_dst[s * SEQ_LOC + j * PT:s * SEQ_LOC + (j + 1) * PT,
                                     0:HEAD_DIM])

    compress_kv = nl.ndarray((CSEQ, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                             name="compress_kv")
    index_k = nl.ndarray((CSEQ, INDEX_DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                         name="index_k")
    gidx = _gather_index(CSEQ)
    nblk = CSEQ // PT
    for b in range(pid * (nblk // CORES), (pid + 1) * (nblk // CORES)):
        tb = nl.ndarray((PT, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=tb,
                      src=ck_dst.ap(pattern=[[HEAD_DIM, PT], [1, HEAD_DIM]],
                                    vector_offset=gidx[0:PT, b:b + 1], indirect_dim=0),
                      dge_mode=nisa.dge_mode.swdge)
        nl.store(compress_kv[b * PT:(b + 1) * PT, 0:HEAD_DIM], value=tb)
        ti = nl.ndarray((PT, INDEX_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=ti,
                      src=ik_dst.ap(pattern=[[INDEX_DIM, PT], [1, INDEX_DIM]],
                                    vector_offset=gidx[0:PT, b:b + 1], indirect_dim=0),
                      dge_mode=nisa.dge_mode.swdge)
        nl.store(index_k[b * PT:(b + 1) * PT, 0:INDEX_DIM], value=ti)
    if CORES > 1:
        nisa.core_barrier(data=compress_kv, cores=tuple(range(CORES)))
        nisa.core_barrier(data=index_k, cores=tuple(range(CORES)))

    msk = nl.ndarray((SEQ_LOC, CSEQ), dtype=nl.uint8, buffer=nl.private_hbm)
    scr = nl.ndarray((SEQ_LOC, 16, NT), dtype=nl.bfloat16, buffer=nl.private_hbm)
    thrb = nl.ndarray((SEQ_LOC, 16), dtype=nl.bfloat16, buffer=nl.private_hbm)
    _index_mask(index_q, index_k, iw, msk, scr, thrb, ATILES, rid)

    q = nl.ndarray((N_HEADS * SEQ_LOC, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    _gemm_head_major(qr_q, wqb, q, QR, Q_LORA, N_HEADS * HEAD_DIM, HEAD_DIM,
                     freqs_hbm=floc)

    winl = nl.ndarray((MT_LOC * 2 * PT, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    og = nl.ndarray((O_GROUPS * SEQ_LOC, HEADS_PER_GROUP * HEAD_DIM), dtype=nl.bfloat16,
                    buffer=nl.private_hbm)
    _sparse_attention(q, kvp, compress_kv, msk, layers_2_attn_attn_sink,
                      floc, og, winl, ATILES, rid)

    z_q = nl.ndarray((SEQ_LOC, O_GROUPS * O_LORA), dtype=nl.bfloat16, buffer=nl.private_hbm)
    for g in range(O_GROUPS):
        _gemm(og, wo_a, z_q, QR, HEADS_PER_GROUP * HEAD_DIM, O_LORA,
              x_row=g * SEQ_LOC, w_row=g * O_LORA, dst_col=g * O_LORA, fp8=True)
    _gemm(z_q, wo_b, cc_src_a, QR, O_GROUPS * O_LORA, CC_HALF, out_dtype=CC_DTYPE)
    if CORES > 1:
        nisa.core_barrier(data=cc_src_a, cores=tuple(range(CORES)))
    ncc.all_gather(srcs=[cc_src_a], dsts=[cc_dst_a], replica_group=replica_group,
                   collective_dim=0)
    _gemm(z_q, wo_b, cc_src_b, QR, O_GROUPS * O_LORA, CC_HALF, w_row=CC_HALF,
          out_dtype=CC_DTYPE)
    if CORES > 1:
        nisa.core_barrier(data=cc_src_b, cores=tuple(range(CORES)))
    ncc.all_gather(srcs=[cc_src_b], dsts=[cc_dst_b], replica_group=replica_group,
                   collective_dim=0)
    for half, cc_dst in ((0, cc_dst_a), (1, cc_dst_b)):
        for j in TILES:
            for s in range(WORLD_SIZE):
                g = WORLD_SIZE * j + s
                nisa.dma_copy(
                    dst=out2[g * PT:(g + 1) * PT, half * CC_HALF:(half + 1) * CC_HALF],
                    src=cc_dst[s * SEQ_LOC + j * PT:s * SEQ_LOC + (j + 1) * PT, 0:CC_HALF])
    if CORES > 1:
        nisa.core_barrier(data=out, cores=tuple(range(CORES)))
    return out


kernel = _impl[CORES]
