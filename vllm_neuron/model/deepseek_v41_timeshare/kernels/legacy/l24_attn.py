"""`layers.24.attention` on 4 ranks, sequence-parallel: every rank holds all 64 heads and the whole
of every weight, and owns a quarter of the **sequence**. `wo_b` stops being row-parallel, so a
rank writes *finished* output rows and the ranks rejoin by `all_gather` rather than
`all_reduce`. A rank's query tiles are interleaved (global tile `WORLD_SIZE * j + rank` is its
local tile `j`), which is what lets one NEFF schedule all four ranks identically.
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
RATIO = 1
CSEQ = SEQ // RATIO
CACHE_ROWS = 4 * 16384
INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_TOPK = 512
EPS = 1e-20
INDEX_SCORE_SCALE = 1.0 / 64.0
NEG_INF = -1.0e30
SM_SCALE = HEAD_DIM ** -0.5
NEG_BOUND = -1.0e30
HEADS_PER_GROUP = N_HEADS // O_GROUPS
OB = 4

PT = 128
NT = 512
WIN_SPAN = 2 * PT
CHUNK = 1

FP8_BLK = 32
FP8_MAX = 448.0
FP8_AMAX_FLOOR = 1e-4
LN2 = 0.6931471805599453
QDQ_CHUNK = 1280

MGROUP_BYTES = 96 * 1024

CORES = 2
QG = NT
NG = SEQ // QG

SEQ_LOC = SEQ // 4
MT_LOC = SEQ_LOC // PT
THR_FLOOR = -1.0e29

WORLD_SIZE = 4
OUT_DTYPE = nl.bfloat16
CC_DTYPE = nl.bfloat16
CC_ROWS = 1024
CC_HALF = DIM // 2


def _blocks(n, step):
    out = []
    off = 0
    while off < n:
        out.append((off, min(step, n - off)))
        off += step
    return out


def _core_runs(pid, cores=CORES):
    per = MT_LOC // (2 * cores)
    a = (pid * per, pid * per + per)
    b = (MT_LOC - (pid + 1) * per, MT_LOC - pid * per)
    runs = [(a[0], b[1])] if a[1] == b[0] else [a, b]
    return ([(t0 * PT, (t1 - t0) * PT) for t0, t1 in runs],
            [t for t0, t1 in runs for t in range(t0, t1)])


def _rank_id():
    rid = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.register_store(rid, ncc.rank_id())
    return rid


def _shard0(t, rank_rows, cols, rid, row, rows):
    return t.reshape((WORLD_SIZE, rank_rows, cols)).ap(
        pattern=[[cols, rows], [1, cols]], offset=row * cols,
        scalar_offset=rid, indirect_dim=0)


def _shard1(t, rows_all, rank_cols, rid, row, rows):
    return t.reshape((rows_all * WORLD_SIZE, rank_cols)).ap(
        pattern=[[WORLD_SIZE * rank_cols, rows], [1, rank_cols]],
        offset=row * WORLD_SIZE * rank_cols, scalar_offset=rid, indirect_dim=0)


def _rtile(t, cols, rid, tile, extra=0, rows=PT):
    return t.reshape((t.shape[0] // PT, PT, cols)).ap(
        pattern=[[cols, rows], [1, cols]],
        offset=(WORLD_SIZE * tile + extra) * PT * cols, scalar_offset=rid, indirect_dim=0)


def _rank_col(rid, rows, scale=1.0):
    r1 = nl.copy(rid.reshape((1, 1)), dtype=nl.float32)
    return nl.multiply(nl.broadcast_to(r1, (rows, 1)), scale)


def _load_bcast(vec_hbm, n, rows):
    return nl.broadcast_to(nl.load(vec_hbm.reshape((1, n)), dtype=nl.float32), (rows, n))


def _load_bcast_shard(vec_hbm, n, rows, rid):
    v = nl.ndarray((1, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=v, src=vec_hbm.reshape((WORLD_SIZE, n)).ap(
        pattern=[[n, 1], [1, n]], offset=0, scalar_offset=rid, indirect_dim=0))
    return nl.broadcast_to(v, (rows, n))


def _rmsnorm(tile, wb, n):
    sq = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.sbuf)
    ss = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=tile, reduce_op=nl.add, reduce_res=ss,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    ms = nl.multiply(ss, 1.0 / n)
    ms = nl.add(ms, EPS)
    y = nl.rsqrt(ms)
    y = nl.multiply(y, nl.subtract(1.5, nl.multiply(nl.multiply(ms, 0.5), nl.multiply(y, y))))
    return nl.multiply(nl.multiply(tile, y), wb)


def _pow2_ceil_log2(x):
    bits = nl.copy(x, dtype=nl.float32).view(nl.int32)
    exp = nl.bitwise_and(nl.right_shift(bits, 23), 0xFF)
    man = nl.bitwise_and(bits, 0x7FFFFF)
    bump = nl.copy(nl.greater(man, 0), dtype=nl.int32)
    e = nl.add(exp, bump)
    scale = nl.copy(nl.left_shift(e, 23)).view(nl.float32)
    inv = nl.copy(nl.left_shift(nl.subtract(254, e), 23)).view(nl.float32)
    return scale, inv


def _i32(shape):
    return nl.ndarray(shape, dtype=nl.int32, buffer=nl.sbuf)


def _decode_e4m3(bytes_u8):
    shape = bytes_u8.shape

    def u8():
        return nl.ndarray(shape, dtype=nl.uint8, buffer=nl.sbuf)

    m7 = u8()
    nisa.tensor_scalar(dst=m7, data=bytes_u8, op0=nl.bitwise_and, operand0=0x7F)
    dn = u8()
    nisa.tensor_scalar(dst=dn, data=m7, op0=nl.maximum, operand0=8, op1=nl.subtract, operand1=8)
    sgn = u8()
    nisa.tensor_scalar(dst=sgn, data=bytes_u8, op0=nl.bitwise_and, operand0=0x80)
    b2 = u8()
    nisa.tensor_tensor(dst=b2, data1=dn, data2=sgn, op=nl.bitwise_or)
    return nl.copy(b2.view(nl.float8_e4m3fn), dtype=nl.bfloat16)


def _block_amax(blocked, floor):
    ab = nl.ndarray(blocked.shape, dtype=blocked.dtype, buffer=nl.sbuf)
    nisa.activation(dst=ab, op=nl.abs, data=blocked)
    amax = nl.max(ab, axis=2, keepdims=True, dtype=nl.float32)
    return nl.maximum(amax, floor)


def _qdq_fp8(tile, rows, n):
    nb = n // FP8_BLK
    blocked = tile.reshape((rows, nb, FP8_BLK))
    amax = _block_amax(blocked, FP8_AMAX_FLOOR)
    scale, rcp = _pow2_ceil_log2(nl.multiply(amax, 2.0 / FP8_MAX))
    inv = rcp.broadcast(2, FP8_BLK)
    wide = scale.broadcast(2, FP8_BLK)
    q8 = nl.ndarray((rows, nb, FP8_BLK), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=q8, data1=blocked, data2=inv, op=nl.multiply)
    out = nl.ndarray((rows, nb, FP8_BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out, data1=q8, data2=wide, op=nl.multiply)
    return out.reshape((rows, n))


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


def _dequant_weight(w_u8, s_u8, dst_hbm, n, k, dst_row=0, src_row=0, shard=None, rid=None,
                    shard_rows=None):
    kb = k // FP8_BLK
    sper = PT // FP8_BLK
    sh = n if shard_rows is None else shard_rows
    sel = _scale_selector()
    for nb in nl.affine_range(n // PT):
        r0 = src_row + nb * PT
        s0 = r0 // FP8_BLK
        if shard is None:
            wq = _decode_e4m3(nl.load(w_u8[r0:r0 + PT, 0:k]))
            sb = nl.load(s_u8[s0:s0 + sper, 0:kb], dtype=nl.int32)
        else:
            wb = nl.ndarray((PT, k), dtype=nl.uint8, buffer=nl.sbuf)
            sb = nl.ndarray((sper, kb), dtype=nl.int32, buffer=nl.sbuf)
            if shard == 0:
                nisa.dma_copy(dst=wb, src=_shard0(w_u8, sh, k, rid, r0, PT))
                nisa.dma_copy(dst=sb, src=_shard0(s_u8, sh // FP8_BLK, kb, rid, s0, sper))
            else:
                nisa.dma_copy(dst=wb, src=_shard1(w_u8, n, k, rid, r0, PT))
                nisa.dma_copy(dst=sb, src=_shard1(s_u8, n // FP8_BLK, kb, rid, s0, sper))
            wq = _decode_e4m3(wb)
        ex = nl.ndarray((PT, kb), dtype=nl.float32, buffer=nl.psum)
        sbf = nl.ndarray((sper, kb), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sbf, data=sb, op0=nl.add, operand0=1)
        nisa.nc_matmul(dst=ex, stationary=sel, moving=sbf, accumulate=False)
        rep = nl.ndarray((PT, kb), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(dst=rep, op=nl.exp, data=ex, bias=-127.0 * LN2, scale=LN2)
        rep = rep.reshape((PT, kb, 1)).broadcast(2, FP8_BLK)
        prod = nl.ndarray((PT, kb, FP8_BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prod, data1=wq.reshape((PT, kb, FP8_BLK)), data2=rep,
                           op=nl.multiply)
        nl.store(dst_hbm[dst_row + nb * PT:dst_row + (nb + 1) * PT, 0:k],
                 value=prod.reshape((PT, k)))


FP4_MAX = 6.0


def _qdq_fp4(tile, rows, n, blk):
    nb = n // blk
    shape = (rows, nb, blk)
    blocked = tile.reshape(shape)
    amax = _block_amax(blocked, 6.0 * (2.0 ** -126))
    sbits = nl.copy(nl.multiply(amax, 1.0 / FP4_MAX), dtype=nl.float32).view(nl.int32)
    e = nl.add(nl.bitwise_and(nl.right_shift(sbits, 23), 0xFF),
               nl.copy(nl.greater(nl.bitwise_and(sbits, 0x7FFFFF), 0), dtype=nl.int32))
    ef = _i32(shape)
    nisa.tensor_scalar(dst=ef, data=blocked.view(nl.int32), op0=nl.bitwise_and,
                       operand0=0x7FFFFFFF, op1=nl.right_shift, operand1=23)
    sf = _i32(shape)
    nisa.tensor_tensor(dst=sf, data1=ef, data2=e.broadcast(2, blk), op=nl.maximum)
    ki = _i32(shape)
    nisa.tensor_scalar(dst=ki, data=sf, op0=nl.left_shift, operand0=23)
    kf = nl.multiply(ki.view(nl.float32), 6291456.0)
    y = nl.add(blocked, kf)
    out = nl.ndarray(shape, dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out, data1=y, data2=kf, op=nl.subtract)
    return out.reshape((rows, n))


def _rope_inplace(t_hbm, freqs_hbm, m, width, pos_stride, inverse, heads=1, row=0, hstride=0):
    half = ROPE_DIM // 2
    lo = width - ROPE_DIM
    for hb in nl.affine_range(heads):
        for mb in nl.affine_range(m // PT):
            r0 = hb * hstride + row + mb * PT
            t = nl.load(t_hbm[r0:r0 + PT, lo:width], dtype=nl.float32).reshape((PT, half, 2))
            a = t[0:PT, 0:half, 0]
            b = t[0:PT, 0:half, 1]
            p0 = (row + mb * PT) * pos_stride
            p1 = p0 + PT * pos_stride
            cos = nl.load(freqs_hbm[p0:p1:pos_stride, 0:half, 0], dtype=nl.float32)
            sin = nl.load(freqs_hbm[p0:p1:pos_stride, 0:half, 1], dtype=nl.float32)
            o = nl.ndarray((PT, half, 2), dtype=nl.float32, buffer=nl.sbuf)
            ac = nl.multiply(a, cos)
            bs = nl.multiply(b, sin)
            as_ = nl.multiply(a, sin)
            bc = nl.multiply(b, cos)
            if inverse:
                nisa.tensor_copy(dst=o[0:PT, 0:half, 0], src=nl.add(ac, bs))
                nisa.tensor_copy(dst=o[0:PT, 0:half, 1], src=nl.subtract(bc, as_))
            else:
                nisa.tensor_copy(dst=o[0:PT, 0:half, 0], src=nl.subtract(ac, bs))
                nisa.tensor_copy(dst=o[0:PT, 0:half, 1], src=nl.add(as_, bc))
            nl.store(t_hbm[r0:r0 + PT, lo:width],
                     value=nl.copy(o.reshape((PT, ROPE_DIM)), dtype=nl.bfloat16))


def _qdq_rows(src_hbm, dst_hbm, m, n, src_row=0, dst_row=0):
    for mb in nl.affine_range(m // PT):
        t = nl.load(src_hbm[src_row + mb * PT:src_row + (mb + 1) * PT, 0:n],
                    dtype=nl.bfloat16)
        nl.store(dst_hbm[dst_row + mb * PT:dst_row + (mb + 1) * PT, 0:n],
                 value=_qdq_fp8(t, PT, n))


def _mgroup(mtiles, ktiles):
    mg = 1
    while mg * 2 <= mtiles and (mg * 2) * ktiles * PT * 2 <= MGROUP_BYTES:
        mg *= 2
    return mg


def _stage_x(xt, x_hbm, ktiles, mg, row, x_col=0):
    sg = 4 if mg % 4 == 0 else (2 if mg % 2 == 0 else 1)
    flat = xt.reshape((PT, ktiles, mg * PT))
    x3 = x_hbm.reshape((x_hbm.shape[0], x_hbm.shape[1] // PT, PT))
    c0 = x_col // PT
    for mi in nl.static_range(mg // sg):
        r0 = row + mi * sg * PT
        nisa.dma_transpose(dst=flat[0:PT, 0:ktiles, mi * sg * PT:(mi + 1) * sg * PT],
                           src=x3[r0:r0 + sg * PT, c0:c0 + ktiles, 0:PT])


def _gemm(x_hbm, w_hbm, dst_hbm, m, k, n, out_dtype=nl.bfloat16,
          x_row=0, x_col=0, w_row=0, dst_row=0, dst_col=0, qdq=False):
    ktiles = k // PT
    if n <= NT:
        nt = n
    elif n % NT == 0:
        nt = NT
    elif n % 256 == 0:
        nt = 256
    else:
        nt = PT
    mg = _mgroup(m // PT, ktiles)
    w3 = w_hbm.reshape((w_hbm.shape[0], ktiles, PT))
    for mgb in nl.affine_range((m // PT) // mg):
        xt = nl.ndarray((PT, ktiles, mg, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
        _stage_x(xt, x_hbm, ktiles, mg, x_row + mgb * mg * PT, x_col=x_col)
        for nb in nl.affine_range(n // nt):
            wt = nl.ndarray((PT, ktiles, nt), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_transpose(dst=wt[0:PT, 0:ktiles, 0:nt],
                               src=w3[w_row + nb * nt:w_row + (nb + 1) * nt, 0:ktiles, 0:PT])
            for mi in nl.static_range(mg):
                acc = nl.ndarray((PT, nt), dtype=nl.float32, buffer=nl.psum)
                for kb in nl.static_range(ktiles):
                    nisa.nc_matmul(dst=acc, stationary=xt[0:PT, kb, mi, 0:PT],
                                   moving=wt[0:PT, kb, 0:nt], accumulate=(kb > 0))
                r = dst_row + (mgb * mg + mi) * PT
                if qdq:
                    rb = nl.copy(acc, dtype=nl.bfloat16)
                    val = _qdq_fp8(rb, PT, nt)
                else:
                    val = nl.copy(acc, dtype=out_dtype)
                nl.store(dst_hbm[r:r + PT, dst_col + nb * nt:dst_col + (nb + 1) * nt],
                         value=val)


def _gemm_head_major(x_hbm, w_hbm, dst_hbm, m, k, n, hdim, row=0, hstride=0, freqs_hbm=None):
    ktiles = k // PT
    half = ROPE_DIM // 2
    rg = hdim // ROPE_DIM
    mg = _mgroup(m // PT, ktiles)
    w3 = w_hbm.reshape((w_hbm.shape[0], ktiles, PT))
    for mgb in nl.affine_range((m // PT) // mg):
        xt = nl.ndarray((PT, ktiles, mg, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
        _stage_x(xt, x_hbm, ktiles, mg, row + mgb * mg * PT)
        if freqs_hbm is not None:
            cs = nl.ndarray((PT, mg, 2, half), dtype=nl.float32, buffer=nl.sbuf)
            for mi in nl.static_range(mg):
                r0 = row + (mgb * mg + mi) * PT
                nisa.tensor_copy(dst=cs[0:PT, mi, 0, 0:half],
                                 src=nl.load(freqs_hbm[r0:r0 + PT, 0:half, 0],
                                             dtype=nl.float32))
                nisa.tensor_copy(dst=cs[0:PT, mi, 1, 0:half],
                                 src=nl.load(freqs_hbm[r0:r0 + PT, 0:half, 1],
                                             dtype=nl.float32))
        for nb in nl.affine_range(n // hdim):
            wt = nl.ndarray((PT, ktiles, hdim), dtype=nl.bfloat16, buffer=nl.sbuf)
            for kb in nl.static_range(ktiles):
                nisa.dma_transpose(
                    dst=wt[0:PT, kb, 0:hdim],
                    src=w_hbm[nb * hdim:(nb + 1) * hdim, kb * PT:(kb + 1) * PT])
            for mi in nl.static_range(mg):
                acc = nl.ndarray((PT, hdim), dtype=nl.float32, buffer=nl.psum)
                for kb in nl.static_range(ktiles):
                    nisa.nc_matmul(dst=acc, stationary=xt[0:PT, kb, mi, 0:PT],
                                   moving=wt[0:PT, kb, 0:hdim], accumulate=(kb > 0))
                r = nb * hstride + row + (mgb * mg + mi) * PT
                if freqs_hbm is None:
                    nl.store(dst_hbm[r:r + PT, 0:hdim], value=nl.copy(acc, dtype=nl.bfloat16))
                else:
                    tb = nl.ndarray((PT, hdim), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=tb, src=acc)
                    v = tb.reshape((PT, rg, half, 2))
                    a = v[0:PT, rg - 1, 0:half, 0]
                    b = v[0:PT, rg - 1, 0:half, 1]
                    c = cs[0:PT, mi, 0, 0:half]
                    s = cs[0:PT, mi, 1, 0:half]
                    ac = nl.multiply(a, c, dtype=nl.float32)
                    bs = nl.multiply(b, s, dtype=nl.float32)
                    as_ = nl.multiply(a, s, dtype=nl.float32)
                    bc = nl.multiply(b, c, dtype=nl.float32)
                    nisa.tensor_tensor(dst=a, data1=ac, data2=bs, op=nl.subtract)
                    nisa.tensor_tensor(dst=b, data1=as_, data2=bc, op=nl.add)
                    nl.store(dst_hbm[r:r + PT, 0:hdim], value=tb)


def _index_q_build(qr_hbm, w_hbm, freqs_hbm, dst_hbm, m, row=0, mg=16, hg_heads=4):
    half = ROPE_DIM // 2
    ktiles = Q_LORA // PT
    hd = hg_heads * INDEX_DIM
    mtiles = m // PT
    while mg > mtiles:
        mg //= 2
    for mgb in nl.affine_range(mtiles // mg):
        xt = nl.ndarray((PT, ktiles, mg, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
        _stage_x(xt, qr_hbm, ktiles, mg, row + mgb * mg * PT)
        cos4 = nl.ndarray((PT, mg, hg_heads, half), dtype=nl.float32, buffer=nl.sbuf)
        sin4 = nl.ndarray((PT, mg, hg_heads, half), dtype=nl.float32, buffer=nl.sbuf)
        for mi in nl.static_range(mg):
            r0 = row + (mgb * mg + mi) * PT
            cos = nl.load(freqs_hbm[r0:r0 + PT, 0:half, 0], dtype=nl.float32)
            sin = nl.load(freqs_hbm[r0:r0 + PT, 0:half, 1], dtype=nl.float32)
            for j in nl.static_range(hg_heads):
                nisa.tensor_copy(dst=cos4[0:PT, mi, j, 0:half], src=cos)
                nisa.tensor_copy(dst=sin4[0:PT, mi, j, 0:half], src=sin)
        for hg in nl.affine_range(INDEX_HEADS // hg_heads):
            wt = nl.ndarray((PT, ktiles, hd), dtype=nl.bfloat16, buffer=nl.sbuf)
            for kb in nl.static_range(ktiles):
                nisa.dma_transpose(
                    dst=wt[0:PT, kb, 0:hd],
                    src=w_hbm[hg * hd:(hg + 1) * hd, kb * PT:(kb + 1) * PT])
            for mi in nl.static_range(mg):
                r0 = row + (mgb * mg + mi) * PT
                acc = nl.ndarray((PT, hd), dtype=nl.float32, buffer=nl.psum)
                for kb in nl.static_range(ktiles):
                    nisa.nc_matmul(dst=acc, stationary=xt[0:PT, kb, mi, 0:PT],
                                   moving=wt[0:PT, kb, 0:hd], accumulate=(kb > 0))
                tb = nl.ndarray((PT, hd), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=tb, src=acc)
                v = tb.reshape((PT, hg_heads, 2, half, 2))
                a = v[0:PT, 0:hg_heads, 1, 0:half, 0]
                b = v[0:PT, 0:hg_heads, 1, 0:half, 1]
                c = cos4[0:PT, mi, 0:hg_heads, 0:half]
                s = sin4[0:PT, mi, 0:hg_heads, 0:half]
                ac = nl.multiply(a, c, dtype=nl.float32)
                bs = nl.multiply(b, s, dtype=nl.float32)
                as_ = nl.multiply(a, s, dtype=nl.float32)
                bc = nl.multiply(b, c, dtype=nl.float32)
                nisa.tensor_tensor(dst=a, data1=ac, data2=bs, op=nl.subtract)
                nisa.tensor_tensor(dst=b, data1=as_, data2=bc, op=nl.add)
                tf = nl.ndarray((PT, hd), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=tf, src=tb)
                nl.store(dst_hbm[r0:r0 + PT, hg * hd:(hg + 1) * hd],
                         value=_qdq_fp4(tf, PT, hd, 32))


def _norm_rows(src_hbm, w_hbm, dst_hbm, m, n, row=0, src_col=0):
    wb = _load_bcast(w_hbm, n, PT)
    for mb in nl.affine_range(m // PT):
        t = nl.load(src_hbm[row + mb * PT:row + (mb + 1) * PT, src_col:src_col + n],
                    dtype=nl.float32)
        nl.store(dst_hbm[row + mb * PT:row + (mb + 1) * PT, 0:n],
                 value=nl.copy(_rmsnorm(t, wb, n), dtype=nl.bfloat16))


def _norm_quant_rows(src_hbm, w_hbm, dst_hbm, m, n, row=0, src_col=0):
    wb = _load_bcast(w_hbm, n, PT)
    for mb in nl.affine_range(m // PT):
        t = nl.load(src_hbm[row + mb * PT:row + (mb + 1) * PT, src_col:src_col + n],
                    dtype=nl.float32)
        tb = nl.copy(_rmsnorm(t, wb, n), dtype=nl.bfloat16)
        nl.store(dst_hbm[row + mb * PT:row + (mb + 1) * PT, 0:n],
                 value=_qdq_fp8(tb, PT, n))


def _norm_rows_both(src_hbm, w_hbm, dst_b, dst_q, m, n, row=0):
    wb = _load_bcast(w_hbm, n, PT)
    for mb in nl.affine_range(m // PT):
        t = nl.load(src_hbm[row + mb * PT:row + (mb + 1) * PT, 0:n], dtype=nl.float32)
        tb = nl.copy(_rmsnorm(t, wb, n), dtype=nl.bfloat16)
        nl.store(dst_b[row + mb * PT:row + (mb + 1) * PT, 0:n], value=tb)
        nl.store(dst_q[row + mb * PT:row + (mb + 1) * PT, 0:n],
                 value=_qdq_fp8(tb, PT, n))


def _index_mask(index_q, ik_hbm, iw_hbm, msk_hbm, scr_hbm, thr_hbm, tiles, rid):
    iq3 = index_q.reshape((SEQ_LOC, INDEX_HEADS, INDEX_DIM))

    kblk = max(tiles) + 1
    ikt_all = nl.ndarray((INDEX_DIM, kblk, NT), dtype=nl.bfloat16, buffer=nl.sbuf)
    for gb in nl.affine_range(kblk):
        nisa.dma_transpose(dst=ikt_all[0:INDEX_DIM, gb, 0:NT],
                           src=ik_hbm[gb * NT:(gb + 1) * NT, 0:INDEX_DIM])

    bpos_i = nl.ndarray((PT, NT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=bpos_i, pattern=[[1, NT]], offset=0, channel_multiplier=0)
    bpos = nl.copy(bpos_i, dtype=nl.float32)
    prow_i = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=prow_i, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    prow = nl.copy(prow_i, dtype=nl.float32)
    neg = nl.full((PT, NT), NEG_INF, dtype=nl.bfloat16)
    clim = nl.add(nl.add(prow, 1.0), _rank_col(rid, PT, float(PT)))
    for j in tiles:
        gbn = j + 1
        q0 = j * PT
        wrow = nl.multiply(nl.load(iw_hbm[q0:q0 + PT, 0:INDEX_HEADS], dtype=nl.float32),
                           INDEX_SCORE_SCALE)
        iqt = nl.ndarray((INDEX_DIM, INDEX_HEADS, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_transpose(dst=iqt[0:INDEX_DIM, 0:INDEX_HEADS, 0:PT],
                           src=iq3[q0:q0 + PT, 0:INDEX_HEADS, 0:INDEX_DIM])
        rows = nl.ndarray((PT, gbn, NT), dtype=nl.bfloat16, buffer=nl.sbuf)
        for gb in nl.static_range(gbn):
            ikt = ikt_all[0:INDEX_DIM, gb, 0:NT]
            tgt = rows[0:PT, gb, 0:NT]
            for hb in nl.static_range(INDEX_HEADS):
                ps = nl.ndarray((PT, NT), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=ps, stationary=iqt[0:INDEX_DIM, hb, 0:PT], moving=ikt,
                               accumulate=False)
                rect = nl.ndarray((PT, NT), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=rect, op=nl.relu, data=ps)
                if hb == 0:
                    nisa.tensor_scalar(dst=tgt, data=rect, op0=nl.multiply,
                                       operand0=wrow[0:PT, hb:hb + 1])
                else:
                    nisa.scalar_tensor_tensor(dst=tgt, data=rect, op0=nl.multiply,
                                              operand0=wrow[0:PT, hb:hb + 1], op1=nl.add,
                                              operand1=tgt)
            if gb == gbn - 1:
                reach = nl.ndarray((PT, NT), dtype=nl.uint8, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=reach, data=bpos[0:PT, 0:NT], op0=nl.less, operand0=clim)
                nisa.tensor_copy(dst=tgt, src=nl.where(reach, tgt, neg))

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


def _sparse_attention(q_hbm, kvp_hbm, ckv_hbm, msk_hbm, sink_hbm, freqs_hbm, og_hbm, winl, tiles,
                      rid):
    dt = HEAD_DIM // PT
    st = NT // PT
    cmax = (max(tiles) + 1) * NT
    q3 = q_hbm.reshape((N_HEADS * SEQ_LOC, dt, PT))
    ckv3 = ckv_hbm.reshape((CACHE_ROWS, dt, PT))
    winl3 = winl.reshape((MT_LOC * 2 * PT, dt, PT))

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
    pwin_gen = nl.copy(nl.greater(inwin, 0.5), dtype=nl.uint8)
    lowok = nl.ndarray((PT, WIN_SPAN), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lowok, data=nonneg, op0=nl.maximum,
                       operand0=nl.copy(nl.greater(_rank_col(rid, PT), 0.5), dtype=nl.float32))
    pwin_zero = nl.copy(nl.greater(nl.multiply(lowok, inwin), 0.5), dtype=nl.uint8)
    sinkb = _load_bcast(sink_hbm, N_HEADS, PT)

    ckv_t = nl.ndarray((PT, dt, cmax), dtype=nl.bfloat16, buffer=nl.sbuf)
    ckv_n = nl.ndarray((PT, cmax // PT, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    for gp in nl.affine_range(cmax // PT):
        nisa.dma_copy(dst=ckv_n[0:PT, gp, 0:HEAD_DIM],
                      src=ckv_hbm[gp * PT:(gp + 1) * PT, 0:HEAD_DIM])
        nisa.dma_transpose(dst=ckv_t[0:PT, 0:dt, gp * PT:(gp + 1) * PT],
                           src=ckv3[gp * PT:(gp + 1) * PT, 0:dt, 0:PT])

    if True:
        for j in tiles:
            gbn = j + 1
            q0 = j * PT
            pwin = pwin_zero if j == 0 else pwin_gen
            win_t = nl.ndarray((PT, dt, WIN_SPAN), dtype=nl.bfloat16, buffer=nl.sbuf)
            win_n = nl.ndarray((PT, 2, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
            for wt in nl.static_range(2):
                nisa.dma_copy(dst=win_n[0:PT, wt, 0:HEAD_DIM],
                              src=_rtile(kvp_hbm, HEAD_DIM, rid, j, extra=wt))
                nl.store(winl[(j * 2 + wt) * PT:(j * 2 + wt + 1) * PT, 0:HEAD_DIM],
                         value=win_n[0:PT, wt, 0:HEAD_DIM])
            for wt in nl.static_range(2):
                w0 = (j * 2 + wt) * PT
                nisa.dma_transpose(dst=win_t[0:PT, 0:dt, wt * PT:(wt + 1) * PT],
                                   src=winl3[w0:w0 + PT, 0:dt, 0:PT])
            kp = nl.load(msk_hbm[q0:q0 + PT, 0:gbn * NT], dtype=nl.uint8)
            rcos = nl.load(freqs_hbm[q0:q0 + PT, 0:ROPE_DIM // 2, 0], dtype=nl.float32)
            rsin = nl.load(freqs_hbm[q0:q0 + PT, 0:ROPE_DIM // 2, 1], dtype=nl.float32)
            rc8 = rcos.reshape((PT, 1, ROPE_DIM // 2)).broadcast(1, OB)
            rs8 = rsin.reshape((PT, 1, ROPE_DIM // 2)).broadcast(1, OB)


            for g in nl.affine_range(O_GROUPS):
              for hbat in nl.static_range(HEADS_PER_GROUP // OB):
                obuf = nl.ndarray((PT, OB, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
                for hi in nl.affine_range(OB):
                    hh = hbat * OB + hi
                    h = g * HEADS_PER_GROUP + hh
                    qt = nl.ndarray((PT, dt, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.dma_transpose(dst=qt[0:PT, 0:dt, 0:PT],
                                       src=q3[h * SEQ_LOC + q0:h * SEQ_LOC + q0 + PT, 0:dt, 0:PT])
                    wmax = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
                    lwp = nl.ndarray((PT, WIN_SPAN), dtype=nl.float32, buffer=nl.psum)
                    for dc in nl.static_range(dt):
                        nisa.nc_matmul(dst=lwp, stationary=qt[0:PT, dc, 0:PT],
                                       moving=win_t[0:PT, dc, 0:WIN_SPAN],
                                       accumulate=(dc > 0))
                    lw = nl.ndarray((PT, WIN_SPAN), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.select_reduce(dst=lw, predicate=pwin, on_true=lwp,
                                       on_false=NEG_BOUND, reduce_res=wmax,
                                       reduce_cmd=nisa.reduce_cmd.reset_reduce,
                                       reduce_op=nl.maximum)
                    lcs = nl.ndarray((PT, gbn * NT), dtype=nl.float32, buffer=nl.sbuf)
                    bmax = nl.ndarray((PT, gbn), dtype=nl.float32, buffer=nl.sbuf)
                    for gb in nl.static_range(gbn):
                        bw = NT
                        keep = kp[0:PT, gb * NT:gb * NT + bw]
                        lcp = nl.ndarray((PT, bw), dtype=nl.float32, buffer=nl.psum)
                        for dc in nl.static_range(dt):
                            nisa.nc_matmul(dst=lcp, stationary=qt[0:PT, dc, 0:PT],
                                           moving=ckv_t[0:PT, dc, gb * NT:gb * NT + bw],
                                           accumulate=(dc > 0))
                        nisa.select_reduce(dst=lcs[0:PT, gb * NT:gb * NT + bw],
                                           predicate=keep, on_true=lcp, on_false=NEG_BOUND,
                                           reduce_res=bmax[0:PT, gb:gb + 1],
                                           reduce_cmd=nisa.reduce_cmd.reset_reduce,
                                           reduce_op=nl.maximum)
                    m = nl.multiply(nl.maximum(wmax, nl.max(bmax, axis=1, keepdims=True)),
                                    SM_SCALE)
                    negm = nl.multiply(m, -1.0)
                    nch = (gbn + CHUNK - 1) // CHUNK
                    dcol = nl.ndarray((PT, nch + 1), dtype=nl.float32, buffer=nl.sbuf)
                    acc = nl.ndarray((PT, HEAD_DIM), dtype=nl.float32, buffer=nl.psum)
                    pw = nl.ndarray((PT, WIN_SPAN), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.activation(dst=pw, op=nl.exp, data=lw, bias=negm, scale=SM_SCALE,
                                    reduce_op=nl.add, reduce_res=dcol[0:PT, 0:1],
                                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
                    pwt = nl.ndarray((PT, 2, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.dma_transpose(dst=pwt[0:PT, 0:2, 0:PT],
                                       src=pw.reshape((PT, 2, PT))[0:PT, 0:2, 0:PT])
                    for wt in nl.static_range(2):
                        nisa.nc_matmul(dst=acc, stationary=pwt[0:PT, wt, 0:PT],
                                       moving=win_n[0:PT, wt, 0:HEAD_DIM],
                                       accumulate=(wt > 0))
                    for ci in range(nch):
                        g0 = ci * CHUNK
                        gk = min(CHUNK, gbn - g0)
                        cw = gk * NT
                        nb = cw // PT
                        pc = nl.ndarray((PT, cw), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.activation(dst=pc, op=nl.exp,
                                        data=lcs[0:PT, g0 * NT:g0 * NT + cw],
                                        bias=negm, scale=SM_SCALE,
                                        reduce_op=nl.add,
                                        reduce_res=dcol[0:PT, ci + 1:ci + 2],
                                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
                        pct = nl.ndarray((PT, nb, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
                        nisa.dma_transpose(dst=pct[0:PT, 0:nb, 0:PT],
                                           src=pc.reshape((PT, nb, PT))[0:PT, 0:nb, 0:PT])
                        for sb in nl.static_range(nb):
                            nisa.nc_matmul(dst=acc, stationary=pct[0:PT, sb, 0:PT],
                                           moving=ckv_n[0:PT, g0 * st + sb, 0:HEAD_DIM],
                                           accumulate=True)
                    esink = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=esink, op=nl.exp, data=sinkb[0:PT, h:h + 1],
                                    bias=negm, scale=1.0)
                    den = nl.add(nl.sum(dcol, axis=1, keepdims=True), esink)
                    nisa.tensor_scalar(dst=obuf[0:PT, hi, 0:HEAD_DIM], data=acc,
                                       op0=nl.multiply, operand0=nl.reciprocal(den))
                ov = obuf.reshape((PT, OB, HEAD_DIM // ROPE_DIM, ROPE_DIM // 2, 2))
                ra = ov[0:PT, 0:OB, HEAD_DIM // ROPE_DIM - 1, 0:ROPE_DIM // 2, 0]
                rb = ov[0:PT, 0:OB, HEAD_DIM // ROPE_DIM - 1, 0:ROPE_DIM // 2, 1]
                rac = nl.multiply(ra, rc8)
                rbs = nl.multiply(rb, rs8)
                ras = nl.multiply(ra, rs8)
                rbc = nl.multiply(rb, rc8)
                nisa.tensor_tensor(dst=ra, data1=rac, data2=rbs, op=nl.add)
                nisa.tensor_tensor(dst=rb, data1=rbc, data2=ras, op=nl.subtract)
                nl.store(og_hbm[g * SEQ_LOC + q0:g * SEQ_LOC + q0 + PT,
                                hbat * OB * HEAD_DIM:(hbat + 1) * OB * HEAD_DIM], value=obuf)


@nki.jit
def kernel(hidden_states, state_shared_attn_candidates, state_shared_attn_compress_kv, state_shared_attn_index_k, state_shared_attn_topk_idxs, layers_24_attn_attn_sink, layers_24_attn_freqs_cis, layers_24_attn_indexer_weights_proj_weight, layers_24_attn_indexer_wq_b_scale, layers_24_attn_indexer_wq_b_weight, layers_24_attn_kv_norm_weight, layers_24_attn_q_norm_weight, layers_24_attn_window_kv_cache, layers_24_attn_wkv_scale, layers_24_attn_wkv_weight, layers_24_attn_wo_a_scale, layers_24_attn_wo_a_weight, layers_24_attn_wo_b_scale, layers_24_attn_wo_b_weight, layers_24_attn_wq_a_scale, layers_24_attn_wq_a_weight, layers_24_attn_wq_b_scale, layers_24_attn_wq_b_weight, layers_24_attn_norm_weight, replica_group):
    x_hbm = hidden_states.reshape((SEQ, DIM))
    ckv_hbm = state_shared_attn_compress_kv.reshape((CACHE_ROWS, HEAD_DIM))
    ik_hbm = state_shared_attn_index_k.reshape((CACHE_ROWS, INDEX_DIM))
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

    pid = nl.program_id(0)
    runs, tiles = _core_runs(pid)
    atiles = sorted(tiles, reverse=True)

    rid = _rank_id()

    QKV_N = Q_LORA + HEAD_DIM
    wqkv = nl.ndarray((QKV_N, DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="wqkv")
    wqb = nl.ndarray((N_HEADS * HEAD_DIM, Q_LORA), dtype=nl.bfloat16,
                     buffer=nl.shared_hbm, name="wqb")
    iwqb = nl.ndarray((INDEX_HEADS * INDEX_DIM, Q_LORA), dtype=nl.bfloat16,
                      buffer=nl.shared_hbm, name="iwqb")
    wo_a = nl.ndarray((O_GROUPS * O_LORA, HEADS_PER_GROUP * HEAD_DIM), dtype=nl.bfloat16,
                      buffer=nl.shared_hbm, name="wo_a")
    wo_b = nl.ndarray((DIM, O_GROUPS * O_LORA), dtype=nl.bfloat16,
                      buffer=nl.shared_hbm, name="wo_b")
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
        _dequant_weight(layers_24_attn_wq_a_weight, layers_24_attn_wq_a_scale, wqkv,
                        Q_LORA, DIM)
        _dequant_weight(layers_24_attn_wkv_weight, layers_24_attn_wkv_scale, wqkv,
                        HEAD_DIM, DIM, dst_row=Q_LORA)
    else:
        _dequant_weight(layers_24_attn_indexer_wq_b_weight,
                        layers_24_attn_indexer_wq_b_scale, iwqb,
                        INDEX_HEADS * INDEX_DIM, Q_LORA)
    _dequant_weight(layers_24_attn_wq_b_weight, layers_24_attn_wq_b_scale, wqb_src,
                    WQB_N // CORES, Q_LORA, dst_row=pid * (WQB_N // CORES),
                    src_row=pid * (WQB_N // CORES), shard=0, rid=rid, shard_rows=WQB_N)
    _dequant_weight(layers_24_attn_wo_a_weight, layers_24_attn_wo_a_scale, woa_src,
                    WOA_N // CORES, HEADS_PER_GROUP * HEAD_DIM,
                    dst_row=pid * (WOA_N // CORES), src_row=pid * (WOA_N // CORES),
                    shard=0, rid=rid, shard_rows=WOA_N)
    _dequant_weight(layers_24_attn_wo_b_weight, layers_24_attn_wo_b_scale, wob_src,
                    WOB_N // CORES, O_GROUPS * O_LORA, dst_row=pid * (WOB_N // CORES),
                    src_row=pid * (WOB_N // CORES), shard=0, rid=rid, shard_rows=WOB_N)
    for wbuf in (wqkv, iwqb, wqb_src, woa_src, wob_src):
        nisa.core_barrier(data=wbuf, cores=tuple(range(CORES)))

    xloc = nl.ndarray((SEQ_LOC, DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    floc = nl.ndarray((SEQ_LOC, ROPE_DIM // 2, 2), dtype=nl.float32, buffer=nl.private_hbm)
    xn = nl.ndarray((SEQ_LOC, DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    xn_q = nl.ndarray((SEQ_LOC, DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    qakv = nl.ndarray((SEQ_LOC, QKV_N), dtype=nl.bfloat16, buffer=nl.private_hbm)
    kvr = nl.ndarray((SEQ_LOC, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    qr_q = nl.ndarray((SEQ_LOC, Q_LORA), dtype=nl.bfloat16, buffer=nl.private_hbm)
    kvp = nl.ndarray((SEQ + PT, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    winl = nl.ndarray((MT_LOC * 2 * PT, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    index_q = nl.ndarray((SEQ_LOC, INDEX_HEADS * INDEX_DIM), dtype=nl.bfloat16,
                         buffer=nl.private_hbm)
    iw = nl.ndarray((SEQ_LOC, INDEX_HEADS), dtype=nl.bfloat16, buffer=nl.private_hbm)
    q = nl.ndarray((N_HEADS * SEQ_LOC, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    msk = nl.ndarray((SEQ_LOC, CSEQ), dtype=nl.uint8, buffer=nl.private_hbm)
    scr = nl.ndarray((SEQ_LOC, 16, NT), dtype=nl.bfloat16, buffer=nl.private_hbm)
    thr = nl.ndarray((SEQ_LOC, 16), dtype=nl.bfloat16, buffer=nl.private_hbm)
    og = nl.ndarray((O_GROUPS * SEQ_LOC, HEADS_PER_GROUP * HEAD_DIM), dtype=nl.bfloat16,
                    buffer=nl.private_hbm)
    z_q = nl.ndarray((SEQ_LOC, O_GROUPS * O_LORA), dtype=nl.bfloat16, buffer=nl.private_hbm)

    xt = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    ft = nl.ndarray((PT, ROPE_DIM), dtype=nl.float32, buffer=nl.sbuf)
    for j in tiles:
        nisa.dma_copy(dst=xt, src=_rtile(x_hbm, DIM, rid, j))
        nl.store(xloc[j * PT:(j + 1) * PT, 0:DIM], value=xt)
        nisa.dma_copy(dst=ft, src=_rtile(layers_24_attn_freqs_cis.reshape(
                                             (layers_24_attn_freqs_cis.shape[0], ROPE_DIM)),
                                         ROPE_DIM, rid, j))
        nl.store(floc.reshape((SEQ_LOC, ROPE_DIM))[j * PT:(j + 1) * PT, 0:ROPE_DIM], value=ft)

    for r0, m in runs:
        _norm_rows_both(xloc, layers_24_attn_norm_weight, xn, xn_q, m, DIM, row=r0)
        _gemm(xn_q, wqkv, qakv, m, DIM, QKV_N, x_row=r0, dst_row=r0)
        _norm_rows(qakv, layers_24_attn_kv_norm_weight, kvr, m, HEAD_DIM,
                   row=r0, src_col=Q_LORA)
        _rope_inplace(kvr, floc, m, HEAD_DIM, 1, False, row=r0)
        _qdq_rows(kvr, kv_src, m, HEAD_DIM, src_row=r0, dst_row=r0)

        _norm_quant_rows(qakv, layers_24_attn_q_norm_weight, qr_q, m, Q_LORA, row=r0)

        _index_q_build(qr_q, iwqb, floc, index_q, m, row=r0)

        _gemm(xn, layers_24_attn_indexer_weights_proj_weight, iw, m, DIM, INDEX_HEADS,
              x_row=r0, dst_row=r0)

    nisa.core_barrier(data=kv_src, cores=tuple(range(CORES)))
    ncc.all_gather(srcs=[kv_src], dsts=[kv_dst], replica_group=replica_group, collective_dim=0)
    if pid == 0:
        nl.store(kvp[0:PT, 0:HEAD_DIM], value=nl.zeros((PT, HEAD_DIM), dtype=nl.bfloat16))
    for j in range(MT_LOC):
        for s in range(WORLD_SIZE):
            g = WORLD_SIZE * j + s
            nisa.dma_copy(dst=kvp[(g + 1) * PT:(g + 2) * PT, 0:HEAD_DIM],
                          src=kv_dst[s * SEQ_LOC + j * PT:s * SEQ_LOC + (j + 1) * PT, 0:HEAD_DIM])
    nisa.core_barrier(data=kvp, cores=tuple(range(CORES)))

    ncc.all_gather(srcs=[wqb_src, woa_src, wob_src], dsts=[wqb, wo_a, wo_b],
                   replica_group=replica_group, collective_dim=0)

    _index_mask(index_q, ik_hbm, iw, msk, scr, thr, atiles, rid)
    for r0, m in runs:
        _gemm_head_major(qr_q, wqb, q, m, Q_LORA, N_HEADS * HEAD_DIM, HEAD_DIM,
                         row=r0, hstride=SEQ_LOC, freqs_hbm=floc)
    _sparse_attention(q, kvp, ckv_hbm, msk, layers_24_attn_attn_sink,
                      floc, og, winl, atiles, rid)

    for r0, m in runs:
        for g in range(O_GROUPS):
            _gemm(og, wo_a, z_q, m, HEADS_PER_GROUP * HEAD_DIM, O_LORA,
                  x_row=g * SEQ_LOC + r0, w_row=g * O_LORA,
                  dst_row=r0, dst_col=g * O_LORA, qdq=True)
    for r0, m in runs:
        _gemm(z_q, wo_b, cc_src_a, m, O_GROUPS * O_LORA, CC_HALF, x_row=r0, dst_row=r0,
              out_dtype=CC_DTYPE)
    nisa.core_barrier(data=cc_src_a, cores=tuple(range(CORES)))
    ncc.all_gather(srcs=[cc_src_a], dsts=[cc_dst_a], replica_group=replica_group,
                   collective_dim=0)
    for r0, m in runs:
        _gemm(z_q, wo_b, cc_src_b, m, O_GROUPS * O_LORA, CC_HALF, x_row=r0, dst_row=r0,
              w_row=CC_HALF, out_dtype=CC_DTYPE)
    nisa.core_barrier(data=cc_src_b, cores=tuple(range(CORES)))
    ncc.all_gather(srcs=[cc_src_b], dsts=[cc_dst_b], replica_group=replica_group,
                   collective_dim=0)
    for half, cc_dst in ((0, cc_dst_a), (1, cc_dst_b)):
        for j in tiles:
            for s in range(WORLD_SIZE):
                g = WORLD_SIZE * j + s
                nisa.dma_copy(
                    dst=out2[g * PT:(g + 1) * PT, half * CC_HALF:(half + 1) * CC_HALF],
                    src=cc_dst[s * SEQ_LOC + j * PT:s * SEQ_LOC + (j + 1) * PT, 0:CC_HALF])
    nisa.core_barrier(data=out, cores=tuple(range(CORES)))
    return out
