"""`layers.20.attention` on 4 ranks, cut along the **query-head** dimension (`FLOORPLAN.md`:
`head x4 (allreduce)`). The four ranks rejoin by summing `wo_b`'s partials with one
`ncc.all_reduce` inside the traced graph; `torch.distributed` only bootstraps the process group,
and no tensor data passes through it.
"""

import torch

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

SEQ = 8192
DIM = 5120
HEAD_DIM = 512
N_HEADS = 16
ROPE_DIM = 64
Q_LORA = 1280
O_LORA = 1024
O_GROUPS = 2
WINDOW = 128
RATIO = 1
CSEQ = SEQ // RATIO
INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_TOPK = 512
EPS = 1e-20
INDEX_SCORE_SCALE = 1.0 / 64.0
NEG_INF = -1.0e30
NEG_BOUND = -1.0e30
SM_SCALE = HEAD_DIM ** -0.5
HEADS_PER_GROUP = N_HEADS // O_GROUPS
BISECT_STEPS = 9

IDX_SCORE_DTYPE = nl.bfloat16
IDX_ACC_DTYPE = nl.bfloat16

ATT_LOGIT_DTYPE = nl.float32

BISECT_SCALAR = 1

BISECT_VFRAC = 0.375

IDX_HG = 4

IDX_DB = 1

IDX_GB = 2

_VEC = nisa.engine.vector

DEQ_SIGN_SCALAR = 1
IDX_VEC_HG = 0
IDX_FUSE = 1
_DONE = object()
_PAIRING = ((0, 1, 6, 7, 8, 9, 14, 15), (2, 3, 4, 5, 10, 11, 12, 13))
PAIRING = 0
OB = 4

PT = 128
NT = 512
WIN_SPAN = 2 * PT
CHUNK = 4

FP8_BLK = 32
FP8_MAX = 448.0
FP8_AMAX_FLOOR = 1e-4
FP4_MAX = 6.0
FP4_BLK_KV = 16
FP4_BLK_IDX = 32

MGROUP_BYTES = 96 * 1024
EPI_MGROUP_BYTES = MGROUP_BYTES
HM_MGROUP = 8

W1_CHUNKS = 2
W1_SETS = 2


EPI_CHUNKS = 1

CORES = 2
MT = SEQ // PT
QG = NT * RATIO
NG = SEQ // QG
QPB = MT // NG

WORLD_SIZE = 4

OUT_DTYPE = nl.bfloat16

CC_DTYPE = nl.bfloat16

CC_ROWS = 1024


def _blocks(n, step):
    out = []
    off = 0
    while off < n:
        out.append((off, min(step, n - off)))
        off += step
    return out


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


def _shard_src(t, shard, n, k, rid, row, rows):
    if shard == 0:
        return _shard0(t, n, k, rid, row, rows)
    return _shard1(t, n, k, rid, row, rows)


def _load_bcast_shard(vec_hbm, n, rows, rid, dt=nl.float32):
    v = nl.ndarray((1, n), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=v, src=vec_hbm.reshape((WORLD_SIZE, n)).ap(
        pattern=[[n, 1], [1, n]], offset=0, scalar_offset=rid, indirect_dim=0))
    wb = nl.ndarray((rows, n), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=wb, src=nl.broadcast_to(v, (rows, n)))
    return wb


def _core_runs(pid, cores=CORES):
    if PAIRING:
        gbs = _PAIRING[pid]
        runs = []
        for g in sorted(gbs):
            if runs and runs[-1][0] + runs[-1][1] == g * QG:
                runs[-1] = (runs[-1][0], runs[-1][1] + QG)
            else:
                runs.append((g * QG, QG))
        return tuple(runs), tuple(gbs)
    per = NG // (2 * cores)
    a = (pid * per, pid * per + per)
    b = (NG - (pid + 1) * per, NG - pid * per)
    gr = [(a[0], b[1])] if a[1] == b[0] else [a, b]
    return (tuple((g0 * QG, (g1 - g0) * QG) for g0, g1 in gr),
            tuple(g for g0, g1 in gr for g in range(g0, g1)))


def _diag_width(mb):
    return min(NT, (mb + 1) * PT // RATIO)


def _load_bcast(vec_hbm, n, rows):
    return nl.broadcast_to(nl.load(vec_hbm.reshape((1, n)), dtype=nl.float32), (rows, n))


def _rmsnorm(tile, wb, n, out_dtype=nl.float32, rinv_hbm=None, row=0):
    rows = tile.shape[0]
    sq = nl.ndarray((rows, n), dtype=nl.float32, buffer=nl.sbuf)
    ss = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=tile, reduce_op=nl.add, reduce_res=ss,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    ms = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ms, data=ss, op0=nl.multiply, operand0=1.0 / n,
                       op1=nl.add, operand1=EPS)
    hn = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=hn, data=ms, op0=nl.multiply, operand0=-0.5)
    y0 = nl.rsqrt(ms)
    t = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t, data=nl.multiply(y0, y0), op0=nl.multiply, operand0=hn,
                       op1=nl.add, operand1=1.5)
    y = nl.multiply(y0, t)
    if rinv_hbm is not None:
        nl.store(rinv_hbm[row:row + rows, 0:1], value=y)
    out = nl.ndarray((rows, n), dtype=out_dtype, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=out, data=tile, op0=nl.multiply, operand0=y,
                              op1=nl.multiply, operand1=wb)
    return out


EXP_MASK = 0x7F800000
RECIP_BITS = 254 << 23
FP4_K_BITS = (22 << 23) + 0x400000
FP4_K_SCALE = 1.5 * (2.0 ** 22)
EXP_MASK_BF16 = 0x7F80


def _i32(shape):
    return nl.ndarray(shape, dtype=nl.int32, buffer=nl.sbuf)


def _ceil_exp2(x_f32):
    bumped = _i32(x_f32.shape)
    nisa.tensor_scalar(dst=bumped, data=x_f32.view(nl.int32), op0=nl.add, operand0=0x7FFFFF)
    e = _i32(x_f32.shape)
    nisa.tensor_scalar(dst=e, data=bumped, op0=nl.bitwise_and, operand0=EXP_MASK)
    return e


def _pow2_ceil_log2(x):
    sbits = _ceil_exp2(x)
    rbits = nl.copy(nl.subtract(RECIP_BITS, sbits))
    return sbits.view(nl.float32), rbits.view(nl.float32)


E4M3_TOP_BINADE = 248.0


def _cast_e4m3(v):
    safe = nl.minimum(nl.maximum(v, -240.0), 240.0)
    direct = nl.copy(nl.copy(safe, dtype=nl.float8_e4m3fn), dtype=nl.float32)
    halved = nl.copy(nl.copy(nl.multiply(v, 0.5), dtype=nl.float8_e4m3fn), dtype=nl.float32)
    return nl.where(nl.greater_equal(nl.abs(v), E4M3_TOP_BINADE),
                    nl.multiply(halved, 2.0), direct)


def _decode_e4m3(bytes_u8):
    shape = bytes_u8.shape

    def u8():
        return nl.ndarray(shape, dtype=nl.uint8, buffer=nl.sbuf)

    sg = u8()
    nisa.tensor_scalar(dst=sg, data=bytes_u8, op0=nl.bitwise_and, operand0=0x80,
                       engine=(nisa.engine.scalar if DEQ_SIGN_SCALAR else nisa.engine.unknown))
    b2 = u8()
    nisa.scalar_tensor_tensor(dst=b2, data=bytes_u8, op0=nl.subtract, operand0=8,
                              op1=nl.maximum, operand1=sg)
    return b2.view(nl.float8_e4m3fn)


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
    scale, rcp = _pow2_ceil_log2(_block_amax(blocked, FP8_AMAX_FLOOR, 2.0 / FP8_MAX))
    inv = rcp.broadcast(2, FP8_BLK)
    wide = scale.broadcast(2, FP8_BLK)
    q8 = nl.ndarray((rows, nb, FP8_BLK), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=q8, data1=blocked, data2=inv, op=nl.multiply)
    out = nl.ndarray((rows, nb, FP8_BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out, data1=q8, data2=wide, op=nl.multiply)
    return out.reshape((rows, n))


def _binade(blocked):
    shape = blocked.shape
    if blocked.dtype == nl.bfloat16:
        pwi = nl.ndarray(shape, dtype=nl.int16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=pwi, data=blocked.view(nl.int16), op0=nl.bitwise_and,
                           operand0=EXP_MASK_BF16)
        return pwi.view(nl.bfloat16)
    pwi = _i32(shape)
    nisa.tensor_scalar(dst=pwi, data=blocked.view(nl.int32), op0=nl.bitwise_and,
                       operand0=EXP_MASK)
    return pwi.view(nl.float32)


def _qdq_fp4(tile, rows, n, blk):
    nb = n // blk
    shape = (rows, nb, blk)
    blocked = tile.reshape(shape)
    e = _ceil_exp2(_block_amax(blocked, 6.0 * (2.0 ** -126), 1.0 / FP4_MAX))
    sc = nl.ndarray((rows, nb, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sc, data=e.view(nl.float32), op0=nl.multiply, operand0=FP4_K_SCALE)
    kf = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=kf, data=_binade(blocked), op0=nl.multiply,
                              operand0=FP4_K_SCALE, op1=nl.maximum,
                              operand1=sc.broadcast(2, blk))
    y = nl.add(blocked, kf, dtype=nl.float32)
    out = nl.ndarray(shape, dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out, data1=y, data2=kf, op=nl.subtract)
    return out.reshape((rows, n))


_E2M1_MID = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E2M1_STEP = (0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 2.0)
_E2M1_TIE_UP = (0, 1, 0, 1, 0, 1, 0)


def _qdq_fp4_e4m3(tile, rows, n, blk):
    nb = n // blk
    shape = (rows, nb, blk)
    blocked = tile.reshape(shape)
    scale = _cast_e4m3(_block_amax(blocked, 6.0 * (2.0 ** -9), 1.0 / FP4_MAX))
    s = scale.broadcast(2, blk)
    av = nl.ndarray(shape, dtype=blocked.dtype, buffer=nl.sbuf)
    nisa.activation(dst=av, op=nl.abs, data=blocked)
    acc = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    for i in range(len(_E2M1_MID)):
        thr = nl.multiply(scale, _E2M1_MID[i]).broadcast(2, blk)
        hit = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=hit, data1=av, data2=thr,
                           op=nl.greater_equal if _E2M1_TIE_UP[i] else nl.greater)
        if i == 0:
            nisa.tensor_scalar(dst=acc, data=hit, op0=nl.multiply, operand0=_E2M1_STEP[i])
        else:
            nisa.scalar_tensor_tensor(dst=acc, data=hit, op0=nl.multiply,
                                      operand0=_E2M1_STEP[i], op1=nl.add, operand1=acc)
    mag = nl.multiply(acc, s)
    out = nl.ndarray(shape, dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out, data1=mag, data2=nl.sign(blocked), op=nl.multiply)
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


def _dequant_weight(w_u8, s_u8, dst_hbm, n, k, dst_row=0, shard=None, rid=None):
    kb = k // FP8_BLK
    sper = PT // FP8_BLK
    sel = _scale_selector()
    for nb in nl.affine_range(n // PT):
        r0 = nb * PT
        if shard is None:
            wraw = nl.load(w_u8[r0:r0 + PT, 0:k], dtype=nl.uint8)
            sb = nl.load(s_u8[r0 // FP8_BLK:r0 // FP8_BLK + sper, 0:kb], dtype=nl.int32)
        else:
            wraw = nl.ndarray((PT, k), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.dma_copy(dst=wraw, src=_shard_src(w_u8, shard, n, k, rid, r0, PT))
            sraw = nl.ndarray((sper, kb), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.dma_copy(dst=sraw, src=_shard_src(s_u8, shard, n // FP8_BLK, kb, rid,
                                                   r0 // FP8_BLK, sper))
            sb = nl.copy(sraw, dtype=nl.int32)
        wq = _decode_e4m3(wraw)
        ex = nl.ndarray((PT, kb), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ex, stationary=sel,
                       moving=nl.copy(nl.add(sb, 1), dtype=nl.bfloat16), accumulate=False)
        sf = nl.copy(nl.left_shift(nl.copy(ex, dtype=nl.int32), 23)).view(nl.float32)
        rep = sf.reshape((PT, kb, 1)).broadcast(2, FP8_BLK)
        prod = nl.ndarray((PT, kb, FP8_BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prod, data1=wq.reshape((PT, kb, FP8_BLK)), data2=rep,
                           op=nl.multiply)
        nl.store(dst_hbm[dst_row + r0:dst_row + r0 + PT, 0:k],
                 value=prod.reshape((PT, k)))


def _mgroup(mtiles, ktiles, budget=MGROUP_BYTES):
    mg = 1
    while mg * 2 <= mtiles and (mg * 2) * ktiles * PT * 2 <= budget:
        mg *= 2
    return mg


def _stage_x(xt, x_hbm, ktiles, mg, row, x_col=0):
    sg = 4 if mg % 4 == 0 else (2 if mg % 2 == 0 else 1)
    flat = xt.reshape((PT, ktiles, mg * PT))
    x3 = x_hbm.reshape((x_hbm.shape[0], x_hbm.shape[1] // PT, PT))
    c0 = x_col // PT
    if mg > sg:
        groups = [(0, 1)]
        done = 1
        while done < mg:
            wd = min(sg, mg - done)
            groups.append((done, wd))
            done += wd
    else:
        groups = [(mi * sg, sg) for mi in range(mg // sg)]
    for m0, wd in groups:
        r0 = row + m0 * PT
        nisa.dma_transpose(dst=flat[0:PT, 0:ktiles, m0 * PT:(m0 + wd) * PT],
                           src=x3[r0:r0 + wd * PT, c0:c0 + ktiles, 0:PT])


def _gemm(x_hbm, w_hbm, dst_hbm, m, k, n, out_dtype=nl.bfloat16,
          x_row=0, x_col=0, w_row=0, dst_row=0, dst_col=0, qdq=False,
          mgb_bytes=MGROUP_BYTES):
    ktiles = k // PT
    if n <= NT:
        nt = n
    elif n % NT == 0:
        nt = NT
    elif n % 256 == 0:
        nt = 256
    else:
        nt = PT
    mg = _mgroup(m // PT, ktiles, mgb_bytes)
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
                    val = _qdq_fp8(acc, PT, nt)
                else:
                    val = nl.copy(acc, dtype=out_dtype)
                nl.store(dst_hbm[r:r + PT, dst_col + nb * nt:dst_col + (nb + 1) * nt],
                         value=val)


def _gemm_head_major(x_hbm, w_hbm, dst_hbm, m, k, n, hdim, row=0, hstride=0, freqs_hbm=None):
    ktiles = k // PT
    half = ROPE_DIM // 2
    rg = hdim // ROPE_DIM
    mg = min(_mgroup(m // PT, ktiles), HM_MGROUP)
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
                nisa.dma_transpose(dst=wt[0:PT, kb, 0:hdim],
                                   src=w_hbm[nb * hdim:(nb + 1) * hdim,
                                             kb * PT:(kb + 1) * PT])
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


def _norm_rows_both(src_hbm, w_hbm, dst_b, dst_q, m, n, row=0, rinv_hbm=None):
    wb = _load_bcast(w_hbm, n, PT)
    for mb in nl.affine_range(m // PT):
        r0 = row + mb * PT
        t = nl.load(src_hbm[r0:r0 + PT, 0:n], dtype=nl.bfloat16)
        tb = _rmsnorm(t, wb, n, out_dtype=nl.bfloat16, rinv_hbm=rinv_hbm, row=r0)
        nl.store(dst_b[r0:r0 + PT, 0:n], value=tb)
        nl.store(dst_q[r0:r0 + PT, 0:n], value=_qdq_fp8(tb, PT, n))


def _norm_quant_rows(src_hbm, w_hbm, dst_hbm, m, n, row=0, src_col=0, rinv_hbm=None):
    wb = _load_bcast(w_hbm, n, PT)
    for mb in nl.affine_range(m // PT):
        r0 = row + mb * PT
        t = nl.load(src_hbm[r0:r0 + PT, src_col:src_col + n], dtype=nl.bfloat16)
        tb = _rmsnorm(t, wb, n, out_dtype=nl.bfloat16, rinv_hbm=rinv_hbm, row=r0)
        nl.store(dst_hbm[r0:r0 + PT, 0:n], value=_qdq_fp8(tb, PT, n))


def _norm_rows(src_hbm, w_hbm, dst_hbm, m, n, row=0, src_col=0):
    wb = _load_bcast(w_hbm, n, PT)
    for mb in nl.affine_range(m // PT):
        r0 = row + mb * PT
        t = nl.load(src_hbm[r0:r0 + PT, src_col:src_col + n], dtype=nl.bfloat16)
        nl.store(dst_hbm[r0:r0 + PT, 0:n],
                 value=_rmsnorm(t, wb, n, out_dtype=nl.bfloat16))


def _norm_rope_quant8_rows(src_hbm, w_hbm, freqs_hbm, dst_hbm, m, n, row=0, src_col=0,
                           dst_row=0):
    half = ROPE_DIM // 2
    rg = n // ROPE_DIM
    wb = _load_bcast(w_hbm, n, PT)
    for mb in nl.affine_range(m // PT):
        r0 = row + mb * PT
        t = nl.load(src_hbm[r0:r0 + PT, src_col:src_col + n], dtype=nl.bfloat16)
        tb = _rmsnorm(t, wb, n, out_dtype=nl.bfloat16)
        cos = nl.load(freqs_hbm[r0:r0 + PT, 0:half, 0], dtype=nl.float32)
        sin = nl.load(freqs_hbm[r0:r0 + PT, 0:half, 1], dtype=nl.float32)
        v = tb.reshape((PT, rg, half, 2))
        a = v[0:PT, rg - 1, 0:half, 0]
        b = v[0:PT, rg - 1, 0:half, 1]
        ac = nl.multiply(a, cos, dtype=nl.float32)
        bs = nl.multiply(b, sin, dtype=nl.float32)
        as_ = nl.multiply(a, sin, dtype=nl.float32)
        bc = nl.multiply(b, cos, dtype=nl.float32)
        nisa.tensor_tensor(dst=a, data1=ac, data2=bs, op=nl.subtract)
        nisa.tensor_tensor(dst=b, data1=as_, data2=bc, op=nl.add)
        nl.store(dst_hbm[dst_row + mb * PT:dst_row + (mb + 1) * PT, 0:n],
                 value=_qdq_fp8(tb, PT, n))


def _norm_rope_quant4_rows(src_hbm, w_hbm, freqs_hbm, dst_hbm, m, width, pos_stride, blk,
                           row=0):
    half = ROPE_DIM // 2
    rg = width // ROPE_DIM
    wb = _load_bcast(w_hbm, width, PT)
    for mb in nl.affine_range(m // PT):
        r0 = row + mb * PT
        t = nl.load(src_hbm[r0:r0 + PT, 0:width], dtype=nl.bfloat16)
        tb = _rmsnorm(t, wb, width, out_dtype=nl.bfloat16)
        p0 = r0 * pos_stride
        p1 = p0 + PT * pos_stride
        cos = nl.load(freqs_hbm[p0:p1:pos_stride, 0:half, 0], dtype=nl.float32)
        sin = nl.load(freqs_hbm[p0:p1:pos_stride, 0:half, 1], dtype=nl.float32)
        v = tb.reshape((PT, rg, half, 2))
        a = v[0:PT, rg - 1, 0:half, 0]
        b = v[0:PT, rg - 1, 0:half, 1]
        ac = nl.multiply(a, cos, dtype=nl.float32)
        bs = nl.multiply(b, sin, dtype=nl.float32)
        as_ = nl.multiply(a, sin, dtype=nl.float32)
        bc = nl.multiply(b, cos, dtype=nl.float32)
        nisa.tensor_tensor(dst=a, data1=ac, data2=bs, op=nl.subtract)
        nisa.tensor_tensor(dst=b, data1=as_, data2=bc, op=nl.add)
        nl.store(dst_hbm[r0:r0 + PT, 0:width], value=_qdq_fp4(tb, PT, width, blk))


def _rope_quant4_e4m3_rows(src_hbm, freqs_hbm, dst_hbm, m, width, pos_stride, blk, row=0):
    half = ROPE_DIM // 2
    rg = width // ROPE_DIM
    for mb in nl.affine_range(m // PT):
        r0 = row + mb * PT
        tb = nl.ndarray((PT, width), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=tb, src=src_hbm[r0:r0 + PT, 0:width])
        p0 = r0 * pos_stride
        p1 = p0 + PT * pos_stride
        cos = nl.load(freqs_hbm[p0:p1:pos_stride, 0:half, 0], dtype=nl.float32)
        sin = nl.load(freqs_hbm[p0:p1:pos_stride, 0:half, 1], dtype=nl.float32)
        v = tb.reshape((PT, rg, half, 2))
        a = v[0:PT, rg - 1, 0:half, 0]
        b = v[0:PT, rg - 1, 0:half, 1]
        ac = nl.multiply(a, cos, dtype=nl.float32)
        bs = nl.multiply(b, sin, dtype=nl.float32)
        as_ = nl.multiply(a, sin, dtype=nl.float32)
        bc = nl.multiply(b, cos, dtype=nl.float32)
        nisa.tensor_tensor(dst=a, data1=ac, data2=bs, op=nl.subtract)
        nisa.tensor_tensor(dst=b, data1=as_, data2=bc, op=nl.add)
        nl.store(dst_hbm[r0:r0 + PT, 0:width], value=_qdq_fp4_e4m3(tb, PT, width, blk))


def _index_q_build(qr_hbm, w_hbm, freqs_hbm, dst_hbm, m, row=0, mg=4, hg_heads=4, bg=2):
    half = ROPE_DIM // 2
    ktiles = Q_LORA // PT
    hd = hg_heads * INDEX_DIM
    for mgb in nl.affine_range((m // PT) // mg):
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
                nisa.dma_transpose(dst=wt[0:PT, kb, 0:hd],
                                   src=w_hbm[hg * hd:(hg + 1) * hd, kb * PT:(kb + 1) * PT])
            for mb0 in nl.static_range(mg // bg):
                tb = nl.ndarray((PT, bg, hd), dtype=nl.bfloat16, buffer=nl.sbuf)
                for mj in nl.static_range(bg):
                    mi = mb0 * bg + mj
                    acc = nl.ndarray((PT, hd), dtype=nl.float32, buffer=nl.psum)
                    for kb in nl.static_range(ktiles):
                        nisa.nc_matmul(dst=acc, stationary=xt[0:PT, kb, mi, 0:PT],
                                       moving=wt[0:PT, kb, 0:hd], accumulate=(kb > 0))
                    nisa.tensor_copy(dst=tb[0:PT, mj, 0:hd], src=acc)
                v = tb.reshape((PT, bg, hg_heads, 2, half, 2))
                a = v[0:PT, 0:bg, 0:hg_heads, 1, 0:half, 0]
                b = v[0:PT, 0:bg, 0:hg_heads, 1, 0:half, 1]
                c = cos4[0:PT, mb0 * bg:(mb0 + 1) * bg, 0:hg_heads, 0:half]
                s = sin4[0:PT, mb0 * bg:(mb0 + 1) * bg, 0:hg_heads, 0:half]
                ac = nl.multiply(a, c, dtype=nl.float32)
                bs = nl.multiply(b, s, dtype=nl.float32)
                as_ = nl.multiply(a, s, dtype=nl.float32)
                bc = nl.multiply(b, c, dtype=nl.float32)
                nisa.tensor_tensor(dst=a, data1=ac, data2=bs, op=nl.subtract)
                nisa.tensor_tensor(dst=b, data1=as_, data2=bc, op=nl.add)
                qq = _qdq_fp4(tb.reshape((PT, bg * hd)), PT, bg * hd,
                              FP4_BLK_IDX).reshape((PT, bg, hd))
                for mj in nl.static_range(bg):
                    r0 = row + (mgb * mg + mb0 * bg + mj) * PT
                    nl.store(dst_hbm[r0:r0 + PT, hg * hd:(hg + 1) * hd],
                             value=qq[0:PT, mj, 0:hd])


def _index_mask(index_q, ik_hbm, front_hbm, rinv_hbm, msk_hbm, gbs):
    for _ in _index_mask_gen(index_q, ik_hbm, front_hbm, rinv_hbm, msk_hbm, gbs):
        pass


def _index_mask_gen(index_q, ik_hbm, front_hbm, rinv_hbm, msk_hbm, gbs):
    iq3 = index_q.reshape((SEQ, INDEX_HEADS, INDEX_DIM))

    kblk = max(gbs) + 1
    ikt_all = nl.ndarray((INDEX_DIM, kblk, NT), dtype=nl.bfloat16, buffer=nl.sbuf)
    for gb in nl.affine_range(kblk):
        nisa.dma_transpose(dst=ikt_all[0:INDEX_DIM, gb, 0:NT],
                           src=ik_hbm[gb * NT:(gb + 1) * NT, 0:INDEX_DIM])

    bpos_i = nl.ndarray((PT, PT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=bpos_i, pattern=[[1, PT]], offset=0, channel_multiplier=0)
    bpos = nl.copy(bpos_i, dtype=nl.float32)
    prow_i = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=prow_i, pattern=[[1, 1]], offset=1, channel_multiplier=1)
    clim = nl.copy(prow_i, dtype=nl.float32)
    neg = nl.full((PT, NT), NEG_INF, dtype=IDX_SCORE_DTYPE)
    big = nl.full((PT, NT), -NEG_INF, dtype=IDX_SCORE_DTYPE)

    for qb in gbs:
        gbn = qb + 1
        for mb in nl.static_range(QPB):
            gmb = qb * QPB + mb
            q0 = gmb * PT
            wrow = nl.ndarray((PT, INDEX_HEADS), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=wrow,
                               data=nl.load(front_hbm[q0:q0 + PT, 0:INDEX_HEADS],
                                            dtype=nl.float32),
                               op0=nl.multiply,
                               operand0=nl.load(rinv_hbm[q0:q0 + PT, 0:1], dtype=nl.float32),
                               op1=nl.multiply, operand1=INDEX_SCORE_SCALE)
            iqt = nl.ndarray((INDEX_DIM, INDEX_HEADS, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_transpose(dst=iqt[0:INDEX_DIM, 0:INDEX_HEADS, 0:PT],
                               src=iq3[q0:q0 + PT, 0:INDEX_HEADS, 0:INDEX_DIM])
            dw = _diag_width(mb)
            rows = nl.ndarray((PT, gbn, NT), dtype=IDX_SCORE_DTYPE, buffer=nl.sbuf)
            nhg = INDEX_HEADS // IDX_HG
            if dw < NT:
                nisa.tensor_copy(dst=rows[0:PT, qb, dw:NT], src=neg[0:PT, 0:NT - dw])
            rowf = rows.reshape((PT, gbn * NT))
            bats = []
            g0 = 0
            while g0 < qb:
                bats.append((g0, min(IDX_GB, qb - g0)))
                g0 += bats[-1][1]
            bats.append((qb, 1))
            for gb0, gk in bats:
                diag = gb0 == qb
                bw = dw if diag else NT
                nw = (gk - 1) * NT + bw
                tgt = rowf[0:PT, gb0 * NT:gb0 * NT + nw]
                part = nl.ndarray((PT, IDX_HG, gk, NT), dtype=IDX_ACC_DTYPE, buffer=nl.sbuf)
                ptf = part.reshape((PT, IDX_HG, gk * NT))
                for hg in nl.static_range(nhg):
                    rect = nl.ndarray((PT, IDX_HG, gk, NT), dtype=nl.bfloat16, buffer=nl.sbuf)
                    rcf = rect.reshape((PT, IDX_HG, gk * NT))
                    for gi in nl.static_range(gk):
                        ps = nl.ndarray((PT, IDX_HG, NT), dtype=nl.float32, buffer=nl.psum)
                        for j in range(IDX_HG):
                            nisa.nc_matmul(dst=ps[0:PT, j, 0:bw],
                                           stationary=iqt[0:INDEX_DIM, hg * IDX_HG + j, 0:PT],
                                           moving=ikt_all[0:INDEX_DIM, gb0 + gi, 0:bw],
                                           accumulate=False)
                        if hg < IDX_VEC_HG:
                            nisa.tensor_scalar(dst=rect[0:PT, 0:IDX_HG, gi, 0:bw],
                                               data=ps[0:PT, 0:IDX_HG, 0:bw], op0=nl.maximum,
                                               operand0=0.0, engine=_VEC)
                        else:
                            nisa.activation(dst=rect[0:PT, 0:IDX_HG, gi, 0:bw], op=nl.relu,
                                            data=ps[0:PT, 0:IDX_HG, 0:bw])
                    for j in range(IDX_HG):
                        hb = hg * IDX_HG + j
                        rj = rcf[0:PT, j, 0:nw]
                        pj = ptf[0:PT, j, 0:nw]
                        if hg == 0:
                            nisa.tensor_scalar(dst=pj, data=rj, op0=nl.multiply,
                                               operand0=wrow[0:PT, hb:hb + 1])
                        else:
                            nisa.scalar_tensor_tensor(dst=pj, data=rj, op0=nl.multiply,
                                                      operand0=wrow[0:PT, hb:hb + 1],
                                                      op1=nl.add, operand1=pj)
                nacc = IDX_HG
                while nacc > 2:
                    nacc //= 2
                    for j in range(nacc):
                        nisa.tensor_tensor(dst=ptf[0:PT, j, 0:nw], data1=ptf[0:PT, j, 0:nw],
                                           data2=ptf[0:PT, j + nacc, 0:nw], op=nl.add)
                nisa.tensor_tensor(dst=tgt, data1=ptf[0:PT, 0, 0:nw],
                                   data2=ptf[0:PT, 1, 0:nw], op=nl.add)
                if diag:
                    reach = nl.ndarray((PT, PT), dtype=nl.uint8, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=reach, data=bpos, op0=nl.less, operand0=clim)
                    tsub = rows[0:PT, qb, dw - PT:dw]
                    nisa.tensor_copy(dst=tsub, src=nl.where(reach, tsub, neg[0:PT, 0:PT]))

            rflat = rows.reshape((PT, gbn * NT))
            hi = nl.max(rflat, axis=1, keepdims=True, dtype=nl.float32)
            r0t = rows[0:PT, 0, 0:NT]
            if gbn == 1:
                lsrc = nl.where(nl.greater(r0t, nl.multiply(big, -0.5)), r0t, big)
            else:
                lsrc = r0t
            lo = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=lo, data=nl.min(lsrc, axis=1, keepdims=True,
                                                   dtype=nl.float32),
                               op0=nl.minimum, operand0=hi, op1=nl.subtract, operand1=1.0)
            dw_ = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=dw_, data=hi, op0=nl.subtract, operand0=lo)
            mid = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            cnt = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            enough = nl.ndarray((PT, 1), dtype=nl.uint8, buffer=nl.sbuf)
            if BISECT_SCALAR:
                nv = min(gbn * NT, ((int(gbn * NT * BISECT_VFRAC) + PT // 2) // PT) * PT)
                ns = gbn * NT - nv
                thr = float(2 * INDEX_TOPK - ns)
                comb = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
                sc = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
                sgn = (nl.ndarray((PT, ns), dtype=nl.bfloat16, buffer=nl.sbuf)
                       if ns else None)
                over = (nl.ndarray((PT, nv), dtype=nl.uint8, buffer=nl.sbuf)
                        if nv else None)
                for bs in nl.static_range(BISECT_STEPS):
                    nisa.tensor_scalar(dst=mid, data=dw_, op0=nl.multiply,
                                       operand0=0.5 ** (bs + 1), op1=nl.add, operand1=lo)
                    if ns:
                        nisa.activation(dst=sgn, op=nl.sign, data=rflat[0:PT, nv:nv + ns],
                                        bias=mid, scale=-1.0,
                                        reduce_op=nl.add, reduce_res=sc,
                                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
                    if nv:
                        nisa.tensor_scalar_reduce(dst=over, data=rflat[0:PT, 0:nv],
                                                  op0=nl.greater, operand0=mid,
                                                  reduce_op=nl.add, reduce_res=cnt)
                    if nv and ns:
                        nisa.scalar_tensor_tensor(dst=comb, data=cnt, op0=nl.multiply,
                                                  operand0=2.0, op1=nl.subtract, operand1=sc)
                    elif nv:
                        nisa.tensor_scalar(dst=comb, data=cnt, op0=nl.multiply, operand0=2.0)
                    else:
                        nisa.tensor_scalar(dst=comb, data=sc, op0=nl.multiply, operand0=-1.0)
                    nisa.tensor_scalar(dst=enough, data=comb, op0=nl.greater_equal,
                                       operand0=thr)
                    nisa.tensor_copy_predicated(src=mid, dst=lo, predicate=enough)
            for bs in nl.static_range(0 if BISECT_SCALAR else BISECT_STEPS):
                nisa.tensor_scalar(dst=mid, data=dw_, op0=nl.multiply,
                                   operand0=0.5 ** (bs + 1), op1=nl.add, operand1=lo)
                over = nl.ndarray((PT, gbn * NT), dtype=nl.uint8, buffer=nl.sbuf)
                nisa.tensor_scalar_reduce(dst=over, data=rflat, op0=nl.greater, operand0=mid,
                                          reduce_op=nl.add, reduce_res=cnt)
                nisa.tensor_scalar(dst=enough, data=cnt, op0=nl.greater_equal,
                                   operand0=float(INDEX_TOPK))
                nisa.tensor_copy_predicated(src=mid, dst=lo, predicate=enough)
            mt_ = nl.ndarray((PT, gbn * NT), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mt_, data=rflat, op0=nl.greater, operand0=lo)
            nl.store(msk_hbm[q0:q0 + PT, 0:gbn * NT], value=mt_)
            yield


def _sparse_attention(q_hbm, kvp_hbm, ckv_hbm, msk_hbm, sink_hbm, freqs_hbm, og_hbm, gbs, rid):
    for _ in _sparse_attention_gen(q_hbm, kvp_hbm, ckv_hbm, msk_hbm, sink_hbm, freqs_hbm,
                                   og_hbm, gbs, rid):
        pass


def _sparse_attention_gen(q_hbm, kvp_hbm, ckv_hbm, msk_hbm, sink_hbm, freqs_hbm, og_hbm, gbs,
                          rid):
    dt = HEAD_DIM // PT
    st = NT // PT
    cmax = (max(gbs) + 1) * NT
    q3 = q_hbm.reshape((N_HEADS * SEQ, dt, PT))
    ckv3 = ckv_hbm.reshape((CSEQ, dt, PT))
    kvp3 = kvp_hbm.reshape((SEQ + PT, dt, PT))

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
    pwin_zero = nl.copy(nl.greater(nl.multiply(nonneg, inwin), 0.5), dtype=nl.uint8)
    sinkb = _load_bcast_shard(sink_hbm, N_HEADS, PT, rid)

    ckv_t = nl.ndarray((PT, dt, cmax), dtype=nl.bfloat16, buffer=nl.sbuf)
    ckv_n = nl.ndarray((PT, cmax // PT, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    for gp in nl.affine_range(cmax // PT):
        nisa.dma_copy(dst=ckv_n[0:PT, gp, 0:HEAD_DIM],
                      src=ckv_hbm[gp * PT:(gp + 1) * PT, 0:HEAD_DIM])
        nisa.dma_transpose(dst=ckv_t[0:PT, 0:dt, gp * PT:(gp + 1) * PT],
                           src=ckv3[gp * PT:(gp + 1) * PT, 0:dt, 0:PT])

    for qb in gbs:
        gbn = qb + 1
        for mb in nl.static_range(QPB):
            gmb = qb * QPB + mb
            q0 = gmb * PT
            pwin = pwin_zero if gmb == 0 else pwin_gen
            dw = _diag_width(mb)
            win_t = nl.ndarray((PT, dt, WIN_SPAN), dtype=nl.bfloat16, buffer=nl.sbuf)
            win_n = nl.ndarray((PT, 2, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
            for wt in nl.static_range(2):
                r0 = q0 + wt * PT
                nisa.dma_copy(dst=win_n[0:PT, wt, 0:HEAD_DIM],
                              src=kvp_hbm[r0:r0 + PT, 0:HEAD_DIM])
                nisa.dma_transpose(dst=win_t[0:PT, 0:dt, wt * PT:(wt + 1) * PT],
                                   src=kvp3[r0:r0 + PT, 0:dt, 0:PT])
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
                                           src=q3[h * SEQ + q0:h * SEQ + q0 + PT, 0:dt, 0:PT])
                        bmax = nl.ndarray((PT, gbn + 1), dtype=nl.float32, buffer=nl.sbuf)
                        wmax = bmax[0:PT, gbn:gbn + 1]
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
                        lcs = nl.ndarray((PT, (gbn - 1) * NT + dw), dtype=ATT_LOGIT_DTYPE,
                                         buffer=nl.sbuf)
                        for gb in nl.static_range(gbn):
                            bw = NT if gb < gbn - 1 else dw
                            keep = kp[0:PT, gb * NT:gb * NT + bw]
                            lcp = nl.ndarray((PT, bw), dtype=nl.float32, buffer=nl.psum)
                            for dc in nl.static_range(dt):
                                nisa.nc_matmul(dst=lcp, stationary=qt[0:PT, dc, 0:PT],
                                               moving=ckv_t[0:PT, dc, gb * NT:gb * NT + bw],
                                               accumulate=(dc > 0))
                            nisa.select_reduce(dst=lcs[0:PT, gb * NT:gb * NT + bw],
                                               predicate=keep, on_true=lcp,
                                               on_false=NEG_BOUND,
                                               reduce_res=bmax[0:PT, gb:gb + 1],
                                               reduce_cmd=nisa.reduce_cmd.reset_reduce,
                                               reduce_op=nl.maximum)
                        negm = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_scalar_reduce(dst=bmax, data=bmax, op0=nl.multiply,
                                                  operand0=-SM_SCALE, reduce_op=nl.min,
                                                  reduce_res=negm)
                        nch = (gbn + CHUNK - 1) // CHUNK
                        dcol = nl.ndarray((PT, nch + 2), dtype=nl.float32, buffer=nl.sbuf)
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
                            cw = gk * NT if g0 + gk < gbn else (gk - 1) * NT + dw
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
                        nisa.activation(dst=dcol[0:PT, nch + 1:nch + 2], op=nl.exp,
                                        data=sinkb[0:PT, h:h + 1], bias=negm, scale=1.0)
                        den = nl.sum(dcol[0:PT, 0:nch + 2], axis=1, keepdims=True)
                        rden = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.reciprocal(dst=rden, data=den)
                        nisa.tensor_scalar(dst=obuf[0:PT, hi, 0:HEAD_DIM], data=acc,
                                           op0=nl.multiply, operand0=rden)
                    ov = obuf.reshape((PT, OB, HEAD_DIM // ROPE_DIM, ROPE_DIM // 2, 2))
                    ra = ov[0:PT, 0:OB, HEAD_DIM // ROPE_DIM - 1, 0:ROPE_DIM // 2, 0]
                    rb = ov[0:PT, 0:OB, HEAD_DIM // ROPE_DIM - 1, 0:ROPE_DIM // 2, 1]
                    rac = nl.multiply(ra, rc8)
                    rbs = nl.multiply(rb, rs8)
                    ras = nl.multiply(ra, rs8)
                    rbc = nl.multiply(rb, rc8)
                    nisa.tensor_tensor(dst=ra, data1=rac, data2=rbs, op=nl.add)
                    nisa.tensor_tensor(dst=rb, data1=rbc, data2=ras, op=nl.subtract)
                    nl.store(og_hbm[g * SEQ + q0:g * SEQ + q0 + PT,
                                    hbat * OB * HEAD_DIM:(hbat + 1) * OB * HEAD_DIM],
                             value=obuf)
            yield


@nki.jit
def _impl(hidden_states, layers_20_attn_attn_sink, layers_20_attn_compress_kv_cache,
          layers_20_attn_compressor_norm_weight, layers_20_attn_compressor_wkv_weight,
          layers_20_attn_freqs_cis, layers_20_attn_indexer_k_cache,
          layers_20_attn_indexer_k_norm_weight, layers_20_attn_indexer_weights_proj_weight,
          layers_20_attn_indexer_wk_weight, layers_20_attn_indexer_wq_b_scale,
          layers_20_attn_indexer_wq_b_weight, layers_20_attn_kv_norm_weight,
          layers_20_attn_q_norm_weight, layers_20_attn_window_kv_cache,
          layers_20_attn_wkv_scale, layers_20_attn_wkv_weight, layers_20_attn_wo_a_scale,
          layers_20_attn_wo_a_weight, layers_20_attn_wo_b_scale, layers_20_attn_wo_b_weight,
          layers_20_attn_wq_a_scale, layers_20_attn_wq_a_weight, layers_20_attn_wq_b_scale,
          layers_20_attn_wq_b_weight, layers_20_attn_norm_weight, front, replica_group):
    pid = nl.program_id(0)
    runs, gbs = _core_runs(pid)
    agbs = tuple(sorted(gbs, reverse=True))

    x_hbm = hidden_states.reshape((SEQ, DIM))
    out = nl.ndarray((1, SEQ, DIM), dtype=OUT_DTYPE, buffer=nl.shared_hbm)
    out2 = out.reshape((SEQ, DIM))
    cc_src = nl.ndarray((SEQ, DIM), dtype=CC_DTYPE, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((SEQ, DIM), dtype=CC_DTYPE, buffer=nl.shared_hbm, name="cc_dst")

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
    if pid == 0:
        _dequant_weight(layers_20_attn_wq_a_weight, layers_20_attn_wq_a_scale, wqkv,
                        Q_LORA, DIM)
        _dequant_weight(layers_20_attn_wkv_weight, layers_20_attn_wkv_scale, wqkv,
                        HEAD_DIM, DIM, dst_row=Q_LORA)
        _dequant_weight(layers_20_attn_indexer_wq_b_weight,
                        layers_20_attn_indexer_wq_b_scale, iwqb,
                        INDEX_HEADS * INDEX_DIM, Q_LORA)
        _dequant_weight(layers_20_attn_wo_a_weight, layers_20_attn_wo_a_scale, wo_a,
                        O_GROUPS * O_LORA, HEADS_PER_GROUP * HEAD_DIM, shard=0, rid=rid)
    else:
        _dequant_weight(layers_20_attn_wq_b_weight, layers_20_attn_wq_b_scale, wqb,
                        N_HEADS * HEAD_DIM, Q_LORA, shard=0, rid=rid)
        _dequant_weight(layers_20_attn_wo_b_weight, layers_20_attn_wo_b_scale, wo_b,
                        DIM, O_GROUPS * O_LORA, shard=1, rid=rid)
    if CORES > 1:
        for wbuf in (wqkv, wqb, iwqb):
            nisa.core_barrier(data=wbuf, cores=tuple(range(CORES)))

    w1 = tuple(
        (nl.ndarray((SEQ, DIM), dtype=nl.bfloat16, buffer=nl.private_hbm),
         nl.ndarray((SEQ, DIM), dtype=nl.bfloat16, buffer=nl.private_hbm),
         nl.ndarray((SEQ, QKV_N), dtype=nl.bfloat16, buffer=nl.private_hbm),
         nl.ndarray((SEQ, Q_LORA), dtype=nl.bfloat16, buffer=nl.private_hbm),
         nl.ndarray((CSEQ, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm),
         nl.ndarray((CSEQ, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm),
         nl.ndarray((CSEQ, INDEX_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm))
        for _ in range(W1_SETS))
    rinv = nl.ndarray((SEQ, 1), dtype=nl.float32, buffer=nl.private_hbm)
    kvp = nl.ndarray((SEQ + PT, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    index_q = nl.ndarray((SEQ, INDEX_HEADS * INDEX_DIM), dtype=nl.bfloat16,
                         buffer=nl.private_hbm)
    q = nl.ndarray((N_HEADS * SEQ, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    msk = nl.ndarray((SEQ, CSEQ), dtype=nl.uint8, buffer=nl.private_hbm)
    msk2 = (nl.ndarray((SEQ, CSEQ), dtype=nl.uint8, buffer=nl.private_hbm)
            if IDX_FUSE else msk)
    og = nl.ndarray((O_GROUPS * SEQ, HEADS_PER_GROUP * HEAD_DIM), dtype=nl.bfloat16,
                    buffer=nl.private_hbm)
    z_qs = tuple(nl.ndarray((SEQ, O_GROUPS * O_LORA), dtype=nl.bfloat16, buffer=nl.private_hbm)
                 for _ in range(W1_SETS if EPI_CHUNKS > 1 else 1))
    index_k = nl.ndarray((CSEQ, INDEX_DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                         name="index_k")
    compress_kv = nl.ndarray((CSEQ, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                             name="compress_kv")

    nl.store(kvp[0:PT, 0:HEAD_DIM], value=nl.zeros((PT, HEAD_DIM), dtype=nl.bfloat16))

    w1_runs = tuple((r0 + ci * (m // W1_CHUNKS), m // W1_CHUNKS, ci == 0 and r0 > 0)
                    for r0, m in runs for ci in range(W1_CHUNKS))

    for ci, (r0, m, pre) in enumerate(w1_runs):
        xn, xn_q, qakv, qr_q, cpa, latent, ika = w1[ci % W1_SETS]
        if pre:
            p0 = r0 - PT
            _norm_quant_rows(x_hbm, layers_20_attn_norm_weight, xn_q, PT, DIM, row=p0)
            _gemm(xn_q, wqkv, qakv, PT, DIM, HEAD_DIM, x_row=p0, w_row=Q_LORA,
                  dst_row=p0, dst_col=Q_LORA)
            _norm_rope_quant8_rows(qakv, layers_20_attn_kv_norm_weight,
                                   layers_20_attn_freqs_cis, kvp, PT, HEAD_DIM,
                                   row=p0, src_col=Q_LORA, dst_row=p0 + PT)

        _norm_rows_both(x_hbm, layers_20_attn_norm_weight, xn, xn_q, m, DIM, row=r0,
                        rinv_hbm=rinv)
        _gemm(xn_q, wqkv, qakv, m, DIM, QKV_N, x_row=r0, dst_row=r0)
        _norm_rope_quant8_rows(qakv, layers_20_attn_kv_norm_weight,
                               layers_20_attn_freqs_cis, kvp, m, HEAD_DIM,
                               row=r0, src_col=Q_LORA, dst_row=r0 + PT)

        _norm_quant_rows(qakv, layers_20_attn_q_norm_weight, qr_q, m, Q_LORA, row=r0)
        _index_q_build(qr_q, iwqb, layers_20_attn_freqs_cis, index_q, m, row=r0)
        _gemm_head_major(qr_q, wqb, q, m, Q_LORA, N_HEADS * HEAD_DIM, HEAD_DIM,
                         row=r0, hstride=SEQ, freqs_hbm=layers_20_attn_freqs_cis)

        _gemm(xn, layers_20_attn_compressor_wkv_weight, cpa, m, DIM, HEAD_DIM,
              x_row=r0, dst_row=r0)
        _norm_rows(cpa, layers_20_attn_compressor_norm_weight, latent, m, HEAD_DIM, row=r0)
        _gemm(latent, layers_20_attn_indexer_wk_weight, ika, m, HEAD_DIM, INDEX_DIM,
              x_row=r0, dst_row=r0)
        _norm_rope_quant4_rows(ika, layers_20_attn_indexer_k_norm_weight,
                               layers_20_attn_freqs_cis, index_k, m, INDEX_DIM, RATIO,
                               FP4_BLK_IDX, row=r0)
        _rope_quant4_e4m3_rows(latent, layers_20_attn_freqs_cis, compress_kv, m, HEAD_DIM,
                               RATIO, FP4_BLK_KV, row=r0)

    if CORES > 1:
        nisa.core_barrier(data=index_k, cores=tuple(range(CORES)))
        nisa.core_barrier(data=compress_kv, cores=tuple(range(CORES)))
        nisa.core_barrier(data=wo_a, cores=tuple(range(CORES)))
        nisa.core_barrier(data=wo_b, cores=tuple(range(CORES)))

    if IDX_FUSE:
        lo_g = agbs[len(agbs) // 2:]
        hi_g = agbs[:len(agbs) // 2]
        _index_mask(index_q, index_k, front, rinv, msk, lo_g)
        ms = _index_mask_gen(index_q, index_k, front, rinv, msk2, hi_g)
        sa = _sparse_attention_gen(q, kvp, compress_kv, msk, layers_20_attn_attn_sink,
                                   layers_20_attn_freqs_cis, og, lo_g, rid)
        if IDX_FUSE == 2:
            while True:
                m = next(ms, _DONE)
                a = next(sa, _DONE)
                if m is _DONE and a is _DONE:
                    break
        else:
            for _ in ms:
                pass
            for _ in sa:
                pass
        _sparse_attention(q, kvp, compress_kv, msk2, layers_20_attn_attn_sink,
                          layers_20_attn_freqs_cis, og, hi_g, rid)
    else:
        _index_mask(index_q, index_k, front, rinv, msk, agbs)
        _sparse_attention(q, kvp, compress_kv, msk, layers_20_attn_attn_sink,
                          layers_20_attn_freqs_cis, og, agbs, rid)

    for ci, (r0, m) in enumerate((r0 + cj * (m // EPI_CHUNKS), m // EPI_CHUNKS)
                                 for r0, m in runs for cj in range(EPI_CHUNKS)):
        z_q = z_qs[ci % len(z_qs)]
        for g in range(O_GROUPS):
            _gemm(og, wo_a, z_q, m, HEADS_PER_GROUP * HEAD_DIM, O_LORA,
                  x_row=g * SEQ + r0, w_row=g * O_LORA,
                  dst_row=r0, dst_col=g * O_LORA, qdq=True,
                  mgb_bytes=EPI_MGROUP_BYTES)
        _gemm(z_q, wo_b, cc_src, m, O_GROUPS * O_LORA, DIM, x_row=r0, dst_row=r0,
              out_dtype=CC_DTYPE, mgb_bytes=EPI_MGROUP_BYTES)

    if CORES > 1:
        nisa.core_barrier(data=cc_src, cores=tuple(range(CORES)))
    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], replica_group=replica_group, op=nl.add)
    for r0, m in runs:
        for off, sz in _blocks(m, CC_ROWS):
            nisa.dma_copy(dst=out2[r0 + off:r0 + off + sz, 0:DIM],
                          src=cc_dst[r0 + off:r0 + off + sz, 0:DIM])

    if CORES > 1:
        nisa.core_barrier(data=out, cores=tuple(range(CORES)))
    return out


def _linear(x, w):
    try:
        lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device="meta", dtype=w.dtype)
    except Exception:
        lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
    del lin.weight
    lin.weight = w
    return lin(x)


def _front(hidden_states, wp_w, norm_w):
    w = (wp_w.to(torch.float32) * norm_w.to(torch.float32)).to(torch.bfloat16)
    return _linear(hidden_states.reshape(SEQ, DIM), w)


_grid = _impl[CORES]

_I_HIDDEN, _I_WP_W, _I_NORM_W = 0, 8, 25


def kernel(*args):
    assert len(args) == 27, f"expected 26 recorded arguments and a replica group, got {len(args)}"
    front = _front(args[_I_HIDDEN], args[_I_WP_W], args[_I_NORM_W])
    return _grid(*args[:26], front, args[26])


kernel.lnc = getattr(_grid, "lnc", CORES)
