"""`layers.3.attention` on 4 ranks. The rank's `wo_b` partial lands in a named `nl.shared_hbm`
scratch buffer, `nki.collectives.all_reduce` sums the four partials inside the traced graph, and
the sum is copied out. A rank's identity lives entirely in the values `inference.py` passes in,
never in the code, so one NEFF runs on all four ranks.
"""

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
RATIO = 2
CSEQ = SEQ // RATIO
CACHE_ROWS = 4 * (16384 // RATIO)
INDEX_TOPK = 512
EPS = 1e-20
SM_SCALE = HEAD_DIM ** -0.5
NEG_BOUND = -1.0e30
HEADS_PER_GROUP = N_HEADS // O_GROUPS

PT = 128
NT = 512
WIN_SPAN = 2 * PT
CHUNK = 4
GATHER_UNROLL = 2

FP8_BLK = 32
FP8_MAX = 448.0
FP8_AMAX_FLOOR = 1e-4
QDQ_CHUNK = 1280

MGROUP_BYTES = 64 * 1024
WLOOK = 8
XLOOK = 2

WORLD_SIZE = 4
CC_DTYPE = nl.bfloat16
CC_ROWS = 1024

SP_SEQ = SEQ // WORLD_SIZE
FREQ_HALF = ROPE_DIM // 2
FREQ_WIDTH = 2 * FREQ_HALF
FREQ_ROWS = 16384


def _blocks(n, step):
    out = []
    off = 0
    while off < n:
        sz = min(step, n - off)
        out.append((off, sz))
        off += sz
    return out


def _live_pos_blocks(t):
    return -(-(PT * (t + 1) // RATIO) // PT)


def _live_rank_tiles(t):
    return min(INDEX_TOPK // PT, -(-(PT * (t + 1) // RATIO) // PT))


def _load_bcast(vec_hbm, n, rows, dt=nl.float32):
    wb = nl.ndarray((rows, n), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=wb, src=nl.broadcast_to(
        nl.load(vec_hbm.reshape((1, n)), dtype=dt), (rows, n)))
    return wb


def _rmsnorm(tile, wb, n, rows=PT, out_dtype=nl.float32):
    sq = nl.ndarray((rows, n), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
    ssq = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=tile, bias=0.0, scale=1.0,
                    reduce_op=nl.add, reduce_res=ssq,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    ms = nl.add(nl.multiply(ssq, 1.0 / n), EPS)
    y = nl.rsqrt(ms)
    y = nl.multiply(y, nl.subtract(1.5, nl.multiply(nl.multiply(ms, 0.5), nl.multiply(y, y))))
    out = nl.ndarray((rows, n), dtype=out_dtype, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=out, data=tile, op0=nl.multiply, operand0=y,
                              op1=nl.multiply, operand1=wb)
    return out


def _pow2_ceil_log2(x):
    bits = nl.copy(x, dtype=nl.float32).view(nl.int32)
    exp = nl.bitwise_and(nl.right_shift(bits, 23), 0xFF)
    man = nl.bitwise_and(bits, 0x7FFFFF)
    bump = nl.copy(nl.greater(man, 0), dtype=nl.int32)
    e = nl.add(exp, bump)
    scale = nl.copy(nl.left_shift(e, 23)).view(nl.float32)
    inv = nl.copy(nl.left_shift(nl.subtract(254, e), 23)).view(nl.float32)
    return scale, inv


FP8_PEAK = FP8_MAX / 2.0


BYTE_HI = 256.0
E4M3_MAG2 = 0x7F007F00
E4M3_SGN2 = -2147450880
SC_EXP_ADD = 120


def _decode_e4m3(byte_tile, rows, k):
    t = nl.ndarray((rows, k), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t, data=byte_tile, op0=nl.multiply, operand0=BYTE_HI,
                       engine=nisa.engine.scalar)
    hi = nl.ndarray((rows, k), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=hi.view(nl.uint32), data=t.view(nl.uint32),
                       op0=nl.bitwise_and, operand0=E4M3_MAG2,
                       op1=nl.right_shift, operand1=4, engine=nisa.engine.vector)
    bits = nl.ndarray((rows, k), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=bits.view(nl.uint32), data=t.view(nl.uint32),
                              op0=nl.bitwise_and, operand0=E4M3_SGN2,
                              op1=nl.bitwise_or, operand1=hi.view(nl.uint32))
    return bits.view(nl.bfloat16)


def _qdq_fp8(tile, rows, n):
    nb = n // FP8_BLK
    blocked = tile.reshape((rows, nb, FP8_BLK))
    ab = nl.ndarray((rows, nb, FP8_BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=ab, op=nl.abs, data=blocked, bias=0.0, scale=1.0)
    amax = nl.maximum(nl.copy(nl.max(ab, axis=2, keepdims=True), dtype=nl.float32),
                      FP8_AMAX_FLOOR)
    scale, rcp = _pow2_ceil_log2(nl.multiply(amax, 1.0 / FP8_PEAK))
    rcp_b = nl.copy(rcp, dtype=nl.bfloat16)
    scale_b = nl.copy(scale, dtype=nl.bfloat16)
    q8 = nl.ndarray((rows, nb, FP8_BLK), dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=q8, data1=blocked, op=nl.multiply,
                       data2=nl.broadcast_to(rcp_b, (rows, nb, FP8_BLK)))
    deq = nl.ndarray((rows, nb, FP8_BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=deq, data1=q8, op=nl.multiply,
                       data2=nl.broadcast_to(scale_b, (rows, nb, FP8_BLK)))
    return deq.reshape((rows, n))


def _act_chunk(k):
    for c in (QDQ_CHUNK, 1024, 512, 256, 128):
        if k % c == 0:
            return c
    return k


def _dequant_weight(w_u8, s_u8, dst_hbm, n, k, dst_row=0):
    kb = k // FP8_BLK
    rows = PT if n % PT == 0 else FP8_BLK
    nsb = rows // FP8_BLK
    for nb in nl.affine_range(n // rows):
        wq = _decode_e4m3(nl.load(w_u8[nb * rows:(nb + 1) * rows, 0:k], dtype=nl.uint8),
                          rows, k)
        sf = nl.ndarray((rows, kb), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i in nl.static_range(nsb):
            sb = nl.load(s_u8[nb * nsb + i:nb * nsb + i + 1, 0:kb], dtype=nl.int32)
            one = nl.copy(nl.left_shift(nl.add(sb, SC_EXP_ADD), 23)).view(nl.float32)
            nisa.tensor_copy(dst=sf[i * FP8_BLK:(i + 1) * FP8_BLK, 0:kb],
                             src=nl.broadcast_to(one, (FP8_BLK, kb)))
        rep = nl.broadcast_to(sf.reshape((rows, kb, 1)), (rows, kb, FP8_BLK))
        prod = nl.ndarray((rows, k), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prod, data1=wq, data2=rep.reshape((rows, k)),
                           op=nl.multiply)
        nl.store(dst_hbm[dst_row + nb * rows:dst_row + (nb + 1) * rows, 0:k],
                 value=prod)


def _kv_rope_quant(src_hbm, w_hbm, freqs3, dst_hbm, m, src_col=0, row=0, dst_row=0):
    wb = _load_bcast(w_hbm, HEAD_DIM, PT, dt=nl.bfloat16)
    half = ROPE_DIM // 2
    lo = HEAD_DIM - ROPE_DIM
    lo2 = lo // 2
    for mb in nl.affine_range(m // PT):
        r0 = row + mb * PT
        t = nl.load(src_hbm[r0:r0 + PT, src_col:src_col + HEAD_DIM], dtype=nl.bfloat16)
        y = _rmsnorm(t, wb, HEAD_DIM, out_dtype=nl.bfloat16)
        cs = nl.load(freqs3[r0:r0 + PT, 0:half, 0:2], dtype=nl.float32)
        cos = nl.copy(cs[0:PT, 0:half, 0])
        sin = nl.copy(cs[0:PT, 0:half, 1])
        y2 = y.reshape((PT, HEAD_DIM // 2, 2))
        a = y2[0:PT, lo2:HEAD_DIM // 2, 0]
        b = y2[0:PT, lo2:HEAD_DIM // 2, 1]

        def _mulf(x, z):
            r = nl.ndarray((PT, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=r, data1=x, data2=z, op=nl.multiply)
            return r

        ac = _mulf(a, cos)
        bs = _mulf(b, sin)
        as_ = _mulf(a, sin)
        bc = _mulf(b, cos)
        o = nl.ndarray((PT, half, 2), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o[0:PT, 0:half, 0], src=nl.subtract(ac, bs))
        nisa.tensor_copy(dst=o[0:PT, 0:half, 1], src=nl.add(as_, bc))
        nisa.tensor_copy(dst=y[0:PT, lo:HEAD_DIM], src=o.reshape((PT, ROPE_DIM)))
        nl.store(dst_hbm[dst_row + mb * PT:dst_row + (mb + 1) * PT, 0:HEAD_DIM],
                 value=_qdq_fp8(y, PT, HEAD_DIM))


def _mgroup(mtiles, ktiles):
    mg = 1
    while mg * 2 <= mtiles and (mg * 2) * ktiles * PT * 2 <= MGROUP_BYTES:
        mg *= 2
    return mg


def _gemm(x_hbm, w_hbm, dst_hbm, m, k, n, out_dtype=nl.bfloat16,
          x_row=0, x_col=0, w_row=0, dst_col=0, dst_row=0, qdq=False):
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
    ntiles = n // nt
    for mgb in nl.affine_range((m // PT) // mg):
        xt = nl.ndarray((PT, mg, ktiles, PT), dtype=nl.bfloat16, buffer=nl.sbuf)

        def stage_x(mi):
            for kb in nl.static_range(ktiles):
                nisa.dma_transpose(
                    dst=xt[0:PT, mi, kb, 0:PT],
                    src=x_hbm[x_row + (mgb * mg + mi) * PT:x_row + (mgb * mg + mi + 1) * PT,
                              x_col + kb * PT:x_col + (kb + 1) * PT])

        def ntile(nb, jit):
            wt = nl.ndarray((PT, ktiles, nt), dtype=nl.bfloat16, buffer=nl.sbuf)

            def stage_w(kb):
                nisa.dma_transpose(
                    dst=wt[0:PT, kb, 0:nt],
                    src=w_hbm[w_row + nb * nt:w_row + (nb + 1) * nt,
                              kb * PT:(kb + 1) * PT])

            for kb in nl.static_range(min(WLOOK, ktiles)):
                stage_w(kb)
            for mi in nl.static_range(mg):
                if jit and mi + XLOOK < mg:
                    stage_x(mi + XLOOK)
                acc = nl.ndarray((PT, nt), dtype=nl.float32, buffer=nl.psum)
                for kb in nl.static_range(ktiles):
                    if mi == 0 and kb + WLOOK < ktiles:
                        stage_w(kb + WLOOK)
                    nisa.nc_matmul(dst=acc, stationary=xt[0:PT, mi, kb, 0:PT],
                                   moving=wt[0:PT, kb, 0:nt], accumulate=(kb > 0))
                row = nl.copy(acc, dtype=out_dtype)
                nl.store(dst_hbm[dst_row + (mgb * mg + mi) * PT:
                                 dst_row + (mgb * mg + mi + 1) * PT,
                                 dst_col + nb * nt:dst_col + (nb + 1) * nt],
                         value=_qdq_fp8(row, PT, nt) if qdq else row)

        for mi0 in nl.static_range(min(XLOOK, mg)):
            stage_x(mi0)
        ntile(0, mg > XLOOK)
        if ntiles > 1:
            for nb in nl.affine_range(ntiles - 1):
                ntile(nb + 1, False)


def _gemm_head_major(x_hbm, w_hbm, dst_hbm, m, k, n, hdim, row=0, hstride=None,
                     rope_freqs=None):
    ktiles = k // PT
    hst = m if hstride is None else hstride
    mg = _mgroup(m // PT, ktiles)
    ntiles = n // hdim
    half = ROPE_DIM // 2
    lo = hdim - ROPE_DIM
    if rope_freqs is not None:
        mt = m // PT
        cosb = nl.ndarray((PT, mt, half), dtype=nl.float32, buffer=nl.sbuf)
        sinb = nl.ndarray((PT, mt, half), dtype=nl.float32, buffer=nl.sbuf)
        for mi in nl.affine_range(mt):
            cs = nl.load(rope_freqs[row + mi * PT:row + (mi + 1) * PT, 0:half, 0:2],
                         dtype=nl.float32)
            nisa.tensor_copy(dst=cosb[0:PT, mi, 0:half], src=cs[0:PT, 0:half, 0])
            nisa.tensor_copy(dst=sinb[0:PT, mi, 0:half], src=cs[0:PT, 0:half, 1])

    def drain(acc, mii):
        rw = nl.copy(acc, dtype=nl.bfloat16)
        if rope_freqs is None:
            return rw
        r4 = rw.reshape((PT, hdim // 2, 2))
        a = r4[0:PT, lo // 2:hdim // 2, 0]
        b = r4[0:PT, lo // 2:hdim // 2, 1]

        def _mulf(x, z):
            r = nl.ndarray((PT, half), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=r, data1=x, data2=z, op=nl.multiply)
            return r

        ac = _mulf(a, cosb[0:PT, mii, 0:half])
        bs = _mulf(b, sinb[0:PT, mii, 0:half])
        as_ = _mulf(a, sinb[0:PT, mii, 0:half])
        bc = _mulf(b, cosb[0:PT, mii, 0:half])
        o = nl.ndarray((PT, half, 2), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o[0:PT, 0:half, 0], src=nl.subtract(ac, bs))
        nisa.tensor_copy(dst=o[0:PT, 0:half, 1], src=nl.add(as_, bc))
        nisa.tensor_copy(dst=rw[0:PT, lo:hdim], src=o.reshape((PT, ROPE_DIM)))
        return rw

    for mgb in nl.affine_range((m // PT) // mg):
        xt = nl.ndarray((PT, mg, ktiles, PT), dtype=nl.bfloat16, buffer=nl.sbuf)

        def stage_x(mi):
            for kb in nl.static_range(ktiles):
                nisa.dma_transpose(
                    dst=xt[0:PT, mi, kb, 0:PT],
                    src=x_hbm[row + (mgb * mg + mi) * PT:row + (mgb * mg + mi + 1) * PT,
                              kb * PT:(kb + 1) * PT])

        def ntile(nb, jit):
            wt = nl.ndarray((PT, ktiles, hdim), dtype=nl.bfloat16, buffer=nl.sbuf)

            def stage_w(kb):
                nisa.dma_transpose(
                    dst=wt[0:PT, kb, 0:hdim],
                    src=w_hbm[nb * hdim:(nb + 1) * hdim, kb * PT:(kb + 1) * PT])

            for kb in nl.static_range(min(WLOOK, ktiles)):
                stage_w(kb)
            for mi in nl.static_range(mg):
                if jit and mi + XLOOK < mg:
                    stage_x(mi + XLOOK)
                acc = nl.ndarray((PT, hdim), dtype=nl.float32, buffer=nl.psum)
                for kb in nl.static_range(ktiles):
                    if mi == 0 and kb + WLOOK < ktiles:
                        stage_w(kb + WLOOK)
                    nisa.nc_matmul(dst=acc, stationary=xt[0:PT, mi, kb, 0:PT],
                                   moving=wt[0:PT, kb, 0:hdim], accumulate=(kb > 0))
                r = nb * hst + row + (mgb * mg + mi) * PT
                nl.store(dst_hbm[r:r + PT, 0:hdim], value=drain(acc, mgb * mg + mi))

        for mi0 in nl.static_range(min(XLOOK, mg)):
            stage_x(mi0)
        ntile(0, mg > XLOOK)
        if ntiles > 1:
            for nb in nl.affine_range(ntiles - 1):
                ntile(nb + 1, False)


def _norm_quant_rows(src_hbm, w_hbm, dst_hbm, m, n, src_col=0, row=0):
    wb = _load_bcast(w_hbm, n, PT, dt=nl.bfloat16)
    chunk = _act_chunk(n)
    for mb in nl.affine_range(m // PT):
        t = nl.load(src_hbm[row + mb * PT:row + (mb + 1) * PT, src_col:src_col + n],
                    dtype=nl.bfloat16)
        y = _rmsnorm(t, wb, n, out_dtype=nl.bfloat16)
        for cb in nl.static_range(n // chunk):
            nl.store(dst_hbm[row + mb * PT:row + (mb + 1) * PT,
                             cb * chunk:(cb + 1) * chunk],
                     value=_qdq_fp8(y[0:PT, cb * chunk:(cb + 1) * chunk], PT, chunk))


QT = 4
QG = QT * PT
GQ = 8
CPAIR = 1


def _sparse_attention_impl(q_hbm, kvp_hbm, ckv_hbm, topk_hbm, sink_hbm, og3_hbm, gbs,
                           ogf_hbm):
    dt = HEAD_DIM // PT
    nrt = INDEX_TOPK // PT
    ncb = CSEQ // PT
    WB = QT + 1
    ncs = [_live_pos_blocks((gb + 1) * QT - 1) for gb in gbs]
    ncmax = max(ncs)
    nrtmax = max(_live_rank_tiles((gb + 1) * QT - 1) for gb in gbs)

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
    eye = nl.copy(nl.equal(dio[0:PT, 0:PT], 0.0), dtype=nl.bfloat16)
    onec = nl.copy(nl.greater_equal(jio[0:PT, 0:1], 0.0), dtype=nl.bfloat16)
    zeroq = nl.copy(nl.less(jio[0:PT, 0:QT], 0.0), dtype=nl.bfloat16)
    zblk = nl.zeros((PT, GQ * ncmax), dtype=nl.bfloat16, buffer=nl.sbuf)
    sinkb = _load_bcast(sink_hbm, N_HEADS, PT)
    esinks = nl.ndarray((PT, N_HEADS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=esinks, op=nl.exp, data=sinkb[0:PT, 0:N_HEADS], bias=0.0, scale=1.0)

    pio2_i = nl.ndarray((PT, PT), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=pio2_i, pattern=[[1, PT]], offset=0, channel_multiplier=0)
    pio2 = nl.copy(pio2_i, dtype=nl.bfloat16)
    pio2g = nl.ndarray((PT, GQ, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pio2g, src=pio2.ap(pattern=[[PT, PT], [0, GQ], [1, PT]]))
    cio2_i = nl.ndarray((PT, ncmax), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=cio2_i, pattern=[[1, ncmax]], offset=0, channel_multiplier=0)
    cio2 = nl.copy(cio2_i, dtype=nl.bfloat16)

    ckv_t = nl.ndarray((PT, dt, ncmax * PT), dtype=nl.bfloat16, buffer=nl.sbuf)
    ckv_n = nl.ndarray((PT, ncmax, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    for gp in nl.affine_range(ncmax):
        nisa.tensor_copy(dst=ckv_n[0:PT, gp, 0:HEAD_DIM],
                         src=nl.load(ckv_hbm[gp * PT:(gp + 1) * PT, 0:HEAD_DIM]))
        for dc in nl.static_range(dt):
            nisa.dma_transpose(dst=ckv_t[0:PT, dc, gp * PT:(gp + 1) * PT],
                               src=ckv_hbm[gp * PT:(gp + 1) * PT, dc * PT:(dc + 1) * PT])

    wadds = []
    for first in range(2):
        live = nl.ndarray((PT, WIN_SPAN), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=live, data=nonneg, op0=nl.maximum,
                           operand0=float(1 - first))
        wadds.append(nl.multiply(nl.subtract(nl.multiply(live, inwin), 1.0), -NEG_BOUND))

    win_t = nl.ndarray((PT, dt, WB * PT), dtype=nl.bfloat16, buffer=nl.sbuf)
    win_n = nl.ndarray((PT, WB, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    loT = nl.ndarray((PT, nrtmax, QG), dtype=nl.bfloat16, buffer=nl.sbuf)
    hiT = nl.ndarray((PT, nrtmax, QG), dtype=nl.bfloat16, buffer=nl.sbuf)
    maskTs = [nl.ndarray((PT, ncmax, QG), dtype=nl.bfloat16, buffer=nl.sbuf)
              for _ in range(2)]
    qt4 = nl.ndarray((PT, dt, QG), dtype=nl.bfloat16, buffer=nl.sbuf)
    PDEPTH = 4
    pexps = [nl.ndarray((PT, CPAIR, QG), dtype=nl.bfloat16, buffer=nl.sbuf)
             for _ in range(PDEPTH)]
    pblks = [nl.ndarray((PT, CPAIR, QG), dtype=nl.bfloat16, buffer=nl.sbuf)
             for _ in range(PDEPTH)]
    pacs = [nl.ndarray((PT, CPAIR, QG), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(2)]
    pwT = nl.ndarray((PT, 2, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
    loTb = [nl.ndarray((PT, GQ, PT), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(2)]
    palls = [nl.ndarray((PT, nrtmax, GQ, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
             for _ in range(2)]
    calls = [nl.ndarray((PT, nrtmax, GQ, ncmax), dtype=nl.bfloat16, buffer=nl.sbuf)
             for _ in range(2)]

    RHALF = ROPE_DIM // 2
    OGLO = HEAD_DIM - ROPE_DIM
    ogcos = nl.ndarray((PT, QT, RHALF), dtype=nl.float32, buffer=nl.sbuf)
    ogsin = nl.ndarray((PT, QT, RHALF), dtype=nl.float32, buffer=nl.sbuf)

    def rope_prep(gb):
        qg0 = gb * QG
        for ti in nl.static_range(QT):
            cs = nl.load(ogf_hbm[qg0 + ti * PT:qg0 + (ti + 1) * PT, 0:RHALF, 0:2],
                         dtype=nl.float32)
            nisa.tensor_copy(dst=ogcos[0:PT, ti, 0:RHALF], src=cs[0:PT, 0:RHALF, 0])
            nisa.tensor_copy(dst=ogsin[0:PT, ti, 0:RHALF], src=cs[0:PT, 0:RHALF, 1])

    def win_prep(gb):
        qg0 = gb * QG
        for wb in nl.static_range(WB):
            r0 = qg0 + (wb - 1) * PT
            if r0 < 0:
                nisa.tensor_copy(dst=win_n[0:PT, wb, 0:HEAD_DIM],
                                 src=nl.zeros((PT, HEAD_DIM), dtype=nl.bfloat16,
                                              buffer=nl.sbuf))
                for dc in nl.static_range(dt):
                    nisa.tensor_copy(dst=win_t[0:PT, dc, wb * PT:(wb + 1) * PT],
                                     src=nl.zeros((PT, PT), dtype=nl.bfloat16,
                                                  buffer=nl.sbuf))
                continue
            nisa.tensor_copy(dst=win_n[0:PT, wb, 0:HEAD_DIM],
                             src=nl.load(kvp_hbm[r0:r0 + PT, 0:HEAD_DIM]))
            for dc in nl.static_range(dt):
                nisa.dma_transpose(
                    dst=win_t[0:PT, dc, wb * PT:(wb + 1) * PT],
                    src=kvp_hbm[r0:r0 + PT, dc * PT:(dc + 1) * PT])

    def key_prep(gb, nrt_t):
        qg0 = gb * QG
        kw = nrt_t * PT
        for ti in nl.static_range(QT):
            q0 = qg0 + ti * PT
            tki = nl.load(topk_hbm[q0:q0 + PT, 0:kw], dtype=nl.int32)
            cand = nl.ndarray((PT, kw), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=cand, data=tki, op0=nl.subtract, operand0=SEQ,
                               op1=nl.maximum, operand1=0)
            base = nl.ndarray((PT, kw), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=base, data=tki, op0=nl.less, operand0=SEQ,
                               op1=nl.multiply, operand1=CSEQ)
            rel_i = nl.ndarray((PT, kw), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=rel_i, data1=cand, data2=base, op=nl.add)
            hi_b = nl.copy(nl.right_shift(rel_i, 7), dtype=nl.bfloat16)
            lo_b = nl.copy(nl.bitwise_and(rel_i, PT - 1), dtype=nl.bfloat16)
            for rt in nl.static_range(nrt_t):
                tp = nl.ndarray((PT, 2, PT), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=tp[0:PT, 0, 0:PT],
                               stationary=hi_b[0:PT, rt * PT:(rt + 1) * PT],
                               moving=eye, accumulate=False)
                nisa.nc_matmul(dst=tp[0:PT, 1, 0:PT],
                               stationary=lo_b[0:PT, rt * PT:(rt + 1) * PT],
                               moving=eye, accumulate=False)
                nisa.tensor_copy(dst=hiT[0:PT, rt, ti * PT:(ti + 1) * PT],
                                 src=tp[0:PT, 0, 0:PT])
                nisa.tensor_copy(dst=loT[0:PT, rt, ti * PT:(ti + 1) * PT],
                                 src=tp[0:PT, 1, 0:PT])

    def mask_compare(nrt_t, nc, qblk, slot):
        qb = qblk * GQ
        for rt in nl.static_range(nrt_t):
            nisa.tensor_copy(
                dst=loTb[rt % 2],
                src=loT.ap(pattern=[[nrtmax * QG, PT], [1, GQ], [0, PT]],
                           offset=rt * QG + qb),
                engine=nisa.scalar_engine)
            nisa.tensor_tensor(
                dst=palls[slot][0:PT, rt, 0:GQ, 0:PT], op=nl.equal,
                data1=pio2g[0:PT, 0:GQ, 0:PT],
                data2=loTb[rt % 2][0:PT, 0:GQ, 0:PT])
            nisa.tensor_tensor(
                dst=calls[slot][0:PT, rt, 0:GQ, 0:nc], op=nl.equal,
                data1=cio2.ap(pattern=[[ncmax, PT], [0, GQ], [1, nc]]),
                data2=hiT.ap(pattern=[[nrtmax * QG, PT], [1, GQ], [0, nc]],
                             offset=rt * QG + qb))

    def mask_scatter(nrt_t, nc, qblk, slot, mdst):
        qb = qblk * GQ
        mp = nl.ndarray((PT, GQ * nc), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=mp, stationary=eye, moving=zblk[0:PT, 0:GQ * nc],
                       accumulate=False)
        for g in nl.static_range(GQ):
            for rt in nl.static_range(nrt_t):
                nisa.nc_matmul(dst=mp[0:PT, g * nc:(g + 1) * nc],
                               stationary=palls[slot][0:PT, rt, g, 0:PT],
                               moving=calls[slot][0:PT, rt, g, 0:nc], accumulate=True)
        nisa.tensor_copy(dst=mdst[0:PT, 0:nc, qb:qb + GQ],
                         src=mp.ap(pattern=[[GQ * nc, PT], [1, nc], [nc, GQ]]),
                         engine=nisa.engine.vector)

    def mask_slice(nrt_t, nc, qblk, mdst):
        mask_compare(nrt_t, nc, qblk, 0)
        mask_scatter(nrt_t, nc, qblk, 0, mdst)

    def head_group(gb, h, nc, wadd0, maskT, inject=None):
        qg0 = gb * QG
        for ti in nl.static_range(QT):
            q0 = qg0 + ti * PT
            for dc in nl.static_range(dt):
                nisa.dma_transpose(
                    dst=qt4[0:PT, dc, ti * PT:(ti + 1) * PT],
                    src=q_hbm[h * SEQ + q0:h * SEQ + q0 + PT, dc * PT:(dc + 1) * PT])
        dws = nl.ndarray((PT, QT), dtype=nl.float32, buffer=nl.sbuf)
        accs = []
        dacc = []
        for ti in nl.static_range(QT):
            wadd = wadds[1] if (gb == 0 and ti == 0) else wadd0
            lwp = nl.ndarray((PT, WIN_SPAN), dtype=nl.float32, buffer=nl.psum)
            for dc in nl.static_range(dt):
                nisa.nc_matmul(dst=lwp, stationary=qt4[0:PT, dc, ti * PT:(ti + 1) * PT],
                               moving=win_t[0:PT, dc, ti * PT:(ti + 2) * PT],
                               accumulate=(dc > 0))
            lw = nl.ndarray((PT, WIN_SPAN), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=lw, data=lwp, op0=nl.multiply,
                                      operand0=SM_SCALE, op1=nl.add, operand1=wadd)
            pw = nl.ndarray((PT, WIN_SPAN), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=pw, op=nl.exp, data=lw, bias=0.0, scale=1.0,
                            reduce_op=nl.add, reduce_res=dws[0:PT, ti:ti + 1],
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)
            twp = nl.ndarray((PT, 2, PT), dtype=nl.float32, buffer=nl.psum)
            for wt in nl.static_range(2):
                nisa.nc_matmul(dst=twp[0:PT, wt, 0:PT],
                               stationary=pw[0:PT, wt * PT:(wt + 1) * PT],
                               moving=eye, accumulate=False)
            nisa.tensor_copy(dst=pwT, src=twp, engine=nisa.scalar_engine)
            acc = nl.ndarray((PT, HEAD_DIM), dtype=nl.float32, buffer=nl.psum)
            for wt in nl.static_range(2):
                nisa.nc_matmul(dst=acc, stationary=pwT[0:PT, wt, 0:PT],
                               moving=win_n[0:PT, ti + wt, 0:HEAD_DIM],
                               accumulate=(wt > 0))
            accs.append(acc)

        for cb in nl.static_range(nc // CPAIR):
            c0 = cb * CPAIR
            sl = cb % PDEPTH
            pexp = pexps[sl]
            pblk = pblks[sl]
            lcp = nl.ndarray((PT, CPAIR, QG), dtype=nl.float32, buffer=nl.psum)
            for cj in nl.static_range(CPAIR):
                for dc in nl.static_range(dt):
                    nisa.nc_matmul(dst=lcp[0:PT, cj, 0:QG],
                                   stationary=ckv_t[0:PT, dc, (c0 + cj) * PT:(c0 + cj + 1) * PT],
                                   moving=qt4[0:PT, dc, 0:QG], accumulate=(dc > 0))
            nisa.activation(dst=pexp, op=nl.exp, data=lcp, bias=0.0, scale=SM_SCALE)
            nisa.tensor_tensor(dst=pblk, data1=pexp,
                               data2=maskT[0:PT, c0:c0 + CPAIR, 0:QG], op=nl.multiply)
            if cb == 0:
                nisa.tensor_copy(dst=pacs[1], src=pblk, engine=nisa.engine.vector)
            else:
                nisa.tensor_tensor(dst=pacs[(cb + 1) % 2], data1=pacs[cb % 2], data2=pblk,
                                   op=nl.add, engine=nisa.engine.vector)
            for cj in nl.static_range(CPAIR):
                for ti in nl.static_range(QT):
                    nisa.nc_matmul(dst=accs[ti],
                                   stationary=pblk[0:PT, cj, ti * PT:(ti + 1) * PT],
                                   moving=ckv_n[0:PT, c0 + cj, 0:HEAD_DIM], accumulate=True)
                if inject is not None:
                    inject(c0 + cj)
        dcol = nl.ndarray((PT, QT), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=dcol[0:PT, 0:QT], stationary=eye, moving=zeroq,
                       accumulate=False)
        pfin = pacs[(nc // CPAIR) % 2]
        for cj in nl.static_range(CPAIR):
            for ti in nl.static_range(QT):
                nisa.nc_matmul(dst=dcol[0:PT, ti:ti + 1],
                               stationary=pfin[0:PT, cj, ti * PT:(ti + 1) * PT],
                               moving=onec, accumulate=True)
        o4 = nl.ndarray((PT, QT, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        dsum = nl.ndarray((PT, QT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=dsum, data=dcol[0:PT, 0:QT], op0=nl.add,
                                  operand0=esinks[0:PT, h:h + 1],
                                  op1=nl.add, operand1=dws[0:PT, 0:QT])
        rq = nl.ndarray((PT, QT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rq, data=dsum)
        for ti in nl.static_range(QT):
            nisa.tensor_scalar(dst=o4[0:PT, ti, 0:HEAD_DIM], data=accs[ti],
                               op0=nl.multiply, operand0=rq[0:PT, ti:ti + 1])
        o5 = o4.reshape((PT, QT, HEAD_DIM // 2, 2))
        oa = o5[0:PT, 0:QT, OGLO // 2:HEAD_DIM // 2, 0]
        ob = o5[0:PT, 0:QT, OGLO // 2:HEAD_DIM // 2, 1]

        def _ogmul(x, z):
            r = nl.ndarray((PT, QT, RHALF), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=r, data1=x, data2=z, op=nl.multiply)
            return r

        oac = _ogmul(oa, ogcos[0:PT, 0:QT, 0:RHALF])
        obs = _ogmul(ob, ogsin[0:PT, 0:QT, 0:RHALF])
        oas = _ogmul(oa, ogsin[0:PT, 0:QT, 0:RHALF])
        obc = _ogmul(ob, ogcos[0:PT, 0:QT, 0:RHALF])
        orot = nl.ndarray((PT, QT, RHALF, 2), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=orot[0:PT, 0:QT, 0:RHALF, 0], data1=oac, data2=obs,
                           op=nl.add)
        nisa.tensor_tensor(dst=orot[0:PT, 0:QT, 0:RHALF, 1], data1=obc, data2=oas,
                           op=nl.subtract)
        nisa.tensor_copy(dst=o4[0:PT, 0:QT, OGLO:HEAD_DIM],
                         src=orot.reshape((PT, QT, ROPE_DIM)))
        for ti in nl.static_range(QT):
            nl.store(og3_hbm[qg0 + ti * PT:qg0 + (ti + 1) * PT, h:h + 1, 0:HEAD_DIM],
                     value=o4[0:PT, ti, 0:HEAD_DIM])

    NQB = QG // GQ
    QPH = NQB // N_HEADS
    ncof = {gb: _live_pos_blocks((gb + 1) * QT - 1) for gb in gbs}
    nrtof = {gb: _live_rank_tiles((gb + 1) * QT - 1) for gb in gbs}

    key_prep(gbs[0], nrtof[gbs[0]])
    for s0 in nl.static_range(NQB):
        mask_slice(nrtof[gbs[0]], ncof[gbs[0]], s0, maskTs[0])

    for i, gb in enumerate(gbs):
        win_prep(gb)
        rope_prep(gb)
        nxt = gbs[i + 1] if i + 1 < len(gbs) else None
        if nxt is not None:
            key_prep(nxt, nrtof[nxt])
        ncur = ncof[gb]
        for h in nl.affine_range(N_HEADS):
            if nxt is None:
                head_group(gb, h, ncur, wadds[0], maskTs[i % 2])
                continue

            def inject(c, _nrt=nrtof[nxt], _nc=ncof[nxt], _m=maskTs[(i + 1) % 2],
                       _h=h, _ncur=ncur):
                for k in range(QPH + 1):
                    if (k * _ncur) // (QPH + 1) != c:
                        continue
                    if k < QPH:
                        mask_compare(_nrt, _nc, _h * QPH + k, k % 2)
                    if k >= 1:
                        mask_scatter(_nrt, _nc, _h * QPH + k - 1, (k - 1) % 2, _m)

            head_group(gb, h, ncur, wadds[0], maskTs[i % 2], inject=inject)


def _sparse_attention(q_hbm, kvp_hbm, ckv_hbm, topk_hbm, sink_hbm, og3_hbm, gbs, ogf_hbm):
    return _sparse_attention_impl(q_hbm, kvp_hbm, ckv_hbm, topk_hbm, sink_hbm, og3_hbm, gbs,
                                  ogf_hbm)


CORES = 2


def _core_runs(pid, cores=CORES):
    ng = SEQ // QG
    per = ng // (2 * cores)
    a = (pid * per, pid * per + per)
    b = (ng - (pid + 1) * per, ng - pid * per)
    runs = [(a[0], b[1])] if a[1] == b[0] else [a, b]
    return [(g0 * QG, (g1 - g0) * QG) for g0, g1 in runs], \
           [list(range(g0, g1)) for g0, g1 in runs]


@nki.jit
def kernel(hidden_states, state_shared_attn_candidates, state_shared_attn_compress_kv, state_shared_attn_index_k, state_shared_attn_topk_idxs, layers_3_attn_attn_sink, layers_3_attn_freqs_cis, layers_3_attn_kv_norm_weight, layers_3_attn_q_norm_weight, layers_3_attn_window_kv_cache, layers_3_attn_wkv_scale, layers_3_attn_wkv_weight, layers_3_attn_wo_a_scale, layers_3_attn_wo_a_weight, layers_3_attn_wo_b_scale, layers_3_attn_wo_b_weight, layers_3_attn_wq_a_scale, layers_3_attn_wq_a_weight, layers_3_attn_wq_b_scale, layers_3_attn_wq_b_weight, layers_3_attn_norm_weight, replica_group):
    xsp_hbm = hidden_states.reshape((WORLD_SIZE, SP_SEQ, DIM))
    fsp_hbm = layers_3_attn_freqs_cis.reshape(
        (FREQ_ROWS // SP_SEQ, SP_SEQ, FREQ_WIDTH))
    ckv_hbm = state_shared_attn_compress_kv.reshape((CACHE_ROWS, HEAD_DIM))
    topk_hbm = state_shared_attn_topk_idxs.reshape((SEQ, INDEX_TOPK))
    out = nl.ndarray((1, SEQ, DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    out2 = out.reshape((SEQ, DIM))
    cc_src = nl.ndarray((SEQ, DIM), dtype=CC_DTYPE, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((SEQ, DIM), dtype=CC_DTYPE, buffer=nl.shared_hbm, name="cc_dst")

    pid = nl.program_id(0)
    runs, run_gbs = _core_runs(pid)

    QKV_N = Q_LORA + HEAD_DIM
    wqkv = nl.ndarray((QKV_N, DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="wqkv")
    wqb = nl.ndarray((N_HEADS * HEAD_DIM, Q_LORA), dtype=nl.bfloat16,
                     buffer=nl.shared_hbm, name="wqb")
    wo_a = nl.ndarray((O_GROUPS * O_LORA, HEADS_PER_GROUP * HEAD_DIM), dtype=nl.bfloat16,
                      buffer=nl.shared_hbm, name="wo_a")
    wo_b = nl.ndarray((DIM, O_GROUPS * O_LORA), dtype=nl.bfloat16,
                      buffer=nl.shared_hbm, name="wo_b")
    if pid == 0:
        _dequant_weight(layers_3_attn_wq_a_weight, layers_3_attn_wq_a_scale, wqkv,
                        Q_LORA, DIM)
        _dequant_weight(layers_3_attn_wkv_weight, layers_3_attn_wkv_scale, wqkv,
                        HEAD_DIM, DIM, dst_row=Q_LORA)
        _dequant_weight(layers_3_attn_wo_b_weight, layers_3_attn_wo_b_scale, wo_b,
                        DIM, O_GROUPS * O_LORA)
    else:
        _dequant_weight(layers_3_attn_wq_b_weight, layers_3_attn_wq_b_scale, wqb,
                        N_HEADS * HEAD_DIM, Q_LORA)
        _dequant_weight(layers_3_attn_wo_a_weight, layers_3_attn_wo_a_scale, wo_a,
                        O_GROUPS * O_LORA, HEADS_PER_GROUP * HEAD_DIM)
    for wbuf in (wqkv, wqb, wo_a, wo_b):
        nisa.core_barrier(data=wbuf, cores=tuple(range(CORES)))

    SP_PER_CORE = SP_SEQ // CORES
    sp0 = pid * SP_PER_CORE
    xloc = nl.ndarray((SP_SEQ, DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    floc = nl.ndarray((SP_SEQ, FREQ_WIDTH), dtype=nl.float32, buffer=nl.private_hbm)
    xn_q = nl.ndarray((SP_SEQ, DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    qakv = nl.ndarray((SP_SEQ, QKV_N), dtype=nl.bfloat16, buffer=nl.private_hbm)
    qr_loc = nl.ndarray((SP_SEQ, Q_LORA), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                        name="qr_loc")
    kv_loc = nl.ndarray((SP_SEQ, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                        name="kv_loc")
    qr_q = nl.ndarray((SEQ, Q_LORA), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="qr_all")
    kvg = nl.ndarray((SEQ, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kv_all")
    q = nl.ndarray((N_HEADS * SEQ, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    og3 = nl.ndarray((SEQ, N_HEADS, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    og = og3.reshape((SEQ, N_HEADS * HEAD_DIM))
    z_q = nl.ndarray((SEQ, O_GROUPS * O_LORA), dtype=nl.bfloat16, buffer=nl.private_hbm)

    rid = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.register_store(rid, ncc.rank_id())
    for off, sz in _blocks(SP_PER_CORE, CC_ROWS):
        nisa.dma_copy(dst=xloc[sp0 + off:sp0 + off + sz, 0:DIM],
                      src=xsp_hbm.ap(pattern=[[DIM, sz], [1, DIM]],
                                     offset=(sp0 + off) * DIM,
                                     scalar_offset=rid, indirect_dim=0))
    nisa.dma_copy(dst=floc[sp0:sp0 + SP_PER_CORE, 0:FREQ_WIDTH],
                  src=fsp_hbm.ap(pattern=[[FREQ_WIDTH, SP_PER_CORE], [1, FREQ_WIDTH]],
                                 offset=sp0 * FREQ_WIDTH,
                                 scalar_offset=rid, indirect_dim=0))
    floc3 = floc.reshape((SP_SEQ, FREQ_HALF, 2))

    _norm_quant_rows(xloc, layers_3_attn_norm_weight, xn_q, SP_PER_CORE, DIM, row=sp0)
    _gemm(xn_q, wqkv, qakv, SP_PER_CORE, DIM, QKV_N, x_row=sp0, dst_row=sp0)
    _kv_rope_quant(qakv, layers_3_attn_kv_norm_weight, floc3, kv_loc, SP_PER_CORE,
                   src_col=Q_LORA, row=sp0, dst_row=sp0)
    _norm_quant_rows(qakv, layers_3_attn_q_norm_weight, qr_loc, SP_PER_CORE, Q_LORA, row=sp0)

    nisa.core_barrier(data=qr_loc, cores=tuple(range(CORES)))
    nisa.core_barrier(data=kv_loc, cores=tuple(range(CORES)))
    ncc.all_gather(srcs=[qr_loc, kv_loc], dsts=[qr_q, kvg],
                   replica_group=replica_group, collective_dim=0)
    nisa.core_barrier(data=qr_q, cores=tuple(range(CORES)))
    nisa.core_barrier(data=kvg, cores=tuple(range(CORES)))

    for r0, m in runs:
        _gemm_head_major(qr_q, wqb, q, m, Q_LORA, N_HEADS * HEAD_DIM, HEAD_DIM,
                         row=r0, hstride=SEQ, rope_freqs=layers_3_attn_freqs_cis)

    allgbs = sorted([gb for gl in run_gbs for gb in gl], reverse=True)
    _sparse_attention(q, kvg, ckv_hbm, topk_hbm, layers_3_attn_attn_sink, og3, allgbs,
                      layers_3_attn_freqs_cis)

    for r0, m in runs:
        for g in range(O_GROUPS):
            _gemm(og, wo_a, z_q, m, HEADS_PER_GROUP * HEAD_DIM, O_LORA,
                  x_row=r0, x_col=g * HEADS_PER_GROUP * HEAD_DIM, w_row=g * O_LORA,
                  dst_row=r0, dst_col=g * O_LORA, qdq=True)
        _gemm(z_q, wo_b, cc_src, m, O_GROUPS * O_LORA, DIM, x_row=r0, dst_row=r0,
              out_dtype=CC_DTYPE)

    nisa.core_barrier(data=cc_src, cores=tuple(range(CORES)))
    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], replica_group=replica_group, op=nl.add)
    for r0, m in runs:
        for off, sz in _blocks(m, CC_ROWS):
            nisa.dma_copy(dst=out2[r0 + off:r0 + off + sz, :],
                          src=cc_dst[r0 + off:r0 + off + sz, :])
    nisa.core_barrier(data=out, cores=tuple(range(CORES)))
    return out
