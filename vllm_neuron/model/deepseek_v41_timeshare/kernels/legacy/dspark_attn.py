"""`mtp.0.attention` on 4 ranks, cut along heads (`FLOORPLAN.md`: `head x4 (allreduce)`). Each rank
computes a quarter of the heads, and `nki.collectives.all_reduce` sums the four `wo_b` partials
inside the traced graph.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

PMAX = 128
SEQ = 5
MAIN_SEQ = 1
START_POS = 128
WIN = 128
NKV = WIN + SEQ
FREQ_OFF = START_POS + MAIN_SEQ
DIM = 5120
Q_LORA = 1280
KV_DIM = 512
N_HEADS = 16
HEAD_DIM = 512
ROPE_OFF = 448
ROPE_PAIRS = 32
O_GROUPS = 2
HEADS_PER_GROUP = 8
O_LORA = 1024
O_GROUP_IN = 4096
O_INNER = 2048
SOFTMAX_SCALE = 0.044194173824159216
NORM_EPS = 1e-20
FP8_MAX = 448.0
NEG_BIG = -1.0e30
KFMAX = 1024
NCHMAX = 512
MAX_KBLK = 8
NBLK_MAX = NCHMAX // PMAX
DEPTH = 2
WD_DEPTH = 4
WDEPTH = 8
KF_QA = 1024
KF_QB = 640
KF_OA = 1024
KF_OB = 1024
NKT_MAX = DIM // 32
N_CORES = 2
WORLD_SIZE = 4
CC_CHUNK = 1024
O_SPLIT = DIM // N_CORES
Q_SPLIT = Q_LORA // N_CORES
KV_SPLIT = KV_DIM // N_CORES
EX_W = Q_SPLIT + KV_SPLIT

XQ_G = 8
XQ_GW = DIM // XQ_G
QA_G = 10
QA_GW = Q_LORA // QA_G
KV_G = 4
KV_GW = KV_DIM // KV_G
OG_G = 8
OG_GW = O_LORA // OG_G

_F32_MANT = 1 << 23
_MANT_1P75 = 6 * (1 << 20)
_RNE_MAGIC = 8388608.0
_BF16_MANT = 1 << 7
_MANT_SHIFT = 4
_SCALE_EXP_BIAS = 120
_E4M3_BITS_MASK = 0x87F0 - (1 << 16)


def _check_shape(cond, msg):
    assert cond, "[NCC_INKI016] Kernel validation exception: " + msg


def div_ceil(n, d):
    return (n + d - 1) // d


def _alloc_scratch():
    sc = {}
    for name in ("i32a", "i32b", "i32c"):
        sc[name] = nl.ndarray((PMAX, KFMAX), dtype=nl.int32, buffer=nl.sbuf)
    for name in ("f32a", "f32b", "f32c", "f32d"):
        sc[name] = nl.ndarray((PMAX, KFMAX), dtype=nl.float32, buffer=nl.sbuf)
    sc["u8"] = nl.ndarray((PMAX, DEPTH, KFMAX), dtype=nl.uint8, buffer=nl.sbuf)
    sc["stg"] = nl.ndarray((PMAX, DEPTH, KFMAX), dtype=nl.int16, buffer=nl.sbuf)
    sc["bits"] = nl.ndarray((PMAX, DEPTH, KFMAX), dtype=nl.int16, buffer=nl.sbuf)
    sc["wd"] = nl.ndarray((PMAX, WD_DEPTH, KFMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    sc["su8"] = nl.ndarray((PMAX, NBLK_MAX, NKT_MAX), dtype=nl.uint8, buffer=nl.sbuf)
    sc["sclb"] = nl.ndarray((PMAX, NBLK_MAX, NKT_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    for name in ("s32a", "s32b", "s32c", "s32d"):
        sc[name] = nl.ndarray((PMAX, KFMAX // 32), dtype=nl.int32, buffer=nl.sbuf)
    for name in ("sfa",):
        sc[name] = nl.ndarray((PMAX, KFMAX // 32), dtype=nl.float32, buffer=nl.sbuf)
    for name in ("r1", "r2", "r3"):
        sc[name] = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    sc["rv"] = nl.ndarray((PMAX, 8), dtype=nl.float32, buffer=nl.sbuf)
    sc["rope"] = nl.ndarray((PMAX, 4, ROPE_PAIRS, 1), dtype=nl.float32, buffer=nl.sbuf)
    sc["wt"] = nl.ndarray((PMAX, WDEPTH, MAX_KBLK, NCHMAX), dtype=nl.bfloat16,
                          buffer=nl.sbuf)
    sc["pm"] = nl.ndarray((PMAX, NCHMAX), dtype=nl.float32, buffer=nl.psum)
    for d in range(DEPTH):
        sc["pt" + str(d)] = nl.ndarray((PMAX, MAX_KBLK, PMAX), dtype=nl.bfloat16,
                                       buffer=nl.psum)
    sc["po"] = nl.ndarray((PMAX, HEAD_DIM), dtype=nl.float32, buffer=nl.psum)
    sc["ps"] = nl.ndarray((PMAX, NKV), dtype=nl.float32, buffer=nl.psum)
    sc["ppt"] = nl.ndarray((PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.psum)
    sc["kci"] = [0]
    return sc


def _bcast_row(dst, src_row, nparts):
    nisa.dma_copy(dst=dst, src=src_row.expand_dim(0).broadcast(0, nparts))


def _transpose_into(sc, src, nblk, dst, dst_blk0, nrow, pb=0):
    pt = sc["pt" + str(pb)]
    grp = 0
    while grp < nblk:
        cnt = min(MAX_KBLK, nblk - grp)
        for j in range(cnt):
            nisa.nc_transpose(dst=pt[0:PMAX, j, 0:nrow],
                              data=src[0:nrow, (grp + j) * PMAX:(grp + j + 1) * PMAX])
        nisa.tensor_copy(dst=dst[0:PMAX, dst_blk0 + grp:dst_blk0 + grp + cnt, 0:nrow],
                         src=pt[0:PMAX, 0:cnt, 0:nrow],
                         engine=nisa.engine.scalar)
        grp += MAX_KBLK


def _to_groups(dst, src, nrow, G, W):
    for g in range(G):
        nisa.dma_copy(dst=dst[g * nrow:(g + 1) * nrow, 0:W],
                      src=src[0:nrow, g * W:(g + 1) * W])


def _from_groups(dst, src, nrow, G, W):
    for g in range(G):
        nisa.dma_copy(dst=dst[0:nrow, g * W:(g + 1) * W],
                      src=src[g * nrow:(g + 1) * nrow, 0:W])


def _load_scale_panel(sc, s_hbm, n_off, nblk, K):
    nkt = K // 32
    rows = s_hbm[n_off // 32:n_off // 32 + 4 * nblk, 0:nkt].reshape_dim(0, (nblk, 4))
    for g in range(4):
        nisa.dma_copy(dst=sc["su8"][g * 32:(g + 1) * 32, 0:nblk, 0:nkt],
                      src=rows[0:nblk, g, 0:nkt].expand_dim(0).broadcast(0, 32))
    nisa.tensor_scalar(dst=sc["sclb"].view(nl.int16)[0:PMAX, 0:nblk, 0:nkt],
                       data=sc["su8"][0:PMAX, 0:nblk, 0:nkt],
                       op0=nl.add, operand0=_SCALE_EXP_BIAS, op1=nl.multiply,
                       operand1=_BF16_MANT)


def _decode_weight(sc, w_hbm, n_off, nb, k_off, kfs, pb, pw):
    nk = kfs // 32
    nisa.dma_copy(dst=sc["u8"][0:PMAX, pb, 0:kfs],
                  src=w_hbm[n_off:n_off + PMAX, k_off:k_off + kfs])
    sx = sc["stg"][0:PMAX, pb, 0:kfs]
    nisa.tensor_copy(dst=sx, src=sc["u8"].view(nl.int8)[0:PMAX, pb, 0:kfs],
                     engine=nisa.engine.scalar)

    bits = sc["bits"][0:PMAX, pb, 0:kfs]
    nisa.tensor_scalar(dst=bits, data=sx, op0=nl.left_shift, operand0=_MANT_SHIFT,
                       op1=nl.bitwise_and, operand1=_E4M3_BITS_MASK)
    val = sc["bits"].view(nl.bfloat16)[0:PMAX, pb, 0:kfs]

    scl = sc["sclb"][0:PMAX, nb, k_off // 32:k_off // 32 + nk]
    nisa.tensor_tensor(dst=sc["wd"][0:PMAX, pw, 0:kfs].reshape_dim(1, (nk, 32)),
                       data1=val.reshape_dim(1, (nk, 32)),
                       data2=scl.expand_dim(2).broadcast(2, 32), op=nl.multiply)


def _gemm(sc, a_t, k_off_blk, K, w_hbm, s_hbm, n_off, N, out_sb, out_off, kf):
    _check_shape(N % PMAX == 0, "GEMM N must be a multiple of 128")
    _check_shape(K % kf == 0 and kf % PMAX == 0, "bad GEMM k chunking")
    _check_shape(kf // PMAX <= MAX_KBLK, "GEMM k chunk wider than the transpose buffer")
    _check_shape(K // 32 <= NKT_MAX, "GEMM K wider than the block-scale panel")
    for n0 in range(0, N, NCHMAX):
        nch = min(NCHMAX, N - n0)
        nblk = nch // PMAX
        _check_shape(nblk <= NBLK_MAX, "more 128-row blocks than scale panel slots")
        _load_scale_panel(sc, s_hbm, n_off + n0, nblk, K)
        pb = 0
        pw = 0
        for kc in range(0, K, kf):
            nkb = kf // PMAX
            wb = sc["kci"][0] % WDEPTH
            for nn in range(nblk):
                _decode_weight(sc, w_hbm, n_off + n0 + nn * PMAX, nn, kc, kf, pb, pw)
                _transpose_into(sc, sc["wd"][0:PMAX, pw, 0:kf], nkb,
                                sc["wt"][0:PMAX, wb, 0:MAX_KBLK,
                                         nn * PMAX:(nn + 1) * PMAX],
                                0, PMAX, pb)
                pb = (pb + 1) % DEPTH
                pw = (pw + 1) % WD_DEPTH
            for j in range(nkb):
                nisa.nc_matmul(dst=sc["pm"][0:SEQ, 0:nch],
                               stationary=a_t[0:PMAX, k_off_blk + kc // PMAX + j, 0:SEQ],
                               moving=sc["wt"][0:PMAX, wb, j, 0:nch],
                               accumulate=(kc > 0 or j > 0))
            sc["kci"][0] += 1
        nisa.tensor_copy(dst=out_sb[0:SEQ, out_off + n0:out_off + n0 + nch],
                         src=sc["pm"][0:SEQ, 0:nch])


def _refine_rsqrt(sc, r, v, steps=2):
    t = sc["r3"][0:SEQ, 0:1]
    for _ in range(steps):
        nisa.tensor_tensor(dst=t, data1=r, data2=r, op=nl.multiply)
        nisa.tensor_tensor(dst=t, data1=t, data2=v, op=nl.multiply)
        nisa.tensor_scalar(dst=t, data=t, op0=nl.multiply, operand0=-0.5,
                           op1=nl.add, operand1=1.5)
        nisa.tensor_tensor(dst=r, data1=r, data2=t, op=nl.multiply)


def _refine_reciprocal(sc, r, d, steps=2):
    t = sc["rv"][0:SEQ, 7:8]
    for _ in range(steps):
        nisa.tensor_tensor(dst=t, data1=d, data2=r, op=nl.multiply)
        nisa.tensor_scalar(dst=t, data=t, op0=nl.multiply, operand0=-1.0,
                           op1=nl.add, operand1=2.0)
        nisa.tensor_tensor(dst=r, data1=r, data2=t, op=nl.multiply)


def _rmsnorm(sc, src, D, w_tile, dst, kf):
    nchunk = div_ceil(D, kf)
    _check_shape(nchunk <= 8, "rmsnorm chunk count too large")
    for c in range(nchunk):
        k0 = c * kf
        kfs = min(kf, D - k0)
        nisa.activation(dst=sc["f32a"][0:SEQ, 0:kfs], op=nl.square,
                        data=src[0:SEQ, k0:k0 + kfs], reduce_op=nl.add,
                        reduce_res=sc["rv"][0:SEQ, c:c + 1],
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
    nisa.tensor_reduce(dst=sc["r1"][0:SEQ, 0:1], op=nl.add,
                       data=sc["rv"][0:SEQ, 0:nchunk], axis=(1,))
    nisa.tensor_scalar(dst=sc["r1"][0:SEQ, 0:1], data=sc["r1"][0:SEQ, 0:1],
                       op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=NORM_EPS)
    nisa.activation(dst=sc["r2"][0:SEQ, 0:1], op=nl.rsqrt, data=sc["r1"][0:SEQ, 0:1])
    _refine_rsqrt(sc, sc["r2"][0:SEQ, 0:1], sc["r1"][0:SEQ, 0:1])
    for c in range(nchunk):
        k0 = c * kf
        kfs = min(kf, D - k0)
        nisa.activation(dst=sc["f32a"][0:SEQ, 0:kfs], op=nl.copy,
                        data=src[0:SEQ, k0:k0 + kfs], scale=sc["r2"][0:SEQ, 0:1])
        nisa.tensor_tensor(dst=dst[0:SEQ, k0:k0 + kfs], data1=sc["f32a"][0:SEQ, 0:kfs],
                           data2=w_tile[0:SEQ, k0:k0 + kfs], op=nl.multiply)


def _act_quant(sc, src, D, dst, kf, nrow=SEQ):
    _check_shape(D % 32 == 0, "act quant needs D % 32 == 0")
    for k0 in range(0, D, kf):
        kfs = min(kf, D - k0)
        nk = kfs // 32
        a = src[0:nrow, k0:k0 + kfs]
        amax = sc["sfa"][0:nrow, 0:nk]
        absa = sc["f32b"][0:nrow, 0:kfs]
        nisa.activation(dst=absa, op=nl.abs, data=a)
        nisa.tensor_reduce(dst=amax, op=nl.maximum,
                           data=absa.reshape_dim(1, (nk, 32)), axis=(2,))
        nisa.tensor_scalar(dst=amax, data=amax, op0=nl.maximum, operand0=1.0e-4)
        abits = sc["sfa"].view(nl.int32)[0:nrow, 0:nk]
        ex = sc["s32a"][0:nrow, 0:nk]
        nisa.tensor_scalar(dst=ex, data=abits, op0=nl.right_shift, operand0=23)
        thrb = sc["s32b"][0:nrow, 0:nk]
        nisa.tensor_scalar(dst=thrb, data=ex, op0=nl.multiply, operand0=_F32_MANT,
                           op1=nl.add, operand1=_MANT_1P75)
        thr = sc["s32b"].view(nl.float32)[0:nrow, 0:nk]
        gt = sc["s32c"][0:nrow, 0:nk]
        nisa.tensor_tensor(dst=gt, data1=amax, data2=thr, op=nl.greater)
        sexp = sc["s32d"][0:nrow, 0:nk]
        nisa.tensor_tensor(dst=sexp, data1=ex, data2=gt, op=nl.add)
        nisa.tensor_scalar(dst=sc["s32a"][0:nrow, 0:nk], data=sexp, op0=nl.add,
                           operand0=-8, op1=nl.multiply, operand1=_F32_MANT)
        sval = sc["s32a"].view(nl.float32)[0:nrow, 0:nk]
        nisa.tensor_scalar(dst=sc["s32b"][0:nrow, 0:nk], data=sexp, op0=nl.subtract,
                           operand0=262, reverse0=True, op1=nl.multiply, operand1=_F32_MANT)
        inv = sc["s32b"].view(nl.float32)[0:nrow, 0:nk]

        x = sc["f32a"][0:nrow, 0:kfs]
        nisa.tensor_tensor(dst=x.reshape_dim(1, (nk, 32)), data1=a.reshape_dim(1, (nk, 32)),
                           data2=inv.expand_dim(2).broadcast(2, 32), op=nl.multiply)
        ax = sc["f32b"][0:nrow, 0:kfs]
        nisa.activation(dst=ax, op=nl.abs, data=x)
        xbits = sc["f32b"].view(nl.int32)[0:nrow, 0:kfs]
        e1 = sc["i32a"][0:nrow, 0:kfs]
        nisa.tensor_scalar(dst=e1, data=xbits, op0=nl.right_shift, operand0=23)
        nisa.tensor_scalar(dst=e1, data=e1, op0=nl.maximum, operand0=121)
        nisa.tensor_scalar(dst=sc["i32b"][0:nrow, 0:kfs], data=e1, op0=nl.subtract,
                           operand0=257, reverse0=True, op1=nl.multiply, operand1=_F32_MANT)
        istep = sc["i32b"].view(nl.float32)[0:nrow, 0:kfs]
        nisa.tensor_scalar(dst=sc["i32c"][0:nrow, 0:kfs], data=e1, op0=nl.add,
                           operand0=-3, op1=nl.multiply, operand1=_F32_MANT)
        step = sc["i32c"].view(nl.float32)[0:nrow, 0:kfs]
        y = sc["f32c"][0:nrow, 0:kfs]
        nisa.tensor_tensor(dst=y, data1=ax, data2=istep, op=nl.multiply)
        fl = sc["f32d"][0:nrow, 0:kfs]
        nisa.tensor_scalar(dst=fl, data=y, op0=nl.add, operand0=_RNE_MAGIC)
        nisa.tensor_scalar(dst=fl, data=fl, op0=nl.subtract, operand0=_RNE_MAGIC)
        nisa.tensor_tensor(dst=fl, data1=fl, data2=step, op=nl.multiply)
        nisa.tensor_scalar(dst=fl, data=fl, op0=nl.minimum, operand0=FP8_MAX)
        sgn = sc["f32c"][0:nrow, 0:kfs]
        nisa.activation(dst=sgn, op=nl.sign, data=x)
        nisa.tensor_tensor(dst=fl, data1=fl, data2=sgn, op=nl.multiply)
        nisa.tensor_tensor(dst=dst[0:nrow, k0:k0 + kfs].reshape_dim(1, (nk, 32)),
                           data1=fl.reshape_dim(1, (nk, 32)),
                           data2=sval.expand_dim(2).broadcast(2, 32), op=nl.multiply)


def _rope(sc, tile, off, cos, sin, inverse):
    seg = tile[0:SEQ, off:off + 2 * ROPE_PAIRS].reshape_dim(1, (ROPE_PAIRS, 2))
    re = seg[0:SEQ, 0:ROPE_PAIRS, 0:1]
    im = seg[0:SEQ, 0:ROPE_PAIRS, 1:2]
    t0 = sc["rope"][0:SEQ, 0, 0:ROPE_PAIRS, 0:1]
    t1 = sc["rope"][0:SEQ, 1, 0:ROPE_PAIRS, 0:1]
    t2 = sc["rope"][0:SEQ, 2, 0:ROPE_PAIRS, 0:1]
    t3 = sc["rope"][0:SEQ, 3, 0:ROPE_PAIRS, 0:1]
    nisa.tensor_tensor(dst=t0, data1=re, data2=cos, op=nl.multiply)
    nisa.tensor_tensor(dst=t1, data1=im, data2=sin, op=nl.multiply)
    nisa.tensor_tensor(dst=t2, data1=re, data2=sin, op=nl.multiply)
    nisa.tensor_tensor(dst=t3, data1=im, data2=cos, op=nl.multiply)
    if inverse:
        nisa.tensor_tensor(dst=re, data1=t0, data2=t1, op=nl.add)
        nisa.tensor_tensor(dst=im, data1=t3, data2=t2, op=nl.subtract)
    else:
        nisa.tensor_tensor(dst=re, data1=t0, data2=t1, op=nl.subtract)
        nisa.tensor_tensor(dst=im, data1=t2, data2=t3, op=nl.add)


@nki.jit
def kernel(hidden_states, mtp_0_attn_attn_sink, mtp_0_attn_freqs_cis,
           mtp_0_attn_kv_norm_weight, mtp_0_attn_q_norm_weight, mtp_0_attn_window_kv_cache,
           mtp_0_attn_wkv_scale, mtp_0_attn_wkv_weight, mtp_0_attn_wo_a_scale,
           mtp_0_attn_wo_a_weight, mtp_0_attn_wo_b_scale, mtp_0_attn_wo_b_weight,
           mtp_0_attn_wq_a_scale, mtp_0_attn_wq_a_weight, mtp_0_attn_wq_b_scale,
           mtp_0_attn_wq_b_weight, mtp_0_attn_norm_weight, replica_group):
    _check_shape(hidden_states.shape == (1, SEQ, DIM), "unexpected hidden_states shape")
    _check_shape(mtp_0_attn_window_kv_cache.shape[1] == WIN, "unexpected ring length")
    _check_shape(mtp_0_attn_attn_sink.shape == (N_HEADS,), "attn_sink is not this rank's")
    _check_shape(mtp_0_attn_wq_b_weight.shape == (N_HEADS * HEAD_DIM, Q_LORA),
                 "wq_b is not this rank's head slice")
    _check_shape(mtp_0_attn_wo_a_weight.shape == (O_GROUPS * O_LORA, O_GROUP_IN),
                 "wo_a is not this rank's group slice")
    _check_shape(mtp_0_attn_wo_b_weight.shape == (DIM, O_INNER),
                 "wo_b is not this rank's reduction slice")

    out_hbm = nl.ndarray((1, SEQ, DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    cc_src = nl.ndarray((SEQ, DIM), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((SEQ, DIM), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_dst")
    sc = _alloc_scratch()
    pid = nl.program_id(0)
    og_hbm = nl.ndarray((N_CORES, SEQ, O_LORA), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    ex_hbm = nl.ndarray((N_CORES, SEQ, EX_W), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    hs = nl.ndarray((SEQ, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=hs, src=hidden_states[0, 0:SEQ, 0:DIM])
    wnorm = nl.ndarray((SEQ, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    _bcast_row(wnorm, mtp_0_attn_norm_weight[0:DIM], SEQ)

    cos = nl.ndarray((SEQ, ROPE_PAIRS, 1), dtype=nl.float32, buffer=nl.sbuf)
    sin = nl.ndarray((SEQ, ROPE_PAIRS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cos,
                  src=mtp_0_attn_freqs_cis[FREQ_OFF:FREQ_OFF + SEQ, 0:ROPE_PAIRS, 0:1])
    nisa.dma_copy(dst=sin,
                  src=mtp_0_attn_freqs_cis[FREQ_OFF:FREQ_OFF + SEQ, 0:ROPE_PAIRS, 1:2])
    sink = nl.ndarray((SEQ, N_HEADS), dtype=nl.float32, buffer=nl.sbuf)
    _bcast_row(sink, mtp_0_attn_attn_sink[0:N_HEADS], SEQ)

    kv_win = nl.ndarray((WIN, KV_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=kv_win, src=mtp_0_attn_window_kv_cache[0, 0:WIN, 0:KV_DIM])

    xq = nl.ndarray((SEQ, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    _rmsnorm(sc, hs, DIM, wnorm, xq, 1024)
    xqg = nl.ndarray((XQ_G * SEQ, XQ_GW), dtype=nl.bfloat16, buffer=nl.sbuf)
    _to_groups(xqg, xq, SEQ, XQ_G, XQ_GW)
    _act_quant(sc, xqg, XQ_GW, xqg, XQ_GW, nrow=XQ_G * SEQ)
    _from_groups(xq, xqg, SEQ, XQ_G, XQ_GW)
    xq_t = nl.ndarray((PMAX, DIM // PMAX, SEQ), dtype=nl.bfloat16, buffer=nl.sbuf)
    _transpose_into(sc, xq, DIM // PMAX, xq_t, 0, SEQ)

    qa = nl.ndarray((SEQ, Q_LORA), dtype=nl.bfloat16, buffer=nl.sbuf)
    kv_new = nl.ndarray((SEQ, KV_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    ex_h = nl.ndarray((SEQ, EX_W), dtype=nl.bfloat16, buffer=nl.sbuf)
    _gemm(sc, xq_t, 0, DIM, mtp_0_attn_wq_a_weight, mtp_0_attn_wq_a_scale,
          pid * Q_SPLIT, Q_SPLIT, ex_h, 0, KF_QA)
    _gemm(sc, xq_t, 0, DIM, mtp_0_attn_wkv_weight, mtp_0_attn_wkv_scale,
          pid * KV_SPLIT, KV_SPLIT, ex_h, Q_SPLIT, KF_QA)
    nisa.dma_copy(dst=ex_hbm[pid, 0:SEQ, 0:EX_W], src=ex_h)
    nisa.core_barrier(data=ex_hbm, cores=(0, 1))
    for c in range(N_CORES):
        nisa.dma_copy(dst=qa[0:SEQ, c * Q_SPLIT:(c + 1) * Q_SPLIT],
                      src=ex_hbm[c, 0:SEQ, 0:Q_SPLIT])
        nisa.dma_copy(dst=kv_new[0:SEQ, c * KV_SPLIT:(c + 1) * KV_SPLIT],
                      src=ex_hbm[c, 0:SEQ, Q_SPLIT:EX_W])
    wqn = nl.ndarray((SEQ, Q_LORA), dtype=nl.bfloat16, buffer=nl.sbuf)
    _bcast_row(wqn, mtp_0_attn_q_norm_weight[0:Q_LORA], SEQ)
    _rmsnorm(sc, qa, Q_LORA, wqn, qa, 1024)
    qag = nl.ndarray((QA_G * SEQ, QA_GW), dtype=nl.bfloat16, buffer=nl.sbuf)
    _to_groups(qag, qa, SEQ, QA_G, QA_GW)
    _act_quant(sc, qag, QA_GW, qag, QA_GW, nrow=QA_G * SEQ)
    _from_groups(qa, qag, SEQ, QA_G, QA_GW)
    qa_t = nl.ndarray((PMAX, Q_LORA // PMAX, SEQ), dtype=nl.bfloat16, buffer=nl.sbuf)
    _transpose_into(sc, qa, Q_LORA // PMAX, qa_t, 0, SEQ)

    wkn = nl.ndarray((SEQ, KV_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    _bcast_row(wkn, mtp_0_attn_kv_norm_weight[0:KV_DIM], SEQ)
    _rmsnorm(sc, kv_new, KV_DIM, wkn, kv_new, 512)
    _rope(sc, kv_new, ROPE_OFF, cos, sin, False)
    kvg = nl.ndarray((KV_G * SEQ, KV_GW), dtype=nl.bfloat16, buffer=nl.sbuf)
    _to_groups(kvg, kv_new, SEQ, KV_G, KV_GW)
    _act_quant(sc, kvg, KV_GW, kvg, KV_GW, nrow=KV_G * SEQ)
    _from_groups(kv_new, kvg, SEQ, KV_G, KV_GW)

    kv_t = nl.ndarray((PMAX, KV_DIM // PMAX, NKV), dtype=nl.bfloat16, buffer=nl.sbuf)
    for j in range(KV_DIM // PMAX):
        nisa.nc_transpose(dst=sc["pt0"][0:PMAX, 0, 0:WIN],
                          data=kv_win[0:WIN, j * PMAX:(j + 1) * PMAX])
        nisa.tensor_copy(dst=kv_t[0:PMAX, j, 0:WIN], src=sc["pt0"][0:PMAX, 0, 0:WIN])
        nisa.nc_transpose(dst=sc["pt0"][0:PMAX, 1, 0:SEQ],
                          data=kv_new[0:SEQ, j * PMAX:(j + 1) * PMAX])
        nisa.tensor_copy(dst=kv_t[0:PMAX, j, WIN:NKV], src=sc["pt0"][0:PMAX, 1, 0:SEQ])

    q_head = nl.ndarray((SEQ, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    o_head = nl.ndarray((SEQ, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    q_t = nl.ndarray((PMAX, HEAD_DIM // PMAX, SEQ), dtype=nl.bfloat16, buffer=nl.sbuf)
    p16 = nl.ndarray((SEQ, NKV), dtype=nl.bfloat16, buffer=nl.sbuf)
    pw_t = nl.ndarray((WIN, SEQ), dtype=nl.bfloat16, buffer=nl.sbuf)
    pn_t = nl.ndarray((SEQ, SEQ), dtype=nl.bfloat16, buffer=nl.sbuf)
    scores = nl.ndarray((SEQ, NKV), dtype=nl.float32, buffer=nl.sbuf)
    p32 = nl.ndarray((SEQ, NKV), dtype=nl.float32, buffer=nl.sbuf)
    o_t = nl.ndarray((PMAX, O_GROUP_IN // PMAX, SEQ), dtype=nl.bfloat16, buffer=nl.sbuf)
    og = nl.ndarray((SEQ, O_LORA), dtype=nl.bfloat16, buffer=nl.sbuf)

    g = pid
    for hh in range(HEADS_PER_GROUP):
        h = g * HEADS_PER_GROUP + hh
        _gemm(sc, qa_t, 0, Q_LORA, mtp_0_attn_wq_b_weight, mtp_0_attn_wq_b_scale,
              h * HEAD_DIM, HEAD_DIM, q_head, 0, KF_QB)
        _rope(sc, q_head, ROPE_OFF, cos, sin, False)
        _transpose_into(sc, q_head, HEAD_DIM // PMAX, q_t, 0, SEQ)

        for j in range(HEAD_DIM // PMAX):
            nisa.nc_matmul(dst=sc["ps"][0:SEQ, 0:NKV],
                           stationary=q_t[0:PMAX, j, 0:SEQ],
                           moving=kv_t[0:PMAX, j, 0:NKV], accumulate=(j > 0))
        nisa.tensor_scalar(dst=scores, data=sc["ps"][0:SEQ, 0:NKV],
                           op0=nl.multiply, operand0=SOFTMAX_SCALE)
        nisa.tensor_reduce(dst=sc["r1"][0:SEQ, 0:1], op=nl.maximum, data=scores,
                           axis=(1,))
        nisa.tensor_scalar(dst=sc["r1"][0:SEQ, 0:1], data=sc["r1"][0:SEQ, 0:1],
                           op0=nl.maximum, operand0=NEG_BIG,
                           op1=nl.multiply, operand1=-1.0)
        nisa.activation(dst=p32, op=nl.exp, data=scores, bias=sc["r1"][0:SEQ, 0:1],
                        reduce_op=nl.add, reduce_res=sc["r2"][0:SEQ, 0:1],
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        nisa.activation(dst=sc["r3"][0:SEQ, 0:1], op=nl.exp,
                        data=sc["r1"][0:SEQ, 0:1], bias=sink[0:SEQ, h:h + 1])
        nisa.tensor_tensor(dst=sc["r2"][0:SEQ, 0:1], data1=sc["r2"][0:SEQ, 0:1],
                           data2=sc["r3"][0:SEQ, 0:1], op=nl.add)
        nisa.tensor_copy(dst=p16, src=p32)
        nisa.nc_transpose(dst=sc["ppt"][0:WIN, 0:SEQ], data=p16[0:SEQ, 0:WIN])
        nisa.tensor_copy(dst=pw_t, src=sc["ppt"][0:WIN, 0:SEQ])
        nisa.nc_transpose(dst=sc["ppt"][0:SEQ, 0:SEQ], data=p16[0:SEQ, WIN:NKV])
        nisa.tensor_copy(dst=pn_t, src=sc["ppt"][0:SEQ, 0:SEQ])
        nisa.nc_matmul(dst=sc["po"][0:SEQ, 0:HEAD_DIM], stationary=pw_t,
                       moving=kv_win[0:WIN, 0:KV_DIM], accumulate=False)
        nisa.nc_matmul(dst=sc["po"][0:SEQ, 0:HEAD_DIM], stationary=pn_t,
                       moving=kv_new[0:SEQ, 0:KV_DIM], accumulate=True)
        nisa.reciprocal(dst=sc["r1"][0:SEQ, 0:1], data=sc["r2"][0:SEQ, 0:1])
        _refine_reciprocal(sc, sc["r1"][0:SEQ, 0:1], sc["r2"][0:SEQ, 0:1])
        nisa.tensor_scalar(dst=o_head, data=sc["po"][0:SEQ, 0:HEAD_DIM],
                           op0=nl.multiply, operand0=sc["r1"][0:SEQ, 0:1])
        _rope(sc, o_head, ROPE_OFF, cos, sin, True)
        _transpose_into(sc, o_head, HEAD_DIM // PMAX, o_t,
                        hh * (HEAD_DIM // PMAX), SEQ)
    _gemm(sc, o_t, 0, O_GROUP_IN, mtp_0_attn_wo_a_weight, mtp_0_attn_wo_a_scale,
          g * O_LORA, O_LORA, og, 0, KF_OA)

    ogg = nl.ndarray((OG_G * SEQ, OG_GW), dtype=nl.bfloat16, buffer=nl.sbuf)
    _to_groups(ogg, og, SEQ, OG_G, OG_GW)
    _act_quant(sc, ogg, OG_GW, ogg, OG_GW, nrow=OG_G * SEQ)
    for g in range(OG_G):
        nisa.dma_copy(dst=og_hbm[pid, 0:SEQ, g * OG_GW:(g + 1) * OG_GW],
                      src=ogg[g * SEQ:(g + 1) * SEQ, 0:OG_GW])
    nisa.core_barrier(data=og_hbm, cores=(0, 1))
    ogf = nl.ndarray((SEQ, O_INNER), dtype=nl.bfloat16, buffer=nl.sbuf)
    for c in range(N_CORES):
        nisa.dma_copy(dst=ogf[0:SEQ, c * O_LORA:(c + 1) * O_LORA],
                      src=og_hbm[c, 0:SEQ, 0:O_LORA])
    og_t = nl.ndarray((PMAX, O_INNER // PMAX, SEQ), dtype=nl.bfloat16, buffer=nl.sbuf)
    _transpose_into(sc, ogf, O_INNER // PMAX, og_t, 0, SEQ)
    out_sb = nl.ndarray((SEQ, O_SPLIT), dtype=nl.float32, buffer=nl.sbuf)
    _gemm(sc, og_t, 0, O_INNER, mtp_0_attn_wo_b_weight, mtp_0_attn_wo_b_scale,
          pid * O_SPLIT, O_SPLIT, out_sb, 0, KF_OB)

    nisa.dma_copy(dst=cc_src[0:SEQ, pid * O_SPLIT:pid * O_SPLIT + O_SPLIT], src=out_sb)
    nisa.core_barrier(data=cc_src, cores=(0, 1))
    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], replica_group=replica_group, op=nl.add)
    for cc0 in range(0, O_SPLIT, CC_CHUNK):
        ccw = min(CC_CHUNK, O_SPLIT - cc0)
        g0 = pid * O_SPLIT + cc0
        nisa.dma_copy(dst=out_hbm[0, 0:SEQ, g0:g0 + ccw],
                      src=cc_dst[0:SEQ, g0:g0 + ccw])
    return out_hbm
