"""`layers.1.engram` on 4 ranks, rejoined by one `nki.collectives.all_reduce`. The cut is `ngram`,
the contraction axis of `wkv`, so it rejoins by summation. The gate's contraction is
indivisible — it is a nonlinear function of the *summed* key — so every rank forms the gate from
the summed key and splits only its output rows.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

TOK = 128
DIM = 5120
HC = 4
NHC = 24
HD = 256
KDIM = NHC * HD
NKEY = DIM * HC
NVAL = DIM

WORLD_SIZE = 4
NHC_SHARD = NHC // WORLD_SIZE
KSL = KDIM // WORLD_SIZE

N_CORES = 2
HC_PER_CORE = HC // N_CORES

NHC_CORE = NHC // N_CORES
KDIM_CORE = NHC_CORE * HD

GSLOT = 3
GPAD_CORE = GSLOT * WORLD_SIZE
GPAD = GPAD_CORE * N_CORES
CCW = NVAL + GPAD

EPS = 1e-20
CLAMP = 1e-6
DIM_RSQRT = 5120 ** -0.5

ABLK = 32
FP8_MAX = 448.0
AMAX_FLOOR = 1e-4
LOG2E = 1.4426950408889634

PAR = 128
NMAX = 512
NW = 2560
NSL = NW // NMAX
NWB = NW // ABLK
RBLK = PAR // ABLK

NKT = KDIM // PAR
NKT_CORE = NKT // N_CORES
NVT = KSL // PAR
NKC = NKEY // NW
NKC_CORE = NKC // N_CORES

EXP_BIAS = 32
FP32_EXP_BIAS = 127.0
FP32_EXP_LSB = 8388608.0

BF16_EXP_LSB = 128
E8M0_SHIFT = 128


def _pow2(e_int, m, n):
    idx = nl.ndarray((m, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=idx, data=e_int,
                       op0=nl.maximum, operand0=float(-EXP_BIAS),
                       op1=nl.minimum, operand1=float(63 - EXP_BIAS))
    res = nl.ndarray((m, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=res.view(nl.uint32), data=idx,
                       op0=nl.add, operand0=FP32_EXP_BIAS,
                       op1=nl.multiply, operand1=FP32_EXP_LSB)
    return res


def _scratch(m, n):
    out = []
    for _ in range(2):
        out.append(nl.ndarray((m, n), dtype=nl.uint8, buffer=nl.sbuf))
    out.append(nl.ndarray((m, n // ABLK, ABLK), dtype=nl.bfloat16, buffer=nl.sbuf))
    return out


def _e8m0_to_bf16x2(raw, dst_u16):
    for k in range(2):
        nisa.tensor_scalar(dst=dst_u16[:, :, k:k + 1], data=raw,
                           op0=nl.multiply, operand0=BF16_EXP_LSB,
                           op1=nl.add, operand1=E8M0_SHIFT)


def _fp8_dequant(byte, pw3, dst3, m, n, s):
    _fix_byte(byte, m, n, s)
    _apply_scale(pw3, dst3, m, n, s)


def _fix_byte(byte, m, n, s):
    nisa.tensor_scalar(dst=s[0].view(nl.uint16), data=byte.view(nl.uint16),
                       op0=nl.bitwise_and, operand0=0x8080)
    nisa.scalar_tensor_tensor(dst=s[1], data=byte, op0=nl.subtract, operand0=8.0,
                              op1=nl.maximum, operand1=s[0])


def _apply_scale(pw32, dst3, m, n, s):
    nb = n // ABLK
    f3 = s[1].view(nl.float8_e4m3).reshape((m, nb, ABLK))
    nisa.tensor_copy(dst=dst3, src=f3, engine=nisa.engine.scalar)
    nisa.tensor_copy(dst=s[2].view(nl.uint32), src=pw32.broadcast(2, ABLK // 2),
                     engine=nisa.engine.scalar)
    nisa.tensor_tensor(dst=dst3, data1=dst3, op=nl.multiply, data2=s[2])


def _stage1(w, i, n0, bytes_, scrs, nrow=0, roff=None):
    d = i % len(bytes_)
    if roff is None:
        nisa.dma_copy(dst=bytes_[d], src=w[i * PAR:(i + 1) * PAR, n0:n0 + NW])
    else:
        nisa.dma_copy(dst=bytes_[d],
                      src=w.ap(pattern=[[nrow, PAR], [1, NW]], offset=i * PAR * nrow,
                               scalar_offset=roff, indirect_dim=1))
    _fix_byte(bytes_[d], PAR, NW, scrs[d])


def _embed(ids2, ncol, ew, es, s, su32, stride=None, base=0):
    st = ncol if stride is None else stride
    emb = nl.ndarray((TOK, ncol, HD // ABLK, ABLK), dtype=nl.bfloat16, buffer=nl.sbuf)
    for j in range(ncol):
        idx = nl.ndarray((TOK, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx, src=ids2.ap(pattern=[[st, TOK], [1, 1]], offset=base + j))

        wb = nl.ndarray((TOK, HD), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(dst=wb, src=ew.ap(pattern=[[HD, TOK], [1, HD]], offset=0,
                                        vector_offset=idx, indirect_dim=0))
        sb = nl.ndarray((TOK, HD // ABLK, 1), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(dst=sb, src=es.ap(pattern=[[HD // ABLK, TOK], [1, HD // ABLK]],
                                        offset=0, vector_offset=idx, indirect_dim=0))

        _e8m0_to_bf16x2(sb, su32.view(nl.uint16))
        _fp8_dequant(wb, su32, emb[:, j, :, :], TOK, HD, s)
    return emb.reshape((TOK, ncol * HD))


def _act_quant(x_bf16, m, k):
    nb = k // ABLK
    x2 = nl.copy(x_bf16, dtype=nl.float32)
    xf = x2.reshape((m, nb, ABLK))
    amax = nl.maximum(nl.max(nl.abs(xf), axis=2), AMAX_FLOOR)

    ratio = nl.multiply(amax, 1.0 / FP8_MAX)
    e = nl.copy(nl.ceil(nl.multiply(nl.log(ratio), LOG2E)), dtype=nl.int32)
    sc = nl.multiply(_pow2(e, m, nb), 2.0)
    inv = nl.multiply(_pow2(nl.subtract(0, e), m, nb), 0.5)

    sb = nl.broadcast_to(sc.reshape((m, nb, 1)), (m, nb, ABLK))
    ib = nl.broadcast_to(inv.reshape((m, nb, 1)), (m, nb, ABLK))
    v = nl.minimum(nl.maximum(nl.multiply(xf, ib), -FP8_MAX * 0.5), FP8_MAX * 0.5)
    q = nl.copy(nl.copy(v, dtype=nl.float8_e4m3), dtype=nl.float32)
    dq = nl.multiply(q, sb)
    out = nl.ndarray((m, k), dtype=nl.bfloat16, buffer=nl.sbuf)
    out[...] = nl.copy(dq.reshape((m, k)), dtype=nl.bfloat16)
    return out


def _transpose_stack(t, k, out=None, t0=0):
    nt = k // PAR
    if out is None:
        out = nl.ndarray((PAR, nt, TOK), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i in range(nt):
        out[:, t0 + i, :] = nl.transpose(t[:, i * PAR:(i + 1) * PAR])
    return out


def _wkv_scale_all(ws, nrow, n0, nti, praw, pw32, roff=None):
    ws3 = ws.reshape((1, RBLK * nti, nrow))
    for r in range(RBLK):
        if roff is None:
            src = ws3.ap(pattern=[[0, ABLK], [RBLK * nrow, nti], [1, NWB]],
                         offset=r * nrow + n0 // ABLK)
        else:
            src = ws3.ap(pattern=[[0, ABLK], [RBLK * nrow, nti], [1, NWB]],
                         offset=r * nrow, scalar_offset=roff, indirect_dim=2)
        nisa.dma_copy(dst=praw[r * ABLK:(r + 1) * ABLK, 0:nti, :, :], src=src)
    h16 = pw32.view(nl.uint16)
    for k in range(2):
        nisa.tensor_scalar(dst=h16[:, 0:nti, :, k:k + 1], data=praw[:, 0:nti, :, :],
                           op0=nl.multiply, operand0=BF16_EXP_LSB,
                           op1=nl.add, operand1=E8M0_SHIFT)


@nki.jit
def kernel(input_0, input_1, input_1_shard,
           layers_1_engram_k_weight, layers_1_engram_q_weight,
           layers_1_engram_wkv_key_scale, layers_1_engram_wkv_key_weight,
           layers_1_engram_wkv_value_scale, layers_1_engram_wkv_value_weight,
           layers_1_engram_embed_scale, layers_1_engram_embed_weight,
           replica_group):
    out = nl.ndarray((1, TOK, HC, DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    a_sh = nl.ndarray((PAR, NKT, TOK), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                      name="a_sh")
    cc_src = nl.ndarray((TOK, CCW), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((TOK, CCW), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_dst")

    pid = nl.program_id(0)

    rk_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.register_store(rk_sb, ncc.rank_id())

    koff_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    koff_sb[...] = nl.copy(nl.add(nl.multiply(rk_sb, N_CORES * NW), pid * NW),
                           dtype=nl.int32)

    voff_sb = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    voff_sb[...] = nl.copy(nl.multiply(rk_sb, NVT), dtype=nl.uint32)

    soff_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    soff_sb[...] = nl.copy(nl.add(nl.multiply(rk_sb, N_CORES * NWB), pid * NWB),
                           dtype=nl.int32)

    ones1 = nl.ndarray((1, TOK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(ones1, 1.0)
    iot = nl.ndarray((1, WORLD_SIZE), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=iot, pattern=[[1, WORLD_SIZE]], channel_multiplier=0)
    oh1 = nl.ndarray((1, WORLD_SIZE), dtype=nl.bfloat16, buffer=nl.sbuf)
    oh1[...] = nl.copy(nl.equal(iot, rk_sb.broadcast(1, WORLD_SIZE)), dtype=nl.bfloat16)
    ohp = nl.ndarray((TOK, WORLD_SIZE), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ohp, stationary=ones1, moving=oh1)
    onehot = nl.ndarray((TOK, WORLD_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=onehot, src=ohp, engine=nisa.engine.scalar)

    h2 = input_0.reshape((TOK, HC * DIM))
    out2 = out.reshape((TOK, HC * DIM))
    ids2 = input_1.reshape((TOK, NHC))
    idsv = input_1_shard.reshape((TOK, NHC_SHARD))
    qw2 = layers_1_engram_q_weight.reshape((1, HC * DIM))
    kw2 = layers_1_engram_k_weight.reshape((1, HC * DIM))

    esc = _scratch(TOK, HD)
    su32 = nl.ndarray((TOK, HD // ABLK, 1), dtype=nl.uint32, buffer=nl.sbuf)
    a_t = nl.ndarray((PAR, NKT, TOK), dtype=nl.bfloat16, buffer=nl.sbuf)
    emb = _embed(ids2, NHC_CORE, layers_1_engram_embed_weight,
                 layers_1_engram_embed_scale, esc, su32,
                 stride=NHC, base=NHC_CORE * pid)
    _transpose_stack(_act_quant(emb, TOK, KDIM_CORE), KDIM_CORE,
                     out=a_t, t0=NKT_CORE * pid)
    nisa.dma_copy(dst=a_sh[:, NKT_CORE * pid:NKT_CORE * (pid + 1), :],
                  src=a_t[:, NKT_CORE * pid:NKT_CORE * (pid + 1), :])
    nisa.core_barrier(data=a_sh, cores=(0, 1))
    oq = NKT_CORE * (N_CORES - 1 - pid)
    nisa.dma_copy(dst=a_t[:, oq:oq + NKT_CORE, :], src=a_sh[:, oq:oq + NKT_CORE, :])
    a_tv = nl.ndarray((PAR, NVT, TOK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=a_tv,
                     src=a_t.ap(pattern=[[NKT * TOK, PAR], [TOK, NVT], [1, TOK]],
                                offset=0, scalar_offset=voff_sb, indirect_dim=1),
                     engine=nisa.engine.scalar)

    praw = nl.ndarray((PAR, NKT, NWB, 1), dtype=nl.uint8, buffer=nl.sbuf)
    pw32 = nl.ndarray((PAR, NKT, NWB, 1), dtype=nl.uint32, buffer=nl.sbuf)
    DEPTH = 2
    wdq = []
    wdq2 = []
    scrs = []
    bytes_ = []
    for _ in range(DEPTH):
        t = nl.ndarray((PAR, NWB, ABLK), dtype=nl.bfloat16, buffer=nl.sbuf)
        wdq.append(t)
        wdq2.append(t.reshape((PAR, NW)))
        scrs.append(_scratch(PAR, NW))
        bytes_.append(nl.ndarray((PAR, NW), dtype=nl.uint8, buffer=nl.sbuf))
    kvb = nl.ndarray((TOK, NW), dtype=nl.bfloat16, buffer=nl.sbuf)
    wqb = nl.ndarray((TOK, NW), dtype=nl.bfloat16, buffer=nl.sbuf)
    wkb = nl.ndarray((TOK, NW), dtype=nl.bfloat16, buffer=nl.sbuf)
    rsq = nl.ndarray((TOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    rdot = nl.ndarray((TOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    rh = nl.ndarray((TOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    gate_scr = []
    for _ in range(3):
        gate_scr.append(nl.ndarray((TOK, NW), dtype=nl.float32, buffer=nl.sbuf))

    if True:
        acc = []
        for sl in range(NSL):
            acc.append(nl.ndarray((TOK, NMAX), dtype=nl.float32, buffer=nl.psum))

        _wkv_scale_all(layers_1_engram_wkv_key_scale, NKEY // ABLK, 0, NKT, praw, pw32,
                       roff=soff_sb)
        _stage1(layers_1_engram_wkv_key_weight, 0, 0, bytes_, scrs, NKEY, koff_sb)
        for i in range(NKT):
            d = i % DEPTH
            if i + 1 < NKT:
                _stage1(layers_1_engram_wkv_key_weight, i + 1, 0, bytes_, scrs,
                        NKEY, koff_sb)
            _apply_scale(pw32[:, i, :, :], wdq[d], PAR, NW, scrs[d])
            for sl in range(NSL):
                nisa.nc_matmul(dst=acc[sl], stationary=a_t[:, i, :],
                               moving=wdq2[d][:, sl * NMAX:(sl + 1) * NMAX],
                               accumulate=(i > 0))

        for sl in range(NSL):
            nisa.tensor_copy(dst=kvb[:, sl * NMAX:(sl + 1) * NMAX], src=acc[sl],
                             engine=nisa.engine.scalar)

        keyf = gate_scr[0]
        hf = gate_scr[1]
        hw = gate_scr[2]
        hf[...] = nl.load(h2.ap(pattern=[[HC * DIM, TOK], [1, NW]], offset=0,
                                scalar_offset=koff_sb, indirect_dim=1),
                          dtype=nl.float32)
        nisa.dma_copy(dst=wqb, src=qw2.ap(pattern=[[0, TOK], [1, NW]], offset=0,
                                          scalar_offset=koff_sb, indirect_dim=1))
        nisa.dma_copy(dst=wkb, src=kw2.ap(pattern=[[0, TOK], [1, NW]], offset=0,
                                          scalar_offset=koff_sb, indirect_dim=1))
        nisa.tensor_tensor(dst=hw, data1=wqb, op=nl.multiply, data2=wkb)
        hw[...] = nl.multiply(hw, hf)
        nisa.tensor_tensor(dst=hw, data1=hw, op=nl.multiply, data2=kvb)
        nisa.activation(dst=keyf, op=nl.square, data=kvb, reduce_op=nl.add,
                        reduce_res=rsq, reduce_cmd=nisa.reduce_cmd.reset_reduce)
        nisa.activation(dst=keyf, op=nl.square, data=hf, reduce_op=nl.add,
                        reduce_res=rh, reduce_cmd=nisa.reduce_cmd.reset_reduce)
        nisa.activation(dst=keyf, op=nl.copy, data=hw, reduce_op=nl.add,
                        reduce_res=rdot, reduce_cmd=nisa.reduce_cmd.reset_reduce)

    gpart = (rsq, rdot, rh)
    pad = nl.ndarray((TOK, GPAD_CORE), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(WORLD_SIZE):
        for k in range(GSLOT):
            pad[:, j * GSLOT + k:j * GSLOT + k + 1] = nl.multiply(onehot[:, j:j + 1],
                                                                  gpart[k])

    value = nl.ndarray((TOK, NVAL), dtype=nl.float32, buffer=nl.sbuf)
    v0 = pid * (NVAL // N_CORES)
    for ci in range(NVAL // N_CORES // NW):
        n0 = v0 + ci * NW
        acc = []
        for sl in range(NSL):
            acc.append(nl.ndarray((TOK, NMAX), dtype=nl.float32, buffer=nl.psum))

        _wkv_scale_all(layers_1_engram_wkv_value_scale, NVAL // ABLK, n0, NVT, praw, pw32)
        _stage1(layers_1_engram_wkv_value_weight, 0, n0, bytes_, scrs)
        for i in range(NVT):
            d = i % DEPTH
            if i + 1 < NVT:
                _stage1(layers_1_engram_wkv_value_weight, i + 1, n0, bytes_, scrs)
            _apply_scale(pw32[:, i, :, :], wdq[d], PAR, NW, scrs[d])
            for sl in range(NSL):
                nisa.nc_matmul(dst=acc[sl], stationary=a_tv[:, i, :],
                               moving=wdq2[d][:, sl * NMAX:(sl + 1) * NMAX],
                               accumulate=(i > 0))

        for sl in range(NSL):
            nisa.tensor_copy(dst=value[:, n0 + sl * NMAX:n0 + (sl + 1) * NMAX],
                             src=acc[sl], engine=nisa.engine.scalar)

    nisa.dma_copy(dst=cc_src[:, v0:v0 + NVAL // N_CORES],
                  src=value[:, v0:v0 + NVAL // N_CORES])
    nisa.dma_copy(dst=cc_src[:, NVAL + pid * GPAD_CORE:NVAL + (pid + 1) * GPAD_CORE],
                  src=pad)
    nisa.core_barrier(data=cc_src, cores=(0, 1))
    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], op=nl.add, replica_group=replica_group)
    nisa.dma_copy(dst=value, src=cc_dst[:, 0:NVAL])
    gpad = nl.ndarray((TOK, GPAD), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=gpad, src=cc_dst[:, NVAL:NVAL + GPAD])

    for cc in range(HC_PER_CORE):
        c = pid * HC_PER_CORE + cc
        b0 = c * GSLOT
        b1 = GPAD_CORE + c * GSLOT
        ksqc = nl.add(gpad[:, b0:b0 + 1], gpad[:, b1:b1 + 1])
        dotc = nl.add(gpad[:, b0 + 1:b0 + 2], gpad[:, b1 + 1:b1 + 2])
        hsqc = nl.add(gpad[:, b0 + 2:b0 + 3], gpad[:, b1 + 2:b1 + 3])
        hsq = nl.multiply(hsqc, 1.0 / DIM)
        kms = nl.multiply(ksqc, 1.0 / DIM)
        rstd = nl.multiply(nl.rsqrt(nl.add(hsq, EPS)), nl.rsqrt(nl.add(kms, EPS)))
        dot = nl.multiply(nl.multiply(dotc, rstd), DIM_RSQRT)
        mag = nl.sqrt(nl.maximum(nl.abs(dot), CLAMP))
        sgn = nl.subtract(nl.multiply(nl.copy(nl.greater_equal(dot, 0.0),
                                              dtype=nl.float32), 2.0), 1.0)
        gate = nl.ndarray((TOK, 1), dtype=nl.float32, buffer=nl.sbuf)
        gate[...] = nl.sigmoid(nl.multiply(mag, sgn))
        for sl in range(DIM // NW):
            n0 = c * DIM + sl * NW
            hf = gate_scr[0]
            yv = gate_scr[1]
            hf[...] = nl.load(h2[:, n0:n0 + NW], dtype=nl.float32)
            nisa.activation(dst=yv, op=nl.copy,
                            data=value[:, sl * NW:(sl + 1) * NW], scale=gate)
            nisa.tensor_tensor(dst=yv, data1=yv, op=nl.add, data2=hf)
            nl.store(out2[:, n0:n0 + NW], value=yv)

    return out
