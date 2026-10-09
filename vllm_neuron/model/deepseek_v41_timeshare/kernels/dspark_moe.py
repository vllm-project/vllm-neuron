"""`mtp.0.ffn` on 4 ranks: the chain `mtp.0.ffn_norm -> mtp.0.ffn` — an RMSNorm, then a 128-expert
top-3 MoE with fp4 routed experts and one fp8 shared expert (the `dspark_*` MoE config, not the
backbone's 384-expert one). Every rank runs the prologue replicated, the routed experts for its
own 32-expert shard, and a quarter of the shared expert; `ncc.all_reduce` sums the four
float32 accumulators.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

DIM = 5120
INTER = 2304
NEXP = 128
NTOK = 5
TOPK = 3
BLK = 32
NBD = DIM // BLK
NBI = INTER // BLK
NRD = INTER // 128
NRO = DIM // 128
CH = 80
TAIL = 1

W13B = 2
W2B = 4

NSLOT = NTOK * TOPK

EST1 = INTER * NBD * 16
EST1S = INTER * NBD
EST2 = DIM * NBI * 16
EST2S = DIM * NBI

WORLD_SIZE = 4
NLOCAL = NEXP // WORLD_SIZE

CORES = 2
FULL_LO = (0, 3)
FULL_HI = (2, NTOK)
PART_TOK = 2
PART_LO = (0, 2)
PART_HI = (2, TOPK)
SHARED_CORE = 1

INTER_PAD = 2560
INTER_L = INTER_PAD // WORLD_SIZE
NRS = INTER_L // 128
NBS = INTER_L // BLK

NORM_EPS = 1e-20
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 10.0
FP8_MAX_INV = 1.0 / 448.0
AMAX_FLOOR = 1e-4
RTNE_MAGIC = 12582912.0

NIB_MASK = 0x0F0F
NIB_MUL = 4160.0
FIELD_MASK = 0x81C081C0 - (1 << 32)

BYTE_HI = 256.0
E4M3_MAG2 = 0x7F007F00
E4M3_SGN2 = 0x80008000 - (1 << 32)

ACT_EXP = 118
W_SHIFT = 126 - ACT_EXP
F8_SHIFT = 120 - ACT_EXP

SHARED_DTYPE = nl.bfloat16


def _chunk(nb):
    n = (nb + CH - 1) // CH
    while nb % n:
        n += 1
    return nb // n


def _fp4_values(packed, nb, nt=1):
    m = packed.shape[0]
    out = nl.ndarray((m, nt, nb, BLK), dtype=nl.uint16, buffer=nl.sbuf)
    o32 = out.view(nl.uint32)
    p16 = packed.view(nl.uint16)
    cs = _chunk(nb)
    nib = nl.ndarray((m, nt, nb, 16), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nib[:, :, :, 0:8], data=p16, op0=nl.bitwise_and, operand0=NIB_MASK)
    nisa.tensor_scalar(dst=nib[:, :, :, 8:16], data=p16, op0=nl.right_shift, operand0=4,
                       op1=nl.bitwise_and, operand1=NIB_MASK)
    nib8 = nib.view(nl.uint8)
    for u in range(nt):
        for c in nl.affine_range(nb // cs):
            pre = nl.ndarray((m, cs, BLK), dtype=nl.uint16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=pre, data=nib8[:, u, c * cs:(c + 1) * cs, :],
                               op0=nl.multiply, operand0=NIB_MUL,
                               engine=nisa.engine.scalar)
            nisa.tensor_scalar(dst=o32[:, u, c * cs:(c + 1) * cs, :], data=pre.view(nl.uint32),
                               op0=nl.bitwise_and, operand0=FIELD_MASK)
    return out.view(nl.bfloat16)


def _fp8_values(raw, nb):
    m = raw.shape[0]
    out = nl.ndarray((m, 1, nb, BLK), dtype=nl.uint16, buffer=nl.sbuf)
    o32 = out.view(nl.uint32)[:, 0]
    wide = nl.ndarray((m, nb, BLK), dtype=nl.uint16, buffer=nl.sbuf)
    for half in range(2):
        nisa.tensor_scalar(dst=wide[:, :, half * 16:(half + 1) * 16],
                           data=raw[:, :, half:BLK:2],
                           op0=nl.multiply, operand0=BYTE_HI,
                           engine=nisa.engine.scalar)
    mag = nl.ndarray((m, nb, BLK), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=mag.view(nl.uint32), data=wide.view(nl.uint32),
                       op0=nl.bitwise_and, operand0=E4M3_MAG2,
                       op1=nl.right_shift, operand1=4)
    nisa.scalar_tensor_tensor(dst=o32, data=wide.view(nl.uint32),
                              op0=nl.bitwise_and, operand0=E4M3_SGN2,
                              op1=nl.bitwise_or, operand1=mag.view(nl.uint32))
    return out.view(nl.bfloat16)


def _pow2_from_e8m0(code, shift, shape=None):
    r = nl.ndarray(code.shape, dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=r, data=code,
                       op0=nl.multiply, operand0=float(1 << 7),
                       op1=nl.add, operand1=float(shift << 7),
                       engine=nisa.engine.scalar)
    return r.view(nl.bfloat16).reshape(
        shape if shape is not None else (code.shape[0], 1, code.shape[1]))


def _act_quant(x, nb):
    xf = nl.copy(x, dtype=nl.float32)
    amax = nl.ndarray((x.shape[0], nb, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=amax, op=nl.maximum, data=nl.abs(xf), axis=(2,), keepdims=True)
    v = nl.multiply(nl.maximum(amax, AMAX_FLOOR), FP8_MAX_INV)
    vi = v.view(nl.int32)
    ce = nl.add(nl.subtract(nl.bitwise_and(nl.right_shift(vi, 23), 255), 127),
                nl.minimum(nl.bitwise_and(vi, 0x7FFFFF), 1))
    s = nl.left_shift(nl.add(ce, 127 + ACT_EXP), 23).view(nl.float32)
    sinv = nl.left_shift(nl.subtract(127, ce), 23).view(nl.float32)
    q = nl.multiply(xf, sinv.broadcast(2, BLK))
    qe = nl.bitwise_and(nl.right_shift(q.view(nl.int32), 23), 255)
    se = nl.maximum(nl.subtract(qe, 130), -9)
    step = nl.left_shift(nl.add(se, 127), 23).view(nl.float32)
    stepinv = nl.left_shift(nl.subtract(127, se), 23).view(nl.float32)
    r = nl.subtract(nl.add(nl.multiply(q, stepinv), RTNE_MAGIC), RTNE_MAGIC)
    val = nl.multiply(nl.multiply(r, step), s.broadcast(2, BLK))
    out = nl.ndarray((x.shape[0], nb, BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out[:, :, 0:BLK // 2], src=val[:, :, 0:BLK:2])
    nisa.tensor_copy(dst=out[:, :, BLK // 2:BLK], src=val[:, :, 1:BLK:2])
    return out


def _broadcast_tokens(src, nb, scratch, ntok):
    nisa.dma_copy(dst=scratch[0, 0:ntok, 0:nb, 0:BLK], src=src)
    return _bcast(scratch, 0, ntok, nb)


def _bcast(scratch, t0, ntok, nb):
    dst = nl.ndarray((128, ntok, nb, BLK), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=dst, src=scratch[0:1, t0:t0 + ntok, 0:nb, 0:BLK].broadcast(0, 128))
    return dst


def _project(wvals, sf, actb, nb, dst, ntok, dtype=nl.float16, wbatch=False):
    m = wvals.shape[0]
    if wbatch:
        w, f = wvals, sf
        a = actb.broadcast(1, ntok) if ntok > 1 else actb
    else:
        w = wvals.broadcast(1, ntok) if ntok > 1 else wvals
        f = sf.broadcast(1, ntok) if ntok > 1 else sf
        a = actb
    cur = nl.ndarray((m, ntok, nb, BLK), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=cur, data1=w, data2=a, op=nl.multiply)
    width = BLK
    while width > TAIL:
        v = cur.reshape((m, ntok, nb, 2, width // 2))
        nxt = nl.ndarray((m, ntok, nb, width // 2), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=nxt, data1=v[:, :, :, 0, :], data2=v[:, :, :, 1, :],
                           op=nl.add)
        cur, width = nxt, width // 2
    if width == 1:
        part = cur.reshape((m, ntok, nb))
    else:
        part = nl.ndarray((m, ntok, nb), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=part, op=nl.add, axis=(3,), data=cur)
    scaled = nl.ndarray((m, ntok, nb), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=scaled, data1=part, data2=f, op=nl.multiply)
    nisa.tensor_reduce(dst=dst, op=nl.add, axis=(2,), data=scaled)


def _compact(rwl):
    livef = nl.ndarray((NLOCAL, NTOK), dtype=nl.float32, buffer=nl.sbuf)
    livef[...] = nl.copy(nl.greater(rwl, 0.0), dtype=nl.float32)
    live = nl.ndarray((NLOCAL, NTOK), dtype=nl.bfloat16, buffer=nl.sbuf)
    live[...] = nl.copy(livef, dtype=nl.bfloat16)

    tri = nl.zeros((NLOCAL, NLOCAL), dtype=nl.bfloat16, buffer=nl.sbuf)
    for f in range(1, NLOCAL):
        nisa.memset(dst=tri[0:f, f:f + 1], value=1.0)
    onec = nl.ndarray((NLOCAL, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=onec, value=1.0)
    pfxp = nl.ndarray((NLOCAL, NTOK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=pfxp, stationary=tri, moving=live)
    iotap = nl.ndarray((NLOCAL, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=iotap, stationary=tri, moving=onec)

    pfx = nl.copy(pfxp, dtype=nl.float32)
    sel = nl.ndarray((NLOCAL, NSLOT), dtype=nl.bfloat16, buffer=nl.sbuf)
    for t in range(NTOK):
        for r in range(TOPK):
            g = TOPK * t + r
            nisa.tensor_copy(
                dst=sel[:, g:g + 1],
                src=nl.multiply(livef[:, t:t + 1],
                                nl.copy(nl.equal(pfx[:, t:t + 1], float(r)),
                                        dtype=nl.float32)))

    ecol = nl.ndarray((NLOCAL, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    ecol[...] = nl.copy(iotap, dtype=nl.bfloat16)
    idxp = nl.ndarray((1, NSLOT), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=idxp, stationary=ecol, moving=sel)
    vldp = nl.ndarray((1, NSLOT), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=vldp, stationary=onec, moving=sel)

    idxf = nl.copy(idxp, dtype=nl.float32)
    vldf = nl.copy(vldp, dtype=nl.float32)
    eidx = nl.copy(nl.minimum(nl.maximum(idxf, 0.0), float(NLOCAL - 1)), dtype=nl.int32)
    ridx = nl.copy(nl.minimum(nl.maximum(nl.add(idxf,
                                                nl.multiply(nl.subtract(1.0, vldf),
                                                            float(NLOCAL))), 0.0),
                              float(NLOCAL)), dtype=nl.int32)
    return eidx, ridx


def _to_tokens(src, ntile, dst, ntok):
    for i in nl.affine_range(ntile):
        ps = nl.ndarray((ntok, 128), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=ps, data=src[:, i, :])
        nisa.tensor_copy(dst=dst[:, i, :], src=ps)


def _bf16(x):
    return nl.copy(nl.copy(x, dtype=nl.bfloat16), dtype=nl.float32)


def _swiglu(gate, up):
    u = nl.minimum(nl.maximum(_bf16(up), -SWIGLU_LIMIT), SWIGLU_LIMIT)
    return nl.multiply(nl.silu(nl.minimum(_bf16(gate), SWIGLU_LIMIT)), u)


def _routed_slot(t, r, lst, xa, rwscr, hscr, yT, w1p, w1s, w3p, w3s, w2p, w2s):
    ed0 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    rd0 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ed0, src=lst[0:1, TOPK * t + r:TOPK * t + r + 1])
    nisa.dma_copy(dst=rd0, src=lst[1:2, TOPK * t + r:TOPK * t + r + 1])
    e = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    rd = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=e, src=ed0)
    nisa.tensor_copy(dst=rd, src=rd0)

    s13 = nl.ndarray((128, W13B, NRD, NBD), dtype=nl.uint8, buffer=nl.sbuf)
    for i in nl.affine_range(NRD):
        nisa.dma_copy(dst=s13[:, 0, i, :], src=w1s[0:1, i * 128:(i + 1) * 128, 0:NBD].ap(
            pattern=[[EST1S, 1], [NBD, 128], [1, NBD]],
            scalar_offset=e, indirect_dim=0).squeeze_dim(0))
        nisa.dma_copy(dst=s13[:, 1, i, :], src=w3s[0:1, i * 128:(i + 1) * 128, 0:NBD].ap(
            pattern=[[EST1S, 1], [NBD, 128], [1, NBD]],
            scalar_offset=e, indirect_dim=0).squeeze_dim(0))
    f13 = _pow2_from_e8m0(s13.reshape((128, W13B * NRD * NBD)), W_SHIFT,
                          (128, W13B, NRD, NBD))

    guT = nl.ndarray((128, NRD, W13B), dtype=nl.float32, buffer=nl.sbuf)
    for i in nl.affine_range(NRD):
        wb = nl.ndarray((128, W13B, NBD, 16), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(dst=wb[:, 0], src=w1p[0:1, i * 128:(i + 1) * 128, 0:NBD, 0:16].ap(
            pattern=[[EST1, 1], [NBD * 16, 128], [16, NBD], [1, 16]],
            scalar_offset=e, indirect_dim=0).squeeze_dim(0))
        nisa.dma_copy(dst=wb[:, 1], src=w3p[0:1, i * 128:(i + 1) * 128, 0:NBD, 0:16].ap(
            pattern=[[EST1, 1], [NBD * 16, 128], [16, NBD], [1, 16]],
            scalar_offset=e, indirect_dim=0).squeeze_dim(0))
        _project(_fp4_values(wb, NBD, W13B), f13[:, :, i, :], xa, NBD,
                 guT[:, i, :], W13B, wbatch=True)
    gateT = guT[:, :, 0:1]
    upT = guT[:, :, 1:2]

    rw = nl.ndarray((128, 1, NTOK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rw[:, 0, :], src=rwscr[0:1, 0:1, 0:NTOK].ap(
        pattern=[[NTOK, 1], [0, 128], [1, NTOK]],
        scalar_offset=rd, indirect_dim=0).squeeze_dim(0))
    hid = nl.multiply(_swiglu(gateT, upT),
                      rw[:, 0:1, t:t + 1].broadcast(1, NRD))
    hsb = nl.ndarray((1, NRD, 128), dtype=nl.float32, buffer=nl.sbuf)
    _to_tokens(hid, NRD, hsb, 1)
    hdqb = _broadcast_tokens(
        _act_quant(nl.copy(hsb.reshape((1, NBI, BLK)), dtype=nl.bfloat16), NBI),
        NBI, hscr, 1)

    s2 = nl.ndarray((128, NRO, NBI), dtype=nl.uint8, buffer=nl.sbuf)
    for j in nl.affine_range(NRO):
        nisa.dma_copy(dst=s2[:, j, :], src=w2s[0:1, j * 128:(j + 1) * 128, 0:NBI].ap(
            pattern=[[EST2S, 1], [NBI, 128], [1, NBI]],
            scalar_offset=e, indirect_dim=0).squeeze_dim(0))
    f2 = _pow2_from_e8m0(s2.reshape((128, NRO * NBI)), W_SHIFT, (128, NRO, NBI))

    accs = nl.ndarray((128, NRO, 1), dtype=nl.float32, buffer=nl.sbuf)
    for j in nl.affine_range(NRO // W2B):
        wb2 = nl.ndarray((128, W2B, NBI, 16), dtype=nl.uint8, buffer=nl.sbuf)
        for u in range(W2B):
            o = (j * W2B + u) * 128
            nisa.dma_copy(dst=wb2[:, u], src=w2p[0:1, o:o + 128, 0:NBI, 0:16].ap(
                pattern=[[EST2, 1], [NBI * 16, 128], [16, NBI], [1, 16]],
                scalar_offset=e, indirect_dim=0).squeeze_dim(0))
        _project(_fp4_values(wb2, NBI, W2B), f2[:, j * W2B:(j + 1) * W2B, :], hdqb, NBI,
                 accs[:, j * W2B:(j + 1) * W2B, 0], W2B, wbatch=True)
    nisa.tensor_copy(dst=yT[:, 0:NRO, t:t + 1],
                     src=nl.add(yT[:, 0:NRO, t:t + 1], _bf16(accs)))


@nki.jit
def kernel(x_in, norm_w, gate_w, gate_b, w1p, w1s, w3p, w3s, w2p, w2s,
           sw1p, sw1s, sw3p, sw3s, sw2p, sw2s, replica_group):
    out = nl.ndarray((1, NTOK, DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    cc_src = nl.ndarray((NTOK, DIM), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((NTOK, DIM), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_dst")
    part = nl.ndarray((CORES * NTOK, DIM), dtype=nl.float32, buffer=nl.shared_hbm, name="part")

    xscr = nl.ndarray((1, NTOK, NBD, BLK), dtype=nl.bfloat16, buffer=nl.private_hbm)
    hscr = nl.ndarray((1, 1, NBI, BLK), dtype=nl.bfloat16, buffer=nl.private_hbm)
    sscr = nl.ndarray((1, NTOK, NBS, BLK), dtype=nl.bfloat16, buffer=nl.private_hbm)
    rwscr = nl.ndarray((NLOCAL + 1, 1, NTOK), dtype=nl.float32, buffer=nl.private_hbm)
    lst = nl.ndarray((2, NSLOT), dtype=nl.int32, buffer=nl.private_hbm)

    pid = nl.program_id(0)

    xt = nl.ndarray((NTOK, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x_in[0, 0:NTOK, 0:DIM])
    nwt = nl.ndarray((NTOK, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=nwt, src=norm_w[0:1, 0:DIM].broadcast(0, NTOK))
    xf = nl.copy(xt, dtype=nl.float32)
    ss = nl.ndarray((NTOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ss, op=nl.add, data=nl.multiply(xf, xf), axis=(1,),
                       keepdims=True)
    rs = nl.rsqrt(nl.add(nl.multiply(ss, 1.0 / DIM), NORM_EPS))
    hx = nl.copy(nl.multiply(nl.multiply(xf, rs.broadcast(1, DIM)),
                             nl.copy(nwt, dtype=nl.float32)), dtype=nl.bfloat16)
    hscr2 = nl.ndarray((1, NTOK, DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)
    nisa.dma_copy(dst=hscr2[0, 0:NTOK, 0:DIM], src=hx)

    gw = nl.ndarray((NEXP, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=gw, src=gate_w[0:NEXP, 0:DIM])
    logitsT = nl.ndarray((NEXP, NTOK), dtype=nl.float32, buffer=nl.sbuf)
    hb = nl.ndarray((NEXP, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    for t in range(NTOK):
        nisa.dma_copy(dst=hb, src=hscr2[0:1, t, 0:DIM].broadcast(0, NEXP))
        nisa.tensor_reduce(dst=logitsT[:, t:t + 1], op=nl.add,
                           data=nl.multiply(gw, hb, dtype=nl.float32),
                           axis=(1,), keepdims=True)
    lps = nl.ndarray((NTOK, NEXP), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=lps, data=logitsT)
    scores = nl.sqrt(nl.softplus(nl.copy(lps, dtype=nl.float32)))

    gb = nl.ndarray((NTOK, NEXP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=gb, src=gate_b[0:1, 0:NEXP].broadcast(0, NTOK))
    biased = nl.add(scores, gb)
    cur = nl.copy(biased, dtype=nl.float32)
    thr = nl.ndarray((NTOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    for r in range(TOPK):
        nisa.tensor_reduce(dst=thr, op=nl.maximum, data=cur, axis=(1,), keepdims=True)
        if r < TOPK - 1:
            hit = nl.copy(nl.equal(cur, thr.broadcast(1, NEXP)), dtype=nl.float32)
            cur = nl.subtract(cur, nl.multiply(hit, 1.0e30))
    keep = nl.copy(nl.greater_equal(biased, thr.broadcast(1, NEXP)), dtype=nl.float32)
    wsel = nl.multiply(scores, keep)
    wsum = nl.ndarray((NTOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=wsum, op=nl.add, data=wsel, axis=(1,), keepdims=True)
    winv = nl.ndarray((NTOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=winv, data=nl.add(wsum, 1.0e-20))
    rweight = nl.multiply(nl.multiply(wsel, winv.broadcast(1, NEXP)), ROUTE_SCALE)
    rwps = nl.ndarray((NEXP, NTOK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=rwps, data=rweight)
    rwl = nl.copy(rwps[0:NLOCAL, 0:NTOK], dtype=nl.float32)
    nisa.dma_copy(dst=rwscr[0:NLOCAL, 0, 0:NTOK], src=rwl)
    nisa.dma_copy(dst=rwscr[NLOCAL:NLOCAL + 1, 0, 0:NTOK],
                  src=nl.zeros((1, NTOK), dtype=nl.float32, buffer=nl.sbuf))
    eidx, ridx = _compact(rwl)
    nisa.dma_copy(dst=lst[0:1, 0:NSLOT], src=eidx)
    nisa.dma_copy(dst=lst[1:2, 0:NSLOT], src=ridx)

    nisa.dma_copy(dst=xscr[0, 0:NTOK, 0:NBD, 0:BLK],
                  src=_act_quant(hx.reshape((NTOK, NBD, BLK)), NBD))

    yT = nl.zeros((128, NRO, NTOK), dtype=nl.float32, buffer=nl.sbuf)

    for t in range(FULL_LO[pid], FULL_HI[pid]):
        xat = _bcast(xscr, t, 1, NBD)
        for r in nl.sequential_range(TOPK):
            _routed_slot(t, r, lst, xat, rwscr, hscr, yT, w1p, w1s, w3p, w3s, w2p, w2s)
    if PART_HI[pid] > PART_LO[pid]:
        xap = _bcast(xscr, PART_TOK, 1, NBD)
        for r in range(PART_LO[pid], PART_HI[pid]):
            _routed_slot(PART_TOK, r, lst, xap, rwscr, hscr, yT,
                         w1p, w1s, w3p, w3s, w2p, w2s)

    if pid == SHARED_CORE:
        xdqb = _bcast(xscr, 0, NTOK, NBD)
        sgateT = nl.ndarray((128, NRS, NTOK), dtype=nl.float32, buffer=nl.sbuf)
        supT = nl.ndarray((128, NRS, NTOK), dtype=nl.float32, buffer=nl.sbuf)
        for i in nl.affine_range(NRS):
            sb = nl.ndarray((128, NBD, BLK), dtype=nl.uint8, buffer=nl.sbuf)
            sq = nl.ndarray((128, NBD), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.dma_copy(dst=sb, src=sw1p[i * 128:(i + 1) * 128, 0:NBD, 0:BLK])
            for g in nl.affine_range(4):
                nisa.dma_copy(dst=sq[g * BLK:(g + 1) * BLK, 0:NBD],
                              src=sw1s[i * 4 + g:i * 4 + g + 1, 0:NBD].broadcast(0, BLK))
            sf1 = _pow2_from_e8m0(sq, F8_SHIFT)
            wv1 = _fp8_values(sb, NBD)
            for t in range(NTOK):
                _project(wv1, sf1, xdqb[:, t:t + 1], NBD, sgateT[:, i, t:t + 1], 1,
                         SHARED_DTYPE)
            nisa.dma_copy(dst=sb, src=sw3p[i * 128:(i + 1) * 128, 0:NBD, 0:BLK])
            for g in nl.affine_range(4):
                nisa.dma_copy(dst=sq[g * BLK:(g + 1) * BLK, 0:NBD],
                              src=sw3s[i * 4 + g:i * 4 + g + 1, 0:NBD].broadcast(0, BLK))
            sf3 = _pow2_from_e8m0(sq, F8_SHIFT)
            wv3 = _fp8_values(sb, NBD)
            for t in range(NTOK):
                _project(wv3, sf3, xdqb[:, t:t + 1], NBD, supT[:, i, t:t + 1], 1,
                         SHARED_DTYPE)

        shsb = nl.ndarray((NTOK, NRS, 128), dtype=nl.float32, buffer=nl.sbuf)
        _to_tokens(_swiglu(sgateT, supT), NRS, shsb, NTOK)
        sdqb = _broadcast_tokens(
            _act_quant(nl.copy(shsb.reshape((NTOK, NBS, BLK)), dtype=nl.bfloat16), NBS),
            NBS, sscr, NTOK)

        for j in nl.affine_range(NRO):
            sb2 = nl.ndarray((128, NBS, BLK), dtype=nl.uint8, buffer=nl.sbuf)
            sq2 = nl.ndarray((128, NBS), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.dma_copy(dst=sb2, src=sw2p[j * 128:(j + 1) * 128, 0:NBS, 0:BLK])
            for g in nl.affine_range(4):
                nisa.dma_copy(dst=sq2[g * BLK:(g + 1) * BLK, 0:NBS],
                              src=sw2s[j * 4 + g:j * 4 + g + 1, 0:NBS].broadcast(0, BLK))
            sf2 = _pow2_from_e8m0(sq2, F8_SHIFT)
            wv2 = _fp8_values(sb2, NBS)
            sacc = nl.ndarray((128, NTOK), dtype=nl.float32, buffer=nl.sbuf)
            _project(wv2, sf2, sdqb[:, 0:NTOK], NBS, sacc, NTOK, SHARED_DTYPE)
            nisa.tensor_copy(dst=yT[:, j, :], src=nl.add(yT[:, j, :], _bf16(sacc)))

    ysb = nl.ndarray((NTOK, NRO, 128), dtype=nl.float32, buffer=nl.sbuf)
    _to_tokens(yT, NRO, ysb, NTOK)
    y = ysb.reshape((NTOK, DIM))

    oth = (1 - pid) * NTOK
    nisa.dma_copy(dst=part[pid * NTOK:pid * NTOK + NTOK, 0:DIM], src=y)
    nisa.core_barrier(data=part, cores=(0, 1))
    nisa.dma_copy(dst=cc_src[0:NTOK, 0:DIM],
                  src=nl.add(y, nl.load(part[oth:oth + NTOK, 0:DIM])))

    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], replica_group=replica_group, op=nl.add)

    nisa.dma_copy(dst=out[0, 0:NTOK, 0:DIM],
                  src=nl.copy(nl.load(cc_dst[0:NTOK, 0:DIM]), dtype=nl.bfloat16))
    return out
