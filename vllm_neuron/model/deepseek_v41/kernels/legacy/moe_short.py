"""`layers.1.ffn` on 4 ranks, rejoining through `nki.collectives`. Every rank runs the whole
`ffn_norm -> router` prologue replicated, the SwiGLU experts for its own 96-expert shard, and a
quarter of the shared expert, accumulating all of it in float32; `ncc.all_reduce` sums the four
accumulators into the module's output on every rank.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

DIM = 5120
INTER = 2304
INTER_PAD = 2560
SHARDS = 4
INTER_L = INTER_PAD // SHARDS
N_EXPERTS = 384
TOPK = 6
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 10.0
NORM_EPS = 1e-20
SUM_EPS = 1e-20
TOK = 128

PAR = 128
NMAX = 512
ABLK = 32
FP8_MAX = 448.0
AMAX_FLOOR = 1e-4
LOG2E = 1.4426950408889634

CH = 96
CORES = 2
CHH = CH // CORES
SH_C0 = (0, 256)
SH_CW = (256, INTER_L - 256)
BPB = ABLK // 2
KBLK1 = DIM // ABLK
KBLK2 = INTER // ABLK
KB1 = DIM // 2
KB2 = INTER // 2
NW1 = 1152
NW2 = 1280
NBG1 = 2
NBG2 = 2
KSP1 = 1
KSP2 = 1

SCENG1 = (nisa.engine.scalar,)
SCENG2 = (nisa.engine.scalar,)
DRNENG = (nisa.engine.scalar,)
TRENG = (nisa.engine.scalar,)
PRENG1 = (nisa.engine.scalar,)
PRENG2 = (nisa.engine.scalar,)

NIBENG = ((nisa.engine.vector,),
          (nisa.engine.vector,))

MULENG = (nisa.engine.vector,)

FP8ENG = nisa.engine.vector

EXP_BIAS = 32
BF_BIAS = 127
F32_MANT = 23
F32_MANT_MAX = (1 << F32_MANT) - 1
RATIO_MAX = float(1 << (EXP_BIAS - 1))
RATIO_MIN_BITS = ((BF_BIAS - EXP_BIAS - 1) << F32_MANT) + 1
BF_STEP = 128.0
BF_HALF_BIAS = float((2 * BF_BIAS - 1) << 7)

NIB_MUL = 4160.0
PAT_MASK2 = -2118090304
QUAD_NIB_MASK = 0x0F0F0F0F
HI_NIB_SHIFT = 4
UNROLL = 24
STOFF = (-2, 0, 0, 1, 3, 5)
DDEP = 1 + STOFF[1] - STOFF[0] + 2
NDEP = 1 + STOFF[3] - STOFF[1] + 3
assert 1 + STOFF[2] - STOFF[0] <= DDEP
assert CH % UNROLL == 0
assert CHH % UNROLL == 0
PDEP1 = 6
PDEP2 = 7
CHSP1 = True
CHSP2 = True
SC_MUL = 128.0
SC_ADD = 16128.0

BYTE_HI = 256.0
E4M3_MAG2 = 0x7F007F00
E4M3_SGN2 = 0x80008000 - (1 << 32)
CCH = 1024
assert DIM % CCH == 0
SC_MUL8 = 128.0
SC_ADD8 = float(120 << 7)


def _act_quant(x_bf16, m, k, chunks):
    nb = k // ABLK
    xf = x_bf16.reshape((m, nb, ABLK))
    mx = nl.ndarray((m, nb), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mx, op=nl.max, data=xf, axis=2)
    nmn = nl.ndarray((m, nb), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=nmn, op=nl.min, data=xf, axis=2, negate=True)
    amax = nl.ndarray((m, nb), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=amax, data=mx, op0=nl.maximum, operand0=AMAX_FLOOR,
                              op1=nl.maximum, operand1=nmn)

    ratio = nl.ndarray((m, nb), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ratio, data=amax, op0=nl.multiply, operand0=1.0 / FP8_MAX,
                       op1=nl.minimum, operand1=RATIO_MAX, engine=nisa.engine.vector)
    eb = nl.ndarray((m, nb), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=eb, data=ratio.view(nl.int32), op0=nl.maximum,
                       operand0=RATIO_MIN_BITS, op1=nl.add, operand1=F32_MANT_MAX,
                       engine=nisa.engine.vector)
    ef = nl.ndarray((m, nb), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ef, data=eb, op0=nl.right_shift, operand0=F32_MANT,
                       engine=nisa.engine.vector)

    sb = nl.ndarray((m, nb), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sb, data=ef, op0=nl.multiply, operand0=BF_STEP,
                       engine=nisa.engine.vector)
    s = sb.view(nl.bfloat16)
    ivb = nl.ndarray((m, nb), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ivb, data=ef, op0=nl.multiply, operand0=-BF_STEP,
                       op1=nl.add, operand1=BF_HALF_BIAS, engine=nisa.engine.vector)
    inv = ivb.view(nl.bfloat16)

    ib = inv.reshape((m, nb, 1)).broadcast(2, ABLK)

    q = nl.ndarray((m, nb, ABLK), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=q, data1=xf, data2=ib, op=nl.multiply,
                       engine=nisa.engine.vector)
    q4 = q.reshape((m, nb, BPB, 2))
    fls = []
    for c in range(len(chunks)):
        kb0 = chunks[c][0]
        nk = chunks[c][1]
        s2 = nl.multiply(s[:, kb0:kb0 + nk], 2.0).reshape((m, 1, 1, nk))
        fl = nl.ndarray((m, 2, BPB, nk), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=fl,
                           data1=q4[:, kb0:kb0 + nk, :, :].permute((0, 3, 2, 1)),
                           data2=s2.broadcast(1, 2).broadcast(2, BPB),
                           op=nl.multiply, engine=nisa.engine.vector)
        fls.append(fl)
    return fls


def _transpose_stack(t, k):
    nt = k // PAR
    out = nl.ndarray((PAR, nt, TOK), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i in range(nt):
        out[:, i, :] = nl.transpose(t[:, i * PAR:(i + 1) * PAR])
    return out


def _swiglu(g_bf16, u_bf16, wcol, dst):
    m, k = g_bf16.shape
    g = nl.ndarray((m, k), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=g, data=g_bf16, op0=nl.minimum, operand0=SWIGLU_LIMIT,
                       engine=nisa.engine.vector)
    u = nl.ndarray((m, k), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=u, data=u_bf16, op0=nl.maximum, operand0=-SWIGLU_LIMIT,
                       op1=nl.minimum, operand1=SWIGLU_LIMIT, engine=nisa.engine.vector)
    sg = nl.ndarray((m, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sg, op=nl.silu, data=g)
    nisa.scalar_tensor_tensor(dst=dst, data=sg, op0=nl.multiply, operand0=wcol,
                              op1=nl.multiply, operand1=u)


def _flat_chunks(kblk, split):
    if not split:
        return [(0, kblk, 0, kblk * BPB // PAR)]
    ch = []
    kb0 = 0
    while kb0 < kblk:
        nk = PAR
        while nk > kblk - kb0:
            nk //= 2
        assert nk * BPB % PAR == 0
        ch.append((kb0, nk, PAR // nk, nk * BPB // PAR))
        kb0 += nk
    return ch


def _flat_plan(kblk, split):
    tiles = []
    if not split:
        nt = kblk * BPB // PAR
        for _ in range(nt):
            tiles.append([])
        for off in range(BPB):
            kb = 0
            while kb < kblk:
                f = off * kblk + kb
                t = f // PAR
                p = f - t * PAR
                ln = min(kblk - kb, PAR - p)
                tiles[t].append((p, off, kb, ln))
                kb += ln
        return tiles
    chs = _flat_chunks(kblk, True)
    for c in range(len(chs)):
        kb0 = chs[c][0]
        nk = chs[c][1]
        nof = chs[c][2]
        for a in range(chs[c][3]):
            segs = []
            for i in range(nof):
                segs.append((i * nk, a * nof + i, kb0, nk))
            tiles.append(segs)
    return tiles


def _assert_covered(segs, name):
    for t in range(len(segs)):
        cov = []
        for _ in range(PAR):
            cov.append(0)
        for si in range(len(segs[t])):
            p = segs[t][si][0]
            ln = segs[t][si][3]
            for i in range(p, p + ln):
                cov[i] += 1
        for i in range(PAR):
            assert cov[i] == 1, \
                "%s tile %d partition %d covered %d times, not once" % (name, t, i, cov[i])


def _scale_plan(segs):
    keys = []
    idx = []
    for t in range(len(segs)):
        key = []
        for si in range(len(segs[t])):
            key.append((segs[t][si][0], segs[t][si][2], segs[t][si][3]))
        key = tuple(key)
        if key not in keys:
            keys.append(key)
        idx.append(keys.index(key))
    order = []
    grp = []
    for p in range(len(keys)):
        for t in range(len(segs)):
            if idx[t] == p:
                order.append(t)
                grp.append(p)
    return len(keys), order, grp


CHK1 = _flat_chunks(KBLK1, CHSP1)
CHK2 = _flat_chunks(KBLK2, CHSP2)
SEG1 = _flat_plan(KBLK1, CHSP1)
SEG2 = _flat_plan(KBLK2, CHSP2)
_assert_covered(SEG1, "SEG1")
_assert_covered(SEG2, "SEG2")
SSC1 = _scale_plan(SEG1)
SSC2 = _scale_plan(SEG2)


def _flat_tr(fls, chunks):
    nch = len(chunks)
    nt = 0
    for c in range(nch):
        nt += chunks[c][3]
    out = nl.ndarray((PAR, 2 * nt, TOK), dtype=nl.bfloat16, buffer=nl.sbuf)
    ci = 0
    t = 0
    for c in range(nch):
        flat = fls[c].reshape((TOK, 2, BPB * chunks[c][1]))
        for a in range(chunks[c][3]):
            for high in range(2):
                tp = nl.ndarray((PAR, TOK), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tp, data=flat[:, high, PAR * a:PAR * a + PAR])
                nisa.tensor_copy(dst=out[:, 2 * t + high, :], src=tp,
                                 engine=TRENG[ci % len(TRENG)])
                ci += 1
            t += 1
    return out


def _quant_layout(x, m, k, chunks):
    return _flat_tr(_act_quant(x, m, k, chunks), chunks)


def _fp4_dequant(q, sc, gw, mul_eng=nisa.engine.vector):
    nisa.tensor_scalar(dst=q.view(nl.uint32), data=q.view(nl.uint32),
                       op0=nl.bitwise_and, operand0=PAT_MASK2, engine=nisa.engine.vector)
    dq = q.view(nl.bfloat16).reshape((PAR, 2, gw))
    nisa.tensor_tensor(dst=dq, data1=dq,
                       data2=sc.view(nl.bfloat16).reshape((PAR, 1, gw)).broadcast(1, 2),
                       op=nl.multiply, engine=mul_eng)


def _dma_buf(nw, pdep):
    bf = []
    for _ in range(pdep):
        bf.append([nl.zeros((PAR, nw), dtype=nl.uint8, buffer=nl.sbuf),
                   nl.zeros((PAR, nw), dtype=nl.uint8, buffer=nl.sbuf)])
    return bf


def _alloc_db(nbg, nw, nj, pdep):
    if nbg > 1:
        bf = []
        for _ in range(nj // nbg):
            bf.append(_dma_buf(nbg * nw, pdep))
        return bf
    bf = []
    for _ in range(nj):
        bf.append(_dma_buf(nw, pdep))
    return bf


def _gemm_fp4(ws, ss, j, kbytes, kblk, n, nw, nbg, segs, ssc, alay, dbufs, outs, yacc,
              sceng, preng, ksp, pdep, d):
    ng = len(ws)
    widths = []
    for q in range(0, nw, NMAX):
        widths.append(min(NMAX, nw - q))
    nt = len(segs)
    nbt = n // nw
    assert nt % ksp == 0
    ntc = nt // ksp
    passes = []
    for gm in range(ng):
        for nb0 in range(0, nbt, nbg):
            passes.append((gm, nb0))
    npass = len(passes)
    npos = npass * ntc
    jnb = []
    jt = []
    ja = []
    for i in range(nbg):
        for kc in range(ksp):
            jnb.append(i * nw)
            jt.append(kc * ntc)
            ja.append(i)
    njob = len(jnb)
    merge = (nbg > 1)
    if merge:
        groups = []
        for kc in range(ksp):
            gl = []
            for i in range(nbg):
                gl.append(kc + ksp * i)
            groups.append(gl)
    else:
        groups = []
        for q in range(njob):
            groups.append([q])
    ngp = len(groups)
    gwid = []
    for gi in range(ngp):
        gwid.append(len(groups[gi]) * nw)
    jobgrp = []
    for q in range(njob):
        for gi in range(ngp):
            for qi in range(len(groups[gi])):
                if groups[gi][qi] == q:
                    jobgrp.append((gi, qi))
    assert len(jobgrp) == njob
    nib = []
    pre = []
    for _ in range(NDEP):
        gn = []
        for gi in range(ngp):
            gn.append(nl.ndarray((PAR, 2 * gwid[gi]), dtype=nl.uint8, buffer=nl.sbuf))
        nib.append(gn)
    for _ in range(pdep):
        gp = []
        for gi in range(ngp):
            gp.append(nl.ndarray((PAR, 2 * gwid[gi]), dtype=nl.uint16, buffer=nl.sbuf))
        pre.append(gp)
    if ksp == 1 and ssc[0] < nt:
        torder = ssc[1]
        tgrp = ssc[2]
    else:
        torder = []
        tgrp = []
        for t in range(ntc):
            torder.append(t)
            tgrp.append(t)
    nkey = tgrp[ntc - 1] + 1
    scb = []
    for _ in range(pdep):
        gs = []
        for gi in range(ngp):
            gs.append(nl.ndarray((PAR, gwid[gi]), dtype=nl.uint16, buffer=nl.sbuf))
        scb.append(gs)
    dsc = []
    dwt = []
    for gm in range(ng):
        hs = []
        hw = []
        for gi in range(ngp):
            gs = []
            gw = []
            q0 = groups[gi][0]
            kc = q0 % ksp
            for c in range(DDEP):
                if merge:
                    gs.append(dbufs[gm][kc][c][0])
                    gw.append(dbufs[gm][kc][c][1].view(nl.uint32))
                else:
                    gs.append(dbufs[gm][q0][c][0])
                    gw.append(dbufs[gm][q0][c][1].view(nl.uint32))
            hs.append(gs)
            hw.append(gw)
        dsc.append(hs)
        dwt.append(hw)
    accs = []
    for _ in range(npass):
        accs.append(None)
    assert 1 + STOFF[1] - STOFF[0] <= DDEP and 1 + STOFF[2] - STOFF[0] <= DDEP
    assert 1 + STOFF[3] - STOFF[1] <= NDEP
    assert 1 + STOFF[5] - STOFF[3] <= pdep
    assert npass * nkey <= pdep
    for g in range(min(STOFF), npos + max(STOFF)):
        for st in range(6):
            pos = g - STOFF[st]
            if pos < 0 or pos >= npos:
                continue
            p = pos // ntc
            lp = pos - p * ntc
            gm = passes[p][0]
            nb0 = passes[p][1]
            t = torder[lp]
            c = pos % pdep
            cd = pos % DDEP
            cn = pos % NDEP
            sb = (p * nkey + tgrp[lp]) % pdep
            sfill = (lp == 0 or tgrp[lp] != tgrp[lp - 1])
            if st == 0:
                for u in range(ksp if merge else njob):
                    nwd = nbg * nw if merge else nw
                    jn0 = nb0 * nw if merge else nb0 * nw + jnb[u]
                    sg = segs[jt[u] + t]
                    for si in range(len(sg)):
                        pp = sg[si][0]
                        off = sg[si][1]
                        kb0 = sg[si][2]
                        ln = sg[si][3]
                        if sfill:
                            nisa.dma_copy(
                                dst=dbufs[gm][u][cd][0][pp:pp + ln, :],
                                src=ss[gm].ap(pattern=[[kblk * n, 1], [n, ln], [1, nwd]],
                                              offset=kb0 * n + jn0 + d * kblk * n,
                                              scalar_offset=j,
                                              indirect_dim=0).squeeze_dim(0))
                        nisa.dma_copy(
                            dst=dbufs[gm][u][cd][1][pp:pp + ln, :],
                            src=ws[gm].ap(pattern=[[kbytes * n, 1], [BPB * n, ln],
                                                   [1, nwd]],
                                          offset=(BPB * kb0 + off) * n + jn0
                                          + d * kbytes * n,
                                          scalar_offset=j,
                                          indirect_dim=0).squeeze_dim(0))
            elif st == 1:
                for gi in range(ngp):
                    b16 = dwt[gm][gi][cd]
                    gwd = gwid[gi]
                    ix = t * ngp + gi
                    nisa.tensor_scalar(dst=nib[cn][gi][:, 0:gwd].view(nl.uint32), data=b16,
                                       op0=nl.bitwise_and, operand0=QUAD_NIB_MASK,
                                       engine=NIBENG[0][ix % len(NIBENG[0])])
                    nisa.tensor_scalar(dst=nib[cn][gi][:, gwd:2 * gwd].view(nl.uint32),
                                       data=b16,
                                       op0=nl.right_shift, operand0=HI_NIB_SHIFT,
                                       op1=nl.bitwise_and, operand1=QUAD_NIB_MASK,
                                       engine=NIBENG[1][ix % len(NIBENG[1])])
            elif st == 2:
                for gi in range(ngp):
                    if sfill:
                        nisa.tensor_scalar(dst=scb[sb][gi], data=dsc[gm][gi][cd],
                                           op0=nl.multiply, operand0=SC_MUL,
                                           op1=nl.add, operand1=SC_ADD,
                                           engine=sceng[(t * ngp + gi) % len(sceng)])
            elif st == 3:
                for gi in range(ngp):
                    nisa.tensor_scalar(dst=pre[c][gi], data=nib[cn][gi],
                                       op0=nl.multiply, operand0=NIB_MUL,
                                       engine=preng[(2 * gi) % len(preng)])
            elif st == 4:
                for gi in range(ngp):
                    _fp4_dequant(pre[c][gi], scb[sb][gi], gwid[gi],
                                 MULENG[(t * ngp + gi) % len(MULENG)])
            else:
                if accs[p] is None:
                    ac = []
                    for _ in range(nbg):
                        ga = []
                        for wi in range(len(widths)):
                            ga.append(nl.ndarray((TOK, widths[wi]), dtype=nl.float32,
                                                 buffer=nl.psum))
                        ac.append(ga)
                    accs[p] = ac
                for q in range(njob):
                    gi = jobgrp[q][0]
                    qw0 = jobgrp[q][1] * nw
                    dqf = pre[c][gi].view(nl.bfloat16)
                    for high in range(2):
                        base = high * gwid[gi] + qw0
                        for wi in range(len(widths)):
                            qw = base + wi * NMAX
                            nisa.nc_matmul(dst=accs[p][ja[q]][wi],
                                           stationary=alay[:, 2 * (jt[q] + t) + high, :],
                                           moving=dqf[:, qw:qw + widths[wi]],
                                           accumulate=(lp > 0 or high > 0 or jt[q] > 0))
        for p in range(npass):
            if g == p * ntc + ntc - 1 + STOFF[5]:
                gm = passes[p][0]
                nb0 = passes[p][1]
                di = 0
                for a in range(nbg):
                    for wi in range(len(widths)):
                        qw = wi * NMAX
                        jn0 = (nb0 + a) * nw
                        sl = slice(jn0 + qw, jn0 + qw + widths[wi])
                        deng = DRNENG[di % len(DRNENG)]
                        di += 1
                        if yacc is None:
                            nisa.tensor_copy(dst=outs[gm][:, sl], src=accs[p][a][wi],
                                             engine=deng)
                        else:
                            nisa.tensor_tensor(dst=yacc[:, sl], data1=yacc[:, sl],
                                               data2=accs[p][a][wi],
                                               op=nl.add, engine=nisa.engine.vector)


def _route_col(j, base, weff):
    wcol = nl.ndarray((TOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=wcol,
                  src=weff.ap(pattern=[[N_EXPERTS, TOK], [1, 1]], offset=base,
                              scalar_offset=j, indirect_dim=1))
    return wcol


def _expert_step(j, d, wcol, w1, s1, w3, s3, w2, s2, seg1, alay1, seg2, db, y):
    g = nl.ndarray((TOK, INTER), dtype=nl.bfloat16, buffer=nl.sbuf)
    u = nl.ndarray((TOK, INTER), dtype=nl.bfloat16, buffer=nl.sbuf)
    _gemm_fp4([w1, w3], [s1, s3], j, KB1, KBLK1, INTER, NW1, NBG1, seg1, SSC1, alay1,
              [db[0], db[1]], [g, u], None, SCENG1, PRENG1, KSP1, PDEP1, d)

    t = nl.ndarray((TOK, INTER), dtype=nl.bfloat16, buffer=nl.sbuf)
    _swiglu(g, u, wcol, t)

    _gemm_fp4([w2], [s2], j, KB2, KBLK2, DIM, NW2, NBG2, seg2, SSC2,
              _quant_layout(t, TOK, INTER, CHK2),
              [db[2]], None, y, SCENG2, PRENG2, KSP2, PDEP2, d)


def _pow2(e_int, m, n):
    idx = nl.minimum(nl.maximum(nl.add(e_int, EXP_BIAS), 0), 63)
    res = nl.ndarray((m, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=res.view(nl.uint32), data=idx,
                       op0=nl.multiply, operand0=float(1 << F32_MANT),
                       op1=nl.add, operand1=float((BF_BIAS - EXP_BIAS) << F32_MANT))
    return res


def _act_quant8(x_bf16, m, k):
    nb = k // ABLK
    x2 = nl.copy(x_bf16, dtype=nl.float32)
    xf = x2.reshape((m, nb, ABLK))
    amax = nl.maximum(nl.max(nl.abs(xf), axis=2), AMAX_FLOOR)

    ratio = nl.multiply(amax, 1.0 / FP8_MAX)
    e = nl.copy(nl.ceil(nl.multiply(nl.log(ratio), LOG2E)), dtype=nl.int32)
    s = _pow2(e, m, nb)
    inv = _pow2(nl.subtract(0, e), m, nb)

    s3 = s.reshape((m, nb, 1))
    i3 = inv.reshape((m, nb, 1))
    sb = nl.broadcast_to(s3, (m, nb, ABLK))
    ib = nl.broadcast_to(i3, (m, nb, ABLK))

    v = nl.multiply(nl.minimum(nl.maximum(nl.multiply(xf, ib), -FP8_MAX), FP8_MAX), 0.5)
    q = nl.copy(nl.copy(v, dtype=nl.float8_e4m3), dtype=nl.float32)
    dq = nl.multiply(nl.multiply(q, 2.0), sb)
    return nl.copy(dq, dtype=nl.bfloat16)


def _scale_rows(s, nt, nbtot, kb0=0, b0=0):
    row = s.shape[1]
    tiles = []
    for i in range(nt):
        raw = nl.ndarray((PAR, nbtot), dtype=nl.uint8, buffer=nl.sbuf)
        for r in range(PAR // ABLK):
            kb = kb0 + i * (PAR // ABLK) + r
            nisa.dma_copy(dst=raw[r * ABLK:(r + 1) * ABLK, :],
                          src=s.ap(pattern=[[0, ABLK], [1, nbtot]], offset=kb * row + b0))
        out = nl.ndarray((PAR, nbtot), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=out.view(nl.uint16), data=raw, op0=nl.multiply, operand0=SC_MUL8,
                           op1=nl.add, operand1=SC_ADD8, engine=nisa.engine.vector)
        tiles.append(out)
    return tiles


def _fp8_dequant(byte, sc, m, nw, nb):
    t = nl.ndarray((m, nw), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t, data=byte, op0=nl.multiply, operand0=BYTE_HI,
                       engine=nisa.engine.scalar)
    hi = nl.ndarray((m, nw), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=hi.view(nl.uint32), data=t.view(nl.uint32),
                       op0=nl.bitwise_and, operand0=E4M3_MAG2,
                       op1=nl.right_shift, operand1=4, engine=nisa.engine.vector)
    bits = nl.ndarray((m, nw), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=bits.view(nl.uint32), data=t.view(nl.uint32),
                              op0=nl.bitwise_and, operand0=E4M3_SGN2,
                              op1=nl.bitwise_or, operand1=hi.view(nl.uint32))
    wdq = nl.ndarray((m, nw), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=wdq.reshape((m, nb, ABLK)),
                       data1=bits.view(nl.bfloat16).reshape((m, nb, ABLK)),
                       data2=sc.reshape((m, nb, 1)).broadcast(2, ABLK),
                       op=nl.multiply, engine=FP8ENG)
    return wdq


def _gemm_fp8(a_t, w, s, k, n, k0=0, n0off=0):
    nt = k // PAR
    nbtot = n // ABLK
    sctiles = _scale_rows(s, nt, nbtot, k0 // ABLK, n0off // ABLK)
    out = nl.ndarray((TOK, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    for n0 in range(0, n, NMAX):
        nw = min(NMAX, n - n0)
        nb = nw // ABLK
        acc = nl.ndarray((TOK, nw), dtype=nl.float32, buffer=nl.psum)
        for i in range(nt):
            byte = nl.ndarray((PAR, nw), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.dma_copy(dst=byte, src=w[k0 + i * PAR:k0 + (i + 1) * PAR,
                                          n0off + n0:n0off + n0 + nw])
            wdq = _fp8_dequant(byte, sctiles[i][:, n0 // ABLK:n0 // ABLK + nb],
                               PAR, nw, nb)
            nisa.nc_matmul(dst=acc, stationary=a_t[:, i, :], moving=wdq,
                           accumulate=(i > 0))
        nisa.tensor_copy(dst=out[:, n0:n0 + nw], src=acc, engine=nisa.engine.scalar)
    return out


def _swiglu8(g_bf16, u_bf16):
    g = nl.minimum(nl.copy(g_bf16, dtype=nl.float32), SWIGLU_LIMIT)
    u = nl.minimum(nl.maximum(nl.copy(u_bf16, dtype=nl.float32), -SWIGLU_LIMIT),
                   SWIGLU_LIMIT)
    return nl.multiply(nl.silu(g), u)


def _shared_shard(h, sw1, ss1, sw2, ss2, sw3, ss3, c0, cw):
    hq = _act_quant8(h, TOK, DIM)
    hq_t = _transpose_stack(hq.reshape((TOK, DIM)), DIM)
    sg = _gemm_fp8(hq_t, sw1, ss1, DIM, cw, n0off=c0)
    su = _gemm_fp8(hq_t, sw3, ss3, DIM, cw, n0off=c0)
    st = nl.copy(_swiglu8(sg, su), dtype=nl.bfloat16)
    stq = _act_quant8(st, TOK, cw)
    return _gemm_fp8(_transpose_stack(stq.reshape((TOK, cw)), cw),
                     sw2, ss2, cw, DIM, k0=c0)


@nki.jit
def kernel(hidden, norm_w, gate_wt, gate_b, w1, s1, w3, s3, w2, s2,
           sw1, ss1, sw2, ss2, sw3, ss3, replica_group):
    out = nl.ndarray((TOK, DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    cc_src = nl.ndarray((TOK, DIM), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((TOK, DIM), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_dst")
    part = nl.ndarray((CORES * TOK, DIM), dtype=nl.float32, buffer=nl.shared_hbm, name="part")

    pid = nl.program_id(0)
    pbase = pid * CHH

    x = nl.load(hidden, dtype=nl.float32)
    gw = nl.load(norm_w, dtype=nl.float32)
    var = nl.multiply(nl.sum(nl.multiply(x, x), axis=1, keepdims=True), 1.0 / DIM)
    hf = nl.multiply(x, nl.rsqrt(nl.add(var, NORM_EPS)))
    h = nl.copy(nl.multiply(hf, nl.broadcast_to(gw, (TOK, DIM))), dtype=nl.bfloat16)

    h_t = _transpose_stack(h, DIM)
    logits = nl.ndarray((TOK, N_EXPERTS), dtype=nl.float32, buffer=nl.psum)
    for i in range(DIM // PAR):
        nisa.nc_matmul(dst=logits, stationary=h_t[:, i, :],
                       moving=nl.load(gate_wt[i * PAR:(i + 1) * PAR, :]),
                       accumulate=(i > 0))
    scores = nl.sqrt(nl.softplus(nl.copy(logits, dtype=nl.float32)))

    sel = nl.add(scores, nl.broadcast_to(nl.load(gate_b, dtype=nl.float32),
                                         (TOK, N_EXPERTS)))
    best = nl.ndarray((TOK, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=best, src=sel)
    sixth = nl.copy(best[:, TOPK - 1:TOPK])
    keep = nl.greater_equal(sel, nl.broadcast_to(sixth, (TOK, N_EXPERTS)))
    picked = nl.ndarray((TOK, N_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    picked[...] = nl.multiply(scores, nl.copy(keep, dtype=nl.float32))
    total = nl.add(nl.sum(picked, axis=1, keepdims=True), SUM_EPS)
    inv = nl.multiply(nl.reciprocal(total), ROUTE_SCALE)
    weff = nl.ndarray((TOK, N_EXPERTS), dtype=nl.float32, buffer=nl.private_hbm)
    nl.store(weff, value=nl.multiply(picked, nl.broadcast_to(inv, (TOK, N_EXPERTS))))

    y = nl.zeros((TOK, DIM), dtype=nl.float32, buffer=nl.sbuf)

    alay1 = _quant_layout(h, TOK, DIM, CHK1)
    dba = _alloc_db(NBG1, NW1, NBG1 * KSP1, DDEP)
    dbb = _alloc_db(NBG1, NW1, NBG1 * KSP1, DDEP)
    db2 = _alloc_db(NBG2, NW2, NBG2 * KSP2, DDEP)
    db = (dba, dbb, db2)

    def body(j):
        for d in range(UNROLL):
            _expert_step(j, d + pbase, _route_col(j, d + pbase, weff), w1, s1, w3, s3, w2, s2,
                         SEG1, alay1, SEG2, db, y)

    nl.fori_loop(0, CHH, body, step=UNROLL)

    shared = _shared_shard(h, sw1, ss1, sw2, ss2, sw3, ss3, SH_C0[pid], SH_CW[pid])
    y[...] = nl.add(y, nl.copy(shared, dtype=nl.float32))

    oth = (1 - pid) * TOK
    nisa.dma_copy(dst=part[pid * TOK:pid * TOK + TOK, :], src=y)
    nisa.core_barrier(data=part, cores=(0, 1))
    for c in range(0, DIM, CCH):
        nisa.dma_copy(dst=cc_src[:, c:c + CCH],
                      src=nl.add(y[:, c:c + CCH], nl.load(part[oth:oth + TOK, c:c + CCH])))

    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], replica_group=replica_group, op=nl.add)
    nl.store(out, value=nl.copy(nl.load(cc_dst), dtype=nl.bfloat16))
    return out
