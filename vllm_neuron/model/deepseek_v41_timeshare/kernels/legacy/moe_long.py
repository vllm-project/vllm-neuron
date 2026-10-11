"""`layers.1.ffn` at 8192 tokens on 4 ranks, rejoining through `nki.collectives`. Every rank runs
the `ffn_norm -> router` prologue replicated, the SwiGLU experts for its own 96-expert shard,
and a quarter of the shared expert, accumulating in float32; `ncc.all_reduce` sums the four.
The work is additionally split across the device's two physical cores.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

DIM = 5120
INTER = 2304
N_EXPERTS = 384
TOPK = 6
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 10.0
NORM_EPS = 1e-20
SUM_EPS = 1e-20
SP_CAP = 20.0
TOK = 128

PAR = 128
NMAX = 512
ABLK = 32
FP8_MAX = 448.0
AMAX_FLOOR = 1e-4

CH = 96
BPB = ABLK // 2
KBLK1 = DIM // ABLK
KBLK2 = INTER // ABLK
KB1 = DIM // 2
KB2 = INTER // 2
NW1 = 768
NW2 = 1024
NBG1 = 1
NBG2 = 1
GTOK = 4
KSP1 = 4
KSP2 = 3
MIDENG = nisa.engine.vector

SCENG1 = (nisa.engine.scalar,)
SCENG2 = (nisa.engine.scalar,)
DRNENG = (nisa.engine.scalar,)
TRENG = (nisa.engine.scalar,)
TRB = 8
PRENG1 = (nisa.engine.scalar,)
PRENG2 = (nisa.engine.scalar,)

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
PAIR_NIB_MASK = 0x0F0F
HI_NIB_SHIFT = 4
PDEP1 = 6
PDEP2 = 3
LEAD1 = 2
LEAD2 = 2
CHSP1 = True
CHSP2 = False
SC_MUL = 128.0
SC_ADD = 16128.0

INTER_PAD = 2560
SHARDS = 4
INTER_L = INTER_PAD // SHARDS
NT1 = DIM // PAR
NT2 = INTER_L // PAR
assert INTER_PAD >= INTER and INTER_PAD % (SHARDS * PAR) == 0

GSH = 4
WDEP = 3
WLEAD = 2
assert WDEP > WLEAD >= 0
PREDQ_W = 2560

BYTE_HI = 256.0
E4M3_MAG2 = 0x7F007F00
E4M3_SGN2 = 0x80008000 - (1 << 32)
SC_MUL8 = 128.0
SC_ADD8 = float(120 << 7)


def _act_quant(x_bf16, m, k, chunks, dst=None, koff=0, natout=None):
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
    if natout is not None:
        nisa.tensor_tensor(dst=natout, data1=q,
                           data2=nl.multiply(s, 2.0).reshape((m, nb, 1)).broadcast(2, ABLK),
                           op=nl.multiply, engine=nisa.engine.vector)
    q4 = q.reshape((m, nb, BPB, 2))
    fls = []
    for c in range(len(chunks)):
        kb0 = chunks[c][0]
        nk = chunks[c][1]
        s2 = nl.multiply(s[:, kb0:kb0 + nk], 2.0).reshape((m, 1, 1, nk))
        if dst is None:
            fl = nl.ndarray((m, 2, BPB, nk), dtype=nl.bfloat16, buffer=nl.sbuf)
            tgt = fl
        else:
            fl = dst[c]
            tgt = fl[:, :, :, koff:koff + nk]
        nisa.tensor_tensor(dst=tgt,
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
                       op1=nl.minimum, operand1=SWIGLU_LIMIT, engine=MIDENG)
    sg = nl.ndarray((m, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sg, op=nl.silu, data=g)
    nisa.scalar_tensor_tensor(dst=dst, data=sg, op0=nl.multiply, operand0=wcol,
                              op1=nl.multiply, operand1=u)


def _swiglu_sh(g_bf16, u_bf16, dst):
    m, k = g_bf16.shape
    g = nl.ndarray((m, k), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=g, data=g_bf16, op0=nl.minimum, operand0=SWIGLU_LIMIT,
                       engine=nisa.engine.vector)
    u = nl.ndarray((m, k), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=u, data=u_bf16, op0=nl.maximum, operand0=-SWIGLU_LIMIT,
                       op1=nl.minimum, operand1=SWIGLU_LIMIT, engine=MIDENG)
    sg = nl.ndarray((m, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sg, op=nl.silu, data=g)
    nisa.tensor_tensor(dst=dst, data1=sg, data2=u, op=nl.multiply,
                       engine=nisa.engine.vector)


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
    return idx


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
    srcs = []
    for c in range(nch):
        flat = fls[c].reshape((TOK, 2, BPB * chunks[c][1]))
        for a in range(chunks[c][3]):
            for high in range(2):
                srcs.append(flat[:, high, PAR * a:PAR * a + PAR])
    ci = 0
    i = 0
    n = len(srcs)
    while i < n:
        b = min(TRB, n - i)
        tp = nl.ndarray((PAR, b, TOK), dtype=nl.bfloat16, buffer=nl.psum)
        for k in range(b):
            nisa.nc_transpose(dst=tp[:, k, :], data=srcs[i + k])
        nisa.tensor_copy(dst=out[:, i:i + b, :], src=tp, engine=TRENG[ci % len(TRENG)])
        ci += 1
        i += b
    return out


def _quant_layout(x, m, k, chunks):
    return _flat_tr(_act_quant(x, m, k, chunks), chunks)


def _fp4_dequant(q, sc):
    nisa.tensor_scalar(dst=q.view(nl.uint32), data=q.view(nl.uint32),
                       op0=nl.bitwise_and, operand0=PAT_MASK2, engine=nisa.engine.vector)
    dq = q.view(nl.bfloat16)
    nisa.tensor_tensor(dst=dq, data1=dq, data2=sc, op=nl.multiply,
                       engine=nisa.engine.vector)
    return dq


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


def _gemm_fp4(ws, ss, j, kbytes, kblk, n, nw, nbg, segs, pid, alays, dbufs, outs, yacc,
              sceng, preng, ksp, pdep, d, lead=0, mid=None):
    ng = len(ws)
    widths = []
    for q in range(0, nw, NMAX):
        widths.append(min(NMAX, nw - q))
    nt = len(segs)
    nbt = n // nw
    assert nt % ksp == 0
    assert nbt % nbg == 0
    gtk = len(alays)
    assert 1 <= gtk <= GTOK
    assert pdep > lead
    assert 4 * ng * nbg * gtk * nw <= 16384
    ntc = nt // ksp
    ntc = nt // ksp
    njob = ng * nbg * ksp
    nib = []
    pre = []
    for _ in range(pdep):
        gn = []
        gp = []
        for _ in range(njob):
            gn.append([nl.ndarray((PAR, nw), dtype=nl.uint8, buffer=nl.sbuf),
                       nl.ndarray((PAR, nw), dtype=nl.uint8, buffer=nl.sbuf)])
            gp.append([nl.ndarray((PAR, nw), dtype=nl.uint16, buffer=nl.sbuf),
                       nl.ndarray((PAR, nw), dtype=nl.uint16, buffer=nl.sbuf)])
        nib.append(gn)
        pre.append(gp)
    plans = []
    for nb0 in range(0, nbt, nbg):
        jg = []
        jn = []
        jt = []
        ja = []
        for g in range(ng):
            for i in range(nbg):
                for kc in range(ksp):
                    jg.append(g)
                    jn.append((nb0 + i) * nw)
                    jt.append(kc * ntc)
                    ja.append(g * nbg + i)
        njob = len(jg)
        acc = []
        for _ in range(ng * nbg):
            gt = []
            for _ in range(gtk):
                ga = []
                for wi in range(len(widths)):
                    ga.append(nl.ndarray((TOK, widths[wi]), dtype=nl.float32, buffer=nl.psum))
                gt.append(ga)
            acc.append(gt)
        keys = []
        for pos in range(ntc):
            for q in range(njob):
                k = (jg[q], jn[q], pid[jt[q] + pos])
                if k not in keys:
                    keys.append(k)
        scb = []
        for _ in range(len(keys)):
            scb.append(nl.ndarray((PAR, nw), dtype=nl.uint16, buffer=nl.sbuf))
        sfl = []
        sbi = []
        filled = []
        for q in range(njob):
            sfl.append([])
            sbi.append([])
        for pos in range(ntc):
            for q in range(njob):
                k = (jg[q], jn[q], pid[jt[q] + pos])
                sbi[q].append(keys.index(k))
                sfl[q].append(k not in filled)
                if k not in filled:
                    filled.append(k)
        merge = (ng == 1 and nbg > 1)
        dsc = []
        dwt = []
        for q in range(njob):
            gs = []
            gw = []
            kc = q % ksp
            i = (q // ksp) % nbg
            for c in range(pdep):
                if merge:
                    w16 = dbufs[kc][c][1].view(nl.uint16)
                    gs.append(dbufs[kc][c][0][:, i * nw:(i + 1) * nw])
                    gw.append(w16[:, i * (nw // 2):(i + 1) * (nw // 2)])
                else:
                    gs.append(dbufs[q][c][0])
                    gw.append(dbufs[q][c][1].view(nl.uint16))
            dsc.append(gs)
            dwt.append(gw)
        plans.append((nb0, jg, jn, jt, ja, acc, scb, sfl, sbi, dsc, dwt, merge))
    ntot = len(plans) * ntc
    for ti in range(5 * (ntot + lead)):
        step = ti // 5
        st = ti % 5
        gpos = step - lead if st == 4 else step
        if gpos < 0 or gpos >= ntot:
            continue
        nbi = gpos // ntc
        pos = gpos % ntc
        nb0, jg, jn, jt, ja, acc, scb, sfl, sbi, dsc, dwt, merge = plans[nbi]
        t = pos
        c = gpos % pdep
        if st == 0:
            for u in range(ksp if merge else njob):
                nwd = nbg * nw if merge else nw
                jn0 = nb0 * nw if merge else jn[u]
                sg = segs[jt[u] + t]
                for si in range(len(sg)):
                    p = sg[si][0]
                    off = sg[si][1]
                    kb0 = sg[si][2]
                    ln = sg[si][3]
                    if sfl[u][pos]:
                        nisa.dma_copy(
                            dst=dbufs[u][c][0][p:p + ln, :],
                            src=ss[jg[u]].ap(pattern=[[kblk * n, 1], [n, ln], [1, nwd]],
                                             offset=kb0 * n + jn0 + d * kblk * n,
                                             scalar_offset=j,
                                             indirect_dim=0).squeeze_dim(0))
                    nisa.dma_copy(
                        dst=dbufs[u][c][1][p:p + ln, :],
                        src=ws[jg[u]].ap(pattern=[[kbytes * n, 1], [BPB * n, ln],
                                                  [1, nwd]],
                                         offset=(BPB * kb0 + off) * n + jn0
                                         + d * kbytes * n,
                                         scalar_offset=j,
                                         indirect_dim=0).squeeze_dim(0))
        elif st == 1:
            for q in range(njob):
                b16 = dwt[q][c]
                nisa.tensor_scalar(dst=nib[c][q][0].view(nl.uint16), data=b16,
                                   op0=nl.bitwise_and, operand0=PAIR_NIB_MASK,
                                   engine=nisa.engine.vector)
                nisa.tensor_scalar(dst=nib[c][q][1].view(nl.uint16), data=b16,
                                   op0=nl.right_shift, operand0=HI_NIB_SHIFT,
                                   op1=nl.bitwise_and, operand1=PAIR_NIB_MASK,
                                   engine=nisa.engine.vector)
        elif st == 2:
            for q in range(njob):
                if sfl[q][pos]:
                    nisa.tensor_scalar(dst=scb[sbi[q][pos]], data=dsc[q][c],
                                       op0=nl.multiply, operand0=SC_MUL,
                                       op1=nl.add, operand1=SC_ADD,
                                       engine=sceng[(t * njob + q) % len(sceng)])
        elif st == 3:
            for q in range(njob):
                for e in range(2):
                    nisa.tensor_scalar(dst=pre[c][q][e], data=nib[c][q][e],
                                       op0=nl.multiply, operand0=NIB_MUL,
                                       engine=preng[(2 * q + e) % len(preng)])
        else:
            wdq = []
            for q in range(njob):
                sc = scb[sbi[q][pos]].view(nl.bfloat16)
                wdq.append([_fp4_dequant(pre[c][q][0], sc),
                            _fp4_dequant(pre[c][q][1], sc)])
            for q in range(njob):
                for high in range(2):
                    for tt in range(gtk):
                        for wi in range(len(widths)):
                            qw = wi * NMAX
                            nisa.nc_matmul(
                                dst=acc[ja[q]][tt][wi],
                                stationary=alays[tt][:, 2 * (jt[q] + t) + high, :],
                                moving=wdq[q][high][:, qw:qw + widths[wi]],
                                accumulate=(t > 0 or high > 0 or jt[q] > 0))
        if st == 4 and pos == ntc - 1:
            di = 0
            for a in range(ng * nbg):
                for tt in range(gtk):
                    for wi in range(len(widths)):
                        qw = wi * NMAX
                        jn0 = (nb0 + a % nbg) * nw
                        sl = slice(jn0 + qw, jn0 + qw + widths[wi])
                        deng = DRNENG[di % len(DRNENG)]
                        di += 1
                        if yacc is None:
                            nisa.tensor_copy(dst=outs[a // nbg][tt][:, sl], src=acc[a][tt][wi],
                                             engine=deng)
                        else:
                            nisa.tensor_copy(dst=yacc[tt][:, sl], src=acc[a][tt][wi], engine=deng)
            if mid is not None:
                gs, us, wc, f2 = mid
                sl = slice(nb0 * nw, (nb0 + nbg) * nw)
                kw = nbg * nw
                tsl = []
                for tt in range(gtk):
                    tb = nl.ndarray((TOK, kw), dtype=nl.bfloat16, buffer=nl.sbuf)
                    _swiglu(gs[tt][:, sl], us[tt][:, sl], wc[tt], tb)
                    tsl.append(tb)
                for tt in range(gtk):
                    _act_quant(tsl[tt], TOK, kw, [(0, kw // ABLK, 0, kw * BPB // (ABLK * PAR))],
                               dst=f2[tt], koff=nb0 * nw // ABLK)


def _route_col(j, base, weff):
    wcol = nl.ndarray((TOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=wcol,
                  src=weff.ap(pattern=[[N_EXPERTS, TOK], [1, 1]], offset=base,
                              scalar_offset=j, indirect_dim=1))
    return wcol


def _expert_step(j, d, wcols, w1, s1, w3, s3, w2, s2, seg1, alay1, seg2, db, ys):
    gtk = len(alay1)
    g = []
    u = []
    for _ in range(gtk):
        g.append(nl.ndarray((TOK, INTER), dtype=nl.bfloat16, buffer=nl.sbuf))
        u.append(nl.ndarray((TOK, INTER), dtype=nl.bfloat16, buffer=nl.sbuf))
    fl2 = []
    for tt in range(gtk):
        rw = []
        for c in range(len(CHK2)):
            rw.append(nl.ndarray((TOK, 2, BPB, CHK2[c][1]), dtype=nl.bfloat16, buffer=nl.sbuf))
        fl2.append(rw)
    _gemm_fp4([w1], [s1], j, KB1, KBLK1, INTER, NW1, NBG1, seg1, SSC1, alay1,
              db[0], [g], None, SCENG1, PRENG1, KSP1, PDEP1, d, LEAD1)
    _gemm_fp4([w3], [s3], j, KB1, KBLK1, INTER, NW1, NBG1, seg1, SSC1, alay1,
              db[1], [u], None, SCENG1, PRENG1, KSP1, PDEP1, d, LEAD1,
              mid=(g, u, wcols, fl2))

    lay2 = []
    for tt in range(gtk):
        lay2.append(_flat_tr(fl2[tt], CHK2))

    _gemm_fp4([w2], [s2], j, KB2, KBLK2, DIM, NW2, NBG2, seg2, SSC2, lay2,
              db[2], None, ys, SCENG2, PRENG2, KSP2, PDEP2, d, LEAD2)


def _scale_rows(s, nt, nbtot):
    row = s.shape[1]
    tiles = []
    for i in range(nt):
        raw = nl.ndarray((PAR, nbtot), dtype=nl.uint8, buffer=nl.sbuf)
        for r in range(PAR // ABLK):
            nisa.dma_copy(dst=raw[r * ABLK:(r + 1) * ABLK, :],
                          src=s.ap(pattern=[[0, ABLK], [1, nbtot]],
                                   offset=(i * (PAR // ABLK) + r) * row))
        out = nl.ndarray((PAR, nbtot), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=out.view(nl.uint16), data=raw, op0=nl.multiply,
                           operand0=SC_MUL8, op1=nl.add, operand1=SC_ADD8,
                           engine=nisa.engine.vector)
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
                       op=nl.multiply, engine=nisa.engine.vector)
    return wdq


def _predq(w, s, k, n, dst):
    nt = k // PAR
    sct = _scale_rows(s, nt, n // ABLK)
    for i in range(nt):
        for n0 in range(0, n, PREDQ_W):
            nw = min(PREDQ_W, n - n0)
            byte = nl.ndarray((PAR, nw), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.dma_copy(dst=byte, src=w[i * PAR:(i + 1) * PAR, n0:n0 + nw])
            wdq = _fp8_dequant(byte, sct[i][:, n0 // ABLK:(n0 + nw) // ABLK],
                               PAR, nw, nw // ABLK)
            nisa.dma_copy(dst=dst[i * PAR:(i + 1) * PAR, n0:n0 + nw], src=wdq)


def _gemm_bf16(ats, w, k, n):
    nt = k // PAR
    nj = len(ats)
    outs = []
    for _ in range(nj):
        outs.append(nl.ndarray((TOK, n), dtype=nl.bfloat16, buffer=nl.sbuf))
    for n0 in range(0, n, NMAX):
        nw = min(NMAX, n - n0)
        accs = []
        for _ in range(nj):
            accs.append(nl.ndarray((TOK, nw), dtype=nl.float32, buffer=nl.psum))
        wb = []
        for _ in range(WDEP):
            wb.append(nl.ndarray((PAR, nw), dtype=nl.bfloat16, buffer=nl.sbuf))
        for i in range(nt + WLEAD):
            if i < nt:
                nisa.dma_copy(dst=wb[i % WDEP], src=w[i * PAR:(i + 1) * PAR, n0:n0 + nw])
            m = i - WLEAD
            if 0 <= m < nt:
                for jj in range(nj):
                    nisa.nc_matmul(dst=accs[jj], stationary=ats[jj][:, m, :],
                                   moving=wb[m % WDEP], accumulate=(m > 0))
        for jj in range(nj):
            nisa.tensor_copy(dst=outs[jj][:, n0:n0 + nw], src=accs[jj],
                             engine=nisa.engine.scalar)
    return outs


TOKENS = 8192
NTILE = TOKENS // TOK

P1U = 2
assert NTILE % P1U == 0

GRP = GTOK * TOK
NPGT = CH + (TOKENS * TOPK - CH) // GRP
CORES = 2
NPGC = (NPGT + CORES - 1) // CORES
NPG = CORES * NPGC
assert NPG >= NPGT

GPASS = (GTOK, GTOK // 2, 1)
NPASS = len(GPASS)
assert GPASS[0] == GTOK and GPASS[-1] >= 1
assert NPASS == 3
assert all(GPASS[i] > GPASS[i + 1] for i in range(NPASS - 1))
GTHR = tuple(list(GPASS[1:]) + [0])

UNROLLQ = (1, 1, 1)
assert len(UNROLLQ) == NPASS and all(NPGC % u == 0 for u in UNROLLQ)
GSTEPQ = tuple(CORES * u for u in UNROLLQ)
GOFFMAX = NPG + sum(GSTEPQ)
GORDN = GOFFMAX + CORES * NPGC
SHT = NTILE // CORES
assert SHT % GSH == 0
BIG = 1.0e7

PKW = 64
PKSTEP = TOK * PKW // GRP
PKR = ((NPG + 1 + PKSTEP - 1) // PKSTEP) * PKSTEP
assert TOK * PKW % GRP == 0 and PKR >= NPG + 1


def _tile3(t3d, t, n1, n2, tt=0):
    if isinstance(t, int):
        return t3d.ap(pattern=[[n2, n1], [1, n2]], offset=(tt + t) * n1 * n2)
    return t3d.ap(pattern=[[n1 * n2, 1], [n2, n1], [1, n2]], offset=tt * n1 * n2,
                  scalar_offset=t, indirect_dim=0).squeeze_dim(0)


def _pack_cols(pk3, g, col, ncol=GTOK):
    dst = nl.ndarray((TOK, ncol), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=dst,
                  src=pk3.ap(pattern=[[GRP * 2, 1], [2, TOK], [TOK * 2, ncol]],
                             offset=col,
                             scalar_offset=g, indirect_dim=0).squeeze_dim(0))
    return dst


def _grp_body(gtk, gptr, ctx, unroll):
    (pk3, eidh, gordh, aqf, w1, s1, w3, s3, w2, s2, db, partf, strash, tbase) = ctx
    for u in range(unroll):
        gv = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gv, src=gordh.ap(pattern=[[1, 1], [1, 1]], offset=0,
                                           scalar_offset=gptr, indirect_dim=0))
        e1 = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(dst=e1, src=eidh.ap(pattern=[[1, 1], [1, 1]], offset=0,
                                          scalar_offset=gv, indirect_dim=0))
        ys = []
        wcols = []
        alay1 = []
        sidx = []
        pkr = _pack_cols(pk3, gv, 0, gtk)
        pkw = _pack_cols(pk3, gv, 1, gtk)
        for tt in range(gtk):
            row = pkr[:, tt:tt + 1]
            wcols.append(pkw[:, tt:tt + 1])
            sidx.append(nl.copy(nl.minimum(nl.add(row, float(tbase * TOK)), strash),
                                dtype=nl.uint32))
            gi = nl.copy(nl.minimum(row, float(TOKENS - 1)), dtype=nl.uint32)
            fls = []
            for c in range(len(CHK1)):
                lc = 2 * BPB * CHK1[c][1]
                fl = nl.ndarray((TOK, 2, BPB, CHK1[c][1]), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=fl, src=aqf[c].ap(pattern=[[lc, TOK], [1, lc]],
                                                    vector_offset=gi, indirect_dim=0))
                fls.append(fl)
            ys.append(nl.ndarray((TOK, DIM), dtype=nl.bfloat16, buffer=nl.sbuf))
            alay1.append(_flat_tr(fls, CHK1))

        _expert_step(e1, 0, wcols, w1, s1, w3, s3, w2, s2, SEG1, alay1, SEG2, db, ys)

        for tt in range(gtk):
            nisa.dma_compute(
                dst=partf.ap(pattern=[[DIM, TOK], [1, DIM]],
                             vector_offset=sidx[tt], indirect_dim=0),
                srcs=[partf.ap(pattern=[[DIM, TOK], [1, DIM]],
                               vector_offset=sidx[tt], indirect_dim=0), ys[tt]],
                reduce_op=nl.add, oob_mode=nisa.oob_mode.skip)

        nisa.tensor_scalar(dst=gptr, data=gptr, op0=nl.add, operand0=CORES)


@nki.jit
def kernel(hidden, norm_w, gate_wt, gate_b, w1, s1, w3, s3, w2, s2,
           sw1, ss1, sw2, ss2, sw3, ss3, replica_group):
    out = nl.ndarray((TOKENS, DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    PT = NTILE + 1
    part = nl.ndarray((CORES * PT, TOK, DIM), dtype=nl.float32, buffer=nl.shared_hbm)
    partf = part.reshape((CORES * PT * TOK, DIM))
    cc_src = nl.ndarray((TOKENS, DIM), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((TOKENS, DIM), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_dst")

    pid = nl.program_id(0)
    tbase = pid * PT
    shlo = pid * SHT

    aql = []
    aqf = []
    for c in range(len(CHK1)):
        aql.append(nl.ndarray((NTILE, TOK, 2 * BPB * CHK1[c][1]), dtype=nl.bfloat16,
                              buffer=nl.shared_hbm, name=f"aql{c}"))
        aqf.append(aql[c].reshape((TOKENS, 2 * BPB * CHK1[c][1])))
    hqt = nl.ndarray((NTILE, PAR, NT1 * TOK), dtype=nl.bfloat16, buffer=nl.private_hbm)
    weffL = nl.ndarray((NTILE, TOK, CH), dtype=nl.float32, buffer=nl.shared_hbm, name="weffL")
    totx = nl.ndarray((CORES, CH), dtype=nl.float32, buffer=nl.shared_hbm, name="totx")
    pk3 = nl.ndarray((PKR, GRP, 2), dtype=nl.float32, buffer=nl.shared_hbm, name="pk3")
    eidh = nl.ndarray((NPG + 1, 1), dtype=nl.float32, buffer=nl.private_hbm)
    nth = nl.ndarray((NPG + 1, 1), dtype=nl.float32, buffer=nl.private_hbm)
    gordh = nl.ndarray((GORDN, 1), dtype=nl.float32, buffer=nl.private_hbm)
    d1 = nl.ndarray((DIM, INTER_L), dtype=nl.bfloat16, buffer=nl.private_hbm)
    d3 = nl.ndarray((DIM, INTER_L), dtype=nl.bfloat16, buffer=nl.private_hbm)
    d2 = nl.ndarray((INTER_L, DIM), dtype=nl.bfloat16, buffer=nl.private_hbm)

    dba = _alloc_db(NBG1, NW1, NBG1 * KSP1, PDEP1)
    dbb = _alloc_db(NBG1, NW1, NBG1 * KSP1, PDEP1)
    db2 = _alloc_db(NBG2, NW2, NBG2 * KSP2, PDEP2)
    db = (dba, dbb, db2)

    _predq(sw1, ss1, DIM, INTER_L, d1)
    _predq(sw3, ss3, DIM, INTER_L, d3)
    _predq(sw2, ss2, INTER_L, DIM, d2)

    zt = nl.ndarray((TOK, PKW, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(zt[:, :, 0], BIG)
    nisa.memset(zt[:, :, 1], 0.0)
    pkflat = pk3.reshape((PKR * GRP, 2))
    PKC = (PKR * GRP) // (TOK * PKW)
    assert PKC * TOK * PKW == PKR * GRP
    for i in range(PKC):
        nisa.dma_copy(dst=pkflat.ap(pattern=[[PKW * 2, TOK], [2, PKW], [1, 2]],
                                    offset=i * TOK * PKW * 2),
                      src=zt)
    ez = nl.ndarray((CH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ez, 0.0)
    for i in range(0, NPG + 1, CH):
        nz = min(CH, NPG + 1 - i)
        nisa.dma_copy(dst=eidh[i:i + nz, :], src=ez[0:nz, :])
        nisa.dma_copy(dst=nth[i:i + nz, :], src=ez[0:nz, :])
    gz = nl.ndarray((CH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(gz, float(NPG))
    for i in range(0, GORDN, CH):
        nz = min(CH, GORDN - i)
        nisa.dma_copy(dst=gordh[i:i + nz, :], src=gz[0:nz, :])

    tot = nl.zeros((1, CH), dtype=nl.float32, buffer=nl.sbuf)

    rit = nl.ndarray((1, N_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(rit, pattern=[[1, N_EXPERTS]], channel_multiplier=0)
    rkey = nl.ndarray((1, N_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    rkey[...] = nl.multiply(nl.add(rit, -float(N_EXPERTS)), -1.0)
    rkb = nl.ndarray((TOK, N_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    rkb[...] = nl.broadcast_to(rkey, (TOK, N_EXPERTS))
    gbb = nl.ndarray((TOK, N_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
    gbb[...] = nl.broadcast_to(nl.load(gate_b, dtype=nl.float32), (TOK, N_EXPERTS))
    oit = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(oit, pattern=[[1, 8]], channel_multiplier=0)
    ohk = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.sbuf)
    ohk[...] = nl.multiply(nl.add(oit, -float(TOPK - 1)), -1.0)

    gwb = nl.ndarray((TOK, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    gwb[...] = nl.broadcast_to(nl.load(norm_w, dtype=nl.bfloat16), (TOK, DIM))

    assert SHT % P1U == 0
    for t in range(shlo, shlo + SHT, P1U):
        for s in range(P1U):
            xb = nl.ndarray((TOK, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=xb, src=_tile3(hidden, t, TOK, DIM, s))

            hf = nl.ndarray((TOK, DIM), dtype=nl.float32, buffer=nl.sbuf)
            ss = nl.ndarray((TOK, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=hf, op=nl.square, data=xb, reduce_op=nl.add, reduce_res=ss,
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)
            var = nl.multiply(ss, 1.0 / DIM)
            nisa.activation(dst=hf, op=nl.copy, data=xb,
                            scale=nl.rsqrt(nl.add(var, NORM_EPS)))
            h = nl.multiply(hf, gwb, dtype=nl.bfloat16)
            hq = nl.ndarray((TOK, DIM // ABLK, ABLK), dtype=nl.bfloat16, buffer=nl.sbuf)
            aq = _act_quant(h, TOK, DIM, CHK1, natout=hq)
            for c in range(len(CHK1)):
                nisa.dma_copy(dst=_tile3(aql[c], t, TOK, 2 * BPB * CHK1[c][1], s),
                              src=aq[c].reshape((TOK, 2 * BPB * CHK1[c][1])))
            nisa.dma_copy(dst=_tile3(hqt, t, PAR, NT1 * TOK, s),
                          src=_transpose_stack(hq.reshape((TOK, DIM)), DIM))

            h_t = _transpose_stack(h, DIM)
            logits = nl.ndarray((TOK, N_EXPERTS), dtype=nl.float32, buffer=nl.psum)
            for i in range(DIM // PAR):
                nisa.nc_matmul(dst=logits, stationary=h_t[:, i, :],
                               moving=nl.load(gate_wt[i * PAR:(i + 1) * PAR, :]),
                               accumulate=(i > 0))
            lg = nl.copy(logits, dtype=nl.float32)
            scores = nl.sqrt(nl.maximum(nl.softplus(nl.minimum(lg, SP_CAP)), lg))

            sel = nl.add(scores, gbb)
            kth = nl.ndarray((TOK, 8), dtype=nl.float32, buffer=nl.sbuf)
            nisa.max8(dst=kth, src=sel)
            kb = nl.broadcast_to(nl.copy(kth[:, TOPK - 1:TOPK]), (TOK, N_EXPERTS))
            gt = nl.greater(sel, kb, dtype=nl.float32)
            cand = nl.multiply(nl.equal(sel, kb, dtype=nl.float32), rkb)
            tk = nl.ndarray((TOK, 8), dtype=nl.float32, buffer=nl.sbuf)
            nisa.max8(dst=tk, src=cand)
            ngt = nl.sum(gt, axis=1, keepdims=True)
            onh = nl.equal(nl.broadcast_to(ngt, (TOK, 8)),
                           nl.broadcast_to(ohk, (TOK, 8)), dtype=nl.float32)
            thr = nl.sum(nl.multiply(tk, onh), axis=1, keepdims=True)
            keep = nl.add(gt, nl.greater_equal(cand, nl.broadcast_to(thr, (TOK, N_EXPERTS)),
                                               dtype=nl.float32))
            picked = nl.ndarray((TOK, N_EXPERTS), dtype=nl.float32, buffer=nl.sbuf)
            picked[...] = nl.multiply(scores, keep)
            total = nl.add(nl.sum(picked, axis=1, keepdims=True), SUM_EPS)
            inv = nl.multiply(nl.reciprocal(total), ROUTE_SCALE)
            wl = nl.ndarray((TOK, CH), dtype=nl.float32, buffer=nl.sbuf)
            wl[...] = nl.multiply(picked[:, 0:CH], nl.broadcast_to(inv, (TOK, CH)))
            nisa.dma_copy(dst=_tile3(weffL, t, TOK, CH, s), src=wl)

            kf = nl.ndarray((TOK, CH), dtype=nl.float32, buffer=nl.sbuf)
            kf[...] = nl.greater(wl, 0.0)
            c = nl.ndarray((1, CH), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_partition_reduce(dst=c, op=nl.add, data=kf)
            tot[...] = nl.add(tot, c)


    yz = nl.ndarray((TOK, DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(yz, 0.0)
    for t in range(SHT):
        nisa.dma_copy(dst=part[tbase + (1 - pid) * SHT + t], src=yz)
    nisa.dma_copy(dst=part[tbase + NTILE], src=yz)

    assert SHT % GSH == 0
    for tb in range(shlo, shlo + SHT, GSH):
        ats = []
        for j in range(GSH):
            a = nl.ndarray((PAR, NT1, TOK), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=a, src=_tile3(hqt, tb, PAR, NT1 * TOK, j))
            ats.append(a)
        gs = _gemm_bf16(ats, d1, DIM, INTER_L)
        us = _gemm_bf16(ats, d3, DIM, INTER_L)
        tts = []
        for j in range(GSH):
            tb16 = nl.ndarray((TOK, INTER_L), dtype=nl.bfloat16, buffer=nl.sbuf)
            _swiglu_sh(gs[j], us[j], tb16)
            tq = nl.ndarray((TOK, INTER_L // ABLK, ABLK), dtype=nl.bfloat16, buffer=nl.sbuf)
            _act_quant(tb16, TOK, INTER_L, [], natout=tq)
            tts.append(_transpose_stack(tq.reshape((TOK, INTER_L)), INTER_L))
        ys = _gemm_bf16(tts, d2, INTER_L, DIM)
        for j in range(GSH):
            yf = nl.ndarray((TOK, DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=yf, src=ys[j], engine=nisa.engine.scalar)
            nisa.dma_copy(dst=_tile3(part, tb, TOK, DIM, j + tbase), src=yf)

    nisa.dma_copy(dst=totx[pid:pid + 1, :], src=tot)
    for c in range(len(CHK1)):
        nisa.core_barrier(data=aql[c], cores=(0, 1))
    nisa.core_barrier(data=weffL, cores=(0, 1))
    nisa.core_barrier(data=totx, cores=(0, 1))
    nisa.core_barrier(data=pk3, cores=(0, 1))
    tot[...] = nl.add(nl.load(totx[0:1, :]), nl.load(totx[1:2, :]))

    ngr = nl.ndarray((1, CH), dtype=nl.float32, buffer=nl.sbuf)
    ngr[...] = nl.floor(nl.multiply(nl.add(tot, GRP - 1.0), 1.0 / GRP))
    cum = nl.ndarray((1, CH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar_cumulative(dst=cum, src=ngr, op0=nl.multiply, imm0=1.0, op1=nl.add)
    gst = nl.ndarray((1, CH), dtype=nl.float32, buffer=nl.sbuf)
    gst[...] = nl.subtract(cum, ngr)
    base = nl.ndarray((1, CH), dtype=nl.float32, buffer=nl.sbuf)
    base[...] = nl.multiply(gst, float(GRP))

    tps = nl.ndarray((CH, 3), dtype=nl.float32, buffer=nl.psum)
    tps[:, 0:1] = nl.transpose(gst)
    tps[:, 1:2] = nl.transpose(ngr)
    tps[:, 2:3] = nl.transpose(tot)
    gstT = nl.copy(tps[:, 0:1])
    ngrT = nl.copy(tps[:, 1:2])
    totT = nl.copy(tps[:, 2:3])
    jid = nl.ndarray((CH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(jid, pattern=[[0, 1]], channel_multiplier=1)
    for k in range(TOKENS // GRP):
        sl = nl.ndarray((CH, 1), dtype=nl.float32, buffer=nl.sbuf)
        sl[...] = nl.minimum(nl.add(nl.add(gstT, float(k)),
                                    nl.multiply(nl.less_equal(ngrT, float(k)), BIG)),
                             float(NPG))
        slu = nl.copy(sl, dtype=nl.uint32)
        nisa.dma_copy(dst=eidh.ap(pattern=[[1, CH], [1, 1]],
                                  vector_offset=slu,
                                  indirect_dim=0),
                      src=jid, oob_mode=nisa.oob_mode.skip)
        ntv = nl.ndarray((CH, 1), dtype=nl.float32, buffer=nl.sbuf)
        ntv[...] = nl.minimum(
            nl.floor(nl.multiply(nl.add(nl.maximum(nl.add(totT, -float(k * GRP)), 0.0),
                                        float(TOK - 1)), 1.0 / TOK)),
            float(GTOK))
        nisa.dma_copy(dst=nth.ap(pattern=[[1, CH], [1, 1]],
                                 vector_offset=slu,
                                 indirect_dim=0),
                      src=ntv, oob_mode=nisa.oob_mode.skip)

    nts = nl.ndarray((1, NPG), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=nts, src=nth.reshape((1, NPG + 1))[0:1, 0:NPG])
    gmsk = []
    for q in range(NPASS):
        m = nl.ndarray((1, NPG), dtype=nl.float32, buffer=nl.sbuf)
        m[...] = nl.greater(nts, float(GTHR[q]))
        gmsk.append(m)
    gsel = []
    gcum = []
    for q in range(NPASS):
        a = nl.ndarray((1, NPG), dtype=nl.float32, buffer=nl.sbuf)
        a[...] = nl.copy(gmsk[q]) if q == 0 else nl.subtract(gmsk[q], gmsk[q - 1])
        gsel.append(a)
        c = nl.ndarray((1, NPG), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar_cumulative(dst=c, src=a, op0=nl.multiply, imm0=1.0, op1=nl.add)
        gcum.append(c)

    goff = []
    gom1 = []
    gcnt = []
    run0 = nl.zeros((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    for q in range(NPASS):
        o = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        o[...] = nl.minimum(nl.maximum(run0, 0.0), float(GOFFMAX))
        goff.append(o)
        d = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        d[...] = nl.add(o, -1.0)
        gom1.append(d)
        nq = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nq[...] = nl.copy(gcum[q][0:1, NPG - 1:NPG])
        gcnt.append(nq)
        nxt = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nxt[...] = nl.add(o, nl.maximum(
            nl.multiply(nl.floor(nl.multiply(nl.add(nq, float(GSTEPQ[q] - 1)),
                                             1.0 / GSTEPQ[q])),
                        float(GSTEPQ[q])),
            float(GSTEPQ[q])))
        run0 = nxt

    gpos = nl.ndarray((1, NPG), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(gpos, float(GORDN - 1))
    for q in range(NPASS):
        tq = nl.ndarray((1, NPG), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=tq, data=gcum[q], op0=nl.add, operand0=gom1[q])
        gpos[...] = nl.add(gpos, nl.multiply(gsel[q],
                                             nl.subtract(tq, float(GORDN - 1))))
    for h in range(0, NPG, PAR):
        gw = min(PAR, NPG - h)
        pt = nl.ndarray((gw, 1), dtype=nl.float32, buffer=nl.psum)
        pt[...] = nl.transpose(gpos[0:1, h:h + gw])
        pu = nl.copy(nl.minimum(nl.maximum(pt, 0.0), float(GORDN - 1)), dtype=nl.uint32)
        gcl = nl.ndarray((gw, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(gcl, pattern=[[0, 1]], channel_multiplier=1)
        gcv = nl.ndarray((gw, 1), dtype=nl.float32, buffer=nl.sbuf)
        gcv[...] = nl.add(gcl, float(h))
        nisa.dma_copy(dst=gordh.ap(pattern=[[1, gw], [1, 1]],
                                   vector_offset=pu, indirect_dim=0),
                      src=gcv, oob_mode=nisa.oob_mode.skip)

    ipat = nl.ndarray((PAR, PAR), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(ipat, pattern=[[1, PAR]], channel_multiplier=-1)
    ut = nl.ndarray((PAR, PAR), dtype=nl.bfloat16, buffer=nl.sbuf)
    ut[...] = nl.greater(ipat, 0.0)
    tokid = nl.ndarray((TOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(tokid, pattern=[[0, 1]], channel_multiplier=1)
    run = nl.zeros((1, CH), dtype=nl.float32, buffer=nl.sbuf)
    eord = nl.ndarray((1, CH), dtype=nl.float32, buffer=nl.sbuf)
    eord[...] = nl.multiply(nl.add(ipat[0:1, 0:CH], -float(CH)), -1.0)
    trash = nl.ndarray((TOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    trash[...] = nl.add(tokid, float(NPG * GRP))

    for t in range(shlo):
        wlc = nl.load(weffL[t], dtype=nl.float32)
        kfc = nl.ndarray((TOK, CH), dtype=nl.float32, buffer=nl.sbuf)
        kfc[...] = nl.greater(wlc, 0.0)
        cc = nl.ndarray((1, CH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_partition_reduce(dst=cc, op=nl.add, data=kfc)
        run[...] = nl.add(run, cc)

    for t in range(shlo, shlo + SHT):
        wl = nl.load(weffL[t], dtype=nl.float32)
        kb = nl.ndarray((TOK, CH), dtype=nl.bfloat16, buffer=nl.sbuf)
        kb[...] = nl.greater(wl, 0.0)
        kf = nl.ndarray((TOK, CH), dtype=nl.float32, buffer=nl.sbuf)
        kf[...] = nl.copy(kb, dtype=nl.float32)
        ps = nl.ndarray((TOK, CH), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ps, stationary=ut, moving=kb)
        off = nl.ndarray((1, CH), dtype=nl.float32, buffer=nl.sbuf)
        off[...] = nl.add(base, run)

        npos = nl.ndarray((TOK, CH), dtype=nl.float32, buffer=nl.sbuf)
        npos[...] = nl.subtract(nl.multiply(nl.less_equal(wl, 0.0), -BIG),
                                nl.add(ps, nl.broadcast_to(off, (TOK, CH))))
        bp = nl.ndarray((TOK, 8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.max8(dst=bp, src=npos)
        slq = nl.ndarray((TOK, 8), dtype=nl.float32, buffer=nl.sbuf)
        slq[...] = nl.minimum(nl.multiply(bp, -1.0), nl.broadcast_to(trash, (TOK, 8)))
        su = nl.copy(slq, dtype=nl.uint32)

        key = nl.ndarray((TOK, CH), dtype=nl.float32, buffer=nl.sbuf)
        key[...] = nl.add(nl.multiply(kf, nl.broadcast_to(eord, (TOK, CH))),
                          nl.multiply(wl, 0.5))
        bk = nl.ndarray((TOK, 8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.max8(dst=bk, src=key)
        wr = nl.multiply(nl.subtract(bk, nl.floor(bk)), 2.0)

        pay = nl.ndarray((TOK, 8, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(pay[:, :, 0], float(t * TOK))
        nisa.tensor_scalar(dst=pay[:, :, 0], data=pay[:, :, 0], op0=nl.add, operand0=tokid)
        pay[:, :, 1] = nl.copy(wr)
        for r in range(8):
            nisa.dma_copy(dst=pk3.reshape((PKR * GRP, 2)).ap(
                              pattern=[[2, TOK], [1, 2]],
                              vector_offset=su[:, r:r + 1], indirect_dim=0),
                          src=pay[:, r, :], oob_mode=nisa.oob_mode.skip)

        c = nl.ndarray((1, CH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_partition_reduce(dst=c, op=nl.add, data=kf)
        run[...] = nl.add(run, c)

    strash = nl.ndarray((TOK, 1), dtype=nl.float32, buffer=nl.sbuf)
    strash[...] = nl.add(tokid, float(tbase * TOK + TOKENS))

    gptrs = []
    nbregs = []
    for q in range(NPASS):
        nbf = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nbf[...] = nl.minimum(
            nl.maximum(nl.floor(nl.multiply(nl.add(gcnt[q], float(GSTEPQ[q] - 1)),
                                            1.0 / GSTEPQ[q])), 1.0),
            float(NPGC // UNROLLQ[q]))
        nbi = nl.copy(nbf, dtype=nl.int32)
        reg = nisa.register_alloc()
        nisa.register_load(reg, nbi)
        nbregs.append(reg)
        pf = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        pf[...] = nl.add(goff[q], float(pid))
        gp = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        gp[...] = nl.copy(pf, dtype=nl.int32)
        gptrs.append(gp)
    gctx = (pk3, eidh, gordh, aqf, w1, s1, w3, s3, w2, s2, db, partf, strash, tbase)

    nisa.core_barrier(data=pk3, cores=(0, 1))

    def pass0(i):
        _grp_body(GPASS[0], gptrs[0], gctx, UNROLLQ[0])

    def pass1(i):
        _grp_body(GPASS[1], gptrs[1], gctx, UNROLLQ[1])

    def pass2(i):
        _grp_body(GPASS[2], gptrs[2], gctx, UNROLLQ[2])

    nl.fori_loop(0, nbregs[0], pass0)
    nl.fori_loop(0, nbregs[1], pass1)
    nl.fori_loop(0, nbregs[2], pass2)

    CCT = NTILE // CORES
    clo = pid * CCT
    nisa.core_barrier(data=part, cores=(0, 1))
    for t in range(clo, clo + CCT):
        tsum = nl.ndarray((TOK, DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=tsum, data1=nl.load(part[t]), data2=nl.load(part[PT + t]),
                           op=nl.add, engine=nisa.engine.vector)
        nisa.dma_copy(dst=cc_src[t * TOK:(t + 1) * TOK, :], src=tsum)
    nisa.core_barrier(data=cc_src, cores=(0, 1))
    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], op=nl.add, replica_group=replica_group)
    for t in range(clo, clo + CCT):
        sl = slice(t * TOK, (t + 1) * TOK)
        nl.store(out[sl, :], value=nl.copy(nl.load(cc_dst[sl, :]), dtype=nl.bfloat16))
    return out, part
