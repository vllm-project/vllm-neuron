"""`layers.1.engram` on 4 ranks at long context: two cuts, two collectives, nothing replicated
twice. The `ngram` contraction axis of `wkv` is sharded on the value half — rank `r` owns
contraction columns `[1536r, 1536r+1536)`, i.e. 6 of the 24 hash columns — while the embedding
table and its e8m0 block scales are replicated.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

L = 8192
HC = 4
DIM = 5120
NCOL = 24
HEAD_DIM = 256
INF = NCOL * HEAD_DIM
XF = HC * DIM
KROWS = XF
VROWS = DIM

WORLD_SIZE = 4
N_CORES = 2

VK = INF // WORLD_SIZE
BS = 32
NBLK = INF // BS
VNBLK = VK // BS
KT = 128
NK = INF // KT
VNK = VK // KT
PT = 128
MB = 512
NBT = 1024
NMB = NBT // MB
SLAB = 512
NRH = 4
TPG = 8
TPGV = 6
HDIM = DIM // NRH
KSLAB = 256
MSUB = 8
STW = MSUB * KT
MSUBV = 10
STWV = MSUBV * KT
NBG = 12
NTT = L // PT
NB_TOT = KROWS // NBT
PWP = NTT // N_CORES
PWA = 6
PW = PWA * PWP
CDEP = 4
KSH = KROWS // WORLD_SIZE
NKB = KSH // NBT
NSLC = (L // KSLAB) // N_CORES

TBLR = 46596
NBS = HEAD_DIM // BS
ROWB = HEAD_DIM + NBS
RPP = 4
TGRP = KT * RPP
NTG = TBLR // TGRP
TTAIL = TBLR - NTG * TGRP
EPS = 1e-20
CLAMP = 1e-6
INV_DIM = 1.0 / DIM
DIM_RSQRT = 0.0139754248593737


def _e8m0_to_bf16(dst_i16, byte_i16, part, nblk):
    nisa.tensor_scalar(dst=dst_i16, data=byte_i16, op0=nl.multiply, operand0=128,
                       op1=nl.add, operand1=1024)


def _dequant(dst, byte_i16, scale_bf, part, nblk, scratch_a, scratch_b, flat=None):
    nf = nblk * BS
    if flat is None:
        flat = byte_i16.ap(pattern=[[nf, part], [1, nf]])
    nisa.tensor_scalar(dst=scratch_a, data=flat, op0=nl.bitwise_and, operand0=0x80,
                       op1=nl.left_shift, operand1=8)
    nisa.tensor_scalar(dst=scratch_b, data=flat, op0=nl.bitwise_and, operand0=0x7F,
                       op1=nl.left_shift, operand1=7)
    nisa.tensor_tensor(dst=scratch_b, data1=scratch_b, data2=scratch_a,
                       op=nl.bitwise_or)
    nisa.tensor_tensor(
        dst=dst,
        data1=scratch_b.view(nl.float16),
        data2=scale_bf.ap(pattern=[[nblk, part], [1, nblk], [0, BS]]),
        op=nl.multiply,
    )


def _transpose_into(tsb, src_sb, m, ntile, tpb, tpg):
    for g in nl.static_range(ntile // tpg):
        for j in nl.static_range(tpg):
            nisa.nc_transpose(dst=tpb[:, j, :],
                              data=src_sb[:, (g * tpg + j) * KT:(g * tpg + j + 1) * KT])
        nisa.tensor_copy(dst=tsb[:, g * tpg:(g + 1) * tpg, m * KT:(m + 1) * KT], src=tpb)


def _store_transposed(dst_hbm, ncol, col0, tsb, ntile, stw, nbg):
    for g in nl.static_range(ntile // nbg):
        nisa.dma_copy(
            dst=dst_hbm.ap(pattern=[[ncol, KT], [KT * ncol, nbg], [1, stw]],
                           offset=g * nbg * KT * ncol,
                           scalar_offset=col0, indirect_dim=1),
            src=tsb[:, g * nbg:(g + 1) * nbg, :],
        )


def _build_table(tbl, wsrc, ssrc):
    for g in nl.static_range(NTG):
        w = nl.ndarray((KT, RPP, HEAD_DIM), dtype=nl.uint8, buffer=nl.sbuf)
        s = nl.ndarray((KT, RPP, NBS), dtype=nl.uint8, buffer=nl.sbuf)
        stg = nl.ndarray((KT, RPP, ROWB), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(dst=w, src=wsrc.ap(
            pattern=[[RPP * HEAD_DIM, KT], [HEAD_DIM, RPP], [1, HEAD_DIM]],
            offset=g * TGRP * HEAD_DIM))
        nisa.dma_copy(dst=s, src=ssrc.ap(
            pattern=[[RPP * NBS, KT], [NBS, RPP], [1, NBS]],
            offset=g * TGRP * NBS))
        nisa.tensor_copy(dst=stg[:, :, 0:HEAD_DIM], src=w)
        nisa.tensor_copy(dst=stg[:, :, HEAD_DIM:ROWB], src=s)
        nisa.dma_copy(dst=tbl.ap(
            pattern=[[RPP * ROWB, KT], [1, RPP * ROWB]],
            offset=g * TGRP * ROWB), src=stg)

    wt = nl.ndarray((TTAIL, HEAD_DIM), dtype=nl.uint8, buffer=nl.sbuf)
    st = nl.ndarray((TTAIL, NBS), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.dma_copy(dst=wt, src=wsrc.ap(pattern=[[HEAD_DIM, TTAIL], [1, HEAD_DIM]],
                                      offset=NTG * TGRP * HEAD_DIM))
    nisa.dma_copy(dst=st, src=ssrc.ap(pattern=[[NBS, TTAIL], [1, NBS]],
                                      offset=NTG * TGRP * NBS))
    nisa.dma_copy(dst=tbl.ap(pattern=[[ROWB, TTAIL], [1, HEAD_DIM]],
                             offset=NTG * TGRP * ROWB), src=wt)
    nisa.dma_copy(dst=tbl.ap(pattern=[[ROWB, TTAIL], [1, NBS]],
                             offset=NTG * TGRP * ROWB + HEAD_DIM), src=st)


def _widen_scales(dst_hbm, scale_hbm, nrows, nblk, GB):
    for g0 in nl.static_range(nrows // (KT * GB)):
        rows = nl.ndarray((KT, GB, nblk), dtype=nl.int16, buffer=nl.sbuf)
        for g in nl.static_range(KT // BS):
            nisa.dma_copy(
                dst=rows[g * BS:(g + 1) * BS, :, :],
                src=scale_hbm.ap(
                    pattern=[[0, BS], [(KT // BS) * nblk, GB], [1, nblk]],
                    offset=(g0 * GB * (KT // BS) + g) * nblk),
            )
        shifted = nl.ndarray((KT, GB, nblk), dtype=nl.int16, buffer=nl.sbuf)
        _e8m0_to_bf16(shifted, rows, KT, nblk)
        nisa.dma_copy(
            dst=dst_hbm.ap(pattern=[[nblk, KT], [KT * nblk, GB], [1, nblk]],
                           offset=g0 * GB * KT * nblk),
            src=shifted.view(nl.bfloat16),
        )


@nki.jit
def kernel(input_0, input_1, shard_offset,
           layers_1_engram_k_weight, layers_1_engram_q_weight,
           layers_1_engram_wkv_key_scale, layers_1_engram_wkv_key_weight,
           layers_1_engram_wkv_value_scale, layers_1_engram_wkv_value_weight,
           layers_1_engram_embed_scale, layers_1_engram_embed_weight,
           replica_group):
    out = nl.ndarray((1, L, HC, DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    out2 = out.reshape((L, XF))

    cc_src = nl.ndarray((L, VROWS), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((L, VROWS), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="cc_dst")

    cc_gs = nl.ndarray((L, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_gs")
    cc_gd = nl.ndarray((WORLD_SIZE * L, 1), dtype=nl.float32, buffer=nl.shared_hbm,
                       name="cc_gd")

    h2 = input_0.reshape((L, XF))
    ids2 = input_1.reshape((L, NCOL))
    qw2 = layers_1_engram_q_weight.reshape((1, XF))
    kw2 = layers_1_engram_k_weight.reshape((1, XF))

    pid = nl.program_id(0)
    nprog = nl.num_programs()
    tok_lo = pid * (L // nprog)
    tok_hi = tok_lo + (L // nprog)
    slab_lo = pid * (L // SLAB // nprog)
    slab_hi = slab_lo + (L // SLAB // nprog)

    voff = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=voff, src=shard_offset)

    koff_f = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    koff = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=koff_f, src=voff)
    nisa.tensor_scalar(dst=koff_f, data=koff_f, op0=nl.multiply,
                       operand0=float(KSH) / float(VK), op1=nl.add, operand1=0.5)
    nisa.tensor_copy(dst=koff, src=koff_f)

    ksc = nl.ndarray((KROWS, NBLK), dtype=nl.bfloat16, buffer=nl.private_hbm)
    vsc = nl.ndarray((VROWS, VNBLK), dtype=nl.bfloat16, buffer=nl.private_hbm)
    embT = nl.ndarray((INF, L), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    vT = nl.ndarray((VK, VROWS), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    wp_hbm = nl.ndarray((1, XF), dtype=nl.float32, buffer=nl.private_hbm)
    tbl = nl.ndarray((TBLR, ROWB), dtype=nl.uint8, buffer=nl.private_hbm)

    _widen_scales(ksc, layers_1_engram_wkv_key_scale, KROWS, NBLK, 16)
    _widen_scales(vsc, layers_1_engram_wkv_value_scale, VROWS, VNBLK, 8)
    _build_table(tbl, layers_1_engram_embed_weight, layers_1_engram_embed_scale)

    qsb = nl.ndarray((1, XF), dtype=nl.bfloat16, buffer=nl.sbuf)
    ksb = nl.ndarray((1, XF), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=qsb, src=qw2)
    nisa.dma_copy(dst=ksb, src=kw2)
    wpsb = nl.ndarray((1, XF), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=wpsb, data1=qsb, data2=ksb, op=nl.multiply)
    nisa.dma_copy(dst=wp_hbm, src=wpsb)

    def embed_body(t0):
        tsb = nl.ndarray((KT, NK, STW), dtype=nl.bfloat16, buffer=nl.sbuf)
        tpb = nl.ndarray((KT, TPG, PT), dtype=nl.bfloat16, buffer=nl.psum)
        idx = nl.ndarray((PT, MSUB, NCOL), dtype=nl.uint32, buffer=nl.sbuf)
        qs = nl.ndarray((PT, NCOL, ROWB), dtype=nl.int16, buffer=nl.sbuf)
        esc = nl.ndarray((PT, NCOL, NBS), dtype=nl.int16, buffer=nl.sbuf)
        sa = nl.ndarray((PT, INF), dtype=nl.int16, buffer=nl.sbuf)
        sb = nl.ndarray((PT, INF), dtype=nl.int16, buffer=nl.sbuf)
        emb = nl.ndarray((PT, INF), dtype=nl.bfloat16, buffer=nl.sbuf)

        nisa.dma_copy(dst=idx, src=ids2.ap(
            pattern=[[NCOL, PT], [PT * NCOL, MSUB], [1, NCOL]],
            scalar_offset=t0, indirect_dim=0))

        for m in nl.static_range(MSUB):
            for j in nl.static_range(NCOL):
                nisa.dma_copy(
                    dst=qs[:, j, :],
                    src=tbl.ap(
                        pattern=[[ROWB, PT], [1, ROWB]],
                        vector_offset=idx[:, m, j:j + 1], indirect_dim=0),
                )

            _e8m0_to_bf16(esc, qs[:, :, HEAD_DIM:ROWB], PT, NBLK)
            _dequant(emb, None, esc.view(nl.bfloat16), PT, NBLK, sa, sb,
                     flat=qs.ap(pattern=[[NCOL * ROWB, PT], [ROWB, NCOL],
                                         [1, HEAD_DIM]]))
            _transpose_into(tsb, emb, m, NK, tpb, TPG)

        _store_transposed(embT, L, t0, tsb, NK, STW, NK)

    nl.fori_loop(tok_lo, tok_hi, embed_body, step=STW)
    if nprog > 1:
        nisa.core_barrier(embT, cores=(0, 1))

    def value_weight_body(r0):
        tsb = nl.ndarray((KT, VNK, STWV), dtype=nl.bfloat16, buffer=nl.sbuf)
        tpb = nl.ndarray((KT, TPGV, KT), dtype=nl.bfloat16, buffer=nl.psum)
        wq = nl.ndarray((KT, VK), dtype=nl.int16, buffer=nl.sbuf)
        wsc = nl.ndarray((KT, VNBLK), dtype=nl.bfloat16, buffer=nl.sbuf)
        sa = nl.ndarray((KT, VK), dtype=nl.int16, buffer=nl.sbuf)
        sb = nl.ndarray((KT, VK), dtype=nl.int16, buffer=nl.sbuf)
        wbf = nl.ndarray((KT, VK), dtype=nl.bfloat16, buffer=nl.sbuf)

        for m in nl.static_range(MSUBV):
            nisa.dma_copy(dst=wq, src=layers_1_engram_wkv_value_weight.ap(
                pattern=[[VK, KT], [1, VK]], offset=m * KT * VK,
                scalar_offset=r0, indirect_dim=0))
            nisa.dma_copy(dst=wsc, src=vsc.ap(
                pattern=[[VNBLK, KT], [1, VNBLK]], offset=m * KT * VNBLK,
                scalar_offset=r0, indirect_dim=0))
            _dequant(wbf, wq, wsc, KT, VNBLK, sa, sb)
            _transpose_into(tsb, wbf, m, VNK, tpb, TPGV)

        _store_transposed(vT, VROWS, r0, tsb, VNK, STWV, VNK)

    v_lo = pid * (VROWS // nprog)
    nl.fori_loop(v_lo, v_lo + (VROWS // nprog), value_weight_body, step=STWV)
    if nprog > 1:
        nisa.core_barrier(vT, cores=(0, 1))

    def value_out_body(n0):
        wsb = nl.ndarray((KT, VNK, VROWS), dtype=nl.bfloat16, buffer=nl.sbuf)
        for kb in nl.static_range(VNK):
            nisa.dma_copy(dst=wsb[:, kb, :],
                          src=vT.ap(pattern=[[VROWS, KT], [1, VROWS]],
                                    offset=kb * KT * VROWS,
                                    scalar_offset=n0, indirect_dim=1))
        esb = nl.ndarray((KT, VNK, SLAB), dtype=nl.bfloat16, buffer=nl.sbuf)
        val0 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        val1 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        vals = (val0, val1)

        base = slab_lo * (SLAB // PT)
        k = 0
        for sl in range(slab_lo, slab_hi):
            for kb in nl.static_range(VNK):
                nisa.dma_copy(dst=esb[:, kb, :],
                              src=embT.ap(pattern=[[L, KT], [1, SLAB]],
                                          offset=kb * KT * L + sl * SLAB,
                                          scalar_offset=voff, indirect_dim=0))
            for m in range(SLAB // PT):
                value = vals[k % 2]
                for nb in nl.static_range(VROWS // MB):
                    ps = nl.ndarray((PT, MB), dtype=nl.float32, buffer=nl.psum)
                    for kb in nl.static_range(VNK):
                        nisa.nc_matmul(
                            dst=ps,
                            stationary=esb[:, kb, m * PT:(m + 1) * PT],
                            moving=wsb[:, kb, nb * MB:(nb + 1) * MB],
                            accumulate=(kb > 0),
                        )
                    nisa.tensor_copy(dst=value[:, nb * MB:(nb + 1) * MB], src=ps,
                                     engine=nisa.engine.vector)

                if k > 0:
                    pv = vals[(k - 1) % 2]
                    pt0 = (base + k - 1) * PT
                    nisa.dma_copy(dst=cc_src[pt0:pt0 + PT, :], src=pv)
                k += 1

        lv = vals[(k - 1) % 2]
        lt0 = (base + k - 1) * PT
        nisa.dma_copy(dst=cc_src[lt0:lt0 + PT, :], src=lv)

    nl.fori_loop(0, VROWS, value_out_body, step=VROWS)

    pa3 = nl.ndarray((PT, NKB * 3, PWP), dtype=nl.float32, buffer=nl.sbuf)
    wsb = nl.ndarray((KT, NK, NBT), dtype=nl.bfloat16, buffer=nl.sbuf)
    wq = nl.ndarray((KT, INF), dtype=nl.int16, buffer=nl.sbuf)
    wsc = nl.ndarray((KT, NBLK), dtype=nl.bfloat16, buffer=nl.sbuf)
    wsa = nl.ndarray((KT, INF), dtype=nl.int16, buffer=nl.sbuf)
    wsx = nl.ndarray((KT, INF), dtype=nl.int16, buffer=nl.sbuf)
    wbf = nl.ndarray((KT, INF), dtype=nl.bfloat16, buffer=nl.sbuf)
    tpb = nl.ndarray((KT, TPG, KT), dtype=nl.bfloat16, buffer=nl.psum)
    wpk = nl.ndarray((KT, NBT), dtype=nl.float32, buffer=nl.sbuf)
    esb0 = nl.ndarray((KT, NK, KSLAB), dtype=nl.bfloat16, buffer=nl.sbuf)
    esb1 = nl.ndarray((KT, NK, KSLAB), dtype=nl.bfloat16, buffer=nl.sbuf)
    esbs = (esb0, esb1)
    hs = nl.ndarray((PT, KSLAB // PT, NBT), dtype=nl.bfloat16, buffer=nl.sbuf)
    sl0 = pid * NSLC

    for kblk in nl.static_range(NKB):
        kbase = kblk * NBT
        for rb in nl.static_range(NBT // KT):
            nisa.dma_copy(dst=wq, src=layers_1_engram_wkv_key_weight.ap(
                pattern=[[INF, KT], [1, INF]], offset=(kbase + rb * KT) * INF,
                scalar_offset=koff, indirect_dim=0))
            nisa.dma_copy(dst=wsc, src=ksc.ap(
                pattern=[[NBLK, KT], [1, NBLK]], offset=(kbase + rb * KT) * NBLK,
                scalar_offset=koff, indirect_dim=0))
            _dequant(wbf, wq, wsc, KT, NBLK, wsa, wsx)
            for g in nl.static_range(NK // TPG):
                for j in nl.static_range(TPG):
                    kb = g * TPG + j
                    nisa.nc_transpose(dst=tpb[:, j, :],
                                      data=wbf[:, kb * KT:(kb + 1) * KT])
                nisa.tensor_copy(dst=wsb[:, g * TPG:(g + 1) * TPG,
                                        rb * KT:(rb + 1) * KT], src=tpb)

        nisa.dma_copy(dst=wpk, src=wp_hbm.ap(pattern=[[0, KT], [1, NBT]],
                                             offset=kbase,
                                             scalar_offset=koff, indirect_dim=1))

        for kb in nl.static_range(NK):
            nisa.dma_copy(dst=esbs[0][:, kb, :],
                          src=embT[kb * KT:(kb + 1) * KT,
                                   sl0 * KSLAB:(sl0 + 1) * KSLAB])

        for sl in nl.static_range(0, NSLC):
            esb = esbs[sl % 2]
            if sl + 1 < NSLC:
                nxt = sl + 1
                eb = esbs[nxt % 2]
                for kb in nl.static_range(NK):
                    nisa.dma_copy(dst=eb[:, kb, :],
                                  src=embT[kb * KT:(kb + 1) * KT,
                                           (sl0 + nxt) * KSLAB:(sl0 + nxt + 1) * KSLAB])
            nisa.dma_copy(dst=hs, src=h2.ap(
                pattern=[[XF, PT], [PT * XF, KSLAB // PT], [1, NBT]],
                offset=(sl0 + sl) * KSLAB * XF + kbase,
                scalar_offset=koff, indirect_dim=1))

            for m in nl.static_range(KSLAB // PT):
                tile = sl * (KSLAB // PT) + m
                osb = nl.ndarray((PT, NBT), dtype=nl.bfloat16, buffer=nl.sbuf)
                ps = nl.ndarray((PT, NBT), dtype=nl.float32, buffer=nl.psum)
                for nb in nl.static_range(NMB):
                    for kb in nl.static_range(NK):
                        nisa.nc_matmul(
                            dst=ps[:, nb * MB:(nb + 1) * MB],
                            stationary=esb[:, kb, m * PT:(m + 1) * PT],
                            moving=wsb[:, kb, nb * MB:(nb + 1) * MB],
                            accumulate=(kb > 0),
                        )
                nisa.tensor_copy(dst=osb, src=ps, engine=nisa.engine.scalar)

                hsb = hs[:, m, :]
                dk = nl.ndarray((PT, NBT), dtype=nl.bfloat16, buffer=nl.sbuf)
                a0 = kblk * 3
                nisa.activation(dst=dk, op=nl.square, data=osb, reduce_op=nl.add,
                                reduce_res=pa3[:, a0, tile:tile + 1],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
                nisa.activation(dst=dk, op=nl.square, data=hsb, reduce_op=nl.add,
                                reduce_res=pa3[:, a0 + 1, tile:tile + 1],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
                prod = nl.ndarray((PT, NBT), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=prod, data1=hsb, data2=wpk, op=nl.multiply)
                nisa.tensor_tensor(dst=prod, data1=prod, data2=osb, op=nl.multiply)
                nisa.tensor_reduce(dst=pa3[:, a0 + 2, tile:tile + 1],
                                   op=nl.add, data=prod, axis=(1,))

    sk = nl.ndarray((PT, PWP), dtype=nl.float32, buffer=nl.sbuf)
    sh = nl.ndarray((PT, PWP), dtype=nl.float32, buffer=nl.sbuf)
    sd = nl.ndarray((PT, PWP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sk, data1=pa3[:, 0, :], data2=pa3[:, 3, :], op=nl.add)
    nisa.tensor_tensor(dst=sh, data1=pa3[:, 1, :], data2=pa3[:, 4, :], op=nl.add)
    nisa.tensor_tensor(dst=sd, data1=pa3[:, 2, :], data2=pa3[:, 5, :], op=nl.add)
    for b in nl.static_range(2, NKB):
        nisa.tensor_tensor(dst=sk, data1=sk, data2=pa3[:, 3 * b, :], op=nl.add)
        nisa.tensor_tensor(dst=sh, data1=sh, data2=pa3[:, 3 * b + 1, :], op=nl.add)
        nisa.tensor_tensor(dst=sd, data1=sd, data2=pa3[:, 3 * b + 2, :], op=nl.add)

    tmp = nl.ndarray((PT, PWP), dtype=nl.float32, buffer=nl.sbuf)
    rh = nl.ndarray((PT, PWP), dtype=nl.float32, buffer=nl.sbuf)
    rk = nl.ndarray((PT, PWP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=tmp, op=nl.sqrt, data=sh, scale=INV_DIM, bias=EPS)
    nisa.reciprocal(dst=rh, data=tmp)
    nisa.activation(dst=tmp, op=nl.sqrt, data=sk, scale=INV_DIM, bias=EPS)
    nisa.reciprocal(dst=rk, data=tmp)

    red = nl.ndarray((PT, PWP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=red, data1=sd, data2=rh, op=nl.multiply)
    nisa.tensor_tensor(dst=red, data1=red, data2=rk, op=nl.multiply)
    nisa.tensor_scalar(dst=red, data=red, op0=nl.multiply, operand0=DIM_RSQRT)

    mag = nl.ndarray((PT, PWP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=mag, op=nl.abs, data=red)
    nisa.tensor_scalar(dst=mag, data=mag, op0=nl.maximum, operand0=CLAMP)
    nisa.activation(dst=mag, op=nl.sqrt, data=mag)
    sgn = nl.ndarray((PT, PWP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sgn, op=nl.sign, data=red)
    nisa.tensor_tensor(dst=mag, data1=mag, data2=sgn, op=nl.multiply)
    gl = nl.ndarray((PT, PWP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=gl, op=nl.sigmoid, data=mag)

    base = pid * PWP
    nisa.dma_copy(dst=cc_gs.ap(pattern=[[1, PT], [PT, PWP]], offset=base * PT), src=gl)
    if nprog > 1:
        nisa.core_barrier(cc_src, cores=(0, 1))
        nisa.core_barrier(cc_gs, cores=(0, 1))
    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], op=nl.add, replica_group=replica_group)
    ncc.all_gather(srcs=[cc_gs], dsts=[cc_gd], replica_group=replica_group,
                   collective_dim=0)

    gt = nl.ndarray((PT, HC * PWP), dtype=nl.float32, buffer=nl.sbuf)
    for c in nl.static_range(HC):
        nisa.dma_copy(dst=gt[:, c * PWP:(c + 1) * PWP],
                      src=cc_gd.ap(pattern=[[1, PT], [PT, PWP]],
                                   offset=c * L + base * PT))

    def combine_body(n0):
        vv0 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        vv1 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        hh0 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        hh1 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        hh2 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        hh3 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        oo0 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        oo1 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        oo2 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        oo3 = nl.ndarray((PT, DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        vvs = (vv0, vv1)
        hhs = (hh0, hh1, hh2, hh3)
        oos = (oo0, oo1, oo2, oo3)

        base = pid * PWP
        NU = PWP * HC
        nisa.dma_copy(dst=vvs[0], src=cc_dst[base * PT:(base + 1) * PT, :])
        for u in range(CDEP - 1):
            nisa.dma_copy(dst=hhs[u], src=h2.ap(
                pattern=[[XF, PT], [1, DIM]],
                offset=(base + u // HC) * PT * XF + (u % HC) * DIM))

        for u in range(NU):
            i = u // HC
            c = u % HC
            if c == 0 and i + 1 < PWP:
                nisa.dma_copy(dst=vvs[(i + 1) % 2],
                              src=cc_dst[(base + i + 1) * PT:(base + i + 2) * PT, :])
            v = u + CDEP - 1
            if v < NU:
                nisa.dma_copy(dst=hhs[v % CDEP],
                              src=h2.ap(pattern=[[XF, PT], [1, DIM]],
                                        offset=(base + v // HC) * PT * XF
                                        + (v % HC) * DIM))
            nisa.scalar_tensor_tensor(dst=oos[u % CDEP], data=vvs[i % 2],
                                      op0=nl.multiply,
                                      operand0=gt[:, c * PWP + i:c * PWP + i + 1],
                                      op1=nl.add, operand1=hhs[u % CDEP])
            nisa.dma_copy(dst=out2.ap(pattern=[[XF, PT], [1, DIM]],
                                      offset=(base + i) * PT * XF + c * DIM),
                          src=oos[u % CDEP])

    nl.fori_loop(0, PT, combine_body, step=PT)

    return out
