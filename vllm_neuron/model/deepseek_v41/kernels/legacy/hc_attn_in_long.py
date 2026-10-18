import nki
import nki.language as nl
import nki.isa as nisa

PMAX = 128

KCHUNK = 128

KGROUP = 640

LGROUP = 2560

LPIPE = 5

CGROUP = 1280

YCHUNK = 1280

TBATCH = 64

NBUF = 2

COLLAPSE_SEED_ENGINE = nisa.scalar_engine

COLLAPSE_SEED_ACT = False

GPIPE = 4

TPIPE = 2

MPIPE = 2

DRAIN_SCALAR_BLOCKS = 0

DRAIN_SCALAR_GROUPS = 3

DRAIN_SPLIT = 1

STAT_ON_PE = True

STAT_SCALAR_GROUPS = 0

PSUM_PACK = 0

ACC_DTYPE = nl.bfloat16

GSEL_IN_PSUM = True

FUSE_MG = True

WSLOT_PAD = 512

FLAT_DIM_FULL = 20480

STATIC_DGE = True


def _static_dma(dst, src):
    if STATIC_DGE:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.none)
    else:
        nisa.dma_copy(dst=dst, src=src)


@nki.jit
def kernel(input_0, input_1, layers_0_hc_attn_in_fn):
    n_tok = input_0.shape[1]
    hc = input_0.shape[2]
    dim = input_0.shape[3]
    flat_dim = hc * dim
    mix_hc = layers_0_hc_attn_in_fn.shape[0]
    n_group = flat_dim // KGROUP
    n_sub = KGROUP // KCHUNK

    assert flat_dim % LGROUP == 0 and LGROUP % KGROUP == 0 and KGROUP % KCHUNK == 0
    assert (dim % LGROUP == 0 or LGROUP % dim == 0)
    assert LGROUP % CGROUP == 0 and dim % CGROUP == 0
    assert dim % YCHUNK == 0 and n_tok % PMAX == 0
    assert layers_0_hc_attn_in_fn.shape[1] == flat_dim
    lmul = LGROUP // KGROUP
    n_coll = LGROUP // CGROUP
    lpre = (LPIPE - 1) * lmul

    assert 0 <= STAT_SCALAR_GROUPS <= n_group
    stat_scalar_g = []
    for g in range(n_group):
        if (g * STAT_SCALAR_GROUPS) % n_group < STAT_SCALAR_GROUPS:
            stat_scalar_g.append(g)
    n_stat = len(stat_scalar_g)
    assert n_stat == STAT_SCALAR_GROUPS
    assert STAT_ON_PE or n_stat == n_group, "with the Gram off, every group must go to Scalar"

    y_out = nl.ndarray((1, n_tok, dim), dtype=input_0.dtype, buffer=nl.shared_hbm)
    mix_out = nl.ndarray((1, n_tok, mix_hc), dtype=nl.float32, buffer=nl.shared_hbm)
    meansq_out = nl.ndarray((1, n_tok, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    x2 = input_0.reshape((n_tok, flat_dim))
    pm2 = input_1.reshape((n_tok, hc))
    y2 = y_out.reshape((n_tok, dim))
    mix2 = mix_out.reshape((n_tok, mix_hc))
    meansq2 = meansq_out.reshape((n_tok, 1))

    wt = nl.ndarray((KCHUNK, n_group * n_sub * mix_hc), dtype=nl.bfloat16, buffer=nl.sbuf)
    for g in range(n_group):
        wg = nl.load(layers_0_hc_attn_in_fn[0:mix_hc, g * KGROUP:(g + 1) * KGROUP],
                     dtype=nl.bfloat16)
        wt_ps = nl.ndarray((KCHUNK, n_sub * mix_hc), dtype=nl.bfloat16, buffer=nl.psum)
        for i in range(n_sub):
            nisa.nc_transpose(dst=wt_ps[:, i * mix_hc:(i + 1) * mix_hc],
                              data=wg[:, i * KCHUNK:(i + 1) * KCHUNK])
        nisa.tensor_copy(dst=wt[:, g * n_sub * mix_hc:(g + 1) * n_sub * mix_hc], src=wt_ps,
                         engine=nisa.scalar_engine)

    WSLOT = WSLOT_PAD if FUSE_MG else KCHUNK + mix_hc
    assert WSLOT >= KCHUNK + mix_hc
    xtw = []
    if FUSE_MG:
        assert STAT_ON_PE and STAT_SCALAR_GROUPS == 0, "the fused matmul has no Scalar-stat path"
        assert DRAIN_SCALAR_BLOCKS == 0 and DRAIN_SPLIT == 1, "the drain writes one strided view"
        assert PSUM_PACK == 0
        for g in range(n_group):
            tw = nl.ndarray((PMAX, n_sub, WSLOT), dtype=nl.bfloat16, buffer=nl.sbuf)
            for i in range(n_sub):
                w0 = g * n_sub * mix_hc + i * mix_hc
                nisa.tensor_copy(dst=tw[:, i, KCHUNK:KCHUNK + mix_hc], src=wt[:, w0:w0 + mix_hc],
                                 engine=nisa.scalar_engine)
            xtw.append(tw)

    diag_pred = None
    if STAT_ON_PE:
        didx = nl.ndarray((PMAX, PMAX), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=didx, pattern=[[1, PMAX]], channel_multiplier=-1)
        diag_pred = nl.ndarray((PMAX, PMAX), dtype=nl.int8, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=diag_pred, data=didx, op0=nl.equal, operand0=0)

    assert (n_tok // PMAX) % TBATCH == 0
    assert TPIPE <= NBUF and MPIPE >= 1 and GPIPE >= 1
    for s in nl.affine_range(n_tok // (PMAX * TBATCH)):
        s0 = s * (PMAX * TBATCH)

        mixb = []
        ssb = []
        gramb = []
        accb = []
        ybfb = []
        mgb = []
        if PSUM_PACK == 2 and STAT_ON_PE:
            mg_w = PMAX + mix_hc
            assert NBUF * mg_w * 4 <= 2048, "one PSUM bank is 2 KiB per partition"
            mg = nl.ndarray((PMAX, NBUF * mg_w), dtype=nl.float32, buffer=nl.psum)
        elif PSUM_PACK == 1:
            mg_w = mix_hc
            assert NBUF * mg_w * 4 <= 2048, "one PSUM bank is 2 KiB per partition"
            mg = nl.ndarray((PMAX, NBUF * mg_w), dtype=nl.float32, buffer=nl.psum)
        for i in range(NBUF):
            if PSUM_PACK == 2 and STAT_ON_PE:
                gramb.append(mg[:, i * mg_w:i * mg_w + PMAX])
                mixb.append(mg[:, i * mg_w + PMAX:(i + 1) * mg_w])
            elif FUSE_MG:
                mgf = nl.ndarray((PMAX, KCHUNK + mix_hc), dtype=nl.float32, buffer=nl.psum)
                mgb.append(mgf)
                gramb.append(mgf[:, 0:KCHUNK])
                mixb.append(mgf[:, KCHUNK:KCHUNK + mix_hc])
            else:
                if PSUM_PACK == 1:
                    mixb.append(mg[:, i * mg_w:(i + 1) * mg_w])
                else:
                    mixb.append(nl.ndarray((PMAX, mix_hc), dtype=nl.float32,
                                           buffer=nl.psum))
                if STAT_ON_PE:
                    gramb.append(nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum))
            ssb.append(nl.ndarray((PMAX, n_group), dtype=nl.float32, buffer=nl.sbuf))
            accb.append(nl.ndarray((PMAX, dim), dtype=ACC_DTYPE, buffer=nl.sbuf))
            ybfb.append(nl.ndarray((PMAX, dim), dtype=input_0.dtype, buffer=nl.sbuf))

        for bf in range(TBATCH + (TPIPE - 1)):
         if bf < TBATCH:
          b = bf
          t0 = s0 + b * PMAX
          mix = mixb[b % NBUF]
          mgt = mgb[b % NBUF] if FUSE_MG else None
          ss = ssb[b % NBUF]
          gram = gramb[b % NBUF] if STAT_ON_PE else None
          acc = accb[b % NBUF]
          ybf = ybfb[b % NBUF]

          pm = nl.ndarray((PMAX, hc), dtype=input_1.dtype, buffer=nl.sbuf)
          _static_dma(pm, pm2[t0:t0 + PMAX, :])

          xb_q = []
          m_base = 0
          xtps_q = []
          xt_q = []
          ks = 0
          n_gram = 0
          for gf in range(n_group + lpre + (GPIPE - 1) + (MPIPE - 1)):
           if gf < n_group and gf % lmul == 0:
            xb_q.append(nl.load(x2[t0:t0 + PMAX, gf * KGROUP:gf * KGROUP + LGROUP]))

           gt = gf - lpre
           if 0 <= gt < n_group:
            xbt = xb_q[gt // lmul - m_base]
            o0 = (gt % lmul) * KGROUP

            if FUSE_MG:
                xt_ps_new = nl.ndarray((PMAX, n_sub, KCHUNK), dtype=nl.bfloat16, buffer=nl.psum)
                for i in range(n_sub):
                    nisa.nc_transpose(dst=xt_ps_new[:, i, :],
                                      data=xbt[:, o0 + i * KCHUNK:o0 + (i + 1) * KCHUNK])
            else:
                xt_ps_new = nl.ndarray((PMAX, KGROUP), dtype=nl.bfloat16, buffer=nl.psum)
                for i in range(n_sub):
                    nisa.nc_transpose(dst=xt_ps_new[:, i * KCHUNK:(i + 1) * KCHUNK],
                                      data=xbt[:, o0 + i * KCHUNK:o0 + (i + 1) * KCHUNK])
            xtps_q.append(xt_ps_new)

           g = gt - (GPIPE - 1)
           if 0 <= g < n_group:
            xt_ps = xtps_q.pop(0)
            if FUSE_MG:
                xt = xtw[g][:, :, 0:KCHUNK]
            else:
                xt = nl.ndarray((PMAX, KGROUP), dtype=nl.bfloat16, buffer=nl.sbuf)
            n_vec = (n_sub - DRAIN_SCALAR_BLOCKS) * KCHUNK
            if n_vec == KGROUP:
                if (g * DRAIN_SCALAR_GROUPS) % n_group < DRAIN_SCALAR_GROUPS:
                    nisa.tensor_copy(dst=xt, src=xt_ps, engine=nisa.scalar_engine)
                elif DRAIN_SPLIT == 1:
                    nisa.tensor_copy(dst=xt, src=xt_ps)
                else:
                    w = KGROUP // DRAIN_SPLIT
                    for q in range(DRAIN_SPLIT):
                        nisa.tensor_copy(dst=xt[:, q * w:(q + 1) * w],
                                         src=xt_ps[:, q * w:(q + 1) * w])
            else:
                if n_vec > 0:
                    nisa.tensor_copy(dst=xt[:, 0:n_vec], src=xt_ps[:, 0:n_vec],
                                     engine=nisa.vector_engine)
                nisa.tensor_copy(dst=xt[:, n_vec:KGROUP], src=xt_ps[:, n_vec:KGROUP],
                                 engine=nisa.scalar_engine)

            xt_q.append(xtw[g] if FUSE_MG else xt)

            m = g // lmul
            q = g % lmul
            assert m == m_base
            xbg = xb_q[0]
            for c in range(n_coll):
             if (c * lmul) // n_coll != q:
                 continue
             c0 = m * LGROUP + c * CGROUP
             xgc = xbg if n_coll == 1 else xbg[:, c * CGROUP:(c + 1) * CGROUP]
             j = c0 // dim
             f0 = c0 - j * dim
             dst = ybf[:, f0:f0 + CGROUP] if j == hc - 1 else acc[:, f0:f0 + CGROUP]
             if j == 0:
                 if COLLAPSE_SEED_ACT:
                     nisa.activation(dst=dst, op=nl.copy, data=xgc, scale=pm[:, j:j + 1])
                 else:
                     nisa.tensor_scalar(dst=dst, data=xgc, op0=nl.multiply,
                                        operand0=pm[:, j:j + 1], engine=COLLAPSE_SEED_ENGINE)
             else:
                 nisa.scalar_tensor_tensor(dst=dst, data=xgc,
                                           op0=nl.multiply, operand0=pm[:, j:j + 1],
                                           op1=nl.add, operand1=acc[:, f0:f0 + CGROUP])

            if g in stat_scalar_g:
                sink = nl.ndarray((PMAX, KGROUP), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=sink, op=nl.square,
                                data=xbg[:, q * KGROUP:(q + 1) * KGROUP],
                                reduce_op=nl.add, reduce_res=ss[:, ks:ks + 1],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
                ks = ks + 1

            if q == lmul - 1:
                xb_q.pop(0)
                m_base = m_base + 1

           gm = gt - (GPIPE - 1) - (MPIPE - 1)
           if gm >= 0:
            xtm = xt_q.pop(0)
            wg0 = gm * n_sub * mix_hc
            if FUSE_MG:
              for i in range(n_sub):
                nisa.nc_matmul(dst=mgt, stationary=xtm[:, i, 0:KCHUNK],
                               moving=xtm[:, i, 0:KCHUNK + mix_hc],
                               accumulate=(gm > 0 or i > 0))
            else:
             for i in range(n_sub):
                nisa.nc_matmul(dst=mix,
                               stationary=xtm[:, i * KCHUNK:(i + 1) * KCHUNK],
                               moving=wt[:, wg0 + i * mix_hc:wg0 + (i + 1) * mix_hc],
                               accumulate=(gm > 0 or i > 0))
                if STAT_ON_PE and gm not in stat_scalar_g:
                    nisa.nc_matmul(dst=gram,
                                   stationary=xtm[:, i * KCHUNK:(i + 1) * KCHUNK],
                                   moving=xtm[:, i * KCHUNK:(i + 1) * KCHUNK],
                                   accumulate=(n_gram > 0))
                    n_gram = n_gram + 1

         bt = bf - (TPIPE - 1)
         if bt >= 0:
          b = bt
          t0 = s0 + b * PMAX
          mix = mixb[b % NBUF]
          mgt = mgb[b % NBUF] if FUSE_MG else None
          ss = ssb[b % NBUF]
          gram = gramb[b % NBUF] if STAT_ON_PE else None
          ybf = ybfb[b % NBUF]

          if STAT_ON_PE:
              sumsq = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
              gsel = gram if GSEL_IN_PSUM else \
                  nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
              nisa.select_reduce(dst=gsel, predicate=diag_pred, on_true=gram, on_false=0.0,
                                 reduce_res=sumsq, reduce_op=nl.max,
                                 reduce_cmd=nisa.reduce_cmd.reset_reduce)
              if n_stat > 0:
                  sumsq = nl.add(sumsq, nl.sum(ss[:, 0:n_stat], axis=1, keepdims=True))
          else:
              sumsq = nl.sum(ss, axis=1, keepdims=True)

          meansq = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
          nisa.tensor_scalar(dst=meansq, data=sumsq, op0=nl.multiply,
                             operand0=1.0 / FLAT_DIM_FULL)

          mixes = nl.ndarray((PMAX, mix_hc), dtype=nl.float32, buffer=nl.sbuf)
          nisa.tensor_copy(dst=mixes, src=mix)

          _static_dma(mix2[t0:t0 + PMAX, :], mixes)
          _static_dma(meansq2[t0:t0 + PMAX, :], meansq)
          for f in range(dim // YCHUNK):
              f0 = f * YCHUNK
              _static_dma(y2[t0:t0 + PMAX, f0:f0 + YCHUNK], ybf[:, f0:f0 + YCHUNK])

    return y_out, mix_out, meansq_out
