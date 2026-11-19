"""Fused Qwen3.5 Gated DeltaNet decode step for one TP rank (NKI, LNC=1).

One kernel covers in_proj (q|k|v|z, a, b) -> depthwise conv + SiLU -> l2norm / gates ->
gated delta-rule recurrence -> gated RMSNorm -> this rank's out_proj slice. Reference:
``gdn_decode_torch.gdn_decode_torch`` (argument layouts converted by ``pack_inputs`` there).

Layouts (all chosen so every DMA moves >= 2 KB contiguous per partition):
  w_proj_p  [128, H/128, 2Kd+2Vd+2vh]  q|k|v|z|a|b in_proj, rows pre-permuted:
                                       w_p[p, h1, c] = W[h1*128+p, c]
  w_out     [Vd, H]                natural (8 KB rows)
  conv_state[B, Cc, K-1]           raw (pre-conv) history, oldest first; loaded token-major, PE-transposed
  rec_state [B, 128k, vh, 128v]    head inside the k row: one vh*512 B run per (token, k)
            [128k, B, vh, 128v]    QWEN35_GDN_KMAJOR=1: one run of (tokens x vh x 512 B) per k

SBUF compute layout is FEATURE-major ([128 features, tokens]): projection output feeds the
conv, l2norm and recurrence with no transposes, and per-channel / per-head parameters are
per-partition scalars. Per-token scalars are broadcast across partitions on the PE.

Recurrence per (token b, v-head h), state S [128k, 128v] fp32 (default QWEN35_GDN_REC=v2, see
_recurrence_v2): states of a super-block of tokens stay resident in SBUF;
    [kS | qS] = S^T [k q]                 (PE, bf16 S as stationary: one LDWEIGHTS per token and head)
    delta     = (v - eg*kS) * beta        (vector, whole super-block per head)
    S'        = eg*S + k (x) delta        (PE rank-1: token-major k of 32 tokens as stationary, delta
                                           masked to the token's row as moving; one DVE pass, in place)
    o         = eg*qS + (q.k) * delta     (no second pass over S')
x^T comes from a DMA transpose (QWEN35_GDN_XT=dma) when x is in HBM and B % 16 == 0.
"""

import os

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert

P = 128
HEAD = 128

# Trace-time switches (read once at import, like the GPT-OSS giga kernel's flags).
_S_BF16 = os.environ.get("QWEN35_GDN_SBF16", "1") == "1"  # bf16 copy of S for the matvec
_DGE = os.environ.get("QWEN35_GDN_DGE", "auto")  # auto | none | swdge | hwdge
_GROUP = int(os.environ.get("QWEN35_GDN_GROUP", "8"))  # tokens per recurrence group (state DMA unit)
_REC = os.environ.get("QWEN35_GDN_REC", "v2")  # v1: per-group recurrence | v2: phase-split, resident states
_R1 = os.environ.get("QWEN35_GDN_R1", "auto")  # v2 rank-1 form: auto | dmask | kmask
_CPY = os.environ.get("QWEN35_GDN_CPY", "act")  # v2 bf16 state copy engine: act | dve | alt (per group)
_RES_KB = int(os.environ.get("QWEN35_GDN_RES_KB", "64"))  # v2 resident fp32 state bytes per partition
_XT = os.environ.get("QWEN35_GDN_XT", "dma")  # x -> x^T in gdn_decode_tkg: dma (DMA transpose) | pe
_IL = os.environ.get("QWEN35_GDN_IL", "1") == "1"  # v2 dmask: interleave rank-1 matmuls over PE row tiles
_KMAJOR = os.environ.get("QWEN35_GDN_KMAJOR", "0") == "1"  # rec_state / new_rec k-major [128k, B, vh, 128v]
_KLOAD = int(os.environ.get("QWEN35_GDN_KLOAD", "0"))  # v2 k-major: tokens per state-load DMA (0: super-block)
_SDGE = os.environ.get("QWEN35_GDN_SDGE", "same")  # v2 state DMAs: same (as _dma) | split (loads SP, stores ACT)
MCH = 512  # fp32 PSUM bank width
_dma_rr = [0]


def _dma(dst, src):
    """Large-transfer DMA with the configured descriptor mode."""
    if _DGE == "auto":
        nisa.dma_copy(dst=dst, src=src)
        return
    mode = getattr(nisa.dge_mode, _DGE)
    if _DGE == "hwdge":
        eng = nisa.engine.sync if _dma_rr[0] % 2 == 0 else nisa.engine.scalar
        _dma_rr[0] += 1
        nisa.dma_copy(dst=dst, src=src, dge_mode=mode, engine=eng)
    else:
        nisa.dma_copy(dst=dst, src=src, dge_mode=mode)


def _state_ap(t, row, n, vh):
    """Access pattern of n consecutive tokens (from token row `row`) of a recurrent-state tensor, as
    [128k, n, vh*128v] (SBUF layout). Token-major t [Bt, 128k, vh, 128v]: one vh*512-byte run per
    (token, k). K-major t [128k, Bt, vh, 128v]: one n*vh*512-byte run per k."""
    VP = vh * P
    if _KMAJOR:
        Bt = t.shape[1]
        return t.ap(pattern=[[Bt * VP, P], [VP, n], [1, VP]], offset=row * VP)
    return t.ap(pattern=[[VP, P], [P * VP, n], [1, VP]], offset=row * P * VP)


def _sdma(dst, src, i):
    """State DMA (v2). split: loads / stores on the two HWDGE rings alternately."""
    if _SDGE == "split":
        eng = nisa.engine.sync if i % 2 == 0 else nisa.engine.scalar
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=eng)
    else:
        _dma(dst=dst, src=src)


@nki.jit
def gdn_decode_tkg(
    x,  # [B, H] bf16
    w_proj_p,  # [128, H/128, 2*Kd + 2*Vd + 2*vh] bf16
    conv_w,  # [Cc, K] bf16
    conv_state,  # [B, Cc, K-1] bf16
    a_log,  # [vh] fp32
    dt_bias,  # [vh] fp32
    norm_w,  # [128] fp32
    rec_state,  # [B, 128, vh, 128] fp32 ([128, B, vh, 128] with QWEN35_GDN_KMAJOR=1)
    w_out,  # [Vd, H] bf16 (or [Vd, H + extra] with folded extra output columns)
    rec_row0=0,  # first row of rec_state this call uses (row-blocked callers pass the full state tensor)
    inplace=False,  # write the new conv / recurrent states over the inputs (same rows) instead of new tensors
):
    B, H = x.shape
    x_sb = None  # DMA transpose: the body transposes x straight from HBM (needs 32-byte output rows)
    if not (_XT == "dma" and B % 16 == 0):
        x_sb = nl.ndarray((B, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        _dma(dst=x_sb, src=x)
    o_sb, new_conv, new_rec = gdn_decode_body(
        x_sb, w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out, x_hbm=x, rec_row0=rec_row0,
        inplace=inplace,
    )
    out = nl.ndarray((B, w_out.shape[1]), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    _dma(dst=out, src=o_sb)
    return out, new_conv, new_rec


def gdn_load_weights(w_proj_p, w_out, vh):
    """SBUF copies of the GDN projection and out-projection weights (for gdn_decode_body's
    w_proj_sb / w_out_sb when one layer runs the body on several row blocks)."""
    H1, NT = w_proj_p.shape[1], w_proj_p.shape[2]
    N_out = w_out.shape[1]
    w_all = nl.ndarray((P, H1, NT), dtype=nl.bfloat16, buffer=nl.sbuf)
    _dma(dst=w_all, src=w_proj_p)
    wo = nl.ndarray((P, vh, N_out), dtype=nl.bfloat16, buffer=nl.sbuf)
    _dma(dst=wo, src=w_out.ap(pattern=[[N_out, P], [P * N_out, vh], [1, N_out]], offset=0))
    return w_all, wo


def gdn_decode_body(
    x_sb,  # [B, H] bf16 SBUF (normed tokens)
    w_proj_p,  # [128, H/128, 2*Kd + 2*Vd + 2*vh] bf16
    conv_w,  # [Cc, K] bf16
    conv_state,  # [B, Cc, K-1] bf16
    a_log,  # [vh] fp32
    dt_bias,  # [vh] fp32
    norm_w,  # [128] fp32
    rec_state,  # [B, 128, vh, 128] fp32 ([128, B, vh, 128] with QWEN35_GDN_KMAJOR=1)
    w_out,  # [Vd, H] bf16 (or [Vd, H + extra] with folded extra output columns)
    x_hbm=None,  # [B, H] bf16 HBM: used (DMA transpose) when x_sb is None
    rec_row0=0,  # rows rec_row0 .. rec_row0 + B of rec_state are this call's states
    w_proj_sb=None,  # optional SBUF copy of w_proj_p (callers running several row blocks load it once)
    w_out_sb=None,  # optional SBUF [128, vh, N_out] copy of w_out
    inplace=False,  # new states overwrite conv_state / rec_state rows rec_row0.. (multi-step decode carries them)
):
    B, H = x_sb.shape if x_sb is not None else x_hbm.shape
    vh = a_log.shape[0]
    Vd = vh * HEAD
    NT = w_proj_p.shape[2]  # all projection columns
    NW = NT - 2 * vh  # q|k|v|z columns
    Kd = (NW - 2 * Vd) // 2
    kh = Kd // HEAD
    gqa = vh // kh
    Cc = 2 * Kd + Vd
    K = conv_w.shape[1]
    Kh = K - 1
    H1 = H // P
    n_ct = NW // P
    n_conv_t = Cc // P
    q_t0, k_t0, v_t0, z_t0 = 0, kh, 2 * kh, 2 * kh + vh
    G = min(_GROUP, B)
    eps = 1e-6
    kernel_assert(B <= P and B % G == 0, f"B must be <= 128 and a multiple of {G}, got {B}")
    kernel_assert(H % P == 0 and Kd % P == 0, "H and key dim must be multiples of 128")
    if _KMAJOR:
        kernel_assert(
            rec_state.shape[0] == HEAD and tuple(rec_state.shape[2:]) == (vh, HEAD)
            and rec_state.shape[1] >= rec_row0 + B,
            "k-major rec_state must be [128, >= rec_row0 + B, vh, 128]",
        )
    else:
        kernel_assert(
            tuple(rec_state.shape[1:]) == (HEAD, vh, HEAD) and rec_state.shape[0] >= rec_row0 + B,
            "rec_state must be [>= rec_row0 + B, 128, vh, 128]",
        )
    kernel_assert(w_proj_p.shape[0] == P and w_proj_p.shape[1] == H1, "w_proj_p must be [128, H/128, cols]")

    N_out = w_out.shape[1]  # H, or H + extra folded columns (low-concurrency router fold)
    if inplace:  # every state row is read (into SBUF) before the update computed from it is stored back
        kernel_assert(conv_state.shape[0] == B, "in-place GDN states: conv_state must be this call's B rows")
        new_conv, new_rec, new_row0 = conv_state, rec_state, rec_row0
    else:
        new_conv = nl.ndarray((B, Cc, Kh), dtype=conv_state.dtype, buffer=nl.shared_hbm)
        new_rec = nl.ndarray(
            (HEAD, B, vh, HEAD) if _KMAJOR else (B, HEAD, vh, HEAD), dtype=nl.float32, buffer=nl.shared_hbm
        )
        new_row0 = 0

    # ---------------- constants ----------------
    ones_col = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_col, 1.0)
    ones_row = nl.ndarray((1, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_row, 1.0)
    iota_g = nl.ndarray((G, G), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(iota_g, pattern=[[1, G]], offset=0, channel_multiplier=-1)
    ident = nl.ndarray((G, G), dtype=nl.float32, buffer=nl.sbuf)  # ident[p, b] = (p == b)
    nisa.tensor_scalar(dst=ident, data=iota_g, op0=nl.equal, operand0=0)
    iota_h = nl.ndarray((vh, vh, P), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(iota_h, pattern=[[1, vh], [0, P]], offset=0, channel_multiplier=-1)
    sel_h = nl.ndarray((vh, vh, P), dtype=nl.float32, buffer=nl.sbuf)  # sel_h[p, h, m] = (p == h)
    nisa.tensor_scalar(dst=sel_h, data=iota_h, op0=nl.equal, operand0=0)

    # ---------------- x^T [128, H1, B] ----------------
    xT = nl.ndarray((P, H1, B), dtype=nl.bfloat16, buffer=nl.sbuf)
    if x_sb is None:  # DMA-engine transpose from HBM (no PE / DVE work)
        nisa.dma_transpose(dst=xT, src=x_hbm.ap(pattern=[[H, B], [P, H1], [1, P]], offset=0), axes=(2, 1, 0))
    else:
        for h1 in range(H1):
            tp = nl.ndarray((P, B), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp, data=x_sb[:, h1 * P : (h1 + 1) * P])
            nisa.tensor_copy(dst=xT[:, h1, :], src=tp)

    # ---------------- projections ----------------
    # x^T[:, h1] is the stationary (one LDWEIGHTS per h1, reused by every 512-wide weight
    # chunk as moving): every distinct stationary has a fixed cost regardless of its size, so
    # the weights-stationary form (one load per 128x128 weight tile) is PE-bound at small B.
    # Output is token-major [B, NW]; one PE transpose per 128 columns makes it feature-major.
    proj = nl.ndarray((P, n_ct, B), dtype=nl.bfloat16, buffer=nl.sbuf)
    SLAB = 8
    MCH = 512
    n_mch = (NT + MCH - 1) // MCH
    ps_tok = [nl.ndarray((B, MCH), dtype=nl.float32, buffer=nl.psum) for _ in range(n_mch)]
    for s0 in range(0, H1, SLAB):
        if w_proj_sb is not None:  # loaded once by the caller for several row blocks
            w_sb = w_proj_sb[:, s0 : s0 + SLAB, :]
        else:
            w_sb = nl.ndarray((P, SLAB, NT), dtype=nl.bfloat16, buffer=nl.sbuf)
            _dma(dst=w_sb, src=w_proj_p[:, s0 : s0 + SLAB, :])
        for hs in range(SLAB):
            for mc in range(n_mch):
                c0, c1 = mc * MCH, min(NT, (mc + 1) * MCH)
                nisa.nc_matmul(ps_tok[mc][:, 0 : c1 - c0], xT[:, s0 + hs, :], w_sb[:, hs, c0:c1])
    tok = nl.ndarray((B, NT), dtype=nl.bfloat16, buffer=nl.sbuf)
    for mc in range(n_mch):
        c0, c1 = mc * MCH, min(NT, (mc + 1) * MCH)
        nisa.tensor_copy(dst=tok[:, c0:c1], src=ps_tok[mc][:, 0 : c1 - c0])
    for ct in range(n_ct):
        tpp = nl.ndarray((P, B), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tpp, data=tok[:, ct * P : (ct + 1) * P])
        nisa.tensor_copy(dst=proj[:, ct, :], src=tpp)

    gate_rows = []
    for gi in range(2):
        # a | b columns of the token-major projection -> [vh, B] rows (head on partitions)
        tpg = nl.ndarray((vh, B), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tpg, data=tok[:, NW + gi * vh : NW + (gi + 1) * vh])
        rows = nl.ndarray((vh, B), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=rows, src=tpg)
        gate_rows.append(rows)
    a_rows, b_rows = gate_rows

    # g = -exp(A_log) * softplus(a + dt_bias); eg = exp(g); beta = sigmoid(b)
    alog_sb = nl.ndarray((vh, 1), dtype=nl.float32, buffer=nl.sbuf)
    dt_sb = nl.ndarray((vh, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=alog_sb, src=a_log.reshape((vh, 1)))
    nisa.dma_copy(dst=dt_sb, src=dt_bias.reshape((vh, 1)))
    neg_exp_a = nl.ndarray((vh, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=neg_exp_a, op=nl.exp, data=alog_sb)
    nisa.tensor_scalar(dst=neg_exp_a, data=neg_exp_a, op0=nl.multiply, operand0=-1.0)
    sp = nl.ndarray((vh, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sp, op=nl.softplus, data=a_rows, bias=dt_sb)
    g_rows = nl.ndarray((vh, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=g_rows, data=sp, op0=nl.multiply, operand0=neg_exp_a)
    eg_rows = nl.ndarray((vh, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=eg_rows, op=nl.exp, data=g_rows)
    beta_rows = nl.ndarray((vh, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=beta_rows, op=nl.sigmoid, data=b_rows)
    eg_bc = nl.ndarray((P, vh, B), dtype=nl.float32, buffer=nl.sbuf)
    beta_bc = nl.ndarray((P, vh, B), dtype=nl.float32, buffer=nl.sbuf)
    for h in range(vh):
        ps_e = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(ps_e, sel_h[:, h, :], eg_rows)
        nisa.tensor_copy(dst=eg_bc[:, h, :], src=ps_e)
        ps_b = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(ps_b, sel_h[:, h, :], beta_rows)
        nisa.tensor_copy(dst=beta_bc[:, h, :], src=ps_b)

    # ---------------- depthwise conv over [h0 h1 h2 x] + SiLU ----------------
    # History arrives token-major ([B, Cc*Kh] contiguous rows); PE-transpose per (tap, tile).
    hist_tok = nl.ndarray((B, Cc, Kh), dtype=conv_state.dtype, buffer=nl.sbuf)
    _dma(dst=hist_tok, src=conv_state)
    cw = nl.ndarray((P, n_conv_t, K), dtype=conv_w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=cw, src=conv_w.ap(pattern=[[K, P], [P * K, n_conv_t], [1, K]], offset=0))
    cwf = nl.ndarray((P, n_conv_t, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cwf, src=cw)
    convo = nl.ndarray((P, n_conv_t, B), dtype=nl.float32, buffer=nl.sbuf)
    newh_tok = nl.ndarray((B, Cc, Kh), dtype=conv_state.dtype, buffer=nl.sbuf)
    for ct in range(n_conv_t):
        hist = nl.ndarray((P, Kh, B), dtype=conv_state.dtype, buffer=nl.sbuf)
        for j in range(Kh):
            tph = nl.ndarray((P, B), dtype=conv_state.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=tph, data=hist_tok[:, ct * P : (ct + 1) * P, j])
            nisa.tensor_copy(dst=hist[:, j, :], src=tph)
        acc = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=acc, data=proj[:, ct, :], op0=nl.multiply, operand0=cwf[:, ct, Kh : Kh + 1])
        for j in range(Kh):
            nisa.scalar_tensor_tensor(
                dst=acc, data=hist[:, j, :], op0=nl.multiply, operand0=cwf[:, ct, j : j + 1], op1=nl.add,
                operand1=acc,
            )
        conv_bf = nl.ndarray((P, B), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(dst=conv_bf, op=nl.silu, data=acc)
        nisa.tensor_copy(dst=convo[:, ct, :], src=conv_bf)
        # new history [h1 h2 x], token-major: shift the old taps, transpose x back
        for j in range(Kh - 1):
            nisa.tensor_copy(dst=newh_tok[:, ct * P : (ct + 1) * P, j], src=hist_tok[:, ct * P : (ct + 1) * P, j + 1])
        tpx = nl.ndarray((B, P), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tpx, data=proj[:, ct, :])
        nisa.tensor_copy(dst=newh_tok[:, ct * P : (ct + 1) * P, Kh - 1], src=tpx)
    _dma(dst=new_conv, src=newh_tok)

    # SiLU(z) for every head now, while ACT still has the SiLU table loaded.
    silu_z_all = nl.ndarray((P, vh, B), dtype=nl.float32, buffer=nl.sbuf)
    for h in range(vh):
        nisa.activation(dst=silu_z_all[:, h, :], op=nl.silu, data=proj[:, z_t0 + h, :])

    # ---------------- l2norm(q), l2norm(k), q.k ----------------
    def _col_sum_bcast(src, n):
        """[P, n] -> PSUM [P, n] with every partition holding the column sums of src."""
        ps_sum = nl.ndarray((1, n), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(ps_sum, ones_col, src)
        row = nl.ndarray((1, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=row, src=ps_sum)
        ps_bc = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(ps_bc, ones_row, row)
        return ps_bc

    qn_f = nl.ndarray((P, kh, B), dtype=nl.float32, buffer=nl.sbuf)
    kn_f = nl.ndarray((P, kh, B), dtype=nl.float32, buffer=nl.sbuf)
    qk_bc = nl.ndarray((P, kh, B), dtype=nl.float32, buffer=nl.sbuf)
    kq_f = nl.ndarray((P, kh, B, 2), dtype=nl.float32, buffer=nl.sbuf)
    kq_bf = nl.ndarray((P, kh, B, 2), dtype=nl.bfloat16, buffer=nl.sbuf)
    kn_bf = nl.ndarray((P, kh, B), dtype=nl.bfloat16, buffer=nl.sbuf)
    for j in range(kh):
        sq2 = nl.ndarray((P, 2, B), dtype=nl.float32, buffer=nl.sbuf)
        for i, src_t in enumerate((q_t0 + j, k_t0 + j)):
            nisa.tensor_tensor(dst=sq2[:, i, :], data1=convo[:, src_t, :], data2=convo[:, src_t, :], op=nl.multiply)
        ps_bc = _col_sum_bcast(sq2.reshape((P, 2 * B)), 2 * B)
        rs2 = nl.ndarray((P, 2, B), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rs2.reshape((P, 2 * B)), op=nl.rsqrt, data=ps_bc, bias=eps)
        for i, (src_t, dst, scale) in enumerate(((q_t0 + j, qn_f, HEAD**-0.5), (k_t0 + j, kn_f, 1.0))):
            nisa.scalar_tensor_tensor(
                dst=dst[:, j, :], data=convo[:, src_t, :], op0=nl.multiply, operand0=scale, op1=nl.multiply,
                operand1=rs2[:, i, :],
            )
        prod = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prod, data1=qn_f[:, j, :], data2=kn_f[:, j, :], op=nl.multiply)
        ps_qk = _col_sum_bcast(prod, B)
        nisa.tensor_copy(dst=qk_bc[:, j, :], src=ps_qk)
        nisa.tensor_copy(dst=kq_f[:, j, :, 0], src=kn_f[:, j, :])
        nisa.tensor_copy(dst=kq_f[:, j, :, 1], src=qn_f[:, j, :])
        nisa.tensor_copy(dst=kq_bf[:, j, :, :], src=kq_f[:, j, :, :])
        nisa.tensor_copy(dst=kn_bf[:, j, :], src=kn_f[:, j, :])

    norm_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=norm_sb, src=norm_w.reshape((P, 1)))
    o_all = nl.ndarray((P, vh, B), dtype=nl.bfloat16, buffer=nl.sbuf)

    # ---------------- recurrence ----------------
    rec_args = dict(
        B=B, vh=vh, kh=kh, gqa=gqa, G=G, v_t0=v_t0, rec_state=rec_state, new_rec=new_rec, kn_bf=kn_bf,
        kq_f=kq_f, kq_bf=kq_bf, kn_f=kn_f, eg_bc=eg_bc, beta_bc=beta_bc, convo=convo, qk_bc=qk_bc, o_all=o_all,
        ident=ident, rec_row0=rec_row0, new_row0=new_row0,
    )
    if _REC == "v1":
        _recurrence_v1(**rec_args)
    else:
        _recurrence_v2(**rec_args)

    # ---------------- gated RMSNorm over the head dim, all heads at once ----------------
    core = nl.ndarray((P, vh, B), dtype=nl.bfloat16, buffer=nl.sbuf)
    osq = nl.ndarray((P, vh * B), dtype=nl.float32, buffer=nl.sbuf)
    o_flat = o_all.reshape((P, vh * B))
    nisa.tensor_tensor(dst=osq, data1=o_flat, data2=o_flat, op=nl.multiply)
    ps_ms = _col_sum_bcast(osq, vh * B)
    rms = nl.ndarray((P, vh * B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rms, op=nl.rsqrt, data=ps_ms, scale=1.0 / HEAD, bias=eps)
    on_bf = nl.ndarray((P, vh * B), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=on_bf, data1=o_flat, data2=rms, op=nl.multiply)
    nisa.scalar_tensor_tensor(
        dst=core.reshape((P, vh * B)), data=on_bf, op0=nl.multiply, operand0=norm_sb, op1=nl.multiply,
        operand1=silu_z_all.reshape((P, vh * B)),
    )

    # ---------------- out_proj: out[B, N_out] = core^T @ w_out ----------------
    if w_out_sb is not None:
        wo = w_out_sb
    else:
        wo = nl.ndarray((P, vh, N_out), dtype=nl.bfloat16, buffer=nl.sbuf)
        _dma(dst=wo, src=w_out.ap(pattern=[[N_out, P], [P * N_out, vh], [1, N_out]], offset=0))
    NCH = 512
    o_sb = nl.ndarray((B, N_out), dtype=nl.bfloat16, buffer=nl.sbuf)
    for c0 in range(0, N_out, NCH):
        cn = min(NCH, N_out - c0)
        ps_o = nl.ndarray((B, NCH), dtype=nl.float32, buffer=nl.psum)
        for h in range(vh):
            nisa.nc_matmul(ps_o[:, 0:cn], core[:, h, :], wo[:, h, c0 : c0 + cn])
        nisa.tensor_copy(dst=o_sb[:, c0 : c0 + cn], src=ps_o[:, 0:cn])
    return o_sb, new_conv, new_rec


def _recurrence_v1(B, vh, kh, gqa, G, v_t0, rec_state, new_rec, kn_bf, kq_f, kq_bf, kn_f, eg_bc, beta_bc, convo,
                   qk_bc, o_all, ident, rec_row0=0, new_row0=0):
    """Original recurrence: groups of G tokens; per (token, head) one bf16 state stationary for the
    matvec and one masked-k stationary for the rank-1 term (two LDWEIGHTS per token and head)."""
    for b0 in range(0, B, G):
        # States of G tokens, all heads: [128k, G, vh*128v]; each (k, token) row is one 2 KB run.
        S_g = nl.ndarray((P, G, vh * P), dtype=nl.float32, buffer=nl.sbuf)
        _dma(dst=S_g, src=_state_ap(rec_state, rec_row0 + b0, G, vh))
        # masked-k stationaries for this group: ksel[p, j, b, m] = (p == b) * k_j[m, b0 + b]
        ksel = nl.ndarray((G, kh, G, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for j in range(kh):
            tpk = nl.ndarray((G, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tpk, data=kn_bf[:, j, b0 : b0 + G])
            k_tok = nl.ndarray((G, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=k_tok, src=tpk)
            for b in range(G):
                nisa.tensor_scalar(
                    dst=ksel[:, j, b, :], data=k_tok, op0=nl.multiply, operand0=ident[:, b : b + 1],
                    engine=nisa.vector_engine if b % 2 == 0 else nisa.scalar_engine,
                )

        ps_kq = nl.ndarray((P, vh, G, 2), dtype=nl.float32, buffer=nl.psum)
        for b in range(G):
            for h in range(vh):
                j = h // gqa
                S_bh = S_g[:, b, h * P : (h + 1) * P]
                if _S_BF16:
                    S_bf = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.activation(dst=S_bf, op=nl.copy, data=S_bh)
                    nisa.nc_matmul(ps_kq[:, h, b, :], S_bf, kq_bf[:, j, b0 + b, :])
                else:
                    nisa.nc_matmul(ps_kq[:, h, b, :], S_bh, kq_f[:, j, b0 + b, :])

        delta_toks = nl.ndarray((G, vh, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for h in range(vh):
            j = h // gqa
            egh = eg_bc[:, h, b0 : b0 + G]
            tmp = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=tmp, data1=egh, data2=ps_kq[:, h, :, 0], op=nl.multiply)
            diff = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=diff, data1=convo[:, v_t0 + h, b0 : b0 + G], data2=tmp, op=nl.subtract)
            delta = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=delta, data1=diff, data2=beta_bc[:, h, b0 : b0 + G], op=nl.multiply)
            # o = eg*qS + (q.k)*delta
            o1 = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=o1, data1=egh, data2=ps_kq[:, h, :, 1], op=nl.multiply)
            o2 = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=o2, data1=qk_bc[:, j, b0 : b0 + G], data2=delta, op=nl.multiply)
            nisa.tensor_tensor(dst=o_all[:, h, b0 : b0 + G], data1=o1, data2=o2, op=nl.add)

            delta_bf = nl.ndarray((P, G), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=delta_bf, src=delta)
            tpd = nl.ndarray((G, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tpd, data=delta_bf)
            nisa.tensor_copy(dst=delta_toks[:, h, :], src=tpd)

        # Rank-1 updates, token outer / head inner: the masked-k stationary of token b is
        # shared by all heads of its k group, so it is loaded once per (token, k head).
        S_new = nl.ndarray((P, G, vh * P), dtype=nl.float32, buffer=nl.sbuf)
        for b in range(G):
            for h in range(vh):
                j = h // gqa
                kd = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(kd, ksel[:, j, b, :], delta_toks[:, h, :])
                # S' = eg*S + k (x) delta in one vector pass
                nisa.scalar_tensor_tensor(
                    dst=S_new[:, b, h * P : (h + 1) * P], data=S_g[:, b, h * P : (h + 1) * P], op0=nl.multiply,
                    operand0=eg_bc[:, h, b0 + b : b0 + b + 1], op1=nl.add, operand1=kd,
                )
        _dma(dst=_state_ap(new_rec, new_row0 + b0, G, vh), src=S_new)




def _recurrence_v2(B, vh, kh, gqa, G, v_t0, rec_state, new_rec, kn_bf, kq_f, kq_bf, kn_f, eg_bc, beta_bc, convo,
                   qk_bc, o_all, ident, rec_row0=0, new_row0=0):
    """Phase-split recurrence over super-blocks of NB tokens whose fp32 states stay resident in SBUF.

    A  per group of G tokens: one state DMA, ONE bf16 copy of the whole group, then the matvecs
       [kS | qS] = S^T [k q] into a PSUM tile that holds the whole super-block (S stationary: the one
       LDWEIGHTS per (token, head) the PE cannot avoid).
    B  delta / o for all NB tokens at once (a handful of [128, NB] vector ops per head instead of
       ~10 per head and group), delta made token-major with one PE transpose per head.
    C  rank-1 terms k (x) delta on the PE and S' = eg*S + k (x) delta in place (one DVE pass per token
       and head), state stores per group.
       dmask: the stationary is the token-major k of a 32-token row group (one LDWEIGHTS per 32 tokens,
              PE row tiling at partition 32*g) and the moving operand is delta masked to the token's
              row: M[p, bl, :] = (p % 32 == bl) * delta[p, :], built for the whole super-block by ONE
              broadcast tensor_tensor; consecutive tokens share one matmul (up to 512 PSUM columns).
       kmask: v1's form (masked-k stationary per token, its heads in one matmul): one LDWEIGHTS per
              token, cheaper when vh is large (dmask's mask costs 32*vh*128 elements per partition).
    """
    VP = vh * P
    NB = min(B, max(G, (_RES_KB * 1024 // (VP * 4)) // G * G))
    r1_mode = _R1
    if r1_mode == "auto":
        r1_mode = "dmask" if vh == 1 else "kmask"

    # identity [128, 128] for the masks
    iota_p = nl.ndarray((P, P), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(iota_p, pattern=[[1, P]], offset=0, channel_multiplier=-1)
    ident_p = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)  # ident_p[p, c] = (p == c)
    nisa.tensor_scalar(dst=ident_p, data=iota_p, op0=nl.equal, operand0=0)

    for sb0 in range(0, B, NB):
        nb = min(NB, B - sb0)
        use_dmask = r1_mode == "dmask" and (nb <= 32 or nb % 32 == 0)
        GM = min(32, nb)

        # ---- A: resident states, bf16 copies, matvecs ----
        S_res = nl.ndarray((P, NB, VP), dtype=nl.float32, buffer=nl.sbuf)
        kernel_assert(vh * NB * 2 * 4 <= 2048, "matvec results of a super-block must fit one PSUM bank")
        ps_kq = nl.ndarray((P, vh, NB, 2), dtype=nl.float32, buffer=nl.psum)
        # k-major: one DMA per KL tokens (contiguous KL*vh*512 bytes per partition); token-major: per group
        KL = (_KLOAD or nb) if _KMAJOR else G
        for li, l0 in enumerate(range(0, nb, KL)):
            ln = min(KL, nb - l0)
            _sdma(dst=S_res[:, l0 : l0 + ln, :], src=_state_ap(rec_state, rec_row0 + sb0 + l0, ln, vh), i=li)
        for gi, g0 in enumerate(range(0, nb, G)):
            S_bf = nl.ndarray((P, G, VP), dtype=nl.bfloat16, buffer=nl.sbuf)
            src_flat = S_res[:, g0 : g0 + G, :]
            if _CPY == "dve" or (_CPY == "alt" and gi % 2 == 1):
                nisa.tensor_copy(dst=S_bf, src=src_flat, engine=nisa.vector_engine)
            else:
                nisa.activation(dst=S_bf, op=nl.copy, data=src_flat)
            for b in range(G):
                bt = sb0 + g0 + b
                for h in range(vh):
                    j = h // gqa
                    nisa.nc_matmul(ps_kq[:, h, g0 + b, :], S_bf[:, b, h * P : (h + 1) * P], kq_bf[:, j, bt, :])

        # ---- B: delta, o for the whole super-block ----
        cs = slice(sb0, sb0 + nb)
        d_tok = nl.ndarray((NB, vh, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for h in range(vh):
            j = h // gqa
            egh = eg_bc[:, h, cs]
            tmp = nl.ndarray((P, NB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=tmp[:, 0:nb], data1=egh, data2=ps_kq[:, h, 0:nb, 0], op=nl.multiply)
            diff = nl.ndarray((P, NB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=diff[:, 0:nb], data1=convo[:, v_t0 + h, cs], data2=tmp[:, 0:nb], op=nl.subtract)
            delta = nl.ndarray((P, NB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=delta[:, 0:nb], data1=diff[:, 0:nb], data2=beta_bc[:, h, cs], op=nl.multiply)
            o1 = nl.ndarray((P, NB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=o1[:, 0:nb], data1=egh, data2=ps_kq[:, h, 0:nb, 1], op=nl.multiply)
            o2 = nl.ndarray((P, NB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=o2[:, 0:nb], data1=qk_bc[:, j, cs], data2=delta[:, 0:nb], op=nl.multiply)
            nisa.tensor_tensor(dst=o_all[:, h, cs], data1=o1[:, 0:nb], data2=o2[:, 0:nb], op=nl.add)
            delta_bf = nl.ndarray((P, NB), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=delta_bf[:, 0:nb], src=delta[:, 0:nb])
            tpd = nl.ndarray((NB, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tpd[0:nb], data=delta_bf[:, 0:nb])
            nisa.tensor_copy(dst=d_tok[0:nb, h, :], src=tpd[0:nb])
        # token-major k of the super-block (rows at partition 0)
        k_tok = nl.ndarray((NB, kh, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for j in range(kh):
            tpk = nl.ndarray((NB, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tpk[0:nb], data=kn_bf[:, j, cs])
            nisa.tensor_copy(dst=k_tok[0:nb, j, :], src=tpk[0:nb])

        # ---- C: rank-1 terms and in-place state update, stores per group ----
        CW = gqa * P  # columns of one k head's v heads
        if use_dmask:
            # maskg[p, bl] = (p % GM == bl): sum of the GM-wide column slices of the identity
            maskg = nl.ndarray((P, GM), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=maskg, src=ident_p[:, 0:GM])
            for q in range(1, P // GM):
                nisa.tensor_tensor(dst=maskg, data1=maskg, data2=ident_p[:, q * GM : (q + 1) * GM], op=nl.add)
            M = nl.ndarray((NB, GM, VP), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=M[0:nb],
                data1=d_tok.ap(pattern=[[VP, nb], [0, GM], [1, VP]], offset=0),
                data2=maskg.ap(pattern=[[GM, nb], [1, GM], [0, VP]], offset=0),
                op=nl.multiply,
                engine=nisa.vector_engine,
            )
        else:
            ksel = nl.ndarray((NB, kh, NB, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        if CW <= MCH:
            tpm = MCH // CW  # tokens per matmul (dmask) / heads chunk = all
            hch = gqa
        else:
            tpm = 1
            hch = MCH // P
        # matmul jobs (store group g0, tokens b0..b1); with PE row tiling, consecutive jobs rotate over
        # the 32-row groups so that matmuls on different row tiles are adjacent in program order
        jobs = [(g0, b0, min(g0 + G, b0 + tpm)) for g0 in range(0, nb, G) for b0 in range(g0, g0 + G, tpm)]
        if use_dmask and nb > 32 and _IL:
            jobs.sort(key=lambda jb: ((jb[1] % GM), jb[1] // GM))
        done = {}
        for g0, b0, b1 in jobs:
            kd_of = {}
            for j in range(kh):
                for hc0 in range(0, gqa, hch):
                    hc1 = min(gqa, hc0 + hch)
                    ncol = (hc1 - hc0) * P
                    if use_dmask:
                        kd = nl.ndarray((P, b1 - b0, ncol), dtype=nl.float32, buffer=nl.psum)
                        r0 = (b0 // GM) * GM
                        bl0, bl1 = b0 - r0, b1 - r0  # b0..b1 never straddle a row group (GM % tpm == 0)
                        c0 = j * CW + hc0 * P
                        mov = M[r0 : r0 + GM, bl0:bl1, c0 : c0 + ncol]
                        if nb > 32:
                            nisa.nc_matmul(
                                kd, k_tok[r0 : r0 + GM, j, :], mov, tile_position=(r0, 0), tile_size=(32, P)
                            )
                        else:
                            nisa.nc_matmul(kd, k_tok[0:nb, j, :], mov)
                        for b in range(b0, b1):
                            for hh in range(hc0, hc1):
                                kd_of[b, j * gqa + hh] = (kd, b - b0, (hh - hc0) * P)
                    else:
                        for b in range(b0, b1):
                            if hc0 == 0:
                                nisa.tensor_scalar(
                                    dst=ksel[0:nb, j, b, :], data=k_tok[0:nb, j, :], op0=nl.multiply,
                                    operand0=ident_p[0:nb, b : b + 1],
                                    engine=nisa.vector_engine if b % 2 == 0 else nisa.scalar_engine,
                                )
                            kd = nl.ndarray((P, hc1 - hc0, P), dtype=nl.float32, buffer=nl.psum)
                            nisa.nc_matmul(kd, ksel[0:nb, j, b, :], d_tok[0:nb, j * gqa + hc0 : j * gqa + hc1, :])
                            for hh in range(hc0, hc1):
                                kd_of[b, j * gqa + hh] = (kd, hh - hc0, 0)
            for b in range(b0, b1):
                bt = sb0 + b
                for h in range(vh):
                    kd, ki, off = kd_of[b, h]
                    hs = slice(h * P, (h + 1) * P)
                    nisa.scalar_tensor_tensor(
                        dst=S_res[:, b, hs], data=S_res[:, b, hs], op0=nl.multiply,
                        operand0=eg_bc[:, h, bt : bt + 1], op1=nl.add, operand1=kd[:, ki, off : off + P],
                    )
            done[g0] = done.get(g0, 0) + (b1 - b0)
            if done[g0] == G:
                _sdma(dst=_state_ap(new_rec, new_row0 + sb0 + g0, G, vh), src=S_res[:, g0 : g0 + G, :], i=g0 // G + 1)
