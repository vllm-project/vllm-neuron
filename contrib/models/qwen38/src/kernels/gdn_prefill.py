"""Qwen3.5 Gated DeltaNet PREFILL for one TP rank (NKI, LNC=1): chunkwise gated delta rule.

Same fused block as ``gdn_decode.gdn_decode_tkg`` (in_proj -> causal conv + SiLU -> l2norm / gates ->
gated delta rule -> gated RMSNorm -> this rank's out_proj slice), for B sequences of L tokens each,
all on this core. It hands off its final states in exactly the layouts ``gdn_decode_tkg`` reads, so
decode continues from them (P -> D disaggregation). Reference: ``gdn_prefill_torch.gdn_prefill_torch``
(the decode recurrence, token by token).

Layouts (inputs as gdn_decode_tkg):
  x          [B*L, H]                 bf16, sequence-major rows (b*L + t)
  w_proj_p   [128, H/128, 2Kd+2Vd+2vh] q|k|v|z|a|b in_proj, rows pre-permuted
  w_out      [Vd, N_out]              N_out = H, or wider (folded extra columns)
  conv_state [B, Cc, K-1]  (optional) raw conv history, oldest first (zeros if omitted)
  rec_state  [B, 128k, vh, 128v] fp32 (optional; [128k, B, vh, 128v] with QWEN35_GDN_KMAJOR=1; zeros if omitted)
Outputs: out [B*L, N_out] bf16, new_conv [B, Cc, K-1] bf16, new_rec in rec_state's layout (fp32).

Padded sequence (``valid_len`` [1, 1] int32, B = 1; edit 0001 in ../PROVENANCE.md): rows t >= valid_len are padding
(a chunked prefill's last chunk in a fixed bucket). They get g = 0, beta = 0 and k = 0, so the recurrent state
passes them unchanged (decay exp(0) = 1, no rank-one update), and new_conv is the K-1 pre-conv rows ending at row
valid_len - 1 (old history rows where valid_len < K-1), selected by one-hot matmuls (exact). Their outputs are
unspecified. The pad tables live in HBM so the on-device block loop reads them with its register.

Algorithm per chunk of C = 128 tokens and v-head (FLA chunk_gated_delta_rule, WY form), with
gc = in-chunk cumsum of g, gam = exp(gc), D[i, s] = exp(gc_i - gc_s) (s <= i):
    A   = strict_lower(beta_i (k_i.k_s) D[i, s]);   T = (I + A)^-1        (Schur merge + one Newton step)
    U   = T (beta v);  W = T (beta gam k)                                    (state independent)
    dlt = U - W S                                                           (chain: 1 matmul + 1 DVE op)
    o   = gam (q S) + (upper_incl(q.k) D) dlt                               (off the chain)
    S'  = gam_C S + ((gam_C / gam) k)^T dlt                                 (chain: 1 matmul + 1 DVE op)
Matmul operands are bf16 (fp32 PSUM); S, the cumulative gates, decays, A and the Newton residual are
fp32. The tokens of a block of NB tokens (QWEN35_GDNP_BLK, default 512) run through one body; blocks of
a sequence run in an on-device loop (nl.fori_loop) with S and the conv history carried in SBUF.

PE operand orientation (dst = stat^T @ mov): projections keep x^T[:, h1, 128 tokens] stationary and
stream all NT weight columns (one LDWEIGHTS per (128 tokens, h1)); every per-chunk matmul has 128-256
moving columns; the out-projection keeps core^T (128 tokens) stationary and streams w_out.
"""

import os

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert

P = 128
HEAD = 128
C = 128  # chunk length
MCH = 512  # fp32 PSUM bank width (moving columns per matmul)
NEG = 1.0e30
XO = 16  # column of xe where the block starts: 32-byte aligned rows / chunk offsets (DMA transpose output rule);
#          the K-1 history columns sit just before it (XO - 3 .. XO - 1)

_BLK = int(os.environ.get("QWEN35_GDNP_BLK", "512"))  # tokens per loop body (multiple of 128)
_XT = os.environ.get("QWEN35_GDNP_XT", "auto")  # x^T: dma (HBM DMA transpose) | sb (row DMA + SBUF DMA transpose) | pe
_NEWTON = os.environ.get("QWEN35_GDNP_NEWTON", "1") == "1"  # one Newton step on the bf16 Schur inverse
_DYN = os.environ.get("QWEN35_GDNP_DYN", "auto")  # block loop: auto (dynamic if > 1 block) | 1 | 0 (unrolled)
_WO_RES = os.environ.get("QWEN35_GDNP_WO_RES", "auto")  # w_out resident in SBUF: auto | 1 | 0 (stream per block)
_KMAJOR = os.environ.get("QWEN35_GDN_KMAJOR", "0") == "1"  # decode's k-major rec_state layout
_WIDE = int(os.environ.get("QWEN35_GDNP_WIDE", "2048"))  # max moving columns per matmul (NC-v4: <= 4096 fp32)
_GCDIAG = os.environ.get("QWEN35_GDNP_GCDIAG", "1") == "1"  # column cumsum = diagonal of the row form (1 fp32 matmul less)
_FMT = os.environ.get("QWEN35_GDNP_FMT", "dma")  # token -> feature-major q|k|v|z: dma (SBUF DMA transpose) | pe
_OBF16 = os.environ.get("QWEN35_GDNP_OBF16", "0") == "1"  # vh == 1: out-projection straight to bf16 PSUM
_POOL = os.environ.get("QWEN35_GDNP_POOL", "1") == "1"  # SBUF-only elementwise work on GpSimd (DVE relief)
# Recurrence restructurings (fewer PE matmuls: TP16 670 -> 518 per body with all three), OFF by default: on trn3 inside
# the TP16 EP32 prefill stack GRP and QT each make the stack slower and VT=dma is within noise.
_GRP = os.environ.get("QWEN35_GDNP_GRP", "0") == "1"  # one matmul per key head (not per v-head) for cumsum / W / dS
_VT = os.environ.get("QWEN35_GDNP_VT", "pe")  # per-chunk token-major k / v: dma (SBUF DMA transpose) | pe
_QT = os.environ.get("QWEN35_GDNP_QT", "0") == "1"  # Schur levels 0-4: X = 32x32 Vector-engine transposes of X^T
QB = 32  # Vector-engine transpose tile
_rr = [0]


def _tt(dst, a, b, op):
    """SBUF-only tensor_tensor, on GpSimd when _POOL (DVE is the busiest engine)."""
    if _POOL:
        nisa.tensor_tensor(dst=dst, data1=a, data2=b, op=op, engine=nisa.gpsimd_engine)
    else:
        nisa.tensor_tensor(dst=dst, data1=a, data2=b, op=op)


def _copy(dst, src):
    """PSUM/SBUF copy, alternated between the vector and scalar engines."""
    _rr[0] += 1
    if _rr[0] % 2:
        nisa.tensor_copy(dst=dst, src=src, engine=nisa.vector_engine)
    else:
        nisa.activation(dst=dst, op=nl.copy, data=src)


def _state_ap(t, row, vh):
    """[128k, vh*128v] view of sequence `row` of a decode-layout recurrent state tensor."""
    VP = vh * P
    if _KMAJOR:
        Bt = t.shape[1]
        return t.ap(pattern=[[Bt * VP, P], [1, VP]], offset=row * VP)
    return t.ap(pattern=[[VP, P], [1, VP]], offset=row * P * VP)


def _hbm_rows(t, row0, n, ncol, blk, nb):
    """[n, ncol] HBM rows row0 + blk*nb .. (+n) of a [rows, ncol] tensor; blk is an int or a loop register."""
    if isinstance(blk, int):
        return t.ap(pattern=[[ncol, n], [1, ncol]], offset=(row0 + blk * nb) * ncol)
    tb = t.reshape((t.shape[0] // nb, nb * ncol))
    return tb.ap(pattern=[[ncol, n], [1, ncol]], offset=row0 * ncol, scalar_offset=blk, indirect_dim=0)


def _hbm_rows_t(t, row0, n, H, blk, nb):
    """x rows as the [n, H1, 128] pattern a DMA transpose turns into x^T [128, H1, n]."""
    H1 = H // P
    if isinstance(blk, int):
        return t.ap(pattern=[[H, n], [P, H1], [1, P]], offset=(row0 + blk * nb) * H)
    tb = t.reshape((t.shape[0] // nb, nb * H))
    return tb.ap(pattern=[[H, n], [P, H1], [1, P]], offset=row0 * H, scalar_offset=blk, indirect_dim=0)


def _consts(vh, NCH):
    """Masks and small constant tiles, built once on device."""
    c = {}
    iota_d = nl.ndarray((P, P), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(iota_d, pattern=[[1, P]], offset=0, channel_multiplier=-1)  # f - p

    def cmp(op, val, dt=nl.float32):
        t = nl.ndarray((P, P), dtype=dt, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=t, data=iota_d, op0=op, operand0=val)
        return t

    c["ident"] = cmp(nl.equal, 0)
    c["utri"] = cmp(nl.greater_equal, 0)  # [s, t]: s <= t (cumsum operator)
    for name, op in (("lowneg", nl.less), ("upneg", nl.greater_equal)):
        m = cmp(op, 0)
        nisa.tensor_scalar(dst=m, data=m, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=NEG)
        c[name] = m  # 0 where kept (lowneg: s < i in [i, s]; upneg: i >= s in [s, i]), -1e30 elsewhere

    def blk(m):  # [i, s] = (i >> m == s >> m)
        if m == 0:
            return c["ident"]
        nb, w = P >> m, 1 << m
        v = nl.ndarray((P, nb, w), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(v, pattern=[[-w, nb], [0, w]], offset=0, channel_multiplier=1)  # i - b*2^m
        ge = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ge, data=v.reshape((P, P)), op0=nl.greater_equal, operand0=0)
        lt = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=lt, data=v.reshape((P, P)), op0=nl.less, operand0=w)
        nisa.tensor_tensor(dst=ge, data1=ge, data2=lt, op=nl.multiply)
        return ge

    blks = [blk(m) for m in range(7)]
    ones = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)
    blks.append(ones)
    offm = []  # off-diagonal quadrant masks of each Schur merge level (block 2^l -> 2^(l+1))
    for lvl in range(7):
        t = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=t, data1=blks[lvl + 1], data2=blks[lvl], op=nl.subtract)
        offm.append(t)
    c["offm"] = offm
    noffm = []
    for t in offm:
        n = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=n, data=t, op0=nl.multiply, operand0=-1.0)
        noffm.append(n)
    c["noffm"] = noffm
    c["ones_f"] = ones
    ones_b = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(ones_b, 1.0)
    c["ones_b"] = ones_b
    return c


def _qt(dst, src):
    """dst = src^T for a [128, 128] tile that is zero off its diagonal 32x32 quadrants (4 Vector-engine transposes)."""
    for q in range(P // QB):
        s = slice(q * QB, (q + 1) * QB)
        nisa.nc_transpose(dst=dst[s, s], data=src[s, s], engine=nisa.vector_engine)


def _inverse_T(A, c):
    """T^T (bf16 [s, i]) for T = (I + A)^-1, A fp32 [i, s] strictly lower.

    Schur merge, block size 1 -> 128: with X the block-diagonal inverse of the current level and R the
    level's off-diagonal quadrants of -A, the merged inverse is X + X R X (X R is nilpotent of order 2).
    X and X^T are both carried in bf16: Mid = R^T X^T, X^T += X^T Mid, X += Mid^T X (3 matmuls per level,
    the last two independent). Then one Newton step on the left residual in fp32,
    T1^T = T0^T + T0^T E^T with E^T = I - T0^T - A^T T0^T (A split into bf16 hi + lo), restores the
    accuracy the bf16 levels lose (final state error 1.4e-2 -> 9e-3 on a hard case).
    With _QT, while the merged blocks fit the 32x32 quadrants (levels 0-4: X block-diagonal in 32-blocks,
    zero off the diagonal quadrants) X is the Vector-engine quadrant transpose of X^T, in place, instead of
    the third matmul (and the level-0 PE transpose): 22 -> 17 matmuls.
    """
    noffm = c["noffm"]
    R0 = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    _tt(R0, A, noffm[0], nl.multiply)
    X = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=X, data1=R0, data2=c["ident"], op=nl.add)
    XT = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    if _QT:
        _tt(XT, R0, c["ident"], nl.add)  # zero off the diagonal quadrants; the transposes overwrite the rest
        _qt(XT, X)
    else:
        tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=X)
        _copy(XT, tp)
    for lvl in range(1, 7):
        R = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        _tt(R, A, noffm[lvl], nl.multiply)
        ps_m = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(ps_m, R, XT)  # R^T X^T
        Mid = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        _copy(Mid, ps_m)
        ps_t = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(ps_t, X, Mid)  # X^T R^T X^T
        last = lvl == 6
        if _QT and (1 << (lvl + 1)) <= QB:  # merged blocks still inside the quadrants
            XT_n = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=XT_n, data1=XT, data2=ps_t, op=nl.add)
            _qt(X, XT_n)  # in place: X's off-diagonal quadrants stay zero
            XT = XT_n
            continue
        if not last or _NEWTON:
            ps_x = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(ps_x, Mid, X)  # X R X
        XT_n = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=XT_n, data1=XT, data2=ps_t, op=nl.add)
        if not last or _NEWTON:
            X_n = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=X_n, data1=X, data2=ps_x, op=nl.add)
            X = X_n
        XT = XT_n
    if not _NEWTON:
        return XT
    A_hi = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=A_hi, src=A, engine=nisa.gpsimd_engine if _POOL else nisa.vector_engine)
    A_lo = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    _tt(A_lo, A, A_hi, nl.subtract)
    ps_p = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(ps_p, A_hi, XT)  # A^T T0^T
    nisa.nc_matmul(ps_p, A_lo, XT)
    tmp = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tmp, data1=ps_p, data2=XT, op=nl.add)
    ET = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ET, data1=c["ident"], data2=tmp, op=nl.subtract)
    ps_e = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(ps_e, X, ET)  # T0^T E^T
    TT = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=TT, data1=XT, data2=ps_e, op=nl.add)
    return TT


@nki.jit
def gdn_prefill_tkg(
    x,  # [B*L, H] bf16 normed tokens, sequence-major
    w_proj_p,  # [128, H/128, 2*Kd + 2*Vd + 2*vh] bf16
    conv_w,  # [Cc, K] bf16
    a_log,  # [vh] fp32
    dt_bias,  # [vh] fp32
    norm_w,  # [128] fp32
    w_out,  # [Vd, N_out] bf16
    conv_state=None,  # [B, Cc, K-1] bf16 initial conv history (zeros if None)
    rec_state=None,  # [B, 128, vh, 128] fp32 initial recurrent state (zeros if None)
    seq_len=0,  # L (0: one sequence of all rows)
    valid_len=None,  # [1, 1] int32 (B == 1): rows >= valid_len are padding (state and conv window untouched)
    state_row=0,  # first row of conv_state / rec_state this call reads (sequence b reads row state_row + b)
):
    T, H = x.shape
    L = seq_len or T
    B = T // L
    vh = a_log.shape[0]
    Vd = vh * HEAD
    NT = w_proj_p.shape[2]
    NW = NT - 2 * vh
    Kd = (NW - 2 * Vd) // 2
    kh = Kd // HEAD
    gqa = vh // kh
    Cc = 2 * Kd + Vd
    K = conv_w.shape[1]
    Kh = K - 1
    H1 = H // P
    n_ct = NW // P
    n_conv_t = Cc // P
    q_t0, k_t0, v_t0 = 0, kh, 2 * kh
    NB = min(_BLK, L)
    NCH = NB // C
    nblk = L // NB
    N_out = w_out.shape[1]
    eps = 1e-6
    kernel_assert(T == B * L and L % NB == 0 and NB % C == 0, f"rows {T} must be B * L with L a multiple of {NB}")
    kernel_assert(H % P == 0 and Kd % P == 0 and vh % kh == 0, "H, key dim must be multiples of 128")
    kernel_assert(w_proj_p.shape[0] == P and w_proj_p.shape[1] == H1, "w_proj_p must be [128, H/128, cols]")
    kernel_assert(Kh == 3, "conv width 4")
    masked = valid_len is not None
    kernel_assert(not masked or B == 1, "valid_len masks the pad rows of one sequence")

    out = nl.ndarray((T, N_out), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    new_conv = nl.ndarray((B, Cc, Kh), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    new_rec = nl.ndarray((HEAD, B, vh, HEAD) if _KMAJOR else (B, HEAD, vh, HEAD), dtype=nl.float32,
                         buffer=nl.shared_hbm)

    c = _consts(vh, NCH)

    # ---------------- weights and per-channel / per-head parameters ----------------
    w_all = nl.ndarray((P, H1, NT), dtype=nl.bfloat16, buffer=nl.sbuf)
    SL = max(1, H1 // 4)
    for s0 in range(0, H1, SL):
        nisa.dma_copy(dst=w_all[:, s0 : s0 + SL, :], src=w_proj_p[:, s0 : s0 + SL, :])
    wo_res = (vh * N_out * 2 <= 16 * 1024) if _WO_RES == "auto" else _WO_RES == "1"
    wo_pat = w_out.ap(pattern=[[N_out, P], [P * N_out, vh], [1, N_out]], offset=0)
    wo = None
    if wo_res:
        wo = nl.ndarray((P, vh, N_out), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wo, src=wo_pat)
    cw = nl.ndarray((P, n_conv_t, K), dtype=conv_w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=cw, src=conv_w.ap(pattern=[[K, P], [P * K, n_conv_t], [1, K]], offset=0))
    cwf = nl.ndarray((P, n_conv_t, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cwf, src=cw)
    norm_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=norm_sb, src=norm_w.reshape((P, 1)))
    # per-head rows broadcast over partitions (token-major gate math), replicated over the block's chunks
    nexpa = nl.ndarray((P, NCH, vh), dtype=nl.float32, buffer=nl.sbuf)
    dtr = nl.ndarray((P, NCH, vh), dtype=nl.float32, buffer=nl.sbuf)
    for ci in range(NCH):
        nisa.dma_copy(dst=nexpa[:, ci, :], src=a_log.ap(pattern=[[0, P], [1, vh]], offset=0))
        nisa.dma_copy(dst=dtr[:, ci, :], src=dt_bias.ap(pattern=[[0, P], [1, vh]], offset=0))
    nisa.activation(dst=nexpa, op=nl.exp, data=nexpa)
    nisa.tensor_scalar(dst=nexpa, data=nexpa, op0=nl.multiply, operand0=-1.0)

    xt_mode = _XT
    use_dyn = nblk > 1 and (_DYN == "1" or (_DYN == "auto"))
    if xt_mode == "auto":
        xt_mode = "sb" if use_dyn else "dma"

    m_hbm = sel_hbm = vl = None
    if masked:
        m_hbm, sel_hbm, vl = _pad_tables(valid_len, L, Kh)

    for b in range(B):
        row0 = b * L
        # ---------------- carried state ----------------
        S_f = nl.ndarray((P, vh, P), dtype=nl.float32, buffer=nl.sbuf)  # [k, h, v]
        if rec_state is not None:
            nisa.dma_copy(dst=S_f.reshape((P, vh * P)), src=_state_ap(rec_state, state_row + b, vh))
        else:
            nisa.memset(S_f, 0.0)
        S_b = nl.ndarray((P, vh, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=S_b, src=S_f)
        # [conv history | block] per q|k|v channel tile; z tiles ride in the trailing slots (no history)
        xe = nl.ndarray((P, n_ct, XO + NB), dtype=nl.bfloat16, buffer=nl.sbuf)
        if conv_state is not None:
            nisa.dma_copy(
                dst=xe[:, 0:n_conv_t, XO - Kh : XO],
                src=conv_state.ap(pattern=[[Kh, P], [P * Kh, n_conv_t], [1, Kh]], offset=(state_row + b) * Cc * Kh),
            )
        else:
            nisa.memset(xe[:, 0:n_conv_t, XO - Kh : XO], 0.0)
        hacc = _hist_init(xe, vl, n_conv_t, Kh) if masked else None

        args = dict(
            x=x, out=out, w_all=w_all, wo=wo, wo_pat=wo_pat, cwf=cwf, norm_sb=norm_sb, nexpa=nexpa, dtr=dtr, c=c,
            S_f=S_f, S_b=S_b, xe=xe, row0=row0, H=H, H1=H1, NT=NT, NW=NW, vh=vh, kh=kh, gqa=gqa, n_ct=n_ct,
            n_conv_t=n_conv_t, q_t0=q_t0, k_t0=k_t0, v_t0=v_t0, NB=NB, NCH=NCH, Kh=Kh, N_out=N_out, eps=eps,
            xt_mode=xt_mode, m_hbm=m_hbm, sel_hbm=sel_hbm, hacc=hacc,
        )
        if use_dyn:
            def body(i, args=args):
                _block(blk=i, **args)

            nl.fori_loop(0, nblk, body)
        else:
            for i in range(nblk):
                _block(blk=i, **args)

        nisa.dma_copy(dst=_state_ap(new_rec, b, vh), src=S_f.reshape((P, vh * P)))
        hist = xe[:, 0:n_conv_t, XO - Kh : XO]
        if masked:  # the window that ends at the last valid row (exact: one-hot selections of bf16 values)
            hist = nl.ndarray((P, n_conv_t, Kh), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=hist, src=hacc)
        nisa.dma_copy(
            dst=new_conv.ap(pattern=[[Kh, P], [P * Kh, n_conv_t], [1, Kh]], offset=b * Cc * Kh), src=hist
        )
    return out, new_conv, new_rec


def _pad_tables(valid_len, L, Kh):
    """Pad-row tables of one sequence with a runtime valid length, in HBM so a block of the on-device loop reads its
    own rows with the loop register (``_hbm_rows``): ``m [L, 1]`` fp32 = (t < valid_len), and ``sel [L, Kh]`` bf16 =
    (t + Kh - j == valid_len), the one-hot of window column j (row valid_len - Kh + j). Returns them and valid_len
    broadcast to every partition ([P, 1] fp32)."""
    vl_i = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=vl_i, src=valid_len.ap(pattern=[[0, P], [1, 1]], offset=0))
    vl = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vl, src=vl_i)
    m_hbm = nl.ndarray((L, 1), dtype=nl.float32, buffer=nl.private_hbm)
    sel_hbm = nl.ndarray((L, Kh), dtype=nl.bfloat16, buffer=nl.private_hbm)
    for c0 in range(0, L, C):
        pi = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(pi, pattern=[[0, 1]], offset=c0, channel_multiplier=1)  # t
        pf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pf, src=pi)
        m = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=m, data=pf, op0=nl.less, operand0=vl)
        nisa.dma_copy(dst=m_hbm[c0 : c0 + C], src=m)
        si = nl.ndarray((P, Kh), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(si, pattern=[[-1, Kh]], offset=c0 + Kh, channel_multiplier=1)  # t + Kh - j
        sf = nl.ndarray((P, Kh), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sf, src=si)
        nisa.tensor_scalar(dst=sf, data=sf, op0=nl.equal, operand0=vl)
        sb = nl.ndarray((P, Kh), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sb, src=sf)
        nisa.dma_copy(dst=sel_hbm[c0 : c0 + C], src=sb)
    return m_hbm, sel_hbm, vl


def _hist_init(xe, vl, n_conv_t, Kh):
    """fp32 accumulator of the final conv window, seeded with the old history's columns the window still covers (a
    valid length below Kh, or none): column j is old column e when valid_len + j == e."""
    hacc = nl.ndarray((P, n_conv_t, Kh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(hacc, 0.0)
    for j in range(Kh):
        for e in range(j, Kh):
            f = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=f, data=vl, op0=nl.equal, operand0=float(e - j))
            nisa.scalar_tensor_tensor(dst=hacc[:, :, j], data=xe[:, 0:n_conv_t, XO - Kh + e], op0=nl.multiply,
                                      operand0=f, op1=nl.add, operand1=hacc[:, :, j])
    return hacc


def _block(x, out, w_all, wo, wo_pat, cwf, norm_sb, nexpa, dtr, c, S_f, S_b, xe, row0, H, H1, NT, NW, vh, kh, gqa,
           n_ct, n_conv_t, q_t0, k_t0, v_t0, NB, NCH, Kh, N_out, eps, xt_mode, blk, m_hbm=None, sel_hbm=None,
           hacc=None):
    """NB tokens of one sequence: projection, conv, norms, chunk recurrence, gated norm, out-projection."""
    WM = max(MCH, _WIDE)  # moving columns per projection / out-projection matmul (one LDWEIGHTS each)
    n_mch = (NT + WM - 1) // WM
    z_t0 = 2 * kh + vh

    # ---------------- x^T and token-major projections, per 128-token chunk ----------------
    tok = nl.ndarray((P, NCH, NT), dtype=nl.bfloat16, buffer=nl.sbuf)
    if xt_mode == "sb":
        x_rows = nl.ndarray((P, NCH, H1, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for ci in range(NCH):
            nisa.dma_copy(dst=x_rows[:, ci, :, :], src=_hbm_rows_t(x, row0 + ci * C, C, H, blk, NB))
    for ci in range(NCH):
        xT = nl.ndarray((P, H1, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        if xt_mode == "dma":
            nisa.dma_transpose(dst=xT, src=_hbm_rows_t(x, row0 + ci * C, C, H, blk, NB), axes=(2, 1, 0))
        elif xt_mode == "sb":
            nisa.dma_transpose(dst=xT, src=x_rows[:, ci, :, :], axes=(2, 1, 0))
        else:
            x_sb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_sb, src=_hbm_rows(x, row0 + ci * C, C, H, blk, NB))
            for h1 in range(H1):
                tpx = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tpx, data=x_sb[:, h1 * P : (h1 + 1) * P])
                _copy(xT[:, h1, :], tpx)
        ps = [nl.ndarray((P, min(WM, NT - mc * WM)), dtype=nl.float32, buffer=nl.psum) for mc in range(n_mch)]
        for h1 in range(H1):
            for mc in range(n_mch):
                c0, c1 = mc * WM, min(NT, (mc + 1) * WM)
                nisa.nc_matmul(ps[mc], xT[:, h1, :], w_all[:, h1, c0:c1])
        for mc in range(n_mch):
            c0, c1 = mc * WM, min(NT, (mc + 1) * WM)
            for e0 in range(c0, c1, MCH):  # evacuate per bank
                e1 = min(c1, e0 + MCH)
                _copy(tok[:, ci, e0:e1], ps[mc][:, e0 - c0 : e1 - c0])

    # ---------------- feature-major q|k|v (conv input) and SiLU(z) ----------------
    silu_z = nl.ndarray((P, vh, NB), dtype=nl.float32, buffer=nl.sbuf)
    if _FMT == "dma":  # DMA-engine transposes (the PE is the bound engine): all q|k|v|z tiles of a chunk at once
        for ci in range(NCH):
            nisa.dma_transpose(
                dst=xe[:, :, XO + ci * C : XO + (ci + 1) * C],
                src=tok.ap(pattern=[[NCH * NT, P], [P, n_ct], [1, P]], offset=ci * NT), axes=(2, 1, 0),
            )
        nisa.activation(dst=silu_z, op=nl.silu, data=xe[:, z_t0:n_ct, XO : XO + NB])
    else:
        for ci in range(NCH):
            for ct in range(n_ct):
                tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tp, data=tok[:, ci, ct * P : (ct + 1) * P])
                if ct < n_conv_t:
                    _copy(xe[:, ct, XO + ci * C : XO + (ci + 1) * C], tp)
                else:
                    nisa.activation(dst=silu_z[:, ct - z_t0, ci * C : (ci + 1) * C], op=nl.silu, data=tp)

    # ---------------- gates, token-major [tok, chunk, head] ----------------
    a_f = nl.ndarray((P, NCH, vh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=a_f, data1=tok[:, :, NW : NW + vh], data2=dtr, op=nl.add)
    nisa.activation(dst=a_f, op=nl.softplus, data=a_f)
    g_all = nl.ndarray((P, NCH, vh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=g_all, data1=a_f, data2=nexpa, op=nl.multiply)
    beta_all = nl.ndarray((P, NCH, vh), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=beta_all, op=nl.sigmoid, data=tok[:, :, NW + vh : NW + 2 * vh])
    m_tok = None
    if m_hbm is not None:
        # pad rows: g = 0 (no decay), beta = 0 and k = 0 (no update), so the state passes them unchanged
        m_tok = nl.ndarray((P, NCH, 1), dtype=nl.float32, buffer=nl.sbuf)
        sel = nl.ndarray((P, NCH, Kh), dtype=nl.bfloat16, buffer=nl.sbuf)
        for ci in range(NCH):
            nisa.dma_copy(dst=m_tok[:, ci, :], src=_hbm_rows(m_hbm, row0 + ci * C, C, 1, blk, NB))
            nisa.dma_copy(dst=sel[:, ci, :], src=_hbm_rows(sel_hbm, row0 + ci * C, C, Kh, blk, NB))
        m_b = m_tok.ap(pattern=[[NCH, P], [1, NCH], [0, vh]], offset=0)
        nisa.tensor_tensor(dst=g_all, data1=g_all, data2=m_b, op=nl.multiply)
        nisa.tensor_tensor(dst=beta_all, data1=beta_all, data2=m_b, op=nl.multiply)
        # the conv window's rows in this block: hacc[ch, j] += sum_t tok[t, ch] sel[t, j] (one nonzero term)
        for ct in range(n_conv_t):
            ps_h = nl.ndarray((P, Kh), dtype=nl.float32, buffer=nl.psum)
            for ci in range(NCH):
                nisa.nc_matmul(ps_h, tok[:, ci, ct * P : (ct + 1) * P], sel[:, ci, :])
            nisa.tensor_tensor(dst=hacc[:, ct, :], data1=hacc[:, ct, :], data2=ps_h, op=nl.add)

    # ---------------- causal depthwise conv over [history | block] + SiLU ----------------
    convo = nl.ndarray((P, n_conv_t, NB), dtype=nl.bfloat16, buffer=nl.sbuf)
    for ct in range(n_conv_t):
        acc = nl.ndarray((P, NB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=acc, data=xe[:, ct, XO : XO + NB], op0=nl.multiply, operand0=cwf[:, ct, Kh : Kh + 1])
        for j in range(Kh):
            nisa.scalar_tensor_tensor(
                dst=acc, data=xe[:, ct, XO - Kh + j : XO - Kh + j + NB], op0=nl.multiply, operand0=cwf[:, ct, j : j + 1], op1=nl.add,
                operand1=acc,
            )
        nisa.activation(dst=convo[:, ct, :], op=nl.silu, data=acc)
    nisa.tensor_copy(dst=xe[:, 0:n_conv_t, XO - Kh : XO], src=xe[:, 0:n_conv_t, XO + NB - Kh : XO + NB])  # history for the next block

    # ---------------- l2norm(k), l2norm(q)/sqrt(d): kq[:, j, 0] = k^T, kq[:, j, 1] = q^T ----------------
    kq = nl.ndarray((P, kh, 2, NB), dtype=nl.bfloat16, buffer=nl.sbuf)
    for j in range(kh):
        for slot, src_t, scale in ((0, k_t0 + j, 1.0), (1, q_t0 + j, HEAD**-0.5)):
            for c0 in range(0, NB, MCH):
                cn = min(MCH, NB - c0)
                sq = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.sbuf)
                _tt(sq[:, 0:cn], convo[:, src_t, c0 : c0 + cn], convo[:, src_t, c0 : c0 + cn], nl.multiply)
                ss = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(ss[:, 0:cn], c["ones_f"], sq[:, 0:cn], is_stationary_onezero=True)
                rs = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=rs[:, 0:cn], op=nl.rsqrt, data=ss[:, 0:cn], bias=eps)
                nisa.scalar_tensor_tensor(
                    dst=kq[:, j, slot, c0 : c0 + cn], data=convo[:, src_t, c0 : c0 + cn], op0=nl.multiply,
                    operand0=scale, op1=nl.multiply, operand1=rs[:, 0:cn],
                )

    # ---------------- chunk recurrence ----------------
    # _GRP: where the stationary operand is shared by the gqa v-heads of a key head (the 0/1 operators, k), the
    # heads' moving operands go side by side through ONE matmul (<= 512 fp32 columns, one PSUM bank, one
    # LDWEIGHTS): the cumsum rows (ones stationary, all heads, and all chunks while they fit),
    # W^T = k^T (bg T^T) and dS = k^T (ratio dlt). The per-token scales move from k onto T^T / dlt (same
    # number of bf16 roundings). Per-head stationaries (T^T, W^T, S, dlt) keep one matmul per head.
    o_blk = nl.ndarray((P, vh, NB), dtype=nl.bfloat16, buffer=nl.sbuf)  # core output, feature-major [v, h, tok]
    grp_rb = _GRP and vh * C <= MCH
    n_rb = max(1, min(NCH, MCH // (vh * C))) if grp_rb else 1  # chunks per cumsum matmul (a divisor of NCH)
    while NCH % n_rb:
        n_rb -= 1
    grp = _GRP and gqa * C <= MCH
    rb_all = None
    for ci in range(NCH):
        cs = slice(ci * C, (ci + 1) * C)
        # in-chunk cumulative gates: column form [t, h] and, per head, row form broadcast to all partitions
        # rbs[h][m, t] = gc_t (row form, every partition)
        if grp_rb:
            if ci % n_rb == 0:  # sum_s ones[s, m] g[s, c, h] utri[s, t]: 0/1 stationary, n_rb * vh * C columns
                nrc = n_rb * vh * C
                gu = nl.ndarray((P, n_rb, vh, C), dtype=nl.float32, buffer=nl.sbuf)
                _tt(gu, g_all.ap(pattern=[[NCH * vh, P], [vh, n_rb], [1, vh], [0, C]], offset=ci * vh),
                    c["utri"].ap(pattern=[[P, P], [0, n_rb], [0, vh], [1, C]], offset=0), nl.multiply)
                ps_rb = nl.ndarray((P, nrc), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(ps_rb, c["ones_f"], gu.reshape((P, nrc)), is_stationary_onezero=True)
                rb_all = nl.ndarray((P, n_rb, vh, C), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=rb_all.reshape((P, nrc)), src=ps_rb)
            rbs = [rb_all[:, ci % n_rb, h, :] for h in range(vh)]
        else:  # per head: fp32 matmul with the 0/1 cumsum operator as moving
            rbs = []
            for h in range(vh):
                ps_rb = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(ps_rb, g_all.ap(pattern=[[NCH * vh, P], [0, P]], offset=ci * vh + h), c["utri"],
                               is_moving_onezero=True)
                rb = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=rb, src=ps_rb)
                rbs.append(rb)
        gc = nl.ndarray((P, vh), dtype=nl.float32, buffer=nl.sbuf)
        if _GCDIAG:  # column form gc[t] = rb[t, t]: one DVE pass with a free-axis sum, no second fp32 matmul
            for h in range(vh):
                dg = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(dst=dg, data=rbs[h], op0=nl.multiply, operand0=1.0, op1=nl.multiply,
                                          operand1=c["ident"], reduce_op=nl.add, reduce_res=gc[:, h : h + 1])
        else:
            ps_gc = nl.ndarray((P, vh), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(ps_gc, c["utri"], g_all[:, ci, :], is_stationary_onezero=True)
            nisa.tensor_copy(dst=gc, src=ps_gc)
        ngc = nl.ndarray((P, vh), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ngc, data=gc, op0=nl.multiply, operand0=-1.0)
        gam = nl.ndarray((P, vh), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=gam, op=nl.exp, data=gc)
        bg = nl.ndarray((P, vh), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=bg, data1=gam, data2=beta_all[:, ci, :], op=nl.multiply)
        if _VT == "dma":  # token-major k (all key heads) and beta v (all v-heads): 2 DMA transposes, no PE
            k_all = nl.ndarray((P, kh, P), dtype=nl.bfloat16, buffer=nl.sbuf)  # [t, j, k]
            nisa.dma_transpose(dst=k_all, src=kq.ap(pattern=[[kh * 2 * NB, P], [2 * NB, kh], [1, C]], offset=ci * C),
                               axes=(2, 1, 0))
            v_all = nl.ndarray((P, vh, P), dtype=nl.bfloat16, buffer=nl.sbuf)  # [t, h, v]
            nisa.dma_transpose(
                dst=v_all, src=convo.ap(pattern=[[n_conv_t * NB, P], [NB, vh], [1, C]], offset=v_t0 * NB + ci * C),
                axes=(2, 1, 0),
            )
            bV_all = nl.ndarray((P, vh, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            _tt(bV_all, v_all, beta_all.ap(pattern=[[NCH * vh, P], [1, vh], [0, P]], offset=ci * vh), nl.multiply)
        for j in range(kh):
            if _VT == "dma":
                k_tok = k_all[:, j, :]
            else:
                tpk = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tpk, data=kq[:, j, 0, cs])
                k_tok = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)  # [t, k]
                _copy(k_tok, tpk)
            if m_tok is not None:  # pad rows' k = 0: they add nothing to the state update (and to W)
                k_m = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=k_m, op=nl.copy, data=k_tok, scale=m_tok[:, ci, :])
                k_tok = k_m
            ps_gk = nl.ndarray((P, 2, C), dtype=nl.float32, buffer=nl.psum)  # [t, (k.k_s | k.q_i)]
            nisa.nc_matmul(ps_gk, kq[:, j, 0, cs], kq[:, j, :, cs])
            G = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=G, src=ps_gk[:, 0, :])
            KQ = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=KQ, src=ps_gk[:, 1, :])
            if grp:
                TTs_g = nl.ndarray((P, gqa * P), dtype=nl.bfloat16, buffer=nl.sbuf)  # [t, (h, i)] bg_t T^T
                dlr_g = nl.ndarray((P, gqa * P), dtype=nl.bfloat16, buffer=nl.sbuf)  # [t, (h, v)] ratio_t dlt
            hd = []
            for hh in range(gqa):
                h = j * gqa + hh
                gcl = gc[:, h : h + 1]
                rb = rbs[h]  # rb[m, t] = gc_t
                eC = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)  # gam_C on every partition
                nisa.activation(dst=eC, op=nl.exp, data=rb[:, C - 1 : C])
                ratio = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)  # gam_C / gam_t
                nisa.activation(dst=ratio, op=nl.exp, data=gcl, scale=-1.0, bias=rb[:, C - 1 : C])
                # decay matrices, masked before the exp (masked entries -> exp(-1e30) = 0)
                Dm = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)  # [i, s] = exp(gc_i - gc_s), s < i
                _tt(Dm, c["lowneg"], rb, nl.subtract)
                nisa.activation(dst=Dm, op=nl.exp, data=Dm, bias=gcl)
                DT = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)  # [s, i] = exp(gc_i - gc_s), s <= i
                _tt(DT, rb, c["upneg"], nl.add)
                nisa.activation(dst=DT, op=nl.exp, data=DT, bias=ngc[:, h : h + 1])
                A = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(dst=A, data=G, op0=nl.multiply, operand0=beta_all[:, ci, h : h + 1],
                                          op1=nl.multiply, operand1=Dm)
                PT = nl.ndarray((P, C), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=PT, data1=KQ, data2=DT, op=nl.multiply)
                gam_row = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=gam_row, op=nl.exp, data=rb)
                QtT = nl.ndarray((P, C), dtype=nl.bfloat16, buffer=nl.sbuf)  # [k, i] = gam_i q_i
                _tt(QtT, kq[:, j, 1, cs], gam_row, nl.multiply)
                if _VT == "dma":
                    bV = bV_all[:, h, :]
                else:
                    tpv = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(dst=tpv, data=convo[:, v_t0 + h, cs])
                    bV = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)  # [t, v]
                    nisa.activation(dst=bV, op=nl.copy, data=tpv, scale=beta_all[:, ci, h : h + 1])
                Kh_ = None
                if not grp:
                    bgK = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)  # [t, k]
                    nisa.activation(dst=bgK, op=nl.copy, data=k_tok, scale=bg[:, h : h + 1])
                    Kh_ = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)  # [t, k] (gam_C / gam_t) k_t
                    nisa.activation(dst=Kh_, op=nl.copy, data=k_tok, scale=ratio)

                TT = _inverse_T(A, c)
                ps_u = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(ps_u, TT, bV)  # U = T (beta v)  [i, v]
                U = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
                _copy(U, ps_u)
                if grp:
                    WT = None
                    nisa.activation(dst=TTs_g[:, hh * P : (hh + 1) * P], op=nl.copy, data=TT, scale=bg[:, h : h + 1])
                else:
                    ps_w = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(ps_w, bgK, TT)  # W^T = (beta gam k)^T T^T  [k, i]
                    WT = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
                    _copy(WT, ps_w)
                hd.append((h, eC, ratio, PT, QtT, U, WT, Kh_))
            if grp:
                ps_w = nl.ndarray((P, gqa * P), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(ps_w, k_tok, TTs_g)  # W^T = k^T (bg T^T)  [k, (h, i)]
                WT_g = nl.ndarray((P, gqa * P), dtype=nl.bfloat16, buffer=nl.sbuf)
                _copy(WT_g, ps_w)

            # ---- state chain ----
            for hh, (h, eC, ratio, PT, QtT, U, WT, Kh_) in enumerate(hd):
                if grp:
                    WT = WT_g[:, hh * P : (hh + 1) * P]
                ps_ws = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(ps_ws, WT, S_b[:, h, :])  # W S  [i, v]
                ps_o = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(ps_o, S_b[:, h, :], QtT)  # (gam q S)^T  [v, i]
                dlt = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=dlt, data1=U, data2=ps_ws, op=nl.subtract)
                nisa.nc_matmul(ps_o, dlt, PT)  # + (P dlt)^T
                if grp:
                    nisa.activation(dst=dlr_g[:, hh * P : (hh + 1) * P], op=nl.copy, data=dlt, scale=ratio)
                else:
                    ps_ds = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(ps_ds, Kh_, dlt)  # [k, v]
                    nisa.scalar_tensor_tensor(dst=S_b[:, h, :], data=S_f[:, h, :], op0=nl.multiply, operand0=eC,
                                              op1=nl.add, operand1=ps_ds)
                    nisa.scalar_tensor_tensor(dst=S_f[:, h, :], data=S_f[:, h, :], op0=nl.multiply, operand0=eC,
                                              op1=nl.add, operand1=ps_ds)
                _copy(o_blk[:, h, cs], ps_o)
            if grp:
                ps_ds = nl.ndarray((P, gqa * P), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(ps_ds, k_tok, dlr_g)  # dS = k^T (ratio dlt)  [k, (h, v)]
                for hh, (h, eC, *_r) in enumerate(hd):
                    dsl = ps_ds[:, hh * P : (hh + 1) * P]
                    nisa.scalar_tensor_tensor(dst=S_b[:, h, :], data=S_f[:, h, :], op0=nl.multiply, operand0=eC,
                                              op1=nl.add, operand1=dsl)
                    nisa.scalar_tensor_tensor(dst=S_f[:, h, :], data=S_f[:, h, :], op0=nl.multiply, operand0=eC,
                                              op1=nl.add, operand1=dsl)

    # ---------------- gated RMSNorm over the head dim ----------------
    core = nl.ndarray((P, vh, NB), dtype=nl.bfloat16, buffer=nl.sbuf)
    for h in range(vh):
        for c0 in range(0, NB, MCH):
            cn = min(MCH, NB - c0)
            osq = nl.ndarray((P, MCH), dtype=nl.bfloat16, buffer=nl.sbuf)
            _tt(osq[:, 0:cn], o_blk[:, h, c0 : c0 + cn], o_blk[:, h, c0 : c0 + cn], nl.multiply)
            ms = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(ms[:, 0:cn], c["ones_b"], osq[:, 0:cn])
            rms = nl.ndarray((P, MCH), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=rms[:, 0:cn], op=nl.rsqrt, data=ms[:, 0:cn], scale=1.0 / HEAD, bias=eps)
            on = nl.ndarray((P, MCH), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=on[:, 0:cn], data1=o_blk[:, h, c0 : c0 + cn], data2=rms[:, 0:cn], op=nl.multiply)
            nisa.scalar_tensor_tensor(
                dst=core[:, h, c0 : c0 + cn], data=on[:, 0:cn], op0=nl.multiply, operand0=norm_sb, op1=nl.multiply,
                operand1=silu_z[:, h, c0 : c0 + cn],
            )

    # ---------------- out-projection: out[tok, N_out] = core^T w_out ----------------
    if wo is None:  # streamed: one DMA per block
        wo_b = nl.ndarray((P, vh, N_out), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wo_b, src=wo_pat)
    else:
        wo_b = wo
    GW = WM  # columns per PSUM group / matmul
    for ci in range(NCH):
        o_sb = nl.ndarray((P, N_out), dtype=nl.bfloat16, buffer=nl.sbuf)
        for g0 in range(0, N_out, GW):
            g1 = min(N_out, g0 + GW)
            pso = nl.ndarray((P, g1 - g0), dtype=nl.bfloat16 if (_OBF16 and vh == 1) else nl.float32, buffer=nl.psum)
            for h in range(vh):
                nisa.nc_matmul(pso, core[:, h, ci * C : (ci + 1) * C], wo_b[:, h, g0:g1])
            for e0 in range(g0, g1, MCH):
                e1 = min(g1, e0 + MCH)
                _copy(o_sb[:, e0:e1], pso[:, e0 - g0 : e1 - g0])
        nisa.dma_copy(dst=_hbm_rows(out, row0 + ci * C, C, N_out, blk, NB), src=o_sb)
