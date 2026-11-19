"""Qwen3.5 gated full-attention PREFILL for one attention TP rank (NKI, LNC=1).

    out_partial[L, N] = o_proj( causal_attention(q, K, V) * sigmoid(x W_gate) )     (this rank's heads)

One sequence of L new tokens (positions ``start .. start+L-1``) per call, ``nq`` query heads sharing
one KV head (TP8: nq=4). The K/V of the L tokens are written into decode's slot-contiguous cache
layouts, ``k_cache_t[slot, :, start:start+L]`` (K transposed) and ``v_cache[slot, start:start+L, :]``,
so ``attn_decode_tkg`` continues from position ``start + L`` with no relayout.

Three phases, all SBUF/PSUM placed by hand (the reused attention core places its buffers by hand
from address 0, so the whole kernel uses direct allocation and each phase owns all of SBUF):

  A. projections, 512-token blocks. x^T of the block is built with PE transposes; then for every
     512-column weight chunk (w_qkv | w_gate, streamed, double buffered) each 128-token tile keeps
     its x^T slice stationary and streams the chunk (N = 512 moving columns per stationary, i.e.
     the stationary load is amortised over a full PSUM bank). A 512-column chunk is exactly two
     256-wide heads, so post-processing runs straight off each PSUM bank: bf16 rounding (as HF),
     zero-centred q/k RMSNorm, partial RoPE (first 64 dims, rotate-half), then
       q    -> HBM scratch q[h, t, :]           (token-major, attention_cte tp_q layout)
       k    -> PE transpose -> k^T scratch [1, d, L] and k_cache_t[slot, :, start + t]
       v    -> v scratch [1, L, d] and v_cache[slot, start + t, :]
       gate -> raw bf16 gate scratch [L, nq*d]
  B. causal attention: nkilib ``attention_cte`` (flash-style: Q group of 128 stationary, K/V tiles
     streamed, fp32 online softmax, causal tile skipping, ACT+DVE exp split on trn3; supports
     head_dim 256 and native GQA: the nq q heads are the batch, the single KV head is batch_kv = 1).
     For continued prefill (start > 0) the slot's cached history 0..start-1 is the prefix-caching
     prior (k_prior/v_prior + prior_used_len), so tokens attend to it plus the causal new block.
  C. gate + o_proj per 128-token tile: attention rows (token-major) * sigmoid(gate) -> bf16,
     8 PE transposes, then the tile stays stationary and the resident w_o streams through in
     512-column chunks -> token-major partial [128, N].

Numerics follow HF / ``attn_decode_torch``: projections leave the PE in bf16, norms/RoPE/softmax in
fp32, attention output, gated product and o_proj output in bf16. The only extra rounding vs the
reference is attention_cte's bf16 probabilities before P.V (relative ~2^-9 per term, averaging).

Inputs (one rank, one sequence):
  x          [L, H] bf16 (post input-RMSNorm)
  w_qkv      [H, (nq+2)*d] bf16  q heads | k | v (natural channel order, as decode)
  q_norm_w   [1, d] fp32 raw zero-centred (kernel uses 1 + w);  k_norm_w likewise
  w_gate_p   [128, H/128, nq*d] bf16 (decode layout: row h1*128 + p on partition p)
  cos, sin   [L, 32] fp32  cos/sin((start + t) * inv_freq)
  k_cache_t  [B, d, S] bf16 (in/out) — or [B, S, d], V's token-major layout, when the pool's K pages are
             token-major (``k_tm``; SGLANG_NEURON_QWEN35_KV_TM, the kv_cp=1 PD prefill serving the kv_cp
             decode's scatter; every K DMA then addresses the page with 2-byte-granular strided patterns like
             V's, the SBUF values bit-identical) — v_cache [B, S, d] bf16 (in/out), S >= start + L
  slot       [1, 1] int32 runtime cache slot of this sequence
  w_o        [nq*d, N] bf16 (N = H, or wider with folded columns as decode supports)
  start      compile-time int, tokens already in the slot's cache (0 = fresh prompt)
Returns (out [L, N] bf16, k_cache_t, v_cache).

Paged mode (edit 0002 in ../PROVENANCE.md; ``write_pages`` given, the caches are page pools ``k_cache_t [NPAGE, d, PG]``
/ ``v_cache [NPAGE, PG, d]``, attn_decode_tkg's "dp" page layout): the chunk starts at a runtime position that is a
multiple of PG, so ``start`` stays 0 and ``slot`` is unused.
  write_pages  [1, L/PG] int32  pool page of the chunk's rows j*PG .. j*PG+PG-1 (page 0 past the request's pages)
  prior_pages  [1, MPP] int32   (optional) the request's pages 0 .. MPP-1, padded with page 0 (finite, masked)
  prior_len    [1, 1] int32     tokens before the chunk (<= MPP*PG): attention_cte's dynamic prior_used_len
  cos, sin     [L, 32]          of the chunk's positions prior_len + t
K/V go to their pages with the page id as each DMA's scalar offset (offsets inside a page are static); the prior is
gathered page by page into the static [1, d, MPP*PG] / [1, MPP*PG, d] scratch (the compile-time prior capacity).
"""

import os

import nki
import nki.isa as nisa
import nki.language as nl

from .core.attention.attention_cte import _attention_cte
from nkilib.core.utils.kernel_assert import kernel_assert

# token-major K pages [.., tokens, head_dim] (sglang_neuron models/qwen35/config.KV_TM_ENV): the kv_cp=1 PD
# prefill serving a kv_cp decode's scatter stores K like V so a CP rank's block of a page is one contiguous run
# on both roles (sglang_patches/kv_cp_scatter). Every K DMA below addresses the token-major page with the same
# 2-byte-granular strided patterns as V's, so the SBUF values are bit-identical to the dim-major kernel's.
_K_TM = os.environ.get("SGLANG_NEURON_QWEN35_KV_TM", "0") == "1"

P = 128
ROT = 64  # rotary dims (partial_rotary_factor 0.25 of head_dim 256)
MCH = 512  # weight / output column chunk = one fp32 PSUM bank
BANK = 2048  # PSUM bank bytes per partition
SBUF_LIMIT = 180 * 1024  # bytes per partition this kernel lets a phase claim
TBLK = 512  # phase-A token block


class _Bump:
    """Direct SBUF allocation: a bump pointer (bytes per partition), 64 B aligned."""

    def __init__(self, base=0):
        self.addr = base

    def take(self, nbytes):
        a = self.addr
        self.addr = a + ((nbytes + 63) // 64) * 64
        return a


def _sb(shape, dtype, addr):
    return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf, address=(0, addr))


def _ps(shape, dtype, bank, off=0):
    return nl.ndarray(shape, dtype=dtype, buffer=nl.psum, address=(0, bank * BANK + off))


def _page_segs(s0, sw, PG):
    """Split chunk rows [s0, s0 + sw) at page boundaries: [(page j, offset in page, width, offset in range)]."""
    segs = []
    s = s0
    while s < s0 + sw:
        jp = s // PG
        o = s - jp * PG
        w = min(PG - o, s0 + sw - s)
        segs.append((jp, o, w, s - s0))
        s += w
    return segs


def _paged_prior(k_cache_t, v_cache, prior_pages, prior_len, MPP, PG, d, DH, base, k_tm=False):
    """The request's pages 0 .. MPP-1 gathered into attention_cte's contiguous prior scratch (phase-A SBUF from
    ``base`` is dead here), and prior_len as its dynamic prior_used_len."""
    bf16 = nl.bfloat16
    cap = MPP * PG
    k_prior = nl.ndarray((1, d, cap), dtype=bf16, buffer=nl.private_hbm)
    v_prior = nl.ndarray((1, cap, d), dtype=bf16, buffer=nl.private_hbm)
    pb = _Bump(base)
    pp = _sb((1, MPP), nl.int32, pb.take(MPP * 4))
    nisa.dma_copy(dst=pp, src=prior_pages)
    kb = _sb((P, PG), bf16, pb.take(PG * 2))
    vb = _sb((P, PG // P, d), bf16, pb.take(PG // P * d * 2))
    for jp in range(MPP):
        page = pp[0:1, jp : jp + 1]
        if k_tm:
            # token-major K page [PG, d]: one contiguous burst read (V's form below; a strided
            # [[1, P], [d, PG]] read would step 2 bytes at a time, much slower on silicon) into the
            # staging tile V is about to reuse, then transpose each P-token chunk on-chip into the
            # same [P, PG] dim tiles the dim-major read lands
            nisa.dma_copy(dst=vb, src=k_cache_t.ap(pattern=[[d, P], [P * d, PG // P], [1, d]], offset=0,
                                                   scalar_offset=page, indirect_dim=0))
            for j in range(DH):
                for cc in range(PG // P):
                    tp = _ps((P, P), bf16, 2)
                    nisa.nc_transpose(dst=tp, data=vb[:, cc, j * P : (j + 1) * P])
                    nisa.tensor_copy(dst=kb[:, cc * P : (cc + 1) * P], src=tp)
                nisa.dma_copy(dst=k_prior.ap(pattern=[[cap, P], [1, PG]], offset=j * P * cap + jp * PG),
                              src=kb)
        else:
            for j in range(DH):
                nisa.dma_copy(dst=kb, src=k_cache_t.ap(pattern=[[PG, P], [1, PG]], offset=j * P * PG,
                                                       scalar_offset=page, indirect_dim=0))
                nisa.dma_copy(dst=k_prior.ap(pattern=[[cap, P], [1, PG]], offset=j * P * cap + jp * PG),
                              src=kb)
        nisa.dma_copy(dst=vb, src=v_cache.ap(pattern=[[d, P], [P * d, PG // P], [1, d]], offset=0, scalar_offset=page,
                                             indirect_dim=0))
        nisa.dma_copy(dst=v_prior.ap(pattern=[[d, P], [P * d, PG // P], [1, d]], offset=jp * PG * d), src=vb)
    return k_prior, v_prior, prior_len.reshape((1,))


@nki.jit
def attn_prefill_tkg(
    x,
    w_qkv,
    q_norm_w,
    k_norm_w,
    w_gate_p,
    cos,
    sin,
    k_cache_t,
    v_cache,
    slot,
    w_o,
    start=0,
    write_pages=None,
    prior_pages=None,
    prior_len=None,
    k_tm=None,
):
    if k_tm is None:
        k_tm = _K_TM  # trace-time: the layout of the pool this kernel is compiled against
    L, H = x.shape
    d = v_cache.shape[2]
    S = v_cache.shape[1]
    nq = w_gate_p.shape[2] // d
    nd = nq * d
    NQKV = (nq + 2) * d
    H1 = H // P
    DH = d // P
    half = ROT // 2
    n_ot = nd // P
    N = w_o.shape[1]
    TB = min(TBLK, L)
    NT = TB // P
    NBLK = L // TB
    NCQ = NQKV // MCH  # q/k/v chunks: nq/2 q chunks, then [k | v]
    NCG = nd // MCH  # gate chunks
    bf16 = nl.bfloat16
    f32 = nl.float32
    kernel_assert(d == 256, f"head_dim must be 256 (two heads per 512-column chunk), got {d}")
    kernel_assert(nq % 2 == 0, f"nq must be even, got {nq}")
    kernel_assert(H % P == 0 and L % P == 0 and L % TB == 0, f"need H%128==0, L%128==0, L%512==0 or L<512 (L={L})")
    paged = write_pages is not None
    PG = S  # page tokens in paged mode
    NWP = L // PG if paged else 0
    MPP = prior_pages.shape[1] if paged and prior_pages is not None else 0
    if paged:
        kernel_assert(start == 0, "paged prefill takes its start at run time (prior_len)")
        kernel_assert(PG % P == 0 and L % PG == 0, f"page {PG} must be a multiple of 128 dividing the chunk {L}")
        kernel_assert(tuple(write_pages.shape) == (1, NWP), f"write_pages must be [1, {NWP}]")
        kernel_assert((prior_pages is None) == (prior_len is None), "prior_pages and prior_len go together")
    kernel_assert(start + L <= S or paged, f"start + L = {start + L} exceeds cache length {S}")
    kernel_assert(start % P == 0, "start must be a multiple of 128")
    if k_tm:
        kernel_assert(k_cache_t.shape[1] == S and k_cache_t.shape[2] == d,
                      "k_cache_t must be [B, S, d] (token-major K)")
    else:
        kernel_assert(k_cache_t.shape[1] == d and k_cache_t.shape[2] == S, "k_cache_t must be [B, d, S]")

    # ---------------- HBM scratch ----------------
    q_hbm = nl.ndarray((nq, L, d), dtype=bf16, buffer=nl.private_hbm)
    k_hbm = nl.ndarray((1, d, L), dtype=bf16, buffer=nl.private_hbm)
    v_hbm = nl.ndarray((1, L, d), dtype=bf16, buffer=nl.private_hbm)
    g_hbm = nl.ndarray((L, nd), dtype=bf16, buffer=nl.private_hbm)
    attn_hbm = nl.ndarray((nq, L, d), dtype=bf16, buffer=nl.private_hbm)

    # =====================================================================================
    # Phase A: QKV + gate projections, q/k norm, RoPE, KV-cache writes
    # =====================================================================================
    al = _Bump()
    slot_sb = _sb((1, 1), nl.int32, al.take(4))
    wp_sb = None
    if paged:
        wp_sb = _sb((1, NWP), nl.int32, al.take(NWP * 4))
        nisa.dma_copy(dst=wp_sb, src=write_pages)
    else:
        nisa.dma_copy(dst=slot_sb, src=slot)
    wnq = _sb((P, d), f32, al.take(d * 4))
    wnk = _sb((P, d), f32, al.take(d * 4))
    nisa.dma_copy(dst=wnq, src=q_norm_w.ap(pattern=[[0, P], [1, d]], offset=0))
    nisa.dma_copy(dst=wnk, src=k_norm_w.ap(pattern=[[0, P], [1, d]], offset=0))
    nisa.tensor_scalar(dst=wnq, data=wnq, op0=nl.add, operand0=1.0)
    nisa.tensor_scalar(dst=wnk, data=wnk, op0=nl.add, operand0=1.0)
    A_XS = al.take(2 * H * 2)
    A_XT = al.take(H1 * TB * 2)
    A_W = al.take(2 * H1 * MCH * 2)
    A_CS = al.take(2 * NT * 2 * 2 * half * 4)
    A_QB = al.take(3 * MCH * 2)
    A_JK = al.take(2 * d * 2)
    A_SS = al.take(2 * 64)
    A_RS = al.take(2 * 64)
    A_NB = al.take(2 * 2 * d * 2)
    A_RT = al.take(2 * 4 * 2 * half * 4)
    A_KT = al.take(2 * DH * TB * 2)
    kernel_assert(al.addr <= SBUF_LIMIT, f"phase A SBUF {al.addr} B/partition exceeds {SBUF_LIMIT}")

    xT = _sb((P, H1, TB), bf16, A_XT)
    ctr = 0  # rotating PSUM accumulator index
    wctr = 0  # weight chunk buffer index
    qctr = 0  # post-processing buffer index
    for blk in range(NBLK):
        t0 = blk * TB
        # ---- x^T of the block: 8 PE transposes per PSUM bank, one copy per bank ----
        for i in range(NT):
            xs = _sb((P, H), bf16, A_XS + (i % 2) * H * 2)
            nisa.dma_copy(dst=xs, src=x[t0 + i * P : t0 + (i + 1) * P, :])
            for g0 in range(0, H1, 8):
                gn = min(8, H1 - g0)
                tp = _ps((P, 8, P), bf16, (g0 // 8) % 2)
                for k in range(gn):
                    nisa.nc_transpose(dst=tp[:, k, :], data=xs[:, (g0 + k) * P : (g0 + k + 1) * P])
                nisa.tensor_copy(dst=xT[:, g0 : g0 + gn, i * P : (i + 1) * P], src=tp[:, 0:gn, :])
        # ---- cos / sin of the block's tokens, broadcast over the two heads of a chunk ----
        csb = _sb((P, NT, 2, 2, half), f32, A_CS + (blk % 2) * NT * 2 * 2 * half * 4)
        for i in range(NT):
            tok = t0 + i * P
            nisa.dma_copy(dst=csb[:, i, 0, :, :], src=cos.ap(pattern=[[half, P], [0, 2], [1, half]], offset=tok * half))
            nisa.dma_copy(dst=csb[:, i, 1, :, :], src=sin.ap(pattern=[[half, P], [0, 2], [1, half]], offset=tok * half))
        kTb = _sb((P, DH, TB), bf16, A_KT + (blk % 2) * DH * TB * 2)

        for c in range(NCQ + NCG):
            wq = _sb((P, H1, MCH), bf16, A_W + (wctr % 2) * H1 * MCH * 2)
            wctr += 1
            if c < NCQ:
                nisa.dma_copy(dst=wq, src=w_qkv.ap(pattern=[[NQKV, P], [P * NQKV, H1], [1, MCH]], offset=c * MCH))
            else:
                g = c - NCQ
                nisa.dma_copy(dst=wq, src=w_gate_p[:, :, g * MCH : (g + 1) * MCH])
            for i in range(NT):
                tok = t0 + i * P
                acc = _ps((P, MCH), f32, 3 + ctr % 5)
                ctr += 1
                for h1 in range(H1):
                    nisa.nc_matmul(acc, xT[:, h1, i * P : (i + 1) * P], wq[:, h1, :])
                qb = _sb((P, MCH), bf16, A_QB + (qctr % 3) * MCH * 2)
                nisa.tensor_copy(dst=qb, src=acc)  # projections leave the PE in bf16 (HF)
                sl = qctr % 2
                qctr += 1
                if c >= NCQ:
                    g = c - NCQ
                    nisa.dma_copy(dst=g_hbm[tok : tok + P, g * MCH : (g + 1) * MCH], src=qb)
                    continue
                is_kv = c == NCQ - 1
                nh = 1 if is_kv else 2  # normed heads in this chunk (k only, or two q heads)
                # ---- zero-centred RMSNorm per head (fp32 math, bf16 out) ----
                ss = _sb((P, 2), f32, A_SS + sl * 64)
                for e in range(nh):
                    jk = _sb((P, d), bf16, A_JK + sl * d * 2)
                    nisa.activation_reduce(
                        dst=jk, op=nl.square, data=qb[:, e * d : (e + 1) * d], reduce_op=nl.add,
                        reduce_res=ss[:, e : e + 1],
                    )
                rs = _sb((P, 2), f32, A_RS + sl * 64)
                nisa.activation(dst=rs[:, 0:nh], op=nl.rsqrt, data=ss[:, 0:nh], scale=1.0 / d, bias=1e-6)
                nb = _sb((P, 2, d), bf16, A_NB + sl * 2 * d * 2)
                for e in range(nh):
                    nisa.scalar_tensor_tensor(
                        dst=nb[:, e, :], data=qb[:, e * d : (e + 1) * d], op0=nl.multiply, operand0=rs[:, e : e + 1],
                        op1=nl.multiply, operand1=wnk if is_kv else wnq,
                    )
                # ---- partial RoPE on dims [0, ROT): rotate-half over those ROT dims ----
                rbase = A_RT + sl * 4 * 2 * half * 4
                t1 = _sb((P, 2, half), f32, rbase)
                t2 = _sb((P, 2, half), f32, rbase + 2 * half * 4)
                r1 = _sb((P, 2, half), f32, rbase + 2 * 2 * half * 4)
                r2 = _sb((P, 2, half), f32, rbase + 3 * 2 * half * 4)
                x1 = nb[:, 0:nh, 0:half]
                x2 = nb[:, 0:nh, half:ROT]
                cc = csb[:, i, 0, 0:nh, :]
                sn = csb[:, i, 1, 0:nh, :]
                nisa.tensor_tensor(dst=t1[:, 0:nh, :], data1=x1, data2=cc, op=nl.multiply)
                nisa.tensor_tensor(dst=t2[:, 0:nh, :], data1=x2, data2=sn, op=nl.multiply)
                nisa.tensor_tensor(dst=r1[:, 0:nh, :], data1=t1[:, 0:nh, :], data2=t2[:, 0:nh, :], op=nl.subtract)
                nisa.tensor_tensor(dst=t1[:, 0:nh, :], data1=x2, data2=cc, op=nl.multiply)
                nisa.tensor_tensor(dst=t2[:, 0:nh, :], data1=x1, data2=sn, op=nl.multiply)
                nisa.tensor_tensor(dst=r2[:, 0:nh, :], data1=t1[:, 0:nh, :], data2=t2[:, 0:nh, :], op=nl.add)
                nisa.tensor_copy(dst=nb[:, 0:nh, 0:half], src=r1[:, 0:nh, :])
                nisa.tensor_copy(dst=nb[:, 0:nh, half:ROT], src=r2[:, 0:nh, :])
                if not is_kv:
                    # q heads 2c, 2c+1 -> q_hbm[h, tok:tok+128, :]
                    nisa.dma_copy(
                        dst=q_hbm.ap(pattern=[[d, P], [L * d, 2], [1, d]], offset=(2 * c) * L * d + tok * d), src=nb
                    )
                else:
                    # K^T into the block buffer (written once per block, 1 KB runs)
                    kp = _ps((P, DH, P), bf16, 2, (i % 2) * DH * P * 2)
                    for j in range(DH):
                        nisa.nc_transpose(dst=kp[:, j, :], data=nb[:, 0, j * P : (j + 1) * P])
                    nisa.tensor_copy(dst=kTb[:, :, i * P : (i + 1) * P], src=kp)
                    # V (no norm / RoPE): scratch + cache row start + tok of the slot
                    v_t = qb[:, d : 2 * d]
                    nisa.dma_copy(dst=v_hbm[0, tok : tok + P, :], src=v_t)
                    if paged:  # row tok % PG of page write_pages[tok // PG]
                        jp = tok // PG
                        v_dst = v_cache.ap(pattern=[[d, P], [1, d]], offset=(tok - jp * PG) * d,
                                           scalar_offset=wp_sb[0:1, jp : jp + 1], indirect_dim=0)
                    else:
                        v_dst = v_cache.ap(
                            pattern=[[d, P], [1, d]], offset=(start + tok) * d, scalar_offset=slot_sb, indirect_dim=0
                        )
                    nisa.dma_copy(dst=v_dst, src=v_t)
                    if k_tm:
                        # token-major K pool rows take the post-norm/RoPE nb tile [P tokens, d] directly
                        # in V's burst form — writing the [P dims, w tokens] kTb tile instead would
                        # free-stride d = 2-byte steps, much slower on silicon. kTb is still built for
                        # the dim-major local-attention scratch below.
                        if paged:
                            k_dst = k_cache_t.ap(pattern=[[d, P], [1, d]], offset=(tok - jp * PG) * d,
                                                 scalar_offset=wp_sb[0:1, jp : jp + 1], indirect_dim=0)
                        else:
                            k_dst = k_cache_t.ap(pattern=[[d, P], [1, d]], offset=(start + tok) * d,
                                                 scalar_offset=slot_sb, indirect_dim=0)
                        nisa.dma_copy(dst=k_dst, src=nb[:, 0, :])
        # ---- block's K^T -> scratch and k_cache_t's K rows of the block's tokens ----
        # kTb[:, j, :] is the block's [dim j block, token] tile. A token-major pool (k_tm) took its K
        # rows from nb in the microblock loop above; the dim-major pool takes kTb with the same
        # 2-byte-granular strided DMA V uses (partition = dim, free = token).
        for j in range(DH):
            nisa.dma_copy(dst=k_hbm.ap(pattern=[[L, P], [1, TB]], offset=j * P * L + t0), src=kTb[:, j, :])
            if k_tm:
                continue
            if paged:  # one DMA per page segment of the block
                for jp, o, w, dc in _page_segs(t0, TB, PG):
                    nisa.dma_copy(
                        dst=k_cache_t.ap(pattern=[[PG, P], [1, w]], offset=j * P * PG + o,
                                         scalar_offset=wp_sb[0:1, jp : jp + 1], indirect_dim=0),
                        src=kTb[:, j, dc : dc + w],
                    )
                continue
            nisa.dma_copy(
                dst=k_cache_t.ap(pattern=[[S, P], [1, TB]], offset=j * P * S + start + t0,
                                 scalar_offset=slot_sb, indirect_dim=0),
                src=kTb[:, j, :],
            )

    # ---------------- continued prefill: the slot's history is the attention prior ----------------
    k_prior = None
    v_prior = None
    pul = None
    if MPP:
        k_prior, v_prior, pul = _paged_prior(k_cache_t, v_cache, prior_pages, prior_len, MPP, PG, d, DH, A_XS,
                                             k_tm=k_tm)
    elif start > 0:
        k_prior = nl.ndarray((1, d, start), dtype=bf16, buffer=nl.private_hbm)
        v_prior = nl.ndarray((1, start, d), dtype=bf16, buffer=nl.private_hbm)
        pul = nl.ndarray((1,), dtype=nl.int32, buffer=nl.private_hbm)
        PCH = 2048
        pb = _Bump(A_XS)  # phase-A buffers are dead here
        kb = _sb((P, PCH), bf16, pb.take(PCH * 2))
        vb = _sb((P, PCH // P, d), bf16, pb.take(PCH // P * d * 2))
        pu = _sb((1, 1), nl.int32, pb.take(4))
        if k_tm:
            # token-major slot K [S, d]: contiguous burst reads (V's form; the strided [[1, P], [d, sw]]
            # read would step 2 bytes at a time, much slower on silicon) into the staging tile V is
            # about to reuse, then transpose each P-token chunk on-chip into the same [P, sw] dim tiles
            # the dim-major read lands (start % P == 0, so every chunk is a full 128 tokens)
            for s0 in range(0, start, PCH):
                sw = min(PCH, start - s0)
                fc = sw // P
                nisa.dma_copy(dst=vb[:, :fc, :],
                              src=k_cache_t.ap(pattern=[[d, P], [P * d, fc], [1, d]], offset=s0 * d,
                                               scalar_offset=slot_sb, indirect_dim=0))
                for j in range(DH):
                    for cc in range(fc):
                        tp = _ps((P, P), bf16, 2)
                        nisa.nc_transpose(dst=tp, data=vb[:, cc, j * P : (j + 1) * P])
                        nisa.tensor_copy(dst=kb[:, cc * P : (cc + 1) * P], src=tp)
                    nisa.dma_copy(dst=k_prior.ap(pattern=[[start, P], [1, sw]], offset=j * P * start + s0),
                                  src=kb[:, 0:sw])
        else:
            for j in range(DH):
                for s0 in range(0, start, PCH):
                    sw = min(PCH, start - s0)
                    nisa.dma_copy(
                        dst=kb[:, 0:sw],
                        src=k_cache_t.ap(pattern=[[S, P], [1, sw]], offset=j * P * S + s0,
                                         scalar_offset=slot_sb, indirect_dim=0),
                    )
                    nisa.dma_copy(dst=k_prior.ap(pattern=[[start, P], [1, sw]], offset=j * P * start + s0),
                                  src=kb[:, 0:sw])
        for r0 in range(0, start, P):
            nisa.dma_copy(
                dst=vb[:, 0, :],
                src=v_cache.ap(pattern=[[d, P], [1, d]], offset=r0 * d, scalar_offset=slot_sb, indirect_dim=0),
            )
            nisa.dma_copy(dst=v_prior[0, r0 : r0 + P, :], src=vb[:, 0, :])
        nisa.iota(pu, pattern=[[0, 1]], offset=start, channel_multiplier=0)
        nisa.dma_copy(dst=pul.reshape((1, 1)), src=pu)

    # =====================================================================================
    # Phase B: causal flash attention (nkilib attention_cte), GQA nq:1, head_dim 256
    # =====================================================================================
    _attention_cte(
        q=q_hbm,
        k=k_hbm,
        v=v_hbm,
        scale=float(d) ** -0.5,  # 1/16: power of two, folded into Q exactly
        causal_mask=True,
        k_prior=k_prior,
        v_prior=v_prior,
        prior_used_len=pul,
        tp_q=True,
        tp_k=False,
        tp_out=False,
        out_o_hbm=attn_hbm,
    )

    # =====================================================================================
    # Phase C: sigmoid gate + o_proj, 128-token tiles, w_o resident
    # =====================================================================================
    cl = _Bump()
    C_WO = cl.take(n_ot * N * 2)
    per_slot = nd * 2 * 4 + nd * 4 + N * 2  # a, g, og, oT (bf16) + sg (fp32) + o_sb
    nbuf = 2 if cl.addr + 2 * per_slot <= SBUF_LIMIT else 1
    C_A = cl.take(nbuf * nd * 2)
    C_G = cl.take(nbuf * nd * 2)
    C_SG = cl.take(nbuf * nd * 4)
    C_OG = cl.take(nbuf * nd * 2)
    C_OT = cl.take(nbuf * nd * 2)
    C_O = cl.take(nbuf * N * 2)
    kernel_assert(cl.addr <= SBUF_LIMIT, f"phase C SBUF {cl.addr} B/partition exceeds {SBUF_LIMIT} (w_o resident)")
    wo = _sb((P, n_ot, N), bf16, C_WO)
    nisa.dma_copy(dst=wo, src=w_o.ap(pattern=[[N, P], [P * N, n_ot], [1, N]], offset=0))
    out = nl.ndarray((L, N), dtype=bf16, buffer=nl.shared_hbm)
    NTB = (n_ot + 7) // 8  # PSUM banks of og transposes per tile
    octr = 0
    for t in range(L // P):
        tok = t * P
        s = t % nbuf
        a_sb = _sb((P, nq, d), bf16, C_A + s * nd * 2)
        nisa.dma_copy(dst=a_sb, src=attn_hbm.ap(pattern=[[d, P], [L * d, nq], [1, d]], offset=tok * d))
        g_sb = _sb((P, nd), bf16, C_G + s * nd * 2)
        nisa.dma_copy(dst=g_sb, src=g_hbm[tok : tok + P, :])
        sg = _sb((P, nd), f32, C_SG + s * nd * 4)
        nisa.activation(dst=sg, op=nl.sigmoid, data=g_sb)
        og = _sb((P, nd), bf16, C_OG + s * nd * 2)
        nisa.tensor_tensor(dst=og, data1=a_sb.reshape((P, nd)), data2=sg, op=nl.multiply)
        oT = _sb((P, n_ot, P), bf16, C_OT + s * nd * 2)
        for b8 in range(NTB):
            k0 = b8 * 8
            kn = min(8, n_ot - k0)
            tpo = _ps((P, 8, P), bf16, (t * NTB + b8) % 2)
            for k in range(kn):
                nisa.nc_transpose(dst=tpo[:, k, :], data=og[:, (k0 + k) * P : (k0 + k + 1) * P])
            nisa.tensor_copy(dst=oT[:, k0 : k0 + kn, :], src=tpo[:, 0:kn, :])
        o_sb = _sb((P, N), bf16, C_O + s * N * 2)
        for c0 in range(0, N, MCH):
            cn = min(MCH, N - c0)
            ps_o = _ps((P, MCH), f32, 2 + octr % 6)
            octr += 1
            for k in range(n_ot):
                nisa.nc_matmul(ps_o[:, 0:cn], oT[:, k, :], wo[:, k, c0 : c0 + cn])
            nisa.tensor_copy(dst=o_sb[:, c0 : c0 + cn], src=ps_o[:, 0:cn])
        nisa.dma_copy(dst=out[tok : tok + P, :], src=o_sb)
    return out, k_cache_t, v_cache
