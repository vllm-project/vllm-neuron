# SPDX-License-Identifier: Apache-2.0
"""The DSpark drafter -- `model.forward_spec` for one sequence -- as a single NKI kernel:
the flat scheme's drafter, launched once per sequence. The pipeline scheme's copy
(`../tp8pp8/draft_kernel.py`) drafts every sequence of a step in one launch; this
one is the pre-batching kernel kept as it was, so the flat scheme's path is unchanged.

    main_x   = main_norm(main_proj(main_hidden))                 the three taps, fp8 GEMM
    x        = embed([seed, noise x (K-1)]) expanded to hc copies  vocab-parallel, all_reduce
    for each of the 3 MTP stages:
        write this step's main-stream KV into the stage's window cache
        (prefill, start_pos == 0: that is all the stage does)
        hc_in -> draft attention over [window | the draft block] -> all_reduce(attn)
              -> hc_out -> hc_in -> 128-expert top-3 MoE -> all_reduce(world) -> hc_out
    head: collapse -> norm -> vocab-parallel head; then, token by token,
          markov bias (vocab-parallel embedding + head), argmax, confidence.

The draft position (`pos + tap`) is data dependent -- `tap` comes out of the
rejection sampler -- so every index that depends on it is computed here: the draft
rows' positions, the window's validity base and the window's cache slots
(`block_table[p // bs] * bs + p % bs`).

The argmax over the full vocabulary never gathers the logits: each rank takes the
(max, index) of its own vocabulary shard and an all_gather of those pairs decides
(ties to the lowest rank, i.e. the lowest vocabulary index, as `argmax` does).

Per-stage weights arrive stacked on a leading axis of 3, except the two kinds the
helpers read through an access pattern (`fn.T`, `gate.T`), which come one per stage.
The fp8 / fp4 weights are in the MX layout (`mk_model.mx_fp8` / `mx_fp4`: `[K/4, N]`
words) for the tensor engine's block-scaled matmul (`mk_mx`); the routed experts run
token-grouped (`mk_moe`); the attention selects its keys on chip (`attn_token_body_ti`).
`flags` = (at_zero, K, win, stage, kver, noise_id, vocab_size, topk, block_size);
stage 9 is the full drafter, 1..3 return the stream after that MTP stage, 4 the
masked local logits and the collapsed hidden. Outputs: `output_ids [1, K+1]` int32,
`confidence [K, 1]` fp32, `dbg`, and the three window caches (aliased, mutated in
place).
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

from . import mk_moe
from . import mk_mx as X
from .mk_common import (
    KTILE,
    PT,
    add_rows,
    attn_index_table,
    attn_token_body_ti,
    bcast_row,
    build_kT,
    cast_rows,
    cdiv,
    engram_gather,
    gather_rows,
    gemm_fp32,
    hc_in_tile,
    hc_out_tile,
    load_fn_t,
    norm_rows,
    pad_to,
    rope_rows,
    route_topk,
    scatter_rows,
    swiglu_rows,
    tiles_of,
)
from .mk_compress import bf16_gemm_rows

CORES = 2
SWIGLU_LIMIT = 10.0
ROUTE_SCALE = 1.5
NEG = -1.0e30


def _half(n, pid):
    tiles = n // PT
    a = (tiles + 1) // 2
    return (0, a * PT) if pid == 0 else (a * PT, n)


def _log2(n):
    k = 0
    while (1 << k) < n:
        k += 1
    assert (1 << k) == n, f"{n} is not a power of two"
    return k


def draft_body(
    meta, main_hidden, row_pos, write_slots, block_table,
    main_proj, main_proj_s, main_norm_w, embed_w, vocab_start,
    hca_fn_t0, hca_fn_t1, hca_fn_t2, hca_base, hca_scale, attn_norm_w,
    hcf_fn_t0, hcf_fn_t1, hcf_fn_t2, hcf_base, hcf_scale, ffn_norm_w,
    wq_a, wq_a_s, q_norm_w, wq_b, wq_b_s, wkv, wkv_s, kv_norm_w, wo_a, wo_a_s, wo_b, wo_b_s,
    attn_sink, freqs,
    gate_w_t0, gate_w_t1, gate_w_t2, gate_b, expert_lo,
    w1, s1, w3, s3, w2, s2, sh_w1, sh_s1, sh_w3, sh_s3, sh_w2, sh_s2,
    cache0, cache1, cache2,
    fnorm_w, head_w, mk_embed_w, mk_head_w, conf_w,
    groups_attn, groups_world, flags,
    phase="decode", scheme="tp64",
):
    """The drafter, traced with `phase` baked: "prefill" is the `at_zero` step (the
    main-stream KV writes only), "decode" the draft proper. `mk_kernels` makes one
    `@nki.jit` entry per phase and scheme. The drafter's groups are the ranks the
    head and sampler run on: the world under tp64, the last stage under tp8pp8."""
    at_zero, K, win, stage, kver, noise_id, vocab_size, topk_e, bs = flags
    assert bool(at_zero) == (phase == "prefill"), f"flags at_zero={at_zero} vs phase {phase}"
    nstages = wq_a.shape[0]
    L = main_hidden.shape[0]
    Lp = pad_to(L, PT)
    Kp = PT
    main_in = main_hidden.shape[1]
    dim = main_proj.shape[1]
    hc = hca_fn_t0.shape[0] // dim
    width = hc * dim
    mix = hca_fn_t0.shape[1]
    # fp8 / fp4 weights: `[S, K/4, N]` words (the MX layout)
    q_lora = wq_a.shape[2]
    hd = wkv.shape[2]
    heads = wq_b.shape[2] // hd
    o_lora_total = wo_a.shape[2]
    g_in = wo_a.shape[1] * 4
    hpg = g_in // hd
    ngroups = heads // hpg
    o_lora = o_lora_total // ngroups
    n_experts = gate_w_t0.shape[1]
    n_local = w1.shape[1]
    inter = w1.shape[3]
    kpad = pad_to(win + K, KTILE)
    ntab = win + K
    ntab_p = pad_to(ntab, PT)
    main_proj4 = main_proj.view(X.FP8_X4)
    wq_a4, wq_b4, wkv4 = wq_a.view(X.FP8_X4), wq_b.view(X.FP8_X4), wkv.view(X.FP8_X4)
    wo_a4, wo_b4 = wo_a.view(X.FP8_X4), wo_b.view(X.FP8_X4)
    sh_w1_4, sh_w3_4, sh_w2_4 = sh_w1.view(X.FP8_X4), sh_w3.view(X.FP8_X4), sh_w2.view(X.FP8_X4)
    w1_4, w3_4, w2_4 = w1.view(X.FP4_X4), w3.view(X.FP4_X4), w2.view(X.FP4_X4)
    part = head_w.shape[0]
    mrank = mk_embed_w.shape[1]
    world = len(groups_world[0])
    W8 = max(world, 8)
    scale = float(hd) ** -0.5
    bs_log2 = _log2(bs)
    caches = (cache0, cache1, cache2)
    hca_fn_ts = (hca_fn_t0, hca_fn_t1, hca_fn_t2)
    hcf_fn_ts = (hcf_fn_t0, hcf_fn_t1, hcf_fn_t2)
    gate_w_ts = (gate_w_t0, gate_w_t1, gate_w_t2)
    assert K <= PT and nstages <= 3 and part % PT == 0 and mrank % PT == 0 and part % 512 == 0
    assert hpg * ngroups == heads

    pid = nl.program_id(0)
    rg_attn = ncc.ReplicaGroup([list(g) for g in groups_attn])
    rg_world = ncc.ReplicaGroup([list(g) for g in groups_world])
    kt = tiles_of(K, pid)             # the draft rows: one partial tile, core 0's
    kt_all = tiles_of(K, 0)
    ptiles = tiles_of(Kp, 0) + tiles_of(Kp, 1)
    lt = tiles_of(L, pid)
    ltp = tiles_of(Lp, 0) + tiles_of(Lp, 1)

    # ------------------------------------------------------------------ outputs
    output_ids = nl.ndarray((1, K + 1), dtype=nl.int32, buffer=nl.shared_hbm)
    confidence = nl.ndarray((K, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    dbg = nl.ndarray((Kp, max(width, part + dim)) if stage != 9 else (1, 1),
                     dtype=nl.float32, buffer=nl.shared_hbm)

    # ------------------------------------------------------------------ scratch
    sc = nl.ndarray((1, 4), dtype=nl.float32, buffer=nl.shared_hbm, name="sc")       # seed, draft_pos, wv
    ids_hbm = nl.ndarray((Kp, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="ids")
    drow_pos = nl.ndarray((Kp, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="drow_pos")
    wvk = nl.ndarray((Kp, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="wvk")
    zcol = nl.ndarray((Kp, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="zcol")
    tkin = nl.ndarray((Kp, K), dtype=nl.int32, buffer=nl.shared_hbm, name="tkin")
    win_slots = nl.ndarray((win, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="win_slots")
    mid = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="mid")
    mh16 = nl.ndarray((Lp, main_in), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="mh16")
    mxr = nl.ndarray((Lp, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="mxr")
    mx = nl.ndarray((Lp, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="mx")
    # One scratch pair per stage: at `at_zero` the three KV writes run back to back
    # with no other barrier between them, and a shared pair would let a core start
    # the next stage's projection while the other core still reads this one's.
    mkvr_s = [nl.ndarray((Lp, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name=f"mkvr{s}") for s in range(nstages)]
    mkv_s = [nl.ndarray((Lp, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name=f"mkv{s}") for s in range(nstages)]
    e_loc = nl.ndarray((Kp, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="e_loc")
    e_sum = nl.ndarray((Kp, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="e_sum")
    st = nl.ndarray((K, width), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="st")
    st2 = nl.ndarray((K, width), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="st2")
    st3 = nl.ndarray((K, width), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="st3")
    pre_mix = nl.ndarray((K, hc), dtype=nl.float32, buffer=nl.shared_hbm, name="pre_mix")
    pre_a = nl.ndarray((K, hc), dtype=nl.float32, buffer=nl.shared_hbm, name="pre_a")
    post_a = nl.ndarray((K, hc), dtype=nl.float32, buffer=nl.shared_hbm, name="post_a")
    comb_a = nl.ndarray((K, hc * hc), dtype=nl.float32, buffer=nl.shared_hbm, name="comb_a")
    pre_f = nl.ndarray((K, hc), dtype=nl.float32, buffer=nl.shared_hbm, name="pre_f")
    post_f = nl.ndarray((K, hc), dtype=nl.float32, buffer=nl.shared_hbm, name="post_f")
    comb_f = nl.ndarray((K, hc * hc), dtype=nl.float32, buffer=nl.shared_hbm, name="comb_f")
    xn = nl.ndarray((Kp, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="xn")
    qa = nl.ndarray((Kp, q_lora), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="qa")
    qr = nl.ndarray((Kp, q_lora), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="qr")
    q = nl.ndarray((Kp, heads * hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="q")
    kvr = nl.ndarray((Kp, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kvr")
    kvd = nl.ndarray((Kp, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kvd")
    kvtab = nl.ndarray((ntab_p, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kvtab")
    idx = nl.ndarray((Kp, kpad), dtype=nl.int32, buffer=nl.shared_hbm, name="idx")
    o = nl.ndarray((Kp, heads * hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="o")
    oq = nl.ndarray((Kp, heads * hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="oq")
    z = nl.ndarray((Kp, o_lora_total), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="z")
    cc_src = nl.ndarray((Kp, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((Kp, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_dst")
    attn_out = nl.ndarray((Kp, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="attn_out")
    # xn2 / wexp / eacc carry one extra zero tile for the routed experts' padding slots.
    xn2 = nl.ndarray((Kp + PT, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="xn2")
    logits = nl.ndarray((Kp, n_experts), dtype=nl.float32, buffer=nl.shared_hbm, name="logits")
    wexp = nl.ndarray((Kp + PT, n_local), dtype=nl.float32, buffer=nl.shared_hbm, name="wexp")
    eg = nl.ndarray((Kp, inter), dtype=nl.bfloat16, buffer=nl.private_hbm)
    eu = nl.ndarray((Kp, inter), dtype=nl.bfloat16, buffer=nl.private_hbm)
    ey = nl.ndarray((Kp, inter), dtype=nl.bfloat16, buffer=nl.private_hbm)
    eo32 = nl.ndarray((Kp, dim), dtype=nl.float32, buffer=nl.private_hbm)
    eacc = nl.ndarray((Kp + PT, dim), dtype=nl.float32, buffer=nl.private_hbm)
    part_ = nl.ndarray((CORES, Kp, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="part")
    cc2_src = nl.ndarray((Kp, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc2_src")
    cc2_dst = nl.ndarray((Kp, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc2_dst")
    ffn_out = nl.ndarray((Kp, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="ffn_out")
    xh = nl.ndarray((Kp, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="xh")       # collapsed, pre-norm
    xhn = nl.ndarray((Kp, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="xhn")     # normed
    lg = nl.ndarray((Kp, part), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="lg")
    lg32 = nl.ndarray((Kp, part), dtype=nl.float32, buffer=nl.shared_hbm, name="lg32")
    mcc_src = nl.ndarray((1, mrank), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="mcc_src")
    mcc_dst = nl.ndarray((1, mrank), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="mcc_dst")
    gv_src = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="gv_src")
    gv_dst = nl.ndarray((world, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="gv_dst")
    gi_src = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="gi_src")
    gi_dst = nl.ndarray((world, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="gi_dst")
    memb = nl.ndarray((Kp, mrank), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="memb")

    def ret():
        return output_ids, confidence, dbg, cache0, cache1, cache2

    # ------------------------------------------------------------------ 0. the draft's indices
    # seed, draft_pos = pos + tap, wv = draft_pos - (win - 1): three scalars every core reads.
    if pid == 0:
        mt = nl.ndarray((1, 3), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=mt, src=meta[0:1, 0:3])
        mtf = nl.ndarray((1, 3), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mtf, src=mt)
        s4 = nl.ndarray((1, 4), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(s4, 0.0)
        nisa.tensor_copy(dst=s4[0:1, 0:1], src=mtf[0:1, 0:1])
        nisa.tensor_tensor(dst=s4[0:1, 1:2], data1=mtf[0:1, 1:2], data2=mtf[0:1, 2:3], op=nl.add)
        nisa.tensor_scalar(dst=s4[0:1, 2:3], data=s4[0:1, 1:2], op0=nl.add, operand0=float(-(win - 1)))
        nisa.dma_copy(dst=sc[0:1, 0:4], src=s4)
    nisa.core_barrier(data=sc, cores=(0, 1))
    if pid == 0:
        ramp = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=ramp, pattern=[[1, 1]], offset=0, channel_multiplier=1)
        rampf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=rampf, src=ramp)
        dp = bcast_row(sc[0:1, 1:2], 1, PT)
        wvb = bcast_row(sc[0:1, 2:3], 1, PT)
        # draft rows sit at draft_pos + 1 + r
        dr = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=dr, data=rampf, op0=nl.add, operand0=dp[0:PT, 0:1], op1=nl.add, operand1=1.0)
        dri = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=dri, src=dr)
        nisa.dma_copy(dst=drow_pos[0:Kp, 0:1], src=dri)
        wvi = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=wvi, src=wvb)
        nisa.dma_copy(dst=wvk[0:Kp, 0:1], src=wvi)
        zi = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(zi, 0)
        nisa.dma_copy(dst=zcol[0:Kp, 0:1], src=zi)
        tk = nl.ndarray((PT, K), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=tk, pattern=[[1, K]], offset=win, channel_multiplier=0)
        nisa.dma_copy(dst=tkin[0:Kp, 0:K], src=tk)
        # draft token ids: the seed, then noise
        idf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(idf, float(noise_id))
        sdb = bcast_row(sc[0:1, 0:1], 1, 1)
        nisa.tensor_copy(dst=idf[0:1, :], src=sdb)
        idi = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idi, src=idf)
        nisa.dma_copy(dst=ids_hbm[0:Kp, 0:1], src=idi)
        # the window's cache slots: p = max(wv + j, 0); block_table[p >> log2(bs)] * bs + (p & (bs - 1))
        for r0, nr in tiles_of(win, 0) + tiles_of(win, 1):
            wp = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=wp[0:nr, :], data=rampf[0:nr, :], op0=nl.add, operand0=wvb[0:nr, 0:1],
                               op1=nl.add, operand1=float(r0))
            nisa.tensor_scalar(dst=wp[0:nr, :], data=wp[0:nr, :], op0=nl.maximum, operand0=0.0)
            wpi = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=wpi[0:nr, :], src=wp[0:nr, :])
            blki = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
            blki[0:nr, :] = nl.right_shift(wpi[0:nr, :], bs_log2)
            offi = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
            offi[0:nr, :] = nl.bitwise_and(wpi[0:nr, :], bs - 1)
            bt = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=bt[0:nr, :], src=block_table.ap(pattern=[[1, nr], [1, 1]],
                                                               vector_offset=blki[0:nr, 0:1], indirect_dim=0))
            btf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=btf[0:nr, :], src=bt[0:nr, :])
            offf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=offf[0:nr, :], src=offi[0:nr, :])
            slot = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=slot[0:nr, :], data=btf[0:nr, :], op0=nl.multiply, operand0=float(bs))
            nisa.tensor_tensor(dst=slot[0:nr, :], data1=slot[0:nr, :], data2=offf[0:nr, :], op=nl.add)
            sloti = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=sloti[0:nr, :], src=slot[0:nr, :])
            nisa.dma_copy(dst=win_slots[r0 : r0 + nr, 0:1], src=sloti[0:nr, :])
    nisa.core_barrier(data=drow_pos, cores=(0, 1))
    nisa.core_barrier(data=wvk, cores=(0, 1))
    nisa.core_barrier(data=zcol, cores=(0, 1))
    nisa.core_barrier(data=tkin, cores=(0, 1))
    nisa.core_barrier(data=ids_hbm, cores=(0, 1))
    nisa.core_barrier(data=win_slots, cores=(0, 1))

    # ------------------------------------------------------------------ 1. main_x and the draft embedding
    cast_rows(main_hidden, mh16, lt, main_in, nl.bfloat16)
    if pid == 0 and L < Lp:
        padm = nl.ndarray((PT, main_in), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(padm, 0.0)
        nisa.dma_copy(dst=mh16[L:Lp, :], src=padm[0 : Lp - L, :])
    nisa.core_barrier(data=mh16, cores=(0, 1))
    lo, hi = _half(dim, pid)
    X.gemm_rows(mh16, Lp, main_in, main_proj4, main_proj_s, mxr, lo, hi)
    nisa.core_barrier(data=mxr, cores=(0, 1))
    norm_rows(mxr, main_norm_w, mx, lt, dim, qdq=False, chunk=1024)
    if pid == 0 and L < Lp:
        padx = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(padx, 0.0)
        nisa.dma_copy(dst=mx[L:Lp, :], src=padx[0 : Lp - L, :])
    nisa.core_barrier(data=mx, cores=(0, 1))

    # This rank's embedding rows of [seed, noise, ...] (zero elsewhere), then the world sum.
    if pid == 0:
        padz = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(padz, 0.0)
        nisa.dma_copy(dst=e_loc[0:Kp, :], src=padz)
        engram_gather(ids_hbm, embed_w, vocab_start, e_loc, kt_all, 1, dim, part)
    nisa.core_barrier(data=e_loc, cores=(0, 1))
    ncc.all_reduce(srcs=[e_loc], dsts=[e_sum], replica_group=rg_world, op=nl.add)
    if pid == 0:
        e = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=e[0:K, :], src=e_sum[0:K, :])
        for c in range(hc):
            nisa.dma_copy(dst=st[0:K, c * dim : (c + 1) * dim], src=e[0:K, :])
        pm = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(pm, 0.0)
        nisa.memset(pm[:, 0:1], 1.0)
        nisa.dma_copy(dst=pre_mix[0:K, :], src=pm[0:K, :])
    nisa.core_barrier(data=st, cores=(0, 1))
    nisa.core_barrier(data=pre_mix, cores=(0, 1))

    cur_st, cur_pre = st, pre_mix
    for s in range(nstages):
        cache = caches[s]
        mkvr, mkv = mkvr_s[s], mkv_s[s]
        # ---- the main stream's KV for this step, into the stage's window cache
        lo, hi = _half(hd, pid)
        X.gemm_rows(mx, Lp, dim, wkv4[s], wkv_s[s], mkvr, lo, hi)
        nisa.core_barrier(data=mkvr, cores=(0, 1))
        norm_rows(mkvr, kv_norm_w[s : s + 1], mkv, lt, hd, qdq=False, chunk=hd)
        rope_rows(mkv, mkv, lt, hd, hd, 1, freqs, row_pos, inverse=False, qdq=True)
        scatter_rows(mkv, write_slots, cache, lt, hd)
        nisa.core_barrier(data=cache, cores=(0, 1))
        if at_zero:
            continue
        nxt_st = st2 if s % 2 == 0 else st3
        # ---- hc_in (attention)
        fnT = load_fn_t(hca_fn_ts[s], width, mix)
        for r0, nr in kt:
            hc_in_tile(r0, nr, cur_st, cur_pre, fnT, hca_base[s : s + 1], hca_scale[s : s + 1],
                       attn_norm_w[s : s + 1], hc, dim, mix, xn, pre_a, post_a, comb_a)
        if pid == 0:
            padn = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(padn, 0.0)
            nisa.dma_copy(dst=xn[K:Kp, :], src=padn[0 : Kp - K, :])
        nisa.core_barrier(data=xn, cores=(0, 1))
        # ---- projections of the draft rows
        if s == 0 and stage == 11:
            cast_rows(xn, dbg[0:Kp, 0:dim], kt, dim, nl.float32)
            nisa.core_barrier(data=dbg, cores=(0, 1))
            return ret()
        lo, hi = _half(q_lora, pid)
        X.gemm_rows(xn, Kp, dim, wq_a4[s], wq_a_s[s], qa, lo, hi)
        lo, hi = _half(hd, pid)
        X.gemm_rows(xn, Kp, dim, wkv4[s], wkv_s[s], kvr, lo, hi)
        nisa.core_barrier(data=qa, cores=(0, 1))
        nisa.core_barrier(data=kvr, cores=(0, 1))
        norm_rows(qa, q_norm_w[s : s + 1], qr, kt, q_lora, qdq=False, chunk=q_lora)
        norm_rows(kvr, kv_norm_w[s : s + 1], kvd, kt, hd, qdq=False, chunk=hd)
        rope_rows(kvd, kvd, kt, hd, hd, 1, freqs, drow_pos, inverse=False, qdq=True)
        if pid == 0:
            padq = nl.ndarray((PT, q_lora), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(padq, 0.0)
            nisa.dma_copy(dst=qr[K:Kp, :], src=padq[0 : Kp - K, :])
        nisa.core_barrier(data=qr, cores=(0, 1))
        nisa.core_barrier(data=kvd, cores=(0, 1))
        lo, hi = _half(heads * hd, pid)
        X.gemm_rows(qr, Kp, q_lora, wq_b4[s], wq_b_s[s], q, lo, hi)
        nisa.core_barrier(data=q, cores=(0, 1))
        rope_rows(q, q, kt, heads * hd, hd, heads, freqs, drow_pos, inverse=False)
        if s == 0 and stage == 12:
            cast_rows(q, dbg[0:Kp, 0 : heads * hd], kt, heads * hd, nl.float32)
            nisa.core_barrier(data=dbg, cores=(0, 1))
            return ret()
        if s == 0 and stage == 13:
            cast_rows(kvd, dbg[0:Kp, 0:hd], kt, hd, nl.float32)
            nisa.core_barrier(data=dbg, cores=(0, 1))
            return ret()
        # ---- the slot table: the window, then the draft block's own rows
        gather_rows(cache, win_slots, kvtab[0:win], tiles_of(win, pid), hd)
        if pid == 0:
            dk = nl.ndarray((PT, hd), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(dk, 0.0)
            nisa.dma_copy(dst=dk[0:K, :], src=kvd[0:K, :])
            nisa.dma_copy(dst=kvtab[win:ntab_p, :], src=dk[0 : ntab_p - win, :])
            attn_index_table(idx, kt_all, zcol, wvk, win, zcol, tkin, K, kpad)
            padi = nl.ndarray((PT, kpad), dtype=nl.int32, buffer=nl.sbuf)
            nisa.memset(padi, 0)
            nisa.dma_copy(dst=idx[K:Kp, :], src=padi[0 : Kp - K, :])
        nisa.core_barrier(data=q, cores=(0, 1))
        nisa.core_barrier(data=kvtab, cores=(0, 1))
        nisa.core_barrier(data=idx, cores=(0, 1))
        # ---- attention over the K draft rows
        sink_t = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sink_t, src=attn_sink[s, 0:heads, 0:1])
        ntok = cdiv(K, CORES)
        tok0 = 0 if pid == 0 else K - ntok
        kTs = build_kT(kvtab, ntab, hd)

        def tok_body(i, tok0=tok0, kTs=kTs):
            attn_token_body_ti(i, tok0, q, kvtab, kTs, idx, o, sink_t, heads, hd, kpad, scale)

        nl.fori_loop(0, ntok, tok_body)
        nisa.core_barrier(data=o, cores=(0, 1))
        if s == 0 and stage == 14:
            cast_rows(o, dbg[0:Kp, 0 : heads * hd], kt, heads * hd, nl.float32)
            nisa.core_barrier(data=dbg, cores=(0, 1))
            return ret()
        rope_rows(o, oq, kt, heads * hd, hd, heads, freqs, drow_pos, inverse=True, qdq=False)
        if pid == 0:
            pado = nl.ndarray((PT, heads * hd), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(pado, 0.0)
            nisa.dma_copy(dst=oq[K:Kp, :], src=pado[0 : Kp - K, :])
        nisa.core_barrier(data=oq, cores=(0, 1))
        for g in range(ngroups):
            lo, hi = _half(o_lora, pid)
            X.gemm_rows(oq, Kp, g_in, wo_a4[s], wo_a_s[s], z, g * o_lora + lo, g * o_lora + hi, x_col0=g * g_in)
        nisa.core_barrier(data=z, cores=(0, 1))
        lo, hi = _half(dim, pid)
        X.gemm_rows(z, Kp, o_lora_total, wo_b4[s], wo_b_s[s], cc_src, lo, hi, out_dtype=nl.float32)
        nisa.core_barrier(data=cc_src, cores=(0, 1))
        if s == 0 and stage == 15:
            cast_rows(cc_src, dbg[0:Kp, 0:dim], kt, dim, nl.float32)
            cast_rows(xn, dbg[0:Kp, dim : 2 * dim], kt, dim, nl.float32)
            nisa.core_barrier(data=dbg, cores=(0, 1))
            return ret()
        ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], replica_group=rg_attn, op=nl.add)
        cast_rows(cc_dst, attn_out, kt, dim, nl.bfloat16)
        # ---- hc_out, hc_in (ffn)
        for r0, nr in kt:
            hc_out_tile(r0, nr, attn_out, cur_st, post_a, comb_a, nxt_st, hc, dim)
        if s == 0 and stage == 16:
            cast_rows(nxt_st, dbg[0:K, 0:width], kt, width, nl.float32)
            nisa.core_barrier(data=dbg, cores=(0, 1))
            return ret()
        fnT2 = load_fn_t(hcf_fn_ts[s], width, mix)
        for r0, nr in kt:
            hc_in_tile(r0, nr, nxt_st, pre_a, fnT2, hcf_base[s : s + 1], hcf_scale[s : s + 1],
                       ffn_norm_w[s : s + 1], hc, dim, mix, xn2, pre_f, post_f, comb_f)
        if pid == 0:
            padn2 = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(padn2, 0.0)
            nisa.dma_copy(dst=xn2[K:Kp, :], src=padn2[0 : Kp - K, :])
            nisa.dma_copy(dst=xn2[Kp : Kp + PT, :], src=padn2)
        nisa.core_barrier(data=xn2, cores=(0, 1))
        # ---- MoE: n_experts routed, top-k, plus the shared expert
        if s == 0 and stage == 17:
            cast_rows(xn2, dbg[0:Kp, 0:dim], kt, dim, nl.float32)
            nisa.core_barrier(data=dbg, cores=(0, 1))
            return ret()
        gemm_fp32(xn2, gate_w_ts[s], logits, kt_all, dim, n_experts)
        route_topk(logits, gate_b[s : s + 1], expert_lo, kt_all, n_experts, topk_e, n_local, ROUTE_SCALE, wexp)
        zw = nl.ndarray((PT, n_local), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(zw, 0.0)
        nisa.dma_copy(dst=wexp[K:Kp, :], src=zw[0 : Kp - K, :])
        nisa.dma_copy(dst=wexp[Kp : Kp + PT, :], src=zw)
        # routed experts over the rows that picked them, into the zeroed accumulator
        lo, hi = _half(inter, pid)
        mk_moe.routed_experts(pid, n_local, Kp, xn2, wexp, w1_4[s], s1[s], w3_4[s], s3[s],
                              w2_4[s], s2[s], dim, inter, lo, hi, SWIGLU_LIMIT, eacc)
        X.gemm_rows(xn2, Kp, dim, sh_w1_4[s], sh_s1[s], eg, lo, hi)
        X.gemm_rows(xn2, Kp, dim, sh_w3_4[s], sh_s3[s], eu, lo, hi)
        swiglu_rows(eg[0:Kp, lo:hi], eu[0:Kp, lo:hi], None, 0, ey[0:Kp, lo:hi], ptiles, hi - lo, SWIGLU_LIMIT)
        # The shared expert's weights are replicated on every rank and the model adds its
        # output ONCE, after the expert all-reduce. Every rank adds it to its partial here,
        # scaled by 1/world, so the world sum carries it exactly once (a power-of-two
        # scale of the fp32 partial is exact).
        X.gemm_rows(ey, Kp, hi - lo, sh_w2_4[s], sh_s2[s], eo32, 0, dim, x_col0=lo, krow0=lo // 4,
                    out_dtype=nl.float32, out_scale=1.0 / world)
        add_rows(eacc, eo32, ptiles, dim, False)
        for r0, nr in ptiles:
            t = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=t[0:nr, :], src=eacc[r0 : r0 + nr, :])
            nisa.dma_copy(dst=part_[pid, r0 : r0 + nr, :], src=t[0:nr, :])
        nisa.core_barrier(data=part_, cores=(0, 1))
        for r0, nr in tiles_of(Kp, pid):
            a = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=a[0:nr, :], src=part_[0, r0 : r0 + nr, :])
            b = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=b[0:nr, :], src=part_[1, r0 : r0 + nr, :])
            nisa.tensor_tensor(dst=a[0:nr, :], data1=a[0:nr, :], data2=b[0:nr, :], op=nl.add)
            nisa.dma_copy(dst=cc2_src[r0 : r0 + nr, :], src=a[0:nr, :])
        nisa.core_barrier(data=cc2_src, cores=(0, 1))
        if s == 0 and stage == 18:
            cast_rows(cc2_src, dbg[0:Kp, 0:dim], kt, dim, nl.float32)
            cast_rows(xn2, dbg[0:Kp, dim : 2 * dim], kt, dim, nl.float32)
            nisa.core_barrier(data=dbg, cores=(0, 1))
            return ret()
        ncc.all_reduce(srcs=[cc2_src], dsts=[cc2_dst], replica_group=rg_world, op=nl.add)
        cast_rows(cc2_dst, ffn_out, kt, dim, nl.bfloat16)
        out_st = st if s % 2 == 0 else st2
        for r0, nr in kt:
            hc_out_tile(r0, nr, ffn_out, nxt_st, post_f, comb_f, out_st, hc, dim)
        nisa.core_barrier(data=out_st, cores=(0, 1))
        nisa.core_barrier(data=pre_f, cores=(0, 1))
        cur_st, cur_pre = out_st, pre_f
        if stage == s + 1:
            cast_rows(cur_st, dbg[0:K, 0:width], kt, width, nl.float32)
            nisa.core_barrier(data=dbg, cores=(0, 1))
            return ret()

    # ------------------------------------------------------------------ head
    # Collapse with the last pre_mix (fp32 sum, bf16 out), then the final norm.
    if pid == 0:
        for r0, nr in kt_all:
            pm = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=pm[0:nr, :], src=cur_pre[r0 : r0 + nr, 0:hc])
            acc = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
            for c in range(hc):
                xc = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=xc[0:nr, :], src=cur_st[r0 : r0 + nr, c * dim : (c + 1) * dim])
                if c == 0:
                    nisa.tensor_scalar(dst=acc[0:nr, :], data=xc[0:nr, :], op0=nl.multiply, operand0=pm[0:nr, 0:1],
                                       engine=nisa.engine.vector)
                else:
                    nisa.scalar_tensor_tensor(dst=acc[0:nr, :], data=xc[0:nr, :], op0=nl.multiply,
                                              operand0=pm[0:nr, c : c + 1], op1=nl.add, operand1=acc[0:nr, :])
            y16 = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(y16, 0.0)
            nisa.tensor_copy(dst=y16[0:nr, :], src=acc[0:nr, :])
            nisa.dma_copy(dst=xh[0:Kp, :], src=y16)
    nisa.core_barrier(data=xh, cores=(0, 1))
    norm_rows(xh, fnorm_w, xhn, kt, dim, qdq=False, chunk=1024)
    if pid == 0:
        padh = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(padh, 0.0)
        nisa.dma_copy(dst=xhn[K:Kp, :], src=padh[0 : Kp - K, :])
    nisa.core_barrier(data=xhn, cores=(0, 1))
    lo, hi = _half(part, pid)
    bf16_gemm_rows(xhn, head_w, lg, Kp, dim, part, lo, hi)
    nisa.core_barrier(data=lg, cores=(0, 1))
    # fp32 logits, the vocabulary pad (columns past the real vocabulary) pushed to -1e30.
    if pid == 0:
        vs1 = bcast_row(vocab_start[0:1, 0:1], 1, PT)
        for r0, nr in kt_all:
            l16 = nl.ndarray((PT, part), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=l16[0:nr, :], src=lg[r0 : r0 + nr, 0:part])
            l32 = nl.ndarray((PT, part), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=l32[0:nr, :], src=l16[0:nr, :])
            ci = nl.ndarray((PT, part), dtype=nl.int32, buffer=nl.sbuf)
            nisa.iota(dst=ci[0:nr, :], pattern=[[1, part]], offset=0, channel_multiplier=0)
            cf = nl.ndarray((PT, part), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=cf[0:nr, :], src=ci[0:nr, :])
            live = nl.ndarray((PT, part), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=live[0:nr, :], data=cf[0:nr, :], op0=nl.add, operand0=vs1[0:nr, 0:1],
                               op1=nl.less, operand1=float(vocab_size))
            dead = nl.ndarray((PT, part), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=dead[0:nr, :], data=live[0:nr, :], op0=nl.add, operand0=-1.0,
                               op1=nl.multiply, operand1=-NEG)
            nisa.tensor_tensor(dst=l32[0:nr, :], data1=l32[0:nr, :], data2=dead[0:nr, :], op=nl.add)
            nisa.dma_copy(dst=lg32[r0 : r0 + nr, 0:part], src=l32[0:nr, :])
    nisa.core_barrier(data=lg32, cores=(0, 1))
    if stage == 4:
        cast_rows(lg32, dbg[0:Kp, 0:part], kt, part, nl.float32)
        cast_rows(xh, dbg[0:Kp, part : part + dim], kt, dim, nl.float32)
        nisa.core_barrier(data=dbg, cores=(0, 1))
        return ret()

    # The markov head transposed once: (PT, mrank // PT, part) bf16.
    mkT = nl.ndarray((PT, mrank // PT, part), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kb in range(mrank // PT):
        for nb in range(part // PT):
            nisa.dma_transpose(dst=mkT[0:PT, kb, nb * PT : (nb + 1) * PT],
                               src=mk_head_w[nb * PT : (nb + 1) * PT, kb * PT : (kb + 1) * PT])
    vs1 = bcast_row(vocab_start[0:1, 0:1], 1, 1)
    ids_sb = nl.ndarray((1, K + 1), dtype=nl.int32, buffer=nl.sbuf)
    sdb = bcast_row(sc[0:1, 0:1], 1, 1)
    nisa.tensor_copy(dst=ids_sb[0:1, 0:1], src=sdb)
    for i in range(K):
        # markov embedding of output_ids[i]: this rank's row (or zero), world-summed.
        if pid == 0:
            nisa.dma_copy(dst=mid[0:1, 0:1], src=ids_sb[0:1, i : i + 1])
            engram_gather(mid, mk_embed_w, vocab_start, mcc_src, [(0, 1)], 1, mrank, part)
        nisa.core_barrier(data=mcc_src, cores=(0, 1))
        ncc.all_reduce(srcs=[mcc_src], dsts=[mcc_dst], replica_group=rg_world, op=nl.add)
        me = nl.ndarray((1, mrank), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=me, src=mcc_dst[0:1, 0:mrank])
        if pid == 0:
            nisa.dma_copy(dst=memb[i : i + 1, 0:mrank], src=me)
        # bias = bf16(me @ mk_head_w.T) on this rank's columns; logits[i] += bias; (max, argmax).
        meT = nl.ndarray((PT, mrank // PT, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        for kb in range(mrank // PT):
            tps = nl.ndarray((PT, 1), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=me[0:1, kb * PT : (kb + 1) * PT])
            nisa.tensor_copy(dst=meT[0:PT, kb, 0:1], src=tps)
        row = nl.ndarray((1, part), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=row, src=lg32[i : i + 1, 0:part])
        for nb in range(part // 512):
            acc = nl.ndarray((1, 512), dtype=nl.float32, buffer=nl.psum)
            for kb in range(mrank // PT):
                nisa.nc_matmul(dst=acc, stationary=meT[0:PT, kb, 0:1], moving=mkT[0:PT, kb, nb * 512 : (nb + 1) * 512],
                               accumulate=(kb > 0))
            b16 = nl.ndarray((1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=b16, src=acc)
            b32 = nl.ndarray((1, 512), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=b32, src=b16)
            nisa.tensor_tensor(dst=row[0:1, nb * 512 : (nb + 1) * 512], data1=row[0:1, nb * 512 : (nb + 1) * 512],
                               data2=b32, op=nl.add)
        best = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.max8(dst=best, src=row)
        bidx = nl.ndarray((1, 8), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.nc_find_index8(dst=bidx, data=row, vals=best)
        bf = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=bf, src=bidx[0:1, 0:1])
        gidx_f = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gidx_f, data1=bf, data2=vs1, op=nl.add)
        if pid == 0:
            nisa.dma_copy(dst=gv_src[0:1, 0:1], src=best[0:1, 0:1])
            nisa.dma_copy(dst=gi_src[0:1, 0:1], src=gidx_f)
        nisa.core_barrier(data=gv_src, cores=(0, 1))
        nisa.core_barrier(data=gi_src, cores=(0, 1))
        ncc.all_gather(srcs=[gv_src], dsts=[gv_dst], replica_group=rg_world, collective_dim=0)
        ncc.all_gather(srcs=[gi_src], dsts=[gi_dst], replica_group=rg_world, collective_dim=0)
        # the global argmax: the highest shard max, ties to the lowest rank (lowest index)
        vrow = nl.ndarray((1, W8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(vrow, NEG)
        nisa.dma_copy(dst=vrow[0:1, 0:world], src=gv_dst.reshape((1, world))[0:1, 0:world])
        trow = nl.ndarray((1, W8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(trow, 0.0)
        nisa.dma_copy(dst=trow[0:1, 0:world], src=gi_dst.reshape((1, world))[0:1, 0:world])
        gbest = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.max8(dst=gbest, src=vrow)
        gidx = nl.ndarray((1, 8), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.nc_find_index8(dst=gidx, data=vrow, vals=gbest)
        tok = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.nc_n_gather(dst=tok, data=trow, indices=gidx[0:1, 0:1])
        nisa.tensor_copy(dst=ids_sb[0:1, i + 1 : i + 2], src=tok)
    nisa.core_barrier(data=memb, cores=(0, 1))
    if pid == 0:
        nisa.dma_copy(dst=output_ids[0:1, 0 : K + 1], src=ids_sb)
        # confidence = bf16( [hidden | markov_embed] . proj_w )
        hcat = nl.ndarray((PT, dim + mrank), dtype=nl.float32, buffer=nl.sbuf)
        h16 = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=h16[0:K, :], src=xh[0:K, 0:dim])
        nisa.tensor_copy(dst=hcat[0:K, 0:dim], src=h16[0:K, :])
        m16 = nl.ndarray((PT, mrank), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=m16[0:K, :], src=memb[0:K, 0:mrank])
        nisa.tensor_copy(dst=hcat[0:K, dim : dim + mrank], src=m16[0:K, :])
        wrow = bcast_row(conf_w[0:1, 0 : dim + mrank], dim + mrank, K)
        prod = nl.ndarray((PT, dim + mrank), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prod[0:K, :], data1=hcat[0:K, :], data2=wrow, op=nl.multiply)
        cf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=cf[0:K, :], op=nl.add, data=prod[0:K, :], axis=(1,), keepdims=True)
        c16 = nl.ndarray((PT, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=c16[0:K, :], src=cf[0:K, :])
        c32 = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=c32[0:K, :], src=c16[0:K, :])
        nisa.dma_copy(dst=confidence[0:K, 0:1], src=c32[0:K, :])
    nisa.core_barrier(data=output_ids, cores=(0, 1))
    nisa.core_barrier(data=confidence, cores=(0, 1))
    return ret()


#: The generic entry (both phases by the `at_zero` flag).
draft_forward = nki.jit(draft_body)
