# SPDX-License-Identifier: Apache-2.0
"""One DeepSeek-V4.1-Flash layer as a single NKI kernel (`_run_block`).

    hc_in -> attention (window [+ compressed]) -> all_reduce(attn group) -> hc_out
          -> hc_in -> MoE (top-6 of 384, this rank's experts + shared) -> all_reduce(world)
          -> hc_out

Both physical cores of the logical NeuronCore run this body (LNC=2); every phase
says how it divides the work (`pid = nl.program_id(0)`). Data that crosses cores
lives in named `shared_hbm` scratch behind `core_barrier`s; the two collectives
are `nki.collectives.all_reduce` over the replica groups passed as tuples.

The quantized projections run on the tensor engine's MX matmul (`mk_mx`: fp8 / fp4
weights in the host-packed block-scaled layout, activations quantized on chip), the
routed experts only over the rows that picked them (`mk_moe`), and the sparse
attention selects its keys on chip through tensor indirection into a resident
transposed slot table (`mk_common.attn_token_body_ti`) when the table fits SBUF.

The KV cache is an input that the kernel writes into and returns, so the caller
sees it mutated in place (operand/output aliasing). The host prepares only
integer bookkeeping: absolute row positions, cache slots of this step's rows,
the slot list of each span's window/compressed rows, and three index columns.

`flags` = (ratio, nspans, window, stage). `stage` < 9 returns early with an intermediate
tensor in the first output slot, for staged validation:
  1 normed-x | 2 q | 3 kv | 4 o (attention) | 5 attn_out (reduced, bf16)
  6 stream2 | 7 normed2 | 8 ffn_out (reduced, bf16) | 9 full layer
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

from .mk_compress import (
    bf16_gemm_rows,
    fp32_proj_rows,
    index_scores_span,
    index_topk,
    pool_groups,
    rope_fp4_put,
    scatter_rows_f32,
)
from . import mk_moe
from . import mk_mx as X
from .mk_common import (
    PT,
    KTILE,
    add_rows,
    attn_index_table,
    attn_token_body,
    attn_token_body_ti,
    build_kT,
    cast_rows,
    cdiv,
    engram_readout,
    tap_tile,
    gather_rows,
    gemm_fp32,
    hc_in_tile,
    hc_out_tile,
    kt_resident,
    load_fn_t,
    norm_rows,
    pad_to,
    rope_rows,
    route_topk,
    scatter_rows,
    swiglu_rows,
    tiles_of,
)

CORES = 2
#: `swiglu_limit`, `routed_scaling_factor`, `num_experts_per_tok`.
SWIGLU_LIMIT = 10.0
ROUTE_SCALE = 1.5
TOPK_EXPERTS = 6


def _half(n, pid):
    """`(lo, hi)` of core `pid`'s share of `n` columns, in whole 128-column tiles."""
    tiles = n // PT
    a = (tiles + 1) // 2
    lo = 0 if pid == 0 else a * PT
    hi = a * PT if pid == 0 else n
    return lo, hi


def layer_body(
    stream, pre_mix,
    hca_fn_t, hca_base, hca_scale, attn_norm_w,
    hcf_fn_t, hcf_base, hcf_scale, ffn_norm_w,
    wq_a, wq_a_s, q_norm_w, wq_b, wq_b_s, wkv, wkv_s, kv_norm_w,
    wo_a, wo_a_s, wo_b, wo_b_s, attn_sink, freqs,
    row_pos, write_slots, win_slots, cmp_slots, twin, wv, cbase, topk_in,
    window_kv_cache, compress_kv_cache,
    pool_kv, pool_score, index_k_cache,
    cm_wkv, cm_wkv_t, cm_wgate_t, cm_norm_w, grp_pos, grp_mslots, grp_wslots,
    ix_wq_b, ix_wq_b_s, ix_wproj_t, ix_wk, ix_knorm_w, clen,
    hash_ids, etable, erow_start, e_wkv, e_wkv_s, e_qk,
    gate_w_t, gate_b, expert_lo, w1, s1, w3, s3, w2, s2,
    sh_w1, sh_s1, sh_w3, sh_s3, sh_w2, sh_s2,
    groups_attn, groups_world, flags,
    phase="prefill", arch=None, scheme="tp64",
):
    """The block, traced with `phase` ("prefill" / "decode"), the block's `arch`
    `(ratio, is_kv_src, is_idx_src, has_engram, tap)` and the `scheme` baked in;
    `mk_kernels` makes one `@nki.jit` entry point per combination. The runtime
    `flags` must agree with `arch`. Under "tp8pp8" `groups_world` is the 8-rank
    stage: the expert reduce, and the engram table's row sum, span the stage."""
    ratio, nspans, win, stage, is_kv_src, is_idx_src, has_engram, tap, kver = flags
    if arch is not None:
        assert (ratio, is_kv_src, is_idx_src, has_engram, tap) == tuple(arch), (
            f"flags {flags} disagree with the baked archetype {arch}")
    # Under both schemes the engram read-out is in-kernel, over `groups_world` (the
    # stage under "tp8pp8", where the n-gram table is sharded over the stage's ranks).
    T, hc, dim = stream.shape
    Tp = pad_to(T, PT)
    width = hc * dim
    mix = hca_fn_t.shape[1]
    # fp8 / fp4 weights arrive as `[K/4, N]` words (`mk_model.mx_fp8` / `mx_fp4`)
    q_lora = wq_a.shape[1]
    hd = wkv.shape[1]
    heads = wq_b.shape[1] // hd
    o_lora_total = wo_a.shape[1]
    g_in = wo_a.shape[0] * 4
    hpg = g_in // hd
    ngroups = heads // hpg
    o_lora = o_lora_total // ngroups
    n_experts = gate_w_t.shape[1]
    n_local = w1.shape[0]
    inter = w1.shape[2]
    nwin = win_slots.shape[0] // nspans
    read_len = cmp_slots.shape[0] // nspans if ratio else 0
    nspan_rows = nwin + read_len
    ntab = nspans * nspan_rows
    ntab_p = pad_to(ntab, PT)
    # Prefill keeps the slot table transposed in SBUF and selects keys on chip;
    # decode's handful of tokens are cheaper with the per-token transposes.
    resident = phase == "prefill" and kt_resident(ntab, hd)
    topk = min(topk_in.shape[1], read_len) if ratio else 0
    kpad = pad_to(win + topk, KTILE)
    L = T // nspans
    half = cdiv(L, CORES)
    ng = grp_wslots.shape[0] // nspans if is_kv_src else 0   # groups a span
    NG = nspans * ng
    NGp = pad_to(max(NG, 1), PT)
    kd = ix_wk.shape[0]
    nh_idx = ix_wproj_t.shape[1]
    IDX_SCALE = float(kd) ** -0.5 * float(nh_idx * groups_attn[0].__len__()) ** -0.5
    scale = float(hd) ** -0.5
    assert hpg * ngroups == heads, "o_groups must divide the local heads"
    assert q_lora % PT == 0 and dim % PT == 0 and hd % PT == 0
    wq_a4, wq_b4, wkv4 = wq_a.view(X.FP8_X4), wq_b.view(X.FP8_X4), wkv.view(X.FP8_X4)
    wo_a4, wo_b4 = wo_a.view(X.FP8_X4), wo_b.view(X.FP8_X4)
    ix_wq_b4 = ix_wq_b.view(X.FP8_X4)
    e_wkv4 = e_wkv.view(X.FP8_X4) if has_engram else None   # a 1x1 dummy otherwise
    sh_w1_4, sh_w3_4, sh_w2_4 = sh_w1.view(X.FP8_X4), sh_w3.view(X.FP8_X4), sh_w2.view(X.FP8_X4)
    w1_4, w3_4, w2_4 = w1.view(X.FP4_X4), w3.view(X.FP4_X4), w2.view(X.FP4_X4)
    world = len(groups_world[0])

    pid = nl.program_id(0)
    rg_attn = ncc.ReplicaGroup([list(g) for g in groups_attn])
    rg_world = ncc.ReplicaGroup([list(g) for g in groups_world])
    mine = tiles_of(T, pid)
    all_tiles = tiles_of(T, 0) + tiles_of(T, 1)

    # ------------------------------------------------------------------ outputs
    out_stream = nl.ndarray((T, hc, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    ffn_pre = nl.ndarray((T, hc), dtype=nl.float32, buffer=nl.shared_hbm)
    dbg_shape = (Tp, max(width, heads * hd)) if stage != 9 else (1, 1)
    dbg = nl.ndarray(dbg_shape, dtype=nl.float32, buffer=nl.shared_hbm)
    topk_out = nl.ndarray((T, topk) if is_idx_src else (1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    tap_out = nl.ndarray((T, dim) if tap else (1, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    # ------------------------------------------------------------------ scratch
    strm = stream.reshape((T, width))
    out2 = out_stream.reshape((T, width))
    xn = nl.ndarray((Tp, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="xn")
    pre_a = nl.ndarray((T, hc), dtype=nl.float32, buffer=nl.shared_hbm, name="pre_a")
    post_a = nl.ndarray((T, hc), dtype=nl.float32, buffer=nl.shared_hbm, name="post_a")
    comb_a = nl.ndarray((T, hc * hc), dtype=nl.float32, buffer=nl.shared_hbm, name="comb_a")
    post_f = nl.ndarray((T, hc), dtype=nl.float32, buffer=nl.shared_hbm, name="post_f")
    comb_f = nl.ndarray((T, hc * hc), dtype=nl.float32, buffer=nl.shared_hbm, name="comb_f")
    qa = nl.ndarray((Tp, q_lora), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="qa")
    qr = nl.ndarray((Tp, q_lora), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="qr")
    q = nl.ndarray((Tp, heads * hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="q")
    kvr = nl.ndarray((Tp, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kvr")
    kv = nl.ndarray((Tp, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kv")
    kvtab = nl.ndarray((ntab_p, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kvtab")
    idx = nl.ndarray((Tp, kpad), dtype=nl.int32, buffer=nl.shared_hbm, name="idx")
    o = nl.ndarray((Tp, heads * hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="o")
    oq = nl.ndarray((Tp, heads * hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="oq")
    z = nl.ndarray((Tp, o_lora_total), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="z")
    cc_src = nl.ndarray((Tp, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((Tp, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_dst")
    attn_out = nl.ndarray((Tp, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="attn_out")
    stream2 = nl.ndarray((T, width), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="stream2")
    # xn2 / wexp / eacc carry one extra zero tile: the padding slots of a routed
    # expert's last row tile gather from and scatter into it (`mk_moe`).
    xn2 = nl.ndarray((Tp + PT, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="xn2")
    logits = nl.ndarray((Tp, n_experts), dtype=nl.float32, buffer=nl.shared_hbm, name="logits")
    wexp = nl.ndarray((Tp + PT, n_local), dtype=nl.float32, buffer=nl.shared_hbm, name="wexp")
    eg = nl.ndarray((Tp, inter), dtype=nl.bfloat16, buffer=nl.private_hbm)
    eu = nl.ndarray((Tp, inter), dtype=nl.bfloat16, buffer=nl.private_hbm)
    ey = nl.ndarray((Tp, inter), dtype=nl.bfloat16, buffer=nl.private_hbm)
    eo32 = nl.ndarray((Tp, dim), dtype=nl.float32, buffer=nl.private_hbm)
    eacc = nl.ndarray((Tp + PT, dim), dtype=nl.float32, buffer=nl.private_hbm)
    part = nl.ndarray((CORES, Tp, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="part")
    cc2_src = nl.ndarray((Tp, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc2_src")
    cc2_dst = nl.ndarray((Tp, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc2_dst")
    ffn_out = nl.ndarray((Tp, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="ffn_out")
    # Source-layer scratch.
    pkv = nl.ndarray((Tp, hd), dtype=nl.float32, buffer=nl.shared_hbm, name="pkv")
    psc = nl.ndarray((Tp, hd), dtype=nl.float32, buffer=nl.shared_hbm, name="psc")
    lat_raw = nl.ndarray((NGp, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="lat_raw")
    lat = nl.ndarray((NGp, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="lat")
    kraw = nl.ndarray((NGp, kd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kraw")
    kn = nl.ndarray((NGp, kd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kn")
    qiraw = nl.ndarray((Tp, nh_idx * kd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="qiraw")
    qi = nl.ndarray((Tp, nh_idx * kd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="qi")
    wiraw = nl.ndarray((Tp, nh_idx), dtype=nl.float32, buffer=nl.shared_hbm, name="wiraw")
    wi = nl.ndarray((Tp, nh_idx), dtype=nl.float32, buffer=nl.shared_hbm, name="wi")
    ktab = nl.ndarray((max(nspans * read_len, 1), kd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="ktab")
    sc_src = nl.ndarray((Tp, max(read_len, 1)), dtype=nl.float32, buffer=nl.shared_hbm, name="sc_src")
    sc_dst = nl.ndarray((Tp, max(read_len, 1)), dtype=nl.float32, buffer=nl.shared_hbm, name="sc_dst")
    sc16 = nl.ndarray((Tp, max(read_len, 1)), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="sc16")
    stream0 = nl.ndarray((T, width) if has_engram else (1, 1), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="stream0")

    # ------------------------------------------------------------------ 0. engram, tap
    if has_engram:
        engram_readout(pid, mine, T, Tp, hc, dim, hash_ids, etable, erow_start, e_wkv4, e_wkv_s, e_qk,
                       strm, stream0, rg_world)
        strm = stream0
    if tap:
        for r0, nr in mine:
            tap_tile(r0, nr, strm, tap_out, hc, dim)
    if stage == 0:
        cast_rows(strm, dbg[0:T, 0:width], mine, width, nl.float32)
        nisa.core_barrier(data=dbg, cores=(0, 1))
        return dbg, ffn_pre, topk_out, tap_out, window_kv_cache, compress_kv_cache, pool_kv, pool_score, index_k_cache

    # ------------------------------------------------------------------ 1. hc_in (attention)
    fnT = load_fn_t(hca_fn_t, width, mix)
    for r0, nr in mine:
        hc_in_tile(r0, nr, strm, pre_mix, fnT, hca_base, hca_scale, attn_norm_w, hc, dim, mix,
                   xn, pre_a, post_a, comb_a)
    if pid == 0 and T < Tp:
        pad = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(pad, 0.0)
        nisa.dma_copy(dst=xn[T:Tp, :], src=pad[0 : Tp - T, :])
    nisa.core_barrier(data=xn, cores=(0, 1))
    if stage == 1:
        cast_rows(xn, dbg[0:Tp, 0:dim], all_tiles if pid == 0 else [], dim, nl.float32)
        nisa.core_barrier(data=dbg, cores=(0, 1))
        return dbg, ffn_pre, topk_out, tap_out, window_kv_cache, compress_kv_cache, pool_kv, pool_score, index_k_cache

    # ------------------------------------------------------------------ 2. q / kv projections
    # MX GEMMs: the activation is quantized on chip as it is staged (each core stages
    # every row and computes its half of the output columns).
    lo, hi = _half(q_lora, pid)
    X.gemm_rows(xn, Tp, dim, wq_a4, wq_a_s, qa, lo, hi)
    lo, hi = _half(hd, pid)
    X.gemm_rows(xn, Tp, dim, wkv4, wkv_s, kvr, lo, hi)
    nisa.core_barrier(data=qa, cores=(0, 1))
    nisa.core_barrier(data=kvr, cores=(0, 1))
    # q_norm (wq_b quantizes its input itself); kv_norm -> rope -> fp8 round trip.
    norm_rows(qa, q_norm_w, qr, mine, q_lora, qdq=False, chunk=q_lora)
    norm_rows(kvr, kv_norm_w, kv, mine, hd, qdq=False, chunk=hd)
    rope_rows(kv, kv, mine, hd, hd, 1, freqs, row_pos, inverse=False, qdq=True)
    if pid == 0 and T < Tp:
        pad = nl.ndarray((PT, q_lora), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(pad, 0.0)
        nisa.dma_copy(dst=qr[T:Tp, :], src=pad[0 : Tp - T, :])
    nisa.core_barrier(data=qr, cores=(0, 1))
    nisa.core_barrier(data=kv, cores=(0, 1))
    # This step's KV into the paged cache (the input, mutated in place).
    scatter_rows(kv, write_slots, window_kv_cache, mine, hd)
    lo, hi = _half(heads * hd, pid)
    X.gemm_rows(qr, Tp, q_lora, wq_b4, wq_b_s, q, lo, hi)
    nisa.core_barrier(data=q, cores=(0, 1))
    nisa.core_barrier(data=window_kv_cache, cores=(0, 1))
    # Rope on each head's trailing 64 columns, by row tile.
    rope_rows(q, q, mine, heads * hd, hd, heads, freqs, row_pos, inverse=False)
    nisa.core_barrier(data=q, cores=(0, 1))

    # ------------------------------------------------------------------ 2b. compressor (kv sources)
    if is_kv_src:
        gtiles = tiles_of(NG, pid)
        if ratio == 2:
            fp32_proj_rows(xn, cm_wkv_t, pkv, mine, dim, hd)
            fp32_proj_rows(xn, cm_wgate_t, psc, mine, dim, hd)
            scatter_rows_f32(pkv, write_slots, pool_kv, mine, hd)
            scatter_rows_f32(psc, write_slots, pool_score, mine, hd)
            nisa.core_barrier(data=pool_kv, cores=(0, 1))
            nisa.core_barrier(data=pool_score, cores=(0, 1))
            pool_groups(pool_kv, pool_score, grp_mslots, gtiles, ratio, hd, lat_raw)
        else:
            lo, hi = _half(hd, pid)
            bf16_gemm_rows(xn, cm_wkv, lat_raw, Tp, dim, hd, lo, hi)
        nisa.core_barrier(data=lat_raw, cores=(0, 1))
        norm_rows(lat_raw, cm_norm_w, lat, gtiles, hd, qdq=False, chunk=hd)
        nisa.core_barrier(data=lat, cores=(0, 1))
        if NG < NGp and pid == 0:
            padl = nl.ndarray((PT, hd), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(padl, 0.0)
            nisa.dma_copy(dst=lat[NG:NGp, :], src=padl[0 : NGp - NG, :])
        nisa.core_barrier(data=lat, cores=(0, 1))
        rope_fp4_put(lat, gtiles, hd, hd, 1, freqs, grp_pos, 16, True, compress_kv_cache, grp_wslots)
        # index keys from the latent
        lo, hi = _half(kd, pid) if kd >= 2 * PT else (0, kd)
        if pid == 0 or kd >= 2 * PT:
            bf16_gemm_rows(lat, ix_wk, kraw, NGp, hd, kd, lo, hi)
        nisa.core_barrier(data=kraw, cores=(0, 1))
        norm_rows(kraw, ix_knorm_w, kn, gtiles, kd, qdq=False, chunk=kd)
        rope_fp4_put(kn, gtiles, kd, kd, 1, freqs, grp_pos, 32, False, index_k_cache, grp_wslots)
        nisa.core_barrier(data=compress_kv_cache, cores=(0, 1))
        nisa.core_barrier(data=index_k_cache, cores=(0, 1))

    # ------------------------------------------------------------------ 2c. index queries (index sources)
    if is_idx_src:
        lo, hi = _half(nh_idx * kd, pid) if nh_idx * kd >= 2 * PT else (0, nh_idx * kd)
        if pid == 0 or nh_idx * kd >= 2 * PT:
            X.gemm_rows(qr, Tp, q_lora, ix_wq_b4, ix_wq_b_s, qiraw, lo, hi)
        fp32_proj_rows(xn, ix_wproj_t, wiraw, mine, dim, nh_idx)
        nisa.core_barrier(data=qiraw, cores=(0, 1))
        rope_fp4_put(qiraw, mine, nh_idx * kd, kd, nh_idx, freqs, row_pos, 32, False, dst_hbm=qi)
        for r0, nr in mine:
            w = nl.ndarray((PT, nh_idx), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w[0:nr, :], src=wiraw[r0 : r0 + nr, 0:nh_idx])
            w16 = nl.ndarray((PT, nh_idx), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=w16[0:nr, :], src=w[0:nr, :])
            nisa.tensor_scalar(dst=w[0:nr, :], data=w16[0:nr, :], op0=nl.multiply, operand0=IDX_SCALE)
            nisa.dma_copy(dst=wi[r0 : r0 + nr, 0:nh_idx], src=w[0:nr, :])
        nisa.core_barrier(data=qi, cores=(0, 1))
        nisa.core_barrier(data=wi, cores=(0, 1))
    if stage == 2:
        cast_rows(q, dbg[0:Tp, 0 : heads * hd], all_tiles if pid == 0 else [], heads * hd, nl.float32)
        nisa.core_barrier(data=dbg, cores=(0, 1))
        return dbg, ffn_pre, topk_out, tap_out, window_kv_cache, compress_kv_cache, pool_kv, pool_score, index_k_cache
    if stage == 3:
        cast_rows(kv, dbg[0:Tp, 0:hd], all_tiles if pid == 0 else [], hd, nl.float32)
        nisa.core_barrier(data=dbg, cores=(0, 1))
        return dbg, ffn_pre, topk_out, tap_out, window_kv_cache, compress_kv_cache, pool_kv, pool_score, index_k_cache

    # ------------------------------------------------------------------ 3. slot table + indices
    # Span b's rows of the slot table: its window rows, then its compressed rows.
    for b in range(nspans):
        gather_rows(window_kv_cache, win_slots[b * nwin : (b + 1) * nwin],
                    kvtab[b * nspan_rows : b * nspan_rows + nwin], tiles_of(nwin, pid), hd)
        if ratio:
            gather_rows(compress_kv_cache, cmp_slots[b * read_len : (b + 1) * read_len],
                        kvtab[b * nspan_rows + nwin : (b + 1) * nspan_rows], tiles_of(read_len, pid), hd)
        if is_idx_src:
            gather_rows(index_k_cache, cmp_slots[b * read_len : (b + 1) * read_len],
                        ktab[b * read_len : (b + 1) * read_len], tiles_of(read_len, pid), kd)
    topk_src = topk_in
    if is_idx_src:
        nisa.core_barrier(data=ktab, cores=(0, 1))
        for b in range(nspans):
            tok0 = b * L + (0 if pid == 0 else L - half)
            index_scores_span(qi, wi, ktab[b * read_len : (b + 1) * read_len], sc_src, tok0, half,
                              nh_idx, read_len, kd)
        nisa.core_barrier(data=sc_src, cores=(0, 1))
        ncc.all_reduce(srcs=[sc_src], dsts=[sc_dst], replica_group=rg_attn, op=nl.add)
        index_topk(sc_dst, clen, sc16, topk_out, T, Tp, read_len, topk, pid)
        topk_src = topk_out
    attn_index_table(idx, mine, twin, wv, win, cbase, topk_src, topk, kpad)
    if pid == 0 and T < Tp:
        padi = nl.ndarray((PT, kpad), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(padi, 0)
        nisa.dma_copy(dst=idx[T:Tp, :], src=padi[0 : Tp - T, :])
    if pid == 1 and ntab < ntab_p:
        padk = nl.ndarray((PT, hd), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(padk, 0.0)
        nisa.dma_copy(dst=kvtab[ntab:ntab_p, :], src=padk[0 : ntab_p - ntab, :])
    nisa.core_barrier(data=kvtab, cores=(0, 1))
    nisa.core_barrier(data=idx, cores=(0, 1))

    # ------------------------------------------------------------------ 4. sparse attention
    sink_t = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sink_t, src=attn_sink[0:heads, 0:1])
    # Core 0 takes rows [0, h), core 1 rows [T - h, T): equal trip counts, and for an
    # odd T the middle row is computed by both (identically).
    ntok = cdiv(T, CORES)
    tok0 = 0 if pid == 0 else T - ntok
    if resident:
        kTs = build_kT(kvtab, ntab, hd)

        def tok_body(i):
            attn_token_body_ti(i, tok0, q, kvtab, kTs, idx, o, sink_t, heads, hd, kpad, scale)
    else:
        def tok_body(i):
            attn_token_body(i, tok0, q, kvtab, idx, o, sink_t, heads, hd, kpad, scale)

    nl.fori_loop(0, ntok, tok_body)
    nisa.core_barrier(data=o, cores=(0, 1))
    if stage == 4:
        cast_rows(o, dbg[0:Tp, 0 : heads * hd], all_tiles if pid == 0 else [], heads * hd, nl.float32)
        nisa.core_barrier(data=dbg, cores=(0, 1))
        return dbg, ffn_pre, topk_out, tap_out, window_kv_cache, compress_kv_cache, pool_kv, pool_score, index_k_cache

    # ------------------------------------------------------------------ 5. output projections
    # Inverse rope, grouped wo_a, wo_b (fp32 partial); the MX GEMMs quantize their inputs.
    rope_rows(o, oq, mine, heads * hd, hd, heads, freqs, row_pos, inverse=True, qdq=False)
    if pid == 0 and T < Tp:
        pad = nl.ndarray((PT, heads * hd), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(pad, 0.0)
        nisa.dma_copy(dst=oq[T:Tp, :], src=pad[0 : Tp - T, :])
    nisa.core_barrier(data=oq, cores=(0, 1))
    for g in range(ngroups):
        # Column split of the group's `o_lora` outputs across the cores; the group's
        # block of `wo_a` is columns [g*o_lora, (g+1)*o_lora) of the MX layout against
        # the group's `g_in` input columns.
        lo, hi = _half(o_lora, pid)
        X.gemm_rows(oq, Tp, g_in, wo_a4, wo_a_s, z, g * o_lora + lo, g * o_lora + hi, x_col0=g * g_in)
    nisa.core_barrier(data=z, cores=(0, 1))
    lo, hi = _half(dim, pid)
    X.gemm_rows(z, Tp, o_lora_total, wo_b4, wo_b_s, cc_src, lo, hi, out_dtype=nl.float32)
    nisa.core_barrier(data=cc_src, cores=(0, 1))
    if stage == 51:
        cast_rows(cc_src, dbg[0:Tp, 0:dim], mine, dim, nl.float32)
        cast_rows(xn, dbg[0:Tp, dim : 2 * dim], mine, dim, nl.float32)
        cast_rows(o, dbg[0:Tp, 2 * dim : 2 * dim + heads * hd], mine, heads * hd, nl.float32)
        nisa.core_barrier(data=dbg, cores=(0, 1))
        return dbg, ffn_pre, topk_out, tap_out, window_kv_cache, compress_kv_cache, pool_kv, pool_score, index_k_cache
    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], replica_group=rg_attn, op=nl.add)
    cast_rows(cc_dst, attn_out, mine, dim, nl.bfloat16)
    if stage == 5:
        cast_rows(attn_out, dbg[0:Tp, 0:dim], mine, dim, nl.float32)
        nisa.core_barrier(data=dbg, cores=(0, 1))
        return dbg, ffn_pre, topk_out, tap_out, window_kv_cache, compress_kv_cache, pool_kv, pool_score, index_k_cache

    # ------------------------------------------------------------------ 6. hc_out (attention)
    for r0, nr in mine:
        hc_out_tile(r0, nr, attn_out, strm, post_a, comb_a, stream2, hc, dim)
    if stage == 6:
        cast_rows(stream2, dbg[0:T, 0:width], mine, width, nl.float32)
        nisa.core_barrier(data=dbg, cores=(0, 1))
        return dbg, ffn_pre, topk_out, tap_out, window_kv_cache, compress_kv_cache, pool_kv, pool_score, index_k_cache

    # ------------------------------------------------------------------ 7. hc_in (ffn)
    fnT2 = load_fn_t(hcf_fn_t, width, mix)
    for r0, nr in mine:
        hc_in_tile(r0, nr, stream2, pre_a, fnT2, hcf_base, hcf_scale, ffn_norm_w, hc, dim, mix,
                   xn2, ffn_pre, post_f, comb_f)
    if pid == 0:
        pad = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(pad, 0.0)
        if T < Tp:
            nisa.dma_copy(dst=xn2[T:Tp, :], src=pad[0 : Tp - T, :])
        nisa.dma_copy(dst=xn2[Tp : Tp + PT, :], src=pad)
    nisa.core_barrier(data=xn2, cores=(0, 1))
    if stage == 7:
        cast_rows(xn2, dbg[0:Tp, 0:dim], all_tiles if pid == 0 else [], dim, nl.float32)
        nisa.core_barrier(data=dbg, cores=(0, 1))
        return dbg, ffn_pre, topk_out, tap_out, window_kv_cache, compress_kv_cache, pool_kv, pool_score, index_k_cache

    # ------------------------------------------------------------------ 8. MoE
    # Router on both cores (fp32 GEMM, tiny), then this core's experts.
    gemm_fp32(xn2, gate_w_t, logits, tiles_of(T, 0) + tiles_of(T, 1), dim, n_experts)
    route_topk(logits, gate_b, expert_lo, tiles_of(T, 0) + tiles_of(T, 1), n_experts,
               TOPK_EXPERTS, n_local, ROUTE_SCALE, wexp)
    zw = nl.ndarray((PT, n_local), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(zw, 0.0)
    if T < Tp:
        nisa.dma_copy(dst=wexp[T:Tp, :], src=zw[0 : Tp - T, :])
    nisa.dma_copy(dst=wexp[Tp : Tp + PT, :], src=zw)
    if stage == 81:
        cast_rows(wexp, dbg[0:Tp, 0:n_local], tiles_of(T, 0) + tiles_of(T, 1) if pid == 0 else [],
                  n_local, nl.float32)
        cast_rows(logits, dbg[0:Tp, n_local : n_local + n_experts],
                  tiles_of(T, 0) + tiles_of(T, 1) if pid == 0 else [], n_experts, nl.float32)
        nisa.core_barrier(data=dbg, cores=(0, 1))
        return dbg, ffn_pre, topk_out, tap_out, window_kv_cache, compress_kv_cache, pool_kv, pool_score, index_k_cache
    ptiles = tiles_of(Tp, 0) + tiles_of(Tp, 1)
    # Routed experts, each over the rows that picked it (this core's experts), into
    # the zeroed fp32 accumulator.
    lo, hi = _half(inter, pid)
    if stage == 84:
        mk_moe.zero_rows(eacc, Tp + PT, dim)
    else:
        mk_moe.routed_experts(pid, n_local, Tp, xn2, wexp, w1_4, s1, w3_4, s3, w2_4, s2,
                              dim, inter, lo, hi, SWIGLU_LIMIT, eacc)
    # Shared expert: this core's half of the intermediate columns (SwiGLU is elementwise
    # along it and w2 contracts over it).
    if stage != 83:
        lo, hi = _half(inter, pid)
        X.gemm_rows(xn2, Tp, dim, sh_w1_4, sh_s1, eg, lo, hi)
        X.gemm_rows(xn2, Tp, dim, sh_w3_4, sh_s3, eu, lo, hi)
        swiglu_rows(eg[0:Tp, lo:hi], eu[0:Tp, lo:hi], None, 0, ey[0:Tp, lo:hi], ptiles, hi - lo, SWIGLU_LIMIT)
        # The shared expert's weights are replicated on every rank and the model adds its
        # output ONCE, after the expert all-reduce. Every rank adds it to its partial here,
        # scaled by 1/world, so the world sum carries it exactly once (a power-of-two
        # scale of the fp32 partial is exact).
        X.gemm_rows(ey, Tp, hi - lo, sh_w2_4, sh_s2, eo32, 0, dim, x_col0=lo, krow0=lo // 4,
                    out_dtype=nl.float32, out_scale=1.0 / world)
        add_rows(eacc, eo32, ptiles, dim, False)
    # Rejoin the two cores' partials, then the ranks'.
    for r0, nr in ptiles:
        t = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t[0:nr, :], src=eacc[r0 : r0 + nr, :])
        nisa.dma_copy(dst=part[pid, r0 : r0 + nr, :], src=t[0:nr, :])
    nisa.core_barrier(data=part, cores=(0, 1))
    for r0, nr in tiles_of(Tp, pid):
        a = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=a[0:nr, :], src=part[0, r0 : r0 + nr, :])
        b = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=b[0:nr, :], src=part[1, r0 : r0 + nr, :])
        nisa.tensor_tensor(dst=a[0:nr, :], data1=a[0:nr, :], data2=b[0:nr, :], op=nl.add)
        nisa.dma_copy(dst=cc2_src[r0 : r0 + nr, :], src=a[0:nr, :])
    nisa.core_barrier(data=cc2_src, cores=(0, 1))
    if stage in (82, 83, 84):
        cast_rows(cc2_src, dbg[0:Tp, 0:dim], mine, dim, nl.float32)
        cast_rows(xn2, dbg[0:Tp, dim : 2 * dim], mine, dim, nl.float32)
        nisa.core_barrier(data=dbg, cores=(0, 1))
        return dbg, ffn_pre, topk_out, tap_out, window_kv_cache, compress_kv_cache, pool_kv, pool_score, index_k_cache
    ncc.all_reduce(srcs=[cc2_src], dsts=[cc2_dst], replica_group=rg_world, op=nl.add)
    cast_rows(cc2_dst, ffn_out, mine, dim, nl.bfloat16)
    if stage == 8:
        cast_rows(ffn_out, dbg[0:Tp, 0:dim], mine, dim, nl.float32)
        nisa.core_barrier(data=dbg, cores=(0, 1))
        return dbg, ffn_pre, topk_out, tap_out, window_kv_cache, compress_kv_cache, pool_kv, pool_score, index_k_cache

    # ------------------------------------------------------------------ 9. hc_out (ffn)
    for r0, nr in mine:
        hc_out_tile(r0, nr, ffn_out, stream2, post_f, comb_f, out2, hc, dim)
    nisa.core_barrier(data=out_stream, cores=(0, 1))
    nisa.core_barrier(data=ffn_pre, cores=(0, 1))
    return out_stream, ffn_pre, topk_out, tap_out, window_kv_cache, compress_kv_cache, pool_kv, pool_score, index_k_cache


#: The generic entry (every archetype, prefill, the flat world): what the unit tests
#: and `compile_layer.py` trace. The model runs the specialized entries of `mk_kernels`.
layer_forward = nki.jit(layer_body)
