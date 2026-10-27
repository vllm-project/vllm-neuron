"""DeepSeek-V4.1-Flash `r1_tap` block, prefill, tp64 scheme -- one NKI megakernel.

The model launches it as `layer_prefill_r1_tap_tp64` for blocks 37-39: a ratio-1 consumer that
emits the DSpark tap. It is the optimized r1 prefill kernel (`../r1_prefill`, which carries the
r2 prefill kernel's optimizations too) with the sections of `../layer_kernel.layer_body` this
archetype has (generated from the template by resolving the archetype's branches), at the same
points of the dataflow, so its other phases are the optimized kernel's. The signature and the
outputs are the model's.

    hc_in -> q / kv projections -> window + top-k compressed sparse attention
          -> all_reduce(8-rank attention group) -> hc_out
          -> hc_in -> MoE (top-6 of 384: this rank's 6 experts + 1/64 of the shared one)
          -> all_reduce(64-rank world) -> hc_out

`flags` = (1, nspans, window, stage=9, 0, 0, 0, 1, kernel_version).
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl


from . import mk_moe
from . import mk_mx as X
from .mk_common import (
    PT,
    KTILE,
    attn_index_table,
    attn_compute,
    attn_unrolled_prescore,
    ATTN_PRESCORE,
    GT_PACKED,
    attn_bufs3,
    attn_fetch,
    attn_pv,
    attn_scores,
    attn_softmax,
    attn_store,
    ATTN_PIPE3,
    ATTN_PIPE3_NBUF,
    ATTN_PIPE3_STEPS,
    ATTN_TI_STEP,
    build_kT_chunks,
    _ti_block,
    attn_step,
    attn_step_bufs,
    attn_step_masks,
    attn_token_body,
    attn_token_body_ti,
    attn_tokens,
    build_kT,
    cdiv,
    gather_rows,
    gemm_fp32,
    hc_in_consts,
    hc_in_tile,
    HC_SINKHORN_BATCHED,
    sinkhorn_batched,
    hc_out_tile,
    kt_resident,
    kv_pair_layout,
    load_fn_t,
    norm_rows,
    pad_to,
    rope_rows,
    route_topk,
    scatter_rows,
    swiglu_rows,
    tap_tile,
    ti_score_psum,
    tiles_of,
)

CORES = 2
#: `swiglu_limit`, `routed_scaling_factor`, `num_experts_per_tok`.
SWIGLU_LIMIT = 10.0
ROUTE_SCALE = 1.5
TOPK_EXPERTS = 6
#: Tokens per step of the attention loop (`attn_step` stacks a step's tokens x heads on
#: partitions, so at most 128 // heads).
TOK_UNROLL = 16
#: Software-pipelined attention loop: ATTN_PIPE_STEPS steps of ATTN_PIPE_U tokens per
#: loop iteration with double-buffered gathers; each step's successor (the next
#: iteration's first step for the last one) is fetched while it computes
#: (`attn_fetch` / `attn_compute`).
ATTN_PIPE = True
ATTN_PIPE_U = 8
#: Steps per pipelined loop iteration (even: the two buffers alternate); more steps
#: amortize the per-iteration tail (last store, cross-core sync at the back-edge).
ATTN_PIPE_STEPS = 32
#: Tokens per pipelined loop iteration at most (a power of two: the q / idx / o scratch
#: rows are padded to a multiple of 3x this, so every iteration view's R rows divide them).
#: Each loop back-edge drains every engine (~30 us at 1k), so fewer, longer iterations
#: win: in1024 attention 1.774 ms (8 steps) -> 1.706 (16) -> 1.665 (32).
ATTN_PIPE_MAX_ROWS = 256
#: Select keys on chip from the resident transposed table (tensor indirection) when it
#: fits. Off: at TOK_UNROLL tokens per step the step is paced by GpSimd issuing DMAs,
#: and the TI path's per-token snake-index DMAs cost more than the PE transposes they
#: save (in1024_bs1: 17.78 ms with TI vs 17.36 ms without).
ATTN_TI = False
#: Gather a token's values with one batched (2-D vector_offset) DMA instead of one per
#: 128 keys (fewer GpSimd DMA instructions per step) when the slot table has at least
#: this many rows. Both are exact; measured: in8192 (6400 rows) 25.4 ms batched vs
#: 28.7 ms per-tile, in1024 (1792 rows) 17.05 batched vs 16.71 per-tile.
ATTN_BATCHED_GATHER_ROWS = 0
#: hc_in(attn)'s batched sinkhorn (`comb_out`) is issued after the q / kv projection
#: GEMMs instead of at the end of hc_in (its comb is first read by hc_out(attn)).
#: (Issuing it, or hc_in(ffn)'s, after the all-reduces made the attention loop 0.5 ms
#: slower: a tile live across the loop.)
COMB_DEFERRED = True
#: The rank's MoE partial (part[0] + part[1]) is formed in the shared expert's w2 GEMM
#: epilogue: the GEMM runs the other core's row tiles first (into `part`), then, after
#: the barrier, its own (the combine and the all-reduce source store in the epilogue),
#: instead of an `eo32` round trip through HBM and a separate combine pass.
PART_FUSED = True
#: hc_out (ffn): all tiles' comb*res accumulators are computed before the expert
#: all-reduce's result is needed -- HCO_KEEP tiles' in SBUF (80 KB per partition each),
#: the rest stored to an fp32 HBM scratch and read back after the all-reduce -- so DVE
#: works during the all-reduce and only the `post * x` terms follow it.
HCO_SPILL = True
HCO_KEEP = 2
#: hc_out (ffn)'s accumulator work is held after the rank partial's last combine
#: (`PART_FUSED`'s epilogue) by a data dependency (its comb tiles + 0 * that output):
#: the compiler otherwise interleaved ~90 us (8k) / ~140 us (1k) of it with the combine
#: adds on DVE, delaying the all-reduce source. (`depends_on(kind=schedule)` on the
#: accumulator ops, ACT-only or all of them, did not change the DVE order.)
HCO_AFTER_PART = True
#: The same for hc_out (attn) during the attention all-reduce (tile 0's accumulators in
#: SBUF, the others' spilled), ahead of the fused hc_out / hc_in (ffn) tiles.
HCOA_SPILL = True
#: DVE us per 1024 columns of a fused hc_out tile queued ahead of each hc_in (ffn) tile
#: (its `post * x` terms), for `mk_common.HC_IN_SPREAD`'s copy model.
HCO_DVE_US = 1.0
#: The slot table's input-only parts (compressed rows, index tables, pad rows) are built
#: during hc_in (attention) instead of after the projections (only the window rows,
#: which hold this step's kv, stay there), when the compressed part has at least this
#: many rows: in8192 (8448 rows) attention ends 40 us earlier; in1024 (1280 rows, a
#: 0.04 ms slot phase) the index math lengthens hc_in / proj more than it saves.
SLOT_EARLY_MIN_ROWS = 4096
#: The pipelined attention steps are fully unrolled (static DMA addresses, so the
#: non-gather DMAs can leave GpSimd's queue, and no loop back-edges) when the core has at
#: most this many steps.
ATTN_UNROLL_MAX_STEPS = 160
#: The shared expert's w1 / w3 GEMMs and SwiGLU issued inside `mk_moe.routed_experts`
#: (between the routed experts' static first tiles and their in-order scatter-adds and
#: loops: it overlaps them instead of following the loops' drains; MoE window -27 us at
#: in8192_bs1, -21 us at in1024_bs1).
SHARED_IN_ROUTED = True


def _half(n, pid):
    """`(lo, hi)` of core `pid`'s share of `n` columns, in whole 128-column tiles."""
    tiles = n // PT
    a = (tiles + 1) // 2
    lo = 0 if pid == 0 else a * PT
    hi = a * PT if pid == 0 else n
    return lo, hi


@nki.jit
def layer_prefill_r1_tap_tp64(
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
):
    ratio, nspans, win, stage, is_kv_src, is_idx_src, has_engram, tap, kver = flags
    assert (ratio, is_kv_src, is_idx_src, has_engram, tap) == (1, 0, 0, 0, 1), (
        f"flags {flags} are not an r1_tap block's")
    assert stage == 9, "the optimized kernel has no staged debug outputs"
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
    read_len = cmp_slots.shape[0] // nspans
    nspan_rows = nwin + read_len
    ntab = nspans * nspan_rows
    ntab_p = pad_to(ntab, PT)
    # Prefill keeps the slot table transposed in SBUF and selects keys on chip when
    # it fits; otherwise the per-token transposes.
    resident = ATTN_TI and kt_resident(ntab, hd)
    batched = ntab_p >= ATTN_BATCHED_GATHER_ROWS
    topk = min(topk_in.shape[1], read_len)
    kpad = pad_to(win + topk, KTILE)
    scale = float(hd) ** -0.5
    assert hpg * ngroups == heads, "o_groups must divide the local heads"
    assert q_lora % PT == 0 and dim % PT == 0 and hd % PT == 0
    wq_a4, wq_b4, wkv4 = wq_a.view(X.FP8_X4), wq_b.view(X.FP8_X4), wkv.view(X.FP8_X4)
    wo_a4, wo_b4 = wo_a.view(X.FP8_X4), wo_b.view(X.FP8_X4)
    sh_w1_4, sh_w3_4, sh_w2_4 = sh_w1.view(X.FP8_X4), sh_w3.view(X.FP8_X4), sh_w2.view(X.FP8_X4)
    w1_4, w3_4, w2_4 = w1.view(X.FP4_X4), w3.view(X.FP4_X4), w2.view(X.FP4_X4)
    world = len(groups_world[0])

    pid = nl.program_id(0)
    rg_attn = ncc.ReplicaGroup([list(g) for g in groups_attn])
    rg_world = ncc.ReplicaGroup([list(g) for g in groups_world])
    mine = tiles_of(T, pid)

    # ------------------------------------------------------------------ outputs
    out_stream = nl.ndarray((T, hc, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    ffn_pre = nl.ndarray((T, hc), dtype=nl.float32, buffer=nl.shared_hbm)
    topk_out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    tap_out = nl.ndarray((T, dim), dtype=nl.float32, buffer=nl.shared_hbm)

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
    # q / idx / idxp carry PT extra rows: the pipelined attention loop's last iteration
    # prefetches (and discards) one step past the core's tokens
    # (rows padded to a multiple of 3 * ATTN_PIPE_MAX_ROWS so the pipelined loops' views
    # of R rows a step-row, R = steps x U tokens, divide them; the rows past Tp + PT are
    # never read)
    Tq = pad_to(Tp + PT, 3 * ATTN_PIPE_MAX_ROWS)
    To = pad_to(Tp, 3 * ATTN_PIPE_MAX_ROWS)
    q = nl.ndarray((Tq, heads * hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="q")
    kvr = nl.ndarray((Tp, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kvr")
    kv = nl.ndarray((Tp, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kv")
    # the slot table, then (batched gathers) 128 zero rows the empty slots gather from
    nzero = PT if batched else 0
    kvtab = nl.ndarray((ntab_p + nzero, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kvtab")
    idx = nl.ndarray((Tq, kpad), dtype=nl.int32, buffer=nl.shared_hbm, name="idx")
    # idx permuted for the attention loop's batched value gathers (`attn_index_table`)
    idxp = nl.ndarray((Tq, kpad), dtype=nl.int32, buffer=nl.shared_hbm, name="idxp")
    # `ATTN_TI_STEP` (slot tables that fit the resident transposed layout): the snake rows
    ti_tab = ATTN_TI_STEP and kv_pair_layout(hd) and batched and not resident and kt_resident(ntab, hd)
    snk = nl.ndarray((Tq, kpad), dtype=nl.int32, buffer=nl.shared_hbm, name="snk") if ti_tab else None
    o = nl.ndarray((To, heads * hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="o")
    oq = nl.ndarray((Tp, heads * hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="oq")
    z = nl.ndarray((Tp, o_lora_total), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="z")
    cc_src = nl.ndarray((Tp, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((Tp, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_dst")
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
    # hc_out (ffn)'s spilled comb*res accumulators (`HCO_SPILL`, this core's rows)
    hacc = nl.ndarray((Tp, hc * dim), dtype=nl.float32, buffer=nl.private_hbm) if HCO_SPILL else None
    # hc_out (attn)'s (`HCOA_SPILL`)
    hacc_a = nl.ndarray((Tp, hc * dim), dtype=nl.float32, buffer=nl.private_hbm) if HCOA_SPILL else None

    # The slot table's compressed rows, the index tables and the pad rows read only
    # inputs (not this step's kv): `SLOT_EARLY` builds them during hc_in (GpSimd gathers
    # from the start, the index math per hc_in tile) instead of after the projections.
    kvp = kv_pair_layout(hd)
    assert not (resident and kvp), "the resident (TI) path reads the natural column order"
    SLOT_EARLY = nspans * read_len >= SLOT_EARLY_MIN_ROWS
    # (not what this step writes: a kv source's compressed rows, an index source's top-k)
    early_cmp = SLOT_EARLY
    early_idx = SLOT_EARLY

    def slot_cmp():
        for b in range(nspans):
            gather_rows(compress_kv_cache, cmp_slots[b * read_len : (b + 1) * read_len],
                        kvtab[b * nspan_rows + nwin : (b + 1) * nspan_rows], tiles_of(read_len, pid), hd,
                        pair=kvp)

    def slot_index(tiles, topk_src):
        attn_index_table(idx, tiles, twin, wv, win, cbase, topk_src, topk, kpad,
                         idxp_hbm=idxp if batched else None, zrow0=ntab_p if batched else None,
                         snk_hbm=snk)

    def slot_pads():
        if pid == 0:
            # pad rows (read only by the attention loop's discarded last prefetch): index 0;
            # the gather table's point at the 128 distinct zero rows (`zrow0 + a` at the
            # transposed position `a * kt + q`), not all at one row (hot-row DMA)
            padi = nl.ndarray((PT, kpad), dtype=nl.int32, buffer=nl.sbuf)
            nisa.memset(padi, 0)
            if batched:
                ktl = kpad // PT
                zp = nl.ndarray((PT, PT), dtype=nl.int32, buffer=nl.sbuf)
                nisa.iota(dst=zp, pattern=[[1, PT]], offset=ntab_p, channel_multiplier=0)
                padp = nl.ndarray((PT, PT, ktl), dtype=nl.int32, buffer=nl.sbuf)
                for qq in range(ktl):
                    nisa.tensor_copy(dst=padp[0:PT, :, qq], src=zp)
                padpf = padp.reshape((PT, kpad))
            for z0 in range(T, Tp + PT, PT):
                z1 = min(z0 + PT, Tp + PT)
                nisa.dma_copy(dst=idx[z0:z1, :], src=padi[0 : z1 - z0, :])
                if batched:
                    nisa.dma_copy(dst=idxp[z0:z1, :], src=padpf[0 : z1 - z0, :])
                if snk is not None:
                    nisa.dma_copy(dst=snk[z0:z1, :], src=padi[0 : z1 - z0, :])
        if pid == 1 and ntab < ntab_p + nzero:
            padk = nl.ndarray((PT, hd), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(padk, 0.0)
            for z0 in range(ntab, ntab_p + nzero, PT):
                z1 = min(z0 + PT, ntab_p + nzero)
                nisa.dma_copy(dst=kvtab[z0:z1, :], src=padk[0 : z1 - z0, :])


    # ------------------------------------------------------------------ 0b. DSpark tap
    for r0, nr in mine:
        tap_tile(r0, nr, strm, tap_out, hc, dim)

    # ------------------------------------------------------------------ 1. hc_in (attention)
    fnT = load_fn_t(hca_fn_t, width, mix)
    hcc = hc_in_consts(hca_base, hca_scale, attn_norm_w, hc, dim)
    nmine = len(mine)

    def t2_buf():
        # the call's pre-sinkhorn comb logits, one slot per row tile (zeros past a
        # partial tile's rows)
        if not HC_SINKHORN_BATCHED or nmine == 0:
            return None
        t = nl.ndarray((PT, nmine, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(t, 0.0)
        return t

    def comb_out(t2all, comb_hbm):
        if t2all is None:
            return
        cb = sinkhorn_batched(t2all, nmine, hc)
        for t, (r0, nr) in enumerate(mine):
            nisa.dma_copy(dst=comb_hbm[r0 : r0 + nr, 0 : hc * hc], src=cb[0:nr, t, :])

    t2a = t2_buf()
    if early_cmp:
        slot_cmp()
    for t, (r0, nr) in enumerate(mine):
        hc_in_tile(r0, nr, strm, pre_mix, fnT, hca_base, hca_scale, attn_norm_w, hc, dim, mix,
                   xn, pre_a, post_a, comb_a, consts=hcc,
                   t2_dst=None if t2a is None else t2a[:, t, :])
        if early_idx:
            slot_index([(r0, nr)], topk_in)
    if SLOT_EARLY:
        slot_pads()
    if not COMB_DEFERRED:
        comb_out(t2a, comb_a)
    if pid == 0 and T < Tp:
        pad = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(pad, 0.0)
        nisa.dma_copy(dst=xn[T:Tp, :], src=pad[0 : Tp - T, :])
    # The routed experts' fp32 accumulator (private to this core) is zeroed here, off
    # the MoE's critical path: these writes overlap the DVE-bound hc_in / projections.
    mk_moe.zero_rows(eacc, Tp + PT, dim)
    nisa.core_barrier(data=xn, cores=(0, 1))

    # ------------------------------------------------------------------ 2. q / kv projections
    # MX GEMMs: the activation is quantized on chip as it is staged (each core stages
    # every row and computes its half of the output columns).
    # wq_a and wkv share one staging of the xn rows.
    X.gemm_rows_multi(xn, Tp, dim, [(wq_a4, wq_a_s, qa) + _half(q_lora, pid),
                                    (wkv4, wkv_s, kvr) + _half(hd, pid)])
    if COMB_DEFERRED:
        # hc_in(attn)'s sinkhorn alongside the projection GEMMs (comb_a is first read by
        # hc_out(attn), after the attention)
        comb_out(t2a, comb_a)
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



    # ------------------------------------------------------------------ 3. slot table + indices
    # Span b's rows of the slot table: its window rows, then its compressed rows.
    # (in the pair-interleaved column order the attention transposes from, `kv_pair_layout`)
    # (`SLOT_EARLY`: everything but the window rows was built during hc_in)
    for b in range(nspans):
        gather_rows(window_kv_cache, win_slots[b * nwin : (b + 1) * nwin],
                    kvtab[b * nspan_rows : b * nspan_rows + nwin], tiles_of(nwin, pid), hd, pair=kvp)
    if not early_cmp:
        slot_cmp()
    # (`layer_kernel`'s index source: this step's top-k from the index keys)
    topk_src = topk_in
    if not early_idx:
        slot_index(mine, topk_src)
    if not SLOT_EARLY:
        slot_pads()
    nisa.core_barrier(data=kvtab, cores=(0, 1))
    nisa.core_barrier(data=idx, cores=(0, 1))
    if batched:
        nisa.core_barrier(data=idxp, cores=(0, 1))
    if snk is not None:
        nisa.core_barrier(data=snk, cores=(0, 1))

    # ------------------------------------------------------------------ 4. sparse attention
    sink_t = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sink_t, src=attn_sink[0:heads, 0:1])
    # Tokens per loop step: TOK_UNROLL, at most 128 stacked (token, head) rows for
    # `attn_step`, a power of two (it divides Tp).
    pipe = ATTN_PIPE and batched and not resident
    U = TOK_UNROLL if resident else max(1, min(ATTN_PIPE_U if pipe else TOK_UNROLL, PT // heads))
    while Tp % U:
        U //= 2
    # the sinks repeated for each token of a step (`attn_step`'s stacked rows)
    sink_u = nl.ndarray((U * heads, 1), dtype=nl.float32, buffer=nl.sbuf)
    for u in range(U):
        nisa.dma_copy(dst=sink_u[u * heads : (u + 1) * heads, :], src=attn_sink[0:heads, 0:1])
    masks = None if resident else attn_step_masks(U, heads, hd, kpad)
    # Core 0 takes rows [0, h), core 1 rows [T - h, T): equal trip counts, and for an
    # odd T the middle row is computed by both (identically).
    ntok = cdiv(T, CORES)
    tok0 = 0 if pid == 0 else T - ntok
    # TOK_UNROLL tokens per loop step (the tables viewed with TOK_UNROLL rows per step),
    # then the remainder one by one; the trip counts depend on T only.
    if resident:
        kTs = build_kT(kvtab, ntab, hd)
        # two PSUM score sets, shared round-robin by the step's tokens (PSUM is 8 banks)
        accs = [ti_score_psum(heads, kpad) for _ in range(min(TOK_UNROLL, 2))]
    else:
        kTs, accs = None, None
    nstep = ntok // U
    qU = q.reshape((Tq // U, U * heads * hd))
    idxU = idx.reshape((Tq // U, U * kpad))
    oU = o.reshape((To // U, U * heads * hd))
    idxpU = idxp.reshape((Tq // U, U * kpad)) if batched else None
    # pipelined: ATTN_PIPE_STEPS steps per iteration (views stepping R rows); the
    # 3-buffer loop ATTN_PIPE3_STEPS (it prefetches up to 3 steps past the core's tokens)
    NB = ATTN_PIPE3_NBUF
    pipe3 = pipe and ATTN_PIPE3 and NB * U <= PT and ntok >= ATTN_PIPE3_STEPS * U
    if pipe3:
        nsp = ATTN_PIPE3_STEPS
    else:
        nsp = ATTN_PIPE_STEPS
        # (more than 128 tokens per iteration only when that leaves no more tokens to the
        # unpipelined remainder loops than 128 would)
        while nsp > 2 and (nsp * U > ATTN_PIPE_MAX_ROWS or ntok < nsp * U
                           or (nsp * U > PT and ntok % (nsp * U) >= PT)):
            nsp //= 2
    npipe = ntok // (nsp * U) if pipe else 0
    # small token counts: the pipelined steps fully unrolled (all `nstep` steps)
    unroll = pipe and not pipe3 and 1 < nstep <= ATTN_UNROLL_MAX_STEPS
    if unroll:
        npipe, nsp = 1, nstep

    def step_body(i):
        if resident:
            attn_tokens(i, [tok0 + u for u in range(U)], qU, kvtab, kTs, accs, idxU, oU,
                        sink_t, heads, hd, kpad, scale)
        else:
            attn_step(i, tok0, U, qU, kvtab, idxU, oU, sink_u, masks, heads, hd, kpad, scale,
                      idxp_hbm=idxpU)

    def tok_body(i):
        if resident:
            attn_token_body_ti(i, tok0, q, kvtab, kTs, accs[0], idx, o, sink_t, heads, hd, kpad, scale)
        else:
            attn_token_body(i, tok0, q, kvtab, idx, o, sink_t, heads, hd, kpad, scale)

    if npipe > 0 and pipe3:
        R = nsp * U
        q2 = q.reshape((Tq // R, R * heads * hd))
        idx2 = idx.reshape((Tq // R, R * kpad))
        o2 = o.reshape((To // R, R * heads * hd))
        idxp2 = idxp.reshape((Tq // R, R * kpad))
        b3 = [attn_bufs3(U, heads, hd, kpad) for _ in range(NB)]
        # prologue: the first NB steps' loads, the first step's scores and softmax
        for st in range(NB):
            attn_fetch(None, tok0 + st * U, U, q, kvtab, idx, heads, hd, kpad, b3[st], idxp_hbm=idxp)
        sts = attn_scores(U, b3[0], masks, heads, hd, kpad)
        attn_softmax(U, sts, sink_u, heads, kpad, scale, b3[0])

        def pipe3_body(k):
            # step s (buffer s % 3, its softmax done one stage earlier): step s+1's
            # scores, step s's PV, step s+3's loads into step s's buffer (a token's gather
            # as soon as the PV has read its values; step nsp.. = the next iteration's),
            # step s+1's softmax, step s's store (after the gathers on GpSimd's queue)
            for st in range(nsp):
                cur, nxt = b3[st % NB], b3[(st + 1) % NB]
                sts = attn_scores(U, nxt, masks, heads, hd, kpad)
                pvs = [attn_pv(U, cur, masks, heads, hd, kpad, g, cur["p16"][g]) for g in range(len(sts))]
                attn_fetch(k, tok0 + (st + NB) * U, U, q2, kvtab, idx2, heads, hd, kpad, cur,
                           idxp_hbm=idxp2)
                attn_softmax(U, sts, sink_u, heads, kpad, scale, nxt)
                for g, acc_o in enumerate(pvs):
                    attn_store(k, tok0 + st * U, U, o2, acc_o, cur["rcp"][g], heads, hd, g)

        nl.fori_loop(0, npipe, pipe3_body)
    elif npipe > 0:
        R = nsp * U
        if not unroll:
            q2 = q.reshape((Tq // R, R * heads * hd))
            idx2 = idx.reshape((Tq // R, R * kpad))
            o2 = o.reshape((To // R, R * heads * hd))
            idxp2 = idxp.reshape((Tq // R, R * kpad))
        # `ATTN_TI_STEP`: keys read from the resident transposed table (small tables)
        ti = snk is not None
        kTsr = build_kT_chunks(kvtab, ntab_p, hd) if ti else None
        tacc = ([nl.ndarray((U * heads, _ti_block(kpad)), dtype=nl.float32, buffer=nl.psum)
                 for _ in range(kpad // _ti_block(kpad))] if ti else None)
        bufA = attn_step_bufs(U, heads, hd, kpad, ti=ti)
        bufB = attn_step_bufs(U, heads, hd, kpad, ti=ti)
        # prologue: the first step's loads
        attn_fetch(None, tok0, U, q, kvtab, idx, heads, hd, kpad, bufA, idxp_hbm=idxp, snk_hbm=snk)
        snk2 = snk.reshape((Tq // R, R * kpad)) if (ti and not unroll) else None

        bufs = (bufA, bufB)

        def pipe_body(k):
            # step s computes from bufs[s % 2] (step 0 fetched by the previous iteration);
            # step s + 1 is fetched into the other buffer first, once step s - 1's PV has
            # read it (step nsp = the next iteration's step 0)
            prev = None
            for st in range(nsp):
                attn_fetch(k, tok0 + (st + 1) * U, U, q2, kvtab, idx2, heads, hd, kpad,
                           bufs[(st + 1) % 2], idxp_hbm=idxp2, snk_hbm=snk2)
                prev = attn_compute(k, tok0 + st * U, U, o2, bufs[st % 2], sink_u, masks, heads, hd,
                                    kpad, scale, after=prev, kTs=kTsr, ti_accs=tacc)

        if unroll and ATTN_PRESCORE and (ti or (kv_pair_layout(hd) and GT_PACKED)):
            # `ATTN_PRESCORE`: step s + 1's first token scored inside step s's softmax gap
            qtm2 = nl.ndarray(masks[0].shape, dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(qtm2, 0.0)
            taccs = None
            if ti:
                taccs = (tacc, [nl.ndarray((U * heads, _ti_block(kpad)), dtype=nl.float32, buffer=nl.psum)
                                for _ in range(kpad // _ti_block(kpad))])
            attn_unrolled_prescore(nstep, tok0, U, q, kvtab, idx, o, sink_u, masks, qtm2, heads, hd,
                                   kpad, scale, idxp, bufs, snk_hbm=snk, kTs=kTsr, taccs=taccs)
        elif unroll:
            # every step unrolled (static addresses: `ATTN_STATIC_HWDGE`; no loop back-edge
            # draining the engines); the last step prefetches nothing
            prev = None
            for s in range(nstep):
                if s + 1 < nstep:
                    attn_fetch(None, tok0 + (s + 1) * U, U, q, kvtab, idx, heads, hd, kpad,
                               bufs[(s + 1) % 2], idxp_hbm=idxp, snk_hbm=snk)
                prev = attn_compute(None, tok0 + s * U, U, o, bufs[s % 2], sink_u, masks, heads, hd,
                                    kpad, scale, after=prev, kTs=kTsr, ti_accs=tacc)
        else:
            nl.fori_loop(0, npipe, pipe_body)
    if nstep > nsp * npipe:
        nl.fori_loop(nsp * npipe, nstep, step_body)
    if ntok > nstep * U:
        nl.fori_loop(nstep * U, ntok, tok_body)
    nisa.core_barrier(data=o, cores=(0, 1))

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
    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], replica_group=rg_attn, op=nl.add)

    # ------------------------------------------------------------------ 6. hc_out (attention)
    # hc_out reads the fp32 all-reduce result into bf16 by DMA: the bf16 cast of the
    # attention output, without a bf16 copy through HBM.
    # ------------------------------------------------------------------ 7. hc_in (ffn)
    # fused with 6 by row tile: hc_in reads the tile hc_out just produced from SBUF (the
    # stream2 rows are still stored for hc_out (ffn)'s residual).
    fnT2 = load_fn_t(hcf_fn_t, width, mix)
    hcc2 = hc_in_consts(hcf_base, hcf_scale, ffn_norm_w, hc, dim)
    # skewed: tile t+1's hc_out is issued before tile t's hc_in
    yss = [None] * len(mine)
    t2f = t2_buf()
    spill_a = HCOA_SPILL and nmine > 1
    if spill_a:
        # every tile's comb*res accumulators while the all-reduce is in flight: tile 0's in
        # SBUF, the others' stored to `hacc_a` (fp32) and read back after it (`HCO_SPILL`)
        acc0 = hc_out_tile(mine[0][0], mine[0][1], cc_dst, strm, post_a, comb_a, stream2, hc, dim,
                           acc_only=True)
        for r0, nr in mine[1:]:
            hc_out_tile(r0, nr, cc_dst, strm, post_a, comb_a, stream2, hc, dim, acc_only=True,
                        acc_hbm=hacc_a)
    for t in range(len(mine) + 1):
        if t < len(mine):
            r0, nr = mine[t]
            if spill_a:
                yss[t] = hc_out_tile(r0, nr, cc_dst, strm, post_a, comb_a, stream2, hc, dim, keep=True,
                                     accs_in=acc0 if t == 0 else None,
                                     acc_hbm=None if t == 0 else hacc_a)
            else:
                yss[t] = hc_out_tile(r0, nr, cc_dst, strm, post_a, comb_a, stream2, hc, dim, keep=True,
                                     acc_first=(t == 0))
        if t > 0:
            r0, nr = mine[t - 1]
            hc_in_tile(r0, nr, stream2, pre_a, fnT2, hcf_base, hcf_scale, ffn_norm_w, hc, dim, mix,
                       xn2, ffn_pre, post_f, comb_f, consts=hcc2, xin=yss[t - 1],
                       t2_dst=None if t2f is None else t2f[:, t - 1, :],
                       v0=HCO_DVE_US * hc * dim / 1024 if t < len(mine) else 0.0)
    comb_out(t2f, comb_f)
    if pid == 0:
        pad = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(pad, 0.0)
        if T < Tp:
            nisa.dma_copy(dst=xn2[T:Tp, :], src=pad[0 : Tp - T, :])
        nisa.dma_copy(dst=xn2[Tp : Tp + PT, :], src=pad)
    nisa.core_barrier(data=xn2, cores=(0, 1))

    # ------------------------------------------------------------------ 8. MoE
    # Router (fp32 GEMM + top-k) split by row tile between the cores, then every core
    # reads all rows' expert weights for its experts.
    gemm_fp32(xn2, gate_w_t, logits, mine, dim, n_experts)
    route_topk(logits, gate_b, expert_lo, mine, n_experts, TOPK_EXPERTS, n_local, ROUTE_SCALE, wexp)
    if pid == 0:
        zw = nl.ndarray((PT, n_local), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(zw, 0.0)
        if T < Tp:
            nisa.dma_copy(dst=wexp[T:Tp, :], src=zw[0 : Tp - T, :])
        nisa.dma_copy(dst=wexp[Tp : Tp + PT, :], src=zw)
    nisa.core_barrier(data=wexp, cores=(0, 1))
    ptiles = tiles_of(Tp, 0) + tiles_of(Tp, 1)
    # Routed experts, each over the rows that picked it (this core's experts), into
    # the zeroed fp32 accumulator.
    lo, hi = _half(inter, pid)

    def shared_up():
        # Shared expert: this core's half of the intermediate columns (SwiGLU is elementwise
        # along it and w2 contracts over it). Independent of the routed accumulator:
        # `SHARED_IN_ROUTED` issues it inside the routed experts (after their static first
        # tiles, before their in-order scatter-adds and loops).
        X.gemm_rows_multi(xn2, Tp, dim, [(sh_w1_4, sh_s1, eg, lo, hi), (sh_w3_4, sh_s3, eu, lo, hi)])
        swiglu_rows(eg[0:Tp, lo:hi], eu[0:Tp, lo:hi], None, 0, ey[0:Tp, lo:hi], ptiles, hi - lo,
                    SWIGLU_LIMIT)

    mk_moe.routed_experts(pid, n_local, Tp, xn2, wexp, w1_4, s1, w3_4, s3, w2_4, s2,
                          dim, inter, lo, hi, SWIGLU_LIMIT, eacc, zero_acc=False,
                          mid=shared_up if SHARED_IN_ROUTED else None)
    if not SHARED_IN_ROUTED:
        shared_up()
    # The shared expert's weights are replicated on every rank and the model adds its
    # output ONCE, after the expert all-reduce. Every rank adds it to its partial here,
    # scaled by 1/world, so the world sum carries it exactly once (a power-of-two
    # scale of the fp32 partial is exact).
    # This core's partial, eacc + eo (fp32), is formed in the w2 GEMM's epilogue (eo never
    # goes through HBM): the other core's rows go to this core's `part` slot, its own rows
    # to `eo32`; after the barrier each core rejoins its own rows, part[0] + part[1], then
    # the ranks'. (Do NOT split the all-reduces by rows to overlap them: the collectives'
    # per-element summation order depends on the message size -- not bit-exact.)
    other = {r for r, _ in tiles_of(Tp, 1 - pid)}
    # `HCO_AFTER_PART`: an exact zero column (0 * every own-row combine's output)
    last_comb = [None]
    if HCO_AFTER_PART and PART_FUSED and HCO_SPILL:
        last_comb[0] = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(last_comb[0], 0.0)

    def w2_epi(r, n0, w, o):
        av = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=av, src=eacc[r : r + PT, n0 : n0 + w])
        nisa.tensor_tensor(dst=av, data1=av, data2=o, op=nl.add, engine=nisa.engine.gpsimd)
        if PART_FUSED and r not in other:
            # this core's own rows (second pass, after the other core's `part` rows are
            # written): the rank's partial part[0] + part[1] formed here, straight into
            # the all-reduce source
            bv = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=bv, src=part[1 - pid, r : r + PT, n0 : n0 + w])
            p0, p1 = (av, bv) if pid == 0 else (bv, av)
            nisa.tensor_tensor(dst=av, data1=p0, data2=p1, op=nl.add)
            if last_comb[0] is not None:
                # `HCO_AFTER_PART`: the running exact zero column depends on every combine
                nisa.scalar_tensor_tensor(dst=last_comb[0], data=av[0:PT, 0:1], op0=nl.multiply,
                                          operand0=0.0, op1=nl.add, operand1=last_comb[0])
            nisa.dma_copy(dst=cc2_src[r : r + PT, n0 : n0 + w], src=av)
            return
        dst = part[pid, r : r + PT, n0 : n0 + w] if r in other else eo32[r : r + PT, n0 : n0 + w]
        nisa.dma_copy(dst=dst, src=av)

    if PART_FUSED:
        # two passes over the w2 GEMM's row tiles: the other core's rows into this core's
        # `part` slot, the barrier, then this core's rows (the combine in the epilogue)
        w2_args = dict(out_dtype=nl.float32, out_scale=1.0 / world, x_col0=lo, krow0=lo // 4)
        X.gemm_row_list(ey, [r for r, _ in tiles_of(Tp, 1 - pid)], hi - lo, sh_w2_4, sh_s2, 0, dim,
                        w2_epi, **w2_args)
        nisa.core_barrier(data=part, cores=(0, 1))
        X.gemm_row_list(ey, [r for r, _ in tiles_of(Tp, pid)], hi - lo, sh_w2_4, sh_s2, 0, dim,
                        w2_epi, **w2_args)
    else:
        X.gemm_rows(ey, Tp, hi - lo, sh_w2_4, sh_s2, eo32, 0, dim, x_col0=lo, krow0=lo // 4,
                    out_dtype=nl.float32, out_scale=1.0 / world, epilogue=w2_epi)
        ach = min(1024, dim)
        nisa.core_barrier(data=part, cores=(0, 1))
        for r0, nr in tiles_of(Tp, pid):
            for c in range(dim // ach):
                c0, c1 = c * ach, (c + 1) * ach
                av = nl.ndarray((PT, ach), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=av[0:nr, :], src=eo32[r0 : r0 + nr, c0:c1])
                bv = nl.ndarray((PT, ach), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=bv[0:nr, :], src=part[1 - pid, r0 : r0 + nr, c0:c1])
                p0, p1 = (av, bv) if pid == 0 else (bv, av)
                nisa.tensor_tensor(dst=av[0:nr, :], data1=p0[0:nr, :], data2=p1[0:nr, :], op=nl.add)
                nisa.dma_copy(dst=cc2_src[r0 : r0 + nr, c0:c1], src=av[0:nr, :])
    nisa.core_barrier(data=cc2_src, cores=(0, 1))
    ncc.all_reduce(srcs=[cc2_src], dsts=[cc2_dst], replica_group=rg_world, op=nl.add)

    # ------------------------------------------------------------------ 9. hc_out (ffn)
    # (the bf16 cast of the ffn output in hc_out's DMA, as in 6)
    # one tile of lookahead: tile t+1's comb*res accumulators are issued before tile t's
    # x-dependent half, so two tiles' accumulate while the all-reduce is in flight
    # (`HCO_AFTER_PART`: scheduled after the rank partial's last combine, so they fill the
    # all-reduce instead of stretching the MoE tail that feeds it)
    # (a data dependency: the accumulators' comb tiles get + that zero column)
    zdep = last_comb[0]

    def hco_acc(t):
        r0, nr = mine[t]
        return hc_out_tile(r0, nr, cc2_dst, stream2, post_f, comb_f, out2, hc, dim, acc_only=True,
                           comb_dep=zdep)

    if HCO_SPILL:
        # every tile's comb*res accumulators before the first all-reduce-dependent op: the
        # first HCO_KEEP tiles' kept in SBUF, the others' stored to `hacc` (fp32) while the
        # all-reduce is in flight and read back by DMA after it (DVE is idle meanwhile)
        nkeep = min(nmine, HCO_KEEP)
        kept = [hco_acc(t) for t in range(nkeep)]
        for t in range(nkeep, nmine):
            r0, nr = mine[t]
            hc_out_tile(r0, nr, cc2_dst, stream2, post_f, comb_f, out2, hc, dim, acc_only=True,
                        acc_hbm=hacc, comb_dep=zdep)
        for t, (r0, nr) in enumerate(mine):
            if t < nkeep:
                hc_out_tile(r0, nr, cc2_dst, stream2, post_f, comb_f, out2, hc, dim, accs_in=kept[t])
            else:
                hc_out_tile(r0, nr, cc2_dst, stream2, post_f, comb_f, out2, hc, dim, acc_hbm=hacc)
    else:
        pend = hco_acc(0) if nmine else None
        for t, (r0, nr) in enumerate(mine):
            nxt = hco_acc(t + 1) if t + 1 < nmine else None
            hc_out_tile(r0, nr, cc2_dst, stream2, post_f, comb_f, out2, hc, dim, accs_in=pend)
            pend = nxt
    nisa.core_barrier(data=out_stream, cores=(0, 1))
    nisa.core_barrier(data=ffn_pre, cores=(0, 1))
    return out_stream, ffn_pre, topk_out, tap_out, window_kv_cache, compress_kv_cache, pool_kv, pool_score, index_k_cache



#: The kernel on both physical cores of the logical NeuronCore (LNC=2).
kernel = layer_prefill_r1_tap_tp64[2]
