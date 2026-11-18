"""DeepSeek-V4.1-Flash `r2_src_idx_e` block, decode, tp64 scheme -- one NKI megakernel.

The model launches it as `layer_decode_r2_src_idx_e_tp64` for block 14: a ratio-2 kv and index
source after the engram read-out. It is the optimized r2 decode kernel (`../r2_decode`) with the
sections of `../layer_kernel.layer_body` this archetype has (generated from the template by
resolving the archetype's branches), at the same points of the dataflow, so its other phases are
the optimized kernel's. The signature and the outputs are the model's.

    hc_in -> q / kv projections -> window + top-k compressed sparse attention
          -> all_reduce(8-rank attention group) -> hc_out
          -> hc_in -> MoE (top-6 of 384: this rank's 6 experts + 1/64 of the shared one)
          -> all_reduce(64-rank world) -> hc_out

`flags` = (2, nspans, window, stage=9, 1, 1, 1, 0, kernel_version).
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
    load_freqs,
    PT,
    KTILE,
    attn_index_table,
    attn_compute,
    attn_bufs3,
    attn_fetch,
    attn_pv,
    attn_scores,
    attn_softmax,
    attn_store,
    ATTN_PIPE3,
    ATTN_PIPE3_NBUF,
    ATTN_PIPE3_STEPS,
    attn_step,
    attn_step_bufs,
    attn_step_masks,
    attn_token_body,
    attn_token_body_ti,
    attn_tokens,
    build_kT,
    cdiv,
    engram_readout,
    gather_rows,
    gather_rows_cores,
    gemm_fp32,
    hc_in_consts,
    hc_in_tile,
    HC_SINKHORN_BATCHED,
    sinkhorn_batched,
    hc_out_tile,
    HC_OUT_SPREAD,
    hc_out_spread_acc,
    hc_out_spread_out,
    spread_of,
    kt_resident,
    kv_pair_layout,
    load_fn_t,
    norm_rows,
    pad_to,
    rope_rows,
    route_topk,
    _sched_after,
    scatter_rows,
    swiglu_rows,
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
ATTN_PIPE_STEPS = 8
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
#: Decode: hc_in(attn)'s sinkhorn is issued right after the attention all-reduce (its
#: comb is first read by hc_out(attn), after it) and hc_in(ffn)'s right after the world
#: all-reduce (read by the final hc_out): the serial 20-iteration DVE/ACT chains run in
#: the all-reduce waits instead of delaying norm_rows and the router. (Only with one row
#: tile: in prefill a live tile across the attention loop slowed it.)
COMB_IN_AR = True
#: The slot table's per-span slot indices loaded by one hardware-DGE DMA per core
#: (`gather_rows_cores`; the gathers stay one per tile). (An earlier version that also
#: gathered 8 tiles per 2-D vector_offset DMA and stored them with one DMA gave WRONG
#: values -- not debugged.)
SLOT_GATHER_GROUPED = True
#: The shared expert's w1 / w3 GEMMs and SwiGLU before the router (w2 stays after the
#: routed experts: its epilogue adds their accumulator).
SHARED_FIRST = True
#: Each core's attention tokens rounded up to whole loop steps (the cores' ranges
#: overlap; a row both compute is written identically by both; rows past T are the
#: zero padding rows and are discarded), so a decode step's 3 or 30 tokens per core
#: run as pipelined U-token steps instead of partly through the per-token loop.
ATTN_WHOLE_STEPS = True
#: With a single row tile (decode), the router GEMM is split by expert columns across
#: the cores instead of by row tiles (where core 1 would have no rows).
ROUTER_COL_SPLIT = True
#: The world all-reduce carries only the T real rows (not the 128-row padding) up to
#: this many rows.
CC2_T_ROWS_MAX = 8
#: The pass-through cache outputs (`window_kv_cache` after this step's rows are
#: scattered in, `compress_kv_cache`, `index_k_cache`) are fresh tensors filled by our own
#: HBM -> HBM copies, split across the cores and issued early on the ACT HWDGE queue,
#: instead of the copies the compiler inserts for a returned input (scheduled late, they
#: held up the MoE combine DMAs before the world all-reduce: ~195 MB a core at
#: in8192_bs10).
CACHE_COPY_EARLY = True
#: The HWDGE queues the cache copies alternate over.
CACHE_COPY_ENGINES = (nisa.engine.scalar, nisa.engine.sync)
#: Rows (of 512 elements) per cache-copy DMA (one DMA runs on one engine at ~40 GB/s: 1 MB
#: pieces, so no 2-4 MB straggler drains alone into the MoE; 4096: in8192_bs10 1.73 -> 1.66 ms).
CACHE_COPY_ROWS = 1024
#: hc_out's x-independent prework (comb * residual) is scheduled after its all-reduce's
#: trigger (schedule-only deps): in v1 the list scheduler hoisted it ahead of the
#: trigger, where its DVE work delayed the wo_b / combine epilogues and so the trigger.
HCO_AFTER_AR = True
#: Elements per row of the cache copies' flattened views.
CACHE_COPY_ELEMS = 8192
#: Where the copies are issued: "early" (unwritten caches at the start, the window cache
#: after its scatter) or "ar" (unwritten caches at the attention all-reduce's trigger,
#: the window cache at the world all-reduce's: rank 0's DMA idles in those waits; early
#: copies collided with the slot-table / attention gathers at in8192_bs10).
CACHE_COPY_AT = "ar"
#: The "ar" copies are schedule-pinned after the barrier before their all-reduce.
CACHE_COPY_PIN = True
#: The cache copies spread over the phases where DMA is light: the index cache and the
#: first half of the compress cache at the start (hc_in), the rest and the window cache at
#: the attention all-reduce's trigger (instead of the window cache at the world
#: all-reduce's, where on the last-launched rank it outlasted the final hc_out).
CACHE_COPY_SPLIT = True
#: The final barriers on the copied caches are scheduled after the world all-reduce's trigger.
CACHE_BAR_AFTER_AR = True
#: The rope cos / sin rows gathered once at the start for the three ropes (kv, q, inverse).
ROPE_FREQS_ONCE = True
#: With one row tile, the MoE input rows are staged once for the shared and the routed experts.
MOE_STAGE_ONCE = True
#: Zero only the routed-expert accumulator rows the world all-reduce carries (not Tp + 128).
EACC_ZERO_T = True
#: Decode (one row tile): hc_in(attn)'s sinkhorn alongside the projection GEMMs instead of
#: in the attention all-reduce wait (a rank that arrives last at the all-reduce has no
#: wait to hide it in: it ran between the all-reduce and hc_out there).
COMB_A_EARLY = True
#: One row tile: hc_in(attn)'s sinkhorn runs on core 1 (no row tiles: idle while core 0
#: runs the row-tile norms / ropes) on the logits core 0 hands over through HBM; comb_a
#: comes back through HBM with a core barrier after the attention all-reduce's trigger.
COMB_A_CORE1 = True
#: ... for at most this many rows (with more rows core 0's own norm / rope work hides its
#: sinkhorn: in8192_bs10 gained nothing and paid the handover)
COMB_A_CORE1_MAX_T = 32
#: One row tile: the shared expert's w2 GEMM (x 1/world) right after the router GEMM, its
#: fp32 result kept in SBUF; after the routed experts only `eacc + o` (the w2 epilogue's
#: add, same operands) remains, over the rows the world all-reduce carries.
SHARED_W2_EARLY = True
#: With SHARED_W2_EARLY: the routed accumulator's epilogue and the cross-core combine
#: cover only the T real rows; the world all-reduce's padding rows T .. T2 of `cc2_src`
#: (summed and discarded) are zeroed once at the start instead of carrying computed zeros.
CC2_PAD_ZERO = True
#: Decode: hc_in(ffn)'s sinkhorn and the final hc_out's x-independent accumulators issued
#: in the MoE (after the routing weights' barrier; the routed experts are weight-DMA
#: bound) instead of after the world all-reduce's trigger.
COMB_F_MOE = False


def _half(n, pid):
    """`(lo, hi)` of core `pid`'s share of `n` columns, in whole 128-column tiles."""
    tiles = n // PT
    a = (tiles + 1) // 2
    lo = 0 if pid == 0 else a * PT
    hi = a * PT if pid == 0 else n
    return lo, hi


def _copy_rows(src, dst, pid, after=None, part=(0, 1)):
    """`dst = src` (HBM -> HBM, `[rows, n]`), core `pid` copying its half in DMAs of
    `CACHE_COPY_ROWS` x 512 elements on the hardware DGE queues `CACHE_COPY_ENGINES`.
    The tensors are viewed as `[rows / g, n * g]` (g rows per view row, up to 16 KiB
    contiguous): far fewer, larger DMA descriptors than one per 1 KiB cache row."""
    rows, n = src.shape
    g = 1
    for cand in (64, 32, 16, 8, 4, 2):
        if rows % cand == 0 and n * cand <= CACHE_COPY_ELEMS:
            g = cand
            break
    src = src.reshape((rows // g, n * g))
    dst = dst.reshape((rows // g, n * g))
    rows = rows // g
    step = max(1, CACHE_COPY_ROWS * 512 // (n * g))
    # `part = (k, m)`: only the k-th of m equal row ranges (each still split across the cores)
    k, m = part
    p0, p1 = rows * k // m, rows * (k + 1) // m
    h = p0 + (p1 - p0 + 1) // 2
    lo = p0 if pid == 0 else h
    hi = h if pid == 0 else p1
    for j, r0 in enumerate(range(lo, hi, step)):
        r1 = min(r0 + step, hi)
        ins = nisa.dma_copy(dst=dst[r0:r1, :], src=src[r0:r1, :], dge_mode=nisa.dge_mode.hwdge,
                            engine=CACHE_COPY_ENGINES[j % len(CACHE_COPY_ENGINES)])
        # schedule-only: without it the list scheduler hoists the copies (no data
        # dependency) into the DMA-bound slot-table / attention / combine phases
        _sched_after(ins, after)


@nki.jit
def layer_decode_r2_src_idx_e_tp64(
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
    assert (ratio, is_kv_src, is_idx_src, has_engram, tap) == (2, 1, 1, 1, 0), (
        f"flags {flags} are not an r2_src_idx_e block's")
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
    L = T // nspans
    half = cdiv(L, CORES)
    ng = grp_wslots.shape[0] // nspans   # compression groups a span
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
    e_wkv4 = e_wkv.view(X.FP8_X4)   # a 1x1 dummy otherwise
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
    topk_out = nl.ndarray((T, topk), dtype=nl.int32, buffer=nl.shared_hbm)
    tap_out = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    # a ratio-2 kv source writes the pooling inputs: returned as copies like the caches
    pool_out = CACHE_COPY_EARLY
    if pool_out:
        pkv_out = nl.ndarray(pool_kv.shape, dtype=pool_kv.dtype, buffer=nl.shared_hbm)
        psc_out = nl.ndarray(pool_score.shape, dtype=pool_score.dtype, buffer=nl.shared_hbm)
    else:
        pkv_out, psc_out = pool_kv, pool_score

    if CACHE_COPY_EARLY:
        wkv_out = nl.ndarray(window_kv_cache.shape, dtype=window_kv_cache.dtype, buffer=nl.shared_hbm)
        ckv_out = nl.ndarray(compress_kv_cache.shape, dtype=compress_kv_cache.dtype, buffer=nl.shared_hbm)
        ik_out = nl.ndarray(index_k_cache.shape, dtype=index_k_cache.dtype, buffer=nl.shared_hbm)
    else:
        wkv_out, ckv_out, ik_out = window_kv_cache, compress_kv_cache, index_k_cache

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
    # (rows padded to a multiple of 3 * 128 so the pipelined loops' views of R rows a
    # step-row, R = steps x U tokens with 3 or 4 steps, divide them; the rows past
    # Tp + PT are never read)
    Tq = pad_to(Tp + PT, 3 * PT)
    To = pad_to(Tp, 3 * PT)
    q = nl.ndarray((Tq, heads * hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="q")
    kvr = nl.ndarray((Tp, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kvr")
    kv = nl.ndarray((Tp, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kv")
    # the slot table, then (batched gathers) 128 zero rows the empty slots gather from
    nzero = PT if batched else 0
    kvtab = nl.ndarray((ntab_p + nzero, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kvtab")
    idx = nl.ndarray((Tq, kpad), dtype=nl.int32, buffer=nl.shared_hbm, name="idx")
    # idx permuted for the attention loop's batched value gathers (`attn_index_table`)
    idxp = nl.ndarray((Tq, kpad), dtype=nl.int32, buffer=nl.shared_hbm, name="idxp")
    o = nl.ndarray((To, heads * hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="o")
    oq = nl.ndarray((Tp, heads * hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="oq")
    z = nl.ndarray((Tp, o_lora_total), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="z")
    # the all-reduce buffers hold the T real rows only (not the 128-row padding)
    cc_src = nl.ndarray((T, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((T, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_dst")
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
    # (the world all-reduce's per-element summation order depends on its message size:
    # T rows only where that was measured exact)
    T2 = T if T <= CC2_T_ROWS_MAX else Tp
    cc2_src = nl.ndarray((T2, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc2_src")
    cc2_dst = nl.ndarray((T2, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc2_dst")
    # Source-layer scratch (`layer_kernel`'s).
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

    # (a kv source's compressed / index caches are copied once its compressor wrote them)
    split_copy = False
    if split_copy:
        _copy_rows(index_k_cache, ik_out, pid)
        _copy_rows(compress_kv_cache, ckv_out, pid, part=(0, 2))

    # ------------------------------------------------------------------ 0. engram read-out
    stream0 = nl.ndarray((T, width), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="stream0")
    engram_readout(pid, mine, T, Tp, hc, dim, hash_ids, etable, erow_start, e_wkv4, e_wkv_s, e_qk,
                   strm, stream0, rg_world)
    strm = stream0


    # ------------------------------------------------------------------ 1. hc_in (attention)
    nmine = len(mine)
    # (a core without row tiles -- core 1 when T <= 128 -- loads none of hc_in's constants)
    fnT = load_fn_t(hca_fn_t, width, mix) if nmine else None
    hcc = hc_in_consts(hca_base, hca_scale, attn_norm_w, hc, dim) if nmine else None
    # The rows' rope cos / sin, gathered once (and early: as each rope's own indirect gather
    # it sat in GpSimd's DMA queue behind the slot-table gathers and held up the q rope)
    fcache = {r0: load_freqs(freqs, row_pos, r0, nr) for r0, nr in mine} if ROPE_FREQS_ONCE else None

    def t2_buf():
        # the call's pre-sinkhorn comb logits, one slot per row tile (zeros past a
        # partial tile's rows)
        if not HC_SINKHORN_BATCHED or nmine == 0:
            return None
        t = nl.ndarray((PT, nmine, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(t, 0.0)
        return t

    def comb_out(t2all, comb_hbm, after=None):
        if t2all is None:
            return
        cb = sinkhorn_batched(t2all, nmine, hc, after=after)
        for t, (r0, nr) in enumerate(mine):
            nisa.dma_copy(dst=comb_hbm[r0 : r0 + nr, 0 : hc * hc], src=cb[0:nr, t, :])

    t2a = t2_buf()
    for t, (r0, nr) in enumerate(mine):
        hc_in_tile(r0, nr, strm, pre_mix, fnT, hca_base, hca_scale, attn_norm_w, hc, dim, mix,
                   xn, pre_a, post_a, comb_a, consts=hcc,
                   t2_dst=None if t2a is None else t2a[:, t, :])
    if not COMB_DEFERRED:
        comb_out(t2a, comb_a)
    core1_a = COMB_A_CORE1 and COMB_IN_AR and HC_SINKHORN_BATCHED and Tp == PT and T <= COMB_A_CORE1_MAX_T
    if core1_a:
        t2a_hbm = nl.ndarray((T, hc * hc), dtype=nl.float32, buffer=nl.shared_hbm, name="t2a_hbm")
        for t, (r0, nr) in enumerate(mine):
            nisa.dma_copy(dst=t2a_hbm[r0 : r0 + nr, :], src=t2a[0:nr, t, :])
    if pid == 0 and T < Tp:
        pad = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(pad, 0.0)
        nisa.dma_copy(dst=xn[T:Tp, :], src=pad[0 : Tp - T, :])
    # The routed experts' fp32 accumulator (private to this core) is zeroed here, off
    # the MoE's critical path: these writes overlap the DVE-bound hc_in / projections.
    # (only the rows the world all-reduce carries: rows past T are never routed, the junk
    # rows past Tp only take the padding slots' scatter-adds, and neither is read back)
    w2_early = SHARED_W2_EARLY and SHARED_FIRST and Tp == PT
    pad_zero = CC2_PAD_ZERO and w2_early and T2 > T
    T2r = T if pad_zero else T2
    mk_moe.zero_rows(eacc, T2r if EACC_ZERO_T else Tp + PT, dim)
    if pad_zero and pid == 1:
        zc = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(zc, 0.0)
        nisa.dma_copy(dst=cc2_src[T:T2, :], src=zc[0 : T2 - T, :])
    nisa.core_barrier(data=xn, cores=(0, 1))
    if core1_a:
        nisa.core_barrier(data=t2a_hbm, cores=(0, 1))

    # ------------------------------------------------------------------ 2. q / kv projections
    # MX GEMMs: the activation is quantized on chip as it is staged (each core stages
    # every row and computes its half of the output columns).
    # wq_a and wkv share one staging of the xn rows.
    X.gemm_rows_multi(xn, Tp, dim, [(wq_a4, wq_a_s, qa) + _half(q_lora, pid),
                                    (wkv4, wkv_s, kvr) + _half(hd, pid)])
    comb_in_ar = COMB_IN_AR and Tp == PT
    if comb_in_ar and COMB_A_EARLY and (not core1_a):
        comb_out(t2a, comb_a)
    if COMB_DEFERRED and (not comb_in_ar):
        # hc_in(attn)'s sinkhorn alongside the projection GEMMs (comb_a is first read by
        # hc_out(attn), after the attention)
        comb_out(t2a, comb_a)
    nisa.core_barrier(data=qa, cores=(0, 1))
    nisa.core_barrier(data=kvr, cores=(0, 1))
    if core1_a and pid == 1:
        # core 1 (no row tiles): hc_in(attn)'s sinkhorn while core 0 runs the norms
        t2c = nl.ndarray((PT, 1, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(t2c, 0.0)
        nisa.dma_copy(dst=t2c[0:T, 0, :], src=t2a_hbm[0:T, :])
        cbc = sinkhorn_batched(t2c, 1, hc)
        nisa.dma_copy(dst=comb_a[0:T, 0 : hc * hc], src=cbc[0:T, 0, :])
    # q_norm (wq_b quantizes its input itself); kv_norm -> rope -> fp8 round trip.
    norm_rows(qa, q_norm_w, qr, mine, q_lora, qdq=False, chunk=q_lora)
    norm_rows(kvr, kv_norm_w, kv, mine, hd, qdq=False, chunk=hd)
    rope_rows(kv, kv, mine, hd, hd, 1, freqs, row_pos, inverse=False, qdq=True, freqs_cache=fcache)
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
    rope_rows(q, q, mine, heads * hd, hd, heads, freqs, row_pos, inverse=False, freqs_cache=fcache)
    nisa.core_barrier(data=q, cores=(0, 1))

    # ------------------------------------------------------------------ 2b. compressor (kv sources)
    # (`layer_kernel`'s: the pooling inputs and the latent of this step's groups, the
    # compressed latents and the index keys into their caches)
    gtiles = tiles_of(NG, pid)
    fp32_proj_rows(xn, cm_wkv_t, pkv, mine, dim, hd)
    fp32_proj_rows(xn, cm_wgate_t, psc, mine, dim, hd)
    scatter_rows_f32(pkv, write_slots, pool_kv, mine, hd)
    scatter_rows_f32(psc, write_slots, pool_score, mine, hd)
    nisa.core_barrier(data=pool_kv, cores=(0, 1))
    nisa.core_barrier(data=pool_score, cores=(0, 1))
    pool_groups(pool_kv, pool_score, grp_mslots, gtiles, ratio, hd, lat_raw)
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

    # ------------------------------------------------------------------ 3. slot table + indices
    # Span b's rows of the slot table: its window rows, then its compressed rows.
    # (in the pair-interleaved column order the attention transposes from, `kv_pair_layout`)
    kvp = kv_pair_layout(hd)
    assert not (resident and kvp), "the resident (TI) path reads the natural column order"
    for b in range(nspans):
        if SLOT_GATHER_GROUPED:
            gather_rows_cores(window_kv_cache, win_slots, b * nwin, kvtab, b * nspan_rows, nwin, pid, hd,
                              pair=kvp)
            if read_len:
                gather_rows_cores(compress_kv_cache, cmp_slots, b * read_len, kvtab, b * nspan_rows + nwin,
                                  read_len, pid, hd, pair=kvp)
            continue
        gather_rows(window_kv_cache, win_slots[b * nwin : (b + 1) * nwin],
                    kvtab[b * nspan_rows : b * nspan_rows + nwin], tiles_of(nwin, pid), hd, pair=kvp)
        if read_len:
            gather_rows(compress_kv_cache, cmp_slots[b * read_len : (b + 1) * read_len],
                        kvtab[b * nspan_rows + nwin : (b + 1) * nspan_rows], tiles_of(read_len, pid), hd,
                        pair=kvp)
    # (`layer_kernel`'s index source: this step's top-k from the index keys)
    topk_src = topk_in
    for b in range(nspans):
        gather_rows(index_k_cache, cmp_slots[b * read_len : (b + 1) * read_len],
                    ktab[b * read_len : (b + 1) * read_len], tiles_of(read_len, pid), kd)
    nisa.core_barrier(data=ktab, cores=(0, 1))
    for b in range(nspans):
        itok0 = b * L + (0 if pid == 0 else L - half)
        index_scores_span(qi, wi, ktab[b * read_len : (b + 1) * read_len], sc_src, itok0, half,
                          nh_idx, read_len, kd)
    nisa.core_barrier(data=sc_src, cores=(0, 1))
    ncc.all_reduce(srcs=[sc_src], dsts=[sc_dst], replica_group=rg_attn, op=nl.add)
    index_topk(sc_dst, clen, sc16, topk_out, T, Tp, read_len, topk, pid)
    topk_src = topk_out
    attn_index_table(idx, mine, twin, wv, win, cbase, topk_src, topk, kpad,
                     idxp_hbm=idxp if batched else None, zrow0=ntab_p if batched else None)
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
    if pid == 1 and ntab < ntab_p + nzero:
        padk = nl.ndarray((PT, hd), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(padk, 0.0)
        for z0 in range(ntab, ntab_p + nzero, PT):
            z1 = min(z0 + PT, ntab_p + nzero)
            nisa.dma_copy(dst=kvtab[z0:z1, :], src=padk[0 : z1 - z0, :])
    nisa.core_barrier(data=kvtab, cores=(0, 1))
    nisa.core_barrier(data=idx, cores=(0, 1))
    if batched:
        nisa.core_barrier(data=idxp, cores=(0, 1))

    # ------------------------------------------------------------------ 4. sparse attention
    sink_t = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sink_t, src=attn_sink[0:heads, 0:1])
    # Tokens per loop step: TOK_UNROLL, at most 128 stacked (token, head) rows for
    # `attn_step`, a power of two (it divides Tp).
    pipe = ATTN_PIPE and batched and not resident
    U = TOK_UNROLL if resident else max(1, min(ATTN_PIPE_U if pipe else TOK_UNROLL, PT // heads))
    while Tp % U:
        U //= 2
    # Core 0 takes rows [0, h), core 1 rows [T - h, T): equal trip counts, and the rows
    # both cover (odd T, or h rounded up to whole steps) are computed by both (identically).
    ntok = cdiv(T, CORES)
    if ATTN_WHOLE_STEPS and (not resident):
        # decode: a core's few tokens in whole steps (U the smallest power of two that
        # covers them), never the one-token-at-a-time remainder loop
        while U > 1 and U // 2 >= ntok:
            U //= 2
        ntok = cdiv(ntok, U) * U
    tok0 = 0 if pid == 0 else max(0, T - ntok)
    # the sinks repeated for each token of a step (`attn_step`'s stacked rows)
    sink_u = nl.ndarray((U * heads, 1), dtype=nl.float32, buffer=nl.sbuf)
    for u in range(U):
        nisa.dma_copy(dst=sink_u[u * heads : (u + 1) * heads, :], src=attn_sink[0:heads, 0:1])
    masks = None if resident else attn_step_masks(U, heads, hd, kpad)
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
        while nsp > 2 and (nsp * U > PT or ntok < nsp * U):
            nsp //= 2
    npipe = ntok // (nsp * U) if pipe else 0

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
        q2 = q.reshape((Tq // R, R * heads * hd))
        idx2 = idx.reshape((Tq // R, R * kpad))
        o2 = o.reshape((To // R, R * heads * hd))
        idxp2 = idxp.reshape((Tq // R, R * kpad))
        bufA = attn_step_bufs(U, heads, hd, kpad)
        bufB = attn_step_bufs(U, heads, hd, kpad)
        # prologue: the first step's loads
        attn_fetch(None, tok0, U, q, kvtab, idx, heads, hd, kpad, bufA, idxp_hbm=idxp)

        bufs = (bufA, bufB)

        def pipe_body(k):
            # step s computes from bufs[s % 2] (step 0 fetched by the previous iteration);
            # step s + 1 is fetched into the other buffer first, once step s - 1's PV has
            # read it (step nsp = the next iteration's step 0)
            prev = None
            for st in range(nsp):
                attn_fetch(k, tok0 + (st + 1) * U, U, q2, kvtab, idx2, heads, hd, kpad,
                           bufs[(st + 1) % 2], idxp_hbm=idxp2)
                prev = attn_compute(k, tok0 + st * U, U, o2, bufs[st % 2], sink_u, masks, heads, hd,
                                    kpad, scale, after=prev)

        nl.fori_loop(0, npipe, pipe_body)
    if nstep > nsp * npipe:
        nl.fori_loop(nsp * npipe, nstep, step_body)
    if ntok > nstep * U:
        nl.fori_loop(nstep * U, ntok, tok_body)
    nisa.core_barrier(data=o, cores=(0, 1))

    # ------------------------------------------------------------------ 5. output projections
    # Inverse rope, grouped wo_a, wo_b (fp32 partial); the MX GEMMs quantize their inputs.
    rope_rows(o, oq, mine, heads * hd, hd, heads, freqs, row_pos, inverse=True, qdq=False,
              freqs_cache=fcache)
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
    X.gemm_rows(z, Tp, o_lora_total, wo_b4, wo_b_s, cc_src, lo, hi, out_dtype=nl.float32, out_rows=T)
    bar1 = nisa.core_barrier(data=cc_src, cores=(0, 1))
    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], replica_group=rg_attn, op=nl.add)
    # (the all-reduce op itself takes no schedule deps: anchor on its source's barrier)
    ar1 = bar1 if HCO_AFTER_AR and hasattr(bar1, "depends_on") else None
    if CACHE_COPY_EARLY and CACHE_COPY_AT == 'ar':
        pin1 = bar1 if CACHE_COPY_PIN and hasattr(bar1, "depends_on") else None
        if split_copy:
            _copy_rows(compress_kv_cache, ckv_out, pid, after=pin1, part=(1, 2))
            _copy_rows(window_kv_cache, wkv_out, pid, after=pin1)
        else:
            _copy_rows(compress_kv_cache, ckv_out, pid, after=pin1)
            _copy_rows(index_k_cache, ik_out, pid, after=pin1)
    if comb_in_ar and (not COMB_A_EARLY) and (not core1_a):
        comb_out(t2a, comb_a)
    if core1_a:
        # core 1's comb_a, before hc_out(attn) reads it
        nisa.core_barrier(data=comb_a, cores=(0, 1))

    # ------------------------------------------------------------------ 6. hc_out (attention)
    # hc_out reads the fp32 all-reduce result into bf16 by DMA: the bf16 cast of the
    # attention output, without a bf16 copy through HBM.
    # ------------------------------------------------------------------ 7. hc_in (ffn)
    # fused with 6 by row tile: hc_in reads the tile hc_out just produced from SBUF (the
    # stream2 rows are still stored for hc_out (ffn)'s residual).
    fnT2 = load_fn_t(hcf_fn_t, width, mix) if nmine else None
    hcc2 = hc_in_consts(hcf_base, hcf_scale, ffn_norm_w, hc, dim) if nmine else None
    # skewed: tile t+1's hc_out is issued before tile t's hc_in
    yss = [None] * len(mine)
    t2f = t2_buf()
    # decode (one tile of few rows): hc_out in the spread layout, hc_in reads its rows back
    spread = HC_OUT_SPREAD and nmine == 1 and Tp == PT and spread_of(mine[0][1], dim) > 1
    if spread:
        r0, nr = mine[0]
        st = hc_out_spread_acc(nr, strm, post_a, comb_a, hc, dim, after=ar1)
        y_last = hc_out_spread_out(st, nr, cc_dst, stream2, hc, dim)
        hc_in_tile(r0, nr, stream2, pre_a, fnT2, hcf_base, hcf_scale, ffn_norm_w, hc, dim, mix,
                   xn2, ffn_pre, post_f, comb_f, consts=hcc2,
                   t2_dst=None if t2f is None else t2f[:, 0, :], x_after=y_last)
    for t in range(0 if spread else len(mine) + 1):
        if t < len(mine):
            r0, nr = mine[t]
            yss[t] = hc_out_tile(r0, nr, cc_dst, strm, post_a, comb_a, stream2, hc, dim, keep=True,
                                 acc_first=(t == 0), after=ar1)
        if t > 0:
            r0, nr = mine[t - 1]
            hc_in_tile(r0, nr, stream2, pre_a, fnT2, hcf_base, hcf_scale, ffn_norm_w, hc, dim, mix,
                       xn2, ffn_pre, post_f, comb_f, consts=hcc2, xin=yss[t - 1],
                       t2_dst=None if t2f is None else t2f[:, t - 1, :])
    if not comb_in_ar:
        comb_out(t2f, comb_f)
    if pid == 0:
        pad = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(pad, 0.0)
        if T < Tp:
            nisa.dma_copy(dst=xn2[T:Tp, :], src=pad[0 : Tp - T, :])
        nisa.dma_copy(dst=xn2[Tp : Tp + PT, :], src=pad)
    nisa.core_barrier(data=xn2, cores=(0, 1))

    # ------------------------------------------------------------------ 8. MoE
    ptiles = tiles_of(Tp, 0) + tiles_of(Tp, 1)

    # One row tile: xn2 staged (transposed + quantized) once for the shared expert's
    # w1 / w3 and every routed expert (`mk_moe.EXPERT_MERGED`; the same per-row values).
    xstage = None
    if MOE_STAGE_ONCE and mk_moe.EXPERT_MERGED and (Tp == PT):
        xstage = X.alloc_x(1, dim)
        X.stage_x(xn2, 0, dim, xstage[0], xstage[1], 0)

    def shared_w13():
        # Shared expert: this core's half of the intermediate columns (SwiGLU is
        # elementwise along it and w2 contracts over it).
        lo, hi = _half(inter, pid)
        if xstage is not None:
            for w_mx, s_mx, out_hbm in ((sh_w1_4, sh_s1, eg), (sh_w3_4, sh_s3, eu)):
                X.gemm(xstage[0], xstage[1], 1, dim, w_mx, s_mx, out_hbm, 0, lo, hi)
        else:
            X.gemm_rows_multi(xn2, Tp, dim, [(sh_w1_4, sh_s1, eg, lo, hi), (sh_w3_4, sh_s3, eu, lo, hi)])
        swiglu_rows(eg[0:Tp, lo:hi], eu[0:Tp, lo:hi], None, 0, ey[0:Tp, lo:hi], ptiles, hi - lo,
                    SWIGLU_LIMIT)

    if SHARED_FIRST:
        # the shared expert's w1 / w3 (weight-DMA-bound, independent of the routing) first:
        # the router's compute overlaps their weight streams
        shared_w13()
    # Router (fp32 GEMM + top-k) split by row tile between the cores, then every core
    # reads all rows' expert weights for its experts.
    if ROUTER_COL_SPLIT and T <= PT and (n_experts % 2 == 0):
        # one row tile: each core computes half of the experts' logits for every row
        # (each logit's fp32 accumulation unchanged) and streams half of gate_w_t
        nh = n_experts // 2
        gemm_fp32(xn2, gate_w_t, logits, tiles_of(T, 0) + tiles_of(T, 1), dim, nh, n0=pid * nh)
        nisa.core_barrier(data=logits, cores=(0, 1))
    else:
        gemm_fp32(xn2, gate_w_t, logits, mine, dim, n_experts)
    if w2_early:
        # the shared expert's w2 partial (this core's contraction half, x 1/world) now,
        # kept in SBUF for the epilogue after the routed experts
        osb = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
        lo2, hi2 = _half(inter, pid)

        def w2_keep(r, n0, w, o):
            nisa.tensor_copy(dst=osb[:, n0 : n0 + w], src=o, engine=nisa.engine.gpsimd)

        X.gemm_rows(ey, Tp, hi2 - lo2, sh_w2_4, sh_s2, eo32, 0, dim, x_col0=lo2, krow0=lo2 // 4,
                    out_dtype=nl.float32, out_scale=1.0 / world, epilogue=w2_keep)
    route_topk(logits, gate_b, expert_lo, mine, n_experts, TOPK_EXPERTS, n_local, ROUTE_SCALE, wexp)
    if pid == 0:
        zw = nl.ndarray((PT, n_local), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(zw, 0.0)
        if T < Tp:
            nisa.dma_copy(dst=wexp[T:Tp, :], src=zw[0 : Tp - T, :])
        nisa.dma_copy(dst=wexp[Tp : Tp + PT, :], src=zw)
    barw = nisa.core_barrier(data=wexp, cores=(0, 1))
    hco_st = None
    if spread and comb_in_ar and COMB_F_MOE:
        pinw = barw if hasattr(barw, "depends_on") else None
        comb_out(t2f, comb_f, after=pinw)
        hco_st = hc_out_spread_acc(mine[0][1], stream2, post_f, comb_f, hc, dim, after=pinw)
    # Routed experts, each over the rows that picked it (this core's experts), into
    # the zeroed fp32 accumulator.
    lo, hi = _half(inter, pid)
    mk_moe.routed_experts(pid, n_local, Tp, xn2, wexp, w1_4, s1, w3_4, s3, w2_4, s2,
                          dim, inter, lo, hi, SWIGLU_LIMIT, eacc, zero_acc=False, T=T, xstage=xstage)
    if not SHARED_FIRST:
        shared_w13()
    # The shared expert's weights are replicated on every rank and the model adds its
    # output ONCE, after the expert all-reduce. Every rank adds it to its partial here,
    # scaled by 1/world, so the world sum carries it exactly once (a power-of-two
    # scale of the fp32 partial is exact).
    # This core's partial, eacc + eo (fp32), is formed in the w2 GEMM's epilogue (eo never
    # goes through HBM): the other core's rows go to this core's `part` slot, its own rows
    # to `eo32`; after the barrier each core rejoins its own rows, part[0] + part[1], then
    # the ranks'. (Do NOT split the all-reduces by rows to overlap them: the collectives'
    # per-element summation order depends on the message size -- not bit-exact.)
    lo, hi = _half(inter, pid)
    other = {r for r, _ in tiles_of(Tp, 1 - pid)}

    def w2_epi(r, n0, w, o):
        av = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=av, src=eacc[r : r + PT, n0 : n0 + w])
        nisa.tensor_tensor(dst=av, data1=av, data2=o, op=nl.add, engine=nisa.engine.gpsimd)
        dst = part[pid, r : r + PT, n0 : n0 + w] if r in other else eo32[r : r + PT, n0 : n0 + w]
        nisa.dma_copy(dst=dst, src=av)

    if w2_early:
        # w2_epi over the rows the world all-reduce carries, o from SBUF; the combine split
        # by column tiles between the cores (each keeps its own tiles' partial in SBUF and
        # hands the other core its tiles through `part`)
        nrw = min(PT, T2r)
        ntl = cdiv(dim, X.NT)
        own_lo, own_hi = (0, cdiv(ntl, 2)) if pid == 0 else (cdiv(ntl, 2), ntl)
        kept = []
        for jt in range(ntl):
            n0 = jt * X.NT
            w = min(X.NT, dim - n0)
            av = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=av[0:nrw, :], src=eacc[0:nrw, n0 : n0 + w])
            nisa.tensor_tensor(dst=av[0:nrw, :], data1=av[0:nrw, :], data2=osb[0:nrw, n0 : n0 + w],
                               op=nl.add, engine=nisa.engine.gpsimd)
            if own_lo <= jt < own_hi:
                kept.append((n0, w, av))
            else:
                nisa.dma_copy(dst=part[pid, 0:nrw, n0 : n0 + w], src=av[0:nrw, :])
        nisa.core_barrier(data=part, cores=(0, 1))
        for n0, w, av in kept:
            bv = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=bv[0:nrw, :], src=part[1 - pid, 0:nrw, n0 : n0 + w])
            p0, p1 = (av, bv) if pid == 0 else (bv, av)
            nisa.tensor_tensor(dst=av[0:nrw, :], data1=p0[0:nrw, :], data2=p1[0:nrw, :], op=nl.add)
            nisa.dma_copy(dst=cc2_src[0:nrw, n0 : n0 + w], src=av[0:nrw, :])
    else:
        X.gemm_rows(ey, Tp, hi - lo, sh_w2_4, sh_s2, eo32, 0, dim, x_col0=lo, krow0=lo // 4,
                    out_dtype=nl.float32, out_scale=1.0 / world, epilogue=w2_epi)
        ach = min(1024, dim)
        nisa.core_barrier(data=part, cores=(0, 1))
        for r0, nr in tiles_of(Tp, pid):
            nr = min(nr, T2 - r0)
            if nr <= 0:
                continue
            for c in range(dim // ach):
                c0, c1 = c * ach, (c + 1) * ach
                av = nl.ndarray((PT, ach), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=av[0:nr, :], src=eo32[r0 : r0 + nr, c0:c1])
                bv = nl.ndarray((PT, ach), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=bv[0:nr, :], src=part[1 - pid, r0 : r0 + nr, c0:c1])
                p0, p1 = (av, bv) if pid == 0 else (bv, av)
                nisa.tensor_tensor(dst=av[0:nr, :], data1=p0[0:nr, :], data2=p1[0:nr, :], op=nl.add)
                nisa.dma_copy(dst=cc2_src[r0 : r0 + nr, c0:c1], src=av[0:nr, :])
    bar2 = nisa.core_barrier(data=cc2_src, cores=(0, 1))
    ncc.all_reduce(srcs=[cc2_src], dsts=[cc2_dst], replica_group=rg_world, op=nl.add)
    ar2 = bar2 if HCO_AFTER_AR and hasattr(bar2, "depends_on") else None
    if CACHE_COPY_EARLY and CACHE_COPY_AT == 'ar':
        pin2 = bar2 if CACHE_COPY_PIN and hasattr(bar2, "depends_on") else None
        if not split_copy:
            _copy_rows(window_kv_cache, wkv_out, pid, after=pin2)
        if pool_out:
            _copy_rows(pool_kv, pkv_out, pid, after=pin2)
            _copy_rows(pool_score, psc_out, pid, after=pin2)
    if comb_in_ar and hco_st is None:
        # (pinned after the trigger: hoisted, its DVE / ACT chain delayed the combine adds)
        comb_out(t2f, comb_f, after=ar2)

    # ------------------------------------------------------------------ 9. hc_out (ffn)
    # (the bf16 cast of the ffn output in hc_out's DMA, as in 6)
    # one tile of lookahead: tile t+1's comb*res accumulators are issued before tile t's
    # x-dependent half, so two tiles' accumulate while the all-reduce is in flight
    def hco_acc(t):
        r0, nr = mine[t]
        return hc_out_tile(r0, nr, cc2_dst, stream2, post_f, comb_f, out2, hc, dim, acc_only=True,
                           after=ar2)

    if spread:
        if hco_st is None:
            hco_st = hc_out_spread_acc(mine[0][1], stream2, post_f, comb_f, hc, dim, after=ar2)
        hc_out_spread_out(hco_st, mine[0][1], cc2_dst, out2, hc, dim)
    pend = hco_acc(0) if nmine and not spread else None
    for t, (r0, nr) in enumerate([] if spread else mine):
        nxt = hco_acc(t + 1) if t + 1 < nmine else None
        hc_out_tile(r0, nr, cc2_dst, stream2, post_f, comb_f, out2, hc, dim, accs_in=pend)
        pend = nxt
    nisa.core_barrier(data=out_stream, cores=(0, 1))
    nisa.core_barrier(data=ffn_pre, cores=(0, 1))
    if CACHE_COPY_EARLY:
        # (schedule-pinned after the world all-reduce's trigger: hoisted ahead of it, a barrier
        # waiting on a copy still in flight held up GpSimd's queue -- the w2 epilogue adds)
        pinb = bar2 if CACHE_BAR_AFTER_AR and hasattr(bar2, "depends_on") else None
        _sched_after(nisa.core_barrier(data=wkv_out, cores=(0, 1)), pinb)
        _sched_after(nisa.core_barrier(data=ckv_out, cores=(0, 1)), pinb)
        _sched_after(nisa.core_barrier(data=ik_out, cores=(0, 1)), pinb)
        if pool_out:
            _sched_after(nisa.core_barrier(data=pkv_out, cores=(0, 1)), pinb)
            _sched_after(nisa.core_barrier(data=psc_out, cores=(0, 1)), pinb)
    return out_stream, ffn_pre, topk_out, tap_out, wkv_out, ckv_out, pkv_out, psc_out, ik_out



#: The kernel on both physical cores of the logical NeuronCore (LNC=2).
kernel = layer_decode_r2_src_idx_e_tp64[2]
