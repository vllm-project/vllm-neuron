"""DeepSeek-V4.1-Flash `r1_idx` block, decode, tp64 scheme -- one NKI megakernel.

The model launches it as `layer_decode_r1_idx_tp64` for blocks 24, 28, 32, 36: a ratio-1 index
source (index scores / top-k over the nearest kv source's keys). It is the optimized r1 decode
kernel (`../r1_decode`) with the sections of `../layer_kernel.layer_body` this archetype has
(generated from the template by resolving the archetype's branches), at the same points of the
dataflow, so its other phases are the optimized kernel's. The signature and the outputs are the
model's.

    hc_in -> q / kv projections -> window + top-k compressed sparse attention
          -> all_reduce(8-rank attention group) -> hc_out
          -> hc_in -> MoE (top-6 of 384: this rank's 6 experts + 1/64 of the shared one)
          -> all_reduce(64-rank world) -> hc_out

`flags` = (1, nspans, window, stage=9, 0, 1, 0, 0, kernel_version).
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

from .mk_compress import (
    fp32_proj_rows,
    IDX_KT_DIRECT,
    index_scores_span,
    index_topk,
    rope_fp4_put,
)
from . import mk_moe
from . import mk_mx as X
from .mk_common import (
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
    cmp_compose_index,
    cmp_gather_onchip,
    CMP_SLOTS_ON_CHIP_MAX,
    gather_rows,
    gather_rows_batched,
    gather_rows_via,
    gemm_fp32,
    hc_in_consts,
    hc_in_tile,
    HC_SINKHORN_BATCHED,
    sinkhorn_batched,
    hc_out_tile,
    hc_out_spread_s,
    hc_out_tile_spread,
    kt_resident,
    kv_pair_layout,
    load_fn_t,
    norm_rows,
    pad_to,
    rope_rows,
    route_topk,
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
#: Both hc_in sinkhorns (`comb_out`, ~45 us of serial DVE / ACT work each at T = 6) issued
#: right after the all-reduce whose result their comb is first read with (the engines
#: idle while it is in flight) instead of on the projection / router chains.
COMB_AT_AR = True
#: Rows of the all-reduce payloads: the real T rows (not the padded Tp) only for a payload
#: of at most CC_SMALL_BYTES. The collectives' per-element summation order changes with the
#: message size above a small-message size: T = 60 rows of fp32 5120 was not bit-exact on
#: either all-reduce (world: out_stream 0.0039; attention group: ffn_pre 6.6e-6 at
#: in1024_bs10_L21, exact at L33), T = 6 rows is exact on both.
CC_SMALL_BYTES = 128 * 1024
CC_ROWS = lambda T, Tp, dim: T if T * dim * 4 <= CC_SMALL_BYTES else Tp
#: Compact slot table: gather only each row's top-k compressed rows (composed indices,
#: `cmp_compose_index`) instead of every span's read_len, when that is fewer rows.
CMP_COMPACT = True
#: (fused launch) With at least two spans, the index keys / scores split between the cores by
#: whole spans (span b to core b % 2: its key-table gather and all its L tokens' scores, one
#: batched `index_scores_span` pass) instead of every span's rows / tokens halved between
#: them (each core then transposed every span's key table and scored half its tokens).
#: Every row's values are unchanged.
IDX_SPAN_SPLIT = True
#: The index-key table's gather (`ktab`) in batches of this many 128-row tiles a DMA
#: (`gather_rows_batched`: one index load, gather and store a batch) when a span's rows
#: are a multiple of the batch; 1: per-tile `gather_rows`. (The per-tile SWDGE gather
#: was ~1.3 us of GpSimd a tile: ~550 us a core at in8192_bs10.)
#: NOTE (iter-4): `gather_rows_batched` loads its [PT, G] index tile row-major (ix[p, g] =
#: slots[p G + g]) but a 2-D vector-offset gather reads the indices column-major
#: (`cmp_gather_onchip`'s comment): its rows come out permuted. It was never exercised (needs
#: read_len % 512 == 0; the recorded cases have 1280 / 8448) and is unreachable with
#: IDX_KT_DIRECT; kept off (1) until its index load is fixed.
IDX_GATHER_BATCH = 1
#: (`mk_compress.IDX_KT_DIRECT`) 128-row tiles a gather DMA of the index keys gathered
#: straight into SBUF.
IDX_DIRECT_BATCH = 1
#: The index-score all-reduce carries only the T real rows when T x read_len fp32 is at most
#: CC_SMALL_BYTES (in1024_bs1: 30 KB instead of the padded 655 KB).
IDX_SC_ROWS_T = True
#: Else (one span, T x read_len / 2 fp32 at most CC_SMALL_BYTES) two all-reduces of the two
#: cores' key halves (in8192_bs1: 2 x 101 KB instead of the padded 4.3 MB).
IDX_SC_SPLIT = True
#: (IDX_KT_DIRECT with the span split) core 0's share of the spans: k4 at in8192_bs10 (5 + 5)
#: had core 0's gathers + scores end ~260 us after core 1's (its DMA / GpSimd shared with
#: hc_in and the projections). (iter-5: with KT_IX_BULK the gathers cost ~30% less GpSimd
#: and 0.4 left core 0 idle ~190 us before the score all-reduce -> 0.5 again.)
IDX_SPAN_CORE0_FRAC = 0.5
#: (SHARE_INDEX_KEYS) The split when the keys are reloaded (no gather on either core).
IDX_SPAN_CORE0_FRAC_LOADED = 0.5
#: (iter-5) With the gathers (the source block), core 0's span count from a cost model instead
#: of IDX_SPAN_CORE0_FRAC: core 0 first runs hc_in (~IDX_CORE0_HEAD_US), then its n0 spans, core 1
#: the other nspans - n0; a span's gather + scores ~ read_len * IDX_SPAN_NS_PER_KEY (m2 profile,
#: in8192_bs10: ~130 us a span of 8448 keys, core 0's gathers starting ~90-100 us late).
#: Measured: in8192_bs10 wants 5 of 10 spans on core 0, in1024_bs10 fewer than 4.
IDX_SPAN_SPLIT_MODEL = True
IDX_CORE0_HEAD_US = 90.0
#: (HC_MIX_CORE1) core 0's head with only the normed rows of hc_in (spread) at the start
IDX_CORE0_HEAD_US_MIX1 = 35.0
IDX_SPAN_NS_PER_KEY = 15.4
#: The attention loop's tokens per step fitted to the core's token count (fewest loop
#: bodies) instead of the largest power of two <= the cap plus single-token leftovers.
ATTN_U_FIT = True
#: The compact table's compressed rows gathered several tiles per DMA (`gather_rows_via`).
#: Off: wrong rows at in8192_bs10 (60 batches of 4 tiles; deterministic max err 0.125),
#: exact at in8192_bs1 (6 batches). Superseded by `cmp_gather_onchip`'s batches
#: (CMP_GATHER_BATCH, exact everywhere: the index tile's column q = the batch's tile q).
GATHER_VIA_BATCHED = False
#: The compact table's slot lookups on chip (`cmp_compose_index(slots_hbm=...)`: each row's
#: span table loaded once, its top-k entries picked by `nc_n_gather`) instead of one
#: indirect 4-byte DMA per 128-row gather tile (GpSimd-issued, half the gather's DMAs).
CMP_SLOTS_ON_CHIP = True
#: The returned window / compressed / index-key caches are copies the kernel makes itself
#: (`_copy_rows`, each core half the rows; issued after the attention all-reduce in program
#: order, but the scheduler starts the read-only caches' copies early, they depend on
#: nothing) instead of returning the inputs (the compiler's copies of those, ~98 MB a core
#: at in8192_bs10, ran at the end, ahead of the world all-reduce, delaying the MoE merge).
CACHE_COPY_AT_AR = True
#: The MoE partials rejoined by both cores, each over half the columns of every row
#: (one 128-row tile at T <= 128 was rejoined by core 0 alone).
MERGE_COL_SPLIT = True
#: The engine executing the core barriers (it blocks only that engine): the sync engine
#: instead of GpSimd, whose in-order stream also issues every software-DGE gather (a
#: barrier queued behind a core's gathers held the other core for 140 us at in8192_bs10).
BARRIER_ENGINE = nisa.engine.sync
#: The compact table's compressed rows (inputs only: `topk_in`, `cbase`, `cmp_slots`, the
#: compressed cache) gathered at the start of the kernel, overlapping hc_in and the
#: projections, instead of after them.
CMP_GATHER_EARLY = False
#: With CMP_SLOTS_ON_CHIP: the compressed gathers' index columns made on chip (PE
#: transposes of the looked-up rows, `cmp_gather_onchip`) instead of stored to a private
#: HBM table and loaded back one 128-row tile at a time.
CMP_IDX_ON_CHIP = True
#: At T <= 128 (one row tile, core 0's alone) the router GEMM split by expert columns
#: between the cores (each loads half the fp32 gate weight).
ROUTER_COL_SPLIT = True
#: CACHE_COPY_AT_AR's copies made data-dependent on the attention all-reduce's source
#: (`cc_src`): without it the scheduler starts the read-only caches' copies at the
#: kernel's start, where they contend with hc_in / gathers / attention for DMA.
CACHE_COPY_DEP = False
#: The shared expert's w1 / w3 GEMMs and SwiGLU issued before the routed experts (whose
#: per-expert dynamic loops end the scheduler's window), so they overlap the router and
#: top-k instead of following the last routed expert; its w2 (whose epilogue reads the
#: routed accumulator) stays after them.
SHARED_FIRST = True
#: With SHARED_FIRST: the shared expert's w2 GEMM too, its fp32 partial parked in HBM
#: (rows < Tc2) and added to the routed accumulator after the routed experts. Only for
#: small all-reduce payloads (Tc2 <= Tp / 2, e.g. T = 6): at bs10 (Tc2 = Tp, 2.6 MB parked
#: and reloaded a core) the chain after the attention all-reduce got 15-44 us longer.
SHARED_W2_FIRST = True
#: SHARED_W2_FIRST also for big payloads when all rows are one row tile (Tp <= 128, e.g.
#: T = 60): the w2 GEMM's output tiles (fp32, 20 KB a partition) stay in SBUF across the
#: routed experts instead of being parked in HBM (for T = 6 too).
SHARED_W2_SBUF = True
#: The engine whose hardware DGE queue issues `_copy_rows`' DMAs (None: the default,
#: GpSimd's software DGE, whose rings the gathers and hc_in loads also use).
CACHE_COPY_ENGINE = None
#: The zero padding rows (T..Tp, and the extra tile of xn2 / wexp) of xn, qr, oq, xn2 and
#: wexp written once at the kernel's start (core 0) instead of before each one's barrier.
EARLY_PADS = False
EARLY_PADS_ENGINE = nisa.engine.scalar
#: (EARLY_PADS OFF: the pads' DMAs on the ACT queue at the start delayed hc_in / the
#: projections, pre-AR chain +70 us at T=6.) Instead, the xn2 / wexp pads (read after the
#: attention all-reduce) are issued right after it (default queue), off the xn2 barrier.
#: Only for small all-reduce payloads (Tc2 * 2 <= Tp, T = 6): at T = 60 the chain after the
#: attention all-reduce got ~43 us longer with the pads moved (start or here).
PADS_AT_AR1 = True
#: (fused launch) The GEMM inputs' zero pad rows (rows T..Tp of xn, qr, oq and xn2, and
#: xn2's extra tile) are not stored to HBM: the GEMMs read only the T real rows and stage
#: the rest as zero rows on chip (`gemm_rows(m_valid=T)`: the same values), and the routed
#: experts' padding slots (which gather xn2's extra tile) are skipped by their scatter-add.
#: The routed accumulator `eacc` is zeroed only on the rows its epilogue reads.
#: (Were ~5 MB of pad stores a block on core 0's queues, xn / qr / oq ahead of barriers.)
PADS_ON_CHIP = True
#: The engine of the shared w2 epilogue's fp32 adds (eacc + the shared output): DVE, off
#: GpSimd, whose in-order stream issues the epilogue's software-DGE DMAs (was GpSimd).
W2_EPI_ENGINE = nisa.engine.vector
#: (fused launch) With COMB_AT_AR at T <= 128 (one row tile, core 0's): hc_in(attn)'s
#: sinkhorn runs on core 1, idle then: core 0 stores its rows' pre-sinkhorn logits (T x 16
#: fp32) to HBM, core 1 loads them after the kvr barrier (while core 0 runs the q / kv
#: norm + rope chain), runs the same `sinkhorn_batched` and stores comb_a; core 0 reads it
#: after a barrier at the attention all-reduce. (On core 0 during the all-reduce its ~45 us
#: of serial DVE / ACT work outlasted the ~22 us all-reduce at T=6.)
COMB_A_CORE1 = True
#: (fused launch) At Tp = 128 (one row tile) the routed experts run densely over the tile
#: (`mk_moe.run_expert_dense`): the shared expert's staged xn2 tile is every routed expert's
#: stationary (no per-expert row gather / re-staging), the routing weights come from an SBUF
#: copy of wexp, and the w2 outputs accumulate in an fp32 SBUF tile (no scatter-add, no eacc
#: in HBM) that the shared w2 epilogue reads. An active expert cost ~80 us a core, ~27 us of
#: it before the first matmul (ids -> gather -> staging) and ~8 us of scatter-add at the end.
MOE_DENSE = True
#: (fused launch) At T <= 128 the q rope and the inverse rope of the attention output
#: split by heads between the cores (core p ropes columns [p, p+1) * heads * hd / 2, its own
#: half of wq_b's output; elementwise, same values) instead of core 0 alone.
ROPE_SPLIT = True
#: (fused launch) At T <= 128 the kv chain (kv_norm -> rope -> fp8 round trip, the window
#: cache scatter) runs on core 1, in parallel with core 0's q norm (both were core 0's).
KV_CORE1 = True
#: With CACHE_COPY_AT_AR: the read-only caches' copies (compressed, index-key) issued first
#: in program order (GpSimd issues them at the kernel's start, while hc_in is PE / DVE
#: bound) instead of where the scheduler put them (at in8192_bs10 half of the 8 MB chunks
#: were issued at ~580-617 us, starving wo_a's weight DMA for ~60 us); the window cache's
#: copy stays after this step's scatter. (OFF: no effect, the scheduler issued the copies at
#: the same times, 101-110 / 243-322 / 586-622 us at in8192_bs10, whatever their program order.)
CACHE_COPY_FIRST = False
#: `_copy_rows`' bytes per DMA.
CACHE_COPY_CHUNK = 8 << 20
#: With COMB_AT_AR: each sinkhorn made data-dependent on the last tile its all-reduce's
#: source is built from (one element DMA'd into a pad partition of the sinkhorn's input,
#: a row whose result is discarded), so it runs while the all-reduce is in flight. Without
#: it the scheduler hoists it (it reads only hc_in's logits) onto the DVE / ACT queues
#: ahead of the MoE merge / shared w2 epilogue (~50 us before the world all-reduce at T=6).
SINKHORN_GATED = True
#: (fused launch) At T <= 128 (one row tile, core 0's) the two hc_outs run on BOTH cores,
#: each core the columns [pid * dim / 2, (pid + 1) * dim / 2) of every stream
#: (`hc_out_tile_spread(col0=, ncols=)`: elementwise, the same values); hc_in(ffn) reads
#: the rows back from `stream2` after a barrier (as it already does for a spread tile).
#: Core 1 reads core 0's post / comb coefficients after barriers on them.
HCO_SPLIT = True
#: (fused launch, COMB_A_CORE1) At T <= 128 hc_in(attn) is split by its two independent
#: halves (`hc_in_tile(part=)`): core 0 makes only the normed rows (the projections'
#: input), core 1 the coefficients (mix GEMM, sum of squares, pre / post, the comb logits
#: straight into its sinkhorn) after its projection GEMMs, instead of core 0 running the
#: ~45-75 us mix GEMM on its PE ahead of its projection half. Same values.
#: (r1_idx: core 1's half right after the index-score all-reduce, `mix1`, where it waits
#: for the scores.)
HC_MIX_CORE1 = True
#: (fused launch, HCO_SPLIT) hc_in(ffn) split the same way: core 1 the coefficients (mix GEMM,
#: ffn_pre / post_f, the comb_f logits and their sinkhorn at the world all-reduce) right
#: after its hc_out half, core 0 the normed rows, then the WHOLE router GEMM and top-k (no
#: column split, so core 0 never waits for core 1's half), the xn2 barrier moved next to the
#: wexp one (one rendezvous). Same values (the router columns' accumulation unchanged).
HC_MIXF_CORE1 = True
#: DMA QoS priority of `_copy_rows`' DMAs (0-3, lower is higher; None: the default). The
#: copies (~98 MB a core at in8192_bs10) share the DMA rings with the gathers and the
#: projection / wo weight loads. (Measured: no effect at 3 -- wo_a's weight packets still
#: waited behind them at in8192_bs10; harmless, kept.)
CACHE_COPY_PRIORITY = 3


def _copy_rows(src, dst, pid, chunk_bytes=CACHE_COPY_CHUNK, dep=None):
    """`dst = src` (HBM -> HBM) for core `pid`'s half of the rows, in DMAs of about
    `chunk_bytes`. With `dep` (a `[1, w]` SBUF tile of dst's dtype) every chunk's first
    elements are first written from `dep`, so the copy (which overwrites them) cannot start
    before whatever `dep` depends on (write-after-write on dst)."""
    rows = src.shape[0]
    h = (rows + 1) // 2
    lo, hi = (0, h) if pid == 0 else (h, rows)
    if hi <= lo:
        return
    row_bytes = 1
    for d in src.shape[1:]:
        row_bytes *= d
    row_bytes *= 4 if src.dtype == nl.float32 else 2
    step = max(1, chunk_bytes // row_bytes)
    for a in range(lo, hi, step):
        b = min(hi, a + step)
        if dep is not None:
            nisa.dma_copy(dst=dst[a : a + 1, 0 : dep.shape[1]], src=dep)
        if CACHE_COPY_ENGINE is None:
            nisa.dma_copy(dst=dst[a:b], src=src[a:b], priority=CACHE_COPY_PRIORITY)
        else:
            nisa.dma_copy(dst=dst[a:b], src=src[a:b], dge_mode=nisa.dge_mode.hwdge,
                          engine=CACHE_COPY_ENGINE)


def _barrier(t):
    """`core_barrier` on `t`, executed by BARRIER_ENGINE."""
    if BARRIER_ENGINE is None:
        nisa.core_barrier(data=t, cores=(0, 1))
    else:
        nisa.core_barrier(data=t, cores=(0, 1), engine=BARRIER_ENGINE)


def _half(n, pid):
    """`(lo, hi)` of core `pid`'s share of `n` columns, in whole 128-column tiles."""
    tiles = n // PT
    a = (tiles + 1) // 2
    lo = 0 if pid == 0 else a * PT
    hi = a * PT if pid == 0 else n
    return lo, hi


#: (fused launch) Blocks reading the same slot-table inputs (top-k, cbase, twin / wv,
#: cmp_slots, the compressed cache) share ONE kvtab / idx / idxp (`share`, a dict the launch
#: passes to its members): the first builds the compressed rows, the index tables and the
#: pads, the later ones only gather their own window rows into it (the same bytes: those
#: rows / tables depend on nothing a block computes). L33-L35 reuse L32's, L37-L39 L36's.
SHARE_SLOT_TABLE = True
#: (fused launch) Index sources reading the same index-key cache and slots (L32 and L36: the
#: launch writes neither) score against the same keys: the first stores each core's
#: gathered, transposed keys (kT, per span) in a shared HBM table, the later one loads them
#: with contiguous row DMAs instead of the SWDGE row gather + PE transposes (the same bytes).
SHARE_INDEX_KEYS = True


def _tab_share(share, key, objs):
    """The (kvtab, idx, idxp) an earlier member built from the same inputs, else None."""
    if share is None or not SHARE_SLOT_TABLE:
        return None
    ent = share.get("tab")
    if ent is None or ent[0] != key or len(ent[1]) != len(objs):
        return None
    if not all(a is b for a, b in zip(ent[1], objs)):
        return None
    return ent[2]


@nki.jit
def layer_decode_r1_idx_tp64(
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
    groups_attn, groups_world, flags, share=None,
):
    ratio, nspans, win, stage, is_kv_src, is_idx_src, has_engram, tap, kver = flags
    assert (ratio, is_kv_src, is_idx_src, has_engram, tap) == (1, 0, 1, 0, 0), (
        f"flags {flags} are not an r1_idx block's")
    assert ratio == 1, "the r1 decode kernels"
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
    topk = min(topk_in.shape[1], read_len)
    L = T // nspans
    half = cdiv(L, CORES)
    kd = ix_wk.shape[0]
    nh_idx = ix_wproj_t.shape[1]
    IDX_SCALE = float(kd) ** -0.5 * float(nh_idx * groups_attn[0].__len__()) ** -0.5
    # compact slot table (only the rows the attention reads): each span's window rows,
    # then each row's top-k compressed rows, when that is fewer rows than every span's
    # read_len compressed rows
    compact = CMP_COMPACT and topk > 0 and T * topk < nspans * read_len
    # (the compressed rows start at a multiple of 128: `gather_rows_via`'s batched stores)
    ncmp0 = pad_to(nspans * nwin, PT) if compact else nspans * nwin
    ntab = ncmp0 + T * topk if compact else nspans * nspan_rows
    ntab_p = pad_to(ntab, PT)
    # Prefill keeps the slot table transposed in SBUF and selects keys on chip when
    # it fits; otherwise the per-token transposes.
    resident = ATTN_TI and kt_resident(ntab, hd)
    batched = ntab_p >= ATTN_BATCHED_GATHER_ROWS
    kpad = pad_to(win + topk, KTILE)
    scale = float(hd) ** -0.5
    assert hpg * ngroups == heads, "o_groups must divide the local heads"
    assert q_lora % PT == 0 and dim % PT == 0 and hd % PT == 0
    wq_a4, wq_b4, wkv4 = wq_a.view(X.FP8_X4), wq_b.view(X.FP8_X4), wkv.view(X.FP8_X4)
    wo_a4, wo_b4 = wo_a.view(X.FP8_X4), wo_b.view(X.FP8_X4)
    ix_wq_b4 = ix_wq_b.view(X.FP8_X4)
    sh_w1_4, sh_w3_4, sh_w2_4 = sh_w1.view(X.FP8_X4), sh_w3.view(X.FP8_X4), sh_w2.view(X.FP8_X4)
    w1_4, w3_4, w2_4 = w1.view(X.FP4_X4), w3.view(X.FP4_X4), w2.view(X.FP4_X4)
    world = len(groups_world[0])

    pid = nl.program_id(0)
    rg_attn = ncc.ReplicaGroup([list(g) for g in groups_attn])
    rg_world = ncc.ReplicaGroup([list(g) for g in groups_world])
    mine = tiles_of(T, pid)
    hsplit = (HCO_SPLIT and T <= PT and not tiles_of(T, 1) and dim % (2 * PT) == 0
              and hc_out_spread_s(T, dim // 2) > 1)

    # ------------------------------------------------------------------ outputs
    out_stream = nl.ndarray((T, hc, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    ffn_pre = nl.ndarray((T, hc), dtype=nl.float32, buffer=nl.shared_hbm)
    topk_out = nl.ndarray((T, topk), dtype=nl.int32, buffer=nl.shared_hbm)
    tap_out = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    if CACHE_COPY_AT_AR:
        # the returned caches as the kernel's own outputs, copied by the kernel at a point
        # of its choosing (returning the inputs makes the compiler copy them at the end,
        # where the copies saturated the DMA queues ahead of the world all-reduce)
        win_out = nl.ndarray(window_kv_cache.shape, dtype=window_kv_cache.dtype, buffer=nl.shared_hbm)
        cmp_out = nl.ndarray(compress_kv_cache.shape, dtype=compress_kv_cache.dtype, buffer=nl.shared_hbm)
        idxk_out = nl.ndarray(index_k_cache.shape, dtype=index_k_cache.dtype, buffer=nl.shared_hbm)
        cache_copies = [(window_kv_cache, win_out), (compress_kv_cache, cmp_out), (index_k_cache, idxk_out)]

    # ------------------------------------------------------------------ scratch
    strm = stream.reshape((T, width))
    out2 = out_stream.reshape((T, width))
    xn = nl.ndarray((Tp, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="xn")
    pre_a = nl.ndarray((T, hc), dtype=nl.float32, buffer=nl.shared_hbm, name="pre_a")
    post_a = nl.ndarray((T, hc), dtype=nl.float32, buffer=nl.shared_hbm, name="post_a")
    comb_a = nl.ndarray((T, hc * hc), dtype=nl.float32, buffer=nl.shared_hbm, name="comb_a")
    comb1 = (COMB_A_CORE1 and HC_SINKHORN_BATCHED and COMB_DEFERRED and COMB_AT_AR
             and T <= PT and not tiles_of(T, 1))
    if comb1:
        t2a_h = nl.ndarray((T, hc * hc), dtype=nl.float32, buffer=nl.shared_hbm, name="t2a_h")
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
    tab_key = (T, nspans, nwin, read_len, topk, win, kpad, ntab, ntab_p, nzero, Tq, compact, ncmp0,
               batched, resident, kv_pair_layout(hd), hd)
    tab_objs = (compress_kv_cache, cmp_slots, cbase, twin, wv)
    # (an index source always builds its own table -- its attention reads its own top-k -- and
    # offers it to the next members once that top-k exists, SHARE_SLOT_TABLE)
    reuse = None
    kvtab = nl.ndarray((ntab_p + nzero, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kvtab")
    idx = nl.ndarray((Tq, kpad), dtype=nl.int32, buffer=nl.shared_hbm, name="idx")
    # idx permuted for the attention loop's batched value gathers (`attn_index_table`)
    idxp = nl.ndarray((Tq, kpad), dtype=nl.int32, buffer=nl.shared_hbm, name="idxp")
    o = nl.ndarray((To, heads * hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="o")
    oq = nl.ndarray((Tp, heads * hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="oq")
    z = nl.ndarray((Tp, o_lora_total), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="z")
    # the all-reduce payloads carry only the CC_ROWS(T) real rows (not the Tp padding)
    Tc = CC_ROWS(T, Tp, dim)
    cc_src = nl.ndarray((Tc, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((Tc, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_dst")
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
    Tc2 = CC_ROWS(T, Tp, dim)
    cc2_src = nl.ndarray((Tc2, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc2_src")
    cc2_dst = nl.ndarray((Tc2, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc2_dst")
    qiraw = nl.ndarray((Tp, nh_idx * kd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="qiraw")
    qi = nl.ndarray((Tp, nh_idx * kd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="qi")
    wiraw = nl.ndarray((Tp, nh_idx), dtype=nl.float32, buffer=nl.shared_hbm, name="wiraw")
    wi = nl.ndarray((Tp, nh_idx), dtype=nl.float32, buffer=nl.shared_hbm, name="wi")
    kt_direct = IDX_KT_DIRECT and read_len % PT == 0 and kd == PT
    ktab = nl.ndarray((1 if kt_direct else max(nspans * read_len, 1), kd), dtype=nl.bfloat16,
                      buffer=nl.shared_hbm, name="ktab")
    # (the index-score all-reduce: only the T real rows when that is a small message, as
    # CC_ROWS; rows past T are neither written nor read)
    Tsc = T if IDX_SC_ROWS_T and T * max(read_len, 1) * 4 <= CC_SMALL_BYTES else Tp
    # (else, with one span and the keys halved between the cores: two all-reduces of a
    # core's half each when that is a small message, e.g. in8192_bs1 2 x 101 KB, not 4.3 MB)
    sc_split = (IDX_SC_SPLIT and Tsc == Tp and T <= PT and nspans == 1 and kt_direct
                and read_len % (2 * PT) == 0 and T * (read_len // 2) * 4 <= CC_SMALL_BYTES)
    if sc_split:
        h2 = read_len // 2
        sc_srcs = [nl.ndarray((T, h2), dtype=nl.float32, buffer=nl.shared_hbm, name="sc_src%d" % i)
                   for i in range(CORES)]
        sc_dsts = [nl.ndarray((T, h2), dtype=nl.float32, buffer=nl.shared_hbm, name="sc_dst%d" % i)
                   for i in range(CORES)]
        Tsc = 1
    sc_src = nl.ndarray((Tsc, max(read_len, 1)), dtype=nl.float32, buffer=nl.shared_hbm, name="sc_src")
    sc_dst = nl.ndarray((Tsc, max(read_len, 1)), dtype=nl.float32, buffer=nl.shared_hbm, name="sc_dst")
    sc16 = nl.ndarray((Tp, max(read_len, 1)), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="sc16")

    kvp = kv_pair_layout(hd)
    # the top-k the compact table and the attention indices read: an index source's own
    # (computed after the projections), else the nearest index source's
    topk_src = [topk_in]

    def cmp_gather():
        # the compact table's compressed part: row r's top-k compressed rows,
        # cmp_slots[cix[r, j]], composed on each core for all rows
        on_chip = CMP_SLOTS_ON_CHIP and read_len <= CMP_SLOTS_ON_CHIP_MAX
        if on_chip and CMP_IDX_ON_CHIP and (topk % PT == 0):
            # the gather indices never leave SBUF
            cmp_gather_onchip(compress_kv_cache, cmp_slots, kvtab, ncmp0, T, cbase,
                              topk_src[0], topk, nspans, nspan_rows, nwin, read_len, pid, hd, pair=kvp,
                              cores=CORES)
            return
        cix = nl.ndarray((T, topk), dtype=nl.int32, buffer=nl.private_hbm)
        # (small span tables: cix holds the looked-up cache rows themselves)
        cmp_compose_index(cix, T, cbase, topk_src[0], topk, nspans, nspan_rows, nwin, read_len,
                          slots_hbm=cmp_slots if on_chip else None)
        if on_chip:
            gather_rows(compress_kv_cache, cix.reshape((T * topk, 1)), kvtab[ncmp0 : ncmp0 + T * topk],
                        tiles_of(T * topk, pid), hd, pair=kvp)
        elif GATHER_VIA_BATCHED:
            gather_rows_via(compress_kv_cache, cmp_slots, cix.reshape((T * topk, 1)), kvtab, ncmp0,
                            T * topk, pid, hd, pair=kvp, cores=CORES)
        else:
            gather_rows(compress_kv_cache, cmp_slots, kvtab[ncmp0 : ncmp0 + T * topk],
                        tiles_of(T * topk, pid), hd, pair=kvp, via=cix.reshape((T * topk, 1)))


    if CACHE_COPY_AT_AR and CACHE_COPY_FIRST and (not CACHE_COPY_DEP):
        for src_c, dst_c in cache_copies[1:]:
            _copy_rows(src_c, dst_c, pid)



    # ------------------------------------------------------------------ 1. hc_in (attention)
    fnT = load_fn_t(hca_fn_t, width, mix)
    hcc = hc_in_consts(hca_base, hca_scale, attn_norm_w, hc, dim, np_=min(PT, T))
    nmine = len(mine)

    def t2_buf():
        # the call's pre-sinkhorn comb logits, one slot per row tile (zeros past a
        # partial tile's rows)
        if not HC_SINKHORN_BATCHED or nmine == 0:
            return None
        t = nl.ndarray((PT, nmine, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(t, 0.0)
        return t

    def gate(t2all, dep):
        # t2all's last tile's pad partition PT-1 (rows past a partial tile's are zeros whose
        # sinkhorn results are discarded; every op is partition-local) <- one element of `dep`
        if not (SINKHORN_GATED and COMB_AT_AR) or t2all is None or dep is None:
            return
        if mine[-1][1] >= PT:
            return
        nisa.dma_copy(dst=t2all[PT - 1 : PT, nmine - 1, 0:1], src=dep[0:1, 0:1])

    def comb_out(t2all, comb_hbm):
        if t2all is None:
            return
        cb = sinkhorn_batched(t2all, nmine, hc)
        for t, (r0, nr) in enumerate(mine):
            nisa.dma_copy(dst=comb_hbm[r0 : r0 + nr, 0 : hc * hc], src=cb[0:nr, t, :])

    if EARLY_PADS and pid == 0:
        # the zero padding rows of the GEMM inputs / routing weights, written at the start
        # (they depend on nothing) on the ACT queue, instead of right before each input's
        # barrier, where the ~1-2.5 MB pad stores held the barrier (xn2: ~30 us at T=6)
        def zpad(dst, r0, r1, n, dtype):
            zt = nl.ndarray((PT, n), dtype=dtype, buffer=nl.sbuf)
            nisa.memset(zt, 0.0, engine=nisa.engine.gpsimd)
            for a in range(r0, r1, PT):
                b = min(r1, a + PT)
                nisa.dma_copy(dst=dst[a:b, :], src=zt[0 : b - a, :], dge_mode=nisa.dge_mode.hwdge,
                              engine=EARLY_PADS_ENGINE)

        zpad(xn, T, Tp, dim, nl.bfloat16)
        zpad(qr, T, Tp, q_lora, nl.bfloat16)
        zpad(oq, T, Tp, heads * hd, nl.bfloat16)
        zpad(xn2, T, Tp + PT, dim, nl.bfloat16)
        zpad(wexp, T, Tp + PT, n_local, nl.float32)
    mixc1 = HC_MIX_CORE1 and comb1
    t2a = t2_buf()
    for t, (r0, nr) in enumerate(mine):
        hc_in_tile(r0, nr, strm, pre_mix, fnT, hca_base, hca_scale, attn_norm_w, hc, dim, mix,
                   xn, pre_a, post_a, comb_a, consts=hcc,
                   t2_dst=None if (t2a is None or mixc1) else t2a[:, t, :],
                   part="xn" if mixc1 else None)
    if comb1 and pid == 0 and not mixc1:
        nisa.dma_copy(dst=t2a_h[0:T, :], src=t2a[0:T, 0, :])
    if not COMB_DEFERRED:
        comb_out(t2a, comb_a)
    if pid == 0 and T < Tp and not EARLY_PADS and not PADS_ON_CHIP:
        pad = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(pad, 0.0)
        nisa.dma_copy(dst=xn[T:Tp, :], src=pad[0 : Tp - T, :])
    # The routed experts' fp32 accumulator (private to this core) is zeroed here, off
    # the MoE's critical path: these writes overlap the DVE-bound hc_in / projections.
    moe_dense = MOE_DENSE and Tp == PT and SHARED_FIRST
    if moe_dense:
        pass  # the routed sum lives in SBUF (`run_expert_dense`)
    elif PADS_ON_CHIP:
        # the rows the w2 epilogue reads (the scatter-add touches only real rows)
        w2f = SHARED_FIRST and SHARED_W2_FIRST and (Tc2 * 2 <= Tp or (SHARED_W2_SBUF and Tp <= PT))
        mk_moe.zero_rows(eacc, Tc2 if w2f else Tp, dim)
    else:
        mk_moe.zero_rows(eacc, Tp + PT, dim)
    _barrier(xn)

    # ------------------------------------------------------------------ 2. q / kv projections
    # MX GEMMs: the activation is quantized on chip as it is staged (each core stages
    # every row and computes its half of the output columns).
    # wq_a and wkv share one staging of the xn rows.
    X.gemm_rows_multi(xn, Tp, dim, [(wq_a4, wq_a_s, qa) + _half(q_lora, pid),
                                    (wkv4, wkv_s, kvr) + _half(hd, pid)],
                                    m_valid=T if PADS_ON_CHIP else None)
    if COMB_DEFERRED and (not COMB_AT_AR):
        # hc_in(attn)'s sinkhorn alongside the projection GEMMs (comb_a is first read by
        # hc_out(attn), after the attention)
        comb_out(t2a, comb_a)
    _barrier(qa)
    _barrier(kvr)
    # q_norm (wq_b quantizes its input itself); kv_norm -> rope -> fp8 round trip.
    kv1 = KV_CORE1 and T <= PT and not tiles_of(T, 1)
    kvt = tiles_of(T, 1 - pid) if kv1 else mine
    norm_rows(qa, q_norm_w, qr, mine, q_lora, qdq=False, chunk=q_lora)
    norm_rows(kvr, kv_norm_w, kv, kvt, hd, qdq=False, chunk=hd)
    rope_rows(kv, kv, kvt, hd, hd, 1, freqs, row_pos, inverse=False, qdq=True)
    if comb1 and not mixc1:
        _barrier(t2a_h)
        if pid == 1:
            # (rows past T zero, as t2_buf's; every op is partition-local)
            t2c = nl.ndarray((PT, 1, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(t2c, 0.0)
            nisa.dma_copy(dst=t2c[0:T, 0, :], src=t2a_h[0:T, :])
            cb1 = sinkhorn_batched(t2c, 1, hc)
            nisa.dma_copy(dst=comb_a[0:T, 0 : hc * hc], src=cb1[0:T, 0, :])
    if pid == 0 and T < Tp and not EARLY_PADS and not PADS_ON_CHIP:
        pad = nl.ndarray((PT, q_lora), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(pad, 0.0)
        nisa.dma_copy(dst=qr[T:Tp, :], src=pad[0 : Tp - T, :])
    _barrier(qr)
    _barrier(kv)
    # This step's KV into the paged cache (the input, mutated in place).
    scatter_rows(kv, write_slots, window_kv_cache, kvt, hd)
    lo, hi = _half(heads * hd, pid)
    X.gemm_rows(qr, Tp, q_lora, wq_b4, wq_b_s, q, lo, hi, m_valid=T if PADS_ON_CHIP else None)
    _barrier(q)
    _barrier(window_kv_cache)
    # Rope on each head's trailing 64 columns, by row tile.
    rsplit = ROPE_SPLIT and T <= PT and not tiles_of(T, 1) and heads % 2 == 0
    if rsplit:
        hw2 = heads * hd // 2
        rope_rows(q, q, tiles_of(T, 0), hw2, hd, heads // 2, freqs, row_pos, inverse=False,
                  col0=pid * hw2)
    else:
        rope_rows(q, q, mine, heads * hd, hd, heads, freqs, row_pos, inverse=False)
    _barrier(q)


    # ------------------------------------------------------------------ 2c. index queries (index sources)
    lo, hi = _half(nh_idx * kd, pid) if nh_idx * kd >= 2 * PT else (0, nh_idx * kd)
    if pid == 0 or nh_idx * kd >= 2 * PT:
        X.gemm_rows(qr, Tp, q_lora, ix_wq_b4, ix_wq_b_s, qiraw, lo, hi, m_valid=T if PADS_ON_CHIP else None)
    fp32_proj_rows(xn, ix_wproj_t, wiraw, mine, dim, nh_idx)
    _barrier(qiraw)
    rope_fp4_put(qiraw, mine, nh_idx * kd, kd, nh_idx, freqs, row_pos, 32, False, dst_hbm=qi)
    for r0, nr in mine:
        w = nl.ndarray((PT, nh_idx), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w[0:nr, :], src=wiraw[r0 : r0 + nr, 0:nh_idx])
        w16 = nl.ndarray((PT, nh_idx), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=w16[0:nr, :], src=w[0:nr, :])
        nisa.tensor_scalar(dst=w[0:nr, :], data=w16[0:nr, :], op0=nl.multiply, operand0=IDX_SCALE)
        nisa.dma_copy(dst=wi[r0 : r0 + nr, 0:nh_idx], src=w[0:nr, :])
    _barrier(qi)
    _barrier(wi)
    # this step's top-k from the index keys (`layer_kernel`'s), before the slot table
    span_split = IDX_SPAN_SPLIT and nspans >= CORES
    all_tiles = tiles_of(read_len, 0) + tiles_of(read_len, 1)
    GB = IDX_GATHER_BATCH
    # (not with the span split: whole spans per core by batches measured slower, e.g.
    # in1024_bs10 9.51 vs 8.80 ms; with one span the batches alternate between the cores)
    # (`kbatched`, not `batched`: that one (above) selects the attention's batched gathers)
    kbatched = (GB > 1 and not span_split and read_len % (GB * PT) == 0
                and cmp_slots.shape[0] % GB == 0 and ktab.shape[0] % GB == 0)
    if kt_direct:
        # each core gathers the keys it scores straight into SBUF: whole spans (span
        # split), else every span's keys halved between the cores (all its tokens)
        kh = pad_to(cdiv(read_len, CORES), PT)
        # (span split: core 0 -- also running hc_in / the projections meanwhile -- takes the
        # first n0 spans (IDX_SPAN_SPLIT_MODEL / IDX_SPAN_CORE0_FRAC), core 1 the rest)
        # (SHARE_INDEX_KEYS: an earlier index source's kT of the same keys: span b's
        # transposed keys in kt_tab[b] (shared: with the span split a core may reload spans
        # the other one gathered; the source block barriers on it after its stores))
        kt_key = (nspans, read_len, kd, span_split, kh, IDX_DIRECT_BATCH)
        kt_tab, kt_mode = None, None
        if share is not None and SHARE_INDEX_KEYS:
            ent = share.get("kT")
            if (ent is not None and ent[0] == kt_key and ent[1][0] is index_k_cache
                    and ent[1][1] is cmp_slots):
                kt_tab, kt_mode = ent[2], "load"
            else:
                kt_tab = nl.ndarray((nspans, PT, read_len), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                                    name="kt_share")
                kt_mode = "save"
                share["kT"] = (kt_key, (index_k_cache, cmp_slots), kt_tab)
        frac = IDX_SPAN_CORE0_FRAC_LOADED if kt_mode == "load" else IDX_SPAN_CORE0_FRAC
        n0 = min(nspans - 1, max(1, int(nspans * frac + 0.5)))
        if IDX_SPAN_SPLIT_MODEL and kt_mode != "load" and nspans >= CORES:
            su = read_len * IDX_SPAN_NS_PER_KEY * 1e-3
            head0 = IDX_CORE0_HEAD_US_MIX1 if mixc1 else IDX_CORE0_HEAD_US

            def _cost(n):
                return (max(head0 + n * su, (nspans - n) * su), -n)

            n0 = min(range(1, nspans), key=_cost)

        def kt_args(b, k0, nk):
            if kt_tab is None:
                return {}
            v = kt_tab[b, 0:PT, k0 : k0 + nk]
            return {"kt_load": v} if kt_mode == "load" else {"kt_save": v}

        for b in range(nspans):
            if span_split:
                if (b < n0) == (pid == 0):
                    index_scores_span(qi, wi, None, sc_src, b * L, L, nh_idx, read_len, kd,
                                      gather=(index_k_cache, cmp_slots, b * read_len, IDX_DIRECT_BATCH),
                                      **kt_args(b, 0, read_len))
                continue
            k0 = 0 if pid == 0 else kh
            nk = min(kh, read_len) if pid == 0 else read_len - kh
            if nk > 0:
                index_scores_span(qi, wi, None, sc_srcs[pid] if sc_split else sc_src, b * L, L, nh_idx,
                                  read_len, kd,
                                  gather=(index_k_cache, cmp_slots, b * read_len, IDX_DIRECT_BATCH),
                                  key0=k0, nkeys=nk, sc_col0=0 if sc_split else None,
                                  **kt_args(b, k0, nk))
        if kt_mode == "save":
            _barrier(kt_tab)
    for b in range(0 if kt_direct else nspans):
        if span_split and b % CORES != pid:
            continue
        if kbatched:
            gather_rows_batched(index_k_cache, cmp_slots, ktab, b * read_len, read_len, kd, GB,
                                pid=0 if span_split else pid, cores=1 if span_split else CORES)
            continue
        gather_rows(index_k_cache, cmp_slots[b * read_len : (b + 1) * read_len],
                    ktab[b * read_len : (b + 1) * read_len],
                    all_tiles if span_split else tiles_of(read_len, pid), kd)
    if not kt_direct:
        _barrier(ktab)
    for b in range(0 if kt_direct else nspans):
        if span_split:
            if b % CORES == pid:
                index_scores_span(qi, wi, ktab[b * read_len : (b + 1) * read_len], sc_src, b * L, L,
                                  nh_idx, read_len, kd)
            continue
        itok0 = b * L + (0 if pid == 0 else L - half)
        index_scores_span(qi, wi, ktab[b * read_len : (b + 1) * read_len], sc_src, itok0, half,
                          nh_idx, read_len, kd)
    def mix1():
        # (HC_MIX_CORE1, index source) hc_in(attn)'s coefficients on core 1 (core 0 made the
        # normed rows) while the index-score all-reduce is in flight, the comb logits straight
        # into the sinkhorn's tile (rows past T zero, as t2_buf's)
        if not (mixc1 and pid == 1):
            return
        t2c = nl.ndarray((PT, 1, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(t2c, 0.0)
        hc_in_tile(0, T, strm, pre_mix, fnT, hca_base, hca_scale, attn_norm_w, hc, dim, mix,
                   xn, pre_a, post_a, comb_a, consts=hcc, t2_dst=t2c[:, 0, :], part="mix")
        cb1 = sinkhorn_batched(t2c, 1, hc)
        nisa.dma_copy(dst=comb_a[0:T, 0 : hc * hc], src=cb1[0:T, 0, :])

    if sc_split:
        for t_ in sc_srcs:
            _barrier(t_)
        for a_, b_ in zip(sc_srcs, sc_dsts):
            ncc.all_reduce(srcs=[a_], dsts=[b_], replica_group=rg_attn, op=nl.add)
        mix1()
        index_topk(tuple(sc_dsts), clen, sc16, topk_out, T, Tp, read_len, topk, pid)
    else:
        _barrier(sc_src)
        ncc.all_reduce(srcs=[sc_src], dsts=[sc_dst], replica_group=rg_attn, op=nl.add)
        mix1()
        index_topk(sc_dst, clen, sc16, topk_out, T, Tp, read_len, topk, pid)
    _barrier(topk_out)
    topk_src[0] = topk_out
    if share is not None and SHARE_SLOT_TABLE:
        share["tab"] = (tab_key, tab_objs + (topk_out,), (kvtab, idx, idxp))

    # ------------------------------------------------------------------ 3. slot table + indices
    # Span b's rows of the slot table: its window rows, then its compressed rows.
    # (in the pair-interleaved column order the attention transposes from, `kv_pair_layout`)
    assert not (resident and kvp), "the resident (TI) path reads the natural column order"
    if compact:
        if not CMP_GATHER_EARLY:
            cmp_gather()
        for b in range(nspans):
            # alternate the cores' (full / partial) window tiles by span
            gather_rows(window_kv_cache, win_slots[b * nwin : (b + 1) * nwin],
                        kvtab[b * nwin : (b + 1) * nwin], tiles_of(nwin, (pid + b) % CORES), hd,
                        pair=kvp)
    else:
        for b in range(nspans):
            gather_rows(window_kv_cache, win_slots[b * nwin : (b + 1) * nwin],
                        kvtab[b * nspan_rows : b * nspan_rows + nwin], tiles_of(nwin, pid), hd, pair=kvp)
            if reuse is None:
                gather_rows(compress_kv_cache, cmp_slots[b * read_len : (b + 1) * read_len],
                            kvtab[b * nspan_rows + nwin : (b + 1) * nspan_rows], tiles_of(read_len, pid), hd,
                            pair=kvp)
    attn_index_table(idx, mine, twin, wv, win, cbase, topk_src[0], topk, kpad,
                     idxp_hbm=idxp if batched else None, zrow0=ntab_p if batched else None,
                     compact=(nspans, nspan_rows, nwin, read_len, ncmp0) if compact else None)
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
    if pid == 1 and ntab < ntab_p + nzero and reuse is None:
        padk = nl.ndarray((PT, hd), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(padk, 0.0)
        for z0 in range(ntab, ntab_p + nzero, PT):
            z1 = min(z0 + PT, ntab_p + nzero)
            nisa.dma_copy(dst=kvtab[z0:z1, :], src=padk[0 : z1 - z0, :])
    _barrier(kvtab)
    _barrier(idx)
    if batched:
        _barrier(idxp)

    # ------------------------------------------------------------------ 4. sparse attention
    sink_t = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sink_t, src=attn_sink[0:heads, 0:1])
    # Tokens per loop step: TOK_UNROLL, at most 128 stacked (token, head) rows for
    # `attn_step`, a power of two (it divides Tp).
    pipe = ATTN_PIPE and batched and not resident
    U = TOK_UNROLL if resident else max(1, min(ATTN_PIPE_U if pipe else TOK_UNROLL, PT // heads))
    # Core 0 takes rows [0, h), core 1 rows [T - h, T): equal trip counts, and for an
    # odd T the middle row is computed by both (identically).
    ntok = cdiv(T, CORES)
    if ATTN_U_FIT and (not resident):
        # the step size (dividing the q / idx / o views' rows) with the fewest loop
        # bodies for this core's ntok tokens (a step's values are the single-token
        # body's, `attn_compute`)
        cands = [u for u in range(1, U + 1) if Tq % u == 0 and To % u == 0]
        U = min(cands, key=lambda u: (ntok // u + ntok % u, -u))
    else:
        while Tp % U:
            U //= 2
    # the sinks repeated for each token of a step (`attn_step`'s stacked rows)
    sink_u = nl.ndarray((U * heads, 1), dtype=nl.float32, buffer=nl.sbuf)
    for u in range(U):
        nisa.dma_copy(dst=sink_u[u * heads : (u + 1) * heads, :], src=attn_sink[0:heads, 0:1])
    masks = None if resident else attn_step_masks(U, heads, hd, kpad)
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
        while nsp > 2 and (nsp * U > PT or ntok < nsp * U or Tq % (nsp * U) or To % (nsp * U)):
            nsp //= 2
    npipe = ntok // (nsp * U) if pipe and Tq % (nsp * U) == 0 and To % (nsp * U) == 0 else 0

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
    _barrier(o)

    # ------------------------------------------------------------------ 5. output projections
    # Inverse rope, grouped wo_a, wo_b (fp32 partial); the MX GEMMs quantize their inputs.
    if rsplit:
        rope_rows(o, oq, tiles_of(T, 0), hw2, hd, heads // 2, freqs, row_pos, inverse=True, qdq=False,
                  col0=pid * hw2)
    else:
        rope_rows(o, oq, mine, heads * hd, hd, heads, freqs, row_pos, inverse=True, qdq=False)
    if pid == 0 and T < Tp and not EARLY_PADS and not PADS_ON_CHIP:
        pad = nl.ndarray((PT, heads * hd), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(pad, 0.0)
        nisa.dma_copy(dst=oq[T:Tp, :], src=pad[0 : Tp - T, :])
    _barrier(oq)
    for g in range(ngroups):
        # Column split of the group's `o_lora` outputs across the cores; the group's
        # block of `wo_a` is columns [g*o_lora, (g+1)*o_lora) of the MX layout against
        # the group's `g_in` input columns.
        lo, hi = _half(o_lora, pid)
        X.gemm_rows(oq, Tp, g_in, wo_a4, wo_a_s, z, g * o_lora + lo, g * o_lora + hi, x_col0=g * g_in,
                    m_valid=T if PADS_ON_CHIP else None)
    _barrier(z)
    lo, hi = _half(dim, pid)

    wo_b_last = [None]

    def wo_b_epi(r, n0, w, o_t):
        nr = min(PT, Tc - r)
        if nr > 0:
            nisa.dma_copy(dst=cc_src[r : r + nr, n0 : n0 + w], src=o_t[0:nr, :])
            wo_b_last[0] = o_t

    X.gemm_rows(z, Tp, o_lora_total, wo_b4, wo_b_s, cc_src, lo, hi, out_dtype=nl.float32,
                epilogue=wo_b_epi)
    _barrier(cc_src)
    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], replica_group=rg_attn, op=nl.add)
    pads_ar1 = PADS_AT_AR1 and not EARLY_PADS and Tc2 * 2 <= Tp
    if pads_ar1 and pid == 0:
        pad = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(pad, 0.0)
        if T < Tp and not PADS_ON_CHIP:
            nisa.dma_copy(dst=xn2[T:Tp, :], src=pad[0 : Tp - T, :])
        if not PADS_ON_CHIP:
            nisa.dma_copy(dst=xn2[Tp : Tp + PT, :], src=pad)
        zw = nl.ndarray((PT, n_local), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(zw, 0.0)
        if T < Tp:
            nisa.dma_copy(dst=wexp[T:Tp, :], src=zw[0 : Tp - T, :])
        nisa.dma_copy(dst=wexp[Tp : Tp + PT, :], src=zw)
    if CACHE_COPY_AT_AR:
        # the returned caches' copies (this core's half of the rows), while the
        # all-reduce is in flight (the window cache after this step's scatter)
        depb = None
        if CACHE_COPY_DEP:
            dep = nl.ndarray((1, 2), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=dep, src=cc_src[0:1, 0:2])
            depb = nl.ndarray((1, 2), dtype=window_kv_cache.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=depb, src=dep)
        late = cache_copies[:1] if CACHE_COPY_FIRST and not CACHE_COPY_DEP else cache_copies
        for src_c, dst_c in late:
            _copy_rows(src_c, dst_c, pid, dep=depb if src_c.dtype == window_kv_cache.dtype else None)
    if COMB_DEFERRED and COMB_AT_AR:
        # hc_in(attn)'s sinkhorn while the all-reduce is in flight (comb_a is first read
        # by hc_out(attn), right after it)
        if comb1:
            _barrier(comb_a)
        else:
            gate(t2a, wo_b_last[0])
            comb_out(t2a, comb_a)
            if hsplit:
                _barrier(comb_a)
    if hsplit or mixc1:
        _barrier(post_a)
    if mixc1:
        # (core 1's, read by core 0's hc_in(ffn))
        _barrier(pre_a)

    # ------------------------------------------------------------------ 6. hc_out (attention)
    # hc_out reads the fp32 all-reduce result into bf16 by DMA: the bf16 cast of the
    # attention output, without a bf16 copy through HBM.
    # ------------------------------------------------------------------ 7. hc_in (ffn)
    # fused with 6 by row tile: hc_in reads the tile hc_out just produced from SBUF (the
    # stream2 rows are still stored for hc_out (ffn)'s residual).
    fnT2 = load_fn_t(hcf_fn_t, width, mix)
    hcc2 = hc_in_consts(hcf_base, hcf_scale, ffn_norm_w, hc, dim, np_=min(PT, T))
    # skewed: tile t+1's hc_out is issued before tile t's hc_in
    yss = [None] * len(mine)
    t2f = t2_buf()
    mixf1 = HC_MIXF_CORE1 and hsplit and HC_SINKHORN_BATCHED
    if hsplit:
        dh = dim // 2
        hc_out_tile_spread(0, T, cc_dst, strm, post_a, comb_a, stream2, hc, dim,
                           hc_out_spread_s(T, dh), col0=pid * dh, ncols=dh)
        _barrier(stream2)
    if mixf1 and pid == 1:
        t2f1 = nl.ndarray((PT, 1, hc * hc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(t2f1, 0.0)
        hc_in_tile(0, T, stream2, pre_a, fnT2, hcf_base, hcf_scale, ffn_norm_w, hc, dim, mix,
                   xn2, ffn_pre, post_f, comb_f, consts=hcc2, t2_dst=t2f1[:, 0, :], part="mix")
    for t in range(len(mine) + 1):
        if hsplit:
            if t < len(mine):
                r0, nr = mine[t]
                hc_in_tile(r0, nr, stream2, pre_a, fnT2, hcf_base, hcf_scale, ffn_norm_w, hc, dim,
                           mix, xn2, ffn_pre, post_f, comb_f, consts=hcc2,
                           t2_dst=None if (t2f is None or mixf1) else t2f[:, t, :],
                           part="xn" if mixf1 else None)
            continue
        if t < len(mine):
            r0, nr = mine[t]
            yss[t] = hc_out_tile(r0, nr, cc_dst, strm, post_a, comb_a, stream2, hc, dim, keep=True,
                                 acc_first=(t == 0))
        if t > 0:
            r0, nr = mine[t - 1]
            hc_in_tile(r0, nr, stream2, pre_a, fnT2, hcf_base, hcf_scale, ffn_norm_w, hc, dim, mix,
                       xn2, ffn_pre, post_f, comb_f, consts=hcc2, xin=yss[t - 1],
                       t2_dst=None if t2f is None else t2f[:, t - 1, :])
    if not COMB_AT_AR:
        comb_out(t2f, comb_f)
    if pid == 0 and (not EARLY_PADS) and (not pads_ar1):
        pad = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(pad, 0.0)
        if T < Tp and not PADS_ON_CHIP:
            nisa.dma_copy(dst=xn2[T:Tp, :], src=pad[0 : Tp - T, :])
        if not PADS_ON_CHIP:
            nisa.dma_copy(dst=xn2[Tp : Tp + PT, :], src=pad)
    if not mixf1:
        _barrier(xn2)

    # ------------------------------------------------------------------ 8. MoE
    # Router (fp32 GEMM + top-k) split by row tile between the cores, then every core
    # reads all rows' expert weights for its experts.
    if ROUTER_COL_SPLIT and (not tiles_of(T, 1)) and (n_experts % 2 == 0) and not mixf1:
        # one row tile: each core the router GEMM's half of the experts' columns (each
        # column's accumulation unchanged), then core 0 its top-k
        nh = n_experts // 2
        gemm_fp32(xn2, gate_w_t, logits, tiles_of(T, 0), dim, nh, n0=pid * nh)
        _barrier(logits)
    else:
        gemm_fp32(xn2, gate_w_t, logits, mine, dim, n_experts)
    route_topk(logits, gate_b, expert_lo, mine, n_experts, TOPK_EXPERTS, n_local, ROUTE_SCALE, wexp)
    if pid == 0 and (not EARLY_PADS) and (not pads_ar1):
        zw = nl.ndarray((PT, n_local), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(zw, 0.0)
        if T < Tp:
            nisa.dma_copy(dst=wexp[T:Tp, :], src=zw[0 : Tp - T, :])
        nisa.dma_copy(dst=wexp[Tp : Tp + PT, :], src=zw)
    if mixf1:
        _barrier(xn2)
    _barrier(wexp)
    ptiles = tiles_of(Tp, 0) + tiles_of(Tp, 1)

    xst = []

    def shared_up():
        # Shared expert: this core's half of the intermediate columns (SwiGLU is
        # elementwise along it and w2 contracts over it).
        lo, hi = _half(inter, pid)
        if moe_dense:
            # the xn2 tile staged once: the shared expert's and every routed expert's
            # stationary (`gemm_rows_multi`'s staging and GEMMs for its single row tile)
            xq2, xs2 = X.alloc_x(1, dim)
            X.stage_x(xn2, 0, dim, xq2, xs2, 0, nvalid=min(PT, T) if PADS_ON_CHIP else PT)
            xst.append((xq2, xs2))
            for w_mx, s_mx, o_hbm in ((sh_w1_4, sh_s1, eg), (sh_w3_4, sh_s3, eu)):
                X.gemm(xq2, xs2, 1, dim, w_mx, s_mx, o_hbm, 0, lo, hi)
        else:
            X.gemm_rows_multi(xn2, Tp, dim, [(sh_w1_4, sh_s1, eg, lo, hi), (sh_w3_4, sh_s3, eu, lo, hi)],
                              m_valid=T if PADS_ON_CHIP else None)
        swiglu_rows(eg[0:Tp, lo:hi], eu[0:Tp, lo:hi], None, 0, ey[0:Tp, lo:hi], ptiles, hi - lo,
                    SWIGLU_LIMIT)

    w2_sbuf = SHARED_W2_SBUF and Tp <= PT
    w2_first = SHARED_FIRST and SHARED_W2_FIRST and (Tc2 * 2 <= Tp or w2_sbuf)
    w2_parked = []
    if SHARED_FIRST:
        shared_up()
        if w2_first and w2_sbuf:
            # the shared expert's w2 partial (scaled) output tiles kept in SBUF until the
            # routed accumulator is complete
            lo, hi = _half(inter, pid)

            def esh_keep(r, n0, w, o):
                if r < Tc2:
                    w2_parked.append((r, n0, w, o))

            X.gemm_rows(ey, Tp, hi - lo, sh_w2_4, sh_s2, None, 0, dim, x_col0=lo, krow0=lo // 4,
                        out_dtype=nl.float32, out_scale=1.0 / world, epilogue=esh_keep)
        elif w2_first:
            # the shared expert's w2 partial (scaled), rows < Tc2 (the only ones merged),
            # parked in fp32 HBM until the routed accumulator is complete
            lo, hi = _half(inter, pid)
            esh = nl.ndarray((Tp, dim), dtype=nl.float32, buffer=nl.private_hbm)

            def esh_epi(r, n0, w, o):
                nr = min(PT, Tc2 - r)
                if nr > 0:
                    nisa.dma_copy(dst=esh[r : r + nr, n0 : n0 + w], src=o[0:nr, :])

            X.gemm_rows(ey, Tp, hi - lo, sh_w2_4, sh_s2, esh, 0, dim, x_col0=lo, krow0=lo // 4,
                        out_dtype=nl.float32, out_scale=1.0 / world, epilogue=esh_epi)
    # Routed experts, each over the rows that picked it (this core's experts), into
    # the zeroed fp32 accumulator.
    lo, hi = _half(inter, pid)
    dense_args = None
    if moe_dense:
        wexp_sb = nl.ndarray((PT, n_local), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=wexp_sb, src=wexp[0:PT, 0:n_local])
        acc_sb = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(acc_sb, 0.0)
        dense_args = (xst[0][0], xst[0][1], wexp_sb, acc_sb)
    mk_moe.routed_experts(pid, n_local, Tp, xn2, wexp, w1_4, s1, w3_4, s3, w2_4, s2,
                          dim, inter, lo, hi, SWIGLU_LIMIT, eacc, zero_acc=False, dense=dense_args,
                          nrows=T)
    if not SHARED_FIRST:
        shared_up()
    lo, hi = _half(inter, pid)
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

    # MERGE_COL_SPLIT: core pid rejoins columns [mlo, mhi) of every row (its own partial's
    # columns there go to eo32, the rest to its `part` slot for the other core)
    mlo, mhi = _half(dim, pid)

    def w2_epi(r, n0, w, o, nr=PT):
        av = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
        if moe_dense:
            # (one row tile: r = 0) the routed sum from its SBUF accumulator
            assert r == 0
            nisa.tensor_tensor(dst=av[0:nr, :], data1=dense_args[3][0:nr, n0 : n0 + w], data2=o[0:nr, :],
                               op=nl.add, engine=W2_EPI_ENGINE)
        else:
            nisa.dma_copy(dst=av[0:nr, :], src=eacc[r : r + nr, n0 : n0 + w])
            nisa.tensor_tensor(dst=av[0:nr, :], data1=av[0:nr, :], data2=o[0:nr, :], op=nl.add,
                               engine=W2_EPI_ENGINE)
        if MERGE_COL_SPLIT:
            for a, b in ((n0, min(n0 + w, mlo)), (max(n0, mlo), min(n0 + w, mhi)),
                         (max(n0, mhi), n0 + w)):
                if b > a:
                    own = mlo <= a and b <= mhi
                    dst = eo32[r : r + nr, a:b] if own else part[pid, r : r + nr, a:b]
                    nisa.dma_copy(dst=dst, src=av[0:nr, a - n0 : b - n0])
            return
        dst = part[pid, r : r + nr, n0 : n0 + w] if r in other else eo32[r : r + nr, n0 : n0 + w]
        nisa.dma_copy(dst=dst, src=av[0:nr, :])

    if w2_first and w2_sbuf:
        # the w2 epilogue over the SBUF-parked tiles (same values and adds)
        for r, n0, w, o in w2_parked:
            w2_epi(r, n0, w, o, min(PT, Tc2 - r))
    elif w2_first:
        # the w2 epilogue over the parked partial (same values and adds as in the GEMM's
        # epilogue), rows < Tc2
        wc = min(1024, dim)
        for r in range(0, Tc2, PT):
            nr = min(PT, Tc2 - r)
            for n0 in range(0, dim, wc):
                w = min(wc, dim - n0)
                o = nl.ndarray((PT, w), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=o[0:nr, :], src=esh[r : r + nr, n0 : n0 + w])
                w2_epi(r, n0, w, o, nr)
    else:
        X.gemm_rows(ey, Tp, hi - lo, sh_w2_4, sh_s2, eo32, 0, dim, x_col0=lo, krow0=lo // 4,
                    out_dtype=nl.float32, out_scale=1.0 / world, epilogue=w2_epi)
    ach = min(1024, dim)
    _barrier(part)
    if MERGE_COL_SPLIT:
        mtiles = [(r0, min(PT, Tc2 - r0)) for r0 in range(0, Tc2, PT)]
        mcols = [(c0, min(c0 + ach, mhi)) for c0 in range(mlo, mhi, ach)]
    else:
        mtiles = tiles_of(Tc2, pid)
        mcols = [(c * ach, (c + 1) * ach) for c in range(dim // ach)]
    merge_last = None
    for r0, nr in mtiles:
        for c0, c1 in mcols:
            cw = c1 - c0
            av = nl.ndarray((PT, ach), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=av[0:nr, 0:cw], src=eo32[r0 : r0 + nr, c0:c1])
            bv = nl.ndarray((PT, ach), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=bv[0:nr, 0:cw], src=part[1 - pid, r0 : r0 + nr, c0:c1])
            p0, p1 = (av, bv) if pid == 0 else (bv, av)
            nisa.tensor_tensor(dst=av[0:nr, 0:cw], data1=p0[0:nr, 0:cw], data2=p1[0:nr, 0:cw], op=nl.add)
            nisa.dma_copy(dst=cc2_src[r0 : r0 + nr, c0:c1], src=av[0:nr, 0:cw])
            merge_last = av
    _barrier(cc2_src)
    ncc.all_reduce(srcs=[cc2_src], dsts=[cc2_dst], replica_group=rg_world, op=nl.add)
    if COMB_AT_AR and mixf1:
        # (core 1's comb_f logits)
        if pid == 1:
            if SINKHORN_GATED and T < PT and merge_last is not None:
                nisa.dma_copy(dst=t2f1[PT - 1 : PT, 0, 0:1], src=merge_last[0:1, 0:1])
            cbf = sinkhorn_batched(t2f1, 1, hc)
            nisa.dma_copy(dst=comb_f[0:T, 0 : hc * hc], src=cbf[0:T, 0, :])
    elif COMB_AT_AR:
        # hc_in(ffn)'s sinkhorn while the world all-reduce is in flight (comb_f is first
        # read by hc_out(ffn)), off the router / MoE chain
        gate(t2f, merge_last)
        comb_out(t2f, comb_f)

    # ------------------------------------------------------------------ 9. hc_out (ffn)
    # (the bf16 cast of the ffn output in hc_out's DMA, as in 6)
    # one tile of lookahead: tile t+1's comb*res accumulators are issued before tile t's
    # x-dependent half, so two tiles' accumulate while the all-reduce is in flight
    def hco_acc(t):
        r0, nr = mine[t]
        return hc_out_tile(r0, nr, cc2_dst, stream2, post_f, comb_f, out2, hc, dim, acc_only=True)

    if hsplit:
        # core 0's hc_in(ffn) coefficients / sinkhorn (during the all-reduce)
        _barrier(post_f)
        _barrier(comb_f)
        dh = dim // 2
        hc_out_tile_spread(0, T, cc2_dst, stream2, post_f, comb_f, out2, hc, dim,
                           hc_out_spread_s(T, dh), col0=pid * dh, ncols=dh)
    pend = hco_acc(0) if nmine and not hsplit else None
    for t, (r0, nr) in enumerate([] if hsplit else mine):
        nxt = hco_acc(t + 1) if t + 1 < nmine else None
        hc_out_tile(r0, nr, cc2_dst, stream2, post_f, comb_f, out2, hc, dim, accs_in=pend)
        pend = nxt
    _barrier(out_stream)
    _barrier(ffn_pre)
    if CACHE_COPY_AT_AR:
        for _, dst_c in cache_copies:
            _barrier(dst_c)
        return (out_stream, ffn_pre, topk_out, tap_out, win_out, cmp_out, pool_kv, pool_score,
                idxk_out)
    return out_stream, ffn_pre, topk_out, tap_out, window_kv_cache, compress_kv_cache, pool_kv, pool_score, index_k_cache



#: The kernel on both physical cores of the logical NeuronCore (LNC=2).
kernel = layer_decode_r1_idx_tp64[2]
