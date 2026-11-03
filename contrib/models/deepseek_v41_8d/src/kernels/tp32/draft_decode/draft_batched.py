# SPDX-License-Identifier: Apache-2.0
"""The DSpark drafter -- `model.forward_spec` for every sequence of a step -- as one NKI kernel.

B sequences a launch (B * K draft rows, one 128-row tile; B <= 25 at K = 5): the
row-wise stages see one tile of rows, each sequence has its own window in the slot
table and its own block table, and the token loop drafts the i-th token of every
sequence at once (one markov all-reduce and one argmax all-gather per token instead
of one per token per sequence).

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
rejection sampler -- so every index that depends on it is computed here from
`meta` (the per-row positions, ids and window validity bases, the window's cache
slots `block_table[b][p // bs] * bs + p % bs` per sequence); the row's sequence
and markov row are host constants (`rows_c`, `mk_model.draft_consts`).

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
masked local logits and the collapsed hidden. Inputs: `meta [B, 3]` int32 (seed,
tap, pos) per sequence, `rows_c [B*K, 4]` int32 (i, b, slot-table base, markov
row) per draft row, `main_hidden [B*L, in]`,
`row_pos` / `write_slots [B*L, 1]`, `block_table [B*nblk, 1]`. Outputs:
`output_ids [B, K+1]` int32, `confidence [B*K, 1]` fp32 (row b*K + i), `dbg`, and the
three window caches (aliased, mutated in place).
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
    attn_rows_dense,
    attn_sink_rows,
    attn_token_body_ti,
    attn_units,
    bcast_row,
    build_kT,
    cast_rows,
    cdiv,
    copy_rows_hw,
    engram_gather,
    gather_rows,
    gemm_fp32,
    hc_in_mix,
    hc_in_mix_sink,
    hc_in_norm,
    hc_in_norm_split,
    hc_in_norm_spread,
    hc_out_chunks,
    hc_out_spread,
    hc_out_tile,
    moe_partial,
    load_fn_t,
    norm_rows,
    norm_rows_sb,
    pad_to,
    rope_rows,
    route_topk,
    scatter_rows,
    sched_after,
    swiglu_rows,
    tiles_of,
    transpose_pe,
)
from .mk_compress import bf16_gemm_rows

CORES = 2
SWIGLU_LIMIT = 10.0
ROUTE_SCALE = 1.5
NEG = -1.0e30
ATTN_UNROLL = 2
#: hc_in_mix of the ffn sublayer on core 1 while core 0 runs its hc_in_norm (else in
#: the world all_reduce's shadow)
HC_MIX_EARLY = False
#: the attention sublayer's hc_in_norm split over both cores by columns
HC_NORM_SPLIT = True
#: hc_in_norm (both sublayers) on core 0 with the elementwise parts spread over the partitions
HC_NORM_SPREAD = True
#: hc_out with the rows' column blocks spread over the partitions
HC_OUT_SPREAD = True
#: The returned caches are fresh outputs filled by our own HBM -> HBM copies (each core
#: half the rows, ~1 MB DMAs on the hardware DGE queues), cache s < last schedule-pinned
#: into stage s + 1's attention all_reduce (DMA-idle), the last one into the head, instead
#: of the compiler's copies of a returned input (in8192_bs10: ~130 MB a core inside the
#: last stage's world all_reduce, slowing it and the hc_out after it)
#: the q norm on chip on both cores, staged straight into wq_b's GEMM (no qr round trip)
Q_NORM_ON_CHIP = True
CACHE_COPY = True
#: ... each cache s in two halves: in stage s's attention and world all_reduce shadows
#: (instead of whole caches in the next stages' attention all_reduces and the head)
CACHE_COPY_SPREAD = True
#: DMA QoS priority of the attention / world all_reduces (0 = the highest: they are on the
#: critical path, the weight prefetches / KV writes / cache copies in their shadow are not)
AR_PRIORITY = 0
#: the attention / world all_reduces carry the R live rows only (else the padded tile)
AR_LIVE_ROWS = True
#: hc_in_mix of both sublayers split: its sum of squares, mix GEMM and sigmoids on core 1
#: while core 0 runs the sublayer's hc_in_norm (core 1 idles there), only the Sinkhorn
#: (~200 dependent instructions) in the following all_reduce's shadow (the whole mix ran
#: past the attention all_reduce's end on core 1: on the critical path)
HC_MIX_SPLIT_ATTN = True
HC_MIX_SPLIT_FFN = True
#: the embedding all_reduce's result read on GpSimd after kv_write(0) (see the prologue)
E_LOAD_LATE = True
#: the stage tiles' pad rows (xn, xn2 + its zero tile, oq, idx) zeroed once in the prologue
PADS_ONCE = True
#: the MoE partial sum (eacc + eo32 per core, core 1 -> part1 -> core 0) with the rows' column
#: blocks spread over the partitions (mk_common.moe_partial)
PARTIAL_SPREAD = True
#: wexp's pad rows (rows R .. Rp + 128: never written by the top-k) zeroed once in the prologue
#: instead of every stage behind the top-k's store (they sat on the expert lists' chain), and
#: the routed accumulator's live rows zeroed at the start of the MoE section (off that chain)
WEXP_PADS_ONCE = True
EACC_ZERO_EARLY = True
#: hc_in_norm_spread's rows split over the cores: core 1's share of the rows in the ffn
#: sublayer (it also runs that sublayer's hc_in_mix there) and in the head (core 1 idle);
#: none in the attention sublayer (measured: core 1's hc_in_mix there plus a third of the
#: rows outlasted core 0's norm, ~+11 us a stage at in8192_bs10)
HC_NORM_CORE1_SHARE_ATTN = 0.0
HC_NORM_CORE1_SHARE_FFN = 1.0 / 3.0
HC_NORM_CORE1_SHARE_HEAD = 0.5


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
    meta, rows_c, main_hidden, row_pos, write_slots, block_table,
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
    phase="decode", scheme="tp32",
):
    """The drafter, traced with `phase` baked: "prefill" is the `at_zero` step (the
    main-stream KV writes only), "decode" the draft proper, for the `B = meta.shape[0]`
    sequences of the step. `mk_kernels` makes one `@nki.jit` entry per phase and scheme.
    The drafter's groups are the ranks the head and sampler run on: the world under
    tp32, the last stage under tp8pp8."""
    at_zero, K, win, stage, kver, noise_id, vocab_size, topk_e, bs = flags
    assert bool(at_zero) == (phase == "prefill"), f"flags at_zero={at_zero} vs phase {phase}"
    nstages = wq_a.shape[0]
    B = meta.shape[0]
    R = B * K                          # the draft rows: row b * K + i is sequence b's i-th
    Rp = PT
    assert R <= PT, f"{B} sequences x {K} draft rows exceed one tile"
    assert meta.shape[1] == 3 and rows_c.shape[0] == R and rows_c.shape[1] == 4
    assert win == PT, f"the window ({win}) is one tile of slots"
    L = main_hidden.shape[0]           # main-stream rows of the step, all sequences
    Lp = pad_to(L, PT)
    nblk = block_table.shape[0] // B
    assert nblk * B == block_table.shape[0]
    main_in = main_hidden.shape[1]
    dim = main_norm_w.shape[1]
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
    ntab = win + K                     # a sequence's slot-table rows: its window, then its draft block
    ntab_p = pad_to(ntab, PT)
    main_proj4 = main_proj.view(X.FP8_X4)
    wq_a4, wq_b4, wkv4 = wq_a.view(X.FP8_X4), wq_b.view(X.FP8_X4), wkv.view(X.FP8_X4)
    wo_a4, wo_b4 = wo_a.view(X.FP8_X4), wo_b.view(X.FP8_X4)
    sh_w1_4, sh_w3_4, sh_w2_4 = sh_w1.view(X.FP8_X4), sh_w3.view(X.FP8_X4), sh_w2.view(X.FP8_X4)
    w1_4, w3_4, w2_4 = w1.view(X.FP4_X4), w3.view(X.FP4_X4), w2.view(X.FP4_X4)
    part = head_w.shape[0]
    mrank = mk_embed_w.shape[1]
    world = len(groups_world[0])
    # main_proj column-sharded over the world (layout.prepare: each rank its dim / world output
    # columns): each rank projects its slice, an all_gather assembles main_x (exact: every
    # output element is the same dot product, computed once)
    mp_cols = main_proj.shape[1]
    mp_tp = mp_cols != dim
    assert not mp_tp or mp_cols * world == dim, f"main_proj {main_proj.shape} vs dim {dim}"
    # the shared expert's gate / up projections column-sharded over the world (layout.prepare:
    # each rank its inter / world columns): each rank projects + swiglus its slice, a world
    # all_gather assembles the hidden rows for the replicated down projection (exact: every
    # hidden element is the same dot product and elementwise chain, computed once)
    sh_cols = sh_w1.shape[2]
    sh_tp = sh_cols != inter
    assert not sh_tp or (sh_cols * world == inter and (inter // PT + 1) // 2 * PT % sh_cols == 0), \
        f"sh_w1 {sh_w1.shape} vs inter {inter}"
    W8 = max(world, 8)
    # the seeds' markov embeddings ride in the draft embedding's all_reduce (B extra rows)
    fold = mrank <= dim
    Re = R + B if fold else R
    GW = 2 + mrank                     # the argmax all_gather row: (max, index, markov embedding)
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
    kt = tiles_of(R, pid)             # the draft rows: one partial tile, core 0's
    kt_all = tiles_of(R, 0)
    ptiles = tiles_of(Rp, 0) + tiles_of(Rp, 1)
    lt = tiles_of(L, pid)
    ltp = tiles_of(Lp, 0) + tiles_of(Lp, 1)

    # ------------------------------------------------------------------ outputs
    output_ids = nl.ndarray((B, K + 1), dtype=nl.int32, buffer=nl.shared_hbm)
    confidence = nl.ndarray((R, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    dbg = nl.ndarray((Rp, max(width, part + dim)) if stage != 9 else (1, 1),
                     dtype=nl.float32, buffer=nl.shared_hbm)

    # ------------------------------------------------------------------ scratch
    sc = nl.ndarray((B, 2), dtype=nl.float32, buffer=nl.shared_hbm, name="sc")        # draft_pos, wv per sequence
    ids_hbm = nl.ndarray((Rp, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="ids")
    drow_pos = nl.ndarray((Rp, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="drow_pos")
    wvk = nl.ndarray((Rp, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="wvk")
    tbase = nl.ndarray((Rp, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="tbase")
    permc = nl.ndarray((Rp, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="permc")
    tkin = nl.ndarray((Rp, K), dtype=nl.int32, buffer=nl.shared_hbm, name="tkin")
    win_slots = nl.ndarray((B * win, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="win_slots")
    sid = nl.ndarray((Re, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="sid")
    mh16 = nl.ndarray((Lp, main_in), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="mh16")
    mxr = nl.ndarray((Lp, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="mxr")
    mx = nl.ndarray((Lp, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="mx")
    if mp_tp:
        mxs = nl.ndarray((L, mp_cols), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="mxs")
        mxg = nl.ndarray((world * L, mp_cols), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="mxg")
    if sh_tp:
        shs = nl.ndarray((R, sh_cols), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="shs")
        shg = nl.ndarray((world * R, sh_cols), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="shg")
    # One scratch pair per stage: at `at_zero` the three KV writes run back to back
    # with no other barrier between them, and a shared pair would let a core start
    # the next stage's projection while the other core still reads this one's.
    mkvr_s = [nl.ndarray((Lp, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name=f"mkvr{s}") for s in range(nstages)]
    mkv_s = [nl.ndarray((Lp, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name=f"mkv{s}") for s in range(nstages)]
    # the embedding all_reduce carries the live rows only (a sum of one nonzero row and
    # zeros: exact in any order); the attention / world all_reduces too with
    # `AR_LIVE_ROWS` (their fp32 summation order changes with the payload -- measured: 3..12
    # bf16 ulps of confidence, the drafts unchanged -- for R of the Rp rows' bytes), else
    # the padded tile, its pad rows zeroed once
    Rc = R if AR_LIVE_ROWS else Rp
    Rw = R if AR_LIVE_ROWS else Rp
    e_loc = nl.ndarray((Re, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="e_loc")
    e_sum = nl.ndarray((Re, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="e_sum")
    st = nl.ndarray((R, width), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="st")
    st2 = nl.ndarray((R, width), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="st2")
    st3 = nl.ndarray((R, width), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="st3")
    pre_mix = nl.ndarray((R, hc), dtype=nl.float32, buffer=nl.shared_hbm, name="pre_mix")
    pre_a = nl.ndarray((R, hc), dtype=nl.float32, buffer=nl.shared_hbm, name="pre_a")
    post_a = nl.ndarray((R, hc), dtype=nl.float32, buffer=nl.shared_hbm, name="post_a")
    comb_a = nl.ndarray((R, hc * hc), dtype=nl.float32, buffer=nl.shared_hbm, name="comb_a")
    pre_f = nl.ndarray((R, hc), dtype=nl.float32, buffer=nl.shared_hbm, name="pre_f")
    post_f = nl.ndarray((R, hc), dtype=nl.float32, buffer=nl.shared_hbm, name="post_f")
    comb_f = nl.ndarray((R, hc * hc), dtype=nl.float32, buffer=nl.shared_hbm, name="comb_f")
    xn = nl.ndarray((Rp, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="xn")
    ysc = nl.ndarray((R, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="ysc")     # hc_in_norm_split's exchange
    rsc = nl.ndarray((R, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="rsc")        # hc_in_norm_spread's rms
    rsch = nl.ndarray((R, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="rsch")      # ... the head's
    qa = nl.ndarray((Rp, q_lora), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="qa")
    qr = nl.ndarray((Rp, q_lora), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="qr")
    q = nl.ndarray((Rp, heads * hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="q")
    kvr = nl.ndarray((Rp, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kvr")
    kvd = nl.ndarray((Rp, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kvd")
    kvtab = nl.ndarray((B * ntab_p, hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kvtab")
    idx = nl.ndarray((Rp, kpad), dtype=nl.int32, buffer=nl.shared_hbm, name="idx")
    o = nl.ndarray((Rp, heads * hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="o")
    oq = nl.ndarray((Rp, heads * hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="oq")
    z = nl.ndarray((Rp, o_lora_total), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="z")
    cc_src = nl.ndarray((Rc, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((Rc, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_dst")
    # xn2 / wexp / eacc carry one extra zero tile for the routed experts' padding slots.
    xn2 = nl.ndarray((Rp + PT, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="xn2")
    logits = nl.ndarray((Rp, n_experts), dtype=nl.float32, buffer=nl.shared_hbm, name="logits")
    wexp = nl.ndarray((Rp + PT, n_local), dtype=nl.float32, buffer=nl.shared_hbm, name="wexp")
    eg = nl.ndarray((Rp, inter), dtype=nl.bfloat16, buffer=nl.private_hbm)
    eu = nl.ndarray((Rp, inter), dtype=nl.bfloat16, buffer=nl.private_hbm)
    ey = nl.ndarray((Rp, inter), dtype=nl.bfloat16, buffer=nl.private_hbm)
    eo32 = nl.ndarray((Rp, dim), dtype=nl.float32, buffer=nl.private_hbm)
    eacc = nl.ndarray((Rp + PT, dim), dtype=nl.float32, buffer=nl.private_hbm)
    part1 = nl.ndarray((R, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="part1")   # core 1's partial
    cc2_src = nl.ndarray((Rw, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc2_src")
    cc2_dst = nl.ndarray((Rw, dim), dtype=nl.float32, buffer=nl.shared_hbm, name="cc2_dst")
    xh = nl.ndarray((Rp, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="xh")       # collapsed, pre-norm
    xhn = nl.ndarray((Rp, dim), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="xhn")     # normed
    lg = nl.ndarray((Rp, part), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="lg")
    lg32 = nl.ndarray((Rp, part), dtype=nl.float32, buffer=nl.shared_hbm, name="lg32")
    if not fold:
        mid = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.shared_hbm, name="mid")
        mcc_src = nl.ndarray((B, mrank), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="mcc_src")
        mcc_dst = nl.ndarray((B, mrank), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="mcc_dst")
    g_src = nl.ndarray((B, GW), dtype=nl.float32, buffer=nl.shared_hbm, name="g_src")
    g_dst = nl.ndarray((world * B, GW), dtype=nl.float32, buffer=nl.shared_hbm, name="g_dst")
    memb = nl.ndarray((Rp, mrank), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="memb")   # row i * B + b
    mperm = nl.ndarray((Rp, mrank), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="mperm") # row b * K + i

    use_copy = CACHE_COPY and stage == 9 and not at_zero
    if use_copy:
        c_out = [nl.ndarray(c.shape, dtype=c.dtype, buffer=nl.shared_hbm) for c in caches]
    else:
        c_out = list(caches)

    def ret():
        return output_ids, confidence, dbg, c_out[0], c_out[1], c_out[2]

    def main_x():
        """main_x = main_norm(main_proj(main_hidden)) into `mx` (only the KV writes read it)."""
        # the taps' projection: replicated (~80 MB of weights, in the embedding all_reduce's
        # shadow) or column-sharded (MAIN_PROJ_TP: this rank's slice, then an all_gather)
        cast_rows(main_hidden, mh16, lt, main_in, nl.bfloat16)
        if pid == 0 and L < Lp:
            padm = nl.ndarray((PT, main_in), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(padm, 0.0)
            nisa.dma_copy(dst=mh16[L:Lp, :], src=padm[0 : Lp - L, :])
        nisa.core_barrier(data=mh16, cores=(0, 1))
        if mp_tp:
            if pid == 0:
                X.gemm_rows(mh16, Lp, main_in, main_proj4, main_proj_s, mxs, 0, mp_cols, m_out=L, m_live=L)
            nisa.core_barrier(data=mxs, cores=(0, 1))
            ncc.all_gather(srcs=[mxs], dsts=[mxg], replica_group=rg_world, collective_dim=0, priority=AR_PRIORITY)
        else:
            lo, hi = _half(dim, pid)
            X.gemm_rows(mh16, Lp, main_in, main_proj4, main_proj_s, mxr, lo, hi, m_live=L)

    def main_x_finish():
        if mp_tp and pid == 0:
            # the gathered slices [position, row, column] -> main_x rows (position p holds
            # the columns of rank groups_world[0][p])
            gw = [int(r) for r in groups_world[0]]
            if gw == list(range(world)):
                nisa.dma_copy(dst=mxr.ap(pattern=[[mp_cols, world], [dim, L], [1, mp_cols]], offset=0),
                              src=mxg.ap(pattern=[[L * mp_cols, world], [mp_cols, L], [1, mp_cols]], offset=0))
            else:
                for q, r in enumerate(gw):
                    nisa.dma_copy(dst=mxr[0:L, r * mp_cols : (r + 1) * mp_cols], src=mxg[q * L : (q + 1) * L, 0:mp_cols])
        nisa.core_barrier(data=mxr, cores=(0, 1))
        norm_rows(mxr, main_norm_w, mx, lt, dim, qdq=False, chunk=1024)
        if pid == 0 and L < Lp:
            padx = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(padx, 0.0)
            nisa.dma_copy(dst=mx[L:Lp, :], src=padx[0 : Lp - L, :])
        nisa.core_barrier(data=mx, cores=(0, 1))

    # MAIN_PROJ_TP: the slice projection and its all_gather FIRST (they need main_hidden only):
    # behind the embedding all_reduce, the all_gather's trigger waited for the all_reduce to
    # complete and held the DMA queues behind it (measured: weight DMAs of both cores stalled
    # until the all_reduce ended)
    if mp_tp:
        main_x()
    # ------------------------------------------------------------------ 0. the draft's indices
    # `meta [B, 3]` = (seed, tap, pos) per sequence, data dependent (the tap comes out
    # of the rejection sampler, on the device); `rows_c [R, 4]` = (i, b, b * ntab_p,
    # i * B + b) per draft row, host constants. draft_pos = pos + tap; a row's position
    # is draft_pos + 1 + i, its window validity base draft_pos - (win - 1), its id the
    # seed for i = 0 and noise after. Pad rows: noise, no window, base 0.
    if pid == 0:
        mt = nl.ndarray((B, 3), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=mt, src=meta[0:B, 0:3])
        mtf = nl.ndarray((B, 3), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mtf, src=mt)
        dpwv = nl.ndarray((B, 2), dtype=nl.float32, buffer=nl.sbuf)       # draft_pos, wv per sequence
        nisa.tensor_tensor(dst=dpwv[0:B, 0:1], data1=mtf[0:B, 1:2], data2=mtf[0:B, 2:3], op=nl.add)
        nisa.tensor_scalar(dst=dpwv[0:B, 1:2], data=dpwv[0:B, 0:1], op0=nl.add, operand0=float(-(win - 1)))
        nisa.dma_copy(dst=sc[0:B, 0:2], src=dpwv)
        for r0, nr in kt_all:
            rc = nl.ndarray((PT, 4), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rc[0:nr, :], src=rows_c[r0 : r0 + nr, 0:4])
            # the row's sequence's meta row, gathered by b
            mr = nl.ndarray((PT, 3), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=mr[0:nr, :], src=meta.ap(pattern=[[3, nr], [1, 3]],
                                                         vector_offset=rc[0:nr, 1:2], indirect_dim=0))
            mrf = nl.ndarray((PT, 3), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=mrf[0:nr, :], src=mr[0:nr, :])
            rcf = nl.ndarray((PT, 4), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=rcf[0:nr, :], src=rc[0:nr, :])
            dp = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=dp[0:nr, :], data1=mrf[0:nr, 1:2], data2=mrf[0:nr, 2:3], op=nl.add)
            col = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            coli = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
            # position = draft_pos + 1 + i
            nisa.tensor_tensor(dst=col[0:nr, :], data1=dp[0:nr, :], data2=rcf[0:nr, 0:1], op=nl.add)
            nisa.tensor_scalar(dst=col[0:nr, :], data=col[0:nr, :], op0=nl.add, operand0=1.0)
            nisa.tensor_copy(dst=coli[0:nr, :], src=col[0:nr, :])
            nisa.dma_copy(dst=drow_pos[r0 : r0 + nr, 0:1], src=coli[0:nr, :])
            # window validity base = draft_pos - (win - 1)
            nisa.tensor_scalar(dst=col[0:nr, :], data=dp[0:nr, :], op0=nl.add, operand0=float(-(win - 1)))
            nisa.tensor_copy(dst=coli[0:nr, :], src=col[0:nr, :])
            nisa.dma_copy(dst=wvk[r0 : r0 + nr, 0:1], src=coli[0:nr, :])
            # id = seed * (i == 0) + noise * (i != 0)   (exact in fp32: ids, positions < 2^24)
            is0 = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)      # 1 where i == 0
            nisa.tensor_scalar(dst=is0[0:nr, :], data=rcf[0:nr, 0:1], op0=nl.less, operand0=1.0)
            nisa.tensor_tensor(dst=col[0:nr, :], data1=mrf[0:nr, 0:1], data2=is0[0:nr, :], op=nl.multiply)
            noi = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)      # noise where i != 0
            nisa.tensor_scalar(dst=noi[0:nr, :], data=is0[0:nr, :], op0=nl.multiply, operand0=-float(noise_id),
                               op1=nl.add, operand1=float(noise_id))
            nisa.tensor_tensor(dst=col[0:nr, :], data1=col[0:nr, :], data2=noi[0:nr, :], op=nl.add)
            nisa.tensor_copy(dst=coli[0:nr, :], src=col[0:nr, :])
            nisa.dma_copy(dst=ids_hbm[r0 : r0 + nr, 0:1], src=coli[0:nr, :])
            nisa.dma_copy(dst=tbase[r0 : r0 + nr, 0:1], src=rc[0:nr, 2:3])
            nisa.dma_copy(dst=permc[r0 : r0 + nr, 0:1], src=rc[0:nr, 3:4])
        if R < Rp:
            padc = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.memset(padc, 0)
            nisa.dma_copy(dst=drow_pos[R:Rp, 0:1], src=padc[0 : Rp - R, :])
            nisa.dma_copy(dst=tbase[R:Rp, 0:1], src=padc[0 : Rp - R, :])
            nisa.dma_copy(dst=permc[R:Rp, 0:1], src=padc[0 : Rp - R, :])
            nisa.memset(padc, -win)
            nisa.dma_copy(dst=wvk[R:Rp, 0:1], src=padc[0 : Rp - R, :])
            nisa.memset(padc, noise_id)
            nisa.dma_copy(dst=ids_hbm[R:Rp, 0:1], src=padc[0 : Rp - R, :])
        tk = nl.ndarray((PT, K), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=tk, pattern=[[1, K]], offset=win, channel_multiplier=0)
        nisa.dma_copy(dst=tkin[0:Rp, 0:K], src=tk)
        if fold:
            nisa.dma_copy(dst=sid[R:Re, 0:1], src=mt[0:B, 0:1])
    # ------------------------------------------------------------------ 1. the draft embedding first
    # (its all_reduce needs only the ids: the window / row tables follow in its shadow)
    # This rank's embedding rows of [seed, noise, ...] (zero elsewhere), then the world sum.
    if pid == 0:
        padz = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(padz, 0.0)
        nisa.dma_copy(dst=e_loc[0:Re, :], src=padz[0:Re, :])
        engram_gather(ids_hbm, embed_w, vocab_start, e_loc, kt_all, 1, dim, part)
        if R < Rw:
            padc = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(padc, 0.0)
            nisa.dma_copy(dst=cc_src[R:Rc, :], src=padc[0 : Rc - R, :])
            nisa.dma_copy(dst=cc2_src[R:Rw, :], src=padc[0 : Rw - R, :])
        if fold:
            engram_gather(sid, mk_embed_w, vocab_start, e_loc, [(R, B)], 1, mrank, part)
    nisa.core_barrier(data=e_loc, cores=(0, 1))
    ncc.all_reduce(srcs=[e_loc], dsts=[e_sum], replica_group=rg_world, op=nl.add, priority=AR_PRIORITY)
    if pid == 0 and PADS_ONCE and not at_zero:
        # the pad rows of the stages' row tiles (never written otherwise) zeroed once, in
        # the all_reduce's shadow (per stage they sat on the DMA queues behind waits)
        pz = nl.ndarray((PT, max(dim, heads * hd)), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(pz, 0.0)
        if R < Rp:
            nisa.dma_copy(dst=xn[R:Rp, :], src=pz[0 : Rp - R, 0:dim])
            nisa.dma_copy(dst=xn2[R:Rp, :], src=pz[0 : Rp - R, 0:dim])
            nisa.dma_copy(dst=oq[R:Rp, :], src=pz[0 : Rp - R, 0 : heads * hd])
            pzi = nl.ndarray((PT, kpad), dtype=nl.int32, buffer=nl.sbuf)
            nisa.memset(pzi, 0)
            nisa.dma_copy(dst=idx[R:Rp, :], src=pzi[0 : Rp - R, :])
        nisa.dma_copy(dst=xn2[Rp : Rp + PT, :], src=pz[:, 0:dim])
    if WEXP_PADS_ONCE and not at_zero:
        # (both cores, as the per-stage writes did: each core's own write precedes its reads)
        zw0 = nl.ndarray((PT, n_local), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(zw0, 0.0)
        if R < Rp:
            nisa.dma_copy(dst=wexp[R:Rp, :], src=zw0[0 : Rp - R, :])
        nisa.dma_copy(dst=wexp[Rp : Rp + PT, :], src=zw0)
    if sh_tp and not at_zero:
        # the shared expert's gathered hidden rows: the staged pad rows (R .. pad32(R)) of
        # each core's private copy zeroed once (the gather writes rows < R only)
        mr_sh = min(Rp, pad_to(R, 32))
        if R < mr_sh:
            pze = nl.ndarray((PT, inter), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(pze, 0.0)
            nisa.dma_copy(dst=ey[R:mr_sh, :], src=pze[0 : mr_sh - R, :])
    nisa.core_barrier(data=sc, cores=(0, 1))
    # the window's cache slots, a sequence per core in turn:
    # p = max(wv + j, 0); block_table[b][p >> log2(bs)] * bs + (p & (bs - 1))
    ramp = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=ramp, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    rampf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rampf, src=ramp)
    for b in range(B):
        if b % CORES != pid:
            continue
        wvb = bcast_row(sc[b : b + 1, 1:2], 1, PT)
        wp = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=wp, data=rampf, op0=nl.add, operand0=wvb[0:PT, 0:1],
                           op1=nl.maximum, operand1=0.0)
        wpi = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=wpi, src=wp)
        blki = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        blki[0:PT, :] = nl.right_shift(wpi[0:PT, :], bs_log2)
        offi = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        offi[0:PT, :] = nl.bitwise_and(wpi[0:PT, :], bs - 1)
        # the sequence's block-table row: block index + b * nblk (through fp32, exact)
        blkf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=blkf, src=blki)
        nisa.tensor_scalar(dst=blkf, data=blkf, op0=nl.add, operand0=float(b * nblk))
        blkb = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=blkb, src=blkf)
        bt = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=bt, src=block_table.ap(pattern=[[1, PT], [1, 1]],
                                                  vector_offset=blkb[0:PT, 0:1], indirect_dim=0))
        btf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=btf, src=bt)
        offf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=offf, src=offi)
        slot = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=slot, data=btf, op0=nl.multiply, operand0=float(bs))
        nisa.tensor_tensor(dst=slot, data1=slot, data2=offf, op=nl.add)
        sloti = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sloti, src=slot)
        nisa.dma_copy(dst=win_slots[b * win : (b + 1) * win, 0:1], src=sloti)
    nisa.core_barrier(data=drow_pos, cores=(0, 1))
    nisa.core_barrier(data=wvk, cores=(0, 1))
    nisa.core_barrier(data=tbase, cores=(0, 1))
    nisa.core_barrier(data=permc, cores=(0, 1))
    nisa.core_barrier(data=tkin, cores=(0, 1))
    nisa.core_barrier(data=ids_hbm, cores=(0, 1))
    nisa.core_barrier(data=win_slots, cores=(0, 1))

    # ------------------------------------------------------------------ 1. main_x and the draft embedding

    def kv_write(s, after=None):
        """The main stream's KV for this step into stage `s`'s window cache: it needs
        only `mx`, so it runs in the shadow of the collective before the stage."""
        cache = caches[s]
        mkvr, mkv = mkvr_s[s], mkv_s[s]
        lo, hi = _half(hd, pid)
        X.gemm_rows(mx, Lp, dim, wkv4[s], wkv_s[s], mkvr, lo, hi, m_live=L, after=after)
        nisa.core_barrier(data=mkvr, cores=(0, 1))
        norm_rows(mkvr, kv_norm_w[s : s + 1], mkv, lt, hd, qdq=False, chunk=hd)
        rope_rows(mkv, mkv, lt, hd, hd, 1, freqs, row_pos, inverse=False, qdq=True)
        scatter_rows(mkv, write_slots, cache, lt, hd)
        return nisa.core_barrier(data=cache, cores=(0, 1))

    if not mp_tp:
        main_x()
    main_x_finish()
    bar_kv0 = kv_write(0) if not at_zero and nstages > 0 else None
    # The markov head transposed once, (PT, mrank // PT, part) bf16: weights only, so in
    # the embedding all_reduce's shadow, long before the head needs it.
    mkT = transpose_pe(mk_head_w, part, mrank)
    if pid == 0:
        e = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        # it waits for the embedding all_reduce: issued from GpSimd after kv_write(0)'s last
        # barrier there (on the Sync queue it held up main_x's norm and kv_write(0) until
        # the all_reduce ended)
        ins = nisa.dma_copy(dst=e[0:R, :], src=e_sum[0:R, :],
                            dge_mode=nisa.dge_mode.swdge if E_LOAD_LATE else nisa.dge_mode.unknown)
        if E_LOAD_LATE:
            sched_after(ins, bar_kv0)
        for c in range(hc):
            nisa.dma_copy(dst=st[0:R, c * dim : (c + 1) * dim], src=e[0:R, :],
                          dge_mode=nisa.dge_mode.swdge if E_LOAD_LATE else nisa.dge_mode.unknown)
        pm = nl.ndarray((PT, hc), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(pm, 0.0)
        nisa.memset(pm[:, 0:1], 1.0)
        nisa.dma_copy(dst=pre_mix[0:R, :], src=pm[0:R, :])
    nisa.core_barrier(data=st, cores=(0, 1))
    nisa.core_barrier(data=pre_mix, cores=(0, 1))

    def norm_split(x2, pm, w, normed, ys, rs, share1):
        """hc_in_norm_spread of the R rows, core 1 the last `share1` of them (the same
        per-row ops on either core)."""
        r1 = min(R - 1, int(R * share1))
        r0n = R - r1
        if pid == 0:
            hc_in_norm_spread(r0n, x2, pm, w, hc, dim, normed, ys, rs)
        elif r1 > 0:
            hc_in_norm_spread(r1, x2, pm, w, hc, dim, normed, ys, rs, r0=r0n)

    cur_st, cur_pre = st, pre_mix
    bar_out = None
    for s in range(nstages):
        cache = caches[s]
        if at_zero:
            kv_write(s)
            continue
        nxt_st = st2 if s % 2 == 0 else st3
        # ---- hc_in (attention): the normed input now, on core 0; its coefficients
        # (pre_a / post_a / comb_a) on core 1 in the attention all_reduce's shadow
        if HC_NORM_SPREAD:
            norm_split(cur_st, cur_pre, attn_norm_w[s : s + 1], xn, ysc, rsc, HC_NORM_CORE1_SHARE_ATTN)
        elif HC_NORM_SPLIT:
            for r0, nr in kt_all:
                hc_in_norm_split(r0, nr, cur_st, cur_pre, attn_norm_w[s : s + 1], hc, dim, xn, ysc, pid)
        else:
            for r0, nr in kt:
                hc_in_norm(r0, nr, cur_st, cur_pre, attn_norm_w[s : s + 1], hc, dim, xn)
        if pid == 0 and not PADS_ONCE:
            padn = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(padn, 0.0)
            nisa.dma_copy(dst=xn[R:Rp, :], src=padn[0 : Rp - R, :])
        if pid == CORES - 1 and HC_MIX_SPLIT_ATTN:
            # the attention coefficients up to the Sinkhorn now, on the core hc_in_norm
            # leaves idle (read only after the attention all_reduce)
            fnT = load_fn_t(hca_fn_ts[s], width, mix)
            t2a = [hc_in_mix(r0, nr, cur_st, fnT, hca_base[s : s + 1], hca_scale[s : s + 1], hc, dim, mix,
                             pre_a, post_a, comb_a, sink=False) for r0, nr in kt_all]
        nisa.core_barrier(data=xn, cores=(0, 1))
        # ---- projections of the draft rows
        if s == 0 and stage == 11:
            cast_rows(xn, dbg[0:Rp, 0:dim], kt, dim, nl.float32)
            nisa.core_barrier(data=dbg, cores=(0, 1))
            return ret()
        lo, hi = _half(q_lora, pid)
        lo2, hi2 = _half(hd, pid)
        X.gemm_rows_shared(xn, Rp, dim, [(wq_a4[s], wq_a_s[s], qa, lo, hi), (wkv4[s], wkv_s[s], kvr, lo2, hi2)],
                           m_live=R)
        nisa.core_barrier(data=qa, cores=(0, 1))
        nisa.core_barrier(data=kvr, cores=(0, 1))
        if Q_NORM_ON_CHIP:
            # each core normalizes the q rows itself and stages them for wq_b on chip (no qr
            # HBM round trip / barrier; the same per-row ops)
            qn = norm_rows_sb(qa, q_norm_w[s : s + 1], R, q_lora, min(PT, ((R + 31) // 32) * 32),
                              chunk=q_lora)
        else:
            norm_rows(qa, q_norm_w[s : s + 1], qr, kt, q_lora, qdq=False, chunk=q_lora)
        norm_rows(kvr, kv_norm_w[s : s + 1], kvd, kt, hd, qdq=False, chunk=hd)
        rope_rows(kvd, kvd, kt, hd, hd, 1, freqs, drow_pos, inverse=False, qdq=True)
        if not Q_NORM_ON_CHIP:
            if pid == 0 and R < Rp:
                padq = nl.ndarray((PT, q_lora), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.memset(padq, 0.0)
                nisa.dma_copy(dst=qr[R:Rp, :], src=padq[0 : Rp - R, :])
            nisa.core_barrier(data=qr, cores=(0, 1))
            nisa.core_barrier(data=kvd, cores=(0, 1))
        lo, hi = _half(heads * hd, pid)
        if Q_NORM_ON_CHIP:
            X.gemm_rows_sb(qn, q_lora, wq_b4[s], wq_b_s[s], q, lo, hi, m_live=R)
            nisa.core_barrier(data=kvd, cores=(0, 1))
        else:
            X.gemm_rows(qr, Rp, q_lora, wq_b4[s], wq_b_s[s], q, lo, hi, m_live=R)
        nisa.core_barrier(data=q, cores=(0, 1))
        rope_rows(q, q, kt, heads * hd, hd, heads, freqs, drow_pos, inverse=False)
        if s == 0 and stage == 12:
            cast_rows(q, dbg[0:Rp, 0 : heads * hd], kt, heads * hd, nl.float32)
            nisa.core_barrier(data=dbg, cores=(0, 1))
            return ret()
        if s == 0 and stage == 13:
            cast_rows(kvd, dbg[0:Rp, 0:hd], kt, hd, nl.float32)
            nisa.core_barrier(data=dbg, cores=(0, 1))
            return ret()
        # ---- the slot table, a sequence at a time: its window, then its draft block's own rows
        for b in range(B):
            if b % CORES != pid:
                continue
            gather_rows(cache, win_slots[b * win : (b + 1) * win, :], kvtab[b * ntab_p : b * ntab_p + win, :],
                        tiles_of(win, 0), hd)
            dk = nl.ndarray((PT, hd), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(dk, 0.0)
            nisa.dma_copy(dst=dk[0:K, :], src=kvd[b * K : b * K + K, :])
            nisa.dma_copy(dst=kvtab[b * ntab_p + win : (b + 1) * ntab_p, :], src=dk[0 : ntab_p - win, :])
        if pid == 0:
            attn_index_table(idx, kt_all, tbase, wvk, win, tbase, tkin, K, kpad)
            if R < Rp and not PADS_ONCE:
                padi = nl.ndarray((PT, kpad), dtype=nl.int32, buffer=nl.sbuf)
                nisa.memset(padi, 0)
                nisa.dma_copy(dst=idx[R:Rp, :], src=padi[0 : Rp - R, :])
        nisa.core_barrier(data=q, cores=(0, 1))
        nisa.core_barrier(data=kvtab, cores=(0, 1))
        nisa.core_barrier(data=idx, cores=(0, 1))
        # ---- attention over the draft rows: dense per sequence, (token, head) pairs as rows
        aunits = attn_units(B, K, heads, pid)
        if aunits:
            sinkr = attn_sink_rows(attn_sink[s], heads, max(nt for _, _, nt in aunits))
        for b, t0, nt in aunits:
            attn_rows_dense(q, kvtab, idx, o, sinkr, heads, hd, kpad, scale, b * K + t0, nt, b * ntab_p)
        nisa.core_barrier(data=o, cores=(0, 1))
        if s == 0 and stage == 14:
            cast_rows(o, dbg[0:Rp, 0 : heads * hd], kt, heads * hd, nl.float32)
            nisa.core_barrier(data=dbg, cores=(0, 1))
            return ret()
        rope_rows(o, oq, kt, heads * hd, hd, heads, freqs, drow_pos, inverse=True, qdq=False)
        if pid == 0 and R < Rp and not PADS_ONCE:
            pado = nl.ndarray((PT, heads * hd), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(pado, 0.0)
            nisa.dma_copy(dst=oq[R:Rp, :], src=pado[0 : Rp - R, :])
        nisa.core_barrier(data=oq, cores=(0, 1))
        for g in range(ngroups):
            lo, hi = _half(o_lora, pid)
            X.gemm_rows(oq, Rp, g_in, wo_a4[s], wo_a_s[s], z, g * o_lora + lo, g * o_lora + hi, x_col0=g * g_in, m_live=R)
        nisa.core_barrier(data=z, cores=(0, 1))
        lo, hi = _half(dim, pid)
        X.gemm_rows(z, Rp, o_lora_total, wo_b4[s], wo_b_s[s], cc_src, lo, hi, out_dtype=nl.float32, m_out=R, m_live=R)
        bar_cc = nisa.core_barrier(data=cc_src, cores=(0, 1))
        if use_copy and CACHE_COPY_SPREAD:
            # the first half of cache s (written before the stage's attention) in this
            # all_reduce's shadow, the second in the world all_reduce's
            copy_rows_hw(caches[s], c_out[s], pid, after=bar_cc, part=(0, 2))
        elif use_copy and 0 < s and s - 1 < nstages - 1:
            # cache s - 1 (written in stage s - 1's prework) in this all_reduce's shadow
            copy_rows_hw(caches[s - 1], c_out[s - 1], pid, after=bar_cc)
        if s == 0 and stage == 15:
            cast_rows(cc_src, dbg[0:Rp, 0:dim], kt, dim, nl.float32)
            cast_rows(xn, dbg[0:Rp, dim : 2 * dim], kt, dim, nl.float32)
            nisa.core_barrier(data=dbg, cores=(0, 1))
            return ret()
        ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], replica_group=rg_attn, op=nl.add, priority=AR_PRIORITY)
        if pid == CORES - 1 and HC_MIX_SPLIT_ATTN:
            for (r0, nr), t2 in zip(kt_all, t2a):
                hc_in_mix_sink(r0, nr, t2, hc, comb_a, after=bar_cc)
        elif pid == CORES - 1:
            fnT = load_fn_t(hca_fn_ts[s], width, mix)
            for r0, nr in kt_all:
                hc_in_mix(r0, nr, cur_st, fnT, hca_base[s : s + 1], hca_scale[s : s + 1], hc, dim, mix,
                          pre_a, post_a, comb_a)
        nisa.core_barrier(data=pre_a, cores=(0, 1))
        nisa.core_barrier(data=post_a, cores=(0, 1))
        nisa.core_barrier(data=comb_a, cores=(0, 1))
        # ---- hc_out (both cores, a column half each; the fp32 sum cast to bf16 on load), hc_in (ffn)
        if HC_OUT_SPREAD:
            hc_out_spread(R, cc_dst, cur_st, post_a, comb_a, nxt_st, hc, dim, pid)
        else:
            hc_out_tile(0, R, cc_dst, cur_st, post_a, comb_a, nxt_st, hc, dim, chunks=hc_out_chunks(dim, pid))
        nisa.core_barrier(data=nxt_st, cores=(0, 1))
        if s == 0 and stage == 16:
            cast_rows(nxt_st, dbg[0:R, 0:width], kt, width, nl.float32)
            nisa.core_barrier(data=dbg, cores=(0, 1))
            return ret()
        # ---- hc_in (ffn): the normed input now, on core 0; its coefficients on core 1
        # in the world all_reduce's shadow
        if HC_NORM_SPREAD:
            norm_split(nxt_st, pre_a, ffn_norm_w[s : s + 1], xn2, ysc, rsc, HC_NORM_CORE1_SHARE_FFN)
        else:
            for r0, nr in kt:
                hc_in_norm(r0, nr, nxt_st, pre_a, ffn_norm_w[s : s + 1], hc, dim, xn2)
        if pid == CORES - 1 and HC_MIX_SPLIT_FFN:
            fnT2 = load_fn_t(hcf_fn_ts[s], width, mix)
            t2f = [hc_in_mix(r0, nr, nxt_st, fnT2, hcf_base[s : s + 1], hcf_scale[s : s + 1], hc, dim, mix,
                             pre_f, post_f, comb_f, sink=False) for r0, nr in kt_all]
        elif pid == CORES - 1 and HC_MIX_EARLY:
            # the ffn coefficients now, on the core hc_in_norm leaves idle (they are only
            # read by hc_out after the world all_reduce)
            fnT2 = load_fn_t(hcf_fn_ts[s], width, mix)
            for r0, nr in kt_all:
                hc_in_mix(r0, nr, nxt_st, fnT2, hcf_base[s : s + 1], hcf_scale[s : s + 1], hc, dim, mix,
                          pre_f, post_f, comb_f)
        if pid == 0 and not PADS_ONCE:
            padn2 = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(padn2, 0.0)
            if R < Rp:
                nisa.dma_copy(dst=xn2[R:Rp, :], src=padn2[0 : Rp - R, :])
            nisa.dma_copy(dst=xn2[Rp : Rp + PT, :], src=padn2)
        nisa.core_barrier(data=xn2, cores=(0, 1))
        # ---- MoE: n_experts routed, top-k, plus the shared expert
        if s == 0 and stage == 17:
            cast_rows(xn2, dbg[0:Rp, 0:dim], kt, dim, nl.float32)
            nisa.core_barrier(data=dbg, cores=(0, 1))
            return ret()
        # the shared expert first (it needs no routing): its weight streaming overlaps the
        # router's fp32 GEMM and the expert lists
        if EACC_ZERO_EARLY:
            mk_moe.zero_live(eacc, R, dim)
        lo, hi = _half(inter, pid)
        if sh_tp:
            # SH_TP: this rank's gate / up slice (core 0, on chip), then its world all_gather
            # (in the router's shadow); the down projection after the top-k reads the rows
            if pid == 0:
                mk_moe.shared_slice(xn2, dim, R, sh_w1_4[s], sh_s1[s], sh_w3_4[s], sh_s3[s], sh_cols,
                                    SWIGLU_LIMIT, shs)
            nisa.core_barrier(data=shs, cores=(0, 1))
            ncc.all_gather(srcs=[shs], dsts=[shg], replica_group=rg_world, collective_dim=0, priority=AR_PRIORITY)
        else:
            X.gemm_rows_shared(xn2, Rp, dim, [(sh_w1_4[s], sh_s1[s], eg, lo, hi), (sh_w3_4[s], sh_s3[s], eu, lo, hi)],
                               m_live=R)
            swiglu_rows(eg[0:Rp, lo:hi], eu[0:Rp, lo:hi], None, 0, ey[0:Rp, lo:hi], ptiles, hi - lo, SWIGLU_LIMIT)
            # The shared expert's weights are replicated on every rank and the model adds its
            # output ONCE, after the expert all-reduce. Every rank adds it to its partial here,
            # scaled by 1/world, so the world sum carries it exactly once (a power-of-two
            # scale of the fp32 partial is exact).
            X.gemm_rows(ey, Rp, hi - lo, sh_w2_4[s], sh_s2[s], eo32, 0, dim, x_col0=lo, krow0=lo // 4,
                        out_dtype=nl.float32, out_scale=1.0 / world, m_out=R, m_live=R)
        # the router logits, the expert columns split between the cores
        ce0 = n_experts // CORES
        gemm_fp32(xn2, gate_w_ts[s], logits, kt_all, dim, ce0 if pid == 0 else n_experts - ce0,
                  c0=0 if pid == 0 else ce0)
        bar_lg = nisa.core_barrier(data=logits, cores=(0, 1))
        route_topk(logits, gate_b[s : s + 1], expert_lo, kt_all, n_experts, topk_e, n_local, ROUTE_SCALE, wexp)
        if not WEXP_PADS_ONCE:
            zw = nl.ndarray((PT, n_local), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(zw, 0.0)
            if R < Rp:
                nisa.dma_copy(dst=wexp[R:Rp, :], src=zw[0 : Rp - R, :])
            nisa.dma_copy(dst=wexp[Rp : Rp + PT, :], src=zw)
        if sh_tp:
            # this core's contraction half of the gathered hidden rows, then the (replicated)
            # down projection as before, scaled by 1/world
            sched_after(mk_moe.shared_gather(shg, ey, R, sh_cols, lo, hi, [int(r) for r in groups_world[0]],
                                             inter), bar_lg)
            X.gemm_rows(ey, Rp, hi - lo, sh_w2_4[s], sh_s2[s], eo32, 0, dim, x_col0=lo, krow0=lo // 4,
                        out_dtype=nl.float32, out_scale=1.0 / world, m_out=R, m_live=R)
        # routed experts over the rows that picked them, into the zeroed accumulator
        lo, hi = _half(inter, pid)
        mk_moe.routed_experts(pid, n_local, Rp, xn2, wexp, w1_4[s], s1[s], w3_4[s], s3[s],
                              w2_4[s], s2[s], dim, inter, lo, hi, SWIGLU_LIMIT, eacc, live=R,
                              zeroed=EACC_ZERO_EARLY)
        # this core's partial (routed + shared) of the live rows, kept on chip; core 1's
        # goes through HBM to core 0, which adds the two (core 0's + core 1's, as before)
        if PARTIAL_SPREAD:
            bar_w = moe_partial(pid, R, dim, eacc, eo32, part1, cc2_src)
        else:
            pa = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=pa[0:R, :], src=eacc[0:R, :])
            pe = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=pe[0:R, :], src=eo32[0:R, :])
            nisa.tensor_tensor(dst=pa[0:R, :], data1=pa[0:R, :], data2=pe[0:R, :], op=nl.add)
            if pid == 1:
                nisa.dma_copy(dst=part1[0:R, :], src=pa[0:R, :])
            nisa.core_barrier(data=part1, cores=(0, 1))
            if pid == 0:
                pb = nl.ndarray((PT, dim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=pb[0:R, :], src=part1[0:R, :])
                nisa.tensor_tensor(dst=pa[0:R, :], data1=pa[0:R, :], data2=pb[0:R, :], op=nl.add)
                nisa.dma_copy(dst=cc2_src[0:R, :], src=pa[0:R, :])
            bar_w = nisa.core_barrier(data=cc2_src, cores=(0, 1))
        if use_copy and CACHE_COPY_SPREAD:
            copy_rows_hw(caches[s], c_out[s], pid, after=bar_w, part=(1, 2))
        if s == 0 and stage == 18:
            cast_rows(cc2_src, dbg[0:Rp, 0:dim], kt, dim, nl.float32)
            cast_rows(xn2, dbg[0:Rp, dim : 2 * dim], kt, dim, nl.float32)
            nisa.core_barrier(data=dbg, cores=(0, 1))
            return ret()
        ncc.all_reduce(srcs=[cc2_src], dsts=[cc2_dst], replica_group=rg_world, op=nl.add, priority=AR_PRIORITY)
        if pid == CORES - 1 and HC_MIX_SPLIT_FFN:
            for (r0, nr), t2 in zip(kt_all, t2f):
                hc_in_mix_sink(r0, nr, t2, hc, comb_f, after=bar_w)
        if s + 1 < nstages:
            kv_write(s + 1, after=bar_w)   # the next stage's, in this all_reduce's shadow
        if pid == CORES - 1 and not HC_MIX_EARLY and not HC_MIX_SPLIT_FFN:
            fnT2 = load_fn_t(hcf_fn_ts[s], width, mix)
            for r0, nr in kt_all:
                hc_in_mix(r0, nr, nxt_st, fnT2, hcf_base[s : s + 1], hcf_scale[s : s + 1], hc, dim, mix,
                          pre_f, post_f, comb_f)
        nisa.core_barrier(data=pre_f, cores=(0, 1))
        nisa.core_barrier(data=post_f, cores=(0, 1))
        nisa.core_barrier(data=comb_f, cores=(0, 1))
        out_st = st if s % 2 == 0 else st2
        if HC_OUT_SPREAD:
            hc_out_spread(R, cc2_dst, nxt_st, post_f, comb_f, out_st, hc, dim, pid)
        else:
            hc_out_tile(0, R, cc2_dst, nxt_st, post_f, comb_f, out_st, hc, dim, chunks=hc_out_chunks(dim, pid))
        bar_out = nisa.core_barrier(data=out_st, cores=(0, 1))
        cur_st, cur_pre = out_st, pre_f
        if stage == s + 1:
            cast_rows(cur_st, dbg[0:R, 0:width], kt, width, nl.float32)
            nisa.core_barrier(data=dbg, cores=(0, 1))
            return ret()

    if use_copy:
        # the last stage's cache (and any not copied yet) in the head (DMA-idle: the token loop)
        for c in range(nstages if CACHE_COPY_SPREAD else max(0, nstages - 1), len(caches)):
            copy_rows_hw(caches[c], c_out[c], pid, after=bar_out)
    # ------------------------------------------------------------------ head
    # Collapse with the last pre_mix (fp32 sum, bf16 out), then the final norm.
    if HC_NORM_SPREAD:
        # the same ops as hc_in_norm (xh = the collapsed rows, the norm's input)
        norm_split(cur_st, cur_pre, fnorm_w, xhn, xh, rsch, HC_NORM_CORE1_SHARE_HEAD)
    else:
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
                nisa.dma_copy(dst=xh[0:Rp, :], src=y16)
        nisa.core_barrier(data=xh, cores=(0, 1))
        norm_rows(xh, fnorm_w, xhn, kt, dim, qdq=False, chunk=1024)
    if pid == 0 and R < Rp:
        padh = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(padh, 0.0)
        nisa.dma_copy(dst=xhn[R:Rp, :], src=padh[0 : Rp - R, :])
    nisa.core_barrier(data=xhn, cores=(0, 1))
    lo, hi = _half(part, pid)
    bf16_gemm_rows(xhn, head_w, lg, Rp, dim, part, lo, hi)
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
        cast_rows(lg32, dbg[0:Rp, 0:part], kt, part, nl.float32)
        cast_rows(xh, dbg[0:Rp, part : part + dim], kt, dim, nl.float32)
        nisa.core_barrier(data=dbg, cores=(0, 1))
        return ret()

    vsB = bcast_row(vocab_start[0:1, 0:1], 1, B)
    # the drafted ids, a sequence per partition: column 0 is the seed
    ids_sb = nl.ndarray((B, K + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ids_sb[0:B, 0:1], src=meta[0:B, 0:1])
    # The markov embeddings of the seeds came with the draft embedding's all_reduce; those
    # of token i + 1 come with token i's argmax all_gather: every rank sends the markov row
    # of its own shard's best token per sequence (a row it holds), and the winner's row is
    # the world sum's (the other ranks' rows are zero there).
    me = nl.ndarray((B, mrank), dtype=nl.bfloat16, buffer=nl.sbuf)
    if fold:
        nisa.dma_copy(dst=me, src=e_sum[R:Re, 0:mrank])
    else:
        if pid == 0:
            nisa.dma_copy(dst=mid[0:B, 0:1], src=ids_sb[0:B, 0:1])
            engram_gather(mid, mk_embed_w, vocab_start, mcc_src, [(0, B)], 1, mrank, part)
        nisa.core_barrier(data=mcc_src, cores=(0, 1))
        ncc.all_reduce(srcs=[mcc_src], dsts=[mcc_dst], replica_group=rg_world, op=nl.add)
        nisa.dma_copy(dst=me, src=mcc_dst[0:B, 0:mrank])
    bramp = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=bramp, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    brampf = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=brampf, src=bramp)
    for i in range(K):
        if pid == 0:
            nisa.dma_copy(dst=memb[i * B : (i + 1) * B, 0:mrank], src=me)
        # bias = bf16(me @ mk_head_w.T) on this rank's columns; logits[b * K + i] += bias[b]; (max, argmax).
        meT = nl.ndarray((PT, mrank // PT, B), dtype=nl.bfloat16, buffer=nl.sbuf)
        for kb in range(mrank // PT):
            tps = nl.ndarray((PT, B), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tps, data=me[0:B, kb * PT : (kb + 1) * PT])
            nisa.tensor_copy(dst=meT[0:PT, kb, 0:B], src=tps)
        # the logits rows of token i: row b * K + i of every sequence
        tkcol = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=tkcol, pattern=[[1, 1]], offset=i, channel_multiplier=K)
        row = nl.ndarray((B, part), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=row, src=lg32.ap(pattern=[[part, B], [1, part]],
                                           vector_offset=tkcol[0:B, 0:1], indirect_dim=0))
        for nb in range(part // 512):
            acc = nl.ndarray((B, 512), dtype=nl.float32, buffer=nl.psum)
            for kb in range(mrank // PT):
                nisa.nc_matmul(dst=acc, stationary=meT[0:PT, kb, 0:B], moving=mkT[0:PT, kb, nb * 512 : (nb + 1) * 512],
                               accumulate=(kb > 0))
            b16 = nl.ndarray((B, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=b16, src=acc)
            # the bf16 bias added straight into the fp32 row (the add reads it as fp32: exact)
            nisa.tensor_tensor(dst=row[0:B, nb * 512 : (nb + 1) * 512], data1=row[0:B, nb * 512 : (nb + 1) * 512],
                               data2=b16, op=nl.add)
        best = nl.ndarray((B, 8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.max8(dst=best, src=row)
        bidx = nl.ndarray((B, 8), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.nc_find_index8(dst=bidx, data=row, vals=best)
        bf = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=bf, src=bidx[0:B, 0:1])
        gidx_f = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gidx_f, data1=bf, data2=vsB, op=nl.add)
        # (max, global index, markov row of the local best token) per sequence -> one all_gather
        bi = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=bi, src=bf)
        el = nl.ndarray((B, mrank), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=el, src=mk_embed_w.ap(pattern=[[mrank, B], [1, mrank]],
                                                vector_offset=bi[0:B, 0:1], indirect_dim=0))
        pk = nl.ndarray((B, GW), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pk[0:B, 0:1], src=best[0:B, 0:1])
        nisa.tensor_copy(dst=pk[0:B, 1:2], src=gidx_f)
        nisa.tensor_copy(dst=pk[0:B, 2:GW], src=el)
        if pid == 0:
            nisa.dma_copy(dst=g_src[0:B, 0:GW], src=pk)
        nisa.core_barrier(data=g_src, cores=(0, 1))
        ncc.all_gather(srcs=[g_src], dsts=[g_dst], replica_group=rg_world, collective_dim=0, priority=AR_PRIORITY)
        # the global argmax per sequence: the highest shard max, ties to the lowest rank
        # (lowest index). The gathered rows are rank-major (row w * B + b), read as [B, world].
        vrow = nl.ndarray((B, W8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(vrow, NEG)
        nisa.dma_copy(dst=vrow[0:B, 0:world], src=g_dst.ap(pattern=[[GW, B], [B * GW, world]], offset=0))
        trow = nl.ndarray((B, W8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(trow, 0.0)
        nisa.dma_copy(dst=trow[0:B, 0:world], src=g_dst.ap(pattern=[[GW, B], [B * GW, world]], offset=1))
        gbest = nl.ndarray((B, 8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.max8(dst=gbest, src=vrow)
        gidx = nl.ndarray((B, 8), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.nc_find_index8(dst=gidx, data=vrow, vals=gbest)
        tok = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.nc_n_gather(dst=tok, data=trow, indices=gidx[0:B, 0:1])
        nisa.tensor_copy(dst=ids_sb[0:B, i + 1 : i + 2], src=tok)
        if i + 1 < K:
            # each sequence's winner's markov row (gathered row winner * B + b): the next token's
            wf = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=wf, src=gidx[0:B, 0:1])
            nisa.scalar_tensor_tensor(dst=wf, data=wf, op0=nl.multiply, operand0=float(B), op1=nl.add,
                                      operand1=brampf)
            wsel = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=wsel, src=wf)
            wr = nl.ndarray((B, mrank), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=wr, src=g_dst.ap(pattern=[[GW, B], [1, mrank]], offset=2,
                                               vector_offset=wsel[0:B, 0:1], indirect_dim=0))
            me = nl.ndarray((B, mrank), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=me, src=wr)
    nisa.core_barrier(data=memb, cores=(0, 1))
    if pid == 0:
        nisa.dma_copy(dst=output_ids[0:B, 0 : K + 1], src=ids_sb)
        # confidence = bf16( [hidden | markov_embed] . proj_w ), row b * K + i
        gather_rows(memb, permc, mperm, kt_all, mrank)
        hcat = nl.ndarray((PT, dim + mrank), dtype=nl.float32, buffer=nl.sbuf)
        h16 = nl.ndarray((PT, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=h16[0:R, :], src=xh[0:R, 0:dim])
        nisa.tensor_copy(dst=hcat[0:R, 0:dim], src=h16[0:R, :])
        m16 = nl.ndarray((PT, mrank), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=m16[0:R, :], src=mperm[0:R, 0:mrank])
        nisa.tensor_copy(dst=hcat[0:R, dim : dim + mrank], src=m16[0:R, :])
        wrow = bcast_row(conf_w[0:1, 0 : dim + mrank], dim + mrank, R)
        prod = nl.ndarray((PT, dim + mrank), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prod[0:R, :], data1=hcat[0:R, :], data2=wrow, op=nl.multiply)
        cf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=cf[0:R, :], op=nl.add, data=prod[0:R, :], axis=(1,), keepdims=True)
        c16 = nl.ndarray((PT, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=c16[0:R, :], src=cf[0:R, :])
        c32 = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=c32[0:R, :], src=c16[0:R, :])
        nisa.dma_copy(dst=confidence[0:R, 0:1], src=c32[0:R, :])
    nisa.core_barrier(data=output_ids, cores=(0, 1))
    bar_end = nisa.core_barrier(data=confidence, cores=(0, 1))
    if use_copy:
        for c in c_out:
            sched_after(nisa.core_barrier(data=c, cores=(0, 1)), bar_end)
    return ret()


#: The generic entry (both phases by the `at_zero` flag): what the unit tests trace.
draft_forward = nki.jit(draft_body)
