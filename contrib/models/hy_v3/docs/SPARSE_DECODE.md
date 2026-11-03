# Block-sparse decode attention (experimental, off)

At 85k context, attention is 55% of an 8-request MTP decode step and 27% of a
single-request one ([PROFILE_85K.md](PROFILE_85K.md)); most of it is reading
the KV history. This study asks whether decode can attend to a fraction of
the history without losing accuracy, before building a fast implementation.
Status: the study below found no accuracy loss at 12.5% of the blocks. The
implementation that reads only the kept blocks
([Implementation](#implementation)) keeps that accuracy on the full model
with FP8 KV + MTP (LongBench v2 54% vs 56%, 98/100 same answers, needle
12/12) and makes 8-request decode steps **25% faster** (ITL 76.7 -> 57.2 ms,
TPOT 51.0 -> 42.9 ms, +20% output throughput), 4-request steps 9% faster;
1 request is 6% slower (it stays dense but keeps the summaries up to date).
Off by default. The same selection for the prefill
([Sparse prefill](#sparse-prefill-hy3_sparse_prefillfraction-experimental-off),
`HY3_SPARSE_PREFILL`, off) makes a cold 85k prefill 15% and prefix-hit
prefills 29-33% faster; with both, 8-request TPOT 51.8 -> 36.3 ms.

## Accuracy study

The study implementation (commit 80c4bdb, since replaced) read the whole K
every step: `HY3_SPARSE_DECODE=<fraction>`, decode
only (prefill stays dense), BF16 KV, no DCP, one token per request per step:

1. In every layer the decode query is computed as the decode kernel does it
   (RMSNorm, STATIC FP8 QKV projection, QK-norm, RoPE).
2. Every 32-token KV block of the request gets the Quest upper bound of
   `q . k` over its tokens: `sum_d max(q_d min_d, q_d max_d)` from the block's
   per-dimension K min / max (each rank holds one KV head and selects for it).
3. `ceil(fraction x blocks)` blocks are kept per request: the first block
   (`HY3_SPARSE_SINK_BLOCKS`, 1) and the last 4 (`HY3_SPARSE_RECENT_BLOCKS`)
   always, the rest by bound.
4. The kept blocks are moved to the front of the block table in their original
   order and the history length is set to the tokens they hold, so the
   unchanged decode megakernel computes exact attention over them.

Evaluation (`scripts/sparse_decode_study.sh`, which now runs the
implementation below: max_model_len 86016, prefix
caching, chunked 4096-token prefill, one request at a time; each fraction is
its own server, the dense run is the reference):

- **needle**: `scripts/needle.py`, passkey at depth 0.1 / 0.5 / 0.9 of 16k /
  32k / 64k / 85k-token prompts (12 cases);
- **continuations**: `scripts/sparse_decode_eval.py`, greedy 64-token
  continuations of 19 long real-text prompts (11 x 85k with a 78k shared
  prefix, 4 x 32k, 4 x 64k ShareGPT), compared to dense;
- **LongBench v2**: `scripts/longbench_v2.py`, 100 multiple-choice items
  (THUDM/LongBench-v2, official no-CoT template) of 10k-81k tokens (median
  30k), greedy; accuracy and the share of items answered like dense;
- **control**: a fraction of 1e-6 keeps only the sink and the 4 recent blocks
  (no selection by the bound).

## Results

| decode attends to | needle | continuations identical to dense | LongBench v2 accuracy | LongBench v2 same answer as dense |
|---|---|---|---|---|
| all blocks (dense) | 12/12 | (reference) | 57% | (reference) |
| 50% of the blocks | 12/12 | 18/19 | 59% | 98% |
| 25% | 12/12 | 19/19 | 60% | 97% |
| **12.5%** | **12/12** | **19/19** | **60%** | **97%** |
| sink + 4 recent blocks only (control) | **0/12** | 4/19 (25% common prefix) | | |

Raw outputs: `results/sparse_decode/`.

- At 12.5% of the blocks (4k of a 32k history, 10.6k of 85k) nothing measured
  drops: all needles found, every 64-token continuation identical to dense,
  and LongBench v2 answers differ on 3 of 100 items, all three from a wrong
  dense answer to the right one (9 items have no parsable answer in any run).
- The control shows that the decode kernel really reads only the compacted
  blocks (no needle, 4/19 identical continuations), so the selection by the
  bound is what keeps the accuracy. In the control the first digits of a key
  are often right: the first answer token comes from the (dense) prefill.
- The one different continuation at 50% (an open-ended 85k ShareGPT prompt,
  diverging after 81 characters) is not repeated at 25% or 12.5%: run-to-run
  variation of the near-tie kind seen elsewhere, not a trend.

Limits: decode only (the prefill, where the needle is read in, is dense); one
model, these task types (retrieval, continuation, long-document multiple
choice); generations of 16-64 tokens; BF16 KV (FP8 packed K would need the
summaries from the unpacked values).

## Implementation

`HY3_SPARSE_DECODE=<fraction>` (src/model.py; no DCP), decode only:

| step | study | implementation |
|---|---|---|
| K min / max | per 32-token block, gathered from the K cache every step | per group of `HY3_SPARSE_GROUP` blocks (4: 128 tokens), `k_sum` [blocks, 2, kv heads, 128] BF16 next to the KV cache, at the row of the group's first block (`bind_attention_kv`); written by prefill (`update_k_summary_prefill`, from the K before FP8 quantization) and by every decode step (`update_k_summary_decode`); 1/16 of the FP8 KV bytes |
| query (and K) for the bound | the kernel's | computed again in torch (`_sparse_decode_blocks`) from own copies of the Q and K weight columns (dequantized to BF16) and the norm gammas, with RoPE from the positions (`prepare_sparse_decode`) |
| kept | a fraction of each request's blocks | `sparse_budget_blocks`: the fraction of the decode context bucket, rounded up to a multiple of 256 blocks (0.125 of 2688 -> 512 blocks = 128 groups, 16k tokens): the first group and the last 2 (`HY3_SPARSE_SINK_GROUPS`, `HY3_SPARSE_RECENT_GROUPS`) and the rest by bound (top-k) |
| decode attention | full-width table, masked | the kept groups' blocks as a [B, 512] block table and `s_prior` 16384: the kernel reads only them. The kept groups are in any order (attention does not depend on it), then the history's last group, then the one after it; a history that fits gets the head of its table |

Every `HY3_SPARSE_SELECT_EVERY` (4) layers select; the 3 after each reuse
its table and mask (all layers share the block table) and only fold their K
into their summaries. With MTP the bound is the max over the 4 verify tokens, and the draft layer
selects for its own cache too (`HY3_SPARSE_DRAFT=0` turns that off). Steps
of fewer than `HY3_SPARSE_MIN_BATCH` (4) requests attend densely (the decode
NEFFs are per batch bucket): the selection costs about the same at any batch
size, the attention it saves grows with it. They still update the summaries.

Why groups: the selection is a handful of small torch ops per layer whose
cost is mostly fixed. `scripts/sparse_select_bench.py` times them on one
NeuronCore at the full-model per-rank shapes (8 requests x 4 verify tokens,
2688 blocks):

| selection (`sparse_select_blocks` and its parts) | time |
|---|---|
| launch + sync of a trivial graph | 0.10 ms |
| per block (group 1, the first version): summary gather / + bound / + top-511 / + sort | 0.24 / 0.27 / 0.86 / 0.98 ms |
| per block, top-k replaced by a 16-step threshold bisection, + scatter or compare compaction | 0.57, + 1.8 or + 0.3 ms |
| per block, whole function | 1.07 ms |
| groups of 4 (672 groups, top-127), whole function | 0.35 ms |

By the 85k profile, attending to 1/5 of the blocks saves ~0.46 ms per layer
at 8 requests, ~0.23 ms at 4 and ~0.06 ms at 1.

Pitfalls found on the way (covered by `test/unit/test_sparse_decode.py` or
comments in the code):

- **Torch ops reading tensors that also feed the decode megakernel** (the
  cos / sin, the QKV weight, the norm gammas) gave wrong attention results;
  hence the own weight copies and RoPE.
- **The kernel reads a step's earlier verify tokens back through the block
  table**: when the 4 MTP tokens cross a block boundary, the block after the
  history must be in the compacted table. At most `n_sel / group - 1` groups
  are kept for that. Without it, LongBench v2 lost 5 of 78 items (3 of them
  unparsable answers), while the needle test still passed.
- **Recycled blocks keep the summary of their previous request**: with
  summaries written at prefill only (the first version), blocks decode wrote
  looked like valid history and could be dropped once an output outgrew the
  forced recent blocks. Decode now folds its K into the summaries (a step
  starting a group restarts it; rejected verify tokens only loosen the bound).
- A block-table width that is not a multiple of 256 makes the kernel split
  the 32-token blocks, which the compacted table does not survive.
- HBM: the summaries (~0.9 GB per rank at 697k tokens) do not fit next to
  the 32768-bucket decode NEFFs (`nrt_tensor_allocate` at runtime), so
  `serve_85k.sh` drops that bucket when `HY3_SPARSE_DECODE` is set.
- The 4-layer mini model (`hy3_mini4_mtp`) does not show such bugs in greedy
  tokens: with MTP even a recompiled dense run flips near-tie tokens there.
  Compare captured activations (without MTP) or use the full model.

### Profile of a sparse step

`scripts/profile_85k.sh` with `HY3_SPARSE_DECODE=0.125` (KV_CAP 0.45 for the
profiler's buffers: 5 requests at 85k, the batch-8 NEFF), rank 0, version
255fbaf (`results/profile_85k_sparse/`, `scripts/profile_attn_breakdown.py`
attributes the attention module's instructions to their model.py function):

| per 5-request step | dense (c6, [PROFILE_85K.md](PROFILE_85K.md)) | sparse |
|---|---|---|
| target NEFF | 72.0 ms | 61.9 ms |
| attention module (union) | 45.7 ms | 30.0 ms |
| NKI attention kernels (`attention_tkg`, `attention_block_tkg`) | ~38 ms | 12.9 ms |
| torch: selection (`sparse_select_blocks`) | | 16.2 ms |
| torch: Q / K for the bound (`_sparse_decode_blocks`) | | 2.4 ms |
| torch: summary update | | 1.4 ms |

Within the selection (80 + 1 layer calls): the summary gather 4.3 ms, top-k
3.6, the sort back into table order 2.6, the empty-summary handling 3.5, the
block-id gather 2.2, the bound 1.7. The sort, the empty-summary handling (a
group without a summary scores <= 0 by itself) and the per-block gather (now
per group) are gone in the current version.

### Results (full model, `serve_85k.sh` FP8 KV + MTP k=3, 0.125)

Accuracy (`results/sparse_decode/lb_full_fp8mtp_*.json`, needle logs):

| | LongBench v2 | same answer as dense | needle |
|---|---|---|---|
| dense | 56% | | |
| per block, prefill summaries (c5ab0c9) | 55% | 95/100 | 12/12 (before the verify-token fix) |
| groups + decode summaries (255fbaf) | 55% | 97/100 | 12/12 |
| + no sort / per-group gather (2395994) | 55% | 97/100 | 12/12 |
| **+ selection every 4 layers (current)** | **54%** | **98/100** | **12/12** |

The differing items are coherent near-ties ("The correct answer is (B)"; the
2 of the current version also differ in the per-layer versions); 8
unparsable answers in every run.

Speed: P/D split, `bench_pd_split.sh "1 4 8"` (85k prompts with a 78k shared
prefix, 256 output tokens, pass 2; `results/bench/pd_split_85k_nodcp_fp8kv_mtp3`
and `..._sparse` (c5ab0c9), `..._sparse_g4` (255fbaf), `..._sparse_g4b`
(2395994), `..._sparse_g4r4` (current)):

| concurrency | ITL dense | c5ab0c9 | 255fbaf | 2395994 | **current** | TPOT dense | 2395994 | **current** | output tok/s dense / current |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 47.4 ms | 100.5 ms | 50.8 ms | 50.7 ms | 50.1 ms | 23.4 ms | 25.6 ms | 24.6 ms | 37.9 / 36.2 |
| 4 | 66.6 ms | 128.1 ms | 66.9 ms | 64.3 ms | **60.6 ms** | 39.2 ms | 37.3 ms | **36.3 ms** | 92.0 / 97.5 |
| 8 | 76.7 ms | 137.1 ms | 66.2 ms | 61.3 ms | **57.2 ms** | 51.0 ms | 44.7 ms | **42.9 ms** | 138.3 / **166.2** |

MTP accepts 2.0 tokens per step in every run. TPOT gains less than ITL at 4
and 8 requests because it also carries the (dense) prefill steps of arriving
requests that interrupt decode (p99 ITL ~600 ms in all runs). At 1 request
the step stays dense; the ~3 ms are the Q / K projection and the summary
update in 80 + 3 layer calls. An NKI kernel for the selection itself does
not pay off much: on one core the NKI top-k kernels
(`vllm_neuron.functional.topk`) are no faster than `torch.topk` at [8, 672],
and the summary gather is a DMA already (descriptor-bound: 5376 rows of 512
bytes per layer). What is left: the K projection and summary update in every
layer (~4 ms per step), which taking K from the decode kernel would remove
(also the 1-request cost; a framework change), and the selection in every
4th layer.

## Sparse prefill (`HY3_SPARSE_PREFILL=<fraction>`, experimental, off)

The same summaries and bound applied to the segmented prefill (FP8 KV, no
DCP). Per layer and chunk, `sparse_prefill_blocks` scores every complete
128-token group of the request's prior context with the largest bound over
the chunk's (non-padding) queries and keeps the first group, the last two,
the incomplete last group and the best others up to the budget (768 of 2688
blocks at 0.25 for an 86016 table; all of them while they fit). Only the kept
blocks are copied to BF16 (`_request_blocks_bf16`), followed by the chunk's
own blocks; the kernel gets the kept-token count as `prior_tokens`, which is
where it finds the chunk in that layout. The chunk itself stays causal and
dense. Decode keeps the summaries up to date also when `HY3_SPARSE_DECODE`
is off (later chunks of a prefix hit read them).

Accuracy (full model, `serve_85k.sh` FP8 KV + MTP k=3, decode dense;
`results/sparse_prefill/`; measured with the first version, 66777fb, which
keeps the same blocks as the current one):

| | needle | LongBench v2 | same answer as dense | right -> wrong / wrong -> right | 64-token continuations identical |
|---|---|---|---|---|---|
| dense | 12/12 | 56% | | | |
| prefill 0.25 | 12/12 | 58% | 88/100 | 4 / 6 | 19/19 |
| prefill 0.125 | 12/12 | 51% | 90/100 | 7 / 2 | 19/19 |

For scale: two dense runs (BF16 and FP8 KV) agree on 94/100, sparse decode
0.125 on 98/100. Sparse prefill changes more answers than sparse decode;
at 0.25 the changes cancel, at 0.125 they lean to losses (7 vs 2, not
significant at 100 items). 0.25 is the setting measured for speed.

Speed (`scripts/cold_ttft.py` for the cold prompt, `bench_pd_split.sh "1 4 8"`;
`results/sparse_prefill/cold_ttft_needle.txt`, `results/bench/pd_split_85k_nodcp_fp8kv_mtp3_rerun`
(dense, same session), `..._sprefill` (prefill 0.25, 66777fb), `..._sprefill_sdecode`
(prefill 0.25 + decode 0.125, 66777fb), `..._sprefill_sdecode_nosort` (the same, current)):

| | dense | prefill 0.25 (66777fb) | + decode 0.125 (66777fb) | **+ decode 0.125 (current)** | decode 0.125 alone |
|---|---|---|---|---|---|
| cold 85k prefill TTFT | 14.34 s | 15.86 s | 15.80 s | **12.13 s (-15%)** | |
| P: 78k prefix hit, TTFT c1 / c4 / c8 | 2.14 / 7.69 / 15.32 s | 1.87 / 6.60 / 13.14 s | 1.87 / 6.58 / 13.12 s | **1.52 / 5.19 / 10.34 s (-29% / -33% / -33%)** | 2.16 / 7.81 / 15.55 s |
| D pass 2, ITL c1 / c4 / c8 | 47.4 / 66.6 / 76.6 ms | 50.1 / 76.6 / 83.1 ms | 50.1 / 60.6 / 57.3 ms | 50.2 / 60.6 / 57.2 ms | 50.1 / 60.6 / 57.2 ms |
| D pass 2, TPOT c1 / c4 / c8 | 23.4 / 39.4 / 51.8 ms | 24.6 / 43.5 / 53.1 ms | 24.9 / 35.9 / 40.8 ms | 24.9 / **34.2** / **36.3** ms | 24.6 / 36.3 / 42.9 ms |
| D pass 2, output tok/s c1 / c4 / c8 | 38.0 / 91.7 / 139.6 | 36.1 / 83.5 / 134.2 | 36.2 / 99.4 / 168.8 | 37.3 / **106.0** / **194.2** | |

(decode 0.125 alone: the `..._sparse_g4r4` run above, before the prefill MoE
change.)

- The first version (66777fb) made a cold 85k prefill 11% slower. Its device
  profile (`results/profile_85k_sparse_prefill/summary.md`) found ~190 ms per
  chunk in the `torch.sort` of the 2688 block ids (2.2 ms per layer on one
  core; the profiler files it under `Sampler`, the first call site of the
  deduplicated custom call). The kept groups need no order, so the current
  version gathers the top-k groups' blocks directly: selection 2.40 -> 0.27 ms
  per layer, the same blocks kept.
- Now a cold 85k prefill is 15% faster and one on a cached 78k prefix (the
  6.7k-token suffix, 2 chunks) 29-33% faster. Shorter prefills interrupt
  decode for less time: with sparse decode, 8-request TPOT 42.9 -> 36.3 ms and
  output 194 tok/s (dense: 51.8 ms, 140 tok/s, +39%). The decode step (ITL)
  is the sparse-decode one.
- Prefill alone (decode dense) is not useful: decode pays the summary updates
  (3-10 ms per step) with nothing to gain from them. Use it together with
  `HY3_SPARSE_DECODE`.
