# DeepSeek-V4.1-Flash tp64 (opt12) — decode-step profile at in1024 bs1, and a plan for a 7 ms step

**Scope.** One steady bs1 decode step at ~1k context on the current src (opt12: TP8 x EP8 decode MoE,
drafter MoE in the same layout; `bench_report.opt12.json`), live server, trn3 (LNC=2, 64 ranks).
Focus: the 40 transformer blocks (5 fused launches) and the DSpark drafter. Everything below is
measured on the device timeline of the live server unless marked *estimate*.

## TL;DR

* A decode step is **16.94 ms** of device-window time (mean of 29 steady steps; the bench's mean
  step for this cell is 17.4 ms). The **40 blocks take 13.98 ms (82.5 %)**, the **drafter 1.53 ms
  (9.1 %)**, and **launch overhead ~1.25 ms** (entry preambles + entry barriers + gaps + the
  prologue / tail graphs).
* The hardware is mostly waiting. Over the step **mean MFU is 6.5 % and mean MBU 15.4 %** (blocks:
  MFU 5–9 %, MBU 14–25 %; drafter 3.2 % / 11 %). No engine is saturated: in the blocks TensorE
  is active 36 % of the time, VectorE 52 %, GpSimd 35 %, ScalarE 24 %, DMA 56 %, and 92 % of the
  10 µs bins are *under-utilised* (MFU and MBU both below half of what the same kernel reaches
  elsewhere).
* What the idle time is: **dependency stalls 40 %** (engines waiting on each other's semaphores —
  long chains of small dependent instructions across 5 engines x 2 cores), **VectorE busy at low
  efficiency 16 %** (Sinkhorn, rope, routing on 6-row tiles), ScalarE / GpSimd / SP busy 12 %,
  TensorE busy at low efficiency 8 % (expert GEMMs with 6-column stationaries), collectives'
  transfer 7 %, DMA queues 7 %. Per engine: the TensorE is idle 55 % of a block, half of it
  waiting for DMA'd data and a quarter for the VectorE; the VectorE waits on the TensorE a third of
  the block; the sync engine idles in cross-core barriers. A normal block issues **~5,000 instructions, ~50 cross-core
  barriers and ~700 semaphore waits** for 75 MB of HBM traffic and 3.9 GFLOP.
* The floors are far below the step: the 40 blocks move **3.5 GB of HBM (2.98 ms at peak
  bandwidth)** and **~160 GFLOP (~1 ms of TensorE)**; the collectives' fastest instances are
  14.4 µs (AR over 8 ranks) and 16.1 µs (AR over 64 ranks), **1.2 ms a step at their floor**.
  So 7 ms is a dependency-chain / dataflow problem, not a bandwidth or compute one.
* **A 7 ms step needs** launch overhead 1.25 → 0.35 ms, the 40 blocks 14.0 → 5.6 ms (a normal
  block 303 → ~125 µs, the 8 index / KV-source / engram blocks 535 → ~200 µs) and the drafter
  1.53 → 0.85 ms. Section 7 lays out the work streams that get there: one persistent launch a step,
  an SBUF-resident residual stream with on-chip cross-core exchange, the hyper-connection and
  routing chains rebuilt off the critical path, the projections without HBM round trips, the
  overflow loop gone, and the special blocks' index path re-scheduled. With the estimated
  per-segment savings the step lands at ~7 ms; the largest items are structural (weeks, not days).

## 1. What was captured

| | |
|---|---|
| Code | branch `deepseek_v41_trn3`, uncommitted opt12 working tree (integration test passed, `bench_report.opt12*.json`) |
| Stack | vLLM 0.29 (`aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0`), neuronx-cc 2.28.7631, runtime 2.35.30, neuron-explorer 2.33.0.1039 |
| Workload | in1024 bs1 (sonnet prompt, 1024 tokens, 192 generated, greedy), `--max-model-len 1280 --max-num-seqs 1`, tp64, async scheduling, DSpark 5 draft tokens |
| Method | serve + `/start_profile` / `/stop_profile` (skill *neuron-framework-profiling-vllm-neuron*, method 1), `neuron_profiler` device + system profile, decode window 0.6 s opened once every stream is in steady decode; NRT profile buffers 512 / 256 MB |
| Captures | **rank 0** (0.6 s, 32 steps; steps 2–30 used) and **8 ranks** 0, 4, 9, 18, 27, 36, 45, 63 (0.4 s, for cross-rank skew) |
| Analysis | ingested with `neuron-explorer view --ingest-only`; per-launch / per-block / 10 µs-bin metrics with the opt10 profiling scripts (`analyze6`, `timeline6`, `tables6`, `util6`, `skew6`) plus `segments12.py` / `milestones12.py` (block anatomy) and `idle12.py` (what each engine waits for) |
| Data | `/mnt/nvme/ds41_profiling_opt12/` (profiles, parquet, `analysis/`, `report/` with every table below) |

Caveats: the per-block tables are rank 0. A fused launch carries no per-member markers, so a block
is cut at its world all-reduce's end. "MBU" is HBM bytes of the DMA-issuing instructions over the
unit's wall time against 1.18 TB/s per LNC; "MFU" is useful (non-transpose) BF16-equivalent FLOPs
against 157 TFLOP/s per LNC. The 8-rank skew is a lower bound (56 of 64 ranks are not captured).
Profiling overhead is small here (16.94 ms device window vs 17.4 ms bench step).

## 2. Where a decode step goes

| Component | ms / step | % |
|---|---:|---:|
| 40 transformer blocks (5 fused launches) | 13.98 | 82.5 |
| DSpark drafter (one launch) | 1.53 | 9.1 |
| fused launches' entry (preamble ~80 µs + entry barrier ~64 µs, x5) | 0.74 | 4.4 |
| device idle between the step's launches (~32 µs turnaround each) | 0.34 | 2.0 |
| final norm + lm_head + greedy argmax (in the last launch) | 0.14 | 0.9 |
| step prologue (NKI launch) | 0.12 | 0.7 |
| decode tail graph (draft verification + drafter inputs) | 0.05 | 0.3 |
| embedding + hc_expand (in the first launch) | 0.03 | 0.2 |
| **Total** | **16.94** | 100 |

Per launch (device time, mean): L00–L07 + embed 2.995 ms, L08–L15 3.191, L16–L23 2.810, L24–L31
2.852, L32–L39 + head 3.053, drafter 1.535, prologue 0.153, tail graph 0.078. Eight device launches
a step; each costs ~80 µs of runtime preamble (descriptor writes, semaphore resets, drains) before
its first instruction, ~60–70 µs in the entry barrier (the arrival spread of the 8 captured ranks
at the entry barrier is ~40 µs) and ~32 µs of turnaround from the previous one.

## 3. The 40 blocks

### 3.1 By block kind

| Block kind | blocks | µs / block | ms / step | mean MFU / MBU % | peak MFU % (10 µs bin) | peak MBU % | TensorE / VectorE / GpSimd active % (approx. mean) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `r2` | 15 | 304 | 4.55 | 8.3 / 18.0 | 49 (`moe_tp._up`) | 87 (`_wdma`) | 40 / 52 / 34 |
| `r1` | 12 | 323 | 3.88 | 7.5 / 16.4 | 49 (`moe_tp._up`) | 76 (`_wdma`) | 37 / 52 / 38 |
| `r1_idx` (L24, L28, L32, L36) | 4 | 396 | 1.58 | 5.7 / 13.6 | 48 | 73 | 32 / 52 / 37 |
| `r1_tap` (L37–L39) | 3 | 326 | 0.98 | 7.5 / 14.8 | 49 | 75 | 36 / 54 / 39 |
| `r2_src_idx` (L02, L08) | 2 | 503 | 1.01 | 4.8 / 14.2 | 47 | 80 | 30 / 49 / 29 |
| `r2_src_idx_e` (L14) | 1 | 711 | 0.71 | 5.1 / 14.0 | 87 | 86 | 24 / 52 / 27 |
| `r0_e` (L01) | 1 | 508 | 0.51 | 6.7 / 16.1 | 87 | 78 | 25 / 56 / 28 |
| `r1_src_idx` (L20) | 1 | 474 | 0.47 | 6.2 / 14.8 | 76 | 66 | 30 / 52 / 36 |
| `r0` (L00) | 1 | 290 | 0.29 | 8.1 / 14.5 | 49 | 92 | 39 / 51 / 33 |

The 32 "normal" blocks (r0 / r1 / r2 / r1_tap) average **303 µs** (9.70 ms a step); the 8 special
blocks (index sources, KV sources, engram) average **535 µs** (4.28 ms) and cost **~1.9 ms a step
above** what 8 normal blocks would.

### 3.2 Anatomy of a normal block (live trace, median of the steady steps)

`r1` block, 323 µs, t = 0 at the previous block's world all-reduce end (both cores):

| Segment | µs | What runs (and the measured engine time in the block) |
|---|---:|---|
| hc_out of the previous FFN + hc_in(attn) | 0 → 47 | stream update, mix GEMM (`hc_in_tile`, block-diagonal `add`), spread norm (`hc_in_x_spread`, `spread_rstd`) |
| q / kv projections, norms, rope | 47 → 67 | `gemm` (wq_a, wkv, wq_b), `norm_rows`, `rope_inplace`, `_qdq_fp8` |
| attention | 67 → 109 | `attn_fetch` (gathers), `key_tile_packed` (key transposes, 20 µs/core), `attn_scores`, `attn_softmax`, `attn_pv`, `attn_store` |
| output projection | 109 → 148 | `gemm` (wo_a, cross-core exchange, wo_b), stores to `cc_src`, barrier |
| **AR-8** (attention out, 123 KB fp32) | 148 → 165 | floor 14.4 µs |
| hc_out(attn) + hc_in(ffn) + router + routing | 165 → 216 | `hc_out_tile_spread`, mix + Sinkhorn on core 1, `router_logits` (4.5 µs of PE), `route_group` (top-6 of 384, group compaction, slot ids into registers) |
| routed + shared experts | 216 → 270 | `_load_up` / `_load_down_half` (dynamic weight DMAs), `_up` (44 µs of PE a core), `_swiglu`, `stage_rows`, `_down_cs` |
| tail | 270 → 289 | output store, the overflow loop's entry / exit (taken zero times, ~8–10 µs: every engine drains), barrier, trigger |
| **AR-64** (MoE out, 123 KB fp32) | 289 → 322 | floor 16.1 µs; the rest is waiting for the last rank |

Summed over the step (40 blocks): attention side (block start → AR-8 trigger) **7.16 ms**, AR-8
**0.74 ms** (floor 0.58), MoE side (AR-8 end → AR-64 trigger) **4.65 ms**, AR-64 **1.32 ms** (floor
0.64), the index blocks' score all-reduce 0.19 ms, the engram all-reduces 0.10 ms.

`r2` blocks (299 µs) are the same shape: AR-8 137 → 155, AR-64 266 → 297; their core 1 runs the
mix from HBM (`hc_mix_hbm`, 68 µs of ScalarE / SP time a block).

### 3.3 The special blocks

Their MoE side is a normal block's (110–126 µs); the extra time is all on the attention side:
`r1_idx` +65 µs (index-query projection, index scores, the score AR-8 at 25–28 µs, top-k on GpSimd
behind the slot-table gathers, the per-row index table built on 6 partitions x 512 columns),
`r2_src_idx` / `r1_src_idx` +160–200 µs (plus the KV compressor), `r0_e` / `r2_src_idx_e` +215–415
µs (plus the engram read-out: a 1.6 MB AR-64 of engram rows, 52 µs, and its gate).

### 3.4 Floors

| per block (rank 0, both cores) | HBM MB | at 1.18 TB/s | GFLOP (BF16-eq) | at 157 TFLOP/s | collectives active |
|---|---:|---:|---:|---:|---:|
| `r2` | 76.9 | 65 µs | 3.9 | 25 µs | 49 µs |
| `r1` | 73.0 | 62 µs | 3.8 | 24 µs | 48 µs |
| `r1_idx` | 73.7 | 63 µs | 3.5 | 23 µs | 84 µs |
| `r2_src_idx_e` | 127.0 | 108 µs | 5.7 | 36 µs | 100 µs |
| drafter (whole launch) | 237.0 | 201 µs | 7.8 | 50 µs | 348 µs |
| **40 blocks** | **3,515 a step** | **2.98 ms** | ~160 | ~1.0 ms | ~2.4 ms (of which floor ~1.2) |

Where a normal block's 75 MB go (`hot6`): attention weights 22–24 MB (`_wdma`: wq_a 6.6 MB and wkv
2.6 MB are replicated on every rank, wq_b / wo_a / wo_b are the rank's 1/8), routed-expert slices
~29 MB (`_load_up` 17.7 + `_load_down_half` 11.1), router weight 3.9 MB (bf16, replicated),
KV / slot-table gathers 4–5 MB, shared-expert block 1 MB, the two hc mix weights `fn.T` (fp32) ~4 MB.

## 4. The drafter

1.535 ms, one launch, 3 MTP stages on 5 draft rows, then the 5-token Markov head:

| Piece | µs | collective after it (µs, fastest) |
|---|---:|---|
| entry barrier | 54 | — |
| prologue: main_x projection, token embedding | 28 + 27 | AG-64 `mxs` 26 (23), AR-64 `e_loc` 35 (28) |
| stage 0: attention side / MoE side | 181 / 133 | AR-8 27 (20) / AR-64 42 (34) |
| stage 1 | 158 / 121 | AR-8 27 (20) / AR-64 42 (35) |
| stage 2 | 169 / 126 | AR-8 27 (21) / AR-64 30 (21) |
| head: norm + lm_head GEMM | 106 | then 5 x (22 µs of argmax / Markov-row work + AG-64 11 µs) |
| tail | 29 | — |

Collectives are 367 µs (24 %) of it; mean MFU 3.2 %, MBU 11 %; engines active: TensorE 31 %,
VectorE 47 %, GpSimd 23 %, DMA 47 %. Its largest items: `hc_out_spread` 167 µs and `hc_in_mix`
157 µs of any-engine wall, `gemm` 135, `interleave_rows` 118 (unpacked activation staging),
`hc_in_norm_spread` 110, Sinkhorn 144; the attention weights it streams are 79 MB (`_wdma`).

## 5. Why the engines idle

### 5.1 What the under-utilised time is spent on (all device time of the step)

| Cause (one per 10 µs bin) | % of the step's device time |
|---|---:|
| dependency stall (no engine >= 70 % busy: engines wait on each other's semaphores) | 40 |
| VectorE busy at low efficiency | 16 |
| ScalarE / GpSimd / SP busy at low efficiency | 12 |
| TensorE busy at low efficiency | 8 |
| collective transfer | 7 |
| DMA queues busy, few bytes | 7 |
| collective waiting for peers / kernel-entry barrier | 1 / 1 |
| (not under-utilised) | 8 |

Largest single items (ms a step): TensorE on `moe_tp._up` 0.83 (6-column stationaries against
288-column moving tiles, both halves of 8 static slots whether or not an expert is active); VectorE on
the Sinkhorn `coldiv` 0.77 + `rowdiv` 0.34 (20 iterations of tiny reductions); dependency stalls in
`mk_mx.gemm` 0.75 (projections waiting for staged activations and weights), `_down_cs` 0.50,
`attn_pv` 0.36, `key_tile_packed` 0.33, `route_group` 0.32, `router_logits` 0.30; AR-8 transfer
0.55; GpSimd issuing `_load_down_half` descriptors 0.39; ScalarE on `hc_mix_hbm` 0.37.

### 5.2 The block is a chain, not a load

Per normal block on rank 0 (both cores): TensorE ~1,700 instructions, SP (sync) ~1,600 (mostly
DMA triggers and semaphores), VectorE ~1,000, GpSimd ~450, ScalarE ~350 — **~5,000
instructions in ~310 µs**, of which ~700 are semaphore waits, and **~50 cross-core barriers**
(`nisa.core_barrier`: one pseudo-instruction per core each, 22 on the sync engine at 2.4–4.0 µs and
28 on GpSimd at 0.6–0.9 µs on average). Data moves between the two cores and between sublayers
through shared HBM (store, barrier, reload: 2–5 µs each), and every phase hands over between
engines (PE → PSUM copy on DVE → DMA → GpSimd → ...). The critical path is therefore a sequence of
hand-offs; shortening or removing hand-offs is what moved this round (the TP8 x EP8 MoE layout, the
Sinkhorn's instruction count, fewer slot DMAs: the bs1 step went from ~25 to ~17 ms), while making
individual instructions cheaper did not (narrower stationaries, stacked projection staging: no
measurable change in the per-launch workspaces).

### 5.3 What each engine waits for

Each idle gap of an engine (no instruction executing on it, per core) is charged to what the
instruction that ends the gap waited for (the first semaphore in its wait condition). Percent of the
block time, mean of the two cores (`report/idle_r0.txt`, `scripts/idle12.py`):

| normal blocks | idle % | … waiting on TensorE | VectorE | ScalarE | GpSimd | a DMA queue | barrier / register sem. | nothing (queue empty) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TensorE | 55 | — | 12.9 | 7.5 | 0 | **27.1** | 2.4 | 5.3 |
| VectorE | 65 | **33.5** | — | 10.8 | 3.0 | 11.3 | 0.2 | 4.0 |
| ScalarE | 79 | 19.0 | **30.9** | — | 8.3 | 8.6 | 3.9 | 8.3 |
| GpSimd | 72 | 3.6 | 9.1 | 13.9 | — | 13.6 | 4.4 | **26.9** |
| SP (sync) | 88 | 1.2 | 1.3 | 7.8 | 5.2 | 17.7 | **37.4** | 16.9 |

(special blocks: the same pattern, TensorE waits on VectorE 26 % and on DMA 27 %, SP on barriers 46 %;
drafter: TensorE waits on DMA 49 %, ScalarE on VectorE 44 %, SP on barriers 33 %.)

Read across: the **TensorE is idle 55 % of a block, half of that waiting for data to arrive by
DMA** (weights after routing, gathered keys, activations staged through HBM) and a quarter waiting
for the VectorE to evacuate PSUM / stage the next operand; the **VectorE waits on the TensorE**
(a third of the block) — the matmul → PSUM copy → next matmul ping-pong of 6-row tiles; the
**sync engine spends its idle time in cross-core barriers**. These are the three hand-off chains the
plan attacks: DMA latency on the PE's path (W2, W4d, W6, W7: data resident or prefetched), the
PE ↔ DVE ping-pong (W3, W5: fused epilogues, fewer, larger PSUM evacuations), and the barriers (W2).

### 5.4 Collectives and skew

* Fastest instances: AR-8 (123 KB) 14.4 µs, AR-64 (123 KB) 16.1 µs; the drafter's AR-64 (102 KB)
  21–35 µs, AG-64 (1 KB) 10 µs.
* Measured means: AR-8 17.4–18.5 µs (excess 0.17 ms a step), AR-64 33 µs (excess **0.68 ms**).
* Across the 8 captured ranks the arrival spread at the MoE AR-64 is median 9.0 µs (p90 15 µs,
  max 62), at the AR-8 9.0 µs (p90 16 µs) — a lower bound for 64 ranks. The work per rank is
  uniform after the TP8 x EP8 layout (MoE side 116 µs on every captured rank); rank 63 is the
  last arrival at ~45 % of the all-reduces (of 8 captured) (its attention side is ~4 µs longer).
* The tail before the AR-64 (19 µs) includes the overflow loop's ~8–10 µs entry / exit even when
  it runs no iteration; `nki.experimental.if_else` to skip it does not compile with this
  compiler ("Detected Block1 with forward edges").

## 6. What a 7 ms step means

At today's ~3.4 tokens a step, 7 ms is a TPOT of ~2.1 ms (~480 tok/s per user). Budget:

| Part | today ms | target ms | needs |
|---|---:|---:|---|
| launch overhead (entries, barriers, gaps, prologue / tail graphs) | 1.25 | 0.35 | one persistent launch a step (or two) |
| 32 normal blocks | 9.70 (303 µs each) | 4.00 (125 µs) | −59 % a block |
| 8 special blocks | 4.28 (535 µs each) | 1.60 (200 µs) | −63 % |
| drafter | 1.53 | 0.85 | −45 % |
| head + embed | 0.17 | 0.15 | — |
| **step** | **16.94** (+ ~0.5 between steps in the bench) | **6.95** | |

A 125 µs normal block still sits above its floors (HBM ~63 µs at peak, PE ~25 µs, collectives
~31 µs at their fastest) but only if weight streaming overlaps the chain (MBU ~50 % inside a block,
3x today) and the two all-reduces are close to their floors. Segment targets for an `r1` block:

| Segment | today µs | target µs | lever (section 7) |
|---|---:|---:|---|
| hc_out(prev) + hc_in(attn) | 47 | 12 | W2, W3 |
| q / kv projections, norms, rope | 20 | 12 | W5 |
| attention | 42 | 20 | W6 |
| output projection | 39 | 12 | W5 |
| AR-8 | 17 | 15 | W4 |
| hc_out(attn) + hc_in(ffn) + router + routing | 51 | 18 | W2, W3, W7 |
| experts | 54 | 28 | W7 |
| tail (store, overflow loop, barrier, trigger) | 19 | 3 | W4, W7 |
| AR-64 | 33 | 17 | W4 |
| **block** | **322** | **137** | |

## 7. The plan

Each work stream lists the evidence, the mechanism, the estimated saving (*estimate*: from the
measured segment times above) and the risk. They overlap in places; the totals are not additive
to the microsecond.

**W1 — One persistent launch a step (−0.9 to −1.0 ms).** Eight device launches cost ~170 µs each
in preamble + entry barrier + turnaround (1.25 ms a step with the prologue / tail graphs). Fold the step prologue, the five fused
launches, the decode tail (verification, drafter inputs) and the drafter into one NKI launch (or
two: target and drafter), using the fused-launch generator that already concatenates members.
Also removes the ~40 µs entry-barrier skew per launch and lets the next layer group's weights
prefetch across today's launch boundaries. Risks: compile time and NEFF / instruction size of a
40-block + drafter kernel, the number of IO tensors (host submission cost grows with them: ~2,000
weight tensors would need to be packed into a few buffers), async-scheduling hooks.

**W2 — Residual stream resident in SBUF, cross-core exchange on chip (−25 to −40 µs a block,
~1.2 ms a step).** Today the 4 hc streams (4 x 6 x 5120 bf16 = 245 KB) live in shared HBM; each
sublayer stores, barriers and reloads them, and the two cores exchange partial results through
HBM + `core_barrier` (~50 barriers a block, several HBM round trips). Keep the streams in SBUF across
a launch's blocks (each core its half of the columns), exchange with `nisa.sendrecv` (as
`COMBINE_SR` / `ROUTER_SR` already do for two hand-offs), and write hc_out's result directly in
the layout the next hc_in and the projections read (spread / packed). Risk: SBUF budget next to
the weight double-buffers; every member family (r0 / r1 / r2 and their idx / src / tap / engram
variants) must take the same layout.

**W3 — Hyper-connection chain off the critical path (−25 to −35 µs a block).** Per sublayer the
critical path carries the mix GEMM, the 20-iteration Sinkhorn (VectorE `coldiv` + `rowdiv` 1.1 ms a
step) and hc_out. The Sinkhorn's output (`comb`) is first needed at the sublayer's hc_out, so it
can run entirely on the core that is idle during the attention / experts and hand over by
`sendrecv`; fuse hc_out of sublayer k with hc_in of sublayer k + 1 into one pass over the
SBUF-resident stream (W2); compute the mix GEMM from SBUF instead of reading the stream back from
HBM (`hc_mix_hbm`, 68 µs of ScalarE / SP a block on r2's core 1). Same arithmetic, so the risk is
schedule-only.

**W4 — Collectives at their floor (−20 to −30 µs a block, ~1 ms a step).** (a) Remove the overflow
loop's zero-trip cost (~8–10 µs a block): either a compiler-supported dynamic `if_else` (request
it: today it fails to compile) or a loop-free overflow scheme (e.g. a second static tier that only
issues DMAs past the active count and whose matmuls are masked). (b) Trigger the AR-64 right after
the last store: today it waits behind the next block's prefetch DMAs and loop bookkeeping in the
sync / GpSimd queues (~3–5 µs). (c) Cut the arrival spread (~9 µs median among 8 ranks): find
rank 63's +4 µs, keep per-rank work identical (the same slot counts, the same DMA queue split).
(d) Use the waits: issue the next block's cached-key gathers and transposes and the next
sublayer's weight loads inside the AR-8 / AR-64 waits (both cores idle ~15–30 µs there).

**W5 — Projections without HBM round trips (−30 to −40 µs a block).** The output projection takes
39 µs for 2 small GEMMs (wo_a: 4096 → 1024, wo_b: 1024 → 5120) because the cores split heads,
store `z` to HBM, barrier and reload before wo_b. Split instead so each core keeps its wo_a
columns and computes a wo_b partial that goes straight into the AR-8 source (the all-reduce sums
partials anyway), or exchange `z` by `sendrecv`. Fuse q_norm / kv_norm / rope into the PSUM
epilogues in the spread layout. The q / kv side (20 µs) gains the same way.

**W6 — Attention from pre-staged keys (−15 to −20 µs a block).** The key gathers and their PE
transposes (`key_tile_packed`, 20 µs of PE a core) depend only on the cache and the slot indices,
which are known before the block's attention for every block that reuses a top-k (all but the 8
index sources) and for the window part everywhere. Gather and transpose the cached keys at the
block's start (or in the previous AR-64 wait, W4d) and append only the step's T new rows at
attention time.

**W7 — MoE chain (−35 to −50 µs a block).** From the AR-8 end to the first expert matmul is 51
µs: hc (W2 / W3), the router GEMM (4.5 µs), and the routing chain (two ACT table switches, top-k,
group compaction, slot ids into registers, ~20 µs) before any expert weight DMA can start. Keep
routing on one engine without table switches (sqrt / softplus via the DVE, or one table), issue the
first pair's weight DMAs as soon as its id is known, and start the shared expert inside that wait.
Expert compute (54 µs) runs 8 static slots whatever the active count (~4.3 experts per group on
average): with W4a's tiering, only active slots run, which halves the PE time (`_up` is the step's
largest TensorE item, 0.83 ms). The tail's store / loop / barrier (19 µs) goes with W4a / W4b.

**W8 — Special blocks (−1.0 to −1.3 ms a step, beyond W2–W7).** The 8 index / KV-source / engram
blocks cost +65 to +415 µs each over a normal block (~1.9 ms a step). In `r1_idx`, the index top-k runs on GpSimd behind the slot-table gathers
queued on the same engine; the per-row index table runs on 6 partitions x 512 columns (spread it);
the score AR-8 (25–28 µs) has a ~10 µs gap after it. Re-queue the gathers, spread the table, move
the index-query projection to its own core, and give the KV compressor (src blocks) and the
engram read-out (1.6 MB AR-64, 52 µs) the W4d treatment.

**W9 — Drafter (−0.6 to −0.7 ms).** Into the step launch (W1: −0.1 ms of entry), its three stages
through W2–W7 (each ~360 µs today → ~200 µs), its prologue's all-gather + all-reduce (117 µs → ~60
µs: fold the token embedding into the main_x gather), packed activation staging for its GEMMs
(`interleave_rows` 118 µs), and the Markov head (5 x 33 µs: keep the Markov table replicated so the
per-token all-gather carries (max, index) only, ~5 x 18 µs).

### Order and milestones (*estimates*)

| Milestone | Work | step ms (in1024 bs1) |
|---|---|---:|
| today | — | 16.9 |
| M1 — schedule-only changes inside the current kernels | W4b, W4d, W6, W7 routing, W8 (queues, table), W9 head + staging | ~14.0 |
| M2 — block dataflow rebuilt | W2, W3, W5, W7 slot tiering (with W4a) | ~9.0 |
| M3 — one launch a step | W1, W9 into the launch, W8 compressor / engram | ~7.0 |

M1 is incremental (each item measurable in the existing per-launch workspaces, as in rounds 3–4).
M2 is a redesign of the block kernel's dataflow shared by all member families and the drafter; it is
where most of the time goes and should start with one family (`r1`) end to end in the L24 workspace.
M3 depends on compile-size and host-submission limits that need a feasibility test first (a 2-launch
step: target + drafter).

## 8. How each step is validated

Per change: the per-launch workspace (recorded inputs and golden outputs of all 64 ranks) for
correctness at the summation-order tolerance and for latency; then, for every version that is
benched, the integration test and the bench (bs1 and bs10: texts, TTFT, acceptance per cell against
the previous version). A fresh live profile like this one after each milestone (rank 0 + 8 ranks) to
re-measure the segments in section 3.2.

Open questions to settle early: SBUF headroom for the resident stream next to the weight buffers
(W2); whether the compiler can support a dynamic branch (W4a) or the tiering must stay loop-based;
NEFF size / compile time of a 40-block launch and the IO-tensor count (W1); the cause of rank 63's
longer attention side.

## Appendix — reproduce

```bash
# capture (rank 0, then 8 ranks): /mnt/nvme/ds41_profiling_opt12/scripts/chain12.sh
#   profile_cell.py --cell in1024_bs1 --isl 1024 --bs 1 --ms 1 --ctx 1280 --decode-window 0.6 --phases decode
# ingest + per-unit metrics + 10 us bins
/mnt/nvme/ds41_profiling_opt12/scripts/process.sh in1024_bs1_r0 "" 0 in1024_bs1_decode
# tables: tables6.phase_tables / util6.megakernel_table (report/tables_r0*.md), block anatomy:
python segments12.py in1024_bs1_decode 2 30 ; python milestones12.py in1024_bs1_decode r1 2 30
# per-function time / HBM: python hot6.py in1024_bs1_decode r1 --steps 2,...,30
# skew: python skew6.py in1024_bs1_mr_skew in1024_bs1_mr_decode_r{0,4,9,18,27,36,45,63}
```
