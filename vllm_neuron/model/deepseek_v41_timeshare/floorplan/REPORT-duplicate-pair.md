# Ranking three deployments of DeepSeek-V4.1-Flash on trn2-16device

## 0. First finding: there are two deployments, not three

`scheme-b.yaml` and `scheme-c.yaml` are **byte-identical apart from the name in the first
comment line**. Same `version`, same `target`, same `notes`, same `runtime`, same 271
placements in the same order with the same units, splits, collectives, tiers and stages. There
is no deployment difference between them at all, cosmetic or otherwise, so any ordering of the
two is a coin flip and I say so rather than inventing a reason.

The real question is scheme-a against scheme-b/c, and it is a narrow one. The two differ in
**exactly three things**, out of 271 placements and four runtime fields:

| | scheme-a | scheme-b / scheme-c |
|---|---|---|
| Δ1 · `layers.0/1.attention`, `mtp.0/1/2.attention` (5 of 43) | `head ×4 · batch ×2` | `head ×8` |
| Δ2 · `layers.1.engram`, `layers.14.engram` | `ngram ×16` | `ngram ×8 · batch ×2` |
| Δ3 · `runtime.prefill_chunk_tokens` | 96 | 80 |

Everything else — the 16-stage pipeline, the device pairing, `expert ×8` on all 43 MoE, `head ×4 ·
batch ×2` on the other 38 attention layers, `hidden ×4` on all 172 hyper-connection modules,
`vocab ×4`/`vocab ×8` on embed/lm_head, `tier: hbm_bank` with no tiering anywhere, 16 units per
Engram table, `decode_micro_batch: 4`, `pipeline_chunks: true`, no `overlap_collectives`
anywhere — is identical. Both schemes are the same floorplan at two adjacent search iterations.

## 1. Verdict

RANKING: scheme-b > scheme-c > scheme-a

**scheme-b (equivalently scheme-c) is the better deployment.** Of the three differences, the two
that are actually *placement* decisions both favour it, and the one that favours scheme-a is a
single runtime scalar that does not describe how the model sits on the machine and can be
retuned per workload in one line.

- **Δ1 is a strict win and it is correctly scoped.** `compress_ratios` in `config.json` is `0`
  for exactly layers 0, 1 and the three MTP stages, and `DSparkAttention.forward` asserts
  `compress_ratio == 0`, so these five layers are the only ones whose KV sweep is a fixed
  128-row ring at every context length. On them, and only on them, the head factor is free.
  scheme-b takes it; scheme-a leaves it.
- **Δ2 is right in the regime that matters for throughput** (many tokens per chunk) and very
  slightly wrong in the regime that matters for latency (batch 1, and all decode). Its capacity
  cost is real but harmless — see §3.4.
- **Δ3 is the only thing scheme-a wins on**, and it wins the four `prefill_8192` points with it.
  But `prefill_chunk_tokens` is not a floorplan property. Ranking a *deployment* above another
  on the strength of one integer in the `runtime:` block, when that deployment's placement is
  strictly worse on five attention layers, is the wrong trade.

The practical recommendation, which the ranking cannot express: **ship scheme-b's placements
with `prefill_chunk_tokens: 96`.** That combination beats both schemes on twelve of the sixteen
points and ties scheme-a on the other four.

Where the ranking is a judgement call: `prefill_8192_b32` (§5) is genuinely too close to call,
and the whole a-vs-b/c gap is small everywhere — these are two iterations of one design, not two
designs.

### A note on the `notes:` blocks

Both YAML files carry long `notes:` fields containing the search's own per-metric millisecond
deltas. I read them, because they are in the files I was handed. I have not ranked by them:
they are the output of the uncalibrated simulator this exercise deliberately withholds, and
several of the quantities they lean on (`weight_divisor`, `matmul_small_k`, scalar-engine
throughput) are not in `systems/` at all. Where my independent reading of the hardware and the
model source lands: I agree with Δ1 without reservation, I agree with Δ2's *direction* only at
batch ≥ 4, and I think Δ3 is a metric-shaping move rather than a hardware improvement — the
notes themselves concede it is a wash in absolute milliseconds (`21319 ms` against `21262 ms`
summed over the eight prefill points) and only looks like a win because each metric is
normalised separately.

## 2. How the winning scheme deploys the model

### 2.1 The pipeline: one stage per device

Sixteen stages, `stage: 0` .. `stage: 15`, one per device, device `d`*i* owning stage *i*. The
43 blocks (40 backbone + 3 MTP) are dealt out 2–3 per stage:

| stage/device | layers | also |
|---|---|---|
| 0 / d0 | 0, 1, 2 | `embed`, `hc_expand`, `other_globals`, `layers.1.engram` |
| 1 / d1 | 3, 4, 5 | |
| 2 / d2 | 6, 7, 8 | |
| 3 / d3 | 9, 10 | |
| 4 / d4 | 11, 12, 13 | |
| 5 / d5 | 14, 15, 16 | `layers.14.engram` |
| 6 / d6 | 17, 18 | |
| 7 / d7 | 19, 20, 21 | |
| 8 / d8 | 22, 23, 24 | |
| 9 / d9 | 25, 26 | |
| 10 / d10 | 27, 28, 29 | |
| 11 / d11 | 30, 31, 32 | |
| 12 / d12 | 33, 34 | |
| 13 / d13 | 35, 36, 37 | |
| 14 / d14 | 38, 39, mtp.0 | `lm_head` |
| 15 / d15 | mtp.1, mtp.2 | `final_norm`, `hc_collapse`, `mtp.2.norm/markov_head/confidence_head` |

`pipeline_chunks: true`, so stage *i* may start chunk *n*+1 before stage *i*+1 finishes chunk
*n*; `decode_micro_batch: 4` splits a decode batch into groups of four that traverse the
pipeline independently.

On the 4×4 torus (device *i* at row *i*//4, column *i*%4 — this reproduces the `PLATFORM.md` hop
table exactly, e.g. d0→d3 = 1 by column wrap, d0→d10 = 4) the stage chain pays one hop for
twelve transitions and two hops for the three row wraps d3→d4, d7→d8, d11→d12: **18 hops per
token traversal**.

### 2.2 The intra-stage layout: XOR-1 device pairs

Every heavyweight module is split 8 ways across its home device's four logical NeuronCores and
the four of the home device's **XOR-1 partner** (d0↔d1, d2↔d3, … d14↔d15). Every one of those
eight pairs is **1 torus hop** — the pairing is by low bit, which keeps both devices in the same
torus row and adjacent in column. Per text constraint 5 a group pays the worst hop distance in
the group on every collective step, and 1 is the minimum for any group spanning more than one
device.

The unit list is interleaved home/partner — `layers.2.attention` is
`[d0.l0, d1.l0, d0.l1, d1.l1, d0.l2, d1.l2, d0.l3, d1.l3]` — and this is load-bearing. With the
first-listed split as the outer axis, shard index = `head_idx·2 + batch_idx`, so **each batch
half's four head shards are the four logical cores of one device**: batch half 0 lives entirely
on d0, half 1 entirely on d1. The expensive collective (the fp32 head all-reduce) therefore
never leaves the chip — 803 GB/s, 1 µs, per constraint 3 — and the split that crosses the
device boundary is the `batch` one, which needs no collective at all.

Everything small is kept strictly inside one device: all 172 `hc_*` modules, the norms, the MTP
heads and `other_globals` are `hidden ×4` (or `vocab ×4`) on the four logical cores of a single
device. That is 172 of 271 placements that never touch the torus.

**Nothing in either scheme uses the `.p0`/`.p1` level.** Every shard is a whole logical
NeuronCore. That is correct, not an oversight: per constraint 2 the pair shares one address
space so splitting across it is collective-free, but it adds no SBUF, and the one thing you
might want it for — running Engram's GPSIMD row gather on one physical core while the Tensor
Engine runs `wkv` on the other — is exactly what constraint 1 forbids, since the 56 MiB SBUF is
the pair's.

### 2.3 Where the weights live

**Every one of the 271 placements is `tier: hbm_bank`, fully resident.** No `resident_fraction`,
no `cache_tier`, no `hit_rate`, nothing in `device_hbm`, `peer_hbm`, `host_dram` or `nvme`.
Both Engram tables — 94.56 GiB each, the two modules `MODEL.md` says cannot fit a bank — are
split 16 ways and held entirely in HBM banks. `layers.1.engram` occupies all sixteen banks of
d0–d3 (torus row 0, a 4-cycle of diameter 2); `layers.14.engram` occupies all sixteen of d4–d7
(row 1). Each group deliberately contains the device that consumes the result, so the
`allgather` rejoin lands the output where layer 1 / layer 14 needs it.

Resulting per-bank occupancy (weights only). `d0.l0` carries `layers.1.engram` at
101.5 GB / 8 = 11.82 GiB, plus a shard of six MoE layers — layers 0/1/2 homed on d0 and 3/4/5
homed on d1, all eight-way on the same unit set — at 6.76 GiB / 8 = 0.845 GiB each = 5.07 GiB,
plus six attention shards (~0.09 GiB) and `embed` (0.15 GiB): **≈17.2 GiB of 24**. The banks of
d8–d15, which carry no Engram, sit at **4–5 GiB of 24**, and d14/d15 at under 3 GiB. Total
charged across the instance is ≈668 GiB of 1536 GiB.

## 3. Why it is the best

### 3.1 The first limiting factor, per regime

Both schemes hit the same wall in the same order; the question is only how hard.

**Decode, any batch: the 725 GB/s bank, multiplied by 16 stages of serialisation.** A decode
token visits all sixteen devices in sequence, and each stage must pull its layers' weights out
of its own bank. The composition of that read is the surprise, and it is the key fact in this
whole report:

- One MoE shard at batch-1 decode reads almost nothing. 6 of 384 experts fire, so one of the
  eight expert shards holds on average 0.75 live experts of its 48: 0.75 × 3 × 5120 × 2304 at
  fp4 = 13.3 MB, plus the shared expert (whose `moe_inter_dim` the expert factor also divides,
  2304/8 = 288) at 4.4 MB, plus the gate. **≈22 MB per ffn shard.**
- One attention shard at `head ×4` reads the head-splittable weights `wq_b` (41.9 MB),
  `wo_a` (33.6 MB) and `wo_b` (41.9 MB) quartered to 29.4 MB, **plus `wq_a` and `wkv`
  replicated in full** — 9.2 MB that no head split touches, because this model has a single
  512-wide latent KV head shared by all 64 query heads. **≈38.6 MB per attention shard.**

So on `d0.l0`, six attention shards read ~232 MB against six ffn shards' ~132 MB. **Attention,
at 1.1% of the model's parameters, is about 64% of the per-bank weight read at decode.** The
MoE's 277 GiB is almost entirely cold; the attention's 5.2 GiB is entirely hot, every layer,
every token.

That is why Δ1 is the most valuable of the three differences and why the head factor is the
single most important number in this floorplan. It also bounds what is achievable: at `head ×8`
a shard still reads 14.7 + 9.2 = 23.9 MB, of which the irreducible replicated latent
projections are 39%. **The winner's decode is within roughly 40% of the floor set by `wq_a` and
`wkv`**, and per §4.1 the head factor cannot go past 8 without breaking `o_groups`. There is no
large win left here.

**Decode at higher batch: pipeline fill.** `decode_micro_batch: 4` puts batch/4 micro-batches in
flight through a 16-stage pipeline. At batch 1 there is one item and fifteen sixteenths of the
machine is idle; at batch 32 there are eight, so the pipe is still only half full. Depth is a
real cost at every decode point in both schemes, worst at batch 1.

**Prefill at 128 tokens: pipeline fill, and nothing else.** Two chunks through sixteen stages is
seventeen stage-slots for two chunks' worth of work. Fifteen of them are bubbles. This is the
one regime where chunk size is pure gain as it shrinks, which is what Δ3 exploits.

**Prefill at 8192 tokens: chunk count × the slowest stage.** 850 MB per expert shard cannot live
in 56 MiB of SBUF, so every chunk re-streams each stage's full expert working set through its
banks: ~2.5 GB per bank per chunk for a three-layer stage, ~3.5 ms at 725 GB/s. Total prefill
≈ `n_chunks × t_slowest`, and **the slowest stages are 0 and 5** — the only two carrying three
layers *and* a 94.56 GiB Engram lookup. That asymmetry is the deployment's real structural
weakness and both schemes have it identically.

### 3.2 Why Δ1 is free, and why it is only free on five layers

`Attention.forward` concatenates a sliding window of raw KV with `index_topk` compressed
positions, and there is one latent KV per position shared by all 64 query heads. So the total
KV bytes a group sweeps is `head_factor × batch_per_shard × kv_rows`: **the head split
replicates the KV sweep and the batch split divides it.** That makes the head factor the one
knob that trades the weight roofline against the KV roofline.

For the 38 layers with `compress_ratio ≥ 1`, `kv_rows` grows with context: at 8192 tokens a
ratio-1 layer holds 8192 compressed latents plus the 128-row window, so at batch 32 a
`head ×4 · batch ×2` shard sweeps 16 × 8320 × 512 B ≈ 68 MB and `head ×8` would sweep 136 MB —
paying +68 MB to save 14.7 MB of weight. `head ×4` is right there, and both schemes use it.

For layers 0, 1 and the three MTP stages, `compress_ratios` is 0. Their entire KV is the
128-slot ring buffer: 64 KiB per batch member, **identical at 128 and 8192 tokens**. Doubling
that to save 14.7 MB of weight stream per shard is not a trade, it is arithmetic. And at batch 1
the `batch ×2` leg was dividing nothing at all while leaving four of eight logical cores with
zero samples. scheme-b applies `head ×8` to precisely this set. I checked the set against
`config.json` and `DSparkAttention`'s `assert self.compress_ratio == 0`; it is exactly right.

The one cost: at `head ×8` the single head group spans both devices of the pair, so the fp32
output all-reduce moves off the 803 GB/s intra-device path onto a 1-hop torus link at
1024 GB/s. Payload is `tokens × 5120 × 4 B`, so this is invisible at decode (one token) and a
few tens of microseconds per chunk at batch 32 prefill. It is the reason Δ1 shows a faint
negative sign on the largest prefill point and a strong positive one everywhere else.

### 3.3 Why Δ2 is a prefill optimisation that mildly hurts decode

`Engram.forward` (`modules/01-Engram/source.py`) is three stages with different parallel
behaviour:

1. **Gather.** 24 hash columns (`(max_ngram_size−1) × n_heads = 3 × 8`) × 256 fp8 + scales =
   6.3 KB per token out of a 384-million-row table. An `ngram` split divides both the table and
   the gather count per core — which matters beyond bytes, because per constraint 1 the GPSIMD
   gather cannot overlap the Tensor Engine on the same logical core.
2. **`wkv`.** `Linear(24 × 256 = 6144 → dim × (hc_mult+1) = 25600)`, 157.3 MB fp8. The `ngram`
   split divides its contraction axis, so a shard holds 157.3/8 = 19.7 MB (scheme-b) or
   157.3/16 = 9.8 MB (scheme-a). **This is a dense read, paid per chunk regardless of token
   count.**
3. **Gate and residual add.** `rsqrt(h²·mean)·rsqrt(key²·mean)`, the weighted dot, the signed
   sqrt, the sigmoid, and `h + gate·value` — all fp32 over `[B, L, 4, 5120]`, ≈330 KB of fp32
   traffic per token on the scalar/vector engines. **This is strictly token-proportional and an
   `ngram` split does not touch it at all**, because it happens after the rejoin. Only `batch`,
   `seq` or `hidden` divide it.

So the two schemes sit on opposite sides of one crossover. Scheme-a's `ngram ×16` minimises the
dense `wkv` read (9.8 MB, ≈13.6 µs at 725 GB/s) and leaves the gate fully replicated on sixteen
shards. Scheme-b's `ngram ×8 · batch ×2` halves the gate at the cost of +9.9 MB of dense read
(≈+13.6 µs). Break-even is around 60 tokens per shard per chunk:

- **Decode**: tokens per step = batch ≤ 32, so the gate saving is at most ~7 µs against a
  +13.6 µs dense read. **scheme-a's Engram split is better at every one of the eight decode
  points.** It is simply outweighed by Δ1.
- **Batch 1**: `batch ×2` divides nothing, so scheme-b pays the doubled read for no return, at
  both context lengths and in both phases.
- **Prefill at batch ≥ 4**: 320–2560 tokens per chunk, and the gate is the largest
  token-proportional term in the file that no other split divides. Halving it shortens stages 0
  and 5 — which §3.1 identifies as the pipeline's critical stages. This is a real win, and it
  lands on the right stage.

### 3.4 The capacity cost of Δ2 is real and does not matter

`batch` is not a weight-partition dimension, so a batch-replicated eighth of the table is
charged honestly: 94.56/8 = 11.82 GiB on each of sixteen banks, 189 GiB per table, 378 GiB for
the two, against scheme-a's 189 GiB. Scheme-b spends an extra **189 GiB, 12.3% of the
instance's memory**, and it spends it on exactly the 32 banks that are already the fullest.

This is worth naming and then dismissing. The instance holds 1536 GiB; the model is 474.5 GiB.
Peak bank occupancy is 17.2 GiB of 24 in scheme-b and 11.3 GiB in scheme-a, while 32 banks on
d8–d15 sit at 4–5 GiB. **The 24 GiB bank limit is not binding in either scheme and HBM is not
the scarce resource here** — a point reinforced by how small the KV cache turns out to be in
this architecture. A 128-slot sliding window at `head_dim` 512 is 64 KiB per layer per batch
member; only the four `kv_source_layers` (2, 8, 14, 20) hold a compressed cache at all, and
`index_topk: 512` caps what any query reads from it. Even at 8192 tokens and batch 32 the whole
KV state is on the order of a gigabyte across the instance. The "batch multiplies the KV cache"
pressure that would normally decide this question is largely designed out of the model.

So text constraint 8's framing — Engram residency in HBM "costs 12% of the instance's memory for
an almost entirely cold table" — is true and irrelevant: there is nothing else that wants the
memory. Both schemes are right to keep the tables in HBM rather than pay `host_dram`'s 13 GB/s
and 2 µs on a scattered, data-dependent lookup on the critical path of layers 1 and 14, and
scheme-b's doubling of that footprint costs nothing real.

### 3.5 Why Δ3 is the wrong direction for long prefill

With `n = ceil(T/c)` chunks through a 16-stage pipeline, latency ≈ `(n + 15) · (a + τc)`, where
`a` is the per-chunk stage-fixed cost (dominated by re-streaming the expert weights) and `τc` is
the chunk's token-proportional work. Expanding gives `aT/c + τT + 15a + 15τc`: the term that
moves is `aT/c`, and **shrinking `c` inflates it**.

- At T = 8192: 8192/80 = 103 chunks against 8192/96 = 86. Seventeen extra full sweeps of every
  stage's expert weights through a 725 GB/s bank, ~3.5 ms each on the critical stage. Smaller
  chunks are better only when `15τc² > aT`, i.e. only at batch well above 32 on these numbers.
  **Scheme-a is right at 8192, at every batch size.**
- At T = 128: both settings give two chunks, so `n` is fixed and the only surviving term is
  `15τc` — the pipeline fill, fifteen stage-slots of it. 80 + 48 beats 96 + 32 by
  `15τ(96−80)`. **Scheme-b is right at 128, at every batch size.**

This is a clean phase-dependent trade with no placement content. It is the sort of thing a
serving stack decides per request-length bucket, which is why I decline to let it decide the
deployment ranking.

## 4. Recipes for every module the winner decomposes

Convention throughout: `len(units)` = product of the split factors; the first-listed split is
the outer axis, so shard index = `outer_idx × inner_factor + inner_idx`, and shard *i* lands on
`units[i]`. Every shard is a whole **logical NeuronCore** (`d<n>.l<m>`); no placement uses the
`.p0`/`.p1` physical-core level.

### 4.1 Attention, `head ×8` — 5 modules (`layers.0`, `layers.1`, `mtp.0/1/2`)

Units: home device and XOR-1 partner interleaved, e.g. `layers.0.attention` on
`[d0.l0, d1.l0, d0.l1, d1.l1, d0.l2, d1.l2, d0.l3, d1.l3]`. Collective: `allreduce`.

Shard *i* takes query heads `8i … 8i+7` of 64. Concretely:

- **`wq_b`** — `ColumnParallelLinear(1280 → 64·512 = 32768)`. Split the **output** axis:
  shard *i* holds rows `[4096i : 4096(i+1)]`, a `[4096, 1280]` fp8 block (5.24 MB). The output
  slice is already complete; no exchange.
- **`attn_sink`** — `[64]` fp32. Shard *i* takes entries `8i … 8i+7`.
- **`wo_a`** — `ColumnParallelLinear(64·512/8 = 4096 → 8·1024 = 8192)`, used as a
  block-diagonal einsum reshaped `[n_local_groups, 1024, 4096]`. At `head ×8`,
  `n_local_groups = o_groups/8 = 1`: **shard *i* owns exactly output group *i***, the
  `[1024, 4096]` block that projects its own eight heads. **This is the hard ceiling on the head
  factor: `o_groups` is 8, so `head ×16` would ask for half a group and is not expressible.**
- **`wo_b`** — `RowParallelLinear(8192 → 5120)`. Split the **reduction** axis: shard *i* holds
  `[5120, 1024]` fp8, columns `[1024i : 1024(i+1)]`, and produces a partial sum.
- **`wq_a` (5120→1280), `q_norm`, `wkv` (5120→512), `kv_norm`, `freqs_cis`** — **replicated on
  all eight shards.** Each shard computes the whole 512-wide latent KV and writes the whole
  128-slot `window_kv_cache`. This replication is what the `head` split cannot divide, and it is
  both the reason the head split saturates and the reason it is free on these five layers (the
  replicated sweep is 128 rows, not a growing cache).
- **Rejoin:** fp32 `all_reduce` of `wo_b`'s `[tokens, 5120]` partial sums — 20.5 KB per token —
  **after** the attention output and before `hc_attn_out`. Group = all eight units, spanning two
  devices **1 torus hop** apart; per constraint 5 the whole collective is priced at 1 hop.

### 4.2 Attention, `head ×4 · batch ×2` — the other 38 modules

Same units and same weight recipe with `head_factor` 4 (shard *i* holds heads `16i … 16i+15`,
`n_local_groups = 2`, `wo_b` holds `[5120, 2048]`).

The **batch** leg has `collective: none` — data parallelism needs no exchange. Batch half *b*
runs samples `[b·B/2 : (b+1)·B/2]`, keeps its own `window_kv_cache`, `compress_kv_cache` and
indexer `k_cache` slices, and never talks to the other half.

The placement of the two legs is the good part: shard = `head_idx·2 + batch_idx`, so head group
*g* sits on `d_home.l_g` and `d_partner.l_g`, and **batch half 0 is the four logical cores of
the home device while batch half 1 is the four cores of the partner.** The head all-reduce
therefore runs *within one device* — 803 GB/s, 1 µs, on-chip per constraint 3 — and the torus is
crossed only by the collective-free batch split. At batch 1 the odd shards hold no samples and
sit idle; that is the defect Δ1 removes on the five layers where it can.

### 4.3 MoE, `expert ×8` — all 43 `*.ffn`

Units: the same interleaved 8-core pair set, home device first. Collective: `all_to_all`.

- **Routed experts.** 384 experts (128 for the MTP stages) dealt out 48 per shard; shard *i*
  owns experts `48i … 48i+47`, each complete — `w1 [2304, 5120]`, `w3 [2304, 5120]`,
  `w2 [5120, 2304]`, fp4 with per-32 `ue8m0` scales, 17.7 MB per expert, 850 MB per shard. The
  expert axis is *not* an axis of any single weight; it selects whole `Expert` modules.
- **`gate`** — `[384, 5120]`, replicated: every shard scores every expert so it knows what it
  owns. `score_func: sqrtsoftplus`, `topk = 6`, with `bias` steering selection and the unbiased
  scores supplying the weights.
- **Shared expert** — one `Expert(5120, 2304)` that every token traverses. Here the expert
  factor is used as tensor parallelism over its intermediate dimension: shard *i* holds
  `w1/w3` rows `[288i : 288(i+1)]` and the matching `w2` columns, i.e. `moe_inter_dim/8 = 288`.
  This is the one place where `expert ×8` genuinely divides a weight matrix rather than a module
  list, and it is why doubling the expert factor halves the shared expert's read as well.
- **Communication, in order:** `gate` locally → an **all-to-all dispatch** sending each token's
  hidden state (`tokens × 6 × 5120` bf16) to the shards owning its six experts → local expert
  GEMMs → an **all-to-all combine** returning the weighted outputs → local fp32 accumulation
  and the shared-expert partial sum. Group = eight units over two devices, **1 hop**.
- **Why 8 and not 16.** `expert ×16` would halve the per-shard weight again, but sixteen units
  means four devices, and on a 4×4 torus the nearest four-device group has diameter 2 — every
  all-to-all step then pays two hops instead of one. Eight units on an XOR-1 pair is the largest
  expert factor available at the minimum group diameter. Both schemes get this right.
- **Engine note.** Per constraint 1 the dispatch gather runs on GPSIMD and cannot overlap the
  Tensor Engine on the same logical core, so the gather and the expert GEMM it feeds serialise.
  At batch-1 decode each shard averages 0.75 live experts, making its GEMM a `K=5120, M≈1,
  N=288` sliver — deep in the `matmul_small_k` regime, far below the 0.181/0.410 achieved
  fractions. Decode-time MoE cost here is gather plus dequantisation plus launch, not FLOPs.

### 4.4 Engram, `ngram ×8 · batch ×2` — `layers.1.engram`, `layers.14.engram`

Units: all sixteen logical cores of one torus row, listed device-major —
`[d0.l0..d0.l3, d1.l0..d1.l3, d2.l0..d2.l3, d3.l0..d3.l3]` for layer 1, d4–d7 for layer 14.
Collectives: `allgather` on the `ngram` leg, `none` on the `batch` leg.

- **`embed.weight` / `embed.scale`** — `[384006168, 256]` fp8 plus `[384006168, 8]` scales,
  94.56 GiB. The `ngram` leg splits the **row** axis eight ways: row shard *g* holds rows
  `[g·48000771 : (g+1)·48000771]`, 11.82 GiB. The `batch` leg **replicates** those rows, so both
  of shard *g*'s units hold the same 11.82 GiB — which is why the plan is charged 189 GiB per
  table, honestly, rather than pretending a batch split divides storage.
- **`wkv`** — `Linear(6144 → 25600)` fp8, 157.3 MB. Its contraction axis *is* the 24 looked-up
  rows × 256 dims, so the row split cuts it on exactly the same boundary: shard *g* holds
  `[25600, 768]`, 19.7 MB, and computes a partial sum over its own rows. This is the term that
  scheme-a's `ngram ×16` halves.
- **`q_weight`, `k_weight`** — `[4, 5120]` each, replicated.
- **Placement, by level.** Shard = `ngram_idx·2 + batch_idx`, so with the device-major unit list
  row shard *g*'s two batch replicas land on **two logical cores of the same device**
  (`d0.l0`/`d0.l1` for *g*=0, `d0.l2`/`d0.l3` for *g*=1, and so on) — which is harmless, since
  the replicas never communicate. The consequence is on the collective: **each batch half's
  eight `ngram` participants are spread across all four devices of the row**, so the allgather
  is priced at the row's diameter of **2 hops**. See §6 for the free fix.
- **Communication, in order:** per-token gather of this shard's share of the 24 hash rows
  (GPSIMD; ~3 rows per token per shard) → dequantise with the per-32 scales → local partial
  `wkv` → **rejoin across the eight `ngram` shards** of this batch half, payload
  `tokens × 25600` → split into `key [4, 5120]` and `value [5120]` → the replicated fp32 gate
  and residual add. Everything after the rejoin is replicated across the eight row shards; only
  the `batch` leg divides it.
- **Why not `hidden`.** `hidden` would divide the gate too, but it does not partition the
  table's rows, so declaring it would under-charge residency two-fold. `seq` divides the gate as
  well and unlike `batch` it also divides at batch 1 — but it needs a second collective that
  `batch` gets for free. `batch` is the honest choice if you accept the batch-1 blind spot.

### 4.5 `embed`, `lm_head`

- **`embed`** — `vocab ×4`, `allreduce`, on `d0.l0..l3` (one device, intra-device link).
  `ParallelEmbedding` gives shard *i* rows `[32320i : 32320(i+1)]` of `[129280, 5120]`; ids
  outside the shard's range are zero-masked and the all-reduce sums the one real row per token.
- **`lm_head`** — `vocab ×8`, `allgather`, on `d14.l0..l3 + d15.l0..l3` (1 hop). `ParallelHead`
  gives shard *i* rows `[16160i : 16160(i+1)]` of a `[129280, 5120]` fp32 weight; the allgather
  concatenates the logit slices along the vocabulary axis. `full_logits` defaults to false, so
  at decode this is one position per sample.

### 4.6 The 172 hyper-connection modules and the norms, `hidden ×4`

`hc_attn_in`, `hc_attn_out`, `hc_ffn_in`, `hc_ffn_out` for all 43 blocks, plus `hc_expand`,
`hc_collapse`, `final_norm`, `other_globals`, `mtp.0.main_norm`, `mtp.0.main_proj`,
`mtp.2.norm`, `mtp.2.markov_head`, `mtp.2.confidence_head`. All `hidden ×4`, `allreduce`, on the
four logical cores of one device — always the device that owns the layer.

Shard *i* holds `fn` columns `[5120i : 5120(i+1)]` of `[24, 4·5120]` fp32 (`mix_hc = (2+hc_mult)
·hc_mult = 24`) and the corresponding quarter of the flattened `hc_mult × dim` stream. Two tiny
all-reduces per module: the `[tokens]` sum of squares for the rsqrt and the `[tokens, 24]`
`mixes` partial sums. `hc_pre`'s reduction is over the `hc` axis, not `dim`, so it needs no
exchange. `hc_split_sinkhorn` then runs 20 iterations on a `[tokens, 4, 4]` fp32 matrix,
replicated — it cannot be divided by `hidden`, and with 86 sublayers that is 1720 serially
dependent normalisation steps per token on the vector engine, a per-token floor both schemes
share.

Keeping all 172 of these on-device is the quietly correct decision in this floorplan: the
payloads are kilobytes, the collectives are frequent, and at 803 GB/s and 1 µs on-chip they
never touch the torus.

## 5. Per-configuration rankings

RANKING[prefill_128_b1]: scheme-b > scheme-c > scheme-a
RANKING[prefill_128_b4]: scheme-b > scheme-c > scheme-a
RANKING[prefill_128_b8]: scheme-b > scheme-c > scheme-a
RANKING[prefill_128_b32]: scheme-b > scheme-c > scheme-a
RANKING[prefill_8192_b1]: scheme-a > scheme-b > scheme-c
RANKING[prefill_8192_b4]: scheme-a > scheme-b > scheme-c
RANKING[prefill_8192_b8]: scheme-a > scheme-b > scheme-c
RANKING[prefill_8192_b32]: scheme-a > scheme-b > scheme-c
RANKING[decode_128_b1]: scheme-b > scheme-c > scheme-a
RANKING[decode_128_b4]: scheme-b > scheme-c > scheme-a
RANKING[decode_128_b8]: scheme-b > scheme-c > scheme-a
RANKING[decode_128_b32]: scheme-b > scheme-c > scheme-a
RANKING[decode_8192_b1]: scheme-b > scheme-c > scheme-a
RANKING[decode_8192_b4]: scheme-b > scheme-c > scheme-a
RANKING[decode_8192_b8]: scheme-b > scheme-c > scheme-a
RANKING[decode_8192_b32]: scheme-b > scheme-c > scheme-a

scheme-b and scheme-c are the same file, so their relative order is arbitrary in all sixteen.

**`prefill_128_b1`** — b/c. Two chunks either way, so `n_chunks` is fixed and the only term that
moves is the 15-stage pipeline fill: 80 + 48 beats 96 + 32. Δ1 compounds it, because at batch 1
`batch ×2` leaves four of eight cores empty on five layers while `head ×8` uses all eight and
halves their weight stream. Δ2 is a small negative (doubled `wkv` read, no gate division at
batch 1) and loses.

**`prefill_128_b4` / `_b8` / `_b32`** — b/c, by widening margins. Same fill argument, and now
Δ2 turns positive: 320 to 2560 tokens per chunk make Engram's replicated fp32 gate the largest
token-proportional term no other split divides, and halving it shortens stages 0 and 5, which
§3.1 identifies as the pipeline's critical stages. Small offset at b32: `head ×8` moves five
layers' all-reduce off the 803 GB/s on-chip path onto a 1-hop torus link.

**`prefill_8192_b1`** — a, decisively; this is scheme-a's best point. 103 chunks against 86, and
every chunk re-streams each stage's ~2.5 GB of fp4 expert weights through a 725 GB/s bank
because 850 MB per expert shard cannot stay in 56 MiB of SBUF. Twenty percent more of the
dominant term, with only 80 tokens per chunk to amortise it. Δ2 contributes nothing at batch 1
and costs a doubled `wkv` read paid 103 times.

**`prefill_8192_b4` / `_b8`** — a, narrowing. The chunk penalty is unchanged in absolute terms
but now amortised over 320–640 tokens per chunk, and Δ2's gate saving exists. Still not enough:
seventeen extra full weight sweeps of the critical stage outweigh halving one vector pass.

**`prefill_8192_b32`** — too close to call; I pick a. On stage 0 the two effects are the same
size — halving Engram's gate saves roughly what seventeen extra chunks of expert-weight
streaming costs, and my estimates of the two differ by less than their own error. I break the
tie toward a because the chunk-count term is bounded by numbers that are in the datasheet
(725 GB/s bank bandwidth, fp4 expert sizes) while the gate term depends on scalar-engine
throughput that is not, and I would rather be wrong on the quantity I cannot check. If Δ2 is
larger than I think, this point flips and scheme-b sweeps twelve of sixteen into thirteen.

**`decode_128_b1`** — b/c. Attention is ~64% of the per-bank weight read at decode (§3.1);
halving it on five of 43 layers and filling four idle cores beats the +13.6 µs per Engram layer
that Δ2 costs. Δ3 does not apply to decode.

**`decode_128_b4` / `_b8` / `_b32`** — b/c, same reason, margin narrowing slightly as batch
amortises the attention weight read and as the pipeline fills (8 micro-batches of 4 at batch
32). Δ2 remains a small negative at every decode point: tokens per step is at most 32, so
Engram never leaves the dense-read-bound regime where `ngram ×16` is the better split.

**`decode_8192_b1` … `_b32`** — b/c. Δ1 is context-independent by construction: the five layers
it touches have `compress_ratio 0`, so their KV sweep is the same 128-slot ring at 8192 as at
128, while every layer whose sweep does grow keeps `head ×4` in both schemes. Nothing about
8192 tokens changes the sign of any of the three differences at decode.

## 6. What I would check first

**On the datasheet's soft numbers**

1. **Every inter-device figure is an extrapolation.** `PLATFORM.md`'s provenance section says
   `links.inter_device.bandwidth_bytes_per_s`, `per_hop_latency_us` and *all collectives wider
   than one device's four logical cores* are unmeasured — this host has one Trainium2. That is
   the entire basis for pricing the 8-unit MoE all-to-all, the 8-unit attention all-reduce and
   the 16-unit Engram allgather. It is also why the `expert ×8` / `head ×4` / 1-hop-pair design
   looks as good as it does: if inter-device bandwidth is materially below 1024 GB/s or the
   per-hop latency above 1 µs, the correct answer shifts toward smaller, strictly on-device
   groups and a different expert factor. This is the single largest uncertainty in the ranking,
   and it affects both schemes identically, so it does not change a-vs-b/c — it changes whether
   either is right.
2. **`links.intra_device` is derived, not measured** — 2.9e12 B/s × a measured DMA efficiency of
   0.2770, with NKI 0.6.0 having no collective primitive and the XLA all-reduce failing to
   compile (`NCC_ISMP902`). Every one of the 172 `hidden ×4` placements and every head
   all-reduce in the 38 `head ×4 · batch ×2` layers is priced against that derivation. §4.2's
   argument that the batch/head leg ordering keeps the expensive collective on-chip is only
   worth what the on-chip number is worth.
3. **The `matmul_small_k` path.** `PLATFORM.md` calls it "the single most important cost-model
   fact for MoE" and gives no figure. At batch-1 decode each expert shard runs a `K=5120, M≈1,
   N=288` GEMM; the whole decode picture depends on a number I cannot see.

**On assumptions the schemes make**

4. **`hit_rate` is not an issue here, and that is worth stating plainly.** All 271 placements in
   both schemes are `tier: hbm_bank` with no `resident_fraction`, so neither makes a locality
   claim about the 384-million-row n-gram tables. There is no hit-rate assumption to justify.
   Given that the instance holds 1536 GiB against 474.5 GiB of weights and the fullest bank in
   the winner is at 17.2 of 24 GiB, I think that is the right call — but it is worth confirming
   that no workload's activation high-water mark pushes the d0–d7 banks, since those are
   carrying 11.82 GiB of Engram each and have only ~7 GiB of headroom. Scheme-a would have
   ~13 GiB.
5. **The split-composition convention.** My §4 recipes assume the first-listed split is the
   outer axis (shard = `outer·inner + inner`). The interleaved unit lists in the attention
   placements only make sense under that reading — they are exactly what you write to keep each
   batch half's head all-reduce inside one device — so I am confident, but the Engram
   conclusions in §4.4 depend on it and it is a one-line thing to verify.
6. **`wo_a`'s dtype accounting.** `Attention.__init__` declares `wo_a` as bf16 and the source
   comment says "convert.py dequantizes it to bf16; an fp8 grouped GEMM would halve the memory",
   yet the module's `param_bytes` (126,753,280 for a plain backbone layer) only reconciles if
   `wo_a` is counted at its fp8 checkpoint size. If the runtime really holds it in bf16, every
   attention module costs ~33 MB more than the plan is charged — 1.4 GiB across the instance.
   Small, but it is an accounting gap rather than a modelling choice.

**Free improvements to the winner, in decreasing confidence**

7. **Reorder the Engram unit lists.** As written, each batch half's eight `ngram` participants
   are spread across all four devices of the torus row, so the allgather is priced at the row's
   diameter of 2 hops. Interleaving the list as
   `[d0.l0, d2.l0, d0.l1, d2.l1, d0.l2, d2.l2, d0.l3, d2.l3, d1.l0, d3.l0, …]` puts batch half 0
   entirely on {d0, d1} and half 1 entirely on {d2, d3}, giving each allgather a diameter of
   **1 hop**. Identical residency, identical work, strictly shorter collective. This is the
   cheapest thing in this list.
8. **Declare `overlap_collectives: true` on the two Engram placements.** Per constraint 4 nothing
   overlaps unless declared, and nothing is declared anywhere in either scheme. The global sweep
   reportedly measured 0.05–0.15%, but a global average hides the one place it should matter:
   `layers.1.engram` spans d0–d3, which in steady-state pipelined prefill are simultaneously
   running stages 1, 2 and 3 on earlier chunks. Its allgather currently serialises against three
   other stages' compute. Test the two Engram placements in isolation rather than trusting the
   271-placement average.
9. **Sweep `decode_micro_batch` downward.** The notes record 5, 6 and 8 being tried and
   rejected; 1, 2 and 3 do not appear. At batch 32 with micro_batch 4 there are eight items in a
   sixteen-stage pipeline — half empty. Micro_batch 2 would fill it, at the cost of smaller
   per-stage GEMMs. Given that §3.1 puts decode firmly on the weight-streaming roofline rather
   than on FLOPs, smaller micro-batches may be nearly free. Untested in the direction that
   matters.
10. **Relabel the pipeline as a boustrophedon.** Assigning stages to devices in the order
    0,1,2,3,7,6,5,4,8,9,10,11,15,14,13,12 makes all fifteen stage transitions 1 hop instead of
    twelve at 1 and three row-wraps at 2 — 15 hops per token traversal instead of 18. It
    preserves every XOR-1 pair and leaves both Engram device sets unchanged. Worth ~3 µs per
    traversal: negligible, but it costs nothing.
11. **Rebalance the pipeline against Engram, not against layer count.** Stages 0 and 5 carry
    three layers *and* a 94.56 GiB Engram lookup while d14/d15 hold under 3 GiB of weights each
    and 32 banks on d8–d15 sit at 4–5 of 24 GiB. In a 16-stage pipeline the slowest stage sets
    the steady-state rate for long prefill, and it is one of those two. Giving stages 0 and 5
    two layers instead of three, and spreading each Engram table over 32 banks rather than 16
    (halving `wkv` again and the per-core gather count), is the largest structural improvement
    available to either scheme. It is not free — a 32-way group spans eight devices and pays
    more hops on the rejoin, and the notes record wider groups losing badly for the *MoE* — but
    Engram's rejoin payload and access pattern are nothing like the MoE's, so that result does
    not transfer. This is the first experiment I would run.

---

# Appendix: the simulator's numbers

*Appended by `floorplan/rank.py` after the ranking above was written. The ranking
agent did not see any of this — it had no access to the simulator, its traces or
these latencies. Two independent judgements of the same schemes.*

## Simulated latencies

All milliseconds, lower is better, across phase x context length x batch size.
**Uncalibrated**: the simulator was built without access to any measured latency for
this model, so these numbers are useful for comparing schemes against each other and
not as predictions of wall-clock time.

| scheme | prefill_128_b1_ms | prefill_128_b4_ms | prefill_128_b8_ms | prefill_128_b32_ms | prefill_8192_b1_ms | prefill_8192_b4_ms | prefill_8192_b8_ms | prefill_8192_b32_ms | decode_128_b1_ms | decode_128_b4_ms | decode_128_b8_ms | decode_128_b32_ms | decode_8192_b1_ms | decode_8192_b4_ms | decode_8192_b8_ms | decode_8192_b32_ms | score | iteration |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `scheme-a` | 27.289 | 81.173 | 157.153 | 613.793 | 617.393 | 1867.114 | 3688.863 | 14619.674 | 11.002 | 11.820 | 14.340 | 29.461 | 13.868 | 22.267 | 26.181 | 49.667 | 1.0606 | -1 |
| `scheme-b` | 24.419 | 69.273 | 133.945 | 523.016 | 627.067 | 1851.414 | 3649.745 | 14440.525 | 10.715 | 11.530 | 14.021 | 28.971 | 13.581 | 21.977 | 25.891 | 49.377 | 1.0108 | 5 |
| `scheme-c` | 24.419 | 69.273 | 133.945 | 523.016 | 627.067 | 1851.414 | 3649.745 | 14440.525 | 10.715 | 11.530 | 14.021 | 28.971 | 13.581 | 21.977 | 25.891 | 49.377 | 1.0108 | -1 |

The score is the mean of every latency normalized against the best value any scheme
achieved for that metric, so it rewards being near the frontier across the whole grid
rather than winning one point.

## Where the two overall rankings agree

| rank | architectural analysis | simulator |
|---|---|---|
| 1 | `scheme-b` | `scheme-c` |
| 2 | `scheme-c` | `scheme-b` |
| 3 | `scheme-a` | `scheme-a` ✓ |

**1 of 3 positions agree.**

## Per-configuration rankings

The agent's call for each point beside the simulator's, best first. `—` means the
report did not rank that configuration; a missing opinion and an opinion that happens
to agree are different things and are not conflated here.

| configuration | architectural analysis | simulator | agree |
|---|---|---|---|
| `prefill_128_b1` | `scheme-b` > `scheme-c` > `scheme-a` | `scheme-c` > `scheme-b` > `scheme-a` | ✗ |
| `prefill_128_b4` | `scheme-b` > `scheme-c` > `scheme-a` | `scheme-b` > `scheme-c` > `scheme-a` | ✓ |
| `prefill_128_b8` | `scheme-b` > `scheme-c` > `scheme-a` | `scheme-c` > `scheme-b` > `scheme-a` | ✗ |
| `prefill_128_b32` | `scheme-b` > `scheme-c` > `scheme-a` | `scheme-c` > `scheme-b` > `scheme-a` | ✗ |
| `prefill_8192_b1` | `scheme-a` > `scheme-b` > `scheme-c` | `scheme-a` > `scheme-c` > `scheme-b` | ✓ |
| `prefill_8192_b4` | `scheme-a` > `scheme-b` > `scheme-c` | `scheme-c` > `scheme-b` > `scheme-a` | ✗ |
| `prefill_8192_b8` | `scheme-a` > `scheme-b` > `scheme-c` | `scheme-b` > `scheme-c` > `scheme-a` | ✗ |
| `prefill_8192_b32` | `scheme-a` > `scheme-b` > `scheme-c` | `scheme-b` > `scheme-c` > `scheme-a` | ✗ |
| `decode_128_b1` | `scheme-b` > `scheme-c` > `scheme-a` | `scheme-c` > `scheme-b` > `scheme-a` | ✗ |
| `decode_128_b4` | `scheme-b` > `scheme-c` > `scheme-a` | `scheme-c` > `scheme-b` > `scheme-a` | ✗ |
| `decode_128_b8` | `scheme-b` > `scheme-c` > `scheme-a` | `scheme-b` > `scheme-c` > `scheme-a` | ✓ |
| `decode_128_b32` | `scheme-b` > `scheme-c` > `scheme-a` | `scheme-c` > `scheme-b` > `scheme-a` | ✗ |
| `decode_8192_b1` | `scheme-b` > `scheme-c` > `scheme-a` | `scheme-b` > `scheme-c` > `scheme-a` | ✓ |
| `decode_8192_b4` | `scheme-b` > `scheme-c` > `scheme-a` | `scheme-b` > `scheme-c` > `scheme-a` | ✓ |
| `decode_8192_b8` | `scheme-b` > `scheme-c` > `scheme-a` | `scheme-b` > `scheme-c` > `scheme-a` | ✓ |
| `decode_8192_b32` | `scheme-b` > `scheme-c` > `scheme-a` | `scheme-b` > `scheme-c` > `scheme-a` | ✓ |

**The agent called 16 of 16 configurations and picked the simulator's winner in 7 of them.**

Where they differ, neither is authoritative. The simulator's constants are uncalibrated, and the architectural reading cannot see queueing or pipeline fill — but a disagreement concentrated in one region of the grid (all the batch-32 points, say, or only long context) is a specific and checkable claim about which effect is being missed.

**The two methods disagree on the winner** — architecture prefers `scheme-b`, the simulator prefers `scheme-c`. This is the most informative outcome available and should not be resolved by preferring one automatically. The simulator is uncalibrated; the architectural analysis cannot see second-order effects like queueing and pipeline fill. Read the report's reasoning against the losing scheme's latencies and decide which argument actually applies.

## Provenance of the hardware numbers

Probed on trn2.3xlarge at 2026-09-25T07:08:04.

Extrapolated rather than measured, because the dev host has one device:

- `links.inter_device.bandwidth_bytes_per_s`
- `links.inter_device.per_hop_latency_us`
- `collectives wider than the 4 logical cores of one device`

**`links.intra_device` is derived, not measured.** device HBM bandwidth (2.9e+12 B/s) x measured dma_large_contiguous (0.2770). NOT MEASURED. NKI 0.6.0 has no collective primitive and the XLA all-reduce fails to compile (NCC_ISMP902), so no intra-device collective can be timed on this host. Bounded by the on-chip path per text constraint 3.

Any conclusion that turns on whether tensor parallelism belongs inside a device or across devices rests on this number, and it is the first thing to measure on real 16-device hardware.
