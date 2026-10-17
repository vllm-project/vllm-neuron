# Ranking three deployments of DeepSeek V4.1-Flash on trn2-16device

## 0. Answer first

```
RANKING: scheme-c > scheme-b > scheme-a
```

`scheme-c` is the best deployment. `scheme-b` is a close second and beats `scheme-c` on the four
long-context prefill points. `scheme-a` is third by a wide and structural margin: it makes the
wrong cut on the MoE, which is 58.4% of the model's parameters, and pays for it at every one of
the sixteen points.

The gap between b and c is a judgement call and I say below exactly which part of c I would
revert. The gap between {b, c} and a is not a judgement call.

## 1. The three schemes are one lineage, not three designs

Before anything else: these are not three independent plans. Diffed against each other they are a
strictly nested chain.

| | difference from its predecessor |
|---|---|
| `scheme-a` | the base (its own notes call it iter-3) |
| `scheme-b` | **a**, with all 43 `layers.*.ffn` changed from `expert x4 * batch x2` to `expert x8`. Nothing else — not one unit, stage, tier or runtime field. |
| `scheme-c` | **b**, plus three things: (1) five attention placements go from `head x4 * batch x2` to `head x8`; (2) both `engram` tables go from `ngram x16` to `ngram x8 * batch x2`; (3) `prefill_chunk_tokens` 96 → 80. |

Everything else is byte-identical across all three: the same 16-stage pipeline on the same 16
devices, the same 8-core tensor-parallel groups, the same `hidden x4` treatment of every
hyper-connection module, the same `vocab` cuts on `embed` and `lm_head`, `decode_micro_batch: 4`,
`pipeline_chunks: true`, and — in all 271 placements of all three files — `tier: hbm_bank` with no
`resident_fraction`, no `cache_tier` and no `hit_rate` anywhere.

So the ranking is not "which floorplan", it is "which of four specific partition decisions are
right". That is a better question and it is answerable from the hardware.

Two notes on method. First, the scheme files carry the search's own annotations, including
relative metric deltas from the uncalibrated simulator. I read them for design intent — they say
what was tried and rejected, which is real information — but I have not ranked by them, and in two
places below my reading of the hardware disagrees with them. I say so where it happens and flag it
in §6. Second, all the arithmetic below is roofline arithmetic from `PLATFORM.md`,
`systems/trn2-16device.yaml` and the model's own `config.json` and `vendor/model.py`; it is there
to establish *which term dominates*, not to predict milliseconds.

## 2. What the model actually is

The partition decisions only make sense against five facts from `modules/*/config.json` and
`vendor/model.py`, so they go first.

**The MoE is the model.** `n_routed_experts: 384`, `n_activated_experts: 6`, `n_shared_experts: 1`,
`moe_inter_dim: 2304`, `expert_dtype: fp4`. One expert is 3 × 5120 × 2304 = 35.4 M params at half
a byte = 17.7 MB; 384 of them is 6.45 GiB per layer, 277 GiB over 43 layers — 58.4% of the model.
A token activates 6 of 384, so 1.6% of it.

**Attention has one shared KV latent, not one per head.** `Attention.wkv` is
`Linear(dim=5120 → head_dim=512)`: a *single* 512-wide K/V vector per position that all
`n_heads: 64` heads read (`vendor/model.py:643`, `:774-780`). Consequence, and it drives two of the
four decisions: **a `head` split divides the weight stream and replicates the KV sweep.** Every
head shard holds its own slice of `wq_b`, `wo_a`, `wo_b` but must sweep the *entire* window and
compressed KV cache.

**How far back a layer looks is per-layer and fixed.** `window_size: 128`, and
`compress_ratios` is `0,0` for layers 0-1, `2` for layers 2-19, `1` for layers 20-39, `0,0,0` for
the three MTP layers. `DSparkAttention.forward` opens with `assert self.compress_ratio == 0`
(`vendor/model.py:1034`). So `kv_rows = 128 + ctx/ratio` for layers 2-39 and **exactly 128, at any
context, for layers 0, 1, mtp.0, mtp.1 and mtp.2** — five layers whose KV sweep does not grow.
That is five of 43, and it is the whole basis of c's change (1).

**The indexer is where the long-context collective traffic is.** `index_source_layers:
[2,8,14,20,24,28,32,36]`, `index_n_heads: 32`, `index_head_dim: 128`, `index_topk: 512`. Each of
those eight layers scores every reachable compressed position against every query
(`index_score = einsum("bshd,btd->bsht", q, index_k)`, then `dist.all_reduce(index_score)`,
`vendor/model.py:556-559`). The all-reduced payload is `queries x compress_rows x fp32` — it grows
with context *and* batch, and it rides on whatever collective the `head` split declares. This is
the hard ceiling on how wide the head split can go on those layers.

**Engram is a 94.56 GiB table that a token touches 6.3 kB of.** Two tables,
`engram_num_embeddings: [384006168, 384016682]`, `engram_head_dim: 256`, fp8 rows plus one fp8
scale per 32 lanes = 264 B/row → 94.56 GiB each. `engram_max_ngram_size: 4` and
`engram_n_heads: 8` give (4−1) × 8 = **24 hash columns per token**, so a token reads 24 rows =
6.3 kB per engram layer. The row axis is not homogeneous: it is 24 disjoint prime-sized buckets of
≈16,000,257 rows (`engram.py:108-118`, `engram_vocab_size: 16000000`), one per
(n-gram size, head) pair, and 24 × 16,000,257 = 384,006,168 exactly.

Two derived hardware ratios I use throughout:

- A logical NeuronCore gets **725 GB/s** to its own 24 GiB bank and **58.7 TFLOPS** of usable fp8
  (1299 TFLOPS/device × 0.181 achieved ÷ 4). Its **break-even arithmetic intensity is 81
  FLOP/byte.**
- A device's four banks at 725 GB/s sum to 2.9 TB/s, which is exactly the `device_hbm` figure. The
  datasheet is self-consistent: four cores streaming at once saturate the device.

## 3. The deployment, walked through (scheme-c)

### Geometry

16 stages on 16 devices, `stage: 0 … 15`. Stage *s* owns device *s*; its tensor-parallel partner is
device *s* XOR 1. On the 4×4 torus (`index = row*4 + col`) XOR-1 always flips the low bit of the
column, so **every tensor-parallel group is two column-adjacent devices, one hop.** Never two, never
the 4-hop diameter.

Stage *s* holds 2 or 3 consecutive transformer blocks:

| stage | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary device | d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | d11 | d12 | d13 | d14 | d15 |
| blocks | 0-2 | 3-5 | 6-8 | 9-10 | 11-13 | 14-16 | 17-18 | 19-21 | 22-24 | 25-26 | 27-29 | 30-32 | 33-34 | 35-37 | 38-39, mtp.0 | mtp.1, mtp.2 |

Because stage *s* and stage *s*XOR1 share the same two devices, the machine is really **eight
device-pairs each running two consecutive pipeline stages**: {d0,d1} runs blocks 0-5, {d2,d3}
blocks 6-10, {d4,d5} blocks 11-16, {d6,d7} 17-21, {d8,d9} 22-26, {d10,d11} 27-32, {d12,d13} 33-37,
{d14,d15} 38-39 plus all three MTP blocks plus `final_norm`, `hc_collapse` and `lm_head`. Five or
six blocks per pair — balanced to about ±10%.

That is a deliberate trade and it is the right one: 16 stages on 16 devices with 4-core
tensor-parallel groups would have halved the TP width and doubled per-bank weight residency, and
decode is bank-bandwidth-bound. Pairing stages buys 8-wide TP at the cost of serializing two
stages per pair.

### Which modules go where

- **`embed`** — `vocab x4` across the four logical cores of d0, stage 0. Each shard holds 32,320 of
  129,280 rows; a token outside a shard's range yields zeros and the rejoin is an `allreduce` of
  `dim x bf16` = 10 kB/token over four cores of one device: intra-device, zero hops.
- **`hc_expand`, `other_globals`** — `hidden x4` on d0, stage 0.
- **Each block's `attention`** — 8 logical cores: all four of the primary device and all four of
  the partner. `head x4 * batch x2` on 38 of 43; `head x8` on layers 0, 1, mtp.0, mtp.1, mtp.2.
- **Each block's `ffn`** — the same 8 logical cores, `expert x8`, `all_to_all`.
- **Each block's four hyper-connection modules** (`hc_attn_in`, `hc_attn_out`, `hc_ffn_in`,
  `hc_ffn_out`) — `hidden x4` on the four cores of the **primary device only**. So each sublayer
  boundary reshards between a 4-core intra-device group and an 8-core two-device group.
- **`layers.1.engram`** — `ngram x8 * batch x2` on all 16 logical cores of d0-d3, stage 0.
  **`layers.14.engram`** — the same shape on all 16 cores of d4-d7, stage 5.
- **`mtp.0.main_norm`, `mtp.0.main_proj`** — `hidden x4` on d14. **`mtp.2.norm`,
  `mtp.2.markov_head`, `mtp.2.confidence_head`, `final_norm`, `hc_collapse`** — `hidden x4` on d15.
- **`lm_head`** — `vocab x8` across d14 and d15 (four cores each), `allgather`, stage 15.

No placement uses `.p0`/`.p1`. Every shard in all three schemes is a whole logical NeuronCore, so
the two physical NeuronCore-v3 inside one are never separately addressed. That is correct here:
constraint 2 says the pair shares one 56 MiB SBUF and one address space, so splitting inside a
logical core adds neither SBUF capacity nor bank bandwidth, and the binding resource is the bank
port.

### Where the weights live

All 271 placements are `tier: hbm_bank`. Nothing is tiered, nothing sits in `device_hbm`,
`peer_hbm`, `host_dram` or `nvme`. The model is 474.5 GiB against 1536 GiB of HBM, so full
residency is affordable and every slower tier is avoided outright — including, per constraint 8,
both Engram tables, whose lookups therefore never pay a tier crossing on the critical path of
layers 1 and 14.

Weight residency per bank, my arithmetic, weights only (attention adds ~0.2 GiB, hyper-connections
~0.02 GiB):

| banks | blocks | engram share | scheme-a | scheme-b | scheme-c |
|---|---|---|---|---|---|
| d0, d1 | 6 | engram.1 | 15.6 | **10.7** | 16.6 |
| d2, d3 | 5 | engram.1 | 14.0 | **9.9** | 15.8 |
| d4, d5 | 6 | engram.14 | 15.6 | **10.7** | 16.6 |
| d6, d7 | 5 | engram.14 | 14.0 | **9.9** | 15.8 |
| d8 … d15 | 5-6 | — | 8.1-9.7 | **4.0-4.8** | 4.0-4.8 |
| **instance total (GiB)** | | | 757 | **479** | 668 |

All three fit the 24 GiB bank. Scheme-c's tightest banks are 70% full before any KV cache or
activation; scheme-b's are 45%.

### The schedule

`prefill_chunk_tokens: 80`, `decode_micro_batch: 4`, `pipeline_chunks: true`. Prefill walks 80-token
chunks (per sample) down the 16 stages with chunk *N*+1 allowed to enter a stage before *N* has
left the next — 2 chunks at 128 tokens, 103 at 8192. Decode splits the batch into micro-batches of
4 and pipelines those; at batch 32 that is 8 micro-batches against 16 stages, at batch 4 one, at
batch 1 nothing to pipeline at all. Stage-to-stage handoff is the rank-4 hyper-connection stream,
`hc_mult(4) x dim(5120)` = 41 kB/token in bf16, plus `pre_mix`; six of the fifteen handoffs are one
hop and the pair-to-pair ones at d3→d4, d7→d8, d11→d12 are two.

## 4. Why this is the best, and what each scheme hits first

### Decode is a weight-streaming problem, and the MoE is the weight

At decode there is one token per sample per step. Per step the machine must read every attention
weight (5.2 GiB), every hyper-connection coefficient, both Engram `wkv` matrices, `embed` and
`lm_head` (2.4 GiB), and — per layer — the gate (3.9 MB), the shared expert, and the 6 of 384
routed experts the token chose (6 × 17.7 MB = 106 MB). Roughly 20 GB of bytes against a few GFLOP
of work: arithmetic intensity of order 1 against a break-even of 81. Decode is nowhere near
compute; it is the 725 GB/s bank port, at every batch size, at both contexts. That is the frame in
which the four decisions have to be judged.

Two things set how fast those bytes arrive: **how many banks are live**, and **how many bytes each
live bank holds**.

How many banks are live is set by the pipeline, and all three schemes are identical here. One stage
is live at a time, and a stage is 8 of 64 banks — **5.8 TB/s of the instance's 46 TB/s.** At batch
32 the 8 micro-batches fill half the 16 stages and this partly recovers; at batch 1 it does not
recover at all. This is the shared ceiling, and it is the *first* limiting factor of every scheme
at `decode_*_b1`. None of the three addresses it, so it cannot separate them — but it is worth
saying that the README's warning about deep pipelines being pure cost at batch 1 is right for a
reason that is not stage-boundary latency: the handoff is 41 kB over a 1024 GB/s link, a couple of
microseconds. The cost is 56 idle bank ports.

How many bytes each live bank holds is exactly what a→b→c moves, and it is where the ranking lives.

### Scheme-a's first limiting factor: it puts the MoE in HBM twice and then idles half of it

Scheme-a splits every `ffn` `expert x4 * batch x2` over 8 cores. `batch` is not a weight-partition
dimension, so the weight divisor is 4, not 8: each shard holds 96 experts = **1.61 GiB per layer
per bank**, and the instance holds 555 GiB of expert weights for a 277 GiB MoE. Scheme-b's
`expert x8` gives each shard 48 experts = **0.806 GiB**, and 277 GiB total. On the term that is
58.4% of the model, streamed through a 725 GB/s port, that is a factor of two.

It is worse than a factor of two at batch 1, and this is the part that decides third place. In
`splits`, shard *i* lands on `units[i]` in row-major order, so with `[expert x4, batch x2]` the
*batch* index varies fastest. Scheme-a's unit lists alternate devices — `d0.l0, d1.l0, d0.l1,
d1.l1, …` — so batch-bucket 0 lands on the four cores of the primary device and **batch-bucket 1
lands entirely on the partner device.** At batch 1 there is nothing for bucket 1: in scheme-a, at
every `decode_*_b1` and `prefill_*_b1` point, the partner device of every stage does no MoE work
at all. The MoE streams 1.61 GiB per layer from **4** banks where scheme-b streams 0.806 GiB from
**8**. Four times the time on 58% of the model, and 2.9 TB/s of the instance's 46 TB/s in use.

The same accounting shows why `expert x8` and not more. A token activates 6 experts. Spread over 8
shards, the expected number of shards with work is 8(1 − (7/8)⁶) ≈ 4.5 of 8; over 16 shards it is
≈4.8 of 16, i.e. 70% idle. **`n_activated_experts: 6` is why 8 is the widest useful expert split**,
and it explains from the hardware what the search found by measurement when `expert x16` on 16
cores lost badly. It also explains why widening the group to four devices is the wrong fix: the
`all_to_all` would pay the 2-hop diameter of a four-device group on every step (constraint 5) to
buy shards that have no work.

What does scheme-a's structure buy? One real thing. Under `expert x4 * batch x2` a shard sees half
the tokens, so the per-token work that expert parallelism *replicates* — the gate over 384 scores,
`ffn_norm`, `act_quant`, the fp32 accumulator over `dim` — is halved. This is why the search
measured scheme-a ahead on the six prefill points at batch ≥ 4, and it is the one place I disagree
with it. Two hardware reasons:

1. **Arithmetic intensity says prefill is weight-bound below about batch 14.** Per chunk per layer
   the expert weights are 6.93 GB; at batch 32 with an 80-token chunk the expert FLOPs are
   ≈1.3 × 10¹². That is 188 FLOP/byte against a break-even of 81 — compute-bound. At batch 8 it is
   47, at batch 4 it is 23, at batch 1 it is 6. So at `prefill_*_b1`, `_b4` and `_b8`, prefill is
   bound by exactly the same thing decode is, and halving the resident expert bytes wins.
2. **At batch 32, where prefill does cross into compute, `expert x8` has the better GEMM shape.**
   Assignments per expert per chunk: scheme-a gives 1280 tokens over 96 experts ≈ 96 rows;
   scheme-b gives 2560 tokens over 48 experts ≈ 384 rows. The achieved fp8 fraction is only 0.181,
   which is what a small-M grouped GEMM looks like; 384 rows fills three 128-row tiles where 96
   underfills one. Scheme-a's genuine edge here is the 2.0 MiB PSUM: a `M x 2304` fp32 accumulator
   holds M ≤ 227 rows, so scheme-b's 384-row GEMM needs two PSUM passes and scheme-a's 96-row one
   needs one. That is a tiling cost, not a bandwidth cost, and it is second-order to halving the
   weight stream.

So my reading is that **b beats a at all sixteen points**, with `prefill_8192_b32` the closest.
This is the one place my judgement diverges from the search's own measurements, and §6 says what I
would check to settle it.

Scheme-a's first limiting factor, then: the 725 GB/s bank port of the *primary* device of each
stage, carrying 2× the expert bytes it needs to, with the partner device's four ports unused at
batch 1.

### Scheme-c's change (1): right, and provable from the source

Five attention placements — layers 0, 1 and the three MTP blocks — go from `head x4 * batch x2` to
`head x8`. These are exactly the five layers with `compress_ratio == 0`, whose `kv_rows` is the
128-slot sliding window at 128 tokens and at 8192 tokens alike.

A `head` split halves the weight stream (`wq_b`, `wo_a`, `wo_b`) and replicates the KV sweep,
because `wkv` produces one shared 512-wide latent. On these five layers the replicated sweep is
128 rows × 512 B, bounded and tiny; the halved weight stream is real. And it removes the
`batch x2` leg, which at batch 1 idled four of the eight cores for the same reason the MoE's did.

The hop accounting is the pretty part, and it also shows why this change must *not* be extended.
With `[head x4, batch x2]` and the alternating unit list, shard (h, b) sits at index 2h+b, so
head-shard *h* with bucket *b* is on `d(primary if b=0 else partner).l h`. **The 4-way head
allreduce group is therefore the four logical cores of one device: intra-device, zero torus hops,
803 GB/s** (constraint 3), and the free `batch` leg is the one that crosses the device boundary.
That is an elegant assignment and all three schemes get it.

Change (1) replaces that with a single `head x8` whose group spans both devices — an 8-way
allreduce, 1024 GB/s, one hop, +1 µs. On these five layers the allreduce carries only `wo_b`'s
partial, `queries x 5120 x bf16` = 10 kB/token, because `compress_ratio == 0` means
`_compress_kv` never runs and there is no indexer and no `index_score`. A microsecond of extra
latency against halving the per-bank attention weight stream on five layers: a clear win, biggest
at batch 1 where it also un-idles four cores.

Extending it to the other 38 layers would be a disaster, and the source says why: on the eight
`index_source_layers` the same allreduce carries `index_score`, `queries x compress_rows x fp32`.
At 8192 context and `compress_ratio: 1` that is 8192 columns of fp32 per query. Doubling the group
width and pushing it across a device boundary on that payload is not a trade anyone should make —
and separately, `head x8` would double the replicated KV sweep on every layer whose `kv_rows`
grows with context. Scheme-c leaves all 38 at `head x4` on an intra-device group. Correct.

### Scheme-c's change (2): the one I would revert

Both Engram tables go from `ngram x16` to `ngram x8 * batch x2` on the same 16 cores. `batch` is
not a weight-partition dimension, so the per-shard charge doubles: **5.91 GiB → 11.82 GiB per
bank, and 94.56 GiB → 189 GiB per table.** Two tables: **378 GiB, 24.6% of the instance's HBM**, on
32 of the 64 banks, taking each of the d0-d7 banks from 9.9-10.7 GiB to 15.8-16.6 GiB — 70% of a
24 GiB bank committed before a single KV row.

What it buys is genuine but small. Under an `ngram` split the row lookup and `wkv`'s contraction
divide, but Engram's gate — a per-(token, hc-copy) normalized dot product over `dim` in fp32,
`vendor/model.py:266-269` — is replicated on all 16 shards, and a `batch` split halves the tokens
each shard sees. At `prefill_8192_b32` that is 16 shards × 262,144 tokens × (4 × 5120 × 4 B) ≈
344 GB of fp32 traffic per table, halved: worth some tens of milliseconds across 16 banks. Real.
The second merit is cleaner and nobody mentions it: **`ngram x8` aligns with the table's 24 prime
buckets at exactly 3 per shard, where `ngram x16` puts 1.5 buckets on a shard** and splits a
(n-gram size, head) bucket across two. Under x8 each of a token's 24 hash columns resolves to
exactly one shard.

Against that: at batch 1 the change is a straight regression, by the same mechanism that made a→b
right. Here the unit list is device-major (`d0.l0, d0.l1, d0.l2, d0.l3, d1.l0, …`) and `batch` is
still the fast axis, so buckets alternate between adjacent logical cores *within* each device. At
batch 1, `l1` and `l3` of d0-d3 (and d4-d7) sit idle holding 11.82 GiB each, while the eight active
shards each stream double the `wkv` bytes they did under x16. So the change costs at
`prefill_128_b1`, `prefill_8192_b1` and `decode_*_b1`, buys something only at batch ≥ 4, buys
almost nothing at decode where a step is 1-4 tokens, and costs 189 GiB everywhere.

Nothing in the measured grid fails because of it — the 24 GiB bank still holds, and I checked the
worst case: `{d4,d5}` at 8192 × batch 32 carries 16.6 GiB of weights, plus layer 14's
`compress_kv_cache` (16 samples × 4096 rows × 512 B), its indexer `k_cache`, six layers of
128-slot window caches and the live activations — on the order of 3 GiB, inside the ~7 GiB left.
But the headroom it consumes is the only thing that would pay for the alternatives this deployment
never tries: replicating hot experts, or `max_seq_len: 16384` which the config supports and at
which the ratio-1 compressed caches double. A plan that doubles its single largest memory
commitment to shave a replicated scalar pass on four of sixteen points is worse engineering, and
`ngram x8 * seq x2` — which the notes record as rejected on a sub-1% basis — divides `work_tokens`
at batch 1 too and costs no replica.

### Scheme-c's change (3): a near-wash that helps the short context

`prefill_chunk_tokens` 96 → 80. Two mechanisms, pulling opposite ways.

At **128 tokens** the 16-stage pipeline has only 2 chunks to fill it either way (96+32, or 80+48),
so almost the whole latency is 16 unoverlapped traversals of the *first* chunk. Modelling stage
time as proportional to tokens: 16·t(96) + t(32) ≈ 12.3·t(128) against 16·t(80) + t(48) ≈
10.4·t(128). Chunk 80 is ~15% better because the chunk that pays the fill is smaller. Real, and it
applies to all four `prefill_128` points.

At **8192 tokens** the chunk count goes 86 → 103, and each extra chunk re-pays the per-chunk
weight read. The re-read is sublinear in chunk size, which softens it: 96 tokens generate 576
assignments over 384 experts and touch ≈78% of them, 80 tokens generate 480 and touch ≈71%. Netting
per-chunk fixed cost against per-token work gives roughly +5% at batch 1 and +3% at batch 32. A
loss, concentrated where the absolute milliseconds are.

Summed in absolute time this is a small net loss; judged per-metric it is a small net gain. Both
chunk settings also underfill a 128-wide query tile (75% and 62.5%), so neither is the real answer
to the `prefill_128` problem — 16 stages is.

### Where that leaves the ranking

- **scheme-a — first limiting factor:** the primary device's bank ports, carrying 2× the expert
  bytes, with the partner device idle in the MoE at batch 1. This is the worst of the three
  because it is on the term that is 58.4% of the parameters, and it hurts at every point.
- **scheme-b — first limiting factor:** pipeline occupancy at decode (8 of 64 banks live), and then
  the `batch x2` leg still on all 43 attention placements, which idles the partner device's
  attention at batch 1. At prefill 8192 it is the per-chunk re-read of ~78% of each layer's 384
  experts, 86 times.
- **scheme-c — first limiting factor:** the same pipeline occupancy at decode; at prefill 8192 the
  same per-chunk re-read, 103 times instead of 86; and structurally, bank headroom — 70% of 32
  banks committed to weights, two-thirds of it to duplicate copies of a table a token reads 6.3 kB
  of.

Scheme-c's is the least bad because its decode limit is strictly better than scheme-b's (the five
ratio-0 layers) and its prefill-8192 limit is worse than scheme-b's by a few percent on four points.
Twelve of sixteen against four. Scheme-a's is worse than both on all sixteen.

## 5. How every split in the winning scheme is actually performed

Shapes: `dim = 5120`, `n_heads = 64`, `head_dim = 512`, `rope_head_dim = 64`, `q_lora_rank = 1280`,
`o_lora_rank = 1024`, `o_groups = 8`, `moe_inter_dim = 2304`, `vocab_size = 129280`, `hc_mult = 4`.
Every shard below is a **logical NeuronCore**; no split uses the two physical NeuronCore-v3 inside
one.

### 5.1 `layers.{2..39 minus 0,1}.attention` — `head x4` × `batch x2`, 8 logical cores

*Dimension.* `head` cuts the 64 query heads 4 ways: 16 heads per shard. `batch` cuts samples 2
ways, no collective.

*Tensors.*
- `wq_a` [1280, 5120] fp8 and `q_norm` — **replicated**: they produce the shared low-rank `qr`
  every head shard needs. 6.5 MB.
- `wq_b` [64·512 = 32768, 1280] fp8 — cut on the **output rows**: [8192, 1280] per shard, 10.5 MB.
- `wkv` [512, 5120] fp8 and `kv_norm` — **replicated**, 2.6 MB. This is the shared KV latent, and
  the reason the head split replicates the sweep: every shard holds the whole 128-slot
  `window_kv_cache` and reads the whole `compress_kv_cache`.
- `attn_sink` [64] fp32 — 16 per shard.
- `wo_a` — block-diagonal over the 8 `o_groups`, viewed as [groups, 1024, 4096]. 64 heads / 8
  groups = 8 heads per group, so 16 heads is **exactly 2 groups**: [2, 1024, 4096] bf16 = 16.8 MB
  per shard. The `head` cut and the group structure line up with no remainder.
- `wo_b` [5120, 8192] fp8 — `RowParallelLinear`, cut on the **contraction**: [5120, 2048] per
  shard, 10.5 MB, producing a partial sum over `dim`.
- On `kv_source_layers` {2,8,14,20}: `Compressor.wkv` and `.wgate` [512, 5120] (fp32 at ratio 2,
  bf16 at ratio 1) — **replicated**, since the latent is shared. On
  `index_source_layers` {2,8,14,20,24,28,32,36}: `Indexer.wq_b` [4096, 1280] cut to
  [1024, 1280] (8 of 32 index heads), `weights_proj` [32, 5120] bf16 cut to 8 rows, `wk`/`k_norm`
  replicated.

*Communication, and when.* Once per sublayer, after `wo_b`: an `allreduce` of the partial
[batch, tokens, 5120] — 10 kB/token in bf16. On the eight index-source layers, additionally an
`allreduce` of `index_score` [batch, queries, compress_rows] fp32 inside the indexer, before the
top-k — this is the payload that grows with context × batch and caps the head factor at 4.

*Hierarchy.* 8 logical cores: `l0..l3` of the primary device and `l0..l3` of the partner, one torus
hop apart. Because `batch` is the fast-varying index, **the head allreduce group is the four
logical cores of a single device** — intra-device, 0 hops, 803 GB/s, contending for that device's
2.9 TB/s per constraint 3. The batch leg is what crosses the hop, and it needs no collective.

### 5.2 `layers.{0,1}.attention` and `mtp.{0,1,2}.attention` — `head x8`, 8 logical cores

*Dimension.* `head` cuts 64 heads 8 ways: 8 heads = exactly 1 `o_group` per shard. No batch leg.

*Tensors.* As above with the factors doubled: `wq_b` → [4096, 1280]; `wo_a` → [1, 1024, 4096]
bf16 = 8.4 MB; `wo_b` → [5120, 1024]; `wq_a`, `wkv`, the norms replicated. No `Compressor`, no
`Indexer` — `compress_ratio == 0` on all five.

*Communication.* One `allreduce` of `wo_b`'s [batch, tokens, 5120] partial per sublayer, 10 kB/token.
Nothing else: no `index_score`, because these layers never enter `_compress_kv`.

*Hierarchy.* The 8-way group now spans both devices of the pair — one torus hop, 1024 GB/s,
+1 µs of latency per allreduce against halving the per-bank attention weight stream. The KV sweep
each shard replicates is the 128-slot window, at any context.

### 5.3 `layers.*.ffn` (all 43) — `expert x8`, 8 logical cores

*Dimension.* `expert` cuts the 384 routed experts 8 ways: **48 experts per shard**. It also cuts
the shared expert's intermediate dimension, which is the standard tensor-parallel cut of a SwiGLU
FFN.

*Tensors.*
- Routed experts: shard *i* owns experts [48*i, 48*i+48). Each is `w1` [2304, 5120],
  `w3` [2304, 5120], `w2` [5120, 2304] in fp4 = 17.7 MB → **0.85 GB per shard per layer**. A shard
  holding no expert a token routed to simply runs nothing for it (`MoE.forward` skips
  `counts[i] == 0`).
- `Gate.weight` [384, 5120] bf16 + `bias` [384] fp32 — **replicated**, 3.9 MB. Every shard computes
  the full routing decision so it knows which of its own 48 to run.
- Shared expert: `moe_inter_dim` 2304 → **288 per shard**. `w1`, `w3` → [288, 5120]
  (column-parallel); `w2` → [5120, 288] (row-parallel). 35.4 MB → 4.4 MB per shard. Its output
  needs a sum across the 8 shards.

*Communication, and when.* After the gate and before the expert GEMMs, an `all_to_all` dispatch
sending each token to the shard owning its chosen expert; after the GEMMs, the symmetric combine.
Payload `tokens x 6 activated x 5120 x bf16 x 2` = 123 kB/token. Plus the shared expert's
reduction over the 8 shards, same rejoin.

*Hierarchy.* 8 logical cores, four per device, two column-adjacent devices. Half the `all_to_all`
traffic stays on the 803 GB/s intra-device path and half crosses the single 1024 GB/s hop. Widening
to 16 cores would pull in a second device pair and pay a 2-hop group diameter (`d0→d2 = 2`) on
every step, for shards that have no work: with 6 activated experts, 8 shards already leaves ~3.5
of 8 idle in expectation, and 16 would leave ~11 of 16.

### 5.4 `layers.{1,14}.engram` — `ngram x8` × `batch x2`, 16 logical cores

*Dimension.* `ngram` cuts the hash table's **row** axis 8 ways. The axis is 24 disjoint prime
buckets of ≈16,000,257 rows, one per (n-gram size, head) pair, so 8 ways is **3 whole buckets per
shard** and every one of a token's 24 hash columns resolves to exactly one shard. `batch` cuts
samples 2 ways, no collective.

*Tensors.*
- `ParallelEngramEmbedding.weight` [48,000,771, 256] fp8 + `.scale` [48,000,771, 8] fp8 =
  **11.82 GiB per shard**. An index outside a shard's `[vocab_start_idx, vocab_end_idx)` is masked
  to zero (`vendor/model.py:312-326`), so a shard contributes only for the columns it owns.
- `wkv` `Linear(24·256 = 6144 → 5120·5 = 25600)` fp8, 157 MB — cut on the **contraction**:
  [25600, 768] per shard, 19.7 MB. Each shard produces a partial of the full 25600-wide output.
- `q_weight`, `k_weight` [4, 5120] fp32 — replicated, negligible.
- The gate (`rstd`, `dot`, `sigmoid` over `dim` in fp32) is **replicated on all 8 ngram shards**
  and is what the `batch` leg halves.

*Communication, and when.* Once, after `wkv`, before the gate: combine the 8 ngram shards' partial
[tokens, 25600] fp32 results — 102 kB/token — and add into the residual stream at the *start* of
blocks 1 and 14 (`vendor/model.py:1262-1263`). The schemes declare `allgather`; the model does
`dist.all_reduce` (`vendor/model.py:325`), i.e. a sum over row ranges, not a concatenation. See §6.

*Hierarchy.* 16 logical cores = all four of d0-d3 (block 1) or d4-d7 (block 14). With `batch` as
the fast axis the ngram group for a given bucket is `l0, l2` of each of four devices, so the
collective's worst hop distance is 2 (`d0→d2`, `d4→d6`) — unavoidable for any 16-unit group on a
4×4 torus, and it sits on the critical path of blocks 1 and 14. The consumer is on the primary
device only (`d0` for block 1, `d5` for block 14), so shards on `d3`/`d7` ship their partials two
hops.

### 5.5 The hyper-connection modules — `hidden x4`, 4 logical cores of one device

172 placements (43 blocks × 4) plus `hc_expand` and `hc_collapse`.

*Dimension.* `hidden` cuts the flattened `hc_mult · dim = 20480` residual stream 4 ways.

*Tensors.* `HyperConnectSublayerIn.fn` [24, 20480] fp32 — cut on the **contraction**: [24, 5120]
per shard, 492 kB; `base` [24] and `scale` [3] replicated. Each shard produces a partial of
`mixes` [tokens, 24] fp32 and a partial of the sum-of-squares behind `rsqrt`.
`HyperConnectSublayerOut` has **no parameters at all**: `hidden` just gives each shard 1280 of the
5120 output columns, and the 4×4 `comb` contraction is over the hc axis, not the split axis.

*Communication.* For `…In`: an `allreduce` of 24 floats plus one float per token — ~100 B/token,
essentially free — then the collapsed `y` over `dim`. For `…Out`: nothing is actually required; the
declared `allreduce` is conservative.

*Hierarchy.* Four logical cores of the **primary device only**, intra-device, zero hops. Note the
consequence: each sublayer boundary reshards between this 4-core single-device group and the
8-core two-device attention/ffn group — an intra-device gather plus a one-hop transfer of
`dim x bf16` = 10 kB/token.

### 5.6 `embed` — `vocab x4`, and `lm_head` — `vocab x8`

`embed`: [129280, 5120] cut on **rows**, 32,320 per shard on `d0.l0..l3`. Out-of-range ids yield
zeros; rejoin is an `allreduce` of `dim x bf16` = 10 kB/token, intra-device, 0 hops.

`lm_head`: [129280, 5120] cut on **rows** 8 ways, 16,160 per shard across `d14.l0..l3` and
`d15.l0..l3`. Each shard produces its own logit slice; rejoin is an `allgather` of 129,280 fp32 =
517 kB per token, over a one-hop two-device group. At batch 32 that is 16.5 MB per step.

### 5.7 The remaining `hidden x4` singletons

- `mtp.0.main_proj` `Linear(5120·3 = 15360 → 5120)` fp8, 78.6 MB — contraction cut to 3840 per
  shard, 19.7 MB; `allreduce` of the [tokens, 5120] partial. On `d14.l0..l3`.
- `mtp.2.markov_head` = `ParallelEmbedding(129280, 256)` + `ParallelHead(129280, 256)`, 132 MB —
  `hidden` cuts the rank 256 to **64 per shard**, so each shard holds [129280, 64] twice, 33 MB.
  The embedding lookup returns 64 of 256 dims and the head contracts over those 64, so the rejoin
  is an `allreduce` of the full 129,280-wide logit bias in fp32, 517 kB per drafted token, over
  four cores of `d15` at 803 GB/s — sub-microsecond, but worth knowing it is 517 kB to save 99 MB
  of residency.
- `mtp.2.confidence_head` `Linear(5376 → 1)` fp32 — contraction cut 4 ways, `allreduce` of one
  scalar per token.
- `final_norm`, `mtp.0.main_norm`, `mtp.2.norm` — `RMSNorm(5120)`, weight cut to 1280 per shard;
  the rejoin is the partial sum of squares for the RMS statistic, one fp32 per token.
- `hc_expand`, `hc_collapse` — no parameters; `hidden x4` just assigns each core a quarter of the
  stream's `dim`.

## 6. What I would check first

**No scheme tiers anything, so there is no `hit_rate` to justify.** All 271 placements in all three
files are `tier: hbm_bank`. The Engram question constraint 8 poses is answered by brute residency,
and the first thing I would check is the *untested opposite*: a tiered Engram with a small
`resident_fraction` cached in `hbm_bank` and backed by **`peer_hbm`** — 1024 GB/s, 1.0 µs — rather
than `host_dram` at 13 GB/s or `nvme` at 9.1 GB/s reached through host DRAM (constraint 6). A token
reads 24 rows of 264 B = 6.3 kB per engram layer, twice per token in the whole forward pass. That
is a tiny, random, cold access on the critical path of blocks 1 and 14. If the crossing costs a few
microseconds per token it buys back 189 GiB in scheme-b and 378 GiB in scheme-c. Nobody measured
it, and it is the single largest unexplored decision in the plan. The `hit_rate` a tiered Engram
would have to claim is genuinely hard — 384 million rows, 24 hashed lookups per token, hashed
specifically so that n-grams spread — so I would expect a *low* resident fraction with no hit-rate
pretence at all, priced at full `peer_hbm` latency.

**Every inter-device number I used is an extrapolation.** `PLATFORM.md`'s own provenance section
says `links.inter_device.bandwidth_bytes_per_s`, `per_hop_latency_us` and *every collective wider
than the four logical cores of one device* are extrapolated, not measured, because this host has
one Trainium2 device. That covers: the one-hop `all_to_all` of every `ffn`, the one-hop `head x8`
allreduce of scheme-c's five layers, the two-hop Engram reduction, the `lm_head` allgather, and
every stage-to-stage handoff. If real inter-device bandwidth is materially below 1024 GB/s, the
8-core two-device tensor-parallel group becomes the binding constraint rather than the bank port,
and scheme-a — whose per-shard `all_to_all` payload is half of b's and c's — closes the gap.

**`intra_device` at 803 GB/s is derived, not measured, and it carries the biggest collective in the
file.** It is `device_hbm` bandwidth × a measured `dma_large_contiguous` efficiency of 0.277, and
NKI 0.6.0 cannot time an intra-device collective at all (`NCC_ISMP902`). The `index_score`
allreduce on the eight index-source layers — `queries x compress_rows x fp32`, growing with context
and batch — runs on exactly that path. Most of my prefill-8192 reasoning, and the whole argument
for why `head x4` must stay `head x4`, rests on this one derived number.

**Whether the achieved fp8 fraction is M-dependent.** I argued `expert x8` wins at
`prefill_8192_b32` partly because 384 rows per expert GEMM fills three 128-row tiles where
scheme-a's 96 rows underfills one, and partly because 0.181 of peak is what a small-M grouped GEMM
looks like. If the cost model applies a flat 0.181 regardless of M, half that argument disappears
and only the weight-byte halving remains — still decisive, but the four `prefill_*_b32` calls get
closer, and this is where my ranking most visibly disagrees with the search's own measurements.

**The routing entropy behind the chunk-size knee.** My +3…5% estimate for chunk 80 at 8192 context
hinges on how many distinct experts a chunk actually touches — I assumed near-uniform routing, so
78% of 384 at 96 tokens and 71% at 80. The gate has a learned correction bias and real prompts are
not uniform, so the true distinct-expert count per chunk is likely lower, the per-chunk fixed cost
smaller, and chunk 80 better than I have it. Measure distinct experts per chunk at 80 and 96 tokens
on a real prompt before believing either setting. Relatedly: both settings underfill a 128-wide
query tile (62.5% and 75%), so I would check whether the attention kernel pads M or handles it
ragged — if it pads, the right move at 128 context is a shallower pipeline, not a smaller chunk.

**Two collectives are declared wrong, identically in all three schemes.** The Engram `ngram` leg
declares `allgather`, but `ParallelEngramEmbedding.forward` ends in `dist.all_reduce` — the rejoin
is a *sum* over disjoint row ranges. And `HyperConnectSublayerOut` needs no collective at all
(`hidden` splits its output `dim`; the only contraction is over the hc axis) yet declares
`allreduce` on 172 placements. Neither moves the ranking, but the Engram one sits on the critical
path of blocks 1 and 14 and the direction of the mispricing is opposite in the two cases.

**How `batch_per_shard` is charged at batch 1.** Every batch-1 conclusion here — a's idle partner
device, c's idle engram shards — depends on a `batch x2` split at batch 1 leaving one shard with
zero samples and the group's time being set by the shard that has one. Worth confirming the cost
model does that rather than averaging, because if it averages, all the batch-1 points are
systematically optimistic for whichever scheme has the most batch legs, which is scheme-a.

**The workload the 480 MiB activation figure in `MODEL.md` refers to.** It is quoted per module
with no workload attached. Whether it is charged per co-resident placement at 8192 × batch 32
decides whether scheme-c's ~7 GiB of remaining bank headroom on d0-d7 is comfortable or marginal.
The gate says it fits at the sixteen measured points; nothing says anything about
`max_seq_len: 16384`, which the config supports and at which the four `kv_source` layers' compressed
caches double. Scheme-c is the scheme that would hit the 24 GiB wall first.

## 7. Per-configuration rankings

Three deltas decide every line: **D_moe** (`expert x8` vs `expert x4 * batch x2`, a against b/c),
**D_attn** (`head x8` on the five ratio-0 layers, c only), **D_engram** (`ngram x8 * batch x2`,
c only), **D_chunk** (80 vs 96, c only).

```
RANKING[prefill_128_b1]:    scheme-c > scheme-b > scheme-a
RANKING[prefill_128_b4]:    scheme-c > scheme-b > scheme-a
RANKING[prefill_128_b8]:    scheme-c > scheme-b > scheme-a
RANKING[prefill_128_b32]:   scheme-c > scheme-b > scheme-a
RANKING[prefill_8192_b1]:   scheme-b > scheme-c > scheme-a
RANKING[prefill_8192_b4]:   scheme-b > scheme-c > scheme-a
RANKING[prefill_8192_b8]:   scheme-b > scheme-c > scheme-a
RANKING[prefill_8192_b32]:  scheme-b > scheme-c > scheme-a
RANKING[decode_128_b1]:     scheme-c > scheme-b > scheme-a
RANKING[decode_128_b4]:     scheme-c > scheme-b > scheme-a
RANKING[decode_128_b8]:     scheme-c > scheme-b > scheme-a
RANKING[decode_128_b32]:    scheme-c > scheme-b > scheme-a
RANKING[decode_8192_b1]:    scheme-c > scheme-b > scheme-a
RANKING[decode_8192_b4]:    scheme-c > scheme-b > scheme-a
RANKING[decode_8192_b8]:    scheme-c > scheme-b > scheme-a
RANKING[decode_8192_b32]:   scheme-c > scheme-b > scheme-a
```

**`prefill_128_b1`** — 128 tokens, one sample, two chunks, 16 stages: almost pure pipeline fill, and
weight-bound (intensity ≈6 against break-even 81). D_chunk gives c the smaller first chunk (80 vs
96) and ~15% of the fill. D_moe is decisive against a, which additionally idles the partner device
of every stage. D_engram costs c a little here — eight of sixteen engram shards idle, doubled `wkv`
per active shard — but only two of 43 blocks have an engram.

**`prefill_128_b4` / `_b8`** — still weight-bound (23 and 47 FLOP/byte). Same order; D_engram now
helps c instead of hurting it, D_chunk still helps c, D_moe still separates a.

**`prefill_128_b32`** — crosses into compute-bound (188 FLOP/byte), which is scheme-a's best case:
its `batch x2` halves the per-token fp32 passes that expert parallelism replicates. Still not
enough — b/c read half the expert bytes *and* get 384-row expert GEMMs against a's 96 under a
128-row tile. c keeps the D_chunk fill advantage. This is a's narrowest loss among the
`prefill_128` points.

**`prefill_8192_b1`** — 103 chunks against 86, each re-reading ~71-78% of every layer's 384
experts, in the most weight-bound regime on the grid. D_chunk costs c roughly 5% here, its largest
penalty anywhere, and D_engram costs it a little more; D_attn does nothing, because layers 0 and 1
are 2 of 40 and `DSparkBlock.forward` at `start_pos == 0` only seeds the window cache
(`vendor/model.py:1123-1125`), so the three MTP blocks barely run during prefill. b wins. a is far
behind: 78% of 6.45 GiB per layer per chunk streamed from four banks instead of eight.

**`prefill_8192_b4` / `_b8`** — same shape, still weight-bound, so the extra 17 chunks still cost c
more than D_engram returns. b > c, narrowing as batch rises.

**`prefill_8192_b32`** — the closest call on the grid, and effectively a tie. D_chunk's penalty is
at its smallest (~3%, the per-chunk fixed cost amortized over 2560 tokens) and D_engram's gain at
its largest (half of ~344 GB of replicated fp32 gate traffic per table). I pick **b** because the
chunk-count penalty is the one I can bound from the hardware while the gate saving depends on an
effective vector-engine bandwidth I have no measurement for — but I would not defend the order, only
the tie.

**`decode_128_b1`** — the most weight-bound point on the grid and scheme-a's worst: the MoE streams
1.61 GiB per layer from four banks where b/c stream 0.806 GiB from eight, on 58.4% of the model.
D_attn gives c five layers at half the attention weight stream with the batch leg removed, against
a one-microsecond wider allreduce that carries no `index_score`. D_engram costs c a fraction of
that. c > b > a.

**`decode_128_b4` / `_b8` / `_b32`** — same ordering and the same mechanism. At 128 tokens every
layer's sweep is the 128-slot window, so nothing pulls against D_attn. At batch 32 the eight
micro-batches half-fill the pipeline and all three improve together, but a still holds 2× the
expert bytes per bank on the dominant term.

**`decode_8192_b1` … `_b32`** — unchanged order. At 8192 the compressed KV sweep enters for the 38
layers with `compress_ratio > 0` (8192 rows × 512 B per sample at ratio 1), which is precisely why
all three correctly keep those at `head x4` — a head split replicates the sweep. c's advantage
narrows to the five ratio-0 layers, but it is still an advantage, and a's deficit is unchanged.
D_chunk is irrelevant at decode. At `decode_8192_b32` the sweep becomes comparable to the active
expert bytes per shard, which compresses all three together; c > b remains, by the five layers.

**Ties worth naming rather than dressing up:** `prefill_8192_b32` for b against c, as above.
Everywhere else the b-vs-c margin is small but has a direction I can name from a mechanism.
Scheme-a is never a tie with either.

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
| `scheme-a` | 33.877 | 76.107 | 147.021 | 573.266 | 753.768 | 1766.021 | 3486.677 | 13810.928 | 11.873 | 12.969 | 15.763 | 32.529 | 14.740 | 23.416 | 27.495 | 51.967 | 1.1038 | 3 |
| `scheme-b` | 27.289 | 81.173 | 157.153 | 613.793 | 617.393 | 1867.114 | 3688.863 | 14619.674 | 11.002 | 11.820 | 14.340 | 29.461 | 13.868 | 22.267 | 26.181 | 49.667 | 1.0606 | 4 |
| `scheme-c` | 24.419 | 69.273 | 133.945 | 523.016 | 627.067 | 1851.414 | 3649.745 | 14440.525 | 10.715 | 11.530 | 14.021 | 28.971 | 13.581 | 21.977 | 25.891 | 49.377 | 1.0108 | 5 |

The score is the mean of every latency normalized against the best value any scheme
achieved for that metric, so it rewards being near the frontier across the whole grid
rather than winning one point.

## Where the two overall rankings agree

| rank | architectural analysis | simulator |
|---|---|---|
| 1 | `scheme-c` | `scheme-c` ✓ |
| 2 | `scheme-b` | `scheme-b` ✓ |
| 3 | `scheme-a` | `scheme-a` ✓ |

**3 of 3 positions agree.**

## Per-configuration rankings

The agent's call for each point beside the simulator's, best first. `—` means the
report did not rank that configuration; a missing opinion and an opinion that happens
to agree are different things and are not conflated here.

| configuration | architectural analysis | simulator | agree |
|---|---|---|---|
| `prefill_128_b1` | `scheme-c` > `scheme-b` > `scheme-a` | `scheme-c` > `scheme-b` > `scheme-a` | ✓ |
| `prefill_128_b4` | `scheme-c` > `scheme-b` > `scheme-a` | `scheme-c` > `scheme-a` > `scheme-b` | ✓ |
| `prefill_128_b8` | `scheme-c` > `scheme-b` > `scheme-a` | `scheme-c` > `scheme-a` > `scheme-b` | ✓ |
| `prefill_128_b32` | `scheme-c` > `scheme-b` > `scheme-a` | `scheme-c` > `scheme-a` > `scheme-b` | ✓ |
| `prefill_8192_b1` | `scheme-b` > `scheme-c` > `scheme-a` | `scheme-b` > `scheme-c` > `scheme-a` | ✓ |
| `prefill_8192_b4` | `scheme-b` > `scheme-c` > `scheme-a` | `scheme-a` > `scheme-c` > `scheme-b` | ✗ |
| `prefill_8192_b8` | `scheme-b` > `scheme-c` > `scheme-a` | `scheme-a` > `scheme-c` > `scheme-b` | ✗ |
| `prefill_8192_b32` | `scheme-b` > `scheme-c` > `scheme-a` | `scheme-a` > `scheme-c` > `scheme-b` | ✗ |
| `decode_128_b1` | `scheme-c` > `scheme-b` > `scheme-a` | `scheme-c` > `scheme-b` > `scheme-a` | ✓ |
| `decode_128_b4` | `scheme-c` > `scheme-b` > `scheme-a` | `scheme-c` > `scheme-b` > `scheme-a` | ✓ |
| `decode_128_b8` | `scheme-c` > `scheme-b` > `scheme-a` | `scheme-c` > `scheme-b` > `scheme-a` | ✓ |
| `decode_128_b32` | `scheme-c` > `scheme-b` > `scheme-a` | `scheme-c` > `scheme-b` > `scheme-a` | ✓ |
| `decode_8192_b1` | `scheme-c` > `scheme-b` > `scheme-a` | `scheme-c` > `scheme-b` > `scheme-a` | ✓ |
| `decode_8192_b4` | `scheme-c` > `scheme-b` > `scheme-a` | `scheme-c` > `scheme-b` > `scheme-a` | ✓ |
| `decode_8192_b8` | `scheme-c` > `scheme-b` > `scheme-a` | `scheme-c` > `scheme-b` > `scheme-a` | ✓ |
| `decode_8192_b32` | `scheme-c` > `scheme-b` > `scheme-a` | `scheme-c` > `scheme-b` > `scheme-a` | ✓ |

**The agent called 16 of 16 configurations and picked the simulator's winner in 13 of them.**

Where they differ, neither is authoritative. The simulator's constants are uncalibrated, and the architectural reading cannot see queueing or pipeline fill — but a disagreement concentrated in one region of the grid (all the batch-32 points, say, or only long context) is a specific and checkable claim about which effect is being missed.

The two methods pick the same winner. That is a genuine cross-check: an architectural argument from the datasheet and a discrete-event simulation of the timeline are not the same evidence, and agreeing on the top scheme means the result does not rest on the simulator's uncalibrated constants alone.

## Provenance of the hardware numbers

Probed on trn2.3xlarge at 2026-09-25T07:08:04.

Extrapolated rather than measured, because the dev host has one device:

- `links.inter_device.bandwidth_bytes_per_s`
- `links.inter_device.per_hop_latency_us`
- `collectives wider than the 4 logical cores of one device`

**`links.intra_device` is derived, not measured.** device HBM bandwidth (2.9e+12 B/s) x measured dma_large_contiguous (0.2770). NOT MEASURED. NKI 0.6.0 has no collective primitive and the XLA all-reduce fails to compile (NCC_ISMP902), so no intra-device collective can be timed on this host. Bounded by the on-chip path per text constraint 3.

Any conclusion that turns on whether tensor parallelism belongs inside a device or across devices rests on this number, and it is the first thing to measure on real 16-device hardware.
