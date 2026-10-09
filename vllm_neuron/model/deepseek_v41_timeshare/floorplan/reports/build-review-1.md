# Build review — iteration 1

Invariant verdict read from `.autohelix/floorplan/build-invariants-1.json`: **10/10 checks pass**
(a–j). Manifest artifact: `/home/ubuntu/workspace/partition-artifact` (target `trn2-16device`).

## What is left

Nothing is failing, so there is no invariant to unblock. But "no check failed" is not the same
question as "will the search be told the truth," so this section reports the honest gaps the
build's own `sim/NOTES.md` already flags, in the order they matter if a future change perturbs
the numbers:

1. **Prefill context scaling is structural, not a missing term (`sim/NOTES.md` §1).** The
   published-grid ratio (`prefill_8192_b1 / prefill_128_b1` = x24.1 for x64 tokens) looks
   sublinear only because `pipeline_chunks: true` overlaps `prefill_8192`'s four 2048-token
   chunks, so its makespan is really one chunk's dependency chain, not the whole context. A
   single non-chunked measurement at the same token count comes out superlinear (x75.9 for x64
   tokens). If a later iteration "fixes" this by inflating a context-dependent term to hit x64 on
   the chunked grid, that would be the regression to watch for — the honest fix, if any, is in the
   runner's chunking semantics, not in `attention.py` or `moe.py`.
2. **Decode's context sensitivity is real but thin** (`decode_8192_b1 / decode_128_b1` = 1.15).
   At 8192 tokens the compressed KV (8.5 MB/layer) is small against per-layer weight (126 MB), so
   decode stays weight-bound at every context in this grid. Correct given the shapes, but a search
   comparing two decode-heavy plans at different contexts will see almost no signal from this term
   — worth knowing before trusting a close decision on that axis.
3. **Cross-module KV placement is invisible** (`sim/NOTES.md` §3). The 34 layers that read
   another layer's `compress_kv` through `SharedAttentionRuntime` are charged against their own
   bank, not wherever `kv_source_layers` {2, 8, 14, 20} actually placed the cache — the partition
   graph carries no edge for that tensor. This makes "keep layers 2–19 near layer 2" invisible to
   the search as a placement decision, though NOTES.md is right that the workaround (charging
   `device_hbm`) would only make it falsely cheaper, not falsely expensive, so leaving it at the
   bank is the safer of the two errors.
4. **`experts_touched` is a closed-form uniform-routing expectation**, not measured routing. It
   only gates the off-bank expert-fetch term now (§4), so its influence is smaller than it once
   was, but it becomes load-bearing again if the deliberate `charge_bank=False` compromise on
   routed experts (next item) is ever revisited.
5. **MoE's routed-expert weight is explicitly not charged at the bank** (`moe.py` lines 193–212,
   `sim/NOTES.md` §1 under "What iteration 1 changed"). This is flagged by the model itself as
   "the least defensible line in the cost model": charging it would make `prefill_128` and
   `prefill_8192` both weight-bound and collapse the batch/context response the invariants check
   for (measured x5.15 batch response against x32 work). The note names the exact next step if
   this needs revisiting: "charge the routed experts too and accept the `prefill_128` columns."
   Anyone reviewing a future iteration should check whether this line moved and whether it was
   re-justified with a new measurement or just deleted.
6. **The torus's `intra_device`/`inter_device` bandwidth figures invert two of eight collective
   directions** (§5) — a hardware-model property (`systems/probed.yaml`'s extrapolated
   `intra_device` figure), not a cost-model bug, and NOTES.md shows the whole-plan effect still
   points the right way (scattering every TP group costs `prefill_8192_b1` 2.7x). Not urgent, but
   it is the one place the frozen hardware model, not the agent's arithmetic, could mislead the
   search on a close call.

None of these are invariant failures. They are the honest residue of a model that had to make
judgment calls where the vendor source and the frozen hardware model did not fully determine an
answer, and the build documented every one of them with the measurement that motivated it rather
than picking silently.

## Is the model honest?

Checked `sim/modules/*.py` against `<artifact>/modules/*/source.py` for the two largest modules by
parameter count and spot-checked the rest, reading as an adversary: is any duration a disguised
constant, is any module cheapened by a missing term, is anything divided by a split factor it
shouldn't be (or not divided by one it should), is KV/activation memory omitted, is any module
emitting near-nothing, is any NKI kernel latency smuggled in as a hardware rate.

**The two largest modules by parameter count** (measured from `partition_graph.yaml`: `mlp` totals
297.7 GB, the two `*.engram` modules alone total 203.6 GB — larger than attention's 5.5 GB total
across all 43 layers):

- **MoE** (`sim/modules/moe.py`). `config.json` gives `n_routed_experts=384`,
  `n_activated_experts=6`, `moe_inter_dim=2304`, `dim=5120`. The model computes the FLOPs: three
  GEMMs per assignment (`expert_w1`, `expert_w3`, `expert_w2`, lines 133–135) at
  `(assignments, inter, model_dim)` / `(assignments, model_dim, inter)`, where `assignments =
  ceil_div(tokens * activated * experts, routed)` (line 107) — i.e. `tokens * 6` assignments
  system-wide, correctly divided by the expert-split share. The shared expert's own three GEMMs
  (fp8, lines 137–139) and the gate (fp32, line 128) are present. Cross-checked against
  `MoE.forward` in `<artifact>/modules/00-MoE/source.py:311-326`: gate → per-expert dispatch →
  `expert(x[idx], weights)` (a full SwiGLU: `w1`, `w3`, `silu(gate)*up`, `w2`) → all-reduce →
  `+= shared_experts(x)`. All three matrices of both the routed and shared experts are costed;
  nothing is dropped.
- **Engram** (`sim/modules/engram.py`). Two 101.5 GB tables (`layers.1.engram`,
  `layers.14.engram`). The model charges 24 row lookups per token
  (`s.engram_hash_cols = (max_ngram_size-1)*n_heads = 3*8 = 24`, line 77 via `_shapes.py:115`) at
  264-byte rows (`engram_head_dim=256` fp8 + its e8m0 scales, `quantized_row_bytes`), then a real
  `wkv` GEMM (`tokens, out_width, contract, "fp8"`, line 110) with `out_width = dim*(hc_mult+1) =
  5120*5 = 25600` and `contract = engram_hash_cols * head_dim = 24*256 = 6144` — matching
  `Engram.__init__` in `<artifact>/vendor/engram.py:250-252` (`ParallelEngramEmbedding(...,
  layout.head_dim)`, `wkv = Linear(n_hash_cols * layout.head_dim, dim*(hc_mult+1))`) exactly. The
  gate's four-pass elementwise cost (line 117) matches `Engram.forward`'s two rsqrt reductions,
  the weighted dot product, and the signed-sqrt sigmoid (`vendor/engram.py:264-272`). Row lookups
  are priced as random access (line 102, correctly: 264 B rows scattered across 384M rows is
  access-bound, not bandwidth-bound), and the dense `wkv` weight is charged separately from the
  table so the table isn't double-counted (lines 123–136).

**Split factors.** Every module routes tensor widths through `share()`/`ctx.shard_of()`
(`_shapes.py:216-226, 454-476`), which excludes `COMMUNICATION_FREE_DIMS` (batch) from the general
divide and requires a named dimension for anything else — the exact mechanism `sim/NOTES.md`
documents catching itself getting wrong twice (attention's `head`/`hidden` conflation coming out
at 0.92x instead of 0.5x; MoE/Engram's un-sequence-parallelized norms coming out at 0.74x/0.96x
instead of dividing). Both are described as found-and-fixed with the before/after ratio, which is
the kind of check an adversarial reviewer would otherwise have to do by hand.

**Missing modules made free.** `finish()` (`_shapes.py:535-556`) always returns at least
`ctx.deps` even when a shard emits zero ops, so a shard can't vanish from the dependency graph, and
the invariant's own finding for check (d) is "271 module(s) contributed work" — all of them,
including the deliberately near-free `other_globals` (`unclassified.py`), which still pays for
its bank occupancy and its token-id scan rather than costing exactly zero.

**KV and activation memory.** `charge_kv` appears in `attention.py` (window ring + compressed
latents + indexer key, lines 336/343/345) and nowhere else — correctly, since it's the only
archetype with a persistent cache. `charge_activation` appears in every module. No plan can look
like it fits in memory by omission.

**NKI kernel latencies.** No file under `sim/modules/` reads a recorded kernel latency or any
NKI-measured number; the only NKI references are `_shapes.py`'s `QUERY_TILE` comment (the SBUF
partition count, a hardware constant read from the system YAML, not a measured kernel time) and
`sim/framework/engine.py`'s comment naming NKI's five engines. Durations are produced exclusively
through `ctx.matmul_seconds` / `elementwise_seconds` / `dma_seconds` / `gather_seconds` /
`collective_cost`, which read rates from the system YAML and the probe overlay — never from a
bootstrapped kernel trace. `sim/NOTES.md` states the same thing was checked mechanically
(`reports/probe.py:check_no_fabricated_durations`, "0" findings).

**Hardcoded shapes.** Every dimension traced back to a `config.json` key or a vendor
module-level constant (`_shapes.py:8-10, 24-29`); none of the GEMMs in `moe.py` or `attention.py`
use a literal in place of a config-derived value.

The one place the model deliberately declines to charge real work — MoE's routed-expert bank
read — is disclosed, measured, and justified in both the module's own comment (`moe.py:192-212`)
and `sim/NOTES.md` (top of the "uncertain" list), with the exact alternative and its cost named.
That is the opposite of hiding a shortcut.

VERDICT: clean
