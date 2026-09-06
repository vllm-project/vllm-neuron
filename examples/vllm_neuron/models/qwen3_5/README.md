# Qwen3.5 on Neuron

Support for **Qwen3.5-2B** and **Qwen3.5-27B** (dense, via
`Qwen3_5ForConditionalGeneration`) and **Qwen3.5-35B-A3B** (sparse, via
`Qwen3_5MoeForConditionalGeneration`), and **Qwen3.5-397B-A17B** (sparse, 512
experts top-10) on a trn2.48xlarge at `world_size = 64, ep_degree = 8`. Sizing and
the constraints that fix that layout are under *Expert parallelism* below.

Qwen3.5 is a *hybrid* decoder. Most layers are gated DeltaNet, a linear
recurrence carrying a fixed-size state, and the rest are ordinary full
attention:

| | layers | DeltaNet | full attention | hidden | heads (q/kv) | head_dim |
|---|---|---|---|---|---|---|
| Qwen3.5-2B | 24 | 18 | 6 | 2048 | 16 / 2 | 128 |
| Qwen3.5-27B | 64 | 48 | 16 | 5120 | 24 / 4 | 256 |
| Qwen3.5-35B-A3B | 40 | 30 | 10 | 2048 | 16 / 2 | 256 |
| Qwen3.5-397B-A17B | 60 | 45 | 15 | 4096 | 32 / 2 | 256 |

That makes it the first model in this plugin to need **two KV cache groups** —
one paged group for the attention layers and one recurrent group whose "blocks"
are per-sequence states — so most of the shared-code changes in this PR are in
the cache plumbing rather than in the model.

## Quick start

```bash
# 2B, TP=4
python examples/vllm_neuron/models/qwen3_5/run.py \
    --model <path-to-checkpoint>/Qwen3.5-2B

# 27B, TP=4 — both extra flags are required on a 4-core instance, see below
python examples/vllm_neuron/models/qwen3_5/run.py \
    --model <path-to-checkpoint>/Qwen3.5-27B \
    --gpu-memory-utilization 0.65 --optlevel 3

# 35B-A3B, TP=4 — memory cap only; the default -O1 compiles fine here
python examples/vllm_neuron/models/qwen3_5/run.py \
    --model <path-to-checkpoint>/Qwen3.5-35B-A3B \
    --gpu-memory-utilization 0.72

# 35B-A3B with expert parallelism, ep_degree=2 x tp_degree=2. Slower than the
# line above and no smaller; see *Expert parallelism* for why it exists.
python examples/vllm_neuron/models/qwen3_5/run.py \
    --model <path-to-checkpoint>/Qwen3.5-35B-A3B \
    --gpu-memory-utilization 0.72 --expert-parallel --ep-degree 2

# 397B-A17B, on a whole trn2.48xlarge. All four settings are required: 64 ranks
# to fit the weights, ep_degree=8 to make 64 legal, the memory cap so the KV
# planner leaves room for activations, and a patched nkilib for top-k > 8.
# See *Expert parallelism*.
python examples/vllm_neuron/models/qwen3_5/run.py \
    --model <path-to-checkpoint>/Qwen3.5-397B-A17B \
    --tensor-parallel-size 64 --expert-parallel --ep-degree 8 \
    --gpu-memory-utilization 0.7
```

`enable_prefix_caching` must stay off (see *Known limitations*).

The examples decide one compiler option **from the checkpoint**, in
`mac_threshold_override()`. For a *dense* checkpoint they pass
`hlo2tensorizer_options: ""`, suppressing the runner's default
`--modular-flow-mac-threshold=10`: with that flag neuronx-cc fails codegen on
the dense decode graph (`NCC_IBTN006`, a `pftranspose` copy that fails backend
verification). Confirmed by recompiling the cached HLO by hand — it fails with
the flag and succeeds without it. The flag exists because NKI kernels do not
report MAC counts, and the dense path calls none. **The sparse checkpoint must
keep it**: its MoE block calls three kernels per layer. Getting this backwards
in either direction is a compile failure, which is why it is keyed off
`num_experts` rather than a command-line flag.

## The two 27B launch settings, and why

Both were needed to get 27B up on a trn2.3xlarge and neither is guessable from
the error message.

**`--gpu-memory-utilization 0.65`.** HBM on Trn2 is **24 GB per logical
NeuronCore**, not 96 GB pooled, and neuronx-cc checks weights + caches +
activations against that per-core limit. At TP=4 the 27B weights are 13.5
GB/rank, so at the default `0.9` the worker hands the block planner
`24 * 0.9 - 13.5 = 8.1 GB` — which vLLM then *fills*: 542 recurrent blocks,
`f32[542, 12, 128, 128]` per layer × 48 layers = 20.5 GB, and the graph needs
25.6 GB. It fails with

```
[NCC_EVRF009] Size of total input and output tensors exceeds HBM limit
```

The blocks are pure waste: `mamba_block_size == max_model_len`, so at
`max_num_seqs=4` only ~5 recurrent blocks are ever reachable.

**`--optlevel 3`.** 27B's decode graph fails ISA validation at `-O1`:

```
[NCC_IINAR001] ISA validation failed: Copy;
inst failed assertion check: 'start_addr_active_channels'
```

Isolated by hand-recompiling the preserved `graph.hlo` (the failed compile
directory is the one with no `.neff`; `command.txt` has the exact invocation):
`-O1` fails, `-O2` and `-O3` both compile, with and without the fp8 cast flag.
So it is the optimization level, not an internal option. `-O2` is unreachable
from the offline API — `apply_config_platform_defaults` lowers vLLM's default
`O2` to `O1` and cannot distinguish an explicit offline `O2` from the default,
because it sniffs `sys.argv` — and its own docstring points at `O3`, hence
`--optlevel 3`. This costs compile time (below); only the decode graph needs
the higher level, so a per-graph optimization level would recover most of it.

2B does not need either setting. 35B-A3B needs the memory cap
(`--gpu-memory-utilization 0.72`, see below) but **not** the higher optimization
level — `NCC_IINAR001` does not recur on its graphs, and the plugin's default
`-O1` compiles.

## The sparse checkpoint (35B-A3B)

40 layers, 256 experts of `moe_intermediate_size` 512, top-8, plus one shared
expert behind a sigmoid gate. Only the MLP differs from the dense stack: the
hybrid DeltaNet/attention decoder, the zero-centred norms, the gated attention
output and the partial interleaved mRoPE are all reused unchanged, and
`Qwen3_5SparseMoeBlock` is dropped in where `Qwen3_5MLP` sits.

Both parallel layouts are implemented, and **35B-A3B should use tensor
parallelism** — the default. Expert parallelism is a net loss at this width and is
here for the wider checkpoint; *Expert parallelism* below has the numbers.

Four details give plausible-but-wrong output rather than a crash, so they are
worth stating explicitly. All four are verified against HF on CPU to 5e-7:

1. **The decode kernel fuses the RMSNorm**, so the block takes the
   *unnormalised* residual plus `gamma = (1.0 + norm.weight).float()`.
   `Qwen3_5RMSNorm` is zero-centred and the kernel's norm is not; passing an
   already-normed input with `gamma = ones` is provably wrong, because
   `normalize(x) * (1 + w)` does not have unit RMS.
2. **The shared expert stays outside the kernel.** The kernel would add that
   branch in internally, but Qwen3.5 first scales it by
   `sigmoid(shared_expert_gate(x))` — an input-dependent scalar that cannot be
   folded into the weights.
3. **Routing is softmax over all 256 logits, then top-k, then L1
   renormalise** (`PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER`), not the kernels'
   default top-k-then-softmax. The two disagree numerically.
4. **Expert parallelism must force the kernel's all-expert path.**
   `moe_block_tkg` only consults `rank_id` when it is loading every expert, so on
   the selective path it would treat global expert ids as local ones.

`BLOCK_SIZE` is 128 rather than gpt-oss's 256: at `T = 2048` that is 381 blocks
at 33.6% occupancy against 318 at 20.1%.

## Expert parallelism, and why 35B-A3B does not want it

`--expert-parallel` switches the block from sharding each expert's intermediate
dimension across all ranks (TP) to giving each rank a disjoint `1/ep_degree` of
the experts, sharded across the `tp_degree = world_size / ep_degree` ranks of its
own partition (EP). `--ep-degree` sets the degree — unset, it resolves to the
world size, i.e. `tp_degree = 1`.

**It is not a speed knob.** Per-rank prefill work `N * block_size * I_local` at
`T = 1024` on 35B-A3B over 4 ranks is 318 blocks × 128 wide = 5.21M at `ep=1`,
191 × 256 = 6.26M at `ep=2`, 127 × 512 = 8.32M at `ep=4`. It *rises* with
`ep_degree`, because `build_blockwise_mapping` bounds blocks at `T * top_k`
whatever `ep_degree` is — worst case, as if every assignment landed on this rank —
while the per-rank intermediate width grows with it. Nor does it save memory:
weights shard `world_size` ways either way, and the all-expert decode read
`E_local * 3 * I_local * H` is `ep_degree`-independent because
`E_local * I_local = E * I / world_size`.

What it does is make a wide `world_size` **legal**, which is the only reason it is
here:

```
397B-A17B, bf16, text-only weights = 738.25 GiB (measured: the 751.39 GiB
checkpoint less the vision tower and the MTP head)

world_size   attention (32 q heads)   I/tp_degree, pure TP   weights/rank
     8       ok                       128  ok                92.3 GiB
    16       ok                        64  fails             46.1 GiB
    32       ok                        32  fails             23.1 GiB
    64       replicated, 2 ranks/head  16  fails             11.5 GiB
```

The fused decode kernel needs `moe_intermediate_size / tp_degree` to be a multiple
of 128, which caps *pure TP* at 8 ranks; 738 GiB needs at least 32 of this
device's cores, whose usable ceiling is **23.50 GiB** each (a hard partition — the
other three cores of a 96 GB device being idle does not raise it). Empty
intersection. EP breaks it — at
`world_size = 64, ep_degree = 8, tp_degree = 8` every guard passes — which also
gives the rule to size by: **take the smallest `ep_degree` that clears the 128
rule, not the largest**, since per-rank work and the number of graphs to compile
both grow with it while footprint does not. The block logs a warning if you leave
the degree at the world size and something smaller would do.

**397B-A17B runs at `world_size = 64, ep_degree = 8`** (so `tp_degree = 8`), which
is the only layout that fits: at 32 its 23.1 GiB/rank leaves ~0.4 GiB of the 23.50
GiB core for KV and activations. 64 ranks is wider than its 32 query heads and its
16 DeltaNet key heads, so those are *replicated* — see the replication notes in
`Qwen3_5Attention.__init__` and `Qwen3_5GatedDeltaNet.__init__`; only a dimension a
row-parallel projection reduces over has to shard cleanly, because the tp
all-reduce sums the per-rank partial products. Measured: **11.93 GiB used / 12.07
GiB free per rank**, engine init 640 s of which 633 s is compilation, correct
greedy output.

Two things it needs from the caller, neither of which the defaults give:

- **`--gpu-memory-utilization 0.7`.** The KV budget is `core capacity * gmu -
  weights` and the planner spends all of it, which for a hybrid model means
  recurrent-state blocks far past `max_num_seqs` — at the default, ~4825 blocks
  for a 4-sequence run, and `neuronx-cc` rejects the graph with `[NCC_EVRF009] ...
  Needed 27,029,780,748 bytes (25 GB) vs. available 25,769,803,776 bytes (24 GB)`.
  The budget does not count the ~4.7 GB of activations, so leave that headroom.
- **A patched `nkilib`.** `nkilib.core.router_topk.router_topk` hard-caps top-k at
  8 (`nisa.max8` / `nisa.nc_find_index8` are 8-per-pass DVE instructions), and
  397B-A17B routes top-10, so the fused decode kernel `moe_block_tkg` cannot serve
  it as shipped. A `ceil(k/8)`-pass cascade over a scratch SBUF copy, erasing each
  pass's winners with `nisa.nc_match_replace8(imm=-inf)` — the idiom
  `vendored_kernels/rotational_topk` already uses — lifts it, and matches
  `nkilib`'s own torch reference at K=10 and K=16 to the same 0.007 relative error
  as the untouched K=8 path. That fix belongs in `nkilib`, not here, and has been
  reported; it is not part of this branch.

Spreading experts across *data-parallel replicas* is still not supported — this
block reduces over the tensor-parallel group only, and raises rather than silently
dropping off-replica tokens.

## Measured

trn2.3xlarge, TP=4, bf16, greedy, 1024-token input / 128-token output,
`max_model_len` 2048, 3 timed rounds after a discarded warmup round, via
`benchmark_latency.py`:

**Qwen3.5-2B**

| batch | TTFT mean | TPOT | E2E | output tok/s (aggregate) | per-stream tok/s |
|---|---|---|---|---|---|
| 1 | 145.3 ms | 3.88 ms | 638 ms | 200.5 | 257.5 |
| 4 | 353.5 ms | 7.62 ms | 1321 ms | 385.5 | 131.3 |
| 8 | 624.3 ms | 12.35 ms | 2192 ms | 464.8 | 81.0 |

TTFT *min* stays at 145 ms for every batch and TTFT *max* grows linearly
(559 ms at batch 4, 1097 ms at batch 8) because prefills are serialised one
bucket at a time.

**Qwen3.5-35B-A3B**, same harness:

| batch | TTFT mean | TPOT | E2E | output tok/s (aggregate) | per-stream tok/s |
|---|---|---|---|---|---|
| 1 | 328.2 ms | 36.17 ms | 4922 ms | 26.0 | 27.7 |
| 4 | 860.4 ms | 77.15 ms | 10658 ms | 47.9 | 13.0 |
| 8 | 1544.4 ms | 142.69 ms | — | 52.0 | 7.0 |

**Decode on the sparse checkpoint is about 6x slower than the active-parameter
count predicts, and that is the top optimization target here, not a property of
the model.** 35B-A3B activates ~3B parameters against 2B's 2B — 1.5x the FLOPs —
yet TPOT is 9.3x worse (36.17 vs 3.88 ms). TPOT also grows roughly linearly with
batch (1 : 2.1 : 3.9), so what dominates is the per-token expert *gather*, not a
fixed weight read. Note `SELECTIVE_LOADING_THRESHOLD = 1.0` means the
all-expert path never engages below batch 32 (`tokens * top_k / num_experts >= 1`
needs 256/8), so every row above took the selective path; forcing the all-expert
path at small batch is untried and is the obvious next measurement.

Memory, at `--gpu-memory-utilization 0.72`: `Neuron HBM: 16.19 GiB used, 7.81 GiB
free` per rank, matching the predicted 16.177 GiB of weights, leaving a 43,417-token
KV cache with the page geometry resolved to block_size 32 → 544.

Cold compile is one-time and cached: 2B ≈ 213 s at `max_model_len` 1024 (77 s
prefill graph + 136 s decode) at the default `-O1`; **27B ≈ 2840 s** at
`max_model_len` 2048 at `-O3`, dominated by the 64-layer prefill graph. A warm
`VLLM_CACHE_ROOT` brings 27B engine init down to ~132 s. 35B-A3B is *cheap* by
comparison — two graphs, 33.7 s + 220.7 s, 353 s of engine init in total — because
`-O1` suffices and the kernels replace what would otherwise be a very wide MLP
graph.

Accuracy on 35B-A3B: 137/160 greedy tokens (85.6%), 3/5 prompts exact, no
first-token mismatch. The reference there is **HF in bfloat16, not float32**,
because float32 would be ~140 GB against 124 GiB of host RAM; that weakens the
comparison symmetrically rather than in our favour, and the check it exists for —
"is the first token right" — survives it.

Expert parallelism on the same box, 35B-A3B on 4 ranks:

| | TP=4 | EP=2 × TP=2 | EP=4 |
|---|---|---|---|
| HBM used per rank | 16.19 GiB | 16.19 GiB | 16.19 GiB |
| distinct graphs | 2 | 4 | 8 |
| rank-0 cold compile | 254 s | 861 s | 3242 s |
| greedy tokens vs HF bf16 | 137/160 | 156/160 | 119/160 |
| prompts exact | 3/5 | 3/5 | 2/5 |

Identical footprint at every degree, as predicted. **Cold compile is where
`ep_degree` shows up**, and worse than linearly: the number of distinct graphs is
`2 * ep_degree`, because `self.ep_rank` enters the trace as a constant — the local
expert range and the kernel's `rank_id` are both built from it — so every EP
partition gets its own HLO and rank 0 compiles all of them. Each is individually
slower too, the per-rank intermediate being `ep_degree` times wider. Warm cache is
unaffected.

**Read the accuracy row as noise, not as a ranking.** All three layouts agree on
three of the five prompts; the whole 119–156 spread comes from the other two,
where the model is near-tied and each layout falls a different way:

| prompt | TP=4 | EP=2 × TP=2 | EP=4 |
|---|---|---|---|
| `def fibonacci(n):` | 12/32 | 31/32 | 12/32 |
| `Once upon a time, there was a` | 32/32 | 32/32 | 14/32 |

A different collective layout reduces bf16 partials in a different order, and
greedy decoding turns a near-tie into a visible fork. Five prompts cannot resolve
that, so the hybrid beating TP here says as little as EP=4 trailing it does. The
real correctness evidence is the float32 CPU check: one block per rank, each
loaded through its own weight loaders, summed over the ranks, agreeing with HF to
5e-7 on both phases with each partition's routed contribution asserted distinct.

Accuracy: `check_generation_vs_hf.py` compares greedy tokens against HF on CPU
and reports both the exact-match prefix length and the token match rate. A
mismatch on the *first* token is a prefill bug; later divergence is expected,
because bf16 and an independent implementation eventually disagree on a
near-tie and greedy decoding amplifies it.

## Sizing the prefill bucket

`num_batched_tokens_buckets` is the list of prefill token counts the plugin
compiles, and every prompt pads up to the nearest one. Left alone the plugin
defaults to powers of two from 128 up to `max_num_batched_tokens`, so a short
prompt lands on a small bucket. `--prefill-buckets` overrides that list and takes
`max_num_batched_tokens` from its largest entry.

Pinning a *single* bucket is the case to watch, not least because it is also how
you would spell NxDI's old `enable_bucketing=False`. With one 2048 bucket a
1024-token prompt is padded to 2048 -- twice the work the request needs -- and on
27B the tiles stop fitting the 24 MB SBUF. A device profile of that graph
attributes **78% of all HBM reads to SBUF spill reloads**: 108.7 GB of reloads
against 13.5 GB/rank of weights, with spill making up 81% of total HBM traffic.
Padding alone would predict 2.0x; the spill cliff on top of it makes the measured
cost 2.8x.

27B at batch 1 on a 1024-token prompt, TP=4:

| prefill buckets | TTFT | first compile |
|---|---|---|
| `2048` pinned (= `max_model_len`) | 789 ms | 84 min |
| `1024` | **284 ms** | 37 min |
| `512` | 287 ms | 21 min |

TPOT is unchanged at 48.5 ms and the decode NEFF is byte-identical, so this is
prefill-only. Compile time falls along with the bucket because every bucket is a
separate prefill graph and the prefill graph dominates the 27B compile.

The same setting helps 35B-A3B less: TTFT 328 -> 230 ms at batch 1, with TPOT
unchanged at 36.17 ms. It activates only ~3B parameters per token, so its prefill
was never as SBUF-bound. 27B beats the 2.0x padding bound because of the spill
cliff on top of the padding; 35B-A3B falls short of it because fixed per-request
overhead is a larger share of its much smaller prefill.

The win holds at batch 4: median TTFT goes from 2101 ms to 829 ms. Prefills
serialise at batch > 1 (see the note below), so TTFT spreads out -- 281 ms for the
first request in the batch, 1236 ms for the last -- but the per-prefill cost is the
same as at batch 1.

Guidance:

- Leave `--prefill-buckets` unset unless you have a reason to set it. The default
  list already covers short prompts.
- Pin a narrow list when compile time matters, using the table above as the guide.
- Which values are legal turns on whether the largest bucket equals
  `max_model_len`. If it does, prefill is single-shot and any strictly ascending
  list is accepted, 128 and 256 included. If it is smaller, Neuron auto-enables
  its segmented-attention kernel, the list must match `kv_segment_size_buckets`,
  and values are restricted to `[512, 1024, 2048, 4096, 8192]` -- 512 is the floor
  in that case only.
- `1024` in the table is the second case, and for a 1024-token prompt it is a
  single chunk, so none of the numbers above measure multi-chunk prefill. When
  chunking does happen it is cheap: 2x512 costs 2.6% more graph time than 1x1024
  (255.9 vs 249.5 ms).

vLLM enables chunked prefill by default, and the Neuron platform logs that it
"only supports chunking prefills with batch size of 1" and will not mix prefill
and decode in the same batch. That warning appears for every configuration here,
pinned or default, so it is not a consequence of this setting.

## Known limitations

Stated plainly rather than left to be discovered:

- **`enable_prefix_caching` must be `False`.** Prefix caching requires
  segmented prefill here, and a DeltaNet layer has no reusable prefix anyway —
  its prefill starts from a zero state. Not verified in any other
  configuration.
- **One batch bucket per process.** Supplying several
  `num_seqs_buckets` hangs during warmup. Measure each batch size in a separate
  process.
- **Decode gathers the full block-table width** rather than only the occupied
  pages. This is the largest known performance item on the dense path and is
  not addressed here.
- **27B at `-O1` miscompiles** (above). `-O3` is a workaround with a real
  compile-time cost, not a fix.
- **`head_dim = 256` on 27B and 35B-A3B exceeds the flash-attention kernel's
  `MAX_HEAD_DIM = 128`**, so their full-attention layers fall back to the torch
  path. Measured at 6–8% of prefill, so it has been left alone.
- **The vision-language path is wired but lightly exercised.** `vl.py` reuses
  the Qwen3-VL encoder, weight loading, block packing and mRoPE helper directly
  rather than copying them; `check_generation_vs_hf.py --vl` compares a
  single-image prompt against HF. The numbers above are text-only.
- **On the sparse checkpoint, decode is gather-bound** (above). Untried:
  forcing the all-expert kernel path at small batch.
- **The sparse expert loaders read `down_proj` chunked over 32 experts at a
  time.** Sharding `intermediate_size` is a last-dim slice, so a per-expert read
  would become `num_experts * hidden_size` tiny safetensors reads.
- **Experts cannot span data-parallel replicas.** `ep_degree > 1` with
  `data_parallel_size > 1` raises: this block reduces over the tensor-parallel
  group only, so tokens routed off-replica would be dropped silently rather than
  loudly. 397B-A17B does not need it: expert parallelism alone reaches all 64
  cores of a trn2.48xlarge.
- **397B-A17B has functional validation only** — correct greedy output at
  `world_size = 64`, no latency, throughput or per-token HF comparison. It also
  needs a patched `nkilib` for top-k > 8 and `--gpu-memory-utilization 0.7`; both
  are under *Expert parallelism*.

## Attribution

This work derives from the Qwen3.5 port in
[`qingzwang/vllm-neuron`](https://github.com/qingzwang/vllm-neuron), branch
`model/Qwen3.5-2B` (Apache-2.0), which is where the implementation of 2B
originates. `nki_deltanet.py`, `nki_deltanet_fused.py`, `vl.py` and `flags.py`
are that fork's files unchanged, and the bulk of `model.py`, `deltanet.py` and
`config.py` is its work as well.

Forward-ported to `release-0.24.0.1.1.0` and extended to 27B and to the sparse
35B-A3B checkpoint by the submitter; `moe.py` is entirely new. See the PR
description for the specific changes.

## Files

```
examples/vllm_neuron/models/qwen3_5/
  run.py                     offline text generation (dense and sparse)
  benchmark_latency.py       streaming TTFT / TPOT / E2E via AsyncLLM
  check_generation_vs_hf.py  greedy tokens on device vs HF on CPU
```
