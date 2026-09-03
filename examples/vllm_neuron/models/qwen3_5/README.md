# Qwen3.5 (dense) on Neuron

Support for the dense Qwen3.5 checkpoints — **Qwen3.5-2B** and **Qwen3.5-27B** —
via `Qwen3_5ForConditionalGeneration`.

Qwen3.5 is a *hybrid* decoder. Most layers are gated DeltaNet, a linear
recurrence carrying a fixed-size state, and the rest are ordinary full
attention:

| | layers | DeltaNet | full attention | hidden | heads (q/kv) | head_dim |
|---|---|---|---|---|---|---|
| Qwen3.5-2B | 24 | 18 | 6 | 2048 | 16 / 2 | 128 |
| Qwen3.5-27B | 64 | 48 | 16 | 5120 | 24 / 4 | 256 |

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
```

`enable_prefix_caching` must stay off (see *Known limitations*), and the
examples pass `hlo2tensorizer_options: ""` to suppress the runner's default
`--modular-flow-mac-threshold=10`: with that flag neuronx-cc fails codegen on
this model's decode graph (`NCC_IBTN006`, a `pftranspose` copy that fails
backend verification). Confirmed by recompiling the cached HLO by hand — it
fails with the flag and succeeds without it. The flag exists for NKI kernels,
and the dense path calls none.

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

2B does not need either setting.

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

Cold compile is one-time and cached: 2B ≈ 213 s at `max_model_len` 1024 (77 s
prefill graph + 136 s decode) at the default `-O1`; **27B ≈ 2840 s** at
`max_model_len` 2048 at `-O3`, dominated by the 64-layer prefill graph. A warm
`VLLM_CACHE_ROOT` brings 27B engine init down to ~132 s.

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
- **`head_dim = 256` on 27B exceeds the flash-attention kernel's
  `MAX_HEAD_DIM = 128`**, so its full-attention layers fall back to the torch
  path. Measured at 6–8% of prefill, so it has been left alone.
- **The vision-language path is wired but lightly exercised.** `vl.py` reuses
  the Qwen3-VL encoder, weight loading, block packing and mRoPE helper directly
  rather than copying them; `check_generation_vs_hf.py --vl` compares a
  single-image prompt against HF. The numbers above are text-only.
- **Sparse (MoE) Qwen3.5 checkpoints are not in this PR.** They need a
  `Qwen3_5SparseMoeBlock` and a second registry entry; that is a follow-up.

## Attribution

This work derives from the Qwen3.5 port in
[`qingzwang/vllm-neuron`](https://github.com/qingzwang/vllm-neuron), branch
`model/Qwen3.5-2B` (Apache-2.0), which is where the implementation of 2B
originates. `nki_deltanet.py`, `nki_deltanet_fused.py`, `vl.py` and `flags.py`
are that fork's files unchanged, and the bulk of `model.py`, `deltanet.py` and
`config.py` is its work as well.

Forward-ported to `release-0.24.0.1.1.0` and extended to 27B by the submitter;
see the PR description for the specific changes.

## Files

```
examples/vllm_neuron/models/qwen3_5/
  run.py                     offline text generation (2B, 27B)
  benchmark_latency.py       streaming TTFT / TPOT / E2E via AsyncLLM
  check_generation_vs_hf.py  greedy tokens on device vs HF on CPU
```
