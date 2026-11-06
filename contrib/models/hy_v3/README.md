# Contrib Model: Hy3 (tencent/Hy3-FP8)

Serves [tencent/Hy3-FP8](https://huggingface.co/tencent/Hy3-FP8) with vLLM on
AWS Trainium through the `vllm-neuron` plugin. Hy3 is a ~295B-parameter MoE
decoder (80 layers, 192 routed experts top-8 + 1 shared expert, GQA 64Q / 8KV)
with a static per-tensor FP8 checkpoint. All linear layers run with the
checkpoint's FP8 weights; the KV cache is BF16.

The architecture, kernel mapping and feature status are in
[src/README.md](src/README.md). A slide-deck summary (in Chinese: model,
how it is split over the 64 ranks, where the time goes, and each
optimization with its measured effect) is
[docs/PRESENTATION.html](docs/PRESENTATION.html), a self-contained page to
open in a browser.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

## Usage

The model needs the full trn2.48xlarge (TP=64 over 64 logical NeuronCores,
`--enable-expert-parallel`, full EP=64).

### Install / register

vLLM already has a GPU implementation registered as `HYV3ForCausalLM`, and the
vLLM engine core and the Neuron workers run in spawned processes, so the Neuron
implementation has to be registered in every process. This folder is a small
Python package that does that through a `vllm.general_plugins` entry point:

```bash
pip install -e contrib/models/hy_v3     # imports as `hy_v3_neuron`
```

After that, both `vllm serve` and the offline `LLM(...)` API pick up the Neuron
implementation with no further code; the integration test relies on it too
(`from hy_v3_neuron import register`). No file outside this folder is changed.

How this differs from [contrib/README.md](../../README.md):

- It is neither Option 1 (`ModelRegistry.register_model` in your own script)
  nor Option 2 (adding the model to `vllm_neuron/model/registry.py`). Option 1
  is not enough here: the call only registers the architecture in the calling
  process, vLLM's own GPU `HYV3ForCausalLM` stays registered under the same
  name, and the engine core and the Neuron workers are spawned processes. The
  entry point is the "register from your own plugin" route the guidelines
  allow, run by vLLM in every process. `register()` is the single function to
  call if you copy `src/` into your own package instead.
- Besides the `ModelRegistry` entry, `register()` appends `"fp8"` (the
  checkpoint's `quant_method`) to `NeuronPlatform.supported_quantization` at
  runtime, because vLLM rejects a checkpoint whose quantization the platform
  does not list and the plugin has no public hook for adding one. vLLM's own
  FP8 linear methods are never used: the model consumes the FP8 weights and
  scales itself. This is the one runtime touch-point beyond the registry.

### Serve with the OpenAI-compatible server

`scripts/serve.sh` and `scripts/serve_latency.sh` wrap the configurations used
for the results below. A minimal launch:

```bash
export NEURON_SKIP_EFA_AFFINITY=1          # only on instances without EFA
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.45
vllm serve tencent/Hy3-FP8 \
    --served-model-name hy3 \
    --tensor-parallel-size 64 \
    --enable-expert-parallel \
    --max-model-len 16384 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 2048 \
    --block-size 32 \
    --enable-prefix-caching \
    --additional-config '{"neuron_config": {
        "num_batched_tokens_buckets": [2048],
        "kv_segment_size_buckets": [2048],
        "num_seqs_buckets": [1, 4, 16, 32],
        "decode_context_length_buckets": [4096, 8192]}}'
```

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "hy3", "messages": [{"role": "user", "content": "What is 17*23?"}],
         "max_tokens": 64}'
```

The first start compiles the model (about 50 minutes on a cold cache).

### Offline inference with the Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="tencent/Hy3-FP8",
    tensor_parallel_size=64,
    enable_expert_parallel=True,
    max_model_len=4096,
    max_num_seqs=4,
    block_size=32,
)
outputs = llm.generate(["The capital of France is"], SamplingParams(max_tokens=32))
print(outputs[0].outputs[0].text)
```

### Configuration notes

- Keep `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` at 0.45 or lower. Some NEFFs
  are loaded lazily on first use and need device memory for their DMA rings;
  at 0.6 the server ran out of memory after about 40 minutes.
- `max_model_len` is always an implicit decode context bucket, so
  `decode_context_length_buckets` must stay strictly below it.
- Set `decode_context_length_buckets` whenever `max_model_len` is well above
  the typical context. Without it each decode batch bucket is a single NEFF
  that reads `max_model_len` worth of KV blocks per request on every step,
  whatever the request's real length (vllm-neuron 0.21+ makes this opt-in;
  the NxDI-based 0.2.x bucketed decode by sequence length by default). The
  configs below all set it.
- Not supported: MTP together with DCP (rejected at startup). Attention DP
  (`--data-parallel-size` with `--enable-expert-parallel`) is experimental and
  works without MTP; see "Disaggregated inference (1P1D) and attention DP". FP8 KV cache needs a calibrated KV scale profile
  (the checkpoint ships none); see "FP8 KV cache" below.
- MTP speculative decoding (`--speculative-config '{"method": "mtp",
  "num_speculative_tokens": 3}'`, `MTP_K=3` for the serve scripts) uses the
  checkpoint's MTP layer as the draft. It needs a vllm-neuron with the MTP
  proposer (`vllm_neuron/vllm/spec_decode/mtp.py`). With `moe_tkg_hits` it
  cuts TPOT at concurrency 1 from 26.3 to 13.8 ms and triples throughput at
  concurrency 4; see "MTP speculative decoding" under Results.
- `--decode-context-parallel-size 8 --cp-kv-cache-interleave-size 32` shards
  the replicated KV heads 8 ways (8x KV capacity) for many concurrent
  long-context requests. On 0.29 it is validated on the 4-layer model
  (`scripts/validate_features.py <ckpt> <refs> 8`: logits cos >= 0.9988 with
  segmented prefill and prefix-cache hits); its full-model performance was
  measured only on 0.24, where it cost ~22% TPOT and ~30% prefill speed at
  equal concurrency. See [src/README.md](src/README.md).
- Decode steps of up to 4 tokens per 3 local experts (EP=64: 4, EP=32: 8)
  compute only the routed experts that live on each rank
  (`src/kernels/moe_tkg_hits.py` for one token, `moe_tkg_hits_batch.py` for
  more; on by default). Larger steps keep the all-expert `moe_tkg`.
  `HY3_DECODE_HITS_MAX_T` changes the limit and `HY3_DECODE_HITS=0` turns
  both paths off. See "Decode: only the routed experts" under Results. With
  MTP, raise the limit to the verify step's size (max-num-seqs x (k + 1));
  `scripts/serve_latency.sh` does this when `MTP_K` is set.
- `"ep_degree": 32` (`EP=32` for `scripts/serve.sh` and
  `scripts/serve_latency.sh`) trades TTFT for TPOT. In config B it gives
  0.4-1.8 ms lower TPOT at concurrency 1 and 4, but 3-6% higher TPOT at
  concurrency 15 and 10-22% higher TTFT everywhere
  (`results/bench/B_multibucket_ep32/`). EP=64 stays the default.
- Below EP=32 the prefill MoE kernel (`moe_cte` `shard_on_i_hybrid`) does
  not compile on neuronx-cc 2.28. `HY3_MOE_CTE_IMPL=shard_on_i` makes EP=16 /
  8 work, at about 2.5x the TTFT and no TPOT gain over EP=32.
- On-device sampling is on by default. If every request is greedy,
  `"on_device_sampling_config": {"all_greedy": true}` swaps the general
  sampler for an argmax and saves about 0.5 ms per token; requests that set
  a temperature are then decoded greedily too.

## FP8 KV cache

`--kv-cache-dtype fp8` stores K and V as FP8 E4M3 with one static scale per
layer for K and one for V, halving KV bytes per token. The W8A8 checkpoint is
unchanged; the scales come from a separate **KV scale profile**
(`src/kv_scales.py`). `kv_profiles/Hy3-FP8_neuron.safetensors` is a calibrated
profile for tencent/Hy3-FP8 (fingerprinted against its attention weights):

```bash
HY3_KV_SCALE_PROFILE=$PWD/contrib/models/hy_v3/kv_profiles/Hy3-FP8_neuron.safetensors \
    KV_DTYPE=fp8 MODEL=/path/to/Hy3-FP8 \
    PREFILL_BUCKETS='[512, 1024, 2048, 4096, 8192]' scripts/serve.sh 16384 32 '[1, 4, 16, 32]' '[4096, 8192]'
```

To calibrate a profile on Neuron (`scripts/calibrate_kv_scales_neuron.py`):
the served W8A8 model runs the calibration mix with a BF16 KV cache, prefill
only, and the per-layer absmax is read from the cache slots the run wrote --
the values an FP8 cache quantizes. The 75 samples of `kv_calib/full.json`
(72,458 tokens) are in `bench_data/kv_calib_full_samples.json.gz`, so no
datasets download is needed; the prefill takes ~16 s after the model compile:

```bash
python scripts/calibrate_kv_scales_neuron.py calibrate /path/to/Hy3-FP8 \
    bench_data/kv_calib_full_samples.json.gz kv_scales.safetensors
```

Or with llm-compressor on a BF16 copy (HF forward on CPU: 27 min for the
4-layer reduced model on a trn2.48xlarge host, so several hours for the full
model). On the reduced model the two methods agree to <= 4.5% per layer
(layers 0-3, K and V; W8A8 vs BF16 weights):

```bash
# 1. BF16 copy for the HF forward (calibration only; ~550 GB)
python -I scripts/dequant_bf16.py /path/to/Hy3-FP8 /path/to/hy3_bf16
# 2. llm-compressor KV calibration -> profile (CPU, ~3 h for the full model)
python -I scripts/calibrate_kv_scales.py --model /path/to/hy3_bf16 --fp8-ckpt /path/to/Hy3-FP8 \
    --config scripts/kv_calib/full.json --out kv_scales.safetensors --report kv_scales.report.json
# 3. serve (same flags as BF16 KV plus the profile and the packed K layout)
HY3_KV_SCALE_PROFILE=$PWD/kv_scales.safetensors KV_DTYPE=fp8 MODEL=/path/to/Hy3-FP8 \
    PREFILL_BUCKETS='[512, 1024, 2048, 4096, 8192]' scripts/serve.sh 16384 32 '[1, 4, 16, 32]' '[4096, 8192]'
```

- **Statistics**: `QuantizationModifier(kv_cache_scheme=...)` only (no weight or
  activation scheme), `static_minmax` observer, per-tensor. K is observed after
  the per-head q/k-norm and RoPE, V after `v_proj`, before the GQA repeat --
  what the Neuron cache stores. The exported scale is the DEQUANT scale for the
  Trn2 range, `absmax / 240` (llm-compressor's own scale is `absmax / 448`).
- **Loading** (`_load_kv_cache_scales`, `kv_scales.resolve_kv_scales`): a
  profile from `HY3_KV_SCALE_PROFILE` (or `<ckpt>/kv_scales.safetensors`);
  else the checkpoint's own `self_attn.k_scale` / `v_scale`, the upstream
  convention, rescaled by 448 / 240 (they are calibrated for e4m3fn); else an
  error. `HY3_KV_SCALE_ALLOW_DEFAULT=1` gives the upstream default instead (1.0
  for every layer, with a warning), which Hy3 cannot use: its early-layer V is
  ~0.1-0.2, in the FP8 subnormal range at scale 1. For a profile the
  loader checks the format, direction, the platform FP8 range and per-layer
  fingerprints of the attention weights (`load_weights_lite` reads the scales
  too). The server log line `Hy3 FP8 KV scales: source=... layers=80 ...` names
  the source. tencent/Hy3-FP8 ships no KV scales, so it needs a profile. Decode passes
  the scales to the megakernel as compile-time values, so a new profile
  compiles new decode NEFFs; prefill reads them from a per-layer runtime tensor
  (`kv_scale_t`), so the prefill NEFFs do not depend on the profile and all 80
  prefill layers stay identical (per-layer constants there kept the compiler
  from reusing one layer for the others: full-model prefill NEFFs 10-16x larger
  and a cold compile of hours).
- **Kernel contract** (installed nkilib, nki 0.7.0): the decode megakernel takes
  QUANT scales and reads the history undequantized, so the softmax scale carries
  `1 / q_k` and the STATIC O-projection gets `input_scale * q_v` and
  `weight_scale / q_v` (scaling only the weight scale, as the kernel docstring
  suggests, clips the O-proj input at `240 / q_v` of its range).
- **Prefill read**: segmented prefill (prefix caching, chunking, DCP) reads a
  BF16 copy of the request's own blocks (exact for FP8 codes) with the BF16
  segmented kernels, folding `1 / q_k` into the softmax scale and dividing the
  output by `q_v`. Under DCP this is required: the KV-parallel kernel returns
  wrong results for chunks with prior KV when it reads the packed FP8 cache.
  Without DCP the in-kernel packed read is correct but makes the full-model
  prefill NEFFs 10-16x larger, so the BF16 copy is used there too.
- **Layouts**: prefix caching / DCP need the packed K layout
  (`"fp8_packed_kv": true`, set by `serve.sh` for FP8), which the decode kernel
  reads directly. Decode buckets the packed kernel cannot take read through an
  unswizzled copy and write the returned K back
  (`HY3_FP8_KV_FORCE_UNPACKED_DECODE=1` forces this path).
- **DCP prefill buckets**: Q=8192 does not compile under DCP (`NCC_IGCA037`,
  see `serve_256k.sh`). The Q=4096 bucket works in the `serve_256k.sh`
  configuration with both KV dtypes (FP8 KV: shared-prefix benchmark below,
  passkey 12/12 at 16k-85k), but in one 16k configuration (max_model_len 16384,
  32 seqs, decode context buckets `[4096, 8192]`, `--num-gpu-blocks-override
  8139`) it gave wrong output for ~8k-token prompts with BF16 KV (chunks with
  prior KV) and hung every rank in a collective with FP8 KV
  (`NRT_EXEC_HW_ERR_DMA_ABORT`; the devices then need a neuron driver reload).
  The trigger was not isolated; prefill buckets <= 2048 were correct in every
  configuration tried. In the `serve_256k.sh` configuration (BF16 KV) the 16
  prompts of `bench_data/prompts/in8192_c4` (8192 tokens, two 4096 chunks, the
  second with prior KV) give 64-token greedy continuations identical to config
  B without DCP, and the 8k passkey test passes 3/3.
- **Startup**: the full-model FP8 cold compile of config B with five prefill
  buckets took ~35 min on a trn2.48xlarge; keep `VLLM_NEURON_BARRIER_TIMEOUT`
  (default 3600 s) above the cold compile time of larger configurations.

- **Acceptance** (`scripts/kv_acceptance.sh` + `scripts/kv_report.py`): the gate
  reads raw artifacts only and fails on missing evidence. It checks the run
  manifests (`kv_manifest.py`: identical args except the KV dtype, server env,
  commit, profile sha256), the exact persistent KV pool (`Hy3 KV pool` log line)
  and the per-category peak device memory over the run (`kv_mem_sampler.py`),
  MATH-500 paired per problem and re-graded (McNemar, bootstrap CI), complete
  identical load-matrix cells, and stability runs of at least 30 minutes.
- **Full-model results** (trn2.48xlarge, re-run passing the gate; config B +
  prefill buckets 512..8192 for DCP=1, prefill <= 2048 for DCP=8; the same
  8139-block pool for both KV dtypes; profile from `kv_calib/full.json`):

  | | BF16 KV | FP8 KV |
  |---|---|---|
  | persistent KV pool per rank (64 ranks) | 10,667,950,080 B (635.9 GiB) | 5,333,975,040 B (317.9 GiB), x 0.5000 |
  | peak device memory over the run, all ranks | 1219.8 GiB | 910.3 GiB (-309.6 GiB = 48.7% of the BF16 pool; FP8-only `model_code` +8.3 GiB, scratchpad unchanged) |
  | MATH-500, 500 problems, greedy, no_think, paired | 97.4% | 97.8% (6 vs 8 discordant, 95% CI of the difference [-1.0, +1.8] pp) |
  | MATH-500, DCP=8 | 96.8% | 97.0% |
  | load matrix 1024-16000 x c 1/4/15, worst P50 change | | TTFT +4.6%, TPOT +4.2% (1024 / c=1); TPOT -6..-8% at c=15, >= 8k |
  | stability, 1100 requests, c=8, 6k shared prefix | 2198 s, 0 failed, TPOT P50 55.9 ms | 1974 s, 0 failed, TPOT P50 49.4 ms |

  Greedy MATH-500 moves by 0.4-0.8 pp between server restarts on this stack;
  across the original and the re-run (6 BF16 / FP8 pairings) FP8 scored the
  same or higher every time.
- **85k shared prefix, DCP=1** (customer D test: its request counts and output
  lengths, max_model_len 90112, the same pool): FP8 KV lowers TPOT by 21-22% at
  concurrency 2-8 (BF16 86-90 ms) and by 30% at 9-13 (141-144 ms -> 98-101 ms);
  TTFT +1-2%. Without DCP every rank reads the whole 85k history each decode
  step, so halving the KV bytes shows in TPOT.
- **85k shared prefix, DCP=8** (`serve_256k.sh` config, the same 8139-block
  pool, `scripts/bench_prefix_fixed.sh "1 4 8"` with a new `cache_salt` per run,
  92.2% hit rate in every cell):

  | conc | TTFT med BF16 -> FP8 | TPOT med BF16 -> FP8 |
  |---|---|---|
  | 1 | 2.58 -> 2.59 s | 31.9 -> 32.9 ms |
  | 4 | 6.18 -> 6.18 s | 59.1 -> 57.4 ms |
  | 8 | 10.81 -> 10.85 s | 85.9 -> 83.8 ms |

  With DCP=8 each rank reads 1/8 of the history, so FP8 KV changes latency by
  at most 3% either way; its gain there is capacity (twice the tokens in the
  same memory). Passkey retrieval (`needle.py`): 12/12 with FP8 KV, as with
  BF16. The FP8 cold compile of this configuration took about 3 h (BF16 about
  1.5 h); keep `VLLM_NEURON_BARRIER_TIMEOUT` above it.
- **Reproduction on a second trn2.48xlarge** (`results/bench/kv_fp8_configB/`;
  config B + prefill 512..8192, DCP=1, the same 8139-block pool, profile
  `kv_profiles/Hy3-FP8_neuron.safetensors` calibrated on Neuron as above; load
  matrix with the fixed real-text prompts, `bench_fixed.sh "1024 4096 8192
  16000" "1 4 15"`):

  | | BF16 KV | FP8 KV |
  |---|---|---|
  | persistent KV pool per rank | 10,667,950,080 B | 5,333,975,040 B (x 0.5000) |
  | peak device memory over the run, all ranks | 1201.7 GiB | 893.7 GiB (-308.0 GiB = 48.4% of the BF16 pool; `model_code` +9.9 GiB) |
  | MATH-500, 500 problems, greedy, no_think, paired | 97.0% | 96.8% (6 vs 5 discordant, exact McNemar p = 1.0) |
  | 16 x 8192-token prompts, 64-token greedy continuation | | identical to BF16 KV (16/16) |
  | passkey at 8k (3 depths) | 3/3 | 3/3 |
  | load matrix, worst P50 change | | TTFT +5% (1024 / c=1, 0.20 -> 0.21 s), TPOT +2.7% (4096 / c=1) |
  | load matrix, concurrency 15 | | TPOT -2.4..-8.5% (8192: 75.0 -> 68.6 ms), ITL -12.5% at 8k / 16k |

  This run is on commit 55772b8 of the change; the later commits (local copies
  of the private helpers, the scale-source order) passed the unit, module and
  integration tests (4/4, two with FP8 KV, the same profile).

## Disaggregated inference (1P1D) and attention DP

### Sparse attention across prefill and decode

The sparse-attention K summaries (`k_sum`) are written by prefill and read by decode, and they
live next to the KV cache, not in it. `model.py` exposes them as
`CONNECTOR_EXTRA_KV_TENSORS = {layer name: k_sum}`. The KV connectors of vllm-neuron
(`NeuronNixlConnector`, `NeuronMooncakeConnector`, vllm-project/vllm-neuron#90)
register each one as one more region of its layer, so it moves block by block with that layer's
K and V. Without it a disaggregated decode has empty summaries for the prompt, and sparse decode
keeps only the sink and recent groups.

Set `HY3_SPARSE_DECODE` / `HY3_SPARSE_PREFILL` on both nodes, because the prefill node writes the
summaries. A summary row belongs to its group's first block: a decode-side prefix hit that ends
inside a 128-token group leaves that group's row as it was on the decode node.

**Measured.** Two trn2.48xlarge, prefill TP=64 / decode TP=64, FP8 KV, MTP k=3,
max_model_len 87040, prefill buckets up to 8192. Workload: 85k-token inputs with a 78k shared
prefix (92% hit), 1024 output tokens, concurrency 8.

| | TTFT med / P99 (s) | TPOT med / P99 (ms) | output tok/s |
|---|---|---|---|
| dense | 2.80 / 13.1 | 42.2 / 49.2 | 171 |
| sparse decode 0.125 | 3.06 / 13.0 | 26.7 / 30.7 | 255 |
| sparse decode 0.125 + sparse prefill 0.25 (NIXL) | 2.46 / 10.2 | 26.8 / 31.8 | 261 |
| the same with Mooncake | 1.96 / 9.7 | 26.1 / 31.8 | 273 |

**Accuracy through PD** (sparse decode 0.125 with `HY3_SPARSE_MIN_BATCH=1`, so single requests
take the sparse path):
- needle: 12/12, at 16k-85k and depths 0.1 / 0.5 / 0.9;
- `sparse_decode_eval.py`: 16/19 continuations identical to the dense PD run. The 3 that differ
  split at near-ties on repetitive prompts.

### Attention DP on the decode node (experimental)

With `--data-parallel-size D --enable-expert-parallel` (e.g. TP=8 x DP=8 on one trn2.48xlarge)
each replica runs attention for its own requests over its own TP group, so each KV head is read
once instead of TP / num_kv_heads times. The routed experts are spread over all TP x DP ranks.

**How the MoE runs on decode:**
1. It gathers the replicas' tokens over the world group. The strided cross-DP replica group has no
   collective algorithm on trn2 ("no_hier no_mesh").
2. It runs the router, the sharded shared expert and the routed experts on all of them.
3. It all-reduces over the world and keeps its replica's rows (`shard_select`).

The prefill path does the same after its sequence-parallel gather.

**Status.** Correct without MTP: needle and 80/80 requests on the 85k workload. At concurrency 8
with sparse decode it is slower than TP=64:

| decode, no MTP, sparse decode + prefill | TPOT med (ms) |
|---|---|
| TP=64 | 46.2 |
| TP=8 x DP=8 | 51.7 |

Sparse decode already reads only 1/8 of the history, so there is little replicated KV left to
save, while DP adds world-wide MoE collectives and 8x larger attention and lm_head shards per
rank. Its gain is capacity: every replica has its own KV pool.

**With MTP:**
- The MTP draft's MoE also crosses the replicas, so idle replicas must run a dummy draft. This
  needs the `_dispatch_dummy_draft` change in `neuron_model_runner.py`, which is part of this
  change.
- Serving then still fails: the draft hits an unwarmed shape (`fail_on_recompile` on
  `layers.80.self_attn`) when the DP padding changes the batch.

So DP + MTP remains future work.

## Fused decode MoE (`HY3_DECODE_MOE_FUSED=1`, experimental, off)

`src/kernels/moe_tkg_fused.py` runs the whole decode MoE block of a layer in one NKI kernel,
on every rank:
1. the post-attention RMSNorm;
2. the router (BF16 logits, sigmoid, expert bias, top-8, renormalization, scaling factor);
3. this rank's routed experts;
4. the shared expert's decode shard.

The output is the rank's partial, combined by the same all-reduce as before
(`HYV3MoE._combine`).

**Why it is faster.** At T = 32 (8 requests with MTP k=3) the production path costs 0.34 ms per
layer: `moe_tkg` at about 200 GB/s of FP8 weights, plus torch ops for the norm and the router.
The kernel changes three things:
- **W8A8.** It quantizes the activations with the checkpoint's calibrated static input scales,
  doubled at load time like every other input scale here. The matmuls then contract 256 rows
  per pass (`matmul_perf_mode.double_row`); BF16 activations allow only 128.
- **Long DMA rows.** It reads the weights in `HYV3MoE`'s own layouts, so no extra HBM is used.
  The two cores of the logical NeuronCore each take one half of H for gate/up, so each DMA row
  is a full 2 x I bytes. They exchange their partial sums (`sendrecv`), and each core keeps half
  of I for the activation and the down projection.
- **No torch ops** between the attention all-reduce and the MoE all-reduce.

The top-8 is a threshold at the 8th largest biased score (`nisa.max8`). `moe_tkg_hits*` stay in
use where they are chosen today (up to `HY3_DECODE_HITS_MAX_T` tokens); `HY3_DECODE_HITS=0`
sends every decode size to the fused kernel. The kernel is not used with attention DP.

One NeuronCore, real layer weights (rank-0 shard of EP=64), T = 32. "Latency" means each layer's
input is the previous layer's output (`test_moe_tkg_fused.py`, `HY3_FUSED_BENCH=1`):

| decode MoE block, per layer | layer-to-layer latency |
|---|---|
| production: torch RMSNorm + `hy3_route` + `moe_tkg` + shared `NF.mlp` | 0.339 ms |
| `moe_tkg_fused` | 0.197 ms |

The time is flat in T. The routed part alone takes 0.127 ms at T = 8 and 16 and 0.142 ms at
T = 64, while `moe_tkg_hits_batch` takes 0.25 ms once all 3 local experts are hit.

On random inputs the output is 3% (relative L2) from an FP32 fake-quant reference. The same W8A8
fake-quant is 4% from a BF16 reference.

`HY3_MOE_STATIC_DGE=1` gives the weight streams static DMA descriptors. That brings the one-core
latency to 0.152 ms, but the descriptors live in HBM: about 1.1 GB more "DMA rings IO" per core
with decode buckets [1, 4, 8]. At `KV_CAP` 0.61 the decode server then fails its first full step
with `Failed to allocate ... dma rings io`. It stays off.

### PD test method and results

**Setup.**
- Two trn2.48xlarge (EFA, SDK 2.33, vLLM 0.29), one per role.
- vllm-project/vllm-neuron#90 (`NeuronMooncakeConnector`) applied to the checkout.
- `mooncake-transfer-engine-non-cuda` 0.3.14 installed.
- `scripts/pd/` (below) drives everything. The sparse K summaries move as extra KV regions
  (see "Sparse attention across prefill and decode").

```bash
COMMON="CONNECTOR=NeuronMooncakeConnector KV_DTYPE=fp8 MTP_K=3 KV_CAP=0.61 \
  HY3_KV_SCALE_PROFILE=$PWD/contrib/models/hy_v3/kv_profiles/Hy3-FP8_mtp_neuron.safetensors \
  HY3_SPARSE_DECODE=0.125 HY3_SPARSE_PREFILL=0.25 MAX_LEN=87040 MODEL=/path/to/Hy3-FP8"
# decode node (start first; stays up). Give each HY3_DECODE_MOE_* setting its own
# VLLM_CACHE_ROOT: the precompile cache does not key on these env flags.
bash contrib/models/hy_v3/scripts/pd/run_pd.sh fused decode-node $COMMON \
  HY3_DECODE_MOE_FUSED=1 HY3_DECODE_HITS=0 VLLM_CACHE_ROOT=/path/cache/fused
# prefill node: prefill server + Mooncake proxy, waits for the decode, then runs the bench
bash contrib/models/hy_v3/scripts/pd/run_pd.sh fused prefill-node $COMMON \
  D_HOST=<decode private IP> BENCH=pd_bench OUT_LEN=1024 CONCS=8
# accuracy through the same pair (needle + sparse_decode_eval)
bash contrib/models/hy_v3/scripts/pd/run_pd.sh fused_acc prefill-node $COMMON \
  D_HOST=<decode private IP> BENCH=pd_acc
```

**Workload.** `pd_bench.sh` runs the 85k / 92% shared-prefix case (`bench_data/prefix`) through
the proxy:
- first a smoke request and a needle request;
- then one warm-up request and 80 requests at concurrency 8, with 1024 output tokens and a new
  `cache_salt` per run;
- it reads the prefill and decode prefix-hit rates and the MTP acceptance from the servers'
  `/metrics`.

**Results**, decode TP=64 / EP=64, prefill TP=64, Mooncake over EFA:

| decode MoE | TTFT med / P99 (s) | TPOT med / P99 (ms) | ITL med (ms) | output tok/s | MTP tok/step |
|---|---|---|---|---|---|
| production (torch + `moe_tkg`) | 1.98 / 9.8 | 26.5 / 30.9 | 57.4 | 267 | 2.26 |
| `HY3_DECODE_MOE_FUSED=1 HY3_DECODE_HITS=0` | 1.95 / 9.9 | **19.9 / 22.6** | **42.7** | **353** | 2.26 |
| the same, this branch (repeat run) | 1.99 / 9.9 | 20.0 / 22.7 | 42.8 | 343 | 2.23 |

Every run completed 80/80 requests, with prefill and decode prefix hits of 92.1%. The JSON
outputs are in `results/pd_fused/`.

A device profile of the production path (`scripts/profile_85k.sh`, 5 requests at 85k on the
batch-8 NEFF) shows where the step went:

| production decode step: 57.2 ms | ms |
|---|---|
| target NEFF | 52.7 |
| MTP draft | 4.6 |
| routed experts | 23.4 |
| shared expert | 4.2 |
| MoE norm + router torch ops | 4.3 |
| attention | 20.5 |

**Accuracy through PD.**
- Decode with `HY3_DECODE_HITS=0`, so single requests take the fused kernel too.
- `HY3_SPARSE_MIN_BATCH` stays at 4: single requests run dense attention.

| | needle (12) | `sparse_decode_eval.py`, 19 greedy continuations |
|---|---|---|
| fused MoE | 12/12 | 16/19 identical to the dense PD run (the 3 near-tie items where sparse decode also differs, same split points) |
| fused MoE vs the torch router in front of the same W8A8 routed experts (development build) | | 19/19 identical |

The selection in the kernel gives the same tokens as `hy3_route`.

## Prefill: the MoE blockwise mapping on the NKI path (`HY3_MOE_MAP_PAD`, on)

`NF.build_blockwise_mapping` maps each routed token to a 256-token block of its expert for the prefill
MoE kernel. Its NKI path (`find_nonzero_indices`) needs an even local expert count, because it splits
the columns over the two cores of the logical NeuronCore. At EP=64, Hy3 has 3 local experts per rank,
so the mapping fell back to torch ops (`_cumsum_matmul`).

On one core that fallback costs only about 0.1 ms per layer. In the server it runs while the MoE
all-gathers its 4096 tokens (32 MB), and that all-gather took 1.8 ms instead of the 0.22 ms of the
same-size attention all-gather. This happened on every rank, so it is not skew: ranks 0 and 36 are
identical. Over 79 layers it cost about 125 ms of a 637 ms prefill step.

`routed_experts_cte` therefore pads one zero column to the affinities for the mapping. The padded
expert is never selected and gets no blocks:
- `masked` stays in the kernel's [T, L] layout;
- `block_to_expert` is clamped to L - 1 for the empty trailing blocks, which have `conditions` 0.

The kernel's output is bit-identical (relative error 0 on one core against the fallback).
`HY3_MOE_MAP_PAD=0` keeps the fallback.

**P-only at C = 8** (`scripts/pd/pd_bench_split.sh`, 85k / 92% shared prefix, output 1, 80
requests):

| prefill configuration | TTFT P50 / P99 | prefill req/s |
|---|---|---|
| before (prefill buckets up to 4096) | 10.25 / 11.30 s (1P1D) | 0.78 |
| prefill buckets up to 8192, no map-pad | 10.03 s (single node) | 0.80 |
| map-pad | 7.54 / 7.67 s (single node) | 1.07 |
| **map-pad + buckets up to 8192, prefill KV_CAP 0.45 (the `scripts/pd` prefill defaults now)** | **5.78 / 6.05 s (1P1D)** | **1.38** |

The 8192-token steps need the HBM that a smaller KV pool leaves. With KV_CAP 0.65 they fail to
allocate. The prefill role of `pd_serve.sh` therefore takes `P_KV_CAP` (default 0.45, 514k tokens)
and `P_PREFILL_BUCKETS` (default 512..8192), and ignores `KV_CAP`, which applies to decode. The PD
commands above are unchanged.

### Trn2 vs H100, 1P1D, the same benchmark

**H100 setup.** Two p5.48xlarge (8x H100, 32 EFA each), vLLM 0.29.0 (`scripts/pd/gpu/Dockerfile`:
`vllm/vllm-openai:v0.29.0` + aws-efa-installer 1.51.0 + GDRCopy 2.6 + Mooncake 0.3.14),
`scripts/pd/gpu/gpu_serve.sh`.
- The customer's reference flags: TP=8, `--tool-call-parser hy_v3 --enable-auto-tool-choice
  --reasoning-parser hy_v3`, `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm`.
- max_model_len 87040, vLLM's `MooncakeConnector` over EFA, BF16 KV, no MTP.

**Trn2 setup.** The `scripts/pd` defaults: FP8 KV, MTP k=3, sparse decode 0.125 + sparse prefill
0.25, decode `HY3_DECODE_MOE_FUSED=1 HY3_DECODE_HITS=0`.

**Shared between the two.** The same proxy file. The same benchmark: `scripts/pd/pd_bench_final.sh`
(P-only at C = 1 / 4 / 8 / 12, the end-to-end 1024-output case at C = 8, then the D-only replay at C = 8). The same prompt files
(md5-checked on every host; `in85000_p78336_c12` was generated with `gen_prefix_prompts.py`, which
regenerates c8 byte-identically). The same `vllm bench serve` (vLLM 0.29.0).

| P-only TTFT P50 / P99 (s) | C = 1 | C = 4 | C = 8 | C = 12 | prefill req/s |
|---|---|---|---|---|---|
| Trn2 | 1.13 / 1.20 | 2.95 / 3.38 | **5.78 / 6.05** | 8.71 / 9.52 | 1.37 |
| H100 | 0.87 / 0.89 | 2.15 / 2.66 | 3.95 / 5.25 | 6.01 / 7.60 | 2.00 |

| end-to-end, 1024 out, C = 8 | TTFT median / P99 | TPOT median / P99 | output tok/s |
|---|---|---|---|
| Trn2 | 1.84 / 6.46 s | 19.8 / 22.2 ms | 359 |
| H100 | 1.17 / 4.41 s | 16.7 / 16.8 ms | 437 |

All runs completed 80/80 (or 10×C) requests, with prefill and decode prefix hits of 92.0-92.2%.
Results are in `results/pd_prefill/`.

**D-only, the customer GPU table** (`scripts/pd/pd_bench_dcust.sh`, the third part of
`pd_bench_final.sh`). All 80 prompts of the c8 file are sent once with output 1, so both nodes cache
them. They are then replayed with the same `cache_salt`, so almost nothing is prefilled or
transferred. 1200 output tokens, `--ignore-eos`, C = 8, 2 warm-ups, temperature 0.9, seed 0, mean
and P99. TPOT is the number to read here. TTFT is only the fixed path overhead, not prefill.

| D-only, 1200 out, C = 8 | TTFT mean / P99 | TPOT mean / P99 | prefix hits P / D |
|---|---|---|---|
| customer GPU reference | 388 ms / - | 15.25 ms / - | - |
| H100 1P1D (the setup above) | 477 ms / 1.64 s | 17.09 / 17.17 ms | 100% / 99.5% |
| Trn2 1P1D | not measured yet with these defaults | | |

On Trn2 the 80 prompts (about 611k tokens) do not all fit in the prefill node KV (514k at
`P_KV_CAP` 0.45), so part of the replay is prefilled again and TTFT reads higher than on H100. TPOT
is not affected. The H100 run is in `results/pd_prefill/h100/dcust/`.

## Compatibility Matrix

| Instance / Neuron SDK version | 2.33.0      | 2.32 and earlier |
| ----------------------------- | ----------- | ---------------- |
| Trn3                          | Not tested  | Not tested       |
| Trn2 (trn2.48xlarge)          | Working     | Not tested with this folder |

Tested with vllm-neuron 0.29.0.1.2.0 (vLLM 0.29.0). Neuron SDK 2.33.0 component
versions: neuronx-cc 2.28.7631, nki 0.7.0, Neuron runtime 2.35.30, collectives
2.35.43, tools 2.33.38, driver 2.31.7; also libtorch-neuronx-lite 2.13.0.1.0,
torch 2.13.0, transformers 5.17.0, Python 3.12.

## Example Checkpoints

- <https://huggingface.co/tencent/Hy3-FP8>

## Results

All numbers: trn2.48xlarge, TP=64, full EP=64 unless noted, FP8 weights, BF16 KV cache,
`VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.45`, on-device sampling, block size 32.

### Accuracy

| Test | Result |
| --- | --- |
| MATH-500 no_think (full model, temp 0.9, max 4096 tokens, config B) | **97.0%** (485/500), 3 truncated |
| 4-layer real-weight model vs HF FP32, flash prefill (`scripts/compare_hf_logits.py`) | logits cosine >= 0.9993; greedy splits only at near-ties (top-2 gap <= 0.14 nats) |
| same, segmented prefill 5000 / 3100-token prompts, cold and prefix-cache hit (4992 / 3072 tokens cached) (`scripts/validate_features.py`) | cosine >= 0.9989, identical cold vs cache hit |
| batch 128 vs batch 4 decode | greedy identical, cosine >= 0.9993 |
| DCP=8, same checks (`validate_features.py ... 8`) | cosine >= 0.9988; greedy splits only at near-ties (gap 0.011-0.18 nats) |
| Integration test (`test/integration/test_model.py`, both cases) | passed: top-K logit error <= 0.06 vs the BF16 golden; batch-1 case (`moe_tkg_hits`) <= 0.045, batch-4 case (`moe_tkg_hits_batch`) <= 0.057 |
| Module tests (`test/module/`) | routed experts rel. error ~3e-3 (`moe_tkg_hits`, 1-3 local hits: 2.5-3.3e-3, 0 hits: exactly 0; `moe_tkg_hits_batch`, 2 / 4 tokens: 2.9e-3); MLP: BF16-activation (TKG) paths ~2e-3, FP8-activation (CTE) paths within 1-2% of an ideal FP8 quantizer |

### Performance

`vllm bench serve`, random dataset, 256 output tokens (`ignore_eos`),
4 x concurrency requests per cell (min 8), a distinct seed per cell
(`scripts/bench_matrix.sh`). TPOT includes the decode stalls caused by other
requests' prefills (prefill and decode never share a step on Neuron); ITL
median is close to the pure decode step time. Raw JSON is in `results/bench/`.

**Recommended: prefix caching + several prefill buckets** (`results/bench/B_multibucket/`):

```bash
PREFILL_BUCKETS='[512, 1024, 2048, 4096, 8192]' MODEL=<ckpt> \
    scripts/serve.sh 16384 32 '[1, 4, 16, 32]' '[4096, 8192]'
```

| input | conc | TTFT med (s) | TTFT P99 (s) | TPOT med (ms) | ITL med (ms) | out tok/s | peak out tok/s | total tok/s |
|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | 0.23 | 0.25 | 22.2 | 21.7 | 44 | 47 | 218 |
| 1024 | 4 | 0.54 | 0.85 | 34.1 | 32.7 | 111 | 124 | 554 |
| 1024 | 15 | 1.59 | 2.94 | 44.9 | 39.4 | 295 | 390 | 1473 |
| 4096 | 1 | 0.57 | 0.58 | 22.2 | 21.9 | 41 | 46 | 697 |
| 4096 | 4 | 1.37 | 2.17 | 37.8 | 34.5 | 93 | 120 | 1583 |
| 4096 | 15 | 4.26 | 7.96 | 56.9 | 42.2 | 205 | 360 | 3479 |
| 8192 | 1 | 1.01 | 1.02 | 23.7 | 23.5 | 36 | 43 | 1198 |
| 8192 | 4 | 2.45 | 3.90 | 42.8 | 36.9 | 77 | 112 | 2528 |
| 8192 | 15 | 7.71 | 14.42 | 78.2 | 51.8 | 139 | 300 | 4582 |
| 16000 | 1 | 2.05 | 2.06 | 23.8 | 23.5 | 32 | 43 | 2007 |
| 16000 | 4 | 4.99 | 7.94 | 48.6 | 36.8 | 59 | 112 | 3746 |
| 16000 | 15 | 15.79 | 29.51 | 105.7 | 51.7 | 90 | 300 | 5707 |

**High concurrency, short context** (max_model_len 2048, 128 sequences,
prefill buckets 512 / 1024 / 2048; `results/bench/C_multibucket/`):

```bash
PREFILL_BUCKETS='[512, 1024, 2048]' MODEL=<ckpt> \
    scripts/serve.sh 2048 128 '[1, 4, 16, 32, 64, 128]' '[1024]'
```

| input | conc | TTFT med (s) | TTFT P99 (s) | TPOT med (ms) | ITL med (ms) | out tok/s | peak out tok/s | total tok/s |
|---|---|---|---|---|---|---|---|---|
| 512 | 32 | 2.37 | 4.44 | 45.0 | 36.4 | 593 | 896 | 1779 |
| 512 | 63 | 4.48 | 8.76 | 61.3 | 44.2 | 802 | 1449 | 2405 |
| 512 | 127 | 8.89 | 17.49 | 95.7 | 61.1 | 976 | 2159 | 2927 |
| 1024 | 32 | 3.23 | 6.20 | 51.1 | 39.3 | 504 | 832 | 2518 |
| 1024 | 63 | 6.19 | 12.13 | 71.1 | 47.5 | 663 | 1380 | 3314 |
| 1024 | 127 | 12.31 | 24.24 | 116.5 | 68.6 | 773 | 1905 | 3866 |

**Prefix-cache hit rate** (same server as the recommended config;
`scripts/bench_prefix_cache.sh "8192 16000" "0 25 50 75 90" "1 8"`,
`results/bench/prefix_cache/`). Each cell sends 16 requests that share one
random prefix of the given share of the input; the first request fills the
cache. The hit rate is measured from the server's `vllm:prefix_cache_hits` /
`vllm:prefix_cache_queries` counters: prefix share x 15/16, rounded down to
32-token blocks.

| input | conc | shared prefix | measured hit rate | TTFT med (s) | TTFT P99 (s) | TPOT med (ms) | out tok/s | total tok/s |
|---|---|---|---|---|---|---|---|---|
| 8192 | 1 | 0% | 0.0% | 1.01 | 1.05 | 27.7 | 32 | 1044 |
| 8192 | 1 | 25% | 23.4% | 1.02 | 1.03 | 27.7 | 32 | 1044 |
| 8192 | 1 | 50% | 46.9% | 0.61 | 0.96 | 27.7 | 33 | 1096 |
| 8192 | 1 | 75% | 70.3% | 0.39 | 0.96 | 27.7 | 34 | 1124 |
| 8192 | 1 | 90% | 84.2% | 0.29 | 0.92 | 27.7 | 35 | 1140 |
| 8192 | 8 | 0% | 0.0% | 4.36 | 7.72 | 65.1 | 98 | 3225 |
| 8192 | 8 | 25% | 23.4% | 4.39 | 7.76 | 65.1 | 98 | 3220 |
| 8192 | 8 | 50% | 46.9% | 2.76 | 4.86 | 59.5 | 114 | 3767 |
| 8192 | 8 | 75% | 70.3% | 1.88 | 3.32 | 56.5 | 126 | 4152 |
| 8192 | 8 | 90% | 84.2% | 1.49 | 2.63 | 55.0 | 132 | 4357 |
| 16000 | 1 | 0% | 0.0% | 2.06 | 2.08 | 27.7 | 28 | 1782 |
| 16000 | 1 | 25% | 23.4% | 1.66 | 2.00 | 27.7 | 29 | 1857 |
| 16000 | 1 | 50% | 46.9% | 1.12 | 1.92 | 27.7 | 31 | 1972 |
| 16000 | 1 | 75% | 70.3% | 0.70 | 1.85 | 27.7 | 33 | 2071 |
| 16000 | 1 | 90% | 84.4% | 0.48 | 1.83 | 27.7 | 34 | 2127 |
| 16000 | 8 | 0% | 0.0% | 8.94 | 15.79 | 78.7 | 71 | 4484 |
| 16000 | 8 | 25% | 23.4% | 7.38 | 13.04 | 73.5 | 78 | 4978 |
| 16000 | 8 | 50% | 46.9% | 5.21 | 9.11 | 66.0 | 93 | 5901 |
| 16000 | 8 | 75% | 70.3% | 3.51 | 6.22 | 60.2 | 109 | 6894 |
| 16000 | 8 | 90% | 84.4% | 2.62 | 4.67 | 57.1 | 119 | 7577 |

- At an 84% hit rate a 16k prompt's median TTFT drops from 2.06 s to 0.48 s
  (4.3x). P99 stays near the cold value because each cell's first request
  misses.
- At concurrency 8, shorter prefills also stall decode less: 16k TPOT drops from
  78.7 to 57.1 ms and total throughput rises from 4484 to 7577 tok/s.
- The uncached remainder is still padded to a prefill bucket. With 8k input
  and a 25% prefix, the remaining 6144 tokens pad to the 8192 bucket, so there
  is no gain. Denser buckets (subject to compile time) close such gaps.

**Single request, lowest latency** (`scripts/serve_latency.sh`: no prefix
caching, prefill buckets 512..16384 so a 16k prompt is one prefill step;
`results/bench/opt_latency/`):

| input | TTFT med | TPOT med |
| --- | --- | --- |
| 1024 | 0.22 s | 22.1 ms |
| 4096 | 0.54 s | 22.4 ms |
| 8192 | 0.97 s | 23.6 ms |
| 16000 | 1.83 s | 23.5 ms |

The multi-bucket prefix-caching config is within 3-12% of this TTFT and
about the same TPOT (within 2%), so it is the better default.

### Decode: only the routed experts (`moe_tkg_hits`, `moe_tkg_hits_batch`)

In a batch-1 decode step at EP=64 the routed experts took 45% of the device
time (`moe_tkg` all-expert, 12.6 of 27.8 ms). Every rank computes its 3 local
experts although a token's 8 experts hit at most one or two of them on any
rank. The step is bound by the weights read, not by FLOPs: a one-token
matmul streams its weight tiles through the PE array at about 180 GB/s per
physical core, so each expert computed costs its 18.9 MB of FP8 weights.
`moe_tkg`'s selective mode does not help: it unrolls all 8 top-k slots, and
the non-local ones run on a dummy expert with affinity 0.

`src/kernels/moe_tkg_hits.py` (forked from nkilib's `selective_expert_impl`,
Apache 2.0) takes the top-k slots with this rank's experts packed first plus
their count, and loops over that count at runtime (`nl.fori_loop` on a
register). A rank with no routed expert does no expert work; the MoE
all-reduce then waits for the busiest rank, which at EP=64 usually has one
expert and sometimes two. The runtime loop adds about 30 us of control-flow
barriers per layer.

`src/kernels/moe_tkg_hits_batch.py` (from nkilib's `all_expert_impl`) does the
same for decode steps of 2 to 4 tokens per 3 local experts (EP=64: up to 4,
EP=32: up to 8). The model packs this rank's experts that any token in the
step routed to first, reorders their affinity columns the same way, and the
kernel reads each hit expert once for the whole batch, scaling its output
per token (0 for tokens that did not route to it). Past that batch size the
busiest rank hits all its experts in most layers, where the runtime loop is
slower than `moe_tkg`'s unrolled all-expert path, so larger steps keep
`moe_tkg`. `HY3_DECODE_HITS_MAX_T` overrides the limit; `HY3_DECODE_HITS=0`
turns both paths off.

Latency config (`scripts/serve_latency.sh`), c=1, `moe_tkg` -> `moe_tkg_hits`
(`results/bench/opt_latency/`, `results/bench/opt_latency_ep32/`):

| input | TTFT, EP=64 | TPOT, EP=64 | TTFT, EP=32 | TPOT, EP=32 |
| --- | --- | --- | --- | --- |
| 1024 | 0.21 -> 0.22 s | 26.3 -> 22.1 ms | 0.26 s | 21.1 ms |
| 4096 | 0.54 -> 0.54 s | 26.3 -> 22.4 ms | 0.65 s | 20.9 ms |
| 8192 | 0.97 -> 0.97 s | 27.7 -> 23.6 ms | 1.08 s | 22.3 ms |
| 16000 | 1.83 -> 1.83 s | 27.7 -> 23.5 ms | 1.88 s | 22.2 ms |

Recommended config B (`scripts/serve.sh`, see above), `HY3_DECODE_HITS=0` ->
default (`results/bench/B_multibucket_moe_tkg/`, `results/bench/B_multibucket/`).
The two runs are on two trn2.48xlarge instances; the concurrency-15 cells,
which the change does not touch, agree within 0.2%. Concurrency 1 gets 14-15%
lower TPOT. At concurrency 4 the
decode steps have 4 tokens, and the 4-token step itself (ITL median) drops
from 42-50 ms to 33-37 ms: TPOT -11% to -35%, output throughput up to +50%.
Concurrency 15 decodes in the 16-token bucket and is unchanged:

| input | conc | TTFT med (s) | TTFT P99 (s) | TPOT med (ms) | ITL med (ms) | out tok/s | peak out tok/s | total tok/s |
|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | 0.22 -> 0.23 | 0.23 -> 0.25 | 26.3 -> 22.2 | 26.3 -> 21.7 | 37 -> 44 | 39 -> 47 | 185 -> 218 |
| 1024 | 4 | 0.54 -> 0.54 | 0.84 -> 0.85 | 52.1 -> 34.1 | 50.2 -> 32.7 | 74 -> 111 | 84 -> 124 | 370 -> 554 |
| 1024 | 15 | 1.59 -> 1.59 | 2.95 -> 2.94 | 44.9 -> 44.9 | 39.4 -> 39.4 | 295 -> 295 | 390 -> 390 | 1473 -> 1473 |
| 4096 | 1 | 0.56 -> 0.57 | 0.57 -> 0.58 | 26.3 -> 22.2 | 26.4 -> 21.9 | 35 -> 41 | 38 -> 46 | 597 -> 697 |
| 4096 | 4 | 1.38 -> 1.37 | 2.19 -> 2.17 | 53.2 -> 37.8 | 49.2 -> 34.5 | 69 -> 93 | 88 -> 120 | 1166 -> 1583 |
| 4096 | 15 | 4.26 -> 4.26 | 7.96 -> 7.96 | 56.9 -> 56.9 | 42.2 -> 42.2 | 205 -> 205 | 360 -> 360 | 3478 -> 3479 |
| 8192 | 1 | 1.01 -> 1.01 | 1.03 -> 1.02 | 27.7 -> 23.7 | 27.8 -> 23.5 | 32 -> 36 | 37 -> 43 | 1046 -> 1198 |
| 8192 | 4 | 2.45 -> 2.45 | 3.90 -> 3.90 | 55.4 -> 42.8 | 48.4 -> 36.9 | 62 -> 77 | 96 -> 112 | 2036 -> 2528 |
| 8192 | 15 | 7.71 -> 7.71 | 14.44 -> 14.42 | 78.3 -> 78.2 | 51.7 -> 51.8 | 139 -> 139 | 300 -> 300 | 4579 -> 4582 |
| 16000 | 1 | 2.06 -> 2.05 | 2.07 -> 2.06 | 27.7 -> 23.8 | 27.8 -> 23.5 | 28 -> 32 | 36 -> 43 | 1785 -> 2007 |
| 16000 | 4 | 5.00 -> 4.99 | 7.96 -> 7.94 | 54.5 -> 48.6 | 42.0 -> 36.8 | 54 -> 59 | 108 -> 112 | 3461 -> 3746 |
| 16000 | 15 | 15.77 -> 15.79 | 29.49 -> 29.51 | 105.6 -> 105.7 | 51.7 -> 51.7 | 90 -> 90 | 300 -> 300 | 5710 -> 5707 |

EP degree, small config (2048 context, batch 1, prefill buckets 1024 /
2048, `all_greedy`), 1024-token prompts:

| EP | prefill MoE | decode MoE | TTFT | TPOT |
| --- | --- | --- | --- | --- |
| 64 | `shard_on_i_hybrid` | `moe_tkg` all-expert | 219 ms | 25.8 ms |
| 64 | `shard_on_i_hybrid` | `moe_tkg_hits` | 211 ms | 21.6 ms |
| 32 | `shard_on_i_hybrid` | `moe_tkg_hits` | 259 ms | 20.2 ms |
| 16 | `shard_on_i` | `moe_tkg_hits` | 518 ms | 20.5 ms |
| 8 | `shard_on_i` | `moe_tkg_hits` | 564 ms | 20.8 ms |

Smaller EP makes each hit cheaper (an expert's I/tp slice) but the busiest
EP group gets more hits, and the per-layer fixed cost (router, shared expert,
loop barriers) stays, so TPOT flattens out at EP=32. The remaining batch-1
step at EP=32 is about 48% MoE fixed cost, 27% waiting for the busiest rank's
hits, 16% attention and 8% all-reduces.

### Long context: max_model_len 256000 (DCP=8)

`scripts/serve_256k.sh`: max_model_len 256000, `--decode-context-parallel-size 8`,
prefix caching, prefill buckets 512..4096, decode context buckets
8192 / 32768 / 65536 / 98304. Without DCP the KV holds 260,448 tokens, one
max-length request (the scheduler admits by worst case, `max_model_len` per
request); DCP=8 shards it to 2,083,583 tokens, 8 concurrent requests. All
prompts are the fixed real-text ones in `bench_data/` (see its README).

- Passkey retrieval (`scripts/needle.py`, 16k / 32k / 64k / 85k x depth
  0.1 / 0.5 / 0.9): 12/12.
- DCP limits prefill steps to 4096 tokens on neuronx-cc 2.28: the 8192 bucket
  fails to compile (`attention_kv_parallel_segmented_cte` needs 324640 bytes
  of SBUF per partition at a 8192-token Q, `NCC_IGCA037`). Capping the
  kernel's Q chunk at 4096 makes 8192 compile and pass the passkey test, but
  is only 2-5% faster (prefill time is close to linear in tokens), so it is
  not applied.

`scripts/bench_fixed.sh "1024 4096 16000 32768 65536 85000" "1 4 8"`
(`results/bench/ctx256k_dcp8/`):

| input | conc | TTFT med (s) | TTFT P99 (s) | TPOT med (ms) | ITL med (ms) | out tok/s | peak out tok/s | total tok/s |
|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | 0.27 | 0.27 | 30.3 | 30.3 | 32 | 33 | 160 |
| 1024 | 4 | 0.66 | 1.03 | 42.9 | 41.3 | 88 | 100 | 441 |
| 1024 | 8 | 1.12 | 1.96 | 50.7 | 47.3 | 146 | 176 | 730 |
| 4096 | 1 | 0.78 | 0.81 | 30.3 | 30.3 | 30 | 34 | 512 |
| 4096 | 4 | 1.94 | 3.11 | 45.9 | 41.2 | 75 | 100 | 1275 |
| 4096 | 8 | 3.50 | 6.20 | 57.9 | 47.3 | 112 | 176 | 1904 |
| 16000 | 1 | 3.52 | 3.56 | 31.0 | 31.0 | 22 | 33 | 1425 |
| 16000 | 4 | 8.70 | 14.11 | 62.7 | 42.1 | 41 | 96 | 2634 |
| 16000 | 8 | 15.82 | 28.13 | 96.8 | 49.1 | 51 | 168 | 3213 |
| 32768 | 1 | 7.05 | 7.35 | 31.4 | 31.5 | 17 | 32 | 2185 |
| 32768 | 4 | 17.73 | 28.80 | 85.0 | 43.3 | 26 | 96 | 3342 |
| 32768 | 8 | 32.22 | 57.90 | 147.5 | 50.9 | 29 | 160 | 3775 |
| 65536 | 1 | 15.39 | 15.86 | 32.0 | 32.1 | 11 | 32 | 2784 |
| 65536 | 4 | 38.03 | 61.59 | 134.3 | 45.1 | 14 | 92 | 3630 |
| 65536 | 8 | 68.25 | 122.49 | 262.9 | 53.6 | 15 | 152 | 3899 |
| 85000 | 1 | 21.29 | 21.72 | 31.9 | 32.1 | 9 | 32 | 2901 |
| 85000 | 4 | 52.16 | 83.97 | 168.7 | 45.2 | 11 | 92 | 3582 |
| 85000 | 8 | 94.53 | 168.76 | 340.2 | 53.6 | 11 | 152 | 3757 |

- The decode step barely depends on the context: ITL 30.3 -> 32.1 ms from 1k
  to 85k at concurrency 1, 47-54 ms at 8. The large TPOT at long inputs and
  concurrency 4 / 8 is decode stalled behind other requests' prefills.
- DCP=8 costs about 8 ms per decode step against config B (22.2 -> 30.3 ms
  at 1k, concurrency 1: an LSE combine across the 8 DCP ranks in each of the
  80 layers) and 72% prefill time (16k TTFT 2.05 -> 3.52 s; 3.37 s with
  8192-token steps).
- Prefill runs about 4k tokens/s and one request per step (the Neuron
  scheduler takes at most one prefill per step, `max_prefills_per_batch = 1`);
  at concurrency 8 the total throughput saturates at 3.6-3.9k tok/s.

**Shared prefix, 85k input, 92% cached** (`scripts/bench_prefix_fixed.sh "1 4 8"`,
`results/bench/prefix85k_dcp8/`): every request is 85000 tokens, one
78336-token prefix (92.2%) + its own suffix, 10 x concurrency requests after
one warm-up request; one `cache_salt` per run.

| conc | requests | measured hit rate | TTFT med (s) | TTFT P99 (s) | TPOT med (ms) | ITL med (ms) | total tok/s |
|---|---|---|---|---|---|---|---|
| 1 | 10 | 92.2% | 2.57 (cold 21.29) | 2.61 | 32.0 (31.9) | 32.2 | 7940 |
| 4 | 40 | 92.2% | 6.13 (cold 52.16) | 9.83 | 59.1 (168.7) | 45.2 | 16088 |
| 8 | 80 | 92.2% | 10.77 (cold 94.53) | 19.07 | 85.9 (340.2) | 53.6 | 20880 |

Each request prefills its 6664 uncached tokens (4096 + 4096 padded) while
attending to the full 85k context. A rerun with a new salt measures 92.2% and
the same TTFT; with the same salt the suffixes hit too (100%, TTFT 0.71 s), so
a rerun on the same server needs a new salt.

**Prefill-only / decode-only split** (`scripts/bench_pd_split.sh "1 4 8"`,
`results/bench/pd_split_85k_dcp8/`; the same data, on the same single
instance). Prefill only, output 1 token, 10 x concurrency requests:

| conc | hit rate (incl. warm-ups) | TTFT P50 (s) | TTFT P99 (s) | requests/s |
|---|---|---|---|---|
| 1 | 91.5% | 2.55 | 2.57 | 0.42 |
| 4 | 91.9% | 9.39 | 9.53 | 0.43 |
| 8 | 92.0% | 18.74 | 18.86 | 0.43 |

One request prefills at a time, so TTFT is concurrency x 2.3 s of queueing and
the throughput stays at 0.43 requests/s; read the prefill capacity from
requests/s. The mean TTFT at concurrency 1 (2.37 s) is below the median
because `--num-warmups` repeats the first request, which the measured run then
hits completely.

Decode only, the same 80 requests sent twice (output 256, temperature 0.9):

| conc | pass | hit rate | TTFT P50 (s) | TPOT P50 (ms) | ITL P50 (ms) |
|---|---|---|---|---|---|
| 1 | 1 | 91.2% | 2.60 | 31.96 | 32.14 |
| 1 | 2 | 100% | 0.72 | 31.95 | 32.14 |
| 4 | 1 | 91.2% | 5.25 | 59.08 | 45.34 |
| 4 | 2 | 100% | 1.41 | 48.15 | 45.32 |
| 8 | 1 | 91.2% | 9.91 | 85.69 | 53.55 |
| 8 | 2 | 100% | 2.31 | 60.01 | 53.56 |

The first pass equals the shared-prefix benchmark above. In the second pass
each request still re-computes its last token as a prefill step, and on Neuron
prefill and decode never share a step, so decode stalls on every arrival:
TPOT is 2.8 / 6.4 ms above the ITL at concurrency 4 / 8. ITL is the decode
step itself. (A disaggregated decode node counts that re-compute into the
decode batch instead, see `_should_skip_prefilling_padding_in_di` in
`vllm_neuron/vllm/core/scheduler.py`.)

**The same split with MTP (no DCP).** MTP does not run with DCP, so
`scripts/serve_85k.sh` serves the same prompts without DCP: max_model_len
86016, max_num_seqs 4, the same prefill buckets. The KV holds 376k tokens
(349k with MTP, `KV_CAP` 0.65 / 0.61), so 4 requests run at once (concurrency
8 queues 4) and the 80 suffixes no longer fit next to the prefix: the replay
pass 2 hits 92.3%, not 100%, and repeats pass 1. Same `bench_pd_split.sh "1 4 8"`
runs (`results/bench/pd_split_85k_nodcp/`, `results/bench/pd_split_85k_nodcp_mtp3/`,
`MTP_K=3`); DCP=8 rows are pass 1 above (same 91% hit rate):

| | DCP=8 | no DCP | no DCP + MTP k=3 |
|---|---|---|---|
| P: TTFT P50, c=1 / 4 / 8 (s) | 2.55 / 9.39 / 18.74 | 2.11 / 7.57 / 15.11 | 2.14 / 7.68 / 15.31 |
| P: requests/s | 0.43 | 0.53 | 0.53 |
| D c=1: TTFT P50 / TPOT P50 / ITL P50 | 2.60 s / 32.0 / 32.1 ms | 2.16 s / 35.2 / 35.4 ms | 2.18 s / **23.9** / 48.2 ms |
| D c=1: output tok/s | 23.9 | 23.0 | **30.7** |
| D c=4: TTFT P50 / TPOT P50 / ITL P50 | 5.25 s / 59.1 / 45.3 ms | 4.35 s / 72.5 / 61.4 ms | 2.38 s / **58.9** / 73.3 ms |
| D c=4: output tok/s | 48.5 | 43.7 | **57.9** |
| D c=8: TTFT P50 / TPOT P50 / ITL P50 | 9.91 s / 85.7 / 53.6 ms | 27.27 s / 72.5 / 61.5 ms | 19.40 s / 60.4 / 73.4 ms |
| D c=8: output tok/s | **63.1** | 44.0 | 58.7 |

MTP acceptance at temperature 0.9 on these ShareGPT texts is 60 / 29 / 14%
per draft position (33% of drafts, 2.0 tokens per step; MATH-500 greedy:
2.8). With MTP a streamed chunk is one step of 1-4 tokens, so ITL is the
step time and TPOT the per-token time.

- Prefill without DCP is 20% faster (0.53 vs 0.43 requests/s); MTP adds
  nothing measurable to it.
- Decode without DCP reads the full 85k KV on every rank, so its step is
  slower than with DCP at this length (35.4 vs 32.1 ms at c=1, 61 vs 45 ms at
  c=4), the reverse of short contexts.
- MTP k=3 makes up for it: TPOT -32% at c=1 (-25% against DCP=8) and -19% at
  c=4 (equal to DCP=8). Requests finish sooner, so queued prefills start
  sooner and TTFT drops too (c=4: 4.35 -> 2.38 s).
- At c=8 DCP=8 still wins on throughput: it runs 8 requests at once against
  4 here.
- `KV_CAP` 0.65 with MTP runs out of HBM at 4 concurrent requests
  (`nrt_tensor_allocate status=4`); 0.60 fits only 3.99 requests (3 at once).
  `serve_85k.sh` defaults to 0.61 with `MTP_K`.

**The same split with an FP8 KV cache (no DCP).** `KV_DTYPE=fp8
HY3_KV_SCALE_PROFILE=kv_profiles/Hy3-FP8_mtp_neuron.safetensors scripts/serve_85k.sh`
(with and without `MTP_K=3`): FP8 halves the KV bytes, so the same `KV_CAP`
(0.60 / 0.61 with MTP) holds 695k / 697k tokens, 8 requests at once
(max_num_seqs 8), and all 80 suffixes stay cached: the replay pass 2 hits
99.9-100%, as with DCP=8. Decode context buckets are only 32768 (+ 86016):
with 8192 / 32768 / 65536 the batch-8 decode NEFFs did not fit next to the KV
(`Allocation Failure` at batch 8, context 65536; the NEFF code is ~2.2 GiB per
core with one bucket). Decode only (pass 2, ~100% hit), same 80 requests
(`results/bench/pd_split_85k_nodcp_fp8kv{,_mtp3}/`):

| | DCP=8 (BF16 KV) | no DCP, FP8 KV | no DCP, FP8 KV + MTP k=3 |
|---|---|---|---|
| c=1: TPOT / ITL P50 | 31.9 / 32.1 ms | 33.8 / 34.0 ms | **23.4** / 47.4 ms |
| c=1: output tok/s | 28.9 | 27.3 | **37.9** |
| c=4: TPOT / ITL P50 | 48.2 / 45.3 ms | 59.6 / 56.8 ms | **39.2** / 66.6 ms |
| c=4: output tok/s | 75.0 | 61.5 | **92.0** |
| c=8: TPOT / ITL P50 | 60.0 / 53.6 ms | 75.4 / 68.8 ms | **51.0** / 76.7 ms |
| c=8: output tok/s | 116.4 | 94.9 | **138.3** |
| c=8: TTFT P50 | 2.31 s | 2.40 s | 0.95 s |
| prefill-only, requests/s | 0.43 | 0.53 | 0.52 |

- FP8 KV + MTP k=3 without DCP beats DCP=8 at every concurrency: TPOT -27% /
  -19% / -15% and output throughput +31% / +23% / +19% at c=1 / 4 / 8, with 20%
  more prefill throughput. (DCP=8 with FP8 KV is within 3% of BF16 KV, see
  "FP8 KV cache" above, so the DCP=8 column stands for both.)
- FP8 KV alone makes the no-DCP step 4-8% cheaper (ITL 35.4 -> 34.0 ms at c=1,
  61.4 -> 56.8 ms at c=4) and doubles the requests that run at once; MTP then
  gives 2.0 tokens per step (temperature 0.9) for a step 40% / 17% / 11% longer
  at c=1 / 4 / 8, i.e. TPOT -31% / -34% / -32% over FP8 KV alone.
- The step itself is still cheapest with DCP=8 (each rank reads 1/8 of the
  history): MTP together with DCP would combine both, see the MTP limitations.
- Where the time goes in this configuration (device profile of a cold 85k
  prefill and of 1- and 6-request decode, MoE load imbalance per layer, and how
  the model is split over the 64 ranks): [docs/PROFILE_85K.md](docs/PROFILE_85K.md).
- The 8-request verify steps have 32 tokens, above the hits limit (16), so they
  run the all-expert `moe_tkg`. Raising `HY3_DECODE_HITS_MAX_T` to 32 is
  slower (c=8 pass 2: ITL 76.7 -> 85.7 ms, TPOT 51.0 -> 56.0 ms, 138 -> 127
  tok/s; c=4 unchanged; `results/bench/pd_split_85k_nodcp_fp8kv_mtp3_maxt32/`):
  at 32 tokens nearly every local expert is hit, so the hits kernel saves no
  weight reads and only adds its runtime loop and cross-rank waiting.
- EP=32 (`EP=32`, 6 experts per EP group, expert TP=2) instead of 64: decode
  pass 2 TPOT 23.4 -> 22.4 ms at c=1 (ITL 47.4 -> 44.9 ms), 39.2 -> 38.8 at c=4,
  51.0 -> 53.3 at c=8 (138 -> 133 tok/s); prefill 13% slower (c=1 TTFT 2.14 ->
  2.41 s, 0.52 -> 0.46 requests/s; `results/bench/pd_split_85k_nodcp_fp8kv_mtp3_ep32/`).
  More experts per rank evens out the single-request hits only a little (2.5 ms
  of the ~10 ms waiting), while the half-width experts cost prefill and the
  8-request all-expert steps, so EP=64 stays.
- MTP with FP8 KV needs scales for the MTP layer (`model.layers.80`):
  `kv_profiles/Hy3-FP8_mtp_neuron.safetensors` is the shipped profile plus that
  layer (`calibrate_kv_scales_neuron.py calibrate --mtp`); the MTP acceptance
  is the same as with BF16 KV (60% of drafts on MATH-500, 33% here).

### MTP speculative decoding

Latency config (`scripts/serve_latency.sh`, EP=64, `MTP_K=k`), MATH-500 chat
prompts (no_think, greedy, up to 512 tokens, `scripts/bench_real_prompts.py`):
20 problems one at a time (c=1) and 40 four at a time (c=4). Random-token
prompts are almost never accepted, so they do not show MTP's gain.

| config | TPOT c=1 | out tok/s c=1 | TPOT c=4 | out tok/s c=4 |
| --- | --- | --- | --- | --- |
| plain, `moe_tkg` | 26.3 ms | 38 | 53.2 ms | 72 |
| plain, `moe_tkg_hits` | 21.7 ms | 45 | 34.0 ms | 111 |
| MTP k=4, `moe_tkg` | 20.6 ms | 49 | 16.9 ms | 220 |
| MTP k=3, `moe_tkg_hits`, limit 4 (default) | 13.6 ms | 70 | 19.1 ms | 198 |
| **MTP k=3, `moe_tkg_hits`, limit 16 (`serve_latency.sh`)** | **13.8 ms** | **70** | **17.1 ms** | **216** |
| MTP k=4, `moe_tkg_hits`, limit 20 (`serve_latency.sh`) | 13.8 ms | 70 | 17.0 ms | 218 |
| MTP k=3, `moe_tkg_hits`, limit 16, FP8 KV (`KV_DTYPE=fp8`) | 13.8 ms | 69 | 17.2 ms | 217 |

c=1 tok/s for the `moe_tkg` rows is 1000 / TPOT. Acceptance per draft
position is 85 / 60 / 34 / 16% (2.8 tokens per step at k=3, 2.9 at k=4), the
same with either MoE kernel. TTFT is 0.18-0.19 s at c=1 and 0.24-0.25 s at
c=4 (plain: 0.16 / 0.22-0.27 s). MTP k=3 with `moe_tkg_hits` is the
recommended latency setting: k=4 does not help (its fourth draft is accepted
16% of the time). An FP8 KV cache (with the MTP-layer profile,
`kv_profiles/Hy3-FP8_mtp_neuron.safetensors`) changes nothing at these short
contexts (acceptance 60% of drafts either way); it pays at long context (see
"The same split with an FP8 KV cache" above). Design, correctness checks and the per-step overhead:
[docs/MTP_DESIGN.md](docs/MTP_DESIGN.md).

#### Draft prefill under sequence parallelism (fixed)

Before this fix, the draft's KV for the whole prompt was wrong:
- At prefill each rank holds a different SP token shard. `_mtp_step` ran `eh_proj` on those shards.
- `eh_proj` is a `ColumnParallelLinear` with a gathered output, so column block r of row i came
  from token i of rank r's shard.
- Every rank ended up with the same spliced rows, and the MTP layer wrote its KV for every prompt
  position from them.
- Decode steps (no SP) were correct. So each draft step attended to a wrong prompt and to correct
  generated tokens only.

The draft now runs the embedding, `enorm` / `hnorm` and `eh_proj` on every token, and keeps its
token shard only after `eh_proj`. Decode is unchanged.

How the draft KV was checked:
- The draft layer's KV cache rows (KV head 0) were read back after prefill.
- They were compared with a CPU reference of the MTP layer (vLLM `hy_v3_mtp` semantics) on the
  same target hidden states.

| draft KV of a 2048-token prompt | before | after |
|---|---|---|
| K cos | 0.80 | 0.9999 |
| V cos | 0.37 | 0.9997 |

Acceptance on one trn2.48xlarge.
- Config: FP8 KV, MTP k=3, sparse decode 0.125 + sparse prefill 0.25, `HY3_DECODE_MOE_FUSED=1`.
- Workload: the `bench_data/prefix` c8 prompts cut to the last N tokens, greedy raw-prompt
  continuation of 256 tokens, 16 requests, C = 8.

| prompt tokens | tokens/step before | tokens/step after | per-position acceptance after |
|---|---|---|---|
| 2048 | 2.38 | **3.14** | 0.95 / 0.71 / 0.48 |
| 32768 | 2.64 | **3.14** | 0.97 / 0.72 / 0.45 |
| 84000 | 2.49 | **2.94** | 0.94 / 0.64 / 0.35 |

H100 1P1D (vLLM 0.29.0, FP8 KV, MTP k=3) gives 3.12-3.15 tokens/step (0.98 / 0.73 / 0.41) on the
same 85k prompts. The MATH-500 acceptance above was measured before this fix.

#### Decode node: async spec decode when a request joins (fixed)

On a PD decode node (`kv_consumer`), a new request joins the running batch with its KV loaded
and no local prefill, so it has no row in the runner's async spec transition cache.
`_drop_transition_spec_decode_if_cache_incomplete` then stripped spec decode for the whole step.

That step's CPU-built `input_ids` read the running requests' tokens at the scheduler's optimistic
`num_computed_tokens`, which are `-1` / `0` placeholders:
- the target decoded those at the running requests' positions, and their outputs derailed;
- without async scheduling, the same `-1` ids fail validation (`Token IDs out of range`) instead.

Every join hit this: 11008 "Missing async spec transition drafts" warnings in one c=8 run. So at
C >= 2 the PD output differed from C = 1 within a few tokens. MTP acceptance fell to the level of
fresh text, because the derailed continuation no longer copied the prompt.

The fix (`neuron_model_runner.py`) keeps spec decode when only new requests miss the cache:
- running requests replay their cached `[bonus, drafts]` rows, and positions are corrected on
  device as in any spec step;
- a new request takes its CPU-built first token (written fresh by `add_request`), followed by
  pad drafts that the first verify rejects.

1P1D check: greedy continuation of the `bench_data/prefix` prompts cut to the last 2048 tokens, 16
requests, 256 tokens each.

| C | tokens/step before | tokens/step after | greedy text vs C = 1 after |
|---|---|---|---|
| 1 | 3.10 | 3.09 | - |
| 2 | 2.32 | 3.11 | identical (all chars) |
| 8 | 2.34 | 3.08 | identical (all chars) |

**1P1D `scripts/pd/pd_bench_final.sh` with both fixes.**
- Setup: two trn2.48xlarge, the `scripts/pd` defaults; C = 8 unless noted.
- "Before" is the same setup without either fix.
- The H100 1P1D reference is vLLM 0.29.0 with FP8 KV + MTP k=3.

| | before | both fixes | H100 FP8 + MTP |
|---|---|---|---|
| P-only TTFT P50, C = 1 / 4 / 8 / 12 (s) | 1.13 / 2.95 / 5.78 / 8.71 | 1.13 / 2.96 / 5.79 / 8.74 | 0.84 / 2.10 / 3.94 / 5.90 |
| end-to-end TPOT median / P99 (ms) | 19.6 / 24.0 | **15.45 / 22.6** | 7.6 / 11.4 |
| end-to-end MTP tokens/step | 2.23 | **2.71** | 2.98 |
| end-to-end output tok/s | 353 | **416** | 795 |
| D-only TPOT mean / P99 (ms) | 19.86 / 23.73 | 19.71 / 25.19 | 11.02 / 13.31 |
| D-only MTP tokens/step | 2.20 | 2.22 | 2.23 |

Notes on the table:
- The end-to-end case continues raw prompts (`/v1/completions`), so it shows the gain.
- The D-only replay samples chat answers at temperature 0.9, where Trn2 and H100 accept the same
  2.2 tokens per step. Its TPOT is bound by the decode step time (about 44 ms at C = 8), not by
  MTP.

### Comparison with the previous implementation (vllm-neuron 0.24, neuronx-cc 2.27)

The same model code previously ran on vllm-neuron 0.24. Same benchmark, same
seeds:

| config | 0.24 | 0.29 (this folder) |
| --- | --- | --- |
| B, prefill bucket 2048, 16k input, c=1: TTFT / TPOT | 4.40 s / 30.6 ms | 2.61 s / 27.7 ms |
| B, 16k input, c=15: TTFT / total tok/s | 34.4 s / 3078 | 20.3 s / 4756 (multi-bucket: 15.8 s / 5707) |
| B, 1k input, c=15: TTFT / total tok/s | 4.16 s / 1036 | 2.45 s / 1309 (multi-bucket: 1.59 s / 1475) |
| C, 1k input, c=127: TTFT / output tok/s | 32.8 s / 378 | 12.3 s / 773 (multi-bucket) |
| C, 512 input, c=127: TTFT / output tok/s | 32.8 s / 388 | 8.9 s / 976 (multi-bucket) |
| latency config, 16k input, c=1: TTFT / TPOT | 1.80 s / 26.9 ms | 1.83 s / 27.7 ms (`moe_tkg_hits`: 1.83 s / 23.5 ms) |

Where the gains come from:

1. **The dynamic-block MoE prefill kernel now works with prefix caching.** On
   neuronx-cc 2.27, segmented attention followed by any kernel with a
   register-driven loop (`moe_cte` `shard_on_i_hybrid`, `pack_tokens`) in the
   same graph failed to compile (`NCC_IBIR735` / `NCC_ICFG001`), so segmented
   prefill had to fall back to the static `shard_on_i` MoE, which computes the
   worst-case block count (~3x slower at 2048 tokens). On neuronx-cc 2.28 the
   minimal repro passes in every combination and the full model compiles, so
   every prefill path uses the hybrid kernel. This gives the ~40% TTFT drop of
   config B at an unchanged configuration.
2. **Prefill buckets no longer have to equal the KV segment size.** With
   prefix caching, 0.24 required `num_batched_tokens_buckets ==
   kv_segment_size_buckets`, so every prompt was padded to a 2048-token step.
   0.29 compiles each (Q bucket, segment size) pair, so short prompts use a
   512 / 1024 bucket and long prompts 4096 / 8192-token steps.
3. With the same all-expert `moe_tkg`, single-request decode is ~0.8 ms (3%)
   slower per token than on 0.24 in the latency config (same model code); the
   cause has not been isolated.

### Status of issues found on 0.24

| Issue on 0.24 | 0.29 |
| --- | --- |
| Segmented attention + register-loop kernel in one graph fails to compile (`NCC_IBIR735` / `NCC_ICFG001`) | **Fixed** (neuronx-cc 2.28): repro and full model compile and run |
| CTE MLP silently wrong when the per-rank intermediate is not a multiple of 128 (I=208: cos ~0.7) | **Fixed**: I=208 unpadded matches the padded result. The model still pads (validated path) |
| `build_blockwise_mapping` with an EP-TP group > 1 fails to trace (`torch.narrow`) | Fixed in the plugin (`index_select`) |
| `NF.moe_tkg` rejects plain `float8_e4m3fn` weights | Still rejected; the model calls the kernel wrapper directly |
| `"fp8"` not in the Neuron platform's supported quantizations | Still missing; `register()` adds it |
| DCP kernels build their `ReplicaGroup` from the caller's group only, so each DCP group compiles its own NEFFs | Unchanged (compile time only, not correctness) |
| Prefill and decode never share a step | Unchanged; at high concurrency TPOT reaches ~2x ITL |

## Testing

Install the package first (`pip install -e contrib/models/hy_v3`).

### Integration test (accuracy)

The full model is too large for an FP32 HuggingFace reference, so the test
builds a 4-layer checkpoint from the real weights (`scripts/make_mini.py`,
FP8 for vLLM plus a dequantized BF16 copy for HF) and runs the template's
three-way logit validation in four cases: flash-attention prefill, and prefix
caching (segmented prefill) with batch-4 decode, each with a BF16 KV cache and
with an FP8 KV cache (packed K). The FP8-KV cases need a KV scale profile in
`HY3_KV_SCALE_PROFILE` (a full-model profile also covers the 4-layer model) and
are skipped without one. It needs the whole trn2.48xlarge and takes about 32
minutes for the four cases (Neuron SDK 2.33.0: 4 passed).

Two settings differ from the template, because the model runs FP8 activations
against a BF16 golden: the top-K bounds are 0.1 / 0.12 / 0.15 / 0.15
(measured <= 0.06), and the argmax divergence tolerance is 0.25 logits instead
of 1 BF16 ULP. On the flat 4-layer distribution FP8 rounding flips near-ties
whose golden top-1 / top-2 gap is 0.06-0.125; a real defect shows gaps of
several logits. The FP8-KV cases pass with the same bounds.

```bash
HY3_CKPT=/path/to/Hy3-FP8 HY3_KV_SCALE_PROFILE=/path/to/kv_scales.safetensors \
    pytest contrib/models/hy_v3/test/integration/test_model.py -v -s --timeout=7200
```

`HY3_MINI_CKPT` sets where the 4-layer checkpoint is built and cached
(default `/tmp/hy3_mini4`).

### Module tests

Single-NeuronCore tests of the FP8 MLP (dense layer, shared expert, 8-way decode
copy; CTE and TKG paths) and the routed experts (`moe_cte` hybrid, `moe_tkg`,
`moe_tkg_hits` with 0-3 local hits, `moe_tkg_hits_batch` with 2 / 4 tokens),
the fused decode MoE block (`test_moe_tkg_fused.py`: T = 8 / 32 against
an FP32 fake-quant reference with the torch router; `HY3_FUSED_BENCH=1` also times 8
chained layers against the production path), and the FP8 KV attention of layer 0 (`test_fp8_kv_attention.py`: cache writes
in flash and segmented prefill, two decode steps on the standard, packed and
packed-fallback layouts, calibrated and unit scales, plus wrong-scale-contract
controls that must miss). They build the kernel inputs from the real
checkpoint (rank-0 shard) through the model's own weight loaders and compare
against an FP32 fake-quant reference.

```bash
HY3_CKPT=/path/to/Hy3-FP8 pytest contrib/models/hy_v3/test/module/ -v -s
```

### Unit tests

CPU only: FP8 weight transforms (halving, scale layout, padding, expert
packing), the router against the HuggingFace `HYV3TopKRouter`, the KV scale
sources (profile validation, checkpoint `k_scale` / `v_scale` with the 448 ->
240 rescale, the error without scales and the opt-in 1.0 default), the packed-K layout and the drift guards that
compare the local copies in `src/fp8_kv_layout.py` and
`src/kernels/moe_tkg_common.py` with the plugin / nkilib originals (skipped when
those are not importable).

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/hy_v3/test/unit/ -v
```

### Scripts

| Script | Purpose |
| --- | --- |
| `scripts/serve.sh` | `vllm serve` with prefix caching (configs B / C / DCP below); `EP=<n>` sets the EP degree |
| `scripts/serve_latency.sh` | single-request latency config (no prefix caching, prefill buckets 512..16384); `MTP_K=<k>` enables MTP |
| `scripts/bench_real_prompts.py` | TTFT / TPOT / throughput on MATH-500 chat prompts, plus MTP acceptance from `/metrics` |
| `scripts/check_mtp.py` | MTP vs plain greedy output (lossless check) and acceptance; `KV_DTYPE=fp8`, `PREFIX_CACHING=1`, `DUMP_IDS` |
| `scripts/profile_85k.sh` / `scripts/profile_85k_client.py` | device-profiling server (85k FP8 KV + MTP config, fewest NEFFs) and the profiled sessions (cold prefill, 1- and N-request decode) |
| `scripts/expert_load.sh` / `expert_load_client.py` / `expert_load.py` | capture each token's routed experts (tensor_capture of `moe_route.<layer>`), real-text workload, and the expert-placement simulation (docs/PROFILE_85K.md, "Expert load") |
| `scripts/sparse_decode_study.sh` / `sparse_decode_eval.py` / `longbench_v2.py` | EXPERIMENTAL block-sparse decode (`HY3_SPARSE_DECODE`, off; same accuracy, 8-request decode steps 25% faster at 85k): server, long-continuation agreement, LongBench v2 multiple choice ([docs/SPARSE_DECODE.md](docs/SPARSE_DECODE.md)) |
| `scripts/cold_ttft.py` | cold-prefill TTFT (own `cache_salt` per request); with `HY3_SPARSE_PREFILL` (EXPERIMENTAL block-sparse prefill, off; cold 85k prefill 15% faster, prefix-hit prefills 29-33% faster; with `HY3_SPARSE_DECODE`, 8-request TPOT 51.8 -> 36.3 ms, [docs/SPARSE_DECODE.md](docs/SPARSE_DECODE.md#sparse-prefill-hy3_sparse_prefillfraction-experimental-off)) |
| `scripts/profile_analyze.py` | tables for docs/PROFILE_85K.md from neuron-explorer parquet exports: per-phase times and MoE waiting per layer and rank |
| `scripts/check_mtp_gaps.py` | HF FP32 log-probs at the first token where the plain and MTP runs differ (near-tie or defect) |
| `scripts/bench_matrix.sh` | `vllm bench serve` matrix, random dataset, a distinct seed per cell so cells never share prefix-cache entries |
| `scripts/serve_256k.sh` | long-context config: max_model_len 256000, DCP=8, prefill buckets up to 4096 (with DCP the 8192 bucket fails to compile, `NCC_IGCA037`) |
| `scripts/bench_fixed.sh` | `vllm bench serve` matrix over the fixed real-text prompts in `bench_data/prompts` (same requests every run) |
| `scripts/gen_bench_prompts.py` | generates `bench_data/prompts`: sglang-style ShareGPT text cut / repeated to an exact token count |
| `scripts/bench_prefix_fixed.sh` | prefix-cache benchmark over `bench_data/prefix` (85k input, 92% shared prefix): warm-up request, one `cache_salt` per run, hit rate from `/metrics` |
| `scripts/pd/pd_bench_final.sh`, `pd_bench_split.sh`, `pd_bench_dcust.sh` | the comparison set: P-only (output 1) at C = 1 / 4 / 8 / 12, the end-to-end 1024-output case at C = 8, then the customer D-only replay (1200 output) at C = 8 ("Trn2 vs H100") |
| `scripts/pd/gpu/` | the H100 1P1D reference: Dockerfile (vLLM 0.29.0 + EFA + GDRCopy + Mooncake) and `gpu_serve.sh` |
| `scripts/pd/` | 1P1D (Mooncake / NIXL) runs on two trn2.48xlarge: servers, proxy, the 85k / 92% bench and the accuracy checks through the proxy ([scripts/pd/README.md](scripts/pd/README.md), "PD test method and results") |
| `src/kernels/moe_tkg_fused.py` | EXPERIMENTAL fused decode MoE block (`HY3_DECODE_MOE_FUSED=1`): norm + router + routed + shared shard, W8A8 + FP8 double-row; 8-request PD TPOT 26.5 -> 19.9 ms ("Fused decode MoE") |
| `scripts/bench_pd_split.sh` | prefill-only (output 1) and decode-only (replay twice) split over `bench_data/prefix`, with per-cell `cache_salt`, hit rates and MTP acceptance |
| `src/kernels/moe_cte_fp8w.py`, `bwmm_shard_on_I_fp8w.py` | prefill routed experts: nkilib's `shard_on_i_hybrid` with the FP8 weight tiles fed to the matmul (no BF16 copy; identical output, routed experts -33% per prefill chunk; `HY3_MOE_CTE_FP8W=0` for the nkilib kernel; [docs/PROFILE_85K.md](docs/PROFILE_85K.md)) |
| `scripts/tensore_mm_bench.py` | one-core TensorE cost of the decode MoE matmul shapes, BF16 x FP8 vs FP8 x FP8 (W8A8, with and without `double_row`): W8A8 is not faster there (`results/w8a8_decode/`, docs/PROFILE_85K.md) |
| `scripts/profile_attn_breakdown.py` | attention-module time of a profiled decode step by model.py function (sparse-decode profile, [docs/SPARSE_DECODE.md](docs/SPARSE_DECODE.md)) |
| `scripts/sparse_select_bench.py` | single-core timing of the sparse-decode block selection variants ([docs/SPARSE_DECODE.md](docs/SPARSE_DECODE.md)) |
| `scripts/serve_85k.sh` | 85k-context config without DCP (max_model_len 86016, 4 requests at once), for MTP on `bench_data/prefix` |
| `scripts/gen_prefix_prompts.py` | generates `bench_data/prefix`: real-text shared prefix + distinct suffixes, exact token counts |
| `scripts/needle.py` | passkey retrieval at 16k-85k (`gen` writes `bench_data/needle.jsonl.gz`, `run` checks a server) |
| `scripts/bench_sharegpt.sh` | `vllm bench serve` on ShareGPT (vLLM keeps prompts <= 1024 tokens only) |
| `scripts/bench_prefix_cache.sh` | prefix-cache benchmark: shared-prefix share sweep, hit rate read from the server's `/metrics` |
| `scripts/summarize_bench.py` | markdown table from benchmark JSONs, optionally against a baseline directory |
| `scripts/eval_math500.py` | MATH-500 through the OpenAI API, graded with math-verify |
| `scripts/make_mini.py` | reduced real-weight checkpoint (first N layers), FP8 + BF16 |
| `scripts/hf_reference.py` | HF CPU reference logits for the BF16 reduced checkpoint |
| `scripts/compare_hf_logits.py` | vLLM Neuron vs the HF reference, per-step logits |
| `scripts/validate_features.py` | segmented prefill, prefix-cache hits, batch-128 vs batch-4 consistency (optionally with DCP) |
| `scripts/dequant_bf16.py` | BF16 copy of the full checkpoint for KV calibration |
| `scripts/calibrate_kv_scales.py` | llm-compressor KV calibration, exports the KV scale profile (`kv_calib/*.json` data mixes) |
| `scripts/calibrate_kv_scales_neuron.py` | KV calibration on Neuron: absmax of the BF16 KV cache slots a prefill-only run writes (`samples` builds the token ids of a `kv_calib/*.json` mix, `calibrate` writes the profile) |
| `scripts/validate_fp8_kv.py`, `compare_kv_runs.py`, `run_kv_mode.sh` | reduced-model BF16-KV vs FP8-KV runs (modes m1 / m2 / m3: full prefill, packed + prefix, DCP 8) and their comparison |
| `scripts/kv_acceptance.sh`, `kv_report.py` | full-model acceptance run (manifest, memory sampling, load matrix, MATH-500, >= 30 min stability) and the BF16 / FP8 gate |
| `scripts/kv_manifest.py`, `kv_mem_sampler.py` | run manifest (args, server env, commit, profile sha256, KV pool bytes) and neuron-monitor memory time series for the gate |
| `scripts/kv_api_check.py` | full-model functional check through the API (long multi-chunk prompts, prefix hits, partial hits, concurrency), e.g. DCP 8 vs DCP 1 |
| `scripts/kv_cache_isolation.sh` | compile-cache isolation of the KV scale profile (same profile reuses NEFFs; another profile does not reuse stale decode NEFFs) |
