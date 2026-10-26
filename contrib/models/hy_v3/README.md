# Contrib Model: Hy3 (tencent/Hy3-FP8)

Serves [tencent/Hy3-FP8](https://huggingface.co/tencent/Hy3-FP8) with vLLM on
AWS Trainium through the `vllm-neuron` plugin. Hy3 is a ~295B-parameter MoE
decoder (80 layers, 192 routed experts top-8 + 1 shared expert, GQA 64Q / 8KV)
with a static per-tensor FP8 checkpoint. All linear layers run with the
checkpoint's FP8 weights; the KV cache is BF16.

The architecture, kernel mapping and feature status are in
[src/README.md](src/README.md).

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
implementation with no further code. The plugin also adds `"fp8"` (the
checkpoint's `quant_method`) to the Neuron platform's supported quantizations;
the model consumes the FP8 weights and scales itself. Nothing outside this
folder is modified.

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
- Not supported: data parallel / attention DP, MTP speculative decoding.
  They are rejected at startup. FP8 KV cache needs a calibrated KV scale
  profile (the checkpoint ships none); see "FP8 KV cache" below.
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
  both paths off. See "Decode: only the routed experts" under Results.
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
(`src/kv_scales.py`), produced with llm-compressor:

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
  convention, rescaled by 448 / 240 (they are calibrated for e4m3fn); else 1.0
  for every layer, with a warning, as in the upstream models. For a profile the
  loader checks the format, direction, the platform FP8 range and per-layer
  fingerprints of the attention weights (`load_weights_lite` reads the scales
  too). The server log line `Hy3 FP8 KV scales (80 layers) from ...` names the
  source. tencent/Hy3-FP8 ships no KV scales, so without a profile it runs at
  scale 1.0. Decode passes
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
  configuration tried.
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

## Compatibility Matrix

| Instance / software | vllm-neuron 0.29.0.1.2.0 (vLLM 0.29.0) | earlier |
| ------------------- | -------------------------------------- | ------- |
| Trn3                | Not tested                             | Not tested |
| Trn2 (trn2.48xlarge) | Working                               | Not tested with this folder |

Tested component versions: neuronx-cc 2.28.7631, nki 0.7.0,
libtorch-neuronx-lite 2.13.0.1.0, Neuron runtime 2.35.30, collectives 2.35.43,
driver 2.31.7, torch 2.13.0, transformers 5.17.0, Python 3.12.

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
three-way logit validation, once with flash-attention prefill and once with
prefix caching (segmented prefill) and batch-4 decode. It needs the whole
trn2.48xlarge and takes about 12 minutes.

Two settings differ from the template, because the model runs FP8 activations
against a BF16 golden: the top-K bounds are 0.1 / 0.12 / 0.15 / 0.15
(measured <= 0.06), and the argmax divergence tolerance is 0.25 logits instead
of 1 BF16 ULP. On the flat 4-layer distribution FP8 rounding flips near-ties
whose golden top-1 / top-2 gap is 0.06-0.125; a real defect shows gaps of
several logits.

```bash
HY3_CKPT=/path/to/Hy3-FP8 pytest contrib/models/hy_v3/test/integration/test_model.py \
    -v -s --timeout=7200
```

### Module tests

Single-NeuronCore tests of the FP8 MLP (dense layer, shared expert, 8-way decode
copy; CTE and TKG paths) and the routed experts (`moe_cte` hybrid, `moe_tkg`,
`moe_tkg_hits` with 0-3 local hits, `moe_tkg_hits_batch` with 2 / 4 tokens).
They build the kernel inputs from the real checkpoint (rank-0 shard) through
the model's own weight loaders and compare against an FP32 fake-quant
reference.

```bash
HY3_CKPT=/path/to/Hy3-FP8 pytest contrib/models/hy_v3/test/module/ -v -s
```

### Unit tests

CPU only: FP8 weight transforms (halving, scale layout, padding, expert
packing) and the router against the HuggingFace `HYV3TopKRouter`.

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/hy_v3/test/unit/ -v
```

### Scripts

| Script | Purpose |
| --- | --- |
| `scripts/serve.sh` | `vllm serve` with prefix caching (configs B / C / DCP below); `EP=<n>` sets the EP degree |
| `scripts/serve_latency.sh` | single-request latency config (no prefix caching, prefill buckets 512..16384) |
| `scripts/bench_matrix.sh` | `vllm bench serve` matrix, random dataset, a distinct seed per cell so cells never share prefix-cache entries |
| `scripts/serve_256k.sh` | long-context config: max_model_len 256000, DCP=8, prefill buckets up to 4096 (with DCP the 8192 bucket fails to compile, `NCC_IGCA037`) |
| `scripts/bench_fixed.sh` | `vllm bench serve` matrix over the fixed real-text prompts in `bench_data/prompts` (same requests every run) |
| `scripts/gen_bench_prompts.py` | generates `bench_data/prompts`: sglang-style ShareGPT text cut / repeated to an exact token count |
| `scripts/bench_prefix_fixed.sh` | prefix-cache benchmark over `bench_data/prefix` (85k input, 92% shared prefix): warm-up request, one `cache_salt` per run, hit rate from `/metrics` |
| `scripts/bench_pd_split.sh` | prefill-only (output 1) and decode-only (replay twice) split over `bench_data/prefix`, with per-cell `cache_salt` and hit rates |
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
| `scripts/validate_fp8_kv.py`, `compare_kv_runs.py`, `run_kv_mode.sh` | reduced-model BF16-KV vs FP8-KV runs (modes m1 / m2 / m3: full prefill, packed + prefix, DCP 8) and their comparison |
| `scripts/kv_acceptance.sh`, `kv_report.py` | full-model acceptance run (manifest, memory sampling, load matrix, MATH-500, >= 30 min stability) and the BF16 / FP8 gate |
| `scripts/kv_manifest.py`, `kv_mem_sampler.py` | run manifest (args, server env, commit, profile sha256, KV pool bytes) and neuron-monitor memory time series for the gate |
| `scripts/kv_api_check.py` | full-model functional check through the API (long multi-chunk prompts, prefix hits, partial hits, concurrency), e.g. DCP 8 vs DCP 1 |
| `scripts/kv_cache_isolation.sh` | compile-cache isolation of the KV scale profile (same profile reuses NEFFs; another profile does not reuse stale decode NEFFs) |
