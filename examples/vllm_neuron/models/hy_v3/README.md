# Hy3-FP8 (tencent/Hy3-FP8) on Trn2

Scripts, raw results and the full test report for the Hy3 static-FP8
implementation in `vllm_neuron/model/hy_v3/`. The detailed report (Chinese) is
in [REPORT.md](REPORT.md).

## Summary

Measured on trn2.48xlarge with TP=64 and full EP=64, FP8 weights and a BF16 KV
cache.

| Item | Result |
|---|---|
| MATH-500, no_think (temp 0.9, max 4096 tok) | 96.8% (DCP=8: 96.0%) |
| MATH-500, high reasoning (temp 0.9, max 15000 tok) | 94.2% (27 of 29 misses are length truncations) |
| 4-layer real-weight model vs HF fp32 | logits cosine >= 0.9986 |
| Single-request decode (TPOT) | ~30 ms (DCP=8: ~37 ms) |
| Prefill throughput | ~3.7k tok/s (DCP=8: ~2.8k tok/s) |
| KV capacity, KV cap 0.45 | 260k tokens (DCP=8: 2.08M tokens) |
| Peak decode throughput | ~1.7k tok/s (batch 127) |

Prefill cannot be batched with decode on the Neuron backend, so end-to-end
throughput at high concurrency is bound by serialized prefill.
Prefill/decode disaggregation is the next step. It needs EFA, even on a single
node (see REPORT.md §9).

## Layout

| Path | What |
|---|---|
| `serve.sh` | `vllm serve` wrapper for the configs used in the report (baseline B/C, DCP=8) |
| `bench_matrix.sh` | `vllm bench serve` matrix (random dataset, per-cell seed so cells never share prefix-cache entries) |
| `eval_math500.py` | MATH-500 against the OpenAI API, graded with math-verify |
| `make_mini.py` | Build a reduced real-weight checkpoint (first N layers), both FP8 and dequantized BF16 |
| `hf_reference.py` | HF transformers CPU reference logits / greedy tokens for the BF16 mini checkpoint |
| `compare_hf_logits.py` | vLLM Neuron vs HF reference, per-step logits comparison |
| `validate_features.py` | Segmented prefill, prefix-cache hit and batch-128 consistency checks (optionally with DCP) |
| `kernel_tests/` | Single-NeuronCore micro-tests of the FP8 kernels with real weight shards |
| `inspect_scales.py` | Checkpoint FP8 scale statistics |
| `di/` | NIXL probe and `libcuda` stub from the single-node P/D investigation |
| `results/bench/` | Raw `vllm bench` JSON. `summary.csv` covers the clean runs; `archive_fixed_seed/` holds the early fixed-seed runs, whose concurrency > 1 cells are prefix-cache contaminated |
| `results/math500/` | Per-problem MATH-500 outputs (gzipped jsonl) |

## Quick start

```bash
# Baseline, long context (config B)
MODEL=/path/to/Hy3-FP8 ./serve.sh 16384 32 '[1, 4, 16, 32]' '[4096, 8192]'

# DCP=8: 8x KV capacity for many long-context requests
MODEL=/path/to/Hy3-FP8 DCP=8 ./serve.sh 32768 64 '[1, 4, 16, 32, 64]' '[4096, 8192, 16384]'

# Benchmarks and accuracy
OUT_DIR=results/bench/mine MODEL=/path/to/Hy3-FP8 ./bench_matrix.sh "1024 4096" "1 4 15" 256
python eval_math500.py no_think math500.jsonl --concurrency 21
```

Keep `KV_CAP` at 0.45 or lower. At 0.6 a NEFF that is loaded lazily later in
the run fails to allocate its DMA rings, and the server dies mid-run.
