# FP8 Weight Quantization for MoE Decode on Trn2

## Overview

MoE expert weights (gate/up/down projections) are quantized from BF16 to FP8, halving HBM memory bandwidth during decode. Enabled via `VLLM_NEURON_FP8_EXPERT_WEIGHTS=1`.

## Implementation

In `vllm_neuron/model/qwen3/model_bf16.py`:

- 128 experts per layer, per-expert absmax scaling (max=240), quantized to `torch.float8_e4m3fn`
- Dual weight storage: BF16 originals for prefill (CTE kernel), FP8 copies for decode (TKG kernel)
- Scale shapes: gate_up `[E, 2, 1]`, down `[E, 1]` — matches kernel's STATIC quant expectation
- Input scales set to 1.0 (passed for detection only, not used by kernel on Trn2)

## Key Technical Findings

1. **MX path (uint32 packed) does NOT work on Trn2** — The `moe_block_tkg_wrapper` reinterprets uint32 → `float8_e4m3fn_x4`, which triggers `_SUPPORTED_MX_DTYPES` detection, which gates on `nisa.get_nc_version() >= gen4` (Trn3+). Hard assertion failure.

2. **STATIC path works on Trn2** — When weights are `float8_e4m3fn` (non-packed, not in `_SUPPORTED_MX_DTYPES`), the kernel detects `QuantizationType.STATIC` via `expert_gate_up_input_scale != None`. Trn2 hardware does native BF16 x FP8 matmul. No gen4 requirement.

3. **neuroncc needs writable TMPDIR** — The compiler writes temp files in CWD by default. When CWD is a read-only bind mount (`/opt/vllm-neuron/`), compilation fails with `[Errno 30] Read-only file system`. Fix: `export TMPDIR=/tmp`.

4. **Neuron cores not released after kill** — After `kill -9` on the vLLM server, neuron cores stay marked as occupied in `neuron-ls` even though the process is gone. Workaround: use a different set of cores for the next run.

## Performance Results

Tongyi-DeepResearch-30B-A3B, trn2.48xlarge, TP=8, batch=1, 200 tokens decode:

| Metric | BF16 Baseline | FP8 Weights | Improvement |
|--------|--------------|-------------|-------------|
| Avg ITL | 38.5ms | 25.0ms | -35% |
| P50 ITL | 38.6ms | 25.0ms | -35% |
| P99 ITL | 39.4ms | 27.4ms | -30% |
| Decode speed | 25.9 tok/s | 40.0 tok/s | +54% |

The 54% throughput gain is close to the theoretical 2x from halving expert weight reads. The gap is due to non-MoE overheads (attention, router, all-reduce) that remain unchanged.
