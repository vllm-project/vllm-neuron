# Contrib Model: Qwen3.5 (dense and MoE, text-only)

This folder serves the Qwen3.5 family with vLLM on AWS Trainium through the
`vllm-neuron` plugin:

| Checkpoint | Architecture | Layers (DeltaNet / attention) | MLP |
| --- | --- | --- | --- |
| Qwen3.5-2B | `Qwen3_5ForConditionalGeneration` | 24 (18 / 6) | dense |
| Qwen3.5-27B | `Qwen3_5ForConditionalGeneration` | 64 (48 / 16) | dense |
| Qwen3.5-35B-A3B | `Qwen3_5MoeForConditionalGeneration` | 40 (30 / 10) | 256 experts, top-8, gated shared expert |
| Qwen3.5-397B-A17B | `Qwen3_5MoeForConditionalGeneration` | 60 (45 / 15) | 512 experts, top-10, gated shared expert |

Qwen3.5 is a **hybrid** decoder. Most layers are gated DeltaNet, a linear
recurrence that keeps a fixed-size state per sequence. The other layers are
full attention with a paged KV cache. vLLM puts the two kinds of layer in
separate KV cache groups (one attention group and `MambaSpec` state groups) that
share one block pool.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

## Requirements

**This model needs the hybrid KV cache support in `vllm-neuron`**, the framework
change that comes with this model ("Support hybrid (attention + recurrent-state)
models on the V1 runner"), plus the two platform / runner fixes it is stacked on
(text-only vision bucket skip, block-table alignment for non-power-of-2 blocks).
The hybrid change adds
`RecurrentLayerSpec` / `KVSpec.recurrent_layers`, the `MambaSpec` allocation
with page-major views in `NeuronModelRunner`, and the hybrid page-size alignment
in `NeuronPlatform`. Without it the model fails at `bind_kv_cache`.

Hybrid models run on the V1 model runner. `model_runner_v2_policy` already routes
them there; do not set `VLLM_USE_V2_MODEL_RUNNER=1`.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level `README.md`).
Then register the two architectures with vLLM (see
[contrib/README.md](../../README.md), "How do I register a custom model with
vLLM?"). The registered classes are `src/factory.py`.

### Offline inference with the Python API (Option 1)

Register the factories from your script, before you construct `LLM(...)`:

```python
from vllm import LLM, ModelRegistry, SamplingParams

# The package is this folder's src/, copied into your application or installed
# with `pip install -e contrib/models/qwen3_5`.
from qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)

ModelRegistry.register_model(
    "Qwen3_5ForConditionalGeneration", Qwen3_5ForConditionalGeneration
)
ModelRegistry.register_model(
    "Qwen3_5MoeForConditionalGeneration", Qwen3_5MoeForConditionalGeneration
)

llm = LLM(
    model="Qwen/Qwen3.5-2B",
    tensor_parallel_size=4,
    max_model_len=1024,
    max_num_batched_tokens=1024,
    max_num_seqs=4,
    enable_prefix_caching=False,
    limit_mm_per_prompt={"image": 0, "video": 0},
    additional_config={
        "neuron_config": {
            "num_batched_tokens_buckets": [1024],
            "num_seqs_buckets": [4],
            "on_device_sampling_config": {"all_greedy": True},
        }
    },
)
outputs = llm.generate(["The color of the sky is"], SamplingParams(max_tokens=32))
print(outputs[0].outputs[0].text)
```

`scripts/run.py` does the same for all four checkpoints, including the
expert-parallel and head-replication settings that 397B-A17B needs:

```bash
python contrib/models/qwen3_5/scripts/run.py --model <path>/Qwen3.5-2B
python contrib/models/qwen3_5/scripts/run.py --model <path>/Qwen3.5-27B \
    --gpu-memory-utilization 0.65
python contrib/models/qwen3_5/scripts/run.py --model <path>/Qwen3.5-35B-A3B \
    --gpu-memory-utilization 0.72
python contrib/models/qwen3_5/scripts/run.py --model <path>/Qwen3.5-397B-A17B \
    --tensor-parallel-size 64 --expert-parallel --ep-degree 8 \
    --gpu-memory-utilization 0.7
```

The registration in your script reaches the engine and worker processes when
vLLM forks them. If they are spawned, use the package install below.

### Serve with the OpenAI-compatible server

`vllm serve` runs no script of yours, so install this folder as a package. Its
`vllm.general_plugins` entry point (`src/plugin.py`) registers both
architectures in every vLLM process, with no change to the plugin's core code:

```bash
pip install -e contrib/models/qwen3_5

vllm serve Qwen/Qwen3.5-2B \
    --tensor-parallel-size 4 \
    --max-model-len 1024 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [1024], "num_seqs_buckets": [4]}}'
```

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen3.5-2B", "prompt": "The color of the sky is", "max_tokens": 32}'
```

### Settings this model needs

- **`limit_mm_per_prompt={"image": 0, "video": 0}`.** The checkpoints carry a
  `vision_config`, but this port is text-only; with every limit at 0 the
  platform skips vision bucket resolution (core branch), which otherwise fails
  for `max_model_len` below the encoder block size (2048).
- **`enable_prefix_caching=False`.** Prefix caching over recurrent state needs
  per-block state copies, which are not implemented. The runner raises if
  `mamba_cache_mode` is not `"none"`.
- **`max_num_batched_tokens` equal to `max_model_len`.** The model does not
  support chunked prefill: each prefill would start DeltaNet from a zero state
  and attend only to its own tokens, which gives wrong output without an error.
  The 0.29 runner turns segmented prefill on whenever `max_num_batched_tokens <
  max_model_len`, so the factory raises at load in that case.
- **No explicit `--optimization-level`.** The model's `compiler_args_for`
  compiles the dense decode graph at `-O3` and without
  `--modular-flow-mac-threshold=10`. On vLLM-neuron 0.24 the dense decode graph
  failed codegen with that flag (`NCC_IBTN006`), and 27B's decode graph failed
  ISA validation at `-O1` (`NCC_IINAR001`). The runner calls this policy only
  when you do not set the level. The sparse checkpoints keep the base
  arguments: their MoE block calls NKI kernels, which is what the threshold is
  for.
- **`--gpu-memory-utilization` below the default for the large checkpoints.**
  HBM on Trn2 is 24 GB per logical NeuronCore. The block planner fills the
  whole KV budget, which for a hybrid model means recurrent-state blocks far
  past `max_num_seqs` (`mamba_block_size == max_model_len`, so each sequence
  uses one state block). Too many blocks fail compilation with `NCC_EVRF009`
  (tensors exceed the HBM limit). Earlier runs used 0.65 for 27B and 0.72 for
  35B-A3B at TP=4, and 0.7 for 397B-A17B at TP=64.
- **397B-A17B only:** `enable_expert_parallel=True` with
  `neuron_config.ep_degree=8` at TP=64, the head-count relaxation in
  `scripts/run.py` (`relax_head_divisibility`, because vLLM rejects 64 ranks
  for 32 query heads). Its top-10 routing is above the fused decode kernel's
  limit (`moe_block_tkg` calls `nkilib`'s `router_topk`, which asserts K <= 8),
  so decode routes in torch and runs the experts through `moe_tkg`
  (`_forward_decode_unfused` in `src/moe.py`); prefill's `NF.router` already
  falls back to torch above 8. No patched `nkilib` is needed.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vllm-neuron 0.29) | 2.32 and earlier (vllm-neuron 0.24) |
| ---------------------- | ----------------------- | ----------------------------------- |
| Trn3                   | Not tested              | Not tested                          |
| Trn2                   | Working (2B, 27B, 35B-A3B, 397B-A17B; see below) | Working (see below) |

On SDK 2.33 (trn2.48xlarge, TP=4, with the core branch): Qwen3.5-2B passes the
logit-validation integration test (`test/integration/test_model.py`, batch 1 and 4,
`max_model_len` 1024, every token within tolerance); Qwen3.5-27B and 35B-A3B give
coherent greedy output from `scripts/run.py`; Qwen3.5-397B-A17B (TP=64, EP=8,
stock `nkilib`) gives coherent greedy output from `scripts/run.py`.

### Measured on SDK 2.33

Setup:

- trn2.48xlarge, bf16, greedy.
- 1024-token input, 128-token output, `max_model_len` 2048.
- `scripts/benchmark_latency.py`: streamed TTFT, then TPOT over the remaining
  tokens; 3 rounds after a warmup round.
- Batch = `--max-num-seqs` = the number of concurrent requests.
- Each run had the instance to itself. The TP=4 checkpoints used one Neuron
  device (`NEURON_VISIBLE_DEVICES=0-3`); running three of them side by side
  added ~20% to 2B's batch-1 TPOT.

| Checkpoint | Parallelism | Batch | TTFT mean | TPOT mean | Output tok/s (aggregate) |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-2B | TP=4 | 1 | 49.3 ms | 7.01 ms | 136.3 |
| Qwen3.5-2B | TP=4 | 4 | 125.6 ms | 14.53 ms | 259.2 |
| Qwen3.5-2B | TP=4 | 8 | 201.2 ms | 25.60 ms | 296.3 |
| Qwen3.5-27B | TP=4 | 1 | 298.0 ms | 53.60 ms | 18.0 |
| Qwen3.5-27B | TP=4 | 4 | 821.0 ms | 173.32 ms | 22.4 |
| Qwen3.5-27B | TP=4 | 8 | 1348.1 ms | 161.67 ms | 46.8 |
| Qwen3.5-35B-A3B | TP=4 | 1 | 232.9 ms | 34.54 ms | 27.7 |
| Qwen3.5-35B-A3B | TP=4 | 4 | 599.2 ms | 77.40 ms | 49.0 |
| Qwen3.5-35B-A3B | TP=4 | 8 | 1058.3 ms | 143.79 ms | 53.0 |
| Qwen3.5-397B-A17B | TP=64, EP=8 | 1 | 225.8 ms | 51.77 ms | 18.8 |
| Qwen3.5-397B-A17B | TP=64, EP=8 | 4 | 563.0 ms | 59.50 ms | 63.0 |
| Qwen3.5-397B-A17B | TP=64, EP=8 | 8 | 959.8 ms | 69.07 ms | 105.2 |

397B used `--prefill-buckets 1024,2048`; the others used the plugin's default
prefill buckets.

Against the 0.24 port (below):

- TTFT is 2.6-3x lower on 2B and ~30% lower on 35B-A3B.
- 35B-A3B decode is unchanged.
- 2B decode is ~1.8x slower (7.0 vs 3.9 ms at batch 1). The decode graph's
  optimization level is not the cause: `QWEN3_5_DENSE_DECODE_OPT_LEVEL=1`
  measures the same 7.0 ms as the default 3. Not yet explained.
- 27B's batch-4 TPOT is above its batch-8 TPOT; reproduced in isolation.

The results below are from the
earlier in-tree port on vllm-neuron 0.24.0.1.1.0, measured on trn2.3xlarge at
TP=4, bf16, greedy, 1024-token input and 128-token output, `max_model_len`
2048:

| Checkpoint | Batch | TTFT mean | TPOT | Output tok/s (aggregate) |
| --- | --- | --- | --- | --- |
| Qwen3.5-2B | 1 | 145.3 ms | 3.88 ms | 200.5 |
| Qwen3.5-2B | 4 | 353.5 ms | 7.62 ms | 385.5 |
| Qwen3.5-2B | 8 | 624.3 ms | 12.35 ms | 464.8 |
| Qwen3.5-35B-A3B | 1 | 328.2 ms | 36.17 ms | 26.0 |
| Qwen3.5-35B-A3B | 4 | 860.4 ms | 77.15 ms | 47.9 |
| Qwen3.5-35B-A3B | 8 | 1544.4 ms | 142.69 ms | 52.0 |

- 27B at batch 1 on a 1024-token prompt, TP=4, 1024 prefill bucket: TTFT
  284 ms, TPOT 48.5 ms.
- 35B-A3B accuracy: 137/160 greedy tokens equal to HF bf16 (85.6%), no
  first-token mismatch.
- 397B-A17B: functional validation only (correct greedy output) on a
  trn2.48xlarge at `world_size=64, ep_degree=8`, 11.93 GiB used per rank, with a
  patched `nkilib` for top-k > 8 (no longer needed, see above).

## Example Checkpoints

- <https://huggingface.co/Qwen/Qwen3.5-2B>
- <https://huggingface.co/Qwen/Qwen3.5-27B>
- <https://huggingface.co/Qwen/Qwen3.5-35B-A3B>
- <https://huggingface.co/Qwen/Qwen3.5-397B-A17B>

## Testing

The integration test needs the hybrid KV cache core change (see
"Requirements") and Neuron hardware. It registers the architectures itself,
serves the model through vLLM and validates its logits against HuggingFace.
The defaults (Qwen3.5-2B, TP=4) fit a trn2.3xlarge.

```bash
pytest contrib/models/qwen3_5/test/integration/test_model.py \
    -v -s --timeout=3600
```

Set `QWEN35_MODEL_PATH` (a local path or a Hub id) and `QWEN35_TP` to test
another checkpoint or tensor-parallel size.

### Optional: unit and module tests

There are no unit or module tests yet. Add them under `test/unit/` and
`test/module/` and run them in CPU mode:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/qwen3_5/test/unit/
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/qwen3_5/test/module/ -v -s --timeout=600
```

## Known limitations

- **Text-only.** The vision tower is not ported. The registered classes do not
  implement `SupportsMultiModal`, so vLLM builds no multimodal processor and
  the checkpoint's `vision_config` is ignored. The earlier in-tree port reused
  the Qwen3-VL encoder from `vllm_neuron/model/qwen3_vl`, which a contrib model
  must not import.
- **No chunked prefill, no prefix caching, no speculative decoding.** The MTP
  head in the checkpoint is ignored.
- **One batch bucket per process.** With several `num_seqs_buckets` warmup hung
  on vllm-neuron 0.24.
- **Decode gathers the full block-table width** rather than only the occupied
  pages. This is the largest known performance item on the dense path.
- **`head_dim = 256`** (27B, 35B-A3B, 397B-A17B) exceeds the flash-attention
  kernel's `MAX_HEAD_DIM = 128`, so attention runs in torch. Measured at 6-8% of
  prefill.
- **Sparse decode is gather-bound.** 35B-A3B decodes about 6x slower than its
  active-parameter count predicts.
- **Expert parallelism compiles one graph per EP partition** (`ep_rank` is a
  trace constant) and cannot span data-parallel replicas.

## Source layout

```text
src/
  __init__.py      exports the two factories and the configs
  factory.py       the registered classes (and vLLM's hybrid-model contract)
  plugin.py        vllm.general_plugins entry point for `vllm serve`
  config.py        HF config parsing (dense and MoE)
  model.py         decoder, attention, KV/state cache binding, weights, compiler args
  deltanet.py      gated DeltaNet mixer (prefill chunked delta rule, decode step)
  moe.py           sparse MoE block (NKI moe_cte / moe_block_tkg kernels)
  flags.py         debug and experiment environment switches
  kernels/         opt-in NKI kernels for the DeltaNet prefill
scripts/run.py     offline text generation for all four checkpoints
test/integration/test_model.py
```

## Attribution

This work derives from the Qwen3.5 port in
[`qingzwang/vllm-neuron`](https://github.com/qingzwang/vllm-neuron), branch
`model/Qwen3.5-2B` (Apache-2.0), which is where the 2B implementation
originates. `kernels/nki_deltanet.py`, `kernels/nki_deltanet_fused.py` and
`flags.py` are that fork's files unchanged, and the bulk of `model.py`,
`deltanet.py` and `config.py` is its work as well. The 27B, 35B-A3B and
397B-A17B support and `moe.py` were added in the in-tree port for
vllm-neuron 0.24, and this folder ports that code to vllm-neuron 0.29.
