# HRM-Text (HRM-Text-1B)

This folder serves the HRM-Text (Hierarchical Reasoning Model, text) model with
vLLM on AWS Trainium through the `vllm-neuron` plugin. The implementation under
`src/` covers the BF16 checkpoint of `sapientinc/HRM-Text-1B`: two stacks (H and
L) of 16 decoder layers each, run as a recurrence (`H_cycles = 2`,
`L_cycles = 3`) over the same token embeddings, with a sigmoid output gate on
the attention and 128 KV cache slots because each pass of a layer writes its
own slot. It is a text-only causal LM and uses standard text generation.
`src/README.md` is the port's own architecture and feature card.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`), then install this folder:

```bash
pip install -e contrib/models/hrm_text
```

The install adds this folder's `src/` as the `hrm_text_neuron` package and a
`vllm.general_plugins` entry point (`hrm_text_neuron.plugin:register`, Option 3
in [contrib/README.md](../../README.md)). vLLM runs that entry point in every
process it starts (the front end, the engine core and each Neuron worker), so
`HrmTextForCausalLM` is registered everywhere and `vllm serve` and `LLM(...)`
need no other setup. `pip uninstall vllm-neuron-contrib-hrm-text` removes the
registration.

The port supports tensor parallelism 1 and 2 only: HRM-Text has 12 attention
heads, and a larger degree leaves an odd number of heads on a rank, which the
decode kernel cannot take.

### Serve with the OpenAI-compatible server

These are the flags of the tested serving run:

```bash
vllm serve sapientinc/HRM-Text-1B \
    --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 2048 --max-num-batched-tokens 1024 --max-num-seqs 1 \
    --no-enable-prefix-caching \
    --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [1024], "num_seqs_buckets": [1]}}'
```

Then send a request:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "sapientinc/HRM-Text-1B",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

The installed plugin registers the architecture, so no `register_model` call is
needed. The engine arguments below are those of the serve command above:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="sapientinc/HRM-Text-1B",
    tensor_parallel_size=2,
    dtype="bfloat16",
    max_model_len=2048,
    max_num_batched_tokens=1024,
    max_num_seqs=1,
    enable_prefix_caching=False,
    additional_config={
        "neuron_config": {
            "num_batched_tokens_buckets": [1024],
            "num_seqs_buckets": [1],
        }
    },
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
for output in llm.generate(["The color of the sky is"], sampling_params):
    print(output.outputs[0].text)
```

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1` in the environment. See the top-level `README.md` and
`AGENTS.md` for the full list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

Everything outside the tested cell is untested, not known-broken.

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn3 | Not tested | Not tested |
| Trn2 | Working (TP 2) | Not tested |

## Example Checkpoints

- <https://huggingface.co/sapientinc/HRM-Text-1B> — the only checkpoint tested,
  at revision `22097cbcecdd1301afe30a19a3ee61b96a9863e5`, which the tests pin.

Other HRM-Text checkpoints are untested. The factory accepts bfloat16 only, and
rejects attention or MLP bias and any RoPE type other than the default.

## Testing

The tests load this folder's `src/` by file location. `test/_local_src.py`
loads it under a module name unique to this model, and each test module loads
that helper by location too, so nothing goes on `sys.path`, no test needs the
repository root and no test needs the `pip install` above. Two contrib models
collected in one pytest run cannot resolve to each other's code.
`test/conftest.py` shuts down the plugin's `MPExecutor` worker pool when the
session ends; without it pytest prints its summary and then does not exit.

The integration test builds one offline engine at TP 2 and validates its logits
against the HuggingFace reference. It needs 2 Neuron cores reachable from the
test process. It registers the architecture from `src/` itself, so its engine
processes get the registration by fork, and it skips when
`VLLM_WORKER_MULTIPROC_METHOD` is set to anything other than `fork`:

```bash
pytest contrib/models/hrm_text/test/integration/test_model.py \
    -v -s --timeout=10800
```

### Optional: unit tests

`test/unit/test_hrm_text_cpu.py` checks the registration, the config parsing and
the KV slot plan, the load of every checkpoint tensor into the full model at
TP 2 against an independent split of the raw tensors, and one decoder layer with
sequence-parallel prefill against independent reference math:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/hrm_text/test/unit/test_hrm_text_cpu.py -v
```

Set `HRM_TEXT_MODEL_PATH` to a local copy of the checkpoint to skip the
download.

### Optional: module tests

`test/module/` holds the device component tests and `common.py`, the helper
module that the unit, module and integration tests share (the reference math,
the MPExecutor worker harnesses and the NKI dispatch recorder).

`test_hrm_text_modules_multinode.py` runs TP 2 with one rank on each of 2
`trn2.48xlarge` nodes, so every collective crosses the nodes, and it asserts the
NKI kernel dispatch of each compiled graph. On each node, set `NODE_RANK` (0 or
1), `MASTER_ADDR` and `NEURON_RT_ROOT_COMM_ID` (the same value on both nodes),
then run:

```bash
pytest -v contrib/models/hrm_text/test/module/test_hrm_text_modules_multinode.py
```

See the [module test guidelines](../../../docs/design/accuracy/module_test_guidelines.md)
for the three-way accuracy pattern these tests follow.

## Validation report

### vLLM Neuron 0.29.0.1.2.0 (2026-11-18)

#### Environment

| Image | Instance | TP | vLLM | vllm-neuron | torch | transformers | neuronx-cc |
|---|---|---|---|---|---|---|---|
| `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` | `trn2.48xlarge` (LNC 2) | 2 | 0.29.0 | 0.29.0.1.2.0 | 2.13.0 | 5.17.0 | 2.28.7631.0+8f7b0cf5 |

#### Results

| Test | Configuration | Result |
|---|---|---|
| CPU unit tests (`test/unit/`) | `VLLM_NEURON_CPU_MODE=1` | 3 passed |
| `test_model_accuracy[1-256]` | batch 1, 256-token context, 2 prompts x 16 greedy tokens, TP 2 | PASSED (622.56 s) |
| `vllm serve` smoke | flags in Usage, TP 2 | Passed (3/3 completions non-empty) |

Not run on 0.29: module tests (`test/module/`), which need 2 `trn2.48xlarge` nodes; this run used one.

#### Logit validation

| Prompt | Overall Status | K5 (≤ 0.011) | K50 (≤ 0.02) | K1000 (≤ 0.03) | All (≤ 0.05) | Divergence (≤ 1 ULP) |
|---|---|---|---|---|---|---|
| 1 (Amazon rainforest) | FAILED | 0.0278 (> 0.011, token 1) | 0.0341 (> 0.02, token 1) | 0.0341 (> 0.03, token 1) | 0.0341 | 0.0000 |
| 2 (human genome) | FAILED | 0.0192 (> 0.011, token 3) | 0.0233 (> 0.02, token 2) | 0.0289 | 0.0717 (> 0.05, token 1) | 0.0000 |

Prompt 1 over K5/K50/K1000 at generated token 1; prompt 2 over K5 at tokens 1-3, K50 at 2, All at 1 (0-based, of 16).
No argmax divergence. Test passed on aggregate σ-ratio 0.7432 ≤ 1.0 (`multi_prompt_logit_validation`); `agg_bc` FAIL, `pp_static_0.03`/`0.05` 1/2.

#### Serve smoke

`/v1/completions`, max_tokens 32, temperature 0; one batched request of all 3 (max_tokens 16) returned prefixes of the same text.

| Prompt | Completion, first 60 chars |
|---|---|
| `The capital of France is` | ` the capital of France is the capital of France is the capit` |
| `1, 2, 3, 4, 5,` | ` 6, 7, 8, 9, 10, 11, 12, 13, 14,` |
| `def fibonacci(n):` | `::::::::::::::::::::::::::::::::` |

#### Changes for 0.29

- `src/compat.py` (new): vendors `hoist_rope_parameters`, `build_per_head_block_table`, `scatter_new_kv` from a later plugin release.
- `src/config.py`, `src/model.py`: import those helpers from `.compat` instead of the plugin; no other change.
- `pyproject.toml`, `src/plugin.py` (new): `vllm.general_plugins` entry point `hrm_text_neuron.plugin:register`.
- `test/conftest.py` (new): session fixture that shuts down the `MPExecutor` worker pool; no test logic changed.

### Model configuration

| Item | Value |
|---|---|
| Architecture class | `HrmTextForCausalLM` (the `architectures` field of the checkpoint) |
| Implementation | BF16; the factory accepts bfloat16 only |
| Parallelism | TP 2 on one node for the 0.29.0.1.2.0 run. The port supports TP 1 and TP 2 only (12 heads; the decode kernel needs an even head count per rank) |
| KV cache | 128 slots (`16 x H_cycles x (L_cycles + 1)`), because each pass of a layer writes its own slot. The component tests use a block size of 32 |
| Context length | 256 tokens for the logit tests; 2,048 with a 1,024-token prefill bucket for serving |
| Registration | `pip install -e contrib/models/hrm_text` (entry point `hrm_text_neuron.plugin:register`). Not in `vllm_neuron/model/registry.py` |
| Model-specific serving args | None beyond `--dtype bfloat16` and the bucket pinning shown in Usage. PrefixLM attention through `token_type_ids` is not implemented, and the model has no quantized path |

### Exceptions

- No dataset accuracy run (no GSM8K or similar) exists for this model. The
  logit comparison is the only accuracy gate.
- No throughput, latency or time-to-first-token figures exist.
- On 0.29.0.1.2.0 the only serving check is the smoke above. No 1,024 in /
  512 out run was made on 0.29.0.1.2.0, and no 8,192 in / 1,024 out run was
  made on any version.
- The module tests and multi-node TP were not run on 0.29.0.1.2.0.
- Only TP 2 was run. TP 1, which the code supports, is untested, and larger
  degrees are unsupported by the decode kernel.
- No evidence for Trn3, or for this folder's code on any version other than
  0.29.0.1.2.0.
- Not covered by any run: quantization (the port has no quantized path),
  PrefixLM attention through `token_type_ids` (not implemented), prefix
  caching, LoRA and speculative decoding.
