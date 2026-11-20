# OLMo2 (Text)

Serve an OLMo 2 text model (`allenai/OLMo-2-0425-1B`) with vLLM on AWS
Trainium through the `vllm-neuron` plugin. The implementation is in `src/`,
and `src/README.md` describes the architecture and the per-feature status.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`), then install this folder:

```bash
pip install -e contrib/models/olmo2
```

The install makes `src/` importable as the `olmo2_neuron` package and adds a
`vllm.general_plugins` entry point (`src/plugin.py`) that registers
`Olmo2ForCausalLM` in every vLLM process. `vllm serve` and `LLM(...)` then
find the architecture with no other setup.

### Serve with the OpenAI-compatible server

These are the flags of the tested serve run. `--host 127.0.0.1` keeps the
server local to the host; it listens on port 8000.

```bash
vllm serve allenai/OLMo-2-0425-1B \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --block-size 32 \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{"neuron_config": {"quantization": "bf16", "on_device_sampling_config": {"all_greedy": true}}}'
```

Then send a request:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "allenai/OLMo-2-0425-1B",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="allenai/OLMo-2-0425-1B",
    tensor_parallel_size=8,
    dtype="bfloat16",
    max_model_len=2048,
    max_num_batched_tokens=2048,
    max_num_seqs=4,
    block_size=32,
    enable_prefix_caching=False,
    additional_config={
        "neuron_config": {
            "quantization": "bf16",
            "on_device_sampling_config": {"all_greedy": True},
        }
    },
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The color of the sky is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

`quantization: "bf16"` is not optional for this model: the published
checkpoint stores FP32 weights and the implementation computes in BF16.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1`. See the top-level `README.md` and `AGENTS.md` for the
full list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn3 | Not tested | Not tested |
| Trn2 | Working (TP 8) | Not tested |

## Example Checkpoints

- <https://huggingface.co/allenai/OLMo-2-0425-1B> — the only checkpoint with
  recorded runs. The unit tests pin revision
  `a1847dff35000b4271fa70afc5db10fd29fedbdf` (set `OLMO2_CHECKPOINT_PATH` to a
  local copy to skip the download); the integration test loads the model by
  its hub name.

Other OLMo 2 checkpoints have the same architecture, but none was run, and the
implementation rejects a checkpoint with tied embeddings, an attention bias or
`rope_scaling`.

## Testing

The integration test serves the model through vLLM on Neuron and validates its
logits against a HuggingFace reference. It needs 8 NeuronCores:

```bash
pytest contrib/models/olmo2/test/integration/test_model.py \
    -v -s --timeout=7200
```

The integration test does not need the `pip install`: it registers the factory
itself, from the test process, before it builds the engine. vLLM's engine-core
and worker processes inherit that registration through the default `fork`
start method, so do not set `VLLM_WORKER_MULTIPROC_METHOD=spawn` for this test.

### Optional: unit tests

`test/unit/test_loading.py` runs the registration contract, the config parse,
the checkpoint mapping coverage, the weight-loader transforms and a full
8-rank CPU load. It needs no Neuron hardware:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/olmo2/test/unit/ \
    -v -s --timeout=3600
```

### Optional: module tests

`test/module/test_components.py` compares layer-0 attention and MLP with a
HuggingFace reference (the three-way pattern in the
[module test guidelines](../../../docs/design/accuracy/module_test_guidelines.md)),
in prefill and in decode. All four cases need 8 NeuronCores; only
`test_attention_prefill` also has a CPU path.

```bash
pytest contrib/models/olmo2/test/module/ -v -s --timeout=3600
```

All three test directories reach the implementation through
`test/_local_src.py`, which loads `src/` by file location under the unique
module name `contrib_olmo2_src`. Nothing in the tests needs the repository root
on `sys.path`, so the folder keeps working when it is copied into an
application.

`test/unit` and `test/module` also share `test/olmo2_test_utils.py`, the port's
own helper, vendored into this folder with the model. It sits at the `test/`
root rather than inside one of them because both import it. `test/conftest.py`
puts the `test/` directory on `sys.path` so that the three test directories
find those two modules. It also provides the `dist_cleanup` fixture that the
plugin's test tree would otherwise supply, and a session fixture that shuts
down the plugin's `MPExecutor` worker pool when the session ends; without it
pytest prints its summary and then does not exit.

## Validation report

### vLLM Neuron 0.29.0.1.2.0 (2026-11-19)

#### Environment

| Image | Instance | TP | vLLM | vllm-neuron | torch | transformers | neuronx-cc |
|---|---|---|---|---|---|---|---|
| `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` | `trn2.48xlarge` (LNC 2), 1 node | 8 | 0.29.0 | 0.29.0.1.2.0 | 2.13.0 | 5.17.0 | 2.28.7631.0+8f7b0cf5 |

#### Results

| Test | Configuration | Result |
|---|---|---|
| CPU unit tests, `test/unit/` | `VLLM_NEURON_CPU_MODE=1` | 4 passed |
| Integration logit test, `test_model_accuracy[1-2048]` | batch 1, max model length 2,048, TP 8 | PASSED |
| Integration logit test, `test_model_accuracy[4-2048]` | batch 4, max model length 2,048, TP 8 | PASSED |
| `vllm serve` smoke | the serve command under Usage | Passed, 3 of 3 prompts |

Not run on 0.29: module tests (`test/module/`), not part of the 0.29 run.

#### Logit validation

| Case | Prompt | Overall Status | Max divergence (1 ULP) | Max K5 (0.03) | Max K50 (0.05) | Max K1000 (0.08) | Max All (0.09) |
|---|---|---|---|---|---|---|---|
| `[1-2048]` | 1/2, `The color of the sky is` | FAILED | 0.0000 | 0.0318 (> 0.03) | 0.0263 | 0.0198 | 0.0127 |
| `[1-2048]` | 2/2, `A short summary of the history of the printing press:` | PASSED | 0.0000 | 0.0253 | 0.0190 | 0.0187 | 0.0177 |
| `[4-2048]` | 1/2, `The color of the sky is` | FAILED | 0.0000 | 0.0318 (> 0.03) | 0.0263 | 0.0198 | 0.0126 |
| `[4-2048]` | 2/2, `A short summary of the history of the printing press:` | PASSED | 0.0000 | 0.0253 | 0.0190 | 0.0187 | 0.0177 |

Prompt 1: K5 0.0318 > 0.03 at generated token 10 (batch 4: all four rows); test passed on aggregate σ-ratio 0.9808 (`[1-2048]`) and 0.9744 (`[4-2048]`) ≤ 1.0 (`multi_prompt_logit_validation`).
16 generated tokens per prompt vs the HF BF16 reference (FP32 baseline); no argmax divergence.

#### Serve smoke

`/v1/completions`, max_tokens 32, temperature 0; 32 tokens returned per prompt. Batched request (all three, max_tokens 16): a prefix of the same text.

| Prompt | Start of the completion (first 60 characters) |
|---|---|
| `The capital of France is` | ` Paris. The French language is spoken in France. The French ` |
| `1, 2, 3, 4, 5,` | ` 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16` |
| `def fibonacci(n):` | ` \n    if n < 0: \n        return 0 \n    elif n == 0: \n       ` |

#### Changes for 0.29

- `src/compat.py` (new): vendors `hoist_rope_parameters` (0.29 has no `vllm_neuron.model.config_utils`), `build_per_head_block_table` and `scatter_new_kv` (not in 0.29's `vllm_neuron.model.kv_cache`).
- `src/config.py`, `src/model.py`: import those helpers from `.compat` instead of from the plugin.
- `pyproject.toml`, `src/plugin.py` (new): install `src/` as the `olmo2_neuron` package with a `vllm.general_plugins` entry point that registers `Olmo2ForCausalLM`.
- `test/conftest.py`: session fixture that shuts down the plugin's `MPExecutor` worker pool (`vllm_neuron.utils.executor._worker_pool`) at session end.
- Model code, weight loaders and test assertions: otherwise unchanged.

### Model configuration

The architecture class is `Olmo2ForCausalLM` (`src/factory.py` validates the
config and selects `src/model.py`). It is a dense model, so there is no expert
parallelism; every recorded run used tensor parallelism only, at TP 8. The
implementation computes in BF16 from the FP32 checkpoint, which is why
`neuron_config.quantization` is `bf16` in every command above.

The tested serving configuration is a maximum model length of 2048 with a
matching 2048-token batched-token limit, `--max-num-seqs` 1 or 4, block size
32, prefix caching off and async scheduling off.

### Exceptions

- **Prompt 1 is over the two-way K5 tolerance.**
- **Module tests not run on 0.29.**
- **No accuracy-benchmark score and no performance figures.** The serve smoke
  test checks only that each completion is non-empty.
- **No 8,192 / 1,024 serving run.** Every run used a 2,048-token maximum
  model length.
- **Dropped from the port:** its GSM8K accuracy test, its performance test,
  its 2-node serving test and its multi-node launcher. They depend on shared
  test helpers outside this folder. The serving test's logit-validation shape
  is what `test/integration/test_model.py` carries now, with locally defined
  prompts. The module tests default to one node with all 8 ranks, so the
  2-node topology cannot be reproduced from this folder alone.
- **Rewritten assertion.** The port's CPU test asserted that
  `vllm_neuron.model.registry.get_models()` maps `Olmo2ForCausalLM` to the
  factory. A contrib model is deliberately absent from that registry, so the
  shipped test checks the registration contract instead (the checkpoint's
  architecture string and the factory that `src/__init__.py` exports).
- **Rewritten checkpoint resolution.** The port resolved the checkpoint
  through the plugin's shared test utilities. The vendored helper in
  `test/olmo2_test_utils.py` downloads the pinned revision from the
  HuggingFace hub instead, or uses `OLMO2_CHECKPOINT_PATH`.
- **Untested configurations.** Prefix caching and the segmented-prefill path
  (every serving run set `--no-enable-prefix-caching`), tensor parallel sizes
  other than 8, context beyond 2,048 tokens, speculative decoding, Trn3, and
  any `vllm-neuron` version other than 0.29.0.1.2.0 for the code in this
  folder.
