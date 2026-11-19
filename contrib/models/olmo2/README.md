# OLMo2 (Text)

Serve an OLMo 2 text model (`allenai/OLMo-2-0425-1B`) with vLLM on AWS
Trainium through the `vllm-neuron` plugin. The implementation is in `src/`,
and `src/README.md` describes the architecture and the per-feature status.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix and the
> validation report below record exactly what was tested.

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

| Instance / version   | vLLM Neuron 0.29.0.1.2.0 (Neuron SDK 2.33) | Other versions |
| -------------------- | ------------------------------------------ | -------------- |
| Trn2 (trn2.48xlarge) | Working, TP 8                              | Not tested     |
| Trn3                 | Not tested                                 | Not tested     |

The earlier runs on `vllm-neuron` 0.31.0.1.2.0 in the validation report used
the port before the 0.29 changes, so that version is not listed as tested for
the code in this folder.

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

### Unit tests

`test/unit/test_loading.py` runs the registration contract, the config parse,
the checkpoint mapping coverage, the weight-loader transforms and a full
8-rank CPU load. It needs no Neuron hardware:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/olmo2/test/unit/ \
    -v -s --timeout=3600
```

### Module tests

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

One `trn2.48xlarge` (logical NeuronCore config 2), in the Neuron Deep Learning
Container
`public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04`
(Neuron SDK 2.33, Python 3.14.6), with the code in this folder at TP 8 on one
node:

| Package | Version |
|---|---|
| vLLM | 0.29.0 |
| `vllm-neuron` | 0.29.0.1.2.0 |
| torch | 2.13.0 |
| transformers | 5.17.0 |
| neuronx-cc | 2.28.7631.0+8f7b0cf5 |
| nki | 0.7.0+37713544100.gfede04bb |
| libtorch-neuronx-lite | 2.13.0.1.0.3112+653bc48f |

The run downloaded `allenai/OLMo-2-0425-1B` at the pinned revision
`a1847dff35000b4271fa70afc5db10fd29fedbdf` before the tests; the integration
test and the serve smoke test load the model by its hub name. The unit and
integration tests ran first, and the folder was then installed with
`pip install -e` for the serve smoke test.

#### Results

| Test | Configuration | Result |
|---|---|---|
| CPU unit tests, `test/unit/` | `VLLM_NEURON_CPU_MODE=1` | 4 passed |
| Integration logit test, `test_model_accuracy[1-2048]` | batch 1, max model length 2,048, TP 8 | PASSED |
| Integration logit test, `test_model_accuracy[4-2048]` | batch 4, max model length 2,048, TP 8 | PASSED |
| `vllm serve` smoke | the serve command under Usage | Passed, 3 of 3 prompts |

The four unit tests are `test_registry_and_config`,
`test_checkpoint_mapping_coverage`, `test_weight_loader_transforms` and
`test_full_model_tp8_load[8]`. The module tests (`test/module/`) were not run
on this version.

The integration test validates 16 generated tokens for each of its two prompts
against the HuggingFace BF16 reference, with the FP32 reference as the
baseline. The logged thresholds are a divergence bound of 1 ULP (dynamic, 7
mantissa bits) and top-K tolerances (rtol) of 0.03 (K5), 0.05 (K50), 0.08
(K1000) and 0.09 (All). Per prompt, as logged:

| Case | Prompt | Overall Status | Max divergence (1 ULP) | Max K5 (0.03) | Max K50 (0.05) | Max K1000 (0.08) | Max All (0.09) |
|---|---|---|---|---|---|---|---|
| `[1-2048]` | 1/2, `The color of the sky is` | FAILED | 0.0000 | 0.0318 (over) | 0.0263 | 0.0198 | 0.0127 |
| `[1-2048]` | 2/2, `A short summary of the history of the printing press:` | PASSED | 0.0000 | 0.0253 | 0.0190 | 0.0187 | 0.0177 |
| `[4-2048]` | 1/2, `The color of the sky is` | FAILED | 0.0000 | 0.0318 (over) | 0.0263 | 0.0198 | 0.0126 |
| `[4-2048]` | 2/2, `A short summary of the history of the printing press:` | PASSED | 0.0000 | 0.0253 | 0.0190 | 0.0187 | 0.0177 |

Prompt 1 exceeds the K5 tolerance at one generated token (token 10, K5 0.0318
against 0.03; at batch 4 the same value on all four rows), with no argmax
divergence, so both cases log `Two-way validation: 1/2 prompts passed`. The
test asserts `result.passed` from `multi_prompt_logit_validation`. With an
FP32 baseline (three-way mode), that result also passes when the cross-prompt
σ-ratio is at most 1.0. The log reports `agg_sigma_ratio: 0.9808 (≤1.0 =
pass)` at batch 1 and `agg_sigma_ratio: 0.9744 (≤1.0 = pass)` at batch 4, and
pytest reports `2 passed`.

The serve smoke test started `vllm serve` with the command under Usage and
sent each prompt to `/v1/completions` with `max_tokens` 32 and `temperature`
0. The check is that every completion is non-empty. Each returned 32 tokens:

| Prompt | Start of the completion (first 60 characters) |
|---|---|
| `The capital of France is` | ` Paris. The French language is spoken in France. The French ` |
| `1, 2, 3, 4, 5,` | ` 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16` |
| `def fibonacci(n):` | ` \n    if n < 0: \n        return 0 \n    elif n == 0: \n       ` |

(`\n` is a newline.) One batched request with all three prompts and
`max_tokens` 16 returned a prefix of the same text for each prompt.

#### Changes for vLLM Neuron 0.29

From the earlier port of this folder, which was written against a later
plugin release:

- `src/compat.py` (new) vendors three plugin helpers that 0.29.0.1.2.0 does
  not ship: `hoist_rope_parameters` (0.29 has no
  `vllm_neuron.model.config_utils`) and `build_per_head_block_table` and
  `scatter_new_kv` (not in 0.29's `vllm_neuron.model.kv_cache`).
  `src/config.py` and `src/model.py` import them from `.compat` instead of
  from the plugin.
- `pyproject.toml` and `src/plugin.py` (new) install `src/` as the
  `olmo2_neuron` package with a `vllm.general_plugins` entry point that
  registers `Olmo2ForCausalLM`.
- `test/conftest.py` adds a session fixture that shuts down the plugin's
  `MPExecutor` worker pool (`vllm_neuron.utils.executor._worker_pool`) at the
  end of the session, so pytest exits after its summary.

The model code, the weight loaders and the test assertions are otherwise
unchanged.

### Model configuration

The architecture class is `Olmo2ForCausalLM` (`src/factory.py` validates the
config and selects `src/model.py`). It is a dense model, so there is no expert
parallelism; every recorded run used tensor parallelism only, at TP 8. The
implementation computes in BF16 from the FP32 checkpoint, which is why
`neuron_config.quantization` is `bf16` in every command above.

The tested serving configuration is a maximum model length of 2048 with a
matching 2048-token batched-token limit, `--max-num-seqs` 1 or 4, block size
32, prefix caching off and async scheduling off.

### Earlier runs: `vllm-neuron` 0.31.0.1.2.0 (vLLM 0.31.0)

These runs used the port before the 0.29 changes above, on `trn2.48xlarge` at
TP 8 with the `neuron_native` runtime backend. Only the verdicts and case
counts were recorded, and the runs used more than one container image without
a record of which image served which row.

| Run | Nodes | Result |
|---|---|---|
| CPU loading and components | 1 | PASS, 5 of 5 |
| Device functional | 2 (4 ranks each) | PASS, 12 of 12 |

The CPU run covers the loading path (now `test/unit/test_loading.py`) and the
one component case with a CPU path. The device run covers the four layer-0
attention and MLP cases of `test/module/test_components.py` on each node, plus
the port's serving cases, which include a 1,024-token prompt with 512
generated tokens. Those compositions are read from the test sources, not from
a per-case record.

### Exceptions

- **Prompt 1 is over the two-way K5 tolerance.** On 0.29 its max K5 error is
  0.0318 against 0.03 at one token, at batch 1 and batch 4. The integration
  test passes on the three-way σ-ratio, as described above.
- **Module tests not run on 0.29.** `test/module/test_components.py` last
  passed in the 0.31.0.1.2.0 device run above.
- **No accuracy-benchmark score and no performance figures.** The serve smoke
  test checks only that each completion is non-empty.
- **No 8,192 / 1,024 serving run.** Every run used a 2,048-token maximum
  model length.
- **Dropped from the port:** its GSM8K accuracy test, its performance test,
  its 2-node serving test and its multi-node launcher. They depend on shared
  test helpers outside this folder. The serving test's logit-validation shape
  is what `test/integration/test_model.py` carries now, with locally defined
  prompts. The module tests default to one node with all 8 ranks, so the
  2-node topology above cannot be reproduced from this folder alone.
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
