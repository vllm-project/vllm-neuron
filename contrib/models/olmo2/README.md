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
`README.md`). The plugin registers the Neuron platform automatically, but a
contrib architecture is not in the plugin's registry, so register
`Olmo2ForCausalLM` yourself first — see
[contrib/README.md](../../README.md), "How do I register a custom model with
vLLM?".

### Serve with the OpenAI-compatible server

`vllm serve` has no script of yours to run the registration call, so this path
needs option 2 in `contrib/README.md`: move `src/` to
`vllm_neuron/model/olmo2/` and add `("Olmo2ForCausalLM",
"vllm_neuron.model.olmo2:Olmo2ForCausalLM")` to `_MODELS` in
`vllm_neuron/model/registry.py`. As `contrib/README.md` notes, that option
edits the plugin's core code and so crosses the "community-owned, opt-in"
boundary; this PR does not make that edit. Then:

```bash
vllm serve allenai/OLMo-2-0425-1B \
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
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "allenai/OLMo-2-0425-1B",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

Your script runs before the engine, so it can register the architecture
itself. Copy `src/` into your application as an importable package, and then:

```python
from vllm import ModelRegistry

# Your copy of contrib/models/olmo2/src/, under whatever name you gave it.
from your_package.factory import Olmo2ForCausalLM

# Register before you construct LLM(...).
ModelRegistry.register_model("Olmo2ForCausalLM", Olmo2ForCausalLM)

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

The registration above runs in your process, and vLLM builds the model in its
engine-core and worker processes. Those processes inherit the registration
because the offline `LLM(...)` path starts them with the `fork` start method,
which is the default for a Python script. If you set
`VLLM_WORKER_MULTIPROC_METHOD=spawn`, the child re-imports from scratch, never
runs your `register_model` call, and `ModelRegistry.resolve_model_cls` fails to
find `Olmo2ForCausalLM`. Leave the start method alone, or use option 2 above.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1`. See the top-level `README.md` and `AGENTS.md` for the
full list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

The recorded runs name a plugin version and a vLLM version, not a Neuron SDK
release, so the columns below are the versions we have evidence for.

| Instance / version | `vllm-neuron` 0.31.0.1.2.0 (vLLM 0.31.0) | Other versions |
| ------------------ | ---------------------------------------- | -------------- |
| Trn2 (trn2.48xlarge) | Working, TP 8                          | Not tested     |
| Trn3                 | Not tested                             | Not tested     |

## Example Checkpoints

- <https://huggingface.co/allenai/OLMo-2-0425-1B> — the only checkpoint with
  recorded runs. The tests pin revision `a1847dff35000b4271fa70afc5db10fd29fedbdf`.

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
find those two modules, and it provides the `dist_cleanup` fixture that the
plugin's test tree would otherwise supply.

## Validation report

### Source and environment

Two plugin commits carry the recorded results, so no single commit covers the
whole table:

- `60e32deb879ef75b67b33628e465f86488a1422e` — the CPU loading/components run
  and the 2-node device functional run.
- `4cfa33adbc61f4f2ce5d166fe7fda1dcb8ca0c52` — the GSM8K accuracy run and the
  performance run.

Every run was on `trn2.48xlarge` with `vllm-neuron` 0.31.0.1.2.0, vLLM 0.31.0
and the `neuron_native` runtime backend. The checkpoint is
`allenai/OLMo-2-0425-1B`; the test constants pin HuggingFace revision
`a1847dff35000b4271fa70afc5db10fd29fedbdf`.

The runs used more than one container image. The evidence does not say which
image served which row.

### Model configuration

The architecture class is `Olmo2ForCausalLM` (`src/factory.py` validates the
config and selects `src/model.py`). It is a dense model, so there is no expert
parallelism; the evidence names tensor parallelism only. Every recorded run
used TP 8 — on one node for the accuracy and performance runs, and as 2 nodes
with 4 ranks each for the device functional run. The implementation computes in
BF16 from the FP32 checkpoint, which is why `neuron_config.quantization` is
`bf16` in every command above.

The serving test constants state a maximum model length of 2048 with a
matching 2048-token batched-token limit, `--max-num-seqs` 1 or 4, block size
32, prefix caching off and async scheduling off. Registration is the only
step outside this folder, and it is the user's: either register the factory
from your own script (offline) or add the architecture to the plugin registry
(online), as described under Usage.

### Component validation

| Run | Commit | Nodes | Cases |
|---|---|---|---|
| CPU Loading and Components (TRN2) | `60e32de` | 1 | PASS, 5 of 5 |
| Device Functional, 2 nodes (TRN2) | `60e32de` | 2 | PASS, 12 of 12 |

The CPU run covers the loading path (registration contract, config parse,
checkpoint mapping coverage, weight-loader transforms, full 8-rank CPU load)
and the one component case that has a CPU path. The device run covers the four
layer-0 attention and MLP component cases on each node, plus the port's serving
cases. Those compositions are read from the test sources — the evidence holds
the aggregate verdict and the case counts, not a per-case breakdown.

### Numerical correctness

No logit-validation run is on record for this model. None of the recorded test
ids identifies a logit-validation run, and we hold `test_verdict` values and
case counts only, so there is no logit result, tolerance or divergence count to
report here.

The test that ships as `test/integration/test_model.py` does perform logit
validation: it compares the served raw logits with a HuggingFace golden
reference and keeps the strict argmax bound of `1e-5`, with top-K logit bounds
of 0.03 (K=5), 0.05 (K=50), 0.08 (K=1000) and 0.09 (all). That test has no
recorded verdict.

Accuracy evidence for this model is the GSM8K run instead. That run evaluated
GSM8K on the native backend at batch 1 and batch 4. It passed, 2 of 2 cases,
1 node, at commit `4cfa33a`. We hold its verdict and case counts only — no
GSM8K score was read out of the measurement artifacts, so none appears here.

### Serving validation

Our required grid is 1,024 input / 512 output and 8,192 input / 1,024 output,
offline and online:

| Workload | Offline (`LLM(...)`) | Online (`vllm serve`) |
|---|---|---|
| 1,024 in / 512 out | No evidence | Evidence: the 2-node device functional run (PASS, 12 of 12, commit `60e32de`) executed a 1,024-token prompt with exactly 512 generated tokens. The test id does not name the shape; it comes from the test source that run executed. |
| 8,192 in / 1,024 out | No evidence | No evidence |

We do not have the 8,192 / 1,024 row for this model in either mode. Every
recorded run was configured with a 2,048-token maximum model length, so that
workload could not have run.

Two further online runs passed at commit `4cfa33a`, both on one node at TP 8
with batch 1 and batch 4. One is the GSM8K accuracy run (2 of 2 cases). The
other is a performance comparison run on the native backend (2 of 2 cases).
This README states only the existence and the verdict of the performance run.
It publishes no performance figure.

### Exceptions

- **No logit-validation run on record.** No recorded test id identifies a
  logit-validation run for this model, and the evidence holds verdicts and case
  counts only. The shipped integration test does logit validation at the strict
  `1e-5` argmax bound, but it has no recorded verdict.
- **No GSM8K score.** The accuracy run's verdict (PASS) and case counts (2 of
  2) are all we hold.
- **No performance figures.** This README states the existence and the verdict
  of the performance run. It publishes no latency or throughput number.
- **No 8,192 / 1,024 serving row**, offline or online, and **no offline
  1,024 / 512 row**. The only recorded serving shape is the online
  1,024 / 512 request inside the device functional run.
- **Dropped: the port's GSM8K accuracy test** (`test_native_comparison_accuracy.py`).
  It imports `test.utils.simple_server` and a campaign-local
  `native_comparison_support` module that carries historical baselines and
  writes to `WORKLOAD_OUTPUT_RW` — shared helpers and infrastructure outside
  the model folder. Its result is the GSM8K row above.
- **Dropped: the port's performance test** (`test_native_comparison_benchmark.py`).
  Same shared harness plus `test.utils.bench`.
- **Dropped: the port's 2-node serving test** (`test_serving.py`). It imports
  `test.utils.network`, `test.utils.simple_server`,
  `test.vllm_neuron.utils.di_utils`, `test.vllm_neuron.utils.logit_test_utils`
  and `test.vllm_neuron.utils.test_prompts`. Its logit-validation shape is what
  `test/integration/test_model.py` carries now, with locally defined prompts.
- **Dropped: the multi-node launcher** (`launch_multinode.py`, imports
  `test.utils.network`). The shipped module tests therefore default to one node
  with all 8 ranks; the recorded 2-node topology cannot be reproduced from this
  folder alone.
- **Rewritten assertion.** The port's CPU test asserted that
  `vllm_neuron.model.registry.get_models()` maps `Olmo2ForCausalLM` to the
  factory. A contrib model is deliberately absent from that registry, so the
  shipped test checks the registration contract instead (the checkpoint's
  architecture string and the factory that `src/__init__.py` exports). The
  original registry assertion is no longer covered by anything here.
- **Rewritten checkpoint resolution.** The port resolved the checkpoint through
  `test.vllm_neuron.utils.logit_test_utils.get_model_checkpoint`. The vendored
  helper downloads the pinned revision from the HuggingFace hub instead, or
  uses `OLMO2_CHECKPOINT_PATH`.
- **Case composition is inferred.** The mapping from the recorded case counts
  (5 and 12) to the test functions in this folder is read from the test
  sources; the evidence records only aggregate verdicts and counts.
- **Untested configurations.** Prefix caching and the segmented-prefill path
  (every recorded serving run set `--no-enable-prefix-caching`), tensor
  parallel sizes other than 8, context beyond 2,048 tokens, speculative
  decoding, Trn3, and any plugin or vLLM version other than 0.31.0.1.2.0 /
  0.31.0.
- **Image ambiguity.** The runs used more than one container image. The
  evidence does not map an image to a test row.
- **Registration reaches the engine by `fork` only.** Option 1 in
  `contrib/README.md` — register the factory from your own script, which is
  what `test/integration/test_model.py` does — relies on vLLM's engine-core
  and worker processes inheriting the registration from the process that made
  it. That holds for the default `fork` start method. Under
  `VLLM_WORKER_MULTIPROC_METHOD=spawn` it cannot work, because a spawned child
  re-imports from scratch and never runs the registration call. This is read
  from the code, not from a run: the evidence records no offline serving run
  for this model at all, so the shipped integration test's registration path
  has no recorded verdict either way. See the note under "Offline inference
  with the Python API".
