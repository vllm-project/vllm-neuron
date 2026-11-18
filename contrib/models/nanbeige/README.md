# Nanbeige4.2 (Text)

This folder serves the Nanbeige4.2 text models with vLLM on AWS Trainium
through the `vllm-neuron` plugin. The implementation under `src/` is a
Llama-style dense decoder with one addition: the stack of physical decoder
layers runs twice with the same weights, and each `(loop, layer)` pair holds
its own KV cache. See `src/README.md` for the architecture and the
feature-status card.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix and the
> validation report below record exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md` for installation). A contrib model is not in the plugin's model
registry, so register the architecture `NanbeigeForCausalLM` yourself, as
described in [contrib/README.md](../../README.md) ("How do I register a custom
model with vLLM?"). The checkpoint ships its own model code, so every entry
point needs `trust_remote_code`.

### Serve with the OpenAI-compatible server

`vllm serve` starts the engine directly, so there is no script of yours in
which to call `ModelRegistry.register_model`, and the CLI's engine-core
process does not inherit a registration made elsewhere (see "Registering this
model" below). Use option 2 of [contrib/README.md](../../README.md): move
`src/` to `vllm_neuron/model/nanbeige/` and add
`("NanbeigeForCausalLM", "vllm_neuron.model.nanbeige:NanbeigeForCausalLM")`
to `_MODELS` in `vllm_neuron/model/registry.py`. That edit is yours to make in
your own checkout; this contrib folder does not make it. Then:

```bash
vllm serve Nanbeige/Nanbeige4.2-3B \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 2048 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching
```

Then send a request:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Nanbeige/Nanbeige4.2-3B",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

Copy `src/` into your own application package first, then register the factory
class before you construct the engine (option 1 of
[contrib/README.md](../../README.md)):

```python
from vllm import LLM, ModelRegistry, SamplingParams

# Adjust this import to the package you copied contrib/models/nanbeige/src/
# into.
from your_package.nanbeige.factory import NanbeigeForCausalLM

ModelRegistry.register_model("NanbeigeForCausalLM", NanbeigeForCausalLM)

llm = LLM(
    model="Nanbeige/Nanbeige4.2-3B",
    trust_remote_code=True,
    dtype="bfloat16",
    tensor_parallel_size=8,
    max_model_len=2048,
    max_num_seqs=1,
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The color of the sky is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

Prefill runs sequence-parallel, so a prompt length must be greater than the
tensor-parallel size and a multiple of it.

### Registering this model

Option 1 above (register, then build `LLM(...)`) works on the offline path,
but only because that path forks. The plugin pins
`distributed_executor_backend="mp"`, so the topology is your process, then an
engine-core process, then one `MultiprocExecutor` worker per rank. The start
method comes from `VLLM_WORKER_MULTIPROC_METHOD`, and the plugin leaves it at
vLLM's `fork` default for every architecture except Qwen3-ASR. A forked child
inherits this process's populated `ModelRegistry`, so the architecture is
already there; `NeuronWorker.__init__` then registers the plugin's own
`get_models()` entries, which overwrite only their own architecture keys, and
the model runner resolves the final class with
`ModelRegistry.resolve_model_cls(...)`.

Under `spawn` the registration is lost: a spawned engine-core re-executes from
scratch and never runs your call, and a class object cannot be carried across
`spawn` at all. vLLM's spawn-safe form is the lazy `"module:Class"` string,
which needs a module importable by name in a fresh interpreter. If you set
`VLLM_WORKER_MULTIPROC_METHOD=spawn`, or you serve with the CLI, use option 2
instead. This paragraph is read from the plugin and vLLM code, not measured by
a run of this model.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1` in the environment. The fused decode megakernel does
not run on CPU, so CPU mode covers the prefill path only. See the top-level
`README.md` and `AGENTS.md` for the full list of `VLLM_NEURON_*` environment
variables.

## Compatibility Matrix

| Instance | Status | Software |
| --- | --- | --- |
| Trn2 (trn2.48xlarge) | Working — see the validation report | `vllm-neuron` 0.31.0.1.2.0, vLLM 0.31.0, `VLLM_NEURON_BACKEND=neuron_native` |
| Trn3 | Not tested | — |

This README states no Neuron SDK release. The recorded runs used a pinned
toolchain, but their results do not name the Neuron SDK release.

## Example Checkpoints

- <https://huggingface.co/Nanbeige/Nanbeige4.2-3B>

## Testing

The integration test serves the model through vLLM on Neuron and validates
its per-token logits against a HuggingFace reference:

```bash
pytest contrib/models/nanbeige/test/integration/test_model.py \
    -v -s --timeout=3600
```

Set `NANBEIGE4_CHECKPOINT_PATH` to a local checkpoint directory to avoid a
Hub download.

Every test here loads this folder's `src/` by file location through
`test/_local_src.py`, so the tests work from any working directory and after
the folder is copied out of the repository. They need the `vllm-neuron` plugin
importable (installed, or the repository root on `sys.path`, which
`pytest.ini` does for a run from the repository root), but nothing else from
the repository.

Run one model's tests per pytest invocation, or pass
`--import-mode=importlib`. The reason is pytest's default import mode, not
this folder: contrib/README.md gives every model the same
`test/integration/test_model.py` filename, and prepend mode cannot import two
test modules with the same basename in one run.

### Unit tests (CPU mode)

```bash
VLLM_NEURON_CPU_MODE=1 VLLM_NEURON_BACKEND=neuron_native \
    NANBEIGE4_CHECKPOINT_PATH=/path/to/Nanbeige4.2-3B \
    pytest contrib/models/nanbeige/test/unit/ -v
```

Four cases: the registration and factory wiring with the config parse, the
factory's refusal of each unsupported option, a TP2 full-model load and
prefill-logit comparison, and a cross-check of the independent reference
against the checkpoint's own model code.

These tests read the real checkpoint and refuse to run without it. They also
check that the checkpoint directory names the pinned revision
(`b82e54bd609793562a75cbf9337970a93369eab5`), so point the variable at a
directory whose path contains that revision. The module tests below have the
same requirement.

### Module tests

`test/module/test_decoder_layer.py` is the per-component test: it runs one
physical decoder layer in isolation at TP2, once for loop 0 and once for
loop 1, and gates on `assert_close_three_way` against an upstream
`LlamaDecoderLayer` reference — for the layer output and for the KV cache that
each loop owns. It is parameterized over the first and the last physical
layer, which is two cases.

```bash
VLLM_NEURON_CPU_MODE=1 VLLM_NEURON_BACKEND=neuron_native \
    NANBEIGE4_CHECKPOINT_PATH=/path/to/Nanbeige4.2-3B \
    pytest contrib/models/nanbeige/test/module/ -v -s --timeout=600
```

The unit tests and the module tests share `test/nanbeige_utils.py`, which sits
at the root of the test tree for that reason. The fused decode megakernel does
not run on CPU, so the module test covers prefill only.

## Validation report

### Source and environment

Both recorded test runs passed at plugin commit
`cbdb21f13cf51702182309bb6f3a329370ac76fd`, so one commit covers all the
results below. The runs used `vllm-neuron` 0.31.0.1.2.0 and vLLM 0.31.0 with
the native runtime backend (`neuron_native`) on `trn2.48xlarge`.

The checkpoint is `Nanbeige/Nanbeige4.2-3B` at revision
`b82e54bd609793562a75cbf9337970a93369eab5`, which the tests pin and verify
against the published `config.json`.

### Model configuration

The architecture class is `NanbeigeForCausalLM` (factory in `src/factory.py`,
implementation in `src/model.py`), BF16 weights, no quantization and no
expert parallelism. The model is dense, so there is no EP degree.
The device run sharded the model over TP8, with 4 ranks on each of two
`trn2.48xlarge` nodes; the CPU tests build it at TP2. The device test
constants set `--max-model-len 2048` for both the logit and the serving
case, with `--max-num-batched-tokens 512 --max-num-seqs 1` for the logit case
and `--max-num-batched-tokens 1024 --max-num-seqs 4` for the serving case.
Model-specific serving arguments: `--trust-remote-code` (the checkpoint ships
its model code), `--dtype bfloat16`, `--no-enable-prefix-caching`, and
on-device greedy sampling for the logit case. Registration is the user's
step; see Usage above.

### Component validation

A CPU-mode run of the model loading and component tests passed 6 of 6 cases
at commit `cbdb21f1`. The 6 cases check these items:

- the factory wiring and the config parse;
- the refusal of each unsupported option;
- a TP2 full-model load that reads all 201 checkpoint tensors, and a
  prefill-logit comparison against an independent upstream-Llama loop
  reference;
- a per-loop decoder-layer and KV-cache comparison for the first and the last
  physical layer (2 cases);
- a cross-check of that reference against the checkpoint's own model code.

The same 6 cases are in this folder, split by kind into the folder structure
that contrib/README.md defines. The 4 loading and configuration cases are in
`test/unit/test_nanbeige_cpu.py`. The 2 per-loop decoder-layer cases are in
`test/module/test_decoder_layer.py`. The split changed no assertion.

A two-node device test on Trn2, sharded at TP8, passed 8 of 8 cases at commit
`cbdb21f1`. This test is not in this folder (see Exceptions). Its test file
has 4 cases, and each of the 2 nodes ran them, which gives 8 cases. The 4
cases are a decoder-layer prefill-and-decode comparison for the first and the
last physical layer, a server logit validation, and a serving run. This
case-to-node mapping comes from the case count and the test file. The run
results give the test verdict and the case counts, but not the result of each
case.

### Numerical correctness

This model has no dataset-level accuracy result of any kind. There is no
GSM8K or other dataset accuracy run, and neither recorded test is a dedicated
logit test. See Exceptions.

The logit result is indirect. The two-node device test that passed 8 of 8 has
a case that starts a TP8 server across both nodes. That case compares the
full-vocabulary logits of each of 32 generated tokens against an independent
HuggingFace reference. The reference is an FP32 baseline plus a same-dtype
teacher-forced pass. The case uses `multi_prompt_logit_validation` with the
default tolerance map of `vllm_neuron.accuracy`. All cases in that test
passed, so this comparison passed. The run results do not include a per-case
metric.

`test/integration/test_model.py` in this folder uses the same default bounds:
`(1e-5, 0.011)` at top-5, `(1e-5, 0.02)` at top-50, `(1e-5, 0.03)` at
top-1000 and `(1e-5, 0.05)` over the full vocabulary. It also keeps the strict
`1e-5` argmax bound. This test has no CI run. Its configuration is
single-node TP8, offline `LLM`, and reference logits from the checkpoint's own
model code. The recorded run used a two-node server instead.

### Serving validation

| Workload | Offline | Online |
| --- | --- | --- |
| 1,024 in / 512 out | No evidence | Passed: 4 concurrent requests, exactly 512 generated tokens each, in the two-node device test (8/8 passed, commit `cbdb21f1`) |
| 8,192 in / 1,024 out | No evidence | No evidence |

In the online 1,024 / 512 case, the device test sent 4 concurrent OpenAI-API
completion requests. Each request had exactly 1,024 prompt tokens.
`min_tokens` and `ignore_eos` forced 512 output tokens. The test asserted the
token counts and a non-empty completion. There is no offline
(`LLM.generate`) measurement at that shape. There is no run at 8,192 / 1,024
in either mode.

### Exceptions

- **No logit-validation test and no accuracy test.** This model has two
  recorded passing tests. Neither is a dedicated logit test, and neither is a
  GSM8K or other dataset accuracy run. So this README states no accuracy
  score. The only logit comparison is the server-logit case in the two-node
  device test, as described under Numerical correctness.
- **The two-node device test is not in this folder.** It needs shared test
  helpers outside the model folder: `test/utils/simple_server`,
  `test/utils/network`, `test/vllm_neuron/utils/server_utils` and
  `test/vllm_neuron/utils/longbench_prompts`. It also needs a two-node harness
  that discovers peer IPs and runs a headless follower server. A copy of that
  orchestration would add several hundred lines of infrastructure to the
  model folder. That code no longer matches its original harness.
  So this folder does not include the test. The device result above comes
  from it, but you cannot run it again from this folder. For the same
  reason, `test/nanbeige_utils.py` omits the helper code that only that test
  used: the online generate function and the multi-node server orchestration.
  That code was the only code that imported
  `test.vllm_neuron.utils.server_utils` and `test.utils.network`.
- **`PROMPTS_2` is vendored.** The two validation prompts come from
  `test/vllm_neuron/utils/test_prompts.py`, and they are byte-identical to
  it. This folder copies them verbatim into `test/nanbeige_utils.py` and into
  the integration test, as contrib/README.md permits.
- **The registry assertion has a different meaning.** The recorded CPU test
  asserted that `vllm_neuron.model.registry.get_models()` maps the
  architecture to the factory. A contrib model is not in `get_models()`. So
  the test in this folder registers the local factory with vLLM's
  `ModelRegistry`. It then asserts that the architecture resolves to the
  factory. The case count is the same.
- **The recorded CPU suite is split across two directories.** The run that
  passed 6 of 6 used one file. Here its 4 loading and configuration cases are
  in `test/unit/` and its 2 per-loop decoder-layer cases are in
  `test/module/`, to match the folder structure contrib/README.md defines.
  The split adds, drops or rewrites no case. No run covers the split layout.
- **Option 1 registration depends on the `fork` start method.** The
  integration test, and the offline example in Usage, register the factory
  class in the user's process and rely on the engine-core and worker
  processes inheriting that registration through `fork`. Under `spawn`, and
  therefore under `vllm serve`, it does not reach them; vLLM's spawn-safe
  lazy `"module:Class"` form needs a module importable by name in a fresh
  interpreter, which the file-location loader this folder uses deliberately
  avoids. Serving needs option 2 (see "Registering this model"). This is
  read from the plugin and vLLM source, not observed in a run of this model.
- **The tests resolve `src/` by file location.** The recorded runs imported
  the model code through the repository-rooted `contrib.models.nanbeige.src`
  path. The tests in this folder load `src/` through `test/_local_src.py`
  instead, under the module name `contrib_nanbeige_src`. Thus the folder
  works after you copy it out of the repository. The recorded runs used the
  old import scheme. A static check verified test collection of the new
  layout outside the repository, with the plugin stubbed. No run on hardware
  or in CI covers the 4 unit cases and the 2 module cases after the move to
  `test/_local_src.py`.
- **The integration test configuration is untested as such.** It is adapted
  from the template and has never been run in CI. Its `(1, 2048)` and
  `(4, 2048)` cases mirror the shapes of the recorded logit and serving runs,
  but those ran on a two-node server, not single-node offline.
- **The integration test's reference depends on the checkpoint's model
  code.** `get_cached_reference_goldens` loads the HuggingFace reference with
  `trust_remote_code=True`. The CPU test in `test/unit/test_nanbeige_cpu.py`
  documents that this remote code does not load on every `transformers`
  version. When it cannot load, that test skips and gives the reason. On such
  a version, the integration test cannot build its goldens. The recorded
  device run used an independent upstream-Llama loop reference instead.
- **Untested configurations.** Only TP8 (device, 2 nodes) and TP2 (CPU) have
  results; no other tensor-parallel degree, no Trn3, no quantized weights, no
  speculative decoding (the code has a rejection-sampling branch with no
  evidence behind it), no data-parallel or disaggregated (prefill/decode)
  deployment, no prefix caching (the runs disabled it), and no context length
  beyond 2,048 tokens, even though the checkpoint config allows far more.
- **No performance data.** No benchmark or latency test ran for this model,
  so this README states no throughput, latency or time-to-first-token figure.
