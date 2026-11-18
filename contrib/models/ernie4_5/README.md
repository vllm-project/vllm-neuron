# Contrib Model: ERNIE 4.5 (dense, text)

This folder contributes the ERNIE 4.5 dense architecture
(`Ernie4_5ForCausalLM`) to `vllm-neuron`, so you can serve a text ERNIE 4.5
checkpoint with vLLM on AWS Trainium. The modeling code is in `src/`, the
validation report at the end of this file records exactly what was tested.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

ERNIE 4.5 dense is a Llama-type decoder whose RoPE is interleaved (GLM style)
rather than rotate-half. The weight loader reorders the Q and K rows of each
head once, at load time, so the Neuron kernels can run their rotate-half RoPE
and still produce the HuggingFace rotation. `src/README.md` describes that
conversion and the rest of the implementation.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md` for installation). The plugin auto-registers with vLLM, but a
contrib architecture is not in the plugin's registry, so you register it
yourself — see
[contrib/README.md](../../README.md) — "How do I register a custom model with
vLLM?".

Which option you need depends on how you start the engine, and for this model
it is worth getting right before you try to serve.

> **Read this before serving.** Upstream vLLM ships its own
> `Ernie4_5ForCausalLM`: the architecture string this folder uses is already in
> vLLM's built-in model table. So a process that lacks your registration does
> **not** tell you the architecture is unsupported, the way it would for a
> contrib name vLLM has never heard of. It resolves vLLM's class instead, and
> then fails inside the Neuron model runner on `from_configs` — a plugin
> convention that vLLM's own model classes do not define. The resulting
> `AttributeError` names nothing about registration, so a missed registration
> shows up as a confusing crash rather than as a clear "unsupported
> architecture" message. On the working path the signal is the reverse: vLLM
> logs "already registered ... will be overwritten" when your registration
> replaces the built-in entry.

- **Option 1 (register from your own script) covers the offline `LLM(...)` API
  below, and only that.** The plugin pins the multi-process executor and the
  Neuron worker (`vllm_neuron/vllm/platform.py`), so the topology is your
  process → an engine-core process → the executor's workers. Nothing copies a
  registry entry between processes; the children get yours by inheriting your
  memory, which requires the `fork` start method. `fork` is
  `VLLM_WORKER_MULTIPROC_METHOD`'s default (read from vLLM 0.31.0 source;
  this repo now pins vLLM 0.32.0, which we did not read), and the only place
  in the plugin that sets that variable forces `spawn` for Qwen3-ASR alone.
- **Option 2 (move `src/` into `vllm_neuron/model/` and add the architecture to
  `_MODELS`) is what `vllm serve` needs.** The reason holds whatever the start
  method is: `vllm serve` gives you no script in which to call
  `register_model` at all. Separately, two comments in the plugin say the CLI
  starts its engine core with `spawn` — in which case an Option 1 registration
  could not reach it even if you had somewhere to put the call — while a third
  comment describes a *forked* engine core, so we report that as the plugin's
  own commentary rather than as a verified fact. Either way, the `vllm serve`
  example below assumes you have done Option 2. That option edits plugin core
  code, which this contribution deliberately does not do — see
  [contrib/README.md](../../README.md) for the trade-off it describes.

These statements were read out of the plugin's and vLLM's code, not out of a
run of this contribution; see Exceptions.

This implementation runs on the `neuron_native` route only, so set
`VLLM_NEURON_BACKEND=neuron_native` for every process, including the server.

### Serve with the OpenAI-compatible server

This needs Option 2 above (the architecture in the plugin's `_MODELS`). With
only an Option 1 registration the server has nowhere to run your registration
call, and because vLLM ships its own `Ernie4_5ForCausalLM` the server resolves
that one and then fails on `from_configs` instead of telling you the
architecture is unregistered.

```bash
VLLM_NEURON_BACKEND=neuron_native vllm serve baidu/ERNIE-4.5-0.3B-PT \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching
```

Then send a request:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "baidu/ERNIE-4.5-0.3B-PT",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

```python
import os

os.environ["VLLM_NEURON_BACKEND"] = "neuron_native"

from vllm import LLM, ModelRegistry, SamplingParams

# Copy contrib/models/ernie4_5/src/ into your own package first, then register
# the factory before you construct the engine. The engine core and its workers
# are forks of this process, so they inherit this registration; do not set
# VLLM_WORKER_MULTIPROC_METHOD=spawn with this arrangement, or the child
# resolves vLLM's own Ernie4_5ForCausalLM and fails on from_configs.
from your_package.ernie4_5 import Ernie4_5ForCausalLM

ModelRegistry.register_model("Ernie4_5ForCausalLM", Ernie4_5ForCausalLM)

llm = LLM(
    model="baidu/ERNIE-4.5-0.3B-PT",
    dtype="bfloat16",
    tensor_parallel_size=1,
    max_model_len=2048,
    max_num_seqs=4,
    enable_prefix_caching=False,
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The color of the sky is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

Prefill runs sequence-parallel, so a prompt must be longer than the
tensor-parallel size and a multiple of it. The model raises a `ValueError`
otherwise.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1` in the environment. See the top-level `README.md` and
`AGENTS.md` for the full list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

Our evidence records container image digests and the plugin and vLLM versions,
not a Neuron SDK release number, so this matrix is keyed on the plugin version
instead of the SDK version. Every result we hold was produced on
`trn2.48xlarge`.

| Instance / plugin version | 0.31.0.1.2.0 (vLLM 0.31.0) | Other versions |
| ------------------------- | -------------------------- | -------------- |
| Trn2                      | Working                    | Not tested     |
| Trn3                      | Not tested                 | Not tested     |

## Example Checkpoints

- <https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT> (revision
  `b565cf6caebdb7a1eadf00100857b1ed5e044f12`, the only checkpoint and revision
  with evidence)

Larger dense ERNIE 4.5 checkpoints use the same architecture string, but none
of them was tested here.

## Testing

The integration test serves the model through vLLM on Neuron and validates its
per-token logits against a HuggingFace reference. It needs Neuron hardware.

```bash
VLLM_NEURON_BACKEND=neuron_native pytest \
    contrib/models/ernie4_5/test/integration/test_model.py \
    -v -s --timeout=3600
```

Set `ERNIE4_5_CHECKPOINT_DIR` to a local snapshot of the pinned revision to
test the exact checkpoint the results below were produced with. Otherwise the
tests download it.

### Unit tests (CPU mode)

`test/unit/` holds the architecture and config checks and the weight-loading
tests. They run on CPU, without Neuron hardware.

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/ernie4_5/test/unit -v
```

### Module tests

`test/module/` holds the per-component tests. They compare one component
(attention prefill, attention decode, MLP) against FP32 and BF16 HuggingFace
references with the three-way pattern of the
[module test guidelines](../../../docs/design/accuracy/module_test_guidelines.md).
`test_components_cpu.py` runs the torch fallback paths on CPU:

```bash
VLLM_NEURON_CPU_MODE=1 pytest \
    contrib/models/ernie4_5/test/module/test_components_cpu.py -v
```

`test_components_multinode.py` runs the same components on Neuron devices
across two nodes (prefill on node 0, decode on node 1) and also asserts that
each component dispatched its NKI kernel rather than a torch fallback. Start
one process per node:

```bash
mpirun -n 2 pytest \
    contrib/models/ernie4_5/test/module/test_components_multinode.py -v
```

### Test layout

The tests resolve this folder's own modules by file location, never through a
`contrib.models.ernie4_5...` namespace path, so they keep working after the
folder is copied out of the repository — which is the whole point of the
guide's self-containment rule. `test/_local_src.py` does that resolution, and
each test module loads `_local_src.py` itself by file location, under the
unique name `contrib_ernie4_5_local_src`. Nothing goes on `sys.path`.

The three helper modules that `test/unit`, `test/module` and `test/integration`
share therefore exist once, at the `test/` root:

| Helper | Reached as | Purpose |
| --- | --- | --- |
| `_local_src.py` | by file location | Loads `src/` under the module name `contrib_ernie4_5_src`, and the two helpers below under `contrib_ernie4_5_test_*`. |
| `_checkpoint.py` | `_local_src.load_helper("checkpoint")` | Resolves and validates the pinned checkpoint, and holds the shared checkpoint facts. |
| `_component_utils.py` | `_local_src.load_helper("component_utils")` | HuggingFace goldens, KV-cache seeding and the picklable `model_load` functions for the component tests. |

Those unique module names are what lets several contrib models load in one
process. `test/conftest.py` holds one thing only: it forces
`VLLM_NEURON_BACKEND=neuron_native` (the only route this implementation
supports) before pytest imports a test module.

One caveat is left, and the loader is not its cause: this folder uses the
conventional test file names (`test/integration/test_model.py`, as the template
ships it), which every contrib model shares, so collecting several contrib
models in one pytest session in the default `prepend` import mode hits pytest's
"import file mismatch" error on the duplicate basenames. Collecting several
contrib models in one run needs `--import-mode=importlib`; otherwise run one
model folder per invocation.

## Validation report

This section records the validation evidence for this contribution.
Every result below is a test verdict or a case count that we read out of our
CI records. No accuracy score, tolerance measurement, latency or throughput
figure was read out of the measurement artifacts, so none is quoted here.

### Source and environment

Four CI tests passed for this model, and they do not share a single plugin
commit, so each row below cites its own:

| Test | Cases | Nodes | Plugin commit |
| --- | --- | --- | --- |
| Loading and components on CPU, BF16 | 6/6 passed, 0 failed | 1 | `60e32deb879ef75b67b33628e465f86488a1422e` |
| Two-node 1P1D logits, serving and components on Trn2, BF16 | 6/6 passed, 0 failed | 2 | `60e32deb879ef75b67b33628e465f86488a1422e` |
| Native-comparison serving performance at TP 1, batch 1 and batch 4 | 2/2 passed, 0 failed | 1 | `0c5752284168a59a0ef65db7e84da3396ebe7da6` |
| Native-comparison GSM8K accuracy at batch 1 and batch 4 | 2/2 passed, 0 failed | 1 | `0c5752284168a59a0ef65db7e84da3396ebe7da6` |

Every run used `trn2.48xlarge`, plugin version `0.31.0.1.2.0` with vLLM
`0.31.0`, and the `neuron_native` runtime backend. The checkpoint is
`baidu/ERNIE-4.5-0.3B-PT` at revision
`b565cf6caebdb7a1eadf00100857b1ed5e044f12`.

The CI record does not link each result to one image or plugin commit.

### Model configuration

The architecture class is `Ernie4_5ForCausalLM`, served in bfloat16; the
factory rejects any other dtype. The performance run used TP 1, and the
disaggregated-inference and component test files each parametrize TP 8 and TP
1 and passed every case, so both degrees ran. No expert parallelism applies:
this is a dense model. The logit, serving and native-comparison runs set
`max_model_len` and `max_num_batched_tokens` to 2,048 with `max_num_seqs` 4.

Registration is the user's job, as described under "Usage" above and in
`contrib/README.md`; nothing outside this folder needs to change. Beyond the
standard arguments, the model needs `VLLM_NEURON_BACKEND=neuron_native`, a
bfloat16 dtype, and prompts whose length is a multiple of the
tensor-parallel size (sequence-parallel prefill).

### Component validation

The CPU run for loading and components passed all 6 of its cases with no
failures, on one node. Those cases are the tests this folder ships as
`test/unit/test_factory.py` (architecture, config fields and config
rejection), `test/unit/test_weight_loaders.py` (the interleaved-to-
rotate-half reorder against the HuggingFace RoPE, the fused QKV loader at TP 8
rank 5, and a full-checkpoint load that checks mapping coverage, dtype and
sampled parameter values) and `test/module/test_components_cpu.py` (attention
prefill and attention decode against FP32/BF16 HuggingFace references).

The two-node 1P1D run for logits, serving and components passed all 6 of its
cases with no failures. Four of those cases come from
`test/module/test_components_multinode.py`, which runs attention and MLP on
Neuron devices at TP 8 and TP 1 with prefill on one node and decode on the
other, and asserts NKI-kernel dispatch for each component. The remaining two
cases are the 1P1D logit and serving cases described below, which this folder
does not ship.

### Numerical correctness

Logit validation ran, inside the two-node 1P1D run: the port's
`test_di_logits_serving.py` teacher-forced two prompts for 16 tokens at batch
4 and compared raw per-token logits against FP32 and BF16 HuggingFace
references. The run passed, 6 of 6 cases, with no failures.

The shipped `test/integration/test_model.py` uses the tolerance map of that
run — `(1e-5, 0.015)` at top-5, `(1e-5, 0.04)` at top-50, `(1e-5, 0.06)` at
top-1000 and `(1e-5, 0.065)` over the full vocabulary — and keeps the strict
`1e-5` argmax-divergence bound. It is an offline, single-node adaptation of
that test, not a rerun of it: see Exceptions.

The GSM8K accuracy run passed both of its cases, at batch 1 and at batch 4.
We hold its verdict only. We did not read a GSM8K score.

### Serving validation

Our required serving grid is 1,024 input / 512 output tokens and 8,192 input /
1,024 output tokens, offline and online:

| Workload | Offline | Online |
| --- | --- | --- |
| 1,024 in / 512 out | No evidence | Evidence: the serving phase of the two-node 1P1D run (4 concurrent requests through the disaggregated proxy, `ignore_eos`, exact token counts asserted). That run passed 6/6. |
| 8,192 in / 1,024 out | No evidence | No evidence |

A separate online `vllm bench serve` performance run at TP 1 passed both of
its cases. It requested 1,024 input and 100 output tokens, with 20 measured
requests at batch 1 and at batch 4. This README does not quote latency or
throughput figures.

### Exceptions

- The shipped `test/integration/test_model.py` has never been run. It is an
  offline single-node adaptation of the two-node 1P1D logit test (same
  prompts, output length, batch and tolerance map, TP 1), written for this
  contribution.
- Dropped: `test_di_logits_serving.py` (the 1P1D logit and serving test). It
  needs a shared multi-node serving harness that does not ship with this
  folder. To keep the test, we would have to copy that harness into the
  folder. The integration test replaces the logit half in offline form. The
  disaggregated serving half has no replacement here.
- Dropped: `test_native_comparison_accuracy.py` (GSM8K), the
  `test_native_comparison_benchmark.py` performance gate, and their shared
  `native_comparison_support.py`. They need shared test helpers that do not
  ship with this folder. They also need a checkpoint path that is not
  available outside the repository. This folder has no replacement for them.
- The port's session `conftest.py` forced `VLLM_NEURON_BACKEND=neuron_native`
  and asserted the selected backend. The forcing line now lives in this model's
  own `test/conftest.py`, which is the same effect at the scope the contrib
  layout allows; the asserting fixture was dropped.
- `test/unit/test_factory.py` no longer asserts that
  `vllm_neuron.model.registry.get_models()` maps the architecture to the
  factory, because a contrib model is deliberately absent from that registry.
  It asserts the pair a user has to register instead — the architecture string
  in `config.json` and the factory class in `src/`. The config-field and
  config-rejection checks are unchanged.
- The checkpoint resolver lost its third source, the shared
  `get_model_checkpoint` helper. It now uses `ERNIE4_5_CHECKPOINT_DIR` or
  `huggingface_hub.snapshot_download` at the pinned revision, with the same
  architecture and file-size checks.
- The port's `test/utils.py` was vendored into this folder as
  `test/_checkpoint.py` (see the checkpoint-resolver bullet above).
  `contrib/README.md` sanctions copying a shared test helper into a model
  folder rather than reaching outside it, which is what that vendoring is.
- This architecture string collides with upstream vLLM. vLLM's built-in model
  table maps `Ernie4_5ForCausalLM` to its own `ernie45` implementation, so
  registering this folder's factory overwrites a built-in entry instead of
  adding a new one. A process that lacks the registration therefore does not
  report an unsupported architecture: it resolves vLLM's class and then fails
  on `from_configs`, the plugin convention that both Neuron model runners call
  and that no vLLM class defines, so the error names nothing about
  registration. It is a confusing crash, not wrong output. The integration
  test skips itself unless `VLLM_WORKER_MULTIPROC_METHOD` is `fork`, so a
  registration problem cannot surface as that crash mid-test, and the README
  leads its usage section with the hazard.
- Registering a contrib architecture is process-start-method dependent, and
  the two things the guide asks for are in tension. The local-src loader keeps
  the tests runnable from a copied-out folder by loading `src/` from its path
  under a synthetic module name, which no fresh interpreter can import by
  name; a registration that survives `spawn` would need the opposite, a module
  importable by name (the lazy `"module:Class"` form that the plugin uses for
  its own models, and which vLLM resolves by importing it in a fresh
  interpreter). So the integration test and the README's offline example both
  register a class object, and both rely on `fork`.
- Sourcing for the statements above: `fork` being the default of
  `VLLM_WORKER_MULTIPROC_METHOD`, the built-in `Ernie4_5ForCausalLM` entry,
  and class-object registration being inspected in-process were read from
  vLLM 0.31.0 source, in which `from_configs` does not appear at all. This
  repo's `requirements/core.txt` now pins vLLM 0.32.0, which we did not read.
  That the vLLM CLI uses `spawn` rests on two
  comments in the plugin, with a third describing a forked engine core, so it
  is reported as commentary, not as verified. One vLLM path that could force
  `spawn` by itself (`_maybe_force_spawn`) was not read; it is not expected to
  fire on Neuron, where there is no CUDA. None of this came from a run of this
  contribution.
- All evidence was produced with this implementation in-tree, at
  `vllm_neuron/model/ernie4_5/`, registered through `get_models()`. The contrib
  layout — code in `contrib/models/ernie4_5/src/`, registered by the user — was
  not itself re-run in CI.
- Untested in our evidence: Trn3; any Neuron SDK release number (none was
  recorded); prefix caching and chunked prefill (the segmented-attention path
  exists in `src/model.py`, and the logit and serving runs set
  `--no-enable-prefix-caching`); speculative decoding (the forward pass has a
  rejection-sampler branch, never exercised); any quantized or FP8-KV
  configuration (none is implemented); LoRA and multimodal input (neither is
  implemented; `inputs_embeds` is rejected); the 8,192/1,024 serving row; the
  offline 1,024/512 serving row.
- `src/model.py` names data parallelism among its supported parallelism modes,
  but the folder contains no data-parallel wiring and our evidence covers
  none. Treat DP as unverified.
- No GSM8K score, measured tolerance, latency or throughput figure appears in
  this README. We hold test verdicts and case counts only.
