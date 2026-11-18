# EXAONE 4.0 (Text)

This folder serves the EXAONE 4.0 text models with vLLM on AWS Trainium
through the `vllm-neuron` plugin. The implementation under `src/` is a dense
decoder with hybrid attention: three sliding-window layers with RoPE followed
by one full layer with no position embedding, repeated through the stack. See
`src/README.md` for the architecture and the feature-status card.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix and the
> validation report below record exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md` for installation). Registration needs one extra thought for this
model: vLLM already has an `Exaone4ForCausalLM` of its own, so the
architecture must be *re-*registered to point at the class in `src/`,
otherwise vLLM builds its own implementation. See
[contrib/README.md](../../README.md) ("How do I register a custom model with
vLLM?").

> **Read this before you serve the model.** Option 1 of
> [contrib/README.md](../../README.md) — calling
> `ModelRegistry.register_model(...)` from your own script — works only for the
> offline `LLM(...)` API, and **cannot** work for `vllm serve` or any other
> vLLM CLI entrypoint. For most contrib models that mistake ends in a clean
> "architecture not registered" error. For this one it does not, because
> upstream vLLM ships this architecture name. A process that misses the
> registration resolves *vLLM's* `Exaone4ForCausalLM` instead of reporting an
> unsupported architecture, and then fails inside the Neuron model runner's
> `load_model` with an `AttributeError` on `from_configs`. Both Neuron model
> runners follow the same pattern: resolve the architecture with
> `ModelRegistry.resolve_model_cls(...)`, then call
> `model_cls.from_configs(...)` (see
> `vllm_neuron/vllm/worker/neuron_model_runner.py` and
> `neuron_model_runner_v2.py`). `from_configs` is the plugin's own factory
> convention — every plugin factory provides it, including `src/factory.py`
> and `vllm_neuron/model/llama3/factory.py`, and vLLM's own model classes do
> not. So the failure is loud, and it does **not** produce wrong output or
> wrong numbers; it is just confusing, because it names a missing attribute
> rather than the registration you skipped. Use option 2 for serving, as
> described below.
>
> Why: the plugin runs the engine core and the workers as separate processes
> (`distributed_executor_backend="mp"`, see `vllm_neuron/vllm/platform.py`).
> When you drive vLLM as a library, those processes are `fork`ed, so they
> inherit your process's `sys.modules` and your populated `ModelRegistry`, and
> your registration is in effect without being repeated. The vLLM CLI instead
> starts the engine core with `spawn`, which re-executes a fresh interpreter
> that never runs your script. A registered *class object* cannot cross
> `spawn` at all; only the lazy `"module:Class"` string form can, which is the
> form option 2 uses. This is read from the plugin's code, not from a recorded
> run — see the Exceptions section.

The 32B checkpoint is about 64 GB in BF16; the recorded runs used TP8 on
`trn2.48xlarge`.

### Serve with the OpenAI-compatible server

`vllm serve` starts the engine directly, so there is no script of yours in
which to call `ModelRegistry.register_model` — and a wrapper script that calls
it before invoking the CLI does not help, because the CLI's engine-core
process is spawned and starts from a fresh interpreter. Use option 2 of
[contrib/README.md](../../README.md): move `src/` to
`vllm_neuron/model/exaone4/` and add
`("Exaone4ForCausalLM", "vllm_neuron.model.exaone4:Exaone4ForCausalLM")` to
`_MODELS` in `vllm_neuron/model/registry.py`. That entry is a lazy
`"module:Class"` string, which a spawned worker can import by name. The plugin
registers its models inside the worker (`get_models()` in
`vllm_neuron/vllm/worker/neuron_worker.py`), after vLLM's own registry is in
place, so the entry replaces vLLM's EXAONE 4.0 class. This PR deliberately
does not make that edit: it crosses the community-owned, opt-in boundary that
the contrib guide draws, so it is the user's step. Then:

```bash
vllm serve LGAI-EXAONE/EXAONE-4.0-32B \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching
```

Then send a request:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "LGAI-EXAONE/EXAONE-4.0-32B",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

```python
from vllm import LLM, ModelRegistry, SamplingParams

# Copy contrib/models/exaone4/src/ into your own application package and
# import the factory from there (option 1 of contrib/README.md).
from your_package.exaone4 import Exaone4ForCausalLM

# Register the contrib architecture before you construct the engine. The call
# must *replace* vLLM's own Exaone4ForCausalLM, not just add an entry.
ModelRegistry.register_model("Exaone4ForCausalLM", Exaone4ForCausalLM)

llm = LLM(
    model="LGAI-EXAONE/EXAONE-4.0-32B",
    dtype="bfloat16",
    tensor_parallel_size=8,
    max_model_len=2048,
    max_num_seqs=4,
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The color of the sky is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

This works because the engine-core and worker processes are `fork`ed from your
process and inherit the registry you populated. It therefore depends on the
default start method: do not set `VLLM_WORKER_MULTIPROC_METHOD=spawn` with
this registration style, or the worker will resolve vLLM's own
`Exaone4ForCausalLM` and crash on `from_configs`, as described above. The
plugin's own `get_models()` has no EXAONE 4.0 entry (contrib models are opt-in
by design), so the worker's registration pass does not evict yours.

Prefill runs sequence-parallel, so a prompt length must be greater than the
tensor-parallel size and a multiple of it.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1` in the environment; the tiny end-to-end unit test
does exactly that. See the top-level `README.md` and `AGENTS.md` for the full
list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

| Instance | Status | Software |
| --- | --- | --- |
| Trn2 (trn2.48xlarge) | Working — see the validation report | `vllm-neuron` 0.31.0.1.2.0, vLLM 0.31.0, `VLLM_NEURON_BACKEND=neuron_native` |
| Trn3 | Not tested | — |

The Neuron SDK release is not part of the evidence we hold for this model. The
container images that the runs used pin the toolchain, and this README does not
list them.

## Example Checkpoints

- <https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B>

## Testing

The integration test serves the model through vLLM on Neuron and validates its
per-token logits against a HuggingFace reference:

```bash
pytest contrib/models/exaone4/test/integration/test_model.py \
    -v -s --timeout=7200
```

Set `EXAONE4_CHECKPOINT_PATH` to a local checkpoint directory to avoid a Hub
download. The test compiles one graph for a 32B model, so allow a long
timeout.

`test/_local_src.py` loads `src/` by file location, under the module name
`contrib_exaone4_src`, and each test module loads that helper by file location
too. Nothing goes on `sys.path`, so the tests do not need the repository root
and keep working after this folder is copied into an application — they only
need `vllm-neuron` itself installed, and several contrib models can be
collected in one pytest run without resolving each other's code.

### Unit tests (CPU mode)

```bash
VLLM_NEURON_CPU_MODE=1 EXAONE4_CHECKPOINT_PATH=/path/to/EXAONE-4.0-32B \
    pytest contrib/models/exaone4/test/unit/ -v
```

Two suites live here. `test_exaone4_loading.py` reads real checkpoint tensors
and skips when `EXAONE4_CHECKPOINT_PATH` is not set.
`test_tiny_exaone4_e2e.py` needs no checkpoint and no hardware: it writes a
random-weight tiny EXAONE 4.0 model and runs the full vLLM pipeline on CPU at
TP2 and TP1.

### Module tests

```bash
VLLM_NEURON_CPU_MODE=1 EXAONE4_CHECKPOINT_PATH=/path/to/EXAONE-4.0-32B \
    pytest contrib/models/exaone4/test/module/ -v -s --timeout=600
```

`test_exaone4_rope.py` is the one module test: it validates a single component
(`Exaone4RotaryEmbedding`) in isolation against the HuggingFace
`Exaone4RotaryEmbedding`, using the Llama3 scaling parameters of the real
config. It skips when `EXAONE4_CHECKPOINT_PATH` is not set.

The remaining component-level checks stay in `test/unit/` because they validate
weight-loading transforms rather than a component's forward pass against a
reference: the TP8 fused-QKV and o_proj loaders, and the padded MLP shards
(which the loading suite also compares numerically against the unpadded
checkpoint MLP). See the Exceptions section.

## Validation report

### Source and environment

Four tests passed for this model, across two plugin commits, so no single
commit covers the whole table:

| Test | Verdict | Cases | Nodes | Plugin commit |
| --- | --- | --- | --- | --- |
| `[vLLM Neuron Functional] EXAONE 4.0 32B BF16 Loading CPU` | PASS | 10 / 10 | 1 | `9bac9de66e9326a800658d109997ba89c697b748` |
| `[vLLM Neuron E2E] EXAONE 4.0 32B BF16 DI 1P1D Logits Serving TRN2` | PASS | 2 / 2 | 2 | `9bac9de66e9326a800658d109997ba89c697b748` |
| `[Performance] EXAONE-4.0-32B TP8 Native comparison b1/b4` | PASS | 2 / 2 | 1 | `f11b15f210826655986794298e77f8730c1f342b` |
| `[Accuracy] LGAI-EXAONE/EXAONE-4.0-32B Native GSM8K b1/b4` | PASS | 2 / 2 | 1 | `f11b15f210826655986794298e77f8730c1f342b` |

Every run used `vllm-neuron` 0.31.0.1.2.0 with vLLM 0.31.0 on the native
runtime backend (`neuron_native`), on `trn2.48xlarge`. The runs used more than
one container image. Our evidence does not record which test used which image.

The checkpoint is `LGAI-EXAONE/EXAONE-4.0-32B`. None of the tests pins a
revision: they resolve the checkpoint by path or Hub id, so the revision the
runs used is not part of our evidence.

### Model configuration

The architecture class is `Exaone4ForCausalLM` (factory in `src/factory.py`,
implementation in `src/model.py`), BF16 weights, no quantization. The model is
dense, so no run names an expert-parallel degree. TP8 is the only degree with
results: one TP8 engine on each node of the two-node disaggregated run (5 Q
heads and 1 KV head per rank), and TP8 single-node for the performance and
GSM8K runs. The disaggregated test constants set `--max-model-len 2048` with
`--max-num-batched-tokens 2048 --max-num-seqs 4`, which keeps the whole
context below the 4,096-token sliding window, so the runner never trims the
sliding-window block table. The performance run used `--max-model-len 4096`
with token buckets up to 4096 and sequence buckets `[1]` or `[1, 4]`.
Model-specific serving arguments are limited to `--dtype bfloat16` and
disabled prefix caching; the logit run also enables on-device greedy sampling
and raw-logit logprobs. Registration is the user's step, and for this model it
must override vLLM's own class; see Usage above.

### Component validation

The CPU loading test passed with 10 of 10 cases (commit `9bac9de6`). Eight of
those cases are the loading suite: registration and resolution, the config
mapping against the HF config (hybrid layer types, the NoPE full layers,
Llama3 RoPE scaling, the untied head), weight-mapping coverage of all 707
checkpoint tensors with shape and dtype checks, the TP8 fused-QKV and o_proj
loader transforms for ranks 0, 3 and 7 on a sliding and a full layer, the
padded-MLP shards (3424 values padded to 3456) checked both element-wise and
numerically against the unpadded checkpoint MLP, and RoPE against the HF
implementation. The remaining two cases are the tiny end-to-end suite: a TP2
run whose logits are compared three-way against an HF reference built from the
same random weights, and a TP1 run that asserts an `[exaone4_dispatch]` line
for every component of every layer in both phases. (That 8 + 2 split is
inferred from the case count and the two test files; our evidence records the
suite verdict and counts, not per-case identities.)

All ten cases ship here, now split across two folders: the RoPE case moved to
`test/module/test_exaone4_rope.py`, and the other seven loading cases plus the
two tiny end-to-end cases stay in `test/unit/`. The split changes where the
cases live, not what they assert.

The two-node run also asserted kernel dispatch: the prefill node must show
prefill lines and the decode node decode lines, for all 8 ranks and all 64
layers, with `kernel=True`. It passed as part of that suite's 2 of 2 cases.

### Numerical correctness

The disaggregated run is the logit evidence: a 1P1D deployment on two
`trn2.48xlarge` nodes, one TP8 engine per node with NIXL moving the KV cache
from prefill to decode, which compares the served logits of 16 generated
tokens per prompt against a HuggingFace reference (FP32 baseline plus a
same-dtype teacher-forced pass) with `multi_prompt_logit_validation`,
replicated to the 4-sequence batch. It reports the aggregate and the stricter
per-prompt result separately and asserts both. The observed verdict is PASS,
2 of 2 cases (one per node), at commit `9bac9de6`. We hold that verdict and
those counts only — no logit metric was read out.

The shipped `test/integration/test_model.py` uses the same tolerance map as
that run — `(1e-5, 0.025)` at top-5, `(1e-5, 0.04)` at top-50, `(1e-5, 0.06)`
at top-1000 and `(1e-5, 0.065)` over the full vocabulary — and keeps the
strict `1e-5` argmax bound. It has not itself been run in CI: it is a
single-node offline `LLM` run, where the evidence is a two-node disaggregated
server.

A GSM8K run also passed (2 of 2 cases, b1 and b4, 100 documents at
`max_tokens=1024`, commit `f11b15f2`), which means dataset-level accuracy was
exercised end to end. Read that verdict narrowly: the test's only score gate
is a 0.0 threshold on the flexible-extract exact match, so a PASS says the run
completed and produced a score, not that the score reached any particular
level. We never read the score itself, so this README states none.

### Serving validation

| Workload | Offline | Online |
| --- | --- | --- |
| 1,024 in / 512 out | No evidence | Evidence: 4 concurrent requests, asserted at 1,024 prompt and 512 completion tokens each, inside the disaggregated run (2/2 PASS, commit `9bac9de6`) |
| 8,192 in / 1,024 out | No evidence | No evidence |

The online 1,024 / 512 row is the required grid's first row. We have no
offline (`LLM.generate`) measurement at that shape, and no run at
8,192 / 1,024 in either mode — the latter would also exceed the context that
any recorded run configured. The performance test is a separate, narrower
workload: `vllm bench serve` with random data at 1,024 input and 100 output
tokens, 20 measured requests after 5 warmups, at TP8 and concurrency 1 and 4.
It passed (2 of 2 cases, commit `f11b15f2`); we hold no latency or throughput
figure from it, so none is stated here.

### Exceptions

- **Three of the port's test files are dropped.** The disaggregated
  (`test_exaone4_di.py`), GSM8K (`test_native_comparison_accuracy.py`) and
  benchmark (`test_native_comparison_benchmark.py`) suites are not shipped,
  together with their shared support module. Each imports test infrastructure
  that lives outside the model folder — `test/utils/simple_server`,
  `test/utils/network`, `test/utils/process`, `test/utils/bench`,
  `test/utils/metric_checks`, `test/vllm_neuron/utils/di_utils`,
  `test/vllm_neuron/utils/logit_test_utils` and
  `test/vllm_neuron/utils/test_prompts` — and the first also needs `mpirun`
  across two nodes with a NIXL side channel, while the other two need the
  `WORKLOAD_OUTPUT_RW` result-publishing contract. Vendoring that
  infrastructure would have meant copying a server launcher, a proxy, an MPI
  harness and a metric-publishing layer into the model folder, so the suites
  were dropped. Their results are still the evidence cited above, but a user
  cannot re-run them from this folder.
- **No logit or serving test in this folder reproduces the recorded
  deployment.** The recorded logit and serving evidence is 1P1D across two
  nodes. The shipped integration test is single-node and offline, and it has
  never been run in CI.
- **The GSM8K gate is 0.0.** As described under Numerical correctness, the
  accuracy PASS is not a quality bar. No GSM8K score appears in this README,
  for this reason and because we never read one.
- **The checkpoint resolver changed.** The port's loading test resolved the
  checkpoint through a shared test helper
  (`test/vllm_neuron/utils/logit_test_utils.get_model_checkpoint`) that does
  not ship here. The test now reads `EXAONE4_CHECKPOINT_PATH` and skips when
  it is unset or does not hold a checkpoint. The case count is unchanged, but
  the suite can now skip instead of running, which the recorded CI run did not
  do.
- **`PROMPTS_2` is vendored.** The two validation prompts came from
  `test/vllm_neuron/utils/test_prompts.py` and are copied verbatim into the
  integration test, as contrib/README.md sanctions.
- **The registry assertion changed meaning.** The port's test asserted that
  `vllm_neuron.model.registry.get_models()` maps the architecture to the
  factory. A contrib model is deliberately not in `get_models()`, so the
  shipped test registers the local factory with vLLM's `ModelRegistry` and
  asserts that the architecture resolves to it instead of to vLLM's own
  EXAONE 4.0 class. The tiny end-to-end test registers it the same way, which
  the in-tree version did not need to do.
- **The tiny test's conftest was inlined.** The port kept the CPU-mode
  environment overrides (`VLLM_ENABLE_V1_MULTIPROCESSING=0`,
  `VLLM_NEURON_MIN_KV_BUDGET_GIB=0`, `NKI_SIMULATOR=1`,
  `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`) in a `conftest.py` beside the test.
  They now live in an autouse fixture inside the test module, so the loading
  suite in the same directory is unaffected.
- **The tiny test's `forked` marker now always takes effect.** In the plugin's
  own test tree, `test/vllm_neuron/conftest.py` blocks `pytest-forked` unless
  `--forked` is passed, so the marker was inert by default. `contrib/` has no
  such conftest, so with `pytest-forked` installed each tiny end-to-end case
  now runs in a forked child. That is what the marker asks for, and the two
  cases were written to be process-isolated, but it is a behaviour change from
  the recorded run: the module-scoped fixtures are then built inside each fork,
  so the tiny checkpoint is written once per case instead of once per module.
- **Only one component test is module-shaped.** `test/module/` holds the RoPE
  case, which compares one component against its HuggingFace counterpart.
  The loader and padded-MLP cases stay in `test/unit/` because they check
  weight-loading transforms against checkpoint slices rather than running a
  component against a reference implementation. There is no module test that
  runs attention or the MLP forward pass against a HuggingFace module in the
  sense of `docs/design/accuracy/module_test_guidelines.md`; the full-model
  evidence (the tiny end-to-end suite and the recorded logit run) is what
  covers those components here.
- **Registration cannot be made spawn-safe and copy-out-safe at once.** Read
  from the plugin's code, not from any run: the offline `LLM(...)` path reaches
  the engine core and the workers by `fork` inheritance, so registering a class
  object from the test process is enough; the model is then resolved through
  vLLM's registry (`ModelRegistry.resolve_model_cls` in
  `vllm_neuron/vllm/worker/neuron_model_runner.py`). Any vLLM CLI entrypoint
  spawns instead, where a class object cannot cross the process boundary at
  all — only a lazy `"module:Class"` string can, and that needs a module
  importable by name in a fresh interpreter, which the `test/_local_src.py`
  convention deliberately avoids. So the folder is copy-out-safe for the
  offline API and for the tests, and serving needs option 2 of
  contrib/README.md. The in-repo precedent for out-of-tree registration
  (`test/vllm_neuron/accuracy/test_tensor_replacement_e2e.py`) is option 1 for
  the same reason. Because upstream vLLM ships this architecture name, a
  process that misses the registration does not report an unsupported
  architecture: it resolves vLLM's class and then fails on `from_configs`,
  which the Neuron runners call and vLLM's classes do not define. That is a
  confusing crash, not wrong output and not wrong numbers. No run in our
  evidence exercised a spawned engine core for this model, so that failure mode
  is a code reading, not an observation — and we could not inspect vLLM's class
  directly here, since vLLM is not installed in this checkout.
- **The test imports were converted to a local loader.** The port imported the
  modeling code as `contrib.models.exaone4.src...`, which resolves only with
  the repository root on `sys.path`. The tests now load `src/` by file location
  through `test/_local_src.py` under the module name `contrib_exaone4_src`, so
  the folder survives being copied out of the repository. The assertions are
  unchanged; the module name they key on is not.
- **Untested configurations.** Only TP8 has device results (TP1 and TP2 appear
  in the CPU tiny test); no other tensor-parallel degree, no Trn3, no
  quantized weights, no FP8 KV cache (the code has that path, with no
  evidence), no speculative decoding (likewise), no prefix caching (every run
  disabled it), and no context at or beyond the 4,096-token sliding window, so
  the window boundary and the block-table trimming above it are unexercised.
- **No performance figures.** A performance test passed, but we hold only its
  verdict and case counts, so this README states no throughput, latency or
  time-to-first-token number.
