# Contrib Model: Granite (dense, text)

This folder contributes the Granite architecture (`GraniteForCausalLM`) to
`vllm-neuron`, so you can serve a text Granite checkpoint with vLLM on AWS
Trainium. It was contributed for `ibm/PowerLM-3b`, which is the only checkpoint
our evidence covers. The modeling code is in `src/`; the validation report at
the end of this file records exactly what was tested.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

Granite is a Llama-style dense decoder with a bias on every attention and MLP
projection and four extra scalars — an embedding multiplier, an attention
(softmax) multiplier, a residual multiplier and a logits divisor.
`src/README.md` describes the implementation and where those scalars apply.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md` for installation). The plugin auto-registers with vLLM, but a
contrib architecture is not in the plugin's registry, so you register it
yourself — see
[contrib/README.md](../../README.md) — "How do I register a custom model with
vLLM?". The two options there are not interchangeable for this model, and
which one you need depends on how the engine starts its subprocesses:

- **Option 1 (register the factory from your own script) covers the offline
  `LLM(...)` API below.** The plugin pins the `mp` executor and leaves the
  multiprocessing start method at `fork`, so the engine-core and worker
  processes inherit the `ModelRegistry` that your script populated, and the
  model runner then resolves the architecture through vLLM's registry. This
  is read out of the plugin code — `vllm_neuron/vllm/platform.py` (the
  executor pin, and a `spawn` override scoped to one unrelated model) and
  `vllm_neuron/vllm/worker/neuron_model_runner.py` /
  `neuron_model_runner_v2.py` (`ModelRegistry.resolve_model_cls`) — together
  with vLLM 0.31.0's `vllm/envs.py`, where `VLLM_WORKER_MULTIPROC_METHOD`
  defaults to `fork`. It is not read from a run of this folder. Pass the
  factory **class** to `register_model`, not a `"module:Class"` string: vLLM
  inspects a string entry by importing that module path in a fresh
  `sys.executable` subprocess, which cannot reach code loaded from a file
  location. `GraniteForCausalLM` is not one of the plugin's own
  `_MODELS` entries (`vllm_neuron/model/registry.py`), so the pass that
  re-registers those inside the worker does not overwrite your class.
- **Option 2 is what `vllm serve` needs.** A vLLM CLI entrypoint starts its
  engine core with `spawn`. So the fresh interpreter never runs your
  `register_model` call. A registered class object cannot cross `spawn` at
  all. Option 2 moves `src/` into `vllm_neuron/model/` and adds the
  architecture to `_MODELS`. This contribution deliberately does not make
  that edit, because it changes plugin core code. Expect the `spawn` failure
  to be confusing rather than explicit. Upstream vLLM ships this
  architecture name. So a process that never ran your registration resolves
  vLLM's class, and does not report an unsupported architecture. The process
  then fails on `from_configs`. The Neuron model runners call `from_configs`
  on whatever class they resolve, and vLLM's own classes do not define it.
  This comes from reading the code, not from a run:
  `vllm_neuron/vllm/worker/neuron_model_runner.py` and
  `neuron_model_runner_v2.py` (`ModelRegistry.resolve_model_cls` followed by
  `model_cls.from_configs(...)`).

The PowerLM-3b checkpoint stores FP32 weights and sets no dtype. The model
runs them in bfloat16: pass `--dtype bfloat16` so the engine agrees with the
weight loader.

### Serve with the OpenAI-compatible server

This needs the Option 2 registration above. Registering from a script does
not reach a server that the CLI starts.

```bash
vllm serve ibm/PowerLM-3b \
    --dtype bfloat16 \
    --tensor-parallel-size 4 \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4
```

Then send a request:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "ibm/PowerLM-3b",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

```python
from vllm import LLM, ModelRegistry, SamplingParams

# Copy contrib/models/granite/src/ into your own package first, then register
# the factory before you construct the engine. The engine's subprocesses are
# forked from this one, so they inherit this registration.
from your_package.granite import GraniteForCausalLM

ModelRegistry.register_model("GraniteForCausalLM", GraniteForCausalLM)

llm = LLM(
    model="ibm/PowerLM-3b",
    dtype="bfloat16",
    tensor_parallel_size=4,
    max_model_len=2048,
    max_num_seqs=4,
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The color of the sky is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

Prefill runs sequence-parallel, so a prompt must be longer than the
tensor-parallel size and a multiple of it.

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

- <https://huggingface.co/ibm/PowerLM-3b> (revision
  `38a624db4c8c9dd15d8bfb51845b7b177348e104`, the only checkpoint and revision
  with evidence)

Other checkpoints that declare the `GraniteForCausalLM` architecture should fit
this implementation, but none of them was tested here.

## Testing

The integration test serves the model through vLLM on Neuron and validates its
per-token logits against a HuggingFace reference. It needs Neuron hardware
(TP 4 fits on one `trn2.48xlarge`).

```bash
pytest contrib/models/granite/test/integration/test_model.py \
    -v -s --timeout=3600
```

Set `GRANITE_CHECKPOINT_DIR` to a local snapshot of the pinned revision to
test the exact checkpoint the results below were produced with. Otherwise the
tests download it.

### Unit tests (CPU mode)

`test/unit/` holds the architecture, config, revision and weight-loading
tests, plus a decoder-layer comparison on CPU ranks. They need no Neuron
hardware, but `test_loading.py` loads the full 3B checkpoint on 4 CPU ranks, so
give it memory and time.

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/granite/test/unit -v
```

### Module tests

`test/module/test_device_modules.py` runs attention, the MLP and a full
decoder layer on Neuron devices at TP 4 spread over two nodes, so every
collective crosses the node boundary. It compares each component against FP32
and BF16 HuggingFace references with the three-way pattern of the
[module test guidelines](../../../docs/design/accuracy/module_test_guidelines.md).
Run the launcher on both nodes with the topology in the environment:

```bash
# node 0
NODE_RANK=0 MASTER_ADDR=<node 0 ip> python3 \
    contrib/models/granite/test/module/launch_multinode.py
# node 1
NODE_RANK=1 MASTER_ADDR=<node 0 ip> python3 \
    contrib/models/granite/test/module/launch_multinode.py
```

The tests skip themselves when `MASTER_ADDR` is unset, so a single-node run
collects them and reports skips rather than failing.

The shared test helpers live once at the `test/` root: `test/_local_src.py`
loads this folder's `src/` by file location, and `test/granite_reference.py`
holds the pinned checkpoint, the raw-tensor reads and the HuggingFace
references. Each test module loads `_local_src.py` by file location too, under
a name prefixed for this model, so nothing goes on `sys.path` and two contrib
models in one pytest session cannot resolve against each other's `src/`.

Collecting more than one contrib model in a single pytest session still needs
`--import-mode=importlib`, or one pytest invocation per model folder. The
helpers no longer collide, but every model has a
`test/integration/test_model.py`, and the default `prepend` import mode
derives a module name from the file name, so the second folder's files clash
with the first folder's.

## Validation report

This section records what was validated for this contribution, and on which
configuration. Every result below is a test verdict or a case count. We did
not read any accuracy score, tolerance measurement, latency or throughput
figure from the test results, so this README quotes none.

### Source and environment

Four test runs passed for this model:

| Validation | Cases | Nodes |
| --- | --- | --- |
| CPU loading and CPU decoder-layer tests (`test/unit/`) | 5/5 passed, 0 failed | 1 |
| Device functional test, TP 4 across two nodes | 12/18 passed, 0 failed | 2 |
| Online serving performance, TP 4, batch 1 and batch 4 | 2/2 passed, 0 failed | 1 |
| GSM8K accuracy, batch 1 and batch 4 | 2/2 passed, 0 failed | 1 |

Every run used `trn2.48xlarge`, plugin version `0.31.0.1.2.0` with vLLM
`0.31.0`, and the `neuron_native` runtime backend. The runs did not all use
the same plugin source commit. The checkpoint is `ibm/PowerLM-3b` at revision
`38a624db4c8c9dd15d8bfb51845b7b177348e104`. The tests fingerprint its
`config.json` and safetensors index by SHA-256.

### Model configuration

The architecture class is `GraniteForCausalLM`, served in bfloat16; the
FP32 checkpoint is cast at load time. Every passing device and performance run
names TP 4. No expert parallelism applies: this is a dense model. The
multi-node full-model server specifications set `max_model_len` 2,048 with
`max_num_seqs` 4, while the native-comparison constants (the performance and
GSM8K runs) set `max_model_len` and `max_num_batched_tokens` to 4,096 with
token buckets up to 4,096. The CPU loading test loads the checkpoint on 4
ranks; the CPU decoder-layer test runs at TP 2.

Registration is the user's job, as described under "Usage" above and in
`contrib/README.md`; nothing outside this folder needs to change. Beyond the
standard arguments the model needs only `--dtype bfloat16` and prompts whose
length is a multiple of the tensor-parallel size (sequence-parallel prefill).

### Component validation

The CPU run passed all 5 of its cases with no failures. Those cases are the
tests that this folder ships in two files:

- `test/unit/test_loading.py`:
  - the architecture string and the factory's config rejection
  - the four Granite scalars and the geometry of the real config
  - the pinned revision and the 642-tensor index
  - a full-checkpoint load on 4 CPU ranks. It checks mapping coverage against
    every checkpoint tensor. It also compares the shards of two layers with
    the raw tensors.
- `test/unit/test_cpu_decoder_layer.py`: a decoder layer at TP 2 on CPU,
  prefill and one decode step, against FP32 and BF16 HuggingFace references.

The two-node device functional run at TP 4 reported 12 of 18 cases passed and
0 failed. So 6 cases neither passed nor failed. The port's multi-node launcher
runs two stages per node, so that run spans both device files:

- `test/module/test_device_modules.py`, which this folder ships. It holds
  three component cases: attention, MLP and a full decoder layer at TP 4
  across two nodes.
- The online full-model test, which this folder drops. It holds the
  reference-logit, prefix-reuse and 1,024/512 serving cases.

We never opened the per-case list. So we cannot say which cases those 6 were,
or which file each of the 12 passing cases came from. See Exceptions.

### Numerical correctness

A logit-validation case ran in that two-node run: the port's
`test_device_full_model.py` teacher-forced two prompts for 16 tokens at batch
4 against FP32 and BF16 HuggingFace references, through a served endpoint with
full-vocabulary raw logits and greedy on-device sampling. The run as a whole
passed with no failures, but because 6 of its 18 cases neither passed nor
failed and we never read the per-case list, we cannot confirm that the logit
case was among the 12 that passed.

That test passed no tolerance map, so it used the plugin default, and the
shipped `test/integration/test_model.py` uses the same values — `(1e-5, 0.011)`
at top-5, `(1e-5, 0.02)` at top-50, `(1e-5, 0.03)` at top-1000 and
`(1e-5, 0.05)` over the full vocabulary — keeping the strict `1e-5`
argmax-divergence bound. It is an offline, single-node adaptation of that
case, not a rerun of it: see Exceptions.

The GSM8K accuracy run passed both its cases, at batch 1 and batch 4. We hold
its verdict only. We did not read its GSM8K score.

### Serving validation

The serving grid covers 1,024 input / 512 output tokens and 8,192 input /
1,024 output tokens, offline and online:

| Workload | Offline | Online |
| --- | --- | --- |
| 1,024 in / 512 out | No evidence | Partial: a 4-concurrent-request case with exact prompt and completion token counts asserted ran inside the two-node device functional run, whose 12-of-18 case count leaves the per-case outcome unconfirmed. |
| 8,192 in / 1,024 out | No evidence | No evidence |

A separate online `vllm bench serve` performance run at TP 4 passed both of
its cases. It requested 1,024 input and 100 output tokens, with 20 measured
requests at batch 1 and batch 4. This README states the verdict only. It
quotes no latency or throughput figure.

### Exceptions

- The multi-node device functional run reported 12 of 18 cases passed with 0
  failed, which leaves 6 cases that neither passed nor failed. We never opened
  the per-case list, so the cause of that gap is unknown and this contribution
  does not claim 18 of 18. Everything the run covers — the device component
  cases, the online logit case, prefix reuse and the 1,024/512 serving case —
  inherits that uncertainty.
- The shipped `test/integration/test_model.py` has never been run. It is an
  offline single-node adaptation of the online logit case (same prompts,
  output length, batch, TP and tolerances), written for this contribution.
- Dropped: `test_device_full_model.py`, which held the online reference-logit,
  prefix-reuse and 1,024/512 serving cases and their node-1 follower
  counterparts. It needs shared harness modules outside the model folder
  (`test.utils.simple_server`, `test.vllm_neuron.utils.logit_test_utils`,
  `…server_utils`, `…test_prompts`), which would mean vendoring a two-node
  serving harness into the folder. The integration test replaces its logit
  half in offline form. Its prefix-reuse and serving halves have no
  replacement here. So you cannot reproduce those parts of the 12-of-18 run
  from this folder. Separately from the case-count gap, those cases served
  the model over two nodes. The contrib layout cannot reach that setup at
  all. A served endpoint needs the Option 2 registration (see "Usage"), not
  the script registration that this folder uses.
- Dropped: `test_native_comparison_accuracy.py` (GSM8K), the
  `test_native_comparison_benchmark.py` performance gate, and their shared
  `native_comparison_support.py`. They need `test.utils.simple_server`,
  `test.utils.bench` and `test.utils.metric_checks`. No replacement.
- `launch_multinode.py` lost its topology auto-discovery, which read the MPIJob
  hostnames through the shared `test.utils.network` helper. The shipped copy
  takes `NODE_RANK` and `MASTER_ADDR` from the environment and runs the one
  device stage that this folder still ships.
- The checkpoint resolver no longer consults the shared `get_model_checkpoint`
  helper. It now uses `GRANITE_CHECKPOINT_DIR` or
  `huggingface_hub.snapshot_download` at the pinned revision, with the same
  SHA-256 fingerprint checks.
- `test/unit/test_loading.py` no longer asserts that
  `vllm_neuron.model.registry.get_models()` maps the architecture to the
  factory, because a contrib model is deliberately absent from that registry.
  It asserts the pair a user has to register instead — the architecture string
  in `config.json` and the factory class in `src/`. The config, revision and
  full-load checks are unchanged.
- Nothing under `test/` is a Python package, and the test modules import the
  shared helpers by file location rather than by name. We tried packages and
  rejected them. A `test/__init__.py` makes this tree import as `test.unit`.
  But `test` is already the plugin's own top-level test package. The
  repository's root `conftest.py` imports that package in every pytest
  session. So collection fails with `No module named 'test.unit'`.
- Option 1 of `contrib/README.md` registers the factory class from your own
  script. It reaches the engine's subprocesses only because they are forked
  from the registering process. A vLLM CLI entrypoint uses a spawned engine
  core, which never runs that call. A class object cannot cross `spawn` at
  all. The lazy `"module:Class"` string does not solve this either, although
  the plugin uses that form for its own models. vLLM resolves such an entry
  by importing the module path in a fresh `sys.executable` subprocess. So the
  entry needs a module that a clean interpreter can import by name. The
  file-location loader that this folder uses deliberately avoids that.
  Registering the class object is therefore the only form that works here.
  It is what `test/integration/test_model.py` does. This comes from reading
  the code, not from a run. The sources are `vllm_neuron/vllm/platform.py`,
  `vllm_neuron/vllm/worker/neuron_worker.py` and the two
  `neuron_model_runner` files. They also include vLLM 0.31.0's
  `vllm/envs.py` (the `fork` default) and
  `vllm/model_executor/models/registry.py` (the two registration forms).
- All evidence was produced with this implementation in-tree, at
  `vllm_neuron/model/granite/`, registered through `get_models()`. The contrib
  layout — code in `contrib/models/granite/src/`, registered by the user — was
  not itself re-run in CI.
- Only `ibm/PowerLM-3b` has evidence. Any other Granite checkpoint, including
  larger ones and instruction-tuned ones, is untested here.
- Untested in our evidence:
  - Trn3
  - any Neuron SDK release number (none was recorded)
  - any tensor-parallel degree other than 4 on device
  - prefix caching and chunked prefill beyond the unconfirmed case above (the
    segmented-attention path exists in `src/model.py`)
  - speculative decoding (the forward pass has a rejection-sampler branch,
    never exercised)
  - any quantized or FP8-KV configuration (none is implemented)
  - LoRA and multimodal input (neither is implemented; `inputs_embeds` is
    rejected)
  - data parallelism (the decode path pins `attention_dp=1`)
  - the 8,192/1,024 serving row
  - the offline 1,024/512 serving row
- No GSM8K score, measured tolerance, latency or throughput figure appears in
  this README. We hold test verdicts and case counts only.
