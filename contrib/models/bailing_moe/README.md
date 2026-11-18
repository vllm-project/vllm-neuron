# BailingMoeV2 (Ling-mini-2.0)

This folder serves the BailingMoeV2 mixture-of-experts text model with vLLM on
AWS Trainium through the `vllm-neuron` plugin. The implementation under `src/`
covers the BF16 checkpoint of `inclusionAI/Ling-mini-2.0`: 20 decoder layers
(layer 0 dense, layers 1 to 19 with 256 routed experts, top-8 routing and one
shared expert), GQA attention with per-head QK RMSNorm, partial RoPE, and a
sigmoid router with group-limited top-k selection. `src/README.md` describes the
architecture and the kernel paths in detail.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix and the
> validation report below record exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`). The plugin auto-registers with vLLM, but a contrib model is not in
the plugin's model registry, so you register the architecture
`BailingMoeV2ForCausalLM` yourself. See
[contrib/README.md](../../README.md) — "How do I register a custom model with
vLLM?".

The recorded runs served this model with 8 ranks (tensor parallelism 8 with
expert parallelism 8) spread over 2 `trn2.48xlarge` nodes, 4 ranks on each node.

### Offline inference with the Python API

Copy `src/` into your own package, then register the factory before you
construct the engine (option 1 in contrib/README.md).

```python
from vllm import LLM, ModelRegistry, SamplingParams

# Your own copy of this folder's src/. The lazy
# "your_package.factory:BailingMoeV2ForCausalLM" string works the same way.
from your_package.factory import BailingMoeV2ForCausalLM

ModelRegistry.register_model("BailingMoeV2ForCausalLM", BailingMoeV2ForCausalLM)

llm = LLM(
    model="inclusionAI/Ling-mini-2.0",
    tensor_parallel_size=8,
    enable_expert_parallel=True,
    trust_remote_code=True,
    block_size=32,
    max_model_len=2048,
    max_num_seqs=4,
    enable_prefix_caching=False,
    additional_config={"neuron_config": {"quantization": "bf16"}},
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
for output in llm.generate(["The color of the sky is"], sampling_params):
    print(output.outputs[0].text)
```

This registration reaches the processes that actually build the model by
`fork` inheritance, and by nothing else. `LLM(...)` runs the engine core in a
child process and the 8 workers in children of that one, because the plugin
pins `distributed_executor_backend="mp"`
(`vllm_neuron/vllm/platform.py`). Those children come from
`VLLM_WORKER_MULTIPROC_METHOD`, whose default is `fork`, and a forked child
inherits the registry of this process, so the architecture is already there
when the worker adds the plugin's own models and when the runner calls
`ModelRegistry.resolve_model_cls`. Leave that variable at its default for
this path.

What goes wrong without that inheritance is worth knowing, because
`BailingMoeV2ForCausalLM` is also a built-in architecture of upstream vLLM: its
registry maps the same string to vLLM's own `bailing_moe` implementation. Your
registration replaces that entry in your process, and the forked children
inherit the replacement. A process that never ran your registration still
holds vLLM's entry, so it does not report an unknown architecture — it resolves
vLLM's implementation and then fails when the Neuron runner calls
`from_configs` on a class that does not define it. The Exceptions section
records all of this as read from the code, not from a run.

### Serve with the OpenAI-compatible server

`vllm serve` starts the engine itself, so there is no script of yours in which
to call `register_model` at all. That alone rules out option 1 for serving,
whatever the process start method is. (The plugin's own comments in
`vllm_neuron/vllm/platform.py` and `vllm_neuron/vllm/model_runner_v2_policy.py`
say the CLI starts its engine core with `spawn`, which would also defeat a
registration made in another process. We did not read that from vLLM's CLI
code.) Serving therefore needs option 2 of contrib/README.md (move `src/` into `vllm_neuron/model/` and add
the architecture to `_MODELS` in `vllm_neuron/model/registry.py`), or the same
registration from a vLLM plugin of your own. This PR does not make that edit:
it changes plugin core code, which the guide keeps outside a contrib
contribution. Once the architecture is registered that way, serve it with the
flags of the port's own two-node serving test, with the node-0 half shown:

```bash
vllm serve inclusionAI/Ling-mini-2.0 \
    --tensor-parallel-size 8 --enable-expert-parallel \
    --nnodes 2 --node-rank 0 --master-addr $NODE0_IP --master-port 29550 \
    --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 1024 \
    --block-size 32 \
    --trust-remote-code --no-enable-prefix-caching --no-async-scheduling \
    --additional-config '{"neuron_config": {"quantization": "bf16"}}' \
    --port 8000
```

Node 1 runs the same command with `--node-rank 1 --headless` and no `--port`.
Then send a request:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "inclusionAI/Ling-mini-2.0",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1` in the environment. See the top-level `README.md` and
`AGENTS.md` for the full list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

The verified evidence for this model records the plugin and vLLM versions and
the container images, not a Neuron SDK release number, so the matrix is keyed
by plugin version. Everything outside the one tested cell is untested, not
known-broken.

| Instance / plugin version | 0.31.0.1.2.0 (vLLM 0.31.0) | Other |
| ------------------------- | -------------------------- | ----- |
| Trn2                      | Working at TP 8 / EP 8     | Not tested |
| Trn3                      | Not tested                 | Not tested |

## Example Checkpoints

- <https://huggingface.co/inclusionAI/Ling-mini-2.0> — the only checkpoint
  tested, at revision `a810f6416bc4e1e29c9d7f271dd2fa7e56e71eab`, which the
  tests pin.

Other BailingMoeV2 checkpoints are untested. The factory rejects a quantized
checkpoint, a checkpoint with tied embeddings, softmax routing, linear bias, or
multi-token-prediction layers.

## Testing

The tests load this folder's own code by file location, through
`test/_local_src.py`, so they need neither the repository root on `sys.path`
nor a particular working directory, and they keep working when the folder is
copied out of the repository. The commands below use repository-relative
paths because that is where the folder ships.

One pytest caveat is not about this folder's own code: every contrib model
names its tests after the template (`test_model.py`, and so on), and pytest's
default `prepend` import mode keys a test module on its basename, so a single
invocation that collects two contrib folders errors on the duplicate names.
Collect one model per invocation, or pass `--import-mode=importlib`.

`test/integration/test_model.py` serves the model through vLLM and validates
its logits. It needs 8 Neuron cores reachable from the test process:

```bash
pytest contrib/models/bailing_moe/test/integration/test_model.py \
    -v -s --timeout=10800
```

`test/integration/test_full_model_prefill.py` is the other full-model test: it
runs a real 20-layer prefill of both logit prompts on 8 ranks and asserts that
every stage is finite. It drives the model directly, not through vLLM, so it
runs through the launcher like the module tests below.

### Unit tests (CPU mode)

`test/unit/test_cpu_loading.py` checks the registration, the config
translation, each custom weight loader, the partial-RoPE head permutation, the
router, and the full `load_weights` path on 8 ranks with EP 8. It runs on one
host with no Neuron device:

```bash
VLLM_NEURON_CPU_MODE=1 VLLM_NEURON_BACKEND=neuron_native \
    pytest contrib/models/bailing_moe/test/unit/test_cpu_loading.py -v
```

### Module tests

`test/module/` holds the per-component tests (attention, the MoE block, the
dense MLP), the kernel-route probe, and the two helper modules that the unit,
module and integration tests share (`utils.py` and the independent torch
reference `reference_model.py`). The component tests are the three-way accuracy
comparisons the guide describes. `test_kernel_routes.py` sits with them because
it is also per-component and also needs no assembled model — it calls the
kernel-eligibility predicate of each component with meta tensors in this
model's rank shapes — but it asserts kernel selection rather than numbers, so
it compares against no reference.

In CPU mode the 8 ranks run on one host and every operation takes its torch
fallback:

```bash
VLLM_NEURON_CPU_MODE=1 VLLM_NEURON_BACKEND=neuron_native \
    pytest contrib/models/bailing_moe/test/module/test_components.py -v
```

On a device the 8 ranks span 2 `trn2.48xlarge` nodes with 4 ranks on each node.
Start them on each node through the repository's inter-node launcher, which
sets `NODE_RANK`, `MASTER_ADDR` and `NEURON_RT_ROOT_COMM_ID`:

```bash
python3 test/vllm_neuron/functional/collectives/launch_inter_node_a2av.py \
    -m pytest -v contrib/models/bailing_moe/test/module/ \
    contrib/models/bailing_moe/test/integration/test_full_model_prefill.py
```

See the [module test guidelines](../../../docs/design/accuracy/module_test_guidelines.md)
for the three-way accuracy pattern these tests follow.

## Validation report

### Source and environment

Both passing tests ran at one plugin commit, `ecb3dfc394f00def7c090a825ba6bffbda322f2b`,
so a single commit covers the whole table below. The runs used plugin version
`0.31.0.1.2.0` with vLLM `0.31.0` and the `neuron_native` runtime backend. The
base image was the `vllm-neuron` dependencies image for that plugin version.
The instance type was `trn2.48xlarge` throughout: one node for the CPU test and
two nodes for the device test.

The checkpoint is `inclusionAI/Ling-mini-2.0`. The revision
`a810f6416bc4e1e29c9d7f271dd2fa7e56e71eab` comes from the tests themselves
(`test/module/utils.py`), which resolve that snapshot. The evidence we hold
records the test ids, verdicts, case counts, node counts, instance type and
images. It does not name the checkpoint revision separately.

The code in `src/` is the implementation from the snapshot where those tests
passed, with two changes. Neither change alters its behaviour. The in-tree
relative import `from ..kv_cache import ...` is now the absolute
`from vllm_neuron.model.kv_cache import ...`, because the contrib folder is not
inside `vllm_neuron/model/`. Also, `ruff format` rewrapped six expressions in
three places in `src/model.py`. This lets the folder pass the repository's own
`ruff format --check` hook. The test code has more changes. The Exceptions
section lists every change.

### Model configuration

| Item | Value |
|---|---|
| Architecture class | `BailingMoeV2ForCausalLM` (the `architectures` field of the checkpoint) |
| Implementation | BF16; the factory rejects any other dtype and any `quantization_config` |
| Parallelism | TP 8 with EP 8, named by both passing test ids, as 4 ranks on each of 2 nodes for the device run |
| KV block size | 32 (`--block-size 32`); the component tests allocate 64 blocks, so 2048 tokens of KV |
| Context length | The test constants state a 256-token context for the full-model prefill and logit geometry, and a 2048-token context with a 1024-token prefill bucket for the serving geometry |
| Registration | Not in `vllm_neuron/model/registry.py`. Register `BailingMoeV2ForCausalLM` with vLLM yourself (see Usage) |
| Model-specific serving args | `--enable-expert-parallel`, `--block-size 32`, `--trust-remote-code`, `--no-enable-prefix-caching`, `--no-async-scheduling`, and `neuron_config.quantization = "bf16"` |

### Component validation

| Test id | Verdict | Cases | Nodes | Plugin commit |
|---|---|---|---|---|
| `[vLLM Neuron Model] BailingMoeV2 Ling-mini-2.0 BF16 CPU Loading and Components TP8 EP8` | PASS | 17 / 17 | 1 | `ecb3dfc` |
| `[vLLM Neuron Model] BailingMoeV2 Ling-mini-2.0 BF16 Device Functional TP8 EP8 2-Node TRN2` | PASS | 32 / 32 | 2 | `ecb3dfc` |

The first id is the CPU-mode run of the port's loading and component tests,
which ship here as `test/unit/test_cpu_loading.py` and
`test/module/test_components.py`. The second id is the device run of the
port's device tests. These are the same component tests on 8 ranks across 2
nodes, plus `test/integration/test_full_model_prefill.py` and
`test/module/test_kernel_routes.py`. This PR moved the prefill test out of
`test/module/` (see Exceptions). We hold the aggregate verdict and case count
of each run, not a per-file or per-case breakdown. Thus we attribute files to
ids by what the ids name and what the snapshot contained.

For the first id, the case count also supports the attribution. Pytest
collects 7 cases from `test/unit/test_cpu_loading.py` and 10 from
`test/module/test_components.py`. Neither file carries a mark that CPU mode
deselects, and 7 + 10 is exactly the recorded 17. The 32 cases of the second
id over 2 nodes do not decompose that way from the files in this folder. Thus
its attribution rests on the id alone.

The component tests compare each component three-way (FP32 reference, BF16
reference, Neuron) against the independent torch transcription in
`test/module/reference_model.py`. On a device, they also assert that the
compiled prefill and decode graphs of every rank call the required NKI kernels.
Thus a torch fallback of a required operation fails the test. We did not read
error magnitudes out of the measurement artifacts.

### Numerical correctness

No recorded test id for this model names logit validation. The evidence
records the device run as "Device Functional". It holds no accuracy numbers of
any kind, only the verdicts and case counts above.

The shipped `test/integration/test_model.py` does perform logit validation
against the independent torch reference. It uses 2 prompts, 16 greedy tokens
and a 256-token context at TP 8 / EP 8. It asserts
`TOL_MAP = {"5": (1e-5, 0.011), "50": (1e-5, 0.02), "1000": (1e-5, 0.03), "all": (1e-5, 0.05)}`.
That is the `DEFAULT_TOLERANCE_MAP` of `vllm_neuron.accuracy`. The port's own
logit test selected this map by passing `tol_map=None`. The argmax bound stays
at the strict `1e-5` of the template. These are the bounds that the shipped
test asserts, not a measured result. No verdict for this test exists in our
evidence. Nobody ran it in the offline single-process shape that it ships in.

The passing runs give two checks, but neither is logit validation. The first is
the three-way component comparison above. The second is the full-model prefill
probe. It asserts that every stage of a real 20-layer prefill is finite on
every rank. The prefill covers both logit prompts, each pad-position mode of
the runner, and the server forward with the on-device sampler.

### Serving validation

Our required grid is 1,024 in / 512 out and 8,192 in / 1,024 out, offline and
online. We have no evidence for any cell:

| Workload | Offline | Online |
|---|---|---|
| 1,024 in / 512 out | No evidence | No evidence |
| 8,192 in / 1,024 out | No evidence | No evidence |

The port does ship a serving test for the first row: 4 concurrent requests of
1,024 input and 512 output tokens, online, over 2 nodes. But no passing test id
names serving for this model. That test is also not in this folder (see
Exceptions). The 8,192 / 1,024 row never ran for this model at all.

### Exceptions

Dropped tests:

- `test_logits.py` (full-model logit validation, online, 2 nodes) and its
  `multinode.py` harness. They import `test.vllm_neuron.utils.logit_test_utils`,
  `test.vllm_neuron.utils.test_prompts` and `test.utils.simple_server` — shared
  helpers outside this folder, the last of them 767 lines of server harness.
  `test/integration/test_model.py` re-expresses the same check offline.
- `test_serving.py` (1,024 in / 512 out, 4 concurrent requests, online, 2
  nodes). It needs the same `multinode.py` and `test.utils.simple_server`, so
  it cannot come along without them; `test/integration/` would otherwise have
  been its slot. No recorded test id names serving for this model, so dropping
  it loses no evidence.

Untested configurations and missing coverage:

- No logit validation and no dataset accuracy run (no GSM8K or similar) appears
  in the evidence for this model.
- Neither serving row of the required grid has evidence, offline or online.
- The shipped integration test has never been run as shipped. It also differs
  in geometry from the recorded runs: it builds one offline engine in one
  process, while the recorded device run drove TP 8 / EP 8 as 4 ranks on each of
  2 nodes.
- Only TP 8 with EP 8 was run. Other TP degrees, the TP (non-EP) path for the
  routed experts, and single-node 8-rank topologies are untested.
- No evidence for Trn3, for any plugin or vLLM version other than
  `0.31.0.1.2.0` / `0.31.0`, or for any Neuron SDK release: the evidence does
  not record an SDK version.
- No recorded test id covers quantization, the FP8 KV-cache path that exists in
  the code, on-device sampling, prefix caching, chunked prefill, LoRA or
  speculative decoding.
- Case counts are complete for what ran (17 / 17 and 32 / 32, zero failures),
  so there is no case-count shortfall to report.

Registration, read from the code and not from a run. No test id in our
evidence exercises any registration path for this model, so every statement
here is a reading of `vllm_neuron/vllm/platform.py`,
`vllm_neuron/vllm/worker/neuron_worker.py`, the two
`ModelRegistry.resolve_model_cls` call sites in the runners, and a copy of the
vLLM 0.31.0 sources. The plugin now pins `vllm==0.32.0`
(`requirements/core.txt`), which we did not read:

- `ModelRegistry.register_model(...)` in a user script reaches the engine core
  and the workers only because they are forked from it, as the Usage section
  describes. The guide's option 1 never says so. The `fork` default is
  vLLM's own (`VLLM_WORKER_MULTIPROC_METHOD` in its `envs.py`), and the only
  place the plugin sets that variable forces `spawn` for Qwen3-ASR alone.
- `vllm serve` cannot use option 1, first because the CLI gives the user no
  script in which to register anything. The plugin's comments also say the CLI
  starts its engine core with `spawn`, which would defeat a registration made
  elsewhere; that part is their claim, not something we read from vLLM's CLI.
  Either way serving needs option 2, which this PR does not perform.
- Upstream vLLM ships `BailingMoeV2ForCausalLM` as a built-in architecture
  (its registry maps the string to vLLM's own `bailing_moe` module), so a
  process without the user's registration resolves that class instead of
  reporting an unknown architecture. It then fails on `from_configs`, which
  the Neuron runners call and vLLM's class does not define. So the exposure is
  a confusing failure rather than wrong numbers — but only because the Neuron
  runner happens to require `from_configs`.
- The only `spawn`-safe form of registration is the lazy `"module:Class"`
  string, which vLLM resolves by importing that name in a fresh interpreter
  (`_LazyRegisteredModel.inspect_model_cls` runs `_run_in_subprocess`). The
  local-src loader deliberately gives this folder's code a module name
  (`contrib_bailing_moe_src.*`) that exists only in the process that loaded it,
  so the string form is unusable here and the tests register the class object,
  which `_RegisteredModel.from_model_cls` inspects in-process. The two
  requirements are in genuine tension: a copy of `src/` installed as a real
  package in the user's own application satisfies both, which is what the Usage
  section tells the user to do.

Changes to the port's tests, beyond moving them into this folder:

- Intra-model imports became file-location loads through the new
  `test/_local_src.py`, which loads `src/` as `contrib_bailing_moe_src` and the
  two shared helpers as `contrib_bailing_moe_test.*`. The originals were
  `test.vllm_neuron.model.bailing_moe[.reference_model]` and
  `vllm_neuron.model.bailing_moe.{config,model}`. An intermediate revision of
  this folder used `contrib.models.bailing_moe.*` namespace paths with a
  `sys.path` bootstrap rooted at the repository; those are gone, because they
  only import while the repository root is importable.
- `test/module/test_full_model_prefill.py` moved to `test/integration/`. It
  drives the assembled 20-layer model, which the guide's `test/module/`
  explicitly is not for ("validate one component ... in isolation, before you
  assemble the full model"). Only the path changed; the test is byte-identical
  apart from its import block and the run command in its docstring.
- Three shared helpers were vendored, which contrib/README.md sanctions:
  `FakeSafeSlice` (from `test/vllm_neuron/model/utils.py`) and the two prompts
  of `PROMPTS_2` (from `test/vllm_neuron/utils/test_prompts.py`) into
  `test/module/utils.py`, and the `dist_cleanup` fixture (from
  `test/vllm_neuron/conftest.py`) into the new `test/conftest.py`. The last one
  is also a fix: that conftest applies only under `test/vllm_neuron/`, so every
  test here that asks for `dist_cleanup` by name would have errored with
  "fixture not found". The recorded runs used the fixture from its original
  location, where the tests then lived.
- `test_cpu_loading.py` asserted that `vllm_neuron.model.registry.get_models()`
  maps the architecture to the factory. A contrib model is not in that
  registry, so the test now registers the architecture with vLLM's
  `ModelRegistry` and asserts that vLLM holds it. The recorded 17-case run
  asserted the in-tree registry, so this one assertion differs from what ran.
- `test/integration/test_model.py` registered the factory as the string
  `"contrib.models.bailing_moe.src.factory:BailingMoeV2ForCausalLM"` and put
  the repository root on `PYTHONPATH` for the engine's children. It now
  registers the class object that the loader returns, and fails early with an
  explanation if `VLLM_WORKER_MULTIPROC_METHOD` is not `fork`. This test has
  never been run in either shape.
