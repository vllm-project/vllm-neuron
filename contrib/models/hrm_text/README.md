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
> models under `vllm_neuron/model/`. The compatibility matrix and the
> validation report below record exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`). The plugin auto-registers with vLLM, but a contrib model is not in
the plugin's model registry, so you register the architecture
`HrmTextForCausalLM` yourself. See [contrib/README.md](../../README.md) — "How
do I register a custom model with vLLM?". Which option you need depends on how
you start the engine: option 1 (register from your own script) works for the
offline `LLM(...)` API, and only option 2 works for `vllm serve`. The two
subsections below say why.

The port supports tensor parallelism 1 and 2 only: HRM-Text has 12 attention
heads, and a larger degree leaves an odd number of heads on a rank, which the
decode kernel cannot take. The recorded runs used TP 2 with one rank on each of
2 `trn2.48xlarge` nodes.

### Offline inference with the Python API

This is option 1 of contrib/README.md. Copy this folder's `src/` into your
application, import the factory from there, and register it before you build
the engine:

```python
from vllm import LLM, ModelRegistry, SamplingParams

# src/ copied into your own package, so the factory imports by name.
from your_package.hrm_text.factory import HrmTextForCausalLM

ModelRegistry.register_model("HrmTextForCausalLM", HrmTextForCausalLM)

llm = LLM(
    model="sapientinc/HRM-Text-1B",
    tensor_parallel_size=2,
    dtype="bfloat16",
    max_model_len=2048,
    max_num_seqs=1,
    enable_prefix_caching=False,
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
for output in llm.generate(["The color of the sky is"], sampling_params):
    print(output.outputs[0].text)
```

That registration lives in your process, and it reaches the engine only by
process inheritance. The plugin pins the `mp` executor, so the engine core and
its workers are separate processes, and the worker resolves the architecture
through vLLM's registry (`ModelRegistry.resolve_model_cls`). Under the default
start method those processes are forks of yours and inherit your registry
entry; the worker then registers the plugin's own models, which does not evict
yours. With `VLLM_WORKER_MULTIPROC_METHOD=spawn` nothing is inherited and the
worker's lookup raises. This paragraph is read from the plugin code, not from a
recorded run.

### Serve with the OpenAI-compatible server

`vllm serve` cannot use the registration above. The CLI starts the engine core
as a freshly spawned process, so no `register_model` call of yours runs in it
and the worker's lookup raises. Serving needs the architecture registered at
plugin startup instead: option 2 of contrib/README.md (move `src/` into
`vllm_neuron/model/` and add it to `get_models()`), or the equivalent
registration from your own vLLM plugin. This contribution does not make that
edit — it would change plugin core code, which the guide keeps out of a contrib
PR.

The port's recorded serving run was in that arrangement, not in option 1: the
implementation was in the plugin tree then, and the snapshot's CPU test
asserted that the plugin's `get_models()` held `HrmTextForCausalLM` (see
Exceptions). So the flags below reproduce the engine arguments of that two-node
serving case, with the node-0 half shown, but only once the architecture is
registered as this subsection describes:

```bash
vllm serve sapientinc/HRM-Text-1B \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --nnodes 2 --node-rank 0 --master-addr $NODE0_IP --master-port 29610 \
    --max-model-len 2048 --max-num-batched-tokens 1024 --max-num-seqs 1 \
    --no-enable-prefix-caching \
    --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [1024], "num_seqs_buckets": [1]}}' \
    --port 8000
```

Node 1 runs the same command with `--node-rank 1 --headless` and no `--port`.
Then send a request:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "sapientinc/HRM-Text-1B",
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
| Trn2                      | Working at TP 2            | Not tested |
| Trn3                      | Not tested                 | Not tested |

## Example Checkpoints

- <https://huggingface.co/sapientinc/HRM-Text-1B> — the only checkpoint tested,
  at revision `22097cbcecdd1301afe30a19a3ee61b96a9863e5`, which the tests pin.

Other HRM-Text checkpoints are untested. The factory accepts bfloat16 only, and
rejects attention or MLP bias and any RoPE type other than the default.

## Testing

The tests load this folder's `src/` by file location. `test/_local_src.py`
loads it under a module name unique to this model, and each test module loads
that helper by location too, so nothing goes on `sys.path` and no test needs
the repository root. The folder therefore keeps working where you copy it, and
two contrib models collected in one pytest run cannot resolve to each other's
code. The commands below use repository-relative paths only because that is
where the files sit here.

The integration test serves the model and validates its logits. It needs 2
Neuron cores reachable from the test process. It registers the architecture
itself (option 1), so it depends on the default `fork` start method and skips
under `VLLM_WORKER_MULTIPROC_METHOD=spawn`:

```bash
pytest contrib/models/hrm_text/test/integration/test_model.py \
    -v -s --timeout=10800
```

### Unit tests (CPU mode)

`test/unit/test_hrm_text_cpu.py` checks the registration, the config parsing and
the KV slot plan, the load of every checkpoint tensor into the full model at
TP 2 against an independent split of the raw tensors, and one decoder layer with
sequence-parallel prefill against independent reference math:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/hrm_text/test/unit/test_hrm_text_cpu.py -v
```

Set `HRM_TEXT_MODEL_PATH` to a local copy of the checkpoint to skip the
download.

### Module tests

`test/module/` holds the device component tests and `common.py`, the helper
module that the unit, module and integration tests share (the reference math,
the MPExecutor worker harnesses and the NKI dispatch recorder).

`test_hrm_text_modules_multinode.py` runs TP 2 with one rank on each of 2
`trn2.48xlarge` nodes, so every collective crosses the nodes, and it asserts the
NKI kernel dispatch of each compiled graph. Start it on each node through the
repository's inter-node launcher, which sets `NODE_RANK`, `MASTER_ADDR` and
`NEURON_RT_ROOT_COMM_ID`:

```bash
python3 test/vllm_neuron/functional/collectives/launch_inter_node_a2av.py \
    -m pytest -v \
    contrib/models/hrm_text/test/module/test_hrm_text_modules_multinode.py
```

That launcher is a repository script outside this folder, and the test does not
import it. Without it, set `NODE_RANK`, `MASTER_ADDR` and
`NEURON_RT_ROOT_COMM_ID` (the same value on both nodes) yourself and run
`pytest` on the file directly.

See the [module test guidelines](../../../docs/design/accuracy/module_test_guidelines.md)
for the three-way accuracy pattern these tests follow.

## Validation report

### Source and environment

Both passing tests ran at one plugin commit, `297a07626398ee113449b3c4b9c4d5312e485d3b`,
so a single commit covers the whole table below. The runs used plugin version
`0.31.0.1.2.0` with vLLM `0.31.0` and the `neuron_native` runtime backend, on
`trn2.48xlarge`: one node for the CPU test and two nodes for the device test.

The runs used more than one container image. The evidence also records two
source commits (`0c3b7cbadcc1cd8eedfac1cde78907fad6502943` and the `297a0762`
above), but it does not link an image to a commit. The two passing tests are
both recorded at `297a0762`.

The checkpoint is `sapientinc/HRM-Text-1B`. The revision
`22097cbcecdd1301afe30a19a3ee61b96a9863e5` comes from the tests themselves
(`test/module/common.py`); the evidence we hold records the test ids, verdicts,
case counts, node counts, instance type and images, and does not name the
checkpoint revision separately.

The code in `src/` is the implementation from the snapshot those tests passed
at, unchanged: every import it makes already resolves outside
`vllm_neuron/model/`, so no rewrite was needed. `src/README.md` is the port's
own card, carried over from the plugin tree with one correction: its closing
example pointed at an `examples/` script that this folder does not ship, and it
now points at this README and the shipped tests.

### Model configuration

| Item | Value |
|---|---|
| Architecture class | `HrmTextForCausalLM` (the `architectures` field of the checkpoint) |
| Implementation | BF16; the factory accepts bfloat16 only |
| Parallelism | TP 2, named by both passing test ids, as one rank on each of 2 nodes for the device run. The port supports TP 1 and TP 2 only (12 heads; the decode kernel needs an even head count per rank) |
| KV cache | 128 slots (`16 x H_cycles x (L_cycles + 1)`), because each pass of a layer writes its own slot. The component tests use a block size of 32 |
| Context length | The test constants state a 256-token context for the logit geometry, and 2,048 (1,024 in + 512 out + 512 headroom) with a 1,024-token prefill bucket for the serving geometry |
| Registration | Not in `vllm_neuron/model/registry.py`. Register `HrmTextForCausalLM` with vLLM yourself (see Usage) |
| Model-specific serving args | None beyond `--dtype bfloat16` and the bucket pinning shown in Usage. PrefixLM attention through `token_type_ids` is not implemented, and the model has no quantized path |

### Component validation

| Test id | Verdict | Cases | Nodes | Plugin commit |
|---|---|---|---|---|
| `[vLLM Neuron Model] HRM-Text 1B Loading and CPU Components TRN2` | PASS | 3 / 3 | 1 | `297a076` |
| `[vLLM Neuron Model] HRM-Text 1B Multi-Node TP2 Components Logits Serving TRN2` | PASS | 14 / 14 | 2 | `297a076` |

The first id is the CPU-mode run of the port's loading and CPU component tests,
which ship here as `test/unit/test_hrm_text_cpu.py`: the registration and config
checks, the full TP 2 checkpoint load, and one decoder layer with
sequence-parallel prefill. The second id names three things, and the port's
device files cover exactly those three: the components in
`test/module/test_hrm_text_modules_multinode.py` (gated attention prefill,
attention decode on a paged cache, attention decode with the runner's partial
block table, and the SwiGLU MLP in prefill and decode), and the logits and
serving cases in the file this folder drops (see Exceptions). We hold the
aggregate verdict and case count of each run, not a per-file or per-case
breakdown, so the attribution of files to ids is by what the ids name and what
the snapshot contained.

The component tests compare each component three-way (FP32 reference, BF16
reference, Neuron) against reference math written from the raw checkpoint
tensors, and they also assert that each compiled graph calls the expected NKI
kernels, so a torch fallback of a required operation fails the test. No error
magnitudes were read out of the measurement artifacts.

### Numerical correctness

This model does have logit evidence: the device test id names Logits, and that
run is recorded PASS with 14 of 14 cases. The logit case in it decoded 16 greedy
tokens for 2 prompts in a 256-token context at TP 2 over 2 nodes and compared
the raw per-token logits with the HuggingFace reference through the plugin's
three-way golden flow (FP32 baseline plus a same-dtype teacher-forced pass). It
passed `tol_map=None`, which selects `vllm_neuron.accuracy`'s
`DEFAULT_TOLERANCE_MAP`, so the PASS was measured against
`{"5": (1e-5, 0.011), "50": (1e-5, 0.02), "1000": (1e-5, 0.03), "all": (1e-5, 0.05)}`
— a `1e-5` argmax (divergence) bound with those top-K bounds. We hold the
verdict only; no error values or divergence counts were read out of the
measurement artifacts.

The shipped `test/integration/test_model.py` asserts that same tolerance map,
written out explicitly, with the same checkpoint, the same 2 prompts, the same
16 output tokens, the same 256-token context and the same TP 2. It differs from
the recorded run in how it drives the engine: the recorded run used an online
`vllm serve` pair across 2 nodes, started by the plugin's shared logit-test flow,
which lives outside this folder (see Exceptions); the shipped test builds one
offline engine in one process. In that offline shape it has not been run.

### Serving validation

Our required grid is 1,024 in / 512 out and 8,192 in / 1,024 out, offline and
online:

| Workload | Offline | Online |
|---|---|---|
| 1,024 in / 512 out | No evidence | Evidence: inside the PASS of the 14-case device run |
| 8,192 in / 1,024 out | No evidence | No evidence |

The online 1,024 / 512 row is the Serving part of the device test id. The port's
case served one request of exactly 1,024 prompt tokens and 512 output tokens
through the production sampling path (no `all_greedy`), at TP 2 over 2 nodes,
and asserted the prompt and completion token counts, the `length` finish reason
and a non-empty completion. It is one request at concurrency 1, not a throughput
or latency measurement: no tokens-per-second, time-to-first-token or latency
figure was read, and none is claimed here. The 8,192 / 1,024 row was never run
for this model, and nothing was run offline.

That online case ran against a `vllm serve` pair, which no option-1
registration can serve (see Usage). Reproducing it needs the architecture
registered at plugin startup, which this contribution does not do.

### Exceptions

Dropped tests:

- `test_hrm_text_e2e_multinode.py`, both of its cases — the logit case and the
  1,024 in / 512 out serving case whose PASS the device test id records. It
  imports six helpers from `test.vllm_neuron.utils.logit_test_flow` (1,041 lines
  of shared online-server and golden flow, five of the six imports private),
  `test.vllm_neuron.utils.test_prompts` and `test.utils.network` — all outside
  this folder. Vendoring that flow into the model folder would copy most of the
  plugin's serving test harness, so the file is dropped.
  `test/integration/test_model.py` re-expresses the logit case offline, with the
  same checkpoint, prompts, output length, context and tolerance map. The
  serving case has no replacement in this folder: the contrib layout has no slot
  for a serving test.

Untested configurations and missing coverage:

- No dataset accuracy run (no GSM8K or similar) appears in the evidence for this
  model. The logit comparison above is the only accuracy gate.
- No offline serving evidence at any shape, and no evidence at all for
  8,192 in / 1,024 out.
- No throughput, latency or time-to-first-token figures exist in the evidence.
- The shipped integration test has never been run as shipped, and the recorded
  logit PASS was produced by the online two-node flow it replaces, not by this
  test.
- Only TP 2 was run. TP 1, which the code supports, is untested, and larger
  degrees are unsupported by the decode kernel.
- No evidence for Trn3, for any plugin or vLLM version other than
  `0.31.0.1.2.0` / `0.31.0`, or for any Neuron SDK release: the evidence does
  not record an SDK version.
- The container images of the passing runs cannot be pinned, as described in
  Source and environment.
- No recorded test id covers quantization (the port has no quantized path),
  PrefixLM attention through `token_type_ids` (not implemented), prefix caching,
  LoRA or speculative decoding.
- Case counts are complete for what ran (3 / 3 and 14 / 14, zero failures), so
  there is no case-count shortfall to report. Two of the 14 device cases belong
  to the dropped file, so the tests in this folder do not cover all 14.

Changes to the port's tests, beyond moving them into this folder:

- Imports of the port code became file-location loads.
  `vllm_neuron.model.hrm_text[.config,.model]` and `from .common import ...`
  became `_local_src.load(...)` and `_local_src.load_common()` calls, where
  `test/_local_src.py` loads `src/` and the shared `test/module/common.py`
  under module names unique to this model, and each test module loads that
  helper by file location. The recorded runs imported the in-tree package
  names, so every test file's import mechanism differs from what ran.
- `common.resolve_checkpoint` used the plugin's shared `get_model_checkpoint`
  helper to resolve the model id (in this tree that helper is
  `test/utils/storage/checkpoints.py`; the snapshot the port ran at reached it
  through the shared logit-test utilities). It is outside this folder either
  way, so the function now downloads the pinned revision with
  `huggingface_hub.snapshot_download`. The `HRM_TEXT_MODEL_PATH` override is
  unchanged.
- `test_hrm_text_cpu.py` asserted that `vllm_neuron.model.registry.get_models()`
  maps the architecture to the factory. A contrib model is not in that
  registry, so the test now registers the architecture with vLLM's
  `ModelRegistry` and asserts that vLLM holds it. The recorded 3-case run
  asserted the in-tree registry, so this one assertion differs from what ran.
- `src/README.md` is carried over from the plugin tree except for its closing
  example, which pointed at `examples/vllm_neuron/models/hrm_text/run.py`, a
  path this folder does not contain. It now points at this README's Usage
  section and the shipped tests.

Registration, read from the plugin code rather than from a run:

- `test/integration/test_model.py` registers a class object, which only a
  forked engine process inherits. The spawn-safe form is the lazy
  `"module:Class"` string, and that needs a module importable by name in a
  fresh interpreter — which loading `src/` by file location deliberately
  avoids. The two requirements are in tension, so the test skips when
  `VLLM_WORKER_MULTIPROC_METHOD` is not `fork` instead of pretending to cover
  that case. Neither arrangement was measured; both statements come from
  reading `vllm_neuron/vllm/platform.py`,
  `vllm_neuron/vllm/worker/neuron_worker.py` and the two model runners.
- The recorded serving case used `vllm serve`, which spawns its engine, so it
  cannot be reproduced through the registration this folder ships. Serving
  needs option 2 of contrib/README.md, an edit to plugin core code that this
  PR does not make.
