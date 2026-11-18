# Phi3 (Text)

Serve a Phi-3-architecture text model (`microsoft/Phi-4-mini-instruct`, whose
`config.json` declares `Phi3ForCausalLM`) with vLLM on AWS Trainium through the
`vllm-neuron` plugin. The implementation is in `src/`, and `src/README.md`
describes the architecture, the partial LongRoPE handling and the per-feature
limits.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix and the
> validation report below record exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`). The plugin registers the Neuron platform automatically, but a
contrib architecture is not in the plugin's registry, so register
`Phi3ForCausalLM` yourself first — see [contrib/README.md](../../README.md),
"How do I register a custom model with vLLM?".

### Serve with the OpenAI-compatible server

`vllm serve` gives you no script in which to call `register_model` at all, so
this path needs option 2 in `contrib/README.md`: move `src/` to
`vllm_neuron/model/phi3/` and add `("Phi3ForCausalLM",
"vllm_neuron.model.phi3:Phi3ForCausalLM")` to `_MODELS` in
`vllm_neuron/model/registry.py`. That edit is the user's to make — it changes
plugin core code, so this contribution does not contain it. (There is a second
reason to expect option 1 not to carry the CLI: comments in
`vllm_neuron/vllm/model_runner_v2_policy.py` say the CLI starts an engine core
that does not inherit the starting process's runtime patches. We did not
confirm the CLI start method ourselves, and the only place in `vllm_neuron/`
that sets `VLLM_WORKER_MULTIPROC_METHOD` forces `spawn` for Qwen3-ASR alone.)
Then:

```bash
vllm serve microsoft/Phi-4-mini-instruct \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 4 \
    --disable-sliding-window \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{"neuron_config": {"on_device_sampling_config": {"all_greedy": true}}}'
```

`--disable-sliding-window` is recommended, not required. The recorded runs used
it. This implementation has no sliding-window attention path. The factory
accepts a window that is not smaller than `max_position_embeddings`.
Phi-4-mini declares a 262,144-token window and a 131,072-token position range,
so the window has no effect for this checkpoint.

Then send a request:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "microsoft/Phi-4-mini-instruct",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

Your script runs before the engine, so it can register the architecture itself
(option 1 in `contrib/README.md`). Copy `src/` into your application under a
name you can import — `phi3_src` below — and register the factory before you
construct `LLM(...)`:

```python
from vllm import ModelRegistry

# Your copy of contrib/models/phi3/src/.
from phi3_src import Phi3ForCausalLM

# Register before you construct LLM(...).
ModelRegistry.register_model("Phi3ForCausalLM", Phi3ForCausalLM)

from vllm import LLM, SamplingParams

llm = LLM(
    model="microsoft/Phi-4-mini-instruct",
    tensor_parallel_size=8,
    dtype="bfloat16",
    max_model_len=4096,
    max_num_batched_tokens=4096,
    max_num_seqs=4,
    disable_sliding_window=True,
    enable_prefix_caching=False,
    additional_config={
        "neuron_config": {"on_device_sampling_config": {"all_greedy": True}}
    },
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The color of the sky is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

Register the factory as a class object, as above, and not as a
`"module:Class"` string: vLLM resolves a string registration in a fresh
interpreter, which cannot import a package that was loaded by file location.

A class registration reaches the engine-core and worker processes only because
the offline path **forks** them: they inherit the registry your script
populated, and `NeuronWorker` then adds the plugin's own architectures, which
reassign only their own architecture keys and so cannot evict yours. This is
read from code, not from a run of this model: `VLLM_WORKER_MULTIPROC_METHOD`
defaults to `fork` in the vLLM 0.31.0 source (`envs.py`) that the recorded runs
used — the repository now pins `vllm==0.32.0`, which we did not read — a class
registration is inspected in-process
(`vllm/model_executor/models/registry.py`), and `vllm_neuron/vllm/platform.py`
forces `spawn` for Qwen3-ASR alone. `test/integration/test_model.py` skips
itself when the start method is not `fork`.

If the registration does not reach the process that builds the model, the
error will not name it. Upstream vLLM ships its own `Phi3ForCausalLM`
(`vllm/model_executor/models/`), so such a process resolves vLLM's class
instead of reporting an unsupported architecture, and then fails with an
`AttributeError` on `from_configs` — the constructor convention the Neuron
model runners call (`vllm_neuron/vllm/worker/neuron_model_runner.py`,
`neuron_model_runner_v2.py`) and vLLM's own classes do not define. So a missed
registration is a confusing crash, not a silently wrong model.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1`. See the top-level `README.md` and `AGENTS.md` for the
full list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

The recorded runs name a plugin version and a vLLM version, not a Neuron SDK
release, so the columns below are the versions we have evidence for.

| Instance / version | `vllm-neuron` 0.31.0.1.2.0 (vLLM 0.31.0) | Other versions |
| ------------------ | ---------------------------------------- | -------------- |
| Trn2 (trn2.48xlarge) | Working: TP 8 on one node, TP 4 across 2 nodes | Not tested |
| Trn3                 | Not tested                             | Not tested     |

## Example Checkpoints

- <https://huggingface.co/microsoft/Phi-4-mini-instruct> — the only checkpoint
  with recorded runs. The tests pin revision
  `cfbefacb99257ffa30c83adab238a50856ac3083`.

The implementation accepts a `Phi3ForCausalLM` checkpoint with standard or
LongRoPE scaling, BF16 weights and a sliding window no smaller than
`max_position_embeddings`. No other checkpoint was run.

## Testing

The three test levels run two different tensor-parallel sizes on purpose. The
integration test uses TP 8 on one node, the topology of the recorded accuracy
and performance runs. The unit and module tests keep the port's TP 4, the
topology of the CPU loading run and the multi-node functional run, so that they
stay the files whose recorded case counts (9 and 14) the validation report
below cites. Both sizes are valid for this implementation (see `src/README.md`,
"Parallelism").

Every test resolves the checkpoint through `test/phi3_utils.py`: it downloads
the pinned revision from the HuggingFace hub, or uses `PHI3_CHECKPOINT_PATH`
when that points at a checkpoint already on disk.

The integration test serves the model through vLLM on Neuron and validates its
logits against a HuggingFace reference. It needs 8 NeuronCores:

```bash
pytest contrib/models/phi3/test/integration/test_model.py \
    -v -s --timeout=7200
```

### Unit tests

`test/unit/test_phi3_cpu.py` runs the registration contract, the config parse,
the fused QKV and gate/up weight transforms, full weight coverage with the tied
LM head, partial LongRoPE against the HuggingFace rotary embedding, and
attention and full-model logits at TP 4. It needs no Neuron hardware:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/phi3/test/unit/ \
    -v -s --timeout=3600
```

### Module tests

`test/module/test_phi3_device_modules.py` compares layer-0 attention (flash
prefill, segmented prefill and decode) and the MLP with a HuggingFace
reference, using the three-way pattern in the
[module test guidelines](../../../docs/design/accuracy/module_test_guidelines.md),
and runs a full-model prefill in the serving pad layout. All four cases need
Neuron hardware and 4 ranks.

```bash
pytest contrib/models/phi3/test/module/ -v -s --timeout=14400
```

The recorded runs put those 4 ranks on 2 nodes, with a multi-node launcher that
lives in the plugin's test tree and is not part of this contribution. Set
`phi3_utils.DEVICE_NNODES = 1` to run all 4 ranks on one node, or set
`NODE_RANK` and the Neuron multi-node environment yourself on each node.

### Test layout

Beside the three test folders the guide describes, `test/` holds two files:

- `test/_local_src.py` — the loaders. `load()` loads `src/` by file location
  under the module name `contrib_phi3_src`, and `load_test_utils()` loads
  `test/phi3_utils.py` under `contrib_phi3_test_utils`. Each test file loads
  `_local_src.py` itself, also by file location, under
  `contrib_phi3_local_src`. Nothing goes on `sys.path` and no module name can
  collide with another contrib model's, so the tests keep importing this
  model's own code after the folder is copied out of the repository. Build any
  string module path (a patch target, a logger name) from `SRC_PACKAGE` rather
  than writing `contrib_phi3_src` out.
- `test/phi3_utils.py` — the port's own helper, vendored into this folder with
  the model. It holds the checkpoint resolution, the MPExecutor topology, the
  worker-side harnesses and the HuggingFace references. Both kept test files
  use most of it, so it sits at the `test/` root instead of being duplicated
  into `test/unit/` and `test/module/`.

This was verified by copying the folder outside the repository and running
`pytest --collect-only test` from inside the copy: it finds the 9 unit cases,
the 4 module cases and the 2 integration cases with no repository root in
`sys.path`. One pytest invocation can collect only one contrib model's
`test/integration/test_model.py` unless you pass `--import-mode=importlib`,
because every contrib model has a file with that name. Running one model's
tests at a time, as the commands above do, is unaffected.

## Validation report

### Source and environment

Two plugin commits carry the recorded results, so no single commit covers the
whole table:

- `60e32deb879ef75b67b33628e465f86488a1422e` — the CPU loading run and the
  2-node multi-node functional run.
- `dcd99e8278489f969560cd698d60afd411560e64` — the GSM8K accuracy run and the
  performance run.

Every run was on `trn2.48xlarge` with `vllm-neuron` 0.31.0.1.2.0, vLLM 0.31.0
and the `neuron_native` runtime backend. The checkpoint is
`microsoft/Phi-4-mini-instruct`; the test constants pin HuggingFace revision
`cfbefacb99257ffa30c83adab238a50856ac3083`.

### Model configuration

The architecture class is `Phi3ForCausalLM` (`src/factory.py` validates the
config and selects `src/model.py`). It is a dense model, so there is no expert
parallelism; the evidence names tensor parallelism only, and weights are BF16
(the implementation refuses any other dtype).

Two topologies are on record, and the test ids name the parallelism for only
one of them:

- TP 8 on one node, with a 4,096-token maximum model length, 4,096 batched
  tokens, `--max-num-seqs` 1 or 4, `--disable-sliding-window`, prefix caching
  off and async scheduling off. The performance test id names TP 8 outright;
  the accuracy run used the same server configuration.
- TP 4 across 2 nodes with 2 ranks each, with a 2,048-token maximum model
  length. This comes from the test constants of the CPU and functional runs,
  not from the test ids.

`src/README.md` first stated that TP 8 is not valid for this implementation,
which the recorded TP 8 runs contradict. That contradiction is now resolved in
favour of the runs: the card's claim was wrong, the card carries the
correction, and Exceptions below keeps both the original claim and the
reasoning. Registration is the only step outside this folder, and it is the
user's: either register the factory from your own script (offline) or add the
architecture to the plugin registry (online), as described under Usage.

### Component validation

| Run | Commit | Nodes | Cases |
|---|---|---|---|
| CPU Loading (TRN2) | `60e32de` | 1 | PASS, 9 of 9 |
| Multi-Node Functional (TRN2) | `60e32de` | 2 | PASS, 14 of 14 |

The nine CPU cases are exactly the nine cases of the file this folder ships as
`test/unit/test_phi3_cpu.py` (six test functions, one of which has four
parameter sets): registration contract, fused weight transforms, full weight
coverage, four LongRoPE ranges, CPU attention and CPU full-model logits. The
fourteen functional cases are seven per node: the four module cases that ship
as `test/module/test_phi3_device_modules.py` plus the port's three serving
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
of 0.03 (K=5), 0.05 (K=50), 0.08 (K=1000) and 0.09 (all) — the same tolerance
map the port's own tests used. That test has no recorded verdict.

Accuracy evidence for this model is the GSM8K run instead: `[Accuracy]
microsoft/Phi-4-mini-instruct Native GSM8K b1/b4` passed, 2 of 2 cases, 1 node,
at commit `dcd99e8`. We hold its verdict and case counts only — no GSM8K score
was read out of the measurement artifacts, so none appears here.

### Serving validation

Our required grid is 1,024 input / 512 output and 8,192 input / 1,024 output,
offline and online:

| Workload | Offline (`LLM(...)`) | Online (`vllm serve`) |
|---|---|---|
| 1,024 in / 512 out | No evidence | Evidence: the 2-node multi-node functional run (PASS, 14 of 14, commit `60e32de`) executed four concurrent chat requests of 1,024 input tokens and 512 output tokens, on both model runners. The test id does not name the shape; it comes from the test source that run executed. |
| 8,192 in / 1,024 out | No evidence | No evidence |

We do not have the 8,192 / 1,024 row for this model in either mode. The
recorded runs were configured with a 4,096-token maximum model length
(single node) and a 2,048-token one (2 nodes), so that workload could not have
run.

Two further online runs passed at commit `dcd99e8`, both on one node at TP 8
with batch 1 and batch 4: the GSM8K accuracy run (2 of 2 cases) and
`[Performance] Phi-4-mini-instruct TP8 Native comparison b1/b4` (2 of 2
cases). This README reports the performance run as existence and verdict only.
No latency or throughput number from it appears in this README.

### Exceptions

- **No logit-validation run on record.** No recorded test id identifies a
  logit-validation run for this model, and the evidence holds verdicts and case
  counts only. The shipped integration test does logit validation at the strict
  `1e-5` argmax bound, but it has no recorded verdict.
- **No GSM8K score.** The accuracy run's verdict (PASS) and case counts (2 of
  2) are all we hold.
- **No performance figures.** This README states the performance run's
  existence and verdict only. It publishes no latency or throughput number.
- **No 8,192 / 1,024 serving row**, offline or online, and **no offline
  1,024 / 512 row**. The only recorded serving shape is the online
  1,024 / 512 request inside the multi-node functional run.
- **Resolved TP discrepancy, read from the code and not from a run.**
  `src/README.md` (written with the port) said the decode kernel needs an even
  number of Q heads per rank, so TP 8 is not valid; the GSM8K and performance
  runs are recorded as passing at TP 8 on one node. We read the code and
  settled it in favour of the runs: the only head-count constraint on the
  decode path is per-rank `q_heads % kv_heads == 0` (the GQA head-split check
  in `vllm_neuron/functional/attention/attention_decode.py`, which runs only
  when a rank holds more than one KV head, and the matching assert inside the
  NKI `attention_block_tkg` kernel, whose only parity requirement is on
  `d_head`). At TP 4 a rank holds 2 KV heads, which makes that check read as
  "an even number of Q heads per rank" — the likely origin of the card's
  sentence. At TP 8 a rank holds 1 KV head and 3 Q heads, which satisfies it,
  and nothing in `config.py`, `factory.py` or `model.py` rejects TP 8. The
  card now states TP 1, 2, 4 and 8 and carries the correction; the original
  claim is kept there and here rather than deleted. Two limits on this
  resolution: it is static reading, not a run, and the recorded TP 8 runs were
  **served** runs whose source carried this model in `vllm_neuron/model/phi3/`
  with `Phi3ForCausalLM` listed in the plugin's `_MODELS` — in-tree
  registration, not the way a contrib user registers it. The factory in that
  source is identical to this folder's `src/factory.py`, so no TP check was
  lost in the move into `contrib/`, but no recorded run went through
  option-1 registration at TP 8. The shipped integration
  test reproduces that TP 8 topology through the offline `LLM(...)` path with
  option-1 registration, and it has no recorded verdict. The unit and module
  tests keep the port's TP 4.
- **Dropped: the port's 2-node serving test** (`device/test_phi3_device_serving.py`).
  It imports `test.vllm_neuron.utils.logit_test_flow`,
  `test.vllm_neuron.utils.logit_test_utils` and
  `test.vllm_neuron.utils.test_prompts` — shared helpers outside the model
  folder. Its logit-validation shape is what `test/integration/test_model.py`
  carries now, with locally defined prompts. Dropping it also drops the only
  run that exercised prefix caching and the V1-model-runner variant: no shipped
  test covers either.
- **Dropped: the port's GSM8K accuracy test** (`device/test_native_comparison_accuracy.py`)
  and **performance test** (`device/test_native_comparison_benchmark.py`), with
  their `native_comparison_support.py` module. They import
  `test.utils.simple_server`, `test.utils.bench` and `test.utils.metric_checks`,
  and they write to `WORKLOAD_OUTPUT_RW` against historical baselines — shared
  helpers and campaign infrastructure outside the model folder. Their results
  are the GSM8K and performance rows above.
- **Dropped: the multi-node launcher.** The recorded 2-node topology was
  started by `test/vllm_neuron/functional/collectives/launch_inter_node_a2av.py`
  in the plugin test tree, which is not part of this contribution. The shipped
  module tests need `NODE_RANK` and the Neuron multi-node environment set by
  hand, or `DEVICE_NNODES = 1` to put all 4 ranks on one node.
- **Rewritten assertion.** The port's CPU test asserted that
  `vllm_neuron.model.registry.get_models()` maps `Phi3ForCausalLM` to the
  factory. A contrib model is deliberately absent from that registry, so the
  shipped test checks the registration contract instead (the checkpoint's
  architecture string and the factory that `src/__init__.py` exports). The
  original registry assertion is no longer covered by anything here.
- **Rewritten prompt source.** The port built its prompts from the plugin's
  shared LongBench corpus (`test/vllm_neuron/utils/data/longbench_prompts.json`,
  several megabytes, outside the model folder). The vendored helper uses a
  local paragraph, repeated to the requested length. The shipped unit and
  module tests therefore run on different token IDs than the recorded runs did,
  and their HuggingFace references are recomputed for those IDs.
- **Rewritten checkpoint resolution.** The port fell back to
  `test.vllm_neuron.utils.logit_test_utils.get_model_checkpoint` when the hub
  download failed. The vendored helper keeps the pinned hub download and adds
  `PHI3_CHECKPOINT_PATH` for a checkpoint that is already on disk; the shared
  resolver is gone.
- **Case composition is inferred.** The mapping from the recorded case counts
  (9 and 14) to the test functions in this folder is read from the test
  sources; the evidence records only aggregate verdicts and counts.
- **Untested configurations.** Prefix caching and the segmented-prefill path in
  a served engine (the native-comparison runs set
  `--no-enable-prefix-caching`), context beyond 4,096 tokens, the model's
  declared 131,072-token position range and its LongRoPE long factor in
  serving, TP 1 and 2, speculative decoding, Trn3, and any plugin or vLLM
  version other than 0.31.0.1.2.0 / 0.31.0.
- **No other checkpoint.** `Phi3ForCausalLM` is the architecture string of
  several Phi-3-family checkpoints, and this folder is named for it, but
  `microsoft/Phi-4-mini-instruct` is the only checkpoint with a recorded run.
- **Option-1 registration depends on `fork`, read from the code and not from a
  run.** The offline example and `test/integration/test_model.py` register a
  class object with `ModelRegistry` and rely on the engine core and the workers
  inheriting that registry. That holds only under the default `fork` start
  method, and no recorded run of this model exercised it: the recorded TP 8
  runs were served runs, which register the architecture the in-tree way. The
  `"module:Class"` string form is not an alternative here — vLLM resolves a
  string registration in a fresh interpreter, which cannot import a package
  loaded by file location — so the class form, which the guide's option 1 and
  the one in-repo out-of-tree precedent
  (`test/vllm_neuron/accuracy/test_tensor_replacement_e2e.py`) both use, is the
  only form available. Serving needs option 2 for a simpler reason that holds
  whatever the start method is: `vllm serve` gives the user no script in which
  to call `register_model`. The integration test skips itself when
  `VLLM_WORKER_MULTIPROC_METHOD` is not `fork` rather than failing obscurely.
  Because upstream vLLM ships its own `Phi3ForCausalLM`, a process that misses
  the registration resolves vLLM's class instead of reporting an unsupported
  architecture, and then fails on `from_configs`, the constructor convention
  the Neuron model runners require and vLLM's class does not define — a
  confusing crash rather than a silently wrong model.
