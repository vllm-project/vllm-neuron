# ModernBERT (granite-embedding-small-english-r2)

This folder serves a ModernBERT text embedding model with vLLM on AWS Trainium
through the `vllm-neuron` plugin. The implementation under `src/` covers the
BF16 checkpoint of `ibm-granite/granite-embedding-small-english-r2`: a
12-layer bidirectional encoder that alternates full-attention and
sliding-attention layers, each with its own RoPE theta, LayerNorm without bias,
and a GeGLU MLP. It is a pooling model — CLS pooling, then L2 normalization —
not a causal LM: there is no LM head, no sampler and no decode phase, so the
registered class is `ModernBertModel` and not a `ForCausalLM`.
`src/README.md` describes the architecture and the kernel paths in detail.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix and the
> validation report below record exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`). The plugin auto-registers with vLLM, but a contrib model is not in
the plugin's model registry, so you register the architecture `ModernBertModel`
yourself. See [contrib/README.md](../../README.md) — "How do I register a custom
model with vLLM?".

Two engine settings are not optional for this model. It needs vLLM's pooling
runner (it rejects the generate runner), and it needs prefix caching off
(the attention is bidirectional). The checkpoint's `modules.json` has no
Normalize module, so vLLM would return raw CLS vectors; the pooler config below
sets the CLS pooling and the L2 normalization that the model card specifies.

The recorded runs served this model with 2 ranks (tensor parallelism 2) spread
over 2 `trn2.48xlarge` nodes, one rank on each node.

### Offline inference with the Python API

This is option 1 of contrib/README.md: copy this folder's `src/` into your
application, import the factory class, and register it before you construct the
engine.

```python
from vllm import LLM, ModelRegistry
from vllm.config import PoolerConfig

from your_package.factory import ModernBertModel

# Register before LLM(...): this process resolves the architecture while it
# builds the engine config, and its child processes inherit this registry.
ModelRegistry.register_model("ModernBertModel", ModernBertModel)

llm = LLM(
    model="ibm-granite/granite-embedding-small-english-r2",
    runner="pooling",
    tensor_parallel_size=2,
    dtype="bfloat16",
    max_model_len=2048,
    max_num_batched_tokens=2048,
    max_num_seqs=4,
    enable_prefix_caching=False,
    pooler_config=PoolerConfig(seq_pooling_type="CLS", use_activation=True),
)

for output in llm.embed(["The color of the sky is blue."]):
    print(len(output.outputs.embedding))
```

Offline `LLM(...)` works with this registration. `vllm serve` needs a
registration hook (see the next section). We read this from the plugin's code
paths, not from a run of this example.

`fork` inheritance carries your registration to the workers. Nothing
propagates your call explicitly. The plugin registers its own architectures
inside the worker process (`NeuronWorker.__init__` iterates `get_models()` in
`vllm_neuron/model/registry.py`). A contrib model is deliberately not in that
list. The worker then resolves the class through vLLM's registry
(`ModelRegistry.resolve_model_cls` in
`vllm_neuron/vllm/worker/neuron_model_runner_v2.py`).

The plugin pins the `mp` executor. Thus the topology is your process, then the
engine-core process, then the `MultiprocExecutor` workers. The start method
comes from `VLLM_WORKER_MULTIPROC_METHOD`. The plugin leaves it at vLLM's
default `fork` for every architecture but one unrelated model. A forked child
inherits the registry you populated. The pooling runner changes nothing here.
`llm.embed` drives the same engine-core and worker processes, and the same
`resolve_model_cls` call site, as a generate model.

If you set that variable to `spawn`, the children are fresh interpreters that
inherit no registry. Then this option stops working. In that case, register
from a `vllm.general_plugins` entry point in your own package. Every process
loads that hook at startup.

### Serve with the OpenAI-compatible server

`vllm serve` needs a registration hook, because option 1 above cannot serve
this model. There is no script of yours in which to call `register_model`.
Also, a vLLM CLI entrypoint starts the engine with `spawn`. The spawned
engine-core process runs no code of yours. Thus a registration from a user
process does not reach it in either form: a class object or a
`"module:Class"` string.

Serving needs one of these registration hooks:

- Option 2 of contrib/README.md. Move the modeling code into
  `vllm_neuron/model/modernbert/`. Then add the architecture to `_MODELS` in
  `vllm_neuron/model/registry.py`. The plugin registers it at worker startup.
- The same registration from the `vllm.general_plugins` entry point of your
  own vLLM plugin.

This PR does not make the option 2 edit. That edit changes the plugin's core
code. This crosses the community-owned, opt-in boundary that contrib/README.md
draws.

After you register the architecture that way, serve it with the flags of the
port's own two-node serving test. The command shows the node-0 half:

```bash
vllm serve ibm-granite/granite-embedding-small-english-r2 \
    --runner pooling \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --nnodes 2 --node-rank 0 --master-addr $NODE0_IP --master-port 29520 \
    --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --pooler-config '{"seq_pooling_type": "CLS", "use_activation": true}' \
    --host 127.0.0.1 --port 8000
```

Node 1 runs the same command with `--node-rank 1 --headless` and no `--port`.
Then request embeddings:

```bash
curl http://localhost:8000/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "model": "ibm-granite/granite-embedding-small-english-r2",
        "input": ["The color of the sky is blue."]
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

- <https://huggingface.co/ibm-granite/granite-embedding-small-english-r2> — the
  only checkpoint tested, at revision
  `2ab6fa8ea2d674564defd37171ae19079b864b33`, which the tests pin.

Other ModernBERT checkpoints are untested. The config rejects a checkpoint with
a non-`gelu` activation, any norm, attention or MLP bias, a non-default RoPE
type, or a layer type other than `full_attention` and `sliding_attention`; the
factory rejects a quantized checkpoint. A larger tensor-parallel degree needs a
head count that divides by it.

## Testing

The commands below are written from the repository root, but the tests do not
depend on it: each test loads this folder's `src/` by file location through
`test/_local_src.py`, under a module name unique to this folder, so the tests
work from any working directory and keep working after the folder is copied
into an application. The only exception is the multi-node launcher used by the
module tests, which is a script of this repository.

One pytest invocation that collects two contrib models at once needs
`--import-mode=importlib`: the template gives every contrib model the same test
file names (`test_model.py` and so on), and pytest's default import mode rejects
two test files that share a basename. Collecting this folder on its own needs
nothing extra.

The integration test serves the model and validates its embeddings against the
HuggingFace reference. It needs 2 Neuron cores reachable from the test process:

```bash
pytest contrib/models/modernbert/test/integration/test_model.py \
    -v -s --timeout=3600
```

### Unit tests (CPU mode)

`test/unit/test_loading.py` checks the config translation and the
registration, the per-query attention bounds against an independently built
mask, the TP 2 weight shards against the checkpoint tensors, two layers (one
full-attention, one sliding-attention) and the whole model on CPU with packed
sequences:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/modernbert/test/unit/test_loading.py -v
```

Set `MODERNBERT_CHECKPOINT` to a local snapshot of the checkpoint to skip the
download.

### Module tests

`test/module/` holds the device component tests and `common.py`, the helper
module that the unit, module and integration tests share (the HF reference, the
input builders and the MPExecutor worker harnesses). `test/_local_src.py` is the
file-location loader that every test uses to reach `src/` and `common.py`.

`test_device.py` runs TP 2 with one rank on each of 2 `trn2.48xlarge` nodes, so
every collective crosses the nodes. Start it on each node through the
repository's inter-node launcher, which sets `NODE_RANK`, `MASTER_ADDR` and
`NEURON_RT_ROOT_COMM_ID`:

```bash
python3 test/vllm_neuron/functional/collectives/launch_inter_node_a2av.py \
    -m pytest -v contrib/models/modernbert/test/module/test_device.py
```

See the [module test guidelines](../../../docs/design/accuracy/module_test_guidelines.md)
for the three-way accuracy pattern these tests follow.

## Validation report

### Source and environment

Both passing tests ran at one plugin commit, `60e32deb879ef75b67b33628e465f86488a1422e`,
so a single commit covers the whole table below. The runs used plugin version
`0.31.0.1.2.0` with vLLM `0.31.0` and the `neuron_native` runtime backend. The
base image was the vLLM Neuron dependencies image for plugin `0.31.0.1.2.0`.
The instance type was `trn2.48xlarge` throughout: one node for the CPU test and
two nodes for the device test.

The checkpoint is `ibm-granite/granite-embedding-small-english-r2`. The revision
`2ab6fa8ea2d674564defd37171ae19079b864b33` comes from the tests themselves
(`test/module/common.py`). The evidence we hold records the test ids, verdicts,
case counts, node counts, instance type and images. It does not name the
checkpoint revision separately.

The code in `src/` is the implementation from the snapshot where those tests
passed, unchanged. Every import it makes already resolves outside
`vllm_neuron/model/`, so it needed no rewrite.

### Model configuration

| Item | Value |
|---|---|
| Architecture class | `ModernBertModel` (the `architectures` field of the checkpoint); the implementation class is `ModernBertForEmbedding` |
| Model kind | Pooling (embedding) model: `is_pooling_model`, CLS sequence pooling, `attn_type = "encoder_only"`. No LM head, no sampler, no decode phase |
| Implementation | BF16; the factory rejects a `quantization_config` |
| Parallelism | TP 2, named by the device test id, as one rank on each of 2 nodes. 12 heads and intermediate 1536 give 6 heads and 768 channels per rank |
| Context length | The test constants state 1,024-token inputs, with a 2,048-token context and batched-token limit for the serving geometry |
| Registration | Not in `vllm_neuron/model/registry.py`. Register `ModernBertModel` with vLLM yourself (see Usage) |
| Model-specific serving args | `--runner pooling` (required), `--no-enable-prefix-caching` (required), `--dtype bfloat16`, and `--pooler-config '{"seq_pooling_type": "CLS", "use_activation": true}'`. The model also rejects an FP8 KV cache and chunked prefill with `max_num_batched_tokens < max_model_len` |

### Component validation

| Test id | Verdict | Cases | Nodes | Plugin commit |
|---|---|---|---|---|
| `[vLLM Neuron Model] ModernBERT granite-embedding-small-english-r2 CPU Loading TRN2` | PASS | 9 / 9 | 1 | `60e32de` |
| `[vLLM Neuron Model] ModernBERT granite-embedding-small-english-r2 TP2 2-Node Device TRN2` | PASS | 12 / 12 | 2 | `60e32de` |

The first id is the CPU-mode run of the port's loading tests, which ship here as
`test/unit/test_loading.py`: config and registration, the attention bounds for
four packing layouts, the TP 2 weight shards, a full-attention and a
sliding-attention layer, and the packed full model on CPU. The second is the
device run, which ships here as `test/module/test_device.py`: the attention of
a full layer and of a sliding layer, the GeGLU MLP, and the full model for one
1,024-token sequence and for two packed sequences with a padding tail. We hold
the aggregate verdict and case count of each run, not a per-file or per-case
breakdown, so the attribution of files to ids is by what the ids name and what
the snapshot contained.

Both sets compare three-way (FP32 HF reference, BF16 HF reference, Neuron)
against the HuggingFace `ModernBertModel` in eager mode, on the hidden states
and on the CLS embeddings. No error magnitudes or cosine values were read out of
the measurement artifacts.

### Numerical correctness

**There is no logit validation for this model, and there cannot be.** This is a
property of the model, not a gap in the runs or a shortcut in the test. The
checkpoint is an embedding (pooling) model: a bidirectional encoder with no LM
head, no sampler and no decode phase. It produces one pooled vector per input
and never a per-token logit row. Thus the logit shape of the contrib template
has nothing to act on:

- `multi_prompt_logit_validation` has no logits to compare.
- `max_logprobs=-1` and `logprobs_mode="raw_logits"` have no logprobs to widen
  or reformat.
- `on_device_sampling_config={"all_greedy": True}` has no sampler to make
  greedy.
- `get_cached_reference_goldens` has no greedy sequence to teacher-force along.

The template's shape would assert something that the model cannot produce. No
logit number appears below because none can exist, not because one is missing.

This is within the guide, not a deviation from it. contrib/README.md asks a PR
for "an accuracy test ... that validates your model by serving it through vLLM
on Neuron and checking the generated output (for example, logit validation or
dataset evaluation)". Logit validation is one example that the guide offers,
not the required metric. Thus, for a pooling model, the right reading is a
pooling-appropriate accuracy gate on what the model actually emits: its
embeddings. `test/integration/test_model.py` therefore keeps the template's
three-way structure (FP32 HF baseline, BF16 HF expected, BF16 Neuron actual)
and swaps only the metric.

**The cosine floor is the only accuracy gate this model has.** The metric is
the plugin's own pooling assertion,
`assert_pooling_close_three_way(task="embed", min_cos=0.99)`. It sets a
per-vector cosine floor against the HF reference. It also reports the cosine
of BF16 HF against FP32 HF. Thus a failure separates a port defect from the
BF16 precision floor. `0.99` is the port's own `MIN_COS`, the floor that its
device and serving tests use. The device component tests that passed asserted
the same floor on the hidden states and on the CLS embeddings. Cosine ranking
uses the embeddings, so the metric should bound their direction. There is no
second gate: no logit validation (above) and no dataset or retrieval-quality
evaluation (see Exceptions).

**We hold the verdicts, not a cosine.** The evidence behind the table above is
the PASS verdicts and the case counts (9 / 9 and 12 / 12). We never read a
cosine value, an error magnitude or a tolerance out of the measurement
artifacts. Thus this report gives none. `0.99` is the floor that the code
asserts, not a measured result.

The shipped integration test has not itself been run. It drives the engine
offline in one process and asks for embeddings through `llm.embed`. The
recorded device run compared module outputs on 2 nodes through MPExecutor.

### Serving validation

Our required grid is 1,024 in / 512 out and 8,192 in / 1,024 out, offline and
online. We have no evidence for any cell:

| Workload | Offline | Online |
|---|---|---|
| 1,024 in / 512 out | No evidence | No evidence |
| 8,192 in / 1,024 out | No evidence | No evidence |

Two qualifications. First, the output-token half of each row has no meaning for
this model: an encoder has no generation phase, so "512 out" and "1,024 out"
cannot be produced at all. The input half is what could apply. Second, the port
does ship an online serving test on the 1,024-token input side — 4 concurrent
1,024-token `/v1/embeddings` requests over 2 nodes, compared with the HF
reference in cosine space — but no passing test id names serving for this model,
and that test is not in this folder (see Exceptions). Nothing was ever run at
8,192 input tokens.

### Exceptions

Dropped tests:

- `test_serving.py` (online `/v1/embeddings`, 4 inputs of 1,024 tokens, TP 2
  over 2 nodes). It imports `test.utils.simple_server` — a 767-line shared
  server harness outside this folder — and the contrib layout has no slot for a
  serving test. No recorded test id names serving for this model, so dropping it
  loses no evidence.

Untested configurations and missing coverage:

- No serving evidence, offline or online, at any input length; and nothing at
  8,192 input tokens.
- No dataset accuracy or retrieval-quality evaluation (no MTEB or similar)
  appears in the evidence for this model. The cosine floor against the HF
  reference is the only accuracy gate.
- There is no logit validation, and there cannot be. "Numerical correctness"
  gives the reason. It also describes the only accuracy gate, the cosine
  floor, and the evidence we hold for it.
- The shipped integration test has never been run as shipped, and it differs in
  geometry from the recorded device run (one offline engine in one process
  versus one rank on each of 2 nodes).
- Only TP 2 was run. TP 1 and larger degrees are untested, and a degree that
  does not divide the 12 heads is rejected by the code.
- No evidence for Trn3, for any plugin or vLLM version other than
  `0.31.0.1.2.0` / `0.31.0`, or for any Neuron SDK release: the evidence does
  not record an SDK version.
- No recorded test id covers the classify pooling task (the plugin's classify
  assertion is not implemented and this checkpoint has no classifier head),
  quantization, an FP8 KV cache, prefix caching or chunked prefill. The last
  four are rejected by the code rather than untested.
- Case counts are complete for what ran (9 / 9 and 12 / 12, zero failures), so
  there is no case-count shortfall to report.

Changes to the port's tests, beyond moving them into this folder:

- Intra-model imports became file-location loads of this folder. The port's
  `test.vllm_neuron.model.modernbert.common` and
  `vllm_neuron.model.modernbert.{config,model}` imports are now
  `test/_local_src.py` loads of `test/module/common.py` and of
  `src/{config,model}.py`, under module names prefixed `contrib_modernbert`.
  No test puts the repository root on `sys.path` any more, so the folder also
  imports after it is copied out of the repository, and two contrib models
  loaded in one pytest session cannot collide in `sys.modules`.
- `test/integration/test_model.py` registers the factory **class object** with
  vLLM rather than a `"module:Class"` string. The string form is the only one
  that can survive a `spawn` start method, but it needs a module importable by
  name in a fresh interpreter, which the file-location convention above
  deliberately prevents; vLLM also inspects a string-registered class in a
  fresh subprocess, which would fail to import it. The two requirements are in
  genuine tension, and this folder resolves it in favour of the copy-out
  convention: the class form works for the offline `LLM(...)` path the test uses
  (fork inheritance), and no user-process registration of either form reaches a
  `vllm serve` engine (see "Serve with the OpenAI-compatible server"). This is
  read from the plugin's code paths, not from a run.
- `common.resolve_checkpoint` used the plugin's shared
  `test.vllm_neuron.utils.logit_test_utils.get_model_checkpoint` to resolve the
  model id. That helper is outside this folder, so the function now downloads
  the pinned revision with `huggingface_hub.snapshot_download`. The
  `MODERNBERT_CHECKPOINT` override is unchanged.
- `test_loading.py` asserted that `vllm_neuron.model.registry.get_models()`
  maps the architecture to the factory. A contrib model is not in that
  registry, so the test now registers the architecture with vLLM's
  `ModelRegistry` and asserts that vLLM holds it. The recorded 9-case run
  asserted the in-tree registry, so this one assertion differs from what ran.
