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
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`). Then install this folder, from the repository root:

```bash
pip install -e contrib/models/modernbert
```

`pyproject.toml` installs `src/` as the `modernbert_neuron` package, and
`src/plugin.py` is a `vllm.general_plugins` entry point that registers the
architecture `ModernBertModel` in every vLLM process: the front end, the
engine core and each Neuron worker (Option 3 in
[contrib/README.md](../../README.md)). `vllm serve` and `LLM(...)` need no
other setup. `pip uninstall vllm-neuron-contrib-modernbert` removes the
registration.

Two engine settings are not optional for this model. It needs vLLM's pooling
runner (it rejects the generate runner), and it needs prefix caching off
(the attention is bidirectional). The checkpoint's `modules.json` has no
Normalize module, so vLLM would return raw CLS vectors; the pooler config below
sets the CLS pooling and the L2 normalization that the model card specifies.

The tested configuration is tensor parallelism 2 on one `trn2.48xlarge`.

### Serve with the OpenAI-compatible server

These are the flags of the tested `vllm serve` run:

```bash
vllm serve ibm-granite/granite-embedding-small-english-r2 \
    --runner pooling \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --pooler-config '{"seq_pooling_type": "CLS", "use_activation": true}' \
    --host 127.0.0.1 --port 8000
```

Then request embeddings:

```bash
curl http://127.0.0.1:8000/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "model": "ibm-granite/granite-embedding-small-english-r2",
        "input": ["The color of the sky is blue."]
    }'
```

### Offline inference with the Python API

With this folder installed, vLLM registers the architecture itself, so the
script needs no `register_model` call:

```python
from vllm import LLM
from vllm.config import PoolerConfig

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

The integration test builds an engine with the same settings, with
`max_num_seqs` set to its batch size.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1` in the environment. See the top-level `README.md` and
`AGENTS.md` for the full list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

Everything outside the one tested cell is untested, not known-broken.

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn3 | Not tested | Not tested |
| Trn2 | Working (TP 2) | Not tested |

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
into an application. The tests register the architecture themselves, so they
do not need the `pip install -e` step. `test/conftest.py` shuts down the
plugin's `MPExecutor` worker pool at the end of the session, so pytest exits
after its summary.

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
every collective crosses the nodes. Run it on each node with `NODE_RANK` (0 or
1), `MASTER_ADDR` and `NEURON_RT_ROOT_COMM_ID` set for the two nodes:

```bash
pytest -v contrib/models/modernbert/test/module/test_device.py
```

The inter-node launcher named in the test's docstring is not part of this
repository.

See the [module test guidelines](../../../docs/design/accuracy/module_test_guidelines.md)
for the three-way accuracy pattern these tests follow.

## Validation report

### vLLM Neuron 0.29.0.1.2.0 (2026-11-19)

#### Environment

| Image | Instance | TP | vLLM | vllm-neuron | torch | transformers | neuronx-cc |
|---|---|---|---|---|---|---|---|
| `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` (Neuron SDK 2.33) | One `trn2.48xlarge`, LNC 2 | 2 | 0.29.0 | 0.29.0.1.2.0 | 2.13.0 | 5.17.0 | 2.28.7631.0+8f7b0cf5 |

#### Results

| Test | Configuration | Result |
|---|---|---|
| CPU unit tests (`test/unit`) | `VLLM_NEURON_CPU_MODE=1` | 9 passed, 0 failed |
| `test_model_embeddings[single_1024]` | TP 2, one 1,024-token input, test-side registration | PASSED |
| `test_model_embeddings[batch_700_300]` | TP 2, 700 + 300 tokens in one batch, test-side registration | PASSED |
| `vllm serve` smoke | Flags under Usage, after `pip install --no-deps -e contrib/models/modernbert` (entry-point registration) | Passed: 3 embeddings of dimension 384, all finite, related pair above unrelated pair |

Not run on 0.29: module tests (`test/module/test_device.py` needs 2 nodes; this run had one).

#### Embedding validation

| Case | Vectors | Overall Status | Neuron vs FP32 HF, min cos (floor 0.99) | Neuron vs FP32 HF, mean cos | BF16 HF vs FP32 HF, min cos (not gated) |
|---|---|---|---|---|---|
| `single_1024` | 1 | PASSED | 0.999995 | 0.999995 | 0.999982 |
| `batch_700_300` | 2 | PASSED | 0.999993 | 0.999993 | 0.999976 |

Gate: per-vector cosine against FP32 HF, `assert_pooling_close_three_way(task="embed", min_cos=0.99)`. No vector below the floor.

#### Serve smoke

`/v1/embeddings`, one request with 3 inputs; cosine of the first input against the other two.

| Pair | Cosine |
|---|---|
| "The capital of France is Paris." and "Paris is the capital city of France." | 0.980422974235414 |
| "The capital of France is Paris." and "Bananas are yellow fruit." | 0.6922163716769365 |

#### Changes for 0.29

- `pyproject.toml`, `src/plugin.py`: new; install `src/` as `modernbert_neuron` and register `ModernBertModel` as `modernbert_neuron.factory:ModernBertModel` from a `vllm.general_plugins` entry point.
- `src/factory.py`: keyword-only `vllm_config` and `prefix` constructor arguments, `forward(input_ids, positions, ...)` passing both to the implementation, `embed_input_ids` stub.
- `src/model.py`: GeGLU MLP runs torch matmuls and `torch.ops.aten.gelu.default` (exact erf GELU) instead of `NF.mlp`; `nkilib` import removed.
- `test/conftest.py`: new; session fixture that shuts down the plugin's `MPExecutor` worker pool so pytest exits.

### Model configuration

| Item | Value |
|---|---|
| Architecture class | `ModernBertModel` (the `architectures` field of the checkpoint); the implementation class is `ModernBertForEmbedding` |
| Model kind | Pooling (embedding) model: `is_pooling_model`, CLS sequence pooling, `attn_type = "encoder_only"`. No LM head, no sampler, no decode phase |
| Implementation | BF16; the factory rejects a `quantization_config` |
| Parallelism | TP 2 on one node for the 0.29.0.1.2.0 run. 12 heads and intermediate 1536 give 6 heads and 768 channels per rank |
| Context length | 1,024-token inputs (and 700 + 300 in one batch), with a 2,048-token context and batched-token limit |
| Registration | Not in `vllm_neuron/model/registry.py`. `pip install -e contrib/models/modernbert` registers `ModernBertModel` through a `vllm.general_plugins` entry point (see Usage) |
| Model-specific serving args | `--runner pooling` (required), `--no-enable-prefix-caching` (required), `--dtype bfloat16`, and `--pooler-config '{"seq_pooling_type": "CLS", "use_activation": true}'`. The model also rejects an FP8 KV cache and chunked prefill with `max_num_batched_tokens < max_model_len` |

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

This is within the guide, not a deviation from it. contrib/README.md asks a PR
for "an accuracy test ... that validates your model by serving it through vLLM
on Neuron and checking the generated output (for example, logit validation or
dataset evaluation)". Logit validation is one example that the guide offers,
not the required metric. For a pooling model, the right reading is a
pooling-appropriate accuracy gate on what the model actually emits: its
embeddings. `test/integration/test_model.py` therefore keeps the template's
three-way structure (FP32 HF baseline, BF16 HF expected, BF16 Neuron actual)
and swaps only the metric.

**The cosine floor is the only accuracy gate this model has.** The metric is
the plugin's own pooling assertion,
`assert_pooling_close_three_way(task="embed", min_cos=0.99)`. It sets a
per-vector cosine floor against the FP32 HF reference. It also reports the
cosine of BF16 HF against FP32 HF, so a failure separates a port defect from
the BF16 precision floor. `0.99` is the port's own `MIN_COS`, the floor that
its device tests also assert on the hidden states and on the CLS embeddings.
Cosine ranking uses the embeddings, so the metric should bound their
direction. There is no second gate: no logit validation (above) and no dataset
or retrieval-quality evaluation (see Exceptions).

### Serving validation

| Mode | Inputs | Evidence |
|---|---|---|
| Offline, `LLM.embed` | One 1,024-token input; 700 + 300 tokens in one batch | Integration test PASSED on 0.29.0.1.2.0 |
| Online, `vllm serve` `/v1/embeddings` | 3 short sentences in one request | Serve smoke passed on 0.29.0.1.2.0 |
| Online at 1,024 input tokens | — | No evidence |
| Any input above the 2,048-token context | — | No evidence |

There is no throughput or latency measurement. An encoder has no generation
phase, so output-token workloads do not apply to this model.

### Exceptions

Dropped tests:

- `test_serving.py` of the earlier port (online `/v1/embeddings`, 4 inputs of
  1,024 tokens, TP 2 over 2 nodes). It imports a 767-line shared server
  harness outside this folder, and the contrib layout has no slot for a
  serving test. No recorded run covered it, so dropping it loses no evidence.

Untested configurations and missing coverage:

- No online serving at 1,024 input tokens, nothing above a 2,048-token
  context, and no throughput or latency measurement.
- No dataset accuracy or retrieval-quality evaluation (no MTEB or similar)
  appears in the evidence for this model. The cosine floor against the HF
  reference is the only accuracy gate.
- There is no logit validation, and there cannot be. "Numerical correctness"
  gives the reason and describes the cosine floor.
- `test/module/test_device.py` (one rank on each of 2 nodes) was not run on
  vLLM Neuron 0.29.0.1.2.0.
- Only TP 2 was run. TP 1 and larger degrees are untested, and a degree that
  does not divide the 12 heads is rejected by the code.
- No evidence for Trn3, or for the current code on any vLLM Neuron version
  other than 0.29.0.1.2.0.
- No run covers the classify pooling task (the plugin's classify assertion is
  not implemented and this checkpoint has no classifier head), quantization, an
  FP8 KV cache, prefix caching or chunked prefill. The last four are rejected
  by the code rather than untested.

Changes to the earlier port's tests, beyond moving them into this folder:

- Intra-model imports became file-location loads of this folder:
  `test/_local_src.py` loads `test/module/common.py` and `src/{config,model}.py`
  under module names prefixed `contrib_modernbert`. No test puts the repository
  root on `sys.path`, so the folder also imports after it is copied out of the
  repository, and two contrib models loaded in one pytest session cannot
  collide in `sys.modules`.
- `test/integration/test_model.py` registers the factory class object with
  vLLM before it builds the engine. The class comes from the file-location
  loader, so the test needs no install.
- `common.resolve_checkpoint` used a shared helper of the plugin's test tree to
  resolve the model id. That helper is outside this folder, so the function now
  downloads the pinned revision with `huggingface_hub.snapshot_download`. The
  `MODERNBERT_CHECKPOINT` override is unchanged.
- `test_loading.py` asserted that `vllm_neuron.model.registry.get_models()`
  maps the architecture to the factory. A contrib model is not in that
  registry, so the test now registers the architecture with vLLM's
  `ModelRegistry` and asserts that vLLM holds it. The 0.29.0.1.2.0 run asserted
  the shipped form.
