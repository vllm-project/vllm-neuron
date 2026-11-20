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
`README.md` for installation), then install this folder:

```bash
pip install -e contrib/models/granite
```

The install adds a `vllm.general_plugins` entry point (`src/plugin.py`) that
registers `GraniteForCausalLM` in every vLLM process — the front end, the
engine core and each Neuron worker — so `vllm serve` and `LLM(...)` need no
other setup. `pip uninstall vllm-neuron-contrib-granite` removes the
registration.

The PowerLM-3b checkpoint stores FP32 weights and sets no dtype. The model
runs them in bfloat16: pass `--dtype bfloat16` so the engine agrees with the
weight loader.

### Serve with the OpenAI-compatible server

These are the flags of the tested serve run:

```bash
vllm serve ibm/PowerLM-3b \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --tensor-parallel-size 4 \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching
```

`--no-enable-prefix-caching` is required on vLLM Neuron 0.29: prefix caching
(on by default) needs segmented prefill, which the plugin does not turn on when
`--max-num-batched-tokens` is not smaller than `--max-model-len`, as here.

Then send a request:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "ibm/PowerLM-3b",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

```python
from vllm import LLM, SamplingParams

# The pip install above registers GraniteForCausalLM; no register_model call
# is needed.
llm = LLM(
    model="ibm/PowerLM-3b",
    dtype="bfloat16",
    tensor_parallel_size=4,
    max_model_len=2048,
    max_num_batched_tokens=2048,
    max_num_seqs=4,
    enable_prefix_caching=False,
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The color of the sky is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

The integration test runs these engine arguments, plus its revision pin,
compile buckets and logit-validation settings.

Prefill runs sequence-parallel, so the prefill token count after padding to a
bucket must be larger than the tensor-parallel size and a multiple of it
(`src/model.py` raises otherwise). The prompt itself can be shorter: the serve
smoke in the validation report sent 5-, 6- and 14-token prompts at TP 4.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1` in the environment. See the top-level `README.md` and
`AGENTS.md` for the full list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

The matrix is keyed on the Neuron SDK of the container and the vLLM Neuron
plugin version it ran. Trn2 was tested on `trn2.48xlarge` at TP 4.

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn3 | Not tested | Not tested |
| Trn2 | Working (TP 4) | Not tested |

## Example Checkpoints

- <https://huggingface.co/ibm/PowerLM-3b> (revision
  `38a624db4c8c9dd15d8bfb51845b7b177348e104`, the only checkpoint and revision
  with evidence)

Other checkpoints that declare the `GraniteForCausalLM` architecture should fit
this implementation, but none of them was tested here.

## Testing

The integration test serves the model through vLLM on Neuron and validates its
per-token logits against a HuggingFace reference. It needs Neuron hardware
(TP 4 fits on one `trn2.48xlarge`). It registers the model class itself, so it
does not need the `pip install` above.

```bash
pytest contrib/models/granite/test/integration/test_model.py \
    -v -s --timeout=3600
```

Set `GRANITE_CHECKPOINT_DIR` to a local snapshot of the pinned revision to
test the exact checkpoint the results below were produced with. Otherwise the
tests download it.

### Optional: unit tests

`test/unit/` holds the architecture, config, revision and weight-loading
tests, plus a decoder-layer comparison on CPU ranks. They need no Neuron
hardware, but `test_loading.py` loads the full 3B checkpoint on 4 CPU ranks, so
give it memory and time.

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/granite/test/unit -v
```

### Optional: module tests

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
`test/conftest.py` shuts down the plugin's worker pool when the session ends,
so pytest exits after its summary.

Collecting more than one contrib model in a single pytest session still needs
`--import-mode=importlib`, or one pytest invocation per model folder. The
helpers no longer collide, but every model has a
`test/integration/test_model.py`, and the default `prepend` import mode
derives a module name from the file name, so the second folder's files clash
with the first folder's.

## Validation report

This section records what was validated for this contribution, and on which
configuration. The 0.29.0.1.2.0 run below quotes the logit errors from its test
log. No GSM8K score, latency or throughput figure appears in this README.

### vLLM Neuron 0.29.0.1.2.0 (2026-11-18)

#### Environment

| Image | Instance | TP | vLLM | vllm-neuron | torch | transformers | neuronx-cc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` | `trn2.48xlarge` | 4 | 0.29.0 | 0.29.0.1.2.0 | 2.13.0 | 5.17.0 | 2.28.7631.0+8f7b0cf5 |

#### Results

| Test | Configuration | Result |
| --- | --- | --- |
| CPU unit tests (`test/unit/`) | `VLLM_NEURON_CPU_MODE=1` | 5 passed, 0 failed |
| `test_model_accuracy[4-2048]` | TP 4, batch 4, `max_model_len` 2,048, 2 prompts × 16 decoded tokens against FP32 and BF16 HuggingFace references | PASSED (per-prompt: 1/2 prompts within bounds; aggregate σ-ratio 0.9358 ≤ 1.0) |
| `vllm serve` smoke | Folder installed with `pip install -e`, flags under "Usage"; 3 completions at `max_tokens` 32, then 1 request with the 3 prompts at `max_tokens` 16 | Passed (3/3 prompts); the batched request returned the first 16 tokens of the same completions |

Not run on 0.29: module tests (`test/module/`), which need two nodes; this run used one.

#### Logit validation

| Prompt | Overall Status | K5 (0.011) | K50 (0.02) | K1000 (0.03) | All (0.05) | Divergence (1 ULP) |
| --- | --- | --- | --- | --- | --- | --- |
| 1, "The Amazon rainforest is ..." | PASSED (64 of 64 tokens) | 0.0104 | 0.0104 | 0.0104 | 0.0110 | 0.0000 |
| 2, "The human genome contains ..." | FAILED | 0.0112 (> 0.011) | 0.0123 | 0.0136 | 0.0159 | 0.0000 |

Prompt 2 over K5 at generated token 6 (0.0112 > 0.011); all four batch rows report the same maxima.
No argmax divergence. Test passed on aggregate σ-ratio 0.9358 ≤ 1.0 (`multi_prompt_logit_validation`).

#### Serve smoke

`/v1/completions`, `max_tokens` 32, `temperature` 0.

| Prompt | Prompt tokens | First 60 characters of the completion |
| --- | --- | --- |
| `The capital of France is` | 5 | `" Paris.\n\nThe capital of Germany is Berlin.\n\nThe capital of S"` |
| `1, 2, 3, 4, 5,` | 14 | `" 6, 7, 8, 9, 10, 11, 12, 13, 14,"` (whole completion) |
| `def fibonacci(n):` | 6 | `"\n    if n <= 0:\n        return 0\n    elif n == 1:\n        re"` |

#### Changes for 0.29

- `src/compat.py` (new): `hoist_rope_parameters`, `build_per_head_block_table` and `scatter_new_kv`, absent from 0.29.0.1.2.0, copied unchanged from a later release.
- `src/config.py`, `src/model.py`: import those helpers from `.compat` instead of `vllm_neuron.model.config_utils` and `vllm_neuron.model.kv_cache`; nothing else in `src/` changed.
- `pyproject.toml`, `src/plugin.py` (new): `vllm.general_plugins` entry point (see "Usage").
- `test/conftest.py` (new): session fixture that shuts down the plugin's `MPExecutor` worker pool; no test changed.

### Exceptions

- Dropped: `test_device_full_model.py`, which held the online reference-logit,
  prefix-reuse and 1,024/512 serving cases and their node-1 follower
  counterparts. It needs shared harness modules outside the model folder
  (`test.utils.simple_server`, `test.vllm_neuron.utils.logit_test_utils`,
  `…server_utils`, `…test_prompts`), which would mean vendoring a two-node
  serving harness into the folder. The integration test replaces its logit
  half in offline form. Its prefix-reuse and serving halves have no
  replacement here.
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
  It asserts the pair that `src/plugin.py` registers instead — the
  architecture string in `config.json` and the factory class in `src/`. The
  config, revision and full-load checks are unchanged.
- Nothing under `test/` is a Python package, and the test modules import the
  shared helpers by file location rather than by name. We tried packages and
  rejected them. A `test/__init__.py` makes this tree import as `test.unit`.
  But `test` is already the plugin's own top-level test package. The
  repository's root `conftest.py` imports that package in every pytest
  session. So collection fails with `No module named 'test.unit'`.

### Not tested

- Any checkpoint other than `ibm/PowerLM-3b`, including larger and
  instruction-tuned Granite checkpoints.
- Trn3.
- This folder on any plugin version other than 0.29.0.1.2.0.
- Any tensor-parallel degree other than 4 on device.
- The module tests (`test/module/`) on 0.29.0.1.2.0.
- Prefix caching and chunked prefill. Every 0.29.0.1.2.0 configuration
  disables prefix caching and sets `max_num_batched_tokens` equal to
  `max_model_len`. The segmented-attention path exists in `src/model.py`.
- Speculative decoding (the forward pass has a rejection-sampler branch,
  never exercised).
- Any quantized or FP8-KV configuration (none is implemented).
- LoRA and multimodal input (neither is implemented; `inputs_embeds` is
  rejected).
- Data parallelism (the decode path pins `attention_dp=1`).
- The 1,024/512 and 8,192/1,024 serving workloads.
