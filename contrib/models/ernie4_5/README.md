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
`README.md` for installation), then install this folder:

```bash
pip install -e contrib/models/ernie4_5
```

The folder's `pyproject.toml` installs `src/` as the `ernie4_5_neuron` package
and declares a `vllm.general_plugins` entry point (`src/plugin.py`). vLLM runs
that entry point in every process it starts (the front end, the engine core
and each Neuron worker), so it registers `Ernie4_5ForCausalLM` everywhere, and
`vllm serve` and `LLM(...)` need no other setup. This is Option 3 in
[contrib/README.md](../../README.md). To remove the registration, run
`pip uninstall vllm-neuron-contrib-ernie4-5`.

Upstream vLLM ships its own `Ernie4_5ForCausalLM`, so the install replaces a
built-in entry of vLLM's model table rather than adding a new one. Without the
install, vLLM does not report an unsupported architecture: it picks its own
class, and the engine then fails in the Neuron model runner on `from_configs`,
a plugin convention that vLLM's own model classes do not define. If you see
that failure, check that the package is installed.

This implementation runs on the `neuron_native` route only, so set
`VLLM_NEURON_BACKEND=neuron_native` for every process, including the server.

### Serve with the OpenAI-compatible server

These are the flags of the serve smoke test in the validation report.
`--host 127.0.0.1` keeps the server reachable from this host only.

```bash
VLLM_NEURON_BACKEND=neuron_native vllm serve baidu/ERNIE-4.5-0.3B-PT \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching
```

Then send a request:

```bash
curl http://127.0.0.1:8000/v1/completions \
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

from vllm import LLM, SamplingParams

# The installed package registers Ernie4_5ForCausalLM in every vLLM process,
# so no register_model call is needed here.
llm = LLM(
    model="baidu/ERNIE-4.5-0.3B-PT",
    dtype="bfloat16",
    tensor_parallel_size=1,
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

Prefill runs sequence-parallel, so a prompt must be longer than the
tensor-parallel size and a multiple of it. The model raises a `ValueError`
otherwise.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1` in the environment. See the top-level `README.md` and
`AGENTS.md` for the full list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

This matrix shows the instance types and versions this folder was tested
with. The Trn2 result was produced on `trn2.48xlarge`.

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn3 | Not tested | Not tested |
| Trn2 | Working (TP 1) | Not tested |

The earlier runs in the validation report used vLLM Neuron 0.31.0.1.2.0 with
this implementation in-tree, not this folder, so that version is not listed
as tested.

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

The test registers the factory from `src/` in its own process, so it does not
need the `pip install`. It skips itself unless `VLLM_WORKER_MULTIPROC_METHOD`
is unset or `fork` (see Exceptions).

Set `ERNIE4_5_CHECKPOINT_DIR` to a local snapshot of the pinned revision to
test the exact checkpoint the results below were produced with. Otherwise the
tests download it.

### Unit tests (CPU mode)

`test/unit/` holds the architecture and config checks, the weight-loading
tests and a decode KV contract test, which checks the block table and the
write-slot tensor (dtype, shape, values) that decode passes to the attention
kernel. They run on CPU, without Neuron hardware.

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
process. `test/conftest.py` does two things: it forces
`VLLM_NEURON_BACKEND=neuron_native` (the only route this implementation
supports) before pytest imports a test module, and at the end of the session
it shuts down the plugin's `MPExecutor` worker pool, whose workers otherwise
outlive the last test and keep pytest from exiting.

One caveat is left, and the loader is not its cause: this folder uses the
conventional test file names (`test/integration/test_model.py`, as the template
ships it), which every contrib model shares, so collecting several contrib
models in one pytest session in the default `prepend` import mode hits pytest's
"import file mismatch" error on the duplicate basenames. Collecting several
contrib models in one run needs `--import-mode=importlib`; otherwise run one
model folder per invocation.

## Validation report

This section records the validation evidence for this contribution, newest
run first.

### vLLM Neuron 0.29.0.1.2.0 (2026-11-19)

#### Environment

| Image | Instance | TP | vLLM | vllm-neuron | torch | transformers | neuronx-cc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` | `trn2.48xlarge` (logical NeuronCore config 2) | 1 | `0.29.0` | `0.29.0.1.2.0` | `2.13.0` | `5.17.0` | `2.28.7631.0+8f7b0cf5` |

#### Results

| Test | Configuration | Result |
| --- | --- | --- |
| CPU unit tests | `VLLM_NEURON_CPU_MODE=1 pytest contrib/models/ernie4_5/test/unit` | 5 passed, 0 failed |
| `test_model_accuracy[1-2048]` | batch 1, `max_model_len` 2,048, 2 prompts x 16 tokens | PASSED (2/2 prompts) |
| `test_model_accuracy[4-2048]` | batch 4, `max_model_len` 2,048, 2 prompts x 16 tokens | PASSED (2/2 prompts) |
| `vllm serve` smoke | flags in Usage, after `pip install -e contrib/models/ernie4_5`; 3 completions at `max_tokens` 32, then the 3 prompts in one request at `max_tokens` 16; `temperature` 0 | Passed: 3/3 completions non-empty |

Not run on 0.29: module tests (`test/module/`), not part of this run (`test_components_multinode.py` needs two nodes; this run had one).

#### Logit validation

| Prompt | Overall Status | Divergence (≤ 1 ULP) | K5 (≤ 0.04) | K50 (≤ 0.04) | K1000 (≤ 0.06) | All (≤ 0.065) |
| --- | --- | --- | --- | --- | --- | --- |
| 1/2 | PASSED | 0.0000 | 0.0350 | 0.0388 | 0.0388 | 0.0601 |
| 2/2 | PASSED | 0.0000 | 0.0165 | 0.0165 | 0.0165 | 0.0165 |

No value over its bound, at batch 1 and in each of the 4 batch-4 rows (same values); maxima at generated token 1 (prompt 1) and token 4 (prompt 2); prompt 2 argmax diverged at token 2, accepted (Δ = 0.0000).

Test passed on two-way validation, 2/2 prompts (`multi_prompt_logit_validation`); informational checks that missed: aggregate σ-ratio 1.0885 (> 1.0), `agg_bc` FAIL, `agg_linf_1.2x` FAIL, `pp_static_0.03` 0/2, `pp_static_0.05` 0/2, `pp_tok_linf_1.5x` 1/2; the rest passed.

#### Serve smoke

`/v1/completions`, `max_tokens` 32, `temperature` 0; output as JSON strings.

| Prompt | First 60 characters |
| --- | --- |
| `The capital of France is` | `" the capital of the French Republic.\n\nThe capital of France "` ... |
| `1, 2, 3, 4, 5,` | `" 6, 7, 8, 9, 10, 11, 12, 13, 14,"` (full output) |
| `def fibonacci(n):` | `"\n\n    if n == 0:\n        return 0\n    elif n == 1:\n        r"` ... |

#### Changes for 0.29

- `pyproject.toml`, `src/plugin.py` (new): `vllm.general_plugins` entry point that registers `Ernie4_5ForCausalLM` in every vLLM process.
- `src/compat.py` (new): vendors `hoist_rope_parameters` and `build_per_head_block_table` (not in 0.29.0.1.2.0; copied unchanged from a later release), imported by `src/config.py` and `src/model.py`.
- `src/model.py`: decode casts the per-head KV write slots to `uint32` (with `int64` slots the runtime refused to load the decode NEFF, "invalid offset").
- `test/unit/test_decode_kv_contract.py` (new): checks dtype, shape and values of the decode write slots and per-head block table at TP 1; runs no kernel.
- `test/conftest.py`: session fixture that shuts down the plugin's `MPExecutor` worker pool, so pytest exits.
- `test/integration/test_model.py`: own full-vocabulary `generate_fn` (all `config.vocab_size` ids) replaces the plugin's `create_offline_vllm_generate_fn` (it requested `tokenizer.vocab_size` = 100,295 logprobs of 103,424 ids and stopped in `torch.stack`); `multi_prompt_logit_validation` unchanged.
- `test/integration/test_model.py`: K5 = 0.04 on 0.29 (template 0.03; the port used 0.015).

### Earlier runs: vLLM Neuron 0.31.0.1.2.0 (implementation in-tree)

These runs tested this implementation in-tree, at
`vllm_neuron/model/ernie4_5/`, registered through `get_models()`, before it
became this folder. Every run used `trn2.48xlarge`, plugin version
`0.31.0.1.2.0` with vLLM `0.31.0`, the `neuron_native` runtime backend, and
`baidu/ERNIE-4.5-0.3B-PT` at revision
`b565cf6caebdb7a1eadf00100857b1ed5e044f12`. Every result in this subsection is
a test verdict or a case count. No accuracy score, tolerance measurement,
latency or throughput figure was read out of these runs, so none is quoted.

| Test | Cases | Nodes |
| --- | --- | --- |
| Loading and components on CPU, BF16 | 6/6 passed, 0 failed | 1 |
| Two-node 1P1D logits, serving and components on Trn2, BF16 | 6/6 passed, 0 failed | 2 |
| Native-comparison serving performance at TP 1, batch 1 and batch 4 | 2/2 passed, 0 failed | 1 |
| Native-comparison GSM8K accuracy at batch 1 and batch 4 | 2/2 passed, 0 failed | 1 |

#### Model configuration

The architecture class is `Ernie4_5ForCausalLM`, served in bfloat16; the
factory rejects any other dtype. The performance run used TP 1, and the
disaggregated-inference and component test files each parametrize TP 8 and TP
1 and passed every case, so both degrees ran. No expert parallelism applies:
this is a dense model. The logit, serving and native-comparison runs set
`max_model_len` and `max_num_batched_tokens` to 2,048 with `max_num_seqs` 4.

Beyond the standard arguments, the model needs
`VLLM_NEURON_BACKEND=neuron_native`, a bfloat16 dtype, and prompts whose length
is a multiple of the tensor-parallel size (sequence-parallel prefill).

#### Component validation

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

#### Numerical correctness

Logit validation ran, inside the two-node 1P1D run: the port's
`test_di_logits_serving.py` teacher-forced two prompts for 16 tokens at batch
4 and compared raw per-token logits against FP32 and BF16 HuggingFace
references. The run passed, 6 of 6 cases, with no failures.

That run used the tolerance map `(1e-5, 0.015)` at top-5, `(1e-5, 0.04)` at
top-50, `(1e-5, 0.06)` at top-1000 and `(1e-5, 0.065)` over the full
vocabulary. The shipped `test/integration/test_model.py` is an offline,
single-node adaptation of that test, not a rerun of it (see Exceptions). It
keeps those bounds and the strict `1e-5` argmax-divergence bound, except at
top-5, which is 0.04 on vLLM Neuron 0.29.0.1.2.0.

The GSM8K accuracy run passed both of its cases, at batch 1 and at batch 4.
We hold its verdict only. We did not read a GSM8K score.

#### Serving validation

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

- The shipped `test/integration/test_model.py` is an offline single-node
  adaptation of the two-node 1P1D logit test of the earlier runs (same
  prompts, output length and TP 1), written for this contribution. It ran on
  vLLM Neuron 0.29.0.1.2.0 as recorded above, with a K5 bound of 0.04 where
  the 1P1D test used 0.015.
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
  It asserts the pair that gets registered instead — the architecture string
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
  registration. It is a confusing crash, not wrong output. The installed
  entry point registers the factory in every process.
- The integration test does not use the entry point. The tests load `src/` by
  file location under a synthetic module name, so they run from a copied-out
  folder, and vLLM inspects a `"module:Class"` registration by importing it in
  a fresh interpreter, which cannot import that name. The test therefore
  registers the class object in its own process, which reaches the engine core
  and the workers only when they are forks of it. It skips itself unless
  `VLLM_WORKER_MULTIPROC_METHOD` is unset or `fork`, so a registration problem
  cannot surface as the crash above mid-test.
- The statements about vLLM's built-in `Ernie4_5ForCausalLM`, the
  `from_configs` failure and the fresh-interpreter inspection of string
  registrations were read from vLLM 0.31.0 source, not reproduced in a run.
- The earlier runs used this implementation in-tree. The vLLM Neuron
  0.29.0.1.2.0 run used this folder: the tests loaded `src/` from it, and the
  serve smoke ran through its installed entry point.
- Untested in our evidence: Trn3; TP > 1 on vLLM Neuron 0.29.0.1.2.0 (TP 8
  ran only in the earlier in-tree runs); the module tests on vLLM Neuron
  0.29.0.1.2.0; the offline `LLM(...)` example with the installed package (the
  integration test registers in-process); prefix caching and chunked prefill
  (the segmented-attention path exists in `src/model.py`, and every run
  disabled prefix caching); speculative decoding (the forward pass has a
  rejection-sampler branch, never exercised); any quantized or FP8-KV
  configuration (none is implemented); LoRA and multimodal input (neither is
  implemented; `inputs_embeds` is rejected); the 8,192/1,024 serving row; the
  offline 1,024/512 serving row.
- `src/model.py` names data parallelism among its supported parallelism modes,
  but the folder contains no data-parallel wiring and our evidence covers
  none. Treat DP as unverified.
- No GSM8K score, latency or throughput figure appears in this README. The
  measured logit errors come from the vLLM Neuron 0.29.0.1.2.0 run; the
  earlier runs are reported as test verdicts and case counts only.
