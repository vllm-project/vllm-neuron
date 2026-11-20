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
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`), then install this folder:

```bash
pip install -e contrib/models/bailing_moe
```

The install adds a `vllm.general_plugins` entry point (`src/plugin.py`) that
registers `BailingMoeV2ForCausalLM` in every vLLM process: the front end, the
engine core and each Neuron worker. `vllm serve` and `LLM(...)` then find the
architecture with no other setup. vLLM also has its own
`BailingMoeV2ForCausalLM`; this registration replaces it while the package is
installed.

The tested configuration is tensor parallelism 8 with expert parallelism 8 on
one `trn2.48xlarge` (see the validation report).

### Serve with the OpenAI-compatible server

```bash
vllm serve inclusionAI/Ling-mini-2.0 \
    --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 8 --enable-expert-parallel \
    --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 1024 \
    --block-size 32 \
    --trust-remote-code --no-enable-prefix-caching --no-async-scheduling \
    --additional-config '{"neuron_config": {"quantization": "bf16"}}'
```

Then send a request:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "inclusionAI/Ling-mini-2.0",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="inclusionAI/Ling-mini-2.0",
    tensor_parallel_size=8,
    enable_expert_parallel=True,
    trust_remote_code=True,
    block_size=32,
    max_model_len=2048,
    max_num_seqs=4,
    max_num_batched_tokens=1024,
    enable_prefix_caching=False,
    async_scheduling=False,
    additional_config={"neuron_config": {"quantization": "bf16"}},
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
for output in llm.generate(["The color of the sky is"], sampling_params):
    print(output.outputs[0].text)
```

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1` in the environment. See the top-level `README.md` and
`AGENTS.md` for the full list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn3                   | Not tested                      | Not tested       |
| Trn2                   | Working (TP 8)                  | Not tested       |

The code in this folder has run on 0.29.0.1.2.0 only.

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

`test/integration/test_model.py` builds a vLLM engine with `LLM(...)` at TP 8 /
EP 8 and validates its logits. It needs 8 Neuron cores reachable from the test
process, which one `trn2.48xlarge` provides. The test registers the
architecture in its own process and fails early unless
`VLLM_WORKER_MULTIPROC_METHOD` is unset or `fork`:

```bash
pytest contrib/models/bailing_moe/test/integration/test_model.py \
    -v -s --timeout=10800
```

`test/integration/test_full_model_prefill.py` is the other full-model test: it
runs a real 20-layer prefill of both logit prompts on 8 ranks and asserts that
every stage is finite. It drives the model directly, not through vLLM, so on a
device it runs across 2 nodes like the module tests below.

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

On a device, the module tests and `test_full_model_prefill.py` place the 8
ranks on 2 `trn2.48xlarge` nodes with 4 ranks on each node
(`DEVICE_NNODES = 2` in `test/module/utils.py`). Each node needs its node rank,
the rendezvous address and the Neuron runtime communication id in the
environment; the module docstring of `test/module/utils.py` names the
variables. The tests name an inter-node launcher
(`test/vllm_neuron/functional/collectives/launch_inter_node_a2av.py`) that is
not part of this repository, so you need your own way to start one
pytest process per node with those variables.

See the [module test guidelines](../../../docs/design/accuracy/module_test_guidelines.md)
for the three-way accuracy pattern these tests follow.

## Validation report

### vLLM Neuron 0.29.0.1.2.0 (2026-11-18)

#### Environment

| Image | Instance | TP | vLLM | vllm-neuron | torch | transformers | neuronx-cc |
|---|---|---|---|---|---|---|---|
| `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` | 1 × `trn2.48xlarge`, LNC 2 | 8 (EP 8) | 0.29.0 | 0.29.0.1.2.0 | 2.13.0 | 5.17.0 | 2.28.7631.0+8f7b0cf5 |

#### Results

| Test | Configuration | Result |
|---|---|---|
| CPU unit tests, `test/unit/` | `VLLM_NEURON_CPU_MODE=1`, `neuron_native`, 8 ranks on the host | 7 passed, 0 failed |
| `test_model_accuracy[1-256]` | TP 8 / EP 8, `neuron_native`, BF16, batch 1, 256-token context, 2 prompts × 16 greedy tokens, on-device greedy sampling | PASSED on aggregate σ-ratio 0.8723 ≤ 1.0 and per-prompt `pp_max_linf_1.5x` / `pp_max_l2_1.5x` 2/2; static per-prompt bounds 0/2 (below). Rerun 2026-11-20 with the per-prompt assertions: PASSED (258.83 s), same values |
| `vllm serve` smoke | The Usage command, installed with `pip install -e contrib/models/bailing_moe` | Passed: 3/3 prompts, 32 completion tokens each; a batched request of all 3 at `max_tokens` 16 returned a prefix of each |

Not run on 0.29: `test/module/` and `test/integration/test_full_model_prefill.py`, CPU or device (not in the one-node 0.29 run; on a device they need 2 nodes).

#### Logit validation

| Prompt | Overall Status | Max K5 (0.011) | Max K50 (0.02) | Max K1000 (0.03) | Max All (0.05) | Argmax divergence (1 ULP) |
|---|---|---|---|---|---|---|
| 1 of 2 | FAILED | 0.0697 (> 0.011) at token 0 | 0.0811 (> 0.02) at token 0 | 0.0896 (> 0.03) at token 0 | 0.1102 (> 0.05) at token 0 | 0.2500 (> 1 ULP) at token 13 |
| 2 of 2 | FAILED | 0.0187 (> 0.011) at token 4 | 0.0580 (> 0.02) at token 4 | 0.0580 (> 0.03) at token 4 | 0.0580 (> 0.05) at token 4 | 0.0000 |

Neuron vs CPU BF16 reference; token = generated token. Prompt 1 over K5, K50, K1000 and All at generated token 0, argmax divergence at generated token 13; prompt 2 over K5, K50, K1000 and All at generated token 4.
Test passed on aggregate σ-ratio 0.8723 ≤ 1.0 (`multi_prompt_logit_validation`) and on the asserted per-prompt checks `pp_max_linf_1.5x` and `pp_max_l2_1.5x` 2/2; `agg_bc` FAIL.

Error against the FP32 reference, per prompt (max over the 16 generated tokens):

| Prompt | CPU BF16 L-inf | Neuron L-inf | Neuron / CPU BF16 | CPU BF16 L2 | Neuron L2 | Neuron / CPU BF16 |
|---|---|---|---|---|---|---|
| 1 of 2 | 0.1116 | 0.0331 | 0.30x | 0.0521 | 0.0392 | 0.75x |
| 2 of 2 | 0.0455 | 0.0520 | 1.14x | 0.0234 | 0.0295 | 1.26x |

Prompt 1, generated token 13 (the argmax divergence), logits of the two candidate tokens:

| Logits | `","` | `" divided"` | Argmax |
|---|---|---|---|
| FP32 reference | 21.4900 | 21.2679 | `","` |
| CPU BF16 reference | 21.500 | 21.625 | `" divided"` |
| Neuron | 21.250 | 21.000 | `","` |

BF16 ULP at 21.5: 0.125. Neuron and FP32 agree; the divergence is against the CPU BF16 reference.

#### Serve smoke

`/v1/completions`, `temperature` 0, `max_tokens` 32; first 60 characters of each output as a Python string literal.

| Prompt | Output |
|---|---|
| `The capital of France is` | `' _.\nA.Paris\nB.London\nC.Berlin\nD.Madrid\nA\nThe problem is a si'` |
| `1, 2, 3, 4, 5,` | `' 6, 7, 8, 9, 10, 11, 12, 13, 14,'` (the whole output) |
| `def fibonacci(n):` | `'\n    """\n    Calculate the nth Fibonacci number using dynami'` |

#### Changes for 0.29

- `src/compat.py` (new): vendors `hoist_rope_parameters`, `build_per_head_block_table` and `scatter_new_kv`, absent from 0.29.0.1.2.0; `src/config.py` and `src/model.py` import them from it.
- `src/factory.py`: vLLM model interface for the front end (keyword-only `vllm_config` and `prefix`, `forward` with explicit `input_ids` and `positions`, `embed_input_ids` and `compute_logits` stubs that raise `NotImplementedError`, not called on the tested paths).
- `pyproject.toml`, `src/plugin.py` (new): install `src/` as the `bailing_moe_neuron` package with a `vllm.general_plugins` entry point that registers `BailingMoeV2ForCausalLM`.
- `test/conftest.py`: session-scoped autouse fixture that shuts down the plugin's `MPExecutor` worker pool (`vllm_neuron.utils.executor._worker_pool`) at session end, so pytest exits after its summary.

### Model configuration

| Item | Value |
|---|---|
| Architecture class | `BailingMoeV2ForCausalLM` (the `architectures` field of the checkpoint) |
| Implementation | BF16; the factory rejects any other dtype and any `quantization_config` |
| Parallelism | TP 8 with EP 8 on one `trn2.48xlarge` on 0.29.0.1.2.0 |
| KV block size | 32 (`--block-size 32`); the component tests allocate 64 blocks, so 2048 tokens of KV |
| Context length | 256 tokens for the integration test and the full-model prefill; `--max-model-len 2048` with `--max-num-batched-tokens 1024` for `vllm serve` |
| Registration | `pip install -e contrib/models/bailing_moe` (a `vllm.general_plugins` entry point); not in `vllm_neuron/model/registry.py` |
| Model-specific serving args | `--enable-expert-parallel`, `--block-size 32`, `--trust-remote-code`, `--no-enable-prefix-caching`, `--no-async-scheduling`, and `neuron_config.quantization = "bf16"` |

### Serving validation

Our required grid is 1,024 in / 512 out and 8,192 in / 1,024 out, offline and
online. We have no evidence for any cell:

| Workload | Offline | Online |
|---|---|---|
| 1,024 in / 512 out | No evidence | No evidence |
| 8,192 in / 1,024 out | No evidence | No evidence |

The 0.29.0.1.2.0 `vllm serve` smoke above is a functional check with prompts
of 5 to 14 tokens and 32 output tokens, not a cell of this grid. The port
ships a serving test for the first row (4 concurrent requests of 1,024 input
and 512 output tokens, online, over 2 nodes), but no recorded run names serving
for this model, and that test is not in this folder (see Exceptions). The
8,192 / 1,024 row never ran for this model at all.

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

- No dataset accuracy run (no GSM8K or similar) appears in the evidence for
  this model.
- Neither serving row of the required grid has evidence, offline or online.
- The offline `LLM(...)` example in Usage was not run as written. The
  integration test builds its engine the same way at TP 8 / EP 8, with a
  256-token context and its own registration.
- Only TP 8 with EP 8 was run, on one node on 0.29.0.1.2.0. Other TP degrees
  and the TP (non-EP) path for the routed experts are untested.
- No evidence for Trn3, or for this folder's code on any version other than
  vLLM Neuron 0.29.0.1.2.0.
- Nothing recorded covers quantization, the FP8 KV-cache path that exists in
  the code, non-greedy on-device sampling, prefix caching, chunked prefill,
  LoRA or speculative decoding.

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
  "fixture not found".
- `test_cpu_loading.py` asserted that `vllm_neuron.model.registry.get_models()`
  maps the architecture to the factory. A contrib model is not in that
  registry, so the test now registers the architecture with vLLM's
  `ModelRegistry` and asserts that vLLM holds it. The 0.29.0.1.2.0 run asserted
  the new form.
- `test/integration/test_model.py` registered the factory as the string
  `"contrib.models.bailing_moe.src.factory:BailingMoeV2ForCausalLM"` and put
  the repository root on `PYTHONPATH` for the engine's children. It now
  registers the class object that the loader returns, and fails early with an
  explanation if `VLLM_WORKER_MULTIPROC_METHOD` is not `fork`. The 0.29.0.1.2.0
  run ran it in this form.
