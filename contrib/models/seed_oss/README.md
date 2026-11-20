# Seed-OSS (Text)

This folder serves the Seed-OSS text models with vLLM on AWS Trainium through
the `vllm-neuron` plugin. The implementation under `src/` is a dense
Llama-style decoder whose Q, K and V projections carry a bias. See
`src/README.md` for the architecture and the feature-status card.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md` for installation), then install this folder:

```bash
pip install -e contrib/models/seed_oss
```

The install registers the architecture `SeedOssForCausalLM` in every vLLM
process (the API server, the engine core and each tensor-parallel worker)
through a `vllm.general_plugins` entry point (`pyproject.toml`,
`src/plugin.py`), so `vllm serve` and `LLM(...)` need no other setup.
`pip uninstall vllm-neuron-contrib-seed-oss` removes the registration.

The 36B checkpoint is about 72 GB in BF16, so it needs several Neuron cores:
the vLLM Neuron 0.29.0.1.2.0 runs used TP8 (integration test) and TP16
(`vllm serve`) on `trn2.48xlarge`.

### Serve with the OpenAI-compatible server

These are the flags of the tested `vllm serve` run:

```bash
vllm serve ByteDance-Seed/Seed-OSS-36B-Instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --tensor-parallel-size 16 \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching
```

Then send a request:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "ByteDance-Seed/Seed-OSS-36B-Instruct",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

```python
from vllm import LLM, SamplingParams

# `pip install -e contrib/models/seed_oss` has registered SeedOssForCausalLM,
# so no ModelRegistry call is needed here.
llm = LLM(
    model="ByteDance-Seed/Seed-OSS-36B-Instruct",
    dtype="bfloat16",
    tensor_parallel_size=8,
    max_model_len=2048,
    max_num_seqs=1,
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The color of the sky is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

Prefill runs sequence-parallel, so every prefill token bucket (by default the
powers of two from 128, then `--max-num-batched-tokens` as the last bucket)
must be larger than the tensor-parallel size and a multiple of it. Prompts are
padded up to a bucket, so the prompt itself can be shorter: the serve smoke
below sent 5- and 14-token prompts at TP16.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1` in the environment. See the top-level `README.md` and
`AGENTS.md` for the full list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn3 | Not tested | Not tested |
| Trn2 | Working (TP 8 and TP 16) | Not tested |

## Example Checkpoints

- <https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct> at revision
  `497f1dca95ebdec98e41d517b9f060ee753c902f`, the revision the 0.29.0.1.2.0
  runs used and whose `config.json` the unit tests embed.

## Testing

Every suite in this folder resolves the local `src/` by file location, through
the small `test/_local_src.py` helper, so the tests run from any working
directory and keep working after the folder is copied out of the repository.

The integration test serves the model through vLLM on Neuron and validates its
per-token logits against a HuggingFace reference:

```bash
pytest contrib/models/seed_oss/test/integration/test_model.py \
    -v -s --timeout=7200
```

Set `SEED_OSS_CHECKPOINT_PATH` to a local checkpoint directory to avoid a Hub
download. The test compiles one graph for a 36B model, so allow a long
timeout. It registers the local `src/` factory itself, so it does not need the
`pip install` above.

### Optional: unit tests

```bash
VLLM_NEURON_CPU_MODE=1 VLLM_NEURON_BACKEND=neuron_native \
    pytest contrib/models/seed_oss/test/unit/ -v
```

Fourteen cases. They need no Neuron device and no 36B weights: registration
and factory resolution, config mapping from the published `config.json`, the
TP8 geometry and checkpoint-key coverage on the meta device, and the fused QKV
bias loader layout.

### Optional: module tests

```bash
VLLM_NEURON_CPU_MODE=1 VLLM_NEURON_BACKEND=neuron_native \
    pytest contrib/models/seed_oss/test/module/ -v
```

Five cases, each validating one component against a HuggingFace reference on a
tiny random Seed-OSS checkpoint that the test itself writes: prefill
attention, decode attention, the MLP in both phases, and then the assembled
model's prefill logits together with the kernel-dispatch log. They are two-way
relative-L2 checks rather than the three-way gate of the
[module test guidelines](../../../docs/design/accuracy/module_test_guidelines.md);
see Exceptions.

## Validation report

### vLLM Neuron 0.29.0.1.2.0 (2026-11-18)

#### Environment

| Image | Instance | TP | vLLM | vllm-neuron | torch | transformers | neuronx-cc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` | `trn2.48xlarge`, LNC 2 | 8 (integration), 16 (serve smoke) | 0.29.0 | 0.29.0.1.2.0 | 2.13.0 | 5.17.0 | 2.28.7631.0+8f7b0cf5 |

#### Results

| Test | Configuration | Result |
| --- | --- | --- |
| CPU unit tests (`test/unit/`) | `VLLM_NEURON_CPU_MODE=1 VLLM_NEURON_BACKEND=neuron_native` | 14 passed |
| `test_model_accuracy[1-2048]` | TP 8, `batch_size=1`, `seq_len=2048` | PASSED (421.45 s) |
| `vllm serve` smoke | TP 16, flags as in Usage, after `pip install -e contrib/models/seed_oss` | passed (3/3 completions non-empty) |

Not run on 0.29: module tests (`test/module/`); the run covered only the unit tests, the integration test and the serve smoke.

#### Logit validation

| Prompt | Overall Status | Max divergence (1 ULP) | Max K5 (0.03) | Max K50 (0.05) | Max K1000 (0.08) | Max All (0.09) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 of 2 | PASSED, all 16 tokens | 0.0000 | 0.0060 | 0.0060 | 0.0060 | 0.0072 |
| 2 of 2 | PASSED, all 16 tokens | 0.0000 | 0.0085 | 0.0118 | 0.0118 | 0.0153 |

No value over its threshold, no argmax divergence; test passed on per-prompt validation, 2/2 prompts (`multi_prompt_logit_validation`).
Informational aggregates in the log: `agg_l2_1.5x` PASS; `agg_bc`, `agg_l2_1.2x`, `agg_linf_1.2x`, `agg_linf_1.5x` FAIL; `agg_sigma_ratio` 1.0007 (> 1.0).

#### Serve smoke

`/v1/completions`, temperature 0; completions JSON-escaped, `...` marks a cut.

| Prompt | Single request, `max_tokens` 32 (first 60 chars) | All three prompts in one request, `max_tokens` 16 (first 60 chars) |
| --- | --- | --- |
| `The capital of France is` | `" Paris.\nThe capital of France is Paris.\nThe capital of Franc"` ... | `" Paris.\nParis is the capital of France.\nWhich of the followi"` ... |
| `1, 2, 3, 4, 5,` | `" 6, 7, 8, 9, 10, 11, 12, 13, 14,"` | `" 6, 7, 8, 9, 10,"` |
| `def fibonacci(n):` | `"\n    \"\"\"\n    Calculate the nth Fibonacci number using recurs"` ... | `"\n    \"\"\"\n    Calculate the nth Fibonacci number using recurs"` ... |

#### Changes for 0.29

- `src/compat.py` (new): `hoist_rope_parameters`, copied unchanged from a later vLLM Neuron release; 0.29.0.1.2.0 has no `vllm_neuron.model.config_utils`.
- `src/config.py`: imports `hoist_rope_parameters` from `src/compat.py` instead of the plugin.
- `pyproject.toml`, `src/plugin.py` (new): install `src/` as package `seed_oss_neuron`; register `SeedOssForCausalLM` via `vllm.general_plugins` entry point `seed_oss_neuron`.
- `src/model.py`, `src/factory.py`, `src/__init__.py`, `test/`: unchanged.

### Exceptions

- **Three of the port's four test suites are dropped.** The disaggregated
  (`test_di_seed_oss.py`), GSM8K (`test_native_comparison_accuracy.py`) and
  benchmark (`test_native_comparison_benchmark.py`) suites are not shipped.
  Each imports shared test infrastructure that lives outside the model folder
  — `test/utils/simple_server`, `test/utils/network`, `test/utils/process`,
  `test/utils/bench`, `test/utils/metric_checks`,
  `test/vllm_neuron/utils/di_utils`, `test/vllm_neuron/utils/logit_test_utils`
  and `test/vllm_neuron/utils/test_prompts` — and the first also needs
  `mpirun` across two nodes with a NIXL side channel, while the other two need
  the `WORKLOAD_OUTPUT_RW` result-publishing contract. Vendoring that
  infrastructure would have meant copying a server launcher, a proxy, an MPI
  harness and a metric-publishing layer into the model folder, so the suites
  were dropped.
- **No logit or serving test in this folder reproduces the disaggregated
  deployment.** The shipped integration test is single-node and offline, and
  the 0.29.0.1.2.0 serve smoke is a single-node server.
- **`PROMPTS_2` is vendored.** The two validation prompts came from
  `test/vllm_neuron/utils/test_prompts.py` and are copied verbatim into the
  integration test, as contrib/README.md sanctions.
- **The registry assertion changed meaning.** The port's CPU test asserted
  that `vllm_neuron.model.registry.get_models()` maps the architecture to the
  factory. A contrib model is deliberately not in `get_models()`, so the
  shipped test registers the local factory with vLLM's `ModelRegistry` and
  asserts that the architecture resolves to it. The case count is unchanged at
  19.
- **The module tests are not the guidelines' three-way pattern.** The
  component cases ship in `test/module/`, where contrib/README.md puts them,
  but they are the port's original two-way checks: a relative-L2 bound of
  2e-2 against an FP32 reference built from the BF16-rounded weights, one
  input per case, on a tiny random checkpoint. The
  [module test guidelines](../../../docs/design/accuracy/module_test_guidelines.md)
  ask instead for `assert_close_three_way` (Bhattacharyya coefficient and
  sigma-ratio) aggregated over about ten inputs on a real checkpoint. The same
  file also carries the full-model prefill-logits case, which is a full-model
  stage rather than a single component, because it shares the tiny-checkpoint
  fixtures with the component cases.
- **Untested configurations.** Only TP8 and TP16 have results (0.29.0.1.2.0:
  the TP8 integration test and the TP16 serve smoke); no other tensor-parallel
  degree, no Trn3, no quantized weights, no speculative decoding (the code has
  a rejection-sampling branch with no evidence behind it), no context length
  beyond 2,048 tokens even though the checkpoint allows 524,288, and no prefix
  caching (every run disabled it). Batches of more than one sequence appear
  only in the 0.29.0.1.2.0 serve smoke (`--max-num-seqs 4`, one three-prompt
  request), never in a logit run, so the shipped integration test keeps a
  single sequence — each `(batch_size, seq_len)` pair also costs a separate
  36B graph compile.
