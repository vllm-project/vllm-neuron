# Seed-OSS (Text)

This folder serves the Seed-OSS text models with vLLM on AWS Trainium through
the `vllm-neuron` plugin. The implementation under `src/` is a dense
Llama-style decoder whose Q, K and V projections carry a bias. See
`src/README.md` for the architecture and the feature-status card.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix and the
> validation report below record exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md` for installation). A contrib model is not in the plugin's model
registry, so register the architecture `SeedOssForCausalLM` yourself, as
described in [contrib/README.md](../../README.md) ("How do I register a custom
model with vLLM?").

The 36B checkpoint is about 72 GB in BF16, so it needs several Neuron cores:
the recorded runs used TP8 (per engine) and TP16 on `trn2.48xlarge`.

### Serve with the OpenAI-compatible server

`vllm serve` starts the engine directly, so there is no script of yours in
which to call `ModelRegistry.register_model`; the plugin's own code also notes
that the CLI starts its engine core and workers in fresh interpreters, where a
class registered in another process is not present (see Exceptions). Use
option 2 of
[contrib/README.md](../../README.md): move `src/` to
`vllm_neuron/model/seed_oss/` and add
`("SeedOssForCausalLM", "vllm_neuron.model.seed_oss:SeedOssForCausalLM")` to
`_MODELS` in `vllm_neuron/model/registry.py`. Then:

```bash
vllm serve ByteDance-Seed/Seed-OSS-36B-Instruct \
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
from vllm import LLM, ModelRegistry, SamplingParams

# Copy contrib/models/seed_oss/src/ into your own application package first,
# then import the factory from there. The code under src/ imports only the
# plugin's shared building blocks, so it needs no other file from this repo.
from your_package.seed_oss import SeedOssForCausalLM

# Register the architecture before you construct the engine.
ModelRegistry.register_model("SeedOssForCausalLM", SeedOssForCausalLM)

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

Option 1 works for this offline path because the plugin's multiprocessing
executor leaves the engine-core and worker processes on the default `fork`
start method,
so they inherit the `ModelRegistry` your script populated, and the model
loader resolves the architecture through vLLM's registry. See Exceptions for
the limits of that.

Prefill runs sequence-parallel, so a prompt length must be greater than the
tensor-parallel size and a multiple of it.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1` in the environment. See the top-level `README.md` and
`AGENTS.md` for the full list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

| Instance | Status | Software |
| --- | --- | --- |
| Trn2 (trn2.48xlarge) | Working — see the validation report | `vllm-neuron` 0.31.0.1.2.0, vLLM 0.31.0, `VLLM_NEURON_BACKEND=neuron_native` |
| Trn3 | Not tested | — |

The evidence that we hold for this model does not record the Neuron SDK
release.

## Example Checkpoints

- <https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct>

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
timeout.

### Unit tests (CPU mode)

```bash
VLLM_NEURON_CPU_MODE=1 VLLM_NEURON_BACKEND=neuron_native \
    pytest contrib/models/seed_oss/test/unit/ -v
```

Fourteen cases. They need no Neuron device and no 36B weights: registration
and factory resolution, config mapping from the published `config.json`, the
TP8 geometry and checkpoint-key coverage on the meta device, and the fused QKV
bias loader layout.

### Module tests (CPU mode)

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

### Source and environment

Four tests passed for this model, across two plugin commits, so no single
commit covers the whole table:

| Test | Verdict | Cases | Nodes | Plugin commit |
| --- | --- | --- | --- | --- |
| `[vLLM Neuron Functional] Seed-OSS-36B BF16 Loading and Component Tests CPU` | PASS | 19 / 19 | 1 | `a9c4da907c3fc287c8a1606f60846082670ae218` |
| `[vLLM Neuron Full Model] Seed-OSS-36B BF16: DI 1P1D Multi-Node TP8 Logits, Serving and Dispatch TRN2` | PASS | 2 / 2 | 2 | `a9c4da907c3fc287c8a1606f60846082670ae218` |
| `[Performance] Seed-OSS-36B-Instruct TP16 Native comparison b1/b4` | PASS | 2 / 2 | 1 | `f9ee9c4727640c752f22e7b6e2960d1a0deca8c0` |
| `[Accuracy] ByteDance-Seed/Seed-OSS-36B-Instruct Native GSM8K b1/b4` | PASS | 2 / 2 | 1 | `f9ee9c4727640c752f22e7b6e2960d1a0deca8c0` |

Every run used `vllm-neuron` 0.31.0.1.2.0 with vLLM 0.31.0 on the native
runtime backend (`neuron_native`), on `trn2.48xlarge`.

The checkpoint is `ByteDance-Seed/Seed-OSS-36B-Instruct`. The loading test
pins its published `config.json` at revision
`497f1dca95ebdec98e41d517b9f060ee753c902f`; the device tests resolve the
checkpoint by path or Hub id and do not pin a revision.

### Model configuration

The architecture class is `SeedOssForCausalLM` (factory in `src/factory.py`,
implementation in `src/model.py`), BF16 weights, no quantization. The model is
dense, so no run names an expert-parallel degree. Two tensor-parallel degrees
have results: TP8 for each of the two engines in the disaggregated run (10 Q
heads and 1 KV head per rank), and TP16 for the single-node performance and
GSM8K runs. The test constants set `--max-model-len 2048` everywhere: the
disaggregated run adds `--max-num-batched-tokens 1024 --max-num-seqs 1`, and
the TP16 runs add `--max-num-batched-tokens 2048` with sequence buckets `[1]`
or `[1, 4]`. Model-specific serving arguments are limited to `--dtype
bfloat16` and disabled prefix caching; the logit run also enables on-device
greedy sampling and raw-logit logprobs. Registration is the user's step; see
Usage above.

### Component validation

The CPU loading-and-component test passed with 19 of 19 cases (commit
`a9c4da90`). It covers factory resolution and config mapping from the
published `config.json`, construction of all 64 layers on the meta device at
TP8 for the first and last rank with parameter-shape checks and full coverage
of the 771 published checkpoint keys, the fused QKV bias loader against the
fused QKV weight loader for every TP8 rank, and five numerics cases on a tiny
random Seed-OSS checkpoint (prefill attention, decode attention, MLP in both
phases, and full-model prefill logits) against an independent HuggingFace
reference, plus the kernel-dispatch log lines. That suite ships here split by
shape: the 14 registration, config, construction and loader cases in
`test/unit/test_seed_oss_loading.py`, and the 5 component and full-model
numerics cases in `test/module/test_seed_oss_components.py` — 19 in total, the
same cases the recorded run ran.

The two-node run also asserted kernel dispatch: each of the 8 ranks on the
prefill node and each of the 8 ranks on the decode node must have selected the
NKI kernel for every main-path operation. It passed as part of that suite's 2
of 2 cases.

### Numerical correctness

The disaggregated run is the logit evidence: a 1P1D deployment on two
`trn2.48xlarge` nodes, one TP8 engine per node, which compares the served
logits of 16 generated tokens per prompt against a HuggingFace reference
(FP32 baseline plus a same-dtype teacher-forced pass) with
`multi_prompt_logit_validation` at a single sequence (`max_num_seqs` is 1 in
that test). It reports both the aggregate and the per-prompt result and
asserts both. The
observed verdict is PASS, 2 of 2 cases (one per node), at commit `a9c4da90`.
We hold that verdict and those counts only — no logit metric was read out.

The shipped `test/integration/test_model.py` uses the same tolerance map as
that run — `(1e-5, 0.03)` at top-5, `(1e-5, 0.05)` at top-50, `(1e-5, 0.08)`
at top-1000 and `(1e-5, 0.09)` over the full vocabulary — and keeps the strict
`1e-5` argmax bound. It has not itself been run in CI: it is a single-node
offline `LLM` run, where the evidence is a two-node disaggregated server.

A GSM8K run also passed (2 of 2 cases, b1 and b4, commit `f9ee9c47`), which
means dataset-level accuracy was exercised end to end. Read that verdict
narrowly: the test's only score gate is a 0.0 threshold on the
flexible-extract exact match, so a PASS says the run completed and produced a
score, not that the score reached any particular level. We never read the
score itself, so this README states none.

### Serving validation

| Workload | Offline | Online |
| --- | --- | --- |
| 1,024 in / 512 out | No evidence | Evidence: one request through the 1P1D proxy, asserted at 1,024 prompt and 512 completion tokens, inside the disaggregated run (2/2 PASS, commit `a9c4da90`) |
| 8,192 in / 1,024 out | No evidence | No evidence |

The online 1,024 / 512 row is the required grid's first row. We have no
offline (`LLM.generate`) measurement at that shape, and no run at
8,192 / 1,024 in either mode. The performance test is a separate, narrower
workload: `vllm bench serve` with random data at 1,024 input and 100 output
tokens, 20 measured requests after 5 warmups, at TP16 and concurrency 1 and 4.
It passed (2 of 2 cases, commit `f9ee9c47`); we hold no latency or throughput
figure from it, so none is stated here.

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
  were dropped. Their results are still the evidence cited above, but a user
  cannot re-run them from this folder.
- **No logit or serving test in this folder reproduces the recorded
  deployment.** The recorded logit and serving evidence is 1P1D across two
  nodes. The shipped integration test is single-node and offline, and it has
  never been run in CI.
- **The GSM8K gate is 0.0.** As described under Numerical correctness, the
  accuracy PASS is not a quality bar. No GSM8K score appears in this README,
  for this reason and because we never read one.
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
  sigma-ratio) aggregated over about ten inputs on a real checkpoint.
  Rewriting the gate would change what the recorded 19-case run validated, so
  the cases ship as they ran. The same file also carries the full-model
  prefill-logits case, which is a full-model stage rather than a single
  component, because it shares the tiny-checkpoint fixtures with the component
  cases.
- **Option 1 works offline only. `vllm serve` needs option 2.** The
  integration test and the offline example register the factory class with
  vLLM's `ModelRegistry` before they build the engine (option 1 of
  contrib/README.md). This registration reaches the engine-core and worker
  processes only because the plugin keeps the default `fork` start method for
  this model. Under `spawn`, a registered class does not reach the new
  processes. The local-src loader that keeps this folder copyable cannot make
  option 1 `spawn`-safe. `vllm serve` starts its engine core and workers in new
  interpreters, so serving needs option 2. This PR does not make the option-2
  edit. Option-1 registration also does not reach the draft-model lookup of
  the untested speculative-decoding branch. We read these limits from the
  plugin's code, not from a run of this folder. No recorded run for this model
  used option 1. A plugin test in CI uses the same pattern on hardware.
- **Untested configurations.** Only TP8 (two-node, one engine per node) and
  TP16 (single node) have results; no other tensor-parallel degree, no Trn3,
  no quantized weights, no speculative decoding (the code has a
  rejection-sampling branch with no evidence behind it), no context length
  beyond 2,048 tokens even though the checkpoint allows 524,288, and no
  prefix caching (every run disabled it). A 4-sequence batch appears only in
  the b4 arms of the performance and GSM8K runs, never in the logit run, so
  the shipped integration test keeps the single sequence of the logit run —
  each `(batch_size, seq_len)` pair also costs a separate 36B graph compile.
- **No performance figures.** A performance test passed, but we hold only its
  verdict and case counts, so this README states no throughput, latency or
  time-to-first-token number.
