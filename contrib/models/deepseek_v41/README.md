# Contrib Model: DeepSeek-V4.1-Flash (text)

Serving [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
with vLLM on AWS Trainium through the `vllm-neuron` plugin. The architecture
string is `DeepseekV41ForCausalLM`, taken from the checkpoint's own
`config.json`.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

This is a large MoE reasoning model with several pieces that have no analogue in
the dense reference ports: multi-head latent attention with a sparse indexer and
compressed KV, hyper-connections, an engram memory, and the DSpark MTP stages
that draft speculative tokens *inside* the target model. See
[`src/README.md`](src/README.md) for the architecture and the feature-status
card.

## Requirements

| | |
| --- | --- |
| Instance | `trn2.48xlarge` — 16 devices, 64 logical NeuronCores at LNC=2 |
| Parallelism | `tensor_parallel_size = 64`, set by `src/config.py` |
| Checkpoint | 475.3 GiB over 48 shards, on **local NVMe** |
| Host memory | ~180 GiB resident at 64 ranks |

Two of these are hard rather than advisory:

- **The whole instance.** The port serves at TP=64, which is every logical core
  on the box. It cannot share the instance with another Neuron job.
- **The checkpoint on instance store, not EBS.** 64 ranks read 475 GiB between
  them at startup. From a gp3 root volume (measured 33–58 MB/s here) that is
  hours; from the instance store it is about a minute.

`vm.overcommit_memory` must be left at the default `0`. Strict overcommit
(`2`) fails this deployment while ~1.9 TB of real memory is still free:
compiling forks `neuronx-cc` out of each of 64 workers, and a fork under strict
mode re-commits the parent's address space.

### Speculative decoding needs one plugin change

The DSpark MTP stages draft **inside** the target model, and the runner's
speculative path is wired to proposers that ship with the plugin — so the model
cannot register its draft proposer from inside its own folder. That change is a
separate pull request, and `plugin-core/` in this folder carries a copy as a
fallback:

```bash
contrib/models/deepseek_v41/plugin-core/apply_plugin_core.sh --check
```

It reports "already in this tree" and does nothing if the upstream change has
merged. **Without it the model still serves correctly** — both imports are
guarded — but it logs that speculation is unavailable and decodes one token per
step, losing the ~49% draft acceptance the benchmark numbers below were measured
with. See [`plugin-core/README.md`](plugin-core/README.md).

## Registering the model

Contrib models are deliberately not in `get_models()`, so you register this one
yourself — see
[contrib/README.md](../../README.md#how-do-i-register-a-custom-model-with-vllm).
**Importing the package is the whole registration step.** `src/__init__.py`
calls, at import time:

- `register_hf_config()` — teaches `transformers.AutoConfig` the `deepseek_v41`
  model type. Without it `ModelConfig` rejects the checkpoint before any worker
  starts, because upstream `transformers` has never seen this architecture.
- `register_architecture()` — `ModelRegistry.register_model(...)`.
- the port's config hooks, which pin segmented prefill, the KV-cache ceiling and
  the DSpark speculative config.

So no edit to `vllm_neuron/model/registry.py` is needed, and none is included in
this contribution.

### Import order matters

Because registration happens on import, **anything that resolves the checkpoint
before the package is imported will fail.** That includes bare `vllm serve`,
which builds `ModelConfig` before it loads any plugin code:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
  Value error, The checkpoint you are trying to load has model type
  `deepseek_v41` but Transformers does not recognize this architecture.
```

There are two ways round it, and the choice decides where `src/` goes.

#### Offline inference — `LLM(...)`

Import the package first, then build the engine. This is contrib
[Option 1](../../README.md#option-1--offline-the-llm-api) and needs no change to
the plugin.

```python
import sys
sys.path.insert(0, "contrib/models/deepseek_v41/test")  # holds the `deepseek_v41` alias
import deepseek_v41  # noqa: F401 - the import *is* the registration

from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/DeepSeek-V4.1-Flash",
    tensor_parallel_size=64,
    max_model_len=2104,
    max_num_seqs=1,
    quantization="fp8",
)
out = llm.generate(
    ["What is the capital of France?"],
    SamplingParams(temperature=0.0, max_tokens=32),
)
print(out[0].outputs[0].text)
```

In your own application, copy `src/` in as a package named `deepseek_v41` and
`import deepseek_v41` normally. The name is not cosmetic: `register_architecture`
registers the factory as `f"{__name__}.factory:DeepseekV41ForCausalLM"`, and vLLM
re-imports that string in the architecture-inspection subprocess and in all 64
spawned workers, so the package has to be importable *by that name* in a fresh
interpreter.

#### Online serving — `vllm serve`

`vllm serve` starts the engine directly, so there is no script of yours to run
the import. Use a thin entry point that imports the package and then hands over
to vLLM's own `api_server` — `test/performance/_serve_main.py` in this folder is
exactly that, and the benchmark uses it:

```bash
PYTHONPATH=contrib/models/deepseek_v41/test \
python -m performance._serve_main \
    --model /path/to/DeepSeek-V4.1-Flash \
    --tensor-parallel-size 64 \
    --max-model-len 2104 \
    --max-num-seqs 1
```

Alternatively, promote the model with contrib
[Option 2](../../README.md#option-2--online-the-vllm-serve-cli) — move `src/` to
`vllm_neuron/model/deepseek_v41/` and add the architecture to `get_models()` —
after which plain `vllm serve` works, since the plugin imports the package at
worker startup.

## Compatibility Matrix

Neuron SDK versions and instance types this model was tested with.

| Instance / SDK version | 2.33       | 2.32       | 2.31 and earlier |
| ---------------------- | ---------- | ---------- | ---------------- |
| Trn3                   | Working    | Not tested | Not tested       |
| Trn2                   | Working    | Working    | Not tested       |

Trn1 is not applicable: the checkpoint does not fit, and the port's kernels
target Trainium2.

The exact stack behind the "Working" cells:

| | Validated | PR base |
| --- | --- | --- |
| `vllm-neuron` | **0.24.0.1.1.0** | main (0.29.0.1.2.0) |
| vLLM | 0.24.0 | 0.29.x |
| Python | 3.12.3 | — |
| `torch` / `torch-xla` | 2.11.0 | — |
| `neuronx-cc` | 2.27.5334.0 | — |
| `libtorch-neuronx-lite` | 2.11.0.1.0.2651 | — |
| `nki` | 0.6.0 | — |
| Driver (`aws-neuronx-dkms`) | 2.31.7.0 | — |
| Runtime (`aws-neuronx-runtime-lib`) | 2.35.30.0 | — |
| Collectives (`aws-neuronx-collectives`) | 2.35.43.0 | — |
| Tools (`aws-neuronx-tools`) | 2.33.38.0 | — |

### Trn3 on SDK 2.33 (vllm-neuron 0.29.0.1.2.0)

Validated 2026-10-22 on a `trn3.48xlarge` (16 devices, 64 logical cores at
LNC=2, 144 GB each) with the integration test below: both expected substrings
matched, the needle included. The stack was the stock DLAMI venv
`aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0`: vLLM 0.29.0, torch and
torch-xla 2.13.0, `libtorch-neuronx-lite` 2.13.0.1.0.3112, `neuronx-cc`
2.28.7631.0, `nki` 0.7.0, runtime 2.35.30.0, driver 2.31.7.0.

Bringing the port up there needed changes in both the plugin and the model;
each is commented at its site with the measurement that motivated it:

- **Plugin** (`vllm_neuron/`): the 0.29 runner's `redecorate_compiled_model`
  skips drafters without `compile_kwargs` (an in-model proposer has none); the
  Trn2 8x8 expert mesh in `neuron_parallel_state._build_2d_mesh` applies to the
  `trn2` family only, since its device pairs are not linked on Trn3's switch
  topology and the runtime refuses the group; a new
  `patches/dynamo_recompile_reason_patch.py` makes torch 2.13's
  recompilation-reason report best-effort.
- **Device numerics** (`src/model.py`): int64 multiply, xor and modulo in
  compiled graphs are evaluated at 32-bit width on this host, so the engram
  n-gram hash runs on the host (`NgramHashState`); compiled fp32 division is
  one ulp off, which flips e2m1 ties, so the e4m3-scaled fp4 path refines its
  quotient (`fp4_quant_dequant`); the compile backend name is resolved outside
  any trace; the logits gather passes a non-negative dim.
- **Reference fidelity** (`src/model.py`, found by per-layer comparison against
  the checkpoint's own `inference/` code on CPU): the attention output is
  rounded through fp8 before the grouped `wo_a` as the reference does; the
  compressor projects and pools in fp32; `hc_mixes`' projection, the router
  scores and the sharded `wo_b` partials stay fp32 (`fp8_linear_f32`).
- **Sampling**: the plugin's distributed top-k sampler returned rank 0's local
  argmax instead of the global one here, so greedy rows take the argmax of the
  gathered logits inside the sampler's graph (`_sampler_entry`). Non-greedy
  rows still go through the plugin sampler.

`DEEPSEEK_V41_TRACE_INPUTS=1`, `DEEPSEEK_V41_DUMP_DIR=<dir>` and
`DEEPSEEK_V41_CHECK_COLLECTIVES=1` are the diagnostics that found these; all
are off by default. The n-gram hash costs one small device-to-host copy of the
step's token ids.

**Megakernels (2026-10-23).** On Trn3 the port now runs every block as one NKI
kernel and the DSpark drafter as another (`src/kernels/layer_kernel.py`,
`src/kernels/draft_kernel.py`, bridged by `src/mk_model.py`; see
`src/README.md` "Megakernels"). The quantized projections run on the tensor
engine's native block-scaled fp8 / fp4 matmul (`nc_matmul_mx`, the weights
re-laid once into its packed layout, activations quantized on chip), the sparse
attention selects each token's keys on chip through tensor indirection into a
resident transposed slot table, and the routed experts run token-grouped: each
expert only over the rows that picked it, in a runtime-bounded loop, so an
expert nobody picked costs nothing. The integration test passes on that path
(needle recalled, trivia answered with the per-op path's exact completion) and
the matrix below was re-measured with it: TTFT 560 / 1064 / 1954 / 8545 ms and
TPOT 31.6 / 31.1 / 29.7 / 33.8 ms at 512 / 1k / 2k / 8k input, batch 1
(the dense-expert megakernels of the morning: 625 / 1157 / 2134 / 9261 ms and
73 / 66 / 69 / 70 ms; the per-op path: 1060 / 1701 / 3037 / 13419 ms and
~100 ms), 28 decode tok/s at short contexts and 10 at 8k (13.3 and 7.0 before),
draft acceptance 42-49% against 45-50% before and the per-op path's 46-52%.
`test/unit/EQUIVALENCE_REPORT.md` has the component-level and per-layer
agreement of the paths; `DEEPSEEK_V41_MEGAKERNEL=0` restores the per-op path.

> **Read this before you deploy.** Every number in this contribution was
> measured on `vllm-neuron` **0.24.0.1.1.0**, which is the branch the model was
> developed against. The code is contributed on top of `main`, and the port's
> imports against `main` have been checked symbol by symbol, but **the model has
> not been executed on `main`** — doing so needs vLLM 0.29, and the validated
> environment is on 0.24. If you are on `main`, treat this as untested and run
> the integration test below before trusting it. If you want the validated
> configuration exactly, check out `release-0.24.0.1.1.0`, where the
> accompanying `spec_decode` change also applies unmodified.

## Example Checkpoints

- <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash> — revision
  `314113f8a7dadb388215396ba5fbee2c610a04d6`, the revision everything here was
  tested against. 475.3 GiB, 48 shards.

The checkpoint ships attention weights in fp8 (`ue8m0` scales, 32x32 blocks) and
the expert weights in fp4, as its `quantization_config` declares. Pass
`quantization="fp8"` or leave it unset; any other value is rejected with a
reason by `src/factory.py`.

Download straight to the fast volume — a 475 GiB fetch into a slow one stalls
behind writeback:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash \
    --revision 314113f8a7dadb388215396ba5fbee2c610a04d6 \
    --local-dir /mnt/nvme/hf
```

## Testing

### Accuracy (integration)

Serves the model through vLLM on all 64 cores and checks the generated text.
Needs the devices to itself.

```bash
DEEPSEEK_V41_CHECKPOINT=/mnt/nvme/hf \
pytest contrib/models/deepseek_v41/test/integration/test_model.py \
    -v -s --timeout=14400
```

`-s` is not optional in practice: 64 workers log progress, and a cold compile
takes tens of minutes during which captured output makes a live run look hung.

What it asserts, all deterministic:

1. every phase built an engine, loaded weights, compiled and ran — a phase that
   raised is a failure;
2. every selected case produced tokens, so a case dropped for want of context
   fails rather than passing silently;
3. every case that declares an expected substring contains it. This is the real
   accuracy signal: `long-needle-2048-0` plants an access code 2008 tokens deep
   and the completion has to recall it, which a model with broken long-context
   attention, a mis-sharded weight or a wrong RoPE cannot do by luck.

Every completion is printed before the checks run, so a failing run shows what
the model actually produced rather than only that it was wrong. `short-prose`
has no single right answer, so nothing is asserted about it — it is printed for
a reader.

The test does **not** use the template's logit-validation helpers. A HuggingFace
reference for this checkpoint would need the fp8/fp4 weights dequantised to an
fp32 baseline — roughly 1.9 TB of parameters — and upstream `transformers` has
no implementation of this architecture to compare against. The test file's
docstring explains the substitution in full.

### Speed (performance)

The port's benchmark harness. It serves the model and drives it with
`vllm bench serve --dataset-name random` over a matrix of input lengths and
batch sizes, reporting TTFT, TPOT and aggregate decode throughput per cell.

```bash
python contrib/models/deepseek_v41/test/performance/run_bench.py \
    --model /mnt/nvme/hf
```

Random input is deliberate: it fixes the token count exactly, so a cell measures
the shape it names, and no two requests share a prefix the prefix cache could
serve instead of computing. See
[`test/performance/README.md`](test/performance/README.md) for the matrix, the
flags and how to read the output. A full eight-cell run takes hours;
`--input-lens` and `--batch-sizes` narrow it to the cells you care about.

### Optional: unit and module tests

None are contributed yet — `test/unit/` and `test/module/` are placeholders. The
port's component-level checking was done against a private reference
implementation that cannot be redistributed, so re-deriving it against a public
reference is left as follow-up work. The accuracy test above is the gate that
runs today.
