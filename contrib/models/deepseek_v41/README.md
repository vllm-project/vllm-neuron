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
| Trn3                   | Not tested | Not tested | Not tested       |
| Trn2                   | Working    | Working    | Not tested       |

Trn1 is not applicable: the checkpoint does not fit, and the port's kernels
target Trainium2.

The exact stacks behind the "Working" cells. Both pass the integration test
below on a `trn2.48xlarge`, with the same system packages:

| | Neuron SDK 2.32 | Neuron SDK 2.33 |
| --- | --- | --- |
| `vllm-neuron` | 0.24.0.1.1.0 | 0.29.0.1.2.0 (`main`) + the plugin-core change |
| vLLM | 0.24.0 | 0.29.0 |
| Python | 3.12.3 | 3.12.3 |
| `torch` | 2.11.0 | 2.13.0 |
| `libtorch-neuronx-lite` | 2.11.0.1.0.2651 | 2.13.0.1.0.3112 |
| `neuronx-cc` | 2.27.5334.0 | 2.28.7631.0 |
| `nki` | 0.6.0 | 0.7.0 |
| Driver (`aws-neuronx-dkms`) | 2.31.7.0 | 2.31.7.0 |
| Runtime (`aws-neuronx-runtime-lib`) | 2.35.30.0 | 2.35.30.0 |
| Collectives (`aws-neuronx-collectives`) | 2.35.43.0 | 2.35.43.0 |
| Tools (`aws-neuronx-tools`) | 2.33.38.0 | 2.33.38.0 |

The SDK 2.33 column is the DLAMI's `aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0`
environment with `pytest-timeout` added. It compiles at vLLM's default `-O2`;
SDK 2.32 compiles at `-O1`, the level its platform defaults to.

> **Read this before you deploy.**
>
> * **Speculative decoding needs the plugin-core change on both stacks**; see
>   [Speculative decoding needs one plugin change](#speculative-decoding-needs-one-plugin-change).
> * **The model runs as NKI megakernels by default** -- one kernel a block a
>   step, one per layer archetype and phase, and one for the DSpark drafter
>   (see [`src/README.md`](src/README.md#megakernels)). Measured on SDK 2.33,
>   that path decodes 1.4-2.2x faster than the per-op path on SDK 2.32 and
>   2.1-3.8x faster than the per-op path on SDK 2.33 ([Speed](#speed-performance)).
>   `DEEPSEEK_V41_MEGAKERNEL=0`, set before the engine starts, restores the
>   per-op path.
> * **On the per-op path, SDK 2.33 is slower than SDK 2.32.** Device traces of
>   one decode step put that gap inside the SDK's execution path, not this
>   model's code: the per-layer graph runs ~25% longer on the device, and the
>   device idles longer between graph submissions. The compiler version, `nki`
>   version, optimization level (`-O1`/`-O2`), backend route (native/XLA), IO
>   ring cache size and lite's index bound checks were each ruled out by
>   measurement.
> * **Segmented prefill at an untested shape fails on both stacks.** A
>   1215-token prompt served with `max_model_len=1599` and
>   `max_num_batched_tokens=1024` (two prefill segments) hit a `neuronx-cc`
>   internal error (`NCC_ILSA902`) on SDK 2.32 and an out-of-bounds indirect
>   DMA on SDK 2.33. The configurations the integration test and benchmark use
>   — one prefill segment per prompt — are unaffected. Prefer a
>   `max_num_batched_tokens` at least as long as your prompts until this is
>   understood.

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
DEEPSEEK_V41_MAX_NUM_SEQS=10 \
python contrib/models/deepseek_v41/test/performance/run_bench.py \
    --model /mnt/nvme/hf
```

`DEEPSEEK_V41_MAX_NUM_SEQS` lifts the model's default one-sequence clamp. Without
it every batch cell is served one sequence at a time and measures queueing.

Random input is deliberate: it fixes the token count exactly, so a cell measures
the shape it names, and no two requests share a prefix the prefix cache could
serve instead of computing. See
[`test/performance/README.md`](test/performance/README.md) for the matrix, the
flags and how to read the output. A full six-cell run takes hours;
`--input-lens` and `--batch-sizes` narrow it to the cells you care about.

Measured on a `trn2.48xlarge` with the plugin-core change,
`DEEPSEEK_V41_MAX_NUM_SEQS=10`, 128 output tokens, warm process. Acceptance is
DSpark's, from vLLM's own counters over the cell.

**Neuron SDK 2.33, megakernel path, every block on its tuned kernel** (the default;
`kernels/tuned/`, `DEEPSEEK_V41_MK_TUNED=0` restores the 2026-10-25 table below), 2026-10-29:

| cell | TTFT p50 (ms) | TPOT p50 (ms) | decode (tok/s, aggregate) | acceptance |
| --- | ---: | ---: | ---: | ---: |
| in512 bs1 | 259.0 | 27.3 | 34.55 | 40% |
| in512 bs10 | 1433.8 | 70.7 | 112.91 | 43% |
| in1024 bs1 | 441.3 | 25.8 | 33.68 | 44% |
| in1024 bs10 | 2418.8 | 70.2 | 104.46 | 47% |
| in8192 bs1 | 4462.4 | 25.8 | 16.48 | 44% |
| in8192 bs4 | 6724.6 | 98.9 | 21.85 | 42% |

Every one of the 18 layer kernels (9 archetypes x prefill / decode) runs from a package
tuned against inputs recorded from this model, chosen per archetype as the fastest that
is bit-exact on that archetype's own recorded cases (88 cases, four configurations each).
Against 2026-10-25: TTFT 1.98-2.34x, decode 1.41-1.86x. The step moves by what the
[profile](test/performance/PROFILE_REPORT.md)'s breakdown predicts: a 1k prefill
1024 -> 436 ms, a 1k bs10 decode step 239 -> 187 ms. At batch 1 the decode step (~84 ms)
is now bound by the host's kernel submission, not the device.

**Neuron SDK 2.33, megakernel path** (`vllm-neuron` 0.29.0.1.2.0, `-O2`),
2026-10-25:

| cell | TTFT p50 (ms) | TPOT p50 (ms) | decode (tok/s, aggregate) | acceptance |
| --- | ---: | ---: | ---: | ---: |
| in512 bs1 | 548.2 | 37.9 | 24.48 | 43% |
| in512 bs10 | 3030.0 | 99.5 | 77.12 | 44% |
| in1024 bs1 | 1016.7 | 36.6 | 22.39 | 44% |
| in1024 bs10 | 5648.5 | 120.9 | 46.58 | 42% |
| in8192 bs1 | 8849.2 | 39.4 | 9.27 | 41% |
| in8192 bs4 | 13373.9 | 161.8 | 11.89 | 41% |

A decode-step layer kernel runs in ~4 ms and a 512-row prefill one in ~14 ms
(`DEEPSEEK_V41_MK_SYNC=1` times each kernel). The `in1024 bs10` window carries
dynamo retraces (192) of the forward's glue as the live sequence count steps
down through new values; no kernel recompiles (`neff_misses=0`).

The per-op path (`DEEPSEEK_V41_MEGAKERNEL=0`), for comparison:

**Neuron SDK 2.32, per-op** (`vllm-neuron` 0.24.0.1.1.0, `-O1`), 2026-10-23:

| cell | TTFT p50 (ms) | TPOT p50 (ms) | decode (tok/s, aggregate) | acceptance |
| --- | ---: | ---: | ---: | ---: |
| in512 bs1 | 596.0 | 85.5 | 11.02 | 44% |
| in512 bs10 | 3293.9 | 226.5 | 35.85 | 48% |
| in1024 bs1 | 966.6 | 78.1 | 11.46 | 49% |
| in1024 bs10 | 5336.8 | 237.0 | 33.73 | 46% |
| in8192 bs1 | 11261.9 | 95.4 | 5.50 | 43% |
| in8192 bs4 | 17141.4 | 326.6 | 7.94 | 41% |

**Neuron SDK 2.33, per-op** (`vllm-neuron` 0.29.0.1.2.0, `-O2`), 2026-10-24:

| cell | TTFT p50 (ms) | TPOT p50 (ms) | decode (tok/s, aggregate) | acceptance |
| --- | ---: | ---: | ---: | ---: |
| in512 bs1 | 666.8 | 124.8 | 7.41 | 44% |
| in512 bs10 | 3809.8 | 392.7 | 20.14 | 44% |
| in1024 bs1 | 1040.7 | 128.1 | 7.52 | 44% |
| in1024 bs10 | 5707.4 | 415.6 | 20.60 | 46% |
| in8192 bs1 | 11900.1 | 139.1 | 4.32 | 41% |
| in8192 bs4 | 24203.8 | 505.3 | 5.68 | 40% |

On the per-op path SDK 2.33 costs 5-12% on prefill and 21-44% on decode, worst
at batch 10, where a step runs the most graphs. A rank-0 system trace of one in512 bs1 decode step
on each SDK puts it below this model: per-graph device time is up (the per-layer
graph runs 2.1 → 2.7 ms) and the device idles longer between graph
submissions, while the work submitted is the same and acceptance matches. The
SDK 2.32 table predates the fixes that let this folder run on SDK 2.33; with
them SDK 2.32's integration-test completions are token-identical, and none of
them changes what a step computes.

Random token ids are close to unpredictable for a drafter, so these figures are
a lower bound on real traffic; the prose probe measured 47% (SDK 2.32) and 46%
(SDK 2.33). The two medians at `in8192 bs4` are bimodal across runs, because 8
requests at concurrency 4 put the p50 between two waves; its throughput and
mean TPOT are the stable figures.

### Optional: unit and module tests

`test/unit/` checks the megakernels against a CPU reference on a few cores,
component by component, from random weights in the checkpoint's formats:

```bash
cd contrib/models/deepseek_v41/test/unit
python test_gemm.py                      # the Trn2 GEMMs, both partitionings
python test_moe.py --T 300               # the token-grouped fp4 experts
python kharness_mp.py 8 test_layer.py --arch c2_source --phase prefill --T 1024 --stage 41
python kharness_mp.py 8 test_draft.py --L 6 --start 200 --stage 18
```

`test_layer.py` takes any archetype (`--arch`, see `layer_kernel.ARCH`) and phase;
`--stage` returns early at a sublayer (`layer_kernel`'s docstring lists them), and
stages 41 / 82 compare the attention and the MoE against a reference fed the
kernel's own inputs -- on random weights a full-layer comparison is dominated by
rows whose top-6 experts differ after an fp8 re-quantization flip, so the
components are the exact check. `probe_*.py` are the device probes of the
hardware features the kernels rely on (dynamic loops, `nonzero_with_count`,
`dma_compute`, in-kernel collectives over rank groups, the GpSimd top-k).
`test/module/` is a placeholder; the accuracy test above is the gate.
