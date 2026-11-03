# Contrib Model: DeepSeek-V4.1-Flash (text) on Trainium3

Serving [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
with vLLM on a `trn3.48xlarge` (16 devices, 64 ranks) through the `vllm-neuron` plugin. The
architecture string is `DeepseekV41ForCausalLM`, taken from the checkpoint's own
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

This folder is the **Trainium3** implementation. It grew out of the Trainium2 port
(`contrib/models/deepseek_v41`, one NKI kernel per op on a `trn2.48xlarge`) and keeps its
modules, weight loading and verifier, but runs the model as NKI megakernels written for
Trainium3 (see [What this implementation adds](#what-this-implementation-adds)). The sibling
folder [`deepseek_v41_8d_trn3`](../deepseek_v41_8d_trn3/README.md) serves the same model on
8 of the 16 devices.

## What this implementation adds

Every item is on by default and has its own switch (`src/README.md`, "Megakernels"):

- **One NKI kernel per block** (`src/kernels/tp64/`), for prefill and decode: the fp8 / fp4
  projections on the tensor engine's native block-scaled matmul (`nc_matmul_mx`, the weights
  re-laid once at load time, the activations quantized on chip), the sparse attention's keys
  selected on chip, the routed experts token-grouped (an expert nobody picked costs nothing),
  and the collectives inside the kernels. Each (phase, block archetype) has a kernel optimized
  on the live server's recorded inputs and outputs.
- **A decode step is a handful of launches.** The 40 blocks run as three fused launches
  (blocks 0-7 with the token embedding, 8-23, and 24-39 with the greedy head and the argmax
  over the full vocabulary), plus the DSpark drafter's launch. The launches were optimized as
  a whole: the blocks share their index tables, hand their output over in SBUF and prefetch
  the next block's weights during the all-reduce. A 6-row span's whole block runs on chip
  (`r1_decode/b1.py`): the residual stream stays in SBUF between blocks and the
  hyper-connection maps run on the tensor engine.
- **Decode MoE in a TP8 x EP8 layout** for steps of up to 16 rows: 8 groups of 8 ranks
  split a group's 48 experts by hidden columns, so every rank computes a slice of each picked
  expert and no rank waits for the one with the most picked experts.
- **The host is off the critical path.** One small NKI launch makes a step's positions, cache
  slots, slot tables and n-gram ids on the device. The engine schedules step N+1 while step N
  runs (vLLM's async scheduling). A step is one RPC from the engine to the workers, and the
  fused launches go straight to their NEFF executors with the same IO tensors every step.
- **An optimized DSpark drafter** (`src/kernels/tp64/draft_decode/`): ~2.7x faster than the
  generic drafter kernel on the live server's recorded launches.

A second parallelization scheme, vLLM pipeline parallelism over 8 stages of 8 ranks
(`DEEPSEEK_V41_SCHEME=tp8pp8`), stays available; the optimizations above are for the default
flat scheme (`tp64`).

## Requirements

| | |
| --- | --- |
| Instance | `trn3.48xlarge` — 16 devices, 64 logical NeuronCores at LNC=2 |
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
contrib/models/deepseek_v41_trn3/plugin-core/apply_plugin_core.sh --check
```

It reports "already in this tree" and does nothing if the upstream change has
merged. **Without it the model still serves correctly** — both imports are
guarded — but it logs that speculation is unavailable and decodes one token per
step, losing the ~49% draft acceptance the benchmark numbers below were measured
with. See [`plugin-core/README.md`](plugin-core/README.md).

The numbers below were also measured with the runner and scheduler changes of this branch's
plugin (the one-RPC step, async scheduling with the in-model proposer, the Trn3 bring-up fixes
listed under the compatibility matrix). On a plugin without them the port still serves,
synchronously and with vLLM's three RPCs a step.

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
sys.path.insert(0, "contrib/models/deepseek_v41_trn3/test")  # holds the `deepseek_v41` alias
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
PYTHONPATH=contrib/models/deepseek_v41_trn3/test \
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

| Instance / SDK version | 2.33       | 2.32 and earlier |
| ---------------------- | ---------- | ---------------- |
| Trn3                   | Working    | Not tested       |
| Trn2                   | Not tested | Not tested       |

The megakernels use Trainium3 features (the block-scaled MX matmul, its SBUF size); for
Trainium2 use the Trainium2 port. Trn1 is not applicable: the checkpoint does not fit.

The stack behind the "Working" cell (the DLAMI venv
`aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0`):

| | Validated |
| --- | --- |
| `vllm-neuron` | 0.29.0.1.2.0 (`main` with this branch's plugin changes) |
| vLLM | 0.29.0 |
| Python | 3.12.3 |
| `torch` / `torch-xla` | 2.13.0 |
| `libtorch-neuronx-lite` | 2.13.0.1.0.3112 |
| `neuronx-cc` | 2.28.7631.0 |
| `nki` | 0.7.0 |
| Driver (`aws-neuronx-dkms`) | 2.31.7.0 |
| Runtime (`aws-neuronx-runtime-lib`) | 2.35.30.0 |
| Collectives (`aws-neuronx-collectives`) | 2.35.43.0 |
| Tools (`aws-neuronx-tools`) | 2.33.38.0 |

### Trn3 bring-up notes

Bringing the port up on Trn3 needed changes in both the plugin and the model;
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
  n-gram hash stays out of compiled graphs: on the host (`NgramHashState`), and
  for a decode step in the step-prologue NKI kernel (bit-exact against the host's,
  `test/unit/test_step_kernel.py`); compiled fp32 division is
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
are off by default.

## Performance

Measured 2026-11-02 on a `trn3.48xlarge` (64 ranks) with every optimization at its default
and the stack above, by `test/performance/run_bench.py` (see [Testing](#speed-performance)):
sonnet prompts, 128 output tokens, greedy, one server per input length, warmed before it is
measured. A batch-1 cell sends 8 requests one at a time; a batch-10 cell sends 10 concurrent
requests to an engine built for ten (`DEEPSEEK_V41_MAX_NUM_SEQS=10`), and its output tok/s is
their aggregate. `accept %` is the share of DSpark drafts accepted (5 drafts a step); the median
inter-token latency is one decode step (a streamed chunk per step).

| Cell | TTFT p50 ms | TPOT p50 ms | output tok/s | accept % | median ITL ms |
|------|-------------|-------------|--------------|----------|---------------|
| in512_bs1 | 134 | 3.77 | 205.4 | 49.5 | 12.70 |
| in1024_bs1 | 230 | 3.71 | 181.9 | 51.2 | 12.63 |
| in2048_bs1 | 417 | 3.84 | 142.1 | 49.8 | 12.68 |
| in8192_bs1 | 1662 | 4.00 | 58.9 | 48.0 | 13.22 |
| in512_bs10 | 720 | 15.83 | 442.7 | 48.5 | 38.98 |
| in1024_bs10 | 1250 | 19.50 | 329.4 | 50.0 | 40.08 |
| in2048_bs10 | 2249 | 26.63 | 217.7 | 48.3 | 41.02 |
| in8192_bs10 | 9141 | 71.82 | 68.5 | 44.2 | 44.59 |

Every request of every cell completed with its 128 tokens, and the texts are coherent answers
to the prompts. The integration test passed on the same code (10 min against a warm compile
cache). For scale: the first Trn3 run of this port, one NKI kernel per op, decoded at ~100 ms
TPOT and 1060 ms TTFT at 512 input tokens.

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
Needs the devices to itself. From the repository root, in the DLAMI venv the numbers
above were measured in:

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
DEEPSEEK_V41_SCHEME=tp64 DEEPSEEK_V41_CHECKPOINT=/mnt/nvme/hf \
python -m pytest contrib/models/deepseek_v41_trn3/test/integration/test_model.py \
    -v -s --timeout=14400
```

`DEEPSEEK_V41_SCHEME=tp64` is the default (`tp8pp8` runs the pipeline scheme); no other
variable is needed -- every optimization above is on by default. The run takes about
10 minutes against a warm compile cache.

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
`vllm bench serve --dataset-name sonnet` over a matrix of input lengths and
batch sizes, reporting TTFT, TPOT and aggregate decode throughput per cell.
The performance table above comes from two runs, batch 1 and batch 10, one after the
other, from the repository root:

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate

# batch 1, input 512 / 1024 / 2048 / 8192 (the default lengths)
DEEPSEEK_V41_SCHEME=tp64 \
python contrib/models/deepseek_v41_trn3/test/performance/run_bench.py --model /mnt/nvme/hf \
    --dataset-path contrib/models/deepseek_v41_trn3/test/performance/sonnet.txt \
    --report contrib/models/deepseek_v41_trn3/test/performance/bench_report.json \
    --bench-timeout 7200 --batch-sizes 1

# batch 10, on engines that admit ten sequences
DEEPSEEK_V41_SCHEME=tp64 DEEPSEEK_V41_MAX_NUM_SEQS=10 \
python contrib/models/deepseek_v41_trn3/test/performance/run_bench.py --model /mnt/nvme/hf \
    --dataset-path contrib/models/deepseek_v41_trn3/test/performance/sonnet.txt \
    --report contrib/models/deepseek_v41_trn3/test/performance/bench_report_ms10.json \
    --bench-timeout 7200 --input-lens 512,1024,2048,8192 --batch-sizes 10
```

`--dataset-path` is needed unless the sonnet text is at the default
`/mnt/nvme/datasets/sonnet.txt` (a copy ships in `test/performance/`), and
`DEEPSEEK_V41_MAX_NUM_SEQS=10` is what makes the batch-10 cells batch 10: without it
the port clamps the engine to one sequence and the cell's ten requests queue.
`--report` names the output (default `test/performance/bench_report.json`; each cell's raw
`vllm bench serve` result and server log land in `cells/` beside it, and neither is
versioned) and `--bench-timeout` only bounds a wedged run. Each cell gets its own server,
sized to its context, and is warmed before it is measured; the batch-1 run takes about 25
minutes and the batch-10 run about 30 against a warm compile cache.

Sonnet input is deliberate: real English gives the drafter something to predict
(random token ids held acceptance to 11-18% against 41-49% on prose, which made
every throughput figure a lower bound), and with `--sonnet-prefix-len 0` no two requests
share a prefix the prefix cache could serve instead of computing. See
[`test/performance/README.md`](test/performance/README.md) for the matrix, the
flags and how to read the output; `--input-lens` and `--batch-sizes` narrow a run
to the cells you care about.

### Optional: unit and module tests

`test/unit/` holds the megakernels' component tests against a CPU reference
(`test_layer.py`, `test_draft.py`, `test_pp.py`; the optimized kernels against
the generic body, `test_opt_kernels_diff.py`; a fused decode launch against its
members launched one by one, `test_fused_decode.py`; the fused greedy head,
`test_head_kernel.py`; all run on N cores through `kharness_mp.py` from `test/unit/`,
e.g. `python kharness_mp.py 8 test_head_kernel.py`,
`python kharness_mp.py 8 test_fused_decode.py --embed --groups r0,r0_e,r2_src_idx,r2,r2,r2,r2,r2`
for the step's first launch and `--head --groups r1_idx,r1,r1,r1,r1_idx,r1_tap,r1_tap,r1_tap`
for its last), the hardware probes that established the kernels' conventions
(`probe_*.py`) and `compile_entries.py`, which traces every specialized kernel
entry without a device. The accuracy test above remains the gate.
