# Contrib Model: DeepSeek-V4.1-Flash (text) on 8 Trainium3 devices

Serving [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
with vLLM on **8 of a trn3.48xlarge's 16 devices** through the `vllm-neuron` plugin.
The architecture string is `DeepseekV41ForCausalLM`, taken from the checkpoint's own
`config.json`.

This folder is the 8-device variant of [`contrib/models/deepseek_v41`](../deepseek_v41/README.md)
(the 16-device, 64-rank port). It was copied from it and changed only where 32 ranks need
it; the architecture, the megakernels and their history are documented there and in
[`src/README.md`](src/README.md). The two folders are independent: different package names
(`deepseek_v41` / `deepseek_v41_8d`), no shared files, and the original is untouched.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

## What differs from the 16-device port

| | `deepseek_v41` (16 devices) | `deepseek_v41_8d` (this folder) |
| --- | --- | --- |
| Ranks | 64 (16 devices x 4 logical cores, LNC=2) | **32** (8 devices x 4), `config.TENSOR_PARALLEL_SIZE = 32` |
| Scheme | `tp64` (default) or `tp8pp8` | **`tp32`** only (`DEEPSEEK_V41_SCHEME`; the 64-rank pipeline scheme is rejected) |
| Attention / dense projections | TP8, replicated 8 times | TP8, replicated 4 times; world all-reduces over 32 ranks |
| Decode MoE (fused bs1 launches) | TP8 x EP8: 8 groups of 48 experts, 288 hidden columns a rank | **TP8 x EP4**: 4 groups of 96 experts, 288 hidden columns a rank |
| Routed experts, deployment layout (prefill, bs10) | 6 a rank | 12 a rank |
| Shared expert, decode layout | 72 blocks of 32 columns over 128 cores: one a core | two a core (`moe_tp_layout.shared`, the kernels read the width from the tensors) |
| Engram tables | bf16, 2 x 2.86 GiB a rank | **fp8 + e8m0 scales as in the checkpoint**, 2 x 2.95 GiB a rank (bf16 would be 2 x 5.72); the kernels dequantize after the row gather, exactly |
| fp8 linears' per-op copies | kept beside the MX copies | freed once the MX copy is made (`DEEPSEEK_V41_FREE_PEROP`) |
| bs1 decode MoE kernel | 4 static expert slots a core | 5 a core, the down GEMM's contraction packed across slots and the two shared blocks in one tile (`b1moe.PACK_DOWN`, `SHARE_PACK`) |
| Vocabulary shard | 2048 rows a rank | 4096 rows a rank |
| bs1 decode step | 6 NEFF executions: prologue, 3 fused launches, the verify graph, the drafter | **4**: the prologue inside the first fused launch, the verify inside the drafter launch (`DEEPSEEK_V41_PROLOGUE_FUSED`, `DEEPSEEK_V41_DRAFT_TAIL`) |
| DSpark drafter kernel (bs1) | as in the 16-device port | the same drafts in ~0.1 ms less (`draft_decode/`: `HC_MIX_STACK_ROWS`, `HEAD_SPLIT`, `KV_ROWS_CORE1`, `BIAS_CAST_ACT`) |
| Batched drafter (bs10) | shared expert sharded over the world (`layout.SH_TP`) | replicated (at 32 ranks the sharded one drafted worse together with `MAIN_PROJ_TP`) |

Why each change is there:

- **HBM.** A logical core has 36 GiB. At 32 ranks the resident weights are twice as many
  bytes a rank (12 routed experts a layer, a twice-larger engram shard), and the fused bs1
  decode launches add their TP x EP copy of every layer's MoE (0.235 GiB a layer, 9.4 GiB).
  As copied, the port ran out of device memory building that copy (28.7 GiB in use before
  it). Keeping the engram shards in fp8 and freeing the per-op fp8 weights brought it to
  20.4 GiB before and ~29.7 GiB after; the expert / engram budgets
  (`DEEPSEEK_V41_EXPERT_CACHE_BYTES`, `DEEPSEEK_V41_ENGRAM_TABLE_BYTES`) default to 12 GiB.
- **The decode MoE.** With 4 groups a group's 96 experts see about twice the active experts
  of a 48-expert group, so a rank streams and multiplies about twice the routed weights a
  step, and more than 8 active experts in a group (the dynamic overflow loop, ~50 us) became
  common. The packed down contraction (3 matmuls a 512-column tile for a core's slots
  instead of one a slot) and one shared term made a static slot cheap enough for a fifth.
- **The shared expert** has 72 blocks of 32 hidden columns: one a core at 64 ranks, two a core
  here (zero blocks where there are fewer).

## Requirements

| | |
| --- | --- |
| Instance | `trn3.48xlarge`; the engine uses **8 devices** (devices 0-7, logical cores 0-31, by default) |
| Parallelism | `tensor_parallel_size = 32`, set by `src/config.py` |
| Checkpoint | 475.3 GiB over 48 shards, on **local NVMe** |
| HBM | ~30 GiB of a logical core's 36 at bs1 decode |

The engine's workers take logical cores `0 .. 31`, i.e. devices 0-7 (the plugin's
`neuron_worker` default; `NEURON_VISIBLE_DEVICES` is its knob for another set of 32 logical
cores, not tested here, and it rejects `NEURON_RT_VISIBLE_CORES`). Devices 0-7 are what
everything below was measured on; the other 8 devices stay free.

The checkpoint has to be on instance store, not EBS (32 ranks read it at startup), and
`vm.overcommit_memory` must stay at `0`: compiling forks `neuronx-cc` out of every worker.
Speculative decoding needs the plugin change described in [`plugin-core/`](plugin-core/README.md)
(already in this branch's plugin); without it the model serves one token a step.

## Registering the model

Importing the package is the whole registration step (see the
[16-device README](../deepseek_v41/README.md#registering-the-model) for what it does and
why the import has to come first). The package here is named **`deepseek_v41_8d`**:
`test/deepseek_v41_8d` is a symlink to `src/`, and `test/model_package.py` loads it and puts
the folder on `PYTHONPATH` for vLLM's subprocesses.

Offline:

```python
import sys
sys.path.insert(0, "contrib/models/deepseek_v41_8d/test")
import deepseek_v41_8d  # noqa: F401 - the import *is* the registration

from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/DeepSeek-V4.1-Flash",
    tensor_parallel_size=32,
    max_model_len=2104,
    max_num_seqs=1,
    quantization="fp8",
)
out = llm.generate(["What is the capital of France?"], SamplingParams(temperature=0.0, max_tokens=32))
print(out[0].outputs[0].text)
```

Online, through the thin entry point the benchmark uses (it imports the package, then runs
vLLM's `api_server`):

```bash
PYTHONPATH=contrib/models/deepseek_v41_8d/test \
python -m performance._serve_main \
    --model /path/to/DeepSeek-V4.1-Flash \
    --tensor-parallel-size 32 \
    --max-model-len 2104 \
    --max-num-seqs 1
```

Both packages register the same `transformers` model type (`deepseek_v41`) and vLLM
architecture, so load one of them per process.

## Compatibility Matrix

| Instance / SDK version | 2.33 | 2.32 and earlier |
| --- | --- | --- |
| Trn3 (8 devices) | Working | Not tested |
| Trn2 | Not tested | Not tested |

Validated 2026-11-03 on a `trn3.48xlarge` with the DLAMI venv
`aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0` (vLLM 0.29.0, torch 2.13.0,
`libtorch-neuronx-lite` 2.13.0.1.0.3112, `neuronx-cc` 2.28.7631.0, `nki` 0.7.0, runtime
2.35.30.0, driver 2.31.7.0) and this branch's `vllm-neuron`, checkpoint revision
`314113f8a7dadb388215396ba5fbee2c610a04d6`.

## Results

Measured 2026-11-03 on devices 0-7 (32 ranks), every optimization at its default
(`test/performance/bench_report.json`, `bench_report_ms10.json`, per-cell results in
`test/performance/cells/`; sonnet input, 128 output tokens, greedy). In parentheses: the first
8-device version (commit 83d1d76):

| Cell | TTFT p50 ms | TPOT p50 ms | decode tok/s | accept % | median ITL ms |
|------|-------------|-------------|--------------|----------|---------------|
| in512_bs1 | 149 (147) | 3.64 (3.67) | 206.7 (204.8) | 51.3 (51.3) | 12.68 (12.98) |
| in1024_bs1 | 245 (244) | 4.10 (4.16) | 165.9 (164.9) | 43.5 (43.5) | 12.70 (12.89) |
| in2048_bs1 | 428 (428) | 4.08 (4.13) | 135.1 (134.1) | 44.6 (44.6) | 12.78 (13.05) |
| in8192_bs1 | 1709 (1707) | 4.23 (4.31) | 56.9 (56.7) | 44.6 (44.6) | 13.25 (13.49) |
| in512_bs10 | 790 (791) | 17.41 (20.81) | 405.4 (318.7) | 46.1 (38.6) | 39.89 (43.97) |
| in1024_bs10 | 1314 (1311) | 20.56 (24.20) | 299.7 (262.1) | 45.7 (40.8) | 41.91 (45.07) |
| in2048_bs10 | 2312 (2313) | 27.05 (29.03) | 214.3 (190.5) | 47.8 (41.0) | 42.61 (46.03) |
| in8192_bs10 | 9401 (9398) | 75.43 (75.80) | 66.8 (64.4) | 46.6 (38.9) | 49.80 (49.70) |

The 16-device port's last validated run (r5r, commit 8e182aa) for comparison: bs1 TPOT
3.77 / 3.71 / 3.84 / 4.00 ms (median ITL 12.70 / 12.63 / 12.68 / 13.22), bs10 TPOT 15.83 / 19.50 /
26.63 / 71.82 ms.

Every cell completed every request, every request emitted its 128 tokens, and the texts are
coherent answers to the sonnet prompts; at batch 1 they are token-for-token the first 8-device
version's in every cell (and so is the draft acceptance: the bs1 changes below keep the arithmetic).
What moved a decode step since that version:

- **Batch 1 (-1.5 to -2.3 % median ITL):** a step is four NEFF executions instead of six. The
  step's verify runs inside the drafter launch (`DEEPSEEK_V41_DRAFT_TAIL`), which the host launches
  straight on its executor (`DEEPSEEK_V41_DRAFT_DIRECT`), and the step prologue is the first member of
  the step's first fused launch (`DEEPSEEK_V41_PROLOGUE_FUSED`). The drafter kernel itself is ~0.1 ms
  shorter (hc_in's mix GEMM without the per-tile PE transposes, the markov token loop split over the
  two cores, the next stage's KV write off core 0's critical path), with the same drafts; the
  target's greedy head does its argmax on both cores with one all-gather. Each
  NEFF switch costs the device ~45-75 us plus a ~50-70 us cross-rank barrier inside the launch.
- **Batch 10:** the batched drafter keeps the shared expert replicated (`layout.SH_TP` off: with it and
  `MAIN_PROJ_TP` together at 32 ranks it drafted worse, 39 % acceptance against 47 %), and a 60-row
  step runs its MoE in the TP x EP layout up to a 4096-token context (`mk_model.moe_tp_max_t()`).

See [`src/README.md`](src/README.md#the-8-device-deployment) for the switches and what was tried and
dropped.

The integration test passes (`test/unit/verifier_reports/tp32_2026-11-03.json`, 7:58 wall):
`short-prose`, `smoke-test` and `trivia-capital-france` are token-for-token the 16-device
port's (32 / 32 / 18 tokens, `...The capital of France is **Paris**.`), and
`long-needle-2048-0` recalls `5591-C67` with the same 32 tokens up to the sentence after the
code, then continues `...The maintenance log for sector 159 ...`, one of the continuations the
16-device port's own validated runs produced; all four are the first 8-device version's texts.

During serving the 32 workers hold `/dev/neuron0` .. `/dev/neuron7` only.

## Testing

### Accuracy (integration)

Serves the model through vLLM on 32 ranks and checks the generated text. Needs devices
0-7 to itself. From the repository root:

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
DEEPSEEK_V41_CHECKPOINT=/mnt/nvme/hf \
python -m pytest contrib/models/deepseek_v41_8d/test/integration/test_model.py \
    -v -s --timeout=14400
```

About 10 minutes against a warm compile cache (20 cold). It asserts that every phase ran,
every case produced tokens and every case declaring an expected substring contains it
(`long-needle-2048-0` must recall an access code planted 2008 tokens deep,
`trivia-capital-france` must say Paris); every completion is printed. See the
[16-device README](../deepseek_v41/README.md#accuracy-integration) for why the test checks
text rather than logits.

### Speed (performance)

`run_bench.py` serves the model and drives it with `vllm bench serve --dataset-name sonnet`
(greedy, one server per input length, warmed before it is measured):

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate

# batch 1, input 512 / 1024 / 2048 / 8192
python contrib/models/deepseek_v41_8d/test/performance/run_bench.py --model /mnt/nvme/hf \
    --dataset-path contrib/models/deepseek_v41_8d/test/performance/sonnet.txt \
    --report contrib/models/deepseek_v41_8d/test/performance/bench_report.json \
    --bench-timeout 7200 --batch-sizes 1

# batch 10, on engines that admit ten sequences
DEEPSEEK_V41_MAX_NUM_SEQS=10 \
python contrib/models/deepseek_v41_8d/test/performance/run_bench.py --model /mnt/nvme/hf \
    --dataset-path contrib/models/deepseek_v41_8d/test/performance/sonnet.txt \
    --report contrib/models/deepseek_v41_8d/test/performance/bench_report_ms10.json \
    --bench-timeout 7200 --input-lens 512,1024,2048,8192 --batch-sizes 10
```

Per-cell results land next to the report (`cells/`). See
[`test/performance/README.md`](test/performance/README.md) for the flags and the output.

### Optional: unit tests

`test/unit/` is the 16-device port's kernel test suite, copied with the package renamed
(`kharness_mp.py <N> <test>` runs a test on N cores); the integration test is the gate.
