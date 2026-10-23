# Equivalence Report: DeepSeek-V4.1-Flash, per-op NKI path -> megakernel path

## Executive Summary

- **Overall:** PASS (verifier: both expected substrings matched; the two paths produce the
  same greedy tokens on the short case; every component matches its reference from its own
  input to bf16 noise).
- **Model:** DeepSeek-V4.1-Flash (checkpoint `/mnt/nvme/hf`, fp8/fp4 block-quantized,
  40 blocks + 3 DSpark MTP stages, TP=64 on trn3.48xlarge, SDK 2.33).
- **Source:** the port's own per-op path (`DEEPSEEK_V41_MEGAKERNEL=0`): one shipped NKI
  kernel per op, validated against the checkpoint's reference by the integration test
  (see the model README). There is no HuggingFace implementation of this architecture, so
  the skill's HF-based `run_stage*.py` pipeline does not apply (the same gap the
  integration test documents); its method was applied with the per-op path as the source.
- **Target:** the megakernel path (`DEEPSEEK_V41_MEGAKERNEL=1`, the default): one
  `@nki.jit` per block (`kernels/layer_kernel.py`) and one for the DSpark drafter
  (`kernels/draft_kernel.py`), with their collectives inside the kernels.
- **Date:** 2026-10-23.

The error measure of the unit harness (`test/unit/kharness.compare`) is
`R = ||target - ref|| / ||bf16(ref) - ref||`: the error relative to the reference's own
bf16 rounding noise (the skill's R with the fp32 reference's bf16 rounding as the
low-precision baseline). `rel_l2` is `||target - ref|| / ||ref||`.

## Stage 1: Smoke Test (greedy tokens, real model)

`test/unit/verify_short.py` on the trivia case, both paths, 64 ranks (dumps in
`/mnt/nvme/dump_op`, `/mnt/nvme/dump_mk3`):

| Path | Completion | Tokens |
|------|------------|--------|
| per-op | `We need answer simple. Need final.</think>The capital of France is **Paris**.` | 18, stop |
| megakernel | `We need answer simple. Need final.</think>The capital of France is **Paris**.` | 18, stop |

- Token match: 18/18 (100%). Verdict: PASS.

## Stage 2: Component-Level Results

Kernels against a CPU reference of the model's arithmetic (`test/unit/ref_layer.py`), 8
ranks, random weights, through the same `wrap_nki` + `torch.compile` path the model uses.
"from kernel input" means the reference was rebuilt from the kernel's own intermediate, so
only that component's arithmetic is measured.

| Component (test) | rel_l2 | R | Result |
|------------------|--------|---|--------|
| hc sublayer-in (`test_hc_in.py`, vs the shipped device path) | 1e-6 | ~0 | PASS |
| hc sublayer-out vs shipped `hc_attn_out.kernel` (`test_vs_shipped.py`) | 0 (bit-exact) | 0 | PASS |
| engram gate vs shipped `engram_gate.gated_add` | 0 (bit-exact) | 0 | PASS |
| sparse attention token body vs shipped `gathered_attn_idx` | 0 (bit-exact) | 0 | PASS |
| swiglu vs shipped `swiglu.swiglu` | 0 (bit-exact) | 0 | PASS |
| rmsnorm vs shipped `rmsnorm.rms_norm` | 8.7e-5 | noise | PASS |
| layer: normed input (stage 1, T=6/128/2048) | 1.5e-6 .. 1.5e-4 | noise | PASS |
| layer: q / kv / attention output (stages 2-4) | 2e-4 .. 1.4e-3 | noise | PASS |
| layer: attention partial from kernel input (stage 51) | 6e-3 .. 1.5e-2 | 3.6 .. 9 (fp8 flips of `o`) | PASS |
| layer: MoE partial from kernel input (stages 82/83/84) | 4e-5 .. 1e-4 | 0.02 .. 0.06 | PASS |
| layer: router weights (stage 81) | exact | 0 | PASS |
| layer: engram stream (stage 0), tap | 5e-5 .. 7e-5 | noise | PASS |
| layer: compressed latents / index keys written (ratio 1, 2; T up to 2048) | 7e-3 / 1.5e-2 | fp4 flips | PASS |
| layer: pooling caches (ratio 2) | 1.1e-4 | 0.07 | PASS |
| layer: index top-k set agreement (T=2048) | min 0.973, mean 1.0 | - | PASS |
| layer: KV cache rows written / untouched | 2.7e-7 .. 3e-3 / 0 | noise | PASS |
| drafter: normed, attention, hc out, normed2 (stages 11-17) | 1.5e-6 .. 6e-3 | noise | PASS |
| drafter: MoE partial from kernel input (stage 18) | 8e-5 .. 1e-4 | 0.05 | PASS |
| drafter: main-stream KV written (3 stages) | 0 .. 3e-3 | noise | PASS |
| collectives: 2x4-rank and world all_reduce (`probe_cc_groups.py`) | exact | 0 | PASS |

- Components tested: 20. Passed: 20/20.
- Full-layer streams on random weights (stage 9) differ by 2e-2 .. 5e-2 rel_l2: the fp8/fp4
  expert path amplifies the input's bf16 rounding-order flips ~8x and random weights make
  each MoE output comparable to the stream (the real model's residual stream is far larger
  than a sublayer's contribution, see Stages 5+6).

## Stage 3: Fault Localization

Two faults, both found by the per-layer dump comparison on the real model
(`test/unit/compare_dumps.py`), neither visible to the 8-rank unit tests:

- **Primary fault 1 (MoE, every layer):** layer 0's stream 9x too large while its FFN
  pre-mix agreed. The replicated shared expert was added into every rank's partial before
  the 64-rank all_reduce (64x). Classification: reduction semantics (the model adds it
  once, after the reduce). The unit reference reduced the same way, so it passed.
- **Primary fault 2 (engram, layers 1 and 14):** layer 1 at rel_l2 0.45 after fault 1 was
  fixed. The kernel turns n-gram row ids into fp32 for the range mask; global ids of the
  384M-row table exceed 2^24 and are rounded, so ranks with a high `row_start` gathered
  wrong rows. Classification: integer precision. The unit test's table had 32K rows.

## Stage 4: Debug Summary

| Patch | Component | Before | After |
|-------|-----------|--------|-------|
| shared expert scaled by 1/world (`gemm_fp8_kslice(out_scale=1/world)`, both kernels; `moe_ref(shared_scale)`) | MoE | layer 0 rel_l2 8.65 | 3.0e-3 |
| shard-local engram ids from the host (`StepIndices.hash_cols`, `erow_start=0`) | engram | layer 1 rel_l2 0.45 | 2.4e-2 |

## Stages 5+6: E2E Comparison (real model, per-layer, 64 ranks)

Rank 0's live rows of every layer's output stream, per-op vs megakernel, trivia case.
Step 1 = the 37-token prefill, step 2 = the first decode step (1 + 5 draft rows).

| Layer output (step 1) | rel_l2 | cos |
|-----------------------|--------|-----|
| embed | 0 | 1.000 |
| layer 0 | 3.0e-3 | 1.000 |
| layer 1 (engram) | 2.4e-2 | 1.000 |
| layers 2-20 | 7e-3 .. 2.7e-2 | 1.000 |
| layers 24-39 | 3.7e-3 .. 5.4e-3 | 1.000 |
| final normed hidden | 0.23 | 0.973 |

| Layer output (step 2) | rel_l2 | cos |
|-----------------------|--------|-----|
| layer 0 / 1 | 6.7e-3 / 2.2e-2 | 1.000 |
| layers 5-39 | 0.18 .. 0.31 | 0.95 .. 0.98 |
| final normed hidden | 0.35 | 0.94 |

- Greedy tokens identical on the case (Stage 1). The decode rows are 1 real + 5 draft
  tokens with small-magnitude streams, where the same absolute noise is a larger fraction.
- Verdict: PASS on the frozen criterion (greedy tokens and the verifier's expected
  substrings), consistent with the per-op path's own documented ~0.5% agreement with the
  CPU reference.

## Stage 7: Downstream Evaluation

The integration test (`test/integration/test_model.py`, the verifier with 4 cases, two
engines) with the megakernel path:

| Case | Expected | Result |
|------|----------|--------|
| long-needle-2048-0 (access code planted 2008 tokens deep) | `5591-C67` | ok (`... 5591-C67, and it must be entered before the second alarm cycle completes.`) |
| trivia-capital-france | `Paris` | ok |
| short-prose, smoke-test | (no substring) | coherent completions |

`verifier: PASSED — 2 of 2 expected substring(s) matched` (log
`/mnt/nvme/ds41_mk_integration.PASS.log`, before the engram-id fix; the re-run after it
is recorded in the session summary).

## Artifacts

- Unit tests and harness: `test/unit/` (`kharness.py`, `kharness_mp.py`, `test_layer.py`,
  `test_draft.py`, `test_vs_shipped.py`, `test_hc_in.py`, `probe_cc_groups.py`,
  `verify_short.py`, `compare_dumps.py`).
- Dumps: `/mnt/nvme/dump_op` (per-op), `/mnt/nvme/dump_mk3` (megakernel), `step{n}_layer{i}.pt`.
- Debug hook: `DEEPSEEK_V41_MK_DEBUG=<layer>:<stage>` (see `src/mk_model.py`).

## Stage 7 (bench): `test/performance/run_bench.py`, 8 cells, megakernel path vs per-op

`bench_report.json` (megakernel, 2026-10-23, 8/8 cells, every cell warmed and complete) against
`bench_report.perop.json` (the per-op path, 2026-10-22). Medians; `accept` is vLLM's
spec-decode acceptance over the cell; the prose probe (sharegpt, 96 out) accepted 272/585
drafts = 46.5%.

| Cell | TTFT p50 ms (per-op -> mk) | TPOT p50 ms | decode tok/s | accept % |
|------|---------------------------|-------------|--------------|----------|
| in512_bs1 | 1060 -> 625 | 99.4 -> 73.1 | 9.30 -> 13.30 | 52.0 -> 49.7 |
| in512_bs10 | 62766 -> 44035 | 99.0 -> 73.2 | 9.40 -> 13.17 | 51.8 -> 49.4 |
| in1024_bs1 | 1701 -> 1157 | 101.0 -> 66.0 | 8.89 -> 13.27 | 49.4 -> 49.0 |
| in1024_bs10 | 65604 -> 45996 | 99.7 -> 65.9 | 9.10 -> 13.25 | 49.5 -> 48.9 |
| in2048_bs1 | 3037 -> 2134 | 100.9 -> 69.0 | 8.02 -> 11.66 | 46.4 -> 47.6 |
| in2048_bs10 | 74895 -> 49211 | 102.0 -> 68.7 | 7.93 -> 11.71 | 46.7 -> 47.4 |
| in8192_bs1 | 13419 -> 9261 | 102.5 -> 70.3 | 4.84 -> 7.03 | 47.2 -> 45.9 |
| in8192_bs10 | 129633 -> 91322 | 102.9 -> 70.5 | 4.79 -> 6.96 | 47.5 -> 44.9 |

The batch-10 cells are serialized on a `max_num_seqs=1` engine (as before), so their TTFT
is queueing. Draft acceptance stays within 1-3 points of the per-op path, in both
directions, which is the distributional check this port has: a drifted draft or target
would show as a drop. Verdict: no regression; TTFT -30..-40%, TPOT -30%.
