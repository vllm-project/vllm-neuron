# Equivalence Report: DeepSeek-V4.1-Flash, per-op NKI path -> megakernel path

> **Round 2 (2026-10-23, evening) is appended at the end:** the megakernels were
> rebuilt on the tensor engine's native block-scaled fp8 / fp4 matmul, keys
> selected on chip through tensor indirection, and token-grouped routed experts.
> The sections up to "Stage 7 (bench)" document round 1 (bf16-dequant GEMMs,
> dense experts); "Round 2" re-establishes every stage on the new kernels.

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

---

# Round 2: native MX matmuls, on-chip key selection, token-grouped experts

Same source (the per-op path), same method, new target: `kernels/mk_mx.py`
(fp8 / fp4 `nc_matmul_mx` with host-packed block-scaled weights and `quantize_mx`
activations), `kernels/mk_common.py:attn_token_body_ti` (QK over a resident
transposed slot table through `.indirect()`), `kernels/mk_moe.py` (row lists by
`nonzero_with_count`, a runtime-bounded loop per expert, `dma_compute`
scatter-adds). Hardware conventions established by device probes
(`test/unit/probe_mx.py`, `probe_dyn.py`, `probe_ti.py`, `probe_lnc.py`):

| Probe | Finding |
|-------|---------|
| fp4 weight x on-chip fp8 activation, MX matmul vs fp32 CPU GEMM | rel_l2 4.9e-5 (nibble order, scale quadrant layout confirmed) |
| fp8 weight x fp8 activation | rel_l2 2.4e-4, R = 0.15 (the engine rounds its 4-element partial products; e4m3 subnormal codes honoured; the same with single-binade weights, so inherent) |
| `quantize_mx` vs the CPU emulation `ref_layer.act_quant_mx` (random bf16) | 0 of 65536 elements differ (scale `2^(floor(log2 amax) - 7)`, e4m3fn RNE) |
| `.indirect()` moving operand, 640 selected columns | exact; index snake `[i % 16, i // 16]` replicated per quadrant |
| `nonzero_with_count` + `fori_loop` (scalar bound, zero trips) + indirect gather + `dma_compute` scatter-add | exact (rel_l2 1.2e-7) |
| LNC=2 dynamic loop with unequal trip counts on the two cores | **hangs** (runtime timeout) -- hence both cores run every expert and split each tile |
| `nc_matmul(perf_mode=double_row)` fp8, unscaled operands (a one-off probe, not kept) | works, rel_l2 1.3e-4 (the same internal partial-product rounding as MX) |
| the checkpoint's 32x32 block scales folded to one scale per column, the only scale double-row can carry | 25% of the nonzero weight codes flush to zero at the checkpoint's 13-binade spread: double-row cannot take these weights exactly; `nc_matmul_mx` applies the block scales in the array, at twice double-row's packing |
| `[128 x 15360] @ [15360 x 5120]`, double-row vs MX, 20 executions | 9.07 vs 8.22 ms (host round trip included) |

## Stage 1 (round 2): smoke test, real model

Integration test (`test/integration/test_model.py`), megakernel path, 2026-10-23
19:11, 15:00 wall (round 1: 17:56):

| Case | Expected | Result |
|------|----------|--------|
| long-needle-2048-0 | `5591-C67` | ok (`... 5591-C67, and it must be entered before the second alarm cycle completes.`, 27 tok, stop) |
| trivia-capital-france | `Paris` | ok -- the completion `We need answer simple. Need final.</think>The capital of France is **Paris**.` (18 tok) is identical to the per-op path's and to round 1's |
| short-prose, smoke-test | (none) | coherent 32-token completions |

`verifier: PASSED -- 2 of 2 expected substring(s) matched` (log
`/mnt/nvme/ds41_mk2_integration.PASS.log`, report `/mnt/nvme/ds41_mk2_verifier_report.json`).

## Stage 2 (round 2): components, 8 ranks, random weights

| Component (test) | rel_l2 | R | Result |
|------------------|--------|---|--------|
| layer: q (stage 2) | 3.8e-3 | - | PASS |
| layer: attention output `o` (stage 4), T = 6 / 128 (ratio 2 source) / 2048 | 4.6e-3 / 9.7e-3 / 8.5e-3 | - | PASS |
| layer: output projections from the kernel's own `o` (stage 51) | 4.3e-3 | 2.6 | PASS |
| layer: MoE partial from the kernel's own normed input (stage 82), T = 6 / 128 / 2048 (multi-tile experts) | 3.4e-3 / 4.1e-3 / 4.4e-3 | 2.0 / 2.5 / 2.7 | PASS |
| layer: shared expert only (stage 83), routed only (stage 84) | 0 (bit-exact) / 3.4e-3 | 0 / 2.0 | PASS |
| layer: router weights / logits (stage 81) | exact / inherited 7e-3 | - | PASS |
| layer: compressed latents / index keys written (ratio 2 source, T = 128) | 6.7e-3 / 1.7e-2 | - | PASS |
| layer: pooling caches, index top-k set agreement | 1.1e-4 / min 1.006 | 0.07 | PASS |
| layer: engram stream (stage 0), tap | 8.0e-4 / 8.0e-4 | - / 0.48 | PASS |
| layer: KV cache rows written / untouched | 2e-3..6e-3 / 0 | - | PASS |
| drafter: normed, q, kv, o (stages 11-14) | 1.5e-6 / 4.5e-3 / 5.4e-4 / 1.4e-2 | - | PASS |
| drafter: MoE partial from the kernel's own input (stage 18) | 3.6e-3 | 2.2 | PASS |
| drafter: head from the kernel's own hidden (stage 4) | 3.2e-4 | - | PASS |
| drafter: main-stream KV written, 3 stages, decode and prefill (`--start 0`) | 1.2e-2..1.4e-2 | - | PASS |
| drafter: prefill ids (`--start 0`) | exact, confidence exact | - | PASS |

On the zero-weight clause: an expert's row list holds exactly the rows whose
routing weight for it is nonzero (`wexp > 0`), so no `(token, expert)` pair with
a zero weight is ever gathered or multiplied; the row lists were checked
element-for-element against `torch.nonzero` in `probe_dyn.py` and the per-expert
counts drive the loop bounds. The only extra rows a tile carries are the
zero-filled padding of an expert's last partial 128-row tile (the tensor engine's
row granularity, as the per-op path's 128-row buckets): they are not token-expert
pairs, they gather a zero row, their result lands in junk accumulator rows that
are never read, and they add no tensor-engine time because the tile's cost is
the expert's weight stream, which is the same for 1 or 128 real rows.

Components agree with the reference from their own inputs at R of 2-3 (round 1:
0.02-9). The difference is the fp8 MX matmul's internal rounding (2.4e-4 a GEMM
output), which every fp8 re-quantization downstream turns into one-ulp flips, so
a layer's full stream on random weights lands at 3-4e-2 (T = 6), 7e-2 (two spans),
9e-2 (T = 2048) against 2-5e-2 in round 1, and the drafter's three-stage stream
at 0.37 against ~0.2 (its drafted ids on random weights: exact at prefill, two
late flips at decode where the reference's top-2 gap is under 0.6). The unit
tests' stream-level tolerances were set to that floor; the component checks
are the gates.

## Stage 3/4 (round 2): faults found and fixed

| Fault | Symptom | Fix |
|-------|---------|-----|
| scale tile only partly written inside the dynamic expert loop | `NCC_IGCA108` (loop-carried dependency) | `memset` the tile in the body |
| experts split by parity across the two cores (unequal trip counts) | 8-rank runs hang until the runtime timeout | both cores run every expert, tile split along `inter` (`probe_lnc.py`) |
| 1x1 engram dummy viewed as the x4 dtype on non-engram layers | trace assertion at the first real forward | view only when `has_engram` |
| drafter's per-stage attention / shared-expert weights not in the MX layout | shape derivation divides by zero at the first draft | `mk_model.draft_weights` stacks `_fp8()` layouts |
| three stages sharing one `mkvr` / `mkv` scratch at prefill (`at_zero`) | main-stream KV of stages 0-2 corrupted (cos 0.5-0.7) | one scratch pair per stage |

## Stages 5+6 (round 2): real model, per layer, prefill of the trivia case

`/mnt/nvme/dump_mk4` against `/mnt/nvme/dump_op` (rank 0, step 1 = 37-token prefill):

| Layer output | round 2 rel_l2 | round 1 rel_l2 | cos (round 2) |
|--------------|---------------|----------------|---------------|
| embed | 0 | 0 | 1.000 |
| layer 0 | 3.1e-2 | 3.0e-3 | 0.9998 |
| layers 1-9 | 2.2e-2 .. 2.9e-2 | 7e-3 .. 2.7e-2 | 1.000 |
| layers 10-14 | 1.6e-2 .. 2.0e-2 | | 1.000 |
| layers 15-19 | 8.0e-3 .. 9.1e-3 | | 1.000 |
| layers 20-39 | 3.8e-3 .. 6.0e-3 (layer 20: 1.6e-2) | 3.7e-3 .. 5.4e-3 | 1.000 |
| final normed hidden | 0.235 | 0.23 | 0.9725 (round 1: 0.973) |

Decode step (step 2 = 1 real + 5 draft rows, `/mnt/nvme/dump_mk5`):

| Layer output (step 2) | round 2 rel_l2 | round 1 rel_l2 | cos (round 2) |
|-----------------------|---------------|----------------|---------------|
| layer 0 / 1 | 1.1e-2 / 3.3e-2 | 6.7e-3 / 2.2e-2 | 0.9999 / 0.9994 |
| layers 5-39 | 0.16 .. 0.29 | 0.18 .. 0.31 | 0.96 .. 0.99 |
| final normed hidden | 0.33 | 0.35 | 0.946 (round 1: 0.94) |

(The decode rows are small-magnitude draft rows, where the same absolute noise is
a larger fraction, as in round 1.) The dump run's own verifier case passed, and
the integration test was re-run on the final code after the dead-code cleanup:
`verifier: PASSED -- 2 of 2 expected substring(s) matched`, 10:05 wall
(`/mnt/nvme/ds41_mk2_integration.final.log`).

The early layers sit higher than in round 1 (the fp8 MX rounding, re-quantized
through the attention projections of a layer whose stream is still small), the
late layers and the end of the stack are where round 1 was. Note that the per-op
path is not the reference either: its `224`-peak activation round trip keeps one
bit less than the hardware's `quantize_mx` in a fifth of the blocks, so part of
the per-layer difference is the megakernel path sitting closer to the model's
own `448` rule.

## Stage 7 (round 2): bench

`test/performance/run_bench.py`, 8 cells, 2026-10-23 19:13-20:27 (round 1: 2h20m),
`bench_report.json` (round 2) against `bench_report.mk1.json` (round 1, dense
experts, bf16-dequant GEMMs) and `bench_report.perop.json`. Medians; `accept` is
vLLM's spec-decode acceptance over the cell.

| Cell | TTFT p50 ms (per-op / round 1 / round 2) | TPOT p50 ms | decode tok/s | accept % |
|------|------------------------------------------|-------------|--------------|----------|
| in512_bs1 | 1060 / 625 / **560** | 99.4 / 73.1 / **31.6** | 9.3 / 13.3 / **28.2** | 52.0 / 49.7 / 46 |
| in512_bs10 | 62766 / 44035 / **21345** | 99.0 / 73.2 / **30.5** | 9.4 / 13.2 / **28.7** | 51.8 / 49.4 / 47 |
| in1024_bs1 | 1701 / 1157 / **1064** | 101.0 / 66.0 / **31.1** | 8.9 / 13.3 / **25.7** | 49.4 / 49.0 / 48 |
| in1024_bs10 | 65604 / 45996 / **22767** | 99.7 / 65.9 / **31.0** | 9.1 / 13.3 / **25.7** | 49.5 / 48.9 / 48 |
| in2048_bs1 | 3037 / 2134 / **1954** | 100.9 / 69.0 / **29.7** | 8.0 / 11.7 / **22.4** | 46.4 / 47.6 / 49 |
| in2048_bs10 | 74895 / 49211 / **27515** | 102.0 / 68.7 / **29.7** | 7.9 / 11.7 / **22.3** | 46.7 / 47.4 / 49 |
| in8192_bs1 | 13419 / 9261 / **8545** | 102.5 / 70.3 / **33.8** | 4.8 / 7.0 / **10.0** | 47.2 / 45.9 / 42 |
| in8192_bs10 | 129633 / 91322 / **65508** | 102.9 / 70.5 / **33.5** | 4.8 / 7.0 / **10.1** | 47.5 / 44.9 / 42 |

The batch-10 cells are serialized on a `max_num_seqs=1` engine (as before), so
their TTFT is queueing. TPOT halves against round 1 (the decode step no longer
decodes six experts' weights to bf16 scratch and multiplies all rows by them; it
reads only the picked experts' fp4 weights and multiplies them natively), TTFT
drops 8-10% (the prefill's expert work is now one or two tiles per expert instead
of sixteen, but the attention projections and the per-token attention dominate
there). Draft acceptance stays within 1-4 points of round 1 in both directions
(the 8k cells lose 4 points, the 2k cells gain 1-2), the distributional check
this port has. Verdict: no regression, TPOT -55%, decode tok/s x2.1 at short
contexts and x1.4 at 8k.
