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

---

# Round 3: one kernel per phase, archetype and scheme; the TP8 x PP8 pipeline

`kernels/mk_kernels.py` traces a distinct `@nki.jit` entry per (phase, block
archetype, scheme) from the shared bodies. `DEEPSEEK_V41_SCHEME=tp8pp8` runs
vLLM pipeline parallelism (`tensor_parallel_size=8, pipeline_parallel_size=8`):
each of the 8 stages builds and loads only its own 5 blocks, reduces over its 8
ranks, and hands its activations to the next stage through the in-kernel pair
collective of `kernels/pp_kernels.py`; the scheduler keeps 8 batches in flight.
The plugin's V1 model runner gained the non-last-stage path for it
(`neuron_model_runner._pp_not_last_stage`). An earlier, model-internal lockstep
version of the scheme (64 tensor-parallel workers, stages run one after another)
was dropped on 2026-10-24: it overlapped nothing. See `src/README.md`
"Megakernels" for the design.

| Probe (`test/unit/probe_pp_subset.py`, 16 ranks = 2 stages) | Result |
|---|---|
| stage 0 runs kernel A three times, stage 1 kernel B (another NEFF) five times, each an all-reduce over its stage with the full 16-rank mesh declared; then the pair hand-off, then the pair hand-off again after stage 1 ran two more B | 16/16 ranks PASS, 28 s (2026-10-24 00:43): a collective runs on the ranks of its group while the other ranks run other kernels |

## Stage 2 (round 3): components, random weights

| Component (test) | rel_l2 | R | Result |
|------------------|--------|---|--------|
| stage boundary, 2 stages x 8 ranks (`test_pp.py`, T = 6 / 128): stream, pre_mix, top-k, compressed-KV and index-key rows at the written slots, untouched rows | 0 (bit-exact) | 0 | PASS |
| final broadcast (stream, pre_mix, taps) from the last stage's lead rank | 0 (bit-exact) | 0 | PASS |
| engram kernel (`engram_tp8pp8`) vs the CPU read-out, table sharded over 16 ranks | 7.8e-4 | - | PASS |
| tp8pp8 block kernel, 48 experts a rank: MoE from the kernel's own input, T = 6 / 128 / 2048 | 5.8e-3 / 3.9e-3 / 3.8e-3 | 3.5 / 2.4 / 2.3 | PASS |
| tp8pp8 block kernel: attention output, T = 128 | 7.7e-3 | - | PASS |
| tp64 decode-phase entry (per-token key transposes) at T = 128: attention output / projections from the kernel's `o` / MoE | 7.7e-3 / 3.5e-3 / 4.5e-3 | - / 2.1 / 2.7 | PASS |
| tp64 generic entry, full layer T = 6 (unchanged by the split) | 3.6e-2 stream (as round 2) | - | PASS |

The stage plan (`mk_model.PipePlan`, checked on the CPU): 5 blocks a stage,
boundaries after blocks 4, 9, ..., 34, the two pair layouts each cover all 64
ranks, the DSpark taps (blocks 37-39) land in the last stage, the kv sources in
stages 0, 1, 2 and 4 with their consumers downstream of them.

## A hang found by the verifier (round 3), and its fix

The first tp64 verifier run of round 3 passed the long-needle phase and hung in
the short phase at the second prompt's prefill (every rank but 63 in a device
timeout, "missing collectives status ... waiting for barrier END"; rank 63 with
no runtime error and one extra graph compile). `TORCH_LOGS=recompiles` named it:
rank 63 alone re-traced the forward's tail frame (collapse, norm, head, logits
all-gather, sampler, draft seeds) on the guard
`not ___dict_contains((2048, float32, neuron:63), self._vocab_pad_slots)`: the
`-inf` bias of the padded last vocab shard was a lazily filled per-rank dict, so
the frame's guards differed between rank 63 and the rest after the first step,
and the rank recompiled a graph with a collective the other 63 were already
executing. Round 2 never saw it because its per-layer Python loop split the
frame differently. Fix: the bias is a non-persistent buffer built in
`allocate_staging_buffers` (`model.py:vocab_pad_bias`). After the fix the three
short cases (`verify_short.py --case short-prose,smoke-test,trivia-capital-france`)
pass end to end (23:31). The rule and the knobs that located it
(`DEEPSEEK_V41_MK_TRACE`, `DEEPSEEK_V41_MK_IN_FLIGHT`, `NEURON_RT_EXEC_TIMEOUT`) are
in `src/README.md` "Megakernels".

## Stage 7 (round 3): verifier and bench, both schemes

### Verifier (`test/integration/test_model.py`, both phases)

| Scheme | Engine | Result |
|---|---|---|
| tp64 | `tensor_parallel_size=64` | PASSED 2026-10-23 23:44 (10:11 wall): needle `5591-C67` recalled, trivia answered `Paris`; the short phase ran its prompts one at a time (`config.MAX_NUM_SEQS` clamps engines to one sequence unless `DEEPSEEK_V41_MAX_NUM_SEQS` is set) (the hang of the first attempt is the vocab-pad bias story above). Report copied to `/mnt/nvme/ds41_s1_verifier_report.json`. |
| tp8pp8 | `tensor_parallel_size=8, pipeline_parallel_size=8`, synchronous scheduling | PASSED 2026-10-24 03:48 (12:21 wall, kernels cached): needle recalled, trivia answered; report copied to `/mnt/nvme/ds41_pp_verifier_report.json` (log `/mnt/nvme/ds41_pp_integration6.log`). |
| tp64 (final code) | per-scheme kernel packages | PASSED 2026-10-24 22:30 (17:20 wall): needle `5591-C67` recalled, trivia answered `Paris`; `test/unit/verifier_reports/tp64_2026-10-24.json` |
| tp8pp8 (final code) | per-scheme kernel packages, output-carried drafts, batched drafter | PASSED 2026-10-24 23:10 (39:54 wall, kernels compiled on the first requests): needle recalled, trivia answered; `test/unit/verifier_reports/tp8pp8_2026-10-24.json`. Both final runs serve one sequence at a time; the batched paths are checked by `verify_short.py` at `max_num_seqs=4` (17:33 / 17:51: both schemes' texts identical to their one-sequence runs) |

The pipeline scheme's bring-up found, in order: the registered model class has to
declare vLLM's `SupportsPP`; the attention metadata is keyed by the stage's first
block; the drafter kernel wants vocab shards in 512-column multiples
(`VOCAB_SHARD_ALIGN`); the async scheduler cannot feed stages before the sampling
one; under the batch queue, drafts must not be attached to a request before
that step's sampled tokens (first a deferral in
`NeuronScheduler.update_draft_token_ids`, replaced on 2026-10-24 by drafts
carried inside each step's `ModelRunnerOutput` -- see "The draft RPC" below);
and the
two sides of a stage boundary must be traced from identical shapes -- the
runtime hashes a program's collective ops per replica group and refused the
first 512-row boundary with "replica group signature mismatch ... between
peers" (the lockstep version's short-phase failure too) because the receiver's
placeholder top-k was `[T, 512]` while the index-source kernel's real top-k is
`min(index_topk, read_len)` = 384 columns at a 608-token context
(`mk_model.topk_width` now sizes both sides; the boundary kernel is also one
program for both roles, the role a runtime weight, and `test_pp.py --T 512
--R 257` covers the shape). With those,
the needle-phase steps reproduce the flat scheme's draft proposals token for
token (seeds 223, 26307, 5770, 305; one trailing draft differs by an fp8 flip),
and the short phase run on its own (`verify_short.py`, 3 prompts, `max_num_seqs=4`)
PASSED at 03:34 with the flat scheme's first four proposals exactly
(`[2581, 3720, ...]`, `[1309, 3287, ...]`, `[16145, 16, ...]`, `[671, 3072, ...]`);
its decode steps ran one request per batch (`config.MAX_NUM_SEQS` clamped the
engine to one sequence, see the batched runs below).

### The draft RPC (2026-10-24, found by the 8192-token pipeline cells)

The pipeline bench's `in8192` cells died on the first request: the last stage
traced the sampler with a 1-row logits tensor against 6-row sampling metadata
(`Size does not match at dimension 0 expected index torch.Size([6, 1]) to be no
larger than self torch.Size([1, 256])`, `server_ctx8448_ms1.log`). The prompt
prefills in 2048-token chunks, and the drafter runs on every chunk; the engine
collects drafts with `take_draft_token_ids` after every scheduled step, so a
non-final chunk's (meaningless) drafts were attached to the request while its
later chunks were still in flight, the request became schedulable with five spec
tokens and no bonus token, and the model treated that 5-row single-request step
as a prefill. The same RPC explains why no two batches ever overlapped under the
batch queue: it is a collective RPC answered by the output rank -- the last
stage -- whose worker only replies after the `execute_model` queued ahead of it,
i.e. after the step has drained all 8 stages, so `EngineCore.post_step` blocked
the engine until each step finished before it could schedule the next one.

Fix (plugin): the last stage's runner puts each step's drafts inside that step's
`ModelRunnerOutput` (`neuron_pp_drafts`, only for requests that committed a token
in the step; the worker-to-engine queue pickles the object, so the attribute
survives), `NeuronScheduler.update_from_output` attaches them right after the
sampled tokens of the same step, and `vllm/patches/pp_draft_patch.py` makes
`post_step` skip the RPC under pipeline parallelism with synchronous scheduling.
The same module sizes the executor's shared-memory RPC ring (10 chunks by
default, freed only once all 64 workers have read them, two messages a step, so
the engine would block after about five steps in flight) to `4 x
pipeline_parallel_size` chunks.

**Batched requests on the pipeline (2026-10-24).** Every pipeline run at
`max_num_seqs=4` -- before and after the hand-off change -- "passed" the
verifier's substring check while producing garbage (`ButTheParis France of
France? is S`, acceptance 2%); `max_num_seqs=1` on the pipeline and both
settings on the flat scheme were exact. Per-stage dumps (`_dump` on every
stage's first rank; `/mnt/nvme/dump_pp3_ms1` vs `dump_pp3_ms4`, forwards
aligned by their input ids) put the prefill and the first two decode steps of
each request bit-exact and the third decode step's *first-stage embedding* at
`[ids[4], ids[5], 0, 0, 0, 0]`: a four-row shift, four being the drafts the step
before had accepted. Root cause in the plugin's `_update_states` on the stages
before the last: vLLM's scheduler ships a cached request only the bonus token
it schedules now, so the accepted drafts never reach `req_state.output_token_ids`
there; a request re-added to the persistent batch after another request's batch
ran (every step, with several requests on the pipeline) is rebuilt from that
short list, its spec tokens land four rows early and the input slice at
`num_computed_tokens` reads past them. One request alone never leaves the batch
(the `token_ids_cpu` path), which is why `max_num_seqs=1` was exact. Fix: the
runner keeps the drafts it scheduled for a request (`pp_prev_spec_token_ids`)
and restores the accepted prefix before appending the bonus token. The flat
scheme is unaffected (its only rank is the last one). With the fix the three
prompts at `max_num_seqs=4` complete **byte-identical** to the one-sequence
pipeline run (`verify_short.py`, 07:21, `/mnt/nvme/ds41_tp8pp8_short_ms4d.log`;
52% of the real draft slots accepted, as on the flat scheme), with the three
batches in flight at once: the last stage finishes one every 45-50 ms and each
request advances every ~200 ms by 3-6 tokens. The trace also shows where a
pipeline step's time goes: stages 0-6 spend ~5-6 ms a batch; the last stage
~50 ms (host prep 4, kernel launches 5, recv + its 5 blocks + norm/head/sampler
~21, the drafter and its read-back 13-18, with outliers to ~120 ms), so the
sampling stage is the pipeline's bottleneck and a lone request's ~95 ms step is
mostly that stage. Measurements with the hand-off and this fix follow in the
bench section.

### The batched drafter (2026-10-24)

With ten requests on the pipeline (`max_num_seqs=10`, `in512_bs10`) the last
stage spent ~150 ms of its ~180 ms step drafting: `run_draft` launched the
drafter kernel once per sequence (~15 ms each, its cost being the kernel's
collectives, not its arithmetic), so the step gave 49 tok/s in aggregate at a
179 ms TPOT (vLLM acceptance 38%; `test/performance/history/reports/r3_05_tp8pp8_ms10_0813_seqdraft.json`).
`kernels/draft_kernel.py` now drafts every sequence of a step in one launch:
the B sequences' draft blocks are the B x 5 rows of one tile, the host builds
their per-row columns on the device (`mk_model._draft_rows`: token id,
position, window validity base, slot-table base, markov row), each sequence
has its own 128-slot window and draft block in the kernel's slot table at
`b * 256`, and the token loop drafts token i of all sequences at once (one
markov all-reduce and one argmax all-gather per token instead of one per token
per sequence). `kernel_version` is now scoped per kernel kind (block / draft /
pp), so a drafter edit no longer invalidates the block kernels' NEFFs.

Unit tests (8 ranks, random weights; final run 16:49-17:24 on the per-scheme kernel packages). The flat scheme's per-sequence drafter (`kernels/tp64/draft_kernel.py`, `test_draft_seq.py`) passes on 8 ranks at decode (start 200) and at the prefill step (start 0), with the same ids as before the pipeline work. The pipeline scheme's batched drafter (`kernels/tp8pp8/draft_kernel.py`, `test_draft.py`):

| Case | Result |
|---|---|
| `--B 1` (decode, start 200) | PASS on 8 ranks: ids identical to the per-sequence kernel's on the same inputs (`[1234, 14338, 12076, 1641, 11658, 6225]`, the two mismatches against the CPU reference sit at top-2 gaps of 0.26 / 0.59, below the 3.0 criterion), main-stream KV rows written within 1.2e-2 of the reference, untouched rows exact |
| `--B 3` (starts 200 / 237 / 274, taps 5 / 4 / 3) | PASS: **batch-invariant bit for bit** -- every sequence's ids, confidence (rel 0) and KV rows equal its own single-sequence launch |
| `--B 1 --start 0` (prefill phase) | PASS (the test now calls the phase-specialized entry `draft_prefill_tp64`; the generic entry's default phase had made this case untestable) |
| `--B 4 --start 300` | PASS: ids and KV rows batch-invariant bit for bit on 8 ranks; the confidence differs from the single-sequence launch at the 1e-3..4e-2 level when sequences share an expert tile (the random-weight MoE amplifies the accumulation order; the proposer does not consume the confidence, so that comparison is informational) |
| `--B 10` (starts 200..245) | PASS on 8 ranks: KV rows batch-invariant bit for bit; drafted ids identical to each sequence's lone launch for 9 of 10 sequences; sequence 7 differs from its fourth token on, where the reference's top-2 gap is 0.142 (a near-tie; the batched launch picks the reference's token there), which the gate accepts with the reference check's criterion (gap <= 3.0) |

The confidence comparison against the CPU reference is informational now: the
per-sequence kernel gives the same 0.9 relative difference on this test (the
confidence inherits the stream's amplified rounding on random weights, see
round 2), so it never was a measure of the kernel.

On the real model (`verify_short.py`, three prompts at `max_num_seqs=4`, 10:56):
the pipeline scheme's three completions are byte-identical to the one-sequence
reference again, and the run accepts exactly the same 42 of 320 draft slots as
the per-sequence drafter did -- the drafts themselves are unchanged, only the
number of launches is (one per step instead of one per sequence). The flat
scheme at `max_num_seqs=4` (three prompts in one batch, 11:06) is byte-identical
to its earlier run as well.

**What it bought, and what it did not (traced `in512_bs10` at `max_num_seqs=10`,
14:13, `test/performance/history/reports/r3_08_tp8pp8_ms10_1353_traced.json`; the trace came from the server log of that run with
`DEEPSEEK_V41_MK_TRACE=1 VLLM_NEURON_PP_TRACE=1`).** The last stage now spends 66-69 ms a batch
whether the batch holds one sequence or ten (host prep 6-8, launches 5-6, recv +
blocks + tail 35-43, draft + read-back 14-19 ms; before, ten sequences drafted
one after another at ~15 ms each). But of 1101 decode steps the stage ran, 965
carried **one** sequence and 136 carried ten: vLLM re-dispatches each request on
its own the moment its previous step's output is in, so with ten requests in
flight the pipeline runs mostly single-sequence batches, and every such batch
is padded to the engine's 60-row decode bucket and costs the same as a full
one. That, not the drafter, is why the batched cells stay at 46-50 tok/s in
aggregate (TPOT 180-195 ms a stream, acceptance 37-38%): ~15 batches/s through
the last stage times ~2 sequences a batch on average. Smaller decode buckets
for small batches, or holding ready requests back briefly so they merge, are the
next levers; neither is attempted here.

### Bench (`test/performance/run_bench.py`, both schemes, 2026-10-24)

Final run 17:51-22:11, with each scheme on its own kernel package
(`kernels/tp64/`, `kernels/tp8pp8/`): the flat scheme's block kernels and its
per-sequence drafter are the round-2 kernels unchanged, and the pipeline
scheme has its own copies (with the batched drafter and the stage boundary).
Each cell runs on its own engine (`max_model_len` = input + 256,
`max_num_seqs=1`): 8 prompts at concurrency 1 for the `bs1` cells, 10 prompts
at concurrency 10 for the `bs10` cells (queued on the one-sequence engine, so
their TTFT is queueing), 128 output tokens, medians, vLLM's spec-decode
acceptance over the cell.

| Cell | TTFT p50 ms (tp64 / tp8pp8) | TPOT p50 ms | decode tok/s | accept % |
|------|-----------------------------|-------------|--------------|----------|
| in512_bs1 | 562 / 918 | 31.9 / 49.4 | 28.0 / 17.4 | 46 / 46 |
| in512_bs10 | 20925 / 33667 | 30.5 / 49.5 | 28.6 / 17.4 | 47 / 46 |
| in1024_bs1 | 1061 / 1465 | 31.4 / 47.9 | 25.7 / 17.0 | 47 / 50 |
| in1024_bs10 | 23058 / 34394 | 31.0 / 47.4 | 25.7 / 17.3 | 48 / 50 |
| in2048_bs1 | 1957 / 2452 | 30.0 / 49.5 | 22.2 / 14.6 | 49 / 41 |
| in2048_bs10 | 27824 / 42867 | 30.0 / 49.6 | 22.2 / 14.7 | 49 / 42 |
| in8192_bs1 | 8549 / 4160 | 33.6 / 51.6 | 10.0 / 11.9 | 42 / 47 |
| in8192_bs10 | 65726 / 53129 | 33.6 / 51.8 | 10.0 / 11.9 | 42 / 46 |

**The flat scheme matches round 2 on every cell** (`bench_report.mk2.json`):
TPOT within 0.3 ms (31.9 / 30.5 / 31.4 / 31.0 / 30.0 / 30.0 / 33.6 / 33.6 ms
against 31.6 / 30.5 / 31.1 / 31.0 / 29.7 / 29.7 / 33.8 / 33.5), decode
throughput within 1%, TTFT within 2%, acceptance within 0.4 points. (An
interim run at 13:00-15:25, while the two schemes still shared the drafter and
a narrower launch window, was ~2 ms a token slower on the flat scheme; that is
what the per-scheme packages removed.) Every run of the day, interim ones included, is indexed with its
report and console log in `test/performance/history/README.md`.

The pipeline's single-stream TPOT is 47-52 ms: a lone request crosses seven
stage boundaries and the sampling stage serially and uses one stage at a
time. Its TTFT is higher at short inputs (the hand-offs again) and lower at
8192 (4.2 s against 8.5 s: the 2048-token prefill chunks pipeline across the
stages).

Ten concurrent requests on engines built for them (`DEEPSEEK_V41_MAX_NUM_SEQS=10`,
`--batch-sizes 10`; `bench_report.<scheme>_ms10.json`). The flat scheme's cells are
the 2026-10-25 rerun after the two fixes below, greedy like every other cell; the
pipeline's cells are still the 2026-10-24 run at temperature 1.0 and are not
comparable (not re-measured):

| Cell (max_num_seqs=10) | Scheme | Temperature | TTFT p50 ms | TPOT p50 ms | decode tok/s | accept % |
|------------------------|--------|-------------|-------------|-------------|--------------|----------|
| in512_bs10 | tp64 | 0 (greedy) | 3071 | 81.0 | 86.6 | 48.0 |
| in2048_bs10 | tp64 | 0 (greedy) | 10784 | 135.4 | 43.9 | 45.0 |
| in512_bs10 | tp8pp8 | 1.0 (2026-10-24) | 1633 | 162.0 | 55.0 | 37.8 |
| in2048_bs10 | tp8pp8 | 1.0 (2026-10-24) | 6013 | 197.7 | 40.1 | 37.2 |

**Batched acceptance on the flat scheme: root cause and fix (2026-10-25).** The batched
cells first measured 23% / 35% acceptance (against 46-49% one at a time) with
letter-split text ("l i st", "S h a k e s p e a r e"). Two bugs, neither specific to
batching:

1. `run_bench.py` sent temperature 0 to cells on a one-sequence engine but **no
   temperature** to cells on a multi-sequence engine -- a workaround for a 2026-10-21
   hang on the old per-op path -- so the server sampled the batched cells at its
   default 1.0 (the checkpoint has no `generation_config.json`) while the
   one-sequence cells were greedy. Every cell now sends the same temperature
   (greedy by default).
2. Temperature sampling drew only from rank 0's vocabulary shard: the plugin's
   vocab-parallel sampler never combined the shards, so every sampled token had an
   id below 2048 (all 1280 tokens checked, at `max_num_seqs` 1 and 10 alike); the
   model had corrected this for greedy rows only. The model now samples
   temperature rows itself from the gathered full-vocabulary logits
   (`_random_sampler_entry`: Gumbel-max with per-row seeds drawn on the host, so
   every rank draws the same token; top-k / top-p over a 256-candidate window).
   The runner passes the step's sampling kind (`sampling_hint`), so all-greedy
   steps keep their original graph and cost. Checked on the host: at temperature
   1.0 the draws match multinomial sampling (total variation 0.106 against 0.106
   for true multinomial draws, 20000 samples over 4096 tokens), top-k / top-p
   draws stay inside their set with the right proportions, and greedy rows are
   unchanged. Not re-measured on the device at temperature 1.0.

Before settling on these, a per-step trace ruled out the batched path itself: every
draft verified was the one proposed for that request (0 mismatches in 1773 verify
steps) and every step's input token was the token committed for that request the
step before. After the fixes the flat scheme's batched cells accept 48.0% / 45.0%
of drafts (one at a time: 46.2-47.0% / 48.8-48.9%), their 128-token outputs read
normally, and aggregate decode went up rather than down, from 59.2 to 86.6 tok/s at
512 input tokens and from 35.9 to 43.9 at 2048. The one-sequence in512 cell is
unchanged after the fix (TPOT 31.6 ms, 28.1 tok/s, 46.4%). Every run is indexed in
`test/performance/history/README.md`.
