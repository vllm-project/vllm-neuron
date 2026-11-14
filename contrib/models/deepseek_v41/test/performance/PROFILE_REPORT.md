# DeepSeek-V4.1-Flash on Trainium2: prefill / decode profile of the megakernel path at TP=64

*2026-10-25. Branch `deepseek_v41_contri` @ `3442695`, with every block and the DSpark drafter running as NKI megakernels (`DEEPSEEK_V41_MEGAKERNEL=1`, the default). Neuron SDK 2.33 venv `~/venvs/ds41-sdk233`: vLLM 0.29.0, vllm-neuron 0.29 plus the plugin-core worktree, neuronx-cc 2.28.7631, NKI 0.7.0, runtime 2.35.30. Instance trn2.48xlarge, TP=64, LNC=2, DSpark with K=5 (decode is 6 rows per sequence per step).*

## Summary

**Device time per window, rank 0, live trace.**

| Cell | Phase | Device time | Same-session bench (unprofiled) | 40 layer kernels | Drafter | Device idle (host) | XLA glue graphs | Collectives inside kernels | Where the time goes |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| in1024_bs1 | prefill | **996.8 ms** (1 sequence, 1 step) | TTFT p50 1014 ms | 987.0 ms (99.0%) | 3.3 ms | 2.8 ms | 3.7 ms | 92 ms (9.2%) | per-token attention body: 55% of engine time, 89% of its TE work is transposes |
| in1024_bs1 | decode | **117.1 ms / step** (≈3.2 tokens) | TPOT p50 36.6 ms | 94.5 ms (80.6%) | 7.1 ms (6.1%) | 11.8 ms (10.1%) | 3.8 ms | 31.2 ms (26.6%) | weight streaming (MBU 13%, software-DGE 1.6 KB packets); padded all-reduces; host step gap |
| in1024_bs10 | prefill | **1.012 s per sequence**, 10 back to back (10.25 s burst) | TTFT p50 5.66 s / p99 10.16 s | 1001 ms per sequence | 3.3 ms | 14.8 ms | 4.9 ms | — | same kernels as bs1; TTFT is FIFO queueing |
| in1024_bs10 | decode | **239.3 ms / step** | TPOT p50 111 ms | 147.2 ms (61.5%) | 72.1 ms (30.1%) | 14.0 ms (5.9%) | 6.0 ms | 55.1 ms (23.0%) | 10 per-sequence drafter kernels |
| in8192_bs1 | prefill | **8.82 s** = 4 chunks × 2.20 s | TTFT p50 8.85 s | 2175 ms per chunk (98.7%) | 10.4 ms per chunk | 12.7 ms per chunk | 6.0 ms per chunk | — | lightning indexer in the 8 index-source layers: 37% of each chunk |
| in8192_bs1 | decode | **119.8 ms / step** | TPOT p50 39.5 ms | 97.4 ms (81.2%) | 7.2 ms (6.0%) | 11.6 ms (9.7%) | 3.6 ms | 31.8 ms (26.5%) | as 1k decode; 8k context adds 2.3% |
| in8192_bs4 | prefill | **2.21 s per chunk**, 16 back to back (35.4 s burst) | TTFT p50 13.4 s / p99 34.8 s | 2175 ms per chunk | 10.7 ms per chunk | 15.8 ms per chunk | 6.1 ms | — | as 8k bs1; TTFT is FIFO queueing |
| in8192_bs4 | decode | **176.7 ms / step** | TPOT p50 170 ms (includes stalls behind other prefills) | 130.0 ms (73.6%) | 29.2 ms (16.5%) | 13.2 ms (7.5%) | 4.4 ms | 41.5 ms (23.5%) | layer kernels plus 4 per-sequence drafters |

**Roofline by kernel kind.** Execution-weighted means over the device-profiled windows. TE peak is 157.3 TFLOP/s and HBM 716 GB/s per logical core; the ridge is 220 FLOP/B.

| Phase | Kernel kinds | MFU | MBU | AI (FLOP/B) | DMA active | Busiest engine | Collective share | Verdict |
|---|---|---:|---:|---:|---:|---|---:|---|
| 1k prefill | 9 layer archetypes | 3.6–14.3% | 7.7–13.0% | 102–242 | 32–43% | Vector 43–53% | 6–17% | latency/dependency-bound; transposes are 7–38% of TE FLOPs (35% in the consumer layers) |
| 1k prefill | drafter (prefill) | 34.6% | 24.9% | 306 | 57% | DMA 57% | 14% | latency/dependency-bound |
| decode (all 4 cells) | 9 layer archetypes | 2.4–11.6% | 6.5–21.4% | 68–120 | 22–56% | DMA or Vector, 28–56% | 12–74% | latency/dependency-bound; layer 0 (`window`) is collective-wait-bound (step-start rank skew) |
| decode (all 4 cells) | drafter (decode) | 4.7–6.3% | 10.5–14.8% | 94–97 | 43–44% | DMA 43–44% | 23–24% | latency/dependency-bound |

**What would move the numbers most** (ranked by measured time; §6 has the detail):
1. Decode weight streaming: 7.6 GB read per step at 13% MBU, through software-DGE descriptors averaging 1.6 KB.
2. The per-sequence drafter: 30% of the bs10 decode step.
3. The decode all-reduces of the padded 128-row fp32 tile: 2 × 2.62 MB per layer even with 6 live rows, 23 ms per bs1 step.
4. The per-token prefill attention body: 55% of prefill engine time, 89% of its TE work transposes.
5. At 8k, the lightning indexer: 37% of every prefill chunk.
6. The host step boundary: ~14–17 ms per decode step.

---

## 1. How this was measured

**Capture.** I used the serve-based method (`neuron-framework-profiling-vllm-neuron`, Method 1): a live `vllm serve`, `/start_profile` and `/stop_profile`, and the NRT profiler on **rank 0 only**. Each cell gets its own engine, built exactly as `test/performance/run_bench.py` builds it:
- context = input + 256 (1280 or 8448)
- `--max-num-seqs` = batch
- prefix caching off
- sonnet prompts, greedy, `ignore_eos`, 128 output tokens

Each engine is warmed with the bench's own warm-up pass, then runs one unprofiled measured pass (the "same-session bench" numbers below), and only then is captured. Driver and scripts are in `~/profile_mk/`.

| Window | How it is isolated | Profiler activities |
|---|---|---|
| Prefill, bs1 engine | Idle server → `/start_profile` → one request with output 1 → `/stop_profile` | device + system |
| Prefill, bs>1 engine | Idle server → `/start_profile` → the bench's burst of *batch* requests at once (output 1), with prompts built before the window opens → `/stop_profile` | system only (see §7) |
| Decode | `/start_profile` once `/metrics` shows every sequence prefilled and at least one decode step after the last prefill. The profiler stops itself after 3 iterations (1k cells) or 2 (8k cells) | device + system |

**Analysis.** Parquet comes from `neuron-explorer view --ingest-only`, analyzed with the `neuron-framework-profile-analysis-vllm-neuron` method:
- **Latency source.** Every number is a *live* device execution window: `ExecutionInfo` / `nc_exec_running` on rank 0. These cross-check to within 0.1 ms per step (117.03/117.16 ms vs 117.08/117.20 ms on in1024_bs1 decode).
- **Labelling.** Every block is its own NEFF. A NEFF is a layer kernel if its instructions cite `layer_kernel.py`, and the drafter if they cite `draft_kernel.py`. The layer id is the execution's position among the 40 layer executions of the step, and is checked against the archetype every layer must have (zero conflicts in every window).

**Metric definitions** (per execution, then averaged per block):

| Metric | Definition |
|---|---|
| Latency | Device execution window of the block's NEFF (includes in-kernel collective waits). |
| Step wall | End of the previous step's last graph → end of this step's last graph, so the host gap before a step's first graph belongs to that step. Device idle = wall − Σ executions. |
| MFU | Non-transpose Tensor-Engine FLOPs (bf16-equivalent; fp32 matmuls count 4×) / (157.3 TFLOP/s × latency). |
| HFU | All TE FLOPs, PE transposes included. The "transpose share" columns show the difference. |
| MBU | HBM read + write bytes moved by DMA / (716 GB/s × latency). |
| AI | Useful FLOPs / HBM bytes. The roofline ridge is 157.3 TFLOP/s / 716 GB/s = **220 FLOP/B**. |
| DMA active % | Fraction of the latency with at least one DMA queue moving data (interval union of all device DMA packets). |
| DMA GB/s | All device DMA bytes (HBM↔SBUF, SBUF↔SBUF, collectives) / latency. |
| TE / Vec / Scal / GpSimd % | Engine active-wall, interval-merged over both physical cores, from `ActiveTime`. Semaphore waits are excluded; I checked this against an instruction-level recomputation that drops `EVENT_SEMAPHORE`/`DRAIN`/`COMPARE_BRANCH`, which agrees to ±2 points. |
| CC % | Wall of the collective ops (`CcOp`) inside the execution, including time waiting for the other 63 ranks. |
| Verdict | *X-bound* if one engine (or DMA) is active ≥ 70% of the latency; *collective wait* if CC % exceeds every engine; otherwise *latency/dependency-bound*, with the busiest engine named. |

Hardware constants come from the profile's `Metadata`: TE 2 × 128×128 PE at 2.4 GHz = **157.3 TFLOP/s** bf16 per logical core, and HBM **716 GB/s** per logical core (one stack shared by the two physical cores).

---

## 2. in1024_bs1 (context 1280, max_num_seqs 1)

Same-session bench, unprofiled: TTFT p50 1014 ms / p99 1019 ms, TPOT p50 36.6 ms / p99 40.8 ms, decode 22.72 tok/s (8 requests, mean input 1009 tokens).

### 2.1 Prefill: one 1014-token request (device-profiled, complete)

**Overall latency breakdown** (device time of the prefill step, descending)

Step wall (mean of 1 step): **996.81 ms** (steps: 996.81 ms)

| # | Component | ms / step | % of step |
|---:|---|---:|---:|
| 1 | layer kernels: c2 (x15) | 350.65 | 35.2 |
| 2 | layer kernels: c1 (x12) | 268.61 | 26.9 |
| 3 | layer kernels: c1_index (x4) | 127.50 | 12.8 |
| 4 | layer kernels: c1_tap (x3) | 67.14 | 6.7 |
| 5 | layer kernels: c2_source (x2) | 64.42 | 6.5 |
| 6 | layer kernels: c2_source_engram (x1) | 37.47 | 3.8 |
| 7 | layer kernels: c1_source (x1) | 32.34 | 3.2 |
| 8 | layer kernels: window_engram (x1) | 22.06 | 2.2 |
| 9 | layer kernels: window (x1) | 16.79 | 1.7 |
| 10 | drafter kernel (DSpark, 3 stages) | 3.30 | 0.3 |
| 11 | device idle between executions (host dispatch / sync) | 2.82 | 0.3 |
| 12 | embed + n-gram hash + hc_expand (XLA graph) | 2.29 | 0.2 |
| 13 | final norm + LM head + sampler (XLA graph) | 1.21 | 0.1 |
| 14 | sampler / draft-meta glue (XLA graphs) | 0.13 | 0.0 |
| 15 | engram hash-id slice, layer 14 (XLA graph) | 0.04 | 0.0 |
| 16 | engram hash-id slice, layer 1 (XLA graph) | 0.03 | 0.0 |

Collective (all-reduce) wall inside executions: 92.01 ms / step (9.2% of step); inside the 40 layer kernels 89.35 ms (9.1% of layer-kernel time).

**Per-block latency** (descending)

| # | Block | Kernel / archetype | Latency mean (ms) | min–max (ms) | % of step |
|---:|---|---|---:|---:|---:|
| 1 | L14 | c2_source_engram | 37.471 | 37.471–37.471 | 3.8 |
| 2 | L02 | c2_source | 32.446 | 32.446–32.446 | 3.3 |
| 3 | L20 | c1_source | 32.344 | 32.344–32.344 | 3.2 |
| 4 | L36 | c1_index | 32.017 | 32.017–32.017 | 3.2 |
| 5 | L32 | c1_index | 31.984 | 31.984–31.984 | 3.2 |
| 6 | L08 | c2_source | 31.979 | 31.979–31.979 | 3.2 |
| 7 | L24 | c1_index | 31.763 | 31.763–31.763 | 3.2 |
| 8 | L28 | c1_index | 31.733 | 31.733–31.733 | 3.2 |
| 9 | L18 | c2 | 23.711 | 23.711–23.711 | 2.4 |
| 10 | L05 | c2 | 23.664 | 23.664–23.664 | 2.4 |
| 11 | L17 | c2 | 23.661 | 23.661–23.661 | 2.4 |
| 12 | L04 | c2 | 23.654 | 23.654–23.654 | 2.4 |
| 13 | L11 | c2 | 23.641 | 23.641–23.641 | 2.4 |
| 14 | L10 | c2 | 23.640 | 23.640–23.640 | 2.4 |
| 15 | L12 | c2 | 23.528 | 23.528–23.528 | 2.4 |
| 16 | L13 | c2 | 23.402 | 23.402–23.402 | 2.3 |
| 17 | L15 | c2 | 23.387 | 23.387–23.387 | 2.3 |
| 18 | L03 | c2 | 23.250 | 23.250–23.250 | 2.3 |
| 19 | L07 | c2 | 23.225 | 23.225–23.225 | 2.3 |
| 20 | L19 | c2 | 23.175 | 23.175–23.175 | 2.3 |
| 21 | L06 | c2 | 23.159 | 23.159–23.159 | 2.3 |
| 22 | L09 | c2 | 22.864 | 22.864–22.864 | 2.3 |
| 23 | L25 | c1 | 22.792 | 22.792–22.792 | 2.3 |
| 24 | L29 | c1 | 22.752 | 22.752–22.752 | 2.3 |
| 25 | L23 | c1 | 22.731 | 22.731–22.731 | 2.3 |
| 26 | L16 | c2 | 22.687 | 22.687–22.687 | 2.3 |
| 27 | L37 | c1_tap | 22.666 | 22.666–22.666 | 2.3 |
| 28 | L39 | c1_tap | 22.606 | 22.606–22.606 | 2.3 |
| 29 | L27 | c1 | 22.479 | 22.479–22.479 | 2.3 |
| 30 | L26 | c1 | 22.472 | 22.472–22.472 | 2.3 |
| 31 | L22 | c1 | 22.455 | 22.455–22.455 | 2.3 |
| 32 | L21 | c1 | 22.309 | 22.309–22.309 | 2.2 |
| 33 | L31 | c1 | 22.252 | 22.252–22.252 | 2.2 |
| 34 | L34 | c1 | 22.224 | 22.224–22.224 | 2.2 |
| 35 | L33 | c1 | 22.113 | 22.113–22.113 | 2.2 |
| 36 | L01 | window_engram | 22.061 | 22.061–22.061 | 2.2 |
| 37 | L35 | c1 | 22.024 | 22.024–22.024 | 2.2 |
| 38 | L30 | c1 | 22.003 | 22.003–22.003 | 2.2 |
| 39 | L38 | c1_tap | 21.871 | 21.871–21.871 | 2.2 |
| 40 | L00 | window | 16.792 | 16.792–16.792 | 1.7 |
| 41 | drafter | draft | 3.301 | 3.301–3.301 | 0.3 |
| 42 | embed + n-gram hash + hc_expand (XLA graph) | glue | 2.294 | 2.294–2.294 | 0.2 |
| 43 | final norm + LM head + sampler (XLA graph) | glue | 1.211 | 1.211–1.211 | 0.1 |
| 44 | sampler / draft-meta glue (XLA graphs) | glue | 0.042 | 0.027–0.067 | 0.0 |
| 45 | engram hash-id slice, layer 14 (XLA graph) | glue | 0.039 | 0.039–0.039 | 0.0 |
| 46 | engram hash-id slice, layer 1 (XLA graph) | glue | 0.029 | 0.029–0.029 | 0.0 |

**Roofline by kernel kind**

| Kernel kind | Executions | Mean latency (ms) | MFU % | MBU % | AI (FLOP/B) | DMA active % | TE % | Vec % | Scal % | GpSimd % | CC % | Transpose share of TE FLOPs | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c2 | 15 | 23.377 | 5.25 | 10.6 | 109 | 43 | 39 | 52 | 18 | 38 | 9 | 36% | latency/dependency-bound (busiest Vector 52%) |
| c1 | 12 | 22.384 | 5.60 | 11.2 | 110 | 39 | 39 | 52 | 19 | 40 | 9 | 35% | latency/dependency-bound (busiest Vector 52%) |
| c1_index | 4 | 31.874 | 4.17 | 8.2 | 112 | 37 | 42 | 50 | 15 | 39 | 6 | 35% | latency/dependency-bound (busiest Vector 50%) |
| c1_tap | 3 | 22.381 | 5.64 | 11.6 | 107 | 40 | 39 | 53 | 19 | 41 | 9 | 35% | latency/dependency-bound (busiest Vector 53%) |
| c2_source | 2 | 32.212 | 4.60 | 8.1 | 124 | 39 | 38 | 46 | 14 | 38 | 8 | 32% | latency/dependency-bound (busiest Vector 46%) |
| c2_source_engram | 1 | 37.471 | 9.19 | 10.0 | 202 | 41 | 40 | 44 | 14 | 38 | 11 | 17% | latency/dependency-bound (busiest Vector 44%) |
| c1_source | 1 | 32.344 | 3.59 | 7.7 | 102 | 35 | 40 | 48 | 13 | 37 | 9 | 38% | latency/dependency-bound (busiest Vector 48%) |
| window_engram | 1 | 22.061 | 14.32 | 13.0 | 242 | 38 | 34 | 43 | 22 | 36 | 13 | 7% | latency/dependency-bound (busiest Vector 43%) |
| window | 1 | 16.792 | 6.96 | 10.2 | 150 | 32 | 30 | 47 | 25 | 35 | 17 | 18% | latency/dependency-bound (busiest Vector 47%) |
| glue | 7 | 0.529 | 0.92 | 19.4 | 10 | 33 | 3 | 24 | 10 | 15 | 59 | 0% | collective wait (59%) |
| draft | 1 | 3.301 | 34.64 | 24.9 | 306 | 57 | 49 | 42 | 14 | 45 | 14 | 4% | latency/dependency-bound (busiest DMA 57%) |

**Roofline per block** (descending by latency)

| Block | Kernel / archetype | Latency (ms) | TFLOP/s | MFU % | HBM GB/s | MBU % | AI (FLOP/B) | DMA active % | DMA GB/s | TE % | Vec % | Scal % | GpSimd % | CC % | Bottleneck |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| L14 | c2_source_engram | 37.471 | 14.46 | 9.19 | 72 | 10.0 | 202 | 41 | 77 | 40 | 44 | 14 | 38 | 11 | latency/dependency-bound (busiest Vector 44%) |
| L02 | c2_source | 32.446 | 6.77 | 4.30 | 56 | 7.9 | 120 | 38 | 59 | 37 | 45 | 14 | 37 | 10 | latency/dependency-bound (busiest Vector 45%) |
| L20 | c1_source | 32.344 | 5.65 | 3.59 | 55 | 7.7 | 102 | 35 | 58 | 40 | 48 | 13 | 37 | 9 | latency/dependency-bound (busiest Vector 48%) |
| L36 | c1_index | 32.017 | 6.38 | 4.06 | 58 | 8.1 | 110 | 37 | 61 | 41 | 50 | 15 | 39 | 7 | latency/dependency-bound (busiest Vector 50%) |
| L32 | c1_index | 31.984 | 6.11 | 3.88 | 57 | 7.9 | 107 | 36 | 60 | 41 | 49 | 14 | 38 | 7 | latency/dependency-bound (busiest Vector 49%) |
| L08 | c2_source | 31.979 | 7.72 | 4.91 | 60 | 8.4 | 128 | 40 | 63 | 39 | 48 | 15 | 39 | 6 | latency/dependency-bound (busiest Vector 48%) |
| L24 | c1_index | 31.763 | 6.15 | 3.91 | 57 | 8.0 | 107 | 37 | 60 | 41 | 50 | 15 | 39 | 7 | latency/dependency-bound (busiest Vector 50%) |
| L28 | c1_index | 31.733 | 7.58 | 4.82 | 62 | 8.7 | 122 | 39 | 65 | 43 | 53 | 17 | 41 | 3 | latency/dependency-bound (busiest Vector 53%) |
| L18 | c2 | 23.711 | 7.74 | 4.92 | 74 | 10.3 | 105 | 41 | 77 | 38 | 50 | 17 | 37 | 12 | latency/dependency-bound (busiest Vector 50%) |
| L05 | c2 | 23.664 | 8.13 | 5.17 | 75 | 10.5 | 108 | 42 | 78 | 39 | 51 | 18 | 38 | 10 | latency/dependency-bound (busiest Vector 51%) |
| L17 | c2 | 23.661 | 8.13 | 5.17 | 75 | 10.5 | 108 | 42 | 78 | 39 | 51 | 18 | 38 | 11 | latency/dependency-bound (busiest Vector 51%) |
| L04 | c2 | 23.654 | 8.52 | 5.42 | 76 | 10.7 | 112 | 43 | 79 | 39 | 52 | 18 | 39 | 10 | latency/dependency-bound (busiest Vector 52%) |
| L11 | c2 | 23.641 | 7.37 | 4.69 | 72 | 10.1 | 102 | 41 | 76 | 37 | 49 | 16 | 36 | 13 | latency/dependency-bound (busiest Vector 49%) |
| L10 | c2 | 23.640 | 7.76 | 4.93 | 74 | 10.3 | 105 | 42 | 77 | 38 | 50 | 17 | 37 | 12 | latency/dependency-bound (busiest Vector 50%) |
| L12 | c2 | 23.528 | 8.18 | 5.20 | 75 | 10.5 | 108 | 42 | 79 | 39 | 51 | 18 | 38 | 10 | latency/dependency-bound (busiest Vector 51%) |
| L13 | c2 | 23.402 | 8.61 | 5.48 | 77 | 10.8 | 112 | 43 | 80 | 40 | 52 | 19 | 39 | 8 | latency/dependency-bound (busiest Vector 52%) |
| L15 | c2 | 23.387 | 8.23 | 5.23 | 76 | 10.6 | 108 | 43 | 79 | 39 | 52 | 18 | 38 | 9 | latency/dependency-bound (busiest Vector 52%) |
| L03 | c2 | 23.250 | 8.28 | 5.26 | 76 | 10.7 | 108 | 43 | 80 | 39 | 52 | 18 | 38 | 9 | latency/dependency-bound (busiest Vector 52%) |
| L07 | c2 | 23.225 | 7.90 | 5.02 | 75 | 10.5 | 105 | 42 | 78 | 39 | 51 | 17 | 38 | 10 | latency/dependency-bound (busiest Vector 51%) |
| L19 | c2 | 23.175 | 8.70 | 5.53 | 78 | 10.9 | 112 | 44 | 81 | 40 | 53 | 19 | 39 | 8 | latency/dependency-bound (busiest Vector 53%) |
| L06 | c2 | 23.159 | 7.92 | 5.04 | 75 | 10.5 | 105 | 43 | 78 | 39 | 51 | 17 | 38 | 9 | latency/dependency-bound (busiest Vector 51%) |
| L09 | c2 | 22.864 | 10.00 | 6.36 | 83 | 11.6 | 120 | 46 | 86 | 42 | 56 | 21 | 42 | 3 | latency/dependency-bound (busiest Vector 56%) |
| L25 | c1 | 22.792 | 8.44 | 5.37 | 78 | 10.9 | 108 | 38 | 81 | 38 | 51 | 18 | 39 | 11 | latency/dependency-bound (busiest Vector 51%) |
| L29 | c1 | 22.752 | 8.46 | 5.38 | 78 | 10.9 | 108 | 38 | 81 | 38 | 51 | 18 | 39 | 11 | latency/dependency-bound (busiest Vector 51%) |
| L23 | c1 | 22.731 | 9.26 | 5.89 | 81 | 11.3 | 114 | 39 | 84 | 39 | 53 | 19 | 41 | 9 | latency/dependency-bound (busiest Vector 53%) |
| L16 | c2 | 22.687 | 8.48 | 5.39 | 78 | 10.9 | 108 | 44 | 81 | 40 | 53 | 18 | 39 | 7 | latency/dependency-bound (busiest Vector 53%) |
| L37 | c1_tap | 22.666 | 8.49 | 5.40 | 81 | 11.3 | 105 | 39 | 86 | 38 | 52 | 18 | 40 | 10 | latency/dependency-bound (busiest Vector 52%) |
| L39 | c1_tap | 22.606 | 10.12 | 6.43 | 87 | 12.2 | 116 | 42 | 92 | 41 | 56 | 21 | 43 | 6 | latency/dependency-bound (busiest Vector 56%) |
| L27 | c1 | 22.479 | 7.76 | 4.93 | 76 | 10.6 | 102 | 37 | 79 | 38 | 50 | 17 | 38 | 13 | latency/dependency-bound (busiest Vector 50%) |
| L26 | c1 | 22.472 | 9.37 | 5.96 | 82 | 11.4 | 114 | 40 | 85 | 40 | 54 | 20 | 41 | 8 | latency/dependency-bound (busiest Vector 54%) |
| L22 | c1 | 22.455 | 10.99 | 6.99 | 88 | 12.2 | 125 | 42 | 91 | 42 | 57 | 23 | 44 | 4 | latency/dependency-bound (busiest Vector 57%) |
| L21 | c1 | 22.309 | 7.82 | 4.97 | 77 | 10.7 | 102 | 38 | 80 | 38 | 50 | 17 | 39 | 12 | latency/dependency-bound (busiest Vector 50%) |
| L31 | c1 | 22.252 | 9.06 | 5.76 | 81 | 11.4 | 111 | 40 | 85 | 40 | 53 | 19 | 41 | 8 | latency/dependency-bound (busiest Vector 53%) |
| L34 | c1 | 22.224 | 7.85 | 4.99 | 77 | 10.8 | 102 | 37 | 80 | 38 | 50 | 16 | 39 | 11 | latency/dependency-bound (busiest Vector 50%) |
| L33 | c1 | 22.113 | 8.29 | 5.27 | 79 | 11.0 | 105 | 38 | 82 | 39 | 52 | 18 | 40 | 10 | latency/dependency-bound (busiest Vector 52%) |
| L01 | window_engram | 22.061 | 22.53 | 14.32 | 93 | 13.0 | 242 | 38 | 102 | 34 | 43 | 22 | 36 | 13 | latency/dependency-bound (busiest Vector 43%) |
| L35 | c1 | 22.024 | 9.56 | 6.08 | 84 | 11.7 | 114 | 40 | 87 | 40 | 54 | 20 | 42 | 6 | latency/dependency-bound (busiest Vector 54%) |
| L30 | c1 | 22.003 | 8.75 | 5.56 | 81 | 11.3 | 108 | 40 | 84 | 40 | 53 | 19 | 41 | 8 | latency/dependency-bound (busiest Vector 53%) |
| L38 | c1_tap | 21.871 | 7.97 | 5.07 | 81 | 11.3 | 98 | 39 | 86 | 39 | 52 | 17 | 40 | 9 | latency/dependency-bound (busiest Vector 52%) |
| L00 | window | 16.792 | 10.95 | 6.96 | 73 | 10.2 | 150 | 32 | 78 | 30 | 47 | 25 | 35 | 17 | latency/dependency-bound (busiest Vector 47%) |
| drafter | draft | 3.301 | 54.48 | 34.64 | 178 | 24.9 | 306 | 57 | 169 | 49 | 42 | 14 | 45 | 14 | latency/dependency-bound (busiest DMA 57%) |
| embed + n-gram hash + hc_expand (XLA graph) | glue | 2.294 | 0.00 | 0.00 | 82 | 11.5 | 0 | 17 | 37 | 0 | 5 | 2 | 8 | 90 | collective wait (90%) |
| final norm + LM head + sampler (XLA graph) | glue | 1.211 | 4.44 | 2.82 | 266 | 37.2 | 17 | 64 | 285 | 10 | 63 | 24 | 26 | 12 | latency/dependency-bound (busiest DMA 64%) |
| sampler / draft-meta glue (XLA graphs) | glue | 0.042 | 0.00 | 0.00 | 0 | 0.0 | 0 | 26 | 3 | 0 | 3 | 4 | 19 | 0 | latency/dependency-bound (busiest DMA 26%) |
| engram hash-id slice, layer 14 (XLA graph) | glue | 0.039 | 0.00 | 0.00 | 25 | 3.5 | 0 | 23 | 26 | 0 | 14 | 22 | 27 | 0 | latency/dependency-bound (busiest GpSimd 27%) |
| engram hash-id slice, layer 1 (XLA graph) | glue | 0.029 | 0.00 | 0.00 | 34 | 4.7 | 0 | 25 | 34 | 0 | 19 | 30 | 37 | 0 | latency/dependency-bound (busiest GpSimd 37%) |

**Where the engine time goes inside the 40 layer kernels**

| Kernel function | Tensor (ms) | Vector (ms) | Scalar (ms) | GpSimd (ms) | Sync (ms) | Total engine-ms | Useful GFLOP | Transpose GFLOP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mk_common.py:attn_token_body` | 932.2 | 379.4 | 69.1 | 392.3 | 0.0 | 1772.9 | 412.3 | 3304.4 |
| `mk_common.py:hc_in_tile` | 198.1 | 51.0 | 22.5 | 1.5 | 1.0 | 274.0 | 322.1 | 429.3 |
| `mk_gemm.py:gemm_staged` | 174.8 | 0.0 | 0.0 | 73.2 | 0.0 | 248.0 | 5514.1 | 0.0 |
| `moe_expert.py:_gemm_fp4` | 53.7 | 12.2 | 50.3 | 43.1 | 0.0 | 159.3 | 1531.1 | 0.0 |
| `mk_compress.py:body` | 97.8 | 37.7 | 0.0 | 19.4 | 0.0 | 154.8 | 9.0 | 0.0 |
| `mk_common.py:hc_out_tile` | 0.0 | 65.7 | 18.1 | 8.8 | 8.5 | 101.1 | 0.0 | 0.0 |
| `blockquant.py:_qdq_fp8` | 0.0 | 44.1 | 9.7 | 0.0 | 0.0 | 53.8 | 0.0 | 0.0 |
| `mk_gemm.py:stage_rows_T` | 29.8 | 16.1 | 0.0 | 0.0 | 1.7 | 47.6 | 0.0 | 313.7 |
| `mk_common.py:rowsum_div` | 0.0 | 18.3 | 27.3 | 0.0 | 0.0 | 45.6 | 0.0 | 0.0 |
| `moe_expert.py:_fp4_dequant` | 0.0 | 42.5 | 0.0 | 0.0 | 0.0 | 42.5 | 0.0 | 0.0 |

<sub>all layer executions of step 0 (40); wall 987.0 ms. Engine busy time per function, summed over both physical cores and all listed executions (semaphore waits, drains and loop branches excluded), so a row can exceed the wall.</sub>

**DMA traffic by queue** (one prefill step)

| Kernels | DMA queue type | Packets | MB / step | Share | Mean packet (B) |
|---|---|---:|---:|---:|---:|
| layer | software_dynamic | 28,467,104 | 50378 | 66% | 1770 |
| layer | hardware_dynamic | 17,357,162 | 22260 | 29% | 1282 |
| layer | collectives | 312,906 | 2227 | 3% | 7118 |
| layer | instruction | 191,119 | 1277 | 2% | 6681 |
| layer | data | 3,552 | 10 | 0% | 2889 |
| layer | input | 640 | 0 | 0% | 36 |
| draft | software_dynamic | 245,280 | 403 | 72% | 1642 |
| draft | hardware_dynamic | 58,430 | 151 | 27% | 2577 |
| draft | data | 472 | 2 | 0% | 4217 |
| draft | instruction | 939 | 2 | 0% | 1872 |
| draft | input | 528 | 1 | 0% | 1994 |
| draft | collectives | 1,128 | 1 | 0% | 823 |

**What the 1k prefill is made of.**
- **Device time is 996.8 ms; 99% of it is the 40 layer kernels.** The same-session bench TTFT is 1014–1016 ms; the remaining ~18 ms is host time before the first graph (request admission, tokenization, scheduling), which the device trace does not cover. Under the profiler, TTFT was 1017 ms, so profiling overhead is negligible.
- **Two layer costs.** Consumer layers (`c2`, `c1`, `c1_tap`) take 22–24 ms each. The eight index-source layers (`c2_source`, `c1_source`, `c1_index`) take 32 ms: the extra ~9 ms is the lightning indexer, i.e. `mk_compress.py:body` scoring every compressed group for every token, plus `index_topk_rows`.
- **No engine is saturated.** The busiest engine is Vector at ~50%; TE runs ~39% and GpSimd ~38%. Useful MFU is 4–6% (14% for `window_engram`), MBU 8–12%, and AI ~105–125 FLOP/B, below the 220 FLOP/B ridge. The kernels are bound by dependencies between engines, not by any engine's throughput.
- **The per-token attention body dominates.** `attn_token_body` uses 1773 engine-ms, 55% of all engine busy time in the 40 layer kernels, and its 932 TE-ms are **89% PE transposes**: 3304 GFLOP of transposes against 412 GFLOP of QK/PV. Each of the 1024 query tokens gathers its own 640 KV rows (128 window + 512 top-k) by indirect DMA and transposes them on the PE. That is also why it reads 685 MB of HBM in layer 3, 39% of that layer's traffic, even though neighbouring tokens share 127 of 128 window rows and most top-k picks.
- **The GEMMs are not the problem.** 5.5 TFLOP of GEMM work runs in 175 TE-ms, about 31 TFLOP/s while TE is active.
- **Collectives cost 84 ms.** Each layer does two fp32 all-reduces of 20 MB (attention output over the 8-rank group, MoE output over all 64 ranks), at 1.05 ms each: 84 ms per prefill, 8.4%.

### 2.2 Decode, bs1 (device-profiled, 2 full steps)

**Overall latency breakdown** (per decode step, descending)

Step wall (mean of 2 steps): **117.14 ms** (steps: 117.08, 117.20 ms)

| # | Component | ms / step | % of step |
|---:|---|---:|---:|
| 1 | layer kernels: c2 (x15) | 32.69 | 27.9 |
| 2 | layer kernels: c1 (x12) | 26.68 | 22.8 |
| 3 | device idle between executions (host dispatch / sync) | 11.82 | 10.1 |
| 4 | layer kernels: c1_index (x4) | 9.53 | 8.1 |
| 5 | drafter kernel (DSpark, 3 stages) | 7.12 | 6.1 |
| 6 | layer kernels: c1_tap (x3) | 6.68 | 5.7 |
| 7 | layer kernels: c2_source (x2) | 5.42 | 4.6 |
| 8 | layer kernels: c2_source_engram (x1) | 3.75 | 3.2 |
| 9 | layer kernels: window (x1) | 3.71 | 3.2 |
| 10 | layer kernels: window_engram (x1) | 3.53 | 3.0 |
| 11 | layer kernels: c1_source (x1) | 2.47 | 2.1 |
| 12 | embed + n-gram hash + hc_expand (XLA graph) | 2.43 | 2.1 |
| 13 | final norm + LM head + sampler (XLA graph) | 1.09 | 0.9 |
| 14 | sampler / draft-meta glue (XLA graphs) | 0.11 | 0.1 |
| 15 | draft-id cast of the previous step (XLA graph) | 0.04 | 0.0 |
| 16 | engram hash-id slice, layer 1 (XLA graph) | 0.04 | 0.0 |
| 17 | engram hash-id slice, layer 14 (XLA graph) | 0.04 | 0.0 |

Collective (all-reduce) wall inside executions: 31.18 ms / step (26.6% of step); inside the 40 layer kernels 27.00 ms (28.6% of layer-kernel time).

**Per-block latency** (descending)

| # | Block | Kernel / archetype | Latency mean (ms) | min–max (ms) | % of step |
|---:|---|---|---:|---:|---:|
| 1 | drafter | draft | 7.124 | 7.007–7.241 | 6.1 |
| 2 | L14 | c2_source_engram | 3.748 | 3.738–3.757 | 3.2 |
| 3 | L00 | window | 3.712 | 3.670–3.755 | 3.2 |
| 4 | L01 | window_engram | 3.535 | 3.418–3.651 | 3.0 |
| 5 | L02 | c2_source | 2.720 | 2.599–2.841 | 2.3 |
| 6 | L08 | c2_source | 2.702 | 2.585–2.818 | 2.3 |
| 7 | L24 | c1_index | 2.475 | 2.356–2.594 | 2.1 |
| 8 | L33 | c1 | 2.475 | 2.468–2.481 | 2.1 |
| 9 | L20 | c1_source | 2.465 | 2.465–2.465 | 2.1 |
| 10 | embed + n-gram hash + hc_expand (XLA graph) | glue | 2.428 | 2.299–2.558 | 2.1 |
| 11 | L35 | c1 | 2.396 | 2.393–2.399 | 2.0 |
| 12 | L32 | c1_index | 2.356 | 2.356–2.356 | 2.0 |
| 13 | L28 | c1_index | 2.355 | 2.355–2.355 | 2.0 |
| 14 | L36 | c1_index | 2.347 | 2.335–2.358 | 2.0 |
| 15 | L03 | c2 | 2.337 | 2.214–2.460 | 2.0 |
| 16 | L15 | c2 | 2.336 | 2.227–2.445 | 2.0 |
| 17 | L34 | c1 | 2.278 | 2.150–2.406 | 1.9 |
| 18 | L19 | c2 | 2.272 | 2.160–2.384 | 1.9 |
| 19 | L37 | c1_tap | 2.268 | 2.266–2.270 | 1.9 |
| 20 | L29 | c1 | 2.224 | 2.223–2.224 | 1.9 |
| 21 | L25 | c1 | 2.221 | 2.215–2.227 | 1.9 |
| 22 | L09 | c2 | 2.218 | 2.215–2.221 | 1.9 |
| 23 | L39 | c1_tap | 2.209 | 2.209–2.210 | 1.9 |
| 24 | L38 | c1_tap | 2.199 | 2.195–2.204 | 1.9 |
| 25 | L27 | c1 | 2.178 | 2.176–2.180 | 1.9 |
| 26 | L30 | c1 | 2.171 | 2.168–2.175 | 1.9 |
| 27 | L26 | c1 | 2.167 | 2.166–2.167 | 1.8 |
| 28 | L06 | c2 | 2.166 | 2.156–2.176 | 1.8 |
| 29 | L23 | c1 | 2.160 | 2.155–2.165 | 1.8 |
| 30 | L22 | c1 | 2.156 | 2.153–2.159 | 1.8 |
| 31 | L31 | c1 | 2.154 | 2.151–2.157 | 1.8 |
| 32 | L17 | c2 | 2.154 | 2.147–2.161 | 1.8 |
| 33 | L12 | c2 | 2.151 | 2.150–2.151 | 1.8 |
| 34 | L13 | c2 | 2.150 | 2.149–2.151 | 1.8 |
| 35 | L04 | c2 | 2.150 | 2.149–2.151 | 1.8 |
| 36 | L05 | c2 | 2.148 | 2.142–2.155 | 1.8 |
| 37 | L11 | c2 | 2.146 | 2.139–2.152 | 1.8 |
| 38 | L18 | c2 | 2.144 | 2.141–2.146 | 1.8 |
| 39 | L07 | c2 | 2.143 | 2.131–2.156 | 1.8 |
| 40 | L16 | c2 | 2.142 | 2.123–2.162 | 1.8 |
| 41 | L21 | c1 | 2.101 | 1.986–2.217 | 1.8 |
| 42 | L10 | c2 | 2.031 | 1.915–2.147 | 1.7 |
| 43 | final norm + LM head + sampler (XLA graph) | glue | 1.087 | 1.086–1.088 | 0.9 |
| 44 | sampler / draft-meta glue (XLA graphs) | glue | 0.036 | 0.028–0.054 | 0.1 |
| 45 | draft-id cast of the previous step (XLA graph) | glue | 0.043 | 0.041–0.044 | 0.0 |
| 46 | engram hash-id slice, layer 1 (XLA graph) | glue | 0.036 | 0.036–0.037 | 0.0 |
| 47 | engram hash-id slice, layer 14 (XLA graph) | glue | 0.035 | 0.035–0.036 | 0.0 |

**Roofline by kernel kind**

| Kernel kind | Executions | Mean latency (ms) | MFU % | MBU % | AI (FLOP/B) | DMA active % | TE % | Vec % | Scal % | GpSimd % | CC % | Transpose share of TE FLOPs | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c2 | 45 | 2.156 | 5.72 | 12.9 | 98 | 41 | 21 | 41 | 22 | 31 | 27 | 12% | latency/dependency-bound (busiest DMA 41%) |
| c1 | 36 | 2.215 | 6.22 | 13.1 | 104 | 42 | 22 | 42 | 23 | 33 | 25 | 11% | latency/dependency-bound (busiest DMA 42%) |
| c1_index | 12 | 2.413 | 5.69 | 12.3 | 102 | 41 | 21 | 41 | 22 | 32 | 26 | 12% | latency/dependency-bound (busiest Vector 41%) |
| draft | 3 | 7.083 | 6.34 | 14.8 | 94 | 44 | 21 | 40 | 18 | 36 | 23 | 11% | latency/dependency-bound (busiest DMA 44%) |
| c1_tap | 9 | 2.223 | 4.97 | 12.2 | 90 | 40 | 20 | 39 | 20 | 29 | 29 | 13% | latency/dependency-bound (busiest DMA 40%) |
| c2_source | 6 | 2.586 | 5.00 | 12.2 | 90 | 39 | 22 | 38 | 20 | 28 | 23 | 12% | latency/dependency-bound (busiest DMA 39%) |
| glue | 23 | 0.492 | 0.92 | 2.2 | 91 | 14 | 4 | 9 | 3 | 14 | 68 | 1% | collective wait (68%) |
| c2_source_engram | 3 | 3.745 | 10.03 | 20.4 | 108 | 51 | 32 | 34 | 16 | 45 | 20 | 5% | latency/dependency-bound (busiest DMA 51%) |
| window_engram | 3 | 3.560 | 11.55 | 21.1 | 120 | 53 | 33 | 36 | 19 | 47 | 20 | 4% | latency/dependency-bound (busiest DMA 53%) |
| window | 3 | 3.458 | 2.82 | 7.3 | 85 | 23 | 11 | 22 | 11 | 16 | 73 | 12% | collective wait (73%) |
| c1_source | 3 | 2.467 | 4.18 | 11.1 | 83 | 38 | 18 | 36 | 17 | 29 | 30 | 15% | latency/dependency-bound (busiest DMA 38%) |

**Roofline per block** (descending by latency)

| Block | Kernel / archetype | Latency (ms) | TFLOP/s | MFU % | HBM GB/s | MBU % | AI (FLOP/B) | DMA active % | DMA GB/s | TE % | Vec % | Scal % | GpSimd % | CC % | Bottleneck |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| drafter | draft | 7.083 | 9.97 | 6.34 | 106 | 14.8 | 94 | 44 | 112 | 21 | 40 | 18 | 36 | 23 | latency/dependency-bound (busiest DMA 44%) |
| L14 | c2_source_engram | 3.745 | 15.78 | 10.03 | 146 | 20.4 | 108 | 51 | 150 | 32 | 34 | 16 | 45 | 20 | latency/dependency-bound (busiest DMA 51%) |
| L01 | window_engram | 3.560 | 18.16 | 11.55 | 151 | 21.1 | 120 | 53 | 156 | 33 | 36 | 19 | 47 | 20 | latency/dependency-bound (busiest DMA 53%) |
| L00 | window | 3.458 | 4.43 | 2.82 | 52 | 7.3 | 85 | 23 | 56 | 11 | 22 | 12 | 16 | 73 | collective wait (73%) |
| L08 | c2_source | 2.665 | 9.33 | 5.93 | 91 | 12.7 | 103 | 41 | 96 | 23 | 41 | 22 | 31 | 22 | latency/dependency-bound (busiest Vector 41%) |
| L24 | c1_index | 2.515 | 9.78 | 6.22 | 88 | 12.4 | 111 | 41 | 94 | 22 | 42 | 23 | 34 | 26 | latency/dependency-bound (busiest Vector 42%) |
| L02 | c2_source | 2.507 | 6.31 | 4.01 | 84 | 11.7 | 75 | 37 | 89 | 20 | 36 | 17 | 26 | 25 | latency/dependency-bound (busiest DMA 37%) |
| embed + n-gram hash + hc_expand (XLA graph) | glue | 2.484 | 0.00 | 0.00 | 0 | 0.1 | 0 | 1 | 0 | 0 | 2 | 1 | 3 | 96 | collective wait (96%) |
| L20 | c1_source | 2.467 | 6.58 | 4.18 | 80 | 11.1 | 83 | 38 | 85 | 18 | 36 | 17 | 29 | 30 | latency/dependency-bound (busiest DMA 38%) |
| L28 | c1_index | 2.425 | 6.41 | 4.08 | 79 | 11.0 | 82 | 36 | 84 | 18 | 35 | 17 | 27 | 33 | latency/dependency-bound (busiest DMA 36%) |
| L35 | c1 | 2.398 | 15.23 | 9.68 | 109 | 15.2 | 140 | 50 | 114 | 29 | 53 | 33 | 43 | 15 | latency/dependency-bound (busiest Vector 53%) |
| L33 | c1 | 2.388 | 7.70 | 4.90 | 82 | 11.5 | 94 | 37 | 88 | 18 | 36 | 19 | 28 | 34 | latency/dependency-bound (busiest DMA 37%) |
| L32 | c1_index | 2.362 | 9.14 | 5.81 | 90 | 12.5 | 102 | 41 | 95 | 22 | 42 | 22 | 32 | 25 | latency/dependency-bound (busiest Vector 42%) |
| L36 | c1_index | 2.348 | 10.48 | 6.66 | 95 | 13.2 | 111 | 44 | 100 | 24 | 45 | 25 | 36 | 21 | latency/dependency-bound (busiest Vector 45%) |
| L19 | c2 | 2.316 | 6.64 | 4.22 | 80 | 11.1 | 83 | 35 | 85 | 17 | 34 | 17 | 26 | 37 | collective wait (37%) |
| L15 | c2 | 2.300 | 6.68 | 4.25 | 80 | 11.2 | 83 | 36 | 86 | 17 | 34 | 17 | 25 | 35 | latency/dependency-bound (busiest DMA 36%) |
| L37 | c1_tap | 2.258 | 6.81 | 4.33 | 83 | 11.5 | 82 | 38 | 88 | 18 | 36 | 18 | 27 | 32 | latency/dependency-bound (busiest DMA 38%) |
| L30 | c1 | 2.242 | 13.59 | 8.64 | 107 | 14.9 | 127 | 48 | 112 | 27 | 50 | 30 | 40 | 16 | latency/dependency-bound (busiest Vector 50%) |
| L34 | c1 | 2.238 | 6.87 | 4.37 | 83 | 11.6 | 83 | 36 | 89 | 17 | 35 | 18 | 27 | 34 | latency/dependency-bound (busiest DMA 36%) |
| L29 | c1 | 2.223 | 6.91 | 4.40 | 84 | 11.7 | 83 | 37 | 89 | 17 | 35 | 18 | 27 | 33 | latency/dependency-bound (busiest DMA 37%) |
| L09 | c2 | 2.212 | 9.68 | 6.16 | 93 | 13.0 | 104 | 42 | 99 | 22 | 42 | 23 | 32 | 26 | latency/dependency-bound (busiest DMA 42%) |
| L39 | c1_tap | 2.210 | 9.69 | 6.16 | 94 | 13.1 | 103 | 43 | 100 | 22 | 44 | 24 | 33 | 24 | latency/dependency-bound (busiest Vector 44%) |
| L38 | c1_tap | 2.201 | 6.99 | 4.44 | 85 | 11.8 | 82 | 38 | 91 | 18 | 37 | 18 | 28 | 31 | latency/dependency-bound (busiest DMA 38%) |
| L03 | c2 | 2.195 | 7.00 | 4.45 | 84 | 11.7 | 83 | 37 | 90 | 18 | 36 | 18 | 27 | 32 | latency/dependency-bound (busiest DMA 37%) |
| L27 | c1 | 2.172 | 12.64 | 8.04 | 105 | 14.7 | 120 | 47 | 111 | 25 | 49 | 28 | 39 | 17 | latency/dependency-bound (busiest Vector 49%) |
| L26 | c1 | 2.168 | 11.27 | 7.17 | 100 | 14.0 | 112 | 45 | 106 | 24 | 46 | 26 | 36 | 21 | latency/dependency-bound (busiest Vector 46%) |
| L23 | c1 | 2.158 | 7.12 | 4.53 | 86 | 12.0 | 83 | 38 | 92 | 18 | 36 | 18 | 28 | 31 | latency/dependency-bound (busiest DMA 38%) |
| L06 | c2 | 2.156 | 9.93 | 6.31 | 95 | 13.3 | 104 | 43 | 101 | 22 | 43 | 24 | 33 | 25 | latency/dependency-bound (busiest Vector 43%) |
| L22 | c1 | 2.155 | 7.13 | 4.53 | 86 | 12.1 | 83 | 38 | 92 | 18 | 36 | 18 | 28 | 32 | latency/dependency-bound (busiest DMA 38%) |
| L17 | c2 | 2.155 | 14.14 | 8.99 | 110 | 15.4 | 128 | 50 | 116 | 28 | 53 | 32 | 42 | 13 | latency/dependency-bound (busiest Vector 53%) |
| L31 | c1 | 2.153 | 7.14 | 4.54 | 86 | 12.1 | 83 | 38 | 92 | 18 | 36 | 18 | 28 | 32 | latency/dependency-bound (busiest DMA 38%) |
| L12 | c2 | 2.152 | 7.14 | 4.54 | 86 | 12.0 | 83 | 38 | 92 | 18 | 36 | 18 | 28 | 32 | latency/dependency-bound (busiest DMA 38%) |
| L13 | c2 | 2.151 | 11.36 | 7.22 | 101 | 14.1 | 113 | 45 | 107 | 24 | 46 | 26 | 36 | 21 | latency/dependency-bound (busiest Vector 46%) |
| L11 | c2 | 2.151 | 7.15 | 4.54 | 86 | 12.0 | 83 | 37 | 92 | 18 | 36 | 18 | 28 | 32 | latency/dependency-bound (busiest DMA 37%) |
| L07 | c2 | 2.147 | 9.97 | 6.34 | 96 | 13.4 | 104 | 43 | 102 | 22 | 43 | 24 | 33 | 24 | latency/dependency-bound (busiest Vector 43%) |
| L05 | c2 | 2.144 | 7.17 | 4.56 | 86 | 12.0 | 83 | 38 | 92 | 18 | 36 | 18 | 28 | 32 | latency/dependency-bound (busiest DMA 38%) |
| L16 | c2 | 2.142 | 10.00 | 6.36 | 96 | 13.4 | 104 | 43 | 102 | 22 | 43 | 24 | 33 | 24 | latency/dependency-bound (busiest Vector 43%) |
| L25 | c1 | 2.141 | 11.41 | 7.25 | 102 | 14.2 | 112 | 46 | 108 | 24 | 46 | 27 | 36 | 19 | latency/dependency-bound (busiest Vector 46%) |
| L21 | c1 | 2.139 | 10.01 | 6.36 | 97 | 13.5 | 103 | 44 | 103 | 22 | 43 | 24 | 34 | 22 | latency/dependency-bound (busiest DMA 44%) |
| L18 | c2 | 2.068 | 8.89 | 5.65 | 94 | 13.2 | 94 | 42 | 101 | 21 | 41 | 22 | 32 | 26 | latency/dependency-bound (busiest DMA 42%) |
| L04 | c2 | 2.056 | 11.88 | 7.55 | 105 | 14.7 | 113 | 47 | 111 | 25 | 48 | 27 | 38 | 17 | latency/dependency-bound (busiest Vector 48%) |
| L10 | c2 | 1.991 | 7.72 | 4.91 | 93 | 13.0 | 83 | 41 | 99 | 20 | 39 | 20 | 30 | 27 | latency/dependency-bound (busiest DMA 41%) |
| final norm + LM head + sampler (XLA graph) | glue | 1.082 | 5.03 | 3.20 | 54 | 7.6 | 93 | 42 | 59 | 13 | 26 | 7 | 38 | 17 | latency/dependency-bound (busiest DMA 42%) |
| draft-id cast of the previous step (XLA graph) | glue | 0.043 | 0.00 | 0.00 | 0 | 0.0 | 0 | 20 | 4 | 0 | 3 | 0 | 6 | 0 | latency/dependency-bound (busiest DMA 20%) |
| engram hash-id slice, layer 1 (XLA graph) | glue | 0.036 | 0.00 | 0.00 | 0 | 0.0 | 0 | 25 | 5 | 0 | 0 | 0 | 26 | 0 | latency/dependency-bound (busiest GpSimd 26%) |
| engram hash-id slice, layer 14 (XLA graph) | glue | 0.036 | 0.00 | 0.00 | 0 | 0.0 | 0 | 27 | 5 | 0 | 0 | 0 | 26 | 0 | latency/dependency-bound (busiest DMA 27%) |
| sampler / draft-meta glue (XLA graphs) | glue | 0.035 | 0.00 | 0.00 | 0 | 0.0 | 0 | 10 | 2 | 0 | 5 | 8 | 25 | 0 | latency/dependency-bound (busiest GpSimd 25%) |

**Where the engine time goes inside the 40 layer kernels**

| Kernel function | Tensor (ms) | Vector (ms) | Scalar (ms) | GpSimd (ms) | Sync (ms) | Total engine-ms | Useful GFLOP | Transpose GFLOP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mk_gemm.py:gemm_staged` | 26.2 | 0.0 | 0.0 | 65.0 | 0.0 | 91.3 | 689.3 | 0.0 |
| `moe_expert.py:_gemm_fp4` | 6.4 | 1.4 | 5.9 | 5.2 | 0.0 | 18.9 | 181.2 | 0.0 |
| `mk_common.py:hc_in_tile` | 11.0 | 4.7 | 2.4 | 0.1 | 0.1 | 18.4 | 1.9 | 0.1 |
| `blockquant.py:_qdq_fp8` | 0.0 | 11.1 | 2.6 | 0.0 | 0.0 | 13.7 | 0.0 | 0.0 |
| `mk_gemm.py:stage_rows_T` | 7.7 | 4.0 | 0.0 | 0.0 | 0.4 | 12.1 | 0.0 | 78.3 |
| `mk_common.py:hc_out_tile` | 0.0 | 8.1 | 1.8 | 0.0 | 0.6 | 10.5 | 0.0 | 0.0 |
| `mk_common.py:attn_token_body` | 5.5 | 2.2 | 0.5 | 2.3 | 0.0 | 10.5 | 2.4 | 19.4 |
| `mk_common.py:rowsum_div` | 0.0 | 2.3 | 3.5 | 0.0 | 0.0 | 5.8 | 0.0 | 0.0 |
| `moe_expert.py:_fp4_dequant` | 0.0 | 5.0 | 0.0 | 0.0 | 0.0 | 5.0 | 0.0 | 0.0 |
| `mk_common.py:gemm_fp32` | 2.4 | 0.3 | 0.0 | 0.1 | 0.0 | 2.8 | 3.8 | 0.0 |

<sub>all layer executions of step 1 (40); wall 94.9 ms. Engine busy time per function, summed over both physical cores and all listed executions (semaphore waits, drains and loop branches excluded), so a row can exceed the wall.</sub>

**DMA traffic by queue** (one decode step)

| Kernels | DMA queue type | Packets | MB / step | Share | Mean packet (B) |
|---|---|---:|---:|---:|---:|
| layer | software_dynamic | 4,124,256 | 6733 | 72% | 1632 |
| layer | hardware_dynamic | 654,528 | 1838 | 20% | 2809 |
| layer | collectives | 64,420 | 271 | 3% | 4215 |
| layer | data | 112,096 | 246 | 3% | 2194 |
| layer | instruction | 50,715 | 196 | 2% | 3865 |
| layer | input | 42,580 | 32 | 0% | 755 |
| draft | software_dynamic | 363,840 | 582 | 74% | 1600 |
| draft | hardware_dynamic | 44,222 | 148 | 19% | 3340 |
| draft | data | 11,307 | 25 | 3% | 2218 |
| draft | collectives | 5,403 | 21 | 3% | 3903 |
| draft | instruction | 3,948 | 14 | 2% | 3622 |
| draft | input | 954 | 1 | 0% | 1298 |

**What a bs1 decode step is made of.**
- **A step is 117.1 ms and emits ~3.2 tokens** (DSpark mean acceptance length 3.0–3.3), which is why the bench TPOT is 36.6 ms. The 40 layer kernels take 94.5 ms (81%), the drafter 7.1 ms (6%), and the LM head and sampler graph 1.1 ms.
- **12–14 ms of every step is the step boundary, not model compute:**
  - **Host gap, ~10 ms.** Device idle is 11.8 ms, almost all of it in one gap before the step's first graph. The system trace shows only 17 tiny readbacks and a ~1 ms submit inside it; the rest is host Python (scheduler, input preparation, spec-decode bookkeeping, `StepIndices`).
  - **Rank-skew wait, ~2.4 ms.** The first graph (embed + n-gram hash + hc_expand) is 96% collective wait: a 1.26 ms BARRIER while the other 63 ranks catch up. Layer 0 (`window`) then spends 71% of its 3.5 ms in collectives for the same reason.
- **Every layer kernel is latency/dependency-bound.** Each takes 1.9–3.7 ms at MBU 11–15% (20–21% for the two engram layers) and MFU 4–12%. AI is 83–120 FLOP/B, on the memory side of the ridge: the bandwidth roof would allow 59–86 TFLOP/s and they reach 5–18, about 10–20% of the roofline.
- **The weight streaming is the bottleneck.**
  - The layers read 7.6 GB of HBM per rank per step. The fp8 linears were decoded to bf16 at load, so they stream twice the fp8 bytes. At 716 GB/s that is a 10.6 ms floor, against 95 ms measured.
  - The weight-tile DMAs are generated in software on GpSimd: 72% of the step's DMA bytes move on `software_dynamic` queues, as 4.1 M packets with a 1.6 KB mean.
  - `gemm_staged` keeps GpSimd busy for 65 engine-ms per step, about a third of each core. The DMA queues are active ~42% of the time yet move only ~90 GB/s.
- **Collectives are 29% of layer time, and mostly padding.** Each layer all-reduces the full padded 128-row fp32 tile (`cc_src[Tp=128, 5120]`, 2.62 MB) twice, although only 6 rows are live. That is 286 µs each, 22.9 ms per step.

---

## 3. in1024_bs10 (context 1280, max_num_seqs 10)

Same-session bench, unprofiled: TTFT p50 5658 ms / p99 10160 ms, TPOT p50 111.4 ms / p99 152.5 ms, decode 60.99 tok/s (10 requests, mean input 1009 tokens).

### 3.1 Prefill: the bench's burst of 10 requests (system trace)

**The ten prefills run strictly one after another, one sequence per step.** Request latencies, sorted, were 1.04, 2.06, 3.08, 4.11, 5.13, 6.16, 7.18, 8.21, 9.23 and 10.25 s: request *i* waits for the *i*−1 prefills ahead of it. Each prefill step is 1.012 s of device time plus a ~12 ms host gap before the next one (1.024 s wall). The same-session bench TTFT, p50 5.66 s and p99 10.16 s, is therefore entirely FIFO queueing behind identical 1-second prefills, not slower prefills.

**Overall latency breakdown** (per prefilled sequence, mean over the burst, descending)

Step wall (mean of 9 steps): **1024.35 ms** (steps: 1026.32, 1023.73, 1022.68, 1025.23, 1029.06, 1023.00, 1022.94, 1022.43, 1023.82 ms)

| # | Component | ms / step | % of step |
|---:|---|---:|---:|
| 1 | layer kernels: c2 (x15) | 355.17 | 34.7 |
| 2 | layer kernels: c1 (x12) | 273.31 | 26.7 |
| 3 | layer kernels: c1_index (x4) | 129.06 | 12.6 |
| 4 | layer kernels: c1_tap (x3) | 68.27 | 6.7 |
| 5 | layer kernels: c2_source (x2) | 64.36 | 6.3 |
| 6 | layer kernels: c2_source_engram (x1) | 37.40 | 3.7 |
| 7 | layer kernels: c1_source (x1) | 32.91 | 3.2 |
| 8 | layer kernels: window_engram (x1) | 21.95 | 2.1 |
| 9 | layer kernels: window (x1) | 18.96 | 1.9 |
| 10 | device idle between executions (host dispatch / sync) | 14.75 | 1.4 |
| 11 | embed + n-gram hash + hc_expand (XLA graph) | 3.48 | 0.3 |
| 12 | drafter kernel (DSpark, 3 stages) | 3.31 | 0.3 |
| 13 | final norm + LM head + sampler (XLA graph) | 1.21 | 0.1 |
| 14 | sampler / draft-meta glue (XLA graphs) | 0.14 | 0.0 |
| 15 | engram hash-id slice, layer 14 (XLA graph) | 0.04 | 0.0 |
| 16 | engram hash-id slice, layer 1 (XLA graph) | 0.03 | 0.0 |

**Per-block latency** (descending)

| # | Block | Kernel / archetype | Latency mean (ms) | min–max (ms) | % of step |
|---:|---|---|---:|---:|---:|
| 1 | L14 | c2_source_engram | 37.403 | 37.244–37.523 | 3.7 |
| 2 | L20 | c1_source | 32.906 | 32.731–32.945 | 3.2 |
| 3 | L36 | c1_index | 32.469 | 32.318–32.609 | 3.2 |
| 4 | L02 | c2_source | 32.370 | 32.013–32.679 | 3.2 |
| 5 | L24 | c1_index | 32.225 | 32.073–32.350 | 3.1 |
| 6 | L28 | c1_index | 32.197 | 32.052–32.387 | 3.1 |
| 7 | L32 | c1_index | 32.173 | 32.074–32.346 | 3.1 |
| 8 | L08 | c2_source | 31.989 | 31.731–32.257 | 3.1 |
| 9 | L05 | c2 | 24.076 | 24.065–24.082 | 2.4 |
| 10 | L17 | c2 | 23.960 | 23.769–24.025 | 2.3 |
| 11 | L11 | c2 | 23.936 | 23.734–24.269 | 2.3 |
| 12 | L15 | c2 | 23.921 | 23.787–24.026 | 2.3 |
| 13 | L04 | c2 | 23.892 | 23.766–24.026 | 2.3 |
| 14 | L18 | c2 | 23.891 | 23.510–24.257 | 2.3 |
| 15 | L10 | c2 | 23.863 | 23.583–24.032 | 2.3 |
| 16 | L13 | c2 | 23.726 | 23.552–23.801 | 2.3 |
| 17 | L12 | c2 | 23.624 | 23.403–23.665 | 2.3 |
| 18 | L03 | c2 | 23.583 | 23.365–23.620 | 2.3 |
| 19 | L07 | c2 | 23.543 | 23.331–23.577 | 2.3 |
| 20 | L06 | c2 | 23.506 | 23.265–23.889 | 2.3 |
| 21 | L19 | c2 | 23.460 | 23.287–23.784 | 2.3 |
| 22 | L29 | c1 | 23.175 | 23.135–23.387 | 2.3 |
| 23 | L25 | c1 | 23.157 | 22.882–23.385 | 2.3 |
| 24 | L09 | c2 | 23.127 | 22.907–23.365 | 2.3 |
| 25 | L16 | c2 | 23.064 | 23.051–23.073 | 2.3 |
| 26 | L37 | c1_tap | 23.053 | 23.035–23.079 | 2.3 |
| 27 | L23 | c1 | 23.041 | 22.820–23.075 | 2.2 |
| 28 | L27 | c1 | 23.026 | 22.629–23.119 | 2.2 |
| 29 | L34 | c1 | 22.878 | 22.375–23.151 | 2.2 |
| 30 | L22 | c1 | 22.836 | 22.821–22.852 | 2.2 |
| 31 | L26 | c1 | 22.777 | 22.585–23.073 | 2.2 |
| 32 | L39 | c1_tap | 22.757 | 22.745–22.768 | 2.2 |
| 33 | L35 | c1 | 22.682 | 22.349–23.060 | 2.2 |
| 34 | L21 | c1 | 22.581 | 22.388–22.886 | 2.2 |
| 35 | L31 | c1 | 22.538 | 22.330–22.605 | 2.2 |
| 36 | L33 | c1 | 22.477 | 22.386–22.651 | 2.2 |
| 37 | L38 | c1_tap | 22.463 | 22.093–22.790 | 2.2 |
| 38 | L30 | c1 | 22.146 | 21.866–22.384 | 2.2 |
| 39 | L01 | window_engram | 21.950 | 21.827–22.101 | 2.1 |
| 40 | L00 | window | 18.963 | 18.045–20.442 | 1.9 |
| 41 | embed + n-gram hash + hc_expand (XLA graph) | glue | 3.478 | 2.421–7.111 | 0.3 |
| 42 | drafter | draft | 3.306 | 3.283–3.343 | 0.3 |
| 43 | final norm + LM head + sampler (XLA graph) | glue | 1.210 | 1.186–1.216 | 0.1 |
| 44 | sampler / draft-meta glue (XLA graphs) | glue | 0.047 | 0.026–0.096 | 0.0 |
| 45 | engram hash-id slice, layer 14 (XLA graph) | glue | 0.035 | 0.029–0.039 | 0.0 |
| 46 | engram hash-id slice, layer 1 (XLA graph) | glue | 0.034 | 0.030–0.036 | 0.0 |

**Roofline:** these are the same single-sequence prefill kernels as §2.1, with the same `T` and the same work; only the KV-cache tensors are larger. The device profiler's ~2.5 GB of HBM on rank 0 does not fit beside this engine (§7). Per-kernel latencies agree with §2.1 to within 1.5% in total (1001 ms vs 987 ms for the 40 layer kernels); the largest difference is layer 0, +2.2 ms of collective wait, so the roofline in §2.1 applies.

### 3.2 Decode, bs10 (device-profiled, 2 full steps)

**Overall latency breakdown** (per decode step, descending)

Step wall (mean of 2 steps): **239.31 ms** (steps: 239.73, 238.89 ms)

| # | Component | ms / step | % of step |
|---:|---|---:|---:|
| 1 | drafter kernels (DSpark, 3 stages, x10 sequences) | 72.06 | 30.1 |
| 2 | layer kernels: c2 (x15) | 49.18 | 20.5 |
| 3 | layer kernels: c1 (x12) | 41.83 | 17.5 |
| 4 | layer kernels: c1_index (x4) | 17.63 | 7.4 |
| 5 | device idle between executions (host dispatch / sync) | 14.04 | 5.9 |
| 6 | layer kernels: c1_tap (x3) | 10.57 | 4.4 |
| 7 | layer kernels: c2_source (x2) | 8.59 | 3.6 |
| 8 | layer kernels: c2_source_engram (x1) | 5.45 | 2.3 |
| 9 | layer kernels: window (x1) | 5.30 | 2.2 |
| 10 | layer kernels: window_engram (x1) | 4.46 | 1.9 |
| 11 | layer kernels: c1_source (x1) | 4.21 | 1.8 |
| 12 | embed + n-gram hash + hc_expand (XLA graph) | 2.73 | 1.1 |
| 13 | final norm + LM head + sampler (XLA graph) | 2.44 | 1.0 |
| 14 | draft-id cast (XLA graph) | 0.68 | 0.3 |
| 15 | sampler / draft-meta glue (XLA graphs) | 0.07 | 0.0 |
| 16 | engram hash-id slice, layer 14 (XLA graph) | 0.04 | 0.0 |
| 17 | engram hash-id slice, layer 1 (XLA graph) | 0.03 | 0.0 |

Collective (all-reduce) wall inside executions: 55.11 ms / step (23.0% of step); inside the 40 layer kernels 34.79 ms (23.6% of layer-kernel time).

**Per-block latency** (descending)

| # | Block | Kernel / archetype | Latency mean (ms) | min–max (ms) | % of step |
|---:|---|---|---:|---:|---:|
| 1 | drafter seq2 | draft | 7.378 | 7.247–7.509 | 3.1 |
| 2 | drafter seq5 | draft | 7.356 | 7.249–7.462 | 3.1 |
| 3 | drafter seq1 | draft | 7.355 | 7.240–7.469 | 3.1 |
| 4 | drafter seq6 | draft | 7.271 | 7.245–7.298 | 3.0 |
| 5 | drafter seq3 | draft | 7.249 | 7.013–7.484 | 3.0 |
| 6 | drafter seq4 | draft | 7.244 | 7.239–7.248 | 3.0 |
| 7 | drafter seq7 | draft | 7.120 | 7.012–7.229 | 3.0 |
| 8 | drafter seq8 | draft | 7.075 | 7.007–7.144 | 3.0 |
| 9 | drafter seq10 | draft | 7.010 | 7.008–7.012 | 2.9 |
| 10 | drafter seq9 | draft | 7.003 | 6.996–7.009 | 2.9 |
| 11 | L14 | c2_source_engram | 5.450 | 5.439–5.461 | 2.3 |
| 12 | L00 | window | 5.303 | 5.101–5.505 | 2.2 |
| 13 | L28 | c1_index | 4.506 | 4.501–4.510 | 1.9 |
| 14 | L24 | c1_index | 4.495 | 4.481–4.508 | 1.9 |
| 15 | L01 | window_engram | 4.458 | 4.454–4.461 | 1.9 |
| 16 | L32 | c1_index | 4.383 | 4.259–4.507 | 1.8 |
| 17 | L02 | c2_source | 4.303 | 4.178–4.428 | 1.8 |
| 18 | L08 | c2_source | 4.286 | 4.169–4.402 | 1.8 |
| 19 | L36 | c1_index | 4.251 | 4.250–4.252 | 1.8 |
| 20 | L20 | c1_source | 4.210 | 4.210–4.210 | 1.8 |
| 21 | L29 | c1 | 3.734 | 3.611–3.858 | 1.6 |
| 22 | L26 | c1 | 3.680 | 3.555–3.806 | 1.5 |
| 23 | L33 | c1 | 3.606 | 3.604–3.608 | 1.5 |
| 24 | L38 | c1_tap | 3.581 | 3.578–3.583 | 1.5 |
| 25 | L27 | c1 | 3.577 | 3.575–3.578 | 1.5 |
| 26 | L15 | c2 | 3.535 | 3.532–3.538 | 1.5 |
| 27 | L37 | c1_tap | 3.519 | 3.404–3.633 | 1.5 |
| 28 | L19 | c2 | 3.478 | 3.477–3.478 | 1.5 |
| 29 | L13 | c2 | 3.475 | 3.471–3.479 | 1.5 |
| 30 | L39 | c1_tap | 3.467 | 3.348–3.586 | 1.4 |
| 31 | L35 | c1 | 3.454 | 3.335–3.573 | 1.4 |
| 32 | L23 | c1 | 3.450 | 3.332–3.569 | 1.4 |
| 33 | L31 | c1 | 3.448 | 3.329–3.567 | 1.4 |
| 34 | L30 | c1 | 3.446 | 3.330–3.562 | 1.4 |
| 35 | L22 | c1 | 3.444 | 3.326–3.562 | 1.4 |
| 36 | L09 | c2 | 3.411 | 3.296–3.526 | 1.4 |
| 37 | L25 | c1 | 3.381 | 3.375–3.387 | 1.4 |
| 38 | L17 | c2 | 3.353 | 3.234–3.472 | 1.4 |
| 39 | L05 | c2 | 3.340 | 3.199–3.480 | 1.4 |
| 40 | L34 | c1 | 3.333 | 3.330–3.336 | 1.4 |
| 41 | L03 | c2 | 3.296 | 3.295–3.298 | 1.4 |
| 42 | L21 | c1 | 3.272 | 3.170–3.375 | 1.4 |
| 43 | L04 | c2 | 3.238 | 3.236–3.240 | 1.4 |
| 44 | L12 | c2 | 3.237 | 3.235–3.240 | 1.4 |
| 45 | L16 | c2 | 3.237 | 3.235–3.239 | 1.4 |
| 46 | L11 | c2 | 3.234 | 3.226–3.242 | 1.4 |
| 47 | L06 | c2 | 3.115 | 2.994–3.236 | 1.3 |
| 48 | L18 | c2 | 3.114 | 2.999–3.228 | 1.3 |
| 49 | L10 | c2 | 3.113 | 2.997–3.230 | 1.3 |
| 50 | L07 | c2 | 3.000 | 2.999–3.002 | 1.3 |
| 51 | embed + n-gram hash + hc_expand (XLA graph) | glue | 2.732 | 2.082–3.382 | 1.1 |
| 52 | final norm + LM head + sampler (XLA graph) | glue | 2.444 | 2.438–2.450 | 1.0 |
| 53 | draft-id cast (XLA graph) | glue | 0.034 | 0.018–0.066 | 0.3 |
| 54 | sampler / draft-meta glue (XLA graphs) | glue | 0.035 | 0.030–0.041 | 0.0 |
| 55 | engram hash-id slice, layer 14 (XLA graph) | glue | 0.037 | 0.034–0.041 | 0.0 |
| 56 | engram hash-id slice, layer 1 (XLA graph) | glue | 0.026 | 0.018–0.034 | 0.0 |

**Roofline by kernel kind**

| Kernel kind | Executions | Mean latency (ms) | MFU % | MBU % | AI (FLOP/B) | DMA active % | TE % | Vec % | Scal % | GpSimd % | CC % | Transpose share of TE FLOPs | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| draft | 30 | 7.188 | 6.25 | 14.6 | 94 | 44 | 21 | 39 | 18 | 36 | 24 | 11% | latency/dependency-bound (busiest DMA 44%) |
| c2 | 45 | 3.268 | 4.95 | 12.2 | 89 | 39 | 27 | 44 | 21 | 40 | 23 | 23% | latency/dependency-bound (busiest Vector 44%) |
| c1 | 36 | 3.439 | 5.65 | 12.8 | 97 | 42 | 28 | 45 | 22 | 42 | 20 | 21% | latency/dependency-bound (busiest Vector 45%) |
| c1_index | 12 | 4.378 | 4.62 | 10.7 | 95 | 40 | 29 | 42 | 19 | 40 | 17 | 20% | latency/dependency-bound (busiest Vector 42%) |
| c1_tap | 9 | 3.494 | 3.86 | 11.5 | 74 | 37 | 24 | 39 | 17 | 37 | 28 | 26% | latency/dependency-bound (busiest Vector 39%) |
| c2_source | 6 | 4.290 | 3.63 | 10.0 | 80 | 35 | 25 | 37 | 16 | 34 | 24 | 24% | latency/dependency-bound (busiest Vector 37%) |
| glue | 77 | 0.264 | 0.52 | 3.4 | 34 | 26 | 2 | 13 | 5 | 14 | 55 | 1% | collective wait (55%) |
| c2_source_engram | 3 | 5.545 | 8.64 | 17.1 | 111 | 48 | 35 | 40 | 18 | 48 | 14 | 10% | latency/dependency-bound (busiest DMA 48%) |
| window | 3 | 5.406 | 3.39 | 6.5 | 114 | 22 | 13 | 23 | 14 | 20 | 74 | 12% | collective wait (74%) |
| window_engram | 3 | 4.381 | 9.58 | 18.4 | 114 | 47 | 31 | 35 | 18 | 44 | 22 | 6% | latency/dependency-bound (busiest DMA 47%) |
| c1_source | 3 | 4.291 | 3.14 | 9.9 | 70 | 37 | 25 | 37 | 15 | 37 | 24 | 27% | latency/dependency-bound (busiest Vector 37%) |

**Roofline per block** (descending by latency)

| Block | Kernel / archetype | Latency (ms) | TFLOP/s | MFU % | HBM GB/s | MBU % | AI (FLOP/B) | DMA active % | DMA GB/s | TE % | Vec % | Scal % | GpSimd % | CC % | Bottleneck |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| drafter seq2 | draft | 7.334 | 9.63 | 6.12 | 103 | 14.3 | 94 | 43 | 108 | 20 | 38 | 17 | 35 | 25 | latency/dependency-bound (busiest DMA 43%) |
| drafter seq5 | draft | 7.332 | 9.63 | 6.12 | 103 | 14.3 | 94 | 43 | 108 | 20 | 38 | 17 | 35 | 25 | latency/dependency-bound (busiest DMA 43%) |
| drafter seq1 | draft | 7.298 | 9.67 | 6.15 | 103 | 14.4 | 94 | 43 | 108 | 21 | 39 | 17 | 35 | 24 | latency/dependency-bound (busiest DMA 43%) |
| drafter seq3 | draft | 7.257 | 9.73 | 6.19 | 104 | 14.5 | 94 | 43 | 109 | 21 | 39 | 17 | 35 | 25 | latency/dependency-bound (busiest DMA 43%) |
| drafter seq7 | draft | 7.241 | 9.75 | 6.20 | 104 | 14.5 | 94 | 43 | 109 | 21 | 39 | 17 | 35 | 24 | latency/dependency-bound (busiest DMA 43%) |
| drafter seq6 | draft | 7.184 | 9.83 | 6.25 | 105 | 14.6 | 94 | 44 | 110 | 21 | 39 | 18 | 36 | 24 | latency/dependency-bound (busiest DMA 44%) |
| drafter seq4 | draft | 7.167 | 9.85 | 6.26 | 105 | 14.7 | 94 | 44 | 110 | 21 | 39 | 18 | 36 | 24 | latency/dependency-bound (busiest DMA 44%) |
| drafter seq8 | draft | 7.054 | 10.01 | 6.36 | 107 | 14.9 | 94 | 44 | 112 | 21 | 40 | 18 | 36 | 23 | latency/dependency-bound (busiest DMA 44%) |
| drafter seq10 | draft | 7.008 | 10.07 | 6.41 | 107 | 15.0 | 94 | 44 | 113 | 21 | 40 | 18 | 36 | 22 | latency/dependency-bound (busiest DMA 44%) |
| drafter seq9 | draft | 7.005 | 10.08 | 6.41 | 107 | 15.0 | 94 | 44 | 113 | 21 | 40 | 18 | 36 | 22 | latency/dependency-bound (busiest DMA 44%) |
| L14 | c2_source_engram | 5.545 | 13.58 | 8.64 | 122 | 17.1 | 111 | 48 | 126 | 35 | 39 | 18 | 48 | 14 | latency/dependency-bound (busiest DMA 48%) |
| L00 | window | 5.406 | 5.33 | 3.39 | 47 | 6.5 | 114 | 22 | 49 | 13 | 23 | 14 | 20 | 74 | collective wait (74%) |
| L28 | c1_index | 4.424 | 4.63 | 2.94 | 67 | 9.4 | 69 | 35 | 70 | 25 | 36 | 14 | 34 | 25 | latency/dependency-bound (busiest Vector 36%) |
| L24 | c1_index | 4.414 | 10.80 | 6.87 | 89 | 12.4 | 122 | 46 | 92 | 34 | 50 | 26 | 46 | 8 | latency/dependency-bound (busiest Vector 50%) |
| L01 | window_engram | 4.381 | 15.07 | 9.58 | 132 | 18.4 | 114 | 47 | 136 | 31 | 35 | 18 | 44 | 22 | latency/dependency-bound (busiest DMA 47%) |
| L32 | c1_index | 4.344 | 7.50 | 4.77 | 78 | 10.9 | 96 | 41 | 81 | 29 | 43 | 20 | 40 | 16 | latency/dependency-bound (busiest Vector 43%) |
| L02 | c2_source | 4.334 | 4.61 | 2.93 | 67 | 9.3 | 69 | 33 | 70 | 24 | 34 | 14 | 32 | 27 | latency/dependency-bound (busiest Vector 34%) |
| L36 | c1_index | 4.328 | 6.13 | 3.90 | 73 | 10.3 | 83 | 39 | 76 | 27 | 40 | 17 | 38 | 20 | latency/dependency-bound (busiest Vector 40%) |
| L20 | c1_source | 4.291 | 4.93 | 3.14 | 71 | 9.9 | 70 | 37 | 74 | 25 | 37 | 15 | 37 | 24 | latency/dependency-bound (busiest Vector 37%) |
| L08 | c2_source | 4.246 | 6.84 | 4.35 | 76 | 10.6 | 90 | 37 | 79 | 27 | 40 | 18 | 36 | 20 | latency/dependency-bound (busiest Vector 40%) |
| L29 | c1 | 3.614 | 7.27 | 4.62 | 83 | 11.6 | 87 | 38 | 86 | 25 | 40 | 19 | 38 | 27 | latency/dependency-bound (busiest Vector 40%) |
| L38 | c1_tap | 3.577 | 4.81 | 3.06 | 76 | 10.6 | 63 | 34 | 80 | 22 | 36 | 15 | 34 | 33 | latency/dependency-bound (busiest Vector 36%) |
| L26 | c1 | 3.563 | 9.07 | 5.77 | 90 | 12.6 | 100 | 42 | 94 | 28 | 44 | 23 | 42 | 22 | latency/dependency-bound (busiest Vector 44%) |
| L33 | c1 | 3.532 | 8.29 | 5.27 | 88 | 12.3 | 94 | 41 | 91 | 27 | 43 | 21 | 40 | 23 | latency/dependency-bound (busiest Vector 43%) |
| L15 | c2 | 3.530 | 4.88 | 3.10 | 72 | 10.1 | 67 | 32 | 76 | 22 | 35 | 15 | 32 | 34 | latency/dependency-bound (busiest Vector 35%) |
| embed + n-gram hash + hc_expand (XLA graph) | glue | 3.522 | 0.00 | 0.00 | 4 | 0.5 | 0 | 6 | 2 | 0 | 9 | 8 | 16 | 95 | collective wait (95%) |
| L27 | c1 | 3.495 | 8.38 | 5.33 | 89 | 12.4 | 94 | 41 | 92 | 27 | 43 | 21 | 40 | 23 | latency/dependency-bound (busiest Vector 43%) |
| L30 | c1 | 3.483 | 12.75 | 8.10 | 105 | 14.6 | 122 | 49 | 108 | 33 | 53 | 30 | 50 | 11 | latency/dependency-bound (busiest Vector 53%) |
| L19 | c2 | 3.477 | 4.95 | 3.15 | 74 | 10.3 | 67 | 33 | 77 | 22 | 36 | 15 | 33 | 34 | latency/dependency-bound (busiest Vector 36%) |
| L37 | c1_tap | 3.477 | 5.82 | 3.70 | 81 | 11.4 | 71 | 37 | 86 | 24 | 38 | 17 | 36 | 28 | latency/dependency-bound (busiest Vector 38%) |
| L39 | c1_tap | 3.429 | 7.66 | 4.87 | 89 | 12.4 | 86 | 40 | 93 | 27 | 43 | 20 | 40 | 24 | latency/dependency-bound (busiest Vector 43%) |
| L35 | c1 | 3.416 | 13.88 | 8.82 | 110 | 15.3 | 126 | 51 | 113 | 35 | 56 | 32 | 52 | 7 | latency/dependency-bound (busiest Vector 56%) |
| L23 | c1 | 3.411 | 5.93 | 3.77 | 82 | 11.4 | 72 | 37 | 85 | 24 | 38 | 17 | 37 | 28 | latency/dependency-bound (busiest Vector 38%) |
| L31 | c1 | 3.406 | 8.60 | 5.47 | 91 | 12.8 | 94 | 42 | 95 | 28 | 44 | 22 | 41 | 21 | latency/dependency-bound (busiest Vector 44%) |
| L22 | c1 | 3.405 | 5.05 | 3.21 | 79 | 11.0 | 64 | 36 | 82 | 23 | 36 | 15 | 35 | 30 | latency/dependency-bound (busiest Vector 36%) |
| L13 | c2 | 3.402 | 13.05 | 8.30 | 103 | 14.4 | 126 | 47 | 107 | 34 | 55 | 30 | 49 | 11 | latency/dependency-bound (busiest Vector 55%) |
| L25 | c1 | 3.379 | 12.24 | 7.78 | 105 | 14.6 | 117 | 49 | 108 | 33 | 53 | 29 | 50 | 10 | latency/dependency-bound (busiest Vector 53%) |
| L09 | c2 | 3.371 | 5.11 | 3.25 | 76 | 10.6 | 67 | 34 | 79 | 23 | 37 | 15 | 34 | 31 | latency/dependency-bound (busiest Vector 37%) |
| L34 | c1 | 3.330 | 6.98 | 4.44 | 87 | 12.2 | 80 | 40 | 91 | 26 | 41 | 19 | 39 | 24 | latency/dependency-bound (busiest Vector 41%) |
| L17 | c2 | 3.312 | 12.49 | 7.94 | 103 | 14.4 | 121 | 47 | 106 | 34 | 54 | 29 | 49 | 11 | latency/dependency-bound (busiest Vector 54%) |
| L05 | c2 | 3.306 | 7.03 | 4.47 | 84 | 11.7 | 84 | 38 | 87 | 26 | 42 | 19 | 38 | 26 | latency/dependency-bound (busiest Vector 42%) |
| L03 | c2 | 3.295 | 5.22 | 3.32 | 78 | 10.8 | 67 | 35 | 81 | 24 | 37 | 16 | 34 | 30 | latency/dependency-bound (busiest Vector 37%) |
| L04 | c2 | 3.238 | 8.11 | 5.16 | 89 | 12.4 | 91 | 40 | 92 | 28 | 45 | 21 | 40 | 22 | latency/dependency-bound (busiest Vector 45%) |
| L12 | c2 | 3.237 | 6.25 | 3.97 | 82 | 11.5 | 76 | 37 | 86 | 25 | 40 | 18 | 37 | 27 | latency/dependency-bound (busiest Vector 40%) |
| L11 | c2 | 3.236 | 5.32 | 3.38 | 79 | 11.0 | 67 | 35 | 83 | 24 | 38 | 16 | 35 | 29 | latency/dependency-bound (busiest Vector 38%) |
| L16 | c2 | 3.236 | 9.99 | 6.35 | 95 | 13.3 | 105 | 43 | 99 | 31 | 49 | 25 | 44 | 17 | latency/dependency-bound (busiest Vector 49%) |
| L21 | c1 | 3.229 | 8.14 | 5.17 | 93 | 13.0 | 87 | 43 | 97 | 29 | 45 | 21 | 43 | 17 | latency/dependency-bound (busiest Vector 45%) |
| L06 | c2 | 3.155 | 8.33 | 5.29 | 91 | 12.7 | 91 | 41 | 95 | 29 | 46 | 22 | 41 | 20 | latency/dependency-bound (busiest Vector 46%) |
| L18 | c2 | 3.153 | 12.16 | 7.73 | 105 | 14.6 | 116 | 48 | 108 | 34 | 54 | 29 | 49 | 9 | latency/dependency-bound (busiest Vector 54%) |
| L10 | c2 | 3.074 | 5.60 | 3.56 | 83 | 11.6 | 67 | 37 | 87 | 25 | 40 | 17 | 37 | 25 | latency/dependency-bound (busiest Vector 40%) |
| L07 | c2 | 2.998 | 8.76 | 5.57 | 96 | 13.4 | 91 | 43 | 100 | 30 | 48 | 23 | 44 | 15 | latency/dependency-bound (busiest Vector 48%) |
| final norm + LM head + sampler (XLA graph) | glue | 2.457 | 2.27 | 1.44 | 62 | 8.6 | 37 | 60 | 79 | 4 | 22 | 3 | 12 | 17 | latency/dependency-bound (busiest DMA 60%) |
| sampler / draft-meta glue (XLA graphs) | glue | 0.035 | 0.00 | 0.00 | 0 | 0.0 | 0 | 10 | 2 | 0 | 7 | 13 | 29 | 0 | latency/dependency-bound (busiest GpSimd 29%) |
| draft-id cast (XLA graph) | glue | 0.034 | 0.00 | 0.00 | 0 | 0.0 | 0 | 14 | 3 | 0 | 1 | 0 | 19 | 0 | latency/dependency-bound (busiest GpSimd 19%) |
| engram hash-id slice, layer 14 (XLA graph) | glue | 0.032 | 0.00 | 0.00 | 1 | 0.1 | 0 | 23 | 5 | 0 | 16 | 0 | 30 | 0 | latency/dependency-bound (busiest GpSimd 30%) |
| engram hash-id slice, layer 1 (XLA graph) | glue | 0.029 | 0.00 | 0.00 | 1 | 0.1 | 0 | 23 | 5 | 0 | 17 | 0 | 33 | 0 | latency/dependency-bound (busiest GpSimd 33%) |

**Where the engine time goes inside the 40 layer kernels**

| Kernel function | Tensor (ms) | Vector (ms) | Scalar (ms) | GpSimd (ms) | Sync (ms) | Total engine-ms | Useful GFLOP | Transpose GFLOP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mk_common.py:attn_token_body` | 54.6 | 22.2 | 4.1 | 23.0 | 0.0 | 104.0 | 24.2 | 193.6 |
| `mk_gemm.py:gemm_staged` | 26.2 | 0.0 | 0.0 | 63.0 | 0.0 | 89.3 | 689.3 | 0.0 |
| `moe_expert.py:_gemm_fp4` | 15.0 | 3.4 | 13.9 | 12.0 | 0.0 | 44.3 | 425.8 | 0.0 |
| `mk_common.py:hc_in_tile` | 17.3 | 5.4 | 3.0 | 0.2 | 0.2 | 26.1 | 18.9 | 11.8 |
| `blockquant.py:_qdq_fp8` | 0.0 | 11.1 | 2.6 | 0.0 | 0.0 | 13.7 | 0.0 | 0.0 |
| `mk_common.py:hc_out_tile` | 0.0 | 8.2 | 2.5 | 1.0 | 1.2 | 12.9 | 0.0 | 0.0 |
| `mk_gemm.py:stage_rows_T` | 7.7 | 4.0 | 0.0 | 0.0 | 0.4 | 12.1 | 0.0 | 78.3 |
| `mk_common.py:gather_rows` | 0.0 | 0.0 | 0.0 | 8.9 | 3.0 | 11.9 | 0.0 | 0.0 |
| `moe_expert.py:_fp4_dequant` | 0.0 | 11.8 | 0.0 | 0.0 | 0.0 | 11.8 | 0.0 | 0.0 |
| `mk_compress.py:body` | 5.7 | 2.2 | 0.0 | 1.2 | 0.0 | 9.1 | 0.5 | 0.0 |

<sub>all layer executions of step 1 (40); wall 146.4 ms. Engine busy time per function, summed over both physical cores and all listed executions (semaphore waits, drains and loop branches excluded), so a row can exceed the wall.</sub>

**What changes at bs10.**
- **A step is 239.3 ms for 10 sequences.** The bench TPOT p50 is 111 ms, about 2.15 tokens per step per sequence.
- **The drafter is the largest component: 72.1 ms, 30% of the step.** It runs once per sequence, ten 7.2 ms kernels back to back. Each re-streams the same 0.75 GB of drafter weights (7.5 GB per step) at MFU 6% and MBU 15%. Batching the drafter over the ten sequences would read those weights once.
- **Layer kernels grow 56% for 10× the rows** (94.5 → 147.2 ms). The GEMMs cost the same, because 60 rows are still one 128-row tile and the same weights stream. The per-token attention body is now the largest layer function (104 engine-ms, again mostly transposes), and `gather_rows`, the dense slot-table staging, appears at 12 engine-ms.
- **Collectives inside the layers are 34.8 ms, 24% of layer time.** The all-reduce payload is still 2.62 MB, now with 60 live rows.

---

## 4. in8192_bs1 (context 8448, max_num_seqs 1)

Same-session bench, unprofiled: TTFT p50 8852 ms / p99 8859 ms, TPOT p50 39.5 ms / p99 43.2 ms, decode 9.26 tok/s (8 requests, mean input 8146 tokens).

### 4.1 Prefill: one 8143-token request = 4 chunks of 2048 tokens (system trace)

**A single 8143-token request is four 2048-token chunk steps** (the port sets `max_num_batched_tokens` to 2048). Each chunk is a full pass over all 40 layer kernels and the drafter. The chunks take 2.22, 2.20, 2.20 and 2.20 s, the device window from the first graph to the last is 8.82 s, and the request latency is 8.85 s, the same as the same-session bench TTFT (p50 8852 ms).
- **The cost per chunk is flat**: chunk 4 attends to 4× the context of chunk 1 and costs the same. The compressed-KV reads and the indexer are sized by the engine's context (8448), not by the tokens already prefilled.
- **The eight index-source layers take 37% of every chunk** (`c1_index`/`c1_source` ~112 ms, `c2_source` ~84 ms, `c2_source_engram` ~96 ms, against ~43 ms for a consumer layer). At 1k they are only 1.4× a consumer layer; at 8k they are 2–2.6×, because the lightning indexer scores 2048 queries against up to 8192 compressed groups.
- **Host and glue are under 1.5% of a chunk**: 12.7 ms of device idle per chunk (mostly an ~11 ms host gap before each chunk step's first graph), plus 16 ms of drafter, embed and head graphs.

**Overall latency breakdown** (per 2048-token chunk, mean of the 4 chunks, descending)

Step wall (mean of 4 steps): **2204.19 ms** (steps: 2219.75, 2200.11, 2199.97, 2196.93 ms)

| # | Component | ms / step | % of step |
|---:|---|---:|---:|
| 1 | layer kernels: c2 (x15) | 640.21 | 29.0 |
| 2 | layer kernels: c1 (x12) | 509.40 | 23.1 |
| 3 | layer kernels: c1_index (x4) | 446.96 | 20.3 |
| 4 | layer kernels: c2_source (x2) | 166.74 | 7.6 |
| 5 | layer kernels: c1_tap (x3) | 126.96 | 5.8 |
| 6 | layer kernels: c1_source (x1) | 112.82 | 5.1 |
| 7 | layer kernels: c2_source_engram (x1) | 95.46 | 4.3 |
| 8 | layer kernels: window_engram (x1) | 42.71 | 1.9 |
| 9 | layer kernels: window (x1) | 33.85 | 1.5 |
| 10 | device idle between executions (host dispatch / sync) | 12.73 | 0.6 |
| 11 | drafter kernel (DSpark, 3 stages) | 10.39 | 0.5 |
| 12 | embed + n-gram hash + hc_expand (XLA graph) | 3.84 | 0.2 |
| 13 | final norm + LM head + sampler (XLA graph) | 1.93 | 0.1 |
| 14 | sampler / draft-meta glue (XLA graphs) | 0.11 | 0.0 |
| 15 | engram hash-id slice, layer 14 (XLA graph) | 0.04 | 0.0 |
| 16 | engram hash-id slice, layer 1 (XLA graph) | 0.04 | 0.0 |

**Per-block latency** (descending, per chunk)

| # | Block | Kernel / archetype | Latency mean (ms) | min–max (ms) | % of step |
|---:|---|---|---:|---:|---:|
| 1 | L20 | c1_source | 112.823 | 112.640–113.297 | 5.1 |
| 2 | L36 | c1_index | 112.784 | 112.499–113.076 | 5.1 |
| 3 | L28 | c1_index | 111.856 | 111.459–112.378 | 5.1 |
| 4 | L24 | c1_index | 111.199 | 110.873–111.791 | 5.0 |
| 5 | L32 | c1_index | 111.124 | 110.782–111.970 | 5.0 |
| 6 | L14 | c2_source_engram | 95.461 | 95.089–96.518 | 4.3 |
| 7 | L08 | c2_source | 83.435 | 82.929–84.285 | 3.8 |
| 8 | L02 | c2_source | 83.302 | 82.871–84.242 | 3.8 |
| 9 | L18 | c2 | 43.475 | 43.069–44.445 | 2.0 |
| 10 | L05 | c2 | 43.464 | 43.105–44.272 | 2.0 |
| 11 | L23 | c1 | 43.367 | 43.192–43.820 | 2.0 |
| 12 | L25 | c1 | 43.282 | 43.023–43.603 | 2.0 |
| 13 | L17 | c2 | 43.177 | 42.869–44.050 | 2.0 |
| 14 | L29 | c1 | 43.040 | 42.708–43.538 | 2.0 |
| 15 | L04 | c2 | 43.010 | 42.639–44.066 | 2.0 |
| 16 | L11 | c2 | 42.955 | 42.457–44.028 | 1.9 |
| 17 | L12 | c2 | 42.861 | 42.484–43.735 | 1.9 |
| 18 | L22 | c1 | 42.811 | 42.667–43.230 | 1.9 |
| 19 | L10 | c2 | 42.807 | 42.179–44.022 | 1.9 |
| 20 | L03 | c2 | 42.807 | 42.460–43.622 | 1.9 |
| 21 | L06 | c2 | 42.800 | 42.479–43.735 | 1.9 |
| 22 | L39 | c1_tap | 42.776 | 42.440–43.294 | 1.9 |
| 23 | L13 | c2 | 42.756 | 42.408–43.553 | 1.9 |
| 24 | L07 | c2 | 42.717 | 42.351–43.543 | 1.9 |
| 25 | L01 | window_engram | 42.714 | 42.691–42.760 | 1.9 |
| 26 | L15 | c2 | 42.628 | 42.143–43.775 | 1.9 |
| 27 | L21 | c1 | 42.611 | 42.219–43.046 | 1.9 |
| 28 | L34 | c1 | 42.611 | 42.419–43.067 | 1.9 |
| 29 | L37 | c1_tap | 42.543 | 42.035–43.325 | 1.9 |
| 30 | L35 | c1 | 42.542 | 42.222–43.037 | 1.9 |
| 31 | L27 | c1 | 42.526 | 42.148–43.042 | 1.9 |
| 32 | L26 | c1 | 42.462 | 41.953–43.011 | 1.9 |
| 33 | L19 | c2 | 41.928 | 41.455–42.835 | 1.9 |
| 34 | L31 | c1 | 41.727 | 41.493–42.362 | 1.9 |
| 35 | L38 | c1_tap | 41.637 | 41.094–42.350 | 1.9 |
| 36 | L16 | c2 | 41.421 | 41.030–42.181 | 1.9 |
| 37 | L09 | c2 | 41.401 | 40.886–42.270 | 1.9 |
| 38 | L33 | c1 | 41.336 | 40.929–42.205 | 1.9 |
| 39 | L30 | c1 | 41.088 | 40.741–41.836 | 1.9 |
| 40 | L00 | window | 33.851 | 33.480–34.141 | 1.5 |
| 41 | drafter | draft | 10.386 | 5.926–12.057 | 0.5 |
| 42 | embed + n-gram hash + hc_expand (XLA graph) | glue | 3.840 | 2.321–7.106 | 0.2 |
| 43 | final norm + LM head + sampler (XLA graph) | glue | 1.934 | 1.927–1.941 | 0.1 |
| 44 | sampler / draft-meta glue (XLA graphs) | glue | 0.037 | 0.027–0.049 | 0.0 |
| 45 | engram hash-id slice, layer 14 (XLA graph) | glue | 0.038 | 0.036–0.042 | 0.0 |
| 46 | engram hash-id slice, layer 1 (XLA graph) | glue | 0.038 | 0.036–0.041 | 0.0 |

**Roofline:** not measured. At 8448 context the device profiler cannot allocate its buffers on rank 0, even after the caching allocator is emptied (§7).

### 4.2 Decode, bs1 (device-profiled, 1 full step; 32 MB instruction buffers)

**Overall latency breakdown** (per decode step, descending)

Step wall (mean of 1 step): **119.81 ms** (steps: 119.81 ms)

| # | Component | ms / step | % of step |
|---:|---|---:|---:|
| 1 | layer kernels: c2 (x15) | 33.06 | 27.6 |
| 2 | layer kernels: c1 (x12) | 27.02 | 22.6 |
| 3 | device idle between executions (host dispatch / sync) | 11.57 | 9.7 |
| 4 | layer kernels: c1_index (x4) | 11.09 | 9.3 |
| 5 | drafter kernel (DSpark, 3 stages) | 7.23 | 6.0 |
| 6 | layer kernels: c1_tap (x3) | 6.69 | 5.6 |
| 7 | layer kernels: c2_source (x2) | 5.79 | 4.8 |
| 8 | layer kernels: c2_source_engram (x1) | 3.89 | 3.3 |
| 9 | layer kernels: window_engram (x1) | 3.65 | 3.0 |
| 10 | layer kernels: window (x1) | 3.16 | 2.6 |
| 11 | layer kernels: c1_source (x1) | 3.00 | 2.5 |
| 12 | embed + n-gram hash + hc_expand (XLA graph) | 2.35 | 2.0 |
| 13 | final norm + LM head + sampler (XLA graph) | 1.09 | 0.9 |
| 14 | sampler / draft-meta glue (XLA graphs) | 0.10 | 0.1 |
| 15 | draft-id cast of the previous step (XLA graph) | 0.04 | 0.0 |
| 16 | engram hash-id slice, layer 1 (XLA graph) | 0.03 | 0.0 |
| 17 | engram hash-id slice, layer 14 (XLA graph) | 0.03 | 0.0 |

Collective (all-reduce) wall inside executions: 31.77 ms / step (26.5% of step); inside the 40 layer kernels 27.63 ms (28.4% of layer-kernel time).

**Per-block latency** (descending)

| # | Block | Kernel / archetype | Latency mean (ms) | min–max (ms) | % of step |
|---:|---|---|---:|---:|---:|
| 1 | drafter | draft | 7.233 | 7.233–7.233 | 6.0 |
| 2 | L14 | c2_source_engram | 3.894 | 3.894–3.894 | 3.3 |
| 3 | L01 | window_engram | 3.649 | 3.649–3.649 | 3.0 |
| 4 | L00 | window | 3.163 | 3.163–3.163 | 2.6 |
| 5 | L08 | c2_source | 3.007 | 3.007–3.007 | 2.5 |
| 6 | L20 | c1_source | 3.004 | 3.004–3.004 | 2.5 |
| 7 | L24 | c1_index | 2.954 | 2.954–2.954 | 2.5 |
| 8 | L02 | c2_source | 2.779 | 2.779–2.779 | 2.3 |
| 9 | L28 | c1_index | 2.714 | 2.714–2.714 | 2.3 |
| 10 | L32 | c1_index | 2.714 | 2.714–2.714 | 2.3 |
| 11 | L36 | c1_index | 2.712 | 2.712–2.712 | 2.3 |
| 12 | L29 | c1 | 2.502 | 2.502–2.502 | 2.1 |
| 13 | L09 | c2 | 2.456 | 2.456–2.456 | 2.1 |
| 14 | L31 | c1 | 2.446 | 2.446–2.446 | 2.0 |
| 15 | L11 | c2 | 2.394 | 2.394–2.394 | 2.0 |
| 16 | embed + n-gram hash + hc_expand (XLA graph) | glue | 2.347 | 2.347–2.347 | 2.0 |
| 17 | L33 | c1 | 2.288 | 2.288–2.288 | 1.9 |
| 18 | L21 | c1 | 2.268 | 2.268–2.268 | 1.9 |
| 19 | L37 | c1_tap | 2.263 | 2.263–2.263 | 1.9 |
| 20 | L15 | c2 | 2.229 | 2.229–2.229 | 1.9 |
| 21 | L34 | c1 | 2.226 | 2.226–2.226 | 1.9 |
| 22 | L03 | c2 | 2.221 | 2.221–2.221 | 1.9 |
| 23 | L38 | c1_tap | 2.220 | 2.220–2.220 | 1.9 |
| 24 | L35 | c1 | 2.216 | 2.216–2.216 | 1.8 |
| 25 | L27 | c1 | 2.215 | 2.215–2.215 | 1.8 |
| 26 | L30 | c1 | 2.211 | 2.211–2.211 | 1.8 |
| 27 | L39 | c1_tap | 2.210 | 2.210–2.210 | 1.8 |
| 28 | L26 | c1 | 2.210 | 2.210–2.210 | 1.8 |
| 29 | L22 | c1 | 2.205 | 2.205–2.205 | 1.8 |
| 30 | L23 | c1 | 2.201 | 2.201–2.201 | 1.8 |
| 31 | L18 | c2 | 2.166 | 2.166–2.166 | 1.8 |
| 32 | L05 | c2 | 2.165 | 2.165–2.165 | 1.8 |
| 33 | L16 | c2 | 2.165 | 2.165–2.165 | 1.8 |
| 34 | L12 | c2 | 2.164 | 2.164–2.164 | 1.8 |
| 35 | L13 | c2 | 2.164 | 2.164–2.164 | 1.8 |
| 36 | L19 | c2 | 2.160 | 2.160–2.160 | 1.8 |
| 37 | L06 | c2 | 2.160 | 2.160–2.160 | 1.8 |
| 38 | L04 | c2 | 2.157 | 2.157–2.157 | 1.8 |
| 39 | L17 | c2 | 2.156 | 2.156–2.156 | 1.8 |
| 40 | L07 | c2 | 2.151 | 2.151–2.151 | 1.8 |
| 41 | L10 | c2 | 2.148 | 2.148–2.148 | 1.8 |
| 42 | L25 | c1 | 2.035 | 2.035–2.035 | 1.7 |
| 43 | final norm + LM head + sampler (XLA graph) | glue | 1.089 | 1.089–1.089 | 0.9 |
| 44 | sampler / draft-meta glue (XLA graphs) | glue | 0.032 | 0.026–0.038 | 0.1 |
| 45 | draft-id cast of the previous step (XLA graph) | glue | 0.042 | 0.042–0.042 | 0.0 |
| 46 | engram hash-id slice, layer 1 (XLA graph) | glue | 0.033 | 0.033–0.033 | 0.0 |
| 47 | engram hash-id slice, layer 14 (XLA graph) | glue | 0.033 | 0.033–0.033 | 0.0 |

**Roofline by kernel kind**

| Kernel kind | Executions | Mean latency (ms) | MFU % | MBU % | AI (FLOP/B) | DMA active % | TE % | Vec % | Scal % | GpSimd % | CC % | Transpose share of TE FLOPs | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c2 | 30 | 2.195 | 5.33 | 13.0 | 90 | 41 | 20 | 39 | 21 | 33 | 29 | 12% | latency/dependency-bound (busiest DMA 41%) |
| c1 | 24 | 2.273 | 5.67 | 13.4 | 93 | 43 | 21 | 40 | 22 | 34 | 27 | 12% | latency/dependency-bound (busiest DMA 43%) |
| c1_index | 8 | 2.776 | 4.09 | 11.1 | 81 | 41 | 21 | 37 | 16 | 30 | 28 | 13% | latency/dependency-bound (busiest DMA 41%) |
| draft | 2 | 7.118 | 6.31 | 14.8 | 94 | 44 | 21 | 40 | 18 | 36 | 23 | 11% | latency/dependency-bound (busiest DMA 44%) |
| c1_tap | 6 | 2.233 | 4.38 | 12.6 | 76 | 40 | 18 | 37 | 18 | 30 | 32 | 14% | latency/dependency-bound (busiest DMA 40%) |
| c2_source | 4 | 2.882 | 4.49 | 11.5 | 86 | 38 | 22 | 37 | 18 | 28 | 26 | 12% | latency/dependency-bound (busiest DMA 38%) |
| c2_source_engram | 2 | 3.786 | 10.94 | 21.4 | 113 | 56 | 35 | 39 | 19 | 49 | 12 | 5% | latency/dependency-bound (busiest DMA 56%) |
| glue | 15 | 0.502 | 0.92 | 2.2 | 90 | 14 | 4 | 9 | 3 | 14 | 67 | 1% | collective wait (67%) |
| window_engram | 2 | 3.768 | 9.38 | 18.8 | 110 | 46 | 28 | 28 | 14 | 40 | 31 | 4% | latency/dependency-bound (busiest DMA 46%) |
| window | 2 | 3.459 | 2.82 | 7.3 | 85 | 23 | 11 | 22 | 11 | 16 | 72 | 12% | collective wait (72%) |
| c1_source | 2 | 3.002 | 5.37 | 11.7 | 101 | 42 | 22 | 40 | 20 | 34 | 25 | 12% | latency/dependency-bound (busiest DMA 42%) |

**Roofline per block** (descending by latency)

| Block | Kernel / archetype | Latency (ms) | TFLOP/s | MFU % | HBM GB/s | MBU % | AI (FLOP/B) | DMA active % | DMA GB/s | TE % | Vec % | Scal % | GpSimd % | CC % | Bottleneck |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| drafter | draft | 7.118 | 9.92 | 6.31 | 106 | 14.8 | 94 | 44 | 111 | 21 | 40 | 18 | 36 | 23 | latency/dependency-bound (busiest DMA 44%) |
| L14 | c2_source_engram | 3.786 | 17.21 | 10.94 | 153 | 21.4 | 113 | 56 | 157 | 35 | 39 | 19 | 49 | 12 | latency/dependency-bound (busiest DMA 56%) |
| L01 | window_engram | 3.768 | 14.75 | 9.38 | 134 | 18.8 | 110 | 46 | 139 | 28 | 29 | 14 | 40 | 31 | latency/dependency-bound (busiest DMA 46%) |
| L00 | window | 3.459 | 4.43 | 2.82 | 52 | 7.3 | 85 | 23 | 56 | 11 | 22 | 11 | 16 | 72 | collective wait (72%) |
| L20 | c1_source | 3.002 | 8.44 | 5.37 | 84 | 11.7 | 101 | 42 | 88 | 22 | 40 | 20 | 34 | 25 | latency/dependency-bound (busiest DMA 42%) |
| L02 | c2_source | 2.882 | 5.49 | 3.49 | 76 | 10.7 | 72 | 36 | 81 | 20 | 34 | 15 | 25 | 29 | latency/dependency-bound (busiest DMA 36%) |
| L08 | c2_source | 2.881 | 8.64 | 5.49 | 88 | 12.2 | 99 | 41 | 92 | 24 | 41 | 20 | 31 | 22 | latency/dependency-bound (busiest DMA 41%) |
| L36 | c1_index | 2.838 | 7.09 | 4.51 | 80 | 11.2 | 88 | 42 | 85 | 21 | 39 | 18 | 31 | 27 | latency/dependency-bound (busiest DMA 42%) |
| L24 | c1_index | 2.833 | 7.10 | 4.52 | 80 | 11.2 | 88 | 41 | 85 | 21 | 38 | 18 | 31 | 27 | latency/dependency-bound (busiest DMA 41%) |
| L32 | c1_index | 2.718 | 5.74 | 3.65 | 78 | 10.9 | 74 | 40 | 83 | 20 | 36 | 15 | 29 | 28 | latency/dependency-bound (busiest DMA 40%) |
| L28 | c1_index | 2.717 | 5.74 | 3.65 | 78 | 10.9 | 74 | 40 | 83 | 20 | 36 | 15 | 29 | 28 | latency/dependency-bound (busiest DMA 40%) |
| embed + n-gram hash + hc_expand (XLA graph) | glue | 2.494 | 0.00 | 0.00 | 1 | 0.1 | 0 | 3 | 0 | 0 | 2 | 1 | 3 | 94 | collective wait (94%) |
| L30 | c1 | 2.453 | 17.35 | 11.03 | 121 | 16.9 | 144 | 56 | 126 | 32 | 58 | 37 | 51 | 8 | latency/dependency-bound (busiest Vector 58%) |
| L29 | c1 | 2.382 | 8.36 | 5.31 | 91 | 12.7 | 92 | 41 | 96 | 20 | 38 | 21 | 33 | 30 | latency/dependency-bound (busiest DMA 41%) |
| L34 | c1 | 2.341 | 6.57 | 4.18 | 86 | 12.0 | 77 | 37 | 91 | 18 | 34 | 17 | 29 | 33 | latency/dependency-bound (busiest DMA 37%) |
| L27 | c1 | 2.339 | 6.57 | 4.18 | 86 | 12.0 | 77 | 38 | 91 | 17 | 34 | 17 | 29 | 35 | latency/dependency-bound (busiest DMA 38%) |
| L09 | c2 | 2.335 | 8.52 | 5.42 | 90 | 12.5 | 95 | 40 | 95 | 20 | 38 | 21 | 32 | 31 | latency/dependency-bound (busiest DMA 40%) |
| L31 | c1 | 2.322 | 6.62 | 4.21 | 86 | 12.1 | 77 | 38 | 92 | 18 | 35 | 17 | 30 | 34 | latency/dependency-bound (busiest DMA 38%) |
| L33 | c1 | 2.290 | 10.67 | 6.78 | 102 | 14.2 | 105 | 46 | 107 | 23 | 44 | 25 | 38 | 23 | latency/dependency-bound (busiest DMA 46%) |
| L11 | c2 | 2.284 | 6.73 | 4.28 | 85 | 11.8 | 80 | 37 | 90 | 17 | 34 | 17 | 29 | 35 | latency/dependency-bound (busiest DMA 37%) |
| L19 | c2 | 2.279 | 6.74 | 4.29 | 85 | 11.8 | 80 | 37 | 90 | 18 | 35 | 17 | 29 | 35 | latency/dependency-bound (busiest DMA 37%) |
| L04 | c2 | 2.273 | 10.75 | 6.83 | 99 | 13.8 | 109 | 45 | 105 | 23 | 44 | 25 | 37 | 24 | latency/dependency-bound (busiest DMA 45%) |
| L37 | c1_tap | 2.269 | 6.77 | 4.31 | 89 | 12.4 | 76 | 40 | 94 | 18 | 36 | 18 | 30 | 32 | latency/dependency-bound (busiest DMA 40%) |
| L21 | c1 | 2.269 | 6.78 | 4.31 | 88 | 12.4 | 77 | 39 | 94 | 18 | 35 | 18 | 30 | 32 | latency/dependency-bound (busiest DMA 39%) |
| L35 | c1 | 2.236 | 12.95 | 8.24 | 111 | 15.5 | 117 | 50 | 117 | 27 | 50 | 29 | 42 | 16 | latency/dependency-bound (busiest DMA 50%) |
| L15 | c2 | 2.227 | 6.90 | 4.39 | 87 | 12.1 | 80 | 39 | 93 | 18 | 35 | 18 | 29 | 32 | latency/dependency-bound (busiest DMA 39%) |
| L03 | c2 | 2.221 | 6.92 | 4.40 | 87 | 12.2 | 80 | 39 | 93 | 18 | 35 | 18 | 29 | 33 | latency/dependency-bound (busiest DMA 39%) |
| L39 | c1_tap | 2.215 | 6.94 | 4.41 | 91 | 12.7 | 76 | 40 | 97 | 19 | 37 | 18 | 31 | 31 | latency/dependency-bound (busiest DMA 40%) |
| L38 | c1_tap | 2.215 | 6.94 | 4.41 | 91 | 12.7 | 76 | 40 | 97 | 18 | 37 | 18 | 31 | 32 | latency/dependency-bound (busiest DMA 40%) |
| L22 | c1 | 2.208 | 6.96 | 4.43 | 91 | 12.7 | 77 | 40 | 97 | 19 | 36 | 18 | 31 | 31 | latency/dependency-bound (busiest DMA 40%) |
| L26 | c1 | 2.206 | 6.97 | 4.43 | 91 | 12.7 | 77 | 40 | 97 | 18 | 36 | 18 | 31 | 31 | latency/dependency-bound (busiest DMA 40%) |
| L23 | c1 | 2.205 | 6.97 | 4.43 | 91 | 12.7 | 77 | 40 | 97 | 19 | 36 | 18 | 31 | 31 | latency/dependency-bound (busiest DMA 40%) |
| L16 | c2 | 2.166 | 7.10 | 4.51 | 89 | 12.5 | 80 | 39 | 95 | 18 | 36 | 18 | 30 | 32 | latency/dependency-bound (busiest DMA 39%) |
| L18 | c2 | 2.164 | 7.10 | 4.52 | 89 | 12.5 | 80 | 39 | 95 | 19 | 36 | 18 | 30 | 32 | latency/dependency-bound (busiest DMA 39%) |
| L13 | c2 | 2.164 | 11.29 | 7.18 | 104 | 14.5 | 109 | 47 | 110 | 25 | 46 | 26 | 39 | 20 | latency/dependency-bound (busiest DMA 47%) |
| L05 | c2 | 2.160 | 7.12 | 4.52 | 89 | 12.5 | 80 | 39 | 95 | 18 | 36 | 18 | 30 | 32 | latency/dependency-bound (busiest DMA 39%) |
| L07 | c2 | 2.155 | 11.34 | 7.21 | 104 | 14.6 | 109 | 47 | 110 | 25 | 46 | 26 | 39 | 20 | latency/dependency-bound (busiest DMA 47%) |
| L17 | c2 | 2.154 | 11.34 | 7.21 | 105 | 14.6 | 109 | 48 | 110 | 25 | 46 | 26 | 38 | 20 | latency/dependency-bound (busiest DMA 48%) |
| L06 | c2 | 2.153 | 9.24 | 5.88 | 97 | 13.6 | 95 | 43 | 103 | 22 | 41 | 22 | 35 | 26 | latency/dependency-bound (busiest DMA 43%) |
| L10 | c2 | 2.148 | 7.16 | 4.55 | 90 | 12.6 | 80 | 40 | 96 | 19 | 37 | 18 | 30 | 31 | latency/dependency-bound (busiest DMA 40%) |
| L12 | c2 | 2.045 | 7.52 | 4.78 | 95 | 13.2 | 80 | 42 | 101 | 20 | 38 | 19 | 32 | 28 | latency/dependency-bound (busiest DMA 42%) |
| L25 | c1 | 2.030 | 9.80 | 6.23 | 107 | 14.9 | 92 | 47 | 113 | 23 | 44 | 24 | 38 | 18 | latency/dependency-bound (busiest DMA 47%) |
| final norm + LM head + sampler (XLA graph) | glue | 1.089 | 5.00 | 3.18 | 54 | 7.5 | 93 | 41 | 58 | 12 | 26 | 7 | 38 | 16 | latency/dependency-bound (busiest DMA 41%) |
| draft-id cast of the previous step (XLA graph) | glue | 0.042 | 0.00 | 0.00 | 0 | 0.0 | 0 | 22 | 4 | 0 | 3 | 0 | 6 | 0 | latency/dependency-bound (busiest DMA 22%) |
| engram hash-id slice, layer 14 (XLA graph) | glue | 0.041 | 0.00 | 0.00 | 0 | 0.0 | 0 | 20 | 4 | 0 | 0 | 0 | 24 | 0 | latency/dependency-bound (busiest GpSimd 24%) |
| sampler / draft-meta glue (XLA graphs) | glue | 0.033 | 0.00 | 0.00 | 0 | 0.0 | 0 | 11 | 2 | 0 | 5 | 8 | 26 | 0 | latency/dependency-bound (busiest GpSimd 26%) |
| engram hash-id slice, layer 1 (XLA graph) | glue | 0.026 | 0.00 | 0.00 | 0 | 0.0 | 0 | 19 | 4 | 0 | 0 | 0 | 39 | 0 | latency/dependency-bound (busiest GpSimd 39%) |

**Where the engine time goes inside the 40 layer kernels**

| Kernel function | Tensor (ms) | Vector (ms) | Scalar (ms) | GpSimd (ms) | Sync (ms) | Total engine-ms | Useful GFLOP | Transpose GFLOP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mk_gemm.py:gemm_staged` | 26.4 | 0.0 | 0.0 | 65.6 | 0.0 | 92.0 | 689.3 | 0.0 |
| `mk_common.py:hc_in_tile` | 11.1 | 4.7 | 2.4 | 0.1 | 0.1 | 18.4 | 1.9 | 0.1 |
| `moe_expert.py:_gemm_fp4` | 5.4 | 1.2 | 5.0 | 4.3 | 0.0 | 15.9 | 154.0 | 0.0 |
| `blockquant.py:_qdq_fp8` | 0.0 | 11.1 | 2.6 | 0.0 | 0.0 | 13.7 | 0.0 | 0.0 |
| `mk_gemm.py:stage_rows_T` | 7.7 | 4.0 | 0.0 | 0.0 | 0.4 | 12.1 | 0.0 | 78.3 |
| `mk_common.py:hc_out_tile` | 0.0 | 8.1 | 1.8 | 0.0 | 0.6 | 10.5 | 0.0 | 0.0 |
| `mk_common.py:attn_token_body` | 5.5 | 2.2 | 0.5 | 2.3 | 0.0 | 10.5 | 2.4 | 19.4 |
| `mk_common.py:gather_rows` | 0.0 | 0.0 | 0.0 | 5.2 | 1.7 | 6.9 | 0.0 | 0.0 |
| `mk_common.py:rowsum_div` | 0.0 | 2.3 | 3.5 | 0.0 | 0.0 | 5.8 | 0.0 | 0.0 |
| `mk_compress.py:body` | 3.9 | 1.4 | 0.0 | 0.1 | 0.0 | 5.4 | 0.3 | 0.0 |

<sub>all layer executions of step 1 (40); wall 97.4 ms. Engine busy time per function, summed over both physical cores and all listed executions (semaphore waits, drains and loop branches excluded), so a row can exceed the wall.</sub>

**What 8k context costs at decode: 2.3%.** A step is 119.8 ms, against 117.1 ms at 1k context.
- **Index-source layers.** `c1_index` and `c1_source` grow 0.4–0.5 ms each, from the indexer scoring 8192 compressed groups instead of 1024.
- **Ratio-1 layers.** These now stage the compressed KV densely into the slot table. `gather_rows` moves 17.6 MB per layer (8.8 MB read + 8.8 MB written), about 9% of the layer's 200 MB of HBM traffic. This is the dense part of the "sparse" attention discussed earlier; at bs1 it is not yet a large cost.
- Everything else matches the 1k step: the ~11.6 ms host gap, the 2.3 ms skew wait in the first graph, and the 7.2 ms drafter.

---

## 5. in8192_bs4 (context 8448, max_num_seqs 4; bs10 does not fit)

bs10 at 8448 context needs 10 × 8448 / 32 = 2640 KV blocks per rank. vLLM's own sizing on this engine leaves 2523, and that is before the profiler takes HBM. So, as in the bench (`BATCH_OVERRIDES = {8192: (1, 4)}`), this cell is bs4.

Same-session bench, unprofiled: TTFT p50 13358 ms / p99 34751 ms, TPOT p50 170.2 ms / p99 402.8 ms, decode 11.73 tok/s (8 requests, mean input 8146 tokens).

### 5.1 Prefill: the bench's burst of 4 requests = 16 chunks (system trace)

**The four requests' prefills run strictly one after another.** The engine prefills one sequence per step, and a 2048-token segment is the step size, so the burst is 16 chunk steps of ~2.20 s each. Measured request latencies were 8.86, 17.70, 26.53 and 35.37 s: request *i* waits for the 4(*i*−1) chunks ahead of it. The same-session bench gives TTFT p50 13.4 s and p99 34.8 s, so at bs4 the p99 is entirely queueing.

**Where a chunk's 2.2 s goes.**
- **The eight index-source layers take 37% of a chunk** (84–113 ms each, 823 ms total). The lightning indexer scores 2048 queries × up to 8192 compressed groups, ~16× the work of the 1k prefill's 1014 × 1024.
- **Consumer layers take 42–44 ms per 2048 tokens**, about 2× the 1k prefill's 22–24 ms for 1014 tokens: they scale linearly with tokens.

**Overall latency breakdown** (per 2048-token chunk, mean over the burst, descending)

Step wall (mean of 15 steps): **2207.60 ms** (steps: 2199.73, 2202.22, 2199.30, 2229.97, 2202.37, 2200.80, 2198.00, 2229.36, 2205.00, 2204.59, 2202.60, 2234.40, 2203.88, 2205.29, 2196.40 ms)

| # | Component | ms / step | % of step |
|---:|---|---:|---:|
| 1 | layer kernels: c2 (x15) | 639.37 | 29.0 |
| 2 | layer kernels: c1 (x12) | 509.09 | 23.1 |
| 3 | layer kernels: c1_index (x4) | 446.85 | 20.2 |
| 4 | layer kernels: c2_source (x2) | 168.31 | 7.6 |
| 5 | layer kernels: c1_tap (x3) | 126.81 | 5.7 |
| 6 | layer kernels: c1_source (x1) | 112.72 | 5.1 |
| 7 | layer kernels: c2_source_engram (x1) | 95.51 | 4.3 |
| 8 | layer kernels: window_engram (x1) | 42.63 | 1.9 |
| 9 | layer kernels: window (x1) | 33.67 | 1.5 |
| 10 | device idle between executions (host dispatch / sync) | 15.84 | 0.7 |
| 11 | drafter kernel (DSpark, 3 stages) | 10.73 | 0.5 |
| 12 | embed + n-gram hash + hc_expand (XLA graph) | 3.95 | 0.2 |
| 13 | final norm + LM head + sampler (XLA graph) | 1.93 | 0.1 |
| 14 | sampler / draft-meta glue (XLA graphs) | 0.11 | 0.0 |
| 15 | engram hash-id slice, layer 14 (XLA graph) | 0.04 | 0.0 |
| 16 | engram hash-id slice, layer 1 (XLA graph) | 0.04 | 0.0 |

**Per-block latency** (descending, per chunk)

| # | Block | Kernel / archetype | Latency mean (ms) | min–max (ms) | % of step |
|---:|---|---|---:|---:|---:|
| 1 | L36 | c1_index | 112.800 | 112.337–113.484 | 5.1 |
| 2 | L20 | c1_source | 112.715 | 112.491–113.296 | 5.1 |
| 3 | L28 | c1_index | 111.857 | 111.188–112.677 | 5.1 |
| 4 | L24 | c1_index | 111.166 | 110.783–111.903 | 5.0 |
| 5 | L32 | c1_index | 111.029 | 110.421–112.058 | 5.0 |
| 6 | L14 | c2_source_engram | 95.509 | 94.907–96.818 | 4.3 |
| 7 | L08 | c2_source | 84.325 | 83.731–85.348 | 3.8 |
| 8 | L02 | c2_source | 83.983 | 83.568–85.324 | 3.8 |
| 9 | L18 | c2 | 43.667 | 43.080–44.686 | 2.0 |
| 10 | L05 | c2 | 43.374 | 42.861–44.267 | 2.0 |
| 11 | L23 | c1 | 43.247 | 42.946–43.808 | 2.0 |
| 12 | L25 | c1 | 43.240 | 42.783–43.877 | 2.0 |
| 13 | L17 | c2 | 43.106 | 42.839–44.032 | 2.0 |
| 14 | L29 | c1 | 43.007 | 42.709–43.786 | 1.9 |
| 15 | L11 | c2 | 42.971 | 42.423–44.286 | 1.9 |
| 16 | L04 | c2 | 42.930 | 42.424–44.069 | 1.9 |
| 17 | L06 | c2 | 42.867 | 42.477–43.997 | 1.9 |
| 18 | L12 | c2 | 42.770 | 42.460–43.719 | 1.9 |
| 19 | L22 | c1 | 42.770 | 42.612–43.259 | 1.9 |
| 20 | L03 | c2 | 42.709 | 42.222–43.638 | 1.9 |
| 21 | L39 | c1_tap | 42.693 | 42.447–43.333 | 1.9 |
| 22 | L13 | c2 | 42.648 | 41.930–43.585 | 1.9 |
| 23 | L10 | c2 | 42.648 | 42.133–43.647 | 1.9 |
| 24 | L07 | c2 | 42.639 | 41.924–43.530 | 1.9 |
| 25 | L01 | window_engram | 42.632 | 42.440–42.742 | 1.9 |
| 26 | L34 | c1 | 42.627 | 42.408–43.319 | 1.9 |
| 27 | L21 | c1 | 42.553 | 42.208–43.038 | 1.9 |
| 28 | L27 | c1 | 42.520 | 41.972–43.269 | 1.9 |
| 29 | L26 | c1 | 42.483 | 41.957–43.275 | 1.9 |
| 30 | L15 | c2 | 42.478 | 41.888–43.566 | 1.9 |
| 31 | L37 | c1_tap | 42.477 | 42.032–43.128 | 1.9 |
| 32 | L35 | c1 | 42.438 | 41.940–43.082 | 1.9 |
| 33 | L19 | c2 | 41.860 | 41.194–42.846 | 1.9 |
| 34 | L31 | c1 | 41.670 | 41.285–42.358 | 1.9 |
| 35 | L38 | c1_tap | 41.641 | 41.051–42.396 | 1.9 |
| 36 | L16 | c2 | 41.392 | 40.779–42.416 | 1.9 |
| 37 | L33 | c1 | 41.369 | 40.936–42.399 | 1.9 |
| 38 | L09 | c2 | 41.309 | 40.830–42.280 | 1.9 |
| 39 | L30 | c1 | 41.167 | 40.744–41.830 | 1.9 |
| 40 | L00 | window | 33.669 | 32.355–37.730 | 1.5 |
| 41 | drafter | draft | 10.730 | 5.928–12.044 | 0.5 |
| 42 | embed + n-gram hash + hc_expand (XLA graph) | glue | 3.948 | 2.423–6.642 | 0.2 |
| 43 | final norm + LM head + sampler (XLA graph) | glue | 1.933 | 1.920–1.945 | 0.1 |
| 44 | sampler / draft-meta glue (XLA graphs) | glue | 0.037 | 0.026–0.052 | 0.0 |
| 45 | engram hash-id slice, layer 14 (XLA graph) | glue | 0.040 | 0.036–0.047 | 0.0 |
| 46 | engram hash-id slice, layer 1 (XLA graph) | glue | 0.039 | 0.034–0.046 | 0.0 |

**Roofline:** not measurable, for the same reason as §4.1.

### 5.2 Decode, bs4 (device-profiled, 1 full step; 32 MB instruction buffers)

**Overall latency breakdown** (per decode step, descending)

Step wall (mean of 1 step): **176.68 ms** (steps: 176.68 ms)

| # | Component | ms / step | % of step |
|---:|---|---:|---:|
| 1 | layer kernels: c2 (x15) | 42.67 | 24.1 |
| 2 | layer kernels: c1 (x12) | 35.12 | 19.9 |
| 3 | drafter kernels (DSpark, 3 stages, x4 sequences) | 29.15 | 16.5 |
| 4 | layer kernels: c1_index (x4) | 17.31 | 9.8 |
| 5 | device idle between executions (host dispatch / sync) | 13.17 | 7.5 |
| 6 | layer kernels: c1_tap (x3) | 9.08 | 5.1 |
| 7 | layer kernels: c2_source (x2) | 7.94 | 4.5 |
| 8 | layer kernels: c2_source_engram (x1) | 5.21 | 3.0 |
| 9 | layer kernels: c1_source (x1) | 4.72 | 2.7 |
| 10 | layer kernels: window (x1) | 4.06 | 2.3 |
| 11 | layer kernels: window_engram (x1) | 3.86 | 2.2 |
| 12 | embed + n-gram hash + hc_expand (XLA graph) | 2.00 | 1.1 |
| 13 | final norm + LM head + sampler (XLA graph) | 1.86 | 1.0 |
| 14 | draft-id cast (XLA graph) | 0.37 | 0.2 |
| 15 | sampler / draft-meta glue (XLA graphs) | 0.08 | 0.0 |
| 16 | engram hash-id slice, layer 14 (XLA graph) | 0.04 | 0.0 |
| 17 | engram hash-id slice, layer 1 (XLA graph) | 0.04 | 0.0 |

Collective (all-reduce) wall inside executions: 41.48 ms / step (23.5% of step); inside the 40 layer kernels 32.41 ms (24.9% of layer-kernel time).

**Per-block latency** (descending)

| # | Block | Kernel / archetype | Latency mean (ms) | min–max (ms) | % of step |
|---:|---|---|---:|---:|---:|
| 1 | drafter seq3 | draft | 7.435 | 7.435–7.435 | 4.2 |
| 2 | drafter seq4 | draft | 7.243 | 7.243–7.243 | 4.1 |
| 3 | drafter seq2 | draft | 7.239 | 7.239–7.239 | 4.1 |
| 4 | drafter seq1 | draft | 7.239 | 7.239–7.239 | 4.1 |
| 5 | L14 | c2_source_engram | 5.214 | 5.214–5.214 | 3.0 |
| 6 | L20 | c1_source | 4.716 | 4.716–4.716 | 2.7 |
| 7 | L32 | c1_index | 4.537 | 4.537–4.537 | 2.6 |
| 8 | L36 | c1_index | 4.284 | 4.284–4.284 | 2.4 |
| 9 | L24 | c1_index | 4.245 | 4.245–4.245 | 2.4 |
| 10 | L28 | c1_index | 4.242 | 4.242–4.242 | 2.4 |
| 11 | L02 | c2_source | 4.101 | 4.101–4.101 | 2.3 |
| 12 | L00 | window | 4.063 | 4.063–4.063 | 2.3 |
| 13 | L01 | window_engram | 3.860 | 3.860–3.860 | 2.2 |
| 14 | L08 | c2_source | 3.843 | 3.843–3.843 | 2.2 |
| 15 | L33 | c1 | 3.177 | 3.177–3.177 | 1.8 |
| 16 | L30 | c1 | 3.146 | 3.146–3.146 | 1.8 |
| 17 | L15 | c2 | 3.135 | 3.135–3.135 | 1.8 |
| 18 | L11 | c2 | 3.075 | 3.075–3.075 | 1.7 |
| 19 | L17 | c2 | 3.069 | 3.069–3.069 | 1.7 |
| 20 | L37 | c1_tap | 3.060 | 3.060–3.060 | 1.7 |
| 21 | L39 | c1_tap | 3.014 | 3.014–3.014 | 1.7 |
| 22 | L38 | c1_tap | 3.006 | 3.006–3.006 | 1.7 |
| 23 | L29 | c1 | 2.967 | 2.967–2.967 | 1.7 |
| 24 | L34 | c1 | 2.927 | 2.927–2.927 | 1.7 |
| 25 | L26 | c1 | 2.916 | 2.916–2.916 | 1.7 |
| 26 | L27 | c1 | 2.915 | 2.915–2.915 | 1.7 |
| 27 | L22 | c1 | 2.913 | 2.913–2.913 | 1.6 |
| 28 | L23 | c1 | 2.911 | 2.911–2.911 | 1.6 |
| 29 | L35 | c1 | 2.908 | 2.908–2.908 | 1.6 |
| 30 | L31 | c1 | 2.905 | 2.905–2.905 | 1.6 |
| 31 | L09 | c2 | 2.887 | 2.887–2.887 | 1.6 |
| 32 | L03 | c2 | 2.879 | 2.879–2.879 | 1.6 |
| 33 | L19 | c2 | 2.843 | 2.843–2.843 | 1.6 |
| 34 | L13 | c2 | 2.842 | 2.842–2.842 | 1.6 |
| 35 | L04 | c2 | 2.840 | 2.840–2.840 | 1.6 |
| 36 | L12 | c2 | 2.839 | 2.839–2.839 | 1.6 |
| 37 | L16 | c2 | 2.837 | 2.837–2.837 | 1.6 |
| 38 | L05 | c2 | 2.834 | 2.834–2.834 | 1.6 |
| 39 | L06 | c2 | 2.830 | 2.830–2.830 | 1.6 |
| 40 | L21 | c1 | 2.731 | 2.731–2.731 | 1.5 |
| 41 | L25 | c1 | 2.701 | 2.701–2.701 | 1.5 |
| 42 | L18 | c2 | 2.601 | 2.601–2.601 | 1.5 |
| 43 | L07 | c2 | 2.588 | 2.588–2.588 | 1.5 |
| 44 | L10 | c2 | 2.568 | 2.568–2.568 | 1.5 |
| 45 | embed + n-gram hash + hc_expand (XLA graph) | glue | 2.004 | 2.004–2.004 | 1.1 |
| 46 | final norm + LM head + sampler (XLA graph) | glue | 1.855 | 1.855–1.855 | 1.0 |
| 47 | draft-id cast (XLA graph) | glue | 0.047 | 0.023–0.139 | 0.2 |
| 48 | sampler / draft-meta glue (XLA graphs) | glue | 0.040 | 0.028–0.052 | 0.0 |
| 49 | engram hash-id slice, layer 14 (XLA graph) | glue | 0.036 | 0.036–0.036 | 0.0 |
| 50 | engram hash-id slice, layer 1 (XLA graph) | glue | 0.036 | 0.036–0.036 | 0.0 |

**Roofline by kernel kind**

| Kernel kind | Executions | Mean latency (ms) | MFU % | MBU % | AI (FLOP/B) | DMA active % | TE % | Vec % | Scal % | GpSimd % | CC % | Transpose share of TE FLOPs | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c2 | 30 | 2.807 | 5.09 | 12.9 | 87 | 42 | 22 | 39 | 20 | 38 | 26 | 16% | latency/dependency-bound (busiest DMA 42%) |
| c1 | 24 | 2.907 | 4.32 | 10.7 | 89 | 47 | 19 | 30 | 16 | 39 | 22 | 15% | latency/dependency-bound (busiest DMA 47%) |
| draft | 8 | 7.227 | 4.65 | 10.5 | 97 | 43 | 17 | 25 | 10 | 31 | 24 | 11% | latency/dependency-bound (busiest DMA 43%) |
| c1_index | 8 | 4.297 | 2.38 | 7.7 | 68 | 43 | 22 | 28 | 9 | 31 | 21 | 17% | latency/dependency-bound (busiest DMA 43%) |
| c1_tap | 6 | 3.064 | 2.93 | 9.3 | 69 | 41 | 15 | 23 | 11 | 33 | 29 | 18% | latency/dependency-bound (busiest DMA 41%) |
| c2_source | 4 | 3.972 | 3.49 | 10.4 | 73 | 38 | 23 | 35 | 15 | 32 | 25 | 17% | latency/dependency-bound (busiest DMA 38%) |
| c2_source_engram | 2 | 5.063 | 8.93 | 18.2 | 108 | 52 | 34 | 38 | 17 | 47 | 12 | 7% | latency/dependency-bound (busiest DMA 52%) |
| glue | 27 | 0.355 | 0.55 | 2.1 | 57 | 29 | 3 | 10 | 4 | 10 | 48 | 1% | collective wait (48%) |
| c1_source | 2 | 4.591 | 2.69 | 7.7 | 77 | 43 | 23 | 29 | 10 | 31 | 20 | 16% | latency/dependency-bound (busiest DMA 43%) |
| window | 2 | 3.985 | 3.96 | 7.8 | 112 | 27 | 14 | 26 | 16 | 22 | 67 | 10% | collective wait (67%) |
| window_engram | 2 | 3.979 | 10.40 | 19.4 | 118 | 49 | 31 | 34 | 18 | 45 | 24 | 5% | latency/dependency-bound (busiest DMA 49%) |

**Roofline per block** (descending by latency)

| Block | Kernel / archetype | Latency (ms) | TFLOP/s | MFU % | HBM GB/s | MBU % | AI (FLOP/B) | DMA active % | DMA GB/s | TE % | Vec % | Scal % | GpSimd % | CC % | Bottleneck |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| drafter seq3 | draft | 7.339 | 7.21 | 4.58 | 74 | 10.4 | 97 | 43 | 108 | 17 | 25 | 10 | 30 | 25 | latency/dependency-bound (busiest DMA 43%) |
| drafter seq2 | draft | 7.241 | 7.31 | 4.65 | 75 | 10.5 | 97 | 43 | 109 | 17 | 25 | 10 | 31 | 24 | latency/dependency-bound (busiest DMA 43%) |
| drafter seq1 | draft | 7.207 | 7.34 | 4.67 | 76 | 10.6 | 97 | 44 | 110 | 17 | 25 | 10 | 31 | 23 | latency/dependency-bound (busiest DMA 44%) |
| drafter seq4 | draft | 7.119 | 7.43 | 4.72 | 77 | 10.7 | 97 | 44 | 111 | 17 | 26 | 11 | 31 | 23 | latency/dependency-bound (busiest DMA 44%) |
| L14 | c2_source_engram | 5.063 | 14.05 | 8.93 | 130 | 18.2 | 108 | 52 | 134 | 34 | 38 | 17 | 47 | 12 | latency/dependency-bound (busiest DMA 52%) |
| L20 | c1_source | 4.591 | 4.23 | 2.69 | 55 | 7.7 | 77 | 43 | 80 | 23 | 29 | 10 | 31 | 20 | latency/dependency-bound (busiest DMA 43%) |
| L32 | c1_index | 4.407 | 2.75 | 1.75 | 51 | 7.1 | 54 | 40 | 74 | 20 | 25 | 7 | 28 | 26 | latency/dependency-bound (busiest DMA 40%) |
| L36 | c1_index | 4.272 | 4.43 | 2.82 | 58 | 8.1 | 76 | 45 | 84 | 23 | 30 | 11 | 32 | 18 | latency/dependency-bound (busiest DMA 45%) |
| L28 | c1_index | 4.255 | 2.85 | 1.81 | 53 | 7.4 | 54 | 42 | 77 | 21 | 25 | 7 | 29 | 23 | latency/dependency-bound (busiest DMA 42%) |
| L24 | c1_index | 4.255 | 4.98 | 3.17 | 60 | 8.4 | 83 | 47 | 88 | 24 | 32 | 12 | 34 | 15 | latency/dependency-bound (busiest DMA 47%) |
| L02 | c2_source | 4.105 | 4.21 | 2.68 | 68 | 9.6 | 61 | 35 | 72 | 21 | 31 | 12 | 29 | 30 | latency/dependency-bound (busiest DMA 35%) |
| L00 | window | 3.985 | 6.23 | 3.96 | 56 | 7.8 | 112 | 27 | 59 | 14 | 26 | 16 | 22 | 67 | collective wait (67%) |
| L01 | window_engram | 3.979 | 16.36 | 10.40 | 139 | 19.4 | 118 | 49 | 143 | 31 | 34 | 19 | 45 | 24 | latency/dependency-bound (busiest DMA 49%) |
| L08 | c2_source | 3.839 | 6.86 | 4.36 | 82 | 11.4 | 84 | 42 | 85 | 26 | 39 | 17 | 35 | 19 | latency/dependency-bound (busiest DMA 42%) |
| L30 | c1 | 3.136 | 11.00 | 6.99 | 88 | 12.3 | 125 | 53 | 121 | 25 | 40 | 25 | 46 | 14 | latency/dependency-bound (busiest DMA 53%) |
| L15 | c2 | 3.133 | 5.10 | 3.24 | 76 | 10.6 | 67 | 34 | 80 | 17 | 30 | 14 | 30 | 38 | collective wait (38%) |
| L39 | c1_tap | 3.128 | 5.96 | 3.79 | 70 | 9.8 | 85 | 43 | 102 | 17 | 27 | 14 | 35 | 26 | latency/dependency-bound (busiest DMA 43%) |
| L33 | c1 | 3.078 | 6.79 | 4.32 | 74 | 10.3 | 92 | 47 | 108 | 19 | 31 | 17 | 38 | 24 | latency/dependency-bound (busiest DMA 47%) |
| L37 | c1_tap | 3.051 | 3.88 | 2.47 | 64 | 9.0 | 60 | 40 | 94 | 14 | 21 | 9 | 31 | 31 | latency/dependency-bound (busiest DMA 40%) |
| L38 | c1_tap | 3.014 | 3.93 | 2.50 | 65 | 9.1 | 60 | 40 | 95 | 14 | 21 | 9 | 31 | 31 | latency/dependency-bound (busiest DMA 40%) |
| L29 | c1 | 2.967 | 6.28 | 3.99 | 74 | 10.3 | 85 | 46 | 107 | 18 | 28 | 15 | 37 | 24 | latency/dependency-bound (busiest DMA 46%) |
| L17 | c2 | 2.960 | 14.58 | 9.27 | 113 | 15.7 | 129 | 52 | 117 | 30 | 53 | 33 | 50 | 11 | latency/dependency-bound (busiest Vector 53%) |
| L34 | c1 | 2.922 | 4.05 | 2.58 | 67 | 9.3 | 61 | 40 | 97 | 15 | 22 | 10 | 32 | 32 | latency/dependency-bound (busiest DMA 40%) |
| L22 | c1 | 2.913 | 4.07 | 2.59 | 67 | 9.4 | 61 | 40 | 98 | 15 | 22 | 10 | 32 | 32 | latency/dependency-bound (busiest DMA 40%) |
| L35 | c1 | 2.909 | 11.08 | 7.04 | 92 | 12.8 | 121 | 57 | 131 | 26 | 42 | 26 | 49 | 7 | latency/dependency-bound (busiest DMA 57%) |
| L23 | c1 | 2.909 | 4.07 | 2.59 | 67 | 9.4 | 61 | 41 | 98 | 15 | 22 | 10 | 33 | 32 | latency/dependency-bound (busiest DMA 41%) |
| L27 | c1 | 2.908 | 6.41 | 4.08 | 75 | 10.5 | 85 | 46 | 109 | 18 | 29 | 15 | 38 | 24 | latency/dependency-bound (busiest DMA 46%) |
| L31 | c1 | 2.905 | 6.42 | 4.08 | 76 | 10.5 | 85 | 46 | 109 | 18 | 29 | 15 | 38 | 24 | latency/dependency-bound (busiest DMA 46%) |
| L09 | c2 | 2.887 | 8.68 | 5.52 | 93 | 13.0 | 93 | 43 | 98 | 23 | 40 | 22 | 38 | 26 | latency/dependency-bound (busiest DMA 43%) |
| L03 | c2 | 2.883 | 5.54 | 3.52 | 82 | 11.5 | 67 | 37 | 87 | 19 | 33 | 16 | 33 | 34 | latency/dependency-bound (busiest DMA 37%) |
| L19 | c2 | 2.850 | 4.81 | 3.06 | 74 | 10.3 | 65 | 37 | 88 | 18 | 31 | 15 | 32 | 34 | latency/dependency-bound (busiest DMA 37%) |
| L04 | c2 | 2.844 | 8.81 | 5.60 | 95 | 13.2 | 93 | 43 | 99 | 23 | 40 | 22 | 39 | 25 | latency/dependency-bound (busiest DMA 43%) |
| L12 | c2 | 2.836 | 5.64 | 3.58 | 84 | 11.7 | 67 | 37 | 88 | 19 | 33 | 16 | 33 | 34 | latency/dependency-bound (busiest DMA 37%) |
| L11 | c2 | 2.835 | 5.64 | 3.58 | 84 | 11.7 | 67 | 37 | 88 | 19 | 33 | 16 | 33 | 33 | latency/dependency-bound (busiest DMA 37%) |
| L06 | c2 | 2.833 | 8.84 | 5.62 | 95 | 13.3 | 93 | 43 | 100 | 23 | 40 | 22 | 39 | 25 | latency/dependency-bound (busiest DMA 43%) |
| L05 | c2 | 2.832 | 5.64 | 3.59 | 84 | 11.7 | 67 | 37 | 88 | 19 | 33 | 16 | 33 | 33 | latency/dependency-bound (busiest DMA 37%) |
| L26 | c1 | 2.798 | 6.66 | 4.24 | 78 | 11.0 | 85 | 48 | 113 | 19 | 30 | 16 | 39 | 20 | latency/dependency-bound (busiest DMA 48%) |
| L21 | c1 | 2.726 | 6.84 | 4.35 | 80 | 11.2 | 85 | 50 | 116 | 20 | 31 | 16 | 40 | 18 | latency/dependency-bound (busiest DMA 50%) |
| L13 | c2 | 2.720 | 12.54 | 7.97 | 111 | 15.5 | 113 | 50 | 115 | 29 | 50 | 29 | 47 | 12 | latency/dependency-bound (busiest DMA 50%) |
| L16 | c2 | 2.720 | 7.54 | 4.80 | 93 | 13.0 | 81 | 42 | 98 | 22 | 38 | 20 | 38 | 26 | latency/dependency-bound (busiest DMA 42%) |
| L25 | c1 | 2.715 | 7.70 | 4.90 | 84 | 11.7 | 92 | 53 | 123 | 22 | 34 | 19 | 43 | 13 | latency/dependency-bound (busiest DMA 53%) |
| L18 | c2 | 2.595 | 11.40 | 7.25 | 110 | 15.4 | 104 | 50 | 115 | 28 | 48 | 27 | 46 | 13 | latency/dependency-bound (busiest DMA 50%) |
| L07 | c2 | 2.591 | 9.66 | 6.14 | 104 | 14.5 | 93 | 47 | 109 | 26 | 44 | 24 | 43 | 17 | latency/dependency-bound (busiest DMA 47%) |
| L10 | c2 | 2.581 | 6.19 | 3.94 | 92 | 12.9 | 67 | 41 | 97 | 21 | 37 | 18 | 36 | 27 | latency/dependency-bound (busiest DMA 41%) |
| embed + n-gram hash + hc_expand (XLA graph) | glue | 2.462 | 0.00 | 0.00 | 3 | 0.4 | 0 | 10 | 2 | 1 | 6 | 5 | 10 | 83 | collective wait (83%) |
| final norm + LM head + sampler (XLA graph) | glue | 1.855 | 2.23 | 1.42 | 36 | 5.0 | 62 | 57 | 58 | 5 | 19 | 3 | 9 | 14 | latency/dependency-bound (busiest DMA 57%) |
| draft-id cast (XLA graph) | glue | 0.044 | 0.00 | 0.00 | 0 | 0.0 | 0 | 10 | 2 | 0 | 1 | 0 | 12 | 0 | latency/dependency-bound (busiest GpSimd 12%) |
| engram hash-id slice, layer 14 (XLA graph) | glue | 0.036 | 0.00 | 0.00 | 0 | 0.0 | 0 | 26 | 5 | 0 | 12 | 0 | 24 | 0 | latency/dependency-bound (busiest DMA 26%) |
| engram hash-id slice, layer 1 (XLA graph) | glue | 0.036 | 0.00 | 0.00 | 0 | 0.0 | 0 | 24 | 5 | 0 | 13 | 0 | 25 | 0 | latency/dependency-bound (busiest GpSimd 25%) |
| sampler / draft-meta glue (XLA graphs) | glue | 0.034 | 0.00 | 0.00 | 0 | 0.0 | 0 | 19 | 4 | 0 | 4 | 4 | 11 | 0 | latency/dependency-bound (busiest DMA 19%) |

**Where the engine time goes inside the 40 layer kernels**

| Kernel function | Tensor (ms) | Vector (ms) | Scalar (ms) | GpSimd (ms) | Sync (ms) | Total engine-ms | Useful GFLOP | Transpose GFLOP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mk_gemm.py:gemm_staged` | 26.4 | 0.0 | 0.0 | 64.4 | 0.0 | 90.8 | 689.3 | 0.0 |
| `mk_common.py:attn_token_body` | 21.8 | 8.9 | 1.7 | 9.2 | 0.0 | 41.6 | 9.7 | 77.4 |
| `moe_expert.py:_gemm_fp4` | 9.9 | 2.2 | 9.1 | 7.9 | 0.0 | 29.1 | 280.9 | 0.0 |
| `mk_common.py:gather_rows` | 0.0 | 0.0 | 0.0 | 19.0 | 6.8 | 25.8 | 0.0 | 0.0 |
| `mk_compress.py:body` | 15.6 | 5.6 | 0.0 | 0.5 | 0.0 | 21.7 | 1.4 | 0.0 |
| `mk_common.py:hc_in_tile` | 12.7 | 4.9 | 2.9 | 0.2 | 0.2 | 20.9 | 7.5 | 1.9 |
| `blockquant.py:_qdq_fp8` | 0.0 | 11.1 | 2.6 | 0.0 | 0.0 | 13.7 | 0.0 | 0.0 |
| `mk_common.py:hc_out_tile` | 0.0 | 8.1 | 2.4 | 1.0 | 1.1 | 12.7 | 0.0 | 0.0 |
| `mk_gemm.py:stage_rows_T` | 7.6 | 4.0 | 0.0 | 0.0 | 0.4 | 12.0 | 0.0 | 78.3 |
| `moe_expert.py:_fp4_dequant` | 0.0 | 7.8 | 0.0 | 0.0 | 0.0 | 7.8 | 0.0 | 0.0 |

<sub>all layer executions of step 0 (40); wall 127.9 ms. Engine busy time per function, summed over both physical cores and all listed executions (semaphore waits, drains and loop branches excluded), so a row can exceed the wall.</sub>

---

## 6. Cross-config findings, ranked by measured time

Every number below is from the tables above. The lever named for each is where the measured time sits, not a tested change.

1. **No kernel is near a roofline. All 18 layer kernels and both drafter kernels are latency/dependency-bound.**
   - Decode: MBU 6.5–21% and MFU 2.4–12% (AI 68–120 FLOP/B, so the HBM roof is the relevant one).
   - Prefill: MBU 8–13% and MFU 4–14%.
   - No engine is active more than ~60% of any kernel. Engines spend the rest waiting on each other's semaphores, the DMA queues and the collectives.
2. **Decode weight streaming is the largest single inefficiency.**
   - At bs1 a step reads 7.6 GB per rank, a 10.6 ms floor at 716 GB/s, but the layer kernels take 95 ms.
   - 72% of the bytes move on GpSimd-generated (`software_dynamic`) DMA queues, as ~4 M packets of ~1.6 KB per step. GpSimd is busy ~⅓ of each core generating them, and the queues average ~90 GB/s while active.
   - Lever: larger contiguous weight DMAs on hardware-DGE or static descriptors. Keeping the fp8 linears as fp8 in HBM and decoding in SBUF would also halve the bytes, if DMA becomes the binding resource.
3. **The drafter runs once per sequence.** 7.0–7.2 ms each re-streams the same 0.75 GB of weights: 30% of the decode step at bs10 (72 ms) and 16.5% at 8k bs4 (29 ms). Lever: one drafter kernel over all B × 6 rows.
4. **The decode all-reduces carry the padded 128-row fp32 tile, 2.62 MB.** Each layer does two of these whether 6 or 60 rows are live: 286 µs each, 22.9 ms per step at bs1 (20% of the step). Counting skew waits, collectives are 24–29% of all layer-kernel time. Lever: reduce only the live rows, and consider a bf16 payload.
5. **The step boundary costs 14–17 ms per decode step** (12% at bs1). That is the ~10–14 ms host gap before the first graph plus the ~2.0–2.7 ms the first graph spends in a BARRIER waiting for slower ranks. It stays constant as batch grows, so its share falls at bs10. Lever: overlap the host preparation of step N+1 (scheduler, `StepIndices`, readbacks) with step N's device time.
6. **Prefill is dominated by the per-token attention body.**
   - At 1k it is 55% of all engine busy time in the layer kernels. 89% of its TE work is PE transposes of gathered KV, and in layer 3 it reads 39% of the layer's HBM bytes, because each query token gathers its own 640 KV rows.
   - Lever: process a block of query tokens per gathered KV tile (neighbours share 127/128 window rows) and keep K in a layout that needs no PE transpose.
7. **At 8k, the lightning indexer dominates prefill.** The eight index-source layers take 37% of every 2048-token chunk (84–113 ms each against 43 ms for a consumer layer), because scoring is O(T × compressed length). And because one segment is 2048 tokens, the 8k prompt makes 4 full passes over the weights.
8. **TTFT under load is FIFO queueing.** One sequence is prefilled per step, and decode is never mixed in. At bs10 1k, request *i* sees ≈ *i* × 1.0 s. At bs4 8k, request *i* sees ≈ *i* × 8.8 s, and the p99 of 34.8 s is the 16 chunks of the burst. This is a scheduling property, not a kernel one.

---

## 7. Capture limits, and what was done about them

- **in8192 runs at bs4, not bs10.** It does not fit: 2640 KV blocks are needed per rank, and 2523 are available before the profiler takes any HBM. This matches the bench's own `BATCH_OVERRIDES`.
- **Device-profiler HBM on rank 0.** At default sizes the device profiler takes ~2.5 GB, shown as "Profiler Buffers" in `HbmUsageSummaryByType`.
  - The **bs10 1k engine** died with `nrt_tensor_allocate status=4` in a prefill kernel's output allocation once those buffers were live.
  - On the **8448-context engines**, `start_profiling` itself fails with `Allocation Failure` once the 2048-token prefill NEFFs are resident. Emptying `libtorch_neuronx_lite`'s caching allocator first did not help: these are live tensors. A failed start then leaks enough memory to crash the next forward.
  - Consequences:
    - Prefill roofline is measured on the 1k bs1 engine only.
    - Prefill on the bs10, bs4 and 8k engines is measured from the system trace (`nc_exec_running`). Each execution is labelled by position against the 1k prefill template, and the equal-NEFF pattern of every step is verified against it.
    - The 8k decode windows ran with 32 MB instruction/DMA and 16 MB event buffers. That leaves one full step at bs1. At bs4 the instruction buffer filled during the second step, so the roofline there uses the first step's executions; latency uses the complete execution table.
- **1k prefill buffers.** The default 128 MB per-engine instruction buffer dropped everything after layer 24. 256 MB captured all of it with zero drops; 320 MB and 512 MB could not be allocated.
- **Two system-trace traps, found and avoided:**
  1. `max_iterations` counts *prefill* steps as well. A 3-iteration limit stops a burst capture after three prefills, and the stop's ~3–4 s trace flush lands mid-burst, inflating the later requests' TTFT. Burst windows ran with `max_iterations=0` and an explicit stop.
  2. `NEURON_RT_INSPECT_ENABLE=1`, which the serve profiler needs, also starts a system trace at runtime init that keeps recording through warm-up. Burst windows set `NEURON_RT_INSPECT_SYSTEM_PROFILE=0` and `NEURON_RT_INSPECT_DEVICE_PROFILE=0`, so recording starts at `/start_profile`.
- **DGE notifications were off**, the skill's default for MoE serving. The ingester warns that dynamic-DMA metadata is partial. DMA packet bytes and queue types are populated; HBM bytes come from the per-instruction counters.
- **No `XLA_IR_DEBUG` / `XLA_HLO_DEBUG` / `NEURON_FRAMEWORK_DEBUG`.** Setting them would re-key every NEFF in the cache. NKI source attribution (`nki_source_location`) is complete for the kernels; only compiler-inserted semaphore, drain and branch instructions are unattributed.
- **One profiling-only hook.** A `sitecustomize.py` (`~/profile_mk/pyhook`, on `PYTHONPATH` for profiling servers only) empties the caching allocator before `start_profiling`. Model, kernels and scheduler are untouched.
- **bs>1 TPOT is not step time divided by tokens.** A sequence's decode stalls while later requests prefill, because a prefill step carries one sequence and is never mixed with decode. That is why the same-session TPOT p50 at bs10 is 111 ms and at 8k bs4 is 161–170 ms, and why bs10 aggregate throughput varies between sessions (37–61 tok/s across the captures here, 46.6 in the archived bench).

**Artifacts.** Captures, parquet, per-execution CSVs and these tables are under `/mnt/nvme/profiles/mk-20261025/<run>/` (`raw/`, `ne/`, `analysis_*/execs.csv`, `sys_execs.csv`). Scripts are in `~/profile_mk/`: `drive.py`, `launch.sh`, `ingest.sh`, `analyze.py`, `sysan.py`, `gen_report.py`, `kernel_attr.py`, `dmaq.py`, `build_report.py`.

| Section | Run directory |
|---|---|
| 2.1 | `in1024_bs1_pf3` |
| 2.2 | `in1024_bs1` |
| 3.1 | `in1024_bs10_pf4` |
| 3.2 | `in1024_bs10_dec` |
| 4.1 | `in8192_bs1_pf3` |
| 4.2 | `in8192_bs1_dec` |
| 5.1 | `in8192_bs4_pf3` |
| 5.2 | `in8192_bs4_dec` |
