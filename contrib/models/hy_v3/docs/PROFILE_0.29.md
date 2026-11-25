# Hy3-FP8 on vllm-neuron 0.29: where the time goes

Device profile of the latency config (`scripts/serve_latency.sh`, TP=64,
EP=64, no prefix caching) on trn2.48xlarge: one 16000-token prefill followed
by decode steps, captured on TP ranks 0, 7, 31 and 63
(`--profiler-config '{"profiler": "cuda"}'` +
`neuron_profiler.activities=["device_profile"]`), exported to parquet with
`neuron-explorer view -d <dir with the NTFF and both NEFFs> --output-format parquet`.
Times are wall time on the device; module attribution is the union of the
instruction intervals tagged with each module path.

## Decode (batch 1, 16k context): 27.8 ms per step

| Part | ms / step | share | per layer |
| --- | --- | --- | --- |
| Routed experts (`moe_tkg`, all 3 local experts) | 12.6 | 45% | 0.16 ms |
| Attention megakernel (QKV, RoPE, attention, O, KV write) | 6.7 | 24% | 0.084 ms |
| Shared expert (8-way decode copy, I=192) | 1.7 | 6% | 0.021 ms |
| Other attributed (sampler, embedding, lm_head, norms) | 0.3 | 1% | |
| Unattributed instructions | 3.1 | 11% | |
| No instruction running (AllReduce 2.9 ms, DMA / semaphore waits) | 7.3 | 26% | |

161 AllReduces per step (2 per layer + 1), median 18 us.

**Biggest lever: routed experts.** At batch 1 a rank holds 3 experts but on
average fewer than 1 of them is selected, and the all-reduce waits for the
slowest rank, which usually has at most 1 selected expert. A `moe_tkg` path
that only streams the selected experts' weights would cut this from 12.6 ms
towards ~4-5 ms (estimate), i.e. ~25-30% TPOT at batch 1. The gain shrinks
as the batch grows and more local experts are hit.

## Prefill (16000 tokens): 1742 ms on device

| Part (TP rank 0) | ms |
| --- | --- |
| Collectives total | 747 |
| - ReduceScatter (161 x 128 MB) | 652 |
| - AllGather | 88 |
| Instructions running (any engine) | 487 |
| - routed experts kernel | 167 |
| - attention | 166 |
| - shared expert | 13 |

The ReduceScatter after attention takes 0.53 ms (pure transfer, ~230 GB/s).
The one after the MoE takes a median of 8.4 ms (up to 16.7 ms). Per layer, on
the four sampled ranks:

| | TP0 | TP7 | TP31 | TP63 |
| --- | --- | --- | --- | --- |
| routed-expert kernel, median | 1.27 ms | 1.64 ms | 1.46 ms | 1.82 ms |
| MoE ReduceScatter, median | 8.3 ms | 7.7 ms | 7.8 ms | 7.4 ms |
| kernel + ReduceScatter, median | 9.5 ms | 9.7 ms | 10.0 ms | 9.6 ms |

All four ranks finish the MoE stage at the same time (~9.6 ms after it starts)
while doing only 1.3-1.8 ms of expert work: they wait for a few hot ranks
elsewhere in the EP=64 group that receive several times the average number of
tokens. Over 79 MoE layers this waiting is ~600 ms, about a third of the
prefill.

### Random tokens exaggerate the imbalance

`vllm bench serve --dataset-name random` (used for every benchmark in the
README, and for this profile) sends random token ids. Prefill latency of
16000 tokens on the same server (`max_tokens=1`, median of 3):

| Prompt | Prefill |
| --- | --- |
| English docs | 1.454 s |
| Python code | 1.488 s |
| Chinese text | 1.431 s |
| MATH-500 problems + solutions | 1.509 s |
| random token ids (4 seeds) | 1.763-1.772 s |

Real text prefills 17-23% faster, so the README's TTFT numbers are
pessimistic for real traffic by roughly that much at long inputs. Real text
still leaves roughly half of the measured waiting, so expert load balancing
is still worth ~15-20% of long-prompt TTFT.

## Ideas, by expected gain

1. Decode: selective expert loading in `moe_tkg` (only selected experts), and
   MTP speculative decoding (the checkpoint ships one MTP layer).
2. Prefill: expert load balancing across EP ranks (replicate hot experts or
   place experts by measured load). Needs a per-rank routing-load measurement
   on real traffic first.
3. Not worth it: fusing the pre-RoPE QK-norm + RoPE into `qkv_proj` saves
   0.01-0.04 ms per layer (< 0.1% of TTFT) at slightly worse numerics.

## Other observations

- Throttling: `ThrottleSummary` reports activity throttling (utilization
  limit 0.5) active ~11% of the profiled window and power throttling ~5%.
  Whether this explains the ~3% single-request TPOT regression vs
  vllm-neuron 0.24 is not established.
- The instruction trace keeps only the first ~2 executions of a session
  (event buffer), so profile one prefill and one decode step per session.
