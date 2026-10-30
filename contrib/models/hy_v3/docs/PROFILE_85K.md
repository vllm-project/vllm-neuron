# Hy3-FP8 at 85k context: where the time goes, and how unbalanced the experts are

Device profile of tencent/Hy3-FP8 on one trn2.48xlarge, vllm-neuron
0.29.0.1.2.0, in the fastest long-context configuration measured so far (FP8
KV cache + MTP k=3, no DCP; README, "The same split with an FP8 KV cache"):
one cold 85k-token prefill, decode of one request, and decode of six requests
at once. Profiled 2026-10-28.

## Summary

| | time | attention | MoE compute | MoE collective (of which waiting) | rest |
|---|---|---|---|---|---|
| cold prefill, 85,000 tokens (21 x 4096) | 14.34 s | 6.81 s (48%) | 5.15 s (36%) | 1.49 s (10%; **1.05 s waiting, 7%**) | 0.44 s attention RS, 0.28 s MTP draft prefill, 0.17 s other |
| decode, 1 request, one MTP step | 47.5 ms | 12.9 ms (27%) | 16.8 ms (35%) | 11.4 ms (24%; **9.6 ms waiting, 20%**) | 1.8 ms attention AR, 3.1 ms MTP draft, 1.5 ms other |
| decode, 6 requests (batch-8 NEFF, same as 8), one MTP step | 76.6 ms | 41.9 ms (55%) | 23.1 ms (30%) | 2.4 ms (3%; **~0 waiting**) | 2.7 ms attention AR, 4.5 ms MTP draft, 2.0 ms other |

Times are wall time on the device of one rank; the three phases of each
layer are delimited by its collectives (see Method), summed over layers 1-79.
Waiting = MoE collective time minus the attention collective of the same size.

- **Load imbalance is not the main cost at 85k, but it is real.** It is 7% of
  the prefill, 20% of a single-request decode step, and ~0 in the 8-request
  decode step, where the MoE runs the all-expert `moe_tkg` kernel (same work on
  every rank). The prefill imbalance is much smaller than the third of the
  prefill measured with random-token prompts at 16k (docs/PROFILE_0.29.md).
- **Attention dominates long context.** Prefill attention grows linearly with
  the context, 89 ms per 4096-token chunk at no prior context to 560 ms at 80k
  (+23.5 ms per 4k of context), so the last chunk costs 2.1x the first. In the
  8-request decode step, attention over 8 x 85k of history is 55% of the step.
- **MoE compute is flat**: 245 ms per prefill chunk whatever the context;
  decode 16.8 ms (1 request, `moe_tkg_hits_batch` + router + shared expert)
  and 23.1 ms (8 rows, all-expert `moe_tkg`).
- **No host gaps**: consecutive NEFFs (target verify, MTP draft) start 0.07-0.08
  ms apart, so the step time is device time. Communication proper (the same-size
  attention collective) is 3% of the prefill and 4% of the decode steps.

## Deployment: how the model is split

One trn2.48xlarge: 16 Trainium2 chips, 64 logical NeuronCores (LNC=2) with
24 GiB of HBM each, connected by NeuronLink (no EFA on this instance). One
vLLM server, TP=64 with full expert parallelism (EP=64, `--enable-expert-parallel`);
every NeuronCore is one TP rank and one EP rank and runs the same NEFFs (SPMD).

| Part | Parameters | Split across the 64 ranks | Per rank |
|---|---|---|---|
| Embedding | 120,832 x 4096, BF16 | vocab-sharded | 1,888 rows |
| Attention, 80 layers | 64 Q heads, 8 KV heads, head_dim 128, QK-norm, FP8 W8A8 | 1 Q head per rank; each KV head on 8 consecutive ranks (KV replicated 8x) | QKV 4096 x 384, O 128 x 4096 |
| Dense MLP, layer 0 | intermediate 13,312, FP8 | intermediate sharded | 208, padded to 256 |
| Router, layers 1-79 | 4096 x 192, sigmoid top-8 + expert bias | replicated | full |
| Routed experts, layers 1-79 | 192 experts, intermediate 1536, FP8 (18.9 MB each) | EP: whole experts, 3 per rank (experts 3r, 3r+1, 3r+2) | 3 experts, 56.6 MB per layer |
| Shared expert, layers 1-79 | intermediate 1536, FP8 | prefill: full copy per rank; decode: a second copy sharded 8 ways over groups of 8 ranks | full + 1/8 (I=192, padded to 256) |
| lm_head | 4096 x 120,832, BF16 | vocab-sharded; on-device sampling per shard | 1,888 columns |
| MTP layer (`model.layers.80`) | enorm, hnorm, eh_proj 8192 -> 4096, one MoE decoder layer, final norm | same as a decoder layer; eh_proj column-parallel (gathered); shares embedding, lm_head and final norm with the target | |

Weights take 6.29 GiB per rank (6.21 without the MTP layer); the replicated
pieces (KV heads, router, shared expert, decode copy) are why this is more
than 1/64 of the checkpoint.

**Data flow per layer.**

- Prefill (sequence parallel): between layers each rank holds 1/64 of the
  chunk's tokens. Attention all-gathers the chunk, computes its head over all
  tokens (reading the request's cached prefix), and reduce-scatters the O
  projection back to the token shards. The router and the full shared expert
  run on the local token shard; the MoE all-gathers tokens and routing weights,
  each rank runs its 3 experts on the tokens routed to them (`moe_cte`
  `shard_on_i_hybrid`), and a reduce-scatter sums the experts' outputs and
  returns each rank its token shard. 161 reduce-scatters per chunk (1 + 2 per
  layer), 128 MB each for a 4096-token chunk.
- Decode: every rank holds every token. Attention for its head, all-reduce of
  the O projection (tokens x 8 KB); router on all tokens
  (replicated); the rank's experts that any token routed to (`moe_tkg_hits`
  for 1 token, `moe_tkg_hits_batch` up to `HY3_DECODE_HITS_MAX_T` tokens, else
  all 3 local experts with `moe_tkg`); its 1/8 of the shared expert, scaled by
  8/64; one all-reduce sums routed + shared partials. 161 all-reduces per step.
- MTP k=3 (`--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'`):
  each decode step is one target NEFF that verifies 1 + 3 tokens per request
  (rejection sampling on device) and one draft NEFF that runs the MTP layer 3
  times (KV update pass + 2 recurrent passes) to propose the next 3 tokens.
  During prefill the draft NEFF runs the MTP layer over each chunk (its KV).

**KV cache.** FP8 E4M3 with per-layer scales
(`kv_profiles/Hy3-FP8_mtp_neuron.safetensors`, 81 layers incl. the MTP
layer), packed K layout, 32-token blocks, prefix caching. Each rank stores
its 1 KV head for all 81 layers: 20.7 KB per token. Per rank, at
`KV_CAP` 0.61 (the benchmark config, `scripts/serve_85k.sh`): 697k tokens, 8
requests of 86,016 tokens; at 0.49 (this profile): 560k tokens, 6 requests.
The scheduler admits requests by worst case (`max_model_len` each).

**HBM per rank** (benchmark config, `neuron-monitor` after warmup): tensors
19.83 GiB (weights 6.29 + KV 13.47 + I/O), NEFF code 2.21 GiB (grows with the
number of NEFFs: buckets x MTP graphs), shared scratchpad 0.25 GiB, other 0.1
GiB: 22.4 of 24 GiB.

## Method

`scripts/profile_85k.sh` serves the benchmark configuration with the fewest
NEFFs the workload needs (prefill buckets 512 / 4096, batch buckets 1 / 8, one
86,016-token decode context bucket) and the Neuron profiler on TP ranks 0, 9,
18, 27, 36, 45, 54, 63 (spread over the EP groups). `scripts/profile_85k_client.py`
runs three sessions on the `bench_data/prefix` prompts (85,000 tokens, 92%
shared prefix, chat template, temperature 0.9):

| session | what |
|---|---|
| c1_cold | one request with a new `cache_salt`: all 85k tokens prefill (21 chunks), then 10 decode steps |
| c1_decode | one request, 78k prefix cached, 40 decode steps profiled while it decodes |
| c6_decode | six such requests decoding at once, 26 steps |

Each rank's session is exported with `neuron-explorer view --output-format
parquet --ignore-dma-trace` and analysed with `scripts/profile_analyze.py`
(raw output: `results/profile_85k/analysis.md`):

- **Phases from collectives** (all executions, all 8 ranks): in layer L the
  attention compute runs from the end of layer L-1's MLP collective to the
  start of the attention collective, the MoE compute (router, shared expert,
  routed experts, token packing) from the end of the attention collective to
  the start of the MoE collective. Layer 0 (dense) and the embedding / lm_head /
  sampler are "other".
- **Waiting**: the attention and MoE collectives of a layer move the same
  bytes (tokens x 4096 x BF16), and attention work is identical on every rank,
  so the attention collective is the no-wait baseline. The MoE collective's
  excess is time spent waiting for the slowest rank. It is reported as the
  minimum and maximum over the 8 sampled ranks; even the least-waiting sampled
  rank waits, so the slowest rank of the 64 is usually not sampled.
- **Module times** from the instruction trace (the first executions of each
  session only; the trace buffer fills up) for the routed-expert kernel.

Limits:

- **6 requests, not 8.** The profiler allocates ~2.9 GB of buffers per profiled
  rank when a session starts (`NEURON_RT_ENABLE_DMA_TRACING=0` does not shrink
  it), which does not fit next to the KV of 8 x 85k (13.3 GiB). With 6 requests
  the step runs the same batch-8 NEFF, and its step time (76.6 ms) matches the
  unprofiled 8-request benchmark (ITL 76.7 ms), so the c6 breakdown stands for
  8 requests.
- **8-request verify steps use `moe_tkg`.** They have 32 tokens, above
  `HY3_DECODE_HITS_MAX_T` = 16 (4 x (k + 1)), so the routed experts run the
  all-expert kernel: no routed-expert imbalance, but every rank reads all 3
  experts. `HY3_DECODE_HITS_MAX_T=32` (the hits kernel at 32 tokens) is
  slower, see below.
- 8 of 64 ranks sampled; per-expert routing counts are not recorded.

## Cold prefill, 85,000 tokens (c1_cold)

21 target executions of 4096 tokens, each followed by a 13 ms MTP draft pass
over the chunk: 14.34 s on the device. Per chunk, layers 1-79, ms (rank 0's chunk time, phases median
over the 8 ranks):

| chunk | prior context | attention | attention RS | MoE compute | MoE RS | MoE waiting (min-max over ranks) | chunk |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 89.5 | 20.7 | 245.1 | 67.4 | 43.9-48.3 | 451 |
| 3 | 8k | 136.5 | 20.5 | 245.0 | 63.4 | 41.6-43.9 | 470 |
| 6 | 20k | 206.8 | 20.7 | 244.6 | 62.0 | 39.9-41.9 | 539 |
| 11 | 40k | 324.4 | 20.9 | 243.3 | 59.5 | 36.3-39.2 | 655 |
| 16 | 60k | 441.9 | 20.9 | 245.0 | 62.4 | 39.6-42.3 | 779 |
| 19 | 72k | 512.5 | 21.1 | 244.8 | 67.2 | 44.9-48.4 | 855 |
| 21 | 80k | 559.7 | 21.2 | 249.4 | 100.7 | 77.7-81.8 | 941 |
| **sum of 21** | | **6,812** | **437** | **5,154** | **1,490** | **~1,049** | **14,061** |

- Attention is 89 ms + 23.5 ms per 4096 tokens of prior context; over the 85k
  prompt it is 48% of the prefill and grows quadratically with prompt length.
- MoE compute is 245 ms per chunk (routed-expert kernel 0.71 ms median per
  layer, the rest router, shared expert, all-gathers and token packing).
- The MoE reduce-scatter waits 36-82 ms per chunk (7% overall). Per layer the
  routed-expert kernel differs by 0.35 ms (median, p90 0.69 ms) between the
  fastest and slowest of the 8 sampled ranks (max / mean 1.31). With real text
  the hot experts change from layer to layer: each sampled rank is the slowest
  of the eight in about 1/8 of the layers, no rank is systematically hot.

## Decode, one request (c1_decode)

A step is the target verify NEFF (4 tokens) + the MTP draft NEFF: 44.2 + 3.1
ms, 47.5 ms from step to step (benchmark ITL 47.4 ms; 2.0 tokens per step at
temperature 0.9, TPOT 23.4 ms). Verify, layers 1-79, mean of 38 steps:

| phase | ms | share of the 47.5 ms step |
|---|---|---|
| attention (85k history, 4 query tokens) | 12.9 | 27% |
| attention all-reduce | 1.8 | 4% |
| MoE compute: router, shared expert (2.7), routed experts `moe_tkg_hits_batch` (8.2), token packing | 16.8 | 35% |
| MoE all-reduce | 11.4 | 24% |
| of which waiting for the slowest rank | 9.1-10.2 | 19-21% |
| layer 0, embedding, lm_head, sampler, rejection | 1.3 | 3% |
| MTP draft NEFF | 3.1 | 7% |

- **Imbalance costs ~10 ms per step.** The hits kernel only computes the local
  experts that the step's 4 tokens routed to, so ranks finish at different
  times; the routed-expert kernel per layer ranges over 0.07 ms (median, p90
  0.13 ms) across the 8 sampled ranks, max / mean 1.56, and the MoE all-reduce
  waits for the busiest of the 64. All sampled ranks wait 9-10 ms per step, so
  the busiest ranks were not sampled; a rank that runs ~0.12 ms longer per
  layer explains it.
- Attention at 85k: 12.9 ms for 80 layers (0.16 ms per layer). The FP8 history
  is 1.7 GB per step per rank (85k x 128 x 2 bytes x 80 layers), ~2.4 ms at
  this rank's HBM bandwidth; the rest is the per-layer fixed cost of the
  attention megakernel (QKV, RoPE, KV write, attention, O).

## Decode, six requests (c6_decode; stands for 8)

Target verify (batch-8 NEFF, 6 x 4 real tokens) + draft: 71.9 + 4.5 ms,
76.6 ms per step. Verify, layers 1-79, mean of 24 steps:

| phase | ms | share of the 76.6 ms step |
|---|---|---|
| attention (6-8 requests x 85k history) | 41.9 | 55% |
| attention all-reduce | 2.7 | 4% |
| MoE compute: router, shared expert, routed experts all-expert `moe_tkg` (instruction spans 5.0 and 21.9 ms, partly concurrent on different engines) | 23.1 | 30% |
| MoE all-reduce | 2.4 | 3% |
| of which waiting | ~0 | 0% |
| layer 0, embedding, lm_head, sampler, rejection | 2.0 | 3% |
| MTP draft NEFF | 4.5 | 6% |

- **Attention is the bottleneck.** 0.53 ms per layer for 6 requests x 85k of
  history, against 0.16 ms for one request: the KV read grows with the
  requests, ~1.7 GB per request per step per rank.
- **No imbalance**: with the all-expert kernel every rank does the same work
  (routed kernel 0.26 ms per layer on every sampled rank, max / mean 1.05) and
  the MoE all-reduce is as short as the attention one. The price is reading all
  3 local experts per layer (56.6 MB) whatever the routing.

## What this means for optimization

| Lever | Targets | Expected effect at 85k |
|---|---|---|
| DCP (or attention DP 8 x TP 8) together with MTP | attention: each rank reads 1/8 of the history | the largest single item at 8 requests (55% of the step) and in prefill (48%). DCP + MTP needs framework work (vllm#50391 upstream; Neuron issue drafted) |
| ~~EP=32 (6 experts per rank group)~~ | the MoE waiting at 1 request | measured: c=1 TPOT -4% (ITL 47.4 -> 44.9 ms, 2.5 of the ~10 ms waiting), c=8 +5%, prefill +13%. EP=64 stays (`results/bench/pd_split_85k_nodcp_fp8kv_mtp3_ep32/`) |
| ~~Expert load balancing (static placement by measured load)~~ | the MoE waiting | measured on captured routing, see "Expert load" below: a placement fitted on the load cuts the busiest rank's hits by ~8% (T=4) and its prefill tokens by ~9% on held-out requests, i.e. ~0.5 ms of the 47.5 ms step and ~1% of the prefill. Not worth implementing |
| ~~`HY3_DECODE_HITS_MAX_T=32` for 8-request MTP steps~~ | MoE compute at 8 requests | measured: **slower**. At 32 tokens (256 expert choices over 192 experts) a rank's 3 experts are almost always hit, so hits saves no reads but keeps its runtime loop and adds waiting: step 76.7 -> 85.7 ms, TPOT 51.0 -> 56.0 ms, 138 -> 127 tok/s (`results/bench/pd_split_85k_nodcp_fp8kv_mtp3_maxt32/`). The default (4 x (k + 1) = 16) stays |
| Block-sparse decode attention (Quest-style, keep 12.5-25% of the KV blocks) | decode attention, 55% of the 8-request step | no accuracy loss at 12.5% (needle, long continuations, LongBench v2), also reading only the kept blocks on FP8 KV + MTP; 8-request decode steps 20% faster (ITL 76.7 -> 61.3 ms), 4 3% faster, 1 request 7% slower ([SPARSE_DECODE.md](SPARSE_DECODE.md)) |
| Sparse / block-sparse prefill attention | prefill attention, quadratic in prompt length | 48% of the cold 85k prefill; needs an NKI kernel and accuracy validation |
| Fewer, fused kernels per layer (norm + router + shared expert) | per-layer fixed cost in decode | part of the 16.8 ms MoE compute and of attention's fixed cost at 1 request |

## Expert load: would a static expert placement help?

`scripts/expert_load.sh` serves the model with a `tensor_capture` of each MoE
layer's routed experts (`moe_route.<layer>` in `HYV3MoE.forward`, a no-op
without the capture config) and `scripts/expert_load_client.py` sends real
text: 112 ShareGPT prompts of 1k / 4k / 8k / 16k tokens and 40 MATH-500
problems, 128 generated tokens each (685k prefill and 19k decode tokens).
`scripts/expert_load.py` replays the captured routing on other placements of
the 192 experts over the 64 ranks; placements are fitted on half of the
requests and evaluated on the other half (`results/expert_load/analysis.md`).

| placement | prefill: busiest rank's tokens / mean | decode, 1 token | 4 tokens (1 request + MTP) | 8 tokens | 32 tokens (8 + MTP) |
|---|---|---|---|---|---|
| current (expert e on rank e // 3) | 4.76 | 1.27 | 2.07 | 2.70 | 3.00 |
| random | 4.75 | 1.27 | 2.07 | 2.70 | 3.00 |
| balanced by load | 4.35 | 1.18 | 1.91 | 2.56 | 3.00 |
| balanced by load + co-selection | 4.33 | 1.18 | 1.90 | 2.55 | 3.00 |
| mean rank (any placement) | 1 | 0.13 | 0.38 | 0.79 | 1.72 |

Decode columns: distinct local experts hit on the busiest rank per step
(mean over steps and layers; a rank holds 3).

- The popularity is skewed (the most used expert of a layer gets 8x the mean,
  the top 10% of experts 41% of the selections), but **the hot experts depend on
  the request**: within one prefill pass the busiest rank gets 4.3-4.5x the
  mean tokens even for 4k-16k chunks, and a placement fitted on all passes
  (in-sample) only brings that to 3.6-3.8x. The current linear placement is
  as good as a random one.
- Decode: in a 4-token step the busiest rank hits 2.07 experts while the mean
  rank hits 0.38; the fitted placement lowers that to 1.91. At ~0.05 ms per
  extra expert read per layer this is ~0.6 ms of the 47.5 ms step (~1%); at 32
  tokens every rank hits all its experts whatever the placement.
- Prefill: 9% fewer tokens on the busiest rank; the MoE waiting is 7% of the
  85k prefill and the routed-expert kernel time grows less than linearly with
  the tokens, so the gain is ~1% of the prefill.
- Replicating hot experts (EPLB-style redundancy) would need per-request
  choices to help, and every extra expert slot costs ~1.5 GB per rank (79
  layers x 18.9 MB), about what separates 8 from 7 concurrent 85k requests.

So the per-step waiting comes from which experts each request routes to, not
from a placement that a static or slowly updated layout can fix.

Communication itself (the same-size attention collectives) is 3-4% of every
phase measured here, and the NEFFs run back to back, so neither the
interconnect nor the host is the limit in this configuration.

## Reproduce

```bash
PYTHONPATH=$PWD MODEL=/path/to/Hy3-FP8 PROFILE_DIR=/path/nrt \
    contrib/models/hy_v3/scripts/profile_85k.sh &           # ~25 min from a warm NEFF cache
python contrib/models/hy_v3/scripts/profile_85k_client.py   # c1_cold, c1_decode, c6_decode
# per rank and session: link the NTFF and all NEFFs into one directory, then
neuron-explorer view -d <dir> --output-format parquet --output-file <out> --ignore-dma-trace --disable-ui
python contrib/models/hy_v3/scripts/profile_analyze.py <out root> <pid_rank.txt> c1_cold,c1_decode,c6_decode
```

`pid_rank.txt` maps each profiled worker's pid to its TP rank (from the server
log, `Worker_TP<r>_EP<r> pid=<pid>`); neuron-explorer keeps serving after the
export, stop it once `ExecutionInfo.parquet` is written.
