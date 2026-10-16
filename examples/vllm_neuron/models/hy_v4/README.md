# Tencent Hy4-preview on Neuron

[`tencent/Hy4-preview`](https://huggingface.co/tencent/Hy4-preview)
(`HYV4ForCausalLM`, 770B total / 49B active) on one trn2.48xlarge, text only.

| | |
|---|---|
| layers | 78: layer 0 dense FFN (18432), 77 MoE |
| MoE | 256 routed experts of 2048, top-8, sigmoid + correction-bias routing, x2.827, SwiGLU clamp 10, 1 shared expert |
| attention | gated MLA, 64 heads, q_lora 2048, kv_lora 512, qk 192+64, v 256, learnable sink per head |
| sparse attention | DSA, top-2048 keys; 20 "full" layers run the indexer, 58 "shared" ones reuse it |
| residual | iHC, 4 streams |
| vocab / hidden | 120832 / 6144 |

## Quick start

```bash
# once: MXFP8 -> Trn2 fp8 (~7 min with 24 workers on a trn2.48xlarge)
python examples/vllm_neuron/models/hy_v4/convert_mxfp8_to_neuron_fp8.py \
    --src <dir>/Hy4-preview-FP8 --dst <dir>/Hy4-preview-neuron-fp8 --workers 24

python examples/vllm_neuron/models/hy_v4/run.py --model <dir>/Hy4-preview-neuron-fp8

# long context (max_model_len 32768, prefill in 2048-token chunks): a pass key
# hidden at the start of ~30K tokens of filler
python examples/vllm_neuron/models/hy_v4/run.py --model <dir>/Hy4-preview-neuron-fp8 \
    --needle 30000
```

Cold start compiles one prefill graph and one decode graph per decode bucket;
at 32K context the prefill one takes **~17 min** at `-O1`, and a cold start
with both decode buckets ~40 min. `run.py` raises vLLM's `distributed_timeout_seconds` (and the CPU
group's) to 4 h because the other 63 ranks wait for rank 0's compile in a gloo
barrier whose default timeout is 3600 s -- an intermediate build's prefill graph
took 85 min, and every rank died with `RuntimeError: Operation timed out!` at
the hour mark.
Set `NEURON_RT_EXEC_TIMEOUT=1200` too when trying graph changes: earlier builds
took 55-150 s per 32K prefill chunk, over the runtime's default execution
timeout, which surfaces as `TOPSP missing collectives status` / `NRT_TIMEOUT`. While
a chunk runs, `neuron-monitor` shows ~0% core utilisation and every worker sits
idle in vLLM's message queue -- it looks hung and is not; count completed
executions over a few minutes before concluding otherwise.

## Environment

The plugin's `pip install -e .` plus, separately, the compiler packages, with
**`islpy` pinned**:

```bash
pip install --extra-index-url https://pip.repos.neuron.amazonaws.com \
    "neuronx-cc==2.27.*" "nki==0.6.0" "islpy==2026.1"
```

`neuronx-cc` 2.27 asks for `islpy~=2026.1`, which pip resolves to 2026.2.x; with
that every compile dies in the Tensorizer with `[NCC_ISMP902] Simplifier error:
is_subset(): incompatible function arguments`. The import-time warning about
`libtorch_neuronx_lite/_compiler/_C.so: undefined symbol
PyObject_ClearManagedDict` (its vendored compiler is built for cp313) is
harmless on Python 3.12.

The compile cache key does not cover this model's source, so after editing
`vllm_neuron/model/hy_v4/` clear `$VLLM_CACHE_ROOT` (or point it elsewhere) --
otherwise the previous trace is compiled again.

## Measured

**32K context** (current code): trn2.48xlarge, world 64, `max_model_len` 32768,
2048-token prefill chunks, 4 sequences, greedy, `--needle`, 32 output tokens:

| prompt | prefill chunks | output | wall time |
|---|---|---|---|
| 1,732 tokens | 1 | `' 71432. I have memorized it.\n\nThe pass key is **71432**.\n\nI have memorized it.\n\nThe pass key is **'` | 43 s |
| 28,804 tokens | 15 | `' 71432. I have memorized it.\n\nThe pass key is **71432**. I have memorized it.\n\nThe pass key is **714'` | 129 s |

KV cache 40,384 tokens at `--gpu-memory-utilization 0.8` (704 values x 78 layers
per token, replicated on every rank); weights 15.08 GiB of each core's 24.

Throughput at a 31,744-token random prompt, 256 output tokens, greedy, median
of 3 (each run within 0.1%). Only one 32K request fits the KV cache, so this
single stream is also the maximum at this length:

| decode buckets | TTFT | input tok/s | TPOT | output tok/s |
|---|---|---|---|---|
| `[4]` | 97.8 s | 325 | 1197 ms | 0.84 |
| `[1, 4]` (`run.py` default) | 97.8 s | 325 | 604 ms | 1.66 |

Decode is attention-bound: per layer at a 32K span, a full (indexer) layer
takes 16.7 ms and a shared layer 3.7 ms at batch 1 (20.7 / 10.0 ms at batch 4).

**2K context**, the first, dense-attention build (before DSA and chunked
prefill; `run.py`'s demo prompts):

```
'The capital of France is'  -> ' __________.\nA. Sydney\nB. New York\nC. Paris\nD. Ottawa 【解析】句意：法国的首都是巴黎。Sydney'
'I am gonna keep counting forever, 1 2 3 4 5' -> ' 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21'
'def fibonacci(n):'         -> '\n    """Return the nth value in the Fibonacci series."""\n    if n == 0:\n        return 0\n    elif n == 1:\n'
'Once upon a time, there was a' -> ' king who was very rich. He had a lot of gold and silver, but he was not happy. He wanted to be the happiest man in the world'
```

Weights load in ~55 s per rank.

## Why the fp8 checkpoint, and why converted

BF16 is 1.56 TB; a trn2.48xlarge has 64 x 24 GB = 1.5 TB of HBM, partitioned per
logical core. The published `Hy4-preview-FP8` is ModelOpt **MXFP8** (one e8m0
exponent per 32 elements), which has no Trn2 kernel, and its values use the OCP
e4m3fn range (448) where Trn2's fp8 tops out at 240.
`convert_mxfp8_to_neuron_fp8.py` therefore:

* requantises the **routed experts** to fp8 with one scale per output row, inside
  the 240 range (`down_proj` also stored transposed as `down_proj_t [E, I, H]`
  so a TP shard is a contiguous read);
* dequantises **everything else** to bf16, which is lossless (e4m3 x 2^k fits
  bf16 exactly);
* drops `quantization_config` from `config.json` and adds
  `neuron_quantization`, which the model checks for.

Requantisation error, MX -> per-row, measured by the converter over 156 expert
tensors: **median 2.56%, max 2.70%** relative (Frobenius). That is on top of the
MXFP8 quantisation itself.

Per rank at world 64: ~11.6 GiB of fp8 experts + ~3.2 GiB bf16 (q_a/kv_a are
replicated) ~= 15 GiB.

## Layout

**Pure TP over 64 ranks.** One attention head per rank. Every rank holds all
256 experts and 32 of each one's 2048 intermediate columns. The MoE runs in
torch, dense over the 256 local experts with zero affinity on the unselected
ones -- exact, one prefill and one decode graph, and independent of whether
Trn2's MoE kernels accept fp8 weights (`NF.moe_tkg` rejects `float8` today).
The kernel path (`moe_cte` shard_on_block FP8 / `moe_tkg`) needs
`2048 / tp_degree` to be a multiple of 128, so `ep_degree >= 4`; not done yet.

**MLA in latent space, with the DSA key alongside.** Per token the cache holds
the normed 512-wide latent, the 64-wide rope key and the 128-wide indexer key --
704 values, stored as the K and V halves of one 352-wide head so a page is
exactly what the planner reserved; layers with a shared indexer leave the last
128 zero. `kv_b_proj` is absorbed into the query and the output. Every rank holds
a full copy, which at one head per rank is ~10% *more* than an expanded per-head
cache would be; sharding the latent's tokens across ranks is what would make
MLA's compression pay here.

**DSA.** A full layer scores every visible key with
`sum_h w_h * relu(q_h . k)` (32 heads of 128, LayerNorm'd key, rope on the last 64
dims) and keeps the top 2048 per query; the next shared layers reuse that mask.
The top-k is a per-row threshold found by bisection -- elementwise passes only,
no k=2048 sort and no scatter -- and is exact up to ties with the 2048th score.
In prefill the scoring and selection are **split by query** across the 64 ranks
and the mask all-gathered: done replicated, they were ~90% of the 32K prefill
graph and pushed it to 15.7M instructions, over neuronx-cc's 10M limit. The
attention itself is dense attention masked to the selection: exact, but its cost
still grows with the context. Prefill runs it over the span in 2048-key blocks
with an online softmax, so the `[chunk, span]` probabilities are never live at
once (that, plus 128-aligned slices of the cache entry, took the 28.8K needle
from 825 s to 129 s).

**Chunked prefill** is the plugin's segmented prefill: one chunk of one request
per step, attending to the `cached_seq_len` tokens already in the cache plus
causally to itself. `max_num_batched_tokens` (`--prefill-chunk`, default 2048)
must be one of 512/1024/2048/4096/8192.

## Known limitations

* **Cost grows with the cache span, not the prompt.** Every prefill chunk
  attends to (and every full layer's indexer scores) the whole 32K span, because
  the graph is static: ~6 s per 2048-token chunk whatever the prefix. Decode
  likewise reads the whole span (604 ms per token). Bucketing the prefix length,
  and a real sparse gather instead of a masked dense attention, are next.
* **Zeroing the gathered context is a 2.8x speed matter, not a correctness
  one.** Keys beyond the prefix are masked out of the scores either way, but a
  build that left the context unmasked ran ~150 s per chunk (2306 s for the
  same needle) with a prefill graph twice the size (596K vs 284K instructions
  per subgraph) and an 85 min compile. Why is not understood.
* **Context is 32K, not 1M.** The latent cache is replicated on every rank, so
  KV capacity (40K tokens at gmu 0.8) is per rank, not per instance.
* **A neuronx-cc miscompile is worked around, not fixed.** In decode, masking
  the gathered `[rows, 32K, 704]` context with `in_prefix` (by `where` or by a
  multiply) made the device compute `in_prefix` *itself* wrongly wherever it was
  reused -- extra keys in 128-key blocks past the prefix, even on a dead row at
  position 0 -- so every decode at 32K attended to the wrong keys while the first
  token (from prefill) was right. The same ops compiled alone, and the CPU, are
  correct. `_zero_beyond` masks the context with a *separately computed*
  predicate (a float subtraction), which compiles correctly; checked on device
  in a standalone graph of the decode attention chain before it went in.
* **RoPE pairing: interleaved, and transformers disagrees.** vLLM's and SGLang's
  Hy4 ports rotate interleaved pairs; transformers' `hy_v4` rotates halves.
  Measured on device with the same weights, `HY4_ROPE_STYLE=neox` degrades
  within a few tokens -- `'... 1 2 3 4 5'` continues `' 6 7 8 9 10 13 20 5 6 8
  ...'`, `def fibonacci(n):` comes out broken -- while interleaved (the default)
  gives the outputs above. So the checkpoint wants interleaved. The CPU check
  below still runs `neox`, because it compares against transformers.
* No MTP (the `mtp_layers.0` tensors are ignored), no prefix caching, no mixed
  prefill/decode batches (the plugin runs one prefill request per step).
* vLLM 0.24 has no class for `HYV4ForCausalLM`, so the plugin registers it from
  a `vllm.general_plugins` entry point (`vllm_neuron:register_models`); reinstall
  with `pip install -e .` so that entry point exists.
* `HY4_DEBUG_CAPTURE=1` (with `HY4_DEBUG_LAYERS=0,1,...` and `run.py
  --capture-dir`) traces `capture_tensor` calls for those layers' attention and
  MLP into the graphs, for device-vs-CPU comparison. Off, nothing is traced.

## Correctness check (CPU, one layer vs transformers)

```bash
HY4_ROPE_STYLE=neox VLLM_NEURON_CPU_MODE=1 python examples/vllm_neuron/models/hy_v4/check_layer_vs_hf.py \
    --model <dir>/Hy4-preview-neuron-fp8 --layer 1
```

Loads one layer through this port's weight loaders, runs a 2600-token prefill in
2048-token chunks (the second reads its prefix back from the paged cache, and
its rows past 2048 attend to a DSA selection) and one paged decode step, and
compares the layer's update against `transformers.models.hy_v4.HYV4DecoderLayer`
in fp32 on all 2601 tokens at once, on the same (dequantised) weights. It also
checks that the query-split selection, emulated over 4 ranks with the
all-gather returning blocks in a shuffled order, equals the unsplit one, and how
many of the decode row's 2048 selected keys match transformers' indexer.

Layers 0 and 1, current code: chunk 1 / later chunks / decode per-token median
0.39-0.47% / 0.43-0.47% / 0.45-0.61%; 2044-2045 of 2048 selected keys agree
with transformers (the rest are ties at the boundary). The table below is the
dense build's, 40 tokens:

| layer | kind | prefill, per-token median | decode |
|---|---|---|---|
| 0 | dense, full indexer | 0.43% | 0.41% |
| 1 | MoE, full indexer | 0.42% (tokens 26, 32 flip an expert) | 0.40% |
| 2 | MoE, shared indexer | 0.42% | 0.42% |
| 5 | MoE, full indexer | 0.49% (token 23 flips) | 0.50% |
| 77 | MoE, full indexer | 0.66% (tokens 1, 29 flip) | 0.84% |

The attention sub-block alone agrees to 0.5%. The few outlier tokens appear
only in MoE layers, only in prefill, and at 20-45% rather than drifting, which
is what an 8th/9th-expert near-tie resolving differently in bf16 and fp32 looks
like; that explanation is inferred, not yet checked expert by expert.
