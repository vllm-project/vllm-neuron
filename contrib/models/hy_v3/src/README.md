# Hy3 (HYV3)

tencent/Hy3-FP8: ~295B-parameter MoE decoder with a static per-tensor FP8 checkpoint.

## Architecture

| Parameter           | Value |
| ------------------- | ----- |
| hidden_size         | 4096 |
| num_attention_heads | 64 |
| num_key_value_heads | 8 |
| head_dim            | 128 |
| num_hidden_layers   | 80 (+1 MTP layer, not loaded) |
| dense / MoE layers  | layer 0 dense (intermediate 13312), layers 1-79 MoE |
| experts             | 192 routed (top-8, intermediate 1536) + 1 shared |
| router              | sigmoid; `expert_bias` used for selection only; renormalized top-k x 2.826 |
| vocab_size          | 120832 |
| RoPE                | default (rotate_half), theta 11158840 |
| QK-norm             | per-head RMSNorm before RoPE |
| Activation          | SiLU-gated |
| tie_word_embeddings | False |

## Implementation notes

- All linear layers (attention, dense MLP, shared and routed experts) run with
  the checkpoint's FP8 weights. lm_head and embeddings stay BF16.
- TRN2's STATIC kernels use the 240-capped `float8_e4m3`. Weights are halved
  at load time, which is an exact exponent shift for normal values, and the
  weight scales are doubled to compensate. Activation input scales are also
  doubled so that calibrated activations don't clip at 240.
- Routed experts compute BF16 activations x FP8 weights:
  - prefill: `moe_cte` (`shard_on_i_hybrid`, per-channel scales). It runs
    one static block per local expert and skips empty dynamic blocks at
    runtime, about 3x faster than `shard_on_i` at 2048 tokens. neuronx-cc
    2.28 compiles it at EP=64 and EP=32 only; at EP=16 / 8 / 4 (I/tp = 384 /
    192 / 96) it fails (`NCC_IGCA108` or a DMA error), and
    `HY3_MOE_CTE_IMPL=shard_on_i` selects the static kernel instead.
  - decode, one token: `kernels/moe_tkg_hits.py`, forked from nkilib's
    selective `moe_tkg`. The model packs the top-k slots whose expert is
    local to this rank first and passes their count; the kernel loops over
    that many slots at runtime (`nl.fori_loop` on a register). At EP=64 a
    rank holds 3 experts and a token's 8 experts hit about 1.3 of them on the
    busiest rank, where `moe_tkg` computes all 3 (all-expert) or 8 slots
    (selective, the non-local ones on a dummy expert). `HY3_DECODE_HITS=0`
    restores `moe_tkg`.
  - decode, 2 to 4 tokens per 3 local experts (EP=64: 4, EP=32: 8):
    `kernels/moe_tkg_hits_batch.py`, from nkilib's `all_expert_impl`. The
    model packs this rank's experts that any token routed to first, with
    their affinity columns; the kernel loops over them at runtime and reads
    each once for the whole batch. `HY3_DECODE_HITS_MAX_T` sets the limit.
  - decode, more tokens: the `moe_tkg` kernel (STATIC, all-expert; selective
    when tokens x top_k < local experts, or `HY3_DECODE_SELECTIVE=1`).
    `NF.moe_tkg` only admits MX / BF16 weights, so the model calls the kernel
    wrapper directly.
- Shared expert:
  - prefill: full weights on every rank, runs on the SP-local tokens
  - decode: a second copy sharded 8 ways (I=192, padded to 256); its partial
    output is folded into the routed experts' all-reduce, so each rank reads
    1/8 of the shared weights per step with no extra collective
- The dense MLP intermediate is zero-padded per rank to a multiple of 128.
  On vllm-neuron 0.24 the CTE MLP kernel gave wrong results otherwise; on
  0.29 the unpadded shape is correct too, the padding is kept as the
  validated path.
- KV cache is BF16 only (the checkpoint has no KV scales).

## Feature Status

| Feature            | Status | Notes |
| ------------------ | ------ | ----- |
| TP (head sharding) | ✅ | validated at TP=64 |
| SP (seq parallel)  | ✅ | |
| EP                 | ✅ | required at TP=64 (`--enable-expert-parallel`); default full EP=64. `ep_degree=32` (expert TP=2) was validated on vllm-neuron 0.24 (slower than EP=64); `ep_degree` 16 / 8 failed to compile in `moe_cte` there. Not re-tested on 0.29 |
| DCP (decode context parallel) | ✅ | `--decode-context-parallel-size 8 --cp-kv-cache-interleave-size <block_size>`; 8x KV capacity at TP=64. On 0.29 validated on the 4-layer model (cos >= 0.9988 incl. prefix-cache hits); full-model performance only measured on 0.24 |
| DP / attention DP / MLP DP | ❌ | rejected at startup |
| MTP / Eagle3 spec decode | ❌ | |
| FP8 KV cache       | ❌ | |
| On-device sampling | ✅ | |
| Prefix caching (segmented prefill) | ✅ | several prefill buckets can share one `kv_segment_size` |

## Example

```bash
vllm serve tencent/Hy3-FP8 \
    --tensor-parallel-size 64 \
    --enable-expert-parallel \
    --max-model-len 4096 \
    --max-num-seqs 4

# Many long-context requests: shard the replicated KV cache 8 ways
vllm serve tencent/Hy3-FP8 \
    --tensor-parallel-size 64 \
    --enable-expert-parallel \
    --block-size 32 \
    --decode-context-parallel-size 8 \
    --cp-kv-cache-interleave-size 32 \
    --max-model-len 32768 \
    --max-num-seqs 64
```

Serving / benchmark / evaluation scripts are in `../scripts/`; results and
the test instructions are in `../README.md`.
