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
  - prefill: `moe_cte` (`shard_on_i`, per-channel scales)
  - decode: `moe_tkg` (STATIC)
- Shared expert keeps full weights on every rank:
  - prefill: runs on the SP-local tokens
  - decode: runs on all tokens
- The dense MLP intermediate is zero-padded per rank to a multiple of 128.
  The CTE MLP kernel gives wrong results otherwise.
- KV cache is BF16 only (the checkpoint has no KV scales).

## Feature Status

| Feature            | Status | Notes |
| ------------------ | ------ | ----- |
| TP (head sharding) | ✅ | validated at TP=64 |
| SP (seq parallel)  | ✅ | |
| EP                 | ✅ | required at TP=64 (`--enable-expert-parallel`); validated full EP=64 |
| DCP (decode context parallel) | ✅ | `--decode-context-parallel-size 8 --cp-kv-cache-interleave-size <block_size>`; 8x KV capacity at TP=64 |
| DP / attention DP / MLP DP | ❌ | rejected at startup |
| MTP / Eagle3 spec decode | ❌ | |
| FP8 KV cache       | ❌ | |
| On-device sampling | ✅ | |
| Prefix caching (segmented prefill) | ✅ | validated with and without DCP |

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

Test scripts, raw results and the full report are in
`examples/vllm_neuron/models/hy_v3/`.
