# ModernBERT

This directory holds the native (`neuron_native`) implementation of ModernBERT.
The reference checkpoint is `ibm-granite/granite-embedding-small-english-r2`
(revision `2ab6fa8ea2d674564defd37171ae19079b864b33`).

ModernBERT is a bidirectional text encoder. This port is the pooling
(embedding) model: it has no LM head, no sampler and no decode phase. It
returns CLS-pooled embeddings through vLLM's pooling runner.

## Architecture

| Item | Value |
|---|---|
| Registry name | `ModernBertModel` (the `architectures` field of the embedding checkpoints) |
| Implementation class | `ModernBertForEmbedding` (backbone plus vLLM's `DispatchPooler`) |
| Layers | 12 for the reference checkpoint |
| Attention | Bidirectional MHA, 12 heads, head dimension 32 (hidden 384) |
| Layer types | `full_attention` every 3rd layer (0, 3, 6, 9), `sliding_attention` otherwise |
| Sliding window | `local_attention / 2` on each side (64 for the reference checkpoint) |
| Position embedding | RoPE with a theta per layer type: `global_rope_theta` for full layers, `local_rope_theta` for sliding layers |
| Normalization | LayerNorm without bias |
| MLP | GeGLU; the checkpoint fuses the input and gate halves in `Wi` |
| Intermediate size | 1536 for the reference checkpoint |
| Pooling | CLS, then L2 normalization (the model card's pooling) |
| Dtype | BF16 |

## Differences from a decoder port

- Bidirectional attention. Each query gets a `[min, max)` KV bound instead of a
  causal mask: `modernbert_attention_bounds` derives the bound of each row from
  the flat positions, so sequences packed in one flat batch never attend across
  each other and a padding token forms its own sequence.
- No KV cache use. `get_kv_spec` and `bind_kv_cache` exist because the runner
  calls them on every model, but the encoder never reads or writes the cache.
- No generation phase. There is one forward per request and no sampler.
- The forward pads the flat token count to the kernel row tile; a padding token
  has position 0, so it does not join a real sequence.

## Feature Status

| Feature | Status |
|---|---|
| Tensor parallelism | TP 2 is the degree that the recorded tests ran (6 heads and 768 MLP channels per rank). The code shards attention heads and MLP channels and all-reduces after `o_proj` and the MLP; it rejects a TP degree that does not divide the head count. Other degrees are untested. |
| Multi-node TP | TP 2 as one rank on each of 2 nodes is the topology the recorded tests ran. |
| Sequence packing | Supported. More than one sequence in one flat batch, with a padding tail, is covered by the CPU and device tests. |
| Pooling tasks | Embedding only. `default_seq_pooling_type` is CLS. The plugin's classify assertion is not implemented, and this checkpoint has no classifier head. |
| Runner | Pooling only. The model raises if `runner_type` is not `pooling`. |
| Quantization | Not supported. The factory accepts `None` or `"bf16"` and rejects a checkpoint with a `quantization_config`. |
| FP8 KV cache | Rejected by the model: the encoder does not use the KV cache. |
| Prefix caching | Rejected by the model: the attention is bidirectional. |
| Chunked prefill | Rejected by the model unless `max_num_batched_tokens >= max_model_len`, for the same reason. |
| Activation | `gelu` (exact, erf) only. The config rejects any other `hidden_activation`. |
| Bias | Not supported. The config rejects `norm_bias`, `attention_bias` or `mlp_bias`. |
| RoPE scaling | Not supported. The config accepts the `default` RoPE type only. |
| Logit validation | Not applicable. The model produces no logits; the accuracy gate is a cosine floor on the embeddings. |

## Kernel Paths

Each main path goes through an `NF` API, which uses its NKI kernel when its
eligibility check passes and its torch fallback otherwise. In CPU mode
(`VLLM_NEURON_CPU_MODE=1`) `can_run_kernel` is false, so every API takes the
torch fallback.

| Operation | NF API |
|---|---|
| Token embedding | `NF.embedding`, with `NF.merge_prompt_embeds` for passed-in embeddings |
| QKV projection with RoPE | `NF.qkv_proj` |
| Bidirectional attention with per-row KV bounds | `NF.flash_attention` |
| Output projection | `NF.o_proj` |
| GeGLU MLP | `NF.mlp` |

These parts stay in torch: the LayerNorm, the residual adds, the row-tile
padding and the per-row attention bounds.
