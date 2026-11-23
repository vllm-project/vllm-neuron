# HyperCLOVAX-SEED-Think-14B

Native (`neuron_native`) port of `naver-hyperclovax/HyperCLOVAX-SEED-Think-14B`
(`HyperCLOVAXForCausalLM`). The model is text-only. The checkpoint stores fp32
weights. The loader casts each weight to bf16.

## Architecture

| Parameter              | Value |
|------------------------|-------|
| hidden_size            | 6144 |
| intermediate_size      | 14336 |
| num_hidden_layers      | 38 |
| num_attention_heads    | 48 |
| num_key_value_heads    | 8 |
| head_dim               | 128 |
| vocab_size             | 110592 |
| rope_theta             | 1e8 |
| attention_multiplier   | 0.0078125 (the softmax scale) |
| embedding_multiplier   | 10.0 |
| logits_scaling         | 0.125 |
| residual_multiplier    | 1.0 |
| use_post_norm          | true |
| tie_word_embeddings    | false |
| Activation             | SiLU (SwiGLU MLP) |
| Normalization          | RMSNorm |

## Key Differences from Llama3

- **Peri-layer normalization.** `post_norm1` normalizes the attention output,
  and `post_norm2` normalizes the MLP output. Each normalization occurs before
  the residual add.
- **muP multipliers.** The model multiplies the embedding output by
  `embedding_multiplier` and the logits by `logits_scaling`. It multiplies each
  residual branch by `residual_multiplier`. The attention softmax scale is
  `attention_multiplier`, not `1 / sqrt(head_dim)`.
- **fp32 checkpoint.** The loader casts the weights to bf16.

## Kernel Paths

| Phase | Component | API |
|-------|-----------|-----|
| Prefill | QKV projection with fused RoPE | `NF.qkv_proj` |
| Prefill | Attention (full) | `NF.flash_attention` |
| Prefill | Attention (chunked) | `NF.segmented_attention` |
| Prefill | Output projection | `o_proj` (`src/o_proj.py`, the plugin's `NF.o_proj` with an fp32-input torch fallback) |
| Decode | QKV, RoPE, attention, cache update and output projection | `NF.attention_decode` |
| Prefill and decode | SwiGLU MLP | `NF.mlp` |

Norms, the embedding, the full-prefill KV cache write, the decode mask and the
`lm_head` use torch.

## Parallelism

- Tensor parallelism shards the Q heads, the KV heads and the MLP intermediate
  dimension. The TP size must divide 8 (the KV head count) and 14336.
- Prefill uses sequence parallelism: all-gather before attention and MLP, and
  reduce-scatter after them. Decode uses all-reduce.
- The attention TP reduction adds the partial sums in fp32. In prefill, the O
  projection kernel gets an fp32 input and writes an fp32 output. In decode,
  the code casts the bf16 output of the attention kernel to fp32 before the
  all-reduce. The softmax scale is 1/128, so the attention output is small. A
  bf16 sum of the partials then adds more error than the base bf16 error.

## Tests

- `test/unit/test_factory.py`, `test/unit/test_checkpoint.py` and
  `test/unit/test_tiny_hyperclovax_e2e.py`: CPU-mode tests.
- `test/integration/test_model.py`: TP8 logit validation on one Trn2 node.

See the folder's top-level `README.md` for the commands.

## Feature Status

| Feature | Status |
|---------|--------|
| BF16 | Supported |
| Prefix caching | Not validated |
| Quantization | Not supported |
| Multimodal | Not applicable (text-only model) |
