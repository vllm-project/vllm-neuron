# GLM-4-0414 (`Glm4ForCausalLM`) on vLLM Neuron

This directory holds the `Glm4ForCausalLM` implementation for the native
route (`neuron_native`, the default route of vLLM Neuron 0.29.0.1.2.0). The
target checkpoint is `zai-org/GLM-4-32B-0414` (`model_type: glm4`). The model
is text only.

## Architecture

| Field | Value |
|-------|-------|
| Layers | 61 |
| Hidden size | 6144 |
| Intermediate size | 23040 (SwiGLU) |
| Attention | GQA, 48 Q heads, 2 KV heads, head_dim 128, no bias |
| RoPE | Partial: the first 64 of 128 head dims, interleaved pairs, theta 10000, no scaling |
| Normalization | RMSNorm, eps 1e-5, four per decoder layer ("sandwich" norms) |
| Vocabulary | 151552 model ids (the tokenizer has 151343), untied `lm_head` |
| Max position embeddings | 32768 |
| Compute dtype | BF16 (the checkpoint stores BF16) |

GLM-4-0414 differs from Llama in three places, and all three are handled in
this folder:

- **Sandwich norms.** Each decoder layer has four RMSNorms:
  `x = x + post_self_attn_layernorm(attn(input_layernorm(x)))` and
  `x = x + post_mlp_layernorm(mlp(post_attention_layernorm(x)))`. The extra
  norms apply to the sublayer output before the residual add.
- **Partial interleaved RoPE.** HF GLM-4 rotates the pairs `(2i, 2i + 1)` of
  the first `head_dim * partial_rotary_factor` dims of each head and passes the
  rest through. The NKI prefill QKV kernel (fused RoPE) and the decode
  attention kernel use full-head "rotate half" RoPE: they rotate dim `j` with
  dim `j + head_dim / 2`. The QKV weight loader permutes the output columns of
  each Q head and each K head (`glm4_rope_permutation`), so that GLM pair
  `(2i, 2i + 1)` lands at `(i, i + head_dim / 2)`; the pass-through dims fill
  the remaining positions. `Glm4RotaryEmbedding` gives those positions a zero
  frequency (cos 1, sin 0). The same permutation applies to Q and K, so every
  Q.K product is unchanged. V is not permuted, so the attention output and
  `o_proj` keep the checkpoint layout. The KV cache holds K in the permuted
  layout, for prefill and for decode.
- **Fused gate/up.** The checkpoint stores `mlp.gate_up_proj.weight` as one
  `[2 * intermediate, hidden]` tensor (gate rows first). The loaders split it
  into the separate gate and up parameters of `NF.mlp`.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Exports the factory `Glm4ForCausalLM` and `Glm4Config`. |
| `plugin.py` | `register()` for the `vllm.general_plugins` entry point. |
| `factory.py` | The class registered with vLLM. Validates the config, then builds `model.Glm4ForCausalLM`. |
| `config.py` | `Glm4Config` dataclass parsed from the HF config (reads `rope_parameters` from transformers 5). |
| `model.py` | RMSNorm, RoPE layout, attention, MLP, decoder layer, backbone, LM head, KV spec, weight loading. |
| `compat.py` | `hoist_rope_parameters`, which vLLM Neuron 0.29.0.1.2.0 does not ship. |

## Kernel paths

| Phase | Component | API |
|---|---|---|
| Prefill | QKV projection and RoPE | `NF.qkv_proj` (fused RoPE) |
| Prefill | Attention | `NF.flash_attention`, or `NF.segmented_attention` for chunked prefill |
| Prefill | Output projection | `NF.o_proj` |
| Prefill and decode | MLP | `NF.mlp` (SiLU) |
| Decode | QKV, RoPE, cache update, attention and output projection | `NF.attention_decode` |

## Parallelism

Tensor parallelism shards the query heads, the MLP intermediate dimension and
the vocabulary. The TP size must divide the attention head count (48), the
intermediate size (23040) and the vocabulary (151552), and with TP >= 2 it
must be a multiple of the KV head count (2). For GLM-4-32B-0414 that leaves
TP 2, 4, 8 and 16; the integration test and the serve example use TP 8.

- With TP >= 2 every rank holds one KV head. At TP 8 each rank has 6 query
  heads, and each of the 2 KV heads is replicated on 4 ranks.
- Prefill uses sequence parallelism: each rank holds `T / TP` of the tokens,
  so the prefill bucket must be larger than TP and divisible by it. The model
  raises a clear error otherwise.
- Prefill attention is `NF.flash_attention` when the prompt fits in one step
  (`max_num_batched_tokens >= max_model_len`), and `NF.segmented_attention`
  over the paged cache otherwise (chunked prefill). In the segmented path the
  QKV kernel writes the post-RoPE K/V into the cache. The runner picks the
  path through `kv_segment_size` in the attention metadata; vLLM Neuron
  0.29.0.1.2.0 sets it by default only when `max_num_batched_tokens` is a
  supported segment size (512, 1024, 2048, 4096 or 8192), so keep the
  prefill chunk at one of those sizes.
- Decode uses the fused `NF.attention_decode` kernel with the in-kernel KV
  cache update (`kv_cache_update_idx` as `uint32`), then a TP all-reduce.
- The embedding and the LM head are sharded on the vocabulary dimension.

## Feature status

The statuses come from the code in this folder and from what the tests in
`test/` run. "Device integration test" means that
`test/integration/test_model.py` runs the feature on Neuron (TP 8, a
9216-token model length, two prompts of 8000 to 8192 tokens, 16 validated
tokens each, batch 1 and 4); its results are in the Validation report of the
model folder's `README.md`.

| Feature | Status |
|---|---|
| BF16 checkpoint and compute | Device integration test |
| TP 8, single node | Device integration test |
| TP 2, 4, 16 | Implemented, not validated on device. CPU mode covers TP 4 with a tiny random-weight model and TP 2 with one tiny decoder layer. |
| TP 1 | Not covered. Both KV heads land on one rank; the decode call passes a 2D block table, which `NF.attention_decode` routes to its torch fallback for more than one KV head per rank. |
| Segmented (chunked) prefill, 1024-token chunks | Device integration test: 8 chunks per prompt, each chunk after the first attends to the earlier chunks in the paged KV cache. CPU mode also covers a 600-token prompt in two chunks of at most 512 tokens with a tiny model. |
| Flash-attention (single-step) prefill | Implemented, not validated on device: the device configuration sets `max_num_batched_tokens` 1024 below `max_model_len` 9216. CPU mode covers it with a tiny model. |
| Fused decode attention kernel | Device integration test (16 decode steps after each 8k-token prompt) |
| On-device greedy sampling | Device integration test |
| Context up to 9216 tokens | Device integration test |
| Context beyond 9216 tokens (up to 32768) | Implemented, not validated |
| Decode batch sizes 1 and 4 | Device integration test; other batch sizes not validated |
| Prefix caching | Implemented through the segmented-prefill path, not validated (the tests and the serve example disable it) |
| `inputs_embeds` | Not supported, pass `input_ids` |
| Attention bias | Rejected by the factory (GLM-4-0414 checkpoints have none) |
| Quantized checkpoints, non-BF16 checkpoints | Rejected by the factory (BF16 only) |
| Speculative decoding | Code path present (`rejection_sampler`), not validated |
| Disaggregated prefill and decode (KV transfer) | Prefill and decode use the same KV layout; not validated |
| LoRA, multimodal input, pooling | Not implemented |
| Trn3 | Not tested |

## Tests

The tests are in the folder above this one:

- `test/unit/test_glm4_cpu.py`: CPU-mode registration contract, config parse,
  factory checks, the RoPE layout against HF, and a tiny decoder layer at TP 2
  (prefill and decode through the KV cache) against the HF `Glm4DecoderLayer`.
- `test/unit/test_glm4_weight_loading.py`: the weight mapping against the
  checkpoint index and (with the full checkpoint) the TP 8 rank-7 loaders
  against independent slices of the checkpoint tensors.
- `test/unit/test_tiny_glm4_e2e.py`: the whole vLLM pipeline in CPU mode at
  TP 4 on a random-weight tiny GLM-4, against HF.
- `test/integration/test_model.py`: serves the model through vLLM on Neuron at
  TP 8 with a 9216-token model length, builds two prompts of 8000 to 8192
  tokens, and compares the raw logits of 16 generated tokens with a
  HuggingFace golden reference at batch 1 and batch 4.

All of them reach this directory through `test/_local_src.py`, which loads it
by file location under the module name `contrib_glm4_src`.
