# GLM-4 (glm-4-9b-chat-hf) on vLLM Neuron

This directory holds the `GlmForCausalLM` implementation for the native
route (`neuron_native`, the default route of vLLM Neuron 0.29.0.1.2.0). The
target checkpoint is `zai-org/glm-4-9b-chat-hf`, the transformers-format
release of GLM-4-9B-Chat (`model_type: glm`). The model is text only.

## Architecture

| Field | Value |
|-------|-------|
| Layers | 40 |
| Hidden size | 4096 |
| Intermediate size | 13696 (SwiGLU) |
| Attention | GQA, 32 Q heads, 2 KV heads, head_dim 128 |
| Attention bias | Q, K and V have a bias; the output projection has none |
| RoPE | Partial: the first 64 of 128 head dims, interleaved pairs, theta 10000, no scaling |
| Normalization | RMSNorm, eps 1.5625e-7, variance in FP32 |
| Vocabulary | 151552 model ids (the tokenizer has 151343), untied `lm_head` |
| Max position embeddings | 131072 |
| Compute dtype | BF16 (the checkpoint stores BF16) |

GLM differs from Llama in three places, and all three are handled in this
folder:

- **QKV bias.** The checkpoint stores separate `q_proj`, `k_proj` and `v_proj`
  weights and biases. `weight_loaders.glm_fused_qkv_loader` fuses them into one
  `qkv_proj_weight` (`[hidden, q + 2 * kv]`) and one `qkv_proj_bias`
  (`[1, q + 2 * kv]`) per rank. Prefill passes the bias to `NF.qkv_proj` and
  decode passes it to `NF.attention_decode` (`bias_qkv`), so both add it before
  RoPE, as HF GLM does.
- **Partial interleaved RoPE.** HF GLM rotates the pairs `(2i, 2i+1)` of the
  first `head_dim * partial_rotary_factor` dims of each head and passes the
  rest through. The Neuron kernels use the contiguous ("rotate half") layout
  over the full head. The loader permutes the rows of each Q and K head (weight
  and bias) at load time, so that source dim `2i` lands at `i` and `2i+1` at
  `i + head_dim / 2`; the pass-through dims fill the remaining positions.
  `GlmRotaryEmbedding` gives those positions a zero frequency (cos 1, sin 0).
  The same permutation applies to Q and K, so every Q.K product is unchanged.
  The KV cache holds K in the permuted layout.
- **Fused gate/up.** The checkpoint stores `mlp.gate_up_proj.weight` as one
  `[2 * intermediate, hidden]` tensor (gate rows first). The loaders split it
  into the separate gate and up parameters of `NF.mlp`.

`GlmRotaryEmbedding` computes cos/sin inside the compiled graph with
out-of-place operations only. An in-place write into a fresh frequency tensor
gave non-finite logits on the device during the port, and
`test_glm_rotary_graph_has_no_in_place_write` keeps that out.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Exports the factory `GlmForCausalLM` and `GlmConfig`. |
| `plugin.py` | `register()` for the `vllm.general_plugins` entry point. |
| `factory.py` | The class registered with vLLM. Validates the config, then builds `model.GlmForCausalLM`. |
| `config.py` | `GlmConfig` dataclass parsed from the HF config (reads `rope_parameters` and `dtype` from transformers 5). |
| `model.py` | RMSNorm, RoPE, attention, MLP, decoder layer, backbone, LM head, KV spec, weight loading. |
| `weight_loaders.py` | The RoPE row permutation, the fused QKV loader and the gate/up loader. |
| `compat.py` | `hoist_rope_parameters`, which vLLM Neuron 0.29.0.1.2.0 does not ship. |

## Parallelism

Tensor parallelism shards the query heads, the MLP intermediate dimension and
the vocabulary.

- With TP >= 2 every rank holds one KV head. At TP 16 each rank has 2 query
  heads, and each of the 2 KV heads is replicated on 8 ranks.
- Prefill uses sequence parallelism: each rank holds `T / TP` of the tokens,
  so the prefill bucket must be larger than TP and divisible by it. The model
  raises a clear error otherwise.
- Prefill attention is `NF.flash_attention` when the prompt fits in one step
  (`max_num_batched_tokens >= max_model_len`), and `NF.segmented_attention`
  over the paged cache otherwise (chunked prefill). Unused block columns of a
  short prompt point at the row's first block, so the kernel never reads a
  stale tile.
- Decode uses the fused `NF.attention_decode` kernel with the in-kernel KV
  cache update (`kv_cache_update_idx` as `uint32`), then a TP all-reduce.
- The embedding and the LM head are sharded on the vocabulary dimension.

## Feature status

The statuses come from the code in this folder and from the recorded test
verdicts in the Validation report of the model folder's `README.md`.
"Exercised" means a recorded run with a PASS verdict covered the path.

| Feature | Status |
|---|---|
| BF16 checkpoint and compute | Exercised |
| TP 16 | Exercised (single node; see the Validation report) |
| TP 2, 4, 8 | Implemented, not covered on device. CPU mode covers TP 4 with a tiny random-weight model. |
| TP 1 | Not covered. Both KV heads land on one rank; the decode call passes a 2D block table, which `NF.attention_decode` routes to its torch fallback for more than one KV head per rank. |
| Segmented (chunked) prefill, prefill bucket 1024 | Exercised |
| Flash-attention prefill | Implemented, not covered on device: the device runs set `max_num_batched_tokens` 1024 below `max_model_len` 2048, so prefill ran segmented attention |
| Fused decode attention kernel with QKV bias | Exercised |
| On-device greedy sampling | Exercised |
| Prefix caching | Implemented through the segmented-prefill path, not covered (every recorded run disabled it) |
| Context beyond 2048 tokens (up to 131072) | Implemented, not covered |
| `inputs_embeds` | Not supported, pass `input_ids` |
| `rope_scaling` other than default | Rejected by the factory |
| Activations other than SiLU | Rejected by the factory |
| Quantized checkpoints, FP32 checkpoints | Rejected by the factory (BF16 only) |
| Speculative decoding | Code path present (`rejection_sampler`), not covered |
| LoRA, multimodal input, pooling | Not implemented |
| Trn3 | Not tested |

## Tests

The tests are in the folder above this one:

- `test/unit/test_glm_cpu.py`: CPU-mode registration contract, config parse,
  factory checks, RoPE / RMSNorm against HF, the TP 16 loaders, the weight
  mapping against the checkpoint index, the padded-prefill KV paths, and (with
  the full checkpoint) a 40-layer TP 16 CPU load with prefill logits compared
  three-way against HF.
- `test/unit/test_tiny_glm_e2e.py`: the whole vLLM pipeline in CPU mode at
  TP 4 on a random-weight tiny GLM, against HF.
- `test/integration/test_model.py`: serves the model through vLLM on Neuron at
  TP 16 and compares the raw logits with a HuggingFace golden reference.

All of them reach this directory through `test/_local_src.py`, which loads it
by file location under the module name `contrib_glm_src`.
