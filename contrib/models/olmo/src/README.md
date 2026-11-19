# OLMo (OLMo-1B-hf) on vLLM Neuron

This directory holds the `OlmoForCausalLM` implementation for the native
route (`neuron_native`). The target checkpoint is `allenai/OLMo-1B-hf`, the
original OLMo in the transformers format (not OLMo 2, which is the separate
`Olmo2ForCausalLM` architecture). The model is text only.

## Architecture

| Field | Value |
|-------|-------|
| Layers | 16 |
| Hidden size | 2048 |
| Intermediate size | 8192 (SwiGLU, SiLU) |
| Attention | MHA, 16 Q heads, 16 KV heads, head_dim 128, no bias |
| RoPE | Standard (rotate-half), theta 10000, no scaling |
| Normalization | Non-parametric LayerNorm, eps 1e-5 |
| Vocabulary | 50304 rows (the tokenizer has 50280 tokens), tied `lm_head` |
| Context | 2048 (`max_position_embeddings`) |
| Compute dtype | BF16 (the checkpoint stores FP32) |

OLMo is a Llama-style decoder. It differs from Llama in these places, and all
of them are in this folder:

- Every norm (before attention, before the MLP and the final norm) is a
  LayerNorm with **no weight and no bias**: it subtracts the mean and divides
  by the standard deviation, in FP32. The checkpoint has no norm tensors.
- The embeddings are tied. The checkpoint has no `lm_head.weight`; the loader
  reads `model.embed_tokens.weight` for the LM head.
- The HF config can clamp Q/K/V (`clip_qkv`). This checkpoint sets it to
  `null`; a checkpoint with a value is rejected at construction, because the
  fused decode kernel has no clamp step.

The checkpoint stores separate `q_proj`, `k_proj` and `v_proj` weights. The
weight loaders fuse them into one `qkv_proj_weight` per rank and transpose each
projection to `[in_features, out_features]`. Each weight is cast once from FP32
to BF16 at load.

## Parallelism

Tensor parallelism shards the heads, the MLP intermediate dimension and the
vocabulary. The model requires `num_attention_heads % TP == 0` with an **even**
head count per rank, so with 16 heads the legal sizes are TP 1, 2, 4 and 8.
TP 3 and TP 16 (one head per rank) are rejected at construction. The recorded
runs used TP 8 (2 Q heads and 2 KV heads per rank).

- Prefill uses sequence parallelism: each rank holds `T / TP` of the tokens.
  The prompt length (after bucketing) must be larger than TP and divisible by
  it.
- Decode uses the fused attention kernel (`NF.attention_decode`: QKV, RoPE,
  attention, output projection and the KV-cache write) and an all-reduce. A
  rank holds more than one KV head, so the kernel gets a per-head block table
  and per-head write slots. The torch fallback (CPU mode without the NKI
  simulator) gets the shared 2D block table instead.
- The embedding and the LM head are sharded on the vocabulary dimension.

## vLLM Neuron 0.29 notes

- `src/compat.py` vendors `hoist_rope_parameters` and
  `build_per_head_block_table`, which vLLM Neuron 0.29.0.1.2.0 does not ship.
- The decode kernel's write slots are passed as `uint32`; vLLM's slot mapping
  is `int64`, and an `int64` index makes the runtime reject the decode graph.

## Feature status

"Covered by CPU tests" means a test in `test/unit/` exercises the path.
"Implemented, not covered" means the code path is here but no test covers it.
"Not supported" means the code refuses it. Device results are in the
Validation report of the model folder's `README.md`.

| Feature | Status |
|---|---|
| BF16 compute from the FP32 checkpoint | Covered by CPU tests (tiny model, TP 2) |
| TP 8, single node | Integration test (device) |
| TP 1, 2, 4 | TP 2 covered by CPU tests; TP 1 and 4 implemented, not covered |
| Prefill (flash attention, sequence parallel) | Covered by CPU tests (tiny model, TP 2) |
| Fused decode attention kernel, per-head block table | Covered by CPU tests (NKI simulator); the torch fallback path too |
| Segmented prefill (the prefix-caching path) | Implemented, not covered — every test sets `enable_prefix_caching=False` |
| On-device greedy sampling | Covered by CPU tests |
| Paged KV cache | Covered by CPU tests |
| Context up to 2048 tokens | Integration test (device) |
| Sliding-window attention | Not supported (OLMo does not use one) |
| `inputs_embeds` | Not supported — pass `input_ids` |
| `clip_qkv`, `rope_scaling`, `attention_bias` | Not supported |
| Quantization other than BF16 (FP8, MXFP4) | Not supported |
| Speculative decoding | Code path present (`rejection_sampler` when the runner passes spec-decode metadata); not tested |
| LoRA, multimodal input, pooling / embedding tasks | Not implemented |

## Tests

The tests that ship with this model are in the folder above this one:

- `test/unit/test_loading.py`: vLLM model protocol, plugin entry point, config
  parse, factory and TP checks, checkpoint mapping coverage (from the
  safetensors header) and TP 8 weight-loader shards.
- `test/unit/test_components.py`: LayerNorm and RoPE against HF.
- `test/unit/test_tiny_e2e.py`: a random-weight tiny OLMo served through vLLM
  at TP 2 in CPU mode, compared with HF.
- `test/integration/test_model.py`: serves `allenai/OLMo-1B-hf` through vLLM
  at TP 8 and compares the raw logits with a HuggingFace golden reference.
