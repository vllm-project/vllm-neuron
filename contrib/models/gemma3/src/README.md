# Gemma3 text (rnj-1-instruct) on vLLM Neuron

This directory holds the `Gemma3ForCausalLM` implementation for the native
route (`neuron_native`, the default route of vLLM Neuron 0.29). The target
checkpoint is `EssentialAI/rnj-1-instruct` (`model_type` `gemma3_text`). The
model is text only.

## Architecture

| Field | Value |
|-------|-------|
| Layers | 32, full attention in every layer |
| Hidden size | 4096 |
| Intermediate size | 16384 (gated, `gelu_pytorch_tanh`) |
| Attention | GQA, 32 Q heads, 8 KV heads, head_dim 128 |
| Softmax scale | `query_pre_attn_scalar ** -0.5` = `128 ** -0.5` |
| QK-norm | Per-head RMSNorm on Q and K, before RoPE |
| RoPE | YaRN, theta 10000, factor 4, original context 8192, beta_fast 64 |
| Normalization | Gemma RMSNorm (`norm(x) * (1 + w)` in FP32), eps 1e-6, four per layer |
| Embedding | Scaled by `sqrt(hidden_size)` = 64 |
| Logits | Final soft cap `tanh(x / 30) * 30` |
| Vocabulary | 128256, LM head tied to the embedding |
| Checkpoint dtype | FP32; served in BF16 |

What differs from the plugin's Qwen3 dense model, which this implementation
started from:

- Every RMSNorm multiplies by `1 + w` (the checkpoint stores zero-centered
  weights). The torch norms do it in FP32; the NKI kernels (the QK-norm in
  the QKV and decode-attention kernels) get the gamma `1 + w` in BF16.
- Each decoder layer has four norms. `post_attention_layernorm` and
  `post_feedforward_layernorm` apply to the sublayer output, before the
  residual add.
- The token embeddings are multiplied by `sqrt(hidden_size)`, rounded to BF16
  once, as the HF scaled embedding does.
- The MLP activation is the tanh approximation of GELU
  (`ActFnType.GELU_Tanh_Approx`).
- YaRN RoPE. The table is built on the CPU in FP32 from the same formula as
  transformers `_compute_yarn_parameters`, and cos/sin include the YaRN
  attention factor (`0.1 * ln(4) + 1`). The checkpoint's `attn_factor` key is
  not a key transformers reads, so it is ignored here too.
- The logits get the final soft cap, on the vocabulary-sharded logits before
  sampling and before the logit gather.
- The LM head is a separate vocabulary-sharded parameter that the weight load
  fills from `model.embed_tokens.weight`.
- The weight load casts each FP32 checkpoint tensor to the served dtype.

The checkpoint stores separate `q_proj`, `k_proj` and `v_proj` weights. The
weight loaders fuse them into one `qkv_proj_weight` per rank and transpose each
projection to `[in_features, out_features]`.

The checkpoint's `config.json` has `layer_type` (singular). transformers
ignores that key and derives `layer_types` from `sliding_window_pattern`,
which is 1, so every layer is a full-attention layer; `src/config.py` follows
the same order.

## Parallelism

Tensor parallelism shards the heads, the MLP intermediate dimension and the
vocabulary. With 8 KV heads, TP 8 gives each rank one KV head, and TP 16 gives
each rank one KV head that two ranks share (`num_kv_replicas` 2).

- Prefill uses sequence parallelism: each rank holds `T / TP` of the tokens.
  The prompt bucket must be larger than TP and divisible by it.
- Decode uses the fused attention block kernel and an all-reduce. With one KV
  head per rank (TP 8 and TP 16) the cache is passed as a 3D cache with a 2D
  block table. With more than one KV head per rank (TP 1, 2, 4) the decode
  path builds a per-head 3D block table and scatters the new K/V itself, with
  the helpers in `compat.py` (not shipped in the 0.29 plugin).
- The embedding and the LM head are sharded on the vocabulary dimension.

## Feature status

"Exercised" means that a run in the validation report of the model folder's
`README.md` (vLLM Neuron 0.29.0.1.2.0, `trn2.48xlarge`, TP 8) covers the path.
"CPU-tested" means that a CPU-mode test in `test/unit/` covers it.
"Implemented, not covered" means that the code path is here but no test covers
it. "Not supported" means that the factory or the code refuses it.

| Feature | Status |
|---|---|
| BF16 served dtype (FP32 checkpoint) | Exercised; CPU-tested |
| TP 8, single node | Exercised (integration test, batch 1 and 4, and `vllm serve`) |
| TP 16 | Implemented, not covered: the port's two-node disaggregated test used it, and that test is not part of this folder |
| One KV head per rank (the TP 8 and TP 16 decode branch) | CPU-tested at TP 2 with a tiny model that has 2 KV heads |
| TP 1, 2, 4 (more than one KV head per rank) | Implemented, not covered. In CPU mode with the NKI simulator (`NKI_SIMULATOR=1`) the decode attention kernel of this branch returns NaN when the block table has inactive (`-1`) entries; the branch was not run on device |
| Flash attention prefill (NKI) | Exercised |
| Fused decode attention kernel | Exercised |
| Segmented prefill (the prefix-caching path) | Implemented, not covered: every recorded run disabled prefix caching |
| On-device greedy sampling | Exercised; CPU-tested |
| Sliding-window layers | Not supported (the factory rejects any `sliding_attention` layer) |
| Attention logit soft cap | Not supported (the factory rejects `attn_logit_softcapping`) |
| Attention bias, other activations | Not supported |
| FP8 KV cache | Code path present (scales from the checkpoint or 1.0); not tested |
| Quantized weights | Not supported |
| Speculative decoding | Code path present (`rejection_sampler`); not tested |
| Multimodal Gemma 3 (`Gemma3ForConditionalGeneration`) | Not implemented |
| Trn3 | Not tested |
