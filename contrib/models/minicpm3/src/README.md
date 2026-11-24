# MiniCPM3 (openbmb/MiniCPM3-4B) on vLLM Neuron

This directory holds the `MiniCPM3ForCausalLM` implementation for the native
route (`neuron_native`). The target checkpoint is `openbmb/MiniCPM3-4B`. The
model is a BF16 dense causal LM with multi-head latent attention (MLA). The
input modality is text only.

## Architecture

| Field | Value |
|-------|-------|
| Layers | 62 |
| Hidden size | 2560 |
| Intermediate size | 6400 (SwiGLU) |
| Attention | MLA, 40 heads; `q_lora_rank` 768, `kv_lora_rank` 256; Q/K head dim 64 + 32 (no-RoPE + RoPE), V head dim 64 |
| RoPE | LongRoPE on the 32 rotary channels of each head, theta 10000 |
| Normalization | RMSNorm, eps 1e-5 (layers), 1e-6 (the two low-rank norms) |
| Vocabulary | 73448, tied embeddings |
| muP scaling | `scale_emb` 12, `scale_depth` 1.4, `dim_model_base` 256 |
| Compute dtype | BF16 (the checkpoint stores BF16) |

## Architecture mapping

| HF component | This implementation | Neuron operator |
|---|---|---|
| `q_a_proj` + `kv_a_proj_with_mqa` | One replicated latent projection | `NF.qkv_proj` |
| `q_a_layernorm`, `kv_a_layernorm` | Torch RMSNorm | - |
| `q_b_proj` + `kv_b_proj` + shared `k_pe` | One fused per-head QKV weight | `NF.qkv_proj` (prefill), `NF.attention_decode` (decode) |
| Prefill attention | Causal attention on padded heads | `NF.segmented_attention` when the runner sets a KV segment size (chunked prefill), otherwise `NF.flash_attention` |
| Decode attention | Paged attention kernel, per-head block table, in-kernel KV write | `NF.attention_decode` |
| `o_proj` | Output projection with zero rows at the pad channels | `NF.o_proj` (prefill), in the decode kernel (decode) |
| SwiGLU MLP | TP-sharded intermediate dimension | `NF.mlp` |
| `scale_emb`, `scale_depth`, `dim_model_base` | Torch scale ops | - |

The implementation pads the head dimension (96 for Q/K, 64 for V) to 128. The
rotary channels go to positions [0, 16) and [64, 80). Thus a full-width RoPE
with `rotate_half` rotates only the rotary channels. The softmax scale stays
`(qk_nope_head_dim + qk_rope_head_dim) ** -0.5`. The KV cache holds standard
per-head K and V, so `kv_heads` equals `num_attention_heads`. The fused QKV
weight holds an identity copy of the shared `k_pe` for each head.

The residual stream stays in FP32. The muP residual scale
(`scale_depth / sqrt(num_layers)`, 0.178) makes each branch small against the
residual, so a BF16 add would lose the low bits of the branch in each of the
124 adds.

## Checkpoint loading

The published checkpoint has one `pytorch_model.bin` and no `.safetensors`
file. The `SafetensorsCheckpoint` of vLLM Neuron 0.29.0.1.2.0 reads only
`.safetensors` files. `checkpoints.py` in this directory is that 0.29 loader
with a `pytorch_model*.bin` fallback added: it opens the `.bin` files with
`torch.load(mmap=True, weights_only=True)` and hands the weight loaders lazy
slice views, so sharding and pipelining work as for safetensors. `model.py`
imports the loader from here. The plugin is not patched; a checkpoint that has
`.safetensors` files loads as with the plugin's own loader.

`compat.py` holds two helpers that 0.29 does not provide:
`hoist_rope_parameters` (reads the transformers 5 `rope_parameters` layout)
and `build_per_head_block_table` (the per-head block table of the fused decode
kernel).

## Parallelism

Tensor parallelism shards the heads, the MLP intermediate dimension and the
vocabulary. The constructors require the TP size to divide the 40 heads, the
intermediate size (6400) and the vocabulary (73448 = 8 x 9181), so the legal
sizes are TP 1, 2, 4 and 8. The latent projection and the two low-rank norms
are replicated on every rank.

- Prefill uses sequence parallelism: each rank holds `T / TP` of the tokens
  of a prefill step. `T` is the padded token count of the step (the prefill
  bucket, 2,048 in the tested configuration); it must be larger than TP and
  divisible by it. The latent input (1280 wide, narrower than the 2560 hidden
  states) is all-gathered before the per-head projection.
- Chunked prefill: with `kv_segment_size_buckets` set, each prefill step runs
  `NF.segmented_attention`, which reads the K and V of the earlier chunks of
  the prompt from the paged KV cache (`cached_seq_len` prior tokens) together
  with the K and V that the step has just written.
- Decode uses the fused attention kernel and an all-reduce.
- The embedding and the LM head are sharded on the vocabulary dimension.

At TP 8 each rank holds 5 heads, an intermediate size of 800 and 9181
vocabulary rows.

## Feature status

| Feature | Status |
|---|---|
| BF16 checkpoint and compute | Implemented |
| `pytorch_model*.bin` checkpoint (the published format) | Implemented (`checkpoints.py`) |
| TP 8, single node | Implemented; the integration test configuration |
| TP 1, 2, 4 | Implemented |
| Context length | 9,216 tokens validated (the integration test); the checkpoint allows 32,768 positions, not validated above 9,216 |
| Chunked prefill with segmented attention (NKI) over the KV cache of the earlier chunks | Implemented; the integration test configuration (2,048-token chunks, prompts of 8,000 to 8,192 tokens) |
| One-shot prefill with flash attention (NKI), used when no KV segment size is set | Implemented; not run by the integration test |
| Fused decode attention kernel with in-kernel KV write | Implemented |
| Prefix caching | Uses the segmented-prefill attention; not validated (no test produces a cache hit) |
| On-device greedy sampling | Implemented |
| LongRoPE with equal long and short factors | Implemented (MiniCPM3-4B) |
| LongRoPE with different long and short factors past `original_max_position_embeddings` | Not supported (raises) |
| `inputs_embeds` | Not supported — pass `input_ids` |
| Checkpoint without `q_lora_rank`, with `attention_bias` or `mlp_bias`, or not BF16 | Not supported (the factory raises) |
| Quantization other than BF16 (FP8, MXFP4) | Not supported |
| Speculative decoding | Code path present (`rejection_sampler`); not validated |
| LoRA, multimodal input, pooling / embedding tasks | Not implemented |

The validation report in the model folder's `README.md` records which of
these the device runs covered.

## Tests

The tests that ship with this model are in `../test/`:

| File | Needs | Coverage |
|---|---|---|
| `test/unit/test_loading.py` | CPU | Registration, factory checks, config and head layout, RoPE buffers, checkpoint key coverage, the `.bin` loader fallback, fused MLA weight math in FP32 against HF |
| `test/unit/test_components.py` | CPU, 4 and 8 ranks | Attention prefill + decode, attention chunked prefill (three 512-token segmented chunks) and MLP prefill + decode (TP 4); full model load and logits (2 random layers at TP 8; optionally the real checkpoint) against HF |
| `test/unit/test_tiny_e2e.py` | CPU | The full vLLM `LLM(...)` pipeline on a tiny random `.bin` checkpoint at TP 1 and TP 2 |
| `test/integration/test_model.py` | 8 NeuronCores | Full-model logits through vLLM against the HF reference: two prompts of 8,000 to 8,192 tokens at a 9,216-token model length, chunked prefill in 2,048-token chunks, batch 1 and 4 |

The reference is the `MiniCPM3ForCausalLM` built into `transformers`. The
tests do not use the remote code of the checkpoint.
