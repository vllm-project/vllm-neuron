# Bloom (bloomz-560m) on vLLM Neuron

This directory holds the `BloomForCausalLM` implementation for the native
route (`neuron_native`, the default route of vLLM Neuron 0.29). The target
checkpoint is `bigscience/bloomz-560m`. The model is text only.

Deployment, compatibility, tests and the validation report: see
[../README.md](../README.md).

## Files

| File | What it holds |
|---|---|
| `__init__.py` | Exports `BloomForCausalLM` (the factory) and `BloomConfig` |
| `plugin.py` | `register()`: registers `BloomForCausalLM` with vLLM's `ModelRegistry`, replacing vLLM's GPU class. Run in every process through the `vllm.general_plugins` entry point in `../pyproject.toml` |
| `config.py` | `BloomConfig`, parsed from the Hugging Face `config.json` (`from_configs`). Accepts the HF aliases `n_layer`, `n_head` and `n_embed` |
| `factory.py` | Registry entry; validates the config and builds the model with `from_configs`. Carries the vLLM model protocol stubs (`embed_input_ids`, `compute_logits`) that vLLM's front end inspects |
| `model.py` | The model: ALiBi helpers and weight loaders, `BloomLayerNorm`, `BloomAttention`, `BloomMLP`, `BloomDecoderLayer`, `BloomModel`, `BloomForCausalLM` (`forward`, `get_kv_spec`, `bind_kv_cache`, `load_weights`) |
| `compat.py` | `build_per_head_block_table` and `scatter_new_kv`, which vLLM Neuron 0.29.0.1.2.0 does not ship in `vllm_neuron.model.kv_cache` |

## Architecture

| Field | bloomz-560m |
|-------|-------|
| Layers | 24 |
| Hidden size | 1024 |
| MLP | Not gated, 1024 -> 4096 -> 1024, GeLU (tanh approximation), with biases |
| Attention | MHA, 16 heads, head_dim 64, fused per-head interleaved QKV with bias, output bias |
| Position encoding | ALiBi (no RoPE) |
| Normalization | LayerNorm with bias, eps 1e-5, also directly after the word embedding |
| Vocabulary | 250880 (the tokenizer has 250680; the last 200 ids are padding), tied `lm_head` |
| Compute dtype | BF16 (the checkpoint stores FP16) |

### ALiBi in the head dimension

Bloom adds `slope[h] * key_position` to each attention score. The plugin's
attention kernels have no bias input, so `model.py` puts the bias into the
head dimension, where the kernels compute it exactly:

- Each head is padded from 64 to 128 lanes.
- The hidden input of the QKV projection gets the extra columns
  `1, pos // 64, pos % 64` and zero padding up to a multiple of 256
  (1024 + 256 = 1280 for bloomz-560m). bfloat16 holds both position parts
  exactly.
- The QKV bias goes into the row of the `1` column.
- Six extra Q lanes hold `64 * t_k` and `t_k` for three bfloat16 terms `t_k`
  whose sum is `slope / scale`. Two extra K lanes hold `pos // 64` and
  `pos % 64`, so the KV cache carries them into decode and segmented prefill.
- The extra V lanes and the matching rows of the output projection are 0.

Then `scale * (q' . k') = scale * (q . k) + slope * key_position`. The decode
kernel applies RoPE, so decode passes `cos = 1, sin = 0`, an exact identity.
Prefill does not apply RoPE. The encoding is exact for positions below
`64 * 256 = 16384`, so `max_model_len` above 16384 is rejected.

## Parallelism

Tensor parallelism shards the heads, the MLP intermediate dimension and the
vocabulary. With 16 heads the legal sizes are TP 1, 2, 4 and 8: the decode
kernel needs an even head count per rank, so TP 16 is rejected at
construction. The ALiBi slopes of a rank go into the QKV weight of that rank
at load time.

- Prefill uses sequence parallelism: each rank holds `T / TP` of the tokens.
  The prompt length (the padded bucket) must be larger than TP and divisible
  by it.
- Prefill attention: all-gather, QKV projection (split into column chunks when
  the fused width is above the QKV kernel's 4096 limit, as at TP 1), flash
  attention (or segmented attention when the runner passes cached blocks),
  output projection, reduce-scatter. The output bias is added after the
  collective.
- Decode: the fused decode attention kernel with a per-head 3D block table,
  then `scatter_new_kv` writes the new K/V into the paged cache, then an
  all-reduce. In CPU mode the torch fallback uses the 2D block table and
  writes the cache in place (`kv_cache_update_idx` as `uint32`).
- On a Neuron device the MLP goes through `vllm_neuron.functional.mlp` (the
  NKI MLP kernel) with `skip_gate=True` and `ActFnType.GELU_Tanh_Approx`. In
  CPU mode it runs a torch path with `gelu_tanh`, the same tanh-approximation
  GeLU written with elementwise ops: `vllm_neuron.functional.mlp`'s own torch
  fallback calls `torch.nn.functional.gelu`, which Dynamo cannot trace when the
  Neuron XLA overrides replace it.

## Feature status

| Feature | Status |
|---|---|
| BF16 compute from an FP16 checkpoint | Implemented |
| TP 2, single node | Integration test target; see the validation report |
| TP 1, 4, 8 | Implemented, not covered by the shipped tests on device |
| Flash attention prefill, fused decode attention (NKI) | Implemented |
| Segmented prefill (the prefix-caching path) | Implemented, not covered — every shipped configuration disables prefix caching |
| On-device greedy sampling | Implemented |
| Max model length up to 16384 | Implemented; the shipped tests use 2048 |
| `inputs_embeds` | Not supported — pass `input_ids` |
| `apply_residual_connection_post_layernorm=True` | Not supported (rejected by the factory) |
| `slow_but_exact=True` with `pretraining_tp > 1` | Not supported (rejected by the factory) |
| head_dim above 122 | Not supported (the ALiBi encoding needs 6 free lanes) |
| Quantized checkpoints, FP8 KV cache | Not supported |
| Speculative decoding | Code path present (`rejection_sampler`), not tested |
| Trn3 | Not tested |
