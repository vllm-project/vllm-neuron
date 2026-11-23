# StableLM (stablelm-zephyr-3b) on vLLM Neuron

This directory holds the `StableLmForCausalLM` implementation for the native
route (`neuron_native`). The target checkpoint is
`stabilityai/stablelm-zephyr-3b`. The model is text only.

## Architecture

| Field | Value |
|-------|-------|
| Layers | 32 |
| Hidden size | 2560 |
| Intermediate size | 6912 (SwiGLU, SiLU) |
| Attention | MHA, 32 Q heads, 32 KV heads, head_dim 80 |
| RoPE | Partial: the first 20 of 80 dims of each head (`partial_rotary_factor` 0.25), theta 10000, no scaling |
| Normalization | LayerNorm with weight and bias, eps 1e-5 |
| Vocabulary | 50304 (the tokenizer has 50277 tokens), untied `lm_head` |
| Compute dtype | BF16 (the checkpoint stores BF16) |

StableLM differs from Llama in three places, and all three are in this folder:

- Each norm (the two per layer and the final norm) is a full LayerNorm with a
  bias, not an RMSNorm.
- RoPE rotates only the first `rotary_dim = head_dim * partial_rotary_factor`
  dims of each head. The weight loader permutes the Q and K output columns of
  each head (`rotary_head_dim_permutation`) so that the rotary pairs sit at
  `(i, i + head_dim / 2)`. The cos/sin tables hold cos=1 and sin=0 for the
  other dims. A full-head rotate_half RoPE then equals the partial RoPE of the
  HuggingFace model. Q and K use the same permutation, so Q.K does not change;
  V and O are not permuted. Prefill and decode use the same layout, so the KV
  cache layout is the same in both phases.
- The decode megakernel (`NF.attention_decode`) is not used: its RoPE stage
  accepts only d_head 64 or 128. Decode runs the QKV projection kernel, then
  RoPE, the paged KV write, paged attention over the block table and the O
  projection in torch. Attention and the O projection and its all-reduce run
  in FP32, with one cast to BF16 after the all-reduce. The KV cache is BF16.

The checkpoint stores separate `q_proj`, `k_proj` and `v_proj` weights. The
weight loaders fuse them into one `qkv_proj_weight` per rank and transpose each
projection to `[in_features, out_features]`.

`compat.py` vendors `hoist_rope_parameters`, which vLLM Neuron 0.29.0.1.2.0
does not ship. transformers 5 keeps `rope_theta` and `partial_rotary_factor`
under `rope_parameters`; `config.py` reads both from there.

## Parallelism

Tensor parallelism shards the heads, the MLP intermediate dimension and the
vocabulary. 32 heads, 6912 MLP columns and 50304 vocabulary rows divide by
TP 1, 2, 4, 8 and 16. The port's device tests used TP 8 (4 heads per rank).

- Prefill uses sequence parallelism: each rank holds `T / TP` of the tokens.
  The prompt bucket must be larger than TP and divisible by it.
- Decode runs the full batch on each rank and all-reduces after the O
  projection and after the MLP.
- The embedding and the LM head are sharded on the vocabulary dimension.

## Feature status

"Covered by a test" names the test in this folder that exercises the path.
Device results go in the Validation report of the model folder's `README.md`.

| Feature | Status |
|---|---|
| BF16 weights and compute | Implemented; CPU unit tests and the device integration test |
| TP 8, one node | Implemented; device integration test and module tests |
| TP 2 | Implemented; CPU unit tests (tiny checkpoint) |
| TP 1, 4, 16 | Implemented, not covered |
| Flash attention prefill (NKI) | Implemented; device module and integration tests |
| Torch paged-attention decode | Implemented; CPU unit tests and device module tests |
| Segmented prefill (the prefix-caching path) | Implemented, not covered — the tests turn prefix caching off |
| On-device greedy sampling | Implemented; CPU tiny E2E test and device integration test |
| Paged KV cache | Implemented |
| Max model length 2048 | Device integration test |
| Longer context (up to `max_position_embeddings` 4096) | Implemented, not covered |
| `inputs_embeds` | Not supported — pass `input_ids` |
| `use_qkv_bias`, `qk_layernorm`, `use_parallel_residual` | Not supported (the factory refuses them) |
| `rope_scaling` / a RoPE type other than `default` | Not supported (the factory refuses it) |
| `hidden_act` other than `silu`, checkpoint dtype other than BF16 | Not supported (the factory refuses them) |
| Quantization (FP8, MXFP4) | Not implemented |
| Speculative decoding | Code path present (`rejection_sampler` when the runner passes spec-decode metadata), not covered |
| LoRA, multimodal input, pooling / embedding tasks | Not implemented |
| Trn3 | Not tested |

## Files

- `__init__.py`: exports the factory and the config.
- `plugin.py`: the `vllm.general_plugins` entry point (`pyproject.toml` in the
  folder above).
- `factory.py`: validates the HF config and builds `model.py`.
- `config.py`: the `StableLmConfig` dataclass.
- `model.py`: LayerNorm, partial RoPE, attention, MLP, decoder, backbone and
  the causal-LM head, KV spec and weight loading.
- `compat.py`: the helper that vLLM Neuron 0.29.0.1.2.0 does not ship.
