# Granite — architecture and feature status

This card describes the implementation in this folder. Everything in it was
read out of `config.py`, `factory.py` and `model.py`, or out of the verified CI
evidence summarized in the model folder's `README.md`. Nothing here is a plan
or an expectation.

## Architecture

`GraniteForCausalLM` is a Llama-style dense decoder — RMSNorm, a SwiGLU MLP
with `silu`, rotary embeddings, tied embeddings — with two differences from
Llama:

- Every attention and MLP projection can carry a bias (`attention_bias`,
  `mlp_bias`). In `ibm/PowerLM-3b` both are set. The row-parallel biases
  (`o_proj`, `down_proj`) are divided by the tensor-parallel size at load time,
  because each rank adds its share before the tensor-parallel reduction.
- Four scalars rescale the network: `embedding_multiplier` multiplies the token
  embeddings, `attention_multiplier` replaces `head_dim ** -0.5` as the softmax
  scale, `residual_multiplier` multiplies each sub-layer output before the
  residual add, and `logits_scaling` divides the logits before sampling. For
  `ibm/PowerLM-3b` they are 12.0, 0.015625, 0.22 and 9.0.

The config dataclass defaults are the `ibm/PowerLM-3b` geometry — 40 layers,
hidden size 2,304, intermediate size 9,216, 36 query and 36 key/value heads
(so multi-head attention, not grouped-query) of head dimension 64, vocabulary
49,152, `rope_theta` 10,000 — and `from_configs` overwrites them from the
checkpoint's `config.json`. That checkpoint stores FP32 weights and sets no
dtype; the config resolves a missing or FP32 dtype to bfloat16 and the weight
loader casts each tensor, so the served model is bfloat16.

| File | Contents |
| --- | --- |
| `config.py` | `GraniteConfig` dataclass and `from_configs`, which accepts a `PretrainedConfig`, a `config.json` path or a dict, and resolves the dtype. |
| `factory.py` | `GraniteForCausalLM` factory registered with vLLM; validates the config and selects the implementation. |
| `model.py` | The fused QKV bias loader, RMSNorm, rotary embedding, attention (prefill, decode, segmented), MLP, decoder layer, backbone, and the `ForCausalLM` with `forward`, `get_kv_spec`, `bind_kv_cache`, `checkpoint_mappings` and `load_weights`. |

Attention shards heads across tensor-parallel ranks and the MLP shards the
intermediate dimension. Prefill is sequence-parallel: each rank takes its slice
of the prompt, so the prompt length must exceed the TP size and divide by it.
Decode reads a paged KV cache; with more than one key/value head per rank it
builds a per-head 3D block table and asks the shared decode helper whether the
fused kernel accepts that geometry, falling back to the 2D table otherwise.
That gate is `vllm_neuron.functional.attention.attention_decode.
_can_use_attention_block_kernel`, a private name in shared plugin code, so a
change there can change this model's decode path.

## Feature status

"Verified" means the model folder's `README.md` cites CI evidence for it.
"Unconfirmed" means a case for it ran inside the two-node device run whose
12-of-18 case count leaves per-case outcomes unknown. "Implemented,
unverified" means the code path exists in this folder but no evidence covers
it. "Not implemented" means the code has no such path.

| Feature | Status |
| --- | --- |
| bfloat16 execution of an FP32 checkpoint | Verified by the CPU loading run, which checks that every checkpoint tensor has a mapping and compares selected shards of two layers, the embedding, the tied head and the final norm against the raw FP32 tensors. |
| Projection biases, including the divided row-parallel biases | Verified by the same shard comparison. |
| The four Granite scalars | Verified against the real config by the CPU loading run, and exercised by every component and end-to-end run. |
| Tensor parallelism | Verified at TP 4 on device (2 nodes) and on 4 CPU ranks; TP 2 on CPU for the decoder-layer test. No other degree was tested. |
| Sequence-parallel prefill | Verified (the component runs slice the prompt per rank). |
| Paged KV cache, BF16 | Verified. |
| Multi-head attention (36 KV heads) with the per-head block table on decode | Verified at TP 4, where each rank holds 9 key/value heads. |
| Full-vocabulary logprobs (`max_logprobs=-1`, `raw_logits`) and on-device greedy sampling | Unconfirmed — the logit case that used them is inside the 12-of-18 run. |
| Prefix caching (`NF.segmented_attention` path) | Unconfirmed — a prefix-reuse case ran inside the 12-of-18 run. |
| Chunked prefill | Implemented, unverified. It shares the segmented-attention path with prefix caching; no case names it. |
| Speculative decoding | Implemented, unverified. `forward` returns `rejection_sampler(...)` when it gets `spec_decode_metadata`; no evidence exercises it. |
| Data parallelism | Not implemented. The decode path pins `attention_dp=1` with no DP group. |
| Quantization (FP8, MXFP4, INT8), FP8 KV cache | Not implemented. There is no quantization module and no quantized variant. |
| LoRA | Not implemented. |
| Multimodal input, prompt embeddings | Not implemented. `forward` rejects `inputs_embeds`. |
| Sliding-window attention | Not implemented. `get_kv_spec` reports `sliding_window_size=None` for every layer. |
| NKI-kernel dispatch assertions | None. The device tests compare numerics; no test asserts that a component took its NKI kernel rather than a torch fallback. |
| Trn3 | Not tested. |

## Configurations the factory rejects

`_validate_config` raises `ValueError` for a dtype other than bfloat16 or
float32, for any `hidden_act` other than `silu`, and for a non-default
`rope_scaling` or `rope_parameters` type. The unit test under
`../test/unit/test_loading.py` checks the activation and RoPE rejections
against the real checkpoint config.
