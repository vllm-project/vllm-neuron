# ERNIE 4.5 (dense) — architecture and feature status

This card describes the implementation in this folder. Everything in it was
read out of `config.py`, `factory.py` and `model.py`, or out of the verified CI
evidence summarized in the model folder's `README.md`. Nothing here is a plan
or an expectation.

## Architecture

`Ernie4_5ForCausalLM` is a Llama-type dense decoder: RMSNorm, grouped-query
attention, a SwiGLU MLP with `silu`, tied embeddings and no projection bias.
The config dataclass defaults are the `baidu/ERNIE-4.5-0.3B-PT` geometry — 18
layers, hidden size 1,024, intermediate size 3,072, 16 query heads and 2
key/value heads of head dimension 128, vocabulary 103,424, `rope_theta`
500,000 and 131,072 maximum positions — and `from_configs` overwrites them
from the checkpoint's `config.json`.

The one structural difference from Llama is RoPE. ERNIE 4.5 rotates the
interleaved dimension pairs `(2i, 2i+1)` of each head (GLM style), while the
Neuron attention kernels rotate the halves `(i, i + Dh/2)`. The weight loader
therefore permutes the output rows of `q_proj` and `k_proj` within each head
into even-then-odd order once, at load time. After that permutation the
rotate-half kernels compute the interleaved rotation, and because attention
uses only the `q · k` dot product, which the shared permutation leaves
unchanged, the attention scores are identical. `V` and `o_proj` are untouched.
The prefill path, the decode path and the KV cache all hold the permuted
layout.

| File | Contents |
| --- | --- |
| `config.py` | `Ernie4_5Config` dataclass and `from_configs`, which accepts a `PretrainedConfig`, a `config.json` path or a dict. |
| `factory.py` | `Ernie4_5ForCausalLM` factory registered with vLLM; validates the config and selects the implementation. |
| `model.py` | RoPE permutation helpers, the fused QKV loader, checkpoint mappings, RMSNorm, rotary embedding, attention (prefill, decode, segmented), MLP, decoder layer, backbone, and the `ForCausalLM` with `forward`, `get_kv_spec`, `bind_kv_cache` and `load_weights`. |

Attention shards query and key/value heads across tensor-parallel ranks; when
the TP size is at least the key/value head count, each rank holds one
key/value head and several ranks replicate it. Prefill is sequence-parallel:
each rank takes its slice of the prompt, so the prompt length must exceed the
TP size and divide by it. The MLP shards the intermediate dimension.

## Feature status

"Verified" means the model folder's `README.md` cites CI evidence for it.
"Implemented, unverified" means the code path exists in this folder but no
evidence covers it. "Not implemented" means the code has no such path.

| Feature | Status |
| --- | --- |
| bfloat16 weights | Verified. The factory rejects every other dtype. |
| Tensor parallelism | Verified at TP 1 and TP 8. |
| Sequence-parallel prefill | Verified (the component and end-to-end runs exercise it). |
| Paged KV cache, BF16 | Verified. |
| Grouped-query attention with replicated KV heads | Verified at TP 8, where 2 key/value heads span 8 ranks. |
| NKI kernels for QKV, flash attention, O projection, decode attention and MLP | Verified. The multi-node component test asserts each of these dispatched its kernel rather than a torch fallback. |
| Torch fallback paths (CPU mode) | Verified by the CPU component tests. |
| On-device greedy sampling | Verified (the logit runs set `all_greedy`). |
| Full-vocabulary logprobs (`max_logprobs=-1`, `raw_logits`) | Verified by the logit run. |
| Disaggregated inference (1P1D, two nodes) | Verified. |
| Prefix caching / chunked prefill | Implemented, unverified. `model.py` routes a nonzero `kv_segment_size` to `NF.segmented_attention`, but the runs we hold disabled prefix caching. |
| Speculative decoding | Implemented, unverified. `forward` returns `rejection_sampler(...)` when it gets `spec_decode_metadata`; no evidence exercises it. |
| Data parallelism | Unknown. The `model.py` header names DP as supported, but the folder has no data-parallel wiring and no evidence covers it. |
| Quantization (FP8, MXFP4, INT8), FP8 KV cache | Not implemented. There is no quantization module and no quantized variant. |
| LoRA | Not implemented. |
| Multimodal input, prompt embeddings | Not implemented. `forward` rejects `inputs_embeds`. |
| Sliding-window attention | Not implemented. `get_kv_spec` reports `sliding_window_size=None` for every layer. |
| `neuron_native` runtime backend | Required. The port supports no other route. |
| Trn3 | Not tested. |

The architecture string `Ernie4_5ForCausalLM` is also an upstream vLLM
built-in, so registering this factory replaces vLLM's own entry rather than
adding a new one. A process without the registration resolves vLLM's class
instead of reporting an unsupported architecture, and then fails on the
`from_configs` classmethod above, which vLLM's own classes do not define; see
the model folder's `README.md` ("Usage" and Exceptions).

## Configurations the factory rejects

`_validate_config` raises `ValueError` for a non-bfloat16 dtype, `use_bias`
set, a non-default `rope_scaling` or `rope_parameters` type, an odd head
dimension, and any `hidden_act` other than `silu`. The unit test under
`../test/unit/test_factory.py` checks each of these against the real
checkpoint config.
