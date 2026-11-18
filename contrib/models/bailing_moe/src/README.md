# BailingMoeV2

This directory holds the native (`neuron_native`) implementation of
BailingMoeV2. The reference checkpoint is `inclusionAI/Ling-mini-2.0`
(revision `a810f6416bc4e1e29c9d7f271dd2fa7e56e71eab`).

BailingMoeV2 is a text-only mixture-of-experts decoder. The implementation
covers the BF16 checkpoint.

## Architecture

| Item | Value |
|---|---|
| Registry name | `BailingMoeV2ForCausalLM` |
| Decoder layers | 20. Layer 0 is a dense SiLU MLP (intermediate 5120); layers 1 to 19 are MoE |
| Routed experts | 256, top-8, intermediate 512 |
| Shared experts | 1 (intermediate 512), replicated on each rank |
| Router | Sigmoid scores in FP32, expert bias on the selection only, 8 groups and top-4 groups, then top-8 experts; weights normalized and scaled by 2.5 |
| Attention | GQA, 16 Q heads, 4 KV heads, head dimension 128, no bias |
| Normalization | RMSNorm, plus per-head QK RMSNorm before RoPE |
| Position embedding | Partial RoPE: the first 64 of the 128 head dims rotate, `rope_theta = 600000` |
| Hidden size | 2048 |
| Vocabulary | 157184, untied LM head |
| Dtype | BF16 |

## Differences from Qwen3

- The checkpoint fuses Q, K and V in one `query_key_value` tensor (Q heads,
  then K heads, then V heads).
- Partial RoPE. The prefill QKV kernel and the decode attention kernel apply
  RoPE with the rotate_half rule over the whole head dimension, so the weight
  loader permutes the dims of each Q and K head (and the QK-norm gammas) and
  the rotary tables put `cos = 1`, `sin = 0` on the pass-through dims. The KV
  cache keeps K in the permuted layout.
- A sigmoid router with an expert bias and group-limited top-k selection, in
  place of a softmax router.

## Feature Status

| Feature | Status |
|---|---|
| Tensor parallelism | TP 8 is the degree that the recorded tests ran. The code shards attention heads, the dense MLP and the shared expert over the TP group; other degrees are untested. |
| Expert parallelism | EP 8 is the degree that the recorded tests ran (32 experts on each rank, full intermediate size, so the prefill MoE kernel applies). The code also has a TP path for the routed experts, which is untested. |
| Sequence parallelism (prefill) | Supported by the code (the attention and MLP blocks exchange SP shards). Exercised by the component and full-model prefill tests. |
| Multi-node TP | TP 8 / EP 8 as 4 ranks on each of 2 nodes is the topology the recorded tests ran. |
| On-device sampling | Supported by the code: with an `on_device_sampling_config` the model runs `vllm_neuron.nn.sampler.Sampler` and skips the logit all-gather. Not covered by a recorded test id. |
| Quantization | Not supported. The factory accepts `None` or `"bf16"` and rejects a checkpoint with a `quantization_config`. |
| FP8 KV cache | Present in the code (quantize-on-write under `kv_cache_dtype=fp8`), untested. No recorded test id names an FP8 KV cache run. |
| Tied embeddings | Not supported. The factory requires a separate `lm_head.weight`. |
| Multi-token prediction | Not supported. The factory rejects `num_nextn_predict_layers > 0`. |
| Softmax routing | Not supported. The factory requires `score_function = "sigmoid"`. |
| Linear bias | Not supported. The factory rejects `use_bias` or `use_qkv_bias`. |
| Prefix caching | Unknown. The port's full-model tests served with prefix caching off, and the code has no prefix-caching-specific path. |
| Chunked prefill | Present in the code (`NF.segmented_attention` on the chunked path), untested. |
| LoRA, speculative decoding | Unknown. The implementation has no code for either. |

## Kernel Paths

Each main path goes through an `NF` API, which uses its NKI kernel when its
eligibility check passes and its torch fallback otherwise. In CPU mode
(`VLLM_NEURON_CPU_MODE=1`) `can_run_kernel` is false, so every API takes the
torch fallback.

| Phase | Operation | NF API |
|---|---|---|
| Prefill | QKV projection with RoPE | `NF.qkv_proj` |
| Prefill | Causal attention | `NF.flash_attention` |
| Prefill | Chunked prefill attention | `NF.segmented_attention` |
| Prefill | Output projection | `NF.o_proj` |
| Decode | Attention mask | `NF.gen_attention_decode_mask` |
| Decode | QKV, RoPE, KV cache write, attention | `NF.attention_decode` |
| Prefill | Routed experts (blockwise) | `NF.moe_cte` with `NF.build_blockwise_mapping` |
| Decode | Routed experts | `NF.moe_tkg` with `NF.get_local_expert_affinities` |
| Prefill and decode | Dense MLP and shared expert | `NF.mlp` |

The device component tests record the NKI call sites of each compiled graph on
each rank and fail if a required operation takes its torch fallback.
`test/module/test_kernel_routes.py` probes the eligibility predicates for the
serving geometry as well.
