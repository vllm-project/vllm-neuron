# Qwen3-MoE (Qwen3-30B-A3B) on vLLM Neuron

This directory holds the `Qwen3MoeForCausalLM` implementation for the native
route (`neuron_native`). The target checkpoint is `Qwen/Qwen3-30B-A3B` (BF16).
The model is text only.

## Architecture

| Field | Qwen3-30B-A3B |
|---|---|
| Hidden size | 2048 |
| Layers | 48. Each layer has a sparse MoE block |
| Attention | GQA, 32 Q heads, 4 KV heads, head_dim 128, per-head Q/K RMSNorm before RoPE, no bias |
| RoPE | Default (rotate_half), theta 1e6, no scaling |
| Experts | 128 routed experts, top-8, expert intermediate size 768, SiLU gate |
| Router | Softmax over all experts, top-8, then renormalization (`norm_topk_prob=True`) |
| Shared expert | None |
| Vocabulary | 151936 (the tokenizer has 151669 tokens), untied LM head |
| Compute dtype | BF16 (the checkpoint stores BF16) |

The attention (`Qwen3Attention`), the norms (`Qwen3RMSNorm`), the RoPE
(`Qwen3RotaryEmbedding`), the backbone forward (`Qwen3Model`) and the
causal-LM forward, KV-cache binding and sampler (`Qwen3ForCausalLM`) are the
Qwen3 dense modules of vLLM Neuron 0.29.0.1.2.0 (`vllm_neuron.model.qwen3`),
imported from the installed plugin. This directory adds:

- `Qwen3MoeSparseMoeBlock`: the router weight, the per-expert gate/up and down
  weights, their loaders and the prefill and decode MoE paths.
- `Qwen3MoeDecoderLayer`: the Qwen3 pre-norm layer with the MoE block in place
  of the dense MLP. In decode the MoE kernel applies the post-attention
  RMSNorm itself.
- `Qwen3MoeModel` and `Qwen3MoeForCausalLM`: the layer stack with the MoE
  layers, the untied LM head, and the HF checkpoint mappings
  (`qwen3_moe_checkpoint_mappings`).
- `Qwen3MoeConfig`: the plugin's `Qwen3Config` plus the MoE fields. It also
  reads `rope_theta` from the transformers 5 `rope_parameters` form.

## Parallelism

- Use `--tensor-parallel-size 16 --enable-expert-parallel`. This gives EP 16
  over the TP group, and each rank holds 8 complete experts. With EP the
  expert-parallel degree must equal the TP size; the model refuses another
  `ep_degree`.
- Attention: TP head sharding. At TP 16 each rank holds 2 query heads and 1
  KV head; each KV head is replicated on 4 ranks.
- Prefill uses sequence parallelism: each rank holds `T / TP` of the tokens.
  The prompt bucket must be larger than TP and divisible by it.
- Without EP, each rank holds all experts with a TP shard of the expert
  intermediate dim. At TP 16 this gives 48 columns per rank. This shard is
  smaller than the minimum of the prefill MoE kernel (128), so prefill then
  uses the torch MoE path, and the model logs a warning. Use EP.
- The model refuses an expert count that the EP degree does not divide, and an
  expert intermediate size that the expert TP degree does not divide.
- Data parallelism is not supported; the model refuses `--data-parallel-size`
  other than 1.

## Kernel paths

| Phase | Operation | Path |
|---|---|---|
| Prefill | QKV, Q/K norm, RoPE | `NF.qkv_proj` (plugin `Qwen3Attention`) |
| Prefill | Attention | `NF.segmented_attention` (chunked prefill) or `NF.flash_attention` |
| Prefill | Output projection | `NF.o_proj` |
| Prefill | Router | `NF.router`, which runs its PyTorch path in vLLM Neuron 0.29.0.1.2.0 (FP32 matmul, softmax, top-k and renorm) |
| Prefill | Experts | `NF.build_blockwise_mapping` and `NF.moe_cte` (`shard_on_block`, block size 256) |
| Decode | Attention | `NF.attention_decode` (plugin `Qwen3Attention`) |
| Decode | RMSNorm, router, experts | `NF.moe_block_tkg` (all-expert mode, BF16 router matmul) |

In prefill the router runs on the sequence-parallel shard. The block then
all-gathers the hidden states and the routing weights, keeps the weights of
the local experts (with EP), builds the blockwise token-to-expert mapping and
runs the experts, and reduce-scatters the result back to the shard. Bucket
padding tokens get no expert work: the runner pads a prefill at the last real
position, so the first maximum of the positions marks the last real token.
Decode runs the fused RMSNorm + router + experts kernel on the full batch on
each rank and all-reduces the partial sums of the local experts.

## Feature status

"Covered by" names the test in this folder that exercises the path. Device
results go in the Validation report of the model folder's `README.md`.

| Feature | Status |
|---|---|
| BF16 weights and compute | Implemented; unit tests (CPU) and the device integration test |
| TP 16 with EP 16, one node | Implemented; device integration test |
| TP 2 with EP 2 and without EP | CPU unit tests (tiny checkpoints) |
| Other TP sizes, TP 16 without EP | Implemented, not covered on device |
| Segmented (chunked) prefill | Implemented; device integration test (2048-token chunks, KV segment 2048), CPU tiny-model test |
| Flash-attention prefill (one step) | Implemented; CPU tiny-model test only |
| On-device greedy sampling | Device integration test |
| Max model length 9216 | Device integration test (prompts of 8000 to 8192 tokens) |
| Longer context (up to `max_position_embeddings` 40960) | Implemented, not covered |
| Quantized checkpoints | Not supported. The factory rejects them |
| Dense MLP layers (`mlp_only_layers`, `decoder_sparse_step` > 1) | Not supported. The factory rejects them |
| `norm_topk_prob: false` | Not supported. The factory rejects it: `NF.router` has no "softmax over all experts, then top-k without renorm" order |
| Sliding-window attention, attention bias, `rope_scaling`, `hidden_act` other than SiLU | Not supported. The factory rejects them |
| Data parallelism | Not supported. The model rejects it |
| Prefix caching, speculative decoding | Not tested |
| Trn3 | Not tested |

## Files

- `__init__.py`: exports the factory and the config.
- `plugin.py`: the `vllm.general_plugins` entry point (`pyproject.toml` in the
  folder above).
- `factory.py`: validates the HF config and builds `model.py`.
- `config.py`: the `Qwen3MoeConfig` dataclass.
- `model.py`: the sparse MoE block, its weight loaders, the decoder layer, the
  backbone and the causal-LM head with the checkpoint mappings.
