# OLMo2 (OLMo-2-0425-1B) on vLLM Neuron

This directory holds the `Olmo2ForCausalLM` implementation for the native
route (`neuron_native`). The target checkpoint is `allenai/OLMo-2-0425-1B`.
The model is text only.

## Architecture

| Field | Value |
|-------|-------|
| Layers | 16 |
| Hidden size | 2048 |
| Intermediate size | 8192 (SwiGLU) |
| Attention | MHA, 16 Q heads, 16 KV heads, head_dim 128 |
| RoPE | Standard, theta 500000, no scaling |
| Normalization | RMSNorm, eps 1e-6 |
| Vocabulary | 100352, untied `lm_head` |
| Compute dtype | BF16 (the checkpoint stores FP32) |

OLMo2 differs from Llama in three places, and all three are in this folder:

- The decoder layer is post-norm: `h = h + norm(attn(h))` and
  `h = h + norm(mlp(h))`. There is no input norm.
- Q and K each have one RMSNorm over the full projected width of all heads,
  before RoPE. Under tensor parallelism each rank owns part of that width, so
  the sum of squares takes an FP32 all-reduce across the TP group.
- `lm_head` is a separate weight. A checkpoint with `tie_word_embeddings=True`
  is rejected at construction.

The checkpoint stores separate `q_proj`, `k_proj` and `v_proj` weights. The
weight loaders fuse them into one `qkv_proj_weight` per rank and transpose each
projection to `[in_features, out_features]`.

## Parallelism

Tensor parallelism shards the heads, the MLP intermediate dimension and the
vocabulary. The attention constructor requires `num_attention_heads % TP == 0`
with an **even** head count per rank, and `num_key_value_heads % TP == 0`, so
with 16 heads the legal sizes are TP 1, 2, 4 and 8. TP 16 (one head per rank)
and TP 3 are rejected at construction. The recorded runs used TP 8.

- Prefill uses sequence parallelism: each rank holds `T / TP` of the tokens.
  The prompt length must be larger than TP and divisible by it.
- Decode uses the fused attention kernel and an all-reduce.
- The embedding and the LM head are sharded on the vocabulary dimension.

## Feature status

The statuses come from two sources only: the code in this folder, and the
recorded test verdicts in the Validation report of the model folder's
`README.md`. "Exercised" means a test that covers the path was part of a run
with a recorded PASS verdict. "Implemented, not covered" means the code path
is here but no recorded run covers it. "Not supported" means the code refuses
it.

| Feature | Status |
|---|---|
| BF16 weights (FP32 checkpoint, BF16 compute) | Exercised |
| TP 8, single node | Exercised |
| TP 8 across 2 nodes | Exercised |
| TP 1, 2, 4 | Implemented, not covered |
| Flash attention prefill (NKI) | Exercised |
| Fused decode attention kernel | Exercised |
| Segmented prefill (the prefix-caching path) | Implemented, not covered — every recorded serving run set `--no-enable-prefix-caching` |
| On-device greedy sampling | Exercised |
| Paged KV cache, block size 32 | Exercised |
| Max model length 2048 | Exercised |
| Longer context (up to `max_position_embeddings` 4096) | Implemented, not covered |
| Sliding-window attention | Not supported (the KV spec sets no window; OLMo-2 does not use one) |
| `inputs_embeds` | Not supported — pass `input_ids` |
| Tied embeddings | Not supported |
| `rope_scaling`, `attention_bias` | Not supported |
| Quantization other than BF16 (FP8, MXFP4) | Not supported |
| Speculative decoding | Code path present (`rejection_sampler` when the runner passes spec-decode metadata); no recorded test, so unknown |
| LoRA, multimodal input, pooling / embedding tasks | Not implemented |
| Trn3 | Unknown — no recorded run |

## Tests

The tests that ship with this model are in the folder above this one:

- `test/unit/test_loading.py`: CPU-mode registration contract, config parse,
  checkpoint mapping coverage, weight-loader transforms and a full 8-rank
  CPU load.
- `test/module/test_components.py`: layer-0 attention and MLP against a
  HuggingFace reference (three-way), prefill and decode.
- `test/integration/test_model.py`: serves the model through vLLM and compares
  the raw logits with a HuggingFace golden reference.

All three reach this directory through `test/_local_src.py`, which loads it by
file location under the unique module name `contrib_olmo2_src`, so no test
needs the repository root on `sys.path`.
