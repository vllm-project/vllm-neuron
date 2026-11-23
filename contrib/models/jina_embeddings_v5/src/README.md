# Jina Embeddings v5 Text Nano (EuroBERT encoder) on vLLM Neuron

This directory holds the `JinaEmbeddingsV5Model` implementation for the pooling
runner. The target checkpoint is `jinaai/jina-embeddings-v5-text-nano`, base
encoder weights only (no task LoRA adapter).

## Architecture

| Field | Value |
|-------|-------|
| Layers | 12, pre-norm encoder layers |
| Hidden size | 768 |
| Intermediate size | 3072 (SiLU gated MLP) |
| Attention | Bidirectional MHA, 12 Q heads, 12 KV heads, head_dim 64, no bias |
| RoPE | Default rotate-half, theta 1,000,000, no scaling |
| Normalization | RMSNorm, eps 1e-5 |
| Vocabulary | 128256, no LM head |
| Pooling | Last token, then L2 normalization |
| Compute dtype | BF16 (`config.json` states float32, the weights are BF16) |

## Files

- `factory.py`: `JinaEmbeddingsV5Model`, the registered class. It validates the
  HF config, requires the pooling runner, and builds the implementation.
- `config.py`: `JinaEmbeddingsV5Config` and `jina_v5_config_from_hf`, which map
  the EuroBERT config fields and reject RoPE scaling.
- `model.py`: the encoder and the pooling model.
- `plugin.py`: the `vllm.general_plugins` entry point that `pyproject.toml`
  installs.

`model.py` copies the TP-only parts of the plugin's Llama RMSNorm, default
RoPE, attention constructor and SiLU MLP, so that the folder imports no other
model's package.

## Forward paths

- **Prefill** (one sequence per step, padded to the token bucket):
  embedding with sequence parallelism (each rank holds `T/TP` tokens), then
  for each layer an all-gather, fused QKV projection and RoPE (`NF.qkv_proj`),
  the `attention_cte` kernel (`NF.flash_attention`) with `causal_mask=False`
  and key bounds `[0, real_len)` that remove the bucket pad tokens, the O
  projection and a reduce-scatter; then the MLP with the same SP pattern. The
  final norm output is all-gathered to `[T, H]` for the pooler.
- **Decode-classified step** (each sequence has 1 token): the attention of a
  1-token sequence equals its value vector, so the layer computes
  `o_proj(v_proj(x))` in float32 with a float32 TP all-reduce. The MLP
  all-reduces.

The encoder never reads or writes the KV cache. The model still reports a KV
spec, because the runner allocates and binds a cache for every model.

## Parallelism

Tensor parallelism shards the heads, the MLP intermediate dimension and the
vocabulary of the embedding. The code requires the head count and the
intermediate size to divide by the TP degree. The tested degree is TP 2. Data
parallel attention, MLP or embedding groups and decode context parallelism
are rejected.

## Unsupported configurations

The factory and the model reject: the generate runner, prefix caching,
chunked prefill with `max_num_batched_tokens < max_model_len`, an FP8 KV
cache, a quantized checkpoint, `is_decoder=True`, attention or MLP bias, an
activation other than SiLU, a RoPE scaling that changes the default
frequencies, and a local checkpoint that has an `adapters/` directory.
