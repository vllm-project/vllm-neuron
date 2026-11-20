# Seed-OSS on vLLM Neuron

This directory holds the native-route (`neuron_native`) implementation of
`SeedOssForCausalLM`. The target checkpoint is
`ByteDance-Seed/Seed-OSS-36B-Instruct` (the loading test pins its published
`config.json` at revision `497f1dca95ebdec98e41d517b9f060ee753c902f`). The
model is text only.

## Architecture

| Field | Value |
|-------|-------|
| Layers | 64 |
| Hidden size | 5120 |
| Intermediate size | 27648 (SwiGLU, `silu`) |
| Attention | GQA, 80 Q heads, 8 KV heads, head_dim 128 |
| QKV bias | Yes (`attention_bias=True`); `o_proj` and MLP have no bias |
| RoPE | Default (no scaling), theta 1.0e7, applied in every layer |
| Vocabulary | 155136, untied LM head |
| Dtype | BF16 weights |

Seed-OSS is a Llama-style dense decoder. The one structural difference from
Llama is the bias on the Q, K and V projections. The implementation fuses QKV
into one parameter, so it needs a bias loader (`fused_qkv_bias_loader`) that
shards the bias exactly like the fused weight: rank `r` gets Q shard `r` and
KV shard `r // num_kv_replicas`, in `[Q, K, V]` order. The published
checkpoint has 771 tensors, and `checkpoint_mappings()` covers all of them.

Each traced graph writes one `[seed_oss_dispatch]` line per main-path
operation of the first and last decoder layer, recording the phase, the
operation, the TP rank and whether the NKI kernel or the torch fallback ran.
The lines come from the same eligibility predicates the `vllm_neuron.functional`
wrappers use, so a test can assert that the kernel path was taken.

## Parallelism

- Prefill uses sequence parallelism: attention and the MLP all-gather the
  tokens and reduce-scatter the result.
- Decode uses the fused attention kernel, which writes the new K/V into the
  paged cache, and the MLP kernel.
- KV heads are replicated when the TP size exceeds the 8 KV heads
  (`num_kv_replicas = world_size // num_key_value_heads`); the TP size must
  divide the head counts, and a size that does not raises at construction.
- The embedding and the LM head are sharded on the vocabulary dimension.

TP8 and TP16 both have recorded results on `trn2.48xlarge`. With vLLM Neuron
0.29.0.1.2.0, the single-node integration logit test ran at TP8 and the
`vllm serve` smoke at TP16. The CPU tests construct the model at TP8 on the
meta device and run the numerics at TP1.

## Limits

The factory refuses a configuration it cannot run, rather than loading and
producing wrong output. It rejects a dtype other than BF16,
`attention_out_bias=True`, `mlp_bias=True`, a `hidden_act` other than `silu`,
and any RoPE type other than `default`.

Other limits:

- There is no quantized path: BF16 weights only.
- No sliding-window or chunked attention (the KV spec sets neither).
- `max_position_embeddings` in the checkpoint is 524288, but no run in our
  evidence went beyond a 2,048-token context.
- The forward pass has a speculative-decoding branch (it calls
  `vllm_neuron.nn.rejection_sampler` when `spec_decode_metadata` is present),
  but no run in our evidence exercised it, so its status is unknown.

## Feature status

| Feature | Status |
|---|---|
| BF16 dense inference, text only | Supported; validated (see the folder README's validation report) |
| Fused QKV with bias | Supported; the loader layout is checked against the fused weight loader for every TP8 rank |
| TP with prefill sequence parallelism | Supported; TP8 and TP16 have recorded runs |
| On-device greedy sampling | Supported; used by the recorded logit run |
| Quantization (FP8, MXFP4, ...) | Not supported (no quantized path in the code) |
| Sliding-window / chunked attention | Not applicable |
| Multimodal input | Not applicable (text-only checkpoint) |
| Speculative decoding | Unknown (code path present, no evidence) |
| Context beyond 2,048 tokens | Unknown (no evidence) |

## Tests

- `../test/unit/test_seed_oss_loading.py`: CPU-mode registration and factory
  resolution, config mapping from the published `config.json`, TP8
  construction on the meta device with parameter shapes and full
  checkpoint-key coverage, and the fused QKV bias loader against the fused
  weight loader (14 cases).
- `../test/module/test_seed_oss_components.py`: CPU-mode component numerics on
  a tiny random checkpoint against a HuggingFace reference — prefill
  attention, decode attention, the MLP in both phases, and the assembled
  model's prefill logits with the dispatch log (5 cases).
- `../test/integration/test_model.py`: single-node TP8 offline logit
  validation through vLLM.

All three resolve this folder by file location through
`../test/_local_src.py`, so they also run after the model folder is copied out
of the repository.
