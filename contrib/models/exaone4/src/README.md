# EXAONE 4.0 on vLLM Neuron

This directory holds the native-route (`neuron_native`) implementation of
`Exaone4ForCausalLM`. The target checkpoint is `LGAI-EXAONE/EXAONE-4.0-32B`.
The model is text only. vLLM ships its own `Exaone4ForCausalLM`, so the
architecture must be re-registered for this class to run (see the folder
README).

## Architecture

| Field | Value |
|-------|-------|
| Layers | 64 |
| Hidden size | 5120 |
| Intermediate size | 27392 (SwiGLU, `silu`), zero-padded to 3456 per rank at TP8 |
| Attention | GQA, 40 Q heads, 8 KV heads, head_dim 128, per-head QK-norm |
| Layer types | Hybrid "LLLG": 3 sliding-window layers (window 4096) then 1 full layer |
| RoPE | Llama3 scaling (factor 16, original context 8192, theta 1.0e6 — the values the port's tiny test takes from the checkpoint); sliding layers only, full layers are NoPE |
| Vocabulary | From the checkpoint (the config default is 102400); untied LM head |
| Dtype | BF16 weights |

The structure follows the Qwen3 dense model (GQA, per-head QK-norm, SiLU
MLP), with four EXAONE-specific differences:

- **Post-norm decoder layer.** There is no input norm; the layer normalizes
  the attention output and the MLP output before each residual add.
- **Hybrid attention.** A sliding-window layer attends to the last 4,096
  keys; a full layer attends to every key. `layer_types` carries one entry per
  layer, and the config derives the HF default pattern when the checkpoint
  omits it.
- **RoPE only on sliding layers.** The full layers use no position embedding.
  The port feeds the identity rotation (cos 1, sin 0) to the same fused
  kernels, so both layer types take one code path.
- **Padded MLP shard.** The per-rank intermediate size at TP8 (27392 / 8 =
  3424) is zero-padded to a multiple of 128 (3456). The zero columns and rows
  contribute exactly zero, which the unit tests check numerically.

Each layer writes one `[exaone4_dispatch]` log line per main component at
trace time, carrying the result of the same eligibility predicate the
`vllm_neuron.functional` wrappers use.

## Parallelism

- Prefill uses sequence parallelism; decode uses the fused attention kernel
  with a per-head block table for the paged KV cache.
- TP8 gives 5 Q heads and 1 KV head per rank. The decode attention kernel
  shards on batch and prior sequence, so it does not require an even Q-head
  count per rank.
- The embedding and the LM head are sharded on the vocabulary dimension.

TP8 is the only degree with recorded results: TP8 single-node in the vLLM
Neuron 0.29.0.1.2.0 logit test and `vllm serve` run. The unit tests check the
loader transforms at TP8 (ranks 0, 3 and 7) and run the tiny end-to-end cases
at TP1 and TP2.

## Dependencies

The implementation imports only shared building blocks, as
[contrib/README.md](../../../README.md) sanctions: `vllm_neuron.functional`,
`vllm_neuron.nn`, `vllm_neuron.model.neuron_config`,
`vllm_neuron.model.kv_cache`, `vllm_neuron.parallel`, `vllm_neuron.utils`
(checkpoint and weight-loader helpers) and vLLM's public APIs. It imports
nothing from another model's folder.

`compat.py` holds three helpers that vLLM Neuron 0.29.0.1.2.0 does not ship:
`hoist_rope_parameters` (used by
`config.py`), and `build_per_head_block_table` and `scatter_new_kv` (used by
`model.py`). `plugin.py` is the `vllm.general_plugins` entry point that
`../pyproject.toml` declares.

`model.py` also imports two enum types from `nkilib`
(`nkilib.core.utils.common_types.NormType` and `QKVOutputLayout`). `nkilib` is
not part of the plugin tree, but it is a hard dependency of the
`vllm_neuron.functional` modules this model calls — `functional/attention/qkv.py`
and `functional/mlp.py` import the same module — and these two enums are the
argument types those wrappers take. Every officially-supported model does the
same (`vllm_neuron/model/llama3/model.py`, `vllm_neuron/model/qwen3/model.py`,
`vllm_neuron/model/gpt_oss/model_bf16.py`). So it is a stable, shared building
block in the sense of the contrib guide, and it is already installed wherever
`vllm_neuron.functional` is importable.

## Limits

The factory refuses a dtype other than BF16, and a `layer_types` list whose
length does not match `num_hidden_layers`. The config accepts only the
`default` and `llama3` RoPE types.

Other limits:

- No quantized weight path: BF16 only.
- There is an FP8 KV-cache quantize-on-write path (used only with
  `kv_cache_dtype=fp8`), but no run in our evidence enabled it, so its status
  is unknown.
- The forward pass has a speculative-decoding branch (it calls
  `vllm_neuron.nn.rejection_sampler` when `spec_decode_metadata` is present),
  with no evidence behind it. Read from the code: this model can only be the
  *target* of speculative decoding, never the draft. A draft architecture is
  looked up in `dict(get_models())` in `vllm_neuron/vllm/spec_decode/eagle.py`,
  which consults the plugin's own model map and not vLLM's `ModelRegistry`. A
  contrib model is deliberately absent from that map, so registering it the
  way the folder README describes does not reach this lookup, and using a
  contrib architecture as the draft raises `ValueError`. That needs option 2
  of contrib/README.md.
- No recorded run went past a 4,096-token context, so the behaviour at the
  sliding-window boundary (and the block-table trimming above it) is
  untested; the checkpoint's `max_position_embeddings` is far larger.

## Feature status

| Feature | Status |
|---|---|
| BF16 dense inference, text only | Supported; validated (see the folder README's validation report) |
| Hybrid sliding-window / full attention with NoPE full layers | Supported; the config mapping and the per-layer dispatch are asserted by the tests |
| Padded per-rank MLP | Supported; checked against the unpadded checkpoint MLP in FP32 |
| Llama3-scaled RoPE | Supported; checked against the HF `Exaone4RotaryEmbedding` |
| TP with prefill sequence parallelism | Supported at TP8 (recorded); TP1 and TP2 in the tiny CPU cases |
| On-device greedy sampling | Supported; used by the recorded logit run |
| Weight quantization (FP8, MXFP4, ...) | Not supported (no quantized weight path) |
| FP8 KV cache | Unknown (code path present, no evidence) |
| Speculative decoding | Unknown (code path present, no evidence) |
| Multimodal input | Not applicable (text-only checkpoint) |
| Context beyond the 4,096-token window | Unknown (no evidence) |

## Tests

- `../test/unit/test_exaone4_loading.py`: CPU-mode registration, config
  mapping against the HF config (hybrid layer types, NoPE layers, Llama3 RoPE,
  untied head), full weight-mapping coverage of the checkpoint with shape and
  dtype checks, and the TP8 attention and padded-MLP loader transforms against
  independent checkpoint slices. Needs the real checkpoint
  (`EXAONE4_CHECKPOINT_PATH`).
- `../test/module/test_exaone4_rope.py`: `Exaone4RotaryEmbedding` in isolation
  against the HF `Exaone4RotaryEmbedding`, with the Llama3 scaling parameters
  of the real config. Needs the real checkpoint's `config.json`.
- `../test/unit/test_tiny_exaone4_e2e.py`: the full vLLM pipeline on CPU with
  a random-weight tiny checkpoint — TP2 logits against an HF reference, and a
  TP1 run that asserts the dispatch lines for both phases of every layer.
  Needs no checkpoint.
- `../test/integration/test_model.py`: single-node TP8 offline logit
  validation through vLLM.
