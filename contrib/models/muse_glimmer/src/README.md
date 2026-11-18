# Muse-Glimmer

BF16 text decoder and vision encoder for Muse-Glimmer-30B (text, images, and
video), on Model Runner V1. Plain PyTorch, no NKI kernels: the math is the same
as the HF transformers reference under `torch.compile(fullgraph=True)`, and the
only change is the paged KV cache.

Deployment, compatibility, and accuracy: see [../README.md](../README.md).

## Files

| File | What it holds |
|---|---|
| `__init__.py` | `register()`: registers the factory with vLLM's `ModelRegistry` (overriding vLLM's GPU class) and adds it to vLLM Neuron's model list |
| `factory.py` | Registry entry `MuseGlimmerForConditionalGeneration`; validates the config; encoder sizing hooks (`SupportsSpatialMerge`, `SupportsEncoderBudget`, `SupportsEncoderCacheLayout`) |
| `config.py` | `MuseGlimmerTextConfig`, parsed from the HF `text_config` |
| `model_bf16.py` | Text decoder, LM head, KV-cache spec, weight loading, `embed_multimodal` |
| `vision_encoder_bf16.py` | HF vision tower + adapter + projection; writes embeddings into the on-device encoder cache |
| `processing.py` | Prompt processing from the transformers processor at the fixed shapes |
| `shapes.py` | The fixed image and video shapes (one compiled vision graph each) |

Because `register()` overrides vLLM's own class for this architecture in every
process, the API server also uses this package's processor.

## Text decoder

From `config.py` (Muse-Glimmer-30B): 52 layers, hidden 6656, 32 query heads,
2 KV heads, head_dim 128, SwiGLU 19968, vocab 202048.

| Item | Muse-Glimmer |
|---|---|
| Layer pattern | `[local, local, local, global]` x 13 |
| Local layers | sliding window 2048, RoPE theta 500000 |
| Global layers | full attention, no positional encoding (NoPE) |
| Attention output | gated: `o_proj(attn * sigmoid(gate_proj(x)))` |
| Q/K norm | weight-less RMSNorm; Q scaled by `qk_scale_factor` (3.87), folded with `1/sqrt(head_dim)` |
| Layer norms | four centered norms per layer, `x * (1 + w)`; post norms use `post_norm_eps` |
| Embedding | weight-less RMSNorm after the embedding |
| Logits | `cap * tanh(logits * output_multiplier / cap)`, `cap = final_logit_softcapping` |

Parallelism follows `qwen3`: TP over heads and MLP, sequence parallel in
prefill (all-gather / reduce-scatter), vocab-sharded embedding, column-parallel
LM head. When TP is larger than the KV-head count, KV heads are replicated.

Attention reads cached K/V through the block table and the chunk's own K/V from
the new tensors (the chunk's slots are written after the read). vLLM creates 4
KV-cache groups (sliding-window and full layers).

## Vision encoder

The HF vision modules run unchanged, with grid metadata precomputed on the host
for each fixed shape and the per-window attention replaced by one SDPA call
under a block-diagonal mask (the form validated with `fullgraph=True`). The
encoder is replicated on every rank. The embeddings fill the item's
encoder-cache blocks in order (`write_block_ids`); blocks hold 256 tokens, so an
item pads its last block, which `encoder_cache_num_blocks` absorbs.

## Validated configuration

Trn3, TP=2, logical NeuronCore config 2, `max_model_len` 2048,
`max_num_seqs` 1, `num_batched_tokens_buckets` [1024],
`encoder_attention_block_size` 1024, `num_encoder_input_buckets` [1024, 1564, 1632, 2080], `encoder_cache_num_blocks` 16, prefix caching off. Serve with
`--reasoning-parser muse_glimmer`.

## Not supported yet

Multi-image prompts, MXFP8, encoder parallelism, EPD, segmented prefill,
speculative decoding, Model Runner V2, TP above 2 (not validated).
