# MuseGlimmer: architecture and feature status

Neuron implementation of `MuseGlimmerForConditionalGeneration`
([`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B)):
image, video and text in, text out. The usage, compatibility matrix and test
instructions are in the [model README](../README.md).

## Architecture

**Text decoder** (`model_bf16.py`):
- 52 layers, hidden 6656, 32 query / 2 KV heads, head_dim 128, SwiGLU MLP (intermediate 19968).
- Sliding-window (2048) and global layers in a 3:1 pattern. RoPE applies only on sliding
  layers; global layers use no positional encoding.
- Weightless per-head QK RMSNorm and a fixed Q scale (`qk_scale_factor`), folded into the
  softmax scale.
- Sigmoid output gate before o_proj: `out = (attn * sigmoid(x @ W_gate)) @ W_o`.
- Centered sandwich RMSNorms, computed as `x * (1 + w)` with `w` in fp32.
- Weightless RMSNorm after the embedding; logit softcap `20 * tanh(logits * mult / 20)`.

**Perception encoder** (`vision_encoder_bf16.py`):
- 1.8B-parameter ViT: 50 layers, width 1536, 16 heads (head_dim 96), q/k/v biases.
- Windowed attention (32 x 32 patches) with full attention every fourth layer; the HF window
  reorder is replaced by a CPU pack order (`utils/vision_preprocessing.py`), so window and
  full layers differ only in their attention bounds.
- 2D axial RoPE, a 4-tap bilinear position table, then a 2x2 pixel shuffle, a 2-layer GELU
  adapter and the projection into the text hidden size.
- Video frames go through the same path, one item per temporal group (temporal patch 2).

**Parallelism (trn2.3xlarge, TP=4):** Q heads, the gate and the MLP are column-sharded,
o_proj and the MLP down projection are row-sharded, and the 2 KV heads are replicated to 4.
Prefill uses sequence parallelism. The encoder runs data-parallel by default or TP=4
(`vision_neuron_config.tp_size`).

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Exports the factory and `register()` (the `vllm.general_plugins` entry point) |
| `config.py` | Text, vision and top-level configs, with load-time asserts on every model-specific field |
| `factory.py` | Registry class: vLLM model stubs and the multimodal processor for the front-end, `from_configs` for the worker |
| `processing.py` | Multimodal processor; image and video prompt expansion identical to HF `MuseGlimmerProcessor` |
| `model_bf16.py` | Text decoder, LM head + softcap, KV spec, weight loading, `embed_multimodal` |
| `vision_encoder_bf16.py` | Perception encoder, adapter, projection, encoder-cache write |
| `weight_loaders.py` | Vision q/k/v fusion and transposed loaders |
| `decode_fusions.py`, `gate_decode.py` | Decode fusion knobs and the gated decode attention call |
| `kernels/nki_attn_gate_tkg.py` | NKI decode megakernel with the output gate, see below |
| `utils/` | Vision pack order, window bounds, RoPE tables; vision block packing; per-head decode table |

## NKI kernel in this folder

`kernels/nki_attn_gate_tkg.py` is a patched copy of nkilib
`experimental/transformer/attention_block_tkg.py` from nkilib 0.7.0 (SDK 2.33). It applies
the sigmoid output gate inside the decode kernel and keeps the attention output in SBUF
through the output projection: `out = (attn * sigmoid(X @ W_gate)) @ W_out`. The gate
projection reads the same normalized input as QKV and is issued before attention, so its
weight DMA overlaps the KV traffic. It is used for decode batch buckets of 16 and above
(`MUSE_GLIMMER_DECODE_FUSE_GATE_MIN_BS`); smaller buckets use the stock kernel with the gate
in torch.

## Sliding-window decode table

The runner trims each sliding-window group's decode block table to the window, aligned to
128 tokens. At LNC=2 the decode kernel shards the prior context over two programs, so it
needs a multiple of 256 tokens: window 2048 with block size 32 trims to 68 blocks (2176
tokens), which the kernel rejects (`Sharded s_prior must be divisible by p_max`).
`pad_swa_decode_table` (`model_bf16.py`) pads the table with inactive (`-1`) blocks to the
next power of two (4096 tokens, 128 blocks). The kernel skips the pad blocks and the mask
never reaches them, so the result equals attention over the wider window
(`test/unit/test_components.py`). A power-of-two width also keeps the kernel's KV DMA chunks
large: it re-tiles the cache into `gcd(block_size, s_prior / (programs * 128))`-token chunks,
which is 16 tokens at 4096 but 1 token at 2304.

## Environment knobs

| Variable | Default | Meaning |
|---|---|---|
| `MUSE_GLIMMER_DECODE_FUSE` | `gate` | `gate`: sigmoid gate and o_proj inside the decode kernel. `off`: the gate runs in torch. `norm`, `early_gate`: other folds, measured slower or neutral. |
| `MUSE_GLIMMER_DECODE_FUSE_GATE_MIN_BS` | 16 | Smallest decode batch bucket that uses the in-kernel gate. |
| `MUSE_GLIMMER_DECODE_OPT` | 3 | neuronx-cc `-O` level for the decode graph only. `off` keeps the engine level. |
| `MUSE_GLIMMER_KV_DEFER` | 2 | Decode KV writes run after the layer loop, one write per physical KV buffer (avoids a per-step copy of the shared hybrid KV buffers). `0` restores per-layer writes. |
| `MUSE_GLIMMER_VIS_MLP` | torch | `nki`: nkilib MLP kernel with exact GELU and the fc1 bias, used while I/rank < 4096. |
| `MUSE_GLIMMER_VIS_LN` | torch | `fused`: LayerNorms fused into the QKV and MLP kernels. |
| `MUSE_GLIMMER_VIS_SMALL_BLOCK` | 0 | `1`: single-block images run the smallest vision bucket that fits. Needs `num_vision_tokens_buckets` below the block size. |
| `MUSE_GLIMMER_VIS_WIN_SPLIT`, `MUSE_GLIMMER_VIS_ROPE`, `MUSE_GLIMMER_VIS_OPT` | off | Exact, but measured slower or neutral. |
| `MUSE_GLIMMER_DECODE_TORCH`, `MUSE_GLIMMER_COMPUTE_DTYPE`, `MUSE_GLIMMER_STRICT_CONFIG` | off | Debug and CPU-test knobs (`COMPUTE_DTYPE=float32` gives an exact structural check against HF FP32). |

## Feature status

| Feature | Status |
|---|---|
| BF16 text, image, video, mixed image + video | Supported |
| TP=4 text, vision DP or TP=4 | Supported (trn2.3xlarge) |
| Chunked prefill (`--max-num-batched-tokens` < `--max-model-len`) | Not validated on device (`serve.sh` uses single-shot prefill) |
| FP8 KV cache | Not in this folder (rejected at load) |
| FP8 weights, speculative decoding | Not in this folder |
| Prefix caching | Not validated |
| EPD (encoder-only pool) | Not supported |
