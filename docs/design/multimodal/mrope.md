# M-RoPE (Multimodal Rotary Position Embedding)

<!-- meta: description: Multimodal RoPE (M-RoPE) design -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

## Overview

Multimodal Rotary Position Embedding (M-RoPE) extends standard RoPE to encode spatial structure for vision-language models such as Qwen3-VL. Instead of a single scalar position per token, M-RoPE uses three axes — temporal, height, and width — so that spatially adjacent image patches receive similar rotary embeddings and attend to each other more strongly.

### Background

M-RoPE was introduced in Qwen2-VL ([arXiv:2409.12191](https://arxiv.org/abs/2409.12191)) and carried forward to Qwen3-VL. The key idea: standard RoPE encodes position as a single scalar, which means two image patches that are spatially adjacent (e.g. same row, adjacent columns) but far apart in the flattened token sequence receive very different rotary embeddings. This weakens the attention signal between semantically related patches.

M-RoPE splits the RoPE frequency spectrum into three sections (controlled by `mrope_section = [24, 20, 20]` for Qwen3-VL, summing to `head_dim // 2 = 64`):

- **Temporal (T)**: 24 frequency pairs — encodes frame index (constant for images)
- **Height (H)**: 20 frequency pairs — encodes row position in the vision grid
- **Width (W)**: 20 frequency pairs — encodes column position in the vision grid

These are interleaved into a single frequency vector as `[T, H, W, T, H, W, ..., T, T, T, T]` so that each group of 3 consecutive frequency slots encodes one component from each axis. The remaining 4 slots (24 - 20 = 4 extra T slots) carry only temporal information.

For text tokens, all three axes share the same sequential value, so M-RoPE degenerates to standard 1D RoPE. For vision tokens, the height and width axes encode the 2D grid position, producing similar rotary embeddings for spatially adjacent patches and stronger attention affinity between them.

## Problem Statement

Standard 1D RoPE assigns monotonically increasing positions to all tokens. This works for text but loses spatial information for vision tokens: two image patches that are vertically adjacent may be hundreds of positions apart in the flattened sequence, producing weak attention affinity despite being semantically related.

M-RoPE solves this by assigning 2D grid coordinates (height, width) to vision tokens while keeping text tokens on a shared sequential counter across all three axes.

## Dual Position ID Design

The Neuron model requires **two separate position inputs** per forward pass:

1. **Sequential positions** (`positions`, shape `[T]`)
    - Monotonically increasing per-request, continuous across prefill and decode.
    - Used for KV cache slot mapping (`slot_mapping = positions → block/offset`).
    - Used for causal attention mask generation in decode (`gen_attention_decode_mask` needs linear ordering to determine which cached K/V each query token can attend to).
2. **Rotary position IDs** (`rotary_position_ids`, shape `[3, T]`)
    - Three axes: temporal (T), height (H), width (W).
    - Used exclusively for RoPE computation (cos/sin embeddings).
    - For text tokens: all 3 axes are identical sequential values (degenerates to standard RoPE).
    - For vision tokens: encodes the spatial grid so nearby patches get similar rotary embeddings.

### Why they cannot be unified

M-RoPE positions are always smaller than sequential positions for vision tokens because multiple tokens share the same grid coordinate (e.g. all tokens in the same row share a height value). This means M-RoPE max \< sequential position after an image. Using M-RoPE values for cache indexing would cause multiple tokens to map to the same slot, corrupting the KV cache. Using sequential positions for RoPE would discard the spatial encoding.

The upstream GPU model avoids this split by passing a single polymorphic `positions` tensor whose shape (1D vs 2D) determines the path inside the rotary embedding kernel. The Neuron model cannot do this because:

- The attention decode kernel (`gen_attention_decode_mask`) requires 1D positions.
- The compiled NEFF graph requires fixed-rank tensor inputs; a single tensor that changes between `[T]` and `[3, T]` would require two separate compilations for what is semantically the same input.

## Concrete Example

Consider a prompt with 12 text tokens, a 16x16 image (256 tokens), and 12 trailing text tokens (total 280 tokens):

``` text
Sequential positions (for cache/mask):
  [0, 1, 2, ..., 11, 12, 13, ..., 267, 268, 269, ..., 279]
   ─── text ───   ──── vision ────   ──── text ────

M-RoPE axis T (temporal):
  [0, 1, ..., 11,  12, 12, ..., 12,  28, 29, ..., 39]
   ─── text ───    constant for img    ── text ──

M-RoPE axis H (height):
  [0, 1, ..., 11,  12, 12, ..., 27,   28, 29, ..., 39]
   ─── text ───    row 0..15 (×16 cols each)  ── text ──

M-RoPE axis W (width):
  [0, 1, ..., 11,  12, 13, ..., 27, 12, ..., 27, ...,  28, 29, ..., 39]
   ─── text ───    col 0..15 repeating per row          ── text ──
```

Key observations:

- The maximum M-RoPE value across all axes after the image is **27** (from H and W axes), not 267.
- Trailing text resumes from `max(all axes) + 1 = 28`, ending at 39.
- Sequential positions reach 279, but M-RoPE max is only 39.
- Position delta = `39 + 1 - 280 = -240` (negative, reflecting compression).

During decode, all 3 axes collapse to the same value (`mrope_position_delta + context_len = -240 + 280 = 40`), since generated text has no spatial structure. Each subsequent decode step increments by 1: 41, 42, 43, ...

## Implementation

### Request initialization

When a new request arrives, `_init_mrope_positions` (in the model runner) calls the model's `get_mrope_input_positions` which:

1. Iterates over multimodal features to find vision token spans.
2. Assigns `np.indices((1, grid_h, grid_w))` grid coordinates to vision regions.
3. Assigns identical sequential values across all 3 axes for text regions.
4. Computes `mrope_position_delta = max(positions) + 1 - len(tokens)`.

The result is stored as `req.mrope_positions` (shape `[3, prompt_len]`) and `req.mrope_position_delta` on the request state.

### Prefill vs Decode: How M-RoPE Differs

Although both phases use the same `[3, T]` tensor format, the **content** of the rotary position IDs is fundamentally different between prefill and decode.

**Prefill** processes the full prompt in one pass. The `[3, T]` tensor contains a mix of spatial and sequential positions:

- Text regions: all 3 axes share the same sequential value (degenerate M-RoPE = standard RoPE).
- Vision regions: each axis encodes a different spatial dimension. The T axis holds a constant frame index, the H axis holds row coordinates, and the W axis holds column coordinates. This makes Q/K dot products between spatially adjacent patches naturally larger, encouraging local attention within the image.

The spatial divergence between axes is what gives M-RoPE its power: two patches in the same row share H values but differ in W, so only the W-frequency components rotate differently — producing a small angular difference and strong attention.

**Decode** generates one token at a time (or a small speculative draft). Since generated text has no spatial structure, all 3 axes collapse to the same scalar value:

``` text
decode step 1:  rotary_position_ids = [[284], [284], [284]]
decode step 2:  rotary_position_ids = [[285], [285], [285]]
...
```

This means during decode, M-RoPE is mathematically equivalent to standard 1D RoPE — the interleaving of T/H/W frequencies produces the same cos/sin values when all three input positions are identical. The 3D format is maintained only for graph compatibility with the compiled NEFF (which always expects `[3, T]`).

**Key implication for correctness**: The spatial information injected during prefill persists in the KV cache. When a decode token attends back to cached vision keys, the rotary dot-product naturally respects the spatial structure encoded at prefill time — decode tokens don't need spatial positions themselves because the spatial encoding is already baked into the cached K vectors.

### Runtime assembly (`_calc_mrope_positions`)

`_calc_mrope_positions` assembles the batch-level `[3, total_tokens]` tensor:

- **Prompt tokens (prefill)**: sliced from pre-computed `req.mrope_positions` which contain the full spatial grid layout.
- **Completion tokens (decode)**: all 3 axes = `delta + context_len + offset` (sequential, identical across axes — no spatial structure for generated text).

The runner then pads to the compilation bucket and passes as a keyword argument:

``` python
model_kwargs["rotary_position_ids"] = rotary_position_ids.to(device)
```

### Warmup (NEFF compilation)

Both prefill and decode warmup create synthetic M-RoPE tensors with the correct shape but dummy values:

``` python
# Prefill warmup: [3, bucket_size]
rotary_position_ids = positions.unsqueeze(0).expand(3, -1).contiguous()

# Decode warmup: [3, batch_size]
rotary_position_ids = positions.unsqueeze(0).expand(3, -1).contiguous()
```

The values are irrelevant for compilation — only the shape matters for NEFF graph tracing. Both produce the same tensor rank and dtype as real inference.

### Model forward pass

Inside `Qwen3VLTextModel.forward`:

``` python
# Compute cos/sin from 3D positions (M-RoPE interleaving)
position_embeddings = self.rotary_emb(rotary_position_ids, ...)

for layer in self.layers:
    layer(hidden_states, positions=positions,
          position_embeddings=position_embeddings, ...)
```

The `Qwen3VLTextRotaryEmbedding` module:

1. Reshapes `[3, T]` → `[3, 1, T]`.
2. Computes per-axis frequencies: `inv_freq @ position_ids` → `[3, 1, T, D/2]`.
3. Interleaves T/H/W frequencies into a single `[1, T, D/2]` tensor using `mrope_section = [24, 20, 20]` (for the 8B model: 24 freq pairs for T, 20 for H, 20 for W, summing to head_dim/2 = 64).
4. Returns `(cos, sin)` each of shape `[T, D/2]`.

### Non-nullable contract

`rotary_position_ids` is a **required** (non-optional) argument to both `Qwen3VLTextModel.forward` and `Qwen3VLForConditionalGeneration.forward`. This ensures that if the runner ever fails to provide M-RoPE positions, the model raises a `TypeError` immediately rather than silently falling back to 1D positions (which would produce incorrect attention patterns for any prompt containing vision tokens).

## Key Files

- Model: `vllm_neuron/model/qwen3_vl/model_bf16.py` (`Qwen3VLTextRotaryEmbedding`, `Qwen3VLTextModel.forward`)
- Runner: `vllm_neuron/vllm/worker/neuron_model_runner.py` (`_init_mrope_positions`, `_calc_mrope_positions`, warmup methods)
