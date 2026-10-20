# SPDX-License-Identifier: Apache-2.0
"""CPU preprocessing functions for the Qwen3-VL vision encoder.

These functions run on CPU before the compiled vision model. They compute:
- 2D rotary position embeddings (RoPE) from grid_thw
- Bilinear interpolation indices/weights for position embeddings
- Attention bounds (bound_min, bound_max) from grid_thw

All functions are pure tensor operations with no learnable parameters.
The actual embedding lookup runs on-device via ``Qwen3VLVisionPosEmbed``.

Reference: HF ``Qwen3VLVisionModel.rot_pos_emb``,
``fast_pos_embed_interpolate``, and ``cu_seqlens`` computation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_rotary_pos_emb(
    grid_thw: torch.Tensor,
    head_dim: int,
    spatial_merge_size: int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 2D rotary position embeddings for vision attention.

    Each token gets a 2D position (row, col) within its image. The RoPE
    frequencies are looked up from a precomputed table and concatenated
    for the two spatial dimensions.

    Matches HF ``Qwen3VLVisionModel.rot_pos_emb`` + the ``emb = cat(...)``
    and ``(emb.cos(), emb.sin())`` in the forward method.

    Args:
        grid_thw: ``[num_images, 3]`` — (temporal, height, width) per image.
        head_dim: Attention head dimension.
        spatial_merge_size: Spatial merge factor (patches are grouped in
            ``merge_size × merge_size`` blocks before the merger).
        theta: RoPE base frequency. Default: 10000.0.

    Returns:
        ``(cos, sin)`` each ``[total_tokens, head_dim]``.
    """
    half_dim = head_dim // 2

    # Frequency table: [max_hw, half_dim]
    max_hw = int(grid_thw[:, 1:].max().item())
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half_dim, 2, dtype=torch.float32) / half_dim)
    )
    seq = torch.arange(max_hw, dtype=torch.float32)
    freq_table = torch.outer(seq, inv_freq)  # [max_hw, half_dim // 2]

    total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long)

    offset = 0
    for num_frames, height, width in grid_thw:
        num_frames = int(num_frames.item())
        height = int(height.item())
        width = int(width.item())
        merged_h = height // spatial_merge_size
        merged_w = width // spatial_merge_size

        block_rows = torch.arange(merged_h)
        block_cols = torch.arange(merged_w)
        intra_row = torch.arange(spatial_merge_size)
        intra_col = torch.arange(spatial_merge_size)

        # Full-resolution positions within the merge grid
        row_idx = (
            block_rows[:, None, None, None] * spatial_merge_size
            + intra_row[None, None, :, None]
        )
        col_idx = (
            block_cols[None, :, None, None] * spatial_merge_size
            + intra_col[None, None, None, :]
        )

        row_idx = row_idx.expand(
            merged_h, merged_w, spatial_merge_size, spatial_merge_size
        ).reshape(-1)
        col_idx = col_idx.expand(
            merged_h, merged_w, spatial_merge_size, spatial_merge_size
        ).reshape(-1)

        coords = torch.stack((row_idx, col_idx), dim=-1)  # [h*w, 2]

        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)

        num_tokens = coords.shape[0]
        pos_ids[offset : offset + num_tokens] = coords
        offset += num_tokens

    # Lookup and flatten: [total_tokens, 2, half_dim//2] → [total_tokens, half_dim]
    embeddings = freq_table[pos_ids]  # [total_tokens, 2, half_dim//2]
    rotary_pos_emb = embeddings.flatten(1)  # [total_tokens, half_dim]

    # Double and compute cos/sin (matches HF: emb = cat(rotary, rotary))
    emb = torch.cat(
        (rotary_pos_emb, rotary_pos_emb), dim=-1
    )  # [total_tokens, head_dim]
    cos = emb.cos()
    sin = emb.sin()

    return cos, sin


def compute_position_indices_and_weights(
    grid_thw: torch.Tensor,
    num_grid_per_side: int,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute bilinear interpolation indices and weights for position embeddings.

    For each image, computes the 4 bilinear corner indices and weights on the
    learned embedding grid, applies time-repeat and spatial-merge permutation.
    The actual embedding lookup is deferred to the Neuron device.

    Matches HF ``Qwen3VLVisionModel.fast_pos_embed_interpolate``
    but returns raw indices and weights instead of the embedded result.

    Args:
        grid_thw: ``[num_images, 3]`` — (temporal, height, width) per image.
        num_grid_per_side: ``int(sqrt(num_position_embeddings))`` — grid side
            length of the learned embedding table (e.g., 48 for 2304 entries).
        spatial_merge_size: Spatial merge factor (e.g., 2).

    Returns:
        ``(idx_tensor, weight_tensor)`` where:
        - ``idx_tensor``: ``[4, total_tokens]`` int32 — indices into the
          position embedding table for the 4 bilinear corners.
        - ``weight_tensor``: ``[4, total_tokens]`` bf16 — bilinear weights.
    """
    idx_chunks: list[torch.Tensor] = []
    weight_chunks: list[torch.Tensor] = []

    for t, h, w in zip(grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]):
        t_val, h_val, w_val = int(t.item()), int(h.item()), int(w.item())

        indices, weights = _compute_bilinear_indices_and_weights(
            h_val, w_val, num_grid_per_side
        )
        indices = _apply_time_repeat_and_spatial_merge(
            indices, t_val, h_val, w_val, spatial_merge_size
        )
        weights = _apply_time_repeat_and_spatial_merge(
            weights, t_val, h_val, w_val, spatial_merge_size
        )

        idx_chunks.append(indices)
        weight_chunks.append(weights)

    idx_tensor = torch.cat(idx_chunks, dim=1).to(torch.int32)
    weight_tensor = torch.cat(weight_chunks, dim=1).to(torch.bfloat16)

    return idx_tensor, weight_tensor


def _compute_bilinear_indices_and_weights(
    h: int, w: int, num_grid_per_side: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute bilinear interpolation indices and weights for a single (h, w) grid.

    Returns:
        indices: ``[4, h*w]`` — grid indices for the 4 bilinear corners.
        weights: ``[4, h*w]`` — bilinear weights for the 4 corners.
    """
    h_idxs = torch.linspace(0, num_grid_per_side - 1, h)
    w_idxs = torch.linspace(0, num_grid_per_side - 1, w)

    h_floor = h_idxs.int()
    w_floor = w_idxs.int()
    h_ceil = (h_floor + 1).clip(max=num_grid_per_side - 1)
    w_ceil = (w_floor + 1).clip(max=num_grid_per_side - 1)

    dh = (h_idxs - h_floor).unsqueeze(1)  # (h, 1)
    dw = (w_idxs - w_floor).unsqueeze(0)  # (1, w)

    base_h = h_floor * num_grid_per_side
    base_h_ceil = h_ceil * num_grid_per_side

    indices = torch.stack(
        [
            (base_h[:, None] + w_floor[None, :]).flatten(),
            (base_h[:, None] + w_ceil[None, :]).flatten(),
            (base_h_ceil[:, None] + w_floor[None, :]).flatten(),
            (base_h_ceil[:, None] + w_ceil[None, :]).flatten(),
        ]
    )  # (4, h*w)

    weights = torch.stack(
        [
            ((1 - dh) * (1 - dw)).flatten(),
            ((1 - dh) * dw).flatten(),
            (dh * (1 - dw)).flatten(),
            (dh * dw).flatten(),
        ]
    )  # (4, h*w)

    return indices, weights


def _apply_time_repeat_and_spatial_merge(
    tensor: torch.Tensor,
    t: int,
    h: int,
    w: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Repeat by temporal dim and apply spatial-merge permutation.

    Args:
        tensor: ``[4, h*w]`` — index or weight tensor for the 4 corners.
        t: Temporal frames.
        h: Height (grid tokens).
        w: Width (grid tokens).
        spatial_merge_size: Merge factor (e.g., 2).

    Returns:
        ``[4, t*h*w]`` — with time-repeat and spatial-merge permutation applied.
    """
    m = spatial_merge_size
    # (4, h*w) → repeat along seq dim → (4, t*h*w)
    repeated = tensor.unsqueeze(1).expand(-1, t, -1).reshape(4, t * h * w)
    # Apply spatial-merge permutation
    repeated = (
        repeated.view(4, t, h // m, m, w // m, m)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .reshape(4, t * h * w)
    )
    return repeated


def compute_attention_bounds(
    grid_thw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token attention bounds from grid_thw.

    Each token attends only to tokens from the same image/frame group.
    Converts the ``cu_seqlens`` representation (used by HF) to per-token
    ``(bound_min, bound_max)`` format (used by our attention module).

    Args:
        grid_thw: ``[num_images, 3]`` — (temporal, height, width) per image.

    Returns:
        ``(bound_min, bound_max)`` each ``[total_tokens, 1]`` int32.
        Token ``i`` attends to positions ``[bound_min[i], bound_max[i])``.
    """
    # Compute cu_seqlens (same as HF)
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    total_tokens = int(cu_seqlens[-1].item())
    bound_min = torch.zeros(total_tokens, 1, dtype=torch.int32)
    bound_max = torch.zeros(total_tokens, 1, dtype=torch.int32)

    for i in range(len(cu_seqlens) - 1):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        bound_min[start:end] = start
        bound_max[start:end] = end

    return bound_min, bound_max
