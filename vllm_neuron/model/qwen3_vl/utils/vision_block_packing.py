# SPDX-License-Identifier: Apache-2.0
"""Block packing utilities for the Qwen3-VL vision encoder.

Packs variable-size images into fixed-size blocks for compile stability.
Each function is independently testable with explicit parameters.

Layers:
  1. ffd_pack_images — pure FFD bin-packing algorithm
  2. scatter_to_blocks — scatter flat tensors into block layout
  3. compute_block_bounds — block-local attention bounds
  4. compute_unpack_indices — post-merger gather indices
  5. select_vision_bucket — bucket selection from configured list
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class BlockAssignment:
    """Result of FFD bin packing: which images go in which blocks.

    Attributes:
        bins: bins[block_idx] = list of image indices assigned to that block.
        num_blocks: Total number of blocks.
        block_size: Max tokens per block.
    """

    bins: list[list[int]]
    num_blocks: int
    block_size: int


def ffd_pack_images(
    tokens_per_image: list[int],
    block_size: int,
    num_blocks: int,
    one_item_per_block: bool = False,
    group_ids: list[int] | None = None,
) -> BlockAssignment:
    """Assign images to blocks using First Fit Decreasing.

    Sorts images by token count (descending) and greedily assigns each
    to the first block with enough remaining capacity.

    Args:
        tokens_per_image: Token count for each image.
        block_size: Max tokens per block.
        num_blocks: Number of available blocks.
        one_item_per_block: If True, each group gets its own dedicated
            block(s) — no cross-group sharing. Each item must fit within
            one block (raises otherwise); a group's items pack densely
            across the group's block run. Used by on-device encoder cache
            (one mm_item per block — no cross-item sharing).
        group_ids: Only with one_item_per_block. Items sharing a group id pack
            densely into that group's block run (e.g. a video's per-frame items
            share the video's blocks). Defaults to one group per item.

    Returns:
        BlockAssignment with image-to-block mapping.

    Raises:
        ValueError: If any image exceeds block_size or images don't fit.
    """
    if one_item_per_block:
        return _pack_one_item_per_block(
            tokens_per_image, block_size, num_blocks, group_ids
        )

    for i, t in enumerate(tokens_per_image):
        if t > block_size:
            raise ValueError(
                f"Image {i} has {t} tokens, exceeds block_size={block_size}"
            )

    sorted_indices = sorted(
        range(len(tokens_per_image)),
        key=lambda i: tokens_per_image[i],
        reverse=True,
    )

    bins: list[list[int]] = [[] for _ in range(num_blocks)]
    remaining = [block_size] * num_blocks

    for idx in sorted_indices:
        size = tokens_per_image[idx]
        placed = False
        for b in range(num_blocks):
            if remaining[b] >= size:
                bins[b].append(idx)
                remaining[b] -= size
                placed = True
                break
        if not placed:
            raise ValueError(
                f"Cannot fit image {idx} ({size} tokens) into {num_blocks} "
                f"blocks of size {block_size}. Increase num_blocks or block_size."
            )

    return BlockAssignment(bins=bins, num_blocks=num_blocks, block_size=block_size)


def _pack_one_item_per_block(
    tokens_per_image: list[int],
    block_size: int,
    num_blocks: int,
    group_ids: list[int] | None = None,
) -> BlockAssignment:
    """Pack items such that each group gets its own dedicated block(s).

    No cross-group sharing: a block only holds items from one group. Without
    group_ids each item is its own group (the original one-item-per-block).
    Each item must fit within one block (raises ValueError otherwise); a group
    spans multiple blocks only by packing whole items densely (used to split a
    video into per-frame items so its frames share a block run without ever
    mixing with another video's frames).

    Blocks are assigned sequentially in item input order (no descending sort).
    This simplifies write_block_ids construction: the caller can flatten
    per-item cache block IDs in the same item order and get a 1:1 mapping
    between VE output block index and cache block index, without needing to
    reverse-map through the assignment.
    """
    if group_ids is None:
        group_ids = list(range(len(tokens_per_image)))

    # Each item must fit within one block: a single image/frame cannot be split
    # across blocks or its block-local attention breaks.
    for i, t in enumerate(tokens_per_image):
        if t > block_size:
            raise ValueError(
                f"Item {i} has {t} tokens, exceeds block_size={block_size}. "
                f"A single image/frame must fit within one block so its attention "
                f"stays complete (different blocks do not attend to each other). "
                f"Increase block_size or the vision bucket."
            )

    bins: list[list[int]] = [[] for _ in range(num_blocks)]
    block_cursor = 0
    prev_group: int | None = None
    remaining = 0

    for idx, tokens in enumerate(tokens_per_image):
        # A new group, or an item that no longer fits, starts a fresh block. Each
        # item is guaranteed <= block_size (checked above); a group spans blocks
        # only by starting fresh blocks for items that don't fit. Items of the
        # same group pack densely.
        if group_ids[idx] != prev_group or remaining < tokens:
            blocks_needed = math.ceil(tokens / block_size)
            if block_cursor + blocks_needed > num_blocks:
                raise ValueError(
                    f"Cannot fit item {idx} ({tokens} tokens, needs "
                    f"{blocks_needed} blocks) starting at block {block_cursor} "
                    f"with only {num_blocks} total blocks. "
                    f"Increase num_blocks or block_size."
                )
            for b in range(block_cursor, block_cursor + blocks_needed):
                bins[b].append(idx)
            block_cursor += blocks_needed
            remaining = blocks_needed * block_size - tokens
            prev_group = group_ids[idx]
        else:
            bins[block_cursor - 1].append(idx)
            remaining -= tokens

    return BlockAssignment(bins=bins, num_blocks=num_blocks, block_size=block_size)


def scatter_to_blocks(
    flat_tensor: torch.Tensor,
    tokens_per_image: list[int],
    assignment: BlockAssignment,
) -> torch.Tensor:
    """Scatter a flat per-token tensor into block-packing layout.

    Images are placed into their assigned blocks in image-index order
    within each block. Unused positions are zero-padded.

    Args:
        flat_tensor: [total_tokens, ...] — concatenated across images in order.
        tokens_per_image: Token count per image (sum = total_tokens).
        assignment: Image-to-block mapping from ffd_pack_images.

    Returns:
        [num_blocks, block_size, ...] — zero-padded block layout.
    """
    trailing_shape = flat_tensor.shape[1:]
    out = flat_tensor.new_zeros(
        (assignment.num_blocks, assignment.block_size, *trailing_shape)
    )

    token_offsets = [0]
    for t in tokens_per_image:
        token_offsets.append(token_offsets[-1] + t)

    for b, block_images in enumerate(assignment.bins):
        block_offset = 0
        for img_idx in sorted(block_images):
            src_start = token_offsets[img_idx]
            src_end = token_offsets[img_idx + 1]
            n = src_end - src_start
            out[b, block_offset : block_offset + n] = flat_tensor[src_start:src_end]
            block_offset += n

    return out


def compute_block_bounds(
    tokens_per_image: list[int],
    assignment: BlockAssignment,
    grid_thw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token attention bounds in block-local coordinates.

    Each image's tokens attend only within that image. When an image has
    multiple frames (T > 1), each frame is a separate attention group.
    Bounds are relative to the start of the block.

    Args:
        tokens_per_image: Token count per image.
        assignment: Image-to-block mapping.
        grid_thw: [num_images, 3] — (T, H, W) per image.

    Returns:
        (bound_min, bound_max) each [num_blocks, block_size, 1] int32.
    """
    bound_min = torch.zeros(
        assignment.num_blocks, assignment.block_size, 1, dtype=torch.int32
    )
    bound_max = torch.zeros(
        assignment.num_blocks, assignment.block_size, 1, dtype=torch.int32
    )

    for b, block_images in enumerate(assignment.bins):
        block_offset = 0
        for img_idx in sorted(block_images):
            t = int(grid_thw[img_idx, 0].item())
            h = int(grid_thw[img_idx, 1].item())
            w = int(grid_thw[img_idx, 2].item())
            tokens_per_frame = h * w

            for frame in range(t):
                frame_start = block_offset + frame * tokens_per_frame
                frame_end = frame_start + tokens_per_frame
                bound_min[b, frame_start:frame_end, 0] = frame_start
                bound_max[b, frame_start:frame_end, 0] = frame_end

            block_offset += tokens_per_image[img_idx]

    return bound_min, bound_max


def compute_unpack_indices(
    tokens_per_image: list[int],
    assignment: BlockAssignment,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, int]:
    """Compute gather indices to restore per-image token order after merger.

    After the merger reduces each block from block_size to
    block_size // merge_factor tokens, unpack_indices maps from
    original image order to the flat merged block layout.

    Args:
        tokens_per_image: Token count per image.
        assignment: Image-to-block mapping.
        spatial_merge_size: Merger spatial merge factor (typically 2).

    Returns:
        (unpack_indices, total_merged_tokens) where:
        - unpack_indices: [merged_seq_len] int64, padded to static size.
          unpack_indices[original_position] = flat_merged_position.
        - total_merged_tokens: Number of valid entries.
    """
    merge_factor = spatial_merge_size**2
    merged_block_size = assignment.block_size // merge_factor
    merged_seq_len = assignment.num_blocks * merged_block_size
    total_merged_tokens = sum(t // merge_factor for t in tokens_per_image)

    pad_index = merged_seq_len - 1
    unpack_indices = torch.full((merged_seq_len,), pad_index, dtype=torch.int64)

    merged_offsets = [0]
    for t in tokens_per_image:
        merged_offsets.append(merged_offsets[-1] + t // merge_factor)

    for b, block_images in enumerate(assignment.bins):
        block_merged_offset = 0
        for img_idx in sorted(block_images):
            num_merged = tokens_per_image[img_idx] // merge_factor
            orig_start = merged_offsets[img_idx]
            packed_start = b * merged_block_size + block_merged_offset

            unpack_indices[orig_start : orig_start + num_merged] = torch.arange(
                packed_start, packed_start + num_merged, dtype=torch.int64
            )
            block_merged_offset += num_merged

    return unpack_indices, total_merged_tokens


def select_vision_bucket(
    total_tokens: int,
    buckets: list[int],
    block_size: int,
    dp_size: int = 1,
) -> tuple[int, int]:
    """Select smallest bucket >= total_tokens and derive num_blocks.

    Args:
        total_tokens: Total vision tokens across all images.
        buckets: Configured bucket sizes (sorted ascending, must not be empty).
        block_size: Vision attention block size.
        dp_size: Vision DP degree. num_blocks is padded to be divisible by dp_size.

    Returns:
        (bucket, num_blocks) tuple.

    Raises:
        ValueError: If buckets is empty or total_tokens exceeds largest bucket.
    """
    if not buckets:
        raise ValueError(
            "num_encoder_input_buckets must be configured (non-empty list)"
        )

    sorted_buckets = sorted(buckets)
    for bucket in sorted_buckets:
        if bucket >= total_tokens:
            num_blocks = math.ceil(bucket / block_size)
            # Pad num_blocks to be divisible by dp_size for even scatter
            if dp_size > 1:
                num_blocks = math.ceil(num_blocks / dp_size) * dp_size
            return bucket, num_blocks

    raise ValueError(
        f"Total vision tokens ({total_tokens}) exceeds largest configured "
        f"bucket ({sorted_buckets[-1]}). Increase num_encoder_input_buckets "
        f"or reduce input image size."
    )
