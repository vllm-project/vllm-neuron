# SPDX-License-Identifier: Apache-2.0
"""M-RoPE (Multimodal Rotary Position Embedding) position computation.

Computes 3D position IDs [3, seq_len] for Qwen3-VL where the three axes
encode temporal, height, and width information. Text tokens get identical
sequential positions on all 3 axes; vision tokens get spatial grid
coordinates derived from their image/video grid dimensions.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.multimodal.inputs import MultiModalFeatureSpec

if TYPE_CHECKING:
    from vllm_neuron.model.qwen3_vl.config import Qwen3VLConfig


def iter_mm_grid_hw(
    input_tokens: list[int],
    mm_features: list[MultiModalFeatureSpec],
    video_token_id: int,
    vision_start_token_id: int,
    vision_end_token_id: int,
    spatial_merge_size: int,
) -> Iterator[tuple[int, int, int, int]]:
    """Iterate over multimodal features and yield grid position info.

    Processes features in sequence order (sorted by offset). For each image,
    yields a single entry. For each video, yields one entry per temporal frame.

    Args:
        input_tokens: Full token sequence for the prompt.
        mm_features: Multimodal features (image/video) with offset metadata.
        video_token_id: Token ID marking video content.
        vision_start_token_id: Token ID marking start of vision region.
        vision_end_token_id: Token ID marking end of vision region.
        spatial_merge_size: Merge factor for spatial dimensions.

    Yields:
        (offset, llm_grid_h, llm_grid_w, actual_num_tokens):
            offset: Position of the first image/video token in the sequence.
            llm_grid_h: Logical grid height (after spatial merge).
            llm_grid_w: Logical grid width (after spatial merge).
            actual_num_tokens: Actual number of image/video tokens in placeholder.
    """
    for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
        offset = mm_feature.mm_position.offset
        if mm_feature.modality == "image":
            t, h, w = mm_feature.data["image_grid_thw"].data.tolist()
            if t != 1:
                raise ValueError(f"Image must have 1 frame, got {t}")
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size
            yield offset, llm_grid_h, llm_grid_w, llm_grid_h * llm_grid_w
        elif mm_feature.modality == "video":
            t, h, w = mm_feature.data["video_grid_thw"].data.tolist()
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size
            for _ in range(t):
                offset = input_tokens.index(vision_start_token_id, offset)
                vision_end_offset = input_tokens.index(vision_end_token_id, offset)
                try:
                    video_offset = input_tokens.index(
                        video_token_id, offset, vision_end_offset
                    )
                    actual_num_tokens = vision_end_offset - video_offset
                except ValueError:
                    video_offset = offset + 1
                    actual_num_tokens = 0
                yield video_offset, llm_grid_h, llm_grid_w, actual_num_tokens
                offset = vision_end_offset + 1


def compute_mrope_positions(
    input_tokens: list[int],
    mm_features: list[MultiModalFeatureSpec],
    config: "Qwen3VLConfig",
) -> tuple[torch.Tensor, int]:
    """Compute 3D M-RoPE positions [3, seq_len] and position delta.

    Text regions get identical sequential positions on all 3 axes.
    Vision regions get spatial grid positions (temporal, height, width).

    The position delta is `max_position + 1 - seq_len`, representing the
    offset needed to compute decode-phase positions that continue from
    the last prefill position.

    Args:
        input_tokens: Full prompt token IDs.
        mm_features: Multimodal features with grid/offset metadata.
        config: Model config providing special token IDs and spatial_merge_size.

    Returns:
        (positions, delta) where positions is [3, seq_len] int64 tensor
        and delta is the offset for computing decode-phase positions.
    """
    if not input_tokens:
        return torch.zeros((3, 0), dtype=torch.long), 0

    llm_pos_ids_list: list[np.ndarray] = []
    cursor = 0
    for (
        offset,
        llm_grid_h,
        llm_grid_w,
        actual_num_tokens,
    ) in iter_mm_grid_hw(
        input_tokens,
        mm_features,
        video_token_id=config.video_token_id,
        vision_start_token_id=config.vision_start_token_id,
        vision_end_token_id=config.vision_end_token_id,
        spatial_merge_size=config.vision_config.spatial_merge_size,
    ):
        if actual_num_tokens == 0:
            continue

        text_len = offset - cursor
        if text_len < 0:
            raise ValueError(
                f"Overlapping or misordered multimodal feature: "
                f"offset={offset} < current position cursor={cursor}"
            )
        next_pos_idx = (
            llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        )
        if text_len > 0:
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + next_pos_idx
            )

        expected_tokens_per_frame = llm_grid_h * llm_grid_w
        if actual_num_tokens > expected_tokens_per_frame:
            num_logical_frames = actual_num_tokens // expected_tokens_per_frame
            remainder = actual_num_tokens % expected_tokens_per_frame
            for _ in range(num_logical_frames):
                grid_indices = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
                llm_pos_ids_list.append(grid_indices + text_len + next_pos_idx)
                next_pos_idx = llm_pos_ids_list[-1].max() + 1
                text_len = 0
            if remainder > 0:
                full_grid = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
                llm_pos_ids_list.append(
                    full_grid[:, :remainder] + text_len + next_pos_idx
                )
        else:
            grid_indices = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
            llm_pos_ids_list.append(grid_indices + text_len + next_pos_idx)

        cursor = offset + actual_num_tokens

    if cursor < len(input_tokens):
        next_pos_idx = (
            llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        )
        text_len = len(input_tokens) - cursor
        llm_pos_ids_list.append(
            np.broadcast_to(np.arange(text_len), (3, text_len)) + next_pos_idx
        )

    llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
    mrope_position_delta = int(llm_positions.max() + 1 - len(input_tokens))
    return torch.from_numpy(llm_positions.copy()), mrope_position_delta
