# SPDX-License-Identifier: Apache-2.0
"""Scatter audio encoder embeddings into hidden_states for Qwen3-ASR prefill.

Audio has no deepstack (the encoder-cache width equals the text hidden size), so
the merge uses the shared encoder scatter and drops the always-None deepstack
return. The buffer, the block ids, and the positions come from the runner's
_gather_mm_embeddings via the encoder_* mm_kwargs. This wraps
merge_multimodal_embeddings, one level simpler.
"""

from __future__ import annotations

import torch

from vllm_neuron.functional.merge_multimodal_embeddings import (
    merge_multimodal_embeddings,
)


def merge_audio_embeddings(
    hidden_states: torch.Tensor,
    audio_embedding_buffer: torch.Tensor,
    audio_block_ids: torch.Tensor,
    audio_positions: torch.Tensor,
    rank: torch.Tensor,
) -> torch.Tensor:
    """Scatter audio encoder embeddings from cache blocks into hidden_states.

    This replaces the audio-placeholder rows (audio_token_id) of the token
    embeddings with the encoder outputs, in order. Audio has no deepstack
    (the encoder-cache width equals the text hidden size), so it reuses the
    shared encoder merge and drops the always-None deepstack return.

    Args:
        hidden_states: [local_len, hidden_size] token embeddings (SP-sharded).
        audio_embedding_buffer: [num_blocks, block_size, hidden_size] cache.
        audio_block_ids: [max_blocks] int32 block indices to gather.
        audio_positions: [max_blocks, block_size] int32 global positions.
        rank: SP rank tensor for the coordinate remap. Pass the group
            rank_in_group_tensor. The NKI merge path needs a runtime tensor.

    Returns:
        hidden_states with the audio embeddings scattered into the placeholders.
    """
    if audio_embedding_buffer.shape[-1] != hidden_states.shape[-1]:
        raise ValueError(
            f"Audio encoder cache width ({audio_embedding_buffer.shape[-1]}) must "
            f"equal the text hidden size ({hidden_states.shape[-1]}). Audio has no "
            f"deepstack; size the encoder cache last dim to output_dim."
        )
    merged, _deepstack = merge_multimodal_embeddings(
        hidden_states,
        audio_embedding_buffer,
        audio_block_ids,
        audio_positions,
        rank,
    )
    return merged
