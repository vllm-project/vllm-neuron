# SPDX-License-Identifier: Apache-2.0
"""On-device position and slot_mapping correction for async spec decode.

In async spec decode, the scheduler optimistically assumes all drafts from
the previous step were accepted when scheduling the current step. The
CPU-built ``positions`` passed to the target verify NEFF reflect this
optimism. This module provides a functional helper that reads the previous
step's rejection sampler output on device, computes the per-request
rejection count, and subtracts it from ``positions`` for the matching
tokens. It then recomputes ``slot_mapping`` from the corrected positions.

Runs inside the compile boundary, so no CPU sync is needed — the caller can
dispatch the target NEFF while the previous step's output is still in
flight, and the Neuron runtime handles the data dependency on
``prev_sampled_token_ids`` transparently.
"""

from __future__ import annotations

import torch


def correct_spec_decode_positions_and_slot_mapping(
    positions: torch.Tensor,
    attn_metadata: dict,
    prev_sampled_token_ids: torch.Tensor,
    prev_num_draft_tokens: torch.Tensor,
    req_indices_per_token: torch.Tensor,
    vocab_size: int,
) -> tuple[torch.Tensor, dict]:
    """Correct positions and slot_mapping on device for async spec decode.

    For the first decode step after prefill (no prior spec step), callers
    pass a dummy ``prev_sampled_token_ids`` filled with valid entries so
    ``valid_count == 1 + num_spec`` and the correction is a no-op.

    Args:
        positions: Optimistic positions ``[total_scheduled]`` int32,
            computed from the scheduler's pre-increment num_computed_tokens.
        attn_metadata: Per-layer attention metadata dict. Each entry's
            ``slot_mapping`` is replaced with the corrected one.
        prev_sampled_token_ids: Previous step's rejection sampler output
            ``[bs, num_spec+1]`` int32, with ``-1`` for rejected positions.
        prev_num_draft_tokens: Previous step's draft count per request
            ``[bs]`` int32.
        req_indices_per_token: For each of the ``total_scheduled`` positions,
            which request it belongs to ``[total_scheduled]`` int64.
        vocab_size: Model vocabulary size. Matches the validity criterion
            used by the rejection sampler.

    Returns:
        ``(corrected_positions, attn_metadata)`` — positions shape is the
        same ``[total_scheduled]`` int32; attn_metadata has its
        slot_mapping entries replaced in-place.

    Example:
        >>> positions = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
        >>> prev_sampled = torch.tensor([[100, 200, -1, -1]], dtype=torch.int32)
        >>> prev_num_draft = torch.tensor([3], dtype=torch.int32)
        >>> req_indices = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
        >>> attn_meta = {"layers.0.self_attn": {
        ...     "block_table_tensor": torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        ...     "block_size": 16,
        ... }}
        >>> corrected_positions, _ = correct_spec_decode_positions_and_slot_mapping(
        ...     positions, attn_meta, prev_sampled, prev_num_draft,
        ...     req_indices, vocab_size=200000,
        ... )
        >>> # valid_count = 2, num_rejected = 4 - 2 = 2.
        >>> # corrected_positions = [8, 9, 10, 11].
    """
    valid_mask = (prev_sampled_token_ids != -1) & (prev_sampled_token_ids < vocab_size)
    valid_count = valid_mask.sum(dim=1).to(torch.int32)  # [bs]
    num_rejected = (1 + prev_num_draft_tokens.to(torch.int32)) - valid_count  # [bs]

    # Expand per-req rejection count to per-scheduled-token.
    per_token_offset = num_rejected[req_indices_per_token]  # [total_scheduled]
    positions = positions - per_token_offset

    # Recompute slot_mapping per layer using corrected positions. For now,
    # recompute per layer — layers in the same KV cache group share the
    # same block_table_tensor object and would produce identical
    # slot_mapping; torch.compile's CSE should elide the redundant work.
    new_attn_metadata = {}
    for layer_name, meta in attn_metadata.items():
        # Use the untrimmed block table when present (SWA models). The
        # SWA-trimmed ``block_table_tensor`` only covers the sliding
        # window, so corrected positions outside that window would index
        # out of range. Non-SWA metadata only has ``block_table_tensor``,
        # so the .get() fallback keeps the existing behavior.
        block_table_tensor = meta.get(
            "full_block_table_tensor", meta["block_table_tensor"]
        )
        block_size = meta["block_size"]
        slot_mapping = _compute_slot_mapping_per_token(
            positions, block_table_tensor, req_indices_per_token, block_size
        )
        new_meta = dict(meta)
        new_meta["slot_mapping"] = slot_mapping
        new_attn_metadata[layer_name] = new_meta
    return positions, new_attn_metadata


def _compute_slot_mapping_per_token(
    positions: torch.Tensor,
    block_table_tensor: torch.Tensor,
    req_indices_per_token: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Compute slot_mapping on-device for per-token (batch-verify) layout.

    Each scheduled token maps to a KV cache slot via:
        slot = block_id * block_size + (position % block_size)
    where ``block_id = block_table[req_idx, position // block_size]``.

    This generalizes the single-position-per-request pattern used by the
    draft recurrent loop to the multi-position-per-request layout of the
    target verify NEFF (1 + num_spec tokens per request).

    Args:
        positions: Token positions ``[total_scheduled]`` int32/int64.
        block_table_tensor: ``[bs, max_blocks_per_seq]`` int32.
        req_indices_per_token: Request index per token ``[total_scheduled]``
            int64.
        block_size: Number of slots per block.

    Returns:
        Slot indices ``[total_scheduled]`` int64. ``-1`` where the looked-up
        block is the inactive-block sentinel (``-1``), so the attention
        kernel DMA-skips that position.
    """
    block_numbers = positions.to(torch.int64) // block_size  # [total_scheduled]
    max_blocks = block_table_tensor.shape[1]
    flat_idx = req_indices_per_token * max_blocks + block_numbers
    block_ids = block_table_tensor.view(-1)[flat_idx]  # [total_scheduled]
    slot = block_ids.to(torch.int64) * block_size + (
        positions.to(torch.int64) % block_size
    )
    return torch.where(block_ids < 0, torch.full_like(slot, -1), slot)
