# SPDX-License-Identifier: Apache-2.0
"""Tensor alignment utilities for shape normalization before comparison."""

import logging
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)


# Modules where HF recomputes full sequence but only the tail is relevant.
TAKE_LAST_MODULES = {"lm_head"}


def hf_reference_reconstruction(
    rank_tensors: List[torch.Tensor],
    module_name: str,
    phase: str = "prefill",
    positions: List[int] = None,
) -> torch.Tensor:
    """Default reconstruction for HF reference (single rank, no parallelism).

    For decode steps, takes only the last token since HF recomputes full sequence.

    Args:
        rank_tensors: List of per-rank tensors (single element for HF)
        module_name: Name of the module
        phase: "prefill" or "decode" (extensible for future phases)

    Returns:
        Reconstructed tensor

    Example:
        >>> t = torch.randn(1, 10, 64)
        >>> hf_reference_reconstruction([t], "layer.0", phase="decode").shape
        torch.Size([1, 1, 64])
    """
    tensor = rank_tensors[0]
    is_prefill = phase == "prefill"
    if not is_prefill and tensor.dim() >= 2:
        return tensor[:, -1:, :] if tensor.dim() == 3 else tensor[-1:, :]
    if module_name in TAKE_LAST_MODULES:
        return tensor[:, -1:, :] if tensor.dim() == 3 else tensor[-1:, :]
    return tensor


def align_and_truncate_hidden(
    baseline: torch.Tensor,
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Align tensors for comparison: truncate hidden dim only.

    Capture reconstruction strips bucket padding and gathers SP shards,
    so sequence dimensions already match. Only hidden dim truncation is
    needed to handle Neuron's padded hidden size.
    """
    tensors = _promote_and_truncate_hidden([baseline, expected, actual])
    match = all(t.shape == tensors[0].shape for t in tensors)
    return tensors[0], tensors[1], tensors[2], match


# --- Private helpers ---


def _promote_and_truncate_hidden(
    tensors: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Promote 2D→3D, truncate hidden dim, squeeze batch=1, cast float32."""
    # Promote 2D to 3D
    max_dim = max(t.dim() for t in tensors)
    if max_dim == 3:
        tensors = [t.unsqueeze(0) if t.dim() == 2 else t for t in tensors]

    # Truncate hidden dim to min
    min_h = min(t.shape[-1] for t in tensors)
    if any(t.shape[-1] != min_h for t in tensors):
        tensors = [t[..., :min_h] for t in tensors]

    # Squeeze batch=1, cast float32
    tensors = [t.squeeze(0) if t.dim() == 3 and t.shape[0] == 1 else t for t in tensors]
    return [t.float() for t in tensors]


def get_seq_dim_size(tensor: torch.Tensor) -> int:
    """Get sequence dimension size from tensor.

    Args:
        tensor: Input tensor (2D or 3D)

    Returns:
        Sequence dimension size

    Example:
        >>> get_seq_dim_size(torch.randn(1, 10, 64))
        10
    """
    if tensor.dim() == 3:
        return tensor.shape[1]
    elif tensor.dim() == 2:
        return tensor.shape[0]
    return 0


def slice_token(tensor: torch.Tensor, token_idx: int) -> torch.Tensor:
    """Extract a single token from a tensor along the sequence dimension.

    Args:
        tensor: Input tensor (2D or 3D)
        token_idx: Token index to extract

    Returns:
        Tensor with single token in sequence dimension

    Example:
        >>> t = torch.randn(1, 5, 8)
        >>> slice_token(t, 2).shape
        torch.Size([1, 1, 8])
    """
    if tensor.dim() == 3:
        return tensor[:, token_idx : token_idx + 1, :]
    return tensor[token_idx : token_idx + 1, :]


def count_real_tokens(positions: List[int]) -> int:
    """Count real (non-padding) tokens from a position list.

    Neuron padding repeats the last real position (e.g. [0,1,2,2,2]).
    Real tokens have strictly increasing positions from the start.
    """
    return sum(1 for i, p in enumerate(positions) if i == 0 or p > positions[i - 1])


def align_decode_captures(ref_captures: List, target_captures: List) -> List:
    """Align reference captures with target by sorting and filtering by position.

    HF and Neuron may produce different numbers of decode captures. This
    function matches them by token position so only comparable decode steps
    are included in the comparison.

    1. Separates prefill and decode captures
    2. Sorts decodes by max position for deterministic order
    3. Keeps only ref decode steps whose position has a matching target decode

    Args:
        ref_captures: Reference captures (e.g., HF FP32 or BF16)
        target_captures: Target captures (e.g., Neuron) to match against

    Returns:
        Filtered ref_captures with only matching decode positions
    """
    ref_prefills = [c for c in ref_captures if c.is_prefill]
    ref_decodes = [c for c in ref_captures if not c.is_prefill]
    target_decodes = [c for c in target_captures if not c.is_prefill]

    ref_decodes.sort(key=lambda c: max(c.metadata.positions))
    target_decodes.sort(key=lambda c: max(c.metadata.positions))

    if not target_decodes or not ref_decodes:
        return ref_prefills + ref_decodes

    if len(ref_decodes) != len(target_decodes):
        logger.warning(
            "Decode step count mismatch: ref has %d, target has %d.",
            len(ref_decodes),
            len(target_decodes),
        )

    target_positions = {max(c.metadata.positions) for c in target_decodes}
    ref_positions = {max(c.metadata.positions) for c in ref_decodes}
    missing_in_target = ref_positions - target_positions
    if missing_in_target:
        logger.warning(
            "Dropping %d ref decode step(s) with no matching target position: %s",
            len(missing_in_target),
            sorted(missing_in_target),
        )

    ref_decodes = [
        c for c in ref_decodes if max(c.metadata.positions) in target_positions
    ]

    return ref_prefills + ref_decodes
