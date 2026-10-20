# SPDX-License-Identifier: Apache-2.0
"""Pure utility functions for speculative decoding in vLLM Neuron.

These functions are extracted from NeuronModelRunner to enable unit testing
without Neuron hardware dependencies.
"""

import torch

from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


def extract_next_token_ids(
    sampled_token_ids: torch.Tensor | list[list[int]],
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the last valid token ID and valid count per request from sampler output.

    The sampler output may be a 2D tensor (spec decode, shape [num_reqs, max_tokens])
    with -1 for rejected positions, or a list of lists (non-spec decode).

    Args:
        sampled_token_ids: Sampled tokens, either a tensor with -1 for invalid
            entries or a list of token ID lists per request.
        vocab_size: Model vocabulary size. Tokens >= vocab_size are treated as
            invalid. Defensive guard from upstream eagle implementation against
            OOB token IDs from accumulation errors in sampling cumsum.

    Returns:
        Tuple of (next_token_ids, valid_sampled_tokens_count), both int32 tensors
        of shape [num_reqs].

    Example:
        >>> ids = torch.tensor([[10, 20, -1], [5, -1, -1]])
        >>> next_ids, counts = extract_next_token_ids(ids, 32000)
        >>> next_ids.tolist()
        [20, 5]
        >>> counts.tolist()
        [2, 1]
    """
    if isinstance(sampled_token_ids, torch.Tensor):
        if sampled_token_ids.ndim == 1:
            sampled_token_ids = sampled_token_ids.unsqueeze(1)
        valid_mask = (sampled_token_ids != -1) & (sampled_token_ids < vocab_size)
        valid_count = valid_mask.sum(dim=1).to(torch.int32)
        last_valid_idx = torch.clamp(valid_count - 1, min=0)
        next_token_ids = (
            sampled_token_ids.gather(1, last_valid_idx.unsqueeze(1).to(torch.long))
            .squeeze(1)
            .to(torch.int32)
        )
        return next_token_ids, valid_count
    else:
        valid_count = torch.tensor(
            [len(x) for x in sampled_token_ids], dtype=torch.int32
        )
        next_list = [int(x[-1]) if len(x) > 0 else 0 for x in sampled_token_ids]
        next_token_ids = torch.tensor(next_list, dtype=torch.int32)
        return next_token_ids, valid_count


def compute_token_indices_to_sample(
    last_token_indices: torch.Tensor,
    spec_decode_metadata: SpecDecodeMetadata | None,
    valid_sampled_tokens_count: torch.Tensor,
) -> torch.Tensor:
    """Compute draft model input indices, adjusting for rejected tokens.

    When spec_decode_metadata is None (prefill or non-spec decode), the indices
    are the same as last_token_indices. When present, we subtract the number of
    rejected tokens per request so the draft model sees the correct position.

    Args:
        last_token_indices: Index of last token per request, shape [num_reqs].
        spec_decode_metadata: Spec decode metadata with num_draft_tokens, or None.
        valid_sampled_tokens_count: Number of valid (accepted) tokens per request.

    Returns:
        Adjusted token indices, shape [num_reqs].

    Example:
        >>> indices = torch.tensor([3, 7])
        >>> compute_token_indices_to_sample(indices, None, torch.tensor([1, 1]))
        tensor([3, 7])
    """
    if spec_decode_metadata is None:
        return last_token_indices

    num_rejected = []
    for i, n_draft in enumerate(spec_decode_metadata.num_draft_tokens):
        if n_draft > 0:
            num_rejected.append(int(n_draft + 1 - valid_sampled_tokens_count[i].item()))
        else:
            num_rejected.append(0)
    return last_token_indices - torch.tensor(num_rejected, dtype=torch.long)


def replicate_per_seq_rows(
    per_seq: torch.Tensor,
    num_repeats: int,
) -> torch.Tensor:
    """Repeat each row of a per-sequence tensor ``num_repeats`` consecutive times.

    Equivalent to ``torch.repeat_interleave(per_seq, num_repeats, dim=0)`` but
    implemented with ``torch.arange`` + ``index_select`` so it traces cleanly
    on Neuron's XLA backend. ``index_select`` is a single XLA primitive that
    produces a contiguous output, avoiding the contiguity assertions that
    ``repeat_interleave`` and view-based idioms (``unsqueeze`` + ``expand`` +
    ``reshape``) trigger on Neuron tensors.

    Args:
        per_seq: Per-sequence tensor of shape ``[num_seqs, ...]``.
        num_repeats: Number of consecutive copies to make per row. Must be
            ``>= 1``. ``1`` is a no-op fast path that returns ``per_seq``
            unchanged (no allocation).

    Returns:
        Tensor of shape ``[num_seqs * num_repeats, ...]``, same dtype and
        device as ``per_seq``. Row ``i * num_repeats + j`` of the output equals
        row ``i`` of the input for all ``j in [0, num_repeats)``.

    Raises:
        ValueError: if ``num_repeats < 1``.

    Example:
        >>> per_seq = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> replicate_per_seq_rows(per_seq, 3)
        tensor([[1., 2.],
                [1., 2.],
                [1., 2.],
                [3., 4.],
                [3., 4.],
                [3., 4.]])
    """
    if num_repeats < 1:
        raise ValueError(f"num_repeats must be >= 1, got {num_repeats}")
    if num_repeats == 1:
        return per_seq
    n_seq = per_seq.shape[0]
    idx = torch.arange(n_seq * num_repeats, device=per_seq.device).div(
        num_repeats, rounding_mode="floor"
    )
    return per_seq.index_select(0, idx)
