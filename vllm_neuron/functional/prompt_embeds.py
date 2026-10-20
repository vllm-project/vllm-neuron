# SPDX-License-Identifier: Apache-2.0
"""Shared prompt embeddings merge utility for model backbones."""

import torch


def merge_prompt_embeds(
    hidden_states: torch.Tensor,
    inputs_embeds: torch.Tensor | None,
    is_token_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Merge token-path and prompt-embed-path hidden states.

    ``is_token_ids`` is the selector: True keeps ``hidden_states`` from
    ``embed_tokens``; False swaps in ``inputs_embeds``.
    Caller is responsible for SP alignment before calling this helper.
    """
    if inputs_embeds is None or is_token_ids is None:
        return hidden_states

    is_token_mask = is_token_ids.unsqueeze(-1)
    return torch.where(
        is_token_mask, hidden_states, inputs_embeds.to(hidden_states.dtype)
    )
