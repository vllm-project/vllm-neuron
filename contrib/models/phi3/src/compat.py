# SPDX-License-Identifier: Apache-2.0
"""Helpers this model needs that vLLM Neuron 0.29.0.1.2.0 does not ship.

They are copied unchanged from a later vLLM Neuron release, so the model
runs on the 0.29 plugin as installed.
"""

import torch


def build_per_head_block_table(
    block_table_2d: torch.Tensor, kv_heads: int
) -> torch.Tensor:
    """Build a per-head 3D block table from a 2D base table.

    The kernel folds the 4D KV cache into a ``[num_blocks * kv_heads]`` pool.
    Each ``(batch, head)`` entry is the global pool index ``base * kv_heads + h``.
    Preserve ``-1`` (inactive) entries unchanged.

    Expects a 2D ``[B, num_blocks]`` table with ``-1`` as the inactive sentinel.
    Returns a ``[B, kv_heads, num_blocks]`` table.

    Example:
        >>> import torch
        >>> t = torch.tensor([[0, 1, -1]])
        >>> build_per_head_block_table(t, kv_heads=2)
        tensor([[[ 0,  2, -1],
                 [ 1,  3, -1]]])
    """
    assert block_table_2d.dim() == 2, (
        f"build_per_head_block_table expects a 2D [B, num_blocks] table, got "
        f"{block_table_2d.dim()}D {tuple(block_table_2d.shape)}"
    )
    base_3d = block_table_2d.unsqueeze(1)  # [B, 1, num_blocks]
    kv_h = torch.arange(
        kv_heads, dtype=block_table_2d.dtype, device=block_table_2d.device
    ).view(1, kv_heads, 1)
    shifted = base_3d * kv_heads + kv_h  # [B, kv_heads, num_blocks]
    return torch.where(base_3d == -1, base_3d, shifted).contiguous()
