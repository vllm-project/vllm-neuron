# SPDX-License-Identifier: Apache-2.0
"""Per-head decode block table for GQA (copied from ``qwen3_vl/utils/decode_kv.py``).

The fused decode megakernel takes GQA (KV heads per rank > 1) only with a
per-head 3D block table over a 4D KV cache. Kept local so the contrib model
does not import from another model's package.
"""

from __future__ import annotations

import torch


def build_per_head_block_table(
    block_table_2d: torch.Tensor, kv_heads: int
) -> torch.Tensor:
    """Expand a 2D ``[B, num_blocks]`` block table into the per-head 3D table the
    GQA fused decode mega-kernel requires: ``[B, kv_heads, num_blocks]``.

    The kernel squeezes the 4D KV cache head-inner into a ``[num_blocks*kv_heads]``
    pool, so each ``(batch, head)`` entry must be the GLOBAL pool index
    ``base*kv_heads + h``; ``-1`` (inactive) is preserved unchanged. 

    Expects a 2D ``[B, num_blocks]`` table with ``-1`` as the inactive sentinel.

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
