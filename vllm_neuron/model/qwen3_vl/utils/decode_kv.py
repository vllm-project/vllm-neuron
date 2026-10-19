# SPDX-License-Identifier: Apache-2.0
"""Shared decode-time KV helpers for Qwen3-VL GQA (bf16 + native-MXFP8).

GQA decode (kv_heads_per_rank > 1, e.g. TP=4 on the 8-KV-head 32B model) reaches
the fused decode mega-kernel only with a PER-HEAD 3D block table + a 4D KV cache;
a 2D ``[B, num_blocks]`` table routes GQA to the slow torch fallback (see
``_can_use_attention_block_kernel``). These two pure-torch, device-agnostic helpers
build that table and scatter the kernel's 4D per-head new-K/V tokens back into the
paged cache. They live here (a shared utils module, not either model file) so the
bf16 (``model_bf16``) and native-MX (``model_mxfp8``) decode paths and the CPU
three-way reference use one definition and cannot drift.

TODO: move these helpers to a model-neutral shared location once the decode-time
KV contract is generalized enough that dense Qwen3 and Qwen3-VL can depend on the
same module without either model importing through the other's package tree.
"""

from __future__ import annotations

import torch

from ...kv_cache import paged_slot_indices


def build_per_head_block_table(
    block_table_2d: torch.Tensor, kv_heads: int
) -> torch.Tensor:
    """Expand a 2D ``[B, num_blocks]`` block table into the per-head 3D table the
    GQA fused decode mega-kernel requires: ``[B, kv_heads, num_blocks]``.

    The kernel squeezes the 4D KV cache head-inner into a ``[num_blocks*kv_heads]``
    pool, so each ``(batch, head)`` entry must be the GLOBAL pool index
    ``base*kv_heads + h``; ``-1`` (inactive) is preserved unchanged. Single source
    of truth shared by ``Qwen3VLTextAttention.forward_decode`` (bf16),
    ``Qwen3VLTextAttentionMX.forward_decode`` and the CPU three-way reference, so
    the kernel's GQA global-index convention is encoded in exactly one place (they
    cannot drift and mask a shared bug).

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


def scatter_new_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    K_new: torch.Tensor,
    V_new: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    nkh: int,
    head_dim: int,
) -> None:
    """Scatter the kernel's per-head new K/V tokens into the paged KV cache.

    The fused decode kernel returns 4D per-head tensors ``K_new [d_head, B, kv_heads, S]``
    / ``V_new [B, kv_heads, S, d_head]``. (When kv_heads == 1 the kernel drops K's
    head axis to 3D ``[d_head, B, S]``; callers re-insert it before calling here, so
    this function always sees the uniform 4D contract — for kv_heads == 1 the head
    axis is a singleton.) Both are reordered to head-major ``[kv_heads, B*S, d_head]``
    so the ``(head, block, pos)`` index_put enumeration below lands each
    ``(batch, token, head)`` row at its paged slot.

    ``slot_mapping`` is ``[B*S]`` in B-major (batch, token) order; row ``i`` of the
    flattened new K/V is ``(head = i // (B*S), token = i % (B*S))`` and is written to
    ``cache[block_of(token), head, pos_of(token)]``. Extracted as a free function so
    the permute/scatter mapping is unit-testable with known values (no kernel/device).

    Example:
        >>> import torch
        >>> # 1 block of size 4, 1 kv-head, d_head=2, B*S=2 new tokens
        >>> k_cache = torch.zeros(1, 1, 4, 2)
        >>> v_cache = torch.zeros(1, 1, 4, 2)
        >>> K_new = torch.ones(2, 1, 1, 2)   # [d_head, B, kv, S]
        >>> V_new = torch.ones(1, 1, 2, 2)   # [B, kv, S, d_head]
        >>> slots = torch.tensor([0, 1])
        >>> scatter_new_kv(k_cache, v_cache, K_new, V_new, slots, 4, 1, 2)
    """
    assert K_new.dim() == 4 and V_new.dim() == 4, (
        f"scatter_new_kv expects 4D per-head K_new/V_new from the GQA kernel; "
        f"got K_new.dim()={K_new.dim()}, V_new.dim()={V_new.dim()}"
    )
    block_indices, position_indices = paged_slot_indices(
        slot_mapping, block_size, k_cache.shape[0]
    )

    # num_tokens = B*S from the kernel tensors (K_new: [d_head, B, kv, S]). The
    # downstream index_put_ enumeration assumes one slot per (batch, token), so
    # slot_mapping must be exactly the B*S decode write set; the assert enforces
    # that LENGTH contract. Padded *values* inside that length are safe —
    # paged_slot_indices redirects them to the null block — but a padded decode
    # batch that changes the length would still need the scatter reworked.
    _, B, _, S = K_new.shape
    num_tokens = B * S
    assert slot_mapping.shape[0] == num_tokens, (
        f"slot_mapping length ({slot_mapping.shape[0]}) must equal B*S ({num_tokens}) "
        f"from K_new {tuple(K_new.shape)}"
    )

    k_new = K_new.permute(2, 1, 3, 0).reshape(nkh, num_tokens, head_dim)
    v_new = V_new.permute(1, 0, 2, 3).reshape(nkh, num_tokens, head_dim)
    k_new_flat = k_new.reshape(-1, head_dim)
    v_new_flat = v_new.reshape(-1, head_dim)

    head_indices_for_put = torch.arange(
        nkh, dtype=torch.long, device=slot_mapping.device
    ).repeat_interleave(num_tokens)
    block_indices_for_put = block_indices.repeat(nkh)
    position_indices_for_put = position_indices.repeat(nkh)

    k_cache.index_put_(
        (block_indices_for_put, head_indices_for_put, position_indices_for_put),
        k_new_flat.to(k_cache.dtype),
    )
    v_cache.index_put_(
        (block_indices_for_put, head_indices_for_put, position_indices_for_put),
        v_new_flat.to(v_cache.dtype),
    )
