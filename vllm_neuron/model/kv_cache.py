# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

import torch
from vllm.v1.kv_cache_interface import MLAAttentionSpec


class KeyOnlyAttentionSpec(MLAAttentionSpec):
    """Key-only KV cache spec that will not merge caches of different widths.

    ``MLAAttentionSpec`` budgets one vector per token rather than a K/V pair,
    which is what a key-only cache needs. Its ``merge`` asserts on the
    quantization fields but takes ``head_size``/``num_kv_heads``/``dtype`` from
    the first spec, and vLLM decides two layers share a KV cache group by
    calling ``merge`` and checking whether it raises
    (``kv_cache_utils.is_kv_cache_spec_uniform``). A 576-wide MLA cache and a
    128-wide indexer cache therefore merge silently into one 576-wide group,
    and every layer in that group is budgeted AND shaped at 576 -- the indexer
    both wastes 448 columns and rejects its own 128-wide write.

    Asserting on the width keeps differing caches out of one group, so vLLM
    falls through to ``UniformTypeKVCacheSpecs``, which budgets each layer at
    its own page size and exposes per-layer specs to the worker.
    """

    @classmethod
    def merge(cls, specs):
        widths = {(s.num_kv_heads, s.head_size, s.dtype) for s in specs}
        assert len(widths) == 1, (
            "KV caches of different widths cannot share a cache group: "
            f"{sorted(str(w) for w in widths)}"
        )
        return super().merge(specs)


@dataclass
class LayerSpec:
    """
    Defines the KV cache specification for a single transformer layer.

    Used to specify the memory requirements and configuration for storing
    key-value pairs in the attention mechanism of a transformer layer.
    """

    name: str
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    sliding_window_size: int | None = None
    chunk_size: int | None = None
    # One vector per token instead of a K/V pair, budgeted and allocated as a
    # single cache. Set by MLA layers, which store one combined
    # [rope | compressed_kv] entry, and by key-only caches such as the DeepSeek
    # V3.2 sparse-attention indexer. bind_kv_cache receives a 1-element list.
    key_only: bool = False


@dataclass
class StateLayerSpec:
    """
    Defines a per-request recurrent-state cache for a non-attention layer
    (e.g. a Gated DeltaNet / Mamba-style layer).

    Each request owns one fixed-size "page" per layer, independent of context
    length. A page packs ``shapes[i]`` tensors of ``dtypes[i]`` back to back, in
    order (per-rank shapes, i.e. already divided by the TP degree).
    """

    name: str
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype, ...]
    # vLLM MambaAttentionBackendEnum member that names the layer type in its MambaSpec.
    mamba_type: str = "GDN_ATTN"


@dataclass
class KVSpec:
    """
    Defines the KV cache needs of a model by specifying all layer configurations.

    Contains a list of LayerSpec objects that collectively define the complete
    KV cache requirements for an entire transformer model.
    """

    layers: list[LayerSpec]
    # Recurrent-state layers of hybrid models. Additive: models without such
    # layers leave this empty and are unaffected.
    state_layers: list[StateLayerSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Paged-KV slot arithmetic
# ---------------------------------------------------------------------------
# Write-side padding target: the code redirects an invalid slot to block 0. The
# scheduler reserves block 0 as the null block for this purpose. This mirrors
# ``NULL_BLOCK_ID`` in ``neuron_model_runner``, whose context-parallel path
# already does ``np.where(oob_mask, NULL_BLOCK_ID, slot_mapping)``.
_NULL_BLOCK_ID = 0


def paged_slot_indices(
    slot_mapping: torch.Tensor,
    block_size: int,
    num_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive bounds-checked ``(block_indices, position_indices)`` for a KV write.

    The scheduler marks an inactive write slot with ``PAD_SLOT_ID`` (-1). A
    negative entry is not safe to use as a cache coordinate. ``-1 // block_size``
    floor-divides to ``-1``, not to ``0``. Eager torch wraps that ``-1`` to the
    LAST block and corrupts the KV of a live sequence. On Neuron the same
    ``index_put_`` becomes an HLO ``scatter`` that runs as an indirect vector-DGE
    copy, so ``-1`` becomes an out-of-range HBM address. The runtime then raises
    "scatter/gather (indirect memory copy via vector DGE) out-of-bound access",
    which wedges the NEFF execution queue and stops the engine core.

    This function redirects an entry outside ``[0, num_blocks * block_size)`` to
    the null block before it splits the slot into a block and a position. The
    write for that token still happens, and the null block discards it. A
    redirect keeps the returned shapes static, so the enclosing ``index_put_``
    stays one traceable operation.

    Call this function instead of a direct ``//`` and ``%`` on a slot mapping.
    It clamps before it derives both coordinates. A clamp after the ``//`` and
    the ``%`` leaves ``position_indices`` at ``block_size - 1`` for a pad entry.

    Args:
        slot_mapping: ``[num_tokens]`` flat slot ids, possibly padded with -1.
        block_size: Tokens per KV-cache block.
        num_blocks: Block count of the target cache (its dim-0 extent).

    Returns:
        ``(block_indices, position_indices)``. Both have the shape and the dtype
        of ``slot_mapping``. Every value is in range for ``num_blocks`` blocks.

    Example:
        >>> import torch
        >>> slots = torch.tensor([0, 5, -1, 9])  # -1 is PAD_SLOT_ID; 9 overflows
        >>> blocks, positions = paged_slot_indices(slots, 4, 2)
        >>> blocks
        tensor([0, 1, 0, 0])
        >>> positions
        tensor([0, 1, 0, 0])
    """
    max_slot = num_blocks * block_size
    safe_slots = torch.where(
        (slot_mapping < 0) | (slot_mapping >= max_slot),
        torch.full_like(slot_mapping, _NULL_BLOCK_ID),
        slot_mapping,
    )
    return safe_slots // block_size, safe_slots % block_size


def build_per_head_slot_mapping(
    slot_mapping: torch.Tensor,
    block_size: int,
    kv_heads: int,
    batch_size: int,
) -> torch.Tensor:
    """Build the per-head ``kv_cache_update_idx`` of the in-kernel KV cache update.

    ``update_cache=True`` makes the fused decode kernel write the new K/V into the
    paged cache itself, so the caller does no scatter. For block KV the kernel wants
    physical slots, and the shape depends on the head count
    (``attention_block_tkg`` docstring):

    * ``kv_heads == 1``: ``[B, S_tkg]``
    * ``kv_heads >  1``: ``[B, kv_heads, S_tkg]``

    A slot must index the SAME flat pool that ``build_per_head_block_table`` targets:
    the kernel folds the 4D cache to ``num_blocks * kv_heads`` blocks, so the global
    block of ``(base_block, h)`` is ``base_block * kv_heads + h``. A slot is then
    ``global_block * block_size + position_in_block``. This function and
    ``build_per_head_block_table`` (in ``qwen3_vl.utils.decode_kv``) must agree.

    Args:
        slot_mapping: ``[B * S_tkg]`` flat slots, ``base_block * block_size + pos``.
        block_size: Tokens in one cache block.
        kv_heads: KV heads of this rank.
        batch_size: Requests of this decode step.

    Returns:
        ``[B, kv_heads, S_tkg]`` slots, or ``[B, S_tkg]`` when ``kv_heads == 1``.

    Example:
        >>> import torch
        >>> # 1 request, 1 token at base block 2 position 3, block_size 4
        >>> slots = torch.tensor([2 * 4 + 3])
        >>> build_per_head_slot_mapping(slots, block_size=4, kv_heads=2, batch_size=1)
        tensor([[[19],
                 [23]]])
    """
    if kv_heads == 1:
        return slot_mapping.view(batch_size, -1)

    base_block = torch.div(slot_mapping, block_size, rounding_mode="floor")
    position = slot_mapping - base_block * block_size

    kv_h = torch.arange(
        kv_heads, dtype=slot_mapping.dtype, device=slot_mapping.device
    ).view(1, kv_heads)
    # [B*S, kv_heads] -> global slot of each (token, head)
    global_block = base_block.unsqueeze(1) * kv_heads + kv_h
    global_slot = global_block * block_size + position.unsqueeze(1)
    # [B*S, kv_heads] -> [B, kv_heads, S_tkg]
    return global_slot.view(batch_size, -1, kv_heads).transpose(1, 2).contiguous()
