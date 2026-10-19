# SPDX-License-Identifier: Apache-2.0
"""
Sharding semantics and deterministic KV fill/verify for DI testing.

Prefill uses ``prefill_fill_kv`` to write deterministic values. Decode uses
``decode_verify_kv_layer`` to check byte-exact correctness after NIXL transfer.
Both sides derive global_head and global_pos independently — no KV connector
data needed.

WARNING: This module is for testing only — not for production inference.
"""

from __future__ import annotations

import torch


# ── Deterministic fill ──────────────────────────────────────────


def kv_fill_tensor(
    head_size: int,
    layer_idx: int,
    kv: int,
    global_positions: torch.Tensor,
    global_head_start: int | torch.Tensor,
    num_local_heads: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Deterministic KV fill tensor: (num_tokens, num_heads, head_size).

    All values in [1, 255] (exact in bfloat16). Each dim slot uses a
    different multiplier so corruptions are caught independently.

    End-to-end example with 4 KV heads, TP=2, head_size=4:

        Each TP rank processes all token positions but only fills its owned
        KV heads. pos = token position in the sequence (not rank).

        Prefill (TP rank 0 owns global heads [0,1]):
          Layer 0, K cache (kv=0):
            pos=0, head=0: [1, 1, 1, 1]
            pos=0, head=1: [4, 5, 6, 7]
            pos=1, head=0: [2, 3, 4, 5]
            pos=1, head=1: [5, 7, 9, 11]

        Prefill (TP rank 1 owns global heads [2,3]):
          Layer 0, K cache (kv=0):
            pos=0, head=2: [7, 9, 11, 13]
            pos=0, head=3: [10, 13, 16, 19]
            pos=1, head=2: [8, 11, 14, 17]
            pos=1, head=3: [11, 15, 19, 23]

        After NIXL transfer, decode rank 0 verifies its KV cache
        matches the exact same values. Any byte mismatch → KV_FAIL.
    """
    compute_device = global_positions.device
    if isinstance(global_head_start, torch.Tensor):
        global_head_start = global_head_start.to(compute_device)
    positions = global_positions.long().to(compute_device).unsqueeze(1).unsqueeze(2)
    heads = (
        (torch.arange(num_local_heads, device=compute_device) + global_head_start)
        .unsqueeze(0)
        .unsqueeze(2)
    )
    dims = torch.arange(head_size, device=compute_device).unsqueeze(0).unsqueeze(0)

    # TODO: values can overflow for large pos/head/dim combinations. Consider
    # a hash-based approach to avoid collisions and stay within bfloat16 range.
    combined = layer_idx + kv * 3 + positions * (dims + 1) + heads * (dims + 3)
    return (combined % 255 + 1).to(dtype)


# ── Head sharding (TP) ─────────────────────────────────────────


def get_kv_heads_for_rank(
    tp_rank: int | torch.Tensor,
    tp_size: int,
    total_kv_heads: int,
) -> tuple[int | torch.Tensor, int]:
    """Given my TP rank, which KV heads are mine?

    Only TP shards KV heads. EP shards MoE experts, not attention.
    Returns ``(first_global_kv_head, num_kv_heads)``.

    Example::

        # 8 KV heads, TP=4 → 2 heads per rank
        >>> get_kv_heads_for_rank(0, 4, 8)
        (0, 2)
        >>> get_kv_heads_for_rank(2, 4, 8)
        (4, 2)

        # GQA: 2 KV heads, TP=4 → multiple ranks share same head
        >>> get_kv_heads_for_rank(0, 4, 2)
        (0, 1)
    """
    if tp_size >= total_kv_heads:
        return tp_rank // (tp_size // total_kv_heads), 1
    heads_per_rank = total_kv_heads // tp_size
    return tp_rank * heads_per_rank, heads_per_rank


# ── Block position mapping ─────────────────────────────────────


def get_block_positions(
    block_ordinal: int,
    block_size: int,
    seq_len: int,
) -> list[int]:
    """Which global token positions are stored in this block?

    Block N holds contiguous positions [N*block_size .. (N+1)*block_size),
    capped at seq_len.

    # TODO: Add cp_rank/cp_size/cp_interleave_size params when CP support is needed.
    # With CP, positions are interleaved across ranks in chunks.

    Example::

        >>> get_block_positions(0, 4, seq_len=16)
        [0, 1, 2, 3]
        >>> get_block_positions(2, 4, seq_len=16)
        [8, 9, 10, 11]
    """
    start = block_ordinal * block_size
    return [p for p in range(start, start + block_size) if p < seq_len]


# ── Layer sharding (PP) ───────────────────────────────────────


def get_my_layers(pp_rank: int, pp_size: int, total_layers: int) -> range:
    """Which layers does this PP stage own?

    Example::

        >>> get_my_layers(0, 2, 8)
        range(0, 4)
        >>> get_my_layers(1, 2, 8)
        range(4, 8)
    """
    per_stage = total_layers // pp_size
    return range(pp_rank * per_stage, (pp_rank + 1) * per_stage)


# ── Prefill: fill KV cache ───────────────────────────────────


def prefill_fill_kv(
    kv_caches: dict[str, list[torch.Tensor]],
    positions: torch.Tensor,
    attn_metadata: dict,
    global_head_start: int,
    num_local_heads: int,
) -> None:
    """Write deterministic values into KV cache during prefill.

    For each layer, maps slot_mapping → (block, offset) and fills K/V
    with values from ``kv_fill_tensor``. Only valid slots (>= 0) are written.

    Example flow (layer 0, TP rank 0 with head 0, block_size=32)::

        slot_mapping = [32, 33, 34, ...]  → block=1, offsets=[0, 1, 2, ...]
        K[1, :, 0] = kv_fill_tensor(head_size, layer=0, kv=0, pos=0, head_start=0, ...)
        V[1, :, 0] = kv_fill_tensor(head_size, layer=0, kv=1, pos=0, head_start=0, ...)
    """
    for layer_name, (k_cache, v_cache) in kv_caches.items():
        layer_idx = int(layer_name.split(".")[1])
        meta = attn_metadata[layer_name]
        slot_mapping = meta["slot_mapping"]
        block_size = meta["block_size"]
        # Packed FP8 K is unpacked_K.view(nb, kv, block//2, d, 2) — a byte-level
        # reinterpret of the contiguous [nb, kv, block, d] cache. View it back so
        # the standard position indexing writes to the right bytes. V is never
        # packed.
        if k_cache.dim() == 5:
            nb, num_kv, hs = k_cache.shape[0], k_cache.shape[1], k_cache.shape[3]
            k_cache = k_cache.view(nb, num_kv, -1, hs)
        head_size = k_cache.shape[3]

        valid = slot_mapping >= 0
        slots = slot_mapping[valid].cpu()
        gpos = positions[: slot_mapping.shape[0]][valid].cpu()
        blocks = slots // block_size
        offsets = slots % block_size

        k_vals = kv_fill_tensor(
            head_size,
            layer_idx,
            0,
            gpos,
            global_head_start,
            num_local_heads,
            dtype=k_cache.dtype,
        )
        v_vals = kv_fill_tensor(
            head_size,
            layer_idx,
            1,
            gpos,
            global_head_start,
            num_local_heads,
            dtype=v_cache.dtype,
        )

        for i in range(len(blocks)):
            b, o = blocks[i].item(), offsets[i].item()
            for h in range(num_local_heads):
                k_cache[b, h, o].copy_(k_vals[i, h])
                v_cache[b, h, o].copy_(v_vals[i, h])

        # Log first/last-transferred values being written
        # Last token (gpos[-1]) is excluded from NIXL transfer (computed locally
        # on decode), so log gpos[-2] as FILL_LAST to match decode's LAST.
        if len(blocks) > 0:
            _log_kv(
                "FILL_FIRST",
                layer_name,
                blocks[0].item(),
                offsets[0].item(),
                gpos[0].item(),
                global_head_start,
                k_vals[0, 0, :8].tolist(),
                v_vals[0, 0, :8].tolist(),
            )
            if len(blocks) > 2:
                _log_kv(
                    "FILL_LAST",
                    layer_name,
                    blocks[-2].item(),
                    offsets[-2].item(),
                    gpos[-2].item(),
                    global_head_start,
                    k_vals[-2, 0, :8].tolist(),
                    v_vals[-2, 0, :8].tolist(),
                )


# ── Decode: verify KV cache ──────────────────────────────────


def decode_verify_kv_layer(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    layer_idx: int,
    block_ids: list[int],
    num_computed: int,
    block_size: int,
    global_positions: torch.Tensor,
    global_head_start: int,
    num_local_heads: int,
) -> list[str]:
    """Verify KV cache for a single layer after NIXL transfer.

    Recomputes expected values via ``kv_fill_tensor`` and compares against
    actual cache contents. Returns list of mismatch descriptions (empty = OK).

    Example::

        # After NIXL transfer, decode rank verifies block 1 of layer 0:
        errors = decode_verify_kv_layer(
            k_cache, v_cache, layer_idx=0, block_ids=[1],
            num_computed=7, block_size=32,
            global_positions=tensor([0,1,2,3,4,5,6]),
            global_head_start=0, num_local_heads=1,
        )
        # errors == [] means byte-exact match
    """
    valid = global_positions < num_computed
    gpos = global_positions[valid]
    n = gpos.shape[0]
    if n == 0:
        return []

    blk_idx = torch.tensor(block_ids).repeat_interleave(block_size)[:n]
    offsets = torch.arange(block_size).repeat(len(block_ids))[:n]
    # Packed FP8 K is a byte-level reinterpret of [nb, kv, block, d]; view it
    # back so the standard position gather reads the right bytes. V is never packed.
    if k_cache.dim() == 5:
        nb, num_kv, hs = k_cache.shape[0], k_cache.shape[1], k_cache.shape[3]
        k_cache = k_cache.view(nb, num_kv, -1, hs)
    head_size = k_cache.shape[3]

    k_exp = kv_fill_tensor(
        head_size,
        layer_idx,
        0,
        gpos,
        global_head_start,
        num_local_heads,
        dtype=k_cache.dtype,
    )
    v_exp = kv_fill_tensor(
        head_size,
        layer_idx,
        1,
        gpos,
        global_head_start,
        num_local_heads,
        dtype=v_cache.dtype,
    )

    k_act = k_cache[blk_idx, :, offsets].cpu()
    v_act = v_cache[blk_idx, :, offsets].cpu()

    layer_name = f"layers.{layer_idx}.self_attn"

    # Log first/last verified entries using the actual values we just computed
    _log_kv(
        "FIRST",
        layer_name,
        blk_idx[0].item(),
        offsets[0].item(),
        gpos[0].item(),
        global_head_start,
        k_act[0, 0, :8].tolist(),
        v_act[0, 0, :8].tolist(),
        k_exp[0, 0, :8].tolist(),
        v_exp[0, 0, :8].tolist(),
    )
    if n > 1:
        _log_kv(
            "LAST",
            layer_name,
            blk_idx[-1].item(),
            offsets[-1].item(),
            gpos[-1].item(),
            global_head_start,
            k_act[-1, 0, :8].tolist(),
            v_act[-1, 0, :8].tolist(),
            k_exp[-1, 0, :8].tolist(),
            v_exp[-1, 0, :8].tolist(),
        )

    errors: list[str] = []
    if not torch.equal(k_act, k_exp):
        n_bad = (k_act != k_exp).sum().item()
        mismatch = (k_act != k_exp).nonzero(as_tuple=False)
        if len(mismatch) > 0:
            idx = tuple(mismatch[0].tolist())
            print(
                f"[SyntheticKV]   K mismatch at {idx}: actual={k_act[idx].item()}, expected={k_exp[idx].item()}, "
                f"blk={blk_idx[idx[0]].item()}, off={offsets[idx[0]].item()}, gpos={gpos[idx[0]].item()}",
                flush=True,
            )
        errors.append(f"{layer_name} K: {n_bad} mismatches")
    if not torch.equal(v_act, v_exp):
        n_bad = (v_act != v_exp).sum().item()
        mismatch = (v_act != v_exp).nonzero(as_tuple=False)
        if len(mismatch) > 0:
            idx = tuple(mismatch[0].tolist())
            print(
                f"[SyntheticKV]   V mismatch at {idx}: actual={v_act[idx].item()}, expected={v_exp[idx].item()}, "
                f"blk={blk_idx[idx[0]].item()}, off={offsets[idx[0]].item()}, gpos={gpos[idx[0]].item()}",
                flush=True,
            )
        errors.append(f"{layer_name} V: {n_bad} mismatches")
    return errors


def _log_kv(
    label: str,
    layer_name: str,
    block_id: int,
    offset: int,
    gpos: int,
    head_start: int,
    k_vals: list,
    v_vals: list,
    k_expected: list | None = None,
    v_expected: list | None = None,
) -> None:
    """Log a KV cache entry for prefill fill or decode verify."""
    header = f"[SyntheticKV]   {label} {layer_name} block={block_id} off={offset} gpos={gpos} head={head_start}:"
    if k_expected is not None:
        match_k = "✓" if k_vals == k_expected else "✗"
        match_v = "✓" if v_vals == v_expected else "✗"
        print(
            f"{header}\n"
            f"[SyntheticKV]     K actual={k_vals} expected={k_expected} {match_k}\n"
            f"[SyntheticKV]     V actual={v_vals} expected={v_expected} {match_v}",
            flush=True,
        )
    else:
        print(
            f"{header}\n"
            f"[SyntheticKV]     K fill={k_vals}\n"
            f"[SyntheticKV]     V fill={v_vals}",
            flush=True,
        )


# ── Helper: build positions for decode verification ───────────


def build_block_positions(
    num_blocks: int,
    block_size: int,
    seq_len: int,
    seq_offset: int = 0,
) -> torch.Tensor:
    """Build global_positions tensor for decode verification across blocks.

    For SWA layers, use ``seq_offset = max(0, num_computed - window_blocks * block_size)``.

    # TODO: Add cp_rank/cp_size/cp_interleave_size params when CP support is needed.

    Example::

        >>> build_block_positions(2, 4, seq_len=8)
        tensor([0, 1, 2, 3, 4, 5, 6, 7])
    """
    all_pos: list[int] = []
    for ordinal in range(num_blocks):
        all_pos.extend(get_block_positions(ordinal, block_size, seq_len))
    if seq_offset > 0:
        all_pos = [p + seq_offset for p in all_pos]
    return torch.tensor(all_pos, dtype=torch.long)
