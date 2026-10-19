# SPDX-License-Identifier: Apache-2.0
"""Build device inputs for the standalone Neuron model runner.

Prepare flat token positions and request indices, batch-ordered block rows,
and slot mappings from :class:`NeuronRequestState`. Perform only host-side
NumPy and integer bookkeeping so input preparation does not read device state
or serialize the asynchronous pipeline. Build decode ``input_ids`` inside the
compiled graph from the per-slot sampled-token store.

Apply the block-table sentinel by position at and beyond each row's valid
block count. Do not replace zero-valued block IDs, because block ID zero is
valid inside the used prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from .neuron_slot_mapping import compute_slot_mapping_cpu

if TYPE_CHECKING:
    from .neuron_request_state import NeuronRequestState

# These write-padding and read-sentinel values are fixed Neuron kernel
# contracts. Keeping them local avoids a dependency on the V1 runner.
NULL_BLOCK_ID = 0
PAD_SLOT_ID = -1

__all__ = [
    "NULL_BLOCK_ID",
    "PAD_SLOT_ID",
    "NeuronModelInput",
    "build_flat_positions",
    "build_prefill_linear_idx",
    "gather_block_rows_raw",
    "build_kernel_block_rows",
    "SWA_ATTENTION_PMAX",
    "compute_swa_num_blocks",
    "build_swa_kernel_block_rows",
    "build_slot_mapping",
    "compute_slot_mapping_cpu",
]


@dataclass(frozen=True, slots=True)
class NeuronModelInput:
    """Store fixed-shape operands and host metadata for one runner step.

    Use one sampled row for prefill and the padded decode row count for decode.
    Permit index values to vary between steps while retaining graph-stable
    tensor shapes.
    """

    # Common to prefill and decode.
    positions: torch.Tensor  # [P] prefill / [B] decode, int32
    block_tables: list[torch.Tensor]  # Per group [1|B, width], int32
    slot_mapping: torch.Tensor  # [P] / [B], int64
    sampling_positions: torch.Tensor  # [R] hidden-state row, int64
    row_to_slot: torch.Tensor  # [R] output row -> stable slot or -1 padding, int64
    input_valid_mask: torch.Tensor  # [P] / [B], bool
    slot_active_row: torch.Tensor  # [max_num_reqs] slot -> output row, int32
    query_len_by_slot: torch.Tensor  # [max_num_reqs], int32
    prefill_len: torch.Tensor  # [max_num_reqs], int32

    # Host bookkeeping.
    idx_mapping_np: np.ndarray  # [num_active] output row -> stable slot
    sample_position_ids: tuple[int, ...]  # Absolute position per sampled row
    is_prefill: bool
    num_active: int
    padded_num_tokens: int
    padded_num_reqs: int
    padded_sample_rows: int
    valid_rows: tuple[bool, ...]  # [R], trims padding/non-final prefill
    cached_seq_len: int
    kv_segment_size: int

    # Prefill-only; None for decode.
    linear_idx: torch.Tensor | None

    # Per-group KV-write slot mappings ([P]/[B] int64 each, one per KV cache
    # group). Each group owns its block table, so writes must resolve slots
    # through their own group's rows. None means single-group legacy input
    # (fall back to ``slot_mapping`` for every group).
    slot_mappings: tuple[torch.Tensor, ...] | None = None
    # Per-group per-row KV position offsets for sliding-window groups on
    # decode ([B] int32); None entries for full-attention groups and prefill.
    swa_kv_pos_offsets: tuple[torch.Tensor | None, ...] | None = None


def build_flat_positions(
    num_computed_tokens: np.ndarray, query_lens: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build flat token positions and request indices in scheduler order.

    Emit each row's positions from its computed-token count through the end of
    its current query. This covers single-token decode, prefix-cache offsets,
    and chunked prefill windows. Concatenate all rows in scheduler order.

    Args:
        num_computed_tokens: ``[num_reqs]`` computed counts in scheduler order.
        query_lens: ``[num_reqs]`` positive query lengths in scheduler order.

    Returns:
        Two ``[sum(query_lens)]`` int64 arrays containing positions and row
        indices.

    Raises:
        AssertionError: If input shapes differ or a query length is not
            positive.
    """
    num_computed_tokens = np.asarray(num_computed_tokens, dtype=np.int64)
    query_lens = np.asarray(query_lens, dtype=np.int64)
    assert num_computed_tokens.shape == query_lens.shape, (
        f"num_computed {num_computed_tokens.shape} != query_lens {query_lens.shape}"
    )
    assert (query_lens >= 1).all(), (
        f"query_lens must be >= 1, got {query_lens.tolist()}"
    )

    # per-token row index: [0]*q0 + [1]*q1 + ... ; positions = base[row] + within-row offset.
    req_indices = np.repeat(np.arange(query_lens.shape[0], dtype=np.int64), query_lens)
    # within-row offset 0..q-1 for each row, via arange minus the row's start in the flat layout.
    starts = np.cumsum(query_lens) - query_lens  # flat start index of each row
    within = np.arange(req_indices.shape[0], dtype=np.int64) - starts[req_indices]
    positions = num_computed_tokens[req_indices] + within
    return positions, req_indices


def build_prefill_linear_idx(
    idx_mapping_np: np.ndarray,
    num_computed_tokens: np.ndarray,
    query_lens: np.ndarray,
    max_model_len: int,
    padded_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the padded prefill gather indices and validity mask.

    Address each token as ``slot * max_model_len + position`` in the flattened
    stable-slot token store. Build indices on the host for fresh prefill,
    prefix-cache offsets, and chunked prefill windows.

    Give padded tokens the in-bounds index zero and a false validity flag.
    Keep the array shape fixed while allowing index values to vary by step.

    Args:
        idx_mapping_np: ``[num_reqs]`` stable slots in scheduler order.
        num_computed_tokens: ``[num_reqs]`` prefill window start positions.
        query_lens: ``[num_reqs]`` prefill window lengths.
        max_model_len: Token-buffer row width.
        padded_tokens: Fixed flat token capacity.

    Returns:
        An int64 ``[padded_tokens]`` gather index and a Boolean
        ``[padded_tokens]`` validity mask.

    Raises:
        AssertionError: If scored tokens exceed the padded capacity or a valid
            position exceeds the model length.
    """
    idx_mapping_np = np.asarray(idx_mapping_np, dtype=np.int64)
    positions, req_indices = build_flat_positions(num_computed_tokens, query_lens)
    total = positions.shape[0]
    assert total <= padded_tokens, (
        f"scored tokens {total} > padded_tokens {padded_tokens}"
    )
    assert total == 0 or positions.max() < max_model_len, (
        f"position >= max_model_len ({max_model_len})"
    )
    linear_idx = np.zeros(padded_tokens, dtype=np.int64)  # in-bounds padding (0)
    valid = np.zeros(padded_tokens, dtype=bool)
    linear_idx[:total] = idx_mapping_np[req_indices] * max_model_len + positions
    valid[:total] = True
    return linear_idx, valid


def gather_block_rows_raw(
    state: NeuronRequestState,
    group: int,
    idx_mapping_np: np.ndarray,
    padded_num_reqs: int,
) -> np.ndarray:
    """Gather one KV group's block rows into a zero-padded host array.

    Preserve scheduler row order and fill unused columns and rows with
    ``NULL_BLOCK_ID``. The slot-mapping helper reads only the valid prefix.
    Apply the kernel's ``-1`` sentinel separately.

    Args:
        state: State that owns the block table.
        group: KV-cache group index.
        idx_mapping_np: ``[num_active]`` stable slots in scheduler order.
        padded_num_reqs: Compiled batch width.

    Returns:
        A zero-padded int32 array with shape
        ``[padded_num_reqs, block_row_width]``.

    Raises:
        AssertionError: If active rows exceed the padded batch width.
    """
    table = state.block_table(group)  # [max_num_reqs, width] int32
    width = table.shape[1]
    out = np.zeros((padded_num_reqs, width), dtype=np.int32)
    num_active = idx_mapping_np.shape[0]
    assert num_active <= padded_num_reqs, (
        f"num_active {num_active} > padded_num_reqs {padded_num_reqs}"
    )
    # Copy only each source row's valid prefix so a shorter row cannot expose the slot's stale tail.
    for row in range(num_active):
        slot = int(idx_mapping_np[row])
        n = state.num_blocks(group, slot)
        if n:
            out[row, :n] = table[slot, :n]
    return out


def build_kernel_block_rows(
    state: NeuronRequestState,
    group: int,
    idx_mapping_np: np.ndarray,
    padded_num_reqs: int,
    *,
    output_device: torch.device | None = None,
) -> torch.Tensor:
    """Build batch-ordered attention-kernel block rows.

    Set every column at or beyond a row's valid block count to
    ``PAD_SLOT_ID`` so the kernel skips inactive slots. Preserve block ID zero
    inside valid prefixes. Fill all padded rows with the sentinel.

    Args:
        state: State that owns the block table.
        group: KV-cache group index.
        idx_mapping_np: ``[num_active]`` stable slots in scheduler order.
        padded_num_reqs: Compiled batch width.
        output_device: Device for the returned tensor. Defaults to
            ``state.device``.

    Returns:
        An int32 tensor on the selected device with shape
        ``[padded_num_reqs, block_row_width]``.

    Raises:
        AssertionError: If active rows exceed the padded batch width.
    """
    table = state.block_table(group)
    width = table.shape[1]
    out = np.full((padded_num_reqs, width), PAD_SLOT_ID, dtype=np.int32)
    num_active = idx_mapping_np.shape[0]
    assert num_active <= padded_num_reqs, (
        f"num_active {num_active} > padded_num_reqs {padded_num_reqs}"
    )
    for row in range(num_active):
        slot = int(idx_mapping_np[row])
        n = state.num_blocks(group, slot)
        if n:
            out[row, :n] = table[
                slot, :n
            ]  # valid prefix copied verbatim; a within-count column survives
            # the positional -1 mask (block 0 is upstream's reserved null block, never a real alloc)
    return torch.from_numpy(out).to(
        state.device if output_device is None else output_device
    )


def build_slot_mapping(
    block_rows_raw: np.ndarray,
    positions: np.ndarray,
    req_indices: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """Build a flat token-to-KV-slot mapping on the host.

    Consume zero-padded raw block rows and index only their valid regions. Do
    not apply the attention kernel's ``-1`` sentinel. Use a context-parallel
    degree of one.

    Args:
        block_rows_raw: ``[num_reqs, width]`` batch-ordered int32 rows.
        positions: ``[num_tokens]`` absolute token positions.
        req_indices: ``[num_tokens]`` batch-row indices.
        block_size: Number of token slots in each KV block.

    Returns:
        An int64 ``[num_tokens]`` slot mapping.
    """
    slot_mapping = np.zeros(positions.shape[0], dtype=np.int64)
    compute_slot_mapping_cpu(
        np.ascontiguousarray(block_rows_raw),
        slot_mapping,
        np.asarray(positions, dtype=np.int64),
        np.asarray(req_indices, dtype=np.int64),
        block_size,
    )
    return slot_mapping


# NKI attention kernel tile constraint: SBUF partition count. Trimmed
# sliding-window block counts must produce a context length divisible by this
# value, or the decode kernel's tiling rejects the shape. Same constant as the
# V1 runner's ``_compute_swa_num_blocks`` and ``bucket_utils``.
SWA_ATTENTION_PMAX = 128


def compute_swa_num_blocks(
    sliding_window: int,
    block_size: int,
    max_model_len: int,
    dcp_size: int = 1,
) -> int:
    """Number of KV blocks for a sliding-window decode group's block table.

    Single source of truth for the sliding-window block-table width: warmup
    (compile) and runtime must call this with the same arguments and agree,
    or the compiled graph shape and the runtime shape diverge.

    Sizing: the window needs ``sliding_window // block_size + 1`` blocks,
    padded up to the attention kernel's partition-tile divisibility, then
    clipped to the full-context block count because the request-state block
    table is only allocated that wide. The clip binds when
    ``sliding_window >= max_model_len`` (small / smoke configs); in that
    regime trimming is a no-op and the full-context width is the only
    storage-backed choice. Mirrors the V1 runner's ``_compute_swa_num_blocks``.

    Args:
        sliding_window: Window size in tokens for the group's layers.
        block_size: Number of token slots per KV block.
        max_model_len: Maximum model sequence length.
        dcp_size: Decode-context-parallel degree (one on the V2 runner).

    Returns:
        The trimmed block-table width for the sliding-window group.
    """
    min_blocks = sliding_window // block_size + 1
    blocks_per_pmax = SWA_ATTENTION_PMAX // block_size
    num_swa_blocks = (
        (min_blocks + blocks_per_pmax - 1) // blocks_per_pmax * blocks_per_pmax
    )
    dcp_block_size = block_size * max(dcp_size, 1)
    full_ctx_blocks = (max_model_len + dcp_block_size - 1) // dcp_block_size
    return min(num_swa_blocks, full_ctx_blocks)


def build_swa_kernel_block_rows(
    state: NeuronRequestState,
    group: int,
    idx_mapping_np: np.ndarray,
    padded_num_reqs: int,
    num_computed_np: np.ndarray,
    num_swa_blocks: int,
    block_size: int,
    *,
    output_device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build window-trimmed kernel block rows for a sliding-window group.

    Copy, for each active row, the ``num_swa_blocks``-wide slice of its block
    table that covers the attention window ending at the current position:
    ``start_block = max(ctx_len // block_size - num_swa_blocks + 1, 0)``.
    Also return the per-row KV position offset ``start_block * block_size``,
    which the model subtracts from position IDs so the causal mask is
    computed in the trimmed frame. The offset is always returned (zero when
    no row is trimmed) so the compiled graph signature does not depend on
    batch content.

    Set every column at or beyond a row's valid trimmed length to
    ``PAD_SLOT_ID``, preserve block ID zero inside valid prefixes, and fill
    padded rows with the sentinel — the same conventions as
    :func:`build_kernel_block_rows`. When ``num_swa_blocks`` is at least the
    state table width the loop degenerates to the untrimmed gather with zero
    offsets. Slot mappings for KV writes must NOT use these rows: writes
    address the full (untrimmed) table.

    Args:
        state: State that owns the block table.
        group: KV-cache group index.
        idx_mapping_np: ``[num_active]`` stable slots in scheduler order.
        padded_num_reqs: Compiled batch width.
        num_computed_np: ``[num_active]`` per-row context lengths (tokens
            already stored in KV), scheduler order.
        num_swa_blocks: Trimmed width from :func:`compute_swa_num_blocks`.
        block_size: Number of token slots per KV block.
        output_device: Device for the returned tensors. Defaults to
            ``state.device``.

    Returns:
        A pair of the int32 ``[padded_num_reqs, width]`` trimmed block rows
        and the int32 ``[padded_num_reqs]`` KV position offsets.
    """
    table = state.block_table(group)
    width = min(num_swa_blocks, table.shape[1])
    out = np.full((padded_num_reqs, width), PAD_SLOT_ID, dtype=np.int32)
    offsets = np.zeros(padded_num_reqs, dtype=np.int32)
    num_active = idx_mapping_np.shape[0]
    assert num_active <= padded_num_reqs, (
        f"num_active {num_active} > padded_num_reqs {padded_num_reqs}"
    )
    assert num_computed_np.shape[0] >= num_active, (
        f"num_computed has {num_computed_np.shape[0]} rows, need {num_active}"
    )
    for row in range(num_active):
        slot = int(idx_mapping_np[row])
        n = state.num_blocks(group, slot)
        if not n:
            continue
        start_block = max(int(num_computed_np[row]) // block_size - width + 1, 0)
        valid = min(start_block + width, n) - start_block
        if valid > 0:
            out[row, :valid] = table[slot, start_block : start_block + valid]
        offsets[row] = start_block * block_size
    device = state.device if output_device is None else output_device
    return (
        torch.from_numpy(out).to(device),
        torch.from_numpy(offsets).to(device),
    )
