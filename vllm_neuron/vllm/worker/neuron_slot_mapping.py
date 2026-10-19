# SPDX-License-Identifier: Apache-2.0
"""Provide CPU slot mapping for the standalone Neuron model runner."""

from __future__ import annotations

import numpy as np

_NULL_BLOCK_ID = 0


# Forked from the V1 model runner's _compute_slot_mapping_cpu helper.
# TODO: Before enabling V2 context parallelism, reconcile the cp_world_size > 1
# branch with the current V1 helper and add V2-specific CP coverage.
def compute_slot_mapping_cpu(
    block_table_cpu: np.ndarray,
    slot_mapping_np: np.ndarray,
    positions: np.ndarray,
    req_indices: np.ndarray,
    block_size: int,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
) -> None:
    """Populate a slot mapping with vectorized NumPy operations.

    Map only tokens owned by ``cp_rank`` when context parallelism is active.
    Route tokens owned by remote ranks to the null block.

    Args:
        block_table_cpu: Block ids arranged as request rows.
        slot_mapping_np: Output array populated in place.
        positions: Absolute token positions.
        req_indices: Request-row index for each token position.
        block_size: Number of slots in each local KV block.
        cp_world_size: Number of context-parallel ranks.
        cp_rank: Context-parallel rank that owns local tokens.
        cp_kv_cache_interleave_size: Consecutive tokens assigned to each rank.

    Example:
        >>> import numpy as np
        >>> block_table = np.array([[5]], dtype=np.int32)
        >>> slots = np.zeros(2, dtype=np.int64)
        >>> compute_slot_mapping_cpu(
        ...     block_table,
        ...     slots,
        ...     np.array([0, 1]),
        ...     np.array([0, 0]),
        ...     block_size=4,
        ... )
        >>> slots.tolist()
        [20, 21]
    """
    max_blocks_per_req = block_table_cpu.shape[1]
    virtual_block_size = block_size * cp_world_size

    block_indices = positions // virtual_block_size
    flat_indices = req_indices * max_blocks_per_req + block_indices
    block_numbers = block_table_cpu.ravel()[flat_indices]

    if cp_world_size > 1:
        virtual_block_offsets = positions % virtual_block_size
        is_local = (
            virtual_block_offsets // cp_kv_cache_interleave_size % cp_world_size
            == cp_rank
        )
        block_offsets = (
            virtual_block_offsets
            // (cp_world_size * cp_kv_cache_interleave_size)
            * cp_kv_cache_interleave_size
            + virtual_block_offsets % cp_kv_cache_interleave_size
        )
        slot_mapping = block_numbers * block_size + block_offsets
        slot_mapping_np[: len(positions)] = np.where(
            is_local, slot_mapping, _NULL_BLOCK_ID
        )
    else:
        block_offsets = positions % block_size
        np.add(
            block_numbers * block_size,
            block_offsets,
            out=slot_mapping_np[: len(positions)],
        )
