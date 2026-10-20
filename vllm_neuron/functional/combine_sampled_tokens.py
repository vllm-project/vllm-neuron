# SPDX-License-Identifier: Apache-2.0
"""Provide on-device consumers of a decode step's sampled-token future.

With async scheduling, dispatch the next model graph before copying the
previous step's sampled token to the host. Assemble dependent inputs on the
device from the in-flight output to avoid a host round trip. These helpers use
plain PyTorch operations traced by ``torch.compile`` inside the model graph.

Use :func:`scatter_sampled_into_slot_store` to update the permanent slot store
on every tensor-parallel rank. Use
:func:`build_decode_input_ids_from_store` to gather row inputs from that
coherent slot-keyed store when the batch composition changes.
"""

from __future__ import annotations

import torch


def scatter_sampled_into_slot_store(
    slot_store: torch.Tensor,
    slot_row_perm: torch.Tensor,
    prev_future: torch.Tensor,
) -> torch.Tensor:
    """Update the persistent sampled-token store on the device.

    Write each active slot's sampled token inside the compiled graph so every
    tensor-parallel rank advances the same slot-indexed state. Preserve idle
    slots. ``slot_row_perm`` is keyed by permanent slot: a nonnegative value
    selects a row from ``prev_future``, while a negative value keeps the
    existing store value.

    Return a new tensor so the caller can alias it back to the persistent
    store. Before calling this function, bit-reinterpret a raw ``uint32``
    sampled future as the store's signed integer dtype. The conversion in this
    function is a value cast and does not perform that reinterpretation.

    Args:
        slot_store: Persistent ``[max_num_reqs, 1]`` integer store.
        slot_row_perm: ``[max_num_reqs]`` mapping from slot to sampled row, or
            ``-1`` to preserve the slot.
        prev_future: ``[padded_num_reqs]`` sampled tokens in batch-row order,
            already reinterpreted to ``slot_store``'s dtype.

    Returns:
        A new ``[max_num_reqs, 1]`` store without mutating ``slot_store``.

    Example:
        >>> import torch
        >>> store = torch.zeros(4, 1, dtype=torch.int32)
        >>> future = torch.tensor([70, 80], dtype=torch.int32)
        >>> perm = torch.tensor([1, -1, 0, -1], dtype=torch.int32)
        >>> scatter_sampled_into_slot_store(store, perm, future).reshape(-1).tolist()
        [80, 0, 70, 0]
    """
    perm = slot_row_perm.to(torch.int64)
    valid = perm >= 0
    fut = prev_future.reshape(-1).to(slot_store.dtype)
    # Clamp negatives to 0 so the gather stays in-bounds; `where` discards those slots.
    gathered = fut[torch.clamp(perm, min=0)].reshape(-1, 1)  # [max_num_reqs, 1]
    return torch.where(valid.reshape(-1, 1), gathered, slot_store)


def build_decode_input_ids_from_store(
    input_ids: torch.Tensor,
    row_to_slot: torch.Tensor,
    slot_store: torch.Tensor,
) -> torch.Tensor:
    """Build decode ``input_ids`` from the permanent device slot store.

    Gather each active row by stable slot inside the compiled graph. Use ``-1``
    for padding or newly-prefilled rows whose existing ``input_ids`` value must
    be preserved. Nonnegative slot values must be smaller than
    ``max_num_reqs``.

    Args:
        input_ids: ``[padded_num_reqs]`` integer placeholder input.
        row_to_slot: ``[padded_num_reqs]`` mapping from row to stable slot, or
            ``-1`` to preserve the input row.
        slot_store: ``[max_num_reqs, 1]`` per-slot sampled-token store.

    Returns:
        New ``[padded_num_reqs]`` input IDs without mutating ``input_ids``.

    Example:
        >>> import torch
        >>> store = torch.tensor([[11], [22], [33]], dtype=torch.int32)
        >>> input_ids = torch.zeros(3, dtype=torch.int32)
        >>> row_to_slot = torch.tensor([2, 0, -1], dtype=torch.int32)
        >>> build_decode_input_ids_from_store(input_ids, row_to_slot, store).tolist()
        [33, 11, 0]
    """
    perm = row_to_slot.to(torch.int64)
    valid = perm >= 0
    store_flat = slot_store.reshape(-1).to(input_ids.dtype)  # [max_num_reqs]
    # Clamp negatives to 0 so the gather stays in-bounds; `where` discards those rows.
    gathered = store_flat[torch.clamp(perm, min=0)]
    return torch.where(valid, gathered, input_ids)
