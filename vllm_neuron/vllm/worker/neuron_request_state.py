# SPDX-License-Identifier: Apache-2.0
"""Maintain slot-stable request state for the standalone Neuron model runner.

Assign each request a fixed slot with :meth:`add_request` and retain it until
:meth:`remove_request` frees it. Recycle slots without compacting rows, and
select each step's active rows through :meth:`build_idx_mapping`.

Authority table:

* Device-authoritative mutable state consists of the ``torch.int32`` tensors
  ``token_ids``, ``num_computed_tokens``, ``total_len``, and
  ``last_sampled_tokens``.
* Host-authoritative immutable metadata consists of the ``np.int32`` arrays
  ``prompt_len``, ``prefill_len``, and ``max_seq_len``.
* ``scheduler_num_computed_tokens`` is host advisory progress used for
  bucketing and control. It never repairs or overwrites the device count.

There is no persistent host mirror of token history or the device count.

Admission copies one ``[1, max_model_len]`` token row into the selected stable
slot. It does not replace the full ``[max_num_reqs, max_model_len]`` state
tensor. Store integer state as ``int32`` to match sampled-token tensors.

Store block IDs and valid counts in a minimal NumPy table for each KV-cache
group. The supported configuration uses a pure row-width calculation without
context-parallel sizing or allocation-to-kernel block expansion.

Slot allocation invariant:

* Always allocate ``min(free_indices)``. Request removals can be processed from
  sets whose iteration order differs across tensor-parallel ranks, so LIFO or
  insertion-order allocation can assign different slots on different ranks.
  Any future data structure must preserve lowest-free-slot selection.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

# Disabled/greedy defaults: top_k disabled = -1, top_p disabled = 1.0, greedy = 0.0. vLLM uses
# top_k == 0 for "disabled"; {None, <= 0} normalize to the -1 sentinel.
_DEFAULT_TOP_K = -1.0
_DEFAULT_TOP_P = 1.0
_DEFAULT_TEMPERATURE = 0.0
NO_LOGPROBS = -1


def _cdiv(a: int, b: int) -> int:
    """Compute ceiling division for nonnegative integers.

    Args:
        a: Dividend.
        b: Divisor.

    Returns:
        The ceiling of ``a / b``.
    """
    return -(-a // b)


def block_row_width(max_model_len: int, block_size: int) -> int:
    """Compute the aligned block-row width for one KV-cache group.

    Round ``ceil(max_model_len / block_size)`` up to a multiple of
    ``128 / block_size`` when ``block_size <= 128``. This calculation covers
    the nonhybrid configuration with a context-parallel degree of one.

    Args:
        max_model_len: Maximum number of tokens in a request.
        block_size: Number of tokens represented by each KV block.

    Returns:
        Number of block IDs allocated to each request row.

    Raises:
        AssertionError: If ``block_size`` is not positive.
    """
    assert block_size > 0, f"block_size must be positive, got {block_size}"
    n = _cdiv(max_model_len, block_size)
    if block_size <= 128:
        align = 128 // block_size
        return _cdiv(n, align) * align
    return n


def _sampling_row(sampling_params, device: torch.device) -> torch.Tensor:
    """Build one device-resident sampling-parameter row.

    Use the greedy or disabled defaults for missing attributes. Normalize
    ``top_k`` values of ``None`` or at most zero to the ``-1`` sentinel.

    Args:
        sampling_params (SamplingParams | None): Sampling values to normalize.
        device: Device on which to create the row.

    Returns:
        A float32 ``[3]`` tensor containing ``top_k``, ``top_p``, and
        ``temperature``.
    """
    top_k = getattr(sampling_params, "top_k", None)
    top_p = getattr(sampling_params, "top_p", None)
    temperature = getattr(sampling_params, "temperature", None)
    top_k = _DEFAULT_TOP_K if (top_k is None or top_k <= 0) else float(top_k)
    top_p = _DEFAULT_TOP_P if top_p is None else float(top_p)
    temperature = _DEFAULT_TEMPERATURE if temperature is None else float(temperature)
    return torch.tensor([top_k, top_p, temperature], dtype=torch.float32, device=device)


class NeuronRequestState:
    """Store per-request decode state in stable slots.

    Follow the module-level authority rules for device state, host metadata,
    and scheduler advisory progress.
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        vocab_size: int,
        device: torch.device,
        block_sizes: list[int],
    ) -> None:
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.vocab_size = vocab_size
        self.device = device
        # Slot bookkeeping. add_request selects min(free_indices) (deterministic lowest free slot),
        # append on free; list ORDER is irrelevant — selection depends only on the free SET, so ranks
        # agree given equal membership. A slot is reused only after its request is removed, so
        # rows never move.
        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        self.free_indices: list[int] = list(range(max_num_reqs))

        # Device-authoritative mutable state. Admission copies one row into
        # these tensors.
        self.token_ids = torch.zeros(
            (max_num_reqs, max_model_len), dtype=torch.int32, device=device
        )
        self.num_computed_tokens = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self.total_len = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.last_sampled_tokens = torch.zeros(
            max_num_reqs, 1, dtype=torch.int32, device=device
        )

        # Host-authoritative immutable metadata.
        self.prompt_len = np.zeros(max_num_reqs, dtype=np.int32)
        self.prefill_len = np.zeros(max_num_reqs, dtype=np.int32)
        self.max_seq_len = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_logprobs = np.full(max_num_reqs, NO_LOGPROBS, dtype=np.int32)

        # Per-request [top_k, top_p, temperature], immutable after admission. Kept DEVICE-resident (V2
        # is async-first: prefer on-device; no host mirror), folded into the one state owner (no
        # separate store). Admission fully overwrites the selected row. Free
        # slots are never selected, so no per-slot reset is needed.
        # The sampler gathers active rows in the compiled graph with
        # `torch.index_select(sampling_params, 0, idx)` where idx is a device int64 tensor typed on
        # host (as the models type their logits index).
        # index_select AND advanced-index gather FAIL eager on Neuron ("Expected self.dtype() ==
        # dst.dtype()"), and an on-device index cast (idx.to(dev).long()) is illegal — so this gather
        # is a COMPILED-path op, not an eager one, and the index must arrive already int64.
        # Build the [max_num_reqs, 3] defaults table CONTIGUOUS on the host (np.repeat of the [1,3] row
        # along axis 0), then a single H2D copy. Building and repeating the
        # default row eagerly on device can produce a non-contiguous layout
        # rejected by Neuron. The host construction guarantees a contiguous
        # copy without an in-place device write.
        self.sampling_params = torch.from_numpy(
            np.repeat(
                np.array(
                    [[_DEFAULT_TOP_K, _DEFAULT_TOP_P, _DEFAULT_TEMPERATURE]],
                    dtype=np.float32,
                ),
                max_num_reqs,
                axis=0,
            )
        ).to(device)

        # Host advisory progress (scheduler-derived upper bound; NOT authoritative).
        self.scheduler_num_computed_tokens = np.zeros(max_num_reqs, dtype=np.int32)

        # Block rows: one [max_num_reqs, width] int32 table per KV group + per-(group, slot) count.
        self.configure_block_sizes(block_sizes)

    @property
    def num_reqs(self) -> int:
        """Return the number of requests that currently hold a slot."""
        return len(self.req_id_to_index)

    # ------------------------------------------------------------------
    # block rows (host NumPy, per KV group)
    # ------------------------------------------------------------------
    def configure_block_sizes(self, block_sizes: list[int]) -> None:
        """Configure empty rows for the resolved KV-cache groups.

        Replace the constructor's placeholder geometry with validated group
        block sizes before admitting requests. Build replacement arrays before
        assignment so validation failures leave the current geometry intact.

        Args:
            block_sizes: Block size for each KV-cache group.

        Raises:
            AssertionError: If requests are active or no block size is given.
        """
        assert self.num_reqs == 0, "cannot change block sizes while requests are active"
        sizes = list(block_sizes)
        assert sizes, "at least one KV cache block size is required"
        tables = [
            np.zeros(
                (self.max_num_reqs, block_row_width(self.max_model_len, block_size)),
                dtype=np.int32,
            )
            for block_size in sizes
        ]
        counts = np.zeros((len(sizes), self.max_num_reqs), dtype=np.int32)
        self.block_sizes = sizes
        self._block_tables = tables
        self._num_blocks = counts

    def block_table(self, group: int) -> np.ndarray:
        """Return the block-ID table for one KV-cache group.

        Args:
            group: KV-cache group index.

        Returns:
            The ``[max_num_reqs, width]`` int32 block-ID array.
        """
        return self._block_tables[group]

    def num_blocks(self, group: int, slot: int) -> int:
        """Return the valid block count for a group and slot.

        Args:
            group: KV-cache group index.
            slot: Stable request slot.

        Returns:
            Number of valid block IDs in the row.
        """
        return int(self._num_blocks[group, slot])

    def has_allocated_blocks(self) -> bool:
        """Return whether any slot currently holds allocated block IDs.

        Returns:
            True when at least one KV-cache group row has a nonzero valid
            block count.

        Example:
            >>> if state.has_allocated_blocks():
            ...     tables = [state.block_rows_snapshot(g) for g in range(2)]
        """
        return bool((self._num_blocks > 0).any())

    def block_rows_snapshot(self, group: int) -> np.ndarray:
        """Return a sentinel-padded copy of one group's block-ID table.

        Rows are slot-indexed ``[max_num_reqs, width]``. Apply the sentinel by
        position at and beyond each row's valid block count (block ID zero is
        valid inside the used prefix, so padding cannot be told apart by
        value). The copy is independent of live state, so KV-cache analysis
        can hold it across request freeing.

        Args:
            group: KV-cache group index.

        Returns:
            An int32 ``[max_num_reqs, width]`` copy with padding entries set
            to ``PAD_SLOT_ID``.

        Example:
            >>> table = state.block_rows_snapshot(0)
            >>> table[slot, : state.num_blocks(0, slot)]  # valid block IDs
        """
        from .neuron_input_build import PAD_SLOT_ID

        table = self._block_tables[group].copy()
        width = table.shape[1]
        pad_mask = (
            np.arange(width, dtype=np.int32)[None, :]
            >= self._num_blocks[group][:, None]
        )
        table[pad_mask] = PAD_SLOT_ID
        return table

    def _check_block_groups(self, block_ids: tuple[list[int], ...]) -> None:
        """Validate that block IDs contain one list per group.

        Args:
            block_ids: Block-ID lists grouped by KV cache.

        Raises:
            AssertionError: If the number of groups does not match the tables.
        """
        assert len(block_ids) == len(self._block_tables), (
            f"block_ids has {len(block_ids)} groups, expected {len(self._block_tables)}"
        )

    def _check_append_capacity(
        self, slot: int, block_ids: tuple[list[int], ...]
    ) -> None:
        """Validate capacity before appending block IDs.

        Args:
            slot: Stable request slot.
            block_ids: Block-ID lists to append by KV-cache group.

        Raises:
            AssertionError: If any append exceeds its block-row width.
        """
        for group, ids in enumerate(block_ids):
            width = self._block_tables[group].shape[1]
            end = int(self._num_blocks[group, slot]) + len(ids)
            assert end <= width, (
                f"block row overflow: group {group} slot {slot} would hold {end} blocks "
                f"> width {width}"
            )

    def _check_fresh_capacity(self, block_ids: tuple[list[int], ...]) -> None:
        """Validate block-row capacity for a fresh admission.

        Validate from an empty row so same-ID replacement can be checked before
        freeing its current slot.

        Args:
            block_ids: Initial block-ID lists by KV-cache group.

        Raises:
            AssertionError: If any group exceeds its block-row width.
        """
        for group, ids in enumerate(block_ids):
            width = self._block_tables[group].shape[1]
            assert len(ids) <= width, (
                f"block row overflow: group {group} needs {len(ids)} blocks > width {width}"
            )

    def _append_block_row(self, slot: int, group: int, ids: list[int]) -> None:
        """Append block IDs to one group row.

        Args:
            slot: Stable request slot.
            group: KV-cache group index.
            ids: Block IDs to append at the current valid count.
        """
        if not ids:
            return
        start = int(self._num_blocks[group, slot])
        self._block_tables[group][slot, start : start + len(ids)] = ids
        self._num_blocks[group, slot] = start + len(ids)

    def _clear_block_row(self, slot: int, group: int) -> None:
        """Clear one group row and reset its valid count.

        Args:
            slot: Stable request slot.
            group: KV-cache group index.
        """
        n = int(self._num_blocks[group, slot])
        if n:
            self._block_tables[group][slot, :n] = 0
        self._num_blocks[group, slot] = 0

    def _append_all_groups(self, slot: int, block_ids: tuple[list[int], ...]) -> None:
        """Append validated block IDs for every KV-cache group.

        Require callers to validate the group count and row capacity before
        calling this method.

        Args:
            slot: Stable request slot.
            block_ids: Block-ID lists to append by KV-cache group.
        """
        for group in range(len(self._block_tables)):
            self._append_block_row(slot, group, block_ids[group])

    def _clear_all_block_rows(self, slot: int) -> None:
        for group in range(len(self._block_tables)):
            self._clear_block_row(slot, group)

    # ------------------------------------------------------------------
    # admission / removal
    # ------------------------------------------------------------------
    def add_request(
        self,
        req_id: str,
        prompt_len: int,
        prefill_token_ids: list[int],
        num_computed_tokens: int,
        max_tokens: int,
        block_ids: tuple[list[int], ...],
        sampling_params=None,
        *,
        release_resident: Callable[[str], None] | None = None,
    ) -> int:
        """Assign a request and copy its state into one stable slot.

        Treat ``prefill_token_ids`` as the complete model input, including
        prompt tokens and any output tokens replayed during resume. Interpret a
        zero computed count as fresh or recomputed prefill and a positive count
        as cached or resumed progress. For an existing request ID, validate all
        inputs before removing and readmitting it; the replacement may receive
        a different slot. Invalid admission leaves all state unchanged.

        Args:
            req_id: Scheduler request identifier.
            prompt_len: Number of original prompt tokens.
            prefill_token_ids: Complete token sequence used for prefill.
            num_computed_tokens: Number of tokens already computed.
            max_tokens: Maximum number of output tokens to generate.
            block_ids: Initial block IDs grouped by KV cache.
            sampling_params (SamplingParams | None): Optional sampling values.
            release_resident: Optional callback that safely releases a
                same-ID resident after validation. The callback must either
                leave an allocatable slot or raise before changing state.

        Returns:
            The selected stable-slot index.

        Raises:
            AssertionError: If admission data is invalid, block capacity is
                insufficient, or no slot is available.
        """
        # Validate all inputs, including block capacity, before changing state.
        # Invalid admission must leave the current request unchanged.
        assert prefill_token_ids is not None, (
            "NeuronRequestStateV2: prefill_token_ids must not be None"
        )
        prefill_len = len(prefill_token_ids)
        assert 0 <= prompt_len <= prefill_len <= self.max_model_len, (
            f"NeuronRequestStateV2: need 0 <= prompt_len ({prompt_len}) <= prefill_len "
            f"({prefill_len}) <= max_model_len ({self.max_model_len})"
        )
        assert 0 <= num_computed_tokens <= prefill_len, (
            f"NeuronRequestStateV2: need 0 <= num_computed ({num_computed_tokens}) <= "
            f"prefill_len ({prefill_len})"
        )
        self._check_block_groups(block_ids)
        self._check_fresh_capacity(block_ids)
        resident = req_id in self.req_id_to_index
        assert resident or self.free_indices, (
            "NeuronRequestStateV2: no free slot (max_num_reqs)"
        )
        requested_logprobs = getattr(sampling_params, "logprobs", None)
        if requested_logprobs == -1:
            requested_logprobs = self.vocab_size

        # Prepare every value that can reject malformed scheduler data before a
        # resident is released.
        padded = prefill_token_ids + [0] * (self.max_model_len - prefill_len)
        token_rows = torch.tensor([padded], dtype=torch.int32, device=self.device)
        num_computed_row = torch.tensor(
            [num_computed_tokens], dtype=torch.int32, device=self.device
        )
        total_len_row = torch.tensor(
            [prefill_len], dtype=torch.int32, device=self.device
        )
        # Seed sampled state from the last computed token. A decode bootstrap
        # reads its first uncached token directly from the token history.
        if 0 < num_computed_tokens <= prefill_len:
            seed = prefill_token_ids[num_computed_tokens - 1]
        else:
            seed = 0
        last_sampled_row = torch.tensor([[seed]], dtype=torch.int32, device=self.device)
        sampling_row = _sampling_row(sampling_params, self.device).unsqueeze(0)

        # All checks passed. Use the caller's safe release path when replacing
        # a request that already has a slot.
        if resident:
            if release_resident is None:
                self.remove_request(req_id)
            else:
                release_resident(req_id)
            assert req_id not in self.req_id_to_index, (
                "resident release callback did not remove the request"
            )
        assert self.free_indices, "NeuronRequestStateV2: no free slot after release"

        # Choose the lowest free slot so every rank makes the same choice.
        slot = min(self.free_indices)
        self.free_indices.remove(slot)
        self.req_id_to_index[req_id] = slot
        self.index_to_req_id[slot] = req_id

        # Host metadata + advisory shadow.
        self.prompt_len[slot] = prompt_len
        self.prefill_len[slot] = prefill_len
        self.max_seq_len[slot] = prompt_len + max_tokens
        self.scheduler_num_computed_tokens[slot] = num_computed_tokens
        self.num_logprobs[slot] = (
            NO_LOGPROBS if requested_logprobs is None else requested_logprobs
        )

        # Block rows: the slot is fresh (just selected from free_indices, whose
        # members were cleared by remove_request), so append from a zero
        # baseline fully populates the row.
        self._append_all_groups(slot, block_ids)

        # Update only the admitted row. The one-row slices avoid full-state
        # replacement tensors and keep the source and destination shapes equal.
        row = slice(slot, slot + 1)
        self.token_ids[row].copy_(token_rows)
        self.num_computed_tokens[row].copy_(num_computed_row)
        self.total_len[row].copy_(total_len_row)
        self.last_sampled_tokens[row].copy_(last_sampled_row)
        self.sampling_params[row].copy_(sampling_row)
        return slot

    def remove_request(self, req_id: str) -> int | None:
        """Free a request slot and clear its block rows.

        Leave other rows in place. Device contents remain stale until the next
        admission overwrites them, but freed slots are not selected. Reset
        block-row counts immediately so recycled slots cannot inherit them.

        Args:
            req_id: Scheduler request identifier to remove.

        Returns:
            The released slot index, or ``None`` if the request is absent.
        """
        slot = self.req_id_to_index.pop(req_id, None)
        if slot is None:
            return None
        self.index_to_req_id.pop(slot, None)
        self._clear_all_block_rows(slot)
        self.free_indices.append(slot)
        return slot

    # ------------------------------------------------------------------
    # cached-request update (host advisory + block append)
    # ------------------------------------------------------------------
    def update_cached_request(
        self,
        req_id: str,
        scheduler_num_computed_tokens: int,
        new_block_ids: tuple[list[int], ...] | None = None,
    ) -> None:
        """Apply scheduler progress to a cached request.

        Advance only the host advisory count because the compiled producer
        advances device-authoritative progress. Append, rather than replace,
        any newly allocated block IDs. Validate all block updates before
        mutating the advisory count or tables.

        Args:
            req_id: Scheduler request identifier.
            scheduler_num_computed_tokens: Scheduler-reported progress.
            new_block_ids: Optional block-ID lists to append by KV-cache group.

        Raises:
            AssertionError: If the request is absent or block data is invalid.
        """
        slot = self.req_id_to_index.get(req_id)
        assert slot is not None, (
            f"NeuronRequestStateV2.update_cached_request: {req_id!r} not present"
        )
        # Preflight blocks BEFORE mutating anything, so a malformed/overflowing append leaves the
        # advisory count and block rows unchanged.
        if new_block_ids is not None:
            self._check_block_groups(new_block_ids)
            self._check_append_capacity(slot, new_block_ids)
        self.scheduler_num_computed_tokens[slot] = scheduler_num_computed_tokens
        if new_block_ids is not None:
            self._append_all_groups(slot, new_block_ids)

    # ------------------------------------------------------------------
    # idx_mapping — select the active rows for a step in scheduler order
    # ------------------------------------------------------------------
    def build_idx_mapping(self, active_req_ids: list[str]) -> np.ndarray:
        """Map scheduler-ordered request IDs to stable slots.

        Args:
            active_req_ids: Request IDs in scheduler batch order.

        Returns:
            An int32 array mapping each batch position to its stable slot.

        Raises:
            AssertionError: If an active request does not hold a slot.
        """
        idx_np = np.empty(len(active_req_ids), dtype=np.int32)
        for batch_pos, req_id in enumerate(active_req_ids):
            slot = self.req_id_to_index.get(req_id)
            assert slot is not None, (
                f"NeuronRequestStateV2.build_idx_mapping: {req_id!r} not present"
            )
            idx_np[batch_pos] = slot
        return idx_np
