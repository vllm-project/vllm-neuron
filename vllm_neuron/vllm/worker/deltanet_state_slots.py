# SPDX-License-Identifier: Apache-2.0
"""Map each request to one row of the DeltaNet state buffers.

A linear attention layer keeps a recurrent state and a convolution state instead
of a KV cache. The model holds both as one buffer with a row for each concurrent
request, and it indexes them with ``batch_slot_indices``. This module owns the
assignment of a request to a row.

This class manages **allocation only**. It tracks which row belongs to which
request and it keeps the free queue. The model reads and writes the state inside
the graph.
"""

from __future__ import annotations

from collections import deque

_LINEAR_ATTENTION = "linear_attention"


def usable_state_rows(num_rows: int, buckets: list[int] | None) -> int:
    """Return the state rows that requests can hold at the same time.

    A decode batch pads up to a compiled bucket. The pool gives each padded row a
    free row, so a padded row cannot overwrite the state of a request. So the pool
    must hold 1 free row while a batch pads, and admission must keep that row free.

    A ladder that holds every count from 1 to `num_rows` never pads a batch, and
    every row is then usable. A ladder with a gap pads a batch whose request count
    falls in the gap, so 1 row must stay free.

    A request that is not in the batch can still hold a row: a decode request waits
    out a prefill step, and a request the scheduler did not schedule keeps its row.
    So the pool can be full while the batch is not, and the padded rows of that
    batch then find no free row.

    Args:
        num_rows: The row count of the pool. Use max(num_seqs_buckets).
        buckets: The compiled `num_seqs_buckets` ladder. None gives 1 free row.

    Returns:
        The largest count of requests that can hold a row at the same time.

    Example:
        >>> usable_state_rows(8, [1, 2, 4, 8])
        7
        >>> usable_state_rows(2, [1, 2])
        2
        >>> usable_state_rows(1, [1])
        1
    """
    ladder = set(buckets or [])
    if ladder and all(count in ladder for count in range(1, num_rows + 1)):
        return num_rows
    return max(1, num_rows - 1)


def state_pool_rows(buckets: list[int] | None) -> int:
    """Return the row count of the DeltaNet state buffers for a bucket ladder.

    The pool holds ``max(buckets)`` rows for the requests, and 1 more row when a
    decode batch can pad (see `usable_state_rows`). That row takes the padded
    rows of a batch, so ``max(buckets)`` requests (``max_num_seqs``) can hold a
    row at the same time. The model, the runner and the scheduler all size the
    pool with this function.

    Args:
        buckets: The compiled `num_seqs_buckets` ladder. None or empty gives 1 row.

    Returns:
        The row count of the state buffers.

    Example:
        >>> state_pool_rows([1, 4, 8])
        9
        >>> state_pool_rows([32])
        33
        >>> state_pool_rows([1, 2])
        2
        >>> state_pool_rows(None)
        1
    """
    if not buckets:
        return 1
    num_rows = max(buckets)
    if usable_state_rows(num_rows, buckets) == num_rows:
        return num_rows
    return num_rows + 1


def count_linear_attention_layers(model_config) -> int:
    """Count the layers that keep a recurrent state instead of a KV cache.

    Read ``layer_types`` from the text config first, then from the top-level
    config. A hybrid checkpoint marks these layers as ``linear_attention``.

    This module holds the helper because it imports nothing from vLLM. Both the
    platform config check and the model runner therefore reuse one definition.

    Args:
        model_config: The vLLM model config.

    Returns:
        The number of linear attention layers. Returns 0 when the config
        declares no ``layer_types``, which is the dense case.

    Example:
        >>> import types
        >>> text = types.SimpleNamespace(layer_types=["linear_attention"] * 3)
        >>> cfg = types.SimpleNamespace(hf_config=types.SimpleNamespace(text_config=text))
        >>> count_linear_attention_layers(cfg)
        3
    """
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return 0
    candidates = (getattr(hf_config, "text_config", None), hf_config)
    for config in candidates:
        if config is None:
            continue
        layer_types = getattr(config, "layer_types", None)
        # Count a list or a tuple of names only. A config object can answer
        # every attribute with a placeholder, and a placeholder is true but it
        # holds no name. Such a value must give the dense answer, not an error.
        if isinstance(layer_types, (list, tuple)) and layer_types:
            return sum(1 for name in layer_types if name == _LINEAR_ATTENTION)
    return 0


class DeltaNetStateSlots:
    """Assign one state row to each active request.

    The pool holds ``num_slots`` rows, which must equal the row count of the
    model state buffers. ``allocate`` is idempotent: a request keeps the same row
    for every prefill chunk and every decode step until ``free`` releases it.

    A released row returns to the back of the free queue. The next request that
    takes the row starts at position 0, so the per-slot reset in the model zeros
    the row before the first read. The pool therefore stores no state itself.

    A decode batch pads up to a compiled bucket, so it can hold more rows than
    there are requests. ``slots_for_batch`` gives each padded row a row that no
    request holds. ``state_pool_rows`` adds that row to the pool when a batch can
    pad, and the model keeps one row for each slot. See ``slots_for_batch``.

    Args:
        num_slots: Number of state rows. Use ``state_pool_rows(num_seqs_buckets)``,
            which is the row count the model allocates.

    Example:
        >>> pool = DeltaNetStateSlots(num_slots=2)
        >>> pool.allocate("req-a")
        0
        >>> pool.allocate("req-a")  # idempotent
        0
        >>> pool.allocate("req-b")
        1
        >>> pool.free("req-a")
        0
        >>> pool.allocate("req-c")  # reuses the released row
        0
    """

    def __init__(self, num_slots: int):
        if num_slots < 1:
            raise ValueError(
                f"DeltaNetStateSlots needs at least one slot, got {num_slots}. "
                "Set num_slots to the row count of the model state buffers."
            )
        self.num_slots = num_slots
        self._slot_of_req: dict[str, int] = {}
        self._free: deque[int] = deque(range(num_slots))

    @property
    def num_free(self) -> int:
        """Return the number of rows that no request holds."""
        return len(self._free)

    @property
    def num_active(self) -> int:
        """Return the number of rows that a request holds."""
        return len(self._slot_of_req)

    def contains(self, req_id: str) -> bool:
        """Return True when the request already holds a row.

        Example:
            >>> pool = DeltaNetStateSlots(num_slots=1)
            >>> pool.contains("req-a")
            False
            >>> _ = pool.allocate("req-a")
            >>> pool.contains("req-a")
            True
        """
        return req_id in self._slot_of_req

    def get(self, req_id: str) -> int | None:
        """Return the row of the request, or None when it holds no row.

        Example:
            >>> pool = DeltaNetStateSlots(num_slots=1)
            >>> pool.get("req-a") is None
            True
        """
        return self._slot_of_req.get(req_id)

    def allocate(self, req_id: str) -> int:
        """Return the row of the request. Take a free row on the first call.

        Args:
            req_id: The request id.

        Returns:
            The row index of this request.

        Raises:
            ValueError: The pool holds no free row.

        Example:
            >>> pool = DeltaNetStateSlots(num_slots=1)
            >>> pool.allocate("req-a")
            0
        """
        slot = self._slot_of_req.get(req_id)
        if slot is not None:
            return slot
        if not self._free:
            raise ValueError(
                f"The DeltaNet state pool holds no free slot for request "
                f"{req_id}. All {self.num_slots} slots are in use. The "
                "scheduler must not admit more requests than the state buffers "
                "hold. Raise max(num_seqs_buckets), or lower --max-num-seqs."
            )
        slot = self._free.popleft()
        self._slot_of_req[req_id] = slot
        return slot

    def free(self, req_id: str) -> int | None:
        """Release the row of the request and return it.

        Return None when the request holds no row. A second call for the same
        request is therefore safe, because the model runner can reap a request
        more than one time.

        Args:
            req_id: The request id.

        Returns:
            The released row index, or None.

        Example:
            >>> pool = DeltaNetStateSlots(num_slots=1)
            >>> _ = pool.allocate("req-a")
            >>> pool.free("req-a")
            0
            >>> pool.free("req-a") is None
            True
        """
        slot = self._slot_of_req.pop(req_id, None)
        if slot is None:
            return None
        self._free.append(slot)
        return slot

    def slots_for(self, req_ids: list[str]) -> list[int]:
        """Return one row for each request id, in the order of the argument.

        Allocate a row for a request that holds none. Use this to build the
        ``batch_slot_indices`` tensor for one forward pass.

        Args:
            req_ids: The request ids of the batch rows, in batch order.

        Returns:
            The row index for each request id, in the same order.

        Example:
            >>> pool = DeltaNetStateSlots(num_slots=2)
            >>> pool.slots_for(["req-a", "req-b"])
            [0, 1]
            >>> pool.slots_for(["req-b", "req-a"])  # order follows the argument
            [1, 0]
        """
        return [self.allocate(req_id) for req_id in req_ids]

    def slots_for_batch(self, req_ids: list[str], padded_size: int) -> list[int]:
        """Return one row for each row of the padded batch.

        A decode batch pads up to a compiled bucket, so it can hold more rows
        than there are requests. The model writes the state of every row back, so
        a padded row must not point at the row of a request. A padded row would
        then overwrite the state of that request with the result of a padded
        token. Give each padded row the same free row instead. The model writes
        only data that no request reads.

        A free row must exist when the batch pads. The request count of the batch
        does not give that row: a request that is not in the batch can hold a row,
        so the pool can be full while the batch holds fewer rows. Admission keeps
        1 row free for this reason. See `usable_state_rows`.

        A full batch has no padding, so it needs no free row.

        The write to the free row is safe. The next request that takes the row
        starts at position 0, so the reset in the model zeros the row first.

        Args:
            req_ids: The request ids of the real batch rows, in batch order.
            padded_size: The row count of the padded batch.

        Returns:
            One row index for each of the ``padded_size`` batch rows.

        Raises:
            ValueError: ``padded_size`` is smaller than the request count, or
                larger than the pool, or the batch has padding and the pool
                holds no free row.

        Example:
            >>> pool = DeltaNetStateSlots(num_slots=4)
            >>> pool.slots_for_batch(["req-a", "req-b"], padded_size=4)
            [0, 1, 2, 2]
            >>> pool.slots_for_batch(["req-a", "req-b"], padded_size=2)
            [0, 1]
        """
        if padded_size < len(req_ids):
            raise ValueError(
                f"The padded batch holds {padded_size} rows, which is fewer than "
                f"the {len(req_ids)} requests of the batch. Pad the decode batch "
                "up to a compiled bucket, not down."
            )
        if padded_size > self.num_slots:
            raise ValueError(
                f"The padded batch holds {padded_size} rows, but the DeltaNet "
                f"state buffers hold {self.num_slots} rows. Make the largest "
                "entry of num_seqs_buckets equal to max_num_seqs."
            )
        slots = [self.allocate(req_id) for req_id in req_ids]
        num_padded_rows = padded_size - len(slots)
        if num_padded_rows == 0:
            return slots
        if not self._free:
            raise ValueError(
                f"The padded batch needs {num_padded_rows} padded rows, but the "
                f"DeltaNet state pool holds no free row. The batch holds "
                f"{len(req_ids)} requests, and other requests hold the remaining "
                f"rows of the {self.num_slots} slots. Admission must keep "
                "1 row free: give the scheduler a limit of "
                f"{usable_state_rows(self.num_slots, None)} rows, or add a bucket "
                "for each count to num_seqs_buckets so that no batch pads."
            )
        # Peek, and do not take the row. The row stays free for the next request.
        scratch_slot = self._free[0]
        return slots + [scratch_slot] * num_padded_rows

    def reset(self) -> None:
        """Release every row. Use this between two independent runs.

        Example:
            >>> pool = DeltaNetStateSlots(num_slots=2)
            >>> _ = pool.slots_for(["req-a", "req-b"])
            >>> pool.reset()
            >>> pool.num_free
            2
        """
        self._slot_of_req.clear()
        self._free = deque(range(self.num_slots))
