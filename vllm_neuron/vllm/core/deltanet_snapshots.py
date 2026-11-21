# SPDX-License-Identifier: Apache-2.0
"""Map a cached prompt prefix to a saved DeltaNet state.

Prefix caching skips the prefill of the tokens that the KV cache already holds.
A DeltaNet layer keeps a recurrent state and a conv state instead of a KV cache,
and the paged KV pool cannot hold this state on Neuron (the state page is larger
than the largest attention page). So the layer keeps a second buffer of state
rows, the snapshot rows. One snapshot row holds the state after exactly ``P``
prompt tokens.

This module owns the host side of the snapshots:

- the table from the block hash at position ``P`` to a snapshot row, with LRU
  eviction;
- the positions where a prefill step must stop, so that the state at that
  position exists at the end of a step and the step can save it.

The model reads and writes the snapshot rows inside the prefill graph. Row 0 is
a scratch row: a step that saves nothing writes row 0, and a step that restores
nothing reads its own state row. So row 0 never holds a valid snapshot.

This module imports nothing from vLLM, so the scheduler and the tests share one
definition.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

#: The snapshot row that holds no valid state.
SCRATCH_ROW = 0

#: The values of the save policy.
SNAPSHOT_POLICIES = ("both", "tail", "junction")


@dataclass
class SnapshotStats:
    """Counts of the snapshot table since the start of the engine."""

    lookups: int = 0
    hits: int = 0
    restored_tokens: int = 0
    clamped_tokens: int = 0
    saves: int = 0
    evictions: int = 0


class DeltaNetSnapshotTable:
    """Find, save and evict the DeltaNet state snapshots of prompt prefixes.

    A snapshot is valid at a position ``P`` that is a multiple of ``align``.
    The key of a snapshot is the block hash of the last full block before ``P``.
    vLLM chains each block hash to the hash of the block before it, so the key
    identifies the whole prefix ``[0, P)``.

    Two rules decide where a prefill step stops to save a snapshot:

    - **Tail:** the last aligned position that is at least ``tail_margin``
      tokens before the prompt end. The end of a chat prompt holds the parts
      that change between requests (a request ID, the chat template suffix), so
      the margin puts the save point before them.
    - **Junction:** the end of the KV cache hit, when the KV cache holds more of
      the prompt than the snapshots do. The next request with the same prefix
      then restores the state at the junction. A junction is used only when it
      is at least ``junction_min_tokens`` past the restored position.

    Args:
        num_rows: The row count of the snapshot buffers, row 0 included.
        block_size: The KV cache block size in tokens.
        align: The alignment of a snapshot position in tokens. It must be a
            multiple of ``block_size``.
        tail_margin: The minimum token count between the tail save point and
            the prompt end.
        policy: "both", "tail" or "junction".
        junction_min_tokens: The minimum gain of a junction save, in tokens.

    Example:
        >>> table = DeltaNetSnapshotTable(4, block_size=64, align=128, tail_margin=64)
        >>> table.split_end(start=0, end=1000, num_prompt_tokens=1000, junction=0)
        896
        >>> hashes = [f"h{i}" for i in range(15)]
        >>> table.save(hashes[896 // 64 - 1])
        1
        >>> table.lookup(hashes, max_tokens=999)
        (896, 1)
    """

    def __init__(
        self,
        num_rows: int,
        block_size: int,
        align: int = 128,
        tail_margin: int = 64,
        policy: str = "both",
        junction_min_tokens: int = 512,
    ) -> None:
        if num_rows < 2:
            raise ValueError(
                f"The snapshot pool has {num_rows} rows. Row 0 is a scratch row, so "
                "the pool needs at least 2 rows. Set "
                "VLLM_NEURON_BACKEND_DELTANET_SNAPSHOT_ROWS to 2 or more."
            )
        if block_size <= 0 or align <= 0 or align % block_size != 0:
            raise ValueError(
                f"The snapshot alignment ({align}) is not a multiple of the KV block "
                f"size ({block_size}). Set VLLM_NEURON_BACKEND_DELTANET_SNAPSHOT_ALIGN "
                "to a multiple of the block size."
            )
        if policy not in SNAPSHOT_POLICIES:
            raise ValueError(
                f"The snapshot policy {policy!r} is not known. Use one of "
                f"{', '.join(SNAPSHOT_POLICIES)}."
            )
        self.num_rows = num_rows
        self.block_size = block_size
        self.align = align
        self.tail_margin = max(0, tail_margin)
        self.policy = policy
        self.junction_min_tokens = max(0, junction_min_tokens)
        # Key -> row, in LRU order (the first entry is the least recently used).
        self._rows: OrderedDict[Hashable, int] = OrderedDict()
        self._free: list[int] = list(range(num_rows - 1, SCRATCH_ROW, -1))
        self.stats = SnapshotStats()

    def reset(self) -> None:
        """Forget every snapshot. Call this when the prefix cache is reset.

        Example:
            >>> table = DeltaNetSnapshotTable(2, block_size=64)
            >>> table.save("a")
            1
            >>> table.reset()
            >>> table.num_valid
            0
        """
        self._rows.clear()
        self._free = list(range(self.num_rows - 1, SCRATCH_ROW, -1))

    @property
    def num_valid(self) -> int:
        """Return the count of rows that hold a valid snapshot."""
        return len(self._rows)

    def _key_at(self, block_hashes: Sequence[Hashable], pos: int) -> Hashable | None:
        index = pos // self.block_size - 1
        if index < 0 or index >= len(block_hashes):
            return None
        return block_hashes[index]

    def tail_point(self, num_prompt_tokens: int) -> int:
        """Return the tail save position of a prompt, or 0 when there is none.

        Args:
            num_prompt_tokens: The prompt length in tokens.

        Returns:
            The last multiple of ``align`` that is at least ``tail_margin``
            tokens before the prompt end.

        Example:
            >>> DeltaNetSnapshotTable(2, 64).tail_point(1000)
            896
            >>> DeltaNetSnapshotTable(2, 64).tail_point(150)
            0
        """
        if self.policy == "junction":
            return 0
        return max(0, (num_prompt_tokens - self.tail_margin) // self.align * self.align)

    def junction_point(self, junction: int, start: int) -> int:
        """Return the aligned junction save position, or 0 when it is not used.

        Args:
            junction: The end of the KV cache hit in tokens, or 0.
            start: The position where the computation of the prompt starts.

        Returns:
            The junction floored to ``align``, when the policy allows a junction
            and the gain over ``start`` is at least ``junction_min_tokens``.

        Example:
            >>> DeltaNetSnapshotTable(2, 64).junction_point(2000, start=0)
            1920
            >>> DeltaNetSnapshotTable(2, 64).junction_point(2000, start=1792)
            0
        """
        if self.policy == "tail" or junction <= 0:
            return 0
        point = junction // self.align * self.align
        if point - start < max(self.junction_min_tokens, 1):
            return 0
        return point

    def split_end(
        self, start: int, end: int, num_prompt_tokens: int, junction: int
    ) -> int:
        """Return the end of a prefill step, cut at the next save point.

        Args:
            start: The first token of the step.
            end: The end of the step before the cut.
            num_prompt_tokens: The prompt length in tokens.
            junction: The junction of the request (see ``junction_point``).

        Returns:
            The earliest save point strictly inside ``(start, end)``, or ``end``.

        Example:
            >>> table = DeltaNetSnapshotTable(2, 64)
            >>> table.split_end(0, 8192, 2500, junction=0)
            2432
            >>> table.split_end(2432, 2500, 2500, junction=0)
            2500
        """
        stops = [self.tail_point(num_prompt_tokens)]
        # The junction is measured from the restored position, which is the
        # start of the first step of the request.
        stops.append(self.junction_point(junction, start))
        inside = [s for s in stops if start < s < end]
        return min(inside) if inside else end

    def is_save_point(self, end: int, num_prompt_tokens: int, junction: int) -> bool:
        """Return True when a step that ends at ``end`` must save the state.

        Args:
            end: The end of the step in tokens.
            num_prompt_tokens: The prompt length in tokens.
            junction: The junction of the request (see ``junction_point``).

        Returns:
            True for the tail point and for an aligned junction inside the
            prompt.

        Example:
            >>> table = DeltaNetSnapshotTable(2, 64)
            >>> table.is_save_point(2432, 2500, junction=0)
            True
            >>> table.is_save_point(2500, 2500, junction=0)
            False
        """
        if end <= 0 or end >= num_prompt_tokens or end % self.align != 0:
            return False
        if end == self.tail_point(num_prompt_tokens):
            return True
        return (
            self.policy != "tail"
            and junction > 0
            and end == junction // self.align * self.align
        )

    def lookup(
        self, block_hashes: Sequence[Hashable], max_tokens: int
    ) -> tuple[int, int]:
        """Find the longest saved prefix of at most ``max_tokens`` tokens.

        A hit moves the snapshot to the most recently used end of the LRU order.

        Args:
            block_hashes: The block hashes of the request, one for each full block.
            max_tokens: The length of the KV cache hit. A snapshot past it has no
                KV, so it cannot serve the request.

        Returns:
            ``(tokens, row)`` of the longest saved prefix, or ``(0, SCRATCH_ROW)``.

        Example:
            >>> table = DeltaNetSnapshotTable(3, block_size=64)
            >>> hashes = [f"h{i}" for i in range(8)]
            >>> table.lookup(hashes, 512)
            (0, 0)
            >>> table.save(hashes[3])
            1
            >>> table.lookup(hashes, 512)
            (256, 1)
        """
        self.stats.lookups += 1
        pos = min(max_tokens, len(block_hashes) * self.block_size)
        pos = pos // self.align * self.align
        while pos > 0:
            key = self._key_at(block_hashes, pos)
            row = self._rows.get(key) if key is not None else None
            if row is not None:
                self._rows.move_to_end(key)
                self.stats.hits += 1
                return pos, row
            pos -= self.align
        return 0, SCRATCH_ROW

    def save(self, key: Hashable, keep_row: int = SCRATCH_ROW) -> int:
        """Give a snapshot row to ``key`` and return it.

        A key that already has a row keeps it. The step then writes the same
        state again. Otherwise the table takes a free row, or it evicts the least
        recently used snapshot.

        Args:
            key: The block hash at the save position.
            keep_row: A row that the same step reads. The table does not evict it.

        Returns:
            The snapshot row that the step must write.

        Example:
            >>> table = DeltaNetSnapshotTable(3, block_size=64)
            >>> table.save("a"), table.save("b"), table.save("c")
            (1, 2, 1)
            >>> table.save("b")
            2
        """
        row = self._rows.get(key)
        if row is not None:
            self._rows.move_to_end(key)
        elif self._free:
            row = self._free.pop()
        else:
            victim = next((k for k, r in self._rows.items() if r != keep_row), None)
            if victim is None:
                return SCRATCH_ROW
            row = self._rows.pop(victim)
            self.stats.evictions += 1
        self._rows[key] = row
        self.stats.saves += 1
        return row

    def save_at(
        self, block_hashes: Sequence[Hashable], pos: int, keep_row: int = SCRATCH_ROW
    ) -> int:
        """Give a snapshot row to the prefix ``[0, pos)`` and return it.

        Args:
            block_hashes: The block hashes of the request.
            pos: The save position. It must be a multiple of ``align``.
            keep_row: A row that the same step reads. The table does not evict it.

        Returns:
            The snapshot row to write, or ``SCRATCH_ROW`` when the request has no
            hash for that position.

        Example:
            >>> table = DeltaNetSnapshotTable(2, block_size=64)
            >>> table.save_at(["h0", "h1"], 128)
            1
            >>> table.save_at(["h0"], 128)
            0
        """
        key = self._key_at(block_hashes, pos)
        if key is None or pos % self.align != 0:
            return SCRATCH_ROW
        return self.save(key, keep_row=keep_row)
