# SPDX-License-Identifier: Apache-2.0
"""On-device block-based encoder cache for multimodal embeddings.

One mm_item per block (no cross-item sharing), pre-allocated contiguous
device buffer. The encoder graph writes directly into this buffer
via input-output aliasing (same pattern as KV cache). Eviction returns
blocks to free queue (no device-side memory ops).

Buffer layout::

    buffer: [num_cache_blocks, block_size, fat_dim]

    ┌──────────────┬──────────────┬──────────────┬───┬──────────────┐
    │ block 0      │ block 1      │ block 2      │...│ block N-1    │
    │[block_size,D]│[block_size,D]│[block_size,D]│   │ (scratch)    │
    └──────────────┴──────────────┴──────────────┴───┴──────────────┘

    slot_map (mm_hash → block_ids):
      "img_hash_A" → [0]         (1 image = 1 block)
      "img_hash_B" → [2]         (1 image = 1 block)
      "video_hash" → [3, 4, 5]   (1 video = multiple blocks, frames packed within)

    block N-1 is the scratch block (absorbs padding writes, never allocated).

Writes into the buffer happen inside the encoder graph via
input-output aliasing (same pattern as KV cache). Reads happen inside the
prefill graph: a pointer to the entire buffer is passed along with a block-id
index tensor, and the graph gathers the needed blocks via ``index_select``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections import deque
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


def mm_hash_fingerprint(mm_hash: str) -> int:
    """Deterministic, nonzero 63-bit fingerprint of an ``mm_hash``.

    This is not re-hashing for security: ``mm_hash`` is already a content hash,
    but an arbitrary-length string; we compress it to one fixed-width value so it
    fits the ``int64`` device tag slot the consumer compares after the RDMA pull.

    Used as the per-block content tag for the EPD RDMA receipt check: the VE
    producer stamps ``fingerprint(mm_hash)`` into a device-resident tag region,
    the PD consumer pulls the tag with the embedding and compares it to the
    fingerprint it expects. It must therefore be a pure function shared by both
    ends. The value is masked into ``[1, 2**63)`` so it fits a signed int64 tag
    tensor and 0 is reserved as the "empty/unwritten" sentinel.

    Args:
        mm_hash: The multimodal content hash keying a cache slot.

    Returns:
        A stable nonzero integer in ``[1, 2**63)``.

    Example:
        >>> fp = mm_hash_fingerprint("abc123")
        >>> fp == mm_hash_fingerprint("abc123") and fp != 0
        True
    """
    digest = hashlib.blake2b(mm_hash.encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "big") & ((1 << 63) - 1)
    # Reserve 0 for "empty"; map a zero digest to 1 so every real tag is nonzero.
    return value or 1


@dataclass
class SlotEntry:
    """Bookkeeping for one cached mm_item."""

    block_ids: list[int]
    tokens_per_block: list[int]
    write_time: float
    # Real (non-pad) merged *tokens* physically stored in each block, one entry
    # per block_id. For videos, when frames don't tile blocks exactly, a
    # non-last block has a trailing pad; for images, this is the dense expansion
    # [block_size, ..., remainder]. Readers use this to skip pad and enforce the
    # invariant: sum(tokens_per_block) == total valid tokens, len(tokens_per_block)
    # == len(block_ids) == block count.


class EncoderCacheBlocks:
    """Block-based on-device encoder cache (one mm_item per block).

    Pre-allocates a contiguous buffer [num_cache_blocks, block_size, fat_dim]
    on device at init time. Each mm_item occupies one or more blocks (one item
    per block, no cross-item packing).

    This class manages **allocation only** — it tracks which blocks are assigned
    to which mm_hash and maintains the free queue. Actual data writes happen
    inside the encoder graph (which takes `self.buffer` as an
    input-output alias and scatter-writes merged embeddings at the allocated
    block positions). Reads happen inside the prefill graph: the model runner
    passes a pointer to the entire buffer along with a block-id index tensor, and
    the graph gathers the needed blocks via ``index_select`` (a single coalesced
    gather rather than one strided DMA copy per block).

    Args:
        num_blocks: Total number of cache blocks.
        block_size: Tokens per block (post-merger, matches the encoder output block size).
        fat_dim: Embedding width = ``out_hidden_size * (1 + num_deepstack_levels)``.
            Concatenates the main encoder embedding with the deepstack
            residuals. Only Qwen3-VL has deepstack levels.
        dtype: Data type for the buffer.
        device: Device to allocate buffer on.
        min_hold_time_ms: Minimum time (ms) a block must stay allocated after
            write before it can be freed. Prevents premature reuse when a
            remote reader hasn't finished pulling. 0 = instant free.
        model_id: Identifier of the model this cache serves (e.g.
            ``model_config.model``). Read by the EPD NIXL handshake to reject a
            PD pool pointed at a VE serving a different model. ``None`` (the
            default) disables that cross-pool check — fine for monolithic and
            existing tests.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        fat_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
        min_hold_time_ms: float = 100.0,
        model_id: str | None = None,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.fat_dim = fat_dim
        self.dtype = dtype
        self._device = device
        self.min_hold_time_ms = min_hold_time_ms
        self.model_id = model_id

        # Last block is reserved as scratch — absorbs padding writes from
        # encoder bucket padding. Never allocated to real items.
        self.scratch_block_id = num_blocks - 1

        self.buffer = torch.zeros(
            (num_blocks, block_size, fat_dim), dtype=dtype, device=device
        )

        # Per-block content tag: fingerprint of the mm_hash owning each block,
        # stamped at allocate, before the encoder writes the embedding.
        # A parallel NIXL-registered region on EPD, so the PD consumer can pull
        # it with the embedding and verify it received the block it expected
        # (see mm_hash_fingerprint). 0 = empty/unwritten.
        self.mm_hash_tags = torch.zeros(num_blocks, dtype=torch.int64, device=device)

        self._slot_map: dict[str, SlotEntry] = {}
        # Free queue excludes the scratch block
        self._free_queue: deque[int] = deque(range(num_blocks - 1))
        # mm_hash → (block_ids, write_time) for blocks in hold-time window
        self._held_blocks: dict[str, tuple[list[int], float]] = {}

        logger.debug(
            "[EncoderCacheBlocks] init: num_blocks=%d (allocatable=%d, scratch=%d), "
            "block_size=%d, fat_dim=%d, min_hold_time_ms=%.0f, "
            "buffer=%.0f MB, device=%s",
            num_blocks,
            num_blocks - 1,
            self.scratch_block_id,
            block_size,
            fat_dim,
            min_hold_time_ms,
            num_blocks * block_size * fat_dim * 2 / (1024 * 1024),
            device,
        )

    @property
    def num_free_blocks(self) -> int:
        """Blocks currently in the free queue (allocatable now)."""
        return len(self._free_queue)

    @property
    def num_active_blocks(self) -> int:
        """Blocks currently allocated to cached mm items."""
        return sum(len(e.block_ids) for e in self._slot_map.values())

    @property
    def num_held_blocks(self) -> int:
        """Blocks freed but still within the min-hold-time window (not yet
        returned to the free queue)."""
        return sum(len(block_ids) for block_ids, _ in self._held_blocks.values())

    @property
    def num_active_real_tokens(self) -> int:
        """Real (non-padding) merged tokens stored across all active blocks.

        Each block holds ``block_size`` slots but an item may use fewer, so this
        is < ``num_active_blocks * block_size``; the difference is padding waste.
        """
        return sum(sum(e.tokens_per_block) for e in self._slot_map.values())

    def _padding_waste_summary(self) -> str:
        """Human-readable snapshot of how much allocated block capacity is
        real tokens vs. padding waste.

        Each cached item occupies whole block(s) regardless of token count, so
        many small items leave most of each block empty. This ratio is the
        clearest signal of the scheduler/block-allocator budget mismatch and
        drives the sizing recommendation in the cache-full error.
        """
        active_blocks = self.num_active_blocks
        if active_blocks == 0:
            return "no active items"
        active_tokens = self.num_active_real_tokens
        capacity = active_blocks * self.block_size
        util_pct = 100.0 * active_tokens / capacity
        return (
            f"{active_tokens}/{capacity} tokens used across {active_blocks} "
            f"allocated block(s) ({util_pct:.0f}% utilized, "
            f"{100.0 - util_pct:.0f}% padding waste)"
        )

    def contains(self, mm_hash: str) -> bool:
        return mm_hash in self._slot_map

    @staticmethod
    def dense_tokens_per_block(num_tokens: int, block_size: int) -> list[int]:
        """Dense per-block token counts for a contiguous item: [block_size, ..., remainder]."""
        n = math.ceil(num_tokens / block_size)
        return [min(block_size, num_tokens - j * block_size) for j in range(n)]

    def allocate(
        self,
        mm_hash: str,
        tokens_per_block: list[int],
    ) -> list[int]:
        """Reserve blocks for an mm_item. No data is written.

        The caller passes the returned block_ids to the encoder graph
        as `write_block_ids` so it can scatter-write directly into
        `self.buffer` at those positions.

        Self-healing: if free blocks are insufficient, attempts to reclaim
        expired held blocks. If still insufficient and held blocks exist
        that haven't expired yet, waits (bounded by min_hold_time_ms)
        for them to expire before giving up.

        Args:
            mm_hash: Unique identifier for this mm_item.
            tokens_per_block: Real (non-pad) merged tokens per reserved block,
                one entry per block; its length sets the block count, which can
                exceed the dense ceil(num_tokens / block_size) when a video's
                whole-frame packing leaves per-block pad. Stored on the
                SlotEntry so readers skip each block's pad tail.

        Returns:
            List of allocated block_ids.
            If mm_hash is already allocated, returns existing block_ids.

        Raises:
            RuntimeError: If the cache cannot satisfy the request after
                reclaiming all held blocks.
        """
        t0 = time.monotonic()
        existing = self._slot_map.get(mm_hash)
        if existing is not None:
            logger.debug(
                "[EncoderCacheBlocks] allocate(%s): cache hit, returning existing block_ids=%s",
                mm_hash[:12],
                existing.block_ids,
            )
            return existing.block_ids

        # If this mm_hash was recently freed and is in held state, reclaim
        # its blocks immediately — re-allocating the same item means the
        # hold is no longer needed (we're about to overwrite the data).
        held = self._held_blocks.pop(mm_hash, None)
        if held is not None:
            held_block_ids, _ = held
            for block_id in held_block_ids:
                self._free_queue.append(block_id)
            logger.debug(
                "[EncoderCacheBlocks] allocate(%s): reclaimed %d held blocks for same hash",
                mm_hash[:12],
                len(held_block_ids),
            )

        num_needed = len(tokens_per_block)

        if num_needed > len(self._free_queue):
            reclaimed = self.reclaim_held()
            logger.debug(
                "[EncoderCacheBlocks] allocate(%s): reclaimed %d expired held blocks",
                mm_hash[:12],
                reclaimed,
            )

        if num_needed > len(self._free_queue) and self._held_blocks:
            logger.warning(
                "[EncoderCacheBlocks] cache pressure: need %d blocks but only "
                "%d free. Waiting up to %.0fms per block for %d held block(s) to expire. "
                "This stalls the inference thread and degrades TTFT. "
                "Cause: per-block padding waste — the scheduler's token-based "
                "budget over-admits small items relative to the physical block "
                "layout (%s). If this recurs, raise the encoder cache capacity "
                "via EncoderNeuronConfig.encoder_cache_num_blocks (currently %d) "
                "in the encoder_neuron_config additional_config.",
                num_needed,
                len(self._free_queue),
                self.min_hold_time_ms,
                len(self._held_blocks),
                self._padding_waste_summary(),
                self.num_blocks,
            )
            self._wait_and_reclaim(num_needed)

        if num_needed > len(self._free_queue):
            deficit = num_needed - len(self._free_queue)
            suggested = self.num_blocks + deficit
            raise RuntimeError(
                f"Encoder cache full: cannot allocate mm_hash={mm_hash} "
                f"({sum(tokens_per_block)} tokens, {num_needed} block(s) needed, "
                f"{len(self._free_queue)} free of {self.num_blocks - 1} allocatable, "
                f"{len(self._slot_map)} active, {len(self._held_blocks)} held).\n"
                f"Cause: each cached item occupies whole block(s) regardless of "
                f"token count, but the scheduler's EncoderCacheManager admits "
                f"items by a token budget that ignores this per-block padding. "
                f"Current buffer state: {self._padding_waste_summary()}. With "
                f"many small images the block allocator fills up before the "
                f"scheduler's token budget does.\n"
                f"Fix: raise the on-device cache capacity by setting "
                f"'encoder_cache_num_blocks' in the encoder "
                f"additional_config to at least {suggested} (currently "
                f"{self.num_blocks}); size it to cover expected concurrent items "
                f"divided by average per-block utilization to absorb padding "
                f"waste. See EncoderNeuronConfig.encoder_cache_num_blocks. "
                f"If held blocks dominate, also consider lowering "
                f"encoder_cache_min_hold_time_ms (currently {self.min_hold_time_ms:.0f}ms)."
            )

        block_ids = [self._free_queue.popleft() for _ in range(num_needed)]

        self._slot_map[mm_hash] = SlotEntry(
            block_ids=block_ids,
            tokens_per_block=tokens_per_block,
            write_time=0.0,
        )
        # Stamp the per-block content tag now, before the encoder writes
        # the embedding. This orders the tag write ahead of the embedding write.
        # If a block's embedding changes, its tag changed first. An EPD consumer
        # reads the tag to confirm the bytes it pulls belong to this mm_hash.
        fp = mm_hash_fingerprint(mm_hash)
        for block_id in block_ids:
            self.mm_hash_tags[block_id] = fp
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug(
            "[EncoderCacheBlocks] allocate(%s): %d tokens → %d blocks, free=%d, active=%d, held=%d, took=%.1fms",
            mm_hash[:12],
            sum(tokens_per_block),
            num_needed,
            len(self._free_queue),
            len(self._slot_map),
            len(self._held_blocks),
            elapsed_ms,
        )
        return block_ids

    def mark_written(self, mm_hash: str) -> None:
        """Update write_time after data is valid in the buffer.

        Called after embed_multimodal() completes — the encoder graph
        has finished writing. This ensures the hold-time guard measures from
        when data is actually available, not from when blocks were reserved.
        """
        entry = self._slot_map.get(mm_hash)
        if entry is not None:
            entry.write_time = time.monotonic()
            logger.debug(
                "[EncoderCacheBlocks] mark_written(%s): block_ids=%s",
                mm_hash[:12],
                entry.block_ids,
            )

    def _wait_and_reclaim(self, num_needed: int) -> None:
        """Wait for held blocks to expire until enough are free.

        Sleeps in short intervals until either enough blocks are reclaimed
        or all held blocks have been reclaimed (and it's still not enough).
        Total wait is bounded by min_hold_time_ms.
        """
        deadline = time.monotonic() + self.min_hold_time_ms / 1000
        poll_interval = 0.001  # 1ms

        while len(self._free_queue) < num_needed and self._held_blocks:
            now = time.monotonic()
            if now >= deadline:
                break
            # Sleep until the next earliest expiry or poll_interval
            earliest_expiry = min(
                wt + self.min_hold_time_ms / 1000
                for _, (_, wt) in self._held_blocks.items()
            )
            sleep_time = min(earliest_expiry - now, poll_interval)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.reclaim_held()

    def get_block_ids(self, mm_hash: str) -> list[int] | None:
        """Get allocated block IDs for a cached mm_item.

        Used by the model runner to build the block-id index tensor for
        the prefill graph input (the whole cache buffer is passed alongside).
        """
        entry = self._slot_map.get(mm_hash)
        if entry is None:
            return None
        return entry.block_ids

    def get_num_tokens(self, mm_hash: str) -> int | None:
        """Get the number of valid tokens stored for an mm_item.

        Used to compute encoder_positions and determine how many tokens
        to gather from each block.
        """
        entry = self._slot_map.get(mm_hash)
        if entry is None:
            return None
        return sum(entry.tokens_per_block)

    def check_num_tokens(self, mm_hash: str, expected_num_tokens: int) -> str | None:
        """Validate a cached item's stored token count against the reader's expectation.

        The stored count comes from the encode/bucket path (``tokens_per_block``
        set at allocate); ``expected_num_tokens`` comes from the sequence
        placeholder the reader is filling. They are produced by independent code
        paths, so a mismatch flags a real fault — a merge/bucket/placeholder bug
        in monolithic, or (on EPD) a ``locator["num_tokens"]`` that disagrees
        with the sequence after a truncated/mis-routed transfer.

        Args:
            mm_hash: The cached item to check.
            expected_num_tokens: Embedding-position count the reader expects.

        Returns:
            ``None`` if the item is resident and its token count matches;
            otherwise a human-readable mismatch string (also covers a missing
            entry, though callers normally detect that via ``get_block_ids``).

        Example:
            >>> cache.check_num_tokens("img_a", 256)  # matches
            >>> cache.check_num_tokens("img_a", 128)  # doctest: +SKIP
            'mm_hash=img_a... token-count mismatch: stored 256 != expected 128'
        """
        stored = self.get_num_tokens(mm_hash)
        if stored is None:
            return f"mm_hash={mm_hash[:12]}... not resident"
        if stored != expected_num_tokens:
            return (
                f"mm_hash={mm_hash[:12]}... token-count mismatch: "
                f"stored {stored} != expected {expected_num_tokens}"
            )
        return None

    def get_tokens_per_block(self, mm_hash: str) -> list[int] | None:
        """Real (non-pad) merged tokens stored in each block for an mm_item.

        For videos, when frames don't tile blocks exactly, non-last blocks may
        have pad; for images, this is the dense expansion [block_size, ..., remainder].
        Readers use this to skip pad. Always an explicit list; sum == get_num_tokens.
        """
        entry = self._slot_map.get(mm_hash)
        if entry is None:
            return None
        return entry.tokens_per_block

    def free(self, mm_hash: str) -> int:
        """Free blocks for an mm_hash, returning them to the free queue.

        If min_hold_time_ms > 0 and the entry was written recently, the
        blocks are held and will be reclaimed on a later reclaim_held() call.

        Returns:
            Number of blocks evicted from the cache (0 if mm_hash was not
            cached). The entry leaves the cache regardless of whether its
            blocks are immediately freed or held, so a held eviction still
            returns its block count. The caller owns metric emission.
        """
        entry = self._slot_map.pop(mm_hash, None)
        if entry is None:
            return 0

        num_evicted = len(entry.block_ids)

        if self.min_hold_time_ms > 0:
            elapsed_ms = (time.monotonic() - entry.write_time) * 1000
            if elapsed_ms < self.min_hold_time_ms:
                self._held_blocks[mm_hash] = (entry.block_ids, entry.write_time)
                logger.info(
                    "[EncoderCacheBlocks] evict(%s): HELD %d blocks=%s "
                    "(elapsed=%.0fms < hold=%.0fms), free=%d, active=%d, held=%d",
                    mm_hash[:12],
                    len(entry.block_ids),
                    entry.block_ids,
                    elapsed_ms,
                    self.min_hold_time_ms,
                    len(self._free_queue),
                    len(self._slot_map),
                    len(self._held_blocks),
                )
                return num_evicted

        elapsed_ms_free = (
            (time.monotonic() - entry.write_time) * 1000
            if entry.write_time > 0.0
            else None
        )
        for block_id in entry.block_ids:
            self._free_queue.append(block_id)
            self.mm_hash_tags[block_id] = 0
        logger.info(
            "[EncoderCacheBlocks] evict(%s): FREED %d blocks=%s (elapsed=%s), "
            "free=%d, active=%d, held=%d",
            mm_hash[:12],
            len(entry.block_ids),
            entry.block_ids,
            f"{elapsed_ms_free:.0f}ms" if elapsed_ms_free is not None else "unwritten",
            len(self._free_queue),
            len(self._slot_map),
            len(self._held_blocks),
        )
        return num_evicted

    def reclaim_held(self) -> int:
        """Reclaim blocks whose hold time has expired.

        Returns:
            Number of blocks reclaimed.
        """
        if not self._held_blocks:
            return 0

        reclaimed = 0
        expired = []
        now = time.monotonic()
        for mm_hash, (block_ids, write_time) in self._held_blocks.items():
            if (now - write_time) * 1000 >= self.min_hold_time_ms:
                expired.append(mm_hash)

        for mm_hash in expired:
            block_ids, _ = self._held_blocks.pop(mm_hash)
            for block_id in block_ids:
                self._free_queue.append(block_id)
                self.mm_hash_tags[block_id] = 0
            reclaimed += len(block_ids)

        return reclaimed

    def clear(self) -> None:
        """Reset all state, returning all blocks to free queue."""
        self._slot_map.clear()
        self._held_blocks.clear()
        self._free_queue = deque(range(self.num_blocks - 1))  # exclude scratch
        self.buffer.zero_()
        self.mm_hash_tags.zero_()

    # ------------------------------------------------------------------
    # Read/write helpers for snapshot, serialization, and prompt_embeds.
    # ------------------------------------------------------------------

    def get(self, mm_hash: str) -> torch.Tensor | None:
        """Read cached entry by gathering from buffer.

        Returns [num_tokens, fat_dim] or None. Used by snapshot/serialization
        path — not the hot path (prefill gathers from the buffer in-graph).
        """
        entry = self._slot_map.get(mm_hash)
        if entry is None:
            return None

        blocks = []
        for block_id, chunk_size in zip(entry.block_ids, entry.tokens_per_block):
            blocks.append(self.buffer[block_id, :chunk_size])

        if len(blocks) == 1:
            return blocks[0]
        return torch.cat(blocks, dim=0)

    def put(self, mm_hash: str, tensor: torch.Tensor) -> None:
        """Allocate blocks and write tensor data into the buffer.

        Used by the prompt_embeds passthrough path, which stores the
        embeddings directly. An encoder graph writes through index_put_
        instead, so this method is not on the encoder hot path.
        """
        if mm_hash in self._slot_map:
            return
        num_tokens = tensor.shape[0]
        block_ids = self.allocate(
            mm_hash, self.dense_tokens_per_block(num_tokens, self.block_size)
        )
        tokens_written = 0
        for block_id in block_ids:
            chunk_size = min(self.block_size, num_tokens - tokens_written)
            self.buffer[block_id, :chunk_size] = tensor[
                tokens_written : tokens_written + chunk_size
            ]
            if chunk_size < self.block_size:
                self.buffer[block_id, chunk_size:] = 0
            tokens_written += chunk_size
        self.mark_written(mm_hash)

    def items(self):
        """Iterate (mm_hash, tensor) pairs. Used by snapshot/serialization."""
        for mm_hash in self._slot_map:
            yield mm_hash, self.get(mm_hash)
