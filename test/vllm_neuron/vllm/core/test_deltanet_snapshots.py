# SPDX-License-Identifier: Apache-2.0
"""Tests for the DeltaNet snapshot table of prefix caching.

The table maps the block hash at a prompt position to a snapshot row, and it
gives the positions where a prefill step must stop to save a snapshot. A wrong
answer here gives a cache hit a state that belongs to a different prefix, so the
model output is wrong with no error.

CPU-only; no Neuron device.
"""

import pytest

from vllm_neuron.vllm.core.deltanet_snapshots import (
    SCRATCH_ROW,
    DeltaNetSnapshotTable,
)

BLOCK = 16


def _hashes(prefix: str, num_blocks: int) -> list[str]:
    """Return a hash chain: entry i stands for the prefix of blocks 0..i."""
    return [f"{prefix}{i}" for i in range(num_blocks)]


def _table(**kwargs) -> DeltaNetSnapshotTable:
    args = dict(num_rows=4, block_size=BLOCK, align=32, tail_margin=16)
    args.update(kwargs)
    return DeltaNetSnapshotTable(**args)


class TestValidation:
    def test_one_row_is_rejected_because_row_0_is_scratch(self):
        with pytest.raises(ValueError, match="at least 2 rows"):
            _table(num_rows=1)

    def test_an_alignment_that_is_not_a_block_multiple_is_rejected(self):
        with pytest.raises(ValueError, match="not a multiple of the KV block"):
            _table(align=24)

    def test_an_unknown_policy_is_rejected(self):
        with pytest.raises(ValueError, match="policy 'all' is not known"):
            _table(policy="all")


class TestSavePoints:
    def test_the_tail_point_stays_a_margin_before_the_prompt_end(self):
        table = _table()
        # (200 - 16) // 32 * 32 = 160
        assert table.tail_point(200) == 160
        assert 200 - table.tail_point(200) >= table.tail_margin

    def test_a_prompt_shorter_than_the_margin_has_no_tail_point(self):
        assert _table().tail_point(40) == 0

    def test_the_junction_policy_has_no_tail_point(self):
        assert _table(policy="junction").tail_point(200) == 0

    def test_a_step_stops_at_the_tail_point(self):
        table = _table()
        assert table.split_end(0, 200, 200, junction=0) == 160
        # The next step runs to the prompt end.
        assert table.split_end(160, 200, 200, junction=0) == 200

    def test_a_step_that_ends_before_the_tail_point_does_not_change(self):
        assert _table().split_end(0, 128, 200, junction=0) == 128

    def test_a_step_stops_at_the_earlier_of_the_junction_and_the_tail(self):
        table = _table(junction_min_tokens=32)
        # Tail point 480, junction 256 (gain 256 >= 32).
        assert table.split_end(0, 512, 512, junction=256) == 256
        assert table.split_end(256, 512, 512, junction=256) == 480

    def test_a_small_junction_gain_does_not_stop_a_step(self):
        table = _table(junction_min_tokens=512)
        assert table.split_end(0, 512, 512, junction=256) == 480

    def test_the_tail_policy_ignores_the_junction(self):
        table = _table(policy="tail", junction_min_tokens=0)
        assert table.split_end(0, 512, 512, junction=256) == 480
        assert not table.is_save_point(256, 512, junction=256)

    def test_the_save_points_are_the_tail_and_the_junction(self):
        table = _table(junction_min_tokens=32)
        assert table.is_save_point(160, 200, junction=0)
        assert table.is_save_point(96, 200, junction=100)
        assert not table.is_save_point(128, 200, junction=0)
        # The prompt end is never a save point: decode changes the state after it.
        assert not table.is_save_point(200, 200, junction=0)
        assert not table.is_save_point(0, 200, junction=0)


class TestLookupAndSave:
    def test_an_empty_table_has_no_hit(self):
        assert _table().lookup(_hashes("a", 12), 192) == (0, SCRATCH_ROW)

    def test_a_hit_returns_the_longest_saved_prefix(self):
        table = _table()
        hashes = _hashes("a", 12)
        row_64 = table.save_at(hashes, 64)
        row_160 = table.save_at(hashes, 160)
        assert table.lookup(hashes, 192) == (160, row_160)
        # A shorter KV hit cannot use the snapshot at 160.
        assert table.lookup(hashes, 159) == (64, row_64)

    def test_a_different_prefix_does_not_hit(self):
        table = _table()
        table.save_at(_hashes("a", 12), 160)
        assert table.lookup(_hashes("b", 12), 192) == (0, SCRATCH_ROW)

    def test_a_hit_must_be_aligned(self):
        table = _table()
        hashes = _hashes("a", 12)
        row = table.save_at(hashes, 160)
        # 176 is a block boundary but not an aligned position.
        assert table.save_at(hashes, 176) == SCRATCH_ROW
        assert table.lookup(hashes, 191) == (160, row)

    def test_row_0_is_never_given_to_a_snapshot(self):
        table = _table(num_rows=3)
        rows = {table.save(f"k{i}") for i in range(10)}
        assert SCRATCH_ROW not in rows
        assert rows == {1, 2}

    def test_the_least_recently_used_snapshot_is_evicted(self):
        table = _table(num_rows=3)
        a, b = _hashes("a", 4), _hashes("b", 4)
        row_a = table.save_at(a, 32)
        table.save_at(b, 32)
        # A hit on a makes b the least recently used snapshot.
        assert table.lookup(a, 64) == (32, row_a)
        table.save_at(_hashes("c", 4), 32)
        assert table.lookup(b, 64) == (0, SCRATCH_ROW)
        assert table.lookup(a, 64) == (32, row_a)
        assert table.stats.evictions == 1

    def test_a_save_does_not_evict_the_row_that_the_same_step_restores(self):
        table = _table(num_rows=3)
        a, b = _hashes("a", 2), _hashes("b", 2)
        row_a = table.save_at(a, 32)
        table.save_at(b, 32)
        # The snapshot of a is the least recently used one, but the step that
        # saves c restores a. So the table evicts b.
        row_c = table.save_at(_hashes("c", 2), 32, keep_row=row_a)
        assert row_c != row_a
        assert table.lookup(a, 32) == (32, row_a)
        assert table.lookup(b, 32) == (0, SCRATCH_ROW)

    def test_a_saved_key_keeps_its_row(self):
        table = _table()
        row = table.save("a")
        assert table.save("a") == row
        assert table.num_valid == 1
