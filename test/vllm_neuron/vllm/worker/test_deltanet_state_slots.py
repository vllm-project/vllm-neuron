# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DeltaNet state slot pool.

A linear attention layer keeps a recurrent state and a convolution state for each
concurrent request. The pool assigns one state row to each request. The model
then indexes the state buffers with ``batch_slot_indices``.

Two properties carry the correctness of the port. First, ``allocate`` must be
idempotent, because one request spans many prefill chunks and many decode steps;
a request that changed row would read the state of another request. Second, a
released row must return to the pool, so a long run does not exhaust it.

CPU-only; no Neuron device and no vLLM import.
"""

import pytest

from vllm_neuron.vllm.worker.deltanet_state_slots import (
    DeltaNetStateSlots,
    usable_state_rows,
)


def test_rejects_an_empty_pool():
    with pytest.raises(ValueError) as excinfo:
        DeltaNetStateSlots(num_slots=0)
    assert "at least one slot" in str(excinfo.value)


def test_allocate_gives_a_row_and_counts_it():
    pool = DeltaNetStateSlots(num_slots=2)
    assert pool.num_free == 2
    assert pool.allocate("req-a") == 0
    assert pool.num_free == 1
    assert pool.num_active == 1
    assert pool.contains("req-a")
    assert pool.get("req-a") == 0


def test_allocate_is_idempotent():
    # One request spans many prefill chunks and decode steps. It must keep one
    # row for all of them.
    pool = DeltaNetStateSlots(num_slots=2)
    first = pool.allocate("req-a")
    for _ in range(58):  # a 117k prompt needs 58 prefill chunks
        assert pool.allocate("req-a") == first
    assert pool.num_active == 1
    assert pool.num_free == 1


def test_distinct_requests_get_distinct_rows():
    pool = DeltaNetStateSlots(num_slots=4)
    rows = [pool.allocate(f"req-{i}") for i in range(4)]
    assert sorted(rows) == [0, 1, 2, 3]
    assert len(set(rows)) == 4


def test_exhaustion_raises_with_the_corrective_action():
    pool = DeltaNetStateSlots(num_slots=1)
    pool.allocate("req-a")
    with pytest.raises(ValueError) as excinfo:
        pool.allocate("req-b")
    message = str(excinfo.value)
    assert "req-b" in message
    assert "num_seqs_buckets" in message


def test_free_releases_the_row_and_is_safe_twice():
    # The model runner can reap a request more than one time.
    pool = DeltaNetStateSlots(num_slots=1)
    pool.allocate("req-a")
    assert pool.free("req-a") == 0
    assert pool.free("req-a") is None
    assert pool.num_free == 1
    assert not pool.contains("req-a")


def test_free_of_an_unknown_request_is_a_no_op():
    pool = DeltaNetStateSlots(num_slots=1)
    assert pool.free("never-seen") is None
    assert pool.num_free == 1


def test_a_released_row_is_reused():
    pool = DeltaNetStateSlots(num_slots=1)
    pool.allocate("req-a")
    pool.free("req-a")
    # The new request starts at position 0, so the per-slot reset in the model
    # zeros the row before the first read.
    assert pool.allocate("req-b") == 0


def test_the_free_queue_is_first_in_first_out():
    pool = DeltaNetStateSlots(num_slots=3)
    pool.slots_for(["x", "y", "z"])
    pool.free("x")
    pool.free("y")
    assert pool.allocate("m") == 0
    assert pool.allocate("n") == 1


def test_slots_for_follows_the_batch_order():
    pool = DeltaNetStateSlots(num_slots=2)
    assert pool.slots_for(["req-a", "req-b"]) == [0, 1]
    # The scheduler can reorder the rows between two steps.
    assert pool.slots_for(["req-b", "req-a"]) == [1, 0]


def test_slots_for_allocates_only_the_new_requests():
    pool = DeltaNetStateSlots(num_slots=3)
    pool.allocate("req-a")
    assert pool.slots_for(["req-a", "req-b"]) == [0, 1]
    assert pool.num_active == 2


def test_reset_releases_every_row():
    pool = DeltaNetStateSlots(num_slots=2)
    pool.slots_for(["req-a", "req-b"])
    pool.reset()
    assert pool.num_free == 2
    assert pool.num_active == 0
    assert pool.allocate("req-c") == 0


def test_padded_rows_get_a_free_row():
    # The decode batch pads up to a compiled bucket. A padded row must not point
    # at the row of a request, or it overwrites the state of that request.
    pool = DeltaNetStateSlots(num_slots=4)
    slots = pool.slots_for_batch(["req-a", "req-b"], padded_size=4)
    assert slots == [0, 1, 2, 2]
    real_rows = set(slots[:2])
    padded_rows = set(slots[2:])
    assert real_rows.isdisjoint(padded_rows)


def test_a_padded_row_does_not_take_the_row():
    # The scratch row stays free, so the next request can take it.
    pool = DeltaNetStateSlots(num_slots=4)
    pool.slots_for_batch(["req-a", "req-b"], padded_size=4)
    assert pool.num_free == 2
    assert pool.allocate("req-c") == 2


def test_a_full_batch_has_no_padded_row():
    pool = DeltaNetStateSlots(num_slots=2)
    assert pool.slots_for_batch(["req-a", "req-b"], padded_size=2) == [0, 1]
    assert pool.num_free == 0


def test_a_free_row_exists_for_every_padded_batch():
    # A free row exists while the batch holds every request that has a row.
    # Padding needs num_reqs < padded_size, and padded_size is at most num_slots,
    # so at least one row is free.
    #
    # A request that is not in the batch breaks that condition. See
    # test_a_request_outside_the_batch_takes_the_padded_row.
    num_slots = 8
    for num_reqs in range(num_slots + 1):
        for padded_size in range(num_reqs, num_slots + 1):
            pool = DeltaNetStateSlots(num_slots=num_slots)
            req_ids = [f"req-{i}" for i in range(num_reqs)]
            slots = pool.slots_for_batch(req_ids, padded_size=padded_size)
            assert len(slots) == padded_size
            assert len(set(slots[:num_reqs])) == num_reqs
            if padded_size > num_reqs:
                assert set(slots[num_reqs:]).isdisjoint(set(slots[:num_reqs]))


def test_a_request_outside_the_batch_takes_the_padded_row():
    # The engine gave a row to 8 requests, a step ran 7 of them, and the batch
    # padded up to the bucket of 8. The 8th row belongs to a request outside the
    # batch, so the padded row finds no free row.
    pool = DeltaNetStateSlots(num_slots=8)
    batch = [f"req-{index}" for index in range(7)]
    for req_id in batch:
        pool.allocate(req_id)
    # This request holds a row, and it is not in the batch of this step.
    pool.allocate("req-incoming")
    assert pool.num_free == 0

    with pytest.raises(ValueError) as excinfo:
        pool.slots_for_batch(batch, padded_size=8)
    message = str(excinfo.value)
    # The message must name the cause and the action.
    assert "Admission must keep" in message
    assert "num_seqs_buckets" in message


def test_the_usable_row_count_keeps_one_row_for_a_ladder_with_a_gap():
    # 7 requests can hold a row of 8. Row 8 stays free for the padded rows of a
    # batch of 7, which pads up to the bucket of 8.
    assert usable_state_rows(8, [1, 2, 4, 8]) == 7


def test_the_usable_row_count_uses_every_row_of_a_full_ladder():
    # A bucket for each count means no batch pads, so no batch needs a free row.
    assert usable_state_rows(4, [1, 2, 3, 4]) == 4
    assert usable_state_rows(1, [1]) == 1


def test_the_usable_row_count_never_reaches_zero():
    # A pool of 1 row must stay usable. A batch of 1 request pads to 1 row, so it
    # has no padded row.
    assert usable_state_rows(1, [2]) == 1
    assert usable_state_rows(1, None) == 1


def test_the_usable_row_count_admits_a_batch_that_pads():
    # Close the loop: the count that `usable_state_rows` gives must let the pool
    # pad a batch up to the largest bucket, with 1 request outside the batch.
    num_slots = 8
    usable = usable_state_rows(num_slots, [1, 2, 4, num_slots])
    pool = DeltaNetStateSlots(num_slots=num_slots)
    admitted = [f"req-{index}" for index in range(usable)]
    for req_id in admitted:
        pool.allocate(req_id)
    slots = pool.slots_for_batch(admitted[:-1], padded_size=num_slots)
    assert len(slots) == num_slots
    held = {pool.get(req_id) for req_id in admitted}
    assert set(slots[usable - 1 :]).isdisjoint(held)


def test_a_batch_larger_than_the_pool_raises():
    pool = DeltaNetStateSlots(num_slots=2)
    with pytest.raises(ValueError) as excinfo:
        pool.slots_for_batch(["req-a"], padded_size=3)
    assert "num_seqs_buckets" in str(excinfo.value)


def test_a_padded_size_below_the_request_count_raises():
    pool = DeltaNetStateSlots(num_slots=4)
    with pytest.raises(ValueError) as excinfo:
        pool.slots_for_batch(["req-a", "req-b"], padded_size=1)
    assert "fewer than" in str(excinfo.value)


def test_slots_for_batch_keeps_the_row_of_a_continuing_request():
    # A request keeps one row while its batch position changes.
    pool = DeltaNetStateSlots(num_slots=4)
    first = pool.slots_for_batch(["req-a", "req-b"], padded_size=4)
    second = pool.slots_for_batch(["req-b", "req-a"], padded_size=4)
    assert first[:2] == [0, 1]
    assert second[:2] == [1, 0]


def test_a_prefill_step_asks_for_one_row():
    pool = DeltaNetStateSlots(num_slots=4)
    pool.allocate("req-a")
    pool.allocate("req-b")
    # The model folds the token block of a prefill into one batch row.
    assert pool.slots_for_batch(["req-b"], padded_size=1) == [1]


def test_a_long_run_does_not_leak_rows():
    # Admit and retire many requests through a pool of one row.
    pool = DeltaNetStateSlots(num_slots=1)
    for i in range(200):
        assert pool.allocate(f"req-{i}") == 0
        pool.free(f"req-{i}")
    assert pool.num_free == 1
    assert pool.num_active == 0


@pytest.mark.parametrize(
    "buckets, rows",
    [([1, 4, 8], 9), ([32], 33), ([1, 2], 2), ([1, 2, 3, 4], 4), (None, 1), ([], 1)],
)
def test_the_pool_adds_one_row_when_a_batch_can_pad(buckets, rows):
    from vllm_neuron.vllm.worker.deltanet_state_slots import state_pool_rows

    assert state_pool_rows(buckets) == rows


@pytest.mark.parametrize("buckets", [[1, 4, 8], [8], [32], [1, 2, 3, 4]])
def test_max_num_seqs_requests_fit_with_a_padded_batch(buckets):
    # Every one of max(buckets) requests holds a row, one waits out the step, and
    # the batch pads up to the largest bucket: the padded rows still find a row.
    from vllm_neuron.vllm.worker.deltanet_state_slots import state_pool_rows

    num_slots = state_pool_rows(buckets)
    max_num_seqs = max(buckets)
    assert usable_state_rows(num_slots, buckets) == max_num_seqs
    pool = DeltaNetStateSlots(num_slots=num_slots)
    admitted = [f"req-{index}" for index in range(max_num_seqs)]
    for req_id in admitted:
        pool.allocate(req_id)
    if max_num_seqs > 1 and num_slots > max_num_seqs:
        slots = pool.slots_for_batch(admitted[:-1], padded_size=max_num_seqs)
        held = {pool.get(req_id) for req_id in admitted}
        assert set(slots[max_num_seqs - 1 :]).isdisjoint(held)
