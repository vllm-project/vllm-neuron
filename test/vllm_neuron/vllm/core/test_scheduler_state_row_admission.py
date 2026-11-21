# SPDX-License-Identifier: Apache-2.0
"""Tests for the recurrent state row admission control of NeuronScheduler.

A model with linear attention keeps one recurrent state row for each request. The
row count limits how many requests the scheduler admits.

The row of a producer is the reason this check exists.
`_release_dispatched_producers` takes a dispatched producer out of the running
queue, so the compute slot is free for the next prefill. The consumer reads the
state of that producer after it leaves the queue, so the producer still holds its
state row. Without this check the scheduler admits the next request, the worker
finds no free row, and the request fails:

    ValueError: The DeltaNet state pool holds no free slot for request <id>.
    All 1 slots are in use.

A device run of a 1P3D topology showed this: 10 of 12 requests failed, and only
the prefill engine reported the error.

CPU-only; no Neuron device.
"""

import types
from collections import deque
from unittest.mock import Mock

import pytest

# Qwen3-Next interleaves one full attention layer after every 3 linear layers.
_HYBRID_LAYERS = ["linear_attention"] * 3 + ["full_attention"]


def _make_hybrid(model_config):
    """Make an existing model config report linear attention layers.

    Change the config in place. The base scheduler reads other fields of it, so a
    replacement object breaks the base __init__.
    """
    model_config.hf_config = types.SimpleNamespace(
        text_config=types.SimpleNamespace(layer_types=_HYBRID_LAYERS * 16)
    )
    return model_config


def _make_scheduler(base_config, num_seqs_buckets, max_num_seqs, kv_cache_config, som):
    """Build a scheduler for a hybrid model with a given state row count."""
    from vllm_neuron.vllm.core.scheduler import NeuronAsyncScheduler

    _make_hybrid(base_config.model_config)
    base_config.scheduler_config.max_num_seqs = max_num_seqs
    base_config.additional_config = {
        "neuron_config": {"num_seqs_buckets": num_seqs_buckets}
    }
    scheduler = NeuronAsyncScheduler(
        vllm_config=base_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=som,
        block_size=32,
        log_stats=False,
    )
    scheduler.requests = {}
    scheduler.waiting = deque()
    scheduler.running = []
    return scheduler


@pytest.fixture
def hybrid_scheduler(
    mock_vllm_config, mock_kv_cache_config, mock_structured_output_manager
):
    """A prefill engine of a hybrid model: 1 state row."""
    return _make_scheduler(
        mock_vllm_config, [1], 1, mock_kv_cache_config, mock_structured_output_manager
    )


@pytest.fixture
def hybrid_scheduler_two_rows(
    mock_vllm_config, mock_kv_cache_config, mock_structured_output_manager
):
    """A decode engine of a hybrid model: 2 state rows."""
    return _make_scheduler(
        mock_vllm_config,
        [1, 2],
        2,
        mock_kv_cache_config,
        mock_structured_output_manager,
    )


@pytest.fixture
def hybrid_scheduler_wide(
    mock_vllm_config, mock_kv_cache_config, mock_structured_output_manager
):
    """A wide decode engine: 8 state rows, and a ladder that has gaps."""
    return _make_scheduler(
        mock_vllm_config,
        [1, 2, 4, 8],
        8,
        mock_kv_cache_config,
        mock_structured_output_manager,
    )


@pytest.fixture
def hybrid_scheduler_full_ladder(
    mock_vllm_config, mock_kv_cache_config, mock_structured_output_manager
):
    """A decode engine of 4 rows with a bucket for each count."""
    return _make_scheduler(
        mock_vllm_config,
        [1, 2, 3, 4],
        4,
        mock_kv_cache_config,
        mock_structured_output_manager,
    )


class TestTheRowCount:
    """The scheduler must read the row count from the bucket list."""

    def test_a_hybrid_model_reads_the_largest_bucket(self, hybrid_scheduler_two_rows):
        assert hybrid_scheduler_two_rows._num_state_rows == 2

    def test_a_prefill_engine_holds_one_row(self, hybrid_scheduler):
        assert hybrid_scheduler._num_state_rows == 1

    def test_a_dense_model_holds_no_row(self, neuron_scheduler):
        # The check must stay off for a model that keeps no recurrent state, so a
        # dense deployment keeps its admission behaviour.
        assert neuron_scheduler._num_state_rows == 0
        assert neuron_scheduler._state_rows_in_use() == 0


class TestThePaddedRowKeepsOneRowFree:
    """A decode batch that pads needs 1 free row, so admission must keep 1 row.

    A device run of a wide decode engine showed this. The client sent 14 requests
    to an engine of 8 rows. The engine admitted 8 requests, a step ran 7 of them,
    and the batch padded up to the bucket of 8. The padded row found no free row,
    and every worker of the engine stopped:

        ValueError: The padded batch needs 1 padded rows, but the DeltaNet state
        pool holds no free row.

    The request count of the batch does not give that row. A request that waits for
    its remote state read holds a row before it runs, so the pool is full while the
    batch is not.
    """

    def test_a_ladder_with_a_gap_keeps_one_row_free(self, hybrid_scheduler_wide):
        assert hybrid_scheduler_wide._num_state_rows == 8
        assert hybrid_scheduler_wide._usable_state_rows == 7

    def test_a_ladder_with_every_count_uses_every_row(
        self, hybrid_scheduler_full_ladder
    ):
        # No batch pads, so no batch needs a free row.
        assert hybrid_scheduler_full_ladder._num_state_rows == 4
        assert hybrid_scheduler_full_ladder._usable_state_rows == 4

    def test_the_engine_admits_the_usable_rows(
        self, hybrid_scheduler_wide, mock_request_factory
    ):
        from vllm_neuron.vllm.core.scheduler import SchedulerState

        hybrid_scheduler_wide._state = SchedulerState.ACTIVE_DECODE
        for index in range(6):
            hybrid_scheduler_wide.running.append(
                mock_request_factory(
                    request_id=f"req-{index}",
                    num_prompt_tokens=100,
                    num_computed_tokens=100,
                )
            )
        assert hybrid_scheduler_wide._state_rows_in_use() == 6
        request = mock_request_factory(request_id="req-6", num_prompt_tokens=100)
        assert hybrid_scheduler_wide.can_schedule(request) is True

    def test_the_engine_defers_the_request_that_takes_the_last_row(
        self, hybrid_scheduler_wide, mock_request_factory
    ):
        from vllm_neuron.vllm.core.scheduler import SchedulerState

        hybrid_scheduler_wide._state = SchedulerState.ACTIVE_DECODE
        for index in range(7):
            hybrid_scheduler_wide.running.append(
                mock_request_factory(
                    request_id=f"req-{index}",
                    num_prompt_tokens=100,
                    num_computed_tokens=100,
                )
            )
        # 7 rows of 8 are in use. Row 8 must stay free for the padded rows of a
        # batch of 7, which pads up to the bucket of 8.
        assert hybrid_scheduler_wide._state_rows_in_use() == 7
        request = mock_request_factory(request_id="req-7", num_prompt_tokens=100)
        assert hybrid_scheduler_wide.can_schedule(request) is False

    def test_the_admitted_requests_can_fill_a_padded_batch(self, hybrid_scheduler_wide):
        # Close the loop against the pool: the largest count that admission allows
        # must leave the pool able to pad a batch up to the largest bucket.
        from vllm_neuron.vllm.worker.deltanet_state_slots import DeltaNetStateSlots

        pool = DeltaNetStateSlots(num_slots=hybrid_scheduler_wide._num_state_rows)
        admitted = [
            f"req-{index}" for index in range(hybrid_scheduler_wide._usable_state_rows)
        ]
        # Every admitted request holds a row. One of them waits for its state read,
        # so the batch runs the other rows and pads up to the bucket of 8.
        for req_id in admitted:
            pool.allocate(req_id)
        slots = pool.slots_for_batch(admitted[:-1], padded_size=8)
        assert len(slots) == 8
        # The padded rows must not point at a row that a request holds.
        held = {pool.get(req_id) for req_id in admitted}
        assert set(slots[len(admitted) - 1 :]).isdisjoint(held)


class TestTheSendLaneHoldsARow:
    """A producer that waits for a remote read still holds its row."""

    def test_a_parked_producer_counts(self, hybrid_scheduler):
        hybrid_scheduler._pending_kv_send.add("req-a")
        assert hybrid_scheduler._state_rows_in_use() == 1

    def test_a_parked_producer_stops_admission(
        self, hybrid_scheduler, mock_request_factory
    ):
        # This is the defect. The running queue is empty, because
        # _release_dispatched_producers moved the producer to the send-lane, so
        # every other check passes. Only the row count can defer this request.
        hybrid_scheduler._pending_kv_send.add("req-a")
        assert hybrid_scheduler.running == []
        request = mock_request_factory(request_id="req-b", num_prompt_tokens=100)
        assert hybrid_scheduler.can_schedule(request) is False

    def test_the_request_is_admitted_after_the_read(
        self, hybrid_scheduler, mock_request_factory
    ):
        # The row returns when the consumer finishes the read, and the next
        # request is then admitted. So the deferral is not a deadlock.
        hybrid_scheduler._pending_kv_send.add("req-a")
        request = mock_request_factory(request_id="req-b", num_prompt_tokens=100)
        assert hybrid_scheduler.can_schedule(request) is False
        hybrid_scheduler._pending_kv_send.discard("req-a")
        assert hybrid_scheduler.can_schedule(request) is True


class TestTheWaitingQueueHoldsARow:
    """A decode engine reads the remote state into the row before the request runs."""

    def test_a_waiting_request_counts(self, hybrid_scheduler_two_rows):
        from vllm_neuron.vllm.core.scheduler import SchedulerState

        hybrid_scheduler_two_rows._state = SchedulerState.ACTIVE_DECODE
        hybrid_scheduler_two_rows.waiting.append(Mock(request_id="req-a"))
        assert hybrid_scheduler_two_rows._state_rows_in_use() == 1

    def test_two_rows_admit_one_running_and_one_landing(
        self, hybrid_scheduler_two_rows, mock_request_factory
    ):
        running = mock_request_factory(
            request_id="req-a", num_prompt_tokens=100, num_computed_tokens=100
        )
        hybrid_scheduler_two_rows.running.append(running)
        # 1 row in use of 2, so the state that lands has a row.
        assert hybrid_scheduler_two_rows._state_rows_in_use() == 1
        request = mock_request_factory(request_id="req-b", num_prompt_tokens=100)
        assert hybrid_scheduler_two_rows.can_schedule(request) is True

    def test_a_third_request_is_deferred(
        self, hybrid_scheduler_two_rows, mock_request_factory
    ):
        from vllm_neuron.vllm.core.scheduler import SchedulerState

        hybrid_scheduler_two_rows._state = SchedulerState.ACTIVE_DECODE
        running = mock_request_factory(
            request_id="req-a", num_prompt_tokens=100, num_computed_tokens=100
        )
        hybrid_scheduler_two_rows.running.append(running)
        hybrid_scheduler_two_rows.waiting.append(Mock(request_id="req-b"))
        assert hybrid_scheduler_two_rows._state_rows_in_use() == 2
        request = mock_request_factory(request_id="req-c", num_prompt_tokens=100)
        assert hybrid_scheduler_two_rows.can_schedule(request) is False


class TestTheWarning:
    """The operator must learn the reason and the action."""

    def test_the_warning_names_the_action(
        self, hybrid_scheduler, mock_request_factory, caplog
    ):
        import logging

        hybrid_scheduler._pending_kv_send.add("req-a")
        request = mock_request_factory(request_id="req-b", num_prompt_tokens=100)
        with caplog.at_level(logging.WARNING):
            hybrid_scheduler.can_schedule(request)
        message = caplog.text
        assert "recurrent state" in message
        assert "num_seqs_buckets" in message
        assert "max-num-seqs" in message

    def test_the_warning_comes_one_time(
        self, hybrid_scheduler, mock_request_factory, caplog
    ):
        import logging

        hybrid_scheduler._pending_kv_send.add("req-a")
        request = mock_request_factory(request_id="req-b", num_prompt_tokens=100)
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                hybrid_scheduler.can_schedule(request)
        assert caplog.text.count("Prefill deferred: 1 requests hold the 1 usable") == 1
