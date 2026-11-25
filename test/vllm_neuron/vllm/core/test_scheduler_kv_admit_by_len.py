# SPDX-License-Identifier: Apache-2.0
"""Tests for the KV admission by request length of NeuronScheduler.

The default admission assumes max_model_len for each request. Thus it admits
floor(pool / max_model_len) requests, also when each request is short. With
max_model_len 16384 and a pool of 332,800 tokens, a decode bucket of 40 rows
holds only 20 requests.

The rule by request length (VLLM_NEURON_BACKEND_KV_ADMIT_BY_REQUEST_LEN=1)
counts the blocks of each request for its prompt plus its max_tokens. It admits
a request while the sum of these counts fits the pool. A request cannot grow
past this length, so a decode request is never preempted.

CPU-only; no Neuron device.
"""

import logging
import types
from collections import deque
from unittest.mock import Mock

import pytest
import torch

from test.vllm_neuron.vllm.core.conftest import (
    SchedulerMockKVCacheConfig,
    _make_mock_vllm_config,
    create_mock_request,
)

BLOCK = 16
MAX_MODEL_LEN = 1024  # 64 blocks for each group


def _full_attention_group():
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    spec = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
    )
    return types.SimpleNamespace(layer_names=["layer.0"], kv_cache_spec=spec)


def _sliding_window_group():
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    spec = SlidingWindowSpec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        sliding_window=64,
    )
    return types.SimpleNamespace(layer_names=["layer.1"], kv_cache_spec=spec)


def _config(dcp_size=1):
    config = _make_mock_vllm_config()
    config.scheduler_config.max_num_seqs = 64
    config.scheduler_config.max_num_batched_tokens = MAX_MODEL_LEN
    config.scheduler_config.max_num_scheduled_tokens = MAX_MODEL_LEN
    config.model_config.max_model_len = MAX_MODEL_LEN
    config.additional_config = {}
    config.parallel_config = Mock(
        data_parallel_rank=0,
        data_parallel_index=0,
        pipeline_parallel_size=1,
        decode_context_parallel_size=dcp_size,
        prefill_context_parallel_size=1,
    )
    config.observability_config = Mock(
        kv_cache_metrics=False,
        kv_cache_metrics_sample=1.0,
        enable_mfu_metrics=False,
    )
    return config


def _make_scheduler(monkeypatch, enabled, num_blocks=101, groups=None, dcp_size=1):
    """Build a scheduler and then apply the rule to a pool of ``groups``.

    The base scheduler builds a KV cache manager from the groups of its config,
    so the scheduler starts with no groups and with the rule off. The test then
    calls the init step of the rule with the groups that it checks.
    """
    from vllm_neuron.vllm.core.scheduler import NeuronScheduler

    monkeypatch.delenv("VLLM_NEURON_BACKEND_KV_ADMIT_BY_REQUEST_LEN", raising=False)
    config = _config(dcp_size)
    scheduler = NeuronScheduler(
        vllm_config=config,
        kv_cache_config=SchedulerMockKVCacheConfig(
            num_blocks=num_blocks, block_size=BLOCK
        ),
        structured_output_manager=Mock(),
        block_size=BLOCK,
    )
    if enabled:
        pool = SchedulerMockKVCacheConfig(
            num_blocks=num_blocks,
            block_size=BLOCK,
            kv_cache_groups=[_full_attention_group()] if groups is None else groups,
        )
        scheduler._init_kv_admission_by_len(config, pool)
    scheduler.requests = {}
    scheduler.waiting = deque()
    scheduler.running = []
    return scheduler


def _decode(request_id, prompt=100, max_tokens=28):
    """A running decode request. 100 + 28 tokens take 8 blocks of 16."""
    return create_mock_request(
        request_id,
        num_prompt_tokens=prompt,
        num_computed_tokens=prompt,
        max_tokens=max_tokens,
    )


def _new(request_id, prompt=100, max_tokens=28):
    return create_mock_request(
        request_id, num_prompt_tokens=prompt, max_tokens=max_tokens
    )


class TestTheRuleIsOptIn:
    def test_off_by_default(self, monkeypatch):
        scheduler = _make_scheduler(monkeypatch, enabled=False)
        assert scheduler._kv_admit_by_len is False

    @pytest.mark.parametrize("value,calls", [(None, 0), ("0", 0), ("1", 1)])
    def test_the_env_variable_starts_the_init_step(self, monkeypatch, value, calls):
        from vllm_neuron.vllm.core.scheduler import NeuronScheduler

        seen = []
        monkeypatch.setattr(
            NeuronScheduler,
            "_init_kv_admission_by_len",
            lambda self, config, pool: seen.append(pool),
        )
        if value is None:
            monkeypatch.delenv(
                "VLLM_NEURON_BACKEND_KV_ADMIT_BY_REQUEST_LEN", raising=False
            )
        else:
            monkeypatch.setenv("VLLM_NEURON_BACKEND_KV_ADMIT_BY_REQUEST_LEN", value)
        pool = SchedulerMockKVCacheConfig(num_blocks=101, block_size=BLOCK)
        scheduler = NeuronScheduler(
            vllm_config=_config(),
            kv_cache_config=pool,
            structured_output_manager=Mock(),
            block_size=BLOCK,
        )
        assert seen == [pool] * calls
        assert scheduler._kv_admit_by_len is False

    def test_default_rule_admits_one_request_per_max_model_len(self, monkeypatch):
        # 101 blocks x 16 = 1616 tokens, so 1 request of max_model_len 1024.
        scheduler = _make_scheduler(monkeypatch, enabled=False)
        assert scheduler._max_kv_concurrent == 1
        scheduler.running.append(_decode("r0"))
        assert scheduler._kv_pool_is_full_for(_new("r1")) is True
        assert scheduler.can_schedule(_new("r1")) is False

    def test_on_for_a_full_attention_pool(self, monkeypatch):
        scheduler = _make_scheduler(monkeypatch, enabled=True)
        assert scheduler._kv_admit_by_len is True
        # One block of the pool is the null block of vLLM.
        assert scheduler._kv_budget_blocks == 100
        assert scheduler._kv_group_block_sizes == (BLOCK,)

    def test_off_for_a_sliding_window_group(self, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING):
            scheduler = _make_scheduler(
                monkeypatch,
                enabled=True,
                groups=[_full_attention_group(), _sliding_window_group()],
            )
        assert scheduler._kv_admit_by_len is False
        assert any("not available" in r.message for r in caplog.records)

    def test_off_for_an_empty_pool(self, monkeypatch):
        scheduler = _make_scheduler(monkeypatch, enabled=True, groups=[])
        assert scheduler._kv_admit_by_len is False

    def test_off_with_decode_context_parallel(self, monkeypatch):
        scheduler = _make_scheduler(monkeypatch, enabled=True, dcp_size=2)
        assert scheduler._kv_admit_by_len is False


class TestTheBlockCount:
    def test_prompt_plus_max_tokens(self, monkeypatch):
        scheduler = _make_scheduler(monkeypatch, enabled=True)
        assert scheduler._kv_blocks_of(_new("r", prompt=100, max_tokens=28)) == 8
        assert scheduler._kv_blocks_of(_new("r", prompt=100, max_tokens=29)) == 9

    def test_capped_at_max_model_len(self, monkeypatch):
        scheduler = _make_scheduler(monkeypatch, enabled=True)
        assert scheduler._kv_blocks_of(_new("r", prompt=1000, max_tokens=500)) == 64

    def test_no_max_tokens_means_max_model_len(self, monkeypatch):
        scheduler = _make_scheduler(monkeypatch, enabled=True)
        assert scheduler._kv_blocks_of(_new("r", prompt=100, max_tokens=None)) == 64

    def test_each_group_counts(self, monkeypatch):
        scheduler = _make_scheduler(
            monkeypatch,
            enabled=True,
            groups=[_full_attention_group(), _full_attention_group()],
        )
        assert scheduler._kv_blocks_of(_new("r", prompt=100, max_tokens=28)) == 16

    def test_lookahead_tokens_count(self, monkeypatch):
        scheduler = _make_scheduler(monkeypatch, enabled=True)
        scheduler.num_lookahead_tokens = 4
        assert scheduler._kv_blocks_of(_new("r", prompt=100, max_tokens=28)) == 9

    def test_send_lane_producer_counts(self, monkeypatch):
        scheduler = _make_scheduler(monkeypatch, enabled=True)
        scheduler.running.append(_decode("r0"))
        parked = _decode("p0", prompt=200, max_tokens=1)  # 201 tokens: 13 blocks
        scheduler.requests["p0"] = parked
        scheduler._pending_kv_send = {"p0", "gone"}
        assert scheduler._kv_blocks_in_use() == 8 + 13


class TestTheAdmission:
    def test_short_requests_fill_the_pool(self, monkeypatch):
        # 12 requests of 8 blocks take 96 of the 100 blocks. A 13th needs 104.
        scheduler = _make_scheduler(monkeypatch, enabled=True)
        for index in range(12):
            request = _new(f"r{index}")
            assert scheduler.can_schedule(request) is True
            scheduler.running.append(_decode(f"r{index}"))
        assert scheduler._kv_blocks_in_use() == 96
        assert scheduler.can_schedule(_new("r12")) is False

    def test_a_request_that_fits_the_rest_is_admitted(self, monkeypatch):
        scheduler = _make_scheduler(monkeypatch, enabled=True)
        for index in range(12):
            scheduler.running.append(_decode(f"r{index}"))
        # 4 blocks are free: 64 tokens fit.
        assert scheduler.can_schedule(_new("small", prompt=40, max_tokens=24)) is True
        assert scheduler.can_schedule(_new("big", prompt=40, max_tokens=25)) is False

    def test_a_long_request_holds_a_full_share(self, monkeypatch):
        scheduler = _make_scheduler(monkeypatch, enabled=True)
        scheduler.running.append(_decode("long", prompt=1000, max_tokens=24))
        assert scheduler._kv_blocks_in_use() == 64
        assert scheduler.can_schedule(_new("r1", prompt=500, max_tokens=76)) is True
        assert scheduler.can_schedule(_new("r2", prompt=500, max_tokens=77)) is False

    def test_the_warning_names_the_blocks(self, monkeypatch, caplog):
        scheduler = _make_scheduler(monkeypatch, enabled=True)
        for index in range(12):
            scheduler.running.append(_decode(f"r{index}"))
        with caplog.at_level(logging.WARNING):
            assert scheduler.can_schedule(_new("r12")) is False
            assert scheduler.can_schedule(_new("r13")) is False
        warnings = [
            r.message for r in caplog.records if "Prefill deferred" in r.message
        ]
        assert warnings == [
            "Prefill deferred: the KV pool cannot hold the largest length of 12 "
            "running requests and the new request (96 + 8 of 100 blocks). Waiting "
            "for decode requests to complete. To increase concurrency, raise "
            "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION or send a smaller max_tokens."
        ]


class TestTheRoomForPrefill:
    def test_default_rule(self, monkeypatch):
        scheduler = _make_scheduler(monkeypatch, enabled=False)
        assert scheduler._kv_pool_has_room_for_prefill() is True
        scheduler.running.append(_decode("r0"))
        assert scheduler._kv_pool_has_room_for_prefill() is False

    def test_counts_the_waiting_request(self, monkeypatch):
        scheduler = _make_scheduler(monkeypatch, enabled=True)
        for index in range(11):
            scheduler.running.append(_decode(f"r{index}"))
        scheduler.waiting.append(_new("w0"))
        assert scheduler._kv_pool_has_room_for_prefill() is True
        scheduler.waiting.append(_new("w1", max_tokens=29))
        assert scheduler._kv_pool_has_room_for_prefill() is False

    def test_an_empty_waiting_queue_assumes_max_model_len(self, monkeypatch):
        # A structured-output request from the skipped-waiting queue has no
        # known length here.
        scheduler = _make_scheduler(monkeypatch, enabled=True)
        for index in range(4):
            scheduler.running.append(_decode(f"r{index}"))
        assert scheduler._kv_pool_has_room_for_prefill() is True
        scheduler.running.append(_decode("r4"))
        assert scheduler._kv_pool_has_room_for_prefill() is False


@pytest.mark.parametrize("limit,full", [(1, True), (2, False)])
def test_a_scheduler_without_the_attribute_keeps_the_default_rule(limit, full):
    # Some tests build a scheduler without __init__. The rule must then stay off.
    from vllm_neuron.vllm.core.scheduler import NeuronScheduler

    scheduler = NeuronScheduler.__new__(NeuronScheduler)
    scheduler.running = [_decode("r0")]
    scheduler._pending_kv_send = set()
    scheduler._max_kv_concurrent = limit
    assert scheduler._kv_pool_is_full_for(_new("r1")) is full
