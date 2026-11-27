# SPDX-License-Identifier: Apache-2.0
"""CPU tests for async speculative decoding near ``max_model_len``.

``NeuronAsyncScheduler`` stops giving draft placeholders to a request whose
next spec step would not fit within ``max_model_len``. These tests check that a
decode step never mixes 0-draft and k-draft requests, since the verify graph
runs ``1 + k`` rows for every request. Run with ``VLLM_NEURON_CPU_MODE=1``; no
device is needed.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_neuron.vllm.core.scheduler import NeuronAsyncScheduler

MAX_MODEL_LEN = 64


def _fake_base_schedule(self, throttle_prefills=False):
    """Schedule every running decode request the way vLLM's schedule() does:
    one token plus its draft placeholders, which move into
    ``scheduled_spec_decode_tokens``."""
    num_scheduled_tokens = {}
    scheduled_spec_decode_tokens = {}
    for request in self.running:
        num_scheduled_tokens[request.request_id] = 1 + len(request.spec_token_ids)
        if request.spec_token_ids:
            scheduled_spec_decode_tokens[request.request_id] = list(
                request.spec_token_ids
            )
            request.spec_token_ids = []
    output = SimpleNamespace(
        num_scheduled_tokens=num_scheduled_tokens,
        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
        pending_structured_output_tokens=False,
    )
    self._update_after_schedule(output)
    return output


def _fake_scheduler_update_after_schedule(self, scheduler_output):
    for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
        self.requests[req_id].num_computed_tokens += num_tokens


@pytest.fixture(autouse=True)
def _fake_vllm_scheduler():
    with (
        patch.object(AsyncScheduler, "schedule", _fake_base_schedule),
        patch.object(
            Scheduler, "_update_after_schedule", _fake_scheduler_update_after_schedule
        ),
    ):
        yield


def _scheduler(num_spec_tokens: int) -> NeuronAsyncScheduler:
    sched = object.__new__(NeuronAsyncScheduler)
    sched.num_spec_tokens = num_spec_tokens
    sched.max_model_len = MAX_MODEL_LEN
    sched._spec_token_placeholders = [-1] * num_spec_tokens
    sched.requests = {}
    sched.running = []
    return sched


def _add_request(
    sched: NeuronAsyncScheduler, req_id: str, num_computed_tokens: int
) -> SimpleNamespace:
    """Add a running decode request that has draft placeholders, as
    ``_update_after_schedule`` leaves one after a spec step."""
    request = SimpleNamespace(
        request_id=req_id,
        num_computed_tokens=num_computed_tokens,
        is_prefill_chunk=False,
        use_structured_output=False,
        num_output_placeholders=0,
        spec_token_ids=sched._spec_token_placeholders,
    )
    sched.requests[req_id] = request
    sched.running.append(request)
    return request


def _draft_counts(output) -> dict[str, int]:
    return {
        req_id: len(output.scheduled_spec_decode_tokens.get(req_id, ()))
        for req_id in output.num_scheduled_tokens
    }


def _safe_num_computed(k: int) -> int:
    # The scheduler's threshold: past it, a request's drafts are disabled.
    return MAX_MODEL_LEN - 3 - 2 * k


@pytest.mark.parametrize("k", [1, 2, 3, 7])
def test_decode_step_never_mixes_draft_counts(k):
    sched = _scheduler(k)
    # A crosses the threshold on the first step; B is far from the limit.
    _add_request(sched, "a", _safe_num_computed(k) - k)
    _add_request(sched, "b", 10)

    for step in range(4):
        counts = _draft_counts(sched._call_base_schedule())
        assert set(counts) == {"a", "b"}
        assert len(set(counts.values())) == 1, (
            f"step {step} mixes draft counts {counts}"
        )
    # The shared placeholder list must never be cleared in place.
    assert sched._spec_token_placeholders == [-1] * k


def test_request_joining_during_pause_gets_no_drafts():
    k = 3
    sched = _scheduler(k)
    a = _add_request(sched, "a", _safe_num_computed(k) + 1)
    a._async_spec_disabled = True
    a.spec_token_ids = []
    # C just finished its prefill and has placeholders for its first decode.
    _add_request(sched, "c", 20)

    output = sched._call_base_schedule()

    assert _draft_counts(output) == {"a": 0, "c": 0}
    assert output.num_scheduled_tokens == {"a": 1, "c": 1}


def test_drafts_resume_after_disabled_request_leaves():
    k = 3
    sched = _scheduler(k)
    a = _add_request(sched, "a", _safe_num_computed(k) - k)
    b = _add_request(sched, "b", 10)
    sched._call_base_schedule()
    sched._call_base_schedule()
    assert a._async_spec_disabled

    sched.running.remove(a)
    del sched.requests["a"]
    output = sched._call_base_schedule()

    assert _draft_counts(output) == {"b": k}
    assert not getattr(b, "_async_spec_disabled", False)


def test_far_from_limit_batch_unchanged():
    k = 3
    steps = 4
    sched = _scheduler(k)
    a = _add_request(sched, "a", 10)
    b = _add_request(sched, "b", 12)

    for _ in range(steps):
        output = sched._call_base_schedule()
        assert _draft_counts(output) == {"a": k, "b": k}

    assert a.num_output_placeholders == steps * (1 + k)
    assert b.num_output_placeholders == steps * (1 + k)


@pytest.mark.parametrize("k", [1, 2, 3, 7])
def test_single_request_near_limit_unchanged(k):
    sched = _scheduler(k)
    _add_request(sched, "a", _safe_num_computed(k) - 2 * (1 + k))

    counts = [_draft_counts(sched._call_base_schedule())["a"] for _ in range(5)]

    # Spec steps until the request crosses the threshold, then non-spec.
    assert counts == [k, k, k, 0, 0]


def test_spec_off_unchanged():
    sched = _scheduler(0)
    a = _add_request(sched, "a", MAX_MODEL_LEN - 6)
    b = _add_request(sched, "b", 10)

    for step in range(1, 4):
        output = sched._call_base_schedule()
        assert output.scheduled_spec_decode_tokens == {}
        assert output.num_scheduled_tokens == {"a": 1, "b": 1}
        assert a.num_output_placeholders == step
        assert b.num_output_placeholders == step
