# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the inputs of an async spec decode step after a non-spec pause.

Near ``max_model_len`` the scheduler runs a batch without drafts for a few
steps, then drafts resume for the requests that are still running. The first
input token of each spec row must be the token the request emitted last, both
when the row comes from the transition cache (batch composition changed) and
when the runner rebuilds the input on the host. Run with
``VLLM_NEURON_CPU_MODE=1``; no device is needed.
"""

from types import SimpleNamespace

import numpy as np
import torch

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

K = 3


def _runner(req_ids: list[str]) -> NeuronModelRunner:
    runner = object.__new__(NeuronModelRunner)
    runner.use_async_scheduling = True
    runner.is_eagle3_spec = True
    runner.is_dflash_spec = False
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=K)
    runner.vocab_size = 1000
    runner._async_spec_transition_full_draft_cache = {}
    runner._async_spec_transition_bonus_source = {}
    runner.async_execution_buffer = {}
    runner._draft_token_ids = None
    runner._async_partial_prefill_req_ids = set()
    runner._async_output_materialized_this_step = True
    runner._batch_composition_changed = False
    runner.requests = {
        req_id: SimpleNamespace(num_prompt_tokens=50) for req_id in req_ids
    }
    runner.input_batch = SimpleNamespace(  # type: ignore[assignment]
        req_ids=list(req_ids),
        num_reqs=len(req_ids),
        num_computed_tokens_cpu=np.full(len(req_ids), 100, dtype=np.int32),
    )
    return runner


def _row(*tokens: int) -> torch.Tensor:
    return torch.tensor(tokens, dtype=torch.int32)


def _fill(runner: NeuronModelRunner, num_reqs: int = 1) -> list[int]:
    """Inputs of a step that resumes drafts after a batch-composition change."""
    runner._batch_composition_changed = True
    metadata = SimpleNamespace(
        num_draft_tokens=[K] * num_reqs,
        draft_token_ids=torch.zeros(K * num_reqs, dtype=torch.int32),
    )
    input_ids = runner._fill_transition_input_and_spec_metadata_from_cache(
        torch.full(((1 + K) * num_reqs,), -1, dtype=torch.int32), metadata
    )
    tokens: list[int] = input_ids.tolist()
    return tokens


def test_no_proposal_step_refreshes_cached_bonus():
    runner = _runner(["b"])
    cached = _row(10, 11, 12, 13)
    runner._async_spec_transition_full_draft_cache["b"] = cached
    # A non-spec decode step: one token per row, padded to the bucket.
    runner.async_execution_buffer["futures_sampled_token_ids"] = _row(42, 0)

    runner._cache_async_spec_transition_drafts(["b"])

    # The step itself leaves the cached row alone; the fill applies the token.
    assert runner._async_spec_transition_full_draft_cache["b"] is cached
    assert _fill(runner) == [42, 11, 12, 13]


def test_verify_step_without_proposal_uses_last_accepted_token():
    runner = _runner(["b"])
    runner._async_spec_transition_full_draft_cache["b"] = _row(10, 11, 12, 13)
    # A verify step whose proposal was skipped.
    runner.async_execution_buffer["futures_sampled_token_ids"] = torch.zeros(
        (1, 1 + K), dtype=torch.int32
    )
    runner.async_execution_buffer["futures_last_accepted_token"] = _row(77)

    runner._cache_async_spec_transition_drafts(["b"])

    assert _fill(runner) == [77, 11, 12, 13]


def test_composition_change_fill_uses_latest_token():
    runner = _runner(["b"])
    runner._async_spec_transition_full_draft_cache["b"] = _row(10, 11, 12, 13)
    runner.async_execution_buffer["futures_sampled_token_ids"] = _row(42, 0)
    runner._cache_async_spec_transition_drafts(["b"])
    # A later step that emits without proposing replaces the recorded token.
    runner.async_execution_buffer["futures_sampled_token_ids"] = _row(43, 0)
    runner._cache_async_spec_transition_drafts(["b"])

    # Drafts resume on a step whose batch composition changed.
    runner._batch_composition_changed = True
    metadata = SimpleNamespace(
        num_draft_tokens=[K], draft_token_ids=torch.zeros(K, dtype=torch.int32)
    )
    input_ids = runner._fill_transition_input_and_spec_metadata_from_cache(
        torch.full((1 + K,), -1, dtype=torch.int32), metadata
    )

    assert input_ids.tolist() == [43, 11, 12, 13]
    assert metadata.draft_token_ids.tolist() == [11, 12, 13]


def test_proposal_step_replaces_rows_unchanged():
    runner = _runner(["a", "b"])
    runner._async_spec_transition_full_draft_cache["b"] = _row(10, 11, 12, 13)
    # A no-proposal step first; the next proposal supersedes its token.
    runner.async_execution_buffer["futures_sampled_token_ids"] = _row(42, 0)
    runner._cache_async_spec_transition_drafts(["b"])
    runner.async_execution_buffer["futures_sampled_token_ids"] = torch.zeros(
        (2, 1 + K), dtype=torch.int32
    )
    runner._draft_token_ids = torch.tensor(
        [[5, 6, 7, 8], [9, 20, 21, 22]], dtype=torch.int32
    )

    runner._cache_async_spec_transition_drafts(["a", "b"])

    cache = runner._async_spec_transition_full_draft_cache
    assert cache["a"].tolist() == [5, 6, 7, 8]
    assert cache["b"].tolist() == [9, 20, 21, 22]
    assert _fill(runner, 2) == [5, 6, 7, 8, 9, 20, 21, 22]


def test_partial_prefill_row_not_refreshed():
    runner = _runner(["a", "b"])
    runner._async_spec_transition_full_draft_cache["a"] = _row(10, 11, 12, 13)
    runner._async_spec_transition_full_draft_cache["b"] = _row(30, 31, 32, 33)
    runner._async_partial_prefill_req_ids = {"a"}
    runner.async_execution_buffer["futures_sampled_token_ids"] = _row(42, 43)

    runner._cache_async_spec_transition_drafts(["a", "b"])

    assert _fill(runner, 2) == [10, 11, 12, 13, 43, 31, 32, 33]


def _set_prev_output(runner: NeuronModelRunner, sampled, req_ids) -> None:
    runner.async_execution_buffer["async_output"] = SimpleNamespace(
        model_runner_output=SimpleNamespace(sampled_token_ids=sampled, req_ids=req_ids)
    )


def test_fallback_fixup_sets_first_token_of_spec_rows():
    runner = _runner(["a", "b"])
    _set_prev_output(runner, [[5, 42], [7]], ["a", "b"])
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens_padded={"a": 1 + K, "b": 1 + K},
        scheduled_spec_decode_tokens={"a": [-1] * K, "b": [-1] * K},
    )
    # The host gather on a rank that did not write the last tokens back.
    input_ids = _row(1, -1, -1, -1, 2, -1, -1, -1)

    out = runner._apply_async_fallback_token_fixup(input_ids, scheduler_output)

    assert out.tolist() == [42, -1, -1, -1, 7, -1, -1, -1]


def test_fallback_fixup_one_token_rows_unchanged():
    runner = _runner(["a", "b"])
    _set_prev_output(runner, [[5, 42], [7]], ["a", "b"])
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens_padded={"a": 1, "b": 1},
        scheduled_spec_decode_tokens={},
    )

    out = runner._apply_async_fallback_token_fixup(_row(1, 2), scheduler_output)

    assert out.tolist() == [42, 7]


def test_fallback_fixup_skips_prefill_bucket():
    runner = _runner(["a"])
    _set_prev_output(runner, [[5]], ["a"])
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens_padded={"a": 128},
        scheduled_spec_decode_tokens={},
    )
    input_ids = torch.tensor(list(range(128)), dtype=torch.int32)

    out = runner._apply_async_fallback_token_fixup(input_ids.clone(), scheduler_output)

    assert torch.equal(out, input_ids)
