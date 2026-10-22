# SPDX-License-Identifier: Apache-2.0
"""Logprobs of on-device sampled tokens (NeuronModelRunner._sampled_token_logprobs, CPU)."""

import types

import pytest
import torch

runner_mod = pytest.importorskip("vllm_neuron.vllm.worker.neuron_model_runner")
from vllm.v1.sample.sampler import Sampler  # noqa: E402

V = 11


def _runner(logprobs_mode="raw_logprobs"):
    runner = object.__new__(runner_mod.NeuronModelRunner)
    runner.input_batch = types.SimpleNamespace(vocab_size=V)
    runner.sampler = Sampler(logprobs_mode=logprobs_mode)
    return runner


def _logits(rows):
    torch.manual_seed(0)
    return torch.randn(rows, V)


def test_verify_step_keeps_accepted_tokens_in_order():
    # 2 requests, k = 2 drafts (3 rows each). Request 0 accepted 1 draft + bonus; request 1
    # accepted everything. A third, padded row must be ignored.
    sampled = torch.tensor([[3, 5, -1], [1, 2, 4], [7, 7, 7]])
    logits = _logits(9)
    emitted = [[3, 5], [1, 2, 4]]
    lp = _runner()._sampled_token_logprobs(sampled, logits, 2, emitted)
    assert lp.cu_num_generated_tokens == [0, 2, 5]
    ref = logits.log_softmax(-1)
    rows, tokens = [0, 1, 3, 4, 5], [3, 5, 1, 2, 4]
    # Column 0 holds the sampled token and its logprob, then the top-2.
    assert lp.logprob_token_ids[:, 0].tolist() == tokens
    assert torch.allclose(lp.logprobs[:, 0], ref[rows, tokens])
    assert torch.equal(lp.logprob_token_ids[:, 1:], ref[rows].topk(2, -1).indices)
    lists = lp.tolists()
    assert len(lists.logprob_token_ids) == 5


def test_plain_decode_and_discarded_rows():
    sampled = torch.tensor([6, 2, 9])  # [bs] on-device samples
    logits = _logits(3)
    emitted = [[6], [], [9]]  # request 1 is a partial-prefill chunk: discarded
    lp = _runner()._sampled_token_logprobs(sampled, logits, 1, emitted)
    assert lp.cu_num_generated_tokens == [0, 1, 1, 2]
    assert lp.logprob_token_ids[:, 0].tolist() == [6, 9]


def test_raw_logits_mode_and_full_vocab():
    sampled = torch.tensor([[4, -1]])
    logits = _logits(2)
    lp = _runner("raw_logits")._sampled_token_logprobs(sampled, logits, -1, [[4]])
    assert lp.logprobs.shape == (1, V + 1)  # sampled token + the whole vocabulary
    assert torch.allclose(lp.logprobs[0, 0], logits[0, 4])
