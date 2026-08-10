# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner


def test_async_ods_computes_requested_logprobs():
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    sampling_metadata = SimpleNamespace(max_num_logprobs=5)
    runner.input_batch = SimpleNamespace(sampling_metadata=sampling_metadata)
    runner.use_async_scheduling = True
    runner.on_device_sampling = True
    runner._on_device_logits = torch.tensor([[1.0, 2.0]])
    expected_logprobs = object()

    def fake_sampler(*, logits, sampling_metadata):
        torch.testing.assert_close(logits, runner._on_device_logits)
        return SimpleNamespace(logprobs_tensors=expected_logprobs)

    runner.sampler = fake_sampler
    sampled_tokens = torch.tensor([1])

    output = runner._sample(sampled_tokens)

    assert output.sampled_token_ids is sampled_tokens
    assert output.logprobs_tensors is expected_logprobs
