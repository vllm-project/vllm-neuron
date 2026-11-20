# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_neuron.functional.sampling_vocabulary import SamplingVocabulary


def test_tp_shards_match_gathered_logits():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 20.0]])
    policy = SamplingVocabulary(6, (5, 1, 5))
    full = policy.mask(logits, 8, None)
    shards = [
        policy.mask(x, 8, torch.tensor([r]))
        for r, x in enumerate(logits.chunk(2, dim=-1))
    ]
    torch.testing.assert_close(torch.cat(shards, dim=-1), full)
    assert full.argmax(-1).tolist() == [4]
    assert full.isneginf().tolist() == [
        [False, True, False, False, False, True, True, True]
    ]
    assert SamplingVocabulary().mask(logits, 8, None) is logits


def test_sampler_intersects_static_vocabulary_and_grammar(monkeypatch):
    from vllm_neuron.model.neuron_config import NeuronConfig
    from vllm_neuron.nn import sampler

    config = NeuronConfig.from_dict(
        {
            "on_device_sampling_config": {
                "sampling_vocab_size": 6,
                "suppressed_token_ids": [5],
            }
        }
    )
    model = sampler.Sampler(config.on_device_sampling_config)
    monkeypatch.setattr(sampler, "sample", lambda logits, **kwargs: logits.argmax(-1))
    logits = torch.arange(8, dtype=torch.float32).view(1, 8)
    grammar = torch.ones_like(logits, dtype=torch.bool)
    grammar[:, 4] = False
    assert model(logits, logit_mask=grammar).item() == 3


@pytest.mark.parametrize("size,suppressed", [(0, ()), (2, (1, 0)), (None, (-1,))])
def test_reject_invalid_vocabulary(size, suppressed):
    with pytest.raises(ValueError):
        SamplingVocabulary(size, suppressed)


def test_sharded_mask_requires_rank():
    with pytest.raises(ValueError, match="TP rank"):
        SamplingVocabulary(6).mask(torch.zeros(1, 4), 8, None)
