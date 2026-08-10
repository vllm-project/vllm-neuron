# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import vllm_neuron.model.qwen3.model_bf16 as qwen3_model
from vllm_neuron.model.qwen3.model_bf16 import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3RotaryEmbedding,
    _sanitize_slot_mapping,
)


def test_qwen3_config_reads_modern_rope_parameters():
    config = Qwen3Config.from_configs(
        {
            "rope_theta": None,
            "rope_parameters": {
                "rope_theta": 1_000_000,
                "rope_type": "default",
            },
        },
        neuron_config=None,
    )

    assert config.rope_theta == 1_000_000


def test_qwen3_config_prefers_legacy_rope_theta():
    config = Qwen3Config.from_configs(
        {
            "rope_theta": 500_000,
            "rope_parameters": {"rope_theta": 1_000_000},
        },
        neuron_config=None,
    )

    assert config.rope_theta == 500_000


def test_qwen3_rotary_phase_is_computed_in_fp32():
    rotary = Qwen3RotaryEmbedding(
        head_dim=128,
        max_position_embeddings=4096,
        base=1_000_000,
    )
    positions = torch.tensor([256, 257, 511, 988], dtype=torch.int32)

    cos, sin = rotary(positions, dtype=torch.bfloat16)
    inv_freq = rotary._compute_inv_freq(positions.device)
    phase = torch.einsum("i,j->ij", positions.float(), inv_freq)
    phase = torch.cat([phase, phase], dim=-1)

    torch.testing.assert_close(cos, phase.cos().to(torch.bfloat16))
    torch.testing.assert_close(sin, phase.sin().to(torch.bfloat16))


def test_qwen3_config_defaults_to_qk_norm():
    config = Qwen3Config.from_configs({}, neuron_config=None)

    assert config.use_qk_norm is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"attention_bias": True}, "attention_bias=True"),
        ({"use_qk_norm": False}, "use_qk_norm=False"),
        ({"torch_dtype": "float16"}, "only torch.bfloat16"),
    ],
)
def test_qwen3_config_rejects_unsupported_variants(overrides, message):
    with pytest.raises(ValueError, match=message):
        Qwen3Config.from_configs(overrides, neuron_config=None)


def test_qwen3_slot_mapping_uses_null_block_for_padding():
    slot_mapping = torch.tensor([-1, 3, 8], dtype=torch.int64)

    sanitized = _sanitize_slot_mapping(slot_mapping, max_slot=8)

    torch.testing.assert_close(sanitized, torch.tensor([0, 3, 0]))


class _FakeTPGroup:
    device_group = object()

    def all_gather(self, tensor, dim):
        assert dim == 1
        return torch.cat((tensor, tensor + 10), dim=dim)


class _FakeQwen3Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.ones((1, 2), dtype=torch.bfloat16), []


class _FakeColumnParallelLinear(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        bias,
        dtype,
        gather_output,
        tp_group,
    ):
        super().__init__()
        self.gather_output = gather_output

    def forward(self, hidden_states):
        return torch.tensor([[1.0, 2.0]], dtype=hidden_states.dtype)


class _FakeSampler(torch.nn.Module):
    def __init__(self, config, process_group):
        super().__init__()

    def forward(self, logits, sampling_params, **kwargs):
        return torch.tensor([1])


def _build_test_model(monkeypatch, on_device_sampling_config, max_logprobs=0):
    monkeypatch.setattr(qwen3_model, "Qwen3Model", _FakeQwen3Model)
    monkeypatch.setattr(
        qwen3_model.neuron_nn,
        "ColumnParallelLinear",
        _FakeColumnParallelLinear,
    )
    monkeypatch.setattr(qwen3_model, "Sampler", _FakeSampler)
    monkeypatch.setattr(qwen3_model, "get_tp_group", _FakeTPGroup)
    monkeypatch.setattr(qwen3_model, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        qwen3_model, "get_tensor_model_parallel_world_size", lambda: 2
    )
    neuron_config = SimpleNamespace(
        on_device_sampling_config=on_device_sampling_config,
        max_logprobs=max_logprobs,
        debug_logits_dir=None,
    )
    config = Qwen3Config(
        hidden_size=2,
        vocab_size=4,
        neuron_config=neuron_config,
    )
    return Qwen3ForCausalLM(config)


def test_qwen3_cpu_sampling_gathers_full_vocab_logits(monkeypatch):
    model = _build_test_model(monkeypatch, on_device_sampling_config=None)

    assert model.lm_head.gather_output is True


def test_qwen3_on_device_sampling_returns_logits_for_logprobs(monkeypatch):
    model = _build_test_model(
        monkeypatch,
        on_device_sampling_config=object(),
        max_logprobs=5,
    )

    sampled_tokens, gathered_logits = model(
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0]),
        sampling_positions=torch.tensor([0]),
    )

    torch.testing.assert_close(sampled_tokens, torch.tensor([1]))
    torch.testing.assert_close(
        gathered_logits,
        torch.tensor([[1.0, 2.0, 11.0, 12.0]], dtype=torch.bfloat16),
    )
