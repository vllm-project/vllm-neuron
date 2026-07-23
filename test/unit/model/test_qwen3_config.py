# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_neuron.model.qwen3.model_bf16 import (
    Qwen3Config,
    Qwen3RotaryEmbedding,
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
