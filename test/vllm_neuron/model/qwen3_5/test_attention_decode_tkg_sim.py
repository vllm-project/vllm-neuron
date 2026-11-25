# SPDX-License-Identifier: Apache-2.0
"""Full attention decode with the attention_tkg kernel against the torch path.

Run with the NKI simulator on CPU:
    NKI_SIMULATOR=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 VLLM_NEURON_CPU_MODE=1 \
        pytest test/vllm_neuron/model/qwen3_5/test_attention_decode_tkg_sim.py
"""

import os

import pytest
import torch

from vllm_neuron.model.qwen3_5 import model_bf16 as m
from vllm_neuron.model.qwen3_5.config import Qwen3_5Config

SIM = os.environ.get("NKI_SIMULATOR") == "1"
BLOCK = 64
HEAD_DIM = 256


class _NoOpGroup:
    world_size = 1
    rank_in_group = 0

    @staticmethod
    def all_reduce(x):
        return x


def _config() -> Qwen3_5Config:
    return Qwen3_5Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=HEAD_DIM,
        partial_rotary_factor=0.25,
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [16, 8, 8],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000,
            "rope_type": "default",
        },
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        torch_dtype=torch.bfloat16,
    )


@pytest.fixture(autouse=True)
def _tp1(monkeypatch):
    monkeypatch.setattr(m, "get_tp_group", lambda: _NoOpGroup())
    monkeypatch.setenv("VLLM_NEURON_CPU_MODE", "1")


def _layer_and_inputs():
    cfg = _config()
    layer = m.Qwen3_5FullAttention(cfg, layer_idx=3)
    torch.manual_seed(42)
    for p in layer.parameters():
        with torch.no_grad():
            p.normal_(0.0, 0.05)
    num_blocks = 9
    torch.manual_seed(1)
    k_cache = torch.randn(num_blocks, 1, BLOCK, HEAD_DIM).to(torch.bfloat16)
    v_cache = torch.randn(num_blocks, 1, BLOCK, HEAD_DIM).to(torch.bfloat16)
    # Rows 0 and 1 are real requests. Row 2 is a padded row: its hidden state is
    # NaN, and its write goes to the null block 0.
    positions = torch.tensor([70, 20, 0])
    block_table = torch.tensor([[1, 2], [3, 4], [-1, -1]])
    slot_mapping = torch.tensor([2 * BLOCK + 70 % BLOCK, 3 * BLOCK + 20, 0])
    hidden = torch.randn(3, cfg.hidden_size).to(torch.bfloat16)
    hidden[2] = float("nan")
    meta = {
        "layers.3.self_attn": {
            "slot_mapping": slot_mapping,
            "block_size": BLOCK,
            "block_table_tensor": block_table,
            "max_query_len": 1,
            "decode_token_threshold": 1,
        }
    }
    rotary = m.Qwen3_5RotaryEmbedding(cfg)
    cos, sin = rotary(positions, device=hidden.device, dtype=hidden.dtype)
    return layer, k_cache, v_cache, hidden, positions, (cos, sin), meta


def _decode(layer, kernel, k_cache, v_cache, hidden, positions, rope, meta):
    layer._attn_decode_kernel = kernel
    layer.k_cache = k_cache.clone()
    layer.v_cache = v_cache.clone()
    with torch.no_grad():
        out = layer.forward_decode(hidden, positions, rope, meta)
    return out.float(), layer.k_cache, layer.v_cache


@pytest.mark.skipif(
    not SIM, reason="Set NKI_SIMULATOR=1 to run the attention_tkg kernel."
)
def test_tkg_decode_matches_torch_decode():
    layer, k_cache, v_cache, hidden, positions, rope, meta = _layer_and_inputs()
    ref, _, _ = _decode(layer, "torch", k_cache, v_cache, hidden, positions, rope, meta)
    out, k_after, v_after = _decode(
        layer, "tkg", k_cache, v_cache, hidden, positions, rope, meta
    )
    real = slice(0, 2)
    assert torch.isfinite(out[real]).all()
    err = ((out[real] - ref[real]).norm() / ref[real].norm()).item()
    assert err < 2e-2
    # The NaN of the padded row must not reach the cache.
    assert torch.isfinite(k_after).all() and torch.isfinite(v_after).all()


def test_torch_decode_keeps_nan_writes():
    # The default path does not change: it writes the padded row as it is.
    layer, k_cache, v_cache, hidden, positions, rope, meta = _layer_and_inputs()
    _, k_after, _ = _decode(
        layer, "torch", k_cache, v_cache, hidden, positions, rope, meta
    )
    assert not torch.isfinite(k_after[0]).all()


def test_attn_decode_kernel_setting_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_BACKEND_QWEN3_5_ATTN_DECODE_KERNEL", "flash")
    with pytest.raises(ValueError):
        m.Qwen3_5FullAttention(_config(), layer_idx=3)
