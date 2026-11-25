# SPDX-License-Identifier: Apache-2.0
"""Chunked (segmented) prefill must equal single-shot prefill for Qwen3_5.

Long-context serving (max_model_len=131072) runs prefill in fixed-size chunks.
Each chunk is a separate forward pass. Two layer types carry state between
chunks:

- ``Qwen3_5FullAttention``: chunk k must attend to the KV of chunks 0..k
  that earlier passes wrote to the paged cache (``kv_segment_size`` branch).
- ``Qwen3_5DeltaNet``: the recurrent state and the conv1d left context
  must flow from chunk k-1 into chunk k through the state buffers.

These tests run the pure-torch paths in CPU mode at TP=1 and compare the
chunked outputs and final states against one pass over the whole sequence.
"""

import os

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import pytest
import torch

from vllm_neuron.model.qwen3_5 import model_bf16 as m
from vllm_neuron.model.qwen3_5.config import Qwen3_5Config


class _NoOpGroup:
    world_size = 1
    rank_in_group = 0

    @staticmethod
    def all_gather(x, dim=0):
        return x

    @staticmethod
    def reduce_scatter(x, dim=0):
        return x

    @staticmethod
    def all_reduce(x):
        return x


def _tiny_config() -> Qwen3_5Config:
    return Qwen3_5Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        partial_rotary_factor=0.25,
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
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
        torch_dtype=torch.float32,
    )


def _randomize(module: torch.nn.Module) -> None:
    torch.manual_seed(42)
    for name, p in module.named_parameters():
        with torch.no_grad():
            if name in ("recurrent_state", "conv_state"):
                p.zero_()
            elif name == "a_log":
                p.uniform_(-2.0, 0.0)
            elif name == "dt_bias":
                p.uniform_(-1.0, 1.0)
            else:
                p.normal_(0.0, 0.05)


@pytest.fixture(autouse=True)
def _tp1(monkeypatch):
    monkeypatch.setattr(m, "get_tp_group", lambda: _NoOpGroup())
    monkeypatch.setenv("VLLM_NEURON_CPU_MODE", "1")


# ── DeltaNet ──────────────────────────────────────────────────────────────────


def _deltanet_prefill(layer, h, positions, mask):
    return layer(
        h,
        positions,
        is_prefill=True,
        batch_slot_indices=torch.tensor([0]),
        deltanet_padding_mask=mask,
    )


def _reset_deltanet(layer):
    with torch.no_grad():
        layer.recurrent_state.zero_()
        layer.conv_state.zero_()


def test_deltanet_two_full_chunks_match_single_shot():
    cfg = _tiny_config()
    layer = m.Qwen3_5DeltaNet(cfg, layer_idx=0)
    _randomize(layer)
    T, C = 16, 8
    torch.manual_seed(1)
    h = torch.randn(T, cfg.hidden_size)
    pos = torch.arange(T)
    ones = torch.ones(T, dtype=torch.bfloat16)

    _reset_deltanet(layer)
    ref = _deltanet_prefill(layer, h, pos, ones)
    ref_rec = layer.recurrent_state.detach().clone()
    ref_conv = layer.conv_state.detach().clone()

    _reset_deltanet(layer)
    out0 = _deltanet_prefill(layer, h[:C], pos[:C], ones[:C])
    out1 = _deltanet_prefill(layer, h[C:], pos[C:], ones[C:])
    out = torch.cat([out0, out1], dim=0)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(layer.recurrent_state, ref_rec, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(layer.conv_state, ref_conv, rtol=1e-4, atol=1e-4)


def test_deltanet_padded_last_chunk_matches_single_shot():
    cfg = _tiny_config()
    layer = m.Qwen3_5DeltaNet(cfg, layer_idx=0)
    _randomize(layer)
    C, real = 8, 5
    T = C + real
    torch.manual_seed(2)
    h = torch.randn(T, cfg.hidden_size)

    _reset_deltanet(layer)
    ref = _deltanet_prefill(
        layer, h, torch.arange(T), torch.ones(T, dtype=torch.bfloat16)
    )
    ref_rec = layer.recurrent_state.detach().clone()
    ref_conv = layer.conv_state.detach().clone()

    _reset_deltanet(layer)
    _deltanet_prefill(
        layer, h[:C], torch.arange(C), torch.ones(C, dtype=torch.bfloat16)
    )
    # Runner padding contract: pad tokens repeat the last real position.
    pad = C - real
    h2 = torch.cat([h[C:], torch.zeros(pad, cfg.hidden_size)])
    pos2 = torch.cat([torch.arange(C, T), torch.full((pad,), T - 1)])
    mask2 = torch.cat([torch.ones(real), torch.zeros(pad)]).to(torch.bfloat16)
    out2 = _deltanet_prefill(layer, h2, pos2, mask2)

    torch.testing.assert_close(out2[:real], ref[C:], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(layer.recurrent_state, ref_rec, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(layer.conv_state, ref_conv, rtol=1e-4, atol=1e-4)


def test_deltanet_continuation_chunk_shorter_than_conv_kernel():
    cfg = _tiny_config()
    layer = m.Qwen3_5DeltaNet(cfg, layer_idx=0)
    _randomize(layer)
    C, real = 8, 2
    T = C + real
    torch.manual_seed(3)
    h = torch.randn(T, cfg.hidden_size)

    _reset_deltanet(layer)
    _deltanet_prefill(layer, h, torch.arange(T), torch.ones(T, dtype=torch.bfloat16))
    ref_conv = layer.conv_state.detach().clone()

    _reset_deltanet(layer)
    _deltanet_prefill(
        layer, h[:C], torch.arange(C), torch.ones(C, dtype=torch.bfloat16)
    )
    pad = C - real
    h2 = torch.cat([h[C:], torch.zeros(pad, cfg.hidden_size)])
    pos2 = torch.cat([torch.arange(C, T), torch.full((pad,), T - 1)])
    mask2 = torch.cat([torch.ones(real), torch.zeros(pad)]).to(torch.bfloat16)
    _deltanet_prefill(layer, h2, pos2, mask2)

    # The saved conv window must span the chunk boundary: one token from
    # chunk 0 and the two real tokens of chunk 1.
    torch.testing.assert_close(layer.conv_state, ref_conv, rtol=1e-4, atol=1e-4)


# ── Full attention ────────────────────────────────────────────────────────────

_BLOCK = 4


def _attn_layer(cfg):
    layer = m.Qwen3_5FullAttention(cfg, layer_idx=3)
    _randomize(layer)
    num_blocks = 8
    layer.k_cache = torch.zeros(num_blocks, 1, _BLOCK, cfg.head_dim)
    layer.v_cache = torch.zeros(num_blocks, 1, _BLOCK, cfg.head_dim)
    return layer


def _attn_meta(layer_idx, positions, block_table, cached_seq_len, kv_segment_size):
    slot_mapping = torch.tensor(
        [block_table[p // _BLOCK] * _BLOCK + p % _BLOCK for p in positions.tolist()]
    )
    return {
        f"layers.{layer_idx}.self_attn": {
            "slot_mapping": slot_mapping,
            "block_size": _BLOCK,
            "block_table_tensor": torch.tensor([block_table]),
            "cached_seq_len": torch.tensor([[cached_seq_len]], dtype=torch.int32),
            "kv_segment_size": kv_segment_size,
            "max_query_len": len(positions),
            "decode_token_threshold": 1,
        }
    }


def _attn_prefill(layer, rotary, h, positions, meta):
    cos, sin = rotary(positions, device=h.device, dtype=h.dtype)
    return layer.forward_prefill(h, positions, (cos, sin), meta)


def test_full_attention_two_chunks_match_single_shot():
    cfg = _tiny_config()
    rotary = m.Qwen3_5RotaryEmbedding(cfg)
    T, C = 16, 8
    blocks = [1, 2, 3, 4]
    torch.manual_seed(4)
    h = torch.randn(T, cfg.hidden_size)
    pos = torch.arange(T)

    layer = _attn_layer(cfg)
    ref = _attn_prefill(
        layer, rotary, h, pos, _attn_meta(3, pos, blocks, 0, kv_segment_size=0)
    )

    layer = _attn_layer(cfg)
    out0 = _attn_prefill(
        layer, rotary, h[:C], pos[:C], _attn_meta(3, pos[:C], blocks, 0, 512)
    )
    out1 = _attn_prefill(
        layer, rotary, h[C:], pos[C:], _attn_meta(3, pos[C:], blocks, C, 512)
    )
    out = torch.cat([out0, out1], dim=0)

    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_full_attention_second_chunk_sees_first_chunk():
    """Sanity check for the test itself: without the prior KV the second
    chunk's output must differ, so a passing equality above is meaningful."""
    cfg = _tiny_config()
    rotary = m.Qwen3_5RotaryEmbedding(cfg)
    T, C = 16, 8
    blocks = [1, 2, 3, 4]
    torch.manual_seed(4)
    h = torch.randn(T, cfg.hidden_size)
    pos = torch.arange(T)

    layer = _attn_layer(cfg)
    ref = _attn_prefill(
        layer, rotary, h, pos, _attn_meta(3, pos, blocks, 0, kv_segment_size=0)
    )
    layer = _attn_layer(cfg)
    alone = _attn_prefill(
        layer, rotary, h[C:], pos[C:], _attn_meta(3, pos[C:], blocks, C, 512)
    )
    assert not torch.allclose(alone, ref[C:], rtol=1e-3, atol=1e-3)
