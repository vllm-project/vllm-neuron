# SPDX-License-Identifier: Apache-2.0
"""The 2 decode writes of the DeltaNet conv window must give the same buffer.

A decode step writes the new conv window of each batch row to the state row of
its request. VLLM_NEURON_BACKEND_DELTANET_CONV_STATE_WRITE selects the write:
"select" (the default) or "scatter" (index_put_). "select" does not use a
scatter, because the compiler lowers the scatter of some decode buckets to
GpSimd.

The writes only move data, so these tests require equal bits:
- each state row of a request gets the window of its batch row,
- a state row that no batch row writes keeps its window,
- the free row that the padded rows share gets the window of one of its writers,
- a full decode step gives the same output and the same state with each write.

These tests run the pure-torch paths in CPU mode at TP=1.
"""

import os

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import pytest
import torch

from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.model.qwen3_5 import model_bf16 as m
from vllm_neuron.model.qwen3_5.config import Qwen3_5Config

ROWS = 6
WRITES = ("scatter", "select")


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
        neuron_config=NeuronConfig(num_seqs_buckets=[ROWS]),
    )


@pytest.fixture(autouse=True)
def _tp1(monkeypatch):
    monkeypatch.setattr(m, "get_tp_group", lambda: _NoOpGroup())
    monkeypatch.setenv("VLLM_NEURON_CPU_MODE", "1")


def _layer(monkeypatch, write: str) -> m.Qwen3_5DeltaNet:
    monkeypatch.setenv("VLLM_NEURON_BACKEND_DELTANET_CONV_STATE_WRITE", write)
    layer = m.Qwen3_5DeltaNet(_tiny_config(), layer_idx=0)
    torch.manual_seed(42)
    for name, p in layer.named_parameters():
        with torch.no_grad():
            if name in ("recurrent_state", "conv_state"):
                p.normal_(0.0, 1.0)
            elif name == "a_log":
                p.uniform_(-2.0, 0.0)
            elif name == "dt_bias":
                p.uniform_(-1.0, 1.0)
            else:
                p.normal_(0.0, 0.05)
    return layer


def _new_windows(layer: m.Qwen3_5DeltaNet, batch: int) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(batch, *layer.conv_state.shape[1:])


@pytest.mark.parametrize("write", WRITES)
def test_each_request_row_gets_the_window_of_its_batch_row(monkeypatch, write):
    layer = _layer(monkeypatch, write)
    old = layer.conv_state.detach().clone()
    # 2 requests in rows 3 and 0, and 3 padded rows that share the free row 5.
    slot_idx = torch.tensor([3, 0, 5, 5, 5])
    new = _new_windows(layer, 5)
    with torch.no_grad():
        layer._write_decode_conv_state(slot_idx, new)
    got = layer.conv_state.detach()
    assert torch.equal(got[3], new[0])
    assert torch.equal(got[0], new[1])
    for row in (1, 2, 4):
        assert torch.equal(got[row], old[row]), f"row {row} has no writer"
    assert any(torch.equal(got[5], new[j]) for j in (2, 3, 4))
    if write == "select":
        assert torch.equal(got[5], new[4]), "select uses the last writer"


def test_the_writes_give_equal_buffers(monkeypatch):
    slot_idx = torch.tensor([4, 1, 2, 0, 5, 5])
    buffers = {}
    for write in WRITES:
        layer = _layer(monkeypatch, write)
        with torch.no_grad():
            layer._write_decode_conv_state(slot_idx, _new_windows(layer, 6))
        buffers[write] = layer.conv_state.detach().clone()
    assert torch.equal(buffers["select"][:5], buffers["scatter"][:5])
    new = _new_windows(_layer(monkeypatch, "select"), 6)
    assert torch.equal(buffers["select"][5], new[5])


def test_a_decode_step_is_the_same_with_each_write(monkeypatch):
    torch.manual_seed(11)
    h = torch.randn(4, 64)
    positions = torch.tensor([20, 31, 9, 9])
    # 3 requests in rows 2, 5 and 0, and 1 padded row on the free row 1.
    slot_idx = torch.tensor([2, 5, 0, 1])
    results = {}
    for write in WRITES:
        layer = _layer(monkeypatch, write)
        with torch.no_grad():
            out = layer(h, positions, is_prefill=False, batch_slot_indices=slot_idx)
        results[write] = (
            out.detach().clone(),
            layer.conv_state.detach().clone(),
            layer.recurrent_state.detach().clone(),
        )
    # Each state row has at most 1 writer here, so all rows must be equal.
    ref_out, ref_conv, ref_rec = results["scatter"]
    assert not torch.equal(ref_conv[2], _layer(monkeypatch, "scatter").conv_state[2])
    out, conv, rec = results["select"]
    assert torch.equal(out, ref_out)
    assert torch.equal(conv, ref_conv)
    assert torch.equal(rec, ref_rec)


def test_a_wrong_write_name_stops_the_layer(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_BACKEND_DELTANET_CONV_STATE_WRITE", "fast")
    with pytest.raises(ValueError, match="CONV_STATE_WRITE"):
        m.Qwen3_5DeltaNet(_tiny_config(), layer_idx=0)


def test_the_default_write_is_select(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_BACKEND_DELTANET_CONV_STATE_WRITE", raising=False)
    layer = m.Qwen3_5DeltaNet(_tiny_config(), layer_idx=0)
    assert layer._dn_conv_write == "select"
