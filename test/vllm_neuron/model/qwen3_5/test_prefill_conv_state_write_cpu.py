# SPDX-License-Identifier: Apache-2.0
"""The 2 prefill writes of the DeltaNet conv window must give the same buffers.

A prefill step writes the last conv window of its request to the state row of the
request and to its snapshot save row (row 0, the scratch row, when it saves
nothing). VLLM_NEURON_BACKEND_DELTANET_CONV_STATE_WRITE selects the write:
"select" (the default) or "scatter" (index_put_). The writes only move data, so
these tests require equal bits:
- the state row and the save row get the window, and every other row keeps its own,
- prefill steps (a first chunk, a continuation, a restore from a snapshot) give
  the same output, conv buffers and recurrent buffers with each write.

These tests run the pure-torch paths in CPU mode at TP=1.
"""

import os

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import pytest
import torch

from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.model.qwen3_5 import model_bf16 as m
from vllm_neuron.model.qwen3_5.config import Qwen3_5Config

ROWS = 4
SNAPSHOT_ROWS = 3
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


def _tiny_config(snapshot_rows: int = SNAPSHOT_ROWS) -> Qwen3_5Config:
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
        neuron_config=NeuronConfig(
            num_seqs_buckets=[ROWS], deltanet_snapshot_rows=snapshot_rows
        ),
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
            if name in (
                "recurrent_state",
                "conv_state",
                "recurrent_snapshot",
                "conv_snapshot",
            ):
                p.normal_(0.0, 1.0)
            elif name == "a_log":
                p.uniform_(-2.0, 0.0)
            elif name == "dt_bias":
                p.uniform_(-1.0, 1.0)
            else:
                p.normal_(0.0, 0.05)
    return layer


def _window(layer: m.Qwen3_5DeltaNet) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(1, *layer.conv_state.shape[1:])


@pytest.mark.parametrize("write", WRITES)
@pytest.mark.parametrize("save_row", [0, 2])
def test_the_state_row_and_the_save_row_get_the_window(monkeypatch, write, save_row):
    layer = _layer(monkeypatch, write)
    old_state = layer.conv_state.detach().clone()
    old_snap = layer.conv_snapshot.detach().clone()
    new = _window(layer)
    with torch.no_grad():
        layer._write_prefill_conv_state(
            torch.tensor([2]), new, torch.tensor([1, save_row])
        )
    state, snap = layer.conv_state.detach(), layer.conv_snapshot.detach()
    assert torch.equal(state[2], new[0])
    for row in (0, 1, 3):
        assert torch.equal(state[row], old_state[row]), f"state row {row} has no writer"
    assert torch.equal(snap[save_row], new[0])
    for row in range(SNAPSHOT_ROWS):
        if row != save_row:
            assert torch.equal(snap[row], old_snap[row]), (
                f"snapshot row {row} has no writer"
            )


@pytest.mark.parametrize("write", WRITES)
def test_no_snapshot_rows_write_the_state_row_only(monkeypatch, write):
    layer = _layer(monkeypatch, write)
    old_snap = layer.conv_snapshot.detach().clone()
    new = _window(layer)
    with torch.no_grad():
        layer._write_prefill_conv_state(torch.tensor([1]), new, None)
    assert torch.equal(layer.conv_state[1], new[0])
    assert torch.equal(layer.conv_snapshot, old_snap)


def _run_steps(layer: m.Qwen3_5DeltaNet):
    torch.manual_seed(11)
    h = torch.randn(24, 64)
    pad = torch.ones(16, dtype=torch.bfloat16)
    pad_tail = torch.cat([torch.ones(5), torch.zeros(3)]).to(torch.bfloat16)
    outs = []
    with torch.no_grad():
        # A first chunk that saves to snapshot row 1, in state row 3.
        outs.append(
            layer(
                h[:16],
                torch.arange(16),
                True,
                torch.tensor([3]),
                pad,
                torch.tensor([0, 1]),
            )
        )
        # A padded continuation chunk (5 real tokens of 8) that saves nothing.
        outs.append(
            layer(
                h[16:],
                torch.arange(16, 24),
                True,
                torch.tensor([3]),
                pad_tail,
                torch.tensor([0, 0]),
            )
        )
        # Another request in state row 0 that restores snapshot row 1 and saves to row 2.
        outs.append(
            layer(
                h[16:],
                torch.arange(16, 24),
                True,
                torch.tensor([0]),
                pad_tail,
                torch.tensor([1, 2]),
            )
        )
    return (
        [o.detach().clone() for o in outs],
        layer.conv_state.detach().clone(),
        layer.conv_snapshot.detach().clone(),
        layer.recurrent_state.detach().clone(),
        layer.recurrent_snapshot.detach().clone(),
    )


def test_prefill_steps_are_the_same_with_each_write(monkeypatch):
    results = {write: _run_steps(_layer(monkeypatch, write)) for write in WRITES}
    ref, got = results["scatter"], results["select"]
    for a, b in zip(ref[0], got[0]):
        assert torch.equal(a, b)
    for a, b in zip(ref[1:], got[1:]):
        assert torch.equal(a, b)
    # The steps did write: the state row of the first request changed.
    fresh = _layer(monkeypatch, "scatter")
    assert not torch.equal(ref[1][3], fresh.conv_state[3])
