# SPDX-License-Identifier: Apache-2.0
"""A prefix cache hit must continue from the state that a miss saved.

Prefix caching skips the prefill of a cached prefix. A DeltaNet layer has no KV
cache, so the layer saves its recurrent state and its conv window at the end of
a prefill step (a snapshot row), and a later request with the same prefix
restores them. ``deltanet_snapshot_rows`` holds ``[restore_row, save_row]``, and
row 0 is a scratch row.

The property that these tests check: a request that restores a snapshot gives
the same output and the same final state, bit for bit, as the request that saved
it and continued from its own state row. The two run the same arithmetic on the
same inputs, so any difference is a defect in the restore or the save.

These tests run the pure-torch paths in CPU mode at TP=1.
"""

import os

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import pytest
import torch

from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.model.qwen3_5 import model_bf16 as m
from vllm_neuron.model.qwen3_5.config import Qwen3_5Config

SNAPSHOT_ROWS = 3


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
            num_seqs_buckets=[1, 2], deltanet_snapshot_rows=snapshot_rows
        ),
    )


def _randomize(module: torch.nn.Module) -> None:
    torch.manual_seed(42)
    for name, p in module.named_parameters():
        with torch.no_grad():
            if name in ("recurrent_state", "conv_state"):
                p.zero_()
            elif name in ("recurrent_snapshot", "conv_snapshot"):
                # Garbage, so a test can see a row that a step must not read.
                p.normal_(0.0, 10.0)
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


@pytest.fixture
def layer(monkeypatch):
    built = m.Qwen3_5DeltaNet(_tiny_config(), layer_idx=0)
    _randomize(built)
    return built


def _prefill(layer, h, start: int, slot: int, rows: tuple[int, int] | None):
    positions = torch.arange(start, start + h.shape[0])
    return layer(
        h,
        positions,
        is_prefill=True,
        batch_slot_indices=torch.tensor([slot]),
        deltanet_padding_mask=torch.ones(h.shape[0], dtype=torch.bfloat16),
        deltanet_snapshot_rows=None if rows is None else torch.tensor(list(rows)),
    )


def _inputs(num_tokens: int, hidden: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(num_tokens, hidden)


def test_the_layer_holds_the_snapshot_rows(layer):
    assert layer.recurrent_snapshot.shape[0] == SNAPSHOT_ROWS
    assert layer.conv_snapshot.shape[0] == SNAPSHOT_ROWS
    assert layer.recurrent_snapshot.shape[1:] == layer.recurrent_state.shape[1:]
    assert layer.conv_snapshot.shape[1:] == layer.conv_state.shape[1:]


def test_a_restored_snapshot_continues_like_the_request_that_saved_it(layer):
    T, C = 24, 16
    h = _inputs(T, layer.hidden_size, seed=1)

    # Request A in state row 0: the first step saves to snapshot row 1, and the
    # second step continues from row 0.
    _prefill(layer, h[:C], 0, slot=0, rows=(0, 1))
    out_a = _prefill(layer, h[C:], C, slot=0, rows=(0, 0))
    rec_a = layer.recurrent_state[0].detach().clone()
    conv_a = layer.conv_state[0].detach().clone()

    # Request B in state row 1 holds garbage. It restores snapshot row 1 and runs
    # the same second step.
    with torch.no_grad():
        layer.recurrent_state[1].normal_(0.0, 10.0)
        layer.conv_state[1].normal_(0.0, 10.0)
    out_b = _prefill(layer, h[C:], C, slot=1, rows=(1, 0))

    assert torch.equal(out_b, out_a)
    assert torch.equal(layer.recurrent_state[1], rec_a)
    assert torch.equal(layer.conv_state[1], conv_a)


def test_a_restore_matches_a_single_pass(layer):
    T, C = 24, 16
    h = _inputs(T, layer.hidden_size, seed=2)
    ref = _prefill(layer, h, 0, slot=0, rows=(0, 0))
    ref_rec = layer.recurrent_state[0].detach().clone()

    _prefill(layer, h[:C], 0, slot=0, rows=(0, 2))
    out = _prefill(layer, h[C:], C, slot=1, rows=(2, 0))

    torch.testing.assert_close(out, ref[C:], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(layer.recurrent_state[1], ref_rec, rtol=1e-4, atol=1e-4)


def test_a_save_writes_the_state_of_the_request_row(layer):
    h = _inputs(16, layer.hidden_size, seed=3)
    _prefill(layer, h, 0, slot=0, rows=(0, 2))
    assert torch.equal(layer.recurrent_snapshot[2], layer.recurrent_state[0])
    assert torch.equal(layer.conv_snapshot[2], layer.conv_state[0])


def test_a_step_that_saves_nothing_leaves_the_snapshots(layer):
    h = _inputs(16, layer.hidden_size, seed=4)
    before = layer.recurrent_snapshot[1:].detach().clone()
    conv_before = layer.conv_snapshot[1:].detach().clone()
    _prefill(layer, h, 0, slot=0, rows=(0, 0))
    assert torch.equal(layer.recurrent_snapshot[1:], before)
    assert torch.equal(layer.conv_snapshot[1:], conv_before)


def test_restore_row_0_reads_the_request_row(layer):
    """Row 0 holds garbage, so a read of it would change the output."""
    T, C = 24, 16
    h = _inputs(T, layer.hidden_size, seed=5)
    _prefill(layer, h[:C], 0, slot=0, rows=None)
    with_rows = _prefill(layer, h[C:], C, slot=0, rows=(0, 0))

    _prefill(layer, h[:C], 0, slot=0, rows=None)
    without_rows = _prefill(layer, h[C:], C, slot=0, rows=None)
    assert torch.equal(with_rows, without_rows)


def test_a_decode_step_ignores_the_snapshot_rows(layer):
    h = _inputs(16, layer.hidden_size, seed=6)
    _prefill(layer, h, 0, slot=0, rows=(0, 1))
    snap = layer.recurrent_snapshot.detach().clone()
    layer(
        _inputs(1, layer.hidden_size, seed=7),
        torch.tensor([16]),
        is_prefill=False,
        batch_slot_indices=torch.tensor([0]),
        deltanet_snapshot_rows=torch.tensor([0, 2]),
    )
    assert torch.equal(layer.recurrent_snapshot, snap)


def test_without_snapshot_rows_the_layer_has_no_snapshots():
    plain = m.Qwen3_5DeltaNet(_tiny_config(snapshot_rows=0), layer_idx=0)
    assert not hasattr(plain, "recurrent_snapshot")
    _randomize(plain)
    h = _inputs(16, plain.hidden_size, seed=8)
    # The input is ignored, so a caller can pass it to any layer.
    out_rows = _prefill(plain, h, 0, slot=0, rows=(1, 2))
    with torch.no_grad():
        plain.recurrent_state.zero_()
        plain.conv_state.zero_()
    out_none = _prefill(plain, h, 0, slot=0, rows=None)
    assert torch.equal(out_rows, out_none)
