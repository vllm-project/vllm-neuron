# SPDX-License-Identifier: Apache-2.0
"""One DeltaNet state row for each request must stay separate from the others.

A linear attention layer keeps a recurrent state and a convolution state for each
concurrent request. The layer holds one row for each row of the decode batch, and
``batch_slot_indices`` gives the row of each batch row. Two properties carry the
correctness of a batch of more than one request:

- **Isolation.** The state of a request changes only from its own tokens. A
  second request in the same batch must not change it.
- **Padded rows.** A decode batch pads up to a compiled bucket, so it can hold
  more rows than there are requests. The layer writes the state of every row
  back, so a padded row must point at a row that no request holds.

These tests run the pure-torch paths in CPU mode at TP=1. They compare a request
that shares a batch with a second request, or with a padded row, against the same
request alone.
"""

import os

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import pytest
import torch

from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.model.qwen3_5 import model_bf16 as m
from vllm_neuron.model.qwen3_5.config import Qwen3_5Config

NUM_SLOTS = 2
HIDDEN_SIZE = 64


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


def _tiny_config(num_seqs_buckets: list[int]) -> Qwen3_5Config:
    """Build a small config that holds one state row for each bucket row."""
    return Qwen3_5Config(
        vocab_size=256,
        hidden_size=HIDDEN_SIZE,
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
        neuron_config=NeuronConfig(num_seqs_buckets=num_seqs_buckets),
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


@pytest.fixture
def layer():
    built = m.Qwen3_5DeltaNet(_tiny_config([1, NUM_SLOTS]), layer_idx=0)
    _randomize(built)
    return built


def _reset(layer) -> None:
    with torch.no_grad():
        layer.recurrent_state.zero_()
        layer.conv_state.zero_()


def _prefill(layer, h, slot: int):
    """Prefill one request. A prefill step carries one request and one row."""
    positions = torch.arange(h.shape[0])
    return layer(
        h,
        positions,
        is_prefill=True,
        batch_slot_indices=torch.tensor([slot]),
        deltanet_padding_mask=torch.ones(h.shape[0], dtype=torch.bfloat16),
    )


def _decode(layer, h, slots: list[int]):
    """Decode one batch. ``h`` holds one row for each entry of ``slots``."""
    positions = torch.full((h.shape[0],), 7, dtype=torch.long)
    return layer(
        h,
        positions,
        is_prefill=False,
        batch_slot_indices=torch.tensor(slots),
    )


def _inputs(num_tokens: int, hidden: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(num_tokens, hidden)


def test_the_state_of_a_request_ignores_a_second_request(layer):
    # Request A shares a decode batch with request B. The output of A and the
    # state row of A must equal the values that A produces alone.
    hidden = HIDDEN_SIZE
    prompt_a = _inputs(6, hidden, seed=1)
    prompt_b = _inputs(6, hidden, seed=2)
    token_a = _inputs(1, hidden, seed=3)
    token_b = _inputs(1, hidden, seed=4)

    _reset(layer)
    _prefill(layer, prompt_a, slot=0)
    alone = _decode(layer, token_a, slots=[0])
    alone_rec = layer.recurrent_state[0].detach().clone()
    alone_conv = layer.conv_state[0].detach().clone()

    _reset(layer)
    _prefill(layer, prompt_a, slot=0)
    _prefill(layer, prompt_b, slot=1)
    shared = _decode(layer, torch.cat([token_a, token_b], dim=0), slots=[0, 1])

    torch.testing.assert_close(shared[:1], alone, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        layer.recurrent_state[0], alone_rec, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(layer.conv_state[0], alone_conv, rtol=1e-5, atol=1e-5)


def test_the_batch_order_of_a_request_does_not_change_its_state(layer):
    # The scheduler can reorder the batch rows between two steps. A request keeps
    # its row, so its state must not change with its batch position.
    hidden = HIDDEN_SIZE
    prompt_a = _inputs(6, hidden, seed=1)
    prompt_b = _inputs(6, hidden, seed=2)
    token_a = _inputs(1, hidden, seed=3)
    token_b = _inputs(1, hidden, seed=4)

    _reset(layer)
    _prefill(layer, prompt_a, slot=0)
    _prefill(layer, prompt_b, slot=1)
    a_first = _decode(layer, torch.cat([token_a, token_b], dim=0), slots=[0, 1])
    rec_after_a_first = layer.recurrent_state.detach().clone()

    _reset(layer)
    _prefill(layer, prompt_a, slot=0)
    _prefill(layer, prompt_b, slot=1)
    b_first = _decode(layer, torch.cat([token_b, token_a], dim=0), slots=[1, 0])

    torch.testing.assert_close(b_first[1:], a_first[:1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(b_first[:1], a_first[1:], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        layer.recurrent_state, rec_after_a_first, rtol=1e-5, atol=1e-5
    )


def test_a_padded_row_on_a_free_row_keeps_the_real_state(layer):
    # One request in a decode bucket of two rows. The padded row points at row 1,
    # which no request holds. Row 0 must be identical to the unpadded run.
    hidden = HIDDEN_SIZE
    prompt_a = _inputs(6, hidden, seed=1)
    token_a = _inputs(1, hidden, seed=3)
    padded_row = _inputs(1, hidden, seed=99)

    _reset(layer)
    _prefill(layer, prompt_a, slot=0)
    alone = _decode(layer, token_a, slots=[0])
    alone_rec = layer.recurrent_state[0].detach().clone()
    alone_conv = layer.conv_state[0].detach().clone()

    _reset(layer)
    _prefill(layer, prompt_a, slot=0)
    padded = _decode(layer, torch.cat([token_a, padded_row], dim=0), slots=[0, 1])

    torch.testing.assert_close(padded[:1], alone, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        layer.recurrent_state[0], alone_rec, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(layer.conv_state[0], alone_conv, rtol=1e-5, atol=1e-5)


def test_a_padded_row_that_shares_the_real_row_corrupts_it(layer):
    # This test gives the reason for the free row above. It sends the padded row
    # to row 0, which the request holds. The state of the request then changes.
    # A test that cannot see this fault cannot protect the free-row rule.
    hidden = HIDDEN_SIZE
    prompt_a = _inputs(6, hidden, seed=1)
    token_a = _inputs(1, hidden, seed=3)
    padded_row = _inputs(1, hidden, seed=99)

    _reset(layer)
    _prefill(layer, prompt_a, slot=0)
    _decode(layer, token_a, slots=[0])
    alone_rec = layer.recurrent_state[0].detach().clone()

    _reset(layer)
    _prefill(layer, prompt_a, slot=0)
    _decode(layer, torch.cat([token_a, padded_row], dim=0), slots=[0, 0])

    assert not torch.allclose(
        layer.recurrent_state[0], alone_rec, rtol=1e-5, atol=1e-5
    ), (
        "A padded row that shares the row of a request left the state unchanged. "
        "The test can no longer show why a padded row needs a free row."
    )


def test_a_reused_row_starts_from_zero_state(layer):
    # A finished request releases its row. The next request that takes the row
    # starts at position 0, so the reset in the prefill zeros the row first.
    hidden = HIDDEN_SIZE
    prompt_a = _inputs(6, hidden, seed=1)
    prompt_b = _inputs(6, hidden, seed=2)

    _reset(layer)
    fresh = _prefill(layer, prompt_b, slot=1)
    fresh_rec = layer.recurrent_state[1].detach().clone()

    _reset(layer)
    _prefill(layer, prompt_a, slot=1)  # request A holds row 1 and then finishes
    reused = _prefill(layer, prompt_b, slot=1)  # request B takes the same row

    torch.testing.assert_close(reused, fresh, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        layer.recurrent_state[1], fresh_rec, rtol=1e-5, atol=1e-5
    )


def test_a_batch_of_more_than_one_row_needs_the_row_map(layer):
    # Without the map the layer would send batch position i to row i. A batch
    # position does not identify a request, so the layer must stop.
    two_rows = _inputs(2, HIDDEN_SIZE, seed=5)
    with pytest.raises(ValueError) as excinfo:
        layer(
            two_rows,
            torch.zeros(2, dtype=torch.long),
            is_prefill=False,
            batch_slot_indices=None,
        )
    assert "batch_slot_indices" in str(excinfo.value)


def test_one_row_still_runs_without_the_row_map(layer):
    # The single-row path keeps working, so a caller that predates the row map
    # does not break.
    one_row = _inputs(1, HIDDEN_SIZE, seed=5)
    out = layer(
        one_row,
        torch.zeros(1, dtype=torch.long),
        is_prefill=False,
        batch_slot_indices=None,
    )
    assert out.shape[0] == 1


@pytest.mark.parametrize(
    "buckets, rows", [([1, 2, 3, 4], 4), ([1, 2, 4], 5), ([1, 4], 5), ([8], 9)]
)
def test_the_state_buffers_hold_the_rows_of_the_pool(buckets, rows):
    # The pool sizes itself with state_pool_rows(num_seqs_buckets): a row for
    # each request, and 1 row for the padded rows when the ladder pads. The
    # buffers must agree, because the pool hands out a row index that indexes
    # these buffers.
    from vllm_neuron.vllm.worker.deltanet_state_slots import state_pool_rows

    assert state_pool_rows(buckets) == rows
    layer = m.Qwen3_5DeltaNet(_tiny_config(buckets), layer_idx=0)
    assert layer.recurrent_state.shape[0] == rows
    assert layer.conv_state.shape[0] == rows


@pytest.mark.parametrize("rows", [2, 4, 8])
def test_the_decode_output_of_a_row_ignores_the_batch_width(rows):
    # The decode step makes the output in the layout [B, H, 1, v_dim]. The next
    # step needs [B, 1, H, v_dim]. The 2 layouts hold the elements in the same
    # order, so the code uses a reshape and not a transpose.
    #
    # A wrong layout mixes the heads of 2 rows. Then a row in a wide batch does
    # not match the same row alone. Test a batch of 4 and a batch of 8: the
    # compiler backend refuses the transpose at these widths, so the reshape is
    # the only path that these widths use.
    built = m.Qwen3_5DeltaNet(_tiny_config([1, rows]), layer_idx=0)
    _randomize(built)

    prompts = [_inputs(6, HIDDEN_SIZE, seed=10 + i) for i in range(rows)]
    tokens = [_inputs(1, HIDDEN_SIZE, seed=50 + i) for i in range(rows)]

    alone = []
    for row in range(rows):
        _reset(built)
        _prefill(built, prompts[row], slot=0)
        alone.append(_decode(built, tokens[row], slots=[0]).detach().clone())

    _reset(built)
    for row in range(rows):
        _prefill(built, prompts[row], slot=row)
    shared = _decode(built, torch.cat(tokens, dim=0), slots=list(range(rows)))

    assert shared.shape[0] == rows
    for row in range(rows):
        torch.testing.assert_close(
            shared[row : row + 1], alone[row], rtol=1e-5, atol=1e-5
        )
