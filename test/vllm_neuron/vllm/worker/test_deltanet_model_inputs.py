# SPDX-License-Identifier: Apache-2.0
"""The DeltaNet model inputs of the runner: warmup inputs and the padding mask.

A hybrid model gets ``batch_slot_indices`` and ``deltanet_padding_mask`` at warmup
and at every real step; the Qwen3.5 prefill reads the mask. A dense model gets
neither, so its traced graph and its compile cache stay the same. CPU only.
"""

from types import SimpleNamespace

import torch

from vllm_neuron.vllm.worker.deltanet_state_slots import DeltaNetStateSlots
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

_CPU = torch.device("cpu")


def _runner(hybrid):
    return SimpleNamespace(
        _deltanet_state_slots=DeltaNetStateSlots(32) if hybrid else None
    )


def test_a_hybrid_prefill_warmup_gets_one_state_row_and_a_full_mask():
    out = NeuronModelRunner._deltanet_warmup_inputs(
        _runner(True), num_slot_rows=1, num_tokens=256, device=_CPU
    )
    assert out["batch_slot_indices"].shape == (1,)
    assert out["batch_slot_indices"].dtype == torch.long
    mask = out["deltanet_padding_mask"]
    assert mask.shape == (256,) and mask.dtype == torch.bfloat16
    assert bool((mask == 1).all())


def test_a_hybrid_decode_warmup_gets_a_row_per_token():
    out = NeuronModelRunner._deltanet_warmup_inputs(
        _runner(True), num_slot_rows=32, num_tokens=32, device=_CPU
    )
    assert out["batch_slot_indices"].shape == (32,)
    assert out["deltanet_padding_mask"].shape == (32,)


def test_a_dense_model_gets_no_deltanet_inputs():
    runner = _runner(False)
    assert (
        NeuronModelRunner._deltanet_warmup_inputs(
            runner, num_slot_rows=1, num_tokens=256, device=_CPU
        )
        == {}
    )
    assert NeuronModelRunner._deltanet_padding_mask_for(runner, torch.arange(8)) is None


def test_the_mask_marks_the_padding_of_a_continuation_chunk():
    # 5 real tokens at positions 100..104, then 3 padding slots that repeat 104.
    positions = torch.tensor([100, 101, 102, 103, 104, 104, 104, 104])
    mask = NeuronModelRunner._deltanet_padding_mask_for(_runner(True), positions)
    assert mask.dtype == torch.bfloat16
    assert mask.tolist() == [1, 1, 1, 1, 1, 0, 0, 0]


def test_no_positions_give_no_mask():
    runner = _runner(True)
    assert NeuronModelRunner._deltanet_padding_mask_for(runner, None) is None
    empty = torch.tensor([], dtype=torch.long)
    assert NeuronModelRunner._deltanet_padding_mask_for(runner, empty) is None
