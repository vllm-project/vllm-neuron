# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the draft proposers' warmup readback under async scheduling.

With async scheduling, ``EagleProposer.warmup`` and ``DFlashProposer.warmup``
read the proposal back to the host to consume the device future. With
``VLLM_NEURON_CPU_COMPILE=1`` the worker runs on the meta device and the
parallel-trace pool calls ``warmup`` on meta inputs, so the proposal is a meta
tensor with no data and no future. Warmup must skip meta outputs and still
read back real ones.

The tests need no Neuron hardware. Run with:
    VLLM_NEURON_CPU_MODE=1 pytest test/unit/vllm/spec_decode/test_draft_warmup_readback.py -v
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from vllm_neuron.vllm.spec_decode.dflash import DFlashProposer
from vllm_neuron.vllm.spec_decode.eagle import EagleProposer

NUM_SPEC = 2
HIDDEN_SIZE = 8
LAYER = "layers.1.self_attn"
PROPOSERS = [EagleProposer, DFlashProposer]
PROPOSER_IDS = ["eagle3", "dflash"]
# (num_tokens, num_reqs): one prefill bucket and one decode batch.
SHAPES = [(16, 1), (2 * (1 + NUM_SPEC), 2)]


class _StubDraftModel(nn.Module):
    """Return the draft outputs on the input device, like the compiled draft."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=HIDDEN_SIZE, block_size=NUM_SPEC + 1)
        # DFlashProposer.propose reads the block size from the backbone config.
        self.model = SimpleNamespace(config=self.config)

    def forward(self, **kwargs):
        # EAGLE3 passes the per-request rows as sampling_positions, DFlash as
        # last_token_indices.
        key = (
            "sampling_positions"
            if "sampling_positions" in kwargs
            else "last_token_indices"
        )
        rows = kwargs[key]
        num_reqs, device = rows.shape[0], rows.device
        draft_token_ids = torch.zeros(
            num_reqs, NUM_SPEC + 1, dtype=torch.int32, device=device
        )
        drafts_only = torch.zeros(num_reqs, NUM_SPEC, dtype=torch.int32, device=device)
        return draft_token_ids, drafts_only, None


def _proposer(cls: Any, device: str, async_scheduling: bool) -> Any:
    """Build a proposer around the stub draft, skipping config parsing and loading."""
    proposer = cls.__new__(cls)
    proposer.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling)
    )
    proposer.speculative_config = SimpleNamespace(model="stub-draft")
    # DFlash sizes the aux hidden states from the draft's target layer ids.
    proposer.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(target_layer_ids=[1, 2, 3])
    )
    proposer.num_speculative_tokens = NUM_SPEC
    proposer.device = torch.device(device)
    proposer.model = _StubDraftModel()
    proposer.attn_layer_names = [LAYER]
    # EAGLE3 passes the TP rank as a tensor on the input device.
    proposer.rank_tensor = torch.tensor(0, dtype=torch.int32, device=device)
    return proposer


@pytest.fixture
def readbacks():
    """Record the device of every tensor that ``Tensor.cpu`` reads back."""
    devices: list[str] = []
    original_cpu = torch.Tensor.cpu

    def _recording_cpu(self, *args, **kwargs):
        devices.append(self.device.type)
        return original_cpu(self, *args, **kwargs)

    with patch.object(torch.Tensor, "cpu", _recording_cpu):
        yield devices


@pytest.mark.parametrize("num_tokens, num_reqs", SHAPES, ids=["prefill", "decode"])
@pytest.mark.parametrize("cls", PROPOSERS, ids=PROPOSER_IDS)
def test_async_warmup_on_meta_skips_readback(cls, num_tokens, num_reqs, readbacks):
    # CPU compile: the parallel-trace pool calls warmup on the meta device.
    proposer = _proposer(cls, "meta", async_scheduling=True)
    proposer.warmup(num_tokens=num_tokens, num_reqs=num_reqs, attn_metadata={LAYER: {}})
    assert readbacks == []


@pytest.mark.parametrize("cls", PROPOSERS, ids=PROPOSER_IDS)
def test_async_warmup_on_device_reads_back_every_output(cls, readbacks):
    # A device output holds the async future, so warmup still reads back both
    # proposal tensors. CPU tensors stand in for device tensors here.
    proposer = _proposer(cls, "cpu", async_scheduling=True)
    proposer.warmup(num_tokens=16, num_reqs=1, attn_metadata={LAYER: {}})
    assert readbacks == ["cpu", "cpu"]


@pytest.mark.parametrize("device", ["meta", "cpu"])
@pytest.mark.parametrize("cls", PROPOSERS, ids=PROPOSER_IDS)
def test_sync_warmup_reads_nothing_back(cls, device, readbacks):
    proposer = _proposer(cls, device, async_scheduling=False)
    proposer.warmup(num_tokens=16, num_reqs=1, attn_metadata={LAYER: {}})
    assert readbacks == []
