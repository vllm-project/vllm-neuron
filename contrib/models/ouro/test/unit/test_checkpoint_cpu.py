# SPDX-License-Identifier: Apache-2.0
"""Ouro-1.4B tests in CPU mode with the real weights (the port's CPU tests).

These tests read the pinned ``ByteDance/Ouro-1.4B`` checkpoint (2.9 GB) from
``OURO_CHECKPOINT_PATH`` and skip when the variable is not set. They check the
checkpoint identity, and compare the real modules with the independent torch
reference in ``ouro_reference.py``:

- ``test_decoder_layer_prefill``: decoder layer 0 on UT step 1 at TP 2 with
  SP. A three-way check applies to the output and to the KV cache write. The
  KV caches of the other UT steps must stay zero.
- ``test_exit_select``: the early-exit selection with the real gate weights
  is bitwise equal to the reference.
- ``test_kv_spec_and_binding``: the 96 KV layers, the binding and the cache
  owners at TP 8.
- ``test_full_weight_load``: ``load_weights`` at TP 2 gives the hand-sliced
  shards of layers 0 and 23 and of the model-level tensors.

Run with:
    OURO_CHECKPOINT_PATH=/path/to/Ouro-1.4B VLLM_NEURON_CPU_MODE=1 \
        pytest contrib/models/ouro/test/unit/test_checkpoint_cpu.py -v
"""

import pytest

from vllm_neuron import envs

from ouro_component_checks import (
    check_decoder_layer_prefill,
    check_exit_select,
    check_full_weight_load,
    check_kv_spec_and_binding,
    load_gate,
)
from ouro_reference import read_config
from ouro_test_utils import real_checkpoint_or_skip

pytestmark = [
    pytest.mark.usefixtures("dist_cleanup"),
    pytest.mark.skipif(
        not envs.VLLM_NEURON_CPU_MODE,
        reason="These tests run in CPU mode (VLLM_NEURON_CPU_MODE=1).",
    ),
]


@pytest.fixture(scope="module")
def model_checkpoint():
    return real_checkpoint_or_skip()


def test_decoder_layer_prefill(model_checkpoint):
    """Three-way check of decoder layer 0 on UT step 1 at TP 2, 128 tokens."""
    check_decoder_layer_prefill(
        model_checkpoint,
        world_size=2,
        device="cpu",
        layer_idx=0,
        ut_step=1,
        seq_len=128,
        block_size=32,
    )


@pytest.mark.parametrize("threshold", [1.0, 0.5, None])
def test_exit_select(model_checkpoint, threshold):
    """The selected hidden state is bitwise equal to the reference."""
    gate_weight, gate_bias = load_gate(model_checkpoint)
    num_ut_steps = read_config(model_checkpoint)["total_ut_steps"]
    check_exit_select(gate_weight, gate_bias, num_ut_steps, threshold)


def test_kv_spec_and_binding(model_checkpoint):
    """96 KV layers (24 layers x 4 UT steps) at TP 8."""
    check_kv_spec_and_binding(model_checkpoint, world_size=8)


def test_full_weight_load(model_checkpoint):
    """load_weights at TP 2 gives the hand-sliced shards."""
    check_full_weight_load(model_checkpoint, world_size=2)
