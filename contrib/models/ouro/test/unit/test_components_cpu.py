# SPDX-License-Identifier: Apache-2.0
"""Ouro component tests in CPU mode, with a tiny random-weight checkpoint.

The tiny checkpoint (``ouro_test_utils.TINY_CONFIG``) has the Ouro
architecture at a small size: 2 decoder layers, 4 UT steps (8 KV cache
layers), sandwich norms and the early-exit gate. Each test builds the real
modules at TP 2 and compares them with the independent torch reference in
``ouro_reference.py`` (``ouro_component_checks.py`` has the checks).

- Decoder layer prefill on UT step 1 (three-way), with the KV cache write.
- Attention: flash prefill, segmented prefill (2 chunks) and decode (TP 4),
  each on one UT step. The KV caches of the other UT steps must stay zero.
- MLP in prefill (SP) and in decode.
- Early-exit selection: bitwise equal to the reference for the thresholds
  1.0, 0.5 and None.
- KV spec, cache binding and weight load of the full model.

No Neuron hardware and no download. Run with:
    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/ouro/test/unit/test_components_cpu.py -v
"""

import pytest

from vllm_neuron import envs

from ouro_component_checks import (
    check_attention_decode,
    check_attention_flash_prefill,
    check_attention_segmented_prefill,
    check_decoder_layer_prefill,
    check_exit_select,
    check_full_weight_load,
    check_kv_spec_and_binding,
    check_mlp,
    load_gate,
)
from ouro_test_utils import TINY_CONFIG, make_tiny_checkpoint

WORLD_SIZE = 2
# The torch fallback of NF.attention_decode (CPU mode) supports one KV head
# per rank with the per-head block table. TP 4 gives one of the 4 KV heads to
# each rank. The device kernel has no such limit.
DECODE_WORLD_SIZE = 4
# The three-way check compares error histograms, so it needs enough output
# elements. 32 decode tokens x hidden 256 gives 8192 elements, the same count
# as the device decode tests (4 tokens x hidden 2048). With 4 tokens of the
# tiny model the histogram is too sparse for a stable result.
DECODE_TOKENS = 32
DEVICE = "cpu"
BLOCK_SIZE = 32

pytestmark = [
    pytest.mark.usefixtures("dist_cleanup"),
    pytest.mark.skipif(
        not envs.VLLM_NEURON_CPU_MODE,
        reason="These tests run in CPU mode (VLLM_NEURON_CPU_MODE=1).",
    ),
]


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    return make_tiny_checkpoint(str(tmp_path_factory.mktemp("tiny_ouro")))


def test_decoder_layer_prefill(tiny_checkpoint):
    """Three-way check of decoder layer 1 on UT step 1, 128 tokens."""
    check_decoder_layer_prefill(
        tiny_checkpoint,
        WORLD_SIZE,
        DEVICE,
        layer_idx=1,
        ut_step=1,
        seq_len=128,
        block_size=BLOCK_SIZE,
    )


def test_attention_flash_prefill(tiny_checkpoint):
    """Flash-attention prefill of layer 0 on UT step 2, 128 tokens."""
    check_attention_flash_prefill(
        tiny_checkpoint,
        WORLD_SIZE,
        DEVICE,
        layer_idx=0,
        ut_step=2,
        seq_len=128,
        block_size=BLOCK_SIZE,
    )


def test_attention_segmented_prefill(tiny_checkpoint):
    """Segmented prefill of layer 1 on UT step 3, two chunks of 64 tokens."""
    check_attention_segmented_prefill(
        tiny_checkpoint,
        WORLD_SIZE,
        DEVICE,
        layer_idx=1,
        ut_step=3,
        segment_len=64,
        num_chunks=2,
        block_size=BLOCK_SIZE,
    )


def test_attention_decode(tiny_checkpoint):
    """Decode attention of layer 0 on UT step 1, batch 32, context 96, TP 4."""
    check_attention_decode(
        tiny_checkpoint,
        DECODE_WORLD_SIZE,
        DEVICE,
        layer_idx=0,
        ut_step=1,
        batch=DECODE_TOKENS,
        context=96,
        blocks_per_seq=4,
        block_size=BLOCK_SIZE,
    )


@pytest.mark.parametrize("is_prefill", [True, False], ids=["prefill", "decode"])
def test_mlp(tiny_checkpoint, is_prefill):
    """MLP of layer 1: 128 tokens in prefill, 32 tokens in decode."""
    check_mlp(
        tiny_checkpoint,
        WORLD_SIZE,
        DEVICE,
        layer_idx=1,
        num_tokens=128 if is_prefill else DECODE_TOKENS,
        is_prefill=is_prefill,
    )


@pytest.mark.parametrize("threshold", [1.0, 0.5, None])
def test_exit_select(tiny_checkpoint, threshold):
    """The selected hidden state is bitwise equal to the reference."""
    gate_weight, gate_bias = load_gate(tiny_checkpoint)
    check_exit_select(gate_weight, gate_bias, TINY_CONFIG["total_ut_steps"], threshold)


def test_kv_spec_and_binding(tiny_checkpoint):
    """8 KV layers (2 layers x 4 UT steps), the binding and the owners."""
    check_kv_spec_and_binding(tiny_checkpoint, WORLD_SIZE)


def test_full_weight_load(tiny_checkpoint):
    """load_weights gives the hand-sliced TP shards of each parameter."""
    check_full_weight_load(tiny_checkpoint, WORLD_SIZE)
