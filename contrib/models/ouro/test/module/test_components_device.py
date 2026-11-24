# SPDX-License-Identifier: Apache-2.0
"""Ouro-1.4B prefill and decode component tests on Neuron devices.

Each test compiles one real Ouro module at TP 8 and compares it with the
independent torch reference in ``ouro_reference.py`` (the checks are in
``ouro_component_checks.py``). A three-way check (fp32 reference, bf16
reference, Neuron) applies with the default thresholds.

| Test | Path | Kernels |
|---|---|---|
| ``test_attention_flash_prefill`` | prefill, 1024 tokens | qkv_proj, flash_attention, o_proj |
| ``test_attention_segmented_prefill`` | prefill, 2 x 512 tokens | qkv_proj, segmented_attention, o_proj |
| ``test_mlp_prefill`` | prefill, 1024 tokens | mlp |
| ``test_attention_decode`` | decode, batch 4, context 96 | attention_decode |
| ``test_mlp_decode`` | decode, batch 4 | mlp |

Each attention test runs on one UT step, and checks that the other UT-step
caches stay zero. The flash prefill and the decode tests also check the KV
cache write.

The tests use one node with 8 NeuronCores and the pinned checkpoint
(``OURO_CHECKPOINT_PATH``, or a download of the pinned revision). Run with:
    pytest contrib/models/ouro/test/module/test_components_device.py -v
"""

import pytest

from vllm_neuron import envs

from ouro_component_checks import (
    check_attention_decode,
    check_attention_flash_prefill,
    check_attention_segmented_prefill,
    check_mlp,
)
from ouro_test_utils import resolve_ouro_checkpoint

WORLD_SIZE = 8
DEVICE = "neuron:0"
BLOCK_SIZE = 32
PREFILL_LEN = 1024
SEGMENT_LEN = 512
DECODE_BATCH = 4
DECODE_CONTEXT = 96
DECODE_BLOCKS_PER_SEQ = 4

pytestmark = [
    pytest.mark.usefixtures("dist_cleanup"),
    pytest.mark.skipif(
        bool(envs.VLLM_NEURON_CPU_MODE),
        reason="Device test: needs Neuron hardware (unset VLLM_NEURON_CPU_MODE).",
    ),
]


@pytest.fixture(scope="module")
def model_checkpoint():
    return resolve_ouro_checkpoint()


def test_attention_flash_prefill(model_checkpoint):
    """Flash-attention prefill of layer 0 on UT step 2, 1024 tokens."""
    check_attention_flash_prefill(
        model_checkpoint,
        WORLD_SIZE,
        DEVICE,
        layer_idx=0,
        ut_step=2,
        seq_len=PREFILL_LEN,
        block_size=BLOCK_SIZE,
    )


def test_attention_segmented_prefill(model_checkpoint):
    """Segmented prefill of layer 12 on UT step 3, two chunks of 512 tokens."""
    check_attention_segmented_prefill(
        model_checkpoint,
        WORLD_SIZE,
        DEVICE,
        layer_idx=12,
        ut_step=3,
        segment_len=SEGMENT_LEN,
        num_chunks=2,
        block_size=BLOCK_SIZE,
    )


def test_mlp_prefill(model_checkpoint):
    """MLP prefill of layer 0, 1024 tokens, with SP."""
    check_mlp(
        model_checkpoint,
        WORLD_SIZE,
        DEVICE,
        layer_idx=0,
        num_tokens=PREFILL_LEN,
        is_prefill=True,
    )


def test_attention_decode(model_checkpoint):
    """Decode attention of layer 0 on UT step 1, batch 4, context 96."""
    check_attention_decode(
        model_checkpoint,
        WORLD_SIZE,
        DEVICE,
        layer_idx=0,
        ut_step=1,
        batch=DECODE_BATCH,
        context=DECODE_CONTEXT,
        blocks_per_seq=DECODE_BLOCKS_PER_SEQ,
        block_size=BLOCK_SIZE,
    )


def test_mlp_decode(model_checkpoint):
    """MLP decode of layer 23, batch 4, with the TP all-reduce."""
    check_mlp(
        model_checkpoint,
        WORLD_SIZE,
        DEVICE,
        layer_idx=23,
        num_tokens=DECODE_BATCH,
        is_prefill=False,
    )
