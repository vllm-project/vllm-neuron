# SPDX-License-Identifier: Apache-2.0
"""StableLM device component tests (Neuron, TP 8, one node).

Each test compares a port module on Neuron at TP 8 with the HF transformers
module that holds the checkpoint weights of layer 0 (three-way: FP32 HF, BF16
HF, BF16 port), as ``docs/design/accuracy/module_test_guidelines.md`` in the
repository describes. The model is text only.

- Attention prefill: ``NF.qkv_proj``, the partial RoPE, the paged KV write,
  ``NF.flash_attention`` and ``NF.o_proj``.
- Attention prefill to decode: the prefill fills the paged KV cache. One
  decode step then reads it with ``NF.qkv_proj``, the torch partial RoPE, the
  paged KV write, torch paged attention over the block table and a torch FP32
  O projection with an FP32 all-reduce. The decode megakernel is not used,
  because its RoPE stage refuses d_head 80.
- MLP: ``NF.mlp`` for a prefill shape and a decode shape.

The same cases run in CPU mode on a tiny checkpoint in
``test/unit/test_components.py``.

The checkpoint comes from ``STABLELM_CHECKPOINT_PATH``, or the pinned revision
is downloaded from the HuggingFace hub. Run with 8 NeuronCores:
    pytest contrib/models/stablelm/test/module/ -v -s --timeout=3600
"""

import pytest

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import assert_close_three_way

from stablelm_test_utils import (
    DEVICE_TP,
    resolve_checkpoint,
    run_attention_prefill,
    run_attention_prefill_to_decode,
    run_mlp,
)

pytestmark = pytest.mark.skipif(
    envs.VLLM_NEURON_CPU_MODE,
    reason="Device test. test/unit/test_components.py has the CPU cases.",
)

SEED = 0


@pytest.fixture(scope="module")
def checkpoint() -> str:
    return resolve_checkpoint()


@pytest.mark.usefixtures("dist_cleanup")
@pytest.mark.parametrize("seq_len", [1024])
def test_attention_prefill(checkpoint, seq_len: int, request):
    """Attention prefill on Neuron at TP 8."""
    baseline, expected, actual = run_attention_prefill(
        checkpoint, DEVICE_TP, seq_len, SEED
    )
    assert_close_three_way(
        baseline=baseline, expected=expected, actual=actual, name=request.node.name
    )


@pytest.mark.usefixtures("dist_cleanup")
@pytest.mark.parametrize("prefill_len", [1024])
def test_attention_prefill_to_decode(checkpoint, prefill_len: int, request):
    """A prefill fills the paged KV cache. One decode step then reads it."""
    baseline, expected, actual = run_attention_prefill_to_decode(
        checkpoint, DEVICE_TP, prefill_len, SEED
    )
    assert_close_three_way(
        baseline=[baseline],
        expected=[expected],
        actual=[actual],
        name=request.node.name,
    )


@pytest.mark.usefixtures("dist_cleanup")
@pytest.mark.parametrize(
    "phase, num_tokens",
    [
        pytest.param("prefill", 1024, id="prefill-1024"),
        pytest.param("decode", 4, id="decode-bs4"),
    ],
)
def test_mlp(checkpoint, phase: str, num_tokens: int, request):
    """SwiGLU MLP on Neuron at TP 8 for the prefill and the decode shape."""
    baseline, expected, actual = run_mlp(checkpoint, DEVICE_TP, phase, num_tokens, SEED)
    assert_close_three_way(
        baseline=baseline, expected=expected, actual=actual, name=request.node.name
    )
