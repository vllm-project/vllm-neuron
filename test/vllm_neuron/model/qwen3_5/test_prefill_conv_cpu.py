# SPDX-License-Identifier: Apache-2.0
"""The prefill conv of the linear attention layers gives the values of F.conv1d.

The prefill step runs the depthwise causal conv as K shifted multiply-adds
(_causal_depthwise_conv), because the native route miscompiles F.conv1d in the
8192-token prefill graph (see the docstring of _causal_depthwise_conv). These tests
check the values on CPU in the token-major layout of the QKV projection: the chunk
with its left context, the window across the first K - 1 tokens, and the dtype.
"""

import os

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

import pytest
import torch
import torch.nn.functional as F

from vllm_neuron.model.qwen3_5.model_bf16 import _causal_depthwise_conv

CHANNELS, KERNEL = 96, 4


def _inputs(tokens, dtype, zero_context=False):
    gen = torch.Generator().manual_seed(tokens)
    conv_in = torch.randn(KERNEL - 1 + tokens, CHANNELS, generator=gen)
    if zero_context:
        conv_in[: KERNEL - 1] = 0
    weight = torch.randn(CHANNELS, 1, KERNEL, generator=gen) * 0.5
    return conv_in.to(dtype), weight.to(dtype)


def _reference(conv_in, weight):
    """F.conv1d over the channels-first transpose, back in the token-major layout."""
    channels_first = conv_in.float().transpose(0, 1).unsqueeze(0)
    out = F.conv1d(
        channels_first, weight.float(), bias=None, padding=0, groups=CHANNELS
    )
    return out.squeeze(0).transpose(0, 1)


@pytest.mark.parametrize("tokens", [1, 3, 4, 256, 4099])
@pytest.mark.parametrize("zero_context", [False, True])
def test_float32_matches_conv1d(tokens, zero_context):
    conv_in, weight = _inputs(tokens, torch.float32, zero_context)
    out = _causal_depthwise_conv(conv_in, weight)
    assert out.shape == (tokens, CHANNELS)
    torch.testing.assert_close(out, _reference(conv_in, weight), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("tokens", [16, 4099])
def test_bfloat16_rounds_the_float32_sum(tokens):
    conv_in, weight = _inputs(tokens, torch.bfloat16)
    out = _causal_depthwise_conv(conv_in, weight)
    assert out.dtype == torch.bfloat16
    # The float32 sum of the bf16 inputs, rounded once to bf16: within half a bf16
    # step (2**-9 relative) of the float32 reference.
    torch.testing.assert_close(
        out.float(), _reference(conv_in, weight), atol=1e-6, rtol=2**-8
    )


def test_the_left_context_feeds_the_first_tokens():
    conv_in, weight = _inputs(8, torch.float32)
    with_context = _causal_depthwise_conv(conv_in, weight)
    conv_in[: KERNEL - 1] = 0
    without_context = _causal_depthwise_conv(conv_in, weight)
    # The window of token t covers t - 3 .. t, so tokens 0, 1 and 2 see the context.
    assert not torch.equal(with_context[: KERNEL - 1], without_context[: KERNEL - 1])
    assert torch.equal(with_context[KERNEL - 1 :], without_context[KERNEL - 1 :])
