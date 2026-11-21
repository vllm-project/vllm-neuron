# SPDX-License-Identifier: Apache-2.0
"""Segmented prefill attention at head_dim 256 (Qwen3.5) runs the vendored kernel.

For head_dim 256 the attention core cuts a segment of more than 4096 keys into
sections of 4096 keys. The nkilib kernel used only the first section, so it
ignored the keys after position 4096 of an active Q chunk, of a full prior
segment and of a partial prior segment. The vendored copy
(vllm_neuron/functional/vendored_kernels/attention_segmented_cte) fixes this, and
segmented_attention takes it for 128 < head_dim <= 256 only.

The kernel cases need the NKI simulator (VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1)
or a Neuron device. On CPU without the simulator they check the torch fallback.
"""

import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.accuracy.testing import assert_close
from vllm_neuron.functional.attention import attention_segmented_cte as seg
from vllm_neuron.functional.attention.attention_segmented_cte import (
    _torch_segmented_attention_impl,
    segmented_attention,
)

DTYPE = torch.bfloat16
DEVICE = "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron:0"
_slow = pytest.mark.slow


def _golden(q, k_cache, v_cache, block_tables, prior_tokens, block_size, scale):
    """Eager fp32 run of the torch fallback, the reference for the kernel."""
    out = _torch_segmented_attention_impl(
        q=q.float(),
        k_cache=k_cache.float(),
        v_cache=v_cache.float(),
        block_tables=block_tables,
        prior_tokens=torch.tensor([[prior_tokens]], dtype=torch.int32),
        block_size=block_size,
        kv_segment_size=q.shape[1],  # unused by the fallback
        scale=scale,
        tp_q=True,
        tp_out=False,
    )
    return out.to(DTYPE)


def _block_cache(num_blocks, block_size, head_dim):
    shape = (num_blocks, 1, block_size, head_dim)
    k_cache = torch.rand(shape, dtype=DTYPE) * 2 - 1
    v_cache = torch.rand(shape, dtype=DTYPE) * 2 - 1
    return k_cache, v_cache


def _block_tables(num_blocks_needed, max_blocks_per_seq):
    block_tables = torch.full((1, max_blocks_per_seq), -1, dtype=torch.int32)
    block_tables[0, :num_blocks_needed] = torch.arange(
        num_blocks_needed, dtype=torch.int32
    )
    return block_tables


_hd256_params = [
    # Full prior segment of 8192 keys: 2 sections.
    pytest.param(512, 8192, 8192, 64, 2, id="hd256-q512-seg8192-prior8192"),
    # Partial prior segment of 6144 keys: 2 sections.
    pytest.param(512, 8192, 6144, 64, 2, id="hd256-q512-seg8192-prior6144"),
    pytest.param(512, 4096, 4096, 64, 2, id="hd256-q512-seg4096-prior4096"),
    # Active Q chunk of 8192 keys: 2 sections.
    pytest.param(8192, 8192, 0, 64, 2, id="hd256-q8192-seg8192-prior0", marks=_slow),
    pytest.param(8192, 4096, 0, 64, 2, id="hd256-q8192-seg4096-prior0", marks=_slow),
    pytest.param(
        8192, 8192, 6144, 128, 2, id="hd256-q8192-seg8192-prior6144-bs128", marks=_slow
    ),
    # 2 full prior segments and a partial prior segment.
    pytest.param(
        2048, 8192, 22528, 64, 2, id="hd256-q2048-seg8192-prior22528", marks=_slow
    ),
    # The TP=4 shape of Qwen3.8-27B: 24 q heads and 4 KV heads -> 6 q heads per KV head.
    pytest.param(
        8192, 8192, 8192, 64, 6, id="hd256-q8192-seg8192-prior8192-q6kv1", marks=_slow
    ),
]


@pytest.mark.parametrize(
    "seqlen_q,kv_segment_size,prior_tokens,block_size,num_q_heads", _hd256_params
)
def test_segmented_hd256(
    seqlen_q, kv_segment_size, prior_tokens, block_size, num_q_heads
):
    torch.manual_seed(42)
    head_dim = 256
    # Scale 1.0: with 1 / sqrt(256) the softmax of these random inputs is almost
    # uniform over thousands of keys, the output is near 0, and the tolerance
    # cannot see a missing section.
    scale = 1.0
    num_blocks_needed = -(-(prior_tokens + seqlen_q) // block_size)
    k_cache, v_cache = _block_cache(num_blocks_needed + 50, block_size, head_dim)
    block_tables = _block_tables(num_blocks_needed, num_blocks_needed + 10)
    q = torch.randn((num_q_heads, seqlen_q, head_dim), dtype=DTYPE)

    expected = _golden(
        q, k_cache, v_cache, block_tables, prior_tokens, block_size, scale
    )
    compiled = torch.compile(
        segmented_attention, backend=envs.get_compile_backend_name(), fullgraph=True
    )
    result = compiled(
        q=q.to(DEVICE),
        k_cache=k_cache.to(DEVICE),
        v_cache=v_cache.to(DEVICE),
        block_tables=block_tables.to(DEVICE),
        prior_tokens=torch.tensor([[prior_tokens]], dtype=torch.int32).to(DEVICE),
        block_size=block_size,
        kv_segment_size=kv_segment_size,
        scale=scale,
        tp_q=True,
        tp_out=False,
    )
    assert_close(result.cpu(), expected, atol=1e-2, rtol=4e-2)


class _FakeKernel:
    """Stands in for a wrapped NKI kernel and records that it ran."""

    def __init__(self):
        self.calls = 0

    def __getitem__(self, grid):
        def run(**kwargs):
            self.calls += 1
            return torch.zeros_like(kwargs["q"])

        return run


@pytest.mark.parametrize("head_dim, wide", [(64, False), (128, False), (256, True)])
def test_head_dim_selects_the_kernel(monkeypatch, head_dim, wide):
    nkilib, vendored = _FakeKernel(), _FakeKernel()
    monkeypatch.setattr(seg, "_wrapped_attention_segmented_cte", nkilib)
    monkeypatch.setattr(seg, "_wrapped_attention_segmented_cte_wide_head", vendored)
    monkeypatch.setattr(seg, "can_run_kernel", lambda q: True)
    block_size, seqlen_q = 64, 512
    k_cache, v_cache = _block_cache(20, block_size, head_dim)
    seg.segmented_attention(
        q=torch.randn((2, seqlen_q, head_dim), dtype=DTYPE),
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=_block_tables(8, 10),
        prior_tokens=torch.tensor([[0]], dtype=torch.int32),
        block_size=block_size,
        kv_segment_size=512,
        scale=1.0,
        tp_q=True,
        tp_out=False,
    )
    assert (nkilib.calls, vendored.calls) == ((0, 1) if wide else (1, 0))


def test_head_dim_above_256_uses_the_torch_fallback():
    q = torch.randn((2, 512, 320), dtype=DTYPE)
    assert not seg._can_use_segmented_kernel(q, kv_segment_size=512, block_size=64)
