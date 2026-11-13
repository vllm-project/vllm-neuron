# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the packed FP8 K layout helpers (src/fp8_kv_layout.py).

    VLLM_NEURON_CPU_MODE=1 pytest contrib/models/hy_v3/test/unit/test_fp8_kv_layout.py
"""

import itertools

import pytest
import torch
from hy_v3_neuron.fp8_kv_layout import (
    packed_decode_viable,
    swizzle_packed_k,
    unswizzle_packed_k,
)

FP8 = torch.float8_e4m3fn


def test_packed_layout_contract():
    # packed[..., j, d, 0] is position 2j, packed[..., j, d, 1] position 2j + 1.
    blocks, kvh, bl, d = 3, 2, 8, 4
    k = torch.arange(blocks * kvh * bl * d, dtype=torch.float32).view(blocks, kvh, bl, d)
    p = swizzle_packed_k(k)
    assert p.shape == (blocks, kvh, bl // 2, d, 2)
    for j in range(bl // 2):
        assert torch.equal(p[..., j, :, 0], k[..., 2 * j, :])
        assert torch.equal(p[..., j, :, 1], k[..., 2 * j + 1, :])


@pytest.mark.parametrize("lead", [(), (5,), (7, 1)])
def test_swizzle_roundtrip_is_exact(lead):
    k = (torch.randn(*lead, 32, 128) * 10).to(FP8)
    p = swizzle_packed_k(k)
    assert p.dtype == FP8
    assert torch.equal(unswizzle_packed_k(p).view(torch.uint8), k.view(torch.uint8))


def test_bad_shapes_rejected():
    with pytest.raises(ValueError, match="even"):
        swizzle_packed_k(torch.zeros(4, 31, 8))
    with pytest.raises(ValueError, match="last dim"):
        unswizzle_packed_k(torch.zeros(4, 16, 8, 3))


@pytest.mark.parametrize("lnc, smallest_viable", [("1", 256), ("2", 512)])
def test_small_contexts_are_not_viable(monkeypatch, lnc, smallest_viable):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    # bs=1 shards s_prior over lnc programs of 128 partitions: the kernel's block
    # is gcd(32, s_prior / (lnc * 128)), which needs s_prior >= 2 * lnc * 128.
    assert packed_decode_viable(32, 1, 8, 1, smallest_viable)
    assert not packed_decode_viable(32, 1, 8, 1, smallest_viable // 2)
    assert packed_decode_viable(32, 1, 8, 1, 11264)  # 90112 / DCP8
    assert not packed_decode_viable(32, 1, 8, 1, 0)


@pytest.mark.parametrize("lnc", ["1", "2"])
def test_matches_the_plugin_helper(monkeypatch, lnc):
    """Drift guard: same answer as vllm_neuron's helper on a grid of buckets."""
    attention_decode = pytest.importorskip("vllm_neuron.functional.attention.attention_decode")
    ref = getattr(attention_decode, "packed_fp8_viable_for_bucket", None)
    if ref is None:
        pytest.skip("vllm_neuron has no packed_fp8_viable_for_bucket")
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    grid = itertools.product(
        [16, 32, 64],  # block_len
        [1, 2, 3, 4, 8, 16, 32],  # bs
        [1, 4, 8, 16],  # q_head
        [1, 2, 4],  # s_active
        [128, 256, 384, 512, 1024, 2048, 4096, 11264, 16384, 65536],  # s_prior
    )
    for args in grid:
        assert packed_decode_viable(*args) == ref(*args), args
