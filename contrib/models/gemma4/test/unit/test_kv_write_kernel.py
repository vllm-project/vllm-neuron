# SPDX-License-Identifier: Apache-2.0
"""NKI KV-cache kernels (row write, block gather) against torch references, on the NKI CPU simulator.

Run with ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``; skipped otherwise.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("NKI_SIMULATOR") != "1", reason="needs NKI_SIMULATOR=1"
)


def _cache(n_blocks, heads, bs, dh, seed=0):
    torch.manual_seed(seed)
    planes = torch.randn(2, n_blocks, heads, bs, dh).bfloat16()
    return planes, planes.reshape(-1).clone(), (n_blocks, heads, bs, dh)


def _reference_write(planes, k_rows, v_rows, blk, pos, heads):
    ref = planes.clone()
    t = blk.shape[0]
    h_idx = torch.arange(heads).repeat_interleave(t)
    for plane, rows in ((0, k_rows), (1, v_rows)):
        ref[plane].index_put_((blk.repeat(heads), h_idx, pos.repeat(heads)), rows)
    return ref


@pytest.mark.parametrize("tokens", [3, 70])  # 6 rows (1 tile) and 140 rows (2 tiles) per plane
def test_write_matches_index_put(tokens):
    from gemma4_neuron.kv_write import write_kv

    n_blocks, heads, bs, dh = 12, 2, 32, 256
    planes, flat, dims = _cache(n_blocks, heads, bs, dh)
    slots = torch.randperm(n_blocks * bs)[:tokens]
    blk, pos = slots // bs, slots % bs
    k_rows = torch.randn(heads * tokens, dh).bfloat16()
    v_rows = torch.randn(heads * tokens, dh).bfloat16()

    token = write_kv(flat, k_rows, v_rows, blk, pos, heads, dims)
    assert torch.equal(flat.view_as(planes), _reference_write(planes, k_rows, v_rows, blk, pos, heads))
    assert token.shape == (128, 1) and not token.any()


def test_padding_slots_never_touch_real_blocks():
    """A padded decode row (slot -1 -> null block 0 through paged_slot_indices) cannot corrupt the LAST block."""
    from gemma4_neuron.kv_write import write_kv
    from vllm_neuron.model.kv_cache import paged_slot_indices

    n_blocks, heads, bs, dh = 6, 2, 32, 256
    planes, flat, dims = _cache(n_blocks, heads, bs, dh)
    slots = torch.tensor([70, -1])
    blk, pos = paged_slot_indices(slots, bs, n_blocks)
    rows = torch.randn(heads * 2, dh).bfloat16()
    write_kv(flat, rows, rows, blk, pos, heads, dims)
    out = flat.view_as(planes)
    assert not torch.equal(out[:, 2], planes[:, 2])  # slot 70 -> block 2 was written
    assert torch.equal(out[:, 3:], planes[:, 3:])    # blocks >= 3, including the last one, untouched


@pytest.mark.parametrize("heads, bs, dh, nb", [(4, 32, 256, 5), (1, 64, 512, 40)])
def test_gather_matches_index_select(heads, bs, dh, nb):
    from gemma4_neuron.kv_write import gather_kv

    planes, flat, dims = _cache(64, heads, bs, dh)
    ids = torch.randint(0, 64, (nb,))
    k_blk, v_blk = gather_kv(flat, ids, dims)
    assert torch.equal(k_blk, planes[0][ids]) and torch.equal(v_blk, planes[1][ids])


def test_write_then_gather_sees_the_write():
    from gemma4_neuron.kv_write import gather_kv, write_kv

    n_blocks, heads, bs, dh, tokens = 10, 2, 32, 256, 4
    planes, flat, dims = _cache(n_blocks, heads, bs, dh)
    slots = torch.randperm(n_blocks * bs)[:tokens]
    blk, pos = slots // bs, slots % bs
    k_rows = torch.randn(heads * tokens, dh).bfloat16()
    v_rows = torch.randn(heads * tokens, dh).bfloat16()
    ref = _reference_write(planes, k_rows, v_rows, blk, pos, heads)

    token = write_kv(flat, k_rows, v_rows, blk, pos, heads, dims)
    ids = torch.tensor([2, 9, 2])
    k_blk, v_blk = gather_kv(flat, ids, dims, after=token)
    assert torch.equal(k_blk, ref[0][ids]) and torch.equal(v_blk, ref[1][ids])
