# SPDX-License-Identifier: Apache-2.0
"""Tests of attention_decode_block_kv against a float64 reference.

Without NKI_SIMULATOR=1 the torch implementation runs. With NKI_SIMULATOR=1 and
VLLM_NEURON_CPU_MODE=1 the nkilib attention_tkg kernel runs in the NKI simulator:
    NKI_SIMULATOR=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 VLLM_NEURON_CPU_MODE=1 \
        pytest test/vllm_neuron/functional/attention/test_attention_decode_block_kv.py
"""

import os

import pytest
import torch

from vllm_neuron.envs import get_compile_backend_name
from vllm_neuron.functional.attention.attention_decode_block_kv import (
    attention_decode_block_kv,
    can_use_attention_decode_block_kv,
    kernel_block_table,
)

SIM = os.environ.get("NKI_SIMULATOR") == "1"


def _inputs(B, H, d, block_len, nb, seed=0, inactive_tail=0, nan_tail=False):
    """Return the op inputs with the new token already in the cache, like the model."""
    torch.manual_seed(seed)
    bf = torch.bfloat16
    num_blocks = B * nb + 1
    q = torch.randn(B, H, d).to(bf)
    k_new = torch.randn(B, d).to(bf)
    v_new = torch.randn(B, d).to(bf)
    k_cache = torch.randn(num_blocks, 1, block_len, d).to(bf)
    v_cache = torch.randn(num_blocks, 1, block_len, d).to(bf)
    # Block 0 is the null block. A real block ID starts at 1.
    table = (1 + torch.randperm(B * nb)).reshape(B, nb)
    hi = (nb - inactive_tail) * block_len - 1
    positions = torch.randint(1, hi, (B,))
    if inactive_tail:
        table[:, nb - inactive_tail :] = 0
    for b in range(B):
        p = int(positions[b])
        blk = int(table[b, p // block_len])
        k_cache[blk, 0, p % block_len] = k_new[b]
        v_cache[blk, 0, p % block_len] = v_new[b]
    if nan_tail:
        # The null block and the unused blocks hold NaN. The kernel must not read them.
        k_cache[0] = float("nan")
        v_cache[0] = float("nan")
        for b in range(B):
            for blk in table[b, (int(positions[b]) // block_len) + 1 :]:
                k_cache[int(blk)] = float("nan")
                v_cache[int(blk)] = float("nan")
    return q, k_new, v_new, k_cache, v_cache, table, positions


def _reference(q, k_cache, v_cache, table, positions):
    B, H, d = q.shape
    block_len = k_cache.shape[-2]
    out = torch.empty(B, H * d, dtype=torch.float64)
    for b in range(B):
        ids = table[b].clamp_min(0).long()
        n = int(positions[b]) + 1
        k = k_cache[ids].double().reshape(-1, d)[:n]
        v = v_cache[ids].double().reshape(-1, d)[:n]
        w = torch.softmax(q[b].double() @ k.t() * d**-0.5, dim=-1)
        out[b] = (w @ v).reshape(-1)
    return out


def _rel_rms(x, ref):
    return ((x.double() - ref).norm() / ref.norm()).item()


def _run(*args):
    compiled = torch.compile(
        attention_decode_block_kv, backend=get_compile_backend_name(), fullgraph=True
    )
    return compiled(*args)


def test_kernel_block_table_marks_dead_blocks_and_pads():
    table = torch.tensor([[3, 5, 7, 0, 0], [2, 4, 6, 8, 9]])
    positions = torch.tensor([70, 300])
    t = kernel_block_table(table, positions, 64)
    # 5 blocks of 64 tokens pad to 8 blocks (2 x 256 tokens).
    assert t.shape == (2, 8) and t.dtype == torch.int32
    # Row 0: block 2 starts at 128 > 70. Row 1: block 4 starts at 256 <= 300.
    assert t[0].tolist() == [3, 5, -1, -1, -1, -1, -1, -1]
    assert t[1].tolist() == [2, 4, 6, 8, 9, -1, -1, -1]


@pytest.mark.parametrize("d", (128, 256))
@pytest.mark.parametrize("B,H", ((1, 3), (2, 6)))
def test_matches_reference(B, H, d):
    q, k_new, v_new, k_cache, v_cache, table, positions = _inputs(B, H, d, 64, 4)
    out = _run(q, k_new, v_new, k_cache, v_cache, table, positions, d**-0.5)
    assert out.shape == (B, H * d)
    assert (
        _rel_rms(out.float(), _reference(q, k_cache, v_cache, table, positions)) < 1e-2
    )


@pytest.mark.skipif(
    SIM, reason="The kernel needs a finite cache. The model writes only finite values."
)
def test_torch_path_ignores_nan_in_unused_blocks():
    q, k_new, v_new, k_cache, v_cache, table, positions = _inputs(
        2, 6, 256, 64, 8, seed=3, inactive_tail=3, nan_tail=True
    )
    out = _run(q, k_new, v_new, k_cache, v_cache, table, positions, 256**-0.5)
    assert torch.isfinite(out).all()
    ref = _reference(q, k_cache, v_cache, table, positions)
    assert _rel_rms(out.float(), ref) < 1e-2


@pytest.mark.skipif(
    not SIM, reason="Set NKI_SIMULATOR=1 to check the kernel selection."
)
def test_kernel_runs_in_simulator():
    q, _, _, k_cache, _, _, _ = _inputs(2, 6, 256, 64, 4)
    assert can_use_attention_decode_block_kv(q, k_cache)


def test_rejects_unsupported_shapes():
    q = torch.randn(2, 6, 192).to(torch.bfloat16)
    assert not can_use_attention_decode_block_kv(
        q, torch.randn(9, 1, 64, 192).to(torch.bfloat16)
    )
    q = torch.randn(2, 6, 128).to(torch.bfloat16)
    assert not can_use_attention_decode_block_kv(
        q, torch.randn(9, 2, 64, 128).to(torch.bfloat16)
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "B,H,block_len,nb",
    ((6, 6, 64, 126), (1, 3, 128, 512)),
    ids=("tp4_ctx8064", "tp8_ctx65536"),
)
def test_matches_reference_at_model_shapes(B, H, block_len, nb):
    q, k_new, v_new, k_cache, v_cache, table, positions = _inputs(
        B, H, 256, block_len, nb, seed=7
    )
    out = _run(q, k_new, v_new, k_cache, v_cache, table, positions, 256**-0.5)
    assert (
        _rel_rms(out.float(), _reference(q, k_cache, v_cache, table, positions)) < 1e-2
    )
