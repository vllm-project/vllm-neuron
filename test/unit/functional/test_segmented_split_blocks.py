# SPDX-License-Identifier: Apache-2.0
"""Sub-block view of large KV blocks for the segmented prefill kernel (CPU).

Runs in a subprocess: importing the real vllm_neuron.functional here would shadow the
vllm / vllm_neuron stubs the qwen3_5 model tests install in the same session.
"""

import subprocess
import sys
import textwrap

import pytest

CHECKS = textwrap.dedent("""
    import pytest, torch
    from vllm_neuron.functional.attention import attention_segmented_cte as seg

    # 1024-token blocks -> 64-token sub-blocks (2-byte K) address the same tokens, as a view.
    nb, bs, d = 6, 1024, 8
    k, v = torch.randn(nb, 1, bs, d), torch.randn(nb, 1, bs, d)
    table = torch.tensor([[3, 1, 5, -1]], dtype=torch.int32)
    k2, v2, t2, bs2 = seg._split_kv_blocks(k, v, table, bs)
    assert bs2 == 64 and t2.shape == (1, 4 * bs // bs2)
    assert k2.data_ptr() == k.data_ptr()
    for c, c2 in ((k, k2), (v, v2)):
        ref = c[table.clamp(min=0).long()[0]].reshape(-1, d)  # -1 -> null block 0
        assert torch.equal(c2[t2[0].long()].reshape(-1, d), ref)

    # Small blocks pass through; several KV heads pass through up to 512 tokens, then fail.
    t0 = torch.zeros(1, 2, dtype=torch.int32)
    k = torch.zeros(4, 1, 32, 8)
    assert seg._split_kv_blocks(k, k, t0, 32)[3] == 32
    k = torch.zeros(4, 2, 256, 8)
    assert seg._split_kv_blocks(k, k, t0, 256)[3] == 256
    k = torch.zeros(4, 2, 1024, 8)
    with pytest.raises(ValueError):
        seg._split_kv_blocks(k, k, t0, 1024)

    # A 1-byte (FP8) K cache prefers 128-token blocks: 128 passes, 512 is split into 4.
    k = torch.zeros(4, 1, 128, 8).to(torch.float8_e4m3fn)
    assert seg._split_kv_blocks(k, k, t0, 128)[3] == 128
    k = torch.zeros(4, 1, 512, 8).to(torch.float8_e4m3fn)
    k2, _, t2, bs2 = seg._split_kv_blocks(k, k, torch.tensor([[2, -1]], dtype=torch.int32), 512)
    assert bs2 == 128 and k2.shape == (16, 1, 128, 8)
    assert t2.tolist() == [[8, 9, 10, 11, 0, 1, 2, 3]]
""")


def test_split_kv_blocks():
    pytest.importorskip("vllm")
    proc = subprocess.run(
        [sys.executable, "-c", CHECKS], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
