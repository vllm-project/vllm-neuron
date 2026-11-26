# SPDX-License-Identifier: Apache-2.0
"""The NKI decode kernels in the NKI CPU simulator (``nki.simulate``) against ``ops.py``, at TP8 per-rank shapes.

Needs the ``nki`` package (Neuron images); skipped elsewhere. The model runs one rank per physical core (LNC 1);
the simulator follows ``NEURON_LOGICAL_NC_CONFIG``, which is set before ``nki`` loads (at LNC 2 a kernel runs on two
shards, and the grouped prefill MoE's scatter-add would add its output twice).
"""

import os

os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"

import pytest  # noqa: E402
import torch  # noqa: E402

nki = pytest.importorskip("nki")

from inkling2_neuron import ops  # noqa: E402
from inkling2_neuron.kernels import attn_decode as K4  # noqa: E402
from inkling2_neuron.kernels import moe_decode_hw as K2  # noqa: E402

H, I, E, K = 2304, 144, 16, 8


def _moe_inputs(T, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(T, H, generator=g) * 0.5).bfloat16()
    gate_up = (torch.randn(E, H, 2 * I, generator=g) * 0.03).bfloat16()
    down = (torch.randn(E, I, H, generator=g) * 0.03).bfloat16()
    idx = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)])
    w = torch.rand(T, K, generator=g)
    return x, idx, w, gate_up, down


@pytest.mark.parametrize("T", [1, 2])
def test_moe_decode_hw_matches_gather(T):
    x, idx, w, gate_up, down = _moe_inputs(T)
    ref = ops.moe_gather(x, idx, w, gate_up, down, 4.0, 25.0)
    out = torch.as_tensor(nki.simulate(K2.inkling2_moe_decode_hw)(
        x, idx.reshape(1, -1).to(torch.int32), w.reshape(1, -1).float(), gate_up, down)).float()
    # the kernel rounds the routing-weighted activation to bf16 before the down projection
    assert (out - ref).abs().max() <= 1e-2 * ref.abs().max()


def test_moe_decode_hw_all_padding_is_zero():
    """Warmup decodes rows that are all padding: every expert load is skipped and the output must stay finite (it
    reaches the null block's caches)."""
    x, idx, w, gate_up, down = _moe_inputs(2, seed=4)
    pad = torch.full_like(idx, E)
    out = torch.as_tensor(nki.simulate(K2.inkling2_moe_decode_hw)(
        x, pad.reshape(1, -1).to(torch.int32), torch.zeros(1, 2 * K), gate_up, down)).float()
    assert torch.equal(out, torch.zeros_like(out))


def test_moe_decode_hw_skips_padding_rows():
    x, idx, w, gate_up, down = _moe_inputs(2, seed=1)
    ref = ops.moe_gather(x, idx, w, gate_up, down, 4.0, 25.0)
    pad = idx.clone()
    pad[1] = E                                      # a padding row: every expert id out of range
    out = torch.as_tensor(nki.simulate(K2.inkling2_moe_decode_hw)(
        x, pad.reshape(1, -1).to(torch.int32), w.reshape(1, -1).float(), gate_up, down)).float()
    assert (out[0] - ref[0]).abs().max() <= 1e-2 * ref[0].abs().max()
    assert torch.isfinite(out[1]).all()


@pytest.mark.parametrize("L,window,q_pos,kv0", [(1280, 1024, [3000, 1100], [2048, 256]),
                                                (1280, 1024, [5, 700], [0, 0]),
                                                (2048, None, [2047, 900], [0, 0])])
def test_attn_decode_matches_core(L, window, q_pos, kv0):
    g = torch.Generator().manual_seed(2)
    B, hq, d, ext = 2, 3, 128, 1024
    q = (torch.randn(B, hq, d, generator=g) * 0.3).bfloat16()
    k = (torch.randn(B, L, d, generator=g) * 0.3).bfloat16()
    v = torch.randn(B, L, d, generator=g).bfloat16()
    rel = torch.randn(B, hq, ext, generator=g).bfloat16()
    sink = torch.randn(B, hq, generator=g)
    qp, ks = torch.tensor(q_pos), torch.tensor(kv0)
    kv_pos = ks.view(-1, 1) + torch.arange(L).view(1, -1)
    ref = ops.attention_core(q.unsqueeze(2), k.unsqueeze(1), v.unsqueeze(1), rel.unsqueeze(2), sink.unsqueeze(-1),
                             qp.view(-1, 1), kv_pos, window, ext).squeeze(2).float()
    relp, start, bounds = K4.prepare(rel, qp, ks, L, window)
    out = torch.as_tensor(nki.simulate(K4.inkling2_attn_decode)(q, k, v, relp, start, bounds, sink.float())).float()
    assert (out - ref).abs().max() <= 1e-2 * ref.abs().max()


@pytest.mark.parametrize("T,E_", [(64, 16), (256, 128)])
def test_moe_prefill_grouped_matches_dense(T, E_):
    from inkling2_neuron.kernels import moe_prefill_grouped as K5

    g = torch.Generator().manual_seed(3)
    x = (torch.randn(T, H, generator=g) * 0.5).bfloat16()
    gate_up = (torch.randn(E_, H, 2 * I, generator=g) * 0.03).bfloat16()
    down = (torch.randn(E_, I, H, generator=g) * 0.03).bfloat16()
    idx = torch.stack([torch.randperm(E_, generator=g)[:K] for _ in range(T)])
    w = torch.rand(T, K, generator=g)
    ref = ops.moe_dense(x, idx, w, gate_up, down, 4.0, 25.0, group=min(E_, 32))
    tok, wt, nt = K5.tables(idx, w, E_)
    out = torch.as_tensor(nki.simulate(K5.inkling2_moe_prefill)(x, tok, wt, nt, gate_up, down)).float()
    assert (out - ref).abs().max() <= 1e-2 * ref.abs().max()
