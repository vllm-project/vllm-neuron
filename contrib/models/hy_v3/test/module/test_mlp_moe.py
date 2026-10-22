# SPDX-License-Identifier: Apache-2.0
"""Module tests: Hy3 static-FP8 MLP and routed experts on one NeuronCore.

Uses layer 0 (dense MLP) and layer 1 (MoE) of the real checkpoint, sharded as
rank 0 of the TP=64 / EP=64 deployment, loaded through the model's weight
loaders, and checks against an FP32 fake-quant reference.

    HY3_CKPT=/path/to/Hy3-FP8 pytest contrib/models/hy_v3/test/module/ -v -s
"""

import types

import pytest
import torch
from conftest import REL_TOL, compile_and_run, dequant, fake_quant, rel_err, rmsnorm
from hy_v3_neuron import model as hy_model
from hy_v3_neuron.config import HYV3Config
from hy_v3_neuron.weight_loaders import (
    expert_down_loader,
    expert_gate_up_loader,
    expert_scale_loader,
)
from nkilib.core.mlp.mlp_parameters import TKG_BS_SEQLEN_THRESHOLD

from vllm_neuron.utils.weight_loader import get_weight_loader

H, TP, EP, E, K = 4096, 64, 64, 192, 8
L = E // EP  # local experts on rank 0
EPS = 1e-5


@pytest.fixture
def single_rank_tp(monkeypatch):
    """Build modules as a single rank: no TP collectives in their forward."""
    group = types.SimpleNamespace(world_size=1, rank_in_group=0)
    monkeypatch.setattr(hy_model, "get_tp_group", lambda: group)


def _load(param: torch.nn.Parameter, slices) -> torch.Tensor:
    return get_weight_loader(param).transform(slices, 0)


def _build_mlp(ckpt, src: str, intermediate: int, num_shards: int):
    """``HYV3StaticFP8MLP`` holding shard 0 of ``num_shards`` of ``src``."""
    mlp = hy_model.HYV3StaticFP8MLP(
        HYV3Config(),
        intermediate,
        tp_sharded=False,
        num_shards=num_shards,
        shard_rank=0,
    )
    keys = {}
    for n in ("gate", "up", "down"):
        keys[f"{n}_proj_weight"] = [f"{src}.{n}_proj.weight"]
        keys[f"{n}_weight_scale"] = [f"{src}.{n}_proj.weight_scale"]
    keys["gate_up_input_scale"] = [
        f"{src}.gate_proj.input_scale",
        f"{src}.up_proj.input_scale",
    ]
    keys["down_input_scale"] = [f"{src}.down_proj.input_scale"]
    for name, ks in keys.items():
        param = getattr(mlp, name)
        param.data = _load(param, [ckpt.slice(k) for k in ks])
    return mlp


def _mlp_reference(ckpt, src, shard, x, ln_w, quant=True):
    """FP32 reference with dequantized weights; ``quant`` fake-quantizes the
    activations with the (doubled) static input scales like the CTE kernel."""
    g = ckpt.get(f"{src}.gate_proj.weight")[:shard]
    u = ckpt.get(f"{src}.up_proj.weight")[:shard]
    d = ckpt.get(f"{src}.down_proj.weight")[:, :shard]
    sg, su, sd = (
        ckpt.scalar(f"{src}.{n}_proj.weight_scale") for n in ("gate", "up", "down")
    )
    # Activation scales are doubled at load time (see weight_loaders.py).
    s_in = ckpt.scalar(f"{src}.gate_proj.input_scale") * 2
    s_d_in = ckpt.scalar(f"{src}.down_proj.input_scale") * 2
    xn = rmsnorm(x, ln_w).to(torch.bfloat16).float()
    xq = fake_quant(xn, s_in) if quant else xn
    inter = torch.nn.functional.silu(xq @ dequant(g, sg).T) * (xq @ dequant(u, su).T)
    return (fake_quant(inter, s_d_in) if quant else inter) @ dequant(d, sd).T


class _MLPWrapper(torch.nn.Module):
    def __init__(self, mlp, ln_w):
        super().__init__()
        self.mlp = mlp
        self.register_buffer("ln_w", ln_w)

    def forward(self, x):
        return self.mlp(x, self.ln_w, EPS, is_prefill=True)


# (label, checkpoint prefix, layernorm, full intermediate, shards, tokens)
MLP_CASES = [
    # Dense layer, TP=64: I=208 per rank, zero-padded to 256.
    ("dense_cte", "model.layers.0.mlp", 0, 13312, TP, 256),
    ("dense_tkg", "model.layers.0.mlp", 0, 13312, TP, 8),
    # Shared expert: full weights (prefill) and the 8-way decode copy.
    ("shared_cte", "model.layers.1.mlp.shared_mlp", 1, 1536, 1, 256),
    ("shared_decode_tkg", "model.layers.1.mlp.shared_mlp", 1, 1536, 8, 8),
    ("shared_decode_cte", "model.layers.1.mlp.shared_mlp", 1, 1536, 8, 128),
]


@pytest.mark.parametrize(
    "label, src, layer, intermediate, shards, tokens",
    MLP_CASES,
    ids=[c[0] for c in MLP_CASES],
)
def test_static_fp8_mlp(
    ckpt, single_rank_tp, label, src, layer, intermediate, shards, tokens
):
    torch.manual_seed(0)
    mlp = _build_mlp(ckpt, src, intermediate, shards)
    ln_w = ckpt.get(f"model.layers.{layer}.post_attention_layernorm.weight")
    x = (torch.randn(tokens, H) * 2.0).to(torch.bfloat16)

    out = compile_and_run(_MLPWrapper(mlp, ln_w), x)
    exact = _mlp_reference(ckpt, src, mlp.intermediate_shard, x, ln_w, quant=False)
    err = rel_err(out, exact)
    if tokens > TKG_BS_SEQLEN_THRESHOLD:
        # CTE path: activations are quantized to FP8 (3 mantissa bits). The
        # kernel does not round bit-identically to torch, so it is held to the
        # error an ideal FP8 fake-quant reference has vs the exact result.
        fq_err = rel_err(
            _mlp_reference(ckpt, src, mlp.intermediate_shard, x, ln_w), exact
        )
        bound = 1.25 * fq_err + 5e-3
    else:
        # TKG path: BF16 activations x FP8 weights.
        fq_err, bound = float("nan"), REL_TOL
    print(
        f"{label}: I_shard={mlp.intermediate_shard} rel_err={err:.3e} "
        f"(ideal fp8 fake-quant {fq_err:.3e}, bound {bound:.3e})"
    )
    assert err < bound


def _expert_tensors(ckpt, layer: int):
    """Rank-0 routed expert weights through the model's loaders (EP=64)."""
    p = f"model.layers.{layer}.mlp.experts"

    def slices(proj, suffix):
        return [ckpt.slice(f"{p}.{e}.{proj}.{suffix}") for e in range(L)]

    gu_s = slices("gate_proj", "weight_scale") + slices("up_proj", "weight_scale")
    d_s = slices("down_proj", "weight_scale")
    I = 1536
    return {
        "gate_up": expert_gate_up_loader(L, I, 0).transform(
            slices("gate_proj", "weight") + slices("up_proj", "weight"), 0
        ),
        "down": expert_down_loader(L, I, 0).transform(slices("down_proj", "weight"), 0),
        "gate_up_scale": expert_scale_loader(
            L, 2, lambda s: s.view(L, 2, 1).expand(L, 2, I).reshape(L, 1, 2 * I)
        ).transform(gu_s, 0),
        "down_scale": expert_scale_loader(
            L, 1, lambda s: s.view(L, 1, 1).expand(L, 1, H)
        ).transform(d_s, 0),
        "gate_up_scale_tkg": expert_scale_loader(
            L, 2, lambda s: s.view(L, 2, 1)
        ).transform(gu_s, 0),
        "down_scale_tkg": expert_scale_loader(L, 1, lambda s: s.view(L, 1)).transform(
            d_s, 0
        ),
    }


def _route(ckpt, layer, x):
    p = f"model.layers.{layer}.mlp"
    return hy_model.hy3_route(
        x,
        ckpt.get(f"{p}.router.gate.weight").float(),
        ckpt.get(f"{p}.expert_bias").float(),
        K,
        True,
        2.826,
    )


def _experts_reference(ckpt, layer, x, aff):
    p = f"model.layers.{layer}.mlp.experts"
    out = torch.zeros(x.shape[0], H)
    for e in range(L):
        w = {
            n: dequant(
                ckpt.get(f"{p}.{e}.{n}_proj.weight"),
                ckpt.scalar(f"{p}.{e}.{n}_proj.weight_scale"),
            )
            for n in ("gate", "up", "down")
        }
        xf = x.float()
        y = torch.nn.functional.silu(xf @ w["gate"].T) * (xf @ w["up"].T) @ w["down"].T
        out += y * aff[:, e : e + 1]
    return out


@pytest.mark.parametrize("tokens", [1024, 4096])
def test_routed_experts_cte(ckpt, tokens):
    torch.manual_seed(0)
    layer = 1
    w = _expert_tensors(ckpt, layer)
    ln_w = ckpt.get(f"model.layers.{layer}.post_attention_layernorm.weight").float()
    x = (torch.randn(tokens, H) * ln_w).to(torch.bfloat16)
    aff, _ = _route(ckpt, layer, x)
    # Make sure every local expert gets tokens (the router alone may skip them).
    for e in range(L):
        aff[e::4, e] = 0.5

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for k, v in w.items():
                self.register_buffer(k, v)
            # tp_degree=1 needs only the group's rank tensor.
            self.group = types.SimpleNamespace(rank_in_group=0)

        def forward(self, x, aff):
            return hy_model.routed_experts_cte(
                x,
                aff,
                self.gate_up,
                self.down,
                self.gate_up_scale,
                self.down_scale,
                top_k=K,
                moe_group=self.group,
                tp_degree=1,
            )

    m = M().to("neuron:0")
    m.group.rank_in_group_tensor = torch.zeros(1, dtype=torch.int32, device="neuron:0")
    out = compile_and_run(m, x, aff[:, :L].contiguous())
    err = rel_err(out, _experts_reference(ckpt, layer, x, aff))
    print(f"moe_cte hybrid T={tokens}: rel_err={err:.3e}")
    assert err < REL_TOL


@pytest.mark.parametrize("tokens", [1, 8, 32])
def test_routed_experts_tkg(ckpt, tokens):
    torch.manual_seed(0)
    layer = 1
    w = _expert_tensors(ckpt, layer)
    ln_w = ckpt.get(f"model.layers.{layer}.post_attention_layernorm.weight").float()
    x = (torch.randn(tokens, H) * ln_w).to(torch.bfloat16)
    aff, idx = _route(ckpt, layer, x)
    # Make sure the local experts are hit in a tiny batch.
    aff[:, 0] = 0.5
    aff[:, 2] = 0.25

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for k, v in w.items():
                self.register_buffer(k, v)

        def forward(self, x, aff, idx):
            rank_id = torch.zeros((1, 1), dtype=torch.int32, device=x.device)
            return hy_model.routed_experts_tkg(
                x,
                aff,
                idx,
                rank_id,
                self.gate_up,
                self.down,
                self.gate_up_scale_tkg,
                self.down_scale_tkg,
            )

    out = compile_and_run(M(), x, aff, idx.to(torch.int32))
    err = rel_err(out, _experts_reference(ckpt, layer, x, aff))
    print(f"moe_tkg T={tokens}: rel_err={err:.3e}")
    assert err < REL_TOL


@pytest.mark.parametrize("hits", [0, 1, 2, 3])
def test_routed_experts_tkg_hits(ckpt, hits):
    """One-token decode over only this rank's routed experts (``moe_tkg_hits``)."""
    torch.manual_seed(0)
    layer = 1
    w = _expert_tensors(ckpt, layer)
    ln_w = ckpt.get(f"model.layers.{layer}.post_attention_layernorm.weight").float()
    x = (torch.randn(1, H) * ln_w).to(torch.bfloat16)
    aff, idx = _route(ckpt, layer, x)
    # Rank 0 holds experts 0..L-1: route exactly `hits` slots to them and the
    # rest elsewhere.
    aff[:, :L] = 0
    for j in range(hits):
        idx[0, j] = j
        aff[0, j] = 0.5 - 0.1 * j
    for j in range(hits, K):
        if idx[0, j] < L:
            idx[0, j] = L + j

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for k, v in w.items():
                self.register_buffer(k, v)

        def forward(self, x, aff, idx):
            rank_id = torch.zeros((1, 1), dtype=torch.int32, device=x.device)
            return hy_model.routed_experts_tkg_hits(
                x,
                aff,
                idx,
                rank_id,
                self.gate_up,
                self.down,
                self.gate_up_scale_tkg,
                self.down_scale_tkg,
            )

    # Reverse the slots so the local hits are last: the packing must find them.
    out = compile_and_run(M(), x, aff, idx.flip(1).to(torch.int32))
    if hits == 0:
        assert out.abs().max().item() == 0
        return
    err = rel_err(out, _experts_reference(ckpt, layer, x, aff))
    print(f"moe_tkg_hits hits={hits}: rel_err={err:.3e}")
    assert err < REL_TOL
