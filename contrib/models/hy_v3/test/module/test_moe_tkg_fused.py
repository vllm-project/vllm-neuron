# SPDX-License-Identifier: Apache-2.0
"""Module test: the fused decode MoE block (kernels/moe_tkg_fused.py) on one NeuronCore.

Real layer-1 weights, sharded as rank 0 of TP=64 / EP=64, loaded through the model's weight
loaders. The reference is FP32 with the same static activation quantization (fake-quant), the
torch RMSNorm and the torch router (``hy3_route``).

    HY3_CKPT=/path/to/Hy3-FP8 pytest contrib/models/hy_v3/test/module/test_moe_tkg_fused.py -v -s

``HY3_FUSED_BENCH=1`` also times 8 chained layers (each layer's input is the previous
output) against the production path: torch RMSNorm + hy3_route + moe_tkg + shared NF.mlp.
"""

import os
import time
import types

import pytest
import torch
from conftest import dequant, fake_quant, rel_err
from hy_v3_neuron import model as hy_model
from hy_v3_neuron.kernels.moe_tkg_fused import moe_tkg_fused
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

import test_mlp_moe as tm

H, E, K, L, I = 4096, 192, 8, 3, 1536
EPS, ROUTE_SCALE, SHARED_W = 1e-5, 2.826, 8 / 64
# W8A8 against the fake-quant reference: the activations round differently on the device
# (about 3% relative L2; the same fake-quant is ~4% from a BF16 reference).
FUSED_TOL = 6e-2


@pytest.fixture(autouse=True)
def single_rank_tp(monkeypatch):
    group = types.SimpleNamespace(world_size=1, rank_in_group=0)
    monkeypatch.setattr(hy_model, "get_tp_group", lambda: group)


def _layer(ckpt, layer):
    """Kernel inputs for ``layer`` (rank 0) and what the reference needs."""
    w = tm._expert_tensors(ckpt, layer)
    p = f"model.layers.{layer}.mlp"
    x2 = lambda key: ckpt.scalar(key) * 2  # input scales are doubled at load time
    s_in = torch.stack([torch.maximum(x2(f"{p}.experts.{e}.gate_proj.input_scale"),
                                      x2(f"{p}.experts.{e}.up_proj.input_scale")) for e in range(L)])
    s_din = torch.stack([x2(f"{p}.experts.{e}.down_proj.input_scale") for e in range(L)])
    gw, uw, dw = w["gate_up_scale_tkg"][:, 0, 0], w["gate_up_scale_tkg"][:, 1, 0], w["down_scale_tkg"][:, 0]
    sh = tm._build_mlp(ckpt, f"{p}.shared_mlp", 1536, 8)
    one = lambda t: t.data.reshape(-1)[0].float()
    sh_in, sh_din = one(sh.gate_up_input_scale), one(sh.down_input_scale)
    sc = torch.cat([
        torch.stack([gw * s_in, uw * s_in / s_din, dw * s_din, 1 / s_in], 1),
        torch.stack([one(sh.gate_weight_scale) * sh_in, one(sh.up_weight_scale) * sh_in / sh_din,
                     one(sh.down_weight_scale) * sh_din * SHARED_W, 1 / sh_in]).view(1, 4),
    ]).reshape(1, 4 * (L + 1))
    router = ckpt.get(f"{p}.router.gate.weight").float()
    return dict(
        w=w, sh=sh, sc=sc.contiguous(), s_in=s_in, s_din=s_din, router=router,
        router_wT=router.to(torch.bfloat16).t().contiguous(),
        bias=ckpt.get(f"{p}.expert_bias").float(),
        ln=ckpt.get(f"model.layers.{layer}.post_attention_layernorm.weight").to(torch.bfloat16).view(1, -1),
    )


def _reference(ckpt, layer, d, x):
    """FP32 fake-quant reference of the block's rank-0 partial, plus the torch top-k."""
    xf = x.float()
    normed = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * d["ln"].float()).to(torch.bfloat16)
    aff, idx = hy_model.hy3_route(normed, d["router"], d["bias"], K, True, ROUTE_SCALE)
    p = f"model.layers.{layer}.mlp.experts"
    out = torch.zeros(x.shape[0], H)
    for e in range(L):
        W = {n: dequant(ckpt.get(f"{p}.{e}.{n}_proj.weight"), ckpt.scalar(f"{p}.{e}.{n}_proj.weight_scale"))
             for n in ("gate", "up", "down")}
        xq = fake_quant(normed.float(), d["s_in"][e])
        inter = torch.nn.functional.silu(xq @ W["gate"].T) * (xq @ W["up"].T)
        out += (fake_quant(inter, d["s_din"][e]) @ W["down"].T) * aff[:, e : e + 1]
    sh = d["sh"]
    one = lambda t: t.data.reshape(-1)[0].float()
    xq = fake_quant(normed.float(), one(sh.gate_up_input_scale))
    g = sh.gate_proj_weight.data.float() * one(sh.gate_weight_scale)
    u = sh.up_proj_weight.data.float() * one(sh.up_weight_scale)
    dn = sh.down_proj_weight.data.float() * one(sh.down_weight_scale)
    inter = torch.nn.functional.silu(xq @ g) * (xq @ u)
    out += (fake_quant(inter, one(sh.down_input_scale)) @ dn) * SHARED_W
    return out, idx


class _Fused(torch.nn.Module):
    def __init__(self, ds, chain):
        super().__init__()
        self.n, self.chain = len(ds), chain
        for i, d in enumerate(ds):
            for k in ("ln", "router_wT", "sc"):
                self.register_buffer(f"{k}_{i}", d[k])
            self.register_buffer(f"bias_{i}", d["bias"].view(1, -1))
            self.register_buffer(f"gu_{i}", d["w"]["gate_up"])
            self.register_buffer(f"dn_{i}", d["w"]["down"])
            for n in ("gate", "up", "down"):
                self.register_buffer(f"sh_{n}_{i}", getattr(d["sh"], f"{n}_proj_weight").data)
        self.register_buffer("ebase", torch.zeros(1, 1))

    def forward(self, x):
        acc, out = x.float(), None
        for i in range(self.n):
            g = lambda k: getattr(self, f"{k}_{i}")
            xin = acc.to(x.dtype) if self.chain and i else x
            out = wrap_nki(moe_tkg_fused)[2](
                xin, g("ln"), g("router_wT"), g("bias"), self.ebase, g("gu"), g("dn"),
                g("sh_gate"), g("sh_up"), g("sh_down"), g("sc"),
                eps=EPS, route_scale=ROUTE_SCALE, top_k=K,
            )
            acc = acc + out.float() * 1e-3
        return out if self.n == 1 else acc


class _Production(torch.nn.Module):
    """torch RMSNorm + hy3_route + moe_tkg (all-expert) + shared NF.mlp, chained."""

    def __init__(self, ds):
        super().__init__()
        self.n = len(ds)
        self.sh = torch.nn.ModuleList([d["sh"] for d in ds])
        for i, d in enumerate(ds):
            self.register_buffer(f"router_{i}", d["router"])
            self.register_buffer(f"bias_{i}", d["bias"])
            self.register_buffer(f"ln_{i}", d["ln"])
            for k in ("gate_up", "down", "gate_up_scale_tkg", "down_scale_tkg"):
                self.register_buffer(f"{k}_{i}", d["w"][k])

    def forward(self, x):
        acc = x.float()
        for i in range(self.n):
            g = lambda k: getattr(self, f"{k}_{i}")
            xin = acc.to(x.dtype) if i else x
            xf = xin.float()
            normed = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * g("ln").float()).to(x.dtype)
            aff, idx = hy_model.hy3_route(normed, g("router"), g("bias"), K, True, ROUTE_SCALE)
            rank = torch.zeros((1, 1), dtype=torch.int32, device=x.device)
            o = hy_model.routed_experts_tkg(normed, aff, idx, rank, g("gate_up"), g("down"),
                                            g("gate_up_scale_tkg"), g("down_scale_tkg"))
            o = o + self.sh[i](xin, g("ln"), EPS, False) * SHARED_W
            acc = acc + o.float() * 1e-3
        return acc


def _compile(m):
    from vllm_neuron.envs import get_compile_backend_name

    return torch.compile(m.to("neuron:0"), backend=get_compile_backend_name(), fullgraph=True)


@pytest.mark.parametrize("tokens", [8, 32])
def test_moe_tkg_fused(ckpt, tokens):
    torch.manual_seed(0)
    layer = 1
    d = _layer(ckpt, layer)
    # Selection-only bias: every token picks local experts 0 and 2, so the routed part of the
    # rank-0 output is exercised.
    d["bias"] = d["bias"].clone()
    d["bias"][0] += 10
    d["bias"][2] += 10
    x = (torch.randn(tokens, H) * torch.linspace(0.5, 4, tokens).view(-1, 1)).to(torch.bfloat16)
    ref, idx = _reference(ckpt, layer, d, x)
    assert bool((idx == 0).any(1).all()) and bool((idx == 2).any(1).all())
    out = _compile(_Fused([d], chain=False))(x.to("neuron:0")).cpu()
    err = rel_err(out, ref)
    print(f"moe_tkg_fused T={tokens}: rel_err={err:.3e}")
    assert err < FUSED_TOL


@pytest.mark.skipif(os.environ.get("HY3_FUSED_BENCH") != "1", reason="set HY3_FUSED_BENCH=1")
def test_moe_tkg_fused_latency(ckpt):
    torch.manual_seed(0)
    T, NL = 32, 8
    ds = [_layer(ckpt, l) for l in range(1, 1 + NL)]
    x = (torch.randn(T, H) * 2).to(torch.bfloat16).to("neuron:0")

    def per_layer(make):
        times = {}
        for n in (2, NL):
            torch._dynamo.reset()
            f = _compile(make(ds[:n]))
            f(x).cpu()
            ts = []
            for _ in range(30):
                t = time.perf_counter()
                f(x).cpu()
                ts.append(time.perf_counter() - t)
            times[n] = sorted(ts)[15] * 1e3
        return (times[NL] - times[2]) / (NL - 2)

    torch._dynamo.config.recompile_limit = 64
    prod = per_layer(_Production)
    fused = per_layer(lambda d: _Fused(d, chain=True))
    print(f"decode MoE block, T={T}, layer-to-layer: production {prod:.3f} ms, fused {fused:.3f} ms")
    assert fused < prod
