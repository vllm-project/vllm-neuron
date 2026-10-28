# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the fp8-resident routed-expert mode (MiMo-V2.5-Pro).

Covers, without Neuron hardware:

* the load-time requantization from the checkpoint's 128x128 blockwise
  ``float8_e4m3fn`` scales to per-output-channel Trn2-range (<= 240) fp8, in the
  layouts the model declares (``gate_up [E_L, H, 2I]`` + ``[E_L, 1, 2I]``,
  ``down [E_L, I, H]`` + ``[E_L, 1, H]``);
* ``moe_fp8.experts_dense_fp8`` against a BF16 reference that mirrors the
  model's existing per-expert ``_experts_torch`` loop;
* the ``auto`` expert-dtype rule for the V2.5 and Pro geometries.

Only ``torch`` and ``pytest`` are needed. The modules under test are loaded
straight from ``src/`` as a bare package (``__init__`` would pull in the
factory and vLLM), and when ``vllm_neuron`` / ``transformers`` are not
importable (a laptop) the two names those modules import are stubbed.

    python3 -m pytest contrib/models/mimo_v2_5/test/unit -q
"""

import dataclasses
import importlib
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
_PKG = "_mimo_v2_5_unit_src"


def _ensure_module(name: str, **attrs) -> None:
    """Import ``name``; if that fails, install a stub carrying ``attrs``."""
    try:
        importlib.import_module(name)
    except ImportError:
        pass
    else:
        return
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        sys.modules.setdefault(parent, types.ModuleType(parent))
    stub = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(stub, key, value)
    sys.modules[name] = stub


@dataclasses.dataclass
class _StubSafetensorsWeightLoader:
    transform: object = None

    def load(self, slices, rank):
        return self.transform(slices, rank).contiguous()


def _stub_missing_deps() -> None:
    _ensure_module(
        "vllm_neuron.utils.weight_loader",
        SafetensorsWeightLoader=_StubSafetensorsWeightLoader,
    )
    _ensure_module(
        "vllm_neuron.model.neuron_config",
        NeuronConfig=type("NeuronConfig", (), {}),
    )
    _ensure_module("transformers", PretrainedConfig=type("PretrainedConfig", (), {}))


def _load(name: str):
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [_SRC]
        sys.modules[_PKG] = pkg
    _stub_missing_deps()
    return importlib.import_module(f"{_PKG}.{name}")


wl = _load("weight_loaders_bf16")
moe_fp8 = _load("moe_fp8")
cfg_mod = _load("config")

E4M3FN_MAX = 448.0
BLOCK = 128


class FakeSlice:
    """The subset of safetensors' ``PySafeSlice`` the loaders use."""

    def __init__(self, tensor: torch.Tensor):
        self.t = tensor

    def get_shape(self):
        return list(self.t.shape)

    def __getitem__(self, idx):
        if self.t.dtype == torch.float8_e4m3fn:
            # Slice through a byte view: CPU fp8 indexing is not universal.
            return self.t.view(torch.uint8)[idx].view(torch.float8_e4m3fn)
        return self.t[idx]


def blockwise_quant(w: torch.Tensor):
    """fp32 ``[R, C]`` -> checkpoint-style (e4m3fn weight, ``[R/128, C/128]`` scale_inv)."""
    r, c = w.shape
    gr, gc = -(-r // BLOCK), -(-c // BLOCK)
    q = torch.empty(r, c, dtype=torch.float8_e4m3fn)
    s = torch.empty(gr, gc, dtype=torch.float32)
    for i in range(gr):
        for j in range(gc):
            blk = w[i * BLOCK : (i + 1) * BLOCK, j * BLOCK : (j + 1) * BLOCK]
            scale = blk.abs().amax().clamp(min=1e-12) / E4M3FN_MAX
            s[i, j] = scale
            q[i * BLOCK : (i + 1) * BLOCK, j * BLOCK : (j + 1) * BLOCK] = (
                blk / scale
            ).to(torch.float8_e4m3fn)
    return q, s


def dequant_blockwise(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return wl.dequant_fp8_blockwise(FakeSlice(q), FakeSlice(s))


def rel_fro(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm())


def _make_expert_checkpoint(n_experts: int, hidden: int, inter: int, seed: int = 0):
    """Per-expert HF tensors: gate/up ``[I, H]``, down ``[H, I]``, blockwise fp8.

    Rows get different magnitudes so per-channel scales actually differ, and
    the block scales vary across a row, which is what the requant must undo.
    """
    g = torch.Generator().manual_seed(seed)
    experts = []
    for _ in range(n_experts):
        tensors = {}
        for name, shape in (
            ("gate", (inter, hidden)),
            ("up", (inter, hidden)),
            ("down", (hidden, inter)),
        ):
            row_mag = torch.rand(shape[0], 1, generator=g) * 4 + 0.05
            w = torch.randn(*shape, generator=g) * row_mag * 1e-2
            tensors[name] = blockwise_quant(w)
        experts.append(tensors)
    return experts


def _gate_up_slices(experts):
    out = []
    for t in experts:
        for name in ("gate", "up"):
            q, s = t[name]
            out += [FakeSlice(q), FakeSlice(s)]
    return out


def _down_slices(experts):
    out = []
    for t in experts:
        q, s = t["down"]
        out += [FakeSlice(q), FakeSlice(s)]
    return out


# -----------------------------------------------------------------------------
# Requantization
# -----------------------------------------------------------------------------


def test_requant_rows_range_and_roundtrip():
    g = torch.Generator().manual_seed(1)
    w = torch.randn(256, 384, generator=g) * (torch.rand(256, 1, generator=g) + 0.1)
    q, s = wl.requant_rows_fp8(w)
    assert q.dtype == torch.float8_e4m3fn and s.shape == (256,)
    assert float(q.float().abs().max()) <= wl.TRN2_FP8_MAX
    # Each row's max lands exactly on 240 (exactly representable in e4m3).
    torch.testing.assert_close(
        q.float().abs().amax(dim=-1), torch.full((256,), wl.TRN2_FP8_MAX)
    )
    assert rel_fro(q.float() * s.unsqueeze(-1), w) < 0.03


def test_requant_handles_zero_row():
    w = torch.zeros(4, 128)
    w[1] = torch.linspace(-1, 1, 128)
    q, s = wl.requant_rows_fp8(w)
    assert torch.isfinite(s).all()
    assert float(q[0].float().abs().max()) == 0.0


@pytest.mark.parametrize("tp_degree", [1, 2])
def test_gate_up_requant_loader_layout_and_error(tp_degree):
    n_exp, hidden, inter = 3, 256, 256
    experts = _make_expert_checkpoint(n_exp, hidden, inter)
    shard = inter // tp_degree
    memo = wl.ExpertRequantMemo()
    rank = tp_degree - 1  # last shard
    loaders = {
        part: wl.expert_gate_up_requant_loader(
            shard_size=2 * shard, num_shards=tp_degree, part=part, memo=memo
        )
        for part in ("weight", "scale")
    }
    weight = loaders["weight"].load(_gate_up_slices(experts), rank)
    scale = loaders["scale"].load(_gate_up_slices(experts), rank)
    assert weight.dtype == torch.float8_e4m3fn
    assert tuple(weight.shape) == (n_exp, hidden, 2 * shard)
    assert tuple(scale.shape) == (n_exp, 1, 2 * shard)
    assert scale.dtype == torch.float32
    assert float(weight.float().abs().max()) <= wl.TRN2_FP8_MAX
    assert not memo._slots  # both parts handed out, slot released

    rows = slice(rank * shard, (rank + 1) * shard)
    for e, t in enumerate(experts):
        gate = dequant_blockwise(*t["gate"])[rows]
        up = dequant_blockwise(*t["up"])[rows]
        ref = torch.cat([gate, up], dim=0).T  # [H, 2I/TP], gate first
        got = weight[e].float() * scale[e]
        err = rel_fro(got, ref)
        assert err < 0.03, f"expert {e}: relative Frobenius error {err:.4f}"


def test_gate_up_requant_matches_bf16_loader_order():
    """fp8 layout must dequantize to the bf16 loader's tensor (same gate/up order)."""
    n_exp, hidden, inter = 2, 128, 256
    experts = _make_expert_checkpoint(n_exp, hidden, inter, seed=3)
    bf16 = wl.expert_gate_up_fp8_loader(
        shard_size=2 * inter, num_shards=1, dtype=torch.float32
    ).load(_gate_up_slices(experts), 0)
    memo = wl.ExpertRequantMemo()
    w = wl.expert_gate_up_requant_loader(2 * inter, 1, "weight", memo).load(
        _gate_up_slices(experts), 0
    )
    s = wl.expert_gate_up_requant_loader(2 * inter, 1, "scale", memo).load(
        _gate_up_slices(experts), 0
    )
    assert rel_fro(w.float() * s, bf16) < 0.03


def test_down_requant_loader_layout_and_error():
    n_exp, hidden, inter = 2, 384, 256
    experts = _make_expert_checkpoint(n_exp, hidden, inter, seed=5)
    memo = wl.ExpertRequantMemo()
    # Scale first: the memo must still hand the weight out afterwards.
    scale = wl.expert_down_requant_loader(inter, 1, "scale", memo).load(
        _down_slices(experts), 0
    )
    weight = wl.expert_down_requant_loader(inter, 1, "weight", memo).load(
        _down_slices(experts), 0
    )
    assert weight.dtype == torch.float8_e4m3fn
    assert tuple(weight.shape) == (n_exp, inter, hidden)
    assert tuple(scale.shape) == (n_exp, 1, hidden)
    for e, t in enumerate(experts):
        ref = dequant_blockwise(*t["down"]).T  # [I, H]
        assert rel_fro(weight[e].float() * scale[e], ref) < 0.03


def test_requant_from_bf16_checkpoint():
    """An unquantized re-export (no scale companions) requantizes too."""
    g = torch.Generator().manual_seed(7)
    hidden, inter = 128, 128
    gate = torch.randn(inter, hidden, generator=g).to(torch.bfloat16)
    up = torch.randn(inter, hidden, generator=g).to(torch.bfloat16)
    memo = wl.ExpertRequantMemo()
    slices = [FakeSlice(gate), FakeSlice(up)]
    w = wl.expert_gate_up_requant_loader(
        2 * inter, 1, "weight", memo, quantized=False
    ).load(slices, 0)
    s = wl.expert_gate_up_requant_loader(
        2 * inter, 1, "scale", memo, quantized=False
    ).load(slices, 0)
    ref = torch.cat([gate, up], dim=0).float().T
    assert rel_fro(w[0].float() * s[0], ref) < 0.03


# -----------------------------------------------------------------------------
# Dense fp8 expert compute
# -----------------------------------------------------------------------------


def _reference_experts_bf16(x, local_aff, gate_up, down):
    """The model's BF16 ``_experts_torch`` loop, on ``[E, H, 2I]`` / ``[E, I, H]``."""
    e_l, hidden, two_i = gate_up.shape
    inter = two_i // 2
    view = gate_up.reshape(e_l, hidden, 2, inter)
    out = torch.zeros(x.shape[0], hidden, dtype=torch.float32)
    for e in range(e_l):
        g = x @ view[e, :, 0, :]
        u = x @ view[e, :, 1, :]
        expert_out = (torch.nn.functional.silu(g.float()) * u.float()) @ down[e].float()
        out = out + expert_out * local_aff[:, e : e + 1]
    return out


def _routed_affinities(tokens, n_local, top_k, seed):
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand(tokens, n_local, generator=g)
    idx = scores.topk(top_k, dim=-1).indices
    w = torch.rand(tokens, top_k, generator=g)
    w = w / w.sum(-1, keepdim=True)
    return torch.zeros(tokens, n_local).scatter(1, idx, w)


@pytest.mark.parametrize("tokens, chunk", [(1, 1024), (7, 1024), (37, 8)])
def test_experts_dense_fp8_matches_bf16_reference(tokens, chunk):
    n_local, hidden, inter = 3, 256, 128
    g = torch.Generator().manual_seed(11)
    gate_up_f = torch.randn(n_local, hidden, 2 * inter, generator=g) * 0.05
    down_f = torch.randn(n_local, inter, hidden, generator=g) * 0.05
    x = torch.randn(tokens, hidden, generator=g).to(torch.bfloat16)
    aff = _routed_affinities(tokens, n_local, top_k=2, seed=12)

    # Quantize per output channel the way the loaders lay it out.
    gu_q, gu_s = zip(*(wl.requant_rows_fp8(gate_up_f[e].T) for e in range(n_local)))
    gu_q = torch.stack([q.view(torch.uint8).T for q in gu_q]).view(torch.float8_e4m3fn)
    gu_s = torch.stack(gu_s).unsqueeze(1)
    d_q, d_s = zip(*(wl.requant_rows_fp8(down_f[e].T) for e in range(n_local)))
    d_q = torch.stack([q.view(torch.uint8).T for q in d_q]).view(torch.float8_e4m3fn)
    d_s = torch.stack(d_s).unsqueeze(1)

    got = moe_fp8.experts_dense_fp8(
        x, aff, gu_q, gu_s, d_q, d_s, compute_dtype=torch.bfloat16, token_chunk=chunk
    )
    assert got.shape == (tokens, hidden) and got.dtype == torch.float32

    # Same fp8 weights, dequantized to bf16 up front, through the bf16 loop.
    gu_deq = (gu_q.float() * gu_s).to(torch.bfloat16)
    d_deq = (d_q.float() * d_s).to(torch.bfloat16)
    ref = _reference_experts_bf16(x, aff, gu_deq, d_deq)
    assert rel_fro(got, ref) < 0.02

    # And against the unquantized weights: the fp8 storage error itself.
    ref_full = _reference_experts_bf16(
        x, aff, gate_up_f.to(torch.bfloat16), down_f.to(torch.bfloat16)
    )
    assert rel_fro(got, ref_full) < 0.06

    # Chunking must not change the result.
    whole = moe_fp8.experts_dense_fp8(x, aff, gu_q, gu_s, d_q, d_s, token_chunk=4096)
    torch.testing.assert_close(got, whole, rtol=0, atol=0)


def test_unrouted_experts_contribute_nothing():
    n_local, hidden, inter, tokens = 2, 128, 128, 5
    g = torch.Generator().manual_seed(21)
    gu = torch.randn(n_local, hidden, 2 * inter, generator=g).to(torch.float8_e4m3fn)
    dn = torch.randn(n_local, inter, hidden, generator=g).to(torch.float8_e4m3fn)
    x = torch.randn(tokens, hidden, generator=g).to(torch.bfloat16)
    out = moe_fp8.experts_dense_fp8(
        x,
        torch.zeros(tokens, n_local),
        gu,
        torch.ones(n_local, 1, 2 * inter),
        dn,
        torch.ones(n_local, 1, hidden),
    )
    assert float(out.abs().max()) == 0.0


# -----------------------------------------------------------------------------
# Expert-dtype mode
# -----------------------------------------------------------------------------


def _v25():
    return cfg_mod.MiMoV2Config(
        hidden_size=4096,
        num_hidden_layers=48,
        moe_layer_freq=[0] + [1] * 47,
        n_routed_experts=256,
    )


def _pro():
    return cfg_mod.MiMoV2Config(
        hidden_size=6144,
        num_hidden_layers=70,
        num_attention_heads=128,
        swa_num_attention_heads=128,
        num_key_value_heads=8,
        moe_layer_freq=[0] + [1] * 69,
        n_routed_experts=384,
        attention_value_scale=0.612,
    )


def test_auto_rule(monkeypatch):
    monkeypatch.delenv(cfg_mod.EXPERT_DTYPE_ENV, raising=False)
    v25, pro = _v25(), _pro()
    # TP=64 / EP=64: 4 and 6 local experts, full I=2048 per rank.
    assert v25.resolve_expert_dtype(4, 2048) == "bf16"
    assert pro.resolve_expert_dtype(6, 2048) == "fp8"
    assert v25.expert_bytes_per_rank(4, 2048, 2) == 9_462_349_824  # 8.81 GiB
    assert pro.expert_bytes_per_rank(6, 2048, 2) == 31_255_953_408  # 29.11 GiB
    assert pro.expert_bytes_per_rank(6, 2048, 1) == 15_627_976_704  # 14.55 GiB
    assert pro.expert_scale_bytes_per_rank(6, 2048) == 16_957_440


def test_env_override(monkeypatch):
    pro, v25 = _pro(), _v25()
    monkeypatch.setenv(cfg_mod.EXPERT_DTYPE_ENV, "bf16")
    assert pro.resolve_expert_dtype(6, 2048) == "bf16"
    monkeypatch.setenv(cfg_mod.EXPERT_DTYPE_ENV, "FP8")
    assert v25.resolve_expert_dtype(4, 2048) == "fp8"
    monkeypatch.setenv(cfg_mod.EXPERT_DTYPE_ENV, "int4")
    with pytest.raises(ValueError):
        v25.resolve_expert_dtype(4, 2048)


def test_pro_quant_config_without_store_dtype_is_fp8():
    pro = _pro()
    pro.quantization_config = {"quant_method": "fp8", "fmt": "e4m3"}
    assert pro.is_fp8_checkpoint
    pro.quantization_config = {"quant_method": "fp8", "store_dtype": "bf16"}
    assert not pro.is_fp8_checkpoint
