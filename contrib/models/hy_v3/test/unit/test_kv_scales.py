# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the FP8 KV scale profile (src/kv_scales.py).

Needs only torch + safetensors (no vLLM / Neuron):
    pytest contrib/models/hy_v3/test/unit/test_kv_scales.py
"""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

_SRC = Path(__file__).resolve().parents[2] / "src" / "kv_scales.py"
_spec = importlib.util.spec_from_file_location("kv_scales", _SRC)
kv_scales = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kv_scales)

N = 3


@pytest.fixture
def ckpt(tmp_path):
    """A tiny fake checkpoint with the fingerprinted attention tensors."""
    d = tmp_path / "ckpt"
    d.mkdir()
    t = {}
    g = torch.Generator().manual_seed(0)
    for i in range(N + 1):
        for k in kv_scales._FINGERPRINT_KEYS:
            shape = (128,) if k.endswith("norm.weight") else ()
            t[f"model.layers.{i}.self_attn.{k}"] = torch.rand(shape, generator=g)
    save_file(t, str(d / "model-00000.safetensors"))
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model-00000.safetensors" for k in t}})
    )
    return str(d)


def _profile(tmp_path, ckpt, k=None, v=None, **meta):
    k = k or [0.05, 0.1, 0.2]
    v = v or [0.01, 0.02, 0.04]
    m = {"target_fp8_max": "240.0", "attn_fingerprints": kv_scales.attn_fingerprints(ckpt, len(k))}
    m.update(meta)
    p = str(tmp_path / "p.safetensors")
    kv_scales.save_profile(p, k, v, m)
    return p


def test_roundtrip_returns_quant_scales(tmp_path, ckpt):
    p = _profile(tmp_path, ckpt)
    qk, qv, meta = kv_scales.load_profile(p, ckpt, N, 240.0)
    assert qk == pytest.approx([20.0, 10.0, 5.0])
    assert qv == pytest.approx([100.0, 50.0, 25.0])
    assert meta["direction"] == "dequant" and len(meta["sha256"]) == 64


def test_prefix_of_layers_is_accepted(tmp_path, ckpt):
    # A reduced model (first 2 layers) can use a profile of the full model.
    p = _profile(tmp_path, ckpt)
    qk, _, _ = kv_scales.load_profile(p, ckpt, 2, 240.0)
    assert len(qk) == 2


def test_wrong_target_range_rejected(tmp_path, ckpt):
    p = _profile(tmp_path, ckpt, target_fp8_max="448.0")
    with pytest.raises(ValueError, match="FP8 max"):
        kv_scales.load_profile(p, ckpt, N, 240.0)


def test_missing_layer_rejected(tmp_path, ckpt):
    p = _profile(tmp_path, ckpt)
    for check in (True, False):
        with pytest.raises(KeyError, match="layers.3"):
            kv_scales.load_profile(p, ckpt, N + 1, 240.0, check_fingerprint=check)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_non_positive_or_non_finite_rejected(tmp_path, ckpt, bad):
    with pytest.raises(ValueError, match="finite and positive"):
        _profile(tmp_path, ckpt, k=[0.05, bad, 0.2])


def test_fingerprint_mismatch_rejected(tmp_path, ckpt):
    p = _profile(tmp_path, ckpt)
    # Change layer 1's k_proj input scale in the checkpoint.
    f = Path(ckpt, "model-00000.safetensors")
    from safetensors.torch import load_file

    t = load_file(str(f))
    t["model.layers.1.self_attn.k_proj.input_scale"] = torch.tensor(9.0)
    save_file(t, str(f))
    with pytest.raises(ValueError, match=r"layers \[1\]"):
        kv_scales.load_profile(p, ckpt, N, 240.0)


def test_not_a_profile_rejected(tmp_path, ckpt):
    p = str(tmp_path / "x.safetensors")
    save_file({"a": torch.ones(1)}, p, metadata={"format": "pt"})
    with pytest.raises(ValueError, match="not a"):
        kv_scales.load_profile(p, ckpt, N, 240.0)


def test_resolve_profile_path(tmp_path, ckpt, monkeypatch):
    monkeypatch.delenv(kv_scales.PROFILE_ENV, raising=False)
    assert kv_scales.resolve_profile_path(ckpt) is None
    Path(ckpt, "kv_scales.safetensors").write_bytes(b"")
    assert kv_scales.resolve_profile_path(ckpt).endswith("kv_scales.safetensors")
    monkeypatch.setenv(kv_scales.PROFILE_ENV, "/x/y.safetensors")
    assert kv_scales.resolve_profile_path(ckpt) == "/x/y.safetensors"
