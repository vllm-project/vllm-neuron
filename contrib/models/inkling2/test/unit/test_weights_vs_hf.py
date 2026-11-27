# SPDX-License-Identifier: Apache-2.0
"""``weights.gate_up`` / ``weights.down`` against Hugging Face's own load ops (``Inkling2LoopDequantizeExperts``,
``Interleave``) on the real checkpoint: one NVFP4 layer and one MXFP8 layer, the 8 ranks' shares concatenated back.
Needs the checkpoint (``INKLING2_CKPT``) and a transformers with the Inkling 2 Loop conversion ops."""

import json
import os

import pytest
import torch

CKPT = os.environ.get("INKLING2_CKPT", "/models/Inkling-2-Mini")
pytest.importorskip("safetensors")
if not os.path.exists(os.path.join(CKPT, "model.safetensors.index.json")):
    pytest.skip("no Inkling 2 Mini checkpoint", allow_module_level=True)
try:
    from transformers.core_model_loading import Inkling2LoopDequantizeExperts, Interleave
except ImportError:  # pragma: no cover
    pytest.skip("transformers without the Inkling 2 Loop load ops", allow_module_level=True)

from safetensors import safe_open

from inkling2_neuron import weights as W

TP, INTER = 8, 1152


def tensors(layer: int, which: str):
    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    base = f"model.llm.layers.{layer}.mlp.experts.{which}"
    out = {}
    for suffix in ("", "_scale", "_amax", "_outer_scale_denominator"):
        k = base + suffix
        if k in idx:
            with safe_open(os.path.join(CKPT, idx[k]), "pt") as f:
                out[suffix] = f.get_tensor(k)
    return out


def hf_reference(t: dict, interleave: bool) -> torch.Tensor:
    """HF's load: every source cast to bf16 first, then the dequant op (and the interleave for w13)."""
    src = {k: (v.to(torch.bfloat16) if k else v) for k, v in t.items()}
    pats = ["w", "w_scale", "w_amax", "w_denom"]
    inp = {pats[0]: src[""], pats[1]: src["_scale"]}
    if "_amax" in src:
        inp[pats[2]], inp[pats[3]] = src["_amax"], src["_outer_scale_denominator"]
    op = Inkling2LoopDequantizeExperts()
    out = op.convert({k: [v] for k, v in inp.items()}, pats, ["out"], model=None, full_layer_name=None)["out"]
    if interleave:
        out = Interleave(dim=1).convert({"out": [out]}, ["out"], ["out"])["out"]
    return out


@pytest.mark.parametrize("layer", [0, 20])
def test_experts_match_hf(layer):
    inter_r = INTER // TP
    t13, t2 = tensors(layer, "w13_weight"), tensors(layer, "w2_weight")
    gu = [W.gate_up(t13[""], t13["_scale"], t13.get("_amax"), t13.get("_outer_scale_denominator"), r, inter_r)
          for r in range(TP)]
    dn = [W.down(t2[""], t2["_scale"], t2.get("_amax"), t2.get("_outer_scale_denominator"), r, inter_r)
          for r in range(TP)]
    gate = torch.cat([g[:, :inter_r] for g in gu], dim=1)
    up = torch.cat([g[:, inter_r:] for g in gu], dim=1)
    ours_gu, ours_dn = torch.cat([gate, up], dim=1), torch.cat(dn, dim=2)
    ref_gu, ref_dn = hf_reference(t13, True), hf_reference(t2, False)
    assert ours_gu.shape == ref_gu.shape and ours_dn.shape == ref_dn.shape
    d_gu = (ours_gu.float() - ref_gu.float()).abs().max().item()
    d_dn = (ours_dn.float() - ref_dn.float()).abs().max().item()
    print(f"layer {layer}: gate_up max |diff| {d_gu:.3e} (|ref| max {ref_gu.float().abs().max():.3e}), "
          f"down {d_dn:.3e}")
    assert d_gu == 0.0 and d_dn == 0.0
