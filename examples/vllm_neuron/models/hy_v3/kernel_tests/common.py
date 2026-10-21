"""Shared helpers for Hy3 kernel micro-tests (single NeuronCore)."""

import json
import os

import torch
from safetensors import safe_open

# Local tencent/Hy3-FP8 checkpoint directory (tests read real weights/scales).
CKPT = os.environ.get("HY3_CKPT", "Hy3-FP8")
FP8 = torch.float8_e4m3fn
TRN2_FP8_MAX = 240.0


class Ckpt:
    def __init__(self, root=CKPT):
        self.root = root
        self.wm = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
        self.handles = {}

    def get(self, key):
        f = self.wm[key]
        if f not in self.handles:
            self.handles[f] = safe_open(f"{self.root}/{f}", "pt")
        return self.handles[f].get_tensor(key)

    def scalar(self, key):
        return self.get(key).float().reshape(())


def dequant(w_fp8, scale):
    return w_fp8.float() * scale.float()


def downscale(w_fp8):
    """Halve an e4m3fn tensor so it fits TRN2's 240-capped e4m3 (exact for normals)."""
    return (w_fp8.float() * 0.5).to(FP8)


def fake_quant(x, in_scale):
    """Reference activation quant matching the kernels: clamp(x/s, ±240) -> fp8 -> *s."""
    q = (x.float() / in_scale).clamp(-TRN2_FP8_MAX, TRN2_FP8_MAX).to(FP8)
    return q.float() * in_scale


def bcast(s, cols=1):
    return s.float().reshape(1, -1).expand(128, cols).contiguous()


def report(name, out, ref):
    out = out.float().cpu()
    ref = ref.float().cpu()
    err = (out - ref).abs()
    rel = err.norm() / ref.norm().clamp_min(1e-12)
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
    status = "OK " if rel < 2e-2 else "BAD"
    print(
        f"[{status}] {name}: rel_l2={rel:.3e} cos={cos:.6f} "
        f"max_abs={err.max():.3e} ref_absmax={ref.abs().max():.3e}",
        flush=True,
    )
    return rel.item()


def compile_and_run(module, *args, device="neuron:0"):
    from vllm_neuron.envs import get_compile_backend_name

    module = module.to(device)
    args = [a.to(device) if isinstance(a, torch.Tensor) else a for a in args]
    compiled = torch.compile(module, backend=get_compile_backend_name(), fullgraph=True)
    out = compiled(*args)
    if isinstance(out, (tuple, list)):
        return [o.cpu() if isinstance(o, torch.Tensor) else o for o in out]
    return out.cpu()
