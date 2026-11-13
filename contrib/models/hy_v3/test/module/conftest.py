# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the Hy3 module tests.

The tests run single kernels on one NeuronCore with real weight shards from the
checkpoint (the TP=64 / EP=64 rank-0 slice), built through the model's own
weight loaders, and compare against an FP32 reference that applies the same
static activation quantization (fake-quant).
"""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

# Some kernels lower to NEFFs with collective ops even on one core, which the
# runtime only loads with a root communicator id.
os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:62182")
os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")

FP8 = torch.float8_e4m3fn
TRN2_FP8_MAX = 240.0
# Relative L2 error bound for paths with BF16 activations (vs an exact FP32
# reference with the dequantized weights).
REL_TOL = 2e-2


class Checkpoint:
    def __init__(self, root: str):
        self.root = root
        index = Path(root, "model.safetensors.index.json").read_text()
        self.weight_map = json.loads(index)["weight_map"]
        self._handles = {}

    def _file(self, key):
        f = self.weight_map[key]
        if f not in self._handles:
            self._handles[f] = safe_open(f"{self.root}/{f}", "pt")
        return self._handles[f]

    def slice(self, key):
        return self._file(key).get_slice(key)

    def get(self, key):
        return self._file(key).get_tensor(key)

    def scalar(self, key):
        return self.get(key).float().reshape(())


@pytest.fixture(scope="session")
def ckpt() -> Checkpoint:
    root = os.environ.get("HY3_CKPT")
    if not root:
        pytest.skip("set HY3_CKPT to the tencent/Hy3-FP8 checkpoint directory")
    return Checkpoint(root)


def dequant(w_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return w_fp8.float() * scale.float()


def fake_quant(x: torch.Tensor, in_scale: torch.Tensor) -> torch.Tensor:
    """Kernel activation quant: clamp(x / s, +-240) -> fp8 -> * s."""
    q = (x.float() / in_scale).clamp(-TRN2_FP8_MAX, TRN2_FP8_MAX).to(FP8)
    return q.float() * in_scale


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    out, ref = out.float().cpu(), ref.float().cpu()
    return ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()


def compile_and_run(module: torch.nn.Module, *args, device="neuron:0"):
    from vllm_neuron.envs import get_compile_backend_name

    module = module.to(device)
    args = [a.to(device) if isinstance(a, torch.Tensor) else a for a in args]
    compiled = torch.compile(module, backend=get_compile_backend_name(), fullgraph=True)
    return compiled(*args).cpu()
