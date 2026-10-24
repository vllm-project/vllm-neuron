# SPDX-License-Identifier: Apache-2.0
"""Hy3 FP8 KV-cache scale profile: format, fingerprint, load and validation.

tencent/Hy3-FP8 declares ``kv_cache_scheme: static`` but ships no K/V scales,
so FP8 KV needs a calibrated profile next to the (unchanged) W8A8 checkpoint.

Profile file: one safetensors file with, per decoder layer ``i``,

    model.layers.{i}.self_attn.k_scale   fp32 scalar
    model.layers.{i}.self_attn.v_scale   fp32 scalar

holding DEQUANT scales ``d`` (``x ~= fp8 * d``) for the target FP8 range
(``target_fp8_max``, 240 on Trn2), and string metadata (``PROFILE_FORMAT``,
``direction``, ``target_fp8_max``, per-layer ``attn_fingerprints`` ...).

Statistic positions (must match what the Neuron cache stores):
    K: after the per-head q/k-norm and RoPE, before the GQA repeat.
    V: the v_proj output.

The model consumes the quantizing direction ``q = 1 / d`` (cache write is
``fp8(clamp(x * q, +-target_fp8_max))``).

Only torch / safetensors / json are imported, so the calibration environment
(llm-compressor, no vLLM) can use this module to write profiles.
"""

import hashlib
import json
import math
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PROFILE_FORMAT = "hy3-kv-scale-profile/v1"
PROFILE_ENV = "HY3_KV_SCALE_PROFILE"
_KEY = "model.layers.{layer}.self_attn.{name}"
# Small per-layer tensors that identify the attention weights a profile was
# calibrated for (identical in the full checkpoint and reduced copies of it).
_FINGERPRINT_KEYS = (
    "k_proj.weight_scale",
    "v_proj.weight_scale",
    "k_proj.input_scale",
    "v_proj.input_scale",
    "k_norm.weight",
)


def scale_key(layer: int, name: str) -> str:
    return _KEY.format(layer=layer, name=name)


class _Index:
    def __init__(self, ckpt_dir: str):
        self.root = ckpt_dir
        index = json.loads(Path(ckpt_dir, "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self._handles = {}

    def get(self, key: str) -> torch.Tensor:
        f = self.weight_map[key]
        if f not in self._handles:
            self._handles[f] = safe_open(os.path.join(self.root, f), "pt")
        return self._handles[f].get_tensor(key)


def attn_fingerprints(ckpt_dir: str, num_layers: int) -> list[str]:
    """Per-layer sha256 over the K/V projection scales and the k_norm weight."""
    idx = _Index(ckpt_dir)
    out = []
    for i in range(num_layers):
        h = hashlib.sha256()
        for k in _FINGERPRINT_KEYS:
            t = idx.get(f"model.layers.{i}.self_attn.{k}").to(torch.float32)
            h.update(k.encode())
            h.update(t.contiguous().numpy().tobytes())
        out.append(h.hexdigest()[:16])
    return out


def save_profile(
    path: str,
    k_dequant: list[float],
    v_dequant: list[float],
    metadata: dict,
) -> str:
    """Write a profile; returns the file's sha256."""
    if len(k_dequant) != len(v_dequant):
        raise ValueError("k/v scale counts differ")
    tensors = {}
    for i, (dk, dv) in enumerate(zip(k_dequant, v_dequant)):
        for name, d in (("k_scale", dk), ("v_scale", dv)):
            if not (math.isfinite(d) and d > 0):
                raise ValueError(f"layer {i} {name}={d} is not finite and positive")
            tensors[scale_key(i, name)] = torch.tensor(d, dtype=torch.float32)
    meta = {k: v if isinstance(v, str) else json.dumps(v) for k, v in metadata.items()}
    meta["format"] = PROFILE_FORMAT
    meta["direction"] = "dequant"
    meta["num_layers"] = str(len(k_dequant))
    save_file(tensors, path, metadata=meta)
    return file_sha256(path)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_profile(
    path: str,
    ckpt_dir: str,
    num_layers: int,
    target_fp8_max: float,
    check_fingerprint: bool = True,
) -> tuple[list[float], list[float], dict]:
    """Load and validate a profile for the first ``num_layers`` layers.

    Returns per-layer QUANT scales ``(q_k, q_v)`` and the metadata. Raises on a
    wrong format/direction, a target range that differs from the hardware's,
    missing / non-finite / non-positive scales, or an attention-weight
    fingerprint that does not match ``ckpt_dir``.
    """
    with safe_open(path, "pt") as f:
        meta = f.metadata() or {}
        if meta.get("format") != PROFILE_FORMAT:
            raise ValueError(f"{path}: not a {PROFILE_FORMAT} file (format={meta.get('format')!r})")
        if meta.get("direction") != "dequant":
            raise ValueError(f"{path}: expected dequant scales, got {meta.get('direction')!r}")
        prof_max = float(meta.get("target_fp8_max", "nan"))
        if prof_max != target_fp8_max:
            raise ValueError(
                f"{path}: calibrated for FP8 max {prof_max}, this platform clamps "
                f"at {target_fp8_max}; recalibrate (or re-export) for this target"
            )
        q_k, q_v = [], []
        keys = set(f.keys())
        for i in range(num_layers):
            for name, dst in (("k_scale", q_k), ("v_scale", q_v)):
                key = scale_key(i, name)
                if key not in keys:
                    raise KeyError(f"{path}: missing {key}")
                d = float(f.get_tensor(key).float().reshape(()))
                if not (math.isfinite(d) and d > 0):
                    raise ValueError(f"{path}: {key}={d} is not finite and positive")
                dst.append(1.0 / d)
    if check_fingerprint:
        want = json.loads(meta.get("attn_fingerprints", "[]"))
        if len(want) < num_layers:
            raise ValueError(f"{path}: fingerprints cover {len(want)} < {num_layers} layers")
        have = attn_fingerprints(ckpt_dir, num_layers)
        bad = [i for i in range(num_layers) if want[i] != have[i]]
        if bad:
            raise ValueError(
                f"{path}: attention weights of layers {bad[:8]} differ from the "
                f"checkpoint the profile was calibrated on ({ckpt_dir})"
            )
    meta = dict(meta)
    meta["sha256"] = file_sha256(path)
    return q_k, q_v, meta


def resolve_profile_path(ckpt_dir: str) -> str | None:
    """``$HY3_KV_SCALE_PROFILE``, else ``<ckpt>/kv_scales.safetensors``."""
    p = os.environ.get(PROFILE_ENV)
    if p:
        return p
    p = os.path.join(ckpt_dir, "kv_scales.safetensors")
    return p if os.path.exists(p) else None
