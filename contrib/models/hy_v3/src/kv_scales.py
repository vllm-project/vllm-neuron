# SPDX-License-Identifier: Apache-2.0
"""Hy3 FP8 KV-cache scales: where they come from, the profile format and validation.

``resolve_kv_scales`` picks, in order:

1. a KV scale profile (``$HY3_KV_SCALE_PROFILE``, else
   ``<ckpt>/kv_scales.safetensors``), the format below;
2. per-layer ``self_attn.k_scale`` / ``v_scale`` in the checkpoint itself (the
   compressed-tensors / llm-compressor convention the upstream models load),
   rescaled from the OCP e4m3fn range they are calibrated for (448) to this
   platform's clamp;
3. otherwise an error, unless ``$HY3_KV_SCALE_ALLOW_DEFAULT=1`` asks for the
   upstream default (1.0 for every layer, logged as a warning).

tencent/Hy3-FP8 declares ``kv_cache_scheme: static`` but ships no K/V scales,
so without a profile it gets case 3. Scale 1.0 is not usable for it: V is
~0.1-0.2 in the early layers, which a direct cast puts in the FP8 subnormal
range. ``kv_profiles/Hy3-FP8_neuron.safetensors`` is a calibrated profile for
it (``scripts/calibrate_kv_scales_neuron.py``).

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
from typing import NamedTuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PROFILE_FORMAT = "hy3-kv-scale-profile/v1"
# FP8 max of the OCP e4m3fn format that checkpoint KV scales are calibrated for.
CHECKPOINT_FP8_MAX = 448.0
PROFILE_ENV = "HY3_KV_SCALE_PROFILE"
ALLOW_DEFAULT_ENV = "HY3_KV_SCALE_ALLOW_DEFAULT"
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
    """Tensor lookup in a sharded (index.json) or single-file checkpoint."""

    def __init__(self, ckpt_dir: str):
        self.root = ckpt_dir
        index = Path(ckpt_dir, "model.safetensors.index.json")
        single = Path(ckpt_dir, "model.safetensors")
        if index.exists():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        elif single.exists():
            with safe_open(str(single), "pt") as f:
                self.weight_map = dict.fromkeys(f.keys(), single.name)
        else:
            self.weight_map = {}
        self._handles = {}

    def __contains__(self, key: str) -> bool:
        return key in self.weight_map

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
            t = idx.get(scale_key(i, k)).to(torch.float32)
            h.update(k.encode())
            h.update(t.contiguous().numpy().tobytes())
        out.append(h.hexdigest()[:16])
    return out


def _positive(what: str, d: float) -> float:
    if not (math.isfinite(d) and d > 0):
        raise ValueError(f"{what}={d} is not finite and positive")
    return d


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
            _positive(f"layer {i} {name}", d)
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
                d = _positive(f"{path}: {key}", float(f.get_tensor(key).float().reshape(())))
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


def checkpoint_kv_dequant(
    ckpt_dir: str, num_layers: int
) -> tuple[list[float], list[float]] | None:
    """Per-layer DEQUANT K/V scales stored in the checkpoint, or None if it has none.

    Keys ``model.layers.{i}.self_attn.k_scale`` / ``v_scale`` (one fp32 scalar
    each), for an FP8 range of ``CHECKPOINT_FP8_MAX``. Some layers having them
    and others not is an error.
    """
    idx = _Index(ckpt_dir)
    keys = [scale_key(i, n) for i in range(num_layers) for n in ("k_scale", "v_scale")]
    missing = [k for k in keys if k not in idx]
    if len(missing) == len(keys):
        return None
    if missing:
        raise KeyError(f"{ckpt_dir}: has KV scales for some layers only; missing {missing[:4]}")
    vals = [_positive(f"{ckpt_dir}: {k}", float(idx.get(k).float().reshape(()))) for k in keys]
    return vals[0::2], vals[1::2]


class KVScales(NamedTuple):
    """Per-layer QUANT scales and where they came from (``source``: profile /
    checkpoint / default; ``detail``: ``key=value`` text for the log)."""

    q_k: list[float]
    q_v: list[float]
    source: str
    detail: str


def resolve_kv_scales(ckpt_dir: str, num_layers: int, target_fp8_max: float) -> KVScales:
    """Per-layer K/V QUANT scales, from the first source that has them.

    See the module docstring for the order. A checkpoint scale ``d`` maps
    ``CHECKPOINT_FP8_MAX`` to the activation range; on a platform that clamps
    at ``target_fp8_max`` the same range needs ``d * CHECKPOINT_FP8_MAX /
    target_fp8_max``, otherwise everything above ``target_fp8_max * d`` clips.
    """
    path = resolve_profile_path(ckpt_dir)
    if path is not None:
        q_k, q_v, meta = load_profile(path, ckpt_dir, num_layers, target_fp8_max)
        return KVScales(q_k, q_v, "profile", f"path={path} sha256={meta['sha256'][:16]}")
    stored = checkpoint_kv_dequant(ckpt_dir, num_layers)
    if stored is not None:
        r = CHECKPOINT_FP8_MAX / target_fp8_max
        q_k, q_v = ([1.0 / (d * r) for d in ds] for ds in stored)
        return KVScales(q_k, q_v, "checkpoint", f"rescale={r:.4g}")
    if os.environ.get(ALLOW_DEFAULT_ENV) != "1":
        raise ValueError(
            f"FP8 KV cache: no KV scales for {ckpt_dir} (no ${PROFILE_ENV}, no "
            "<ckpt>/kv_scales.safetensors, no self_attn.k_scale / v_scale in the "
            "checkpoint). Hy3's V is too small for scale 1.0 (FP8 subnormals): set "
            f"{PROFILE_ENV} to a calibrated profile, e.g. contrib/models/hy_v3/"
            "kv_profiles/Hy3-FP8_neuron.safetensors, or calibrate one with "
            f"scripts/calibrate_kv_scales_neuron.py; {ALLOW_DEFAULT_ENV}=1 forces 1.0."
        )
    return KVScales([1.0] * num_layers, [1.0] * num_layers, "default", "")


def resolve_profile_path(ckpt_dir: str) -> str | None:
    """``$HY3_KV_SCALE_PROFILE``, else ``<ckpt>/kv_scales.safetensors``."""
    p = os.environ.get(PROFILE_ENV)
    if p:
        return p
    p = os.path.join(ckpt_dir, "kv_scales.safetensors")
    return p if os.path.exists(p) else None
