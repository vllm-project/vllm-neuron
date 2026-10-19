# SPDX-License-Identifier: Apache-2.0
"""Rewrite ``zai-org/GLM-5.3`` (FP8, 128 x 128 block scales) into what Trn2 can run.

The checkpoint stores every linear weight as ``float8_e4m3fn`` with a companion
``weight_scale_inv [ceil(N/128), ceil(K/128)]`` (one fp32 per 128 x 128 tile).
Trn2's fp8 tops out at 240, not e4m3fn's 448, so the checkpoint is rewritten once:

* **Routed experts** are requantised to one scale per output row inside the 240
  range and fused per layer into the layout ``vllm_neuron/model/glm_dsa`` loads
  (the same one the Hy4 port uses): ``mlp.experts.gate_up_proj [E, 2I, H]`` with
  ``gate_up_proj_scale [E, 2I]`` (gate rows, then up rows), and
  ``mlp.experts.down_proj_t [E, I, H]`` -- down transposed so a rank's shard of I
  is a row slice -- with ``down_proj_scale [E, H]``.
* **Everything else that is fp8** is dequantised to bf16 (exact: an e4m3 value
  times an fp32 tile scale rounds once).
* Unquantised tensors are copied. The MTP layer (``model.layers.78``) is dropped.

One output shard per decoder layer, plus ``rest.safetensors``. Resumable: a shard
is written under ``.partial`` and renamed. The script reports the per-row
requantisation error of the experts (block -> row), sampled per layer.

    python convert_fp8_block_to_neuron_fp8.py --src <dir>/GLM-5.3 \\
        --dst <dir>/GLM-5.3-neuron-fp8 --workers 24
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX_TRN2 = 240.0
BLOCK = 128
EXPERT_CHUNK = 16


def dequant_block(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """fp8 ``[N, K]`` x fp32 ``[ceil(N/128), ceil(K/128)]`` -> fp32 ``[N, K]``."""
    n, k = w.shape
    full = s.repeat_interleave(BLOCK, 0)[:n].repeat_interleave(BLOCK, 1)[:, :k]
    return w.float() * full


def requant_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``[..., K]`` -> (fp8 ``[..., K]`` within +-240, fp32 scale ``[...]``)."""
    amax = w.abs().amax(dim=-1).clamp(min=1e-12)
    scale = amax / FP8_MAX_TRN2
    q = (w / scale.unsqueeze(-1)).clamp(-FP8_MAX_TRN2, FP8_MAX_TRN2)
    return q.to(torch.float8_e4m3fn), scale.float()


class Sources:
    def __init__(self, root: Path, weight_map: dict[str, str]):
        self.root, self.weight_map, self.handles = root, weight_map, {}

    def get(self, key: str) -> torch.Tensor:
        fname = self.weight_map[key]
        if fname not in self.handles:
            self.handles[fname] = safe_open(self.root / fname, framework="pt")
        return self.handles[fname].get_tensor(key)


def _plain(src: Sources, key: str) -> torch.Tensor:
    t = src.get(key)
    if t.dtype == torch.float8_e4m3fn:
        return dequant_block(t, src.get(key + "_scale_inv")).to(torch.bfloat16)
    return t


def convert_layer(job: tuple) -> tuple[str, dict, list, int]:
    root, dst, weight_map, layer, num_experts = job
    name = f"layer-{layer:03d}.safetensors" if layer >= 0 else "rest.safetensors"
    out_path = Path(dst) / name
    if out_path.exists():
        with safe_open(out_path, framework="pt") as f:
            return name, {k: name for k in f.keys()}, [], out_path.stat().st_size
    torch.set_num_threads(max(1, int(os.environ.get("CONVERT_THREADS", "6"))))
    src = Sources(Path(root), weight_map)
    prefix = f"model.layers.{layer}." if layer >= 0 else None
    keys = [
        k
        for k in weight_map
        if not k.endswith("_scale_inv")
        and (k.startswith(prefix) if prefix else not k.startswith("model.layers."))
    ]
    out: dict[str, torch.Tensor] = {}
    stats: list = []
    expert_keys = [k for k in keys if ".mlp.experts." in k]
    for k in keys:
        if k not in expert_keys:
            out[k] = _plain(src, k)
    if expert_keys:
        base = f"{prefix}mlp.experts."
        g0 = src.get(f"{base}0.gate_proj.weight")
        inter, hidden = g0.shape
        gate_up = torch.empty(num_experts, 2 * inter, hidden, dtype=torch.float8_e4m3fn)
        gate_up_s = torch.empty(num_experts, 2 * inter, dtype=torch.float32)
        down_t = torch.empty(num_experts, inter, hidden, dtype=torch.float8_e4m3fn)
        down_s = torch.empty(num_experts, hidden, dtype=torch.float32)
        for e in range(num_experts):
            p = f"{base}{e}."
            gu = torch.cat(
                (
                    dequant_block(src.get(p + "gate_proj.weight"), src.get(p + "gate_proj.weight_scale_inv")),
                    dequant_block(src.get(p + "up_proj.weight"), src.get(p + "up_proj.weight_scale_inv")),
                )
            )  # [2I, H]
            dn = dequant_block(src.get(p + "down_proj.weight"), src.get(p + "down_proj.weight_scale_inv"))  # [H, I]
            q, s = requant_rows(gu)
            gate_up[e], gate_up_s[e] = q, s
            qd, sd = requant_rows(dn)  # per output row of down: [H]
            down_t[e] = qd.view(torch.uint8).t().contiguous().view(torch.float8_e4m3fn)
            down_s[e] = sd
            if e % 64 == 0:
                for ref, qq, ss, what in ((gu, q, s, "gate_up"), (dn, qd, sd, "down")):
                    back = qq.float() * ss.unsqueeze(-1)
                    stats.append((f"{p}{what}", ((back - ref).norm() / ref.norm()).item()))
        out[f"{base}gate_up_proj"] = gate_up
        out[f"{base}gate_up_proj_scale"] = gate_up_s
        out[f"{base}down_proj_t"] = down_t
        out[f"{base}down_proj_scale"] = down_s
    tmp = out_path.with_suffix(".partial")
    save_file(out, str(tmp), metadata={"format": "pt"})
    tmp.rename(out_path)
    return name, {k: name for k in out}, stats, out_path.stat().st_size


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--layers", default=None, help="comma list, e.g. 0,3 (default: all)")
    args = p.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    config = json.loads((src / "config.json").read_text())
    weight_map = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    layers = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        else [-1] + list(range(config["num_hidden_layers"]))
    )
    jobs = [(str(src), str(dst), weight_map, l, config["n_routed_experts"]) for l in layers]
    t0 = time.time()
    index, all_stats, total = {}, [], 0
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        for name, keys, stats, nbytes in pool.imap_unordered(convert_layer, jobs):
            index.update(keys)
            all_stats += stats
            total += nbytes
            print(f"{name}: {len(keys)} tensors, {nbytes / 2**30:.1f} GiB ({time.time() - t0:.0f} s)", flush=True)
    if all_stats:
        errs = sorted(e for _, e in all_stats)
        print(f"requant error (block -> row), {len(errs)} samples: median {errs[len(errs) // 2]:.4f} max {errs[-1]:.4f}")
    if args.layers is None:
        config.pop("quantization_config", None)
        config["num_nextn_predict_layers"] = 0
        config["neuron_quantization"] = {
            "format": "fp8_row_experts",
            "routed_experts": "fp8_e4m3 per output row, |x| <= 240",
            "other": "bf16 (dequantised from fp8 128x128 blocks)",
            "source": "convert_fp8_block_to_neuron_fp8.py",
        }
        (dst / "config.json").write_text(json.dumps(config, indent=2))
        (dst / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total}, "weight_map": index}, indent=2)
        )
        for f in src.iterdir():
            if f.suffix in (".json", ".jinja", ".txt", ".model") and f.name not in (
                "config.json",
                "model.safetensors.index.json",
            ):
                shutil.copy(f, dst / f.name)
    print(f"done in {time.time() - t0:.0f} s, {total / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
