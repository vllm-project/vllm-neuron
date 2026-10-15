# SPDX-License-Identifier: Apache-2.0
"""Rewrite ``tencent/Hy4-preview-FP8`` (ModelOpt MXFP8) into what Trn2 can run.

The published FP8 checkpoint is **MXFP8**: ``float8_e4m3fn`` values with one
``e8m0`` exponent per 32 consecutive input elements, stored next to each weight
as ``<name>_scale`` (``uint8``). Trn2 has no MX matmul -- the MX kernels in this
plugin are Trn3-only -- and its fp8 is the legacy ``e4m3`` that tops out at
**240**, not the OCP ``e4m3fn`` 448. So the checkpoint is rewritten once, on the
host, into two kinds of tensor:

* **Routed experts** (``mlp.experts.gate_up_proj`` / ``mlp.experts.down_proj``)
  stay fp8, requantised to one scale per *output row* inside the 240 range:
  ``gate_up_proj [E, 2I, H]`` gets ``gate_up_proj_scale [E, 2I]`` and
  ``down_proj [E, H, I]`` gets ``down_proj_scale [E, H]``; the down weight is
  also stored transposed, as ``down_proj_t [E, I, H]``, so that a rank's shard
  of ``I`` is a contiguous read. That is the "FP8 row"
  layout ``moe_cte``'s ``shard_on_block`` and ``moe_tkg`` take on Trn2, and per
  output row means a TP shard of the *input* dim keeps its scale unchanged. The
  experts are 95% of the model; this is what lets 770B fit one trn2.48xlarge.
* **Everything else that was MXFP8** is dequantised to bf16. The product of an
  e4m3 value (3 mantissa bits) and a power of two is exactly representable in
  bf16, so this step is lossless. It costs ~36 GB on disk and, since most of it
  shards 64 ways, little per rank.

Tensors that were never quantised (norms, router, sinks, the iHC parameters,
embeddings, lm_head, the gate projection) are copied through untouched.

``config.json`` loses its ``quantization_config`` -- vLLM would otherwise try to
build a ModelOpt MXFP8 method it has no Neuron kernel for -- and gains a
``neuron_quantization`` block that ``vllm_neuron/model/hy_v4`` checks for, so
the original MXFP8 directory is rejected with a pointer back here rather than
loading scales as weights.

Requantising MX to per-row loses precision wherever one row spans several MX
exponents; the script reports the relative error it introduces (on a sample of
experts per layer) so that cost is a number, not a guess.

Usage::

    python convert_mxfp8_to_neuron_fp8.py \\
        --src /mnt/nvme/models/Hy4-preview-FP8 \\
        --dst /mnt/nvme/models/Hy4-preview-neuron-fp8 --workers 24
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
MX_BLOCK = 32
EXPERT_SUFFIXES = (".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")
# Experts are processed this many at a time: one layer's gate_up_proj is
# 256 x 4096 x 6144, which is 25 GB once widened to fp32.
EXPERT_CHUNK = 16


def _dequant_mx(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """MXFP8 ``(fp8 [..., K], e8m0 [..., K/32])`` -> fp32 ``[..., K]``."""
    *lead, k = weight.shape
    if scale.shape[-1] * MX_BLOCK != k:
        raise ValueError(
            f"MX scale {tuple(scale.shape)} does not tile weight {tuple(weight.shape)} "
            f"in blocks of {MX_BLOCK}"
        )
    w = weight.float().view(*lead, k // MX_BLOCK, MX_BLOCK)
    s = torch.exp2(scale.float() - 127.0).unsqueeze(-1)
    return (w * s).view(*lead, k)


def _requant_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``[..., K]`` -> (fp8 ``[..., K]``, fp32 scale ``[...]``), one scale per row.

    The values stay below 240, where ``e4m3`` and ``e4m3fn`` encode identically,
    so the bytes are stored as torch's ``float8_e4m3fn`` and Trn2 reads them as
    its own ``e4m3`` with no conversion.
    """
    amax = w.abs().amax(dim=-1).clamp(min=1e-12)
    scale = amax / FP8_MAX_TRN2
    q = (w / scale.unsqueeze(-1)).clamp(-FP8_MAX_TRN2, FP8_MAX_TRN2)
    return q.to(torch.float8_e4m3fn), scale.float()


class _Sources:
    """Lazily opened ``safe_open`` handles, for scales stored in another shard."""

    def __init__(self, root: Path, weight_map: dict[str, str]):
        self.root = root
        self.weight_map = weight_map
        self.handles = {}

    def get(self, key: str) -> torch.Tensor:
        fname = self.weight_map[key]
        if fname not in self.handles:
            self.handles[fname] = safe_open(self.root / fname, framework="pt")
        return self.handles[fname].get_tensor(key)


def _convert_expert(
    key: str, weight: torch.Tensor, scale: torch.Tensor, stats: list
) -> tuple[torch.Tensor, torch.Tensor]:
    q_out = torch.empty(weight.shape, dtype=torch.float8_e4m3fn)
    s_out = torch.empty(weight.shape[:-1], dtype=torch.float32)
    for first in range(0, weight.shape[0], EXPERT_CHUNK):
        last = min(first + EXPERT_CHUNK, weight.shape[0])
        ref = _dequant_mx(weight[first:last], scale[first:last])
        q, s = _requant_rows(ref)
        q_out[first:last] = q
        s_out[first:last] = s
        if first == 0:
            # Error of the requantisation alone, measured on the first chunk.
            back = q.float() * s.unsqueeze(-1)
            rel = ((back - ref).norm() / ref.norm().clamp(min=1e-12)).item()
            stats.append((key, rel))
    return q_out, s_out


def _existing(path: Path) -> tuple[dict[str, str], int] | None:
    """Keys and size of an already-written shard, or None if it is absent or torn."""
    try:
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())
            nbytes = 0
            for k in keys:
                s = f.get_slice(k)
                n = 1
                for d in s.get_shape():
                    n *= d
                nbytes += n * {"F8_E4M3": 1, "BF16": 2, "F32": 4, "U8": 1}.get(
                    s.get_dtype(), 4
                )
        return {k: path.name for k in keys}, nbytes
    except Exception:
        return None


def convert_shard(args: tuple) -> tuple[str, dict[str, str], list, int]:
    src, dst, fname, weight_map = args
    done = _existing(Path(dst) / fname)
    if done is not None:
        return fname, done[0], [], done[1]
    torch.set_num_threads(max(1, int(os.environ.get("CONVERT_THREADS", "8"))))
    sources = _Sources(Path(src), weight_map)
    out: dict[str, torch.Tensor] = {}
    stats: list = []
    with safe_open(Path(src) / fname, framework="pt") as f:
        for key in f.keys():
            if key.endswith("_scale") and key[: -len("_scale")] in weight_map:
                continue  # consumed with its weight
            tensor = f.get_tensor(key)
            scale_key = key + "_scale"
            if tensor.dtype != torch.float8_e4m3fn:
                out[key] = tensor
                continue
            if scale_key not in weight_map:
                raise KeyError(f"{key} is fp8 but has no {scale_key}")
            scale = sources.get(scale_key)
            if key.endswith(EXPERT_SUFFIXES):
                q, s = _convert_expert(key, tensor, scale, stats)
                if key.endswith(".down_proj"):
                    # [E, H, I] -> [E, I, H]: a TP shard of I becomes a row slice,
                    # instead of E * H 32-element reads or every rank reading
                    # the whole tensor to keep 1/64 of it.
                    key, q = (
                        key + "_t",
                        q.view(torch.uint8)
                        .transpose(1, 2)
                        .contiguous()
                        .view(torch.float8_e4m3fn),
                    )
                out[key], out[scale_key] = q, s
            else:
                out[key] = _dequant_mx(tensor, scale).to(torch.bfloat16)
    # Written under a temporary name and renamed, so an interrupted run leaves
    # no torn shard for the resume check above to accept.
    tmp = Path(dst) / (fname + ".partial")
    save_file(out, str(tmp), metadata={"format": "pt"})
    tmp.rename(Path(dst) / fname)
    nbytes = sum(t.numel() * t.element_size() for t in out.values())
    return fname, {k: fname for k in out}, stats, nbytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="convert only these shard filenames (for a quick check)",
    )
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]

    config = json.loads((src / "config.json").read_text())
    quant = config.pop("quantization_config", None)
    if quant is None or quant.get("quantization", {}).get("quant_algo") != "MXFP8":
        raise SystemExit(f"{src} is not the ModelOpt MXFP8 checkpoint: {quant}")
    config["neuron_quantization"] = {
        "format": "fp8_row_experts",
        "fp8_max": FP8_MAX_TRN2,
        "source": "modelopt-mxfp8",
    }
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for path in src.iterdir():
        if (
            path.is_file()
            and not path.name.endswith(".safetensors")
            and path.name
            not in (
                "config.json",
                "model.safetensors.index.json",
            )
        ):
            shutil.copy2(path, dst / path.name)

    shards = sorted(set(weight_map.values()))
    if args.only:
        shards = [s for s in shards if s in set(args.only)]
    todo = [(str(src), str(dst), s, weight_map) for s in shards]

    new_map: dict[str, str] = {}
    all_stats: list = []
    total = 0
    start = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, (fname, keys, stats, nbytes) in enumerate(
            pool.imap_unordered(convert_shard, todo), 1
        ):
            new_map.update(keys)
            all_stats.extend(stats)
            total += nbytes
            print(
                f"[{i}/{len(todo)}] {fname} {nbytes / 2**30:.1f} GiB "
                f"({time.time() - start:.0f}s)",
                flush=True,
            )

    if not args.only:
        (dst / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": total}, "weight_map": new_map}, indent=2
            )
            + "\n"
        )

    if all_stats:
        rels = sorted(r for _, r in all_stats)
        worst = sorted(all_stats, key=lambda kv: kv[1])[-3:]
        print(
            f"expert requant rel. error over {len(rels)} tensors: "
            f"median {rels[len(rels) // 2]:.4f}, max {rels[-1]:.4f}"
        )
        for key, rel in worst:
            print(f"  worst: {key} {rel:.4f}")
    print(f"done: {total / 2**30:.1f} GiB in {time.time() - start:.0f}s -> {dst}")


if __name__ == "__main__":
    main()
