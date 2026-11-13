"""Write a BF16 (dequantized) copy of the full Hy3-FP8 checkpoint for HF calibration.

Same transform as make_mini.py's ``bf16`` variant (``w_bf16 = w_fp8 * weight_scale``;
scale tensors dropped; no quantization_config), for all decoder layers and
without the FP8 copy. The MTP layer is dropped (num_nextn_predict_layers = 0).
Only used as the forward model for KV calibration; serving keeps the W8A8 files.

Usage: python -I dequant_bf16.py <src_ckpt> <out_dir> [workers=16]
"""

import json
import os
import re
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, out = sys.argv[1], sys.argv[2]
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
cfg = json.loads(Path(f"{src}/config.json").read_text())
n_layers = cfg["num_hidden_layers"]
wm = json.loads(Path(f"{src}/model.safetensors.index.json").read_text())["weight_map"]


def keep(k):
    m = re.match(r"model\.layers\.(\d+)\.", k)
    return m is None or int(m.group(1)) < n_layers


by_file = {}
for k in sorted(k for k in wm if keep(k)):
    by_file.setdefault(wm[k], []).append(k)


def convert(job):
    i, f, ks = job
    with safe_open(f"{src}/{f}", "pt") as h:
        t = {k: h.get_tensor(k) for k in ks}
    # Scales may live in another shard than their weight.
    out_t = {}
    for k, v in t.items():
        if k.endswith((".weight_scale", ".input_scale")):
            continue
        if v.dtype == torch.float8_e4m3fn:
            sk = k[: -len("weight")] + "weight_scale"
            if sk in t:
                s = t[sk]
            else:
                with safe_open(f"{src}/{wm[sk]}", "pt") as h2:
                    s = h2.get_tensor(sk)
            out_t[k] = (v.float() * s.float()).to(torch.bfloat16)
        else:
            out_t[k] = v
    name = f"model-{i:05d}.safetensors"
    save_file(out_t, f"{out}/{name}", metadata={"format": "pt"})
    return name, list(out_t)


if __name__ == "__main__":
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(src):
        if (
            f.endswith((".json", ".jinja", ".txt", ".model", ".tiktoken"))
            and not f.startswith("model.safetensors")
            and f != "config.json"
        ):
            shutil.copy(f"{src}/{f}", out)
    c = dict(cfg)
    c.pop("quantization_config", None)
    c["num_nextn_predict_layers"] = 0
    Path(f"{out}/config.json").write_text(json.dumps(c, indent=2))
    jobs = [(i, f, ks) for i, (f, ks) in enumerate(sorted(by_file.items()))]
    index = {}
    with Pool(workers) as pool:
        for name, keys in pool.imap_unordered(convert, jobs):
            index.update({k: name for k in keys})
            print(f"wrote {name}: {len(keys)} tensors", flush=True)
    Path(f"{out}/model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": index}, indent=1)
    )
    print("done")
