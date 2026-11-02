"""Build a reduced Hy3 checkpoint (first N decoder layers) from tencent/Hy3-FP8.

Writes two copies:
  <out>/fp8   - original FP8 tensors/scales (for vLLM Neuron)
  <out>/bf16  - weights dequantized to BF16, no quantization_config (HF reference)

Usage: python -I make_mini.py <src_ckpt> <out_dir> <num_layers> [--keep-mtp]

--keep-mtp also copies the MTP layer (model.layers.<N_full>.*) as layer
<num_layers>, so the reduced checkpoint can serve MTP speculative decoding.
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, out, n_layers = sys.argv[1], sys.argv[2], int(sys.argv[3])
keep_mtp = "--keep-mtp" in sys.argv[4:]
wm = json.loads(Path(f"{src}/model.safetensors.index.json").read_text())["weight_map"]
full_cfg = json.loads(Path(f"{src}/config.json").read_text())
mtp_idx = full_cfg["num_hidden_layers"]  # the MTP layer follows the decoder stack


def keep(k):
    m = re.match(r"model\.layers\.(\d+)\.", k)
    if m is None:
        return True
    idx = int(m.group(1))
    return idx < n_layers or (keep_mtp and idx == mtp_idx)


def rename(k):
    """Move the MTP layer right after the reduced stack."""
    if keep_mtp:
        return re.sub(rf"^model\.layers\.{mtp_idx}\.", f"model.layers.{n_layers}.", k)
    return k


keys = sorted(k for k in wm if keep(k))
by_file = {}
for k in keys:
    by_file.setdefault(wm[k], []).append(k)

cfg = json.loads(Path(f"{src}/config.json").read_text())
cfg["num_hidden_layers"] = n_layers
cfg["num_nextn_predict_layers"] = 1 if keep_mtp else 0
if "mlp_layer_types" in cfg:
    cfg["mlp_layer_types"] = cfg["mlp_layer_types"][:n_layers]

for variant in ("fp8", "bf16"):
    d = f"{out}/{variant}"
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(src):
        if (
            f.endswith((".json", ".jinja", ".txt", ".model", ".tiktoken"))
            and not f.startswith("model.safetensors")
            and f != "config.json"
        ):
            shutil.copy(f"{src}/{f}", d)
    c = dict(cfg)
    if variant == "bf16":
        c.pop("quantization_config", None)
    Path(f"{d}/config.json").write_text(json.dumps(c, indent=2))

index = {"fp8": {}, "bf16": {}}
for i, (f, ks) in enumerate(sorted(by_file.items())):
    fp8_tensors, bf16_tensors = {}, {}
    with safe_open(f"{src}/{f}", "pt") as h:
        for k in ks:
            fp8_tensors[rename(k)] = h.get_tensor(k)
    for k, t in fp8_tensors.items():
        if k.endswith((".weight_scale", ".input_scale")):
            continue
        if t.dtype == torch.float8_e4m3fn:
            s = fp8_tensors[k[: -len("weight")] + "weight_scale"].float()
            bf16_tensors[k] = (t.float() * s).to(torch.bfloat16)
        else:
            bf16_tensors[k] = t
    name = f"model-{i:05d}.safetensors"
    save_file(fp8_tensors, f"{out}/fp8/{name}", metadata={"format": "pt"})
    save_file(bf16_tensors, f"{out}/bf16/{name}", metadata={"format": "pt"})
    index["fp8"].update({k: name for k in fp8_tensors})
    index["bf16"].update({k: name for k in bf16_tensors})
    print(f"wrote {name}: {len(fp8_tensors)} tensors", flush=True)

for variant in ("fp8", "bf16"):
    Path(f"{out}/{variant}/model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": index[variant]}, indent=1)
    )
print("done")
