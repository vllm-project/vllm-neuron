"""Build a reduced Hy3 checkpoint (first N decoder layers) from tencent/Hy3-FP8.

Writes two copies:
  <out>/fp8   - original FP8 tensors/scales (for vLLM Neuron)
  <out>/bf16  - weights dequantized to BF16, no quantization_config (HF reference)

Usage: python -I make_mini.py <src_ckpt> <out_dir> <num_layers>
"""

import json
import os
import re
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, out, n_layers = sys.argv[1], sys.argv[2], int(sys.argv[3])
wm = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]


def keep(k):
    m = re.match(r"model\.layers\.(\d+)\.", k)
    return m is None or int(m.group(1)) < n_layers


keys = sorted(k for k in wm if keep(k))
by_file = {}
for k in keys:
    by_file.setdefault(wm[k], []).append(k)

cfg = json.load(open(f"{src}/config.json"))
cfg["num_hidden_layers"] = n_layers
cfg["num_nextn_predict_layers"] = 0
if "mlp_layer_types" in cfg:
    cfg["mlp_layer_types"] = cfg["mlp_layer_types"][:n_layers]

for variant in ("fp8", "bf16"):
    d = f"{out}/{variant}"
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith(
            (".json", ".jinja", ".txt", ".model", ".tiktoken")
        ) and not f.startswith("model.safetensors"):
            if f != "config.json":
                shutil.copy(f"{src}/{f}", d)
    c = dict(cfg)
    if variant == "bf16":
        c.pop("quantization_config", None)
    json.dump(c, open(f"{d}/config.json", "w"), indent=2)

index = {"fp8": {}, "bf16": {}}
for i, (f, ks) in enumerate(sorted(by_file.items())):
    fp8_tensors, bf16_tensors = {}, {}
    with safe_open(f"{src}/{f}", "pt") as h:
        for k in ks:
            fp8_tensors[k] = h.get_tensor(k)
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
    json.dump(
        {"metadata": {}, "weight_map": index[variant]},
        open(f"{out}/{variant}/model.safetensors.index.json", "w"),
        indent=1,
    )
print("done")
