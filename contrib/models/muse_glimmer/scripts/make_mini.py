# SPDX-License-Identifier: Apache-2.0
"""Build a reduced MuseGlimmer checkpoint with real weights: the first N text and vision layers.

    python scripts/make_mini.py /path/to/Muse-Glimmer-30B /tmp/muse_glimmer_mini4 4

Keeps the released layer pattern (3 sliding + 1 full text layers per group of 4; 3 window +
1 full vision layers), the embeddings, the LM head, the vision adapter and the projection.
The tokenizer, chat template and processor files are copied. The result loads in HF
transformers and in vLLM, so the integration test can compare the two on real weights at a
size the HF FP32 reference can run on the host.
"""

import json
import os
import re
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file


def main(src: str, dst: str, n: int) -> None:
    if n % 4:
        raise SystemExit("N must be a multiple of 4 to keep the 3:1 layer pattern")
    os.makedirs(dst, exist_ok=True)
    cfg = json.load(open(os.path.join(src, "config.json")))
    for sub in ("text_config", "vision_config"):
        c = cfg[sub]
        c["num_hidden_layers"] = n
        c["layer_types"] = c["layer_types"][:n]
        if "layer_rope_theta" in c:
            c["layer_rope_theta"] = c["layer_rope_theta"][:n]
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)

    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    keep = {}
    layer = re.compile(r"\.layers\.(\d+)\.")
    for name, shard in index.items():
        m = layer.search(name)
        if m is None or int(m.group(1)) < n:
            keep.setdefault(shard, []).append(name)
    tensors = {}
    for shard, names in keep.items():
        with safe_open(os.path.join(src, shard), "pt") as f:
            for name in names:
                tensors[name] = f.get_tensor(name)
    save_file(tensors, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
    for f in os.listdir(src):
        if f.endswith((".jinja", ".txt")) or f in (
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "processor_config.json", "preprocessor_config.json", "video_preprocessor_config.json",
            "generation_config.json",
        ):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    print(f"wrote {len(tensors)} tensors ({n} text + {n} vision layers) to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
