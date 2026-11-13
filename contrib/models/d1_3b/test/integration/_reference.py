# SPDX-License-Identifier: Apache-2.0
"""fp32 CPU reference for test_model.py, run as a separate process.

vLLM forks its engine processes. If the test process has already run torch CPU
ops (the fp32 reference), the forked engine can deadlock the first time it runs
torch on the CPU (the HF image processor during multimodal profiling): an OpenMP
thread pool does not survive ``fork``. Running the reference here keeps the test
process free of torch CPU work before vLLM starts, and frees its memory on exit.

Usage: python _reference.py <request.json> <result.json>
  request: {"model_path", "text": [{"name", "state", "spec"}],
            "images": [{"name", "size", "spec"}]}
  result:  {"text": [{"name", "ids", "probs"}],
            "images": [{"name", "text", "probs"}]}
"""

from __future__ import annotations

import importlib
import json
import sys


def test_image(size: tuple[int, int]):
    """A deterministic picture: a red circle, a blue square and a caption."""
    from PIL import Image, ImageDraw

    w, h = size
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.ellipse([w * 0.1, h * 0.2, w * 0.45, h * 0.65], fill="red")
    d.rectangle([w * 0.55, h * 0.3, w * 0.9, h * 0.8], fill="blue")
    d.text((w * 0.1, h * 0.85), "order #1042", fill="black")
    return im


def main(req_path: str, out_path: str) -> None:
    import torch
    from transformers import AutoModel

    req = json.load(open(req_path))
    model = AutoModel.from_pretrained(
        req["model_path"], trust_remote_code=True, dtype=torch.float32
    )
    engine = model.engine
    pkg = type(model).__module__.rsplit(".", 1)[0]
    prompt = importlib.import_module(f"{pkg}.prompt")
    runner = importlib.import_module(f"{pkg}.runner")

    out = {"text": [], "images": []}
    for c in req["text"]:
        q = prompt.as_question(c["spec"])
        ids = engine.tokenizer.encode(
            engine.render(c["state"], q), add_special_tokens=False
        )
        probs = engine.probabilities(c["state"], [q])[0]
        out["text"].append({"name": c["name"], "ids": ids, "probs": probs})

    for c in req["images"]:
        # One question over one picture, rendered as SystemOne._request does.
        pics = [runner.cap_pixels(test_image(tuple(c["size"])))]
        q = prompt.as_question(c["spec"])
        prefix = prompt.prefix_text(
            engine.tokenizer,
            None,
            engine.bos,
            engine.state_style,
            engine.system,
            engine._image_markup(len(pics)),
        )
        text = prefix + prompt.suffix_text(
            engine.tokenizer, q, engine.lead, engine.option_style
        )
        probs = engine.probabilities(None, [q], images=pics)[0]
        out["images"].append({"name": c["name"], "text": text, "probs": probs})

    json.dump(out, open(out_path, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
