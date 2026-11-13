"""HF transformers CPU reference for a (reduced) Hy3 BF16 checkpoint.

Usage: python -I hf_reference.py <bf16_ckpt_dir> <out.pt> [dtype=float32]
Env: HY3_PROMPTS=<json list of prompt strings> overrides the default prompts.
"""

import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "人工智能的未来发展趋势是",
    "In 1969, Neil Armstrong became the first person to",
]
PROMPTS = (
    json.loads(Path(os.environ["HY3_PROMPTS"]).read_text())
    if os.environ.get("HY3_PROMPTS")
    else DEFAULT_PROMPTS
)
GEN_TOKENS = 8

ckpt, out = sys.argv[1], sys.argv[2]
dtype = getattr(torch, sys.argv[3]) if len(sys.argv) > 3 else torch.float32
torch.set_num_threads(96)
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=dtype).eval()
print(
    type(model).__name__,
    "moe_fp32_combine:",
    model.config.enable_moe_fp32_combine,
    flush=True,
)

results = []
with torch.no_grad():
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        gen = model.generate(
            ids,
            max_new_tokens=GEN_TOKENS,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
        )
        logits = torch.stack(gen.logits, 1)[0].float()  # [GEN, V]
        new = gen.sequences[0, ids.shape[1] :].tolist()
        results.append(
            {
                "prompt": p,
                "prompt_ids": ids[0].tolist(),
                "gen_ids": new,
                "logits": logits,
            }
        )
        print(repr(p), "->", new, repr(tok.decode(new)), flush=True)
torch.save(results, out)
