"""Where check_mtp.py's plain and MTP runs differ: HF FP32 log-probs of both tokens.

Greedy MTP matches plain decoding up to numerics, so the first differing token
should be a near-tie in an independent reference: both choices among the HF
top tokens, a few tenths of a nat apart (a defect shows gaps of several nats).
Usage: python -I check_mtp_gaps.py <BF16 checkpoint> <ids.json from DUMP_IDS>
(the BF16 copy of the same checkpoint, e.g. make_mini.py --keep-mtp output).
"""

import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt, path = sys.argv[1], sys.argv[2]
with open(path) as f:
    d = json.load(f)
torch.set_num_threads(96)
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.float32).eval()
for i, (p, a, b) in enumerate(zip(d["prompts"], d["plain"], d["mtp"])):
    j = next((j for j, (u, v) in enumerate(zip(a, b)) if u != v), None)
    if j is None:
        print(f"prompt {i}: identical"); continue
    ids = tok(p, return_tensors="pt").input_ids[0].tolist() + a[:j]
    with torch.no_grad():
        lg = model(torch.tensor([ids])).logits[0, -1].float().log_softmax(-1)
    top = lg.topk(3)
    print(f"prompt {i}: diverges at {j}: plain {a[j]} ({lg[a[j]]:.3f}), mtp {b[j]} ({lg[b[j]]:.3f}); "
          f"HF top-3 {top.indices.tolist()} {[round(x, 3) for x in top.values.tolist()]}; "
          f"plain-mtp gap {lg[a[j]] - lg[b[j]]:.3f} nats", flush=True)
