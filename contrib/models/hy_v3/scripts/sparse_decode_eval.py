"""Accuracy check of the EXPERIMENTAL block-sparse decode (HY3_SPARSE_DECODE).

run:     greedy 64-token continuations of long real-text prompts (the 10
         85k shared-prefix prompts of bench_data/prefix and 4 + 4 ShareGPT
         prompts of 32k / 64k tokens), saved to a JSON file.
compare: each run against a reference run (dense decode): exact matches and
         the length of the common prefix of the generated text.

Usage:
    python sparse_decode_eval.py run <out.json> [base_url]
    python sparse_decode_eval.py compare <dense.json> <run.json> [<run.json> ...]
"""

import gzip
import json
import sys
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent


def prompts():
    with gzip.open(HERE.parent / "bench_data/prefix/in85000_p78336_c1.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    out = [("85k", rows[0]["prefix"] + r["suffix"]) for r in rows[1:]]
    for cell in ("in32768_c1", "in65536_c1"):
        with gzip.open(HERE.parent / f"bench_data/prompts/{cell}.jsonl.gz", "rt") as f:
            out += [(cell[2:7], json.loads(line)["prompt"]) for line in list(f)[:4]]
    return out


def run(out, base_url="http://127.0.0.1:8000"):
    client = OpenAI(base_url=base_url + "/v1", api_key="EMPTY", timeout=3600)
    res = []
    for i, (kind, p) in enumerate(prompts()):
        r = client.completions.create(model="hy3", prompt=p, max_tokens=64, temperature=0.0)
        res.append({"i": i, "kind": kind, "text": r.choices[0].text})
        print(i, kind, repr(r.choices[0].text[:60]), flush=True)
    with open(out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


def compare(ref_path, *paths):
    with open(ref_path) as f:
        ref = json.load(f)
    for p in paths:
        with open(p) as f:
            got = json.load(f)
        same, pref = 0, []
        for a, b in zip(ref, got):
            n = next((i for i, (x, y) in enumerate(zip(a["text"], b["text"])) if x != y),
                     min(len(a["text"]), len(b["text"])))
            same += a["text"] == b["text"]
            pref.append(n / max(1, len(a["text"])))
        print(f"{p}: {same}/{len(ref)} identical; common prefix {100 * sum(pref) / len(pref):.0f}% "
              f"of the dense text on average (min {100 * min(pref):.0f}%)")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(*sys.argv[2:])
    elif sys.argv[1] == "compare":
        compare(*sys.argv[2:])
    else:
        sys.exit(__doc__)
