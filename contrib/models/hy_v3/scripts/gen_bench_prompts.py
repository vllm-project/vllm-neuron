# SPDX-License-Identifier: Apache-2.0
"""Fixed real-text benchmark prompts (sglang-style `random`) for bench_fixed.sh.

vLLM's `random` dataset decodes runs of consecutive token ids, so its prompts
are meaningless text; the MoE routing they produce is not representative, and
the routing-dependent decode kernels (`moe_tkg_hits*`) time differently on
them. This follows sglang/benchmark/datasets/random.py (`random_sample=True`)
instead: the first user turn of ShareGPT conversations with >= 2 turns,
shuffled, truncated to the target token count, or repeated and truncated when
shorter. The decoded text is then re-encoded and trimmed until it is exactly
the target length (decode/re-encode can merge tokens).

One file per (input length, concurrency) cell, max(8, 4 x concurrency) prompts
(as bench_matrix.sh), seeded with L * 1000 + C so cells never share
prefix-cache entries. The output is deterministic for a given tokenizer and
ShareGPT file, and gzip-ed with mtime 0, so regenerating gives identical bytes.

Usage:
    python gen_bench_prompts.py <tokenizer> <ShareGPT_V3_unfiltered_cleaned_split.json> \
        <out_dir> "<input_lens>" "<concurrencies>"
"""

import gzip
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer


def exact_text(tok, ids, n):
    for _ in range(8):
        text = tok.decode(ids[:n])
        re_ids = tok.encode(text, add_special_tokens=False)
        if len(re_ids) == n:
            return text
        ids = re_ids if len(re_ids) > n else re_ids + ids[len(re_ids) :]
    return None


def main():
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    with open(sys.argv[2]) as f:
        data = json.load(f)
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    lens = [int(x) for x in sys.argv[4].split()]
    concs = [int(x) for x in sys.argv[5].split()]
    convs = [d.get("conversations", d.get("conversation", [])) for d in data]
    firsts = [c[0]["value"] for c in convs if len(c) >= 2]
    for L in lens:
        for C in concs:
            n = max(8, 4 * C)
            rng = random.Random(L * 1000 + C)
            pool = firsts[:]
            rng.shuffle(pool)
            rows = []
            for p in pool:
                if len(rows) == n:
                    break
                ids = tok.encode(p, add_special_tokens=False)
                if not ids:
                    continue
                ids = (ids * ((L + len(ids) - 1) // len(ids)))[: L + 64]
                text = exact_text(tok, ids, L)
                if text is not None:
                    rows.append({"prompt": text})
            body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            path = out / f"in{L}_c{C}.jsonl.gz"
            with (
                open(path, "wb") as f,
                gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz,
            ):
                gz.write(body.encode())
            print(f"{path.name}: {len(rows)} prompts", flush=True)


if __name__ == "__main__":
    main()
