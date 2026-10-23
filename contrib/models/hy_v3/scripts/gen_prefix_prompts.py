# SPDX-License-Identifier: Apache-2.0
"""Fixed real-text shared-prefix prompts for bench_prefix_fixed.sh.

Per concurrency C: one shared prefix of exactly P tokens (ShareGPT
conversations joined into one long document, ending in a blank line) and
10 x C requests of exactly L tokens = prefix + a distinct suffix (the first
user turn of another ShareGPT conversation, cut or repeated to L - P tokens).
Every prompt is checked to tokenize to exactly L tokens with its first P
tokens equal to the prefix, so the prefix-cache hit is exact; samples whose
suffix merges with the prefix at the boundary are dropped. Row 0 of each file
is a warm-up request (same prefix, its own suffix) that fills the cache and is
not measured. The prefix is stored once (line 0: {"prefix": ...}); each
following line is {"suffix": ...} and the prompt is prefix + suffix. Each C has its own prefix (seed), so cells never hit each
other. P should be a multiple of the prefix-cache block (block_size x DCP,
256 tokens for the 256k config). Deterministic, gzip-ed with mtime 0.

Usage:
    python gen_prefix_prompts.py <tokenizer> <ShareGPT_V3_unfiltered_cleaned_split.json> \
        <out_dir> <total_len> <prefix_len> "<concurrencies>" [requests_per_conc=10]
Writes in{L}_p{P}_c{C}.jsonl.gz.
"""

import gzip
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

SEP = "\n\n"


def exact_text(tok, ids, n):
    """Text that re-encodes to exactly n tokens, from the token ids `ids`."""
    for _ in range(8):
        text = tok.decode(ids[:n])
        re_ids = tok.encode(text, add_special_tokens=False)
        if len(re_ids) == n:
            return text
        ids = re_ids if len(re_ids) > n else re_ids + ids[len(re_ids) :]
    return None


def build_prefix(tok, convs, n):
    """A document of n tokens ending in SEP, from whole conversations."""
    parts, total = [], 0
    for c in convs:
        turns = [t["value"] for t in c if t.get("value")]
        if not turns:
            continue
        parts.append(SEP.join(turns))
        total += len(tok.encode(parts[-1], add_special_tokens=False))
        if total > n + 1024:
            break
    sep_len = len(tok.encode(SEP, add_special_tokens=False))
    body_ids = tok.encode(SEP.join(parts), add_special_tokens=False)
    for k in range(n - sep_len, n - sep_len - 64, -1):
        text = tok.decode(body_ids[:k]).rstrip() + SEP
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) == n:
            return text, ids
    raise RuntimeError("no prefix of the exact length")


def build_prompt(tok, prefix_text, prefix_ids, question, total):
    """prefix + question cut / repeated so the prompt is exactly total tokens."""
    q_ids = tok.encode(question, add_special_tokens=False)
    if not q_ids:
        return None
    s = total - len(prefix_ids)
    q_ids = (q_ids * ((s + len(q_ids) - 1) // len(q_ids)))[: s + 64]
    suffix = exact_text(tok, q_ids, s)
    if suffix is None:
        return None
    for _ in range(4):
        prompt = prefix_text + suffix
        ids = tok.encode(prompt, add_special_tokens=False)
        if ids[: len(prefix_ids)] != prefix_ids:
            return None
        if len(ids) == total:
            return prompt
        s += total - len(ids)
        suffix = exact_text(tok, q_ids, s)
        if suffix is None:
            return None
    return None


def main():
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    with open(sys.argv[2]) as f:
        data = json.load(f)
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    total, prefix_len = int(sys.argv[4]), int(sys.argv[5])
    concs = [int(x) for x in sys.argv[6].split()]
    per_conc = int(sys.argv[7]) if len(sys.argv) > 7 else 10
    convs = [d.get("conversations", d.get("conversation", [])) for d in data]
    convs = [c for c in convs if len(c) >= 2]
    for C in concs:
        rng = random.Random(total * 1000 + prefix_len + C)
        pool = convs[:]
        rng.shuffle(pool)
        # First half for prefixes, second half for suffixes: never the same text.
        half = len(pool) // 2
        prefix_text, prefix_ids = build_prefix(tok, pool[:half], prefix_len)
        rows = []
        for c in pool[half:]:
            if len(rows) == per_conc * C + 1:
                break
            prompt = build_prompt(tok, prefix_text, prefix_ids, c[0]["value"], total)
            if prompt is not None:
                rows.append({"suffix": prompt[len(prefix_text) :]})
        rows[0]["warmup"] = True
        rows.insert(0, {"prefix": prefix_text})
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        path = out / f"in{total}_p{prefix_len}_c{C}.jsonl.gz"
        with (
            open(path, "wb") as f,
            gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz,
        ):
            gz.write(body.encode())
        print(f"{path.name}: 1 warm-up + {len(rows) - 2} requests", flush=True)


if __name__ == "__main__":
    main()
