"""LongBench v2 multiple choice (THUDM/LongBench-v2), no chain of thought, through the OpenAI API.

fetch:   download THUDM/LongBench-v2 to a .jsonl.gz (needs `datasets`).
build:   tokenize every item with the official no-CoT template (chat
         template applied) and keep those of at most --max-tokens tokens,
         --num of them spread over the lengths; writes a .jsonl.gz.
run:     greedy answers; writes <out.json> with the parsed letter per item.
compare: accuracy of each run, and how often a run picks the same letter as
         the reference run (paired, more sensitive than the accuracy).

Usage:
    python longbench_v2.py fetch <raw.jsonl.gz>
    python longbench_v2.py build <tokenizer> <raw.jsonl.gz> <items.jsonl.gz> [--max-tokens 82000] [--num 100]
    python longbench_v2.py run <items.jsonl.gz> <out.json> [base_url]
    python longbench_v2.py compare <ref.json> <run.json> [<run.json> ...]
"""

import argparse
import gzip
import json
import re
import sys

TEMPLATE = (
    "Please read the following text and answer the question below.\n\n<text>\n{context}\n</text>\n\n"
    "What is the correct answer to this question: {question}\nChoices:\n(A) {choice_A}\n(B) {choice_B}\n"
    "(C) {choice_C}\n(D) {choice_D}\n\n"
    'Format your response as follows: "The correct answer is (insert answer here)".'
)


def fetch(out):
    from datasets import load_dataset

    with gzip.open(out, "wt") as f:
        for r in load_dataset("THUDM/LongBench-v2", split="train"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build(tokenizer, raw, out, max_tokens, num):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    rows = []
    with gzip.open(raw, "rt") as f:
        items = [json.loads(line) for line in f]
    for r in items:
        prompt = TEMPLATE.format(**r)
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
        n = len(ids["input_ids"] if hasattr(ids, "keys") else ids)
        if n <= max_tokens:
            rows.append({"id": r["_id"], "tokens": n, "difficulty": r["difficulty"],
                         "domain": r["domain"], "answer": r["answer"], "prompt": prompt})
    rows.sort(key=lambda r: r["tokens"])
    step = max(1, len(rows) / num)
    pick = [rows[int(i * step)] for i in range(min(num, len(rows)))]
    with gzip.open(out, "wt") as f:
        for r in pick:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    toks = [r["tokens"] for r in pick]
    print(f"{len(rows)} items <= {max_tokens} tokens; wrote {len(pick)}: tokens {min(toks)}-{max(toks)}, "
          f"median {sorted(toks)[len(toks) // 2]}")


def parse(text):
    m = re.search(r"correct answer is \(?([ABCD])\)?", text) or re.search(r"\(([ABCD])\)", text)
    return m.group(1) if m else None


def run(items, out, base_url="http://127.0.0.1:8000"):
    from openai import OpenAI

    client = OpenAI(base_url=base_url + "/v1", api_key="EMPTY", timeout=3600)
    with gzip.open(items, "rt") as f:
        rows = [json.loads(line) for line in f]
    res, ok = [], 0
    for i, r in enumerate(rows):
        c = client.chat.completions.create(
            model="hy3", messages=[{"role": "user", "content": r["prompt"]}],
            max_tokens=32, temperature=0.0)
        text = c.choices[0].message.content or ""
        pred = parse(text)
        ok += pred == r["answer"]
        res.append({"id": r["id"], "tokens": r["tokens"], "answer": r["answer"], "pred": pred, "text": text})
        print(i, r["tokens"], r["answer"], pred, f"acc {ok}/{i + 1}", flush=True)
        with open(out, "w") as f:  # after every item: a crashed server keeps the answers so far
            json.dump(res, f, ensure_ascii=False, indent=1)


def compare(ref_path, *paths):
    with open(ref_path) as f:
        ref = {r["id"]: r for r in json.load(f)}
    for p in [ref_path, *paths]:
        with open(p) as f:
            got = json.load(f)
        acc = sum(r["pred"] == r["answer"] for r in got) / len(got)
        same = sum(r["pred"] == ref[r["id"]]["pred"] for r in got) / len(got)
        print(f"{p}: accuracy {100 * acc:.1f}% ({len(got)} items); same answer as the reference {100 * same:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "fetch":
        fetch(*rest)
    elif cmd == "build":
        ap = argparse.ArgumentParser()
        ap.add_argument("tokenizer")
        ap.add_argument("raw")
        ap.add_argument("out")
        ap.add_argument("--max-tokens", type=int, default=82000)
        ap.add_argument("--num", type=int, default=100)
        a = ap.parse_args(rest)
        build(a.tokenizer, a.raw, a.out, a.max_tokens, a.num)
    elif cmd == "run":
        run(*rest)
    elif cmd == "compare":
        compare(*rest)
    else:
        sys.exit(__doc__)
