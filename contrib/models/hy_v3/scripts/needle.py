# SPDX-License-Identifier: Apache-2.0
"""Long-context passkey retrieval (needle in a haystack) through the OpenAI API.

A random 5-digit pass key is hidden at depth 0.1 / 0.5 / 0.9 of a repeated
filler text; the chat-templated prompt is the target length within +-16
tokens. The prompts are fixed in bench_data/needle.jsonl.gz.

Usage:
    python needle.py gen <tokenizer> <out.jsonl.gz> [prompt_lens...]
    python needle.py run <needle.jsonl.gz> [base_url]   (served model name: hy3)
"""

import gzip
import json
import random
import sys
import time
import urllib.request

FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again. "
)
DEPTHS = [0.1, 0.5, 0.9]
QUESTION = "\n\nWhat is the pass key? Answer with the number only."


def prompt_len(tok, text):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
    )
    return len(ids["input_ids"] if hasattr(ids, "keys") else ids)


def build(tok, n_tokens, depth, key):
    needle = f" The pass key is {key}. Remember it. {key} is the pass key. "
    filler_ids = tok.encode(FILLER, add_special_tokens=False)
    budget = n_tokens
    for _ in range(8):
        hay = (filler_ids * (budget // len(filler_ids) + 1))[:budget]
        cut = int(len(hay) * depth)
        text = tok.decode(hay[:cut]) + needle + tok.decode(hay[cut:]) + QUESTION
        diff = n_tokens - prompt_len(tok, text)
        if abs(diff) <= 16:
            break
        budget += int(diff * 1.05)
    return text


def gen(tokenizer, out, lengths):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    rng = random.Random(0)
    rows = []
    for n in lengths:
        for d in DEPTHS:
            key = str(rng.randint(10000, 99999))
            text = build(tok, n, d, key)
            rows.append(
                {
                    "target_len": n,
                    "prompt_len": prompt_len(tok, text),
                    "depth": d,
                    "key": key,
                    "prompt": text,
                }
            )
            print(f"len={n} prompt_len={rows[-1]['prompt_len']} depth={d}", flush=True)
    body = "".join(json.dumps(r) + "\n" for r in rows)
    with (
        open(out, "wb") as f,
        gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz,
    ):
        gz.write(body.encode())


def run(path, base_url):
    with gzip.open(path, "rt") as f:
        rows = [json.loads(line) for line in f]
    ok = 0
    for r in rows:
        body = json.dumps(
            {
                "model": "hy3",
                "messages": [{"role": "user", "content": r["prompt"]}],
                "max_tokens": 16,
                "temperature": 0,
            }
        ).encode()
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            body,
            {"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as resp:
            out = json.load(resp)
        dt = time.time() - t0
        ans = out["choices"][0]["message"]["content"]
        got = r["key"] in ans
        ok += got
        print(
            f"prompt_tokens={out['usage']['prompt_tokens']:>6} depth={r['depth']:.1f} "
            f"key={r['key']} answer={ans.strip()!r} {'PASS' if got else 'FAIL'} "
            f"({dt:.1f}s)",
            flush=True,
        )
    print(f"{ok}/{len(rows)} passed")
    return ok == len(rows)


def main():
    if sys.argv[1] == "gen":
        lengths = [int(x) for x in sys.argv[4:]] or [16000, 32768, 65536, 85000]
        gen(sys.argv[2], sys.argv[3], lengths)
    else:
        base_url = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8000"
        sys.exit(0 if run(sys.argv[2], base_url) else 1)


if __name__ == "__main__":
    main()
