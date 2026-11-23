"""Real-text prompts of exact token lengths, as a `vllm bench serve` custom-dataset JSONL.

`vllm bench serve --dataset-name random` builds each prompt from consecutive
vocabulary ids ((offset + i) % vocab_size), i.e. text no one writes. A MoE router
spreads such tokens unlike real text, so expert load (and anything that depends
on it) is off, and a draft model accepts almost nothing, so speculative decoding
looks useless. This script builds prompts of the same exact lengths from real
conversations instead.

Each prompt concatenates whole conversations from a ShareGPT-format JSON
(``[{"conversations": [{"from": ..., "value": ...}, ...]}, ...]``), starting at a
random one, until it reaches its length, then cuts it there. Different
conversations are concatenated rather than one short prompt being repeated:
repetition is easy for a draft model and inflates speculative-decoding
acceptance. Lengths are counted the way the benchmark counts them
(``len(tokenizer(prompt).input_ids)``, special tokens included) and sampled
uniformly from ``[ISL * (1 - ratio), ISL * (1 + ratio)]``.

    python make_prompts.py --tokenizer Qwen/Qwen3.5-2B --corpus ShareGPT_V3_unfiltered_cleaned_split.json \
        --isl 8192 --num-prompts 64 --seed 8192004 --out prompts.jsonl
"""

import argparse
import json
import math

import numpy as np


def load_corpus(path: str) -> list[str]:
    with open(path) as fh:
        data = json.load(fh)
    docs = []
    for item in data:
        turns = [t.get("value", "") for t in item.get("conversations", [])]
        text = "\n\n".join(t.strip() for t in turns if t and t.strip())
        if text:
            docs.append(text)
    if not docs:
        raise SystemExit(f"{path}: no conversations")
    return docs


def build_prompt(tokenizer, docs, start: int, target: int) -> tuple[str, int]:
    """Concatenate docs from ``start`` until ``target`` tokens; return (prompt, its token count)."""
    special = len(tokenizer("").input_ids)  # BOS etc. the benchmark's count includes
    want = max(1, target - special)
    ids: list[int] = []
    j = start
    while len(ids) < want + 16:
        piece = ("\n\n" if ids else "") + docs[j % len(docs)]
        ids.extend(tokenizer(piece, add_special_tokens=False).input_ids)
        j += 1
    # Decoding then re-encoding can merge or split tokens at the cut, so search
    # for the cut whose re-encoded length is exactly the target.
    n = want
    prompt, got = "", 0
    for _ in range(32):
        prompt = tokenizer.decode(ids[:n])
        got = len(tokenizer(prompt).input_ids)
        if got == target:
            break
        n += target - got
        n = max(1, min(n, len(ids)))
    return prompt, got


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--corpus", required=True, help="ShareGPT-format JSON")
    p.add_argument("--isl", type=int, required=True)
    p.add_argument("--input-ratio", type=float, default=0.03)
    p.add_argument("--num-prompts", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    docs = load_corpus(a.corpus)
    rng = np.random.default_rng(a.seed)
    lo = max(1, math.floor(a.isl * (1 - a.input_ratio)))
    hi = math.ceil(a.isl * (1 + a.input_ratio))
    targets = rng.integers(lo, hi + 1, size=a.num_prompts)
    starts = rng.choice(len(docs), size=a.num_prompts, replace=a.num_prompts > len(docs))

    exact, lens = 0, []
    with open(a.out, "w") as fh:
        for target, start in zip(targets.tolist(), starts.tolist()):
            prompt, got = build_prompt(tokenizer, docs, start, target)
            exact += got == target
            lens.append(got)
            fh.write(json.dumps({"prompt": prompt}) + "\n")
    print(f"{a.out}: {a.num_prompts} prompts, {exact} at the exact sampled length, "
          f"tokens min/mean/max {min(lens)}/{sum(lens) / len(lens):.0f}/{max(lens)}")


if __name__ == "__main__":
    main()
