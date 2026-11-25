"""Workload for the routed-expert statistics (against scripts/expert_load.sh).

Real text only (the routing depends on it): the fixed ShareGPT prompts of
bench_data/prompts (1k / 4k / 8k / 16k tokens, raw completions) and MATH-500
problems (chat), each generating --max-tokens tokens at temperature 0.9 (MATH-500:
greedy), --concurrency requests at a time.
Usage: python expert_load_client.py [--base-url URL] [--math500 math500.jsonl] [--max-tokens 128]
"""

import argparse
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent
CELLS = ["in1024_c8", "in4096_c8", "in8192_c8", "in16000_c4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--math500", help="MATH-500 jsonl (problem field); default: the HF dataset")
    ap.add_argument("--num-math", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    client = OpenAI(base_url=a.base_url + "/v1", api_key="EMPTY", timeout=3600)

    jobs = []
    for cell in CELLS:
        with gzip.open(HERE.parent / f"bench_data/prompts/{cell}.jsonl.gz", "rt") as f:
            jobs += [("text", json.loads(line)["prompt"]) for line in f]
    if a.math500:
        with open(a.math500) as f:
            problems = [json.loads(line)["problem"] for line in f]
    else:
        from datasets import load_dataset

        problems = [r["problem"] for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]
    jobs += [("math", p) for p in problems[: a.num_math]]

    def run(job):
        kind, prompt = job
        if kind == "text":
            r = client.completions.create(model="hy3", prompt=prompt, max_tokens=a.max_tokens,
                                          temperature=0.9, extra_body={"ignore_eos": True})
            return r.usage.completion_tokens
        r = client.chat.completions.create(
            model="hy3", messages=[{"role": "user", "content": prompt}],
            max_tokens=a.max_tokens, temperature=0.0)
        return r.usage.completion_tokens

    t0 = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex:
        n = sum(ex.map(run, jobs))
    print(f"{len(jobs)} requests, {n} generated tokens, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
