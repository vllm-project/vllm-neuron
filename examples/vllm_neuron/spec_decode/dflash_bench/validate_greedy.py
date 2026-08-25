#!/usr/bin/env python3
"""Capture greedy completions from a running vLLM server.

Greedy decoding must be invariant to speculative decoding: DFlash proposes
tokens, the target verifies them, and rejected proposals are discarded. So the
baseline and the DFlash run must emit the same token sequence for the same
prompt. Running this against both servers and diffing the JSON is the
correctness gate that has to pass before any timing number means anything.
"""

import argparse
import json
import sys
import time

import requests

PROMPTS = [
    "Explain why speculative decoding preserves the output distribution of\n"
    "the target model.",
    "Write a Python function that merges two sorted lists into one sorted list.",
    "List the first ten prime numbers, separated by commas.",
    "Summarize the difference between tensor parallelism and pipeline parallelism.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    "The capital of Australia is",
    "Translate to French: 'The weather is nice today and I would like to go outside.'",
    "Count from 1 to 20 in words, one per line.",
]


def complete(base_url, model, prompt, max_tokens):
    r = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "max_tokens": max_tokens,
            # Exact token IDs, not logprobs: with Neuron on-device sampling the
            # sampled token is not guaranteed to appear in the returned top-k,
            # and vLLM's logprobs formatter then raises KeyError. Token IDs are
            # also the stronger comparison — identical text can in principle
            # come from a different tokenization.
            "return_token_ids": True,
        },
        timeout=1800,
    )
    r.raise_for_status()
    choice = r.json()["choices"][0]
    return {
        "text": choice["text"],
        "tokens": choice.get("token_ids"),
        "finish_reason": choice.get("finish_reason"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = []
    for i, prompt in enumerate(PROMPTS):
        t0 = time.perf_counter()
        try:
            out = complete(args.base_url, args.model, prompt, args.max_tokens)
        except Exception as exc:
            print(f"[{i}] FAILED: {exc}", file=sys.stderr)
            raise
        out["prompt"] = prompt
        out["wall_s"] = round(time.perf_counter() - t0, 3)
        results.append(out)
        n = len(out["tokens"]) if out["tokens"] else -1
        print(f"[{i}] {n} tokens in {out['wall_s']}s")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
