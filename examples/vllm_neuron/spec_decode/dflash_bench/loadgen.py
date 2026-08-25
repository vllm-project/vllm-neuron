#!/usr/bin/env python3
"""Fixed-concurrency load generator for the DFlash A/B.

Deliberately not `vllm bench serve --dataset-name random`: random token
prompts make the continuation unpredictable, which understates a drafter's
acceptance rate and would make DFlash look worse than it is on real traffic.
This drives natural-language and code prompts instead, and reports the
streaming metrics that speculative decoding actually moves (TTFT is unchanged
by design; TPOT/ITL is where the win shows up).
"""

import argparse
import json
import statistics
import threading
import time

import requests

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number using\n"
    "memoization, and explain the time complexity.",
    "Explain how a B-tree index speeds up range queries in a relational\ndatabase.",
    "Summarize the tradeoffs between tensor parallelism and pipeline\n"
    "parallelism for serving a 20B parameter model.",
    "Write a bash script that finds all files larger than 100MB under a\n"
    "directory and prints them sorted by size.",
    "Describe, step by step, how HTTPS establishes a secure connection.",
    "Implement a least-recently-used cache in Python with O(1) get and put.",
    "What are the main differences between a process and a thread? Give\n"
    "concrete examples.",
    "Write a SQL query that returns the top 3 highest-paid employees per department.",
    "Explain what a Kalman filter does, in plain language, with one worked intuition.",
    "Refactor this into idiomatic Python:\n\nresult = []\n"
    "for i in range(len(items)):\n    if items[i] % 2 == 0:\n"
    "        result.append(items[i] * 2)\n",
    "Explain the CAP theorem and why 'CA' systems are usually a misnomer in practice.",
    "Write a unit test suite for a function that parses ISO 8601 durations.",
]


def one_chat(base_url, model, prompt, max_tokens):
    """Same measurement over /v1/chat/completions.

    gpt-oss is a harmony-format chat model and the DFlash drafter was trained
    on responses this target generated, so raw completions with ignore_eos sit
    off-distribution and understate acceptance. This path exercises the traffic
    the drafter was actually fitted to.
    """
    t0 = time.perf_counter()
    ttft, itls, prev, nchunk, ntok = None, [], t0, 0, 0
    with requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        stream=True,
        timeout=1800,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens"):
                ntok = usage["completion_tokens"]
            if not chunk.get("choices"):
                continue
            delta = chunk["choices"][0].get("delta") or {}
            if not (delta.get("content") or delta.get("reasoning_content")):
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            else:
                itls.append(now - prev)
            prev = now
            nchunk += 1
    e2e = time.perf_counter() - t0
    if not ntok:
        ntok = nchunk
    return {"ttft": ttft, "itls": itls, "e2e": e2e, "ntok": ntok, "nchunk": nchunk}


def one(base_url, model, prompt, max_tokens):
    """Stream one completion.

    Token counts come from the usage record, not from counting stream chunks:
    with speculative decoding one chunk carries a whole accepted block, so
    counting chunks undercounts tokens and reports inter-chunk latency as if it
    were inter-token latency. Chunk timings are still returned, separately
    labelled, since they are what a streaming client actually perceives.
    """
    t0 = time.perf_counter()
    ttft = None
    itls = []
    prev = t0
    nchunk = 0
    ntok = 0
    with requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "min_tokens": max_tokens,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        stream=True,
        timeout=1800,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens"):
                ntok = usage["completion_tokens"]
            if not chunk.get("choices"):
                continue
            text = chunk["choices"][0].get("text", "")
            if not text:
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            else:
                itls.append(now - prev)
            prev = now
            nchunk += 1
    e2e = time.perf_counter() - t0
    if not ntok:  # server did not report usage; fall back to the chunk count
        ntok = nchunk
    return {"ttft": ttft, "itls": itls, "e2e": e2e, "ntok": ntok, "nchunk": nchunk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--chat",
        action="store_true",
        help="drive /v1/chat/completions instead of raw completions",
    )
    args = ap.parse_args()

    jobs = [PROMPTS[i % len(PROMPTS)] for i in range(args.requests)]
    results, lock, idx = [], threading.Lock(), [0]

    def worker():
        while True:
            with lock:
                if idx[0] >= len(jobs):
                    return
                job = jobs[idx[0]]
                idx[0] += 1
            fn = one_chat if args.chat else one
            results.append(fn(args.base_url, args.model, job, args.max_tokens))

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    all_itl = [x for r in results for x in r["itls"]]
    ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
    total_tok = sum(r["ntok"] for r in results)
    total_chunk = sum(r["nchunk"] for r in results)
    # Mean time per output token after the first -- the metric speculative
    # decoding is meant to move.
    tpots = [
        (r["e2e"] - r["ttft"]) / (r["ntok"] - 1)
        for r in results
        if r["ttft"] is not None and r["ntok"] > 1
    ]

    def pct(xs, p):
        if not xs:
            return None
        return round(statistics.quantiles(xs, n=100)[p - 1] * 1000, 3)

    summary = {
        "concurrency": args.concurrency,
        "requests": len(results),
        "max_tokens": args.max_tokens,
        "wall_s": round(wall, 3),
        "output_tok_per_s": round(total_tok / wall, 2),
        "output_tok_per_s_per_req": round(total_tok / wall / args.concurrency, 2),
        "ttft_ms_mean": round(statistics.fmean(ttfts) * 1000, 3) if ttfts else None,
        "ttft_ms_p50": pct(ttfts, 50),
        "tpot_ms_mean": round(statistics.fmean(tpots) * 1000, 3) if tpots else None,
        "tokens_per_chunk": round(total_tok / total_chunk, 3) if total_chunk else None,
        "total_stream_chunks": total_chunk,
        "chunk_itl_ms_mean": round(statistics.fmean(all_itl) * 1000, 3)
        if all_itl
        else None,
        "chunk_itl_ms_p50": pct(all_itl, 50),
        "chunk_itl_ms_p90": pct(all_itl, 90),
        "e2e_s_mean": round(statistics.fmean(r["e2e"] for r in results), 3),
        "total_output_tokens": total_tok,
    }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "raw": results}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
