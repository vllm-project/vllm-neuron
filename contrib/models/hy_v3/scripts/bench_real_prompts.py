"""Latency on real prompts (MATH-500 problems), plus MTP acceptance.

Random-token benchmarks say nothing about speculative-decoding acceptance, so
this sends real chat prompts one at a time (greedy, streaming) and reports the
median TTFT / TPOT. When the server runs MTP it also reports the acceptance
from the server's spec-decode counters (/metrics).

Usage: python bench_real_prompts.py <out.json> [--num 20] [--max-tokens 512]
           [--concurrency 1] [--data math500.jsonl] [--base-url http://127.0.0.1:8000]
Requires: openai (and datasets when --data is not given).
"""

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI


def counters(base_url: str) -> dict[str, float]:
    """Spec-decode counters summed over engines; per-position ones keyed by position."""
    import re

    text = urllib.request.urlopen(f"{base_url}/metrics").read().decode()
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = re.match(
            r"(vllm:spec_decode_num_[a-z_]+?)(?:_total)?(\{[^}]*\})? (\S+)$", line
        )
        if not m or m.group(1).endswith("_created"):
            continue
        pos = re.search(r'position="(\d+)"', m.group(2) or "")
        key = m.group(1) + (f"[{pos.group(1)}]" if pos else "")
        out[key] = out.get(key, 0.0) + float(m.group(3))
    return out


def main(args):
    if args.data:
        rows = [json.loads(x) for x in Path(args.data).read_text().splitlines()]
    else:
        from datasets import load_dataset

        rows = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    rows = rows[: args.num]
    client = OpenAI(base_url=f"{args.base_url}/v1", api_key="EMPTY")

    def one(r):
        t0 = time.perf_counter()
        first, n_chunks, text = None, 0, ""
        stream = client.chat.completions.create(
            model="hy3",
            messages=[{"role": "user", "content": r["problem"]}],
            max_tokens=args.max_tokens,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"reasoning_effort": "no_think"}},
        )
        usage = None
        for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.choices and chunk.choices[0].delta.content:
                if first is None:
                    first = time.perf_counter()
                n_chunks += 1
                text += chunk.choices[0].delta.content
        end = time.perf_counter()
        n_out = usage.completion_tokens if usage else n_chunks
        ttft = (first or end) - t0
        tpot = (end - (first or end)) / max(n_out - 1, 1)
        print(
            f"ttft {ttft * 1e3:.0f} ms, tpot {tpot * 1e3:.1f} ms, {n_out} tokens",
            flush=True,
        )
        return {"ttft": ttft, "tpot": tpot, "output_tokens": n_out, "text": text}

    before = counters(args.base_url)
    wall0 = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(one, rows))
    wall = time.perf_counter() - wall0
    after = counters(args.base_url)
    delta = {k: after[k] - before.get(k, 0.0) for k in after}
    drafts = delta.get("vllm:spec_decode_num_draft_tokens", 0.0)
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens", 0.0)
    summary = {
        "requests": len(results),
        "median_ttft_ms": statistics.median(x["ttft"] for x in results) * 1e3,
        "median_tpot_ms": statistics.median(x["tpot"] for x in results) * 1e3,
        "mean_output_tokens": statistics.mean(x["output_tokens"] for x in results),
        "concurrency": args.concurrency,
        "wall_s": wall,
        "output_tok_per_s": sum(x["output_tokens"] for x in results) / wall,
        "spec_decode": delta,
        "acceptance_rate": accepted / drafts if drafts else None,
    }
    print(
        json.dumps({k: v for k, v in summary.items() if k != "spec_decode"}, indent=1)
    )
    Path(args.out).write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--num", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--data", default=None)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    main(p.parse_args())
