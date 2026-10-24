"""MATH-500 eval against an OpenAI-compatible server, graded with math-verify.

Usage: python eval_math500.py <reasoning_effort> <out.jsonl>
           [--concurrency 64] [--max-tokens 4096] [--temperature 0.9] [--limit N]
           [--data math500.jsonl] [--base-url http://127.0.0.1:8000/v1]
           [--seed S]  (per-request seed S + problem index; same across servers)

Requires: openai, math-verify[antlr4_13_2], and datasets (only when --data is
not given; HuggingFaceH4/MATH-500 is then downloaded).
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from math_verify import parse, verify
from openai import AsyncOpenAI

PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def last_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def grade(pred_text: str, gold: str) -> bool:
    boxed = last_boxed(pred_text)
    if boxed is None:
        return False
    try:
        return bool(verify(parse(f"${gold}$"), parse(f"${boxed}$")))
    except Exception:  # noqa: BLE001 - unparsable answers count as wrong
        return False


async def main(args):
    if args.data:
        rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    else:
        from datasets import load_dataset

        rows = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    if args.limit:
        rows = rows[: args.limit]
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=7200)
    sem = asyncio.Semaphore(args.concurrency)
    results = [None] * len(rows)
    done = 0
    t0 = time.time()

    async def one(i, row):
        nonlocal done
        async with sem:
            r = await client.chat.completions.create(
                model="hy3",
                messages=[
                    {"role": "user", "content": PROMPT.format(problem=row["problem"])}
                ],
                temperature=args.temperature,
                top_p=1.0,
                max_tokens=args.max_tokens,
                **({"seed": args.seed + i} if args.seed is not None else {}),
                extra_body={
                    "chat_template_kwargs": {"reasoning_effort": args.reasoning_effort}
                },
            )
        text = r.choices[0].message.content or ""
        ok = grade(text, row["answer"])
        results[i] = {
            "unique_id": row["unique_id"],
            "level": row["level"],
            "subject": row["subject"],
            "answer": row["answer"],
            "pred": last_boxed(text),
            "correct": ok,
            "completion_tokens": r.usage.completion_tokens,
            "finish_reason": r.choices[0].finish_reason,
            "text": text,
        }
        done += 1
        if done % 50 == 0:
            acc = sum(x["correct"] for x in results if x) / done
            print(
                f"{done}/{len(rows)} acc so far {acc:.3f} ({time.time() - t0:.0f}s)",
                flush=True,
            )

    await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)))
    Path(args.out).write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results)
    )

    n = len(results)
    acc = sum(x["correct"] for x in results) / n
    toks = [x["completion_tokens"] for x in results]
    trunc = sum(x["finish_reason"] == "length" for x in results)
    print(
        f"=== MATH-500 reasoning_effort={args.reasoning_effort} temp={args.temperature}: "
        f"accuracy {acc * 100:.1f}% ({sum(x['correct'] for x in results)}/{n}); "
        f"avg completion tokens {sum(toks) / n:.0f}, truncated {trunc}, wall {time.time() - t0:.0f}s"
    )
    for lvl in range(1, 6):
        sub = [x for x in results if x["level"] == lvl]
        if sub:
            print(
                f"    level {lvl}: {sum(x['correct'] for x in sub) / len(sub) * 100:.1f}% (n={len(sub)})"
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("reasoning_effort", choices=["no_think", "low", "high"])
    p.add_argument("out")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--data", default=None, help="local MATH-500 jsonl (default: download)"
    )
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--seed", type=int, default=None)
    asyncio.run(main(p.parse_args()))
