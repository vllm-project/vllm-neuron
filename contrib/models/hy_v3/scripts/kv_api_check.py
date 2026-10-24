"""Full-model FP8-KV functional check through the OpenAI API (e.g. DCP 8 vs DCP 1).

Usage:
  python kv_api_check.py run <base_url> <out.json> [--tokenizer <ckpt>] [--concurrency 8] [--logprobs N]
  python kv_api_check.py compare <ref.json> <test.json> [--max-acc-drop 0.10]

``run`` sends a fixed set of greedy completions built from GSM8K train
few-shot contexts (natural text):
  - cold: contexts of ~125 to ~15,000 tokens (multi-chunk prefill near
    max_model_len), 32 tokens,
  - repeat: each prompt again (full prefix-cache hit),
  - partial: contexts cut at non-block-aligned lengths with a new question,
  - again: the cold prompts once more after the other phases,
  - gsm8k: 10 new questions after each of the 6 contexts (60 prompts sharing
    the context as a cached prefix; partial last block), answered up to 320
    tokens and graded against the GSM8K answer -- a wrong KV read shows up as
    wrong answers, independent of near-tie token splits,
  all with ``--concurrency`` requests in flight (block reuse across requests).
It fails on any HTTP error, an empty completion, a degenerate one (one token
repeated over half the completion) or, with ``--logprobs``, a non-finite logprob.
Token ids come from ``return_token_ids``. Logprobs are off by default: the
Neuron model only returns them with ``neuron_config.max_logprobs`` != 0, which
changes the compiled graphs.

``compare`` reports the per-prompt greedy-prefix agreement between two servers
(informational: tiny numeric differences, e.g. a different prefill chunking on
the same server, already split greedy at near-ties) and the GSM8K accuracy; it
exits non-zero on errors, missing prompts or a GSM8K accuracy more than
``--max-acc-drop`` below the reference.
"""

import argparse
import asyncio
import json
import math
import sys
import time

TARGETS = [200, 1500, 3000, 6000, 10000, 15000]  # prompt tokens (max_model_len 16384)
PARTIAL_CUTS = [(3, 0.61), (4, 0.83), (5, 0.37)]  # (prompt index, fraction of examples)
QUESTIONS_PER_CONTEXT = 10
GSM8K_MAX_TOKENS = 320


def gsm8k_answer(text: str):
    """The number after '####' (GSM8K format), commas and a trailing '.' removed."""
    if "####" not in text:
        return None
    tail = text.split("####", 1)[1].strip().split()
    return tail[0].replace(",", "").rstrip(".").lstrip("$") if tail else None


def prompts(tokenizer_path: str):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    ds = load_dataset("openai/gsm8k", "main", split="train")
    shots = [f"Question: {r['question']}\nAnswer: {r['answer']}\n\n" for r in ds]

    def ntok(s):
        return len(tok(s, add_special_tokens=False).input_ids)

    def context(start, target):
        parts, n, j = [], 0, start
        while n < target - 120:
            parts.append(shots[j])
            n += ntok(shots[j])
            j += 1
        return parts, j

    base, cursor = [], 0
    for t in TARGETS:
        parts, cursor = context(cursor, t)
        q = shots[cursor].split("\nAnswer:")[0] + "\nAnswer:"
        cursor += 1
        base.append((parts, q))
    out = [("cold", i, "".join(p) + q) for i, (p, q) in enumerate(base)]
    out += [("repeat", i, "".join(p) + q) for i, (p, q) in enumerate(base)]
    for j, (i, cut) in enumerate(PARTIAL_CUTS):
        parts, _ = base[i]
        q = shots[cursor + j].split("\nAnswer:")[0] + "\nAnswer:"
        out.append(("partial", j, "".join(parts[: int(len(parts) * cut)]) + q))
    out += [("again", i, "".join(p) + q) for i, (p, q) in enumerate(base)]
    start = cursor + len(PARTIAL_CUTS)
    for i, (parts, _) in enumerate(base):
        for j in range(QUESTIONS_PER_CONTEXT):
            k = start + i * QUESTIONS_PER_CONTEXT + j
            q, gold = shots[k].split("\nAnswer:")
            out.append(("gsm8k", i * QUESTIONS_PER_CONTEXT + j, "".join(parts) + q + "\nAnswer:",
                        gsm8k_answer(gold)))
    return [x if len(x) == 4 else (*x, None) for x in out]


async def run(base_url, out_path, tokenizer, concurrency, logprobs):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url + "/v1", api_key="EMPTY", timeout=3600)
    sem = asyncio.Semaphore(concurrency)
    items = prompts(tokenizer)
    results = [None] * len(items)
    errors = []

    async def one(k, phase, i, p, gold):
        long = phase == "gsm8k"
        async with sem:
            try:
                r = await client.completions.create(
                    model="hy3", prompt=p, max_tokens=GSM8K_MAX_TOKENS if long else 32,
                    temperature=0.0, stop=["\n\nQuestion:"] if long else None,
                    logprobs=logprobs or None, extra_body={"return_token_ids": True},
                )
            except Exception as e:  # noqa: BLE001
                errors.append(f"{phase} {i}: {e!r}")
                return
        c = r.choices[0].model_dump()
        ids = c.get("token_ids") or []
        lps = (c.get("logprobs") or {}).get("token_logprobs") or []
        if not c["text"] or not ids:
            errors.append(f"{phase} {i}: empty completion or no token ids")
        elif max(ids.count(t) for t in set(ids)) > len(ids) // 2 and len(set(ids)) <= 2:
            errors.append(f"{phase} {i}: degenerate completion {c['text'][:40]!r}")
        if logprobs and any(x is None or not math.isfinite(x) for x in lps):
            errors.append(f"{phase} {i}: non-finite logprobs")
        results[k] = {
            "phase": phase, "i": i, "prompt_tokens": r.usage.prompt_tokens,
            "token_ids": ids, "token_logprobs": lps, "text": c["text"],
        }
        if gold is not None:
            pred = gsm8k_answer(c["text"])
            results[k].update(gold=gold, pred=pred, correct=pred == gold)

    t0 = time.time()
    # Phases in order (cold first so repeat / partial / again can hit the cache),
    # requests within a phase concurrently.
    for phase in ("cold", "repeat", "partial", "again", "gsm8k"):
        await asyncio.gather(
            *(one(k, ph, i, p, g) for k, (ph, i, p, g) in enumerate(items) if ph == phase)
        )
    json.dump({"base_url": base_url, "results": results, "errors": errors, "seconds": time.time() - t0},
              open(out_path, "w"), indent=1)
    for r in results:
        if r:
            print(f"{r['phase']:8s} {r['i']} prompt_tokens {r['prompt_tokens']:6d} -> {r['text'][:60]!r}")
    # Within-run: repeat / again (prefix-cache hits) against cold.
    by = {(r["phase"], r["i"]): r for r in results if r}
    for ph in ("repeat", "again"):
        same = sum(by.get((ph, i), {}).get("token_ids") == by.get(("cold", i), {}).get("token_ids")
                   for i in range(len(TARGETS)))
        print(f"--- {ph} vs cold: {same}/{len(TARGETS)} identical")
    acc = gsm8k_accuracy(results)
    print(f"--- gsm8k: {acc['correct']}/{acc['n']} correct; by context tokens: "
          + ", ".join(f"{t}: {c}/{n}" for t, (c, n) in acc["by_len"].items()))
    print(f"=== {len(by)}/{len(items)} ok, {len(errors)} error(s) in {time.time() - t0:.0f}s")
    for e in errors:
        print("ERROR", e)
    return 1 if errors or len(by) != len(items) else 0


def gsm8k_accuracy(results):
    rows = [r for r in results if r and r["phase"] == "gsm8k"]
    by_len = {}
    for r in rows:
        t = r["prompt_tokens"] // 1000 * 1000
        c, n = by_len.get(t, (0, 0))
        by_len[t] = (c + r["correct"], n + 1)
    return {"n": len(rows), "correct": sum(r["correct"] for r in rows), "by_len": dict(sorted(by_len.items()))}


def compare(ref_path, test_path, max_acc_drop):
    ra, rb = json.load(open(ref_path)), json.load(open(test_path))
    ref = {(r["phase"], r["i"]): r for r in ra["results"] if r}
    test = {(r["phase"], r["i"]): r for r in rb["results"] if r}
    fails, agrees = [], []
    if not ref or ra["errors"]:
        fails.append(f"reference run incomplete ({len(ref)} results, {len(ra['errors'])} errors)")
    if rb["errors"]:
        fails.append(f"test run had {len(rb['errors'])} errors")
    for key in sorted(k for k in ref if k[0] != "gsm8k"):
        a, b = ref[key], test.get(key)
        if b is None:
            fails.append(f"{key} missing")
            continue
        x, y = a["token_ids"], b["token_ids"]
        m = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        agrees.append(m / max(len(x), 1))
        print(f"{key[0]:8s} {key[1]} tokens {a['prompt_tokens']:6d}/{b['prompt_tokens']:6d}: greedy {m}/{len(x)}")
        if a["prompt_tokens"] != b["prompt_tokens"]:
            fails.append(f"{key}: different prompt token counts")
    mean = sum(agrees) / len(agrees) if agrees else 0.0
    full = sum(a == 1.0 for a in agrees)
    ga, gb = gsm8k_accuracy(ra["results"]), gsm8k_accuracy(rb["results"])
    same = sum(1 for k in ref if k[0] == "gsm8k" and k in test and ref[k]["pred"] == test[k]["pred"])
    if gb["n"] != ga["n"]:
        fails.append(f"gsm8k: {gb['n']} vs {ga['n']} answers")
    drop = (ga["correct"] - gb["correct"]) / max(ga["n"], 1)
    if drop > max_acc_drop:
        fails.append(f"gsm8k accuracy {gb['correct']}/{gb['n']} vs {ga['correct']}/{ga['n']}")
    print(f"gsm8k (context-prefixed, graded): ref {ga['correct']}/{ga['n']}, test {gb['correct']}/{gb['n']}, "
          f"same final answer {same}/{ga['n']}")
    print(f"=== mean greedy-prefix agreement {mean:.3f} ({full}/{len(agrees)} identical, 32 tokens); "
          f"gsm8k {gb['correct']}/{gb['n']} vs ref {ga['correct']}/{ga['n']}; {len(fails)} failure(s)")
    for f in fails:
        print("FAIL", f)
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("base_url")
    r.add_argument("out")
    r.add_argument("--tokenizer", default="tencent/Hy3-FP8", help="tokenizer path or HF id")
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument("--logprobs", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("test")
    c.add_argument("--max-acc-drop", type=float, default=0.10, help="fraction (60 questions)")
    a = ap.parse_args()
    if a.cmd == "run":
        sys.exit(asyncio.run(run(a.base_url, a.out, a.tokenizer, a.concurrency, a.logprobs)))
    sys.exit(compare(a.ref, a.test, a.max_acc_drop))
