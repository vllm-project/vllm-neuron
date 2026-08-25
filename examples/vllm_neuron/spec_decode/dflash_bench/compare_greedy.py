#!/usr/bin/env python3
"""Diff two validate_greedy.py captures token by token.

Reports, per prompt, the index of the first divergent token. A DFlash run that
diverges from the baseline is a correctness bug in the drafter/verifier
handshake, not a tuning issue.
"""

import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
assert len(a) == len(b), f"{len(a)} vs {len(b)} prompts"

bad = 0
for i, (x, y) in enumerate(zip(a, b)):
    if x["tokens"] == y["tokens"] and x["text"] == y["text"]:
        print(f"[{i}] MATCH  ({len(x['tokens'] or [])} tokens)")
        continue
    bad += 1
    tx, ty = x["tokens"] or [], y["tokens"] or []
    first = next(
        (j for j in range(min(len(tx), len(ty))) if tx[j] != ty[j]),
        min(len(tx), len(ty)),
    )
    print(f"[{i}] DIVERGE at token {first} (lens {len(tx)} vs {len(ty)})")
    print(f"      base: {tx[first : first + 6]}")
    print(f"      test: {ty[first : first + 6]}")

print(f"\n{len(a) - bad}/{len(a)} prompts match exactly")
sys.exit(1 if bad else 0)
