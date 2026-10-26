"""Compare a FP8-KV validation dump against the BF16-KV dump of the same mode.

Usage: python compare_kv_runs.py <bf16.pt> <fp8.pt> [--min-cos 0.99]

Checks, per phase and prompt:
  * every logit finite on the computed steps (no NaN / inf);
  * FP8 vs BF16: greedy prefix match, and logits cosine over the shared prefix;
    where greedy diverges, the BF16 top-2 gap at that step (near-tie or not);
  * within each run: ``repeat`` / ``reuse`` reproduce ``cold`` (prefix-cache
    hits and recycled blocks read the same cache), and prefix phases report
    the expected cached-token counts.
Exits non-zero on a failure.
"""

import argparse
import sys

import torch


def cos(a, b):
    m = torch.isfinite(a) & torch.isfinite(b)
    return torch.nn.functional.cosine_similarity(a[m], b[m], dim=0).item()


def prefix_match(a, b):
    return next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bf16")
    ap.add_argument("fp8")
    ap.add_argument("--min-cos", type=float, default=0.99)
    ap.add_argument("--near-tie", type=float, default=0.5, help="nats")
    args = ap.parse_args()
    base, test = torch.load(args.bf16), torch.load(args.fp8)
    assert base["mode"] == test["mode"], "different modes"
    fails = []
    worst = 1.0
    for ph, recs in test["phases"].items():
        for i, (t, b) in enumerate(zip(recs, base["phases"][ph])):
            m = prefix_match(t["gen_ids"], b["gen_ids"])
            line = f"[{ph} {i}] len {t['prompt_len']} cached {t['cached']}/{b['cached']} greedy {m}/{len(b['gen_ids'])}"
            if "logits" in t:
                lt, lb = t["logits"], b["logits"]
                n = min(len(lt), len(lb))
                if not torch.isfinite(lt[:n].max(dim=-1).values).all() or torch.isnan(lt[:n]).any():
                    fails.append(f"{ph} {i}: non-finite FP8 logits")
                steps = range(min(m + 1, n))
                cs = [cos(lt[s], lb[s]) for s in steps]
                worst = min(worst, min(cs))
                line += f" cos min {min(cs):.5f}"
                if min(cs) < args.min_cos:
                    fails.append(f"{ph} {i}: logits cos {min(cs):.5f} < {args.min_cos}")
                if m < min(len(t["gen_ids"]), len(b["gen_ids"])) and m < n:
                    top2 = lb[m].topk(2).values
                    gap = (top2[0] - top2[1]).item()
                    line += f" diverge@{m} bf16 top2 gap {gap:.3f}"
                    if gap > args.near_tie:
                        fails.append(f"{ph} {i}: greedy diverges at step {m} with gap {gap:.3f}")
            print(line)
    # Within-run consistency: cache hits / recycled blocks reproduce cold.
    def self_cos(run, name, i):
        a, c = run["phases"][name][i], run["phases"]["cold"][i]
        k = min(prefix_match(a["gen_ids"], c["gen_ids"]) + 1, len(a["logits"]), len(c["logits"]))
        return a["gen_ids"] == c["gen_ids"], min(cos(a["logits"][s], c["logits"][s]) for s in range(k))

    ph = test["phases"]
    for name in ("repeat", "reuse"):
        if name not in ph:
            continue
        for i, a in enumerate(ph[name]):
            same, cmin = self_cos(test, name, i)
            _, cbase = self_cos(base, name, i)
            print(f"[{name} vs cold {i}] fp8 cos min {cmin:.5f} (identical greedy {same}); "
                  f"bf16 {cbase:.5f}; cached {a['cached']}")
            if name == "reuse" and a["cached"] == 0:
                # Same chunking as cold: recycled blocks must give the same result.
                if cmin < 0.999:
                    fails.append(f"fp8 reuse {i} differs from cold (cos {cmin:.5f})")
            else:
                # A prefix hit changes the chunking (summation order) on BF16 KV
                # too; FP8 must not be worse than BF16's own X-vs-cold. (With
                # DCP the larger per-request pool can keep blocks cached in the
                # reuse phase, which then behaves like repeat.)
                if cmin < min(cbase - 2e-3, 0.999) or cmin < 0.995:
                    fails.append(f"fp8 {name} {i}: cos {cmin:.5f} vs bf16 {cbase:.5f}")
    if "repeat" in ph and not any(r["cached"] > 0 for r in ph["repeat"]):
        fails.append("repeat phase had no prefix-cache hits")
    print(f"=== worst FP8-vs-BF16 logits cosine {worst:.5f}; {len(fails)} failure(s)")
    for f in fails:
        print("FAIL", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
