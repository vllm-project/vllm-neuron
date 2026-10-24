"""BF16-KV vs FP8-KV acceptance report from two kv_acceptance.sh output dirs.

Usage: python kv_report.py <bf16_dir> <fp8_dir> [--max-regress 5] [--max-acc-drop 1.0]

Prints markdown: KV pool / device memory, the load matrix (TTFT / TPOT P50 and
P99, throughput, % change), MATH-500 and stability, with PASS / FAIL against
the acceptance thresholds. Exits non-zero on a FAIL.
"""

import argparse
import json
import re
import sys
from pathlib import Path


def device_bytes(d: Path):
    """Sum of neuron_device bytes over runtimes in the first neuron-monitor report."""
    p = d / "neuron_monitor.json"
    if not p.exists() or not p.read_text().strip():
        return None, {}
    rep = json.loads(p.read_text().splitlines()[0])
    total, breakdown = 0, {}
    for rt in rep.get("neuron_runtime_data", []):
        mu = rt["report"]["memory_used"]["neuron_runtime_used_bytes"]
        total += mu["neuron_device"]
        dev = mu.get("usage_breakdown", {}).get("neuroncore_memory_usage", {})
        for core in dev.values():
            for k, v in core.items():
                breakdown[k] = breakdown.get(k, 0) + v
    return total, breakdown


def pool_tokens(d: Path):
    t = (d / "kv_pool_log.txt").read_text() if (d / "kv_pool_log.txt").exists() else ""
    m = re.search(r"KV cache size: ([\d,]+) tokens", t)
    c = re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", t)
    return (int(m.group(1).replace(",", "")) if m else None, float(c.group(2)) if c else None)


def bench(d: Path):
    out = {}
    for f in sorted((d / "bench").glob("in*_out*_c*.json")):
        j = json.loads(f.read_text())
        m = re.match(r"in(\d+)_out(\d+)_c(\d+)", f.stem)
        out[(int(m.group(1)), int(m.group(3)))] = j
    return out


def math(d: Path, name: str):
    p = d / f"math500_{name}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return {
        "n": len(rows),
        "acc": 100.0 * sum(r["correct"] for r in rows) / len(rows),
        "trunc": sum(r["finish_reason"] == "length" for r in rows),
        "tokens": sum(r["completion_tokens"] for r in rows) / len(rows),
    }


def pct(a, b):
    return 100.0 * (b - a) / a if a else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bf16")
    ap.add_argument("fp8")
    ap.add_argument("--max-regress", type=float, default=5.0)
    ap.add_argument("--max-acc-drop", type=float, default=1.0)
    ap.add_argument(
        "--bf16-kv-bytes-per-token",
        type=int,
        default=64 * 80 * 2 * 128 * 2,
        help="BF16 K+V bytes per cached token over all runtimes (Hy3 TP64: 64 ranks x "
        "80 layers x K,V x 1 KV head x 128 x 2 bytes)",
    )
    a = ap.parse_args()
    B, F = Path(a.bf16), Path(a.fp8)
    fails = []
    print("## KV pool and device memory\n")
    tb, cb = pool_tokens(B)
    tf, cf = pool_tokens(F)
    db, bb = device_bytes(B)
    df, bf = device_bytes(F)
    print("| | BF16 KV | FP8 KV |\n|---|---|---|")
    print(f"| KV pool (tokens) | {tb:,} | {tf:,} |" if tb and tf else f"| KV pool (tokens) | {tb} | {tf} |")
    print(f"| max concurrency @16384 | {cb} | {cf} |")
    if db and df:
        print(f"| device memory, all runtimes (GiB) | {db / 2**30:.2f} | {df / 2**30:.2f} |")
        for k in sorted(set(bb) | set(bf)):
            print(f"| &nbsp;&nbsp;{k} (GiB) | {bb.get(k, 0) / 2**30:.2f} | {bf.get(k, 0) / 2**30:.2f} |")
    print()
    if db and df and tb and tb == tf and "tensors" in bb and "tensors" in bf:
        # Same checkpoint and the same pool size: the device "tensors" delta is the KV delta.
        kv_b = tb * a.bf16_kv_bytes_per_token
        saved = bb["tensors"] - bf["tensors"]
        ratio = saved / kv_b
        print(f"BF16 KV pool from the geometry: {kv_b / 2**30:.2f} GiB; measured device-tensor "
              f"saving {saved / 2**30:.2f} GiB = {100 * ratio:.1f}% of it "
              f"(non-KV tensors {(bb['tensors'] - kv_b) / 2**30:.2f} / {(bf['tensors'] - kv_b / 2) / 2**30:.2f} GiB).")
        ok = abs(ratio - 0.5) <= 0.02
        if not ok:
            fails.append(f"KV memory saving {100 * ratio:.1f}% (expected ~50%)")
        print(f"KV memory at the same cached token count: {'PASS' if ok else 'FAIL'} (expect 50% +- 2)\n")
    else:
        fails.append("KV memory not comparable (pool sizes or monitor data missing)")

    print("## Load matrix (256 output tokens)\n")
    rb, rf = bench(B), bench(F)
    print("| input | conc | TTFT P50 ms BF16 / FP8 (Δ%) | TTFT P99 ms | TPOT P50 ms | TPOT P99 ms | out tok/s | ok |")
    print("|---|---|---|---|---|---|---|---|")
    for key in sorted(rb):
        if key not in rf:
            fails.append(f"missing FP8 cell {key}")
            continue
        x, y = rb[key], rf[key]
        cells, worst = [], 0.0
        for metric in ("median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms"):
            d = pct(x[metric], y[metric])
            if metric.startswith("median"):
                worst = max(worst, d)
            cells.append(f"{x[metric]:.1f} / {y[metric]:.1f} ({d:+.1f}%)")
        t = pct(x["output_throughput"], y["output_throughput"])
        cells.append(f"{x['output_throughput']:.1f} / {y['output_throughput']:.1f} ({t:+.1f}%)")
        okc = y["completed"] == x["completed"] and y.get("failed", 0) == 0
        ok = okc and worst <= a.max_regress
        if not ok:
            fails.append(f"matrix {key}: P50 regression {worst:+.1f}% or failed requests")
        print(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + f" | {'PASS' if ok else 'FAIL'} |")
    print(f"\nPass rule: P50 TTFT and P50 TPOT regress by at most {a.max_regress}% per cell; P99 reported.\n")

    print("## MATH-500 (full, no_think, max 4096 tokens)\n")
    print("| run | BF16 KV | FP8 KV | Δ (pp) | ok |\n|---|---|---|---|---|")
    for name, label in (("greedy", "greedy (temperature 0)"), ("t09", "temperature 0.9, seeded")):
        mb, mf = math(B, name), math(F, name)
        if not (mb and mf):
            continue
        dlt = mf["acc"] - mb["acc"]
        ok = dlt >= -a.max_acc_drop
        if name == "greedy" and not ok:
            fails.append(f"MATH-500 greedy drop {dlt:.1f} pp")
        print(f"| {label} | {mb['acc']:.1f}% (trunc {mb['trunc']}) | {mf['acc']:.1f}% (trunc {mf['trunc']}) | {dlt:+.1f} | {'PASS' if ok else 'FAIL'} |")
    print()

    print("## Stability (30 min, prefix reuse)\n")
    for lab, d in (("BF16", B), ("FP8", F)):
        p = d / "stability.json"
        if p.exists():
            j = json.loads(p.read_text())
            print(f"- {lab}: {j['completed']} ok / {j.get('failed', 0)} failed in {j['duration']:.0f}s, "
                  f"TTFT P50 {j['median_ttft_ms']:.0f} ms P99 {j['p99_ttft_ms']:.0f} ms, "
                  f"TPOT P50 {j['median_tpot_ms']:.1f} ms P99 {j['p99_tpot_ms']:.1f} ms, "
                  f"out {j['output_throughput']:.1f} tok/s")
            if lab == "FP8" and j.get("failed", 0):
                fails.append("stability run had failed requests")
        m = d / "stability_metrics.txt"
        if m.exists():
            print("  " + " ".join(l.strip() for l in m.read_text().splitlines()))
    print()
    print(f"**{'PASS' if not fails else 'FAIL'}**" + ("" if not fails else ": " + "; ".join(fails)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
