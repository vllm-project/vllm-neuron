"""Tables from `vllm bench serve` result JSONs written by bench_matrix.sh.

Usage:
    # one run, one row per (input, output, concurrency)
    python summarize.py results/bench/hy3-ep64

    # baseline -> new, per cell present in both
    python summarize.py results/bench/new --baseline results/bench/old

    # several models side by side on one metric
    python summarize.py results/bench/hy3 results/bench/mimo-v2.6-flash --pivot median_tpot_ms

    # every cell of every run, as CSV
    python summarize.py results/bench/* --csv all.csv

A result dir may be given as LABEL=DIR to override the label, which otherwise is
the "label" metadata saved by bench_matrix.sh (or the dir name).
"""

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

COLS = [
    ("TTFT med (s)", "median_ttft_ms", 1e-3, "{:.2f}"),
    ("TTFT P99 (s)", "p99_ttft_ms", 1e-3, "{:.2f}"),
    ("TPOT med (ms)", "median_tpot_ms", 1, "{:.1f}"),
    ("TPOT P99 (ms)", "p99_tpot_ms", 1, "{:.1f}"),
    ("ITL med (ms)", "median_itl_ms", 1, "{:.1f}"),
    ("out tok/s", "output_throughput", 1, "{:.0f}"),
    ("peak out tok/s", "max_output_tokens_per_s", 1, "{:.0f}"),
    ("total tok/s", "total_token_throughput", 1, "{:.0f}"),
]
CELL = re.compile(r"in(\d+)_out(\d+)_c(\d+)\.json")


def load(spec):
    """``DIR`` or ``LABEL=DIR`` -> (label, {(isl, osl, conc): result})."""
    label, _, d = spec.rpartition("=") if "=" in spec else ("", "", spec)
    out = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        m = CELL.match(os.path.basename(f))
        if m:
            r = json.loads(Path(f).read_text())
            out[tuple(int(g) for g in m.groups())] = r
            label = label or r.get("label") or ""
    return label or os.path.basename(os.path.normpath(d)), out


def fmt(r, key, mul, f):
    v = r.get(key)
    return "-" if v is None else f.format(v * mul)


def table(runs, baseline):
    multi = len(runs) > 1
    head = (["model"] if multi else []) + ["input", "output", "conc"] + [c[0] for c in COLS]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    base = baseline[1] if baseline else {}
    for label, cells in runs:
        for key in sorted(cells):
            row = [label] if multi else []
            row += [str(k) for k in key]
            failed = cells[key].get("failed") or 0
            if failed:
                row[-1] += f" ({failed} failed)"
            for _, k, mul, f in COLS:
                v = fmt(cells[key], k, mul, f)
                if key in base:
                    v = fmt(base[key], k, mul, f) + " → " + v
                row.append(v)
            print("| " + " | ".join(row) + " |")


def pivot(runs, metric):
    keys = sorted({k for _, cells in runs for k in cells})
    print(f"{metric}\n")
    print("| input | output | conc | " + " | ".join(label for label, _ in runs) + " |")
    print("|" + "---|" * (3 + len(runs)))
    for key in keys:
        vals = []
        for _, cells in runs:
            v = cells.get(key, {}).get(metric)
            vals.append("-" if v is None else f"{v:.1f}")
        print("| " + " | ".join(str(k) for k in key) + " | " + " | ".join(vals) + " |")


def write_csv(runs, path):
    fields = ["model", "input", "output", "concurrency", "completed", "failed"] + [c[1] for c in COLS]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for label, cells in runs:
            for key in sorted(cells):
                r = cells[key]
                w.writerow([label, *key, r.get("completed"), r.get("failed")] + [r.get(c[1]) for c in COLS])
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", help="result dirs (DIR or LABEL=DIR)")
    p.add_argument("--baseline", help="result dir to compare a single run against")
    p.add_argument("--pivot", metavar="METRIC", help="one column per run for this JSON key, e.g. median_tpot_ms")
    p.add_argument("--csv", metavar="FILE", help="also write every cell as CSV")
    a = p.parse_args()
    runs = [load(d) for d in a.dirs]
    if a.pivot:
        pivot(runs, a.pivot)
    else:
        table(runs, load(a.baseline) if a.baseline else None)
    if a.csv:
        write_csv(runs, a.csv)


if __name__ == "__main__":
    main()
