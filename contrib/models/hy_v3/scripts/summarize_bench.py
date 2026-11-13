"""Print a markdown table from `vllm bench serve` result JSONs (bench_matrix.sh output).

Usage: python summarize_bench.py <result_dir> [<baseline_dir>]
With a baseline dir, each cell shows "baseline -> new" for cells present in both.
"""

import glob
import json
import os
import re
import sys
from pathlib import Path

COLS = [
    ("TTFT med (s)", "median_ttft_ms", 1e-3, "{:.2f}"),
    ("TTFT P99 (s)", "p99_ttft_ms", 1e-3, "{:.2f}"),
    ("TPOT med (ms)", "median_tpot_ms", 1, "{:.1f}"),
    ("ITL med (ms)", "median_itl_ms", 1, "{:.1f}"),
    ("out tok/s", "output_throughput", 1, "{:.0f}"),
    ("peak out tok/s", "max_output_tokens_per_s", 1, "{:.0f}"),
    ("total tok/s", "total_token_throughput", 1, "{:.0f}"),
]


def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        m = re.match(r"in(\d+)_out(\d+)_c(\d+)\.json", os.path.basename(f))
        if m:
            out[(int(m.group(1)), int(m.group(3)))] = json.loads(Path(f).read_text())
    return out


new = load(sys.argv[1])
base = load(sys.argv[2]) if len(sys.argv) > 2 else {}
print("| input | conc | " + " | ".join(c[0] for c in COLS) + " |")
print("|" + "---|" * (len(COLS) + 2))
for key in sorted(new):
    cells = []
    for _, k, mul, fmt in COLS:
        v = fmt.format(new[key][k] * mul)
        if key in base:
            v = fmt.format(base[key][k] * mul) + " → " + v
        cells.append(v)
    print(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + " |")
