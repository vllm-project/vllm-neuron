"""Attention-module time of traced decode executions, by source function (EXPERIMENTAL sparse decode).

Input: a neuron-explorer parquet export (see docs/PROFILE_85K.md) and the
model.py the profiled server ran (its line numbers attribute the torch ops).
Per traced execution: the union of the instruction intervals of every
HYV3Attention instruction, grouped by the model.py function its source line
falls in (the sparse-decode helpers), the NKI kernel file, or "no source";
plus the top source lines of the sparse selection.

Usage: python profile_attn_breakdown.py <export dir with Instruction.parquet> <model.py> [exec index ...]
"""

import ast
import re
import sys

import pandas as pd


def union(iv):
    tot, cs, ce = 0, None, None
    for s, t in sorted(iv):
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, t
        else:
            ce = max(ce, t)
    return tot + (ce - cs if cs is not None else 0)


def functions(path):
    """[(first line, last line, name)] of the module- and class-level functions."""
    with open(path) as f:
        tree = ast.parse(f.read())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            out.append((node.lineno, node.end_lineno, node.name))
    return sorted(out)


def main():
    base, model_py = sys.argv[1].rstrip("/") + "/", sys.argv[2]
    funcs = functions(model_py)
    with open(model_py) as f:
        lines = f.read().split("\n")

    def group(src):
        if not isinstance(src, str) or not src:
            return "no source"
        m = re.search(r"model\.py:(\d+)", src)
        if m:
            n = int(m.group(1))
            inner = [f for f in funcs if f[0] <= n <= f[1]]
            return "torch: " + (min(inner, key=lambda f: f[1] - f[0])[2] if inner else "model.py")
        return "NKI: " + src.split("/")[-1].split(":")[0]

    ins = pd.read_parquet(base + "Instruction.parquet",
                          columns=["execution_index", "start_ts", "end_ts", "layer", "nki_source_location"])
    execs = [int(x) for x in sys.argv[3:]] or sorted(set(ins.execution_index))[:2]
    for ex in execs:
        d = ins[(ins.execution_index == ex) & ins.layer.fillna("").str.contains("HYV3Attention")]
        if d.empty:
            continue
        g = d.nki_source_location.map(group)
        print(f"\nexec {ex}: attention module {union(zip(d.start_ts, d.end_ts)) / 1e6:.2f} ms (union)\n")
        print("| source | ms | instructions |\n|---|---|---|")
        parts = sorted(((union(zip(x.start_ts, x.end_ts)) / 1e6, k, len(x)) for k, x in d.groupby(g)), reverse=True)
        for ms, k, cnt in parts:
            if ms >= 0.05:
                print(f"| {k} | {ms:.2f} | {cnt} |")
        sel = d[g == "torch: sparse_select_blocks"]
        if len(sel):
            print("\nsparse_select_blocks by line:\n\n| line | ms | code |\n|---|---|---|")
            n = sel.nki_source_location.str.extract(r"model\.py:(\d+)")[0].astype(int)
            rows = sorted(((union(zip(x.start_ts, x.end_ts)) / 1e6, ln) for ln, x in sel.groupby(n)), reverse=True)
            for ms, ln in rows[:10]:
                print(f"| {ln} | {ms:.2f} | `{lines[ln - 1].strip()[:70]}` |")


if __name__ == "__main__":
    main()
