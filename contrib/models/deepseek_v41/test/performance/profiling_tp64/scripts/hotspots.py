#!/usr/bin/env python3
"""Instruction time and HBM bytes per kernel function for chosen executions.

  hotspots.py <run> <role-or-block> [--step S] [--top N]

Source locations (NKI debug info) are mapped to the enclosing function of the
kernel package (kernels/tp64/*.py; the NEFFs record kernels/<file>, same content).
"""
import argparse, ast, glob, os, re
import duckdb, pandas as pd

ROOT = "/mnt/nvme/ds41_profiling"
KDIR = "/home/ubuntu/vllm-neuron/contrib/models/deepseek_v41/src/kernels/tp64"


def func_map():
    m = {}
    for path in glob.glob(f"{KDIR}/*.py"):
        tree = ast.parse(open(path).read())
        spans = []

        def visit(node, prefix):
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    q = f"{prefix}.{ch.name}" if prefix else ch.name
                    spans.append((ch.lineno, ch.end_lineno, q))
                    visit(ch, q)
                else:
                    visit(ch, prefix)
        visit(tree, "")
        m[os.path.basename(path)] = sorted(spans, key=lambda s: s[1] - s[0])
    return m


FM = func_map()


def resolve(loc):
    if not isinstance(loc, str):
        return "(no source location)"
    mm = re.search(r"([^/]+\.py):(\d+)", loc)
    if not mm:
        return loc
    f, ln = mm.group(1), int(mm.group(2))
    for a, b, name in FM.get(f, []):
        if a <= ln <= b:
            return f"{f}:{name}"
    return f"{f}:?"


def hotspots(run, block, steps=None):
    """(table per function, number of executions, mean device ms)."""
    class A: pass
    a = A(); a.run, a.block = run, block
    ex = pd.read_parquet(f"{ROOT}/analysis/{a.run}/execs.parquet")
    ex["blk"] = [f"L{int(r.layer):02d}" if r.role == "layer" else r.role for r in ex.itertuples()]
    sel = ex[(ex.blk == a.block) | (ex.role == a.block) | (ex.arch == a.block)]
    if steps is not None:
        sel = sel[sel.step.isin(steps)]
    sel = sel[sel.n_inst.notna()]
    ids = ",".join(str(int(e)) for e in sel.exec)
    dev = glob.glob(f"{ROOT}/data/{a.run}/profiles/global/*session_*@latest")[0]
    con = duckdb.connect()
    con.sql("set threads to 32")
    I = f"'{dev}/Instruction.parquet'"
    locs = con.sql(f"""select distinct coalesce(nki_source_location, bir_debug_info_source_location) loc from {I}
        where execution_index in ({ids}) and execution_unit='compute'""").df()
    locs["func"] = locs["loc"].map(resolve)
    con.register("locs", locs)
    base = f"""select l.func, i.engine, i.pcore_idx, i.execution_index ex, i.start_ts s, i.end_ts e,
        coalesce(i.hbm_read_bytes,0)+coalesce(i.hbm_write_bytes,0) hbm, coalesce(i.adjusted_flops,0) fl
        from {I} i join locs l on coalesce(i.nki_source_location, i.bir_debug_info_source_location) is not distinct from l.loc
        where i.execution_index in ({ids}) and i.execution_unit='compute'"""
    # busy wall per (function, engine): interval-merge each physical core's instructions, then average the cores
    wall = con.sql(f"""
      with p as ({base}),
      q as (select func, engine, pcore_idx, ex, s, e, max(e) over (partition by func, engine, pcore_idx, ex order by s, e
                 rows between unbounded preceding and 1 preceding) pm from p),
      r as (select *, sum(case when pm is null or s > pm then 1 else 0 end)
                 over (partition by func, engine, pcore_idx, ex order by s, e rows unbounded preceding) g from q)
      select func, engine, sum(ge - gs) ns from (select func, engine, pcore_idx, ex, g, min(s) gs, max(e) ge from r group by all)
      group by all""").df()
    tot = con.sql(f"select func, sum(hbm) hbm, sum(fl) fl from ({base}) group by 1").df().set_index("func")
    # wall during which any engine of either core runs the function (its footprint on the block's timeline)
    anyw = con.sql(f"""
      with p as ({base}),
      q as (select func, ex, s, e, max(e) over (partition by func, ex order by s, e
                 rows between unbounded preceding and 1 preceding) pm from p),
      r as (select *, sum(case when pm is null or s > pm then 1 else 0 end)
                 over (partition by func, ex order by s, e rows unbounded preceding) g from q)
      select func, sum(ge - gs) ns from (select func, ex, g, min(s) gs, max(e) ge from r group by all)
      group by all""").df().set_index("func")
    n = len(sel)
    p = wall.pivot_table(index="func", columns="engine", values="ns", aggfunc="sum").fillna(0) / n / 1e6 / 2  # per physical core
    p["any-engine wall ms"] = anyw.ns / n / 1e6
    p["HBM MB"] = tot.hbm / n / 1e6
    p["GFLOP(bf16-eq)"] = tot.fl / n / 1e9
    p["busiest-engine ms"] = p[[c for c in p.columns if c in ("Tensor", "Vector", "Scalar", "GpSimd", "Sync")]].max(axis=1)
    p = p.sort_values("any-engine wall ms", ascending=False)
    return p, n, sel.dev_ns.mean() / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("block")
    ap.add_argument("--steps", default=None)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    p, n, ms = hotspots(a.run, a.block, [int(s) for s in a.steps.split(",")] if a.steps else None)
    print(f"{a.run} {a.block}: {n} executions, mean device ms {ms:.3f}")
    print(p.head(a.top).round(3).to_string())


if __name__ == "__main__":
    main()
