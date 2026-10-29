#!/usr/bin/env python3
"""Instruction time and HBM bytes per NKI function for chosen units (opt6).

  hot6.py <run> <unit-or-arch-or-role> [--steps S,..] [--top N]

A unit is a prefill launch or a member of a fused decode launch (analysis/<run>/units.parquet);
its instructions are those that start inside its time window. Source locations are mapped to
the enclosing function of the kernel package (timeline6.resolve).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import duckdb, pandas as pd
from timeline6 import resolve

ROOT = "/mnt/nvme/ds41_profiling_opt6"


def hotspots(run, block, steps=None):
    U = pd.read_parquet(f"{ROOT}/analysis/{run}/units.parquet")
    meta = json.load(open(f"{ROOT}/analysis/{run}/meta.json"))
    sel = U[(U.unit == block) | (U.arch == block) | (U.role == block) | (U.kernel == block)]
    if steps is not None:
        sel = sel[sel.step.isin(steps)]
    sel = sel[sel.n_inst.notna() & (sel.dur_ns > 0)]
    con = duckdb.connect()
    con.sql("set threads to 24")
    con.register("S", sel[["u", "exec", "t0", "t1"]])
    I = f"read_parquet('{meta['dev_dir']}/Instruction.parquet')"
    locs = con.sql(f"""select distinct nki_source_location loc from {I} i join S on i.execution_index = S.exec
        and i.start_ts >= S.t0 and i.start_ts < S.t1 where execution_unit='compute'""").df()
    locs["func"] = locs["loc"].map(resolve)
    con.register("locs", locs)
    base = f"""select l.func, i.engine, i.pcore_idx, S.u ex, i.start_ts s, least(i.end_ts, S.t1) e,
        coalesce(i.hbm_read_bytes,0)+coalesce(i.hbm_write_bytes,0) hbm, coalesce(i.adjusted_flops,0) fl
        from {I} i join S on i.execution_index = S.exec and i.start_ts >= S.t0 and i.start_ts < S.t1
        join locs l on i.nki_source_location is not distinct from l.loc where i.execution_unit='compute'"""
    wall = con.sql(f"""
      with p as ({base}),
      q as (select func, engine, pcore_idx, ex, s, e, max(e) over (partition by func, engine, pcore_idx, ex order by s, e
                 rows between unbounded preceding and 1 preceding) pm from p),
      r as (select *, sum(case when pm is null or s > pm then 1 else 0 end)
                 over (partition by func, engine, pcore_idx, ex order by s, e rows unbounded preceding) g from q)
      select func, engine, sum(ge - gs) ns from (select func, engine, pcore_idx, ex, g, min(s) gs, max(e) ge from r group by all)
      group by all""").df()
    tot = con.sql(f"select func, sum(hbm) hbm, sum(fl) fl from ({base}) group by 1").df().set_index("func")
    anyw = con.sql(f"""
      with p as ({base}),
      q as (select func, ex, s, e, max(e) over (partition by func, ex order by s, e
                 rows between unbounded preceding and 1 preceding) pm from p),
      r as (select *, sum(case when pm is null or s > pm then 1 else 0 end)
                 over (partition by func, ex order by s, e rows unbounded preceding) g from q)
      select func, sum(ge - gs) ns from (select func, ex, g, min(s) gs, max(e) ge from r group by all)
      group by all""").df().set_index("func")
    n = len(sel)
    p = wall.pivot_table(index="func", columns="engine", values="ns", aggfunc="sum").fillna(0) / n / 1e6 / 2
    p["any-engine wall ms"] = anyw.ns / n / 1e6
    p["HBM MB"] = tot.hbm / n / 1e6
    p["GFLOP(bf16-eq)"] = tot.fl / n / 1e9
    p = p.sort_values("any-engine wall ms", ascending=False)
    return p, n, sel.dur_ns.mean() / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("block")
    ap.add_argument("--steps", default=None)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    p, n, ms = hotspots(a.run, a.block, [int(s) for s in a.steps.split(",")] if a.steps else None)
    print(f"{a.run} {a.block}: {n} units, mean ms {ms:.3f}")
    print(p.head(a.top).round(3).to_string())


if __name__ == "__main__":
    main()
