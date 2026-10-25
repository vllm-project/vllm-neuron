#!/usr/bin/env python3
"""Per-block collective time by kind: kernel-entry barrier, 8-rank attention all-reduce,
64-rank world all-reduce (and any other), from CcOp rows inside each execution.

  cc_breakdown.py <run> <steps>   e.g. 1,2,3   -> json + table
"""
import glob, json, sys
import pandas as pd

ROOT = "/mnt/nvme/ds41_profiling"


def kind(r):
    if r.operation == "BARRIER":
        return "barrier"
    g = r.replica_group
    n = len(str(g).split("], [")[0].split(",")) if isinstance(g, str) else None
    return f"{r.operation.lower()}-{n}r"


def main(run, steps):
    ex = pd.read_parquet(f"{ROOT}/analysis/{run}/execs.parquet")
    ex = ex[ex.step.isin(steps) & ex.n_inst.notna()]
    dev = glob.glob(f"{ROOT}/data/{run}/profiles/global/*session_*@latest")[0]
    cc = pd.read_parquet(f"{dev}/CcOp.parquet")
    cc = cc[cc.operation != "Invalid"].sort_values("start_ts")
    cc["exec"] = pd.merge_asof(cc[["start_ts"]], ex[["t0", "exec"]].sort_values("t0"), left_on="start_ts", right_on="t0")["exec"].values
    cc = cc[cc.exec.isin(ex.exec)]
    cc = cc.merge(ex[["exec", "role", "layer", "t1"]], on="exec")
    cc = cc[cc.start_ts < cc.t1]
    cc["kind"] = [kind(r) for r in cc.itertuples()]
    cc["grp"] = cc.role.where(cc.role != "layer", "layer")
    n_exec = ex.groupby("role").size()
    t = cc.groupby(["grp", "kind"]).agg(n=("duration_ns", "size"), us=("duration_ns", "sum"),
                                         bytes=("input_size", "max"), dtype=("dtype", "first")).reset_index()
    t["per_exec_us"] = t.us / t.grp.map(n_exec) / 1e3
    t["mean_us"] = t.us / t.n / 1e3
    t["count_per_exec"] = t.n / t.grp.map(n_exec)
    t["MB"] = t.bytes / 1e6
    print(t[["grp", "kind", "dtype", "MB", "count_per_exec", "mean_us", "per_exec_us"]].round(2).to_string(index=False))
    json.dump(t.to_dict("records"), open(f"{ROOT}/analysis/{run}/cc_{'_'.join(map(str, steps))}.json", "w"), default=str, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], [int(s) for s in sys.argv[2].split(",")])
