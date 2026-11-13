"""Tables for docs/PROFILE_85K.md from neuron-explorer parquet exports.

Input: one parquet export per (rank, session), directories named
``<anything>_pid_<pid>_<session id>`` under <pq dir>, and a ``<pid> Worker_TP<r>_...``
map (from the server log). Sessions are told apart by their start order.

Usage: python profile_analyze.py <pq dir> <pid_rank.txt> [session names, comma-separated]

Per session and NEFF: executions and median time. Per instruction-traced
execution of the target NEFF: wall time by module (union of instruction
intervals), collectives and idle time. Per layer and rank: the MoE collective
against the attention collective of the same size (its excess is waiting for
other ranks), and the routed-expert kernel time.
"""

import glob
import re
import sys
from collections import defaultdict

import pandas as pd

pq_dir, rank_map = sys.argv[1], sys.argv[2]
names = sys.argv[3].split(",") if len(sys.argv) > 3 else None
pid_rank = {}
with open(rank_map) as f:
    for line in f:
        pid, w = line.split()
        pid_rank[pid] = int(re.search(r"TP(\d+)", w).group(1))

by_pid = defaultdict(list)  # pid -> [(session id, parquet dir)]; session ids differ per rank
for d in sorted(glob.glob(f"{pq_dir}/*_pid_*")):
    m = re.search(r"_pid_(\d+)_(\d+)$", d)
    if m:
        by_pid[m.group(1)].append((int(m.group(2)), d))
runs = defaultdict(dict)  # session index -> rank -> parquet dir
for pid, lst in by_pid.items():
    for i, (_, d) in enumerate(sorted(lst)):
        sub = glob.glob(d + "/*/ExecutionInfo.parquet")
        if sub:
            runs[i][pid_rank[pid]] = sub[0].rsplit("/", 1)[0] + "/"
sessions = sorted(runs)
names = names or [f"session{i}" for i in range(len(sessions))]


def md(df):
    """Markdown table without the tabulate dependency."""
    cols = [str(c) for c in df.columns]
    out = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, row in df.iterrows():
        out.append(f"| {idx} | " + " | ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(out)


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


def category(layer, src):
    layer = layer or ""
    if "shared_mlp" in layer:
        return "shared expert"
    if "HYV3MoE" in layer:
        return "routed experts (kernel)" if src else "router / MoE glue"
    if "HYV3DenseMLP" in layer:
        return "dense MLP (layer 0)"
    if "self_attn" in layer or "HYV3Attention" in layer:
        return "attention"
    if layer in ("", "Unknown"):
        return "unattributed"
    return "other (embed, norm, lm_head, sampler, MTP glue)"


def load(base):
    e = pd.read_parquet(base + "ExecutionInfo.parquet").sort_values("execution_start_ts").reset_index(drop=True)
    c = pd.read_parquet(base + "CcOp.parquet")
    c = c[c.operation.isin(["ALLREDUCE", "REDUCE_SCATTER", "ALLGATHER"])]
    return e, c


def target_neff(e, c):
    """The NEFF whose executions carry 161 all-reduces or reduce-scatters (the 80-layer target)."""
    best = None
    for neff, g in e.groupby("neff_name"):
        a, b = g.iloc[0].execution_start_ts, g.iloc[0].execution_end_ts
        n = ((c.start_ts >= a) & (c.end_ts <= b) & c.operation.isin(["ALLREDUCE", "REDUCE_SCATTER"])).sum()
        if n == 161:
            best = neff if best is None or g.ms.median() > e[e.neff_name == best].ms.median() else best
    return best


for sid, name in zip(sessions, names):
    ranks = runs[sid]
    print(f"\n## {name} (ranks {sorted(ranks)})\n")
    r0 = min(ranks)
    e, c = load(ranks[r0])
    e["ms"] = (e.execution_end_ts - e.execution_start_ts) / 1e6
    tgt = target_neff(e, c)
    kind = {n: ("target" if n == tgt else f"other {n[:8]}") for n in e.neff_name.unique()}
    print("| NEFF | executions | median ms | min | max |\n|---|---|---|---|---|")
    for n, g in e.groupby("neff_name"):
        print(f"| {kind[n]} | {len(g)} | {g.ms.median():.2f} | {g.ms.min():.2f} | {g.ms.max():.2f} |")
    t = e[e.neff_name == tgt]
    print(f"\ntarget executions in order (ms): {', '.join(f'{x:.0f}' for x in t.ms)}")

    # Module breakdown of the traced target executions (rank r0).
    ins = pd.read_parquet(ranks[r0] + "Instruction.parquet",
                          columns=["execution_index", "start_ts", "end_ts", "layer", "nki_source_location"])
    traced = sorted(set(ins.execution_index) & set(t.execution_index))
    rows = []
    for ex in traced:
        row = e[e.execution_index == ex].iloc[0]
        a, b = row.execution_start_ts, row.execution_end_ts
        d = ins[ins.execution_index == ex]
        cats = d.apply(lambda r: category(r.layer, r.nki_source_location), axis=1)
        res = {k: union(zip(g.start_ts, g.end_ts)) / 1e6 for k, g in d.groupby(cats)}
        cc = c[(c.start_ts >= a) & (c.end_ts <= b)]
        busy = union(list(zip(d.start_ts, d.end_ts)) + list(zip(cc.start_ts, cc.end_ts))) / 1e6
        res["collectives (any overlap)"] = union(zip(cc.start_ts, cc.end_ts)) / 1e6
        res["idle (no instruction, no collective)"] = (b - a) / 1e6 - busy
        res["_total"] = (b - a) / 1e6
        rows.append(pd.Series(res, name=f"exec {ex}"))
    if rows:
        bd = pd.DataFrame(rows).T.fillna(0).round(2)
        print(f"\nTP{r0} target, wall ms by module (instruction-interval union; modules overlap little):\n")
        print(md(bd))

    # Per layer and rank: MoE vs attention collective, routed-kernel time.
    per = []
    for r, base in sorted(ranks.items()):
        er, cr = load(base)
        tr = er[er.neff_name == tgt]
        for _, row in tr.iterrows():
            a, b = row.execution_start_ts, row.execution_end_ts
            cc = cr[(cr.start_ts >= a) & (cr.end_ts <= b) & cr.operation.isin(["ALLREDUCE", "REDUCE_SCATTER"])]
            cc = cc.sort_values("start_ts").reset_index(drop=True)
            if len(cc) != 161:
                continue
            # Layer L: attention compute, attention collective (1 + 2L), MLP / MoE
            # compute, MLP collective (2 + 2L); collective 0 is the embedding's.
            for L in range(1, 80):
                prev, attn, moe = cc.iloc[2 * L], cc.iloc[1 + 2 * L], cc.iloc[2 + 2 * L]
                per.append({"rank": r, "exec": int(row.execution_index), "L": L,
                            "attn_phase_ms": (attn.start_ts - prev.end_ts) / 1e6,
                            "moe_phase_ms": (moe.start_ts - attn.end_ts) / 1e6,
                            "attn_cc_ms": attn.duration_ns / 1e6, "moe_cc_ms": moe.duration_ns / 1e6,
                            "moe_cc_start_ms": (moe.start_ts - a) / 1e6})
    p = pd.DataFrame(per)
    if p.empty:
        continue
    p["wait_ms"] = p.moe_cc_ms - p.attn_cc_ms
    by_exec = p.groupby(["exec", "rank"]).agg(
        attn_phase=("attn_phase_ms", "sum"), moe_phase=("moe_phase_ms", "sum"),
        attn_cc=("attn_cc_ms", "sum"), moe_cc=("moe_cc_ms", "sum"), wait=("wait_ms", "sum"))
    s = by_exec.groupby("exec").agg(
        attention=("attn_phase", "median"), attn_collective=("attn_cc", "median"),
        moe_compute=("moe_phase", "median"), moe_collective=("moe_cc", "median"),
        moe_wait_min_rank=("wait", "min"), moe_wait_max_rank=("wait", "max")).round(2)
    s["target_ms"] = t.set_index("execution_index").ms.reindex(s.index).round(1)
    print("\nPer target execution, layers 1-79 by phase (ms, median over ranks; the MoE collective minus "
          "the same-size attention collective is waiting for other ranks, min / max over sampled ranks):\n")
    print(md(s))
    print("\nmedian over executions:", s.median().round(2).to_dict())
    # Skew: per layer, spread of the MoE-collective start across ranks (relative to each rank's exec start).
    sk = p.pivot_table(index=["exec", "L"], columns="rank", values="moe_cc_start_ms")
    spread = (sk.max(axis=1) - sk.min(axis=1)) * 1e3
    print(f"\nMoE collective start spread across the {len(ranks)} ranks, per layer (us): "
          f"median {spread.median():.0f}, p90 {spread.quantile(.9):.0f}, max {spread.max():.0f}")
    # Which rank waits least (= is the slowest) per layer.
    w = p.pivot_table(index=["exec", "L"], columns="rank", values="wait_ms")
    slowest = w.idxmin(axis=1).value_counts().sort_index()
    print("slowest sampled rank (least waiting) per layer, count:", dict(slowest))

    # Routed-expert kernel time per layer and rank (traced executions only).
    k = []
    for r, base in sorted(ranks.items()):
        ii = pd.read_parquet(base + "Instruction.parquet",
                             columns=["execution_index", "start_ts", "end_ts", "layer", "nki_source_location"])
        er, _ = load(base)
        ex_t = set(er[er.neff_name == tgt].execution_index)
        ii = ii[ii.execution_index.isin(ex_t) & ii.layer.fillna("").str.contains("HYV3MoE")
                & ~ii.layer.fillna("").str.contains("shared") & ii.nki_source_location.notna()]
        if ii.empty:
            continue
        ii = ii.assign(L=ii.layer.str.extract(r"HYV3DecoderLayer\[(\d+)\]")[0].astype(float))
        for (ex, L), g in ii.groupby(["execution_index", "L"]):
            k.append({"rank": r, "exec": int(ex), "L": int(L), "kernel_ms": union(zip(g.start_ts, g.end_ts)) / 1e6})
    kd = pd.DataFrame(k)
    if not kd.empty:
        kp = kd.pivot_table(index=["exec", "L"], columns="rank", values="kernel_ms")
        print(f"\nRouted-expert kernel per layer (ms), traced executions, {kp.shape[1]} ranks with module tags:")
        print(md(kp.describe().loc[["mean", "50%", "max"]].round(3)))
        rng = (kp.max(axis=1) - kp.min(axis=1))
        print(f"per-layer max - min across ranks: median {rng.median():.3f} ms, p90 {rng.quantile(.9):.3f} ms; "
              f"max/mean median {(kp.max(axis=1) / kp.mean(axis=1)).median():.2f}")
