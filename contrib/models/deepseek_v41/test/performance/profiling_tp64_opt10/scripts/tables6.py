#!/usr/bin/env python3
"""Markdown tables for one (config, phase) of the opt10 report, from analysis/<run>/units.parquet.

  phase_tables(lat_run, roof_runs, phase, steps, client_ms) -> (markdown dict, data dict, meta)

lat_run    the capture whose window holds the whole phase (latency of every execution)
roof_runs  [(run, steps)] captures whose device data give the roofline columns
A "block" row is one transformer block: a prefill launch, or a member of a fused decode launch.
"""
import json
import numpy as np, pandas as pd

ROOT = "/mnt/nvme/ds41_profiling_opt10"
ENG = [("tensor", "TensorE"), ("vector", "VectorE"), ("scalar", "ScalarE"), ("gpsimd", "GpSimd"),
       ("sync", "SP/Sync"), ("dma", "DMA"), ("cc", "Collectives")]
COMP = {  # unit -> breakdown component
    "kernel-entry barrier": "kernel-entry barriers of the fused launches (waiting for all 64 ranks)",
    "embed + hc_expand (fused)": "embedding + hc_expand (fused into the first launch)",
    "greedy head (fused)": "final norm + lm_head + greedy argmax (fused into the last launch)",
    "embedding + hc_expand": "embedding + hc_expand",
    "final norm + lm_head + sampler": "final norm + lm_head + sampler",
    "tail all-gather graph": "final norm + lm_head + sampler",
    "rejection sampler": "glue graphs (rejection sampler, gathers, drafter meta)",
    "gather/concat graph": "glue graphs (rejection sampler, gathers, drafter meta)",
    "drafter meta": "glue graphs (rejection sampler, gathers, drafter meta)",
    "glue": "glue graphs (rejection sampler, gathers, drafter meta)",
    "step prologue": "step prologue (NKI: the step's index columns, n-gram ids, starts)",
    "decode tail graph": "decode tail graph (draft verification + drafter inputs)",
    "token history put": "glue graphs (rejection sampler, gathers, drafter meta)",
}
BLOCKS = "40 transformer blocks (megakernels)"
DRAFT = "DSpark drafter (megakernel)"


def load(run):
    U = pd.read_parquet(f"{ROOT}/analysis/{run}/units.parquet")
    ex = pd.read_parquet(f"{ROOT}/analysis/{run}/execs.parquet")
    meta = json.load(open(f"{ROOT}/analysis/{run}/meta.json"))
    U["block"] = U.unit.where(U.role != "draft", DRAFT)
    U["comp"] = [BLOCKS if r.layer == r.layer and r.layer is not None else
                 (DRAFT if r.role == "draft" else COMP.get(r.unit, r.unit)) for r in U.itertuples()]
    # latency: a whole execution from the live trace (nc_exec_running), a member from the device timeline
    sys_ns = U.exec.map(ex.set_index("exec").sys_ns)
    single = U.groupby("exec").u.transform("size") == 1
    U["lat_ns"] = np.where(single & sys_ns.notna(), sys_ns, U.dur_ns)
    return U, ex, meta


def steps_of(ex):
    f = ex.groupby("step").t0.min().sort_index()
    l = ex.groupby("step").t1.max()
    s = list(f.index)
    return {k: (f[k], f[s[i + 1]] if i + 1 < len(s) else l[k], i + 1 < len(s)) for i, k in enumerate(s)}


def breakdown(U, ex, phase, steps, client_ms=None):
    sw = steps_of(ex)
    rows = []
    for s in steps:
        e = ex[ex.step == s]
        u = U[U.step == s]
        t0, t1, _ = sw[s]
        r = u.groupby("comp").lat_ns.sum().to_dict()
        busy = u.lat_ns.sum()
        r["idle_in"] = max(0, (e.t1.max() - e.t0.min()) - busy)
        r["idle_step"] = max(0, t1 - e.t1.max())
        r["win"] = t1 - t0
        rows.append(r)
    L = pd.DataFrame(rows).fillna(0)
    if phase == "decode":
        per = L.mean()
        total = per["win"]
    else:
        per = L.sum()
        e = ex[ex.step.isin(steps)]
        total = e.t1.max() - e.t0.min()
        per["idle_step"] = total - per.drop(["win", "idle_step"]).sum()
    if phase == "decode":     # asynchronous scheduling: the next launch is already queued when one ends
        names = {"idle_in": "device idle between the step's launches (launch turnaround; the next launch is queued)",
                 "idle_step": "device idle between steps (drafter end → next step prologue)"}
    else:
        names = {"idle_in": "device idle between launches (host submission + launch turnaround)",
                 "idle_step": "device idle between iterations (vLLM scheduler, output processing, input prep)"}
    b = per.drop("win").rename(names)
    if client_ms is not None:
        b["host time outside the device window (HTTP, tokenizer, first schedule, response)"] = client_ms * 1e6 - total
        total = client_ms * 1e6
    b = b[b > 1e3].sort_values(ascending=False)
    return b, total


def per_arch(U, phase, steps):
    lay = U[U.step.isin(steps) & U.layer.notna()]
    k = len(steps) if phase == "decode" else 1
    a = lay.groupby("arch").agg(n=("lat_ns", "size"), ns=("lat_ns", "sum"))
    a["n"] //= k
    a["ns"] /= k
    a["per"] = a.ns / a.n
    return a.sort_values("ns", ascending=False)


def roof_metrics(d, meta):
    te, bw = meta["te_peak_bf16_flops"], meta["hbm_bw"]
    dmapk = meta["dma_bw_per_pcore"] * meta["num_physical_cores"]
    d = d.copy()
    t = d.dur_ns / 1e9
    d["useful"] = d.adj_flops - d.adj_flops_tr
    d["mfu"] = d.useful / (t * te)
    d["hfu"] = d.adj_flops / (t * te)
    d["hbm"] = d.hbm_rd + d.hbm_wr
    d["mbu"] = d.hbm / (t * bw)
    d["dma_act"] = d.act_dma / d.dur_ns
    d["dma_bw"] = d.dma_bytes / (t * dmapk)
    d["ai"] = d.useful / d.hbm.replace(0, np.nan)
    for k, lab in ENG:
        d[f"p_{lab}"] = d[f"act_{k}"] / d.dur_ns if f"act_{k}" in d else np.nan
    return d


def block_table(U, phase, steps, roof):
    e = U[U.step.isin(steps)]
    k = len(steps) if phase == "decode" else 1
    lat = e.groupby("block").agg(ns=("lat_ns", "sum"), n=("lat_ns", "size"))
    lat["ns"] /= k
    lat["n"] //= k
    cols = ["mfu", "hfu", "mbu", "dma_act", "dma_bw", "ai", "hbm", "useful"] + [f"p_{lab}" for _, lab in ENG]
    r = roof.groupby("block")[cols].mean()
    t = lat.join(r, how="left")
    pe = [f"p_{lab}" for _, lab in ENG]
    sub = t[pe].astype(float)
    t["bneck"] = sub.fillna(-1).idxmax(axis=1).str[2:].where(sub.notna().any(axis=1))
    t["bneck_pct"] = sub.max(axis=1)
    return t.sort_values("ns", ascending=False)


def pct(x):
    return "–" if pd.isna(x) else f"{100 * x:.1f}"


def md_breakdown(b, total):
    out = ["| Component | ms | % |", "|---|---:|---:|"]
    for k, v in b.items():
        out.append(f"| {k} | {v / 1e6:,.2f} | {100 * v / total:.1f} |")
    out.append(f"| **Total** | **{total / 1e6:,.2f}** | 100 |")
    return "\n".join(out)


def md_arch(a):
    tot = a.ns.sum()
    out = ["| Block archetype | blocks | ms (all blocks of this type) | ms per block | % of block time |",
           "|---|---:|---:|---:|---:|"]
    for k, r in a.iterrows():
        out.append(f"| `{k}` | {int(r.n)} | {r.ns / 1e6:,.2f} | {r.per / 1e6:.3f} | {100 * r.ns / tot:.1f} |")
    return "\n".join(out)


def md_blocks(t, total, lat_label):
    out = [f"| Block | launches | {lat_label} | % of total | MFU % | HFU % | MBU % | DMA active % | DMA BW % | AI (FLOP/B) | "
           f"Bottleneck engine (busy %) | Collectives % | TensorE % | VectorE % | GpSimd % |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for b, r in t.iterrows():
        bn = "–" if pd.isna(r.bneck_pct) else f"{r.bneck} ({100 * r.bneck_pct:.0f})"
        ai = "–" if pd.isna(r.ai) else f"{r.ai:.0f}"
        out.append(f"| {b} | {int(r.n)} | {r.ns / 1e6:,.3f} | {100 * r.ns / total:.1f} | {pct(r.mfu)} | {pct(r.hfu)} | "
                   f"{pct(r.mbu)} | {pct(r.dma_act)} | {pct(r.dma_bw)} | {ai} | {bn} | "
                   f"{pct(r.p_Collectives)} | {pct(r.p_TensorE)} | {pct(r.p_VectorE)} | {pct(r.p_GpSimd)} |")
    return "\n".join(out)


def phase_tables(lat_run, roof_runs, phase, steps, client_ms=None):
    U, ex, meta = load(lat_run)
    b, total = breakdown(U, ex, phase, steps, client_ms)
    a = per_arch(U, phase, steps)
    rs = []
    for run, rsteps in roof_runs:
        RU, rex, rmeta = load(run)
        d = RU[RU.n_inst.notna() & RU.step.isin(rsteps) & (RU.dur_ns > 0)]
        rs.append(roof_metrics(d, rmeta))
    roof = pd.concat(rs) if rs else pd.DataFrame()
    t = block_table(U, phase, steps, roof)
    lat_label = "ms per decode step" if phase == "decode" else "ms (whole phase)"
    md = {"breakdown": md_breakdown(b, total), "arch": md_arch(a), "blocks": md_blocks(t, total, lat_label)}
    data = {"breakdown_ms": {k: v / 1e6 for k, v in b.items()}, "total_ms": total / 1e6,
            "arch": a.assign(ms=a.ns / 1e6, per_ms=a.per / 1e6)[["n", "ms", "per_ms"]].to_dict("index"),
            "blocks": t.assign(ms=t.ns / 1e6).to_dict("index")}
    return md, data, meta


if __name__ == "__main__":
    import sys
    run, phase, steps = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(",")]
    md, data, meta = phase_tables(run, [(run, steps)], phase, steps)
    print(md["breakdown"]); print(md["arch"]); print(md["blocks"])
