#!/usr/bin/env python3
"""Markdown tables for one (config, phase) of the report.

  phase_tables(lat_run, roof_runs, phase, steps, roof_steps, client_ms) -> (markdown, data)

lat_run    the capture whose window holds the whole phase (system trace: every execution)
roof_runs  [(run, steps)] captures whose device data give the roofline columns
"""
import json
import numpy as np, pandas as pd

ROOT = "/mnt/nvme/ds41_profiling"
ENG = [("tensor", "TensorE"), ("vector", "VectorE"), ("scalar", "ScalarE"), ("gpsimd", "GpSimd"),
       ("sync", "SP/Sync"), ("dma", "DMA"), ("cc", "Collectives")]
ROLE_NAME = {"embed+hc_expand": "embedding + hc_expand", "head+sampler": "final norm + lm_head + sampler",
             "draft kernel": "DSpark drafter (megakernel)", "draft glue": "drafter glue graphs"}


def load(run):
    ex = pd.read_parquet(f"{ROOT}/analysis/{run}/execs.parquet")
    meta = json.load(open(f"{ROOT}/analysis/{run}/meta.json"))
    ex["block"] = [f"L{int(r.layer):02d} {r.arch}" if r.role == "layer" else ROLE_NAME.get(r.role, r.role)
                   for r in ex.itertuples()]
    ex["lat_ns"] = ex.sys_ns.fillna(ex.dev_ns)
    return ex, meta


def steps_of(ex):
    f = ex.groupby("step").t0.min().sort_index()
    l = ex.groupby("step").t1.max()
    s = list(f.index)
    return {k: (f[k], f[s[i + 1]] if i + 1 < len(s) else l[k], i + 1 < len(s)) for i, k in enumerate(s)}


def breakdown(ex, phase, steps, client_ms=None):
    sw = steps_of(ex)
    rows = []
    for s in steps:
        e = ex[ex.step == s]
        t0, t1, _ = sw[s]
        r = {k: g.lat_ns.sum() for k, g in e.groupby("role")}
        busy = e.lat_ns.sum()
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
    names = {"layer": "40 transformer blocks (megakernels)", "draft kernel": ROLE_NAME["draft kernel"],
             "head+sampler": ROLE_NAME["head+sampler"], "embed+hc_expand": ROLE_NAME["embed+hc_expand"],
             "draft glue": ROLE_NAME["draft glue"], "idle_in": "device idle inside the forward (host dispatch / sync)",
             "idle_step": "device idle between iterations (vLLM scheduler, sampling output, input prep)"}
    b = per.drop("win").rename(names)
    if client_ms is not None:
        b["host time outside the device window (HTTP, tokenizer, first schedule, response)"] = client_ms * 1e6 - total
        total = client_ms * 1e6
    b = b[b > 1e3].sort_values(ascending=False)
    return b, total


def per_arch(ex, phase, steps):
    lay = ex[ex.step.isin(steps) & (ex.role == "layer")]
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
    t = d.dev_ns / 1e9
    d["useful"] = d.adj_flops - d.adj_flops_tr
    d["mfu"] = d.useful / (t * te)
    d["hfu"] = d.adj_flops / (t * te)
    d["hbm"] = d.hbm_rd + d.hbm_wr
    d["mbu"] = d.hbm / (t * bw)
    d["dma_act"] = d.act_dma / d.dev_ns
    d["dma_bw"] = d.dma_bytes / (t * dmapk)
    d["ai"] = d.useful / d.hbm.replace(0, np.nan)
    d["attain"] = np.minimum(te, d.ai * bw)
    d["roof_pct"] = (d.useful / t) / d.attain
    for k, lab in ENG:
        d[f"p_{lab}"] = d[f"act_{k}"] / d.dev_ns if f"act_{k}" in d else np.nan
    return d


def block_table(lat_ex, phase, steps, roof):
    """roof: DataFrame of roof_metrics rows (possibly from other runs)."""
    e = lat_ex[lat_ex.step.isin(steps)]
    k = len(steps) if phase == "decode" else 1
    lat = e.groupby("block").agg(ns=("lat_ns", "sum"), n=("lat_ns", "size"))
    lat["ns"] /= k
    lat["n"] //= k
    cols = ["mfu", "hfu", "mbu", "dma_act", "dma_bw", "ai", "roof_pct", "hbm", "useful"] + [f"p_{lab}" for _, lab in ENG]
    r = roof.groupby("block")[cols].mean()
    t = lat.join(r, how="left")
    pe = [f"p_{lab}" for _, lab in ENG]
    t["bneck"] = t[pe].idxmax(axis=1, skipna=True).str[2:] if t[pe].notna().any().any() else None
    t["bneck_pct"] = t[pe].max(axis=1)
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
    ex, meta = load(lat_run)
    b, total = breakdown(ex, phase, steps, client_ms)
    a = per_arch(ex, phase, steps)
    rs = []
    for run, rsteps in roof_runs:
        rex, rmeta = load(run)
        d = rex[rex.n_inst.notna() & rex.step.isin(rsteps)]
        rs.append(roof_metrics(d, rmeta))
    roof = pd.concat(rs) if rs else pd.DataFrame()
    t = block_table(ex, phase, steps, roof)
    lat_label = "ms per decode step" if phase == "decode" else "ms (whole phase)"
    md = {"breakdown": md_breakdown(b, total), "arch": md_arch(a), "blocks": md_blocks(t, total, lat_label)}
    data = {"breakdown_ms": {k: v / 1e6 for k, v in b.items()}, "total_ms": total / 1e6,
            "arch": a.assign(ms=a.ns / 1e6, per_ms=a.per / 1e6)[["n", "ms", "per_ms"]].to_dict("index"),
            "blocks": t.drop(columns=[]).assign(ms=t.ns / 1e6).to_dict("index")}
    return md, data, meta


if __name__ == "__main__":
    md, data, meta = phase_tables("in1024_bs10_decode_r2", [("in1024_bs10_decode_r2", [1, 2, 3, 4])], "decode", [1, 2, 3, 4])
    print(md["breakdown"]); print(md["arch"]); print(md["blocks"][:3000])
