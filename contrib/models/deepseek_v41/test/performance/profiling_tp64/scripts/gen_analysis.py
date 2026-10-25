#!/usr/bin/env python3
"""Cross-configuration tables for the report (written to report_parts/80_tables.md)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import tables
from cc_breakdown import kind
import glob

ROOT = "/mnt/nvme/ds41_profiling"
CONFIGS = json.load(open(f"{ROOT}/report_parts/configs.json"))


def overview():
    rows = []
    for c in CONFIGS:
        ex, meta = tables.load(c["lat_run"])
        b, total = tables.breakdown(ex, c["phase"], c["steps"], c.get("client_ms"))
        get = lambda pref: sum(v for k, v in b.items() if k.startswith(pref)) / 1e6
        rex, rmeta = tables.load(c["roof"][0][0])
        d = tables.roof_metrics(rex[rex.n_inst.notna() & rex.step.isin(c["roof"][0][1]) & (rex.role == "layer")], rmeta)
        w = d.dev_ns
        rows.append({
            "Config / phase": c["short"], "Total ms": total / 1e6,
            "Blocks ms": get("40 transformer"), "Drafter ms": get("DSpark drafter"),
            "Head+sampler ms": get("final norm"), "Embed ms": get("embedding"),
            "Device idle ms": get("device idle"), "Host outside ms": get("host time"),
            "ms / block": get("40 transformer") / (40 * (len(c["steps"]) if c["phase"] == "prefill" else 1)),
            "Block MFU %": 100 * (d.useful.sum() / (w.sum() / 1e9 * rmeta["te_peak_bf16_flops"])),
            "Block MBU %": 100 * (d.hbm.sum() / (w.sum() / 1e9 * rmeta["hbm_bw"])),
            "Block DMA active %": 100 * (d.act_dma.sum() / w.sum()),
            "Block CC %": 100 * (d.act_cc.sum() / w.sum()),
            "Block TensorE %": 100 * (d.act_tensor.sum() / w.sum()),
            "Block VectorE %": 100 * (d.act_vector.sum() / w.sum()),
            "Block GpSimd %": 100 * (d.act_gpsimd.sum() / w.sum()),
            "HBM GB / block": d.hbm.mean() / 1e9,
        })
    return pd.DataFrame(rows)


def cc_table(run, steps):
    ex = pd.read_parquet(f"{ROOT}/analysis/{run}/execs.parquet")
    ex = ex[ex.step.isin(steps) & ex.n_inst.notna()]
    dev = glob.glob(f"{ROOT}/data/{run}/profiles/global/*session_*@latest")[0]
    cc = pd.read_parquet(f"{dev}/CcOp.parquet")
    cc = cc[cc.operation != "Invalid"].sort_values("start_ts")
    cc["exec"] = pd.merge_asof(cc[["start_ts"]], ex[["t0", "exec"]].sort_values("t0"), left_on="start_ts",
                               right_on="t0")["exec"].values
    cc = cc.merge(ex[["exec", "role", "t1"]], on="exec")
    cc = cc[cc.start_ts < cc.t1]
    cc["kind"] = [kind(r) for r in cc.itertuples()]
    lay = cc[cc.role == "layer"]
    n = (ex.role == "layer").sum()
    t = lay.groupby("kind").agg(us=("duration_ns", "sum"), cnt=("duration_ns", "size"), mb=("input_size", "sum"),
                                cdtype=("dtype", "first"))
    t["per_block_us"] = t.us / n / 1e3
    t["mean_us"] = t.us / t.cnt / 1e3
    t["per_block_cnt"] = t.cnt / n
    t["MB"] = t.mb / n / 1e6
    return t, ex[ex.role == "layer"].dev_ns.mean() / 1e3


def fmt(df, floatfmt="{:.2f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def spill_table():
    rows = []
    for c in CONFIGS:
        run, steps = c["roof"][0]
        ex, m = tables.load(run)
        d = ex[ex.n_inst.notna() & ex.step.isin(steps)]
        for role, lab in (("layer", "blocks"), ("draft kernel", "drafter")):
            g = d[d.role == role]
            if len(g) == 0:
                continue
            hbm = (g.hbm_rd + g.hbm_wr).sum()
            sp = (g.spill_save + g.spill_reload).sum()
            rows.append({"Config / phase": c["short"], "Kernel": lab,
                         "HBM GB / launch": hbm / len(g) / 1e9, "of which scratch round-trips GB": sp / len(g) / 1e9,
                         "scratch %": 100 * sp / hbm,
                         "DMA: DGE hw %": 100 * g.dma_hardware_dynamic.sum() / g.dma_bytes.sum(),
                         "DMA: GpSimd sw-DGE %": 100 * g.dma_software_dynamic.sum() / g.dma_bytes.sum(),
                         "DMA: collectives %": 100 * g.dma_collectives.sum() / g.dma_bytes.sum()})
    return pd.DataFrame(rows)


def chunk_table():
    out = []
    for lab, a, b in (("in8192 bs1", ("in8192_bs1_prefill", [0]), ("in8192_bs1_prefill_c4", [0])),
                      ("in8192 bs10", ("in8192_bs10_prefill", [0]), ("in8192_bs10_prefill_c4", [0]))):
        cols = []
        for (run, steps), tag in ((a, "chunk 1"), (b, "chunk 4")):
            ex, m = tables.load(run)
            d = tables.roof_metrics(ex[ex.n_inst.notna() & ex.step.isin(steps)], m)
            d["kind"] = d.arch.fillna(d.role)
            g = d.groupby("kind").agg(ms=("dev_ns", lambda s: s.mean() / 1e6), mfu=("mfu", "mean"), mbu=("mbu", "mean"),
                                      hbm=("hbm", lambda s: s.mean() / 1e9))
            g.columns = [f"{tag} {c}" for c in ("ms", "MFU", "MBU", "HBM GB")]
            cols.append(g)
        t = pd.concat(cols, axis=1)
        t = t[t.index.isin(["r0", "r0_e", "r1", "r1_idx", "r1_src_idx", "r1_tap", "r2", "r2_src_idx", "r2_src_idx_e", "draft kernel", "head+sampler", "embed+hc_expand"])]
        for c in t.columns:
            if "MFU" in c or "MBU" in c:
                t[c] = 100 * t[c]
        t = t.sort_values("chunk 4 ms", ascending=False).reset_index().rename(columns={"kind": f"{lab}: block kind"})
        out.append(fmt(t.round(2)))
    return "\n\n".join(out)


def main():
    open(f"{ROOT}/report_parts/80_spill.md", "w").write(fmt(spill_table().round(2)) + "\n")
    open(f"{ROOT}/report_parts/80_chunks.md", "w").write(chunk_table() + "\n")
    o = overview()
    md = ["### Overview across configurations\n", fmt(o.round(2)), ""]
    md.append("\n### Collectives inside the transformer blocks (per block, rank 0)\n")
    md.append("| Config / phase | block µs | kernel-entry barrier µs | 8-rank all-reduce µs (count, MB, dtype) | 64-rank all-reduce µs (count, MB, dtype) | CC share of block |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for c in CONFIGS:
        run, steps = c["roof"][0]
        t, blk = cc_table(run, steps)
        def cell(k):
            if k not in t.index:
                return "–"
            r = t.loc[k]
            return f"{r.per_block_us:.0f} ({r.per_block_cnt:.1f}×, {r.MB:.1f} MB {r.cdtype})"
        bar = t.loc["barrier"].per_block_us if "barrier" in t.index else 0
        tot = t.per_block_us.sum()
        md.append(f"| {c['short']} | {blk:.0f} | {bar:.0f} | {cell('allreduce-8r')} | {cell('allreduce-64r')} | {100 * tot / blk:.0f} % |")
    open(f"{ROOT}/report_parts/80_tables.md", "w").write("\n".join(md) + "\n")
    o.to_csv(f"{ROOT}/analysis/overview.csv", index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
