#!/usr/bin/env python3
"""Cross-configuration tables for the opt10 report -> report_parts/80_*.md"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import tables6

ROOT = "/mnt/nvme/ds41_profiling_opt10"
CONFIGS = json.load(open(f"{ROOT}/report_parts/configs.json"))


def fmt(df, floatfmt="{:.2f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                     for v in r) + " |")
    return "\n".join(out)


def roof_blocks(c):
    rs = []
    for run, steps in c["roof"]:
        U, ex, meta = tables6.load(run)
        d = U[U.n_inst.notna() & U.step.isin(steps) & U.layer.notna() & (U.dur_ns > 0)]
        rs.append(tables6.roof_metrics(d, meta))
    return pd.concat(rs), meta


def overview():
    rows = []
    for c in CONFIGS:
        U, ex, meta = tables6.load(c["lat_run"])
        b, total = tables6.breakdown(U, ex, c["phase"], c["steps"], c.get("client_ms"))
        get = lambda pref: sum(v for k, v in b.items() if k.startswith(pref)) / 1e6
        d, rmeta = roof_blocks(c)
        w = d.dur_ns
        nb = 40 * (len(c["steps"]) if c["phase"] == "prefill" else 1)
        rows.append({
            "Config / phase": c["short"], "Total ms": total / 1e6,
            "Blocks ms": get("40 transformer"), "Drafter ms": get("DSpark drafter"),
            "Entry barriers ms": get("kernel-entry"), "Head ms": get("final norm"), "Embed ms": get("embedding"),
            "Device idle ms": get("device idle"), "Host outside ms": get("host time"),
            "ms / block": get("40 transformer") / nb,
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
    U = pd.read_parquet(f"{ROOT}/analysis/{run}/units.parquet")
    cc = pd.read_parquet(f"{ROOT}/analysis/{run}/ccops.parquet")
    lay = U[U.step.isin(steps) & U.layer.notna() & U.n_inst.notna()]
    c = cc[cc.u.isin(lay.u)].copy()
    import re
    c["var"] = [(re.search(r"srcs=\[(\w+)\]", s).group(1) if isinstance(s, str) and "srcs=[" in s else k)
                for s, k in zip(c.src_text, c.kind)]
    n = len(lay)
    t = c.groupby(["kind", "var"]).agg(us=("duration_ns", "sum"), cnt=("duration_ns", "size"), b=("input_size", "mean"),
                                       cdtype=("dtype", "first")).reset_index()
    t["per_block_us"] = t.us / n / 1e3
    t["per_block_cnt"] = t.cnt / n
    t["MB"] = t.b / 1e6
    return t, lay.dur_ns.mean() / 1e3, lay.act_cc.sum() / lay.dur_ns.sum()


def spill_table():
    rows = []
    for c in CONFIGS:
        rs = []
        for run, steps in c["roof"]:
            U, ex, m = tables6.load(run)
            rs.append(U[U.n_inst.notna() & U.step.isin(steps)])
        d = pd.concat(rs)
        for sel, lab in ((d.layer.notna(), "blocks"), (d.role == "draft", "drafter")):
            g = d[sel]
            if len(g) == 0:
                continue
            hbm = (g.hbm_rd + g.hbm_wr).sum()
            sp = (g.spill_save + g.spill_reload).sum()
            col = lambda q: g[q].sum() if q in g else 0.0
            rows.append({"Config / phase": c["short"], "Kernel": lab,
                         "HBM GB / launch": hbm / len(g) / 1e9, "of which scratch round-trips GB": sp / len(g) / 1e9,
                         "scratch %": 100 * sp / hbm,
                         "DMA: DGE hw %": 100 * col("dma_hardware_dynamic") / g.dma_bytes.sum(),
                         "DMA: GpSimd sw-DGE %": 100 * col("dma_software_dynamic") / g.dma_bytes.sum(),
                         "DMA: collectives %": 100 * col("dma_collectives") / (g.dma_bytes.sum() + col("dma_collectives"))})
    return pd.DataFrame(rows)


def chunk_table():
    """in8192: per block kind, latency of each chunk (live trace) and roofline of chunk 1 vs chunk 3 (device trace)"""
    out = []
    for lab, run in (("in8192 bs1", "in8192_bs1_prefill"), ("in8192 bs10", "in8192_bs10_prefill")):
        U, ex, m = tables6.load(run)
        U["kind"] = U.arch.where(U.arch.notna(), U.block)
        lat = U[U.step.isin([0, 1, 2, 3])].groupby(["kind", "step"]).lat_ns.mean().unstack() / 1e6
        lat.columns = [f"chunk {int(c) + 1} ms" for c in lat.columns]
        cols = [lat]
        for st in (0, 2):
            d = tables6.roof_metrics(U[U.n_inst.notna() & (U.step == st) & (U.dur_ns > 0)], m)
            g = d.groupby("kind").agg(mfu=("mfu", "mean"), mbu=("mbu", "mean"), hbm=("hbm", lambda x: x.mean() / 1e9))
            g.columns = [f"chunk {st + 1} {c}" for c in ("MFU %", "MBU %", "HBM GB")]
            cols.append(g)
        t = pd.concat(cols, axis=1)
        for c in t.columns:
            if "%" in c:
                t[c] = 100 * t[c]
        t = t[t["chunk 4 ms"].notna() & (t["chunk 4 ms"] > 0.05)]
        t = t.sort_values("chunk 4 ms", ascending=False).reset_index().rename(columns={"kind": f"{lab}: block kind"})
        out.append(fmt(t.round(2)))
    return "\n\n".join(out)


def summary_table():
    from tl_summary import load as tl_load, summarize
    rows = ["| Config · phase | Total | Transformer blocks | Drafter | Entry barriers | Device idle (+ host outside, prefill) | Rest | Per block | "
            "Block MFU / MBU (mean) | Peak MFU / MBU (10 µs) | Under-utilised device time |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in CONFIGS:
        U, ex, meta = tables6.load(c["lat_run"])
        b, total = tables6.breakdown(U, ex, c["phase"], c["steps"], c.get("client_ms"))
        get = lambda pref: sum(v for k, v in b.items() if k.startswith(pref))
        blk, dr, bar = get("40 transformer"), get("DSpark drafter"), get("kernel-entry")
        idle = get("device idle") + get("host time")
        rest = total - blk - dr - bar - idle
        d, rmeta = roof_blocks(c)
        w = d.dur_ns
        mfu = 100 * d.useful.sum() / (w.sum() / 1e9 * rmeta["te_peak_bf16_flops"])
        mbu = 100 * d.hbm.sum() / (w.sum() / 1e9 * rmeta["hbm_bw"])
        run, steps = c["roof"][0]
        T, cc, m = tl_load(run, steps)
        T["all"] = "all"
        R2, C2 = summarize(T, "all", m)
        r = R2.iloc[0]
        nb = 40 * (len(c["steps"]) if c["phase"] == "prefill" else 1)
        tot = f"{total / 1e9:.2f} s" if total > 1e9 else f"{total / 1e6:.1f} ms"
        lab = c["short"].replace(" prefill", " · prefill (TTFT)" if "bs1 " in c["short"] + " " else " · prefill (all requests)")
        lab = lab.replace(" decode", " · decode (per step)")
        if c["phase"] == "prefill":
            RU, _, _ = tables6.load(run)
            ru = RU[RU.step.isin(steps) & RU.n_inst.notna()]
            barc = f"({100 * ru.act_barrier.sum() / ru.dur_ns.sum():.1f} % inside the launches)"
        else:
            barc = f"{100 * bar / total:.1f} %"
        rows.append(f"| {lab} | {tot} | {100 * blk / total:.1f} % | {100 * dr / total:.1f} % | {barc} | "
                    f"{100 * idle / total:.1f} % | {100 * rest / total:.1f} % | {blk / nb / 1e6:.2f} ms | {mfu:.1f} % / {mbu:.1f} % | "
                    f"{100 * r.peak_mfu:.0f} % / {100 * r.peak_mbu:.0f} % | {r.under_pct:.0f} % |")
    return "\n".join(rows)


def cause_matrix():
    """% of the step (decode: incl. device idle) / of the profiled device time (prefill) per blocking cause"""
    from tl_summary import load as tl_load, summarize
    import util6
    cats = [("device idle: no launch on the core", None)] + [(c, cs) for c, cs in util6.CAT]
    names = {"CC wait": "collective: waiting for peer ranks", "entry barrier": "kernel-entry barrier (all ranks must launch)",
             "CC transfer": "collective: transfer", "TensorE busy": "TensorE busy at low efficiency",
             "VectorE busy": "VectorE busy", "ScalarE/GpSimd/SP busy": "ScalarE / GpSimd / SP busy",
             "DMA queues": "DMA queues busy, few bytes", "dep. stall": "dependency stall (no engine >= 70 % busy)"}
    cols, heads = [], []
    for c in CONFIGS:
        run, steps = c["roof"][0]
        T, cc, m = tl_load(run, steps)
        T["all"] = "all"
        R2, C2 = summarize(T, "all", m)
        k = len(steps) if c["phase"] == "decode" else 1
        dev = R2.ms.iloc[0] / k
        idle = 0.0
        if c["phase"] == "decode":
            U, ex, _ = tables6.load(c["lat_run"])
            b, total = tables6.breakdown(U, ex, "decode", c["steps"])
            idle = sum(v for kk, v in b.items() if kk.startswith("device idle")) / 1e6
        tot = dev + idle
        col = {"device idle: no launch on the core": 100 * idle / tot}
        for cn, cs in util6.CAT:
            col[cn] = 100 * C2[C2.cause.isin(cs)].ms.sum() / k / tot
        col["not under-utilised (MFU or MBU >= 1/2 of peak)"] = 100 - sum(col.values())
        cols.append(col)
        heads.append(c["short"] + (" (per step)" if c["phase"] == "decode" else " (profiled iteration)"))
    rows = ["| Where the time goes | " + " | ".join(heads) + " |", "|---|" + "---:|" * len(heads)]
    keys = list(cols[0].keys())
    for kk in keys:
        rows.append(f"| {names.get(kk, kk)} | " + " | ".join(f"{c[kk]:.1f} %" for c in cols) + " |")
    return "\n".join(rows)


def peak_matrix():
    """per megakernel and config: peak MFU / peak MBU (best 10 us bin), mean in parentheses"""
    from tl_summary import load as tl_load, summarize
    import util6
    tabs = {}
    for phase in ("prefill", "decode"):
        cols, heads = {}, []
        for c in [c for c in CONFIGS if c["phase"] == phase]:
            run, steps = c["roof"][0]
            T, cc, m = tl_load(run, steps)
            T["g"] = util6.mk_group(T, "launch")
            R, C = summarize(T, "g", m)
            T["all"] = "**all device time**"
            R2, _ = summarize(T, "all", m)
            R = pd.concat([R, R2])
            heads.append(c["short"].replace(f" {phase}", ""))
            for r in R.itertuples():
                if r.group in ("glue", "drafter meta", "rejection sampler", "gather/concat graph", "decode tail graph", "token history put"):
                    continue
                cols.setdefault(r.group, {})[heads[-1]] = (f"{100 * r.peak_mfu:.0f} / {100 * r.peak_mbu:.0f} "
                                                           f"({100 * r.mean_mfu:.0f} / {100 * r.mean_mbu:.0f})")
        import re as _re

        def _key(g):
            m = _re.search(r"L(\d\d)", str(g))
            return (g.startswith("**"), not str(g).startswith(("layer", "layers")), int(m.group(1)) if m else 99, str(g))
        order = sorted(cols, key=_key)
        rows = [f"| {phase} megakernel | " + " | ".join(heads) + " |", "|---|" + "---:|" * len(heads)]
        for g in order:
            name = g if g.startswith("**") else f"`{g}`"
            rows.append(f"| {name} | " + " | ".join(cols[g].get(h, "–") for h in heads) + " |")
        tabs[phase] = "\n".join(rows)
    return tabs["prefill"] + "\n\n" + tabs["decode"]


OPT6 = "/home/ubuntu/vllm-neuron/contrib/models/deepseek_v41/test/performance/profiling_tp64_opt6/report_data.json"


def vs_opt6():
    """opt6 -> opt10, same configurations and windows (opt6 numbers from its report_data.json)"""
    o6 = json.load(open(OPT6))
    grp = [("Total", None), ("Blocks", "40 transformer"), ("Drafter", "DSpark drafter"), ("Entry barriers", "kernel-entry"),
           ("Device idle", "device idle"), ("Host outside", "host time")]
    rows = ["| Config · phase | " + " | ".join(f"{g} ms: opt6 → opt10" for g, _ in grp) + " |",
            "|---|" + "---:|" * len(grp)]
    arch = {}
    for c in CONFIGS:
        U, ex, meta = tables6.load(c["lat_run"])
        b, total = tables6.breakdown(U, ex, c["phase"], c["steps"], c.get("client_ms"))
        new = {k: v / 1e6 for k, v in b.items()}
        old = o6[c["id"]]["breakdown_ms"]
        cells = []
        for g, pref in grp:
            if pref is None:
                a, n = o6[c["id"]]["total_ms"], total / 1e6
            else:
                a = sum(v for k, v in old.items() if k.startswith(pref))
                n = sum(v for k, v in new.items() if k.startswith(pref))
            f = (lambda x: f"{x / 1e3:,.2f} s") if a > 2000 else (lambda x: f"{x:,.1f}" if x >= 10 else f"{x:.2f}")
            cells.append(f"{f(a)} → **{f(n)}**" if a or n else "–")
        rows.append(f"| {c['short']} | " + " | ".join(cells) + " |")
        a6 = o6[c["id"]]["arch"]
        a10 = tables6.per_arch(U, c["phase"], c["steps"])
        for k, r in a10.iterrows():
            if k in a6:
                arch.setdefault(k, {})[c["short"]] = f"{a6[k]['per_ms']:.2f} → **{r.per / 1e6:.2f}**"
    heads = [c["short"] for c in CONFIGS]
    order = ["r0", "r0_e", "r2", "r2_src_idx", "r2_src_idx_e", "r1_src_idx", "r1_idx", "r1", "r1_tap"]
    rows2 = ["| ms per block, opt6 → opt10 | " + " | ".join(heads) + " |", "|---|" + "---:|" * len(heads)]
    for k in order:
        if k in arch:
            rows2.append(f"| `{k}` | " + " | ".join(arch[k].get(h, "–") for h in heads) + " |")
    return "\n".join(rows) + "\n\n" + "\n".join(rows2)


def main():
    os.makedirs(f"{ROOT}/report_parts", exist_ok=True)
    o = overview()
    o.to_csv(f"{ROOT}/analysis/overview.csv", index=False)
    md = ["### Overview across configurations\n", fmt(o.round(2)), ""]
    md.append("\n### Collectives inside the transformer blocks (per block, rank 0)\n")
    md.append("| Config / phase | block µs | attention AR (8 ranks) µs: count, MB | MoE AR (64 ranks) µs: count, MB | "
              "indexer-scores AR (8 ranks) µs | engram AR µs | CC share of block |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for c in CONFIGS:
        run, steps = c["roof"][0]
        t, blk, share = cc_table(run, steps)

        def cell(var, kind=None):
            s = t[(t["var"] == var) & ((t.kind == kind) if kind else True)]
            if not len(s):
                return "–"
            r = s.iloc[0]
            return f"{r.per_block_us:.0f} ({r.per_block_cnt:.2f}×, {r.MB:.2f} MB {r.cdtype})"
        md.append(f"| {c['short']} | {blk:.0f} | {cell('cc_src')} | {cell('cc2_src')} | {cell('sc_src')} | {cell('emb')} | "
                  f"{100 * share:.0f} % |")
    open(f"{ROOT}/report_parts/80_tables.md", "w").write("\n".join(md) + "\n")
    open(f"{ROOT}/report_parts/80_spill.md", "w").write(fmt(spill_table().round(2)) + "\n")
    open(f"{ROOT}/report_parts/80_chunks.md", "w").write(chunk_table() + "\n")
    open(f"{ROOT}/report_parts/05_summary_table.md", "w").write(summary_table() + "\n")
    open(f"{ROOT}/report_parts/06_cause_matrix.md", "w").write(cause_matrix() + "\n")
    open(f"{ROOT}/report_parts/07_peak_matrix.md", "w").write(peak_matrix() + "\n")
    open(f"{ROOT}/report_parts/80_vs_opt6.md", "w").write(vs_opt6() + "\n")
    print("\n".join(md))
    print(open(f"{ROOT}/report_parts/80_spill.md").read())


if __name__ == "__main__":
    main()
