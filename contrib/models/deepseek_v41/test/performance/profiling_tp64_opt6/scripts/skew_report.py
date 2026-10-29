#!/usr/bin/env python3
"""Cross-rank tables from skew6.py output -> report_parts/80_skew_<name>.md (+ printed)

  skew_report.py <out_name> <phase> [--steps a,b,..]

1. per collective kind: arrival spread over the captured ranks, rank 0's lower-bound wait, which rank
   arrives last, and rank 0's duration-based wait estimate (duration - fastest instance) for comparison
2. the MoE world all-reduce: why the last rank is late -- each block's span from its start (the previous
   block's world all-reduce end, a common instant on every rank) to its attention all-reduce trigger
   (attention), to its world all-reduce trigger (MoE), and the HBM bytes the rank read in the MoE span
   (its routed experts' weights: ~18.8 MB per expert that got >= 1 token)
3. the kernel-entry barrier: device start of the launch on each rank vs the host's submission
   (nrt_model_submit / kbl_exec_post in the rank's system trace)
"""
import argparse, json, re, sys
import duckdb, numpy as np, pandas as pd

ROOT = "/mnt/nvme/ds41_profiling_opt6"


def var_of(s):
    m = re.search(r"srcs=\[(\w+)\]", s) if isinstance(s, str) else None
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("phase")
    ap.add_argument("--steps", default=None)
    a = ap.parse_args()
    D = f"{ROOT}/analysis/{a.name}"
    S = pd.read_parquet(f"{D}/skew_cc.parquet")
    SU = pd.read_parquet(f"{D}/skew_units.parquet")
    info = json.load(open(f"{D}/skew_info.json"))
    steps = [int(s) for s in a.steps.split(",")] if a.steps else sorted(S.step.unique())[1:-1]
    S = S[S.step.isin(steps)].copy()
    S["var"] = S.src.map(var_of)
    from tl_summary import TENSOR
    ex0r = pd.read_parquet(f"{ROOT}/analysis/{info['runs'][str(info['ranks'][0])]}/execs.parquet").set_index("exec")
    S["role"] = S.exec0.map(ex0r.role)

    def _lab(r):
        if r.kind == "barrier":
            return "kernel-entry barrier"
        op = {"allreduce": "AR", "allgather": "AG"}.get(r.kind.split("-")[0], r.kind) + "-" + r.kind.split("-")[1][:-1]
        sz = f"{r.size / 1e6:.1f} MB" if r.size >= 1e6 else f"{r.size / 1e3:.0f} KB" if r.size >= 1e3 else f"{r.size} B"
        var = r.var if isinstance(r.var, str) else None
        if var is None:
            return f"{op} XLA-graph collective ({sz})"
        return f"{op} `{var}` ({TENSOR.get(var, var)}, {sz})" + (" — drafter" if r.role == "draft" else "")
    S["label"] = [_lab(r) for r in S.itertuples()]
    ranks = info["ranks"]
    r0 = ranks[0]
    k = len(steps) if a.phase == "decode" else 1
    out = []
    # ---- 1. per collective kind
    g = S.groupby(["exec0", "i"])
    S["spread"] = g.arrive.transform("max") - g.arrive.transform("min")
    s0 = S[S["rank"] == r0].copy()
    floor = s0.groupby("label").dur.min()
    s0["dur_wait"] = s0.dur - s0.label.map(floor)
    lastc = S[S.is_last].groupby("label")["rank"].agg(lambda x: x.value_counts().to_dict())
    t = s0.groupby("label").agg(n=("dur", "size"), dur_us=("dur", "mean"), floor_us=("dur", "min"),
                                wait_est_us=("dur_wait", "mean"), wait_lb_us=("wait_lb", "mean")).reset_index()
    sp = S.drop_duplicates(["exec0", "i"]).groupby("label").spread.agg(["mean", "median", lambda x: x.quantile(0.9)])
    sp.columns = ["spread_mean", "spread_p50", "spread_p90"]
    t = t.join(sp, on="label")
    for c in ("dur_us", "floor_us", "wait_est_us", "wait_lb_us", "spread_mean", "spread_p50", "spread_p90"):
        t[c] = t[c] / 1e3
    t["per_step_ms_wait_est"] = t.wait_est_us * t.n / k / 1e3
    t = t.sort_values("per_step_ms_wait_est", ascending=False)
    out.append(f"**Alignment**: ranks {ranks}; after aligning on the 64-rank all-reduce ends, those ends agree "
               f"within {info['end_spread_us_p50']:.1f} µs (median; p90 {info['end_spread_us_p90']:.1f} µs).\n")
    out.append("| Collective (rank 0) | per step | mean µs | fastest µs | rank 0 wait, duration-based µs | rank 0 wait vs last of 8 captured ranks µs | "
               "arrival spread over 8 ranks µs: mean / p50 / p90 | last to arrive (rank: times) |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in t.itertuples():
        lc = lastc.get(r.label, {})
        lcs = ", ".join(f"{kk}: {vv}" for kk, vv in sorted(lc.items(), key=lambda x: -x[1])[:4])
        out.append(f"| {r.label} | {r.n / k:.1f} | {r.dur_us:.1f} | {r.floor_us:.1f} | {r.wait_est_us:.1f} | {r.wait_lb_us:.1f} | "
                   f"{r.spread_mean:.1f} / {r.spread_p50:.1f} / {r.spread_p90:.1f} | {lcs} |")
    tab1 = "\n".join(out)

    # ---- 2. MoE world all-reduce: attention / MoE spans per rank and block
    w = S[(S["var"] == "cc2_src") & (S.gsize == 64)][["exec0", "i", "rank", "u", "arrive", "end", "wait_lb", "is_last"]]
    at = S[(S["var"] == "cc_src") & (S.gsize == 8)][["exec0", "i", "rank", "arrive", "end"]]
    rows = []
    for (e0, rk), ww in w.groupby(["exec0", "rank"]):
        aa = at[(at.exec0 == e0) & (at["rank"] == rk)].sort_values("i")
        ww = ww.sort_values("i")
        if len(aa) != len(ww):
            continue
        u = SU[(SU.exec0 == e0) & (SU["rank"] == rk)].sort_values("member")
        blocks = u[u.unit.str.match(r"L\d\d ")]
        if len(blocks) != len(ww):
            continue
        for (_, wr), (_, ar), (_, br) in zip(ww.iterrows(), aa.iterrows(), blocks.iterrows()):
            rows.append(dict(exec0=e0, rank=rk, block=br.unit, t_start=br.t0, attn_arrive=ar.arrive, attn_end=ar.end,
                             moe_arrive=wr.arrive, world_end=wr.end, wait_lb=wr.wait_lb, is_last=wr.is_last))
    B = pd.DataFrame(rows)
    tab2 = ""
    if len(B):
        B["attn_span"] = B.attn_arrive - B.t_start
        B["moe_span"] = B.moe_arrive - B.attn_end
        B["key"] = B.exec0.astype(str) + "|" + B.block
        # HBM bytes read in the MoE span, per rank (Instruction attribution)
        hb = []
        for rk in B["rank"].unique():
            run = info["runs"][str(rk)] if str(rk) in info["runs"] else info["runs"][rk]
            meta = json.load(open(f"{ROOT}/analysis/{run}/meta.json"))
            off = info["offsets_ns"][str(rk)]
            q = B[B["rank"] == rk][["key", "attn_end", "moe_arrive"]].copy()
            q["s"] = q.attn_end - off
            q["e"] = q.moe_arrive - off
            con = duckdb.connect()
            con.sql("set threads to 16")
            con.register("Q", q[["key", "s", "e"]])
            r = con.sql(f"""select Q.key, sum(coalesce(i.hbm_read_bytes,0)) rd from read_parquet('{meta['dev_dir']}/Instruction.parquet') i
                join Q on i.start_ts >= Q.s and i.start_ts < Q.e group by 1""").df()
            r["rank"] = rk
            hb.append(r)
        if hb:
            B = B.merge(pd.concat(hb), on=["key", "rank"], how="left")
        B["moe_rd_MB"] = B.get("rd", np.nan) / 1e6
        g = B.groupby("key")
        for c in ("attn_span", "moe_span", "moe_rd_MB"):
            B[f"{c}_rel"] = B[c] - g[c].transform("median")
        B["arr_rel"] = B.moe_arrive - g.moe_arrive.transform("median")
        last = B[B.is_last]
        corr_moe = np.corrcoef(B.arr_rel, B.moe_span_rel)[0, 1]
        corr_att = np.corrcoef(B.arr_rel, B.attn_span_rel)[0, 1]
        corr_rd = np.corrcoef(B.moe_span_rel.fillna(0), B.moe_rd_MB_rel.fillna(0))[0, 1] if B.moe_rd_MB.notna().any() else np.nan
        tab2 = "\n".join([
            "| Over all blocks and steps | mean | p50 | p90 |", "|---|---:|---:|---:|",
            f"| arrival spread at the MoE world all-reduce over the 8 ranks, µs | {(g.moe_arrive.max() - g.moe_arrive.min()).mean() / 1e3:.1f} | {(g.moe_arrive.max() - g.moe_arrive.min()).median() / 1e3:.1f} | {(g.moe_arrive.max() - g.moe_arrive.min()).quantile(0.9) / 1e3:.1f} |",
            f"| last rank's lateness vs the median rank, µs | {last.arr_rel.mean() / 1e3:.1f} | {last.arr_rel.median() / 1e3:.1f} | {last.arr_rel.quantile(0.9) / 1e3:.1f} |",
            f"|  … of which its MoE span (attention AR end → MoE AR trigger) is longer by, µs | {last.moe_span_rel.mean() / 1e3:.1f} | {last.moe_span_rel.median() / 1e3:.1f} | {last.moe_span_rel.quantile(0.9) / 1e3:.1f} |",
            f"|  … of which its attention span (block start → attention AR trigger) is longer by, µs | {last.attn_span_rel.mean() / 1e3:.1f} | {last.attn_span_rel.median() / 1e3:.1f} | {last.attn_span_rel.quantile(0.9) / 1e3:.1f} |",
            f"| last rank's extra HBM reads in its MoE span vs the median rank, MB | {last.moe_rd_MB_rel.mean():.1f} | {last.moe_rd_MB_rel.median():.1f} | {last.moe_rd_MB_rel.quantile(0.9):.1f} |",
            f"| MoE span per rank, µs (all ranks) | {B.moe_span.mean() / 1e3:.1f} | {B.moe_span.median() / 1e3:.1f} | {B.moe_span.quantile(0.9) / 1e3:.1f} |",
            f"| MoE-span HBM reads per rank, MB (all ranks) | {B.moe_rd_MB.mean():.1f} | {B.moe_rd_MB.median():.1f} | {B.moe_rd_MB.quantile(0.9):.1f} |",
            "",
            f"Correlation over all (block, rank) samples: arrival delay vs MoE span **{corr_moe:.2f}**, vs attention span "
            f"**{corr_att:.2f}**; MoE span vs MoE-span HBM reads **{corr_rd:.2f}**."])
        B.to_parquet(f"{D}/skew_blocks.parquet")

    # ---- 3. kernel-entry barrier vs host submission
    tab3 = ""
    bar = S[S.kind == "barrier"].copy()
    if len(bar):
        sub = []
        for rk in ranks:
            run = info["runs"][str(rk)] if str(rk) in info["runs"] else info["runs"][rk]
            meta = json.load(open(f"{ROOT}/analysis/{run}/meta.json"))
            ev = pd.read_parquet(f"{meta['sys_dir']}/SystemProfileEvents.parquet")
            ev = ev[ev.name.isin(["nrt_model_submit", "kbl_exec_pre", "nrta_execute_schedule"])][["name", "start_ts", "end_ts"]]
            off = info["offsets_ns"][str(rk)]
            ev["start_a"] = ev.start_ts + off
            ev["end_a"] = ev.end_ts + off
            ev["rank"] = rk
            sub.append(ev)
        E = pd.concat(sub)
        rows = []
        for r in bar.itertuples():
            e = E[(E["rank"] == r.rank) & (E.name == "nrt_model_submit") & (E.start_a < r.arrive)]
            if not len(e):
                continue
            x = e.loc[e.start_a.idxmax()]           # the latest submission before the launch started
            rows.append(dict(exec0=r.exec0, rank=r.rank, arrive=r.arrive, sub_start=x.start_a, sub_end=x.end_a,
                             wait_lb=r.wait_lb, is_last=r.is_last, end=r.end))
        Bb = pd.DataFrame(rows)
        if len(Bb):
            ex0 = pd.read_parquet(f"{ROOT}/analysis/{info['runs'][str(r0)]}/execs.parquet").set_index("exec")
            Bb["name"] = Bb.exec0.map(ex0.name)
            if a.phase == "prefill":
                lay0 = ex0.layer.reindex(Bb.exec0).values
                Bb["pos"] = np.where(Bb.name == "embedding + hc_expand", "E", np.where(lay0 == 0, "F0",
                                     np.where(Bb.name.str.startswith("draft"), "drafter", np.where(np.isnan(lay0), "other", "F1-F4"))))
            else:
                Bb["pos"] = np.where(Bb.name.str.contains("_embed_"), "F0", np.where(Bb.name.str.startswith("draft"), "drafter", "F1-F4"))
            g = Bb.groupby("exec0")
            Bb["arr_rel"] = Bb.arrive - g.arrive.transform("min")
            Bb["subend_rel"] = Bb.sub_end - g.sub_end.transform("min")
            Bb["sub_to_dev"] = Bb.arrive - Bb.sub_end
            lines = ["| Kernel-entry barrier, per launch | spread of the launch's device start over the 8 ranks, µs (mean / p90) | "
                     "rank 0's barrier, µs (mean) | correlation: a rank's device-start delay vs its host submission-end delay |",
                     "|---|---:|---:|---:|"]
            dur0 = S[(S.kind == "barrier") & (S["rank"] == r0)].set_index("exec0").dur
            labs = ((("E", "embedding graph (the iteration's first launch)"), ("F0", "first block launch (L00)"), ("F1-F4", "block launches L01–L39"),
                     ("drafter", "drafter launch")) if a.phase == "prefill" else
                    (("F0", "first fused launch of the step (`…_embed_L00_L07`)"), ("F1-F4", "fused launches 2–5"),
                     ("drafter", "drafter launch")))
            for pos, lab in labs:
                d = Bb[Bb.pos == pos]
                if not len(d):
                    continue
                sp = d.groupby("exec0").arrive.agg(lambda x: x.max() - x.min()) / 1e3
                c = np.corrcoef(d.arr_rel, d.subend_rel)[0, 1] if d.arr_rel.std() > 0 else np.nan
                lines.append(f"| {lab} | {sp.mean():.0f} / {sp.quantile(0.9):.0f} | {dur0.reindex(d.exec0.unique()).mean() / 1e3:.0f} | {c:.2f} |")
            # host time between steps, per rank: previous drafter end (common: it ends in collectives) -> F0 submitted
            f0 = Bb[Bb.pos == "F0"]
            rows = []
            for rk in ranks:
                run = info["runs"][str(rk)]
                exr = pd.read_parquet(f"{ROOT}/analysis/{run}/execs.parquet").sort_values("t0")
                off = info["offsets_ns"][str(rk)]
                dr = exr[exr.name.str.startswith("draft_")][["t1"]] + off
                for x in f0[f0["rank"] == rk].itertuples():
                    prev = dr[dr.t1 < x.arrive].t1
                    if len(prev):
                        rows.append(dict(rank=rk, exec0=x.exec0, host_ms=(x.sub_end - prev.max()) / 1e6))
            H = pd.DataFrame(rows)
            if len(H):
                hs = H.groupby("exec0").host_ms.agg(["min", "max", "mean"])
                lines += ["", "| Host time between steps (previous drafter end → the step's first launch submitted), ms | mean | p50 | p90 |",
                          "|---|---:|---:|---:|",
                          f"| fastest of the 8 ranks | {hs['min'].mean():.2f} | {hs['min'].median():.2f} | {hs['min'].quantile(0.9):.2f} |",
                          f"| slowest of the 8 ranks | {hs['max'].mean():.2f} | {hs['max'].median():.2f} | {hs['max'].quantile(0.9):.2f} |",
                          f"| rank 0 | {H[H['rank'] == r0].host_ms.mean():.2f} | {H[H['rank'] == r0].host_ms.median():.2f} | {H[H['rank'] == r0].host_ms.quantile(0.9):.2f} |",
                          "", "Slowest rank per step: " + ", ".join(f"{k}: {v}" for k, v in
                                                                 H.loc[H.groupby('exec0').host_ms.idxmax()]["rank"].value_counts().items())]
                H.to_parquet(f"{D}/skew_host.parquet")
            # host time inside the step: last fused launch end (common: it ends in collectives) -> drafter submitted
            dl = Bb[Bb.pos == "drafter"]
            rows = []
            for rk in ranks:
                run = info["runs"][str(rk)]
                exr = pd.read_parquet(f"{ROOT}/analysis/{run}/execs.parquet").sort_values("t0")
                off = info["offsets_ns"][str(rk)]
                hd = exr[exr.name.str.contains("_head_")][["t1"]] + off
                for x in dl[dl["rank"] == rk].itertuples():
                    prev = hd[hd.t1 < x.arrive].t1
                    if len(prev):
                        rows.append(dict(rank=rk, exec0=x.exec0, host_ms=(x.sub_end - prev.max()) / 1e6))
            H2 = pd.DataFrame(rows)
            hs = H2.groupby("exec0").host_ms.agg(["min", "max"]) if len(H2) else None
            if hs is not None and hs["min"].mean() > 0:     # (at bs1 the drafter is queued before the blocks end)
                lines += ["", "| Host time inside the step (last fused launch end → drafter launch submitted), ms | mean | p50 | p90 |",
                          "|---|---:|---:|---:|",
                          f"| fastest of the 8 ranks | {hs['min'].mean():.2f} | {hs['min'].median():.2f} | {hs['min'].quantile(0.9):.2f} |",
                          f"| slowest of the 8 ranks | {hs['max'].mean():.2f} | {hs['max'].median():.2f} | {hs['max'].quantile(0.9):.2f} |",
                          f"| rank 0 | {H2[H2['rank'] == r0].host_ms.mean():.2f} | {H2[H2['rank'] == r0].host_ms.median():.2f} | {H2[H2['rank'] == r0].host_ms.quantile(0.9):.2f} |"]
            tab3 = "\n".join(lines)
            Bb.to_parquet(f"{D}/skew_barrier.parquet")
    md = "\n\n".join(x for x in (tab1, tab2, tab3) if x)
    open(f"{ROOT}/report_parts/80_skew_{a.name}.md", "w").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
