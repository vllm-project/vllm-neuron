#!/usr/bin/env python3
"""Who waits for whom: collectives across the ranks of a multi-rank capture.

  skew6.py <out_name> <run_r0> <run_rA> ...  (runs = analyze6 outputs of the same window, one per rank)
     -> analysis/<out_name>/skew_cc.parquet, skew_units.parquet

Clocks: every rank's timestamps are relative to its own session origin
(SystemProfileMetadata.first_ts, wall clock); that aligns the ranks to ~0.1 ms. The offset
is then refined so that the END of the 64-rank all-reduces agrees across ranks (an
all-reduce completes on every rank within a few us of the others); the residual spread of
the end times is reported as the alignment error.

Per collective instance (matched by execution and order) and rank:
  arrive    the trigger time (the sequencer's doorbell write; the barrier's start)
  wait      last captured arrival - own arrival (a LOWER bound: 56 of the 64 ranks are not
            captured, the true last arrival can only be later)
Per block member and rank, the time from the block's start to its world all-reduce's
trigger and the HBM bytes read in that span: the work that decides who arrives last.
"""
import json, os, re, sys
import numpy as np, pandas as pd

ROOT = "/mnt/nvme/ds41_profiling_opt6"


def origin_ns(run):
    meta = json.load(open(f"{ROOT}/analysis/{run}/meta.json"))
    sm = pd.read_parquet(f"{meta['sys_dir']}/SystemProfileMetadata.parquet").iloc[0]
    ts = pd.Timestamp(sm.first_ts.replace(" +0000 UTC", ""))
    frac = re.search(r"\.(\d+)", sm.first_ts)
    ns = int(ts.value // 1_000_000_000 * 1_000_000_000) + (int(frac.group(1).ljust(9, "0")[:9]) if frac else 0)
    return ns - int(sm.first_ts_ns)


def rank_of(run):
    m = re.search(r"_r(\d+)$", run)
    return int(m.group(1))


def main(out, runs):
    os.makedirs(f"{ROOT}/analysis/{out}", exist_ok=True)
    R = {}
    for run in runs:
        ex = pd.read_parquet(f"{ROOT}/analysis/{run}/execs.parquet")
        U = pd.read_parquet(f"{ROOT}/analysis/{run}/units.parquet")
        cc = pd.read_parquet(f"{ROOT}/analysis/{run}/ccops.parquet")
        o = origin_ns(run)
        R[rank_of(run)] = dict(run=run, ex=ex, U=U, cc=cc, origin=o)
    ranks = sorted(R)
    r0 = ranks[0]
    base = R[r0]["origin"]
    # coarse offsets from the wall-clock origins
    for r in ranks:
        R[r]["off"] = R[r]["origin"] - base
    # match executions to rank r0's by kernel and nearest start
    ex0 = R[r0]["ex"]
    for r in ranks:
        ex = R[r]["ex"].copy()
        ex["ta"] = ex.t0 + R[r]["off"]
        R[r]["ex"] = ex
    ex0 = R[r0]["ex"]
    match = {r: {} for r in ranks}
    for r in ranks:
        exr = R[r]["ex"]
        for e in ex0.itertuples():
            c = exr[(exr.neff_name == e.neff_name) & ((exr.ta - e.ta).abs() < 15_000_000)]
            if len(c):
                match[r][e.exec] = int(c.iloc[(c.ta - e.ta).abs().argmin()].exec)
    # refine: align the ends of the 64-rank all-reduces
    for r in ranks:
        if r == r0:
            R[r]["fine"] = 0
            continue
        d = []
        c0 = R[r0]["cc"]
        cr = R[r]["cc"]
        for e0, er in match[r].items():
            a = c0[(c0.exec == e0) & (c0.kind != "barrier")].sort_values("start_ts")
            b = cr[(cr.exec == er) & (cr.kind != "barrier")].sort_values("start_ts")
            if len(a) != len(b) or len(a) == 0:
                continue
            ok = (a.kind.values == b.kind.values) & (a.gsize.values == 64)
            d.extend(((b.end_ts.values + R[r]["off"]) - a.end_ts.values)[ok])
        R[r]["fine"] = float(np.median(d)) if d else 0.0
        R[r]["off"] -= R[r]["fine"]
    rows = []
    for e0 in ex0.exec:
        if not all(e0 in match[r] for r in ranks):
            continue
        per = {}
        for r in ranks:
            cr = R[r]["cc"]
            per[r] = cr[cr.exec == match[r][e0]].sort_values("start_ts").reset_index(drop=True)
        n = {len(v) for v in per.values()}
        if len(n) != 1:
            continue
        for i in range(n.pop()):
            k = per[r0].kind[i]
            if any(per[r].kind[i] != k for r in ranks):
                break
            for r in ranks:
                x = per[r].iloc[i]
                arrive = (x.start_ts if k == "barrier" else x.trigger_start_ts) + R[r]["off"]
                rows.append(dict(exec0=e0, i=i, rank=r, kind=k, gsize=x.gsize, dtype=x["dtype"], size=x.input_size,
                                 src=x.src_text, u=x.u, arrive=arrive, end=x.end_ts + R[r]["off"],
                                 dur=x.duration_ns, step=ex0.set_index("exec").step[e0]))
    S = pd.DataFrame(rows)
    g = S.groupby(["exec0", "i"])
    S["last"] = g.arrive.transform("max")
    S["first"] = g.arrive.transform("min")
    S["wait_lb"] = S["last"] - S.arrive
    S["end_spread"] = g.end.transform("max") - g.end.transform("min")
    S["is_last"] = S.arrive == S["last"]
    # block context: the member's start and HBM reads up to the world all-reduce's trigger
    urows = []
    for r in ranks:
        U = R[r]["U"]
        for e0, er in match[r].items():
            for x in U[U.exec == er].itertuples():
                urows.append(dict(rank=r, exec0=e0, member=x.member, unit=x.unit, t0=x.t0 + R[r]["off"],
                                  t1=x.t1 + R[r]["off"], hbm_rd=x.hbm_rd, dma_bytes=x.dma_bytes,
                                  act_cc=x.act_cc, dur=x.dur_ns))
    SU = pd.DataFrame(urows)
    S.to_parquet(f"{ROOT}/analysis/{out}/skew_cc.parquet")
    SU.to_parquet(f"{ROOT}/analysis/{out}/skew_units.parquet")
    info = {"ranks": ranks, "runs": {r: R[r]["run"] for r in ranks},
            "fine_offset_ns": {r: R[r]["fine"] for r in ranks},
            "offsets_ns": {r: R[r]["off"] for r in ranks},
            "matched_execs": {r: len(match[r]) for r in ranks},
            "end_spread_us_p50": float(S[S.gsize == 64].end_spread.median() / 1e3) if len(S) else None,
            "end_spread_us_p90": float(S[S.gsize == 64].end_spread.quantile(0.9) / 1e3) if len(S) else None}
    json.dump(info, open(f"{ROOT}/analysis/{out}/skew_info.json", "w"), indent=1, default=str)
    print(json.dumps(info, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
