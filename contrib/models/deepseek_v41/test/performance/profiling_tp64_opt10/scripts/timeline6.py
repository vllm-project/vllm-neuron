#!/usr/bin/env python3
"""Utilisation timeline of one ingested capture: MFU / MBU / engine busy / collectives in
fixed time bins, and what the core is blocked on whenever MFU and MBU are far below peak.

  timeline6.py <run_name> [--bin-us 10]  ->  analysis/<run_name>/bins.parquet

Per bin (bins anchored at each execution's start; a fused launch's members own the bins
whose midpoint they contain):
  flops      useful TensorE FLOPs (BF16-equivalent, transposes excluded), each instruction's
             FLOPs spread uniformly over its duration
  dma_b      DMA bytes of the core's (non-collective, non-host) queues, each packet's bytes spread
             over its duration; `hbm_b` = dma_b x min(1, unit HBM bytes / unit DMA bytes)
             (Instruction.hbm_*_bytes): a bin never carries more HBM bytes than the DMA it moved
  busy_<e>   active ns of engine e, per physical core (ActiveTime) -> u_<e> = mean over the two
             cores, m_<e> = the busier core
  dma_act    ns with >= 1 packet in flight on the core's DMA queues (mean over the two cores)
  cc_*       the collective in flight longest in the bin (CcOp), and the split of its time into
             "waiting for peers" (its first dur - floor ns, floor = fastest instance of the same
             collective in the run) and "transfer" (the last floor ns)
  func       the NKI function with the most instruction time in the bin (any engine), and the
             busiest engine's top function
"""
import argparse, ast, glob, json, os, re, sys
import duckdb, numpy as np, pandas as pd

ROOT = "/mnt/nvme/ds41_profiling_opt10"
KROOT = "/home/ubuntu/vllm-neuron/contrib/models/deepseek_v41/src/kernels"
ENG = ["tensor", "vector", "scalar", "gpsimd", "sync"]


def func_map():
    m = {}
    for path in glob.glob(f"{KROOT}/**/*.py", recursive=True):
        try:
            tree = ast.parse(open(path).read())
        except SyntaxError:
            continue
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
        m[os.path.relpath(path, KROOT)] = sorted(spans, key=lambda s: s[1] - s[0])
    return m


FM = func_map()


def resolve(loc):
    """NKI source location -> 'file.py:function' (file without the archetype package)."""
    if not isinstance(loc, str):
        return "(no source location)"
    mm = re.search(r"kernels/(.+\.py):(\d+)", loc)
    if not mm:
        mm2 = re.search(r"([^/]+\.py):(\d+)", loc)
        return f"{mm2.group(1)}:?" if mm2 else loc
    rel, ln = mm.group(1), int(mm.group(2))
    spans = FM.get(rel) or FM.get(f"tp64/{rel}") or []
    base = os.path.basename(rel)
    for a, b, name in spans:
        if a <= ln <= b:
            return f"{base}:{name}"
    return f"{base}:<module>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--bin-us", type=float, default=10.0)
    a = ap.parse_args()
    run, B = a.run, int(a.bin_us * 1000)
    out = f"{ROOT}/analysis/{run}"
    meta = json.load(open(f"{out}/meta.json"))
    dev = meta["dev_dir"]
    U = pd.read_parquet(f"{out}/units.parquet")
    ex = pd.read_parquet(f"{out}/execs.parquet")
    cc = pd.read_parquet(f"{out}/ccops.parquet")
    X = ex[ex.n_inst.notna() & ~ex.partial][["exec", "t0", "t1"]].copy()
    con = duckdb.connect()
    con.sql("set threads to 24")
    con.sql("set memory_limit='200GB'")
    con.register("X", X)
    I = f"read_parquet('{dev}/Instruction.parquet')"
    AT = f"read_parquet('{dev}/ActiveTime.parquet')"
    D = f"read_parquet('{dev}/DmaPacket.parquet')"

    def explode(src, cols):
        """intervals (exec, s, e, cols...) -> (exec, b, ov, cols...) pieces per bin"""
        return f"""select exec, b, {cols},
              least(e, t0 + (b + 1) * {B}) - greatest(s, t0 + b * {B}) ov, e - s len
            from (select q.*, X.t0, unnest(range((greatest(q.s, X.t0) - X.t0) // {B},
                                                (least(q.e, X.t1) - 1 - X.t0) // {B} + 1)) b
                  from ({src}) q join X using (exec) where q.e > q.s)"""

    # ---- TensorE FLOPs
    fl = con.sql(f"""select exec, b, sum(f * ov / len) flops, sum(ftr * ov / len) flops_tr from ({explode(
        f'''select execution_index exec, start_ts s, greatest(end_ts, start_ts + 1) e,
               case when tensor_instruction_type = 'TRANSPOSE' then 0 else coalesce(adjusted_flops, 0) end f,
               case when tensor_instruction_type = 'TRANSPOSE' then coalesce(adjusted_flops, 0) else 0 end ftr
            from {I} where engine = 'Tensor' and execution_unit = 'compute' and coalesce(adjusted_flops, 0) > 0''',
        "f, ftr")}) group by all""").df()
    # ---- engine busy per pcore (ActiveTime is merged within a core)
    at = con.sql(f"""select exec, b, engine, pcore_idx, sum(ov) busy from ({explode(
        f'''select x.exec, a.start_ts s, a.end_ts e, a.engine, a.pcore_idx from {AT} a
            asof join X x on a.start_ts >= x.t0 where a.start_ts < x.t1''', "engine, pcore_idx")}) group by all""").df()
    # ---- DMA bytes and active-wall per pcore (non-collective, non-host queues)
    con.sql(f"""create temp table dp as select x.exec, d.start_ts s, greatest(d.end_ts, d.start_ts + 1) e,
        d.pcore_idx, d.transfer_bytes nb, coalesce(d.queue_type, '') = 'collectives' iscc
        from {D} d asof join X x on d.start_ts >= x.t0 where d.start_ts < x.t1 and not d.is_host_device_dma""")
    db = con.sql(f"""select exec, b, sum(case when not iscc then nb * ov / len else 0 end) dma_b,
        sum(case when iscc then nb * ov / len else 0 end) ccq_b from ({explode('select * from dp', 'nb, iscc')})
        group by all""").df()
    dm = con.sql(f"""
      with q as (select exec, pcore_idx, s, e, max(e) over (partition by exec, pcore_idx order by s, e
                 rows between unbounded preceding and 1 preceding) pm from dp where not iscc),
      r as (select *, sum(case when pm is null or s > pm then 1 else 0 end)
                 over (partition by exec, pcore_idx order by s, e rows unbounded preceding) g from q),
      m as (select exec, pcore_idx, min(s) s, max(e) e from r group by exec, pcore_idx, g)
      select exec, b, pcore_idx, sum(ov) dact from ({explode('select * from m', 'pcore_idx')}) group by all""").df()
    # ---- instruction time per function (any engine) and per engine
    locs = con.sql(f"select distinct nki_source_location loc from {I} where execution_unit = 'compute'").df()
    locs["func"] = locs["loc"].map(resolve)
    con.register("locs", locs)
    fn = con.sql(f"""select exec, b, engine, func, sum(ov) t from ({explode(
        f'''select i.execution_index exec, i.start_ts s, greatest(i.end_ts, i.start_ts + 1) e, i.engine, l.func
            from {I} i join locs l on i.nki_source_location is not distinct from l.loc
            where i.execution_unit = 'compute' ''', "engine, func")}) group by all""").df()

    # ---- assemble the bin table
    bins = []
    for r in X.itertuples():
        nb = int((r.t1 - r.t0 + B - 1) // B)
        b = np.arange(nb)
        bins.append(pd.DataFrame({"exec": r.exec, "b": b, "bs": r.t0 + b * B,
                                  "be": np.minimum(r.t0 + (b + 1) * B, r.t1)}))
    T = pd.concat(bins, ignore_index=True)
    T["blen"] = T.be - T.bs
    mid = (T.bs + T.be) // 2
    # unit of each bin (units tile their execution)
    Us = U[U.exec.isin(X.exec)].sort_values("t0")
    T = T.sort_values("bs")
    T["mid"] = mid.loc[T.index]
    T = pd.merge_asof(T.sort_values("mid"), Us[["t0", "u"]].rename(columns={"t0": "ut0"}), left_on="mid",
                      right_on="ut0", direction="backward")
    T = T.drop(columns=["ut0"])
    key = ["exec", "b"]
    T = T.merge(fl, on=key, how="left").merge(db, on=key, how="left")
    a2 = at.assign(engine=at.engine.str.lower()).pivot_table(index=key, columns=["engine", "pcore_idx"],
                                                             values="busy", aggfunc="sum").fillna(0)
    for e in ENG:
        cols = [c for c in a2.columns if c[0] == e]
        if cols:
            v = a2[cols]
            T = T.merge(pd.DataFrame({f"u_{e}": v.mean(axis=1) if len(cols) == 2 else v.sum(axis=1) / 2,
                                      f"m_{e}": v.max(axis=1)}).reset_index(), on=key, how="left")
    d2 = dm.pivot_table(index=key, columns="pcore_idx", values="dact", aggfunc="sum").fillna(0)
    T = T.merge(pd.DataFrame({"dma_act": d2.sum(axis=1) / 2, "dma_act_max": d2.max(axis=1)}).reset_index(),
                on=key, how="left")
    T = T.fillna(0)
    for e in ENG + ["dma"]:
        c = f"u_{e}" if e != "dma" else "dma_act"
        if c in T:
            T[c] = T[c] / T.blen
    for e in ENG:
        if f"m_{e}" in T:
            T[f"m_{e}"] = T[f"m_{e}"] / T.blen
    T["dma_act_max"] = T.dma_act_max / T.blen

    # HBM bytes: the unit's DMA bytes scaled to its HBM bytes (Instruction attribution)
    ub = T.groupby("u").dma_b.sum()
    Uu = U.set_index("u")
    ratio = ((Uu.hbm_rd + Uu.hbm_wr) / ub.reindex(Uu.index)).replace([np.inf], np.nan)
    T["hbm_ratio"] = T.u.map(ratio).fillna(0).clip(upper=1.0)   # (never more HBM bytes than DMA bytes in a bin)
    T["hbm_b"] = T.dma_b * T.hbm_ratio
    te, bw = meta["te_peak_bf16_flops"], meta["hbm_bw"]
    T["mfu"] = T.flops / (T.blen / 1e9 * te)
    T["mbu"] = T.hbm_b / (T.blen / 1e9 * bw)

    # ---- collectives: the op in flight longest in each bin, wait vs transfer split
    c = cc[cc.exec.isin(X.exec)].copy()
    c["sig"] = c.kind + "|" + c.dtype.astype(str) + "|" + c.input_size.astype(str) + "|" + c.src.fillna("")
    floor = c[c.kind != "barrier"].groupby("sig").duration_ns.min()
    c["floor"] = c.sig.map(floor).fillna(0)
    c.loc[c.kind == "barrier", "floor"] = 0
    c["wait_end"] = c.start_ts + (c.duration_ns - c.floor)
    c = c.reset_index(drop=True)
    c["cid"] = np.arange(len(c))          # (CcOp ids are 64-bit: no float round trip)
    c.to_parquet(f"{out}/cc_{int(a.bin_us)}us.parquet")
    pieces = []
    t0 = X.set_index("exec").t0
    for r in c.itertuples():
        e0 = t0[r.exec]
        for b in range(int((r.start_ts - e0) // B), int((r.end_ts - 1 - e0) // B) + 1):
            bs, be = e0 + b * B, e0 + (b + 1) * B
            ov = min(r.end_ts, be) - max(r.start_ts, bs)
            if ov <= 0:
                continue
            ovw = max(0, min(r.wait_end, be) - max(r.start_ts, bs))
            pieces.append((r.exec, b, r.cid, ov, ovw))
    P = pd.DataFrame(pieces, columns=["exec", "b", "cc_id", "cc_ov", "cc_wait"])
    if len(P):
        cov = P.groupby(key)[["cc_ov"]].sum().rename(columns={"cc_ov": "cc_any"})  # (overlapping ops may double count)
        top = P.sort_values("cc_ov").groupby(key).tail(1).set_index(key)
        T = T.merge(cov.reset_index(), on=key, how="left").merge(top.reset_index(), on=key, how="left")
    else:
        T["cc_any"], T["cc_id"], T["cc_ov"], T["cc_wait"] = 0, np.nan, 0, 0
    T[["cc_any", "cc_ov", "cc_wait"]] = T[["cc_any", "cc_ov", "cc_wait"]].fillna(0)
    T["cc_id"] = T.cc_id.fillna(-1).astype("int64")
    T["cc_frac"] = (T.cc_any / T.blen).clip(upper=1)
    T["cc_wait_frac"] = (T.cc_wait / T.blen).clip(upper=1)

    # ---- dominant functions
    fa = fn.groupby(key + ["func"]).t.sum().reset_index().sort_values("t").groupby(key).tail(1)
    T = T.merge(fa.rename(columns={"func": "func_any", "t": "func_any_t"}), on=key, how="left")
    fe = fn.assign(engine=fn.engine.str.lower()).sort_values("t").groupby(key + ["engine"]).tail(1)
    fe = fe.pivot_table(index=key, columns="engine", values="func", aggfunc="first")
    fe.columns = [f"func_{c}" for c in fe.columns]
    T = T.merge(fe.reset_index(), on=key, how="left")
    T = T.merge(Us[["u", "unit", "role", "kernel", "member", "layer", "arch", "step"]], on="u", how="left")
    T.to_parquet(f"{out}/bins_{int(a.bin_us)}us.parquet")
    print(f"{run}: {len(T)} bins of {a.bin_us} us; peak MFU {100 * T.mfu.max():.1f} % / MBU {100 * T.mbu.max():.1f} %")


if __name__ == "__main__":
    main()
