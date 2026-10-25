#!/usr/bin/env python3
"""Per-execution device metrics for one ingested capture (rank 0).

  analyze.py <run_name>  ->  /mnt/nvme/ds41_profiling/analysis/<run_name>/execs.parquet + meta.json

One row per NEFF execution in the capture window:
  latency    device time (ExecutionInfo; cross-checked against the live nc_exec_running system
             trace) and the device-idle gap before it (host dispatch / sync)
  engines    active-wall per engine, interval-merged over both physical cores of the LNC=2 core
  DMA        active-wall of all NeuronCore DMA packets (merged), bytes by queue type
  HBM        bytes read/written from HBM by DMA-issuing instructions (Instruction.hbm_*_bytes)
  TensorE    raw and BF16-equivalent (adjusted) FLOPs, transposes separately
  collectives CcOp active-wall (merged)
plus a label: kernel name from the NEFF, role, layer id, archetype, step index.
"""
import glob, io, json, os, re, sys, tarfile
import duckdb, numpy as np, pandas as pd

ROOT = "/mnt/nvme/ds41_profiling"
CFG = json.load(open("/mnt/nvme/hf/config.json"))["text_config"]


def arch_of(layer):
    parts = [f"r{CFG['compress_ratios'][layer]}"]
    if layer in CFG["kv_source_layer_ids"]:
        parts.append("src")
    if layer in CFG["index_source_layer_ids"]:
        parts.append("idx")
    if layer in CFG["engram_layer_ids"]:
        parts.append("e")
    if layer in CFG["dspark_target_layer_ids"]:
        parts.append("tap")
    return "_".join(parts)


_NAME_RE = re.compile(rb"(layer_(?:prefill|decode)_[a-z0-9_]+?_tp64|draft_(?:prefill|decode)_tp64|hc_expand|_sampler)")


def neff_kernel(path):
    b = open(path, "rb").read()
    try:
        t = tarfile.open(fileobj=io.BytesIO(b[1024:]))
    except Exception:
        return None
    for m in t.getmembers():
        if m.isfile() and m.size < 50_000_000:
            hit = _NAME_RE.search(t.extractfile(m).read())
            if hit:
                return hit.group(1).decode()
    return None


def merged_wall(con, table_sql, key="exec"):
    """Sum of merged [s, e) intervals per key; table_sql yields (key, s, e)."""
    return con.sql(f"""
      with p as ({table_sql}),
      q as (select {key}, s, e, max(e) over (partition by {key} order by s, e
                 rows between unbounded preceding and 1 preceding) pm from p),
      r as (select {key}, s, e, sum(case when pm is null or s > pm then 1 else 0 end)
                 over (partition by {key} order by s, e rows unbounded preceding) g from q)
      select {key}, sum(ge - gs) wall from (select {key}, g, min(s) gs, max(e) ge from r group by all)
      group by {key}""").df().set_index(key)["wall"]


def main(run):
    G = f"{ROOT}/data/{run}/profiles/global"
    dev = glob.glob(f"{G}/*session_*@latest")[0]
    sysd = f"{G}/{run}@latest"
    out = f"{ROOT}/analysis/{run}"
    os.makedirs(out, exist_ok=True)
    con = duckdb.connect()
    con.sql("set threads to 32")
    md = pd.read_parquet(f"{dev}/Metadata.parquet").iloc[0]
    ex = pd.read_parquet(f"{dev}/ExecutionInfo.parquet").sort_values("execution_start_ts").reset_index(drop=True)
    ex = ex.rename(columns={"execution_index": "exec", "execution_start_ts": "t0", "execution_end_ts": "t1"})
    ex["dev_ns"] = ex.t1 - ex.t0
    con.register("ex", ex[["exec", "t0", "t1"]])

    # live system trace: nc_exec_running on one pcore, aligned by order
    ev = pd.read_parquet(f"{sysd}/SystemProfileEvents.parquet")
    nc = ev[ev.name == "nc_exec_running"]
    pc = nc[nc.pcore_idx == nc.pcore_idx.min()].sort_values("start_ts").reset_index(drop=True)
    if len(pc) != len(ex):
        print(f"WARNING: {len(pc)} nc_exec_running vs {len(ex)} device executions", file=sys.stderr)
    n = min(len(pc), len(ex))
    ex["sys_ns"] = np.nan
    ex.loc[: n - 1, "sys_ns"] = pc.duration_ns.values[:n]
    ex["sys_start"] = np.nan
    ex.loc[: n - 1, "sys_start"] = pc.start_ts.values[:n].astype("float64")
    ex["gap_before_ns"] = ex.t0 - ex.t1.shift(1)

    # instruction aggregates (execution_index is carried by every instruction)
    I = f"'{dev}/Instruction.parquet'"
    ins = con.sql(f"""select execution_index exec, count(*) n_inst,
        sum(coalesce(hbm_read_bytes,0)) hbm_rd, sum(coalesce(hbm_write_bytes,0)) hbm_wr,
        sum(coalesce(spill_save_bytes,0)) spill_save, sum(coalesce(spill_reload_bytes,0)) spill_reload,
        sum(case when engine='Tensor' then coalesce(raw_flops,0) else 0 end) raw_flops,
        sum(case when engine='Tensor' then coalesce(adjusted_flops,0) else 0 end) adj_flops,
        sum(case when engine='Tensor' and tensor_instruction_type='TRANSPOSE' then coalesce(adjusted_flops,0) else 0 end) adj_flops_tr,
        sum(case when engine='Tensor' and tensor_instruction_type='TRANSPOSE' then coalesce(raw_flops,0) else 0 end) raw_flops_tr,
        sum(case when engine='Tensor' and opcode='MATMUL_MX' then coalesce(raw_flops,0) else 0 end) raw_flops_mx,
        sum(case when engine='Tensor' and src2_dtype='fp8e4' then coalesce(raw_flops,0) else 0 end) raw_flops_fp8
        from {I} group by 1""").df().set_index("exec")
    ex = ex.join(ins, on="exec")

    # engine active-wall, merged across both pcores, clipped to the execution window
    AT = f"'{dev}/ActiveTime.parquet'"
    engines = [r[0] for r in con.sql(f"select distinct engine from {AT}").fetchall()]
    for eng in engines:
        w = merged_wall(con, f"""select ex.exec, greatest(a.start_ts, ex.t0) s, least(a.end_ts, ex.t1) e
            from {AT} a asof join ex on a.start_ts >= ex.t0 where a.engine='{eng}' and a.start_ts < ex.t1""")
        ex[f"act_{eng}"] = ex.exec.map(w).fillna(0)

    # DMA: all NeuronCore DMA packets (not host<->device), merged; bytes by queue type
    D = f"'{dev}/DmaPacket.parquet'"
    w = merged_wall(con, f"""select ex.exec, greatest(d.start_ts, ex.t0) s, least(d.end_ts, ex.t1) e
        from {D} d asof join ex on d.start_ts >= ex.t0 where not d.is_host_device_dma and d.start_ts < ex.t1""")
    ex["act_dma"] = ex.exec.map(w).fillna(0)
    qb = con.sql(f"""select ex.exec, coalesce(d.queue_type, 'host') qt, sum(d.transfer_bytes) b
        from {D} d asof join ex on d.start_ts >= ex.t0 where d.start_ts < ex.t1 group by all""").df()
    qb = qb.pivot(index="exec", columns="qt", values="b").fillna(0)
    qb.columns = [f"dma_{c}" for c in qb.columns]
    ex = ex.join(qb, on="exec")
    ex["dma_bytes"] = ex[[c for c in qb.columns if c != "dma_host"]].sum(axis=1)

    # collectives
    C = f"'{dev}/CcOp.parquet'"
    w = merged_wall(con, f"""select ex.exec, greatest(c.start_ts, ex.t0) s, least(c.end_ts, ex.t1) e
        from {C} c asof join ex on c.start_ts >= ex.t0 where c.operation != 'Invalid' and c.start_ts < ex.t1""")
    ex["act_cc"] = ex.exec.map(w).fillna(0)
    cc = con.sql(f"""select ex.exec, count(*) n_cc, sum(greatest(c.input_size, c.output_size)) cc_bytes
        from {C} c asof join ex on c.start_ts >= ex.t0 where c.operation != 'Invalid' and c.start_ts < ex.t1 group by 1""").df().set_index("exec")
    ex = ex.join(cc, on="exec")

    # labels
    neffs = {os.path.basename(p): p for p in glob.glob(f"{ROOT}/profiles/*/output/neffs/**/*.neff", recursive=True)}
    names = {}
    for nf in ex.neff_name.unique():
        names[nf] = neff_kernel(neffs[nf]) if nf in neffs else None
    ex["kernel"] = [names.get(nf) for nf in ex.neff_name]
    role, layer, arch, step = [], [], [], []
    cur_step, li = -1, None
    for k in ex.kernel:
        k = k if isinstance(k, str) else None
        if k == "hc_expand":
            cur_step += 1
            li = 0
            role.append("embed+hc_expand"); layer.append(None); arch.append(None)
        elif k and k.startswith("layer_"):
            role.append("layer"); layer.append(li); arch.append(arch_of(li) if li is not None else None)
            li = None if li is None else li + 1
        elif k and k.startswith("draft_"):
            role.append("draft kernel"); layer.append(None); arch.append(None)
        elif k == "_sampler" or (k is None and li == 40):
            role.append("head+sampler"); layer.append(None); arch.append(None); li = 41
        else:
            role.append("draft glue"); layer.append(None); arch.append(None)
        step.append(cur_step)
    ex["role"], ex["layer"], ex["arch"], ex["step"] = role, layer, arch, step
    ex["layer"] = ex["layer"].astype("float")
    for k in {k for k in ex.kernel if isinstance(k, str)}:
        if k.startswith("layer_"):
            got = set(ex.loc[ex.kernel == k, "arch"].dropna())
            want = k[len("layer_"):-len("_tp64")].split("_", 1)[1]
            if got - {want}:
                print(f"WARNING: {k} executed at layers of archetype {got}", file=sys.stderr)
    # The device buffers fill in time order: when later executions have no instruction data,
    # the last one that has some was cut part-way through -- drop its device metrics.
    has = ex.n_inst.notna()
    ex["partial"] = False
    if has.any() and (~has[has[has].index.max():]).any():
        last = has[has].index.max()
        ex.loc[last, "partial"] = True
        dev_cols = [c for c in ex.columns if c.startswith(("act_", "dma_", "hbm_", "spill_", "raw_flops", "adj_flops", "n_inst", "n_cc", "cc_bytes"))]
        ex.loc[last, dev_cols] = np.nan
    ex.to_parquet(f"{out}/execs.parquet")
    te_peak = md.tensor_engine_num_rows * md.tensor_engine_num_cols * 2 * md.tensor_engine_clock_freq * 1e9 * md.num_physical_cores
    meta = {"run": run, "te_peak_bf16_flops": te_peak, "hbm_bw": float(md.hbm_ddr_bandwidth),
            "dma_bw_per_pcore": float(md.dma_ddr_bandwidth), "num_physical_cores": int(md.num_physical_cores),
            "instance_type": md.instance_type, "runtime_version": md.runtime_version, "compiler_version": md.compiler_version,
            "warnings": pd.read_parquet(f"{dev}/Warning.parquet").message.tolist(),
            "n_exec": len(ex), "n_exec_with_instructions": int(ex.n_inst.notna().sum()), "engines": engines}
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    print(f"{run}: {len(ex)} executions ({meta['n_exec_with_instructions']} with instruction data), steps {ex.step.max()+1}")


if __name__ == "__main__":
    main(sys.argv[1])
