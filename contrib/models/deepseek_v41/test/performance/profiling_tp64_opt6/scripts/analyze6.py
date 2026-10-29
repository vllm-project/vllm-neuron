#!/usr/bin/env python3
"""Per-execution and per-unit device metrics for one ingested capture (one rank), opt6 code.

  analyze6.py <run_name>  ->  analysis/<run_name>/{execs,units,ccops}.parquet + meta.json

execs   one row per NEFF execution in the window: kernel label, role, step, device time
        (ExecutionInfo, cross-checked against the live nc_exec_running trace), device-idle gap
units   the unit the report tables are about: a NEFF execution, except that a fused decode
        launch (`layers_decode[_embed]_Lxx_Lyy[_head]_tp64`, 8 blocks in one NKI kernel) is
        split into its members -- [kernel-entry barrier] [embed] block ... block [head] --
        at the end of each block's world all-reduce (the block's last collective; the next
        member needs its output stream). Every unit carries the same metric columns:
  engines    active-wall per engine, interval-merged over both physical cores
  DMA        active-wall of the core's DMA packets (merged), bytes by queue type
  HBM        bytes read/written by DMA-issuing instructions (Instruction.hbm_*_bytes)
  TensorE    BF16-equivalent (adjusted) FLOPs, transposes separately
  CC         active-wall of collectives (merged) and the kernel-entry barrier's share
ccops   every collective with its unit, kind (barrier / allreduce-8r / allreduce-64r / ...),
        payload and the NKI source line of the instruction that triggered it
"""
import glob, io, json, os, re, sys, tarfile
import duckdb, numpy as np, pandas as pd

ROOT = "/mnt/nvme/ds41_profiling_opt6"
CFG = json.load(open("/mnt/nvme/hf/config.json"))["text_config"]
DIM = CFG["hidden_size"]


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


# kernel names, most specific first (a fused NEFF also contains its members' names)
_PATS = [re.compile(p) for p in (
    rb"layers_(?:prefill|decode)(?:_embed)?_L\d\d_L\d\d(?:_head)?_tp64",
    rb"draft_(?:prefill|decode)(?:_batched)?_tp64",
    rb"layer_(?:prefill|decode)_[a-z0-9_]+?_tp64",
    rb"hc_expand",
    rb"vllm_neuron/nn/rejection_sampler\.py",
    rb"_sampler",
    rb"deepseek_v41/mk_model\.",
    rb"_all-gather\.\d+",
    rb"_gather\.\d+",
)]


def neff_kernel(path):
    b = open(path, "rb").read()
    try:
        t = tarfile.open(fileobj=io.BytesIO(b[1024:]))
    except Exception:
        return None
    blobs = [t.extractfile(m).read() for m in t.getmembers() if m.isfile() and m.size < 80_000_000]
    for p in _PATS:
        for blob in blobs:
            hit = p.search(blob)
            if hit:
                return hit.group(0).decode()
    return None


def classify(k):
    """kernel label -> (role, short name)"""
    if not isinstance(k, str):
        return "glue", "glue"
    if k.startswith("layers_"):
        return "fused", k
    if k.startswith("layer_"):
        return "layer", k
    if k.startswith("draft_"):
        return "draft", k
    if k == "hc_expand":
        return "embed", "embedding + hc_expand"
    if "rejection_sampler" in k:
        return "glue", "rejection sampler"
    if k == "_sampler":
        return "head", "final norm + lm_head + sampler"
    if "mk_model" in k:
        return "glue", "drafter meta"
    if k.startswith("_all-gather"):
        return "head", "tail all-gather graph"
    return "glue", "gather/concat graph"


def merged_wall(con, table_sql, key="u"):
    """Sum of merged [s, e) intervals per key; table_sql yields (key, s, e)."""
    return con.sql(f"""
      with p as ({table_sql}),
      q as (select {key}, s, e, max(e) over (partition by {key} order by s, e
                 rows between unbounded preceding and 1 preceding) pm from p),
      r as (select {key}, s, e, sum(case when pm is null or s > pm then 1 else 0 end)
                 over (partition by {key} order by s, e rows unbounded preceding) g from q)
      select {key}, sum(ge - gs) wall from (select {key}, g, min(s) gs, max(e) ge from r group by all)
      group by {key}""").df().set_index(key)["wall"]


def group_size(rg):
    if not isinstance(rg, str) or not rg.startswith("[["):
        return None
    return len(rg[2:].split("]")[0].split(","))


def cc_kind(r):
    if r.operation == "BARRIER":
        return "barrier"
    n = group_size(r.replica_group)
    return f"{r.operation.lower()}-{n}r"


def source_lines():
    """kernel package path suffix -> file lines (for naming a collective's tensor)"""
    kroot = "/home/ubuntu/vllm-neuron/contrib/models/deepseek_v41/src/kernels"
    out = {}
    for p in glob.glob(f"{kroot}/**/*.py", recursive=True):
        out[os.path.relpath(p, kroot)] = open(p).read().splitlines()
    return out


_SRC = source_lines()


def src_text(loc):
    """'/.../kernels/tp64/r2_decode/r2_decode.py:864' -> (rel path:line, stripped line text)"""
    if not isinstance(loc, str):
        return None, None
    m = re.search(r"kernels/(.+\.py):(\d+)", loc)
    if not m:
        return loc, None
    rel, ln = m.group(1), int(m.group(2))
    lines = _SRC.get(rel)
    if lines is None and f"tp64/{rel}" in _SRC:   # the drafter's NEFF records kernels/<file>
        rel = f"tp64/{rel}"
        lines = _SRC[rel]
    txt = lines[ln - 1].strip() if lines and 0 < ln <= len(lines) else None
    if lines and txt is not None and "ncc." not in txt:
        # a NEFF built from an earlier revision of the file (the drafter's: 4 lines shorter):
        # the collective is the nearest `ncc.` call
        near = [(abs(j - ln), j) for j in range(max(1, ln - 10), min(len(lines), ln + 10) + 1)
                if "ncc." in lines[j - 1]]
        if near:
            j = min(near)[1]
            return f"{rel}:{j}", lines[j - 1].strip()
    return f"{rel}:{ln}", txt


def main(run):
    G = f"{ROOT}/data/{run}/profiles/global"
    dev = glob.glob(f"{G}/*session_*@latest")[0]
    sysd = glob.glob(f"{G}/{run}@latest")[0]
    out = f"{ROOT}/analysis/{run}"
    os.makedirs(out, exist_ok=True)
    con = duckdb.connect()
    con.sql("set threads to 24")
    md = pd.read_parquet(f"{dev}/Metadata.parquet").iloc[0]
    ex = pd.read_parquet(f"{dev}/ExecutionInfo.parquet").sort_values("execution_start_ts").reset_index(drop=True)
    ex = ex.rename(columns={"execution_index": "exec", "execution_start_ts": "t0", "execution_end_ts": "t1"})
    ex["dev_ns"] = ex.t1 - ex.t0

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

    # labels
    neffs = {os.path.basename(p): p for p in glob.glob(f"{ROOT}/profiles/*/output/neffs/**/*.neff", recursive=True)}
    names = {nf: (neff_kernel(neffs[nf]) if nf in neffs else None) for nf in ex.neff_name.unique()}
    ex["kernel"] = [names.get(nf) for nf in ex.neff_name]
    rs = [classify(k) for k in ex.kernel]
    ex["role"] = [r for r, _ in rs]
    ex["name"] = [s for _, s in rs]
    # steps: a prefill iteration starts at the embedding graph, a decode step at its first
    # fused launch (the one that embeds)
    step, cur, li = [], -1, None
    layer = []
    for r in ex.itertuples():
        k = r.kernel if isinstance(r.kernel, str) else ""
        if r.role == "embed" or (r.role == "fused" and "_embed_" in k):
            cur += 1
            li = 0
        if r.role == "layer":
            layer.append(li)
            li = None if li is None else li + 1
        else:
            layer.append(None)
        step.append(cur)
    ex["step"], ex["layer"] = step, layer
    ex["layer"] = ex["layer"].astype("float")
    ex["arch"] = [arch_of(int(l)) if not np.isnan(l) else None for l in ex.layer]
    for r in ex[ex.role == "layer"].itertuples():
        want = r.kernel[len("layer_prefill_"):-len("_tp64")] if r.kernel.startswith("layer_prefill_") else None
        if want and r.arch != want:
            print(f"WARNING: exec {r.exec} kernel {r.kernel} at layer {r.layer} ({r.arch})", file=sys.stderr)
            break

    # instruction presence (device buffers fill in time order)
    I = f"'{dev}/Instruction.parquet'"
    ninst = con.sql(f"select execution_index exec, count(*) n from {I} group by 1").df().set_index("exec").n
    ex["n_inst"] = ex.exec.map(ninst)

    # ---- collectives, with the NKI source of their trigger instruction
    cc = pd.read_parquet(f"{dev}/CcOp.parquet")
    cc = cc[cc.operation != "Invalid"].sort_values("start_ts").reset_index(drop=True)
    cc["exec"] = pd.merge_asof(cc[["start_ts"]], ex[["t0", "exec"]], left_on="start_ts", right_on="t0")["exec"].values
    cc = cc.merge(ex[["exec", "t1"]], on="exec", how="left")
    cc = cc[cc.start_ts < cc.t1].drop(columns="t1")
    cc["kind"] = [cc_kind(r) for r in cc.itertuples()]
    cc["gsize"] = [group_size(g) for g in cc.replica_group]
    con.register("ccs", cc[["id", "trigger_engine", "trigger_start_ts"]].assign(eng=cc.trigger_engine))
    # the trigger is the sequencer WRITE to the collective doorbell at trigger_start_ts; it
    # carries the `ncc.all_reduce(...)` / `all_gather(...)` source line
    locs = con.sql(f"""select c.id, i.nki_source_location loc, c.trigger_start_ts - i.start_ts dt from ccs c asof join
        (select engine, start_ts, nki_source_location from {I}
         where execution_unit='sequencer' and opcode='WRITE' and nki_source_location is not null) i
        on i.engine = c.eng and i.start_ts <= c.trigger_start_ts""").df().set_index("id")
    cc["trigger_loc"] = cc.id.map(locs["loc"].where(locs.dt.abs() < 2000))
    st = [src_text(l) for l in cc.trigger_loc]
    cc["src"] = [a for a, _ in st]
    cc["src_text"] = [b for _, b in st]

    # ---- units: executions, fused ones split into members
    units = []
    for r in ex.itertuples():
        base = dict(exec=r.exec, step=r.step, role=r.role, kernel=r.kernel)
        e_cc = cc[cc.exec == r.exec]
        bar = e_cc[e_cc.kind == "barrier"]
        if r.role != "fused":
            units.append(dict(base, unit=r.name if r.role != "layer" else f"L{int(r.layer):02d} {r.arch}",
                              layer=r.layer, arch=r.arch, member=0, t0=r.t0, t1=r.t1))
            continue
        m = re.match(r"layers_(\w+?)(_embed)?_L(\d\d)_L(\d\d)(_head)?_tp64", r.kernel)
        lo, hi = int(m.group(3)), int(m.group(4))
        ids = list(range(lo, hi + 1))
        nz = e_cc[e_cc.kind != "barrier"].sort_values("start_ts")
        # a block's world all-reduce: 64 ranks, fp32, [T, dim] (T = this step's rows)
        w = nz[(nz.gsize == 64) & (nz.dtype == "fp32") & (nz.input_size % (4 * DIM) == 0)]
        sizes = w.input_size.value_counts()
        size = next((s for s, c in sizes.items() if c >= len(ids)), None)
        ends = w[w.input_size == size].end_ts.tolist() if size else []
        if len(ends) != len(ids):
            print(f"WARNING: exec {r.exec} {r.kernel}: {len(ends)} world all-reduces for {len(ids)} blocks",
                  file=sys.stderr)
            units.append(dict(base, unit=r.kernel, layer=None, arch=None, member=0, t0=r.t0, t1=r.t1))
            continue
        t = r.t0
        mi = 0
        if len(bar):
            be = int(bar.end_ts.max())
            units.append(dict(base, unit="kernel-entry barrier", layer=None, arch=None, member=mi, t0=t, t1=be))
            t, mi = be, mi + 1
        if m.group(2):
            # the embedding member ends at its own all-reduce's end (the first non-barrier CC)
            emb_end = int(nz.end_ts.iloc[0])
            units.append(dict(base, unit="embed + hc_expand (fused)", layer=None, arch=None, member=mi, t0=t, t1=emb_end))
            t, mi = emb_end, mi + 1
        for lid, e in zip(ids, ends):
            units.append(dict(base, unit=f"L{lid:02d} {arch_of(lid)}", layer=float(lid), arch=arch_of(lid),
                              member=mi, t0=t, t1=int(e)))
            t, mi = int(e), mi + 1
        if m.group(5):
            units.append(dict(base, unit="greedy head (fused)", layer=None, arch=None, member=mi, t0=t, t1=r.t1))
        elif t < r.t1:
            units[-1]["t1"] = r.t1
    U = pd.DataFrame(units)
    U["u"] = np.arange(len(U))
    U["dur_ns"] = U.t1 - U.t0
    con.register("U", U[["u", "exec", "t0", "t1"]])

    # ---- metrics per unit (window clipped)
    ins = con.sql(f"""select U.u, count(*) n_inst,
        sum(coalesce(hbm_read_bytes,0)) hbm_rd, sum(coalesce(hbm_write_bytes,0)) hbm_wr,
        sum(coalesce(spill_save_bytes,0)) spill_save, sum(coalesce(spill_reload_bytes,0)) spill_reload,
        sum(case when engine='Tensor' then coalesce(raw_flops,0) else 0 end) raw_flops,
        sum(case when engine='Tensor' then coalesce(adjusted_flops,0) else 0 end) adj_flops,
        sum(case when engine='Tensor' and tensor_instruction_type='TRANSPOSE' then coalesce(adjusted_flops,0) else 0 end) adj_flops_tr
        from {I} i join U on i.execution_index = U.exec and i.start_ts >= U.t0 and i.start_ts < U.t1
        group by 1""").df().set_index("u")
    U = U.join(ins, on="u")
    AT = f"'{dev}/ActiveTime.parquet'"
    engines = [r[0] for r in con.sql(f"select distinct engine from {AT}").fetchall()]
    for eng in engines:
        wl = merged_wall(con, f"""select U.u, greatest(a.start_ts, U.t0) s, least(a.end_ts, U.t1) e
            from {AT} a join U on a.end_ts > U.t0 and a.start_ts < U.t1 where a.engine='{eng}'""")
        U[f"act_{eng}"] = U.u.map(wl).fillna(0)
    D = f"'{dev}/DmaPacket.parquet'"
    wl = merged_wall(con, f"""select U.u, greatest(d.start_ts, U.t0) s, least(d.end_ts, U.t1) e
        from {D} d join U on d.end_ts > U.t0 and d.start_ts < U.t1
        where not d.is_host_device_dma and coalesce(d.queue_type,'') != 'collectives'""")
    U["act_dma"] = U.u.map(wl).fillna(0)
    qb = con.sql(f"""select U.u, coalesce(d.queue_type, 'other') qt, sum(d.transfer_bytes) b
        from {D} d join U on d.start_ts >= U.t0 and d.start_ts < U.t1 where not d.is_host_device_dma group by all""").df()
    qb = qb.pivot(index="u", columns="qt", values="b").fillna(0)
    qb.columns = [f"dma_{c}" for c in qb.columns]
    U = U.join(qb, on="u")
    U["dma_bytes"] = U[[c for c in qb.columns if c != "dma_collectives"]].sum(axis=1)
    con.register("CC", cc[["start_ts", "end_ts", "kind"]])
    wl = merged_wall(con, """select U.u, greatest(c.start_ts, U.t0) s, least(c.end_ts, U.t1) e
        from CC c join U on c.end_ts > U.t0 and c.start_ts < U.t1""")
    U["act_cc"] = U.u.map(wl).fillna(0)
    wl = merged_wall(con, """select U.u, greatest(c.start_ts, U.t0) s, least(c.end_ts, U.t1) e
        from CC c join U on c.end_ts > U.t0 and c.start_ts < U.t1 where c.kind = 'barrier'""")
    U["act_barrier"] = U.u.map(wl).fillna(0)
    # units without instruction data (buffers full) carry no device metrics
    has = U.exec.map(ninst).notna()
    dev_cols = [c for c in U.columns if c.startswith(("act_", "dma_", "hbm_", "spill_", "raw_flops", "adj_flops", "n_inst"))]
    U.loc[~has, dev_cols] = np.nan
    # the last execution with instruction data may have been cut part-way through
    last = ex[ex.n_inst.notna()].exec.max() if ex.n_inst.notna().any() else None
    ex["partial"] = False
    if last is not None and (ex.n_inst.isna() & (ex.exec > last)).any():
        ex.loc[ex.exec == last, "partial"] = True
        U.loc[U.exec == last, dev_cols] = np.nan
    cc = cc.merge(U[["u", "exec", "t0", "t1"]], on="exec", how="left")
    cc = cc[(cc.start_ts >= cc.t0) & (cc.start_ts < cc.t1)].drop(columns=["t0", "t1"])
    U.to_parquet(f"{out}/units.parquet")
    ex.to_parquet(f"{out}/execs.parquet")
    cc.drop(columns=["replica_group", "src_target_pairs"]).assign(
        replica_group=cc.replica_group.astype(str).str[:200]).to_parquet(f"{out}/ccops.parquet")
    te_peak = md.tensor_engine_num_rows * md.tensor_engine_num_cols * 2 * md.tensor_engine_clock_freq * 1e9 * md.num_physical_cores
    meta = {"run": run, "dev_dir": dev, "sys_dir": sysd, "te_peak_bf16_flops": te_peak, "hbm_bw": float(md.hbm_ddr_bandwidth),
            "dma_bw_per_pcore": float(md.dma_ddr_bandwidth), "num_physical_cores": int(md.num_physical_cores),
            "instance_type": md.instance_type, "runtime_version": md.runtime_version, "compiler_version": md.compiler_version,
            "explorer_version": md.explorer_version, "neuron_device_idx": int(md.neuron_device_idx),
            "neuroncore_idx": int(md.neuroncore_idx), "first_hw_timestamp": int(md.first_hw_timestamp),
            "warnings": pd.read_parquet(f"{dev}/Warning.parquet").message.tolist()[:50],
            "n_exec": len(ex), "n_exec_with_instructions": int(ex.n_inst.notna().sum()), "engines": engines}
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1, default=str)
    print(f"{run}: {len(ex)} executions ({meta['n_exec_with_instructions']} with instruction data), "
          f"{len(U)} units, steps {ex.step.max() + 1}")


if __name__ == "__main__":
    main(sys.argv[1])
