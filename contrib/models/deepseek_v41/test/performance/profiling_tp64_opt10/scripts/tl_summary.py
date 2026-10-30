#!/usr/bin/env python3
"""Peak MFU / MBU and the anatomy of the under-utilised time, per megakernel and overall.

  tl_summary.py <run> [--steps 1,2,...] [--group kernel|unit|arch]

A bin is UNDER-UTILISED when both its MFU and its MBU are below half of the peak of its
group (the group's best 10 us bin), i.e. neither the TensorE nor HBM is anywhere near what
that kernel reaches elsewhere. Each under-utilised bin is charged to ONE cause, in order:
  1. engine-bound     an engine is >= 70 % busy on the busier physical core
                      ("TensorE busy" = busy at low efficiency: small / fp32 / transposed matmuls)
  2. collective       a collective is in flight >= 50 % of the bin; split into
                      "waiting for peers" (the op's time beyond the fastest instance of the same
                      collective in the run) and "transfer"; the kernel-entry barrier separately
  3. DMA-queue-bound  the core's DMA queues are active >= 70 % of the bin (but move little)
  4. dependency stall none of the above: engines wait on each other's semaphores
and named by the collective (its tensor) or the NKI function on the deciding engine.
"""
import argparse, json, re, sys
import numpy as np, pandas as pd

ROOT = "/mnt/nvme/ds41_profiling_opt10"
ENG_NAME = {"tensor": "TensorE", "vector": "VectorE", "scalar": "ScalarE", "gpsimd": "GpSimd", "sync": "SP(sync)"}

# tensors behind the collectives (variable in the ncc call -> what it carries)
TENSOR = {
    "cc_src": "attention out",
    "cc2_src": "MoE out",
    "sc_src": "indexer scores",
    "e_src": "token embedding",
    "e_loc": "draft-token embedding",
    "emb": "engram rows",
    "gv_src": "argmax values",
    "gi_src": "argmax indices",
    "mcc_src": "draft Markov embedding",
}


def cc_label(r):
    if r.kind == "barrier":
        return "kernel-entry barrier"
    m = re.search(r"srcs=\[(\w+)\]", r.src_text if isinstance(r.src_text, str) else "")
    var = m.group(1) if m else None
    op = {"allreduce": "AR", "allgather": "AG"}.get(r.kind.split("-")[0], r.kind.split("-")[0])
    op = f"{op}-{r.kind.split('-')[1][:-1]}" if "-" in r.kind else op
    sz = r.input_size
    szs = f"{sz / 1e6:.1f} MB" if sz >= 1e6 else f"{sz / 1e3:.0f} KB" if sz >= 1e3 else f"{sz} B"
    if var is None:
        return f"{op} XLA graph collective ({r['dtype']}, {szs})"
    return f"{op} `{var}` ({TENSOR.get(var, var)}, {r['dtype']} {szs})"


def classify(T, cc):
    lab = cc.set_index("cid").apply(cc_label, axis=1)
    kind = cc.set_index("cid").kind
    eng = [e for e in ("tensor", "vector", "scalar", "gpsimd", "sync") if f"m_{e}" in T]
    m = T[[f"m_{e}" for e in eng]].to_numpy()
    top = m.argmax(axis=1)
    topv = m.max(axis=1)
    cause, item = [], []
    for i, r in enumerate(T.itertuples()):
        if topv[i] >= 0.7:
            e = eng[top[i]]
            cause.append(f"{ENG_NAME[e]}-bound")
            item.append(getattr(r, f"func_{e}", None) or r.func_any or "?")
        elif r.cc_frac >= 0.5 and r.cc_id >= 0:
            if kind.get(r.cc_id) == "barrier":
                cause.append("kernel-entry barrier (waiting for all ranks to launch)")
                item.append("kernel-entry barrier")
            elif r.cc_wait >= 0.5 * r.blen:
                cause.append("collective: waiting for peer ranks")
                item.append(lab.get(r.cc_id, "?"))
            else:
                cause.append("collective: transfer")
                item.append(lab.get(r.cc_id, "?"))
        elif r.dma_act_max >= 0.7:
            cause.append("DMA-queue-bound (queues busy, low bytes)")
            item.append(r.func_any if isinstance(r.func_any, str) else "(DMA only)")
        else:
            cause.append("dependency stall (no engine >= 70 % busy)")
            item.append(r.func_any if isinstance(r.func_any, str) else "(idle)")
    T = T.copy()
    T["cause"], T["item"] = cause, item
    return T


def group_key(T, how):
    if how == "kernel":
        k = T.kernel.fillna(T.unit)
        k = k.where(T.role != "fused", T.kernel)
        return k
    if how == "unit":
        return T.unit
    if how == "arch":
        return T.arch.fillna(T.unit)
    raise ValueError(how)


def summarize(T, keys, meta, peak_from=None):
    """per group: peaks, means, under-utilised time and its causes"""
    rows, causes = [], []
    for g, d in T.groupby(keys, sort=False):
        w = d.blen
        tot = w.sum()
        pk = peak_from.loc[g] if peak_from is not None else (d.mfu.max(), d.mbu.max())
        pmfu, pmbu = pk
        low = (d.mfu <= 0.5 * pmfu) & (d.mbu <= 0.5 * pmbu)
        near = ~low
        top_mfu = d.loc[d.mfu.idxmax()]
        top_mbu = d.loc[d.mbu.idxmax()]
        rows.append({"group": g, "ms": tot / 1e6, "bins": len(d),
                     "peak_mfu": pmfu, "peak_mfu_at": f"{top_mfu.unit} @ {top_mfu.func_tensor if isinstance(top_mfu.func_tensor, str) else top_mfu.func_any}",
                     "peak_mbu": pmbu, "peak_mbu_at": f"{top_mbu.unit} @ {top_mbu.func_any if isinstance(top_mbu.func_any, str) else '(DMA only)'}",
                     "p95_mfu": d.mfu.quantile(0.95), "p95_mbu": d.mbu.quantile(0.95),
                     "mean_mfu": (d.flops.sum()) / (tot / 1e9 * meta["te_peak_bf16_flops"]),
                     "mean_mbu": (d.hbm_b.sum()) / (tot / 1e9 * meta["hbm_bw"]),
                     "near_peak_pct": 100 * w[near].sum() / tot, "under_pct": 100 * w[low].sum() / tot})
        L = d[low]
        if len(L):
            c = L.groupby(["cause", "item"]).blen.sum().reset_index()
            c["group"] = g if not isinstance(g, tuple) else "|".join(map(str, g))
            c["ms"] = c.blen / 1e6
            c["pct_of_group"] = 100 * c.blen / tot
            causes.append(c.drop(columns="blen"))
    R = pd.DataFrame(rows)
    C = pd.concat(causes, ignore_index=True) if causes else pd.DataFrame()
    return R, C


def load(run, steps=None, bin_us=10):
    T = pd.read_parquet(f"{ROOT}/analysis/{run}/bins_{bin_us}us.parquet")
    cc = pd.read_parquet(f"{ROOT}/analysis/{run}/cc_{bin_us}us.parquet")
    meta = json.load(open(f"{ROOT}/analysis/{run}/meta.json"))
    if steps is not None:
        T = T[T.step.isin(steps)]
    return classify(T, cc), cc, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--steps", default=None)
    ap.add_argument("--group", default="kernel")
    a = ap.parse_args()
    steps = [int(s) for s in a.steps.split(",")] if a.steps else None
    T, cc, meta = load(a.run, steps)
    T["g"] = group_key(T, a.group)
    R, C = summarize(T, "g", meta)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print(R.round(3).to_string(index=False))
    T["all"] = "all"
    R2, C2 = summarize(T, "all", meta)
    print(R2.round(3).to_string(index=False))
    print(C2.sort_values("ms", ascending=False).head(25).round(3).to_string(index=False))
    print(C.groupby(["group", "cause"]).ms.sum().round(3).to_string())


if __name__ == "__main__":
    main()
