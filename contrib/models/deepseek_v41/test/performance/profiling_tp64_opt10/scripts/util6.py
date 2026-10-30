#!/usr/bin/env python3
"""Peak-utilisation and blocking-cause tables for the opt10 report (markdown)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import tables6
from tl_summary import load as tl_load, summarize

CAT = [  # (column, causes)
    ("CC wait", ["collective: waiting for peer ranks"]),
    ("entry barrier", ["kernel-entry barrier (waiting for all ranks to launch)"]),
    ("CC transfer", ["collective: transfer"]),
    ("TensorE busy", ["TensorE-bound"]),
    ("VectorE busy", ["VectorE-bound"]),
    ("ScalarE/GpSimd/SP busy", ["ScalarE-bound", "GpSimd-bound", "SP(sync)-bound"]),
    ("DMA queues", ["DMA-queue-bound (queues busy, low bytes)"]),
    ("dep. stall", ["dependency stall (no engine >= 70 % busy)"]),
]
SHORT = {"kernel-entry barrier (waiting for all ranks to launch)": "entry barrier",
         "collective: waiting for peer ranks": "waiting for peers",
         "collective: transfer": "CC transfer",
         "DMA-queue-bound (queues busy, low bytes)": "DMA queues busy",
         "dependency stall (no engine >= 70 % busy)": "dependency stall"}


def kernel_label(r):
    k = r.kernel if isinstance(r.kernel, str) else r.unit
    if r.role == "draft":
        return "DSpark drafter"
    if r.role == "fused":
        return k
    if r.role == "layer":
        return k
    return r.unit


def mk_group(T, how):
    if how == "launch":
        return [kernel_label(r) for r in T[["kernel", "unit", "role"]].itertuples()]
    if how == "member":
        g = T.arch.where(T.arch.notna(), T.unit)
        return g.where(T.role != "draft", "DSpark drafter")
    raise ValueError(how)


def fmt_pct(x):
    return "–" if x is None or pd.isna(x) else f"{100 * x:.1f}"


def short_item(cause, item):
    if not isinstance(item, str):
        return SHORT.get(cause, cause)
    item = item.replace("all-reduce ", "AR ").replace("all-gather ", "AG ")
    if cause.startswith("collective"):
        return f"{SHORT.get(cause, cause)} @ {item}"
    if cause.startswith("kernel-entry"):
        return "entry barrier"
    return f"{cause.replace('-bound', '')}: `{item}`" if "bound" in cause else f"{SHORT.get(cause, cause)}: `{item}`"


def megakernel_table(run, steps, phase, how, top_items=2, idle_ns_per_step=None):
    """markdown table: one row per megakernel / block kind, plus 'all' (overall)"""
    T, cc, meta = tl_load(run, steps)
    T["g"] = mk_group(T, how)
    R, C = summarize(T, "g", meta)
    T["all"] = "all device time"
    R2, C2 = summarize(T, "all", meta)
    k = len(steps) if phase == "decode" else 1
    unit = "ms / step" if phase == "decode" else "ms"
    head = (f"| Megakernel | {unit} | peak MFU % (10 µs bin, where) | peak MBU % (where) | mean MFU / MBU % | "
            f"under-utilised % | " + " | ".join(c for c, _ in CAT) + " | largest blocking items |")
    out = [head, "|---|---:|---|---|---:|---:|" + "---:|" * len(CAT) + "---|"]
    allr = pd.concat([R.sort_values("ms", ascending=False), R2])
    allc = pd.concat([C, C2.assign(group="all device time")]) if len(C2) else C
    for r in allr.itertuples():
        cg = allc[allc.group == r.group] if len(allc) else pd.DataFrame()
        cells = []
        for col, causes in CAT:
            v = cg[cg.cause.isin(causes)].pct_of_group.sum() if len(cg) else 0
            cells.append(f"{v:.0f}" if v >= 0.5 else "·")
        top = cg.sort_values("ms", ascending=False).head(top_items) if len(cg) else []
        items = "; ".join(f"{short_item(t.cause, t.item)} ({t.ms / k:.2f} ms)" for t in
                          (top.itertuples() if len(cg) else []))
        name = f"**{r.group}**" if r.group == "all device time" else f"`{r.group}`" if "_" in str(r.group) else r.group
        out.append(f"| {name} | {r.ms / k:,.2f} | {100 * r.peak_mfu:.0f} ({_at(r.peak_mfu_at)}) | "
                   f"{100 * r.peak_mbu:.0f} ({_at(r.peak_mbu_at)}) | {100 * r.mean_mfu:.1f} / {100 * r.mean_mbu:.1f} | "
                   f"{r.under_pct:.0f} | " + " | ".join(cells) + f" | {items} |")
    return "\n".join(out), R, C, R2, C2, T


def _at(s):
    """'L14 r2_src_idx_e @ mk_compress.py:fp32_proj_rows' -> compact"""
    if not isinstance(s, str):
        return "?"
    unit, _, f = s.partition(" @ ")
    f = f.replace("(DMA only)", "DMA only").replace("nan", "DMA only")
    unit = unit.replace("draft_decode_tp64", "drafter").replace("draft_prefill_tp64", "drafter")
    return f"{unit}: {f}" if len(unit) < 22 else f


def overall_causes(R2, C2, steps, phase, idle_ms=None, outside_ms=None, top=14, total_ms=None, base=None):
    """markdown: the largest blocking items over the whole phase, incl. device idle"""
    k = len(steps) if phase == "decode" else 1
    dev_ms = R2.ms.iloc[0] / k
    rows = C2.sort_values("ms", ascending=False).head(top)
    tot = total_ms if total_ms else dev_ms + (idle_ms or 0) + (outside_ms or 0)
    unit = "ms / step" if phase == "decode" else "ms"
    base = base or ("step" if phase == "decode" else "phase")
    out = [f"| # | Blocked on | What exactly | {unit} | % of the {base} |",
           "|---:|---|---|---:|---:|"]
    n = 0
    extra = []
    if idle_ms:
        extra.append(("device idle (no launch on the core)",
                      "launch turnaround between queued launches" if phase == "decode" else
                      "host submission + launch turnaround between launches", idle_ms))
    if outside_ms:
        extra.append(("host outside the device window", "HTTP, tokenizer, first schedule, response", outside_ms))
    items = [(r.cause, r.item, r.ms / k) for r in rows.itertuples()] + [(a, b, c) for a, b, c in extra]
    items.sort(key=lambda x: -x[2])
    for cause, item, ms in items:
        n += 1
        what = item if isinstance(item, str) else ""
        if cause.startswith("collective") or cause.startswith("kernel-entry"):
            what = what
        elif "bound" in cause or "stall" in cause:
            what = f"`{what}`"
        lab = SHORT.get(cause, cause.replace("-bound", " busy (low efficiency)") if cause.endswith("-bound") else cause)
        out.append(f"| {n} | {lab} | {what} | {ms:,.2f} | {100 * ms / tot:.1f} |")
    return "\n".join(out)
