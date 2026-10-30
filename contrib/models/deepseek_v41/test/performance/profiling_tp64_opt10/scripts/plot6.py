#!/usr/bin/env python3
"""Figures for the opt10 report.

  plot6.py roofline <out.png>                 every block / drafter / other unit of every configuration
  plot6.py timeline <run> <step> <out.png> [--t0-ms A --t1-ms B] [--title T]
           MFU and MBU per 10 us bin over one decode step (or a prefill window), and below
           it one lane per blocking cause marking the under-utilised bins; launches labelled.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import tables6
from tl_summary import load as tl_load

ROOT = "/mnt/nvme/ds41_profiling_opt10"
# reference palette, light mode: the first three categorical slots (validated all-pairs)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False})

LANES = [  # (lane label, causes)
    ("kernel-entry barrier", ["kernel-entry barrier (waiting for all ranks to launch)"]),
    ("collective: waiting for peers", ["collective: waiting for peer ranks"]),
    ("collective: transfer", ["collective: transfer"]),
    ("TensorE busy, low efficiency", ["TensorE-bound"]),
    ("VectorE busy", ["VectorE-bound"]),
    ("ScalarE / GpSimd / SP busy", ["ScalarE-bound", "GpSimd-bound", "SP(sync)-bound"]),
    ("DMA queues busy, few bytes", ["DMA-queue-bound (queues busy, low bytes)"]),
    ("dependency stall", ["dependency stall (no engine >= 70 % busy)"]),
    ("device idle (no launch running)", ["__idle__"]),
]


def roofline(out):
    cfgs = json.load(open(f"{ROOT}/report_parts/configs.json"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    te = bw = None
    kinds = [("transformer block", BLUE), ("DSpark drafter", ORANGE), ("embed / head / glue", AQUA)]
    for ax, phase in zip(axes, ("prefill", "decode")):
        markers = iter(["o", "s", "^", "D"])
        for c in [c for c in cfgs if c["phase"] == phase]:
            mk = next(markers)
            for run, steps in c["roof"]:
                U, ex, meta = tables6.load(run)
                te, bw = meta["te_peak_bf16_flops"], meta["hbm_bw"]
                d = tables6.roof_metrics(U[U.n_inst.notna() & U.step.isin(steps) & (U.dur_ns > 20_000)], meta)
                d = d[(d.hbm > 0) & (d.useful > 0) & (d.unit != "kernel-entry barrier")]
                kind = np.where(d.layer.notna(), "transformer block", np.where(d.role == "draft", "DSpark drafter", "embed / head / glue"))
                for k, col in kinds:
                    g = d[kind == k]
                    ax.scatter(g.ai, g.useful / (g.dur_ns / 1e9) / 1e12, s=16, marker=mk, alpha=0.55, color=col,
                               edgecolors="none")
            ax.scatter([], [], marker=mk, color=MUTED, label=c["short"])
        for k, col in kinds:
            ax.scatter([], [], marker="o", color=col, label=k)
        x = np.logspace(-1, 4, 200)
        ax.plot(x, np.minimum(te, x * bw) / 1e12, color=INK, lw=1.5, label="roofline (BF16-eq. TensorE, HBM)")
        ax.axvline(te / bw, color=MUTED, ls=":", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1, 1e4); ax.set_ylim(1e-2, 300)
        ax.set_xlabel("arithmetic intensity (useful BF16-eq. FLOP / HBM byte)")
        ax.set_title(f"{phase}: every block / launch (rank 0)", color=INK, loc="left")
        ax.legend(fontsize=7, loc="lower right", frameon=False)
    axes[0].set_ylabel("achieved useful TFLOP/s (BF16-equivalent)")
    fig.suptitle(f"DeepSeek-V4.1-Flash tp64 opt10 on trn3, one LNC=2 core — ridge {te / bw:.0f} FLOP/B", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("wrote", out)


def timeline(run, step, out, t0_ms=None, t1_ms=None, title=None, smooth_us=50):
    T, cc, meta = tl_load(run, [step])
    U, ex, _ = tables6.load(run)
    e = ex[ex.step == step].sort_values("t0")
    nxt = ex[ex.step == step + 1]
    s0 = e.t0.min()
    s1 = nxt.t0.min() if len(nxt) else e.t1.max()
    T = T.sort_values("bs")
    T["x"] = (T.bs - s0) / 1e6
    if (s1 - s0) > 100e6:      # a prefill iteration: smooth more (the classification still uses 10 us bins)
        smooth_us = 250
    pk_mfu, pk_mbu = T.mfu.max(), T.mbu.max()
    low = (T.mfu <= 0.5 * pk_mfu) & (T.mbu <= 0.5 * pk_mbu)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    # MFU / MBU, smoothed to `smooth_us` for legibility (the classification uses the 10 us bins)
    k = max(1, int(smooth_us // 10))
    for col, color, lab in (("mfu", BLUE, "MFU"), ("mbu", ORANGE, "MBU")):
        y = T[col].rolling(k, center=True, min_periods=1).mean() * 100
        a1.plot(T.x, y, color=color, lw=1.0, label=f"{lab} ({smooth_us:.0f} µs mean)")
    a1.set_ylabel("% of peak hardware rate")
    a1.set_ylim(0, 100)
    a1.legend(loc="upper right", frameon=False, ncol=2)
    # launches
    for r in e.itertuples():
        x0, x1 = (r.t0 - s0) / 1e6, (r.t1 - s0) / 1e6
        if x1 - x0 < 0.003 * (s1 - s0) / 1e6:
            continue
        a1.axvspan(x0, x1, color=GRID, alpha=0.25, lw=0)
        nm = r.name.replace("layers_decode_", "").replace("_tp64", "").replace("embed_", "emb+").replace("_head", "+head")
        nm = nm.replace("draft_decode", "drafter").replace("drafter_batched", "drafter").replace("draft_prefill", "drafter")
        nm = {"embedding + hc_expand": "emb", "tail all-gather graph": "head"}.get(nm, nm)
        if r.role in ("glue", "prologue") or nm in ("emb", "head"):
            continue
        if r.role == "layer":
            if r.kernel and "idx" not in r.kernel and "_e_" not in r.kernel and (r.layer or 0) % 5:
                continue        # label the index / engram blocks and every 5th block only
            nm = f"L{int(r.layer):02d}"
        a1.text((x0 + x1) / 2, 101, nm, ha="center", va="bottom", fontsize=7, color=INK2, clip_on=False)
    # cause lanes
    labels = [l for l, _ in LANES]
    for i, (lab, causes) in enumerate(LANES):
        if causes == ["__idle__"]:
            prev = None
            for r in e.itertuples():
                if prev is not None and r.t0 > prev:
                    a2.broken_barh([((prev - s0) / 1e6, (r.t0 - prev) / 1e6)], (i - 0.35, 0.7), color=MUTED)
                prev = r.t1
            if s1 > prev:
                a2.broken_barh([((prev - s0) / 1e6, (s1 - prev) / 1e6)], (i - 0.35, 0.7), color=MUTED)
            continue
        m = low & T.cause.isin(causes)
        xs = T.x[m].to_numpy()
        w = (T.blen[m] / 1e6).to_numpy()
        a2.broken_barh(list(zip(xs, w)), (i - 0.35, 0.7), color=BLUE if i < 3 else (ORANGE if i < 7 else AQUA))
    a2.set_yticks(range(len(labels)))
    a2.set_yticklabels(labels, fontsize=8)
    a2.invert_yaxis()
    a2.grid(axis="y", visible=False)
    a2.set_xlabel("time in the step (ms)")
    a2.set_title("under-utilised 10 µs bins (MFU and MBU both < ½ of their peak), by what the core is blocked on",
                 fontsize=8, color=INK2, loc="left")
    if t0_ms is not None:
        a2.set_xlim(t0_ms, t1_ms)
    else:
        a2.set_xlim(0, (s1 - s0) / 1e6)
    fig.suptitle(title or f"{run} step {step}", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("args", nargs="+")
    ap.add_argument("--t0-ms", type=float, default=None)
    ap.add_argument("--t1-ms", type=float, default=None)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    if a.what == "roofline":
        roofline(a.args[0])
    else:
        timeline(a.args[0], int(a.args[1]), a.args[2], a.t0_ms, a.t1_ms, a.title)
