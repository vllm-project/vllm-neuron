#!/usr/bin/env python3
"""Roofline plot: every block execution of every configuration (BF16-equivalent FLOPs vs HBM bytes)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import tables

ROOT = "/mnt/nvme/ds41_profiling"
CONFIGS = json.load(open(f"{ROOT}/report_parts/configs.json"))


def main(out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    te, bw = None, None
    colors = {"layer": "#1f77b4", "draft kernel": "#d62728", "head+sampler": "#2ca02c", "embed+hc_expand": "#9467bd"}
    for ax, phase in zip(axes, ("prefill", "decode")):
        markers = iter(["o", "s", "^", "D"])
        for c in [c for c in CONFIGS if c["phase"] == phase]:
            m = next(markers)
            for run, steps in c["roof"]:
                ex, meta = tables.load(run)
                te, bw = meta["te_peak_bf16_flops"], meta["hbm_bw"]
                d = tables.roof_metrics(ex[ex.n_inst.notna() & ex.step.isin(steps)], meta)
                d = d[d.role.isin(colors) & (d.hbm > 0)]
                for role, g in d.groupby("role"):
                    ax.scatter(g.ai, g.useful / (g.dev_ns / 1e9) / 1e12, s=14, marker=m, alpha=0.6,
                               color=colors[role])
            ax.scatter([], [], marker=m, color="grey", label=c["short"])
        for role, col in colors.items():
            ax.scatter([], [], marker="o", color=col, label=role.replace("layer", "transformer block"))
        x = np.logspace(-1, 4, 200)
        ax.plot(x, np.minimum(te, x * bw) / 1e12, "k-", lw=1.5, label="roofline (BF16 TensorE, HBM)")
        ax.axvline(te / bw, color="grey", ls=":", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1, 1e4); ax.set_ylim(1e-2, 300)
        ax.set_xlabel("arithmetic intensity (useful BF16-eq. FLOP / HBM byte)")
        ax.set_title(f"{phase}: every block execution (rank 0)")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=6, loc="lower right")
    axes[0].set_ylabel("achieved useful TFLOP/s (BF16-eq.)")
    fig.suptitle("DeepSeek-V4.1-Flash tp64 on trn3 (one LNC=2 core): ridge %.0f FLOP/B" % (te / bw))
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
