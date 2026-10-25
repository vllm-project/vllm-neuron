#!/usr/bin/env python3
"""Assemble the profiling report: hand-written text (report_parts/*.md) + generated tables.

  build_report.py <out.md>
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import tables
from hotspots import hotspots

ROOT = "/mnt/nvme/ds41_profiling"
PARTS = f"{ROOT}/report_parts"

# (title, phase, latency run, latency steps, roofline [(run, steps)], client ms, note)
CONFIGS = json.load(open(f"{PARTS}/configs.json"))


def hot_md(run, block, steps, top=8):
    p, n, ms = hotspots(run, block, steps)
    eng = [c for c in ("Tensor", "Vector", "Scalar", "GpSimd", "Sync") if c in p.columns]
    out = [f"*{block}* — {n} executions, mean {ms:.3f} ms. *Footprint* = time during which any engine of either core "
           f"executes the function (functions overlap, so footprints do not add up to the block); the engine columns are "
           f"each engine's busy time on that function, per physical core.",
           "", "| Kernel function | footprint ms | " + " | ".join(f"{e} ms" for e in eng) + " | HBM MB | GFLOP (BF16-eq.) |",
           "|---|---:|" + "---:|" * (len(eng) + 2)]
    for f, r in p.head(top).iterrows():
        out.append(f"| `{f}` | {r['any-engine wall ms']:.3f} | " + " | ".join(f"{r[e]:.3f}" for e in eng)
                   + f" | {r['HBM MB']:.1f} | {r['GFLOP(bf16-eq)']:.2f} |")
    return "\n".join(out)


def main(out_path):
    text = []
    text.append(open(f"{PARTS}/00_header.md").read())
    text.append(open(f"{PARTS}/01_method.md").read())
    text.append("\n## Results per configuration\n")
    text.append(open(f"{PARTS}/02_results_intro.md").read() if os.path.exists(f"{PARTS}/02_results_intro.md") else "")
    data = {}
    for c in CONFIGS:
        md, d, meta = tables.phase_tables(c["lat_run"], [tuple(x) for x in c["roof"]], c["phase"], c["steps"], c.get("client_ms"))
        data[c["id"]] = d
        text.append(f"\n### {c['title']}\n")
        if c.get("note"):
            text.append(c["note"] + "\n")
        text.append("**1. Overall latency breakdown** (descending)\n")
        text.append(md["breakdown"] + "\n")
        text.append("Transformer blocks by archetype (descending):\n")
        text.append(md["arch"] + "\n")
        text.append("**2–3. Per-block latency and roofline** (descending by latency)\n")
        text.append(md["blocks"] + "\n")
        if c.get("hot"):
            text.append("**Where the time goes inside the dominant blocks** (NKI source attribution)\n")
            for blk in c["hot"]:
                text.append(hot_md(c["roof"][0][0], blk, c["roof"][0][1]) + "\n")
    for extra in ("80_analysis.md",):
        if os.path.exists(f"{PARTS}/{extra}"):
            body = open(f"{PARTS}/{extra}").read()
            for key, fname in (("{{TABLES}}", "80_tables.md"), ("{{SPILL}}", "80_spill.md"), ("{{CHUNKS}}", "80_chunks.md")):
                if key in body:
                    body = body.replace(key, open(f"{PARTS}/{fname}").read())
            text.append(body)
    tail = f"{PARTS}/90_tail.md"
    if os.path.exists(tail):
        text.append(open(tail).read())
    open(out_path, "w").write("\n".join(text))
    json.dump(data, open(f"{ROOT}/analysis/report_data.json", "w"), default=float, indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1])
