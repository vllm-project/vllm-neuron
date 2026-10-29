#!/usr/bin/env python3
"""Assemble PROFILE_REPORT_tp64_opt6.md: hand-written parts (report_parts/*.md) + generated tables.

  build6.py <out.md>
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import tables6, util6
from hot6 import hotspots

ROOT = "/mnt/nvme/ds41_profiling_opt6"
PARTS = f"{ROOT}/report_parts"
CONFIGS = json.load(open(f"{PARTS}/configs.json"))
FIGREL = "profiling_tp64_opt6"


def hot_md(run, block, steps, top=8):
    p, n, ms = hotspots(run, block, steps)
    eng = [c for c in ("Tensor", "Vector", "Scalar", "GpSimd", "Sync") if c in p.columns]
    label = {"draft": "DSpark drafter"}.get(block, block)
    out = [f"*{label}* — {n} executions, mean {ms:.3f} ms. *Footprint* = time during which any engine of either core "
           f"executes the function (functions overlap, so footprints do not add up to the block); the engine columns are "
           f"each engine's busy time on that function, per physical core.",
           "", "| Kernel function | footprint ms | " + " | ".join(f"{e} ms" for e in eng) + " | HBM MB | GFLOP (BF16-eq.) |",
           "|---|---:|" + "---:|" * (len(eng) + 2)]
    for f, r in p.head(top).iterrows():
        out.append(f"| `{f}` | {r['any-engine wall ms']:.3f} | " + " | ".join(f"{r[e]:.3f}" for e in eng)
                   + f" | {r['HBM MB']:.1f} | {r['GFLOP(bf16-eq)']:.2f} |")
    return "\n".join(out)


def part(name):
    p = f"{PARTS}/{name}"
    return open(p).read() if os.path.exists(p) else ""


def main(out_path):
    text = [part("00_header.md").replace("{{SUMMARY}}", part("05_summary_table.md").strip())
            .replace("{{CAUSES}}", part("06_cause_matrix.md").strip())
            .replace("{{PEAKS}}", part("07_peak_matrix.md").strip()), part("01_method.md"), "\n## Results per configuration\n", part("02_results_intro.md")]
    data = {}
    for c in CONFIGS:
        md, d, meta = tables6.phase_tables(c["lat_run"], [tuple(x) for x in c["roof"]], c["phase"], c["steps"],
                                           c.get("client_ms"))
        data[c["id"]] = d
        text.append(f"\n### {c['title']}\n")
        if c.get("note"):
            text.append(c["note"] + "\n")
        text.append(part(f"10_{c['id']}_lead.md"))
        text.append("**1. Overall latency breakdown** (descending)\n")
        text.append(md["breakdown"] + "\n")
        text.append("Transformer blocks by archetype (descending):\n")
        text.append(md["arch"] + "\n")
        text.append("**2–3. Per-block latency and roofline** (descending by latency)\n")
        text.append(md["blocks"] + "\n")
        if c.get("hot"):
            text.append("**Where the time goes inside the dominant blocks** (NKI source attribution)\n")
            run, steps = c["roof"][0]
            for blk in c["hot"]:
                text.append(hot_md(run, blk, steps) + "\n")
        # ---- 4-5: peak utilisation and what blocks the core
        run, steps = c["roof"][0]
        text.append("**4. Peak MFU / MBU per megakernel, and what the core is blocked on when it is under-utilised**\n")
        text.append(part("11_peak_legend.md"))
        how = "launch"
        md4, R, C, R2, C2, T = util6.megakernel_table(run, steps, c["phase"], how)
        text.append(md4 + "\n")
        if c["phase"] == "decode":
            text.append("The same, by block kind inside the fused launches (members):\n")
            md4b, *_ = util6.megakernel_table(run, steps, c["phase"], "member")
            text.append(md4b + "\n")
        b = pd.Series(d["breakdown_ms"])
        idle = sum(v for k, v in b.items() if k.startswith("device idle"))
        # (prefill: the roofline run may be a 4th-chunk capture, or cover fewer iterations than the
        # latency run; the device idle is added only where both cover the same span)
        with_idle = c["phase"] == "decode" or (run == c["lat_run"] and len(c["steps"]) == 1)
        base = ("step" if c["phase"] == "decode" else
                "profiled 4th chunk (device time)" if run != c["lat_run"] else
                "profiled iteration" if with_idle else
                f"{len(steps)} device-profiled iterations (device time)")
        text.append(f"**5. Overall: the largest blocking items over the {base}** (rank 0's device timeline"
                    f"{' plus the time with no launch on the core' if with_idle else ''})\n")
        text.append(util6.overall_causes(R2, C2, steps, c["phase"], idle_ms=idle if with_idle else None,
                                         outside_ms=None, base=base) + "\n")
        fig = f"{FIGREL}/tl_{c['id']}.png"
        if os.path.exists(f"{ROOT}/analysis/fig/tl_{c['id']}.png"):
            text.append(f"![utilisation timeline: {c['short']}]({fig})\n")
        text.append(part(f"12_{c['id']}_analysis.md"))
    text.append(part("80_analysis.md").replace("{{TABLES}}", part("80_tables.md")).replace("{{SPILL}}", part("80_spill.md"))
                .replace("{{SKEW}}", part("80_skew.md")).replace("{{CHUNKS}}", part("80_chunks.md")))
    text.append(part("90_tail.md"))
    open(out_path, "w").write("\n".join(t for t in text if t is not None))
    json.dump(data, open(f"{ROOT}/analysis/report_data.json", "w"), default=float, indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1])
