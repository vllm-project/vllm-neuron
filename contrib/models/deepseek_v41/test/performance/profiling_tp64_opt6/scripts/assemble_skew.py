#!/usr/bin/env python3
"""report_parts/80_skew.md from the per-capture skew_report outputs."""
import os
P = "/mnt/nvme/ds41_profiling_opt6/report_parts"
order = [("in1024_bs1_decode", "in1024 bs1 · decode"), ("in8192_bs1_decode", "in8192 bs1 · decode"),
         ("in1024_bs10_decode", "in1024 bs10 · decode"), ("in8192_bs10_decode", "in8192 bs10 · decode"),
         ("in1024_bs1_prefill", "in1024 bs1 · prefill (one iteration)")]
out = []
for k, title in order:
    f = f"{P}/80_skew_skew_{k}.md"
    if os.path.exists(f):
        out.append(f"### {title}\n\n" + open(f).read().strip() + "\n")
open(f"{P}/80_skew.md", "w").write("\n".join(out))
print("\n".join(o.splitlines()[0] for o in out))
