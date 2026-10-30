#!/usr/bin/env python3
"""All figures of the report -> analysis/fig/ (roofline + one utilisation timeline per configuration)."""
import json, os, subprocess, sys
ROOT = "/mnt/nvme/ds41_profiling_opt10"
PY = sys.executable
F = f"{ROOT}/analysis/fig"
os.makedirs(F, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([PY, f"{here}/plot6.py", "roofline", f"{F}/roofline_tp64_opt10.png"], check=True)
for c in json.load(open(f"{ROOT}/report_parts/configs.json")):
    if c["phase"] == "decode":
        run, steps = c["roof"][0]
        st = steps[len(steps) // 2]
        title = f"{c['short']} — decode step {st} (rank 0), {len(steps)} steady steps in the window"
    else:
        run, steps = c["roof"][0]
        st = steps[0]
        what = (f"prefill chunk {st + 1} (positions {2048 * st}–{2048 * st + 2047})" if c["id"].startswith("in8192")
                else "the prefill iteration" if len(c["steps"]) == 1 else f"prefill iteration {st + 1} of {len(c['steps'])}")
        title = f"{c['short']} — {what} (rank 0)"
    subprocess.run([PY, f"{here}/plot6.py", "timeline", run, str(st), f"{F}/tl_{c['id']}.png", "--title", title], check=True)
