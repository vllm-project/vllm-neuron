# SPDX-License-Identifier: Apache-2.0
"""Compare two `DEEPSEEK_V41_DUMP_DIR` directories step by step and layer by layer.

    python compare_dumps.py /mnt/nvme/dump_op /mnt/nvme/dump_mk [--steps 1,2]
"""

import argparse
import os
import re
import sys

import torch


def metrics(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    if a.numel() != b.numel():
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    err = (a - b).norm() / max(float(b.norm()), 1e-30)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0) if b.norm() > 0 else torch.tensor(1.0)
    return f"rel_l2={float(err):.3e} cos={float(cos):.6f} max_abs={float((a - b).abs().max()):.4g} nan={bool(torch.isnan(a).any())}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("got")
    ap.add_argument("--steps", default="1")
    args = ap.parse_args()
    for step in [int(s) for s in args.steps.split(",")]:
        names = sorted(
            {f[len(f"step{step}_"):-3] for f in os.listdir(args.ref) if f.startswith(f"step{step}_")}
            & {f[len(f"step{step}_"):-3] for f in os.listdir(args.got) if f.startswith(f"step{step}_")},
            key=lambda n: (re.sub(r"\d+", "", n), int(re.search(r"\d+", n).group()) if re.search(r"\d+", n) else -1),
        )
        print(f"== step {step}: {len(names)} common dumps")
        for name in names:
            a = torch.load(os.path.join(args.ref, f"step{step}_{name}.pt"))
            b = torch.load(os.path.join(args.got, f"step{step}_{name}.pt"))
            if a.dtype in (torch.int32, torch.int64):
                same = (a.reshape(-1) == b.reshape(-1)).float().mean() if a.numel() == b.numel() else float("nan")
                print(f"  {name:>16}: ids equal {float(same):.3f}")
            else:
                print(f"  {name:>16}: {metrics(b, a)}")


if __name__ == "__main__":
    main()
