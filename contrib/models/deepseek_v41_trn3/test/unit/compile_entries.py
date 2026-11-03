# SPDX-License-Identifier: Apache-2.0
"""Trace every kind of specialized megakernel entry (no device): the block kernel per
phase and scheme, the drafter per phase, and the pipeline kernels.

    python compile_entries.py
"""
import os
import sys
import time

os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "8")
import torch  # noqa: E402

from kharness import compile_only  # noqa: E402
from deepseek_v41.kernels import mk_kernels  # noqa: E402
import compile_layer_args as CL  # noqa: E402


def trace(kernel, *args, grid=2):
    t0 = time.time()
    r = compile_only(kernel, *args, grid=grid)
    print(f"  {kernel.func.__name__}: traced in {time.time() - t0:.1f}s, {len(r.return_types)} outputs", flush=True)


def main():
    # block kernel: prefill / decode, plain and engram archetypes, both schemes
    for phase, T in (("prefill", 128), ("decode", 6)):
        for arch, eng in (((0, 0, 0, 0, 0), False), ((0, 0, 0, 1, 0), True)):
            args = CL.layer_args(T=T, engram=eng, stage=9)
            trace(mk_kernels.layer(phase, arch, "tp64"), *args)
            trace(mk_kernels.layer(phase, arch, "tp8pp8"), *args)
    args = CL.layer_args(T=128, ratio=2, src=True, stage=9)
    trace(mk_kernels.layer("prefill", (2, 1, 1, 0, 0), "tp64"), *args)
    trace(mk_kernels.layer("prefill", (2, 1, 1, 0, 0), "tp8pp8"), *args)
    # drafter
    for phase in ("decode", "prefill"):
        for scheme in ("tp64", "tp8pp8"):
            trace(mk_kernels.draft(phase, scheme), *CL.draft_args(at_zero=(phase == "prefill"), scheme=scheme))
    # the stage boundary (one program for both sides), both pair layouts
    hc, dim = 4, 5120
    pairs = tuple((r, r + 8) for r in range(8))
    for T, R in ((6, 4), (128, 128), (512, 257)):
        stream = torch.zeros(T, hc, dim, dtype=torch.bfloat16)
        pre_mix = torch.zeros(T, hc)
        topk = torch.zeros(T, 256, dtype=torch.int32)
        cache = torch.zeros(2048, 512, dtype=torch.bfloat16)
        icache = torch.zeros(2048, 128, dtype=torch.bfloat16)
        slots = torch.zeros(R, 1, dtype=torch.int32)
        role = torch.zeros(1, 1)
        for parity in (0, 1):
            trace(mk_kernels.boundary(parity), stream, pre_mix, topk, cache, icache, slots, role, pairs, (0,))
    print("ALL TRACED", flush=True)


if __name__ == "__main__":
    sys.exit(main())
