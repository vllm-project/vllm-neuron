# SPDX-License-Identifier: Apache-2.0
"""Trace every kind of specialized megakernel entry (no device): the block kernel per
phase, the drafter per phase (per-sequence and batched).

    python compile_entries.py
"""
import os
import sys
import time

os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "8")

from kharness import compile_only  # noqa: E402
from deepseek_v41_8d.kernels import mk_kernels  # noqa: E402
import compile_layer_args as CL  # noqa: E402


def trace(kernel, *args, grid=2):
    t0 = time.time()
    r = compile_only(kernel, *args, grid=grid)
    print(f"  {kernel.func.__name__}: traced in {time.time() - t0:.1f}s, {len(r.return_types)} outputs", flush=True)


def main():
    # block kernel: prefill / decode, plain and engram archetypes
    for phase, T in (("prefill", 128), ("decode", 6)):
        for arch, eng in (((0, 0, 0, 0, 0), False), ((0, 0, 0, 1, 0), True)):
            args = CL.layer_args(T=T, engram=eng, stage=9)
            trace(mk_kernels.layer(phase, arch, "tp32"), *args)
    args = CL.layer_args(T=128, ratio=2, src=True, stage=9)
    trace(mk_kernels.layer("prefill", (2, 1, 1, 0, 0), "tp32"), *args)
    # drafter: per sequence (`draft`) and batched (the generic body, `draft_batched_generic`)
    for phase in ("decode", "prefill"):
        at_zero = phase == "prefill"
        trace(mk_kernels.draft(phase, "tp32"), *CL.draft_args(at_zero=at_zero, scheme="tp32"))
        trace(mk_kernels.draft_batched_generic(phase), *CL.draft_args(at_zero=at_zero, scheme="batched"))
    print("ALL TRACED", flush=True)


if __name__ == "__main__":
    sys.exit(main())
