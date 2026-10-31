# SPDX-License-Identifier: Apache-2.0
"""Does an LNC=2 kernel hang when its two cores run a dynamic loop with different
trip counts?  (The 0.6 dynamic-loop migration guide: "For LNC2 kernels, confirm
both cores iterate the same count.")

    KH_CORE=2 NEURON_RT_EXEC_TIMEOUT=90 python probe_lnc.py

Core `pid` loops `cnt[pid]` times, each iteration adding 1.0 into its own row of
the output; an execution timeout means the hang is real.
"""

import time
import traceback

from kharness import run_nki  # noqa: E402 - sets the environment first
import torch

import nki
import nki.isa as nisa
import nki.language as nl

PT = 128


@nki.jit
def k_lnc(cnt):
    pid = nl.program_id(0)
    out = nl.ndarray((2, PT), dtype=nl.float32, buffer=nl.shared_hbm)
    acc = nl.ndarray((2, PT), dtype=nl.float32, buffer=nl.shared_hbm, name="acc")
    z = nl.ndarray((1, PT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(z, 0.0)
    nisa.dma_copy(dst=acc[pid : pid + 1, :], src=z)
    c = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c, src=cnt[pid : pid + 1, 0:1])
    cs = nisa.scalar_load(c)

    def body(i):
        t = nl.ndarray((1, PT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=acc[pid : pid + 1, :])
        nisa.tensor_scalar(dst=t, data=t, op0=nl.add, operand0=1.0)
        nisa.dma_copy(dst=acc[pid : pid + 1, :], src=t)

    nl.fori_loop(0, cs, body)
    nisa.core_barrier(data=acc, cores=(0, 1))
    t2 = nl.ndarray((2, PT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t2, src=acc)
    nisa.dma_copy(dst=out, src=t2)
    return out


def main():
    for cnt in ([3, 3], [3, 0], [0, 5], [7, 2]):
        t0 = time.time()
        try:
            got = run_nki(k_lnc, torch.tensor(cnt, dtype=torch.int32).reshape(2, 1))
            print(f"LNC counts {cnt}: core sums {got[:, 0].tolist()} in {time.time() - t0:.1f}s -> "
                  f"{'OK' if got[:, 0].tolist() == [float(cnt[0]), float(cnt[1])] else 'WRONG'}", flush=True)
        except Exception as e:
            print(f"LNC counts {cnt}: FAILED after {time.time() - t0:.1f}s: {str(e)[:300]}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
