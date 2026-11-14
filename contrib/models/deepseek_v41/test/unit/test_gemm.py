# SPDX-License-Identifier: Apache-2.0
"""`kernels/mk_gemm` (the megakernels' Trn2 GEMMs) against a CPU reference, both
partitionings, on one logical core (grid 2).

    python test_gemm.py
"""

import torch

from kharness import compare, run_nki  # noqa: E402 - sets the environment first

import nki
import nki.isa as nisa
import nki.language as nl

from deepseek_v41.kernels import mk_gemm as G
from deepseek_v41.kernels.mk_common import tiles_of
import ref_layer as R


@nki.jit
def k_cols(x, wk, flags):
    nr, k = x.shape
    n = wk.shape[1]
    (fp32,) = flags
    pid = nl.program_id(0)
    out = nl.ndarray((nr, n), dtype=nl.float32 if fp32 else nl.bfloat16, buffer=nl.shared_hbm)
    lo, hi = G.half_cols(n, pid)
    G.gemm_cols(x, nr, k, wk, out, lo, hi, qdq=True,
                out_dtype=nl.float32 if fp32 else nl.bfloat16, out_scale=0.5 if fp32 else 1.0)
    return out


@nki.jit
def k_rows(x, wk):
    T, k = x.shape
    n = wk.shape[1]
    pid = nl.program_id(0)
    out = nl.ndarray((T, n), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    G.gemm_rows(x, tiles_of(T, pid), k, wk, out, n, qdq=True)
    return out


def main():
    torch.manual_seed(0)
    ok = True
    for nr, k, n, fp32 in ((6, 5120, 1280, False), (60, 1280, 4096, False), (6, 1024, 5120, True)):
        x = (torch.randn(nr, k) * 2).to(torch.bfloat16)
        w = (torch.randn(n, k) * 0.03).to(torch.bfloat16)
        ref = R.act_quant_t2(x).float() @ w.float().t()
        got = run_nki(k_cols, x, w.t().contiguous(), (int(fp32),), grid=2)
        if fp32:
            ref = ref * 0.5
        else:
            ref = ref.to(torch.bfloat16)
        ok &= compare(f"cols nr={nr} k={k} n={n} fp32={fp32}", got, ref, tol_rel=1e-2 if not fp32 else 1e-4)
    for T, k, n in ((512, 1280, 4096), (1000, 5120, 512)):
        x = (torch.randn(T, k) * 2).to(torch.bfloat16)
        w = (torch.randn(n, k) * 0.03).to(torch.bfloat16)
        ref = (R.act_quant_t2(x).float() @ w.float().t()).to(torch.bfloat16)
        got = run_nki(k_rows, x, w.t().contiguous(), grid=2)
        ok &= compare(f"rows T={T} k={k} n={n}", got, ref, tol_rel=1e-2)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
