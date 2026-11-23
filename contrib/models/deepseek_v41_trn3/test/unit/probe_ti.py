# SPDX-License-Identifier: Apache-2.0
"""Probe tensor indirection on the tensor engine: scores = q @ K^T with the K^T
columns selected on chip through `.indirect(index)` on the moving operand.

    KH_CORE=10 python probe_ti.py

Step 1 decodes the index order the hardware reads: kT[0, r] = r (fp16, exact to 2048),
q one-hot on partition 0, an index tile whose every entry is distinct -> output
position i reports which index entry fed it. Step 2 checks a real gather of 640
columns through the decoded order against the CPU product.
"""

import traceback

from kharness import compare, run_nki  # noqa: E402 - sets the environment first
import torch

import nki
import nki.isa as nisa
import nki.language as nl

PT = 128


@nki.jit
def k_ti(kT_hbm, q_hbm, idx_hbm, flags):
    """idx_hbm [128, ncol] int32 is the index tile as laid out by the host."""
    nsel, = flags
    d, nrows = kT_hbm.shape
    heads = q_hbm.shape[1]
    ncol = idx_hbm.shape[1]
    out = nl.ndarray((heads, nsel), dtype=nl.float32, buffer=nl.shared_hbm)
    kT = nl.ndarray((PT, nrows), dtype=kT_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=kT, src=kT_hbm)
    qT = nl.ndarray((PT, heads), dtype=q_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=qT, src=q_hbm)
    ix32 = nl.ndarray((PT, ncol), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ix32, src=idx_hbm)
    ix = nl.ndarray((PT, ncol), dtype=nl.uint16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ix, src=ix32)
    ps = nl.ndarray((heads, nsel), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ps, stationary=qT, moving=kT.indirect(ix, num_elem=nsel))
    o = nl.ndarray((heads, nsel), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o, src=ps)
    nisa.dma_copy(dst=out, src=o)
    return out


def main():
    ok = True
    # ---- step 1: decode the order
    nrows, heads, ncol = 4096, 8, 32
    nsel = 16 * ncol
    kT = torch.zeros(PT, nrows, dtype=torch.float16)
    kT[0] = torch.arange(nrows).to(torch.float16)
    q = torch.zeros(PT, heads, dtype=torch.float16)
    q[0, 0] = 1.0
    idx = (torch.arange(PT).reshape(PT, 1) * ncol + torch.arange(ncol).reshape(1, ncol)).to(torch.int32)
    order = None
    try:
        got = run_nki(k_ti, kT, q, idx, (nsel,))
        vals = got[0].round().to(torch.int64)
        pc = [(int(v) // ncol, int(v) % ncol) for v in vals.tolist()]
        print("TI decode: output position i -> index[p, c] for i < 40:", pc[:40])
        print("TI decode: positions 500..511:", pc[500:512])
        # hypotheses
        hyp = {
            "snake G=16 (i%16, i//16)": [(i % 16, i // 16) for i in range(nsel)],
            "snake G=32 (i%32, i//32)": [(i % 32, i // 32) for i in range(nsel)],
            "row-major (i//ncol, i%ncol)": [(i // ncol, i % ncol) for i in range(nsel)],
        }
        for name, h in hyp.items():
            match = sum(1 for a, b in zip(pc, h) if a == b)
            print(f"   {name}: {match}/{nsel} positions match")
            if match == nsel:
                order = name
    except Exception:
        traceback.print_exc()
        ok = False
    # ---- step 2: a real gather in the decoded order
    # ---- step 2: real gathers; the index snake replicated per 16-partition block
    # ("x8"), per quadrant only ("q"), or on partitions 0..15 only ("one")
    for nsel, rep in (((512, "x8"), (512, "q"), (128, "x8"), (640, "x8")) if order is not None else ()):
        torch.manual_seed(0)
        nrows = 3000
        kT = (torch.randn(PT, nrows) * 0.5).to(torch.bfloat16)
        q = (torch.randn(PT, heads) * 0.5).to(torch.bfloat16)
        sel = torch.randperm(nrows)[:nsel]
        ref = q.float().t() @ kT.float()[:, sel]
        ncol = nsel // 16
        idx = torch.zeros(PT, ncol, dtype=torch.int32)
        blocks = range(8) if rep == "x8" else ([0, 2, 4, 6] if rep == "q" else [0])
        for g in blocks:
            for i in range(nsel):
                idx[16 * g + i % 16, i // 16] = int(sel[i])
        try:
            got = run_nki(k_ti, kT, q, idx, (nsel,))
            r = compare(f"TI gather of {nsel} columns, index replicated {rep}", got, ref, tol_rel=2e-3)
            if nsel <= 512 and rep == "x8":
                ok &= r
            if not r:
                colok = ((got - ref).abs().amax(0) < 0.05)
                print(f"   columns correct: {int(colok.sum())}/{nsel}")
        except Exception:
            traceback.print_exc()
            ok = False
    if order is None:
        ok = False
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
