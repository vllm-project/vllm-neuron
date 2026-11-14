# SPDX-License-Identifier: Apache-2.0
"""Probe the GpSimd top-k (`nisa.topk`) the index selection relies on, on Trn2.

    python probe_topk.py

Eight rows of `n` bf16 scores, laid out as eight 16-partition snakes, the top
`k` of each against `torch.topk` on the same bf16 values.
"""

import traceback

from kharness import run_nki  # noqa: E402 - sets the environment first
import torch

import nki
import nki.isa as nisa
import nki.language as nl

PT = 128


@nki.jit
def k_topk(sc, flags):
    rows, n = sc.shape
    (k,) = flags
    n16 = n // 16
    k16 = -(-k // 16)
    out = nl.ndarray((rows, k), dtype=nl.int32, buffer=nl.shared_hbm)
    vals_out = nl.ndarray((rows, k), dtype=nl.float32, buffer=nl.shared_hbm)
    snake = nl.ndarray((PT, n16), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(snake, -1.0e30)
    for q in range(rows):
        nisa.dma_copy(dst=snake[16 * q : 16 * q + 16, 0:n16],
                      src=sc.ap(pattern=[[1, 16], [16, n16]], offset=q * n))
    vals = nl.ndarray((PT, k), dtype=nl.bfloat16, buffer=nl.sbuf)
    idx = nl.ndarray((PT, k), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.topk(val_dst=vals, idx_dst=idx, src=snake, n=n, sorted=False)
    vf = nl.ndarray((PT, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vf, src=vals)
    for q in range(rows):
        nisa.dma_copy(dst=out.ap(pattern=[[1, 16], [16, k16]], offset=q * k),
                      src=idx.view(nl.int32)[16 * q : 16 * q + 16, 0:k16])
        nisa.dma_copy(dst=vals_out.ap(pattern=[[1, 16], [16, k16]], offset=q * k),
                      src=vf[16 * q : 16 * q + 16, 0:k16])
    return out, vals_out


def main():
    ok = True
    torch.manual_seed(0)
    for n, k in ((1024, 512), (4224, 512), (8448, 512)):
        try:
            sc = torch.randn(8, n).to(torch.bfloat16)
            idx, vals = run_nki(k_topk, sc, (k,))
            for r in range(8):
                want = set(torch.topk(sc[r].float(), k).indices.tolist())
                got = set(idx[r].tolist())
                thr = torch.topk(sc[r].float(), k).values.min()
                # ties at the threshold may pick either member
                extra = [j for j in got - want if sc[r, j].float() != thr]
                good = len(got) == k and not extra and all(0 <= j < n for j in got)
                if not good:
                    print(f"n={n} row {r}: {len(got & want)} of {k} agree, extra={extra[:5]}")
                ok &= good
            print(f"n={n} k={k}: {'OK' if ok else 'WRONG'}", flush=True)
        except Exception:
            traceback.print_exc()
            ok = False
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
