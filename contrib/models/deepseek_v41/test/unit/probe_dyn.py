# SPDX-License-Identifier: Apache-2.0
"""Probe the data-dependent MoE machinery on the device.

    KH_CORE=8 python probe_dyn.py

B1  nonzero_with_count over per-expert mask rows placed on partitions 16e.
B2  a register-bounded fori_loop (including zero trips) whose body gathers rows
    by an index list, runs a bf16 matmul and scatter-adds (dma_compute) into an
    fp32 HBM accumulator through unique, padded indices.
"""

import traceback

from kharness import compare, run_nki  # noqa: E402 - sets the environment first
import torch

import nki
import nki.isa as nisa
import nki.language as nl

PT = 128


@nki.jit
def k_nonzero(wexp):
    """B1: wexp [Tp, E] fp32 -> glist [E, Tp + 1] int32 (indices, -1 pad, count)."""
    Tp, E = wexp.shape
    nt = Tp // PT
    glist = nl.ndarray((E, Tp + PT), dtype=nl.int32, buffer=nl.shared_hbm)
    maskT = nl.ndarray((PT, Tp), dtype=nl.float32, buffer=nl.sbuf)
    for i in range(nt):
        w = nl.ndarray((PT, E), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w, src=wexp[i * PT : (i + 1) * PT, 0:E])
        wide = nl.ndarray((PT, PT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(wide, 0.0)
        for e in range(E):
            nisa.tensor_scalar(dst=wide[:, 16 * e : 16 * e + 1], data=w[:, e : e + 1], op0=nl.greater, operand0=0.0)
        tp = nl.ndarray((PT, PT), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=wide)
        nisa.tensor_copy(dst=maskT[:, i * PT : (i + 1) * PT], src=tp)
    nz = nl.ndarray((PT, Tp + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(nz, -7)
    nisa.nonzero_with_count(dst=nz, src=maskT, index_offset=0, padding_val=-1)
    for e in range(E):
        nisa.dma_copy(dst=glist[e : e + 1, 0 : Tp + 1], src=nz[16 * e : 16 * e + 1, :])
    return glist


@nki.jit
def k_dynloop(x, glist, cnt, w, flags):
    """B2: for each expert e: for tiles i < ceil(cnt[e]/128): rows = x[glist[e, i*128 + q]],
    y = rows @ w.T (bf16), acc[rows] += y. Padded slots gather zero rows into junk rows."""
    E, Tp = flags
    Tpad = Tp + PT
    n_rows, k = x.shape      # x has Tpad rows: the last 128 are zero
    n = w.shape[0]
    ntmax = Tp // PT
    acc = nl.ndarray((Tpad, n), dtype=nl.float32, buffer=nl.shared_hbm)
    z = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(z, 0.0)
    for i in range(Tpad // PT):
        nisa.dma_copy(dst=acc[i * PT : (i + 1) * PT, :], src=z)
    # w^T resident (k on partitions)
    wT = nl.ndarray((PT, k // PT, n), dtype=nl.bfloat16, buffer=nl.sbuf)
    for kb in range(k // PT):
        nisa.dma_transpose(dst=wT[:, kb, :], src=w[0:n, kb * PT : (kb + 1) * PT])
    ramp = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=ramp, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    rampf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rampf, src=ramp)
    g3 = glist.reshape((E, ntmax + 1, PT))
    for e in range(E):
        c = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=c, src=cnt[e : e + 1, 0:1])
        # ntiles = (c + 127) // 128 on the scalar unit
        cs = nisa.scalar_load(c)
        nt_s = (cs + (PT - 1)) // PT

        def body(i, e=e):
            gi = nl.ndarray((PT, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=gi, src=g3.ap(pattern=[[1, PT], [1, 1]], offset=e * (ntmax + 1) * PT,
                                           scalar_offset=i, indirect_dim=1))
            # padded (-1) -> Tp + q, unique within the tile
            gf = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gf, src=gi)
            neg = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=neg, data=gf, op0=nl.less, operand0=0.0)
            fix = nl.ndarray((PT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=fix, data=rampf, op0=nl.add, operand0=float(Tp + 1))
            nisa.tensor_tensor(dst=fix, data1=fix, data2=neg, op=nl.multiply)
            nisa.tensor_tensor(dst=gf, data1=gf, data2=fix, op=nl.add)
            nisa.tensor_copy(dst=gi, src=gf)
            rows = nl.ndarray((PT, k), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=rows, src=x.ap(pattern=[[k, PT], [1, k]], vector_offset=gi, indirect_dim=0))
            ps = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.psum)
            for kb in range(k // PT):
                tp = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tp, data=rows[:, kb * PT : (kb + 1) * PT])
                xT = nl.ndarray((PT, PT), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=xT, src=tp)
                nisa.nc_matmul(dst=ps, stationary=xT, moving=wT[:, kb, :], accumulate=(kb > 0))
            y = nl.ndarray((PT, n), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=y, src=ps)
            tgt = acc.ap(pattern=[[n, PT], [1, n]], vector_offset=gi, indirect_dim=0)
            nisa.dma_compute(dst=tgt, srcs=[tgt, y], reduce_op=nl.add)

        nl.fori_loop(0, nt_s, body)
    return acc


def main():
    ok = True
    torch.manual_seed(0)
    Tp, E = 512, 6
    wexp = torch.zeros(Tp, E)
    hits = torch.rand(Tp, E) < 0.08
    wexp[hits] = torch.rand(int(hits.sum())) + 0.5
    wexp[:, 5] = 0.0                     # an expert with no rows
    wexp[:, 0] = 1.0                     # an expert with every row (4 tiles)
    wexp[300:, 1] = 0.0
    try:
        glist = run_nki(k_nonzero, wexp)
        good = True
        for e in range(E):
            idx = torch.nonzero(wexp[:, e] > 0).reshape(-1)
            c = int(glist[e, Tp])
            got = glist[e, :c]
            good &= c == idx.numel() and torch.equal(got.to(torch.int64), idx) and bool((glist[e, c:Tp] == -1).all())
            print(f"B1 expert {e}: count {c} (want {idx.numel()}) first {glist[e, :4].tolist()}")
        print("B1 nonzero_with_count:", "OK" if good else "WRONG")
        ok &= good
    except Exception:
        traceback.print_exc()
        ok = False
        return
    try:
        k, n = 256, 384
        x = torch.randn(Tp + PT, k).to(torch.bfloat16)
        x[Tp:] = 0
        w = torch.randn(n, k).to(torch.bfloat16)
        cnt = glist[:, Tp].reshape(E, 1).contiguous()
        got = run_nki(k_dynloop, x, glist, cnt, w, (E, Tp))
        ref = torch.zeros(Tp, n)
        for e in range(E):
            idx = torch.nonzero(wexp[:, e] > 0).reshape(-1)
            if idx.numel():
                ref.index_add_(0, idx, x[idx].float() @ w.float().t())
        ok &= compare("B2 dynamic expert loop (live rows)", got[:Tp], ref, tol_rel=2e-3)
        print("B2 junk rows finite:", bool(torch.isfinite(got[Tp:]).all()))
    except Exception:
        traceback.print_exc()
        ok = False
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
