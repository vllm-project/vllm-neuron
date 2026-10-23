"""Variant A: exact reference seam. Variant B: both cores write the whole source."""
import os, sys
from kharness_mp import init_rank
rank, world = init_rank()
import torch
import nki, nki.isa as nisa, nki.language as nl
import nki.collectives as ncc
from kharness import run_nki

N, D = 128, 1024
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "A"

@nki.jit
def ar_a(x, groups):
    rg = ncc.ReplicaGroup([list(g) for g in groups])
    n, d = x.shape
    out = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.shared_hbm)
    cc_src = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_dst")
    pid = nl.program_id(0)
    half = n // 2
    r0 = pid * half
    t = nl.ndarray((half, d), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x[r0:r0 + half, :])
    nisa.dma_copy(dst=cc_src[r0:r0 + half, :], src=t)
    nisa.core_barrier(data=cc_src, cores=(0, 1))
    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], replica_group=rg, op=nl.add)
    nisa.dma_copy(dst=out[r0:r0 + half, :], src=cc_dst[r0:r0 + half, :])
    nisa.core_barrier(data=out, cores=(0, 1))
    return out

@nki.jit
def ar_b(x, groups):
    rg = ncc.ReplicaGroup([list(g) for g in groups])
    n, d = x.shape
    out = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.shared_hbm)
    cc_src = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_dst")
    t = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    nisa.dma_copy(dst=cc_src, src=t)
    nisa.core_barrier(data=cc_src, cores=(0, 1))
    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], replica_group=rg, op=nl.add)
    u = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=u, src=cc_dst)
    nisa.dma_copy(dst=out, src=u)
    return out

groups = tuple(tuple(range(world)),)
groups = (tuple(range(world)),)
x = torch.full((N, D), float(rank + 1), dtype=torch.float32)
out = run_nki(ar_a if VARIANT == "A" else ar_b, x, groups)
expect = float(sum(r + 1 for r in range(world)))
print(f"[rank {rank}] variant {VARIANT} out unique={out.unique().tolist()} expect={expect} -> {'OK' if torch.all(out == expect) else 'WRONG'}", flush=True)
