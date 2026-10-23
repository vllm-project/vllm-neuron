"""In-kernel nki.collectives.all_reduce through wrap_nki + torch.compile, N ranks."""
import os, sys
from kharness_mp import init_rank
rank, world = init_rank()
import torch
import nki, nki.isa as nisa, nki.language as nl
import nki.collectives as ncc
ncc.ReplicaGroup.__hash__ = lambda self: hash(tuple(tuple(r) for r in self._value))
from kharness import run_nki

N, D = 128, 1024

@nki.jit
def ar_kernel(x, groups):
    n, d = x.shape
    rg = ncc.ReplicaGroup([list(g) for g in groups])
    out = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.shared_hbm)
    rid_out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
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
    nisa.core_barrier(data=cc_dst, cores=(0, 1))
    u = nl.ndarray((half, d), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=u, src=cc_dst[r0:r0 + half, :])
    nisa.dma_copy(dst=out[r0:r0 + half, :], src=u)
    if pid == 0:
        rid = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.register_store(rid, ncc.rank_id())
        nisa.dma_copy(dst=rid_out, src=rid)
    nisa.core_barrier(data=out, cores=(0, 1))
    return out, rid_out

groups = [list(range(world))]
if len(sys.argv) > 1 and sys.argv[1] == "half":
    h = world // 2
    groups = [list(range(h)), list(range(h, world))]
rg = tuple(tuple(g) for g in groups)
x = torch.full((N, D), float(rank + 1), dtype=torch.float32)
out, rid = run_nki(ar_kernel, x, rg)
mine = [g for g in groups if rank in g][0]
expect = float(sum(r + 1 for r in mine))
print(f"[rank {rank}] rank_id()={int(rid)} groups={groups} out unique={out.unique().tolist()} expect={expect} -> {'OK' if torch.all(out == expect) else 'WRONG'}", flush=True)
