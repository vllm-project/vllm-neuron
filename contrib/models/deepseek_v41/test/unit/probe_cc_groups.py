"""In-kernel all_reduce over a ReplicaGroup of several groups (the model's 8-rank
attention groups inside a 64-rank world), on N ranks split into groups of G.

    python kharness_mp.py 8 probe_cc_groups.py 4
"""
import sys

from kharness_mp import init_rank

rank, world = init_rank()
import torch  # noqa: E402

import nki  # noqa: E402
import nki.collectives as ncc  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402

from kharness import run_nki  # noqa: E402

N, D = 128, 1024
G = int(sys.argv[1]) if len(sys.argv) > 1 else 4


@nki.jit
def ar(x, groups_small, groups_world):
    rg = ncc.ReplicaGroup([list(g) for g in groups_small])
    rgw = ncc.ReplicaGroup([list(g) for g in groups_world])
    n, d = x.shape
    out = nl.ndarray((2, n, d), dtype=nl.float32, buffer=nl.shared_hbm)
    cc_src = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_src")
    cc_dst = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_dst")
    cc_dst2 = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.shared_hbm, name="cc_dst2")
    t = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    nisa.dma_copy(dst=cc_src, src=t)
    nisa.core_barrier(data=cc_src, cores=(0, 1))
    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst], replica_group=rg, op=nl.add)
    u = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=u, src=cc_dst)
    nisa.dma_copy(dst=out[0], src=u)
    # then the world, from the group result (so the two are ordered)
    nisa.dma_copy(dst=cc_src, src=u)
    nisa.core_barrier(data=cc_src, cores=(0, 1))
    ncc.all_reduce(srcs=[cc_src], dsts=[cc_dst2], replica_group=rgw, op=nl.add)
    v = nl.ndarray((n, d), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=v, src=cc_dst2)
    nisa.dma_copy(dst=out[1], src=v)
    return out


groups_small = tuple(tuple(range(g * G, (g + 1) * G)) for g in range(world // G))
groups_world = (tuple(range(world)),)
x = torch.full((N, D), float(rank + 1), dtype=torch.float32)
out = run_nki(ar, x, groups_small, groups_world)
mine = groups_small[rank // G]
exp_small = float(sum(r + 1 for r in mine))
exp_world = float(sum(sum(r + 1 for r in g) * len(g) for g in groups_small))
ok1 = bool(torch.all(out[0] == exp_small))
ok2 = bool(torch.all(out[1] == exp_world))
print(f"[rank {rank}] group {mine}: group sum unique={out[0].unique().tolist()} expect={exp_small} -> "
      f"{'OK' if ok1 else 'WRONG'}; world-after unique={out[1].unique().tolist()} expect={exp_world} -> "
      f"{'OK' if ok2 else 'WRONG'}", flush=True)
sys.exit(0 if ok1 and ok2 else 1)
