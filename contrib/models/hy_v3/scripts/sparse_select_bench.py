"""Single-core timing of the EXPERIMENTAL sparse-decode block selection (docs/SPARSE_DECODE.md).

At the full-model per-rank shapes (1 Q head, 1 KV head, head_dim 128, 2688
32-token blocks in the 86016 context bucket, 512 kept), each variant compiled
as its own graph on neuron:0 and timed over 48 calls (synchronized every 8;
the "trivial" line is the per-call launch + sync overhead):

  per-block, cumulative: summary gather -> bound -> top-511 -> + sort
  per-block, bisection:  16-step threshold bisection instead of top-k, then a
                         cumsum + scatter or cumsum + compare compaction
  model:                 hy_v3_neuron.model.sparse_select_blocks (groups of
                         --group blocks, top-(512 / group - 1))

Usage: python sparse_select_bench.py [--batch 8] [--tokens 4] [--group 4]
"""

import argparse
import os
import time

os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:62182")
os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")

import torch
from hy_v3_neuron.model import sparse_select_blocks

from vllm_neuron.envs import get_compile_backend_name

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--tokens", type=int, default=4, help="tokens per request per step (MTP k + 1)")
ap.add_argument("--group", type=int, default=4)
a = ap.parse_args()

DEV = "neuron:0"
B, S, G = a.batch, a.tokens, a.group
NH, KVH, D, BS = 1, 1, 128, 32
NB, NSEL, NBLK = 2688, 512, 21800


def timeit(fn, args, n=48):
    c = torch.compile(fn, backend=get_compile_backend_name(), fullgraph=True)

    def sync(out):
        return out.cpu() if torch.is_tensor(out) else [o.cpu() for o in out]

    with torch.no_grad():
        for _ in range(5):
            out = c(*args)
        sync(out)
        t0 = time.time()
        for i in range(n):
            out = c(*args)
            if i % 8 == 7:
                sync(out)
        sync(out)
    return (time.time() - t0) / n * 1e3


def score(q, k_sum, t):
    s = k_sum[torch.clamp(t.to(torch.long), min=0)].to(torch.float32)  # [B, nb, 2, kvh, D]
    qg = q.view(KVH, NH // KVH, B, S, D).permute(2, 0, 1, 3, 4).reshape(B, KVH, -1, D)
    bound = torch.einsum("bhqd,bnhd->bnhq", qg.clamp(min=0), s[:, :, 1]) + torch.einsum(
        "bhqd,bnhd->bnhq", qg.clamp(max=0), s[:, :, 0])
    return bound.amax(dim=(2, 3))


def bisect(q, k_sum, t, iters=16):
    sc = score(q, k_sum, t)
    lo, hi = sc.amin(dim=1, keepdim=True), sc.amax(dim=1, keepdim=True)
    for _ in range(iters):
        mid = (lo + hi) * 0.5
        over = (sc > mid).sum(dim=1, keepdim=True) > NSEL - 1
        lo, hi = torch.where(over, mid, lo), torch.where(over, hi, mid)
    return sc > hi


def compact_scatter(q, k_sum, t):
    keep = bisect(q, k_sum, t)
    dest = torch.where(keep, torch.cumsum(keep.to(torch.int32), dim=1) - 1, NSEL).to(torch.long)
    return torch.zeros(B, NSEL + 1, dtype=t.dtype, device=t.device).scatter(1, dest, t)[:, :NSEL]


def compact_compare(q, k_sum, t):
    csum = torch.cumsum(bisect(q, k_sum, t).to(torch.float32), dim=1)
    jj = torch.arange(1, NSEL + 1, device=t.device, dtype=torch.float32).view(1, NSEL, 1)
    pos = (csum.unsqueeze(1) < jj).to(torch.float32).sum(-1)
    return t.gather(1, torch.clamp(pos, max=NB - 1).to(torch.long))


g = torch.Generator().manual_seed(0)
lo = torch.randn(NBLK, KVH, D, generator=g) - 1
k_sum = torch.stack([lo, lo + 2], dim=1).to(torch.bfloat16).to(DEV)
t = torch.stack([torch.randperm(NBLK, generator=g)[:NB] for _ in range(B)]).to(torch.int32).to(DEV)
n_hist = torch.full((B,), 85000, dtype=torch.int32).to(DEV)
q = torch.randn(NH, B, S, D, generator=g).to(DEV)

print(f"batch {B}, {S} tokens per request, {NB} blocks, {NSEL} kept")
cases = [
    ("trivial", lambda q: q * 2, (q,)),
    ("block: gather", lambda q, s, t: k_sum[torch.clamp(t.to(torch.long), min=0)].float().sum(), (q, k_sum, t)),
    ("block: + bound", score, (q, k_sum, t)),
    ("block: + top-511", lambda q, s, t: torch.topk(score(q, s, t), NSEL - 1, dim=1).indices, (q, k_sum, t)),
    ("block: + sort", lambda q, s, t: torch.sort(torch.topk(score(q, s, t), NSEL - 1, dim=1).indices, dim=1).values,
     (q, k_sum, t)),
    ("block: bisection", bisect, (q, k_sum, t)),
    ("block: + scatter", compact_scatter, (q, k_sum, t)),
    ("block: + compare", compact_compare, (q, k_sum, t)),
    (f"model, group {G}", lambda q, s, t, n: sparse_select_blocks(q, s, t, n, BS, G, NSEL, 1, 2),
     (q, k_sum, t, n_hist)),
]
for name, fn, args in cases:
    print(f"{name:20s} {timeit(fn, args):7.3f} ms", flush=True)
