# SPDX-License-Identifier: Apache-2.0
"""The stage-boundary kernel (`kernels/tp8pp8/pp_kernels.py`) on two stages of 8 ranks.

    python kharness_mp.py 16 test_pp.py --T 128
    python kharness_mp.py 16 test_pp.py --T 512 --R 257   (the verifier's short-phase prefill)

Stage 0 sends, stage 1 receives (pairs `(t, 8 + t)`); both run the same kernel with
the role as a runtime input. The receiver's outputs must be the sender's inputs bit
for bit and its cache replicas must hold the sender's rows at the written slots,
nothing else touched; the sender's outputs are its own inputs and its caches are
unchanged (it writes its own rows back).
"""

import argparse
import sys

if __name__ == "__main__":
    from kharness_mp import init_rank

    rank, world = init_rank()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from kharness import compare, run_nki  # noqa: E402
from deepseek_v41 import mk_model  # noqa: E402
from deepseek_v41.config import DeepseekV41Config  # noqa: E402
from deepseek_v41.kernels import mk_kernels  # noqa: E402


def gather_all(t):
    parts = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(parts, t.contiguous())
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--R", type=int, default=None, help="cache rows the boundary carries (default T)")
    ap.add_argument("--nslots", type=int, default=4096)
    args = ap.parse_args()
    T = args.T
    nrows = args.R or T
    cfg = DeepseekV41Config()
    hc, dim, hd, kd, nt = cfg.hc_mult, cfg.hidden_size, cfg.head_dim, cfg.index_head_dim, cfg.index_topk
    assert world == 16, "two stages of 8 ranks"
    tp, pp = 8, 2
    plan = mk_model.PipePlan("tp8pp8", world, rank, tp, pp, rank // tp, rank % tp, [0] * 5 + [1] * 5)
    kver = 0
    g = torch.Generator().manual_seed(100 + rank)
    rn = lambda *s, sd=1.0: (torch.randn(*s, generator=g) * sd)  # noqa: E731
    ok = True
    tag = f"[rank {rank} stage {plan.stage}]"

    stream = rn(T, hc, dim).to(torch.bfloat16)
    pre_mix = torch.softmax(rn(T, hc), -1)
    topk = torch.randint(-1, 2000, (T, nt), generator=g).to(torch.int32)
    nslots = args.nslots
    cache = rn(nslots, hd, sd=0.3).to(torch.bfloat16)
    icache = rn(nslots, kd, sd=0.3).to(torch.bfloat16)
    slots = torch.randperm(nslots, generator=torch.Generator().manual_seed(7))[:nrows].to(torch.int32).reshape(nrows, 1)
    sender = plan.stage == 0
    role = torch.full((1, 1), 1.0 if sender else 0.0, dtype=torch.float32)
    groups = plan.boundary_groups(0)
    cache_dev, icache_dev = cache.to("neuron:0"), icache.to("neuron:0")
    out = run_nki(mk_kernels.boundary(0), stream, pre_mix, topk, cache_dev, icache_dev, slots, role, groups, (kver,))
    o_stream, o_mix, o_topk = out[0], out[1], out[2]
    cache_after, icache_after = cache_dev.cpu(), icache_dev.cpu()
    all_stream = gather_all(stream.float())
    all_mix = gather_all(pre_mix)
    all_topk = gather_all(topk)
    all_cache = gather_all(cache.float())
    all_icache = gather_all(icache.float())
    src = rank if sender else rank - tp
    ok &= compare(f"{tag} boundary stream (from rank {src})", o_stream.float(), all_stream[src], tol_rel=0.0)
    ok &= compare(f"{tag} boundary pre_mix", o_mix, all_mix[src], tol_rel=0.0)
    same_topk = torch.equal(o_topk, all_topk[src])
    print(f"{tag} boundary topk identical: {same_topk}", flush=True)
    ok &= same_topk
    sl = slots.reshape(-1).long()
    exp_cache = cache.float().clone()
    exp_icache = icache.float().clone()
    if not sender:
        exp_cache[sl] = all_cache[src][sl]
        exp_icache[sl] = all_icache[src][sl]
    ok &= compare(f"{tag} compressed cache after boundary", cache_after.float(), exp_cache, tol_rel=0.0)
    ok &= compare(f"{tag} index cache after boundary", icache_after.float(), exp_icache, tol_rel=0.0)

    print(f"{tag} {'PASS' if ok else 'FAIL'}", flush=True)
    dist.all_reduce(torch.zeros(1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
