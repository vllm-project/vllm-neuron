# SPDX-License-Identifier: Apache-2.0
"""The greedy head (`head_kernel.head_greedy`: collapse, final norm, vocab-parallel head,
global argmax) and the embedding (`head_kernel.embed_expand`) on N ranks against CPU
references of the paths they replace.

    python kharness_mp.py 8 test_head_greedy.py --T 12

Every rank builds every rank's head shard from the same seeds, so each can compute the
full-vocabulary reference: `x = bf16(sum_c pre_mix * stream)`, `xn = bf16(rmsnorm(x) * w)`,
`logits = fp32(bf16(xn @ head_w.T))` with the vocabulary pad at -1e30, `argmax`. A row
passes when its id is the reference argmax, or -- the kernel's norm and GEMM round
differently from the CPU's -- when the reference logit at its id is within `--tol` of the
row's max (a near-tie); at most `--max-near` rows may be near-ties. The ids must also be
the same on every rank. The embedding must be exact: `stream[t, c] = embed[ids[t]]` for
every copy `c` (one rank holds each row, so the all-reduce adds zeros) and `pre_mix` the
one-hot of copy 0.
"""

import argparse
import os
import sys

if __name__ == "__main__":
    from kharness_mp import init_rank

    rank, world = init_rank()
else:
    rank, world = 0, int(os.environ.get("KH_WORLD", "8"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from kharness import run_nki  # noqa: E402
from deepseek_v41.config import DeepseekV41Config  # noqa: E402
from deepseek_v41.kernels import head_kernel  # noqa: E402
from deepseek_v41 import mk_host, model as M  # noqa: E402
from test_draft import VOCAB  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--tol", type=float, default=0.06)
    ap.add_argument("--max-near", type=int, default=2)
    ap.add_argument("--iters", type=int, default=1,
                    help="launch the head this many times, each on a new stream: a value left in a "
                         "gather buffer by the previous launch shows up as a wrong row")
    args = ap.parse_args()
    T = args.T

    cfg = DeepseekV41Config()
    dim, hc = cfg.hidden_size, cfg.hc_mult
    part = M.vocab_shard_width(VOCAB, world)
    pad = world * part - VOCAB

    g0 = torch.Generator().manual_seed(0)
    norm_w = (1 + 0.1 * torch.randn(dim, generator=g0)).to(torch.bfloat16)
    heads = [(torch.randn(part, dim, generator=torch.Generator().manual_seed(500 + r)) * 0.02).to(torch.bfloat16)
             for r in range(world)]
    bias = torch.zeros(world * part)
    if pad:
        bias[-pad:] = -1.0e30
    groups = mk_host.replica_groups(world, world)
    tag = f"[rank {rank} T={T}]"
    ok, stale = True, 0
    for it in range(args.iters):
        ok_it, n_bad = head_once(args, T, cfg, dim, hc, part, norm_w, heads, bias, groups, rank, world, tag, it,
                                 torch.Generator().manual_seed(1000 * it))
        ok &= ok_it
        stale += n_bad
    if args.iters > 1:
        print(f"{tag} {args.iters} launches: {stale} wrong rows in all", flush=True)
    g0 = torch.Generator().manual_seed(0)

    # ---- embed_expand ---------------------------------------------------------------------
    ok &= _embed_check(T, dim, hc, part, groups, rank, world, tag, g0)

    print(f"{tag} {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


def head_once(args, T, cfg, dim, hc, part, norm_w, heads, bias, groups, rank, world, tag, it, g):
    """One head launch on a new stream; `(ok, wrong rows)`."""
    stream = (torch.randn(T, hc, dim, generator=g) * 2.0).to(torch.bfloat16)
    pre_mix = torch.rand(T, hc, generator=g) * 0.8 + 0.1
    # ---- reference ------------------------------------------------------------------------
    x = (stream.float() * pre_mix.unsqueeze(-1)).sum(1).to(torch.bfloat16).float()
    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
    xn = (x * rstd * norm_w.float()).to(torch.bfloat16)
    logits = torch.cat([(xn.float() @ h.float().t()).to(torch.bfloat16).float() for h in heads], 1) + bias
    ref = logits.argmax(-1)

    # ---- kernel ---------------------------------------------------------------------------
    out = run_nki(head_kernel.head_greedy, stream, pre_mix, norm_w.reshape(1, dim), heads[rank],
                  bias[rank * part : (rank + 1) * part].reshape(1, part).contiguous(),
                  torch.tensor([[rank * part]], dtype=torch.int32), groups, (0,), grid=2)
    ids = (out[0] if isinstance(out, tuple) else out).reshape(-1).to(torch.int64)

    every = [torch.zeros_like(ids) for _ in range(world)]
    dist.all_gather(every, ids)
    same = all(torch.equal(e, ids) for e in every)
    exact = ids == ref
    rows = torch.arange(T)
    best = logits.max(-1).values
    near = (~exact) & (ids < VOCAB) & (logits[rows, ids.clamp(0, VOCAB - 1)] >= best - args.tol)
    bad = ~(exact | near)
    if args.iters == 1 or bool(bad.any()):
        print(f"{tag} launch {it} ids={ids.tolist()}", flush=True)
        print(f"{tag} launch {it} ref={ref.tolist()}", flush=True)
    if args.iters == 1 or bool(bad.any()) or not same:
        print(f"{tag} launch {it} exact {int(exact.sum())}/{T}, near-ties {int(near.sum())}, "
              f"wrong {int(bad.sum())} (rows {bad.nonzero().reshape(-1).tolist()}), same on every rank: {same}",
              flush=True)
    return same and not bool(bad.any()) and int(near.sum()) <= args.max_near, int(bad.sum())


def _embed_check(T, dim, hc, part, groups, rank, world, tag, g0):
    tables = [(torch.randn(part, dim, generator=torch.Generator().manual_seed(900 + r)) * 0.5).to(torch.bfloat16)
              for r in range(world)]
    full = torch.cat(tables)
    tok = torch.randint(0, VOCAB, (T,), generator=g0)
    tok[0], tok[-1] = 0, VOCAB - 1
    pairs = tok.to(torch.int64).view(torch.int32).reshape(T, 2).contiguous()
    st, pm = run_nki(head_kernel.embed_expand, pairs, tables[rank],
                     torch.tensor([[rank * part]], dtype=torch.int32), groups, (hc, 0), grid=2)
    want = full[tok].unsqueeze(1).expand(T, hc, dim)
    one_hot = torch.zeros(T, hc)
    one_hot[:, 0] = 1.0
    e_ok = torch.equal(st.view(torch.int16), want.contiguous().view(torch.int16)) and torch.equal(pm, one_hot)
    print(f"{tag} embed_expand {'exact' if e_ok else 'DIFFERS'}", flush=True)
    return e_ok


if __name__ == "__main__":
    main()
