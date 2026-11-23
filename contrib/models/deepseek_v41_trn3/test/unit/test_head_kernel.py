# SPDX-License-Identifier: Apache-2.0
"""The greedy head kernel (`kernels/tp64/head_kernel.head_greedy`) against a CPU reference
of `model.forward`'s tail (collapse -> norm -> vocab-parallel head -> argmax over the
gathered logits), on N ranks.

    python kharness_mp.py 8 test_head_kernel.py [--rows 6,60,128] [--vocab 16000]

The vocabulary is small enough that one rank's shard fits the kernel's SBUF tiles at
8 ranks (`part` = 2048 columns, the flat scheme's width at 64 ranks); the last rank's
shard is part padding, whose head rows are made huge so an unmasked pad would win. Two
real columns on different ranks get identical, dominant head rows: the kernel must pick
the lower index, as `argmax` does. A row whose token differs from the reference's must
be a near tie in the reference's own logits (the device and CPU round the bf16 logits
independently).
"""

import argparse

if __name__ == "__main__":
    from kharness_mp import init_rank

    rank, world = init_rank()

import torch  # noqa: E402

import kharness as KH  # noqa: E402
from deepseek_v41 import mk_host, model as M  # noqa: E402
from deepseek_v41.kernels.tp64 import head_kernel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="6,60,128")
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--dim", type=int, default=5120)
    args = ap.parse_args()
    hc, dim, vocab = 4, args.dim, args.vocab
    part = M.vocab_shard_width(vocab, world)
    groups = mk_host.replica_groups(world, world)
    g = torch.Generator().manual_seed(1234)
    w_full = (torch.randn(world * part, dim, generator=g) * 0.02).to(torch.bfloat16)
    w_full[vocab:] = (torch.randn(world * part - vocab, dim, generator=g) * 2.0).to(torch.bfloat16)  # the pad
    a, b = part // 3, 5 * part + 7          # a tie across ranks 0 and 5
    w_full[b] = w_full[a] = (torch.randn(dim, generator=g) * 0.08).to(torch.bfloat16)
    norm_w = (1.0 + 0.1 * torch.randn(1, dim, generator=g)).float()
    mine = w_full[rank * part : (rank + 1) * part].contiguous()
    vstart = torch.tensor([[rank * part]], dtype=torch.int32)
    kernel = KH._entry(head_kernel.head_greedy, 2)
    ok = True
    for T in (int(v) for v in args.rows.split(",")):
        stream = (torch.randn(T, hc, dim, generator=g)).to(torch.bfloat16)
        pre_mix = (0.25 + 0.5 * torch.rand(T, hc, generator=g)).float()
        got = kernel(KH.to_dev(stream), KH.to_dev(pre_mix), KH.to_dev(norm_w), KH.to_dev(mine),
                     KH.to_dev(vstart), groups, (vocab, 0))
        got = (got[0] if isinstance(got, (tuple, list)) else got).cpu().reshape(-1).long()
        # reference: collapse (fp32 sum, bf16), norm (fp32, bf16), head (fp32 acc, bf16), argmax
        x = (stream.float() * pre_mix.unsqueeze(-1)).sum(1).to(torch.bfloat16).float()
        y = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-20) * norm_w).to(torch.bfloat16).float()
        logits = (y @ w_full.float().T).to(torch.bfloat16).float()
        logits[:, vocab:] = float("-inf")
        ref = logits.argmax(-1)
        diff = (got != ref).nonzero().reshape(-1).tolist()
        bad = []
        for t in diff:
            gv, rv = float(logits[t, got[t]]) if got[t] < vocab else float("-inf"), float(logits[t, ref[t]])
            if not (rv - gv <= max(1e-2 * abs(rv), 1e-3)):
                bad.append((t, int(got[t]), int(ref[t]), gv, rv))
        ties = int((ref == a).sum())
        tie_ok = bool((got[ref == a] == a).all()) and not bool((got == b).any())
        row_ok = not bad and len(diff) <= max(1, T // 20) and tie_ok
        ok &= row_ok
        if rank == 0:
            print(f"[T={T}] {'PASS' if row_ok else 'FAIL'}: {T - len(diff)}/{T} rows equal the reference"
                  f"{'' if not diff else f' ({len(diff)} near-tie rows differ)'}; tie rows {ties} "
                  f"(lowest index {'kept' if tie_ok else 'LOST'}){'' if not bad else f'; WRONG {bad[:4]}'}",
                  flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
