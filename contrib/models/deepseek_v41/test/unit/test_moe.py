# SPDX-License-Identifier: Apache-2.0
"""`kernels/mk_moe` (token-grouped fp4 routed experts) against a CPU reference,
on one logical core (grid 2).

    python test_moe.py [--T 6]
"""

import argparse

import torch

from kharness import compare, run_nki  # noqa: E402 - sets the environment first

import nki
import nki.isa as nisa
import nki.language as nl

from deepseek_v41.kernels import mk_moe
from deepseek_v41.weight_loader import dequant_block_fp4
import ref_layer as R
from packs import fp4_pack

PT = 128


@nki.jit
def k_moe(x, wexp, w1, s1, w3, s3, w2, s2):
    Tp1, dim = x.shape
    Tp = Tp1 - PT
    n_local = w1.shape[0]
    pid = nl.program_id(0)
    eacc = nl.ndarray((Tp + PT, dim), dtype=nl.float32, buffer=nl.shared_hbm)
    experts = [(w1[e], s1[e], w3[e], s3[e], w2[e], s2[e]) for e in range(n_local)]
    mk_moe.routed_experts(pid, n_local, Tp, x, wexp, experts, eacc)
    return eacc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=6)
    args = ap.parse_args()
    torch.manual_seed(0)
    T, dim, inter, E = args.T, 5120, 2304, 6
    Tp = -(-T // PT) * PT
    g = torch.Generator().manual_seed(1)
    ws = {}
    for name, (n, k) in (("w1", (inter, dim)), ("w3", (inter, dim)), ("w2", (dim, inter))):
        packs = [fp4_pack((torch.randn(n, k, generator=g) * 0.03).to(torch.bfloat16)) for _ in range(E)]
        ws[name] = torch.stack([p[0] for p in packs])            # [E, n, k/2]
        ws["s" + name[1]] = torch.stack([p[1] for p in packs])   # [E, n, k/32]
    x = torch.zeros(Tp + PT, dim, dtype=torch.bfloat16)
    x[:T] = (torch.randn(T, dim, generator=g) * 1.5).to(torch.bfloat16)
    wexp = torch.zeros(Tp + PT, E)
    for t in range(T):
        picks = torch.randperm(E, generator=g)[: 2 if t % 3 else 0]   # some rows pick nothing here
        for e in picks.tolist():
            wexp[t, e] = 0.2 + torch.rand(1, generator=g).item()
    wexp[:, 5] = 0.0   # an expert nobody picked
    # reference
    xq = R.act_quant_t2(x[:T])
    ref = torch.zeros(T, dim)
    for e in range(E):
        we = wexp[:T, e : e + 1]
        if not bool((we > 0).any()):
            continue
        gg = R.linear_fp4(xq, ws["w1"][e], ws["s1"][e]).float()
        uu = R.linear_fp4(xq, ws["w3"][e], ws["s3"][e]).float()
        y = (torch.nn.functional.silu(gg.clamp(max=10.0)) * uu.clamp(-10.0, 10.0) * we).to(torch.bfloat16)
        ref += (R.linear_fp4(R.act_quant_t2(y), ws["w2"][e], ws["s2"][e]).float()) * (we > 0).float()
    km = lambda t: t.transpose(1, 2).contiguous()  # noqa: E731 - [E, n, k'] -> [E, k', n]
    got = run_nki(k_moe, x, wexp, km(ws["w1"]), km(ws["s1"]), km(ws["w3"]), km(ws["s3"]),
                  km(ws["w2"]), km(ws["s2"]), grid=2)
    ok = compare(f"routed experts T={T}", got[:T], ref, tol_rel=2e-2)
    ok &= bool(torch.isfinite(got).all())
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
