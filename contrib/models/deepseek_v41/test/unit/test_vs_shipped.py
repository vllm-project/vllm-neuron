# SPDX-License-Identifier: Apache-2.0
"""The megakernel helpers against the port's shipped per-op kernels, one core.

    KH_CORE=8 python test_vs_shipped.py [--T 40]

Each helper is wrapped in a tiny `@nki.jit` and compared with the per-op kernel
the model ran before (`hc_attn_out.kernel`, `engram_gate.gated_add`,
`sparse_attn.gathered_attn_idx`, `swiglu.swiglu`, `rmsnorm.rms_norm`) on the
same random inputs. The algorithms are the same, so the outputs should agree to
bf16 noise or exactly.
"""

import argparse
import sys

import torch

from kharness import compare, run_nki  # noqa: E402
from deepseek_v41 import model as M  # noqa: E402
from deepseek_v41.config import DeepseekV41Config  # noqa: E402
from deepseek_v41.kernels import engram_gate, hc_attn_out, rmsnorm, sparse_attn, swiglu  # noqa: E402
from deepseek_v41.kernels import mk_common as C  # noqa: E402

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402


@nki.jit
def k_hc_out(x, res, post, comb):
    T, dim = x.shape
    hc = post.shape[1]
    y = nl.ndarray((T, hc * dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for r0, nr in C.tiles_of(T, 0) + C.tiles_of(T, 1):
        C.hc_out_tile(r0, nr, x, res, post, comb, y, hc, dim)
    return y


@nki.jit
def k_engram_gate(h, kv, w):
    T, width = h.shape
    dim = kv.shape[1] - width
    hc = width // dim
    y = nl.ndarray((T, width), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for r0, nr in C.tiles_of(T, 0) + C.tiles_of(T, 1):
        C.engram_gate_tile(r0, nr, h, kv, w, y, hc, dim)
    return y


@nki.jit
def k_attn(q, kvtab, idx, sink, flags):
    T, heads, d = flags
    Tp, hw = q.shape
    kpad = idx.shape[1]
    o = nl.ndarray((Tp, hw), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    sink_t = nl.ndarray((heads, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sink_t, src=sink[0:heads, 0:1])
    scale = float(d) ** -0.5

    def body(i):
        C.attn_token_body(i, 0, q, kvtab, idx, o, sink_t, heads, d, kpad, scale)

    nl.fori_loop(0, T, body)
    return o


@nki.jit
def k_swiglu(gate, up, w, limit):
    T, inter = gate.shape
    y = nl.ndarray((T, inter), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    C.swiglu_rows(gate, up, w, 0, y, C.tiles_of(T, 0) + C.tiles_of(T, 1), inter, limit)
    return y


@nki.jit
def k_norm(x, w):
    T, n = x.shape
    y = nl.ndarray((T, n), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    C.norm_rows(x, w, y, C.tiles_of(T, 0) + C.tiles_of(T, 1), n, qdq=False, chunk=min(n, 1024))
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=40)
    args = ap.parse_args()
    cfg = DeepseekV41Config()
    dim, hc, hd = cfg.hidden_size, cfg.hc_mult, cfg.head_dim
    T = args.T
    g = torch.Generator().manual_seed(11)
    rn = lambda *s, sd=1.0: (torch.randn(*s, generator=g) * sd)  # noqa: E731
    ok = True

    # ---- hc sublayer-out
    x = rn(T, dim).to(torch.bfloat16)
    res = rn(T, hc, dim, sd=1.5).to(torch.bfloat16)
    post = 2 * torch.sigmoid(rn(T, hc))
    comb = torch.softmax(rn(T, hc, hc), -1)
    ref = run_nki(hc_attn_out.kernel, x.unsqueeze(0), res.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0))
    got = run_nki(k_hc_out, x, res.reshape(T, -1), post, comb.reshape(T, -1))
    ok &= compare("hc_out vs shipped", got, ref.reshape(T, -1), tol_rel=1e-3)

    # ---- engram gate
    h = rn(T, hc * dim, sd=1.5).to(torch.bfloat16)
    kv = rn(T, (hc + 1) * dim, sd=0.5).to(torch.bfloat16)
    w = (torch.ones(hc * dim) + 0.1 * rn(hc * dim)).float().reshape(1, -1)
    ref = run_nki(engram_gate.gated_add, h, kv[:, : hc * dim].contiguous(), kv[:, hc * dim :].contiguous(), w)
    got = run_nki(k_engram_gate, h, kv, w)
    ok &= compare("engram gate vs shipped", got, ref, tol_rel=1e-3)

    # ---- sparse attention over a slot table with empty slots
    heads, win, topk = 8, cfg.sliding_window, 512
    kpad = C.pad_to(win + topk, C.KTILE)
    n = 1024
    q = rn(T, heads, hd).to(torch.bfloat16)
    kvtab = M.act_quant_dequant((rn(n, hd) * 0.3).to(torch.bfloat16), 32)
    idx = torch.full((T, kpad), -1, dtype=torch.int64)
    for t in range(T):
        k = win + min(topk, 1 + 7 * t)
        picks = torch.randperm(n, generator=g)[:k]
        idx[t, :k] = picks
        idx[t, : max(0, 100 - 3 * t)] = -1          # leading empties, as a short window
    sink = rn(heads, sd=0.5)
    bias = (idx < 0).float() * M._NEG_BOUND
    bias = torch.cat([bias.new_full((T, 16), hd ** -0.5), bias], 1)
    ref = run_nki(sparse_attn.gathered_attn_idx, q.reshape(T * heads, hd), kvtab,
                  idx.clamp_min(0).to(torch.int32), bias, sink.reshape(heads, 1))
    Tp = C.pad_to(T, C.PT)
    qp = torch.zeros(Tp, heads * hd, dtype=torch.bfloat16)
    qp[:T] = q.reshape(T, -1)
    idxp = torch.zeros(Tp, kpad, dtype=torch.int32)
    idxp[:T] = idx.to(torch.int32)
    got = run_nki(k_attn, qp, kvtab, idxp, sink.reshape(heads, 1), (T, heads, hd))
    ok &= compare("sparse attention vs shipped", got[:T], ref.reshape(T, -1), tol_rel=2e-3)

    # ---- swiglu with a per-row weight
    inter = cfg.moe_intermediate_size
    gate = rn(T, inter, sd=3.0).to(torch.bfloat16)
    up = rn(T, inter, sd=3.0).to(torch.bfloat16)
    wrow = torch.rand(T, 1, generator=g) * 1.5
    limit = torch.tensor([[cfg.swiglu_limit]])
    ref = run_nki(swiglu.swiglu, gate, up, wrow, limit)
    got = run_nki(k_swiglu, gate, up, wrow, float(cfg.swiglu_limit))
    ok &= compare("swiglu vs shipped", got, ref, tol_rel=1e-3)

    # ---- rmsnorm
    xr = rn(T, dim, sd=2.0).to(torch.bfloat16)
    wn = (1 + 0.1 * rn(dim)).to(torch.bfloat16).reshape(1, -1)
    ref = run_nki(rmsnorm.rms_norm, xr, wn, torch.tensor([[cfg.rms_norm_eps]]))
    got = run_nki(k_norm, xr, wn)
    ok &= compare("rmsnorm vs shipped", got, ref, tol_rel=1e-3)

    print("PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
