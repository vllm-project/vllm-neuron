# SPDX-License-Identifier: Apache-2.0
"""The optimized r2 prefill kernel (`kernels/tp64/r2_prefill`) against the generic block
body specialized to (prefill, r2, tp64), on identical inputs, on N ranks.

    python kharness_mp.py 8 test_r2_prefill_diff.py [--cases 7:0,512:0,2x300:400]

The optimized kernel must be **bit-identical** to the generic one: same arithmetic, only
scheduling / data movement changed (that is the bar it was optimized under). The recorded
cases of its workspace cover T = 1024 / 2048 only; this sweeps the token count (incl. T
below one 128-row tile, odd T, T not a multiple of the attention step), the start position
(read length) and several spans. Inputs are built as `test_layer.py` builds them (random
weights per rank, 8 heads and 6 routed experts a rank, the attention group = the world).
Each case: `[rank r case] PASS` when every output and every cache is byte-equal.
"""

import argparse

if __name__ == "__main__":
    from kharness_mp import init_rank

    rank, world = init_rank()

import torch  # noqa: E402

from kharness import run_nki  # noqa: E402
from deepseek_v41.config import DeepseekV41Config  # noqa: E402
from deepseek_v41.kernels import mk_kernels  # noqa: E402
from deepseek_v41 import mk_host, model as M  # noqa: E402
from test_layer import make_params, mx4_stack, mx8  # noqa: E402

#: (spans, tokens per span, start position of the first span)
DEFAULT_CASES = "7:0,23:200,56:2048,129:0,333:500,512:0,777:1300,1500:100,2x300:400"
OUT_NAMES = ["out_stream", "ffn_pre", "topk_out", "tap_out", "window_kv_cache",
             "compress_kv_cache", "pool_kv", "pool_score", "index_k_cache"]
UNWRITTEN = {"topk_out", "tap_out"}


def parse_cases(spec):
    cases = []
    for item in spec.split(","):
        size, start = item.split(":")
        b, l = (int(v) for v in size.split("x")) if "x" in size else (1, int(size))
        cases.append((b, l, int(start)))
    return cases


def build_args(cfg, p, B, L, start, world):
    """The kernel's positional arguments (CPU tensors) for one r2 prefill step."""
    dim, hc, hd = cfg.hidden_size, cfg.hc_mult, cfg.head_dim
    win, block_size, ratio = cfg.sliding_window, 32, 2
    T = B * L
    g0 = torch.Generator().manual_seed(T * 7919 + start)
    x = (torch.randn(T, hc, dim, generator=g0) * 1.5).to(torch.bfloat16)
    pre_mix = torch.softmax(torch.randn(T, hc, generator=g0), -1)
    nblk = max(16, -(-(start + 37 * (B - 1) + L) // block_size))
    num_slots = max(64, B * nblk) * block_size
    cache = M.act_quant_dequant((torch.randn(num_slots, hd, generator=g0) * 0.3).to(torch.bfloat16), 32)
    spans, tables, slots = [], [], []
    for b in range(B):
        s = start + 37 * b
        spans.append((b * L, L, s))
        table = torch.arange(b * nblk, (b + 1) * nblk)
        tables.append(table)
        slots.append(mk_host.slots_of(table, torch.arange(s, s + L), block_size))
    compress_len = (start + 37 * (B - 1) + L) // ratio
    read_len = max(128, -(-compress_len // 128) * 128)
    idx = mk_host.build_attn_indices(spans, tables, torch.cat(slots), block_size, win, read_len, ratio)
    topk = torch.full((T, cfg.index_topk), -1, dtype=torch.int64)
    for t in range(T):
        b = t // L
        clen = (spans[b][2] + t - b * L + 1) // ratio
        k = min(cfg.index_topk, clen)
        if k > 0:
            topk[t, :k] = torch.randperm(clen, generator=g0)[:k]
    ccache = M.fp4_quant_dequant((torch.randn(num_slots, hd, generator=g0) * 0.3).to(torch.bfloat16), 16,
                                 scale_e4m3=True)
    kcache = M.fp4_quant_dequant((torch.randn(num_slots, 128, generator=g0) * 0.3).to(torch.bfloat16), 32)
    one = torch.zeros(1, 1, dtype=torch.int32)
    groups = mk_host.replica_groups(world, world)
    flags = (ratio, B, win, 9, 0, 0, 0, 0, 0)
    return [
        x, pre_mix,
        p["hca_fn"].t().contiguous(), p["hca_base"].reshape(1, -1), p["hca_scale"].reshape(1, 3), p["attn_norm_w"].reshape(1, -1),
        p["hcf_fn"].t().contiguous(), p["hcf_base"].reshape(1, -1), p["hcf_scale"].reshape(1, 3), p["ffn_norm_w"].reshape(1, -1),
        *mx8(p["wq_a"], p["wq_a_s"]), p["q_norm_w"].reshape(1, -1), *mx8(p["wq_b"], p["wq_b_s"]), *mx8(p["wkv"], p["wkv_s"]), p["kv_norm_w"].reshape(1, -1),
        *mx8(p["wo_a"], p["wo_a_s"]), *mx8(p["wo_b"], p["wo_b_s"]), p["attn_sink"].reshape(-1, 1), p["freqs"],
        idx.row_pos, idx.write_slots, idx.win_slots, idx.cmp_slots, idx.twin, idx.wv, idx.cbase, topk.to(torch.int32).contiguous(),
        cache, ccache, torch.zeros(1, 1), torch.zeros(1, 1), kcache,
        p["cm_wkv"], p["cm_wkv"].t().float().contiguous(), p["cm_wgate"].t().float().contiguous(), p["cm_norm_w"].reshape(1, -1),
        one, one, one,
        *mx8(p["ix_wq_b"], p["ix_wq_b_s"]), p["ix_wproj"].t().float().contiguous(), p["ix_wk"], p["ix_knorm_w"].reshape(1, -1),
        idx.clen,
        one, torch.zeros(1, 1, dtype=torch.bfloat16), one, torch.zeros(1, 1, dtype=torch.uint8),
        torch.zeros(1, 1, dtype=torch.uint8), torch.zeros(1, 1),
        p["gate_w"].t().float().contiguous(), p["gate_b"].reshape(1, -1), torch.tensor([[float(p["expert_lo"])]]),
        *mx4_stack(p["w1"], p["s1"]), *mx4_stack(p["w3"], p["s3"]), *mx4_stack(p["w2"], p["s2"]),
        *mx8(p["sh_w1"], p["sh_s1"]), *mx8(p["sh_w3"], p["sh_s3"]), *mx8(p["sh_w2"], p["sh_s2"]),
        groups, groups, flags,
    ]


def launch(kernel, args):
    """Run `kernel` on fresh device copies of the (in-place written) caches."""
    fresh = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
    return run_nki(kernel, *fresh, grid=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=DEFAULT_CASES, help="comma list of [<spans>x]<tokens>:<start>")
    args = ap.parse_args()
    cfg = DeepseekV41Config()
    heads = cfg.num_attention_heads // world if world <= 8 else 8
    ngroups = max(1, cfg.o_groups // world) if world <= 8 else 1
    p = make_params(cfg, rank, world, heads, ngroups, 6, 4096)
    arch = (2, 0, 0, 0, 0)
    generic = mk_kernels.layer_generic("prefill", arch, "tp64")
    optimized = mk_kernels._OPTIMIZED[("prefill", arch, "tp64")]
    all_ok = True
    for B, L, start in parse_cases(args.cases):
        tag = f"[rank {rank} B={B} L={L} start={start}]"
        kargs = build_args(cfg, p, B, L, start, world)
        ref = launch(generic, kargs)
        got = launch(optimized, kargs)
        ok, notes = True, []
        for name, r, g in zip(OUT_NAMES, ref, got):
            if name in UNWRITTEN:
                continue
            if r.shape != g.shape or r.dtype != g.dtype:
                ok = False
                notes.append(f"{name} shape/dtype {tuple(g.shape)} {g.dtype} vs {tuple(r.shape)} {r.dtype}")
            elif not torch.equal(r, g):
                ok = False
                err = (g.double() - r.double()).abs()
                notes.append(f"{name} max_abs {float(err.max()):.4g} ({int((err > 0).sum())} elements differ)")
        all_ok &= ok
        print(f"{tag} {'PASS' if ok else 'FAIL'} {'; '.join(notes)}", flush=True)
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
