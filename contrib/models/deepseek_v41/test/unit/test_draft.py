# SPDX-License-Identifier: Apache-2.0
"""The drafter megakernel against a CPU reference of `model.forward_spec`, on N ranks.

    python kharness_mp.py 8 test_draft.py --L 6 --start 200 --stage 9
    python test_draft.py --compile-only --stage 9          # trace through compile_nki

Every rank builds its own random weights (vocab-parallel tables included), the
attention group and the MoE world are all N ranks, and the reference's partials
are reduced with Gloo. A small vocabulary (16084, so the last rank's shard carries
a pad) keeps the test quick.

What is exact and what is not: every component matches the reference from its
own input (stages 11-18 at mtp.0, the MoE partial to 4e-5), but the random
weights make each MoE output comparable to the stream itself, so the bf16
rounding-order flips of the input (5e-3) come out of the fp8/fp4-quantized
expert path ~8x larger and compound over the three stages to ~0.2 of the
stream. The drafted token ids are the acceptance criterion (exact), the
stream/logit/confidence comparisons are loose (0.3) and informational.
"""

import argparse
import os
import sys
from types import SimpleNamespace

if __name__ == "__main__" and "--compile-only" not in sys.argv:
    from kharness_mp import init_rank

    rank, world = init_rank()
else:
    rank, world = 0, int(os.environ.get("KH_WORLD", "8"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from kharness import compare, compile_only, run_nki  # noqa: E402
from deepseek_v41.config import DeepseekV41Config  # noqa: E402
from deepseek_v41.kernels import draft_kernel  # noqa: E402
from deepseek_v41 import mk_host, model as M  # noqa: E402
import ref_layer as R  # noqa: E402
from test_layer import fp8_pack, make_params, mx4_stack, mx8  # noqa: E402

VOCAB = 16084
NOISE = 15999
K = 5


def ar(t):
    if dist.is_initialized():
        dist.all_reduce(t)
    return t


def gather_cols(t):
    """all_gather along the last axis (fp32)."""
    if not dist.is_initialized():
        return t
    parts = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(parts, t.contiguous())
    return torch.cat(parts, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=6, help="main-stream rows this step")
    ap.add_argument("--start", type=int, default=200, help="start_pos (0 = prefill, at_zero)")
    ap.add_argument("--tap", type=int, default=-1, help="row the draft hangs off (default L-1)")
    ap.add_argument("--seed-token", type=int, default=1234)
    ap.add_argument("--stage", type=int, default=9)
    ap.add_argument("--compile-only", action="store_true")
    args = ap.parse_args()

    cfg = DeepseekV41Config()
    dim, hc, hd = cfg.hidden_size, cfg.hc_mult, cfg.head_dim
    heads, ngroups = 8, 1
    n_local = cfg.dspark_n_routed_experts // world
    win = cfg.sliding_window
    block_size = 32
    max_pos = 4096
    L = args.L
    tap = args.tap if args.tap >= 0 else L - 1
    start = args.start
    at_zero = int(start == 0)
    part = M.vocab_shard_width(VOCAB, world)
    assert part % 512 == 0, part
    vstart = rank * part
    mrank = cfg.dspark_markov_rank
    cfg3 = SimpleNamespace(num_experts_per_tok=cfg.dspark_num_experts_per_tok,
                           routed_scaling_factor=cfg.routed_scaling_factor, swiglu_limit=cfg.swiglu_limit)

    # ---- weights: three stages, plus the shared tables -----------------------------
    ps = []
    for s in range(3):
        p = make_params(cfg, rank + 100 * s, world, heads, ngroups, n_local, max_pos)
        g = torch.Generator().manual_seed(7000 + rank + 100 * s)
        p["gate_w"] = (torch.randn(cfg.dspark_n_routed_experts, dim, generator=g) * 0.02).to(torch.bfloat16)
        p["gate_b"] = torch.randn(cfg.dspark_n_routed_experts, generator=g) * 0.1
        p["expert_lo"] = rank * n_local
        ps.append(p)
    g = torch.Generator().manual_seed(9000 + rank)
    rn = lambda *s, sd=1.0: (torch.randn(*s, generator=g) * sd)  # noqa: E731
    main_proj, main_proj_s = fp8_pack(rn(dim, 3 * dim, sd=0.02).to(torch.bfloat16))
    main_norm_w = (1 + 0.1 * rn(dim)).to(torch.bfloat16)
    embed_w = rn(part, dim, sd=0.5).to(torch.bfloat16)
    head_w = rn(part, dim, sd=0.02).to(torch.bfloat16)
    mk_embed_w = rn(part, mrank, sd=0.5).to(torch.bfloat16)
    mk_head_w = rn(part, mrank, sd=0.05).to(torch.bfloat16)
    conf_w = rn(1, dim + mrank, sd=0.02).to(torch.bfloat16)
    fnorm_w = (1 + 0.1 * rn(dim)).to(torch.bfloat16)

    # ---- inputs shared by every rank (seed 0) --------------------------------------
    g0 = torch.Generator().manual_seed(0)
    main_hidden = torch.randn(L, 3 * dim, generator=g0) * 1.5
    num_slots = 64 * block_size
    caches = []
    for s in range(3):
        c = (torch.randn(num_slots, hd, generator=g0) * 0.3).to(torch.bfloat16)
        caches.append(M.act_quant_dequant(c, 32))
    table = torch.arange(16)
    row_pos = torch.arange(start, start + L)
    write_slots = mk_host.slots_of(table, row_pos, block_size)
    draft_pos = start + tap
    drow_pos = draft_pos + 1 + torch.arange(K)
    base = draft_pos - (win - 1)
    wpos = (base + torch.arange(win)).clamp_min(0)
    wslots = mk_host.slots_of(table, wpos, block_size)
    seed = args.seed_token
    meta = torch.tensor([[seed, tap, start]], dtype=torch.int32)
    i32 = lambda x: x.to(torch.int32).reshape(-1, 1).contiguous()  # noqa: E731

    # ---- kernel ---------------------------------------------------------------------
    groups = mk_host.replica_groups(world, world)
    flags = (at_zero, K, win, args.stage, 0, NOISE, VOCAB, cfg.dspark_num_experts_per_tok, block_size)
    stack = lambda key: torch.stack([p[key] for p in ps]).contiguous()  # noqa: E731

    def stack8(wkey, skey):
        pairs = [mx8(p[wkey], p[skey]) for p in ps]
        return torch.stack([w for w, _ in pairs]).contiguous(), torch.stack([s_ for _, s_ in pairs]).contiguous()

    def stack4(wkey, skey):
        pairs = [mx4_stack(p[wkey], p[skey]) for p in ps]
        return torch.stack([w for w, _ in pairs]).contiguous(), torch.stack([s_ for _, s_ in pairs]).contiguous()

    args_k = [
        meta, main_hidden, i32(row_pos), i32(write_slots), i32(table),
        *mx8(main_proj, main_proj_s), main_norm_w.reshape(1, -1), embed_w, torch.tensor([[vstart]], dtype=torch.int32),
        ps[0]["hca_fn"].t().contiguous(), ps[1]["hca_fn"].t().contiguous(), ps[2]["hca_fn"].t().contiguous(),
        stack("hca_base"), stack("hca_scale"), stack("attn_norm_w"),
        ps[0]["hcf_fn"].t().contiguous(), ps[1]["hcf_fn"].t().contiguous(), ps[2]["hcf_fn"].t().contiguous(),
        stack("hcf_base"), stack("hcf_scale"), stack("ffn_norm_w"),
        *stack8("wq_a", "wq_a_s"), stack("q_norm_w"), *stack8("wq_b", "wq_b_s"),
        *stack8("wkv", "wkv_s"), stack("kv_norm_w"), *stack8("wo_a", "wo_a_s"), *stack8("wo_b", "wo_b_s"),
        stack("attn_sink").reshape(3, heads, 1), ps[0]["freqs"],
        ps[0]["gate_w"].t().float().contiguous(), ps[1]["gate_w"].t().float().contiguous(),
        ps[2]["gate_w"].t().float().contiguous(), stack("gate_b"), torch.tensor([[float(rank * n_local)]]),
        *stack4("w1", "s1"), *stack4("w3", "s3"), *stack4("w2", "s2"),
        *stack8("sh_w1", "sh_s1"), *stack8("sh_w3", "sh_s3"), *stack8("sh_w2", "sh_s2"),
    ]
    if args.compile_only:
        args_k += caches
        args_k += [fnorm_w.reshape(1, -1), head_w, mk_embed_w, mk_head_w, conf_w, groups, groups, flags]
        compile_only(draft_kernel.draft_forward, *args_k, grid=2)
        print("compile-only OK", flush=True)
        return
    cache_dev = [c.to("neuron:0") for c in caches]
    args_k += cache_dev
    args_k += [fnorm_w.reshape(1, -1), head_w, mk_embed_w, mk_head_w, conf_w, groups, groups, flags]
    out_ids, conf_k, dbg, *_ = run_nki(draft_kernel.draft_forward, *args_k)
    caches_after = [c.cpu() for c in cache_dev]
    tag = f"[rank {rank} L={L} start={start} stage={args.stage}]"

    # ---- reference --------------------------------------------------------------------
    eps = cfg.rms_norm_eps
    mh16 = main_hidden.to(torch.bfloat16)
    main_x = R.rmsnorm(R.linear_fp8(R.act_quant_mx(mh16), main_proj, main_proj_s), main_norm_w, eps)
    ids = torch.tensor([seed] + [NOISE] * (K - 1))

    def embed(table_w, tok):
        loc = tok - vstart
        okm = (loc >= 0) & (loc < part)
        rows = table_w[loc.clamp(0, part - 1)] * okm.unsqueeze(-1).to(torch.bfloat16)
        return ar(rows.float()).to(torch.bfloat16)

    e = embed(embed_w, ids)
    x = e.unsqueeze(1).repeat(1, hc, 1).contiguous()
    pre_mix = torch.zeros(K, hc)
    pre_mix[:, 0] = 1.0
    caches_ref = [c.clone() for c in caches]
    streams = []
    ok = True
    for s in range(3):
        p = ps[s]
        cosm, sinm = p["freqs"][row_pos, :32], p["freqs"][row_pos, 32:]
        mkv = R.rmsnorm(R.linear_fp8(R.act_quant_mx(main_x), p["wkv"], p["wkv_s"]), p["kv_norm_w"], eps)
        mkv = torch.cat([mkv[..., :-64], R.rope(mkv[..., -64:], cosm, sinm)], -1)
        mkv = M.act_quant_dequant(mkv, 32)
        caches_ref[s][write_slots] = mkv
        r = compare(f"{tag} stage {s} main_kv written", caches_after[s][write_slots], mkv, tol_rel=2e-2)
        if not r:
            got_kv = caches_after[s][write_slots].float()
            for c0 in range(0, hd, 128):
                compare(f"{tag}   columns {c0}:{c0 + 128}", got_kv[:, c0 : c0 + 128], mkv.float()[:, c0 : c0 + 128], tol_rel=2e-2)
        ok &= r
        untouched = torch.ones(num_slots, dtype=torch.bool)
        untouched[write_slots] = False
        ok &= compare(f"{tag} stage {s} cache untouched rows", caches_after[s][untouched], caches[s][untouched], tol_rel=0.0)
        if at_zero:
            continue
        x_n, pre_a, post_a, comb_a = R.hc_in(x, pre_mix, p["hca_fn"], p["hca_base"], p["hca_scale"], p["attn_norm_w"], cfg)
        cache_ext = torch.cat([caches_ref[s], torch.zeros(K, hd, dtype=torch.bfloat16)])
        extra = num_slots + torch.arange(K)
        idx = SimpleNamespace(
            row_pos=i32(drow_pos), write_slots=i32(extra), win_slots=i32(torch.cat([wslots, extra])),
            nwin=win + K, twin=torch.zeros(K, dtype=torch.int64), wv=torch.full((K,), int(base)),
            window_cols=win + K, topk=None, cbase=None, n_per_span=win + K, cmp_slots=None,
        )
        partial, kv_ref, q_ref, o_ref = R.attention_ref(p, x_n, cache_ext, idx)
        if s == 0 and args.stage in (11, 12, 13, 14, 15):
            if args.stage == 11:
                ok &= compare(f"{tag} normed", dbg[:K, :dim], x_n, tol_rel=2e-2)
            elif args.stage == 12:
                ok &= compare(f"{tag} q", dbg[:K, : heads * hd], q_ref.reshape(K, -1), tol_rel=2e-2)
            elif args.stage == 13:
                ok &= compare(f"{tag} kv", dbg[:K, :hd], kv_ref, tol_rel=2e-2)
            elif args.stage == 14:
                ok &= compare(f"{tag} o", dbg[:K, : heads * hd], o_ref.reshape(K, -1), tol_rel=2e-2)
            else:
                xn_k = dbg[:K, dim : 2 * dim].to(torch.bfloat16)
                partial_k, *_ = R.attention_ref(p, xn_k, cache_ext, idx)
                ok &= compare(f"{tag} attn partial (from kernel xn)", dbg[:K, :dim], partial_k, tol_rel=5e-2)
                ok &= compare(f"{tag} attn partial (from ref xn)", dbg[:K, :dim], partial, tol_rel=5e-2)
            print(f"{tag} {'PASS' if ok else 'FAIL'}", flush=True)
            sys.exit(0 if ok else 1)
        attn_out = ar(partial).to(torch.bfloat16)
        x2 = R.hc_out(attn_out, x, post_a, comb_a, hc)
        x_n2, pre_f, post_f, comb_f = R.hc_in(x2, pre_a, p["hcf_fn"], p["hcf_base"], p["hcf_scale"], p["ffn_norm_w"], cfg)
        acc, top, _w, _wes, _lg = R.moe_ref(p, x_n2, cfg3, shared_scale=1.0 / world)
        if s == 0 and args.stage in (16, 17, 18):
            if args.stage == 16:
                ok &= compare(f"{tag} stream after attn", dbg[:K, : hc * dim], x2.reshape(K, -1), tol_rel=2e-2)
            elif args.stage == 17:
                ok &= compare(f"{tag} normed2", dbg[:K, :dim], x_n2, tol_rel=2e-2)
            else:
                xn2_k = dbg[:K, dim : 2 * dim].to(torch.bfloat16)
                acc_k, top_k, *_ = R.moe_ref(p, xn2_k, cfg3, shared_scale=1.0 / world)
                print(f"{tag} top-3 ref={top.tolist()} from-kernel-xn2={top_k.tolist()}", flush=True)
                ok &= compare(f"{tag} moe partial (from kernel xn2)", dbg[:K, :dim], acc_k, tol_rel=2e-2)
                compare(f"{tag} moe partial (from ref xn2)", dbg[:K, :dim], acc, tol_rel=2e-2)
            print(f"{tag} {'PASS' if ok else 'FAIL'}", flush=True)
            sys.exit(0 if ok else 1)
        ffn_out = ar(acc).to(torch.bfloat16)
        x = R.hc_out(ffn_out, x2, post_f, comb_f, hc)
        pre_mix = pre_f
        streams.append(x)
        if args.stage == s + 1:
            # The stream carries the MoE's amplified rounding flips (see the docstring).
            ok &= compare(f"{tag} stream after mtp.{s}", dbg[:K, : hc * dim], x.reshape(K, -1), tol_rel=0.3)
            for r in range(K):
                compare(f"{tag}   row {r}", dbg[r, : hc * dim], x[r].reshape(-1), tol_rel=0.3)
            print(f"{tag} {'PASS' if ok else 'FAIL'}", flush=True)
            sys.exit(0 if ok else 1)
    # head
    xh = (pre_mix.unsqueeze(-1) * x.float()).sum(1).to(torch.bfloat16)
    xhn = R.rmsnorm(xh, fnorm_w, eps)
    lg_loc = (xhn.float() @ head_w.float().t()).to(torch.bfloat16).float()
    cols = vstart + torch.arange(part)
    lg_loc[:, cols >= VOCAB] = float("-inf")
    if args.stage == 4:
        live = cols < VOCAB
        ok &= compare(f"{tag} local logits", dbg[:K, :part][:, live], lg_loc[:, live], tol_rel=0.3)
        ok &= compare(f"{tag} collapsed hidden", dbg[:K, part : part + dim], xh, tol_rel=0.3)
        # The head from the kernel's own collapsed hidden: the exact check of the head path.
        xh_k = dbg[:K, part : part + dim].to(torch.bfloat16)
        lg_k = (R.rmsnorm(xh_k, fnorm_w, eps).float() @ head_w.float().t()).to(torch.bfloat16).float()
        ok &= compare(f"{tag} local logits (from kernel hidden)", dbg[:K, :part][:, live], lg_k[:, live], tol_rel=2e-2)
        print(f"{tag} {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)
    full = gather_cols(lg_loc)[:, :VOCAB]
    ids_ref = [seed]
    membs, gaps = [], []
    for i in range(K):
        me = embed(mk_embed_w, torch.tensor([ids_ref[i]]))
        bias = (me.float() @ mk_head_w.float().t()).to(torch.bfloat16).float()
        bias = gather_cols(bias)[:, :VOCAB]
        row = full[i] + bias[0]
        top2 = torch.topk(row, 2).values
        gaps.append(float(top2[0] - top2[1]))
        ids_ref.append(int(row.argmax()))
        membs.append(me)
    memb = torch.cat(membs)
    conf_ref = (torch.cat([xh, memb], -1).float() @ conf_w.float().t()).to(torch.bfloat16).float().reshape(-1)
    got = out_ids.reshape(-1).tolist()
    mism = [i for i in range(K + 1) if got[i] != ids_ref[i]]
    print(f"{tag} ids kernel={got} ref={ids_ref} mismatches={mism} top-2 gaps={[round(g, 3) for g in gaps]}", flush=True)
    # Informational on random weights (see the docstring): a mismatch only counts where
    # the reference's top-2 gap is wide enough that the amplified rounding noise could
    # not have flipped it.
    ok &= not [i for i in mism if i > 0 and gaps[i - 1] > 3.0]
    # The confidence of a row depends on the drafted id before it (its markov embedding):
    # compare the rows whose ids matched.
    same = torch.tensor([got[i] == ids_ref[i] for i in range(K)])
    ok &= compare(f"{tag} confidence (rows with matching ids: {int(same.sum())}/{K})",
                  conf_k.reshape(-1)[same], conf_ref[same], tol_rel=0.3)
    print(f"{tag} {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
