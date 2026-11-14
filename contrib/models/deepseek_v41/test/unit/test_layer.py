# SPDX-License-Identifier: Apache-2.0
"""One layer kernel (`kernels/layer_kernel`, any archetype, either phase) against the
CPU reference, on N ranks.

    python kharness_mp.py 8 test_layer.py --arch window --phase decode --T 6 --stage 9
    python kharness_mp.py 8 test_layer.py --arch c2_source --phase prefill --T 512

Every rank builds its own random weights (seeded by rank), the attention group is all
N ranks and so is the MoE world (rank r owns experts [6r, 6r+6)), the reference
partials are reduced with Gloo, and each rank compares the kernel's output with the
reduced reference.
"""

import argparse
import inspect
import sys

if __name__ == "__main__":
    from kharness_mp import init_rank

    rank, world = init_rank()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from kharness import compare, run_nki  # noqa: E402
from deepseek_v41.config import DeepseekV41Config  # noqa: E402
from deepseek_v41.kernels import layer_kernel  # noqa: E402
from deepseek_v41 import mk_host, model as M  # noqa: E402
from deepseek_v41.weight_loader import dequant_block_fp8  # noqa: E402
import ref_layer as R  # noqa: E402
from packs import fp4_pack, fp8_pack  # noqa: E402


def kb16(w, s):
    """An fp8 (weight, scale) pair as the kernels take it: dequantized, bf16, k-major."""
    return dequant_block_fp8(w, s, 32, torch.float32).t().contiguous().to(torch.bfloat16)


def kmajor4(ws):
    """Stacked fp4 experts `[E, n, k']` -> the k-major `[E, k', n]` the kernels read."""
    return ws.transpose(1, 2).contiguous()


def make_params(cfg, r, world, heads, ngroups, n_local, max_pos):
    g = torch.Generator().manual_seed(1000 + r)
    dim, hd, ql, ol, inter = cfg.hidden_size, cfg.head_dim, cfg.q_lora_rank, cfg.o_lora_rank, cfg.moe_intermediate_size
    hc = cfg.hc_mult
    rn = lambda *s, sd=1.0: (torch.randn(*s, generator=g) * sd)  # noqa: E731
    p = {}
    for name in ("hca", "hcf"):
        p[f"{name}_fn"] = rn(M._mix_width(hc), hc * dim, sd=0.02)
        p[f"{name}_base"] = rn(M._mix_width(hc), sd=0.5)
        p[f"{name}_scale"] = torch.tensor([1.0, 0.8, 1.2])
    p["attn_norm_w"] = (1 + 0.1 * rn(dim)).to(torch.bfloat16)
    p["ffn_norm_w"] = (1 + 0.1 * rn(dim)).to(torch.bfloat16)
    p["wq_a"], p["wq_a_s"] = fp8_pack(rn(ql, dim, sd=0.03).to(torch.bfloat16))
    p["q_norm_w"] = (1 + 0.1 * rn(ql)).to(torch.bfloat16)
    p["wq_b"], p["wq_b_s"] = fp8_pack(rn(heads * hd, ql, sd=0.03).to(torch.bfloat16))
    p["wkv"], p["wkv_s"] = fp8_pack(rn(hd, dim, sd=0.03).to(torch.bfloat16))
    p["kv_norm_w"] = (1 + 0.1 * rn(hd)).to(torch.bfloat16)
    hpg = heads // ngroups
    p["wo_a"], p["wo_a_s"] = fp8_pack(rn(ngroups * ol, hpg * hd, sd=0.02).to(torch.bfloat16))
    p["wo_b"], p["wo_b_s"] = fp8_pack(rn(dim, ngroups * ol, sd=0.02).to(torch.bfloat16))
    p["attn_sink"] = rn(heads, sd=0.5)
    cos, sin = M.yarn_freqs_cis(cfg.qk_rope_head_dim, max_pos, 0, cfg.rope_theta, cfg.rope_factor,
                                cfg.beta_fast, cfg.beta_slow, "cpu")
    p["freqs"] = torch.cat([cos, sin], -1).contiguous()
    p["gate_w"] = rn(cfg.n_routed_experts, dim, sd=0.02).to(torch.bfloat16)
    p["gate_b"] = rn(cfg.n_routed_experts, sd=0.1)
    p["expert_lo"] = r * n_local
    w1, s1, w3, s3, w2, s2 = [], [], [], [], [], []
    for _ in range(n_local):
        a, b = fp4_pack(rn(inter, dim, sd=0.03).to(torch.bfloat16)); w1.append(a); s1.append(b)
        a, b = fp4_pack(rn(inter, dim, sd=0.03).to(torch.bfloat16)); w3.append(a); s3.append(b)
        a, b = fp4_pack(rn(dim, inter, sd=0.03).to(torch.bfloat16)); w2.append(a); s2.append(b)
    p["w1"], p["s1"], p["w3"], p["s3"], p["w2"], p["s2"] = [torch.stack(t) for t in (w1, s1, w3, s3, w2, s2)]
    nh_idx = cfg.index_n_heads // world if world <= 8 else 4
    p["cm_wkv"] = rn(hd, dim, sd=0.03).to(torch.bfloat16)
    p["cm_wgate"] = rn(hd, dim, sd=0.03).to(torch.bfloat16)
    p["cm_norm_w"] = (1 + 0.1 * rn(hd)).to(torch.bfloat16)
    p["ix_wq_b"], p["ix_wq_b_s"] = fp8_pack(rn(nh_idx * 128, ql, sd=0.03).to(torch.bfloat16))
    p["ix_wproj"] = rn(nh_idx, dim, sd=0.03).to(torch.bfloat16)
    p["ix_wk"] = rn(128, hd, sd=0.05).to(torch.bfloat16)
    p["ix_knorm_w"] = (1 + 0.1 * rn(128)).to(torch.bfloat16)
    p["e_wkv"], p["e_wkv_s"] = fp8_pack(rn(dim * (hc + 1), 24 * 256, sd=0.03).to(torch.bfloat16))
    p["e_qk"] = (torch.ones(hc * dim) + 0.1 * rn(hc * dim)).float()
    p["sh_w1"], p["sh_s1"] = fp8_pack(rn(inter, dim, sd=0.03).to(torch.bfloat16))
    p["sh_w3"], p["sh_s3"] = fp8_pack(rn(inter, dim, sd=0.03).to(torch.bfloat16))
    p["sh_w2"], p["sh_s2"] = fp8_pack(rn(dim, inter, sd=0.03).to(torch.bfloat16))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="window", choices=sorted(layer_kernel.ARCH))
    ap.add_argument("--phase", default="decode", choices=("decode", "prefill"))
    ap.add_argument("--T", type=int, default=6)
    ap.add_argument("--spans", type=int, default=1)
    ap.add_argument("--start", type=int, default=200, help="start_pos of the (first) span")
    ap.add_argument("--stage", type=int, default=9)
    args = ap.parse_args()
    ratio, src, idx_src, use_engram, use_tap = layer_kernel.ARCH[args.arch]

    cfg = DeepseekV41Config()
    dim, hc, hd = cfg.hidden_size, cfg.hc_mult, cfg.head_dim
    heads = cfg.num_attention_heads // world if world <= 8 else 8
    ngroups = max(1, cfg.o_groups // world) if world <= 8 else 1
    n_local = 6
    win = cfg.sliding_window
    block_size = 32
    T, B = args.T, args.spans
    L = T // B
    max_pos = max(4096, args.start + 37 * B + L + 64)
    p = make_params(cfg, rank, world, heads, ngroups, n_local, max_pos)

    # One shared input stream and cache contents on every rank (seed 0).
    g0 = torch.Generator().manual_seed(0)
    x = (torch.randn(T, hc, dim, generator=g0) * 1.5).to(torch.bfloat16)
    pre_mix = torch.softmax(torch.randn(T, hc, generator=g0), -1)
    nblk = max(16, -(-(args.start + 37 * (B - 1) + L) // block_size))
    num_slots = max(64, B * nblk) * block_size
    cache = (torch.randn(num_slots, hd, generator=g0) * 0.3).to(torch.bfloat16)
    cache = M.act_quant_dequant(cache, 32)
    spans, tables, slots = [], [], []
    for b in range(B):
        start = args.start + 37 * b
        spans.append((b * L, L, start))
        table = torch.arange(b * nblk, (b + 1) * nblk)
        tables.append(table)
        slots.append(mk_host.slots_of(table, torch.arange(start, start + L), block_size))
    slot_mapping = torch.cat(slots)
    read_len = 0
    if ratio:
        compress_len = (args.start + 37 * (B - 1) + L) // ratio
        read_len = max(128, -(-compress_len // 128) * 128)
    idx = mk_host.build_attn_indices(spans, tables, slot_mapping, block_size, win, read_len, ratio)
    idx.window_cols = win
    idx.topk = None
    topk_k = torch.zeros(T, 1, dtype=torch.int32)
    ccache = (torch.randn(num_slots, hd, generator=g0) * 0.3).to(torch.bfloat16)
    ccache = M.fp4_quant_dequant(ccache, 16, scale_e4m3=True)
    if ratio:
        # A random valid selection per row: up to 512 distinct compressed positions below
        # the row's own compress_len, the rest empty (-1), as an index source would emit.
        topk = torch.full((T, cfg.index_topk), -1, dtype=torch.int64)
        for t in range(T):
            b = t // L
            clen = (spans[b][2] + t - b * L + 1) // ratio
            k = min(cfg.index_topk, clen)
            if k > 0:
                topk[t, :k] = torch.randperm(clen, generator=g0)[:k]
        idx.topk = topk
        topk_k = topk.to(torch.int32).contiguous()
    erow_width = 4096
    etable = (torch.randn(erow_width, 256, generator=torch.Generator().manual_seed(500 + rank)) * 0.5).to(torch.bfloat16)
    erow_start = torch.tensor([[rank * erow_width]], dtype=torch.int32)
    hash_ids = torch.randint(0, world * erow_width, (T, 24), generator=g0).to(torch.int32)
    x_in = x
    if use_engram:
        loc = hash_ids.long() - rank * erow_width
        okm = (loc >= 0) & (loc < erow_width)
        rows = etable[loc.clamp(0, erow_width - 1)] * okm.unsqueeze(-1).to(torch.bfloat16)
        rows = rows.reshape(T, -1).float()
        dist.all_reduce(rows)
        rows = rows.to(torch.bfloat16)
        ekv = R.linear_fp8(R.act_quant_t2(rows), p["e_wkv"], p["e_wkv_s"])
        key, value = ekv[:, : hc * dim].float().reshape(T, hc, dim), ekv[:, hc * dim :].float()
        hf = x.float()
        w = p["e_qk"].reshape(hc, dim)
        rstd = torch.rsqrt(hf.square().mean(-1, keepdim=True) + cfg.rms_norm_eps) * torch.rsqrt(key.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
        dot = (hf * w * key).sum(-1, keepdim=True) * rstd * dim ** -0.5
        gate = torch.sigmoid(torch.sign(dot) * torch.sqrt(dot.abs().clamp_min(1e-6)))
        x_in = (hf + gate * value.unsqueeze(1)).to(torch.bfloat16)
    kcache = (torch.randn(num_slots, 128, generator=g0) * 0.3).to(torch.bfloat16)
    kcache = M.fp4_quant_dequant(kcache, 32)
    pool_kv = torch.randn(num_slots, hd, generator=g0)
    pool_score = torch.randn(num_slots, hd, generator=g0)
    if idx_src:
        topk_k = torch.zeros(T, cfg.index_topk, dtype=torch.int32)

    # ---- CPU reference -------------------------------------------------------------
    x_n, pre_a, post_a, comb_a = R.hc_in(x_in, pre_mix, p["hca_fn"], p["hca_base"], p["hca_scale"], p["attn_norm_w"], cfg)
    ccache_in, kcache_in = ccache, kcache
    if idx_src:
        xq0 = R.act_quant_t2(x_n)
        qr0 = R.rmsnorm(R.linear_fp8(xq0, p["wq_a"], p["wq_a_s"]), p["q_norm_w"], 1e-20)
        if src:
            lat_ref, pool_kv_ref, pool_score_ref = R.compressor_ref(p, x_n, idx, ratio, pool_kv, pool_score, cfg)
            ccache, kcache, rot_ref, k_ref = R.source_caches_ref(p, lat_ref, idx, ccache, kcache, cfg)
        score_part = R.index_ref(p, x_n, qr0, idx, kcache, ratio, read_len, cfg, world)
        score_sum = score_part.clone()
        dist.all_reduce(score_sum)
        picks = R.topk_ref(score_sum, idx, cfg.index_topk)
        idx.topk = picks
    attn_part, kv_new, q_ref, o_ref = R.attention_ref(p, x_n, cache, idx, ccache)
    attn_sum = attn_part.clone()
    dist.all_reduce(attn_sum)
    attn_out = attn_sum.to(torch.bfloat16)
    stream2 = R.hc_out(attn_out, x_in, post_a, comb_a, hc)
    x_n2, pre_f, post_f, comb_f = R.hc_in(stream2, pre_a, p["hcf_fn"], p["hcf_base"], p["hcf_scale"], p["ffn_norm_w"], cfg)
    moe_part, top, w, we_ref, logits_ref = R.moe_ref(p, x_n2, cfg, shared_scale=1.0 / world)
    moe_sum = moe_part.clone()
    dist.all_reduce(moe_sum)
    ffn_out = moe_sum.to(torch.bfloat16)
    out_ref = R.hc_out(ffn_out, stream2, post_f, comb_f, hc)
    if rank == 0:
        print(f"[rank 0] reference built: arch={args.arch} phase={args.phase} T={T} B={B} "
              f"heads={heads} read_len={read_len}", flush=True)

    # ---- kernel --------------------------------------------------------------------
    groups = mk_host.replica_groups(world, world)
    kernel = layer_kernel.KERNELS[(args.arch, args.phase)]
    cache_dev = cache.to("neuron:0")
    ccache_dev = ccache_in.to("neuron:0")
    kcache_dev = kcache_in.to("neuron:0")
    pkv_dev, psc_dev = pool_kv.to("neuron:0"), pool_score.to("neuron:0")
    named = dict(
        stream=x, pre_mix=pre_mix,
        hca_fn_t=p["hca_fn"].t().contiguous(), hca_base=p["hca_base"].reshape(1, -1),
        hca_scale=p["hca_scale"].reshape(1, 3), attn_norm_w=p["attn_norm_w"].reshape(1, -1),
        hcf_fn_t=p["hcf_fn"].t().contiguous(), hcf_base=p["hcf_base"].reshape(1, -1),
        hcf_scale=p["hcf_scale"].reshape(1, 3), ffn_norm_w=p["ffn_norm_w"].reshape(1, -1),
        wq_a=kb16(p["wq_a"], p["wq_a_s"]), q_norm_w=p["q_norm_w"].reshape(1, -1),
        wq_b=kb16(p["wq_b"], p["wq_b_s"]), wkv=kb16(p["wkv"], p["wkv_s"]),
        kv_norm_w=p["kv_norm_w"].reshape(1, -1),
        wo_a=kb16(p["wo_a"], p["wo_a_s"]), wo_b=kb16(p["wo_b"], p["wo_b_s"]),
        attn_sink=p["attn_sink"].reshape(-1, 1), freqs=p["freqs"],
        row_pos=idx.row_pos, write_slots=idx.write_slots, win_slots=idx.win_slots,
        twin=idx.twin, wv=idx.wv, window_kv_cache=cache_dev,
        cmp_slots=idx.cmp_slots, cbase=idx.cbase, topk_in=topk_k, compress_kv_cache=ccache_dev,
        pool_kv=pkv_dev, pool_score=psc_dev,
        cm_wkv=p["cm_wkv"].t().contiguous(), cm_wkv_t=p["cm_wkv"].t().float().contiguous(),
        cm_wgate_t=p["cm_wgate"].t().float().contiguous(), cm_norm_w=p["cm_norm_w"].reshape(1, -1),
        grp_pos=idx.grp_pos, grp_mslots=idx.grp_mslots, grp_wslots=idx.grp_wslots,
        ix_wk=p["ix_wk"].t().contiguous(), ix_knorm_w=p["ix_knorm_w"].reshape(1, -1),
        index_k_cache=kcache_dev, ix_wq_b=kb16(p["ix_wq_b"], p["ix_wq_b_s"]),
        ix_wproj_t=p["ix_wproj"].t().float().contiguous(), clen=idx.clen,
        hash_ids=hash_ids, etable=etable, erow_start=erow_start,
        e_wkv=kb16(p["e_wkv"], p["e_wkv_s"]), e_qk=p["e_qk"].reshape(1, -1),
        gate_w_t=p["gate_w"].t().float().contiguous(), gate_b=p["gate_b"].reshape(1, -1),
        expert_lo=torch.tensor([[float(p["expert_lo"])]]), live=torch.tensor([[float(T)]]),
        w1=kmajor4(p["w1"]), s1=kmajor4(p["s1"]), w3=kmajor4(p["w3"]), s3=kmajor4(p["s3"]),
        w2=kmajor4(p["w2"]), s2=kmajor4(p["s2"]),
        sh_w1=kb16(p["sh_w1"], p["sh_s1"]), sh_w3=kb16(p["sh_w3"], p["sh_s3"]),
        sh_w2=kb16(p["sh_w2"], p["sh_s2"]),
        groups_attn=groups, groups_world=groups, flags=(B, win, args.stage, 0),
    )
    names = list(inspect.signature(kernel.func if hasattr(kernel, "func") else kernel).parameters)
    out = run_nki(kernel, *[named[n] for n in names], grid=2)
    res = list(out)
    out_k, ffn_pre_k = res[0], res[1]
    k = 3
    if src:
        k += 4 if ratio == 2 else 2
    topk_k_out = res[k] if idx_src else None
    tap_k = res[k + int(idx_src)] if use_tap else None
    cache_after = cache_dev.cpu()
    st = args.stage
    ok = True
    tag = f"[rank {rank} {args.arch}/{args.phase} T={T} B={B} stage={st}]"
    if src and st >= 3:
        ccache_after, kcache_after = ccache_dev.cpu(), kcache_dev.cpu()
        gw = idx.grp_wslots.reshape(-1).long()
        live = gw != 0
        ok &= compare(f"{tag} compressed latents written", ccache_after[gw[live]], rot_ref[live], tol_rel=5e-2)
        ok &= compare(f"{tag} index keys written", kcache_after[gw[live]], k_ref[live], tol_rel=5e-2)
        if ratio == 2:
            ws = idx.write_slots.reshape(-1).long()
            ok &= compare(f"{tag} pool_kv written", pkv_dev.cpu()[ws], pool_kv_ref[ws], tol_rel=1e-3)
            ok &= compare(f"{tag} pool_score written", psc_dev.cpu()[ws], pool_score_ref[ws], tol_rel=1e-3)
    if idx_src and st >= 4:
        pk = topk_k_out.long()
        agree = [len(set(pk[t].tolist()) & set(picks[t].tolist())) / max(1, (picks[t] >= 0).sum().item()) for t in range(T)]
        print(f"{tag} top-k set agreement per row (kernel vs torch fp32 top-k): min={min(agree):.3f} mean={sum(agree)/len(agree):.3f}", flush=True)
        ok &= min(agree) > 0.95  # bf16 top-k against an fp32 one: near-ties differ
    if use_tap:
        ok &= compare(f"{tag} tap", tap_k, x_in.float().mean(1), tol_rel=1e-3)
    def tile_report(name, got, ref):
        if T <= 128:
            return
        g, r = got.float().reshape(T, -1), ref.float().reshape(T, -1)
        rels = []
        for t0 in range(0, T, 128):
            e = (g[t0 : t0 + 128] - r[t0 : t0 + 128]).norm() / r[t0 : t0 + 128].norm().clamp_min(1e-30)
            rels.append(f"{float(e):.1e}")
        print(f"{tag} {name} per-tile rel: {' '.join(rels)}", flush=True)

    if st == 2:
        tile_report("q", out_k[:T, : heads * hd], q_ref.reshape(T, -1))
    elif st == 3:
        tile_report("kv", out_k[:T, :hd], kv_new)
    elif st == 7:
        tile_report("normed2", out_k[:T, :dim], x_n2)
    elif st == 8:
        tile_report("ffn_out", out_k[:T, :dim], ffn_out)
    if st == 31:
        xk = out_k[:T, :dim].to(torch.bfloat16)
        ok &= compare(f"{tag} xn", xk, x_n, tol_rel=2e-2)
        kvr_ref = R.linear_fp8(R.act_quant_t2(xk), p["wkv"], p["wkv_s"])
        ok &= compare(f"{tag} kvr vs ref(kernel xn)", out_k[:T, dim + hd : dim + 2 * hd], kvr_ref, tol_rel=2e-2)
        kvr_k = out_k[:T, dim + hd : dim + 2 * hd].to(torch.bfloat16)
        cos = p["freqs"][idx.row_pos.reshape(-1).long(), :32]
        sin = p["freqs"][idx.row_pos.reshape(-1).long(), 32:]
        kv2 = R.rmsnorm(kvr_k, p["kv_norm_w"], 1e-20)
        kv2 = torch.cat([kv2[..., :-64], R.rope(kv2[..., -64:], cos, sin)], -1)
        kv2 = M.act_quant_dequant(kv2, 32)
        ok &= compare(f"{tag} kv vs ref(kernel kvr)", out_k[:T, dim : dim + hd], kv2, tol_rel=2e-2)
        tile_report("kv vs ref(kernel kvr)", out_k[:T, dim : dim + hd], kv2)
    if st == 82:
        xn2_k = out_k[:T, dim : 2 * dim].to(torch.bfloat16)
        ok &= compare(f"{tag} xn2(kernel) vs xn2(cpu)", xn2_k, x_n2, tol_rel=3e-2)
        ref_k, top_k, *_ = R.moe_ref(p, xn2_k, cfg, shared_scale=1.0 / world)
        same = torch.equal(torch.sort(top_k, -1).values, torch.sort(top, -1).values)
        nflip = int((torch.sort(top_k, -1).values != torch.sort(top, -1).values).any(-1).sum())
        print(f"{tag} routing from kernel xn2 identical to cpu routing: {same} ({nflip} rows differ)", flush=True)
        ok &= compare(f"{tag} moe partial vs ref(kernel xn2)", out_k[:T, :dim], ref_k, tol_rel=1e-2)
        tile_report("moe partial vs ref(kernel xn2)", out_k[:T, :dim], ref_k)
    if st == 41:
        # the attention alone: the kernel's own q against the kernel's cache contents
        qk = out_k[:T, : heads * hd].to(torch.bfloat16).reshape(T, heads, hd)
        ok_ = out_k[:T, heads * hd : 2 * heads * hd]
        ok &= compare(f"{tag} q", qk.reshape(T, -1), q_ref.reshape(T, -1), tol_rel=2e-2)
        Bn = idx.win_slots.shape[0] // idx.nwin
        parts = []
        # the kernel's own compressed rows and selection (a source layer writes and picks them)
        ccache_k = ccache_dev.cpu()
        sel = topk_k_out.long() if idx_src else (idx.topk if idx.topk is not None else None)
        for b in range(Bn):
            parts.append(cache_after[idx.win_slots[b * idx.nwin : (b + 1) * idx.nwin].reshape(-1).long()])
            if sel is not None:
                rl = idx.n_per_span - idx.nwin
                parts.append(ccache_k[idx.cmp_slots[b * rl : (b + 1) * rl].reshape(-1).long()])
        table = torch.cat(parts)
        o2 = torch.zeros(T, heads, hd)
        sink = p["attn_sink"].float().reshape(heads)
        for t in range(T):
            wi = idx.twin[t].item() + torch.arange(win)
            valid = idx.wv[t].item() + torch.arange(win) >= 0
            ids = torch.where(valid, wi, torch.full_like(wi, -1))
            if sel is not None:
                tk = sel[t].long()
                ids = torch.cat([ids, torch.where(tk >= 0, tk + idx.cbase[t].item(), torch.full_like(tk, -1))])
            gtab = table[ids.clamp_min(0)].float()
            sc = qk[t].float() @ gtab.t() * hd ** -0.5 + torch.where(ids < 0, -1e30, 0.0).unsqueeze(0)
            mx = sc.max(-1, keepdim=True).values
            pr = torch.exp(sc - mx)
            den = pr.sum(-1, keepdim=True) + torch.exp(sink.unsqueeze(1) - mx)
            o2[t] = (pr.to(torch.bfloat16).float() @ gtab) / den
        ok &= compare(f"{tag} o vs ref(kernel q, kernel cache)", ok_, o2.to(torch.bfloat16).reshape(T, -1), tol_rel=1e-2)
        tile_report("o vs ref(kernel q)", ok_, o2.to(torch.bfloat16).reshape(T, -1))
    if st == 1:
        ok &= compare(f"{tag} normed", out_k[:T, :dim], x_n, tol_rel=2e-2)
    elif st == 2:
        ok &= compare(f"{tag} q", out_k[:T, : heads * hd], q_ref.reshape(T, -1), tol_rel=2e-2)
    elif st == 3:
        ok &= compare(f"{tag} kv", out_k[:T, :hd], kv_new, tol_rel=2e-2)
    elif st == 4:
        ok &= compare(f"{tag} o", out_k[:T, : heads * hd], o_ref.reshape(T, -1), tol_rel=2e-2)
    elif st == 5:
        ok &= compare(f"{tag} attn_out", out_k[:T, :dim], attn_out, tol_rel=5e-2)
    elif st == 6:
        ok &= compare(f"{tag} stream2", out_k[:T, : hc * dim], stream2.reshape(T, -1), tol_rel=2e-2)
    elif st == 7:
        ok &= compare(f"{tag} normed2", out_k[:T, :dim], x_n2, tol_rel=2e-2)
    elif st == 8:
        ok &= compare(f"{tag} ffn_out", out_k[:T, :dim], ffn_out, tol_rel=5e-2)
    elif st == 9:
        ok &= compare(f"{tag} stream_out", out_k, out_ref, tol_rel=5e-2 if args.phase == "decode" else 1.5e-1)
        ok &= compare(f"{tag} ffn_pre", ffn_pre_k, pre_f, tol_rel=5e-3)
    if st >= 3:
        ws = idx.write_slots.reshape(-1).long()
        ok &= compare(f"{tag} cache rows written", cache_after[ws], kv_new, tol_rel=2e-2)
        untouched = torch.ones(num_slots, dtype=torch.bool)
        untouched[ws] = False
        ok &= compare(f"{tag} cache rows untouched", cache_after[untouched], cache[untouched], tol_rel=1e-6)
    print(f"{tag} {'PASS' if ok else 'FAIL'}", flush=True)
    dist.all_reduce(torch.zeros(1))  # a barrier; dist.barrier() trips the Neuron backend hook
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
