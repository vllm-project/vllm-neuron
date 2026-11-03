# SPDX-License-Identifier: Apache-2.0
"""The layer megakernel against the CPU reference, on N ranks.

    python kharness_mp.py 8 test_layer.py --T 6 --stage 9

Every rank builds its own random weights (seeded by rank), the attention group
is all N ranks and so is the MoE world (rank r owns experts [6r, 6r+6)), the
reference partials are reduced with Gloo, and each rank compares the kernel's
output with the reduced reference.
"""

import argparse
import os
import sys

if __name__ == "__main__":
    from kharness_mp import init_rank

    rank, world = init_rank()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from kharness import compare, run_nki  # noqa: E402
from deepseek_v41.config import DeepseekV41Config  # noqa: E402
from deepseek_v41.kernels import mk_kernels  # noqa: E402
from deepseek_v41 import mk_host, mk_model, model as M  # noqa: E402
import ref_layer as R  # noqa: E402


def mx8(w, s):
    """An fp8 (weight, scale) pair in the kernels' MX layout."""
    return mk_model.mx_fp8(w, s)


def mx4_stack(ws, ss):
    """Stacked fp4 experts `[E, ...]` in the MX layout."""
    out_w, out_s = [], []
    for e in range(ws.shape[0]):
        w, s_ = mk_model.mx_fp4(ws[e], ss[e])
        out_w.append(w)
        out_s.append(s_)
    return torch.stack(out_w), torch.stack(out_s)


def fp8_pack(w: torch.Tensor, block=32):
    """bf16 weight -> (uint8 e4m3fn, uint8 e8m0 scales per 32x32 block)."""
    n, k = w.shape
    wb = w.float().reshape(n // block, block, k // block, block)
    amax = wb.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-4)
    e = torch.ceil(torch.log2(amax / 448.0))
    scale = torch.exp2(e)
    q = (wb / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (
        q.reshape(n, k).view(torch.uint8).contiguous(),
        (e.reshape(n // block, k // block) + 127).to(torch.uint8).contiguous(),
    )


def fp4_pack(w: torch.Tensor, block=32):
    """bf16 weight -> (uint8 packed e2m1 pairs, uint8 e8m0 scale per row per 32)."""
    n, k = w.shape
    wb = w.float().reshape(n, k // block, block)
    amax = wb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    e = torch.ceil(torch.log2(amax / 6.0))
    scale = torch.exp2(e)
    x = (wb / scale).clamp(-6, 6)
    mags = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    mids = (mags[1:] + mags[:-1]) / 2
    code = torch.bucketize(x.abs(), mids)
    code = torch.where(x < 0, code + 8, code).reshape(n, k).to(torch.uint8)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, (e.reshape(n, k // block) + 127).to(torch.uint8).contiguous()


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
    ap.add_argument("--T", type=int, default=6)
    ap.add_argument("--spans", type=int, default=1)
    ap.add_argument("--start", type=int, default=200, help="start_pos of the (first) span")
    ap.add_argument("--stage", type=int, default=9)
    ap.add_argument("--ratio", type=int, default=0, help="compressed-attention consumer at this ratio")
    ap.add_argument("--src", action="store_true", help="kv + index source layer (compressor, indexer, top-k)")
    ap.add_argument("--engram", action="store_true", help="engram layer (n-gram table read-out before the block)")
    ap.add_argument("--tap", action="store_true", help="emit the DSpark tap")
    ap.add_argument("--n-local", type=int, default=6, help="routed experts a rank (6: tp64, 48: a tp8pp8 stage)")
    ap.add_argument("--scheme", default="tp64", help="tp64 (the generic entry) or tp8pp8 (the stage kernel)")
    ap.add_argument("--phase", default=None, help="prefill / decode entry of the scheme (default by T)")
    args = ap.parse_args()

    cfg = DeepseekV41Config()
    dim, hc, hd = cfg.hidden_size, cfg.hc_mult, cfg.head_dim
    heads = cfg.num_attention_heads // world if world <= 8 else 8
    ngroups = max(1, cfg.o_groups // world) if world <= 8 else 1
    n_local = args.n_local
    win = cfg.sliding_window
    block_size = 32
    max_pos = 4096
    T, B = args.T, args.spans
    L = T // B
    p = make_params(cfg, rank, world, heads, ngroups, n_local, max_pos)

    # One shared input stream and cache contents on every rank (seed 0).
    g0 = torch.Generator().manual_seed(0)
    x = (torch.randn(T, hc, dim, generator=g0) * 1.5).to(torch.bfloat16)
    pre_mix = torch.softmax(torch.randn(T, hc, generator=g0), -1)
    # Enough blocks per span for its positions (at least 16 = 512 positions).
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
    ratio = args.ratio
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
    src = args.src
    erow_width = 4096
    etable = (torch.randn(erow_width, 256, generator=torch.Generator().manual_seed(500 + rank)) * 0.5).to(torch.bfloat16)
    erow_start = torch.tensor([[rank * erow_width]], dtype=torch.int32)
    hash_ids = torch.randint(0, world * erow_width, (T, 24), generator=g0).to(torch.int32)
    x_in = x
    if args.engram:
        # reference: gather this rank's rows, reduce, fp8 wkv, gate
        loc = hash_ids.long() - rank * erow_width
        okm = (loc >= 0) & (loc < erow_width)
        rows = etable[loc.clamp(0, erow_width - 1)] * okm.unsqueeze(-1).to(torch.bfloat16)
        rows = rows.reshape(T, -1).float()
        dist.all_reduce(rows)
        rows = rows.to(torch.bfloat16)
        ekv = R.linear_fp8(R.act_quant_mx(rows), p["e_wkv"], p["e_wkv_s"])
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
    if src:
        assert ratio, "--src needs --ratio"
        topk_k = torch.zeros(T, cfg.index_topk, dtype=torch.int32)

    # ---- CPU reference -------------------------------------------------------------
    x_n, pre_a, post_a, comb_a = R.hc_in(x_in, pre_mix, p["hca_fn"], p["hca_base"], p["hca_scale"], p["attn_norm_w"], cfg)
    ccache_in, kcache_in = ccache, kcache
    if src:
        xq0 = R.act_quant_mx(x_n)
        qr0 = R.rmsnorm(R.linear_fp8(xq0, p["wq_a"], p["wq_a_s"]), p["q_norm_w"], 1e-20)
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
    moe_routed, *_ = R.moe_ref(p, x_n2, cfg, shared=False)
    moe_sum = moe_part.clone()
    dist.all_reduce(moe_sum)
    ffn_out = moe_sum.to(torch.bfloat16)
    out_ref = R.hc_out(ffn_out, stream2, post_f, comb_f, hc)
    if rank == 0:
        print(f"[rank 0] reference built: T={T} B={B} heads={heads} groups={ngroups} top={top[0].tolist()}", flush=True)

    # ---- kernel --------------------------------------------------------------------
    groups = mk_host.replica_groups(world, world)
    flags = (ratio, B, win, args.stage, int(src), int(src), int(args.engram), int(args.tap), 0)
    cache_dev = cache.to("neuron:0")
    ccache_dev = ccache_in.to("neuron:0")
    kcache_dev = kcache_in.to("neuron:0")
    pkv_dev, psc_dev = pool_kv.to("neuron:0"), pool_score.to("neuron:0")
    G = idx.groups_per_span if ratio else 0
    one = torch.zeros(1, 1, dtype=torch.int32)
    args_k = [
        x, pre_mix,
        p["hca_fn"].t().contiguous(), p["hca_base"].reshape(1, -1), p["hca_scale"].reshape(1, 3), p["attn_norm_w"].reshape(1, -1),
        p["hcf_fn"].t().contiguous(), p["hcf_base"].reshape(1, -1), p["hcf_scale"].reshape(1, 3), p["ffn_norm_w"].reshape(1, -1),
        *mx8(p["wq_a"], p["wq_a_s"]), p["q_norm_w"].reshape(1, -1), *mx8(p["wq_b"], p["wq_b_s"]), *mx8(p["wkv"], p["wkv_s"]), p["kv_norm_w"].reshape(1, -1),
        *mx8(p["wo_a"], p["wo_a_s"]), *mx8(p["wo_b"], p["wo_b_s"]), p["attn_sink"].reshape(-1, 1), p["freqs"],
        idx.row_pos, idx.write_slots, idx.win_slots, idx.cmp_slots, idx.twin, idx.wv, idx.cbase, topk_k,
        cache_dev, ccache_dev,
        pkv_dev, psc_dev, kcache_dev,
        p["cm_wkv"], p["cm_wkv"].t().float().contiguous(), p["cm_wgate"].t().float().contiguous(), p["cm_norm_w"].reshape(1, -1),
        idx.grp_pos if src else one, idx.grp_mslots if src else one, idx.grp_wslots if src else one,
        *mx8(p["ix_wq_b"], p["ix_wq_b_s"]), p["ix_wproj"].t().float().contiguous(), p["ix_wk"], p["ix_knorm_w"].reshape(1, -1),
        idx.clen if ratio else one,
        hash_ids, etable, erow_start, *mx8(p["e_wkv"], p["e_wkv_s"]), p["e_qk"].reshape(1, -1),
        p["gate_w"].t().float().contiguous(), p["gate_b"].reshape(1, -1), torch.tensor([[float(p["expert_lo"])]]),
        *mx4_stack(p["w1"], p["s1"]), *mx4_stack(p["w3"], p["s3"]), *mx4_stack(p["w2"], p["s2"]),
        *mx8(p["sh_w1"], p["sh_s1"]), *mx8(p["sh_w3"], p["sh_s3"]), *mx8(p["sh_w2"], p["sh_s2"]),
        groups, groups, flags,
    ]
    if args.scheme == "tp64" and args.phase is None:
        kernel = mk_kernels.pkg(args.scheme).layer_kernel.layer_forward
    else:
        phase = args.phase or ("decode" if L <= 6 else "prefill")
        kernel = mk_kernels.layer(phase, (ratio, int(src), int(src), int(args.engram), int(args.tap)), args.scheme)
    out, ffn_pre_k, topk_k_out, tap_k, *_ = run_nki(kernel, *args_k)
    cache_after = cache_dev.cpu()
    st = args.stage
    ok = True
    tag = f"[rank {rank} T={T} B={B} stage={st}]"
    if src and st >= 3:
        ccache_after, kcache_after = ccache_dev.cpu(), kcache_dev.cpu()
        gw = idx.grp_wslots.reshape(-1).long()
        live = gw != 0
        ok &= compare(f"{tag} compressed latents written", ccache_after[gw[live]], rot_ref[live], tol_rel=2e-2)
        ok &= compare(f"{tag} index keys written", kcache_after[gw[live]], k_ref[live], tol_rel=2e-2)
        if ratio == 2:
            ws = idx.write_slots.reshape(-1).long()
            ok &= compare(f"{tag} pool_kv written", pkv_dev.cpu()[ws], pool_kv_ref[ws], tol_rel=1e-3)
            ok &= compare(f"{tag} pool_score written", psc_dev.cpu()[ws], pool_score_ref[ws], tol_rel=1e-3)
        pk = topk_k_out.long()
        agree = [len(set(pk[t].tolist()) & set(picks[t].tolist())) / max(1, (picks[t] >= 0).sum().item()) for t in range(T)]
        print(f"{tag} top-k set agreement per row (kernel vs torch fp32 top-k): min={min(agree):.3f} mean={sum(agree)/len(agree):.3f}", flush=True)
        ok &= min(agree) > 0.97
    if args.tap:
        # The tap is the mean over the hc copies of the kernel's own (engram) stream.
        tap_ref = out[:T, : hc * dim].reshape(T, hc, dim).mean(1) if st == 0 else x_in.float().mean(1)
        ok &= compare(f"{tag} tap", tap_k, tap_ref, tol_rel=1e-5 if st == 0 else 1e-3)
    if st == 0:
        ok &= compare(f"{tag} engram stream", out[:T, : hc * dim], x_in.reshape(T, -1), tol_rel=2e-2)
    elif st == 1:
        ok &= compare(f"{tag} normed", out[:T, :dim], x_n, tol_rel=2e-2)
    elif st == 2:
        ok &= compare(f"{tag} q", out[:T, : heads * hd], q_ref.reshape(T, -1), tol_rel=2e-2)
    elif st == 3:
        ok &= compare(f"{tag} kv", out[:T, :hd], kv_new, tol_rel=2e-2)
    elif st == 4:
        ok &= compare(f"{tag} o", out[:T, : heads * hd], o_ref.reshape(T, -1), tol_rel=2e-2)
        ok_ = out[:T, : heads * hd].reshape(T, heads, hd).float()
        or_ = o_ref.float()
        err = (ok_ - or_).norm(dim=-1) / or_.norm(dim=-1).clamp_min(1e-9)   # [T, heads]
        print(f"{tag} per-(token,head) rel err:\n" + "\n".join(
            "  t%d: " % t + " ".join("%.1e" % v for v in err[t].tolist()) for t in range(T)), flush=True)
        print(f"{tag} sink={p['attn_sink'].tolist()}", flush=True)
    elif st == 51:
        ok &= compare(f"{tag} attn_part(f32) vs ref(cpu xn)", out[:T, :dim], attn_part, tol_rel=5e-2)
        xn_k = out[:T, dim : 2 * dim].to(torch.bfloat16)
        ok &= compare(f"{tag} xn(kernel) vs xn(cpu)", xn_k, x_n, tol_rel=1e-2)
        part_k, *_ = R.attention_ref(p, xn_k, cache, idx, ccache)
        compare(f"{tag} attn_part(f32) vs ref(kernel xn)", out[:T, :dim], part_k, tol_rel=1e-2)
        o_k = out[:T, 2 * dim : 2 * dim + heads * hd].to(torch.bfloat16).reshape(T, heads, hd)
        compare(f"{tag} o(kernel) vs o(ref from kernel xn)", o_k.reshape(T, -1), R.attention_ref(p, xn_k, cache, idx, ccache)[3].reshape(T, -1), tol_rel=1e-2)
        part_o, _ = R.oproj_ref(p, o_k, idx)
        ok &= compare(f"{tag} attn_part(f32) vs ref(kernel o)", out[:T, :dim], part_o, tol_rel=1e-2)
    elif st == 5:
        ok &= compare(f"{tag} attn_out", out[:T, :dim], attn_out, tol_rel=5e-2)
    elif st == 6:
        ok &= compare(f"{tag} stream2", out[:T, : hc * dim], stream2.reshape(T, -1), tol_rel=2e-2)
    elif st == 7:
        ok &= compare(f"{tag} normed2", out[:T, :dim], x_n2, tol_rel=2e-2)
    elif st == 81:
        ok &= compare(f"{tag} wexp", out[:T, :n_local], we_ref, tol_rel=1e-3)
        # the logits inherit xn2's drift from the attention path (fp8 re-quantization flips)
        ok &= compare(f"{tag} logits", out[:T, n_local : n_local + cfg.n_routed_experts], logits_ref, tol_rel=2e-2)
    elif st in (82, 83, 84):
        xn2_k = out[:T, dim : 2 * dim].to(torch.bfloat16)
        ok &= compare(f"{tag} xn2(kernel) vs xn2(cpu)", xn2_k, x_n2, tol_rel=3e-2)   # inherits the attention path's drift
        ref_k, top_k, *_ = R.moe_ref(p, xn2_k, cfg, shared=(st != 83), routed=(st != 84), shared_scale=1.0 / world)
        same = torch.equal(torch.sort(top_k, -1).values, torch.sort(top, -1).values)
        print(f"{tag} routing identical from kernel xn2: {same}")
        ok &= compare(f"{tag} moe stage {st} vs ref(kernel xn2)", out[:T, :dim], ref_k, tol_rel=1e-2)
    elif st == 8:
        ok &= compare(f"{tag} ffn_out", out[:T, :dim], ffn_out, tol_rel=5e-2)
    else:
        # Full-stream agreement on random weights: the fp8 MX GEMMs' internal rounding
        # (~2.4e-4 a GEMM) is amplified by every fp8 re-quantization downstream, so the
        # stream lands at 3-4e-2 (components match their own inputs far closer: stage 51 / 82).
        # (informational at larger T / denser expert worlds: the drift grows with the
        # context and with the share of the 384 experts a rank holds; the component
        # stages are the gates)
        ok &= compare(f"{tag} stream_out", out, out_ref, tol_rel=5e-2 if T <= 6 else 0.2)
        ok &= compare(f"{tag} ffn_pre", ffn_pre_k, pre_f, tol_rel=5e-3 if T <= 6 else 1e-2)
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
