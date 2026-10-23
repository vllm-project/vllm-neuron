# SPDX-License-Identifier: Apache-2.0
"""CPU torch reference of one layer, with the same arithmetic as the megakernel
(and as the shipped per-op path): fp8 activation round trips before the
quantized GEMMs, bf16 GEMM outputs, fp32 interiors, bf16 roundings where the
model has them. Partial sums (attention, MoE) are returned unreduced; the test
reduces them across ranks with Gloo.
"""

import torch
import torch.nn.functional as F

from deepseek_v41 import model as M
from deepseek_v41.weight_loader import dequant_block_fp4, dequant_block_fp8


def rmsnorm(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * w.float()).to(torch.bfloat16)


def linear_fp8(x_q, w, s):
    """`x_q` bf16 (already round-tripped) @ dequant(w, s).T, bf16 out."""
    return F.linear(x_q.float(), dequant_block_fp8(w, s, 32, torch.float32)).to(torch.bfloat16)


def linear_fp4(x_q, w, s):
    return F.linear(x_q.float(), dequant_block_fp4(w, s, 32, torch.float32)).to(torch.bfloat16)


def hc_in(x, pre_mix, fn, base, scale, norm_w, cfg):
    hc = cfg.hc_mult
    flat = x.float().reshape(x.shape[0], -1)
    rsq = torch.rsqrt(flat.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
    mixes = (flat @ fn.float().t()) * rsq
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + cfg.hc_eps
    post = 2 * torch.sigmoid(mixes[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (mixes[:, 2 * hc :] * scale[2] + base[2 * hc :]).reshape(-1, hc, hc)
    comb = torch.softmax(comb, -1) + cfg.hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + cfg.hc_eps)
    for _ in range(cfg.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + cfg.hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + cfg.hc_eps)
    y = (pre_mix.unsqueeze(-1) * x.float()).sum(1).to(torch.bfloat16)
    return rmsnorm(y, norm_w, cfg.rms_norm_eps), pre, post, comb.reshape(-1, hc * hc)


def hc_out(x, residual, post, comb, hc):
    """x [T, dim] bf16, residual [T, hc, dim] bf16 -> [T, hc, dim] bf16."""
    comb3 = comb.reshape(-1, hc, hc)
    broadcast = post.unsqueeze(-1) * x.float().unsqueeze(1)
    mixed = torch.einsum("tij,tid->tjd", comb3, residual.float())
    return (broadcast + mixed).to(torch.bfloat16)


def rope(x, cos, sin, inverse=False):
    """x [T, ..., 64] bf16 with cos/sin [T, 32] fp32 -> bf16 (adjacent pairs)."""
    if inverse:
        sin = -sin
    pairs = x.float().unflatten(-1, (-1, 2))
    a, b = pairs[..., 0], pairs[..., 1]
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [cos.shape[-1]]
    c, s_ = cos.reshape(shape), sin.reshape(shape)
    return torch.stack([a * c - b * s_, a * s_ + b * c], dim=-1).flatten(-2).to(x.dtype)


def attention_ref(p, x_n, cache, idx, ccache=None):
    """The attention sublayer for one rank: returns the fp32 `wo_b` partial [T, dim]
    and the new `kv` rows (what the kernel writes to the cache).

    `p` holds this rank's weights (dict), `x_n` the normalized input [T, dim] bf16,
    `cache` the flat KV cache [num_slots, 512] bf16 (already holding earlier rows),
    `idx` the `AttnIndices`, `pre_rows_rank` unused (kept for symmetry).
    """
    T = x_n.shape[0]
    hd, rd = 512, 64
    cos = p["freqs"][idx.row_pos.reshape(-1).long(), :32]
    sin = p["freqs"][idx.row_pos.reshape(-1).long(), 32:]
    xq = M.act_quant_dequant(x_n, 32)
    qa = linear_fp8(xq, p["wq_a"], p["wq_a_s"])
    qr = rmsnorm(qa, p["q_norm_w"], 1e-20)
    q = linear_fp8(M.act_quant_dequant(qr, 32), p["wq_b"], p["wq_b_s"])
    heads = q.shape[1] // hd
    q = q.reshape(T, heads, hd)
    q = torch.cat([q[..., :-rd], rope(q[..., -rd:], cos, sin)], -1)
    kv = linear_fp8(xq, p["wkv"], p["wkv_s"])
    kv = rmsnorm(kv, p["kv_norm_w"], 1e-20)
    kv = torch.cat([kv[..., :-rd], rope(kv[..., -rd:], cos, sin)], -1)
    kv = M.act_quant_dequant(kv, 32)
    cache = cache.clone()
    cache[idx.write_slots.reshape(-1).long()] = kv
    B = idx.win_slots.shape[0] // idx.nwin
    parts = []
    for b in range(B):
        parts.append(cache[idx.win_slots[b * idx.nwin : (b + 1) * idx.nwin].reshape(-1).long()])
        if idx.topk is not None:
            rl = idx.n_per_span - idx.nwin
            parts.append(ccache[idx.cmp_slots[b * rl : (b + 1) * rl].reshape(-1).long()])
    table = torch.cat(parts)
    # per-row slot indices into the table (the kernel's attn_index_table)
    rows = []
    for t in range(T):
        wi = idx.twin[t].item() + torch.arange(idx.window_cols)
        valid = idx.wv[t].item() + torch.arange(idx.window_cols) >= 0
        wi = torch.where(valid, wi, torch.full_like(wi, -1))
        if idx.topk is not None:
            tk = idx.topk[t].long()
            ci = torch.where(tk >= 0, tk + idx.cbase[t].item(), torch.full_like(tk, -1))
            wi = torch.cat([wi, ci])
        rows.append(wi)
    sidx = torch.stack(rows)
    scale = hd ** -0.5
    sink = p["attn_sink"].float().reshape(heads)
    o = torch.zeros(T, heads, hd, dtype=torch.bfloat16)
    for t in range(T):
        ids = sidx[t]
        g = table[ids.clamp_min(0)].float()                      # [k, 512]
        sc = q[t].float() @ g.t() * scale                        # [heads, k]
        sc = sc + torch.where(ids < 0, -1e30, 0.0).unsqueeze(0)
        mx = sc.max(-1, keepdim=True).values
        pr = torch.exp(sc - mx)
        den = pr.sum(-1, keepdim=True) + torch.exp(sink.unsqueeze(1) - mx)
        p16 = pr.to(torch.bfloat16).float()
        o[t] = ((p16 @ g) / den).to(torch.bfloat16)
    o_pre = o
    o = torch.cat([o[..., :-rd], rope(o[..., -rd:], cos, sin, True)], -1)
    oq = M.act_quant_dequant(o.reshape(T, heads * hd), 32)
    ngroups = p["wo_a"].shape[0] // 1024
    hpg = heads // ngroups
    zs = []
    for gi in range(ngroups):
        z = linear_fp8(oq[:, gi * hpg * hd : (gi + 1) * hpg * hd],
                       p["wo_a"][gi * 1024 : (gi + 1) * 1024], p["wo_a_s"][gi * 32 : (gi + 1) * 32])
        zs.append(z)
    z = torch.cat(zs, -1)
    zq = M.act_quant_dequant(z, 32)
    partial = F.linear(zq.float(), dequant_block_fp8(p["wo_b"], p["wo_b_s"], 32, torch.float32))
    return partial, kv, q, o_pre


def moe_ref(p, x_n, cfg, shared=True, routed=True, shared_scale=1.0):
    """This rank's MoE partial [T, dim] fp32: routed local experts + the shared expert
    scaled by `shared_scale` (the kernels add the replicated shared expert on every
    rank as 1/world of it, so the world sum carries it once, as the model does)."""
    T = x_n.shape[0]
    logits = x_n.float() @ p["gate_w"].float().t()
    scores = F.softplus(logits).sqrt()
    sel = scores + p["gate_b"].float()
    top = torch.topk(sel, cfg.num_experts_per_tok, dim=-1).indices
    w = scores.gather(1, top)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * cfg.routed_scaling_factor
    xq = M.act_quant_dequant(x_n, 32)
    acc = torch.zeros(T, x_n.shape[1], dtype=torch.float32)
    lo = int(p["expert_lo"])
    wes = []
    for e in range(p["w1"].shape[0]):
        we = ((top == lo + e).float() * w).sum(-1, keepdim=True)   # [T, 1]
        wes.append(we)
        if not routed:
            continue
        g = linear_fp4(xq, p["w1"][e], p["s1"][e]).float()
        u = linear_fp4(xq, p["w3"][e], p["s3"][e]).float()
        y = (F.silu(g.clamp(max=cfg.swiglu_limit)) * u.clamp(-cfg.swiglu_limit, cfg.swiglu_limit) * we)
        y = y.to(torch.bfloat16)
        yq = M.act_quant_dequant(y, 32)
        acc += linear_fp4(yq, p["w2"][e], p["s2"][e]).float()
    if shared:
        g = linear_fp8(xq, p["sh_w1"], p["sh_s1"]).float()
        u = linear_fp8(xq, p["sh_w3"], p["sh_s3"]).float()
        y = (F.silu(g.clamp(max=cfg.swiglu_limit)) * u.clamp(-cfg.swiglu_limit, cfg.swiglu_limit)).to(torch.bfloat16)
        yq = M.act_quant_dequant(y, 32)
        acc += shared_scale * F.linear(yq.float(), dequant_block_fp8(p["sh_w2"], p["sh_s2"], 32, torch.float32))
    return acc, top, w, torch.cat(wes, -1), logits


def compressor_ref(p, x_n, idx, ratio, pool_kv, pool_score, cfg):
    """Latents of this step's groups (bf16, pre-rope), the updated pool caches, the
    group positions; `p["cm_*"]` are the compressor weights."""
    T = x_n.shape[0]
    pool_kv, pool_score = pool_kv.clone(), pool_score.clone()
    G = idx.groups_per_span
    B = idx.grp_wslots.shape[0] // G
    if ratio == 1:
        lat = F.linear(x_n.float(), p["cm_wkv"].float()).to(torch.bfloat16)   # bf16 GEMM
        lat = rmsnorm(lat, p["cm_norm_w"], cfg.rms_norm_eps)
        return lat, pool_kv, pool_score
    xf = x_n.float()
    kv_proj = xf @ p["cm_wkv"].float().t()
    sc_proj = xf @ p["cm_wgate"].float().t()
    ws = idx.write_slots.reshape(-1).long()
    pool_kv[ws] = kv_proj
    pool_score[ws] = sc_proj
    lat = torch.zeros(B * G, 512)
    for g in range(B * G):
        ms = idx.grp_mslots.reshape(-1).long()[g * ratio : (g + 1) * ratio]
        kv = pool_kv[ms]
        sc = pool_score[ms]
        lat[g] = (kv * torch.softmax(sc, dim=0)).sum(0)
    lat = rmsnorm(lat.to(torch.bfloat16), p["cm_norm_w"], cfg.rms_norm_eps)
    return lat, pool_kv, pool_score


def source_caches_ref(p, lat, idx, ccache, kcache, cfg):
    """Rope + fp4 the latents into the compressed cache and the index keys into the
    key cache (clones)."""
    rd = 64
    pos = idx.grp_pos.reshape(-1).long()
    cos, sin = p["freqs"][pos, :32], p["freqs"][pos, 32:]
    rot = torch.cat([lat[:, :-rd], rope(lat[:, -rd:], cos, sin)], -1)
    rot = M.fp4_quant_dequant(rot, 16, scale_e4m3=True)
    ccache = ccache.clone()
    ccache[idx.grp_wslots.reshape(-1).long()] = rot
    k = F.linear(lat.float(), p["ix_wk"].float()).to(torch.bfloat16)
    k = rmsnorm(k, p["ix_knorm_w"], cfg.rms_norm_eps)
    k = torch.cat([k[:, :-rd], rope(k[:, -rd:], cos, sin)], -1)
    k = M.fp4_quant_dequant(k, 32)
    kcache = kcache.clone()
    kcache[idx.grp_wslots.reshape(-1).long()] = k
    return ccache, kcache, rot, k


def index_ref(p, x_n, qr, idx, kcache, ratio, read_len, cfg, world):
    """This rank's index-score partial [T, read_len] (fp32) and the per-head pieces."""
    T = x_n.shape[0]
    rd = 64
    nh = p["ix_wq_b"].shape[0] // 128
    cos = p["freqs"][idx.row_pos.reshape(-1).long(), :32]
    sin = p["freqs"][idx.row_pos.reshape(-1).long(), 32:]
    q = linear_fp8(M.act_quant_dequant(qr, 32), p["ix_wq_b"], p["ix_wq_b_s"]).reshape(T, nh, 128)
    q = torch.cat([q[..., :-rd], rope(q[..., -rd:], cos, sin)], -1)
    q = M.fp4_quant_dequant(q, 32)
    w = F.linear(x_n.float(), p["ix_wproj"].float()).to(torch.bfloat16).float()
    w = w * (128 ** -0.5 * cfg.index_n_heads ** -0.5)
    B = idx.cmp_slots.shape[0] // read_len
    L = T // B
    score = torch.zeros(T, read_len)
    for b in range(B):
        kt = kcache[idx.cmp_slots[b * read_len : (b + 1) * read_len].reshape(-1).long()].float()  # [read_len, 128]
        s = torch.einsum("thd,jd->thj", q[b * L : (b + 1) * L].float(), kt).relu()
        score[b * L : (b + 1) * L] = (s * w[b * L : (b + 1) * L].unsqueeze(-1)).sum(1)
    return score


def topk_ref(score_sum, idx, topk):
    """The shipped selection on the reduced scores: mask j >= clen, top-k, -1 beyond clen."""
    T, read_len = score_sum.shape
    clen = idx.clen.reshape(-1).long()
    masked = score_sum.clone()
    j = torch.arange(read_len)
    masked[j.unsqueeze(0) >= clen.unsqueeze(1)] = -1e30
    k = min(topk, read_len)
    picks = torch.topk(masked, k, dim=-1).indices
    picks = torch.where(picks < clen.unsqueeze(1), picks, torch.full_like(picks, -1))
    return picks
