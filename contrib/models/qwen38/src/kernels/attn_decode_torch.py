"""PyTorch reference for one rank's Qwen3.5 gated full-attention decode step.

HF ``Qwen3_5MoeAttention`` for a single new token per sequence, restricted to this rank's
``nq`` query heads and the single KV head they share, plus this rank's ``o_proj`` row slice
(the output is a partial sum reduced over TP). Paged KV: ``k_cache``/``v_cache``
``[num_blocks, block_len, d]`` addressed by ``block_table [B, max_blocks]``; sequence ``b``
attends to positions ``0..pos[b]`` where the new token is written at ``pos[b]``.
"""

import torch
import torch.nn.functional as F

HEAD = 256
ROT = 64  # partial_rotary_factor 0.25
THETA = 1e7


def _rmsnorm_zc(x, w, eps=1e-6):
    """Zero-centred RMSNorm (HF Qwen3_5MoeRMSNorm): x/rms * (1 + w), fp32, cast back."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())
    return out.to(x.dtype)


def rope_cos_sin(pos):
    """[B] positions -> cos/sin [B, ROT] (HF default rope, text-only mRoPE == plain RoPE)."""
    inv = 1.0 / THETA ** (torch.arange(0, ROT, 2, dtype=torch.float32) / ROT)
    f = pos.float()[:, None] * inv[None, :]
    emb = torch.cat([f, f], dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x):
    return torch.cat([-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]], dim=-1)


def attn_decode_torch(x, w_q, w_gate, w_k, w_v, q_norm_w, k_norm_w, w_o, k_cache, v_cache, block_table, pos):
    """Returns (out [B, H] bf16 partial, new k_cache, new v_cache)."""
    bf = torch.bfloat16
    B, H = x.shape
    nq = w_q.shape[1] // HEAD
    blk = k_cache.shape[1]
    xb = x.to(bf).float()
    q = (xb @ w_q.to(bf).float()).to(bf).view(B, nq, HEAD)
    gate = (xb @ w_gate.to(bf).float()).to(bf).view(B, nq * HEAD)
    k = (xb @ w_k.to(bf).float()).to(bf).view(B, 1, HEAD)
    v = (xb @ w_v.to(bf).float()).to(bf).view(B, 1, HEAD)
    q = _rmsnorm_zc(q, q_norm_w)
    k = _rmsnorm_zc(k, k_norm_w)
    cos, sin = rope_cos_sin(pos)
    cos, sin = cos[:, None, :], sin[:, None, :]

    def rope(t):
        tr, tp = t[..., :ROT].float(), t[..., ROT:]
        tr = tr * cos + _rotate_half(tr) * sin
        return torch.cat([tr.to(t.dtype), tp], dim=-1)

    q, k = rope(q), rope(k)

    k_cache, v_cache = k_cache.clone(), v_cache.clone()
    out = torch.zeros(B, nq, HEAD)
    for b in range(B):
        p = int(pos[b])
        blk_id, off = int(block_table[b, p // blk]), p % blk
        k_cache[blk_id, off] = k[b, 0].to(k_cache.dtype)
        v_cache[blk_id, off] = v[b, 0].to(v_cache.dtype)
        ks = torch.stack([k_cache[int(block_table[b, t // blk]), t % blk] for t in range(p + 1)]).float()
        vs = torch.stack([v_cache[int(block_table[b, t // blk]), t % blk] for t in range(p + 1)]).float()
        s = (q[b].float() @ ks.T) * HEAD**-0.5  # [nq, p+1]
        out[b] = F.softmax(s, dim=-1) @ vs
    o = out.to(bf).view(B, nq * HEAD)
    o = (o.float() * torch.sigmoid(gate.float())).to(bf)
    res = (o.float() @ w_o.to(bf).float()).to(bf)
    return res, k_cache, v_cache


def attn_verify_torch(x, w_q, w_gate, w_k, w_v, q_norm_w, k_norm_w, w_o, k_cache, v_cache, block_table, pos):
    """Speculative verify: ``x`` [B*S, H] sequence-major, token s of sequence b at position pos[b] + s.

    Token s attends to positions 0 .. pos[b] + s (history + new tokens 0..s); all S new K/V rows
    are written. Equivalent to S sequential single-token decode steps. Returns (out [B*S, H], k, v).
    """
    B = pos.shape[0]
    S = x.shape[0] // B
    xs = x.view(B, S, -1)
    outs = []
    for s in range(S):
        o, k_cache, v_cache = attn_decode_torch(
            xs[:, s], w_q, w_gate, w_k, w_v, q_norm_w, k_norm_w, w_o, k_cache, v_cache, block_table, pos + s
        )
        outs.append(o)
    return torch.stack(outs, dim=1).reshape(B * S, -1), k_cache, v_cache
