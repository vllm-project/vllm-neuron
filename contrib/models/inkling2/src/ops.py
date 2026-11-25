# SPDX-License-Identifier: Apache-2.0
"""The Inkling 2 Mini text decoder's arithmetic, in plain torch, for one tensor-parallel rank.

Nothing here knows about vLLM: ``model.py`` supplies the weights of this rank, the cached keys and values and the
short-convolution windows, and does the all-reduces. With ``tp_size = 1`` and contiguous caches these functions are
the whole model, which is how ``test/unit`` checks them against the Hugging Face implementation
(``transformers.models.inkling2_loop``) on a small random configuration.

The numerics follow the Hugging Face eager code (rounding points included where they are cheap to keep): fp32 for the
norms, clips, convolutions, gates, router, block residuals and softmaxes; bf16 for the projections and the attention
matmuls.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

CONV_TAPS = 4          # every short convolution has kernel 4
CONV_WINDOW = 3        # inputs a convolution needs from before the current token


def rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Unweighted RMS norm over the last dim, in fp32, returned in fp32."""
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float, dtype=torch.bfloat16) -> torch.Tensor:
    """RMSNorm with a plain weight (``w * x_hat``), output in ``dtype``."""
    return (w * rms(x, eps)).to(dtype)


def l2_clip(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Scale each last-dim vector down to ``threshold`` when its norm is larger. fp32 in, fp32 out."""
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * (threshold / (n + 1e-6)).clamp(max=1.0)


def log_scale(positions: torch.Tensor, alpha: float, n_floor: int) -> torch.Tensor:
    """``1 + alpha * ln(max((pos + 1) / n_floor, 1))`` per token, fp32 ``[T]``."""
    return 1.0 + alpha * torch.log(((positions + 1).float() / n_floor).clamp(min=1.0))


def head_clip(x: torch.Tensor, scale: torch.Tensor, threshold: float, tau: torch.Tensor | None,
              dtype=torch.bfloat16) -> torch.Tensor:
    """``ScaledL2Clip`` (``tau`` None) or ``LogScaledL2Clip`` over ``x [T, heads, d]``: learned scale, L2 clip, one
    rounding to ``dtype``, then the per-token log scale and a second rounding."""
    y = l2_clip(x.float() * scale.float(), threshold).to(dtype)
    if tau is not None:
        y = (y.float() * tau.view(-1, 1, 1)).to(dtype)
    return y


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    return torch.tanh(x / cap) * cap


# -- block and message residuals (per token, across depth) --

def depth_attention(states: list[torch.Tensor], partial: torch.Tensor, query: torch.Tensor,
                    eps: float) -> torch.Tensor:
    """Softmax over the normalized stacked states (``states`` then ``partial``) with learned queries ``[nq, H]``.
    Returns ``[nq, T, H]`` fp32."""
    s = torch.stack([*(t.float() for t in states), partial.float()], dim=1)    # [T, N+1, H]
    s = rms(s, eps)
    q = query.float().view(-1, s.shape[-1])                                     # [nq, H]
    w = torch.softmax(torch.einsum("qh,tnh->qtn", q, s) * s.shape[-1] ** -0.5, dim=-1)
    return torch.einsum("qtn,tnh->qth", w, s)


def message_gate(dest: torch.Tensor, message: torch.Tensor, down: torch.Tensor, forget: torch.Tensor,
                 gain: torch.Tensor, eps: float) -> torch.Tensor:
    """``(1 - tanh(f)) * dest + tanh(m) * message``; the low-rank code from ``[rms(dest), rms(message)]`` in bf16."""
    dest, message = dest.float(), message.float()
    g = torch.cat([rms(dest, eps), rms(message, eps)], dim=-1)
    g = (g.to(down.dtype) @ down.t()).float()
    f = torch.tanh(g @ forget.float().t())
    m = torch.tanh(g @ gain.float().t())
    return (1 - f) * dest + m * message


# -- short convolutions --

def static_conv(x: torch.Tensor, w: torch.Tensor, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal depthwise conv of kernel 4 over ``x [B, S, C]`` with the ``window [B, 3, C]`` of earlier inputs; tap 3
    weighs the current token. ``(y + x) / sqrt(5)`` in fp32, returned in x's dtype, and the inputs ``[B, 3 + S, C]``
    (fp32) whose last valid three are the next window."""
    xf = x.float()
    seq = torch.cat([window.float(), xf], dim=1)
    S = x.shape[1]
    wf = w.float()                                                              # [C, 4]
    y = sum(seq[:, k:k + S] * wf[:, k] for k in range(CONV_TAPS))
    return ((y + xf) / math.sqrt(CONV_TAPS + 1)).to(x.dtype), seq


def dynamic_conv(x: torch.Tensor, dyn_in: torch.Tensor, static_w: torch.Tensor, w_proj: torch.Tensor,
                 heads: int, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-layer K/V conv: taps by backward lag (lag 0 = current token) = ``static_w [4, C]`` plus per-token,
    per-head taps ``dyn_in @ w_proj.T`` (rows head * 4 + lag). ``(y + x) / 3``. Shapes as ``static_conv``."""
    B, S, C = x.shape
    xf = x.float()
    seq = torch.cat([window.float(), xf], dim=1)
    dw = (dyn_in @ w_proj.t()).float().view(B, S, heads, 1, CONV_TAPS)          # [B, S, h, 1, lag]
    taps = static_w.float().t().reshape(heads, C // heads, CONV_TAPS) + dw       # [B, S, h, d, lag]
    y = sum(seq[:, CONV_WINDOW - lag:CONV_WINDOW - lag + S].view(B, S, heads, C // heads) * taps[..., lag]
            for lag in range(CONV_TAPS))
    return ((y.reshape(B, S, C) + xf) / math.sqrt(2 * CONV_TAPS + 1)).to(x.dtype), seq


def next_window(seq: torch.Tensor, valid: torch.Tensor | int) -> torch.Tensor:
    """The three inputs before position ``valid`` of the new tokens: ``seq[:, valid : valid + 3]`` (``seq`` holds
    the old window then the new tokens)."""
    if isinstance(valid, int):
        return seq[:, valid:valid + CONV_WINDOW]
    idx = valid.view(-1, 1) + torch.arange(CONV_WINDOW, device=seq.device).view(1, -1)        # [B, 3]
    return torch.gather(seq, 1, idx.unsqueeze(-1).expand(-1, -1, seq.shape[-1]))


# -- attention --

def relative_bias(rel_logits: torch.Tensor, dist: torch.Tensor, extent: int) -> torch.Tensor:
    """``rel_logits [B, h, S, extent]`` read at ``dist [B, S, L]`` (query minus key position); 0 outside
    ``0 <= dist < extent``. Returns ``[B, h, S, L]``."""
    B, h, S, _ = rel_logits.shape
    idx = dist.clamp(0, extent - 1).unsqueeze(1).expand(B, h, S, dist.shape[-1])
    bias = torch.gather(rel_logits, -1, idx)
    keep = ((dist >= 0) & (dist < extent)).unsqueeze(1)
    return torch.where(keep, bias, torch.zeros((), dtype=bias.dtype, device=bias.device))


def relative_bias_shift(rel_logits: torch.Tensor, q_start: torch.Tensor, kv_len: int, extent: int,
                        kv_start: torch.Tensor | None = None) -> torch.Tensor:
    """``relative_bias`` for queries at consecutive positions ``q_start + s`` against keys at consecutive positions
    ``kv_start .. kv_start + kv_len - 1`` (``kv_start`` 0 when None),
    without an element gather (an indirect copy per element on Neuron): the Transformer-XL relative shift.

    With ``F[s, u] = rel_logits[s, extent - 1 - u]`` (distance ``extent - 1 - u``) and the band's first key
    ``w0 = q_start - extent + 1``, query ``s`` reads ``F[s, j - w0 - s]``. Padding ``F`` with ``S`` zero columns and
    reading the flat buffer with row stride ``extent + S - 1`` shifts row ``s`` right by ``s``, which gives the band
    ``[S, extent + S - 1]`` over keys ``w0 ...``; one ``index_select`` of ``kv_len`` columns places it.
    ``q_start [B]``; returns ``[B, h, S, kv_len]``."""
    B, h, S, E = rel_logits.shape
    f = torch.flip(rel_logits, dims=[-1])
    f = F.pad(f, (0, S)).reshape(B, h, S * (E + S))[..., :S * (E + S - 1)].view(B, h, S, E + S - 1)
    f = F.pad(f, (0, 1))                                                       # column E + S - 1: zero, for outside
    key_pos = torch.arange(kv_len, device=rel_logits.device).view(1, -1)
    if kv_start is not None:
        key_pos = key_pos + kv_start.view(-1, 1)
    col = key_pos - (q_start.view(-1, 1) - E + 1)                                                       # [B, L]
    col = torch.where((col >= 0) & (col < E + S - 1), col, torch.full_like(col, E + S - 1))
    out = torch.stack([f[b].index_select(-1, col[b]) for b in range(B)]) if B > 1 else f.index_select(-1, col[0])
    return out


def attention_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rel_logits: torch.Tensor,
                   sink: torch.Tensor, q_pos: torch.Tensor, kv_pos: torch.Tensor, window: int | None,
                   extent: int) -> torch.Tensor:
    """Attention of ``q [B, hq, S, d]`` over ``k, v [B, hkv, L, d]`` at positions ``q_pos [B, S]``, ``kv_pos [B, L]``:
    bf16 logits scaled by ``d**-0.5``, the relative bias, a causal (and ``window``) mask, the per-token sink logit
    ``sink [B, hq, S]`` in an fp32 softmax. Returns ``[B, hq, S, d]``."""
    B, hq, S, d = q.shape
    hkv = k.shape[1]
    k = k.repeat_interleave(hq // hkv, dim=1)
    v = v.repeat_interleave(hq // hkv, dim=1)
    dist = q_pos.unsqueeze(-1) - kv_pos.unsqueeze(1)                           # [B, S, L]
    logits = torch.matmul(q, k.transpose(-1, -2)) * d ** -0.5
    # the S queries of a row are consecutive positions (one request's chunk, or one decode token), and so are its
    # keys (the whole context, or a sliding layer's window from kv_pos[:, 0])
    logits = logits + relative_bias_shift(rel_logits, q_pos[:, 0], k.shape[2], extent, kv_pos[:, 0])
    keep = dist >= 0
    if window is not None:
        keep = keep & (dist < window)
    neg = torch.finfo(torch.float32).min
    logits = logits.float().masked_fill(~keep.unsqueeze(1), neg)
    p = torch.softmax(torch.cat([logits, sink.float().unsqueeze(-1)], dim=-1), dim=-1)[..., :-1]
    return torch.matmul(p.to(q.dtype), v)


def window_blocks(block_table: torch.Tensor, q_start: torch.Tensor, S: int, window: int, block: int):
    """The blocks a sliding-window layer reads for queries ``q_start .. q_start + S - 1``: from the block holding
    ``q_start - window + 1`` on, ``ceil((S + window - 1) / block) + 1`` blocks (a static count). Returns
    ``(blocks [B, n], first_pos [B])``: block ids (indices past the table repeat its last entry; their positions are
    past every query, so the causal mask drops them) and the position of the first key read."""
    n = -(-(S + window - 1) // block) + 1
    nb = block_table.shape[1]
    start = torch.div((q_start - window + 1).clamp(min=0), block, rounding_mode="floor")          # [B]
    idx = (start.view(-1, 1) + torch.arange(n, device=block_table.device).view(1, -1)).clamp(max=nb - 1)
    return torch.gather(block_table, 1, idx), start * block


# -- MoE (this rank's slice of every expert's intermediate dim) --

def route(x: torch.Tensor, gate_w: torch.Tensor, bias: torch.Tensor, top_k: int, scale: float):
    """Sigmoid scores; experts chosen by ``score + bias``, weighted by the unbiased scores renormalized and times
    ``scale``. Returns ``(index [T, k], weight [T, k] fp32)``."""
    s = torch.sigmoid(x.float() @ gate_w.float().t())
    idx = torch.topk(s + bias.float(), top_k, dim=-1).indices
    w = torch.gather(s, 1, idx)
    return idx, w / (w.sum(-1, keepdim=True) + 1e-20) * scale


def _act(gu: torch.Tensor, inter: int, gate_cap: float, up_cap: float) -> torch.Tensor:
    gate, up = gu[..., :inter].float(), gu[..., inter:].float()
    return (softcap(gate, gate_cap) * torch.sigmoid(gate) * softcap(up, up_cap)).to(gu.dtype)


def moe_gather(x, idx, w, gate_up, down, gate_cap, up_cap):
    """Each token's top-k experts only (decode): ``gate_up [E, 2I_r, H]``, ``down [E, H, I_r]``. fp32 partial sum."""
    inter = down.shape[-1]
    gu = torch.einsum("th,tkoh->tko", x, gate_up[idx])
    y = torch.einsum("tki,tkhi->tkh", _act(gu, inter, gate_cap, up_cap), down[idx])
    return (y.float() * w.unsqueeze(-1)).sum(1)


def moe_dense(x, idx, w, gate_up, down, gate_cap, up_cap, group: int = 32):
    """Every expert on every token, weighted by the router (prefill): static shapes, experts in groups so no
    ``[T, E, H]`` tensor is formed. fp32 partial sum."""
    E, inter = gate_up.shape[0], down.shape[-1]
    combine = torch.zeros(x.shape[0], E, dtype=torch.float32, device=x.device).scatter_add(1, idx, w)
    out = torch.zeros(x.shape[0], down.shape[1], dtype=torch.float32, device=x.device)
    for g0 in range(0, E, group):
        gu = torch.einsum("th,eoh->teo", x, gate_up[g0:g0 + group])
        a = (_act(gu, inter, gate_cap, up_cap).float() * combine[:, g0:g0 + group, None]).to(x.dtype)
        out = out + torch.einsum("tei,ehi->th", a, down[g0:g0 + group]).float()
    return out


# -- decoder layers and the loop schedule --

class LayerIO:
    """What a forward needs from its caller (contiguous caches in tests, the paged KV cache in vLLM).

    ``positions [B, S]`` are the new tokens' positions; ``kv(c, k, v, window)`` stores this step's keys and values of
    cache layer ``c`` (``[B, hkv, S, d]``) and returns the context ``(k, v, kv_pos)`` the queries may attend: the
    request's whole context, or (``window`` set) at least the last ``window`` positions before each query; ``window(c, slot)``
    and ``put_window(c, slot, seq)`` read and store a convolution's inputs (slots: 0 keys, 1 values, 2 attention
    output, 3 MoE output); ``all_reduce`` sums this rank's partial ``[B, S, H]`` over the TP group (fp32)."""

    positions: torch.Tensor

    def kv(self, c, k, v, window=None):
        raise NotImplementedError

    def window(self, c, slot):
        raise NotImplementedError

    def put_window(self, c, slot, seq):
        raise NotImplementedError

    def all_reduce(self, x):
        return x


def attention(p: dict, cfg, full: bool, xq, xk, xv, c: int, io: LayerIO):
    """One attention block (``xq, xk, xv [B, S, H]`` bf16): this rank's heads, the K/V convolutions, the cache, the
    sigmoid output gate and o_proj. Returns the all-reduced ``[B, S, H]`` bf16."""
    B, S, _ = xq.shape
    d, hq, hkv = cfg.head_dim, cfg.q_heads_per_rank, cfg.kv_heads_per_rank
    pos = io.positions
    tau = log_scale(pos.reshape(-1), cfg.log_alpha, cfg.log_floor) if full else None
    q = head_clip((xq @ p["wq"].t()).view(B * S, hq, d), p["q_scale"], cfg.qk_clip, tau, xq.dtype).view(B, S, hq, d)
    k_lin, v_lin = xk @ p["wk"].t(), xv @ p["wv"].t()                            # [B, S, hkv * d]
    if full:
        k_c, k_seq = dynamic_conv(k_lin, xk, p["k_static"], p["k_wproj"], hkv, io.window(c, 0))
        v_c, v_seq = dynamic_conv(v_lin, xv, p["v_static"], p["v_wproj"], hkv, io.window(c, 1))
    else:
        k_c, k_seq = static_conv(k_lin, p["k_conv"], io.window(c, 0))
        v_c, v_seq = static_conv(v_lin, p["v_conv"], io.window(c, 1))
    io.put_window(c, 0, k_seq)
    io.put_window(c, 1, v_seq)
    k = l2_clip(k_c.view(B, S, hkv, d).float(), cfg.qk_clip).to(xq.dtype).transpose(1, 2)
    v = rms(v_c.view(B, S, hkv, d), cfg.eps).to(xq.dtype).transpose(1, 2)
    k_ctx, v_ctx, kv_pos = io.kv(c, k, v, None if full else cfg.window)
    r = head_clip((xq @ p["wr"].t()).view(B * S, hq, cfg.d_rel), p["r_scale"], cfg.rel_clip, tau, xq.dtype)
    rel = (r @ p["r_proj"]).view(B, S, hq, -1).transpose(1, 2)                  # [B, hq, S, extent]
    sink = (xq @ p["wsink"].t()).transpose(1, 2)                                 # [B, hq, S]
    o = attention_core(q.transpose(1, 2), k_ctx, v_ctx, rel, sink, pos, kv_pos,
                       None if full else cfg.window, cfg.rel_extent)
    o = o.transpose(1, 2).reshape(B, S, hq * d)
    o = (o.float() * torch.sigmoid((xq @ p["wgate"].t()).float())).to(xq.dtype)
    return io.all_reduce((o @ p["wo"].t()).float()).to(xq.dtype)


def moe(p: dict, cfg, x: torch.Tensor, io: LayerIO, dense: bool) -> torch.Tensor:
    B, S, H = x.shape
    flat = x.reshape(B * S, H)
    idx, w = route(flat, p["router"], p["router_bias"], cfg.top_k, cfg.route_scale)
    fn = moe_dense if dense else moe_gather
    out = fn(flat, idx, w, p["gate_up"], p["down"], cfg.gate_cap, cfg.up_cap)
    return io.all_reduce(out.view(B, S, H)).to(x.dtype)


def decoder_layer(p: dict, cfg, L: int, c: int, h: torch.Tensor, blocks: list, msgs, io: LayerIO, dense: bool):
    """Model layer ``L`` on cache layer ``c``; ``h [B, S, H]`` bf16, ``blocks`` the stored block states,
    ``msgs`` None or ``[4, B, S, H]`` (query, key, value, mlp messages). Returns the new ``h`` (bf16)."""
    full = cfg.is_full(L)
    eps, dt = cfg.eps, h.dtype
    B, S, H = h.shape
    flat = lambda t: t.reshape(B * S, H)  # noqa: E731
    shape = lambda t: t.reshape(B, S, H)  # noqa: E731
    bs = [flat(b) for b in blocks]
    if full:
        q_in, k_in, v_in = depth_attention(bs, flat(h), p["attn_query"], eps)
        if msgs is not None:
            q_in = message_gate(q_in, flat(msgs[0]), *p["gate_q"], eps)
            k_in = message_gate(k_in, flat(msgs[1]), *p["gate_k"], eps)
            v_in = message_gate(v_in, flat(msgs[2]), *p["gate_v"], eps)
        xq, xk, xv = (shape(rms_norm(t, p[n], eps, dt)) for t, n in ((q_in, "q_norm"), (k_in, "k_norm"), (v_in, "v_norm")))
    else:
        a_in = flat(h).float() if L == 0 else depth_attention(bs, flat(h), p["attn_query"], eps)[0]
        if msgs is None:
            xq = xk = xv = shape(rms_norm(a_in, p["in_norm"], eps, dt))
        else:
            xq = shape(rms_norm(message_gate(a_in, flat(msgs[0]), *p["gate_q"], eps), p["in_norm"], eps, dt))
            xk = shape(rms_norm(message_gate(a_in, flat(msgs[1]), *p["gate_k"], eps), p["in_norm"], eps, dt))
            xv = shape(rms_norm(message_gate(a_in, flat(msgs[2]), *p["gate_v"], eps), p["in_norm"], eps, dt))
    a = attention(p, cfg, full, xq, xk, xv, c, io)
    a, seq = static_conv(a, p["post_attn_conv"], io.window(c, 2))
    io.put_window(c, 2, seq)
    if cfg.is_boundary(L):
        bs = bs + [flat(h)]
        r = flat(a).float()
    else:
        r = flat(h).float() + flat(a).float()
    m_in = depth_attention(bs, r, p["mlp_query"], eps)[0]
    if msgs is not None:
        m_in = message_gate(m_in, flat(msgs[3]), *p["gate_mlp"], eps)
    m = moe(p, cfg, shape(rms_norm(m_in, p["post_norm"], eps, dt)), io, dense)
    m, seq = static_conv(m, p["post_mlp_conv"], io.window(c, 3))
    io.put_window(c, 3, seq)
    return shape((r + flat(m).float()).to(h.dtype))


def text_model(params: dict, cfg, h: torch.Tensor, io: LayerIO, dense: bool) -> torch.Tensor:
    """The loop schedule over the 24 layers (48 visits): prelude, ``passes`` of the core with per-pass cache layers
    and the previous pass's messages, the coda, then the final depth attention and norm. ``h [B, S, H]`` is the
    normed embedding. Returns ``[B, S, H]`` bf16."""
    layers, eps = params["layers"], cfg.eps
    B, S, H = h.shape
    blocks: list = []

    def visit(L, c, h, blocks, msgs):
        new = decoder_layer(layers[L], cfg, L, c, h, blocks, msgs, io, dense)
        if cfg.is_boundary(L):
            blocks = blocks + [h]
        return new, blocks

    for L in range(cfg.loop_start):
        h, blocks = visit(L, L, h, blocks, None)
    pre_h, pre_blocks = h, blocks
    for L in range(cfg.loop_start, cfg.loop_end):
        h, blocks = visit(L, L, h, blocks, None)
    for p_idx in range(1, cfg.passes):
        prev_blocks = blocks[len(pre_blocks) + 1:]
        prev_h = h
        h, blocks = pre_h, pre_blocks
        offset = cfg.num_layers + (p_idx - 1) * cfg.loop_layers - cfg.loop_start
        for i, L in enumerate(range(cfg.loop_start, cfg.loop_end)):
            msgs = depth_attention([b.reshape(B * S, H) for b in prev_blocks], prev_h.reshape(B * S, H),
                                   params["message_query"][i], eps).view(4, B, S, H)
            h, blocks = visit(L, L + offset, h, blocks, msgs)
    for L in range(cfg.loop_end, cfg.num_layers):
        h, blocks = visit(L, L, h, blocks, None)
    out = depth_attention([b.reshape(B * S, H) for b in blocks], h.reshape(B * S, H), params["final_query"], eps)[0]
    return rms_norm(out, params["norm"], eps, h.dtype).view(B, S, H)
