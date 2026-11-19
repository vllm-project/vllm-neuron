"""PyTorch reference for one rank's Qwen3.5 Gated DeltaNet decode step (TP-sharded heads).

Math follows HF ``Qwen3_5MoeGatedDeltaNet.forward`` for ``seq_len == 1`` with a cache,
restricted to this rank's heads, plus this rank's slice of ``out_proj`` (the result is a
partial sum that the caller reduces over TP). Rounding boundaries mirror the BF16 model:
projections, the convolution output and the core output are rounded to BF16; the
recurrence and both normalisations run in FP32.

The convolution history holds ``K - 1`` raw (pre-conv) values
per channel, oldest first.
"""

import os

import torch
import torch.nn.functional as F

HEAD_DIM = 128


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def gdn_decode_torch(
    x,  # [B, H] bf16, post input-RMSNorm hidden
    w_qkvz,  # [H, 2*Kd + 2*Vd] local q|k|v|z columns
    w_a,  # [H, vh]
    w_b,  # [H, vh]
    conv_w,  # [Cc, K] with Cc = 2*Kd + Vd
    conv_state,  # [B, Cc, K-1] raw history, oldest first
    a_log,  # [vh] fp32
    dt_bias,  # [vh] fp32
    norm_w,  # [128] RMSNormGated weight (plain, not zero-centred)
    rec_state,  # [B, vh, 128, 128] fp32
    w_out,  # [Vd, H]
    eps: float = 1e-6,
):
    """Returns (out [B, H] bf16 partial, new_conv_state, new_rec_state)."""
    B, H = x.shape
    vh = a_log.shape[0]
    Vd = vh * HEAD_DIM
    Kd = (w_qkvz.shape[1] - 2 * Vd) // 2
    kh = Kd // HEAD_DIM
    Cc = 2 * Kd + Vd
    bf = torch.bfloat16

    xb = x.to(bf).float()
    proj = (xb @ w_qkvz.to(bf).float()).to(bf)  # [B, 2Kd+2Vd]
    mixed, z = proj[:, :Cc], proj[:, Cc:]
    a = (xb @ w_a.to(bf).float()).to(bf).float()
    b = (xb @ w_b.to(bf).float()).to(bf).float()

    window = torch.cat([conv_state.to(bf).float(), mixed.float().unsqueeze(-1)], dim=-1)  # [B, Cc, K]
    conv = (window * conv_w.to(bf).float().unsqueeze(0)).sum(-1)
    conv = F.silu(conv).to(bf).float()
    new_conv_state = window[:, :, 1:].to(bf)

    q = conv[:, :Kd].view(B, kh, HEAD_DIM)
    k = conv[:, Kd : 2 * Kd].view(B, kh, HEAD_DIM)
    v = conv[:, 2 * Kd :].view(B, vh, HEAD_DIM)
    gqa = vh // kh
    q = _l2norm(q).repeat_interleave(gqa, dim=1) / (HEAD_DIM**0.5)
    k = _l2norm(k).repeat_interleave(gqa, dim=1)

    beta = torch.sigmoid(b)  # [B, vh]
    g = -a_log.float().exp() * F.softplus(a + dt_bias.float())
    S = rec_state.float() * g.exp()[..., None, None]
    kv = (S * k.unsqueeze(-1)).sum(-2)
    delta = (v - kv) * beta.unsqueeze(-1)
    S = S + k.unsqueeze(-1) * delta.unsqueeze(-2)
    o = (S * q.unsqueeze(-1)).sum(-2)  # [B, vh, 128]

    o = o.to(bf).float()  # core_attn_out leaves the recurrence in the model dtype
    o = (o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps)).to(bf).float()
    o = norm_w.to(bf).float() * o
    o = (o * F.silu(z.float().view(B, vh, HEAD_DIM))).to(bf)

    out = (o.reshape(B, Vd).float() @ w_out.to(bf).float()).to(bf)
    return out, new_conv_state, S


def gdn_kmajor():
    """QWEN35_GDN_KMAJOR=1: the decode kernel's recurrent state is k-major [128k, B, vh, 128v]."""
    return os.environ.get("QWEN35_GDN_KMAJOR", "0") == "1"


def _perm(a, axes):
    return a.permute(*axes) if isinstance(a, torch.Tensor) else a.transpose(axes)


def state_to_kernel(nat, kmajor=None):
    """Natural [B, vh, 128k, 128v] -> kernel layout: [B, 128k, vh, 128v] (token-major) or
    [128k, B, vh, 128v] (k-major)."""
    kmajor = gdn_kmajor() if kmajor is None else kmajor
    return _perm(nat, (2, 0, 1, 3) if kmajor else (0, 2, 1, 3))


def state_from_kernel(ker, kmajor=None):
    """Kernel layout -> natural [B, vh, 128k, 128v] (inverse of ``state_to_kernel``)."""
    kmajor = gdn_kmajor() if kmajor is None else kmajor
    return _perm(ker, (1, 2, 0, 3) if kmajor else (0, 2, 1, 3))


def kernel_state_token_major(ker, kmajor=None):
    """Kernel layout -> token-major kernel layout [B, 128k, vh, 128v] (row ops index dim 0)."""
    kmajor = gdn_kmajor() if kmajor is None else kmajor
    return _perm(ker, (1, 0, 2, 3)) if kmajor else ker


def unpack_inputs(w_proj_p, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out, kmajor=None):
    """Inverse of ``pack_inputs`` for the weight/state arguments (torch tensors)."""
    vh = a_log.shape[0]
    P_, H1, NT = w_proj_p.shape
    w = w_proj_p.permute(1, 0, 2).reshape(H1 * P_, NT)
    return {
        "w_qkvz": w[:, : NT - 2 * vh],
        "w_a": w[:, NT - 2 * vh : NT - vh],
        "w_b": w[:, NT - vh :],
        "conv_w": conv_w,
        "conv_state": conv_state,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "norm_w": norm_w,
        "rec_state": state_from_kernel(rec_state, kmajor),
        "w_out": w_out,
    }


def pack_inputs(x, w_qkvz, w_a, w_b, conv_w, conv_state, a_log, dt_bias, norm_w, rec_state, w_out):
    """Natural-layout numpy arrays -> ``gdn_decode_tkg`` argument layouts."""
    import numpy as np

    H = w_qkvz.shape[0]
    H1 = H // HEAD_DIM

    def perm_rows(w):  # [H, C] -> [128, H/128, C] with w_p[p, h1, c] = w[h1*128 + p, c]
        return np.ascontiguousarray(w.reshape(H1, HEAD_DIM, w.shape[1]).transpose(1, 0, 2))

    return {
        "x": x,
        "w_proj_p": perm_rows(np.concatenate([w_qkvz, w_a, w_b], axis=1)),
        "conv_w": conv_w,
        "conv_state": conv_state,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "norm_w": norm_w,
        "rec_state": np.ascontiguousarray(state_to_kernel(rec_state)),  # [B, 128k, vh, 128v] or k-major
        "w_out": w_out,
    }


def gdn_verify_torch(
    x,  # [B*S, H] bf16, sequence-major rows (b*S + s)
    w_qkvz, w_a, w_b, conv_w,
    conv_in,  # [B, NS, Cc, K-1]
    a_log, dt_bias, norm_w,
    rec_in,  # [B, NS, vh, 128, 128] fp32
    src_slot,  # [B] int: slot of conv_in / rec_in each sequence starts from
    w_out,
    eps: float = 1e-6,
):
    """Speculative-verify GDN: S sequential decode steps per sequence from a selected state slot.

    Returns (out [B*S, H] bf16, conv_out [B, S, Cc, K-1], rec_out [B, S, vh, 128, 128]) where
    ``*_out[:, s]`` is the state after the sequence's token s (rollback = pick a slot).
    """
    B = conv_in.shape[0]
    S = x.shape[0] // B
    idx = torch.as_tensor(src_slot).long().view(B)
    conv = conv_in[torch.arange(B), idx]
    rec = rec_in[torch.arange(B), idx]
    xs = x.view(B, S, -1)
    outs, convs, recs = [], [], []
    for s in range(S):
        o, conv, rec = gdn_decode_torch(
            xs[:, s], w_qkvz, w_a, w_b, conv_w, conv, a_log, dt_bias, norm_w, rec, w_out, eps
        )
        outs.append(o)
        convs.append(conv)
        recs.append(rec)
    out = torch.stack(outs, dim=1).reshape(B * S, -1)
    return out, torch.stack(convs, dim=1), torch.stack(recs, dim=1)
