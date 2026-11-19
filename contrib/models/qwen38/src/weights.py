# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8's checkpoint tensors in the layouts the Qwen3.5 kernels read.

The checkpoint prefixes the language model with ``model.language_model.`` and carries a ``model.visual.`` tower that
text-only serving does not load. Two packings are not a plain shard:

- **GDN.** ``gdn_decode_tkg`` and ``gdn_prefill_tkg`` take one projection operand,
  ``w_proj_p [128, H/128, 2Kd + 2Vd + 2vh]``, holding this rank's q, k, v, z, a and b columns with the rows permuted
  so that ``w_p[p, h1, c] = W[h1 * 128 + p, c]`` (``kernels/gdn_decode_torch.pack_inputs``). The checkpoint has them as four
  linears: ``in_proj_qkv`` (q | k | v), ``in_proj_z``, ``in_proj_a`` and ``in_proj_b``.
- **Attention.** ``q_proj`` holds the query and the output gate folded together: HF views its output as
  ``[..., heads, 2 * head_dim]`` and splits the last axis, so head ``h`` owns rows ``h * 2d .. h * 2d + d`` (query)
  then ``h * 2d + d .. (h + 1) * 2d`` (gate). ``attn_decode_tkg`` wants them apart: ``w_qkv [H, (nq + 2) * d]`` as
  query heads, key, value, and ``w_gate_p [128, H/128, nq * d]`` row-permuted like the GDN projection.

Head sharding: GDN value heads and key heads split over the attention TP (the GQA ratio holds inside a rank, so the
key heads a rank keeps are the ones its value heads group over), attention query heads split, and with more ranks
than KV heads each KV head is replicated over an even group of ranks.
"""

from __future__ import annotations

import torch

GDN_HEAD = 128


def row_permute(w: torch.Tensor) -> torch.Tensor:
    """``[rows, cols] -> [128, rows/128, cols]`` with ``out[p, h1, c] = w[h1 * 128 + p, c]`` (the kernels' projection
    layout). ``rows`` is the hidden size here, a multiple of 128."""
    rows, cols = w.shape
    return w.reshape(rows // GDN_HEAD, GDN_HEAD, cols).transpose(0, 1).contiguous()


def kv_rank(rank: int, tp_size: int, num_kv_heads: int) -> int:
    """The KV shard a rank holds: its own with at least one KV head per rank, else the one its query heads use."""
    return rank if num_kv_heads >= tp_size else rank // (tp_size // num_kv_heads)


def gdn_projection(qkv: torch.Tensor, z: torch.Tensor, a: torch.Tensor, b: torch.Tensor, *, rank: int, tp_size: int,
                   num_k_heads: int, num_v_heads: int, head_dim: int = GDN_HEAD) -> torch.Tensor:
    """``w_proj_p`` for one rank from the checkpoint's four GDN input projections (each ``[out, H]``)."""
    kd = num_k_heads * head_dim
    vh, kh = num_v_heads // tp_size, num_k_heads // tp_size
    v_lo, k_lo = rank * vh, rank * kh
    q_cols = qkv[:kd].t()[:, k_lo * head_dim:(k_lo + kh) * head_dim]
    k_cols = qkv[kd:2 * kd].t()[:, k_lo * head_dim:(k_lo + kh) * head_dim]
    v_cols = qkv[2 * kd:].t()[:, v_lo * head_dim:(v_lo + vh) * head_dim]
    z_cols = z.t()[:, v_lo * head_dim:(v_lo + vh) * head_dim]
    a_cols = a.t()[:, v_lo:v_lo + vh]
    b_cols = b.t()[:, v_lo:v_lo + vh]
    return row_permute(torch.cat([q_cols, k_cols, v_cols, z_cols, a_cols, b_cols], dim=1).contiguous())


def gdn_conv(conv_w: torch.Tensor, *, rank: int, tp_size: int, num_k_heads: int, num_v_heads: int,
             head_dim: int = GDN_HEAD) -> torch.Tensor:
    """``conv_w [Cc, K]`` for one rank: its q, k and v channel blocks, in that order. The checkpoint's depthwise
    ``conv1d.weight`` is ``[conv_dim, 1, K]`` over all heads."""
    w = conv_w.squeeze(1) if conv_w.dim() == 3 else conv_w
    kd = num_k_heads * head_dim
    vh, kh = num_v_heads // tp_size, num_k_heads // tp_size
    k_off, v_off = rank * kh * head_dim, rank * vh * head_dim
    return torch.cat([w[:kd][k_off:k_off + kh * head_dim],
                      w[kd:2 * kd][k_off:k_off + kh * head_dim],
                      w[2 * kd:][v_off:v_off + vh * head_dim]], dim=0).contiguous()


def gdn_out(out_proj: torch.Tensor, *, rank: int, tp_size: int, num_v_heads: int,
            head_dim: int = GDN_HEAD) -> torch.Tensor:
    """``w_out [Vd/tp, H]``: this rank's value-head rows of ``out_proj`` ([H, value_dim] in the checkpoint)."""
    vh = num_v_heads // tp_size
    lo = rank * vh * head_dim
    return out_proj.t()[lo:lo + vh * head_dim, :].contiguous()


def attention_qkv(q_proj: torch.Tensor, k_proj: torch.Tensor, v_proj: torch.Tensor, *, rank: int, tp_size: int,
                  num_q_heads: int, num_kv_heads: int, head_dim: int, padded_q_heads: int = 0):
    """``(w_qkv [H, (nq + 2) * d], w_gate [H, nq * d])`` for one rank. ``q_proj`` is ``[nq * 2d, H]`` with each head's
    query rows followed by its gate rows. ``padded_q_heads`` (0: none) appends all-zero query heads up to that count,
    as ``config.Qwen38Config.kernel_q_heads`` asks."""
    nq = num_q_heads // tp_size
    lo = rank * nq
    q_heads = q_proj.reshape(num_q_heads, 2 * head_dim, -1)[lo:lo + nq]
    q_cols = q_heads[:, :head_dim].reshape(nq * head_dim, -1).t()
    gate_cols = q_heads[:, head_dim:].reshape(nq * head_dim, -1).t()
    extra = max(0, padded_q_heads - nq) * head_dim
    if extra:
        zeros = torch.zeros(q_cols.shape[0], extra, dtype=q_cols.dtype)
        q_cols, gate_cols = torch.cat([q_cols, zeros], dim=1), torch.cat([gate_cols, zeros], dim=1)
    kv = kv_rank(rank, tp_size, num_kv_heads)
    nkv = max(1, num_kv_heads // tp_size)
    k_cols = k_proj[kv * nkv * head_dim:(kv + 1) * nkv * head_dim].t()
    v_cols = v_proj[kv * nkv * head_dim:(kv + 1) * nkv * head_dim].t()
    return (torch.cat([q_cols, k_cols, v_cols], dim=1).contiguous(), gate_cols.contiguous())


def attention_out(o_proj: torch.Tensor, *, rank: int, tp_size: int, num_q_heads: int, head_dim: int,
                  padded_q_heads: int = 0) -> torch.Tensor:
    """``w_o [nq * d, H]``: this rank's query-head rows of ``o_proj`` ([H, nq_total * d] in the checkpoint), plus
    zero rows for the padding heads (``attention_qkv``), so they add nothing to the output."""
    nq = num_q_heads // tp_size
    lo = rank * nq * head_dim
    w = o_proj.t()[lo:lo + nq * head_dim, :]
    extra = max(0, padded_q_heads - nq) * head_dim
    if extra:
        w = torch.cat([w, torch.zeros(extra, w.shape[1], dtype=w.dtype)], dim=0)
    return w.contiguous()


def column_shard(w: torch.Tensor, *, rank: int, tp_size: int) -> torch.Tensor:
    """A column-parallel projection ``[out, in]`` -> this rank's ``[in, out/tp]`` (gate and up of the MLP)."""
    rows = w.shape[0] // tp_size
    return w[rank * rows:(rank + 1) * rows].t().contiguous()


def row_shard(w: torch.Tensor, *, rank: int, tp_size: int) -> torch.Tensor:
    """A row-parallel projection ``[out, in]`` -> this rank's ``[in/tp, out]`` (down of the MLP)."""
    cols = w.shape[1] // tp_size
    return w[:, rank * cols:(rank + 1) * cols].t().contiguous()
