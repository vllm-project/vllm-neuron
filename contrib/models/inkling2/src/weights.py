# SPDX-License-Identifier: Apache-2.0
"""This rank's share of each Inkling 2 Mini checkpoint tensor, in the layouts ``ops.py`` reads.

Head-major layouts split contiguously by head: query channels ``h * 128 + d`` (3 heads a rank at TP 8), key / value
channels ``kvh * 128 + d`` (one KV head a rank), the relative-bias projection rows ``h * 16 + r``, the dynamic
convolution's tap rows ``kvh * 4 + lag``. The experts keep all 128 experts on every rank and split the intermediate
dim (1152 / TP).

Experts are stored NVFP4 (layers 0 to 19) or MXFP8 (20 to 23) and dequantized to bf16 here, as Hugging Face does at
load (``Inkling2LoopDequantizeExperts``): the per-expert ``amax`` is rounded to bf16 first, as HF's load casts it.
``w13`` interleaves the rows (gate ``i`` at row ``2i``, up ``i`` at ``2i + 1``), so one rank's rows are one
contiguous block of ``2 * I_r``; the result is ``[gate (I_r) | up (I_r)]`` per expert.
"""

from __future__ import annotations

import torch

DTYPE = torch.bfloat16
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def rows(t, rank: int, n: int) -> torch.Tensor:
    return t[rank * n:(rank + 1) * n]


def cols(t, rank: int, n: int) -> torch.Tensor:
    return t[:, rank * n:(rank + 1) * n]


def nvfp4(codes: torch.Tensor, scale: torch.Tensor, amax: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    """``codes [E, R, C/2]`` uint8 (even element in the low nibble), ``scale [E, R, C/16]`` e4m3, per-expert
    ``amax [E]`` and scalar ``denom``: ``E2M1[code] * scale * amax / denom`` in fp32, returned bf16 ``[E, R, C]``."""
    table = torch.tensor(E2M1, dtype=torch.float32)
    c = codes.to(torch.uint8).long()
    vals = torch.stack((table[c & 0xF], table[c >> 4]), dim=-1).reshape(*c.shape[:-1], -1)   # [E, R, C]
    s = scale.to(torch.bfloat16).float().repeat_interleave(16, dim=-1)
    factor = amax.to(torch.bfloat16).float() / denom.to(torch.bfloat16).float().reshape(())
    return (vals * s * factor.view(-1, 1, 1)).to(DTYPE)


def mxfp8(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``values [E, R, C]`` e4m3, ``scale [E, R, C/32]`` E8M0: ``value * 2 ** (scale - 127)``, returned bf16."""
    s = torch.exp2(scale.to(torch.uint8).float() - 127).repeat_interleave(32, dim=-1)
    return (values.float() * s).to(DTYPE)


def gate_up(w13, scale, amax, denom, rank: int, inter_r: int) -> torch.Tensor:
    """This rank's ``[E, 2 * I_r, H]``: ``[gate | up]`` of intermediate channels ``rank * I_r ...``."""
    r0, r1 = 2 * inter_r * rank, 2 * inter_r * (rank + 1)
    if amax is not None:
        w = nvfp4(w13[:, r0:r1, :], scale[:, r0:r1, :], amax[:], denom[:])
    else:
        w = mxfp8(w13[:, r0:r1, :], scale[:, r0:r1, :])
    return torch.cat([w[:, 0::2], w[:, 1::2]], dim=1).contiguous()


def down(w2, scale, amax, denom, rank: int, inter_r: int) -> torch.Tensor:
    """This rank's ``[E, H, I_r]`` of the down projection (input channels ``rank * I_r ...``)."""
    if amax is not None:
        c0, c1 = inter_r * rank, inter_r * (rank + 1)
        return nvfp4(w2[:, :, c0 // 2:c1 // 2], scale[:, :, c0 // 16:c1 // 16], amax[:], denom[:]).contiguous()
    return mxfp8(w2[:], scale[:])[:, :, inter_r * rank:inter_r * (rank + 1)].contiguous()


def message_queries(q, kv) -> torch.Tensor:
    """``middle_cycle_cross_pass_queries`` and ``..._kv_queries`` (``[2 * loop_layers, H]`` each) as
    ``[loop_layers, 4, H]`` fp32 (query, key, value, mlp)."""
    q, kv = q[:].float(), kv[:].float()
    q, kv = q.view(-1, 2, q.shape[-1]), kv.view(-1, 2, kv.shape[-1])
    return torch.cat([q[:, 0:1], kv, q[:, 1:2]], dim=1).contiguous()


def lm_head(w, rank: int, rows_per_rank: int, unpadded: int) -> torch.Tensor:
    """Rows ``rank * R ...`` of the first ``unpadded`` rows of the unembedding, zero-padded to ``R``."""
    r0 = rank * rows_per_rank
    r1 = min(r0 + rows_per_rank, unpadded)
    hidden = (w.get_shape() if hasattr(w, "get_shape") else w.shape)[-1]   # a safetensors slice or a tensor
    out = torch.zeros(rows_per_rank, hidden, dtype=DTYPE)
    if r1 > r0:
        out[:r1 - r0] = w[r0:r1].to(DTYPE)
    return out
