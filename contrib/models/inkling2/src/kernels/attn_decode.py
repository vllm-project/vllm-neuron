# SPDX-License-Identifier: Apache-2.0
"""The decode attention core as one NKI kernel: logits, relative-position bias, causal / window mask, sink softmax,
and the value sum, for one query token a row against its gathered context.

The torch core (``ops.attention_core``) is about a dozen small ops a visit (the relative shift alone is a flip, two
pads, reshapes and an index_select); each is a separate instruction stream in the decode graph. Here:

    logits[h, j] = q[h] . k[j] / sqrt(d) + rel[h, off - j]      (0 <= off - j < E, else + 0)
    keep j iff off - j >= 0 (and off - j < window)              (range_select against per-row bounds)
    p = softmax([logits, sink])[:-1];  o[h] = sum_j p[h, j] v[j]

with ``off = q_pos - kv_pos[0]`` (keys at consecutive positions from ``kv_pos[0]``). The bias row is read from
``relp = [0 x L, flip(rel), 0 x L]`` at a runtime start ``L + E - 1 - off`` (one DMA a row, start in a register).

Inputs (HBM): ``q [B, hq, d]`` bf16, ``k, v [B, L, d]`` bf16, ``relp [B, hq, E + 2 L]`` bf16, ``start [1, B]``
int32, ``bounds [B, 2]`` fp32 (keep ``lo <= j < hi``), ``sink [B, hq]`` fp32. ``L`` a multiple of 128, ``d = 128``.
Output ``[B, hq, d]`` bf16.
"""

from __future__ import annotations

import math

import torch

P = 128
N_TILE = 512
NEG = -3.4028234663852886e38

try:
    import os

    import nki
    import nki.isa as nisa
    import nki.language as nl

    # the bias row's runtime-offset DMA: hardware descriptors by default; INKLING2_ATTN_BIAS_DGE=sw|auto to compare
    # (K2 + K4 in one graph hung on the dynamic hardware DMA queues once, see the notes)
    _BIAS_DGE = {"sw": nisa.dge_mode.swdge, "auto": nisa.dge_mode.unknown}.get(
        os.environ.get("INKLING2_ATTN_BIAS_DGE", ""), nisa.dge_mode.hwdge)

    @nki.jit
    def inkling2_attn_decode(q, k, v, relp, start, bounds, sink):
        B, hq, d = q.shape
        L = k.shape[1]
        E2 = relp.shape[2]
        C = L // P
        scale = 1.0 / math.sqrt(d)
        out = nl.ndarray((B, hq, d), dtype=q.dtype, buffer=nl.shared_hbm)
        relp2 = relp.reshape((B * hq, E2))

        start_sb = nl.ndarray((1, B), dtype=start.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=start_sb, src=start)

        for b in range(B):
            # keys and values with positions on partitions: [128, C, d] (position c * 128 + p)
            k_sb = nl.ndarray((P, C, d), dtype=k.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=k_sb, src=k.ap(pattern=[[d, P], [P * d, C], [1, d]], offset=b * L * d))
            v_sb = nl.ndarray((P, C, d), dtype=v.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=v_sb, src=v.ap(pattern=[[d, P], [P * d, C], [1, d]], offset=b * L * d))
            # q^T [d, hq]
            qT = nl.ndarray((d, hq), dtype=q.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=qT, src=q.ap(pattern=[[1, d], [d, hq]], offset=b * hq * d))
            # K^T [d, L] by Tensor Engine transposes of [128, d] tiles
            kT = nl.ndarray((d, L), dtype=k.dtype, buffer=nl.sbuf)
            for c in range(C):
                tp = nl.ndarray((d, P), dtype=k.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=tp, data=k_sb[:, c, :])
                nisa.tensor_copy(dst=kT[:, c * P:(c + 1) * P], src=tp)

            # logits * scale + bias
            lg = nl.ndarray((hq, L), dtype=nl.float32, buffer=nl.sbuf)
            for n0 in range(0, L, N_TILE):
                nsz = min(N_TILE, L - n0)
                ps = nl.ndarray((hq, N_TILE), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=ps[:, 0:nsz], stationary=qT, moving=kT[:, n0:n0 + nsz])
                nisa.activation(dst=lg[:, n0:n0 + nsz], op=nl.copy, data=ps[:, 0:nsz], scale=scale)
            bias = nl.ndarray((hq, L), dtype=nl.float32, buffer=nl.sbuf)
            reg = nisa.register_alloc()
            nisa.register_load(reg, start_sb[0:1, b:b + 1])
            nisa.dma_copy(dst=bias, src=relp2[b * hq:(b + 1) * hq, nl.ds(reg, L)], dge_mode=_BIAS_DGE)
            nisa.tensor_tensor(dst=lg, data1=lg, data2=bias, op=nl.add)

            # mask (keep lo <= j < hi) and the row max
            bnd = nl.ndarray((hq, 2), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=bnd, src=bounds.ap(pattern=[[0, hq], [1, 2]], offset=b * 2))
            mx = nl.ndarray((hq, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.range_select(dst=lg, on_true_tile=lg, comp_op0=nl.greater_equal, comp_op1=nl.less,
                              bound0=bnd[:, 0:1], bound1=bnd[:, 1:2], reduce_res=mx, reduce_op=nl.maximum,
                              on_false_value=NEG)
            sk = nl.ndarray((hq, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=sk, src=sink.ap(pattern=[[1, hq], [1, 1]], offset=b * hq))
            nisa.tensor_tensor(dst=mx, data1=mx, data2=sk, op=nl.maximum)
            negm = nl.ndarray((hq, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=negm, data=mx, op0=nl.multiply, operand0=-1.0)
            # exp and the row sum; the sink's own term
            ex = nl.ndarray((hq, L), dtype=nl.float32, buffer=nl.sbuf)
            ssum = nl.ndarray((hq, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(dst=ex, op=nl.exp, data=lg, reduce_op=nl.add, reduce_res=ssum, bias=negm)
            es = nl.ndarray((hq, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=es, op=nl.exp, data=sk, bias=negm)
            nisa.tensor_tensor(dst=ssum, data1=ssum, data2=es, op=nl.add)
            rs = nl.ndarray((hq, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=rs, data=ssum)
            pb = nl.ndarray((hq, L), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=pb, data=ex, op0=nl.multiply, operand0=rs)

            # o = p @ v, contracting over positions (p^T tiles on partitions)
            o_ps = nl.ndarray((hq, d), dtype=nl.float32, buffer=nl.psum)
            for c in range(C):
                tp = nl.ndarray((P, hq), dtype=q.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=tp, data=pb[:, c * P:(c + 1) * P])
                pT = nl.ndarray((P, hq), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pT, src=tp)
                nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_sb[:, c, :], accumulate=(c > 0))
            o_sb = nl.ndarray((hq, d), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=o_sb, src=o_ps)
            nisa.dma_copy(dst=out[b], src=o_sb)
        return out
except ImportError:  # pragma: no cover
    inkling2_attn_decode = None


def prepare(rel: torch.Tensor, q_pos: torch.Tensor, kv0: torch.Tensor, L: int, window: int | None):
    """The kernel's bias table, bias start and mask bounds from ``rel [B, hq, E]`` and positions ``[B]``."""
    B, hq, E = rel.shape
    relp = torch.nn.functional.pad(torch.flip(rel, dims=[-1]), (L, L))                  # [B, hq, E + 2L]
    off = (q_pos - kv0).to(torch.int64)                                                  # [B]
    start = (L + E - 1 - off).to(torch.int32).view(1, B)
    hi = (off + 1).float()
    lo = (off - window + 1).float() if window is not None else torch.full_like(hi, -1.0)
    return relp, start, torch.stack([lo, hi], dim=1)


def attn_decode(wrap, q, k, v, rel, sink, q_pos, kv0, window):
    """``q [B, hq, d]``, ``k, v [B, L, d]``, ``rel [B, hq, E]``, ``sink [B, hq]``, ``q_pos, kv0 [B]``. Returns
    ``[B, hq, d]``."""
    relp, start, bounds = prepare(rel, q_pos, kv0, k.shape[1], window)
    return wrap(inkling2_attn_decode)(q.contiguous(), k.contiguous(), v.contiguous(), relp.contiguous(), start,
                                      bounds.contiguous(), sink.float().contiguous())
