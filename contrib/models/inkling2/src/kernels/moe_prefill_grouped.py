# SPDX-License-Identifier: Apache-2.0
"""The prefill MoE grouped by expert, as one NKI kernel: each expert runs only on the tokens routed to it.

The torch prefill MoE (``ops.moe_dense``) multiplies every token of a chunk by all 128 experts (16 times the routed
work at top 8). Here, per expert ``e`` (a static loop), the expert's weights are loaded once, and a dynamic loop
runs over its tiles of 32 routed tokens (``ceil(n_e / 32)``, from a table built in torch):

    x_t   = x[tok[e, tile]]                                  (gather of 32 rows, software DGE)
    a     = r * 4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25),  [g | u] = x_t @ gate_up[e]
    out[tok[e, tile]] += a @ down[e]                         (scatter-add of 32 rows, DMA read-modify-write)

Padding slots of a tile hold an out-of-range token id: the gather and the scatter skip them (``oob_mode=skip``).
A token's experts are distinct, so the token ids within a tile are unique (the scatter-add requires it).

Inputs (HBM): ``x [T, H]`` bf16, ``tok [E, T // 32, 32]`` int32, ``wt [E, T // 32, 32]`` fp32, ``ntile [1, E]``
int32, ``gate_up [E, H, 2 I_r]``, ``down [E, I_r, H]`` bf16. Output ``[T, H]`` fp32 (this rank's partial sum).
Experimental, opt-in (``INKLING2_NKI_MOE_PREFILL=1``).
"""

from __future__ import annotations

import torch

GATE_CAP, UP_CAP = 4.0, 25.0
P = 128
TT = 32               # tokens a tile
N_TILE = 512
OOB = 1 << 24         # token id of a padding slot

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    @nki.jit
    def inkling2_moe_prefill(x, tok, wt, ntile, gate_up, down):
        T, H = x.shape
        E, NT, _ = tok.shape
        I2 = gate_up.shape[2]
        I = I2 // 2
        H1 = H // P
        out = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.shared_hbm)
        zero = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=zero, value=0.0)
        for r0 in range(0, T, P):
            nisa.dma_copy(dst=out[r0:r0 + min(P, T - r0), :], src=zero[0:min(P, T - r0), :])

        nt_sb = nl.ndarray((1, E), dtype=ntile.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=nt_sb, src=ntile)
        gu_view = gate_up.reshape((E, P, H1 * I2))
        n_tiles = [(c * N_TILE, min(N_TILE, H - c * N_TILE)) for c in range((H + N_TILE - 1) // N_TILE)]
        i_parts = [(s, min(P, I - s)) for s in range(0, I, P)]

        for e in range(E):
            wgu = nl.ndarray((P, H1 * I2), dtype=gate_up.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=wgu, src=gu_view[e])
            wd = []
            for s, sz in i_parts:
                t = nl.ndarray((P, H), dtype=down.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=t[0:sz, :], src=down[e, s:s + sz, :])
                wd.append(t)
            reg = nisa.register_alloc()
            nisa.register_load(reg, nt_sb[0:1, e:e + 1])
            for ti in nl.dynamic_range(reg):
                ids = nl.ndarray((TT, 1), dtype=tok.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=ids, src=tok[e].select(0, ti).reshape((TT, 1)))
                rw = nl.ndarray((TT, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=rw, src=wt[e].select(0, ti).reshape((TT, 1)))
                xs = nl.ndarray((TT, H), dtype=x.dtype, buffer=nl.sbuf)
                nisa.memset(dst=xs, value=0.0)
                nisa.dma_copy(dst=xs, src=x.ap(pattern=[[H, TT], [1, H]], offset=0, vector_offset=ids, indirect_dim=0),
                              oob_mode=nisa.oob_mode.skip)
                # x^T blocks: hidden h = p * H1 + j on partition p (the weights' layout)
                gu_ps = nl.ndarray((TT, I2), dtype=nl.float32, buffer=nl.psum)
                for j in range(H1):
                    tp = nl.ndarray((P, TT), dtype=x.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=tp, data=xs[:, j:H:H1])
                    xT = nl.ndarray((P, TT), dtype=x.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=xT, src=tp)
                    nisa.nc_matmul(dst=gu_ps, stationary=xT, moving=wgu[:, j * I2:(j + 1) * I2], accumulate=(j > 0))
                sig = nl.ndarray((TT, I), dtype=nl.float32, buffer=nl.sbuf)
                g = nl.ndarray((TT, I), dtype=nl.float32, buffer=nl.sbuf)
                u = nl.ndarray((TT, I), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=sig, op=nl.sigmoid, data=gu_ps[:, 0:I])
                nisa.activation(dst=g, op=nl.tanh, data=gu_ps[:, 0:I], scale=1.0 / GATE_CAP)
                nisa.tensor_tensor(dst=g, data1=g, data2=sig, op=nl.multiply)
                nisa.activation(dst=u, op=nl.tanh, data=gu_ps[:, I:I2], scale=1.0 / UP_CAP)
                nisa.tensor_scalar(dst=u, data=u, op0=nl.multiply, operand0=GATE_CAP * UP_CAP,
                                   op1=nl.multiply, operand1=rw)
                a = nl.ndarray((TT, I), dtype=down.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=a, data1=g, data2=u, op=nl.multiply)
                aT = []
                for s, sz in i_parts:
                    tp = nl.ndarray((sz, TT), dtype=down.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=tp, data=a[:, s:s + sz])
                    sb = nl.ndarray((sz, TT), dtype=down.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=sb, src=tp)
                    aT.append(sb)
                y = nl.ndarray((TT, H), dtype=nl.float32, buffer=nl.sbuf)
                for n0, nsz in n_tiles:
                    ps = nl.ndarray((TT, N_TILE), dtype=nl.float32, buffer=nl.psum)
                    for pi, (s, sz) in enumerate(i_parts):
                        nisa.nc_matmul(dst=ps[:, 0:nsz], stationary=aT[pi][0:sz, :], moving=wd[pi][0:sz, n0:n0 + nsz],
                                       accumulate=(pi > 0))
                    nisa.tensor_copy(dst=y[:, n0:n0 + nsz], src=ps[:, 0:nsz])
                dst = out.ap(pattern=[[H, TT], [1, H]], offset=0, vector_offset=ids, indirect_dim=0)
                nisa.dma_compute(dst=dst, srcs=[dst, y], reduce_op=nl.add, oob_mode=nisa.oob_mode.skip)
        return out
except ImportError:  # pragma: no cover
    inkling2_moe_prefill = None


def tables(idx: torch.Tensor, w: torch.Tensor, E: int):
    """Per expert, its routed tokens and weights in tiles of ``TT`` (``tok``, ``wt [E, T // TT, TT]``; padding:
    token ``OOB``, weight 0) and its tile count (``ntile [1, E]``), from ``idx``, ``w [T, k]``."""
    T, k = idx.shape
    flat_e = idx.reshape(-1)
    flat_t = torch.arange(T, device=idx.device).repeat_interleave(k)
    onehot = torch.nn.functional.one_hot(flat_e, E).to(torch.int32)                     # [T k, E]
    rank = (torch.cumsum(onehot, 0) - 1).gather(1, flat_e.view(-1, 1)).view(-1)          # slot within its expert
    counts = onehot.sum(0)
    tok = torch.full((E, T), OOB, dtype=torch.int32, device=idx.device)
    tok[flat_e, rank] = flat_t.to(torch.int32)
    wt = torch.zeros(E, T, dtype=torch.float32, device=idx.device)
    wt[flat_e, rank] = w.reshape(-1).float()
    ntile = ((counts + TT - 1) // TT).to(torch.int32).view(1, E)
    return tok.view(E, T // TT, TT), wt.view(E, T // TT, TT), ntile


def moe_prefill(wrap, x, idx, w, gate_up, down):
    """``x [T, H]`` bf16 (``T`` a multiple of 32), ``idx``, ``w [T, k]``; returns this rank's partial ``[T, H]``."""
    tok, wt, ntile = tables(idx, w, gate_up.shape[0])
    return wrap(inkling2_moe_prefill)(x.contiguous(), tok, wt, ntile, gate_up, down)
