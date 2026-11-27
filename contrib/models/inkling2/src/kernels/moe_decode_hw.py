# SPDX-License-Identifier: Apache-2.0
"""The decode MoE as one NKI kernel that loads each routed expert with hardware DMA descriptors.

Why: in the torch decode graph the experts' weights are gathered with ``index_select``, which the compiler lowers to
a software-generated DMA (descriptors written by the GpSimd engine during execution) with one descriptor per
576-byte row of an expert's ``gate_up`` slice (``[H, 2 I_r]``, ``I_r = 144``): 2304 descriptors an expert, 1.9 M a
decode step (device profile). Here an expert's ``gate_up`` is read as ``[128, 18 * 2 I_r]`` (hidden index
``h = p * 18 + j`` on partition ``p``), so each partition gets one contiguous 10 KB run, and the DMA uses
hardware descriptor generation (``dge_mode=hwdge``) with the expert id in a register.

Per token ``t`` and routed slot ``k`` (expert ``e``, routing weight ``r``)::

    gu   = x[t] @ gate_up[e]                       # [2 I_r], 18 matmuls of [128, 1] x [128, 2 I_r]
    a    = r * 4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25)
    y[t] += a @ down[e]                            # PSUM accumulation over the token's k experts

Inputs (HBM): ``x [T, H]`` bf16, ``idx [1, T * K]`` int32 (row-major ``[T, K]``), ``w [1, T * K]`` fp32,
``gate_up [E, H, 2 I_r]`` bf16 (``[in, out]``, gate then up), ``down [E, I_r, H]`` bf16. Output ``[T, H]`` fp32
(this rank's partial sum). ``H`` a multiple of 128, ``2 I_r <= 512``, ``T <= 4``. An expert id ``>= E`` (a padding
row) skips that expert's weight loads (``oob_mode=skip``); the row's output is then not meaningful.
"""

from __future__ import annotations

import torch

GATE_CAP, UP_CAP = 4.0, 25.0
P = 128
N_TILE = 512          # PSUM bank: 512 fp32 a partition
ROW = 32              # PSUM partition of token t: t * ROW (matmul column tiling of 32)
RING = 4              # expert weight buffers in flight

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    @nki.jit
    def inkling2_moe_decode_hw(x, idx, w, gate_up, down):
        T, H = x.shape
        TK = idx.shape[1]
        K = TK // T
        E, _, I2 = gate_up.shape
        I = I2 // 2
        H1 = H // P
        out = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.shared_hbm)

        # x^T: hidden h = p * H1 + j at (partition p, free j), one [P, H1] block a token
        xT = nl.ndarray((P, T, H1), dtype=x.dtype, buffer=nl.sbuf)
        for t in range(T):
            nisa.dma_copy(dst=xT[:, t, :], src=x[t:t + 1, :].reshape((P, H1)))
        idx_sb = nl.ndarray((1, TK), dtype=idx.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx_sb, src=idx)
        w_sb = nl.ndarray((1, TK), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_sb, src=w)

        gu_view = gate_up.reshape((E, P, H1 * I2))          # partition p: rows p * H1 .. p * H1 + H1 - 1
        n_tiles = [(c * N_TILE, min(N_TILE, H - c * N_TILE)) for c in range((H + N_TILE - 1) // N_TILE)]
        i_parts = [(s, min(P, I - s)) for s in range(0, I, P)]
        y_ps = [nl.ndarray((T * ROW, N_TILE), dtype=nl.float32, buffer=nl.psum) for _ in n_tiles]

        wgu = [nl.ndarray((P, H1 * I2), dtype=gate_up.dtype, buffer=nl.sbuf) for _ in range(RING)]
        wd = [[nl.ndarray((P, H), dtype=down.dtype, buffer=nl.sbuf) for _ in i_parts] for _ in range(RING)]
        # a skipped (padding) expert computes on whatever its buffer holds: start them at zero, so a padding row
        # never yields inf / NaN (it would reach the null block's KV and state, which every request reads past its
        # length)
        for r in range(RING):
            nisa.memset(dst=wgu[r], value=0.0)
            for t_ in wd[r]:
                nisa.memset(dst=t_, value=0.0)

        for t in range(T):
            for k in range(K):
                n = t * K + k
                slot = n % RING
                reg = nisa.register_alloc()
                nisa.register_load(reg, idx_sb[0:1, n:n + 1])
                nisa.dma_copy(dst=wgu[slot], src=gu_view.select(0, reg), dge_mode=nisa.dge_mode.hwdge,
                              oob_mode=nisa.oob_mode.skip)
                d_e = down.select(0, reg)                     # [I, H]
                for pi, (s, sz) in enumerate(i_parts):
                    nisa.dma_copy(dst=wd[slot][pi][0:sz, :], src=d_e[s:s + sz, :], dge_mode=nisa.dge_mode.hwdge,
                                  oob_mode=nisa.oob_mode.skip)

                # gate / up of this token: [1, 2I] in PSUM, accumulated over the H1 hidden blocks
                gu_ps = nl.ndarray((1, I2), dtype=nl.float32, buffer=nl.psum)
                for j in range(H1):
                    nisa.nc_matmul(dst=gu_ps, stationary=xT[:, t, j:j + 1],
                                   moving=wgu[slot][:, j * I2:(j + 1) * I2], accumulate=(j > 0))
                sig = nl.ndarray((1, I), dtype=nl.float32, buffer=nl.sbuf)
                g = nl.ndarray((1, I), dtype=nl.float32, buffer=nl.sbuf)
                u = nl.ndarray((1, I), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=sig, op=nl.sigmoid, data=gu_ps[0:1, 0:I])
                nisa.activation(dst=g, op=nl.tanh, data=gu_ps[0:1, 0:I], scale=1.0 / GATE_CAP)
                nisa.tensor_tensor(dst=g, data1=g, data2=sig, op=nl.multiply)
                nisa.activation(dst=u, op=nl.tanh, data=gu_ps[0:1, I:I2], scale=1.0 / UP_CAP)
                # u * (GATE_CAP * UP_CAP) * routing weight
                nisa.tensor_scalar(dst=u, data=u, op0=nl.multiply, operand0=GATE_CAP * UP_CAP,
                                   op1=nl.multiply, operand1=w_sb[0:1, n:n + 1])
                a = nl.ndarray((1, I), dtype=down.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=a, data1=g, data2=u, op=nl.multiply)

                # a^T on partitions (the down projection contracts over I)
                aT = []
                for s, sz in i_parts:
                    tp = nl.ndarray((sz, 1), dtype=down.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=tp, data=a[0:1, s:s + sz])
                    sb = nl.ndarray((sz, 1), dtype=down.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=sb, src=tp)
                    aT.append(sb)

                for c, (n0, nsz) in enumerate(n_tiles):
                    for pi, (s, sz) in enumerate(i_parts):
                        first = k == 0 and pi == 0
                        nisa.nc_matmul(dst=y_ps[c][t * ROW:t * ROW + 1, 0:nsz], stationary=aT[pi][0:sz, 0:1],
                                       moving=wd[slot][pi][0:sz, n0:n0 + nsz], accumulate=not first,
                                       tile_position=(0, t * ROW), tile_size=(P, ROW))

        for t in range(T):
            row = nl.ndarray((1, H), dtype=nl.float32, buffer=nl.sbuf)
            for c, (n0, nsz) in enumerate(n_tiles):
                nisa.tensor_copy(dst=row[0:1, n0:n0 + nsz], src=y_ps[c][t * ROW:t * ROW + 1, 0:nsz])
            nisa.dma_copy(dst=out[t:t + 1, :], src=row)
        return out
except ImportError:  # pragma: no cover
    inkling2_moe_decode_hw = None


def moe_decode_hw(wrap, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, gate_up: torch.Tensor,
                  down: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """``x [T, H]`` bf16, ``idx [T, k]``, ``w [T, k]`` fp32, ``gate_up [E, H, 2 I_r]``, ``down [E, I_r, H]``;
    ``valid [T]`` (bool, optional) marks the real rows; ``wrap`` turns the kernel into a callable (the caller's ``wrap_nki(...)[lnc]``). Returns this rank's partial
    ``[T, H]`` fp32."""
    T, k = idx.shape
    if valid is not None:   # padding rows: no weight traffic, zero routing weight
        idx = torch.where(valid.view(T, 1), idx, torch.full_like(idx, gate_up.shape[0]))
        w = torch.where(valid.view(T, 1), w, torch.zeros_like(w))
    return wrap(inkling2_moe_decode_hw)(x.contiguous(), idx.reshape(1, T * k).to(torch.int32),
                                        w.reshape(1, T * k).float(), gate_up, down)
