# SPDX-License-Identifier: Apache-2.0
#
# Vendored verbatim from the NxDI Qwen3.5 reference port, Apache-2.0:
#   https://github.com/qingzwang/neuronx-distributed-inference
#   branch qwen3.5-2b-hybrid-deltanet
#   contrib/models/Qwen3.5-2B/src/nki_kernels/nki_deltanet_fused.py
#
# The reference's *current* kernel, as opposed to the legacy one in
# ``nki_deltanet.py``. The two differ in the part that matters for speed: how they
# apply ``(I - A)^-1`` inside a chunk.
#
#   legacy  forward substitution, P_MAX sequential steps, each a full
#           128x128x128 matmul with all but one row of the result masked away
#   fused   ``_hierarchical_kkt_solve128``: Neumann by power-doubling on 32x32
#           leaves (``_leaf_inverse32_t``, 5 squaring rounds), then Schur
#           composition 32 -> 64 -> 128 (``_offdiag_combine_t`` computes
#           ``left @ cross @ right``, the off-diagonal block of the block inverse)
#
# The fused scheme is far more parallel, which is why it is worth measuring even
# though the legacy one lost badly to torch.
#
# Note the input contract differs from the legacy kernel: q and k arrive **raw**
# here and the l2-norm plus the 1/sqrt(dk) query scale happen in-kernel
# (``q_norm = q_c * q_inv_norm * QUERY_SCALE``).
#
# Caveat carried over from the reference: it documents the *multihead* variant in
# this file as numerically unstable on real vision embeddings. This port uses only
# the single-head ``deltanet_fused_chunked_fwd`` entry.

"""Fused single-kernel DeltaNet chunked forward for CTE (context encoding).

SSD-style architecture: processes ALL chunks for one (batch, head) pair in
a single NKI kernel call.  State (128x128) persists in SBUF across chunks —
no HBM round-trips for inter-chunk state propagation.

Key optimizations over nki_deltanet_chunked.py:
  1. Single kernel call per (B,H) instead of B*H*num_chunks calls
  2. State in SBUF across all chunks (no HBM state read/write per chunk)
  3. In-kernel cumsum via tensor_tensor_scan (no PyTorch cumsum)
  4. Masks and constants loaded once, reused across chunks
  5. Uses tensor_scalar for partition-broadcast (no explicit broadcast loops)
  6. nc_transpose (Vector Engine) for all 128x128 transposes instead of
     nc_matmul(moving=eye) (Tensor Engine) — frees TE for actual math

neuronx-cc 2.26.6360 / nki 0.5.0. k_dim = v_dim = 128 = P_MAX exactly.
Chunk size = 128 = P_MAX (one tile per chunk).

Mathematical framework:
  Per-chunk direct triangular solve for intra-chunk correction:
    QK_decay[i,j] = QK[i,j] * exp(gc[i] - gc[j]) for i > j
    A = -QK_decay * lower_mask
    v_new = solve((I - A), v_beta - (k_beta * exp(gc)) @ state)

  Inter-chunk state propagation:
    attn_inter = (q * exp(gc)) @ state
    attn_intra = (q @ k^T) * (strict_decay + I)
    output = attn_inter + attn_intra @ v_new
    state = exp(g_last) * (state + k_raw_decay^T @ v_new)
"""

import os

import numpy as np

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128  # Partition dim = k_dim = v_dim
CHUNK_SIZE = int(os.environ.get("QWEN36_DELTANET_CHUNK_SIZE", "128"))
L2_EPS_SQUARED = 1.0e-12
QUERY_SCALE = P_MAX ** -0.5
SOLVE_BLOCK_SIZE = int(os.environ.get("QWEN36_DELTANET_SOLVE_BLOCK_SIZE", "32"))
if (
    CHUNK_SIZE <= 0
    or P_MAX % CHUNK_SIZE != 0
    or CHUNK_SIZE % 32 != 0
    or SOLVE_BLOCK_SIZE <= 0
    or CHUNK_SIZE % SOLVE_BLOCK_SIZE != 0
    or SOLVE_BLOCK_SIZE % 32 != 0
    or SOLVE_BLOCK_SIZE & (SOLVE_BLOCK_SIZE - 1) != 0
):
    raise ValueError(
        "QWEN36_DELTANET_CHUNK_SIZE must be a positive divisor of P_MAX "
        "and a multiple of the 32-partition broadcast group, while "
        "QWEN36_DELTANET_SOLVE_BLOCK_SIZE must be positive, divide "
        "CHUNK_SIZE, be a power of two, and be a multiple of 32; "
        f"P_MAX={P_MAX}, CHUNK_SIZE={CHUNK_SIZE}, got {SOLVE_BLOCK_SIZE}"
    )
MAX_SOLVE_SCAN_STEPS = SOLVE_BLOCK_SIZE.bit_length() - 1
SOLVE_SCAN_STEPS = int(
    os.environ.get("QWEN36_DELTANET_SOLVE_SCAN_STEPS", str(MAX_SOLVE_SCAN_STEPS))
)
if SOLVE_SCAN_STEPS <= 0 or SOLVE_SCAN_STEPS > MAX_SOLVE_SCAN_STEPS:
    raise ValueError(
        "QWEN36_DELTANET_SOLVE_SCAN_STEPS must be in "
        f"[1, {MAX_SOLVE_SCAN_STEPS}] for SOLVE_BLOCK_SIZE={SOLVE_BLOCK_SIZE}; "
        f"got {SOLVE_SCAN_STEPS}"
    )
SOLVE_ACTIVE_PREFIX_K = os.environ.get(
    "QWEN36_DELTANET_SOLVE_ACTIVE_PREFIX_K",
    "0",
).lower() not in ("0", "false", "no", "off")
SOLVE_MODE = os.environ.get("QWEN36_DELTANET_SOLVE_MODE", "doubling").lower()
AUTOCP_CP_CHUNKS = int(os.environ.get("QWEN36_DELTANET_AUTOCP_CP_CHUNKS", "4"))
if SOLVE_MODE not in ("doubling", "kkt_hier"):
    raise ValueError(
        "QWEN36_DELTANET_SOLVE_MODE must be one of "
        "('doubling', 'kkt_hier'); "
        f"got {SOLVE_MODE!r}"
    )
SOLVE_KKT_HIER = SOLVE_MODE == "kkt_hier"
if SOLVE_KKT_HIER and (SOLVE_BLOCK_SIZE != P_MAX or CHUNK_SIZE != P_MAX):
    raise ValueError(
        "QWEN36_DELTANET_SOLVE_MODE=kkt_hier currently expects "
        f"QWEN36_DELTANET_CHUNK_SIZE={P_MAX} and "
        f"QWEN36_DELTANET_SOLVE_BLOCK_SIZE={P_MAX}; "
        f"got CHUNK_SIZE={CHUNK_SIZE}, SOLVE_BLOCK_SIZE={SOLVE_BLOCK_SIZE}"
    )
if AUTOCP_CP_CHUNKS <= 0:
    raise ValueError(
        "QWEN36_DELTANET_AUTOCP_CP_CHUNKS must be positive; "
        f"got {AUTOCP_CP_CHUNKS}"
    )

# Broadcast partition 0 to all partitions in a 32-wide group
_BROADCAST_MASK = [0] * 32


def _make_lower_mask():
    """Strict lower triangular active chunk block in a 128x128 constant."""
    mask = np.zeros((P_MAX, P_MAX), dtype=np.float32)
    mask[:CHUNK_SIZE, :CHUNK_SIZE] = np.tril(
        np.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32), k=-1
    )
    return mask


def _make_lower_mask_diag():
    """Lower triangular active chunk block with diagonal in a 128x128 constant."""
    mask = np.zeros((P_MAX, P_MAX), dtype=np.float32)
    mask[:CHUNK_SIZE, :CHUNK_SIZE] = np.tril(
        np.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32), k=0
    )
    return mask


def _make_identity():
    """Identity active chunk block in a 128x128 constant."""
    identity = np.zeros((P_MAX, P_MAX), dtype=np.float32)
    identity[:CHUNK_SIZE, :CHUNK_SIZE] = np.eye(CHUNK_SIZE, dtype=np.float32)
    return identity


def _matmul_square(dst, left, right, size):
    left_trans_psum = nl.ndarray((size, size), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=left_trans_psum, data=left)
    left_trans = nl.ndarray((size, size), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=left_trans, src=left_trans_psum)

    out_psum = nl.ndarray((size, size), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=out_psum, stationary=left_trans, moving=right)
    nisa.tensor_copy(dst=dst, src=out_psum)


def _offdiag_combine_t(dst, left_t, cross_t, right_t, size):
    tmp = nl.ndarray((size, size), dtype=nl.float32, buffer=nl.sbuf)
    _matmul_square(tmp, left_t, cross_t, size)
    _matmul_square(dst, tmp, right_t, size)


def _leaf_inverse32_t(dst, A_T, Imat, start):
    nisa.tensor_copy(dst=dst, src=Imat[0:32, 0:32])

    power_t = nl.ndarray((32, 32), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=power_t, src=A_T[start : start + 32, start : start + 32])

    power_psum = nl.ndarray((32, 32), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=power_psum, data=power_t)
    power = nl.ndarray((32, 32), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=power, src=power_psum)

    for _scan_i in nl.static_range(5):
        correction = nl.ndarray((32, 32), dtype=nl.float32, buffer=nl.sbuf)
        _matmul_square(correction, dst, power_t, 32)

        next_inv_t = nl.ndarray((32, 32), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=next_inv_t, data1=dst, data2=correction, op=nl.add)
        nisa.tensor_copy(dst=dst, src=next_inv_t)

        if _scan_i != 4:
            power_next = nl.ndarray((32, 32), dtype=nl.float32, buffer=nl.sbuf)
            _matmul_square(power_next, power, power, 32)

            power_next_t_psum = nl.ndarray(
                (32, 32), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_transpose(dst=power_next_t_psum, data=power_next)
            power_next_t = nl.ndarray((32, 32), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=power_next_t, src=power_next_t_psum)

            nisa.tensor_copy(dst=power, src=power_next)
            nisa.tensor_copy(dst=power_t, src=power_next_t)


def _inverse64_t(dst, A_T, Imat, start):
    nisa.memset(dst=dst, value=0.0)

    for leaf_idx in nl.static_range(2):
        leaf_offset = leaf_idx * 32
        leaf_start = start + leaf_offset
        leaf_t = nl.ndarray((32, 32), dtype=nl.float32, buffer=nl.sbuf)
        _leaf_inverse32_t(leaf_t, A_T, Imat, leaf_start)
        nisa.tensor_copy(
            dst=dst[leaf_offset : leaf_offset + 32, leaf_offset : leaf_offset + 32],
            src=leaf_t,
        )

    left32_t = nl.ndarray((32, 32), dtype=nl.float32, buffer=nl.sbuf)
    right32_t = nl.ndarray((32, 32), dtype=nl.float32, buffer=nl.sbuf)
    cross32_t = nl.ndarray((32, 32), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=left32_t, src=dst[0:32, 0:32])
    nisa.tensor_copy(dst=right32_t, src=dst[32:64, 32:64])
    nisa.tensor_copy(dst=cross32_t, src=A_T[start : start + 32, start + 32 : start + 64])

    off32_t = nl.ndarray((32, 32), dtype=nl.float32, buffer=nl.sbuf)
    _offdiag_combine_t(off32_t, left32_t, cross32_t, right32_t, 32)
    nisa.tensor_copy(dst=dst[0:32, 32:64], src=off32_t)


def _hierarchical_kkt_solve128(v_new, A_T, Imat, solve_rhs, dim):
    n_lo_t = nl.ndarray((64, 64), dtype=nl.float32, buffer=nl.sbuf)
    n_hi_t = nl.ndarray((64, 64), dtype=nl.float32, buffer=nl.sbuf)
    _inverse64_t(n_lo_t, A_T, Imat, 0)
    _inverse64_t(n_hi_t, A_T, Imat, 64)

    n128_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=n128_t, value=0.0)
    nisa.tensor_copy(dst=n128_t[0:64, 0:64], src=n_lo_t)
    nisa.tensor_copy(dst=n128_t[64:128, 64:128], src=n_hi_t)

    cross64_t = nl.ndarray((64, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cross64_t, src=A_T[0:64, 64:128])

    off64_t = nl.ndarray((64, 64), dtype=nl.float32, buffer=nl.sbuf)
    _offdiag_combine_t(off64_t, n_lo_t, cross64_t, n_hi_t, 64)
    nisa.tensor_copy(dst=n128_t[0:64, 64:128], src=off64_t)

    solved_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=solved_psum, stationary=n128_t, moving=solve_rhs)
    nisa.tensor_copy(dst=v_new, src=solved_psum)


def _blocked_doubling_solve(v_new, A_T, solve_rhs, dim):
    for solve_block in nl.static_range(CHUNK_SIZE // SOLVE_BLOCK_SIZE):
        block_start = solve_block * SOLVE_BLOCK_SIZE
        block_end = block_start + SOLVE_BLOCK_SIZE

        prev_contrib = nl.ndarray(
            (SOLVE_BLOCK_SIZE, dim), dtype=nl.float32, buffer=nl.sbuf
        )
        if solve_block == 0:
            nisa.memset(dst=prev_contrib, value=0.0)
        else:
            prev_psum = nl.ndarray(
                (SOLVE_BLOCK_SIZE, dim), dtype=nl.float32, buffer=nl.psum
            )
            if SOLVE_ACTIVE_PREFIX_K:
                nisa.nc_matmul(
                    dst=prev_psum,
                    stationary=A_T[0:block_start, block_start:block_end],
                    moving=v_new[0:block_start, 0:dim],
                )
            else:
                nisa.nc_matmul(
                    dst=prev_psum,
                    stationary=A_T[0:CHUNK_SIZE, block_start:block_end],
                    moving=v_new[0:CHUNK_SIZE, 0:dim],
                )
            nisa.tensor_copy(dst=prev_contrib, src=prev_psum)

        solve_rhs_block = nl.ndarray(
            (SOLVE_BLOCK_SIZE, dim), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_copy(
            dst=solve_rhs_block,
            src=solve_rhs[block_start:block_end, 0:dim],
        )

        residual_block = nl.ndarray(
            (SOLVE_BLOCK_SIZE, dim), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=residual_block,
            data1=solve_rhs_block,
            data2=prev_contrib,
            op=nl.add,
        )

        A_diag_T = nl.ndarray(
            (SOLVE_BLOCK_SIZE, SOLVE_BLOCK_SIZE),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        nisa.tensor_copy(
            dst=A_diag_T,
            src=A_T[block_start:block_end, block_start:block_end],
        )

        A_power_T = nl.ndarray(
            (SOLVE_BLOCK_SIZE, SOLVE_BLOCK_SIZE),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        nisa.tensor_copy(dst=A_power_T, src=A_diag_T)

        A_power_psum = nl.ndarray(
            (SOLVE_BLOCK_SIZE, SOLVE_BLOCK_SIZE),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        nisa.nc_transpose(dst=A_power_psum, data=A_power_T)
        A_power = nl.ndarray(
            (SOLVE_BLOCK_SIZE, SOLVE_BLOCK_SIZE),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        nisa.tensor_copy(dst=A_power, src=A_power_psum)

        local_v = nl.ndarray(
            (SOLVE_BLOCK_SIZE, dim), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_copy(dst=local_v, src=residual_block)

        for _scan_i in nl.static_range(SOLVE_SCAN_STEPS):
            correction_psum = nl.ndarray(
                (SOLVE_BLOCK_SIZE, dim), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(
                dst=correction_psum,
                stationary=A_power_T,
                moving=local_v,
            )
            correction = nl.ndarray(
                (SOLVE_BLOCK_SIZE, dim), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=correction, src=correction_psum)

            local_next = nl.ndarray(
                (SOLVE_BLOCK_SIZE, dim), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_tensor(
                dst=local_next, data1=local_v, data2=correction, op=nl.add
            )

            nisa.tensor_copy(dst=local_v, src=local_next)

            if _scan_i == SOLVE_SCAN_STEPS - 2:
                A_power_next_T_psum = nl.ndarray(
                    (SOLVE_BLOCK_SIZE, SOLVE_BLOCK_SIZE),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                nisa.nc_matmul(
                    dst=A_power_next_T_psum,
                    stationary=A_power,
                    moving=A_power_T,
                )
                nisa.tensor_copy(dst=A_power_T, src=A_power_next_T_psum)
            elif _scan_i != SOLVE_SCAN_STEPS - 1:
                A_power_next_psum = nl.ndarray(
                    (SOLVE_BLOCK_SIZE, SOLVE_BLOCK_SIZE),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                nisa.nc_matmul(
                    dst=A_power_next_psum,
                    stationary=A_power_T,
                    moving=A_power,
                )
                A_power_next = nl.ndarray(
                    (SOLVE_BLOCK_SIZE, SOLVE_BLOCK_SIZE),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                nisa.tensor_copy(dst=A_power_next, src=A_power_next_psum)

                A_power_next_T_psum = nl.ndarray(
                    (SOLVE_BLOCK_SIZE, SOLVE_BLOCK_SIZE),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                nisa.nc_transpose(dst=A_power_next_T_psum, data=A_power_next)
                A_power_next_T = nl.ndarray(
                    (SOLVE_BLOCK_SIZE, SOLVE_BLOCK_SIZE),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                nisa.tensor_copy(dst=A_power_next_T, src=A_power_next_T_psum)

                nisa.tensor_copy(dst=A_power, src=A_power_next)
                nisa.tensor_copy(dst=A_power_T, src=A_power_next_T)

        nisa.tensor_copy(
            dst=v_new[block_start:block_end, 0:dim],
            src=local_v[0:SOLVE_BLOCK_SIZE, 0:dim],
        )


@nki.jit
def deltanet_fused_chunked_fwd(
    query: nl.ndarray,  # (S, 128) float32 — raw Q; normalized in-kernel
    key: nl.ndarray,  # (S, 128) float32 — raw K; normalized in-kernel
    value: nl.ndarray,  # (S, 128) float32
    g_in: nl.ndarray,  # (S, 1)   float32 — per-token log-decay (NOT cumsum)
    beta_in: nl.ndarray,  # (S, 1)   float32 — per-token write gate
    initial_state: nl.ndarray,  # (128, 128) float32 — recurrent checkpoint or zeros
    lower_mask: nl.ndarray,  # (128, 128) float32 — strict lower tri
    identity: nl.ndarray,  # (128, 128) float32 — identity
    lower_mask_diag: nl.ndarray,  # (128, 128) float32 — lower tri with diag
):
    """Fused chunked DeltaNet forward — single kernel call per (batch, head).

    Processes all chunks sequentially within the kernel, keeping the recurrent
    state (128x128) in SBUF across chunks.  Returns per-token output and
    final state.

    Input requirements:
      - S must be divisible by 128 (pad before calling)
      - query/key are raw projected chunks; l2-norm and Q scale are in-kernel
      - g_in is RAW log-decay (cumsum computed in-kernel via tensor_tensor_scan)
      - beta_in is sigmoid(b) (write gate)
      - initial_state is zero for cold prefill, or the restored GDN checkpoint

    Returns:
        output:      (S, 128) float32
        final_state: (128, 128) float32
    """
    seq_len = query.shape[0]
    dim = query.shape[1]  # 128
    num_chunks = seq_len // CHUNK_SIZE

    # Output tensors in HBM
    output = nl.ndarray((seq_len, dim), dtype=query.dtype, buffer=nl.shared_hbm)
    final_state_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm)

    # ================================================================
    # Load constant masks into SBUF once (reused across all chunks)
    # ================================================================
    Lmask = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask, src=lower_mask)

    UMask_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=UMask_psum, data=Lmask)
    UMask = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=UMask, src=UMask_psum)

    Imat = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Imat, src=identity)

    # Ones vector for cumsum scan: (1, CHUNK_SIZE)
    ones_1xC = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_1xC, value=1.0)

    # Zero initial for cumsum scan
    zero_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero_11, value=0.0)

    # ================================================================
    # Initialize recurrent state in SBUF — persists across ALL chunks
    # ================================================================
    state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=state, src=initial_state)

    # ================================================================
    # Sequential chunk processing
    # ================================================================
    for i_chunk in nl.sequential_range(num_chunks):
        chunk_start = i_chunk * CHUNK_SIZE

        # ---- Load chunk data from HBM ----
        q_c = nl.ndarray((P_MAX, dim), dtype=query.dtype, buffer=nl.sbuf)
        if CHUNK_SIZE == P_MAX:
            nisa.dma_copy(
                dst=q_c,
                src=query[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
            )
        else:
            nisa.memset(dst=q_c, value=0.0)
            nisa.dma_copy(
                dst=q_c[0:CHUNK_SIZE, 0:dim],
                src=query[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
            )

        k_c = nl.ndarray((P_MAX, dim), dtype=key.dtype, buffer=nl.sbuf)
        if CHUNK_SIZE == P_MAX:
            nisa.dma_copy(
                dst=k_c,
                src=key[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
            )
        else:
            nisa.memset(dst=k_c, value=0.0)
            nisa.dma_copy(
                dst=k_c[0:CHUNK_SIZE, 0:dim],
                src=key[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
            )

        q_square = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=q_square, data1=q_c, data2=q_c, op=nl.multiply)
        q_norm_sq = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=q_norm_sq, data=q_square, op=nl.add, axis=1)
        q_norm_sq_clamped = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=q_norm_sq_clamped,
            data=q_norm_sq,
            op0=nl.maximum,
            operand0=L2_EPS_SQUARED,
            engine=nisa.vector_engine,
        )
        q_inv_norm = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=q_inv_norm,
            data=q_norm_sq_clamped,
            op0=nl.rsqrt,
            operand0=0.0,
            engine=nisa.gpsimd_engine,
        )
        q_norm = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=q_norm,
            data=q_c,
            op0=nl.multiply,
            operand0=q_inv_norm,
            op1=nl.multiply,
            operand1=QUERY_SCALE,
            engine=nisa.vector_engine,
        )

        k_square = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=k_square, data1=k_c, data2=k_c, op=nl.multiply)
        k_norm_sq = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=k_norm_sq, data=k_square, op=nl.add, axis=1)
        k_norm_sq_clamped = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_norm_sq_clamped,
            data=k_norm_sq,
            op0=nl.maximum,
            operand0=L2_EPS_SQUARED,
            engine=nisa.vector_engine,
        )
        k_inv_norm = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_inv_norm,
            data=k_norm_sq_clamped,
            op0=nl.rsqrt,
            operand0=0.0,
            engine=nisa.gpsimd_engine,
        )
        k_norm = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_norm,
            data=k_c,
            op0=nl.multiply,
            operand0=k_inv_norm,
            engine=nisa.vector_engine,
        )

        v_c = nl.ndarray((P_MAX, dim), dtype=value.dtype, buffer=nl.sbuf)
        if CHUNK_SIZE == P_MAX:
            nisa.dma_copy(
                dst=v_c,
                src=value[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
            )
        else:
            nisa.memset(dst=v_c, value=0.0)
            nisa.dma_copy(
                dst=v_c[0:CHUNK_SIZE, 0:dim],
                src=value[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
            )

        # g: (CHUNK_SIZE, 1) — raw log-decay per token
        g_chunk_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        if CHUNK_SIZE != P_MAX:
            nisa.memset(dst=g_chunk_p, value=0.0)
        nisa.dma_copy(
            dst=g_chunk_p[0:CHUNK_SIZE, 0:1],
            src=g_in[chunk_start : chunk_start + CHUNK_SIZE, 0:1],
        )

        # beta: (CHUNK_SIZE, 1) — write gate scalar per token
        beta_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        if CHUNK_SIZE != P_MAX:
            nisa.memset(dst=beta_p, value=0.0)
        nisa.dma_copy(
            dst=beta_p[0:CHUNK_SIZE, 0:1],
            src=beta_in[chunk_start : chunk_start + CHUNK_SIZE, 0:1],
        )

        # ---- In-kernel cumsum of g via tensor_tensor_scan ----
        # Need g as (1, CHUNK_SIZE) for scan along free dim. Use direct
        # vector transpose instead of padding through a full 128x128 tile.
        g_tp_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=g_tp_psum, data=g_chunk_p)

        g_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=g_row[0:1, 0:CHUNK_SIZE],
            src=g_tp_psum[0:1, 0:CHUNK_SIZE],
        )

        # cumsum: gc_row[t] = 1.0 * gc_row[t-1] + g_row[t]
        gc_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor_scan(
            dst=gc_row[0:1, 0:CHUNK_SIZE],
            data0=ones_1xC[0:1, 0:CHUNK_SIZE],
            data1=g_row[0:1, 0:CHUNK_SIZE],
            initial=zero_11[0:1, 0:1],
            op0=nl.multiply,
            op1=nl.add,
        )

        # Transpose gc back to (CHUNK_SIZE, 1) partition layout.
        gc_tp_psum = nl.ndarray((CHUNK_SIZE, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=gc_tp_psum, data=gc_row)

        # gc_p: (P_MAX, 1) — cumulative sum of g per token in this chunk
        gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        if CHUNK_SIZE != P_MAX:
            nisa.memset(dst=gc_p, value=0.0)
        nisa.tensor_copy(
            dst=gc_p[0:CHUNK_SIZE, 0:1],
            src=gc_tp_psum[0:CHUNK_SIZE, 0:1],
        )

        # g_last = gc[-1] (scalar) — needed for state decay
        gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=gl_11[0:1, 0:1],
            src=gc_row[0:1, CHUNK_SIZE - 1 : CHUNK_SIZE],
        )

        # ---- Compute exp(gc) and exp(g_last) as (P_MAX, 1) scalars ----
        # These (P_MAX, 1) tensors are used with tensor_scalar to broadcast
        # across the free dimension without explicit (P_MAX, dim) copies.

        exp_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=exp_gc_p[0:P_MAX, 0:1],
            op=nl.exp,
            data=gc_p[0:P_MAX, 0:1],
            bias=None,
            scale=1.0,
        )

        exp_gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=exp_gl_11,
            op=nl.exp,
            data=gl_11,
            bias=None,
            scale=1.0,
        )

        gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=gl_11[0:1, 0:1],
                dst=gl_p[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                shuffle_mask=_BROADCAST_MASK,
            )

        exp_gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=exp_gl_11[0:1, 0:1],
                dst=exp_gl_p[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                shuffle_mask=_BROADCAST_MASK,
            )

        # ============================================================
        # Stable pairwise decay factors from cumulative log-decay.
        #
        # The original fused path used split scaling:
        #   exp(gc[i]) * exp(-gc[j])
        # That can materialize huge unused intermediates.  Build the same
        # causal decay matrices as the per-chunk kernel using exp(gc[i]-gc[j])
        # and mask after the exp so upper-triangular values cannot leak into
        # later matmuls.
        # ============================================================
        gc_row_broadcast = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        if CHUNK_SIZE != P_MAX:
            nisa.memset(dst=gc_row_broadcast, value=0.0)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=gc_row[0:1, 0:CHUNK_SIZE],
                dst=gc_row_broadcast[
                    i_shuf * 32 : i_shuf * 32 + 32, 0:CHUNK_SIZE
                ],
                shuffle_mask=_BROADCAST_MASK,
            )

        gc_col_strict_t = nl.ndarray(
            (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=gc_col_strict_t,
            data1=gc_row_broadcast,
            data2=UMask,
            op=nl.multiply,
        )
        gc_row_strict_t = nl.ndarray(
            (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=gc_row_strict_t,
            data=UMask,
            op0=nl.multiply,
            operand0=gc_p,
            engine=nisa.vector_engine,
        )
        g_diff_strict_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=g_diff_strict_t,
            data1=gc_col_strict_t,
            data2=gc_row_strict_t,
            op=nl.subtract,
        )
        decay_strict_t_raw = nl.ndarray(
            (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.activation(
            dst=decay_strict_t_raw,
            op=nl.exp,
            data=g_diff_strict_t,
            bias=None,
            scale=1.0,
        )
        decay_strict_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=decay_strict_t,
            data1=decay_strict_t_raw,
            data2=UMask,
            op=nl.multiply,
        )

        decay_diag_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=decay_diag_t, data1=decay_strict_t, data2=Imat, op=nl.add
        )

        # ============================================================
        # k_beta = K * beta, v_beta = V * beta
        # tensor_scalar broadcasts beta_p (P_MAX, 1) across free dim
        # ============================================================
        k_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_beta,
            data=k_norm,
            op0=nl.multiply,
            operand0=beta_p,
            engine=nisa.vector_engine,
        )

        v_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=v_beta,
            data=v_c,
            op0=nl.multiply,
            operand0=beta_p,
            engine=nisa.vector_engine,
        )

        # ============================================================
        # Phase 1: Build A matrix (intra-chunk correction)
        # Transpose K and K_beta for matmul
        # ============================================================
        kb_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=kb_T_psum, data=k_beta)
        kb_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kb_T, src=kb_T_psum)

        k_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=k_T_psum, data=k_norm)
        k_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=k_T, src=k_T_psum)

        # QK_T[j, i] = k_norm[j] @ k_beta[i]. Build the transposed solve
        # matrix directly, avoiding a full A -> A_T transpose per chunk.
        QK_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=QK_T_psum, stationary=k_T, moving=kb_T)
        QK_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=QK_T, src=QK_T_psum)

        # A_T[j, i] = -QK[i, j] * exp(gc[i] - gc[j]) for i > j.
        QK_decay_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=QK_decay_t, data1=QK_T, data2=decay_strict_t, op=nl.multiply
        )

        A_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=A_T,
            data=QK_decay_t,
            op0=nl.multiply,
            operand0=-1.0,
            engine=nisa.vector_engine,
        )
        # ============================================================
        # Build the single RHS needed for v_new.
        #
        # Materializing N = inv(I - A) would compute:
        #   value_corr = N @ v_beta
        #   k_cumdecay = N @ (k_beta * exp(gc))
        #   v_new = value_corr - k_cumdecay @ state
        #
        # By associativity:
        #   v_new = N @ (v_beta - (k_beta * exp(gc)) @ state)
        #
        # Solve this RHS directly. This is equivalent to the nilpotent
        # Neumann series, but avoids repeated matrix squaring, which is
        # numerically unstable for realistic Qwen decay gates.
        # ============================================================
        kb_exp_gc = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=kb_exp_gc,
            data=k_beta,
            op0=nl.multiply,
            operand0=exp_gc_p,
            engine=nisa.vector_engine,
        )

        kbe_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=kbe_T_psum, data=kb_exp_gc)
        kbe_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kbe_T, src=kbe_T_psum)

        kbe_state_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=kbe_state_psum, stationary=kbe_T, moving=state)
        kbe_state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kbe_state, src=kbe_state_psum)

        solve_rhs = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=solve_rhs, data1=v_beta, data2=kbe_state, op=nl.subtract)

        # ============================================================
        # Blocked forward substitution for:
        #   v_new = solve((I - A), solve_rhs)
        #
        # A is strictly lower triangular. Compute each solve block's
        # contribution from previously solved rows with one dense matmul, then
        # solve the small diagonal block row-by-row. This keeps the algebra
        # exact while moving the wide part of the triangular solve onto TE
        # tiles, closer to the FlashQLA/FLA blocked chunked-prefill structure.
        # ============================================================
        v_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=v_new, value=0.0)

        if SOLVE_KKT_HIER:
            _hierarchical_kkt_solve128(v_new, A_T, Imat, solve_rhs, dim)
        else:
            _blocked_doubling_solve(v_new, A_T, solve_rhs, dim)

        # ============================================================
        # Phase 2: Inter-chunk state propagation
        # attn_intra = (q @ k^T) * (strict_decay + identity)
        # ============================================================
        q_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=q_T_psum, data=q_norm)
        q_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q_T, src=q_T_psum)

        # ai_T[j, i] = (q[i] @ k[j]) * transpose(decay_diag)[j, i].
        qk_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=qk_T_psum, stationary=k_T, moving=q_T)
        qk_raw_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qk_raw_t, src=qk_T_psum)

        ai_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=ai_T, data1=qk_raw_t, data2=decay_diag_t, op=nl.multiply
        )

        # ============================================================
        # attn_inter = (q * exp(gc)) @ state   (state is in SBUF!)
        # ============================================================
        q_exp = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=q_exp,
            data=q_norm,
            op0=nl.multiply,
            operand0=exp_gc_p,
            engine=nisa.vector_engine,
        )

        qe_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=qe_T_psum, data=q_exp)
        qe_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qe_T, src=qe_T_psum)

        ai_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ai_psum, stationary=qe_T, moving=state)
        attn_inter = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=attn_inter, src=ai_psum)

        # ============================================================
        # attn_intra @ v_new
        # ============================================================
        intra_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=intra_psum, stationary=ai_T, moving=v_new)
        intra_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=intra_out, src=intra_psum)

        # ============================================================
        # chunk_output = attn_inter + intra_out
        # ============================================================
        chunk_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=chunk_out, data1=attn_inter, data2=intra_out, op=nl.add)

        # Store output chunk to HBM
        if CHUNK_SIZE == P_MAX:
            nisa.dma_copy(
                dst=output[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
                src=chunk_out,
            )
        else:
            nisa.dma_copy(
                dst=output[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
                src=chunk_out[0:CHUNK_SIZE, 0:dim],
            )

        # ============================================================
        # State update: state = exp(g_last) * (state + k_raw_decay^T @ v_new)
        # state is updated IN-PLACE in SBUF — no HBM round-trip!
        # ============================================================

        # k_raw_decay contributes as exp(g_last) * (k * exp(-gc))^T @ v_new.
        # Compute the equivalent stable form k * exp(g_last - gc) directly so
        # no exp(-gc) intermediate can overflow.
        exp_gl_minus_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=exp_gl_minus_gc_p[0:P_MAX, 0:1],
            op=nl.exp,
            data=gc_p[0:P_MAX, 0:1],
            bias=gl_p[0:P_MAX, 0:1],
            scale=-1.0,
        )

        # k_raw_decay = k * exp(g_last - gc)
        k_raw_decay = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_raw_decay,
            data=k_norm,
            op0=nl.multiply,
            operand0=exp_gl_minus_gc_p,
            engine=nisa.vector_engine,
        )

        # k_raw_decay^T @ v_new → (dim, dim) outer product sum
        # nc_matmul: result[M,N] = sum_K stationary[K,M] * moving[K,N]
        # stationary=k_raw_decay (P_MAX, dim), moving=v_new (P_MAX, dim)
        # Result: sum over tokens -> (dim, dim)
        kv_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=kv_psum, stationary=k_raw_decay, moving=v_new)
        kv_outer = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kv_outer, src=kv_psum)

        # state = state * exp(g_last) + kv_outer
        state_decayed = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=state_decayed,
            data=state,
            op0=nl.multiply,
            operand0=exp_gl_p,
            engine=nisa.vector_engine,
        )
        nisa.tensor_tensor(dst=state, data1=state_decayed, data2=kv_outer, op=nl.add)

    # ---- Write final state to HBM ----
    nisa.dma_copy(dst=final_state_out, src=state)

    return output, final_state_out


@nki.jit
def deltanet_autocp_affine_chunk(
    query: nl.ndarray,  # (128, 128) float32 - raw Q; normalized in-kernel
    key: nl.ndarray,  # (128, 128) float32 - raw K; normalized in-kernel
    value: nl.ndarray,  # (128, 128) float32
    g_in: nl.ndarray,  # (128, 1) float32 - per-token log-decay
    beta_in: nl.ndarray,  # (128, 1) float32 - per-token write gate
    lower_mask: nl.ndarray,  # (128, 128) float32 - strict lower tri
    identity: nl.ndarray,  # (128, 128) float32 - identity
    lower_mask_diag: nl.ndarray,  # (128, 128) float32 - lower tri with diag
):
    """Build one chunk's state-independent AutoCP affine pieces.

    For one 128-token DeltaNet chunk:
      output = output_base + output_state @ state
      next_state = state_matrix @ state + state_bias

    This probe deliberately mirrors the fused CTE chunk math and returns the
    four intermediate tensors to HBM for isolated correctness validation before
    wiring an AutoCP prepass into serving.
    """
    dim = query.shape[1]

    output_base_out = nl.ndarray(
        (P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )
    output_state_out = nl.ndarray(
        (P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )
    state_matrix_out = nl.ndarray(
        (P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )
    state_bias_out = nl.ndarray(
        (P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )

    Lmask = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask, src=lower_mask)

    Imat = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Imat, src=identity)

    Lmask_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask_diag, src=lower_mask_diag)

    ones_1xC = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_1xC, value=1.0)

    zero_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero_11, value=0.0)

    q_c = nl.ndarray((P_MAX, dim), dtype=query.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_c, src=query[0:CHUNK_SIZE, 0:dim])

    k_c = nl.ndarray((P_MAX, dim), dtype=key.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_c, src=key[0:CHUNK_SIZE, 0:dim])

    q_square = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=q_square, data1=q_c, data2=q_c, op=nl.multiply)
    q_norm_sq = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=q_norm_sq, data=q_square, op=nl.add, axis=1)
    q_norm_sq_clamped = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=q_norm_sq_clamped,
        data=q_norm_sq,
        op0=nl.maximum,
        operand0=L2_EPS_SQUARED,
        engine=nisa.vector_engine,
    )
    q_inv_norm = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=q_inv_norm,
        data=q_norm_sq_clamped,
        op0=nl.rsqrt,
        operand0=0.0,
        engine=nisa.gpsimd_engine,
    )
    q_norm = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=q_norm,
        data=q_c,
        op0=nl.multiply,
        operand0=q_inv_norm,
        op1=nl.multiply,
        operand1=QUERY_SCALE,
        engine=nisa.vector_engine,
    )

    k_square = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=k_square, data1=k_c, data2=k_c, op=nl.multiply)
    k_norm_sq = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=k_norm_sq, data=k_square, op=nl.add, axis=1)
    k_norm_sq_clamped = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=k_norm_sq_clamped,
        data=k_norm_sq,
        op0=nl.maximum,
        operand0=L2_EPS_SQUARED,
        engine=nisa.vector_engine,
    )
    k_inv_norm = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=k_inv_norm,
        data=k_norm_sq_clamped,
        op0=nl.rsqrt,
        operand0=0.0,
        engine=nisa.gpsimd_engine,
    )
    k_norm = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=k_norm,
        data=k_c,
        op0=nl.multiply,
        operand0=k_inv_norm,
        engine=nisa.vector_engine,
    )

    v_c = nl.ndarray((P_MAX, dim), dtype=value.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=v_c, src=value[0:CHUNK_SIZE, 0:dim])

    g_chunk_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=g_chunk_p[0:CHUNK_SIZE, 0:1], src=g_in[0:CHUNK_SIZE, 0:1])

    beta_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=beta_p[0:CHUNK_SIZE, 0:1], src=beta_in[0:CHUNK_SIZE, 0:1])

    g_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=g_padded, value=0.0)
    nisa.tensor_copy(dst=g_padded[0:CHUNK_SIZE, 0:1], src=g_chunk_p)

    g_tp_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=g_tp_psum, data=g_padded)

    g_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=g_row[0:1, 0:CHUNK_SIZE], src=g_tp_psum[0:1, 0:CHUNK_SIZE])

    gc_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor_scan(
        dst=gc_row[0:1, 0:CHUNK_SIZE],
        data0=ones_1xC[0:1, 0:CHUNK_SIZE],
        data1=g_row[0:1, 0:CHUNK_SIZE],
        initial=zero_11[0:1, 0:1],
        op0=nl.multiply,
        op1=nl.add,
    )

    gc_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=gc_padded, value=0.0)
    nisa.tensor_copy(dst=gc_padded[0:1, 0:CHUNK_SIZE], src=gc_row)

    gc_tp_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=gc_tp_psum, data=gc_padded)

    gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gc_p[0:CHUNK_SIZE, 0:1], src=gc_tp_psum[0:CHUNK_SIZE, 0:1])

    gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=gl_11[0:1, 0:1],
        src=gc_row[0:1, CHUNK_SIZE - 1 : CHUNK_SIZE],
    )

    exp_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=exp_gc_p[0:P_MAX, 0:1],
        op=nl.exp,
        data=gc_p[0:P_MAX, 0:1],
        bias=None,
        scale=1.0,
    )

    gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    for i_shuf in nl.static_range(P_MAX // 32):
        nisa.nc_stream_shuffle(
            src=gl_11[0:1, 0:1],
            dst=gl_p[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
            shuffle_mask=_BROADCAST_MASK,
        )

    exp_gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=exp_gl_11,
        op=nl.exp,
        data=gl_11,
        bias=None,
        scale=1.0,
    )

    exp_gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    for i_shuf in nl.static_range(P_MAX // 32):
        nisa.nc_stream_shuffle(
            src=exp_gl_11[0:1, 0:1],
            dst=exp_gl_p[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
            shuffle_mask=_BROADCAST_MASK,
        )

    gc_row_broadcast = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    for i_shuf in nl.static_range(P_MAX // 32):
        nisa.nc_stream_shuffle(
            src=gc_row[0:1, 0:P_MAX],
            dst=gc_row_broadcast[i_shuf * 32 : i_shuf * 32 + 32, 0:P_MAX],
            shuffle_mask=_BROADCAST_MASK,
        )

    gc_col_strict = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=gc_col_strict,
        data=Lmask,
        op0=nl.multiply,
        operand0=gc_p,
        engine=nisa.vector_engine,
    )
    gc_row_strict = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=gc_row_strict, data1=gc_row_broadcast, data2=Lmask, op=nl.multiply
    )
    g_diff_strict = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=g_diff_strict,
        data1=gc_col_strict,
        data2=gc_row_strict,
        op=nl.subtract,
    )
    decay_strict_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=decay_strict_raw,
        op=nl.exp,
        data=g_diff_strict,
        bias=None,
        scale=1.0,
    )
    decay_strict = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=decay_strict, data1=decay_strict_raw, data2=Lmask, op=nl.multiply
    )

    gc_col_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=gc_col_diag,
        data=Lmask_diag,
        op0=nl.multiply,
        operand0=gc_p,
        engine=nisa.vector_engine,
    )
    gc_row_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=gc_row_diag,
        data1=gc_row_broadcast,
        data2=Lmask_diag,
        op=nl.multiply,
    )
    g_diff_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=g_diff_diag,
        data1=gc_col_diag,
        data2=gc_row_diag,
        op=nl.subtract,
    )
    decay_diag_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=decay_diag_raw,
        op=nl.exp,
        data=g_diff_diag,
        bias=None,
        scale=1.0,
    )
    decay_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=decay_diag, data1=decay_diag_raw, data2=Lmask_diag, op=nl.multiply
    )

    k_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=k_beta,
        data=k_norm,
        op0=nl.multiply,
        operand0=beta_p,
        engine=nisa.vector_engine,
    )

    v_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=v_beta,
        data=v_c,
        op0=nl.multiply,
        operand0=beta_p,
        engine=nisa.vector_engine,
    )

    kb_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=kb_T_psum, data=k_beta)
    kb_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kb_T, src=kb_T_psum)

    k_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=k_T_psum, data=k_norm)
    k_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=k_T, src=k_T_psum)

    QK_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=QK_psum, stationary=kb_T, moving=k_T)
    QK = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=QK, src=QK_psum)

    QK_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=QK_decay, data1=QK, data2=decay_strict, op=nl.multiply)
    neg_QK_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=neg_QK_decay,
        data=QK_decay,
        op0=nl.multiply,
        operand0=-1.0,
        engine=nisa.vector_engine,
    )

    A_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=A_T_psum, data=neg_QK_decay)
    A_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=A_T, src=A_T_psum)

    value_u = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=value_u, value=0.0)
    if SOLVE_KKT_HIER:
        _hierarchical_kkt_solve128(value_u, A_T, Imat, v_beta, dim)
    else:
        _blocked_doubling_solve(value_u, A_T, v_beta, dim)

    kb_exp_gc = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=kb_exp_gc,
        data=k_beta,
        op0=nl.multiply,
        operand0=exp_gc_p,
        engine=nisa.vector_engine,
    )
    state_w = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=state_w, value=0.0)
    if SOLVE_KKT_HIER:
        _hierarchical_kkt_solve128(state_w, A_T, Imat, kb_exp_gc, dim)
    else:
        _blocked_doubling_solve(state_w, A_T, kb_exp_gc, dim)

    q_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=q_T_psum, data=q_norm)
    q_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=q_T, src=q_T_psum)

    qk_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=qk_psum, stationary=q_T, moving=k_T)
    qk_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qk_raw, src=qk_psum)

    attn_intra = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=attn_intra, data1=qk_raw, data2=decay_diag, op=nl.multiply
    )

    ai_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=ai_T_psum, data=attn_intra)
    ai_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ai_T, src=ai_T_psum)

    output_base_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=output_base_psum, stationary=ai_T, moving=value_u)
    output_base = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=output_base, src=output_base_psum)

    q_exp = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=q_exp,
        data=q_norm,
        op0=nl.multiply,
        operand0=exp_gc_p,
        engine=nisa.vector_engine,
    )

    output_state_corr_psum = nl.ndarray(
        (P_MAX, dim), dtype=nl.float32, buffer=nl.psum
    )
    nisa.nc_matmul(dst=output_state_corr_psum, stationary=ai_T, moving=state_w)
    output_state_corr = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=output_state_corr, src=output_state_corr_psum)

    output_state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=output_state,
        data1=q_exp,
        data2=output_state_corr,
        op=nl.subtract,
    )

    gl_minus_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=gl_minus_gc_p,
        data1=gl_p,
        data2=gc_p,
        op=nl.subtract,
    )
    exp_gl_minus_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=exp_gl_minus_gc_p[0:P_MAX, 0:1],
        op=nl.exp,
        data=gl_minus_gc_p[0:P_MAX, 0:1],
        bias=None,
        scale=1.0,
    )

    k_raw_decay = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=k_raw_decay,
        data=k_norm,
        op0=nl.multiply,
        operand0=exp_gl_minus_gc_p,
        engine=nisa.vector_engine,
    )

    state_bias_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=state_bias_psum, stationary=k_raw_decay, moving=value_u)
    state_bias = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=state_bias, src=state_bias_psum)

    state_corr_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=state_corr_psum, stationary=k_raw_decay, moving=state_w)
    state_corr = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=state_corr, src=state_corr_psum)

    exp_gl_identity = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=exp_gl_identity,
        data=Imat,
        op0=nl.multiply,
        operand0=exp_gl_p,
        engine=nisa.vector_engine,
    )
    state_matrix = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=state_matrix,
        data1=exp_gl_identity,
        data2=state_corr,
        op=nl.subtract,
    )

    nisa.dma_copy(dst=output_base_out, src=output_base)
    nisa.dma_copy(dst=output_state_out, src=output_state)
    nisa.dma_copy(dst=state_matrix_out, src=state_matrix)
    nisa.dma_copy(dst=state_bias_out, src=state_bias)

    return output_base_out, output_state_out, state_matrix_out, state_bias_out


@nki.jit
def deltanet_autocp_affine_sequence(
    query: nl.ndarray,  # (S, 128) float32 - raw Q; normalized in-kernel
    key: nl.ndarray,  # (S, 128) float32 - raw K; normalized in-kernel
    value: nl.ndarray,  # (S, 128) float32
    g_in: nl.ndarray,  # (S, 1) float32 - per-token log-decay
    beta_in: nl.ndarray,  # (S, 1) float32 - per-token write gate
    lower_mask: nl.ndarray,  # (128, 128) float32 - strict lower tri
    identity: nl.ndarray,  # (128, 128) float32 - identity
    lower_mask_diag: nl.ndarray,  # (128, 128) float32 - kept for call compatibility
):
    """Build AutoCP affine pieces for one sequence with LNC-striped chunks."""
    seq_len = query.shape[0]
    dim = query.shape[1]
    num_chunks = seq_len // CHUNK_SIZE
    program_idx = nl.program_id(axis=0)
    num_programs = nl.num_programs(axes=0)
    chunks_per_program = num_chunks // num_programs

    output_base_out = nl.ndarray(
        (num_chunks, P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )
    output_state_out = nl.ndarray(
        (num_chunks, P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )
    state_matrix_out = nl.ndarray(
        (num_chunks, P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )
    state_bias_out = nl.ndarray(
        (num_chunks, P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )

    Lmask = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask, src=lower_mask)

    Imat = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Imat, src=identity)

    ones_1xC = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_1xC, value=1.0)

    zero_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero_11, value=0.0)

    for chunk_loop in nl.sequential_range(chunks_per_program):
        chunk_idx = program_idx * chunks_per_program + chunk_loop
        chunk_start = chunk_idx * CHUNK_SIZE

        q_c = nl.ndarray((P_MAX, dim), dtype=query.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=q_c,
            src=query[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
        )

        k_c = nl.ndarray((P_MAX, dim), dtype=key.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=k_c,
            src=key[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
        )

        q_square = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=q_square, data1=q_c, data2=q_c, op=nl.multiply)
        q_norm_sq = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=q_norm_sq, data=q_square, op=nl.add, axis=1)
        q_norm_sq_clamped = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=q_norm_sq_clamped,
            data=q_norm_sq,
            op0=nl.maximum,
            operand0=L2_EPS_SQUARED,
            engine=nisa.vector_engine,
        )
        q_inv_norm = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=q_inv_norm,
            data=q_norm_sq_clamped,
            op0=nl.rsqrt,
            operand0=0.0,
            engine=nisa.gpsimd_engine,
        )
        q_norm = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=q_norm,
            data=q_c,
            op0=nl.multiply,
            operand0=q_inv_norm,
            op1=nl.multiply,
            operand1=QUERY_SCALE,
            engine=nisa.vector_engine,
        )

        k_square = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=k_square, data1=k_c, data2=k_c, op=nl.multiply)
        k_norm_sq = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=k_norm_sq, data=k_square, op=nl.add, axis=1)
        k_norm_sq_clamped = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_norm_sq_clamped,
            data=k_norm_sq,
            op0=nl.maximum,
            operand0=L2_EPS_SQUARED,
            engine=nisa.vector_engine,
        )
        k_inv_norm = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_inv_norm,
            data=k_norm_sq_clamped,
            op0=nl.rsqrt,
            operand0=0.0,
            engine=nisa.gpsimd_engine,
        )
        k_norm = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_norm,
            data=k_c,
            op0=nl.multiply,
            operand0=k_inv_norm,
            engine=nisa.vector_engine,
        )

        v_c = nl.ndarray((P_MAX, dim), dtype=value.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=v_c,
            src=value[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
        )

        g_chunk_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=g_chunk_p[0:CHUNK_SIZE, 0:1],
            src=g_in[chunk_start : chunk_start + CHUNK_SIZE, 0:1],
        )

        beta_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=beta_p[0:CHUNK_SIZE, 0:1],
            src=beta_in[chunk_start : chunk_start + CHUNK_SIZE, 0:1],
        )

        g_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=g_padded, value=0.0)
        nisa.tensor_copy(dst=g_padded[0:CHUNK_SIZE, 0:1], src=g_chunk_p)

        g_tp_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=g_tp_psum, data=g_padded)

        g_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=g_row[0:1, 0:CHUNK_SIZE], src=g_tp_psum[0:1, 0:CHUNK_SIZE])

        gc_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor_scan(
            dst=gc_row[0:1, 0:CHUNK_SIZE],
            data0=ones_1xC[0:1, 0:CHUNK_SIZE],
            data1=g_row[0:1, 0:CHUNK_SIZE],
            initial=zero_11[0:1, 0:1],
            op0=nl.multiply,
            op1=nl.add,
        )

        gc_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=gc_padded, value=0.0)
        nisa.tensor_copy(dst=gc_padded[0:1, 0:CHUNK_SIZE], src=gc_row)

        gc_tp_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=gc_tp_psum, data=gc_padded)

        gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gc_p[0:CHUNK_SIZE, 0:1], src=gc_tp_psum[0:CHUNK_SIZE, 0:1])

        gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=gl_11[0:1, 0:1],
            src=gc_row[0:1, CHUNK_SIZE - 1 : CHUNK_SIZE],
        )

        exp_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=exp_gc_p[0:P_MAX, 0:1],
            op=nl.exp,
            data=gc_p[0:P_MAX, 0:1],
            bias=None,
            scale=1.0,
        )

        exp_gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=exp_gl_11,
            op=nl.exp,
            data=gl_11,
            bias=None,
            scale=1.0,
        )

        gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=gl_11[0:1, 0:1],
                dst=gl_p[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                shuffle_mask=_BROADCAST_MASK,
            )

        exp_gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=exp_gl_11[0:1, 0:1],
                dst=exp_gl_p[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                shuffle_mask=_BROADCAST_MASK,
            )

        gc_row_broadcast = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=gc_row[0:1, 0:P_MAX],
                dst=gc_row_broadcast[i_shuf * 32 : i_shuf * 32 + 32, 0:P_MAX],
                shuffle_mask=_BROADCAST_MASK,
            )

        gc_col_strict = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=gc_col_strict,
            data=Lmask,
            op0=nl.multiply,
            operand0=gc_p,
            engine=nisa.vector_engine,
        )
        gc_row_strict = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=gc_row_strict, data1=gc_row_broadcast, data2=Lmask, op=nl.multiply
        )
        g_diff_strict = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=g_diff_strict,
            data1=gc_col_strict,
            data2=gc_row_strict,
            op=nl.subtract,
        )
        decay_strict_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=decay_strict_raw,
            op=nl.exp,
            data=g_diff_strict,
            bias=None,
            scale=1.0,
        )
        decay_strict = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=decay_strict, data1=decay_strict_raw, data2=Lmask, op=nl.multiply
        )

        decay_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=decay_diag, data1=decay_strict, data2=Imat, op=nl.add)

        k_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_beta,
            data=k_norm,
            op0=nl.multiply,
            operand0=beta_p,
            engine=nisa.vector_engine,
        )

        v_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=v_beta,
            data=v_c,
            op0=nl.multiply,
            operand0=beta_p,
            engine=nisa.vector_engine,
        )

        kb_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=kb_T_psum, data=k_beta)
        kb_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kb_T, src=kb_T_psum)

        k_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=k_T_psum, data=k_norm)
        k_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=k_T, src=k_T_psum)

        QK_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=QK_psum, stationary=kb_T, moving=k_T)
        QK = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=QK, src=QK_psum)

        QK_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=QK_decay, data1=QK, data2=decay_strict, op=nl.multiply)
        neg_QK_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=neg_QK_decay,
            data=QK_decay,
            op0=nl.multiply,
            operand0=-1.0,
            engine=nisa.vector_engine,
        )

        A_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=A_T_psum, data=neg_QK_decay)
        A_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=A_T, src=A_T_psum)

        value_u = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=value_u, value=0.0)
        if SOLVE_KKT_HIER:
            _hierarchical_kkt_solve128(value_u, A_T, Imat, v_beta, dim)
        else:
            _blocked_doubling_solve(value_u, A_T, v_beta, dim)

        kb_exp_gc = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=kb_exp_gc,
            data=k_beta,
            op0=nl.multiply,
            operand0=exp_gc_p,
            engine=nisa.vector_engine,
        )
        state_w = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=state_w, value=0.0)
        if SOLVE_KKT_HIER:
            _hierarchical_kkt_solve128(state_w, A_T, Imat, kb_exp_gc, dim)
        else:
            _blocked_doubling_solve(state_w, A_T, kb_exp_gc, dim)

        q_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=q_T_psum, data=q_norm)
        q_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q_T, src=q_T_psum)

        qk_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=qk_psum, stationary=q_T, moving=k_T)
        qk_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qk_raw, src=qk_psum)

        attn_intra = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=attn_intra, data1=qk_raw, data2=decay_diag, op=nl.multiply
        )

        ai_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=ai_T_psum, data=attn_intra)
        ai_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ai_T, src=ai_T_psum)

        output_base_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=output_base_psum, stationary=ai_T, moving=value_u)
        output_base = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=output_base, src=output_base_psum)

        q_exp = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=q_exp,
            data=q_norm,
            op0=nl.multiply,
            operand0=exp_gc_p,
            engine=nisa.vector_engine,
        )

        output_state_corr_psum = nl.ndarray(
            (P_MAX, dim), dtype=nl.float32, buffer=nl.psum
        )
        nisa.nc_matmul(dst=output_state_corr_psum, stationary=ai_T, moving=state_w)
        output_state_corr = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=output_state_corr, src=output_state_corr_psum)

        output_state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=output_state,
            data1=q_exp,
            data2=output_state_corr,
            op=nl.subtract,
        )

        gl_minus_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=gl_minus_gc_p,
            data1=gl_p,
            data2=gc_p,
            op=nl.subtract,
        )
        exp_gl_minus_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=exp_gl_minus_gc_p[0:P_MAX, 0:1],
            op=nl.exp,
            data=gl_minus_gc_p[0:P_MAX, 0:1],
            bias=None,
            scale=1.0,
        )

        k_raw_decay = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_raw_decay,
            data=k_norm,
            op0=nl.multiply,
            operand0=exp_gl_minus_gc_p,
            engine=nisa.vector_engine,
        )

        state_bias_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=state_bias_psum, stationary=k_raw_decay, moving=value_u)
        state_bias = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=state_bias, src=state_bias_psum)

        state_corr_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=state_corr_psum, stationary=k_raw_decay, moving=state_w)
        state_corr = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=state_corr, src=state_corr_psum)

        exp_gl_identity = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=exp_gl_identity,
            data=Imat,
            op0=nl.multiply,
            operand0=exp_gl_p,
            engine=nisa.vector_engine,
        )
        state_matrix = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=state_matrix,
            data1=exp_gl_identity,
            data2=state_corr,
            op=nl.subtract,
        )

        nisa.dma_copy(dst=output_base_out[chunk_idx, 0:P_MAX, 0:dim], src=output_base)
        nisa.dma_copy(dst=output_state_out[chunk_idx, 0:P_MAX, 0:dim], src=output_state)
        nisa.dma_copy(dst=state_matrix_out[chunk_idx, 0:P_MAX, 0:dim], src=state_matrix)
        nisa.dma_copy(dst=state_bias_out[chunk_idx, 0:P_MAX, 0:dim], src=state_bias)

    return output_base_out, output_state_out, state_matrix_out, state_bias_out


@nki.jit
def deltanet_autocp_state_summary_sequence(
    key: nl.ndarray,  # (S, 128) float32 - raw K; normalized in-kernel
    value: nl.ndarray,  # (S, 128) float32
    g_in: nl.ndarray,  # (S, 1) float32 - per-token log-decay
    beta_in: nl.ndarray,  # (S, 1) float32 - per-token write gate
    lower_mask: nl.ndarray,  # (128, 128) float32 - strict lower tri
    identity: nl.ndarray,  # (128, 128) float32 - identity
):
    """Build compact AutoCP segment state summaries.

    This is the first production-shaped AutoCP prepass: it skips query/output
    affine pieces and emits only per-segment state transforms:

        state_{seg+1} = segment_matrix_seg @ state_seg + segment_bias_seg

    Segment replay can then use the existing recurrent fused kernel from the
    corrected segment initial states.
    """
    seq_len = key.shape[0]
    dim = key.shape[1]
    num_chunks = seq_len // CHUNK_SIZE
    num_segments = num_chunks // AUTOCP_CP_CHUNKS
    program_idx = nl.program_id(axis=0)
    num_programs = nl.num_programs(axes=0)
    segments_per_program = num_segments // num_programs

    segment_matrix_out = nl.ndarray(
        (num_segments, P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )
    segment_bias_out = nl.ndarray(
        (num_segments, P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )

    Lmask = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask, src=lower_mask)

    Imat = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Imat, src=identity)

    ones_1xC = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_1xC, value=1.0)

    zero_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero_11, value=0.0)

    for segment_loop in nl.sequential_range(segments_per_program):
        segment_idx = program_idx * segments_per_program + segment_loop
        first_chunk = segment_idx * AUTOCP_CP_CHUNKS

        segment_matrix = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=segment_matrix, src=Imat)

        segment_bias = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=segment_bias, value=0.0)

        for local_chunk in nl.sequential_range(AUTOCP_CP_CHUNKS):
            chunk_idx = first_chunk + local_chunk
            chunk_start = chunk_idx * CHUNK_SIZE

            k_c = nl.ndarray((P_MAX, dim), dtype=key.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=k_c,
                src=key[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
            )

            k_square = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=k_square, data1=k_c, data2=k_c, op=nl.multiply)
            k_norm_sq = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=k_norm_sq, data=k_square, op=nl.add, axis=1)
            k_norm_sq_clamped = nl.ndarray(
                (P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=k_norm_sq_clamped,
                data=k_norm_sq,
                op0=nl.maximum,
                operand0=L2_EPS_SQUARED,
                engine=nisa.vector_engine,
            )
            k_inv_norm = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=k_inv_norm,
                data=k_norm_sq_clamped,
                op0=nl.rsqrt,
                operand0=0.0,
                engine=nisa.gpsimd_engine,
            )
            k_norm = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=k_norm,
                data=k_c,
                op0=nl.multiply,
                operand0=k_inv_norm,
                engine=nisa.vector_engine,
            )

            v_c = nl.ndarray((P_MAX, dim), dtype=value.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=v_c,
                src=value[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
            )

            g_chunk_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=g_chunk_p[0:CHUNK_SIZE, 0:1],
                src=g_in[chunk_start : chunk_start + CHUNK_SIZE, 0:1],
            )

            beta_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=beta_p[0:CHUNK_SIZE, 0:1],
                src=beta_in[chunk_start : chunk_start + CHUNK_SIZE, 0:1],
            )

            g_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=g_padded, value=0.0)
            nisa.tensor_copy(dst=g_padded[0:CHUNK_SIZE, 0:1], src=g_chunk_p)

            g_tp_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=g_tp_psum, data=g_padded)

            g_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=g_row[0:1, 0:CHUNK_SIZE],
                src=g_tp_psum[0:1, 0:CHUNK_SIZE],
            )

            gc_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor_scan(
                dst=gc_row[0:1, 0:CHUNK_SIZE],
                data0=ones_1xC[0:1, 0:CHUNK_SIZE],
                data1=g_row[0:1, 0:CHUNK_SIZE],
                initial=zero_11[0:1, 0:1],
                op0=nl.multiply,
                op1=nl.add,
            )

            gc_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=gc_padded, value=0.0)
            nisa.tensor_copy(dst=gc_padded[0:1, 0:CHUNK_SIZE], src=gc_row)

            gc_tp_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=gc_tp_psum, data=gc_padded)

            gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=gc_p[0:CHUNK_SIZE, 0:1],
                src=gc_tp_psum[0:CHUNK_SIZE, 0:1],
            )

            gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=gl_11[0:1, 0:1],
                src=gc_row[0:1, CHUNK_SIZE - 1 : CHUNK_SIZE],
            )

            exp_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=exp_gc_p[0:P_MAX, 0:1],
                op=nl.exp,
                data=gc_p[0:P_MAX, 0:1],
                bias=None,
                scale=1.0,
            )

            exp_gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=exp_gl_11,
                op=nl.exp,
                data=gl_11,
                bias=None,
                scale=1.0,
            )

            gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            for i_shuf in nl.static_range(P_MAX // 32):
                nisa.nc_stream_shuffle(
                    src=gl_11[0:1, 0:1],
                    dst=gl_p[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                    shuffle_mask=_BROADCAST_MASK,
                )

            exp_gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            for i_shuf in nl.static_range(P_MAX // 32):
                nisa.nc_stream_shuffle(
                    src=exp_gl_11[0:1, 0:1],
                    dst=exp_gl_p[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                    shuffle_mask=_BROADCAST_MASK,
                )

            gc_row_broadcast = nl.ndarray(
                (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
            )
            for i_shuf in nl.static_range(P_MAX // 32):
                nisa.nc_stream_shuffle(
                    src=gc_row[0:1, 0:P_MAX],
                    dst=gc_row_broadcast[i_shuf * 32 : i_shuf * 32 + 32, 0:P_MAX],
                    shuffle_mask=_BROADCAST_MASK,
                )

            gc_col_strict = nl.ndarray(
                (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=gc_col_strict,
                data=Lmask,
                op0=nl.multiply,
                operand0=gc_p,
                engine=nisa.vector_engine,
            )
            gc_row_strict = nl.ndarray(
                (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_tensor(
                dst=gc_row_strict,
                data1=gc_row_broadcast,
                data2=Lmask,
                op=nl.multiply,
            )
            g_diff_strict = nl.ndarray(
                (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_tensor(
                dst=g_diff_strict,
                data1=gc_col_strict,
                data2=gc_row_strict,
                op=nl.subtract,
            )
            decay_strict_raw = nl.ndarray(
                (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.activation(
                dst=decay_strict_raw,
                op=nl.exp,
                data=g_diff_strict,
                bias=None,
                scale=1.0,
            )
            decay_strict = nl.ndarray(
                (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_tensor(
                dst=decay_strict,
                data1=decay_strict_raw,
                data2=Lmask,
                op=nl.multiply,
            )

            k_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=k_beta,
                data=k_norm,
                op0=nl.multiply,
                operand0=beta_p,
                engine=nisa.vector_engine,
            )

            v_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=v_beta,
                data=v_c,
                op0=nl.multiply,
                operand0=beta_p,
                engine=nisa.vector_engine,
            )

            kb_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=kb_T_psum, data=k_beta)
            kb_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=kb_T, src=kb_T_psum)

            k_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=k_T_psum, data=k_norm)
            k_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=k_T, src=k_T_psum)

            QK_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=QK_psum, stationary=kb_T, moving=k_T)
            QK = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=QK, src=QK_psum)

            QK_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=QK_decay,
                data1=QK,
                data2=decay_strict,
                op=nl.multiply,
            )
            neg_QK_decay = nl.ndarray(
                (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=neg_QK_decay,
                data=QK_decay,
                op0=nl.multiply,
                operand0=-1.0,
                engine=nisa.vector_engine,
            )

            A_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=A_T_psum, data=neg_QK_decay)
            A_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=A_T, src=A_T_psum)

            value_u = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=value_u, value=0.0)
            if SOLVE_KKT_HIER:
                _hierarchical_kkt_solve128(value_u, A_T, Imat, v_beta, dim)
            else:
                _blocked_doubling_solve(value_u, A_T, v_beta, dim)

            kb_exp_gc = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=kb_exp_gc,
                data=k_beta,
                op0=nl.multiply,
                operand0=exp_gc_p,
                engine=nisa.vector_engine,
            )
            state_w = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=state_w, value=0.0)
            if SOLVE_KKT_HIER:
                _hierarchical_kkt_solve128(state_w, A_T, Imat, kb_exp_gc, dim)
            else:
                _blocked_doubling_solve(state_w, A_T, kb_exp_gc, dim)

            gl_minus_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=gl_minus_gc_p,
                data1=gl_p,
                data2=gc_p,
                op=nl.subtract,
            )
            exp_gl_minus_gc_p = nl.ndarray(
                (P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.activation(
                dst=exp_gl_minus_gc_p[0:P_MAX, 0:1],
                op=nl.exp,
                data=gl_minus_gc_p[0:P_MAX, 0:1],
                bias=None,
                scale=1.0,
            )

            k_raw_decay = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=k_raw_decay,
                data=k_norm,
                op0=nl.multiply,
                operand0=exp_gl_minus_gc_p,
                engine=nisa.vector_engine,
            )

            state_bias = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            state_bias_psum = nl.ndarray(
                (P_MAX, dim), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(dst=state_bias_psum, stationary=k_raw_decay, moving=value_u)
            nisa.tensor_copy(dst=state_bias, src=state_bias_psum)

            state_corr_psum = nl.ndarray(
                (P_MAX, dim), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(dst=state_corr_psum, stationary=k_raw_decay, moving=state_w)
            state_corr = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=state_corr, src=state_corr_psum)

            exp_gl_identity = nl.ndarray(
                (P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_scalar(
                dst=exp_gl_identity,
                data=Imat,
                op0=nl.multiply,
                operand0=exp_gl_p,
                engine=nisa.vector_engine,
            )
            state_matrix = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=state_matrix,
                data1=exp_gl_identity,
                data2=state_corr,
                op=nl.subtract,
            )

            state_matrix_t_psum = nl.ndarray(
                (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_transpose(dst=state_matrix_t_psum, data=state_matrix)
            state_matrix_t = nl.ndarray(
                (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=state_matrix_t, src=state_matrix_t_psum)

            composed_matrix_psum = nl.ndarray(
                (P_MAX, dim), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(
                dst=composed_matrix_psum,
                stationary=state_matrix_t,
                moving=segment_matrix,
            )
            composed_matrix = nl.ndarray(
                (P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=composed_matrix, src=composed_matrix_psum)

            propagated_bias_psum = nl.ndarray(
                (P_MAX, dim), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(
                dst=propagated_bias_psum,
                stationary=state_matrix_t,
                moving=segment_bias,
            )
            propagated_bias = nl.ndarray(
                (P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=propagated_bias, src=propagated_bias_psum)

            composed_bias = nl.ndarray(
                (P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_tensor(
                dst=composed_bias,
                data1=propagated_bias,
                data2=state_bias,
                op=nl.add,
            )

            nisa.tensor_copy(dst=segment_matrix, src=composed_matrix)
            nisa.tensor_copy(dst=segment_bias, src=composed_bias)

        nisa.dma_copy(
            dst=segment_matrix_out[segment_idx, 0:P_MAX, 0:dim],
            src=segment_matrix,
        )
        nisa.dma_copy(
            dst=segment_bias_out[segment_idx, 0:P_MAX, 0:dim],
            src=segment_bias,
        )

    return segment_matrix_out, segment_bias_out


@nki.jit
def deltanet_autocp_state_prefix(
    state_matrix: nl.ndarray,  # (N, 128, 128) float32
    state_bias: nl.ndarray,  # (N, 128, 128) float32
    initial_state: nl.ndarray,  # (128, 128) float32
):
    """Apply per-chunk AutoCP state transforms and emit chunk initial states.

    Given per-chunk transforms:
      state_{i+1} = state_matrix_i @ state_i + state_bias_i

    returns:
      chunk_states[i] = state_i
      final_state = state_N

    This is the isolated state-prefix correctness probe. A later production
    path can replace the loop body with a tree/parallel prefix over the same
    HBM interface.
    """
    num_chunks = state_matrix.shape[0]
    dim = initial_state.shape[1]

    chunk_states_out = nl.ndarray(
        (num_chunks, P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )
    final_state_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm)

    state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=state, src=initial_state[0:P_MAX, 0:dim])

    for i_chunk in nl.sequential_range(num_chunks):
        nisa.dma_copy(
            dst=chunk_states_out[i_chunk, 0:P_MAX, 0:dim],
            src=state,
        )

        matrix = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=matrix,
            src=state_matrix[i_chunk, 0:P_MAX, 0:dim],
        )

        bias = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=bias,
            src=state_bias[i_chunk, 0:P_MAX, 0:dim],
        )

        matrix_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=matrix_t_psum, data=matrix)
        matrix_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=matrix_t, src=matrix_t_psum)

        propagated_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=propagated_psum, stationary=matrix_t, moving=state)
        propagated = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=propagated, src=propagated_psum)

        next_state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=next_state,
            data1=propagated,
            data2=bias,
            op=nl.add,
        )
        nisa.tensor_copy(dst=state, src=next_state)

    nisa.dma_copy(dst=final_state_out, src=state)

    return chunk_states_out, final_state_out


@nki.jit
def deltanet_autocp_apply_output(
    output_base: nl.ndarray,  # (N, 128, 128) float32
    output_state: nl.ndarray,  # (N, 128, 128) float32
    chunk_states: nl.ndarray,  # (N, 128, 128) float32
):
    """Apply AutoCP chunk initial states to state-dependent output terms."""
    num_chunks = output_base.shape[0]
    dim = output_base.shape[2]

    output = nl.ndarray(
        (num_chunks * CHUNK_SIZE, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )

    for i_chunk in nl.sequential_range(num_chunks):
        base = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=base,
            src=output_base[i_chunk, 0:P_MAX, 0:dim],
        )

        state_coeff = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=state_coeff,
            src=output_state[i_chunk, 0:P_MAX, 0:dim],
        )

        state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=state,
            src=chunk_states[i_chunk, 0:P_MAX, 0:dim],
        )

        coeff_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=coeff_t_psum, data=state_coeff)
        coeff_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=coeff_t, src=coeff_t_psum)

        state_out_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=state_out_psum, stationary=coeff_t, moving=state)
        state_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=state_out, src=state_out_psum)

        chunk_output = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=chunk_output,
            data1=base,
            data2=state_out,
            op=nl.add,
        )

        nisa.dma_copy(
            dst=output[i_chunk * CHUNK_SIZE : i_chunk * CHUNK_SIZE + CHUNK_SIZE, 0:dim],
            src=chunk_output,
        )

    return output


@nki.jit
def deltanet_autocp_prefix_apply_output(
    output_base: nl.ndarray,  # (N, 128, 128) float32
    output_state: nl.ndarray,  # (N, 128, 128) float32
    state_matrix: nl.ndarray,  # (N, 128, 128) float32
    state_bias: nl.ndarray,  # (N, 128, 128) float32
    initial_state: nl.ndarray,  # (128, 128) float32
):
    """Fused AutoCP state-prefix and output-apply pass.

    This removes the intermediate chunk_states HBM tensor and one custom-call
    from the AutoCP probe path. It intentionally remains an exact sequential
    prefix over dense 128x128 chunk transforms; a matrix-affine prefix cannot be
    represented by tensor_tensor_scan's elementwise recurrence.
    """
    num_chunks = output_base.shape[0]
    dim = output_base.shape[2]

    output = nl.ndarray(
        (num_chunks * CHUNK_SIZE, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )
    final_state_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm)

    state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=state, src=initial_state[0:P_MAX, 0:dim])

    for i_chunk in nl.sequential_range(num_chunks):
        base = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=base,
            src=output_base[i_chunk, 0:P_MAX, 0:dim],
        )

        state_coeff = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=state_coeff,
            src=output_state[i_chunk, 0:P_MAX, 0:dim],
        )

        coeff_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=coeff_t_psum, data=state_coeff)
        coeff_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=coeff_t, src=coeff_t_psum)

        state_out_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=state_out_psum, stationary=coeff_t, moving=state)
        state_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=state_out, src=state_out_psum)

        chunk_output = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=chunk_output,
            data1=base,
            data2=state_out,
            op=nl.add,
        )
        nisa.dma_copy(
            dst=output[i_chunk * CHUNK_SIZE : i_chunk * CHUNK_SIZE + CHUNK_SIZE, 0:dim],
            src=chunk_output,
        )

        matrix = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=matrix,
            src=state_matrix[i_chunk, 0:P_MAX, 0:dim],
        )

        bias = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=bias,
            src=state_bias[i_chunk, 0:P_MAX, 0:dim],
        )

        matrix_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=matrix_t_psum, data=matrix)
        matrix_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=matrix_t, src=matrix_t_psum)

        propagated_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=propagated_psum, stationary=matrix_t, moving=state)
        propagated = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=propagated, src=propagated_psum)

        next_state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=next_state,
            data1=propagated,
            data2=bias,
            op=nl.add,
        )
        nisa.tensor_copy(dst=state, src=next_state)

    nisa.dma_copy(dst=final_state_out, src=state)

    return output, final_state_out


@nki.jit
def deltanet_fused_chunked_fwd_multihead(
    query: nl.ndarray,  # (BH, S, 128) float32 — raw Q; normalized in-kernel
    key: nl.ndarray,  # (BH, S, 128) float32 — raw K; normalized in-kernel
    value: nl.ndarray,  # (BH, S, 128) float32
    g_in: nl.ndarray,  # (BH, S, 1) float32
    beta_in: nl.ndarray,  # (BH, S, 1) float32
    initial_state: nl.ndarray,  # (BH, 128, 128) float32
    lower_mask: nl.ndarray,  # (128, 128) float32
    identity: nl.ndarray,  # (128, 128) float32
    lower_mask_diag: nl.ndarray,  # (128, 128) float32
):
    """Fused chunked DeltaNet forward for one or more heads with SPMD sharding."""
    num_heads = query.shape[0]
    seq_len = query.shape[1]
    dim = query.shape[2]
    num_chunks = seq_len // CHUNK_SIZE
    head_idx = nl.program_id(axis=0)

    output = nl.ndarray(
        (num_heads, seq_len, dim), dtype=query.dtype, buffer=nl.shared_hbm
    )
    final_state_out = nl.ndarray(
        (num_heads, P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )

    Lmask = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask, src=lower_mask)

    UMask_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=UMask_psum, data=Lmask)
    UMask = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=UMask, src=UMask_psum)

    Imat = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Imat, src=identity)

    ones_1xC = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_1xC, value=1.0)

    zero_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero_11, value=0.0)

    state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=state, src=initial_state[head_idx, 0:P_MAX, 0:dim])

    for i_chunk in nl.sequential_range(num_chunks):
        chunk_start = i_chunk * CHUNK_SIZE

        q_c = nl.ndarray((P_MAX, dim), dtype=query.dtype, buffer=nl.sbuf)
        if CHUNK_SIZE == P_MAX:
            nisa.dma_copy(
                dst=q_c,
                src=query[head_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
            )
        else:
            nisa.memset(dst=q_c, value=0.0)
            nisa.dma_copy(
                dst=q_c[0:CHUNK_SIZE, 0:dim],
                src=query[head_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
            )

        k_c = nl.ndarray((P_MAX, dim), dtype=key.dtype, buffer=nl.sbuf)
        if CHUNK_SIZE == P_MAX:
            nisa.dma_copy(
                dst=k_c,
                src=key[head_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
            )
        else:
            nisa.memset(dst=k_c, value=0.0)
            nisa.dma_copy(
                dst=k_c[0:CHUNK_SIZE, 0:dim],
                src=key[head_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
            )

        q_square = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=q_square, data1=q_c, data2=q_c, op=nl.multiply)
        q_norm_sq = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=q_norm_sq, data=q_square, op=nl.add, axis=1)
        q_norm_sq_clamped = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=q_norm_sq_clamped,
            data=q_norm_sq,
            op0=nl.maximum,
            operand0=L2_EPS_SQUARED,
            engine=nisa.vector_engine,
        )
        q_inv_norm = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=q_inv_norm,
            data=q_norm_sq_clamped,
            op0=nl.rsqrt,
            operand0=0.0,
            engine=nisa.gpsimd_engine,
        )
        q_norm = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=q_norm,
            data=q_c,
            op0=nl.multiply,
            operand0=q_inv_norm,
            op1=nl.multiply,
            operand1=QUERY_SCALE,
            engine=nisa.vector_engine,
        )

        k_square = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=k_square, data1=k_c, data2=k_c, op=nl.multiply)
        k_norm_sq = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=k_norm_sq, data=k_square, op=nl.add, axis=1)
        k_norm_sq_clamped = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_norm_sq_clamped,
            data=k_norm_sq,
            op0=nl.maximum,
            operand0=L2_EPS_SQUARED,
            engine=nisa.vector_engine,
        )
        k_inv_norm = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_inv_norm,
            data=k_norm_sq_clamped,
            op0=nl.rsqrt,
            operand0=0.0,
            engine=nisa.gpsimd_engine,
        )
        k_norm = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_norm,
            data=k_c,
            op0=nl.multiply,
            operand0=k_inv_norm,
            engine=nisa.vector_engine,
        )

        v_c = nl.ndarray((P_MAX, dim), dtype=value.dtype, buffer=nl.sbuf)
        if CHUNK_SIZE == P_MAX:
            nisa.dma_copy(
                dst=v_c,
                src=value[head_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
            )
        else:
            nisa.memset(dst=v_c, value=0.0)
            nisa.dma_copy(
                dst=v_c[0:CHUNK_SIZE, 0:dim],
                src=value[head_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
            )

        g_chunk_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        if CHUNK_SIZE != P_MAX:
            nisa.memset(dst=g_chunk_p, value=0.0)
        nisa.dma_copy(
            dst=g_chunk_p[0:CHUNK_SIZE, 0:1],
            src=g_in[head_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:1],
        )

        beta_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        if CHUNK_SIZE != P_MAX:
            nisa.memset(dst=beta_p, value=0.0)
        nisa.dma_copy(
            dst=beta_p[0:CHUNK_SIZE, 0:1],
            src=beta_in[head_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:1],
        )

        g_tp_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=g_tp_psum, data=g_chunk_p)

        g_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=g_row[0:1, 0:CHUNK_SIZE],
            src=g_tp_psum[0:1, 0:CHUNK_SIZE],
        )

        gc_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor_scan(
            dst=gc_row[0:1, 0:CHUNK_SIZE],
            data0=ones_1xC[0:1, 0:CHUNK_SIZE],
            data1=g_row[0:1, 0:CHUNK_SIZE],
            initial=zero_11[0:1, 0:1],
            op0=nl.multiply,
            op1=nl.add,
        )

        gc_tp_psum = nl.ndarray((CHUNK_SIZE, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=gc_tp_psum, data=gc_row)

        gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        if CHUNK_SIZE != P_MAX:
            nisa.memset(dst=gc_p, value=0.0)
        nisa.tensor_copy(
            dst=gc_p[0:CHUNK_SIZE, 0:1],
            src=gc_tp_psum[0:CHUNK_SIZE, 0:1],
        )

        gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=gl_11[0:1, 0:1],
            src=gc_row[0:1, CHUNK_SIZE - 1 : CHUNK_SIZE],
        )

        exp_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=exp_gc_p[0:P_MAX, 0:1],
            op=nl.exp,
            data=gc_p[0:P_MAX, 0:1],
            bias=None,
            scale=1.0,
        )

        exp_gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=exp_gl_11,
            op=nl.exp,
            data=gl_11,
            bias=None,
            scale=1.0,
        )

        gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=gl_11[0:1, 0:1],
                dst=gl_p[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                shuffle_mask=_BROADCAST_MASK,
            )

        exp_gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=exp_gl_11[0:1, 0:1],
                dst=exp_gl_p[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                shuffle_mask=_BROADCAST_MASK,
            )

        gc_row_broadcast = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        if CHUNK_SIZE != P_MAX:
            nisa.memset(dst=gc_row_broadcast, value=0.0)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=gc_row[0:1, 0:CHUNK_SIZE],
                dst=gc_row_broadcast[
                    i_shuf * 32 : i_shuf * 32 + 32, 0:CHUNK_SIZE
                ],
                shuffle_mask=_BROADCAST_MASK,
            )

        gc_col_strict_t = nl.ndarray(
            (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=gc_col_strict_t,
            data1=gc_row_broadcast,
            data2=UMask,
            op=nl.multiply,
        )
        gc_row_strict_t = nl.ndarray(
            (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=gc_row_strict_t,
            data=UMask,
            op0=nl.multiply,
            operand0=gc_p,
            engine=nisa.vector_engine,
        )
        g_diff_strict_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=g_diff_strict_t,
            data1=gc_col_strict_t,
            data2=gc_row_strict_t,
            op=nl.subtract,
        )
        decay_strict_t_raw = nl.ndarray(
            (P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.activation(
            dst=decay_strict_t_raw,
            op=nl.exp,
            data=g_diff_strict_t,
            bias=None,
            scale=1.0,
        )
        decay_strict_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=decay_strict_t,
            data1=decay_strict_t_raw,
            data2=UMask,
            op=nl.multiply,
        )

        decay_diag_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=decay_diag_t, data1=decay_strict_t, data2=Imat, op=nl.add
        )

        k_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_beta,
            data=k_norm,
            op0=nl.multiply,
            operand0=beta_p,
            engine=nisa.vector_engine,
        )

        v_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=v_beta,
            data=v_c,
            op0=nl.multiply,
            operand0=beta_p,
            engine=nisa.vector_engine,
        )

        kb_T_psum = nl.ndarray((P_MAX, CHUNK_SIZE), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=kb_T_psum, data=k_beta[0:CHUNK_SIZE, 0:dim])
        kb_T = nl.ndarray((P_MAX, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kb_T, src=kb_T_psum)

        k_T_psum = nl.ndarray((P_MAX, CHUNK_SIZE), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=k_T_psum, data=k_norm[0:CHUNK_SIZE, 0:dim])
        k_T = nl.ndarray((P_MAX, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=k_T, src=k_T_psum)

        QK_T_psum = nl.ndarray(
            (CHUNK_SIZE, CHUNK_SIZE), dtype=nl.float32, buffer=nl.psum
        )
        nisa.nc_matmul(dst=QK_T_psum, stationary=k_T, moving=kb_T)
        QK_T = nl.ndarray((CHUNK_SIZE, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=QK_T, src=QK_T_psum)

        QK_decay_t = nl.ndarray(
            (CHUNK_SIZE, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=QK_decay_t,
            data1=QK_T,
            data2=decay_strict_t[0:CHUNK_SIZE, 0:CHUNK_SIZE],
            op=nl.multiply,
        )

        A_T = nl.ndarray((CHUNK_SIZE, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=A_T,
            data=QK_decay_t,
            op0=nl.multiply,
            operand0=-1.0,
            engine=nisa.vector_engine,
        )
        kb_exp_gc = nl.ndarray((CHUNK_SIZE, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=kb_exp_gc,
            data=k_beta[0:CHUNK_SIZE, 0:dim],
            op0=nl.multiply,
            operand0=exp_gc_p[0:CHUNK_SIZE, 0:1],
            engine=nisa.vector_engine,
        )

        kbe_T_psum = nl.ndarray((P_MAX, CHUNK_SIZE), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=kbe_T_psum, data=kb_exp_gc)
        kbe_T = nl.ndarray((P_MAX, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kbe_T, src=kbe_T_psum)

        kbe_state_psum = nl.ndarray((CHUNK_SIZE, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=kbe_state_psum, stationary=kbe_T, moving=state)
        kbe_state = nl.ndarray((CHUNK_SIZE, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kbe_state, src=kbe_state_psum)

        solve_rhs = nl.ndarray((CHUNK_SIZE, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=solve_rhs,
            data1=v_beta[0:CHUNK_SIZE, 0:dim],
            data2=kbe_state,
            op=nl.subtract,
        )

        v_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=v_new, value=0.0)

        if SOLVE_KKT_HIER:
            _hierarchical_kkt_solve128(v_new, A_T, Imat, solve_rhs, dim)
        else:
            _blocked_doubling_solve(v_new, A_T, solve_rhs, dim)

        q_T_psum = nl.ndarray((P_MAX, CHUNK_SIZE), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=q_T_psum, data=q_norm[0:CHUNK_SIZE, 0:dim])
        q_T = nl.ndarray((P_MAX, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q_T, src=q_T_psum)

        qk_T_psum = nl.ndarray(
            (CHUNK_SIZE, CHUNK_SIZE), dtype=nl.float32, buffer=nl.psum
        )
        nisa.nc_matmul(dst=qk_T_psum, stationary=k_T, moving=q_T)
        qk_raw_t = nl.ndarray(
            (CHUNK_SIZE, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_copy(dst=qk_raw_t, src=qk_T_psum)

        ai_T = nl.ndarray((CHUNK_SIZE, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=ai_T,
            data1=qk_raw_t,
            data2=decay_diag_t[0:CHUNK_SIZE, 0:CHUNK_SIZE],
            op=nl.multiply,
        )

        q_exp = nl.ndarray((CHUNK_SIZE, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=q_exp,
            data=q_norm[0:CHUNK_SIZE, 0:dim],
            op0=nl.multiply,
            operand0=exp_gc_p[0:CHUNK_SIZE, 0:1],
            engine=nisa.vector_engine,
        )

        qe_T_psum = nl.ndarray((P_MAX, CHUNK_SIZE), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=qe_T_psum, data=q_exp)
        qe_T = nl.ndarray((P_MAX, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qe_T, src=qe_T_psum)

        ai_psum = nl.ndarray((CHUNK_SIZE, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ai_psum, stationary=qe_T, moving=state)
        attn_inter = nl.ndarray((CHUNK_SIZE, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=attn_inter, src=ai_psum)

        intra_psum = nl.ndarray((CHUNK_SIZE, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=intra_psum,
            stationary=ai_T,
            moving=v_new[0:CHUNK_SIZE, 0:dim],
        )
        intra_out = nl.ndarray((CHUNK_SIZE, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=intra_out, src=intra_psum)

        chunk_out = nl.ndarray((CHUNK_SIZE, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=chunk_out, data1=attn_inter, data2=intra_out, op=nl.add)

        nisa.dma_copy(
            dst=output[head_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
            src=chunk_out,
        )

        exp_gl_minus_gc_p = nl.ndarray(
            (CHUNK_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.activation(
            dst=exp_gl_minus_gc_p[0:CHUNK_SIZE, 0:1],
            op=nl.exp,
            data=gc_p[0:CHUNK_SIZE, 0:1],
            bias=gl_p[0:CHUNK_SIZE, 0:1],
            scale=-1.0,
        )

        k_raw_decay = nl.ndarray((CHUNK_SIZE, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_raw_decay,
            data=k_norm[0:CHUNK_SIZE, 0:dim],
            op0=nl.multiply,
            operand0=exp_gl_minus_gc_p,
            engine=nisa.vector_engine,
        )

        kv_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=kv_psum,
            stationary=k_raw_decay,
            moving=v_new[0:CHUNK_SIZE, 0:dim],
        )
        kv_outer = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kv_outer, src=kv_psum)

        state_decayed = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=state_decayed,
            data=state,
            op0=nl.multiply,
            operand0=exp_gl_p,
            engine=nisa.vector_engine,
        )
        nisa.tensor_tensor(dst=state, data1=state_decayed, data2=kv_outer, op=nl.add)

    nisa.dma_copy(dst=final_state_out[head_idx, 0:P_MAX, 0:dim], src=state)

    return output, final_state_out
