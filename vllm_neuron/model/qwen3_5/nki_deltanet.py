# SPDX-License-Identifier: Apache-2.0
#
# Vendored verbatim from the NxDI Qwen3.5 reference port, Apache-2.0:
#   https://github.com/qingzwang/neuronx-distributed-inference
#   branch qwen3.5-2b-hybrid-deltanet
#   contrib/models/Qwen3.5-2B/src/nki_kernels/nki_deltanet_fused_legacy.py
#
# Vendored rather than imported because that repository is not a dependency of
# this plugin. The body below is unmodified so it can be diffed against upstream;
# everything Qwen3.5-on-vLLM-specific lives in ``deltanet.py``, which calls this.
#
# This is the *legacy single-head* kernel, chosen deliberately over the newer
# fused multihead one. The reference documents that the multihead variant is
# numerically unstable on real vision embeddings — it emits degenerate repeated
# tokens — and forces QWEN36_DELTANET_CTE_IMPL=legacy_direct plus
# QWEN36_DELTANET_MULTIHEAD_CTE=0 for its own VL path. This port has working VL,
# so it starts from the variant that is known to survive vision inputs. It costs
# one kernel launch per (batch, head) instead of one per head group.
#
# "legacy_direct" also means the caller does the l2-norm and the 1/sqrt(dk) query
# scale, and passes RAW per-token log-decay (the cumsum happens in-kernel).

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
    attn_intra = (q @ k^T) * decay_mask * lower_mask_diag
    output = attn_inter + attn_intra @ v_new
    state = exp(g_last) * (state + k_raw_decay^T @ v_new)
"""

import numpy as np

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128  # Partition dim = chunk_size = k_dim = v_dim
CHUNK_SIZE = 128

# Broadcast partition 0 to all partitions in a 32-wide group
_BROADCAST_MASK = [0] * 32


def _make_lower_mask():
    """Strict lower triangular (128x128) as numpy constant."""
    return np.tril(np.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32), k=-1)


def _make_lower_mask_diag():
    """Lower triangular with diagonal (128x128) as numpy constant."""
    return np.tril(np.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32), k=0)


def _make_identity():
    """Identity matrix (128x128) as numpy constant."""
    return np.eye(CHUNK_SIZE, dtype=np.float32)


@nki.jit
def deltanet_fused_chunked_fwd(
    query: nl.ndarray,  # (S, 128) float32 — l2-normed and scaled
    key: nl.ndarray,  # (S, 128) float32 — l2-normed
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
      - query must be l2-normed and scaled by 1/sqrt(k_dim)
      - key must be l2-normed
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
    eye = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=eye, src=identity)

    Lmask = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask, src=lower_mask)

    Lmask_d = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask_d, src=lower_mask_diag)

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
        nisa.dma_copy(
            dst=q_c,
            src=query[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
        )

        k_c = nl.ndarray((P_MAX, dim), dtype=key.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=k_c,
            src=key[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
        )

        v_c = nl.ndarray((P_MAX, dim), dtype=value.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=v_c,
            src=value[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
        )

        # g: (CHUNK_SIZE, 1) — raw log-decay per token
        g_chunk_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=g_chunk_p[0:CHUNK_SIZE, 0:1],
            src=g_in[chunk_start : chunk_start + CHUNK_SIZE, 0:1],
        )

        # beta: (CHUNK_SIZE, 1) — write gate scalar per token
        beta_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=beta_p[0:CHUNK_SIZE, 0:1],
            src=beta_in[chunk_start : chunk_start + CHUNK_SIZE, 0:1],
        )

        # ---- In-kernel cumsum of g via tensor_tensor_scan ----
        # Need g as (1, CHUNK_SIZE) for scan along free dim.
        # Transpose: (CHUNK_SIZE, 1) -> (1, CHUNK_SIZE) via nc_transpose
        g_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=g_padded, value=0.0)
        nisa.tensor_copy(
            dst=g_padded[0:CHUNK_SIZE, 0:1],
            src=g_chunk_p[0:CHUNK_SIZE, 0:1],
        )

        g_tp_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=g_tp_psum, data=g_padded)

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

        # Transpose gc back to (CHUNK_SIZE, 1) partition layout
        gc_padded = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=gc_padded, value=0.0)
        nisa.tensor_copy(
            dst=gc_padded[0:1, 0:CHUNK_SIZE],
            src=gc_row[0:1, 0:CHUNK_SIZE],
        )

        gc_tp_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=gc_tp_psum, data=gc_padded)

        # gc_p: (P_MAX, 1) — cumulative sum of g per token in this chunk
        gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
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

        # g_last: scalar, then broadcast to (P_MAX, 1) for direct
        # exp(g_last - gc) in the state update.
        gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=gl_11[0:1, 0:1],
                dst=gl_p[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                shuffle_mask=_BROADCAST_MASK,
            )

        # exp(g_last): scalar, then broadcast to (P_MAX, 1)
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
            data=Lmask_d,
            op0=nl.multiply,
            operand0=gc_p,
            engine=nisa.vector_engine,
        )
        gc_row_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=gc_row_diag, data1=gc_row_broadcast, data2=Lmask_d, op=nl.multiply
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
            dst=decay_diag, data1=decay_diag_raw, data2=Lmask_d, op=nl.multiply
        )

        # ============================================================
        # k_beta = K * beta, v_beta = V * beta
        # tensor_scalar broadcasts beta_p (P_MAX, 1) across free dim
        # ============================================================
        k_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_beta,
            data=k_c,
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
        nisa.nc_transpose(dst=k_T_psum, data=k_c)
        k_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=k_T, src=k_T_psum)

        # QK = k_beta^T @ k  (contract over features)
        QK_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=QK_psum, stationary=kb_T, moving=k_T)
        QK = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=QK, src=QK_psum)

        # QK_decay[i,j] = QK[i,j] * exp(gc[i] - gc[j]) for i > j.
        QK_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=QK_decay, data1=QK, data2=decay_strict, op=nl.multiply)

        # A = -QK_decay * lower_mask
        neg_QK_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=neg_QK_decay,
            data=QK_decay,
            op0=nl.multiply,
            operand0=-1.0,
            engine=nisa.vector_engine,
        )
        A_mat = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=A_mat, data1=neg_QK_decay, data2=Lmask, op=nl.multiply)

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
        # Direct forward substitution for:
        #   v_new = solve((I - A_mat), solve_rhs)
        #
        # A_mat is strictly lower triangular, so row i only depends on rows
        # < i. The full-matmul plus row-select form keeps the shape static
        # and compiler-safe while updating exactly one solved row per step.
        # ============================================================
        v_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=v_new, value=0.0)

        A_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=A_T_psum, data=A_mat)
        A_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=A_T, src=A_T_psum)

        for solve_i in nl.static_range(P_MAX):
            row_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=row_psum, stationary=A_T, moving=v_new)
            row_prod = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=row_prod, src=row_psum)

            row_with_rhs = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=row_with_rhs,
                data1=row_prod,
                data2=solve_rhs,
                op=nl.add,
            )

            row_mask = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=row_mask[0:P_MAX, 0:1],
                src=eye[0:P_MAX, solve_i : solve_i + 1],
            )

            row_update = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=row_update,
                data=row_with_rhs,
                op0=nl.multiply,
                operand0=row_mask,
                engine=nisa.vector_engine,
            )

            v_next = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=v_next, data1=v_new, data2=row_update, op=nl.add)
            nisa.tensor_copy(dst=v_new, src=v_next)

        # ============================================================
        # Phase 2: Inter-chunk state propagation
        # attn_intra = (q @ k^T) * decay_mask * lower_mask_diag
        # ============================================================
        q_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=q_T_psum, data=q_c)
        q_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q_T, src=q_T_psum)

        qk_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=qk_psum, stationary=q_T, moving=k_T)
        qk_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qk_raw, src=qk_psum)

        # qk_decay[i,j] = (q @ k^T)[i,j] * exp(gc[i] - gc[j]) for i >= j.
        qk_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=qk_decay, data1=qk_raw, data2=decay_diag, op=nl.multiply)

        attn_intra = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=attn_intra, data1=qk_decay, data2=Lmask_d, op=nl.multiply
        )

        # ============================================================
        # attn_inter = (q * exp(gc)) @ state   (state is in SBUF!)
        # ============================================================
        q_exp = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=q_exp,
            data=q_c,
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
        ai_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=ai_T_psum, data=attn_intra)
        ai_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ai_T, src=ai_T_psum)

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
        nisa.dma_copy(
            dst=output[chunk_start : chunk_start + CHUNK_SIZE, 0:dim],
            src=chunk_out,
        )

        # ============================================================
        # State update: state = exp(g_last) * (state + k_raw_decay^T @ v_new)
        # state is updated IN-PLACE in SBUF — no HBM round-trip!
        # ============================================================

        # k_raw_decay contributes as exp(g_last) * (k * exp(-gc))^T @ v_new.
        # Compute the equivalent stable form k * exp(g_last - gc) directly so
        # no exp(-gc) intermediate can overflow.
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

        # k_raw_decay = k * exp(g_last - gc)
        k_raw_decay = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=k_raw_decay,
            data=k_c,
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
        # tensor_scalar broadcasts exp_gl_p (P_MAX, 1) across free dim.
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
