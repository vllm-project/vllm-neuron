"""Fused chunked DeltaNet prefill kernel, version 2.

Version 2 computes the same recurrence as version 1
(`fused.deltanet_fused_chunked_fwd_multihead`). It changes the launch and the
data flow:

  1. One launch for each layer. Each program of the grid owns whole k-head
     groups and loops over its heads inside the kernel. Version 1 needs one
     launch for each pair of heads.
  2. GQA by index. q and k have one row for each k-head. v, g, beta and the
     state have one row for each v-head. The v-head h reads the k-head h // R.
     The caller does not materialize the repeated q and k.
  3. The kernel normalizes q and k once for each k-head. The R v-heads of the
     group share the result.
  4. Two formulations:
       "v1math"  The chunk body of version 1. The output is bit-identical to
                 version 1.
       "fused"   Fewer transposes, PSUM copies and Vector operations. The
                 output agrees with version 1 to the rounding error.
  5. Opt-in matmul precisions for "fused": `gemm_dtype` and `inverse_dtype`.
     NKI 0.7 has no TF32 matmul. The MLIR verifier rejects a tfloat32
     operand ("'stationary' must be a float type, got '!nisa.float32r'").

Every NKI range is fully unrolled, so the kernel is one flat instruction
stream. The order of the instructions controls how much the engines overlap.
For this reason the kernel emits the chunk work for all heads together.

Precondition for "fused": g <= 0. The model sets g = -exp(A_log) * softplus()
and sets g = 0 at padding, so the condition holds. The "fused" decay clamps
the exponent at 0 before the exp. A positive g gives a wrong decay.

The persistent NKI cache key hashes the source of this kernel function only,
not the source of the helper functions. Increase `revision` when you change a
helper function.
"""

import numpy as np

import nki
import nki.isa as nisa
import nki.language as nl

from vllm_neuron.nki.deltanet.fused import (
    _apply_schur_inverse_t,
    _load_schur_masks,
    _schur_inverse_t,
)

P_MAX = 128
CHUNK = 128
L2_EPS_SQUARED = 1.0e-12
QUERY_SCALE = P_MAX**-0.5
_BROADCAST_MASK = [0] * 32
_SCALAR_ENGINE = getattr(nisa, "scalar_engine", None)

FORMULATIONS = ("v1math", "fused")
GEMM_DTYPES = ("fp32", "bf16")
INVERSE_DTYPES = ("fp32", "bf16")


def make_v2_constants() -> dict:
    """Return the constant mask tiles of the v2 kernel as numpy arrays.

    The model builds the same tiles with torch ops on the device (see
    `_nki_chunked_forward_v2` in the Qwen3-Next model). Tests use this function.

    Example:
        c = make_v2_constants()
        c["lower_mask"].shape   # (128, 128)
    """
    rows = np.arange(P_MAX)[:, None]
    cols = np.arange(P_MAX)[None, :]
    level_masks = np.zeros((7 * P_MAX, P_MAX), dtype=np.float32)
    for plane, b in enumerate([1, 2, 4, 8, 16, 32, 64]):
        sel = ((rows // b) % 2 == 1) & ((cols // b) == (rows // b) - 1)
        level_masks[plane * P_MAX : (plane + 1) * P_MAX, :] = sel
    return {
        "lower_mask": (cols < rows).astype(np.float32),
        "identity": np.eye(P_MAX, dtype=np.float32),
        "upper_diag_mask": (cols >= rows).astype(np.float32),
        "level_masks": level_masks,
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _l2_normalize_qk(q_c, k_c, dim):
    """Normalize one chunk of q and k with the op sequence of version 1.

    The clamp and the rsqrt run once on a (128, 2) tile for q and k together.
    Each element gets the same operations as in version 1, so the result is
    bit-identical.
    """
    norm_sq = nl.ndarray((P_MAX, 2), dtype=nl.float32, buffer=nl.sbuf)

    q_square = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=q_square, data1=q_c, data2=q_c, op=nl.multiply)
    nisa.tensor_reduce(dst=norm_sq[0:P_MAX, 0:1], data=q_square, op=nl.add, axis=1)

    k_square = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=k_square, data1=k_c, data2=k_c, op=nl.multiply)
    nisa.tensor_reduce(dst=norm_sq[0:P_MAX, 1:2], data=k_square, op=nl.add, axis=1)

    clamped = nl.ndarray((P_MAX, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=clamped,
        data=norm_sq,
        op0=nl.maximum,
        operand0=L2_EPS_SQUARED,
        engine=nisa.vector_engine,
    )
    inv_norm = nl.ndarray((P_MAX, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=inv_norm,
        data=clamped,
        op0=nl.rsqrt,
        operand0=0.0,
        engine=nisa.gpsimd_engine,
    )

    q_norm = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=q_norm,
        data=q_c,
        op0=nl.multiply,
        operand0=inv_norm[0:P_MAX, 0:1],
        op1=nl.multiply,
        operand1=QUERY_SCALE,
        engine=nisa.vector_engine,
    )
    k_norm = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=k_norm,
        data=k_c,
        op0=nl.multiply,
        operand0=inv_norm[0:P_MAX, 1:2],
        engine=nisa.vector_engine,
    )
    return q_norm, k_norm


def _transpose_to_sbuf(src, dtype):
    """Transpose a (128, 128) tile on the Tensor Engine and copy it to SBUF."""
    t_psum = nl.ndarray((P_MAX, P_MAX), dtype=src.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=t_psum, data=src)
    t_sb = nl.ndarray((P_MAX, P_MAX), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=t_sb, src=t_psum)
    return t_sb


def _broadcast_scalar(src_11, dst_p1):
    """Copy a (1, 1) value to every partition of a (128, 1) tile."""
    for i_shuf in nl.static_range(P_MAX // 32):
        nisa.nc_stream_shuffle(
            src=src_11[0:1, 0:1],
            dst=dst_p1[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
            shuffle_mask=_BROADCAST_MASK,
        )


def _load_chunk(src, row, chunk_start, dim):
    """Load one (128, dim) chunk of a (rows, S, dim) HBM tensor to SBUF."""
    tile = nl.ndarray((P_MAX, dim), dtype=src.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=src[row, chunk_start : chunk_start + CHUNK, 0:dim])
    return tile


# ---------------------------------------------------------------------------
# Formulation "v1math": the chunk body of version 1
# ---------------------------------------------------------------------------


def _v1math_head_chunk(
    state,
    q_norm,
    k_norm,
    k_T,
    q_T,
    qk_raw_t,
    value,
    g_in,
    beta_in,
    output,
    vh,
    chunk_start,
    dim,
    UMask,
    Imat,
    schur_masks,
    ones_1xC,
    zero_11,
):
    """Run the version 1 chunk body for one v-head and update `state` in place.

    The operations and their operands are the same as in
    `deltanet_fused_chunked_fwd_multihead`. The k transpose, the q transpose and
    the raw q k^T come from the caller, which computes them once for each
    k-head. The values are the same, so the result is bit-identical.
    """
    v_c = _load_chunk(value, vh, chunk_start, dim)

    g_chunk_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=g_chunk_p[0:CHUNK, 0:1],
        src=g_in[vh, chunk_start : chunk_start + CHUNK, 0:1],
    )
    beta_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=beta_p[0:CHUNK, 0:1],
        src=beta_in[vh, chunk_start : chunk_start + CHUNK, 0:1],
    )

    g_tp_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=g_tp_psum, data=g_chunk_p)
    g_row = nl.ndarray((1, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=g_row[0:1, 0:CHUNK], src=g_tp_psum[0:1, 0:CHUNK])

    gc_row = nl.ndarray((1, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor_scan(
        dst=gc_row[0:1, 0:CHUNK],
        data0=ones_1xC[0:1, 0:CHUNK],
        data1=g_row[0:1, 0:CHUNK],
        initial=zero_11[0:1, 0:1],
        op0=nl.multiply,
        op1=nl.add,
    )
    gc_tp_psum = nl.ndarray((CHUNK, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=gc_tp_psum, data=gc_row)
    gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gc_p[0:CHUNK, 0:1], src=gc_tp_psum[0:CHUNK, 0:1])

    gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gl_11[0:1, 0:1], src=gc_row[0:1, CHUNK - 1 : CHUNK])

    exp_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=exp_gc_p[0:P_MAX, 0:1],
        op=nl.exp,
        data=gc_p[0:P_MAX, 0:1],
        bias=None,
        scale=1.0,
    )
    exp_gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=exp_gl_11, op=nl.exp, data=gl_11, bias=None, scale=1.0)

    gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    _broadcast_scalar(gl_11, gl_p)
    exp_gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    _broadcast_scalar(exp_gl_11, exp_gl_p)

    gc_row_broadcast = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    for i_shuf in nl.static_range(P_MAX // 32):
        nisa.nc_stream_shuffle(
            src=gc_row[0:1, 0:CHUNK],
            dst=gc_row_broadcast[i_shuf * 32 : i_shuf * 32 + 32, 0:CHUNK],
            shuffle_mask=_BROADCAST_MASK,
        )

    gc_col_strict_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=gc_col_strict_t, data1=gc_row_broadcast, data2=UMask, op=nl.multiply
    )
    gc_row_strict_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
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
    decay_strict_t_raw = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=decay_strict_t_raw, op=nl.exp, data=g_diff_strict_t, bias=None, scale=1.0
    )
    decay_strict_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=decay_strict_t, data1=decay_strict_t_raw, data2=UMask, op=nl.multiply
    )
    decay_diag_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=decay_diag_t, data1=decay_strict_t, data2=Imat, op=nl.add)

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

    kb_T = _transpose_to_sbuf(k_beta[0:CHUNK, 0:dim], nl.float32)

    QK_T_psum = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=QK_T_psum, stationary=k_T, moving=kb_T)
    QK_T = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=QK_T, src=QK_T_psum)
    QK_decay_t = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=QK_decay_t,
        data1=QK_T,
        data2=decay_strict_t[0:CHUNK, 0:CHUNK],
        op=nl.multiply,
    )
    A_T = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=A_T,
        data=QK_decay_t,
        op0=nl.multiply,
        operand0=-1.0,
        engine=nisa.vector_engine,
    )

    kb_exp_gc = nl.ndarray((CHUNK, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=kb_exp_gc,
        data=k_beta[0:CHUNK, 0:dim],
        op0=nl.multiply,
        operand0=exp_gc_p[0:CHUNK, 0:1],
        engine=nisa.vector_engine,
    )
    kbe_T = _transpose_to_sbuf(kb_exp_gc, nl.float32)
    kbe_state_psum = nl.ndarray((CHUNK, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=kbe_state_psum, stationary=kbe_T, moving=state)
    kbe_state = nl.ndarray((CHUNK, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kbe_state, src=kbe_state_psum)
    solve_rhs = nl.ndarray((CHUNK, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=solve_rhs, data1=v_beta[0:CHUNK, 0:dim], data2=kbe_state, op=nl.subtract
    )

    # The apply writes every row of v_new, so the version 1 memset is not needed.
    v_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    inv_t = _schur_inverse_t(A_T, Imat, schur_masks)
    _apply_schur_inverse_t(v_new, inv_t, solve_rhs, dim)

    ai_T = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=ai_T, data1=qk_raw_t, data2=decay_diag_t[0:CHUNK, 0:CHUNK], op=nl.multiply
    )

    q_exp = nl.ndarray((CHUNK, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=q_exp,
        data=q_norm[0:CHUNK, 0:dim],
        op0=nl.multiply,
        operand0=exp_gc_p[0:CHUNK, 0:1],
        engine=nisa.vector_engine,
    )
    qe_T = _transpose_to_sbuf(q_exp, nl.float32)

    ai_psum = nl.ndarray((CHUNK, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ai_psum, stationary=qe_T, moving=state)
    attn_inter = nl.ndarray((CHUNK, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=attn_inter, src=ai_psum)

    intra_psum = nl.ndarray((CHUNK, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=intra_psum, stationary=ai_T, moving=v_new[0:CHUNK, 0:dim])
    intra_out = nl.ndarray((CHUNK, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=intra_out, src=intra_psum)

    chunk_out = nl.ndarray((CHUNK, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=chunk_out, data1=attn_inter, data2=intra_out, op=nl.add)
    nisa.dma_copy(
        dst=output[vh, chunk_start : chunk_start + CHUNK, 0:dim], src=chunk_out
    )

    exp_gl_minus_gc_p = nl.ndarray((CHUNK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=exp_gl_minus_gc_p[0:CHUNK, 0:1],
        op=nl.exp,
        data=gc_p[0:CHUNK, 0:1],
        bias=gl_p[0:CHUNK, 0:1],
        scale=-1.0,
    )
    k_raw_decay = nl.ndarray((CHUNK, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=k_raw_decay,
        data=k_norm[0:CHUNK, 0:dim],
        op0=nl.multiply,
        operand0=exp_gl_minus_gc_p,
        engine=nisa.vector_engine,
    )
    kv_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=kv_psum, stationary=k_raw_decay, moving=v_new[0:CHUNK, 0:dim])
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


def _run_v1math(
    query,
    key,
    value,
    g_in,
    beta_in,
    initial_state,
    lower_mask,
    identity,
    level_masks,
    output,
    final_state_out,
    kh0,
    kh_count,
    vh0,
    group,
    num_chunks,
    dim,
):
    Lmask = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask, src=lower_mask)
    UMask = _transpose_to_sbuf(Lmask, nl.float32)
    Imat = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Imat, src=identity)
    schur_masks = _load_schur_masks(level_masks)
    ones_1xC = nl.ndarray((1, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_1xC, value=1.0)
    zero_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero_11, value=0.0)

    states = []
    for jv in nl.static_range(kh_count * group):
        state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=state, src=initial_state[vh0 + jv, 0:P_MAX, 0:dim])
        states.append(state)

    for i_chunk in nl.sequential_range(num_chunks):
        chunk_start = i_chunk * CHUNK
        for jk in nl.static_range(kh_count):
            kh = kh0 + jk
            q_c = _load_chunk(query, kh, chunk_start, dim)
            k_c = _load_chunk(key, kh, chunk_start, dim)
            q_norm, k_norm = _l2_normalize_qk(q_c, k_c, dim)
            k_T = _transpose_to_sbuf(k_norm[0:CHUNK, 0:dim], nl.float32)
            q_T = _transpose_to_sbuf(q_norm[0:CHUNK, 0:dim], nl.float32)
            qk_T_psum = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=qk_T_psum, stationary=k_T, moving=q_T)
            qk_raw_t = nl.ndarray((CHUNK, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=qk_raw_t, src=qk_T_psum)
            for r in nl.static_range(group):
                jv = jk * group + r
                _v1math_head_chunk(
                    states[jv],
                    q_norm,
                    k_norm,
                    k_T,
                    q_T,
                    qk_raw_t,
                    value,
                    g_in,
                    beta_in,
                    output,
                    vh0 + jv,
                    chunk_start,
                    dim,
                    UMask,
                    Imat,
                    schur_masks,
                    ones_1xC,
                    zero_11,
                )

    for jv in nl.static_range(kh_count * group):
        nisa.dma_copy(dst=final_state_out[vh0 + jv, 0:P_MAX, 0:dim], src=states[jv])


# ---------------------------------------------------------------------------
# Formulation "fused"
# ---------------------------------------------------------------------------


class _Evac:
    """Alternate PSUM-to-SBUF copies between the Vector and the Scalar engine.

    Both engines can read PSUM. Alternating them spreads the copy work, which
    is the largest share of the Vector Engine time in this kernel.
    """

    def __init__(self):
        self.count = 0

    def __call__(self, dst, src):
        if _SCALAR_ENGINE is not None and self.count % 2 == 1:
            nisa.tensor_copy(dst=dst, src=src, engine=_SCALAR_ENGINE)
        else:
            nisa.tensor_copy(dst=dst, src=src, engine=nisa.vector_engine)
        self.count += 1


def _low_copy(src, kind, evac):
    """Return `src` in the matmul operand dtype `kind`. Copy only for bf16."""
    if kind == "bf16":
        dst = nl.ndarray(src.shape, dtype=nl.bfloat16, buffer=nl.sbuf)
        evac(dst, src)
        return dst
    return src


def _gate_tables(g_in, beta_in, vh, num_chunks, consts, evac):
    """Build the gate tables of one v-head for all chunks of the sequence.

    Column c of each (128, num_chunks) table holds the value of chunk c, with
    the token of the chunk on the partition axis. One scan computes the
    in-chunk cumulative sum of g for every chunk.

    Returns a list:
        [gc, exp(gc), exp(gl), exp(gl - gc), -beta, -beta * exp(gc), beta]
    where gl is the last cumulative value of the chunk, on every partition.
    """
    num_vh = g_in.shape[0]
    g_rows_src = g_in.reshape((num_vh, num_chunks, CHUNK))
    beta_rows_src = beta_in.reshape((num_vh, num_chunks, CHUNK))

    g_rows = nl.ndarray((num_chunks, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=g_rows, src=g_rows_src[vh, 0:num_chunks, 0:CHUNK])
    gc_rows = nl.ndarray((num_chunks, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor_scan(
        dst=gc_rows,
        data0=consts["ones_rows"][0:num_chunks, 0:CHUNK],
        data1=g_rows,
        initial=consts["zero_col"][0:num_chunks, 0:1],
        op0=nl.multiply,
        op1=nl.add,
    )

    gc_psum = nl.ndarray((CHUNK, num_chunks), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=gc_psum, data=gc_rows)
    gc = nl.ndarray((CHUNK, num_chunks), dtype=nl.float32, buffer=nl.sbuf)
    evac(gc, gc_psum)

    gl_psum = nl.ndarray((1, num_chunks), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=gl_psum, data=gc_rows[0:num_chunks, CHUNK - 1 : CHUNK])
    gl_row = nl.ndarray((1, num_chunks), dtype=nl.float32, buffer=nl.sbuf)
    evac(gl_row, gl_psum)
    gl_bc_psum = nl.ndarray((CHUNK, num_chunks), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=gl_bc_psum, stationary=consts["ones_row"], moving=gl_row)
    gl_bc = nl.ndarray((CHUNK, num_chunks), dtype=nl.float32, buffer=nl.sbuf)
    evac(gl_bc, gl_bc_psum)

    exp_gc = nl.ndarray((CHUNK, num_chunks), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=exp_gc, op=nl.exp, data=gc, bias=None, scale=1.0)
    exp_gl = nl.ndarray((CHUNK, num_chunks), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=exp_gl, op=nl.exp, data=gl_bc, bias=None, scale=1.0)
    gl_minus_gc = nl.ndarray((CHUNK, num_chunks), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=gl_minus_gc, data1=gl_bc, data2=gc, op=nl.subtract)
    exp_gl_minus_gc = nl.ndarray((CHUNK, num_chunks), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=exp_gl_minus_gc, op=nl.exp, data=gl_minus_gc, bias=None, scale=1.0
    )

    beta_rows = nl.ndarray((num_chunks, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=beta_rows, src=beta_rows_src[vh, 0:num_chunks, 0:CHUNK])
    beta_psum = nl.ndarray((CHUNK, num_chunks), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=beta_psum, data=beta_rows)
    beta = nl.ndarray((CHUNK, num_chunks), dtype=nl.float32, buffer=nl.sbuf)
    evac(beta, beta_psum)
    neg_beta = nl.ndarray((CHUNK, num_chunks), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=neg_beta,
        data=beta,
        op0=nl.multiply,
        operand0=-1.0,
        engine=nisa.vector_engine,
    )
    neg_beta_exp_gc = nl.ndarray((CHUNK, num_chunks), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=neg_beta_exp_gc, data1=neg_beta, data2=exp_gc, op=nl.multiply
    )

    return [gc, exp_gc, exp_gl, exp_gl_minus_gc, neg_beta, neg_beta_exp_gc, beta]


def _fused_khead_chunk(query, key, kh, chunk_start, dim, gemm_dtype, evac):
    """Normalize q and k of one k-head chunk and build the shared tiles.

    Returns [k_norm, k_T, q_T, kk, qk_t] where k_T and q_T have the dk axis on
    the partitions, kk[i, j] = k_i . k_j and qk_t[j, i] = k_j . q_i.
    """
    q_c = _load_chunk(query, kh, chunk_start, dim)
    k_c = _load_chunk(key, kh, chunk_start, dim)
    q_norm, k_norm = _l2_normalize_qk(q_c, k_c, dim)

    op_dtype = nl.bfloat16 if gemm_dtype == "bf16" else nl.float32
    if gemm_dtype == "bf16":
        k_src = nl.ndarray((P_MAX, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        evac(k_src, k_norm)
        q_src = nl.ndarray((P_MAX, dim), dtype=nl.bfloat16, buffer=nl.sbuf)
        evac(q_src, q_norm)
    else:
        k_src = k_norm
        q_src = q_norm

    # k^T and q^T side by side in one PSUM tile, copied out once.
    kq_t_psum = nl.ndarray((P_MAX, 2 * CHUNK), dtype=op_dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=kq_t_psum[0:P_MAX, 0:CHUNK], data=k_src)
    nisa.nc_transpose(dst=kq_t_psum[0:P_MAX, CHUNK : 2 * CHUNK], data=q_src)
    kq_t = nl.ndarray((P_MAX, 2 * CHUNK), dtype=op_dtype, buffer=nl.sbuf)
    evac(kq_t, kq_t_psum)
    k_T = kq_t[0:P_MAX, 0:CHUNK]
    q_T = kq_t[0:P_MAX, CHUNK : 2 * CHUNK]

    # [kk | qk_t] = k_T^T @ [k_T | q_T], one matmul of width 256.
    kk_qk_psum = nl.ndarray((CHUNK, 2 * CHUNK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=kk_qk_psum, stationary=k_T, moving=kq_t)
    kk_qk = nl.ndarray((CHUNK, 2 * CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    evac(kk_qk, kk_qk_psum)
    return [
        k_norm,
        k_T,
        q_T,
        kk_qk[0:CHUNK, 0:CHUNK],
        kk_qk[0:CHUNK, CHUNK : 2 * CHUNK],
    ]


def _fused_phase_a(
    query,
    key,
    value,
    kh0,
    kh_count,
    vh0,
    group,
    i_chunk,
    dim,
    tables,
    consts,
    gemm_dtype,
    inverse_dtype,
    evac,
):
    """Emit the state-independent work of one chunk for every local head.

    Returns [khead_tiles, vhead_tiles]. khead_tiles[jk] is the list of
    `_fused_khead_chunk`. vhead_tiles[jv] is [inv_t, ai_t, kd, beta_v], where
    inv_t is already in the matmul operand form of `inverse_dtype`.
    The Schur levels run level-major across the v-heads of each k-head group,
    so independent chains interleave in the instruction stream.
    """
    chunk_start = i_chunk * CHUNK
    c = i_chunk
    op_dtype = nl.bfloat16 if gemm_dtype == "bf16" else nl.float32
    khead_tiles = []
    vhead_tiles = []
    for jk in nl.static_range(kh_count):
        kt = _fused_khead_chunk(
            query, key, kh0 + jk, chunk_start, dim, gemm_dtype, evac
        )
        khead_tiles.append(kt)
        k_norm = kt[0]
        kk = kt[3]
        qk_t = kt[4]

        a_list = []
        rest = []
        for r in nl.static_range(group):
            jv = jk * group + r
            tab = tables[jv]
            gc_p = tab[0][0:CHUNK, c : c + 1]

            # gc_b[p, i] = gc_i: ones^T @ diag(gc).
            gc_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=gc_diag,
                data=consts["identity"],
                op0=nl.multiply,
                operand0=gc_p,
                engine=nisa.vector_engine,
            )
            gc_b_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=gc_b_psum, stationary=consts["ones_mat"], moving=gc_diag)

            # Row layout: wr[i, j] = exp(min(gc_i - gc_j, 0)).
            wr_arg = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=wr_arg,
                data=gc_b_psum,
                op0=nl.subtract,
                operand0=gc_p,
                reverse0=True,
                op1=nl.minimum,
                operand1=0.0,
                engine=nisa.vector_engine,
            )
            wr = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=wr, op=nl.exp, data=wr_arg, bias=None, scale=1.0)

            # A[i, j] = -beta_i (k_i . k_j) exp(gc_i - gc_j).
            a_mat = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=a_mat,
                data=kk,
                op0=nl.multiply,
                operand0=tab[4][0:CHUNK, c : c + 1],
                op1=nl.multiply,
                operand1=wr,
            )
            a_list.append(a_mat)

            # T layout: wt[j, i] = exp(min(gc_i - gc_j, 0)) for i >= j, else 0.
            wt_arg = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=wt_arg,
                data=gc_b_psum,
                op0=nl.subtract,
                operand0=gc_p,
                op1=nl.minimum,
                operand1=0.0,
                engine=nisa.vector_engine,
            )
            wt = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=wt, op=nl.exp, data=wt_arg, bias=None, scale=1.0)
            wt_masked = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=wt_masked, data1=wt, data2=consts["upper_diag"], op=nl.multiply
            )
            ai_t = nl.ndarray((P_MAX, P_MAX), dtype=op_dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=ai_t, data1=wt_masked, data2=qk_t, op=nl.multiply)

            kd = nl.ndarray((P_MAX, dim), dtype=op_dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=kd,
                data=k_norm,
                op0=nl.multiply,
                operand0=tab[3][0:CHUNK, c : c + 1],
                engine=nisa.vector_engine,
            )
            v_c = _load_chunk(value, vh0 + jv, chunk_start, dim)
            beta_v = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=beta_v,
                data=v_c,
                op0=nl.multiply,
                operand0=tab[6][0:CHUNK, c : c + 1],
                engine=nisa.vector_engine,
            )
            rest.append([ai_t, kd, beta_v])

        inv_list = _schur_inverse_t_multi(a_list, consts, inverse_dtype, evac)
        for r in nl.static_range(group):
            inv_op = _low_copy(inv_list[r], inverse_dtype, evac)
            vhead_tiles.append([inv_op, rest[r][0], rest[r][1], rest[r][2]])
    return [khead_tiles, vhead_tiles]


def _schur_inverse_t_multi(a_list, consts, inverse_dtype, evac):
    """Return (I - A_h)^-T as an fp32 tile for each strictly lower-triangular A_h.

    Each `a_list[h]` holds A_h in row layout: a[i, j] = A[i, j]. The block merge
    is the one of `fused._schur_inverse_t`. That routine first transposes its
    A_T input to this layout, so this routine saves one transpose for each
    head. The upper triangle of A_h is not read, because every mask plane is
    strictly lower-triangular.

    Each level is emitted for every head before the next level, so the
    independent chains of the heads interleave in the instruction stream.
    The accumulator stays in fp32. For `inverse_dtype` "bf16", the matmul
    operands are bf16.
    """
    schur_masks = consts["schur_masks"]
    Imat = consts["identity"]
    op_dtype = nl.bfloat16 if inverse_dtype == "bf16" else nl.float32
    n = len(a_list)

    inv_t_list = []
    inv_op_list = []
    for h in nl.static_range(n):
        masked0 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=masked0, data1=a_list[h], data2=schur_masks[0], op=nl.multiply
        )
        inv = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=inv, data1=Imat, data2=masked0, op=nl.add)
        inv_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=inv_t_psum, data=inv)
        inv_t = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        evac(inv_t, inv_t_psum)
        inv_t_list.append(inv_t)
        inv_op_list.append(_low_copy(inv, inverse_dtype, evac))

    for level in nl.static_range(6):
        for h in nl.static_range(n):
            if level != 0:
                inv_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=inv_psum, data=inv_t_list[h])
                inv_next = nl.ndarray((P_MAX, P_MAX), dtype=op_dtype, buffer=nl.sbuf)
                evac(inv_next, inv_psum)
                inv_op_list[h] = inv_next

            r_tile = nl.ndarray((P_MAX, P_MAX), dtype=op_dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=r_tile,
                data1=a_list[h],
                data2=schur_masks[level + 1],
                op=nl.multiply,
            )

            mid_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=mid_psum,
                stationary=r_tile,
                moving=_low_copy(inv_t_list[h], inverse_dtype, evac),
            )
            mid = nl.ndarray((P_MAX, P_MAX), dtype=op_dtype, buffer=nl.sbuf)
            evac(mid, mid_psum)

            off_t_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=off_t_psum, stationary=inv_op_list[h], moving=mid)
            inv_t_next = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=inv_t_next, data1=inv_t_list[h], data2=off_t_psum, op=nl.add
            )
            inv_t_list[h] = inv_t_next
    return inv_t_list


def _fused_phase_b(
    output,
    pa,
    states,
    states_op,
    vh0,
    kh_count,
    group,
    i_chunk,
    dim,
    tables,
    gemm_dtype,
    inverse_dtype,
    evac,
):
    """Emit the state chain of one chunk for every local head.

    `states[jk]` is the fp32 state of k-head group jk as a (128, R * 128) tile,
    with the R v-heads side by side on the free axis. `states_op[jk]` is the
    matmul operand form of the same state. The function returns the new
    [states, states_op].
    """
    chunk_start = i_chunk * CHUNK
    c = i_chunk
    khead_tiles = pa[0]
    vhead_tiles = pa[1]
    width = group * dim
    op_dtype = nl.bfloat16 if gemm_dtype == "bf16" else nl.float32
    new_states = []
    new_states_op = []
    for jk in nl.static_range(kh_count):
        kt = khead_tiles[jk]
        k_T = kt[1]
        q_T = kt[2]
        s_op = states_op[jk]

        # One matmul for the R v-heads: [k S_0 | k S_1 | ...].
        ks_psum = nl.ndarray((CHUNK, width), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ks_psum, stationary=k_T, moving=s_op)
        qs_psum = nl.ndarray((CHUNK, width), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=qs_psum, stationary=q_T, moving=s_op)
        qs = nl.ndarray((CHUNK, width), dtype=nl.float32, buffer=nl.sbuf)
        evac(qs, qs_psum)

        s_new = nl.ndarray((P_MAX, width), dtype=nl.float32, buffer=nl.sbuf)
        if gemm_dtype == "bf16":
            s_new_op = nl.ndarray((P_MAX, width), dtype=nl.bfloat16, buffer=nl.sbuf)
        else:
            s_new_op = s_new

        for r in nl.static_range(group):
            jv = jk * group + r
            tab = tables[jv]
            vt = vhead_tiles[jv]
            inv_t = vt[0]
            ai_t = vt[1]
            kd = vt[2]
            beta_v = vt[3]
            lo = r * dim
            hi = lo + dim

            # rhs = beta v - beta exp(gc) (k S).
            rhs_dtype = nl.bfloat16 if inverse_dtype == "bf16" else nl.float32
            rhs = nl.ndarray((CHUNK, dim), dtype=rhs_dtype, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=rhs,
                data=ks_psum[0:CHUNK, lo:hi],
                op0=nl.multiply,
                operand0=tab[5][0:CHUNK, c : c + 1],
                op1=nl.add,
                operand1=beta_v,
            )
            v_new_psum = nl.ndarray((CHUNK, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=v_new_psum, stationary=inv_t, moving=rhs)
            v_new = nl.ndarray((CHUNK, dim), dtype=op_dtype, buffer=nl.sbuf)
            evac(v_new, v_new_psum)

            intra_psum = nl.ndarray((CHUNK, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=intra_psum, stationary=ai_t, moving=v_new)

            # out = exp(gc) (q S) + intra.
            chunk_out = nl.ndarray((CHUNK, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=chunk_out,
                data=qs[0:CHUNK, lo:hi],
                op0=nl.multiply,
                operand0=tab[1][0:CHUNK, c : c + 1],
                op1=nl.add,
                operand1=intra_psum,
            )
            nisa.dma_copy(
                dst=output[vh0 + jv, chunk_start : chunk_start + CHUNK, 0:dim],
                src=chunk_out,
            )

            kv_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=kv_psum, stationary=kd, moving=v_new)

            # S = exp(gl) S + kd^T v_new. The bf16 operand copy comes from the
            # same inputs, so it does not wait for the fp32 result.
            egl = tab[2][0:P_MAX, c : c + 1]
            s_old = states[jk][0:P_MAX, lo:hi]
            nisa.scalar_tensor_tensor(
                dst=s_new[0:P_MAX, lo:hi],
                data=s_old,
                op0=nl.multiply,
                operand0=egl,
                op1=nl.add,
                operand1=kv_psum,
            )
            if gemm_dtype == "bf16":
                nisa.scalar_tensor_tensor(
                    dst=s_new_op[0:P_MAX, lo:hi],
                    data=s_old,
                    op0=nl.multiply,
                    operand0=egl,
                    op1=nl.add,
                    operand1=kv_psum,
                )
        new_states.append(s_new)
        new_states_op.append(s_new_op)
    return [new_states, new_states_op]


def _run_fused(
    query,
    key,
    value,
    g_in,
    beta_in,
    initial_state,
    identity,
    upper_diag_mask,
    level_masks,
    output,
    final_state_out,
    kh0,
    kh_count,
    vh0,
    group,
    num_chunks,
    dim,
    gemm_dtype,
    inverse_dtype,
    lookahead,
):
    evac = _Evac()
    consts = {}
    Imat = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Imat, src=identity)
    consts["identity"] = Imat
    upper_diag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=upper_diag, src=upper_diag_mask)
    consts["upper_diag"] = upper_diag
    consts["schur_masks"] = _load_schur_masks(level_masks)
    ones_mat = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_mat, value=1.0)
    consts["ones_mat"] = ones_mat
    ones_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_row, value=1.0)
    consts["ones_row"] = ones_row
    ones_rows = nl.ndarray((num_chunks, CHUNK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_rows, value=1.0)
    consts["ones_rows"] = ones_rows
    zero_col = nl.ndarray((num_chunks, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero_col, value=0.0)
    consts["zero_col"] = zero_col

    tables = []
    for jv in nl.static_range(kh_count * group):
        tables.append(_gate_tables(g_in, beta_in, vh0 + jv, num_chunks, consts, evac))

    width = group * dim
    states = []
    states_op = []
    for jk in nl.static_range(kh_count):
        s = nl.ndarray((P_MAX, width), dtype=nl.float32, buffer=nl.sbuf)
        for r in nl.static_range(group):
            nisa.dma_copy(
                dst=s[0:P_MAX, r * dim : (r + 1) * dim],
                src=initial_state[vh0 + jk * group + r, 0:P_MAX, 0:dim],
            )
        states.append(s)
        if gemm_dtype == "bf16":
            s_op = nl.ndarray((P_MAX, width), dtype=nl.bfloat16, buffer=nl.sbuf)
            evac(s_op, s)
            states_op.append(s_op)
        else:
            states_op.append(s)

    # lookahead=1 emits phase A of chunk c + 1 before phase B of chunk c.
    pa_args = (query, key, value, kh0, kh_count, vh0, group)
    pa_next = _fused_phase_a(
        *pa_args, 0, dim, tables, consts, gemm_dtype, inverse_dtype, evac
    )
    for i_chunk in nl.sequential_range(num_chunks):
        pa = pa_next
        has_next = i_chunk + 1 < num_chunks
        if lookahead and has_next:
            pa_next = _fused_phase_a(
                *pa_args,
                i_chunk + 1,
                dim,
                tables,
                consts,
                gemm_dtype,
                inverse_dtype,
                evac,
            )
        sb = _fused_phase_b(
            output,
            pa,
            states,
            states_op,
            vh0,
            kh_count,
            group,
            i_chunk,
            dim,
            tables,
            gemm_dtype,
            inverse_dtype,
            evac,
        )
        states = sb[0]
        states_op = sb[1]
        if not lookahead and has_next:
            pa_next = _fused_phase_a(
                *pa_args,
                i_chunk + 1,
                dim,
                tables,
                consts,
                gemm_dtype,
                inverse_dtype,
                evac,
            )

    for jk in nl.static_range(kh_count):
        for r in nl.static_range(group):
            nisa.dma_copy(
                dst=final_state_out[vh0 + jk * group + r, 0:P_MAX, 0:dim],
                src=states[jk][0:P_MAX, r * dim : (r + 1) * dim],
            )


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@nki.jit
def deltanet_prefill_v2(
    query: nl.ndarray,  # (num_kh, S, 128) bfloat16 or float32, raw (not normalized)
    key: nl.ndarray,  # (num_kh, S, 128) bfloat16 or float32, raw (not normalized)
    value: nl.ndarray,  # (num_vh, S, 128) bfloat16 or float32
    g_in: nl.ndarray,  # (num_vh, S, 1) float32, log decay, g <= 0
    beta_in: nl.ndarray,  # (num_vh, S, 1) float32
    initial_state: nl.ndarray,  # (num_vh, 128, 128) float32
    lower_mask: nl.ndarray,  # (128, 128) float32, strict lower triangle
    identity: nl.ndarray,  # (128, 128) float32
    upper_diag_mask: nl.ndarray,  # (128, 128) float32, upper triangle with diagonal
    level_masks: nl.ndarray,  # (7 * 128, 128) float32, Schur block-merge masks
    formulation="v1math",
    gemm_dtype="fp32",
    inverse_dtype="fp32",
    lookahead=0,
    revision=1,
):
    """Chunked DeltaNet prefill for all heads of the launch, version 2.

    Each program of the grid owns num_kh / num_programs k-heads and their
    num_vh / num_programs v-heads. v-head h reads k-head h // (num_vh / num_kh).
    Returns (output, final_state), both float32, with shapes (num_vh, S, 128)
    and (num_vh, 128, 128).
    """
    num_kh = query.shape[0]
    seq_len = query.shape[1]
    dim = query.shape[2]
    num_vh = value.shape[0]
    assert dim == P_MAX, "DeltaNet v2 needs head_dim 128"
    assert seq_len % CHUNK == 0, (
        "DeltaNet v2 needs a sequence length that is a multiple of 128"
    )
    assert num_vh % num_kh == 0, "DeltaNet v2 needs num_vh to be a multiple of num_kh"
    assert formulation in FORMULATIONS
    assert gemm_dtype in GEMM_DTYPES
    assert inverse_dtype in INVERSE_DTYPES
    group = num_vh // num_kh
    num_chunks = seq_len // CHUNK

    n_prog = nl.num_programs(axes=0)
    assert num_kh % n_prog == 0, "DeltaNet v2 needs num_kh to be a multiple of the grid"
    pid = nl.program_id(axis=0)
    kh_count = num_kh // n_prog
    kh0 = pid * kh_count
    vh0 = pid * kh_count * group

    output = nl.ndarray((num_vh, seq_len, dim), dtype=nl.float32, buffer=nl.shared_hbm)
    final_state_out = nl.ndarray(
        (num_vh, P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )

    if formulation == "v1math":
        _run_v1math(
            query,
            key,
            value,
            g_in,
            beta_in,
            initial_state,
            lower_mask,
            identity,
            level_masks,
            output,
            final_state_out,
            kh0,
            kh_count,
            vh0,
            group,
            num_chunks,
            dim,
        )
    else:
        assert num_chunks <= P_MAX, "DeltaNet v2 fused needs S <= 128 * 128"
        assert group * dim <= 512, "DeltaNet v2 fused needs R * 128 <= 512"
        _run_fused(
            query,
            key,
            value,
            g_in,
            beta_in,
            initial_state,
            identity,
            upper_diag_mask,
            level_masks,
            output,
            final_state_out,
            kh0,
            kh_count,
            vh0,
            group,
            num_chunks,
            dim,
            gemm_dtype,
            inverse_dtype,
            lookahead,
        )
    return output, final_state_out
