"""NKI kernels for DeltaNet gated delta rule recurrent forward.

NKI v3 (SDK 2.29, NKI 0.3.0). Processes a SINGLE (batch, head) pair per kernel call.
The caller loops over (B, H) in PyTorch and calls this kernel for each pair.

Input layout: All inputs are 2D contiguous tensors (S, 128).
Each call processes one (batch, head) element's full sequence.

k_dim = v_dim = 128, which matches SBUF tile partition dimension exactly.
g and beta are scalars per token, expanded to (S, 128) by the caller.

Two kernel variants:
  deltanet_recurrent_fwd        -- returns output only (original)
  deltanet_recurrent_fwd_state  -- returns (output, final_state) for CTE->TKG carry-over
  deltanet_recurrent_step       -- one-token state-in/state-out decode step
  deltanet_recurrent_step_batched -- one-token batched-head decode step
"""

import nki
import nki.isa as nisa
import nki.language as nl

# Partition dimension max (NeuronCore SBUF tile width)
P_MAX = 128

# Shuffle mask: broadcast partition 0 to all partitions in a 32-wide group
_BROADCAST_MASK = [0] * 32


@nki.jit
def _deltanet_recurrent_step_batched_kernel(
    query: nl.ndarray,  # (BH*128, 1) float32 — columns stacked vertically
    key: nl.ndarray,  # (BH*128, 1) float32
    value: nl.ndarray,  # (BH*128, 1) float32
    g_in: nl.ndarray,  # (BH, 1) float32, log-decay scalar per head
    beta_in: nl.ndarray,  # (BH, 1) float32, write-gate scalar per head
    state_in: nl.ndarray,  # (BH * 128, 128) float32/bfloat16
):
    """Single-launch batched-head one-token DeltaNet decode step.

    Uses standard NKI tensor indexing (no .ap()) for torch.compile compatibility.
    Inputs q/k/v are pre-transposed to (BH*128, 1) column layout by the caller
    so that each head's 128-element vector is already along the partition dimension.
    """
    batch_heads, _ = g_in.shape
    dim = P_MAX

    output = nl.ndarray(query.shape, dtype=query.dtype, buffer=nl.shared_hbm)
    state_out = nl.ndarray(state_in.shape, dtype=state_in.dtype, buffer=nl.shared_hbm)

    for bh in nl.sequential_range(batch_heads):
        head_offset = bh * P_MAX

        q_t = nl.ndarray((P_MAX, 1), dtype=query.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_t, src=query[head_offset : head_offset + P_MAX, 0:1])

        k_t = nl.ndarray((P_MAX, 1), dtype=key.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_t, src=key[head_offset : head_offset + P_MAX, 0:1])

        v_t = nl.ndarray((P_MAX, 1), dtype=value.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_t, src=value[head_offset : head_offset + P_MAX, 0:1])

        g_t = nl.ndarray((P_MAX, 1), dtype=g_in.dtype, buffer=nl.sbuf)
        g_scalar = nl.ndarray((1, 1), dtype=g_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=g_scalar, src=g_in[bh : bh + 1, 0:1])
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=g_scalar[0:1, 0:1],
                dst=g_t[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                shuffle_mask=_BROADCAST_MASK,
            )

        beta_t = nl.ndarray((P_MAX, 1), dtype=beta_in.dtype, buffer=nl.sbuf)
        beta_scalar = nl.ndarray((1, 1), dtype=beta_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=beta_scalar, src=beta_in[bh : bh + 1, 0:1])
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=beta_scalar[0:1, 0:1],
                dst=beta_t[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                shuffle_mask=_BROADCAST_MASK,
            )

        state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=state,
            src=state_in[head_offset : head_offset + P_MAX, 0:dim],
        )

        exp_g = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_g, op=nl.exp, data=g_t, bias=None, scale=1.0)

        state_decayed = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=state_decayed,
            data=state,
            op0=nl.multiply,
            operand0=exp_g,
            engine=nisa.vector_engine,
        )

        kv_mem_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=kv_mem_psum, stationary=state_decayed, moving=k_t)
        kv_mem = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kv_mem, src=kv_mem_psum)

        v_sub = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=v_sub, data1=v_t, data2=kv_mem, op=nl.subtract)

        delta = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=delta,
            data=v_sub,
            op0=nl.multiply,
            operand0=beta_t,
            engine=nisa.vector_engine,
        )

        delta_row_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=delta_row_psum, data=delta)

        delta_row_sb = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=delta_row_sb, src=delta_row_psum)

        delta_broadcast = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=delta_row_sb[0:1, 0:P_MAX],
                dst=delta_broadcast[i_shuf * 32 : i_shuf * 32 + 32, 0:P_MAX],
                shuffle_mask=_BROADCAST_MASK,
            )

        state_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=state_new,
            data=delta_broadcast,
            op0=nl.multiply,
            operand0=k_t,
            op1=nl.add,
            operand1=state_decayed,
        )

        o_t_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=o_t_psum, stationary=state_new, moving=q_t)
        o_t = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_t, src=o_t_psum)

        nisa.dma_copy(
            dst=output[head_offset : head_offset + P_MAX, 0:1],
            src=o_t,
        )
        nisa.dma_copy(
            dst=state_out[head_offset : head_offset + P_MAX, 0:dim],
            src=state_new,
        )

    return output, state_out


def deltanet_recurrent_step_batched(
    query,
    key,
    value,
    g_in,
    beta_in,
    state_in,
):
    """Launch the one-token DeltaNet decode step across flattened BH heads."""
    return _deltanet_recurrent_step_batched_kernel(
        query,
        key,
        value,
        g_in,
        beta_in,
        state_in,
    )


@nki.jit
def deltanet_recurrent_step(
    query: nl.ndarray,  # (1, 128) float32
    key: nl.ndarray,  # (1, 128) float32
    value: nl.ndarray,  # (1, 128) float32
    g_in: nl.ndarray,  # (1, 128) float32, log-decay broadcast to 128
    beta_in: nl.ndarray,  # (1, 128) float32, write gate broadcast to 128
    state_in: nl.ndarray,  # (128, 128) float32
):
    """Stateful one-token DeltaNet decode step.

    This is the token-generation equivalent of one iteration from
    ``deltanet_recurrent_fwd_state``.  The caller supplies the recurrent
    state restored from the decode cache and receives the updated state.

    Returns:
        output:    (1, 128) float32
        state_out: (128, 128) float32
    """
    _, dim = query.shape

    output = nl.ndarray((1, dim), dtype=query.dtype, buffer=nl.shared_hbm)
    state_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm)

    q_t = nl.ndarray((P_MAX, 1), dtype=query.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_t, src=query.ap(pattern=[[1, P_MAX]], offset=0))

    k_t = nl.ndarray((P_MAX, 1), dtype=key.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_t, src=key.ap(pattern=[[1, P_MAX]], offset=0))

    v_t = nl.ndarray((P_MAX, 1), dtype=value.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=v_t, src=value.ap(pattern=[[1, P_MAX]], offset=0))

    g_t = nl.ndarray((P_MAX, 1), dtype=g_in.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=g_t, src=g_in.ap(pattern=[[1, P_MAX]], offset=0))

    beta_t = nl.ndarray((P_MAX, 1), dtype=beta_in.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=beta_t, src=beta_in.ap(pattern=[[1, P_MAX]], offset=0))

    state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=state, src=state_in)

    exp_g = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=exp_g, op=nl.exp, data=g_t, bias=None, scale=1.0)

    state_decayed = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=state_decayed,
        data=state,
        op0=nl.multiply,
        operand0=exp_g,
        engine=nisa.vector_engine,
    )
    nisa.tensor_copy(dst=state, src=state_decayed)

    kv_mem_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=kv_mem_psum, stationary=state, moving=k_t)
    kv_mem = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=kv_mem, src=kv_mem_psum)

    v_sub = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=v_sub, data1=v_t, data2=kv_mem, op=nl.subtract)

    delta = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=delta,
        data=v_sub,
        op0=nl.multiply,
        operand0=beta_t,
        engine=nisa.vector_engine,
    )

    delta_row_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=delta_row_psum, data=delta)

    delta_row_sb = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=delta_row_sb, src=delta_row_psum)

    delta_broadcast = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    for i_shuf in nl.static_range(P_MAX // 32):
        nisa.nc_stream_shuffle(
            src=delta_row_sb[0:1, 0:P_MAX],
            dst=delta_broadcast[i_shuf * 32 : i_shuf * 32 + 32, 0:P_MAX],
            shuffle_mask=_BROADCAST_MASK,
        )

    outer_prod = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=outer_prod,
        data=delta_broadcast,
        op0=nl.multiply,
        operand0=k_t,
        engine=nisa.vector_engine,
    )

    state_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=state_new, data1=state, data2=outer_prod, op=nl.add)
    nisa.tensor_copy(dst=state, src=state_new)

    o_t_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=o_t_psum, stationary=state, moving=q_t)
    o_t = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o_t, src=o_t_psum)

    nisa.dma_copy(dst=output.ap(pattern=[[1, dim]], offset=0), src=o_t)
    nisa.dma_copy(dst=state_out, src=state)

    return output, state_out


@nki.jit
def deltanet_recurrent_fwd(
    query: nl.ndarray,  # (S, 128) float32
    key: nl.ndarray,  # (S, 128) float32
    value: nl.ndarray,  # (S, 128) float32
    g_in: nl.ndarray,  # (S, 128) float32, log-decay broadcast to 128
    beta_in: nl.ndarray,  # (S, 128) float32, write gate broadcast to 128
) -> nl.ndarray:
    """NKI kernel for DeltaNet recurrent forward -- single (batch, head).

    Iterates over sequence tokens with sequential_range.
    State matrix (128 x 128) lives in SBUF.

    Args:
        query:    (S, 128) float32
        key:      (S, 128) float32
        value:    (S, 128) float32
        g_in:     (S, 128) float32
        beta_in:  (S, 128) float32

    Returns:
        output:   (S, 128) float32
    """
    seq_len, dim = query.shape

    # Output tensor in HBM
    output = nl.ndarray((seq_len, dim), dtype=query.dtype, buffer=nl.shared_hbm)

    # Stride: for 2D (S, D), dim0 stride = D=128, dim1 stride = 1
    seq_stride = dim

    # Initialize recurrent state in SBUF: (128, 128)
    state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=state, value=0.0)

    # Sequential loop over tokens (state-dependent)
    for t in nl.sequential_range(seq_len):
        tok_offset = t * seq_stride

        # ---- Load inputs for token t ----
        q_t = nl.ndarray((P_MAX, 1), dtype=query.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=q_t,
            src=query.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        k_t = nl.ndarray((P_MAX, 1), dtype=key.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=k_t,
            src=key.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        v_t = nl.ndarray((P_MAX, 1), dtype=value.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=v_t,
            src=value.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        g_t = nl.ndarray((P_MAX, 1), dtype=g_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=g_t,
            src=g_in.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        beta_t = nl.ndarray((P_MAX, 1), dtype=beta_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=beta_t,
            src=beta_in.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        # ---- Step 1: Decay state -- state = state * exp(g_t) ----
        exp_g = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_g, op=nl.exp, data=g_t, bias=None, scale=1.0)

        state_decayed = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=state_decayed,
            data=state,
            op0=nl.multiply,
            operand0=exp_g,
            engine=nisa.vector_engine,
        )
        nisa.tensor_copy(dst=state, src=state_decayed)

        # ---- Step 2: Read memory -- kv_mem = state^T @ k_t ----
        kv_mem_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=kv_mem_psum, stationary=state, moving=k_t)
        kv_mem = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kv_mem, src=kv_mem_psum)

        # ---- Step 3: delta = (v_t - kv_mem) * beta_t ----
        v_sub = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=v_sub, data1=v_t, data2=kv_mem, op=nl.subtract)

        delta = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=delta,
            data=v_sub,
            op0=nl.multiply,
            operand0=beta_t,
            engine=nisa.vector_engine,
        )

        # ---- Step 4: state += outer(k_t, delta) ----
        # Broadcast multiply: outer[i,j] = k_t[i] * delta[j]
        # 1) Transpose delta (128,1) -> (1,128) in PSUM
        # 2) Copy PSUM (1,128) -> SBUF (128,128) -- partition broadcast
        # 3) Multiply by k_t (128,1) which broadcasts across free dim
        # This avoids the nc_matmul P=1 outer product (wastes 127/128 TE lanes).

        # Transpose delta to get values along free dimension
        delta_row_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=delta_row_psum, data=delta)

        # Copy PSUM (1, 128) -> SBUF (1, 128) first (NKI 0.3.0 requires matching P dims)
        delta_row_sb = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=delta_row_sb, src=delta_row_psum)

        # Broadcast (1, 128) SBUF -> (128, 128) SBUF via nc_stream_shuffle
        # Each partition row gets the same delta values
        delta_broadcast = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=delta_row_sb[0:1, 0:P_MAX],
                dst=delta_broadcast[i_shuf * 32 : i_shuf * 32 + 32, 0:P_MAX],
                shuffle_mask=_BROADCAST_MASK,
            )

        # Element-wise multiply: outer[i,j] = delta_broadcast[i,j] * k_t[i,0]
        # tensor_scalar broadcasts (P,1) k_t across all F columns
        outer_prod = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=outer_prod,
            data=delta_broadcast,
            op0=nl.multiply,
            operand0=k_t,
            engine=nisa.vector_engine,
        )

        # Accumulate into state
        state_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=state_new, data1=state, data2=outer_prod, op=nl.add)
        nisa.tensor_copy(dst=state, src=state_new)

        # ---- Step 5: o_t = state^T @ q_t ----
        o_t_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=o_t_psum, stationary=state, moving=q_t)
        o_t = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_t, src=o_t_psum)

        # ---- Store output for token t ----
        nisa.dma_copy(
            dst=output.ap(pattern=[[1, dim]], offset=tok_offset),
            src=o_t,
        )

    return output


@nki.jit
def deltanet_recurrent_fwd_state(
    query: nl.ndarray,  # (S, 128) float32
    key: nl.ndarray,  # (S, 128) float32
    value: nl.ndarray,  # (S, 128) float32
    g_in: nl.ndarray,  # (S, 128) float32, log-decay broadcast to 128
    beta_in: nl.ndarray,  # (S, 128) float32, write gate broadcast to 128
):
    """NKI kernel for DeltaNet recurrent forward with final state output.

    Same recurrence as deltanet_recurrent_fwd, but ALSO writes the final
    recurrent state (128, 128) to an output HBM buffer.  This enables
    CTE -> TKG state carry-over.

    Returns:
        output:      (S, 128)   float32 -- per-token output
        final_state: (128, 128) float32 -- recurrent state after last token
    """
    seq_len, dim = query.shape

    # Output tensors in HBM
    output = nl.ndarray((seq_len, dim), dtype=query.dtype, buffer=nl.shared_hbm)
    final_state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm)

    # Stride: for 2D (S, D), dim0 stride = D=128, dim1 stride = 1
    seq_stride = dim

    # Initialize recurrent state in SBUF: (128, 128)
    state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=state, value=0.0)

    # Sequential loop over tokens (state-dependent)
    for t in nl.sequential_range(seq_len):
        tok_offset = t * seq_stride

        # ---- Load inputs for token t ----
        q_t = nl.ndarray((P_MAX, 1), dtype=query.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=q_t,
            src=query.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        k_t = nl.ndarray((P_MAX, 1), dtype=key.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=k_t,
            src=key.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        v_t = nl.ndarray((P_MAX, 1), dtype=value.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=v_t,
            src=value.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        g_t = nl.ndarray((P_MAX, 1), dtype=g_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=g_t,
            src=g_in.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        beta_t = nl.ndarray((P_MAX, 1), dtype=beta_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=beta_t,
            src=beta_in.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        # ---- Step 1: Decay state -- state = state * exp(g_t) ----
        exp_g = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_g, op=nl.exp, data=g_t, bias=None, scale=1.0)

        state_decayed = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=state_decayed,
            data=state,
            op0=nl.multiply,
            operand0=exp_g,
            engine=nisa.vector_engine,
        )
        nisa.tensor_copy(dst=state, src=state_decayed)

        # ---- Step 2: Read memory -- kv_mem = state^T @ k_t ----
        kv_mem_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=kv_mem_psum, stationary=state, moving=k_t)
        kv_mem = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kv_mem, src=kv_mem_psum)

        # ---- Step 3: delta = (v_t - kv_mem) * beta_t ----
        v_sub = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=v_sub, data1=v_t, data2=kv_mem, op=nl.subtract)

        delta = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=delta,
            data=v_sub,
            op0=nl.multiply,
            operand0=beta_t,
            engine=nisa.vector_engine,
        )

        # ---- Step 4: state += outer(k_t, delta) ----
        # Broadcast multiply: outer[i,j] = k_t[i] * delta[j]
        delta_row_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=delta_row_psum, data=delta)

        # Copy PSUM (1, 128) -> SBUF (1, 128) first (NKI 0.3.0 requires matching P dims)
        delta_row_sb = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=delta_row_sb, src=delta_row_psum)

        # Broadcast (1, 128) SBUF -> (128, 128) SBUF via nc_stream_shuffle
        delta_broadcast = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=delta_row_sb[0:1, 0:P_MAX],
                dst=delta_broadcast[i_shuf * 32 : i_shuf * 32 + 32, 0:P_MAX],
                shuffle_mask=_BROADCAST_MASK,
            )

        outer_prod = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=outer_prod,
            data=delta_broadcast,
            op0=nl.multiply,
            operand0=k_t,
            engine=nisa.vector_engine,
        )

        state_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=state_new, data1=state, data2=outer_prod, op=nl.add)
        nisa.tensor_copy(dst=state, src=state_new)

        # ---- Step 5: o_t = state^T @ q_t ----
        o_t_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=o_t_psum, stationary=state, moving=q_t)
        o_t = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_t, src=o_t_psum)

        # ---- Store output for token t ----
        nisa.dma_copy(
            dst=output.ap(pattern=[[1, dim]], offset=tok_offset),
            src=o_t,
        )

    # ---- Write final state to HBM ----
    # state is (128, 128) in SBUF, copy to final_state in HBM
    # Use dma_copy with full tile: P_MAX rows, dim cols
    nisa.dma_copy(dst=final_state, src=state)

    return output, final_state
