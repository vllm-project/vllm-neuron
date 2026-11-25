# SPDX-License-Identifier: Apache-2.0
"""DeltaNet decode step for the whole batch in one call, with the state updated in place.

The math is the math of decode_batched.py (the same float32 steps). The differences:

- The kernel reads the state rows of the batch from the full state buffer by slot index,
  and writes the new rows back to the same rows. The kernel returns the buffer, so the
  buffer is aliased and no gather or index_put_ around the call is needed. Each batch row
  is one indirect DMA (k on the partitions). A DMA with a dynamic offset costs about
  2 us to issue, so one DMA for each (row, head) pair would be much slower.
- All rows of a program go through one call, in tiles of at most 128 (row, head) pairs.
- The inputs have one row for each (batch row, v-head) pair, as for decode_batched.py.
  An HBM DMA needs one outer dimension that splits over the partitions, so the DMA
  cannot do the GQA repeat of q and k.
- The outer product k delta^T of each pair is one bf16 matmul with a contraction of 3
  rows: k_hi delta_hi + k_hi delta_lo + k_lo delta_hi (hi is the bf16 part, lo the bf16
  of the rest), accumulated in float32. decode_batched.py does a float32 matmul over all
  rows of the program with a one-hot delta, and one activation for each pair to make
  the one-hot.

Every NKI range is fully unrolled. The persistent NKI cache key hashes the source of the
kernel function only, so this file has no helper functions.
"""

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128
DIM = 128
L2_EPS_SQUARED = 1.0e-12
QUERY_SCALE = DIM**-0.5


@nki.jit
def deltanet_decode_inplace(
    query: nl.ndarray,  # (B * Hv, 128) raw (not normalized), repeated to the v-heads
    key: nl.ndarray,  # (B * Hv, 128) raw (not normalized), repeated to the v-heads
    value: nl.ndarray,  # (B * Hv, 128)
    z: nl.ndarray,  # (B * Hv, 128) input of the output gate
    a: nl.ndarray,  # (B * Hv, 1) raw decay projection
    b: nl.ndarray,  # (B * Hv, 1) raw beta projection
    a_log: nl.ndarray,  # (B * Hv, 1) float32, A_log of the v-head of each row
    dt_bias: nl.ndarray,  # (B * Hv, 1) float32, dt_bias of the v-head of each row
    norm_weight: nl.ndarray,  # (1, V) gated-norm weight
    state_buf: nl.ndarray,  # (N, Hv, K, V) state buffer, updated in place at the rows slot[b]
    slot: nl.ndarray,  # (B, 1) int32 state row of each batch row
    identity: nl.ndarray,  # (128, 128) float32
    norm_eps=1e-6,
    revision=1,
):
    """One DeltaNet decode step for B rows. Returns (out (B * Hv, V), state_buf).

    Program p of the grid owns batch rows [p * B / n, (p + 1) * B / n). Row r of the
    inputs and of out is batch r // Hv, v-head r % Hv. out is after the gated norm, in the
    dtype of query.
    """
    Hv = state_buf.shape[1]
    K = state_buf.shape[2]
    V = state_buf.shape[3]
    B = slot.shape[0]
    assert K == DIM and V == DIM, "The in-place decode kernel needs head_dim 128"
    assert query.shape[0] == B * Hv, (
        "The inputs need one row for each (batch row, v-head) pair"
    )
    n_prog = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    rows = B // n_prog
    assert rows * n_prog == B, (
        "The in-place decode kernel needs B to be a multiple of the grid"
    )
    assert Hv <= P_MAX, "The in-place decode kernel needs at most 128 v-heads"
    b0 = pid * rows
    max_tile_rows = P_MAX // Hv
    n_tiles = (rows + max_tile_rows - 1) // max_tile_rows
    tile_rows = (rows + n_tiles - 1) // n_tiles
    p_max = tile_rows * Hv

    out = nl.ndarray((B * Hv, V), dtype=query.dtype, buffer=nl.shared_hbm)
    # hi / lo bf16 parts of k and delta of a tile: k_hi | k_lo | delta_hi | delta_lo.
    scratch = nl.ndarray((p_max, 4 * K), dtype=nl.bfloat16, buffer=nl.hbm)

    # The state rows of the program: one indirect DMA for each batch row. k on the
    # partitions, (head, v) on the free axis.
    slot_sb = nl.ndarray((1, rows), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=slot_sb, src=slot.ap(pattern=[[1, 1], [1, rows]], offset=b0))
    row_pat = [[V, K], [K * V, Hv], [1, V]]
    s_rows = []
    for r in nl.static_range(rows):
        s = nl.ndarray((K, Hv * V), dtype=state_buf.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=s,
            src=state_buf.ap(
                pattern=row_pat,
                offset=0,
                scalar_offset=slot_sb.ap(pattern=[[rows, 1], [1, 1]], offset=r),
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.hwdge,
        )
        s_rows.append(s)

    ident = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ident, src=identity[0:P_MAX, 0:P_MAX])
    consts = nl.ndarray(
        (p_max, 2), dtype=nl.float32, buffer=nl.sbuf
    )  # L2 eps^2, norm eps
    nisa.memset(
        dst=consts[0:p_max, 0:1], value=L2_EPS_SQUARED, engine=nisa.vector_engine
    )
    nisa.memset(dst=consts[0:p_max, 1:2], value=norm_eps, engine=nisa.vector_engine)
    ones = nl.ndarray((p_max, DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0, engine=nisa.vector_engine)
    w_rows = nl.ndarray((p_max, V), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=w_rows, src=norm_weight.ap(pattern=[[0, p_max], [1, V]], offset=0)
    )
    # The 3 contraction rows of the outer products of a tile: KS = [k_hi; k_hi; k_lo],
    # DS = [delta_hi; delta_lo; delta_hi], pair p in columns p * 128 to p * 128 + 127.
    ks = nl.ndarray((3, p_max * K), dtype=nl.bfloat16, buffer=nl.sbuf)
    ds = nl.ndarray((3, p_max * V), dtype=nl.bfloat16, buffer=nl.sbuf)

    for t in nl.static_range(n_tiles):
        tr0 = t * tile_rows
        nr = tile_rows
        if tr0 + tile_rows > rows:
            nr = rows - tr0
        P = nr * Hv
        g0 = b0 + tr0  # first batch row of the tile

        # Loads. Pair p = (row p // Hv, v-head p % Hv) on the partitions.
        p0 = g0 * Hv
        q_in = nl.ndarray((P, K), dtype=query.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_in, src=query[p0 : p0 + P, 0:K])
        k_in = nl.ndarray((P, K), dtype=key.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_in, src=key[p0 : p0 + P, 0:K])
        v_in = nl.ndarray((P, V), dtype=value.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_in, src=value[p0 : p0 + P, 0:V])
        z_in = nl.ndarray((P, V), dtype=z.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=z_in, src=z[p0 : p0 + P, 0:V])
        scalars = nl.ndarray(
            (P, 4), dtype=nl.float32, buffer=nl.sbuf
        )  # a, b, a_log, dt_bias
        nisa.dma_copy(dst=scalars[0:P, 0:1], src=a[p0 : p0 + P, 0:1])
        nisa.dma_copy(dst=scalars[0:P, 1:2], src=b[p0 : p0 + P, 0:1])
        nisa.dma_copy(dst=scalars[0:P, 2:3], src=a_log[p0 : p0 + P, 0:1])
        nisa.dma_copy(dst=scalars[0:P, 3:4], src=dt_bias[p0 : p0 + P, 0:1])

        # l2norm of q and k: x * rsqrt(sum(x^2) + 1e-12). q also gets 1 / sqrt(128).
        sum_sq = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
        sq_q = nl.ndarray((P, DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(
            dst=sq_q,
            op=nl.square,
            data=q_in,
            reduce_op=nl.add,
            reduce_res=sum_sq[0:P, 0:1],
        )
        sq_k = nl.ndarray((P, DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(
            dst=sq_k,
            op=nl.square,
            data=k_in,
            reduce_op=nl.add,
            reduce_res=sum_sq[0:P, 1:2],
        )
        inv_norm = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=inv_norm[0:P, 0:1],
            op=nl.rsqrt,
            data=sum_sq[0:P, 0:1],
            bias=consts[0:P, 0:1],
            scale=1.0,
        )
        nisa.activation(
            dst=inv_norm[0:P, 1:2],
            op=nl.rsqrt,
            data=sum_sq[0:P, 1:2],
            bias=consts[0:P, 0:1],
            scale=1.0,
        )
        qn = nl.ndarray((P, DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=qn,
            data=q_in,
            op0=nl.multiply,
            operand0=inv_norm[0:P, 0:1],
            op1=nl.multiply,
            operand1=QUERY_SCALE,
            engine=nisa.vector_engine,
        )
        kn = nl.ndarray((P, DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=kn,
            data=k_in,
            op0=nl.multiply,
            operand0=inv_norm[0:P, 1:2],
            engine=nisa.vector_engine,
        )

        # Gates: g = -exp(a_log) * softplus(a + dt_bias), beta = sigmoid(b).
        softplus = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=softplus,
            op=nl.softplus,
            data=scalars[0:P, 0:1],
            bias=scalars[0:P, 3:4],
            scale=1.0,
        )
        exp_a = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_a, op=nl.exp, data=scalars[0:P, 2:3], scale=1.0)
        g = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=g,
            data=exp_a,
            op0=nl.multiply,
            operand0=-1.0,
            op1=nl.multiply,
            operand1=softplus,
        )
        eg = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=eg, op=nl.exp, data=g, scale=1.0)
        neg_beta = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=neg_beta, op=nl.sigmoid, data=scalars[0:P, 1:2], scale=1.0)
        nisa.tensor_scalar(
            dst=neg_beta,
            data=neg_beta,
            op0=nl.multiply,
            operand0=-1.0,
            engine=nisa.vector_engine,
        )

        # Column layouts of k and q for the Tensor Engine (k on the partitions).
        # Columns 4p to 4p + 3 are k_hi, k_lo, q_hi and q_lo of pair p.
        kt_ps = nl.ndarray((DIM, P), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=kt_ps, data=kn)
        qt_ps = nl.ndarray((DIM, P), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=qt_ps, data=qn)
        kq_t = nl.ndarray((DIM, 4 * P), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=kq_t[0:DIM, 0 : 4 * P : 4], src=kt_ps, engine=nisa.vector_engine
        )
        nisa.tensor_tensor(
            dst=kq_t[0:DIM, 1 : 4 * P : 4],
            data1=kt_ps,
            data2=kq_t[0:DIM, 0 : 4 * P : 4],
            op=nl.subtract,
            engine=nisa.vector_engine,
        )
        nisa.tensor_copy(
            dst=kq_t[0:DIM, 2 : 4 * P : 4], src=qt_ps, engine=nisa.vector_engine
        )
        nisa.tensor_tensor(
            dst=kq_t[0:DIM, 3 : 4 * P : 4],
            data1=qt_ps,
            data2=kq_t[0:DIM, 2 : 4 * P : 4],
            op=nl.subtract,
            engine=nisa.vector_engine,
        )

        # e^g on every partition: eg_bcast[m, p] = e^g of pair p.
        diag = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=diag,
            data=ident[0:P, 0:P],
            op0=nl.multiply,
            operand0=eg,
            engine=nisa.vector_engine,
        )
        eg_bcast_ps = nl.ndarray((DIM, P), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=eg_bcast_ps, stationary=ones[0:P, 0:DIM], moving=diag, accumulate=False
        )
        eg_bcast = nl.ndarray((DIM, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=eg_bcast, src=eg_bcast_ps, engine=nisa.vector_engine)

        # Phase 1: S^T k and S^T q of each pair, with v on the partitions.
        p1 = nl.ndarray((DIM, 4 * P), dtype=nl.float32, buffer=nl.psum)
        for p in nl.static_range(P):
            sr = s_rows[tr0 + p // Hv]
            h = p % Hv
            nisa.nc_matmul(
                dst=p1[0:DIM, 4 * p : 4 * (p + 1)],
                stationary=sr[0:K, h * V : (h + 1) * V],
                moving=kq_t[0:DIM, 4 * p : 4 * (p + 1)],
                accumulate=False,
            )
        p1_sb = nl.ndarray((DIM, 4 * P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=p1_sb, src=p1, engine=nisa.vector_engine)
        kvm_t = nl.ndarray((DIM, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=kvm_t,
            data1=p1_sb[0:DIM, 0 : 4 * P : 4],
            data2=p1_sb[0:DIM, 1 : 4 * P : 4],
            op=nl.add,
            engine=nisa.vector_engine,
        )
        qs_t = nl.ndarray((DIM, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=qs_t,
            data1=p1_sb[0:DIM, 2 : 4 * P : 4],
            data2=p1_sb[0:DIM, 3 : 4 * P : 4],
            op=nl.add,
            engine=nisa.vector_engine,
        )
        kvm_ps = nl.ndarray((P, DIM), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=kvm_ps, data=kvm_t)
        qs_ps = nl.ndarray((P, DIM), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=qs_ps, data=qs_t)

        # delta = (v - e^g * S^T k) * beta
        kvm_minus_v = nl.ndarray((P, DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=kvm_minus_v,
            data=kvm_ps,
            op0=nl.multiply,
            operand0=eg,
            op1=nl.subtract,
            operand1=v_in,
        )
        delta = nl.ndarray((P, DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=delta,
            data=kvm_minus_v,
            op0=nl.multiply,
            operand0=neg_beta,
            engine=nisa.vector_engine,
        )

        # hi / lo bf16 parts of k and delta, then the 3 contraction rows (one partition
        # each, the pairs on the free axis). The rows go through an HBM scratch tensor: a
        # DMA from SBUF to SBUF needs the same shape on both sides.
        hl = nl.ndarray((P, 4 * K), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=hl[0:P, 0:K], src=kn, engine=nisa.vector_engine)
        nisa.tensor_tensor(
            dst=hl[0:P, K : 2 * K],
            data1=kn,
            data2=hl[0:P, 0:K],
            op=nl.subtract,
            engine=nisa.vector_engine,
        )
        nisa.tensor_copy(
            dst=hl[0:P, 2 * K : 3 * K], src=delta, engine=nisa.vector_engine
        )
        nisa.tensor_tensor(
            dst=hl[0:P, 3 * K : 4 * K],
            data1=delta,
            data2=hl[0:P, 2 * K : 3 * K],
            op=nl.subtract,
            engine=nisa.vector_engine,
        )
        nisa.dma_copy(dst=scratch[0:P, 0 : 4 * K], src=hl)
        nisa.dma_copy(
            dst=ks[0:1, 0 : P * K],
            src=scratch.ap(pattern=[[4 * K * p_max, 1], [4 * K, P], [1, K]], offset=0),
        )
        nisa.dma_copy(
            dst=ks[1:2, 0 : P * K],
            src=scratch.ap(pattern=[[4 * K * p_max, 1], [4 * K, P], [1, K]], offset=0),
        )
        nisa.dma_copy(
            dst=ks[2:3, 0 : P * K],
            src=scratch.ap(pattern=[[4 * K * p_max, 1], [4 * K, P], [1, K]], offset=K),
        )
        nisa.dma_copy(
            dst=ds[0:1, 0 : P * V],
            src=scratch.ap(
                pattern=[[4 * K * p_max, 1], [4 * K, P], [1, K]], offset=2 * K
            ),
        )
        nisa.dma_copy(
            dst=ds[1:2, 0 : P * V],
            src=scratch.ap(
                pattern=[[4 * K * p_max, 1], [4 * K, P], [1, K]], offset=3 * K
            ),
        )
        nisa.dma_copy(
            dst=ds[2:3, 0 : P * V],
            src=scratch.ap(
                pattern=[[4 * K * p_max, 1], [4 * K, P], [1, K]], offset=2 * K
            ),
        )

        # Phase 2: S_new = e^g * S + k delta^T for each pair, in place in SBUF. After the
        # last head of a row, write the row back to its state row.
        for p in nl.static_range(P):
            r = tr0 + p // Hv
            h = p % Hv
            sr = s_rows[r]
            outer_ps = nl.ndarray((DIM, DIM), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(
                dst=outer_ps,
                stationary=ks[0:3, p * K : (p + 1) * K],
                moving=ds[0:3, p * V : (p + 1) * V],
                accumulate=False,
            )
            nisa.scalar_tensor_tensor(
                dst=sr[0:K, h * V : (h + 1) * V],
                data=sr[0:K, h * V : (h + 1) * V],
                op0=nl.multiply,
                operand0=eg_bcast[0:DIM, p : p + 1],
                op1=nl.add,
                operand1=outer_ps,
            )
            if h == Hv - 1:
                nisa.dma_copy(
                    dst=state_buf.ap(
                        pattern=row_pat,
                        offset=0,
                        scalar_offset=slot_sb.ap(pattern=[[rows, 1], [1, 1]], offset=r),
                        indirect_dim=0,
                    ),
                    src=sr,
                    dge_mode=nisa.dge_mode.hwdge,
                )

        # o = e^g (S^T q) + (q . k) delta = q^T S_new, then the gated norm:
        # o * rsqrt(mean(o^2) + eps) * weight * silu(z).
        qk = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        qk_prod = nl.ndarray((P, DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=qk_prod, data1=qn, data2=kn, op=nl.multiply, engine=nisa.vector_engine
        )
        nisa.tensor_reduce(dst=qk, data=qk_prod, op=nl.add, axis=1)
        qk_delta = nl.ndarray((P, DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=qk_delta,
            data=delta,
            op0=nl.multiply,
            operand0=qk,
            engine=nisa.vector_engine,
        )
        o = nl.ndarray((P, DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=o,
            data=qs_ps,
            op0=nl.multiply,
            operand0=eg,
            op1=nl.add,
            operand1=qk_delta,
        )
        o_ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        o_sq = nl.ndarray((P, DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(
            dst=o_sq, op=nl.square, data=o, reduce_op=nl.add, reduce_res=o_ss
        )
        rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=rstd, op=nl.rsqrt, data=o_ss, bias=consts[0:P, 1:2], scale=1.0 / DIM
        )
        silu_z = nl.ndarray((P, DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=silu_z, op=nl.silu, data=z_in, scale=1.0)
        normed = nl.ndarray((P, DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=normed,
            data=o,
            op0=nl.multiply,
            operand0=rstd,
            op1=nl.multiply,
            operand1=w_rows[0:P, 0:V],
        )
        y = nl.ndarray((P, DIM), dtype=query.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=y, data1=normed, data2=silu_z, op=nl.multiply, engine=nisa.vector_engine
        )
        nisa.dma_copy(dst=out[p0 : p0 + P, 0:V], src=y)

    return out, state_buf
