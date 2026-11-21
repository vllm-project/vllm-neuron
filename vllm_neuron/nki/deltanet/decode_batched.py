# SPDX-License-Identifier: Apache-2.0
"""DeltaNet decode step for all heads of a program in one pass (one token for each row).

The other decode kernels do one head at a time. Each head is a chain of about 16
operations on different engines, so the time is the chain latency. This kernel
puts the heads of a program on the partitions, and does each step of the chain
once for all heads:

- The l2norm of q and k, the gates and the gated norm run once on a
  (heads, 128) tile.
- The Tensor Engine does S^T [k | q] for each head into one PSUM tile. The output
  is o = e^g (S^T q) + (q . k) delta, which is q^T S_new. So the output does not
  wait for the state update.
- The state update S_new = e^g S + k delta^T is one outer-product matmul and one
  Vector operation for each head. The heads do not depend on each other.

With a bf16 state, the S^T [k | q] matmul uses the bf16 state as the stationary
tile, which is exact, and a hi/lo bf16 split of k and q as the moving tile. The
split keeps k and q to about 2^-16, and the matmul takes 1 pass. With a float32
state, the matmul is a float32 matmul.

Every NKI range is fully unrolled. The persistent NKI cache key hashes the
source of the kernel function only, not the source of the helper functions.
Increase `revision` when you change a helper function.
"""

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128
DIM = 128
L2_EPS_SQUARED = 1.0e-12
QUERY_SCALE = DIM**-0.5


def _gated_norm_output(
    out, qn, kn, delta, qs_ps, eg, consts, z_in, w_rows, rows, r0, out_dtype
):
    """Compute o = e^g (S^T q) + (q . k) delta, apply the gated norm and write the rows to out."""
    qk = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    qk_prod = nl.ndarray((rows, DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=qk_prod, data1=qn, data2=kn, op=nl.multiply)
    nisa.tensor_reduce(dst=qk, data=qk_prod, op=nl.add, axis=1)
    qk_delta = nl.ndarray((rows, DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=qk_delta,
        data=delta,
        op0=nl.multiply,
        operand0=qk,
        engine=nisa.vector_engine,
    )
    o = nl.ndarray((rows, DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=o, data=qs_ps, op0=nl.multiply, operand0=eg, op1=nl.add, operand1=qk_delta
    )
    # Gated norm: o * rsqrt(mean(o^2) + eps) * weight * silu(z).
    o_ss = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    o_sq = nl.ndarray((rows, DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(
        dst=o_sq, op=nl.square, data=o, reduce_op=nl.add, reduce_res=o_ss
    )
    rstd = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=rstd, op=nl.rsqrt, data=o_ss, bias=consts[0:rows, 1:2], scale=1.0 / DIM
    )
    silu_z = nl.ndarray((rows, DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=silu_z, op=nl.silu, data=z_in, scale=1.0)
    normed = nl.ndarray((rows, DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=normed,
        data=o,
        op0=nl.multiply,
        operand0=rstd,
        op1=nl.multiply,
        operand1=w_rows,
    )
    y = nl.ndarray((rows, DIM), dtype=out_dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=y, data1=normed, data2=silu_z, op=nl.multiply)
    nisa.dma_copy(dst=out[r0 : r0 + rows, 0:DIM], src=y)


@nki.jit
def deltanet_decode_batched(
    query: nl.ndarray,  # (BH, 128), raw (not normalized), repeated to the v-heads
    key: nl.ndarray,  # (BH, 128), raw (not normalized), repeated to the v-heads
    value: nl.ndarray,  # (BH, 128)
    z: nl.ndarray,  # (BH, 128), input of the output gate
    a: nl.ndarray,  # (BH, 1) float32, raw decay projection
    b: nl.ndarray,  # (BH, 1) float32, raw beta projection
    a_log: nl.ndarray,  # (BH, 1) float32, A_log of the v-head of each row
    dt_bias: nl.ndarray,  # (BH, 1) float32, dt_bias of the v-head of each row
    norm_weight: nl.ndarray,  # (BH, 128) float32, gated-norm weight for each row
    state: nl.ndarray,  # (BH, 128, 128) bfloat16 or float32, state[row, k, v]
    identity: nl.ndarray,  # (128, 128) float32
    norm_eps=1e-6,
    revision=1,
):
    """One DeltaNet decode step with the gates, the l2norm and the gated norm.

    BH is the number of rows: batch * v-heads, batch-major. Program p of the
    grid owns rows [p * BH / n, (p + 1) * BH / n), and each program can own at
    most 128 rows. Returns the output after the gated norm (BH, 128) in the dtype
    of query, and the new state (BH, 128, 128) in the dtype of state.
    """
    BH = query.shape[0]
    assert query.shape[1] == DIM and value.shape[1] == DIM, (
        "The batched decode kernel needs head_dim 128"
    )
    n_prog = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    rows = BH // n_prog
    assert rows * n_prog == BH, (
        "The batched decode kernel needs BH to be a multiple of the grid"
    )
    assert rows <= P_MAX, (
        "The batched decode kernel needs at most 128 rows for each program"
    )
    r0 = pid * rows
    bf16_state = state.dtype == nl.bfloat16

    out = nl.ndarray((BH, DIM), dtype=query.dtype, buffer=nl.shared_hbm)
    state_out = nl.ndarray((BH, DIM, DIM), dtype=state.dtype, buffer=nl.shared_hbm)

    # Loads. q, k, v and z put the rows on the partitions. The state puts k on
    # the partitions and (row, v) on the free axis.
    q_in = nl.ndarray((rows, DIM), dtype=query.dtype, buffer=nl.sbuf)
    k_in = nl.ndarray((rows, DIM), dtype=key.dtype, buffer=nl.sbuf)
    v_in = nl.ndarray((rows, DIM), dtype=value.dtype, buffer=nl.sbuf)
    z_in = nl.ndarray((rows, DIM), dtype=z.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_in, src=query[r0 : r0 + rows, 0:DIM])
    nisa.dma_copy(dst=k_in, src=key[r0 : r0 + rows, 0:DIM])
    nisa.dma_copy(dst=v_in, src=value[r0 : r0 + rows, 0:DIM])
    nisa.dma_copy(dst=z_in, src=z[r0 : r0 + rows, 0:DIM])
    scalars = nl.ndarray(
        (rows, 4), dtype=nl.float32, buffer=nl.sbuf
    )  # a, b, a_log, dt_bias
    nisa.dma_copy(dst=scalars[0:rows, 0:1], src=a[r0 : r0 + rows, 0:1])
    nisa.dma_copy(dst=scalars[0:rows, 1:2], src=b[r0 : r0 + rows, 0:1])
    nisa.dma_copy(dst=scalars[0:rows, 2:3], src=a_log[r0 : r0 + rows, 0:1])
    nisa.dma_copy(dst=scalars[0:rows, 3:4], src=dt_bias[r0 : r0 + rows, 0:1])
    w_rows = nl.ndarray((rows, DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_rows, src=norm_weight[r0 : r0 + rows, 0:DIM])
    ident = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ident, src=identity[0:P_MAX, 0:P_MAX])
    s_in = nl.ndarray((DIM, rows * DIM), dtype=state.dtype, buffer=nl.sbuf)
    for h in nl.static_range(rows):
        nisa.dma_copy(
            dst=s_in[0:DIM, h * DIM : (h + 1) * DIM], src=state[r0 + h, 0:DIM, 0:DIM]
        )

    consts = nl.ndarray(
        (rows, 2), dtype=nl.float32, buffer=nl.sbuf
    )  # L2 eps^2, norm eps
    nisa.memset(dst=consts[0:rows, 0:1], value=L2_EPS_SQUARED)
    nisa.memset(dst=consts[0:rows, 1:2], value=norm_eps)

    # l2norm of q and k: x * rsqrt(sum(x^2) + 1e-12). q also gets 1 / sqrt(128).
    sum_sq = nl.ndarray((rows, 2), dtype=nl.float32, buffer=nl.sbuf)
    sq_q = nl.ndarray((rows, DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(
        dst=sq_q,
        op=nl.square,
        data=q_in,
        reduce_op=nl.add,
        reduce_res=sum_sq[0:rows, 0:1],
    )
    sq_k = nl.ndarray((rows, DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(
        dst=sq_k,
        op=nl.square,
        data=k_in,
        reduce_op=nl.add,
        reduce_res=sum_sq[0:rows, 1:2],
    )
    inv_norm = nl.ndarray((rows, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=inv_norm[0:rows, 0:1],
        op=nl.rsqrt,
        data=sum_sq[0:rows, 0:1],
        bias=consts[0:rows, 0:1],
        scale=1.0,
    )
    nisa.activation(
        dst=inv_norm[0:rows, 1:2],
        op=nl.rsqrt,
        data=sum_sq[0:rows, 1:2],
        bias=consts[0:rows, 0:1],
        scale=1.0,
    )
    qn = nl.ndarray((rows, DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=qn,
        data=q_in,
        op0=nl.multiply,
        operand0=inv_norm[0:rows, 0:1],
        op1=nl.multiply,
        operand1=QUERY_SCALE,
        engine=nisa.vector_engine,
    )
    kn = nl.ndarray((rows, DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=kn,
        data=k_in,
        op0=nl.multiply,
        operand0=inv_norm[0:rows, 1:2],
        engine=nisa.vector_engine,
    )

    # Gates: g = -exp(a_log) * softplus(a + dt_bias), beta = sigmoid(b).
    softplus = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=softplus,
        op=nl.softplus,
        data=scalars[0:rows, 0:1],
        bias=scalars[0:rows, 3:4],
        scale=1.0,
    )
    exp_a = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=exp_a, op=nl.exp, data=scalars[0:rows, 2:3], scale=1.0)
    g = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=g,
        data=exp_a,
        op0=nl.multiply,
        operand0=-1.0,
        op1=nl.multiply,
        operand1=softplus,
    )
    eg = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=eg, op=nl.exp, data=g, scale=1.0)
    neg_beta = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=neg_beta, op=nl.sigmoid, data=scalars[0:rows, 1:2], scale=1.0)
    nisa.tensor_scalar(
        dst=neg_beta,
        data=neg_beta,
        op0=nl.multiply,
        operand0=-1.0,
        engine=nisa.vector_engine,
    )

    # Column layouts of k and q for the Tensor Engine (k on the partitions).
    kt_ps = nl.ndarray((DIM, rows), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=kt_ps, data=kn)
    qt_ps = nl.ndarray((DIM, rows), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=qt_ps, data=qn)
    if bf16_state:
        # Columns 4h to 4h + 3 are k_hi, k_lo, q_hi and q_lo of row h.
        n_mv = 4
        kq_t = nl.ndarray((DIM, n_mv * rows), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kq_t[0:DIM, 0 : n_mv * rows : n_mv], src=kt_ps)
        nisa.tensor_tensor(
            dst=kq_t[0:DIM, 1 : n_mv * rows : n_mv],
            data1=kt_ps,
            data2=kq_t[0:DIM, 0 : n_mv * rows : n_mv],
            op=nl.subtract,
        )
        nisa.tensor_copy(dst=kq_t[0:DIM, 2 : n_mv * rows : n_mv], src=qt_ps)
        nisa.tensor_tensor(
            dst=kq_t[0:DIM, 3 : n_mv * rows : n_mv],
            data1=qt_ps,
            data2=kq_t[0:DIM, 2 : n_mv * rows : n_mv],
            op=nl.subtract,
        )
        s_mm = s_in
    else:
        # Columns 2h and 2h + 1 are k and q of row h.
        n_mv = 2
        kq_t = nl.ndarray((DIM, n_mv * rows), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kq_t[0:DIM, 0 : n_mv * rows : n_mv], src=kt_ps)
        nisa.tensor_copy(dst=kq_t[0:DIM, 1 : n_mv * rows : n_mv], src=qt_ps)
        s_mm = s_in

    # e^g on every partition: eg_bcast[m, h] = e^g of row h.
    diag = nl.ndarray((rows, rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=diag,
        data=ident[0:rows, 0:rows],
        op0=nl.multiply,
        operand0=eg,
        engine=nisa.vector_engine,
    )
    ones = nl.ndarray((rows, DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    eg_bcast_ps = nl.ndarray((DIM, rows), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=eg_bcast_ps, stationary=ones, moving=diag, accumulate=False)
    eg_bcast = nl.ndarray((DIM, rows), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=eg_bcast, src=eg_bcast_ps)

    # Phase 1: S^T k and S^T q of each row, with v on the partitions.
    p1 = nl.ndarray((DIM, n_mv * rows), dtype=nl.float32, buffer=nl.psum)
    for h in nl.static_range(rows):
        nisa.nc_matmul(
            dst=p1[0:DIM, n_mv * h : n_mv * (h + 1)],
            stationary=s_mm[0:DIM, h * DIM : (h + 1) * DIM],
            moving=kq_t[0:DIM, n_mv * h : n_mv * (h + 1)],
            accumulate=False,
        )
    kvm_t = nl.ndarray((DIM, rows), dtype=nl.float32, buffer=nl.sbuf)
    qs_t = nl.ndarray((DIM, rows), dtype=nl.float32, buffer=nl.sbuf)
    if bf16_state:
        p1_sb = nl.ndarray((DIM, n_mv * rows), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=p1_sb, src=p1)
        nisa.tensor_tensor(
            dst=kvm_t,
            data1=p1_sb[0:DIM, 0 : n_mv * rows : n_mv],
            data2=p1_sb[0:DIM, 1 : n_mv * rows : n_mv],
            op=nl.add,
        )
        nisa.tensor_tensor(
            dst=qs_t,
            data1=p1_sb[0:DIM, 2 : n_mv * rows : n_mv],
            data2=p1_sb[0:DIM, 3 : n_mv * rows : n_mv],
            op=nl.add,
        )
    else:
        nisa.tensor_copy(dst=kvm_t, src=p1[0:DIM, 0 : n_mv * rows : n_mv])
        nisa.tensor_copy(dst=qs_t, src=p1[0:DIM, 1 : n_mv * rows : n_mv])
    kvm_ps = nl.ndarray((rows, DIM), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=kvm_ps, data=kvm_t)
    qs_ps = nl.ndarray((rows, DIM), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=qs_ps, data=qs_t)

    # delta = (v - e^g * S^T k) * beta
    kvm_minus_v = nl.ndarray((rows, DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=kvm_minus_v,
        data=kvm_ps,
        op0=nl.multiply,
        operand0=eg,
        op1=nl.subtract,
        operand1=v_in,
    )
    delta = nl.ndarray((rows, DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=delta,
        data=kvm_minus_v,
        op0=nl.multiply,
        operand0=neg_beta,
        engine=nisa.vector_engine,
    )

    # Phase 2: S_new = e^g * S + k delta^T for each row. The outer product of row
    # h is kn^T (delta * one_hot(h)), so the contraction runs over the rows.
    s_new = nl.ndarray((DIM, rows * DIM), dtype=state.dtype, buffer=nl.sbuf)
    for h in nl.static_range(rows):
        delta_h = nl.ndarray((rows, DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=delta_h, op=nl.copy, data=delta, scale=ident[0:rows, h : h + 1]
        )
        outer_ps = nl.ndarray((DIM, DIM), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=outer_ps, stationary=kn, moving=delta_h, accumulate=False)
        nisa.scalar_tensor_tensor(
            dst=s_new[0:DIM, h * DIM : (h + 1) * DIM],
            data=s_in[0:DIM, h * DIM : (h + 1) * DIM],
            op0=nl.multiply,
            operand0=eg_bcast[0:DIM, h : h + 1],
            op1=nl.add,
            operand1=outer_ps,
        )
        nisa.dma_copy(
            dst=state_out[r0 + h, 0:DIM, 0:DIM],
            src=s_new[0:DIM, h * DIM : (h + 1) * DIM],
        )

    _gated_norm_output(
        out, qn, kn, delta, qs_ps, eg, consts, z_in, w_rows, rows, r0, query.dtype
    )
    return out, state_out
