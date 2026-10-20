# SPDX-License-Identifier: Apache-2.0
"""Qwen3-32B MXFP8 decode (single-token-generation) RMSNorm + swizzle stage helpers.

The sequence-parallel composers (``vllm_neuron/_kernels/qwen3_32b_sp/``) import
``_load_gamma``, ``_compute_rmsnorm`` and ``_swizzle_to_mlp_layout`` and keep every
intermediate in SBUF: RMSNorm the ``[128, T, H // 128]`` activation, then reorder to
the ``[128, H // 512, T, 4]`` layout the in-tree decode MLP kernel and hardware
``quantize_mx`` read (``out[p, t512, tok, q]`` holds natural hidden index
``t512 * 512 + p * 4 + q``). Inputs MUST be in ``pre_shuffle_h`` order (applied at
weight load, see ``vllm_neuron/model/qwen3/weight_loaders_mxfp8.py``): a
natural-order input produces a valid-looking but silently wrong result, and the
shuffle guarantees each MX group of 8 partitions x 4 lanes covers 32 contiguous
natural hidden elements, the grouping the checkpoint's E8M0 weight scales assume.
Port notes (vs nkilib ``rmsnorm_mx_quantize_tkg``)
--------------------------------------------------
Ported from the nkilib subkernel keeping its op order. The deliberate
deviations are marked inline with ``PORT CHANGE`` comments; the numbers keep
the original port list (the others lived in the removed HBM-spill test entry
point):

1. No residual add: the composers add the residual before these stages (nkilib
   fuses it, but needs ``B*S >= 256`` for it).
2. No LNC B*S shard: the composers are LNC=1-only, and each core normalizes
   the full token shard, so the norm needs no cross-core collective.
5. One strided swizzle copy into the ``[128, H // 512, T, 4]`` MLP layout
   (nkilib issues ``n512 * 4`` per-lane copies).
"""

import nki.isa as nisa
import nki.language as nl

# Share the MXFP8 constants and the quantize helper with the decode MLP kernel.
from vllm_neuron._kernels.mlp.qwen3_32b_mxfp8_mlp_tkg import (
    _P,
    _Q_WIDTH,
)


def _load_gamma(gamma: nl.ndarray, h1: int) -> nl.ndarray:
    """Load the shuffled ``[1, H]`` gamma into SBUF as ``[128, H1]``, matching the
    ``_load_hidden`` layout so one elementwise multiply applies gamma to every token.

    Args:
        gamma: bf16 HBM ``[1, H]`` RMSNorm weight in ``pre_shuffle_h`` order.
        h1: ``H // 128``.

    Returns:
        SBUF ``[128, H1]`` gamma at the dtype of ``gamma``.

    Example:
        >>> # called inside the kernel; not a standalone doctest
    """
    gamma_sb = nl.ndarray((_P, h1), dtype=gamma.dtype, buffer=nl.sbuf, name="gamma_sb")
    nisa.dma_transpose(
        dst=gamma_sb.reshape((_P, 1, 1, h1)),
        src=gamma.reshape((h1, 1, 1, _P)),
    )
    return gamma_sb


def _compute_rmsnorm(
    hidden_sb: nl.ndarray,
    gamma_sb: nl.ndarray,
    eps: float,
    hidden_actual: int,
    t: int,
    h1: int,
) -> nl.ndarray:
    """Normalize the ``[128, T, H1]`` activation tile in the loaded layout as
    ``(hidden * gamma) * rsqrt(mean(hidden ^ 2) + eps)``, keeping the nkilib
    ``_compute_rmsnorm`` op order and intermediate dtypes (fp32 square, gamma
    applied BEFORE the rsqrt multiply, cross-partition sum on the PE) because that
    order fixes the bf16 rounding of the result; overwrites ``hidden_sb`` with
    ``hidden * gamma``, as nkilib does, to save one ``[128, T, H1]`` SBUF buffer.

    Args:
        hidden_sb: SBUF ``[128, T, H1]`` activation from :func:`_load_hidden`.
        gamma_sb: SBUF ``[128, H1]`` gamma from :func:`_load_gamma`.
        eps: Epsilon of the RMSNorm denominator.
        hidden_actual: Divisor of the mean. Equals ``H`` here.
        t: Token count.
        h1: ``H // 128``.

    Returns:
        SBUF ``[128, T, H1]`` normalized activation at the dtype of ``hidden_sb``.

    Example:
        >>> # called inside the kernel; not a standalone doctest
    """
    zero_bias = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf, name="zero_bias")
    eps_bias = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf, name="eps_bias")
    # Cross-partition sum on the PE: the all-ones nc_matmul (dst[i, tok] = sum_p ones[p, i] * reduced[p, tok]) puts the full sum on every partition.
    reduce_ones = nl.ndarray(
        (_P, _P), dtype=nl.float32, buffer=nl.sbuf, name="reduce_ones"
    )
    nisa.memset(dst=zero_bias, value=0.0)
    nisa.memset(dst=eps_bias, value=eps)
    nisa.memset(dst=reduce_ones, value=1.0)

    square = nl.ndarray((_P, t, h1), dtype=nl.float32, buffer=nl.sbuf, name="square")
    reduced = nl.ndarray((_P, t), dtype=nl.float32, buffer=nl.sbuf, name="reduced")
    total = nl.ndarray((_P, t), dtype=nl.float32, buffer=nl.psum, name="total")
    inv_rms = nl.ndarray((_P, t), dtype=nl.float32, buffer=nl.sbuf, name="inv_rms")
    normed = nl.ndarray(
        (_P, t, h1), dtype=hidden_sb.dtype, buffer=nl.sbuf, name="normed"
    )

    # Square first, because the multiply below overwrites hidden_sb.
    nisa.activation(dst=square, op=nl.square, data=hidden_sb, bias=zero_bias)

    gamma_view = gamma_sb.reshape((_P, 1, h1)).broadcast(dim=1, size=t)
    nisa.tensor_tensor(dst=hidden_sb, data1=hidden_sb, data2=gamma_view, op=nl.multiply)

    nisa.tensor_reduce(dst=reduced, op=nl.add, data=square, axis=2)
    nisa.nc_matmul(dst=total, stationary=reduce_ones, moving=reduced)

    # The mean divide and the eps add fold into the activation.
    nisa.activation(
        dst=inv_rms,
        op=nl.rsqrt,
        data=total,
        scale=(1.0 / hidden_actual),
        bias=eps_bias,
    )

    inv_rms_view = inv_rms.reshape((_P, t, 1)).broadcast(dim=2, size=h1)
    nisa.tensor_tensor(dst=normed, data1=hidden_sb, data2=inv_rms_view, op=nl.multiply)
    return normed


def _swizzle_to_mlp_layout(
    normed: nl.ndarray, n512: int, t: int, h1: int
) -> nl.ndarray:
    """Reorder the free axis of ``[128, T, H1]`` into ``[128, H // 512, T, 4]``: the
    shuffled load leaves the free axis as ``(tok, i)`` with ``i = q * n512 + t512``,
    and only the split with ``q`` OUTER and ``t512`` INNER makes the quantized output
    equal the in-tree decode MLP kernel's layout
    ``hidden[p, t512, tok, q] = x[tok, t512 * 512 + p * 4 + q]`` — the other split
    builds the same shape but a silently wrong result.

    Args:
        normed: SBUF ``[128, T, H1]`` normalized activation.
        n512: ``H // 512``.
        t: Token count.
        h1: ``H // 128``, equal to ``n512 * 4``.

    Returns:
        SBUF ``[128, n512, T, 4]`` activation at the dtype of ``normed``.

    Example:
        >>> # called inside the kernel; not a standalone doctest
    """
    swizzled = nl.ndarray(
        (_P, n512, t, _Q_WIDTH), dtype=normed.dtype, buffer=nl.sbuf, name="swizzled"
    )
    # PORT CHANGE #5: one strided read (normed holds (tok, q, t512) at free offset tok * h1 + q * n512 + t512, so dst axes (t512, tok, q) read src strides (1, h1, n512)), not the n512 * 4 per-lane copies of nkilib.
    nisa.tensor_copy(
        dst=swizzled,
        src=normed.ap([[t * h1, _P], [1, n512], [h1, t], [n512, _Q_WIDTH]]),
    )
    return swizzled
