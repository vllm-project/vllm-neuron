# SPDX-License-Identifier: Apache-2.0
"""Torch reference for the in-tree Qwen3-32B MXFP8 decode MLP kernel.

Kept in a separate module from the kernel (mirrors nkilib's ``*_torch.py``
convention) so the kernel module stays free of the reference's ``torch`` /
``numpy`` / ``nkilib`` code. The kernel test imports
``qwen3_32b_mxfp8_mlp_tkg_torch_ref`` from here and compares it to the device
kernel. ``nkilib`` (via ``quantize_to_mx``) is imported lazily inside the
reference, so importing this module does not require ``nkilib``; only calling
the reference (in the test) does.
"""

from __future__ import annotations

import torch

from vllm_neuron._kernels.mlp.qwen3_32b_mxfp8_mlp_tkg import (
    _P,
    _Q_HEIGHT,
    _Q_WIDTH,
    _RING_DEPTH_DEFAULT,
    _SBUF_QUADRANT,
    _SCALE_FOLD_ROWS,
    _SCALE_FOLD_SLOTS,
    _SCALES_PER_TILE,
    _TILE,
)


def _e8m0_to_scale(scale_u8: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 exponents into fp32 powers of two.

    Args:
        scale_u8: uint8 E8M0 exponents.

    Returns:
        fp32 scale factors ``2 ** (scale_u8 - 127)``.

    Example:
        >>> import torch
        >>> _e8m0_to_scale(torch.tensor([127], dtype=torch.uint8)).item()
        1.0
    """
    return torch.exp2(scale_u8.to(torch.float32) - 127.0)


def _gather_scattered_scale(scale_u8: torch.Tensor) -> torch.Tensor:
    """Pull the 16 real E8M0 rows out of the 128-partition scatter.

    This is the inverse of the host scatter (``_scatter_scale`` in the test). The
    kernel consumes the 128-partition layout directly. The torch reference needs
    the 16 packed rows to rebuild the per-8-partition scale.

    Args:
        scale_u8: uint8 ``[128, ...]`` pre-scattered E8M0 scale.

    Returns:
        uint8 ``[16, ...]`` scale with one row per 8 contraction partitions.

    Example:
        >>> import torch
        >>> _gather_scattered_scale(torch.zeros(128, 2, dtype=torch.uint8)).shape
        torch.Size([16, 2])
    """
    n_quadrants = _P // _SBUF_QUADRANT
    rows_per_quad = _SCALES_PER_TILE // n_quadrants
    idx = torch.tensor(
        [
            _SBUF_QUADRANT * (j // rows_per_quad) + (j % rows_per_quad)
            for j in range(_SCALES_PER_TILE)
        ]
    )
    return scale_u8[idx]


def _unfold_gate_up_scale(scale_u8: torch.Tensor, i_size: int) -> torch.Tensor:
    """Pull the 16 real E8M0 rows out of the folded gate/up scale.

    Inverse of ``fold_scale_gateup_tkg``. Row ``16q + 4m + r`` of the folded scale
    holds logical row ``4q + r`` for the I columns of 128-lane sub-block ``m``.

    Args:
        scale_u8: uint8 ``[64, n_tile, n_i512 * 128]`` folded E8M0 scale.
        i_size: Padded I extent of the weight.

    Returns:
        uint8 ``[16, n_tile, i_size]`` scale, one row per 8 contraction partitions.

    Example:
        >>> import torch
        >>> _unfold_gate_up_scale(torch.zeros(64, 2, 128, dtype=torch.uint8), 128).shape
        torch.Size([16, 2, 128])
    """
    n_quadrants = _P // _SBUF_QUADRANT
    rows_per_quad = _SCALES_PER_TILE // n_quadrants
    n_tile = scale_u8.shape[1]
    n_i512 = scale_u8.shape[2] // _P
    out = torch.zeros((_SCALES_PER_TILE, n_tile, i_size), dtype=scale_u8.dtype)
    for q in range(n_quadrants):
        for m in range(_SCALE_FOLD_SLOTS):
            src0 = q * _SCALE_FOLD_SLOTS * rows_per_quad + m * rows_per_quad
            for t in range(n_i512):
                lo = t * _TILE + m * _P
                if lo >= i_size:
                    continue
                hi = min(lo + _P, i_size)
                out[q * rows_per_quad : (q + 1) * rows_per_quad, :, lo:hi] = scale_u8[
                    src0 : src0 + rows_per_quad, :, t * _P : t * _P + (hi - lo)
                ]
    return out


def _unfold_down_scale(scale_u8: torch.Tensor, n_i512: int) -> torch.Tensor:
    """Pull the 16 real E8M0 rows out of the folded down scale.

    Inverse of ``fold_scale_down_tkg``. Row ``16q + 4(k % 4) + r`` of buffer
    ``k // 4`` holds logical row ``4q + r`` of contraction tile ``k``.

    Args:
        scale_u8: uint8 ``[64, ceil(n_i512/4), H]`` folded E8M0 scale.
        n_i512: Number of contraction tiles.

    Returns:
        uint8 ``[16, n_i512, H]`` scale, one row per 8 contraction partitions.

    Example:
        >>> import torch
        >>> _unfold_down_scale(torch.zeros(64, 2, 8, dtype=torch.uint8), 7).shape
        torch.Size([16, 7, 8])
    """
    n_quadrants = _P // _SBUF_QUADRANT
    rows_per_quad = _SCALES_PER_TILE // n_quadrants
    width = scale_u8.shape[2]
    out = torch.zeros((_SCALES_PER_TILE, n_i512, width), dtype=scale_u8.dtype)
    for q in range(n_quadrants):
        for k in range(n_i512):
            src0 = (
                q * _SCALE_FOLD_SLOTS * rows_per_quad
                + (k % _SCALE_FOLD_SLOTS) * rows_per_quad
            )
            out[q * rows_per_quad : (q + 1) * rows_per_quad, k, :] = scale_u8[
                src0 : src0 + rows_per_quad, k // _SCALE_FOLD_SLOTS, :
            ]
    return out


def _unpack_x4(weight_u32: torch.Tensor) -> torch.Tensor:
    """Unpack a uint32 x4 tensor into 4 fp32 lanes on a new trailing axis.

    Bit-exact inverse of ``x4_pack_fp8``: byte 0 is the low 8 bits (lane 0), byte
    3 the high 8 bits (lane 3). Each byte is an ``float8_e4m3fn`` value.

    Args:
        weight_u32: uint32 tensor of any shape.

    Returns:
        fp32 tensor with a trailing length-4 axis appended.

    Example:
        >>> import torch
        >>> _unpack_x4(torch.zeros(2, dtype=torch.uint32)).shape
        torch.Size([2, 4])
    """
    u = weight_u32.to(torch.int64)
    lanes = torch.stack([(u >> (8 * k)) & 0xFF for k in range(_Q_WIDTH)], dim=-1)
    return lanes.to(torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)


def _dequant_gate_up_weight(
    weight_u32: torch.Tensor, scale_u8: torch.Tensor, h_size: int
) -> torch.Tensor:
    """Reconstruct a logical ``[H, I]`` gate/up weight from its MXFP8 packing.

    ``gate_up_wl[h, i]`` equals the checkpoint weight ``W[i, h]`` (the down/gate
    projection contracts over H), so ``x[T, H] @ gate_up_wl`` gives ``[T, I]``.

    Args:
        weight_u32: uint32 ``[128, H // 512, I]`` packed MXFP8 weight.
        scale_u8: uint8 ``[128, H // 512, I]`` pre-scattered E8M0 scale.
        h_size: Hidden size.

    Returns:
        fp32 ``[H, I]`` dequantized weight in logical order.

    Example:
        >>> # used by the torch reference; not a standalone doctest
    """
    i_size = weight_u32.shape[2]
    # [128, n_h512, I, 4] with lanes (p, t512, i, q); logical h = t512*512 + p*4 + q.
    w_fp8 = _unpack_x4(weight_u32)
    # Apply the scale in the partition domain: one E8M0 per 8 partitions x 4 lanes.
    # A 64-partition scale is the folded layout; 128 is the pre-scattered one.
    if scale_u8.shape[0] == _SCALE_FOLD_ROWS:
        scale16 = _unfold_gate_up_scale(scale_u8, i_size)
    else:
        scale16 = _gather_scattered_scale(scale_u8)
    scale_full = _e8m0_to_scale(scale16).repeat_interleave(
        _Q_HEIGHT, dim=0
    )  # [128, n_h512, I]
    w_fp8 = w_fp8 * scale_full.unsqueeze(-1)
    return w_fp8.permute(1, 0, 3, 2).reshape(h_size, i_size)  # [H, I]


def _dequant_down_weight(
    weight_u32: torch.Tensor, scale_u8: torch.Tensor, i_size: int
) -> torch.Tensor:
    """Reconstruct a logical ``[I, H]`` down weight from its MXFP8 packing.

    ``down_wl[i, h]`` equals the checkpoint weight ``W_down[h, i]``, so
    ``inter[T, I] @ down_wl`` gives the ``[T, H]`` output.

    Args:
        weight_u32: uint32 ``[128, I // 512, H]`` packed MXFP8 weight.
        scale_u8: uint8 ``[128, I // 512, H]`` pre-scattered E8M0 scale.
        i_size: Padded intermediate size (a multiple of 512).

    Returns:
        fp32 ``[I, H]`` dequantized weight in logical order.

    Example:
        >>> # used by the torch reference; not a standalone doctest
    """
    h_size = weight_u32.shape[2]
    # [128, n_i512, H, 4] with lanes (p, i512, h, q); logical i = i512*512 + q*128 + p.
    w_fp8 = _unpack_x4(weight_u32)
    # The MX block is 8 partitions x 4 lanes, so the scale groups by partition,
    # NOT by contiguous logical I. Expand it in the partition domain.
    # A 64-partition scale is the folded layout; 128 is the pre-scattered one.
    if scale_u8.shape[0] == _SCALE_FOLD_ROWS:
        scale16 = _unfold_down_scale(scale_u8, weight_u32.shape[1])
    else:
        scale16 = _gather_scattered_scale(scale_u8)
    scale_full = _e8m0_to_scale(scale16).repeat_interleave(
        _Q_HEIGHT, dim=0
    )  # [128, n_i512, H]
    w_fp8 = w_fp8 * scale_full.unsqueeze(-1)
    return w_fp8.permute(1, 3, 0, 2).reshape(i_size, h_size)  # [I, H]


def qwen3_32b_mxfp8_mlp_tkg_torch_ref(
    hidden: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    gate_s: torch.Tensor,
    up_s: torch.Tensor,
    down_s: torch.Tensor,
    weight_ring_depth: int = _RING_DEPTH_DEFAULT,
    gamma: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Torch reference for :func:`qwen3_32b_mxfp8_mlp_tkg`.

    The signature matches the kernel's, as ``nkilib_testing`` asserts. The
    reference reconstructs the logical activation and dequantizes the exact
    MXFP8 weights the kernel uses, then computes the MLP in float32 and casts to
    bf16. ``weight_ring_depth`` does not change the result, so the reference
    ignores it. The reference quantizes the activation and the intermediate to
    MXFP8 as well, to match the kernel's in-kernel quantize.

    Args:
        hidden: bf16 ``[128, H // 512, T_pad, 4]`` activation.
        gate_w: uint32 ``[128, H // 512, I]`` MXFP8 gate weight.
        up_w: uint32 ``[128, H // 512, I]`` MXFP8 up weight.
        down_w: uint32 ``[128, I // 512, H]`` MXFP8 down weight.
        gate_s: uint8 ``[128, H // 512, I]`` pre-scattered E8M0 gate scale.
        up_s: uint8 ``[128, H // 512, I]`` pre-scattered E8M0 up scale.
        down_s: uint8 ``[128, I // 512, H]`` pre-scattered E8M0 down scale.
        weight_ring_depth: Ignored (does not change the result).
        gamma: Optional bf16 RMSNorm weight in the activation's x4 H layout.
        eps: RMSNorm epsilon when ``gamma`` is given.

    Returns:
        Dict with key ``"out"`` holding a bf16 ``[T, H]`` tensor.

    Example:
        >>> # driven by the test framework with matching kernel inputs
    """
    import numpy as np
    import nki.language as nl
    from nkilib.core.utils.mx_torch_common import quantize_to_mx

    p, n_h512, t_pad, q = hidden.shape
    h_size = n_h512 * _TILE
    i_size = gate_w.shape[2]
    # gate/up pads its I axis to a multiple of 128, while the down weight keeps
    # whole 512-tiles, so the two extents differ when the last tile is partial.
    n_i512 = -(-i_size // _TILE)
    i_down = n_i512 * _TILE

    def _mx_round_trip(x: torch.Tensor) -> torch.Tensor:
        """Quantize to MXFP8 and back, mirroring the in-kernel quantize_mx.

        ``x`` is ``[128, N]`` with 4 consecutive free elements forming one x4.
        """
        packed_np, scale_np = quantize_to_mx(x.numpy(), nl.float8_e4m3fn_x4)
        packed = torch.from_numpy(packed_np.view(np.uint32).astype(np.int64)).to(
            torch.uint32
        )
        fp8 = _unpack_x4(packed)  # [128, N//4, 4]
        scale = _e8m0_to_scale(
            torch.from_numpy(scale_np.astype(np.uint8))
        )  # [16, N//4]
        scale = scale.repeat_interleave(_Q_HEIGHT, dim=0)  # [128, N//4]
        return fp8 * scale.unsqueeze(-1)  # [128, N//4, 4]

    # Reconstruct the logical activation x[T, H] from the shuffled layout.
    x_shuf = hidden.to(torch.float32)  # [128, n_h512, t_pad, 4], x[p, t512, t, q]

    # The fused kernel normalizes the raw bf16 activation before MX quantization,
    # so every per-32 scale derives from normalized values. Gamma shares the
    # activation's (t512, p, q) layout and must be reconstructed in the same order.
    if gamma is not None:
        gamma_logical = gamma.to(torch.float32).permute(1, 0, 2).reshape(h_size)
        x_raw = x_shuf.permute(2, 1, 0, 3).reshape(t_pad, h_size)
        ms = x_raw.pow(2).mean(dim=1, keepdim=True)
        x_raw = x_raw * torch.rsqrt(ms + eps) * gamma_logical.unsqueeze(0)
        # Round to bf16, then widen back to fp32. The two casts are not redundant.
        # _rmsnorm_full computes in fp32 but writes a bf16 SBUF buffer, so the
        # kernel pays this rounding before quantize_mx reads the values. A direct
        # fp32 path would keep more mantissa than the kernel has, and the
        # reference would then be more precise than the code it checks.
        x_shuf = (
            x_raw.to(torch.bfloat16)
            .to(torch.float32)
            .reshape(t_pad, n_h512, p, _Q_WIDTH)
            .permute(2, 1, 0, 3)
        )

    # Quantize/dequantize to match the kernel.
    x_flat = x_shuf.reshape(p, n_h512 * t_pad * _Q_WIDTH)
    x_deq = _mx_round_trip(x_flat).reshape(p, n_h512, t_pad, _Q_WIDTH)
    # h = t512 * 512 + p * 4 + q  ->  [t_pad, H].
    x_logical = x_deq.permute(2, 1, 0, 3).reshape(t_pad, h_size)  # [t_pad, H]

    gate_wl = _dequant_gate_up_weight(gate_w, gate_s, h_size)  # [H, I]
    up_wl = _dequant_gate_up_weight(up_w, up_s, h_size)  # [H, I]
    down_wl = _dequant_down_weight(down_w, down_s, i_down)  # [I_down, H]

    gate = x_logical @ gate_wl  # [t_pad, I]
    up = x_logical @ up_wl  # [t_pad, I]
    inter = torch.nn.functional.silu(gate) * up  # [t_pad, I]
    # Zero the lanes of a partial last tile, as the kernel memsets them. Those
    # lanes must be zero BEFORE the quantize, because the 4 lanes of one x4
    # element share one E8M0 scale.
    if i_down != i_size:
        inter = torch.nn.functional.pad(inter, (0, i_down - i_size))

    # Quantize the intermediate the same way the kernel does. Reshape into the
    # [128, n_i512, t_pad, 4] layout the down projection uses: i = i512*512 + q*128 + p.
    inter_shuf = inter.reshape(t_pad, n_i512, _Q_WIDTH, _P).permute(
        3, 1, 0, 2
    )  # [128, n_i512, t_pad, 4]
    inter_flat = inter_shuf.reshape(_P, n_i512 * t_pad * _Q_WIDTH)
    inter_deq = _mx_round_trip(inter_flat).reshape(_P, n_i512, t_pad, _Q_WIDTH)
    inter_logical = inter_deq.permute(2, 1, 3, 0).reshape(
        t_pad, i_down
    )  # [t_pad, I_down]

    out = inter_logical @ down_wl  # [t_pad, H]
    return {"out": out.to(torch.bfloat16)}
