# SPDX-License-Identifier: Apache-2.0
"""Fused RMSNorm + FP8 quantization functional API.

Wraps the ``rmsnorm_quant`` NKI kernel from nkilib with a torch-compatible
entry point and a PyTorch CPU fallback.

Supports ``QuantizationType.STATIC`` and ``QuantizationType.ROW``.
``NONE`` (and the TRN3-only MX variants) raise ``NotImplementedError``
so callers fail loudly instead of silently getting the wrong math. The
NKI kernel itself asserts ROW or STATIC, so the enum is the right
dispatch key on the vllm-neuron side.
"""

from typing import Optional

import nki
import torch
from torch import Tensor

from nkilib.core.rmsnorm.rmsnorm_quant import (
    RmsNormQuantKernelArgs,
    rmsnorm_quant_kernel,
)
from nkilib.core.rmsnorm.rmsnorm_quant_constants import RMSNormQuantConstants
from nkilib.core.utils.common_types import DtypeMode, NormType, QuantizationType

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel


# Matches ``quant_data_type_range`` in the kernel (TRN2 float8_e4m3).
# The kernel clamps static-quant output to ±240 before casting to fp8.
_FP8_E4M3_MAX = 240.0

# Partition dimension size — matches ``nl.tile_size.pmax``. The NKI kernel
# expects ``input_dequant_scale`` pre-broadcast to ``(_PMAX, 1)`` so the
# scale can be loaded directly onto the partition dimension without a
# runtime broadcast.
_PMAX = 128


# ---------------------------------------------------------------------------
# NKI entry-point (runs on NeuronCore)
# ---------------------------------------------------------------------------


@nki.jit
def _torch_compatible_rmsnorm_quant_kernel(
    hidden: Tensor,
    ln_w: Tensor,
    input_dequant_scale: Optional[Tensor] = None,
    norm_type: NormType = NormType.RMS_NORM,
    quantization_type: QuantizationType = QuantizationType.STATIC,
    eps: float = 1e-6,
    lower_bound: float = 0.0,
):
    """Torch-friendly ``@nki.jit`` wrapper around ``rmsnorm_quant_kernel``.

    The library kernel takes a ``RmsNormQuantKernelArgs`` dataclass which
    cannot be traced through ``torch.compile`` / FX cleanly. Mirroring the
    ``_maybe_build_sbm`` pattern in ``attention_decode``, we accept plain
    scalars / enums and reconstruct the dataclass inside the ``@nki.jit``
    boundary so tracing only sees ``Tensor``s and primitive constants.

    STATIC and ROW quant are supported end-to-end. The public API
    validates inputs before reaching this function.
    """
    kargs = RmsNormQuantKernelArgs(
        norm_type=norm_type,
        quantization_type=quantization_type,
        eps=eps,
        lower_bound=lower_bound,
    )
    return rmsnorm_quant_kernel(
        hidden=hidden,
        ln_w=ln_w,
        kargs=kargs,
        input_dequant_scale=input_dequant_scale,
        dtype_mode=DtypeMode.AUTO,
    )


# ---------------------------------------------------------------------------
# PyTorch fallback implementation
# ---------------------------------------------------------------------------


def _torch_rmsnorm_quant_static_impl(
    hidden: Tensor,
    ln_w: Tensor,
    input_dequant_scale: Tensor,
    eps: float,
) -> Tensor:
    """CPU reference for the STATIC quant path of ``rmsnorm_quant``.

    Mirrors the kernel's static-quant formula::

        x_norm = rsqrt(mean(x^2) + eps) * x
        x_norm *= gamma
        q = clamp(x_norm / scale, ±240).to(fp8e4m3fn)

    ``input_dequant_scale`` is the ``[128, 1]`` pre-broadcast scalar dequant
    factor described in the public-API docstring. This function reads
    position 0 and ignores the remaining 127 entries — all 128 are required
    to be equal by the public-API contract, which is not re-checked here
    (validation lives at the public entry point so the kernel and fallback
    share the same contract).
    """
    scale = input_dequant_scale.to(torch.float32).flatten()[0]

    x = hidden.to(torch.float32)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(var + eps)
    x_norm = x_norm * ln_w.to(torch.float32).reshape(-1)
    q = (x_norm / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    return q.to(torch.float8_e4m3fn)


# FP8 E4M3FN (OCP) max -- used by the ROW fallback for TRN3 workloads.
_FP8_E4M3FN_MAX = 448.0


def _torch_rmsnorm_quant_row_impl(
    hidden: Tensor,
    ln_w: Tensor,
    eps: float,
) -> Tensor:
    """CPU reference for the ROW quant path of ``rmsnorm_quant``.

    Mirrors the kernel's row-quant formula::

        x_norm = rsqrt(mean(x^2) + eps) * x
        x_norm *= gamma
        row_scale = absmax(x_norm) / FP8_MAX
        q = clamp(x_norm / row_scale, +/-FP8_MAX).to(fp8e4m3fn)
        output = [q || scale_as_4_fp8_bytes]  (packed along last dim)

    Returns:
        Packed FP8 tensor of shape ``[B, S, H+4]`` (or ``[T, H+4]`` if
        input was 2D before promotion). The last 4 elements of each row
        are the float32 dequant scale reinterpreted as 4 fp8 bytes.
    """
    x = hidden.to(torch.float32)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x * torch.rsqrt(var + eps)
    x_norm = x_norm * ln_w.to(torch.float32).reshape(-1)

    # Per-row absmax scaling
    row_absmax = x_norm.abs().amax(dim=-1, keepdim=True)
    dequant_scale = row_absmax / _FP8_E4M3FN_MAX
    dequant_scale = torch.clamp(dequant_scale, min=1e-12)
    quant_scale = 1.0 / dequant_scale

    quantized = (x_norm * quant_scale).clamp(-_FP8_E4M3FN_MAX, _FP8_E4M3FN_MAX)
    quantized = quantized.to(torch.float8_e4m3fn)

    # Pack: [data_H_bytes | scale_4_bytes] along last dim
    shape = list(hidden.shape)
    scale_bytes = (
        dequant_scale.to(torch.float32).view(torch.uint8).reshape(*shape[:-1], 4)
    )
    data_bytes = quantized.view(torch.uint8)
    packed_bytes = torch.cat([data_bytes, scale_bytes], dim=-1)
    return packed_bytes.view(torch.float8_e4m3fn)


# ---------------------------------------------------------------------------
# Kernel eligibility check
# ---------------------------------------------------------------------------


def _can_use_kernel(hidden_in: Tensor) -> bool:
    """Check whether the NKI ``rmsnorm_quant`` kernel can handle the input.

    Returns ``True`` when every kernel constraint is satisfied and the
    tensor lives on a NeuronCore device (or CPU with the NKI simulator
    enabled); ``False`` otherwise, in which case the public API falls back
    to :func:`_torch_rmsnorm_quant_static_impl`.

    ``hidden_in`` is expected to be the **post-promotion** tensor (3D) that
    the public API constructs via ``hidden.unsqueeze(0)`` for 2D inputs,
    i.e. always ``[B, S, H]``. The ``input_dequant_scale`` and ``ln_w``
    shape contracts are enforced at the public-API boundary because they
    also apply to the CPU fallback — this helper only gates kernel-specific
    constraints.

    Constraints mirror ``_validate_kernel_input`` and the ``MAX_B`` /
    ``MAX_S`` / ``MAX_H`` defaults on ``RMSNormQuantConstants`` in
    ``nkilib.core.rmsnorm.rmsnorm_quant`` /
    ``nkilib.core.rmsnorm.rmsnorm_quant_constants``.
    """
    if not can_run_kernel(hidden_in):
        return False

    # Promotion invariant: the public API always passes a 3D tensor. Guard
    # in case a future caller bypasses the promotion.
    if hidden_in.dim() < 3:
        return False

    B, S, H = hidden_in.shape[0], hidden_in.shape[1], hidden_in.shape[-1]
    if B > RMSNormQuantConstants.MAX_B:
        return False
    if S > RMSNormQuantConstants.MAX_S:
        return False
    if H > RMSNormQuantConstants.MAX_H:
        return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rmsnorm_quant(
    hidden: Tensor,
    ln_w: Tensor,
    input_dequant_scale: Optional[Tensor] = None,
    eps: float = 1e-6,
    quantization_type: QuantizationType = QuantizationType.STATIC,
) -> Tensor:
    """Fused RMSNorm + FP8 quantization.

    Computes ``quantize(rms_norm(hidden) * ln_w)`` with FP8 e4m3 output on
    Neuron via the ``rmsnorm_quant`` NKI kernel, or a PyTorch fallback on CPU.

    Args:
        hidden: Input tensor, ``[T, H]`` or ``[B, S, H]``. ``[T, H]`` is
            promoted to ``[1, T, H]`` for the kernel and squeezed back on
            return.
        ln_w: RMSNorm gamma, ``[H]`` or ``[1, H]``. Shape is validated at
            the public-API boundary so the kernel and CPU fallback share
            the same contract.
        input_dequant_scale: FP8 dequantization scale, shape ``(128, 1)``
            fp32. This is a **scalar** dequant factor pre-broadcast across
            the partition dimension — all 128 values must be identical.
            Pre-broadcasting is a performance optimization that lets the
            NKI kernel load the scale directly onto the partition
            dimension without a runtime broadcast. The CPU fallback reads
            position 0 and ignores the remaining entries. Shape and
            non-``None`` are enforced at the public-API boundary;
            value-uniformity is the caller's responsibility (not checked
            at runtime to avoid a device sync on the hot path).
        eps: Epsilon for numerical stability.
        quantization_type: ``QuantizationType.STATIC`` or
            ``QuantizationType.ROW``. Other values raise
            ``NotImplementedError``.

    Returns:
        For STATIC: FP8 quantized tensor with the same shape as ``hidden``.
        For ROW: FP8 packed tensor with shape ``[..., H+4]`` -- the last 4
            elements of each row store the float32 per-row dequant scale
            reinterpreted as 4 fp8 bytes.

    Dispatch:
        The wrapper transparently falls back to a PyTorch CPU implementation
        when :func:`_can_use_kernel` reports that kernel constraints are not
        met (CPU tensors, NKI disabled, or ``B`` / ``S`` / ``H`` exceeds the
        kernel limits from ``RMSNormQuantConstants``).

    Raises:
        NotImplementedError: for any ``quantization_type`` other than
            ``STATIC``.
        ValueError: if ``input_dequant_scale`` is ``None``,
            ``input_dequant_scale`` does not have shape ``(128, 1)``, or
            ``ln_w`` is not ``[H]`` / ``[1, H]`` with last dim equal to
            ``hidden``'s last dim.
    """
    if quantization_type not in (QuantizationType.STATIC, QuantizationType.ROW):
        raise NotImplementedError(
            "rmsnorm_quant supports QuantizationType.STATIC and ROW; "
            f"got {quantization_type.name}"
        )

    # ------------------------------------------------------------------
    # Public-API contract validation — applies to both kernel and CPU
    # fallback paths so callers get the same behavior either way.
    # ------------------------------------------------------------------

    # input_dequant_scale: required for STATIC, not used for ROW.
    if quantization_type == QuantizationType.STATIC:
        if input_dequant_scale is None:
            raise ValueError(
                "rmsnorm_quant requires input_dequant_scale for QuantizationType.STATIC"
            )
        if tuple(input_dequant_scale.shape) != (_PMAX, 1):
            raise ValueError(
                f"input_dequant_scale must have shape ({_PMAX}, 1) "
                "(scalar dequant factor broadcast across the partition "
                f"dimension); got {tuple(input_dequant_scale.shape)}"
            )

    # ln_w: rank 1 or 2, last dim == H, leading dim == 1 when 2D.
    # ``hidden.shape[-1]`` is H for both [T, H] and [B, S, H] inputs.
    H = hidden.shape[-1]
    if ln_w.dim() not in (1, 2):
        raise ValueError(
            f"ln_w must be 1D [H] or 2D [1, H]; got shape {tuple(ln_w.shape)}"
        )
    if ln_w.shape[-1] != H:
        raise ValueError(
            f"ln_w last dim must equal hidden's last dim ({H}); "
            f"got shape {tuple(ln_w.shape)}"
        )
    if ln_w.dim() == 2 and ln_w.shape[0] != 1:
        raise ValueError(
            f"ln_w leading dim must be 1 when 2D; got shape {tuple(ln_w.shape)}"
        )

    # Kernel expects at least 2D; [T, H] inputs are promoted to [1, T, H]
    # and squeezed back on output. This matches the llama3 integration and
    # the kernel's internal _collapse_shape_major_dimensions handling.
    squeeze_out = hidden.dim() == 2
    hidden_in = hidden.unsqueeze(0) if squeeze_out else hidden

    if not _can_use_kernel(hidden_in):
        if quantization_type == QuantizationType.ROW:
            out = _torch_rmsnorm_quant_row_impl(hidden_in, ln_w, eps)
        else:
            out = _torch_rmsnorm_quant_static_impl(
                hidden_in, ln_w, input_dequant_scale, eps
            )
        return out.squeeze(0) if squeeze_out else out

    # Match the process LNC degree so decode-side RMSNormQuant can run under
    # heterogeneous LNC setups (e.g. prefill LNC2 -> decode LNC1).
    wrapped = wrap_nki(_torch_compatible_rmsnorm_quant_kernel)
    out = wrapped[get_logical_nc_config()](
        hidden=hidden_in,
        ln_w=ln_w,
        input_dequant_scale=input_dequant_scale,
        norm_type=NormType.RMS_NORM,
        quantization_type=quantization_type,
        eps=eps,
        lower_bound=0.0,
    )

    return out.squeeze(0) if squeeze_out else out
