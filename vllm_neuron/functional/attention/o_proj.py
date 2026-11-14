# SPDX-License-Identifier: Apache-2.0
from typing import Optional
from torch import Tensor

from nkilib.core.output_projection.output_projection_cte import (
    output_projection_cte as output_projection,
)
from nkilib.core.utils.common_types import (
    DtypeMode,
    OProjAttentionLayout,
    QuantizationType,
)

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel
from vllm_neuron.utils.hardware_config import get_logical_nc_config

P_MAX = 128
MAX_VALIDATED_H_SIZE = 16384 + 4321
MAX_VALIDATED_B_TIMES_S_SIZE = 128 * 1024
MAX_VALIDATED_N_SIZE = 17


def _can_use_kernel(
    active: Tensor,
    weight: Tensor,
    quantization_type: QuantizationType = QuantizationType.NONE,
    attention_layout: Optional[OProjAttentionLayout] = None,
) -> bool:
    """
    Check if the kernel can be used.

    Returns False when any NKI kernel constraint is violated or the tensors
    live on the CPU.
    """
    lnc_size = 2

    if not can_run_kernel(active):
        return False

    if active.dim() != 4:
        return False

    B, N, D, S = _unpack_attention_shape(
        active,
        quantization_type,
        attention_layout,
    )
    ND_weight, H = weight.shape

    if quantization_type == QuantizationType.MX:
        # MX packed weight: [ND//4, H] uint32 → effective ND = ND_weight * 4
        if N * D != ND_weight * 4:
            return False
    else:
        if N * D != ND_weight:
            return False

    if H > MAX_VALIDATED_H_SIZE:
        return False

    if B * S > MAX_VALIDATED_B_TIMES_S_SIZE:
        return False

    if N > MAX_VALIDATED_N_SIZE:
        return False

    if D > P_MAX:
        return False

    if H % lnc_size != 0:
        return False

    return True


def _unpack_attention_shape(
    active: Tensor,
    quantization_type: QuantizationType,
    attention_layout: Optional[OProjAttentionLayout],
) -> tuple[int, int, int, int]:
    """Return the attention dimensions as B, N, D, and S."""
    if attention_layout is None:
        attention_layout = (
            OProjAttentionLayout.BSNd
            if quantization_type == QuantizationType.ROW
            else OProjAttentionLayout.BNdS
        )

    if attention_layout == OProjAttentionLayout.BSNd:
        B, S, N, D = active.shape
    elif attention_layout == OProjAttentionLayout.BNSd:
        B, N, S, D = active.shape
    elif attention_layout == OProjAttentionLayout.BNdS:
        B, N, D, S = active.shape
    else:
        raise ValueError(
            f"Attention layout {attention_layout} is not supported. "
            "Use BNdS, BSNd, or BNSd."
        )

    return B, N, D, S


def _torch_o_proj_impl(
    active: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    input_scales: Optional[Tensor] = None,
    weight_scales: Optional[Tensor] = None,
    attention_layout: Optional[OProjAttentionLayout] = None,
) -> Tensor:
    """
    PyTorch implementation of output projection.

    Args:
        active: Input tensor with shape [B, N, D, S], [B, S, N, D],
            [B, N, S, D], or [B, S, N*D].
        weight: Weight tensor with shape [N*D, H]
        bias: Optional bias tensor with shape [1, H]
        quantization_type: Must be QuantizationType.NONE — quantization is
            not supported in the PyTorch fallback path.
        input_scales: Must be None — not supported in the PyTorch fallback.
        weight_scales: Must be None — not supported in the PyTorch fallback.
        attention_layout: Layout of a 4D attention tensor. None uses the
            layout that the quantization type implies.

    Returns:
        Output tensor with shape [B, S, H]

    Raises:
        AssertionError: If quantization parameters are provided.
    """
    assert quantization_type == QuantizationType.NONE, (
        f"Quantization is not supported in the PyTorch fallback path. "
        f"Got quantization_type={quantization_type}"
    )
    assert input_scales is None, (
        "input_scales must be None for the PyTorch fallback path — "
        "quantization is not supported."
    )
    assert weight_scales is None, (
        "weight_scales must be None for the PyTorch fallback path — "
        "quantization is not supported."
    )

    if active.dim() == 3:
        # [B, S, N*D] @ [N*D, H] -> [B, S, H]
        output = active @ weight
    elif active.dim() == 4:
        B, N, D, S = _unpack_attention_shape(
            active,
            quantization_type,
            attention_layout,
        )
        if attention_layout in (None, OProjAttentionLayout.BNdS):
            x = active.permute(0, 3, 1, 2).reshape(B, S, N * D)
        elif attention_layout == OProjAttentionLayout.BSNd:
            x = active.reshape(B, S, N * D)
        elif attention_layout == OProjAttentionLayout.BNSd:
            x = active.permute(0, 2, 1, 3).reshape(B, S, N * D)
        else:
            raise ValueError(f"Attention layout {attention_layout} is not supported.")
        # [B, S, N*D] @ [N*D, H] -> [B, S, H]
        output = x @ weight
    else:
        raise ValueError(f"Expected 3D or 4D input, got {active.dim()}D")

    if bias is not None:
        output = output + bias

    return output


def o_proj(
    active: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    input_scales: Optional[Tensor] = None,
    weight_scales: Optional[Tensor] = None,
    attention_layout: Optional[OProjAttentionLayout] = None,
) -> Tensor:
    """
    Output Projection API that automatically selects between NKI kernels and PyTorch fallback.

    This function checks kernel constraints and dispatches to:
    - output_projection: When all constraints are satisfied and running on Neuron
    - PyTorch implementation: When constraints are violated or not on Neuron

    Computes: output = active @ weight + bias

    Dimensions:
        B: Batch size
        N: Number of heads
        S: Sequence length
        H: Hidden dimension size
        D: Head dimension size

    Args:
        active: Input tensor with shape [B, N, D, S], [B, S, N, D],
            [B, N, S, D], or [B, S, N*D].
        weight: Weight tensor with shape [N*D, H]. Dtype depends on quantization:
            - NONE: bf16 / fp16
            - STATIC: ``torch.float8_e4m3fn`` in row-major layout
            - STATIC_MX (TRN3-only): ``torch.float8_e4m3fn`` pre-shuffled
              host-side via the loader's reshape-transpose (see
              ``weight_loaders_mxfp8.py:_mx_shuffle_o_proj``). The kernel
              consumes the 2D ``[N*D, H]`` tensor and reshapes it
              internally to ``[N*D//4, H, 4]``.
        bias: Optional bias tensor with shape [1, H]
        quantization_type: Type of quantization (NONE / STATIC / STATIC_MX).
            Quantization is only supported on the NKI kernel path; the PyTorch
            fallback will raise an AssertionError if a value other than NONE is
            supplied.
        input_scales: Per-tensor input scale, fp32. Required for STATIC and
            STATIC_MX. The o-proj CTE kernel asserts shape ``(128, 1)`` for
            both quantization types (broadcast scalar; see
            ``output_projection_cte_parameters.py``).
        weight_scales: Per-tensor weight dequant scale, fp32. Same shape
            contract as ``input_scales``: ``(128, 1)`` for STATIC and STATIC_MX.
        attention_layout: Layout of a 4D attention tensor. None uses the
            layout that the quantization type implies.

    Returns:
        Output tensor with shape [B, S, H]

    Constraints (falls back to PyTorch if violated):
        - weight must be 2D tensor with shape [N*D, H]
        - Head dimension D must be <= 128
        - H must be divisible by LNC size
        - bias (if provided) must have shape [1, H]
        - FP8 static quantization requires N or D to be even

    Notes:
        LNC sharding is performed on the weight tensor's hidden dimension.
        If active is 3D [B, S, N*D], it is reshaped to [B, N, D, S] for the
        NKI kernel using D inferred from weight shape.

    Example:
        >>> output = o_proj(
        ...     active,
        ...     weight,
        ...     attention_layout=OProjAttentionLayout.BNSd,
        ... )
    """
    # Reshape 3D [B, S, N*D] to 4D [B, N, D, S] for kernel compatibility
    if active.dim() == 3:
        B, S_len, ND = active.shape
        D = min(128, ND)  # head_dim is at most 128 (P_MAX)
        while ND % D != 0 and D > 1:
            D -= 1
        N = ND // D
        active = active.transpose(1, 2).reshape(B, N, D, S_len)
        attention_layout = OProjAttentionLayout.BNdS

    if _can_use_kernel(active, weight, quantization_type, attention_layout):
        wrapped_output_projection = wrap_nki(output_projection)

        return wrapped_output_projection[get_logical_nc_config()](
            attention=active,
            weight=weight,
            bias=bias,
            quantization_type=quantization_type,
            input_scales=input_scales,
            weight_scales=weight_scales,
            output_dtype=None,
            dtype_mode=DtypeMode.NON_OCP,
            compact_weight_scales=False,
            attention_layout=attention_layout,
        )

    return _torch_o_proj_impl(
        active,
        weight,
        bias=bias,
        quantization_type=quantization_type,
        input_scales=input_scales,
        weight_scales=weight_scales,
        attention_layout=attention_layout,
    )
