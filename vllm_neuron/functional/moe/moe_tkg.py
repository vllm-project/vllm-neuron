# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch
from torch import Tensor

from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    MoEAllToAllVStrategy,
)

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel


def moe_tkg(
    hidden_input: Tensor,
    expert_gate_up_weights: Tensor,
    expert_down_weights: Tensor,
    expert_affinities: Tensor,
    expert_index: Tensor,
    is_all_expert: bool,
    rank_id: Optional[Tensor] = None,
    expert_gate_up_bias: Optional[Tensor] = None,
    expert_down_bias: Optional[Tensor] = None,
    expert_gate_up_weights_scale: Optional[Tensor] = None,
    expert_down_weights_scale: Optional[Tensor] = None,
    hidden_input_scale: Optional[Tensor] = None,
    expert_gate_up_input_scale: Optional[Tensor] = None,
    expert_down_input_scale: Optional[Tensor] = None,
    mask_unselected_experts: bool = False,
    expert_affinities_eager: Optional[Tensor] = None,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.NO_SCALE,
    activation_fn: ActFnType = ActFnType.SiLU,
    output_dtype: Optional[torch.dtype] = None,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    output_in_sbuf: bool = False,
    is_all_expert_dynamic: bool = False,
    block_size: Optional[int] = None,
    input_dequant_scale: Optional[Tensor] = None,
    all_to_all_v_strategy: MoEAllToAllVStrategy = MoEAllToAllVStrategy.DISABLED,
) -> Tensor:
    """MoE expert MLP token generation kernel API.

    Supports MXFP4 (uint16) and STATIC_MX FP8 (uint32) weights on Trn3, and BF16.

    This functional API should be used instead of the moe_block_tkg functional API when different
    sharding schemes are used in the norm/router region of the MoE block and in the expert MLPs region.
    For example, this API should be used when norm/router are batch/sequence sharded, and All2All is used
    to dispatch tokens to EP ranks. Otherwise, moe_block_tkg should be used.

    Dimensions:
        T: Number of tokens (batch_size * seq_len)
        H: Hidden dimension
        I: Intermediate dimension
        E: Number of global experts
        E_L: Number of local experts
        K: Top-k experts per token
        I_p: I//4 if I <= 512 else 128
        H_concat: H + H/4 + E_L * 2 + 4. Number of columns when hidden_input, hidden_input_scale,
            local expert affinities, and global token index are concatenated and bitcast to a common fp8 dtype.

    Args:
        hidden_input: [T, H] or [T, H_concat] in HBM or [H0, T, H1] in SBUF, Input hidden states tensor.
            When all_to_all_v_strategy != DISABLED, input is expected to have layout [T, H_concat] and fp8 dtype,
            where H_concat = H + H/4 + E_L * 2 + 4 (hidden_quant | hidden_scale | expert_affinities | token_indices).
        expert_gate_up_weights: [E_L, H, 2, I] for bf16/fp16 or [E_L, 128, 2, ceil(H/512), I] for MxFP4,
            Fused gate and up projection weights.
        expert_down_weights: [E_L, I, H] for bf16/fp16 or [E_L, I_p, ceil(I/512), H] for MxFP4,
            Down projection weights.
        expert_affinities: [T, E], Expert routing weights/affinities. None when
            all_to_all_v_strategy != DISABLED (affinities are packed in hidden_input). For all-expert mode with
            affinity scaling, this will be sliced to [T, E_L] internally.
        expert_index: [T, K], Top-K expert indices per token. None when
            all_to_all_v_strategy != DISABLED.
        is_all_expert: If True, process all experts for all tokens; otherwise, process only selected
            top-k experts.
        rank_id: [1, 1], Rank ID tensor specifying which worker processes experts
            [E_L * rank_id, E_L * (rank_id + 1)). Required for all-expert mode with affinity scaling enabled.
        expert_gate_up_bias: [E_L, 2, I] for non-MX or [E_L, I_p, 2, ceil(I/512), 4]
            for MX, Bias for gate/up projections.
        expert_down_bias: [E_L, H], Bias for down projection.
        expert_gate_up_weights_scale: [E_L, 2, I] for FP8 row quantization, [E_L, 2, 1] for
            FP8 static quantization, or [E_L, 128/8, 2, ceil(H/512), I] for MxFP4, Quantization scales for
            gate/up weights.
        expert_down_weights_scale: [E_L, H] for FP8 row quantization, [E_L, 1] for FP8 static
            quantization, or [E_L, I_p/8, ceil(I/512), H] for MxFP4, Quantization scales for down weights.
        hidden_input_scale: [H0, H/512, T], MX quantization scale for pre-quantized
            hidden_input in SBUF. When provided with MX weights in all-expert mode, indicates that hidden_input
            is already quantized and skips internal swizzle + quantization. The hidden_input buffer must be in
            SBUF when hidden_input_scale is provided. dtype: nl.uint8.
        expert_gate_up_input_scale: [E_L, 1], FP8 dequantization scales for gate/up input.
            Used for static quantization.
        expert_down_input_scale: [E_L, 1], FP8 dequantization scales for down input. Used for
            static quantization.
        mask_unselected_experts: Whether to apply expert affinity masking based on expert_index. When
            True, affinities are masked to zero for experts not selected by each token. Only used in all-expert
            mode with affinity scaling. (default: False)
        expert_affinities_eager: [T, K], Eager expert affinities. Not used in
            all_expert mode.
        expert_affinities_scaling_mode: When to apply affinity scaling. Supported
            values: NO_SCALE, POST_SCALE. (default: NO_SCALE)
        activation_fn: Activation function type. (default: SiLU)
        output_dtype: Output tensor data type. Defaults to None; if None, uses hidden_input dtype.
        gate_clamp_upper_limit: Upper bound value to clamp gate projection results.
        gate_clamp_lower_limit: Lower bound value to clamp gate projection results.
        up_clamp_upper_limit: Upper bound value to clamp up projection results.
        up_clamp_lower_limit: Lower bound value to clamp up projection results.
        output_in_sbuf: If True, allocate output in SBUF with same shape as hidden_input. If False
            (default), allocate output in HBM with shape [T, H].
        is_all_expert_dynamic: If True, configures all-expert algorithm to use dynamic control flow.
            If False (default), utilizes all-expert algorithm without dynamic control flow. Only valid when is_all_expert=True.
        block_size: Block size for all-expert dynamic algorithm, used to group tokens for dynamic control flow. Required argument
            when is_all_expert_dynamic=True. block_size must:
            - Evenly divide T, resulting in at least 2 blocks.
            - Be divisible by 8 and less than 32, divisible by 32 and less than 128, or divisible by 128.
        input_dequant_scale: [128, 1] in SBUF, Pre-computed input FP8 dequantization
            scale for STATIC_MX mode. Passed from moe_block_tkg which computes it during the fused
            RMSNorm+quantize step. Used by the all-expert MX path to combine with per-expert weight
            dequant scales for post-matmul dequantization. Derived from expert_gate_up_input_scale.
        all_to_all_v_strategy: Input/output permutation strategy when all_to_all_v (A2A-v) is used.
            Currently only supported on Trn3 with MX weights.
            - DISABLED: Default; A2A-v is not used.
            - PRESERVE_ROW_ORDER: Output row ordering matches input row ordering. Token indices are
              appended as trailing 2 columns of output.
            - PACK_OUTPUT_ROWS: Output rows are packed, with routed tokens placed in the first N rows,
              where N is the number of routed tokens. Final T-N rows are padded with 0s. Token indices
              are appended as trailing 2 columns of output. When this strategy is used, the final 4
              elements of hidden_input must be 0 for all padded rows, and the real token indices must
              be 1-indexed.

    Returns:
        output: [T, H] MoE output tensor. When all_to_all_v_strategy != DISABLED, output is [T, H+2]
            with trailing 2 columns containing bitcast token indices.

    Example:
        >>> output = moe_tkg(
        ...     hidden_input=hidden_states,              # [T, H]
        ...     expert_gate_up_weights=gate_up_weight,   # [E_L, H, 2, I]
        ...     expert_down_weights=down_weight,         # [E_L, I, H]
        ...     expert_affinities=affinities,            # [T, E]
        ...     expert_index=indices,                    # [T, K]
        ...     is_all_expert=True,
        ...     rank_id=torch.zeros((1, 1), dtype=torch.int32, device="neuron"),
        ...     expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        ... )
    """
    can_use = _can_use_kernel(
        hidden_input=hidden_input,
        expert_down_weights=expert_down_weights,
    )

    if can_use:
        from vllm_neuron.functional.moe.moe_tkg_wrapper import moe_tkg_wrapper

        wrapped = wrap_nki(moe_tkg_wrapper)

        return wrapped[2](
            hidden_input=hidden_input,
            expert_gate_up_weights=expert_gate_up_weights,
            expert_down_weights=expert_down_weights,
            expert_affinities=expert_affinities,
            expert_index=expert_index,
            is_all_expert=is_all_expert,
            rank_id=rank_id,
            expert_gate_up_bias=expert_gate_up_bias,
            expert_down_bias=expert_down_bias,
            expert_gate_up_weights_scale=expert_gate_up_weights_scale,
            expert_down_weights_scale=expert_down_weights_scale,
            hidden_input_scale=hidden_input_scale,
            expert_gate_up_input_scale=expert_gate_up_input_scale,
            expert_down_input_scale=expert_down_input_scale,
            mask_unselected_experts=mask_unselected_experts,
            expert_affinities_eager=expert_affinities_eager,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            activation_fn=activation_fn,
            output_dtype=output_dtype,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            output_in_sbuf=output_in_sbuf,
            is_all_expert_dynamic=is_all_expert_dynamic,
            block_size=block_size,
            input_dequant_scale=input_dequant_scale,
            all_to_all_v_strategy=all_to_all_v_strategy,
        )
    else:
        raise NotImplementedError(
            "moe_tkg only supports uint16 (MXFP4), uint32 (STATIC_MX FP8), or "
            f"bfloat16 weights, but got {expert_down_weights.dtype=}"
        )


def _can_use_kernel(
    hidden_input: Tensor,
    expert_down_weights: Tensor,
) -> bool:
    """
    Check if the moe_tkg NKI kernel can be used.

    Returns False when any NKI kernel constraint is violated or the tensors
    live on the CPU without the NKI simulator enabled.

    Kernel constraints checked:
        - Must be running on Neuron device or CPU with NKI simulator
        - Must use MXFP4 (uint16), STATIC_MX FP8 (uint32), or BF16 weights
    """
    if not can_run_kernel(hidden_input):
        return False

    return expert_down_weights.dtype in (torch.uint16, torch.uint32, torch.bfloat16)
