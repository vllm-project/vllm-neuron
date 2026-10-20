# SPDX-License-Identifier: Apache-2.0
import nki
import torch
import nki.language as nl

from typing import Any, Optional
from torch import Tensor

from nkilib.core.moe.moe_cte.moe_cte import (
    MoECTESpec,
    MoECTEImplementation,
    SkipMode,
    ShardOnBlockConfig,
    ShardOnIConfig,
    QuantizationConfig,
    ExpertAffinityScaleMode,
    ActFnType,
    BlockShardStrategy,
)
from nkilib.core.utils.common_types import QuantizationType

from nkilib.core.moe.moe_cte.moe_cte import moe_cte as nki_moe_cte_kernel

# STATIC_MX kwargs are not forwarded by the public moe_cte dispatcher, so the
# STATIC_MX branch calls this inner kernel directly.
from nkilib.core.moe.moe_cte.bwmm_shard_on_block_mx import (
    bwmm_shard_on_block_mx as nki_bwmm_shard_on_block_mx_kernel,
)

from vllm_neuron import envs
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel


def moe_cte(
    hidden_states: Tensor,
    expert_affinities_masked: Optional[Tensor],
    gate_up_proj_weight: Tensor,
    down_proj_weight: Tensor,
    token_position_to_id: Tensor,
    block_to_expert: Tensor,
    block_size: int,
    implementation: MoECTEImplementation,
    # shard_on_block parameters
    n_block_per_iter: int = 1,
    block_sharding_strategy: BlockShardStrategy = BlockShardStrategy.PING_PONG,
    # shard_on_block_mx parameters (MxFP4/MxFP8 quantization)
    n_static_blocks: int = -1,
    n_dynamic_blocks: int = -1,
    # Routing shape for the kernel's auto-computed static/dynamic block split
    # (shard_on_block_mx only; consumed when n_dynamic_blocks is out of range).
    top_k: int = 1,
    ep_degree: int = 1,
    # EP-rank ([1, 1] int32 TENSOR) for the packed-affinity read; the kernel derives the
    # global expert base on-device (ep_rank * num_local_experts). None on non-packed / decode
    # paths, which the kernel treats as EP-rank 0.
    ep_rank: Optional[Tensor] = None,
    # shard_on_i and shard_on_i_dropping parameters
    checkpoint_activation: bool = False,
    expert_affinity_multiply_on_I: bool = False,
    # shard_on_i_hybrid parameters
    num_static_block: Optional[int] = None,
    conditions: Optional[Tensor] = None,
    gate_and_up_proj_bias: Optional[Tensor] = None,
    down_proj_bias: Optional[Tensor] = None,
    # Quantization scales
    gate_up_proj_scale: Optional[Tensor] = None,
    down_proj_scale: Optional[Tensor] = None,
    gate_up_activations_T: Optional[Tensor] = None,
    down_activations: Optional[Tensor] = None,
    activation_function: ActFnType = ActFnType.SiLU,
    # DMA skipping
    skip_token: bool = False,
    skip_weight: bool = False,
    compute_dtype: Any = nl.bfloat16,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    # STATIC_MX (TRN3, shard_on_block_mx only): dequant runs outside the matmul
    # (matmul uses all-127 dummy MX scales). Weight scales go in
    # gate_up_proj_scale / down_proj_scale; input scales in gate_up_in_scale /
    # down_in_scale. The gate_w_scale / up_w_scale / down_w_scale kwargs below
    # are unused by this dispatcher (kept for signature compatibility).
    quantization_type: QuantizationType = QuantizationType.NONE,
    gate_w_scale: Optional[Tensor] = None,
    up_w_scale: Optional[Tensor] = None,
    down_w_scale: Optional[Tensor] = None,
    gate_up_in_scale: Optional[Tensor] = None,
    down_in_scale: Optional[Tensor] = None,
    use_packed_scales: bool = False,
):
    """
    MoE CTE Projection API that automatically selects between NKI kernel and PyTorch fallback.

    This function checks kernel constraints and dispatches to:
    - NKI MoE CTE kernel: When all constraints are satisfied and running on Neuron
    - PyTorch implementation: When on CPU or constraints are violated

    This kernel implements blockwise matrix multiplication for Mixture-of-Experts layers,
    processing tokens through expert-specific gate, up, and down projections:
    output[token] += act_fn(hidden @ W_gate) * (hidden @ W_up) @ W_down * affinity

    Dimensions:
        T: Total number of input tokens
        H: Hidden dimension size
        B: Block size (tokens per block)
        E: Number of experts (E_local when using expert parallelism)
        N: Total number of blocks (N * B >= T, may include padding blocks)
        I_TP: Intermediate size divided by tensor parallelism degree

    Args:
        hidden_states: Input token embeddings with shape [T+1, H] or [T, H].
            The T+1 variant reserves the last row as a padding sink.
        expert_affinities_masked: Expert routing weights with shape [T*E, 1].
            Entry [t*E + e, 0] is the affinity of token t for expert e.
            Zero for non-selected experts. Pass ``None`` for the packed-affinity path
            (shard_on_block_mx), where the kernel reads the affinities from the packed
            ``hidden_states`` tail instead of a separate tensor.
        gate_up_proj_weight: Combined gate and up projection weights. Shape depends on implementation:
            - shard_on_block, shard_on_i variants: [E, H, 2, I_TP]
            - shard_on_block_mx, shard_on_i_mx variants: [E, 128, 2, H//512, I_TP]
              Pre-quantized MXFP4 tiled format, dtype uint16. The kernel reinterprets the
              dtype to the appropriate NKI MX type (e.g., float4_e2m1fn_x4).
        down_proj_weight: Down projection weights. Shape depends on implementation:
            - shard_on_block, shard_on_i variants: [E, I_TP, H]
            - shard_on_block_mx, shard_on_i_mx variants: [E, p_I, num_I_tiles, H]
              Pre-quantized MXFP4 tiled format, dtype uint16, where p_I and num_I_tiles
              are determined by the I_TP tiling configuration.
        token_position_to_id: Mapping from block positions to token IDs with shape [N*B],
            dtype int32. Padding slots should map to index T (or -1 when skip_token=True).
        block_to_expert: Expert assignment per block with shape [N, 1] or [N], dtype int32.
        block_size: Number of tokens per block (B).
        implementation: Which kernel implementation variant to use:
            - shard_on_block: Block-level sharding (TRN2)
            - shard_on_i: Intermediate dimension sharding (TRN2)
            - shard_on_i_hybrid: Hybrid static/dynamic I-sharding (TRN2)
            - shard_on_i_dropping: I-sharding for token-dropping layers (TRN2)
            - shard_on_block_mx: Block sharding with MxFP4/MxFP8 quantization (TRN3)
            - shard_on_i_mx: I-sharding with MxFP4/MxFP8 quantization (TRN3)
            - shard_on_i_mx_hybrid: Hybrid I-sharding with MxFP4/MxFP8 (TRN3)
        n_block_per_iter: Blocks processed per iteration (shard_on_block only). Default: 1
        block_sharding_strategy: Block distribution strategy across cores:
            - PING_PONG: Alternates blocks between shards (0, 1, 0, 1, ...)
            - HI_LO: First half to shard 0, second half to shard 1
        n_static_blocks: Number of blocks for static loop in shard_on_block_mx.
            Set to -1 to auto-compute. Default: -1
        n_dynamic_blocks: Maximum dynamic-loop blocks for shard_on_block_mx.
            Default: -1 → let the kernel auto-compute the best-case static/dynamic
            split (any out-of-range value, <0 or >N, triggers the kernel's
            auto-compute flow).
        top_k: Top-K experts per token (shard_on_block_mx only). Feeds the kernel's
            best-case static-block estimate
            ``max(1, ceil(ceil(T*top_k / ep_degree) / block_size))`` when
            n_dynamic_blocks is out of range. Ignored otherwise. Default: 1
        ep_degree: Expert-parallelism degree the routing is sharded across
            (shard_on_block_mx only). Used with top_k for the static-block
            estimate above. Default: 1
        checkpoint_activation: Save gate/up activations for backward pass
            (shard_on_i, shard_on_i_dropping). Default: False
        expert_affinity_multiply_on_I: Apply expert affinity scaling on intermediate
            dimension instead of hidden dimension (shard_on_i, shard_on_i_dropping). Default: False
        num_static_block: Non-padded block count for shard_on_i_hybrid.
            None means auto-compute as (N - E). Default: None
        conditions: Block padding indicators with shape [N+2] for hybrid/dynamic-loop
            implementations. Required by shard_on_i_hybrid and shard_on_block_mx when
            dynamic loop is desired. Should be padded with 2 trailing zeros. Default: None
        gate_and_up_proj_bias: Optional bias for gate/up projections. Shape depends on implementation:
            - shard_on_block, shard_on_i variants: [E, 2, I_TP]
            - shard_on_block_mx, shard_on_i_mx variants: [E, I_TP_block_size//4, 2, num_I_TP_blocks, 4]
              where num_I_TP_blocks = ceil(I_TP / 512), I_TP_block_size = I_TP // num_I_TP_blocks.
              For Swish activation, up_bias should have +1 pre-added (i.e., store up_bias + 1).
            Must be provided together with down_proj_bias (both or neither). Default: None
        down_proj_bias: Optional bias for down projection with shape [E, H].
            Must be provided together with gate_and_up_proj_bias (both or neither). Default: None
        gate_up_proj_scale: Dequantization scales for gate/up weights. Shape depends on implementation:
            - shard_on_block: [E, 1, 2*I_TP] (optional, for FP8)
            - shard_on_i: [E, 1, 2*I_TP] (optional, for FP8). The flat per-channel
              scales are passed as-is; the kernel reshapes them to [E, 2, I_TP]
              internally (added by CR-268896139).
            - shard_on_block_mx: [E, 16, 2, H//512, I_TP], dtype uint8.
            Required for all MX quantized implementations. Must be provided together
            with down_proj_scale. Default: None
        down_proj_scale: Dequantization scales for down weights. Shape depends on implementation:
            - shard_on_block: [E, 1, H] (optional, for FP8)
            - shard_on_i: [E, 1, H] (optional, for FP8).
            - shard_on_block_mx: [E, q_blocks_per_I_tile, num_I_tiles, H], dtype uint8.
            Required for all MX quantized implementations. Must be provided together
            with gate_up_proj_scale. Default: None
        gate_up_activations_T: Optional storage for gate/up activations. Default: None
        down_activations: Optional storage for down-projection activations with
            shape [N, B, H]. Default: None
        activation_function: Activation function type (SiLU, GELU, Swish, etc.). Default: SiLU
        skip_token: Enable DMA skip for out-of-bounds token indices. When True, padding
            token IDs should be set to -1 instead of T. Default: False
        skip_weight: Enable DMA skip for weight loads (weight reuse optimization). Default: False
        compute_dtype: Data type for internal computations. Default: nl.bfloat16
        is_tensor_update_accumulating: Enable accumulation for topK > 1 scenarios.
            When True, output has extra shard dimension for cross-core reduction. Default: True
        expert_affinities_scaling_mode: When to apply expert affinity weights:
            - POST_SCALE: After down projection
            - PRE_SCALE: Before gate/up projection (scales hidden_states)
            - PRE_SCALE_DELAYED: After gate/up matmul but before activation
        gate_clamp_upper_limit: Upper clamp for gate projection output. Default: None
        gate_clamp_lower_limit: Lower clamp for gate projection output. Default: None
        up_clamp_upper_limit: Upper clamp for up projection output. Default: None
        up_clamp_lower_limit: Lower clamp for up projection output. Default: None

    Returns:
        Output tensor with shape [T, H].

    Kernel Constraints (falls back to PyTorch if on CPU for non-MX implementations):
        - hidden_states must be 2D with shape [T, H] or [T+1, H]
        - I_TP must be divisible by 16 (vector DGE partition alignment)
        - block_size must be a multiple of 128
        - len(token_position_to_id) must be divisible by block_size
        - len(block_to_expert) must equal N = len(token_position_to_id) // block_size
        - gate_and_up_proj_bias and down_proj_bias must both be provided or neither
        - gate_up_proj_scale and down_proj_scale must both be provided or neither
        - For shard_on_block: only PING_PONG strategy is currently supported
        - For shard_on_block_mx: H must be in [512, 8192] and divisible by 512,
          is_tensor_update_accumulating must be True, requires exactly 2 NeuronCores,
          weights must be pre-quantized in MXFP4/MXFP8 tiled format
        - For shard_on_block_mx with conditions: conditions.shape[0] must equal N + 2
        - For shard_on_block_mx: n_static_blocks > 0 must be < N
        - For shard_on_i_hybrid: conditions tensor is required
        - For MX quantized implementations (shard_on_block_mx, shard_on_i_mx,
          shard_on_i_mx_hybrid): scales must be provided and CPU is not supported

    Notes:
        - shard_on_block, shard_on_i, shard_on_i_hybrid, shard_on_i_dropping target TRN2
        - shard_on_block_mx, shard_on_i_mx, shard_on_i_mx_hybrid target TRN3
        - MX implementations expect weights already in quantized tiled format (e.g., uint16).
          The kernel reinterprets the dtype (e.g., uint16 -> float4_e2m1fn_x4) but does NOT
          perform quantization.
        - For Swish activation with MX implementations, the up projection bias should have
          +1 pre-added (i.e., store bias + 1 instead of just bias).

    Usage:
        **Example — MXFP4 quantized prefill**

        >>> # ── Routing ──
        >>> # hidden_states: [T, H]  (after layernorm)
        >>> expert_affinities = router(
        ...     hidden_states=hidden_states,
        ...     router_weights=router_weight.T,       # [H, E]
        ...     top_k=num_experts_per_tok,
        ...     router_bias=router_bias,
        ...     activation="softmax",
        ...     computation_dtype=torch.float32,
        ... )  # [T, E]
        >>>
        >>> # ── Blockwise mapping ──
        >>> (
        ...     expert_affinities_masked,   # [T*E_local, 1]
        ...     token_position_to_id,       # [N*block_size]
        ...     block_to_expert,            # [N]
        ...     conditions,                 # [N]
        ... ) = build_blockwise_mapping(
        ...     expert_affinities=expert_affinities,
        ...     num_local_experts=num_local_experts,
        ...     num_experts_per_token=num_experts_per_tok,
        ...     block_size=block_size,
        ... )
        >>>
        >>> # ── Call moe_cte with blockwise args ──
        >>> output = moe_cte(
        ...     implementation=MoECTEImplementation.shard_on_block_mx,
        ...     hidden_states=hidden_states,
        ...     expert_affinities_masked=expert_affinities_masked,
        ...     gate_up_proj_weight=gate_up_proj_weight,   # [E_local, 128, 2, H//512, I_TP] uint16
        ...     down_proj_weight=down_proj_weight,         # [E_local, p_I, num_I_tiles, H] uint16
        ...     gate_and_up_proj_bias=gate_up_proj_bias_reshaped,
        ...     down_proj_bias=down_proj_bias,             # [E_local, H]
        ...     gate_up_proj_scale=gate_up_proj_scale,     # [E_local, 16, 2, H//512, I_TP] uint8
        ...     down_proj_scale=down_proj_scale,           # [E_local, q, num_I_tiles, H] uint8
        ...     token_position_to_id=token_position_to_id.to(dtype=torch.int32),
        ...     block_to_expert=block_to_expert.to(dtype=torch.int32),
        ...     block_size=block_size,
        ...     conditions=conditions,
        ...     activation_function=ActFnType.Swish,
        ...     expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        ...     skip_token=True,
        ...     is_tensor_update_accumulating=True,
        ...     compute_dtype=nl.bfloat16,
        ...     gate_clamp_upper_limit=8.0,
        ...     up_clamp_upper_limit=9.0,
        ...     up_clamp_lower_limit=-7.0,
        ... )  # output: [T, H]
    """

    # ── STATIC_MX direct-call bypass ──
    # STATIC_MX has no torch fallback — the weights are uint32-packed FP8 in a
    # kernel-specific layout with per-expert fp32 scalar scales that
    # ``_torch_moe_impl`` cannot consume — so any condition that disables the
    # kernel is fatal here.
    if quantization_type == QuantizationType.STATIC_MX:
        if implementation != MoECTEImplementation.shard_on_block_mx:
            raise RuntimeError(
                "STATIC_MX is only supported on shard_on_block_mx, got "
                f"{implementation}"
            )
        if not can_run_kernel(hidden_states):
            # ``can_run_kernel`` covers both CPU device and the
            # ``VLLM_NEURON_DISABLE_NKI_KERNELS=1`` escape hatch.
            raise RuntimeError(
                "STATIC_MX MoE requires NKI kernels on a Neuron device "
                "(no torch fallback). Got device "
                f"{hidden_states.device}; VLLM_NEURON_DISABLE_NKI_KERNELS="
                f"{int(envs.VLLM_NEURON_DISABLE_NKI_KERNELS)}."
            )
        wrapped_kernel = wrap_nki(_torch_compatible_moe_cte_static_mx_kernel)
        output = wrapped_kernel[2](
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=gate_up_proj_weight,
            down_proj_weight=down_proj_weight,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
            block_size=block_size,
            n_static_blocks=n_static_blocks,
            n_dynamic_blocks=n_dynamic_blocks,
            top_k=top_k,
            ep_degree=ep_degree,
            conditions=conditions,
            gate_and_up_proj_bias=gate_and_up_proj_bias,
            down_proj_bias=down_proj_bias,
            gate_up_activations_T=gate_up_activations_T,
            down_activations=down_activations,
            activation_function=activation_function,
            skip_token=skip_token,
            skip_weight=skip_weight,
            compute_dtype=compute_dtype,
            is_tensor_update_accumulating=is_tensor_update_accumulating,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            use_packed_scales=use_packed_scales,
            gate_up_proj_scale=gate_up_proj_scale,
            down_proj_scale=down_proj_scale,
            gate_up_in_scale=gate_up_in_scale,
            down_in_scale=down_in_scale,
        )
        # ``bwmm_shard_on_block_mx`` returns a [2, T, H] accumulating buffer
        # under ``is_tensor_update_accumulating=True``; trim to [T, H] to
        # match the standard return.
        if is_tensor_update_accumulating:
            T, H = hidden_states.shape[0], hidden_states.shape[-1]
            output = output[0, :T, :H]
        return output

    if _can_use_moe_cte_kernel(
        implementation,
        hidden_states,
        gate_up_proj_weight,
        gate_up_proj_scale,
        down_proj_scale,
    ):
        wrapped_kernel = wrap_nki(_torch_compatible_moe_cte_kernel)
        output = wrapped_kernel[2](
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=gate_up_proj_weight,
            down_proj_weight=down_proj_weight,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
            block_size=block_size,
            implementation=implementation,
            n_block_per_iter=n_block_per_iter,
            block_sharding_strategy=block_sharding_strategy,
            n_static_blocks=n_static_blocks,
            n_dynamic_blocks=n_dynamic_blocks,
            top_k=top_k,
            ep_degree=ep_degree,
            ep_rank=ep_rank,
            checkpoint_activation=checkpoint_activation,
            expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
            num_static_block=num_static_block,
            conditions=conditions,
            gate_and_up_proj_bias=gate_and_up_proj_bias,
            down_proj_bias=down_proj_bias,
            gate_up_proj_scale=gate_up_proj_scale,
            down_proj_scale=down_proj_scale,
            gate_up_activations_T=gate_up_activations_T,
            down_activations=down_activations,
            activation_function=activation_function,
            skip_token=skip_token,
            skip_weight=skip_weight,
            compute_dtype=compute_dtype,
            is_tensor_update_accumulating=is_tensor_update_accumulating,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            use_packed_scales=use_packed_scales,
        )

        if is_tensor_update_accumulating and implementation in [
            MoECTEImplementation.shard_on_block_mx,
            MoECTEImplementation.shard_on_i_mx,
            MoECTEImplementation.shard_on_i_mx_hybrid,
        ]:
            output = output[0, ...]
        elif (
            is_tensor_update_accumulating
            and implementation == MoECTEImplementation.shard_on_block
        ):
            # Non-MX shard-on-block kernel allocates output as [T, 2, H+E] where E is
            # the number of experts. The extra E columns hold fused expert affinities
            # in shared HBM (compiler cannot return a partial shared_hbm tensor).
            # Trim back to the original hidden dim H. The shard_on_i kernels return
            # a plain [T, H] tensor and need no trimming.
            H = hidden_states.shape[-1]
            output = output[:, 0, :H]

        return output

    else:
        return _torch_moe_impl(
            hidden_states=hidden_states,
            expert_affinities=expert_affinities_masked,
            gate_up_proj_weight=gate_up_proj_weight,
            down_proj_weight=down_proj_weight,
            gate_and_up_proj_bias=gate_and_up_proj_bias,
            down_proj_bias=down_proj_bias,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            activation_function=activation_function,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        )


def _apply_activation(gate: Tensor, activation_function: ActFnType) -> Tensor:
    """
    Apply the gated activation function to the gate projection output.

    Args:
        gate: Gate projection output with shape [..., I_TP].
        activation_function: Which activation to use.

    Returns:
        Activated gate tensor with the same shape as input.

    Raises:
        ValueError: If activation_function is not a supported type.
    """
    if activation_function == ActFnType.SiLU:
        return torch.nn.functional.silu(gate)
    elif activation_function == ActFnType.GELU:
        return torch.nn.functional.gelu(gate)
    elif activation_function == ActFnType.Swish:
        swiglu_alpha = 1.702  # This value is baked into the NKI ISA
        return gate * torch.sigmoid(gate * swiglu_alpha)
    else:
        raise ValueError(f"Unsupported activation function: {activation_function}")


def _torch_moe_impl(
    hidden_states: Tensor,
    expert_affinities: Tensor,
    gate_up_proj_weight: Tensor,
    down_proj_weight: Tensor,
    gate_and_up_proj_bias: Optional[Tensor] = None,
    down_proj_bias: Optional[Tensor] = None,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    activation_function: ActFnType = ActFnType.SiLU,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
) -> Tensor:
    """
    PyTorch fallback implementation of MoE computation.

    Supports three expert affinity scaling modes that control where the routing
    weight (affinity) is multiplied into the computation:

    POST_SCALE (default):
        gate = hidden @ W_gate
        up   = hidden @ W_up
        intermediate = act(gate) * up
        output += (intermediate @ W_down) * affinity
        Affinity is applied after the down projection on dimension H.

    PRE_SCALE:
        scaled_hidden = hidden * affinity
        gate = scaled_hidden @ W_gate
        up   = scaled_hidden @ W_up
        intermediate = act(gate) * up
        output += intermediate @ W_down
        Affinity is applied before any projection. NOTE: this is NOT mathematically
        equivalent to POST_SCALE because the nonlinear activation breaks the
        linearity: act(a * x) != a * act(x).

    PRE_SCALE_DELAYED:
        gate = hidden @ W_gate
        up   = hidden @ W_up
        intermediate = affinity * act(gate) * up
        output += intermediate @ W_down
        Affinity is applied after activation but before the down projection, on
        dimension I_TP. This IS mathematically equivalent to POST_SCALE because
        the down projection is linear: (a * x) @ W = a * (x @ W).

    Args:
        hidden_states:          [T, H] or [T+1, H]
        expert_affinities:      [T*E, 1]
        gate_up_proj_weight:    [E, H, 2, I_TP]
        down_proj_weight:       [E, I_TP, H]
        gate_and_up_proj_bias:  [E, 2, I_TP]  (optional)
        down_proj_bias:         [E, 1, H]      (optional)
        gate_clamp_upper_limit: Upper clamp for gate projection (optional)
        gate_clamp_lower_limit: Lower clamp for gate projection (optional)
        up_clamp_upper_limit:   Upper clamp for up projection (optional)
        up_clamp_lower_limit:   Lower clamp for up projection (optional)
        activation_function:    Activation type (SiLU, GELU). Default: SiLU
        expert_affinities_scaling_mode: Where to apply the expert affinity.
            One of POST_SCALE, PRE_SCALE, PRE_SCALE_DELAYED. Default: POST_SCALE

    Returns:
        output: [T, H]
    """
    num_experts = gate_up_proj_weight.shape[0]
    hidden_size = hidden_states.shape[-1]
    dtype = hidden_states.dtype
    device = hidden_states.device

    # [T*E, 1] -> [T, E]
    num_tokens = expert_affinities.shape[0] // num_experts
    expert_affinities = expert_affinities.reshape(num_tokens, num_experts)

    hidden_states = hidden_states[:num_tokens]

    # [E, H, 2, I] -> gate [E, H, I], up [E, H, I]
    gate_w = gate_up_proj_weight[:, :, 0, :]  # [E, H, I]
    up_w = gate_up_proj_weight[:, :, 1, :]  # [E, H, I]

    # ── PRE_SCALE: multiply affinity into hidden_states before projections ──
    if expert_affinities_scaling_mode == ExpertAffinityScaleMode.PRE_SCALE:
        # expert_affinities: [T, E] -> [E, T, 1] for broadcasting with [T, H]
        affinity_etx = expert_affinities.T.unsqueeze(-1)  # [E, T, 1]
        # Cast back after the multiply: fp32 router affinities would otherwise
        # promote the activations and the einsum against low-precision weights
        # raises a dtype-mismatch RuntimeError.
        scaled_hidden = (hidden_states.unsqueeze(0) * affinity_etx).to(
            dtype
        )  # [E, T, H]
        gate = torch.einsum("eth,ehi->eti", scaled_hidden, gate_w)  # [E, T, I]
        up = torch.einsum("eth,ehi->eti", scaled_hidden, up_w)  # [E, T, I]
    else:
        # POST_SCALE and PRE_SCALE_DELAYED both project the original hidden_states
        gate = torch.einsum("th,ehi->eti", hidden_states, gate_w)  # [E, T, I]
        up = torch.einsum("th,ehi->eti", hidden_states, up_w)  # [E, T, I]

    # ── Bias ──
    if gate_and_up_proj_bias is not None:
        gate = gate + gate_and_up_proj_bias[:, 0, :].unsqueeze(1)  # [E, 1, I]
        up = up + gate_and_up_proj_bias[:, 1, :].unsqueeze(1)

    # ── Clamping ──
    if gate_clamp_lower_limit is not None or gate_clamp_upper_limit is not None:
        gate = gate.clamp(min=gate_clamp_lower_limit, max=gate_clamp_upper_limit)

    if up_clamp_lower_limit is not None or up_clamp_upper_limit is not None:
        up = up.clamp(min=up_clamp_lower_limit, max=up_clamp_upper_limit)

    # ── Activation + element-wise gate * up ──
    activated_gate = _apply_activation(gate, activation_function)  # [E, T, I]
    intermediate = up * activated_gate  # [E, T, I]

    # ── PRE_SCALE_DELAYED: multiply affinity into intermediate before down proj ──
    if expert_affinities_scaling_mode == ExpertAffinityScaleMode.PRE_SCALE_DELAYED:
        # expert_affinities: [T, E] -> [E, T, 1]
        affinity_etx = expert_affinities.T.unsqueeze(-1)  # [E, T, 1]
        # Cast back after the multiply (see PRE_SCALE above): keeps the down
        # projection einsum operands in the same dtype.
        intermediate = (intermediate * affinity_etx).to(dtype)  # [E, T, I]

    # ── Down projection ──
    next_states = torch.einsum(
        "eti,eih->eth", intermediate, down_proj_weight
    )  # [E, T, H]

    if down_proj_bias is not None:
        next_states = next_states + down_proj_bias

    # ── Accumulate across experts ──
    # The accumulator starts in the input dtype. On the POST_SCALE path the
    # fp32 router affinities implicitly promote it to fp32 on the first
    # iteration (unchanged from the original behavior); PRE_SCALE and
    # PRE_SCALE_DELAYED accumulate in the input dtype throughout. The final
    # cast restores the contract for POST_SCALE, where the promotion used to
    # leak fp32 into the residual stream (the NKI kernel path always returns
    # the input dtype).
    output = torch.zeros(num_tokens, hidden_size, device=device, dtype=dtype)

    if expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
        # Affinity applied per-expert after down projection
        for e in range(num_experts):
            output = output + next_states[e] * expert_affinities[:, e : e + 1]
    else:
        # PRE_SCALE and PRE_SCALE_DELAYED already baked affinity in earlier
        for e in range(num_experts):
            output = output + next_states[e]

    return output.to(dtype)


def _can_use_moe_cte_kernel(
    implementation: MoECTEImplementation,
    hidden_states: Tensor,
    gate_up_proj_weight: Tensor,
    gate_up_proj_scale: Optional[Tensor],
    down_proj_scale: Optional[Tensor],
):
    I_TP = gate_up_proj_weight.shape[-1]

    if I_TP < 128:
        return False

    if implementation in [
        MoECTEImplementation.shard_on_block_mx,
        MoECTEImplementation.shard_on_i_mx,
        MoECTEImplementation.shard_on_i_mx_hybrid,
    ]:
        scales_provided = gate_up_proj_scale is not None and down_proj_scale is not None

        assert scales_provided, (
            "gate_up_proj_scale and down_proj_scale must be provided for MX quantized compute"
        )

        if str(hidden_states.device) == "cpu":
            raise RuntimeError("MX quantized compute not supported on CPU")

        return True
    else:
        if not can_run_kernel(hidden_states):
            return False

        return True


def _build_moe_cte_spec(
    implementation: MoECTEImplementation,
    # shard_on_block parameters
    n_block_per_iter: int = 1,
    block_sharding_strategy: BlockShardStrategy = BlockShardStrategy.PING_PONG,
    # shard_on_block_mx parameters (MxFP4/MxFP8 quantization)
    n_static_blocks: int = -1,
    n_dynamic_blocks: int = -1,
    top_k: int = 1,
    ep_degree: int = 1,
    # shard_on_i and shard_on_i_dropping parameters
    checkpoint_activation: bool = False,
    expert_affinity_multiply_on_I: bool = False,
    # shard_on_i_hybrid parameters
    num_static_block: Optional[int] = None,
    # shard_on_block_mx: packed-scale HBM layout
    use_packed_scales: bool = False,
) -> MoECTESpec:
    shard_on_block = ShardOnBlockConfig(
        n_block_per_iter=n_block_per_iter,
        block_sharding_strategy=block_sharding_strategy,
        n_static_blocks=n_static_blocks,
        n_dynamic_blocks=n_dynamic_blocks,
        top_k=top_k,
        ep_degree=ep_degree,
        use_packed_scales=use_packed_scales,
    )
    shard_on_I = ShardOnIConfig(
        checkpoint_activation=checkpoint_activation,
        expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
        num_static_block=num_static_block,
    )
    return MoECTESpec(
        implementation=implementation,
        shard_on_block=shard_on_block,
        shard_on_I=shard_on_I,
    )


@nki.jit(mode="trace")
def _torch_compatible_moe_cte_kernel(
    hidden_states: nl.NkiTensor,
    expert_affinities_masked: Optional[nl.NkiTensor],
    gate_up_proj_weight: nl.NkiTensor,
    down_proj_weight: nl.NkiTensor,
    token_position_to_id: nl.NkiTensor,
    block_to_expert: nl.NkiTensor,
    block_size: int,
    implementation: MoECTEImplementation,
    # shard_on_block parameters
    n_block_per_iter: int = 1,
    block_sharding_strategy: BlockShardStrategy = BlockShardStrategy.PING_PONG,
    # shard_on_block_mx parameters (MxFP4/MxFP8 quantization)
    n_static_blocks: int = -1,
    n_dynamic_blocks: int = -1,
    top_k: int = 1,
    ep_degree: int = 1,
    # PACKED-affinity EP-rank tensor ([1, 1] int32); None on the older kernel / non-EP path.
    ep_rank: Optional[nl.NkiTensor] = None,
    # shard_on_i and shard_on_i_dropping parameters
    checkpoint_activation: bool = False,
    expert_affinity_multiply_on_I: bool = False,
    # shard_on_i_hybrid parameters
    num_static_block: Optional[int] = None,
    conditions: Optional[nl.NkiTensor] = None,
    gate_and_up_proj_bias: Optional[nl.NkiTensor] = None,
    down_proj_bias: Optional[nl.NkiTensor] = None,
    # Quantization scales
    gate_up_proj_scale: Optional[nl.NkiTensor] = None,
    down_proj_scale: Optional[nl.NkiTensor] = None,
    gate_up_activations_T: Optional[nl.NkiTensor] = None,
    down_activations: Optional[nl.NkiTensor] = None,
    activation_function: ActFnType = ActFnType.SiLU,
    # DMA skipping
    skip_token: bool = False,
    skip_weight: bool = False,
    compute_dtype: Any = nl.bfloat16,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    use_packed_scales: bool = False,
):
    spec = _build_moe_cte_spec(
        implementation=implementation,
        n_block_per_iter=n_block_per_iter,
        block_sharding_strategy=block_sharding_strategy,
        n_static_blocks=n_static_blocks,
        n_dynamic_blocks=n_dynamic_blocks,
        top_k=top_k,
        ep_degree=ep_degree,
        checkpoint_activation=checkpoint_activation,
        expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
        num_static_block=num_static_block,
        use_packed_scales=use_packed_scales,
    )

    quantization_config = QuantizationConfig(
        gate_up_proj_scale=gate_up_proj_scale, down_proj_scale=down_proj_scale
    )

    skip_dma = SkipMode(skip_token=skip_token, skip_weight=skip_weight)

    # ep_rank is a first-class moe_cte kernel arg (the packed-affinity read derives the global
    # expert base as ep_rank * num_local_experts). It is None on non-packed / decode paths, which
    # the kernel treats as EP-rank 0.
    return nki_moe_cte_kernel(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        block_size=block_size,
        spec=spec,
        conditions=conditions,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        quantization_config=quantization_config,
        gate_up_activations_T=gate_up_activations_T,
        down_activations=down_activations,
        activation_function=activation_function,
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        ep_rank=ep_rank,
    )


@nki.jit(mode="trace")
def _torch_compatible_moe_cte_static_mx_kernel(
    hidden_states: nl.NkiTensor,
    expert_affinities_masked: nl.NkiTensor,
    gate_up_proj_weight: nl.NkiTensor,
    down_proj_weight: nl.NkiTensor,
    token_position_to_id: nl.NkiTensor,
    block_to_expert: nl.NkiTensor,
    block_size: int,
    n_static_blocks: int = -1,
    n_dynamic_blocks: int = 55,
    top_k: int = 1,
    ep_degree: int = 1,
    conditions: Optional[nl.NkiTensor] = None,
    gate_and_up_proj_bias: Optional[nl.NkiTensor] = None,
    down_proj_bias: Optional[nl.NkiTensor] = None,
    gate_up_activations_T: Optional[nl.NkiTensor] = None,
    down_activations: Optional[nl.NkiTensor] = None,
    activation_function: ActFnType = ActFnType.SiLU,
    skip_token: bool = False,
    skip_weight: bool = False,
    compute_dtype: Any = nl.bfloat16,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    use_packed_scales: bool = False,
    # gate_up_proj_scale = [E, 2, 1] fp32 (gate=idx0, up=idx1); down_proj_scale = [E, 1] fp32.
    gate_up_proj_scale: Optional[nl.NkiTensor] = None,
    down_proj_scale: Optional[nl.NkiTensor] = None,
    gate_up_in_scale: Optional[nl.NkiTensor] = None,
    down_in_scale: Optional[nl.NkiTensor] = None,
):
    """STATIC_MX direct call into ``bwmm_shard_on_block_mx`` (bypasses the public
    dispatcher, which does not forward STATIC_MX kwargs)."""
    skip_dma = SkipMode(skip_token=skip_token, skip_weight=skip_weight)

    return nki_bwmm_shard_on_block_mx_kernel(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        conditions=conditions,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        block_size=block_size,
        n_static_blocks=n_static_blocks,
        n_dynamic_blocks=n_dynamic_blocks,
        top_k=top_k,
        ep_degree=ep_degree,
        gate_up_activations_T=gate_up_activations_T,
        down_activations=down_activations,
        activation_function=activation_function,
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        use_packed_scales=use_packed_scales,
        quantization_type=QuantizationType.STATIC_MX,
        gate_up_in_scale=gate_up_in_scale,
        down_in_scale=down_in_scale,
    )
