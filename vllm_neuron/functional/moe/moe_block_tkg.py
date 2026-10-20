# SPDX-License-Identifier: Apache-2.0
import torch
from torch import Tensor
from typing import Optional, Tuple

from vllm_neuron.functional.moe.moe_block_tkg_wrapper import (
    moe_block_tkg_wrapper,
)
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    RouterActFnType,
)

import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel


def _can_use_kernel(
    inp: Tensor,
    expert_down_weights: Tensor,
) -> bool:
    """
    Check if the moe_block_tkg NKI kernel can be used.

    Returns False when any NKI kernel constraint is violated or the tensors
    live on the CPU without the NKI simulator enabled.

    Kernel constraints checked:
        - Must be running on Neuron device or CPU with NKI simulator
        - T <= 128 for non-MX modes
        - H must be divisible by 256 (H_par * n_prgs = 128 * 2 for LNC-2)
        - I must be divisible by 128
    """
    if not can_run_kernel(inp):
        return False

    # TODO: validate MXFP8 with MoE kernel, then auto-enable
    if expert_down_weights.dtype == torch.uint16:
        return True  # MXFP4 does not have a torch flow currently, always try kernel.

    if inp.dim() == 3:
        B, S, H = inp.shape
        T = B * S
    else:
        T, H = inp.shape

    # H must be divisible by 256 for LNC-2 sharding
    if H % 256 != 0:
        return False

    return True


def _torch_moe_block_tkg_impl(
    inp: Tensor,
    gamma: Tensor,
    router_weights: Tensor,
    expert_gate_up_weights: Tensor,
    expert_down_weights: Tensor,
    router_bias: Optional[Tensor],
    expert_gate_up_bias: Optional[Tensor],
    expert_down_bias: Optional[Tensor],
    eps: float,
    top_k: int,
    router_act_fn: RouterActFnType,
    router_pre_norm: bool,
    norm_topk_prob: bool,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode,
    hidden_act_fn: ActFnType,
    gate_clamp_upper_limit: Optional[float],
    gate_clamp_lower_limit: Optional[float],
    up_clamp_upper_limit: Optional[float],
    up_clamp_lower_limit: Optional[float],
    hidden_actual: Optional[int],
    skip_router_logits: bool,
    is_all_expert: bool,
    rank_id: Optional[Tensor],
) -> Tuple[Tensor, ...]:
    """
    PyTorch fallback implementation of MoE block TKG computation.

    Performs RMSNorm + Router TopK + Expert MLPs using batched einsum.
    Processes all local experts and uses affinity weighting for correctness
    (selective mode gets the same result since unselected experts have zero
    affinity).

    Does not support quantization, shared experts, or residual fusion.

    Args:
        inp: [B, S, H] or [T, H] input tensor.
        gamma: [1, H] RMSNorm weights.
        router_weights: [H, E] router projection weights.
        expert_gate_up_weights: [E_L, H, 2, I] fused gate/up weights (local experts).
        expert_down_weights: [E_L, I, H] down weights (local experts).
        router_bias: [1, E] optional router bias.
        expert_gate_up_bias: [E_L, 2, I] optional gate/up bias.
        expert_down_bias: [E_L, H] optional down bias.
        eps: RMSNorm epsilon.
        top_k: Number of top experts per token.
        router_act_fn: Router activation function (SOFTMAX or SIGMOID).
        router_pre_norm: If True, apply activation before TopK; otherwise after.
        norm_topk_prob: Whether to normalize top-k affinities.
        expert_affinities_scaling_mode: Affinity scaling strategy.
        hidden_act_fn: Expert activation function.
        gate_clamp_*: Clamping limits for gate projection.
        up_clamp_*: Clamping limits for up projection.
        hidden_actual: Actual hidden dim for padded inputs.
        skip_router_logits: Whether to skip returning router logits.
        is_all_expert: Whether all-expert mode is used.
        rank_id: [1, 1] rank ID for expert parallelism.

    Returns:
        Tuple of (output, [router_logits]):
            - output: [T, H] MoE output tensor
            - router_logits: [T, E] (included when skip_router_logits is False)

    Raises:
        NotImplementedError: If an unsupported activation function is specified.
    """
    # Flatten to [T, H] if 3D
    if inp.dim() == 3:
        B, S, H = inp.shape
        inp = inp.reshape(B * S, H)

    num_tokens, hidden_size = inp.shape
    dtype = inp.dtype
    device = inp.device

    E_L, _, _, I = expert_gate_up_weights.shape
    E_global = router_weights.shape[1]

    # Step 1: RMSNorm
    if hidden_actual is None:
        hidden_actual = hidden_size

    hidden_states_float = inp.to(torch.float32)
    hidden_states_norm = hidden_states_float.clone()

    # Compute variance only on actual (non-padded) dimensions
    variance = hidden_states_float[..., :hidden_actual].pow(2).mean(-1, keepdim=True)
    hidden_states_norm[..., :hidden_actual] = hidden_states_norm[
        ..., :hidden_actual
    ] * torch.rsqrt(variance + eps)

    # Apply gamma and cast back
    gamma_squeezed = gamma.squeeze(0) if gamma.dim() == 2 else gamma
    hidden_states_norm = (gamma_squeezed * hidden_states_norm).to(dtype)
    hidden_states_norm[..., hidden_actual:] = 0.0

    # Step 2: Router computation
    # router_weights: [H, E] — matmul gives [T, E]
    router_logits = torch.matmul(
        hidden_states_norm.to(torch.float32),
        router_weights.to(torch.float32),
    )
    if router_bias is not None:
        bias = router_bias.to(torch.float32)
        if bias.dim() == 2:
            bias = bias.squeeze(0)
        router_logits = router_logits + bias

    # Step 3: Apply activation and TopK based on router_pre_norm
    if router_pre_norm:
        # Apply activation on full logits first, then select top-k
        if router_act_fn == RouterActFnType.SOFTMAX:
            activated_logits = torch.nn.functional.softmax(
                router_logits, dim=-1, dtype=torch.float32
            )
        elif router_act_fn == RouterActFnType.SIGMOID:
            activated_logits = torch.sigmoid(router_logits)
        else:
            raise NotImplementedError(
                f"Router activation {router_act_fn} is not supported in the "
                f"PyTorch fallback path."
            )

        router_top_value, router_indices = torch.topk(activated_logits, top_k, dim=-1)

        if norm_topk_prob:
            router_top_value = router_top_value / (
                router_top_value.sum(dim=-1, keepdim=True) + 1e-20
            )
    else:
        # Select top-k first, then apply activation on selected values
        router_top_value, router_indices = torch.topk(router_logits, top_k, dim=-1)

        if router_act_fn == RouterActFnType.SOFTMAX:
            router_top_value = torch.nn.functional.softmax(
                router_top_value, dim=-1, dtype=torch.float32
            )
        elif router_act_fn == RouterActFnType.SIGMOID:
            router_top_value = torch.sigmoid(router_top_value)
        else:
            raise NotImplementedError(
                f"Router activation {router_act_fn} is not supported in the "
                f"PyTorch fallback path."
            )

        if norm_topk_prob:
            router_top_value = router_top_value / (
                router_top_value.sum(dim=-1, keepdim=True) + 1e-20
            )

    # Step 4: Build local expert affinities [T, E_L]
    if E_L < E_global and rank_id is not None:
        # Expert parallelism: slice to local experts
        rid = rank_id.squeeze().long()
        start = E_L * rid

        full_expert_affinities = torch.zeros(
            num_tokens, E_global, device=device, dtype=torch.float32
        ).scatter_(1, router_indices, router_top_value)

        expert_affinities = full_expert_affinities[:, start : start + E_L]
    else:
        start = 0
        expert_affinities = torch.zeros(
            num_tokens, E_L, device=device, dtype=torch.float32
        ).scatter_(1, router_indices, router_top_value)

    # Step 5: Expert MLPs (batched einsum)
    # Reshape gate_up: [E_L, H, 2, I] -> [E_L, H, 2*I]
    gate_up_weights_flat = expert_gate_up_weights.reshape(E_L, hidden_size, 2 * I)

    # Gate + Up projection: [T, H] x [E_L, H, 2*I] -> [E_L, T, 2*I]
    gate_up = torch.einsum(
        "th,ehi->eti",
        hidden_states_norm.to(torch.float32),
        gate_up_weights_flat.to(torch.float32),
    )

    if expert_gate_up_bias is not None:
        # [E_L, 2, I] -> [E_L, 1, 2*I]
        gate_up = gate_up + expert_gate_up_bias.reshape(E_L, 2 * I).unsqueeze(1).to(
            torch.float32
        )

    # Split gate and up projections
    gate, up = torch.chunk(gate_up, chunks=2, dim=-1)  # Each: [E_L, T, I]

    # Apply clamping
    if gate_clamp_lower_limit is not None or gate_clamp_upper_limit is not None:
        gate = gate.clamp(min=gate_clamp_lower_limit, max=gate_clamp_upper_limit)

    if up_clamp_lower_limit is not None or up_clamp_upper_limit is not None:
        up = up.clamp(min=up_clamp_lower_limit, max=up_clamp_upper_limit)

    # Apply activation function
    if hidden_act_fn == ActFnType.SiLU:
        activated = torch.nn.functional.silu(gate)
    elif hidden_act_fn == ActFnType.GELU:
        activated = torch.nn.functional.gelu(gate)
    elif hidden_act_fn == ActFnType.Swish:
        swiglu_alpha = 1.702  # This value is hardcoded in the nisa.
        activated = gate * torch.sigmoid(gate * swiglu_alpha)
    else:
        raise NotImplementedError(
            f"Activation function {hidden_act_fn} is not supported in the "
            f"PyTorch fallback path for moe_block_tkg."
        )

    intermediate = up * activated  # [E_L, T, I]

    # Down projection: [E_L, T, I] @ [E_L, I, H] -> [E_L, T, H]
    next_states = torch.einsum(
        "eti,eih->eth",
        intermediate,
        expert_down_weights.to(torch.float32),
    )

    if expert_down_bias is not None:
        next_states = next_states + expert_down_bias.unsqueeze(1).to(
            torch.float32
        )  # [E_L, 1, H]

    # Step 6: Weighted sum by expert affinities
    if expert_affinities_scaling_mode == ExpertAffinityScaleMode.NO_SCALE:
        # No scaling: each selected expert contributes with weight 1.0,
        # but scatter_ already placed the affinity values; for NO_SCALE
        # we should use binary mask instead
        binary_affinities = (expert_affinities > 0).to(torch.float32)
        output = torch.zeros(
            num_tokens, hidden_size, device=device, dtype=torch.float32
        )
        for e in range(E_L):
            output = output + next_states[e] * binary_affinities[:, e : e + 1]
    else:
        # POST_SCALE: weight by affinity values
        output = torch.zeros(
            num_tokens, hidden_size, device=device, dtype=torch.float32
        )
        for e in range(E_L):
            output = output + next_states[e] * expert_affinities[:, e : e + 1]

    output = output.to(dtype)

    if skip_router_logits:
        return output

    return output, router_logits


def moe_block_tkg(
    inp: Tensor,
    gamma: Tensor,
    router_weights: Tensor,
    expert_gate_up_weights: Tensor,
    expert_down_weights: Tensor,
    rank_id: Optional[Tensor] = None,
    top_k: int = 1,
    shared_expert_gate_w: Optional[Tensor] = None,
    shared_expert_up_w: Optional[Tensor] = None,
    shared_expert_down_w: Optional[Tensor] = None,
    expert_gate_up_weights_scale: Optional[Tensor] = None,
    expert_down_weights_scale: Optional[Tensor] = None,
    router_bias: Optional[Tensor] = None,
    expert_gate_up_bias: Optional[Tensor] = None,
    expert_down_bias: Optional[Tensor] = None,
    shared_expert_gate_bias: Optional[Tensor] = None,
    shared_expert_up_bias: Optional[Tensor] = None,
    shared_expert_down_bias: Optional[Tensor] = None,
    eps: float = 1e-6,
    router_act_fn: RouterActFnType = RouterActFnType.SIGMOID,
    router_pre_norm: bool = True,
    norm_topk_prob: bool = False,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.NO_SCALE,
    hidden_act_fn: ActFnType = ActFnType.SiLU,
    hidden_act_scale_factor: Optional[float] = None,
    hidden_act_bias: Optional[float] = None,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    router_mm_dtype=nl.bfloat16,
    hidden_actual: Optional[int] = None,
    skip_router_logits: bool = False,
    is_all_expert: bool = False,
    residual: Optional[Tensor] = None,
    expert_gate_up_input_scale: Optional[Tensor] = None,
    expert_down_input_scale: Optional[Tensor] = None,
) -> Tuple[Tensor, ...]:
    """
    MoE Block kernel API for token generation (decode phase).

    Performs a complete fused Mixture of Experts forward pass:
    RMSNorm + Router TopK + Expert MLPs + optional Shared Expert.

    Automatically selects between the NKI kernel and a PyTorch fallback based
    on hardware constraints and input characteristics.

    When using the NKI kernel:
        - Fused RMSNorm preprocessing
        - Router-based expert selection with top-K routing
        - All expert MLP computations with affinity weighting
        - Optional shared expert computation
        - LNC-2 sharding on hidden dimension
        - Supports MxFP4 quantization and FP8 row quantization
        - Optional fused residual add (MXFP all-expert mode)

    When using the PyTorch fallback:
        - Uses einsum-based expert computation
        - Supports all input sizes and runs on CPU
        - No quantization or shared expert support

    Dimensions:
        B: Batch size
        S: Sequence length
        T: Total tokens (B * S)
        H: Hidden dimension (must be divisible by 256 for kernel)
        I: Intermediate dimension (must be divisible by 128 for kernel)
        E: Number of global experts
        E_L: Number of local experts
        K: Top-K experts per token

    Args:
        inp: Input tensor [B, S, H].
        gamma: RMSNorm weights [1, H].
        router_weights: Router projection weights [H, E].
        expert_gate_up_weights: Fused gate/up weights [E_L, H, 2, I] for bf16/fp16
            or [E_L, 128, 2, ceil(H/512), I] for MxFP4.
        expert_down_weights: Down projection weights [E_L, I, H] for bf16/fp16
            or [E_L, I_p, ceil(I/512), H] for MxFP4.
        rank_id: Rank ID tensor [1, 1] for expert parallelism. Required when
            is_all_expert=True. Default: None.
        top_k: Number of top experts to select per token. Default: 1.
        shared_expert_gate_w: Shared expert gate weights [H, I]. Default: None.
        shared_expert_up_w: Shared expert up weights [H, I]. Default: None.
        shared_expert_down_w: Shared expert down weights [I, H]. Default: None.
        expert_gate_up_weights_scale: MxFP/FP8 quantization scales for gate/up
            weights. Default: None.
        expert_down_weights_scale: MxFP/FP8 quantization scales for down
            weights. Default: None.
        router_bias: Router bias [1, E]. Default: None.
        expert_gate_up_bias: Gate/up bias [E_L, 2, I]. Default: None.
        expert_down_bias: Down projection bias [E_L, H]. Default: None.
        shared_expert_gate_bias: Shared expert gate bias [1, I]. Default: None.
        shared_expert_up_bias: Shared expert up bias [1, I]. Default: None.
        shared_expert_down_bias: Shared expert down bias [1, H]. Default: None.
        eps: RMSNorm epsilon. Default: 1e-6.
        router_act_fn: Router activation function. Default: SIGMOID.
        router_pre_norm: Apply activation before TopK. Default: True.
        norm_topk_prob: Normalize top-K affinities. Default: False.
        expert_affinities_scaling_mode: Affinity scaling strategy.
            Default: NO_SCALE.
        hidden_act_fn: Expert activation function. Default: SiLU.
        hidden_act_scale_factor: Activation scale factor (placeholder, must be
            None). Default: None.
        hidden_act_bias: Activation bias (placeholder, must be None).
            Default: None.
        gate_clamp_upper_limit: Upper bound for gate clamping. Default: None.
        gate_clamp_lower_limit: Lower bound for gate clamping. Default: None.
        up_clamp_upper_limit: Upper bound for up clamping. Default: None.
        up_clamp_lower_limit: Lower bound for up clamping. Default: None.
        router_mm_dtype: Router matmul dtype. Default: bfloat16.
        hidden_actual: Actual hidden dim for padded inputs. Default: None.
        skip_router_logits: Skip returning router logits. Default: False.
        is_all_expert: If True, use all-expert mode. Default: False.
        residual: Residual tensor [B, S, H] or [T, H] for fused add
            (MXFP all-expert mode only). Default: None.

    Returns:
        Tuple containing:
            - output: [T, H] MoE output tensor
            - router_logits: [T, E] (included when skip_router_logits is False)
            - residual_out: [T, H] (included when residual is provided)

    Constraints (falls back to PyTorch if violated):
        - inp must not be on CPU
        - T <= 128 for non-MX modes
        - H must be divisible by 256 (LNC-2 with 128 partition size)
        - I must be divisible by 128

    Notes:
        - H must be divisible by 128 (partition size) and 256 (128 * n_prgs for LNC-2)
        - Selective-expert mode: T <= 128
        - All-expert mode with MXFP: T must be divisible by 4
        - All-expert mode without MXFP: T <= 128
        - hidden_act_scale_factor and hidden_act_bias are placeholders (must be None)
        - The caller is responsible for all-reduce after calling (if using TP/EP)

    Example:
        >>> outputs = moe_block_tkg(
        ...     inp=hidden_states,                       # [B, S, H]
        ...     gamma=layernorm_weight.unsqueeze(0),     # [1, H]
        ...     router_weights=router_weight,            # [H, E]
        ...     expert_gate_up_weights=gate_up_weight,   # [E_L, H, 2, I]
        ...     expert_down_weights=down_weight,         # [E_L, I, H]
        ...     rank_id=torch.zeros((1, 1), dtype=torch.int32, device="xla"),
        ...     top_k=2,
        ...     router_bias=router_bias.unsqueeze(0),    # [1, E]
        ...     expert_gate_up_bias=gate_up_bias,        # [E_L, 2, I]
        ...     expert_down_bias=down_bias,              # [E_L, H]
        ...     expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        ...     gate_clamp_upper_limit=7.0,
        ...     up_clamp_upper_limit=8.0,
        ...     up_clamp_lower_limit=-6.0,
        ...     is_all_expert=True,
        ... )
        >>> output = outputs[0]
        >>> router_logits = outputs[1]  # if skip_router_logits=False
        >>> # Then apply all_reduce for TP/EP
    """
    can_use_kernel = _can_use_kernel(
        inp=inp,
        expert_down_weights=expert_down_weights,
    )

    # FIXME: [NCC_INKI016] Kernel validation exception: skip_store_router_logits=True is not currently supported due to a compiler limitation NCC_IGCA090
    if can_use_kernel:
        # Ensure input is 3D [B, S, H] for kernel
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)

        wrapped = wrap_nki(moe_block_tkg_wrapper)

        # Tiled MX path returns a single tensor (no router logits); other paths
        # return ``(output, router_logits[, residual_out])``. Unpack defensively.
        result = wrapped[2](
            inp=inp,
            gamma=gamma,
            router_weights=router_weights,
            expert_gate_up_weights=expert_gate_up_weights,
            expert_down_weights=expert_down_weights,
            shared_expert_gate_w=shared_expert_gate_w,
            shared_expert_up_w=shared_expert_up_w,
            shared_expert_down_w=shared_expert_down_w,
            expert_gate_up_weights_scale=expert_gate_up_weights_scale,
            expert_down_weights_scale=expert_down_weights_scale,
            router_bias=router_bias,
            expert_gate_up_bias=expert_gate_up_bias,
            expert_down_bias=expert_down_bias,
            shared_expert_gate_bias=shared_expert_gate_bias,
            shared_expert_up_bias=shared_expert_up_bias,
            shared_expert_down_bias=shared_expert_down_bias,
            eps=eps,
            top_k=top_k,
            router_act_fn=router_act_fn,
            router_pre_norm=router_pre_norm,
            norm_topk_prob=norm_topk_prob,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            hidden_act_fn=hidden_act_fn,
            hidden_act_scale_factor=hidden_act_scale_factor,
            hidden_act_bias=hidden_act_bias,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            router_mm_dtype=router_mm_dtype,
            hidden_actual=hidden_actual,
            skip_router_logits=False,
            is_all_expert=is_all_expert,
            rank_id=rank_id,
            residual=residual,
            expert_gate_up_input_scale=expert_gate_up_input_scale,
            expert_down_input_scale=expert_down_input_scale,
        )

        if isinstance(result, (tuple, list)):
            outputs = result[0]
            router_logits = result[1] if len(result) > 1 else None
        else:
            outputs = result
            router_logits = None

        if not skip_router_logits:
            return outputs, router_logits

        return outputs

    # --- PyTorch fallback path ---
    if shared_expert_gate_w is not None or shared_expert_down_w is not None:
        raise NotImplementedError(
            "Shared experts are not supported in the PyTorch fallback path "
            "for moe_block_tkg."
        )

    if (
        expert_gate_up_weights_scale is not None
        or expert_down_weights_scale is not None
    ):
        raise NotImplementedError(
            "Weight quantization scales are not supported in the PyTorch "
            "fallback path for moe_block_tkg."
        )

    if expert_gate_up_input_scale is not None or expert_down_input_scale is not None:
        raise NotImplementedError(
            "STATIC_MX input scales are not supported in the PyTorch "
            "fallback path for moe_block_tkg."
        )

    if residual is not None:
        raise NotImplementedError(
            "Fused residual add is not supported in the PyTorch fallback "
            "path for moe_block_tkg (MXFP all-expert only feature)."
        )

    if hidden_act_scale_factor is not None or hidden_act_bias is not None:
        raise NotImplementedError(
            "hidden_act_scale_factor and hidden_act_bias are placeholders "
            "and must be None."
        )

    return _torch_moe_block_tkg_impl(
        inp=inp,
        gamma=gamma,
        router_weights=router_weights,
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        router_bias=router_bias,
        expert_gate_up_bias=expert_gate_up_bias,
        expert_down_bias=expert_down_bias,
        eps=eps,
        top_k=top_k,
        router_act_fn=router_act_fn,
        router_pre_norm=router_pre_norm,
        norm_topk_prob=norm_topk_prob,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        hidden_act_fn=hidden_act_fn,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        hidden_actual=hidden_actual,
        skip_router_logits=skip_router_logits,
        is_all_expert=is_all_expert,
        rank_id=rank_id,
    )
