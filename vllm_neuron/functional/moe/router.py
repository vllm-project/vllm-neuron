# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from typing import Optional, Union, Tuple, Callable

import torch
import torch.nn.functional as F
from torch import Tensor
import nki

from nkilib.core.router_topk.router_topk import router_topk
from nkilib.core.utils.common_types import RouterActFnType

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel

router_topk_jit = nki.jit(router_topk)


class RouterComputationOrder(Enum):
    """
    Enum specifying the computation order for MoE router operations.

    This enum determines the sequence of operations applied during routing:

    - ``PRENORM_LINEAR_TOPK_ACT_SCATTER``: RMSNorm (optional) → Linear → TopK → Activation → Scatter
        - Default behavior, applies optional RMSNorm to hidden states first
        - Projects to router logits, selects top-k experts
        - Applies activation (softmax/sigmoid) only to selected top-k values
        - Scatters activated values to full expert affinity matrix

    - ``PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER``: RMSNorm (optional) → Linear → Activation → TopK → Renorm → Scatter
        - Applies optional RMSNorm to hidden states first
        - Projects to router logits
        - Applies activation to ALL expert logits before selection
        - Selects top-k experts from activated values
        - L1 renormalizes selected values so they sum to 1.0
        - Scatters to full expert affinity matrix

    - ``PRENORM_LINEAR_TOPK_SCATTER_ACT``: RMSNorm (optional) → Linear → TopK → Scatter → Activation
        - Applies optional RMSNorm to hidden states first
        - Projects to router logits
        - Selects top-k experts based on raw logits
        - Scatters raw logit values to full matrix (zeros elsewhere)
        - Applies activation to the full sparse matrix

    Usage Examples:
        >>> from vllm_neuron.functional.moe.router import router, RouterComputationOrder
        >>>
        >>> # Default computation order (PRENORM_LINEAR_TOPK_ACT_SCATTER)
        >>> affinities = router(hidden_states, router_weights, top_k=2)
        >>>
        >>> # Activation before TopK selection with optional RMSNorm
        >>> affinities = router(
        ...     hidden_states, router_weights, top_k=2,
        ...     router_computation_order=RouterComputationOrder.PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER,
        ...     gamma=gamma,  # Optional RMSNorm
        ... )
        >>>
        >>> # Scatter before activation with optional RMSNorm
        >>> affinities = router(
        ...     hidden_states, router_weights, top_k=2,
        ...     router_computation_order=RouterComputationOrder.PRENORM_LINEAR_TOPK_SCATTER_ACT,
        ...     gamma=gamma  # Optional RMSNorm
        ... )
    """

    PRENORM_LINEAR_TOPK_ACT_SCATTER = "prenorm_linear_topk_act_scatter"
    PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER = "prenorm_linear_act_topk_renorm_scatter"
    PRENORM_LINEAR_TOPK_SCATTER_ACT = "prenorm_linear_topk_scatter_act"


def router(
    hidden_states: Tensor,
    router_weights: Tensor,
    top_k: int,
    router_bias: Optional[Tensor] = None,
    activation: Union[str, Callable[[Tensor], Tensor]] = "softmax",
    return_logits: bool = False,
    gamma: Optional[Tensor] = None,
    eps: float = 1e-6,
    computation_dtype: torch.dtype = torch.float32,
    router_computation_order: RouterComputationOrder = RouterComputationOrder.PRENORM_LINEAR_TOPK_ACT_SCATTER,
    shard_on_tokens: Optional[bool] = None,
    transposed_hidden_states: bool = False,
    x_sb_layout: Optional[int] = None,
    use_column_tiling: Optional[bool] = None,
    use_indirect_dma_scatter: Optional[bool] = None,
    use_PE_broadcast_w_bias: Optional[bool] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Router API for Mixture of Experts (MoE) expert selection.

    This function computes routing probabilities for selecting experts in MoE layers.
    It performs top-k expert selection and returns routing affinities. It supports optional RMSNorm
    preprocessing and flexible activation functions for different routing strategies.

    The function uses an optimized NKI kernel when constraints are met.
    Falls back to PyTorch implementation otherwise.

    Args:
        hidden_states: Input hidden states tensor with shape [T, H]
            where T is the number of tokens and H is the hidden dimension.
            Can be [H, T] if transposed_hidden_states is set to True
        router_weights: Router projection weights with shape [H, E]
            where E is the number of experts
        top_k: Number of top experts to select per token (typically 1 or 2)
        router_bias: Optional router projection bias with shape [E]
            Default: None (no bias)
        activation: Activation function to apply to router scores.
            Can be either a string or a callable function:
            - String options: "softmax" (default) for standard MoE routing,
              "sigmoid" for alternative routing strategies
            - Callable: Any function that takes a Tensor and returns a Tensor
              Examples: F.softmax, torch.sigmoid, or custom functions
        return_logits: Whether to return raw router logits in addition to affinities
            Default: False (return only affinities)
        gamma: Optional RMSNorm weights with shape [H] for input preprocessing.
            If provided, applies RMSNorm before router computation for all
            computation orders.
            Default: None (no normalization)
        eps: Epsilon value for RMSNorm numerical stability
            Default: 1e-6
        computation_dtype: Data type for computation (float32, float16, or bfloat16)
            Default: torch.float32
        router_computation_order: Specifies the order of operations in routing computation.
            See RouterComputationOrder enum for details.
            - PRENORM_LINEAR_TOPK_ACT_SCATTER: RMSNorm (optional) → Linear → TopK → Activation → Scatter (default)
            - PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER: RMSNorm (optional) → Linear → Activation → TopK → L1 Renorm → Scatter
            - PRENORM_LINEAR_TOPK_SCATTER_ACT: RMSNorm (optional) → Linear → TopK → Scatter → Activation
            Default: RouterComputationOrder.PRENORM_LINEAR_TOPK_ACT_SCATTER
        shard_on_tokens: [Kernel only arg] Enable LNC sharding across token dimension
        transposed_hidden_states: If True, hidden_states should be [H, T] instead of [T, H]
        x_sb_layout: [Kernel only arg] Layout of input x in SBUF (0, 1, or 2)
        use_column_tiling: [Kernel only arg] Enable PE array column tiling for small T
        use_indirect_dma_scatter: [Kernel only arg] Use indirect DMA for expert affinity scatter
        use_PE_broadcast_w_bias: [Kernel only arg] Use tensor engine for bias broadcast

    Returns:
        If return_logits=False:
            expert_affinities: Tensor with shape [T, E] containing routing probabilities.
                              Non-zero values for selected experts, zeros elsewhere.

        If return_logits=True:
            Tuple containing:
            - expert_affinities: [T, E] routing probabilities as above
            - router_logits: [T, E] raw router logits before top-k selection

    Raises:
        ValueError: If activation is not a valid string ("softmax" or "sigmoid")
                   or a callable function, or if other input parameters are invalid.

    Usage Examples:
        >>> # Basic router usage with softmax activation (default PRENORM_LINEAR_TOPK_ACT_SCATTER order)
        >>> hidden_states = torch.randn(128, 768)  # 128 tokens, 768 hidden dim
        >>> router_weights = torch.randn(768, 8)   # 8 experts
        >>>
        >>> affinities = router(
        ...     hidden_states=hidden_states,
        ...     router_weights=router_weights,
        ...     top_k=2,
        ...     activation="softmax"
        ... )
        >>> print(affinities.shape)  # torch.Size([128, 8])
        >>> print((affinities > 0).sum(dim=1))  # Each token routes to exactly 2 experts

        >>> # Router with RMSNorm preprocessing and bias
        >>> gamma = torch.ones(768)
        >>> router_bias = torch.zeros(8)
        >>>
        >>> affinities, logits = router(
        ...     hidden_states=hidden_states,
        ...     router_weights=router_weights,
        ...     top_k=2,
        ...     router_bias=router_bias,
        ...     gamma=gamma,
        ...     eps=1e-5,
        ...     return_logits=True
        ... )
        >>> print(affinities.shape, logits.shape)  # torch.Size([128, 8]) torch.Size([128, 8])

        >>> # Router with PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER computation order
        >>> # Applies activation to ALL logits before TopK selection, then L1 renormalizes
        >>> affinities = router(
        ...     hidden_states=hidden_states,
        ...     router_weights=router_weights,
        ...     top_k=2,
        ...     activation="softmax",
        ...     router_computation_order=RouterComputationOrder.PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER,
        ...     gamma=gamma,  # Optional RMSNorm preprocessing
        ... )
        >>> print(affinities.shape)  # torch.Size([128, 8])

        >>> # Router with PRENORM_LINEAR_TOPK_SCATTER_ACT computation order
        >>> # Scatters to full matrix first, then applies activation
        >>> affinities = router(
        ...     hidden_states=hidden_states,
        ...     router_weights=router_weights,
        ...     top_k=2,
        ...     activation="softmax",
        ...     router_computation_order=RouterComputationOrder.PRENORM_LINEAR_TOPK_SCATTER_ACT,
        ...     gamma=gamma  # Optional RMSNorm preprocessing
        ... )
        >>> print(affinities.shape)  # torch.Size([128, 8])

        >>> # Router with callable activation functions
        >>> import torch.nn.functional as F
        >>>
        >>> # Using F.softmax as callable (equivalent to "softmax" string)
        >>> affinities = router(
        ...     hidden_states=hidden_states,
        ...     router_weights=router_weights,
        ...     top_k=2,
        ...     activation=lambda x: F.softmax(x, dim=-1)
        ... )
        >>> print(affinities.shape)  # torch.Size([128, 8])

        >>> # Using torch.sigmoid as callable (equivalent to "sigmoid" string)
        >>> affinities = router(
        ...     hidden_states=hidden_states,
        ...     router_weights=router_weights,
        ...     top_k=2,
        ...     activation=torch.sigmoid
        ... )
        >>> print(affinities.shape)  # torch.Size([128, 8])

        >>> # Router with custom computation dtype for reduced precision
        >>> affinities = router(
        ...     hidden_states=hidden_states,
        ...     router_weights=router_weights,
        ...     top_k=2,
        ...     computation_dtype=torch.float16
        ... )
        >>> print(affinities.dtype)  # torch.float16
        >>> print(affinities.shape)  # torch.Size([128, 8])
    """
    # Validate inputs
    _validate_router_inputs(
        hidden_states,
        router_weights,
        top_k,
        router_bias,
        gamma,
        computation_dtype,
        router_computation_order,
        transposed_hidden_states,
    )

    # Check if kernel can be used
    can_use_kernel = _can_use_kernel(
        hidden_states,
        router_weights,
        top_k,
        activation,
        gamma,
        router_bias,
        router_computation_order,
        transposed_hidden_states,
    )

    hidden_states = hidden_states.to(computation_dtype)

    if can_use_kernel:
        expert_affinities, router_logits = _nki_router_impl(
            hidden_states=hidden_states,
            router_weights=router_weights,
            top_k=top_k,
            router_bias=router_bias,
            activation=activation,
            computation_dtype=computation_dtype,
            router_computation_order=router_computation_order,
            skip_store_router_logits=not return_logits,
            shard_on_tokens=shard_on_tokens,
            x_hbm_layout=0 if transposed_hidden_states else 1,
            x_sb_layout=x_sb_layout,
            use_column_tiling=use_column_tiling,
            use_indirect_dma_scatter=use_indirect_dma_scatter,
            use_PE_broadcast_w_bias=use_PE_broadcast_w_bias,
        )
    else:
        # PyTorch fallback implementation
        expert_affinities, router_logits = _torch_router_impl(
            hidden_states=hidden_states.T
            if transposed_hidden_states
            else hidden_states,
            router_weights=router_weights,
            top_k=top_k,
            router_bias=router_bias,
            activation=activation,
            gamma=gamma,
            eps=eps,
            computation_dtype=computation_dtype,
            router_computation_order=router_computation_order,
        )

    if return_logits:
        return expert_affinities, router_logits
    else:
        return expert_affinities


def _torch_router_impl(
    hidden_states: Tensor,
    router_weights: Tensor,
    top_k: int,
    router_bias: Optional[Tensor],
    activation: Union[str, Callable[[Tensor], Tensor]],
    gamma: Optional[Tensor],
    eps: float,
    computation_dtype: torch.dtype,
    router_computation_order: RouterComputationOrder,
) -> Tuple[Tensor, Tensor]:
    """
    PyTorch implementation of router computation with configurable computation order.

    Dispatches to the appropriate implementation based on router_computation_order:
    - PRENORM_LINEAR_TOPK_ACT_SCATTER: RMSNorm (optional) → Linear → TopK → Activation → Scatter
    - PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER: RMSNorm (optional) → Linear → Activation → TopK → L1 Renorm → Scatter
    - PRENORM_LINEAR_TOPK_SCATTER_ACT: RMSNorm (optional) → Linear → TopK → Scatter → Activation

    Args:
        hidden_states: Input tensor [T, H]
        router_weights: Router projection weights [H, E]
        top_k: Number of experts per token
        router_bias: Optional router bias [E]
        activation: Activation function ("softmax", "sigmoid", or callable)
        gamma: Optional RMSNorm weights [H]
        eps: RMSNorm epsilon
        computation_dtype: Data type for computation
        router_computation_order: Specifies the order of operations

    Returns:
        Tuple containing:
        - expert_affinities: [T, E] routing probabilities with zeros for non-selected experts
        - router_logits: [T, E] raw router logits
    """
    if (
        router_computation_order
        == RouterComputationOrder.PRENORM_LINEAR_TOPK_ACT_SCATTER
    ):
        return _torch_router_impl_prenorm_linear_topk_act_scatter(
            hidden_states=hidden_states,
            router_weights=router_weights,
            top_k=top_k,
            router_bias=router_bias,
            activation=activation,
            gamma=gamma,
            eps=eps,
            computation_dtype=computation_dtype,
        )
    elif (
        router_computation_order
        == RouterComputationOrder.PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER
    ):
        return _torch_router_impl_prenorm_linear_act_topk_renorm_scatter(
            hidden_states=hidden_states,
            router_weights=router_weights,
            top_k=top_k,
            router_bias=router_bias,
            activation=activation,
            gamma=gamma,
            eps=eps,
            computation_dtype=computation_dtype,
        )
    elif (
        router_computation_order
        == RouterComputationOrder.PRENORM_LINEAR_TOPK_SCATTER_ACT
    ):
        return _torch_router_impl_prenorm_linear_topk_scatter_act(
            hidden_states=hidden_states,
            router_weights=router_weights,
            top_k=top_k,
            router_bias=router_bias,
            activation=activation,
            gamma=gamma,
            eps=eps,
            computation_dtype=computation_dtype,
        )
    else:
        raise ValueError(
            f"Unknown router_computation_order: {router_computation_order}"
        )


def _torch_router_impl_prenorm_linear_topk_act_scatter(
    hidden_states: Tensor,
    router_weights: Tensor,
    top_k: int,
    router_bias: Optional[Tensor],
    activation: Union[str, Callable[[Tensor], Tensor]],
    gamma: Optional[Tensor],
    eps: float,
    computation_dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    """
    PyTorch implementation: RMSNorm (optional) → Linear → TopK → Activation → Scatter

    This is the default computation order. Applies optional RMSNorm to hidden states,
    projects to router logits, selects top-k experts, applies activation only to
    the selected top-k values, then scatters to the full affinity matrix.

    Args:
        hidden_states: Input tensor [T, H]
        router_weights: Router projection weights [H, E]
        top_k: Number of experts per token
        router_bias: Optional router bias [E]
        activation: Activation function ("softmax", "sigmoid", or callable)
        gamma: Optional RMSNorm weights [H]
        eps: RMSNorm epsilon
        computation_dtype: Data type for computation

    Returns:
        Tuple of (expert_affinities [T, E], router_logits [T, E])
    """
    T, H = hidden_states.shape
    E = router_weights.shape[1]
    device = hidden_states.device

    # Step 1: Optional RMSNorm preprocessing
    if gamma is not None:
        hidden_states = _torch_rms_norm(hidden_states, gamma, eps)

    # Step 2: Router linear projection
    router_logits = F.linear(
        hidden_states.to(computation_dtype),
        router_weights.T.to(computation_dtype),
        router_bias.to(computation_dtype) if router_bias is not None else None,
    )  # [T, E]

    # Step 3: Top-k expert selection
    router_top_values, router_indices = torch.topk(
        router_logits, top_k, dim=-1
    )  # [T, top_k]

    # Step 4: Apply activation function to top-k values only
    router_top_probs = _apply_activation(activation, router_top_values)  # [T, top_k]

    # Step 5: Scatter to full expert affinity matrix
    expert_affinities = torch.zeros(T, E, device=device, dtype=computation_dtype)
    expert_affinities.scatter_(1, router_indices, router_top_probs)  # [T, E]

    return expert_affinities, router_logits


def _torch_router_impl_prenorm_linear_act_topk_renorm_scatter(
    hidden_states: Tensor,
    router_weights: Tensor,
    top_k: int,
    router_bias: Optional[Tensor],
    activation: Union[str, Callable[[Tensor], Tensor]],
    gamma: Optional[Tensor],
    eps: float,
    computation_dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    """
    PyTorch implementation: RMSNorm (optional) → Linear → Activation → TopK → L1 Renorm → Scatter

    Applies optional RMSNorm to hidden states, projects to router logits, applies activation
    to ALL expert logits (not just top-k), selects top-k from activated values, L1-normalizes
    the selected values so they sum to 1.0, then scatters.

    Args:
        hidden_states: Input tensor [T, H]
        router_weights: Router projection weights [H, E]
        top_k: Number of experts per token
        router_bias: Optional router bias [E]
        activation: Activation function ("softmax", "sigmoid", or callable)
        gamma: Optional RMSNorm weights [H]
        eps: RMSNorm epsilon
        computation_dtype: Data type for computation

    Returns:
        Tuple of (expert_affinities [T, E], router_logits [T, E])
    """
    T, H = hidden_states.shape
    E = router_weights.shape[1]
    device = hidden_states.device

    # Step 1: Optional RMSNorm preprocessing
    if gamma is not None:
        hidden_states = _torch_rms_norm(hidden_states, gamma, eps)

    # Step 2: Router linear projection
    router_logits = F.linear(
        hidden_states.to(computation_dtype),
        router_weights.T.to(computation_dtype),
        router_bias.to(computation_dtype) if router_bias is not None else None,
    )  # [T, E]

    # Step 3: Apply activation function to ALL logits
    router_probs = _apply_activation(activation, router_logits)  # [T, E]

    # Step 4: Top-k selection from activated values
    router_top_probs, router_indices = torch.topk(
        router_probs, top_k, dim=-1
    )  # [T, top_k]

    # Step 5: L1 renormalization of top-k probabilities (always applied for this computation order)
    router_top_probs = router_top_probs / router_top_probs.sum(dim=-1, keepdim=True)

    # Step 6: Scatter to full expert affinity matrix
    expert_affinities = torch.zeros(T, E, device=device, dtype=computation_dtype)
    expert_affinities.scatter_(1, router_indices, router_top_probs)  # [T, E]

    return expert_affinities, router_logits


def _torch_router_impl_prenorm_linear_topk_scatter_act(
    hidden_states: Tensor,
    router_weights: Tensor,
    top_k: int,
    router_bias: Optional[Tensor],
    activation: Union[str, Callable[[Tensor], Tensor]],
    gamma: Optional[Tensor],
    eps: float,
    computation_dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    """
    PyTorch implementation: RMSNorm (optional) → Linear → TopK → Scatter → Activation

    Applies optional RMSNorm to hidden states, projects to router logits, selects top-k
    based on raw logits, scatters raw logit values to the full matrix (zeros elsewhere),
    then applies activation to the full sparse matrix.

    Args:
        hidden_states: Input tensor [T, H]
        router_weights: Router projection weights [H, E]
        top_k: Number of experts per token
        router_bias: Optional router bias [E]
        activation: Activation function ("softmax", "sigmoid", or callable)
        gamma: Optional RMSNorm weights [H]
        eps: RMSNorm epsilon
        computation_dtype: Data type for computation

    Returns:
        Tuple of (expert_affinities [T, E], router_logits [T, E])
    """
    T, H = hidden_states.shape
    E = router_weights.shape[1]
    device = hidden_states.device

    # Step 1: Optional RMSNorm preprocessing
    if gamma is not None:
        hidden_states = _torch_rms_norm(hidden_states, gamma, eps)

    # Step 2: Router linear projection
    router_logits = F.linear(
        hidden_states.to(computation_dtype),
        router_weights.T.to(computation_dtype),
        router_bias.to(computation_dtype) if router_bias is not None else None,
    )  # [T, E]

    # Step 3: Top-k selection based on raw logits
    router_top_values, router_indices = torch.topk(
        router_logits, top_k, dim=-1
    )  # [T, top_k]

    # Step 4: Scatter raw logit values to matrix initialized with -inf (not zeros)
    expert_affinities = torch.full(
        (T, E), float("-inf"), device=device, dtype=computation_dtype
    )
    expert_affinities.scatter_(1, router_indices, router_top_values)  # [T, E]

    # Step 5: Apply activation function to the full sparse matrix
    expert_affinities = _apply_activation(activation, expert_affinities)  # [T, E]

    return expert_affinities, router_logits


def _apply_activation(
    activation: Union[str, Callable[[Tensor], Tensor]], x: Tensor
) -> Tensor:
    """
    Apply activation function to input tensor.

    Args:
        activation: Activation function ("softmax", "sigmoid", or callable)
        x: Input tensor

    Returns:
        Activated tensor

    Raises:
        ValueError: If activation is not a valid string or callable
    """
    if isinstance(activation, str):
        if activation == "softmax":
            return F.softmax(x, dim=-1)
        elif activation == "sigmoid":
            return torch.sigmoid(x)
        else:
            raise ValueError(
                f"Unsupported activation function: {activation}. Use 'softmax' or 'sigmoid'."
            )
    elif callable(activation):
        return activation(x)
    else:
        raise ValueError(
            f"Activation must be either a string ('softmax' or 'sigmoid') or a callable function. Got: {type(activation)}"
        )


def _validate_router_inputs(
    hidden_states: Tensor,
    router_weights: Tensor,
    top_k: int,
    router_bias: Optional[Tensor] = None,
    gamma: Optional[Tensor] = None,
    computation_dtype: torch.dtype = torch.float32,
    router_computation_order: RouterComputationOrder = RouterComputationOrder.PRENORM_LINEAR_TOPK_ACT_SCATTER,
    transposed_hidden_states: bool = False,
) -> None:
    """
    Validate input parameters for router function.

    This function performs comprehensive input validation for the router function,
    ensuring all tensors have correct dimensions and shapes, and that parameters
    are within valid ranges.

    Args:
        hidden_states: Input hidden states tensor, expected shape [T, H]
        router_weights: Router projection weights, expected shape [H, E]
        top_k: Number of top experts to select per token
        router_bias: Optional router projection bias, expected shape [E]
        gamma: Optional RMSNorm weights, expected shape [H]
        computation_dtype: Data type for computation
        router_computation_order: Specifies the order of operations

    Raises:
        ValueError: If any input parameter has invalid shape or value
    """
    # Validate hidden_states dimensions
    if hidden_states.dim() != 2:
        raise ValueError(
            f"Expected hidden_states to be 2D [T, H], got shape {hidden_states.shape}"
        )

    # Validate router_weights dimensions
    if router_weights.dim() != 2:
        raise ValueError(
            f"Expected router_weights to be 2D [H, E], got shape {router_weights.shape}"
        )

    if transposed_hidden_states:
        H, T = hidden_states.shape
    else:
        T, H = hidden_states.shape
    H_w, E = router_weights.shape

    # Validate dimension compatibility
    if H != H_w:
        raise ValueError(
            f"Hidden dimension mismatch: hidden_states has {H}, router_weights has {H_w}"
        )

    # Validate optional router_bias shape
    if router_bias is not None and router_bias.shape != (E,):
        raise ValueError(f"Expected router_bias shape [E], got {router_bias.shape}")

    # Validate optional gamma shape
    if gamma is not None and gamma.shape != (H,):
        raise ValueError(f"Expected gamma shape [H], got {gamma.shape}")

    # Validate top_k range
    if top_k < 1 or top_k > E:
        raise ValueError(f"top_k must be between 1 and {E}, got {top_k}")

    # Validate computation_dtype
    supported_dtypes = {torch.float32, torch.float16, torch.bfloat16}
    if computation_dtype not in supported_dtypes:
        raise ValueError(
            f"computation_dtype must be one of {supported_dtypes}, got {computation_dtype}"
        )

    # Validate router_computation_order type
    if not isinstance(router_computation_order, RouterComputationOrder):
        raise ValueError(
            f"router_computation_order must be a RouterComputationOrder enum, got {type(router_computation_order)}"
        )


def _torch_rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """RMSNorm implementation in PyTorch."""
    original_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    x_normed = x_normed * weight
    return x_normed.to(original_dtype)


def _nki_router_impl(
    hidden_states: Tensor,
    router_weights: Tensor,
    top_k: int,
    router_bias: Optional[Tensor],
    activation: str,
    computation_dtype: torch.dtype,
    router_computation_order: RouterComputationOrder,
    skip_store_router_logits: bool,
    shard_on_tokens: Optional[bool],
    x_hbm_layout: int,
    x_sb_layout: Optional[int],
    use_column_tiling: Optional[bool],
    use_indirect_dma_scatter: Optional[bool],
    use_PE_broadcast_w_bias: Optional[bool],
) -> Tuple[Tensor, Tensor]:
    """
    NKI kernel implementation of router computation.

    Currently only supports PRENORM_LINEAR_TOPK_ACT_SCATTER and PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER computation orders.

    Args:
        hidden_states: [T, H] input tensor
        router_weights: [H, E] weight tensor
        top_k: Number of experts per token
        router_bias: Optional [E] bias tensor
        activation: "softmax" or "sigmoid"
        computation_dtype: Computation dtype
        router_computation_order: Specifies the order of operations
        skip_store_router_logits: Skips storing router logits to HBM
        shard_on_tokens: Enable LNC sharding across token dimension
        x_hbm_layout: Layout of input x in HBM (0=[H,T], 1=[T,H])
        x_sb_layout: Layout of input x in SBUF
        use_column_tiling: Enable PE array column tiling for small T
        use_indirect_dma_scatter: Use indirect DMA for expert affinity scatter
        use_PE_broadcast_w_bias: Use tensor engine for bias broadcast

    Returns:
        Tuple of (expert_affinities [T, E], router_logits [T, E])
    """
    # HBM layout dictates the expected hidden states shape
    if x_hbm_layout == 0:
        H, T = hidden_states.shape
    else:
        T, H = hidden_states.shape
    E = router_weights.shape[1]
    device = hidden_states.device

    # Set kernel args to reasonable defaults if not provided
    if shard_on_tokens is None:
        shard_on_tokens = T >= 128  # Enable LNC sharding when using a high token count
    if x_sb_layout is None:
        x_sb_layout = 0
    if use_column_tiling is None:
        # TODO: Default to True once NKILIB-584 is resolved
        use_column_tiling = False
    if use_indirect_dma_scatter is None:
        # TODO: Default to False once NKILIB-615 is resolved
        use_indirect_dma_scatter = True
    if use_PE_broadcast_w_bias is None:
        use_PE_broadcast_w_bias = False

    act_fn = (
        RouterActFnType.SOFTMAX if activation == "softmax" else RouterActFnType.SIGMOID
    )

    router_logits = torch.zeros(T, E, dtype=computation_dtype, device=device)
    expert_affinities = torch.zeros(T, E, dtype=computation_dtype, device=device)
    expert_index = torch.zeros(T, top_k, dtype=torch.int32, device=device)

    w_bias = router_bias.unsqueeze(0) if router_bias is not None else None

    # Map computation order to kernel's router_pre_norm parameter
    # router_pre_norm=True -> PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER (activation before topk)
    # router_pre_norm=False -> PRENORM_LINEAR_TOPK_ACT_SCATTER (activation after topk)
    router_pre_norm = (
        router_computation_order
        == RouterComputationOrder.PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER
    )

    # For PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER, always apply L1 renormalization
    norm_topk_prob = router_pre_norm

    router_topk_nki = wrap_nki(router_topk_jit)

    router_logits, expert_index, expert_affinities = router_topk_nki[2](
        x=hidden_states,
        w=router_weights,
        w_bias=w_bias,
        router_logits=router_logits,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        act_fn=act_fn,
        k=top_k,
        x_hbm_layout=x_hbm_layout,
        x_sb_layout=x_sb_layout,
        router_pre_norm=router_pre_norm,
        norm_topk_prob=norm_topk_prob,
        use_indirect_dma_scatter=use_indirect_dma_scatter,
        use_column_tiling=use_column_tiling,
        shard_on_tokens=shard_on_tokens,
        skip_store_router_logits=skip_store_router_logits,
        skip_store_expert_index=True,
        use_PE_broadcast_w_bias=use_PE_broadcast_w_bias,
    )

    return expert_affinities, router_logits


def _can_use_kernel(
    hidden_states: Tensor,
    router_weights: Tensor,
    top_k: int,
    activation: Union[str, Callable],
    gamma: Optional[Tensor],
    router_bias: Optional[Tensor] = None,
    router_computation_order: RouterComputationOrder = RouterComputationOrder.PRENORM_LINEAR_TOPK_ACT_SCATTER,
    transposed_hidden_states: bool = False,
) -> bool:
    """
    Check if the NKI kernel can be used for router computation.

    Kernel constraints from router_topk_kernel_nki:
    - K <= 8
    - T <= 128 or (T <= 2048 and T % 128 == 0)
    - E <= 512
    - (H % 128) == 0
    - Activation must be "softmax" or "sigmoid" (string only)
    - No RMSNorm support (gamma must be None)
    - Device must be XLA (Neuron)
    - router_bias must be None or shape [E]
    - Only PRENORM_LINEAR_TOPK_ACT_SCATTER and PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER computation orders supported

    Returns:
        bool: True if kernel can be used, False otherwise
    """

    # TODO: Remove this after debugging compilation issue on TRN3
    return False

    if not can_run_kernel(hidden_states):
        return False

    if transposed_hidden_states:
        H, T = hidden_states.shape
    else:
        T, H = hidden_states.shape
    E = router_weights.shape[1]

    if top_k > 8 or E > 512 or H % 128 != 0:
        return False

    # TODO: Remove T <= 2048 requirements when NKILIB-618 is resolved
    if T > 128 and (T > 2048 or T % 128 != 0):
        return False

    if not isinstance(activation, str) or activation not in ["softmax", "sigmoid"]:
        return False

    if gamma is not None:
        return False

    # Bias shape validation
    if router_bias is not None and router_bias.shape != (E,):
        return False

    # Only PRENORM_LINEAR_TOPK_ACT_SCATTER and PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER are supported by kernel
    # PRENORM_LINEAR_TOPK_SCATTER_ACT requires PyTorch fallback
    if (
        router_computation_order
        == RouterComputationOrder.PRENORM_LINEAR_TOPK_SCATTER_ACT
    ):
        return False

    return True
