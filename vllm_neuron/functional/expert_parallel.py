# SPDX-License-Identifier: Apache-2.0
"""
Expert parallelism utilities for MoE models.

This module provides utilities for distributing experts across ranks in expert parallelism.
"""

import torch


def calculate_local_expert_indices(
    ep_rank: int | torch.Tensor,
    ep_degree: int,
    total_num_experts: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Calculate the local expert indices for a given EP rank. This function assigns experts in
    contiguous order.

    For example, with ep_rank=1, ep_degree=4, total_num_experts=8, this returns experts [2, 3].

    Args:
        ep_rank: The EP rank for the local rank, can be int or tensor
        ep_degree: Expert parallelism degree (number of EP groups)
        total_num_experts: Total number of experts
        device: Device to place the result tensor on

    Returns:
        A tensor containing the local expert indices.

    Example:
        >>> import torch
        >>> import vllm_neuron.functional as NF
        >>>
        >>> ep_rank = 1
        >>> ep_degree = 4
        >>> total_num_experts = 8
        >>>
        >>> # Calculate local expert indices for the local EP rank
        >>> local_expert_indices = NF.calculate_local_expert_indices(ep_rank, ep_degree, total_num_experts, device=torch.device("cpu"))
    """
    experts_per_rank = total_num_experts // ep_degree

    # Use torch.arange with fixed size and add offset
    local_indices = torch.arange(experts_per_rank, dtype=torch.long, device=device)
    start_idx = ep_rank * experts_per_rank

    return local_indices + start_idx


def validate_expert_parallelism_config(
    ep_degree: int, num_experts: int, world_size: int
) -> None:
    """
    Validate expert parallelism configuration.

    Args:
        ep_degree: Expert parallelism degree
        num_experts: Total number of experts
        world_size: Number of available ranks

    Raises:
        ValueError: If configuration is invalid

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> import vllm_neuron.functional as NF
        >>>
        >>> ep_degree = 8
        >>> num_experts = 128
        >>> world_size = dist.get_world_size()
        >>>
        >>> # Validate this configuration is valid for EP
        >>> ep_rank = NF.validate_expert_parallelism_config(ep_degree, num_experts, world_size)
    """
    if not isinstance(ep_degree, int) or ep_degree < 1:
        raise ValueError(f"ep_degree must be a positive integer, got {ep_degree}")

    if not isinstance(num_experts, int) or num_experts < 1:
        raise ValueError(f"num_experts must be a positive integer, got {num_experts}")

    if ep_degree > world_size:
        raise ValueError(
            f"ep_degree ({ep_degree}) cannot exceed world_size ({world_size})"
        )

    if world_size % ep_degree != 0:
        raise ValueError(
            f"World size ({world_size}) must be divisible by "
            f"expert parallelism degree ({ep_degree})"
        )

    if num_experts % ep_degree != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be evenly divisible by "
            f"ep_degree ({ep_degree})"
        )


def get_local_expert_affinities(
    expert_affinities: torch.Tensor, local_expert_indices: torch.Tensor
) -> torch.Tensor:
    """
    Map global routing weights to local expert indices.

    Args:
        expert_affinities: Tensor of shape [T, num_global_experts]
        local_expert_indices: Tensor of expert indices assigned to this rank

    Returns:
        Tensor of shape [T, num_experts] with routing weights for local experts

    Raises:
        ValueError: If input validation fails

    Example:
        >>> import torch
        >>> import vllm_neuron.functional as NF
        >>>
        >>> num_total_experts = 128
        >>> nun_local_experts = 8
        >>> num_tokens = 512
        >>> local_expert_indices = torch.arange(0, 8, dtype=torch.long, device="cpu")
        >>>
        >>> # Calculate scores for all experts
        >>> router_logits = torch.randn((num_tokens, num_total_experts), dtype=torch.bfloat16)
        >>> router_top_value, router_indices = torch.topk(router_logits, top_k, dim=-1)
        >>> router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=torch.float32)
        >>>
        >>> # Get local expert affinities
        >>> full_expert_affinities = torch.zeros(num_tokens, num_total_experts, device="cpu", dtype=torch.float32).scatter_(
        >>>     1, router_indices, router_top_value
        >>> )
        >>> local_expert_affinities = NF.get_local_expert_affinities(
        >>>     full_expert_affinities, local_expert_indices
        >>> )
        >>> # Returns the local expert scores with shape [T, E_local]
    """
    if expert_affinities.dim() != 2:
        raise ValueError(
            f"expert_affinities must be 2D tensor, got {expert_affinities.dim()}D"
        )

    if local_expert_indices.numel() == 0:
        raise ValueError("local_expert_indices cannot be empty")

    # Select the routing weights of the local experts: [T, num_experts].
    local_expert_affinities = torch.index_select(
        expert_affinities, dim=1, index=local_expert_indices
    )

    return local_expert_affinities
