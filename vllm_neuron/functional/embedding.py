# SPDX-License-Identifier: Apache-2.0
import torch.nn.functional as F
from torch import Tensor


def embedding(
    input: Tensor,
    weight: Tensor,
) -> Tensor:
    """
    Embedding lookup API for inference workloads.

    This function performs embedding table lookup for inference. It automatically
    falls back to PyTorch implementation (no kernel support yet).

    Args:
        input: Input token IDs tensor with shape [T]
            where T is the total number of tokens (batch_size * sequence_length)
        weight: Embedding weights tensor with shape [vocab_size, embedding_dim]
            Can be sharded along vocabulary dimension using sharding_weight_loader

    Returns:
        Embedded tokens tensor with shape [T, embedding_dim]

    Raises:
        ValueError: If input parameters have invalid shapes or values

    Usage Examples:
        Basic embedding lookup::

            >>> import torch
            >>> from vllm_neuron.functional import embedding
            >>> # Create embedding weights [vocab_size=100, embedding_dim=64]
            >>> weight = torch.randn(100, 64)
            >>> # Token IDs with shape [T=8]
            >>> input_ids = torch.tensor([1, 5, 12, 0, 42, 99, 3, 7])
            >>> # Perform embedding lookup
            >>> output = embedding(input_ids, weight)
            >>> output.shape
            torch.Size([8, 64])
    """
    # Check if kernel can be used
    can_use_kernel = _can_use_kernel()

    # Validate inputs
    _validate_embedding_inputs(input, weight)

    if can_use_kernel:
        # TODO: Add kernel support
        raise NotImplementedError("Kernel implementation not available")
    else:
        # PyTorch fallback implementation
        return _torch_embedding_impl(input, weight)


def _torch_embedding_impl(
    input: Tensor,
    weight: Tensor,
) -> Tensor:
    """
    PyTorch implementation of embedding lookup.

    This function performs standard embedding table lookup using PyTorch's
    F.embedding function. It handles the flattened token dimension [T]
    and preserves the output shape [T, embedding_dim].

    Args:
        input: Token IDs tensor with shape [T]
        weight: Embedding weights tensor with shape [vocab_size, embedding_dim]

    Returns:
        Embedded tokens tensor with shape [T, embedding_dim]
    """
    # Use PyTorch's embedding function directly
    # input: [T], weight: [vocab_size, embedding_dim] -> output: [T, embedding_dim]
    return F.embedding(input, weight)


def _validate_embedding_inputs(
    input: Tensor,
    weight: Tensor,
) -> None:
    """
    Validate input parameters for embedding function.

    This function performs comprehensive input validation for the embedding function,
    ensuring all tensors have correct dimensions and shapes, and that parameters
    are within valid ranges.

    Args:
        input: Input token IDs tensor, expected shape [T]
        weight: Embedding weights tensor, expected shape [vocab_size, embedding_dim]

    Raises:
        ValueError: If any input parameter has invalid shape or value
    """
    # Validate input dimensions
    if input.dim() != 1:
        raise ValueError(f"Expected input to be 1D [T], got shape {input.shape}")

    # Validate weight dimensions
    if weight.dim() != 2:
        raise ValueError(
            f"Expected weight to be 2D [vocab_size, embedding_dim], got shape {weight.shape}"
        )

    # Validate input dtype (must be integer type for indexing)
    if input.is_floating_point():
        raise ValueError(f"Expected input dtype to be integer, got {input.dtype}")


def _can_use_kernel() -> bool:
    """
    Check if the NKI kernel can be used for embedding computation.

    Returns:
        bool: Always False, triggering torch fallback.
    """
    return False
