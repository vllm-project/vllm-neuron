# SPDX-License-Identifier: Apache-2.0
import logging
import math

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._functional_collectives import (
    all_gather_tensor,
    all_reduce,
    reduce_scatter_tensor,
)

from vllm_neuron.envs import is_native_backend
from vllm_neuron.functional.embedding import embedding
from vllm_neuron.utils.weight_loader import (
    set_weight_loader,
    sharding_weight_loader,
    SafetensorsWeightLoader,
)

logger = logging.getLogger(__name__)


class VocabDimShardedEmbedding(nn.Module):
    """
    Vocabulary-dimension sharded embedding layer for tensor parallelism.

    This embedding layer shards the embedding table along the vocabulary dimension,
    distributing vocabulary entries across multiple tensor parallel ranks. Each rank
    handles a contiguous portion of the vocabulary, and the embeddings are combined
    using collective operations.

    The sharding strategy works as follows:
    - The vocabulary is divided into contiguous chunks across ranks
    - Each rank stores embeddings for its vocabulary range [vocab_start_idx, vocab_end_idx)
    - During forward pass, each rank computes embeddings only for tokens in its range
    - Results are combined via reduce_scatter (sequence parallelism) or all_reduce (tensor parallelism)

    Args:
        vocab_size: Total vocabulary size across all ranks.
        embed_dim: Embedding dimension (not sharded, full dimension on each rank).
        device: Device to place the embedding weights on.
        dtype: Data type for embedding weights.
        tp_group: Tensor parallel process group. If None, uses the world group.

    Example:
        >>> # Create vocabulary-sharded embedding layer
        >>> embedding = VocabDimShardedEmbedding(
        ...     vocab_size=32000,
        ...     embed_dim=4096,
        ...     dtype=torch.bfloat16,
        ... )
        >>>
        >>> # Forward pass with input tokens [T]
        >>> input_ids = torch.tensor([1, 100, 500, 1000], dtype=torch.long)
        >>> # With sequence parallelism (reduce_scatter)
        >>> output = embedding(input_ids, scatter_tokens=True)  # [T/tp_size, embed_dim]
        >>> # Without sequence parallelism (all_reduce)
        >>> output = embedding(input_ids, scatter_tokens=False)  # [T, embed_dim]
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        device=None,
        dtype=None,
        tp_group=None,
    ):
        """
        Initialize the vocabulary-dimension sharded embedding layer.

        Args:
            vocab_size: Total vocabulary size across all ranks.
            embed_dim: Embedding dimension (full dimension on each rank).
            device: Device to place the embedding weights on.
            dtype: Data type for embedding weights.
            tp_group: Tensor parallel process group. Defaults to world group.
        """
        super().__init__()

        # Use TP group if provided, otherwise fall back to world
        # TODO: Extract this out into a utility function
        if dist.is_initialized():
            self.tp_group = tp_group if tp_group is not None else dist.group.WORLD
            self.tp_size = dist.get_world_size(self.tp_group)
            self.tp_rank = dist.get_rank(self.tp_group)
        else:
            self.tp_group = None
            self.tp_size = 1
            self.tp_rank = 0

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Read the backend at construction, not in the compiled forward.
        # is_native_backend() can log a warning. Dynamo cannot trace the log
        # and makes a graph break. The backend is set before construction.
        self._native = is_native_backend()

        # Vocabulary sharding: each rank handles a portion of vocabulary
        self.vocab_size_per_rank = math.ceil(vocab_size / self.tp_size)

        # Store current rank's values for weight loading and initialization
        self.vocab_start_idx = self.tp_rank * self.vocab_size_per_rank
        self.vocab_end_idx = min(
            self.vocab_start_idx + self.vocab_size_per_rank, vocab_size
        )

        # Actual vocabulary size handled by this rank (excluding padding)
        self.actual_vocab_size_per_rank = max(
            0, self.vocab_end_idx - self.vocab_start_idx
        )

        # Create parameter with sharded vocabulary dimension
        self.weight = nn.Parameter(
            torch.empty(
                (self.vocab_size_per_rank, embed_dim), device=device, dtype=dtype
            )
        )

        self.reset_parameters()

        weight_loader: SafetensorsWeightLoader = sharding_weight_loader(
            shard_dim=0,  # Shard along vocabulary dimension
            shard_size=self.vocab_size_per_rank,
            num_shards=self.tp_size,
            pad_shard=True,
        )
        set_weight_loader(self.weight, weight_loader)

    def reset_parameters(self) -> None:
        """Initialize embedding parameters using normal distribution."""
        # Standard embedding initialization
        nn.init.normal_(self.weight)

        # Initialize padding entries (vocabulary indices beyond actual vocab_size) to zero
        if self.actual_vocab_size_per_rank < self.vocab_size_per_rank:
            with torch.no_grad():
                self.weight[self.actual_vocab_size_per_rank :].fill_(0)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Load parameters from state dict with automatic sharding."""
        weight_key = prefix + "weight"

        if weight_key in state_dict:
            full_weight = state_dict[weight_key]
            # We only shard if the weight is not pre-sharded
            if full_weight.shape != self.weight.data.shape:
                # Extract the actual vocabulary portion for this rank
                start_idx = self.vocab_start_idx
                end_idx = self.vocab_end_idx
                actual_weight = full_weight[start_idx:end_idx]

                # Create padded weight tensor with same shape as self.weight
                padded_weight = torch.zeros_like(self.weight.data)
                padded_weight[: actual_weight.shape[0]] = actual_weight

                state_dict[weight_key] = padded_weight

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        input: torch.Tensor,
        scatter_tokens: bool = True,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through vocabulary sharded embedding.

        Args:
            input: Input token IDs tensor with shape [T] where T is total tokens
            scatter_tokens: Whether to scatter tokens using reduce_scatter for sequence parallelism.
                If True, uses reduce_scatter_tensor for sequence parallel coordination.
                If False, uses all_reduce for tensor parallel coordination.
                Default: True (scatter tokens enabled by default)
            rank: Rank tensor with shape [1].

        Returns:
            Embedded tokens tensor with shape [T, embed_dim] or [T/tp_size, embed_dim] if scatter_tokens=True
        """
        # Validate input dimensions
        if input.dim() != 1:
            raise ValueError(f"Expected input to be 1D [T], got shape {input.shape}")

        if scatter_tokens:
            # Assert T % self.tp_size == 0
            assert input.numel() % self.tp_size == 0, (
                f"Number of tokens ({input.numel()}) must be evenly divisible by tp_size ({self.tp_size}) "
                f"when using sequence parallelism (scatter_tokens=True)"
            )

        # Calculate vocab range based on rank
        if rank is None:
            # Fallback for single-node tests
            vocab_start_idx = self.vocab_start_idx
            vocab_end_idx = self.vocab_end_idx
        else:
            # Calculate from rank tensor
            vocab_start_idx = rank * self.vocab_size_per_rank
            vocab_end_idx = (vocab_start_idx + self.vocab_size_per_rank).clamp(
                max=self.vocab_size
            )

        # Create mask for tokens in this rank's vocabulary range
        input_mask = (input >= vocab_start_idx) & (input < vocab_end_idx)

        # Tokens on other ranks produce negative or too-large local indices.
        local_input = input - vocab_start_idx
        # A collapsed SPMD graph can issue the embedding load before the
        # runtime mask resolves. Bound the index before the load.
        local_input = local_input.clamp(min=0, max=self.vocab_size_per_rank - 1)

        # Set tokens from other ranks to zero.
        local_input = torch.where(input_mask, local_input, 0)

        # Use functional embedding API
        local_output = embedding(local_input, self.weight)

        # Zero out embeddings for tokens not in this rank's range
        local_output = torch.where(
            input_mask.unsqueeze(-1).expand_as(local_output).contiguous(),
            local_output,
            torch.zeros_like(local_output),
        )

        if self.tp_size == 1:
            return local_output

        if scatter_tokens:
            # Sequence parallelism: reduce_scatter along sequence dimension
            return reduce_scatter_tensor(
                local_output, "sum", scatter_dim=0, group=self.tp_group
            )
        else:
            # Tensor parallelism: sum partial embeddings across ranks.
            if self._native:
                # Use the string reduceOp. Dynamo cannot pickle a ReduceOp
                # object into a precompile package. A guard on it breaks the
                # warm start.
                return all_reduce(local_output, reduceOp="sum", group=self.tp_group)
            # XLA route: nothing pickles the ReduceOp. Keep the in-place op that
            # the XLA graph already lowers.
            torch.distributed.all_reduce(
                local_output, op=dist.ReduceOp.SUM, group=self.tp_group
            )
            return local_output


class EmbedDimShardedEmbedding(nn.Module):
    """
    Embedding-dimension sharded embedding layer for tensor parallelism.

    This embedding layer shards the embedding table along the embedding dimension,
    distributing embedding features across multiple tensor parallel ranks. Each rank
    stores and computes a portion of the embedding dimensions, and the full embeddings
    are reconstructed using all_gather.

    The sharding strategy works as follows:
    - The embedding dimension is divided across ranks using ceiling division
    - Each rank stores a weight matrix of shape [vocab_size, ceil(embed_dim/tp_size)]
    - During forward pass, each rank computes partial embeddings
    - Results are gathered via all_gather to reconstruct full embeddings
    - If embed_dim is not evenly divisible by tp_size, padding is used and
      the output is sliced to return exactly [T, embed_dim]

    Args:
        vocab_size: Total vocabulary size (full vocabulary on each rank).
        embed_dim: Total embedding dimension across all ranks.
        device: Device to place the embedding weights on.
        dtype: Data type for embedding weights.
        tp_group: Tensor parallel process group. If None, uses the world group.

    Example:
        >>> # Create embedding-dimension sharded embedding layer
        >>> embedding = EmbedDimShardedEmbedding(
        ...     vocab_size=32000,
        ...     embed_dim=4096,
        ...     dtype=torch.bfloat16,
        ... )
        >>>
        >>> # Forward pass with input tokens [T]
        >>> input_ids = torch.tensor([1, 100, 500, 1000], dtype=torch.long)
        >>> output = embedding(input_ids)  # [T, embed_dim]
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        device=None,
        dtype=None,
        tp_group=None,
    ):
        """
        Initialize the embedding-dimension sharded embedding layer.

        Args:
            vocab_size: Total vocabulary size (full vocabulary on each rank).
            embed_dim: Total embedding dimension across all ranks. When not evenly
                divisible by tp_size, padding is automatically applied.
            device: Device to place the embedding weights on.
            dtype: Data type for embedding weights.
            tp_group: Tensor parallel process group. Defaults to world group.
        """
        super().__init__()

        # Use TP group if provided, otherwise fall back to world
        # TODO: Extract this out into a utility function
        if dist.is_initialized():
            self.tp_group = tp_group if tp_group is not None else dist.group.WORLD
            self.tp_size = dist.get_world_size(self.tp_group)
            self.tp_rank = dist.get_rank(self.tp_group)
        else:
            self.tp_group = None
            self.tp_size = 1
            self.tp_rank = 0

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Embedding dimension sharding: each rank handles a portion of embedding dims
        # Use ceiling division to support cases where embed_dim % tp_size != 0
        self.embed_dim_per_rank = math.ceil(embed_dim / self.tp_size)
        self.embed_start_idx = self.tp_rank * self.embed_dim_per_rank
        self.embed_end_idx = min(
            self.embed_start_idx + self.embed_dim_per_rank, embed_dim
        )

        # Actual embedding dimensions handled by this rank (excluding padding)
        self.actual_embed_dim_per_rank = max(
            0, self.embed_end_idx - self.embed_start_idx
        )

        # Create parameter with sharded embedding dimension
        self.weight = nn.Parameter(
            torch.empty(
                (vocab_size, self.embed_dim_per_rank), device=device, dtype=dtype
            )
        )

        self.reset_parameters()

        # Set weight loader for embedding dimension sharding
        weight_loader: SafetensorsWeightLoader = sharding_weight_loader(
            shard_dim=1,  # Shard along embedding dimension
            shard_size=self.embed_dim_per_rank,
            num_shards=self.tp_size,
            pad_shard=True,
        )
        set_weight_loader(self.weight, weight_loader)

    def reset_parameters(self) -> None:
        """Initialize embedding parameters using normal distribution."""
        # Standard embedding initialization
        nn.init.normal_(self.weight)

        # Initialize padding entries (embedding dims beyond actual embed_dim) to zero
        if self.actual_embed_dim_per_rank < self.embed_dim_per_rank:
            with torch.no_grad():
                self.weight[:, self.actual_embed_dim_per_rank :].fill_(0)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Load parameters from state dict with automatic sharding."""
        weight_key = prefix + "weight"

        if weight_key in state_dict:
            full_weight = state_dict[weight_key]
            # We only shard if the weight is not pre-sharded
            if full_weight.shape != self.weight.data.shape:
                # Extract the actual embedding dimension portion for this rank
                start_idx = self.embed_start_idx
                end_idx = self.embed_end_idx
                actual_weight = full_weight[:, start_idx:end_idx]

                # Create padded weight tensor with same shape as self.weight
                padded_weight = torch.zeros_like(self.weight.data)
                padded_weight[:, : actual_weight.shape[1]] = actual_weight

                state_dict[weight_key] = padded_weight

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through embedding dimension sharded embedding.

        Args:
            input: Input token IDs tensor with shape [T] where T is total tokens

        Returns:
            Embedded tokens tensor with shape [T, embed_dim]
        """
        # Validate input dimensions
        if input.dim() != 1:
            raise ValueError(f"Expected input to be 1D [T], got shape {input.shape}")

        # Use functional embedding API to get partial embeddings
        local_output = embedding(input, self.weight)

        if self.tp_size == 1:
            return local_output

        # Gather along embedding dimension (dim 1)
        gathered_output = all_gather_tensor(
            local_output, gather_dim=1, group=self.tp_group
        )

        # Slice to remove padding if embed_dim is not evenly divisible by tp_size
        # all_gather produces [T, embed_dim_per_rank * tp_size], we need [T, embed_dim]
        if gathered_output.shape[1] != self.embed_dim:
            gathered_output = gathered_output[:, : self.embed_dim]

        return gathered_output
