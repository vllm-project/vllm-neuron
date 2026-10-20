# SPDX-License-Identifier: Apache-2.0
"""
Builder for MPMD RowParallelLinear model.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from base import ModelBuilder
from mpmd_rpl import MPMDRowParallelLinear


class MPMDRPLBuilder(ModelBuilder):
    """
    Builder for MPMD RowParallelLinear model.

    This builder handles the creation of MPMD RowParallelLinear models,
    their reference implementations, checkpoints, and test inputs.

    MPMD (Multiple Program Multiple Data) approach: Only rank 0 has bias
    to avoid duplication during all-reduce operations.
    """

    def __init__(
        self, in_features: int = 128, out_features: int = 512, batch_size: int = 2
    ):
        """
        Initialize RPLBuilder with model configuration.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            batch_size: Batch size for test inputs
        """
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size

    def build_model(self) -> nn.Module:
        """Build the distributed MPMD RowParallelLinear model."""
        return MPMDRowParallelLinear(self.in_features, self.out_features)

    def build_reference(self) -> nn.Module:
        """Build the reference nn.Linear model."""
        return nn.Linear(self.in_features, self.out_features)

    def create_checkpoint(self) -> dict[str, torch.Tensor]:
        """Create a checkpoint compatible with both RPL and Linear models."""
        return {
            "weight": torch.randn(self.out_features, self.in_features),
            "bias": torch.randn(self.out_features),
        }

    def create_inputs(self, num_steps: int = 3) -> list[torch.Tensor]:
        """
        Create test input tensors.

        For RPL, we create full inputs that will be sharded during distributed execution.
        The runner will handle the sharding based on rank.
        """
        return [
            torch.randn(self.batch_size, self.in_features) for _ in range(num_steps)
        ]

    def shard_input(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Shard input tensor along the last dimension for the current rank.

        Args:
            input_tensor: Full input tensor to be sharded

        Returns:
            torch.Tensor: Sharded input for current rank
        """
        # Query current distributed state dynamically
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            return input_tensor  # No sharding needed

        in_features_per_rank = self.in_features // world_size
        start_idx = rank * in_features_per_rank
        end_idx = start_idx + in_features_per_rank
        return input_tensor[:, start_idx:end_idx]

    @property
    def name(self) -> str:
        """Return the model name."""
        return "MPMD RowParallelLinear"
