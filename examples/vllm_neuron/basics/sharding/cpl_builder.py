# SPDX-License-Identifier: Apache-2.0
"""
Builder for ColumnParallelLinear model.
"""

import torch
import torch.nn as nn
from base import ModelBuilder
from cpl import ColumnParallelLinear


class CPLBuilder(ModelBuilder):
    """
    Builder for ColumnParallelLinear model.

    This builder handles the creation of ColumnParallelLinear models,
    their reference implementations, checkpoints, and test inputs.
    """

    def __init__(
        self, in_features: int = 128, out_features: int = 512, batch_size: int = 2
    ):
        """
        Initialize CPLBuilder with model configuration.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            batch_size: Batch size for test inputs
        """
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size

    def build_model(self) -> nn.Module:
        """Build the distributed ColumnParallelLinear model."""
        return ColumnParallelLinear(self.in_features, self.out_features)

    def build_reference(self) -> nn.Module:
        """Build the reference nn.Linear model."""
        return nn.Linear(self.in_features, self.out_features)

    def create_checkpoint(self) -> dict[str, torch.Tensor]:
        """Create a checkpoint compatible with both CPL and Linear models."""
        return {
            "weight": torch.randn(self.out_features, self.in_features),
            "bias": torch.randn(self.out_features),
        }

    def create_inputs(self, num_steps: int = 3) -> list[torch.Tensor]:
        """Create test input tensors."""
        return [
            torch.randn(self.batch_size, self.in_features) for _ in range(num_steps)
        ]

    @property
    def name(self) -> str:
        """Return the model name."""
        return "ColumnParallelLinear"
