# SPDX-License-Identifier: Apache-2.0
"""
4-Layer MLP with Tensor Parallelism implementation.

This implementation creates a 4-layer MLP where each layer consists of:
- Column Parallel Linear (CPL) - splits output features across ranks
- SiLU activation function
- SPMD Row Parallel Linear (RPL) - splits input features across ranks

Architecture:
Input (128) → [CPL → SiLU → RPL] × 4 → Output (128)
Hidden dimensions: 512 for all intermediate layers
"""

import torch.nn as nn
import torch.nn.functional as F
from cpl import ColumnParallelLinear
from spmd_rpl import SPMDRowParallelLinear


class MLPTPLayer(nn.Module):
    """
    Single Tensor Parallel Layer: CPL → SiLU → RPL

    This module encapsulates the pattern of:
    - Column Parallel Linear (splits output features across ranks)
    - SiLU activation function
    - SPMD Row Parallel Linear (splits input features across ranks)

    Args:
        in_features: Input feature dimension
        hidden_features: Hidden feature dimension (output of CPL, input to RPL)
        out_features: Output feature dimension
        bias: Whether to use bias in linear layers
        device: Device to place the layers on
        dtype: Data type for the layers
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.cpl = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            gather_output=False,
            device=device,
            dtype=dtype,
        )
        self.rpl = SPMDRowParallelLinear(
            hidden_features, out_features, bias=bias, device=device, dtype=dtype
        )

    def forward(self, x):
        """
        Forward pass through the tensor parallel layer.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        x = self.cpl(x)
        x = F.silu(x)
        x = self.rpl(x)
        return x


class MLP4LayersTP(nn.Module):
    """
    4-Layer MLP with Tensor Parallelism.

    Each layer follows the pattern: CPL → SiLU → RPL
    - Layer 1: 128 → 512 → 512
    - Layer 2: 512 → 512 → 512
    - Layer 3: 512 → 512 → 512
    - Layer 4: 512 → 512 → 128
    """

    def __init__(
        self,
        input_features: int = 128,
        hidden_features: int = 512,
        output_features: int = 128,
        num_layers: int = 4,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        self.num_layers = num_layers

        # Create layers using ModuleList for proper parameter registration
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            # Determine input and output dimensions for each layer
            if i == 0:
                # First layer: input_features → hidden_features → hidden_features
                in_feat = input_features
                hidden_feat = hidden_features
                out_feat = hidden_features
            elif i == num_layers - 1:
                # Last layer: hidden_features → hidden_features → output_features
                in_feat = hidden_features
                hidden_feat = hidden_features
                out_feat = output_features
            else:
                # Middle layers: hidden_features → hidden_features → hidden_features
                in_feat = hidden_features
                hidden_feat = hidden_features
                out_feat = hidden_features

            # Create MLPTPLayer with appropriate dimensions
            layer = MLPTPLayer(
                in_feat, hidden_feat, out_feat, bias=bias, device=device, dtype=dtype
            )
            self.layers.append(layer)

    def forward(self, x):
        """
        Forward pass through all MLP layers.

        Args:
            x: Input tensor of shape (batch_size, input_features)

        Returns:
            Output tensor of shape (batch_size, output_features)
        """
        for layer in self.layers:
            x = layer(x)
        return x
