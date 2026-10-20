# SPDX-License-Identifier: Apache-2.0
"""
Builder for MLP4LayersTP model.
"""

import torch
import torch.nn as nn
from base import ModelBuilder
from mlp_4layer_tp import MLP4LayersTP


class MLP4LayerTPBuilder(ModelBuilder):
    """
    Builder for MLP4LayersTP model.

    This builder handles the creation of 4-layer MLP models with tensor parallelism,
    their reference implementations, checkpoints, and test inputs.

    Architecture: 4 layers of [CPL → SiLU → RPL]
    - Input: 128 features
    - Hidden: 512 features (all intermediate layers)
    - Output: 128 features
    """

    def __init__(
        self,
        input_features: int = 128,
        hidden_features: int = 512,
        output_features: int = 128,
        batch_size: int = 2,
        num_layers: int = 4,
    ):
        """
        Initialize MLP4LayerTPBuilder with model configuration.

        Args:
            input_features: Input feature dimension
            hidden_features: Hidden feature dimension for all intermediate layers
            output_features: Output feature dimension
            batch_size: Batch size for test inputs
            num_layers: Number of MLP layers
        """
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        self.batch_size = batch_size
        self.num_layers = num_layers

    def build_model(self) -> nn.Module:
        """Build the distributed MLP4LayersTP model."""
        return MLP4LayersTP(
            input_features=self.input_features,
            hidden_features=self.hidden_features,
            output_features=self.output_features,
            num_layers=self.num_layers,
        )

    def build_reference(self) -> nn.Module:
        """Build the reference (non-distributed) model using standard PyTorch layers."""
        layers = []

        for i in range(self.num_layers):
            # Determine input and output dimensions for each layer
            if i == 0:
                # First layer: input_features → hidden_features → hidden_features
                in_feat = self.input_features
                hidden_feat = self.hidden_features
                out_feat = self.hidden_features
            elif i == self.num_layers - 1:
                # Last layer: hidden_features → hidden_features → output_features
                in_feat = self.hidden_features
                hidden_feat = self.hidden_features
                out_feat = self.output_features
            else:
                # Middle layers: hidden_features → hidden_features → hidden_features
                in_feat = self.hidden_features
                hidden_feat = self.hidden_features
                out_feat = self.hidden_features

            # Add CPL → SiLU → RPL pattern for each layer
            layers.extend(
                [
                    nn.Linear(in_feat, hidden_feat),  # CPL equivalent
                    nn.SiLU(),  # SiLU activation
                    nn.Linear(hidden_feat, out_feat),  # RPL equivalent
                ]
            )

        return nn.Sequential(*layers)

    def create_checkpoint(self) -> dict[str, torch.Tensor]:
        """
        Create a checkpoint compatible with both MLP4LayersTP and reference models.

        The checkpoint contains weights and biases for all linear layers in the TPLayer structure.
        """
        checkpoint = {}

        for i in range(self.num_layers):
            # Determine dimensions for this layer
            if i == 0:
                # First layer dimensions
                cpl_in_features = self.input_features
                cpl_out_features = self.hidden_features
                rpl_in_features = self.hidden_features
                rpl_out_features = self.hidden_features
            elif i == self.num_layers - 1:
                # Last layer dimensions
                cpl_in_features = self.hidden_features
                cpl_out_features = self.hidden_features
                rpl_in_features = self.hidden_features
                rpl_out_features = self.output_features
            else:
                # Middle layer dimensions
                cpl_in_features = self.hidden_features
                cpl_out_features = self.hidden_features
                rpl_in_features = self.hidden_features
                rpl_out_features = self.hidden_features

            # Create checkpoint entries for TPLayer structure: layers.i.cpl and layers.i.rpl
            checkpoint[f"layers.{i}.cpl.weight"] = torch.randn(
                cpl_out_features, cpl_in_features
            )
            checkpoint[f"layers.{i}.cpl.bias"] = torch.randn(cpl_out_features)
            checkpoint[f"layers.{i}.rpl.weight"] = torch.randn(
                rpl_out_features, rpl_in_features
            )
            checkpoint[f"layers.{i}.rpl.bias"] = torch.randn(rpl_out_features)

        return checkpoint

    def create_inputs(self, num_steps: int = 3) -> list[torch.Tensor]:
        """
        Create test input tensors.

        Args:
            num_steps: Number of test input samples to generate

        Returns:
            List of input tensors of shape (batch_size, input_features)
        """
        return [
            torch.randn(self.batch_size, self.input_features) for _ in range(num_steps)
        ]

    def shard_input(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Shard input tensor along the last dimension for the current rank.

        This is needed for the first layer's RPL which expects sharded inputs.

        Args:
            input_tensor: Full input tensor to be sharded

        Returns:
            torch.Tensor: Input tensor (no sharding needed for first CPL)
        """
        # For MLP4LayersTP, the first layer is CPL which takes full inputs
        # No sharding is needed at the input level
        return input_tensor

    def _create_reference_checkpoint_mapping(
        self, checkpoint: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Create a checkpoint mapping for the reference Sequential model.

        Maps from MLP4LayersTP TPLayer parameter names to Sequential model parameter names.
        """
        reference_checkpoint = {}

        for i in range(self.num_layers):
            # Each layer has 3 components in Sequential: Linear (CPL), SiLU, Linear (RPL)
            # So layer i starts at sequential index i*3
            seq_idx = i * 3

            # Map TPLayer checkpoint keys to Sequential model keys
            reference_checkpoint[f"{seq_idx}.weight"] = checkpoint[
                f"layers.{i}.cpl.weight"
            ]
            reference_checkpoint[f"{seq_idx}.bias"] = checkpoint[f"layers.{i}.cpl.bias"]
            reference_checkpoint[f"{seq_idx + 2}.weight"] = checkpoint[
                f"layers.{i}.rpl.weight"
            ]
            reference_checkpoint[f"{seq_idx + 2}.bias"] = checkpoint[
                f"layers.{i}.rpl.bias"
            ]

        return reference_checkpoint

    @property
    def name(self) -> str:
        """Return the model name."""
        return f"MLP {self.num_layers}-Layer with Tensor Parallelism"
