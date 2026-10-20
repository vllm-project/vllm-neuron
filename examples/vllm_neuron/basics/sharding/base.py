# SPDX-License-Identifier: Apache-2.0
"""
Abstract base class for model builders.

This module defines the interface that all model builders must implement.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ModelBuilder(ABC):
    """
    Abstract base class for model builders.

    Each model type should implement a concrete builder that handles:
    - Model instantiation (both distributed and reference versions)
    - Checkpoint generation with proper structure
    - Input data generation

    This pattern scales well for complex models with multiple layers/modules.
    """

    @abstractmethod
    def build_model(self) -> nn.Module:
        """
        Build and return the distributed model instance.

        Returns:
            nn.Module: The distributed model to be tested
        """
        pass

    @abstractmethod
    def build_reference(self) -> nn.Module:
        """
        Build and return the reference (non-distributed) model instance.

        This is used for comparison to verify distributed model correctness.

        Returns:
            nn.Module: The reference model (e.g., standard nn.Linear)
        """
        pass

    @abstractmethod
    def create_checkpoint(self) -> dict[str, torch.Tensor]:
        """
        Create a checkpoint dictionary compatible with both models.

        For complex models, this can return nested dictionaries with
        state_dicts for multiple modules/layers.

        Returns:
            Dict[str, torch.Tensor]: Checkpoint state dictionary
        """
        pass

    @abstractmethod
    def create_inputs(self, num_steps: int = 3) -> list[torch.Tensor]:
        """
        Create input tensors for testing.

        Args:
            num_steps: Number of test input samples to generate

        Returns:
            List[torch.Tensor]: List of input tensors
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the human-readable name of this model."""
        pass
