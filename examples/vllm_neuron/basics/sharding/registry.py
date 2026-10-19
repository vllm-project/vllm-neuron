# SPDX-License-Identifier: Apache-2.0
"""
Model registry for managing available model builders.

This module provides a central registry of all available model builders
and functions to access them.
"""

from base import ModelBuilder
from cpl_builder import CPLBuilder
from mlp_4layer_tp_builder import MLP4LayerTPBuilder
from mpmd_rpl_builder import MPMDRPLBuilder
from spmd_rpl_builder import SPMDRPLBuilder

# ============================================================================
# Model Registry
# ============================================================================

MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "cpl": CPLBuilder(),
    "mpmd_rpl": MPMDRPLBuilder(),
    "spmd_rpl": SPMDRPLBuilder(),
    "mlp_4layer_tp": MLP4LayerTPBuilder(),
    # Add more models here as you create new builders:
    # "cpl_large": CPLBuilder(in_features=1024, out_features=4096),
    # "multi_layer": MultiLayerTransformerBuilder(num_layers=4),
}


def get_model_builder(model_name: str) -> ModelBuilder:
    """
    Get builder for a specific model.

    Args:
        model_name: Name of the model to retrieve

    Returns:
        ModelBuilder: The builder instance for the requested model

    Raises:
        ValueError: If the model name is not found in the registry
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]


def list_available_models() -> dict[str, str]:
    """
    List all available models with their names.

    Returns:
        Dict[str, str]: Dictionary mapping model keys to their display names
    """
    return {key: builder.name for key, builder in MODEL_REGISTRY.items()}
