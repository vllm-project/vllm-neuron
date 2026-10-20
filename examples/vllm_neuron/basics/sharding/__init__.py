# SPDX-License-Identifier: Apache-2.0
"""
Sharding examples package.

This package provides a modular framework for testing distributed models
using the Model Builder pattern.

Main components:
- base: Abstract ModelBuilder base class
- registry: Central model registry
- runner: Test orchestration and execution
- Model implementations: cpl.py, etc.
- Model builders: cpl_builder.py, etc.

Usage:
    python runner.py --model cpl --world_size 2
"""

from base import ModelBuilder
from registry import MODEL_REGISTRY, get_model_builder, list_available_models

__all__ = [
    "ModelBuilder",
    "get_model_builder",
    "list_available_models",
    "MODEL_REGISTRY",
]
