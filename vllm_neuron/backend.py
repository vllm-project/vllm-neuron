# SPDX-License-Identifier: Apache-2.0
"""
Backend implementations for vLLM Neuron plugin.

This module provides the vLLM Neuron backend implementation.

The backend is selected via the VLLM_NEURON_BACKEND environment variable.
"""

import logging
import os
from enum import Enum

logger = logging.getLogger(__name__)


class NeuronBackend(Enum):
    """Supported Neuron backend implementations."""

    VLLM_NEURON = "vllm_neuron"
    NEURON_NATIVE = "neuron_native"


def get_backend() -> NeuronBackend:
    """
    Determine which backend to use based on environment variable.

    Priority:
    1. VLLM_NEURON_BACKEND environment variable (explicit selection)
    2. Default to neuron_native

    Returns:
        NeuronBackend: The backend to use

    Raises:
        ImportError: If vllm_neuron is not available
        ValueError: If an invalid backend is specified
    """
    env_backend = os.environ.get("VLLM_NEURON_BACKEND", "").lower()

    vllm_neuron_available = _is_vllm_neuron_available()

    if env_backend:
        if env_backend == "vllm_neuron":
            if not vllm_neuron_available:
                raise ImportError(
                    "VLLM_NEURON_BACKEND=vllm_neuron but vllm_neuron package is not installed. "
                    "Please install vllm_neuron or change VLLM_NEURON_BACKEND."
                )
            logger.info(
                "Using vllm_neuron backend (explicitly set via VLLM_NEURON_BACKEND)"
            )
            return NeuronBackend.VLLM_NEURON
        elif env_backend == "neuron_native":
            if not vllm_neuron_available:
                raise ImportError(
                    "VLLM_NEURON_BACKEND=neuron_native but vllm_neuron package is not installed. "
                    "Please install vllm_neuron or change VLLM_NEURON_BACKEND."
                )
            logger.info(
                "Using neuron_native backend (explicitly set via VLLM_NEURON_BACKEND)"
            )
            return NeuronBackend.NEURON_NATIVE
        else:
            raise ValueError(
                f"Invalid VLLM_NEURON_BACKEND value: '{env_backend}'. "
                f"Valid options are: 'vllm_neuron', 'neuron_native'"
            )

    # Default to the native route
    if vllm_neuron_available:
        logger.info("Using neuron_native backend (default)")
        return NeuronBackend.NEURON_NATIVE
    else:
        raise ImportError(
            "No Neuron backend package is installed. Please install 'vllm_neuron'."
        )


def _is_vllm_neuron_available() -> bool:
    """vLLM Neuron is always available as it's bundled in this package."""
    return True


def get_platform_class() -> str:
    """
    Get the platform class path based on the selected backend.

    Returns:
        str: Fully qualified class path for the platform
    """
    # vLLM Neuron platform is now in the vllm subfolder
    return "vllm_neuron.vllm.platform.NeuronPlatform"
