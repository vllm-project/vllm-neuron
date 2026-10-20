# SPDX-License-Identifier: Apache-2.0
"""
Logging configuration for vLLM Neuron.

This module provides centralized logging configuration for the vLLM Neuron project.
"""

import logging
import sys

from vllm_neuron import envs


def setup_logging(level=None, format=None):
    """
    Set up logging configuration for vLLM Neuron.

    Args:
        level: Logging level (default: INFO, or from VLLM_NEURON_LOG_LEVEL env var)
        format: Log format string (optional)
    """
    # Determine log level from environment variable or default
    if level is None:
        level_name = envs.VLLM_NEURON_LOG_LEVEL
        level = getattr(logging, level_name, logging.INFO)

    # Default format if none provided
    if format is None:
        format = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Override any existing configuration
    )

    # Set specific levels for vLLM Neuron loggers.
    # Add a dedicated handler to ensure logs are always emitted (even if
    # vLLM reconfigures the root logger). Keep propagate=True so logs also
    # flow through vLLM's root logger and pick up worker prefix formatting
    # (e.g., "(Worker_TP0 pid=...)") when available.
    vllm_neuron_logger = logging.getLogger("vllm_neuron")
    vllm_neuron_logger.setLevel(level)
    if not vllm_neuron_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(format))
        vllm_neuron_logger.addHandler(handler)


def get_logger(name):
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Auto-configure logging when this module is imported
# This ensures logging works by default
if not logging.getLogger().hasHandlers():
    setup_logging()
