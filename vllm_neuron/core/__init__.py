# SPDX-License-Identifier: Apache-2.0
"""Core module for vllm_neuron."""

from .scheduler import ContinuousBatchingNeuronScheduler

__all__ = ["ContinuousBatchingNeuronScheduler"]
