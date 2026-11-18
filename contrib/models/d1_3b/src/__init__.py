# SPDX-License-Identifier: Apache-2.0
"""LiquidAI d1-3B (LFM2 hybrid decision model) for vLLM on Neuron."""

from .factory import ARCHITECTURE, D1ForConditionalGeneration, register

__all__ = ["ARCHITECTURE", "D1ForConditionalGeneration", "register"]
