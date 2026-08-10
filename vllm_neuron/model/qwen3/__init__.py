# SPDX-License-Identifier: Apache-2.0
"""Qwen3 and Qwen3Moe model support for vLLM-Neuron."""

from .model_bf16 import Qwen3ForCausalLM

__all__ = ["Qwen3ForCausalLM"]
