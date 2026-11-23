# SPDX-License-Identifier: Apache-2.0
"""Factory for Qwen3.5 model selection.

Phase 1 has a single implementation (the eager hybrid model). The factory
wrapper mirrors ``qwen3/factory.py`` so the registry entry keeps the same
``from_configs`` shape and a quantization-driven dispatch can be added later
(mxfp8 / NKI) without touching the registry.
"""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Qwen3_5ForCausalLM(nn.Module):
    """Factory that validates config and builds the Qwen3.5 implementation."""

    def __init__(
        self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None = None
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None = None
    ) -> nn.Module:
        return cls._select_implementation(hf_config, neuron_config)

    @staticmethod
    def _select_implementation(
        hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        Qwen3_5ForCausalLM._validate_config(hf_config)

        from .model import Qwen3_5ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @staticmethod
    def _validate_config(hf_config: PretrainedConfig) -> None:
        layer_types = getattr(hf_config, "layer_types", None)
        num_layers = getattr(hf_config, "num_hidden_layers", None)
        if layer_types is None or num_layers is None:
            raise ValueError(
                "Qwen3.5 checkpoint must carry layer_types and num_hidden_layers "
                "to drive the hybrid (full/linear attention) dispatch."
            )
        if len(layer_types) != num_layers:
            raise ValueError(
                f"layer_types length ({len(layer_types)}) != num_hidden_layers "
                f"({num_layers})."
            )
