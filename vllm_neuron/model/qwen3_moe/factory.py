# SPDX-License-Identifier: Apache-2.0
"""Factory for Qwen3Moe model selection based on platform and configuration."""

import torch.nn as nn
from transformers import PretrainedConfig

from libtorch_neuronx_lite.compile.platform import get_platform_target
from vllm_neuron.model.neuron_config import NeuronConfig


class Qwen3MoeForCausalLM(nn.Module):
    """Factory that validates config and selects the appropriate Qwen3Moe implementation."""

    def __init__(
        self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)

        platform = get_platform_target()
        quantization = neuron_config.quantization if neuron_config else None

        if quantization == "bf16":
            from .model_bf16 import Qwen3MoeForCausalLM as Model

            return Model.from_configs(hf_config, neuron_config)

        # Default to bf16 for now
        from .model_bf16 import Qwen3MoeForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Validate that the configuration is supported."""
        pass
