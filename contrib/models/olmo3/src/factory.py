# SPDX-License-Identifier: Apache-2.0
"""Factory for OLMo-3 model selection based on platform and configuration."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

_SUPPORTED_LAYER_TYPES = ("sliding_attention", "full_attention")


class Olmo3ForCausalLM(nn.Module):
    """Factory that validates the config and selects the OLMo-3 implementation.

    This class extends nn.Module to satisfy the vLLM ModelRegistry requirements.
    The factory keeps the selected implementation and sends forward() calls to it.
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
        *,
        vllm_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *args,
        **kwargs,
    ):
        """Send the forward pass to the selected implementation."""
        return self._model(*args, input_ids=input_ids, positions=positions, **kwargs)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Create the model from the configs. Return the implementation directly."""
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Validate the config, then make the implementation."""
        cls._validate_config(hf_config, neuron_config)

        from .model import Olmo3ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Validate that the configuration is supported.

        The model raises the TP divisibility errors when it builds the layers,
        because only the TP group gives the TP degree.
        """
        torch_dtype = getattr(hf_config, "torch_dtype", None) or getattr(
            hf_config, "dtype", None
        )
        if torch_dtype is not None and torch_dtype not in (
            torch.bfloat16,
            "bfloat16",
        ):
            raise ValueError(
                f"OLMo-3 dtype {torch_dtype!r} is not supported. "
                f"Use a bfloat16 checkpoint."
            )

        if getattr(hf_config, "attention_bias", False):
            raise ValueError(
                "OLMo-3 attention_bias=True is not supported. "
                "Use a checkpoint without attention bias."
            )

        layer_types = getattr(hf_config, "layer_types", None)
        num_hidden_layers = getattr(hf_config, "num_hidden_layers", None)
        if layer_types is not None:
            if num_hidden_layers is not None and len(layer_types) != num_hidden_layers:
                raise ValueError(
                    f"layer_types length ({len(layer_types)}) is not equal to "
                    f"num_hidden_layers ({num_hidden_layers}). "
                    f"Give one layer type for each decoder layer."
                )
            unknown = sorted(set(layer_types) - set(_SUPPORTED_LAYER_TYPES))
            if unknown:
                raise ValueError(
                    f"OLMo-3 layer types {unknown} are not supported. "
                    f"Use only {list(_SUPPORTED_LAYER_TYPES)}."
                )
