# SPDX-License-Identifier: Apache-2.0
"""Factory for EXAONE 4.0 model selection based on platform and configuration."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Exaone4ForCausalLM(nn.Module):
    """Factory that validates the config and selects the EXAONE 4.0 implementation.

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
        """Create the model from the configs. Return the selected implementation."""
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Validate the config, then make the BF16 implementation."""
        cls._validate_config(hf_config, neuron_config)

        from .model import Exaone4ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Make sure that this port supports the configuration.

        The check stops a bad configuration at construction. Otherwise, it
        shows later as a shape mismatch in a kernel.
        """
        torch_dtype = getattr(hf_config, "torch_dtype", None)
        if torch_dtype is not None and torch_dtype not in (
            torch.bfloat16,
            "bfloat16",
        ):
            raise ValueError(
                f"EXAONE 4.0 on Neuron supports only bfloat16, but the dtype is "
                f"{torch_dtype!r}. Use the bfloat16 checkpoint."
            )

        layer_types = getattr(hf_config, "layer_types", None)
        num_hidden_layers = getattr(hf_config, "num_hidden_layers", None)
        if (
            layer_types is not None
            and num_hidden_layers is not None
            and len(layer_types) != num_hidden_layers
        ):
            raise ValueError(
                f"layer_types has {len(layer_types)} entries, but "
                f"num_hidden_layers is {num_hidden_layers}. Supply one entry "
                "for each decoder layer."
            )
