# SPDX-License-Identifier: Apache-2.0
"""Factory for HRM-Text model selection based on platform and configuration."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


def _get(hf_config, name: str, default=None):
    if isinstance(hf_config, dict):
        return hf_config.get(name, default)
    return getattr(hf_config, name, default)


class HrmTextForCausalLM(nn.Module):
    """Factory that validates config and selects the HRM-Text implementation.

    This class extends nn.Module to satisfy vLLM's ModelRegistry requirements.
    The factory stores the selected implementation and delegates forward() calls to it.
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
        """Delegate forward pass to the selected implementation."""
        return self._model(*args, input_ids=input_ids, positions=positions, **kwargs)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Create model from configs. Returns the selected implementation directly."""
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Select and instantiate the implementation based on config."""
        cls._validate_config(hf_config, neuron_config)

        from .model import HrmTextForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Reject a configuration that the implementation does not support."""
        torch_dtype = _get(hf_config, "torch_dtype") or _get(hf_config, "dtype")
        if torch_dtype is not None and torch_dtype not in (
            torch.bfloat16,
            "bfloat16",
        ):
            raise ValueError(
                f"HRM-Text on Neuron supports only bfloat16, but the config has "
                f"{torch_dtype!r}. Pass --dtype bfloat16."
            )

        if _get(hf_config, "attention_bias", False) or _get(
            hf_config, "mlp_bias", False
        ):
            raise ValueError(
                "HRM-Text on Neuron supports only projections without bias. "
                "Set attention_bias and mlp_bias to false."
            )

        rope_parameters = _get(hf_config, "rope_parameters") or {}
        rope_type = rope_parameters.get("rope_type", "default")
        if rope_type != "default":
            raise ValueError(
                f"HRM-Text on Neuron supports only the default RoPE, but the "
                f"config has rope_type {rope_type!r}."
            )
