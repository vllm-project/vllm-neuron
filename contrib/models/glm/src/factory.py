# SPDX-License-Identifier: Apache-2.0
"""Factory for GLM model selection based on platform and configuration."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


def _get(hf_config, name, default=None):
    if isinstance(hf_config, dict):
        return hf_config.get(name, default)
    return getattr(hf_config, name, default)


class GlmForCausalLM(nn.Module):
    """Factory that validates the config and selects the GLM implementation.

    This class extends nn.Module to satisfy the vLLM ModelRegistry requirements.
    The factory keeps the selected implementation and sends forward() calls to it.

    Example::

        model = GlmForCausalLM.from_configs(hf_config, neuron_config)
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
        """Create the model from the configs. Return the selected implementation.

        Example::

            model = GlmForCausalLM.from_configs(hf_config, neuron_config)
        """
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Validate the config, then make the implementation."""
        cls._validate_config(hf_config, neuron_config)

        from .model import GlmForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Make sure that this implementation supports the configuration.

        Stop at construction with a clear error. Without this check, a bad
        configuration shows as a shape mismatch in the kernels.
        """
        # Only the BF16 dense variant has an implementation.
        torch_dtype = _get(hf_config, "torch_dtype") or _get(hf_config, "dtype")
        if torch_dtype is not None and torch_dtype not in (
            torch.bfloat16,
            "bfloat16",
        ):
            raise ValueError(
                f"GLM on Neuron supports only bfloat16, got {torch_dtype!r}. "
                "Use a bfloat16 checkpoint."
            )

        hidden_act = _get(hf_config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                f"GLM on Neuron supports only the silu activation, got {hidden_act!r}."
            )

        rope_scaling = _get(hf_config, "rope_scaling")
        if isinstance(rope_scaling, dict):
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
            if rope_type not in (None, "default"):
                raise ValueError(
                    f"GLM on Neuron does not support rope_scaling type {rope_type!r}. "
                    "Use a checkpoint with default RoPE."
                )

        num_heads = _get(hf_config, "num_attention_heads")
        hidden_size = _get(hf_config, "hidden_size")
        head_dim = _get(hf_config, "head_dim") or (
            hidden_size // num_heads if hidden_size and num_heads else None
        )
        factor = _get(hf_config, "partial_rotary_factor")
        if factor is None:
            rope_parameters = _get(hf_config, "rope_parameters")
            if isinstance(rope_parameters, dict):
                factor = rope_parameters.get("partial_rotary_factor")
        if factor is None:
            factor = 0.5
        if head_dim is not None:
            rotary_dim = int(head_dim * factor)
            if rotary_dim % 2 != 0 or rotary_dim > head_dim or rotary_dim <= 0:
                raise ValueError(
                    f"Rotary dimension {rotary_dim} is not valid for head_dim "
                    f"{head_dim}. It must be even, positive and not larger than "
                    "head_dim."
                )
