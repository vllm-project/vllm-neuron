# SPDX-License-Identifier: Apache-2.0
"""Factory for GLM-4 model selection based on platform and configuration."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Glm4ForCausalLM(nn.Module):
    """Factory that validates the config and selects the GLM-4 implementation.

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
        """Create the model from configs. Return the selected implementation."""
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Validate the config, then make the implementation."""
        cls._validate_config(hf_config, neuron_config)

        from .model import Glm4ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Make sure that the implementation supports the configuration.

        Stop at construction with a clear error. A wrong configuration can
        otherwise cause a shape mismatch deep in the kernels.
        """
        # Only the BF16 dense variant is implemented.
        torch_dtype = getattr(hf_config, "torch_dtype", None)
        if torch_dtype is not None and torch_dtype not in (
            torch.bfloat16,
            "bfloat16",
        ):
            raise ValueError(
                f"GLM-4 on Neuron supports only bfloat16. The config has "
                f"{torch_dtype!r}. Use a bfloat16 checkpoint."
            )

        quantization_config = getattr(hf_config, "quantization_config", None)
        if quantization_config:
            raise ValueError(
                "GLM-4 on Neuron does not support a quantized checkpoint. "
                "Use the bfloat16 checkpoint."
            )

        # The attention has no bias path. GLM-4-0414 checkpoints have no QKV bias.
        if getattr(hf_config, "attention_bias", False):
            raise ValueError(
                "GLM-4 on Neuron does not support attention_bias=True. "
                "Use a GLM-4-0414 checkpoint without a QKV bias."
            )

        # The fused RoPE layout permutes the rotary part of each Q/K head.
        # The rotary part must be an even number of dimensions inside the head.
        head_dim = getattr(hf_config, "head_dim", None) or (
            hf_config.hidden_size // hf_config.num_attention_heads
        )
        factor = getattr(hf_config, "partial_rotary_factor", None)
        if factor is None:
            rope_parameters = getattr(hf_config, "rope_parameters", None) or {}
            factor = rope_parameters.get("partial_rotary_factor", 1.0)
        rotary_dim = int(head_dim * factor)
        if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
            raise ValueError(
                f"Rotary dimension {rotary_dim} is not valid for head_dim "
                f"{head_dim}. Set partial_rotary_factor to give an even value "
                f"from 2 to head_dim."
            )
