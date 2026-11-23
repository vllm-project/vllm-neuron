# SPDX-License-Identifier: Apache-2.0
"""Factory for Granite model selection based on platform and configuration."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

# Granite checkpoints can store FP32 weights. The model runs these weights in
# bfloat16 (see GraniteConfig).
_SUPPORTED_DTYPES = (
    None,
    torch.bfloat16,
    torch.float32,
    "bfloat16",
    "float32",
)


class GraniteForCausalLM(nn.Module):
    """Factory that validates the config and selects the Granite implementation.

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
        """Validate the config. Then make the Granite implementation."""
        cls._validate_config(hf_config, neuron_config)

        from .model import GraniteForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Make sure that the implementation supports the configuration.

        Stop at construction with a clear error. A configuration that is not
        supported otherwise causes a shape mismatch deep in the kernels.
        """
        torch_dtype = getattr(hf_config, "torch_dtype", None)
        if torch_dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"Granite on Neuron does not support dtype {torch_dtype!r}. "
                "Use a bfloat16 or float32 checkpoint."
            )

        hidden_act = getattr(hf_config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                f"Granite on Neuron does not support hidden_act {hidden_act!r}. "
                "Use a checkpoint with hidden_act 'silu'."
            )

        rope_scaling = getattr(hf_config, "rope_scaling", None)
        rope_parameters = getattr(hf_config, "rope_parameters", None)
        rope_type = None
        if isinstance(rope_scaling, dict):
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        elif isinstance(rope_parameters, dict):
            rope_type = rope_parameters.get("rope_type")
        if rope_type not in (None, "default"):
            raise ValueError(
                f"Granite on Neuron does not support RoPE type {rope_type!r}. "
                "Use a checkpoint with the default RoPE."
            )
