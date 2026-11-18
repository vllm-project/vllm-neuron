# SPDX-License-Identifier: Apache-2.0
"""Factory for OLMo2 model selection based on platform and configuration."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

_SUPPORTED_CHECKPOINT_DTYPES = (torch.float32, torch.bfloat16, "float32", "bfloat16")


class Olmo2ForCausalLM(nn.Module):
    """Factory that validates the config and selects the OLMo2 implementation.

    This class extends nn.Module to satisfy the vLLM ModelRegistry requirements.
    The factory keeps the selected implementation and delegates forward() to it.
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
        """Delegate the forward pass to the selected implementation."""
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

        from .model import Olmo2ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Make sure that the implementation supports the configuration.

        Stop at construction with a clear error. A configuration that is not
        supported otherwise gives a shape mismatch deep in the kernels. The
        model checks the tensor parallel size, because only the model knows
        the TP group.
        """
        torch_dtype = getattr(hf_config, "torch_dtype", None)
        if torch_dtype is not None and torch_dtype not in _SUPPORTED_CHECKPOINT_DTYPES:
            raise ValueError(
                f"OLMo2 checkpoint dtype {torch_dtype!r} is not supported. "
                "Use a float32 or bfloat16 checkpoint."
            )

        # A recent transformers can report the default RoPE as a dict.
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if rope_scaling:
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
            if rope_type not in (None, "default"):
                raise ValueError(
                    f"OLMo2 rope_scaling {rope_scaling!r} is not supported. "
                    "Use a checkpoint without rope_scaling."
                )

        if getattr(hf_config, "attention_bias", False):
            raise ValueError(
                "OLMo2 attention_bias=True is not supported. "
                "Use a checkpoint without attention bias."
            )

        if getattr(hf_config, "tie_word_embeddings", False):
            raise ValueError(
                "OLMo2 tie_word_embeddings=True is not supported. "
                "Use a checkpoint with a separate lm_head weight."
            )
