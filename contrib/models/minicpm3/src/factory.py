# SPDX-License-Identifier: Apache-2.0
"""Factory for MiniCPM3 model selection based on platform and configuration."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class MiniCPM3ForCausalLM(nn.Module):
    """Factory that validates config and selects the MiniCPM3 implementation.

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

        from .model import MiniCPM3ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Validate that the configuration is supported.

        Stop at construction with a clear error. A configuration error must not
        show as a shape mismatch in a kernel.
        """
        # Only the BF16 dense variant is implemented.
        torch_dtype = getattr(hf_config, "torch_dtype", None) or getattr(
            hf_config, "dtype", None
        )
        if torch_dtype is not None and torch_dtype not in (
            torch.bfloat16,
            "bfloat16",
        ):
            raise ValueError(
                f"MiniCPM3 on Neuron supports only bfloat16, got {torch_dtype!r}. "
                f"Set --dtype bfloat16."
            )

        # This port needs a low-rank Q projection (q_a_proj and q_b_proj).
        if getattr(hf_config, "q_lora_rank", 768) is None:
            raise ValueError(
                "MiniCPM3 on Neuron needs q_lora_rank. A checkpoint without the "
                "low-rank Q projection is not supported."
            )

        # The checkpoint has no bias on the attention or MLP projections.
        for name in ("attention_bias", "mlp_bias"):
            if getattr(hf_config, name, False):
                raise ValueError(
                    f"MiniCPM3 on Neuron does not support {name}=True. "
                    f"Use a checkpoint without projection bias."
                )
