# SPDX-License-Identifier: Apache-2.0
"""Factory for Gemma3 (text) model selection based on the configuration."""

from typing import Any

import torch
import torch.nn as nn

from vllm_neuron.model.neuron_config import NeuronConfig

from .config import FULL_ATTENTION, Gemma3Config, resolve_layer_types


class Gemma3ForCausalLM(nn.Module):
    """Factory that validates the config and selects the Gemma3 implementation.

    This class extends nn.Module to satisfy the vLLM ModelRegistry. The factory
    keeps the selected implementation and sends forward() calls to it.
    """

    def __init__(
        self,
        hf_config: Any,
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
        cls, hf_config: Any, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Create the model from the configs. Return the selected implementation."""
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: Any, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Validate the config, then make the BF16 implementation."""
        cls._validate_config(hf_config, neuron_config)

        from .model import Gemma3ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: Any, neuron_config: NeuronConfig | None
    ) -> None:
        """Make sure that the configuration is supported.

        Stop at construction with a clear error. Do not let an unsupported
        config continue to a shape mismatch in a kernel.
        """
        config = Gemma3Config.from_configs(hf_config, neuron_config)

        if config.torch_dtype != torch.bfloat16:
            raise ValueError(
                f"Gemma3 on Neuron supports only the bfloat16 served dtype, not "
                f"{config.torch_dtype}. Set --dtype bfloat16."
            )

        # Only the full-attention variant is implemented. A sliding-window
        # layer needs a separate KV spec and a sliding-window mask.
        layer_types = resolve_layer_types(
            {"layer_types": config.layer_types}, config.num_hidden_layers
        )
        other = sorted({t for t in layer_types if t != FULL_ATTENTION})
        if len(layer_types) != config.num_hidden_layers or other:
            raise ValueError(
                f"Gemma3 on Neuron supports only full attention in all "
                f"{config.num_hidden_layers} layers. The config has layer types "
                f"{other}. Use a checkpoint with full attention in all layers."
            )

        if config.attn_logit_softcapping is not None:
            raise ValueError(
                f"attn_logit_softcapping {config.attn_logit_softcapping} is not "
                "supported. The attention kernels have no soft cap. Use a "
                "checkpoint without an attention soft cap."
            )

        if config.hidden_activation not in ("gelu_pytorch_tanh", "gelu_tanh"):
            raise ValueError(
                f"Activation {config.hidden_activation!r} is not supported. "
                "Gemma3 on Neuron supports only gelu_pytorch_tanh."
            )

        if config.attention_bias:
            raise ValueError(
                "attention_bias is not supported. Use a checkpoint without "
                "attention bias."
            )
