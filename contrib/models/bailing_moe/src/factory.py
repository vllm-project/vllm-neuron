# SPDX-License-Identifier: Apache-2.0
"""Factory for BailingMoeV2 model selection based on platform and configuration."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


def _hf_value(hf_config, name: str, default=None):
    """Return the field ``name`` from a PretrainedConfig, a dict or another object."""
    if isinstance(hf_config, dict):
        return hf_config.get(name, default)
    return getattr(hf_config, name, default)


class BailingMoeV2ForCausalLM(nn.Module):
    """Factory that validates config and selects the BailingMoeV2 implementation.

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
        """Select and instantiate the appropriate implementation based on config."""
        cls._validate_config(hf_config, neuron_config)

        from .model import BailingMoeV2ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Validate that the configuration is supported.

        Only the BF16 checkpoint with sigmoid routing, untied embeddings and no
        multi-token-prediction layers has an implementation.
        """
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16"):
            raise ValueError(
                f"quantization={quantization!r} is not supported for BailingMoeV2. "
                "Use None or 'bf16'."
            )
        if _hf_value(hf_config, "quantization_config") is not None:
            raise ValueError(
                "Quantized BailingMoeV2 checkpoints are not supported. "
                "Use the BF16 checkpoint."
            )

        torch_dtype = _hf_value(hf_config, "torch_dtype")
        if torch_dtype is not None and torch_dtype not in (
            torch.bfloat16,
            "bfloat16",
        ):
            raise ValueError(
                f"BailingMoeV2 on Neuron supports bfloat16 only, got {torch_dtype!r}."
            )

        if _hf_value(hf_config, "tie_word_embeddings", False):
            raise ValueError(
                "BailingMoeV2 with tied embeddings is not supported. "
                "Use a checkpoint that has a separate lm_head.weight."
            )

        score_function = _hf_value(hf_config, "score_function", "sigmoid")
        if score_function != "sigmoid":
            raise ValueError(
                f"BailingMoeV2 router score_function={score_function!r} is not "
                "supported. Use a checkpoint with sigmoid routing."
            )

        if _hf_value(hf_config, "num_nextn_predict_layers", 0):
            raise ValueError(
                "BailingMoeV2 multi-token-prediction layers are not supported. "
                "Use a checkpoint with num_nextn_predict_layers=0."
            )

        if _hf_value(hf_config, "use_bias", False) or _hf_value(
            hf_config, "use_qkv_bias", False
        ):
            raise ValueError(
                "BailingMoeV2 with linear bias is not supported. "
                "Use a checkpoint with use_bias=False and use_qkv_bias=False."
            )

        num_experts = _hf_value(hf_config, "num_experts")
        n_group = _hf_value(hf_config, "n_group", 1)
        if num_experts is not None and n_group and num_experts % n_group != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must divide by n_group ({n_group})."
            )
