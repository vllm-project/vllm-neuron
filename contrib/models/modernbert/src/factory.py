# SPDX-License-Identifier: Apache-2.0
"""Factory for the ModernBERT model selection."""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class ModernBertModel(nn.Module):
    """Factory that validates the config and selects the ModernBERT implementation.

    The HF ``architectures`` field of a ModernBERT embedding checkpoint is
    ``ModernBertModel``. The only implementation is the pooling (embedding)
    model. This class extends nn.Module to satisfy vLLM's ModelRegistry.
    """

    # vLLM reads these class attributes from the registered class.
    is_pooling_model = True
    default_seq_pooling_type = "CLS"
    attn_type = "encoder_only"

    def __init__(
        self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)
        from .model import ModernBertForEmbedding as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Validate that the configuration is supported.

        The port supports unquantized ModernBERT checkpoints only.
        """
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16"):
            raise ValueError(
                f"quantization={quantization!r} is not supported for ModernBERT. "
                "Use an unquantized checkpoint (None or 'bf16')."
            )
        quant_cfg = getattr(hf_config, "quantization_config", None)
        if quant_cfg is None and hasattr(hf_config, "to_dict"):
            quant_cfg = hf_config.to_dict().get("quantization_config")
        if quant_cfg is not None:
            raise ValueError(
                "A quantized ModernBERT checkpoint is not supported. "
                "Use an unquantized checkpoint."
            )
