# SPDX-License-Identifier: Apache-2.0
"""Factory for the Qwen3.5-family dense hybrid model (Qwen3.6/3.8-27B)."""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Qwen3_5ForCausalLM(nn.Module):
    """Factory that validates config and selects the Qwen3.5 implementation.

    Extends nn.Module to satisfy vLLM's ModelRegistry requirements.
    """

    def __init__(
        self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None = None,
        *,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: object | None = None,
    ) -> nn.Module:
        # The checkpoint is Qwen3_5ForConditionalGeneration, so the runner sees
        # hf_config.vision_config and calls the multimodal form
        # (text_neuron_config=..., encoder_neuron_config=...). This port is the
        # language model only: take the text config and ignore the encoder one.
        if neuron_config is None:
            neuron_config = text_neuron_config
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)
        from .model_bf16 import Qwen3_5ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16"):
            raise ValueError(
                f"quantization={quantization!r} is not supported for Qwen3.5 "
                "(DeltaNet layers require bf16). Only bf16 is implemented."
            )


class Qwen3_5ForConditionalGeneration(Qwen3_5ForCausalLM):
    """Factory for the checkpoint architecture name ``Qwen3_5ForConditionalGeneration``.

    Only the language model is served. The vision tower is not loaded, so serve
    with ``--limit-mm-per-prompt '{"image": 0, "video": 0}'``.
    """
