# SPDX-License-Identifier: Apache-2.0
"""Factory for DeepSeek-V4.1-Flash (timeshare) model selection."""

from __future__ import annotations

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class DeepseekV41ForCausalLM(nn.Module):
    """Validates config and selects the DeepSeek-V4.1-Flash implementation.

        This model does not support `--runner generate`
    """

    def __init__(
        self,
        hf_config: PretrainedConfig | None = None,
        neuron_config: NeuronConfig | None = None,
        *,
        vllm_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hf_config is None:
            if vllm_config is None:
                raise ValueError(
                    "DeepseekV41ForCausalLM needs either hf_config (the Neuron "
                    "path) or vllm_config."
                )
            hf_config = vllm_config.model_config.hf_config
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(
        self,
        input_ids=None,
        positions=None,
        *args,
        **kwargs,
    ):
        return self._model(input_ids, positions, *args, **kwargs)

    def embed_input_ids(self, input_ids):
        """Token embeddings, as vLLM's `VllmModel` protocol names them."""
        return self._model.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states):
        """Unembedding, as vLLM's `VllmModelForTextGeneration` names it."""
        return self._model.compute_logits(hidden_states)

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

        from .model import DeepseekV41ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Reject configurations this port cannot serve, with a clear reason."""
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "fp8"):
            raise ValueError(
                f"quantization={quantization!r} is not supported for "
                "DeepSeek-V4.1-Flash (timeshare). The checkpoint stores "
                "attention in fp8 (ue8m0 scales, 32x32 blocks) and expert "
                "weights in fp4; pass quantization=None or 'fp8'."
            )

        if neuron_config is not None and getattr(
            neuron_config, "mm_encoder_only", False
        ):
            raise ValueError(
                "DeepSeek-V4.1-Flash (timeshare) is text-only: the vision tower "
                "is off the tokens -> logits path and is not placed by the "
                "floorplan. Serve this checkpoint as a causal LM."
            )
