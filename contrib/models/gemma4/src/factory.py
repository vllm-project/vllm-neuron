# SPDX-License-Identifier: Apache-2.0
"""vLLM ModelRegistry entry for Gemma 4 (text decoder only)."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Gemma4ForConditionalGeneration(nn.Module):
    """Factory: validates config, returns the text-decoder implementation.

    Registered for both ``Gemma4ForConditionalGeneration`` (the HF checkpoint
    architecture) and ``Gemma4ForCausalLM`` (a text-only config). Only the text
    decoder is implemented; vision/audio tower weights are skipped at load time.

    vLLM inspects the registered class when it builds ModelConfig (in the API
    process, before any Neuron worker exists). Its VllmModelForTextGeneration
    protocol (vLLM 0.29 interfaces_base.py) requires ``__init__`` accepting
    ``vllm_config``, ``embed_input_ids``, ``forward(input_ids, positions, ...)``
    and ``compute_logits``. The Neuron runner never instantiates this factory
    (it calls ``from_configs``), so those methods only exist to satisfy the checks.
    """

    is_text_generation_model = True

    def __init__(
        self,
        hf_config: PretrainedConfig | None = None,
        neuron_config: NeuronConfig | None = None,
        *,
        vllm_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hf_config is None and vllm_config is not None:
            hf_config = vllm_config.model_config.hf_config
        self._model = (
            self.from_configs(hf_config, neuron_config) if hf_config is not None else None
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._model.model.embed_tokens(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, *args, **kwargs):
        return self._model(input_ids, positions, *args, **kwargs)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("vllm-neuron runs the LM head inside forward()")

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None = None,
        *,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config=None,
    ) -> nn.Module:
        if neuron_config is None:
            neuron_config = text_neuron_config
        cls._validate_config(hf_config, neuron_config)

        from .model import Gemma4ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        quantization = getattr(neuron_config, "quantization", None) if neuron_config else None
        if quantization not in (None, "bf16"):
            raise ValueError(
                f"Gemma 4 contrib model supports BF16 only, got quantization={quantization!r}"
            )
