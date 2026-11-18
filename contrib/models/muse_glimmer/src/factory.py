# SPDX-License-Identifier: Apache-2.0
"""vLLM ModelRegistry entry for MuseGlimmerForConditionalGeneration (the HF architecture)."""

import math

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm_neuron.model.neuron_config import NeuronConfig

from .processing import (
    MuseGlimmerNeuronDummyInputs,
    MuseGlimmerNeuronProcessingInfo,
    MuseGlimmerNeuronProcessor,
)


@MULTIMODAL_REGISTRY.register_processor(
    MuseGlimmerNeuronProcessor,
    info=MuseGlimmerNeuronProcessingInfo,
    dummy_inputs=MuseGlimmerNeuronDummyInputs,
)
class MuseGlimmerForConditionalGeneration(nn.Module):
    """Factory: validates config, returns the implementation (text, images, video).

    vLLM inspects the registered class when it builds ModelConfig (in the API process,
    before any Neuron worker exists). Its VllmModelForTextGeneration protocol (vLLM 0.29
    interfaces_base.py) requires: __init__ accepting `vllm_config`, `embed_input_ids`,
    `forward(input_ids, positions, ...)` and `compute_logits`; `supports_multimodal` +
    `get_placeholder_str` mark it multimodal. The Neuron runner never instantiates this
    factory — it calls `from_configs` — so those methods only exist to satisfy the checks.

    The classmethods below are vllm-neuron's SupportsSpatialMerge / SupportsEncoderBudget
    hooks; the runner sizes the encoder cache from them (looked up via the plugin's
    get_models(), which this package extends — see __init__.py).
    """

    is_text_generation_model = True
    supports_multimodal = True

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
        self._model = self.from_configs(hf_config, neuron_config)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return (
                "<|patch|>"  # the chat template's image part; the processor expands it
            )
        if modality.startswith("video"):
            return "<|video|>"
        raise ValueError(
            f"Muse-Glimmer on vLLM Neuron: modality {modality!r} is not supported"
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._model.model.embed_tokens(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, **kwargs):
        return self._model(input_ids=input_ids, positions=positions, **kwargs)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("vllm-neuron runs the LM head inside forward()")

    # ── vllm-neuron encoder sizing hooks ──

    @classmethod
    def get_vision_token_merge_factor(cls, hf_config) -> int:
        return hf_config.vision_config.merge_size**2

    @classmethod
    def get_max_encoder_embed_budget(
        cls, hf_config, encoder_neuron_config
    ) -> int | None:
        buckets = encoder_neuron_config.num_encoder_input_buckets
        return (
            max(buckets) // cls.get_vision_token_merge_factor(hf_config)
            if buckets
            else None
        )

    @classmethod
    def get_max_encoder_blocks_per_request(
        cls, hf_config, encoder_neuron_config
    ) -> int:
        from vllm_neuron.utils.mm_utils import get_encoder_cache_block_size

        buckets = encoder_neuron_config.num_encoder_input_buckets
        if not buckets:
            return 0
        embeds = max(buckets) // cls.get_vision_token_merge_factor(hf_config)
        block = get_encoder_cache_block_size(
            hf_config, encoder_neuron_config.encoder_attention_block_size
        )
        return math.ceil(embeds / block)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None = None,
        *,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config=None,
    ) -> nn.Module:
        # The runner calls (hf_config, neuron_config) for text models and
        # (hf_config=, text_neuron_config=, encoder_neuron_config=) once an encoder config exists.
        if neuron_config is None:
            neuron_config = text_neuron_config
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16"):
            raise ValueError(
                f"quantization={quantization!r} is not supported; Muse-Glimmer runs BF16"
            )
        from .model_bf16 import MuseGlimmerForCausalLM

        return MuseGlimmerForCausalLM.from_configs(
            hf_config, neuron_config, encoder_neuron_config
        )
