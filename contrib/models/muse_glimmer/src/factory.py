# SPDX-License-Identifier: Apache-2.0
"""Factory for MuseGlimmer.

The class is registered with vLLM's ``ModelRegistry`` under
``MuseGlimmerForConditionalGeneration`` (see ``register()`` in ``__init__.py``).
It plays two roles:

* in the Neuron worker, the model runner calls ``from_configs`` to build the
  Neuron implementation (``model_bf16.py``);
* in the front-end and the engine core, vLLM only inspects the class. For that
  it must look like a vLLM model (``__init__(*, vllm_config, prefix)``,
  ``forward(input_ids, positions)``, ``embed_input_ids``, ``compute_logits``)
  and carry the multimodal processor (``processing.py``). vLLM ships its own GPU
  ``MuseGlimmerForCausalLM`` under the same architecture name, whose processor
  emits different multimodal kwargs, so this class must replace it in every
  process.
"""

from __future__ import annotations

import math
from typing import ClassVar, Literal

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_neuron.model.interfaces import (
    SupportsEncoderBudget,
    SupportsMaxPixels,
    SupportsSpatialMerge,
)
from vllm_neuron.model.neuron_config import NeuronConfig, EncoderNeuronConfig

from .processing import (
    IMAGE_TOKEN,
    VIDEO_TOKEN,
    MuseGlimmerDummyInputsBuilder,
    MuseGlimmerMultiModalProcessor,
    MuseGlimmerProcessingInfo,
)

ARCHITECTURE = "MuseGlimmerForConditionalGeneration"


@MULTIMODAL_REGISTRY.register_processor(
    MuseGlimmerMultiModalProcessor,
    info=MuseGlimmerProcessingInfo,
    dummy_inputs=MuseGlimmerDummyInputsBuilder,
)
class MuseGlimmerForConditionalGeneration(
    nn.Module, SupportsSpatialMerge, SupportsMaxPixels, SupportsEncoderBudget
):
    """Validates the config and builds the MuseGlimmer Neuron implementation."""

    # vLLM front-end interface tags (inspected, never instantiated there).
    supports_multimodal: ClassVar[Literal[True]] = True
    supports_multimodal_raw_input_only: ClassVar[bool] = False
    supports_encoder_tp_data: ClassVar[bool] = False
    requires_raw_input_tokens: ClassVar[bool] = False
    is_text_generation_model: ClassVar[bool] = True

    def __init__(self, *, vllm_config=None, prefix: str = "", **kwargs) -> None:
        raise RuntimeError(
            "MuseGlimmerForConditionalGeneration is a factory; the Neuron model "
            "runner builds it with from_configs()."
        )

    # -- VllmModel protocol stubs (front-end inspection only) -------------
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, **kwargs):
        raise RuntimeError("factory stub")

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("factory stub")

    def compute_logits(self, hidden_states: torch.Tensor):
        raise RuntimeError("factory stub")

    def embed_multimodal(self, **kwargs):
        raise RuntimeError("factory stub")

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_TOKEN
        if modality.startswith("video"):
            return VIDEO_TOKEN
        raise ValueError(f"Unsupported modality {modality!r}")

    # -- plugin entry point -----------------------------------------------
    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: EncoderNeuronConfig | None = None,
    ) -> nn.Module:
        cls._validate_config(hf_config, text_neuron_config)
        from .model_bf16 import MuseGlimmerForConditionalGeneration as Model

        return Model.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            encoder_neuron_config=encoder_neuron_config,
        )

    @classmethod
    def get_vision_token_merge_factor(cls, hf_config: PretrainedConfig) -> int:
        return hf_config.vision_config.merge_size**2

    # -- encoder budget (vllm-neuron 0.29 contract) ------------------------
    # Buckets are raw pre-merge patches; merge_size**2 patches -> one embed.
    @classmethod
    def get_max_encoder_embed_budget(cls, hf_config, encoder_neuron_config) -> int | None:
        buckets = encoder_neuron_config.num_encoder_input_buckets
        if not buckets:
            return None
        return max(buckets) // cls.get_vision_token_merge_factor(hf_config)

    @classmethod
    def get_max_encoder_blocks_per_request(cls, hf_config, encoder_neuron_config) -> int:
        from vllm_neuron.utils.mm_utils import get_encoder_cache_block_size

        buckets = encoder_neuron_config.num_encoder_input_buckets
        if not buckets:
            return 0
        mu = cls.get_vision_token_merge_factor(hf_config)
        cache_block_size = get_encoder_cache_block_size(
            hf_config, encoder_neuron_config.encoder_attention_block_size
        )
        return math.ceil((max(buckets) // mu) / cache_block_size)

    @classmethod
    def get_max_pixels_token_count(cls, hf_config: PretrainedConfig, max_pixels: int) -> int:
        patch = hf_config.vision_config.patch_size
        return max_pixels // (patch**2)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, text_neuron_config: NeuronConfig | None
    ) -> None:
        archs = list(getattr(hf_config, "architectures", None) or [])
        if archs and archs[0] != ARCHITECTURE:
            raise ValueError(f"Expected architecture {ARCHITECTURE}, got {archs}")
        quantization = text_neuron_config.quantization if text_neuron_config else None
        if quantization not in (None, "bf16"):
            raise ValueError(
                f"Unknown quantization {quantization!r} for MuseGlimmer. "
                "Expected None or 'bf16'."
            )
