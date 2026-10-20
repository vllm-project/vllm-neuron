# SPDX-License-Identifier: Apache-2.0
"""Factory for Qwen3-VL model selection based on platform and configuration."""

import math

import torch.nn as nn
from transformers import PretrainedConfig
from vllm.multimodal.inputs import MultiModalKwargsItem

from vllm_neuron.model.interfaces import (
    SupportsDisaggEncoder,
    SupportsEncoderBudget,
    SupportsMaxPixels,
    SupportsSpatialMerge,
)
from vllm_neuron.model.neuron_config import NeuronConfig, EncoderNeuronConfig


class Qwen3VLForConditionalGeneration(
    nn.Module,
    SupportsSpatialMerge,
    SupportsMaxPixels,
    SupportsDisaggEncoder,
    SupportsEncoderBudget,
):
    """Factory that validates config and selects the appropriate Qwen3-VL implementation.

    The model runner passes `text_neuron_config` and `encoder_neuron_config`
    for multimodal models.
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: EncoderNeuronConfig | None = None,
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(
            hf_config, text_neuron_config, encoder_neuron_config
        )

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: EncoderNeuronConfig | None = None,
    ) -> nn.Module:
        return cls._select_implementation(
            hf_config, text_neuron_config, encoder_neuron_config
        )

    @classmethod
    def _select_implementation(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None,
        encoder_neuron_config: EncoderNeuronConfig | None,
    ) -> nn.Module:
        cls._validate_config(hf_config, text_neuron_config)

        quantization = text_neuron_config.quantization if text_neuron_config else None

        if quantization == "mxfp8":
            from .model_mxfp8 import Qwen3VLForConditionalGeneration as Model
        else:
            from .model_bf16 import Qwen3VLForConditionalGeneration as Model

        return Model.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            encoder_neuron_config=encoder_neuron_config,
        )

    @classmethod
    def get_vision_token_merge_factor(cls, hf_config: PretrainedConfig) -> int:
        return hf_config.vision_config.spatial_merge_size**2

    @classmethod
    def get_max_encoder_embed_budget(
        cls,
        hf_config: PretrainedConfig,
        encoder_neuron_config: EncoderNeuronConfig,
    ) -> int | None:
        # Vision buckets are raw pre-merge tokens. merge_factor raw tokens
        # collapse into one embed token, so the budget is max // merge_factor.
        buckets = encoder_neuron_config.num_encoder_input_buckets
        if not buckets:
            return None
        return max(buckets) // cls.get_vision_token_merge_factor(hf_config)

    @classmethod
    def get_max_encoder_blocks_per_request(
        cls,
        hf_config: PretrainedConfig,
        encoder_neuron_config: EncoderNeuronConfig,
    ) -> int:
        # One cache block holds cache_block_size embed tokens. The largest
        # bucket is raw pre-merge tokens, so convert to embeds, then to blocks.
        from vllm_neuron.utils.mm_utils import get_encoder_cache_block_size

        buckets = encoder_neuron_config.num_encoder_input_buckets
        if not buckets:
            return 0
        merge_factor = cls.get_vision_token_merge_factor(hf_config)
        max_bucket = max(buckets)
        assert max_bucket % merge_factor == 0, (
            f"Largest VE bucket ({max_bucket}) must be divisible by "
            f"merge_factor ({merge_factor})"
        )
        cache_block_size = get_encoder_cache_block_size(
            hf_config, encoder_neuron_config.encoder_attention_block_size
        )
        return math.ceil((max_bucket // merge_factor) / cache_block_size)

    @classmethod
    def get_max_pixels_token_count(
        cls, hf_config: PretrainedConfig, max_pixels: int
    ) -> int:
        patch_size = hf_config.vision_config.patch_size
        return max_pixels // (patch_size**2)

    @classmethod
    def get_epd_kwargs(cls, item: MultiModalKwargsItem) -> MultiModalKwargsItem:
        # M-RoPE on the LM pool needs this item's grid_thw; send it over HTTP.
        for grid_key in ("image_grid_thw", "video_grid_thw"):
            if grid_key in item:
                return MultiModalKwargsItem({grid_key: item[grid_key]})

    @classmethod
    def get_placeholder_is_embed(
        cls,
        hf_config: PretrainedConfig,
        modality: str,
        placeholder_token_ids: list[int],
    ) -> list[bool] | None:
        # Qwen3-VL video placeholder interleaves timestamp text tokens
        # with video_pad per frame, so only the video_token_id positions carry embeddings
        if modality != "video":
            return None
        video_token_id = hf_config.video_token_id
        return [tok == video_token_id for tok in placeholder_token_ids]

    @classmethod
    def _validate_config(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None,
    ) -> None:
        # Reject an unrecognized quantization string at config-load time. Without
        # this, a typo (e.g. "mxfp-8") silently routes to bf16 and produces a
        # baffling baseline measurement instead of failing loudly. None / "bf16"
        # are the default bf16 path; "mxfp8" selects the CPU-dequant model
        # (model_mxfp8).
        quantization = text_neuron_config.quantization if text_neuron_config else None
        if quantization not in (None, "bf16", "mxfp8"):
            raise ValueError(
                f"Unknown quantization {quantization!r} for Qwen3-VL. "
                "Expected one of: None, 'bf16', 'mxfp8'."
            )
