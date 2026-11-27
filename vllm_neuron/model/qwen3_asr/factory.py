# SPDX-License-Identifier: Apache-2.0
"""Factory for Qwen3-ASR model selection.

Qwen3-ASR has a single BF16 implementation today. The factory always routes to
``vllm_neuron.model.qwen3_asr.model_bf16.Qwen3ASRForConditionalGeneration``.

The model runner passes ``text_neuron_config`` and ``encoder_neuron_config`` for a
multimodal model. Qwen3-ASR uses ``encoder_neuron_config`` for its audio encoder.

The factory is a plain ``nn.Module``. The audio encoder uses merge factor 1, and
the text decoder uses standard 1D RoPE.
"""

import torch.nn as nn
from transformers import PretrainedConfig
from vllm.multimodal.inputs import MultiModalFeatureSpec

from vllm_neuron.model.interfaces import (
    SupportsEncoderBudget,
    SupportsEncoderCacheLayout,
)
from vllm_neuron.model.neuron_config import NeuronConfig, EncoderNeuronConfig


class Qwen3ASRForConditionalGeneration(
    nn.Module, SupportsEncoderBudget, SupportsEncoderCacheLayout
):
    """Factory that validates the config and selects the Qwen3-ASR implementation."""

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
        cls._validate_config(text_neuron_config)

        from .model_bf16 import Qwen3ASRForConditionalGeneration as Model

        return Model.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            encoder_neuron_config=encoder_neuron_config,
        )

    @classmethod
    def get_max_encoder_embed_budget(
        cls,
        hf_config: PretrainedConfig,
        encoder_neuron_config: EncoderNeuronConfig,
    ) -> int | None:
        # Audio buckets are block counts. A block holds slots_per_block chunk
        # slots of num_tokens_per_chunk real post-CNN tokens each (merge factor is 1),
        # so budget = max_blocks * slots_per_block * num_tokens_per_chunk. This is the
        # exact per-block capacity. A single request can pack this many real
        # tokens, so the scheduler cap must not fall below it.
        from vllm_neuron.model.qwen3_asr.utils.audio_preprocessing import (
            get_max_real_tokens_per_block,
        )

        buckets = encoder_neuron_config.num_encoder_input_buckets
        if not buckets:
            return None
        tokens_per_block = get_max_real_tokens_per_block(
            hf_config, encoder_neuron_config.encoder_attention_block_size
        )
        return max(buckets) * tokens_per_block

    @classmethod
    def get_max_encoder_blocks_per_request(
        cls,
        hf_config: PretrainedConfig,
        encoder_neuron_config: EncoderNeuronConfig,
    ) -> int:
        # Audio buckets are block counts already. One inference window fills a
        # whole block, so a request uses at most max_bucket blocks.
        buckets = encoder_neuron_config.num_encoder_input_buckets
        if not buckets:
            return 0
        return max(buckets)

    @classmethod
    def get_encoder_cache_item_blocks(
        cls,
        hf_config: PretrainedConfig,
        encoder_neuron_config: EncoderNeuronConfig,
        mm_feature: MultiModalFeatureSpec,
    ) -> int | None:
        # A clip owns a run of whole blocks, and a block holds whole inference
        # windows plus pad. embed_multimodal sizes that run with
        # plan_audio_blocks, which depends on the clip only through its post-CNN
        # length. That length is the clip's placeholder length, which the
        # scheduler always has: with mm_processor_cache_type="shm",
        # mm_feature.data holds only an address.
        from vllm_neuron.model.qwen3_asr.utils.audio_preprocessing import (
            clip_num_cache_blocks,
            get_audio_window_tokens,
        )

        num_tokens = mm_feature.mm_position.get_num_embeds()
        if num_tokens <= 0:
            return None
        audio_config = hf_config.thinker_config.audio_config
        return clip_num_cache_blocks(
            num_tokens,
            block_size=encoder_neuron_config.encoder_attention_block_size,
            window_tokens=get_audio_window_tokens(hf_config),
            n_window=audio_config.n_window,
            n_window_infer=audio_config.n_window_infer,
        )

    @classmethod
    def _validate_config(cls, text_neuron_config: NeuronConfig | None) -> None:
        # Reject an unknown quantization string at config-load time. Without this
        # check a typo routes to BF16 and produces a wrong baseline instead of a
        # loud failure. None and "bf16" both select the BF16 path.
        quantization = text_neuron_config.quantization if text_neuron_config else None
        if quantization not in (None, "bf16"):
            raise ValueError(
                f"Unknown quantization {quantization!r} for Qwen3-ASR. "
                "Expected one of: None, 'bf16'."
            )
