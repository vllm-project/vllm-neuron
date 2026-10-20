# SPDX-License-Identifier: Apache-2.0
"""Multimodal encoder utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from transformers import PretrainedConfig

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalKwargsItem

    from vllm_neuron.model.neuron_config import EncoderNeuronConfig


def get_vision_token_merge_factor(hf_config: PretrainedConfig) -> int:
    """Return the vision token merge factor for a model.

    Looks up the model class via registry and calls its
    SupportsSpatialMerge.get_vision_token_merge_factor classmethod.
    Returns 1 if the model doesn't implement the protocol.

    Args:
        hf_config: HuggingFace PretrainedConfig for the model.

    Returns:
        Number of raw vision tokens that collapse into one embed token.
        E.g., 4 for Qwen3-VL (spatial_merge_size=2, so 2x2=4).
    """
    from vllm_neuron.model.interfaces import SupportsSpatialMerge
    from vllm_neuron.model.registry import get_models

    arch = hf_config.architectures[0]
    model_cls = dict(get_models()).get(arch)
    if model_cls and issubclass(model_cls, SupportsSpatialMerge):
        return model_cls.get_vision_token_merge_factor(hf_config)
    return 1


def get_encoder_cache_block_size(
    hf_config: PretrainedConfig, encoder_attention_block_size: int
) -> int:
    """Return the merged tokens that one encoder-cache block holds.

    The encoder-cache buffer stores post-merger embeds while the encoder
    attention block size counts raw tokens, so divide by the merge factor. The
    platform and the model runner both need this number and must agree on it:
    the platform converts a block count into the scheduler's embed budget, and
    the runner converts that budget back into a block count. A mismatch would
    make the scheduler retain a different number of items than the buffer holds.

    Args:
        hf_config: HuggingFace PretrainedConfig for the model.
        encoder_attention_block_size: Raw encoder tokens per attention block, from
            EncoderNeuronConfig.

    Returns:
        Merged (post-merger) tokens per encoder-cache block.

    Raises:
        ValueError: If the attention block size is not divisible by the merge
            factor, which would make a block hold a fractional token count.
    """
    merge_factor = get_vision_token_merge_factor(hf_config)
    if encoder_attention_block_size % merge_factor != 0:
        raise ValueError(
            f"encoder_attention_block_size ({encoder_attention_block_size}) "
            f"must be divisible by merge_factor ({merge_factor})"
        )
    return encoder_attention_block_size // merge_factor


def get_max_encoder_embed_budget(
    hf_config: PretrainedConfig, encoder_neuron_config: "EncoderNeuronConfig"
) -> int | None:
    """Return the per-step encoder budget in embed tokens for a model, or None.

    Looks up the model class via registry and calls its
    SupportsEncoderBudget.get_max_encoder_embed_budget classmethod. The scheduler
    caps the encoder compute budget at this value, so one prefill step never
    sends more encoder tokens than a compiled bucket holds. Each modality states
    its buckets in a different unit, so the model owns the conversion. Returns
    None when the model does not implement the protocol, so the scheduler does
    not cap the budget.

    Args:
        hf_config: HuggingFace PretrainedConfig for the model.
        encoder_neuron_config: The encoder neuron config with the buckets.

    Returns:
        The embed-token budget, or None if unsupported.
    """
    from vllm_neuron.model.interfaces import SupportsEncoderBudget
    from vllm_neuron.model.registry import get_models

    arch = hf_config.architectures[0]
    model_cls = dict(get_models()).get(arch)
    if model_cls and issubclass(model_cls, SupportsEncoderBudget):
        return model_cls.get_max_encoder_embed_budget(hf_config, encoder_neuron_config)
    return None


def get_max_encoder_blocks_per_request(
    hf_config: PretrainedConfig, encoder_neuron_config: "EncoderNeuronConfig"
) -> int:
    """Return the largest encoder-cache block count one request uses, or 0.

    Looks up the model class via registry and calls its
    SupportsEncoderBudget.get_max_encoder_blocks_per_request classmethod. The
    model runner sizes the per-request block gather with this value. Each
    modality states its buckets in a different unit, so the model owns the
    conversion. Returns 0 when the model does not implement the protocol.

    Args:
        hf_config: HuggingFace PretrainedConfig for the model.
        encoder_neuron_config: The encoder neuron config with the buckets.

    Returns:
        The block count, or 0 if unsupported.
    """
    from vllm_neuron.model.interfaces import SupportsEncoderBudget
    from vllm_neuron.model.registry import get_models

    arch = hf_config.architectures[0]
    model_cls = dict(get_models()).get(arch)
    if model_cls and issubclass(model_cls, SupportsEncoderBudget):
        return model_cls.get_max_encoder_blocks_per_request(
            hf_config, encoder_neuron_config
        )
    return 0


def get_max_pixels_token_count(
    hf_config: PretrainedConfig, max_pixels: int
) -> int | None:
    """Return the raw vision-token count for a max_pixels cap, or None.

    Looks up the model class via registry and calls its
    SupportsMaxPixels.get_max_pixels_token_count classmethod. Returns None
    when the model does not implement the protocol.

    Args:
        hf_config: HuggingFace PretrainedConfig for the model.
        max_pixels: Per-image pixel cap from mm_processor_kwargs.

    Returns:
        Raw (pre-merge) vision token count, or None if unsupported.
    """
    from vllm_neuron.model.interfaces import SupportsMaxPixels
    from vllm_neuron.model.registry import get_models

    arch = hf_config.architectures[0]
    model_cls = dict(get_models()).get(arch)
    if model_cls and issubclass(model_cls, SupportsMaxPixels):
        return model_cls.get_max_pixels_token_count(hf_config, max_pixels)
    return None


def get_epd_kwargs(
    hf_config: PretrainedConfig, item: "MultiModalKwargsItem"
) -> "MultiModalKwargsItem":
    """Return the subset of an item's mm kwargs the EPD LM pool needs.

    Looks up the model class via registry and calls its
    SupportsDisaggEncoder.get_epd_kwargs classmethod, which drops encoder-only
    tensors (e.g. pixel_values) the LM pool pulls out-of-band. Returns the item
    unchanged when the model doesn't implement the protocol.

    Args:
        hf_config: HuggingFace PretrainedConfig for the model.
        item: The full preprocessed mm kwargs for one media item.

    Returns:
        The LM-relevant mm kwargs item.
    """
    from vllm_neuron.model.interfaces import SupportsDisaggEncoder
    from vllm_neuron.model.registry import get_models

    arch = hf_config.architectures[0]
    model_cls = dict(get_models()).get(arch)
    if model_cls and issubclass(model_cls, SupportsDisaggEncoder):
        return model_cls.get_epd_kwargs(item)
    return item


def get_placeholder_is_embed(
    hf_config: PretrainedConfig,
    modality: str,
    placeholder_token_ids: list[int],
) -> list[bool] | None:
    """Re-derive the EPD placeholder is_embed mask via the model's rule.

    The disagg wire drops the renderer's is_embed mask, so the LM pool rebuilds
    it from the placeholder token ids by delegating to the model class's
    SupportsDisaggEncoder.get_placeholder_is_embed. Returns None (dense span)
    when the model doesn't implement the protocol or the modality is dense.

    Args:
        hf_config: HuggingFace PretrainedConfig for the model.
        modality: The item's modality ("image" / "video").
        placeholder_token_ids: The item's placeholder-span prompt token ids.

    Returns:
        A list[bool] mask over the span, or None if dense.
    """
    from vllm_neuron.model.interfaces import SupportsDisaggEncoder
    from vllm_neuron.model.registry import get_models

    arch = hf_config.architectures[0]
    model_cls = dict(get_models()).get(arch)
    if model_cls and issubclass(model_cls, SupportsDisaggEncoder):
        return model_cls.get_placeholder_is_embed(
            hf_config, modality, placeholder_token_ids
        )
    return None
