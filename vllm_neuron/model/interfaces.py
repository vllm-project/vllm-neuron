# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from transformers import PretrainedConfig
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem

from vllm_neuron.model.neuron_config import EncoderNeuronConfig


@runtime_checkable
class SupportsEncoderWarmup(Protocol):
    """Models that support encoder warmup implement this interface."""

    def get_encoder(self) -> torch.nn.Module | None:
        """Return the encoder tower that warmup compiles and calls.

        Each modality names its own tower (``visual``, ``audio_tower``), so this
        gives the shared warmup path one name. None when the pool builds no
        encoder, for example a language-only PD pool.
        """
        ...

    def build_encoder_synthetic_inputs(
        self,
        bucket: int,
        encoder_neuron_config: EncoderNeuronConfig,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Construct shape-only tensors matching the encoder forward signature."""
        ...


@runtime_checkable
class SupportsSpatialMerge(Protocol):
    """Models that perform spatial token merging in the vision encoder."""

    @classmethod
    def get_vision_token_merge_factor(cls, hf_config: PretrainedConfig) -> int:
        """Return how many raw vision tokens collapse into one embed token."""
        ...


@runtime_checkable
class SupportsEncoderBudget(Protocol):
    """Models that cap the per-step encoder budget to one compiled bucket.

    The encoder is compiled for fixed bucket sizes. Each modality states its
    buckets in a different unit. Vision buckets are raw pre-merge tokens. Audio
    buckets are block counts. The model owns the conversion from a bucket to
    embed (post-merge) tokens and to encoder-cache blocks.
    """

    @classmethod
    def get_max_encoder_embed_budget(
        cls,
        hf_config: PretrainedConfig,
        encoder_neuron_config: EncoderNeuronConfig,
    ) -> int | None:
        """Return the largest bucket capacity in embed (post-merge) token space.

        The scheduler caps the encoder compute budget at this value. One prefill
        step then never sends more encoder tokens than a compiled bucket holds.

        Args:
            hf_config: HuggingFace PretrainedConfig for the model.
            encoder_neuron_config: The encoder neuron config with the buckets.

        Returns:
            The embed-token budget, or None when no buckets are configured.
        """
        ...

    @classmethod
    def get_max_encoder_blocks_per_request(
        cls,
        hf_config: PretrainedConfig,
        encoder_neuron_config: EncoderNeuronConfig,
    ) -> int:
        """Return the largest encoder-cache block count that one request uses.

        The model runner sizes the per-request block gather with this value.

        Args:
            hf_config: HuggingFace PretrainedConfig for the model.
            encoder_neuron_config: The encoder neuron config with the buckets.

        Returns:
            The block count, or 0 when no buckets are configured.
        """
        ...


@runtime_checkable
class SupportsMaxPixels(Protocol):
    """Models whose image processor supports a max_pixels resize cap.

    Models that honor it own the math that maps the pixel budget to a raw
    (pre-merge) vision-token count, since patch tiling differs across
    architectures.
    """

    @classmethod
    def get_max_pixels_token_count(
        cls, hf_config: PretrainedConfig, max_pixels: int
    ) -> int:
        """Convert a max_pixels cap into a raw vision-token count.

        Args:
            hf_config: HuggingFace PretrainedConfig for the model.
            max_pixels: Per-image pixel cap from mm_processor_kwargs.

        Returns:
            Raw (pre-merge) vision token count for that pixel budget.
        """
        ...


@runtime_checkable
class SupportsDisaggEncoder(Protocol):
    """Models split across an encoder pool and a language-model pool (EPD).

    The LM pool pulls the encoded embedding out-of-band (over NIXL), so it needs
    only the lightweight per-item metadata, not the heavy encoder-side tensors.
    """

    @classmethod
    def get_epd_kwargs(cls, item: MultiModalKwargsItem) -> MultiModalKwargsItem:
        """Return the subset of an item's mm kwargs the LM pool needs.

        Args:
            item: The full preprocessed mm kwargs for one media item.

        Returns:
            The LM-relevant subset (e.g. image_grid_thw for M-RoPE), dropping
            encoder-only tensors like pixel_values that the LM pool pulls over
            NIXL instead.
        """
        ...

    @classmethod
    def get_placeholder_is_embed(
        cls,
        hf_config: PretrainedConfig,
        modality: str,
        placeholder_token_ids: list[int],
    ) -> list[bool] | None:
        """Re-derive the sparse embedding mask for one item's placeholder span.

        EPD's disagg wire carries only (offset, length) and drops the renderer's
        is_embed mask, so the LM pool rebuilds it from the placeholder's token
        ids. Return None when the span is dense -- every position is an embedding,
        e.g. an image's contiguous pad run. Return a per-position bool when the
        layout is sparse, e.g. Qwen3-VL video, whose span interleaves
        timestamp/marker text tokens with the vision tokens.

        Args:
            hf_config: HuggingFace PretrainedConfig for the model.
            modality: The item's modality ("image" / "video").
            placeholder_token_ids: The prompt token ids of this item's
                placeholder span (length == PlaceholderRange.length).

        Returns:
            A list[bool] of len(placeholder_token_ids), or None if dense.
        """
        ...


@runtime_checkable
class SupportsMRoPE(Protocol):
    """Models that provide 3D multimodal rotary position embeddings."""

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        """Compute 3D M-RoPE positions for the given prompt.

        Args:
            input_tokens: Full prompt token IDs.
            mm_features: Multimodal features attached to the request.

        Returns:
            (positions, delta) where positions is [3, seq_len] int64 tensor
            and delta is the offset for computing decode-phase positions.
        """
        ...
