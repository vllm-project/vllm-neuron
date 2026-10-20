# SPDX-License-Identifier: Apache-2.0
"""Tokenizer-boundary codec for the EPD Router.

The two directions of the renderer's tokenizer boundary:

* encode -- build_renderer + render_request turn OpenAI chat messages
  into prompt_token_ids + per-item multimodal records (request side).
* decode -- Detokenizer turns PD's token_ids back into text (response side),
  incrementally so the streaming and non-stream paths share one implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from vllm.v1.engine.detokenizer import detokenize_incrementally

from vllm_neuron.vllm.disaggregated_encoder.protocol import (
    PreprocessedItem,
    PreprocessedRequest,
)

if TYPE_CHECKING:
    from vllm.renderers.base import BaseRenderer


def align_placeholders_to_tokens(
    pre: PreprocessedRequest, override_ids: Sequence[int]
) -> PreprocessedRequest:
    """Re-derive item placeholder offsets from an overridden token stream.

    Validation-only: when a teacher-forcing caller replaces the rendered prompt
    tokens, the renderer's placeholder offsets (computed against the renderer's
    layout) may no longer point at the image-pad runs in the new stream. Locate
    the contiguous image-pad runs in override_ids (the pad token id is read
    from the renderer's own stream at the first item's offset, so this is
    model-agnostic) and reassign offsets in order, preserving each item's
    mm_hash / mm_kwargs_item / length. Raises if the run count or lengths don't
    match the items -- a loud failure beats silently misaligning vision blocks.

    Args:
        pre: The renderer-derived preprocessed request (items in prompt order).
        override_ids: The token stream that will actually be driven to PD.

    Returns:
        A new PreprocessedRequest with prompt_token_ids=override_ids and item
        offsets re-homed onto that stream.
    """
    override = list(override_ids)
    if not pre.items:
        return replace(pre, prompt_token_ids=override)
    pad_id = pre.prompt_token_ids[pre.items[0].offset]
    # Collect contiguous runs of pad_id in the override stream.
    runs: list[tuple[int, int]] = []  # (offset, length)
    i, n = 0, len(override)
    while i < n:
        if override[i] == pad_id:
            j = i
            while j < n and override[j] == pad_id:
                j += 1
            runs.append((i, j - i))
            i = j
        else:
            i += 1
    if len(runs) != len(pre.items):
        raise ValueError(
            f"override token stream has {len(runs)} image-pad runs but request "
            f"has {len(pre.items)} media items"
        )
    new_items = []
    for item, (offset, length) in zip(pre.items, runs):
        if length != item.length:
            raise ValueError(
                f"image-pad run length {length} at offset {offset} does not "
                f"match item length {item.length} (mm_hash={item.mm_hash})"
            )
        new_items.append(replace(item, offset=offset))
    return replace(pre, prompt_token_ids=override, items=new_items)


def build_renderer(
    model: str,
    renderer_num_workers: int,
    quantization: str | None = None,
    *,
    media_io_kwargs: Mapping[str, Any] | None = None,
    mm_processor_kwargs: Mapping[str, Any] | None = None,
) -> "BaseRenderer":
    """Build a vLLM renderer for HF preprocessing.

    Builds a VllmConfig from the model id and hands it to renderer_from_config, which
    loads the tokenizer + HF multimodal processor. Run once at
    startup; call render_request per request.

    Args:
        model: HF model id / path (must match the VE/PD model so mm_hash and the
            preprocessed tensors are exactly what the VE expects).
        renderer_num_workers: Size of the renderer's HF-preprocessing thread pool.
        quantization: the PD/VE neuron_config.quantization (e.g. "mxfp8"). The
            quantization_config is rejected by the Neuron platform validator
            unless the CPU-dequant quantization is declared here.
        media_io_kwargs: Per-modality media-loader kwargs forwarded to the HF
            media loader, e.g. {"video": {"num_frames": 60}}.
        mm_processor_kwargs: Extra kwargs for the HF multimodal processor, e.g.
            {"max_pixels": 262144} to cap per-frame resolution so a high-res input
            fits the VE's compiled num_encoder_input_buckets.

    Returns:
        A constructed BaseRenderer.
    """
    from vllm.engine.arg_utils import EngineArgs
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.renderers.registry import renderer_from_config

    engine_kwargs: dict = dict(model=model)
    if quantization:
        engine_kwargs["additional_config"] = {
            "neuron_config": {"quantization": quantization}
        }
    if media_io_kwargs:
        engine_kwargs["media_io_kwargs"] = dict(media_io_kwargs)
    if mm_processor_kwargs:
        engine_kwargs["mm_processor_kwargs"] = dict(mm_processor_kwargs)
    engine_args = EngineArgs(**engine_kwargs)
    vllm_config = engine_args.create_engine_config()
    # Offload HF preprocessing across threads (native decode/transform release the GIL).
    vllm_config.model_config.renderer_num_workers = renderer_num_workers
    renderer = renderer_from_config(vllm_config)

    # Force a local "processor_only" cache: the default single-process "lru" cache drops pixel_values on a hit, but the Router must forward pixels to the VE every request.
    mm_processor = getattr(renderer, "mm_processor", None)
    if mm_processor is not None:
        mm_processor.cache = MULTIMODAL_REGISTRY.processor_only_cache_from_config(
            vllm_config
        )
    return renderer


async def render_request(
    renderer: "BaseRenderer",
    messages: Sequence[Mapping[str, Any]],
    *,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> PreprocessedRequest:
    """Run HF preprocessing once for a chat request → token ids + per-item records.

    Drives the renderer's render_chat_async (chat template → tokenize → HF multimodal
    processing), then flattens into one PreprocessedItem per media item.

    Args:
        renderer: A renderer from build_renderer.
        messages: OpenAI chat messages (one conversation).
        chat_template_kwargs: Extra chat-template kwargs (e.g.
            {"add_generation_prompt": True}).

    Returns:
        A PreprocessedRequest. Text-only requests yield items == [].
    """
    from vllm.renderers.params import ChatParams

    chat_params = ChatParams(chat_template_kwargs=dict(chat_template_kwargs or {}))
    mm_config = getattr(renderer.model_config, "multimodal_config", None)
    if mm_config is not None:
        chat_params = chat_params.with_defaults(
            default_media_io_kwargs=getattr(mm_config, "media_io_kwargs", None),
            default_mm_processor_kwargs=getattr(mm_config, "mm_processor_kwargs", None),
        )
    _conversations, eng_prompts = await renderer.render_chat_async(
        [list(messages)], chat_params
    )
    eng = eng_prompts[0]

    # Flatten the renderer's per-modality, index-aligned maps into one PreprocessedItem
    # per media item. Modality is the dict key; item index is the list position.
    token_ids = list(eng["prompt_token_ids"])
    mm_kwargs = eng.get("mm_kwargs") or {}
    mm_hashes = eng.get("mm_hashes") or {}
    mm_placeholders = eng.get("mm_placeholders") or {}

    items: list[PreprocessedItem] = []
    for modality, kwargs_items in mm_kwargs.items():
        hashes = mm_hashes.get(modality, [])
        placeholders = mm_placeholders.get(modality, [])
        # These three maps are produced index-aligned by the renderer. A length mismatch
        # means a media item would be silently dropped from routing (no error, wrong
        # answer downstream), so fail loud instead of letting zip truncate to shortest.
        if not len(kwargs_items) == len(hashes) == len(placeholders):
            raise ValueError(
                f"misaligned multimodal maps for modality {modality!r}: "
                f"{len(kwargs_items)} kwargs, {len(hashes)} hashes, "
                f"{len(placeholders)} placeholders"
            )
        # TODO: future long-video per-frame splitting — split a video item into per-frame
        # mm_hash'd sub-items here so each frame routes independently to distribute load.
        for kwargs_item, mm_hash, placeholder in zip(
            kwargs_items, hashes, placeholders, strict=True
        ):
            items.append(
                PreprocessedItem(
                    mm_hash=mm_hash,
                    mm_kwargs_item=kwargs_item,
                    offset=placeholder.offset,
                    length=placeholder.length,
                    modality=modality,
                )
            )
    return PreprocessedRequest(prompt_token_ids=token_ids, items=items)


class Detokenizer:
    """Incremental detokenizer for one output sequence (one OpenAI choice).

    Wraps vLLM's detokenize_incrementally, holding the running token-offset
    state so appended tokens decode into text without re-scanning the whole
    sequence.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
    ) -> None:
        self._tokenizer = tokenizer
        self._skip_special_tokens = skip_special_tokens
        self._spaces_between_special_tokens = spaces_between_special_tokens
        self._all_token_ids: list[int] = []
        self._prev_tokens: list[str] | None = None
        self._prefix_offset = 0
        self._read_offset = 0

    def decode_delta(self, new_token_ids: Sequence[int]) -> str:
        """Append new_token_ids and return only the newly-decoded text.

        Args:
            new_token_ids: The token ids emitted since the last call.

        Returns:
            The incremental text for those tokens (empty string if the tokens
            do not yet complete a printable unit).
        """
        text = ""
        for token_id in new_token_ids:
            self._all_token_ids.append(token_id)
            new_tokens, delta_text, self._prefix_offset, self._read_offset = (
                detokenize_incrementally(
                    self._tokenizer,
                    self._all_token_ids,
                    self._prev_tokens,
                    self._prefix_offset,
                    self._read_offset,
                    skip_special_tokens=self._skip_special_tokens,
                    spaces_between_special_tokens=self._spaces_between_special_tokens,
                )
            )
            if self._prev_tokens is None:
                self._prev_tokens = new_tokens
            else:
                self._prev_tokens.extend(new_tokens)
            text += delta_text
        return text
