# SPDX-License-Identifier: Apache-2.0
"""Front-end (vLLM) multimodal processor registration for MuseGlimmer.

vLLM 0.24 has no native MuseGlimmer model. Registering the plugin factory
class with this processor in the front-end process (see
``NeuronPlatform.pre_register_and_update``) makes ``ModelConfig`` resolve the
architecture to the plugin class (``supports_multimodal=True``) instead of the
generic ``TransformersMultiModalForCausalLM``.

Image path. The HF ``MuseGlimmerProcessor`` performs the prompt
expansion itself: the chat template renders one ``<|patch|>`` per image, and
the processor replaces it with
``<|image_start|> + <|patch|> * (t*h*w / 4) + <|image_end|>``. vLLM then
locates that exact token run (``_get_prompt_updates``) and marks only the
``<|patch|>`` tokens as embeddings. Token ids therefore match HF
``apply_chat_template(..., tokenize=True)`` exactly, for ``LLM.chat`` requests
and for raw prompts holding the HF-rendered chat text (with or without the
leading ``<|begin_of_text|>``: vLLM tokenizes with ``add_special_tokens=True``
and a doubled leading BOS is collapsed, see ``_postprocess_prompt``).

Video path. Same split: the chat template renders one
``<|video|>`` per video and HF ``MuseGlimmerProcessor.replace_video_token``
expands it to

    <|vid_start|> Time: t0s <|video|>*n <|vid_frame_separator|> Time: t1s ...
    <|video|>*n <|vid_end|>

with one ``Time: {:.1f}s`` per temporal group (``timestamps[::2]``) and
``n = h*w/4`` (<= ``max_video_frame_tokens``). Only the ``<|video|>`` tokens
are embeds (``is_embed``); the timestamp text and the markers are ordinary
text tokens. vLLM hands each video over as ``(frames[T,H,W,3] uint8, metadata)``
(``video_needs_metadata``); metadata follows HF ``VideoMetadata`` (``fps``,
``total_num_frames``, optional ``frames_indices`` / ``duration``) plus vLLM's
``do_sample_frames`` flag. Raw frames (``do_sample_frames`` absent / True) are
sampled by the HF processor (``fps`` / ``num_frames`` kwargs, default 2 FPS and
96 frames); frames already sampled by vLLM's video loader are passed with
``do_sample_frames=False`` and their original ``frames_indices`` so the
timestamps stay in source time. Each video is run through the HF processor on
its own (prompt ``<|video|>``), so the expansion is produced by HF itself; the
resulting token ids are kept per item (``video_prompt_ids``) so the vLLM
processor cache can rebuild the prompt update without re-running HF.

Budget / limits are computed from processor caps, without the HF processor,
so text-only runs (including ``skip_tokenizer_init=True``) work.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from transformers import BatchFeature
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

IMAGE_TOKEN = "<|patch|>"
IMAGE_START_TOKEN = "<|image_start|>"
IMAGE_END_TOKEN = "<|image_end|>"
VIDEO_TOKEN = "<|video|>"
VIDEO_START_TOKEN = "<|vid_start|>"
VIDEO_END_TOKEN = "<|vid_end|>"
VIDEO_SEP_TOKEN = "<|vid_frame_separator|>"

# HF image/video processor defaults (image_processing_muse_glimmer.py,
# video_processing_muse_glimmer.py).
DEFAULT_MAX_IMAGE_TOKENS = 4096
DEFAULT_MAX_VIDEO_FRAME_TOKENS = 144
DEFAULT_NUM_FRAMES = 96
DEFAULT_TEMPORAL_PATCH = 2
DEFAULT_VIDEO_FPS = 2.0

# HF ``VideoMetadata`` fields; anything else (e.g. vLLM's ``do_sample_frames``) is
# stripped before the metadata reaches ``VideoMetadata(**d)``.
_HF_VIDEO_METADATA_KEYS = (
    "total_num_frames", "fps", "width", "height", "duration", "video_backend", "frames_indices",
)


class MuseGlimmerDataParser(MultiModalDataParser):
    """``video_needs_metadata`` parser that tolerates a bare frame array.

    A video without metadata gets ``fps=None`` and all frames as
    ``frames_indices``; the HF processor then assumes 24 FPS (with a warning),
    exactly like HF ``apply_chat_template`` on a bare array.
    """

    def _get_video_with_metadata(self, video):
        frames, metadata = super()._get_video_with_metadata(video)
        if metadata is None:
            metadata = {"total_num_frames": len(frames), "fps": None,
                        "frames_indices": list(range(len(frames)))}
        return frames, metadata


class MuseGlimmerProcessingInfo(BaseProcessingInfo):
    def _mm_kwargs(self) -> Mapping[str, Any]:
        mm_config = self.ctx.model_config.multimodal_config
        return (mm_config.mm_processor_kwargs or {}) if mm_config else {}

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_data_parser(self) -> MultiModalDataParser:
        return MuseGlimmerDataParser(
            video_needs_metadata=True,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_max_image_tokens(self) -> int:
        return int(self._mm_kwargs().get("max_image_tokens", DEFAULT_MAX_IMAGE_TOKENS))

    def get_max_video_tokens(self) -> int:
        kw = self._mm_kwargs()
        per_group = int(kw.get("max_video_frame_tokens", DEFAULT_MAX_VIDEO_FRAME_TOKENS))
        frames = int(kw.get("num_frames", DEFAULT_NUM_FRAMES))
        groups = -(-frames // DEFAULT_TEMPORAL_PATCH)
        return per_group * groups

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int]):
        # Computed from processor caps, so no tokenizer / HF processor needed.
        return {"image": self.get_max_image_tokens(), "video": self.get_max_video_tokens()}


class MuseGlimmerDummyInputsBuilder(BaseDummyInputsBuilder[MuseGlimmerProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_TOKEN * mm_counts.get("image", 0) + VIDEO_TOKEN * mm_counts.get(
            "video", 0
        )

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        side = 14 * 2 * 32  # 896 px -> 64x64 patches -> 1024 merged tokens
        out = {
            "image": self._get_dummy_images(
                width=side,
                height=side,
                num_images=num_images,
                overrides=(mm_options or {}).get("image"),
            )
        }
        if num_videos:
            # 16:9 frames at the per-group cap (448x252 -> 18x32 patches -> 144 tokens),
            # max frames at the sampling FPS (no resampling: do_sample_frames=False).
            kw = self.info._mm_kwargs()
            nf = int(kw.get("num_frames", DEFAULT_NUM_FRAMES))
            fps = float(kw.get("fps", DEFAULT_VIDEO_FPS))
            frames = np.full((nf, 252, 448, 3), 255, dtype=np.uint8)
            meta = {"fps": fps, "total_num_frames": nf, "duration": nf / fps,
                    "frames_indices": list(range(nf)), "do_sample_frames": False}
            out["video"] = [(frames.copy(), dict(meta)) for _ in range(num_videos)]
        return out


class MuseGlimmerMultiModalProcessor(BaseMultiModalProcessor[MuseGlimmerProcessingInfo]):
    # vLLM 0.29 contract: the renderer tokenizes the prompt; ``_apply_hf_processor_main``
    # only sees the media (no prompt text), and ``_postprocess_prompt`` sees the ids.
    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        valid = mm_items.select({k for k, c in mm_items.get_all_counts().items() if c > 0})
        mm_data, passthrough = self._get_hf_mm_data(valid)
        mm_data = dict(mm_data)
        videos = mm_data.pop("videos", None) or []
        if not videos:
            return super()._apply_hf_processor_main(mm_items, hf_processor_mm_kwargs)

        ctx = self.info.ctx
        hf_proc = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        # Each video on its own through the HF processor (text = one <|video|>): HF
        # itself samples the frames and builds the timestamp/marker expansion.
        video_repl, pv_list, grid_list = [], [], []
        for frames, metadata in videos:
            metadata = dict(metadata or {})
            kw = dict(hf_processor_mm_kwargs)
            if "do_sample_frames" in metadata and "do_sample_frames" not in kw:
                kw["do_sample_frames"] = bool(metadata["do_sample_frames"])
            meta = {k: metadata[k] for k in _HF_VIDEO_METADATA_KEYS if metadata.get(k) is not None}
            meta.setdefault("total_num_frames", len(frames))
            meta.setdefault("fps", None)
            if "frames_indices" not in meta:
                meta["frames_indices"] = list(range(len(frames)))
            out = ctx.call_hf_processor(
                hf_proc,
                {"text": VIDEO_TOKEN, "videos": [frames], "video_metadata": [meta]},
                {**kw, "add_special_tokens": False},
            )
            ids = out["input_ids"]
            ids = ids.tolist() if hasattr(ids, "tolist") else ids
            video_repl.append(torch.tensor(ids[0], dtype=torch.int64))
            pv_list.append(out["pixel_values_videos"])
            grid_list.append(out["video_grid_thw"])

        if mm_data:  # images in the same request
            text_kwargs = {k: v for k, v in hf_processor_mm_kwargs.items() if k not in ("fps", "num_frames")}
            processed = ctx.call_hf_processor(hf_proc, mm_data, text_kwargs)
            processed.pop("input_ids", None)
            processed.pop("attention_mask", None)
        else:
            processed = BatchFeature({})
        processed["pixel_values_videos"] = torch.cat(pv_list)
        processed["video_grid_thw"] = torch.cat(grid_list)
        processed["video_prompt_ids"] = video_repl
        processed.update(passthrough)
        return processed

    def _postprocess_prompt(self, prompt: list[int]) -> list[int]:
        # The renderer tokenizes multimodal prompts with add_special_tokens=True for both
        # the chat and the completion route (``info.default_tok_params``). The MuseGlimmer
        # chat template already renders <|begin_of_text|>, and HF apply_chat_template
        # tokenizes without special tokens, so the chat route starts with a doubled BOS.
        # Collapse it so prompt ids equal HF exactly (a prompt without BOS is unaffected).
        bos = self.info.get_tokenizer().bos_token_id
        if bos is not None and len(prompt) >= 2 and prompt[0] == bos == prompt[1]:
            return list(prompt[1:])
        return prompt

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_thw.prod(-1)
            ),
            image_grid_thw=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
                "video", video_grid_thw.prod(-1)
            ),
            video_grid_thw=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
            # HF expansion of each <|video|> (timestamps + markers + embeds), front-end only.
            video_prompt_ids=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        vocab = self.info.get_tokenizer().get_vocab()
        patch_id = vocab[IMAGE_TOKEN]
        start_id = vocab[IMAGE_START_TOKEN]
        end_id = vocab[IMAGE_END_TOKEN]
        merge = self.info.get_hf_config().vision_config.merge_size ** 2

        def image_replacement(item_idx: int):
            grid = out_mm_kwargs["image"][item_idx]["image_grid_thw"].data
            n = int(grid.prod()) // merge
            return PromptUpdateDetails.select_token_id(
                [start_id] + [patch_id] * n + [end_id], embed_token_id=patch_id
            )

        video_id = vocab[VIDEO_TOKEN]

        def video_replacement(item_idx: int):
            ids = out_mm_kwargs["video"][item_idx]["video_prompt_ids"].data
            return PromptUpdateDetails.select_token_id(
                [int(t) for t in ids.tolist()], embed_token_id=video_id
            )

        return [
            PromptReplacement(
                modality="image", target=[patch_id], replacement=image_replacement
            ),
            PromptReplacement(
                modality="video", target=[video_id], replacement=video_replacement
            ),
        ]
