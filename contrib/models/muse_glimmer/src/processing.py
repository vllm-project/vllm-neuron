# SPDX-License-Identifier: Apache-2.0
"""Multimodal prompt processing for Muse-Glimmer on vLLM Neuron.

Pixels and prompt blocks come from the transformers MuseGlimmerProcessor — the one the
bring-up validated (R3-R5 images, V3-V5 video) — not vLLM's own processor implementation.

  image: resized to the fixed shape (shapes.IMAGE_SHAPES) closest to its aspect ratio
         (448x448 -> 256 tokens, 644x476 -> 391, 476x672 -> 408);
         block = <|image_start|> <|patch|> x N <|image_end|>
  video: vLLM hands over ALL frames + metadata (num_frames=-1, video_needs_metadata);
         VIDEO_FRAMES frames are picked like the bring-up (np.linspace(...).round(), not
         vLLM's truncating linspace), resized to VIDEO_W x VIDEO_H -> grid [4, 20, 26] ->
         520 tokens; the transformers processor gets the real fps and frame indices, so the
         block has the real timestamps:
         <|vid_start|> Time: t0 <|video|> x 130 <|vid_frame_separator|> ... <|vid_end|>

vLLM's MuseGlimmerMultiModalProcessor is the base for the prompt side (placeholder
expansion). Its video replacement assumes evenly spaced timestamps at its own processor's
sampling fps, so the video replacement here rebuilds the transformers block from the stored
timestamps instead.
"""

from functools import cached_property

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, BatchFeature
from transformers.video_utils import VideoMetadata

from vllm.model_executor.models import muse_glimmer as vmg
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import ImageSize, MultiModalDataParser
from vllm.multimodal.processing import PromptReplacement, PromptUpdateDetails

from .shapes import (
    IMAGE_SIZE_MOST_TOKENS,
    IMAGE_TOKENS,
    VIDEO_FRAMES,
    VIDEO_GRID,
    VIDEO_H,
    VIDEO_TOKENS,
    VIDEO_W,
    pick_image_shape,
)


def _sample_like_bringup(meta: dict, n_have: int):
    """Pick VIDEO_FRAMES of the n_have decoded frames the way muse_inputs.load_video does."""
    pick = np.linspace(0, n_have - 1, VIDEO_FRAMES).round().astype(int).tolist()
    have_idx = list(meta.get("frames_indices") or range(n_have))
    return pick, [int(have_idx[i]) for i in pick]


class _TransformersProcessor:
    """The transformers MuseGlimmerProcessor; images/videos resized to the validated sizes."""

    def __init__(self, hf):
        self._hf = hf

    def __getattr__(self, name):
        return getattr(self._hf, name)

    def image(self, img):
        img = (
            img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img))
        ).convert("RGB")
        out = self._hf(
            text=self._hf.image_token,
            images=[img.resize(pick_image_shape(*img.size))],
            add_special_tokens=False,
            return_tensors="pt",
        )
        return out["input_ids"][0].tolist(), out["pixel_values"]

    def video(self, frames, meta: dict):
        frames = np.asarray(frames)
        pick, orig_idx = _sample_like_bringup(meta, len(frames))
        sel = np.stack(
            [
                np.asarray(
                    Image.fromarray(frames[i]).convert("RGB").resize((VIDEO_W, VIDEO_H))
                )
                for i in pick
            ]
        )
        md = VideoMetadata(
            total_num_frames=int(meta.get("total_num_frames") or len(frames)),
            fps=float(meta["fps"]),
            frames_indices=orig_idx,
            width=VIDEO_W,
            height=VIDEO_H,
        )
        out = self._hf(
            text=self._hf.video_token,
            videos=[sel],
            video_metadata=[md],
            do_sample_frames=False,
            add_special_tokens=False,
            return_tensors="pt",
        )
        # one timestamp per temporal group (patch_temporal=2), as the processor writes them
        ts = torch.tensor(
            list(md.timestamps[::2])[: VIDEO_GRID[0]], dtype=torch.float32
        )
        return out["input_ids"][0].tolist(), out["pixel_values_videos"], ts


class MuseGlimmerNeuronProcessingInfo(vmg.MuseGlimmerProcessingInfo):
    @cached_property
    def _transformers_processor(self):
        return _TransformersProcessor(
            AutoProcessor.from_pretrained(self.ctx.model_config.model)
        )

    def get_hf_processor(self, **kwargs):
        return self._transformers_processor

    def get_data_parser(self):
        return MultiModalDataParser(
            video_needs_metadata=True
        )  # fps + frame indices -> timestamps

    def get_supported_mm_limits(self):
        return {"image": None, "video": None}

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        return {"image": IMAGE_TOKENS, "video": VIDEO_TOKENS}

    def get_image_size_with_most_features(self):
        width, height = IMAGE_SIZE_MOST_TOKENS
        return ImageSize(width=width, height=height)

    def get_num_frames_with_most_features(self, seq_len, mm_counts):
        return VIDEO_FRAMES


class MuseGlimmerNeuronDummyInputs(vmg.MuseGlimmerDummyInputsBuilder):
    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        return {
            "image": self._get_dummy_images(
                width=IMAGE_SIZE_MOST_TOKENS[0],
                height=IMAGE_SIZE_MOST_TOKENS[1],
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            ),
            "video": self._get_dummy_videos(
                width=VIDEO_W,
                height=VIDEO_H,
                num_frames=VIDEO_FRAMES,
                num_videos=mm_counts.get("video", 0),
                overrides=mm_options.get("video"),
            ),
        }


class MuseGlimmerNeuronProcessor(vmg.MuseGlimmerMultiModalProcessor):
    def _apply_hf_processor_main(self, mm_items, hf_processor_mm_kwargs):
        valid = mm_items.select(
            {k for k, c in mm_items.get_all_counts().items() if c > 0}
        )
        mm_data, passthrough = self._get_hf_mm_data(valid)
        prompt_text = self.dummy_inputs.get_dummy_text(mm_items.get_all_counts())
        proc = self.info.get_hf_processor()
        tok = proc.tokenizer
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        img_sentinel = tok.convert_tokens_to_ids(vmg.IMAGE_TOKEN)
        vid_sentinel = tok.convert_tokens_to_ids(vmg.VIDEO_TOKEN)
        cfg = self.info.get_hf_config()

        images, videos = (
            list(mm_data.get("images", ())),
            list(mm_data.get("videos", ())),
        )
        img_blocks, img_pix, img_sizes = [], [], []
        for img in images:
            ids, pix = proc.image(img)
            img_blocks.append(ids)
            img_pix.append(pix)
            img_sizes.append(ids.count(cfg.image_token_id))
        vid_blocks, vid_pix, vid_sizes, vid_ts = [], [], [], []
        for v in videos:
            frames, meta = v if isinstance(v, tuple) else (v, {})
            if not meta.get("fps"):
                raise ValueError(
                    "Muse-Glimmer video needs fps metadata for its timestamps"
                )
            ids, pix, ts = proc.video(frames, meta)
            vid_blocks.append(ids)
            vid_pix.append(pix)
            vid_sizes.append(ids.count(cfg.video_token_id))
            vid_ts.append(ts)

        out, ii, vi = [], 0, 0
        for t in prompt_ids:
            if t == img_sentinel:
                out.extend(img_blocks[ii])
                ii += 1
            elif t == vid_sentinel:
                out.extend(vid_blocks[vi])
                vi += 1
            else:
                out.append(t)
        data = {"input_ids": [out]}
        if img_pix:
            data.update(
                image_pixel_values=img_pix, image_feature_sizes=torch.tensor(img_sizes)
            )
        if vid_pix:
            data.update(
                video_pixel_values=vid_pix,
                video_feature_sizes=torch.tensor(vid_sizes),
                video_timestamps=vid_ts,
            )
        res = BatchFeature(data=data, tensor_type=None)
        res.update(passthrough)
        return res

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        fields = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        if "video_timestamps" in hf_inputs:
            fields["video_timestamps"] = MultiModalFieldConfig.batched(
                "video", keep_on_cpu=True
            )
        return fields

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        updates = [
            u
            for u in super()._get_prompt_updates(
                mm_items, hf_processor_mm_kwargs, out_mm_kwargs
            )
            if u.modality != "video"
        ]
        tok = self.info.get_tokenizer()
        vocab = tok.get_vocab()
        cfg = self.info.get_hf_config()
        start, sep, end = (
            vocab["<|vid_start|>"],
            vocab["<|vid_frame_separator|>"],
            vocab["<|vid_end|>"],
        )

        def video_replacement(item_idx):
            item = out_mm_kwargs["video"][item_idx]
            ts = item["video_timestamps"].data.tolist()
            per_group = int(item["video_feature_sizes"].data) // len(ts)
            rep = [start]
            for g, t in enumerate(ts):  # same block the transformers processor writes
                rep.extend(tok.encode(f"Time: {t:.1f}s", add_special_tokens=False))
                rep.extend([cfg.video_token_id] * per_group)
                rep.append(sep if g < len(ts) - 1 else end)
            return PromptUpdateDetails.select_token_id(rep, cfg.video_token_id)

        updates.append(
            PromptReplacement(
                modality="video",
                target=[vocab[vmg.VIDEO_TOKEN]],
                replacement=video_replacement,
            )
        )
        return updates
