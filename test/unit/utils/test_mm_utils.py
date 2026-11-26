# SPDX-License-Identifier: Apache-2.0
"""Encoder-cache block counts: the device block count and per-item blocks."""

import math
import types

import pytest
import torch
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)

from vllm_neuron.model.neuron_config import EncoderNeuronConfig
from vllm_neuron.model.qwen3_asr.utils.audio_preprocessing import (
    _get_feat_extract_output_lengths,
    clip_num_cache_blocks,
    plan_audio_blocks,
)
from vllm_neuron.utils.mm_utils import (
    get_encoder_cache_item_blocks,
    get_encoder_cache_num_blocks,
)


def _feature(modality: str, num_embeds: int, **fields) -> MultiModalFeatureSpec:
    data = None
    if fields:
        data = MultiModalKwargsItem(
            {
                key: MultiModalFieldElem(data=value, field=MultiModalBatchedField())
                for key, value in fields.items()
            }
        )
    return MultiModalFeatureSpec(
        data=data,
        modality=modality,
        identifier=f"{modality}-{num_embeds}",
        mm_position=PlaceholderRange(offset=0, length=num_embeds),
    )


def _encoder_config(block_size: int) -> EncoderNeuronConfig:
    return EncoderNeuronConfig.from_dict(
        {
            "num_encoder_input_buckets": [block_size],
            "encoder_attention_block_size": block_size,
        }
    )


@pytest.mark.parametrize(
    "cache_size, configured, expected",
    [(16384, None, 33), (16384, 64, 64), (16384, 8, 33), (1000, None, 3)],
)
def test_get_encoder_cache_num_blocks(cache_size, configured, expected):
    assert get_encoder_cache_num_blocks(cache_size, 512, configured) == expected


@pytest.mark.parametrize("num_embeds, expected", [(1, 1), (512, 1), (513, 2)])
def test_item_blocks_default_to_dense_layout(num_embeds, expected):
    # A model without SupportsEncoderCacheLayout, merge factor 1.
    hf_config = types.SimpleNamespace(architectures=["UnknownForCausalLM"])
    feature = _feature("image", num_embeds)
    assert get_encoder_cache_item_blocks(hf_config, _encoder_config(512), feature) == (
        expected
    )


QWEN3_VL = types.SimpleNamespace(
    architectures=["Qwen3VLForConditionalGeneration"],
    vision_config=types.SimpleNamespace(spatial_merge_size=2),
)


def test_qwen3_vl_image_takes_one_block():
    # 32x32 patches = 1024 raw tokens = 256 embeds, one 512-embed block.
    feature = _feature("image", 256, image_grid_thw=torch.tensor([1, 32, 32]))
    assert get_encoder_cache_item_blocks(QWEN3_VL, _encoder_config(2048), feature) == 1


VIDEO_TOKEN_ID = 151656


def _shm_item() -> MultiModalKwargsItem:
    # What the scheduler holds with mm_processor_cache_type="shm":
    # ShmObjectStoreSenderCache.address_as_item, no item tensors.
    return MultiModalKwargsItem(
        {
            "address": MultiModalFieldElem(data=0, field=MultiModalBatchedField()),
            "monotonic_id": MultiModalFieldElem(data=0, field=MultiModalBatchedField()),
        }
    )


def _item(cache_type: str, **fields) -> MultiModalKwargsItem | None:
    if cache_type == "shm":
        return _shm_item()
    if cache_type == "none":
        return None
    return MultiModalKwargsItem(
        {
            key: MultiModalFieldElem(data=value, field=MultiModalBatchedField())
            for key, value in fields.items()
        }
    )


def _video_feature(t: int, h: int, w: int, cache_type: str) -> MultiModalFeatureSpec:
    """A Qwen3-VL video item as vLLM's processor places it in the prompt."""
    from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor

    # Any tokenizer works: the timestamp tokens are never video tokens.
    tokenizer = types.SimpleNamespace(
        encode=lambda text, add_special_tokens=False: [7] * (len(text) % 3 + 1)
    )
    repl = Qwen3VLMultiModalProcessor.get_video_repl(
        tokens_per_frame=[h * w // 4] * t,
        timestamps=[1.0 + 2 * i for i in range(t)],
        tokenizer=tokenizer,
        vision_start_token_id=5,
        vision_end_token_id=6,
        video_token_id=VIDEO_TOKEN_ID,
        select_token_id=True,
    )
    return MultiModalFeatureSpec(
        data=_item(cache_type, video_grid_thw=torch.tensor([t, h, w])),
        modality="video",
        identifier=f"video-{t}x{h}x{w}",
        mm_position=PlaceholderRange(
            offset=3, length=len(repl.full), is_embed=repl.is_embed(repl.full)
        ),
    )


CACHE_TYPES = ["lru", "shm", "none"]


@pytest.mark.parametrize("cache_type", CACHE_TYPES)
def test_qwen3_vl_video_packs_whole_frames_per_block(cache_type):
    from vllm_neuron.model.qwen3_vl.utils.vision_block_packing import (
        video_cache_tokens_per_block,
    )

    # 7 frames of 24x24 = 576 raw tokens (144 embeds) each. A 2048-token block
    # holds 3 whole frames, so the video takes 3 blocks, while its 1008 embeds
    # would fill only ceil(1008 / 512) = 2 dense blocks.
    assert video_cache_tokens_per_block(7, 576, 2048, 4) == [432, 432, 144]
    feature = _video_feature(7, 24, 24, cache_type)
    assert feature.mm_position.get_num_embeds() == 7 * 144
    assert get_encoder_cache_item_blocks(QWEN3_VL, _encoder_config(2048), feature) == 3


@pytest.mark.parametrize("cache_type", CACHE_TYPES)
@pytest.mark.parametrize("t, h, w", [(1, 32, 34), (4, 32, 34), (9, 20, 30), (3, 2, 2)])
def test_qwen3_vl_video_blocks_match_encoder_layout(cache_type, t, h, w):
    # The block count embed_multimodal allocates, from video_grid_thw.
    from vllm_neuron.model.qwen3_vl.utils.vision_block_packing import (
        video_cache_tokens_per_block,
    )

    expected = len(video_cache_tokens_per_block(t, h * w, 2048, 4))
    feature = _video_feature(t, h, w, cache_type)
    assert get_encoder_cache_item_blocks(QWEN3_VL, _encoder_config(2048), feature) == (
        expected
    )


def test_qwen3_vl_video_frame_larger_than_block_falls_back_to_dense():
    # embed_multimodal rejects a frame larger than one block. The scheduler
    # still gets a block count rather than an error.
    feature = _video_feature(2, 48, 48, "lru")
    assert get_encoder_cache_item_blocks(QWEN3_VL, _encoder_config(2048), feature) == 3


QWEN3_ASR = types.SimpleNamespace(
    architectures=["Qwen3ASRForConditionalGeneration"],
    thinker_config=types.SimpleNamespace(
        audio_config=types.SimpleNamespace(n_window=50, n_window_infer=800)
    ),
)
MEL_LENS = [37, 100, 800, 801, 1600, 2000, 5000, 12345]


def _audio_feature(mel_len: int, cache_type: str) -> MultiModalFeatureSpec:
    # vLLM's Qwen3-ASR processor puts one audio token per post-CNN step.
    num_embeds = int(_get_feat_extract_output_lengths(torch.tensor(mel_len)))
    return MultiModalFeatureSpec(
        data=_item(cache_type, audio_feature_lengths=torch.tensor(mel_len)),
        modality="audio",
        identifier=f"audio-{mel_len}",
        mm_position=PlaceholderRange(offset=0, length=num_embeds),
    )


@pytest.mark.parametrize("block_size", [128, 256])
@pytest.mark.parametrize("cache_type", CACHE_TYPES)
def test_qwen3_asr_clip_blocks_match_audio_plan(block_size, cache_type):
    # embed_multimodal plans all clips of a batch together and gives clip i
    # ceil(len(clip_slots[i]) / slots_per_block) cache blocks.
    plan = plan_audio_blocks(
        MEL_LENS,
        block_size=block_size,
        window_tokens=104,
        n_window=50,
        n_window_infer=800,
        one_item_per_block=True,
    )
    expected = [
        math.ceil(len(slots) / plan.slots_per_block) for slots in plan.clip_slots
    ]
    actual = [
        get_encoder_cache_item_blocks(
            QWEN3_ASR, _encoder_config(block_size), _audio_feature(mel_len, cache_type)
        )
        for mel_len in MEL_LENS
    ]
    assert actual == expected


@pytest.mark.parametrize("block_size", [128, 256, 512, 1024])
def test_clip_num_cache_blocks_matches_audio_plan(block_size):
    # Every mel length up to 60 s of audio, and a few longer clips.
    mel_lens = list(range(1, 6001)) + [9999, 12345, 30001]
    plan = plan_audio_blocks(
        mel_lens,
        block_size=block_size,
        window_tokens=104,
        n_window=50,
        n_window_infer=800,
        one_item_per_block=True,
    )
    expected = [
        math.ceil(len(slots) / plan.slots_per_block) for slots in plan.clip_slots
    ]
    actual = [
        clip_num_cache_blocks(
            int(_get_feat_extract_output_lengths(torch.tensor(mel_len))),
            block_size=block_size,
            window_tokens=104,
            n_window=50,
            n_window_infer=800,
        )
        for mel_len in mel_lens
    ]
    assert actual == expected


def test_qwen3_asr_long_clip_takes_more_than_dense_blocks():
    # 12345 mel frames = 1605 embeds: 15 full 104-token windows and a tail. A
    # 128-token block holds one window, so the clip takes 16 blocks, not
    # ceil(1605 / 128) = 13.
    feature = _audio_feature(12345, "shm")
    num_embeds = feature.mm_position.get_num_embeds()
    blocks = get_encoder_cache_item_blocks(QWEN3_ASR, _encoder_config(128), feature)
    assert num_embeds == 1605
    assert blocks == 16 > math.ceil(num_embeds / 128)
