# SPDX-License-Identifier: Apache-2.0
"""Muse-Glimmer fixed vision shapes, encoder sizing hooks, and registration (no device needed)."""

from types import SimpleNamespace

import pytest
from vllm import ModelRegistry

from muse_glimmer_neuron import ARCH, shapes
from muse_glimmer_neuron.factory import MuseGlimmerForConditionalGeneration
from vllm_neuron.model.neuron_config import EncoderNeuronConfig
from vllm_neuron.model.registry import get_models

BUCKETS = [1024, 1564, 1632, 2080]


def _hf_config():
    return SimpleNamespace(
        architectures=[ARCH], vision_config=SimpleNamespace(merge_size=2)
    )


def _encoder_config():
    return EncoderNeuronConfig(
        encoder_attention_block_size=1024, num_encoder_input_buckets=BUCKETS
    )


def test_every_shape_has_its_own_patch_count():
    # The vision graph is picked by patch count, so the counts must differ.
    assert sorted(shapes.GRIDS) == BUCKETS
    assert shapes.GRIDS[1024] == (1, 32, 32)
    assert shapes.GRIDS[1564] == (1, 34, 46)
    assert shapes.GRIDS[1632] == (1, 48, 34)
    assert shapes.GRIDS[2080] == shapes.VIDEO_GRID


def test_token_counts():
    assert shapes.IMAGE_TOKENS == 408  # portrait 476x672
    assert shapes.IMAGE_TOKENS_MIN == 256  # square 448x448
    assert shapes.VIDEO_TOKENS == 520
    assert shapes.IMAGE_SIZE_MOST_TOKENS == (476, 672)


@pytest.mark.parametrize(
    "size, expected",
    [
        ((640, 480), (644, 476)),
        ((640, 427), (644, 476)),
        ((500, 375), (644, 476)),
        ((480, 640), (476, 672)),
        ((427, 640), (476, 672)),
        ((500, 500), (448, 448)),
        ((612, 612), (448, 448)),
    ],
)
def test_pick_image_shape_keeps_the_aspect_ratio(size, expected):
    assert shapes.pick_image_shape(*size) == expected


def test_shapes_are_multiples_of_the_merged_patch():
    # The transformers processor keeps a size that is a multiple of 28
    # (patch 14 x merge 2), so the fixed shapes pass through unchanged.
    for w, h in shapes.IMAGE_SHAPES:
        assert w % 28 == 0 and h % 28 == 0


def test_register_adds_the_model():
    # test/conftest.py calls register()
    assert dict(get_models())[ARCH] is MuseGlimmerForConditionalGeneration
    assert ARCH in ModelRegistry.get_supported_archs()


def test_merge_factor_and_encoder_budget():
    cls = MuseGlimmerForConditionalGeneration
    assert cls.get_vision_token_merge_factor(_hf_config()) == 4
    assert cls.get_max_encoder_embed_budget(_hf_config(), _encoder_config()) == 520


def test_encoder_blocks_per_request():
    # 520 embeddings (the video, the largest item) in cache blocks of
    # encoder_attention_block_size / merge factor = 256 tokens.
    cls = MuseGlimmerForConditionalGeneration
    assert cls.get_max_encoder_blocks_per_request(_hf_config(), _encoder_config()) == 3


def test_rejects_unknown_quantization():
    with pytest.raises(ValueError):
        MuseGlimmerForConditionalGeneration.from_configs(
            SimpleNamespace(), SimpleNamespace(quantization="fp8")
        )
