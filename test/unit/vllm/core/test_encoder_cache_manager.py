# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side encoder cache accounting against the on-device block cache.

``EncoderCacheBlocks`` stores every item from a fresh block. These tests drive
``NeuronScheduler``'s encoder cache manager and the device cache together, the
way vLLM's scheduling loop and ``NeuronModelRunner`` do, and check that the
scheduler never keeps more items cached than the device has blocks for.

Sizes follow the Qwen3-VL tutorial: encoder_attention_block_size 2048 raw tokens
with merge factor 4 gives 512 embeds per cache block, and encoder_cache_size
16384 (2 * max_num_batched_tokens) gives 32 allocatable blocks plus one scratch
block.
"""

import math
import types
from unittest import mock

import pytest
import torch
from vllm.config import EncoderCacheManagerConfig
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_neuron.vllm.core.scheduler import NeuronScheduler
from vllm_neuron.vllm.worker.encoder_cache_blocks import EncoderCacheBlocks

ENCODER_ATTENTION_BLOCK_SIZE = 2048
CACHE_BLOCK_SIZE = 512
CACHE_SIZE = 16384
# The runner's default block count: ceil(CACHE_SIZE / CACHE_BLOCK_SIZE) + 1.
NUM_DEVICE_BLOCKS = 33


def _encoder_neuron_config(**overrides) -> dict:
    config = {
        "num_encoder_input_buckets": [ENCODER_ATTENTION_BLOCK_SIZE],
        "encoder_attention_block_size": ENCODER_ATTENTION_BLOCK_SIZE,
    }
    config.update(overrides)
    return config


def _vllm_config(encoder_neuron_config=None, ec_manager_config=None):
    additional_config = {}
    if encoder_neuron_config is not None:
        additional_config["encoder_neuron_config"] = encoder_neuron_config
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            model="Qwen/Qwen3-VL-8B-Instruct",
            max_model_len=8192,
            hf_config=types.SimpleNamespace(
                architectures=["Qwen3VLForConditionalGeneration"],
                vision_config=types.SimpleNamespace(spatial_merge_size=2),
            ),
        ),
        scheduler_config=types.SimpleNamespace(
            policy="fcfs", max_num_seqs=4, max_num_batched_tokens=8192
        ),
        parallel_config=types.SimpleNamespace(decode_context_parallel_size=1),
        cache_config=types.SimpleNamespace(block_size=32),
        kv_transfer_config=None,
        additional_config=additional_config,
        ec_manager_config=ec_manager_config or EncoderCacheManagerConfig(),
    )


def _neuron_scheduler(
    vllm_config, cache_size: int = CACHE_SIZE, base_manager_cls=EncoderCacheManager
) -> NeuronScheduler:
    """Build a NeuronScheduler over the encoder cache the base scheduler builds.

    vLLM's ``Scheduler.__init__`` needs a full engine config, so it is replaced
    by the part that matters here: with no ``ec_manager_config`` it builds
    ``EncoderCacheManager(cache_size=MultiModalBudget.encoder_cache_size)``.
    The rest of ``NeuronScheduler.__init__`` runs as is.
    """

    def base_init(self, vllm_config, **kwargs):
        self.encoder_cache_manager = base_manager_cls(cache_size=cache_size)
        self.max_num_encoder_input_tokens = cache_size
        self.structured_output_manager = None

    with mock.patch.object(Scheduler, "__init__", base_init):
        return NeuronScheduler(
            vllm_config=vllm_config,
            kv_cache_config=types.SimpleNamespace(kv_cache_groups=[], num_blocks=1024),
            structured_output_manager=None,
            block_size=32,
        )


def _image(identifier: str, num_embeds: int) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=0, length=num_embeds),
    )


def _video(identifier: str, t: int, h: int, w: int, data) -> MultiModalFeatureSpec:
    """A Qwen3-VL video item with the placeholder vLLM's processor builds."""
    from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor

    video_token_id = 151656
    repl = Qwen3VLMultiModalProcessor.get_video_repl(
        tokens_per_frame=[h * w // 4] * t,
        timestamps=[1.0 + 2 * i for i in range(t)],
        tokenizer=types.SimpleNamespace(
            encode=lambda text, add_special_tokens=False: [7, 8]
        ),
        vision_start_token_id=5,
        vision_end_token_id=6,
        video_token_id=video_token_id,
        select_token_id=True,
    )
    return MultiModalFeatureSpec(
        data=data,
        modality="video",
        identifier=identifier,
        mm_position=PlaceholderRange(
            offset=0, length=len(repl.full), is_embed=repl.is_embed(repl.full)
        ),
    )


def _request(request_id: str, sizes: list[int]):
    features = [_image(f"{request_id}-{i}", n) for i, n in enumerate(sizes)]
    return _request_of(request_id, features)


def _request_of(request_id: str, features: list[MultiModalFeatureSpec]):
    return types.SimpleNamespace(
        request_id=request_id,
        mm_features=features,
        get_num_encoder_embeds=lambda i: features[i].mm_position.get_num_embeds(),
    )


def _device_cache(num_blocks: int = NUM_DEVICE_BLOCKS) -> EncoderCacheBlocks:
    # Monolithic serving: no hold time, freed blocks return at once.
    return EncoderCacheBlocks(
        num_blocks=num_blocks,
        block_size=CACHE_BLOCK_SIZE,
        fat_dim=4,
        min_hold_time_ms=0,
    )


def _serve_one_item_requests(
    manager, cache, sizes: list[int], compute_budget: int, used: list[int]
) -> None:
    """Serve one single-image request at a time, as in the device reproduction."""
    items = [
        (
            _image(f"req{k}-0", num_embeds),
            EncoderCacheBlocks.dense_tokens_per_block(num_embeds, CACHE_BLOCK_SIZE),
        )
        for k, num_embeds in enumerate(sizes)
    ]
    _serve_items(manager, cache, items, compute_budget, used)


def _serve_items(manager, cache, items, compute_budget: int, used: list[int]) -> None:
    """Serve one single-item request per ``(feature, tokens_per_block)`` item.

    Each request is scheduled, encoded and finished before the next one starts,
    so its item stays cached on both sides until the scheduler evicts it. The
    encoder allocates ``tokens_per_block``. The device blocks in use after each
    request are appended to ``used``.
    """
    for k, (feature, tokens_per_block) in enumerate(items):
        req = _request_of(f"req{k}", [feature])
        # Scheduler._try_schedule_encoder_inputs, then allocate after scheduling.
        assert not manager.check_and_update_cache(req, 0)
        assert manager.can_allocate(req, 0, compute_budget, 0)
        manager.allocate(req, 0)
        # The runner frees the evicted items, then the encoder allocates blocks.
        for mm_hash in manager.get_freed_mm_hashes():
            cache.free(mm_hash)
        cache.allocate(feature.identifier, tokens_per_block)
        # The request finishes; its item stays cached until evicted.
        manager.free(req)
        used.append(cache.num_active_blocks)


def test_scheduler_keeps_distinct_images_within_device_blocks():
    """40 distinct 256-embed images, one request at a time.

    Each image fills one 512-embed block. Counting embeds, the scheduler holds
    64 such images in its 16384-embed budget, but the device has 32 blocks, so
    the 33rd image raised "Encoder cache full". Counting blocks, the scheduler
    evicts the oldest image from then on.
    """
    scheduler = _neuron_scheduler(_vllm_config(_encoder_neuron_config()))
    used: list[int] = []
    _serve_one_item_requests(
        scheduler.encoder_cache_manager,
        _device_cache(),
        [256] * 40,
        scheduler.max_num_encoder_input_tokens,
        used,
    )
    assert used == list(range(1, 33)) + [32] * 8


def test_scheduler_mixed_item_sizes_within_device_blocks():
    scheduler = _neuron_scheduler(_vllm_config(_encoder_neuron_config()))
    sizes = [64, 513, 256, 1, 8192, 511, 512, 1500, 3000, 1024, 16384, 77] * 4
    used: list[int] = []
    _serve_one_item_requests(
        scheduler.encoder_cache_manager, _device_cache(), sizes, CACHE_SIZE, used
    )
    assert max(used) == NUM_DEVICE_BLOCKS - 1


@pytest.mark.parametrize("cache_type", ["lru", "shm"])
def test_scheduler_keeps_videos_within_device_blocks(cache_type):
    """17 distinct 4x32x34 videos (1088 embeds), one request at a time.

    The encoder puts whole 1088-token frames (272 embeds) in 2048-token blocks,
    one frame per block, so each video takes 4 blocks where the dense layout
    gives 3. With the
    shm processor cache the scheduler's item holds only a shared-memory address,
    so the block count has to come from the placeholder.
    """
    if cache_type == "shm":
        data = MultiModalKwargsItem(
            {
                "address": MultiModalFieldElem(data=0, field=MultiModalBatchedField()),
                "monotonic_id": MultiModalFieldElem(
                    data=0, field=MultiModalBatchedField()
                ),
            }
        )
    else:
        data = MultiModalKwargsItem(
            {
                "video_grid_thw": MultiModalFieldElem(
                    data=torch.tensor([4, 32, 34]), field=MultiModalBatchedField()
                )
            }
        )
    # What embed_multimodal allocates: one frame per block.
    layout = [272] * 4
    items = [(_video(f"video{k}", 4, 32, 34, data), layout) for k in range(17)]
    # Encoder buckets [2048, 8192], as in the device video run, let one step
    # encode a whole 4352-token video.
    scheduler = _neuron_scheduler(
        _vllm_config(_encoder_neuron_config(num_encoder_input_buckets=[2048, 8192]))
    )
    used: list[int] = []
    _serve_items(
        scheduler.encoder_cache_manager,
        _device_cache(),
        items,
        scheduler.max_num_encoder_input_tokens,
        used,
    )
    assert used == list(range(4, 33, 4)) + [32] * 9


@pytest.mark.parametrize(
    "configured, expected_capacity",
    [(None, 32), (64, 63), (8, 32)],
)
def test_capacity_is_device_allocatable_blocks(configured, expected_capacity):
    from vllm_neuron.utils.mm_utils import get_encoder_cache_num_blocks

    overrides = {}
    if configured is not None:
        overrides["encoder_cache_num_blocks"] = configured
    scheduler = _neuron_scheduler(_vllm_config(_encoder_neuron_config(**overrides)))
    # The runner sizes EncoderCacheBlocks with the same call, plus one scratch.
    device_blocks = get_encoder_cache_num_blocks(
        CACHE_SIZE, CACHE_BLOCK_SIZE, configured
    )
    capacity = scheduler.encoder_cache_manager.cache_size
    assert capacity == device_blocks - 1 == expected_capacity


def test_scheduler_keeps_vllm_manager_without_device_cache():
    # No encoder_neuron_config: the runner builds no device cache.
    scheduler = _neuron_scheduler(_vllm_config())
    assert type(scheduler.encoder_cache_manager) is EncoderCacheManager

    # All modalities disabled: the encoder cache budget is 0.
    scheduler = _neuron_scheduler(_vllm_config(_encoder_neuron_config()), cache_size=0)
    assert type(scheduler.encoder_cache_manager) is EncoderCacheManager

    # Encoder-decoder models keep vLLM's encoder-decoder manager.
    scheduler = _neuron_scheduler(
        _vllm_config(_encoder_neuron_config()),
        base_manager_cls=EncoderDecoderCacheManager,
    )
    assert type(scheduler.encoder_cache_manager) is EncoderDecoderCacheManager

    # A manager the user configured through ec_manager_config is kept.
    user = EncoderCacheManagerConfig(encoder_cache_manager_cls="pkg.UserManager")
    scheduler = _neuron_scheduler(
        _vllm_config(_encoder_neuron_config(), ec_manager_config=user)
    )
    assert type(scheduler.encoder_cache_manager) is EncoderCacheManager


def test_items_of_one_request_counted_in_blocks():
    """Items approved earlier in the same scheduling call count in blocks."""
    from vllm_neuron.vllm.core.encoder_cache_manager import NeuronEncoderCacheManager

    def item_blocks(mm_feature):
        return math.ceil(mm_feature.mm_position.get_num_embeds() / CACHE_BLOCK_SIZE)

    manager = NeuronEncoderCacheManager(num_blocks=4, item_blocks=item_blocks)
    req = _request("a", [513, 513, 513])
    # 2 blocks, then 4 blocks. The third item needs 6 blocks, although its
    # 1539 embeds would fit 4 blocks' worth (2048) of tokens.
    assert manager.can_allocate(req, 0, 4096, 0)
    assert manager.can_allocate(req, 1, 4096 - 513, 513)
    assert not manager.can_allocate(req, 2, 4096 - 1026, 1026)
    manager.allocate(req, 0)
    manager.allocate(req, 1)
    assert manager.num_free_slots == 0

    # The compute budget is still in embeds.
    assert not manager.can_allocate(_request("b", [600]), 0, 599, 0)

    # A finished request leaves its items freeable. A new item evicts the
    # oldest one only, and a new request does not inherit earlier pending items.
    manager.free(req)
    assert manager.num_freeable_slots == 4
    assert manager.can_allocate(_request("c", [1024]), 0, 4096, 0)
    assert manager.get_freed_mm_hashes() == ["a-0"]
    assert manager.num_free_slots == 2
