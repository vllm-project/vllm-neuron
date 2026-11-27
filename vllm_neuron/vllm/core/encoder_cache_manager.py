# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side encoder cache manager that counts on-device cache blocks.

vLLM's ``EncoderCacheManager`` sizes the encoder cache, and decides when to
evict, by embed count. On Neuron the worker stores encoder outputs in
``EncoderCacheBlocks``, where every item starts on a fresh block, so an item of
``n`` embeds occupies at least ``ceil(n / block_size)`` blocks. Counting embeds,
the scheduler keeps more finished items cached than the device has blocks for,
and ``EncoderCacheBlocks.allocate`` then fails with "Encoder cache full".

``NeuronEncoderCacheManager`` keeps vLLM's logic (cache hits, freeable entries,
oldest-first eviction) and changes only the unit: capacity is the device's
allocatable block count, and each item is charged the blocks the encoder
allocates for it. The encoder compute budget stays in embeds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from vllm.v1.core.encoder_cache_manager import EncoderCacheManager

if TYPE_CHECKING:
    from collections.abc import Callable

    from vllm.config import VllmConfig
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.v1.request import Request


class _BlockCountRequest:
    """View of a request whose ``get_num_encoder_embeds`` returns cache blocks.

    The base manager reads an item's size only through
    ``request.get_num_encoder_embeds``. Passing this view to the base methods
    makes them count blocks. Every other attribute is the request's own.

    NOTE: coupled to upstream ``vllm.v1.core.encoder_cache_manager.
    EncoderCacheManager`` reading item sizes through
    ``request.get_num_encoder_embeds``. If a later vLLM reads them elsewhere
    (for example ``mm_position.get_num_embeds()``), charges fall back to embeds
    against a capacity in blocks and requests stall. The CPU tests in
    ``test/unit/vllm/core/test_encoder_cache_manager.py`` run the real base
    class and fail in that case.
    """

    __slots__ = ("_request", "_item_blocks")

    def __init__(
        self,
        request: Request,
        item_blocks: Callable[[MultiModalFeatureSpec], int],
    ) -> None:
        self._request = request
        self._item_blocks = item_blocks

    def __getattr__(self, name: str) -> Any:
        return getattr(self._request, name)

    def get_num_encoder_embeds(self, input_id: int) -> int:
        return self._item_blocks(self._request.mm_features[input_id])


class NeuronEncoderCacheManager(EncoderCacheManager):
    """``EncoderCacheManager`` that counts on-device encoder-cache blocks.

    ``cache_size``, ``num_free_slots`` and ``num_freeable_slots`` are in blocks.

    Args:
        num_blocks: Allocatable blocks of the device cache (scratch excluded).
        item_blocks: Returns the blocks the encoder allocates for one item.
    """

    @classmethod
    def create_manager(
        cls, *, cache_size: int, vllm_config: VllmConfig
    ) -> NeuronEncoderCacheManager:
        """Build the manager the way the model runner sizes the device cache.

        Args:
            cache_size: The scheduler's encoder cache budget in embeds, the
                ``encoder_cache_size`` the runner derives its block count from.
            vllm_config: The engine configuration. Its additional_config must
                hold an ``encoder_neuron_config``.
        """
        from vllm_neuron.model.neuron_config import EncoderNeuronConfig
        from vllm_neuron.utils.mm_utils import (
            get_encoder_cache_block_size,
            get_encoder_cache_item_blocks,
            get_encoder_cache_num_blocks,
        )

        hf_config = vllm_config.model_config.hf_config
        encoder_neuron_config = EncoderNeuronConfig.from_dict(
            vllm_config.additional_config["encoder_neuron_config"]
        )
        cache_block_size = get_encoder_cache_block_size(
            hf_config, encoder_neuron_config.encoder_attention_block_size
        )
        num_blocks = get_encoder_cache_num_blocks(
            cache_size,
            cache_block_size,
            encoder_neuron_config.encoder_cache_num_blocks,
        )

        def item_blocks(mm_feature: MultiModalFeatureSpec) -> int:
            return get_encoder_cache_item_blocks(
                hf_config, encoder_neuron_config, mm_feature
            )

        # One block is scratch: the encoder writes padding there, and it never
        # holds an item.
        return cls(num_blocks=num_blocks - 1, item_blocks=item_blocks)

    def __init__(
        self,
        num_blocks: int,
        item_blocks: Callable[[MultiModalFeatureSpec], int],
    ) -> None:
        super().__init__(cache_size=num_blocks)
        self._item_blocks = item_blocks
        # Blocks of the items that can_allocate approved earlier in the current
        # _try_schedule_encoder_inputs call for this request. The scheduler
        # passes the same items as num_embeds_to_schedule, but in embeds.
        self._pending_request_id: str | None = None
        self._pending_blocks = 0

    def _view(self, request: Request) -> Request:
        return cast("Request", _BlockCountRequest(request, self._item_blocks))

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        # The compute budget counts embeds the encoder runs on this step.
        if request.get_num_encoder_embeds(input_id) > encoder_compute_budget:
            return False

        # num_embeds_to_schedule is 0 on the first call for a request in a
        # scheduling step, so a nonzero value always refers to items approved
        # just before for the same request.
        pending = 0
        if num_embeds_to_schedule and self._pending_request_id == request.request_id:
            pending = self._pending_blocks
        num_blocks = self._item_blocks(request.mm_features[input_id])
        # The compute budget is already checked above, so pass the item's own
        # block count to let the base check pass on it.
        if not super().can_allocate(self._view(request), input_id, num_blocks, pending):
            return False
        self._pending_request_id = request.request_id
        self._pending_blocks = pending + num_blocks
        return True

    def allocate(self, request: Request, input_id: int) -> None:
        super().allocate(self._view(request), input_id)

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        super().free_encoder_input(self._view(request), input_id)

    def reset(self) -> None:
        super().reset()
        self._pending_request_id = None
        self._pending_blocks = 0
