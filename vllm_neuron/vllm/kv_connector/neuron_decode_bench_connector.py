# SPDX-License-Identifier: Apache-2.0
"""
NeuronDecodeBenchConnector — DecodeBenchConnector adapted for Neuron devices.

A KV connector is vLLM's plug-in point for "this request's KV cache comes from
somewhere external" — in real disaggregated inference (DI) that source is the
prefill node shipping KV to the decode node. This connector abuses that
interface to *pretend a prefill already happened*, so decode can be benchmarked
in isolation (no prefill compute, no prefill server, no prefill/decode
contamination in the numbers).

Despite the "DecodeBench" name, the load-bearing behavior is skipping prefill,
not filling KV. What each piece does, and why it is needed:

1. Scheduler reports the whole prompt as already-present external KV
   (get_num_new_matched_tokens returns (all_uncomputed_tokens, is_async=True)).
   → vLLM skips prefill and sends the request straight to the decode loop.
     Without it, every request runs a real prefill and the "decode" benchmark
     measures prefill+decode mixed together (and needs a prefill NEFF/server).

2. is_async=True (upstream returns False) makes the first scheduled step a
   clean 1-token decode rather than a 1-token "prefill".
   → With kv_role=kv_consumer there is no prefill warmup, so no prefill NEFF
     was compiled; a 1-token prefill step would dispatch a graph that does not
     exist and fail. (This also pairs with the scheduler-side last-token
     reclassify in vllm/core/scheduler.py.)

3. Worker builds the HMA (hybrid KV cache manager) group-to-layers mapping
   from kv_cache_config (register_kv_caches).
   → GPT-OSS splits KV into multiple groups (e.g. full-attention vs
     sliding-window) with different block layouts. The upstream parent assumes
     a single group; on a hybrid model that mapping is wrong. This is the one
     override that does real functional work on Neuron.

4. Worker tracks "filled" requests and reports them finished (start_fill_kv /
   get_finished) to satisfy the connector handshake.
   → Without the "transfer complete" signal the engine waits forever for KV
     that never arrives and requests hang instead of decoding.

5. _fill_blocks is a NO-OP. Upstream writes dummy values into the "received"
   KV blocks; on Neuron host-side KV writes fail with nrt_tensor_read (see
   _fill_blocks), so we skip it.
   → Decode runs against uninitialized KV, producing garbage logits — which is
     irrelevant here: the fixed-shape decode NEFF does the same compute
     regardless of KV contents, so throughput is unaffected and correctness is
     not checked. fill_mean/fill_std are therefore inert (only the overridden
     upstream _fill_blocks reads them).

Usage:
    vllm serve <model> --kv-transfer-config '{
        "kv_connector": "NeuronDecodeBenchConnector",
        "kv_connector_module_path":
            "vllm_neuron.vllm.kv_connector.neuron_decode_bench_connector",
        "kv_role": "kv_consumer"
    }'
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

# SupportsHMA and the DecodeBenchConnector* base classes are provided by the
# pinned vllm (vllm==0.21.0 in requirements/core.txt); both paths are present
# there. Re-verify these private vllm.* paths on any vLLM version bump.
from vllm.distributed.kv_transfer.kv_connector.v1 import SupportsHMA
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnector,
    DecodeBenchConnectorMetadata,
    DecodeBenchConnectorScheduler,
    DecodeBenchConnectorWorker,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class NeuronDecodeBenchConnectorScheduler(DecodeBenchConnectorScheduler):
    """Scheduler that reports all tokens as async-external.

    This makes the first scheduled step a pure decode (1 output token) rather
    than a 1-token "prefill" that triggers a missing prefill NEFF when
    kv_role=kv_consumer (no prefill warmup).
    """

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        req_id = request.request_id

        if req_id in self._filled_requests:
            return 0, False

        num_uncomputed_tokens = request.num_tokens - num_computed_tokens
        if num_uncomputed_tokens == 0:
            return 0, False

        logger.debug(
            "NeuronDecodeBenchConnector[SCHED]: get_num_new_matched_tokens "
            "req=%s num_tokens=%d num_computed=%d returning (%d, True)",
            req_id,
            request.num_tokens,
            num_computed_tokens,
            num_uncomputed_tokens,
        )
        return num_uncomputed_tokens, True


class NeuronDecodeBenchConnectorWorker(DecodeBenchConnectorWorker):
    """Worker side: tracks which requests are "filled" and builds the HMA
    group-to-layers mapping. The actual KV fill is a no-op on Neuron
    (see _fill_blocks)."""

    def __init__(
        self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig | None" = None
    ):
        super().__init__(vllm_config)
        self._kv_cache_config = kv_cache_config
        self._filled_req_ids: set[str] = set()

    def start_fill_kv(self, metadata: DecodeBenchConnectorMetadata):
        """Fill KV and track which requests are done."""
        logger.debug(
            "NeuronDecodeBenchConnector[WORKER]: start_fill_kv reqs=%s",
            list(metadata.reqs_to_fill.keys()) if metadata.reqs_to_fill else [],
        )
        for req_id, (block_ids_per_group, num_tokens) in metadata.reqs_to_fill.items():
            logger.debug(
                "  req=%s num_tokens=%d groups=%d block_counts=%s",
                req_id,
                num_tokens,
                len(block_ids_per_group),
                [len(b) for b in block_ids_per_group],
            )
        super().start_fill_kv(metadata)
        self._filled_req_ids.update(metadata.reqs_to_fill.keys())
        logger.debug(
            "NeuronDecodeBenchConnector[WORKER]: fill complete, reporting %d done",
            len(self._filled_req_ids),
        )

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Report all filled requests as finished receiving."""
        done = self._filled_req_ids.copy()
        self._filled_req_ids.clear()
        if done:
            logger.debug(
                "NeuronDecodeBenchConnector[WORKER]: get_finished returning %d recving done",
                len(done),
            )
        return set(), done

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Build correct group-to-layers mapping for HMA compatibility.

        Intentionally does not call ``super().register_kv_caches``: the parent
        only sets ``self.kv_caches`` and ``self.group_to_layers`` (the latter
        always to a single group 0), both of which we overwrite here with the
        real multi-group HMA mapping. There is no other parent-side state to
        preserve, so skipping super() is safe.
        """
        self.kv_caches = kv_caches
        if self._kv_cache_config is not None:
            self.group_to_layers = {
                idx: group.layer_names
                for idx, group in enumerate(self._kv_cache_config.kv_cache_groups)
            }
        else:
            self.group_to_layers = {0: list(kv_caches.keys())}
        logger.debug(
            "NeuronDecodeBenchConnector[WORKER]: register_kv_caches: "
            "%d layers, %d groups, group_sizes=%s",
            len(kv_caches),
            len(self.group_to_layers),
            {idx: len(names) for idx, names in self.group_to_layers.items()},
        )
        for idx, names in self.group_to_layers.items():
            if names:
                sample_cache = kv_caches.get(names[0])
                shape = sample_cache.shape if sample_cache is not None else "NOT_FOUND"
                logger.debug(
                    "  group %d: %d layers, sample_layer=%s, cache_shape=%s",
                    idx,
                    len(names),
                    names[0],
                    shape,
                )

    def _fill_blocks(self, group_idx: int, block_ids: list[int], num_tokens: int):
        """No-op: we do not fill the KV cache, decode runs against whatever the
        default block allocation contains.

        Filling from host-side Python was attempted but fails with nrt_tensor_read
        errors: on Neuron the KV cache lives in device storage that the NEFF's DMA
        reads, and host-side PyTorch writes don't land in that region. (KV is
        normally written on-device from inside a compiled NEFF, e.g. the model's
        index_put_ during real prefill.)

        This is acceptable here because the benchmark measures decode *throughput*,
        which is set by the fixed-shape decode NEFF, not by KV contents. The
        resulting logits are garbage, but correctness is not checked.

        Note: ``fill_mean``/``fill_std`` from the kv_connector_extra_config are
        therefore inert — they are only read by the upstream _fill_blocks we
        override here.

        A possible alternative (not yet tried) is to fill KV on-device via a
        synthetic prefill, like sharding.py:prefill_fill_kv.
        """
        pass


class NeuronDecodeBenchConnector(DecodeBenchConnector, SupportsHMA):
    """DecodeBenchConnector adapted for Neuron: skips prefill via the scheduler,
    no-op KV fill, async scheduling, and HMA support."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: "KVConnectorRole",
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = NeuronDecodeBenchConnectorScheduler(vllm_config)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = NeuronDecodeBenchConnectorWorker(
                vllm_config, kv_cache_config
            )

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Report filled requests as finished receiving."""
        if self.connector_worker is not None:
            return self.connector_worker.get_finished()
        return None, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None
