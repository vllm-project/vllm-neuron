# SPDX-License-Identifier: Apache-2.0
"""``NeuronNixlConnector`` for Hy4's key-only KV cache (disaggregated inference).

vllm-neuron's ``NeuronNixlConnector`` registers every layer's cache as two NIXL regions,
the K and the V slab of a ``(2, num_blocks, heads, block, head)`` allocation. Hy4's
cache is key-only -- ``(1, num_blocks, 1, block, 704)``, one ``[index | latent | rope]``
entry per token -- so ``register_kv_caches`` asserts on it. This subclass registers a
key-only layer as ONE region whose block is the whole page; everything after
registration (descriptor ids ``region * num_blocks + block``, the TP / DCP read logic)
is region-count generic and unchanged.

``register_kv_caches`` is the base method from vllm-neuron 0.29.0.1.2.0 with only the
region loop changed (marked ``HY4``). Select it on both servers with::

    --kv-transfer-config '{"kv_connector": "Hy4NixlConnector",
        "kv_connector_module_path": "hy4_preview.kv_connector", ...}'
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING

from vllm_neuron.vllm.kv_connector import neuron_nixl_connector as base
from vllm_neuron.vllm.kv_connector.neuron_nixl_connector import (  # noqa: F401
    NeuronNixlAgentMetadata,
    NixlConnector,
    NixlHandshakePayload,
    _agent_metadata_kwargs,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = logging.getLogger(__name__)


class Hy4NixlConnectorWorker(base.NeuronNixlConnectorWorker):
    def register_kv_caches(self, kv_caches):
        """Register Neuron KV caches as two NIXL regions per layer (K, V).

        The base connector assumes one blocks-first region per layer with
        K/V packed into the content dim. Neuron allocates
        (2, num_blocks, heads, block_size, head_size) with K and V as two
        separate num_blocks-sized slabs, so one logical block spans two
        non-contiguous byte ranges and cannot be described by a single NIXL
        descriptor. Registering each slab as its own region keeps every
        descriptor a single contiguous span; all downstream desc-id math
        (region * num_blocks + block) is region-count generic on both ends.
        """
        import msgspec
        from vllm.distributed.kv_transfer.kv_connector.utils import (
            TransferTopology,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
            compute_nixl_compatibility_hash,
        )
        from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

        assert not self._has_mamba, "Neuron K/V-slab registration: no mamba support"
        assert not self.use_host_buffer, (
            "Neuron K/V-slab registration assumes device buffers "
            f"(kv_buffer_device={self.kv_buffer_device!r})"
        )

        first_cache = next(iter(kv_caches.values()))
        self.transfer_topo = TransferTopology(
            tp_rank=self.tp_rank,
            tp_size=self.world_size,
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=self.attn_backends,
            # Per-slab shape: ONE K (or V) slab, 4-D blocks-first
            # (num_blocks, num_kv_heads, block_size, head_size). K and V are
            # separate slabs, each registered as its own NIXL region (see
            # register_kv_caches).
            tensor_shape=first_cache[0].shape,
            is_mamba=False,
        )
        # ``cross_layers_blocks`` became a required argument in a later vLLM, so pass it only when
        # the installed one takes it. False, because this backend registers one slab for each K and V.
        _compat_kwargs = {"transfer_mode": self._TRANSFER_MODE}
        if (
            "cross_layers_blocks"
            in inspect.signature(compute_nixl_compatibility_hash).parameters
        ):
            _compat_kwargs["cross_layers_blocks"] = False
        self.compat_hash = compute_nixl_compatibility_hash(
            self.vllm_config,
            self.backend_name,
            **_compat_kwargs,
        )

        # A later vLLM keeps ``block_len_per_layer`` and no longer creates
        # ``block_stride_per_layer``. Create whichever is absent, and never overwrite the base's.
        for _attr in ("block_len_per_layer", "block_stride_per_layer"):
            if not hasattr(self, _attr):
                setattr(self, _attr, [])

        caches_data = []
        seen_base_addresses: list[int] = []
        slab_size_bytes = None

        for layer_name, cache in kv_caches.items():
            # _layer_specs is populated by NixlBaseConnectorWorker.__init__
            # (from the KV cache specs at worker construction), not by the
            # base register_kv_caches -- safe to read without super().
            layer_spec = self._layer_specs.get(layer_name)
            if layer_spec is None:
                # Layer shares its KV cache tensor with another layer.
                continue
            if isinstance(layer_spec, UniformTypeKVCacheSpecs):
                layer_spec = layer_spec.kv_cache_specs[layer_name]

            assert cache.dim() == 5 and cache.shape[0] in (1, 2), (
                f"Expected (2 or 1, num_blocks, ...) Neuron KV cache for "
                f"{layer_name}, got shape {tuple(cache.shape)}"
            )
            assert cache.shape[1] == self.num_blocks, (
                f"KV cache num_blocks mismatch for {layer_name}: "
                f"expected {self.num_blocks}, got shape {tuple(cache.shape)}"
            )
            # HY4: a key-only cache (1, num_blocks, ...) is ONE slab whose block is the
            # whole page; a K/V cache is two slabs of half a page each.
            num_slabs = cache.shape[0]
            half_page = layer_spec.page_size_bytes // num_slabs

            for slab in tuple(cache[i] for i in range(num_slabs)):
                base_addr = slab.data_ptr()
                if base_addr in seen_base_addresses:
                    # Pooled tensor shared across layers: register once.
                    continue
                seen_base_addresses.append(base_addr)
                self.block_len_per_layer.append(half_page)
                self.block_stride_per_layer.append(half_page)
                self._region_is_mla.append(False)

                curr_size_bytes = self.num_blocks * half_page
                if slab_size_bytes is None:
                    slab_size_bytes = curr_size_bytes
                assert slab_size_bytes == curr_size_bytes, (
                    "All K/V slabs must have the same size"
                )
                self.device_id = max(cache.get_device(), 0)
                caches_data.append((base_addr, curr_size_bytes, self.device_id, ""))

        self.kv_caches_base_addr[self.engine_id][self.tp_rank] = seen_base_addresses
        self.num_regions = len(caches_data)
        self.num_descs = self.num_regions * self.num_blocks

        if self.pp_size > 1:
            start_layer, end_layer = self.model_config.get_layers_start_end_indices(
                self.vllm_config.parallel_config
            )
            num_local_layers = end_layer - start_layer
            assert num_local_layers > 0 and self.num_regions % num_local_layers == 0
            self._remote_region_offset = (
                self.num_regions // num_local_layers
            ) * start_layer

        descs = self.nixl_wrapper.get_reg_descs(caches_data, self.nixl_memory_type)
        self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
        self._registered_descs.append(descs)
        logger.info(
            "Registered %d KV slab regions with %d descs",
            self.num_regions,
            self.num_descs,
        )

        self.device_kv_caches = kv_caches
        self.dst_num_blocks[self.engine_id] = self.num_blocks

        # Local/src descriptors for NIXL xfer.
        (
            self.src_xfer_handles_by_block_size[self.block_size],
            self.src_blocks_data,
        ) = self.register_local_xfer_handler(self.block_size)

        agent_metadata = NeuronNixlAgentMetadata(
            **_agent_metadata_kwargs(
                engine_id=self.engine_id,
                agent_metadata=self.nixl_wrapper.get_agent_metadata(),
                device_id=self.device_id,
                kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id][
                    self.tp_rank
                ],
                num_blocks=self.num_blocks,
                block_lens=self.block_len_per_layer,
                block_strides=self.block_stride_per_layer,
                kv_cache_layout=self.kv_cache_layout,
                block_size=self.block_size,
                ssm_sizes=self._mamba_ssm_size,
                attn_backend_name=self.backend_name,
                physical_blocks_per_logical_kv_block=(
                    self._physical_blocks_per_logical_kv_block
                ),
            )
        )
        assert self.compat_hash is not None
        encoder = msgspec.msgpack.Encoder()
        self.xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=encoder.encode(agent_metadata),
        )

        dcp = self.vllm_config.parallel_config.decode_context_parallel_size
        is_prefill = self.kv_transfer_config.kv_role == "kv_producer"

        if dcp > 1 and is_prefill:
            agent_meta = NeuronNixlAgentMetadata(
                **_agent_metadata_kwargs(
                    engine_id=self.engine_id,
                    agent_metadata=self.nixl_wrapper.get_agent_metadata(),
                    device_id=self.device_id,
                    kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id][
                        self.tp_rank
                    ],
                    num_blocks=self.num_blocks,
                    block_lens=self.block_len_per_layer,
                    block_strides=self.block_stride_per_layer,
                    kv_cache_layout=self.kv_cache_layout
                    if not self.use_host_buffer
                    else self.host_buffer_kv_cache_layout,
                    block_size=self.block_size,
                    ssm_sizes=self._mamba_ssm_size,
                    attn_backend_name="neuron",
                    physical_blocks_per_logical_kv_block=1,
                    dcp_size=dcp,
                )
            )
            encoder = msgspec.msgpack.Encoder()
            self.xfer_handshake_metadata = NixlHandshakePayload(
                compatibility_hash=self.compat_hash,
                agent_metadata_bytes=encoder.encode(agent_meta),
            )
            logger.info("DCP NIXL: prefill metadata tagged with dcp_size=%d", dcp)
        else:
            # Decode: store local DCP rank for block filtering. The DCP group
            # may not be initialized on non-DCP deployments.
            from vllm.distributed.parallel_state import get_dcp_group

            try:
                g = get_dcp_group()
                self._local_dcp_size = g.world_size
                self._local_dcp_rank = g.rank_in_group
            except AssertionError:
                self._local_dcp_size = 1
                self._local_dcp_rank = 0

        # Patch tp_ratio to always return 1, preventing upstream from
        # applying its own head splitting logic. We handle all DCP
        # topologies in our _read_blocks_for_req / add_remote_agent.
        if self.transfer_topo is not None:
            self.transfer_topo.tp_ratio = lambda remote_tp_size: 1

        if self._local_dcp_size > 1:
            logger.info(
                "DCP NIXL: local decode dcp_size=%d, dcp_rank=%d, tp_rank=%d",
                self._local_dcp_size,
                self._local_dcp_rank,
                self.tp_rank,
            )

    # ── Override: _validate_remote_agent_handshake ────────────────────────


class Hy4NixlConnector(base.NeuronNixlConnector):
    """``NeuronNixlConnector`` whose worker registers key-only caches."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
            KVConnectorRole,
        )

        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.kv_cache_config = kv_cache_config
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self.kv_transfer_config = vllm_config.kv_transfer_config

        if role == KVConnectorRole.SCHEDULER:
            # Subclass carries the DI sliding-window fix. See
            # NeuronNixlConnectorScheduler.
            self.connector_scheduler = base.NeuronNixlConnectorScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = Hy4NixlConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )
