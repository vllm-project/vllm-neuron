# SPDX-License-Identifier: Apache-2.0
"""
NeuronNixlConnector — standalone NIXL connector for Neuron with DCP support.

Replaces the monkey-patch approach. Users specify this connector directly:

    --kv-transfer-config '{
        "kv_connector": "NeuronNixlConnector",
        "kv_connector_module_path": "vllm_neuron.vllm.kv_connector.neuron_nixl_connector",
        ...
    }'

Supports all DCP DI topologies via unified head_ratio / seq_ratio math:
  - DCP prefill → TP decode
  - DCP prefill → DCP decode (same or different DCP degrees)
  - TP prefill → DCP decode
  - Standard TP → TP (passthrough)
"""

from __future__ import annotations

import inspect
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
    NixlConnectorMetadata,
    NixlHandshakePayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.scheduler import (
    NixlConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
    TPMapping,
)
from vllm.logger import init_logger

from vllm_neuron import envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorRole,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Serialises the binding swap below, which is process-global for the length of one block. Two
# connectors built at once would otherwise each restore the other's wrapper.
_NIXL_AGENT_SWAP_LOCK = threading.Lock()


@contextmanager
def _nixl_agent_config_overrides():
    """Build NIXL's agent with Neuron's config overrides, for the duration of this block.

    Two fields of ``nixlAgentConfig`` are set here, because ``nixl._api.nixl_agent.__init__``
    fills the config itself and the base worker passes no argument that reaches either one. The
    config object, at the moment NIXL consumes it, is the only place left.

    Thread sync (``VLLM_NEURON_NIXL_THREAD_SYNC_RW``, on by default). The base worker builds the
    agent without a ``sync_mode``, so NIXL picks ``NIXL_THREAD_SYNC_NONE`` and takes no lock of
    its own. Two threads use the agent anyway: the handshake executor loads each new remote
    (``loadRemoteMD``, ``prepXferDlist``) while the main thread posts reads (``postXferReq``,
    which releases the GIL). The remote-agent map is then read during a rehash, and the lookup
    either fails with ``NIXL_ERR_NOT_FOUND`` or segfaults the worker. ``RW`` takes a writer lock
    for the metadata calls and a reader lock for the transfer calls. A mode NIXL already picked
    for itself (``STRICT``, with a listen thread) is kept.

    Progress-thread delay (``VLLM_NEURON_NIXL_PROGRESS_THREAD_DELAY_US``, off by default).
    NIXL's libfabric progress thread loops ``progressActiveRails(); sleep_for(pthrDelay)``, and
    NIXL defaults ``pthrDelay`` to 0, so it never sleeps: an agent holds a core even with no
    transfers posted. There is one agent per rank, so a wide deployment asks for close to a core
    per rank, which on a GPT-OSS 120B decode engine at DP16 x TP8 is most of the host. A small
    delay still reaps far more often than once a step. It is opt-in for a performance run, not a
    service default.

    Yields:
        None.

    Example:
        >>> with _nixl_agent_config_overrides():  # doctest: +SKIP
        ...     super().__init__(vllm_config, engine_id, kv_cache_config)
    """
    delay_us = envs.VLLM_NEURON_NIXL_PROGRESS_THREAD_DELAY_US
    sync_rw = envs.VLLM_NEURON_NIXL_THREAD_SYNC_RW
    if delay_us <= 0 and not sync_rw:
        yield
        return

    try:
        from nixl import _api as nixl_api

        bindings = nixl_api.nixlBind
        if not hasattr(bindings, "nixlAgent"):
            raise AttributeError("nixlBind exposes no nixlAgent")
    except Exception as exc:
        logger.warning(
            "Neuron: cannot set the NIXL agent config (%s). The agent keeps NIXL's defaults: "
            "no thread sync, and a progress thread that holds a core for each rank.",
            exc,
        )
        yield
        return

    sync_none = getattr(bindings, "NIXL_THREAD_SYNC_NONE", None)
    sync_rw_mode = getattr(bindings, "NIXL_THREAD_SYNC_RW", None)
    if sync_rw and (sync_none is None or sync_rw_mode is None):
        logger.warning(
            "Neuron: this NIXL has no NIXL_THREAD_SYNC_RW, so the agent keeps NIXL's own "
            "thread sync."
        )
        sync_rw = False
        if delay_us <= 0:
            yield
            return

    built: list[str] = []
    synced: list[str] = []

    with _NIXL_AGENT_SWAP_LOCK:
        # Capture the original inside the lock and close over it. Reading it outside would let
        # a second connector record this block's wrapper as its own original.
        original_agent_cls = bindings.nixlAgent

        def agent_with_overrides(agent_name, agent_config):
            if delay_us > 0:
                agent_config.pthrDelay = delay_us
            if sync_rw and agent_config.syncMode == sync_none:
                agent_config.syncMode = sync_rw_mode
                synced.append(agent_name)
            built.append(agent_name)
            return original_agent_cls(agent_name, agent_config)

        bindings.nixlAgent = agent_with_overrides
        try:
            yield
        finally:
            bindings.nixlAgent = original_agent_cls

    if not built:
        # Silence would hide the overrides becoming a no-op if upstream stops building the
        # agent inside __init__.
        logger.warning(
            "Neuron: no NIXL agent was built while the agent config overrides were set, so "
            "they did NOT take effect."
        )
        return
    if synced:
        logger.info("Neuron: NIXL thread sync set to RW for %d agent(s).", len(synced))
    if delay_us > 0:
        logger.info(
            "Neuron: NIXL progress-thread delay set to %d us for %d agent(s).",
            delay_us,
            len(built),
        )


def dcp_seq_ratio_fragments(
    local_groups: list[list[int]],
    remote_groups: list[list[int]],
    group_decode_block_sizes: list[int],
    group_prefill_block_sizes: list[int],
    prefill_dcp: int,
    local_dcp: int,
    local_dcp_rank: int,
    num_tokens: int,
    num_computed_tokens: int = 0,
) -> dict[int, list[tuple[int, int, int, int, int, int]]]:
    """Heterogeneous-block-size (prefill_bs != decode_bs) DCP-prefill read map,
    for dense / full-KV-retention models (no sliding window).

    When the decode per-rank block_size ``dbs`` is a multiple ``R = dbs // pbs`` of
    the prefill per-rank block_size ``pbs`` (prefill DCP > decode DCP, matched
    effective blocks), a single decode block is NOT a whole prefill block: its
    ``dbs`` tokens are assembled from ``R`` token-fragments of ``pbs`` tokens each,
    scattered across the ``G = prefill_dcp // local_dcp`` prefill ranks this decode
    rank reads from.

    decode rank ``dr`` owns global decode-vblocks ``{v : v % local_dcp == dr}`` in
    ascending order; its (possibly APC-trimmed) local block ``L`` is the
    ``(dec_skip_vb + L)``-th such, i.e. global decode-vblock
    ``vbd = (dec_skip_vb + L) * local_dcp + dr``, covering global tokens
    ``[vbd*dbs, vbd*dbs+dbs)``. ``dec_skip_vb = (num_computed_tokens // dbs) //
    local_dcp`` counts the cached decode-vblocks this rank owns on an APC prefix
    hit (0 on cold / full-miss warm); it offsets the trimmed local list back onto
    its true global decode-vblocks. Without it a partial-hit warm read maps the
    tail blocks onto PREFIX logical blocks, corrupting the leading decode tokens.
    APC hashes on a ``dbs * local_dcp``-token granularity, so num_computed_tokens
    is always a multiple of it and dec_skip_vb is exact + identical across ranks.

    For each token ``t`` in a decode block: prefill rank = ``(t // pbs) % P``,
    prefill logical block = ``t // (pbs * P)``, prefill offset 0 (whole block),
    decode offset = ``t % dbs``. Tokens sharing (prefill rank, logical) are a
    contiguous ``pbs``-run, so each fragment is a whole prefill block copied into
    a ``pbs``-wide slice of the decode block.

    ``num_tokens`` (the prompt length) bounds the read so trailing fragments of a
    partial last decode block -- which cover tokens the prefill never stored -- are
    skipped.

    Returns ``frags_by_rank``: ``{prefill_rank: [(gi, local_block_id, local_tok_off,
    remote_block_id, remote_tok_off, n_tok), ...]}`` with ``n_tok == pbs`` and
    ``remote_tok_off == 0``; ``gi`` is the KV-cache group index (the reader keys
    per-group byte geometry off it).

    Example:
        prefill_dcp=8, local_dcp=4, local_dcp_rank=0, dbs=8, pbs=2 (R=4). A cold
        read of one decode block (physical id 100) whose 8 tokens are the four
        pbs=2 fragments scattered across prefill ranks 0..3, all in prefill
        logical block 0:

        >>> dcp_seq_ratio_fragments(
        ...     local_groups=[[100]],
        ...     remote_groups=[[0]],
        ...     group_decode_block_sizes=[8],
        ...     group_prefill_block_sizes=[2],
        ...     prefill_dcp=8,
        ...     local_dcp=4,
        ...     local_dcp_rank=0,
        ...     num_tokens=8,
        ... )
        {0: [(0, 100, 0, 0, 0, 2)],
         1: [(0, 100, 2, 0, 0, 2)],
         2: [(0, 100, 4, 0, 0, 2)],
         3: [(0, 100, 6, 0, 0, 2)]}
    """
    P = prefill_dcp
    frags_by_rank: dict = {}
    n_g = len(local_groups)
    for gi in range(n_g):
        lgroup = local_groups[gi]
        rgroup = remote_groups[gi]
        nloc = len(lgroup)
        nrem = len(rgroup)
        dbs = group_decode_block_sizes[gi]
        pbs = group_prefill_block_sizes[gi]
        R = dbs // pbs
        pblk = pbs * P
        # APC cached-prefix offset: skip the leading decode-vblocks this rank owns
        # that were already cached (the base scheduler trimmed them off the local
        # list). 0 on cold / full-miss.
        dec_skip_vb = (num_computed_tokens // dbs) // local_dcp
        for L in range(nloc):
            vbd = (dec_skip_vb + L) * local_dcp + local_dcp_rank
            for r in range(R):  # R fragments per decode block
                gtok = vbd * dbs + r * pbs  # global token at fragment start
                if gtok >= num_tokens:
                    # Fragment lies past the prompt end (trailing pbs-fragments of
                    # the last partial decode block cover never-stored tokens).
                    continue
                logical = gtok // pblk  # remote-list position
                pr = (gtok // pbs) % P  # prefill rank owning the token
                if logical < 0 or logical >= nrem:
                    continue  # out of advertised range
                frags_by_rank.setdefault(pr, []).append(
                    (gi, lgroup[L], r * pbs, rgroup[logical], 0, pbs)
                )
    return frags_by_rank


@dataclass
class NeuronNixlAgentMetadata(NixlAgentMetadata):
    """Pass-through of the upstream handshake metadata.

    0.29 carries every field Neuron needs natively, so this adds nothing;
    it keeps call sites on a Neuron-owned name in case fields return later
    (a wire-visible field also needs the base-module rebind back).
    """


def _agent_metadata_kwargs(**kwargs) -> dict:
    """Return the subset of ``kwargs`` the installed ``NeuronNixlAgentMetadata`` declares.
    A field a release removed raises ``TypeError`` in the dataclass constructor.
    """
    declared = getattr(NeuronNixlAgentMetadata, "__struct_fields__", None)
    if declared is None:
        declared = getattr(NeuronNixlAgentMetadata, "__dataclass_fields__", None)
    if not declared:
        return dict(kwargs)
    known = set(declared)
    dropped = sorted(k for k in kwargs if k not in known)
    if dropped:
        logger.info(
            "[GK-NIXL] this vLLM's agent metadata does not declare %s; not sending it",
            ", ".join(dropped),
        )
    return {k: v for k, v in kwargs.items() if k in known}


def _supported_kwargs(fn, **kwargs) -> dict:
    """Return the subset of ``kwargs`` that ``fn`` accepts.
    Passing an argument a release removed raises ``TypeError`` deep in the handshake or heartbeat loop.
    """
    accepted = inspect.signature(fn).parameters
    # A ``**kwargs`` catch-all takes every name, so drop nothing.
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        return dict(kwargs)
    dropped = sorted(k for k in kwargs if k not in accepted)
    if dropped:
        logger.info(
            "[GK-NIXL] %s does not take %s; not passing it",
            getattr(fn, "__name__", fn),
            ", ".join(dropped),
        )
    return {k: v for k, v in kwargs.items() if k in accepted}


class NeuronNixlConnectorWorker(NixlConnectorWorker):
    """NIXL connector worker with DCP-aware KV transfer.

    Uses unified head_ratio / seq_ratio logic to handle all topologies:
    - head_ratio > 1: split (remote block has more heads, read a subset)
    - head_ratio < 1: merge (remote block has fewer heads, read multiple)
    - seq_ratio > 1: read from multiple remote DCP ranks
    - seq_ratio <= 1: read from subset of one remote DCP rank's blocks
    """

    def __init__(self, *args, **kwargs):
        # NIXL builds the agent, and starts its progress thread, inside this call.
        with _nixl_agent_config_overrides():
            super().__init__(*args, **kwargs)
        self._local_dcp_size: int = 1
        self._local_dcp_rank: int = 0
        self._remote_dcp: dict[str, int] = {}
        self._head_ratio: dict[str, float] = {}
        # Remote (prefill) per-rank page size in tokens. The read path derives the
        # page ratio R = decode_bs // prefill_bs from this to route whole-block vs
        # token-fragment reads when prefill DCP > decode DCP.
        self._remote_prefill_bs: dict[str, int] = {}
        self._remote_tp_size: dict[str, int] = {}
        self._inverse_handle_cache: dict[str, list] = {}
        # Decode-side cache {remote_engine_id: {producer_rank R: fan_in(R)}}, where
        # fan_in(R) = the number of decode ranks that read from (and therefore
        # notify) producer rank R. A pure function of the P/D topology, so it is
        # computed once per remote engine and is identical on every decode rank;
        # passed to the base read as expected_consumers, which puts it on the wire
        # so the producer frees on the true per-rank reader count.
        self._producer_fanin: dict[str, dict[int, int]] = {}
        # Per-engine remote geometry for our DCP paths. add_remote_agent fully
        # replaces the base (no super()), so we own these dicts and their
        # cleanup (see _cleanup_remote_engine).
        self._tp_size: dict[str, int] = {}
        self._block_size: dict[str, int] = {}

    # ── Override: register_kv_caches ─────────────────────────────────────

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

            assert cache.dim() == 5 and cache.shape[0] == 2, (
                f"Expected (2, num_blocks, ...) Neuron KV cache for "
                f"{layer_name}, got shape {tuple(cache.shape)}"
            )
            assert cache.shape[1] == self.num_blocks, (
                f"KV cache num_blocks mismatch for {layer_name}: "
                f"expected {self.num_blocks}, got shape {tuple(cache.shape)}"
            )
            # One K (or V) slab per block; page_size_bytes covers K+V.
            half_page = layer_spec.page_size_bytes // 2

            for slab in (cache[0], cache[1]):
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
            "Registered %d K/V slab regions (%d layers x 2) with %d descs",
            self.num_regions,
            self.num_regions // 2,
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

    def _validate_remote_agent_handshake(self, nixl_agent_meta, remote_tp_size):
        """Skip upstream strict validation for all DCP topologies as the upstream doesn't consider DCP configs."""
        assert self.transfer_topo is not None
        # Upstream's handshake rejects prefill TP > decode TP (fractional
        # tp_ratio). This worker replaces that validation to admit DCP
        # topologies (where prefill TP legitimately exceeds decode TP, e.g.
        # tp32/dcp4 -> tp8), so the plain-TP downshift must be re-rejected
        # here: without DCP the extra prefill ranks have no reader and the
        # deployment half-works before dying mid-serve.
        remote_dcp = getattr(nixl_agent_meta, "dcp_size", 0) or 1
        if (
            remote_dcp == 1
            and self._local_dcp_size == 1
            and remote_tp_size > self.world_size
        ):
            raise RuntimeError(
                f"Unsupported DI topology: prefill TP ({remote_tp_size}) > "
                f"decode TP ({self.world_size}) without decode-context "
                "parallelism. Use prefill TP <= decode TP, or shard the "
                "prefill KV with DCP."
            )
        # both should have num_kv_cache_groups entres
        assert len(nixl_agent_meta.kv_caches_base_addr) == len(
            self.block_len_per_layer
        ), "KV group count mismatch between P and D"
        remote_block_len = nixl_agent_meta.block_lens[0]
        local_block_len = self.block_len_per_layer[0]
        assert all(bl == remote_block_len for bl in nixl_agent_meta.block_lens), (
            f"Remote block_lens not uniform: {nixl_agent_meta.block_lens}"
        )
        assert all(bl == local_block_len for bl in self.block_len_per_layer), (
            f"Local block_lens not uniform: {self.block_len_per_layer}"
        )
        ratio = remote_block_len / local_block_len
        inv_ratio = local_block_len / remote_block_len
        assert ratio == int(ratio) or inv_ratio == int(inv_ratio), (
            f"Block lens must be integer-divisible: "
            f"remote_block_len={remote_block_len}, local_block_len={local_block_len}"
        )

    # ── Override: add_remote_agent ────────────────────────────────────────

    def add_remote_agent(
        self,
        nixl_agent_meta,
        remote_tp_rank=0,
        remote_tp_size=1,
        remote_dcp_size=None,
    ):
        """Register remote agent with head_ratio-aware descriptors."""
        # This override fully replaces the base's add_remote_agent, so the
        # base's call to _validate_remote_agent_handshake never runs on this
        # path. Invoke the hook here, before any per-engine state is written,
        # so a rejected handshake (e.g. plain-TP downshift) fails fast and
        # leaves no partial registration behind.
        self._validate_remote_agent_handshake(nixl_agent_meta, remote_tp_size)
        eid = nixl_agent_meta.engine_id
        # ``dcp_size`` is not a field of every metadata type that reaches this. 1 is the right
        # default, because no DCP means one decode-context-parallel rank.
        remote_dcp = remote_dcp_size or getattr(nixl_agent_meta, "dcp_size", 0) or 1
        remote_block_len = nixl_agent_meta.block_lens[0]
        local_block_len = self.block_len_per_layer[0]

        # Decompose the raw block-length ratio into the head/dtype factor and the
        # page-size factor. block_len = block_size * n_kv_heads * d_head * dtype, so
        # a pure page-size mismatch (decode_bs = R * prefill_bs, e.g. prefill DCP >
        # decode DCP with matched effective blocks) would otherwise masquerade as a
        # head split/merge. Normalize each side to its per-TOKEN length so
        # head_ratio reflects ONLY the head/dtype layout; the page ratio
        # R = decode_bs // prefill_bs is handled in the read path.
        remote_bs = getattr(nixl_agent_meta, "block_size", 0) or 1
        local_bs = self.block_size or 1
        remote_per_token = max(1, remote_block_len // remote_bs)
        local_per_token = max(1, local_block_len // local_bs)
        head_ratio = remote_per_token / local_per_token

        if eid not in self._remote_dcp:
            self._remote_dcp[eid] = remote_dcp
            self._head_ratio[eid] = head_ratio
            self._remote_prefill_bs[eid] = remote_bs
            self._remote_tp_size[eid] = remote_tp_size
            # Seed an inert tp_mappings[eid] (we never read it) so the base's
            # unguarded `del tp_mappings[eid]` on TTL eviction doesn't KeyError —
            # our add_remote_agent replaces the base and never populated it.
            self.tp_mappings[eid] = TPMapping(
                source_ranks_per_group=(),
                all_source_ranks=(),
                rank_to_attention_slot={},
                rank_offset_factor=0,
            )
            logger.info(
                "DCP NIXL: engine %s head_ratio=%.2f, remote_dcp=%d, "
                "remote_block_len=%d, local_block_len=%d, "
                "P_TP=%d, D_TP=%d, D_DCP=%d",
                eid,
                head_ratio,
                remote_dcp,
                remote_block_len,
                local_block_len,
                remote_tp_size,
                self.world_size,
                self._local_dcp_size,
            )

        # For split/merge/match we handle descriptor registration ourselves.
        # Match case (head_ratio=1) also goes through our path to avoid
        # upstream's _group_spec_types / tp_mapping logic which breaks for DCP.
        if eid not in self.dst_num_blocks:
            self.dst_num_blocks[eid] = (
                nixl_agent_meta.num_blocks
            )  # total capacity (all slots in the KV cache), not specific to a certain request

        self.kv_caches_base_addr[eid][remote_tp_rank] = (
            nixl_agent_meta.kv_caches_base_addr
        )

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )
        # Don't write self._remote_agents here. The base publishes the full
        # per-rank dict atomically (from our return value) once the background
        # handshake finishes; an incremental write flips its completion guard
        # early and reads start mid-handshake -> KeyError on unregistered ranks.

        # Register with TransferTopology so _read_blocks can resolve get_engine_info()
        assert self.transfer_topo is not None
        if eid not in self.transfer_topo._engines:
            from vllm.distributed.kv_transfer.kv_connector.utils import (
                EngineTransferInfo,
            )

            self.transfer_topo.register_remote_engine(
                eid,
                EngineTransferInfo(
                    remote_tp_size=remote_tp_size,
                    remote_block_len=nixl_agent_meta.block_lens[0],
                    remote_block_size=nixl_agent_meta.block_size,
                    remote_physical_blocks_per_logical=1,
                ),
            )

        if head_ratio > 1:
            # Split: remote block has more heads than local.
            num_kv_heads = self.vllm_config.model_config.get_total_num_kv_heads()
            D_ranks_per_dcp_group = self.world_size // self._local_dcp_size
            D_num_kv_replica = max(1, D_ranks_per_dcp_group // num_kv_heads)
            D_kv_head_rank = self.tp_rank // (self._local_dcp_size * D_num_kv_replica)
            head_idx = D_kv_head_rank % int(head_ratio)
            head_offset = head_idx * local_block_len
            self._register_remote_descs(
                eid,
                nixl_agent_meta,
                remote_agent_name,
                remote_tp_rank,
                desc_len=local_block_len,
                offset=head_offset,
            )
        else:
            # Merge: register full remote block.
            self._register_remote_descs(
                eid,
                nixl_agent_meta,
                remote_agent_name,
                remote_tp_rank,
                desc_len=remote_block_len,
                offset=0,
            )

        return remote_agent_name

    def _register_remote_descs(
        self,
        eid,
        nixl_agent_meta,
        remote_agent_name,
        remote_tp_rank,
        desc_len,
        offset,
    ):
        """Register NIXL descriptors for a remote agent's blocks."""
        kv_addrs = nixl_agent_meta.kv_caches_base_addr
        blocks_data = []
        for kv_group_idx, base_addr in enumerate(kv_addrs):
            page_size = nixl_agent_meta.block_lens[kv_group_idx]
            for block_id in range(nixl_agent_meta.num_blocks):
                addr = base_addr + block_id * page_size + offset
                blocks_data.append((addr, desc_len, nixl_agent_meta.device_id))
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        handle = self.nixl_wrapper.prep_xfer_dlist(remote_agent_name, descs)
        self.dst_xfer_side_handles[eid][remote_tp_rank] = handle

    # ── Override: _cleanup_remote_engine ──────────────────────────────────

    def _cleanup_remote_engine(self, engine_id, *args, **kwargs):
        """Tear down per-engine state on TTL eviction / shutdown.

        Since we own add_remote_agent, we own the matching cleanup: release our
        own NIXL dlist handles (_inverse_handle_cache — the base doesn't know
        about them, so they'd leak), delegate the base-owned state to super(),
        then clear our extra geometry dicts so a re-registered engine starts
        clean. Handle release goes first in a try/finally so it always happens
        even if super() raises.
        """
        try:
            for handle in self._inverse_handle_cache.get(engine_id, []):
                self.nixl_wrapper.release_dlist_handle(handle)
        finally:
            super()._cleanup_remote_engine(engine_id, *args, **kwargs)
            self._inverse_handle_cache.pop(engine_id, None)
            self._producer_fanin.pop(engine_id, None)
            self._remote_dcp.pop(engine_id, None)
            self._head_ratio.pop(engine_id, None)
            self._remote_prefill_bs.pop(engine_id, None)
            self._remote_tp_size.pop(engine_id, None)
            self._tp_size.pop(engine_id, None)
            self._block_size.pop(engine_id, None)

    # ── Override: _nixl_handshake ─────────────────────────────────────────

    def _nixl_handshake(self, *args, **kwargs):
        """Connect to all remote ranks unconditionally, since the remote DCP degree is unknown here.
        Binds ``*args`` to the BASE signature. A fixed list gives a positional the wrong
        name when vLLM drops one.
        """
        base = super()._nixl_handshake
        bound = inspect.signature(base).bind(*args, **kwargs)
        bound.apply_defaults()

        # ``handshake_target_ranks`` takes the remote TP size on every version seen so far.
        self.transfer_topo.handshake_target_ranks = lambda rtp, rdcp=1: list(range(rtp))
        try:
            result = base(*bound.args, **bound.kwargs)
        finally:
            if "handshake_target_ranks" in self.transfer_topo.__dict__:
                del self.transfer_topo.__dict__["handshake_target_ranks"]
        return result

    # ── Override: _send_heartbeats ────────────────────────────────────────

    def _send_heartbeats(self, metadata: NixlConnectorMetadata) -> None:
        """Send heartbeats, tolerating concurrent remote-agent registration.

        Upstream iterates ``self._remote_agents[engine_id].values()`` directly.
        Handshakes run on a background ``_handshake_initiation_executor`` thread
        whose done-callback assigns ``self._remote_agents[eid] = ...``, so a
        handshake completing mid-iteration raises ``RuntimeError: dictionary
        changed size during iteration``, which kills the decode EngineCore and
        surfaces to the client as a 500. This is most likely with the wide
        remote-TP producers used in DCP DI (e.g. tp32_dcp4). Snapshotting the
        agent list before iterating closes that race; a heartbeat that misses a
        just-registered agent is picked up on the next cycle.
        """
        for engine_id, hb_info in metadata.heartbeat_by_engine.items():
            # Proactive handshake so the next heartbeat for this remote can go
            # through; skip sending while it is still pending.
            if (
                self._ensure_handshake(
                    engine_id,
                    hb_info.host,
                    hb_info.port,
                    hb_info.tp_size,
                    # ``dcp_size`` is gone from HeartbeatInfo and from the base signature in vLLM
                    # 0.28. 1 is the right default, because no DCP means one such rank.
                    **_supported_kwargs(
                        self._ensure_handshake,
                        dcp_size=getattr(hb_info, "dcp_size", 0) or 1,
                        pp_size=hb_info.pp_size,
                        notif_agents_only=(
                            self._hb_handshake_notif_only and hb_info.pp_size > 1
                        ),
                    ),
                )
                is not None
            ):
                continue

            hb_msg = ("HB:" + ",".join(hb_info.req_ids)).encode()
            for agent_name in list(self._remote_agents[engine_id].values()):
                try:
                    self.nixl_wrapper.send_notif(agent_name, notif_msg=hb_msg)
                except Exception:
                    logger.debug(
                        "Failed to send heartbeat to engine %s",
                        engine_id,
                        exc_info=True,
                    )

    # ── Override: _read_blocks_for_req ────────────────────────────────────

    def _read_blocks_for_req(self, req_id, meta):
        """Unified DCP-aware block reading using head topology and seq_ratio."""
        assert meta.remote is not None and self.transfer_topo is not None
        eid = meta.remote.engine_id
        # Per-producer-rank fan-in this decode rank hands the base read as
        # expected_consumers, so the producer frees the KV once all real readers
        # finish (see _producer_fanin_map). fanin_map[p_rank_id] is always
        # present: p_rank_id is a rank THIS decode rank reads from, hence in the
        # enumerated union by construction.
        fanin_map = self._producer_fanin_map(eid)

        P_DCP = self._remote_dcp.get(eid, 1)
        D_DCP = self._local_dcp_size
        head_ratio = self._head_ratio.get(eid, 1.0)

        P_TP = self._remote_tp_size[eid]
        seq_ratio = P_DCP / D_DCP

        # Heterogeneous page size (prefill DCP > decode DCP): decode block_size =
        # R * prefill block_size, so one decode block is assembled from R prefill-
        # block fragments scattered across DCP ranks, the whole-block map below
        # cannot express it. Route to the fragment reader. head_ratio is already
        # normalized to the pure head/dtype factor (see add_remote_agent), so R is
        # the isolated page ratio here.
        prefill_bs = self._remote_prefill_bs.get(eid) or self.block_size
        page_R = self.block_size // prefill_bs if prefill_bs else 1
        if seq_ratio > 1 and page_R > 1:
            if head_ratio != 1:
                raise NotImplementedError(
                    "het-page DCP read (page_R>1) with head split/merge "
                    f"(head_ratio={head_ratio}) is not supported"
                )
            self._read_blocks_dcp_prefill_hetbs(
                req_id, meta, eid, P_DCP, D_DCP, prefill_bs, page_R, fanin_map
            )
            return

        D_rank = self.tp_rank
        D_DCP_rank = self._local_dcp_rank

        num_kv_heads = self.vllm_config.model_config.get_total_num_kv_heads()
        D_ranks_per_dcp_group = self.world_size // D_DCP
        P_ranks_per_dcp_group = P_TP // P_DCP
        D_num_kv_replica = max(1, D_ranks_per_dcp_group // num_kv_heads)
        P_num_kv_replica = max(1, P_ranks_per_dcp_group // num_kv_heads)
        D_kv_head_rank = D_rank // (D_DCP * D_num_kv_replica)

        # --- Determine which prefill KV head ranks to read from ---
        if head_ratio >= 1:
            P_kv_head_ranks_to_read = [D_kv_head_rank // int(head_ratio)]
            split_local_handles = None
            remote_block_size = self.transfer_topo.get_engine_info(
                eid
            ).remote_block_size
            local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                remote_block_size
            ]
        else:
            inverse_head_ratio = int(1 / head_ratio)
            P_kv_head_ranks_to_read = [
                D_kv_head_rank * inverse_head_ratio + j
                for j in range(inverse_head_ratio)
            ]
            split_local_handles = self._get_or_create_inverse_handles(
                eid, inverse_head_ratio
            )

        # --- Determine which prefill DCP ranks and block mappings ---
        # Anchor num_groups on the LOCAL side (matching upstream): a DI pull can
        # arrive with zero local blocks (prefix-cache hit), and anchoring on the
        # remote count would then IndexError the empty local.
        num_groups = len(meta.local_physical_block_ids)
        if num_groups == 0:
            # Nothing to write, but still notify the prefill rank(s) this decode
            # rank maps to so the producer releases the KV (same p_rank_id as the
            # real path). Empty _read_blocks issues the notif without recording a
            # transfer; recording one would trip the scheduler's req-ownership
            # assert for a request this engine no longer owns (DP>1).
            if seq_ratio > 1:
                p_dcp_ranks = [D_DCP_rank + i * D_DCP for i in range(int(seq_ratio))]
            else:
                p_dcp_ranks = [D_DCP_rank % P_DCP]
            for p_dcp_rank in p_dcp_ranks:
                for p_kv_head_rank in P_kv_head_ranks_to_read:
                    p_rank_id = p_kv_head_rank * (P_DCP * P_num_kv_replica) + p_dcp_rank
                    self._read_blocks(
                        read_spec=ReadSpec(
                            remote_rank=p_rank_id,
                            local_block_ids=[],
                            remote_block_ids=[],
                        ),
                        dst_engine_id=eid,
                        request_id=req_id,
                        remote_request_id=meta.remote.request_id,
                        local_xfer_side_handle=(
                            split_local_handles[0]
                            if split_local_handles is not None
                            else local_xfer_side_handle
                        ),
                        remote_xfer_side_handle=self.dst_xfer_side_handles[eid][
                            p_rank_id
                        ],
                        # ``expected_consumers`` tells the producer how many decode ranks read from
                        # it. Pass it only where the installed vLLM accepts it.
                        **_supported_kwargs(
                            self._read_blocks,
                            expected_consumers=fanin_map[p_rank_id],
                        ),
                    )
            return

        if seq_ratio > 1:
            # DCP>1 isn't used by GPT-OSS today (always DCP=1, the branch below),
            # so this path still front-aligns and has the same prefix-cache-hit bug
            # we fix there. Doing it right here means handling the round-robin block
            # interleave across DCP ranks; leaving it until we have a config to test.
            int_seq_ratio = int(seq_ratio)
            P_DCP_ranks_to_read = [D_DCP_rank + i * D_DCP for i in range(int_seq_ratio)]
            remote_blk_id_indices_per_dcp_rank = []
            local_blk_id_indices_per_dcp_rank = []
            for k in range(int_seq_ratio):
                remote_indices_per_group = []
                local_indices_per_group = []
                for g in range(num_groups):
                    num_remote_blks = len(meta.remote.block_ids[g])
                    num_local_blks = len(meta.local_physical_block_ids[g])
                    local_indices = [
                        k + i * int_seq_ratio
                        for i in range(num_remote_blks)
                        if k + i * int_seq_ratio < num_local_blks
                    ]
                    remote_indices_per_group.append(list(range(len(local_indices))))
                    local_indices_per_group.append(local_indices)
                remote_blk_id_indices_per_dcp_rank.append(remote_indices_per_group)
                local_blk_id_indices_per_dcp_rank.append(local_indices_per_group)
        else:
            P_DCP_ranks_to_read = [D_DCP_rank % P_DCP]
            step = D_DCP // P_DCP
            offset = D_DCP_rank // P_DCP
            remote_indices_per_group = []
            local_indices_per_group = []
            for g in range(num_groups):
                num_local_blks = len(meta.local_physical_block_ids[g])
                num_remote_blks = len(meta.remote.block_ids[g])
                # This rank's slice of the remote blocks ([0..N-1] when DCP=1).
                strided = list(range(offset, num_remote_blks, step))
                # On a partial prefix-cache hit the decode side already has the
                # leading blocks, so only the tail is left to pull -- read the last
                # num_local blocks, matching upstream _apply_prefix_caching
                # (remote[-num_local:]). Reading from the front would copy the
                # cached prefix into the tail slots and decode the wrong tokens.
                # A full pull is unchanged.
                if num_local_blks <= 0:
                    remote_indices = []
                elif num_local_blks < len(strided):
                    remote_indices = strided[-num_local_blks:]
                else:
                    remote_indices = strided
                remote_indices_per_group.append(remote_indices)
                local_indices_per_group.append(list(range(len(remote_indices))))
            remote_blk_id_indices_per_dcp_rank = [remote_indices_per_group]
            local_blk_id_indices_per_dcp_rank = [local_indices_per_group]

        def get_p_rank_id(p_dcp_rank, p_kv_head_rank):
            # Prefill now uses the SAME full-TP + token-interleaved-KV layout
            # as decode: each KV-head group is a block of consecutive ranks,
            # with dcp_rank the inner (fastest-varying) dimension:
            #   rank = kv_head_rank * (P_DCP * P_num_kv_replica)
            #          + replica_idx * P_DCP + dcp_rank
            # Pick replica 0 (replicas hold identical KV). For P_DCP == 1 this
            # reduces to the standard-TP mapping kv_head_rank * P_num_kv_replica.
            return p_kv_head_rank * (P_DCP * P_num_kv_replica) + p_dcp_rank

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Unified read: req=%s, D_rank=%d, D_DCP_rank=%d/%d, "
                "D_kv_head_rank=%d, seq_ratio=%.2f, "
                "P_kv_head_ranks=%s, P_DCP_ranks=%s",
                req_id,
                D_rank,
                D_DCP_rank,
                D_DCP,
                D_kv_head_rank,
                seq_ratio,
                P_kv_head_ranks_to_read,
                P_DCP_ranks_to_read,
            )

        # --- Issue reads ---
        for i, p_dcp_rank in enumerate(P_DCP_ranks_to_read):
            remote_indices_per_group = remote_blk_id_indices_per_dcp_rank[i]
            local_indices_per_group = local_blk_id_indices_per_dcp_rank[i]

            if not any(local_indices_per_group):
                continue

            remote_block_ids = [
                [meta.remote.block_ids[g][idx] for idx in remote_indices_per_group[g]]
                for g in range(num_groups)
            ]
            local_block_ids = [
                [
                    meta.local_physical_block_ids[g][idx]
                    for idx in local_indices_per_group[g]
                ]
                for g in range(num_groups)
            ]
            # This override does its own prefix-cache trimming above, so the base
            # _read_blocks skipping _apply_prefix_caching when DCP is active is
            # expected. Nothing downstream re-checks the per-group pairing, so
            # assert it here: a mismatch would post unequal descriptor counts.
            assert all(
                len(local_block_ids[g]) == len(remote_block_ids[g])
                for g in range(num_groups)
            ), (
                f"per-group block count mismatch for req={req_id} p_dcp_rank="
                f"{p_dcp_rank}: local={[len(b) for b in local_block_ids]} "
                f"remote={[len(b) for b in remote_block_ids]}"
            )

            for j, p_kv_head_rank in enumerate(P_kv_head_ranks_to_read):
                p_rank_id = get_p_rank_id(p_dcp_rank, p_kv_head_rank)
                remote_xfer_side_handle = self.dst_xfer_side_handles[eid][p_rank_id]

                if split_local_handles is not None:
                    cur_local_handle = split_local_handles[j]
                else:
                    cur_local_handle = local_xfer_side_handle

                self._read_blocks(
                    read_spec=ReadSpec(
                        remote_rank=p_rank_id,
                        local_block_ids=local_block_ids,
                        remote_block_ids=remote_block_ids,
                    ),
                    dst_engine_id=eid,
                    request_id=req_id,
                    remote_request_id=meta.remote.request_id,
                    local_xfer_side_handle=cur_local_handle,
                    remote_xfer_side_handle=remote_xfer_side_handle,
                    # ``expected_consumers`` tells the producer how many decode ranks read from
                    # it. Pass it only where the installed vLLM accepts it.
                    **_supported_kwargs(
                        self._read_blocks,
                        expected_consumers=fanin_map[p_rank_id],
                    ),
                )

        # When D_DCP > P_DCP a short prompt can leave some decode DCP ranks with
        # no blocks to read (empty stride above), so they post no transfer and
        # create no _recving_transfers key. get_finished then never reports the
        # req from those ranks, and since completion needs all TP ranks, the req
        # hangs until TTL eviction. Register an empty entry (reported done
        # immediately) for reqs this engine owns. The _recving_metadata gate +
        # the num_groups==0 early-return above keep this off the disowned-abort
        # path, so it can't reintroduce the DP>1 assert this used to trip.
        if req_id in self._recving_metadata and req_id not in self._recving_transfers:
            self._recving_transfers[req_id] = []

    def _read_blocks_dcp_prefill_hetbs(
        self, req_id, meta, eid, P_DCP, D_DCP, prefill_bs, page_R, fanin_map
    ):
        """DCP-prefill read when decode block_size = page_R * prefill block_size
        (prefill DCP > decode DCP). A decode block holds ``page_R`` token-fragments
        of ``prefill_bs`` tokens each, scattered across the prefill DCP ranks; each
        fragment is a WHOLE prefill block copied into the r-th ``prefill_bs``-token
        slice of the decode block.

        Reuses the head-merge local-split machinery: ``_get_or_create_inverse_handles
        (eid, page_R)`` yields ``page_R`` local descriptors, handle ``r`` targeting
        decode-block byte offset ``r * (local_block_len // page_R)``, exactly the
        r-th fragment slice. head_ratio == 1 on this route (the fragment gate), so
        add_remote_agent registered whole prefill blocks (the merge/match branch),
        which match the fragment length.

        Requires matched effective blocks (``page_R == seq_ratio``, i.e.
        ``d_bs * D_DCP == p_bs * P_DCP``): then each decode rank reads the contiguous
        prefill-rank block ``{dr*R .. dr*R+R-1}`` and every prefill rank has fan-in 1.
        Non-matched effective blocks are not yet supported.
        """
        seq_ratio = P_DCP // D_DCP
        if page_R != seq_ratio:
            raise NotImplementedError(
                "DCP het-page read requires matched effective blocks "
                f"(page_R == seq_ratio); got page_R={page_R}, seq_ratio={seq_ratio}. "
                "Set decode/prefill block_size so d_bs*D_DCP == p_bs*P_DCP."
            )

        dr = self._local_dcp_rank
        groups = self.kv_cache_config.kv_cache_groups
        n_g = len(meta.local_physical_block_ids)
        decode_block_sizes = [
            groups[gi].kv_cache_spec.block_size if gi < len(groups) else self.block_size
            for gi in range(n_g)
        ]
        # Same page ratio across groups (both sides rescale block_size by the same
        # n_kv*d_head factor), so per-group prefill_bs = decode_bs // page_R.
        prefill_block_sizes = [bs // page_R for bs in decode_block_sizes]

        num_tokens = getattr(meta.remote, "num_tokens", 0)
        num_computed_tokens = getattr(meta.remote, "num_computed_tokens", 0)

        frags_by_rank = dcp_seq_ratio_fragments(
            local_groups=meta.local_physical_block_ids,
            remote_groups=meta.remote.block_ids,
            group_decode_block_sizes=decode_block_sizes,
            group_prefill_block_sizes=prefill_block_sizes,
            prefill_dcp=P_DCP,
            local_dcp=D_DCP,
            local_dcp_rank=dr,
            num_tokens=num_tokens,
            num_computed_tokens=num_computed_tokens,
        )

        # page_R local sub-descriptors: handle[r] -> decode-block offset r*prefill_bs.
        split_local_handles = self._get_or_create_inverse_handles(eid, page_R)

        # ``pr`` from the fragment map is a prefill DCP rank in [0, P_DCP). Map it to
        # the full NIXL rank id, adding this decode rank's KV-head-group offset
        # (head_ratio == 1 here, so decode kv-head-rank maps to the same prefill
        # kv-head-rank; replica 0).
        num_kv_heads = self.vllm_config.model_config.get_total_num_kv_heads()
        P_TP = self._remote_tp_size[eid]
        P_num_kv_replica = max(1, (P_TP // P_DCP) // num_kv_heads)
        D_num_kv_replica = max(1, (self.world_size // D_DCP) // num_kv_heads)
        D_kv_head_rank = self.tp_rank // (D_DCP * D_num_kv_replica)

        def p_rank_of(p_dcp):
            return D_kv_head_rank * (P_DCP * P_num_kv_replica) + p_dcp

        # Touch the recv-transfers key so a short prompt that produces no fragments
        # still registers.
        if req_id in self._recving_metadata and req_id not in self._recving_transfers:
            self._recving_transfers[req_id] = []

        # Group fragments by (prefill_rank, fragment slot r), then per group build
        # matched local/remote block-id lists and issue one read per (pr, r). The
        # local handle[r] carries the intra-decode-block byte offset; the remote
        # handle is the whole prefill block on rank pr.
        for pr, frags in frags_by_rank.items():
            p_rank_id = p_rank_of(pr)
            by_r: dict[int, list] = {}
            for gi, lblk, loff, rblk, _roff, _ntok in frags:
                r = loff // prefill_block_sizes[gi]
                per_group = by_r.setdefault(r, [([], []) for _ in range(n_g)])
                per_group[gi][0].append(lblk)
                per_group[gi][1].append(rblk)
            for r, per_group in by_r.items():
                local_block_ids = [lg for (lg, _rg) in per_group]
                remote_block_ids = [rg for (_lg, rg) in per_group]
                if not any(local_block_ids):
                    continue
                self._read_blocks(
                    read_spec=ReadSpec(
                        remote_rank=p_rank_id,
                        local_block_ids=local_block_ids,
                        remote_block_ids=remote_block_ids,
                    ),
                    dst_engine_id=eid,
                    request_id=req_id,
                    remote_request_id=meta.remote.request_id,
                    local_xfer_side_handle=split_local_handles[r],
                    remote_xfer_side_handle=self.dst_xfer_side_handles[eid][p_rank_id],
                    expected_consumers=fanin_map[p_rank_id],
                )

    def _get_or_create_inverse_handles(self, eid, inverse_head_ratio):
        """Create local split handles for merge (inverse_head_ratio > 1)."""
        if eid in self._inverse_handle_cache:
            return self._inverse_handle_cache[eid]

        remote_block_len = self.block_len_per_layer[0] // inverse_head_ratio
        local_base_addrs = self.kv_caches_base_addr[self.engine_id][self.tp_rank]
        split_handles = []
        for hp in range(inverse_head_ratio):
            blocks_data = []
            for kv_group_idx, base_addr in enumerate(local_base_addrs):
                local_page_size = self.block_len_per_layer[kv_group_idx]
                for block_id in range(self.num_blocks):
                    addr = (
                        base_addr + block_id * local_page_size + hp * remote_block_len
                    )
                    blocks_data.append((addr, remote_block_len, self.device_id))
            descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
            handle = self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs)
            split_handles.append(handle)
        self._inverse_handle_cache[eid] = split_handles
        return split_handles

    # ── Override: producer-side early-free (DCP/replica-correct notif accounting) ──

    def _producer_fanin_map(self, eid: str) -> dict[int, int]:
        """``{producer_rank R: number of decode ranks that read from R}`` -- how
        many read-done notifs R must see before freeing a request's KV.

        Base derives this from ``tp_ratio`` (TP world sizes only), blind to DCP,
        KV-head replication, and head split/merge (and this connector monkeypatches
        it to 1 on the read path), so it frees too early (corruption) or never (30s
        lease) on those topologies. Instead we replay the exact producer-rank routing
        of ``_read_blocks_for_req`` per decode rank and tally each producer rank's
        notifs, so the fan-in can't drift from the reads issued. Pure topology ->
        computed once per engine, identical on every decode rank (which embeds
        ``fan_in(R)`` in its notif to R).
        """
        cached = self._producer_fanin.get(eid)
        if cached is not None:
            return cached

        P_TP = self._remote_tp_size[eid]
        P_DCP = self._remote_dcp.get(eid, 1)
        D_TP = self.world_size  # full decode TP (DCP is nested within TP)
        D_DCP = self._local_dcp_size
        head_ratio = self._head_ratio.get(eid, 1.0)
        seq_ratio = P_DCP / D_DCP
        # Page ratio: >1 means the fragment read (decode block = page_R prefill
        # blocks). It reads a different prefill-rank set than the whole-block map,
        # so the fan-in replay below must match it.
        prefill_bs = self._remote_prefill_bs.get(eid) or self.block_size
        page_R = self.block_size // prefill_bs if prefill_bs else 1
        num_kv_heads = self.vllm_config.model_config.get_total_num_kv_heads()
        D_num_kv_replica = max(1, (D_TP // D_DCP) // num_kv_heads)
        P_num_kv_replica = max(1, (P_TP // P_DCP) // num_kv_heads)

        fanin: dict[int, int] = {}
        for d in range(D_TP):
            D_kv_head_rank = d // (D_DCP * D_num_kv_replica)
            D_DCP_rank = d % D_DCP
            if head_ratio >= 1:
                p_kv_head_ranks = [D_kv_head_rank // int(head_ratio)]
            else:
                inv = int(1 / head_ratio)
                p_kv_head_ranks = [D_kv_head_rank * inv + j for j in range(inv)]
            if seq_ratio > 1 and page_R > 1:
                # Fragment read (matched effective blocks, page_R == seq_ratio):
                # decode rank dr reads the contiguous prefill-DCP-rank block
                # {dr*R .. dr*R+R-1}, so each prefill rank has exactly one reader.
                p_dcp_ranks = [D_DCP_rank * page_R + r for r in range(page_R)]
            elif seq_ratio > 1:
                p_dcp_ranks = [D_DCP_rank + i * D_DCP for i in range(int(seq_ratio))]
            else:
                p_dcp_ranks = [D_DCP_rank % P_DCP]
            for p_dcp in p_dcp_ranks:
                for p_kv in p_kv_head_ranks:
                    R = p_kv * (P_DCP * P_num_kv_replica) + p_dcp
                    fanin[R] = fanin.get(R, 0) + 1

        self._producer_fanin[eid] = fanin
        return fanin


class NeuronNixlConnectorScheduler(NixlConnectorScheduler):
    """Scheduler-side connector carrying the DI sliding-window fix.

    On a hybrid SWA model the decode node admits a DI request at the full prompt
    length ``N`` but recomputes the last token at ``N - 1``, so the sliding-window
    skip (anchored at ``N``) evicts the window's oldest block one step too early
    at the ``N % block_size == block_size - 1`` boundary; that block is never
    transferred and decode reads uninitialised K -> garbled / empty output.

    The fix makes both nodes operate at ``N - 1``: the prefill node
    (``do_remote_decode``) pops the last prompt token, the decode node
    (``do_remote_prefill``) lowers the admitted count by one (prompt kept, since
    the decoder recomputes ``N - 1``). Both then reserve the same block count and
    the eviction lines up. It lives here, not in the old ``allocate_slots``
    monkeypatch, so it touches no block-manager internals.

    If a future vLLM does the N-1 dance itself, both sides self-protect: the
    decode gate stops firing once the base returns ``N - 1``, and the prefill pop
    has a tripwire that raises if the base also mutates the prompt length.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        from vllm.v1.kv_cache_interface import SlidingWindowSpec

        # True iff this model has a sliding-window group (the DI over-eviction).
        self._has_swa = any(
            isinstance(g.kv_cache_spec, SlidingWindowSpec)
            for g in kv_cache_config.kv_cache_groups
        )
        # vLLM reserves block_id 0 as the null block and HMA marks SWA-skipped
        # blocks with it at the front of a group; we only see ids here, so keep
        # the named constant. Deferred import matches the SlidingWindowSpec above.
        from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

        self._null_block_id = NULL_BLOCK_ID
        # req_id -> locally-cached prefix length (num_new_local_computed_tokens),
        self._num_computed_by_req: dict[str, int] = {}
        # req_id -> full prompt length, so fragmented read can skip trailing
        # fragments past the prompt end.
        self._num_tokens_by_req: dict[str, int] = {}
        if self._has_swa:
            logger.info(
                "NeuronNixlConnectorScheduler: sliding-window DI truncation active "
                "(%d KV-cache group(s)).",
                len(kv_cache_config.kv_cache_groups),
            )

    def on_new_request(self, request: "Request") -> None:
        """Prefill node: drop the last prompt token at add time so it ships
        ``h(0 .. N-2)``, matching the decode side's N-1 admission.

        Must run here, not in get_num_new_matched_tokens: the base sizes the
        prefix-cache hit before calling that, so popping later could leave
        ``num_new_tokens == 0`` and trip its assert (a prefill crash at
        ``N % block_size == 1``). Idempotent via the ``_swa_di_truncated`` marker.
        """
        params = request.kv_transfer_params
        if (
            self._has_swa
            and params is not None
            and params.get("do_remote_decode")
            and not params.get("_swa_di_truncated")
            and request.num_prompt_tokens > 1
            and request.prompt_token_ids is not None
        ):
            request.prompt_token_ids.pop()
            request._all_token_ids.pop()
            request.num_prompt_tokens -= 1
            params["_swa_di_truncated"] = True
        # Preserve the base's heartbeat tracking (no-op for do_remote_decode reqs).
        super().on_new_request(request)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        params = request.kv_transfer_params
        self._num_computed_by_req[request.request_id] = num_computed_tokens
        self._num_tokens_by_req[request.request_id] = request.num_prompt_tokens

        # The prefill-side pop now happens in on_new_request (before the base's
        # prefix-cache hit is computed), so nothing to truncate here.
        prompt_len_before = request.num_prompt_tokens
        count, load_async = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )

        # Tripwire: we popped in on_new_request; if a future vLLM also truncates
        # this prompt, super() would take it to N-2. Fail loud (explicit raise so
        # it survives ``python -O``).
        if (
            self._has_swa
            and params is not None
            and params.get("do_remote_decode")
            and params.get("_swa_di_truncated")
            and request.num_prompt_tokens != prompt_len_before
        ):
            raise RuntimeError(
                "NeuronNixlConnectorScheduler: the base scheduler mutated the "
                f"prompt length ({prompt_len_before} -> {request.num_prompt_tokens})"
                " after this subclass already truncated it for the sliding-window "
                "DI fix. vLLM may now truncate these requests itself; the fix "
                "double-applies. Remove this subclass or re-validate."
            )

        # Decode node: admit N-1 (prompt intact) to retain the window-low block.
        # The ``>= num_prompt_tokens`` guard decrements only a full-prompt pull
        # (a partial prefix-cache pull is untouched), and is False if the base
        # already returns N-1, so we never stack to N-2.
        if (
            self._has_swa
            and params is not None
            and params.get("do_remote_prefill")
            and count > 0
            and num_computed_tokens + count >= request.num_prompt_tokens
        ):
            count -= 1
            if count == 0:
                # Nothing left to pull; load_async=True with count 0 would trip
                # the base's ``assert num_external_computed_tokens > 0``.
                load_async = False

        return count, load_async

    def build_connector_meta(self, scheduler_output):
        """Stamp the locally-cached prefix length onto each recv RemoteMeta.

        On a decode-side APC hit the base scheduler trims the recv local block
        list to the uncached TAIL, while the remote (prefill) block list stays
        the full logical table. The DCP fragment read (prefill DCP > decode DCP)
        needs ``num_computed_tokens`` to offset the trimmed list back onto its
        true global decode-vblocks (``dec_skip_vb``). The count is captured in
        ``get_num_new_matched_tokens`` and dropped at the base metadata boundary,
        so re-attach it here. ReqMeta / RemoteMeta are in-process plain
        dataclasses (bind_connector_metadata just assigns the object, no
        serialization), so adding the attribute is safe. Defaults to 0 when
        absent so the worker falls back to the full-read map (cold behavior).
        """
        meta = super().build_connector_meta(scheduler_output)
        for req_id, req_meta in meta.reqs_to_recv.items():
            if req_meta.remote is not None:
                req_meta.remote.num_computed_tokens = self._num_computed_by_req.get(
                    req_id, 0
                )
                req_meta.remote.num_tokens = self._num_tokens_by_req.get(req_id, 0)
        # Drop entries for reqs whose recv meta is now emitted; keep the dicts
        # from growing unbounded.
        for req_id in meta.reqs_to_recv:
            self._num_computed_by_req.pop(req_id, None)
            self._num_tokens_by_req.pop(req_id, None)
        return meta

    def _strip_leading_null_blocks(self, blocks: list[int]) -> list[int]:
        """Drop the leading run of null blocks from one SWA group's clipped list.

        HMA front-pads SWA groups with null blocks and the base keeps the last
        ``blocks_per_sw`` entries — exact only for a full window. Under N-1
        truncation the window is one block short at ``N % block_size == 1``, so the
        base slice keeps a leading null that the worker mispairs (dropping the
        newest real block). Only the front is ever null, so a full window is
        unchanged (strict no-op).
        """
        n_leading_null = 0
        for block_id in blocks:
            if block_id == self._null_block_id:
                n_leading_null += 1
            else:
                break
        return blocks[n_leading_null:] if n_leading_null else blocks

    def get_exchange_clipped_blocks(
        self, block_ids: BlockIds, clip_ssm: bool = True
    ) -> BlockIds:
        """Base SWA unpad, then strip any leading null it left behind (see
        ``_strip_leading_null_blocks``). No-op for full-attn groups and the
        untruncated path, which have no leading null.
        """
        clipped = super().get_exchange_clipped_blocks(block_ids, clip_ssm=clip_ssm)
        if not self._has_swa:
            return clipped
        return tuple(self._strip_leading_null_blocks(list(group)) for group in clipped)


class NeuronNixlConnector(NixlConnector):
    """Neuron NIXL connector with DCP support."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: "KVConnectorRole",
        kv_cache_config: "KVCacheConfig",
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
            self.connector_scheduler = NeuronNixlConnectorScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = NeuronNixlConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )
