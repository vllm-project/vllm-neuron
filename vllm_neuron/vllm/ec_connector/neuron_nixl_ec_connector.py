# SPDX-License-Identifier: Apache-2.0
"""
NeuronNixlECConnector — standalone NIXL encoder-cache (EC) connector for Neuron.

Moves a finished vision embedding from a Vision-Encoder (VE) pool to a
Prefill/Decode (PD) pool over NIXL (HBM->HBM), reusing the EC connector slot
(vllm.distributed.ec_transfer). a thin dispatcher delegating by role to a
scheduler-side helper and a worker-side transport.

Users select it via ec_transfer_config:
    --ec-transfer-config '{"ec_connector": "NeuronNixlECConnector",
                           "ec_role": "ec_producer" | "ec_consumer", ...}'
(platform.py auto-injects ec_connector_module_path for this name.)

Classes in this file:
- NeuronECWorkerMetadata      — ECConnectorWorkerMetadata subclass carrying the locator
                                worker->scheduler (survives the pickle transport).
- NeuronECConnectorMetadata   — scheduler->worker per-step load list (mm_hash -> locator).
- NeuronNixlECConnector       — dispatcher; subclasses ECConnectorBase, delegates by role.
- NeuronNixlECConnectorScheduler — scheduler-side helper (cache admission + per-step metadata).
- NeuronNixlECConnectorWorker — worker-side transport: NIXL register / handshake / READ /
                                completion barrier (wait_for_load).
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import msgspec

from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
    ECConnectorWorkerMetadata,
)
from vllm.logger import init_logger

from vllm_neuron.vllm.worker.encoder_cache_blocks import mm_hash_fingerprint

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Control-plane request sent by a PD consumer to a VE producer's metadata server
# to fetch its NIXL agent metadata (the one-time handshake).
GET_META_MSG = b"get_ec_meta"


class NeuronNixlECAgentMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    """The VE worker's NIXL agent info, served to a PD consumer over the handshake.


    - agent_metadata: opaque bytes from nixl_wrapper.get_agent_metadata().
    - base_addr / num_blocks / block_len: the embedding region's address + geometry,
      so the consumer can build remote descriptors at READ time.
    - model_id / dtype: producer identity, so the consumer can reject a pool serving
      a different model or dtype at handshake time (before any bytes move).
    - tag_base_addr / tag_block_len: the parallel per-block mm_hash tag region, so the
      consumer can pull the tag with the embedding and validate receipt (Part D).
    """

    engine_id: str
    agent_metadata: bytes
    base_addr: int
    num_blocks: int
    block_len: int
    device_id: int
    model_id: str
    dtype: str
    tag_base_addr: int
    tag_block_len: int


@dataclass
class NeuronECWorkerMetadata(ECConnectorWorkerMetadata):
    """Worker->scheduler carrier for encoder-cache locators.

    embed_locator: mm_hash -> {engine_id, host, port, remote_block_ids, num_tokens}.
    remote_block_ids index the producer's buffer (remote addressing only); the
    consumer builds the NIXL descriptor from them + the producer base_addr +
    page_stride, and supplies its own local landing positions at READ time.

    This metadata is aggregated across worker ranks (aggregate merges the
    locator maps) before reaching the scheduler-side connector.
    """

    embed_locator: dict[str, Any] = field(default_factory=dict)

    def aggregate(self, other: ECConnectorWorkerMetadata) -> "NeuronECWorkerMetadata":
        assert isinstance(other, NeuronECWorkerMetadata)
        self.embed_locator.update(other.embed_locator)
        return self


@dataclass
class NeuronECConnectorMetadata(ECConnectorMetadata):
    """Scheduler->worker per-step load list: mm_hash -> locator dict.

    Built by the scheduler-side helper in build_connector_meta and bound onto the
    worker connector before model execution (bind_connector_metadata). The worker
    reads it to know which encoder caches to NIXL-READ this step.
    """

    load: dict[str, Any] = field(default_factory=dict)


class NeuronNixlECConnectorScheduler:
    """Scheduler-side helper: tracks which encoder caches the consumer must load.

    Runs in the scheduler process. The loadable set is driven by the per-request
    locator carried from the VE, then packed into the per-step metadata for the
    worker. Mirrors upstream ECExampleConnector's scheduler-side shape, but keyed
    on the locator rather than a shared on-disk store.

    The locator rides on the request's KV-transfer params (under "ec_locator")
    because the pinned vllm has no dedicated EC-transfer param; borrowing that
    field is not the same as using the KV connector. TODO(epd-transport): switch
    to the dedicated ec_transfer_params field once vllm #40695 lands.
    """

    def __init__(self, vllm_config: "VllmConfig"):
        self.vllm_config = vllm_config
        self.ec_config = vllm_config.ec_transfer_config
        # mm_hash -> locator dict, awaiting a worker-side load this step.
        self._need_load: dict[str, dict] = {}

    def has_cache_item(self, identifier: str) -> bool:
        # A consumer (PD) has no vision tower, so it can only pull -> always True.
        # A producer (VE) encodes items itself and is the source others pull from,
        # so it never pulls -> always False.
        # The actual READ is driven by the per-item locator in update_state_after_alloc.
        # TODO(epd-transport): vllm #40695 changes this to
        # has_cache_item(identifier, request), letting us admit only items that
        # carry a valid locator and reject the rest here at admission, instead of
        # admitting all and surfacing a missing locator later.
        return bool(self.ec_config.is_ec_consumer)

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        # Only the consumer (PD) loads an external encoder cache; the producer
        # (VE) has nothing to fetch.
        if not self.ec_config.is_ec_consumer:
            return
        mm_hash = request.mm_features[index].identifier
        locator = self._locator_from_request(request, mm_hash)
        if locator is not None:
            self._need_load[mm_hash] = dict(locator)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> NeuronECConnectorMetadata:
        # Drain the accumulated load list into this step's metadata and reset, so
        # a load is requested exactly once.
        meta = NeuronECConnectorMetadata(load=dict(self._need_load))
        self._need_load.clear()
        return meta

    def request_finished(
        self, request: "Request"
    ) -> tuple[bool, dict[str, Any] | None]:
        # The locator is emitted onto the response by NeuronScheduler
        # (core/scheduler.py), not here. Nothing to hold asynchronously, so
        # return the no-delay-free default (no async save in flight).
        return False, None

    @staticmethod
    def _locator_from_request(request: "Request", mm_hash: str) -> dict | None:
        # Locator carrier is temporary -- see the class docstring.
        params = getattr(request, "kv_transfer_params", None) or {}
        ec_locator = params.get("ec_locator")
        if not ec_locator:
            return None
        return ec_locator.get(mm_hash)


class NeuronNixlECConnectorWorker:
    """Worker-side EC transport: register one region, hand-shake, READ it HBM->HBM.

    Does NOT inherit NixlConnectorWorker -- its __init__ needs KV-cache geometry an
    embedding region can't supply. We compose the raw nixl_wrapper and apply its
    transport patterns to ONE contiguous region (no per-layer loop).
    """

    def __init__(self, vllm_config: "VllmConfig"):
        self.vllm_config = vllm_config
        self.ec_config = vllm_config.ec_transfer_config
        self.engine_id = self.ec_config.engine_id

        # NIXL agent + memory settings, resolved lazily (see _ensure_nixl) so the
        # connector can be constructed on a CPU box without the nixl package.
        self.nixl_wrapper: Any = None
        self.nixl_memory_type: str | None = None
        # Neuron uses the LIBFABRIC backend (EFA), not UCX -- UCX cannot register
        # the Neuron device pointer (ucp_mem_map fails). Matches the DI KV launch
        # config; overridable via ec_connector_extra_config["backends"].
        self.nixl_backends = self.ec_config.get_from_extra_config(
            "backends", ["LIBFABRIC"]
        )
        self.device_id: int = 0
        self._transfer_timeout_s: float = float(
            self.ec_config.get_from_extra_config("transfer_timeout_sec", 1)
        )
        self._poll_interval_s: float = float(
            self.ec_config.get_from_extra_config("wait_poll_interval_sec", 0.0005)
        )

        # Local region state, set by register_encoder_cache.
        self._registered_descs: list[Any] = []
        self._local_xfer_handle: int | None = None
        self._agent_metadata: NeuronNixlECAgentMetadata | None = None
        self._block_len: int = 0
        # Local identity, compared against the remote at handshake time.
        self._model_id: str = self.vllm_config.model_config.model
        self._dtype: str = ""
        # Parallel per-block mm_hash tag region. The tag has its own xfer
        # handle and READs AFTER the embedding lands, so the consumer reads the
        # embedding before the tag. With the producer writing the tag before the
        # embedding, a matching tag then proves the embedding matches too.
        # self._local_tags is the local tag tensor, read to validate receipt.
        # It is None when there is no tag region (tag_block_len stays 0), which
        # is the single "tags disabled" signal.
        self._local_tag_xfer_handle: int | None = None
        self._local_tags: Any = None
        self._tag_block_len: int = 0
        # Tag READs deferred until the embedding lands:
        # mm_hash -> (remote_tag_handle, local_block_ids, remote_block_ids).
        self._pending_tag_reads: dict[str, tuple[int, list[int], list[int]]] = {}
        # Per in-flight READ: mm_hash -> (local_block_ids, expected_fingerprint).
        # The tag compare runs after the tag transfer lands.
        self._pending_validation: dict[str, tuple[list[int], int]] = {}

        # Producer handshake-server address (set by start_handshake_server), put on
        # the locator so a consumer can open the one-time metadata handshake. Port
        # is per-rank (base_port + tp_rank) to avoid collisions on one host.
        self._side_channel_host: str = self.ec_config.get_from_extra_config(
            "side_channel_host", "127.0.0.1"
        )
        self._side_channel_port: int | None = None
        self._handshake_stop = threading.Event()

        # Remote agents discovered via handshake, cached per engine_id (one
        # handshake per producer engine,
        # reused for every later request). {engine_id: (agent_name, xfer_handle)}.
        self._remote_agents: dict[str, tuple[str, int]] = {}
        # In-flight READs: mm_hash -> list[nixl xfer handle]. Keyed by mm_hash.
        self._recving_transfers: dict[str, list[int]] = defaultdict(list)

    def _ensure_nixl(self) -> None:
        """Build the NIXL agent on first use (import deferred -- needs hardware)."""
        if self.nixl_wrapper is not None:
            return
        from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config

        config = (
            nixl_agent_config(backends=self.nixl_backends)
            if nixl_agent_config is not None
            else None
        )
        self.nixl_wrapper = NixlWrapper(str(uuid.uuid4()), config)
        from vllm.platforms import current_platform

        self.nixl_memory_type = current_platform.get_nixl_memory_type() or "VRAM"

    def register_encoder_cache(self, transfer_buffer, tag_buffer=None) -> None:
        """Register the encoder-cache buffer as ONE contiguous NIXL region.

        transfer_buffer is the device tensor backing the cache (the VE's
        EncoderCacheBlocks buffer, or the PD landing buffer): shape
        [num_blocks, block_size, fat_dim]. We register the whole tensor as a
        single region; per-block descriptors are derived at READ time from the
        block stride. After this, the VE has agent metadata to serve over the
        handshake.

        tag_buffer is the parallel per-block mm_hash tag tensor (int64
        [num_blocks]); it is registered as a second region and pulled alongside
        the embedding so the consumer can validate receipt (Part D).
        """
        self._ensure_nixl()
        base_addr = transfer_buffer.data_ptr()
        num_blocks = transfer_buffer.shape[0]
        region_len = transfer_buffer.numel() * transfer_buffer.element_size()
        # Bytes per block row -- the unit a single embedding block transfers.
        self._block_len = region_len // num_blocks
        self.device_id = max(transfer_buffer.get_device(), 0)
        self._dtype = str(transfer_buffer.dtype)

        descs = self.nixl_wrapper.get_reg_descs(
            [(base_addr, region_len, self.device_id, "")], self.nixl_memory_type
        )
        self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
        self._registered_descs.append(descs)

        # Register the parallel mm_hash tag region, if it exists. The consumer
        # pulls the tag with the embedding for the Part D receipt check.
        self._local_tags = tag_buffer
        tag_base_addr = 0
        if tag_buffer is not None:
            tag_base_addr = tag_buffer.data_ptr()
            tag_region_len = tag_buffer.numel() * tag_buffer.element_size()
            self._tag_block_len = tag_region_len // num_blocks
            tag_descs = self.nixl_wrapper.get_reg_descs(
                [(tag_base_addr, tag_region_len, self.device_id, "")],
                self.nixl_memory_type,
            )
            self.nixl_wrapper.register_memory(tag_descs, backends=self.nixl_backends)
            self._registered_descs.append(tag_descs)

        # Prep the LOCAL embedding xfer handle: one descriptor per block row,
        # indexed by block id. A READ targets local positions by these ids.
        local_blocks = [
            (base_addr + i * self._block_len, self._block_len, self.device_id)
            for i in range(num_blocks)
        ]
        local_descs = self.nixl_wrapper.get_xfer_descs(
            local_blocks, self.nixl_memory_type
        )
        self._local_xfer_handle = self.nixl_wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT", local_descs
        )

        # Prep a separate LOCAL tag xfer handle, one 8-byte descriptor per block.
        # The tag READs on its own transfer, after the embedding lands.
        if self._local_tags is not None:
            local_tag_blocks = [
                (
                    tag_base_addr + i * self._tag_block_len,
                    self._tag_block_len,
                    self.device_id,
                )
                for i in range(num_blocks)
            ]
            local_tag_descs = self.nixl_wrapper.get_xfer_descs(
                local_tag_blocks, self.nixl_memory_type
            )
            self._local_tag_xfer_handle = self.nixl_wrapper.prep_xfer_dlist(
                "NIXL_INIT_AGENT", local_tag_descs
            )

        self._agent_metadata = NeuronNixlECAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            base_addr=base_addr,
            num_blocks=num_blocks,
            block_len=self._block_len,
            device_id=self.device_id,
            model_id=self._model_id,
            dtype=self._dtype,
            tag_base_addr=tag_base_addr,
            tag_block_len=self._tag_block_len,
        )

    def register_and_serve(
        self, transfer_buffer, tag_buffer=None, tp_rank: int = 0
    ) -> None:
        """Register the cache buffer and (producer) start the handshake server.

        Called once at worker init, after the encoder-cache device buffer exists.
        Both roles register the same on-device EncoderCacheBlocks buffer with NIXL
        (the VE serves it as the READ source; the PD reads into it), plus the
        parallel mm_hash tag region. The producer also starts its ZMQ metadata
        server and records its host/port (per TP rank: base_port + tp_rank) so the
        locator can carry the address. Calling again is a no-op (returns early if
        already registered).
        """
        if self._local_xfer_handle is not None:
            return  # already registered
        self.register_encoder_cache(transfer_buffer, tag_buffer)
        if self.ec_config.is_ec_producer:
            base_port = int(
                self.ec_config.get_from_extra_config("side_channel_port", 0) or 0
            )
            port = base_port + tp_rank if base_port else 0
            ready = threading.Event()
            self._side_channel_port = self._start_handshake_server(
                host=self._side_channel_host,
                port=port,
                ready=ready,
                stop=self._handshake_stop,
            )

    # ------------------------------------------------------------------
    # Handshake -- the one-time control-plane bootstrap (per producer engine).
    #
    # Producer (VE) hosts a ZMQ ROUTER serving its agent metadata; consumer (PD)
    # fetches it once per engine_id over a ZMQ REQ, registers the remote agent
    # with NIXL, and caches it. Later requests to the same engine skip this --
    # there is no per-request control hop. This is the worker-side placement the
    # upstream EC connector uses (the EC scheduler-helper is thin and the
    # producer's address rides the locator).
    # ------------------------------------------------------------------

    def _start_handshake_server(self, host, port, ready, stop):
        """Start the ROUTER metadata server in a daemon thread; return its port.

        Replies to each GET_META_MSG with the msgpack-encoded agent metadata for
        this VE's single region. port=0 binds an ephemeral port (returned).
        """
        import zmq

        assert self._agent_metadata is not None, "register_encoder_cache first"
        encoded = msgspec.msgpack.encode(self._agent_metadata)
        bound_port = port

        def _serve():
            nonlocal bound_port
            ctx = zmq.Context()
            sock = ctx.socket(zmq.ROUTER)
            try:
                if port == 0:
                    bound_port = sock.bind_to_random_port(f"tcp://{host}")
                else:
                    sock.bind(f"tcp://{host}:{port}")
                sock.setsockopt(zmq.RCVTIMEO, 1000)
                ready.set()
                while not stop.is_set():
                    try:
                        identity, _, _msg = sock.recv_multipart()
                    except zmq.Again:
                        continue
                    sock.send_multipart((identity, b"", encoded))
            finally:
                ctx.destroy(linger=0)

        self._handshake_server_thread = threading.Thread(
            target=_serve, daemon=True, name="ec-nixl-handshake-server"
        )
        self._handshake_server_thread.start()
        if not ready.wait(timeout=5.0):
            raise RuntimeError(
                f"EC handshake server did not become ready within 5s "
                f"(host={host}, port={port}); cannot serve agent metadata."
            )
        return bound_port

    def _fetch_remote_metadata(self, host, port) -> bytes:
        """REQ the producer's ROUTER for its agent metadata (raw msgpack bytes)."""
        import zmq

        ctx = zmq.Context()
        try:
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, 5000)
            sock.connect(f"tcp://{host}:{port}")
            sock.send(GET_META_MSG)
            return sock.recv()
        finally:
            ctx.destroy(linger=0)

    def _ensure_handshake(self, engine_id, host, port) -> tuple[str, int, int]:
        """Handshake to `engine_id` once; cache (agent_name, remote_handle, remote_tag_handle).

        Done once per producer engine: a later request to the same engine finds
        the cached entry and skips the handshake. Returns the cached tuple.
        """
        cached = self._remote_agents.get(engine_id)
        if cached is not None:
            return cached
        raw = self._fetch_remote_metadata(host, port)
        meta = msgspec.msgpack.Decoder(NeuronNixlECAgentMetadata).decode(raw)
        # Safety: VE and PD must agree on the per-block byte stride (block_size x
        # fat_dim x itemsize). A mismatch (divergent bucket/merge-factor/dtype
        # between pools) would silently READ wrong-sized rows. Reject it loudly at
        # handshake time rather than corrupt the embedding at runtime.
        assert self._block_len, (
            "register_encoder_cache must run before _ensure_handshake "
            "(local EC region not registered; _block_len is 0)"
        )
        if meta.block_len != self._block_len:
            raise ValueError(
                f"EC region block_len mismatch: remote engine {engine_id} "
                f"advertises {meta.block_len} bytes/block but local region is "
                f"{self._block_len}. VE and PD must share block_size x fat_dim x "
                f"itemsize (check bucket/merge-factor/dtype agreement)."
            )
        # Identity: block_len equality cannot distinguish two different models (or
        # dtypes) that happen to share byte geometry. Reject incorrectly wired pool here,
        # before any embedding bytes move.
        if meta.model_id != self._model_id:
            raise ValueError(
                f"EC model_id mismatch: remote engine {engine_id} serves model "
                f"{meta.model_id!r} but local pool expects {self._model_id!r}. "
                f"The VE and PD pools must serve the same model."
            )
        if meta.dtype != self._dtype:
            raise ValueError(
                f"EC dtype mismatch: remote engine {engine_id} serves dtype "
                f"{meta.dtype!r} but local region is {self._dtype!r}."
            )
        # A tag-enabled consumer needs a tag-enabled producer. Reject the mismatch
        # at handshake time. Both pools must run the same encoder-cache build.
        if self._local_tags is not None and not meta.tag_block_len:
            raise ValueError(
                f"EC tag region missing: remote engine {engine_id} advertises no "
                f"mm_hash tag region, but the local pool expects one. Run the same "
                f"encoder-cache build on both pools."
            )
        agent_name = self.nixl_wrapper.add_remote_agent(meta.agent_metadata)
        # Prep the remote embedding xfer handle: one descriptor per remote block.
        blocks_data = [
            (meta.base_addr + i * meta.block_len, meta.block_len, meta.device_id)
            for i in range(meta.num_blocks)
        ]
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        remote_handle = self.nixl_wrapper.prep_xfer_dlist(agent_name, descs)
        # Prep a separate remote tag xfer handle for the deferred tag READ.
        remote_tag_handle = -1
        if self._local_tags is not None:
            tag_blocks = [
                (
                    meta.tag_base_addr + i * meta.tag_block_len,
                    meta.tag_block_len,
                    meta.device_id,
                )
                for i in range(meta.num_blocks)
            ]
            tag_descs = self.nixl_wrapper.get_xfer_descs(
                tag_blocks, self.nixl_memory_type
            )
            remote_tag_handle = self.nixl_wrapper.prep_xfer_dlist(agent_name, tag_descs)
        self._remote_agents[engine_id] = (agent_name, remote_handle, remote_tag_handle)
        return self._remote_agents[engine_id]

    def start_load_caches(self, encoder_cache, metadata=None, **kwargs) -> None:
        """Issue the embedding NIXL READ for each item in the load list.

        Each item's locator gives the producer (engine_id + host/port) and the
        remote block ids (remote_block_ids); the consumer lifecycle supplies the
        PD-local landing positions (local_block_ids). The tag READ does not run
        here. wait_for_load issues it after the embedding lands, so the consumer
        reads the embedding before the tag.
        """
        if metadata is None:
            return
        for mm_hash, locator in metadata.load.items():
            engine_id = locator["engine_id"]
            _agent_name, remote_handle, remote_tag_handle = self._ensure_handshake(
                engine_id, locator["host"], locator["port"]
            )
            remote_ids = list(locator["remote_block_ids"])
            local_ids = list(locator["local_block_ids"])
            handle = self.nixl_wrapper.make_prepped_xfer(
                "READ",
                self._local_xfer_handle,
                local_ids,
                remote_handle,
                remote_ids,
                mm_hash.encode(),
            )
            self.nixl_wrapper.transfer(handle)
            self._recving_transfers[mm_hash].append(handle)
            if self._local_tags is not None:
                # Defer the tag READ until the embedding lands (see wait_for_load).
                self._pending_tag_reads[mm_hash] = (
                    remote_tag_handle,
                    local_ids,
                    remote_ids,
                )
                self._pending_validation[mm_hash] = (
                    local_ids,
                    mm_hash_fingerprint(mm_hash),
                )

    def save_caches(self, encoder_cache, mm_hash, **kwargs) -> None:
        # Nothing to persist here: the embedding is already in the registered
        # region; the consumer pulls it via READ.
        #
        # KNOWN LIMITATION (prototype): the VE has no transfer-completion pin, so
        # there is no guarantee a block is fully READ by the consumer before the
        # VE evicts/reuses it. We rely on the cache's wall-clock min_hold_time as
        # a best-effort window. The prod fix (pin the block until consumption,
        # driven by a READ-completion signal) is planned for next milestone.
        return

    def _poll_recving_once(self) -> set[str]:
        """Advance every in-flight READ one step; return mm_hashes that failed.

        Polls each handle: DONE -> released (success); PROC -> kept; any other
        state -> released + marked failed. An mm_hash resolves only once ALL its
        handles leave PROC; a failed one is returned, a successful one is simply
        dropped from the in-flight map.
        """
        failed_mm_hashes: set[str] = set()
        for mm_hash, handles in list(self._recving_transfers.items()):
            in_progress = []
            failed = False
            for handle in handles:
                state = self.nixl_wrapper.check_xfer_state(handle)
                if state == "DONE":
                    self.nixl_wrapper.release_xfer_handle(handle)
                elif state == "PROC":
                    in_progress.append(handle)
                else:
                    logger.error(
                        "EC NIXL transfer failed for mm_hash %s (state=%s)",
                        mm_hash,
                        state,
                    )
                    self.nixl_wrapper.release_xfer_handle(handle)
                    failed = True
            if in_progress:
                self._recving_transfers[mm_hash] = in_progress
                continue
            del self._recving_transfers[mm_hash]
            if failed:
                failed_mm_hashes.add(mm_hash)
        return failed_mm_hashes

    def _drain_recving(self) -> set[str]:
        """Poll the in-flight transfers to completion; return the failed mm_hashes.

        Apply the transfer timeout as a backstop: a stuck transfer fails loud
        rather than hangs. Return as soon as no transfer remains in flight.
        """
        failed: set[str] = set()
        if not self._recving_transfers:
            return failed
        deadline = time.monotonic() + self._transfer_timeout_s
        while self._recving_transfers:
            failed |= self._poll_recving_once()
            if not self._recving_transfers:
                break
            if time.monotonic() >= deadline:
                # Backstop: a stuck transfer should fail loud, not hang forever.
                # Fail the still-outstanding mm_hashes and stop waiting.
                for mm_hash, handles in list(self._recving_transfers.items()):
                    for handle in handles:
                        self.nixl_wrapper.release_xfer_handle(handle)
                    del self._recving_transfers[mm_hash]
                    failed.add(mm_hash)
                logger.error(
                    "EC transfer timed out after %.1fs; failing mm_hashes %s",
                    self._transfer_timeout_s,
                    failed,
                )
                break
            time.sleep(self._poll_interval_s)
        return failed

    def wait_for_load(self) -> set[str]:
        """Block until this step's NIXL READs land; return the failed mm_hashes.

        The READ is async, so the prefill gather must wait here before it reads a
        landing block. A step with no READ (text-only / cache-hit / no locator)
        returns at once. Implements the upstream ECConnectorBase.wait_for_load
        (#40695).

        The tag READ runs in a second phase, after the embedding lands, so the
        consumer reads the embedding before the tag. With the producer writing the
        tag before the embedding, a matching tag then proves the embedding matches.
        """
        # Phase 1: the embedding READs land.
        failed = self._drain_recving()
        # Phase 2: issue the tag READ for each item whose embedding landed. The
        # embedding is now in HBM, so this read observes memory after the embedding
        # read did.
        for mm_hash, (
            remote_tag_handle,
            local_ids,
            remote_ids,
        ) in self._pending_tag_reads.items():
            if mm_hash in failed:
                continue
            tag_handle = self.nixl_wrapper.make_prepped_xfer(
                "READ",
                self._local_tag_xfer_handle,
                local_ids,
                remote_tag_handle,
                remote_ids,
                (mm_hash + ":tag").encode(),
            )
            self.nixl_wrapper.transfer(tag_handle)
            self._recving_transfers[mm_hash].append(tag_handle)
        self._pending_tag_reads = {}
        # Phase 3: the tag READs land.
        failed |= self._drain_recving()
        # Phase 4: compare each pulled tag to the expected fingerprint.
        failed |= self._validate_pulled_tags(failed)
        return failed

    def _validate_pulled_tags(self, already_failed: set[str]) -> set[str]:
        """Compare each successfully-pulled mm_hash tag to its expected fingerprint.

        A mismatch means the remote block we READ no longer held (or never held)
        this mm_hash's embedding -- a stale/reused/wrong-content block that
        block_len geometry and transfer-state checks cannot catch. Returns the
        mm_hashes whose tag did not match; drains the pending-validation map.
        """
        newly_failed: set[str] = set()
        pending = self._pending_validation
        self._pending_validation = {}
        if self._local_tags is None:
            return newly_failed
        to_check = {h: v for h, v in pending.items() if h not in already_failed}
        if not to_check:
            return newly_failed
        # Read every needed tag slot to host in a SINGLE device->host copy, then
        # compare on the host. The tag comparison must reach the host to gate the
        # raise, so one sync is unavoidable; batching keeps it to one per step
        # (not one per mm_hash), and it runs after wait_for_load already drained
        # the transfers, off the transfer-overlap path.
        needed_ids = sorted(
            {i for local_ids, _ in to_check.values() for i in local_ids}
        )
        tag_by_id = dict(zip(needed_ids, self._local_tags[needed_ids].tolist()))
        for mm_hash, (local_ids, expected_fp) in to_check.items():
            got = [tag_by_id[i] for i in local_ids]
            if any(tag != expected_fp for tag in got):
                logger.error(
                    "EC receipt validation failed for mm_hash %s: expected "
                    "fingerprint %d, got tags %s at local blocks %s (stale/reused "
                    "remote block or wrong content)",
                    mm_hash,
                    expected_fp,
                    got,
                    local_ids,
                )
                newly_failed.add(mm_hash)
        return newly_failed


class NeuronNixlECConnector(ECConnectorBase):
    """EC-native NIXL connector dispatcher. Thin; delegates each ABC method by role.

    Built per-pool by ECConnectorFactory.create_connector(config, role): the
    scheduler process gets the scheduler-side helper, each worker process gets the
    worker-side transport.
    """

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        super().__init__(vllm_config, role)
        self.engine_id = vllm_config.ec_transfer_config.engine_id
        if role == ECConnectorRole.SCHEDULER:
            self.connector_scheduler: NeuronNixlECConnectorScheduler | None = (
                NeuronNixlECConnectorScheduler(vllm_config)
            )
            self.connector_worker: NeuronNixlECConnectorWorker | None = None
        else:
            self.connector_scheduler = None
            self.connector_worker = NeuronNixlECConnectorWorker(vllm_config)

    # ==============================
    # Scheduler-side (called by the engine Scheduler)
    # ==============================

    def has_cache_item(self, identifier: str) -> bool:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.has_cache_item(identifier)

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_state_after_alloc(request, index)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> NeuronECConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self, request: "Request"
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request)

    # ==============================
    # Worker-side
    # ==============================

    def register_and_serve(
        self, transfer_buffer, tag_buffer=None, tp_rank: int = 0
    ) -> None:
        assert self.connector_worker is not None
        self.connector_worker.register_and_serve(
            transfer_buffer, tag_buffer, tp_rank=tp_rank
        )

    @property
    def side_channel_address(self) -> tuple[str | None, int | None]:
        """(host, port) of this worker's handshake server, for the producer locator."""
        if self.connector_worker is None:
            return None, None
        return (
            self.connector_worker._side_channel_host,
            self.connector_worker._side_channel_port,
        )

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        assert self.connector_worker is not None
        self.connector_worker.start_load_caches(encoder_cache, **kwargs)

    def save_caches(self, encoder_cache, mm_hash: str, **kwargs) -> None:
        assert self.connector_worker is not None
        self.connector_worker.save_caches(encoder_cache, mm_hash, **kwargs)

    def wait_for_load(self) -> set[str]:
        assert self.connector_worker is not None
        return self.connector_worker.wait_for_load()
