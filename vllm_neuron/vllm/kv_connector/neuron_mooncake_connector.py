# SPDX-License-Identifier: Apache-2.0
"""NeuronMooncakeConnector: vLLM's MooncakeConnector over Neuron's K/V-slab KV caches.

K/V slabs
---------
vLLM's MooncakeConnector registers each layer's cache as blocks-first: ``cache[0]`` is one
block, and when that block is not one dense span (a heads-first layout) it registers one
region per ``cache[:, i]`` instead, each blocks-first with ``block_len = stride(0)``.

vllm-neuron hands the connector ``(2, num_blocks, kv_heads, block_size, head_size)`` per layer
(``get_kv_cache_view_for_connector_registration``): the K and V halves are two num_blocks-long
slabs. Taken as is, ``cache[0]`` is the whole K slab, which is dense, so the base would register
the layer as one region whose "block" is the entire K slab.

Handing it ``cache.transpose(0, 1)`` instead, ``(num_blocks, 2, ...)``, makes ``cache[0]`` one
block's K and V, which are not one span, so the base takes its per-region path with
``cache[:, 0]`` = the K slab and ``cache[:, 1]`` = the V slab, each blocks-first with
``block_len`` = one block's K (or V) bytes. That is the same two-regions-per-layer registration
NeuronNixlConnector does by hand. Packed FP8 K is a view of the K half with the same bytes per
block, so it moves unchanged.

Prefill without DCP into a DCP decode
-------------------------------------
Set ``"remote_dcp_size": <decode DCP>`` in the PREFILL side's ``kv_connector_extra_config``
(default 1: homogeneous, the base behaviour). Supported: equal TP on both sides, prefill DCP 1,
decode DCP D with ``D <= TP / num_kv_heads`` (the KV-head replication), and the decode's
``cp_kv_cache_interleave_size`` equal to the block size.

With KV heads replicated R = TP / num_kv_heads times, prefill rank ``p = R * h + i`` holds the
whole KV of head ``h`` (replica i), and a DCP=D decode rank ``d = R * h + i`` holds head ``h``'s
DCP shard ``i % D`` (token-interleaved by block: its block ``k`` = tokens
``[(k * D + i % D) * B, + B)`` of the request, B = block size). The base already pairs decode
rank d with prefill rank d (equal TP), and both ranks see the same head, so only the block map
changes: the prefill rank pushes its blocks ``k * D + i % D`` into the decode rank's blocks
``k``. Every prefill rank sends 1/D of the blocks, so all 64 ranks (and their NICs) share the
transfer. A decode partial prefix hit sends only its uncached tail blocks, aligned on the last
block as in the base (``dcp_block_map``).

Registration
------------
Every buffer is registered as one 2 MiB-aligned span per ``/dev/neuron`` mapping, so EFA gets
2 MiB pages, and model tensors listed in ``CONNECTOR_EXTRA_KV_TENSORS`` move as extra regions of
their layer; both are described in ``neuron_hbm_registration``.

Mooncake itself recognises Neuron HBM (the /dev/neuron<N> mapping) and registers it with
FI_HMEM_NEURON over EFA (kvcache-ai/Mooncake#3888, mooncake-transfer-engine-non-cuda >= 0.3.14rc3);
use ``"mooncake_protocol": "efa"``. With mooncake < 0.3.14 also set
``MC_EFA_NIC_SELECTION=local`` (the default since kvcache-ai/Mooncake#5102), so each HBM buffer is
registered only on its topology-local NICs, the ones its transfers use.

Not supported: sliding window, mamba, prefill DCP > 1, unequal TP (the NeuronNixlConnector
scheduler-side fixes for SWA / DCP are not carried over).

    --kv-transfer-config '{"kv_connector":"NeuronMooncakeConnector",
        "kv_connector_module_path":"vllm_neuron.vllm.kv_connector.neuron_mooncake_connector",
        "kv_role":"kv_producer",
        "kv_connector_extra_config":{"mooncake_protocol":"efa","remote_dcp_size":8}}'
"""

from __future__ import annotations

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector as mc
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnector,
    MooncakeConnectorWorker,
)
from vllm.logger import init_logger

from vllm_neuron.vllm.kv_connector.neuron_hbm_registration import (
    coalesce_enabled,
    coalesce_neuron_spans,
    extra_kv_tensors,
)

logger = init_logger(__name__)


def _blocks_first(name: str, cache: torch.Tensor) -> torch.Tensor:
    if cache.dim() == 5 and cache.shape[0] == 2:
        return cache.transpose(0, 1)
    raise AssertionError(
        f"NeuronMooncakeConnector expects a (2, num_blocks, heads, block, head_size) KV cache "
        f"for {name}, got {tuple(cache.shape)}"
    )


class _DeferredRegistration:
    """Stands in for the TransferEngine while the caches are registered: records the buffers,
    which ``register_kv_caches`` then registers as coalesced spans."""

    def __init__(self, engine):
        self._engine = engine
        self.ptrs: list[int] = []
        self.lens: list[int] = []

    def batch_register_memory(self, ptrs, lens):
        self.ptrs += list(ptrs)
        self.lens += list(lens)
        return 0

    def __getattr__(self, name):
        return getattr(self._engine, name)


def dcp_block_map(
    local_block_ids: list[int], remote_block_ids: list[int], dcp_size: int, dcp_rank: int
) -> tuple[list[int], list[int]]:
    """Pair this prefill rank's blocks with a DCP decode rank's blocks.

    ``local_block_ids``: all of the request's prefill blocks, in token order (block j = tokens
    [j * B, (j + 1) * B)). ``remote_block_ids``: the decode's uncached tail blocks (block k =
    the decode's k-th, of ceil(len(local) / dcp_size); on this decode rank it holds tokens
    [(k * dcp_size + dcp_rank) * B, + B)). Returns (local ids, remote ids) to copy, pairwise;
    trailing decode blocks past the prompt on this rank are skipped.
    """
    n_local = len(local_block_ids)
    n_decode = -(-n_local // dcp_size)
    k0 = n_decode - len(remote_block_ids)
    if k0 < 0:
        raise ValueError(
            f"decode asks for {len(remote_block_ids)} blocks, the prompt has {n_decode} "
            f"DCP={dcp_size} blocks ({n_local} prefill blocks)"
        )
    src, dst = [], []
    for i, remote_id in enumerate(remote_block_ids):
        j = (k0 + i) * dcp_size + dcp_rank
        if j < n_local:
            src.append(local_block_ids[j])
            dst.append(remote_id)
    return src, dst


class NeuronMooncakeConnectorWorker(MooncakeConnectorWorker):
    """The base worker plus the prefill-side block map into a DCP decode (``remote_dcp_size``)."""

    def _setup_remote_dcp(self) -> None:
        cfg = self.vllm_config.kv_transfer_config
        self.remote_dcp_size = int(cfg.get_from_extra_config("remote_dcp_size", 1))
        if self.remote_dcp_size == 1:
            return
        par = self.vllm_config.parallel_config
        num_kv_heads = self.model_config.get_total_num_kv_heads()
        replicas = max(1, self.tp_size // num_kv_heads)
        interleave = getattr(par, "cp_kv_cache_interleave_size", self.block_size)
        problems = []
        if not self.is_kv_producer:
            problems.append("remote_dcp_size is a prefill-side (kv_producer) setting")
        if par.decode_context_parallel_size != 1:
            problems.append(f"prefill DCP must be 1, is {par.decode_context_parallel_size}")
        if replicas % self.remote_dcp_size:
            problems.append(
                f"decode DCP {self.remote_dcp_size} must divide the KV-head replication "
                f"{replicas} (TP {self.tp_size} / {num_kv_heads} KV heads)"
            )
        if interleave != self.block_size:
            problems.append(f"needs cp_kv_cache_interleave_size == block size {self.block_size}")
        if self.kv_cache_config.has_mamba_layers:
            problems.append("no mamba")
        if problems:
            raise ValueError("NeuronMooncakeConnector remote_dcp_size: " + "; ".join(problems))
        logger.info(
            "NeuronMooncakeConnector: prefill (DCP 1) -> decode DCP %d; prefill rank r pushes "
            "its blocks k * %d + r %% %d to decode rank r",
            self.remote_dcp_size, self.remote_dcp_size, self.remote_dcp_size,
        )

    async def _build_transfer_params(self, ready_reqs, agent_meta, local_regions, remote_regions):
        if getattr(self, "remote_dcp_size", 1) == 1:
            return await super()._build_transfer_params(
                ready_reqs, agent_meta, local_regions, remote_regions
            )
        if agent_meta.remote_tp_size != self.tp_size:
            msg = f"remote_dcp_size needs equal TP (P {self.tp_size}, D {agent_meta.remote_tp_size})"
            return [], [], [], [d for d, _ in ready_reqs], msg
        dcp_rank = agent_meta.remote_tp_rank % self.remote_dcp_size
        src_ptrs: list[int] = []
        dst_ptrs: list[int] = []
        lengths: list[int] = []
        err_reqs: list[str] = []
        err_msg = None
        for d_req_id, send_meta in ready_reqs:
            _, remote_groups = agent_meta.req_blocks[d_req_id]
            if not remote_groups or all(len(g) == 0 for g in remote_groups):
                continue
            if len(send_meta.local_block_ids) != len(remote_groups):
                err_reqs.append(d_req_id)
                err_msg = err_msg or "KV group count mismatch"
                continue
            try:
                pairs = [
                    dcp_block_map(list(lg), list(rg), self.remote_dcp_size, dcp_rank)
                    for lg, rg in zip(send_meta.local_block_ids, remote_groups)
                ]
            except ValueError as e:
                logger.error("req %s: %s", d_req_id, e)
                err_reqs.append(d_req_id)
                err_msg = err_msg or str(e)
                continue
            for local_region, remote_region in zip(local_regions, remote_regions):
                local_ids, remote_ids = pairs[local_region.group_index]
                if not local_ids:
                    continue
                # Equal TP: each block is copied whole (same per-block bytes on both sides).
                n = local_region.kv_block_len
                assert n == remote_region.kv_block_len, "per-block KV bytes differ"
                coalesce = local_region.block_len == remote_region.block_len == n
                for lrun, rrun in zip(*mc.group_concurrent_contiguous(local_ids, remote_ids)):
                    if coalesce:
                        src_ptrs.append(local_region.base_addr + lrun[0] * local_region.block_len)
                        dst_ptrs.append(remote_region.base_addr + rrun[0] * remote_region.block_len)
                        lengths.append(n * len(lrun))
                    else:
                        for lb, rb in zip(lrun, rrun):
                            src_ptrs.append(local_region.base_addr + lb * local_region.block_len)
                            dst_ptrs.append(remote_region.base_addr + rb * remote_region.block_len)
                            lengths.append(n)
        return src_ptrs, dst_ptrs, lengths, err_reqs, err_msg


    # A failed pull: the base worker only logs it, so the decode-side request stays in
    # WAITING_FOR_REMOTE_KVS forever (e.g. "Timeout waiting for P side ready" when the prefill
    # queue outlasts VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT, or an EAGAIN from the transfer). Report
    # it finished with its blocks as load errors instead, so kv_load_failure_policy applies
    # (recompute: the decode re-computes the prompt; fail: the request errors out).

    def _mark_pull_failed(self, req_id, pull_meta) -> None:
        errs = self.__dict__.setdefault("_load_error_blocks", set())
        for group in pull_meta.local_block_ids:
            errs.update(int(b) for b in group if b)
        pull_meta.pull_tasks_count -= 1
        if pull_meta.pull_tasks_count <= 0:
            self.finished_recving_reqs.add(pull_meta.d_req_id)
        logger.warning(
            "NeuronMooncakeConnector: KV pull for %s failed; %d blocks reported as load errors",
            req_id, sum(len(g) for g in pull_meta.local_block_ids),
        )

    def process_pulling_result(self, response, pull_metas):
        super().process_pulling_result(response, pull_metas)
        seen = self.__dict__.setdefault("_pull_seen", {}).setdefault(id(pull_metas), set())
        seen.update(response.ok_reqs or [])
        for req_id in response.err_reqs or []:
            if req_id in pull_metas and req_id not in seen:
                seen.add(req_id)
                self._mark_pull_failed(req_id, pull_metas[req_id])

    async def receive_kv_from_single_worker(self, worker_addr, pull_metas):
        try:
            await super().receive_kv_from_single_worker(worker_addr, pull_metas)
        finally:
            # An ERROR response or an exception returns without a result for these requests.
            seen = self.__dict__.setdefault("_pull_seen", {}).pop(id(pull_metas), set())
            for req_id, pull_meta in pull_metas.items():
                if req_id not in seen:
                    self._mark_pull_failed(req_id, pull_meta)

    def take_load_error_blocks(self) -> set[int]:
        errs = self.__dict__.get("_load_error_blocks")
        if not errs:
            return set()
        self._load_error_blocks = set()
        return errs


class NeuronMooncakeConnector(MooncakeConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        if self.connector_worker is not None:
            # The base builds its worker inside __init__; extend it in place (no extra state
            # beyond what _setup_remote_dcp adds).
            self.connector_worker.__class__ = NeuronMooncakeConnectorWorker
            self.connector_worker._setup_remote_dcp()

    def get_block_ids_with_load_errors(self) -> set[int]:
        w = self.connector_worker
        return w.take_load_error_blocks() if w is not None else set()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if not coalesce_enabled():
            return self._register_kv_caches(kv_caches)
        w = self.connector_worker
        real = w.engine
        deferred = _DeferredRegistration(real)
        w.engine = deferred
        try:
            self._register_kv_caches(kv_caches)
        finally:
            w.engine = real
        ptrs, lens = coalesce_neuron_spans(deferred.ptrs, deferred.lens)
        if real.batch_register_memory(ptrs, lens) != 0:
            raise RuntimeError("Mooncake registration of the coalesced KV spans failed")
        logger.info(
            "NeuronMooncakeConnector: %d buffers registered as %d coalesced spans (%s)",
            len(deferred.ptrs), len(ptrs), ", ".join(f"{p:#x}+{n >> 20}MiB" for p, n in zip(ptrs, lens)),
        )

    def _register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        views = {name: _blocks_first(name, cache) for name, cache in kv_caches.items()}
        first = next(iter(views.values()))
        logger.info(
            "NeuronMooncakeConnector: %d layers as K/V slab regions, block view %s, "
            "%d bytes per K (or V) block",
            len(views),
            tuple(first.shape),
            first[:, 0].stride(0) * first.element_size(),
        )
        super().register_kv_caches(views)
        self._register_extra_kv_tensors(views)

    def _register_extra_kv_tensors(self, views: dict[str, torch.Tensor]) -> None:
        w = self.connector_worker
        extra = extra_kv_tensors()
        names = [n for n in views if n in extra and n in w._layer_specs]
        if not names:
            return
        ptrs, lens = [], []
        for name in names:
            t = extra[name]
            assert t.shape[0] == views[name].shape[0], (
                f"{name}: extra tensor {tuple(t.shape)} is not indexed by the KV's "
                f"{views[name].shape[0]} blocks"
            )
            assert t.is_contiguous(), f"{name}: extra tensor must be contiguous"
            block_len = t.stride(0) * t.element_size()
            w.kv_caches_base_addr.append(t.data_ptr())
            w.block_len_per_layer.append(block_len)
            w.kv_block_len_per_layer.append(block_len)
            w.registered_layer_names.append(name)
            w.registered_layer_indices.append(mc.extract_layer_index(name))
            w.registered_group_indices.append(w._layer_group_indices[name])
            ptrs.append(t.data_ptr())
            lens.append(t.numel() * t.element_size())
        if w.engine.batch_register_memory(ptrs, lens) != 0:
            raise RuntimeError("Mooncake registration of the extra KV tensors failed")
        w.seen_base_addresses.extend(ptrs)
        logger.info(
            "NeuronMooncakeConnector: %d extra per-layer tensors as regions (%s, %d bytes per "
            "block), e.g. %s",
            len(names), tuple(extra[names[0]].shape), lens[0] // extra[names[0]].shape[0], names[0],
        )
