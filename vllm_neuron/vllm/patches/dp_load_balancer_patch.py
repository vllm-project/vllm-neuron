# SPDX-License-Identifier: Apache-2.0
"""Keep internal-DP routing admissions consistent across API workers.

With ``VLLM_NEURON_DP_PREFIX_AFFINITY=1``, a request also goes to the engine
that holds the longest prefix of its prompt (see dp_prefix_affinity.py).
"""

import contextlib
import logging
import multiprocessing
import time
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

_SHARED_INFLIGHT_ARG = "_vllm_neuron_dp_lb_shared_inflight"
_SHARED_INFLIGHT_CONFIG = "dp_lb_shared_inflight"
_ROUTING_ENABLED_ATTR = "_vllm_neuron_dp_lb_routing_enabled"
_AFFINITY_ATTR = "_vllm_neuron_dp_prefix_affinity"
_AFFINITY_LOG_INTERVAL_S = 60.0

_applied = False
_original_engine_core_run_busy_loop: Any = None
_original_dplb_client_init: Any = None
_original_dplb_get_core_engine_for_request: Any = None
_original_dplb_process_engine_outputs: Any = None
_original_api_server_manager_init: Any = None


def _maybe_publish_request_counts(self) -> None:
    if not self.publish_dp_lb_stats:
        return

    counts = self.scheduler.get_request_counts()
    if counts == getattr(self, "_dp_lb_last_counts", (0, 0)):
        return

    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.metrics.stats import SchedulerStats

    self._dp_lb_last_counts = counts
    stats = SchedulerStats(*counts)
    self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))


def _engine_core_run_busy_loop(self) -> None:
    if not self.publish_dp_lb_stats:
        return _original_engine_core_run_busy_loop(self)

    while self._handle_shutdown():
        self._process_input_queue()
        self._maybe_publish_request_counts()
        self._process_engine_step()
        self._maybe_publish_request_counts()

    raise SystemExit


def _dplb_client_init(
    self,
    vllm_config,
    executor_class,
    log_stats: bool,
    client_addresses: dict[str, Any] | None = None,
    client_count: int = 1,
    client_index: int = 0,
) -> None:
    routing_enabled = not vllm_config.model_config.is_moe
    setattr(self, _ROUTING_ENABLED_ATTR, routing_enabled)
    if not routing_enabled:
        _original_dplb_client_init(
            self,
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )
        return

    self.shared_engine_inflight = (
        client_addresses.get(_SHARED_INFLIGHT_CONFIG)
        if client_addresses is not None
        else None
    )
    self.engine_inflight = Counter()

    _original_dplb_client_init(
        self,
        vllm_config,
        executor_class,
        log_stats,
        client_addresses,
        client_count,
        client_index,
    )

    if self.shared_engine_inflight is not None:
        assert len(self.shared_engine_inflight) == len(self.core_engines)
    self.engine_indices = {
        engine: index for index, engine in enumerate(self.core_engines)
    }
    setattr(
        self,
        _AFFINITY_ATTR,
        _make_prefix_affinity(vllm_config, len(self.core_engines), client_count),
    )


def _make_prefix_affinity(vllm_config, num_engines: int, client_count: int):
    """Build the prefix index when VLLM_NEURON_DP_PREFIX_AFFINITY is set.

    Returns:
        A PrefixAffinityIndex, or None when the routing is off.
    """
    from vllm_neuron import envs
    from vllm_neuron.vllm.patches.dp_prefix_affinity import (
        CHUNK_TOKENS,
        DEFAULT_MAX_CHUNKS_PER_ENGINE,
        PrefixAffinityIndex,
    )

    if not envs.VLLM_NEURON_DP_PREFIX_AFFINITY:
        return None
    cache_config = vllm_config.cache_config
    if not getattr(cache_config, "enable_prefix_caching", False):
        logger.warning(
            "DP prefix affinity: off, because prefix caching is off "
            "(VLLM_NEURON_DP_PREFIX_AFFINITY is set)."
        )
        return None

    slack = envs.VLLM_NEURON_DP_PREFIX_AFFINITY_SLACK
    if slack is None:
        slack = max(1, vllm_config.scheduler_config.max_num_seqs // 2)
    # The handshake sums the KV blocks of the engines of this client.
    num_blocks = getattr(cache_config, "num_gpu_blocks", None) or 0
    block_size = getattr(cache_config, "block_size", None) or 0
    if num_blocks and block_size:
        engine_tokens = num_blocks * block_size // num_engines
        max_chunks = max(1, engine_tokens // CHUNK_TOKENS)
    else:
        max_chunks = DEFAULT_MAX_CHUNKS_PER_ENGINE
    if client_count > 1:
        logger.warning(
            "DP prefix affinity: each of the %d API servers keeps its own prefix "
            "index, so a session that moves between API servers can lose its "
            "engine. Serve with --api-server-count 1 for full affinity.",
            client_count,
        )
    logger.info(
        "DP prefix affinity: on for %d engines, slack %s requests, %d-token "
        "chunks, %d chunks per engine",
        num_engines,
        slack,
        CHUNK_TOKENS,
        max_chunks,
    )
    return PrefixAffinityIndex(num_engines, slack, max_chunks_per_engine=max_chunks)


def _affinity_prefix(affinity, request):
    """Return the prompt prefix of a request, or None if it cannot be matched."""
    token_ids = request.prompt_token_ids
    if (
        affinity is None
        or request.sampling_params is None
        or not token_ids
        or request.mm_features
        or getattr(request, "prompt_embeds", None) is not None
    ):
        return None
    lora = request.lora_request
    return affinity.prefix(
        token_ids,
        lora_name=lora.lora_name if lora is not None else None,
        cache_salt=request.cache_salt,
    )


def _maybe_log_affinity(self, affinity) -> None:
    now = time.monotonic()
    if now - getattr(self, "_dp_affinity_last_log", 0.0) < _AFFINITY_LOG_INTERVAL_S:
        return
    self._dp_affinity_last_log = now
    s = affinity.stats
    logger.info(
        "DP prefix affinity: %d load-balanced requests, %d with a prefix on some "
        "engines, %d sent to that engine (%.0f matched tokens each), %d sent to "
        "the least loaded engine by the load guard",
        s.requests,
        s.matched,
        s.followed,
        s.matched_tokens / max(1, s.followed),
        s.overridden,
    )


def _observed_engine_load(counts: list[int | float]) -> float:
    waiting, running = counts[:2]
    if len(counts) == 2:
        # Preserve the pinned vLLM 0.24 queue penalty.
        return float(waiting * 4 + running)

    kv_cache_usage = float(counts[2])
    return float(waiting + running + waiting * 6.0 * max(0.0, kv_cache_usage - 0.5))


def _get_core_engine_for_request(self, request):
    if not getattr(self, _ROUTING_ENABLED_ATTR):
        return _original_dplb_get_core_engine_for_request(self, request)

    from vllm.v1.pool.late_interaction import get_late_interaction_engine_index

    eng_index = request.data_parallel_rank
    if eng_index is None:
        eng_index = get_late_interaction_engine_index(
            request.pooling_params, len(self.core_engines)
        )
    load_balanced = eng_index is None

    affinity = getattr(self, _AFFINITY_ATTR, None)
    if affinity is not None:
        affinity.resize(len(self.core_engines))
    # Hash the prompt before taking the lock shared with the other API servers.
    prefix = _affinity_prefix(affinity, request)

    routing_lock = (
        self.shared_engine_inflight.get_lock()
        if self.shared_engine_inflight is not None
        else contextlib.nullcontext()
    )
    with routing_lock:
        if load_balanced:
            num_engines = len(self.lb_engines)
            min_score = (float("inf"), float("inf"))
            eng_index = 0
            order = [
                (self.eng_start_index + offset) % num_engines
                for offset in range(num_engines)
            ]
            scores: list[tuple[float, float]] = [min_score] * num_engines

            for index in order:
                observed_load = _observed_engine_load(self.lb_engines[index])
                if self.shared_engine_inflight is not None:
                    score = (
                        float(self.shared_engine_inflight[index]),
                        observed_load,
                    )
                else:
                    local_inflight = (
                        self.client_count
                        * self.engine_inflight[self.core_engines[index]]
                    )
                    score = (max(float(local_inflight), observed_load), 0.0)
                scores[index] = score

                if score < min_score:
                    min_score = score
                    eng_index = index

            if prefix is not None:
                eng_index = affinity.choose(prefix, scores, eng_index, order)

            if self.shared_engine_inflight is None:
                self.lb_engines[eng_index][0] += self.client_count
            self.eng_start_index = (self.eng_start_index + 1) % num_engines

        chosen_engine = self.core_engines[eng_index]
        self.reqs_in_flight[request.request_id] = chosen_engine
        self.engine_inflight[chosen_engine] += 1
        if self.shared_engine_inflight is not None:
            self.shared_engine_inflight[eng_index] += 1

    if prefix is not None:
        affinity.admit(request.request_id, eng_index, prefix)
        _maybe_log_affinity(self, affinity)
    return chosen_engine


async def _process_engine_outputs(self, outputs) -> None:
    if not getattr(self, _ROUTING_ENABLED_ATTR):
        await _original_dplb_process_engine_outputs(self, outputs)
        return

    affinity = getattr(self, _AFFINITY_ATTR, None)
    if affinity is not None:
        # The engine caches the generated tokens too, so the next turn of a
        # session, which starts with them, finds this engine.
        for output in outputs.outputs:
            if output.new_token_ids:
                affinity.extend(output.request_id, output.new_token_ids)
        for request_id in outputs.finished_requests or ():
            affinity.finish(request_id)

    # vLLM reports aborted requests in finished_requests after scheduler cleanup.
    # Release here so an abort stays reserved until the engine acknowledges it.
    if not outputs.finished_requests or not self.reqs_in_flight:
        return

    for request_id in outputs.finished_requests:
        engine = self.reqs_in_flight.pop(request_id, None)
        if engine is None:
            continue

        self.engine_inflight[engine] -= 1
        if self.shared_engine_inflight is not None:
            with self.shared_engine_inflight.get_lock():
                eng_index = self.engine_indices[engine]
                self.shared_engine_inflight[eng_index] -= 1


def _api_server_manager_init(
    self,
    listen_address: str,
    sock: Any,
    args,
    num_servers: int,
    input_addresses: list[str],
    output_addresses: list[str],
    target_server_fn=None,
    stats_update_address: str | None = None,
    tensor_queue=None,
) -> None:
    external_dp = getattr(args, "data_parallel_external_lb", False)
    local_engines_only = external_dp or getattr(args, "data_parallel_hybrid_lb", False)
    size_attr = (
        "data_parallel_size_local" if local_engines_only else "data_parallel_size"
    )
    managed_engine_count = int(getattr(args, size_attr, None) or 1)

    if external_dp or managed_engine_count <= 1:
        _original_api_server_manager_init(
            self,
            listen_address,
            sock,
            args,
            num_servers,
            input_addresses,
            output_addresses,
            target_server_fn,
            stats_update_address,
            tensor_queue,
        )
        return

    shared_inflight = (
        multiprocessing.get_context("spawn").Array("q", managed_engine_count)
        if num_servers > 1
        else None
    )
    setattr(args, _SHARED_INFLIGHT_ARG, shared_inflight)
    self.shared_dp_lb_inflight = shared_inflight

    if target_server_fn is None:
        target_server_fn = _run_api_server_worker_proc

    _original_api_server_manager_init(
        self,
        listen_address,
        sock,
        args,
        num_servers,
        input_addresses,
        output_addresses,
        target_server_fn,
        stats_update_address,
        tensor_queue,
    )


def _run_api_server_worker_proc(
    listen_address, sock, args, client_config=None, **uvicorn_kwargs
) -> None:
    # Spawned API workers import this module with fresh globals.
    from vllm_neuron.vllm.patches import apply_patches

    apply_patches()

    shared_inflight = getattr(args, _SHARED_INFLIGHT_ARG, None)
    if shared_inflight is not None:
        client_config = dict(client_config or {})
        client_config[_SHARED_INFLIGHT_CONFIG] = shared_inflight

    from vllm.v1.utils import run_api_server_worker_proc

    run_api_server_worker_proc(
        listen_address,
        sock,
        args,
        client_config,
        **uvicorn_kwargs,
    )


def apply_dp_load_balancer_patch() -> None:
    """Apply the internal-DP routing fix to the pinned vLLM version.

    vLLM loads general plugins before runtime configuration is available.
    Patched entry points therefore gate on internal-DP state when invoked and
    delegate unchanged configurations to their original implementations.

    Example:
        ``apply_dp_load_balancer_patch()``
    """
    global _applied
    global _original_api_server_manager_init
    global _original_dplb_client_init
    global _original_dplb_get_core_engine_for_request
    global _original_dplb_process_engine_outputs
    global _original_engine_core_run_busy_loop

    if _applied:
        return

    from vllm.v1 import utils as vllm_utils
    from vllm.v1.engine.core import EngineCoreProc
    from vllm.v1.engine.core_client import DPLBAsyncMPClient

    _original_engine_core_run_busy_loop = EngineCoreProc.run_busy_loop
    _original_dplb_client_init = DPLBAsyncMPClient.__init__
    _original_dplb_get_core_engine_for_request = (
        DPLBAsyncMPClient.get_core_engine_for_request
    )
    _original_dplb_process_engine_outputs = DPLBAsyncMPClient.process_engine_outputs
    _original_api_server_manager_init = vllm_utils.APIServerProcessManager.__init__

    EngineCoreProc._maybe_publish_request_counts = _maybe_publish_request_counts
    EngineCoreProc.run_busy_loop = _engine_core_run_busy_loop
    DPLBAsyncMPClient.__init__ = _dplb_client_init
    DPLBAsyncMPClient.get_core_engine_for_request = _get_core_engine_for_request
    DPLBAsyncMPClient.process_engine_outputs = staticmethod(_process_engine_outputs)
    vllm_utils.APIServerProcessManager.__init__ = _api_server_manager_init

    _applied = True
    logger.info("Applied internal-DP load-balancer patch")
