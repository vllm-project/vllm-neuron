# SPDX-License-Identifier: Apache-2.0
"""
NeuronWorker - Worker implementation for vLLM integration

This provides the main worker class that implements vLLM's WorkerBase interface
for AWS Neuron/Trainium hardware integration through vLLM Neuron.
"""

import contextlib
import logging
import os
import sys
import math
import re
import time
from datetime import timedelta
from functools import wraps
from typing import Any

import torch
import vllm.distributed.parallel_state as parallel_state
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    has_kv_transfer_group,
    get_kv_transfer_group,
)
from vllm.profiler.wrapper import TorchProfilerWrapper, WorkerProfiler
from vllm.utils.torch_utils import set_random_seed
from vllm.tasks import SupportedTask
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase
from vllm.v1.outputs import (
    DraftTokenIds,
    ModelRunnerOutput,
    AsyncModelRunnerOutput,
)
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.v1.core.sched.output import GrammarOutput

from vllm_neuron import envs
from vllm_neuron.metrics import STARTUP_TIME
from vllm_neuron.model.interfaces import SupportsEncoderWarmup
from vllm_neuron.utils.bucket_utils import segmented_prefill_supports_q_len
from vllm_neuron.utils.neuron_utils import in_lane_grad_mode
from vllm_neuron.utils.hardware_config import (
    get_efa_interface,
    get_logical_cores_per_hbm_stack,
    parse_range_list,
)
from vllm.utils.network_utils import get_open_port
from vllm_neuron.parallel.neuron_parallel_state import tp_barrier
from vllm_neuron.vllm.patches.shm_reader_spin_patch import (
    apply_shm_reader_spin_patch,
)
from vllm_neuron.vllm.platform import NeuronPlatform
from vllm_neuron.vllm.worker.neuron_profiler import (
    NeuronProfilerConfig,
    NeuronProfiler,
)

logger = logging.getLogger(__name__)

# Largest GIL switch interval this accepts, in seconds. CPython's own default is 0.005 s, and the
# point of the knob is to go shorter, so anything above a second is a typo rather than a setting.
_MAX_GIL_SWITCH_INTERVAL_S = 1.0


def _apply_gil_switch_interval() -> None:
    """Shorten the GIL switch interval of this process, when asked to.

    A shorter interval makes a thread holding the GIL inside a native device readback yield more
    often, which shortens the wait a step thread can lose repeatedly. It is off unless
    ``VLLM_NEURON_GIL_SWITCH_INTERVAL_S`` is set, because the interval is global to the
    interpreter: it changes thread scheduling for every model this worker serves, while the
    measurement behind it covers one decode configuration. On that configuration it is worth
    nothing measurable, so the default costs no performance. CPython's own default is 0.005 s.

    Call it one time in each worker process.

    Example:
        >>> _apply_gil_switch_interval()   # doctest: +SKIP
    """
    interval = envs.VLLM_NEURON_GIL_SWITCH_INTERVAL_S
    if interval <= 0:
        return
    if not math.isfinite(interval) or interval > _MAX_GIL_SWITCH_INTERVAL_S:
        # ``sys.setswitchinterval`` accepts inf and nan without complaining, and either one
        # leaves the interpreter barely switching threads at all. Refuse instead: a value this
        # far from CPython's 0.005 s is a typo, not a tuning decision.
        logger.error(
            "VLLM_NEURON_GIL_SWITCH_INTERVAL_S=%r is outside (0, %s], so CPython's %.6f s is "
            "kept. A GIL switch interval near this is a typo, and an infinite one would stop "
            "this process from switching threads.",
            interval,
            _MAX_GIL_SWITCH_INTERVAL_S,
            sys.getswitchinterval(),
        )
        return
    previous = sys.getswitchinterval()
    sys.setswitchinterval(interval)
    logger.info(
        "GIL switch interval %.6f s -> %.6f s",
        previous,
        sys.getswitchinterval(),
    )


_LOOPBACK_ADDRS = (None, "", "127.0.0.1", "localhost", "::1")
_HEADROOM_ENV = "VLLM_NEURON_RESERVED_HEADROOM_GIB"
_HBM_ALLOCATION_ERROR_SIGNATURES = (
    "ncc_eoom002",
    "nrt_resource",
    "nrt_load_util",
    "nrt_tensor_allocate status=4",
    "allocation failure",
    "maximum peak hbm usage",
)


def _exception_chain_text(exc: BaseException) -> str:
    """Return searchable text from an exception and its chained causes."""
    messages = []
    seen = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current))
        current = current.__cause__ or current.__context__
    return "\n".join(messages).lower()


def _is_hbm_allocation_error(exc: BaseException) -> bool:
    """Return whether an exception contains a known Neuron HBM OOM signature."""
    text = _exception_chain_text(exc)
    return any(signature in text for signature in _HBM_ALLOCATION_ERROR_SIGNATURES)


def _headroom_tuning_guidance(env_name: str, current_gib: float) -> str:
    """Build actionable guidance for a compile/load HBM allocation failure."""
    return (
        "Neuron HBM allocation failed while loading, initializing, compiling, "
        "or warming up the model. The KV cache may be leaving too little HBM "
        f"for non-KV allocations. Increase {env_name} above the current "
        f"{current_gib:g} GiB value, starting in 1 GiB increments. Each "
        "additional 1 GiB of headroom reduces the KV cache budget by 1 GiB. "
        "Rerun the same model, sharding, backend, and hardware configuration, "
        "then keep the smallest value that passes. Do not increase headroom for "
        "'Computed KV cache budget is below minimum threshold'; decrease it for "
        "that error."
    )


def _add_headroom_guidance_to_hbm_errors(func):
    """Annotate known Neuron HBM allocation errors with tuning guidance."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as exc:
            if _is_hbm_allocation_error(exc) and envs.get_kv_headroom_env() is not None:
                headroom_bytes, _ = self._get_reserved_headroom_bytes()
                exc.add_note(
                    _headroom_tuning_guidance(
                        self._active_headroom_env(),
                        headroom_bytes / (1024**3),
                    )
                )
            raise

    return wrapper


def validate_cross_node_master_addr(
    nnodes: int,
    master_addr: str | None,
    data_parallel_master_ip: str | None,
) -> None:
    """Assert ``parallel_config.master_addr`` is set for the cross-node
    _WORLD rendezvous.

    vllm core's ``init_distributed_environment`` uses
    ``parallel_config.master_addr`` (not ``data_parallel_master_ip``) for the
    flat ``world_size_across_dp = TP*PP*DP`` group when ``nnodes>1``.
    ``master_addr`` defaults to the loopback ``"127.0.0.1"`` and is populated
    only by the ``--master-addr`` CLI flag — so a multi-pod DP launch that
    omits ``--master-addr`` leaves ``master_addr`` on loopback and the
    secondary pod's ranks deadlock rendezvousing on their own loopback.
    ``master_addr`` and ``data_parallel_master_ip`` represent two different
    things in vLLM (the TP/PP/DP world rendezvous vs. the DP coordinator), so
    we surface a clear error rather than silently substituting one for the
    other.
    """
    if nnodes > 1 and master_addr in _LOOPBACK_ADDRS:
        raise ValueError(
            f"nnodes={nnodes} but parallel_config.master_addr={master_addr!r} "
            "is the loopback default — the cross-node _WORLD rendezvous would "
            "deadlock on the secondary pod's own loopback. Pass "
            "--master-addr <HEAD_NODE_IP> on every pod (typically the same IP "
            f"as --data-parallel-address, e.g. {data_parallel_master_ip!r})."
        )


def resolve_ep_degree(
    ep_degree: int,
    world_size_across_dp: int,
    enable_expert_parallel: bool,
) -> int:
    """Resolve and validate the expert-parallel degree.

    Rules:
      * ``ep_degree > 1`` requires ``--enable-expert-parallel``.
      * When EP is enabled and ``ep_degree`` is left at its default of 1,
        it expands to ``world_size_across_dp`` (TP * DP) so experts span
        every rank, including across DP replicas.
      * The only hard constraint on an explicit ``ep_degree`` is that
        ``world_size_across_dp`` be divisible by it. Wide-EP layouts where
        ``ep_degree < TP * DP`` (e.g. TP=1 DP=128 EP=32 -> ep_tp=4) are
        valid: the model shards the MoE intermediate dim within each EP
        partition via ``_NEURON_EP_TP`` collectives. The previous strict
        ``ep_degree == world_size_across_dp`` check was the only thing
        blocking those topologies from configuring.

    Returns the resolved ``ep_degree``.
    """
    if not enable_expert_parallel:
        if ep_degree > 1:
            raise ValueError(
                f"ep_degree={ep_degree} requires --enable-expert-parallel to be set."
            )
        return ep_degree
    if ep_degree == 1:
        ep_degree = world_size_across_dp
    if world_size_across_dp % ep_degree != 0:
        raise ValueError(
            f"world_size (TP*DP={world_size_across_dp}) "
            f"must be divisible by ep_degree ({ep_degree})."
        )
    return ep_degree


_CCOM_BASE_PORT = 61234
"""Base port for deterministic CCOM bootstrap rendezvous.

When ``NEURON_VISIBLE_DEVICES`` is set the ephemeral path derives its port as
``_CCOM_BASE_PORT + first_device`` (see ``_ephemeral_ccom_addr``).  This is
the same base used by the pytest-xdist harness in ``test/conftest.py`` and the
DI logit tests, so the values stay in sync across the codebase.
"""


def _ephemeral_ccom_addr(master_addr: str) -> str:
    """Return a deterministic CCOM root address for the current engine.

    Preferred over :func:`get_open_port` when ``NEURON_VISIBLE_DEVICES`` is
    set, because probing a free port without holding it open causes a race when
    multiple DI engines (prefill + decode(s)) start concurrently on the same
    host: all engines can observe the same free port and pin their
    ``NEURON_RT_ROOT_COMM_ID`` to it, deadlocking the CCOM bootstrap.

    When ``NEURON_VISIBLE_DEVICES`` is set the first device index is unique per
    engine (each engine gets a non-overlapping slice), so keying the port on it
    guarantees distinct root endpoints without any inter-process coordination.

    Falls back to :func:`get_open_port` when ``NEURON_VISIBLE_DEVICES`` is not
    set (single-engine or legacy configurations).

    Args:
        master_addr: Hostname or IP to use as the CCOM root address.

    Returns:
        A ``"host:port"`` string to use as ``NEURON_RT_ROOT_COMM_ID``.

    Examples:
        >>> # NEURON_VISIBLE_DEVICES="4-7" → first device is 4
        >>> os.environ["NEURON_VISIBLE_DEVICES"] = "4-7"
        >>> _ephemeral_ccom_addr("localhost")
        'localhost:61238'
        >>> # NEURON_VISIBLE_DEVICES="0,1,2,3" → first device is 0
        >>> os.environ["NEURON_VISIBLE_DEVICES"] = "0,1,2,3"
        >>> _ephemeral_ccom_addr("localhost")
        'localhost:61234'
    """
    addr = _ccom_addr_from_visible_devices(master_addr)
    if addr is not None:
        return addr
    port = get_open_port()
    return f"{master_addr}:{port}"


def _ccom_addr_from_visible_devices(master_addr: str) -> str | None:
    """Derive a per-engine CCOM address from ``NEURON_VISIBLE_DEVICES``.

    Returns ``master_addr:(_CCOM_BASE_PORT + first_device)`` when
    ``NEURON_VISIBLE_DEVICES`` names a numeric first device, else ``None``.
    The first device index is unique per co-located engine, so this yields a
    distinct, deterministic root socket for each engine with no inter-process
    coordination. Split out so both the ephemeral path and
    :func:`rendezvous_ccom_bootstrap` can consult it.
    """
    visible = os.environ.get("NEURON_VISIBLE_DEVICES", "")
    if not visible:
        return None
    first_token = visible.split(",")[0].split("-")[0].strip()
    if not first_token.isdigit():
        return None
    return f"{master_addr}:{_CCOM_BASE_PORT + int(first_token)}"


def _offset_comm_id_port(addr: str, data_parallel_index: int) -> str:
    """Offset the port of a ``host:port`` CCOM address by ``data_parallel_index``.

    DP index 0 keeps the pinned port unchanged (preserving the single-replica
    and TP-only behaviour); each higher replica shifts to a distinct port so
    independent DP gloo worlds don't race onto the same root socket. Returns
    ``addr`` unchanged if it has no parseable ``host:port`` form.
    """
    if data_parallel_index == 0:
        return addr
    host, sep, port = addr.rpartition(":")
    if not sep or not port.isdigit():
        return addr
    return f"{host}:{int(port) + data_parallel_index}"


def rendezvous_ccom_bootstrap(data_parallel_index: int = 0):
    """Broadcast NEURON_RT_ROOT_COMM_ID from rank 0 via gloo.

    Must be called after ``init_distributed_environment()`` (gloo is up) and
    before the first NEFF execution (which triggers CCOM bootstrap).

    Rank 0 constructs the CCOM address (from a pinned base or ``MASTER_ADDR``
    plus a free port), then broadcasts it to all workers in its gloo world so
    every worker shares one CCOM bootstrap endpoint.

    DP topology on Neuron — each DP replica is its OWN gloo world:
      - **No DP / external DP**: each process has its own gloo world with its
        own rank 0, so each independently picks its own port.
      - **Internal DP** (``--data-parallel-size > 1``): vLLM spawns one
        ``EngineCore`` process *per* DP replica, and each runs its own
        MultiprocExecutor with its own gloo world of size TP (NOT one flat
        TP*DP world) and its own rank 0. The replicas form independent CCOM
        communicators, so each needs a **distinct** root endpoint.

    ``NEURON_RT_ROOT_COMM_ID`` may be pinned in the environment (e.g. the
    parallel-test harness pins it per core allocation to avoid an ephemeral
    port race between concurrent servers). Because the pinned value is
    inherited identically by every DP engine, rank 0 offsets the pinned port
    by ``data_parallel_index`` so the independent replicas don't collide on
    the same root socket — which would otherwise leave all-but-one replica
    hanging at ``[N-ranks bootstrap: rank K awaiting root parameters]``.

    The ephemeral (unpinned) path uses :func:`_ephemeral_ccom_addr`. When
    ``NEURON_VISIBLE_DEVICES`` is set (always true in DI mode) it derives a
    deterministic port from the engine's first device index, giving each
    co-located engine a distinct root endpoint without any inter-process
    coordination. This avoids the race where concurrent ``get_open_port()``
    probes all observe the same free port and deadlock the CCOM bootstrap.
    When ``NEURON_VISIBLE_DEVICES`` is not set it falls back to
    ``get_open_port()`` (safe for single-engine configurations).

    ``data_parallel_index`` (not ``data_parallel_rank``) is the caller's source
    of truth: vLLM resets ``data_parallel_rank`` to 0 for dense (non-MoE) DP
    engines — which it treats as independent DP=1 servers — while
    ``data_parallel_index`` always reflects the true DP rank for both dense and
    MoE engines.

    No-op in CPU mode (no CCOM needed).
    """
    import torch.distributed as dist

    if envs.VLLM_NEURON_CPU_MODE or envs.VLLM_NEURON_CPU_COMPILE:
        return

    if dist.get_rank() == 0:
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        # Precedence:
        #   1. NEURON_VISIBLE_DEVICES-derived per-engine port (DI mode). This
        #      MUST win over any inherited NEURON_RT_ROOT_COMM_ID: the runtime
        #      dep ``libtorch_neuronx_lite`` presets that var at import time to
        #      MASTER_ADDR:MASTER_PORT (default ``localhost:62182``) — the SAME
        #      value in every co-located engine's process. Treating that shared
        #      preset as a "pin" is exactly what deadlocks the CCOM bootstrap
        #      (all engines' rank 0 race onto one root socket). The per-device
        #      port is unique per engine, so prefer it whenever NVD is set.
        #   2. Otherwise honor an explicitly pinned NEURON_RT_ROOT_COMM_ID (the
        #      xdist harness pins it per core allocation), offsetting by the DP
        #      index so independent DP replicas don't share a root socket.
        #   3. Otherwise fall back to an ephemeral free port.
        #
        # The DP-index offset is applied only on the pinned path: every DP
        # replica inherits the SAME pinned base, so they must be shifted apart.
        # The NVD-derived port is already unique per engine (distinct core
        # slice), so it needs no offset.
        addr = _ccom_addr_from_visible_devices(master_addr)
        if addr is None:
            pinned = os.environ.get("NEURON_RT_ROOT_COMM_ID")
            if pinned:
                addr = _offset_comm_id_port(pinned, data_parallel_index)
            else:
                addr = _ephemeral_ccom_addr(master_addr)
    else:
        addr = None

    addr_list = [addr]
    dist.broadcast_object_list(addr_list, src=0, device=torch.device("cpu"))
    addr = addr_list[0]

    os.environ["NEURON_RT_ROOT_COMM_ID"] = addr
    logger.info(
        "CCOM bootstrap rendezvous: NEURON_RT_ROOT_COMM_ID=%s (rank %d/%d)",
        addr,
        dist.get_rank(),
        dist.get_world_size(),
    )


_NRT_INIT_RETRY_TIMEOUT_S = 60
_NRT_INIT_RETRY_POLL_S = 2


def _nrt_initialize() -> None:
    """Create and initialize the Neuron runtime binding."""
    runtime = torch.classes.neuron.Runtime()
    runtime.initialize()


def _initialize_neuron_runtime_with_retry(
    timeout_s: float = _NRT_INIT_RETRY_TIMEOUT_S,
    poll_s: float = _NRT_INIT_RETRY_POLL_S,
) -> None:
    """Initialize the Neuron runtime, retrying while NeuronCores are reclaimed.

    NeuronCore release lags engine-process exit: the driver reclaims cores
    asynchronously after the previous engine shuts down, so a new engine
    started back-to-back (same process or same host) can fail ``nrt_init``
    with cores still held. A failed ``nrt_init`` leaves the runtime
    uninitialized, so retrying in-process is safe. Genuine initialization
    failures (driver missing, no devices) still surface after ``timeout_s``.
    Retry only triggers on failure, so servers running in parallel on other
    cores are unaffected.

    Args:
        timeout_s: Total time budget for retries before re-raising.
        poll_s: Delay between attempts.

    Raises:
        RuntimeError: The last initialization error once ``timeout_s`` elapses.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            _nrt_initialize()
            return
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            logger.warning(
                "Neuron runtime initialization failed (NeuronCores may still "
                "be held by a previous engine); retrying in %.0fs",
                poll_s,
            )
            time.sleep(poll_s)


class _SuppressModelRegistryOverwrite(logging.Filter):
    """Filter to suppress expected model registration overwrite warnings from vLLM.

    This filter is used when registering vLLM Neuron's Neuron-optimized model implementations
    with vLLM's ModelRegistry. Since we intentionally override vLLM's default models,
    the "already registered" warnings are expected and not actionable.
    """

    # Matches: "Model architecture LlamaForCausalLM is already registered, and will be overwritten..."
    _PATTERN = re.compile(
        r"Model architecture \w+ is already registered.*will be overwritten"
    )

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False to suppress matching messages, True to allow them through."""
        return not self._PATTERN.search(record.getMessage())


class NeuronWorker(WorkerBase):
    """
    Worker implementation for AWS Neuron/Trainium hardware.
    This class implements the vLLM WorkerBase interface.
    """

    def __init__(
        self,
        vllm_config: Any | None = None,
        local_rank: int = 0,
        rank: int = 0,
        distributed_init_method: str | None = None,
        is_driver_worker: bool = True,
    ):
        """
        Initialize the Neuron worker.

        Args:
            vllm_config: vLLM configuration object
            local_rank: Local rank of this worker
            rank: Global rank of this worker
            distributed_init_method: Method for distributed initialization
            is_driver_worker: Whether this is the driver worker
        """
        # vLLM serve spawns workers in fresh interpreters. The cached bool
        # survives config pickling, but the class patch does not, so reinstall
        # the Neuron getter before this process reads use_v2_model_runner.
        from vllm_neuron.vllm.model_runner_v2_policy import (
            patch_vllm_config_for_neuron,
        )

        patch_vllm_config_for_neuron()
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        self.use_async_scheduling = vllm_config.scheduler_config.async_scheduling

        self.vllm_config = vllm_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        self._startup_start_time = time.perf_counter()

        # Model runner will be created in init_device()
        # Following GPU/TPU worker pattern
        self.model_runner = None
        self.device = None

        # Register models with ModelRegistry
        # Put it in the worker as vLLM's model registry overwrites vllm_neuron model registry
        # if we put it too early in the platform.py.
        # We need to use or create a platform API for OOT platform model registry.
        # Tracked as TODO below.
        # TODO [CHRYS-72]: Investigate vLLM model registry
        from vllm import ModelRegistry
        from vllm_neuron.model import registry

        # Suppress the vLLM warning about overwriting existing model registrations.
        # This is expected behavior - we intentionally override vLLM's default model
        # implementations with vLLM Neuron's Neuron-optimized versions.
        vllm_registry_logger = logging.getLogger("vllm.model_executor.models.registry")
        suppress_filter = _SuppressModelRegistryOverwrite()
        vllm_registry_logger.addFilter(suppress_filter)
        try:
            for arch, model_cls in registry.get_models():
                ModelRegistry.register_model(arch, model_cls)
        finally:
            vllm_registry_logger.removeFilter(suppress_filter)

        self._profiler: WorkerProfiler | None = None

        from vllm_neuron.manifest import ManifestWriter

        self._manifest_writer = ManifestWriter.from_env()

        logger.info(
            "Initialized NeuronWorker with rank %s, local_rank %s", rank, local_rank
        )

    def _set_efa_affinity(self, visible_devices) -> str:
        """
        Set EFA interface affinity for optimal network performance.

        Delegates the full device-to-EFA-interface resolution to
        hardware_config.get_efa_interface().

        Args:
            visible_devices: List of visible Neuron device indices

        Returns:
            str: Name of the EFA interface assigned to this host worker

        Raises:
            RuntimeError: If the Neuron device cannot be accessed or EFA interface not found
            IndexError: If the computed device index is out of bounds
        """
        efa = get_efa_interface(self.local_rank, visible_devices)
        os.environ["FI_EFA_IFACE"] = efa
        logger.debug(
            "Set EFA interface to %s for local_rank %s",
            efa,
            self.local_rank,
        )
        return efa

    def _set_cpu_affinity(self) -> None:
        """
        Set CPU affinity for all threads based on EFA's NUMA node.

        Pins all threads in the current process to CPUs in the same NUMA node as
        the EFA interface to minimize interconnection overhead and improve performance.
        This is critical for both single-instance and disaggregated inference workloads.

        Queries the EFA interface name from the FI_EFA_IFACE environment variable.
        """
        efa = os.environ.get("FI_EFA_IFACE")
        numa_node_path = (
            f"/sys/class/infiniband/{efa}/device/numa_node" if efa else None
        )

        if efa and numa_node_path and os.path.exists(numa_node_path):
            with open(f"/sys/class/infiniband/{efa}/device/numa_node") as f:
                numa = int(f.read())
            with open(f"/sys/devices/system/node/node{numa}/cpulist") as f:
                cpulist = parse_range_list(f.read())

            # Set affinity for all existing threads in the process
            # This handles any vLLM threads that may have been created before init_device()
            # e.g., ZeroMQ I/O or reaper threads
            threads_set = 0
            try:
                thread_ids = os.listdir("/proc/self/task")
                logger.debug(
                    "Setting affinity for %s threads to NUMA node %s",
                    len(thread_ids),
                    numa,
                )
                for tid in thread_ids:
                    try:
                        os.sched_setaffinity(int(tid), cpulist)
                        threads_set += 1
                    except OSError as ose:
                        # Thread may have exited or we don't have permission
                        logger.warning(
                            "Failed to set affinity for thread %s: %s", tid, ose
                        )
            except FileNotFoundError as fnfe:
                logger.warning(
                    "Could not enumerate threads (/proc/self/task not found): %s", fnfe
                )
                os.sched_setaffinity(0, cpulist)
                threads_set = 1

            logger.debug(
                "Set CPU affinity for %s thread(s) to NUMA node %s, CPUs: %s",
                threads_set,
                numa,
                cpulist,
            )
        else:
            logger.warning(
                "EFA device %s not found, skipping CPU affinity settings", efa
            )

    def init_device(self) -> None:
        """
        Initialize the device for this worker.

        Sets up Neuron environment variables, device, and creates model runner.
        Follows the vLLM GPU/TPU worker pattern.

        Initialization order:
        1. ``_init_neuron_distributed_environment_and_runtime()`` – affinity,
           env vars, gloo init, rendezvous_ccom_bootstrap, runtime.initialize(),
           ensure_kv_transfer_initialized
        2. Create torch device and model runner
        """
        # Pin device-worker CPU threads to 1. Workers inherit the frontend's
        # OMP_NUM_THREADS default (set for multimodal patchify), which skips
        # vLLM's own worker cap; without this reset the workers would
        # oversubscribe the host. Mirrors vllm_neuron/utils/executor.py.
        os.environ["OMP_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        # init_device runs one time in each worker process, so set the interval here.
        _apply_gil_switch_interval()
        # WorkerProc builds its MessageQueue reader AFTER init_device and after load_model,
        # so this is a deterministic point at which the reader default still applies.
        apply_shm_reader_spin_patch()

        if envs.VLLM_NEURON_CPU_MODE:
            self.device = torch.device("cpu")
        elif envs.VLLM_NEURON_CPU_COMPILE:
            self.device = torch.device("meta")

        # Install replica-mesh patches before any process group is created.
        # Deferred from plugin-load time (vllm_neuron/__init__.py) because
        # vllm.distributed cannot be safely imported while the vLLM plugin
        # loader is still mid-init (circular vllm.utils.torch_utils). By the
        # time init_device() runs, vLLM has finished importing.
        from vllm_neuron.vllm.patches.replica_mesh_patch import (
            apply_replica_mesh_patches,
        )

        apply_replica_mesh_patches()

        self._init_neuron_distributed_environment_and_runtime(
            self.vllm_config, self.rank, self.distributed_init_method, self.local_rank
        )

        if self._use_neuron_device():
            self.device = torch.device(f"neuron:{self.local_rank}")

        # Create model runner with vllm_config and device
        if self.vllm_config.use_v2_model_runner:
            from .neuron_model_runner_v2 import NeuronModelRunnerV2

            self.model_runner = NeuronModelRunnerV2(
                vllm_config=self.vllm_config, device=self.device
            )
        else:
            from .neuron_model_runner import NeuronModelRunner

            self.model_runner = NeuronModelRunner(
                vllm_config=self.vllm_config, device=self.device
            )

        # Set random seed
        set_random_seed(self.model_config.seed)

        logger.debug(
            "Device initialized: device=%s, local_rank=%s, rank=%s",
            self.device,
            self.local_rank,
            self.rank,
        )

    def _get_visible_devices(self):
        if "NEURON_RT_VISIBLE_CORES" in os.environ:
            raise RuntimeError(
                "NEURON_RT_VISIBLE_CORES cannot be used with multi-processing execution on vLLM. "
                "Set NEURON_VISIBLE_DEVICES to control device visibility."
            )
        parallel_config = self.vllm_config.parallel_config
        # Cores this engine process spans on the local node (TP*PP within one
        # data-parallel replica).
        num_local_ranks = parallel_config.local_world_size

        # With data parallelism, several replica engine processes are launched
        # on the same node and all inherit the node-wide NEURON_VISIBLE_DEVICES
        # (e.g. "0-7" for 2 replicas x 4 cores). vLLM slices the node's cores
        # per DP-local rank and records this replica's disjoint window in
        # parallel_config.assigned_physical_gpu_ids (see
        # get_physical_gpu_ids_for_local_dp_rank). Prefer it so replicas bind
        # non-overlapping cores. It is only populated when DP places multiple
        # replicas on a node; otherwise fall back to NEURON_VISIBLE_DEVICES.
        assigned = parallel_config.assigned_physical_gpu_ids
        if assigned:
            neuron_visible_devices = list(assigned)
            # The per-replica slice lives on the config, not the environment,
            # so narrow NEURON_VISIBLE_DEVICES to this replica's cores for
            # libtorchneuron and other consumers that read the env var.
            os.environ["NEURON_VISIBLE_DEVICES"] = ",".join(
                str(c) for c in neuron_visible_devices
            )
        else:
            if "NEURON_VISIBLE_DEVICES" not in os.environ:
                # Defaulting to cores 0..num_local_ranks-1 is only safe when
                # this is the sole replica on the node. With multiple local DP
                # replicas, vLLM must supply assigned_physical_gpu_ids; without
                # it, every replica would default to the same cores and collide,
                # so fail loudly instead.
                if parallel_config.data_parallel_size_local > 1:
                    raise RuntimeError(
                        "Neuron does not support this data-parallel "
                        "configuration: vLLM did not provide per-replica device "
                        "assignment."
                    )
                # NEURON_VISIBLE_DEVICES must be set for libtorchneuron
                os.environ["NEURON_VISIBLE_DEVICES"] = ",".join(
                    str(i) for i in range(num_local_ranks)
                )
            neuron_visible_devices = parse_range_list(
                os.getenv("NEURON_VISIBLE_DEVICES")
            )

        if len(neuron_visible_devices) != num_local_ranks:
            raise RuntimeError(
                f"Resolved {len(neuron_visible_devices)} Neuron cores "
                f"{neuron_visible_devices} for this replica, but "
                f"local_world_size={num_local_ranks} "
                f"(NEURON_VISIBLE_DEVICES={os.getenv('NEURON_VISIBLE_DEVICES')!r}, "
                f"assigned_physical_gpu_ids={assigned})"
            )
        return neuron_visible_devices

    def get_kv_connector_handshake_metadata(self) -> dict | None:
        """Get KV connector metadata from this worker if available."""

        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()
        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        if (metadata := connector.get_handshake_metadata()) is None:
            return None

        pp_rank = get_pp_group().rank_in_group
        tp_rank = get_tp_group().rank_in_group
        return {(pp_rank, tp_rank): metadata}

    def _patch_in_same_node_as_function(self):
        # WORKAROUND: Patch in_the_same_node_as() to avoid PyTorch barrier() bug with Neuron device backend
        #
        # THE PROBLEM:
        # vLLM's parallel_state.in_the_same_node_as() function calls torch.distributed.barrier(group=pg)
        # to synchronize ranks and find ranks that share the same shared memory.
        # However, PyTorch's barrier() implementation
        # (in torch/distributed/distributed_c10d.py, line ~4792) unconditionally calls:
        #     device = torch._C._get_accelerator()
        # BEFORE checking the backend type or dispatching to the backend-specific barrier implementation.
        #
        # For Neuron (which uses PyTorch's PrivateUse1 extension mechanism), _get_accelerator() requires
        # PrivateUse1HooksInterface to be registered. Since we haven't registered it, this causes:
        #     RuntimeError: Please register PrivateUse1HooksInterface by `RegisterPrivateUse1HooksInterface` first.
        #
        # This error occurs even though:
        # 1. The process group (pg) is using the Gloo backend (CPU-based)
        # 2. Gloo's barrier implementation doesn't need device/accelerator information
        # 3. The barrier would work fine if PyTorch didn't try to detect the device first
        #
        # CALL CHAIN WHERE THIS BREAKS:
        # init_model_parallel_group()
        #   → GroupCoordinator.__init__()
        #   → MessageQueue.create_from_process_group()
        #   → in_the_same_node_as(pg)
        #   → torch.distributed.barrier(group=pg)
        #   → torch._C._get_accelerator()  ← CRASH HERE
        #
        # THIS WORKAROUND:
        # We patch in_the_same_node_as() to bypass the barrier() call entirely and calculate
        # the result based on ParallelConfig, rather than probe the system. This allows:
        # - MessageQueue to use shared memory for inter-rank communication
        # - Process group initialization to complete successfully
        # - Single-node and multi-node distributed inference/training to work
        #
        # PROPER SOLUTIONS (either one will work):
        # 1. Register PrivateUse1HooksInterface for Neuron:
        #    - Will come with consolidation with Torch Eager
        # 2. Register vLLM Neuron plugin from a separate python package
        #    - Will mean that PrivateUse1 is not bound and hence error isn't hit.
        def patched_in_the_same_node_as(pg, source_rank=0):
            # Compute ranks_per_node from the cluster-wide world (TP*PP*DP),
            # not parallel_config.world_size (TP*PP per DP replica). With
            # DP>1, an EP/cross-DP group spans all nodes, and we need to
            # map each group-local rank back to its global rank to assign
            # it to a physical node correctly. Mirrors the patch in
            # neuron_parallel_state._ensure_vllm_parallel_state.
            parallel_config = self.vllm_config.parallel_config
            nnodes = parallel_config.nnodes
            ranks_per_node = max(parallel_config.world_size_across_dp // nnodes, 1)
            global_ranks = [
                torch.distributed.get_global_rank(pg, r)
                for r in range(torch.distributed.get_world_size(group=pg))
            ]
            source_node = global_ranks[source_rank] // ranks_per_node
            return [g // ranks_per_node == source_node for g in global_ranks]

        parallel_state.in_the_same_node_as = patched_in_the_same_node_as

    def _init_neuron_distributed_environment_and_runtime(
        self,
        vllm_config: VllmConfig,
        rank: int,
        distributed_init_method: str | None,
        local_rank: int,
    ) -> None:
        """Initialize the distributed environment and Neuron runtime.

        For non-CPU mode the full sequence is:
        1. Set EFA affinity for network performance
        2. Set CPU affinity to match EFA's NUMA node (all threads including main)
        3. Set visible cores, device count, and HBM mapping
        4. Initialize gloo distributed backend
        5. rendezvous_ccom_bootstrap  →  sets unique COMM_ID per server
        6. runtime.initialize()       →  latches COMM_ID, maps HBM
        7. ensure_kv_transfer_initialized  →  NIXL registers HBM memory
        """

        # --- Prepare runtime environment (affinity, env vars) ---------------
        if self._use_neuron_device():
            visible_devices = self._get_visible_devices()

            if os.environ.get("NEURON_SKIP_EFA_AFFINITY", "0") == "0":
                self._set_efa_affinity(visible_devices)
            self._set_cpu_affinity()

            # Set visible cores and platform device count. The worker thread
            # has one visible core, while the platform device count is the
            # number of visible cores on the local node.
            NeuronPlatform.set_device_count(len(visible_devices))
            os.environ["NEURON_RT_VISIBLE_CORES"] = (
                f"{visible_devices[self.local_rank]}"
            )

            # Enable HBM mapping (required for RDMA in disaggregated
            # inference and other use cases)
            os.environ["NEURON_RT_MAP_HBM"] = "1"

            # Default the runtime execution queue depth (8 -> 32) to avoid
            # intermittent Execution Queue Full issues under the async flow.
            # Temporary mitigation, tracking P443231617.
            os.environ.setdefault("NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS", "32")

            # Default the DMA I/O ring cache (1 -> 32) so the runtime reuses
            # the per-step DMA ring across decode steps instead of rebuilding
            # it every step (kbl_exec_pre), which otherwise dominates
            # async-decode model-submit time. A cache-size sweep (see PR
            # description) shows 32 is the knee: it captures the full
            # latency win while the TPOT/HBM curves have both flattened, so
            # going higher only costs HBM for no gain. Override via env.
            # See P453665601.
            os.environ.setdefault("NEURON_RT_IO_RING_CACHE_SIZE", "32")

            # Disable the per-exec execution barrier: the cross-rank rendezvous
            # wait (enc_barrier) is the other major component of async-decode
            # model-submit time. Override via env. See P453665601.
            os.environ.setdefault("NEURON_RT_DISABLE_EXECUTION_BARRIER", "1")

            # GK decode: run the collectives' DMA queues at the highest udma QoS
            # service level (compute queues stay at the default 0). The decode
            # step's five collectives per layer are ~20 us/step shorter; the
            # data phases are unchanged. Override via env.
            if vllm_config.additional_config.get("neuron_config", {}).get("gk_decode"):
                os.environ.setdefault("NEURON_RT_DBG_CC_DMA_QOS_SERVICE_LEVEL", "3")

        elif (
            envs.VLLM_NEURON_CPU_COMPILE and "NEURON_VISIBLE_DEVICES" not in os.environ
        ):
            # CPU-compile leaves NEURON_VISIBLE_DEVICES unset, so every DP
            # replica gets an empty get_server_prefix() and names its per-rank
            # compile-cache dir identically ("<hash>/rank<N>") in the shared
            # tree — the replicas then race and clobber each other's rank dir,
            # failing with `neff_packager: failed to open output file`. Give
            # each replica a disjoint synthetic core window so its prefix is
            # distinct. Compile-cache namespacing only (no device is opened);
            # keyed on data_parallel_index, not data_parallel_rank (reset to 0
            # for dense DP). CPU-compile only — plain CPU_MODE is single-process
            # and setting the var would leak into pooled-worker tests.
            num_local_ranks = vllm_config.parallel_config.local_world_size
            dp_index = vllm_config.parallel_config.data_parallel_index
            first = dp_index * num_local_ranks
            local_ranks = range(first, first + num_local_ranks)
            os.environ["NEURON_VISIBLE_DEVICES"] = ",".join(str(c) for c in local_ranks)

        # --- Distributed environment init -----------------------------------
        parallel_config = vllm_config.parallel_config

        # vllm/distributed/parallel_state.py init_distributed_environment
        # rendezvouses the flat cross-node _WORLD group on
        # parallel_config.master_addr when nnodes>1; that field defaults to
        # "127.0.0.1" and is only set by --master-addr. Surface a clear error
        # (rather than letting the secondary pod deadlock on loopback) if
        # --master-addr was omitted. master_addr and data_parallel_master_ip
        # are distinct in vLLM, so we don't auto-substitute.
        validate_cross_node_master_addr(
            nnodes=parallel_config.nnodes,
            master_addr=parallel_config.master_addr,
            data_parallel_master_ip=parallel_config.data_parallel_master_ip,
        )

        self._patch_in_same_node_as_function()

        logger.debug(
            f"Initializing distributed environment: world_size={parallel_config.world_size}, {rank=}, {local_rank=}"
        )

        from vllm_neuron.parallel.neuron_parallel_state import (
            init_neuron_distributed_environment,
        )
        from vllm_neuron.envs import get_dist_backend

        neuron_config_dict = vllm_config.additional_config.get("neuron_config", {})
        dist_backend = get_dist_backend()

        ods_config = neuron_config_dict.get("on_device_sampling_config", {})
        sampling_dp_degree = (
            int(ods_config.get("sampling_dp_degree", 1)) if ods_config else 1
        )

        # Resolve ep_degree (defaults to TP*DP under EP) and validate it
        # against the cluster-wide world. See resolve_ep_degree for the rules.
        ep_degree = resolve_ep_degree(
            ep_degree=neuron_config_dict.get("ep_degree", 1),
            world_size_across_dp=parallel_config.world_size_across_dp,
            enable_expert_parallel=parallel_config.enable_expert_parallel,
        )

        attention_dp_size = neuron_config_dict.get("attention_dp_size", 1)
        embedding_dp_size = neuron_config_dict.get("embedding_dp_size", 1)
        lm_head_dp_size = neuron_config_dict.get("lm_head_dp_size", 1)
        mlp_dp_size = neuron_config_dict.get("mlp_dp_size", 1)
        draft_mlp_dp_size = neuron_config_dict.get("draft_mlp_dp_size")

        encoder_config_dict = vllm_config.additional_config.get(
            "encoder_neuron_config", {}
        )
        from vllm_neuron.model.neuron_config import EncoderNeuronConfig

        encoder_neuron_config = EncoderNeuronConfig.from_dict(encoder_config_dict or {})
        encoder_tp_size, encoder_dp_size = encoder_neuron_config.resolve_tp_dp(
            parallel_config.world_size
        )

        init_neuron_distributed_environment(
            rank=rank,
            local_rank=local_rank,
            distributed_init_method=distributed_init_method,
            world_size=parallel_config.world_size,
            backend=dist_backend,
            tensor_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
            decode_context_parallel_size=parallel_config.decode_context_parallel_size,
            ep_degree=ep_degree,
            sampling_dp_degree=sampling_dp_degree,
            attention_dp_size=attention_dp_size,
            embedding_dp_size=embedding_dp_size,
            lm_head_dp_size=lm_head_dp_size,
            mlp_dp_size=mlp_dp_size,
            draft_mlp_dp_size=draft_mlp_dp_size,
            encoder_tp_size=encoder_tp_size,
            encoder_dp_size=encoder_dp_size,
            nnodes=parallel_config.nnodes,
            timeout=(
                timedelta(seconds=parallel_config.distributed_timeout_seconds)
                if parallel_config.distributed_timeout_seconds is not None
                else None
            ),
        )

        # Rendezvous CCOM bootstrap address via gloo (now that dist is up).
        # Pass the DP index so independent DP replicas (each its own gloo world)
        # derive distinct root endpoints from a pinned NEURON_RT_ROOT_COMM_ID.
        # Use data_parallel_index, NOT data_parallel_rank: vLLM resets
        # data_parallel_rank to 0 for dense (non-MoE) DP engines (they are
        # treated as independent DP=1 servers), whereas data_parallel_index
        # always reflects the true DP rank for both dense and MoE engines.
        rendezvous_ccom_bootstrap(
            data_parallel_index=parallel_config.data_parallel_index
        )

        # Initialize the Neuron runtime NOW — after rendezvous has set the
        # correct NEURON_RT_ROOT_COMM_ID but before NIXL/libfabric tries to
        # register device memory.  The runtime latches COMM_ID during
        # initialize() and never re-reads it, so the ordering is critical:
        #   1. gloo init  (above)
        #   2. rendezvous_ccom_bootstrap  →  sets unique COMM_ID per server
        #   3. runtime.initialize()       →  latches COMM_ID, maps HBM
        if self._use_neuron_device():
            _initialize_neuron_runtime_with_retry()

    @_add_headroom_guidance_to_hbm_errors
    def load_model(self) -> None:
        """
        Load the model on this worker.

        This initializes the model runner and loads the model.
        After loading, attempts to fetch custom prefill buckets from
        the model if available.
        """
        from vllm.config import set_current_vllm_config

        logger.info("Loading model on NeuronWorker")
        # Point TORCHINDUCTOR_CACHE_DIR at a per-configuration, per-rank
        # directory before the first torch.compile() runs in load_model().
        # Native backend only: the XLA/eager routes never compile through
        # TorchInductor, so the redirect would be inert there. CPU mode skips it
        # too, so the early platform/version probes stay off the runtime.
        if envs.is_native_backend() and not envs.VLLM_NEURON_CPU_MODE:
            self._configure_torchinductor_cache_dir()
        with set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model()
            self.model_runner.init_tensor_capture()
            self.model_runner.init_tensor_replacement()

    def _configure_torchinductor_cache_dir(self) -> None:
        """Isolate this worker's TorchInductor cache by configuration and rank.

        Runs before the first torch.compile() in load_model(). The model runner
        already holds the resolved NeuronConfig. Use self.rank for the rank leaf:
        it is unique per worker inside a replica, and dp_index disambiguates the
        replicas.
        """
        from vllm_neuron.vllm.inductor_cache import (
            configure_torchinductor_cache_dir,
            resolve_dp_index,
        )

        runner = self.model_runner
        configure_torchinductor_cache_dir(
            vllm_config=self.vllm_config,
            neuron_config=runner.neuron_config,
            dp_index=resolve_dp_index(self.vllm_config.parallel_config),
            tp_rank=self.rank,
            encoder_neuron_config=getattr(runner, "encoder_neuron_config", None),
        )

    def _save_compile_cache_artifacts(self) -> None:
        """Save this worker's torch compile-cache package after warmup.

        Call this one time, after the last warmup bucket. The gate is the native
        backend, not CPU mode, and active caching-precompile. The method is a
        no-op when the gate is not met.
        """
        from vllm_neuron.vllm.inductor_cache import (
            save_compile_cache_artifacts,
            should_save_compile_cache,
        )

        if should_save_compile_cache():
            save_compile_cache_artifacts()

    def _get_local_worker_count(self) -> int:
        """Return number of worker processes sharing host memory on this node."""
        return max(self.vllm_config.parallel_config.local_world_size, 1)

    def _query_host_runtime_memory(self) -> int:
        """Query host available memory and return this worker's fair-share bytes."""
        from vllm_neuron.utils.memory import get_available_memory_bytes

        host_available = get_available_memory_bytes()

        if host_available <= 0:
            raise RuntimeError(
                f"Non-positive host available memory detected: {host_available} bytes."
            )

        local_workers = self._get_local_worker_count()
        fair_share = host_available // local_workers
        if fair_share <= 0:
            raise RuntimeError(
                "Host memory fair-share per worker is non-positive. "
                f"host_available={host_available} bytes, local_workers={local_workers}."
            )

        logger.info(
            "Host memory: %.2f GiB available, %d local worker(s), %.2f GiB fair-share per worker",
            host_available / (1024**3),
            local_workers,
            fair_share / (1024**3),
        )
        return fair_share

    def _query_runtime_memory(self) -> int:
        """Query Neuron runtime for available HBM (raw bytes_free)."""
        _, bytes_free = self._query_runtime_memory_stats()
        return bytes_free

    def _query_runtime_memory_stats(self) -> tuple[int, int]:
        """Query Neuron runtime for used/free HBM bytes.

        The runtime reports the full HBM stack, and it counts only the memory that
        this process holds. At LNC 1 two logical cores share one stack, so both
        siblings get the same total and neither sees the memory of the other. A
        caller that sizes a per-core budget must scale the total, which
        ``_compute_kv_budget`` does.
        """
        runtime = torch.classes.neuron.Runtime()
        bytes_used, bytes_free = runtime.get_vnc_memory_stats()

        if bytes_used < 0 or bytes_free <= 0:
            raise RuntimeError(
                f"Neuron runtime reported unusable memory stats: "
                f"used={bytes_used}, free={bytes_free}"
            )

        logger.info(
            "Neuron HBM: %.2f GiB used, %.2f GiB free",
            bytes_used / (1024**3),
            bytes_free / (1024**3),
        )
        return bytes_used, bytes_free

    def _active_headroom_env(self) -> str:
        """Return the headroom variable shared by both backends and CPU.

        Returns:
            The name of the environment variable in effect.

        Examples:
            >>> worker._active_headroom_env()
            'VLLM_NEURON_RESERVED_HEADROOM_GIB'
        """
        return _HEADROOM_ENV

    def _get_reserved_headroom_bytes(self) -> tuple[int, str]:
        """Return the per-worker memory headroom and its source."""
        env_name = self._active_headroom_env()
        headroom_gib = envs.VLLM_NEURON_RESERVED_HEADROOM_GIB
        source = "explicit"

        if headroom_gib is None:
            raise RuntimeError(
                "Set an explicit headroom value to enable headroom sizing."
            )
        if not math.isfinite(headroom_gib) or headroom_gib < 0:
            raise RuntimeError(
                "The KV cache headroom must be a finite nonnegative value. "
                f"Set {env_name} to zero or more. Got {headroom_gib}."
            )

        return int(headroom_gib * (1024**3)), source

    def _get_kv_cap_fraction(self) -> float:
        """Return the validated fractional cap for default KV sizing."""
        fraction = envs.VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise RuntimeError(
                "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION must be in (0, 1]. "
                f"Got {fraction}."
            )
        return fraction

    def _determine_available_memory_cpu(self) -> int:
        """Compute the CPU-mode KV budget from available host memory.

        Host available memory already excludes loaded model weights and other
        current allocations, so only future-allocation headroom is subtracted.
        """
        host_memory_fair_share = self._query_host_runtime_memory()
        if envs.get_kv_headroom_env() is None:
            total_budget = int(
                host_memory_fair_share * self.cache_config.gpu_memory_utilization
            )
            cap_fraction = self._get_kv_cap_fraction()
            available = int(total_budget * cap_fraction)
            logger.info(
                "CPU mode KV cache budget: %.2f GiB host memory fair share "
                "with gpu_memory_utilization=%.2f and cap_fraction=%.2f = %.2f GiB",
                host_memory_fair_share / (1024**3),
                self.cache_config.gpu_memory_utilization,
                cap_fraction,
                available / (1024**3),
            )
            self._validate_min_kv_budget(
                available,
                memory_before_headroom_bytes=host_memory_fair_share,
                budget_formula=(
                    "host_memory_fair_share * gpu_memory_utilization * cap_fraction"
                ),
            )
            return available

        headroom_bytes, headroom_source = self._get_reserved_headroom_bytes()
        available = max(host_memory_fair_share - headroom_bytes, 0)

        logger.info(
            "CPU mode KV cache budget: %.2f GiB host memory fair share "
            "- %.2f GiB headroom (%s) = %.2f GiB",
            host_memory_fair_share / (1024**3),
            headroom_bytes / (1024**3),
            headroom_source,
            available / (1024**3),
        )

        self._validate_min_kv_budget(
            available,
            memory_before_headroom_bytes=host_memory_fair_share,
            budget_formula="host_memory_fair_share - headroom_bytes",
        )
        return available

    def _validate_min_kv_budget(
        self,
        available_bytes: int,
        memory_before_headroom_bytes: int,
        budget_formula: str,
    ) -> None:
        """Fail early when the computed KV budget is below the configured minimum."""
        min_kv_gib = envs.VLLM_NEURON_MIN_KV_BUDGET_GIB
        if min_kv_gib <= 0:
            return

        min_kv_bytes = int(min_kv_gib * (1024**3))
        if available_bytes >= min_kv_bytes:
            return

        if envs.get_kv_headroom_env() is None:
            raise RuntimeError(
                "Computed KV cache budget is below minimum threshold. "
                f"effective={available_bytes / (1024**3):.2f} GiB, "
                f"minimum={min_kv_gib:.2f} GiB.\n"
                f"KV budget = {budget_formula}.\n"
                "Increase --gpu-memory-utilization or "
                "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION, or lower "
                "VLLM_NEURON_MIN_KV_BUDGET_GIB."
            )

        max_headroom_gib = max(
            (memory_before_headroom_bytes - min_kv_bytes) / (1024**3),
            0,
        )
        raise RuntimeError(
            "Computed KV cache budget is below minimum threshold. "
            f"effective={available_bytes / (1024**3):.2f} GiB, "
            f"minimum={min_kv_gib:.2f} GiB.\n"
            f"KV budget = max({budget_formula}, 0).\n"
            f"Do not increase {self._active_headroom_env()}; increasing "
            "headroom makes the KV cache budget smaller.\n"
            f"Set {self._active_headroom_env()} to at most "
            f"{max_headroom_gib:.2f}, or lower "
            "VLLM_NEURON_MIN_KV_BUDGET_GIB."
        )

    def _compute_kv_budget(
        self,
        total_hbm_bytes: int,
        model_weight_bytes: int,
    ) -> int:
        """Compute the Neuron HBM KV budget.

        Scale HBM to the logical-core share. Default sizing applies memory
        utilization and the fractional cap. Explicit headroom selects reserve sizing.

        Both callers report the memory of a full HBM stack. At LNC 2 one logical
        core owns the stack. At LNC 1 two logical cores share it, so each must
        plan against half of the stack. Without this scaling, both siblings size
        a KV cache against the same HBM and can fail during model load.
        """
        cores_per_stack = get_logical_cores_per_hbm_stack()
        if cores_per_stack > 1:
            logger.info(
                "LNC-1 HBM budget: %d logical cores share one HBM stack, so scale "
                "the %.2f GiB stack total to %.2f GiB for this core.",
                cores_per_stack,
                total_hbm_bytes / (1024**3),
                total_hbm_bytes / cores_per_stack / (1024**3),
            )
        total_hbm_bytes = total_hbm_bytes // cores_per_stack

        if envs.get_kv_headroom_env() is None:
            total_budget = int(
                total_hbm_bytes * self.cache_config.gpu_memory_utilization
            )
            user_budget = max(total_budget - model_weight_bytes, 0)
            cap_fraction = self._get_kv_cap_fraction()
            heuristic_cap = int(total_budget * cap_fraction)
            available = max(min(user_budget, heuristic_cap), 0)
            logger.info(
                "KV cache fractional budget: %.2f GiB logical-core HBM "
                "with gpu_memory_utilization=%.2f, %.2f GiB model weights, "
                "and cap_fraction=%.2f = %.2f GiB",
                total_hbm_bytes / (1024**3),
                self.cache_config.gpu_memory_utilization,
                model_weight_bytes / (1024**3),
                cap_fraction,
                available / (1024**3),
            )
            self._validate_min_kv_budget(
                available,
                memory_before_headroom_bytes=max(
                    total_hbm_bytes - model_weight_bytes, 0
                ),
                budget_formula=(
                    "min(max(logical_core_hbm * gpu_memory_utilization "
                    "- model_weight_bytes, 0), "
                    "logical_core_hbm * gpu_memory_utilization * cap_fraction)"
                ),
            )
            return available

        memory_before_headroom = max(total_hbm_bytes - model_weight_bytes, 0)
        headroom_bytes, headroom_source = self._get_reserved_headroom_bytes()
        available = max(memory_before_headroom - headroom_bytes, 0)

        logger.info(
            "KV cache budget: %.2f GiB logical-core HBM - %.2f GiB model weights "
            "- %.2f GiB headroom (%s) = %.2f GiB",
            total_hbm_bytes / (1024**3),
            model_weight_bytes / (1024**3),
            headroom_bytes / (1024**3),
            headroom_source,
            available / (1024**3),
        )

        self._validate_min_kv_budget(
            available,
            memory_before_headroom_bytes=memory_before_headroom,
            budget_formula=("logical_core_hbm - model_weight_bytes - headroom_bytes"),
        )
        return available

    def _get_byte_used_from_model(self) -> int:
        """
        Use the model params and buffers to calculate the size of tensors
        allocated on device.
        """
        bytes_used_params = sum(p.nbytes for p in self.model_runner.model.parameters())
        bytes_used_buffers = sum(b.nbytes for b in self.model_runner.model.buffers())
        # Add spec decode model weights.
        if self.model_runner.drafter is not None:
            bytes_used_params += sum(
                p.nbytes for p in self.model_runner.drafter.model.parameters()
            )
            bytes_used_buffers += sum(
                b.nbytes for b in self.model_runner.drafter.model.buffers()
            )
        return bytes_used_params + bytes_used_buffers

    def _estimate_available_memory_neuron(self) -> int:
        """
        Estimate KV budget from static HBM size and model parameter bytes.
        This is used during CPU Compilation flow and computes bytes used
        based on the model params and buffer sizes.
        """
        from libtorch_neuronx_lite.compile.platform import get_total_available_memory

        total_hbm_bytes = int(get_total_available_memory() * 1024 * 1024 * 1024)
        model_weight_bytes = self._get_byte_used_from_model()
        return self._compute_kv_budget(total_hbm_bytes, model_weight_bytes)

    def _determine_available_memory_neuron(self) -> int:
        """
        Compute Neuron-mode KV memory budget using static HBM size and model size

        Uses model parameter and buffer sizes for bytes_used (same as the CPU
        compile estimate path) instead of runtime.get_vnc_memory_stats().
        This ensures the KV cache block count is identical between CPU compile
        and execution, which is required for compilation cache hits.

        Without this, mismatch happens because runtime.get_vnc_memory_stats()
        itself uses ~3MB of device memory which goes unaccounted in CPU compile.
        Ignoring this is safe as the total memory is in order of GBs.
        """
        runtime_bytes_used, bytes_free = self._query_runtime_memory_stats()
        total_hbm_bytes = runtime_bytes_used + bytes_free
        model_weight_bytes = self._get_byte_used_from_model()
        return self._compute_kv_budget(total_hbm_bytes, model_weight_bytes)

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """
        Determine the amount of memory available for KV cache.

        The default uses memory utilization and a fractional cap. An explicit
        headroom value selects the reserve policy. CPU inference uses each
        worker's host memory fair share. Neuron uses logical-core HBM and weights.

        Returns:
            int: Available memory in bytes.
        """
        if envs.VLLM_NEURON_CPU_COMPILE:
            available = self._estimate_available_memory_neuron()
        elif envs.VLLM_NEURON_CPU_MODE:
            available = self._determine_available_memory_cpu()
        else:
            available = self._determine_available_memory_neuron()
        return available

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """
        Initialize the KV cache with the given number of blocks.

        Args:
            num_gpu_blocks: Number of GPU blocks to allocate
            num_cpu_blocks: Number of CPU blocks to allocate
        """
        logger.debug(
            "Initializing cache with %s GPU blocks, %s CPU blocks",
            num_gpu_blocks,
            num_cpu_blocks,
        )
        # TODO: Determine if initialize_cache() needs implementation for Neuron
        # Why required: Clarify whether explicit cache initialization is needed or handled by vLLM Neuron
        # Current: Empty method implementation
        # Target: Either implement cache allocation or document that it's handled elsewhere

    @_add_headroom_guidance_to_hbm_errors
    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache from configuration object.

        This is the vLLM v1 API method that replaces initialize_cache.
        Required by vLLM v1 WorkerBase interface.

        Args:
            kv_cache_config: KV cache configuration object
        """
        logger.debug("Initializing KV cache from config (NeuronWorker)")

        # Initialize KV transfer connector before KV cache so register_kv_caches
        # can find the connector (has_kv_transfer_group() must be True).
        # Skip in CPU compile mode — no device memory to register.
        if not envs.VLLM_NEURON_CPU_COMPILE:
            ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
        # Delegate to model runner to initialize KV cache
        self.model_runner.initialize_kv_cache(kv_cache_config)

    @_add_headroom_guidance_to_hbm_errors
    # warmup and compilation: this is what compiles, so it sets the mode the lane compiled in
    @in_lane_grad_mode
    def compile_or_warm_up_model(self) -> CompilationTimes:
        """
        Compile or warm up the model for optimal performance.

        Pre-compiles the model for all prefill bucket sizes and decode batch sizes
        to ensure optimal performance on Neuron hardware. Each bucket/batch size
        triggers a separate compilation that is cached by the Neuron compiler.

        In disaggregated inference (DI) mode, only the relevant warmup phase
        runs: prefill-only servers skip decode warmup, and decode-only servers
        skip prefill warmup. Servers with kv_role="kv_both" run both.

        Set VLLM_NEURON_SKIP_PREFILL_WARMUP=1 to skip prefill compilation
        without requiring a kv-transfer-config (useful for decode-only profiling).
        Set VLLM_NEURON_SKIP_DECODE_WARMUP=1 to skip decode compilation
        without requiring a kv-transfer-config (useful for prefill-only profiling).

        This method calls the model directly without using SchedulerOutput or
        execute_model(), avoiding unnecessary side effects on internal state.

        Returns:
            CompilationTimes: Compilation/warmup time.


        """
        if self.model_runner is None or self.model_runner.model is None:
            logger.warning(
                "Model not loaded yet. Skipping warmup. "
                "This may result in compilation happening during first inference."
            )
            return CompilationTimes(language_model=0.0, encoder=0.0)

        # SyntheticNeuronModel: skip warmup (no compiled graphs needed)
        from vllm_neuron.model.synthetic import SyntheticNeuronModel

        if isinstance(self.model_runner.model, SyntheticNeuronModel):
            logger.info("SyntheticNeuronModel detected — skipping warmup")
            return CompilationTimes(language_model=0.0, encoder=0.0)

        # CPU eager mode: skip warmup (no NEFFs to compile, shapes are dynamic)
        if envs.VLLM_NEURON_CPU_MODE and self.vllm_config.model_config.enforce_eager:
            logger.info("CPU eager mode — skipping warmup")
            return CompilationTimes(language_model=0.0, encoder=0.0)

        # Determine which warmup phases to run based on kv_role.
        # kv_producer = prefill server → only prefill warmup
        # kv_consumer = decode server  → only decode warmup
        # kv_both / no DI             → both
        warmup_start = time.perf_counter()
        kv_cfg = getattr(self.vllm_config, "kv_transfer_config", None)
        # Construction-axis flags. A vision-only (VE)
        # server has no language model at all, so skip both LM warmup phases.
        mm_encoder_only = getattr(self.model_runner, "mm_encoder_only", False)
        mm_language_model_only = getattr(
            self.model_runner, "mm_language_model_only", False
        )
        skip_language_model_warmup = (
            mm_encoder_only or envs.VLLM_NEURON_SKIP_PREFILL_DECODE_WARMUP
        )
        skip_prefill_warmup = (
            skip_language_model_warmup
            or (
                kv_cfg is not None
                and kv_cfg.is_kv_consumer
                and not kv_cfg.is_kv_producer
            )
            or envs.VLLM_NEURON_SKIP_PREFILL_WARMUP
        )
        skip_decode_warmup = (
            skip_language_model_warmup
            or (
                kv_cfg is not None
                and kv_cfg.is_kv_producer
                and not kv_cfg.is_kv_consumer
            )
            or envs.VLLM_NEURON_SKIP_DECODE_WARMUP
            # Pooling models are prefill-only (no autoregressive
            # decode), so skip decode graph extraction + warmup entirely.
            or self.model_runner.is_pooling_model
        )

        # Point the compile-ledger writer at this engine's directory (all ranks)
        # before any graph is captured, so the graphs compiled below are recorded
        # for the manifest emitted after warmup. No writer when the feature is off.
        if getattr(self, "_manifest_writer", None) is not None:
            self._manifest_writer.arm_ledger()

        # === Graph extraction (capture HLO for prefill + decode + encoder) ===
        # All extracts run before any warmup so the parallel-trace fork
        # children inherit a clean NRT/torch_xla state.

        # Skip the encoder warmup when:
        #   - no encoder is configured; or
        #   - the SKIP_ENCODER_WARMUP lever is set; or
        #   - language-only (PD): the encoder is not built; or
        #   - a prefill-skipping server that is NOT a vision-only (VE) pool.
        # The last term reuses skip_prefill_warmup as a shortcut for "decode-only
        # server, which never runs the encoder". A VE also skips prefill
        # (it has no language model) but DOES run the encoder, so it must
        # be exempted -- hence `and not mm_encoder_only`.
        skip_encoder_warmup = (
            self.model_runner.encoder_neuron_config is None
            or envs.VLLM_NEURON_SKIP_ENCODER_WARMUP
            or mm_language_model_only
            or (skip_prefill_warmup and not mm_encoder_only)
        )
        has_capture_backend = (
            self.model_runner.capture_backend_model is not None
            or self.model_runner.encoder_capture_backend is not None
        )
        # The native route has no graph-capture sidecar: it precompiles the
        # torch.compile wrapper directly to StableHLO/NEFF. CPU mode has no
        # native backend to precompile against: ``get_compile_backend_name()``
        # and ``get_dist_backend()`` both return the XLA value in CPU mode, so
        # keep the same carve-out here. CPU mode then warms up in process, like
        # the XLA route does.
        native_compile_only = envs.is_native_backend() and not envs.VLLM_NEURON_CPU_MODE
        if has_capture_backend or native_compile_only:
            if skip_prefill_warmup:
                logger.info(
                    "Skipping prefill graph extraction (kv_role=%s, decode-only server)",
                    kv_cfg.kv_role if kv_cfg is not None else "kv_consumer",
                )
            if skip_decode_warmup:
                logger.info(
                    "Skipping decode graph extraction (kv_role=%s, prefill-only server)",
                    kv_cfg.kv_role if kv_cfg is not None else "kv_producer",
                )
            self._extract_graphs(
                skip_prefill=skip_prefill_warmup,
                skip_decode=skip_decode_warmup,
                skip_encoder=skip_encoder_warmup,
                native_compile_only=native_compile_only,
            )
            logger.info("Barrier: waiting for all ranks to finish graph extraction")
            tp_barrier()
            if not native_compile_only:
                # graph.hlo -> XLA compilation. Native fork workers have
                # already persisted their StableHLO-derived NEFFs.
                self.model_runner.parallel_compile()
            logger.info("Barrier: waiting for all ranks to finish compilation")
            tp_barrier()

        # Re-decorate the compiled model so a fresh wrapper loads the just-flushed
        # package and dispatches each warmup bucket, in place of a re-trace.
        # CPU-compile runs no warmup, so it has no use for the fresh wrapper. Its
        # package restore would also build a device executor on a host with none.
        if (
            native_compile_only
            and not envs.VLLM_NEURON_CPU_COMPILE
            and hasattr(self.model_runner, "redecorate_compiled_model")
        ):
            self.model_runner.redecorate_compiled_model()
            logger.info("Re-decorated the compiled model after the package flush.")

        # === Prefill warmup ===
        if skip_prefill_warmup:
            logger.info(
                "Skipping prefill warmup (kv_role=%s, this is a decode-only server)",
                kv_cfg.kv_role if kv_cfg is not None else "kv_consumer",
            )
        elif not envs.VLLM_NEURON_CPU_COMPILE:
            self._warmup_prefill()

        # === Decode warmup ===
        if skip_decode_warmup:
            logger.info(
                "Skipping decode warmup (kv_role=%s, this is a prefill-only server)",
                kv_cfg.kv_role if kv_cfg is not None else "kv_producer",
            )
        elif not envs.VLLM_NEURON_CPU_COMPILE:
            # Optional: wrap decode warmup in an NRT profile capture. Warmup
            # DRIVES the decode NEFF on-device (synthetic inputs), so this
            # reliably records the decode kernel's execution — unlike a
            # /start_profile window around `vllm bench serve`, whose async
            # request flow can close the window before decode runs on the
            # profiled rank (yields an empty NTFF: NeffNodes=0). Gated on
            # VLLM_NEURON_PROFILE_WARMUP_DECODE so normal runs are unaffected.
            _profile_warmup = (
                envs.VLLM_NEURON_PROFILE_WARMUP_DECODE and self._should_profile()
            )
            if _profile_warmup:
                logger.info(
                    "[GK] profiling decode warmup (NRT capture around _warmup_decode)"
                )
                self.profile(is_start=True)
            try:
                self._warmup_decode()
            finally:
                if _profile_warmup:
                    self.profile(is_start=False)

        # === Encoder warmup ===
        if skip_encoder_warmup:
            if self.model_runner.encoder_neuron_config is not None:
                logger.info(
                    "Skipping encoder warmup (kv_role=%s, this is a decode-only server)",
                    kv_cfg.kv_role if kv_cfg is not None else "kv_consumer",
                )
        elif not envs.VLLM_NEURON_CPU_COMPILE:
            self._warmup_encoder()

        # Enable tensor capture after warmup (skip warmup captures)
        if self.model_runner.neuron_config.tensor_capture:
            self.model_runner.enable_capture()
            logger.info("Tensor capture enabled after warmup")

        # Save the torch compile-cache package once, after the last warmup
        # bucket. Native backend and caching-precompile only. A later warm start
        # loads the package automatically, so no explicit load call runs.
        self._save_compile_cache_artifacts()

        num_batched_tokens_buckets = (
            self.model_runner.neuron_config.num_batched_tokens_buckets
        )
        num_seqs_buckets = self.model_runner.neuron_config.num_seqs_buckets
        logger.info("=" * 80)
        logger.info(
            "Model warmup completed: %s num_batched_tokens_buckets, %s num_seqs_buckets",
            len(num_batched_tokens_buckets),
            len(num_seqs_buckets) if num_seqs_buckets else 0,
        )
        logger.info("=" * 80)

        STARTUP_TIME.labels(
            model_name=self.vllm_config.model_config.model,
        ).set(time.perf_counter() - self._startup_start_time)

        # EPD EC: register the encoder-cache device buffer with NIXL and (producer)
        # start the handshake server. Done post-warmup, when the buffer is a
        # materialized HBM tensor with a stable data_ptr. Skipped in CPU-compile
        # mode (no device memory); a no-op when no EC connector is configured.
        if not envs.VLLM_NEURON_CPU_COMPILE:
            self.model_runner.maybe_register_ec_cache()

        if getattr(self, "_manifest_writer", None) is not None and self.local_rank == 0:
            self._manifest_writer.emit_and_cleanup(vllm_config=self.vllm_config)

        return CompilationTimes(
            language_model=time.perf_counter() - warmup_start, encoder=0.0
        )

    def _prefill_buckets(self) -> tuple[list[int], list[int]]:
        num_batched_tokens_buckets = (
            self.model_runner.neuron_config.num_batched_tokens_buckets
        )
        kv_segment_size_buckets = (
            self.model_runner.neuron_config.kv_segment_size_buckets
        )
        # Use [0] as a sentinel for non-segmented mode (cached_seq_len=0)
        effective_kv_buckets = (
            kv_segment_size_buckets if kv_segment_size_buckets else [0]
        )
        return num_batched_tokens_buckets, effective_kv_buckets

    def _build_trace_job(
        self,
        model_input,
        *,
        native_compile_only: bool,
        graph_type: "str | None" = None,
    ) -> tuple:
        """Pair the active runner's trace model with one input.

        Args:
            model_input: Runner-specific synthetic model input.
            native_compile_only: Whether to compile a native graph directly.
            graph_type: The graph type, for example ``"decode"``. The runner
                resolves its concrete compiler arguments here and rides them on
                the trace job's third element, so the fork child compiles the
                graph with its own arguments. ``None`` uses the base arguments.

        Returns:
            The trace callable and its keyword arguments, plus (XLA capture
            path only) the resolved compiler arguments as a third element.
        """
        if self.vllm_config.use_v2_model_runner:
            # MRv2 pairs the adapter ABI with a compiled adapter callable: the
            # NATIVE precompile job drives the serving adapter (_forward_adapter,
            # compiled with the native Lite backend), and the XLA capture job
            # drives capture_backend_model. Selecting the raw model here would
            # feed adapter-only kwargs (token_store, num_computed_tokens, ...) to
            # a callable that does not accept them.
            target_model = (
                self.model_runner._forward_adapter
                if native_compile_only
                else self.model_runner.capture_backend_model
            )
            if target_model is None:
                route = (
                    "native _forward_adapter"
                    if native_compile_only
                    else "XLA capture_backend_model"
                )
                raise RuntimeError(
                    f"MRv2 compile target is unavailable: {route} is None. "
                    "_compile_forward_adapter() must publish it before trace jobs "
                    f"are built (native_compile_only={native_compile_only}, "
                    "use_v2_model_runner=True)."
                )
            input_device = model_input.positions.device
            kwargs = self.model_runner._build_forward_adapter_kwargs(
                model_input,
                state_device=input_device if input_device.type == "meta" else None,
            )
        else:
            # MRv1 pairs the raw model kwargs with the compiled model (native) or
            # the capture wrapper (XLA); the runner has no forward adapter.
            target_model = (
                self.model_runner.model
                if native_compile_only
                else self.model_runner.capture_backend_model
            )
            kwargs = model_input
        # Both routes' parallel_trace open a scoped_compiler_args scope from the
        # third element, so the lane compiles the graph with the arguments the
        # in-process warmup scopes. A re-decorated wrapper serves the lane's NEFF,
        # so the two must agree.
        compiler_args = self.model_runner.resolve_graph_compiler_args(graph_type)
        return target_model, kwargs, compiler_args

    def _build_prefill_trace_jobs(
        self,
        buckets: list[tuple[int, int]],
        *,
        native_compile_only: bool = False,
    ) -> list[tuple]:
        """Build prefill jobs for XLA capture or native precompile.

        XLA-capture jobs use meta inputs and drive the capture sidecar to write
        ``graph.hlo``. Native jobs use the model runner's logical device (so
        Dynamo fakeifies inputs and parameters on the same device) and drive
        the real compiled model to persist a StableHLO-derived NEFF. Eagle
        (drafter) extraction stays sequential on the XLA path and is not built
        here.

        Buckets are emitted **largest-first** (longest-processing-time-
        first scheduling). The fork pool's wall-clock equals the slowest
        lane in each wave; LPT minimizes the long-pole by ensuring big
        jobs get scheduled first while small jobs fill in around them.
        """
        # Sort descending. Trace cost grows with bucket_size primarily
        # and kv_seg_size secondarily; tuple ordering captures both.
        device = (
            self.model_runner.device if native_compile_only else torch.device("meta")
        )

        jobs: list[tuple] = []
        for bucket_size, kv_seg_size in sorted(buckets, reverse=True):
            model_input = self.model_runner._build_prefill_synthetic_inputs(
                bucket_size, kv_seg_size, device=device
            )
            jobs.append(
                self._build_trace_job(
                    model_input,
                    native_compile_only=native_compile_only,
                )
            )
        return jobs

    def _build_decode_trace_jobs(
        self,
        targets: list[tuple[int, int]],
        *,
        native_compile_only: bool = False,
    ) -> list[tuple]:
        """Build decode jobs for XLA capture or native precompile.

        XLA-capture jobs use meta inputs and drive the capture sidecar; native
        jobs use the model runner's logical device (so Dynamo fakeifies inputs
        and parameters on the same device) and drive the real compiled model.
        ``native_compile_only`` selects which.

        For spec-decode configs the target model has both with-spec and
        no-spec decode shapes — both flow through the pool. The drafter's
        decode graph runs sequentially in the post-pool fallback.

        Targets are emitted **largest-first** (LPT) — see
        ``_build_prefill_trace_jobs`` for rationale.
        """
        if not targets:
            return []

        device = (
            self.model_runner.device if native_compile_only else torch.device("meta")
        )

        if self.vllm_config.use_v2_model_runner:
            jobs: list[tuple] = []
            for batch_size, _ctx_bucket in sorted(targets, reverse=True):
                model_input = self.model_runner._build_decode_synthetic_inputs(
                    batch_size,
                    device=device,
                )
                jobs.append(
                    self._build_trace_job(
                        model_input,
                        native_compile_only=native_compile_only,
                        graph_type="decode",
                    )
                )
            return jobs

        spec_decode_enabled = (
            self.model_runner.drafter is not None
            and self.model_runner.speculative_config is not None
        )
        # Sort descending. Decode cost grows with both batch_size and
        # ctx_bucket; tuple ordering captures both.
        targets = sorted(targets, reverse=True)
        jobs: list[tuple] = []
        for batch_size, ctx_bucket in targets:
            kwargs = self.model_runner._build_decode_synthetic_inputs(
                batch_size,
                256,
                spec_decode_enabled=spec_decode_enabled,
                ctx_bucket=ctx_bucket,
                device=device,
                compiled_graph_input=True,
            )
            jobs.append(
                self._build_trace_job(
                    kwargs,
                    native_compile_only=native_compile_only,
                    graph_type="decode",
                )
            )
            if spec_decode_enabled:
                # Target model decode WITHOUT spec decode — separate
                # cache key, different decode_token_threshold.
                kwargs_no_spec = self.model_runner._build_decode_synthetic_inputs(
                    batch_size,
                    256,
                    spec_decode_enabled=False,
                    decode_token_threshold=1,
                    ctx_bucket=ctx_bucket,
                    device=device,
                    compiled_graph_input=True,
                )
                jobs.append(
                    self._build_trace_job(
                        kwargs_no_spec,
                        native_compile_only=native_compile_only,
                        graph_type="decode",
                    )
                )
        return jobs

    def _unwrap_encoder_model(self) -> "SupportsEncoderWarmup | None":
        """Unwrap model wrappers and return it if it implements SupportsEncoderWarmup."""
        m = self.model_runner.model
        if type(m).__name__ == "OptimizedModule":
            m = m._orig_mod
        return m if isinstance(m, SupportsEncoderWarmup) else None

    # VISIONGRAD: one owner of the grad context for encoder work.
    #
    # The encoder tower is traced in a parallel-trace child and then called again by
    # warmup and by serving. A Dynamo guard encodes the grad mode as GLOBAL_STATE,
    # and encodes each input tensor's dispatch key set, which carries the autograd
    # and view keys. A trace in one mode therefore builds guards that reject a call
    # made in another mode. The installed graph is discarded and the frame is traced
    # again, with nothing reported.
    #
    # The context cannot wrap the fork pool, because _extract_graphs submits
    # prefill, decode and encoder jobs to ONE pool and the language-model warmups run
    # in the ambient mode. So the context travels with the encoder job instead.
    # parallel_trace documents a job as "(callable, kwargs)", so a plain callable
    # that enters the context is within its contract.
    @staticmethod
    @contextlib.contextmanager
    def _encoder_grad_context():
        """Own the grad mode for every encoder trace and every encoder call.

        no_grad rather than inference_mode. The two produce different dispatch
        key sets on a new tensor: inference_mode drops ADInplaceOrView, no_grad
        keeps it. Every installed encoder guard measured on this route expects
        ADInplaceOrView present, so no_grad is the mode the packaged graphs were
        traced in, and the model forward already declares no_grad. Choosing
        inference_mode here would invalidate those guards instead of matching
        them.
        """
        with torch.no_grad():
            yield

    @classmethod
    def _encoder_job_callable(cls, target_model):
        """Return a job callable that enters the encoder grad context itself."""

        def _call(**kwargs):
            with cls._encoder_grad_context():
                return target_model(**kwargs)

        return _call

    def _build_encoder_trace_jobs(
        self, *, native_compile_only: bool = False
    ) -> list[tuple]:
        """Build encoder jobs for XLA capture or native precompile.

        Native jobs invoke the separately compiled encoder tower that parent
        warmup uses, resolved through ``get_encoder()``; XLA-capture jobs use
        ``encoder_capture_backend``. Buckets are sorted largest-first (LPT
        scheduling).
        """
        if (
            not native_compile_only
            and self.model_runner.encoder_capture_backend is None
        ):
            return []
        encoder_neuron_config = self.model_runner.encoder_neuron_config
        if (
            encoder_neuron_config is None
            or encoder_neuron_config.num_encoder_input_buckets is None
        ):
            return []

        unwrapped = self._unwrap_encoder_model()
        if unwrapped is None:
            return []

        # get_encoder() gives None when the pool builds no encoder tower, for
        # example a language-only PD pool. A None target reaches the fork child
        # and fails there, so return no job instead.
        if native_compile_only and unwrapped.get_encoder() is None:
            return []

        target_model = (
            unwrapped.get_encoder()
            if native_compile_only
            else self.model_runner.encoder_capture_backend
        )
        # Resolve the "encoder" arguments once (-O3) and ride them on the job so
        # the fork child compiles the encoder graph with them (see _build_trace_job).
        encoder_compiler_args = self.model_runner.resolve_graph_compiler_args("encoder")
        device = (
            self.model_runner.device if native_compile_only else torch.device("meta")
        )
        cache = self.model_runner.encoder_cache
        jobs: list[tuple] = []
        for bucket in sorted(
            encoder_neuron_config.num_encoder_input_buckets, reverse=True
        ):
            kwargs = unwrapped.build_encoder_synthetic_inputs(
                bucket, encoder_neuron_config, device
            )
            num_blocks = (
                math.ceil(
                    math.ceil(
                        bucket / encoder_neuron_config.encoder_attention_block_size
                    )
                    / encoder_neuron_config.dp_size
                )
                * encoder_neuron_config.dp_size
            )
            kwargs["encoder_cache_buffer"] = (
                cache.buffer
                if native_compile_only
                else torch.empty(
                    cache.num_blocks,
                    cache.block_size,
                    cache.fat_dim,
                    dtype=cache.dtype,
                    device=device,
                )
            )
            # A model can give its own write_block_ids when one block does not
            # hold encoder_attention_block_size of its tokens.
            kwargs.setdefault(
                "write_block_ids",
                torch.zeros(num_blocks, dtype=torch.int64, device=device),
            )
            if native_compile_only:
                # VISIONGRAD: the grad mode a tensor is created in decides its
                # dispatch key set, and the guard checks that set, so build the
                # inputs in the mode the call uses. The XLA branch is untouched.
                with self._encoder_grad_context():
                    kwargs = unwrapped.build_encoder_synthetic_inputs(
                        bucket, encoder_neuron_config, device
                    )
                    kwargs["encoder_cache_buffer"] = cache.buffer
                    kwargs.setdefault(
                        "write_block_ids",
                        torch.zeros(num_blocks, dtype=torch.int64, device=device),
                    )
                jobs.append(
                    (
                        self._encoder_job_callable(target_model),
                        kwargs,
                        encoder_compiler_args,
                    )
                )
            else:
                jobs.append((target_model, kwargs, encoder_compiler_args))
        return jobs

    def _run_parallel_trace_jobs(
        self, jobs: list[tuple], *, native_compile_only: bool = False
    ) -> bool:
        """Send target-model trace jobs to the parallel-trace fork pool.

        Returns True if the pool ran, False if it was bypassed via
        ``VLLM_NEURON_DISABLE_PARALLEL_TRACE`` (caller falls back to
        the in-process sequential path).
        """
        from libtorch_neuronx_lite.compile.parallel_trace import parallel_trace

        from vllm_neuron.parallel.neuron_parallel_state import (
            spmd_rank_tensors_on_meta,
        )

        if envs.VLLM_NEURON_DISABLE_PARALLEL_TRACE:
            return False
        # native_compile_only exists only in the native lite wheel's
        # parallel_trace(). The XLA route runs on the base image's default lite,
        # whose older signature would raise TypeError on the kwarg, so only pass
        # it on the native path. The native path traces on the real device
        # (see build_prefill/build_decode_trace_jobs), so it doesn't hit the
        # meta/real-device rank-tensor collision the meta-trace path does.
        if native_compile_only:
            # The file-store rendezvous provides no port usable as a socket
            # namespace, so give the native parallel-trace pool a stable
            # per-engine compile-service socket (keyed by the executor parent
            # pid, shared by all of this engine's workers).
            os.environ.setdefault(
                "NEURON_COMPILE_SERVICE_SOCKET",
                f"/tmp/neuron_compile_service_{os.getuid()}_{os.getppid()}.sock",
            )
            parallel_trace(
                jobs=jobs,
                parent_rank=self.rank,
                native_compile_only=True,
            )
        else:
            # The fork-children meta-swap the model's params/buffers, but the SPMD
            # rank tensors live on the vLLM group objects (not the model), so they
            # stay on the real device and collide with the meta-traced graph. Swap
            # them to meta for the duration of the (synchronous) trace pool — forked
            # children inherit the meta tensors via copy-on-write; the parent's
            # originals are restored on exit so the real per-rank value is bound at
            # runtime.
            with spmd_rank_tensors_on_meta():
                parallel_trace(jobs=jobs, parent_rank=self.rank)
        return True

    def _prefill_compile_targets(self) -> list[tuple[int, int]]:
        """Enumerate (num_batched_tokens, kv_segment_size) pairs to compile.

        Cartesian product of the configured prefill buckets and KV
        segment buckets. Mirror of ``_decode_compile_targets`` — both
        return the flat tuple list the trace pool consumes.

        Q buckets the segmented kernel cannot tile (not a multiple of 128, see
        ``segmented_prefill_supports_q_len``) are compiled at
        ``kv_segment_size=0`` instead: such a bucket still needs a NEFF, just a
        full-prefill one. Emitting the segmented pair would fail warmup with
        ``[NCC_INKI016] Query seqlen N must be a multiple of 128`` before any
        request arrived. This mirrors the runtime gate in
        ``_build_attention_metadata``, which leaves ``kv_segment_size`` at 0 for
        exactly these steps — the two must agree or we would dispatch a shape
        that was never compiled.
        """
        num_batched_tokens_buckets, effective_kv_buckets = self._prefill_buckets()
        # Deduped via dict keys, not a set: collapsing several kv_seg_size
        # buckets onto the same full-prefill target is expected, but warmup
        # order must stay deterministic (it is what the "[i/N] Warming up for
        # prefill" logs report) and the return type mirrors the ordered list
        # ``_decode_compile_targets`` hands the trace pool.
        targets: dict[tuple[int, int], None] = {}
        for kv_seg_size in effective_kv_buckets:
            for bucket_size in num_batched_tokens_buckets:
                if kv_seg_size and not segmented_prefill_supports_q_len(bucket_size):
                    kv_seg_size_for_bucket = 0
                else:
                    kv_seg_size_for_bucket = kv_seg_size
                targets[(bucket_size, kv_seg_size_for_bucket)] = None
        return list(targets)

    def _extract_graphs(
        self,
        *,
        skip_prefill: bool,
        skip_decode: bool,
        skip_encoder: bool,
        native_compile_only: bool = False,
    ) -> None:
        """Extract all target-model HLOs (prefill + decode + encoder) in a
        single parallel-trace pass. On the native route the drafter graphs join
        the same pass; on the XLA route they are captured in this process
        afterwards.

        Combining encoder + prefill + decode into one fork pool is strictly better
        than three: it amortizes the per-child meta-swap (the dominant
        per-fork cost — ~3-4 GB / child) and gives the round-robin
        partitioner a bigger work pool to balance against, so the
        long-pole bucket has less idle time around it.

        Sequential fallback (when ``VLLM_NEURON_DISABLE_PARALLEL_TRACE``
        is set): runs extraction in the existing order, including drafter
        graph capture interleaved per-bucket.
        """
        prefill_buckets = [] if skip_prefill else self._prefill_compile_targets()
        decode_targets = [] if skip_decode else self._decode_compile_targets()

        encoder_jobs = (
            []
            if skip_encoder
            else self._build_encoder_trace_jobs(native_compile_only=native_compile_only)
        )

        if prefill_buckets:
            logger.info(
                "Extracting graphs for %d prefill buckets: %s",
                len(prefill_buckets),
                prefill_buckets,
            )
        if decode_targets:
            logger.info(
                "Extracting %d decode compile targets: %s",
                len(decode_targets),
                decode_targets,
            )

        if encoder_jobs:
            logger.info(
                "Extracting graphs for %d encoder buckets",
                len(encoder_jobs),
            )

        prefill_jobs = self._build_prefill_trace_jobs(
            prefill_buckets, native_compile_only=native_compile_only
        )
        decode_jobs = self._build_decode_trace_jobs(
            decode_targets, native_compile_only=native_compile_only
        )

        target_jobs = prefill_jobs + decode_jobs + encoder_jobs

        # Native only: without this the parent compiles every drafter graph
        # serially during warmup. The XLA route captures the drafter through
        # graph_extract, which needs the capture sidecar native does not have.
        if native_compile_only and self.model_runner.drafter is not None:
            target_jobs += self._build_drafter_trace_jobs(
                prefill_buckets, decode_targets
            )

        if self._run_parallel_trace_jobs(
            target_jobs, native_compile_only=native_compile_only
        ):
            # The native pool already compiled the drafter graphs. The XLA route
            # captures them here, sequentially against ``self.device``.
            if native_compile_only:
                return
            has_drafter = self.model_runner.drafter is not None
            if has_drafter and not skip_prefill:
                self._extract_prefill_drafter_graphs(prefill_buckets)
            if has_drafter and not skip_decode:
                self._extract_decode_drafter_graphs(decode_targets)
            return

        # Native fallback deliberately does no extraction here. Parent warmups
        # retrace each bucket, compile StableHLO sequentially, and execute on NRT.
        if native_compile_only:
            return

        # Legacy sequential fallback: in-process extraction (target + drafter
        # together) for each prefill bucket, then each decode target.
        if not skip_prefill:
            self._extract_prefill_graphs_sequential(prefill_buckets)
        if not skip_decode:
            self._extract_decode_graphs_sequential(decode_targets)
        if not skip_encoder:
            self._extract_encoder_graphs()

    def _extract_prefill_graphs_sequential(
        self, buckets: list[tuple[int, int]]
    ) -> None:
        """In-process prefill extraction for every bucket. Used when
        the parallel-trace pool is disabled (workers<=1).

        Delegates to ``model_runner.extract_prefill_graphs`` which
        captures both the target and drafter graphs together against
        ``self.device``."""
        total = len(buckets)
        for i, (bucket_size, kv_seg_size) in enumerate(buckets, 1):
            logger.info(
                "\n[%s/%s] Extracting for prefill: bucket_size=%s, kv_segment_size=%s",
                i,
                total,
                bucket_size,
                kv_seg_size,
            )
            try:
                self.model_runner.extract_prefill_graphs(bucket_size, kv_seg_size)
                logger.info(
                    "  Successfully extracted graph for prefill bucket %s "
                    "with kv_segment_size %s",
                    bucket_size,
                    kv_seg_size,
                )
            except Exception as e:
                logger.error(
                    "  Failed to extract graph for prefill bucket %s "
                    "with kv_segment_size %s: %s",
                    bucket_size,
                    kv_seg_size,
                    e,
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Model graph extraction failed for prefill bucket {bucket_size} "
                    f"with kv_segment_size {kv_seg_size}. Error: {e}"
                ) from e

    def _build_drafter_trace_jobs(
        self,
        prefill_buckets: list[tuple[int, int]],
        decode_targets: list[tuple[int, int]],
    ) -> list[tuple]:
        """Build one ``drafter.warmup`` job per prefill bucket and decode target.

        Each job passes the arguments the parent warmup would, largest first.
        """
        runner = self.model_runner
        args = []
        for bucket_size, kv_seg_size in sorted(prefill_buckets, reverse=True):
            inputs = runner._build_prefill_synthetic_inputs(bucket_size, kv_seg_size)
            args.append((bucket_size, 1, inputs["attn_metadata"]))
        # Each request carries the bonus token plus the speculative tokens.
        tokens_per_req = 1 + runner.speculative_config.num_speculative_tokens
        for batch_size, ctx_bucket in sorted(decode_targets, reverse=True):
            num_tokens = batch_size * tokens_per_req
            attn_metadata = runner._build_warmup_attention_metadata(
                num_tokens=num_tokens, num_reqs=batch_size, ctx_bucket=ctx_bucket
            )
            args.append((num_tokens, batch_size, attn_metadata))
        warmup = runner.drafter.warmup
        return [
            (warmup, {"num_tokens": n, "num_reqs": r, "attn_metadata": m})
            for n, r, m in args
        ]

    def _extract_prefill_drafter_graphs(self, buckets: list[tuple[int, int]]) -> None:
        """Sequentially extract the eagle drafter prefill graphs.

        Runs in the parent process after the target-model fork pool
        completes, on ``self.device``. Caller must ensure a drafter is
        attached.
        """
        if self.model_runner.drafter is None:
            raise RuntimeError(
                "_extract_prefill_drafter_graphs called without a drafter "
                "attached; gate the call on self.model_runner.drafter."
            )
        for bucket_size, kv_seg_size in buckets:
            kwargs = self.model_runner._build_prefill_synthetic_inputs(
                bucket_size, kv_seg_size
            )
            logger.info(
                "Capturing EAGLE3 prefill graphs for bucket size: %d", bucket_size
            )
            self.model_runner.drafter.graph_extract(
                num_tokens=bucket_size,
                num_reqs=1,
                attn_metadata=kwargs["attn_metadata"],
            )

    def _extract_encoder_graphs(self) -> None:
        """Extract encoder HLO graphs.

        Drives the capture backend directly with synthetic per-bucket inputs.
        """
        from libtorch_neuronx_lite.compile.capture_backend import CaptureComplete
        from libtorch_neuronx_lite.compile.compiler_args import scoped_compiler_args

        encoder_neuron_config = self.model_runner.encoder_neuron_config
        unwrapped = self._unwrap_encoder_model()
        if unwrapped is None:
            return

        sorted_buckets = sorted(encoder_neuron_config.num_encoder_input_buckets)
        logger.info(
            "Extracting encoder graphs for %d buckets: %s",
            len(sorted_buckets),
            sorted_buckets,
        )

        capture_backend = self.model_runner.encoder_capture_backend
        meta = torch.device("meta")
        for i, bucket in enumerate(sorted_buckets, 1):
            logger.info(
                "  [%s/%s] Extracting encoder graph: bucket=%s",
                i,
                len(sorted_buckets),
                bucket,
            )
            encoder_inputs = unwrapped.build_encoder_synthetic_inputs(
                bucket, encoder_neuron_config, meta
            )
            num_blocks = (
                math.ceil(
                    math.ceil(
                        bucket / encoder_neuron_config.encoder_attention_block_size
                    )
                    / encoder_neuron_config.dp_size
                )
                * encoder_neuron_config.dp_size
            )
            cache = self.model_runner.encoder_cache
            encoder_inputs["encoder_cache_buffer"] = torch.empty(
                cache.num_blocks,
                cache.block_size,
                cache.fat_dim,
                dtype=cache.dtype,
                device=meta,
            )
            # A model can give its own write_block_ids when one block does not
            # hold encoder_attention_block_size of its tokens.
            encoder_inputs.setdefault(
                "write_block_ids",
                torch.zeros(num_blocks, dtype=torch.int64, device=meta),
            )
            try:
                # Scope the "encoder" compiler args, so the capture records -O3.
                with scoped_compiler_args(
                    self.model_runner.resolve_graph_compiler_args("encoder")
                ):
                    capture_backend(**encoder_inputs)
            except CaptureComplete:
                pass
            logger.info(
                "  Successfully extracted encoder graph for bucket %s",
                bucket,
            )

    def _warmup_prefill(self) -> None:
        """Run prefill warmup for all bucket and KV segment size combinations."""
        num_batched_tokens_buckets, effective_kv_buckets = self._prefill_buckets()

        logger.info(
            "Warming up model for %s num_batched_tokens_buckets: %s",
            len(num_batched_tokens_buckets),
            num_batched_tokens_buckets,
        )

        total_warmups = len(effective_kv_buckets) * len(num_batched_tokens_buckets)
        warmup_count = 0

        for kv_seg_size in effective_kv_buckets:
            for bucket_size in num_batched_tokens_buckets:
                warmup_count += 1
                logger.info(
                    "\n[%s/%s] Warming up for prefill: bucket_size=%s, "
                    "kv_segment_size=%s",
                    warmup_count,
                    total_warmups,
                    bucket_size,
                    kv_seg_size,
                )
                try:
                    self.model_runner.warmup_prefill(bucket_size, kv_seg_size)
                    logger.info(
                        "  Successfully warmed up for prefill bucket %s "
                        "with kv_segment_size %s",
                        bucket_size,
                        kv_seg_size,
                    )
                except Exception as e:
                    logger.error(
                        "  Failed to warmup for prefill bucket %s "
                        "with kv_segment_size %s: %s",
                        bucket_size,
                        kv_seg_size,
                        e,
                        exc_info=True,
                    )
                    raise RuntimeError(
                        f"Model warmup failed for prefill bucket {bucket_size} "
                        f"with kv_segment_size {kv_seg_size}. Error: {e}"
                    ) from e

    def _decode_compile_targets(self) -> list[tuple[int, int]]:
        """Enumerate (batch_bucket, ctx_bucket) pairs to compile.

        Without ``decode_context_length_buckets``: one pair per batch bucket,
        seq=max_model_len (today's behavior).

        With ``decode_context_length_buckets`` set: cartesian product of batch
        buckets × (configured seq buckets + max_model_len fallback).

        Returns:
            List of (batch_bucket, ctx_bucket) tuples. Empty when
            ``num_seqs_buckets`` is None.
        """
        num_seqs_buckets = self.model_runner.neuron_config.num_seqs_buckets
        if not num_seqs_buckets:
            return []

        decode_ctx_buckets = (
            self.model_runner.neuron_config.decode_context_length_buckets
        )
        max_model_len = self.model_runner.max_model_len

        if decode_ctx_buckets is None:
            return [(b, max_model_len) for b in num_seqs_buckets]

        ctx_buckets_with_fallback = list(decode_ctx_buckets) + [max_model_len]
        return [(b, s) for b in num_seqs_buckets for s in ctx_buckets_with_fallback]

    def _extract_decode_graphs_sequential(self, targets: list[tuple[int, int]]) -> None:
        """In-process decode extraction for every (batch, seq) compile
        target. Used when the parallel-trace pool is disabled.

        Delegates to ``model_runner.extract_decode_graphs`` which
        captures both target and drafter graphs together against
        ``self.device``."""
        for i, (batch_size, ctx_bucket) in enumerate(targets, 1):
            logger.info(
                "\n[%s/%s] Extracting graph for decode (batch=%s, seq=%s)",
                i,
                len(targets),
                batch_size,
                ctx_bucket,
            )
            try:
                self.model_runner.extract_decode_graphs(
                    batch_size, ctx_bucket=ctx_bucket
                )
                logger.info(
                    "  Extracted graph for decode (batch=%s, seq=%s)",
                    batch_size,
                    ctx_bucket,
                )
            except Exception as e:
                logger.error(
                    "  Failed to extract graph for decode (batch=%s, seq=%s): %s",
                    batch_size,
                    ctx_bucket,
                    e,
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Model graph extraction failed for decode "
                    f"(batch={batch_size}, seq={ctx_bucket}). Error: {e}"
                ) from e

    def _extract_decode_drafter_graphs(self, targets: list[tuple[int, int]]) -> None:
        """Sequentially extract eagle drafter decode graphs.

        Runs in the parent process after the target-model fork pool, on
        ``self.device``. Caller must ensure a drafter is attached.

        TODO: migrate the drafter to the parallel-trace fork pool. Eagle
        wraps ``propose()`` rather than calling ``capture_backend_model``
        directly, so it doesn't fit the ``(model, kwargs)`` primitive
        without an adapter that exposes ``_orig_mod`` for the meta swap.
        """
        runner = self.model_runner
        if runner.drafter is None or runner.speculative_config is None:
            raise RuntimeError(
                "_extract_decode_drafter_graphs called without a drafter "
                "attached; gate the call on self.model_runner.drafter."
            )
        num_spec_tokens = runner.speculative_config.num_speculative_tokens
        for batch_size, ctx_bucket in targets:
            draft_num_tokens = batch_size * (1 + num_spec_tokens)
            draft_attn_metadata = runner._build_warmup_attention_metadata(
                num_tokens=draft_num_tokens,
                num_reqs=batch_size,
                ctx_bucket=ctx_bucket,
            )
            logger.info("Capturing EAGLE3 decode graphs for batch size: %d", batch_size)
            runner.drafter.graph_extract(
                num_tokens=draft_num_tokens,
                num_reqs=batch_size,
                attn_metadata=draft_attn_metadata,
            )

    def _warmup_decode(self) -> None:
        """Run decode warmup for all (batch, seq) compile targets."""
        targets = self._decode_compile_targets()
        if not targets:
            return
        logger.info(
            "\nWarming up %d decode compile targets: %s",
            len(targets),
            targets,
        )

        for i, (batch_size, ctx_bucket) in enumerate(targets, 1):
            logger.info(
                "\n[%s/%s] Warming up decode (batch=%s, seq=%s)",
                i,
                len(targets),
                batch_size,
                ctx_bucket,
            )
            try:
                self.model_runner.warmup_decode(batch_size, ctx_bucket=ctx_bucket)
                logger.info(
                    "  Warmed up decode (batch=%s, seq=%s)",
                    batch_size,
                    ctx_bucket,
                )
            except Exception as e:
                logger.error(
                    "  Failed to warmup decode (batch=%s, seq=%s): %s",
                    batch_size,
                    ctx_bucket,
                    e,
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Model warmup failed for decode "
                    f"(batch={batch_size}, seq={ctx_bucket}). Error: {e}"
                ) from e

    # VISIONGRAD: the grad mode comes from _encoder_grad_context, which the trace
    # path uses too, so the two cannot drift apart.
    def _warmup_encoder(self) -> None:
        """Warm up the encoder by loading NEFFs for each bucket.

        Drives the compiled encoder directly with synthetic per-bucket inputs.
        """
        from libtorch_neuronx_lite.compile.compiler_args import scoped_compiler_args

        encoder_neuron_config = self.model_runner.encoder_neuron_config
        unwrapped = self._unwrap_encoder_model()
        has_encoder_warmup = unwrapped is not None

        if encoder_neuron_config.num_encoder_input_buckets is None:
            if has_encoder_warmup:
                raise ValueError(
                    "Model supports an encoder but num_encoder_input_buckets is not "
                    "configured. The encoder requires bucket configuration "
                    "for Neuron compilation."
                )
            logger.info("No encoder token buckets configured, skipping encoder warmup")
            return

        if not has_encoder_warmup:
            logger.info("Model does not support encoder warmup, skipping")
            return

        # get_encoder() gives None when the pool builds no encoder tower, for
        # example a language-only PD pool. Return here, because each site below
        # calls the tower.
        if unwrapped.get_encoder() is None:
            logger.info("Model builds no encoder tower, skipping encoder warmup")
            return

        sorted_buckets = sorted(encoder_neuron_config.num_encoder_input_buckets)
        logger.info(
            "Warming up the encoder for %d buckets: %s",
            len(sorted_buckets),
            sorted_buckets,
        )

        device = next(unwrapped.get_encoder().parameters()).device
        for i, bucket in enumerate(sorted_buckets):
            num_blocks = math.ceil(
                bucket / encoder_neuron_config.encoder_attention_block_size
            )
            # Pad num_blocks to be divisible by dp_size for even DP scatter
            if encoder_neuron_config.dp_size > 1:
                num_blocks = (
                    math.ceil(num_blocks / encoder_neuron_config.dp_size)
                    * encoder_neuron_config.dp_size
                )
            logger.info(
                "  Encoder warmup [%d/%d]: bucket=%d, num_blocks=%d",
                i + 1,
                len(sorted_buckets),
                bucket,
                num_blocks,
            )

            with self._encoder_grad_context():
                encoder_inputs = unwrapped.build_encoder_synthetic_inputs(
                    bucket, encoder_neuron_config, device
                )
            try:
                encoder_inputs["encoder_cache_buffer"] = (
                    self.model_runner.encoder_cache.buffer
                )
                # A model can give its own write_block_ids when one block does not
                # hold encoder_attention_block_size of its tokens.
                encoder_inputs.setdefault(
                    "write_block_ids",
                    torch.zeros(num_blocks, dtype=torch.int64, device=device),
                )
                # Scope the "encoder" compiler args, so warmup loads the -O3 NEFF.
                with (
                    self._encoder_grad_context(),
                    scoped_compiler_args(
                        self.model_runner.resolve_graph_compiler_args("encoder")
                    ),
                ):
                    _ = unwrapped.get_encoder()(**encoder_inputs)
                logger.info("  Successfully warmed up encoder bucket %d", bucket)
            except Exception as e:
                logger.error(
                    "  Failed to warmup encoder bucket %d: %s",
                    bucket,
                    e,
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Encoder warmup failed for bucket {bucket}. Error: {e}"
                ) from e

        logger.info("Encoder warmup complete")

    # serving: must match what warmup compiled
    @in_lane_grad_mode
    def execute_model(
        self,
        scheduler_output: Any,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """
        Execute the model with the given scheduler output.

        If this method returns None, sample_tokens should be called immediately after
        to obtain the ModelRunnerOutput.

        NOTE (vllm) Note that this design may be changed in future if/when structured outputs
        parallelism is re-architected.

        Args:
            scheduler_output: Output from the vLLM scheduler

        Returns:
            - ModelRunnerOutput
            - or AsyncModelRunnerOutput
            - or None
        """
        if self._profiler is not None:
            self._profiler.step()

        logger.debug("Executing model on NeuronWorker")
        output = self.model_runner.execute_model(scheduler_output)

        return output

    def _create_profiler(self) -> WorkerProfiler:
        """Create the appropriate profiler based on config.

        Called lazily on first /start_profile request.

        Raises:
            ValueError: If profiler type is not recognized.
        """
        profiler_config = self.vllm_config.profiler_config

        if profiler_config.profiler == "cuda":
            neuron_profiler_config = NeuronProfilerConfig(
                self.vllm_config.additional_config.get("neuron_profiler")
            )
            return NeuronProfiler(profiler_config, neuron_profiler_config)
        elif profiler_config.profiler == "torch":
            worker_name = f"neuron-rank-{self.rank}"
            return TorchProfilerWrapper(
                profiler_config,
                worker_name=worker_name,
                local_rank=self.local_rank,
                activities=["CPU"],
            )
        else:
            raise ValueError(
                f"Unknown profiler type: '{profiler_config.profiler}'. "
                "Supported types: 'cuda', 'torch'."
            )

    def _should_profile(self) -> bool:
        """Return True if this worker should participate in profiling."""
        profiler_config = self.vllm_config.profiler_config
        if profiler_config is None or profiler_config.profiler is None:
            return False

        if profiler_config.profiler == "cuda":
            neuron_profiler_config = NeuronProfilerConfig(
                self.vllm_config.additional_config.get("neuron_profiler")
            )
            neuron_cores = neuron_profiler_config.neuron_cores
            if neuron_cores is not None:
                return self.local_rank in neuron_cores
            return self.local_rank == 0

        return True

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        """Toggle profiling on/off via /start_profile and /stop_profile.

        Supports two profiler types:
        - "cuda": NeuronProfiler for device/system profiling (NRT inspect APIs)
        - "torch": TorchProfilerWrapper for CPU-side torch profiler spans

        The profiler is created lazily on the first start call.
        """
        if not self._should_profile():
            return

        if is_start:
            if self._profiler is None:
                self._profiler = self._create_profiler()
            self._profiler.start()
        else:
            if self._profiler is None:
                logger.warning("Profiler was not started, nothing to stop.")
                return
            self._profiler.stop()

    # serving phase 2, same engine step as execute_model
    @in_lane_grad_mode
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Get KV cache specifications for all model layers.

        This method is required by vLLM v1 WorkerBase interface.

        Returns:
            Dictionary mapping layer names to their KV cache specifications
        """
        logger.debug("Getting KV cache spec from NeuronWorker")
        return self.model_runner.get_kv_cache_spec()

    def get_supported_kv_cache_layouts(self) -> list[str]:
        """Pin the KV cache layout to layer-outermost.

        Every Neuron KV init path (per-layer, MRv2, GK stacking) reinterprets a
        layer's region as one contiguous ``(2, num_blocks, ...)`` view, which only a
        layer-outermost layout provides. The base implementation would derive this
        from the attention backends, and Neuron models do not populate
        ``static_forward_context``, so it would fall through to the selector default.
        """
        return ["LBNHC"]

    # draft readback, covered for uniformity at the boundary
    @in_lane_grad_mode
    def take_draft_token_ids(self) -> DraftTokenIds | None:
        """
        Get draft token ids from the model runner.

        This method is required by vLLM v1 WorkerBase interface.
        Delegates to the model runner to get draft token ids.

        Returns:
            DraftTokenIds or None if not available
        """
        logger.debug("Getting draft token ids from NeuronWorker")
        return self.model_runner.take_draft_token_ids()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """
        Get the tasks supported by this worker.

        This method is required by vLLM v1 WorkerBase interface.
        Delegates to the model runner to determine supported tasks.

        Returns:
            Tuple of supported tasks
        """
        logger.debug("Getting supported tasks from NeuronWorker")
        return self.model_runner.get_supported_tasks()

    # idle-rank dispatch, inside the fail_on_recompile stance
    @in_lane_grad_mode
    def execute_dummy_batch(self) -> None:
        """
        Execute a dummy batch for Data Parallel synchronization.

        This method is called by the vLLM engine when Data Parallel workload
        imbalance is detected (i.e., this rank has no real requests but other
        DP ranks do). It runs a minimal dummy forward pass to keep all ranks
        synchronized during collective operations.

        The implementation simply delegates to the model runner's
        execute_dummy_batch method, which handles the actual dummy execution.
        """
        logger.debug("Executing dummy batch on NeuronWorker")
        self.model_runner.execute_dummy_batch()

    def shutdown(self) -> None:
        """Clean up resources held by the worker."""
        if self._profiler is not None:
            self._profiler.shutdown()

        # Disable tensor capture to prevent writes during teardown
        if self.model_runner is not None:
            self.model_runner.disable_capture()

        self.model_runner.ensure_kv_transfer_shutdown()

        # This triggers an unexpected state error. Lets try avoiding unsafe_close()
        #
        # if not envs.VLLM_NEURON_CPU_MODE:
        #     # force the neuron profiles to dump the profiles
        #     torch.classes.neuron.Runtime().unsafe_close()

    def get_kv_caches(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Extract KV cache tensors from this worker's model (CPU tensors for serialization).

        Example:
            >>> kv = worker.get_kv_caches()
        """
        return self.model_runner.get_kv_caches()

    def get_block_table_info(self) -> dict:
        """Extract block table and sequence info for KV cache reconstruction.

        Example:
            >>> info = worker.get_block_table_info()
        """
        return self.model_runner.get_block_table_info()

    def get_kv_cache_config(self) -> dict:
        """Return KVCacheConfig for model-agnostic KV reconstruction.

        Example:
            >>> cfg = worker.get_kv_cache_config()
        """
        return self.model_runner.get_kv_cache_config()

    def enable_kv_snapshot(self) -> None:
        """Enable block-table snapshotting on subsequent forward passes.

        Example:
            >>> worker.enable_kv_snapshot()
        """
        self.model_runner.enable_kv_snapshot()

    def get_block_tables(self) -> list[bytes]:
        """Return block tables for all KV cache groups.

        Example:
            >>> tables = worker.get_block_tables()
        """
        return self.model_runner.get_block_tables()

    def clear_kv_snapshot(self) -> None:
        """Release block table snapshot memory and disable snapshotting.

        Example:
            >>> worker.clear_kv_snapshot()
        """
        self.model_runner.clear_kv_snapshot()

    def get_encoder_cache(self) -> dict[str, bytes]:
        """Return encoder cache (multimodal embeddings) as serialized bytes.

        Example:
            >>> enc = worker.get_encoder_cache()
        """
        return self.model_runner.get_encoder_cache()

    def enable_encoder_cache_snapshot(self) -> None:
        """Enable encoder cache snapshotting (lightweight, no serialization).

        Example:
            >>> worker.enable_encoder_cache_snapshot()
        """
        self.model_runner.enable_encoder_cache_snapshot()

    def clear_encoder_cache_snapshot(self) -> None:
        """Release encoder cache snapshot memory and disable snapshotting.

        Example:
            >>> worker.clear_encoder_cache_snapshot()
        """
        self.model_runner.clear_encoder_cache_snapshot()

    def get_async_scheduling_stats(self) -> dict:
        """Return async scheduling step counters from the model runner.

        Returns a dict with ``async_steps`` and ``sync_fallback_steps``.
        Returns zeros when async scheduling is not enabled.

        Example:
            >>> stats = worker.get_async_scheduling_stats()
            >>> stats["async_steps"]
            42
        """
        if not getattr(self.model_runner, "use_async_scheduling", False):
            return {"async_steps": 0, "sync_fallback_steps": 0}
        return {
            "async_steps": self.model_runner._async_steps,
            "sync_fallback_steps": self.model_runner._sync_fallback_steps,
        }

    def _use_neuron_device(self) -> bool:
        """
        VLLM_NEURON_CPU_MODE forces cpu as the device with vLLM.
        VLLM_NEURON_CPU_COMPILE forces meta as the device type with vLLM.

        This check returns if neuron is the device type being used with vLLM.
        """
        return not envs.VLLM_NEURON_CPU_MODE and not envs.VLLM_NEURON_CPU_COMPILE
