# SPDX-License-Identifier: Apache-2.0
"""Multi-process executor for distributed Neuron model inference.

This module provides MPExecutor, which manages multiple worker processes
for running distributed models across Neuron cores. Each process handles
a separate Neuron core and communicates via queues.

This module is intended for use in integration tests (such as module tests).
To run models, use standard vLLM interfaces for offline/online inference.

Typical usage:
    1. Define a model loading function that creates and compiles your model
    2. Create MPExecutor with desired world_size
    3. Use dispatch() to send inputs and collect() to get results
    4. Call shutdown() when done

CPU Mode:
    Pass device="cpu" to MPExecutor to run on CPU instead of Neuron devices.
    This is useful for testing and debugging without Neuron hardware.
"""

import logging
import os
import time
import traceback
from queue import Empty

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import ml_dtypes

import vllm.config  # noqa: F401 — Pre-load so forked children inherit it
import vllm.v1.executor  # noqa: F401 — Eager-load scipy chain (~570ms)
import vllm.v1.executor.ray_utils  # noqa: F401 — Eager-load ray chain (~310ms)

# Pin fork: py3.14 defaults to forkserver on Linux, but MPExecutor requires
# fork semantics (module pre-loading, fork-after-NRT-init diagnostics).
mp = mp.get_context("fork")

from vllm_neuron import envs
from vllm_neuron.parallel.neuron_parallel_state import stamp_spmd_rank_tensors

logger = logging.getLogger(__name__)

# fp8 dtypes require ml_dtypes for np conversion since np has no native fp8.
_FP8_NUMPY_DTYPES = {
    torch.float8_e5m2: ml_dtypes.float8_e5m2,
    torch.float8_e4m3fn: ml_dtypes.float8_e4m3fn,
}


# Port 29500 is the conventional PyTorch MASTER_PORT default used by most distributed
# examples (including helloworld_dist.py). We must avoid it because _WorkerPool
# persists across tests and holds a TCPStore on its assigned port — if we used 29500,
# any test that spawns its own process group on the default port would get EADDRINUSE.
# Port 61234 + core_id is reserved for NEURON_RT_ROOT_COMM_ID.
#
# VLLM_NEURON_MPEXECUTOR_PORT_OFFSET separates the base ports per concurrent
# MPExecutor group. Each derived port is base + offset + core_id. The offset
# must keep the highest root-comm port inside the valid TCP range. Reject an
# offset that moves a port out of range.
_MP_EXECUTOR_PORT_OFFSET = envs.VLLM_NEURON_MPEXECUTOR_PORT_OFFSET
_MAX_CORE_ID = 128
_MAX_PORT_OFFSET = 65535 - 61234 - _MAX_CORE_ID
if not 0 <= _MP_EXECUTOR_PORT_OFFSET <= _MAX_PORT_OFFSET:
    raise ValueError(
        f"VLLM_NEURON_MPEXECUTOR_PORT_OFFSET={_MP_EXECUTOR_PORT_OFFSET} moves a "
        f"port out of the valid TCP range. Use a value from 0 to "
        f"{_MAX_PORT_OFFSET}."
    )
_BASE_COLLECTIVES_PORT = 48920 + _MP_EXECUTOR_PORT_OFFSET
_BASE_ROOT_COMM_PORT = 61234 + _MP_EXECUTOR_PORT_OFFSET


def _get_unique_root_comm_id(core_id: int) -> str:
    """Derive the CCOM bootstrap endpoint from the first core ID of the allocation.

    Keyed on the group's first core so every worker in the group agrees on one
    endpoint and separate allocations get separate ports.

    Args:
        core_id: The first logical core ID of the allocation.

    Returns:
        str: A ``"host:port"`` CCOM root address.

    Example:
        >>> _get_unique_root_comm_id(0)
        '127.0.0.1:61234'
        >>> _get_unique_root_comm_id(8)
        '127.0.0.1:61242'
    """
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    return f"{master_addr}:{_BASE_ROOT_COMM_PORT + core_id}"


def _get_unique_collectives_port(core_id: int) -> str:
    """Derive a deterministic port for distributed communication from the core ID.

    Uses base port + first core ID of the allocation. This avoids the race
    condition inherent in ephemeral port discovery and ensures parallel tests
    on different cores don't conflict.

    Note: Port 61234 + core_id is reserved for NEURON_RT_ROOT_COMM_ID.
    We use 48920 + core_id to avoid colliding with the conventional
    MASTER_PORT default of 29500 used by PyTorch examples.

    Args:
        core_id: The first logical core ID of the allocation.

    Returns:
        str: A unique port number as a string.

    Example:
        >>> _get_unique_collectives_port(0)
        '48920'
        >>> _get_unique_collectives_port(8)
        '48928'
    """
    return str(_BASE_COLLECTIVES_PORT + core_id)


def _is_picklable(obj):
    """Check if an object can be pickled (required for mp.Queue)."""
    import pickle

    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, AttributeError, TypeError):
        return False


class _ReinitFailed(Exception):
    """Raised when REINIT_PARALLEL fails in any worker (error or timeout).

    Signals _WorkerPool.acquire() to fall back to respawning workers rather
    than propagating the failure to the caller. After this is raised the
    pool is in an unknown state and callers must tear it down.
    """


# Teardown budgets for the pooled worker group. The Neuron core is free only
# after the worker process exits. Each budget belongs to the whole group, not to
# one worker, so the teardown cost does not grow with the worker count.
_WORKER_STOP_TIMEOUT_S = 5.0
_WORKER_TERMINATE_TIMEOUT_S = 10.0
_WORKER_KILL_TIMEOUT_S = 10.0
_LOAD_TIMEOUT_S = 300.0


def _join_group(processes: list, timeout: float) -> list:
    """Join every process against one shared deadline. Return the survivors.

    The budget is shared, so a large worker group costs the same as a small one.
    A per-worker timeout gives a 64-worker group a 1600s worst case. An outer
    timeout can then kill the parent before the last worker gets a signal, and
    that leaks the Neuron cores which this teardown must free.

    Args:
        processes: The worker processes to wait for.
        timeout: The budget in seconds for the whole group.

    Returns:
        The processes that are still alive after the budget expires.

    Example:
        >>> survivors = _join_group(pool._processes, 5.0)
        >>> [p.pid for p in survivors]
        []
    """
    deadline = time.monotonic() + timeout
    for p in processes:
        p.join(timeout=max(0.0, deadline - time.monotonic()))
    return [p for p in processes if p.is_alive()]


# TODO(CHRS-746): Make WorkerPool threadsafe.
class _WorkerPool:
    """Module-level singleton that keeps at most one worker group alive.

    When a matching config is requested, existing workers are reused and
    the new model is loaded via a LOAD command. When the config differs,
    the old group is torn down first.

    The config key is (world_size, device). For example, consecutive
    test_collectives tests with world_size=2 and device="cpu" share the
    same key (2, "cpu").
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._key = None
            cls._instance._parallel_kwargs = None
            cls._instance._processes = []
            cls._instance._input_queues = []
            cls._instance._output_pipes = []
            cls._instance._master_port = None
            cls._instance._num_local_ranks = 0
        return cls._instance

    def acquire(
        self,
        world_size,
        model_load,
        device="neuron:0",
        nnodes=1,
        node_rank=0,
        ep_degree=1,
        sampling_dp_degree=1,
        dp_size=1,
        attention_dp_size=1,
        embedding_dp_size=1,
        lm_head_dp_size=1,
        mlp_dp_size=1,
        draft_mlp_dp_size=None,
        encoder_tp_size=1,
        encoder_dp_size=1,
        dcp_size=1,
        load_timeout=_LOAD_TIMEOUT_S,
    ):
        """Acquire workers. Reuses existing group if config matches, else respawns.

        Returns:
            tuple: (num_local_ranks, processes, input_queues, output_pipes)
        """
        num_local_ranks = world_size // nnodes
        # Hard key: params that determine process topology — must respawn.
        key = (world_size, device)
        # Soft key: Neuron sub-group params — can reinit in-place.
        parallel_kwargs = dict(
            ep_degree=ep_degree,
            dp_size=dp_size,
        )

        if self._key == key and self._processes:
            # Reuse existing workers — just load the new model
            # If model_load is not picklable (e.g. a closure), fall back to
            # respawning workers so it can be passed via mp.Process (fork).
            if _is_picklable(model_load):
                logger.info(
                    "WorkerPool: reusing %d workers for key %s",
                    num_local_ranks,
                    key,
                )
                self._drain_stale_data(num_local_ranks)
                try:
                    # Reinit Neuron groups if parallel config changed.
                    if parallel_kwargs != self._parallel_kwargs:
                        logger.info(
                            "WorkerPool: reinitializing parallel groups: %s",
                            parallel_kwargs,
                        )
                        for rank in range(num_local_ranks):
                            self._input_queues[rank].put(
                                ("REINIT_PARALLEL", parallel_kwargs)
                            )
                        self._wait_for_reinit(num_local_ranks)
                        self._parallel_kwargs = parallel_kwargs
                    if model_load is not None:
                        for rank in range(num_local_ranks):
                            self._input_queues[rank].put(("LOAD", model_load))
                        self._wait_for_load(num_local_ranks, load_timeout)
                    return (
                        num_local_ranks,
                        self._processes,
                        self._input_queues,
                        self._output_pipes,
                    )
                except (_ReinitFailed, RuntimeError, TimeoutError) as e:
                    # Workers are in an unknown state after a failed
                    # reinit or load. Tear them down and fall through
                    # to the respawn path.
                    logger.warning(
                        "WorkerPool: reuse failed (%s); "
                        "falling back to respawning workers.",
                        e,
                    )
                    self._shutdown_workers()
            else:
                logger.warning(
                    "WorkerPool: model_load is not picklable, respawning workers. "
                    "Define model_load as a top-level function to enable worker "
                    "reuse and faster test execution."
                )
                self._shutdown_workers()

        # Config changed or no pool — tear down old and spawn new
        if self._processes:
            logger.info("WorkerPool: config changed, shutting down old workers")
            self._shutdown_workers()

        self._spawn_workers(
            world_size,
            num_local_ranks,
            device,
            node_rank,
            ep_degree,
            sampling_dp_degree,
            dp_size=dp_size,
            attention_dp_size=attention_dp_size,
            embedding_dp_size=embedding_dp_size,
            lm_head_dp_size=lm_head_dp_size,
            mlp_dp_size=mlp_dp_size,
            draft_mlp_dp_size=draft_mlp_dp_size,
            encoder_tp_size=encoder_tp_size,
            encoder_dp_size=encoder_dp_size,
            dcp_size=dcp_size,
            nnodes=nnodes,
            model_load=model_load,
        )
        if model_load is not None:
            self._wait_for_load(num_local_ranks, load_timeout)
        self._key = key
        self._parallel_kwargs = parallel_kwargs
        return (
            num_local_ranks,
            self._processes,
            self._input_queues,
            self._output_pipes,
        )

    def _wait_for_load(self, num_local_ranks, load_timeout=_LOAD_TIMEOUT_S):
        """Wait for LOAD_OK/ERROR from all workers."""
        timeout = float(load_timeout)
        for rank in range(num_local_ranks):
            if self._output_pipes[rank].poll(timeout=timeout):
                item = self._output_pipes[rank].recv()
                if item[0] == "LOAD_OK":
                    continue
                elif item[0] == "ERROR":
                    raise RuntimeError(f"Worker {rank} failed to load model: {item[1]}")
                else:
                    raise RuntimeError(
                        f"Worker {rank} sent unexpected response during LOAD: {item}"
                    )
            else:
                raise TimeoutError(
                    f"Worker {rank} did not respond to LOAD within timeout of "
                    f"{timeout:g} seconds."
                )

    def _wait_for_reinit(self, num_local_ranks):
        """Wait for REINIT_OK/ERROR from all workers.

        Raises _ReinitFailed on any worker error or response timeout so that
        acquire() can fall back to respawning the pool.
        """
        for rank in range(num_local_ranks):
            if self._output_pipes[rank].poll(timeout=60.0):
                item = self._output_pipes[rank].recv()
                if item[0] == "REINIT_OK":
                    continue
                elif item[0] == "ERROR":
                    raise _ReinitFailed(
                        f"Worker {rank} failed to reinit parallel state: {item[1]}"
                    )
                else:
                    raise _ReinitFailed(
                        f"Worker {rank} sent unexpected response during REINIT_PARALLEL: {item}"
                    )
            else:
                raise _ReinitFailed(
                    f"Worker {rank} did not respond to REINIT_PARALLEL within 60s."
                )

    def _drain_stale_data(self, num_local_ranks):
        """Drain stale commands and results when reusing workers.

        We drain input queues first, then drain output pipes.
        This ensures _wait_for_load reads the correct LOAD_OK.
        """
        for rank in range(num_local_ranks):
            i_count = 0
            while not self._input_queues[rank].empty():
                self._input_queues[rank].get_nowait()
                i_count += 1
            o_count = 0
            while self._output_pipes[rank].poll(timeout=0):
                self._output_pipes[rank].recv()
                o_count += 1
            if i_count or o_count:
                logger.warning(
                    "WorkerPool: drained worker %d: %d input cmd(s), %d output msg(s)",
                    rank,
                    i_count,
                    o_count,
                )

    def _spawn_workers(
        self,
        world_size,
        num_local_ranks,
        device,
        node_rank,
        ep_degree,
        sampling_dp_degree,
        dp_size=1,
        attention_dp_size=1,
        embedding_dp_size=1,
        lm_head_dp_size=1,
        mlp_dp_size=1,
        draft_mlp_dp_size=None,
        encoder_tp_size=1,
        encoder_dp_size=1,
        dcp_size=1,
        nnodes=1,
        model_load=None,
    ):
        visible_cores_env = os.environ.get("NEURON_RT_VISIBLE_CORES")
        explicit_core_ids = None
        if visible_cores_env:
            if "," in visible_cores_env:
                explicit_core_ids = [
                    int(c.strip()) for c in visible_cores_env.split(",")
                ]
                start_core_id = explicit_core_ids[0]
                num_visible = len(explicit_core_ids)
            elif "-" in visible_cores_env:
                start, end = visible_cores_env.split("-")
                start_core_id = int(start)
                num_visible = int(end) - start_core_id + 1
            else:
                start_core_id = int(visible_cores_env)
                num_visible = 1
            if num_visible < num_local_ranks:
                raise ValueError(
                    f"NEURON_RT_VISIBLE_CORES specifies {num_visible} cores "
                    f"but num_local_ranks is {num_local_ranks}"
                )
        else:
            start_core_id = 0

        self._master_port = _get_unique_collectives_port(start_core_id)
        root_comm_id = _get_unique_root_comm_id(start_core_id)

        self._processes = []
        self._input_queues = []
        self._output_pipes = []
        self._num_local_ranks = num_local_ranks

        rank_offset = node_rank * num_local_ranks
        for local_rank in range(num_local_ranks):
            rank = rank_offset + local_rank
            core_id = (
                explicit_core_ids[local_rank]
                if explicit_core_ids
                else start_core_id + local_rank
            )
            input_queue = mp.Queue()
            parent_conn, child_conn = mp.Pipe(duplex=False)
            p = mp.Process(
                target=worker_process,
                args=(
                    rank,
                    world_size,
                    input_queue,
                    child_conn,
                    self._master_port,
                    device,
                    core_id,
                    ep_degree,
                    sampling_dp_degree,
                    dp_size,
                    attention_dp_size,
                    embedding_dp_size,
                    lm_head_dp_size,
                    mlp_dp_size,
                    draft_mlp_dp_size,
                    encoder_tp_size,
                    encoder_dp_size,
                    dcp_size,
                    nnodes,
                    model_load,
                    root_comm_id,
                ),
            )
            p.start()
            child_conn.close()
            self._processes.append(p)
            self._input_queues.append(input_queue)
            self._output_pipes.append(parent_conn)

        logger.info("WorkerPool: spawned %d workers", num_local_ranks)

    def _shutdown_workers(self):
        # Escalate over the whole group in phases. A live worker keeps its Neuron
        # core, so the next test fails in nrt_init with "cores busy, ret=-16".
        # Signal every survivor first, then wait once, so one wedged worker
        # cannot delay the signal to the others.
        for q in self._input_queues:
            q.put(("STOP",))
        alive = _join_group(self._processes, _WORKER_STOP_TIMEOUT_S)

        if alive:
            # Give SIGTERM time to unwind the runtime before the forced exit.
            for p in alive:
                p.terminate()
            alive = _join_group(alive, _WORKER_TERMINATE_TIMEOUT_S)

        if alive:
            logger.warning(
                "WorkerPool: %d worker(s) did not exit after SIGTERM (pids=%s). "
                "Sending SIGKILL to release the Neuron cores.",
                len(alive),
                [p.pid for p in alive],
            )
            for p in alive:
                p.kill()
            alive = _join_group(alive, _WORKER_KILL_TIMEOUT_S)

        if alive:
            logger.error(
                "WorkerPool: %d worker(s) survived SIGKILL (pids=%s). The Neuron "
                "cores stay busy for the next test. The workers are most likely "
                "wedged in a driver call.",
                len(alive),
                [p.pid for p in alive],
            )
        for pipe in self._output_pipes:
            pipe.close()
        self._processes = []
        self._input_queues = []
        self._output_pipes = []
        self._key = None
        self._parallel_kwargs = None
        logger.info("WorkerPool: workers shut down")

    def shutdown_all(self):
        """Kill all pooled workers. Called at session teardown."""
        if self._processes:
            self._shutdown_workers()


# Module-level singleton
_worker_pool = _WorkerPool()


class MPExecutor:
    """Multi-process executor for distributed model inference on Neuron devices.

    MPExecutor manages multiple worker processes, each running on a separate Neuron core,
    to execute distributed models in parallel. It handles process creation, communication
    via queues, and distributed coordination.

    Workers are pooled via a module-level ``_WorkerPool`` singleton. When consecutive
    MPExecutor instances share the same config (world_size, device),
    existing workers are reused and only the model is reloaded. workers are cleaned
    up at session teardown via ``_worker_pool.shutdown_all()``.

    CPU Mode:
        Pass device="cpu" to run on CPU instead of Neuron devices. This is useful
        for testing and debugging without hardware. When CPU mode is enabled,
        inputs are kept on CPU and Neuron-specific environment variables are not set.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> from vllm_neuron.utils.executor import MPExecutor
        >>>
        >>> # Define your model
        >>> class MyModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = nn.Linear(128, 64)
        ...
        ...     def forward(self, x):
        ...         return self.linear(x)
        >>>
        >>> # Model loading function
        >>> def model_load():
        ...     model = MyModel()
        ...     model.to("neuron:0")
        ...     return torch.compile(model, backend=envs.get_compile_backend_name())
        >>>
        >>> # Create executor
        >>> executor = MPExecutor(
        ...     world_size=2,
        ...     model_load=model_load,
        ... )
        >>>
        >>> # Run inference
        >>> input_tensor = torch.randn(2, 128)
        >>> executor.dispatch(input_tensor)
        >>> outputs = executor.collect()
        >>>
        >>> # Cleanup
        >>> executor.shutdown()

    Args:
        world_size (int): Number of processes/Neuron cores to use
        model_load (callable): Function that loads and compiles the model.
            Should return compiled model.
        device (str): Device to run on. Defaults to "neuron:0".
            Use "cpu" for testing without Neuron hardware.
        nnodes (int): The number of nodes used to run the model.
            Defaults to 1.
        node_rank (int): The rank of this node. Defaults to 0.
        ep_degree (int): Expert parallelism degree.
            Defaults to 1 (disabled). Passed to initialize_neuron_parallel_state.
        sampling_dp_degree (int): Sampling data-parallelism degree.
            Defaults to 1 (disabled). Passed to initialize_neuron_parallel_state.
        encoder_tp_size (int): Encoder tensor-parallelism degree.
            Defaults to 1 (disabled). Passed to initialize_neuron_parallel_state.
        encoder_dp_size (int): Encoder data-parallelism degree.
            Defaults to 1 (disabled). Passed to initialize_neuron_parallel_state.
        load_timeout (float): Seconds to wait for each worker's LOAD handshake.
            Defaults to 300. Covers reading the checkpoint
    """

    def __init__(
        self,
        world_size,
        model_load,
        device="neuron:0",
        nnodes=1,
        node_rank=0,
        ep_degree=1,
        sampling_dp_degree=1,
        dp_size=1,
        attention_dp_size=1,
        embedding_dp_size=1,
        lm_head_dp_size=1,
        mlp_dp_size=1,
        draft_mlp_dp_size=None,
        encoder_tp_size=1,
        encoder_dp_size=1,
        dcp_size=1,
        load_timeout=_LOAD_TIMEOUT_S,
    ) -> None:
        self.world_size = world_size
        self.dp_size = dp_size
        self.num_local_ranks = world_size // nnodes
        self.device = device

        (
            self.num_local_ranks,
            self.processes,
            self.input_queues,
            self.output_pipes,
        ) = _worker_pool.acquire(
            world_size=world_size,
            model_load=model_load,
            device=device,
            nnodes=nnodes,
            node_rank=node_rank,
            ep_degree=ep_degree,
            sampling_dp_degree=sampling_dp_degree,
            dp_size=dp_size,
            attention_dp_size=attention_dp_size,
            embedding_dp_size=embedding_dp_size,
            lm_head_dp_size=lm_head_dp_size,
            mlp_dp_size=mlp_dp_size,
            draft_mlp_dp_size=draft_mlp_dp_size,
            encoder_tp_size=encoder_tp_size,
            encoder_dp_size=encoder_dp_size,
            dcp_size=dcp_size,
            load_timeout=load_timeout,
        )

    def dispatch(self, *args):
        """Send inputs to all worker processes for distributed execution.

        Broadcasts the same inputs to all processes. For SPMD (Single Program,
        Multiple Data) execution, each process receives identical inputs but
        operates on different model shards.

        Args:
            *args: Input arguments to pass to the model
        """
        for rank in range(self.num_local_ranks):
            self.input_queues[rank].put(("RUN",) + args)

    def dispatch_per_rank(self, rank_inputs: list[tuple]):
        """Send different inputs to each worker process.

        Sends rank-specific inputs to each process. Useful for testing
        distributed operations where each rank needs different data.

        Args:
            rank_inputs: List of input tuples, one per rank. Length must equal num_local_ranks.
                        Each element is a tuple of arguments for that rank.

        Raises:
            ValueError: If length of rank_inputs doesn't match num_local_ranks
        """
        if len(rank_inputs) != self.num_local_ranks:
            raise ValueError(
                f"rank_inputs length {len(rank_inputs)} must match num_local_ranks {self.num_local_ranks}"
            )

        for rank in range(self.num_local_ranks):
            self.input_queues[rank].put(("RUN",) + rank_inputs[rank])

    def collect(self, timeout: float = 300.0):
        """Collect outputs from all worker processes.

        Waits for and collects results from all processes.

        Args:
            timeout (float): Timeout in seconds for each worker. Defaults to 300.0.

        Returns:
            list[torch.Tensor] or list[tuple]: List of outputs, one per process rank.
                Returns list of tuples if model output is a tuple.

        Raises:
            RuntimeError: If any process returns an error
            TimeoutError: If any process doesn't respond within timeout
        """
        outputs = [None] * self.num_local_ranks
        for rank in range(self.num_local_ranks):
            # Wait for data from each worker's pipe
            if self.output_pipes[rank].poll(timeout=timeout):
                item = self.output_pipes[rank].recv()
                if item[0] == "ERROR":
                    raise RuntimeError(f"Error in worker process {rank}: {item[1]}")

                _, output_data, dtype_or_flag = item

                # Handle tuple outputs
                if dtype_or_flag == "tuple":
                    output_tuple = []
                    for out_np, dtype_str in output_data:
                        output_tuple.append(_numpy_to_torch(out_np, dtype_str))
                    outputs[rank] = tuple(output_tuple)
                else:
                    # Single tensor output
                    outputs[rank] = _numpy_to_torch(output_data, dtype_or_flag)
            else:
                raise TimeoutError(f"Worker {rank} did not respond within timeout")

        return outputs

    def start_profile(
        self,
        output_dir: str = "./neuron_profiles",
        activities: list[str] | None = None,
        ranks: list[int] | None = None,
        sys_trace_max_events_per_nc: int | None = None,
        neff_cache_dir: str | None = None,
    ) -> None:
        """Start NRT profiling on specified workers.

        Args:
            output_dir: Directory for NTFF output files.
            activities: Activity types to capture.
                Defaults to ["device_profile", "system_profile"].
            ranks: Which local worker ranks to profile. Defaults to [0] (rank 0 only).
                These are local ranks on this node (0 to num_local_ranks-1).
                In multi-node setups, each node's MPExecutor manages its own
                local workers independently.
                Pass list(range(executor.num_local_ranks)) to profile all.
            sys_trace_max_events_per_nc: Max system trace events per NeuronCore.
            neff_cache_dir: Source dir to copy NEFFs from. Defaults to the
                Neuron compile cache directory.

        Raises:
            RuntimeError: If any worker fails to start profiling or in CPU mode.
            TimeoutError: If any worker does not respond within 30 seconds.

        Example:
            >>> executor = MPExecutor(world_size=2, model_load=model_load)
            >>> executor.start_profile(output_dir="/tmp/profiles")
            >>> executor.dispatch(input_tensor)
            >>> executor.collect()
            >>> executor.stop_profile()
        """
        if self.device == "cpu":
            raise RuntimeError("Neuron profiling is not available in CPU mode.")

        if ranks is None:
            ranks = [0]

        if neff_cache_dir is None:
            import libtorch_neuronx_lite.envs as libtorch_envs
            from vllm_neuron import envs

            neff_cache_dir = libtorch_envs.get_neff_cache_dir(
                backend=envs.get_compile_backend_name()
            )

        profile_kwargs = {
            "output_dir": output_dir,
            "activities": activities or ["device_profile", "system_profile"],
            "sys_trace_max_events_per_nc": sys_trace_max_events_per_nc,
            "neff_cache_dir": neff_cache_dir,
        }

        for rank in ranks:
            if rank >= self.num_local_ranks or rank < 0:
                raise ValueError(
                    f"rank {rank} is out of range [0, {self.num_local_ranks})"
                )

        for rank in ranks:
            self.input_queues[rank].put(("PROFILE_START", profile_kwargs))

        for rank in ranks:
            if self.output_pipes[rank].poll(timeout=30.0):
                item = self.output_pipes[rank].recv()
                if item[0] == "ERROR":
                    raise RuntimeError(
                        f"Worker {rank} failed to start profiling: {item[1]}"
                    )
            else:
                raise TimeoutError(
                    f"Worker {rank} did not ack PROFILE_START within 30s."
                )

        logger.info(
            "Profiling started on %d workers (ranks %s). Output: %s",
            len(ranks),
            ranks,
            output_dir,
        )

    def stop_profile(self) -> None:
        """Stop NRT profiling on all workers and save traces.

        Sends PROFILE_STOP to all local workers. NRT stop_profiling is a
        no-op on workers that are not actively profiling, so this is safe
        to call regardless of which ranks were started.

        Raises:
            RuntimeError: If any worker fails to stop profiling or in CPU mode.
            TimeoutError: If any worker does not respond within 60 seconds.

        Example:
            >>> executor.stop_profile()
        """
        if self.device == "cpu":
            raise RuntimeError("Neuron profiling is not available in CPU mode.")

        for rank in range(self.num_local_ranks):
            self.input_queues[rank].put(("PROFILE_STOP",))

        for rank in range(self.num_local_ranks):
            if self.output_pipes[rank].poll(timeout=60.0):
                item = self.output_pipes[rank].recv()
                if item[0] == "ERROR":
                    raise RuntimeError(
                        f"Worker {rank} failed to stop profiling: {item[1]}"
                    )
            else:
                raise TimeoutError(
                    f"Worker {rank} did not ack PROFILE_STOP within 60s."
                )

        logger.info("Profiling stopped on %d workers.", self.num_local_ranks)

    # TODO(CHRS-606): shutdown will be removed since worker pool will be shutdown after
    # all the tests finished.
    def shutdown(self):
        """Shut down workers and release Neuron cores."""
        _worker_pool._shutdown_workers()


# NRT failures that mean this worker was forked after the parent initialized the
# runtime (the child inherits a poisoned NRT handle — like CUDA fork-unsafety).
_FORK_NRT_SIGNATURES = (
    "NRT_STATE_CHILD",
    "nrt_tensor_allocate status=1",
    "could not be initialized",
    "nrt_allocate_neuron_cores",
    "cores busy",
)

_FORK_NRT_DIAGNOSIS = (
    "[MPExecutor fork-safety] Worker forked after the parent initialized NRT "
    "(a prior test likely ran a Neuron op directly in the pytest parent). "
    "MPExecutor tests need a 'world_size' pytest parameter so conftest orders "
    "them ahead of in-parent-device tests (see test/vllm_neuron/conftest.py)."
)


def _enrich_worker_error(exc: Exception) -> str:
    """Append the fork-safety diagnosis to the error string when it matches a known
    fork-after-NRT-init signature; otherwise return the bare message unchanged."""
    message = str(exc)
    haystack = f"{message}\n{traceback.format_exc()}"
    if any(sig in haystack for sig in _FORK_NRT_SIGNATURES):
        return f"{message}\n{_FORK_NRT_DIAGNOSIS}"
    return message


def _move_to_device(obj, device):
    if hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, (tuple, list)):
        return type(obj)(_move_to_device(item, device) for item in obj)
    elif isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _torch_to_numpy(tensor):
    """Convert a torch tensor to numpy, handling bfloat16 and fp8 dtypes."""
    t = tensor.detach().to("cpu")
    if t.dtype == torch.bfloat16:
        return t.float().numpy()
    if t.dtype in _FP8_NUMPY_DTYPES:
        return t.view(torch.uint8).numpy().view(_FP8_NUMPY_DTYPES[t.dtype])
    return t.numpy()


def _numpy_to_torch(arr, dtype_str):
    """Convert a numpy array back to a torch tensor, handling fp8 dtypes.

    Args:
        arr: numpy array (may use ml_dtypes for fp8).
        dtype_str: string like "torch.bfloat16" or "torch.float8_e5m2".
    """
    target_dtype = eval(dtype_str)
    # fp8 ml_dtypes arrays must go through uint8 view for torch.from_numpy
    if target_dtype in _FP8_NUMPY_DTYPES:
        return torch.from_numpy(arr.view("uint8")).view(target_dtype)
    t = torch.from_numpy(arr)
    if t.dtype != target_dtype:
        t = t.to(target_dtype)
    return t


def _restore_process_env(baseline: dict[str, str]) -> None:
    """Reset ``os.environ`` to ``baseline`` so a ``model_load``'s env mutations
    don't leak into the next test that reuses this pooled worker."""
    for key in [k for k in os.environ if k not in baseline]:
        del os.environ[key]
    for key, value in baseline.items():
        if os.environ.get(key) != value:
            os.environ[key] = value


def worker_process(
    rank,
    world_size,
    input_queue,
    output_pipe,
    master_port,
    device,
    core_id=None,
    ep_degree=1,
    sampling_dp_degree=1,
    dp_size=1,
    attention_dp_size=1,
    embedding_dp_size=1,
    lm_head_dp_size=1,
    mlp_dp_size=1,
    draft_mlp_dp_size=None,
    encoder_tp_size=1,
    encoder_dp_size=1,
    dcp_size=1,
    nnodes=1,
    initial_model_load=None,
    root_comm_id=None,
):
    """Worker for MPExecutor. Initializes runtime once, then loops
    over LOAD/RUN/STOP commands.

    Args:
        rank (int): Process rank (0 to num_local_ranks-1)
        world_size (int): Total number of processes
        input_queue (mp.Queue): Queue to receive input tensors
        output_pipe (mp.connection.Connection): Pipe to send output tensors
        master_port (str): Port number for distributed communication
        device (str): Device to run on ("cpu" or "neuron:0")
        core_id (int): The core ID to set NEURON_RT_VISIBLE_CORES
        nnodes (int): Number of nodes in the distributed cluster.
        ep_degree (int): Expert parallelism degree
        sampling_dp_degree (int): Sampling data-parallelism degree
        dp_size (int): vLLM data parallelism degree
        initial_model_load (callable, optional): Function to load the model
            at spawn time. If None, model is loaded via LOAD command.
        root_comm_id (str, optional): CCOM bootstrap endpoint shared by the
            whole worker group.
    """

    # Limit threads to avoid OpenMP conflicts in forked processes
    torch.set_num_threads(1)

    # Determine if running in CPU mode
    cpu_mode = device == "cpu"
    num_local_ranks = world_size // nnodes
    local_rank = rank % num_local_ranks

    from vllm_neuron.envs import get_dist_backend, is_native_backend

    native_backend = not cpu_mode and is_native_backend()

    # Configure environment for this process.
    # Use the deterministic port derived from core allocation (61234 + start_core_id)
    # rather than reading/mutating MASTER_PORT from the parent environment.
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = master_port
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ.setdefault("VLLM_TARGET_DEVICE", "neuron")

    if core_id is None:
        core_id = rank
    if not cpu_mode:
        os.environ["NEURON_RT_NUM_CORES"] = "1"
        # Pin this child to its allocated physical core before process-group
        # initialization, which initializes NRT for the native backend.
        os.environ["NEURON_RT_VISIBLE_CORES"] = f"{core_id}"

        # Set here rather than inherited: a forked child only gets what the parent
        # exported before mp.Process.start(), and NRT refuses to load a NEFF with
        # collective operations without it. setdefault so an explicit pin wins.
        if root_comm_id is not None:
            os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", root_comm_id)

        if native_backend:
            # These are topology metadata; the process-local tensor device
            # remains neuron:0 after the physical core is narrowed above.
            os.environ["LOCAL_RANK"] = str(local_rank)
            os.environ["LOCAL_WORLD_SIZE"] = str(num_local_ranks)

    os.environ["NCCL_DEBUG"] = "ERROR"

    dist_backend = get_dist_backend()
    master_addr = os.environ["MASTER_ADDR"]
    init_method = f"tcp://{master_addr}:{master_port}"
    dist.init_process_group(
        dist_backend, init_method=init_method, rank=rank, world_size=world_size
    )

    from vllm_neuron.parallel.neuron_parallel_state import (
        initialize_neuron_parallel_state,
    )

    # With DP, partition ranks into DP replicas. Each DP replica forms a TP group.
    tp_size = world_size // dp_size
    dp_rank = rank // tp_size
    tp_global_ranks = [dp_rank * tp_size + i for i in range(tp_size)]

    initialize_neuron_parallel_state(
        ep_degree=ep_degree,
        sampling_dp_degree=sampling_dp_degree,
        dp_size=dp_size,
        attention_dp_size=attention_dp_size,
        embedding_dp_size=embedding_dp_size,
        lm_head_dp_size=lm_head_dp_size,
        mlp_dp_size=mlp_dp_size,
        draft_mlp_dp_size=draft_mlp_dp_size,
        encoder_tp_size=encoder_tp_size,
        encoder_dp_size=encoder_dp_size,
        dcp_size=dcp_size,
        nnodes=nnodes,
        tp_global_ranks=tp_global_ranks,
        local_rank=local_rank,
    )

    if native_backend:
        # Lite initializes NRT while constructing the native process group and
        # exposes one process-local device after narrowing visible cores.
        # torch.accelerator.current_device_index() remains -1 in compile-only
        # mode, and physical/local rank IDs are not valid tensor device indices.
        device = "neuron:0"

    # Stamp each parallel group's rank_in_group onto it as an on-device tensor, for SPMD compatibility
    _spmd_rank_tensors = stamp_spmd_rank_tensors(device)  # noqa: F841

    from vllm.config import VllmConfig, ParallelConfig, set_current_vllm_config

    vllm_config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=tp_size,
            data_parallel_size=dp_size,
            decode_context_parallel_size=dcp_size,
        )
    )

    # Pristine env, captured before any model_load runs, so each LOAD can
    # restore it and drop env mutations left by the previous test's model_load.
    _baseline_env = dict(os.environ)

    def _load_model(load_fn, reset_dynamo=False):
        """Load a model and send LOAD_OK/ERROR to the parent."""
        nonlocal model
        try:
            # Always start from the pristine env so a prior model_load's
            # mutations can't leak in.
            _restore_process_env(_baseline_env)
            if reset_dynamo:
                torch._dynamo.reset()
            with set_current_vllm_config(vllm_config):
                model = load_fn()
            if model is None:
                raise RuntimeError(
                    "model_load function returned None — it must return a callable model"
                )
            logger.info("Worker %d: model loaded successfully", rank)
            output_pipe.send(("LOAD_OK", rank))
        except Exception as e:
            logger.error("Worker %d: model load failed: %s", rank, e)
            traceback.print_exc()
            output_pipe.send(("ERROR", _enrich_worker_error(e)))

    # --- Load initial model if provided at spawn time (avoids pickling via queue) ---
    model = None
    if initial_model_load is not None:
        _load_model(initial_model_load)

    # --- Command loop ---
    try:
        while True:
            try:
                cmd = input_queue.get(timeout=1.0)
            except Empty:
                continue

            if cmd[0] == "STOP":
                break

            elif cmd[0] == "REINIT_PARALLEL":
                # Reinitialize parallel groups in-place without respawning.
                # Only Neuron groups are always rebuilt; vLLM groups (_TP, _DP, _PP)
                # are only rebuilt when dp_size changes (since they depend on it).
                # _WORLD and patches from _ensure_vllm_parallel_state persist.
                _, parallel_kwargs = cmd
                try:
                    from vllm_neuron.parallel.neuron_parallel_state import (
                        destroy_neuron_parallel_state,
                        _create_neuron_groups,
                        _reinit_vllm_model_parallel,
                        _patch_getters,
                        _patch_destroy,
                    )
                    import vllm.distributed.parallel_state as vllm_parallel_state

                    new_dp_size = parallel_kwargs.pop("dp_size", dp_size)

                    # destroy_neuron_parallel_state also restores the
                    # original vLLM destroy_model_parallel function.
                    destroy_neuron_parallel_state()

                    if new_dp_size != dp_size:
                        vllm_parallel_state.destroy_model_parallel()
                        _reinit_vllm_model_parallel(
                            tp_size=world_size // new_dp_size,
                            dp_size=new_dp_size,
                        )
                        dp_size = new_dp_size

                    _create_neuron_groups(**parallel_kwargs)
                    _patch_getters()
                    _patch_destroy()
                    # Re-stamp groups
                    _spmd_rank_tensors = stamp_spmd_rank_tensors(device)
                    logger.info("Worker %d: parallel groups reinitialized", rank)
                    output_pipe.send(("REINIT_OK", rank))
                except Exception as e:
                    logger.error("Worker %d: parallel reinit failed: %s", rank, e)
                    traceback.print_exc()
                    output_pipe.send(("ERROR", _enrich_worker_error(e)))

            elif cmd[0] == "LOAD":
                _, load_fn = cmd
                _load_model(load_fn, reset_dynamo=True)

            elif cmd[0] == "RUN":
                args = cmd[1:]
                try:
                    if model is None:
                        raise RuntimeError(
                            "No model loaded. Call MPExecutor with a model_load "
                            "function or send a LOAD command before dispatching."
                        )
                    with torch.no_grad():
                        args = [_move_to_device(arg, device) for arg in args]
                        output = model(*args)

                    if isinstance(output, tuple):
                        output_list = [
                            (_torch_to_numpy(out), str(out.dtype)) for out in output
                        ]
                        output_pipe.send((rank, output_list, "tuple"))
                    else:
                        output_pipe.send(
                            (rank, _torch_to_numpy(output), str(output.dtype))
                        )

                except Exception as e:
                    traceback.print_exc()
                    output_pipe.send(("ERROR", _enrich_worker_error(e)))

            elif cmd[0] == "PROFILE_START":
                _, profile_kwargs = cmd
                try:
                    if cpu_mode:
                        raise RuntimeError(
                            "Neuron profiling is not available in CPU mode."
                        )
                    output_dir = profile_kwargs["output_dir"]
                    activities = profile_kwargs["activities"]
                    sys_trace_max_events = profile_kwargs.get(
                        "sys_trace_max_events_per_nc"
                    )
                    neff_cache_dir = profile_kwargs.get("neff_cache_dir")
                    os.makedirs(output_dir, exist_ok=True)
                    runtime = torch.classes.neuron.Runtime()
                    runtime.start_profiling(
                        output_dir,
                        activities,
                        None,  # neuron_cores — profile all visible (1 per worker)
                        sys_trace_max_events,
                        neff_cache_dir,
                    )
                    logger.info(
                        "Worker %d: profiling started. Output dir: %s", rank, output_dir
                    )
                    output_pipe.send(("PROFILE_START_OK", rank))
                except Exception as e:
                    traceback.print_exc()
                    output_pipe.send(("ERROR", _enrich_worker_error(e)))

            elif cmd[0] == "PROFILE_STOP":
                try:
                    if cpu_mode:
                        raise RuntimeError(
                            "Neuron profiling is not available in CPU mode."
                        )
                    runtime = torch.classes.neuron.Runtime()
                    runtime.stop_profiling()
                    logger.info("Worker %d: profiling stopped.", rank)
                    output_pipe.send(("PROFILE_STOP_OK", rank))
                except Exception as e:
                    traceback.print_exc()
                    output_pipe.send(("ERROR", _enrich_worker_error(e)))

            else:
                logger.error("Worker %d: unknown command %r", rank, cmd[0])
                output_pipe.send(("ERROR", f"Unknown command: {cmd[0]}"))

    except Exception as e:
        traceback.print_exc()
        output_pipe.send(("ERROR", _enrich_worker_error(e)))
    finally:
        output_pipe.close()
        dist.destroy_process_group()
