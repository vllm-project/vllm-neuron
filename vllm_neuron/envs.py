# SPDX-License-Identifier: Apache-2.0
"""
vLLM Neuron Environment Variables Configuration

This module provides centralized environment variable management for vLLM Neuron

All environment variables are:
- Lazily evaluated when accessed
- Type-safe with proper validation
- Prefixed with VLLM_NEURON_ for namespace isolation
"""

import functools
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    # Core System Variables
    VLLM_NEURON_CPU_MODE: bool = False
    VLLM_NEURON_CPU_COMPILE: bool = False
    VLLM_NEURON_LOG_LEVEL: str = "INFO"
    VLLM_NEURON_DEBUG_MODE: bool = False
    # Experimental: allow MoE / expert parallelism with data_parallel_size > 1
    # on the V2 runner. Single-phase traffic only: mixed prefill/decode phases
    # across the EP-spanning DP group raise at runtime (no cross-rank phase
    # coordination; production wide-EP separates phases via disaggregated
    # inference, which V2 does not support yet).
    VLLM_NEURON_ALLOW_V2_WIDE_EP: bool = False
    VLLM_NEURON_BARRIER_TIMEOUT: int = 3600
    VLLM_NEURON_DISABLE_PARALLEL_TRACE: bool = False
    # Select a guard filter for torch.compile. See
    # get_precompile_guard_filter_fn(). "" keeps every guard.
    VLLM_NEURON_PRECOMPILE_GUARD_FILTER: str = ""
    # Number of spawn worker processes for the EPD Router's HF preprocessing pool.
    VLLM_NEURON_EPD_PREPROC_WORKERS: int = 8
    VLLM_NEURON_EPD_PREPROC_WORKER_THREADS: int = 4
    VLLM_NEURON_EPD_SHM_BUFFER_BYTES: int = 8 * 1024 * 1024 * 1024
    # TODO: Remove VLLM_NEURON_SWITCH_CC and derive topology from instance type.
    VLLM_NEURON_SWITCH_CC: bool = False
    VLLM_NEURON_MIN_KV_BUDGET_GIB: float = 1.0
    VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION: float = 0.30
    VLLM_NEURON_RESERVED_HEADROOM_GIB: float | None = None
    VLLM_NEURON_WORKER_TERMINATION_TIMEOUT: int = 5
    VLLM_NEURON_NIXL_PROGRESS_THREAD_DELAY_US: int = 0
    # Build the NIXL agent with reader-writer thread sync. On by default; 0 keeps NIXL's NONE.
    VLLM_NEURON_NIXL_THREAD_SYNC_RW: bool = True
    VLLM_NEURON_BATCH_QUEUE_DEPTH: int = 0
    # Bound the vLLM shm_broadcast reader spin. On by default, because an unbounded
    # spin holds one core per reader. Set it to 0 to keep upstream's timing.
    VLLM_NEURON_SHM_READER_SPIN_PATCH: bool = True
    # sys.setswitchinterval for a worker process. 0 keeps CPython's default.
    VLLM_NEURON_GIL_SWITCH_INTERVAL_S: float = 0.0
    # Offset added to the MPExecutor collective and root-comm base ports.
    # `make test` sets a different offset per concurrent build-time pass so the
    # passes do not collide on a rendezvous port. Default 0 keeps the production
    # ports.
    VLLM_NEURON_MPEXECUTOR_PORT_OFFSET: int = 0
    VLLM_NEURON_DISABLE_NKI_KERNELS: bool = False
    # The fused GPT-OSS tail has ONE flag, and it defaults ON. Its entry in
    # environment_variables below carries the default and the parsing.
    VLLM_NEURON_GPT_OSS_TAIL_KERNEL: bool = True
    # GPT-OSS decode giga-kernel fusions. Each one is OFF by default, because each
    # one removes a host computation and needs its own accuracy gate.
    VLLM_NEURON_MEGA_FUSE_EMBED: bool = False
    VLLM_NEURON_MEGA_FUSE_EMBED_HIDDEN: bool = False
    VLLM_NEURON_MEGA_FUSE_ROPE: bool = False
    VLLM_NEURON_QWEN3_INTREE_MLP: bool = True
    # LNC1 a2a-fused sequence-parallel decode boundaries. ON by default; set "0"
    # to disable sequence-parallel decode entirely (kill switch): the a2a
    # composers are the only sequence-parallel decode path, so a refused shape
    # or a disabled flag keeps the all-reduce path.
    VLLM_NEURON_QWEN3_SP_A2A: bool = True
    # Gather the fused sampling tail's rows with ONE all_gather over the 128-rank lm_head TP
    # group, in place of all_gather[TP] followed by all_gather[lm_head_dp].
    VLLM_NEURON_TAIL_FLAT_ROW_GATHER: bool = False
    VLLM_NEURON_ATTN_MASK_TORCH: bool = False
    VLLM_NEURON_ATTN_PV_SWAP: bool = False
    VLLM_NEURON_SKIP_PREFILL_WARMUP: bool = False
    VLLM_NEURON_SKIP_DECODE_WARMUP: bool = False
    VLLM_NEURON_PROFILE_WARMUP_DECODE: bool = False
    VLLM_NEURON_SKIP_PREFILL_DECODE_WARMUP: bool = False
    VLLM_NEURON_SKIP_ENCODER_WARMUP: bool = False
    # Force the STATIC FP8 (non-MX) attention path on TRN3 even when STATIC_MX
    # kernels are available. Used by the llama3 factory as an escape hatch.
    VLLM_NEURON_FORCE_STATIC_FP8: bool = False
    # Snapshot Capture Variables
    VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_ENABLE: bool = False
    VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_AT_CALL: Optional[str] = None
    VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_TOKEN: Optional[str] = None
    VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_REQUEST: Optional[str] = None
    VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_MAX_CAPTURES: str = "4"
    VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_RANKS: Optional[str] = None
    VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_FORMAT: str = "pt"
    VLLM_NEURON_EFA_INSTANCE_FAMILY: str = ""
    # Compile-Artifact Manifest Variables
    # Directory where per-engine compile-artifact ledgers and manifests are
    # written. A path enables the feature; unset (None) or empty string ("")
    # disables it. There is no default location.
    VLLM_NEURON_MANIFEST_DIR: Optional[str] = None
    # GPT-OSS giga-kernel decode DMA-order schedule
    VLLM_NEURON_MEGA_DMA_ORDER_DERIVE: bool = False
    VLLM_NEURON_MEGA_DMA_ORDER_CONSTRAINTS_FILE: Optional[str] = None
    VLLM_NEURON_MEGA_SBUF_ALLOC: bool = True


def maybe_convert_bool(value: str | None) -> bool | None:
    """
    Safely convert string to boolean using numeric conversion.

    Args:
        value: String value to convert ("0", "1") or None

    Returns:
        True if value is "1", False if value is "0", None if value is None

    Raises:
        ValueError: If value cannot be converted to int

    Examples:
        >>> maybe_convert_bool("1")
        True
        >>> maybe_convert_bool("0")
        False
        >>> maybe_convert_bool(None)
        None
        >>> maybe_convert_bool("invalid")  # Raises ValueError
    """
    if value is None:
        return None
    return bool(int(value))


def maybe_convert_int(value: str | None) -> int | None:
    """
    Safely convert string to integer.

    Args:
        value: String value to convert or None

    Returns:
        Integer value if conversion successful, None if value is None

    Raises:
        ValueError: If value cannot be converted to int

    Examples:
        >>> maybe_convert_int("600")
        600
        >>> maybe_convert_int("0")
        0
        >>> maybe_convert_int(None)
        None
        >>> maybe_convert_int("invalid")  # Raises ValueError
    """
    if value is None:
        return None
    return int(value)


def optional_perf_int(name: str) -> int:
    """Read an opt-in performance knob, and return 0 for anything unusable.

    These knobs turn off at 0 and are read while the platform registers, so a raise there takes
    the server down before it serves. An operator who exports the name empty, or with a typo,
    means "leave it alone". A malformed value warns, because it was an attempt to set something.

    Args:
        name: Environment variable name.

    Returns:
        The value, or 0 when unset, empty or not an integer.

    Examples:
        >>> optional_perf_int("VLLM_NEURON_NOT_SET_ANYWHERE")
        0
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer, so it is ignored.", name, raw)
        return 0


def optional_perf_bool(name: str, default: bool) -> bool:
    """Read an on/off performance knob, and fall back to ``default`` for anything unusable.

    The boolean member of the :func:`optional_perf_int` family, and it exists for the same
    reason: these knobs are read while the platform registers, so a raise there takes the server
    down before it serves. That matters most for a knob that is on by default, because the
    variable is then the documented way to turn the behaviour off, and an operator reaching for
    it must not be able to stop the server with a spelling.

    Accepts 0/1, true/false, on/off and yes/no in any case.

    Args:
        name: Environment variable name.
        default: Value to use when unset, empty or unrecognised.

    Returns:
        The requested setting, or ``default``.

    Examples:
        >>> optional_perf_bool("VLLM_NEURON_NOT_SET_ANYWHERE", True)
        True
    """
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "on", "yes"):
        return True
    if raw in ("0", "false", "off", "no"):
        return False
    logger.warning(
        "%s=%r is not a recognised on/off value, so %s is kept.", name, raw, default
    )
    return default


def optional_perf_float(name: str) -> float:
    """Read an opt-in performance knob that takes a fraction, and return 0.0 if unusable.

    The float counterpart of :func:`optional_perf_int`, with the same contract: 0.0 means leave
    the behaviour alone, and a malformed value warns rather than raising.

    Args:
        name: Environment variable name.

    Returns:
        The value, or 0.0 when unset, empty or not a number.

    Examples:
        >>> optional_perf_float("VLLM_NEURON_NOT_SET_ANYWHERE")
        0.0
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number, so it is ignored.", name, raw)
        return 0.0


def maybe_convert_float(value: str | None) -> float | None:
    """Safely convert string to float."""
    if value is None:
        return None
    return float(value)


environment_variables: dict[str, Callable[[], Any]] = {
    # ================== Core System Variables ==================
    # Enable CPU fallback mode instead of using Neuron accelerators
    "VLLM_NEURON_CPU_MODE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_CPU_MODE")) or False
    ),
    # Enable CPU graph capture and compilation
    "VLLM_NEURON_CPU_COMPILE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_CPU_COMPILE")) or False
    ),
    # Logging level for vLLM Neuron components
    "VLLM_NEURON_LOG_LEVEL": lambda: os.getenv("VLLM_NEURON_LOG_LEVEL", "INFO").upper(),
    # Enable debug mode for verbose output and additional diagnostics
    "VLLM_NEURON_DEBUG_MODE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_DEBUG_MODE")) or False
    ),
    "VLLM_NEURON_ALLOW_V2_WIDE_EP": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_ALLOW_V2_WIDE_EP")) or False
    ),
    # Admit the packed-PV (P-stationary) MM2 attention path at d_head 128. Off by default.
    "VLLM_NEURON_ATTN_PV_SWAP": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_ATTN_PV_SWAP")) or False
    ),
    # Guard filter for torch.compile: "", "off", "portable" or "skip_all"
    "VLLM_NEURON_PRECOMPILE_GUARD_FILTER": lambda: (
        os.getenv("VLLM_NEURON_PRECOMPILE_GUARD_FILTER", "").strip().lower()
    ),
    # tp_barrier timeout in seconds (raise for long serial warmups)
    "VLLM_NEURON_BARRIER_TIMEOUT": lambda: (
        maybe_convert_int(os.getenv("VLLM_NEURON_BARRIER_TIMEOUT")) or 3600
    ),
    # Offset for the MPExecutor collective and root-comm base ports. `make test`
    # sets a different offset per concurrent build-time pass to prevent a
    # rendezvous-port collision between the passes.
    "VLLM_NEURON_MPEXECUTOR_PORT_OFFSET": lambda: (
        maybe_convert_int(os.getenv("VLLM_NEURON_MPEXECUTOR_PORT_OFFSET")) or 0
    ),
    # Number of spawn worker processes for EPD Router HF preprocessing
    "VLLM_NEURON_EPD_PREPROC_WORKERS": lambda: (
        maybe_convert_int(os.getenv("VLLM_NEURON_EPD_PREPROC_WORKERS"))
        if os.getenv("VLLM_NEURON_EPD_PREPROC_WORKERS") is not None
        else 8
    ),
    "VLLM_NEURON_EPD_PREPROC_WORKER_THREADS": lambda: (
        maybe_convert_int(os.getenv("VLLM_NEURON_EPD_PREPROC_WORKER_THREADS")) or 4
    ),
    # Per-worker shm ring size (bytes) for the Router->VE shm pixel handoff.
    "VLLM_NEURON_EPD_SHM_BUFFER_BYTES": lambda: (
        maybe_convert_int(os.getenv("VLLM_NEURON_EPD_SHM_BUFFER_BYTES"))
        or 8 * 1024 * 1024 * 1024
    ),
    # When True, disable the parallel-trace fork pool entirely and run
    # graph extraction sequentially in the parent process.
    "VLLM_NEURON_DISABLE_PARALLEL_TRACE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_DISABLE_PARALLEL_TRACE")) or False
    ),
    # Minimum KV budget (GiB) guardrail
    "VLLM_NEURON_MIN_KV_BUDGET_GIB": lambda: (
        maybe_convert_float(os.getenv("VLLM_NEURON_MIN_KV_BUDGET_GIB"))
        if os.getenv("VLLM_NEURON_MIN_KV_BUDGET_GIB") is not None
        else 1.0
    ),
    # Default KV cap for the memory-utilization budget.
    "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION": lambda: (
        maybe_convert_float(os.getenv("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION"))
        if os.getenv("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION") is not None
        else 0.30
    ),
    # An explicit reserve enables headroom sizing for this worker.
    "VLLM_NEURON_RESERVED_HEADROOM_GIB": lambda: (
        maybe_convert_float(os.getenv("VLLM_NEURON_RESERVED_HEADROOM_GIB"))
        if os.getenv("VLLM_NEURON_RESERVED_HEADROOM_GIB") is not None
        else None
    ),
    # Local cache directory for model checkpoints
    "VLLM_NEURON_CHECKPOINT_CACHE": lambda: os.getenv(
        "NXDI_CHECKPOINT_CACHE", "/tmp/vllm_neuron-checkpoints"
    ),
    # Golden cache directory (disk tier)
    "VLLM_NEURON_GOLDEN_CACHE_DIR": lambda: os.path.expandvars(
        os.getenv("VLLM_NEURON_GOLDEN_CACHE_DIR", "/tmp/vllm_neuron-goldens-$USER")
    ),
    # Golden cache S3 URI (secondary tier, empty = disabled)
    "VLLM_NEURON_S3_GOLDENS_URI": lambda: os.getenv("VLLM_NEURON_S3_GOLDENS_URI", ""),
    # TODO: Remove VLLM_NEURON_SWITCH_CC and derive topology from instance type.
    # When True, uses contiguous groups for 8x8 topology instead of custom TRN2 mesh.
    "VLLM_NEURON_SWITCH_CC": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_SWITCH_CC")) or False
    ),
    # Timeout in seconds for worker termination (SIGTERM→SIGKILL).
    # Default is 5s (upstream vLLM uses 4s for workers, 5s for API servers).
    # Increase when system profiling is enabled (NEURON_RT_INSPECT_ENABLE=1),
    # as the Neuron runtime needs time to flush profiling data after SIGTERM.
    "VLLM_NEURON_WORKER_TERMINATION_TIMEOUT": lambda: (
        maybe_convert_int(os.getenv("VLLM_NEURON_WORKER_TERMINATION_TIMEOUT")) or 5
    ),
    # Sleep between two sweeps of the NIXL libfabric progress thread, in microseconds.
    # 0 keeps NIXL's own default, which never sleeps and holds a core per rank. Opt-in only:
    # it reaches into NIXL's bindings, so a performance run sets it, not a service.
    "VLLM_NEURON_NIXL_PROGRESS_THREAD_DELAY_US": lambda: optional_perf_int(
        "VLLM_NEURON_NIXL_PROGRESS_THREAD_DELAY_US"
    ),
    # Build the NIXL agent with NIXL_THREAD_SYNC_RW in place of NIXL's NONE. On by default,
    # because NONE leaves the agent's remote-agent map unlocked while two threads use it: the
    # handshake thread loads a new remote (loadRemoteMD) as the main thread posts a read
    # (postXferReq, which releases the GIL). A lookup during a rehash then fails with
    # NIXL_ERR_NOT_FOUND or segfaults the worker, as in the GPT-OSS 120B DI throughput run.
    # Set it to 0 to keep NIXL's own default.
    "VLLM_NEURON_NIXL_THREAD_SYNC_RW": lambda: optional_perf_bool(
        "VLLM_NEURON_NIXL_THREAD_SYNC_RW", True
    ),
    # Number of batches the async engine keeps in flight. 0 keeps the upstream depth, which is
    # the only depth every configuration can serve: the runner holds one step of sampled tokens,
    # and upstream's 2 leaves exactly one step outstanding. A deeper queue hides more host time,
    # but it makes the runner read a stale step. Opt-in, per configuration, once measured.
    "VLLM_NEURON_BATCH_QUEUE_DEPTH": lambda: optional_perf_int(
        "VLLM_NEURON_BATCH_QUEUE_DEPTH"
    ),
    # Bound the shm_broadcast reader spin. On by default, unlike the two knobs above, because
    # upstream's one-second spin window is never reached at a millisecond step, so every reader
    # holds a core it is not using. Measured worth on the GPT-OSS 120B decode reference config:
    # 2.3x the request throughput and a third of the p99. Non-raising on purpose: this variable
    # is the documented way to turn a default-on behaviour off, so a misspelling must not take
    # the server down before it serves.
    "VLLM_NEURON_SHM_READER_SPIN_PATCH": lambda: optional_perf_bool(
        "VLLM_NEURON_SHM_READER_SPIN_PATCH", True
    ),
    # sys.setswitchinterval for a worker process, in seconds. 0 keeps CPython's own default.
    # Opt-in, like the two knobs above: the setting is global to the interpreter, so it changes
    # thread scheduling for every model the worker ever serves, not only the one it was measured
    # on. Measured worth on the GPT-OSS 120B decode reference config: nothing. 5 legs at 0.0001 s
    # and 5 legs at CPython's 0.005 s both read p50 4.64 ms, so a service pays no price for
    # leaving it alone.
    "VLLM_NEURON_GIL_SWITCH_INTERVAL_S": lambda: optional_perf_float(
        "VLLM_NEURON_GIL_SWITCH_INTERVAL_S"
    ),
    # Disable NKI kernels — forces can_run_kernel() to return False
    "VLLM_NEURON_DISABLE_NKI_KERNELS": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_DISABLE_NKI_KERNELS")) or False
    ),
    # Fuse the GPT-OSS decode tail into NKI kernels (norm + lm_head + top-k +
    # sampler). ON by default; set the variable to 0 to take the torch tail back.
    # `is not False` rather than `or True`: maybe_convert_bool returns None when the
    # variable is unset, and `False or True` is True, so `or` cannot express default-on
    # and would make "=0" impossible to honour. Grep "[gpt_oss_tail]" for the per-trace
    # decision and, on a fallback, the reason.
    "VLLM_NEURON_GPT_OSS_TAIL_KERNEL": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_GPT_OSS_TAIL_KERNEL")) is not False
    ),
    # Build the token embedding inside the GPT-OSS decode giga-kernel, which keeps the hidden
    # state out of HBM. Needs embedding_dp_size == 1, which the model asserts.
    "VLLM_NEURON_MEGA_FUSE_EMBED": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_MEGA_FUSE_EMBED")) or False
    ),
    # Shard the fused embedding by the HIDDEN dim, not the vocab dim, so each rank holds
    # [vocab, H/tp_size] and the kernel does one hidden-dim all_gather. Needs the fused embedding.
    "VLLM_NEURON_MEGA_FUSE_EMBED_HIDDEN": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_MEGA_FUSE_EMBED_HIDDEN")) or False
    ),
    # Compute the RoPE cos/sin tables inside the GPT-OSS decode giga-kernel, from
    # pos_ids and inv_freq. The host then does not run the rotary embedding.
    "VLLM_NEURON_MEGA_FUSE_ROPE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_MEGA_FUSE_ROPE")) or False
    ),
    # Use the in-tree Qwen3-32B MXFP8 decode MLP kernel. On by default; set to 0
    # to fall back to the shared nkilib kernel. Unset means on, so default to
    # True unless the value is explicitly false.
    "VLLM_NEURON_QWEN3_INTREE_MLP": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_QWEN3_INTREE_MLP")) is not False
    ),
    "VLLM_NEURON_QWEN3_SP_A2A": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_QWEN3_SP_A2A")) is not False
    ),
    # Gather the fused sampling tail's rows with ONE all_gather over the 128-rank lm_head TP group,
    # in place of all_gather[TP=8] in the giga-kernel followed by all_gather[lm_head_dp=16].
    "VLLM_NEURON_TAIL_FLAT_ROW_GATHER": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_TAIL_FLAT_ROW_GATHER")) or False
    ),
    # Build the decode attention mask in torch, because gen_mask_tkg_hbm gives NaN
    # logits on the packed-FP8 QK-swap path (NKI-1460). Off by default.
    "VLLM_NEURON_ATTN_MASK_TORCH": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_ATTN_MASK_TORCH")) or False
    ),
    # Skip prefill warmup/compilation without requiring kv-transfer-config.
    # Useful for decode-only profiling workflows.
    "VLLM_NEURON_SKIP_PREFILL_WARMUP": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_SKIP_PREFILL_WARMUP")) or False
    ),
    # Skip both prefill and decode (language-model) warmup without requiring a
    # transfer config. Used by EPD vision-only (VE) pools, which have no
    # language model (mirrors the prefill/decode levers).
    "VLLM_NEURON_SKIP_PREFILL_DECODE_WARMUP": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_SKIP_PREFILL_DECODE_WARMUP")) or False
    ),
    # Skip encoder warmup without requiring a transfer config. Used by
    # EPD language-only (PD) pools, which have no encoder.
    "VLLM_NEURON_SKIP_ENCODER_WARMUP": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_SKIP_ENCODER_WARMUP")) or False
    ),
    # Skip decode warmup/compilation without requiring kv-transfer-config.
    # Useful for prefill-only profiling workflows.
    "VLLM_NEURON_SKIP_DECODE_WARMUP": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_SKIP_DECODE_WARMUP")) or False
    ),
    # Wrap decode warmup in an NRT profile capture. Warmup drives the decode
    # NEFF on-device with synthetic inputs, so this reliably records the decode
    # kernel's execution into the neuron_profiler output_dir (a /start_profile
    # window around `vllm bench serve` can miss it and produce an empty NTFF).
    # Requires --profiler-config '{"profiler":"cuda"}' + additional-config
    # neuron_profiler.output_dir.
    "VLLM_NEURON_PROFILE_WARMUP_DECODE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_PROFILE_WARMUP_DECODE")) or False
    ),
    # Force the STATIC FP8 (non-MX) attention path on TRN3 even when STATIC_MX
    # kernels are available (llama3 factory escape hatch).
    "VLLM_NEURON_FORCE_STATIC_FP8": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_FORCE_STATIC_FP8")) or False
    ),
    # ================== Snapshot Capture Variables ==================
    # Master switch for NRT-boundary input snapshot capture; off by default.
    "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_ENABLE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_ENABLE"))
        or False
    ),
    # Comma-separated 0-based call indices to capture (-1 = all); unset selects
    # the first post-warmup call.
    "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_AT_CALL": lambda: os.getenv(
        "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_AT_CALL", None
    ),
    # Comma-separated generated-token indices to capture (decode only); the
    # forward whose inputs *produce* the token is captured. Unset disables it.
    "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_TOKEN": lambda: os.getenv(
        "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_TOKEN", None
    ),
    # Comma-separated request_ids to capture; fires on any forward whose batch
    # contains the request. Matched against the caller's base id, ignoring the
    # unique suffix vLLM appends. Unset disables it.
    "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_REQUEST": lambda: os.getenv(
        "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_CAPTURE_REQUEST", None
    ),
    # Hard cap on captured calls per process, so a broad rule cannot fill disk
    # or stall the engine. Defaults to 4; a non-positive or malformed override
    # falls back to that default.
    "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_MAX_CAPTURES": lambda: os.getenv(
        "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_MAX_CAPTURES", "4"
    ),
    # Comma-separated tp_rank set to capture; unset captures all ranks (each
    # worker writes its own global-rank directory). Filtered on tp_rank, so
    # with DP>1 a value selects that tp position in every DP group.
    "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_RANKS": lambda: os.getenv(
        "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_RANKS", None
    ),
    # On-disk artifact format the serialize op writes for each tensor{i}:
    # "pt" (pickled, any dtype) or "npy" (raw element bytes, value-preserving;
    # bf16/fp8 stored as |V2/|V1). "npy" pulls in a numpy read dependency on
    # replay. Defaults "pt". Validated against the write_tensors op's accepted
    # formats, so an unsupported value fails at startup.
    "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_FORMAT": lambda: os.getenv(
        "VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_FORMAT", "pt"
    ),
    # Explicit instance-family override. When set, get_instance_family()
    # returns this verbatim instead of resolving from the sysfs product_name
    # (which defaults any trn3* to trn3pds and any trn2* to trn2).
    "VLLM_NEURON_EFA_INSTANCE_FAMILY": lambda: os.getenv(
        "VLLM_NEURON_EFA_INSTANCE_FAMILY", ""
    ),
    # Directory where per-engine compile-artifact ledgers and manifests are
    # written. Returns the raw configured value: a path enables the feature,
    # while None (unset) or "" (empty) disables it.
    "VLLM_NEURON_MANIFEST_DIR": lambda: os.getenv("VLLM_NEURON_MANIFEST_DIR"),
    # Emit the compiler's OWN DMA order for the GPT-OSS giga-kernel decode graph, as a
    # constraint JSON, instead of applying one.
    "VLLM_NEURON_MEGA_DMA_ORDER_DERIVE": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_MEGA_DMA_ORDER_DERIVE")) or False
    ),
    # Path of a DMA-order constraint JSON to apply to that graph. Unset or "" applies none.
    "VLLM_NEURON_MEGA_DMA_ORDER_CONSTRAINTS_FILE": lambda: os.getenv(
        "VLLM_NEURON_MEGA_DMA_ORDER_CONSTRAINTS_FILE"
    ),
    # Pair the DMA-order constraint file with the BIR SBUF allocator, so the pass reads the
    # file's own ``sbuf_alloc_limit``. ON by default; set the variable to 0 to turn it off.
    "VLLM_NEURON_MEGA_SBUF_ALLOC": lambda: (
        maybe_convert_bool(os.getenv("VLLM_NEURON_MEGA_SBUF_ALLOC")) is not False
    ),
}


def _force_attn_tkg_no_swap() -> None:
    """nkilib's attention_tkg QK-swap MM1 path (packed FP8 KV) has a
    vector-DGE out-of-bounds defect that swap-on/off testing has not fully
    cleared. Kept off unconditionally rather than gated by a vllm_neuron
    flag, so no CI manifest/config has to opt in.

    nkilib's attention_tkg reads NKILIB_EXPERIMENTAL_ATTN_TKG_NO_SWAP as a
    module-level constant the moment attention_tkg_utils is first imported,
    before any vLLM CLI arg or NeuronConfig is parsed. That import happens
    while vllm_neuron/model/registry.py loads the model modules (which import
    attention_decode -> attention_tkg_utils), so setting the flag from any
    model's own code, or from anywhere reachable only after the registry
    loads, is always too late. This module is imported at the top of
    vllm_neuron/__init__.py, before vllm_neuron.model.registry is ever
    touched, so this is the earliest point in this package where the write is
    guaranteed to land before that import chain.

    setdefault keeps an explicit value from the environment for every engine
    except one. The GPT-OSS decode giga-kernel is correct only on the kernel's
    default QK-swap path, so NeuronPlatform.check_and_update_config always sets
    the variable to "0" for that engine before its worker processes start. An
    explicit "1" from the environment does not reach the giga-kernel decode.
    """
    os.environ.setdefault("NKILIB_EXPERIMENTAL_ATTN_TKG_NO_SWAP", "1")


_force_attn_tkg_no_swap()


def __getattr__(name: str) -> Any:
    """
    Gets environment variables lazily.

    Falls through to libtorch_neuronx_lite.envs for NEURON_LIBTORCH_* names
    so callers can access libtorch env vars via this module.

    Args:
        name: Name of the environment variable to retrieve

    Returns:
        Value of the environment variable (type depends on variable)

    Raises:
        AttributeError: If environment variable name is not defined

    Examples:
        >>> import vllm_neuron.envs as envs
        >>> cpu_mode = envs.VLLM_NEURON_CPU_MODE  # Returns bool
        >>> log_level = envs.VLLM_NEURON_LOG_LEVEL  # Returns str
        >>> envs.VLLM_NEURON_NONEXISTENT  # Raises AttributeError
    """
    if name in environment_variables:
        return environment_variables[name]()
    if name.startswith("NEURON_LIBTORCH_"):
        import libtorch_neuronx_lite.envs as libtorch_envs

        return getattr(libtorch_envs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """
    Return list of available environment variables.

    Returns:
        List of all defined environment variable names

    Examples:
        >>> import vllm_neuron.envs as envs
        >>> variables = dir(envs)
        >>> 'VLLM_NEURON_CPU_MODE' in variables
        True
    """
    return list(environment_variables.keys())


def is_set(name: str) -> bool:
    """
    Check if an environment variable is explicitly set.

    Args:
        name: Name of the environment variable to check

    Returns:
        True if environment variable is explicitly set, False otherwise

    Raises:
        AttributeError: If environment variable name is not defined

    Examples:
        >>> import os
        >>> import vllm_neuron.envs as envs
        >>> os.environ['VLLM_NEURON_CPU_MODE'] = '1'
        >>> envs.is_set('VLLM_NEURON_CPU_MODE')
        True
        >>> envs.is_set('VLLM_NEURON_LOG_LEVEL')  # Not set, uses default
        False
        >>> envs.is_set('VLLM_NEURON_NONEXISTENT')  # Raises AttributeError
    """
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@functools.lru_cache
def _valid_backend_values() -> tuple[str, ...]:
    """Return the route names that VLLM_NEURON_BACKEND accepts.

    The names come from `vllm_neuron.backend.NeuronBackend`, which is the one
    definition of the routes. A second copy of the names here can drift from
    that enum, so this function reads the enum instead.

    The import is local, because the `vllm_neuron` package imports this module.
    A module-level import makes an import cycle.

    Returns:
        The accepted values of VLLM_NEURON_BACKEND, in lower case.

    Examples:
        >>> import vllm_neuron.envs as envs
        >>> envs._valid_backend_values()
        ('vllm_neuron', 'neuron_native')
    """
    from vllm_neuron.backend import NeuronBackend

    return tuple(member.value.lower() for member in NeuronBackend)


# Short names that also select a route. `ci/dispatch.py` accepts the same 2
# aliases in `canonical_backend`, and `ci/buildspec/pre-build.sh` rewrites
# `neuron_native` to `neuron` for the image tag. That build alias reaches the
# container, because `Dockerfile.plugin` copies it into VLLM_NEURON_BACKEND. So
# this module must accept `neuron`, or a native image warns and takes the other
# route. Keep this map equal to `_BACKEND_ALIASES` in `ci/dispatch.py`.
_BACKEND_ALIASES = {"neuron": "neuron_native", "nxdi": "vllm_neuron"}


def _canonical_backend(value: str) -> str:
    """Return the route that a VLLM_NEURON_BACKEND value names.

    An alias maps to its route. Any other value passes through, so the caller
    still sees an invalid value and can warn about it.

    Args:
        value: The value of VLLM_NEURON_BACKEND, in lower case.

    Returns:
        The route name, or the value itself when the value is not an alias.

    Examples:
        >>> import vllm_neuron.envs as envs
        >>> envs._canonical_backend("neuron")
        'neuron_native'
        >>> envs._canonical_backend("nxdi")
        'vllm_neuron'
        >>> envs._canonical_backend("neuron_native")
        'neuron_native'
        >>> envs._canonical_backend("neuron_libtorch")
        'neuron_libtorch'
    """
    return _BACKEND_ALIASES.get(value, value)


@functools.lru_cache
def _warn_invalid_backend(value: str) -> None:
    """Write one warning for an invalid VLLM_NEURON_BACKEND value.

    `functools.lru_cache` holds the values that already got a warning, so each
    distinct value gets one warning only. This matters because some callers run
    `is_native_backend()` for each step, and one of them is a per-test fixture.
    A warning on every call floods the log.

    A test that asserts on the warning must call
    `envs._warn_invalid_backend.cache_clear()` first.

    Args:
        value: The invalid value of VLLM_NEURON_BACKEND, in lower case.

    Examples:
        >>> import vllm_neuron.envs as envs
        >>> envs._warn_invalid_backend.cache_clear()
        >>> envs._warn_invalid_backend("neuron_libtorch")
    """
    logger.warning(
        "VLLM_NEURON_BACKEND='%s' is not a valid route. The native route runs "
        "now. Set VLLM_NEURON_BACKEND to one of %s, or to an alias of %s. To "
        "use the default native route, remove the variable.",
        value,
        ", ".join(repr(name) for name in _valid_backend_values()),
        ", ".join(repr(name) for name in sorted(_BACKEND_ALIASES)),
    )


def is_native_backend() -> bool:
    """Return True for the native route.

    The native route is the default. Only VLLM_NEURON_BACKEND=vllm_neuron and
    its alias `nxdi` select the XLA route, which is on the deprecation path.

    This function accepts the 2 aliases of `_BACKEND_ALIASES`, so `neuron`
    selects the native route. A native image carries `VLLM_NEURON_BACKEND=neuron`
    in its own environment, because the build normalizes the route name for the
    image tag. Without the alias that image warns and then takes the XLA route.

    This function writes a warning for a value that is not a route and not an
    alias, because a silent fallback hides the mistake. One example is a
    `torch.compile` backend name such as `neuron_libtorch`. An unknown value
    gets the default route. `vllm_neuron.backend.get_backend()` rejects such a
    value, but the plugin entry point does not call that function, so this
    check is the only one that runs.

    Returns:
        True for the native route. False for the XLA route.

    Examples:
        >>> import os
        >>> import vllm_neuron.envs as envs
        >>> os.environ['VLLM_NEURON_BACKEND'] = 'neuron_native'
        >>> envs.is_native_backend()
        True
        >>> os.environ['VLLM_NEURON_BACKEND'] = 'vllm_neuron'
        >>> envs.is_native_backend()
        False
        >>> os.environ['VLLM_NEURON_BACKEND'] = 'neuron'
        >>> envs.is_native_backend()
        True
        >>> del os.environ['VLLM_NEURON_BACKEND']
        >>> envs.is_native_backend()
        True
    """
    value = os.getenv("VLLM_NEURON_BACKEND", "").lower()
    if not value:
        return True
    route = _canonical_backend(value)
    if route not in _valid_backend_values():
        _warn_invalid_backend(value)
        return True
    return route == "neuron_native"


def get_compile_backend_name() -> str:
    """Return the torch.compile backend name.

    The native route registers lite's backend on demand and returns its name.
    That name is intentionally not "neuron": "neuron" stays reserved for the
    torch_neuronx package so a separate install keeps upstream semantics.
    The XLA route uses "neuron_libtorch", registered at lite import.
    """
    if is_native_backend() and not environment_variables["VLLM_NEURON_CPU_MODE"]():
        from libtorch_neuronx_lite.compile.native_backend import register

        # register() is idempotent: repeated calls return the same name
        # without re-registering, so calling it from this getter is safe.
        return register()
    return "neuron_libtorch"


# The guard filter that each VLLM_NEURON_PRECOMPILE_GUARD_FILTER value selects.
# The value maps to a torch.compiler function name.
_GUARD_FILTERS: dict[str, str] = {
    "portable": "keep_portable_guards_unsafe",
    "skip_all": "skip_all_guards_unsafe",
}


def get_precompile_guard_filter_fn() -> Optional[Callable]:
    """Return the torch.compile guard filter. Return None to keep every guard.

    VLLM_NEURON_PRECOMPILE_GUARD_FILTER selects the filter. "off" or "" keeps
    every guard. "portable" keeps the tensor-metadata, dynamic-shape and
    global-context guards. "skip_all" drops every guard. A filter makes the
    warm start faster, because torch pickles fewer guards. A filter is safe on
    Neuron, because warmup runs every shape bucket.

    Examples:
        >>> if (fn := get_precompile_guard_filter_fn()) is not None:
        ...     compile_options = {"guard_filter_fn": fn}
    """
    value = environment_variables["VLLM_NEURON_PRECOMPILE_GUARD_FILTER"]()
    if value in ("", "off"):
        return None

    name = _GUARD_FILTERS.get(value)
    if name is None:
        logger.warning(
            "VLLM_NEURON_PRECOMPILE_GUARD_FILTER='%s' is not a valid filter. "
            "Every guard is kept. Set the variable to 'off', or to one of %s.",
            value,
            sorted(_GUARD_FILTERS),
        )
        return None

    import torch

    filter_fn = getattr(torch.compiler, name, None)
    if filter_fn is None:
        logger.warning(
            "torch.compiler.%s is not available in torch %s. Every guard is "
            "kept. Upgrade torch, or set "
            "VLLM_NEURON_PRECOMPILE_GUARD_FILTER=off.",
            name,
            torch.__version__,
        )
        return None

    logger.info("torch.compile drops guards with torch.compiler.%s", name)
    return filter_fn


def get_dist_backend() -> str:
    """Return the world process group's backend, which is Gloo on every route.

    Neuron model-parallel groups do not come from here; they hardcode Gloo in
    neuron_parallel_state.py. Collectives are traced into the NEFF rather than
    dispatched at runtime, so no route needs a "neuron" process group -- and
    asking for one would initialize the Neuron runtime, which a driverless
    compile host does not have.
    """
    return "gloo"


def get_kv_headroom_env() -> str | None:
    """Return the explicit headroom variable for either backend.

    An unset reserve keeps the memory-utilization and fractional-cap policy.
    Native, XLA, and CPU use the same variable. Zero enables headroom sizing.

    Returns:
        The active variable name, or None for fractional-cap sizing.

    Examples:
        >>> get_kv_headroom_env()  # doctest: +SKIP
        'VLLM_NEURON_RESERVED_HEADROOM_GIB'
    """
    module = sys.modules[__name__]
    if module.VLLM_NEURON_RESERVED_HEADROOM_GIB is not None:
        return "VLLM_NEURON_RESERVED_HEADROOM_GIB"
    return None


def get_neuron_compile_cache_dir() -> str:
    """Return the local compile cache directory derived from VLLM_CACHE_ROOT.

    If ``VLLM_CACHE_ROOT`` is explicitly set by the user, returns
    ``$VLLM_CACHE_ROOT/neuron/compile_cache`` unconditionally — the user's
    explicit choice is always respected.

    If ``VLLM_CACHE_ROOT`` is not set (default ``~/.cache/vllm``), probes
    the home directory filesystem type.  On multi-node clusters,the home directory
    can be NFS-mounted, which is incompatible with FileLock semantics.  In that case
    the function falls back to ``/tmp/vllm_neuron_wdir_$USER/neuron/compile_cache`` and logs a warning.

    Returns:
        str: Absolute path to the local Neuron compile cache directory.

    Examples:
        >>> import os
        >>> os.environ["VLLM_CACHE_ROOT"] = "/tmp/my_cache"
        >>> get_neuron_compile_cache_dir()
        '/tmp/my_cache/neuron/compile_cache'
    """
    # Cache by VLLM_CACHE_ROOT so the FS probe + NFS-fallback warning only
    # runs once per cache root per process.
    return _resolve_neuron_cache_subdir(
        os.environ.get("VLLM_CACHE_ROOT"), "neuron/compile_cache"
    )


@functools.lru_cache
def _resolve_neuron_cache_subdir(cache_root_override: Optional[str], leaf: str) -> str:
    """Resolve ``<cache_root>/<leaf>``, honoring VLLM_CACHE_ROOT with an NFS fallback.

    An explicit ``VLLM_CACHE_ROOT`` is honored unconditionally. Otherwise the
    default home location is used, with a ``/tmp`` fallback when home is on a
    remote filesystem (NFS or Lustre), which is incompatible with FileLock. The
    result is cached per (override, leaf), so the FS probe runs once each.
    """
    if cache_root_override is not None:
        return os.path.join(cache_root_override, leaf)

    # Probe ~ (guaranteed to exist), not ~/.cache/vllm (may not exist yet).
    if _path_is_remote_filesystem(os.path.expanduser("~")):
        fallback = os.path.join(os.path.expandvars("/tmp/vllm_neuron_wdir_$USER"), leaf)
        logger.warning(
            "Default cache path ~/.cache/vllm/%s is on a remote filesystem "
            "(NFS or Lustre). Falling back to %s.",
            leaf,
            fallback,
        )
        return fallback

    return os.path.join(os.path.expanduser("~/.cache/vllm"), leaf)


def _path_is_remote_filesystem(path: str) -> bool:
    """Return True if *path* is on a remote filesystem (NFS or Lustre).

    Args:
        path: An existing filesystem path to check.

    Returns:
        bool: True if the filesystem is NFS (any variant) or Lustre
        (including FSx for Lustre), False otherwise.
    """
    result = subprocess.check_output(
        ["stat", "-f", "-c", "%T", path],
        stderr=subprocess.STDOUT,
        text=True,
    )
    fs_type = result.strip().lower()
    return fs_type.startswith("nfs") or fs_type == "lustre"


def get_neuron_snapshot_dir() -> str:
    """Return the snapshot bundle root, derived like the compile cache dir.

    Snapshots reference the HLO/NEFF that live under the compile cache, so they
    resolve off the same ``VLLM_CACHE_ROOT`` base and inherit the same NFS/Lustre
    fallback. The two trees sit side by side (``neuron/compile_cache`` and
    ``neuron/snapshots``) under one root.

    Examples:
        >>> import os
        >>> os.environ["VLLM_CACHE_ROOT"] = "/tmp/my_cache"
        >>> get_neuron_snapshot_dir()
        '/tmp/my_cache/neuron/snapshots'
    """
    return _resolve_neuron_cache_subdir(
        os.environ.get("VLLM_CACHE_ROOT"), "neuron/snapshots"
    )


def get_neuron_torch_compile_cache_dir() -> str:
    """Return the TorchInductor cache root under ``VLLM_CACHE_ROOT``.

    The TorchInductor cache holds the FX-level artifacts of ``torch.compile``.
    It sits beside ``neuron/compile_cache`` and ``neuron/snapshots`` under one
    ``VLLM_CACHE_ROOT``. This function returns the shared root only. The caller
    adds the per-configuration and per-rank sub-path. See
    ``vllm_neuron/vllm/inductor_cache.py``.

    Returns:
        str: Absolute path to the TorchInductor cache root.

    Examples:
        >>> import os
        >>> os.environ["VLLM_CACHE_ROOT"] = "/tmp/my_cache"
        >>> get_neuron_torch_compile_cache_dir()
        '/tmp/my_cache/neuron/torch_compile'
    """
    return _resolve_neuron_cache_subdir(
        os.environ.get("VLLM_CACHE_ROOT"), "neuron/torch_compile"
    )
