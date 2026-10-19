# SPDX-License-Identifier: Apache-2.0
"""Scope and save the torch compile cache for the native backend.

This module does two things:

1. It points TORCHINDUCTOR_CACHE_DIR at a per-configuration, per-rank directory.
   See the VLLM_CACHE_ROOT row in AGENTS.md for the directory layout.
2. It saves the torch compile-cache artifacts once, after the last warmup
   bucket. A later warm start loads the package automatically. No explicit load
   call runs.

Both features are active on the native backend only, and only when torch
caching-precompile is on. The hook is inert otherwise.

Environment contract for the save and warm-start path:

- ``TORCH_CACHING_PRECOMPILE=1`` turns on torch caching-precompile. torch then
  sets ``torch._dynamo.config.caching_precompile``, which this module reads.
- ``TORCHINDUCTOR_AUTOGRAD_CACHE=1`` turns on the AOTAutograd cache, so the warm
  start reuses the autograd artifacts. torch reads the variable one time, into
  ``torch._functorch.config.enable_autograd_cache``, and that config is what this
  module reads. A component can set the config directly instead of the variable.
"""

import dataclasses
import hashlib
import json
import logging
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from vllm_neuron import envs
from vllm_neuron.utils.hardware_config import get_logical_nc_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)

# torch._dynamo.guards logs this at WARNING for every guard that caching_precompile
# drops, once per graph and rank: about 1,760 lines per rank on GPT-OSS 120B.
# Dropping these guards is how precompile works, so the line has nothing to act on.
_DROPPED_GUARD_MSG = "%s guard on %s is dropped with caching_precompile=True."
_GUARDS_LOGGER = "torch._dynamo.guards"


class _DroppedGuardToDebug(logging.Filter):
    """Demote torch's dropped-guard notice to DEBUG.

    The record keeps its text and passes when DEBUG is on for vllm_neuron
    (``VLLM_NEURON_LOG_LEVEL=DEBUG``) or for torch's guards logger
    (``TORCH_LOGS=+torch._dynamo.guards``).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg != _DROPPED_GUARD_MSG:
            return True
        record.levelno = logging.DEBUG
        record.levelname = logging.getLevelName(logging.DEBUG)
        return any(
            logging.getLogger(name).isEnabledFor(logging.DEBUG)
            for name in ("vllm_neuron", _GUARDS_LOGGER)
        )


def demote_dropped_guard_warnings() -> None:
    """Install the dropped-guard filter on torch's guards logger, once per process.

    Examples:
        >>> demote_dropped_guard_warnings()  # doctest: +SKIP
    """
    guards_logger = logging.getLogger(_GUARDS_LOGGER)
    if not any(isinstance(f, _DroppedGuardToDebug) for f in guards_logger.filters):
        guards_logger.addFilter(_DroppedGuardToDebug())


_ENV_VAR = "TORCHINDUCTOR_CACHE_DIR"

# Length of the hex fingerprint. 128 bits (like the neuronx-cc cache key) make
# a collision negligible across a fleet that shares a cache root.
_FINGERPRINT_LEN = 32


def _platform_target() -> str:
    """Return the compile platform target, or "cpu" in CPU mode."""
    # get_platform_target() reads the Neuron runtime, which a CPU-mode host
    # does not have. Return a fixed token there.
    if envs.VLLM_NEURON_CPU_MODE:
        return "cpu"
    from libtorch_neuronx_lite.compile.platform import get_platform_target

    return get_platform_target()


def _plugin_version() -> str:
    """Return the installed vllm-neuron version, or "unknown" if not installed."""
    # importlib.metadata raises for a distribution that pip did not register,
    # for example an uninstalled editable checkout.
    try:
        return version("vllm-neuron")
    except PackageNotFoundError:
        return "unknown"


def _versions() -> dict[str, str]:
    """Return the framework, compiler, and plugin versions that gate reuse."""
    versions = {"torch": torch.__version__, "vllm_neuron": _plugin_version()}
    # Neuron compiler and Lite are not present in CPU mode. The platform target
    # is already "cpu" there, so a compiler version adds no information.
    if envs.VLLM_NEURON_CPU_MODE:
        versions["libtorch_neuronx_lite"] = "cpu"
        versions["neuronxcc"] = "cpu"
        versions["nki"] = "cpu"
        return versions
    from libtorch_neuronx_lite.compile.platform import (
        get_libtorch_neuronx_lite_version,
        get_neuronxcc_version,
        get_nki_version,
    )

    versions["libtorch_neuronx_lite"] = get_libtorch_neuronx_lite_version()
    versions["neuronxcc"] = get_neuronxcc_version()
    # nki ships as its own wheel and drives kernel codegen; DiskDynamoCache keys
    # only on the source id, so an nki-only bump must give a new fingerprint.
    versions["nki"] = get_nki_version()
    return versions


def _config_field_names(config: Any) -> list[str]:
    """Return the field names of a dataclass config, or its instance attrs."""
    if dataclasses.is_dataclass(config):
        return [f.name for f in dataclasses.fields(config)]
    return list(vars(config))


def _collect_buckets(*configs: Any) -> dict[str, Any]:
    """Return every resolved ``*_buckets`` field across the given configs.

    Collect by the ``_buckets`` name suffix so a new bucket field is picked up
    automatically. A missing config is skipped. Sort the keys and each value: a
    bucket set is order-insensitive by design (the sizes, not their order, drive
    the compile), so two configs differing only in bucket order share a
    fingerprint, and the value stays stable even if a field is an unordered
    iterable (e.g. a set).
    """
    buckets: dict[str, Any] = {}
    for config in configs:
        if config is None:
            continue
        for name in _config_field_names(config):
            if not name.endswith("_buckets"):
                continue
            value = getattr(config, name)
            buckets[name] = sorted(value) if value is not None else None
    return dict(sorted(buckets.items()))


def _compiler_options(vllm_config: "VllmConfig") -> dict[str, Any]:
    """Return the torch.compile and neuronx-cc options that affect cache reuse.

    Reads the LNC degree from the env, not the runner's compile_options: that
    dict is built later, inside load_model, so it is empty here. neuronx-cc keys
    the NEFF cache on the canonicalized compiler args (including the LNC flag),
    so a new degree must give a new fingerprint.
    """
    opt_level = getattr(vllm_config, "optimization_level", None)
    return {
        "backend": envs.get_compile_backend_name(),
        "optimization_level": getattr(opt_level, "value", opt_level),
        "guard_filter": envs.VLLM_NEURON_PRECOMPILE_GUARD_FILTER,
        "fullgraph": not envs.VLLM_NEURON_DEBUG_MODE,
        "logical_nc_config": get_logical_nc_config(),
    }


def resolve_dp_index(parallel_config: Any) -> int:
    """Return the true DP replica index from a ParallelConfig.

    Prefer ``data_parallel_index``: vLLM's ``reconfigure_for_independent_dp_rank``
    zeroes ``data_parallel_rank`` and drops ``data_parallel_size`` to 1 for dense
    (non-MoE) DP engines, so neither the rank nor ``compute_hash()`` identifies
    the replica. A bool is rejected because ``isinstance(True, int)`` is True.

    Examples:
        >>> from types import SimpleNamespace
        >>> resolve_dp_index(SimpleNamespace(data_parallel_index=2))
        2
        >>> resolve_dp_index(
        ...     SimpleNamespace(data_parallel_index=None, data_parallel_rank=1))
        1
    """
    dp_index = getattr(parallel_config, "data_parallel_index", None)
    if not isinstance(dp_index, int) or isinstance(dp_index, bool):
        dp_index = getattr(parallel_config, "data_parallel_rank", 0)
    return dp_index


def _neff_cache_dir() -> str:
    """Return the directory that holds the native NEFF artifacts.

    A precompile package names its NEFFs by cache key and does not hold them, and
    the 2 caches have separate roots, so a run that keeps the package directory and
    moves the NEFF directory restores a package whose NEFFs are absent. Making this
    a fingerprint input compiles instead. See P524534517.

    Ask lite rather than reading an env var: the location has 3 sources
    (``TORCH_NEURONX_NEFF_CACHE_DIR``, ``NEURON_LIBTORCH_CACHE_ROOT``, and
    ``VLLM_CACHE_ROOT``). Pass the native backend name, since lite maps its XLA
    name to a different directory. Returns "" when lite is absent, which keeps the
    fingerprint stable in CPU mode.
    """
    try:
        from libtorch_neuronx_lite.envs import NATIVE_BACKEND_NAME, get_neff_cache_dir

        return get_neff_cache_dir(NATIVE_BACKEND_NAME)
    except Exception:  # noqa: BLE001 - a fingerprint must not break a startup
        logger.debug("Cannot resolve the NEFF cache directory for the fingerprint.")
        return ""


def build_fingerprint_inputs(
    vllm_config: "VllmConfig",
    neuron_config: Any,
    encoder_neuron_config: Any = None,
) -> dict[str, Any]:
    """Assemble every input that the compile-cache fingerprint covers.

    The inputs are the vLLM config hash, the resolved bucket sets, the platform
    target, the compiler options, the versions, and the NEFF cache directory. The
    bucket sets cover the Neuron config and the encoder config. The versions cover
    the framework, the compiler, and the plugin. The result is rank-independent, so
    every rank derives the same fingerprint.

    Examples:
        >>> inputs = build_fingerprint_inputs(vllm_config, neuron_config)  # doctest: +SKIP
        >>> set(inputs) == {"config_hash", "buckets", "platform_target",
        ...                  "compiler_options", "versions", "neff_cache_dir"}  # doctest: +SKIP
        True
    """
    return {
        "config_hash": vllm_config.compute_hash(),
        "buckets": _collect_buckets(neuron_config, encoder_neuron_config),
        "platform_target": _platform_target(),
        "compiler_options": _compiler_options(vllm_config),
        "versions": _versions(),
        "neff_cache_dir": _neff_cache_dir(),
    }


def compute_fingerprint(inputs: dict[str, Any]) -> str:
    """Return a deterministic hex fingerprint for the compile inputs.

    Serialize with sorted keys so an equivalent input dict always gives the same
    fingerprint. An input that is not JSON-serializable raises, so a
    non-deterministic type cannot slip in silently.

    Examples:
        >>> compute_fingerprint({"a": 1}) == compute_fingerprint({"a": 1})
        True
        >>> compute_fingerprint({"a": 1}) != compute_fingerprint({"a": 2})
        True
    """
    payload = json.dumps(inputs, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:_FINGERPRINT_LEN]


def resolve_cache_dir(fingerprint: str, dp_index: int, tp_rank: int) -> Path:
    """Return the cache directory for one fingerprint, DP replica, and TP rank.

    Examples:
        >>> str(resolve_cache_dir("abc123", 1, 3)).endswith(
        ...     "neuron/torch_compile/abc123/dp_1/rank_3")
        True
    """
    root = Path(envs.get_neuron_torch_compile_cache_dir())
    return root / fingerprint / f"dp_{dp_index}" / f"rank_{tp_rank}"


def configure_torchinductor_cache_dir(
    vllm_config: "VllmConfig",
    neuron_config: Any,
    dp_index: int,
    tp_rank: int,
    encoder_neuron_config: Any = None,
) -> str:
    """Set TORCHINDUCTOR_CACHE_DIR for this worker before the first compile.

    Call this once per worker, before the first ``torch.compile()``. It computes
    the fingerprint, creates the per-rank directory, and sets the env var.

    Args:
        vllm_config: The engine configuration.
        neuron_config: The resolved Neuron configuration (carries the buckets).
        dp_index: The data-parallel replica index of this worker.
        tp_rank: The tensor-parallel rank of this worker.
        encoder_neuron_config: The resolved encoder configuration, or None.

    Returns:
        str: The absolute cache directory that the env var now points at.

    Examples:
        >>> configure_torchinductor_cache_dir(  # doctest: +SKIP
        ...     vllm_config, neuron_config, dp_index=0, tp_rank=0)
        '/home/user/.cache/vllm/neuron/torch_compile/<fingerprint>/dp_0/rank_0'
    """
    inputs = build_fingerprint_inputs(vllm_config, neuron_config, encoder_neuron_config)
    fingerprint = compute_fingerprint(inputs)
    cache_dir = resolve_cache_dir(fingerprint, dp_index, tp_rank)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # TorchInductor reads this process-global env var on the first
    # torch.compile(). Call this once per worker process, before that compile.
    os.environ[_ENV_VAR] = str(cache_dir)
    logger.info(
        "Set %s=%s (fingerprint=%s, dp=%d, rank=%d)",
        _ENV_VAR,
        cache_dir,
        fingerprint,
        dp_index,
        tp_rank,
    )
    log_precompile_state()
    demote_dropped_guard_warnings()
    return str(cache_dir)


def precompile_state() -> dict[str, object]:
    """Return what decides whether a warm start can reuse compiled graphs.

    Every field is read at call time. ``precompile`` is the torch flag, which
    torch reads once from ``TORCH_CACHING_PRECOMPILE`` when ``torch._dynamo`` is
    imported, so setting the variable later has no effect and this reports the
    value that is actually in force.

    Examples:
        >>> sorted(precompile_state())  # doctest: +SKIP
        ['autograd_cache', 'cache_dir', 'precompile', 'will_save']
    """
    return {
        "precompile": caching_precompile_enabled(),
        "autograd_cache": autograd_cache_enabled(),
        "cache_dir": os.environ.get(_ENV_VAR, ""),
        "will_save": should_save_compile_cache(),
    }


def autograd_cache_enabled() -> bool:
    """Return True when the AOTAutograd cache is on.

    Read ``torch._functorch.config.enable_autograd_cache``, not the environment.
    torch reads ``TORCHINDUCTOR_AUTOGRAD_CACHE`` one time into that config, so the
    config holds the value in force. A component can also set the config
    directly: libtorch-neuron-lite does that and never sets the variable. A read
    of the variable alone then reports off on every native run, and the caller
    logs a warning that does not apply.

    Fall back to the variable when the config has no such field, which keeps an
    older torch working.

    Returns:
        bool: True when the AOTAutograd cache is on.

    Examples:
        >>> autograd_cache_enabled()  # doctest: +SKIP
        True
    """
    config = getattr(getattr(torch, "_functorch", None), "config", None)
    if config is not None and hasattr(config, "enable_autograd_cache"):
        return bool(config.enable_autograd_cache)
    return os.environ.get("TORCHINDUCTOR_AUTOGRAD_CACHE", "0") == "1"


def log_precompile_state() -> dict[str, object]:
    """Log the precompile state once, whether it is on or off.

    Logged unconditionally, which is the point. Nothing else in a log says
    precompile is OFF: the cache directory is set on every native run whether or
    not precompile is active, and the save hook prints only when it is active, so
    its absence is ambiguous. Without this line the only clue is a torch guard
    warning that happens to name caching_precompile, buried in a few thousand
    others.

    Returns the same mapping it logged, so a caller can assert on it.
    """
    state = precompile_state()
    logger.info(
        "Precompile: %s (autograd_cache=%s, will_save=%s, cache_dir=%s)",
        "ON" if state["precompile"] else "OFF",
        "on" if state["autograd_cache"] else "off",
        state["will_save"],
        state["cache_dir"] or "unset",
    )
    if state["precompile"] and not state["autograd_cache"]:
        # Off records no backend artifact, the package saves bypassed=True, and
        # the warm start retraces everything. Name the config in the action: an
        # export of TORCHINDUCTOR_AUTOGRAD_CACHE has no effect after torch reads
        # it, so a caller that starts late must set the config.
        logger.warning(
            "Precompile is on and the AOTAutograd cache is off, so no backend "
            "artifact is recorded and a warm start retraces. Set "
            "torch._functorch.config.enable_autograd_cache, or export "
            "TORCHINDUCTOR_AUTOGRAD_CACHE=1 before the first torch import."
        )
    return state


def caching_precompile_enabled() -> bool:
    """Return True when torch caching-precompile is active.

    torch reads ``TORCH_CACHING_PRECOMPILE`` and sets
    ``torch._dynamo.config.caching_precompile``. The save hook and the warm start
    both depend on this flag.

    Examples:
        >>> caching_precompile_enabled()  # doctest: +SKIP
        True
    """
    return bool(getattr(torch._dynamo.config, "caching_precompile", False))


def should_save_compile_cache() -> bool:
    """Return True when the worker must save the torch compile cache.

    The gate is the native backend, not CPU mode, and active caching-precompile.
    The save hook is inert when any condition is False.

    Examples:
        >>> should_save_compile_cache()  # doctest: +SKIP
        True
    """
    return (
        envs.is_native_backend()
        and not envs.VLLM_NEURON_CPU_MODE
        and caching_precompile_enabled()
    )


# PT2 cache telemetry lives in torch._dynamo.utils.counters, a nested counter.
# These keys hold the Dynamo precompile cache and the AOTAutograd cache hit,
# miss, error, and save counts. DynamoCache.load() records three outcomes: a hit,
# an error when the package exists but does not deserialize, and a miss when none
# exists. The error count matters because a bad load is not a cold start.
_CACHE_COUNTER_KEYS = (
    ("dynamo_cache", "dynamo_cache_hit"),
    ("dynamo_cache", "dynamo_cache_miss"),
    ("dynamo_cache", "dynamo_cache_error"),
    ("aot_autograd", "autograd_cache_hit"),
    ("aot_autograd", "autograd_cache_miss"),
    ("aot_autograd", "autograd_cache_bypass"),
    ("aot_autograd", "autograd_cache_saved"),
)


def collect_cache_counters() -> dict[str, int]:
    """Return the PT2 cache hit, miss, bypass, and save counts.

    The keys are ``<group>.<counter>``, for example ``dynamo_cache.dynamo_cache_hit``.
    A counter that torch did not record reads as 0.

    Examples:
        >>> counts = collect_cache_counters()  # doctest: +SKIP
        >>> counts["dynamo_cache.dynamo_cache_hit"]  # doctest: +SKIP
        4
    """
    from torch._dynamo.utils import counters

    return {
        f"{group}.{key}": counters[group][key] for group, key in _CACHE_COUNTER_KEYS
    }


def _count_saved_artifacts(info: Any) -> int:
    """Return the total artifact count in a torch CacheInfo object.

    CacheInfo holds one field, ``artifacts``, mapping kind to a list of keys. The
    per-kind attributes are properties over it, so sum the mapping not the fields.

    Examples:
        >>> _count_saved_artifacts(None)
        0
    """
    artifacts = getattr(info, "artifacts", None)
    if not isinstance(artifacts, dict):
        return 0
    return sum(len(keys) for keys in artifacts.values())


def save_compile_cache_artifacts() -> int:
    """Save the torch compile-cache artifacts once, and log the cache counts.

    Call this one time per worker, after the last warmup bucket. torch writes a
    precompile package under TORCHINDUCTOR_CACHE_DIR. A later warm start loads the
    package automatically, so no explicit load call is necessary.

    The log line reports the save count and the hit, miss, and error counts. It
    prints on a cold start and on a warm start, so the cache lifecycle is visible.

    Returns:
        int: The number of packages or artifacts that torch saved. 0 means torch
        saved nothing, for example when no compile ran.

    Examples:
        >>> save_compile_cache_artifacts()  # doctest: +SKIP
        128
    """
    # Best-effort save. A disk or torch error must not stop the model from
    # serving, so catch it, log it, and continue with a save count of 0.
    try:
        if caching_precompile_enabled():
            # Two calls, deliberately. The public one does the save, so a rename
            # in the private path costs the count and not the package. Its return
            # value is CacheArtifactManager state, which precompile never fills,
            # so it always reads 0; the private call gives the real count and is
            # idempotent.
            saved = 0
            torch.compiler.save_cache_artifacts()
            try:
                from torch._dynamo.precompile_context import PrecompileContext

                saved = len(PrecompileContext.save_to_dynamo_cache().get("dynamo", ()))
            except Exception:
                logger.warning(
                    "Saved the torch compile cache but could not count the "
                    "packages; the reported count is 0.",
                    exc_info=True,
                )
        else:
            result = torch.compiler.save_cache_artifacts()
            saved = 0
            if result is not None:
                # save_cache_artifacts() returns (serialized_bytes, cache_info).
                # Nothing consumes the bytes, because the package is on disk.
                _serialized_bytes, cache_info = result
                saved = _count_saved_artifacts(cache_info)
    except Exception:
        logger.warning("Failed to save the torch compile cache", exc_info=True)
        return 0
    # Read the counters after the save, so autograd_cache_saved reflects it.
    logger.info(
        "Saved torch compile cache: %d artifacts. Cache counters: %s",
        saved,
        collect_cache_counters(),
    )
    return saved
