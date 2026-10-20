# SPDX-License-Identifier: Apache-2.0
"""Make ``torch.accelerator.empty_cache()`` survive Neuron's allocator.

vLLM's ``cleanup_dist_env_and_memory`` (every engine shutdown) calls
``torch.accelerator.empty_cache()`` on all non-CPU platforms. torch asserts
there that the active accelerator's allocator is a ``DeviceAllocator``
(``CachingDeviceAllocator.h``: "Allocator for neuron is not a
DeviceAllocator") — Neuron's privateuse1 backend registers a plain allocator,
so the assert fires as an internal-assert ``RuntimeError``.

The crash is worse than a failed test: teardown aborts before the worker
releases its Neuron cores, so every subsequent engine or server in the same
pod fails NRT initialization ("The PyTorch Neuron Runtime could not be
initialized"), cascading one teardown crash into whole-session failures.

There is nothing to flush — Neuron device memory is not managed by a torch
caching allocator — so swallowing exactly this assert is behavior-preserving.
Any other error still propagates, and platforms with a real caching allocator
are untouched because their call succeeds. Applied at import time so it
survives spawn-mode re-imports in EngineCore subprocesses.
"""

import functools
import logging

import torch

logger = logging.getLogger(__name__)

_applied = False


def apply_accelerator_empty_cache_patch() -> None:
    global _applied
    if _applied:
        return
    _applied = True

    accelerator = getattr(torch, "accelerator", None)
    orig = getattr(accelerator, "empty_cache", None)
    if orig is None:
        return

    @functools.wraps(orig)
    def _allocator_safe_empty_cache() -> None:
        try:
            orig()
        except RuntimeError as e:
            if "is not a DeviceAllocator" in str(e):
                logger.debug(
                    "Skipped torch.accelerator.empty_cache(): no torch "
                    "caching allocator on this accelerator (%s)",
                    e,
                )
                return
            raise

    torch.accelerator.empty_cache = _allocator_safe_empty_cache
    # torch.accelerator re-exports from torch.accelerator.memory; keep direct
    # users of the submodule consistent with the package attribute.
    memory_mod = getattr(accelerator, "memory", None)
    if memory_mod is not None and getattr(memory_mod, "empty_cache", None) is orig:
        memory_mod.empty_cache = _allocator_safe_empty_cache
    logger.debug("Applied allocator-safe torch.accelerator.empty_cache patch")
