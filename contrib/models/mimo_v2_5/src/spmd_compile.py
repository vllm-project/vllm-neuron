# SPDX-License-Identifier: Apache-2.0
"""Compile each SPMD graph once per server rather than once per rank.

On the native route every rank traces and submits its own graphs. This model's
graphs are SPMD (every per-rank offset comes from a stamped rank tensor), so the
submissions of all ranks are byte-identical and share one NEFF cache key, but
they arrive at the compilation service at the same moment, before any NEFF
exists, and the service compiles every one of them (seen as several identical
neuronx-cc jobs at once; on Qwen3.5-397B at TP=64 one decode graph was compiled
64 times, 8 at a time).

Here rank 0 submits as usual and the other ranks hold each submission back until
rank 0's NEFF for that key is in the persistent cache, then submit and hit it.
Rank 0 marks each key it submits, so a graph that rank 0 does not compile (not
SPMD) is only held back for ``MIMO_V2_5_SPMD_COMPILE_LEAD_S`` (default 30 min).

``MIMO_V2_5_SPMD_LEADER_COMPILE=0`` turns it off.
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

_LEAD_S = float(os.environ.get("MIMO_V2_5_SPMD_COMPILE_LEAD_S", 1800))
_WAIT_S = float(os.environ.get("MIMO_V2_5_SPMD_COMPILE_WAIT_S", 12 * 3600))
_POLL_S = 10.0


def install_leader_compile() -> None:
    """Wrap the native backend's MLIR submission (idempotent)."""
    if os.environ.get("MIMO_V2_5_SPMD_LEADER_COMPILE", "1") == "0":
        return
    try:
        from libtorch_neuronx_lite._compiler.neuron_dynamo_backend import compile as ndc
    except ImportError:  # CPU-only installs
        return
    if getattr(ndc._submit_mlir_bytes, "_mimo_v2_5_leader", False):
        return
    original = ndc._submit_mlir_bytes

    def submit(mlir_bytes, *, has_collectives, use_tensorizer=False):
        _hold_until_leader_compiled(ndc._content_key(mlir_bytes))
        return original(
            mlir_bytes, has_collectives=has_collectives, use_tensorizer=use_tensorizer
        )

    submit._mimo_v2_5_leader = True
    ndc._submit_mlir_bytes = submit


def _rank() -> int | None:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return None
    return dist.get_rank()


def _hold_until_leader_compiled(key: str) -> None:
    rank = _rank()
    if rank is None:
        return
    from libtorch_neuronx_lite.compile.native_backend import _key_index_dir

    index_dir = _key_index_dir(key)
    marker = os.path.join(
        index_dir.split(os.sep + "cache_key_index" + os.sep)[0], "spmd_submitted", key
    )
    if rank == 0:
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        open(marker, "w").close()
        return

    t0 = time.monotonic()
    while not os.path.exists(marker):
        if time.monotonic() - t0 > _LEAD_S:
            logger.info("SPMD compile: rank 0 never submitted %s; compiling here", key[:12])
            return
        time.sleep(_POLL_S)
    while not (os.path.isdir(index_dir) and os.listdir(index_dir)):
        if time.monotonic() - t0 > _WAIT_S:
            logger.warning("SPMD compile: no NEFF from rank 0 for %s; compiling here", key[:12])
            return
        time.sleep(_POLL_S)
    logger.info(
        "SPMD compile: rank %d reuses rank 0's NEFF for %s after %.0f s",
        rank, key[:12], time.monotonic() - t0,
    )
