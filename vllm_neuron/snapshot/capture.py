# SPDX-License-Identifier: Apache-2.0
"""Drive input-snapshot capture from the lite execution hook.

One :class:`SnapshotCapturer` is shared by every executable in the process; the
lite backend passes ``ExecuteMetadata`` on each forward so the capturer can tell
which NEFF it is running for. Bundles are written to the snapshot dir keyed by
the graph's compilation id, ``output_dir/<neff_id>/rank<N>/call<M>``. The
per-rank subdir keeps workers sharing a compilation from colliding.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional, Sequence

from vllm_neuron.snapshot.config import get_snapshot_config

if TYPE_CHECKING:
    from libtorch_neuronx_lite.compile.execute_context import ExecuteMetadata
    from vllm_neuron.snapshot.config import CaptureSelector

logger = logging.getLogger(__name__)


class WorkerRanks(NamedTuple):
    """This worker's ranks. ``global_rank`` keys the output dir; ``tp``/``dp``
    are descriptive labels recorded in meta.json.
    """

    global_rank: int
    tp: int
    dp: int


@dataclass(frozen=True)
class CaptureScope:
    """Process-constant capture settings, resolved once: which rank this worker
    is, plus the selector, budget, and format shared by every executable.
    """

    global_rank: int
    tp_rank: int
    dp_rank: int
    selector: "CaptureSelector"
    max_captures: int
    fmt: str
    output_dir: str


def resolve_capture_scope() -> Optional[CaptureScope]:
    """Resolve this worker's capture scope, or ``None`` when it does not capture.

    ``None`` when capture is disabled or this worker's tp-rank is out of the
    configured set. Resolved lazily on the first real forward (not at import),
    so the enable flag and the parallel-state ranks are both settled by then.
    """
    config = get_snapshot_config()
    if not config.is_active():
        return None

    try:
        ranks = current_ranks()
    except Exception as exc:  # defensive: never fail a forward over labeling
        logger.warning("Snapshot disabled: %s", exc)
        return None
    # RANKS filters on tp_rank, not global rank: with DP>1 a value like [0]
    # selects tp-rank 0 in every DP group (several worker processes), each still
    # writing to its own global-rank directory.
    if config.ranks is not None and ranks.tp not in config.ranks:
        return None

    return CaptureScope(
        global_rank=ranks.global_rank,
        tp_rank=ranks.tp,
        dp_rank=ranks.dp,
        selector=config.selector,
        max_captures=config.max_captures,
        fmt=config.fmt,
        output_dir=config.output_dir,
    )


class SnapshotCapturer:
    """Shared input capturer installed as the lite pre-execute hook.

    Keeps a per-NEFF call index keyed by the compilation id lite passes, so one
    shared instance counts each graph's forwards independently. Owns the whole
    capture decision: context lookup, selection, budget, the tensor write, and
    meta.json. ``pre_execute`` runs just before the NEFF executes.

    Capture requires synchronous scheduling (enforced at startup), so forwards
    do not overlap and the call-index table is touched by one forward at a time.
    """

    def __init__(self) -> None:
        self._call_index: Dict[str, int] = {}
        self._scope: Optional[CaptureScope] = None
        self._scope_resolved = False

    def _capture_scope(self) -> Optional[CaptureScope]:
        """Resolve the process-constant scope once, on first use."""
        if not self._scope_resolved:
            self._scope_resolved = True
            self._scope = resolve_capture_scope()
        return self._scope

    def pre_execute(self, inputs: Sequence[Any], metadata: "ExecuteMetadata") -> None:
        """Serialize ``inputs`` for this forward when it is selected.

        A no-op when no forward context is published (warmup, or a runner that
        does not publish one), so the per-NEFF index counts only real forwards.
        When a context is present the index for this NEFF advances, then the OR
        of the per-forward token/request verdict and this NEFF's call-index rule
        (bounded by the process-global budget) decides whether to write.
        """
        from vllm_neuron.snapshot.context import (
            get_current_forward,
            try_consume_capture_budget,
        )

        ctx = get_current_forward()
        if ctx is None:
            return

        scope = self._capture_scope()
        if scope is None:
            return

        neff_id = metadata.neff_id
        call_index = self._call_index.get(neff_id, 0)
        self._call_index[neff_id] = call_index + 1

        reasons = {match.reason for match in ctx.matches} if ctx.capture else set()
        if scope.selector.call_index_match(call_index):
            reasons.add("call_index")
        selected_by = sorted(reasons)
        if not selected_by or not try_consume_capture_budget(scope.max_captures):
            return

        import torch

        from vllm_neuron.snapshot.meta import write_call_meta

        # Write under the snapshot dir keyed by this graph's compilation id, in a
        # per-global-rank subdir so workers sharing a compilation never collide.
        fmt = scope.fmt
        call_dir = os.path.join(
            scope.output_dir,
            neff_id,
            f"rank{scope.global_rank}",
            f"call{call_index}",
        )
        logger.info(
            "Snapshot: capturing %s (%s) tensors=%d",
            call_dir,
            ",".join(selected_by),
            len(inputs),
        )

        # Copy the inputs to host before execute so the captured bytes are
        # exactly what this forward will run. Only holds under sync scheduling
        # (enforced at startup); async could overwrite the buffers first.
        torch.ops.neuron.write_tensors(inputs, call_dir, fmt)

        # Record dtype/shape so the on-disk bytes can be reinterpreted on replay
        # (bf16/fp8 .npy carry no numpy type label).
        input_specs = [
            {"index": i, "dtype": str(t.dtype), "shape": list(t.shape)}
            for i, t in enumerate(inputs)
        ]

        # meta.json last so a complete bundle (tensors + meta) exists once it is
        # written; if this raises, the tensors are on disk but the run aborts.
        write_call_meta(
            call_dir,
            compilation_hash=neff_id,
            fmt=fmt,
            global_rank=scope.global_rank,
            tp_rank=scope.tp_rank,
            dp_rank=scope.dp_rank,
            call_index=call_index,
            selected_by=selected_by,
            inputs=input_specs,
            context=ctx,
        )


def current_ranks() -> WorkerRanks:
    """Return this worker's ranks, best-effort.

    ``global_rank`` (unique per process) keys the output dir; ``tp``/``dp`` are
    descriptive. An unresolved rank falls back to 0 — the dir stays unique via
    ``global_rank``, so a mislabel cannot cause a clobber.
    """
    return WorkerRanks(
        global_rank=_global_rank(),
        tp=_group_rank("get_tp_group"),
        dp=_group_rank("get_dp_group"),
    )


def _global_rank() -> int:
    """This worker's global rank, or 0 when not running distributed."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def _group_rank(getter_name: str) -> int:
    """Rank within a vLLM parallel group, or 0 when the group is unavailable."""
    try:
        import vllm.distributed.parallel_state as parallel_state

        return getattr(parallel_state, getter_name)().rank_in_group
    except Exception:
        return 0
