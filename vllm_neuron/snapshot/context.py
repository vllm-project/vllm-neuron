# SPDX-License-Identifier: Apache-2.0
"""Per-forward request identity, published by the model runner for capture.

The captured tensors live at the NRT boundary as an anonymous positional
vector; the request/token identity that makes a snapshot answerable for an
accuracy regression ("request X, token 312") exists only in the model runner,
above the compiled boundary that passes tensors but not Python objects. The
runner and the compiled executable run on the same thread within one
synchronous forward, so the runner evaluates the token/request verdict and
publishes it (with identity) into this process-global holder before calling the
model; each executable reads it while deciding whether to dump and what to tag.

The token/request verdict is per-forward (the same for every NEFF that forward),
so it is resolved once here at the top; the call-index rule is per-NEFF and
stays in the executable. The capture budget is process-global so a broad rule
cannot dump without bound.

Kept import-light (no torch/numpy) because the worker touches this on every
forward, including capture-disabled ones.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotMatch:
    """One batch row that satisfied a token/request rule."""

    batch_index: int
    request_id: str
    position: int
    reason: str  # "token" or "request"


@dataclass(frozen=True)
class SnapshotForwardContext:
    """Identity and capture verdict for one forward.

    ``capture`` is the resolved token/request verdict for this forward, shared
    by every NEFF in it. ``matches`` records which rows triggered it (for the
    metadata tag). ``req_ids`` / ``positions`` are recorded as the runner
    assembles them: on decode there is one position per request; on prefill a
    single request owns the position range. ``global_step`` is shared by every
    NEFF in the same forward so per-rank, per-NEFF dumps line up to one step.
    """

    global_step: int
    is_prompt: bool
    capture: bool
    req_ids: List[str] = field(default_factory=list)
    positions: List[int] = field(default_factory=list)
    matches: List[SnapshotMatch] = field(default_factory=list)


_current: Optional[SnapshotForwardContext] = None
_step_counter = itertools.count()
_captures_done = 0
_budget_exhausted_logged = False


def next_global_step() -> int:
    """Monotonic per-process forward id, advanced once per published forward."""
    return next(_step_counter)


def set_current_forward(ctx: Optional[SnapshotForwardContext]) -> None:
    """Publish the verdict + identity for the forward about to run."""
    global _current
    _current = ctx


def get_current_forward() -> Optional[SnapshotForwardContext]:
    """Verdict + identity of the in-flight forward, or ``None`` if unpublished."""
    return _current


def clear_current_forward() -> None:
    """Drop the published context once the forward has returned."""
    global _current
    _current = None


def try_consume_capture_budget(limit: Optional[int]) -> bool:
    """Reserve one capture against the process-global budget.

    Called only after an executable has decided to capture, so ``True`` both
    authorizes and accounts for it. ``None`` means unbounded. Bounding here
    (not per rule) keeps a broad selection from dumping without end.
    """
    global _captures_done, _budget_exhausted_logged
    if limit is not None and _captures_done >= limit:
        # Log once so a user who set a rule but sees capture stop knows why,
        # instead of silently getting no further bundles.
        if not _budget_exhausted_logged:
            _budget_exhausted_logged = True
            logger.info(
                "Snapshot: capture budget (%d) exhausted; no further forwards "
                "will be captured. Raise VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_MAX_CAPTURES "
                "to capture more.",
                limit,
            )
        return False
    _captures_done += 1
    return True
