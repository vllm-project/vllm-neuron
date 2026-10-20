# SPDX-License-Identifier: Apache-2.0
"""Write the per-call identity tag (``meta.json``) beside a captured bundle.

The dumped tensors are an anonymous positional vector; the path carries the
compilation hash and rank, and the filename carries position, but none of that
says *which request or token* the forward ran for. That identity is what an
accuracy regression is reported against, so it is recorded here next to the
tensors. ``global_step`` ties the same logical step across NEFFs and (within a
tensor-parallel group) across ranks.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import List, Optional

from vllm_neuron.snapshot.context import SnapshotForwardContext

logger = logging.getLogger(__name__)

_META_FILENAME = "meta.json"
_SCHEMA = 1


def write_call_meta(
    call_dir: str,
    *,
    compilation_hash: str,
    fmt: str,
    global_rank: int,
    tp_rank: int,
    dp_rank: int,
    call_index: int,
    selected_by: List[str],
    inputs: Optional[List[dict]] = None,
    context: Optional[SnapshotForwardContext],
) -> str:
    """Write ``{call_dir}/meta.json`` describing what this capture ran for.

    ``fmt`` (``"npy"``/``"pt"``) records the on-disk artifact extension so a
    reader can locate ``tensor{i}.{fmt}``; ``inputs`` lists each input's
    ``index``, ``dtype`` and ``shape`` so those bytes can be reinterpreted
    (bf16/fp8 ``.npy`` carry no numpy type label). ``context`` is ``None`` for a
    call-index capture with no published forward; known fields are still
    written.

    Raises on any write failure: a requested capture that cannot be finalized is
    a hard error, not a silently incomplete bundle (see
    ``SnapshotCapturer.pre_execute``).
    """
    path = os.path.join(call_dir, _META_FILENAME)
    payload = {
        "schema": _SCHEMA,
        "hash": compilation_hash,
        "format": fmt,
        "global_rank": global_rank,
        "tp_rank": tp_rank,
        "dp_rank": dp_rank,
        "call_index": call_index,
        "selected_by": selected_by,
        "inputs": inputs or [],
        "global_step": context.global_step if context else None,
        "is_prompt": context.is_prompt if context else None,
        "matches": [asdict(m) for m in context.matches] if context else [],
        "req_ids": list(context.req_ids) if context else [],
        "positions": list(context.positions) if context else [],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    logger.info("Snapshot: finalized %s", call_dir)
    return path
