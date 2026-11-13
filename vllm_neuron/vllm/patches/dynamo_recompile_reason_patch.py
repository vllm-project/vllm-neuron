# SPDX-License-Identifier: Apache-2.0
"""Make dynamo's recompilation-reason report best-effort.

On every recompilation ``torch._dynamo.convert_frame._compile`` calls
``get_and_maybe_log_recompilation_reasons`` to say *why* the cached entries'
guards failed. When the failing guard is the no-tensor-aliasing one, torch 2.13
re-evaluates each aliasing source of the **old** entry against the **new**
frame's locals (``guards.recompilation_reason_for_no_tensor_aliasing_guard``),
with no exception handling:

    tensor_id = id(eval(tensor_source, global_scope, scope))
    KeyError: 'wgate'

The sources are attribute paths through ``nn.Module`` ``_modules`` dicts, so a
frame re-entered with a module that *lacks* a submodule the earlier entry had
— a decoder layer without a KV compressor, called through the same per-block
helper as one with it — raises ``KeyError`` out of the report, and the forward
dies on a diagnostic. The recompilation itself is legitimate; only the
explanation of it is broken.

This wraps that reporter so a source that cannot be evaluated against the new
frame is skipped rather than fatal. Every other path in
``get_guard_fail_reason_helper`` already evaluates its parts under
``try/except Exception``; this gives the aliasing path the same treatment.
Applied at import time so it survives spawn-mode re-imports in worker
processes.
"""

import functools
import logging

logger = logging.getLogger(__name__)

_applied = False


def apply_dynamo_recompile_reason_patch() -> None:
    global _applied
    if _applied:
        return
    _applied = True

    try:
        from torch._dynamo import guards as dynamo_guards
    except Exception:  # noqa: BLE001 - no dynamo, nothing to patch
        return

    orig = getattr(
        dynamo_guards, "recompilation_reason_for_no_tensor_aliasing_guard", None
    )
    if orig is None or getattr(orig, "_vllm_neuron_best_effort", False):
        return

    @functools.wraps(orig)
    def _best_effort(guard_manager, scope):  # noqa: ANN001 - torch's signature
        try:
            return orig(guard_manager, scope)
        except Exception as e:  # noqa: BLE001 - a diagnostic must not kill the forward
            logger.debug(
                "Skipped dynamo's no-tensor-aliasing recompilation reason: "
                "a guard source does not evaluate against the new frame (%s: %s)",
                type(e).__name__,
                e,
            )
            return []

    _best_effort._vllm_neuron_best_effort = True  # type: ignore[attr-defined]
    dynamo_guards.recompilation_reason_for_no_tensor_aliasing_guard = _best_effort
    logger.debug("Applied best-effort dynamo recompilation-reason patch")
