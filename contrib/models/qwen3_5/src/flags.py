# SPDX-License-Identifier: Apache-2.0
"""Environment switches for the Qwen3.5 port, in one place.

These are debugging and experiment controls, not tuning knobs: every default is
the shipped configuration, and each one exists because it answered a question
that cost a lot to answer otherwise. Keeping them together makes the debug
surface discoverable, and stops the two modules that need the same flag from
reading the environment independently.

Note the deliberate split between constants and functions. ``SEQUENCE_PARALLEL``
and ``ABLATE_MIXERS`` change the shape of the compiled graph, so they are read
once at import — a process runs one configuration. The NKI switches are read per
call, because ``probe_nki_deltanet.py`` flips them inside a single process to A/B
the two paths against each other.
"""

from __future__ import annotations

import os

# Sequence parallelism: the embedding scatters tokens across ranks and each
# mixer/MLP all-gathers on entry and reduce-scatters on exit. Setting
# VLLM_NEURON_QWEN35_DISABLE_SP=1 keeps the full sequence on every rank and
# all-reduces instead — mathematically equivalent, just more memory and
# bandwidth.
#
# Kept because SP is the one part of this model the CPU checks cannot exercise
# (they force world_size=1), so toggling it separates "the collectives are
# misplaced" from every other hypothesis in one device run. That is how the
# collectives were ruled out while hunting the split() miscompile.
SEQUENCE_PARALLEL = os.environ.get("VLLM_NEURON_QWEN35_DISABLE_SP") != "1"

# Mixer ablation, for comparing a *reduced* model between device and CPU. The
# output is meaningless text either way; only the agreement matters.
#
#   "all" (or "1")  drop every mixer, leaving embed -> (residual + MLP) x 24 ->
#                   norm -> lm_head. Isolates the "spine" that the per-layer CPU
#                   checks cannot cover at TP>1.
#   "delta"/"attn"  drop one mixer kind, to say which of the two is at fault.
#
# This is what proved the spine was correct on device and localised the bug to a
# mixer. Also used with ``probe_device_model.py --time`` to attribute prefill
# cost between the two mixer kinds.
ABLATE_MIXERS = os.environ.get("VLLM_NEURON_QWEN35_ABLATE_MIXERS", "")
if ABLATE_MIXERS == "1":
    ABLATE_MIXERS = "all"


def nki_delta_rule_enabled() -> bool:
    """Whether to use a vendored NKI kernel for the prefill delta rule.

    Opt **in**: measured on device, both vendored kernels are correct but slower
    than the batched torch path (0.43x for ``fused``, 0.08x for ``legacy``). See
    ``deltanet._nki_chunk_gated_delta_rule`` for the accounting.
    """
    return os.environ.get("VLLM_NEURON_QWEN35_ENABLE_NKI") == "1"


def nki_delta_rule_variant() -> str:
    """Which vendored kernel: ``"fused"`` (default) or ``"legacy"``.

    They differ in how they apply the in-chunk ``(I - A)^-1``; that difference is
    5.2x. Again see ``deltanet._nki_chunk_gated_delta_rule``.
    """
    variant = os.environ.get("VLLM_NEURON_QWEN35_NKI_VARIANT", "fused")
    if variant not in ("fused", "legacy"):
        raise ValueError(
            f"VLLM_NEURON_QWEN35_NKI_VARIANT must be 'fused' or 'legacy', "
            f"got {variant!r}"
        )
    return variant


__all__ = [
    "ABLATE_MIXERS",
    "SEQUENCE_PARALLEL",
    "nki_delta_rule_enabled",
    "nki_delta_rule_variant",
]
