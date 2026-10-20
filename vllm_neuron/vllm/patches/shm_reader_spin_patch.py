# SPDX-License-Identifier: Apache-2.0
"""Bound the vLLM ``shm_broadcast`` reader spin, so an idle reader releases its core.

``SpinCondition.wait()`` yields the CPU until ``busy_loop_s`` has passed since the last read,
then blocks on a notification socket. Upstream defaults that to one second and neither
construction site overrides it, so on any shorter step the deadline is always in the future and
the blocking branch is unreachable: the hybrid design degrades into an unbounded spin and every
reader holds a core it is not using. That costs more than the CPU, because a step waits for the
slowest rank, so oversubscribing the host lets descheduling pick which rank that is.

This replaces the default value rather than wrapping the function, so upstream's ``__init__``
runs unchanged. The writer is unaffected: its branch assigns ``busy_loop_s = 0`` directly.

Blocking does need a bounded wait, or one dropped notification hangs the reader for good. The
notifier is best-effort, but ``acquire_read`` tests the authoritative shared-memory flag before
waiting and the writer sets it before notifying, so a notification is only a hint. Shortening the
spin window is what makes that bound matter: at upstream's one second a decode reader never
reached the blocking branch at all, so this module owns both halves of the change.

The bound is upstream's own ``SHM_READER_RECHECK_INTERVAL_MS``, which exists from vLLM 0.27.1
onwards and ships at 5000 ms. This module lowers that value rather than wrapping ``wait()``,
because its only reader loads it as a module global at call time, so upstream keeps applying its
own ``min`` against the caller's deadline. A vLLM without the constant returns ``None`` from
``timeout_ms`` and cannot be bounded at all, so there this module declines entirely rather than
shorten the spin window and leave a dropped notification able to hang a reader for good.
"""

import logging

from vllm_neuron import envs as neuron_envs

logger = logging.getLogger(__name__)

# The reader spins for this long after its last read, then blocks. Upstream uses one second,
# which a short step never reaches. This keeps a brief opportunistic window for the case where
# the next chunk is already in the ring, and blocks for everything else.
READER_BUSY_LOOP_S = 0.0002

# Upper bound for one idle poll. A reader re-tests the authoritative shared-memory flag when the
# poll returns, so a lost notification costs one extra test of the flag and at most this much
# delay. Upstream ships 5000, which is about a thousand steps at the step period this serves.
READER_POLL_BOUND_MS = 5

# Upstream's own recheck interval, and this module's capability test. Present from 0.27.1; the
# pinned vLLM has it. Its absence means the idle poll cannot be bounded, so the patch declines.
_UPSTREAM_RECHECK_ATTR = "SHM_READER_RECHECK_INTERVAL_MS"

# The upstream default this module expects to replace. A mismatch means the signature moved.
_EXPECTED_UPSTREAM_BUSY_LOOP_S = 1

_applied = False


def apply_shm_reader_spin_patch() -> None:
    """Bound the reader spin of every ``SpinCondition`` built after this call.

    Call it before the process builds a ``MessageQueue`` reader. ``NeuronWorker.init_device``
    covers each worker and the Neuron ``run_engine_core`` wrapper covers each engine; both run
    before their queues exist, and this is idempotent, so both call sites are safe.

    It never raises. Doing nothing is identical to upstream behaviour, and a performance change
    must not stop a server from starting. An unrecognised signature is reported at ERROR level
    instead, because doing nothing costs a held core per rank and that must not be silent.

    Example:
        >>> apply_shm_reader_spin_patch()  # doctest: +SKIP
    """
    global _applied
    if _applied:
        return

    if not neuron_envs.VLLM_NEURON_SHM_READER_SPIN_PATCH:
        # Say it at INFO. The cost of being off is a held core per reader, which is invisible
        # from the outside, so the one line that explains it has to be in the log.
        logger.info(
            "Neuron: shm_broadcast reader spin patch is disabled by "
            "VLLM_NEURON_SHM_READER_SPIN_PATCH. Each reader keeps upstream's %s s spin "
            "window and holds a CPU while it waits.",
            _EXPECTED_UPSTREAM_BUSY_LOOP_S,
        )
        return

    try:
        from vllm.distributed.device_communicators import shm_broadcast
    except Exception as exc:
        # Not only ImportError. At plugin-import time vLLM can be partially initialised, and
        # a circular import then raises AttributeError. A later call site retries.
        logger.debug(
            "shm reader spin patch deferred. shm_broadcast is not ready: %s", exc
        )
        return

    spin_condition_cls = getattr(shm_broadcast, "SpinCondition", None)
    if spin_condition_cls is None:
        logger.debug("shm reader spin patch skipped. This vLLM has no SpinCondition.")
        return

    defaults = spin_condition_cls.__init__.__defaults__
    if defaults != (_EXPECTED_UPSTREAM_BUSY_LOOP_S,):
        # Refuse rather than guess: replacing an unknown default could set the wrong
        # parameter. Say so loudly, because staying silent loses the fix with no symptom.
        logger.error(
            "Neuron: the shm_broadcast reader spin is NOT bounded. This module expects "
            "SpinCondition.__init__ to default busy_loop_s to %s, but the defaults are %r. "
            "Each reader will hold a CPU while it waits, which costs one core for every "
            "rank. Re-measure, then update READER_BUSY_LOOP_S and "
            "_EXPECTED_UPSTREAM_BUSY_LOOP_S together.",
            _EXPECTED_UPSTREAM_BUSY_LOOP_S,
            defaults,
        )
        return

    previous_recheck = getattr(shm_broadcast, _UPSTREAM_RECHECK_ATTR, None)
    if previous_recheck is None:
        # Check this BEFORE touching busy_loop_s. Shortening the spin window is what moves a
        # reader onto the blocking branch, and on a vLLM with no recheck interval that branch
        # polls with no timeout, so one dropped notification would hang the reader for good.
        # Declining leaves upstream's timing exactly as it was, which is slow but safe.
        logger.error(
            "Neuron: the shm_broadcast reader spin patch is DECLINED. This vLLM has no "
            "%s, so an idle poll cannot be bounded and shortening the spin window would let "
            "one dropped notification hang a reader. Each reader keeps upstream's %s s spin "
            "window and holds a CPU while it waits. %s exists from vLLM 0.27.1 onwards.",
            _UPSTREAM_RECHECK_ATTR,
            _EXPECTED_UPSTREAM_BUSY_LOOP_S,
            _UPSTREAM_RECHECK_ATTR,
        )
        return

    spin_condition_cls.__init__.__defaults__ = (READER_BUSY_LOOP_S,)
    setattr(shm_broadcast, _UPSTREAM_RECHECK_ATTR, READER_POLL_BOUND_MS)

    _applied = True
    logger.info(
        "Neuron: shm_broadcast reader busy_loop_s %s -> %s (an idle reader now blocks "
        "instead of holding a core), and %s %s -> %s ms.",
        _EXPECTED_UPSTREAM_BUSY_LOOP_S,
        READER_BUSY_LOOP_S,
        _UPSTREAM_RECHECK_ATTR,
        previous_recheck,
        READER_POLL_BOUND_MS,
    )
