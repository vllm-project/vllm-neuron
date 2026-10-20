# SPDX-License-Identifier: Apache-2.0
"""Force vLLM's cached PIN_MEMORY constant off on Neuron.

The pooling path (``vllm.v1.pool.metadata.build_pooling_cursor``) allocates
``torch.zeros(..., pin_memory=PIN_MEMORY)``. ``PIN_MEMORY`` is a module-level
constant in ``vllm.utils.torch_utils`` computed once at import from
``current_platform.is_pin_memory_available()``. If that module is imported
before the Neuron platform is active, the constant caches ``True``, and pinning
host memory then routes to Neuron's privateuse1 backend — which is registered
without ``PrivateUse1HooksInterface`` in CPU mode — raising
``RuntimeError: Please register PrivateUse1HooksInterface ... first``.

Neuron never wants pinned host memory (``NeuronPlatform.is_pin_memory_available``
returns ``False``), so we overwrite the cached constant to match.
``build_pooling_cursor`` reads the module global at call time, so the override
takes effect immediately. We patch the already-loaded module via ``sys.modules``
(no fresh import) so this can't re-trigger platform resolution or plugin
discovery during plugin initialization.
"""

import logging
import sys

logger = logging.getLogger(__name__)

_applied = False


def apply_pin_memory_patch() -> None:
    global _applied
    if _applied:
        return
    _applied = True

    # If the module isn't loaded yet there's nothing cached to override: when it
    # loads later, current_platform (NeuronPlatform) supplies
    # is_pin_memory_available() == False, so PIN_MEMORY is computed as False.
    # Only patch when the module is already loaded holding a stale True value.
    mod = sys.modules.get("vllm.utils.torch_utils")
    if mod is not None and getattr(mod, "PIN_MEMORY", False):
        mod.PIN_MEMORY = False
        logger.debug(
            "Neuron: forced vllm.utils.torch_utils.PIN_MEMORY = False "
            "(privateuse1 has no pinned-memory hooks in CPU mode)."
        )
