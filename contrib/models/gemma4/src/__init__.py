# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 31B (text) for vLLM Neuron (contrib model).

Contrib models are not in ``vllm_neuron/model/registry.py``. ``register()`` makes
this one available without changing the plugin:

  1. vLLM's ``ModelRegistry``: overrides vLLM's own (GPU) classes for the
     ``Gemma4ForConditionalGeneration`` and ``Gemma4ForCausalLM`` architectures,
     as vLLM Neuron does for its own models.
  2. vLLM Neuron's model list (``vllm_neuron.model.registry.get_models``), so the
     platform and runner treat it like a built-in model.

``pip install -e contrib/models/gemma4`` adds a ``vllm.general_plugins`` entry
point, so vLLM calls ``register()`` in every process (``vllm serve`` and
``LLM(...)``).
"""

import logging

ARCHS = ("Gemma4ForConditionalGeneration", "Gemma4ForCausalLM")
IMPL = "gemma4_neuron.factory:Gemma4ForConditionalGeneration"


class _DropOverwriteWarning(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not any(arch in msg for arch in ARCHS)


def _extend_neuron_model_list() -> None:
    from vllm_neuron.model import registry

    if getattr(registry.get_models, "_gemma4", False):
        return
    original = registry.get_models

    def get_models():
        from .factory import Gemma4ForConditionalGeneration

        models = [m for m in original() if m[0] not in ARCHS]
        return models + [(arch, Gemma4ForConditionalGeneration) for arch in ARCHS]

    get_models._gemma4 = True  # type: ignore[attr-defined]
    registry.get_models = get_models


def register() -> None:
    """Make Gemma 4 available to vLLM Neuron. Safe to call more than once."""
    from vllm import ModelRegistry

    log = logging.getLogger("vllm.model_executor.models.registry")
    quiet = _DropOverwriteWarning()
    log.addFilter(quiet)
    try:
        for arch in ARCHS:
            # Intentional override of vLLM's GPU class.
            ModelRegistry.register_model(arch, IMPL)
    finally:
        log.removeFilter(quiet)
    _extend_neuron_model_list()


__all__ = ["ARCHS", "register"]
