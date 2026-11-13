# SPDX-License-Identifier: Apache-2.0
"""Muse-Glimmer-30B for vLLM Neuron (contrib model).

Contrib models are not in ``vllm_neuron/model/registry.py``. ``register()`` makes
this one available without changing the plugin:

  1. vLLM's ``ModelRegistry``: overrides vLLM's own (GPU) class for the
     ``MuseGlimmerForConditionalGeneration`` architecture, as vLLM Neuron does for
     its own models. The API server then also uses this package's multimodal
     processor.
  2. vLLM Neuron's model list (``vllm_neuron.model.registry.get_models``): the
     platform and the runner size the on-device encoder cache from the listed
     class; an unlisted model gets no encoder cache.

``pip install -e contrib/models/muse_glimmer`` adds a ``vllm.general_plugins``
entry point, so vLLM calls ``register()`` in every process (``vllm serve`` and
``LLM(...)``).
"""

import logging

ARCH = "MuseGlimmerForConditionalGeneration"
IMPL = "muse_glimmer_neuron.factory:MuseGlimmerForConditionalGeneration"


class _DropOverwriteWarning(logging.Filter):
    def filter(self, record):
        return ARCH not in record.getMessage()


def _extend_neuron_model_list() -> None:
    from vllm_neuron.model import registry

    if getattr(registry.get_models, "_muse_glimmer", False):
        return
    original = registry.get_models

    def get_models():
        from .factory import MuseGlimmerForConditionalGeneration

        models = [m for m in original() if m[0] != ARCH]
        return models + [(ARCH, MuseGlimmerForConditionalGeneration)]

    get_models._muse_glimmer = True  # type: ignore[attr-defined]
    registry.get_models = get_models


def register() -> None:
    """Make Muse-Glimmer available to vLLM Neuron. Safe to call more than once."""
    from vllm import ModelRegistry

    log = logging.getLogger("vllm.model_executor.models.registry")
    quiet = _DropOverwriteWarning()
    log.addFilter(quiet)
    try:
        ModelRegistry.register_model(
            ARCH, IMPL
        )  # intentional override of vLLM's GPU class
    finally:
        log.removeFilter(quiet)
    _extend_neuron_model_list()


__all__ = ["ARCH", "register"]
