# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini (text) for vLLM Neuron (contrib model).

``register()`` makes the model available without changing the plugin: it registers the factory with vLLM's
``ModelRegistry`` for ``Inkling2LoopForConditionalGeneration`` (overriding vLLM's GPU class, as vLLM Neuron does for its
own models) and adds it to vLLM Neuron's model list. ``pip install -e contrib/models/inkling2`` adds a
``vllm.general_plugins`` entry point, so vLLM calls ``register()`` in every process.
"""

import logging

ARCHS = ("Inkling2LoopForConditionalGeneration",)
IMPL = "inkling2_neuron.factory:Inkling2ForConditionalGeneration"


class _DropOverwriteWarning(logging.Filter):
    def filter(self, record):
        return not any(arch in record.getMessage() for arch in ARCHS)


def _extend_neuron_model_list() -> None:
    from vllm_neuron.model import registry

    if getattr(registry.get_models, "_inkling2", False):
        return
    original = registry.get_models

    def get_models():
        from .factory import Inkling2ForConditionalGeneration

        models = [m for m in original() if m[0] not in ARCHS]
        return models + [(arch, Inkling2ForConditionalGeneration) for arch in ARCHS]

    get_models._inkling2 = True  # type: ignore[attr-defined]
    registry.get_models = get_models


def register() -> None:
    """Make Inkling 2 available to vLLM Neuron. Safe to call more than once."""
    from vllm import ModelRegistry

    from .config import register_hf_configs

    # the image's transformers predates inkling2_loop: register the config classes vLLM needs to read config.json
    register_hf_configs()

    log = logging.getLogger("vllm.model_executor.models.registry")
    quiet = _DropOverwriteWarning()
    log.addFilter(quiet)
    try:
        for arch in ARCHS:
            ModelRegistry.register_model(arch, IMPL)
    finally:
        log.removeFilter(quiet)
    _extend_neuron_model_list()


__all__ = ["ARCHS", "register"]
