# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 27B (text) for vLLM Neuron (contrib model).

``register()`` makes the model available without changing the plugin: it registers the factory with vLLM's
``ModelRegistry`` for ``Qwen3_5ForConditionalGeneration`` (overriding vLLM's GPU class, as vLLM Neuron does for its
own models) and adds it to vLLM Neuron's model list. ``pip install -e contrib/models/qwen38`` adds a
``vllm.general_plugins`` entry point, so vLLM calls ``register()`` in every process.
"""

import logging

ARCHS = ("Qwen3_5ForConditionalGeneration",)
IMPL = "qwen38_neuron.factory:Qwen38ForConditionalGeneration"


class _DropOverwriteWarning(logging.Filter):
    def filter(self, record):
        return not any(arch in record.getMessage() for arch in ARCHS)


def _extend_neuron_model_list() -> None:
    from vllm_neuron.model import registry

    if getattr(registry.get_models, "_qwen38", False):
        return
    original = registry.get_models

    def get_models():
        from .factory import Qwen38ForConditionalGeneration

        models = [m for m in original() if m[0] not in ARCHS]
        return models + [(arch, Qwen38ForConditionalGeneration) for arch in ARCHS]

    get_models._qwen38 = True  # type: ignore[attr-defined]
    registry.get_models = get_models


def register() -> None:
    """Make Qwen3.8 available to vLLM Neuron. Safe to call more than once."""
    from vllm import ModelRegistry

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
