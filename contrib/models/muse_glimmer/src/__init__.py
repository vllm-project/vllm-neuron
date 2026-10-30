# SPDX-License-Identifier: Apache-2.0
"""MuseGlimmer (``MuseGlimmerForConditionalGeneration``) for vLLM on Neuron (contrib)."""

from .config import MuseGlimmerConfig, MuseGlimmerTextConfig, MuseGlimmerVisionConfig
from .factory import ARCHITECTURE, MuseGlimmerForConditionalGeneration

__all__ = [
    "ARCHITECTURE",
    "MuseGlimmerConfig",
    "MuseGlimmerForConditionalGeneration",
    "MuseGlimmerTextConfig",
    "MuseGlimmerVisionConfig",
    "register",
]


def register() -> None:
    """Register MuseGlimmer with vLLM. Safe to call more than once.

    vLLM ships its own GPU implementation under the same architecture name
    (``MuseGlimmerForConditionalGeneration`` -> ``MuseGlimmerForCausalLM``), and
    the engine core and the Neuron workers run in spawned processes. Each of
    those processes resolves the architecture on its own: the front-end and the
    engine core build the multimodal processor and the encoder budget from it,
    and the worker builds the model. The Neuron class must win in all of them.

    Installing this folder (``pip install -e contrib/models/muse_glimmer``)
    exposes this function as a ``vllm.general_plugins`` entry point, which vLLM
    calls in every process at startup.
    """
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        ARCHITECTURE, f"{__name__}.factory:MuseGlimmerForConditionalGeneration"
    )
