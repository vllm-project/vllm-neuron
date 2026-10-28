# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2.5 text decoder for vllm-neuron (contrib).

Importing the package is cheap on purpose: it exports the factory class that
vLLM's ``ModelRegistry`` builds, and the factory imports ``model`` (NKI,
nkilib) only when a model is actually constructed. ``plugin.register`` runs in
every vLLM process, including the API server, so it must stay light.
"""

from .config import MiMoV2Config
from .factory import MiMoV2ForCausalLM

ARCHITECTURE = "MiMoV2ForCausalLM"

__all__ = ["ARCHITECTURE", "MiMoV2Config", "MiMoV2ForCausalLM"]
