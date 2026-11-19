# SPDX-License-Identifier: Apache-2.0
"""Register the Neuron ``StableLmForCausalLM`` with vLLM.

Installed as a ``vllm.general_plugins`` entry point (``pyproject.toml`` in the
folder above), so vLLM runs it in every process it starts: the front end, the
engine core, each Neuron worker, and ``vllm serve``. Registering only from a
user script is not enough, because the engine core and workers are separate
processes. vLLM also ships its own GPU ``StableLmForCausalLM``; this
registration replaces it with the Neuron implementation.
"""

ARCHITECTURE = "StableLmForCausalLM"
IMPLEMENTATION = "stablelm_neuron.factory:StableLmForCausalLM"


def register() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCHITECTURE, IMPLEMENTATION)
