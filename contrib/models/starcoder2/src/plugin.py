# SPDX-License-Identifier: Apache-2.0
"""Register the Neuron ``Starcoder2ForCausalLM`` with vLLM.

Installed as a ``vllm.general_plugins`` entry point (``pyproject.toml`` in the
folder above), so vLLM runs it in every process it starts: the front end, the
engine core, each Neuron worker, and ``vllm serve``. Registering only from a
user script is not enough, because the engine core and workers are separate
processes.

vLLM ships its own ``Starcoder2ForCausalLM``. This call replaces that entry in
the registry, so the Neuron class is the one vLLM builds.
"""


def register() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Starcoder2ForCausalLM", "starcoder2_neuron.factory:Starcoder2ForCausalLM"
    )
