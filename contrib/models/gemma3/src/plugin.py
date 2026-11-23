# SPDX-License-Identifier: Apache-2.0
"""Register the Neuron ``Gemma3ForCausalLM`` with vLLM.

Installed as a ``vllm.general_plugins`` entry point (``pyproject.toml`` in the
folder above), so vLLM runs it in every process it starts: the front end, the
engine core, each Neuron worker, and ``vllm serve``. Registering only from a
user script is not enough, because the engine core and workers are separate
processes.
"""


def register() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Gemma3ForCausalLM", "gemma3_neuron.factory:Gemma3ForCausalLM"
    )
