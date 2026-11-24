# SPDX-License-Identifier: Apache-2.0
"""Register the Neuron ``Qwen3MoeForCausalLM`` with vLLM.

Installed as a ``vllm.general_plugins`` entry point (``pyproject.toml`` in the
folder above), so vLLM runs it in every process it starts: the front end, the
engine core, each Neuron worker, and ``vllm serve``. Registering only from a
user script is not enough, because the engine core and workers are separate
processes. vLLM ships its own GPU ``Qwen3MoeForCausalLM``; this registration
replaces it.
"""


def register() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Qwen3MoeForCausalLM", "qwen3_moe_neuron.factory:Qwen3MoeForCausalLM"
    )
