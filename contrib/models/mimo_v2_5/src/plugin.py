# SPDX-License-Identifier: Apache-2.0
"""Register ``MiMoV2ForCausalLM`` with vLLM.

Installed as a ``vllm.general_plugins`` entry point (``pyproject.toml`` in this
folder), so vLLM runs it in every process it starts -- the API server, the
engine core and each worker -- before it resolves the architecture. That is
what makes ``vllm serve`` work without editing ``vllm_neuron/model/registry.py``.
An offline script can instead call ``ModelRegistry.register_model`` itself
before constructing ``LLM`` (Option 1 in ``contrib/README.md``); that covers the
worker processes only while vLLM forks them, which is the Neuron default.

The class is registered as a lazy ``"module:Class"`` string so this function
imports nothing heavy. It overrides vLLM's in-tree GPU ``MiMoV2ForCausalLM``
entry, which is the point: on Neuron the plugin's model must be the one built.
"""

ARCHITECTURE = "MiMoV2ForCausalLM"
FACTORY = "mimo_v2_5.factory:MiMoV2ForCausalLM"


def register() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCHITECTURE, FACTORY)
    from .spmd_compile import install_leader_compile

    install_leader_compile()
