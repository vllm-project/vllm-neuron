# SPDX-License-Identifier: Apache-2.0
"""Register ``GlmMoeDsaForCausalLM`` with vLLM.

Installed as a ``vllm.general_plugins`` entry point (``pyproject.toml`` in this
folder), so vLLM runs it in every process it starts -- the engine, each worker,
and ``vllm serve`` -- before it resolves the architecture. vLLM has its own (GPU)
class for this architecture; registering replaces it with this one.
"""

ARCHITECTURE = "GlmMoeDsaForCausalLM"


def register() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCHITECTURE, "glm_dsa.factory:GLMDsaForCausalLM")
