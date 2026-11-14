# SPDX-License-Identifier: Apache-2.0
"""Register ``HYV4ForCausalLM`` with vLLM.

Installed as a ``vllm.general_plugins`` entry point (``pyproject.toml`` in this
folder), so vLLM runs it in every process it starts -- the engine, each worker,
and ``vllm serve`` -- before it resolves the architecture. Without it a worker
cannot build the model: an architecture registered only in the user's script
exists in that process alone.
"""

ARCHITECTURE = "HYV4ForCausalLM"


def register() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCHITECTURE, "hy4_preview.factory:HYV4ForCausalLM")
