# SPDX-License-Identifier: Apache-2.0
"""Register the Qwen3.5 architectures with vLLM.

Installed as a ``vllm.general_plugins`` entry point (``pyproject.toml`` in this
folder), so vLLM runs it in every process it starts (the frontend, the engine
core and each worker, also under ``vllm serve``) before it resolves the
architecture. An architecture that is registered only in a user's script
exists only in that process and in the processes it forks.
"""

ARCHITECTURES = (
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
)


def register() -> None:
    from vllm import ModelRegistry

    for arch in ARCHITECTURES:
        ModelRegistry.register_model(arch, f"qwen3_5.factory:{arch}")
