# SPDX-License-Identifier: Apache-2.0
"""Contrib model package.

``register()`` is optional. It is the ``vllm.general_plugins`` entry point in
``pyproject.toml``: after ``pip install -e contrib/models/<your_model_name>``,
vLLM calls it in every process. See contrib/README.md, "Option 3".
"""

# TODO: the architecture string from the checkpoint's config.json, and the
# factory class as "<package>.<module>:<class>". A string keeps register()
# cheap: vLLM imports the class only when it builds the model.
ARCH = "YourModelForCausalLM"
IMPL = "your_model_neuron.factory:YourModelForCausalLM"


def register() -> None:
    """Register the model with vLLM. Safe to call more than once."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCH, IMPL)


__all__ = ["ARCH", "register"]
