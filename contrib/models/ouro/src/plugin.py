# SPDX-License-Identifier: Apache-2.0
"""Register the Neuron ``OuroForCausalLM`` with vLLM.

Installed as a ``vllm.general_plugins`` entry point (``pyproject.toml`` in the
folder above), so vLLM runs it in every process it starts: the front end, the
engine core, each Neuron worker, and ``vllm serve``. Registering only from a
user script is not enough, because the engine core and workers are separate
processes.

``register()`` does two things:

* It imports ``hf_config``, which registers the ``"ouro"`` model type with
  transformers' ``AutoConfig``. transformers does not include Ouro, and the
  checkpoint's own ``configuration_ouro.py`` loads only with
  ``--trust-remote-code``. With the registration, vLLM parses the checkpoint
  ``config.json`` without remote code. vLLM loads the general plugins in
  ``EngineArgs`` before it reads the model config, so the registration is in
  place in time.
* It registers the factory for the ``OuroForCausalLM`` architecture. vLLM
  removed its own Ouro implementation, so without this registration vLLM
  rejects the architecture as no longer supported.
"""

ARCHITECTURE = "OuroForCausalLM"


def register() -> None:
    from vllm import ModelRegistry

    from . import hf_config  # noqa: F401  (registers "ouro" with AutoConfig)

    ModelRegistry.register_model(ARCHITECTURE, "ouro_neuron.factory:OuroForCausalLM")
