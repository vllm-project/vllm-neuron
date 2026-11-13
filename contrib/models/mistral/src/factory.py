# SPDX-License-Identifier: Apache-2.0
"""Factory for the Mistral contrib model.

Adapted from ``vllm_neuron/model/llama3/factory.py``. Only the main
``MistralForCausalLM`` factory is kept: the Llama Eagle3 / P-EAGLE factories
import draft-model files that this port does not carry.

This class plays two roles, like the factory in
``contrib/models/muse_glimmer/src/factory.py``:

* In the Neuron worker, the model runner calls ``from_configs`` to build the
  Neuron implementation in ``model.py``. ``__init__`` is never used there.
* In vLLM's front end and engine core, vLLM only *inspects* the registered
  class to decide whether it is a text-generation model. It must look like a
  vLLM model: ``__init__(*, vllm_config, prefix)``, ``forward(input_ids,
  positions)``, ``embed_input_ids`` and ``compute_logits``. The official
  Neuron models never hit this because the plugin registers them only inside
  the Neuron worker, so the front end inspects vLLM's own GPU class instead.
  This contrib model is registered in every process through the
  ``vllm.general_plugins`` entry point, so the front end sees this class.
  Without the stubs below vLLM fails with "This model does not support
  `--runner generate`".
"""

from typing import ClassVar

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class MistralForCausalLM(nn.Module):
    """Validates the config and builds the Mistral Neuron implementation.

    vLLM's front end inspects this class (never instantiates it), and the
    Neuron model runner calls :meth:`from_configs`.
    """

    # vLLM front-end interface tag (inspected, never instantiated there).
    is_text_generation_model: ClassVar[bool] = True

    def __init__(self, *, vllm_config=None, prefix: str = "", **kwargs) -> None:
        raise RuntimeError(
            "MistralForCausalLM (contrib) is a factory; the Neuron model runner "
            "builds the model with MistralForCausalLM.from_configs()."
        )

    # -- VllmModel / VllmModelForTextGeneration protocol stubs ------------
    # Present only so vLLM's front-end inspection recognises a generative
    # model. They are never called: the runner uses the object returned by
    # from_configs(), which is the real model in model.py.
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, **kwargs):
        raise RuntimeError("factory stub: use the model built by from_configs()")

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("factory stub: use the model built by from_configs()")

    def compute_logits(self, hidden_states: torch.Tensor):
        raise RuntimeError("factory stub: use the model built by from_configs()")

    # -- plugin entry point -------------------------------------------------
    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Create model from configs. Returns the selected implementation directly."""
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Select and instantiate the implementation based on config."""
        cls._validate_config(hf_config, neuron_config)

        from .model import MistralForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Reject configurations this port does not implement.

        * ``sliding_window``: Mistral-7B-v0.1 uses a 4096-token sliding
          window. The attention code here is full causal attention, so a
          non-null value would give wrong outputs past 4096 tokens.
        * ``quantization_config``: only BF16 checkpoints are supported.
        """
        del neuron_config  # reserved for future validation rules
        if hf_config is None:
            return

        sliding_window = getattr(hf_config, "sliding_window", None)
        if sliding_window is not None:
            raise NotImplementedError(
                f"sliding_window={sliding_window} is not supported by the Mistral "
                "contrib port. Use a checkpoint without a sliding window, such as "
                "Mistral-7B-v0.3."
            )

        quant_cfg = getattr(hf_config, "quantization_config", None)
        if quant_cfg is None and hasattr(hf_config, "to_dict"):
            quant_cfg = hf_config.to_dict().get("quantization_config")
        if quant_cfg is not None:
            raise NotImplementedError(
                "Quantized Mistral checkpoints are not supported by the Mistral "
                "contrib port."
            )
