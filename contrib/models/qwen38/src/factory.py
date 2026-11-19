# SPDX-License-Identifier: Apache-2.0
"""vLLM ModelRegistry entry for Qwen3.8 (text decoder only).

The class does not implement vLLM's ``IsHybrid`` interface: that would make vLLM allocate a Mamba state cache the
Neuron runner cannot use. The GDN state is declared as KV-cache layers instead (``state.py``), which needs the
prefix cache off: two requests that shared a cached prefix would share its first block, and so its state region.
"""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Qwen38ForConditionalGeneration(nn.Module):
    """Factory: validates the config and returns the text decoder.

    vLLM inspects the registered class in the API process (its text-generation protocol needs ``__init__`` with
    ``vllm_config``, ``embed_input_ids``, ``forward`` and ``compute_logits``); the Neuron runner only calls
    ``from_configs``.
    """

    is_text_generation_model = True

    def __init__(self, hf_config: PretrainedConfig | None = None, neuron_config: NeuronConfig | None = None, *,
                 vllm_config=None, prefix: str = "") -> None:
        super().__init__()
        if hf_config is None and vllm_config is not None:
            hf_config = vllm_config.model_config.hf_config
        self._model = self.from_configs(hf_config, neuron_config) if hf_config is not None else None

    def embed_input_ids(self, input_ids):
        return self._model.embed_tokens(input_ids)

    def forward(self, input_ids, positions, *args, **kwargs):
        return self._model(input_ids, positions, *args, **kwargs)

    def compute_logits(self, hidden_states):
        raise NotImplementedError("vllm-neuron runs the LM head inside forward()")

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None = None, *,
                     text_neuron_config: NeuronConfig | None = None, encoder_neuron_config=None) -> nn.Module:
        if neuron_config is None:
            neuron_config = text_neuron_config
        cls._validate(neuron_config)
        from .model import Qwen38ForCausalLM

        return Qwen38ForCausalLM.from_configs(hf_config, neuron_config)

    @staticmethod
    def _validate(neuron_config) -> None:
        quantization = getattr(neuron_config, "quantization", None) if neuron_config else None
        if quantization not in (None, "bf16"):
            raise ValueError(f"the Qwen3.8 contrib model supports BF16 only, got quantization={quantization!r}")
        try:
            from vllm.config import get_current_vllm_config

            cache = get_current_vllm_config().cache_config
        except Exception:  # noqa: BLE001 (no vLLM config outside an engine, e.g. unit tests)
            return
        level = getattr(getattr(get_current_vllm_config(), "optimization_level", None), "value", None)
        if level is not None and int(level) != 1:
            raise ValueError(
                f"the Qwen3.8 contrib model needs --optimization-level 1 (got {level}): compiled at -O2, the prefill "
                "graphs gave logits that disagree with Hugging Face on prompts past 512 tokens (README.md)")
        if getattr(cache, "enable_prefix_caching", False):
            raise ValueError("the Qwen3.8 contrib model keeps the GDN state in the request's first KV block; "
                             "serve with --no-enable-prefix-caching")
