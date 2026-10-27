# SPDX-License-Identifier: Apache-2.0
"""The classes vLLM's ``ModelRegistry`` builds for Qwen3.5 checkpoints.

The dense checkpoints (2B, 27B) declare ``Qwen3_5ForConditionalGeneration`` and
the sparse ones (35B-A3B, 397B-A17B) declare
``Qwen3_5MoeForConditionalGeneration``. Both build the same text-only hybrid
decoder in ``model.py``; ``Qwen3_5DecoderLayer`` picks the dense MLP or the MoE
block from the checkpoint config.

Register these classes before the engine starts, either from your script
(``ModelRegistry.register_model``) or with the ``vllm.general_plugins`` entry
point in this folder's ``pyproject.toml``. See ``plugin.py``.
"""

from __future__ import annotations

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Qwen3_5ForConditionalGeneration(nn.Module):
    """Factory for the dense Qwen3.5 checkpoints.

    Registering this class replaces vLLM's own ``Qwen3_5ForConditionalGeneration``
    in every process, including the frontend. It declares the plugin's
    ``HasRecurrentState`` (the DeltaNet layers are reported as
    ``KVSpec.state_layers``; the runner allocates and sizes their state pages)
    rather than vLLM's ``IsHybrid``: with ``IsHybrid`` vLLM's own Mamba config
    hook would resize the attention blocks as well, against the runner's
    padding. ``ServesTextOnly`` keeps the platform from configuring vision
    buckets for the multimodal checkpoint architecture.

    The class does not implement ``SupportsMultiModal``, so vLLM treats the
    checkpoint as text-only and builds no multimodal processor.

    vLLM also inspects the registered class for its text-generation protocol
    (``is_text_generation_model``): ``__init__`` must take ``vllm_config``, and
    the class needs ``embed_input_ids``, ``compute_logits`` and a ``forward``
    with ``input_ids``/``positions``. Without them vLLM does not treat the
    architecture as a generative model. The Neuron runner never calls
    ``__init__``, ``embed_input_ids`` or ``compute_logits``; it builds the model
    with ``from_configs`` and calls ``forward``.
    """

    # Native compile route: --model-type=generic, as on the XLA route. The default
    # "transformer" partitions the prefill graph per layer and remats.
    NATIVE_COMPILER_MODEL_TYPE = "generic"

    @classmethod
    def num_recurrent_state_layers(cls, hf_config: PretrainedConfig) -> int:
        """``interfaces.HasRecurrentState``: the number of DeltaNet layers."""
        from .config import LINEAR_ATTENTION

        text = getattr(hf_config, "text_config", None) or hf_config
        return list(text.layer_types).count(LINEAR_ATTENTION)

    @classmethod
    def serves_text_only(cls) -> bool:
        """``interfaces.ServesTextOnly``: only the text tower is served."""
        return True

    def __init__(
        self,
        hf_config: PretrainedConfig | None = None,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config=None,
        *,
        vllm_config=None,
        prefix: str = "",
    ) -> None:
        # ``vllm_config``/``prefix`` exist only for vLLM's protocol check (see
        # the class docstring).
        if hf_config is None:
            raise TypeError(f"build {type(self).__name__} with from_configs()")
        super().__init__()
        self._model = self.from_configs(
            hf_config, text_neuron_config, encoder_neuron_config
        )

    def forward(self, input_ids, positions, *args, **kwargs):
        return self._model(input_ids, positions, *args, **kwargs)

    def embed_input_ids(self, input_ids):
        return self._model.language_model.embed_tokens(input_ids, scatter_tokens=False)

    def compute_logits(self, hidden_states):
        return self._model.lm_head(hidden_states)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config=None,
    ) -> nn.Module:
        """Build the text-only implementation.

        The runner passes ``encoder_neuron_config`` when the platform resolved
        vision buckets for the checkpoint's ``vision_config``. It is ignored:
        this port builds no vision tower.
        """
        from .model import Qwen3_5ForCausalLM

        return Qwen3_5ForCausalLM.from_configs(
            hf_config, text_neuron_config=text_neuron_config
        )


class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Factory for the sparse Qwen3.5 checkpoints (``model_type: qwen3_5_moe``).

    A separate class only because the checkpoint declares a different
    architecture name. The decoder and the state geometry are the same.
    """


__all__ = [
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
]
