# SPDX-License-Identifier: Apache-2.0
"""Factory for DeepSeek-V4.1-Flash (timeshare) model selection.

Generic by design: nothing here depends on how `model.py` is implemented, so
this file is complete as written and does not need revisiting when the scaffold
in `model.py` becomes a real implementation. It validates the config and defers
importing the implementation until selection time, which is also what keeps
`import vllm_neuron.model.registry` cheap.

Mirrors `qwen3/factory.py`; `nn.Module` is extended to satisfy vLLM's
`ModelRegistry`, which expects a module class.

The class name matches the checkpoint's `architectures` entry,
`DeepseekV41ForCausalLM` — that string is the registry key, and the
`_timeshare` package path is what distinguishes this port.
"""

from __future__ import annotations

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class DeepseekV41ForCausalLM(nn.Module):
    """Validates config and selects the DeepSeek-V4.1-Flash implementation.

    Also the class vLLM *inspects*. `ModelConfig` resolves this architecture
    through `vllm.ModelRegistry` and asks `is_text_generation_model(cls)`
    whether it can serve `--runner generate`; that check reads four things off
    the class — a `vllm_config` keyword on `__init__`, an `embed_input_ids`
    method, `input_ids`/`positions` keywords on `forward`, and a
    `compute_logits` method — and rejects the model with

        This model does not support `--runner generate`

    if any is missing. The three methods below exist to answer that question
    truthfully; the Neuron runtime never calls them, because it builds the model
    through `from_configs` and drives `forward` itself. Every other
    architecture this backend serves is also in vLLM's in-tree registry, so
    vLLM inspects *its* class instead and the requirement never surfaces.
    """

    def __init__(
        self,
        hf_config: PretrainedConfig | None = None,
        neuron_config: NeuronConfig | None = None,
        *,
        vllm_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # vLLM's generic construction signature, accepted so the interface
        # check sees it. Only the Neuron path (hf_config/neuron_config, or
        # `from_configs`) is actually exercised.
        if hf_config is None:
            if vllm_config is None:
                raise ValueError(
                    "DeepseekV41ForCausalLM needs either hf_config (the Neuron "
                    "path) or vllm_config."
                )
            hf_config = vllm_config.model_config.hf_config
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(
        self,
        input_ids=None,
        positions=None,
        *args,
        **kwargs,
    ):
        return self._model(input_ids, positions, *args, **kwargs)

    def embed_input_ids(self, input_ids):
        """Token embeddings, as vLLM's `VllmModel` protocol names them."""
        return self._model.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states):
        """Unembedding, as vLLM's `VllmModelForTextGeneration` names it."""
        return self._model.compute_logits(hidden_states)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)

        # Imported here, not at module scope: the registry imports this file at
        # worker start-up for every architecture, and only the selected one
        # should pay for pulling in torch kernels.
        from .model import DeepseekV41ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Reject configurations this port cannot serve, with a clear reason.

        Checked up front so a user gets a named limitation here instead of a
        cryptic failure deep in weight loading or kernel compilation.

        Note what is deliberately *not* rejected: the official config always
        carries a `vision_config`, so its presence says nothing about how the
        model is being served. This port simply does not model the vision tower
        — it is off the `tokens -> logits` path — and `config.from_configs`
        keeps that sub-config as an opaque dict. Rejecting on it would refuse
        every real checkpoint.
        """
        # The checkpoint ships fp8 attention with fp4 experts
        # (quantization_config.quant_method = "fp8"). Any other quantization
        # request would silently change what the kernels under kernels/ were
        # written and measured against.
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "fp8"):
            raise ValueError(
                f"quantization={quantization!r} is not supported for "
                "DeepSeek-V4.1-Flash (timeshare). The checkpoint stores "
                "attention in fp8 (ue8m0 scales, 32x32 blocks) and expert "
                "weights in fp4; pass quantization=None or 'fp8'."
            )

        # Multimodal serving needs the vision tower this port does not model.
        # Detected from the request, not from the config's vision_config.
        if neuron_config is not None and getattr(
            neuron_config, "mm_encoder_only", False
        ):
            raise ValueError(
                "DeepSeek-V4.1-Flash (timeshare) is text-only: the vision tower "
                "is off the tokens -> logits path and is not placed by the "
                "floorplan. Serve this checkpoint as a causal LM."
            )
