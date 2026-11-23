# SPDX-License-Identifier: Apache-2.0
"""Factory for the Jina Embeddings v5 text model (EuroBERT encoder)."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class JinaEmbeddingsV5Model(nn.Module):
    """Factory that validates the config and selects the implementation.

    The class extends ``nn.Module`` and has the constructor, ``forward`` and
    ``embed_input_ids`` signatures of the vLLM model protocol, so that the vLLM
    ``ModelRegistry`` accepts it. The pooling attributes let vLLM classify the
    architecture as an embedding model before the model is built. A pooling
    model has no ``compute_logits``.
    """

    is_pooling_model = True
    default_seq_pooling_type = "LAST"

    def __init__(
        self,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
        *,
        vllm_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *args,
        **kwargs,
    ):
        """Delegate the forward pass to the selected implementation."""
        return self._model(*args, input_ids=input_ids, positions=positions, **kwargs)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

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

        from vllm.config import get_current_vllm_config

        model_config = get_current_vllm_config().model_config
        runner_type = getattr(model_config, "runner_type", None)
        if runner_type != "pooling":
            raise ValueError(
                f"Runner type {runner_type!r} is not supported for Jina "
                "Embeddings v5. The model is an encoder without an LM head. "
                "Start it with --runner pooling."
            )

        from .model import JinaEmbeddingsV5ForEmbedding as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Reject the configurations that this port does not implement."""
        get = (
            hf_config.get
            if isinstance(hf_config, dict)
            else lambda key, default=None: getattr(hf_config, key, default)
        )
        if get("is_decoder", False):
            raise NotImplementedError(
                "Jina Embeddings v5 with is_decoder=True is not supported. This "
                "port implements the EuroBERT bidirectional encoder only."
            )
        if get("attention_bias", False) or get("mlp_bias", False):
            raise NotImplementedError(
                "Attention bias and MLP bias are not supported for Jina "
                "Embeddings v5. Use a checkpoint without projection bias."
            )
        hidden_act = get("hidden_act", "silu")
        if hidden_act != "silu":
            raise NotImplementedError(
                f"hidden_act={hidden_act!r} is not supported for Jina Embeddings "
                "v5. The Neuron MLP kernel uses SiLU."
            )
        if get("quantization_config", None) is not None:
            raise NotImplementedError(
                "Quantized Jina Embeddings v5 checkpoints are not supported. "
                "Use the BF16 checkpoint."
            )
        if neuron_config is not None:
            for name in ("attention_dp_size", "mlp_dp_size", "embedding_dp_size"):
                size = getattr(neuron_config, name, 1) or 1
                if size != 1:
                    raise NotImplementedError(
                        f"{name}={size} is not supported for Jina Embeddings v5. "
                        f"Set {name} to 1."
                    )
        try:
            from vllm.distributed.parallel_state import get_dcp_group

            dcp_size = get_dcp_group().world_size
        except (AssertionError, AttributeError):
            dcp_size = 1
        if dcp_size > 1:
            raise NotImplementedError(
                f"Decode context parallel size {dcp_size} is not supported for "
                "Jina Embeddings v5. Set the decode context parallel size to 1."
            )
