# SPDX-License-Identifier: Apache-2.0
"""Factory for ERNIE 4.5 dense model selection based on configuration."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


def _get(hf_config, key: str, default=None):
    if isinstance(hf_config, dict):
        return hf_config.get(key, default)
    return getattr(hf_config, key, default)


class Ernie4_5ForCausalLM(nn.Module):
    """Factory that validates the config and selects the ERNIE 4.5 implementation.

    This class extends nn.Module to satisfy the vLLM ModelRegistry requirements.
    The factory keeps the selected implementation and sends forward() calls to it.
    """

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
        """Send the forward pass to the selected implementation."""
        return self._model(*args, input_ids=input_ids, positions=positions, **kwargs)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Create the model from configs. Return the selected implementation."""
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Validate the config, then make the implementation."""
        cls._validate_config(hf_config, neuron_config)

        from .model import Ernie4_5ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Make sure that the implementation supports the configuration.

        Stop at construction with a clear error. A wrong configuration can
        otherwise cause a shape mismatch deep in the kernels.
        """
        # A config can have "torch_dtype" and "dtype" together. Check each key,
        # because the two values can be different.
        for dtype_key in ("torch_dtype", "dtype"):
            torch_dtype = _get(hf_config, dtype_key)
            if torch_dtype is not None and torch_dtype not in (
                torch.bfloat16,
                "bfloat16",
            ):
                raise ValueError(
                    f"ERNIE 4.5 on Neuron supports only bfloat16. The config has "
                    f"{dtype_key}={torch_dtype!r}. Set --dtype bfloat16."
                )

        if _get(hf_config, "use_bias", False):
            raise ValueError(
                "ERNIE 4.5 with use_bias=True is not supported. Use a checkpoint "
                "without projection bias."
            )

        rope_scaling = _get(hf_config, "rope_scaling")
        rope_parameters = _get(hf_config, "rope_parameters")
        rope_type = None
        if isinstance(rope_scaling, dict):
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        elif isinstance(rope_parameters, dict):
            rope_type = rope_parameters.get("rope_type", "default")
        if rope_type not in (None, "default"):
            raise ValueError(
                f"ERNIE 4.5 RoPE type {rope_type!r} is not supported. Only the "
                "default RoPE is supported."
            )

        hidden_size = _get(hf_config, "hidden_size")
        num_heads = _get(hf_config, "num_attention_heads")
        head_dim = _get(hf_config, "head_dim") or (
            hidden_size // num_heads if hidden_size and num_heads else None
        )
        if head_dim is not None and head_dim % 2 != 0:
            raise ValueError(
                f"ERNIE 4.5 head_dim ({head_dim}) is odd. RoPE needs an even head_dim."
            )

        hidden_act = _get(hf_config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                f"ERNIE 4.5 activation {hidden_act!r} is not supported. Only silu "
                "is supported."
            )
