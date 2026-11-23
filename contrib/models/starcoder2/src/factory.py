# SPDX-License-Identifier: Apache-2.0
"""Factory for StarCoder2 model selection based on platform and configuration."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

# Checkpoint dtypes that the loader can cast to the BF16 compute dtype.
_SUPPORTED_CHECKPOINT_DTYPES = (torch.float32, torch.bfloat16, "float32", "bfloat16")

# HF activation names that are GELU with the tanh approximation.
_SUPPORTED_HIDDEN_ACTS = ("gelu_pytorch_tanh", "gelu_new", "gelu_fast")


class Starcoder2ForCausalLM(nn.Module):
    """Factory that validates the config and selects the StarCoder2 implementation.

    This class extends nn.Module to satisfy the ModelRegistry requirements of vLLM.
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
        """Make the model from the configs. Return the selected implementation."""
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Validate the config, then make the BF16 implementation."""
        cls._validate_config(hf_config, neuron_config)

        from .model import Starcoder2ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Validate that the implementation supports the configuration.

        Stop at construction with a clear error. A bad configuration can
        otherwise show as a shape mismatch deep in a kernel.
        """
        torch_dtype = getattr(hf_config, "torch_dtype", None) or getattr(
            hf_config, "dtype", None
        )
        if torch_dtype is not None and torch_dtype not in _SUPPORTED_CHECKPOINT_DTYPES:
            raise ValueError(
                f"StarCoder2 checkpoint dtype {torch_dtype!r} is not supported. "
                "Use a float32 or bfloat16 checkpoint."
            )

        if neuron_config is not None and neuron_config.quantization not in (
            None,
            "bf16",
        ):
            raise ValueError(
                f"StarCoder2 quantization {neuron_config.quantization!r} is not "
                'supported. Set neuron_config quantization to "bf16".'
            )

        hidden_act = getattr(hf_config, "hidden_act", "gelu_pytorch_tanh")
        if hidden_act not in _SUPPORTED_HIDDEN_ACTS:
            raise ValueError(
                f"StarCoder2 hidden_act {hidden_act!r} is not supported. "
                f"Use one of {_SUPPORTED_HIDDEN_ACTS}."
            )

        if not getattr(hf_config, "use_bias", True):
            raise ValueError(
                "StarCoder2 with use_bias=False is not supported. "
                "Use a checkpoint that has linear biases."
            )

        rope_parameters = getattr(hf_config, "rope_parameters", None) or {}
        rope_type = rope_parameters.get("rope_type", "default")
        if rope_type != "default":
            raise ValueError(
                f"StarCoder2 rope_type {rope_type!r} is not supported. "
                'Use a checkpoint with rope_type "default".'
            )

        num_heads = hf_config.num_attention_heads
        num_kv_heads = hf_config.num_key_value_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_heads}) is not a multiple of "
                f"num_key_value_heads ({num_kv_heads})."
            )

        # The implementation does not apply the sliding window. The result is
        # correct only when each sequence fits in the window.
        sliding_window = getattr(hf_config, "sliding_window", None)
        if sliding_window:
            from vllm.config import get_current_vllm_config_or_none

            vllm_config = get_current_vllm_config_or_none()
            if vllm_config is not None and vllm_config.model_config is not None:
                max_model_len = vllm_config.model_config.max_model_len
                if max_model_len > sliding_window:
                    raise ValueError(
                        f"max_model_len ({max_model_len}) is larger than the "
                        f"StarCoder2 sliding_window ({sliding_window}). The "
                        "Neuron implementation does not apply a sliding window. "
                        f"Set --max-model-len to {sliding_window} or less."
                    )
