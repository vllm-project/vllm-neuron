# SPDX-License-Identifier: Apache-2.0
"""Factory for OLMo model selection based on platform and configuration."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

# The checkpoint stores FP32 or bf16 weights. The model computes in bf16.
_SUPPORTED_CHECKPOINT_DTYPES = (torch.float32, torch.bfloat16, "float32", "bfloat16")


class OlmoForCausalLM(nn.Module):
    """Factory that validates the config and selects the OLMo implementation.

    This class extends nn.Module to satisfy vLLM's ModelRegistry requirements.
    The factory stores the selected implementation and sends forward() calls to it.
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
        """Validate the config. Then make the bf16 implementation."""
        cls._validate_config(hf_config, neuron_config)

        from .model import OlmoForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Make sure that this port supports the configuration.

        Stop at construction with a clear error. A configuration that this port
        does not support otherwise gives incorrect numerics or a kernel shape error.
        """
        torch_dtype = getattr(hf_config, "torch_dtype", None)
        if torch_dtype is not None and torch_dtype not in _SUPPORTED_CHECKPOINT_DTYPES:
            raise ValueError(
                f"OLMo checkpoint dtype {torch_dtype!r} is not supported. "
                "Use a float32 or bfloat16 checkpoint."
            )

        # HF clamps Q/K/V to [-clip_qkv, clip_qkv] when clip_qkv is set. The
        # fused decode kernel has no clamp step, so reject the setting.
        clip_qkv = getattr(hf_config, "clip_qkv", None)
        if clip_qkv is not None:
            raise ValueError(
                f"OLMo clip_qkv={clip_qkv} is not supported. "
                "Use a checkpoint with clip_qkv set to null."
            )

        if getattr(hf_config, "attention_bias", False):
            raise ValueError(
                "OLMo attention_bias=True is not supported. "
                "Use a checkpoint without attention bias."
            )

        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if rope_scaling and rope_scaling.get("rope_type", "default") != "default":
            raise ValueError(
                f"OLMo rope_scaling {rope_scaling!r} is not supported. "
                "Use a checkpoint with default RoPE."
            )
