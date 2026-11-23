# SPDX-License-Identifier: Apache-2.0
"""Factory for HyperCLOVAX model selection based on platform and configuration."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

# The checkpoint dtypes that the loader can cast to the bfloat16 compute dtype.
_SUPPORTED_CHECKPOINT_DTYPES = (torch.float32, torch.bfloat16, "float32", "bfloat16")


class HyperCLOVAXForCausalLM(nn.Module):
    """Factory that validates config and selects the HyperCLOVAX implementation.

    This class extends nn.Module to satisfy vLLM's ModelRegistry requirements.
    The factory stores the selected implementation and delegates forward() calls to it.
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
        """Delegate forward pass to the selected implementation."""
        return self._model(*args, input_ids=input_ids, positions=positions, **kwargs)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

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
        """Select and instantiate the appropriate implementation based on config."""
        cls._validate_config(hf_config, neuron_config)

        from .model import HyperCLOVAXForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Validate that the configuration is supported.

        Stop at construction with a clear error. An unsupported configuration
        otherwise causes a shape mismatch deep in the kernels.
        """
        checkpoint_dtype = getattr(hf_config, "dtype", None) or getattr(
            hf_config, "torch_dtype", None
        )
        if (
            checkpoint_dtype is not None
            and checkpoint_dtype not in _SUPPORTED_CHECKPOINT_DTYPES
        ):
            raise ValueError(
                f"HyperCLOVAX on Neuron supports float32 and bfloat16 checkpoints only. "
                f"The checkpoint dtype is {checkpoint_dtype!r}. Use a float32 or a "
                f"bfloat16 checkpoint."
            )

        if getattr(hf_config, "attention_bias", False) or getattr(
            hf_config, "mlp_bias", False
        ):
            raise ValueError(
                "HyperCLOVAX on Neuron does not support attention_bias or mlp_bias. "
                "Use a checkpoint without projection biases."
            )

        rope_parameters = getattr(hf_config, "rope_parameters", None) or {}
        rope_scaling = getattr(hf_config, "rope_scaling", None) or {}
        for rope_cfg in (rope_parameters, rope_scaling):
            rope_type = rope_cfg.get("rope_type", rope_cfg.get("type", "default"))
            if rope_type != "default":
                raise ValueError(
                    f"HyperCLOVAX on Neuron supports the default RoPE only. "
                    f"The config sets rope_type={rope_type!r}."
                )

        hidden_act = getattr(hf_config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                f"HyperCLOVAX on Neuron supports hidden_act='silu' only. "
                f"The config sets {hidden_act!r}."
            )
