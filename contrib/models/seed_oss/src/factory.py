# SPDX-License-Identifier: Apache-2.0
"""Factory for Seed-OSS model selection based on platform and configuration."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


def _get(hf_config, name: str, default=None):
    if isinstance(hf_config, dict):
        return hf_config.get(name, default)
    return getattr(hf_config, name, default)


class SeedOssForCausalLM(nn.Module):
    """Factory that validates the config and selects the Seed-OSS implementation.

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
        """Validate the config, then make the BF16 implementation."""
        cls._validate_config(hf_config, neuron_config)

        from .model import SeedOssForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Make sure that the port supports the configuration.

        Stop at construction with a clear error. A configuration that the port
        does not support otherwise causes a shape error in a kernel.
        """
        torch_dtype = _get(hf_config, "torch_dtype")
        if torch_dtype is not None and torch_dtype not in (
            torch.bfloat16,
            "bfloat16",
        ):
            raise ValueError(
                f"Seed-OSS on Neuron supports only bfloat16. The config has "
                f"{torch_dtype!r}. Use a bfloat16 checkpoint."
            )

        if _get(hf_config, "attention_out_bias", False):
            raise ValueError(
                "Seed-OSS on Neuron does not support attention_out_bias=True. "
                "Use a checkpoint with attention_out_bias=False."
            )

        if _get(hf_config, "mlp_bias", False):
            raise ValueError(
                "Seed-OSS on Neuron does not support mlp_bias=True. "
                "Use a checkpoint with mlp_bias=False."
            )

        hidden_act = _get(hf_config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                f"Seed-OSS on Neuron supports only hidden_act='silu'. The config "
                f"has {hidden_act!r}."
            )

        rope_scaling = _get(hf_config, "rope_scaling") or {}
        rope_parameters = _get(hf_config, "rope_parameters") or {}
        for rope_cfg in (rope_scaling, rope_parameters):
            if not isinstance(rope_cfg, dict):
                continue
            rope_type = rope_cfg.get("rope_type", rope_cfg.get("type", "default"))
            if rope_type not in (None, "default"):
                raise ValueError(
                    f"Seed-OSS on Neuron supports only the default RoPE type. "
                    f"The config has rope_type={rope_type!r}."
                )
