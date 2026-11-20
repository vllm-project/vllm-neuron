# SPDX-License-Identifier: Apache-2.0
"""Factory for Nanbeige model selection based on configuration."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

# Each of these optional Nanbeige features changes the forward pass. The
# Neuron implementation does not support them, so the factory refuses a
# configuration that enables one of them.
_UNSUPPORTED_FLAGS = (
    "qk_layernorm",
    "attention_bias",
    "mlp_bias",
    "enable_hyper_connection",
    "enable_mhc",
    "enable_depth_attention",
    "enable_double_loop_split",
    "loop_share_kv",
)


def _get(hf_config, name, default=None):
    if isinstance(hf_config, dict):
        return hf_config.get(name, default)
    return getattr(hf_config, name, default)


class NanbeigeForCausalLM(nn.Module):
    """Factory that validates the config and selects the Nanbeige implementation.

    This class extends nn.Module to satisfy the ModelRegistry requirements of
    vLLM. The factory stores the selected implementation and delegates
    forward() calls to it.
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
        """Delegate the forward pass to the selected implementation."""
        return self._model(*args, input_ids=input_ids, positions=positions, **kwargs)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Create the model from configs. Return the implementation directly."""
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)

        from .model import NanbeigeForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Refuse a configuration that the Neuron implementation does not support.

        A refusal at construction gives a clear error. Without it, the model
        can load and then give incorrect output.
        """
        torch_dtype = _get(hf_config, "torch_dtype") or _get(hf_config, "dtype")
        if torch_dtype is not None and torch_dtype not in (
            torch.bfloat16,
            "bfloat16",
        ):
            raise ValueError(
                f"Nanbeige dtype {torch_dtype!r} is not supported on Neuron. "
                f"Use bfloat16."
            )

        for flag in _UNSUPPORTED_FLAGS:
            if _get(hf_config, flag, False):
                raise ValueError(
                    f"Nanbeige option {flag}=True is not supported on Neuron. "
                    f"Use a checkpoint that sets {flag} to False."
                )

        if _get(hf_config, "emb_neighbor_num") is not None:
            raise ValueError(
                "Nanbeige n-gram embeddings (emb_neighbor_num) are not supported "
                "on Neuron. Use a checkpoint without n-gram embeddings."
            )

        if _get(hf_config, "loop_loss_weights"):
            raise ValueError(
                "Nanbeige loop_loss_weights is not supported on Neuron. "
                "Set num_loops and keep loop_loss_weights empty."
            )

        hidden_act = _get(hf_config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                f"Nanbeige hidden_act {hidden_act!r} is not supported on Neuron. "
                f"Use silu."
            )

        rope_scaling = _get(hf_config, "rope_scaling")
        if isinstance(rope_scaling, dict):
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
            if rope_type not in (None, "default"):
                raise ValueError(
                    f"Nanbeige RoPE scaling {rope_type!r} is not supported on "
                    f"Neuron. Use a checkpoint without rope_scaling."
                )
