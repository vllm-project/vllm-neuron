# SPDX-License-Identifier: Apache-2.0
"""Factory for StableLM model selection based on platform and configuration."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


def _get_hf_field(hf_config: PretrainedConfig, name: str, default=None):
    """Read a field from the top level or from ``rope_parameters``."""
    value = getattr(hf_config, name, None)
    if value is not None:
        return value
    rope_parameters = getattr(hf_config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and name in rope_parameters:
        return rope_parameters[name]
    return default


class StableLmForCausalLM(nn.Module):
    """Factory that validates the config and selects the StableLM implementation.

    This class extends nn.Module to satisfy the ModelRegistry requirements of vLLM.
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
        """Create the model from the configs. Return the implementation directly."""
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Validate the config. Then make the BF16 implementation."""
        cls._validate_config(hf_config, neuron_config)

        from .model import StableLmForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Stop at construction if the config has an unsupported option.

        A clear error here is better than a shape error in a kernel later.
        """
        torch_dtype = getattr(hf_config, "torch_dtype", None)
        if torch_dtype is not None and torch_dtype not in (
            torch.bfloat16,
            "bfloat16",
        ):
            raise ValueError(
                f"StableLM on Neuron supports bfloat16 only. The config has "
                f"{torch_dtype!r}. Use a bfloat16 checkpoint."
            )

        for flag in ("use_qkv_bias", "qk_layernorm", "use_parallel_residual"):
            if getattr(hf_config, flag, False):
                raise ValueError(
                    f"StableLM option {flag}=True is not supported on Neuron. "
                    f"Use a checkpoint that sets {flag}=False."
                )

        hidden_act = getattr(hf_config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                f"StableLM activation {hidden_act!r} is not supported on Neuron. "
                "Use a checkpoint with hidden_act='silu'."
            )

        # transformers >= 5 keeps the RoPE type in rope_parameters, earlier
        # versions in rope_scaling. This implementation has the default RoPE only.
        for key in ("rope_parameters", "rope_scaling"):
            rope = getattr(hf_config, key, None)
            if isinstance(rope, dict):
                rope_type = rope.get("rope_type", rope.get("type"))
                if rope_type not in (None, "default"):
                    raise ValueError(
                        f"StableLM {key} {rope!r} is not supported on Neuron. "
                        "Use a checkpoint with the default RoPE (no rope_scaling)."
                    )

        hidden_size = hf_config.hidden_size
        num_heads = hf_config.num_attention_heads
        head_dim = getattr(hf_config, "head_dim", None) or hidden_size // num_heads
        factor = _get_hf_field(hf_config, "partial_rotary_factor", 1.0)
        rotary_dim = int(head_dim * factor)
        if rotary_dim % 2 != 0 or rotary_dim > head_dim or rotary_dim == 0:
            raise ValueError(
                f"Rotary dim {rotary_dim} is not valid for head_dim {head_dim}. "
                "The rotary dim must be even, larger than 0 and not larger "
                "than head_dim."
            )
