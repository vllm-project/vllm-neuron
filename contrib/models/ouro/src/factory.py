# SPDX-License-Identifier: Apache-2.0
"""Factory for Ouro model selection based on platform and configuration."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


def _get(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


class OuroForCausalLM(nn.Module):
    """Factory that validates the config and selects the Ouro implementation.

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

        from .model import OuroForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Make sure that the implementation supports the configuration.

        Stop at construction with a clear error. A configuration that the
        implementation does not support can otherwise give a shape error in a
        kernel, or a silent numerical difference.
        """
        torch_dtype = _get(hf_config, "torch_dtype")
        if torch_dtype is not None and torch_dtype not in (
            torch.bfloat16,
            "bfloat16",
        ):
            raise ValueError(
                f"Ouro on Neuron supports only bfloat16. The config has {torch_dtype!r}."
            )

        total_ut_steps = _get(hf_config, "total_ut_steps", 4)
        if total_ut_steps is None or int(total_ut_steps) < 1:
            raise ValueError(
                f"total_ut_steps is {total_ut_steps!r}. Set it to 1 or more."
            )

        if _get(hf_config, "early_exit_step") is not None:
            raise ValueError(
                "early_exit_step is set. The Neuron implementation supports only "
                "threshold exit. Remove early_exit_step from the config."
            )

        # A recent transformers version keeps rope_theta in rope_scaling or in
        # rope_parameters, with rope_type "default". Accept the default type only.
        for key in ("rope_scaling", "rope_parameters"):
            rope = _get(hf_config, key)
            if rope is None:
                continue
            rope_type = (
                rope.get("rope_type", rope.get("type", "default"))
                if isinstance(rope, dict)
                else rope
            )
            if rope_type != "default":
                raise ValueError(
                    f"{key} is {rope!r}. The Neuron implementation supports only "
                    "default RoPE. Remove the RoPE scaling from the config."
                )

        if _get(hf_config, "use_sliding_window", False):
            raise ValueError(
                "use_sliding_window is true. The Neuron implementation supports "
                "only full attention. Set use_sliding_window to false."
            )

        layer_types = _get(hf_config, "layer_types")
        if layer_types is not None and any(t != "full_attention" for t in layer_types):
            raise ValueError(
                "layer_types contains a layer that is not full_attention. The "
                "Neuron implementation supports only full attention."
            )

        hidden_act = _get(hf_config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                f"hidden_act is {hidden_act!r}. The Neuron implementation supports "
                "only silu."
            )
