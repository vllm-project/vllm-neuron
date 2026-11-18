# SPDX-License-Identifier: Apache-2.0
"""Factory for Phi3 model selection based on platform and configuration."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .config import SUPPORTED_ROPE_TYPES, Phi3Config


class Phi3ForCausalLM(nn.Module):
    """Factory that validates the config and selects the Phi3 implementation.

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
        """Validate the config. Then make the BF16 implementation."""
        cls._validate_config(hf_config, neuron_config)

        from .model import Phi3ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Make sure that the implementation supports the configuration.

        An unsupported configuration stops here with a clear error. Otherwise it
        can cause a shape mismatch deep in the kernels.
        """
        config = Phi3Config.from_configs(hf_config, None)

        if config.torch_dtype != torch.bfloat16:
            raise ValueError(
                f"Phi3 on Neuron supports only bfloat16. The config dtype is "
                f"{config.torch_dtype!r}. Use a bfloat16 checkpoint or --dtype bfloat16."
            )

        if config.rope_type not in SUPPORTED_ROPE_TYPES:
            raise ValueError(
                f"Phi3 on Neuron does not support rope_type={config.rope_type!r}. "
                f"Use one of {SUPPORTED_ROPE_TYPES}."
            )

        rotary_dim = config.rotary_dim
        if (
            rotary_dim % 2 != 0
            or config.head_dim % 2 != 0
            or not 0 < rotary_dim <= config.head_dim
        ):
            raise ValueError(
                f"Rotary dimension {rotary_dim} is not supported. It must be even, "
                f"larger than 0 and not larger than head_dim ({config.head_dim})."
            )

        if config.rope_type == "longrope":
            half = rotary_dim // 2
            for name in ("rope_short_factor", "rope_long_factor"):
                factor = getattr(config, name)
                if factor is None or len(factor) != half:
                    raise ValueError(
                        f"LongRoPE {name} has length "
                        f"{None if factor is None else len(factor)}. It must have "
                        f"rotary_dim / 2 = {half} entries."
                    )

        # The model has no sliding window attention path. A window that is not
        # smaller than the position range has no effect, so accept it.
        if (
            config.sliding_window is not None
            and config.sliding_window < config.max_position_embeddings
        ):
            raise ValueError(
                f"Phi3 sliding_window={config.sliding_window} is smaller than "
                f"max_position_embeddings={config.max_position_embeddings}. Sliding "
                f"window attention is not supported. Use --disable-sliding-window."
            )
