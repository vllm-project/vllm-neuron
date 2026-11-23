# SPDX-License-Identifier: Apache-2.0
"""Factory for Bloom model selection based on platform and configuration."""

import torch
import torch.nn as nn

from vllm_neuron.model.neuron_config import NeuronConfig

from .config import BloomConfig

# The ALiBi key position goes into the head dimension as two bfloat16 values
# a and b, with position = 64 * a + b. bfloat16 holds each integer exactly
# up to 256, so the encoding is exact for positions below 64 * 256.
BLOOM_MAX_EXACT_POSITION = 64 * 256

# The ALiBi encoding needs 6 free lanes in the 128-wide padded head
# (BLOOM_PADDED_HEAD_DIM - BLOOM_ALIBI_LANES in model.py).
BLOOM_MAX_HEAD_DIM = 128 - 6

_SUPPORTED_CHECKPOINT_DTYPES = (
    None,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    "float16",
    "bfloat16",
    "float32",
)


class BloomForCausalLM(nn.Module):
    """Factory that validates config and selects the Bloom implementation.

    This class extends nn.Module to satisfy vLLM's ModelRegistry requirements.
    The factory stores the selected implementation and delegates forward() calls to it.
    """

    def __init__(
        self,
        hf_config,
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
    def from_configs(cls, hf_config, neuron_config: NeuronConfig | None) -> nn.Module:
        """Create model from configs. Returns the selected implementation directly."""
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Select and instantiate the appropriate implementation based on config."""
        cls._validate_config(hf_config, neuron_config)

        from .model import BloomForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(cls, hf_config, neuron_config: NeuronConfig | None) -> None:
        """Validate that the configuration is supported.

        Stop at construction with a clear error. A wrong configuration can
        otherwise give a shape mismatch deep in the kernels.
        """
        torch_dtype = getattr(hf_config, "torch_dtype", None)
        if torch_dtype not in _SUPPORTED_CHECKPOINT_DTYPES:
            raise ValueError(
                f"Bloom checkpoint dtype {torch_dtype!r} is not supported. "
                "Use a float16, bfloat16 or float32 checkpoint."
            )

        if getattr(hf_config, "apply_residual_connection_post_layernorm", False):
            raise ValueError(
                "Bloom with apply_residual_connection_post_layernorm=True is not "
                "supported. Use a checkpoint that sets this field to False."
            )

        if (
            getattr(hf_config, "slow_but_exact", False)
            and getattr(hf_config, "pretraining_tp", 1) > 1
        ):
            raise ValueError(
                "Bloom with slow_but_exact=True and pretraining_tp > 1 is not "
                "supported. Set slow_but_exact to False."
            )

        config = BloomConfig.from_configs(hf_config, neuron_config)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) is not divisible by the "
                f"number of heads ({config.num_attention_heads})."
            )
        if config.head_dim > BLOOM_MAX_HEAD_DIM:
            # The ALiBi encoding writes lanes head_dim .. head_dim + 5 of the
            # 128-wide padded head (see BLOOM_ALIBI_LANES in model.py).
            raise ValueError(
                f"Bloom head_dim {config.head_dim} is larger than "
                f"{BLOOM_MAX_HEAD_DIM}. The ALiBi encoding needs 6 free lanes in "
                "a 128-wide head."
            )


def validate_max_model_len(max_model_len: int | None) -> None:
    """Check that the ALiBi position encoding is exact for max_model_len.

    Args:
        max_model_len: The maximum sequence length of the engine, or None.

    Raises:
        ValueError: If max_model_len is larger than BLOOM_MAX_EXACT_POSITION.

    Example:
        >>> validate_max_model_len(2048)
    """
    if isinstance(max_model_len, int) and max_model_len > BLOOM_MAX_EXACT_POSITION:
        raise ValueError(
            f"max_model_len {max_model_len} is larger than "
            f"{BLOOM_MAX_EXACT_POSITION}. The ALiBi position encoding is exact "
            f"only below this value. Set --max-model-len to "
            f"{BLOOM_MAX_EXACT_POSITION} or less."
        )
