# SPDX-License-Identifier: Apache-2.0
"""HuggingFace configuration class for Ouro.

The Ouro checkpoint has ``model_type: "ouro"``, and transformers does not
include this model type. The checkpoint supplies a ``configuration_ouro.py``
through ``auto_map``. This module gives a local configuration class, so that
vLLM can parse the checkpoint config without ``--trust-remote-code``.

Importing this module registers the class with ``AutoConfig``.
"""

from typing import Any

from transformers import AutoConfig, PretrainedConfig


class OuroConfig(PretrainedConfig):
    """Configuration for an Ouro looped (Universal Transformer) language model.

    The fields match the ``config.json`` of ``ByteDance/Ouro-1.4B``. The model
    applies its stack of ``num_hidden_layers`` decoder layers
    ``total_ut_steps`` times. An early-exit gate after each step selects the
    hidden state that the language-model head uses.

    Example:
        >>> config = OuroConfig(num_hidden_layers=2, total_ut_steps=4)
        >>> config.model_type
        'ouro'
    """

    model_type = "ouro"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 49152,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int | None = 16,
        head_dim: int | None = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 65536,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1000000.0,
        rope_scaling: dict[str, Any] | None = None,
        use_sliding_window: bool = False,
        sliding_window: int | None = None,
        max_window_layers: int = 24,
        layer_types: list[str] | None = None,
        attention_dropout: float = 0.0,
        total_ut_steps: int = 4,
        early_exit_threshold: float | None = 1.0,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers
        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers
        self.layer_types = layer_types
        self.attention_dropout = attention_dropout
        self.total_ut_steps = total_ut_steps
        self.early_exit_threshold = early_exit_threshold
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


AutoConfig.register("ouro", OuroConfig, exist_ok=True)
