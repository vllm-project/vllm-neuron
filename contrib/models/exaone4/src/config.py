# SPDX-License-Identifier: Apache-2.0
"""
EXAONE 4.0 Config
=================

<-- MODEL-SPECIFIC: All fields in this config are EXAONE 4.0 specific.
EXAONE-4.0-32B: 64 layers, 40 Q heads / 8 KV heads, head_dim=128, hidden=5120.
Key differences from Qwen3:
  - Post-norm decoder layer. There is no input norm. The layer normalizes the
    attention output and the MLP output before each residual add.
  - Hybrid attention. ``layer_types`` gives a sliding-window layer or a full
    layer for each index (pattern "LLLG", window 4096).
  - RoPE applies only on the sliding-window layers. The full layers use no
    position embedding (NoPE).
  - Llama3 RoPE scaling (factor 16, original context 8192).
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.config_utils import hoist_rope_parameters
from vllm_neuron.model.neuron_config import NeuronConfig

SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"

_SUPPORTED_ROPE_TYPES = ("default", "llama3")


@dataclass
class Exaone4Config:
    # <-- MODEL-SPECIFIC: EXAONE-4.0-32B architecture parameters
    vocab_size: int = 102400
    hidden_size: int = 5120
    intermediate_size: int = 27392
    num_hidden_layers: int = 64
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    rope_scaling: dict | None = None
    sliding_window: int | None = 4096
    layer_types: list[str] | None = field(default=None)
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # Framework config
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.layer_types is None:
            # HF default: every 4th layer is a full layer when a window is set.
            self.layer_types = [
                FULL_ATTENTION
                if self.sliding_window is None or (i + 1) % 4 == 0
                else SLIDING_ATTENTION
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries, but "
                f"num_hidden_layers is {self.num_hidden_layers}. Supply one "
                "entry for each decoder layer."
            )
        unknown = set(self.layer_types) - {SLIDING_ATTENTION, FULL_ATTENTION}
        if unknown:
            raise ValueError(
                f"layer_types has unsupported values {sorted(unknown)}. Use "
                f"'{SLIDING_ATTENTION}' or '{FULL_ATTENTION}'."
            )

    def is_sliding(self, layer_idx: int) -> bool:
        """Return True when the layer uses sliding-window attention."""
        return (
            self.sliding_window is not None
            and self.layer_types[layer_idx] == SLIDING_ATTENTION
        )

    def uses_rope(self, layer_idx: int) -> bool:
        """Return True when the layer applies RoPE.

        HF applies RoPE on every layer when no window is set. Otherwise HF
        applies RoPE only on the sliding-window layers.
        """
        return self.sliding_window is None or self.is_sliding(layer_idx)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig = None
    ):
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
            if hasattr(hf_config, "torch_dtype") and hf_config.torch_dtype is not None:
                config_dict["torch_dtype"] = hf_config.torch_dtype
        else:
            config_dict = dict(hf_config)

        if "torch_dtype" not in config_dict and config_dict.get("dtype") is not None:
            config_dict["torch_dtype"] = config_dict["dtype"]

        config_dict = hoist_rope_parameters(config_dict, _SUPPORTED_ROPE_TYPES)
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        if filtered_dict.get("layer_types") is not None:
            filtered_dict["layer_types"] = list(filtered_dict["layer_types"])

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)
