# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 config for the Neuron backend.

Qwen3.5 is a hybrid model: ``config.layer_types`` lists either
``"full_attention"`` or ``"linear_attention"`` per layer. Linear-attention
layers use the GatedDeltaNet (GDN) recurrence, sized by the ``linear_*``
fields. Full-attention layers are *gated* attention (the Q projection doubles
as a sigmoid gate) with partial RoPE, and share the Qwen3.5 RMSNorm
(``x * (1 + weight)``).

Unlike Qwen3, the RoPE geometry lives in ``rope_parameters`` (dict), not a bare
``rope_theta``: it carries ``rope_theta``, ``rope_type``, and
``partial_rotary_factor`` (0.25 by default — only the first quarter of each
head is rotated).
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class Qwen3_5Config:
    # Dense full-attention geometry (gated attention).
    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # RoPE. rope_theta/partial_rotary_factor are convenience mirrors of the
    # rope_parameters dict; from_configs fills them from rope_parameters.
    rope_theta: float = 10_000.0
    partial_rotary_factor: float = 0.25
    rope_parameters: dict | None = None

    # GatedDeltaNet (linear-attention) geometry.
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # Hybrid layer-type map: "full_attention" or "linear_attention" per layer.
    layer_types: list[str] = field(default_factory=list)

    # Framework config.
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

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
            config_dict = hf_config

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        # rope_parameters is authoritative for Qwen3.5 RoPE geometry. The
        # top-level ``partial_rotary_factor`` is a BC default injected by
        # transformers (0.25), so it must not mask a value carried in
        # rope_parameters.
        rope_params = filtered_dict.get("rope_parameters") or {}
        if isinstance(rope_params, str):
            rope_params = {}
        for key in ("rope_theta", "partial_rotary_factor"):
            if key in rope_params:
                filtered_dict[key] = rope_params[key]

        # Legacy fallback: top-level rope_scaling (Qwen2/Qwen3-era) for any key
        # rope_parameters does not carry.
        if "rope_scaling" in filtered_dict and isinstance(
            filtered_dict["rope_scaling"], dict
        ):
            for key in ("rope_theta", "partial_rotary_factor", "rope_type"):
                if key not in rope_params and key in filtered_dict["rope_scaling"]:
                    filtered_dict[key] = filtered_dict["rope_scaling"][key]

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)
