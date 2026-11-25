# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 Config (dense hybrid, Qwen3.6/3.8-27B)
===============================================

Hybrid DeltaNet + Full Attention architecture.
64 layers: 48 linear_attention (DeltaNet) + 16 full_attention (GQA).
Pattern: [3 DeltaNet + 1 GQA] repeated 16 times.

Full attention: 24 Q heads / 4 KV heads, head_dim=256, partial_rotary_factor=0.25
DeltaNet: 16 key heads (dim=128), 48 value heads (dim=128), conv1d kernel=4
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class Qwen3_5Config:
    # Architecture
    vocab_size: int = 248320
    hidden_size: int = 5120
    intermediate_size: int = 17408
    num_hidden_layers: int = 64
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # Full attention parameters
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    max_position_embeddings: int = 262144

    # RoPE / M-RoPE
    rope_theta: float = 10_000_000.0
    rope_scaling: dict | None = None
    rope_parameters: dict | None = field(
        default_factory=lambda: {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000,
            "rope_type": "default",
        }
    )

    # DeltaNet parameters
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_conv_kernel_dim: int = 4
    output_gate_type: str = "swish"
    attn_output_gate: bool = True

    # Layer type pattern
    layer_types: list[str] = field(
        default_factory=lambda: [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
        * 16
    )
    full_attention_interval: int = 4

    # Framework config
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def num_full_attention_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "full_attention")

    @property
    def num_linear_attention_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "linear_attention")

    def is_full_attention(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "full_attention"

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
            # HF config nests text params under text_config for VL models
            if "text_config" in config_dict:
                text_dict = config_dict["text_config"]
                for k, v in text_dict.items():
                    if k not in config_dict or k == "model_type":
                        config_dict[k] = v
        else:
            config_dict = hf_config

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        # Handle dtype field from HF (text_config uses "dtype" not "torch_dtype")
        if "torch_dtype" not in filtered_dict and "dtype" in config_dict:
            dtype_val = config_dict["dtype"]
            if isinstance(dtype_val, str):
                filtered_dict["torch_dtype"] = getattr(torch, dtype_val)

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)
