# SPDX-License-Identifier: Apache-2.0
"""
StableLM Config
===============

<-- MODEL-SPECIFIC: All fields in this config are StableLM-specific.

StableLM is a dense decoder. It differs from Llama in three areas:

- Each norm is a full LayerNorm with a bias, not an RMSNorm.
- RoPE rotates only the first ``partial_rotary_factor * head_dim`` dims of
  each head. The other dims pass through unchanged.
- The config can enable a QKV bias, a per-head QK LayerNorm and a parallel
  residual. The factory refuses these options, because this implementation
  does not have them.
"""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .compat import hoist_rope_parameters


@dataclass
class StableLmConfig:
    # <-- MODEL-SPECIFIC: Architecture parameters (stablelm-zephyr-3b defaults)
    vocab_size: int = 50304
    hidden_size: int = 2560
    intermediate_size: int = 6912
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int | None = None
    max_position_embeddings: int = 4096
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 0.25
    hidden_act: str = "silu"
    use_qkv_bias: bool = False
    qk_layernorm: bool = False
    use_parallel_residual: bool = False
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # Framework config
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @property
    def rotary_dim(self) -> int:
        """Number of dims in each head that RoPE rotates."""
        return int(self.head_dim * self.partial_rotary_factor)

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

        # transformers >= 5 keeps partial_rotary_factor in rope_parameters.
        # hoist_rope_parameters moves rope_theta only, so move this key here.
        rope_parameters = config_dict.get("rope_parameters")
        if isinstance(rope_parameters, dict):
            if "partial_rotary_factor" in rope_parameters:
                config_dict.setdefault(
                    "partial_rotary_factor", rope_parameters["partial_rotary_factor"]
                )
        config_dict = hoist_rope_parameters(config_dict)

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)
