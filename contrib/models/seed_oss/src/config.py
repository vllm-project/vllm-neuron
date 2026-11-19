# SPDX-License-Identifier: Apache-2.0
"""
Seed-OSS Config
===============

<-- MODEL-SPECIFIC: All fields in this config are Seed-OSS-specific.

Seed-OSS is a Llama-style dense decoder. It adds a bias to the Q, K, and V
projections (``attention_bias``). The output projection and the MLP have no
bias.
"""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .compat import hoist_rope_parameters


@dataclass
class SeedOssConfig:
    # <-- MODEL-SPECIFIC: Architecture parameters (Seed-OSS-36B defaults)
    vocab_size: int = 155136
    hidden_size: int = 5120
    intermediate_size: int = 27648
    num_hidden_layers: int = 64
    num_attention_heads: int = 80
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 524288
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    rope_scaling: dict | None = None
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # <-- MODEL-SPECIFIC: Bias flags. The Neuron port supports
    # attention_bias=True or False, attention_out_bias=False and mlp_bias=False.
    attention_bias: bool = True
    attention_out_bias: bool = False
    mlp_bias: bool = False

    # Framework config
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
            config_dict = dict(hf_config)

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
