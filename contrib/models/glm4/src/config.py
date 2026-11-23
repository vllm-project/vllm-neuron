# SPDX-License-Identifier: Apache-2.0
"""
GLM-4 Config
============

<-- MODEL-SPECIFIC: All fields in this config are GLM-4-specific.

GLM-4 (``Glm4ForCausalLM``) is a dense GQA decoder with two differences from Llama:

- Each decoder layer has four RMSNorms ("sandwich" norms). The extra norms
  apply to the attention output and to the MLP output before each residual add.
- RoPE rotates only the first ``partial_rotary_factor * head_dim`` dimensions of
  each head, and it uses interleaved (even, odd) pairs.
"""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .compat import hoist_rope_parameters


@dataclass
class Glm4Config:
    # <-- MODEL-SPECIFIC: Architecture parameters (defaults: GLM-4-32B-0414)
    vocab_size: int = 151552
    hidden_size: int = 6144
    intermediate_size: int = 23040
    num_hidden_layers: int = 61
    num_attention_heads: int = 48
    num_key_value_heads: int = 2
    head_dim: int = 128
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 0.5
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # Framework config
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.partial_rotary_factor is None:
            self.partial_rotary_factor = 1.0

    @property
    def rotary_dim(self) -> int:
        """Number of dimensions in each head that RoPE rotates."""
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

        # transformers >= 5 can move partial_rotary_factor into rope_parameters.
        rope_parameters = config_dict.get("rope_parameters")
        if (
            config_dict.get("partial_rotary_factor") is None
            and isinstance(rope_parameters, dict)
            and "partial_rotary_factor" in rope_parameters
        ):
            config_dict["partial_rotary_factor"] = rope_parameters[
                "partial_rotary_factor"
            ]

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
