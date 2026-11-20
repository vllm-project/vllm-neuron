# SPDX-License-Identifier: Apache-2.0
"""
Granite Config
==============

<-- MODEL-SPECIFIC: All fields in this config are Granite-specific.

Granite (for example ibm/PowerLM-3b) is a Llama-style dense model with four
extra scalars:

- ``embedding_multiplier`` multiplies the token embeddings.
- ``attention_multiplier`` is the softmax scale. It replaces ``head_dim**-0.5``.
- ``residual_multiplier`` multiplies each sub-layer output before the
  residual add.
- ``logits_scaling`` divides the final logits.

The checkpoint can also have a bias on each attention and MLP projection.
"""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .compat import hoist_rope_parameters


@dataclass
class GraniteConfig:
    # <-- MODEL-SPECIFIC: Architecture parameters (defaults: ibm/PowerLM-3b)
    vocab_size: int = 49152
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 40
    num_attention_heads: int = 36
    num_key_value_heads: int = 36
    head_dim: int | None = None
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_act: str = "silu"
    torch_dtype: torch.dtype = torch.bfloat16

    # <-- MODEL-SPECIFIC: Granite scalars
    embedding_multiplier: float = 1.0
    attention_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    logits_scaling: float = 1.0

    # Framework config
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if isinstance(self.torch_dtype, str):
            self.torch_dtype = getattr(torch, self.torch_dtype)
        # Some Granite checkpoints (for example ibm/PowerLM-3b) keep FP32
        # weights and set no dtype. Run these checkpoints in bfloat16. The
        # weight loader casts each tensor to this dtype.
        if self.torch_dtype is None or self.torch_dtype == torch.float32:
            self.torch_dtype = torch.bfloat16

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
