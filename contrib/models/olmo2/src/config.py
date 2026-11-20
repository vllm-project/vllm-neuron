# SPDX-License-Identifier: Apache-2.0
"""
OLMo2 Config
============

<-- MODEL-SPECIFIC: All fields in this config are OLMo2-specific.

OLMo2 differs from Llama in two places:
- Each decoder layer applies RMSNorm after the attention and after the MLP
  (post-norm). The layer has no input norm.
- Q and K each have an RMSNorm over the full projected width (all heads),
  before RoPE.
"""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .compat import hoist_rope_parameters


@dataclass
class Olmo2Config:
    # <-- MODEL-SPECIFIC: Architecture parameters (OLMo-2-0425-1B defaults)
    vocab_size: int = 100352
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int | None = None
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = False
    # The compute dtype. The checkpoint can store FP32 weights. The loader
    # casts each weight to this dtype.
    torch_dtype: torch.dtype = torch.bfloat16

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
        else:
            config_dict = dict(hf_config)

        config_dict = hoist_rope_parameters(config_dict)
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        # The checkpoint stores FP32 weights. The Neuron implementation computes
        # in BF16, so ignore the checkpoint dtype here.
        filtered_dict["torch_dtype"] = torch.bfloat16

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)
