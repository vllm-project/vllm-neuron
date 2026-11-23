# SPDX-License-Identifier: Apache-2.0
"""
OLMo Config
===========

<-- MODEL-SPECIFIC: All fields in this config are OLMo-specific.

OLMo (``OlmoForCausalLM``) is a Llama-style decoder with these differences:

- Each norm is a non-parametric LayerNorm. It has no weight and no bias.
- The checkpoint can clamp Q/K/V with ``clip_qkv``. This port supports
  ``clip_qkv=None`` only. The factory rejects other values.

The original checkpoint stores FP32 weights. The NKI kernels need bf16, so
the compute dtype is always bf16. The loader casts each weight once at load.
"""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .compat import hoist_rope_parameters

# HF ``OlmoLayerNorm`` uses a fixed epsilon. The config does not hold it.
OLMO_LAYER_NORM_EPS = 1e-5


@dataclass
class OlmoConfig:
    # <-- MODEL-SPECIFIC: Architecture parameters (allenai/OLMo-1B-hf defaults)
    vocab_size: int = 50304
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    attention_bias: bool = False
    clip_qkv: float | None = None
    tie_word_embeddings: bool = True
    layer_norm_eps: float = OLMO_LAYER_NORM_EPS
    torch_dtype: torch.dtype = torch.bfloat16

    # Framework config
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
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

        # <-- MODEL-SPECIFIC: Compute in bf16 for each checkpoint dtype. The
        # loader casts the FP32 checkpoint weights to bf16 one time.
        filtered_dict["torch_dtype"] = torch.bfloat16
        filtered_dict["layer_norm_eps"] = OLMO_LAYER_NORM_EPS

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)
