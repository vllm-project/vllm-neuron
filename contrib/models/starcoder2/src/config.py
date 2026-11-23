# SPDX-License-Identifier: Apache-2.0
"""
StarCoder2 Config
=================

<-- MODEL-SPECIFIC: All fields in this config are StarCoder2-specific.

StarCoder2 is a dense GQA decoder. It is different from Llama in these items:

- LayerNorm with a bias, in place of RMSNorm (``norm_epsilon``).
- A non-gated MLP (``c_fc`` -> GELU tanh -> ``c_proj``).
- A bias on each linear projection (``use_bias``).
- A sliding window field (``sliding_window``).
"""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .compat import hoist_rope_parameters


@dataclass
class Starcoder2Config:
    # <-- MODEL-SPECIFIC: Architecture parameters (defaults are StarCoder2-3B)
    vocab_size: int = 49152
    hidden_size: int = 3072
    intermediate_size: int = 12288
    num_hidden_layers: int = 30
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int | None = None
    max_position_embeddings: int = 16384
    norm_epsilon: float = 1e-5
    rope_theta: float = 999999.4420358813
    hidden_act: str = "gelu_pytorch_tanh"
    use_bias: bool = True
    sliding_window: int | None = None
    tie_word_embeddings: bool = True

    # The checkpoint stores FP32 weights. The Neuron implementation computes in
    # BF16 only. The loader casts each weight to this dtype.
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

        # The compute dtype is always BF16. Ignore the checkpoint dtype here.
        filtered_dict.pop("torch_dtype", None)

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)
