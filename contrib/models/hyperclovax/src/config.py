# SPDX-License-Identifier: Apache-2.0
"""
HyperCLOVAX Config
==================

<-- MODEL-SPECIFIC: All fields in this config are HyperCLOVAX-specific.

HyperCLOVAX is a Llama-style dense decoder with these differences:

- MuP scale factors: ``embedding_multiplier`` (embedding output),
  ``attention_multiplier`` (softmax scale), ``residual_multiplier`` (branch
  output) and ``logits_scaling`` (LM head output, multiply).
- Peri-layer normalization (``use_post_norm``): an extra RMSNorm on the output
  of each attention branch and each MLP branch, before the residual add.
"""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .compat import hoist_rope_parameters


@dataclass
class HyperCLOVAXConfig:
    # <-- MODEL-SPECIFIC: Architecture parameters
    vocab_size: int = 110592
    hidden_size: int = 6144
    intermediate_size: int = 14336
    num_hidden_layers: int = 38
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int | None = 128
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 100000000.0
    tie_word_embeddings: bool = False

    # <-- MODEL-SPECIFIC: MuP scale factors and peri-layer normalization
    embedding_multiplier: float = 1.0
    attention_multiplier: float | None = None
    residual_multiplier: float = 1.0
    logits_scaling: float = 1.0
    use_post_norm: bool = True

    # The compute dtype. The released checkpoint stores float32 weights. The
    # Neuron implementation computes in bfloat16, and the loader casts each
    # weight to bfloat16. vLLM does not write the served dtype into hf_config,
    # thus the checkpoint dtype does not set this field.
    torch_dtype: torch.dtype = torch.bfloat16

    # Framework config
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.attention_multiplier is None:
            self.attention_multiplier = self.head_dim**-0.5

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
        # The checkpoint dtype does not set the compute dtype (see torch_dtype).
        field_names.discard("torch_dtype")
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)
