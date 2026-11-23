# SPDX-License-Identifier: Apache-2.0
"""
Bloom Config
============

<-- MODEL-SPECIFIC: All fields in this config are Bloom-specific.

Bloom uses ALiBi position biases in place of RoPE, LayerNorm with bias on
every norm, a LayerNorm after the word embedding, a fused per-head
interleaved QKV projection with bias, and a non-gated GeLU (tanh) MLP.

The HF config uses the names ``n_layer`` and ``n_head``. This config also
accepts ``num_hidden_layers`` and ``num_attention_heads``.
"""

import json
from dataclasses import dataclass

import torch

from vllm_neuron.model.neuron_config import NeuronConfig

# The Neuron port always computes in bfloat16. The published Bloom checkpoints
# store float16 weights. The loader casts them to bfloat16 one time.
BLOOM_COMPUTE_DTYPE = torch.bfloat16

# Map the HF Bloom field names to the names that this config uses.
_HF_FIELD_ALIASES = {
    "n_layer": "num_hidden_layers",
    "n_head": "num_attention_heads",
    "n_embed": "hidden_size",
}


@dataclass
class BloomConfig:
    # <-- MODEL-SPECIFIC: Architecture parameters (bloomz-560m defaults)
    vocab_size: int = 250880
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    layer_norm_epsilon: float = 1e-5
    apply_residual_connection_post_layernorm: bool = False
    pretraining_tp: int = 1
    slow_but_exact: bool = False
    tie_word_embeddings: bool = True
    torch_dtype: torch.dtype = BLOOM_COMPUTE_DTYPE

    # Framework config
    neuron_config: NeuronConfig | None = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def intermediate_size(self) -> int:
        # Bloom has no intermediate size field. The MLP width is 4 x hidden.
        return 4 * self.hidden_size

    @classmethod
    def from_configs(cls, hf_config, neuron_config: NeuronConfig | None = None):
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, dict):
            config_dict = dict(hf_config)
        else:
            config_dict = hf_config.to_dict()

        for hf_name, name in _HF_FIELD_ALIASES.items():
            if hf_name in config_dict and name not in config_dict:
                config_dict[name] = config_dict[hf_name]

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        # The checkpoint dtype (usually float16) does not select the compute
        # dtype. The port always computes in bfloat16.
        filtered_dict["torch_dtype"] = BLOOM_COMPUTE_DTYPE

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)
