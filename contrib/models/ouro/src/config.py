# SPDX-License-Identifier: Apache-2.0
"""
Ouro Config
===========

<-- MODEL-SPECIFIC: All fields in this config are Ouro-specific.

Ouro is a looped language model. One stack of decoder layers runs
``total_ut_steps`` times with shared weights. Each decoder layer has two
extra RMSNorms (``input_layernorm_2`` and ``post_attention_layernorm_2``)
that apply to the attention output and to the MLP output.
"""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .compat import hoist_rope_parameters


@dataclass
class OuroConfig:
    # <-- MODEL-SPECIFIC: Architecture parameters (ByteDance/Ouro-1.4B)
    vocab_size: int = 49152
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 128
    max_position_embeddings: int = 65536
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # <-- MODEL-SPECIFIC: Universal Transformer loop and early exit.
    # The stack of layers runs total_ut_steps times. The exit step for each
    # token is the first step where the cumulative exit probability is at or
    # above early_exit_threshold. If no step reaches the threshold, the model
    # uses the last step. A threshold of None also selects the last step, as
    # in the HF model.
    total_ut_steps: int = 4
    early_exit_threshold: float | None = 1.0

    # Framework config
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

    @property
    def num_kv_layers(self) -> int:
        """Number of KV cache layers. Each UT step has its own KV cache."""
        return self.num_hidden_layers * self.total_ut_steps

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
