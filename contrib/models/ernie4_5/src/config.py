# SPDX-License-Identifier: Apache-2.0
"""
ERNIE 4.5 Config
================

<-- MODEL-SPECIFIC: All fields in this config are ERNIE 4.5 dense fields.

ERNIE 4.5 dense is a Llama-type decoder. It uses interleaved RoPE (GLM style)
and has no bias by default. See ``model.py`` for the RoPE layout conversion.
"""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.config_utils import hoist_rope_parameters
from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class Ernie4_5Config:
    # <-- MODEL-SPECIFIC: Architecture parameters (defaults: ERNIE-4.5-0.3B)
    vocab_size: int = 103424
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 18
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int | None = 128
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    rope_scaling: dict | None = None
    use_bias: bool = False
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True
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
        """Create the config from a HF config, a config.json path, or a dict.

        Args:
            hf_config: A ``PretrainedConfig``, a path to ``config.json``, or a dict.
            neuron_config: The optional Neuron framework config.

        Returns:
            The ``Ernie4_5Config``.

        Example::

            config = Ernie4_5Config.from_configs("/path/to/model/config.json")
        """
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
            if hasattr(hf_config, "torch_dtype") and hf_config.torch_dtype is not None:
                config_dict["torch_dtype"] = hf_config.torch_dtype
        else:
            config_dict = dict(hf_config)

        # The HF config.json writes "dtype". The dataclass uses "torch_dtype".
        if config_dict.get("torch_dtype") is None and "dtype" in config_dict:
            config_dict["torch_dtype"] = config_dict["dtype"]

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
