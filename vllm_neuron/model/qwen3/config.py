# SPDX-License-Identifier: Apache-2.0
"""
Qwen3 Config
============

<-- MODEL-SPECIFIC: All fields in this config are Qwen3-specific.
Qwen3-32B: 64 layers, 64 Q heads / 8 KV heads, head_dim=128, hidden=5120.
Key differences from Llama:
  - QK-norm applied per-head (q_norm / k_norm on head_dim)
  - Standard RoPE (no Llama3-style scaling)
  - rope_theta = 1_000_000
  - tie_word_embeddings = False
  - no sliding window for the 32B dense variant (use_sliding_window=False)
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.config_utils import hoist_rope_parameters
from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class Qwen3Config:
    # <-- MODEL-SPECIFIC: Qwen3-32B architecture parameters
    vocab_size: int = 151936
    hidden_size: int = 5120
    intermediate_size: int = 25600
    num_hidden_layers: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    rope_scaling: dict | None = None
    tie_word_embeddings: bool = False
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
            if hasattr(hf_config, "torch_dtype") and hf_config.torch_dtype is not None:
                config_dict["torch_dtype"] = hf_config.torch_dtype
        else:
            config_dict = hf_config

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


@dataclass
class DFlashConfig(Qwen3Config):
    """
    Config for z-lab DFlash checkpoints using the Qwen3-based drafter architecture.
    Reference checkpoint: z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat
    """

    # Drafter Qwen3 architecture params
    vocab_size: int = 128256
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    attention_bias: bool = False

    # DFlash params
    block_size: int = 10
    mask_token_id: int | None = 128002
    target_layer_ids: list[int] = field(default_factory=lambda: [1, 8, 15, 22, 29])

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig = None
    ):
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
            if getattr(hf_config, "torch_dtype", None) is not None:
                config_dict["torch_dtype"] = hf_config.torch_dtype
        else:
            config_dict = dict(hf_config)

        nested = config_dict.get("dflash_config", {}) or {}

        if "block_size" not in config_dict and "block_size" not in nested:
            raise ValueError(
                "DFlash draft config must define block_size either directly "
                "or in dflash_config"
            )

        for key in ("block_size", "mask_token_id", "target_layer_ids"):
            if key in nested:
                config_dict[key] = nested[key]

        return super().from_configs(config_dict, neuron_config)
