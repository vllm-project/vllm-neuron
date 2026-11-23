# SPDX-License-Identifier: Apache-2.0
"""
GLM Config
==========

<-- MODEL-SPECIFIC: All fields in this config are GLM-specific.

GLM (``GlmForCausalLM``, for example ``zai-org/glm-4-9b-chat-hf``) is a dense
GQA decoder. It differs from Llama in three ways:

- Q, K and V projections have a bias. The output projection has no bias.
- RoPE rotates only the first ``head_dim * partial_rotary_factor`` dimensions
  of each head, and it rotates interleaved pairs (2i, 2i+1).
- The checkpoint stores the gate and up projections as one fused tensor
  (``mlp.gate_up_proj.weight``, gate rows first).
"""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .compat import hoist_rope_parameters


@dataclass
class GlmConfig:
    # <-- MODEL-SPECIFIC: Architecture parameters (defaults: glm-4-9b-chat-hf)
    vocab_size: int = 151552
    hidden_size: int = 4096
    intermediate_size: int = 13696
    num_hidden_layers: int = 40
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1.5625e-07
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 0.5
    attention_bias: bool = True
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # Framework config
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @property
    def rotary_dim(self) -> int:
        """Number of head dimensions that RoPE rotates (HF GLM: int(head_dim * factor))."""
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

        # transformers >= 5 writes "dtype" in place of "torch_dtype".
        if config_dict.get("torch_dtype") is None and config_dict.get("dtype"):
            config_dict["torch_dtype"] = config_dict["dtype"]

        # transformers >= 5 can keep partial_rotary_factor in rope_parameters.
        rope_parameters = config_dict.get("rope_parameters")
        if (
            config_dict.get("partial_rotary_factor") is None
            and isinstance(rope_parameters, dict)
            and rope_parameters.get("partial_rotary_factor") is not None
        ):
            config_dict["partial_rotary_factor"] = rope_parameters[
                "partial_rotary_factor"
            ]
        if config_dict.get("partial_rotary_factor") is None:
            config_dict.pop("partial_rotary_factor", None)

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
