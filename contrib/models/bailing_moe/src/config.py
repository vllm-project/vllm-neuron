# SPDX-License-Identifier: Apache-2.0
"""
BailingMoeV2 Config
===================

<-- MODEL-SPECIFIC: All fields in this config are BailingMoeV2-specific.
Reference model: inclusionAI/Ling-mini-2.0 (20 layers, hidden 2048, 16 Q heads,
4 KV heads, head_dim 128). Layer 0 is a dense MLP. Layers 1 to 19 are MoE layers
with 256 routed experts, top-8 routing and one shared expert.

Differences from Qwen3:
  - Fused query_key_value checkpoint tensor (Q heads, then K heads, then V heads).
  - Partial RoPE: only the first ``partial_rotary_factor * head_dim`` dims rotate.
  - Sigmoid router with an expert bias and group-limited top-k selection.
"""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .compat import hoist_rope_parameters


@dataclass
class BailingMoeV2Config:
    # <-- MODEL-SPECIFIC: Ling-mini-2.0 architecture parameters
    vocab_size: int = 157184
    hidden_size: int = 2048
    intermediate_size: int = 5120
    num_hidden_layers: int = 20
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int | None = 128
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 600000.0
    rope_scaling: dict | None = None
    partial_rotary_factor: float = 0.5
    use_qk_norm: bool = True
    use_bias: bool = False
    use_qkv_bias: bool = False
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # <-- MODEL-SPECIFIC: MoE parameters
    first_k_dense_replace: int = 1
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    num_shared_experts: int | None = 1
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    score_function: str = "sigmoid"
    moe_router_enable_expert_bias: bool = True
    num_nextn_predict_layers: int = 0

    # Framework config
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @property
    def rotary_dim(self) -> int:
        """Return the number of head dims that RoPE rotates."""
        return int(self.head_dim * self.partial_rotary_factor)

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Return True when the decoder layer ``layer_idx`` uses the MoE block."""
        return self.num_experts is not None and layer_idx >= self.first_k_dense_replace

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

        # transformers >= 5 can keep partial_rotary_factor in rope_parameters.
        rope_parameters = config_dict.get("rope_parameters")
        if (
            isinstance(rope_parameters, dict)
            and "partial_rotary_factor" in rope_parameters
        ):
            config_dict = dict(config_dict)
            config_dict.setdefault(
                "partial_rotary_factor", rope_parameters["partial_rotary_factor"]
            )
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
