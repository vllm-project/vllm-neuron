# SPDX-License-Identifier: Apache-2.0
"""Qwen3 Configuration."""

from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class Qwen3Config:
    """Qwen3 configuration matching HF transformers."""

    # Architecture
    vocab_size: int = 151936
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 11008
    moe_intermediate_size: int = 0
    rms_norm_eps: float = 1e-6

    # MoE (0 = dense)
    num_local_experts: int = 0
    num_experts_per_tok: int = 0
    use_qk_norm: bool = True
    norm_topk_prob: bool = False
    decoder_sparse_step: int = 1
    mlp_only_layers: list[int] = field(default_factory=list)

    # Feature flags
    attention_bias: bool = False
    mlp_bias: bool = False

    # RoPE
    rope_theta: float = 10000.0
    max_position_embeddings: int = 32768

    # Runtime
    torch_dtype: torch.dtype = torch.bfloat16
    neuron_config: NeuronConfig | None = None

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        """Build from HF + Neuron configs."""
        if isinstance(hf_config, dict):
            config_dict = hf_config
        else:
            config_dict = hf_config.to_dict()

        # Extract known fields
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in field_names}

        # Parse torch_dtype
        if "torch_dtype" in filtered and isinstance(filtered["torch_dtype"], str):
            filtered["torch_dtype"] = getattr(torch, filtered["torch_dtype"])

        filtered["neuron_config"] = neuron_config
        filtered["num_local_experts"] = config_dict.get(
            "num_experts", config_dict.get("num_local_experts", 0)
        )
        filtered["num_experts_per_tok"] = config_dict.get(
            "num_experts_per_tok", 0
        )
        filtered["moe_intermediate_size"] = config_dict.get(
            "moe_intermediate_size", 0
        )
        filtered["use_qk_norm"] = config_dict.get("use_qk_norm", True)

        # Newer Transformers versions store Qwen3's RoPE base in
        # ``rope_parameters`` rather than the legacy top-level
        # ``rope_theta`` field.
        rope_theta = config_dict.get("rope_theta")
        if rope_theta is None:
            rope_parameters = config_dict.get("rope_parameters") or {}
            rope_theta = rope_parameters.get("rope_theta")
        if rope_theta is not None:
            filtered["rope_theta"] = float(rope_theta)

        config = cls(**filtered)
        if config.attention_bias:
            raise ValueError("Qwen3 attention_bias=True is not supported")
        if not config.use_qk_norm:
            raise ValueError("Qwen3 use_qk_norm=False is not supported")
        if config.torch_dtype != torch.bfloat16:
            raise ValueError(
                "Qwen3 currently supports only torch.bfloat16, "
                f"got {config.torch_dtype}"
            )
        return config
