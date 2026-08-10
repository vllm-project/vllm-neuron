# SPDX-License-Identifier: Apache-2.0
"""
GPT-OSS Configuration
================================
<-- MODEL-SPECIFIC: All fields in this config are model-specific.
When porting to a new model, replace these with the new model's hyperparameters.
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class GptOssConfig:
    """
    Configuration for GPT-OSS model.

    <-- MODEL-SPECIFIC: These parameters define the GPT-OSS architecture.
    When porting a new model, update these fields to match the new architecture.

    Architecture overview:
    - Transformer decoder with MoE (Mixture of Experts) feed-forward layers
    - GQA (Grouped Query Attention) with separate Q and KV head counts
    - YaRN/NTK RoPE for position encoding
    - Learnable attention sinks per head
    - Sliding window attention on alternating layers (even-indexed)
    - SwiGLU activation with clamping in expert feed-forward
    - Pre-attention and pre-MLP RMSNorm
    """

    # ── Model architecture (MODEL-SPECIFIC) ──────────────────────────────
    vocab_size: int = 32000
    hidden_size: int = 4096  # Padded hidden size (aligned for hardware)
    unpadded_hidden_size: int = 4096  # Original hidden size before padding
    num_hidden_layers: int = 32
    num_attention_heads: int = 32  # Q heads
    num_key_value_heads: int | None = None  # KV heads (None = same as Q)
    head_dim: int | None = None  # Per-head dimension
    intermediate_size: int = 11008  # Padded intermediate size
    unpadded_intermediate_size: int = 11008  # Original intermediate size
    rms_norm_eps: float = 1e-6
    torch_dtype: torch.dtype = torch.bfloat16

    # ── MoE configuration (MODEL-SPECIFIC) ───────────────────────────────
    num_local_experts: int = 8  # Total number of experts per layer
    num_experts_per_tok: int = 4  # Top-k experts per token

    # ── Attention features (MODEL-SPECIFIC) ──────────────────────────────
    sliding_window: int | None = None  # Sliding window size (applied to even layers)
    attention_bias: bool = True  # Whether attention layers have bias terms
    mlp_bias: bool = True  # Whether MLP layers have bias terms

    # ── RoPE settings (MODEL-SPECIFIC: YaRN scaling) ─────────────────────
    rope_parameters: dict = field(
        default_factory=lambda: {"rope_type": "yarn", "rope_theta": 10000.0}
    )

    # ── Activation settings (MODEL-SPECIFIC: SwiGLU clamping) ────────────
    swiglu_limit: float = 7.0
    swiglu_alpha: float = 1.702

    # ── Special tokens ───────────────────────────────────────────────────
    pad_token_id: int | None = None
    bos_token_id: int = 1
    eos_token_id: int = 2

    # ── Sequence settings ────────────────────────────────────────────────
    max_position_embeddings: int = 8192

    # ── Framework config (not model-specific) ────────────────────────────
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        """Create config from HuggingFace config + NeuronConfig.

        <-- MODEL-SPECIFIC: The padding logic and field mapping are model-specific.
        Different models may have different padding requirements or field names.
        """
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            if (
                hasattr(hf_config, "quantization_config")
                and hf_config.quantization_config is None
            ):
                delattr(hf_config, "quantization_config")
                config_dict = hf_config.to_dict()
                hf_config.quantization_config = None
            else:
                config_dict = hf_config.to_dict()
        else:
            config_dict = hf_config

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        filtered_dict["neuron_config"] = neuron_config

        # Read attention_bias and mlp_bias from HF config (default True for backward compatibility)
        filtered_dict["attention_bias"] = config_dict.get("attention_bias", True)
        filtered_dict["mlp_bias"] = config_dict.get("mlp_bias", True)

        # <-- MODEL-SPECIFIC: GPT-OSS pads hidden and intermediate to 3072
        filtered_dict["unpadded_intermediate_size"] = filtered_dict["intermediate_size"]
        filtered_dict["intermediate_size"] = 3072
        filtered_dict["unpadded_hidden_size"] = filtered_dict["hidden_size"]
        filtered_dict["hidden_size"] = 3072

        return cls(**filtered_dict)
