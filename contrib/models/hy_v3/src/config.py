# SPDX-License-Identifier: Apache-2.0
"""
Hy3 (HYV3) Configuration
========================
<-- MODEL-SPECIFIC: All fields in this config are model-specific.

Reference checkpoint: tencent/Hy3-FP8 (``HYV3ForCausalLM``, ``model_type=hy_v3``).
"""

from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class HYV3Config:
    """Configuration for the Hy3 MoE model.

    Architecture overview:
    - Decoder-only transformer, pre-norm RMSNorm
    - GQA attention (64 Q / 8 KV heads, head_dim 128) with per-head QK RMSNorm
      applied BEFORE standard (rotate_half) RoPE
    - First ``first_k_dense_replace`` layers use a dense SiLU-gated MLP
    - Remaining layers: MoE with sigmoid router, a selection-only expert bias
      (DeepSeek-V3 ``noaux_tc`` style), renormalized top-k weights scaled by
      ``router_scaling_factor``, plus ``num_shared_experts`` always-on experts
    - Optional MTP (next-n prediction) layers stored after the decoder stack
      (not used by this implementation)
    """

    # ── Model architecture (MODEL-SPECIFIC) ──────────────────────────────
    vocab_size: int = 120832
    hidden_size: int = 4096
    intermediate_size: int = 13312  # Dense MLP intermediate size
    moe_intermediate_size: int = 1536  # Per-expert intermediate size
    num_hidden_layers: int = 80
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # ── RoPE (MODEL-SPECIFIC: default rope, no scaling) ──────────────────
    rope_parameters: dict = field(
        default_factory=lambda: {"rope_type": "default", "rope_theta": 11158840.0}
    )

    # ── MoE configuration (MODEL-SPECIFIC) ───────────────────────────────
    num_experts: int = 192
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    router_scaling_factor: float = 2.826
    route_norm: bool = True
    first_k_dense_replace: int = 1
    mlp_layer_types: list[str] | None = None

    # ── Quantization (checkpoint format) ─────────────────────────────────
    quantization_config: dict | None = None

    # ── Special tokens ───────────────────────────────────────────────────
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    # ── Framework config (not model-specific) ────────────────────────────
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.mlp_layer_types is None:
            self.mlp_layer_types = [
                "dense" if i < self.first_k_dense_replace else "sparse"
                for i in range(self.num_hidden_layers)
            ]

    @property
    def rope_theta(self) -> float:
        return self.rope_parameters["rope_theta"]

    def is_moe_layer(self, layer_idx: int) -> bool:
        return self.mlp_layer_types[layer_idx] == "sparse"

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> "HYV3Config":
        config_dict = hf_config.to_dict()
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in field_names}

        dtype = config_dict.get("torch_dtype") or config_dict.get("dtype")
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if dtype is not None:
            filtered["torch_dtype"] = dtype

        if filtered.get("rope_parameters") is None:
            filtered.pop("rope_parameters", None)
        # MTP layers live after the decoder stack and are not part of
        # mlp_layer_types; keep only the decoder entries.
        if filtered.get("mlp_layer_types") is not None:
            filtered["mlp_layer_types"] = list(filtered["mlp_layer_types"])[
                : filtered.get("num_hidden_layers", cls.num_hidden_layers)
            ]

        filtered["neuron_config"] = neuron_config
        return cls(**filtered)
