# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM3 Config
======================

<-- MODEL-SPECIFIC: All fields in this config are MiniCPM3-specific.

MiniCPM3 uses multi-head latent attention (MLA) with low-rank Q and KV
projections, LongRoPE on the rotary part of each head, and muP scaling of the
embeddings, the residual branches and the logits.

The HF config sets ``head_dim`` to the rotary head dimension (32). This port
does not use ``head_dim``. It uses ``qk_nope_head_dim``, ``qk_rope_head_dim``
and ``v_head_dim``.
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .compat import hoist_rope_parameters

_SUPPORTED_ROPE_TYPES = ("default", "longrope")


@dataclass
class MiniCPM3Config:
    # <-- MODEL-SPECIFIC: Architecture parameters (defaults are MiniCPM3-4B)
    vocab_size: int = 73448
    hidden_size: int = 2560
    intermediate_size: int = 6400
    num_hidden_layers: int = 62
    num_attention_heads: int = 40
    num_key_value_heads: int = 40
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: dict | None = None
    tie_word_embeddings: bool = True
    torch_dtype: torch.dtype = torch.bfloat16

    # <-- MODEL-SPECIFIC: MLA geometry
    q_lora_rank: int = 768
    kv_lora_rank: int = 256
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int | None = None

    # <-- MODEL-SPECIFIC: muP scaling
    scale_emb: float = 12.0
    scale_depth: float | None = 1.4
    dim_model_base: int | None = 256

    # <-- MODEL-SPECIFIC: LongRoPE factors (one factor per rotary frequency)
    rope_short_factor: list[float] | None = field(default=None)
    rope_long_factor: list[float] | None = field(default=None)
    rope_original_max_position_embeddings: int | None = None

    # Framework config
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.v_head_dim is None:
            self.v_head_dim = self.hidden_size // self.num_attention_heads
        if self.scale_depth is None:
            self.scale_depth = float(self.num_hidden_layers) ** 0.5
        if self.dim_model_base is None:
            self.dim_model_base = self.hidden_size
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        scaling = self.rope_scaling or {}
        rope_type = scaling.get("rope_type", scaling.get("type", "default"))
        if rope_type not in _SUPPORTED_ROPE_TYPES:
            raise ValueError(
                f"RoPE type {rope_type!r} is not supported for MiniCPM3. "
                f"Use one of {_SUPPORTED_ROPE_TYPES}."
            )
        if rope_type == "longrope":
            if self.rope_short_factor is None:
                self.rope_short_factor = list(scaling["short_factor"])
            if self.rope_long_factor is None:
                self.rope_long_factor = list(scaling["long_factor"])
            if self.rope_original_max_position_embeddings is None:
                self.rope_original_max_position_embeddings = scaling.get(
                    "original_max_position_embeddings", self.max_position_embeddings
                )
            expected = self.qk_rope_head_dim // 2
            for name in ("rope_short_factor", "rope_long_factor"):
                if len(getattr(self, name)) != expected:
                    raise ValueError(
                        f"{name} has {len(getattr(self, name))} values. "
                        f"Supply qk_rope_head_dim // 2 = {expected} values."
                    )

    @property
    def qk_head_dim(self) -> int:
        """Return the Q/K head dimension that the softmax scale uses."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def residual_scale(self) -> float:
        """Return the scale of each residual branch (muP)."""
        return self.scale_depth / (float(self.num_hidden_layers) ** 0.5)

    @property
    def logits_scaling(self) -> float:
        """Return the divisor of the final hidden states before the LM head."""
        return self.hidden_size / self.dim_model_base

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig = None
    ):
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
            dtype = getattr(hf_config, "torch_dtype", None) or getattr(
                hf_config, "dtype", None
            )
            if dtype is not None:
                config_dict["torch_dtype"] = dtype
        else:
            config_dict = dict(hf_config)

        if config_dict.get("torch_dtype") is None and config_dict.get("dtype"):
            config_dict["torch_dtype"] = config_dict["dtype"]

        config_dict = hoist_rope_parameters(config_dict, _SUPPORTED_ROPE_TYPES)
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)
