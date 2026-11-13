# SPDX-License-Identifier: Apache-2.0
"""
Gemma4 Config
======================

Model-specific config adapter for Gemma4 31B.

Architecture highlights:
  - Heterogeneous layers: SWA (head_dim=256, 16 KV heads) and Global
    (head_dim=512, 4 KV heads)
  - attention_k_eq_v on global layers (V copies K, no v_proj in checkpoint)
  - QK normalization (RMSNorm), V normalization (RMSNorm without learnable scale)
  - layer_scalar per layer (learned multiplicative factor)
  - final_logit_softcapping = 30.0
  - Partial RoPE for global layers (factor=0.25)
  - GeGLU activation (gelu_pytorch_tanh)
  - Scaled embeddings (multiply by sqrt(hidden_size))
  - 4 norms per layer
  - tie_word_embeddings=True
  - vocab_size=262144

HF config nests text parameters under text_config. This config class
extracts them to top-level for convenience.
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


# Global-layer geometry that some transformers releases (e.g. 5.17) drop when
# parsing the Gemma 4 config. Without them the global layers would silently get
# the sliding-layer KV head count.
_GLOBAL_LAYER_FIELDS = ("global_head_dim", "num_global_key_value_heads")


def _raw_fields_dropped_by_transformers(hf_config: PretrainedConfig, parsed: dict) -> dict:
    nested = parsed.get("text_config") if isinstance(parsed.get("text_config"), dict) else {}
    missing = [k for k in _GLOBAL_LAYER_FIELDS if k not in parsed and k not in nested]
    name_or_path = getattr(hf_config, "_name_or_path", None)
    if not missing or not name_or_path:
        return {}
    from transformers.utils import cached_file

    path = cached_file(name_or_path, "config.json", _raise_exceptions_for_missing_entries=False)
    if path is None:
        return {}
    with open(path) as f:
        raw = json.load(f)
    raw = {**raw, **raw.get("text_config", {})}
    return {k: raw[k] for k in missing if k in raw}


@dataclass
class Gemma4Config:
    vocab_size: int = 262144
    hidden_size: int = 5376
    intermediate_size: int = 21504
    num_hidden_layers: int = 60
    num_attention_heads: int = 32
    num_key_value_heads: int = 16  # SWA layers
    head_dim: int = 256  # SWA layers
    global_head_dim: int = 512  # Global layers
    num_global_key_value_heads: int | None = None  # Global layers (None = use num_key_value_heads)
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    sliding_window: int | None = 1024
    final_logit_softcapping: float = 30.0
    attention_k_eq_v: bool = False
    tie_word_embeddings: bool = True
    torch_dtype: torch.dtype = torch.bfloat16

    # Per-layer type list: "sliding_attention" or "full_attention"
    layer_types: list[str] = field(default_factory=list)

    # Per-layer input embeddings (PLE). The 31B checkpoint sets these to 0 or None;
    # the PLE path in model.py is not used by it and is not tested.
    hidden_size_per_layer_input: int | None = 0
    vocab_size_per_layer_input: int | None = None
    num_kv_shared_layers: int | None = 0

    # RoPE parameters per layer type
    rope_parameters: dict = field(default_factory=lambda: {
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
        },
        "sliding_attention": {
            "rope_theta": 10000.0,
        },
    })

    # Framework config
    neuron_config: NeuronConfig | None = None

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
            config_dict.update(_raw_fields_dropped_by_transformers(hf_config, config_dict))
        else:
            config_dict = hf_config

        # Gemma4 nests text params under text_config
        text_config = config_dict.get("text_config", {})
        if isinstance(text_config, dict):
            for k, v in text_config.items():
                if k not in config_dict:
                    config_dict[k] = v

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        # Debug override: limit num_hidden_layers via env var.
        import os
        override_layers = os.environ.get("GEMMA4_NUM_LAYERS")
        if override_layers is not None:
            filtered_dict["num_hidden_layers"] = int(override_layers)
            # Truncate layer_types to match.
            if "layer_types" in filtered_dict:
                filtered_dict["layer_types"] = filtered_dict["layer_types"][:int(override_layers)]
            # Disable PLE (its embedding dim depends on num_layers).
            filtered_dict["hidden_size_per_layer_input"] = 0

        return cls(**filtered_dict)

    def get_layer_head_dim(self, layer_idx: int) -> int:
        if self.layer_types[layer_idx] == "full_attention":
            return self.global_head_dim
        return self.head_dim

    def get_layer_num_kv_heads(self, layer_idx: int) -> int:
        if self.layer_types[layer_idx] == "full_attention":
            v = self.num_global_key_value_heads
            return v if v is not None else self.num_key_value_heads
        return self.num_key_value_heads

    def get_layer_rope_theta(self, layer_idx: int) -> float:
        layer_type = self.layer_types[layer_idx]
        params = self.rope_parameters.get(layer_type, {})
        if layer_type == "full_attention":
            return params.get("rope_theta", 1000000.0)
        return params.get("rope_theta", 10000.0)

    def get_layer_partial_rotary_factor(self, layer_idx: int) -> float:
        layer_type = self.layer_types[layer_idx]
        if layer_type == "full_attention":
            params = self.rope_parameters.get("full_attention", {})
            return params.get("partial_rotary_factor", 0.25)
        return 1.0

    def is_global_layer(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "full_attention"
