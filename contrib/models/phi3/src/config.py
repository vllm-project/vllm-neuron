# SPDX-License-Identifier: Apache-2.0
"""
Phi3 Config
===========

<-- MODEL-SPECIFIC: All fields in this config are Phi3-specific.

Phi3 (for example Phi-4-mini-instruct) is a dense GQA decoder. It differs from
Llama in three ways:

- The checkpoint stores fused ``qkv_proj`` and fused ``gate_up_proj`` weights.
- RoPE applies to the first ``partial_rotary_factor * head_dim`` dimensions only.
- RoPE uses LongRoPE scaling with a short factor and a long factor.
"""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

SUPPORTED_ROPE_TYPES = ("default", "longrope")


@dataclass
class Phi3Config:
    # <-- MODEL-SPECIFIC: Architecture parameters (Phi-4-mini defaults)
    vocab_size: int = 200064
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 32
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: int | None = None
    max_position_embeddings: int = 131072
    original_max_position_embeddings: int | None = None
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 1.0
    rope_type: str = "default"
    rope_short_factor: list[float] | None = None
    rope_long_factor: list[float] | None = None
    rope_attention_factor: float | None = None
    sliding_window: int | None = None
    tie_word_embeddings: bool = True
    torch_dtype: torch.dtype = torch.bfloat16

    # Framework config
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.original_max_position_embeddings is None:
            self.original_max_position_embeddings = self.max_position_embeddings

    @property
    def rotary_dim(self) -> int:
        """Number of head dimensions that RoPE rotates."""
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

        config_dict = dict(config_dict)
        config_dict.update(extract_rope_settings(config_dict))

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])
        if filtered_dict.get("torch_dtype") is None:
            dtype = config_dict.get("dtype")
            filtered_dict["torch_dtype"] = (
                getattr(torch, dtype) if isinstance(dtype, str) else torch.bfloat16
            )

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)


def extract_rope_settings(config_dict: dict) -> dict:
    """Read the Phi3 RoPE settings from a HF config dictionary.

    transformers < 5 keeps ``rope_theta`` and ``partial_rotary_factor`` at the
    top level and the LongRoPE factors in ``rope_scaling``. transformers >= 5
    keeps all of them in ``rope_parameters``. This function reads both layouts.
    A value in ``rope_parameters`` wins over the top-level value.

    Args:
        config_dict: HF config as a dictionary.

    Returns:
        A dictionary with the ``Phi3Config`` RoPE fields that the input sets.

    Example::

        >>> d = extract_rope_settings({
        ...     "rope_parameters": {"rope_type": "longrope", "rope_theta": 1e4,
        ...                         "short_factor": [1.0], "long_factor": [2.0]}})
        >>> d["rope_type"], d["rope_short_factor"]
        ('longrope', [1.0])
    """
    out = {}
    for key in ("rope_theta", "partial_rotary_factor"):
        if config_dict.get(key) is not None:
            out[key] = config_dict[key]
    if config_dict.get("original_max_position_embeddings") is not None:
        out["original_max_position_embeddings"] = config_dict[
            "original_max_position_embeddings"
        ]

    for source_key in ("rope_scaling", "rope_parameters"):
        source = config_dict.get(source_key)
        if not isinstance(source, dict):
            continue
        rope_type = source.get("rope_type", source.get("type"))
        if rope_type is not None:
            out["rope_type"] = rope_type
        if source.get("rope_theta") is not None:
            out["rope_theta"] = source["rope_theta"]
        if source.get("partial_rotary_factor") is not None:
            out["partial_rotary_factor"] = source["partial_rotary_factor"]
        if source.get("original_max_position_embeddings") is not None:
            out["original_max_position_embeddings"] = source[
                "original_max_position_embeddings"
            ]
        if source.get("short_factor") is not None:
            out["rope_short_factor"] = list(source["short_factor"])
        if source.get("long_factor") is not None:
            out["rope_long_factor"] = list(source["long_factor"])
        if source.get("attention_factor") is not None:
            out["rope_attention_factor"] = source["attention_factor"]
    return out
