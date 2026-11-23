# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 (text) Config
====================

<-- MODEL-SPECIFIC: All fields in this config are Gemma3-specific.

Reference checkpoint: EssentialAI/rnj-1-instruct (``Gemma3ForCausalLM``,
``model_type`` ``gemma3_text``). The checkpoint has full attention in all
layers, YaRN RoPE, a final logit soft cap and tied word embeddings.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm_neuron.model.neuron_config import NeuronConfig

logger = logging.getLogger(__name__)

FULL_ATTENTION = "full_attention"


def _to_dict(hf_config: Any) -> dict:
    """Return a plain dict from a config path, a config object, or a dict."""
    if isinstance(hf_config, (str, bytes)):
        with open(hf_config) as f:
            return json.load(f)
    if isinstance(hf_config, dict):
        return dict(hf_config)
    config_dict = hf_config.to_dict()
    for name in ("torch_dtype", "dtype"):
        value = getattr(hf_config, name, None)
        if value is not None:
            config_dict[name] = value
    # The layer types and the RoPE parameters are attributes that the HF config
    # computes. to_dict() can omit them, so read the attributes.
    for name in ("layer_types", "rope_parameters", "rope_scaling", "rope_theta"):
        value = getattr(hf_config, name, None)
        if value is not None:
            config_dict[name] = value
    return config_dict


def resolve_rope_parameters(config_dict: dict) -> dict:
    """Return the RoPE parameters of the full-attention layers.

    transformers 5 stores one RoPE dict for each layer type in
    ``rope_parameters``. An older config stores ``rope_theta`` and
    ``rope_scaling`` at the top level. Both layouts give one flat dict with
    ``rope_theta`` and ``rope_type``.
    """
    params = config_dict.get("rope_parameters")
    if isinstance(params, dict) and FULL_ATTENTION in params:
        params = params[FULL_ATTENTION]
    if isinstance(params, dict) and params:
        out = dict(params)
    else:
        out = dict(config_dict.get("rope_scaling") or {})
        if "type" in out and "rope_type" not in out:
            out["rope_type"] = out.pop("type")
    if "rope_theta" not in out:
        out["rope_theta"] = config_dict.get("rope_theta", 10000.0)
    out.setdefault("rope_type", "default")
    return out


def resolve_layer_types(config_dict: dict, num_hidden_layers: int) -> list[str]:
    """Return the attention type of each layer.

    The rnj-1 checkpoint stores ``layer_type`` (singular). transformers 5
    ignores that key and computes ``layer_types`` from
    ``sliding_window_pattern``. This function uses the same order:
    ``layer_types``, then the pattern, then full attention.
    """
    layer_types = config_dict.get("layer_types")
    if layer_types:
        return list(layer_types)
    pattern = config_dict.get("sliding_window_pattern")
    if pattern:
        return [
            "sliding_attention" if bool((i + 1) % pattern) else FULL_ATTENTION
            for i in range(num_hidden_layers)
        ]
    return [FULL_ATTENTION] * num_hidden_layers


def resolve_served_dtype(default: torch.dtype = torch.bfloat16) -> torch.dtype:
    """Return the dtype that vLLM serves the model in.

    The checkpoint stores float32 weights, but the server runs in the dtype
    of ``--dtype``. Read that dtype from the current vLLM config. Use
    ``default`` when no vLLM config is set.
    """
    try:
        from vllm.config import get_current_vllm_config

        model_config = get_current_vllm_config().model_config
        dtype = getattr(model_config, "dtype", None) if model_config else None
        if isinstance(dtype, torch.dtype):
            return dtype
    except Exception:  # noqa: BLE001 - no vLLM config outside the server
        pass
    return default


@dataclass
class Gemma3Config:
    # <-- MODEL-SPECIFIC: Architecture parameters (defaults are rnj-1-instruct)
    vocab_size: int = 128256
    hidden_size: int = 4096
    intermediate_size: int = 16384
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    hidden_activation: str = "gelu_pytorch_tanh"
    query_pre_attn_scalar: float = 128.0
    final_logit_softcapping: float | None = 30.0
    attn_logit_softcapping: float | None = None
    attention_bias: bool = False
    tie_word_embeddings: bool = True

    # <-- MODEL-SPECIFIC: RoPE (full-attention layers)
    rope_theta: float = 10000.0
    rope_type: str = "yarn"
    rope_factor: float = 1.0
    rope_original_max_position_embeddings: int | None = None
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_attention_factor: float | None = None
    rope_mscale: float | None = None
    rope_mscale_all_dim: float | None = None
    rope_truncate: bool = True

    layer_types: list[str] = field(default_factory=list)

    # The served dtype (the checkpoint dtype is float32).
    torch_dtype: torch.dtype = torch.bfloat16

    # Framework config
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if not self.layer_types:
            self.layer_types = [FULL_ATTENTION] * self.num_hidden_layers

    @property
    def embed_scale(self) -> float:
        """Return the embedding multiplier ``sqrt(hidden_size)``."""
        return float(self.hidden_size) ** 0.5

    @property
    def attn_scaling(self) -> float:
        """Return the attention softmax scale ``query_pre_attn_scalar ** -0.5``."""
        return float(self.query_pre_attn_scalar) ** -0.5

    @classmethod
    def from_configs(cls, hf_config: Any, neuron_config: NeuronConfig | None = None):
        config_dict = _to_dict(hf_config)

        num_layers = config_dict.get("num_hidden_layers", cls.num_hidden_layers)
        rope = resolve_rope_parameters(config_dict)

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        values = {k: v for k, v in config_dict.items() if k in field_names}
        values.pop("torch_dtype", None)
        values.pop("rope_theta", None)
        values.pop("layer_types", None)
        if "hidden_activation" not in values and config_dict.get("hidden_act"):
            values["hidden_activation"] = config_dict["hidden_act"]

        values["layer_types"] = resolve_layer_types(config_dict, num_layers)
        values["rope_theta"] = float(rope["rope_theta"])
        values["rope_type"] = rope.get("rope_type", "default")
        values["rope_factor"] = float(rope.get("factor", 1.0))
        values["rope_original_max_position_embeddings"] = rope.get(
            "original_max_position_embeddings"
        )
        values["rope_beta_fast"] = float(rope.get("beta_fast", 32.0))
        values["rope_beta_slow"] = float(rope.get("beta_slow", 1.0))
        values["rope_attention_factor"] = rope.get("attention_factor")
        values["rope_mscale"] = rope.get("mscale")
        values["rope_mscale_all_dim"] = rope.get("mscale_all_dim")
        values["rope_truncate"] = bool(rope.get("truncate", True))

        values["torch_dtype"] = resolve_served_dtype()
        if neuron_config is not None:
            values["neuron_config"] = neuron_config
        return cls(**values)
