# SPDX-License-Identifier: Apache-2.0
"""
OLMo-3 Config
=============

<-- MODEL-SPECIFIC: All fields in this config are OLMo-3-specific.
Reference model: allenai/Olmo-3-7B-Instruct.
OLMo-3-7B: 32 layers, 32 Q heads / 32 KV heads (MHA), head_dim=128, hidden=4096.

Differences from Llama:
  - Post-norm decoder layer. The layer normalizes the output of the attention
    and of the MLP before the residual add. No input norm exists.
  - q_norm and k_norm are RMSNorm over the full projection width
    (num_heads * head_dim), not over one head.
  - Three sliding-window layers, then one full-attention layer
    (``layer_types``). The window is 4096 tokens.
  - YaRN RoPE applies to the full-attention layers only. The sliding-window
    layers use the default RoPE with the same ``rope_theta``.
"""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .compat import hoist_rope_parameters

SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"

# Compile OLMo-3 at neuronx-cc -O3. At O2, the compiled full model has a large
# error at layers 20, 21 and 24, but each of these layers is correct alone and
# in a traced graph. -O3 disables the modular flow, the probable cause. The
# model runner applies a model's per-graph policy (compiler_args_for) to the
# decode and encoder graphs only, not to prefill. Thus give this level to vLLM
# as an explicit --optimization-level (optimization_level= offline), which
# the runner applies to every graph.
OLMO3_OPTIMIZATION_LEVEL = 3


@dataclass
class Olmo3Config:
    # <-- MODEL-SPECIFIC: OLMo-3-7B architecture parameters
    vocab_size: int = 100278
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int | None = None
    max_position_embeddings: int = 65536
    rms_norm_eps: float = 1e-6
    rope_theta: float = 500000.0
    rope_scaling: dict | None = None
    sliding_window: int | None = 4096
    layer_types: list[str] | None = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    hidden_act: str = "silu"
    torch_dtype: torch.dtype = torch.bfloat16

    # Framework config
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        # HF default: every fourth layer is a full-attention layer.
        if self.layer_types is None:
            self.layer_types = [
                SLIDING_ATTENTION if (i + 1) % 4 != 0 else FULL_ATTENTION
                for i in range(self.num_hidden_layers)
            ]

    def is_sliding(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == SLIDING_ATTENTION

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

        config_dict = _hoist_olmo3_rope_parameters(config_dict)
        if config_dict.get("torch_dtype") is None and config_dict.get("dtype"):
            config_dict["torch_dtype"] = config_dict["dtype"]

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)


def _hoist_olmo3_rope_parameters(config_dict: dict) -> dict:
    """Return ``config_dict`` with the RoPE settings in the pre-v5 layout.

    transformers 5 keeps the OLMo-3 RoPE settings per layer type, for example
    ``rope_parameters = {"full_attention": {...}, "sliding_attention": {...}}``.
    The full-attention entry holds the YaRN settings. This function moves
    them to ``rope_theta`` and ``rope_scaling``. A flat ``rope_parameters``
    dict goes to :func:`hoist_rope_parameters`.
    """
    rope_parameters = config_dict.get("rope_parameters")
    if not isinstance(rope_parameters, dict):
        return config_dict
    if FULL_ATTENTION not in rope_parameters:
        return hoist_rope_parameters(config_dict, supported_rope_types=("yarn",))

    out = dict(config_dict)
    full = dict(rope_parameters.get(FULL_ATTENTION) or {})
    sliding = dict(rope_parameters.get(SLIDING_ATTENTION) or {})
    if "rope_theta" in full:
        out.setdefault("rope_theta", full["rope_theta"])
    elif "rope_theta" in sliding:
        out.setdefault("rope_theta", sliding["rope_theta"])
    if sliding.get("rope_type", "default") != "default":
        raise ValueError(
            f"OLMo-3 sliding-window layers use rope_type="
            f"{sliding.get('rope_type')!r}, but only 'default' is supported. "
            f"Use a checkpoint with the default RoPE on the sliding-window layers."
        )
    scaling = {k: v for k, v in full.items() if k != "rope_theta"}
    if (
        scaling.get("rope_type", "default") != "default"
        and out.get("rope_scaling") is None
    ):
        out["rope_scaling"] = scaling
    return out
