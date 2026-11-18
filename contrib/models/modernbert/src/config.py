# SPDX-License-Identifier: Apache-2.0
"""
ModernBERT Config
=================

<-- MODEL-SPECIFIC: All fields in this config are ModernBERT-specific.
ModernBERT is a bidirectional encoder. It has no LM head in the embedding
checkpoints and no decode phase. Key differences from Llama:
  - Bidirectional attention. A layer is "full_attention" or
    "sliding_attention". A sliding layer attends to keys at a distance of
    ``local_attention // 2`` or less on each side.
  - Each layer type has its own RoPE theta (global and local).
  - LayerNorm without bias (``norm_bias=False``), not RMSNorm.
  - Fused ``Wqkv`` and a GeGLU MLP (``Wi`` holds the input and gate halves).
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"


@dataclass
class ModernBertConfig:
    # <-- MODEL-SPECIFIC: ModernBERT architecture parameters (HF defaults)
    vocab_size: int = 50368
    hidden_size: int = 768
    intermediate_size: int = 1152
    num_hidden_layers: int = 22
    num_attention_heads: int = 12
    hidden_activation: str = "gelu"
    max_position_embeddings: int = 8192
    norm_eps: float = 1e-5
    norm_bias: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    local_attention: int = 128
    global_attn_every_n_layers: int = 3
    layer_types: list[str] | None = None
    global_rope_theta: float = 160_000.0
    local_rope_theta: float = 10_000.0
    pad_token_id: int | None = 50283
    torch_dtype: torch.dtype = torch.bfloat16

    # Framework config
    neuron_config: NeuronConfig | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                SLIDING_ATTENTION
                if i % self.global_attn_every_n_layers
                else FULL_ATTENTION
                for i in range(self.num_hidden_layers)
            ]
        self._validate()

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def sliding_window(self) -> int:
        # HF ModernBertConfig.sliding_window: the half window on each side.
        return self.local_attention // 2

    def _validate(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) is not a multiple of "
                f"num_attention_heads ({self.num_attention_heads}). "
                "Use a checkpoint with a whole number of channels per head."
            )
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries, but "
                f"num_hidden_layers is {self.num_hidden_layers}. "
                "Make the two values agree in config.json."
            )
        unknown = sorted(set(self.layer_types) - {FULL_ATTENTION, SLIDING_ATTENTION})
        if unknown:
            raise ValueError(
                f"layer_types contains an unsupported value: {unknown}. "
                f"Use only '{FULL_ATTENTION}' or '{SLIDING_ATTENTION}'."
            )
        if self.hidden_activation != "gelu":
            raise NotImplementedError(
                f"hidden_activation={self.hidden_activation!r} is not supported. "
                "The Neuron ModernBERT port implements the exact (erf) 'gelu' only."
            )
        if self.norm_bias or self.attention_bias or self.mlp_bias:
            raise NotImplementedError(
                "A ModernBERT checkpoint with norm, attention, or MLP bias is not "
                "supported. The port implements norm_bias, attention_bias, and "
                "mlp_bias equal to False only."
            )

    @staticmethod
    def _rope_thetas(config_dict: dict) -> dict:
        """Return the global and local RoPE theta from a HF config dict.

        transformers >= 5 keeps the values in ``rope_parameters``, keyed by
        layer type. Older configs keep ``global_rope_theta`` and
        ``local_rope_theta`` at the top level.
        """
        out = {}
        rope_parameters = config_dict.get("rope_parameters")
        if isinstance(rope_parameters, dict):
            for layer_type, key in (
                (FULL_ATTENTION, "global_rope_theta"),
                (SLIDING_ATTENTION, "local_rope_theta"),
            ):
                params = rope_parameters.get(layer_type) or {}
                rope_type = params.get("rope_type", params.get("type", "default"))
                if rope_type != "default":
                    raise NotImplementedError(
                        f"RoPE type {rope_type!r} for {layer_type} is not supported. "
                        "The Neuron ModernBERT port implements the 'default' RoPE only."
                    )
                if "rope_theta" in params:
                    out[key] = float(params["rope_theta"])
        for key in ("global_rope_theta", "local_rope_theta"):
            if key not in out and config_dict.get(key) is not None:
                out[key] = float(config_dict[key])
        return out

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig = None
    ):
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
            hf_dtype = getattr(hf_config, "torch_dtype", None)
            if hf_dtype is not None:
                config_dict["torch_dtype"] = hf_dtype
        else:
            config_dict = dict(hf_config)

        # transformers >= 5 writes "dtype" in place of "torch_dtype".
        if config_dict.get("torch_dtype") is None and config_dict.get("dtype"):
            config_dict["torch_dtype"] = config_dict["dtype"]

        config_dict.update(cls._rope_thetas(config_dict))

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {
            k: v
            for k, v in config_dict.items()
            if k in field_names and v is not None and k != "neuron_config"
        }

        if isinstance(filtered_dict.get("torch_dtype"), str):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)
