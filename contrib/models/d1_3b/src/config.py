# SPDX-License-Identifier: Apache-2.0
"""Config for the d1-3B language model (LFM2 hybrid backbone) on Neuron."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class D1TextConfig:
    """LFM2 text backbone of LiquidAI/d1-3B (``text_config`` of ``lfm2_vl``).

    30 layers: ``conv`` (gated short conv, kernel 3) or ``full_attention`` (GQA with
    per-head q/k RMSNorm before RoPE). Every layer has a dense SwiGLU MLP.
    """

    vocab_size: int = 128000
    hidden_size: int = 2048
    intermediate_size: int = 10752
    num_hidden_layers: int = 30
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int | None = None
    max_position_embeddings: int = 32768
    norm_eps: float = 1e-5
    rope_theta: float = 1e6
    conv_L_cache: int = 3
    conv_bias: bool = False
    layer_types: list = field(default_factory=list)
    tie_word_embeddings: bool = True
    torch_dtype: torch.dtype = torch.bfloat16
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries, expected "
                f"{self.num_hidden_layers}"
            )
        unknown = set(self.layer_types) - {"conv", "full_attention"}
        if unknown:
            raise ValueError(f"Unsupported layer types: {sorted(unknown)}")
        if self.conv_bias:
            raise NotImplementedError("conv_bias=True is not implemented")
        if self.conv_L_cache != 3:
            raise NotImplementedError(
                f"conv_L_cache={self.conv_L_cache}; only kernel size 3 is implemented"
            )
        if not self.tie_word_embeddings:
            raise NotImplementedError("Untied lm_head is not implemented")

    @property
    def attention_layer_ids(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    @classmethod
    def from_configs(
        cls, hf_config, neuron_config: NeuronConfig | None = None
    ) -> D1TextConfig:
        """Parse an ``lfm2_vl`` (or bare ``lfm2``) HF config.

        Args:
            hf_config: ``PretrainedConfig``, a dict, or a path to ``config.json``.
            neuron_config: The plugin's Neuron config, stored on the result.
        """
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                d = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            d = hf_config.to_dict()
        else:
            d = dict(hf_config)
        # lfm2_vl nests the language model config.
        d = dict(d.get("text_config") or d)
        rope = d.get("rope_parameters") or {}
        if "rope_theta" in rope:
            d["rope_theta"] = rope["rope_theta"]
        if rope.get("rope_type", "default") != "default":
            raise NotImplementedError(
                f"rope_type={rope['rope_type']} is not implemented"
            )
        names = {f.name for f in cls.__dataclass_fields__.values()}
        kw = {k: v for k, v in d.items() if k in names and k != "torch_dtype"}
        kw["torch_dtype"] = torch.bfloat16
        kw["neuron_config"] = neuron_config
        return cls(**kw)
