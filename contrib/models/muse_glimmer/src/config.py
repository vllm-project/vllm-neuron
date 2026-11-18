# SPDX-License-Identifier: Apache-2.0
"""Muse-Glimmer text-decoder config for vLLM Neuron.

Parsed from the HF ``MuseGlimmerConfig`` (its ``text_config``). Phase 1 serves text only;
the vision fields are ignored here.

Muse-Glimmer-30B text decoder: 52 layers, hidden 6656, 32 Q / 2 KV heads, head_dim 128,
SwiGLU 19968, vocab 202048. Layer pattern [local, local, local, global] x 13:
  local  = sliding window 2048, RoPE theta 500000
  global = full attention, NoPE (layer_rope_theta == 0)
"""

from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class MuseGlimmerTextConfig:
    vocab_size: int = 202048
    hidden_size: int = 6656
    intermediate_size: int = 19968
    num_hidden_layers: int = 52
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5  # embedding norm, pre-norms, final norm, Q/K norm
    post_norm_eps: float = 1e-8  # post-attention / post-feedforward norms
    qk_scale_factor: float = 3.87  # Q is scaled after its norm
    output_multiplier: float = 0.19611613513818404
    final_logit_softcapping: float = 20.0
    sliding_window: int = 2048
    layer_types: list = field(default_factory=list)
    layer_rope_theta: list = field(default_factory=list)
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16
    neuron_config: NeuronConfig | None = None

    def is_sliding(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "sliding_attention"

    def has_rope(self, layer_idx: int) -> bool:
        return bool(self.layer_rope_theta[layer_idx])

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None = None
    ):
        top = (
            hf_config.to_dict()
            if isinstance(hf_config, PretrainedConfig)
            else dict(hf_config)
        )
        text = top.get(
            "text_config", top
        )  # top-level MuseGlimmerConfig nests the decoder
        names = {f.name for f in cls.__dataclass_fields__.values()}
        kw = {k: v for k, v in text.items() if k in names}
        rope = text.get("rope_parameters") or {}
        if "rope_theta" in rope:
            kw["rope_theta"] = rope["rope_theta"]
        thetas = {t for t in kw.get("layer_rope_theta", []) if t}
        if len(thetas) > 1:
            raise ValueError(
                f"expected one RoPE theta across RoPE layers, got {sorted(thetas)}"
            )
        dtype = (
            top.get("dtype")
            or top.get("torch_dtype")
            or getattr(hf_config, "torch_dtype", None)
        )
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if dtype is not None:
            kw["torch_dtype"] = dtype
        kw["neuron_config"] = neuron_config
        cfg = cls(**kw)
        n = cfg.num_hidden_layers
        if len(cfg.layer_types) != n or len(cfg.layer_rope_theta) != n:
            raise ValueError(
                "layer_types / layer_rope_theta must have one entry per layer"
            )
        return cfg
