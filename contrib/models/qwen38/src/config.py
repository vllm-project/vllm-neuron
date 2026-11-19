# SPDX-License-Identifier: Apache-2.0
"""The shape of a Qwen3.8 dense hybrid checkpoint, and what each TP rank holds.

Qwen3.8-27B is the dense member of the Qwen3.5 family: the same per-head geometry and config field names (the
architecture string is ``Qwen3_5ForConditionalGeneration``), with a dense SwiGLU MLP in place of the MoE block.
64 layers as ``[3 linear_attention + 1 full_attention] x 16``: 48 gated delta net (GDN) layers (16 key heads and 48
value heads of 128, conv kernel 4) and 16 gated attention layers (24 query heads and 4 KV heads of 256, partial RoPE
over a quarter of the head, theta 1e7).

The layers run the Qwen3.5 per-layer NKI kernels in ``kernels/``, which are built for one rank per physical
NeuronCore (LNC1) and for exactly this geometry. ``Qwen38Config.problems`` names every field that leaves it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Tuple

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

# the kernels' fixed geometry (kernels/gdn_decode.HEAD, kernels/attn_decode.ROT)
GDN_HEAD_DIM = 128
GDN_CONV_KERNEL = 4
ATTENTION_HEAD_DIM = 256
ROTARY_DIMS = 64
FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"


def _text_dict(hf_config) -> dict:
    if isinstance(hf_config, (str, bytes)):
        with open(hf_config) as f:
            d = json.load(f)
    elif isinstance(hf_config, PretrainedConfig):
        d = hf_config.to_dict()
    else:
        d = dict(hf_config)
    text = d.get("text_config") or {}
    if not isinstance(text, dict):
        text = text.to_dict()
    return {**d, **text}


@dataclass
class Qwen38Config:
    hidden_size: int
    num_hidden_layers: int
    layer_types: Tuple[str, ...]
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    tie_word_embeddings: bool
    tp_size: int = 1
    torch_dtype: torch.dtype = torch.bfloat16
    neuron_config: NeuronConfig | None = None

    @classmethod
    def from_configs(cls, hf_config, neuron_config: NeuronConfig | None = None) -> "Qwen38Config":
        t = _text_dict(hf_config)
        n = int(t["num_hidden_layers"])
        interval = int(t.get("full_attention_interval") or 4)
        types = t.get("layer_types") or [
            FULL_ATTENTION if i % interval == interval - 1 else LINEAR_ATTENTION for i in range(n)
        ]
        # debugging only: a prefix of the layers (a multiple of the 4-layer pattern keeps both kinds)
        override = os.environ.get("QWEN38_NUM_LAYERS")
        if override:
            n = int(override)
            types = types[:n]
        rope = t.get("rope_parameters") or t.get("rope_scaling") or {}
        head_dim = int(t["head_dim"])
        partial = float(rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 0.25)))
        tp = getattr(neuron_config, "tp_size", None) or 1
        return cls(
            hidden_size=int(t["hidden_size"]), num_hidden_layers=n, layer_types=tuple(types),
            linear_num_key_heads=int(t["linear_num_key_heads"]),
            linear_num_value_heads=int(t["linear_num_value_heads"]),
            linear_key_head_dim=int(t["linear_key_head_dim"]),
            linear_value_head_dim=int(t["linear_value_head_dim"]),
            linear_conv_kernel_dim=int(t["linear_conv_kernel_dim"]),
            num_attention_heads=int(t["num_attention_heads"]),
            num_key_value_heads=int(t["num_key_value_heads"]),
            head_dim=head_dim, rotary_dim=int(head_dim * partial),
            rope_theta=float(rope.get("rope_theta", t.get("rope_theta", 1e7))),
            intermediate_size=int(t["intermediate_size"]), vocab_size=int(t["vocab_size"]),
            rms_norm_eps=float(t.get("rms_norm_eps", 1e-6)),
            tie_word_embeddings=bool(t.get("tie_word_embeddings", False)),
            tp_size=int(tp), neuron_config=neuron_config,
        )

    # layer kinds
    @property
    def gdn_layers(self) -> Tuple[int, ...]:
        return tuple(i for i, k in enumerate(self.layer_types) if k == LINEAR_ATTENTION)

    @property
    def attention_layers(self) -> Tuple[int, ...]:
        return tuple(i for i, k in enumerate(self.layer_types) if k == FULL_ATTENTION)

    # per rank
    @property
    def gdn_v_heads(self) -> int:
        return self.linear_num_value_heads // self.tp_size

    @property
    def gdn_k_heads(self) -> int:
        return self.linear_num_key_heads // self.tp_size

    @property
    def conv_channels(self) -> int:
        """GDN conv channels of one rank, in the kernels' order q | k | v."""
        return 2 * self.gdn_k_heads * self.linear_key_head_dim + self.gdn_v_heads * self.linear_value_head_dim

    @property
    def q_heads(self) -> int:
        return self.num_attention_heads // self.tp_size

    @property
    def kernel_q_heads(self) -> int:
        """Query heads a rank runs: its own, plus one all-zero head when that count is odd. The attention prefill
        kernel requires an even count; Qwen3.8 at TP8 has 3 a rank. The extra head's output projection rows are
        zero, so it adds nothing to the layer output."""
        return self.q_heads + self.q_heads % 2

    @property
    def kv_heads(self) -> int:
        """KV heads a rank holds; with more ranks than KV heads each head is replicated."""
        return max(1, self.num_key_value_heads // self.tp_size)

    @property
    def intermediate_per_rank(self) -> int:
        return self.intermediate_size // self.tp_size

    def problems(self) -> list:
        """Every reason the kernels cannot serve this config at this TP degree."""
        tp, out = self.tp_size, []
        if self.linear_key_head_dim != GDN_HEAD_DIM or self.linear_value_head_dim != GDN_HEAD_DIM:
            out.append(f"GDN head dims {self.linear_key_head_dim}/{self.linear_value_head_dim}: the kernels are "
                       f"built for {GDN_HEAD_DIM}")
        if self.linear_conv_kernel_dim != GDN_CONV_KERNEL:
            out.append(f"GDN conv kernel {self.linear_conv_kernel_dim}: the kernels keep a 3-token window")
        if self.head_dim != ATTENTION_HEAD_DIM:
            out.append(f"attention head dim {self.head_dim}: the kernels are built for {ATTENTION_HEAD_DIM}")
        if self.rotary_dim != ROTARY_DIMS:
            out.append(f"{self.rotary_dim} rotary dims: the attention kernels rotate {ROTARY_DIMS}")
        for name, value in (("GDN key heads", self.linear_num_key_heads),
                            ("GDN value heads", self.linear_num_value_heads),
                            ("attention query heads", self.num_attention_heads),
                            ("intermediate size", self.intermediate_size)):
            if value % tp:
                out.append(f"{name} {value} do not shard over TP {tp}")
        if self.num_key_value_heads % tp and tp % self.num_key_value_heads:
            out.append(f"{self.num_key_value_heads} KV heads neither shard over nor replicate across TP {tp}")
        if self.hidden_size % GDN_HEAD_DIM:
            out.append(f"hidden size {self.hidden_size} is not a multiple of {GDN_HEAD_DIM}")
        if self.kv_heads != 1:
            out.append(f"{self.kv_heads} KV heads per rank: the attention kernels take one")
        unknown = sorted(set(self.layer_types) - {FULL_ATTENTION, LINEAR_ATTENTION})
        if unknown:
            out.append(f"layer kinds {unknown}")
        if len(self.gdn_layers) % 2:
            out.append(f"{len(self.gdn_layers)} GDN layers: the state cache splits them over two planes")
        return out
