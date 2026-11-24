# SPDX-License-Identifier: Apache-2.0
"""Config for Tencent Hy4 (``model_type: hy_v4``, ``HYV4ForCausalLM``).

Hy4-preview is 770B total / 49B active: 78 decoder layers, each

    iHC pre -> RMSNorm -> gated MLA (+ DSA indexer) -> iHC post
    iHC pre -> RMSNorm -> MoE (layer 0: dense SwiGLU)  -> iHC post

with the residual carried as ``hc_mult`` = 4 parallel streams rather than one
(identity Hyper-Connections), collapsed by ``hc_head`` before the final norm.

Read from the HF config rather than hardcoded, but a few of the HF fields are
traps and are normalised here:

* ``head_dim`` is rewritten to ``qk_rope_head_dim`` (64) by HF's
  ``HYV4Config.__post_init__`` -- it is the *rotary* width, and the rotary
  frequencies are sized from it. The attention width is ``qk_head_dim`` =
  ``qk_nope_head_dim + qk_rope_head_dim`` = 256 and the value width is
  ``v_head_dim`` = 256.
* ``num_key_value_heads`` is forced to ``num_attention_heads`` by the same
  ``__post_init__``; the ``8`` in ``config.json`` is not used by anything. MLA
  has one latent "KV head" regardless.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

# Which RoPE pairing the MLA rope dims use. The vLLM and SGLang Hy4 ports both
# use interleaved pairs (``is_neox_style=False``, DeepSeek's convention); the
# transformers port uses ``rotate_half``. Interleaved is what the checkpoint
# wants: on device, ``neox`` degrades within a few tokens (counting derails
# after 10) where interleaved stays coherent. ``HY4_ROPE_STYLE=neox`` remains
# for comparing against transformers (``check_layer_vs_hf.py``).
ROPE_STYLE = os.environ.get("HY4_ROPE_STYLE", "interleaved")
if ROPE_STYLE not in ("interleaved", "neox"):
    raise ValueError(
        f"HY4_ROPE_STYLE must be 'interleaved' or 'neox', got {ROPE_STYLE!r}"
    )


def _dtype_of(cfg: PretrainedConfig, default: torch.dtype) -> torch.dtype:
    raw = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None)
    if isinstance(raw, torch.dtype):
        return raw
    if isinstance(raw, str):
        return getattr(torch, raw.replace("torch.", ""))
    return default


@dataclass
class HYV4Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    rms_norm_eps: float

    # Gated MLA
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rope_theta: float
    learnable_sink: bool
    gated_mla: bool

    # DSA indexer (not used yet: see ``HYV4Attention``)
    index_topk: int
    index_n_heads: int
    index_head_dim: int
    indexer_types: tuple[str, ...]

    # MoE
    mlp_layer_types: tuple[str, ...]
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    swiglu_limit: float

    # iHC
    hc_mult: int
    hc_magnitude: float
    hc_eps: float

    tie_word_embeddings: bool
    lm_head_fp32: bool
    max_position_embeddings: int
    torch_dtype: torch.dtype
    neuron_quantization: dict | None = None
    # The indexer's rope dims: the LAST qk_rope_head_dim of its 128 (Hy4) or the FIRST
    # (GLM-5.x, interleaved).
    indexer_rope_first: bool = False

    neuron_config: object | None = field(default=None, repr=False)

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def kv_latent_dim(self) -> int:
        """What MLA caches per token: the normed latent plus the shared rope key."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    def is_moe_layer(self, layer_idx: int) -> bool:
        return self.mlp_layer_types[layer_idx] == "sparse"

    @classmethod
    def from_hf(cls, hf: PretrainedConfig) -> HYV4Config:
        if isinstance(hf, (str, bytes, os.PathLike)):
            with open(hf) as f:
                hf = PretrainedConfig.from_dict(json.load(f))
        get = lambda name, default=None: getattr(hf, name, default)

        n_layers = hf.num_hidden_layers
        mlp_layer_types = get("mlp_layer_types") or (
            ["dense"] * min(1, n_layers) + ["sparse"] * max(n_layers - 1, 0)
        )
        indexer_types = get("indexer_types") or [
            "full" if i == 0 or (i - 1) % 4 == 0 else "shared" for i in range(n_layers)
        ]
        rope = get("rope_parameters") or {}
        if rope.get("rope_type", "default") != "default":
            raise NotImplementedError(
                f"Hy4 on Neuron implements the default RoPE only, got {rope}"
            )
        if not get("enable_ihc", True):
            raise NotImplementedError("Hy4 with enable_ihc=False is not implemented")
        if get("gating_type", "elementwise") != "elementwise":
            raise NotImplementedError(
                f"gating_type={get('gating_type')!r}: only the elementwise "
                f"(one gate per value channel) output gate is implemented"
            )
        if get("n_group", 1) != 1 or get("topk_group", 1) != 1:
            raise NotImplementedError("grouped expert routing is not implemented")

        return cls(
            vocab_size=hf.vocab_size,
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            num_hidden_layers=n_layers,
            rms_norm_eps=hf.rms_norm_eps,
            num_attention_heads=hf.num_attention_heads,
            q_lora_rank=hf.q_lora_rank,
            kv_lora_rank=hf.kv_lora_rank,
            qk_nope_head_dim=hf.qk_nope_head_dim,
            qk_rope_head_dim=hf.qk_rope_head_dim,
            v_head_dim=hf.v_head_dim,
            rope_theta=float(rope.get("rope_theta", get("rope_theta", 10000.0))),
            learnable_sink=bool(get("learnable_sink", True)),
            gated_mla=bool(get("gated_mla", True)),
            index_topk=get("index_topk", 2048),
            index_n_heads=get("index_n_heads", 32),
            index_head_dim=get("index_head_dim", 128),
            indexer_types=tuple(indexer_types),
            mlp_layer_types=tuple(mlp_layer_types),
            n_routed_experts=hf.n_routed_experts,
            n_shared_experts=get("n_shared_experts", 1),
            num_experts_per_tok=hf.num_experts_per_tok,
            moe_intermediate_size=hf.moe_intermediate_size,
            routed_scaling_factor=float(get("routed_scaling_factor", 1.0)),
            norm_topk_prob=bool(get("norm_topk_prob", True)),
            swiglu_limit=float(get("swiglu_limit", 0.0) or 0.0),
            hc_mult=get("hc_mult", 4),
            hc_magnitude=float(get("hc_magnitude", 2.0)),
            hc_eps=float(get("hc_eps", 1e-6)),
            tie_word_embeddings=bool(get("tie_word_embeddings", False)),
            lm_head_fp32=bool(get("enable_lm_head_fp32", False)),
            max_position_embeddings=get("max_position_embeddings", 1048576),
            torch_dtype=_dtype_of(hf, torch.bfloat16),
            neuron_quantization=get("neuron_quantization"),
        )

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: object | None = None
    ) -> HYV4Config:
        config = cls.from_hf(hf_config)
        config.neuron_config = neuron_config
        return config
