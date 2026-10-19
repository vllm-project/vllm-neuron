# SPDX-License-Identifier: Apache-2.0
"""GLM-5.x (``GlmMoeDsaForCausalLM``, ``model_type: glm_moe_dsa``) in the Hy4 port's config.

GLM-5.3 is the Hy4 port's architecture minus three things and with a few values changed,
so it reuses ``HYV4Config`` (and the attention, MoE and DSA code that reads it):

* no iHC: a plain pre-norm residual (``GLMDecoderLayer``);
* no output gate, no attention sinks (``gated_mla`` / ``learnable_sink`` False);
* the indexer rotates its FIRST ``qk_rope_head_dim`` dims, interleaved
  (``indexer_rope_first``); the MLA rope is interleaved too, theta 8e6, no YaRN;
* 3 dense layers (12288), 21 full-indexer layers ([0, 1, 2, 6, 10, ..., 74]),
  routed_scaling_factor 2.5, no SwiGLU clamp.
"""

from __future__ import annotations

import json
import os

import torch
from transformers import PretrainedConfig

from ..hy_v4.config import HYV4Config, _dtype_of


def glm_config(hf: PretrainedConfig, neuron_config: object | None = None) -> HYV4Config:
    if isinstance(hf, (str, bytes, os.PathLike)):
        with open(hf) as f:
            hf = PretrainedConfig.from_dict(json.load(f))
    get = lambda name, default=None: getattr(hf, name, default)  # noqa: E731
    n_layers = hf.num_hidden_layers
    rope = get("rope_parameters") or {}
    if rope.get("rope_type", "default") != "default":
        raise NotImplementedError(f"GLM on Neuron implements the default RoPE only, got {rope}")
    if get("n_group", 1) != 1 or get("topk_group", 1) != 1:
        raise NotImplementedError("grouped expert routing is not implemented")
    if get("scoring_func", "sigmoid") != "sigmoid":
        raise NotImplementedError(f"scoring_func={get('scoring_func')!r}")
    if not get("rope_interleave", True) or not get("indexer_rope_interleave", True):
        raise NotImplementedError("GLM on Neuron implements interleaved RoPE only")
    first_dense = get("first_k_dense_replace", 3)
    mlp_layer_types = get("mlp_layer_types") or (
        ["dense"] * first_dense + ["sparse"] * (n_layers - first_dense)
    )
    indexer_types = get("indexer_types")
    if not indexer_types:
        freq, offset = get("index_topk_freq", 4), get("index_skip_topk_offset", 3)
        indexer_types = [
            "full" if max(i - offset + 1, 0) % freq == 0 else "shared" for i in range(n_layers)
        ]
    config = HYV4Config(
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
        rope_theta=float(rope.get("rope_theta", get("rope_theta", 8.0e6))),
        learnable_sink=False,
        gated_mla=False,
        index_topk=get("index_topk", 2048),
        index_n_heads=get("index_n_heads", 32),
        index_head_dim=get("index_head_dim", 128),
        indexer_types=tuple(indexer_types),
        mlp_layer_types=tuple(mlp_layer_types),
        n_routed_experts=hf.n_routed_experts,
        n_shared_experts=get("n_shared_experts", 1),
        num_experts_per_tok=hf.num_experts_per_tok,
        moe_intermediate_size=hf.moe_intermediate_size,
        routed_scaling_factor=float(get("routed_scaling_factor", 2.5)),
        norm_topk_prob=bool(get("norm_topk_prob", True)),
        swiglu_limit=0.0,
        hc_mult=1,
        hc_magnitude=0.0,
        hc_eps=0.0,
        tie_word_embeddings=bool(get("tie_word_embeddings", False)),
        lm_head_fp32=False,
        max_position_embeddings=get("max_position_embeddings", 1048576),
        torch_dtype=_dtype_of(hf, torch.bfloat16),
        neuron_quantization=get("neuron_quantization"),
        indexer_rope_first=True,
    )
    config.neuron_config = neuron_config
    return config
