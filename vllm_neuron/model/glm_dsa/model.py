# SPDX-License-Identifier: Apache-2.0
"""GLM-5.x (``GlmMoeDsaForCausalLM``) on one trn2.48xlarge, on the Hy4 port's machinery.

Everything below the decoder layer is the Hy4 port's: ``HYV4Attention`` (absorbed MLA,
the DSA indexer and its selection, chunked prefill, the paged cache), ``HYV4MoE``
(DeepSeek routing, per-row fp8 experts) and ``HYV4MLP``. What is
GLM's own is the plain pre-norm residual, the stack around it, and the checkpoint names.
Weights come from ``examples/vllm_neuron/models/glm_dsa/convert_fp8_block_to_neuron_fp8.py``
(the published FP8 checkpoint's 128 x 128 blocks requantised per row for the experts,
dequantised to bf16 elsewhere). The MTP layer is not loaded.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.distributed.parallel_state import get_tp_group

from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.utils.weight_loader import set_weight_loader, sharding_weight_loader

from ..hy_v4.config import HYV4Config
from ..hy_v4.model import (
    PERF_SKIP,
    HYV4Attention,
    HYV4ForCausalLM,
    HYV4RMSNorm,
    HYV4RotaryEmbedding,
    _all_reduce,
    _capture,
)
from ..hy_v4.moe import HYV4MLP, HYV4MoE
from .config import glm_config


class GLMDecoderLayer(nn.Module):
    """``x + attn(norm(x))``, then ``x + mlp(norm(x))``."""

    def __init__(self, config: HYV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        dtype = config.torch_dtype
        self.input_layernorm = HYV4RMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
        self.post_attention_layernorm = HYV4RMSNorm(
            config.hidden_size, config.rms_norm_eps, dtype
        )
        self.self_attn = HYV4Attention(config, layer_idx)
        self.mlp = (
            HYV4MoE(config)
            if config.is_moe_layer(layer_idx)
            else HYV4MLP(config.hidden_size, config.intermediate_size, self.world_size, dtype)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
        selected: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (hidden, DSA key selection for the layers that share it)."""
        if "attn" not in PERF_SKIP:
            x, selected = self.self_attn(
                self.input_layernorm(hidden),
                positions,
                position_embeddings,
                attn_metadata,
                selected,
                rank,
            )
            hidden = hidden + x.to(hidden.dtype)
        x = self.post_attention_layernorm(hidden)
        _capture(self.layer_idx, "mlp_in", x)
        x = torch.zeros_like(x) if "moe" in PERF_SKIP else self.mlp(x)
        if self.world_size > 1:
            x = _all_reduce(self.tp_group, x)
        _capture(self.layer_idx, "mlp_out", x)
        return hidden + x.to(hidden.dtype), selected


class GLMTextModel(nn.Module):
    def __init__(self, config: HYV4Config):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.embed_tokens.tp_size,
            ),
        )
        self.layers = nn.ModuleList(
            GLMDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = HYV4RMSNorm(config.hidden_size, config.rms_norm_eps, config.torch_dtype)
        self.rotary_emb = HYV4RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: dict,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        position_embeddings = self.rotary_emb(positions)
        selected = None
        for layer in self.layers:
            hidden, selected = layer(
                hidden, positions, position_embeddings, attn_metadata, selected, rank
            )
        return self.norm(hidden)


def checkpoint_mappings(config: HYV4Config) -> dict[str, str]:
    """Port parameter name -> converted-checkpoint key, for every loaded parameter."""
    m: dict[str, str] = {}
    for i in range(config.num_hidden_layers):
        hf = ours = f"model.layers.{i}"
        m[f"{ours}.input_layernorm.weight"] = f"{hf}.input_layernorm.weight"
        m[f"{ours}.post_attention_layernorm.weight"] = f"{hf}.post_attention_layernorm.weight"
        a = ha = f"{ours}.self_attn"
        m[f"{a}.q_a_proj_weight"] = f"{ha}.q_a_proj.weight"
        m[f"{a}.q_a_layernorm.weight"] = f"{ha}.q_a_layernorm.weight"
        m[f"{a}.q_b_proj_weight"] = f"{ha}.q_b_proj.weight"
        m[f"{a}.kv_a_proj_weight"] = f"{ha}.kv_a_proj_with_mqa.weight"
        m[f"{a}.kv_a_layernorm.weight"] = f"{ha}.kv_a_layernorm.weight"
        m[f"{a}.w_uk"] = f"{ha}.kv_b_proj.weight"
        m[f"{a}.w_uv"] = f"{ha}.kv_b_proj.weight"
        m[f"{a}.o_proj_weight"] = f"{ha}.o_proj.weight"
        if config.indexer_types[i] == "full":
            ia = hi = f"{a}.indexer"
            m[f"{ia}.wq_b_weight"] = f"{hi}.wq_b.weight"
            m[f"{ia}.wk_weight"] = f"{hi}.wk.weight"
            m[f"{ia}.k_norm_weight"] = f"{hi}.k_norm.weight"
            m[f"{ia}.k_norm_bias"] = f"{hi}.k_norm.bias"
            m[f"{ia}.weights_proj_weight"] = f"{hi}.weights_proj.weight"
        mlp = hm = f"{ours}.mlp"
        if config.is_moe_layer(i):
            m[f"{mlp}.router_weight"] = f"{hm}.gate.weight"
            m[f"{mlp}.e_score_correction_bias"] = f"{hm}.gate.e_score_correction_bias"
            m[f"{mlp}.experts_gate_up_weight"] = f"{hm}.experts.gate_up_proj"
            m[f"{mlp}.experts_gate_up_scale"] = f"{hm}.experts.gate_up_proj_scale"
            m[f"{mlp}.experts_down_weight"] = f"{hm}.experts.down_proj_t"
            m[f"{mlp}.experts_down_scale"] = f"{hm}.experts.down_proj_scale"
            shared = hs = f"{mlp}.shared_experts"
        else:
            shared = hs = mlp
        for p in ("gate_proj", "up_proj", "down_proj"):
            m[f"{shared}.{p}_weight"] = f"{hs}.{p}.weight"
    m["model.embed_tokens.weight"] = "model.embed_tokens.weight"
    m["model.norm.weight"] = "model.norm.weight"
    m["lm_head.weight"] = (
        "model.embed_tokens.weight" if config.tie_word_embeddings else "lm_head.weight"
    )
    return m


class GLMDsaForCausalLM(HYV4ForCausalLM):
    """GLM-5.x text model (``GlmMoeDsaForCausalLM``), on the converted Trn2 fp8 checkpoint."""

    text_model_cls = GLMTextModel
    checkpoint_mappings = staticmethod(checkpoint_mappings)

    @classmethod
    def from_configs(cls, hf_config, text_neuron_config=None, vision_neuron_config=None):
        if vision_neuron_config is not None:
            raise NotImplementedError("GLM-5.x is served text-only here")
        config = glm_config(hf_config, neuron_config=text_neuron_config)
        quant = config.neuron_quantization or {}
        if quant.get("format") != "fp8_row_experts":
            raise ValueError(
                "GLMDsaForCausalLM on Neuron loads the Trn2 fp8 checkpoint written by "
                "examples/vllm_neuron/models/glm_dsa/convert_fp8_block_to_neuron_fp8.py "
                "(config.json has no neuron_quantization.format=fp8_row_experts)."
            )
        return cls(config)


__all__ = ["GLMDsaForCausalLM", "GLMTextModel"]
