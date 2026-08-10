# SPDX-License-Identifier: Apache-2.0
"""Weight loading logic for Qwen3 BF16 models."""

import logging

import torch

from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

logger = logging.getLogger(__name__)


def load_weights_bf16(
    model,
    checkpoint_path: str,
    device: torch.device,
    cache_dir: str | None,
) -> None:
    """Load BF16 weights from HF checkpoint into a Qwen3ForCausalLM model."""
    tp_rank = model.rank
    tp_size = model.world_size

    logger.info(f"Qwen3 load_weights: tp_rank={tp_rank}, tp_size={tp_size}")

    mappings = {}

    # Embeddings
    mappings["model.embed_tokens.weight"] = "model.embed_tokens.weight"
    mappings["lm_head.weight"] = "lm_head.weight"
    mappings["model.norm.weight"] = "model.norm.weight"

    for layer_id in range(model.config.num_hidden_layers):
        prefix = f"model.layers.{layer_id}"

        # Attention (weight loaders handle the fusion)
        mappings[f"{prefix}.self_attn.qkv_proj_weight"] = [
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
        ]
        mappings[f"{prefix}.self_attn.o_proj_weight"] = (
            f"{prefix}.self_attn.o_proj.weight"
        )
        mappings[f"{prefix}.self_attn.q_layernorm.weight"] = (
            f"{prefix}.self_attn.q_norm.weight"
        )
        mappings[f"{prefix}.self_attn.k_layernorm.weight"] = (
            f"{prefix}.self_attn.k_norm.weight"
        )
        mappings[f"{prefix}.input_layernorm.weight"] = (
            f"{prefix}.input_layernorm.weight"
        )

        # MLP
        layer = model.model.layers[layer_id]
        if layer.mlp.is_moe:
            mappings[f"{prefix}.mlp.experts.post_attention_layernorm.weight"] = (
                f"{prefix}.post_attention_layernorm.weight"
            )
            mappings[f"{prefix}.mlp.experts.router_weight"] = (
                f"{prefix}.mlp.gate.weight"
            )
            expert_ids = (
                layer.mlp.experts.local_expert_indices
                if hasattr(layer.mlp.experts, "local_expert_indices")
                else range(model.config.num_local_experts)
            )
            expert_ids = list(expert_ids)
            mappings[f"{prefix}.mlp.experts.gate_up_proj_weight"] = [
                f"{prefix}.mlp.experts.{e}.gate_proj.weight" for e in expert_ids
            ] + [
                f"{prefix}.mlp.experts.{e}.up_proj.weight" for e in expert_ids
            ]
            mappings[f"{prefix}.mlp.experts.down_proj_weight"] = [
                f"{prefix}.mlp.experts.{e}.down_proj.weight" for e in expert_ids
            ]
        else:
            mappings[f"{prefix}.post_attention_layernorm.weight"] = (
                f"{prefix}.post_attention_layernorm.weight"
            )
            mappings[f"{prefix}.mlp.dense_mlp.gate_proj_weight"] = (
                f"{prefix}.mlp.gate_proj.weight"
            )
            mappings[f"{prefix}.mlp.dense_mlp.up_proj_weight"] = (
                f"{prefix}.mlp.up_proj.weight"
            )
            mappings[f"{prefix}.mlp.dense_mlp.down_proj_weight"] = (
                f"{prefix}.mlp.down_proj.weight"
            )

    # Load checkpoint
    checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
    rank_sharded_checkpoint = checkpoint.load_sharded_pipelined(
        tp_rank, tp_size, model, mappings, device
    ).state_dict

    _load_kv_cache_scales(model, checkpoint, device)

    # Apply weights
    model.load_state_dict(rank_sharded_checkpoint, strict=False, assign=True)

    logger.info(f"Successfully loaded Qwen3 weights from {checkpoint_path}")


def _load_kv_cache_scales(
    model, checkpoint: SafetensorsCheckpoint, device: torch.device
) -> None:
    """Load KV cache quantization scales from checkpoint if provided."""
    from vllm_neuron.utils.dtype_utils import QUANTIZED_KV_CACHE_DTYPES

    try:
        from vllm.config import get_current_vllm_config
        vllm_config = get_current_vllm_config()
        cache_dtype = vllm_config.cache_config.cache_dtype
    except Exception:
        return

    if cache_dtype not in QUANTIZED_KV_CACHE_DTYPES:
        return

    for layer_id in range(model.config.num_hidden_layers):
        attn = model.model.layers[layer_id].self_attn

        for scale_name in ("k_scale", "v_scale"):
            key = f"model.layers.{layer_id}.self_attn.{scale_name}"
            if key in checkpoint._tensor_name_to_file:
                # Invert: checkpoint stores scales for (tensor / scale),
                # but kernel quantizes via (tensor * scale).
                val = 1.0 / checkpoint._get_slice(key)[:].to(
                    dtype=torch.bfloat16, device=device
                )
            else:
                val = torch.ones(1, dtype=torch.bfloat16, device=device)
            setattr(attn, scale_name, val.reshape(1, 1))

        attn.k_scale_float = attn.k_scale.item()
        attn.v_scale_float = attn.v_scale.item()
