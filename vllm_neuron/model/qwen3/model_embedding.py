# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-Embedding model (pooling / embedding runner).

Qwen3-Embedding-8B reuses the exact Qwen3 decoder backbone — the only difference from the
generative ``Qwen3ForCausalLM`` is at the output stage:

    Generative:  backbone -> [T, H] -> index_select + lm_head -> [B, vocab]
    Embedding:   backbone -> [T, H]                            (returned as-is)

The flattened ``[T, H]`` post-norm hidden states are then consumed by the
NeuronModelRunner ``_pool`` path, which runs the upstream pooler
(LAST-token gather + L2 normalize) via the reused ``DispatchPooler``.
"""

import logging

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

from .config import Qwen3Config
from .model import Qwen3Model

logger = logging.getLogger(__name__)


class Qwen3ForEmbedding(nn.Module):
    """Qwen3 backbone with a pooler head (no LM head) for embedding tasks.

    >>> PARALLELISM: reuses Qwen3Model's SP/TP backbone unchanged.
    <-- MODEL-SPECIFIC: drops lm_head; attaches upstream DispatchPooler.
    """

    is_pooling_model = True

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        from vllm.config import get_current_vllm_config
        from vllm.model_executor.layers.pooler import DispatchPooler

        vllm_config = get_current_vllm_config()

        # Embedding is prefill-only: it never loads the FP8 k_scale/v_scale params
        # the quantized attention path requires, so an fp8 KV cache would crash
        # deep in attention. Reject it up front with a clear message.
        cache_dtype = getattr(vllm_config.cache_config, "cache_dtype", None)
        if (cache_dtype or "").startswith("fp8"):
            raise ValueError(
                f"kv_cache_dtype={cache_dtype!r} is not supported for the Qwen3 "
                "pooling/embedding model. It is prefill-only and does not "
                "implement FP8 KV cache; use the default (auto/bf16)."
            )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None, (
            "pooler_config is None — Qwen3ForEmbedding requires the pooling "
            "runner (launch with --runner pooling, or a checkpoint whose "
            "sentence-transformers modules.json resolves to a pooling model)."
        )
        # LAST-token pooling + L2 normalize.
        self.pooler = DispatchPooler.for_embedding(pooler_config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return the flattened ``[T, H]`` post-norm backbone hidden states."""
        hidden_states, _ = self.model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
        )
        return hidden_states

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = Qwen3Config.from_configs(hf_config, neuron_config)
        return cls(config)

    # ── KV Cache ──────────────────────────────────────────────────────────
    # Embedding is prefill-only, so the KV cache is never read back. These two hooks exist only because the
    # NeuronModelRunner calls get_kv_spec / bind_kv_cache UNCONDITIONALLY on
    # self.model when building and binding the cache. The factory rejects that combination
    # in _validate_config up front.

    def get_kv_spec(self):
        layers = []
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            layers.append(
                LayerSpec(
                    name=layer_name,
                    num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                    head_size=layer.self_attn.head_dim,
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor, torch.Tensor]]):
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise KeyError(f"KV cache for layer {layer_name} not initialized")
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # ── Weight Loading ────────────────────────────────────────────────────

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load Qwen3 backbone weights from the HF checkpoint, skipping lm_head.

        Mirrors ``Qwen3ForCausalLM.load_weights`` but omits the ``lm_head.weight``
        mapping — an embedding model has no LM head. Qwen3-Embedding-8B ships no
        Dense/Projection module (modules.json: Transformer -> LAST Pooling -> L2
        Normalize), so weight loading is pure backbone and the pooler is
        weightless.
        """
        tp_rank = self.rank
        tp_size = self.world_size

        mappings = {}

        # Map model parameter names to checkpoint keys. Qwen3-Embedding-8B's checkpoint stores
        # weights WITHOUT the "model." prefix , so the checkpoint side drops it.
        # Embedding + final norm (NO lm_head — embedding model has no LM head).
        mappings["model.embed_tokens.weight"] = "embed_tokens.weight"
        mappings["model.norm.weight"] = "norm.weight"

        for layer_id in range(len(self.model.layers)):
            prefix = f"model.layers.{layer_id}"
            ckpt = f"layers.{layer_id}"

            # <-- MODEL-SPECIFIC: Separate Q, K, V → fused QKV
            mappings[f"{prefix}.self_attn.qkv_proj_weight"] = [
                f"{ckpt}.self_attn.q_proj.weight",
                f"{ckpt}.self_attn.k_proj.weight",
                f"{ckpt}.self_attn.v_proj.weight",
            ]
            mappings[f"{prefix}.self_attn.o_proj_weight"] = (
                f"{ckpt}.self_attn.o_proj.weight"
            )

            # <-- MODEL-SPECIFIC: QK-norm weights (no TP sharding)
            mappings[f"{prefix}.self_attn.q_norm.weight"] = (
                f"{ckpt}.self_attn.q_norm.weight"
            )
            mappings[f"{prefix}.self_attn.k_norm.weight"] = (
                f"{ckpt}.self_attn.k_norm.weight"
            )

            # <-- MODEL-SPECIFIC: Layer norm names
            mappings[f"{prefix}.input_layernorm.weight"] = (
                f"{ckpt}.input_layernorm.weight"
            )
            mappings[f"{prefix}.post_attention_layernorm.weight"] = (
                f"{ckpt}.post_attention_layernorm.weight"
            )

            # <-- MODEL-SPECIFIC: Dense MLP weight names
            mappings[f"{prefix}.mlp.gate_proj_weight"] = f"{ckpt}.mlp.gate_proj.weight"
            mappings[f"{prefix}.mlp.up_proj_weight"] = f"{ckpt}.mlp.up_proj.weight"
            mappings[f"{prefix}.mlp.down_proj_weight"] = f"{ckpt}.mlp.down_proj.weight"

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        load_result = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, mappings, device
        )
        rank_sharded = load_result.state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self.load_state_dict(rank_sharded, strict=False, assign=True)
