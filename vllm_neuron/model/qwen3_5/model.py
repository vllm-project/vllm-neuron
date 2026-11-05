# SPDX-License-Identifier: Apache-2.0
"""Native Qwen3.5 hybrid (gated-attention + GatedDeltaNet) model for Neuron.

Qwen3.5 differs from Qwen3 in three ways this module handles:

* **Gated full attention** — ``q_proj`` emits ``2 * num_heads * head_dim``
  values: the query plus a per-token sigmoid gate that scales the attention
  output. Q/K/V are therefore separate (not fused), and Q is *not* a plain
  reuse of Qwen3's attention.
* **Qwen3.5 RMSNorm** — ``x * (1 + weight)`` with zero-init weights, not
  Qwen3's ``weight * x``.
* **Partial RoPE** — only ``partial_rotary_factor`` of each head (0.25 by
  default) is rotated; the rest passes through unchanged.

Linear-attention layers use the eager ``Qwen3_5GatedDeltaNet`` in ``.gdn``.

Phase-1 scope: correct, eager, single-core (TP=1), text-only, prefill only.
TP sharding, NKI kernels, the recurrent-state paged cache (decode), and the
VLM wrapper are follow-ups.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed.parallel_state import get_tp_group
from transformers import PretrainedConfig

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import set_weight_loader, sharding_weight_loader

import vllm_neuron.nn as neuron_nn
from .config import Qwen3_5Config
from .gdn import Qwen3_5GatedDeltaNet

logger = logging.getLogger(__name__)


def _attention_metadata(attn_metadata: dict, layer_types: list[str]) -> dict:
    """Return the per-step metadata of the first full-attention layer.

    Every full-attention layer sees the same ``max_query_len`` in a step, so any
    one determines prefill vs. decode. Qwen3.5 keeps full-attention layers (the
    default pattern places the first at index 3), so ``index()`` is safe.
    """
    idx = layer_types.index("full_attention")
    return attn_metadata[f"layers.{idx}.self_attn"]


def _is_prefill(attn_metadata: dict, layer_types: list[str]) -> tuple[bool, int]:
    md = _attention_metadata(attn_metadata, layer_types)
    max_query_len = md["max_query_len"]
    decode_token_threshold = md["decode_token_threshold"]
    return max_query_len > decode_token_threshold, max_query_len


# =============================================================================
# Norms
# =============================================================================


class Qwen3_5RMSNorm(nn.Module):
    """Qwen3.5 RMSNorm: ``x_norm * (1 + weight)``, zero-init weight."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (hidden_states * (1.0 + self.weight.float())).to(input_dtype)


# =============================================================================
# Partial rotary embedding
# =============================================================================


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_partial_rotary(q, k, cos, sin):
    """Rotate the first ``rotary_dim`` dims of q/k; pass the rest through."""
    cos = cos.unsqueeze(1)  # [T, 1, rotary_dim]
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q = torch.cat([q_rot * cos + _rotate_half(q_rot) * sin, q_pass], dim=-1)
    k = torch.cat([k_rot * cos + _rotate_half(k_rot) * sin, k_pass], dim=-1)
    return q, k


class Qwen3_5RotaryEmbedding(nn.Module):
    """Partial RoPE: cos/sin cover only ``rotary_dim = head_dim * factor``."""

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim
        self.rotary_dim = int(config.head_dim * config.partial_rotary_factor)

    def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
        return 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device=device)
                / self.rotary_dim
            )
        )

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self._compute_inv_freq(device)  # [rotary_dim/2]
        inv_freq_expanded = inv_freq[:, None].float()
        positions_expanded = position_ids[None, :].float()
        freqs = (inv_freq_expanded @ positions_expanded).transpose(0, 1)
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, rotary_dim]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


# =============================================================================
# Gated full attention
# =============================================================================


class Qwen3_5Attention(nn.Module):
    """Gated attention with per-head QK-norm and partial RoPE.

    Weights are stored in checkpoint layout ``[out, in]`` and loaded replicated
    (TP=1). Phase 1 computes a stateless causal attention over the prompt.
    """

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5
        q_dim = self.num_attention_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim

        self.q_proj_weight = nn.Parameter(
            torch.empty(q_dim * 2, self.hidden_size, dtype=self.dtype)
        )
        self.k_proj_weight = nn.Parameter(
            torch.empty(kv_dim, self.hidden_size, dtype=self.dtype)
        )
        self.v_proj_weight = nn.Parameter(
            torch.empty(kv_dim, self.hidden_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, q_dim, dtype=self.dtype)
        )

        self.q_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        # Replicated (TP=1); explicit num_shards=1 documents the Phase-1 intent.
        for name, out_dim in (
            ("q_proj_weight", self.num_attention_heads * self.head_dim * 2),
            ("k_proj_weight", self.num_key_value_heads * self.head_dim),
            ("v_proj_weight", self.num_key_value_heads * self.head_dim),
            ("o_proj_weight", self.hidden_size),
        ):
            set_weight_loader(
                getattr(self, name),
                sharding_weight_loader(shard_dim=0, shard_size=out_dim, num_shards=1),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """Stateless prefill (Phase 1). KV cache write + decode are follow-ups."""
        tokens, _ = hidden_states.shape
        hidden_states = hidden_states.to(self.dtype)

        qkv = F.linear(hidden_states, self.q_proj_weight)
        qkv = qkv.view(tokens, self.num_attention_heads, self.head_dim * 2)
        q, gate = torch.chunk(qkv, 2, dim=-1)
        q = q.reshape(tokens, self.num_attention_heads, self.head_dim)
        gate = gate.reshape(tokens, self.num_attention_heads * self.head_dim)

        k = F.linear(hidden_states, self.k_proj_weight).view(
            tokens, self.num_key_value_heads, self.head_dim
        )
        v = F.linear(hidden_states, self.v_proj_weight).view(
            tokens, self.num_key_value_heads, self.head_dim
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = position_embeddings
        q, k = _apply_partial_rotary(q, k, cos, sin)

        # [1, heads, tokens, head_dim] for causal SDPA.
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        # GQA: repeat K/V heads to match the Q head count.
        n_rep = self.num_attention_heads // self.num_key_value_heads
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=self.scaling
        )
        attn = attn.squeeze(0).transpose(0, 1)  # [tokens, heads, head_dim]
        attn = attn.reshape(tokens, self.num_attention_heads * self.head_dim)

        attn = attn * torch.sigmoid(gate)

        return F.linear(attn, self.o_proj_weight)


# =============================================================================
# Dense MLP
# =============================================================================


class Qwen3_5MLP(nn.Module):
    """SiLU-gated dense MLP (same shape as Qwen3, eager Phase-1 path)."""

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.dtype = config.torch_dtype

        self.gate_proj_weight = nn.Parameter(
            torch.empty(self.intermediate_size, self.hidden_size, dtype=self.dtype)
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(self.intermediate_size, self.hidden_size, dtype=self.dtype)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.intermediate_size, dtype=self.dtype)
        )

        for name in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
            set_weight_loader(
                getattr(self, name),
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=(
                        self.intermediate_size
                        if name != "down_proj_weight"
                        else self.hidden_size
                    ),
                    num_shards=1,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.linear(x, self.gate_proj_weight)
        up = F.linear(x, self.up_proj_weight)
        return F.linear(F.silu(gate) * up, self.down_proj_weight)


# =============================================================================
# Decoder layer
# =============================================================================


class Qwen3_5DecoderLayer(nn.Module):
    """One hybrid decoder layer: gated attention or GatedDeltaNet."""

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        if self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx=layer_idx)
        elif self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"unknown layer_type {self.layer_type!r}")
        self.mlp = Qwen3_5MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "full_attention":
            hidden_states = self.self_attn(
                hidden_states, positions, position_embeddings, attn_metadata
            )
        else:
            hidden_states = self.linear_attn(hidden_states, is_prefill=is_prefill)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Model backbone
# =============================================================================


class Qwen3_5Model(nn.Module):
    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        if self.world_size != 1:
            raise NotImplementedError(
                "Qwen3.5 is TP=1 only in this phase; TP sharding of the gated "
                "attention and GDN projections is not implemented."
            )

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )

        self.layers = nn.ModuleList(
            [
                Qwen3_5DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = Qwen3_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Qwen3_5RotaryEmbedding(config)

        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=False,
                pad_shard=True,
            ),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_prefill, _ = _is_prefill(attn_metadata, self.config.layer_types)

        hidden_states = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        if inputs_embeds is not None:
            # Multimodal merge is deferred to the VLM wrapper (Phase 1 is text-only).
            raise NotImplementedError(
                "inputs_embeds (multimodal) is not yet supported for Qwen3.5."
            )

        position_embeddings = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
                is_prefill=is_prefill,
            )

        return self.norm(hidden_states)


# =============================================================================
# LM head + weight loading
# =============================================================================


class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config
            if config.neuron_config
            else None
        )
        self._gather_logits = (
            config.neuron_config is not None and config.neuron_config.max_logprobs != 0
        )

        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=config.vocab_size // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=False,
            ),
        )

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=self.tp_group.device_group,
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        positions = positions.to(torch.int32)
        is_prefill, max_query_len = _is_prefill(attn_metadata, self.config.layer_types)

        T = input_ids.shape[0]
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt Length ({T}) must be > world_size ({self.world_size}) for SP."
            )

        hidden_states = self.model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
        )

        if (
            not is_prefill
            and max_query_len == 1
            and hidden_states.shape[0] == sampling_positions.shape[0]
        ):
            hidden_states_for_logits = hidden_states
        else:
            hidden_states_for_logits = torch.index_select(
                hidden_states, dim=0, index=sampling_positions
            )

        logits = self.lm_head(hidden_states_for_logits)

        if self.on_device_sampling_config is None:
            return logits

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )

        gathered_logits = None
        if self._gather_logits:
            gathered_logits = self.tp_group.all_gather(logits, dim=1)

        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = Qwen3_5Config.from_configs(hf_config, neuron_config)
        return cls(config)

    # ── KV cache (full-attention layers only) ─────────────────────────────

    def get_kv_spec(self):
        layers = []
        for i, layer in enumerate(self.model.layers):
            if layer.layer_type != "full_attention":
                continue
            layers.append(
                LayerSpec(
                    name=f"layers.{i}.self_attn",
                    num_kv_heads=layer.self_attn.num_key_value_heads,
                    head_size=layer.self_attn.head_dim,
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor, torch.Tensor]]):
        for i, layer in enumerate(self.model.layers):
            if layer.layer_type != "full_attention":
                continue
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise KeyError(f"KV cache for layer {layer_name} not initialized")
            # Phase 1 attention does not read/write these yet; store for later.
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # ── Weight loading ────────────────────────────────────────────────────

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        tp_rank = self.rank
        tp_size = self.world_size

        mappings = {}
        mappings["model.embed_tokens.weight"] = "model.embed_tokens.weight"
        mappings["lm_head.weight"] = "lm_head.weight"
        mappings["model.norm.weight"] = "model.norm.weight"

        for layer_id, layer in enumerate(self.model.layers):
            prefix = f"model.layers.{layer_id}"

            mappings[f"{prefix}.input_layernorm.weight"] = (
                f"{prefix}.input_layernorm.weight"
            )
            mappings[f"{prefix}.post_attention_layernorm.weight"] = (
                f"{prefix}.post_attention_layernorm.weight"
            )
            mappings[f"{prefix}.mlp.gate_proj_weight"] = (
                f"{prefix}.mlp.gate_proj.weight"
            )
            mappings[f"{prefix}.mlp.up_proj_weight"] = f"{prefix}.mlp.up_proj.weight"
            mappings[f"{prefix}.mlp.down_proj_weight"] = (
                f"{prefix}.mlp.down_proj.weight"
            )

            if layer.layer_type == "full_attention":
                mappings[f"{prefix}.self_attn.q_proj_weight"] = (
                    f"{prefix}.self_attn.q_proj.weight"
                )
                mappings[f"{prefix}.self_attn.k_proj_weight"] = (
                    f"{prefix}.self_attn.k_proj.weight"
                )
                mappings[f"{prefix}.self_attn.v_proj_weight"] = (
                    f"{prefix}.self_attn.v_proj.weight"
                )
                mappings[f"{prefix}.self_attn.o_proj_weight"] = (
                    f"{prefix}.self_attn.o_proj.weight"
                )
                mappings[f"{prefix}.self_attn.q_norm.weight"] = (
                    f"{prefix}.self_attn.q_norm.weight"
                )
                mappings[f"{prefix}.self_attn.k_norm.weight"] = (
                    f"{prefix}.self_attn.k_norm.weight"
                )
            else:
                mappings[f"{prefix}.linear_attn.in_proj_qkv_weight"] = (
                    f"{prefix}.linear_attn.in_proj_qkv.weight"
                )
                mappings[f"{prefix}.linear_attn.in_proj_z_weight"] = (
                    f"{prefix}.linear_attn.in_proj_z.weight"
                )
                mappings[f"{prefix}.linear_attn.in_proj_b_weight"] = (
                    f"{prefix}.linear_attn.in_proj_b.weight"
                )
                mappings[f"{prefix}.linear_attn.in_proj_a_weight"] = (
                    f"{prefix}.linear_attn.in_proj_a.weight"
                )
                mappings[f"{prefix}.linear_attn.conv1d_weight"] = (
                    f"{prefix}.linear_attn.conv1d.weight"
                )
                mappings[f"{prefix}.linear_attn.dt_bias"] = (
                    f"{prefix}.linear_attn.dt_bias"
                )
                mappings[f"{prefix}.linear_attn.A_log"] = f"{prefix}.linear_attn.A_log"
                mappings[f"{prefix}.linear_attn.norm.weight"] = (
                    f"{prefix}.linear_attn.norm.weight"
                )
                mappings[f"{prefix}.linear_attn.out_proj_weight"] = (
                    f"{prefix}.linear_attn.out_proj.weight"
                )

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, mappings, device
        ).state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            # dt_bias / A_log stay float32 for the log-space decay gate.
            if name.endswith(".dt_bias") or name.endswith(".A_log"):
                continue
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self.load_state_dict(rank_sharded, strict=False, assign=True)

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        checkpoint._ensure_indexed()
