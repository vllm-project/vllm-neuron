# SPDX-License-Identifier: Apache-2.0
"""
Granite BF16 Implementation
===========================

Implementation of Granite (GraniteForCausalLM, for example ibm/PowerLM-3b)
for the Neuron backend. The structure comes from the SmolLM3 and Qwen3 dense
models.

Granite is a Llama-style model with these differences:

- A bias on the Q, K, V, O, gate, up and down projections (``attention_bias``
  and ``mlp_bias``).
- ``embedding_multiplier`` multiplies the token embeddings.
- ``attention_multiplier`` is the softmax scale.
- ``residual_multiplier`` multiplies each sub-layer output before the
  residual add.
- ``logits_scaling`` divides the logits before the sampler.

Supported parallelism: TP and SP.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    Granite-specific. Change when porting.

PARALLELISM SHARDING:
  Attention heads are sharded. The MLP intermediate dimension is sharded.
  Row-parallel biases (o_proj, down_proj) are divided by the TP size, because
  each rank adds its bias before the TP reduction.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.functional.attention.attention_decode import (
    _can_use_attention_block_kernel,
)
from vllm_neuron.model.kv_cache import (
    KVSpec,
    LayerSpec,
    build_per_head_block_table,
    paged_slot_indices,
    scatter_new_kv,
)
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    fused_qkv_weight_loader,
    scaled_bias_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import GraniteConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Section 0: Weight loaders
# <-- MODEL-SPECIFIC: Granite has separate 1D Q, K and V biases.
# =============================================================================


def fused_qkv_bias_loader(
    q_size: int, kv_size: int, num_shards: int, num_kv_replicas: int = 1
) -> SafetensorsWeightLoader:
    """Fuse the 1D Q, K and V biases into the per-rank layout of qkv_proj_weight.

    The result is ``[q_size + 2 * kv_size]``: the Q heads of this rank, then
    the K heads, then the V heads. This is the column order of the fused weight.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 3, "fused_qkv_bias_loader expects [Q, K, V] slices"
        local_rank = rank % num_shards
        kv_rank = local_rank // num_kv_replicas
        q = slices[0][local_rank * q_size : (local_rank + 1) * q_size]
        k = slices[1][kv_rank * kv_size : (kv_rank + 1) * kv_size]
        v = slices[2][kv_rank * kv_size : (kv_rank + 1) * kv_size]
        return torch.cat([q, k, v], dim=0)

    return SafetensorsWeightLoader(transform=transform)


# =============================================================================
# Section 1: RMS Normalization
# <-- MODEL-SPECIFIC: Granite uses standard RMSNorm
# =============================================================================


class GraniteRMSNorm(nn.Module):
    """Standard RMSNorm (no mean subtraction, no bias)."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            hidden_states, self.weight.shape, self.weight, self.variance_epsilon
        )


# =============================================================================
# Section 2: Rotary Position Embedding (standard RoPE, no scaling)
# =============================================================================


class GraniteRotaryEmbedding(nn.Module):
    """Standard RoPE (rotate_half layout) without frequency scaling."""

    def __init__(self, config: GraniteConfig):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim

    def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
        return 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
                / self.head_dim
            )
        )

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin for the positions. Both have shape [T, head_dim]."""
        inv_freq = self._compute_inv_freq(device)
        inv_freq_expanded = inv_freq[:, None].float()
        positions_expanded = position_ids[None, :].float()
        freqs = (inv_freq_expanded @ positions_expanded).transpose(0, 1)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


# =============================================================================
# Section 3: Attention
# =============================================================================


class GraniteAttention(nn.Module):
    """Multi-head attention with TP head sharding and QKV/O biases.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads are sharded across the TP ranks.
    - Prefill: all-gather -> fused QKV+RoPE -> flash attention -> O proj ->
      reduce-scatter.
    - Decode: fused decode megakernel -> all-reduce.

    <-- MODEL-SPECIFIC:
    - softmax scale = attention_multiplier.
    - Bias on Q, K, V and O.
    """

    def __init__(self, config: GraniteConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        # <-- MODEL-SPECIFIC: Granite softmax scale
        self.scaling = config.attention_multiplier
        self.attention_bias = config.attention_bias

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        if self.num_attention_heads % self.world_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) is not divisible "
                f"by the TP size ({self.world_size}). Use a TP size that divides it."
            )

        # >>> PARALLELISM: Head sharding <<<
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
        if self.world_size >= self.num_key_value_heads:
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            self.num_key_value_heads_per_rank = (
                self.num_key_value_heads // self.world_size
            )
            self.num_kv_replicas = 1
        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
        )

        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim
        qkv_size = q_size + 2 * kv_size
        o_proj_in_features = self.num_attention_heads_per_rank * self.head_dim

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_proj_in_features, self.hidden_size, dtype=self.dtype)
        )
        if self.attention_bias:
            self.qkv_proj_bias = nn.Parameter(torch.zeros(qkv_size, dtype=self.dtype))
            self.o_proj_bias = nn.Parameter(
                torch.zeros(self.hidden_size, dtype=self.dtype)
            )
        else:
            self.qkv_proj_bias = None
            self.o_proj_bias = None

        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        # KV caches are bound externally through bind_kv_cache().
        self.k_cache = None
        self.v_cache = None

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach the checkpoint-to-parameter weight loaders."""
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
                shard_dim=1,
                num_shards=self.world_size,
                is_storage_transposed=True,
                num_kv_replicas=self.num_kv_replicas,
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.num_attention_heads_per_rank * self.head_dim,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        if self.attention_bias:
            set_weight_loader(
                self.qkv_proj_bias,
                fused_qkv_bias_loader(
                    q_size=self.q_size,
                    kv_size=self.kv_size,
                    num_shards=self.world_size,
                    num_kv_replicas=self.num_kv_replicas,
                ),
            )
            # >>> PARALLELISM: Each rank adds the O bias before the TP
            # reduction. Divide it by the TP size so that the sum is correct.
            set_weight_loader(self.o_proj_bias, scaled_bias_loader(self.world_size))

    def _qkv_bias_2d(self) -> torch.Tensor | None:
        return None if self.qkv_proj_bias is None else self.qkv_proj_bias.view(1, -1)

    def _o_bias_2d(self) -> torch.Tensor | None:
        return None if self.o_proj_bias is None else self.o_proj_bias.view(1, -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ):
        """Select the prefill path or the decode path from the metadata."""
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states, positions, position_embeddings, attn_metadata
            )

        # >>> PARALLELISM: All-gather from SP before attention <<<
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(
            hidden_states, positions, position_embeddings, attn_metadata
        )

    # -- Prefill path ---------------------------------------------------------

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """Prefill: fused QKV+bias+RoPE, flash attention and O projection."""
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, _ = hidden_states.shape

        # Step 1: Fused QKV projection + bias + RoPE. The bias is added before
        # RoPE, as in the HF reference.
        cos, sin = position_embeddings
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=self._qkv_bias_2d(),
            d_head=self.head_dim,
            cos_cache=cos.unsqueeze(0),
            sin_cache=sin.unsqueeze(0),
            num_q_heads=self.num_attention_heads_per_rank,
            num_kv_heads=self.num_key_value_heads_per_rank,
        ).squeeze(0)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )

        # Step 2: Update the KV cache
        # >>> PARALLELISM: The cache is per rank (TP-sharded KV heads) <<<
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        block_indices, position_indices = paged_slot_indices(
            slot_mapping, block_size, self.k_cache.shape[0]
        )
        k_flat = k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
        v_flat = v.reshape(-1, self.head_dim).to(self.v_cache.dtype)

        head_indices_for_put = torch.arange(
            self.num_key_value_heads_per_rank,
            dtype=torch.long,
            device=hidden_states.device,
        ).repeat_interleave(slot_mapping.shape[0])
        block_indices_for_put = block_indices.repeat(self.num_key_value_heads_per_rank)
        position_indices_for_put = position_indices.repeat(
            self.num_key_value_heads_per_rank
        )
        self.k_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            k_flat,
        )
        self.v_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            v_flat,
        )

        # Step 3: Attention. Granite has no sinks and no sliding window.
        if kv_segment_size:
            # Segmented prefill reads the prior KV from the cache. Prefix
            # caching and chunked prefill use this path.
            attn_output = NF.segmented_attention(
                q,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=block_table,
                prior_tokens=cached_seq_len,
                block_size=block_size,
                kv_segment_size=kv_segment_size,
                scale=self.scaling,
                tp_q=True,
                tp_out=True,
            )
        else:
            k = k.repeat_interleave(self.num_key_value_groups, dim=0)
            v = v.repeat_interleave(self.num_key_value_groups, dim=0)
            attn_output = NF.flash_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v,
                scale=self.scaling,
                tp_q=False,
                tp_out=True,
            )

        # Step 4: Output projection + bias
        attn_output = NF.o_proj(
            attn_output.unsqueeze(0), self.o_proj_weight, self._o_bias_2d()
        ).squeeze(0)

        # >>> PARALLELISM: Reduce-scatter to the SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    # -- Decode path ----------------------------------------------------------

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
    ):
        """Decode: fused decode megakernel with the QKV and O biases.

        >>> PARALLELISM: The kernel uses TP-sharded weights. An all-reduce
        follows the kernel. <<<
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        assert tokens == B * S_decode, (
            f"Decode token count ({tokens}) is not a multiple of the batch size ({B})."
        )

        hidden_states = hidden_states.to(self.dtype)
        S_ctx = max_blocks_per_seq * block_size
        nkh = self.num_key_value_heads_per_rank
        X = hidden_states.view(B, S_decode, hidden)

        # RoPE in the kernel layout: [T, Dh] -> [Dh/2, B, S]
        cos, sin = position_embeddings
        half_d = self.head_dim // 2
        cos_kernel = (
            cos[:, :half_d]
            .view(B, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )
        sin_kernel = (
            sin[:, :half_d]
            .view(B, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )

        attention_mask = NF.gen_attention_decode_mask(
            pos_ids=positions.view(1, B * S_decode).to(torch.float32),
            bs=B,
            q_head=self.num_attention_heads_per_rank,
            s_active=S_decode,
            s_prior=S_ctx,
            start_pos=None,
            block_len=block_size,
        )

        if nkh > 1:
            # GQA/MHA with more than one KV head on a rank reaches the fused
            # kernel only with a per-head 3D block table and the 4D cache.
            # The torch fallback accepts only the 2D table. Use the same gate
            # as attention_decode to select the table.
            k_cache, v_cache = self.k_cache, self.v_cache
            per_head_table = build_per_head_block_table(block_table, nkh).to(
                torch.int32
            )
            if _can_use_attention_block_kernel(
                X, v_cache, per_head_table, attention_dp=1
            ):
                active_blocks_table = per_head_table
            else:
                active_blocks_table = block_table.to(torch.int32)
        else:
            # One KV head on a rank: the kernel takes a 3D cache and a 2D table.
            active_blocks_table = block_table.to(torch.int32)
            k_cache = (
                self.k_cache.squeeze(1) if self.k_cache.dim() == 4 else self.k_cache
            )
            v_cache = (
                self.v_cache.squeeze(1) if self.v_cache.dim() == 4 else self.v_cache
            )

        output, K_new, V_new = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            rmsnorm_X_enabled=False,
            W_qkv=self.qkv_proj_weight,
            bias_qkv=self._qkv_bias_2d(),
            rmsnorm_QK_pre_rope_enabled=False,
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=active_blocks_table,
            K_cache=k_cache,
            V_cache=v_cache,
            attention_mask=attention_mask,
            softmax_scale=self.scaling,
            sink=None,
            update_cache=False,
            W_out=self.o_proj_weight,
            bias_out=self._o_bias_2d(),
            transposed_out=False,
            out_in_sb=False,
            attention_dp=1,
            attention_dp_group=None,
            attention_dp_rank=0,
            kv_needs_a2a=False,
        )

        if K_new.dim() == 3:
            # The kernel at kv_heads == 1 and the torch fallback give
            # K_new [d_head, B * kv_heads, S]. scatter_new_kv needs
            # [d_head, B, kv_heads, S]. The batch axis is the outer axis.
            K_new = K_new.view(self.head_dim, B, nkh, S_decode)

        scatter_new_kv(
            self.k_cache,
            self.v_cache,
            K_new,
            V_new,
            slot_mapping,
            block_size,
            nkh,
            self.head_dim,
        )

        # >>> PARALLELISM: TP all-reduce after the kernel <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 4: MLP (dense SwiGLU with bias)
# =============================================================================


class GraniteMLP(nn.Module):
    """Dense SwiGLU MLP with TP intermediate sharding and optional bias.

    >>> PARALLELISM: TP <<<
    - gate_proj, up_proj: hidden -> intermediate/TP on each rank.
    - down_proj: intermediate/TP -> hidden on each rank. Its bias is divided
      by the TP size, because each rank adds it before the reduction.
    - Prefill: all-gather -> compute -> reduce-scatter (SP).
    - Decode: compute -> all-reduce.
    """

    def __init__(self, config: GraniteConfig):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        if config.intermediate_size % self.world_size != 0:
            raise ValueError(
                f"intermediate_size ({config.intermediate_size}) is not divisible "
                f"by the TP size ({self.world_size}). Use a TP size that divides it."
            )

        self.hidden_size = config.hidden_size
        self.intermediate_size_per_rank = config.intermediate_size // self.world_size
        self.mlp_bias = config.mlp_bias
        dtype = config.torch_dtype

        self.gate_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size, self.intermediate_size_per_rank, dtype=dtype
            )
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size, self.intermediate_size_per_rank, dtype=dtype
            )
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.intermediate_size_per_rank, config.hidden_size, dtype=dtype
            )
        )
        if self.mlp_bias:
            self.gate_proj_bias = nn.Parameter(
                torch.zeros(self.intermediate_size_per_rank, dtype=dtype)
            )
            self.up_proj_bias = nn.Parameter(
                torch.zeros(self.intermediate_size_per_rank, dtype=dtype)
            )
            self.down_proj_bias = nn.Parameter(
                torch.zeros(config.hidden_size, dtype=dtype)
            )
        else:
            self.gate_proj_bias = None
            self.up_proj_bias = None
            self.down_proj_bias = None

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """>>> PARALLELISM: TP sharding of the MLP weights and biases. <<<"""
        gate_up_loader = sharding_weight_loader(
            shard_dim=1,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        down_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        set_weight_loader(self.gate_proj_weight, gate_up_loader)
        set_weight_loader(self.up_proj_weight, gate_up_loader)
        set_weight_loader(self.down_proj_weight, down_loader)

        if self.mlp_bias:
            gate_up_bias_loader = sharding_weight_loader(
                shard_dim=0,
                shard_size=self.intermediate_size_per_rank,
                num_shards=self.world_size,
            )
            set_weight_loader(self.gate_proj_bias, gate_up_bias_loader)
            set_weight_loader(self.up_proj_bias, gate_up_bias_loader)
            set_weight_loader(self.down_proj_bias, scaled_bias_loader(self.world_size))

    @staticmethod
    def _as_row(bias: torch.Tensor | None) -> torch.Tensor | None:
        return None if bias is None else bias.view(1, -1)

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        # >>> PARALLELISM: All-gather from SP for the full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        # <-- MODEL-SPECIFIC: down(silu(gate(x)) * up(x)), each with bias
        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
            gate_b=self._as_row(self.gate_proj_bias),
            up_b=self._as_row(self.up_proj_bias),
            down_b=self._as_row(self.down_proj_bias),
        )

        # >>> PARALLELISM: Combine the TP shards <<<
        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                self.tp_group.all_reduce(output)
        return output


# =============================================================================
# Section 5: Decoder Layer
# =============================================================================


class GraniteDecoderLayer(nn.Module):
    """One Granite decoder layer.

    <-- MODEL-SPECIFIC:
        h = h + residual_multiplier * attn(norm(h))
        h = h + residual_multiplier * mlp(norm(h))
    """

    def __init__(self, config: GraniteConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = GraniteRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = GraniteRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.self_attn = GraniteAttention(config, layer_idx=layer_idx)
        self.mlp = GraniteMLP(config)
        self.layer_idx = layer_idx
        self.residual_multiplier = config.residual_multiplier

    def _is_decode(self, attn_metadata) -> bool:
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
        return max_query_len <= decode_token_threshold

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        is_decode = self._is_decode(attn_metadata)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=not is_decode)
        hidden_states = residual + hidden_states * self.residual_multiplier

        return hidden_states


# =============================================================================
# Section 6: Model Backbone
# =============================================================================


class GraniteModel(nn.Module):
    """Granite transformer backbone.

    >>> PARALLELISM: SP <<<
    Prefill: the embedding scatters the tokens across the TP ranks. The model
    all-gathers the tokens after the final norm.
    Decode: each rank processes all tokens.
    """

    def __init__(self, config: GraniteConfig):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        self.embedding_multiplier = config.embedding_multiplier

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        self.layers = nn.ModuleList(
            [GraniteDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = GraniteRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = GraniteRotaryEmbedding(config)

        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=False,
            ),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
        )
        # <-- MODEL-SPECIFIC: Granite embedding multiplier
        hidden_states = hidden_states * self.embedding_multiplier

        position_embeddings = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
            )

        hidden_states = self.norm(hidden_states)

        # >>> PARALLELISM: SP all-gather of the full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states


# =============================================================================
# Section 7: Language Model Head
# =============================================================================


class GraniteForCausalLM(nn.Module):
    """Granite model with the language-model head.

    <-- MODEL-SPECIFIC: Tied embeddings. The logits are divided by
    logits_scaling before the sampler and before they are returned.
    """

    def __init__(self, config: GraniteConfig):
        super().__init__()
        self.config = config
        self.model = GraniteModel(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        self.logits_scaling = config.logits_scaling

        if config.vocab_size % self.world_size != 0:
            raise ValueError(
                f"vocab_size ({config.vocab_size}) is not divisible by the TP size "
                f"({self.world_size}). Use a TP size that divides it."
            )

        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config
            if config.neuron_config
            else None
        )
        debug_logits_enabled = (
            config.neuron_config is not None
            and config.neuron_config.debug_logits_dir is not None
        )
        self._gather_logits = (
            config.neuron_config is not None and config.neuron_config.max_logprobs != 0
        ) or debug_logits_enabled

        # The lm_head loads from model.embed_tokens.weight when the embeddings
        # are tied (see load_weights). It is a separate vocab-sharded parameter.
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
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is not None:
            raise ValueError(
                "Granite on Neuron does not support inputs_embeds. "
                "Give input_ids instead."
            )

        positions = positions.to(torch.int32)

        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        T = input_ids.shape[0]
        # >>> PARALLELISM: SP length check <<<
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt length ({T}) is not a multiple of the TP size "
                f"({self.world_size}) that is larger than the TP size. "
                "Use prefill buckets that are multiples of the TP size."
            )

        hidden_states = self.model(
            input_ids, positions, attn_metadata=attn_metadata, rank=rank
        )
        hidden_states_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )

        logits = self.lm_head(hidden_states_for_logits)
        # <-- MODEL-SPECIFIC: Granite logits scaling
        logits = logits / self.logits_scaling

        if self.on_device_sampling_config is None:
            return logits

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )

        gathered_logits = None
        if self._gather_logits:
            if self.tp_group is not None:
                gathered_logits = self.tp_group.all_gather(logits, dim=1)
            else:
                gathered_logits = logits

        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            return rejection_sampler(spec_decode_metadata, sampled_tokens)

        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = GraniteConfig.from_configs(hf_config, neuron_config)
        return cls(config)

    # -- KV Cache Management --------------------------------------------------

    def get_kv_spec(self):
        layers = []
        for i, layer in enumerate(self.model.layers):
            layers.append(
                LayerSpec(
                    name=f"layers.{i}.self_attn",
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
                raise RuntimeError(
                    f"The KV cache for layer {layer_name} is not initialized. "
                    "Allocate the KV cache before the forward pass."
                )
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # -- Weight Loading -------------------------------------------------------

    def checkpoint_mappings(self) -> dict[str, str | list[str]]:
        """Return the parameter-name to checkpoint-key mappings."""
        mappings: dict[str, str | list[str]] = {}
        for layer_id in range(len(self.model.layers)):
            prefix = f"model.layers.{layer_id}"
            attn = f"{prefix}.self_attn"
            mappings[f"{attn}.qkv_proj_weight"] = [
                f"{attn}.q_proj.weight",
                f"{attn}.k_proj.weight",
                f"{attn}.v_proj.weight",
            ]
            mappings[f"{attn}.o_proj_weight"] = f"{attn}.o_proj.weight"
            if self.config.attention_bias:
                mappings[f"{attn}.qkv_proj_bias"] = [
                    f"{attn}.q_proj.bias",
                    f"{attn}.k_proj.bias",
                    f"{attn}.v_proj.bias",
                ]
                mappings[f"{attn}.o_proj_bias"] = f"{attn}.o_proj.bias"

            mappings[f"{prefix}.input_layernorm.weight"] = (
                f"{prefix}.input_layernorm.weight"
            )
            mappings[f"{prefix}.post_attention_layernorm.weight"] = (
                f"{prefix}.post_attention_layernorm.weight"
            )

            for proj in ("gate_proj", "up_proj", "down_proj"):
                mappings[f"{prefix}.mlp.{proj}_weight"] = f"{prefix}.mlp.{proj}.weight"
                if self.config.mlp_bias:
                    mappings[f"{prefix}.mlp.{proj}_bias"] = f"{prefix}.mlp.{proj}.bias"

        mappings["model.embed_tokens.weight"] = "model.embed_tokens.weight"
        mappings["model.norm.weight"] = "model.norm.weight"
        if self.config.tie_word_embeddings:
            mappings["lm_head.weight"] = "model.embed_tokens.weight"
        else:
            mappings["lm_head.weight"] = "lm_head.weight"
        return mappings

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load the TP-sharded weights from the checkpoint.

        The checkpoint can be FP32. Each tensor is cast to the model dtype.
        """
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, self.checkpoint_mappings(), device
        ).state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self.load_state_dict(rank_sharded, strict=False, assign=True)
