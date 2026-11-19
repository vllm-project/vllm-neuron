# SPDX-License-Identifier: Apache-2.0
"""
OLMo BF16 Implementation
========================

Implementation of OLMo (``OlmoForCausalLM``) for the Neuron native backend.
Adapted from the SmolLM3 and Qwen3-VL BF16 dense models.

OLMo is a Llama-style decoder with one important difference: each norm is a
non-parametric LayerNorm (mean subtraction, no weight, no bias). The checkpoint
has no norm tensors.

Supported parallelism: TP, SP.
Starting model: allenai/OLMo-1B-hf (dense, MHA 16Q/16KV, SiLU, tied embeddings).

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    OLMo-specific. Change when porting.

KERNEL PATHS:
  Prefill: NF.qkv_proj -> torch RoPE -> NF.flash_attention or
           NF.segmented_attention -> NF.o_proj. MLP: NF.mlp.
  Decode:  NF.attention_decode (fused QKV, RoPE, attention, O proj, KV write)
           with per-head block tables. MLP: NF.mlp.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
from vllm_neuron.model.kv_cache import (
    KVSpec,
    LayerSpec,
    build_per_head_slot_mapping,
    paged_slot_indices,
)
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.neuron_utils import can_run_kernel
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from transformers import PretrainedConfig
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.sampler import Sampler

import vllm_neuron.nn as neuron_nn
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding

from .compat import build_per_head_block_table
from .config import OlmoConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1: Non-parametric LayerNorm
# <-- MODEL-SPECIFIC: OLMo LayerNorm has no weight and no bias
# =============================================================================


class OlmoLayerNorm(nn.Module):
    """Non-parametric LayerNorm.

    <-- MODEL-SPECIFIC: This is HF ``OlmoLayerNorm``. It subtracts the mean,
    divides by the standard deviation, and has no affine parameters. It
    computes in FP32 and casts back to the input dtype.
    """

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.normalized_shape = (hidden_size,)
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        return F.layer_norm(
            hidden_states.to(torch.float32),
            self.normalized_shape,
            None,
            None,
            eps=self.eps,
        ).to(orig_dtype)


# =============================================================================
# Section 2: Rotary Position Embedding (Standard RoPE)
# <-- MODEL-SPECIFIC: OLMo uses default RoPE (rotate_half style, no scaling)
# =============================================================================


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotate_half RoPE to one tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to the query and key tensors.

    Args:
        q: [Nh, T, Dh], k: [Nkv, T, Dh]
        cos: [T, Dh], sin: [T, Dh]

    Returns:
        Rotated (q, k) tensors
    """
    cos = cos.unsqueeze(0)  # [1, T, Dh]
    sin = sin.unsqueeze(0)  # [1, T, Dh]
    return _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)


class OlmoRotaryEmbedding(nn.Module):
    """Rotary position embedding with default RoPE (no scaling)."""

    def __init__(self, config: OlmoConfig):
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
        """Compute the cos/sin embeddings for the given positions.

        Returns:
            cos, sin: both of shape [T, head_dim]
        """
        inv_freq = self._compute_inv_freq(device)  # [Hd/2]
        inv_freq_expanded = inv_freq[:, None].float()  # [Hd/2, 1]
        positions_expanded = position_ids[None, :].float()  # [1, T]
        freqs = (inv_freq_expanded @ positions_expanded).transpose(0, 1)  # [T, Hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, Hd]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


# =============================================================================
# Section 3: Attention
# Mixed: PARALLELISM (TP head sharding, SP, collectives) +
#        MODEL-SPECIFIC (no bias, no clip_qkv, no sinks, no sliding window)
# =============================================================================


class OlmoAttention(nn.Module):
    """Multi-head attention with TP head sharding.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads are sharded across TP ranks
    - KV heads are replicated when there are fewer KV heads than TP ranks
    - Prefill: all-gather input -> QKV proj -> attention -> O proj -> reduce-scatter
    - Decode: fused megakernel with TP all-reduce

    <-- MODEL-SPECIFIC:
    - Separate Q/K/V checkpoint tensors, no bias
    - Default RoPE, no sinks, no sliding window
    - A rank can hold more than one KV head (for example 2 at TP 8). The decode
      kernel then needs per-head block tables and per-head write slots.
    """

    def __init__(self, config: OlmoConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: Head sharding calculation <<<
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

        # >>> PARALLELISM: QKV weight shapes for TP <<<
        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim
        qkv_size = q_size + 2 * kv_size
        o_proj_in_features = (
            self.num_attention_heads * self.head_dim
        ) // self.world_size

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_proj_in_features, self.hidden_size, dtype=self.dtype)
        )

        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        # KV caches are bound externally via bind_kv_cache()
        self.k_cache = None
        self.v_cache = None

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach the weight loaders that shard the checkpoint tensors.

        >>> PARALLELISM: Weight loaders handle TP sharding of checkpoint tensors.
        <-- MODEL-SPECIFIC: OLMo stores separate Q, K, V weights (no bias).
        """
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
                shard_size=(self.num_attention_heads * self.head_dim)
                // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    # -- Forward dispatch -----------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ):
        """Send the call to the prefill path or to the decode path.

        >>> PARALLELISM: Dispatch logic <<<
        - Prefill: all-gather for SP before attention, reduce-scatter after
        - Decode: the fused megakernel handles TP internally
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
            )

        # >>> PARALLELISM: All-gather from SP before attention <<<
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return self.forward_prefill(
            hidden_states,
            positions,
            position_embeddings,
            attn_metadata,
        )

    # -- Prefill path ---------------------------------------------------------

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """Prefill: full-sequence attention.

        Pipeline:
        1. QKV projection (NF.qkv_proj)   >>> PARALLELISM: TP sharded heads <<<
        2. RoPE (torch)
        3. KV cache update                >>> PARALLELISM: per-rank cache <<<
        4. NF.flash_attention, or NF.segmented_attention for chunked prefill
        5. Output projection (NF.o_proj)  >>> PARALLELISM: reduce-scatter <<<
        """
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # -- Step 1: QKV Projection -------------------------------------------
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
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

        # -- Step 2: Apply RoPE -----------------------------------------------
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # -- Step 3: Update KV Cache ------------------------------------------
        # >>> PARALLELISM: Cache is per-rank (TP sharded KV heads) <<<
        # The 4D cache is [num_blocks, nkh, block_size, d_head]. The decode
        # kernel folds it to [num_blocks * nkh] blocks, so block b and head h
        # are block b * nkh + h. This write uses the same rows.
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

        # -- Step 4: Attention ------------------------------------------------
        if kv_segment_size:
            # Chunked prefill: attend over the prior KV in the paged cache.
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
            )  # [Nh, Dh, T]
        else:
            k = k.repeat_interleave(self.num_key_value_groups, dim=0)
            v = v.repeat_interleave(self.num_key_value_groups, dim=0)

            attn_output = NF.flash_attention(
                q.transpose(1, 2),  # [Nh, Dh, T]
                k.transpose(1, 2),  # [Nh, Dh, T]
                v,  # [Nh, T, Dh]
                scale=self.scaling,
                tp_q=False,
                tp_out=True,
            )  # [Nh, Dh, T]

        # -- Step 5: Output Projection ----------------------------------------
        attn_output = attn_output.unsqueeze(0)  # [1, Nh, Dh, T]
        attn_output = NF.o_proj(attn_output, self.o_proj_weight, None)  # [1, T, H]
        attn_output = attn_output.squeeze(0)  # [T, H]

        # >>> PARALLELISM: Reduce-scatter to return to SP layout <<<
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
        """Decode: fused megakernel for token generation.

        >>> PARALLELISM: The megakernel handles TP internally. <<<

        The kernel accepts a rank with more than one KV head only with a 3D
        per-head block table. The torch fallback takes the 2D table.
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        assert tokens == B * S_decode

        hidden_states = hidden_states.to(self.dtype)
        nkh = self.num_key_value_heads_per_rank

        X = hidden_states.view(B, S_decode, hidden)

        # Change the RoPE layout for the megakernel: [T, Hd] -> [Hd/2, B, S]
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

        # The kernel makes the causal mask on chip from the prior-token count.
        pos_ids_kernel = positions.view(B, S_decode).to(torch.float32)

        # The kernel accepts a rank with more than one KV head only with a
        # per-head table [B, nkh, num_blocks] and per-head write slots. Both use
        # the block numbering base_block * nkh + h of the folded cache pool. The
        # torch fallback (CPU mode without the NKI simulator, or NKI kernels
        # disabled) accepts only the shared 2D table and the flat write slots.
        # The kernel reads the write slots as uint32; vLLM's slot mapping is
        # int64, and an int64 index makes the runtime reject the decode graph.
        if can_run_kernel(X):
            active_blocks_table = build_per_head_block_table(block_table, nkh).to(
                torch.int32
            )
            kv_cache_update_idx = build_per_head_slot_mapping(
                slot_mapping, block_size, nkh, B
            ).to(torch.uint32)
        else:
            active_blocks_table = block_table.to(torch.int32)
            kv_cache_update_idx = slot_mapping.view(B, S_decode).to(torch.uint32)

        # >>> PARALLELISM: Fused megakernel with TP-sharded weights <<<
        # The kernel writes the new K/V into the cache (update_cache=True).
        output = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            rmsnorm_X_enabled=False,
            W_qkv=self.qkv_proj_weight,
            bias_qkv=None,
            rmsnorm_QK_pre_rope_enabled=False,
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=active_blocks_table,
            K_cache=self.k_cache,
            V_cache=self.v_cache,
            pos_ids=pos_ids_kernel,
            swa_start_pos_ids=None,
            softmax_scale=self.scaling,
            sink=None,
            update_cache=True,
            kv_cache_update_idx=kv_cache_update_idx,
            W_out=self.o_proj_weight,
            bias_out=None,
            transposed_out=False,
            out_in_sb=False,
        )

        # >>> PARALLELISM: TP all-reduce after megakernel <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 4: MLP (Dense SwiGLU)
# =============================================================================


class OlmoMLP(nn.Module):
    """Dense SwiGLU MLP with TP intermediate sharding.

    >>> PARALLELISM: TP <<<
    - gate_proj, up_proj: hidden -> intermediate/TP per rank
    - down_proj: intermediate/TP -> hidden per rank
    - Prefill: all-gather before -> compute -> reduce-scatter after (SP)
    - Decode: compute -> all-reduce

    <-- MODEL-SPECIFIC: down(silu(gate(x)) * up(x)), no bias.
    """

    def __init__(self, config: OlmoConfig):
        super().__init__()

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.hidden_size = config.hidden_size
        self.intermediate_size_per_rank = config.intermediate_size // self.world_size

        self.gate_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_size_per_rank,
                dtype=config.torch_dtype,
            )
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_size_per_rank,
                dtype=config.torch_dtype,
            )
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.intermediate_size_per_rank,
                config.hidden_size,
                dtype=config.torch_dtype,
            )
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """>>> PARALLELISM: TP sharding of MLP weights. <<<"""
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

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        """
        >>> PARALLELISM: SP all-gather/reduce-scatter for prefill, all-reduce for decode <<<
        """
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        )

        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 5: Decoder Layer
# =============================================================================


class OlmoDecoderLayer(nn.Module):
    """Single transformer decoder layer.

    Architecture (MODEL-SPECIFIC):
        x -> LayerNorm -> Attention -> residual -> LayerNorm -> MLP -> residual
    Each LayerNorm is non-parametric.
    """

    def __init__(self, config: OlmoConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = OlmoLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.post_attention_layernorm = OlmoLayerNorm(
            config.hidden_size, config.layer_norm_eps
        )
        self.self_attn = OlmoAttention(config, layer_idx=layer_idx)
        self.mlp = OlmoMLP(config)
        self.layer_idx = layer_idx

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
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=not is_decode)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Section 6: Model Backbone
# =============================================================================


class OlmoModel(nn.Module):
    """OLMo transformer backbone.

    >>> PARALLELISM: SP (Sequence Parallelism) <<<
    During prefill:
    - After embedding: split the sequence across TP ranks (T/world_size each)
    - After all layers: all-gather to make the full sequence again
    During decode:
    - No SP; all ranks process all tokens
    """

    def __init__(self, config: OlmoConfig):
        super().__init__()
        self.config = config

        # >>> PARALLELISM: TP group for SP <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: Vocab-sharded embedding <<<
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )

        self.layers = nn.ModuleList(
            [
                OlmoDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # <-- MODEL-SPECIFIC: Final non-parametric LayerNorm
        self.norm = OlmoLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.rotary_emb = OlmoRotaryEmbedding(config)

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

        # >>> PARALLELISM: VocabDimShardedEmbedding handles SP internally <<<
        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
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
            )

        hidden_states = self.norm(hidden_states)

        # >>> PARALLELISM: SP - all-gather to make the full sequence again <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states


# =============================================================================
# Section 7: Language Model Head
# =============================================================================


class OlmoForCausalLM(nn.Module):
    """OLMo model with a language modeling head.

    >>> PARALLELISM: Column-parallel linear for LM head <<<

    <-- MODEL-SPECIFIC: OLMo-1B ties the embeddings. The checkpoint has no
    ``lm_head.weight``, so the loader reads ``model.embed_tokens.weight``.
    """

    def __init__(self, config: OlmoConfig):
        super().__init__()
        self.config = config

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        self._validate_tp(config)

        self.model = OlmoModel(config)

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

    def _validate_tp(self, config: OlmoConfig) -> None:
        """Make sure that the TP degree shards each dimension evenly."""
        tp = self.world_size
        heads_per_rank, heads_rem = divmod(config.num_attention_heads, tp)
        kv = config.num_key_value_heads
        checks = {
            "num_attention_heads % tp": heads_rem == 0,
            "(num_attention_heads / tp) is even": heads_per_rank % 2 == 0,
            "kv heads divide or replicate": kv % tp == 0 or tp % kv == 0,
            "intermediate_size % tp": config.intermediate_size % tp == 0,
            "vocab_size % tp": config.vocab_size % tp == 0,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise ValueError(
                f"OLMo does not support tensor_parallel_size={tp}. "
                f"These checks fail: {failed}. Use tensor_parallel_size 2, 4 or 8."
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
                "OLMo does not support inputs_embeds. Give input_ids instead."
            )

        positions = positions.to(torch.int32)

        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        T = input_ids.shape[0]

        # >>> PARALLELISM: SP length validation <<<
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt length ({T}) must be larger than world_size "
                f"({self.world_size}) and divide by it for SP."
            )

        hidden_states = self.model(
            input_ids, positions, attn_metadata=attn_metadata, rank=rank
        )

        hidden_states_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )

        logits = self.lm_head(hidden_states_for_logits)

        if self.on_device_sampling_config is None:
            return logits

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )

        # >>> PARALLELISM: Gather sharded logits for logprobs computation <<<
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
        config = OlmoConfig.from_configs(hf_config, neuron_config)
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
                    f"KV cache for layer {layer_name} is not initialized. "
                    "Allocate the KV cache before the forward pass."
                )
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # -- Weight Loading -------------------------------------------------------

    @staticmethod
    def checkpoint_mappings(
        num_layers: int, tie_word_embeddings: bool
    ) -> dict[str, str | list[str]]:
        """Return the map from model parameter names to checkpoint keys.

        <-- MODEL-SPECIFIC: OLMo has no norm tensors in the checkpoint.
        """
        mappings: dict[str, str | list[str]] = {}
        for layer_id in range(num_layers):
            prefix = f"model.layers.{layer_id}"
            mappings[f"{prefix}.self_attn.qkv_proj_weight"] = [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ]
            mappings[f"{prefix}.self_attn.o_proj_weight"] = (
                f"{prefix}.self_attn.o_proj.weight"
            )
            mappings[f"{prefix}.mlp.gate_proj_weight"] = (
                f"{prefix}.mlp.gate_proj.weight"
            )
            mappings[f"{prefix}.mlp.up_proj_weight"] = f"{prefix}.mlp.up_proj.weight"
            mappings[f"{prefix}.mlp.down_proj_weight"] = (
                f"{prefix}.mlp.down_proj.weight"
            )
        mappings["model.embed_tokens.weight"] = "model.embed_tokens.weight"
        mappings["lm_head.weight"] = (
            "model.embed_tokens.weight" if tie_word_embeddings else "lm_head.weight"
        )
        return mappings

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load the weights from the checkpoint and cast them to bf16.

        >>> PARALLELISM: Weight loaders handle TP sharding <<<
        """
        mappings = self.checkpoint_mappings(
            len(self.model.layers), self.config.tie_word_embeddings
        )

        # strict=True (the default) raises when a mapped checkpoint key is absent.
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, mappings, device
        ).state_dict

        # The original checkpoint is FP32. Cast each weight one time to bf16.
        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self.load_state_dict(rank_sharded, strict=False, assign=True)
