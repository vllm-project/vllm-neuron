# SPDX-License-Identifier: Apache-2.0
"""
HyperCLOVAX BF16 Implementation
===============================

Implementation of HyperCLOVAX (dense, GQA 48Q/8KV, SiLU, untied LM head) for
the native Neuron backend. The structure follows the SmolLM3 and Llama3 BF16
models. The attention prefill path uses the fused-RoPE QKV kernel of Llama3.

HyperCLOVAX differs from Llama in these items:

- The embedding output is multiplied by ``embedding_multiplier``.
- The softmax scale is ``attention_multiplier``, not ``head_dim ** -0.5``.
- Peri-layer normalization: ``post_norm1`` and ``post_norm2`` normalize the
  attention output and the MLP output before the residual add.
- Each branch output is multiplied by ``residual_multiplier``.
- The LM head output is multiplied by ``logits_scaling``.

The model applies each multiplier at runtime as a scalar multiply, as the HF
reference does. The weights stay unchanged.

Supported parallelism: TP, SP.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code.
  # <-- MODEL-SPECIFIC: ...    HyperCLOVAX-specific.
"""

import logging

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import HyperCLOVAXConfig
from .o_proj import o_proj

logger = logging.getLogger(__name__)

# The decode attention mask is the same for all layers in one forward. The
# first layer stores it in attn_metadata, and the other layers use it again.
_DECODE_MASK_CACHE_KEY = "_cached_decode_mask"


# =============================================================================
# Section 1: RMS Normalization
# <-- MODEL-SPECIFIC: HF HyperCLOVAXRMSNorm. Compute in float32, cast back to
# the input dtype, then multiply by the weight.
# =============================================================================


class HyperCLOVAXRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# =============================================================================
# Section 2: Rotary Position Embedding (default RoPE, rotate_half layout)
# =============================================================================


class HyperCLOVAXRotaryEmbedding(nn.Module):
    """Default RoPE. The QKV prefill kernel and the decode kernel apply it."""

    def __init__(self, config: HyperCLOVAXConfig):
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
        """Return cos and sin, both of shape [T, head_dim]."""
        inv_freq = self._compute_inv_freq(device)  # [Hd/2]
        inv_freq_expanded = inv_freq[:, None].float()  # [Hd/2, 1]
        positions_expanded = position_ids[None, :].float()  # [1, T]
        freqs = (inv_freq_expanded @ positions_expanded).transpose(0, 1)  # [T, Hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, Hd]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


# =============================================================================
# Section 3: Attention
# =============================================================================


class HyperCLOVAXAttention(nn.Module):
    """GQA attention with TP head sharding.

    >>> PARALLELISM: TP <<<
    - Prefill: all-gather input -> fused QKV+RoPE kernel -> flash or segmented
      attention kernel -> O proj kernel -> reduce-scatter
    - Decode: fused attention_block_tkg megakernel, then TP all-reduce

    <-- MODEL-SPECIFIC: softmax scale is attention_multiplier. No bias.
    """

    def __init__(self, config: HyperCLOVAXConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        # <-- MODEL-SPECIFIC: MuP softmax scale
        self.scaling = config.attention_multiplier

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        if self.num_attention_heads % self.world_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) is not divisible "
                f"by the TP size ({self.world_size}). Use a TP size that divides it."
            )

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

        # KV caches bound externally via bind_kv_cache()
        self.k_cache = None
        self.v_cache = None

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach the loaders that shard the checkpoint tensors for TP.

        <-- MODEL-SPECIFIC: The checkpoint stores separate Q, K, V weights.
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

    def _write_kv_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ) -> None:
        """Write K [Nkv, T, Dh] and V [Nkv, T, Dh] to the paged cache."""
        blk_idx = slot_mapping // block_size
        pos_idx = slot_mapping % block_size
        num_slots = slot_mapping.shape[0]
        nkh = self.num_key_value_heads_per_rank

        k_f = k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
        v_f = v.reshape(-1, self.head_dim).to(self.v_cache.dtype)
        h_idx = torch.arange(nkh, dtype=torch.long, device=k.device).repeat_interleave(
            num_slots
        )
        index = (blk_idx.repeat(nkh), h_idx, pos_idx.repeat(nkh))
        self.k_cache.index_put_(index, k_f)
        self.v_cache.index_put_(index, v_f)

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """Prefill: fused QKV+RoPE, then flash or segmented attention, then O proj."""
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, _ = hidden_states.shape

        cos, sin = position_embeddings
        cos_cache = cos.unsqueeze(0)
        sin_cache = sin.unsqueeze(0)

        # >>> PARALLELISM: Cache is per-rank (TP sharded KV heads) <<<
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        if kv_segment_size:
            # Segmented prefill: the QKV kernel writes the post-RoPE K/V to the
            # cache. The kernel returns (q, k_cache, v_cache). The FX aliasing
            # pass threads the in-place cache write back to self.k_cache and
            # self.v_cache for segmented_attention.
            max_slot = self.k_cache.shape[0] * block_size
            slot_mapping_clamped = torch.where(
                (slot_mapping < 0) | (slot_mapping >= max_slot),
                torch.zeros_like(slot_mapping),
                slot_mapping,
            ).to(torch.int32)

            q_hbm, _, _ = NF.qkv_proj(
                hidden=hidden_states.unsqueeze(0),
                qkv_weights=self.qkv_proj_weight,
                bias=None,
                d_head=self.head_dim,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                num_q_heads=self.num_attention_heads_per_rank,
                num_kv_heads=self.num_key_value_heads_per_rank,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                k_scale=None,
                v_scale=None,
                fp8_max=None,
                fp8_min=None,
                kv_dtype=None,
                use_block_kv=True,
                fp8_packed=False,
                block_size=block_size,
                slot_mapping=slot_mapping_clamped.unsqueeze(0),
            )
            q = (
                q_hbm.squeeze(0)
                .view(tokens, self.num_attention_heads_per_rank, self.head_dim)
                .transpose(0, 1)
            )

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
                allow_vector_exp=False,
            )  # [Nh, Dh, T]
        else:
            # Full prefill: fused QKV+RoPE kernel, cache write, flash attention.
            qkv = NF.qkv_proj(
                hidden=hidden_states.unsqueeze(0),
                qkv_weights=self.qkv_proj_weight,
                bias=None,
                d_head=self.head_dim,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                num_q_heads=self.num_attention_heads_per_rank,
                num_kv_heads=self.num_key_value_heads_per_rank,
            ).squeeze(0)

            q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
            q = q.view(
                tokens, self.num_attention_heads_per_rank, self.head_dim
            ).transpose(0, 1)
            k = k.view(
                tokens, self.num_key_value_heads_per_rank, self.head_dim
            ).transpose(0, 1)
            v = v.view(
                tokens, self.num_key_value_heads_per_rank, self.head_dim
            ).transpose(0, 1)

            self._write_kv_cache(k, v, slot_mapping, block_size)

            k = k.repeat_interleave(self.num_key_value_groups, dim=0)
            v = v.repeat_interleave(self.num_key_value_groups, dim=0)

            attn_output = NF.flash_attention(
                q.transpose(1, 2),  # [Nh, Dh, T]
                k.transpose(1, 2),  # [Nh, Dh, T]
                v,  # [Nh, T, Dh]
                scale=self.scaling,
                tp_q=False,
                tp_out=True,
                allow_vector_exp=False,
            )  # [Nh, Dh, T]

        # Give the O projection an fp32 input. The kernel casts the input to the
        # weight dtype for the matmul, and it writes the fp32 PSUM sum without a
        # bf16 rounding. The reduce-scatter then adds the TP partials in fp32.
        # With bf16 partials and a bf16 reduction, the attention error is 1.3x
        # the HF bf16 error, because the 1/128 softmax scale makes the output
        # rounding the main error source.
        # src/o_proj.py carries the O projection with a torch fallback that
        # accepts this fp32 input (the plugin's fallback does not).
        attn_output = attn_output.unsqueeze(0).to(torch.float32)  # [1, Nh, Dh, T]
        attn_output = o_proj(attn_output, self.o_proj_weight, None)  # [1, T, H]
        attn_output = attn_output.squeeze(0)  # [T, H] fp32

        # >>> PARALLELISM: Reduce-scatter to return to SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.to(self.dtype).contiguous()

    # -- Decode path ----------------------------------------------------------

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
    ):
        """Decode: fused attention_block_tkg megakernel, then TP all-reduce."""
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        assert tokens == B * S_decode

        hidden_states = hidden_states.to(self.dtype)
        S_ctx = max_blocks_per_seq * block_size

        X = hidden_states.view(B, S_decode, hidden)

        # Megakernel RoPE layout: [T, Hd] -> [Hd/2, B, S]
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

        attention_mask = attn_metadata.get(_DECODE_MASK_CACHE_KEY)
        if attention_mask is None:
            pos_ids = positions.view(1, B * S_decode)
            attention_mask = NF.gen_attention_decode_mask(
                pos_ids=pos_ids.to(torch.float32),
                bs=B,
                q_head=self.num_attention_heads_per_rank,
                s_active=S_decode,
                s_prior=S_ctx,
                start_pos=None,
                block_len=block_size,
            )
            attn_metadata[_DECODE_MASK_CACHE_KEY] = attention_mask

        # >>> PARALLELISM: Fused megakernel with TP-sharded weights <<<
        # update_cache=True: the kernel writes the new K/V in place, and the FX
        # aliasing pass threads the write back to the caller's cache.
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
            active_blocks_table=block_table,
            K_cache=self.k_cache,
            V_cache=self.v_cache,
            attention_mask=attention_mask,
            softmax_scale=self.scaling,
            sink=None,
            update_cache=True,
            kv_cache_update_idx=slot_mapping.view(B, S_decode).to(torch.uint32),
            W_out=self.o_proj_weight,
            bias_out=None,
            transposed_out=False,
            out_in_sb=False,
        )

        # >>> PARALLELISM: TP all-reduce after megakernel <<<
        # The megakernel writes a bf16 partial. Add the partials in fp32, so
        # that the collective does not round each partial sum to bf16.
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output.to(torch.float32))
            output = output.to(self.dtype)

        return output


# =============================================================================
# Section 4: MLP (dense SwiGLU)
# =============================================================================


class HyperCLOVAXMLP(nn.Module):
    """Dense SwiGLU MLP with TP intermediate sharding.

    >>> PARALLELISM: TP <<<
    - Prefill: all-gather before -> NF.mlp -> reduce-scatter after (SP)
    - Decode: NF.mlp -> all-reduce
    """

    def __init__(self, config: HyperCLOVAXConfig):
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
        # >>> PARALLELISM: All-gather from SP for full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        )

        # >>> PARALLELISM: Combine TP shards <<<
        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 5: Decoder Layer
# <-- MODEL-SPECIFIC: peri-layer normalization and residual_multiplier
# =============================================================================


class HyperCLOVAXDecoderLayer(nn.Module):
    """One decoder layer.

    h = h + post_norm1(attn(input_layernorm(h))) * residual_multiplier
    h = h + post_norm2(mlp(post_attention_layernorm(h))) * residual_multiplier
    """

    def __init__(self, config: HyperCLOVAXConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.residual_multiplier = float(config.residual_multiplier)
        self.use_post_norm = config.use_post_norm

        self.input_layernorm = HyperCLOVAXRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = HyperCLOVAXRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        if self.use_post_norm:
            self.post_norm1 = HyperCLOVAXRMSNorm(
                config.hidden_size, config.rms_norm_eps, config.torch_dtype
            )
            self.post_norm2 = HyperCLOVAXRMSNorm(
                config.hidden_size, config.rms_norm_eps, config.torch_dtype
            )
        self.self_attn = HyperCLOVAXAttention(config, layer_idx=layer_idx)
        self.mlp = HyperCLOVAXMLP(config)

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
        if self.use_post_norm:
            hidden_states = self.post_norm1(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=not is_decode)
        if self.use_post_norm:
            hidden_states = self.post_norm2(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier

        return hidden_states


# =============================================================================
# Section 6: Model Backbone
# =============================================================================


class HyperCLOVAXModel(nn.Module):
    """HyperCLOVAX backbone.

    >>> PARALLELISM: SP <<<
    Prefill: the embedding scatters tokens across TP ranks, and the final
    all-gather rebuilds the full sequence. Decode: no SP.
    """

    def __init__(self, config: HyperCLOVAXConfig):
        super().__init__()
        self.config = config
        self.embedding_multiplier = float(config.embedding_multiplier)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        self.layers = nn.ModuleList(
            [
                HyperCLOVAXDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = HyperCLOVAXRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = HyperCLOVAXRotaryEmbedding(config)

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
        # <-- MODEL-SPECIFIC: MuP embedding multiplier
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

        # >>> PARALLELISM: SP - all-gather to rebuild the full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states


# =============================================================================
# Section 7: Language Model Head
# =============================================================================


class HyperCLOVAXForCausalLM(nn.Module):
    """HyperCLOVAX with an untied, column-parallel LM head.

    <-- MODEL-SPECIFIC: logits = lm_head(h) * logits_scaling.
    """

    def __init__(self, config: HyperCLOVAXConfig):
        super().__init__()
        self.config = config
        self.model = HyperCLOVAXModel(config)
        self.logits_scaling = float(config.logits_scaling)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config
            if config.neuron_config
            else None
        )
        # Gather logits if max_logprobs != 0 OR if debug logits is enabled
        debug_logits_enabled = (
            config.neuron_config is not None
            and config.neuron_config.debug_logits_dir is not None
        )
        self._gather_logits = (
            config.neuron_config is not None and config.neuron_config.max_logprobs != 0
        ) or debug_logits_enabled

        if config.vocab_size % self.world_size != 0:
            raise ValueError(
                f"vocab_size ({config.vocab_size}) is not divisible by the TP size "
                f"({self.world_size}). Use a TP size that divides it."
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
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is not None:
            raise ValueError(
                "HyperCLOVAX on Neuron does not support inputs_embeds. "
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

        # >>> PARALLELISM: SP length validation <<<
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt length ({T}) is not valid for SP. The length must be "
                f"larger than world_size ({self.world_size}) and divisible by it."
            )

        hidden_states = self.model(
            input_ids, positions, attn_metadata=attn_metadata, rank=rank
        )

        hidden_states_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )

        # <-- MODEL-SPECIFIC: MuP logits scaling (multiply)
        logits = self.lm_head(hidden_states_for_logits) * self.logits_scaling

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
        config = HyperCLOVAXConfig.from_configs(hf_config, neuron_config)
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
                raise RuntimeError(f"KV cache for layer {layer_name} not initialized")
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # -- Weight Loading -------------------------------------------------------

    def get_weight_mappings(self) -> dict[str, str | list[str]]:
        """Return the parameter name -> HF checkpoint key(s) map.

        A parameter that is not in the map loads from the key with its own name
        (model.embed_tokens.weight, model.norm.weight, lm_head.weight).
        """
        mappings: dict[str, str | list[str]] = {}
        for layer_id in range(len(self.model.layers)):
            prefix = f"model.layers.{layer_id}"
            mappings[f"{prefix}.self_attn.qkv_proj_weight"] = [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ]
            mappings[f"{prefix}.self_attn.o_proj_weight"] = (
                f"{prefix}.self_attn.o_proj.weight"
            )
            for norm in ("input_layernorm", "post_attention_layernorm"):
                mappings[f"{prefix}.{norm}.weight"] = f"{prefix}.{norm}.weight"
            if self.config.use_post_norm:
                for norm in ("post_norm1", "post_norm2"):
                    mappings[f"{prefix}.{norm}.weight"] = f"{prefix}.{norm}.weight"
            mappings[f"{prefix}.mlp.gate_proj_weight"] = (
                f"{prefix}.mlp.gate_proj.weight"
            )
            mappings[f"{prefix}.mlp.up_proj_weight"] = f"{prefix}.mlp.up_proj.weight"
            mappings[f"{prefix}.mlp.down_proj_weight"] = (
                f"{prefix}.mlp.down_proj.weight"
            )

        # <-- MODEL-SPECIFIC: tie_word_embeddings is false for the released
        # checkpoint, which stores lm_head.weight. Keep the tie path for configs
        # that set it.
        if self.config.tie_word_embeddings:
            mappings["lm_head.weight"] = "model.embed_tokens.weight"
        return mappings

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load and shard the weights. Cast each float32 weight to bfloat16."""
        mappings = self.get_weight_mappings()

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, mappings, device
        ).state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        # load_state_dict(strict=False) does not report a missing parameter.
        # Check that each parameter has a loaded tensor of the correct shape.
        expected = dict(self.named_parameters())
        missing = sorted(set(expected) - set(rank_sharded))
        if missing:
            raise RuntimeError(
                f"The checkpoint load did not give {len(missing)} parameter(s): "
                f"{missing[:8]}. Make sure that the checkpoint is a complete "
                f"HyperCLOVAX checkpoint."
            )
        bad_shapes = [
            (name, tuple(rank_sharded[name].shape), tuple(param.shape))
            for name, param in expected.items()
            if tuple(rank_sharded[name].shape) != tuple(param.shape)
        ]
        if bad_shapes:
            raise RuntimeError(
                f"Loaded weight shape is not equal to the parameter shape for "
                f"{len(bad_shapes)} parameter(s): {bad_shapes[:4]}. Make sure "
                f"that the checkpoint config agrees with the model config."
            )

        self.load_state_dict(rank_sharded, strict=False, assign=True)

        on_meta = [n for n, p in self.named_parameters() if p.is_meta]
        if on_meta:
            raise RuntimeError(
                f"{len(on_meta)} parameter(s) stay on the meta device after the "
                f"load: {on_meta[:8]}. Make sure that each parameter has a "
                f"checkpoint key."
            )
