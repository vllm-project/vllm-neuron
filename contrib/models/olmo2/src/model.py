# SPDX-License-Identifier: Apache-2.0
"""
OLMo2 BF16 Implementation
=========================

Implementation of OLMo2 for the Neuron backend. The structure follows the
SmolLM3 and Qwen3 BF16 models.

OLMo2 differs from Llama in these places:
- Post-norm decoder layer: ``h = h + norm(attn(h))`` and
  ``h = h + norm(mlp(h))``. The layer has no input norm.
- Q and K each have one RMSNorm over the full projected width (all heads),
  before RoPE. Under TP, each rank owns a part of that width, so the sum of
  squares needs a TP all-reduce.
- Untied lm_head.

Supported parallelism: TP, SP.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code.
  # <-- MODEL-SPECIFIC: ...    OLMo2-specific code.
"""

import logging

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
from vllm_neuron.model.kv_cache import (
    KVSpec,
    LayerSpec,
    build_per_head_block_table,
    scatter_new_kv,
)
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
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

from .config import Olmo2Config

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1: RMS Normalization
# <-- MODEL-SPECIFIC: OLMo2 RMSNorm multiplies by the weight before the cast.
# =============================================================================


class Olmo2RMSNorm(nn.Module):
    """RMSNorm that computes in FP32 and casts the weighted result.

    <-- MODEL-SPECIFIC: The HF Olmo2RMSNorm returns
    ``(weight * normed_fp32).to(input_dtype)``. Llama casts before the weight
    multiply. This order changes the BF16 rounding, so keep it.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.to(torch.float32) * x).to(input_dtype)


# =============================================================================
# Section 2: Rotary Position Embedding (Standard RoPE)
# =============================================================================


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """<-- MODEL-SPECIFIC: rotate_half RoPE."""
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
        cos: [T, Dh], sin: [T, Dh] in FP32

    Returns:
        The rotated (q, k) tensors, in the dtype of q and k.
    """
    # <-- MODEL-SPECIFIC: HF OLMo2 keeps cos/sin in FP32. It rotates q and k in
    # FP32 and rounds one time to the input dtype. Do the same calculation.
    cos = cos.float().unsqueeze(0)  # [1, T, Dh]
    sin = sin.float().unsqueeze(0)  # [1, T, Dh]
    q_rot = _apply_rotary_emb(q.float(), cos, sin).to(q.dtype)
    k_rot = _apply_rotary_emb(k.float(), cos, sin).to(k.dtype)
    return q_rot, k_rot


class Olmo2RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with standard RoPE (no scaling)."""

    def __init__(self, config: Olmo2Config):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim

    def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
                / self.head_dim
            )
        )
        return inv_freq

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the cos/sin embeddings for the positions.

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
# =============================================================================


def _all_reduce_fp32(tp_group, x: torch.Tensor) -> torch.Tensor:
    """All-reduce the decode partial sums in FP32. Return the input dtype.

    <-- MODEL-SPECIFIC: A BF16 all-reduce rounds each partial sum of the ring.
    HF rounds the full projection output one time. The FP32 sum keeps one
    final rounding. This keeps the decode logits near the HF BF16 reference.
    """
    return tp_group.all_reduce(x.to(torch.float32)).to(x.dtype)


class Olmo2Attention(nn.Module):
    """Multi-head attention with TP head sharding and full-width Q/K RMSNorm.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads are sharded across TP ranks. Each rank owns whole KV heads.
    - Prefill: all-gather input -> QKV proj -> QK norm -> RoPE -> attention
      -> O proj -> reduce-scatter
    - Decode: QKV proj -> QK norm -> RoPE -> fused decode megakernel
      -> all-reduce

    <-- MODEL-SPECIFIC:
    - q_norm and k_norm normalize over all heads (width num_heads * head_dim).
      The decode megakernel has only a per-head pre-RoPE norm, so the norm
      runs before the megakernel. The megakernel then gets the normalized
      Q/K/V as its input, with a 0/1 selection matrix as W_qkv.
    """

    def __init__(self, config: Olmo2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5
        self.rms_norm_eps = config.rms_norm_eps

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self._validate_tp()

        # >>> PARALLELISM: Head sharding calculation <<<
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
        self.num_key_value_heads_per_rank = self.num_key_value_heads // self.world_size
        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
        )

        # >>> PARALLELISM: QKV weight shapes for TP <<<
        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim
        qkv_size = q_size + 2 * kv_size

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(q_size, self.hidden_size, dtype=self.dtype)
        )

        # <-- MODEL-SPECIFIC: Full-width Q/K norm weights, sharded like the heads.
        self.q_norm_weight = nn.Parameter(torch.ones(q_size, dtype=self.dtype))
        self.k_norm_weight = nn.Parameter(torch.ones(kv_size, dtype=self.dtype))

        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_size = qkv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        # The decode megakernel input width. Keep the model hidden size, so
        # that the megakernel input width and output width are equal.
        self.decode_kernel_width = max(self.hidden_size, qkv_size)

        # KV caches bound externally via bind_kv_cache()
        self.k_cache = None
        self.v_cache = None

        self._setup_weight_loaders()

    def _validate_tp(self) -> None:
        """Stop with a clear error for a TP size that this model cannot use."""
        ws = self.world_size
        if self.num_attention_heads % ws != 0 or (self.num_attention_heads // ws) % 2:
            raise ValueError(
                f"Tensor parallel size {ws} gives an odd or fractional count of "
                f"attention heads per rank ({self.num_attention_heads} heads). "
                "Use a tensor parallel size that gives an even head count per rank."
            )
        # The full-width K norm sums the squares across ranks. A replicated KV
        # head would count more than one time, so each rank must own whole KV heads.
        if self.num_key_value_heads % ws != 0:
            raise ValueError(
                f"Tensor parallel size {ws} does not divide the "
                f"{self.num_key_value_heads} KV heads. OLMo2 does not support KV "
                "head replication. Use a tensor parallel size that divides the KV heads."
            )

    def _setup_weight_loaders(self):
        """Attach the weight loaders for the checkpoint -> parameter transform.

        >>> PARALLELISM: The weight loaders do the TP sharding. <<<
        <-- MODEL-SPECIFIC: Separate Q, K, V weights, no bias. The q_norm and
        k_norm weights have the full projected width, so shard them like Q and K.
        """
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
                shard_dim=1,
                num_shards=self.world_size,
                is_storage_transposed=True,
                num_kv_replicas=1,
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.q_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        set_weight_loader(
            self.q_norm_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.q_size,
                num_shards=self.world_size,
            ),
        )
        set_weight_loader(
            self.k_norm_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.kv_size,
                num_shards=self.world_size,
            ),
        )

    # -- Q/K norm -------------------------------------------------------------

    def _qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the full-width OLMo2 RMSNorm to the local Q and K slices.

        <-- MODEL-SPECIFIC: HF normalizes q_proj(x) and k_proj(x) over the full
        width. Each rank computes the FP32 sum of squares of its slice.
        >>> PARALLELISM: One TP all-reduce gives the global sum for Q and K. <<<

        Args:
            q: [T, q_size] local Q slice.
            k: [T, kv_size] local K slice.

        Returns:
            The normalized (q, k), in the input dtype.
        """
        input_dtype = q.dtype
        q32 = q.to(torch.float32)
        k32 = k.to(torch.float32)
        sumsq = torch.cat(
            (
                q32.pow(2).sum(dim=-1, keepdim=True),
                k32.pow(2).sum(dim=-1, keepdim=True),
            ),
            dim=-1,
        )  # [T, 2]
        if self.world_size > 1:
            sumsq = self.tp_group.all_reduce(sumsq)
        q_var = sumsq[:, 0:1] / (self.q_size * self.world_size)
        k_var = sumsq[:, 1:2] / (self.kv_size * self.world_size)
        q32 = q32 * torch.rsqrt(q_var + self.rms_norm_eps)
        k32 = k32 * torch.rsqrt(k_var + self.rms_norm_eps)
        q_out = (self.q_norm_weight.to(torch.float32) * q32).to(input_dtype)
        k_out = (self.k_norm_weight.to(torch.float32) * k32).to(input_dtype)
        return q_out, k_out

    # -- Forward dispatch -----------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ):
        """Dispatch to the prefill path or the decode path from the metadata."""
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
        1. QKV projection (NKI qkv kernel)
        2. Full-width Q/K RMSNorm   <-- MODEL-SPECIFIC
        3. RoPE
        4. KV cache update
        5. Flash attention or segmented attention (NKI)
        6. Output projection (NKI) and reduce-scatter
        """
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, _ = hidden_states.shape

        # -- Step 1: QKV Projection -------------------------------------------
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
        ).squeeze(0)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        # -- Step 2: Full-width Q/K RMSNorm -----------------------------------
        q, k = self._qk_norm(q, k)

        q = q.reshape(tokens, self.num_attention_heads_per_rank, self.head_dim)
        q = q.transpose(0, 1)
        k = k.reshape(tokens, self.num_key_value_heads_per_rank, self.head_dim)
        k = k.transpose(0, 1)
        v = v.reshape(tokens, self.num_key_value_heads_per_rank, self.head_dim)
        v = v.transpose(0, 1)

        # -- Step 3: RoPE -----------------------------------------------------
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # -- Step 4: Update KV Cache -----------------------------------------
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size

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

        # -- Step 5: Attention ------------------------------------------------
        if kv_segment_size:
            # Segmented prefill: attend over the KV cache segments.
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

        # -- Step 6: Output Projection ----------------------------------------
        attn_output = attn_output.unsqueeze(0)  # [1, Nh, Dh, T]
        attn_output = NF.o_proj(attn_output, self.o_proj_weight, None)  # [1, T, H]
        attn_output = attn_output.squeeze(0)  # [T, H]

        # >>> PARALLELISM: Reduce-scatter to return to the SP layout <<<
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
    ) -> torch.Tensor:
        """Decode: QKV projection, Q/K norm, then the fused decode megakernel.

        <-- MODEL-SPECIFIC: The megakernel cannot do the full-width Q/K norm.
        This path computes Q/K/V with the NKI qkv kernel and normalizes Q/K.
        Then it applies RoPE in FP32 and puts ``[q, k, v, 0]`` into the
        megakernel input X. The megakernel W_qkv is a 0/1 selection matrix, so
        its projection gives ``[q, k, v]`` again without a numerical change.
        The megakernel does the paged attention and the output projection.

        >>> PARALLELISM: TP all-reduce after the megakernel. <<<
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens, _ = hidden_states.shape
        S_decode = tokens // B
        assert tokens == B * S_decode, (
            f"decode token count ({tokens}) must be a multiple of batch size "
            f"({B}); got S_decode={S_decode}"
        )

        hidden_states = hidden_states.to(self.dtype)
        S_ctx = max_blocks_per_seq * block_size
        nkh = self.num_key_value_heads_per_rank

        # -- Q/K/V projection and full-width Q/K norm -------------------------
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
        ).squeeze(0)  # [T, qkv_size]
        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
        q, k = self._qk_norm(q, k)

        # <-- MODEL-SPECIFIC: Do RoPE here in FP32, as prefill and HF OLMo2 do.
        # The megakernel RoPE uses BF16 cos/sin, which adds error at each decode
        # step. The megakernel gets cos=None and does not apply RoPE again.
        cos, sin = position_embeddings
        q = q.reshape(tokens, self.num_attention_heads_per_rank, self.head_dim)
        k = k.reshape(tokens, nkh, self.head_dim)
        q, k = apply_rotary_pos_emb(q.transpose(0, 1), k.transpose(0, 1), cos, sin)
        q = q.transpose(0, 1).reshape(tokens, -1)
        k = k.transpose(0, 1).reshape(tokens, -1)

        pad_width = self.decode_kernel_width - self.qkv_size
        parts = [q, k, v]
        if pad_width > 0:
            parts.append(
                torch.zeros(
                    tokens, pad_width, dtype=self.dtype, device=hidden_states.device
                )
            )
        X = torch.cat(parts, dim=-1).view(B, S_decode, self.decode_kernel_width)

        # Row i of the selection matrix selects column i of X. The rows for the
        # zero padding are zero.
        W_select = torch.eye(
            self.decode_kernel_width,
            self.qkv_size,
            dtype=self.dtype,
            device=hidden_states.device,
        )

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

        if nkh > 1:
            # GQA layout: the megakernel needs a per-head 3D block table and the
            # 4D cache. A 2D table sends this case to the torch fallback.
            active_blocks_table = build_per_head_block_table(block_table, nkh).to(
                torch.int32
            )
            k_cache, v_cache = self.k_cache, self.v_cache
        else:
            active_blocks_table = block_table.to(torch.int32)
            k_cache = (
                self.k_cache.squeeze(1) if self.k_cache.dim() == 4 else self.k_cache
            )
            v_cache = (
                self.v_cache.squeeze(1) if self.v_cache.dim() == 4 else self.v_cache
            )

        output, K_new, V_new = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.decode_kernel_width,
            rmsnorm_X_enabled=False,
            W_qkv=W_select,
            bias_qkv=None,
            rmsnorm_QK_pre_rope_enabled=False,
            rmsnorm_QK_post_rope_enabled=False,
            cos=None,
            sin=None,
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
            bias_out=None,
            transposed_out=False,
            out_in_sb=False,
            attention_dp=1,
            attention_dp_group=None,
            attention_dp_rank=0,
            kv_needs_a2a=False,
        )

        if K_new.dim() == 3:
            # One KV head: the kernel drops the head axis of K. Put it back.
            K_new = K_new.unsqueeze(2)
        if V_new.dim() == 3:
            V_new = V_new.unsqueeze(1)
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

        output = output.reshape(tokens, self.hidden_size)

        # >>> PARALLELISM: TP all-reduce after the megakernel <<<
        if self.world_size > 1:
            output = _all_reduce_fp32(self.tp_group, output)

        return output


# =============================================================================
# Section 4: MLP (Dense)
# =============================================================================


class Olmo2MLP(nn.Module):
    """Dense SwiGLU MLP with TP intermediate sharding.

    >>> PARALLELISM: TP <<<
    - gate_proj, up_proj: hidden -> intermediate/TP per rank
    - down_proj: intermediate/TP -> hidden per rank
    - Prefill: all-gather before -> compute -> reduce-scatter after (SP)
    - Decode: compute -> all-reduce

    <-- MODEL-SPECIFIC: down(silu(gate(x)) * up(x)), no bias.
    """

    def __init__(self, config: Olmo2Config):
        super().__init__()

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.hidden_size = config.hidden_size
        self.intermediate_size_per_rank = config.intermediate_size // self.world_size
        if config.intermediate_size % self.world_size != 0:
            raise ValueError(
                f"Tensor parallel size {self.world_size} does not divide "
                f"intermediate_size {config.intermediate_size}. "
                "Use a tensor parallel size that divides intermediate_size."
            )

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
        """>>> PARALLELISM: TP sharding of the MLP weights. <<<"""
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
        """>>> PARALLELISM: SP all-gather/reduce-scatter for prefill, all-reduce for decode <<<"""
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
                output = _all_reduce_fp32(self.tp_group, output)

        return output


# =============================================================================
# Section 5: Decoder Layer
# =============================================================================


class Olmo2DecoderLayer(nn.Module):
    """Single OLMo2 decoder layer.

    Architecture (MODEL-SPECIFIC, post-norm):
        h = h + post_attention_layernorm(attn(h))
        h = h + post_feedforward_layernorm(mlp(h))

    Parallelism (TP + SP):
        - Prefill input arrives in the SP layout (T/world_size tokens per rank).
        - The norms and the residual adds operate on the SP-local slice.
    """

    def __init__(self, config: Olmo2Config, layer_idx: int):
        super().__init__()
        self.self_attn = Olmo2Attention(config, layer_idx=layer_idx)
        self.mlp = Olmo2MLP(config)
        self.post_attention_layernorm = Olmo2RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_feedforward_layernorm = Olmo2RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
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
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(hidden_states, is_prefill=not is_decode)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Section 6: Model Backbone
# =============================================================================


class Olmo2Model(nn.Module):
    """OLMo2 transformer backbone.

    >>> PARALLELISM: SP (Sequence Parallelism) <<<
    During prefill:
    - After the embedding, each TP rank gets T/world_size tokens.
    - After all layers, an all-gather gives the full sequence.
    During decode:
    - No SP. All ranks process all tokens.
    """

    def __init__(self, config: Olmo2Config):
        super().__init__()
        self.config = config

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
                Olmo2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = Olmo2RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Olmo2RotaryEmbedding(config)

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

        # Keep cos/sin in FP32, as HF OLMo2 does. Prefill and decode both
        # apply RoPE in FP32.
        position_embeddings = self.rotary_emb(
            positions, device=hidden_states.device, dtype=torch.float32
        )

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
            )

        hidden_states = self.norm(hidden_states)

        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states


# =============================================================================
# Section 7: Language Model Head
# =============================================================================


def olmo2_checkpoint_mappings(num_layers: int) -> dict[str, str | list[str]]:
    """Map each OLMo2 model parameter to its HF checkpoint key or keys.

    The mapping covers every parameter. Parameters with the same name in the
    model and in the checkpoint also have an entry, so a test can compare the
    mapping against the checkpoint index.
    """
    mappings: dict[str, str | list[str]] = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
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
        mappings[f"{prefix}.self_attn.q_norm_weight"] = (
            f"{prefix}.self_attn.q_norm.weight"
        )
        mappings[f"{prefix}.self_attn.k_norm_weight"] = (
            f"{prefix}.self_attn.k_norm.weight"
        )
        mappings[f"{prefix}.post_attention_layernorm.weight"] = (
            f"{prefix}.post_attention_layernorm.weight"
        )
        mappings[f"{prefix}.post_feedforward_layernorm.weight"] = (
            f"{prefix}.post_feedforward_layernorm.weight"
        )
        mappings[f"{prefix}.mlp.gate_proj_weight"] = f"{prefix}.mlp.gate_proj.weight"
        mappings[f"{prefix}.mlp.up_proj_weight"] = f"{prefix}.mlp.up_proj.weight"
        mappings[f"{prefix}.mlp.down_proj_weight"] = f"{prefix}.mlp.down_proj.weight"
    return mappings


class Olmo2ForCausalLM(nn.Module):
    """OLMo2 model with the language modeling head.

    >>> PARALLELISM: Column-parallel linear for the LM head <<<
    <-- MODEL-SPECIFIC: Untied lm_head.
    """

    def __init__(self, config: Olmo2Config):
        super().__init__()
        self.config = config
        self.model = Olmo2Model(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

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
                "OLMo2 does not support input embeddings. Pass input_ids instead."
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
                f"({self.world_size}) and divisible by it for SP."
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
        config = Olmo2Config.from_configs(hf_config, neuron_config)
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
                    "Bind a KV cache for each layer."
                )
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # -- Weight Loading -------------------------------------------------------

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load the TP-sharded weights from the checkpoint.

        The checkpoint can store FP32 weights. Each weight is cast to the
        compute dtype after the shard load.
        """
        mappings = olmo2_checkpoint_mappings(len(self.model.layers))

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, mappings, device
        ).state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        missing = set(dict(self.named_parameters())) - set(rank_sharded)
        if missing:
            raise RuntimeError(
                f"Checkpoint load left {len(missing)} parameters without a value "
                f"(for example {sorted(missing)[:3]}). Check the checkpoint keys."
            )

        self.load_state_dict(rank_sharded, strict=False, assign=True)
