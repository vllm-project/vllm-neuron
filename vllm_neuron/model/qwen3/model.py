# SPDX-License-Identifier: Apache-2.0
"""
Qwen3 BF16 Implementation
=========================

Annotated implementation of Qwen3 for the Neuron backend.
Adapted from Llama (dense) template. Dense, GQA, QK-norm, SiLU, standard RoPE.

Supported parallelism: TP, SP, DP.
Reference model: Qwen/Qwen3-32B (dense, 64Q/8KV GQA, head_dim=128, no sliding window).

Key difference from Llama:
  - Per-head QK-norm (q_norm / k_norm) applied BEFORE RoPE.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    Qwen3-specific. Change when porting.
"""

import logging

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group
from transformers import PretrainedConfig
from vllm.model_executor.models.interfaces import SupportsEagle3

import vllm_neuron.functional as NF
from nkilib.core.utils.common_types import NormType
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.parallel.neuron_parallel_state import shard_select
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

import vllm_neuron.nn as neuron_nn
from vllm_neuron.vllm.spec_decode.decorator import async_speculative_decoding
from ..kv_cache import paged_slot_indices
from ..qwen3_vl.utils.decode_kv import build_per_head_block_table, scatter_new_kv
from .config import Qwen3Config

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1: RMS Normalization
# <-- MODEL-SPECIFIC: Qwen3 uses standard RMSNorm (no hidden dim padding).
#     QK-norm (per-head, dim=head_dim) is handled inside Attention.
# =============================================================================


class Qwen3RMSNorm(nn.Module):
    """RMS Normalization — standard, no padding."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


# =============================================================================
# Section 2: Rotary Position Embedding
# <-- MODEL-SPECIFIC: Qwen3 uses standard RoPE (rotate_half, no Llama3 scaling).
#     rope_theta = 1_000_000. QK-norm is applied BEFORE RoPE inside Attention.
# =============================================================================


class Qwen3RotaryEmbedding(nn.Module):
    """Standard RoPE for Qwen3 (no Llama3-style piecewise scaling).

    <-- MODEL-SPECIFIC: rope_theta=1_000_000, no rope_scaling.
    """

    def __init__(self, config: Qwen3Config):
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
        inv_freq = self._compute_inv_freq(device)  # [Hd/2]
        inv_freq_expanded = inv_freq[:, None].float()  # [Hd/2, 1]
        positions_expanded = position_ids[None, :].float()  # [1, T]
        freqs = (inv_freq_expanded @ positions_expanded).transpose(0, 1)  # [T, Hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, Hd]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


# =============================================================================
# Section 3: Attention
# Mixed: PARALLELISM (TP head sharding, SP, collectives) +
#        MODEL-SPECIFIC (GQA, QK-norm per head before RoPE, no bias, no sinks)
# =============================================================================


class Qwen3Attention(nn.Module):
    """Multi-head attention with TP head sharding and per-head QK-norm.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads sharded across TP ranks
    - KV heads replicated when fewer than TP size (GQA)
    - Prefill: all-gather → QKV → norm → RoPE → attn → O proj → reduce-scatter
    - Decode: fused megakernel

    <-- MODEL-SPECIFIC:
    - GQA: 64 Q heads / 8 KV heads (for 32B)
    - Per-head QK-norm (head_dim RMSNorm) applied BEFORE RoPE
    - No attention bias, no sinks, no sliding window
    - Standard RoPE (rotate_half)
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
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

        # >>> PARALLELISM: QKV weight shapes <<<
        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim
        qkv_size = q_size + 2 * kv_size
        o_proj_in_features = self.num_attention_heads * self.head_dim // self.world_size

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_proj_in_features, self.hidden_size, dtype=self.dtype)
        )

        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        # <-- MODEL-SPECIFIC: Per-head QK-norm (on head_dim, not hidden_size)
        self.q_norm = Qwen3RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)
        self.k_norm = Qwen3RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)

        # KV caches bound externally
        self.k_cache = None
        self.v_cache = None
        self.k_scale = None
        self.v_scale = None
        self.k_scale_float = 1.0
        self.v_scale_float = 1.0

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """>>> PARALLELISM: TP sharding of QKV and O-proj weights. <<<
        <-- MODEL-SPECIFIC: Separate Q, K, V (no bias). HF stores transposed.
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
                shard_size=self.num_attention_heads * self.head_dim // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    # ── Forward dispatch ──────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ):
        """>>> PARALLELISM: Dispatch to prefill or decode. <<<"""
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states, positions, position_embeddings, attn_metadata
            )
        else:
            # >>> PARALLELISM: All-gather from SP <<<
            if self.world_size > 1:
                hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
            return self.forward_prefill(
                hidden_states, positions, position_embeddings, attn_metadata
            )

    # ── Prefill path ──────────────────────────────────────────────────────

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # Step 1: Fused QKV projection + pre-RoPE QK-norm + RoPE
        cos, sin = position_embeddings
        cos_cache = cos.unsqueeze(0)  # [1, T, Dh]
        sin_cache = sin.unsqueeze(0)  # [1, T, Dh]

        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
            d_head=self.head_dim,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            num_q_heads=self.num_attention_heads_per_rank,
            num_kv_heads=self.num_key_value_heads_per_rank,
            qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_eps=self.q_norm.variance_epsilon,
            qk_norm_pre_rope_q_gamma=self.q_norm.weight.view(1, -1),
            qk_norm_pre_rope_k_gamma=self.k_norm.weight.view(1, -1),
        ).squeeze(0)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        # Reshape to per-head layout: [T, Nh, Dh] → [Nh, T, Dh]
        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )

        # Step 3: Update KV cache
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        block_indices, position_indices = paged_slot_indices(
            slot_mapping, block_size, self.k_cache.shape[0]
        )

        # FP8 KV cache quantize-on-write.
        # This path triggers only under kv_cache_dtype=fp8.
        if self.k_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            k_flat = (
                (k.reshape(-1, self.head_dim) * self.k_scale)
                .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                .to(self.k_cache.dtype)
            )
            v_flat = (
                (v.reshape(-1, self.head_dim) * self.v_scale)
                .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                .to(self.v_cache.dtype)
            )
        else:
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

        # Step 4: Attention — no sinks, no sliding window
        if kv_segment_size:
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

            q_flash = q.transpose(1, 2)
            k_flash = k.transpose(1, 2)
            v_flash = v

            attn_output = NF.flash_attention(
                q_flash,
                k_flash,
                v_flash,
                scale=self.scaling,
                tp_q=False,
                tp_out=True,
            )

        # Step 5: Output projection
        attn_output = attn_output.unsqueeze(0)
        attn_output = NF.o_proj(attn_output, self.o_proj_weight, None)
        attn_output = attn_output.squeeze(0)

        # >>> PARALLELISM: Reduce-scatter to SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    # ── Decode path ───────────────────────────────────────────────────────

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
        attention_mask: torch.Tensor | None = None,
    ):
        """>>> PARALLELISM: Fused megakernel with TP. <<<
        <-- MODEL-SPECIFIC: QK-norm via rmsnorm_QK_pre_rope_enabled=True.
        No sinks, no sliding window.

        Callers may supply a precomputed mask to reuse this path with custom
        attention semantics (e.g. DFlash bidirectional attention).
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        # Eagle3 makes S_decode 1 + num_speculative_tokens, not 1.
        assert tokens == B * S_decode, (
            f"decode token count ({tokens}) must be a multiple of batch size "
            f"({B}); got S_decode={S_decode}"
        )

        hidden_states = hidden_states.to(self.dtype)
        S_ctx = max_blocks_per_seq * block_size
        nkh = self.num_key_value_heads_per_rank

        X = hidden_states.view(B, S_decode, hidden)

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

        if attention_mask is None:
            # Positions are required to construct the causal attention mask.
            assert positions is not None
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

        # GQA (nkh > 1, e.g. TP=4 on the 8-KV-head 32B model) reaches the fused
        # decode megakernel only via a per-head 3D block table + the 4D cache
        # kept as-is; a 2D table routes GQA to the slow torch fallback.
        if nkh > 1:
            active_blocks_table = build_per_head_block_table(block_table, nkh).to(
                torch.int32
            )

            output, K_new, V_new = NF.attention_decode(
                X=X,
                X_hidden_dim_actual=self.hidden_size,
                # Input RMSNorm intentionally NOT fused here (#3803 regressed BF16
                # LongBench). See the decoder-layer forward.
                rmsnorm_X_enabled=False,
                W_qkv=self.qkv_proj_weight,
                bias_qkv=None,
                rmsnorm_QK_pre_rope_enabled=True,
                rmsnorm_QK_pre_rope_eps=self.q_norm.variance_epsilon,
                rmsnorm_QK_pre_rope_W_Q=self.q_norm.weight.view(1, -1),
                rmsnorm_QK_pre_rope_W_K=self.k_norm.weight.view(1, -1),
                rmsnorm_QK_post_rope_enabled=False,
                cos=cos_kernel,
                sin=sin_kernel,
                rope_contiguous_layout=True,
                K_cache_transposed=False,
                active_blocks_table=active_blocks_table,
                K_cache=self.k_cache,
                V_cache=self.v_cache,
                attention_mask=attention_mask,
                softmax_scale=self.scaling / self.k_scale_float,
                sink=None,
                update_cache=False,
                W_out=self.o_proj_weight / self.v_scale_float,
                bias_out=None,
                transposed_out=False,
                out_in_sb=False,
                k_scale=self.k_scale,
                v_scale=self.v_scale,
                attention_dp=1,
                attention_dp_group=None,
                attention_dp_rank=0,
                kv_needs_a2a=False,
            )

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

            if self.world_size > 1:
                self.tp_group.all_reduce(output)

            return output

        # nkh == 1 (TP >= num_kv_heads): squeeze the head axis for the kernel's
        # 3D [num_blocks, block_len, d_head] cache + 2D [B, num_blocks] table.
        k_cache = self.k_cache.squeeze(1) if self.k_cache.dim() == 4 else self.k_cache
        v_cache = self.v_cache.squeeze(1) if self.v_cache.dim() == 4 else self.v_cache

        active_blocks_table = block_table.to(torch.int32)

        # <-- MODEL-SPECIFIC: rmsnorm_QK_pre_rope_enabled=True for QK-norm;
        # must pass learned gamma weights so decode matches prefill.
        output, K_new, V_new = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            # Input RMSNorm intentionally NOT fused here (#3803 regressed BF16
            # LongBench). See the decoder-layer forward.
            rmsnorm_X_enabled=False,
            W_qkv=self.qkv_proj_weight,
            bias_qkv=None,
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=self.q_norm.variance_epsilon,
            rmsnorm_QK_pre_rope_W_Q=self.q_norm.weight.view(1, -1),
            rmsnorm_QK_pre_rope_W_K=self.k_norm.weight.view(1, -1),
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=active_blocks_table,
            K_cache=k_cache,
            V_cache=v_cache,
            attention_mask=attention_mask,
            softmax_scale=self.scaling / self.k_scale_float,
            sink=None,
            update_cache=False,
            W_out=self.o_proj_weight / self.v_scale_float,
            bias_out=None,
            transposed_out=False,
            out_in_sb=False,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
            attention_dp=1,
            attention_dp_group=None,
            attention_dp_rank=0,
            kv_needs_a2a=False,
        )

        # Manual KV cache update
        block_indices, position_indices = paged_slot_indices(
            slot_mapping, block_size, self.k_cache.shape[0]
        )
        num_tokens = slot_mapping.shape[0]

        k_new = (
            K_new.permute(1, 2, 0)
            .reshape(B, nkh, S_decode, self.head_dim)
            .transpose(0, 1)
            .reshape(nkh, B * S_decode, self.head_dim)
        )
        k_new_flat = k_new.reshape(-1, self.head_dim)
        v_new_flat = V_new.transpose(0, 1).reshape(-1, self.head_dim)

        head_indices_for_put = torch.arange(
            nkh, dtype=torch.long, device=hidden_states.device
        ).repeat_interleave(num_tokens)
        block_indices_for_put = block_indices.repeat(nkh)
        position_indices_for_put = position_indices.repeat(nkh)

        self.k_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            k_new_flat.to(self.k_cache.dtype),
        )
        self.v_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            v_new_flat.to(self.v_cache.dtype),
        )

        # >>> PARALLELISM: TP all-reduce <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 4: Dense MLP
# Mixed: PARALLELISM (TP intermediate sharding, SP collectives) +
#        MODEL-SPECIFIC (SiLU gate/up/down, no bias)
# =============================================================================


class Qwen3MLP(nn.Module):
    """Dense MLP with TP intermediate sharding.

    >>> PARALLELISM: TP <<<
    - gate_proj, up_proj: hidden → intermediate/TP per rank
    - down_proj: intermediate/TP → hidden per rank
    - Prefill: all-gather → compute → reduce-scatter (SP)
    - Decode: compute → all-reduce

    <-- MODEL-SPECIFIC: SiLU activation (gate * silu(up)), no bias.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        self.hidden_size = config.hidden_size
        # >>> PARALLELISM: Intermediate dim sharded across TP <<<
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

        self._setup_weight_loaders(config)

    def _setup_weight_loaders(self, config: Qwen3Config):
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
        """>>> PARALLELISM: SP all-gather/reduce-scatter for prefill, all-reduce for decode. <<<"""
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        # <-- MODEL-SPECIFIC: SiLU gated MLP
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
# <-- MODEL-SPECIFIC: pre-norm, residual connections identical to Llama
# =============================================================================


class Qwen3DecoderLayer(nn.Module):
    """Single Qwen3 transformer decoder layer.

    Architecture: hidden → RMSNorm → Attention → residual → RMSNorm → MLP → residual
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.self_attn = Qwen3Attention(config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
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
        # NOTE: Do NOT fuse this input RMSNorm into the decode megakernel on the
        # BF16 path. #3803 did that (passing gamma/eps so the megakernel folds
        # the norm into the QKV matmul) and it regressed the Qwen3-32B BF16
        # LongBench accuracy test (0.363 vs. the 0.378 gate): the fused norm
        # perturbs the decode logits enough to shift the generated answers.
        # Keep the standalone RMSNorm here. If you re-attempt the fusion for the
        # decode perf win, gate it on the BF16 LongBench test first. (The MXFP8
        # path in model_mxfp8.py keeps the fusion.)
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
# Mixed: PARALLELISM (SP embedding/chunking) + MODEL-SPECIFIC (layer stack, norm)
# =============================================================================


def _decode_sp_viable(tokens: int, world_size: int) -> bool:
    """Return True when ``tokens`` splits evenly over the tensor-parallel ranks.

    The decode token count is ``batch_bucket * s_decode``. Both are trace-time
    constants, so each compiled decode graph uses one branch. A count that does
    not divide keeps the all-reduce. A ragged shard needs a pad, and only the
    runner can add it.

    Example (Qwen3-32B, TP=8): 32 tokens give a 4-token shard and return True.
    4 tokens do not divide by 8, so the layer all-reduces.
    """

    return world_size > 1 and tokens % world_size == 0


class Qwen3Model(nn.Module):
    """Qwen3 transformer backbone.

    >>> PARALLELISM: SP during prefill — embed → chunk → layers → all-gather. <<<
    """

    def __init__(self, config: Qwen3Config):
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

        # <-- MODEL-SPECIFIC: 64 decoder layers
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # <-- MODEL-SPECIFIC: Final RMSNorm
        self.norm = Qwen3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Qwen3RotaryEmbedding(config)

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

        # Set by the runner via set_aux_hidden_state_layers().
        self.aux_hidden_state_layers = []

        # Sequence-parallel decode. The MXFP8 subclass turns this on.
        self.decode_sp_enabled = False

    def _sp_layout(self, is_prefill: bool, num_tokens: int, max_query_len: int) -> bool:
        """Return True when the layer stack works on a token shard: prefill
        always, decode per :meth:`_decode_sp_layout`."""
        if is_prefill:
            return True
        return self._decode_sp_layout(num_tokens, max_query_len)

    def _decode_sp_layout(self, num_tokens: int, max_query_len: int) -> bool:
        """Return True when a decode step works on a token shard:
        ``decode_sp_enabled`` is set, the token count divides by the
        tensor-parallel degree, and each sequence contributes one token
        (``max_query_len == 1``). Speculative decode (Eagle3) widens the query to
        ``1 + num_speculative_tokens`` and keeps the all-reduce path. A subclass
        that gates its layers further must override this to match, or the
        backbone shards what the layers treat as full."""
        return (
            self.decode_sp_enabled
            and max_query_len == 1
            and _decode_sp_viable(num_tokens, self.world_size)
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold
        # ``positions`` is never sharded, so both gates see the same token count.
        sp = self._sp_layout(is_prefill, positions.numel(), max_query_len)

        # >>> PARALLELISM: VocabDimShardedEmbedding handles SP internally; scatter_tokens reduce-scatters, half the bytes <<<
        hidden_states = self.embed_tokens(input_ids, scatter_tokens=sp, rank=rank)

        # merge_prompt_embeds has no shape check: an unsharded operand would broadcast.
        if (
            sp
            and self.world_size > 1
            and inputs_embeds is not None
            and is_token_ids is not None
        ):
            inputs_embeds = shard_select(inputs_embeds, 0, self.tp_group)
            is_token_ids = shard_select(is_token_ids, 0, self.tp_group)

        hidden_states = NF.merge_prompt_embeds(
            hidden_states, inputs_embeds, is_token_ids
        )

        position_embeddings = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )

        # Eagle3 aux states are the input to each configured layer, not its output.
        aux_hidden_states = []
        # MXFP8 a2a decode: a layer may return (mlp_partials, residual) as a carry.
        sp_partials = None
        for idx, decoder_layer in enumerate(self.layers):
            if idx in self.aux_hidden_state_layers:
                # The a2a carry cannot be open at an aux capture: mid-chain the
                # carrier is FLAT pre-shuffled while a reduce-scatter result is
                # natural order, so closing the carry here would add mismatched
                # layouts. Unreachable by construction — only Eagle3 sets aux
                # layers, and speculative decode keeps sequence-parallel decode
                # off (max_query_len > 1) — so assert instead of silently
                # mixing layouts.
                assert sp_partials is None, (
                    "sequence-parallel carry open at an Eagle3 aux-capture "
                    "layer: the flat pre-shuffled carrier cannot be closed "
                    "with a natural-order reduce-scatter. The speculative "
                    "query-width gate should have kept SP off."
                )
                aux_hidden_states.append(hidden_states)
            layer_kwargs = {}
            if sp_partials is not None:
                layer_kwargs["sp_partials"] = sp_partials
                sp_partials = None
            out = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
                **layer_kwargs,
            )
            if isinstance(out, tuple):
                sp_partials, hidden_states = out
            else:
                hidden_states = out
        # Close a carry left open by the last layer.
        if sp_partials is not None:
            hidden_states = hidden_states + self.tp_group.reduce_scatter(
                sp_partials, dim=0
            )

        hidden_states = self.norm(hidden_states)

        # >>> PARALLELISM: SP — mandatory all-gather to full sequence; Qwen3ForCausalLM skips decode index_select, a shard reads OOB <<<
        if sp and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
            # TODO: let the drafter take SP-partitioned aux states instead.
            aux_hidden_states = [
                self.tp_group.all_gather(aux, dim=0) for aux in aux_hidden_states
            ]

        return hidden_states, aux_hidden_states


# =============================================================================
# Section 7: LM Head + Weight Loading
# Mixed: PARALLELISM (column-parallel LM head) + MODEL-SPECIFIC (weight mappings)
# =============================================================================


@async_speculative_decoding
class Qwen3ForCausalLM(nn.Module, SupportsEagle3):
    """Qwen3 model with LM head.

    >>> PARALLELISM: Column-parallel LM head. <<<
    <-- MODEL-SPECIFIC: No tied embeddings (tie_word_embeddings=False).

    Implements ``SupportsEagle3``, so this model can be the *target* (verifier)
    for an Eagle3 draft head: the backbone collects aux hidden states at the
    configured layers and ``forward`` returns them concatenated. The draft is a
    separate ``Eagle3LlamaForCausalLM`` from the speculator checkpoint.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)

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

        # <-- MODEL-SPECIFIC: No tied embeddings — separate lm_head weight
        # >>> PARALLELISM: Column-parallel LM head <<<
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
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,  # @async_speculative_decoding injects async-spec args
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
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
                f"Prompt Length ({T}) must be > world_size ({self.world_size}) for SP."
            )

        hidden_states, aux_hidden_states = self.model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
        )

        # Standard one-token decode is row-aligned by construction. MRv2's
        # _prepare_model_input uses arange(row_bucket); MRv1 computes the
        # cumulative indices of one token per active request. MRv1 may append
        # non-identity suffix indices for fake bucket-padding requests, but
        # the sampling step ignores those output rows. Thus every observable
        # row already maps to the same hidden row. Skip the gather.
        if (
            not is_prefill
            and max_query_len == 1
            and spec_decode_metadata is None
            and hidden_states.shape[0] == sampling_positions.shape[0]
        ):
            hidden_states_for_logits = hidden_states
        else:
            hidden_states_for_logits = torch.index_select(
                hidden_states, dim=0, index=sampling_positions
            )

        logits = self.lm_head(hidden_states_for_logits)

        # ── No on-device sampling: return logits directly ─────────────────
        if self.on_device_sampling_config is None:
            if len(aux_hidden_states) > 0:
                # cat on-device so the NEFF emits one tensor, not a list.
                aux_hidden_states_concat = torch.cat(aux_hidden_states, dim=-1)
                return logits, aux_hidden_states_concat
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

        # ── Speculative decoding: rejection sampling ──────────────────────
        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            rejection_sampled_tokens = rejection_sampler(
                spec_decode_metadata, sampled_tokens
            )
            if len(aux_hidden_states) > 0:
                aux_hidden_states_concat = torch.cat(aux_hidden_states, dim=-1)
                return (
                    rejection_sampled_tokens,
                    aux_hidden_states_concat,
                    gathered_logits,
                )
            return rejection_sampled_tokens

        # ── Standard return (with optional Eagle3 aux states) ─────────────
        if len(aux_hidden_states) > 0:
            aux_hidden_states_concat = torch.cat(aux_hidden_states, dim=-1)
            return sampled_tokens, aux_hidden_states_concat, gathered_logits

        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = Qwen3Config.from_configs(hf_config, neuron_config)
        return cls(config)

    # ── Eagle3 Target Interface ───────────────────────────────────────────

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        if layers is not None:
            self.model.aux_hidden_state_layers = list(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        if self.model.aux_hidden_state_layers:
            return tuple(self.model.aux_hidden_state_layers)
        num_layers = len(self.model.layers)
        assert num_layers >= 3, (
            f"Eagle3 needs >= 3 target layers to source 3 aux hidden states, "
            f"got {num_layers}"
        )
        # Early/middle/late; falls back to evenly spaced if too few layers.
        preferred = (2, num_layers // 2, num_layers - 3)
        if len(set(preferred)) == 3 and all(0 <= i < num_layers for i in preferred):
            return preferred
        step = num_layers / 3
        return tuple(int(step * (k + 0.5)) for k in range(3))

    # ── KV Cache ──────────────────────────────────────────────────────────

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
        """Load weights from Qwen3 HF checkpoint.

        >>> PARALLELISM: Weight loaders handle TP sharding. <<<
        <-- MODEL-SPECIFIC: Qwen3 HF key → model parameter mappings.
        """
        tp_rank = self.rank
        tp_size = self.world_size

        mappings = {}

        # Embedding and LM head
        mappings["model.embed_tokens.weight"] = "model.embed_tokens.weight"
        mappings["lm_head.weight"] = "lm_head.weight"
        mappings["model.norm.weight"] = "model.norm.weight"

        for layer_id in range(len(self.model.layers)):
            prefix = f"model.layers.{layer_id}"

            # <-- MODEL-SPECIFIC: Separate Q, K, V → fused QKV
            mappings[f"{prefix}.self_attn.qkv_proj_weight"] = [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ]
            mappings[f"{prefix}.self_attn.o_proj_weight"] = (
                f"{prefix}.self_attn.o_proj.weight"
            )

            # <-- MODEL-SPECIFIC: QK-norm weights (no TP sharding — per-head, not per-rank)
            mappings[f"{prefix}.self_attn.q_norm.weight"] = (
                f"{prefix}.self_attn.q_norm.weight"
            )
            mappings[f"{prefix}.self_attn.k_norm.weight"] = (
                f"{prefix}.self_attn.k_norm.weight"
            )

            # <-- MODEL-SPECIFIC: Layer norm names
            mappings[f"{prefix}.input_layernorm.weight"] = (
                f"{prefix}.input_layernorm.weight"
            )
            mappings[f"{prefix}.post_attention_layernorm.weight"] = (
                f"{prefix}.post_attention_layernorm.weight"
            )

            # <-- MODEL-SPECIFIC: Dense MLP weight names
            mappings[f"{prefix}.mlp.gate_proj_weight"] = (
                f"{prefix}.mlp.gate_proj.weight"
            )
            mappings[f"{prefix}.mlp.up_proj_weight"] = f"{prefix}.mlp.up_proj.weight"
            mappings[f"{prefix}.mlp.down_proj_weight"] = (
                f"{prefix}.mlp.down_proj.weight"
            )

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, mappings, device
        ).state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self._load_kv_cache_scales(checkpoint, device)
        self.load_state_dict(rank_sharded, strict=False, assign=True)

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Lightweight weight loading used during CPU compile.

        Only loads KV cache scales so the scale-derived compile-time constant
        ``softmax_scale = self.scaling / self.k_scale_float`` is baked correctly
        instead of the 1.0 default. Full weights are not needed for tracing.
        """
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        # load_weights() indexes via load_sharded_pipelined; the lite path skips
        # that, so index explicitly or the scale lookups silently miss (-> 1.0).
        checkpoint._ensure_indexed()
        self._load_kv_cache_scales(checkpoint, device)

    def _load_kv_cache_scales(
        self, checkpoint: SafetensorsCheckpoint, device: torch.device
    ):
        from vllm_neuron.utils.dtype_utils import QUANTIZED_KV_CACHE_DTYPES
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()

        for layer_id in range(len(self.model.layers)):
            attn = self.model.layers[layer_id].self_attn

            if vllm_config.cache_config.cache_dtype not in QUANTIZED_KV_CACHE_DTYPES:
                continue

            for scale_name in ("k_scale", "v_scale"):
                key = f"model.layers.{layer_id}.self_attn.{scale_name}"
                if key in checkpoint._tensor_name_to_file:
                    val = 1.0 / checkpoint._get_slice(key)[:].to(
                        dtype=torch.bfloat16, device=device
                    )
                else:
                    val = torch.ones(1, dtype=torch.bfloat16, device=device)
                setattr(attn, scale_name, val.reshape(1, 1))

            attn.k_scale_float = attn.k_scale.item()
            attn.v_scale_float = attn.v_scale.item()
