# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR composed model (BF16).

Top-level ``Qwen3ASRForConditionalGeneration``: audio encoder + Qwen3 dense text
decoder + (audio-placeholder) fusion.

ASR's decoder is standard dense Qwen3 (GQA 16Q/8KV, head_dim=128, QK-norm, SiLU,
standard 1D RoPE).

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    Qwen3-ASR-specific. Change when porting.
"""

from __future__ import annotations

import logging
import time as _time

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from nkilib.core.utils.common_types import NormType
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig, EncoderNeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.parallel.neuron_parallel_state import shard_select
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
    sharding_weight_loader_with_padding,
)

from ..kv_cache import build_per_head_slot_mapping, paged_slot_indices
from ..qwen3_vl.utils.decode_kv import build_per_head_block_table
from .audio_encoder_bf16 import Qwen3ASREncoder
from .config import Qwen3ASRConfig, Qwen3ASRTextConfig
from .utils.audio_preprocessing import plan_audio_blocks
from .utils.merge_audio_embeds import merge_audio_embeddings

logger = logging.getLogger(__name__)

# HF checkpoint prefixes. The thinker text decoder lives under `thinker.model.*`
# and its head under `thinker.lm_head.*`
HF_TEXT_PREFIX = "thinker.model"
HF_LM_HEAD_KEY = "thinker.lm_head.weight"


# =============================================================================
# Text decoder
# =============================================================================


class Qwen3ASRTextRMSNorm(nn.Module):
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


class Qwen3ASRTextRotaryEmbedding(nn.Module):
    """Standard RoPE for the Qwen3-ASR decoder (no Llama3-style scaling).

    <-- MODEL-SPECIFIC: rope_theta=1_000_000, no rope_scaling.
    """

    def __init__(self, config: Qwen3ASRTextConfig):
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


class Qwen3ASRTextAttention(nn.Module):
    """Multi-head attention with TP head sharding and per-head QK-norm.

    >>> PARALLELISM: TP + SP <<<
    - Q/K/V heads sharded across TP ranks
    - KV heads replicated when fewer than TP size (GQA)
    - Prefill: all-gather from SP → QKV (with fused input norm) → QK-norm →
      RoPE → attn → O proj → reduce-scatter to SP
    - Decode: fused megakernel (with fused input norm) → all-reduce (no SP)

    <-- MODEL-SPECIFIC:
    - GQA: 16 Q heads / 8 KV heads (Qwen3-ASR-1.7B)
    - Per-head QK-norm (head_dim RMSNorm) applied BEFORE RoPE
    - No attention bias, no sinks, no sliding window
    - Standard RoPE (rotate_half)
    """

    def __init__(self, config: Qwen3ASRTextConfig, layer_idx: int):
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

        # Pre-attention input RMSNorm
        self.input_layernorm = Qwen3ASRTextRMSNorm(
            self.hidden_size, config.rms_norm_eps, self.dtype
        )

        # <-- MODEL-SPECIFIC: Per-head QK-norm (on head_dim, not hidden_size)
        self.q_norm = Qwen3ASRTextRMSNorm(
            self.head_dim, config.rms_norm_eps, self.dtype
        )
        self.k_norm = Qwen3ASRTextRMSNorm(
            self.head_dim, config.rms_norm_eps, self.dtype
        )

        # KV caches bound externally
        self.k_cache = None
        self.v_cache = None
        # 3D views of the caches, for the nkh == 1 decode kernel. bind_kv_cache
        # makes them, because the decode graph must not make the view itself.
        self.k_cache_3d = None
        self.v_cache_3d = None

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
        """>>> PARALLELISM: Dispatch to prefill or decode. <<<

        This module owns ``input_layernorm`` and folds it into the QKV kernel on
        each path, so the caller must not normalize ``hidden_states`` first.
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states, positions, position_embeddings, attn_metadata
            )
        else:
            # >>> PARALLELISM: All-gather from SP <<<
            # The gather precedes the fused norm, so each rank normalizes every
            # token instead of its own shard. RMSNorm is per token, so the
            # result is the same.
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

        # Step 1: Fused input RMSNorm + QKV projection + pre-RoPE QK-norm + RoPE
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
            norm_type=NormType.RMS_NORM,
            norm_weights=self.input_layernorm.weight.view(1, -1),
            eps=self.input_layernorm.variance_epsilon,
            qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_eps=self.q_norm.variance_epsilon,
            qk_norm_pre_rope_q_gamma=self.q_norm.weight.view(1, -1),
            qk_norm_pre_rope_k_gamma=self.k_norm.weight.view(1, -1),
        ).squeeze(0)

        # TODO: check profile for transpose, QKV kernel can output NBSD directly
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

        # Fold the pre-attention RMSNorm into the megakernel. The kernel reads
        # gamma as [1, H]. Both call sites below share these 3 arguments.
        rmsnorm_X_gamma = self.input_layernorm.weight.view(1, -1)
        rmsnorm_X_eps = self.input_layernorm.variance_epsilon

        # GQA (nkh > 1, e.g. TP=4 on the 8-KV-head model) reaches the fused
        # decode megakernel only via a per-head 3D block table + the 4D cache
        # kept as-is; a 2D table routes GQA to the slow torch fallback.
        # In-kernel KV cache update (update_cache=True): the kernel writes the new
        # K/V into the paged cache, so no scatter runs after it and K/V never go to
        # HBM and back. The kernel then returns the attention output only.
        # The slots index the flat num_blocks * kv_heads pool, the same pool that
        # the per-head block table targets. The kernel reads uint32.
        kv_cache_update_idx = build_per_head_slot_mapping(
            slot_mapping, block_size, nkh, B
        ).to(torch.uint32)

        if nkh > 1:
            active_blocks_table = build_per_head_block_table(block_table, nkh).to(
                torch.int32
            )

            output = NF.attention_decode(
                X=X,
                X_hidden_dim_actual=self.hidden_size,
                rmsnorm_X_enabled=True,
                rmsnorm_X_eps=rmsnorm_X_eps,
                rmsnorm_X_gamma=rmsnorm_X_gamma,
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
                softmax_scale=self.scaling,
                sink=None,
                update_cache=True,
                kv_cache_update_idx=kv_cache_update_idx,
                W_out=self.o_proj_weight,
                bias_out=None,
                transposed_out=False,
                out_in_sb=False,
                attention_dp=1,
                attention_dp_group=None,
                attention_dp_rank=0,
                kv_needs_a2a=False,
            )

            if self.world_size > 1:
                self.tp_group.all_reduce(output)

            return output

        # nkh == 1 (TP >= num_kv_heads): the kernel takes a 3D
        # [num_blocks, block_len, d_head] cache + a 2D [B, num_blocks] table.
        k_cache = self.k_cache_3d if self.k_cache_3d is not None else self.k_cache
        v_cache = self.v_cache_3d if self.v_cache_3d is not None else self.v_cache

        active_blocks_table = block_table.to(torch.int32)

        # <-- MODEL-SPECIFIC: rmsnorm_QK_pre_rope_enabled=True for QK-norm;
        # must pass learned gamma weights so decode matches prefill.
        output = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            rmsnorm_X_enabled=True,
            rmsnorm_X_eps=rmsnorm_X_eps,
            rmsnorm_X_gamma=rmsnorm_X_gamma,
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
            softmax_scale=self.scaling,
            sink=None,
            update_cache=True,
            kv_cache_update_idx=kv_cache_update_idx,
            W_out=self.o_proj_weight,
            bias_out=None,
            transposed_out=False,
            out_in_sb=False,
            attention_dp=1,
            attention_dp_group=None,
            attention_dp_rank=0,
            kv_needs_a2a=False,
        )

        # >>> PARALLELISM: TP all-reduce <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        return output


class Qwen3ASRTextMLP(nn.Module):
    """Dense MLP with TP intermediate sharding.

    >>> PARALLELISM: TP + SP <<<
    - gate_proj, up_proj: hidden → intermediate/TP per rank
    - down_proj: intermediate/TP → hidden per rank
    - Prefill: all-gather from SP → compute → reduce-scatter to SP
    - Decode: compute → all-reduce

    <-- MODEL-SPECIFIC: SiLU activation (gate * silu(up)), no bias.
    """

    def __init__(self, config: Qwen3ASRTextConfig):
        super().__init__()

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        self.hidden_size = config.hidden_size
        # >>> PARALLELISM: Intermediate dim sharded across TP <<<
        self.intermediate_size_per_rank = config.intermediate_size // self.world_size

        # The post-attention RMSNorm
        self.post_attention_layernorm = Qwen3ASRTextRMSNorm(
            self.hidden_size, config.rms_norm_eps, config.torch_dtype
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

        self._setup_weight_loaders(config)

    def _setup_weight_loaders(self, config: Qwen3ASRTextConfig):
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        is_prefill: bool,
        *,
        fused_add_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """>>> PARALLELISM: SP all-gather/reduce-scatter for prefill, all-reduce for decode. <<<

        This module owns ``post_attention_layernorm``. The caller must not normalize first.
        Decode folds that norm into the kernel. ``fused_add_tensor`` also folds the
        attention residual in. The return is then ``(output, sum)``. The layer uses the
        sum as its next residual. Prefill norms before the gather, which keeps the norm on the
        local shard, and it takes no fused add, because the residual is on the
        shard while the kernel input is gathered.

        TODO: fuse both on prefill after the removal of sequence parallelism.
        """
        if is_prefill and fused_add_tensor is not None:
            raise ValueError(
                "The prefill path must not fuse the residual add. Give "
                "fused_add_tensor on decode only, and add the residual after "
                "this call."
            )

        if is_prefill:
            # Normalize on the SP shard, before the gather, then tell the kernel
            # that the activation is already normalized.
            hidden_states = self.post_attention_layernorm(hidden_states)
            if self.world_size > 1:
                hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
            norm_type = NormType.NO_NORM
            ln_w = None
        else:
            norm_type = NormType.RMS_NORM
            ln_w = self.post_attention_layernorm.weight.view(1, -1)

        # <-- MODEL-SPECIFIC: SiLU gated MLP
        result = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
            # Turn the column tiling off which introduces gate-up product transpose
            # before the down projection
            column_tiling=False,
            eps=self.post_attention_layernorm.variance_epsilon,
            ln_w=ln_w,
            norm_type=norm_type,
            fused_add_tensor=fused_add_tensor,
            store_fused_add_result=fused_add_tensor is not None,
        )
        if fused_add_tensor is not None:
            output, fused_add_result = result
        else:
            output, fused_add_result = result, None

        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                self.tp_group.all_reduce(output)

        if fused_add_result is not None:
            return output, fused_add_result
        return output


class Qwen3ASRTextDecoderLayer(nn.Module):
    """Single Qwen3-ASR decoder layer.

    Architecture: hidden → RMSNorm → Attention → residual → RMSNorm → MLP → residual

    ``self_attn`` owns the pre-attention norm and ``mlp`` owns the
    post-attention norm, so this layer calls neither one. Decode folds each norm
    into the kernel that follows it, and folds the attention residual add into
    the MLP kernel. Prefill folds the pre-attention norm only.
    """

    def __init__(self, config: Qwen3ASRTextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3ASRTextAttention(config, layer_idx=layer_idx)
        self.mlp = Qwen3ASRTextMLP(config)
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
        # Decode gives the attention residual to the MLP kernel, which adds it
        # before the norm and returns the sum. That removes 1 add operator and
        # the gap between the 2 kernels. The second add stays here.
        if is_decode:
            hidden_states, residual = self.mlp(
                hidden_states, is_prefill=False, fused_add_tensor=residual
            )
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp(hidden_states, is_prefill=True)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3ASRTextModel(nn.Module):
    """Qwen3-ASR text decoder backbone.

    >>> PARALLELISM: TP + sequence parallelism (SP) on the prefill residual. <<<
    Prefill shards the residual on dim 0. Decode is TP-only. This matches
    ``qwen3/model.py``. SP needs ``T % world_size == 0`` and ``T > world_size``
    on the prompt length.
    """

    def __init__(self, config: Qwen3ASRTextConfig):
        super().__init__()
        self.config = config

        # >>> PARALLELISM: TP group <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        # >>> PARALLELISM: Vocab-sharded embedding <<<
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )

        # <-- MODEL-SPECIFIC: 28 decoder layers (Qwen3-ASR-1.7B)
        self.layers = nn.ModuleList(
            [
                Qwen3ASRTextDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # <-- MODEL-SPECIFIC: Final RMSNorm
        self.norm = Qwen3ASRTextRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Qwen3ASRTextRotaryEmbedding(config)

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
        # <-- On-device audio encoder cache inputs (prefill-only).
        encoder_embedding_buffer: torch.Tensor | None = None,
        encoder_block_ids: torch.Tensor | None = None,
        encoder_positions: torch.Tensor | None = None,
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

        # Audio embedding merge (prefill only; decode has no audio inputs). Scatter
        # the encoder outputs from the on-device cache blocks into the audio
        # placeholder positions. The merge does the SP coordinate remap internally,
        # so it runs on the already-SP-sharded hidden_states.
        if (
            is_prefill
            and encoder_embedding_buffer is not None
            and encoder_block_ids is not None
            and encoder_positions is not None
        ):
            hidden_states = merge_audio_embeddings(
                hidden_states,
                encoder_embedding_buffer,
                encoder_block_ids,
                encoder_positions,
                rank=self.tp_group.rank_in_group_tensor,
            )

        if (
            is_prefill
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

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
            )

        hidden_states = self.norm(hidden_states)

        # >>> PARALLELISM: SP — all-gather to reconstruct full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states


# =============================================================================
# Composed model — audio encoder + text decoder + fusion + sampler.
# =============================================================================


class Qwen3ASRForConditionalGeneration(nn.Module):
    """Composed Qwen3-ASR model: audio encoder + Qwen3 text decoder.

    Decode-only today (see module docstring). ``SupportsEncoderWarmup`` and the
    audio-fusion path are added in the fusion/bucket follow-up commits.
    """

    # Bare flag, not the SupportsTranscription protocol: the upstream Qwen3-ASR
    # class owns the prompt methods, so this class only advertises the task.
    supports_transcription: bool = True
    supports_transcription_only: bool = False

    def __init__(self, config: Qwen3ASRConfig):
        super().__init__()

        # >>> PARALLELISM: EPD role flags (audio-encoder-only vs LM-only pools) <<<
        # Read the EPD role flags off the audio encoder's neuron config.
        # Both default False for a single-pool model.
        audio_neuron_config = (
            config.audio_config.neuron_config if config.audio_config else None
        )
        self.mm_encoder_only = bool(
            audio_neuron_config and audio_neuron_config.mm_encoder_only
        )
        self.mm_language_model_only = bool(
            audio_neuron_config and audio_neuron_config.mm_language_model_only
        )

        self.config = config
        self.text_config = config.text_config

        # <-- MODEL-SPECIFIC: audio tower. Built unless this is the LM-only EPD
        #     pool. dtype is governed by config.audio_config.torch_dtype (bf16).
        if config.audio_config is not None and not self.mm_language_model_only:
            self.audio_tower = Qwen3ASREncoder(config.audio_config)
        else:
            self.audio_tower = None

        if not self.mm_encoder_only:
            self.language_model = Qwen3ASRTextModel(config.text_config)
        else:
            self.language_model = None

        # >>> PARALLELISM: text TP group (lm_head + logits gather) <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        nc = config.text_config.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        debug_logits_enabled = nc is not None and nc.debug_logits_dir is not None
        self._gather_logits = (
            nc is not None and nc.max_logprobs != 0
        ) or debug_logits_enabled

        if not self.mm_encoder_only:
            # >>> PARALLELISM: vocab-dim sharded lm_head (column parallel) <<<
            self.lm_head = neuron_nn.ColumnParallelLinear(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                bias=False,
                dtype=config.text_config.torch_dtype,
                gather_output=not self.on_device_sampling_config,
                tp_group=self.tp_group.device_group,
                weight_layout="in_out",
            )
            transposed = self.lm_head.weight_layout == "in_out"
            set_weight_loader(
                self.lm_head.weight,
                sharding_weight_loader_with_padding(
                    shard_dim=1 if transposed else 0,
                    shard_size=self.text_config.vocab_size // self.world_size,
                    num_shards=self.world_size,
                    is_storage_transposed=transposed,
                    pad_dim=0 if transposed else 1,
                    padded_size=self.text_config.hidden_size,
                    unpadded_size=self.text_config.hidden_size,
                ),
            )

            if self.on_device_sampling_config is not None:
                self.sampler = Sampler(
                    self.on_device_sampling_config,
                    process_group=self.tp_group.device_group,
                )
            else:
                self.sampler = None
        else:
            self.lm_head = None
            self.sampler = None

    # ------------------------------------------------------------------
    # Compiler arguments (per-graph neuronx-cc policy)
    # ------------------------------------------------------------------

    def compiler_args_for(self, graph_type, example_inputs, base):
        """Give each graph its own neuronx-cc arguments.

        This is the model's per-graph policy. The runner resolves it one time
        per graph, before the cache hash. It passes the graph type, the example
        inputs, and the base argument list. Return the full argument list for
        that graph.

        ``base`` holds the arguments at the caller level. Derive the per-graph
        arguments from it. These arguments are a default, not an override. An
        explicit ``--optimization-level`` from the caller wins, so the runner
        does not call this policy.

        Qwen3-ASR compiles the decode graph and the audio-encoder graph at
        ``-O3``. ``-O3`` disables neuronx-cc modular flow. The ``-O1`` default
        enables it, which adds a wrapper on each collective. ``-O3`` improves
        the latency of these two graphs. Prefill keeps the base level. The
        runner tags the audio-encoder graph ``"encoder"``: one tag serves each
        modality, because every encoder shares one graph path.

        Example:
            >>> model.compiler_args_for("decode", None, ["-O1"])
            ['-O3']
            >>> model.compiler_args_for("prefill", None, ["-O1"])
            ['-O1']
        """
        from vllm_neuron.utils.compiler_args_ctx import with_optimization_level

        base = list(base or [])
        if graph_type in ("decode", "encoder"):
            return with_optimization_level(base, 3)
        return base

    def _scoped_compiler_args(self, graph_type):
        """Scope this graph's resolved arguments for an in-process compile.

        Read the base arguments that the runner set on the model. Delegate to
        :meth:`compiler_args_for`. Open a libtorch scope with the result. Open a
        null scope when the runner did not enable per-graph tuning, so the base
        arguments apply.

        Example:
            >>> with model._scoped_compiler_args("encoder"):
            ...     model.audio_tower(...)  # doctest: +SKIP
        """
        from libtorch_neuronx_lite.compile.compiler_args import scoped_compiler_args

        base = getattr(self, "_neuron_base_compiler_args", None)
        args = None if base is None else self.compiler_args_for(graph_type, None, base)
        return scoped_compiler_args(args)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig = None,
        encoder_neuron_config: EncoderNeuronConfig = None,
    ):
        """Build the composed model from a raw HF config + NeuronConfig(s).

        The keyword ``encoder_neuron_config`` is the runner seam
        (``neuron_model_runner.py`` calls
        ``from_configs(hf_config=, text_neuron_config=, encoder_neuron_config=)``),
        so we accept that literal name and forward it internally into the audio
        slot.
        """
        config = Qwen3ASRConfig.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            audio_neuron_config=encoder_neuron_config,
        )
        return cls(config)

    # ------------------------------------------------------------------
    # Forward (decode-only: text backbone -> sampling)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        # <-- On-device audio encoder cache inputs (prefill-only), injected by
        #     the runner as mm_kwargs.
        encoder_embedding_buffer: torch.Tensor | None = None,
        encoder_block_ids: torch.Tensor | None = None,
        encoder_positions: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Audio-prefill / text-decode step."""
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
                f"Prompt length {T} is too short for sequence parallelism (world_size={self.world_size}). "
                f"Use a prompt with length > {self.world_size} and divisible by {self.world_size}."
            )

        hidden_states = self.language_model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
            encoder_embedding_buffer=encoder_embedding_buffer,
            encoder_block_ids=encoder_block_ids,
            encoder_positions=encoder_positions,
        )

        hidden_states_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )
        logits = self.lm_head(hidden_states_for_logits)

        gathered_logits = None
        if self._gather_logits:
            gathered_logits = self.tp_group.all_gather(logits, dim=1)

        if self.on_device_sampling_config is None:
            return logits

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )

        return sampled_tokens, gathered_logits

    # ------------------------------------------------------------------
    # KV cache wiring (text layers)
    # ------------------------------------------------------------------

    def get_kv_spec(self) -> KVSpec:
        if self.language_model is None:
            return KVSpec(layers=[])
        layers = []
        for i, layer in enumerate(self.language_model.layers):
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

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
        if self.language_model is None:
            return
        for i, layer in enumerate(self.language_model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise KeyError(
                    f"KV cache for layer {layer_name} is not initialized. "
                    "Call bind_kv_cache() before the first forward pass."
                )
            k_cache = kv_caches[layer_name][0]
            v_cache = kv_caches[layer_name][1]
            layer.self_attn.k_cache = k_cache
            layer.self_attn.v_cache = v_cache
            # Prefill keeps the 4D cache, because its index_put_ gives a head
            # index. Decode at nkh == 1 needs a 3D cache. Make that view here,
            # one time, so the decode graph gets a plain input and the in-place
            # cache update needs no copy of the pool.
            if k_cache.dim() == 4 and k_cache.shape[1] == 1:
                layer.self_attn.k_cache_3d = k_cache.squeeze(1)
                layer.self_attn.v_cache_3d = v_cache.squeeze(1)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    @classmethod
    def build_text_weight_mappings(
        cls, num_hidden_layers: int, tie_word_embeddings: bool
    ) -> dict[str, str | list[str]]:
        """Composed-model param name -> HF checkpoint key(s) for the text decoder."""
        mappings: dict[str, str | list[str]] = {
            "language_model.embed_tokens.weight": f"{HF_TEXT_PREFIX}.embed_tokens.weight",
            "language_model.norm.weight": f"{HF_TEXT_PREFIX}.norm.weight",
        }

        # tie_word_embeddings=True for Qwen3-ASR-1.7B: lm_head reads the shared
        # embedding weight; otherwise a dedicated thinker.lm_head.weight.
        if tie_word_embeddings:
            mappings["lm_head.weight"] = f"{HF_TEXT_PREFIX}.embed_tokens.weight"
        else:
            mappings["lm_head.weight"] = HF_LM_HEAD_KEY

        for layer_id in range(num_hidden_layers):
            our = f"language_model.layers.{layer_id}"
            hf = f"{HF_TEXT_PREFIX}.layers.{layer_id}"

            # Fused QKV: list value triggers fused_qkv_weight_loader on the param.
            mappings[f"{our}.self_attn.qkv_proj_weight"] = [
                f"{hf}.self_attn.q_proj.weight",
                f"{hf}.self_attn.k_proj.weight",
                f"{hf}.self_attn.v_proj.weight",
            ]
            mappings[f"{our}.self_attn.o_proj_weight"] = f"{hf}.self_attn.o_proj.weight"

            # Per-head QK-norm (identity name; no TP shard).
            mappings[f"{our}.self_attn.q_norm.weight"] = f"{hf}.self_attn.q_norm.weight"
            mappings[f"{our}.self_attn.k_norm.weight"] = f"{hf}.self_attn.k_norm.weight"

            # The 2 layer norms live in the modules that fold them in, so our
            # names carry the ``self_attn`` and ``mlp`` prefix. The HF names stay
            # at the layer level.
            mappings[f"{our}.self_attn.input_layernorm.weight"] = (
                f"{hf}.input_layernorm.weight"
            )
            mappings[f"{our}.mlp.post_attention_layernorm.weight"] = (
                f"{hf}.post_attention_layernorm.weight"
            )

            mappings[f"{our}.mlp.gate_proj_weight"] = f"{hf}.mlp.gate_proj.weight"
            mappings[f"{our}.mlp.up_proj_weight"] = f"{hf}.mlp.up_proj.weight"
            mappings[f"{our}.mlp.down_proj_weight"] = f"{hf}.mlp.down_proj.weight"

        return mappings

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None = None
    ) -> None:
        """Load text-decoder weights (this class) + delegate audio-tower weights.

        Text: strip ``thinker.model.`` -> ``language_model.``, fuse q/k/v, tie
        lm_head to the shared embedding. Audio: delegate to the encoder's own
        loader (uses the encoder TP group, CPU-side).
        """
        if not self.mm_encoder_only:
            mappings = self.build_text_weight_mappings(
                len(self.language_model.layers),
                self.text_config.tie_word_embeddings,
            )

            checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
            rank_sharded = checkpoint.load_sharded_pipelined(
                self.rank,
                self.world_size,
                self,
                mappings,
                device,
                strict=False,
            ).state_dict

            # Cast any tensor not already the text dtype (matches VL / qwen3).
            for name, tensor in rank_sharded.items():
                if tensor.dtype != self.text_config.torch_dtype:
                    rank_sharded[name] = tensor.to(self.text_config.torch_dtype)

            self.load_state_dict(rank_sharded, strict=False, assign=True)

        # Audio tower loads its own thinker.audio_tower.* weights (encoder TP
        # group, CPU-side), independent of the text TP shard.
        if self.audio_tower is not None:
            self.audio_tower.load_weights(checkpoint_path, device="cpu", cpu_mode=True)

    # ------------------------------------------------------------------
    # Audio-fusion path
    # ------------------------------------------------------------------
    def embed_multimodal(
        self,
        input_audio_features: torch.Tensor | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        encoder_cache=None,
        mm_hashes: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Encode audio clips and allocate+write into the on-device cache.

        The runner groups mm_kwargs by modality, so one call carries only audio items.
        It plans the block-packed chunk layout, reserves cache blocks per clip, packs
        the mel into one whole-chunk stream, and runs the encoder in cache-write mode.
        The encoder scatter-writes into cache.buffer.

        Args:
            input_audio_features: [num_mel_bins, sum_F] flat mel, concatenated over
                clips along the frame axis (upstream flat_from_sizes, dim=1),
                unpadded and not chunk-aligned.
            audio_feature_lengths: [num_clips] valid mel frames per clip.
            feature_attention_mask: [num_clips, max_F] mel mask; the fallback source
                for the per-clip length (matches upstream _parse_and_validate_audio).
            encoder_cache: EncoderCacheBlocks (buffer, allocate, scratch_block_id).
            mm_hashes: one identifier per clip, in clip order.
        """
        from vllm_neuron.vllm.worker.encoder_cache_blocks import EncoderCacheBlocks

        if input_audio_features is None:
            raise ValueError(
                "embed_multimodal requires input_audio_features for the audio modality."
            )
        cache: EncoderCacheBlocks = encoder_cache
        ac = self.config.audio_config
        audio_neuron_config = ac.neuron_config
        t_start = _time.perf_counter()

        # 1. Per-clip valid mel lengths. Prefer explicit lengths; else the mask.
        if audio_feature_lengths is not None:
            feature_lens = [int(v) for v in audio_feature_lengths.flatten().tolist()]
        elif feature_attention_mask is not None:
            feature_lens = [int(v) for v in feature_attention_mask.sum(-1).tolist()]
        else:
            raise ValueError(
                "embed_multimodal needs audio_feature_lengths or "
                "feature_attention_mask to derive the per-clip mel length."
            )
        num_clips = len(feature_lens)
        if mm_hashes is None or len(mm_hashes) != num_clips:
            raise ValueError(
                f"embed_multimodal expects one mm_hash per clip: got "
                f"{0 if mm_hashes is None else len(mm_hashes)} hashes for "
                f"{num_clips} clips."
            )

        block_size = audio_neuron_config.encoder_attention_block_size
        dp_size = audio_neuron_config.dp_size

        # 2. Plan the block-packed layout and 1D-bucket it in one pass. The slot
        #    unit is one base chunk. plan packs segments (full windows and clip
        #    tails) into blocks, then picks the smallest bucket that fits. Buckets
        #    hold block counts. one_item_per_block is required by the cache (one
        #    mm_item per block run).
        buckets = audio_neuron_config.num_encoder_input_buckets
        if not buckets:
            raise ValueError(
                "num_encoder_input_buckets must be configured (non-empty) for audio."
            )
        plan = plan_audio_blocks(
            feature_lens,
            block_size=block_size,
            window_tokens=ac.window_tokens,
            n_window=ac.n_window,
            n_window_infer=ac.n_window_infer,
            buckets=buckets,
            one_item_per_block=True,
            dp_size=dp_size,
        )
        num_blocks = plan.num_blocks
        needed_blocks = plan.min_blocks
        slots_per_block = plan.slots_per_block

        # 3. Allocate cache blocks per clip. The block count follows the plan (one
        #    clip per block run), NOT dense output tokens, so the cache reserves
        #    the same blocks the encoder writes. Real tokens per block sum the
        #    clip's chunk lengths in that block and skip the pad tail.
        cache_block_map: list[list[int]] = []
        for i in range(num_clips):
            slots = plan.clip_slots[i]
            tokens_per_block = [
                sum(plan.chunk_lens[s] for s in slots[b : b + slots_per_block])
                for b in range(0, len(slots), slots_per_block)
            ]
            cache_block_map.append(cache.allocate(mm_hashes[i], tokens_per_block))

        # 4. CPU preprocessing off the plan.
        chunk_slots = plan.chunk_slots()
        bound_min, bound_max = plan.block_bounds()

        # 5. Pack the mel into a whole-chunk stream, then 1D-bucket it. Split the
        #    flat mel by valid lengths, pad each clip up to whole base chunks (1s
        #    granularity), and concat -> the compact real-chunk stream. Then pad
        #    up to num_slots chunks so F = num_slots * num_mel_frames_per_chunk: num_chunks ==
        #    num_slots is pinned by the bucketed num_blocks, so the conv and
        #    index_select shapes are static. The conv stays block-bucketed (no
        #    chunk-count compile dimension), so it also runs on the pad chunks; the
        #    bounds mask them and chunk_slots never references them.
        num_mel_frames_per_chunk = 2 * ac.n_window  # mel frames in one base chunk
        num_mel_bins = input_audio_features.shape[0]
        clip_mels = torch.split(input_audio_features, feature_lens, dim=1)
        parts = []
        for clip_mel, valid in zip(clip_mels, feature_lens):
            padded = -(-valid // num_mel_frames_per_chunk) * num_mel_frames_per_chunk
            buf = input_audio_features.new_zeros(num_mel_bins, padded)
            buf[:, :valid] = clip_mel
            parts.append(buf)
        input_features = torch.cat(parts, dim=1).unsqueeze(0)  # [1, num_mel_bins, F]
        target_F = (
            plan.num_slots * num_mel_frames_per_chunk
        )  # num_blocks * slots_per_block * 100
        if input_features.shape[-1] < target_F:
            input_features = torch.nn.functional.pad(
                input_features, (0, target_F - input_features.shape[-1])
            )

        # 6. write_block_ids: flatten the per-clip cache blocks in clip order; pad
        #    the trailing (bucket) blocks with the scratch block id.
        flat_cache_blocks = [b for blocks in cache_block_map for b in blocks]
        write_block_ids = torch.tensor(
            flat_cache_blocks
            + [cache.scratch_block_id] * (num_blocks - len(flat_cache_blocks)),
            dtype=torch.int64,
        )

        # 7. Move to device, dispatch the encoder in cache-write mode (input-output
        #    alias on cache.buffer). The last dim is output_dim (post-projector).
        #    Scope the "encoder" graph arguments, so the compiled audio-encoder
        #    graph hashes to the same arguments (-O3) as the capture and the warmup.
        device = next(self.audio_tower.parameters()).device
        with self._scoped_compiler_args("encoder"):
            self.audio_tower(
                input_features.to(dtype=ac.torch_dtype, device=device),
                bound_min.to(device),
                bound_max.to(device),
                chunk_slots.to(device),
                cache.buffer,
                write_block_ids.to(device),
            )
        logger.debug(
            "[embed_multimodal] %d clips, %d blocks (needed %d): %.1fms",
            num_clips,
            num_blocks,
            needed_blocks,
            (_time.perf_counter() - t_start) * 1000,
        )
