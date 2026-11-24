# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 (text) BF16 Implementation
=================================

Implementation of the Gemma3 text decoder for the native Neuron backend.
Adapted from the Qwen3 dense template (GQA, per-head QK-norm before RoPE).

Supported parallelism: TP, SP (prefill).
Reference checkpoint: EssentialAI/rnj-1-instruct (32 layers, 32Q/8KV GQA,
head_dim 128, full attention in all layers, YaRN RoPE, tied embeddings).

Differences from Qwen3:
  - RMSNorm multiplies by ``(1 + weight)`` in fp32. This applies to the four
    norms of each layer, the final norm, and the per-head QK-norm.
  - Each decoder layer has four norms. ``post_attention_layernorm`` and
    ``post_feedforward_layernorm`` apply to the sublayer output, before the
    residual add.
  - The embedding output is multiplied by ``sqrt(hidden_size)``.
  - The MLP activation is ``gelu_pytorch_tanh``.
  - The attention scale is ``query_pre_attn_scalar ** -0.5``.
  - YaRN RoPE: cos and sin include the YaRN attention factor.
  - The logits get a final soft cap: ``tanh(logits / cap) * cap``.
  - The LM head is tied to the embedding.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code (same as Qwen3).
  # <-- MODEL-SPECIFIC: ...    Gemma3-specific.
"""

import logging
import math

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from nkilib.core.utils.common_types import ActFnType, NormType
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec, paged_slot_indices
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

from .compat import build_per_head_block_table, scatter_new_kv
from .config import Gemma3Config

logger = logging.getLogger(__name__)

# <-- MODEL-SPECIFIC: HF "gelu_pytorch_tanh" is the tanh approximation of GELU.
GEMMA3_MLP_ACT_FN = ActFnType.GELU_Tanh_Approx


# =============================================================================
# Section 1: RMS Normalization
# <-- MODEL-SPECIFIC: Gemma3 multiplies by (1 + weight) in fp32, then casts.
# =============================================================================


def gemma3_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply the Gemma3 RMSNorm: ``norm(x.float()) * (1 + w.float())``."""
    out = x.float()
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
    out = out * (1.0 + weight.float())
    return out.to(x.dtype)


def gemma3_norm_gamma(weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Return the effective gamma ``1 + w`` in the layout of the NKI kernels."""
    return (1.0 + weight.float()).to(dtype).view(1, -1)


class Gemma3RMSNorm(nn.Module):
    """Gemma3 RMSNorm. The weight is stored as in the checkpoint (zero-centered)."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return gemma3_rms_norm(hidden_states, self.weight, self.variance_epsilon)


# =============================================================================
# Section 2: Rotary Position Embedding (YaRN)
# <-- MODEL-SPECIFIC: Same math as transformers _compute_yarn_parameters.
# =============================================================================


def compute_rope_inv_freq(config: Gemma3Config) -> tuple[torch.Tensor, float]:
    """Return the fp32 inverse frequencies and the cos/sin scale factor.

    For ``rope_type == "yarn"`` the result is the same as transformers
    ``_compute_yarn_parameters``. For ``"default"`` the scale factor is 1.0.
    """
    dim = config.head_dim
    base = config.rope_theta
    # Build the table on the CPU. The runner builds the model in a meta device
    # context, and a meta tensor has no values to read.
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float, device="cpu") / dim)
    if config.rope_type == "default":
        return 1.0 / pos_freqs, 1.0

    factor = config.rope_factor
    original_max = config.rope_original_max_position_embeddings
    if factor is None:
        factor = config.max_position_embeddings / original_max

    def get_mscale(scale: float, mscale: float = 1.0) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    attention_factor = config.rope_attention_factor
    if attention_factor is None:
        if config.rope_mscale and config.rope_mscale_all_dim:
            attention_factor = float(
                get_mscale(factor, config.rope_mscale)
                / get_mscale(factor, config.rope_mscale_all_dim)
            )
        else:
            attention_factor = get_mscale(factor)

    beta_fast = config.rope_beta_fast or 32
    beta_slow = config.rope_beta_slow or 1

    def find_correction_dim(num_rotations: float) -> float:
        return (dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = find_correction_dim(beta_fast)
    high = find_correction_dim(beta_slow)
    if config.rope_truncate:
        low = math.floor(low)
        high = math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001  # Prevent a division by zero.
    ramp = torch.clamp(
        (torch.arange(dim // 2, dtype=torch.float32, device="cpu") - low)
        / (high - low),
        0,
        1,
    )
    extrapolation_factor = 1 - ramp

    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    inv_freq = (
        inv_freq_interpolation * (1 - extrapolation_factor)
        + inv_freq_extrapolation * extrapolation_factor
    )
    return inv_freq, float(attention_factor)


class Gemma3RotaryEmbedding(nn.Module):
    """RoPE table for the full-attention layers (YaRN for rnj-1).

    <-- MODEL-SPECIFIC: cos and sin include the YaRN attention factor, as in
    HF ``Gemma3RotaryEmbedding``.
    """

    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.head_dim = config.head_dim
        inv_freq, attention_factor = compute_rope_inv_freq(config)
        # compute_rope_inv_freq builds on the CPU, so the buffer has real values
        # in the meta device context of the runner. The runner moves the model
        # to the device after the weight load, as for the GPT-OSS YaRN buffer.
        # Do not build the table from a Python list in the forward pass. That
        # puts a constant tensor into the compiled graph.
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = attention_factor

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[:, None].float()  # [Hd/2, 1]
        positions_expanded = position_ids[None, :].float()  # [1, T]
        freqs = (inv_freq_expanded @ positions_expanded).transpose(0, 1)  # [T, Hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, Hd]
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


# =============================================================================
# Section 3: Attention
# Mixed: PARALLELISM (TP head sharding, SP, collectives) +
#        MODEL-SPECIFIC (QK-norm gamma 1 + w, scale query_pre_attn_scalar^-0.5)
# =============================================================================


class Gemma3Attention(nn.Module):
    """Multi-head attention with TP head sharding and per-head QK-norm.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads sharded across TP ranks
    - KV heads replicated when fewer than TP size (GQA)
    - Prefill: all-gather -> QKV kernel -> attention kernel -> O proj -> reduce-scatter
    - Decode: fused attention block kernel -> all-reduce

    <-- MODEL-SPECIFIC:
    - Per-head QK-norm before RoPE with gamma ``1 + w``
    - Scale ``query_pre_attn_scalar ** -0.5``
    - No attention bias, no sinks, no sliding window, no attention soft cap
    """

    def __init__(self, config: Gemma3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        # <-- MODEL-SPECIFIC
        self.scaling = config.attn_scaling

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

        # <-- MODEL-SPECIFIC: Per-head QK-norm (zero-centered weight, gamma 1 + w)
        self.q_norm = Gemma3RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)
        self.k_norm = Gemma3RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)

        # KV caches bound externally
        self.k_cache = None
        self.v_cache = None
        self.k_scale = None
        self.v_scale = None
        self.k_scale_float = 1.0
        self.v_scale_float = 1.0

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """>>> PARALLELISM: TP sharding of QKV and O-proj weights. <<<"""
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

    def _qk_gammas(self) -> tuple[torch.Tensor, torch.Tensor]:
        # <-- MODEL-SPECIFIC: the kernels multiply by gamma; Gemma3 uses 1 + w.
        return (
            gemma3_norm_gamma(self.q_norm.weight, self.dtype),
            gemma3_norm_gamma(self.k_norm.weight, self.dtype),
        )

    # -- Forward dispatch ------------------------------------------------------

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
        # >>> PARALLELISM: All-gather from SP <<<
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(
            hidden_states, positions, position_embeddings, attn_metadata
        )

    # -- Prefill path ----------------------------------------------------------

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

        # Step 1: Fused QKV projection + pre-RoPE QK-norm + RoPE (NKI kernel)
        cos, sin = position_embeddings
        cos_cache = cos.unsqueeze(0)  # [1, T, Dh]
        sin_cache = sin.unsqueeze(0)  # [1, T, Dh]
        q_gamma, k_gamma = self._qk_gammas()

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
            qk_norm_pre_rope_q_gamma=q_gamma,
            qk_norm_pre_rope_k_gamma=k_gamma,
        ).squeeze(0)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        # Reshape to per-head layout: [T, Nh, Dh] -> [Nh, T, Dh]
        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )

        # Step 2: Update KV cache
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        block_indices, position_indices = paged_slot_indices(
            slot_mapping, block_size, self.k_cache.shape[0]
        )

        # FP8 KV cache quantize-on-write (kv_cache_dtype=fp8 only).
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

        # Step 3: Causal attention (NKI kernel). Full attention in all layers.
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

            attn_output = NF.flash_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v,
                scale=self.scaling,
                tp_q=False,
                tp_out=True,
            )

        # Step 4: Output projection (NKI kernel)
        attn_output = attn_output.unsqueeze(0)
        attn_output = NF.o_proj(attn_output, self.o_proj_weight, None)
        attn_output = attn_output.squeeze(0)

        # >>> PARALLELISM: Reduce-scatter to SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    # -- Decode path -----------------------------------------------------------

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
    ):
        """>>> PARALLELISM: Fused attention block kernel with TP. <<<
        <-- MODEL-SPECIFIC: QK-norm via rmsnorm_QK_pre_rope_enabled with gamma 1 + w.
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
            f"Decode token count ({tokens}) is not a multiple of the batch size "
            f"({B}). Make the decode token count a multiple of the batch size."
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

        q_gamma, k_gamma = self._qk_gammas()

        if nkh > 1:
            # GQA with more than one KV head per rank: per-head 3D block table
            # and the 4D cache, as in Qwen3.
            active_blocks_table = build_per_head_block_table(block_table, nkh).to(
                torch.int32
            )
            k_cache = self.k_cache
            v_cache = self.v_cache
        else:
            # nkh == 1 (TP >= num_kv_heads): squeeze the head axis for the 3D
            # [num_blocks, block_len, d_head] cache and the 2D [B, num_blocks] table.
            k_cache = (
                self.k_cache.squeeze(1) if self.k_cache.dim() == 4 else self.k_cache
            )
            v_cache = (
                self.v_cache.squeeze(1) if self.v_cache.dim() == 4 else self.v_cache
            )
            active_blocks_table = block_table.to(torch.int32)

        output, K_new, V_new = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            # The input RMSNorm is a Gemma3 norm (1 + w), so keep it outside.
            rmsnorm_X_enabled=False,
            W_qkv=self.qkv_proj_weight,
            bias_qkv=None,
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=self.q_norm.variance_epsilon,
            rmsnorm_QK_pre_rope_W_Q=q_gamma,
            rmsnorm_QK_pre_rope_W_K=k_gamma,
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

        if nkh > 1:
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
        else:
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
#        MODEL-SPECIFIC (gelu_pytorch_tanh gate, no bias)
# =============================================================================


class Gemma3MLP(nn.Module):
    """Dense MLP with TP intermediate sharding.

    >>> PARALLELISM: TP <<<
    - Prefill: all-gather -> NKI MLP kernel -> reduce-scatter (SP)
    - Decode: NKI MLP kernel -> all-reduce

    <-- MODEL-SPECIFIC: down(gelu_tanh(gate(x)) * up(x)), no bias.
    """

    def __init__(self, config: Gemma3Config):
        super().__init__()

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

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
        """>>> PARALLELISM: SP all-gather/reduce-scatter for prefill, all-reduce for decode. <<<"""
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        # <-- MODEL-SPECIFIC: GELU (tanh approximation) gated MLP
        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
            act_fn=GEMMA3_MLP_ACT_FN,
        )

        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 5: Decoder Layer
# <-- MODEL-SPECIFIC: four norms; the post norms apply before the residual add.
# =============================================================================


class Gemma3DecoderLayer(nn.Module):
    """Single Gemma3 transformer decoder layer.

    x = x + post_attention_layernorm(attn(input_layernorm(x)))
    x = x + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)))
    """

    def __init__(self, config: Gemma3Config, layer_idx: int):
        super().__init__()
        eps = config.rms_norm_eps
        dtype = config.torch_dtype
        self.input_layernorm = Gemma3RMSNorm(config.hidden_size, eps, dtype)
        self.post_attention_layernorm = Gemma3RMSNorm(config.hidden_size, eps, dtype)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps, dtype)
        self.post_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps, dtype)
        self.self_attn = Gemma3Attention(config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config)
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
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=not is_decode)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Section 6: Model Backbone
# Mixed: PARALLELISM (SP embedding/chunking) + MODEL-SPECIFIC (embed scale)
# =============================================================================


class Gemma3Model(nn.Module):
    """Gemma3 text backbone.

    >>> PARALLELISM: SP during prefill -- embed -> chunk -> layers -> all-gather. <<<
    """

    def __init__(self, config: Gemma3Config):
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
        # <-- MODEL-SPECIFIC: HF casts sqrt(hidden_size) to the weight dtype.
        # Round the scale here once and keep a Python float. The product with
        # a rounded scalar is the same as the HF product with a scalar tensor.
        self.embed_scale = float(
            torch.tensor(config.embed_scale, dtype=config.torch_dtype, device="cpu")
        )

        self.layers = nn.ModuleList(
            [
                Gemma3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = Gemma3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Gemma3RotaryEmbedding(config)

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
        # <-- MODEL-SPECIFIC: scale the token embeddings only. HF does not scale
        # caller-supplied prompt embeddings.
        hidden_states = hidden_states * self.embed_scale

        # merge_prompt_embeds has no shape check: an unsharded operand would broadcast.
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

        # >>> PARALLELISM: SP -- all-gather to the full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states


# =============================================================================
# Section 7: LM Head + Weight Loading
# =============================================================================


def apply_final_logit_softcap(
    logits: torch.Tensor, softcap: float | None
) -> torch.Tensor:
    """Return ``tanh(logits / softcap) * softcap`` in fp32, cast to the logits dtype.

    The operation is elementwise, so it is correct on vocab-sharded logits.
    """
    if not softcap:
        return logits
    capped = torch.tanh(logits.float() / softcap) * softcap
    return capped.to(logits.dtype)


def build_weight_mappings(
    num_layers: int, tie_word_embeddings: bool
) -> dict[str, str | list[str]]:
    """Return the map from model parameter names to HF checkpoint keys."""
    mappings: dict[str, str | list[str]] = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.norm.weight": "model.norm.weight",
        # <-- MODEL-SPECIFIC: tied embeddings load lm_head from embed_tokens.
        "lm_head.weight": (
            "model.embed_tokens.weight" if tie_word_embeddings else "lm_head.weight"
        ),
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
        for norm in ("q_norm", "k_norm"):
            mappings[f"{prefix}.self_attn.{norm}.weight"] = (
                f"{prefix}.self_attn.{norm}.weight"
            )
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            mappings[f"{prefix}.{norm}.weight"] = f"{prefix}.{norm}.weight"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            mappings[f"{prefix}.mlp.{proj}_weight"] = f"{prefix}.mlp.{proj}.weight"
    return mappings


class Gemma3ForCausalLM(nn.Module):
    """Gemma3 text model with a column-parallel, tied LM head.

    >>> PARALLELISM: Column-parallel LM head. <<<
    <-- MODEL-SPECIFIC: tied embeddings and a final logit soft cap.
    """

    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        self.model = Gemma3Model(config)

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

        # >>> PARALLELISM: Column-parallel LM head <<<
        # With tied embeddings the weight is a distinct vocab-sharded parameter
        # that load_weights fills from the embed_tokens checkpoint key.
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
                f"Prompt length ({T}) is not a multiple of world_size "
                f"({self.world_size}) larger than world_size. Use a prefill bucket "
                "that is a multiple of the tensor-parallel size."
            )

        hidden_states = self.model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
        )

        # One-token decode is row-aligned by construction (see Qwen3). Skip the gather.
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
        # <-- MODEL-SPECIFIC: final logit soft cap before sampling and gather.
        logits = apply_final_logit_softcap(logits, self.config.final_logit_softcapping)

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
    def from_configs(cls, hf_config: object, neuron_config: NeuronConfig | None):
        config = Gemma3Config.from_configs(hf_config, neuron_config)
        return cls(config)

    # -- KV Cache --------------------------------------------------------------

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
                raise KeyError(
                    f"KV cache for layer {layer_name} is not set. Bind the KV "
                    "cache of each layer before the forward pass."
                )
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # -- Weight Loading --------------------------------------------------------

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load and shard the HF Gemma3 checkpoint, then cast to the served dtype."""
        mappings = build_weight_mappings(
            len(self.model.layers), self.config.tie_word_embeddings
        )

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, mappings, device
        ).state_dict

        # The checkpoint is float32. Cast each tensor to the served dtype.
        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        missing = sorted(
            name for name, _ in self.named_parameters() if name not in rank_sharded
        )
        if missing:
            raise RuntimeError(
                f"The checkpoint at {checkpoint_path} has no tensor for "
                f"{len(missing)} parameters, for example {missing[:4]}. Make sure "
                "that the checkpoint is a complete Gemma3 text checkpoint."
            )

        self._load_kv_cache_scales(checkpoint, device)
        self.load_state_dict(rank_sharded, strict=False, assign=True)

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load only the KV cache scales for a CPU compile (no full weights)."""
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        checkpoint._ensure_indexed()
        self._load_kv_cache_scales(checkpoint, device)

    def _load_kv_cache_scales(
        self, checkpoint: SafetensorsCheckpoint, device: torch.device
    ):
        from vllm.config import get_current_vllm_config

        from vllm_neuron.utils.dtype_utils import QUANTIZED_KV_CACHE_DTYPES

        vllm_config = get_current_vllm_config()
        if vllm_config.cache_config.cache_dtype not in QUANTIZED_KV_CACHE_DTYPES:
            return

        for layer_id in range(len(self.model.layers)):
            attn = self.model.layers[layer_id].self_attn
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
