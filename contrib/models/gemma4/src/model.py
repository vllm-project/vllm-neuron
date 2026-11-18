# SPDX-License-Identifier: Apache-2.0
"""
Gemma 4 31B text decoder in BF16 for vLLM Neuron.

Layers repeat five sliding-window layers and one global layer:
  - sliding: head_dim 256, 16 KV heads, 1024-token window, RoPE theta 1e4
  - global: head_dim 512, 4 KV heads, V = K (no v_proj in the checkpoint), partial RoPE
    (25% of the dims are rotated), RoPE theta 1e6

Other features of the architecture:
  - QK normalization (RMSNorm) after the projection; V normalization (RMSNorm without scale)
  - layer_scalar per layer (learned multiplicative factor)
  - scaling = 1.0 (no 1/sqrt(head_dim) query scaling)
  - GeGLU activation (gelu_pytorch_tanh), scaled embeddings (times sqrt(hidden_size))
  - four norms per layer (input, post_attn, pre_ffn, post_ffn)
  - final_logit_softcapping = 30.0, tie_word_embeddings = True

Both head dims exceed the 128 limit of the NKI attention kernels, so attention is PyTorch over the
gathered paged blocks. The KV cache is accessed through the kernels in kv_write.py, on one flat tensor
per shared buffer (see README.md in this folder for why).

Validated on one trn2.3xlarge with TP=4. Prefill uses sequence parallelism (all-gather before and
reduce-scatter after each block).
"""

import logging
import math

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec, paged_slot_indices
from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from nkilib.core.utils.common_types import ActFnType, NormType

from transformers import PretrainedConfig
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.sampler import Sampler

import vllm_neuron.nn as nxdi_nn
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding

from .config import Gemma4Config
from .kv_write import gather_kv, write_kv

logger = logging.getLogger(__name__)



# =============================================================================
# Section 1: RMS Normalization
# Gemma4 uses standard RMSNorm (weight * normed) not (1+weight) * normed.
# =============================================================================


class Gemma4RMSNorm(nn.Module):
    """Standard RMSNorm with weight scaling."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Gemma4VNorm(nn.Module):
    """RMSNorm WITHOUT learnable scale (applied to V states in attention)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(input_dtype)


# =============================================================================
# Section 2: Rotary Position Embedding
# Gemma4 has per-layer RoPE: SWA uses standard theta=10000, full rotation.
# Global uses theta=1000000, partial_rotary_factor=0.25.
# =============================================================================


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Interleaved RoPE (rotate_half style)."""
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
    """Apply rotary position embeddings to query and key tensors.

    For proportional RoPE (global layers), cos/sin span the full head_dim.
    Non-rotary positions have cos=1, sin=0, so they pass through unchanged.

    Args:
        q: [Nh, T, Dh], k: [Nkv, T, Dh]
        cos: [T, head_dim], sin: [T, head_dim]

    Returns:
        Rotated (q, k) tensors
    """
    cos = cos.unsqueeze(0)  # [1, T, head_dim]
    sin = sin.unsqueeze(0)  # [1, T, head_dim]
    return _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)


class Gemma4RotaryEmbedding(nn.Module):
    """Per-layer Rotary Position Embedding (proportional RoPE).

    Each layer has its own rope_theta and head_dim. For global layers with
    partial_rotary_factor=0.25, only 25% of the head dimensions are rotated,
    but the cos/sin tensors span the full head_dim (with cos=1, sin=0 for
    non-rotary positions).

    This matches the HuggingFace "proportional" RoPE implementation:
      - inv_freq denominator is always head_dim (not rotary_dim)
      - Non-rotary dimensions get zero-frequency entries (cos=1, sin=0)
      - rotate_half is applied to the full head_dim vector
    """

    def __init__(self, head_dim: int, rope_theta: float, partial_rotary_factor: float):
        super().__init__()
        self.rope_theta = rope_theta
        self.head_dim = head_dim
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_angles = int(partial_rotary_factor * head_dim // 2)
        self.nope_angles = head_dim // 2 - self.rope_angles

    def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
        inv_freq_rotated = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, 2 * self.rope_angles, 2, dtype=torch.float, device=device)
                / self.head_dim
            )
        )
        if self.nope_angles > 0:
            inv_freq = torch.cat([
                inv_freq_rotated,
                torch.zeros(self.nope_angles, dtype=torch.float, device=device),
            ])
        else:
            inv_freq = inv_freq_rotated
        return inv_freq

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin embeddings for given positions.

        Returns:
            cos, sin: both of shape [T, head_dim]
            Non-rotary positions have cos=1, sin=0 (pass-through).
        """
        inv_freq = self._compute_inv_freq(device)  # [head_dim//2]
        inv_freq_expanded = inv_freq[:, None]  # [head_dim//2, 1]
        positions_expanded = position_ids[None, :].float()  # [1, T]
        freqs = (inv_freq_expanded @ positions_expanded).transpose(0, 1)  # [T, head_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, head_dim]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


# =============================================================================
# Section 3: Attention
# Heterogeneous: each layer has its own head_dim, KV heads, RoPE config.
# QK norm + V norm applied after projection.
# Scaling = 1.0 (no 1/sqrt(head_dim) applied in attention).
# =============================================================================


class Gemma4Attention(nn.Module):
    """Multi-head attention with per-layer heterogeneous dimensions.

    Key features:
      - head_dim varies per layer (256 for SWA, 512 for global)
      - KV head count varies per layer (16 for SWA, 4 for global)
      - QK normalization via RMSNorm after projection
      - V normalization (RMSNorm without learnable scale)
      - Partial RoPE for global layers (factor=0.25)
      - scaling=1.0 (passed to attention kernels)
    """

    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads

        # Per-layer dimensions
        self.head_dim = config.get_layer_head_dim(layer_idx)
        self.num_key_value_heads = config.get_layer_num_kv_heads(layer_idx)
        self.is_global = config.is_global_layer(layer_idx)
        self.k_eq_v = self.is_global and config.attention_k_eq_v

        # Gemma 4 uses scaling = 1.0 (no 1/sqrt(head_dim) factor on the query).
        self.scaling = 1.0

        # TP group setup
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # Head sharding
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

        # QKV weight shapes for TP
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

        # QK norms (with learnable weight)
        self.q_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)
        self.k_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)

        # V norm (without learnable weight)
        self.v_norm = Gemma4VNorm(self.head_dim, config.rms_norm_eps)

        # Sliding window: only for SWA layers, global layers use full context
        self.sliding_window = config.sliding_window if not self.is_global else None

        # Per-layer RoPE
        rope_theta = config.get_layer_rope_theta(layer_idx)
        partial_rotary_factor = config.get_layer_partial_rotary_factor(layer_idx)
        self.rotary_emb = Gemma4RotaryEmbedding(
            self.head_dim, rope_theta, partial_rotary_factor
        )

        # KV caches bound externally
        # Bound by Gemma4ForCausalLM.bind_kv_cache: one flat tensor per shared KV buffer.
        self.kv_flat = None
        self.kv_dims = None  # (n_blocks, heads, block_size, head_dim) of this layer
        self.cache_dtype = None
        self.cache_num_blocks = None

        # FP8 KV cache quantization scales (set during weight loading if enabled)
        self.register_buffer("k_scale", None, persistent=False)
        self.register_buffer("v_scale", None, persistent=False)
        self.k_scale_float = 1.0
        self.v_scale_float = 1.0

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
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

    def _apply_qk_norm(self, q, k):
        """Apply per-head QK normalization.

        q: [Nh, T, Dh], k: [Nkv, T, Dh]
        QK-norm weights are stored in f32 to force the Neuron compiler
        to keep the computation in f32 (prevents bf16 precision loss).
        """
        q = self.q_norm(q)
        k = self.k_norm(k)
        return q, k

    def _apply_partial_rotary(self, q, k, cos, sin):
        """Apply rotary embedding (proportional RoPE).

        For both SWA and global layers, cos/sin now span the full head_dim.
        For global layers with partial_rotary_factor < 1.0, non-rotary
        positions in cos/sin are 1/0 respectively, so those dimensions
        pass through unchanged when rotate_half is applied to the full vector.
        """
        return apply_rotary_pos_emb(q, k, cos, sin)

    def _manual_sdpa(self, q, k, v, attn_mask):
        """Attention in float32 with PyTorch matmuls.

        The plugin's NKI attention kernels accept head_dim up to 128 (MAX_HEAD_DIM in
        functional/attention/attention_cte.py); this model uses 256 and 512.

        q, k, v: [Nh, T, Dh] (k and v already expanded for GQA)
        attn_mask: [T, T] with -inf for masked positions, 0 for valid
        Returns: [Nh, T, Dh]
        """
        scores = torch.bmm(q.float(), k.float().transpose(1, 2))
        scores = scores * self.scaling
        scores = scores + attn_mask.float()
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        out = torch.bmm(attn_weights, v.float())
        return out.to(q.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attn_metadata: object | None = None,
    ):
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
        else:
            if self.world_size > 1:
                hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

            return self.forward_prefill(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
            )

    # -- Prefill path --

    def _kv_views(self):
        """K and V ``[blocks, heads, block_size, head_dim]`` views derived from the flat cache (read-only use)."""
        n, h, bs, d = self.kv_dims
        half = n * h * bs * d
        return (
            self.kv_flat[:half].view(n, h, bs, d),
            self.kv_flat[half:].view(n, h, bs, d),
        )

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # Step 1: QKV projection (torch matmul)
        qkv = torch.matmul(hidden_states, self.qkv_proj_weight)

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

        # Step 2: QK Normalization (before RoPE)
        q, k = self._apply_qk_norm(q, k)

        # Step 3: V Normalization (operates on dim=-1, works directly on 3D)
        v = self.v_norm(v)

        # Step 4: Apply RoPE (per-layer, possibly partial)
        cos, sin = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )
        q, k = self._apply_partial_rotary(q, k, cos, sin)

        # Step 5: Update KV Cache
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        block_indices, position_indices = paged_slot_indices(
            slot_mapping, block_size, self.cache_num_blocks
        )

        # Quantize K/V before writing to FP8 cache: fp8(clamp(tensor * scale))
        if self.cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            k_flat = (
                (k.reshape(-1, self.head_dim) * self.k_scale)
                .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                .to(self.cache_dtype)
            )
            v_flat = (
                (v.reshape(-1, self.head_dim) * self.v_scale)
                .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                .to(self.cache_dtype)
            )
        else:
            k_flat = k.reshape(-1, self.head_dim).to(self.cache_dtype)
            v_flat = v.reshape(-1, self.head_dim).to(self.cache_dtype)

        write_kv(
            self.kv_flat, k_flat, v_flat, block_indices, position_indices,
            self.num_key_value_heads_per_rank, self.kv_dims,
        )
        k_cache_r, v_cache_r = self._kv_views()

        # Step 6: Attention
        # Segmented (chunked) prefill: gate on the INT kv_segment_size ONLY
        # (compile-time constant per NEFF bucket). cached_seq_len is a TENSOR and
        # must NOT appear in the Python branch (Dynamo data-dependent branching,
        # which is what blocked chunked prefill >16K). NF.segmented_attention
        # routes head_dim>128 (gemma4 is 256/512) to its trace-safe static-shape
        # path, handling GQA + sliding window + causal over the paged KV cache.
        if kv_segment_size:
            attn_output = NF.segmented_attention(
                q,
                k_cache=k_cache_r,
                v_cache=v_cache_r,
                block_tables=block_table,
                prior_tokens=cached_seq_len,
                block_size=block_size,
                kv_segment_size=kv_segment_size,
                scale=self.scaling,
                tp_q=True,
                tp_out=False,
                sliding_window=self.sliding_window,
                sink=None,
            )
        else:
            # Full prefill: expand KV for GQA + torch SDPA (head_dim 256/512
            # exceeds the NKI flash-attention limit of 128).
            nkv_heads = k.shape[0]
            k = (
                k.unsqueeze(1)
                .expand(nkv_heads, self.num_key_value_groups, tokens, self.head_dim)
                .reshape(self.num_attention_heads_per_rank, tokens, self.head_dim)
            )
            v = (
                v.unsqueeze(1)
                .expand(nkv_heads, self.num_key_value_groups, tokens, self.head_dim)
                .reshape(self.num_attention_heads_per_rank, tokens, self.head_dim)
            )

            # q, k, v: [Nh, T, Dh]; the attention matmuls expect [..., S, Dh]
            if self.sliding_window is not None:
                # Build sliding window causal mask: attend only within window
                row_idx = torch.arange(tokens, device=hidden_states.device).unsqueeze(1)
                col_idx = torch.arange(tokens, device=hidden_states.device).unsqueeze(0)
                causal = col_idx <= row_idx
                in_window = (row_idx - col_idx) < self.sliding_window
                mask = causal & in_window  # [T, T]
                attn_mask = torch.where(
                    mask, torch.zeros(1, dtype=self.dtype, device=hidden_states.device),
                    torch.full((1,), float("-inf"), dtype=self.dtype, device=hidden_states.device),
                )
                attn_output = self._manual_sdpa(q, k, v, attn_mask)
            else:
                # Build causal mask explicitly
                row_idx = torch.arange(tokens, device=hidden_states.device).unsqueeze(1)
                col_idx = torch.arange(tokens, device=hidden_states.device).unsqueeze(0)
                causal_mask = torch.where(
                    col_idx <= row_idx,
                    torch.zeros(1, dtype=self.dtype, device=hidden_states.device),
                    torch.full((1,), float("-inf"), dtype=self.dtype, device=hidden_states.device),
                )
                attn_output = self._manual_sdpa(q, k, v, causal_mask)
        # attn_output: [Nh, T, Dh]

        # Reshape to [T, H] for output projection
        attn_output = attn_output.transpose(0, 1).contiguous().view(
            tokens, self.num_attention_heads_per_rank * self.head_dim
        )

        # Step 7: Output Projection
        attn_output = torch.matmul(attn_output, self.o_proj_weight)

        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output

    # -- Decode path --

    def _grouped_decode_attention(
        self, q, k, v, positions, swa_kv_pos_offset, B, S_decode, S_ctx
    ):
        """Decode attention without materializing GQA-expanded K/V.

        q: [Nh, B*S_decode, Dh]; k, v: [B, Nkv, S_ctx, Dh] (cache dtype, bf16).
        Math stays in fp32 like the expanded path (Gemma 4 uses scaling=1.0 with
        QK-norm, so scores are large and bf16 scores would change the softmax),
        but K/V are converted once per KV head instead of once per query head.
        Returns [B*S_decode, Nh*Dh].
        """
        nkh, groups, dh = self.num_key_value_heads_per_rank, self.num_key_value_groups, self.head_dim
        # [Nh, B*S, Dh] -> [B, Nkv, G*S, Dh]; heads are ordered kv-major (h = kv*G + g).
        qg = (
            q.view(nkh, groups, B, S_decode, dh)
            .permute(2, 0, 1, 3, 4)
            .reshape(B, nkh, groups * S_decode, dh)
        )
        scores = torch.matmul(qg.float(), k.float().transpose(-2, -1)) * self.scaling

        pos = positions.view(B, 1, S_decode)
        ctx_pos = torch.arange(S_ctx, device=positions.device).view(1, S_ctx, 1)
        if swa_kv_pos_offset is not None:
            ctx_pos = ctx_pos + swa_kv_pos_offset.view(B, 1, 1)
        keep = ctx_pos.transpose(1, 2) <= pos.transpose(1, 2)  # [B, S_decode, S_ctx]
        if self.sliding_window is not None:
            keep = keep & (ctx_pos.transpose(1, 2) > pos.transpose(1, 2) - self.sliding_window)
        keep = keep.unsqueeze(1).repeat(1, 1, groups, 1).view(B, 1, groups * S_decode, S_ctx)
        scores = scores.masked_fill(~keep, float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v.float()).to(q.dtype)  # [B, Nkv, G*S, Dh]
        return (
            out.view(B, nkh, groups, S_decode, dh)
            .permute(0, 3, 1, 2, 4)
            .reshape(B * S_decode, nkh * groups * dh)
        )

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attn_metadata: object,
    ):
        """Decode attention in PyTorch (head_dim 256 and 512 are above the 128 that the plugin's
        NKI attention kernels accept): QKV projection, QK norm, V norm, RoPE, KV write and
        block gather through the kernels in kv_write.py, then attention over the gathered blocks.
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        swa_kv_pos_offset = attn_metadata[layer_name].get("swa_kv_pos_offset")

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        assert tokens == B * S_decode

        hidden_states = hidden_states.to(self.dtype)
        nkh = self.num_key_value_heads_per_rank

        # Step 1: QKV projection (torch matmul)
        qkv = torch.matmul(hidden_states, self.qkv_proj_weight)
        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(
            0, 1
        )  # [Nh, T, Dh]
        k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )  # [Nkv, T, Dh]
        v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )  # [Nkv, T, Dh]

        # Step 2: QK Normalization
        q, k = self._apply_qk_norm(q, k)

        # Step 3: V Normalization (operates on dim=-1, works directly on 3D)
        v = self.v_norm(v)

        # Step 4: RoPE (per-layer)
        cos, sin = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )
        q, k = self._apply_partial_rotary(q, k, cos, sin)

        # Step 5: KV cache update
        block_indices, position_indices = paged_slot_indices(
            slot_mapping, block_size, self.cache_num_blocks
        )

        # Quantize K/V before writing to FP8 cache
        if self.cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            k_flat = (
                (k.reshape(-1, self.head_dim) * self.k_scale)
                .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                .to(self.cache_dtype)
            )
            v_flat = (
                (v.reshape(-1, self.head_dim) * self.v_scale)
                .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                .to(self.cache_dtype)
            )
        else:
            k_flat = k.reshape(-1, self.head_dim).to(self.cache_dtype)
            v_flat = v.reshape(-1, self.head_dim).to(self.cache_dtype)

        # Write the new rows, then gather the blocks this step attends to. Both go through kernels on
        # the flat cache; the token orders the gather after the write.
        token = write_kv(
            self.kv_flat, k_flat, v_flat, block_indices, position_indices, nkh, self.kv_dims
        )
        S_ctx = max_blocks_per_seq * block_size
        # Clamp -1 (sentinel for unallocated blocks) to 0; masked positions are ignored by the attention mask.
        block_ids = torch.clamp(block_table.reshape(-1), min=0)
        k_blocks, v_blocks = gather_kv(self.kv_flat, block_ids, self.kv_dims, after=token)
        # Reshape: [B, max_blocks, nkh, block_size, head_dim] -> [B, nkh, S_ctx, head_dim]
        k_gathered = (
            k_blocks.view(B, max_blocks_per_seq, nkh, block_size, self.head_dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(B, nkh, S_ctx, self.head_dim)
        )
        v_gathered = (
            v_blocks.view(B, max_blocks_per_seq, nkh, block_size, self.head_dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(B, nkh, S_ctx, self.head_dim)
        )

        # Dequantize FP8 cache values back to compute dtype for attention
        if self.cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            k_gathered = k_gathered.to(self.dtype) / self.k_scale_float
            v_gathered = v_gathered.to(self.dtype) / self.v_scale_float

        attn_output = self._grouped_decode_attention(
            q, k_gathered, v_gathered, positions, swa_kv_pos_offset, B, S_decode, S_ctx
        )
        output = torch.matmul(attn_output, self.o_proj_weight)
        if self.world_size > 1:
            self.tp_group.all_reduce(output)
        return output


# =============================================================================
# Section 4: MLP (GeGLU, gelu_pytorch_tanh)
# =============================================================================


_MLP_PSUM_BANKS = 8
_MLP_I_TILE = 512


def _num_intermediate_chunks(intermediate_size: int) -> int:
    """Fewest equal, 128-aligned chunks of I that each fit the CTE MLP kernel."""
    for n in range(1, intermediate_size // 128 + 1):
        size, rem = divmod(intermediate_size, n)
        if rem == 0 and size % 128 == 0 and math.ceil(size / _MLP_I_TILE) <= _MLP_PSUM_BANKS:
            return n
    return 1


class Gemma4MLP(nn.Module):
    """Dense MLP with TP intermediate sharding.

    GeGLU: down_proj(gelu_tanh(gate_proj(x)) * up_proj(x)).
    No bias on any projection.
    """

    def __init__(self, config: Gemma4Config):
        super().__init__()

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.hidden_size = config.hidden_size
        self.intermediate_size_per_rank = config.intermediate_size // self.world_size
        self.prefill_mlp_chunks = _num_intermediate_chunks(self.intermediate_size_per_rank)

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

    def _setup_weight_loaders(self, config):
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
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 1e-6,
    ) -> torch.Tensor:
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        norm_kwargs = {}
        if norm_weight is not None:
            norm_kwargs = dict(
                eps=norm_eps, ln_w=norm_weight.unsqueeze(0), norm_type=NormType.RMS_NORM
            )

        # Prefill runs the CTE MLP kernel, whose intermediate dim must fit in the
        # PSUM banks (ceil(I / 512) <= 8). At TP=4 the per-rank I is 5376, so split
        # I into chunks that fit and sum the partial down projections, which is
        # exact because GeGLU is elementwise over I. Otherwise NF.mlp falls back
        # to torch GELU, which the Neuron compile path cannot trace.
        chunks = self.prefill_mlp_chunks if is_prefill else 1
        if chunks == 1:
            output = NF.mlp(
                hidden_states,
                self.gate_proj_weight,
                self.up_proj_weight,
                self.down_proj_weight,
                act_fn=ActFnType.GELU_Tanh_Approx,
                **norm_kwargs,
            )
        else:
            step = self.intermediate_size_per_rank // chunks
            output = None
            for c in range(chunks):
                sl = slice(c * step, (c + 1) * step)
                part = NF.mlp(
                    hidden_states,
                    self.gate_proj_weight[:, sl],
                    self.up_proj_weight[:, sl],
                    self.down_proj_weight[sl, :],
                    act_fn=ActFnType.GELU_Tanh_Approx,
                    **norm_kwargs,
                )
                output = part if output is None else output + part

        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 5: Decoder Layer
# 4 norms per layer + layer_scalar at the end.
# =============================================================================


class Gemma4DecoderLayer(nn.Module):
    """Single decoder layer with 4 norms and per-layer scalar.

    Architecture:
        hidden -> input_layernorm -> Attention -> post_attention_layernorm -> residual
               -> pre_feedforward_layernorm -> MLP -> post_feedforward_layernorm -> residual
               -> multiply by layer_scalar
    """

    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        self.input_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.self_attn = Gemma4Attention(config, layer_idx=layer_idx)
        self.mlp = Gemma4MLP(config)
        self.layer_idx = layer_idx

        # Per-layer learned scalar (multiplied at end of forward)
        self.layer_scalar = nn.Parameter(
            torch.ones(1, dtype=config.torch_dtype), requires_grad=False
        )

        # ---- Per-layer input embeddings (PLE). Not used by the 31B checkpoint, not tested. ----
        # If hidden_size_per_layer_input is set, this layer carries the
        # per-layer modulator: gate(hidden) * per_layer_input -> projection
        # -> norm, then added to hidden_states before layer_scalar.
        self.hidden_size_per_layer_input = getattr(
            config, "hidden_size_per_layer_input", 0
        ) or 0
        if self.hidden_size_per_layer_input > 0:
            tp_dg = get_tp_group().device_group
            # hidden_size (2560) -> hidden_size_per_layer_input (256)
            self.per_layer_input_gate = nxdi_nn.ColumnParallelLinear(
                config.hidden_size,
                self.hidden_size_per_layer_input,
                bias=False,
                gather_output=True,
                dtype=config.torch_dtype,
                tp_group=tp_dg,
            )
            # 256 -> hidden_size (2560)
            self.per_layer_projection = nxdi_nn.RowParallelLinear(
                self.hidden_size_per_layer_input,
                config.hidden_size,
                bias=False,
                input_is_parallel=False,
                dtype=config.torch_dtype,
                tp_group=tp_dg,
            )
            # Post-PLE norm (over hidden_size).
            self.post_per_layer_input_norm = Gemma4RMSNorm(
                config.hidden_size,
                config.rms_norm_eps,
                config.torch_dtype,
            )
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None

    def _is_decode(self, attn_metadata) -> bool:
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
        return max_query_len <= decode_token_threshold

    def _fused_norm_residual(
        self, norm: Gemma4RMSNorm, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        """Fused: residual + norm(hidden_states). Single memory pass."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + norm.variance_epsilon)
        return residual + (norm.weight * hidden_states.to(input_dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attn_metadata: object | None = None,
        per_layer_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_decode = self._is_decode(attn_metadata)

        # Attention with pre/post norms (post-norm + residual fused)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
        )
        hidden_states = self._fused_norm_residual(
            self.post_attention_layernorm, hidden_states, residual
        )

        # MLP with fused pre-norm + post norm + residual fused
        residual = hidden_states
        hidden_states = self.mlp(
            hidden_states,
            is_prefill=not is_decode,
            norm_weight=self.pre_feedforward_layernorm.weight,
            norm_eps=self.pre_feedforward_layernorm.variance_epsilon,
        )
        hidden_states = self._fused_norm_residual(
            self.post_feedforward_layernorm, hidden_states, residual
        )

        # ---- Per-layer input injection (PLE; not used by the 31B checkpoint) ----
        # Applied before the per-layer scalar.
        if (
            per_layer_input is not None
            and self.per_layer_input_gate is not None
        ):
            gate = self.per_layer_input_gate(hidden_states)
            # tanh-approx GeLU inlined (Dynamo can't trace
            # torch._C._nn.gelu in this stack).
            gate = 0.5 * gate * (
                1.0
                + torch.tanh(
                    0.7978845608028654
                    * (gate + 0.044715 * gate * gate * gate)
                )
            )
            gated = gate * per_layer_input
            contribution = self.per_layer_projection(gated)
            contribution = self.post_per_layer_input_norm(contribution)
            hidden_states = hidden_states + contribution

        # Per-layer scalar
        hidden_states = hidden_states * self.layer_scalar

        return hidden_states


# =============================================================================
# Section 6: Model Backbone
# Scaled embeddings (sqrt(hidden_size)), heterogeneous layers.
# =============================================================================


class Gemma4Model(nn.Module):
    """Gemma4 transformer backbone."""

    def __init__(self, config: Gemma4Config):
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

        # Gemma4: scale embedding output by sqrt(hidden_size).
        # The reference (HuggingFace) stores this as a bf16 buffer, causing
        # truncation (e.g., sqrt(5376)=73.32 -> bf16: 73.5). We replicate the
        # bf16 rounding on CPU to avoid .item() on meta tensors during init.
        self.embed_scale = torch.tensor(
            config.hidden_size**0.5, dtype=config.torch_dtype, device="cpu"
        ).float().item()

        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )

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

        # ---- Per-layer input embeddings (PLE) ----
        # If hidden_size_per_layer_input is set, the model carries a parallel
        # "per-layer input" stream: each token is also embedded into a per-layer
        # space and projected through a per-layer modulator inside each decoder
        # layer. The 31B checkpoint sets it to 0 or None, so this path is not
        # used by it and is not tested.
        self.hidden_size_per_layer_input = getattr(
            config, "hidden_size_per_layer_input", 0
        ) or 0
        if self.hidden_size_per_layer_input > 0:
            num_layers = config.num_hidden_layers
            ple_total_dim = num_layers * self.hidden_size_per_layer_input
            # PLE vocab embedding can be a smaller vocab than main embed.
            self.vocab_size_per_layer_input = getattr(
                config, "vocab_size_per_layer_input", config.vocab_size
            )
            # Vocab embedding for the per-layer stream.
            # Output shape per token: [num_layers * per_layer_dim].
            self.embed_tokens_per_layer = VocabDimShardedEmbedding(
                vocab_size=self.vocab_size_per_layer_input,
                embed_dim=ple_total_dim,
                dtype=config.torch_dtype,
                tp_group=self.tp_group.device_group,
            )
            # Scaled embedding factor (sqrt of the per-layer dim), stored as
            # a plain float (not buffer) to avoid meta-tensor .to() issues.
            self.embed_scale_per_layer = float(
                self.hidden_size_per_layer_input**0.5
            )
            # Projection from hidden_size → total_ple_dim.
            # Upstream uses ColumnParallelLinear with gather_output=True.
            self.per_layer_model_projection = nxdi_nn.ColumnParallelLinear(
                config.hidden_size,
                ple_total_dim,
                bias=False,
                gather_output=True,
                dtype=config.torch_dtype,
                tp_group=self.tp_group.device_group,
            )
            # Norm over the per-layer slice (acts on size per_layer_dim).
            self.per_layer_projection_norm = Gemma4RMSNorm(
                self.hidden_size_per_layer_input,
                config.rms_norm_eps,
                config.torch_dtype,
            )
            # Combination scale: (projection + embeds) * rsqrt(2)
            # Stored as a plain float (not buffer) for the same reason.
            self.per_layer_input_scale = float(2.0**-0.5)
            # Projection scale: divide projection output by sqrt(hidden_size).
            self.per_layer_projection_scale = float(config.hidden_size**-0.5)
            # Sharded loader for the PLE vocab embedding.
            set_weight_loader(
                self.embed_tokens_per_layer.weight,
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=(
                        self.embed_tokens_per_layer.vocab_size_per_rank
                    ),
                    num_shards=self.world_size,
                    is_storage_transposed=False,
                    pad_shard=True,
                ),
            )
        else:
            self.embed_tokens_per_layer = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None
            self.vocab_size_per_layer_input = None

    def _compute_per_layer_input(
        self,
        input_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        is_prefill: bool,
        rank: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Compute the per-layer input stream (PLE; not used by the 31B checkpoint, not tested).

        Steps:
          1. Look up input_ids in embed_tokens_per_layer →
             [T, num_layers * per_layer_dim], scale by sqrt(per_layer_dim),
             reshape to [T, num_layers, per_layer_dim].
          2. Project hidden_states (post-embedding inputs_embeds) via
             per_layer_model_projection → [T, num_layers * per_layer_dim],
             scale by 1/sqrt(hidden_size), reshape to
             [T, num_layers, per_layer_dim], normalize.
          3. Combine (projection + embeds) * rsqrt(2).

        Returns None when hidden_size_per_layer_input is 0.

        NOTE on shapes: in prefill mode the embeddings are scatter_tokens=True
        (each rank sees T/world_size tokens). We use the actual local
        token dim after embedding, NOT input_ids.shape[0].
        """
        if self.embed_tokens_per_layer is None:
            return None

        num_layers = self.config.num_hidden_layers
        per_layer_dim = self.hidden_size_per_layer_input

        # Step 1: per-layer embedding from input ids.
        # Mask out-of-vocab ids (PLE vocab can be smaller).
        if (
            self.vocab_size_per_layer_input is not None
            and self.vocab_size_per_layer_input < self.config.vocab_size
        ):
            ple_mask = torch.logical_and(
                input_ids >= 0,
                input_ids < self.vocab_size_per_layer_input,
            )
            ple_input_ids = torch.where(
                ple_mask, input_ids, torch.zeros_like(input_ids)
            )
        else:
            ple_input_ids = input_ids

        per_layer_embeds = self.embed_tokens_per_layer(
            ple_input_ids, scatter_tokens=is_prefill, rank=rank
        )
        # Local T (matches hidden_states' first dim after scatter).
        T_local = per_layer_embeds.shape[0]
        per_layer_embeds = per_layer_embeds * self.embed_scale_per_layer
        per_layer_embeds = per_layer_embeds.view(
            T_local, num_layers, per_layer_dim
        )

        # Step 2: project hidden_states → reshape → normalize.
        per_layer_projection = self.per_layer_model_projection(hidden_states)
        per_layer_projection = (
            per_layer_projection * self.per_layer_projection_scale
        )
        per_layer_projection = per_layer_projection.view(
            T_local, num_layers, per_layer_dim
        )
        per_layer_projection = self.per_layer_projection_norm(
            per_layer_projection
        )

        # Step 3: combine.
        return (
            per_layer_projection + per_layer_embeds
        ) * self.per_layer_input_scale

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
        )

        # Scale embedding output by sqrt(hidden_size)
        hidden_states = hidden_states * self.embed_scale

        # Per-layer input stream (None when hidden_size_per_layer_input is 0, as in the 31B checkpoint).
        # It is computed before the layer loop, from the scaled `hidden_states` that go into the first layer.
        per_layer_input = self._compute_per_layer_input(
            input_ids,
            hidden_states=hidden_states,
            is_prefill=is_prefill,
            rank=rank,
        )

        # No shared position embeddings: each layer computes its own RoPE
        for layer_idx, decoder_layer in enumerate(self.layers):
            ple_slice = (
                per_layer_input[:, layer_idx, :]
                if per_layer_input is not None
                else None
            )
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=None,
                attn_metadata=attn_metadata,
                per_layer_input=ple_slice,
            )

        hidden_states = self.norm(hidden_states)

        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states, []


# =============================================================================
# Section 7: Language Model Head
# Logit softcapping + tied embeddings.
# =============================================================================


class Gemma4ForCausalLM(nn.Module):
    """Gemma4 model with language modeling head.

    Includes final logit soft-capping: logits = cap * tanh(logits / cap)
    where cap = 30.0.
    """

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.model = Gemma4Model(config)
        self.final_logit_softcapping = config.final_logit_softcapping

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

        # Tied embeddings: lm_head weight will be loaded from embed_tokens
        self.lm_head = nxdi_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=self.tp_group.device_group,
            )

    def _apply_logit_softcapping(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply final logit soft-capping: logits = cap * tanh(logits / cap)."""
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = logits.float()
            logits = cap * torch.tanh(logits / cap)
        return logits

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
            raise ValueError("Input Embedding as Inputs is Not Supported Yet.")

        positions = positions.to(torch.int32)

        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        T = input_ids.shape[0]

        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt Length ({T}) must be > world_size ({self.world_size}) for SP."
            )

        hidden_states, _ = self.model(
            input_ids, positions, attn_metadata=attn_metadata, rank=rank
        )

        hidden_states_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )

        logits = self.lm_head(hidden_states_for_logits)

        # Apply final logit soft-capping
        logits = self._apply_logit_softcapping(logits)

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

            rejection_sampled_tokens = rejection_sampler(
                spec_decode_metadata, sampled_tokens
            )
            return rejection_sampled_tokens

        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = Gemma4Config.from_configs(hf_config, neuron_config)
        return cls(config)

    # -- KV Cache Management --

    def get_kv_spec(self):
        """Return per-layer KV cache specifications.

        Gemma4 has heterogeneous layers with different head_dim and KV heads:
        - SWA: head_dim=256, num_kv_heads=16 -> 2 per rank (TP=8), sliding_window=1024
        - Global: head_dim=512, num_kv_heads=4 -> 1 per rank (TP=8), full context
        """
        layers = []
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            layers.append(
                LayerSpec(
                    name=layer_name,
                    num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                    head_size=layer.self_attn.head_dim,
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=layer.self_attn.sliding_window,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    @staticmethod
    def _flat_kv(k, v, cache):
        """1-D view over the K and V planes (K then V, contiguous): the same shape as the compiler's base.

        One object per shared buffer: distinct overlapping inputs would be merged into a synthetic base.
        """
        assert v.storage_offset() == k.storage_offset() + k.numel(), "K and V planes must be adjacent"
        # Key on bytes, not on the layer's own shape: SWA ([N,4,32,256]) and global ([N,1,64,512])
        # layers over one buffer must get the SAME tensor object.
        key = ("flat", k.untyped_storage().data_ptr(), k.storage_offset(), 2 * k.numel(), k.dtype)
        if key not in cache:
            cache[key] = k.as_strided((2 * k.numel(),), (1,), k.storage_offset())
        return cache[key]

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor, torch.Tensor]]):
        """Give every layer the flat tensor of its shared KV buffer, and nothing else.

        The per-layer K and V views are deliberately not kept: two overlapping views of different
        shapes over one buffer make the compiler write the buffer back in full on every cache write.
        """
        shared: dict = {}
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise Exception(f"KV cache for layer {layer_name} not initialized")
            k, v = kv_caches[layer_name][0], kv_caches[layer_name][1]
            attn = layer.self_attn
            attn.kv_flat = self._flat_kv(k, v, shared)
            attn.kv_dims = tuple(int(x) for x in k.shape)
            attn.cache_dtype = k.dtype
            attn.cache_num_blocks = int(k.shape[0])

    # -- Weight Loading --

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load weights from checkpoint.

        Gemma4 checkpoint keys are prefixed with 'model.language_model.'.
        Key transformations:
          - Strip 'model.language_model.' prefix
          - Fuse Q/K/V into QKV (with KV replication for global layers)
          - Copy K weights to V for global layers (attention_k_eq_v)
          - Load QK norm weights (q_norm -> q_norm, k_norm -> k_norm)
          - Load layer_scalar per layer
          - Tied embeddings: lm_head from embed_tokens
        """
        tp_rank = self.rank
        tp_size = self.world_size

        mappings = dict()
        for layer_id in range(len(self.model.layers)):
            prefix = f"model.language_model.layers.{layer_id}"
            target_prefix = f"model.layers.{layer_id}"

            # Fused QKV weights
            qkv_sources = [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
            ]
            # For global layers with k_eq_v, v_proj is absent from checkpoint.
            # We duplicate k_proj as v_proj.
            is_global = self.config.is_global_layer(layer_id)
            if is_global and self.config.attention_k_eq_v:
                qkv_sources.append(f"{prefix}.self_attn.k_proj.weight")
            else:
                qkv_sources.append(f"{prefix}.self_attn.v_proj.weight")

            mappings[f"{target_prefix}.self_attn.qkv_proj_weight"] = qkv_sources
            mappings[f"{target_prefix}.self_attn.o_proj_weight"] = (
                f"{prefix}.self_attn.o_proj.weight"
            )

            # QK norm weights
            mappings[f"{target_prefix}.self_attn.q_norm.weight"] = (
                f"{prefix}.self_attn.q_norm.weight"
            )
            mappings[f"{target_prefix}.self_attn.k_norm.weight"] = (
                f"{prefix}.self_attn.k_norm.weight"
            )

            # 4 norm weights per layer
            mappings[f"{target_prefix}.input_layernorm.weight"] = (
                f"{prefix}.input_layernorm.weight"
            )
            mappings[f"{target_prefix}.post_attention_layernorm.weight"] = (
                f"{prefix}.post_attention_layernorm.weight"
            )
            mappings[f"{target_prefix}.pre_feedforward_layernorm.weight"] = (
                f"{prefix}.pre_feedforward_layernorm.weight"
            )
            mappings[f"{target_prefix}.post_feedforward_layernorm.weight"] = (
                f"{prefix}.post_feedforward_layernorm.weight"
            )

            # MLP weights
            mappings[f"{target_prefix}.mlp.gate_proj_weight"] = (
                f"{prefix}.mlp.gate_proj.weight"
            )
            mappings[f"{target_prefix}.mlp.up_proj_weight"] = (
                f"{prefix}.mlp.up_proj.weight"
            )
            mappings[f"{target_prefix}.mlp.down_proj_weight"] = (
                f"{prefix}.mlp.down_proj.weight"
            )

            # layer_scalar
            mappings[f"{target_prefix}.layer_scalar"] = f"{prefix}.layer_scalar"

            # ---- Per-layer input embeddings (PLE), only when hidden_size_per_layer_input > 0 ----
            # The 31B config sets hidden_size_per_layer_input to 0, so these modules do not exist for it.
            if (
                getattr(self.config, "hidden_size_per_layer_input", 0) or 0
            ) > 0:
                mappings[
                    f"{target_prefix}.per_layer_input_gate.weight"
                ] = f"{prefix}.per_layer_input_gate.weight"
                mappings[
                    f"{target_prefix}.per_layer_projection.weight"
                ] = f"{prefix}.per_layer_projection.weight"
                mappings[
                    f"{target_prefix}.post_per_layer_input_norm.weight"
                ] = f"{prefix}.post_per_layer_input_norm.weight"

        # Embedding
        mappings["model.embed_tokens.weight"] = (
            "model.language_model.embed_tokens.weight"
        )

        # ---- Model-level per-layer input embedding components (only when hidden_size_per_layer_input > 0) ----
        if (
            getattr(self.config, "hidden_size_per_layer_input", 0) or 0
        ) > 0:
            mappings["model.embed_tokens_per_layer.weight"] = (
                "model.language_model.embed_tokens_per_layer.weight"
            )
            mappings["model.per_layer_model_projection.weight"] = (
                "model.language_model.per_layer_model_projection.weight"
            )
            mappings["model.per_layer_projection_norm.weight"] = (
                "model.language_model.per_layer_projection_norm.weight"
            )

        # Final norm
        mappings["model.norm.weight"] = "model.language_model.norm.weight"

        # Tied embeddings: lm_head from embed_tokens
        mappings["lm_head.weight"] = "model.language_model.embed_tokens.weight"

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        load_result = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, mappings, device
        )
        rank_sharded = load_result.state_dict

        # Convert to model dtype
        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self.load_state_dict(rank_sharded, strict=False, assign=True)

        self._load_kv_cache_scales(checkpoint, device)

    def _load_kv_cache_scales(
        self, checkpoint: SafetensorsCheckpoint, device: torch.device
    ):
        """Load KV cache quantization scales from checkpoint if provided."""
        from vllm_neuron.utils.dtype_utils import QUANTIZED_KV_CACHE_DTYPES
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()

        for layer_id in range(len(self.model.layers)):
            attn = self.model.layers[layer_id].self_attn

            if vllm_config.cache_config.cache_dtype not in QUANTIZED_KV_CACHE_DTYPES:
                continue

            for scale_name in ("k_scale", "v_scale"):
                key = f"model.language_model.layers.{layer_id}.self_attn.{scale_name}"
                if key in checkpoint._tensor_name_to_file:
                    val = 1.0 / checkpoint._get_slice(key)[:].to(
                        dtype=torch.bfloat16, device=device
                    )
                else:
                    val = torch.ones(1, dtype=torch.bfloat16, device=device)
                setattr(attn, scale_name, val.reshape(1, 1))

            attn.k_scale_float = attn.k_scale.item()
            attn.v_scale_float = attn.v_scale.item()
