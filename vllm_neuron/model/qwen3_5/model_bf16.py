# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5-family dense hybrid model (Qwen3.6/3.8-27B), BF16 implementation
=========================================================================

Hybrid architecture: 48 DeltaNet (linear recurrent attention) + 16 Full Attention (GQA).
Pattern: [3 DeltaNet + 1 GQA] repeated 16 times across 64 layers.

Full attention: 24 Q heads / 4 KV heads, head_dim=256, partial RoPE (factor=0.25)
DeltaNet: 16 key heads (dim=128), 48 value heads (dim=128), conv1d(4), gated delta rule

Key differences from Qwen3 (dense):
  - Hybrid layer types (not all layers are the same)
  - DeltaNet layers have recurrent state (not KV cache)
  - DeltaNet uses NKI kernel for chunked delta rule (fp32 accumulation)
  - Partial RoPE on full attention layers (only 25% of head_dim rotated)
  - RMSNorm uses (1+w)*x pattern (zero-initialized weights)
  - head_dim=256 for full attention (exceeds SBUF 128 limit for fused decode)
  - M-RoPE with sections [11,11,10] (multimodal-compatible positions)

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    Qwen3.5-specific. Change when porting.
"""

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed.parallel_state import get_tp_group
from transformers import PretrainedConfig

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron import envs
from vllm_neuron.model.interfaces import SupportsMRoPE
from vllm_neuron.model.kv_cache import HybridKVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.neuron_utils import can_run_kernel
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import Qwen3_5Config

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1: RMS Normalization
# <-- MODEL-SPECIFIC: Qwen3.5 uses TWO norm patterns:
#   - Standard layers: (1+weight) * rms_norm(x) — zero-initialized weights
#   - DeltaNet internal (gated_rmsnorm): standard weight * rms_norm(x)
# =============================================================================


class Qwen3_5RMSNorm(nn.Module):
    """RMSNorm with (1+weight) pattern for Qwen3.5.

    Qwen3.5 uses zero-initialized norm weights with (1+w)*x semantics.
    This means at initialization the norm is identity (1+0=1).
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # .float() on the weight so (1 + w) is formed exactly in fp32, matching the
        # reference (modeling_qwen3_5.py:753). Without it the sum rounds in bf16
        # before the multiply, across all 66 instances of this module: R=1.3518 on
        # real checkpoint weights vs tau_R=1.2. Exactly one .float(), on the weight
        # only -- an extra one the reference lacks would overshoot into
        # over-precision (R < 1.0) rather than land at 1.0.
        return ((1.0 + self.weight.float()) * hidden_states).to(input_dtype)


class Qwen3_5StandardRMSNorm(nn.Module):
    """Standard RMSNorm (weight * x) for DeltaNet internal use."""

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
# <-- MODEL-SPECIFIC: Partial RoPE (25% of head_dim=256 → 64 dims rotated).
#   M-RoPE sections [11,11,10] for multimodal compatibility (text uses
#   identical values across all 3 dimensions).
# =============================================================================


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply partial rotary embedding — only first rotary_dim dims are rotated.

    Args:
        q: [Nh, T, Dh]  k: [Nkv, T, Dh]
        cos: [T, rotary_dim]  sin: [T, rotary_dim]
        rotary_dim: number of dimensions to rotate (head_dim * partial_rotary_factor)
    """
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    cos = cos.unsqueeze(0)  # [1, T, rotary_dim]
    sin = sin.unsqueeze(0)

    q_rot = _apply_rotary_emb(q_rot, cos, sin)
    k_rot = _apply_rotary_emb(k_rot, cos, sin)

    q = torch.cat([q_rot, q_pass], dim=-1)
    k = torch.cat([k_rot, k_pass], dim=-1)
    return q, k


class Qwen3_5RotaryEmbedding(nn.Module):
    """Partial M-RoPE for Qwen3.5 full attention layers.

    Only rotates partial_rotary_factor (0.25) of head_dim=256 → 64 dims.
    Uses rope_theta=10M. M-RoPE sections [11,11,10] sum to 32 (=rotary_dim/2).
    Supports both 1D positions [T] and 3D M-RoPE positions [3, T].
    """

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim  # 64
        rope_params = config.rope_parameters or {}
        self.mrope_section = rope_params.get("mrope_section", [11, 11, 10])
        self.mrope_interleaved = rope_params.get("mrope_interleaved", True)

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
        """Compute cos/sin for partial M-RoPE.

        Args:
            position_ids: [T] for standard, [3, T] for M-RoPE.

        Returns cos, sin of shape [T, rotary_dim].
        """
        inv_freq = self._compute_inv_freq(device)  # [rotary_dim/2 = 32]

        if position_ids.ndim == 1:
            # Standard 1D positions: [T]
            inv_freq_expanded = inv_freq[:, None].float()  # [32, 1]
            positions_expanded = position_ids[None, :].float()  # [1, T]
            freqs = (inv_freq_expanded @ positions_expanded).transpose(0, 1)  # [T, 32]
        else:
            # M-RoPE: position_ids is [3, T]
            if position_ids.ndim == 2 and position_ids.shape[0] == 3:
                position_ids = position_ids.unsqueeze(1)  # [3, 1, T]

            inv_freq_expanded = (
                inv_freq[None, None, :, None]
                .float()
                .expand(3, position_ids.shape[1], -1, 1)
            )  # [3, bs, 32, 1]
            position_ids_expanded = position_ids[:, :, None, :].float()  # [3, bs, 1, T]
            per_dim_freqs = (inv_freq_expanded @ position_ids_expanded).transpose(
                2, 3
            )  # [3, bs, T, 32]

            # Interleave T/H/W frequencies
            freqs = self._interleave_mrope(per_dim_freqs)  # [bs, T, 32]
            if freqs.shape[0] == 1:
                freqs = freqs.squeeze(0)  # [T, 32]

        emb = torch.cat((freqs, freqs), dim=-1)  # [T, 64]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)

    def _interleave_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        """Interleave T/H/W frequencies per mrope_section [11, 11, 10]."""
        last_dim = freqs.shape[-1]
        indices = torch.arange(last_dim, device=freqs.device, dtype=torch.int64)

        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            mask = (indices % 3 == offset) & (indices < length)
            freqs_t = torch.where(mask, freqs[dim], freqs_t)

        return freqs_t


def _causal_depthwise_conv(conv_in: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Run the depthwise causal conv of a prefill chunk as K shifted multiply-adds.

    The result has the values of ``F.conv1d`` over the channels-first transpose of
    ``conv_in``. The sum accumulates in float32 and is returned in the dtype of
    ``conv_in``.

    The layout is token-major, the layout of the QKV projection. A shift by j
    tokens is then a contiguous block of rows. On the native route a shift along
    the last axis of a channels-first [1, C, T] tensor is a strided copy: each of
    the K slices becomes DMA transfers of one row per channel, about 11 ms for
    each linear attention layer at 8192 tokens against about 1 ms on the XLA route.

    The native route uses this form, and the XLA route keeps ``F.conv1d``. The
    8192-token prefill graph is split at token 4096, and each route lowers one of
    the two forms wrong across that split. On the native route ``F.conv1d`` loses
    the conv window at the split: tokens 4096, 4097 and 4098 get a wrong conv
    output in each linear attention layer, and the recurrent state carries the
    error to the rest of the chunk. On the XLA route this form gives zero keys and
    values for every token from 4096 on. Either error makes every prompt of more
    than 4096 tokens give wrong output. ``-O3`` (no neuronx-cc modular flow) is
    right, but the 64-layer graph then exceeds the instruction limit.

    Args:
        conv_in: [K - 1 + T, C]. The K - 1 tokens of left context, then the T
            tokens of the chunk.
        weight: [C, 1, K].

    Returns:
        [T, C].
    """
    kernel = weight.shape[-1]
    tokens = conv_in.shape[0] - (kernel - 1)
    w = weight.squeeze(1).float().transpose(0, 1)  # [K, C]
    x = conv_in.float()
    out = x[0:tokens] * w[0]
    for j in range(1, kernel):
        out = out + x[j : j + tokens] * w[j]
    return out.to(conv_in.dtype)


def _swap_adjacent_dims(tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Exchange dimension `dim` and dimension `dim` + 1 of a tensor.

    Both attention layers of this model exchange a head dimension and a token
    dimension. The linear attention layer converts between (B, S, H, dim) and
    (B, H, S, dim). The full attention layer converts between (T, H, dim) and
    (H, T, dim), and it also converts the gathered key cache and value cache.

    Use a reshape when one of the 2 dimensions has the length 1. The 2 layouts then
    hold the elements in the same order, so the reshape gives the same result as a
    transpose. A transpose makes a Copy, and the compiler backend refuses that Copy
    at a decode batch of 4 or more rows:

        [NCC_IINAR001] ISA validation failed: Copy;
        inst failed assertion check: 'start_addr_active_channels'

    A decode step has one token for each row, so a token dimension has the length 1
    and the reshape applies. A prefill step has many tokens, so the transpose
    applies. A rank that holds 1 key head also takes the reshape path.

    The reshape also removes a data copy from each attention layer.

    Args:
        tensor: A tensor that has `dim` + 2 or more dimensions.
        dim: The first of the 2 dimensions to exchange. The default is 1.

    Returns:
        The tensor with the 2 dimensions exchanged.

    Example:
        >>> import torch
        >>> _swap_adjacent_dims(torch.zeros(4, 1, 6, 128)).shape
        torch.Size([4, 6, 1, 128])
        >>> _swap_adjacent_dims(torch.zeros(4, 1, 256), dim=0).shape
        torch.Size([1, 4, 256])
    """
    shape = list(tensor.shape)
    if shape[dim] == 1 or shape[dim + 1] == 1:
        shape[dim], shape[dim + 1] = shape[dim + 1], shape[dim]
        return tensor.reshape(shape)
    return tensor.transpose(dim, dim + 1).contiguous()


# =============================================================================
# Section 3: Full Attention (GQA with QK-norm and partial RoPE)
# Mixed: PARALLELISM (TP head sharding, SP, collectives) +
#        MODEL-SPECIFIC (GQA 24Q/4KV, head_dim=256, partial RoPE, QK-norm)
# =============================================================================


_ATTN_DECODE_KERNELS = ("torch", "tkg")


class Qwen3_5FullAttention(nn.Module):
    """Full attention with GQA, QK-norm, and partial RoPE.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads sharded across TP ranks
    - Prefill: all-gather → compute → reduce-scatter
    - Decode: manual (no fused megakernel due to head_dim=256 > 128 SBUF limit)

    <-- MODEL-SPECIFIC:
    - 24 Q heads / 4 KV heads (GQA ratio = 6)
    - head_dim=256 (exceeds SBUF 128 limit for NF.attention_decode)
    - Partial RoPE: only 64 of 256 dims rotated
    - Per-head QK-norm before RoPE
    - Output gate (sigmoid) gating the attention output
    """

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim  # 256
        self.rotary_dim = config.rotary_dim  # 64
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads  # 24
        self.num_key_value_heads = config.num_key_value_heads  # 4
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

        # <-- MODEL-SPECIFIC: Per-head QK-norm (also uses (1+w) pattern)
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)

        # <-- MODEL-SPECIFIC: Output gate (sigmoid)
        # HF q_proj is [num_heads * head_dim * 2, hidden] packing query + gate.
        # We split at load time into separate qkv_proj (for Q/K/V) and gate_proj.
        self.has_output_gate = config.attn_output_gate
        if self.has_output_gate:
            gate_size = self.num_attention_heads_per_rank * self.head_dim
            self.gate_proj_weight = nn.Parameter(
                torch.empty(self.hidden_size, gate_size, dtype=self.dtype)
            )

        # KV caches bound externally
        self.k_cache = None
        self.v_cache = None

        # Resolve the decode kernel here, before tracing. forward_decode() reads
        # the attribute, because a read of the envs module in a traced function
        # can break the graph.
        self._attn_decode_kernel = envs.VLLM_NEURON_BACKEND_QWEN3_5_ATTN_DECODE_KERNEL
        if self._attn_decode_kernel not in _ATTN_DECODE_KERNELS:
            raise ValueError(
                "VLLM_NEURON_BACKEND_QWEN3_5_ATTN_DECODE_KERNEL="
                f"{self._attn_decode_kernel!r} is not valid. Set it to one of {_ATTN_DECODE_KERNELS}."
            )
        layer_types = getattr(config, "layer_types", None) or []
        if "full_attention" in layer_types and layer_idx == layer_types.index(
            "full_attention"
        ):
            logger.info(
                "Full attention decode kernel at construction: ATTN_DECODE_KERNEL=%r",
                self._attn_decode_kernel,
            )

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
                shard_size=self.num_attention_heads * self.head_dim // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

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
        else:
            if self.world_size > 1:
                hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
            return self.forward_prefill(
                hidden_states, positions, position_embeddings, attn_metadata
            )

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

        # QKV projection
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
        ).squeeze(0)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        q = _swap_adjacent_dims(
            q.view(tokens, self.num_attention_heads_per_rank, self.head_dim), dim=0
        )
        # A rank that holds 1 key head takes the reshape path, and a decode step of
        # 1 token also takes it. See _swap_adjacent_dims.
        k = _swap_adjacent_dims(
            k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim), dim=0
        )
        v = _swap_adjacent_dims(
            v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim), dim=0
        )

        # <-- MODEL-SPECIFIC: QK-norm before partial RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        # <-- MODEL-SPECIFIC: Partial RoPE (only rotary_dim=64 of head_dim=256)
        cos, sin = position_embeddings
        q, k = apply_partial_rotary_pos_emb(q, k, cos, sin, self.rotary_dim)

        # KV cache update
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size

        k_flat = self._kv_for_cache(k.reshape(-1, self.head_dim).to(self.k_cache.dtype))
        v_flat = self._kv_for_cache(v.reshape(-1, self.head_dim).to(self.v_cache.dtype))

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

        if kv_segment_size:
            # Segmented (chunked) prefill. This chunk's Q must attend to every
            # token of the request written so far, not only to this chunk. Read
            # K/V back from the paged cache (the index_put_ above already stored
            # this chunk); cached_seq_len is the token count of the earlier
            # chunks. GQA head expansion happens inside segmented_attention.
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
            # Single-shot prefill: the chunk is the whole request.
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

        # <-- MODEL-SPECIFIC: Output gate (sigmoid applied to gate projection of input)
        # attn_output from flash_attention with tp_out=True is [heads, head_dim, seq]
        if self.has_output_gate:
            gate = torch.matmul(hidden_states, self.gate_proj_weight)
            gate = torch.sigmoid(gate)
            # gate is [seq, heads_per_rank * head_dim], reshape to [heads, head_dim, seq]
            num_heads = attn_output.shape[0]
            head_dim = attn_output.shape[1]
            gate = gate.view(-1, num_heads, head_dim).permute(1, 2, 0)
            attn_output = attn_output * gate

        # Output projection
        attn_output = attn_output.unsqueeze(0)
        attn_output = NF.o_proj(attn_output, self.o_proj_weight, None)
        attn_output = attn_output.squeeze(0)

        # >>> PARALLELISM: Reduce-scatter to SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
    ) -> torch.Tensor:
        """Decode path for full attention layers.

        NOTE: head_dim=256 exceeds the 128-dim SBUF limit for the fused
        NF.attention_decode megakernel. We use a manual decode path instead:
        QKV proj → QK-norm → partial RoPE → KV cache update → manual attention.
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B

        hidden_states = hidden_states.to(self.dtype)

        # QKV projection
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
        ).squeeze(0)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        q = _swap_adjacent_dims(
            q.view(tokens, self.num_attention_heads_per_rank, self.head_dim), dim=0
        )
        # A rank that holds 1 key head takes the reshape path, and a decode step of
        # 1 token also takes it. See _swap_adjacent_dims.
        k = _swap_adjacent_dims(
            k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim), dim=0
        )
        v = _swap_adjacent_dims(
            v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim), dim=0
        )

        # QK-norm + partial RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = position_embeddings
        q, k = apply_partial_rotary_pos_emb(q, k, cos, sin, self.rotary_dim)

        # KV cache update
        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size
        num_tokens = slot_mapping.shape[0]

        k_flat = self._kv_for_cache(k.reshape(-1, self.head_dim).to(self.k_cache.dtype))
        v_flat = self._kv_for_cache(v.reshape(-1, self.head_dim).to(self.v_cache.dtype))

        nkh = self.num_key_value_heads_per_rank
        head_indices_for_put = torch.arange(
            nkh, dtype=torch.long, device=hidden_states.device
        ).repeat_interleave(num_tokens)
        block_indices_for_put = block_indices.repeat(nkh)
        position_indices_for_put = position_indices.repeat(nkh)

        self.k_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            k_flat,
        )
        self.v_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            v_flat,
        )

        if (
            getattr(self, "_attn_decode_kernel", "torch") == "tkg"
            and S_decode == 1
            and nkh == 1
        ):
            q_rows = q.reshape(
                self.num_attention_heads_per_rank, B, self.head_dim
            ).transpose(0, 1)
            if NF.can_use_attention_decode_block_kv(q_rows, self.k_cache):
                attn_output = NF.attention_decode_block_kv(
                    q_rows,
                    k_flat,
                    v_flat,
                    self.k_cache,
                    self.v_cache,
                    block_table,
                    positions.reshape(B),
                    self.scaling,
                )
                return self._decode_output(hidden_states, attn_output)

        # Manual attention (no fused megakernel due to head_dim=256)
        # Gather KV from paged cache for this sequence
        max_blocks = block_table.shape[1]
        S_ctx = max_blocks * block_size

        # Reconstruct K, V from paged cache
        # block_table: [B, max_blocks] → indices into k_cache/v_cache
        # Unallocated entries hold the -1 sentinel. Clamp them to the null
        # block (0): a negative gather index is out of bounds on device and
        # reads memory outside the cache, which turns the masked tail of the
        # softmax into NaN. The mask below hides these slots either way.
        # _swap_adjacent_dims takes the reshape path when this rank holds 1 key
        # head. It then saves a copy of the full gathered cache.
        block_ids = block_table.clamp_min(0).reshape(-1)
        k_gathered = _swap_adjacent_dims(
            self.k_cache[block_ids].reshape(
                B, max_blocks, nkh, block_size, self.head_dim
            )
        ).reshape(B, nkh, S_ctx, self.head_dim)

        v_gathered = _swap_adjacent_dims(
            self.v_cache[block_ids].reshape(
                B, max_blocks, nkh, block_size, self.head_dim
            )
        ).reshape(B, nkh, S_ctx, self.head_dim)

        # Unwritten slots inside the gathered blocks can hold any bit pattern,
        # and 0 * NaN in P@V poisons the whole row. Sanitize the cache reads.
        k_gathered = torch.nan_to_num(k_gathered, nan=0.0, posinf=0.0, neginf=0.0)
        v_gathered = torch.nan_to_num(v_gathered, nan=0.0, posinf=0.0, neginf=0.0)

        # GQA expansion
        k_gathered = k_gathered.repeat_interleave(self.num_key_value_groups, dim=1)
        v_gathered = v_gathered.repeat_interleave(self.num_key_value_groups, dim=1)

        # q: [Nh, B*S_decode, Dh] → [B, Nh, S_decode, Dh]
        q_attn = q.reshape(
            self.num_attention_heads_per_rank, B, S_decode, self.head_dim
        ).permute(1, 0, 2, 3)

        # Scaled dot product attention with causal mask
        attn_weights = torch.matmul(q_attn, k_gathered.transpose(-2, -1)) * self.scaling

        # Mask out unfilled KV cache positions (positions > current decode position)
        # positions: [B*S_decode] = [B] for single-step decode
        pos = positions.reshape(B, S_decode)  # [B, S_decode]
        kv_positions = torch.arange(S_ctx, device=positions.device).unsqueeze(
            0
        )  # [1, S_ctx]
        # Valid if kv_position <= current_position (causal)
        causal_mask = kv_positions <= pos[:, -1:]  # [B, S_ctx]
        causal_mask = causal_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S_ctx]
        attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))

        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        )
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0).to(self.dtype)
        attn_output = torch.matmul(attn_weights, v_gathered)  # [B, Nh, S_decode, Dh]

        # Reshape to [tokens, Nh*Dh]. A decode step takes the reshape path,
        # because it has one token. See _swap_adjacent_dims.
        attn_output = _swap_adjacent_dims(attn_output).reshape(tokens, -1)
        return self._decode_output(hidden_states, attn_output)

    def _decode_output(
        self, hidden_states: torch.Tensor, attn_output: torch.Tensor
    ) -> torch.Tensor:
        """Apply the output gate, o_proj and the TP all-reduce to a [tokens, Nh*Dh] output."""
        # <-- MODEL-SPECIFIC: Output gate
        if self.has_output_gate:
            gate = torch.matmul(hidden_states, self.gate_proj_weight)
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        # Output projection
        attn_output = torch.matmul(attn_output, self.o_proj_weight)

        # >>> PARALLELISM: TP all-reduce <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(attn_output)

        return attn_output

    def _kv_for_cache(self, x: torch.Tensor) -> torch.Tensor:
        """Return the keys or values to write to the cache.

        The attention_tkg decode kernel can read a cached value with an attention
        weight of 0, also from a block that is not in the table of the row, and
        0 * NaN is NaN. A padded row can give NaN or Inf, and its write goes to
        the null block. So with the "tkg" decode kernel, replace NaN and Inf by 0.
        """
        if getattr(self, "_attn_decode_kernel", "torch") == "tkg":
            return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x


# =============================================================================
# Section 4: DeltaNet Linear Attention (Placeholder)
# <-- MODEL-SPECIFIC: Gated DeltaNet with NKI kernel
# =============================================================================


def _schur_level_masks_device(device: torch.device) -> torch.Tensor:
    """Build the Schur block-merge masks with device ops, as a (7*128, 128) tile.

    The layout must match `_make_schur_level_masks` in the kernel: plane 0 is
    block size 1, planes 1 to 6 are block sizes 2, 4, 8, 16, 32 and 64, and for
    block size b entry (i, j) is 1 when (i // b) is odd and (j // b) equals
    (i // b) - 1.

    The masks are built with torch ops, not from a numpy constant. A numpy
    constant enters the traced graph as a CPU tensor, and the FX to HLO pass
    then fails with "Tensor on device meta is not on the expected device".
    The neighbouring masks in the caller are built the same way.

    Example:
        masks = _schur_level_masks_device(query.device)   # (896, 128) float32
        level0 = masks[0:128, :]
    """
    size = 128
    rows = torch.arange(size, device=device).unsqueeze(1)
    cols = torch.arange(size, device=device).unsqueeze(0)
    planes = []
    for b in (1, 2, 4, 8, 16, 32, 64):
        row_block = rows // b
        col_block = cols // b
        sel = (row_block % 2 == 1) & (col_block == row_block - 1)
        planes.append(sel.to(torch.float32))
    return torch.cat(planes, dim=0)


_DELTANET_PREFILL_KERNELS = ("v1", "v2")
_DELTANET_FORMULATIONS = ("v1math", "fused")
_DELTANET_GEMM_DTYPES = ("fp32", "bf16")
_DELTANET_INVERSE_DTYPES = ("fp32", "bf16")
_DELTANET_DECODE_KERNELS = ("v1", "tkg", "batched", "inplace")
_DELTANET_CONV_STATE_WRITES = ("scatter", "select")
# The batched DeltaNet decode kernel puts the rows of a program on the 128 partitions.
_BATCHED_DECODE_MAX_ROWS = 128
# The in-place DeltaNet decode kernel keeps the state rows of its program in SBUF:
# rows x v-heads x value_dim elements on each partition. Keep them within half of SBUF.
_INPLACE_DECODE_MAX_STATE_BYTES = 96 * 1024
# nkilib gdn_tkg uses this epsilon for the l2norm and for the gated norm.
_GDN_TKG_EPS = 1e-6


def _check_deltanet_prefill_settings(
    kernel: str,
    formulation: str,
    gemm_dtype: str,
    inverse_dtype: str,
    lookahead: int,
    kgroups_per_launch: int,
) -> None:
    """Raise ValueError for a DeltaNet prefill kernel setting that is not valid.

    Example:
        _check_deltanet_prefill_settings("v2", "fused", "fp32", "fp32", 1, 0)
    """
    checks = (
        (
            "VLLM_NEURON_BACKEND_DELTANET_PREFILL_KERNEL",
            kernel,
            _DELTANET_PREFILL_KERNELS,
        ),
        (
            "VLLM_NEURON_BACKEND_DELTANET_FORMULATION",
            formulation,
            _DELTANET_FORMULATIONS,
        ),
        ("VLLM_NEURON_BACKEND_DELTANET_GEMM_DTYPE", gemm_dtype, _DELTANET_GEMM_DTYPES),
        (
            "VLLM_NEURON_BACKEND_DELTANET_INVERSE_DTYPE",
            inverse_dtype,
            _DELTANET_INVERSE_DTYPES,
        ),
        ("VLLM_NEURON_BACKEND_DELTANET_LOOKAHEAD", lookahead, (0, 1)),
    )
    for name, value, allowed in checks:
        if value not in allowed:
            raise ValueError(
                f"{name}={value!r} is not valid. Set it to one of {allowed}."
            )
    if kgroups_per_launch < 0:
        raise ValueError(
            "VLLM_NEURON_BACKEND_DELTANET_KGROUPS_PER_LAUNCH="
            f"{kgroups_per_launch!r} is not valid. Set it to 0 or a positive integer."
        )
    if (
        kernel == "v2"
        and formulation == "v1math"
        and (gemm_dtype != "fp32" or inverse_dtype != "fp32")
    ):
        raise ValueError(
            "The DeltaNet v1math formulation supports only fp32 matmuls. "
            "Set VLLM_NEURON_BACKEND_DELTANET_FORMULATION=fused to use a bf16 dtype."
        )


def _deltanet_v2_masks_device(device: torch.device) -> tuple[torch.Tensor, ...]:
    """Build the constant mask tiles of the v2 DeltaNet prefill kernel on the device.

    The tiles match `make_v2_constants` in `vllm_neuron/nki/deltanet/fused_v2.py`.
    They are built with torch ops for the reason given in
    `_schur_level_masks_device`.

    Returns:
        (lower_mask, identity, upper_diag_mask, level_masks), all float32.

    Example:
        lower, eye, upper_diag, levels = _deltanet_v2_masks_device(query.device)
    """
    ones = torch.ones(128, 128, dtype=torch.float32, device=device)
    lower = torch.tril(ones, diagonal=-1)
    identity = torch.eye(128, dtype=torch.float32, device=device)
    upper_diag = torch.triu(ones)
    return lower, identity, upper_diag, _schur_level_masks_device(device)


class Qwen3_5DeltaNet(nn.Module):
    """Gated DeltaNet linear attention layer.

    Components:
    - Causal conv1d (kernel_size=4) on input
    - QKV projection with DeltaNet-specific split [key, key, value]
    - Q/Gate interleaved split from q_proj
    - Chunked delta rule via NKI kernel (fp32 internal accumulation)
    - Output gate (sigmoid)
    - Gated RMSNorm (standard w*x)
    - Output projection

    State:
    - Recurrent state: (batch_size, num_v_heads, key_dim, value_dim) per layer
    - Conv1d state: (batch_size, conv_kernel-1, hidden_size) shift register
    """

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size

        # Resolve the prefill kernel settings here, before tracing. forward()
        # reads them from these attributes.
        self._dn_prefill_kernel = envs.VLLM_NEURON_BACKEND_DELTANET_PREFILL_KERNEL
        self._dn_formulation = envs.VLLM_NEURON_BACKEND_DELTANET_FORMULATION
        self._dn_gemm_dtype = envs.VLLM_NEURON_BACKEND_DELTANET_GEMM_DTYPE
        self._dn_inverse_dtype = envs.VLLM_NEURON_BACKEND_DELTANET_INVERSE_DTYPE
        self._dn_lookahead = envs.VLLM_NEURON_BACKEND_DELTANET_LOOKAHEAD
        self._dn_kgroups_per_launch = (
            envs.VLLM_NEURON_BACKEND_DELTANET_KGROUPS_PER_LAUNCH
        )
        _check_deltanet_prefill_settings(
            self._dn_prefill_kernel,
            self._dn_formulation,
            self._dn_gemm_dtype,
            self._dn_inverse_dtype,
            self._dn_lookahead,
            self._dn_kgroups_per_launch,
        )
        self._dn_decode_kernel = envs.VLLM_NEURON_BACKEND_DELTANET_DECODE_KERNEL
        if self._dn_decode_kernel not in _DELTANET_DECODE_KERNELS:
            raise ValueError(
                f"VLLM_NEURON_BACKEND_DELTANET_DECODE_KERNEL={self._dn_decode_kernel!r} "
                f"is not valid. Set it to one of {_DELTANET_DECODE_KERNELS}."
            )
        if self._dn_decode_kernel == "tkg" and config.rms_norm_eps != _GDN_TKG_EPS:
            raise ValueError(
                f"The tkg DeltaNet decode kernel uses eps {_GDN_TKG_EPS}, but rms_norm_eps "
                f"is {config.rms_norm_eps}. Set VLLM_NEURON_BACKEND_DELTANET_DECODE_KERNEL=v1."
            )
        self._dn_conv_write = envs.VLLM_NEURON_BACKEND_DELTANET_CONV_STATE_WRITE
        if self._dn_conv_write not in _DELTANET_CONV_STATE_WRITES:
            raise ValueError(
                f"VLLM_NEURON_BACKEND_DELTANET_CONV_STATE_WRITE={self._dn_conv_write!r} "
                f"is not valid. Set it to one of {_DELTANET_CONV_STATE_WRITES}."
            )

        # Log the kernel toggles once, here: forward() is traced, and a log call
        # there is a Dynamo graph break. forward() re-reads the first three from
        # os.environ, so this is accurate unless the environment changes later.
        if layer_idx == 0:
            logger.info(
                "DeltaNet kernel toggles at construction: "
                "DISABLE_NKI_DELTANET=%r NKI_PREFILL=%r NKI_DECODE=%r CPU_MODE=%r "
                "PREFILL_KERNEL=%r FORMULATION=%r GEMM_DTYPE=%r INVERSE_DTYPE=%r "
                "LOOKAHEAD=%r KGROUPS_PER_LAUNCH=%r DECODE_KERNEL=%r CONV_STATE_WRITE=%r",
                os.environ.get("VLLM_NEURON_DISABLE_NKI_DELTANET", ""),
                os.environ.get("VLLM_NEURON_NKI_PREFILL", "1"),
                os.environ.get("VLLM_NEURON_NKI_DECODE", "1"),
                os.environ.get("VLLM_NEURON_CPU_MODE", ""),
                self._dn_prefill_kernel,
                self._dn_formulation,
                self._dn_gemm_dtype,
                self._dn_inverse_dtype,
                self._dn_lookahead,
                self._dn_kgroups_per_launch,
                self._dn_decode_kernel,
                self._dn_conv_write,
            )

        # DeltaNet dimensions
        self.key_dim = config.linear_key_head_dim  # 128
        self.value_dim = config.linear_value_head_dim  # 128
        self.num_key_heads = config.linear_num_key_heads  # 16
        self.num_value_heads = config.linear_num_value_heads  # 48
        self.conv_kernel = config.linear_conv_kernel_dim  # 4

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: Head sharding for DeltaNet <<<
        self.num_key_heads_per_rank = self.num_key_heads // self.world_size
        self.num_value_heads_per_rank = self.num_value_heads // self.world_size

        # Projections — matches HF checkpoint structure:
        # in_proj_qkv: [10240, 5120] = fused [K_all, K_all, V_all] = [16*128 + 16*128 + 48*128]
        # in_proj_z: [6144, 5120] = output gate (48 value_heads * 128)
        # in_proj_a: [48, 5120] = beta (one per value_head)
        # in_proj_b: [48, 5120] = key scaling (one per value_head)
        qkv_size_per_rank = (
            self.num_key_heads_per_rank * self.key_dim * 2  # Q + K
            + self.num_value_heads_per_rank * self.value_dim  # V
        )
        z_size_per_rank = self.num_value_heads_per_rank * self.value_dim

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size_per_rank, dtype=self.dtype)
        )
        self.z_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, z_size_per_rank, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(
                self.num_value_heads_per_rank * self.value_dim,
                self.hidden_size,
                dtype=self.dtype,
            )
        )

        # Beta/key-scale projections (one scalar per value_head)
        self.a_proj_weight = nn.Parameter(
            torch.empty(
                self.hidden_size, self.num_value_heads_per_rank, dtype=self.dtype
            )
        )
        self.b_proj_weight = nn.Parameter(
            torch.empty(
                self.hidden_size, self.num_value_heads_per_rank, dtype=self.dtype
            )
        )

        # Log decay and timestep bias (per value_head)
        self.a_log = nn.Parameter(
            torch.empty(self.num_value_heads_per_rank, dtype=torch.float32)
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.num_value_heads_per_rank, dtype=torch.float32)
        )

        # Conv1d on QKV concat: [qkv_total, 1, kernel_size] (no bias in checkpoint)
        qkv_total = (
            self.num_key_heads * self.key_dim * 2
            + self.num_value_heads * self.value_dim
        )
        qkv_total_per_rank = qkv_size_per_rank
        self.conv_weight = nn.Parameter(
            torch.empty(qkv_total_per_rank, 1, self.conv_kernel, dtype=self.dtype)
        )

        # Gated RMSNorm (standard w*x, applied per-head on value_dim)
        # Weight is [value_dim=128], applied to last dim of [B, S, num_v_heads, value_dim]
        self.group_norm = Qwen3_5StandardRMSNorm(
            self.value_dim,
            config.rms_norm_eps,
            self.dtype,
        )

        # Recurrent state buffers — nn.Parameters with requires_grad=False so they
        # appear as graph inputs/outputs for torch.compile.
        # One row for each concurrent request, and one more row for the padded
        # rows of a decode batch when the ladder pads (state_pool_rows). The
        # caller gives the row of each batch row in `batch_slot_indices`, because
        # a request keeps one row for its whole life while its position in the
        # batch can change. The forward stops a batch of more than one row that
        # has no row map (see below).
        from vllm_neuron.vllm.worker.deltanet_state_slots import state_pool_rows

        max_batch_size = state_pool_rows(
            config.neuron_config.num_seqs_buckets if config.neuron_config else None
        )
        self.recurrent_state = nn.Parameter(
            torch.zeros(
                max_batch_size,
                self.num_value_heads_per_rank,
                self.key_dim,
                self.value_dim,
                dtype=self.dtype,
            ),
            requires_grad=False,
        )
        qkv_size_per_rank = (
            self.num_key_heads_per_rank * self.key_dim * 2
            + self.num_value_heads_per_rank * self.value_dim
        )
        # Do not flatten the last 2 axes of this buffer to make the decode faster.
        # It looks attractive, because the compiler refuses the fast offset DGE for
        # the dynamic index (`slot_idx`) of this buffer at a small batch. A flat row
        # of `qkv_size_per_rank * (conv_kernel - 1)` was measured at context 2048:
        # the compiler gave the same 96 diagnostics ("unsupported partition shape
        # for offset dge") for each decode graph at the same batch thresholds, the
        # step time moved less than 1% at TP=4 and at TP=8, and
        # `average_partition_utilization` was the same to 14 digits. The tiler
        # gives the gather the layout its consumer needs, so this shape cannot
        # move the decision. The cause is the tiling of the decode body.
        #
        # The refusal rule is a THRESHOLD on the decode batch, not a divisor. Read
        # from every graph of one TP=4 compile cache: the buckets of 3 and 4 are
        # refused, with 96 diagnostics each, and the buckets of 5, 6 and 7 are
        # granted. At TP=8 every bucket is granted. An earlier copy of this comment
        # said the refusal needs a partition count that divides by 16, which no
        # bucket of 3 to 7 satisfies. That paraphrase was wrong.
        #
        # The 2 refused graphs also change the output: they make a request depend
        # on how many other requests share its decode step. The fix is in the
        # bucket ladder, which must not hold a refused bucket at TP=4. See
        # `_QWEN36_27B_DECODE_MIN_BUCKET_TP4` in the online test configurations.
        self.conv_state = nn.Parameter(
            torch.zeros(
                max_batch_size,
                qkv_size_per_rank,
                self.conv_kernel - 1,
                dtype=self.dtype,
            ),
            requires_grad=False,
        )

        # State snapshots for prefix caching: row P holds the state after a saved
        # prompt prefix. Only a prefill step reads and writes them, so the decode
        # graphs do not change. Row 0 is a scratch row: a step that saves nothing
        # writes row 0, and a step that restores nothing reads its own state row.
        # The scheduler gives the rows (see vllm_neuron/vllm/core/deltanet_snapshots.py).
        # The platform resolves the row count (NeuronPlatform._apply_qwen3_5_defaults).
        neuron_config = getattr(config, "neuron_config", None)
        self.num_snapshot_rows = (
            neuron_config.deltanet_snapshot_rows if neuron_config is not None else 0
        )
        if self.num_snapshot_rows >= 2:
            self.recurrent_snapshot = nn.Parameter(
                torch.zeros(
                    self.num_snapshot_rows,
                    self.num_value_heads_per_rank,
                    self.key_dim,
                    self.value_dim,
                    dtype=self.dtype,
                ),
                requires_grad=False,
            )
            self.conv_snapshot = nn.Parameter(
                torch.zeros(
                    self.num_snapshot_rows,
                    qkv_size_per_rank,
                    self.conv_kernel - 1,
                    dtype=self.dtype,
                ),
                requires_grad=False,
            )

        # Constant masks for NKI fused chunked kernel (128x128 fp32)
        P_MAX = 128
        lower = np.tril(np.ones((P_MAX, P_MAX), dtype=np.float32), k=-1)
        self.register_buffer(
            "_nki_lower_mask",
            torch.from_numpy(lower),
            persistent=False,
        )
        self.register_buffer(
            "_nki_identity",
            torch.eye(P_MAX, dtype=torch.float32),
            persistent=False,
        )
        lower_diag = np.tril(np.ones((P_MAX, P_MAX), dtype=np.float32), k=0)
        self.register_buffer(
            "_nki_lower_mask_diag",
            torch.from_numpy(lower_diag),
            persistent=False,
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        qkv_size_per_rank = (
            self.num_key_heads_per_rank * self.key_dim * 2
            + self.num_value_heads_per_rank * self.value_dim
        )

        # ── QKV repack loader ──
        # HF stores in_proj_qkv as [Q_all | K_all | V_all, hidden] = [10240, 5120].
        # A naive contiguous shard gives each rank the WRONG head mix (e.g. rank 0
        # gets 10 Q heads instead of 2Q+2K+6V at TP=8).
        # Fix: reorder to [rank0_Q|rank0_K|rank0_V|rank1_Q|...] then slice per rank.
        num_k_heads = self.num_key_heads
        num_v_heads = self.num_value_heads
        head_k_dim = self.key_dim
        head_v_dim = self.value_dim
        world_size = self.world_size

        def _qkv_repack_transform(slices, rank):
            weight = slices[0][:]  # [10240, 5120]
            key_dim = num_k_heads * head_k_dim  # 2048
            value_dim = num_v_heads * head_v_dim  # 6144
            local_k_heads = num_k_heads // world_size
            local_v_heads = num_v_heads // world_size
            local_key_dim = local_k_heads * head_k_dim
            local_value_dim = local_v_heads * head_v_dim

            q_all = weight[:key_dim]  # [2048, 5120]
            k_all = weight[key_dim : 2 * key_dim]  # [2048, 5120]
            v_all = weight[2 * key_dim :]  # [6144, 5120]

            q_rank = q_all[rank * local_key_dim : (rank + 1) * local_key_dim]
            k_rank = k_all[rank * local_key_dim : (rank + 1) * local_key_dim]
            v_rank = v_all[rank * local_value_dim : (rank + 1) * local_value_dim]

            # Stack as [Q_local | K_local | V_local] then transpose for our layout
            shard = torch.cat([q_rank, k_rank, v_rank], dim=0)  # [qkv_per_rank, 5120]
            return shard.T  # [5120, qkv_per_rank]

        qkv_loader = SafetensorsWeightLoader(transform=_qkv_repack_transform)

        # ── Conv1d repack loader ──
        # HF stores conv1d.weight as [Q_all|K_all|V_all, 1, kernel] = [10240, 1, 4].
        # Same reorder needed as QKV along dim 0.
        def _conv_repack_transform(slices, rank):
            weight = slices[0][:]  # [10240, 1, 4]
            key_dim = num_k_heads * head_k_dim
            value_dim = num_v_heads * head_v_dim
            local_k_heads = num_k_heads // world_size
            local_v_heads = num_v_heads // world_size
            local_key_dim = local_k_heads * head_k_dim
            local_value_dim = local_v_heads * head_v_dim

            q_all = weight[:key_dim]
            k_all = weight[key_dim : 2 * key_dim]
            v_all = weight[2 * key_dim :]

            q_rank = q_all[rank * local_key_dim : (rank + 1) * local_key_dim]
            k_rank = k_all[rank * local_key_dim : (rank + 1) * local_key_dim]
            v_rank = v_all[rank * local_value_dim : (rank + 1) * local_value_dim]

            return torch.cat([q_rank, k_rank, v_rank], dim=0)  # [qkv_per_rank, 1, 4]

        conv_loader = SafetensorsWeightLoader(transform=_conv_repack_transform)

        # in_proj_z: HF [6144, 5120] → param [5120, z_per_rank]
        z_loader = sharding_weight_loader(
            shard_dim=1,
            shard_size=self.num_value_heads_per_rank * self.value_dim,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        # out_proj: HF [5120, 6144] → param [z_per_rank, 5120]
        o_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.num_value_heads_per_rank * self.value_dim,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        # in_proj_a/b: HF [48, 5120] → param [5120, num_v_heads_per_rank]
        ab_loader = sharding_weight_loader(
            shard_dim=1,
            shard_size=self.num_value_heads_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        # A_log and dt_bias: [48] → [num_v_heads_per_rank] (1D, no transpose)
        scalar_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.num_value_heads_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=False,
        )

        set_weight_loader(self.qkv_proj_weight, qkv_loader)
        set_weight_loader(self.z_proj_weight, z_loader)
        set_weight_loader(self.o_proj_weight, o_loader)
        set_weight_loader(self.a_proj_weight, ab_loader)
        set_weight_loader(self.b_proj_weight, ab_loader)
        set_weight_loader(self.a_log, scalar_loader)
        set_weight_loader(self.dt_bias, scalar_loader)
        set_weight_loader(self.conv_weight, conv_loader)

    def _l2norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1, eps=1e-6)

    def _write_decode_conv_state(
        self, slot_idx: torch.Tensor, new_conv_state: torch.Tensor
    ) -> None:
        """Write the new conv window of each decode row to its state row.

        VLLM_NEURON_BACKEND_DELTANET_CONV_STATE_WRITE selects the write:
        - "select" (the default): one torch.where over the full buffer for each
          batch row, then a write of the full buffer. This uses no dynamic
          index. A state row that no batch row writes keeps its window.
        - "scatter": index_put_ with slot_idx.

        At TP=4, the compiler lowers the index_put_ of some decode buckets (7, 8,
        10, 12 and 20) to about 415,000 GpSimd instructions on one core, and a
        decode step takes about 580 ms. "select" keeps these buckets at 48 to
        60 ms. A gather of the new windows in state row order (no scatter) was
        also measured. It removed the GpSimd instructions at bucket 10 but not
        at bucket 20.

        The padded rows of a batch share one free state row (see
        DeltaNetStateSlots.slots_for_batch), so a state row can have more than
        one writer. Both writes give that row the window of one of its writers.
        "select" uses the last writer.
        """
        new_conv_state = new_conv_state.to(self.conv_state.dtype)
        if getattr(self, "_dn_conv_write", "select") == "scatter":
            self.conv_state.index_put_((slot_idx,), new_conv_state)
            return
        rows = self.conv_state.shape[0]
        row_ids = torch.arange(rows, device=slot_idx.device).reshape(rows, 1)
        full = self.conv_state
        for j in range(new_conv_state.shape[0]):
            hit = (slot_idx[j : j + 1].long() == row_ids).reshape(rows, 1, 1)
            full = torch.where(hit, new_conv_state[j : j + 1], full)
        self.conv_state.copy_(full)

    def _write_prefill_conv_state(
        self,
        slot_idx: torch.Tensor,
        new_conv_state: torch.Tensor,
        deltanet_snapshot_rows: torch.Tensor | None,
    ) -> None:
        """Write the conv window of the prefill request to its state row and snapshot row.

        A prefill step carries one request, so ``slot_idx[:1]`` is its state row and
        ``deltanet_snapshot_rows[1:2]`` its save row (row 0, the scratch row, when the
        step saves nothing). VLLM_NEURON_BACKEND_DELTANET_CONV_STATE_WRITE selects the
        write, as for a decode step (see `_write_decode_conv_state`):
        - "select" (the default): a torch.where over each full buffer, then a write
          of the full buffer.
        - "scatter": index_put_ at the row.

        neuronx-cc 2.29 lowers the index_put_ of the 256-token prefill bucket to one
        dynamic store for each element of the window (10,240 for each write, 48 layers,
        2 writes). At TP=4 the 256 bucket then takes about 1,060 ms on the device, and
        every request pays it once, because its last prefill chunk is at most 256
        tokens. The larger buckets compile the same index_put_ to a row write. The
        earlier compiler (vLLM 0.27.1 image) did so at 256 too, and the bucket took
        83 ms. Both writes give the buffers the same values.
        """
        if getattr(self, "_dn_conv_write", "select") == "scatter":
            self.conv_state.index_put_(
                (slot_idx,), new_conv_state.to(self.conv_state.dtype)
            )
            if deltanet_snapshot_rows is not None:
                self.conv_snapshot.index_put_(
                    (deltanet_snapshot_rows[1:2],),
                    new_conv_state.to(self.conv_snapshot.dtype),
                )
            return

        def _select_write(buf: torch.Tensor, row: torch.Tensor) -> None:
            rows = buf.shape[0]
            row_ids = torch.arange(rows, device=row.device).reshape(rows, 1, 1)
            hit = row[:1].long().reshape(1, 1, 1) == row_ids
            buf.copy_(torch.where(hit, new_conv_state.to(buf.dtype), buf))

        _select_write(self.conv_state, slot_idx)
        if deltanet_snapshot_rows is not None:
            _select_write(self.conv_snapshot, deltanet_snapshot_rows[1:2])

    def _recurrent_step(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step recurrent update for decode (PyTorch fallback)."""
        query = self._l2norm(query)
        key = self._l2norm(key)
        scale = 1.0 / (query.shape[-1] ** 0.5)
        query = query * scale

        q_t = query[:, :, 0]  # [B, H, k_dim]
        k_t = key[:, :, 0]
        v_t = value[:, :, 0]  # [B, H, v_dim]
        g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
        beta_t = beta[:, :, 0].unsqueeze(-1)  # [B, H, 1]

        new_state = recurrent_state * g_t
        kv_mem = (new_state * k_t.unsqueeze(-1)).sum(dim=-2)  # [B, H, v_dim]
        delta = (v_t - kv_mem) * beta_t
        new_state = new_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        output = (new_state * q_t.unsqueeze(-1)).sum(dim=-2)  # [B, H, v_dim]

        return output.unsqueeze(2), new_state

    def _nki_recurrent_step(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step recurrent update using NKI batched decode kernel."""
        from vllm_neuron.nki.deltanet.recurrent import (
            _deltanet_recurrent_step_batched_kernel,
        )

        query = self._l2norm(query)
        key = self._l2norm(key)
        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]
        scale = 1.0 / (k_dim**0.5)
        query = query * scale

        BH = B * H
        # Select token 0 (decode), then transpose to (BH*128, 1) column layout
        # for standard NKI indexing (no .ap() needed)
        q_2d = query.reshape(BH, S, k_dim)[:, 0, :]  # (BH, 128)
        k_2d = key.reshape(BH, S, k_dim)[:, 0, :]
        v_2d = value.reshape(BH, S, v_dim)[:, 0, :]
        query_col = q_2d.reshape(BH * k_dim, 1).contiguous()
        key_col = k_2d.reshape(BH * k_dim, 1).contiguous()
        value_col = v_2d.reshape(BH * v_dim, 1).contiguous()
        g_flat = g.reshape(BH, S)[:, 0:1].contiguous()
        beta_flat = beta.reshape(BH, S)[:, 0:1].contiguous()
        state_flat = recurrent_state.reshape(BH * k_dim, v_dim).contiguous()

        _nki_step = wrap_nki(_deltanet_recurrent_step_batched_kernel)
        output_col, state_flat_out = _nki_step[1](
            query_col,
            key_col,
            value_col,
            g_flat,
            beta_flat,
            state_flat,
        )

        output = output_col.reshape(BH, k_dim).reshape(B, H, S, v_dim)
        new_state = state_flat_out.reshape(B, H, k_dim, v_dim)
        return output, new_state

    def _nki_tkg_decode_step(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        z: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Do one decode step with the nkilib gdn_tkg kernel.

        The kernel does the gates, the l2norm of q and k, the recurrence and the
        gated norm. Give it the tensors before these operations: query and key
        (B, 1, Hk, K) for each k-head, value (B, 1, Hv, V), the raw projections a
        and b (B, Hv), the output gate input z (B, Hv * V) and the state
        (B, Hv, K, V).

        Returns:
            The output after the gated norm (B, Hv * V) in bf16, and the new
            state (B, Hv, K, V) in float32.
        """
        from nkilib.experimental.gdn import gdn_tkg

        from vllm_neuron.utils.hardware_config import get_logical_nc_config

        B = query.shape[0]
        Hk = self.num_key_heads_per_rank
        Hv = self.num_value_heads_per_rank
        K, V = self.key_dim, self.value_dim
        BH = B * Hv
        # The kernel takes q and k for each v-head, with the v-head index fastest in
        # each batch row. v-head h uses k-head h // R, as in the GQA repeat of forward().
        q = (
            query.reshape(B, Hk, 1, K)
            .expand(B, Hk, Hv // Hk, K)
            .reshape(BH, K)
            .contiguous()
        )
        k = (
            key.reshape(B, Hk, 1, K)
            .expand(B, Hk, Hv // Hk, K)
            .reshape(BH, K)
            .contiguous()
        )
        out, state_out = wrap_nki(gdn_tkg)[get_logical_nc_config()](
            q,
            k,
            value.reshape(BH, V).contiguous(),
            b.reshape(BH).float().contiguous(),
            a.reshape(BH).float().contiguous(),
            self.a_log.float().contiguous(),
            self.dt_bias.float().contiguous(),
            z.reshape(BH, V).contiguous(),
            self.group_norm.weight.float().contiguous(),
            recurrent_state.float().reshape(BH, K, V).contiguous(),
        )
        return out.reshape(B, Hv * V), state_out.reshape(B, Hv, K, V)

    def _batched_decode_grid(self, query: torch.Tensor) -> int:
        """Return the grid of the batched decode kernel for query (B, ...), or 0 if it cannot run."""
        from vllm_neuron.utils.hardware_config import get_logical_nc_config

        if not can_run_kernel(query):
            return 0
        if self.key_dim != 128 or self.value_dim != 128:
            return 0
        rows = query.shape[0] * self.num_value_heads_per_rank
        for grid in (get_logical_nc_config(), 1):
            if rows % grid == 0 and rows // grid <= _BATCHED_DECODE_MAX_ROWS:
                return grid
        return 0

    def _inplace_decode_grid(self, query: torch.Tensor) -> int:
        """Return the grid of the in-place decode kernel for query (B, ...), or 0 if it cannot run."""
        from vllm_neuron.utils.hardware_config import get_logical_nc_config

        if not can_run_kernel(query):
            return 0
        if self.key_dim != 128 or self.value_dim != 128:
            return 0
        if self.recurrent_state.dtype != torch.bfloat16:
            return 0
        hv = self.num_value_heads_per_rank
        if hv > _BATCHED_DECODE_MAX_ROWS:
            return 0
        batch = query.shape[0]
        row_bytes = hv * self.value_dim * self.recurrent_state.element_size()
        for grid in (get_logical_nc_config(), 1):
            if (
                batch % grid == 0
                and batch // grid * row_bytes <= _INPLACE_DECODE_MAX_STATE_BYTES
            ):
                return grid
        return 0

    def _nki_inplace_decode_step(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        z: torch.Tensor,
        slot_idx: torch.Tensor,
        grid: int,
    ) -> torch.Tensor:
        """Do one decode step with the in-place DeltaNet decode kernel.

        The kernel does the gates, the l2norm of q and k, the recurrence and the
        gated norm for the whole batch in one call. It reads the state rows
        slot_idx of self.recurrent_state and writes the new rows back in place.

        Returns:
            The output after the gated norm (B, Hv * V).
        """
        from vllm_neuron.nki.deltanet.decode_inplace import deltanet_decode_inplace

        B = query.shape[0]
        Hk = self.num_key_heads_per_rank
        Hv = self.num_value_heads_per_rank
        K, V = self.key_dim, self.value_dim
        BH = B * Hv
        # Row r = batch * Hv + v-head. v-head h uses k-head h // R, as in the GQA repeat of forward().
        q = (
            query.reshape(B, Hk, 1, K)
            .expand(B, Hk, Hv // Hk, K)
            .reshape(BH, K)
            .contiguous()
        )
        k = (
            key.reshape(B, Hk, 1, K)
            .expand(B, Hk, Hv // Hk, K)
            .reshape(BH, K)
            .contiguous()
        )
        out, new_state = wrap_nki(deltanet_decode_inplace)[grid](
            q,
            k,
            value.reshape(BH, V).contiguous(),
            z.reshape(BH, V).contiguous(),
            a.reshape(BH, 1).contiguous(),
            b.reshape(BH, 1).contiguous(),
            self.a_log.float().repeat(B).reshape(BH, 1).contiguous(),
            self.dt_bias.float().repeat(B).reshape(BH, 1).contiguous(),
            self.group_norm.weight.reshape(1, V),
            self.recurrent_state,
            slot_idx.to(torch.int32).reshape(B, 1),
            torch.eye(128, dtype=torch.float32, device=query.device),
            norm_eps=float(self.group_norm.variance_epsilon),
        )
        if envs.VLLM_NEURON_CPU_MODE:
            # The NKI simulator returns a copy of the buffer. On the device the
            # kernel output aliases the buffer, so the copy is not needed there.
            with torch.no_grad():
                self.recurrent_state.copy_(new_state)
        return out.reshape(B, Hv * V)

    def _batched_decode_groups(
        self, query: torch.Tensor
    ) -> tuple[int, tuple[int, ...]]:
        """Return the grid and the batch size of each call of the batched decode kernel.

        One call takes at most grid x 128 rows (batch x v-heads of the rank). A
        larger batch takes more calls, and the batch sizes of the calls differ by
        at most 1. A batch that one call can take gives one call, as before.

        Returns:
            (grid, batch sizes), or (0, ()) if the kernel cannot run.
        """
        from vllm_neuron.utils.hardware_config import get_logical_nc_config

        if not can_run_kernel(query):
            return 0, ()
        batch = query.shape[0]
        grid = self._batched_decode_grid(query)
        if grid:
            return grid, (batch,)
        if self.key_dim != 128 or self.value_dim != 128:
            return 0, ()
        hv = self.num_value_heads_per_rank
        for grid in (get_logical_nc_config(), 1):
            max_batch = grid * _BATCHED_DECODE_MAX_ROWS // hv
            if max_batch == 0:
                continue
            n_calls = -(-batch // max_batch)
            base, extra = divmod(batch, n_calls)
            sizes = tuple(base + 1 if i < extra else base for i in range(n_calls))
            if all(size * hv % grid == 0 for size in sizes):
                return grid, sizes
        return 0, ()

    def _nki_batched_decode_groups(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        z: torch.Tensor,
        slot_idx: torch.Tensor,
        grid: int,
        sizes: tuple[int, ...],
    ) -> torch.Tensor:
        """Do one decode step with one call of the batched decode kernel for each group.

        Each group is a range of batch rows. A call reads the state rows of its
        group from the state buffer, and writes the new state rows back in place.
        Thus no copy of the state of the full batch occurs.

        Returns:
            The output after the gated norm (B, Hv * V).
        """
        outputs = []
        b0 = 0
        for size in sizes:
            b1 = b0 + size
            rows = slot_idx[b0:b1]
            out, new_state = self._nki_batched_decode_step(
                query[b0:b1],
                key[b0:b1],
                value[b0:b1],
                a[b0:b1],
                b[b0:b1],
                z[b0:b1],
                self.recurrent_state[rows],
                grid,
            )
            self.recurrent_state.index_put_(
                (rows,), new_state.to(self.recurrent_state.dtype)
            )
            outputs.append(out)
            b0 = b1
        return torch.cat(outputs, dim=0)

    def _nki_batched_decode_step(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        z: torch.Tensor,
        recurrent_state: torch.Tensor,
        grid: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Do one decode step with the batched DeltaNet decode kernel.

        The kernel does the gates, the l2norm of q and k, the recurrence and the
        gated norm for all rows of a program at once. It takes the same tensors as
        _nki_tkg_decode_step, and it reads and writes the state in its own dtype.

        Returns:
            The output after the gated norm (B, Hv * V), and the new state
            (B, Hv, K, V) in the dtype of recurrent_state.
        """
        from vllm_neuron.nki.deltanet.decode_batched import deltanet_decode_batched

        B = query.shape[0]
        Hk = self.num_key_heads_per_rank
        Hv = self.num_value_heads_per_rank
        K, V = self.key_dim, self.value_dim
        BH = B * Hv
        # Row r = batch * Hv + v-head. v-head h uses k-head h // R, as in the GQA repeat of forward().
        q = (
            query.reshape(B, Hk, 1, K)
            .expand(B, Hk, Hv // Hk, K)
            .reshape(BH, K)
            .contiguous()
        )
        k = (
            key.reshape(B, Hk, 1, K)
            .expand(B, Hk, Hv // Hk, K)
            .reshape(BH, K)
            .contiguous()
        )
        out, state_out = wrap_nki(deltanet_decode_batched)[grid](
            q,
            k,
            value.reshape(BH, V).contiguous(),
            z.reshape(BH, V).contiguous(),
            a.reshape(BH, 1).float().contiguous(),
            b.reshape(BH, 1).float().contiguous(),
            self.a_log.float().repeat(B).reshape(BH, 1).contiguous(),
            self.dt_bias.float().repeat(B).reshape(BH, 1).contiguous(),
            self.group_norm.weight.float().reshape(1, V).expand(BH, V).contiguous(),
            recurrent_state.reshape(BH, K, V).contiguous(),
            torch.eye(128, dtype=torch.float32, device=query.device),
            norm_eps=float(self.group_norm.variance_epsilon),
        )
        return out.reshape(B, Hv * V), state_out.reshape(B, Hv, K, V)

    def _chunked_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequential DeltaNet forward for prefill (pure PyTorch, no NKI)."""
        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]

        query = self._l2norm(query)
        key = self._l2norm(key)
        scale = 1.0 / (k_dim**0.5)
        query = query * scale

        if initial_state is None:
            state = torch.zeros(
                B, H, k_dim, v_dim, dtype=torch.float32, device=query.device
            )
        else:
            state = initial_state.float()

        outputs = []
        for t in range(S):
            q_t = query[:, :, t]  # [B, H, k_dim]
            k_t = key[:, :, t]  # [B, H, k_dim]
            v_t = value[:, :, t]  # [B, H, v_dim]
            g_t = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
            beta_t = beta[:, :, t].unsqueeze(-1)  # [B, H, 1]

            state = state * g_t
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)  # [B, H, v_dim]
            delta = (v_t - kv_mem) * beta_t
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            output_t = (state * q_t.unsqueeze(-1)).sum(dim=-2)  # [B, H, v_dim]
            outputs.append(output_t)

        output = torch.stack(outputs, dim=2)  # [B, H, S, v_dim]
        return output, state

    def _nki_chunked_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused chunked DeltaNet prefill via NKI kernel.

        Replaces _chunked_forward's Python loop with a single NKI kernel call
        per head-group, eliminating the loop unrolling that causes ISGV902.
        """
        from vllm_neuron.nki.deltanet.fused import (
            deltanet_fused_chunked_fwd_multihead,
        )

        B, H, S, k_dim = query.shape
        v_dim = value.shape[-1]

        if initial_state is None:
            initial_state = torch.zeros(
                B, H, k_dim, v_dim, dtype=torch.float32, device=query.device
            )
        else:
            initial_state = initial_state.float()

        # Kernel expects (BH, S, 128) for q/k/v, (BH, S, 1) for g/beta,
        # (BH, 128, 128) for initial_state
        BH = B * H
        q_flat = query.reshape(BH, S, k_dim).contiguous()
        k_flat = key.reshape(BH, S, k_dim).contiguous()
        v_flat = value.reshape(BH, S, v_dim).contiguous()
        g_flat = g.reshape(BH, S, 1).contiguous()
        beta_flat = beta.reshape(BH, S, 1).contiguous()
        state_flat = initial_state.reshape(BH, k_dim, v_dim).contiguous()

        # Launch with head_group_size matching LNC (=2 on trn2) to match NxDI.
        # Grid=2 compiles the kernel for 2 logical NeuronCores.
        head_group_size = int(
            os.environ.get("QWEN36_DELTANET_MULTIHEAD_GROUP_SIZE", "2")
        )
        wrapped_kernel = wrap_nki(deltanet_fused_chunked_fwd_multihead)

        pair_outputs = []
        pair_states = []
        # Masks as local tensors (XLA constants) instead of model attributes (graph inputs).
        lower_mask = torch.tril(
            torch.ones(128, 128, dtype=torch.float32, device=query.device), diagonal=-1
        )
        identity_mat = torch.eye(128, dtype=torch.float32, device=query.device)
        lower_mask_diag = torch.tril(
            torch.ones(128, 128, dtype=torch.float32, device=query.device)
        )
        level_masks = _schur_level_masks_device(query.device)

        for bh_start in range(0, BH, head_group_size):
            bh_end = min(bh_start + head_group_size, BH)
            out_pair, state_pair = wrapped_kernel[bh_end - bh_start](
                q_flat[bh_start:bh_end],
                k_flat[bh_start:bh_end],
                v_flat[bh_start:bh_end],
                g_flat[bh_start:bh_end],
                beta_flat[bh_start:bh_end],
                state_flat[bh_start:bh_end],
                lower_mask,
                identity_mat,
                lower_mask_diag,
                level_masks,
            )
            pair_outputs.append(out_pair)
            pair_states.append(state_pair)

        output_flat = torch.cat(pair_outputs, dim=0)
        final_state_flat = torch.cat(pair_states, dim=0)

        output = output_flat.reshape(B, H, S, v_dim)
        new_state = final_state_flat.reshape(B, H, k_dim, v_dim)
        return output, new_state

    def _nki_chunked_forward_v2(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused chunked DeltaNet prefill via the v2 NKI kernel.

        query and key are [B, S, Hk, dk] without the GQA repeat, and value is
        [B, S, Hv, dv], all in the model dtype. g and beta are [B, Hv, S] float32.
        The kernel reads k-head h // (Hv // Hk) for v-head h, so the repeated
        q and k are not materialized. One kernel call covers all heads, unless
        VLLM_NEURON_BACKEND_DELTANET_KGROUPS_PER_LAUNCH splits the heads.

        Returns the output [B, Hv, S, dv] and the final state [B, Hv, dk, dv],
        both float32, like `_nki_chunked_forward`.
        """
        from vllm_neuron.nki.deltanet.fused_v2 import deltanet_prefill_v2

        q = _swap_adjacent_dims(query)
        k = _swap_adjacent_dims(key)
        v = _swap_adjacent_dims(value)
        B, Hk, S, k_dim = q.shape
        Hv = v.shape[1]
        v_dim = v.shape[-1]
        group = Hv // Hk

        if initial_state is None:
            initial_state = torch.zeros(
                B, Hv, k_dim, v_dim, dtype=torch.float32, device=query.device
            )
        else:
            initial_state = initial_state.float()

        BHk = B * Hk
        q_flat = q.reshape(BHk, S, k_dim).contiguous()
        k_flat = k.reshape(BHk, S, k_dim).contiguous()
        v_flat = v.reshape(B * Hv, S, v_dim).contiguous()
        g_flat = g.reshape(B * Hv, S, 1).float().contiguous()
        beta_flat = beta.reshape(B * Hv, S, 1).float().contiguous()
        state_flat = initial_state.reshape(B * Hv, k_dim, v_dim).contiguous()

        # The grid is the logical core pair. Each core owns whole k-head groups.
        head_group_size = int(
            os.environ.get("QWEN36_DELTANET_MULTIHEAD_GROUP_SIZE", "2")
        )
        grid = 2 if head_group_size == 2 and BHk % 2 == 0 else 1
        kgroups = getattr(self, "_dn_kgroups_per_launch", 0)
        kh_per_launch = BHk if kgroups == 0 else min(BHk, kgroups * grid)
        if BHk % kh_per_launch != 0:
            kh_per_launch = BHk

        lower_mask, identity_mat, upper_diag_mask, level_masks = (
            _deltanet_v2_masks_device(query.device)
        )
        wrapped_kernel = wrap_nki(deltanet_prefill_v2)
        options = dict(
            formulation=getattr(self, "_dn_formulation", "fused"),
            gemm_dtype=getattr(self, "_dn_gemm_dtype", "fp32"),
            inverse_dtype=getattr(self, "_dn_inverse_dtype", "fp32"),
            lookahead=getattr(self, "_dn_lookahead", 0),
            revision=1,
        )

        outputs = []
        states = []
        for kh_start in range(0, BHk, kh_per_launch):
            kh_end = kh_start + kh_per_launch
            vh_start = kh_start * group
            vh_end = kh_end * group
            out_part, state_part = wrapped_kernel[grid](
                q_flat[kh_start:kh_end],
                k_flat[kh_start:kh_end],
                v_flat[vh_start:vh_end],
                g_flat[vh_start:vh_end],
                beta_flat[vh_start:vh_end],
                state_flat[vh_start:vh_end],
                lower_mask,
                identity_mat,
                upper_diag_mask,
                level_masks,
                **options,
            )
            outputs.append(out_part)
            states.append(state_part)

        output_flat = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        final_state_flat = states[0] if len(states) == 1 else torch.cat(states, dim=0)
        output = output_flat.reshape(B, Hv, S, v_dim)
        new_state = final_state_flat.reshape(B, Hv, k_dim, v_dim)
        return output, new_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        is_prefill: bool = True,
        batch_slot_indices: torch.Tensor | None = None,
        deltanet_padding_mask: torch.Tensor | None = None,
        deltanet_snapshot_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """DeltaNet forward: conv1d → projections → delta rule → gate → out_proj.

        ``deltanet_snapshot_rows`` is ``[restore_row, save_row]`` of a prefill step
        with prefix caching. A restore row other than 0 gives the initial state.
        The step writes its final state to the save row (row 0 is a scratch row).
        """
        if not is_prefill or getattr(self, "num_snapshot_rows", 0) < 2:
            deltanet_snapshot_rows = None
        # >>> PARALLELISM: All-gather for SP during prefill <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape
        if is_prefill:
            batch_size = 1
            seq_len = tokens
        else:
            batch_size = tokens
            seq_len = 1

        # State slot indices: which state buffer row each batch row reads and
        # writes. The caller gives them, because a request keeps one row for its
        # whole life while its position in the batch can change between steps.
        #
        # The fallback `arange` maps batch position i to row i. That is correct
        # for one row only. With two rows the map changes when the batch order
        # changes, and a request then reads the state of another request. Stop
        # here instead. `batch_size` is a Python int and the None test is a
        # Python test, so this raises while Dynamo traces, not at run time.
        if batch_slot_indices is None:
            if batch_size > 1:
                raise ValueError(
                    f"A batch of {batch_size} rows needs batch_slot_indices, and "
                    "the caller gave none. Each row of the DeltaNet state belongs "
                    "to one request, and a batch position does not identify a "
                    "request. Pass batch_slot_indices, or use a batch of one row."
                )
            slot_idx = torch.arange(batch_size, device=hidden_states.device)
        else:
            slot_idx = batch_slot_indices

        # ── Projections ──
        # QKV: [tokens, hidden] @ [hidden, qkv_per_rank] → [tokens, qkv_per_rank]
        qkv = torch.matmul(hidden_states, self.qkv_proj_weight)
        # Z (output gate): [tokens, hidden] @ [hidden, z_per_rank] → [tokens, z_per_rank]
        z = torch.matmul(hidden_states, self.z_proj_weight)
        # A, B (decay/beta projections): [tokens, hidden] @ [hidden, num_v_heads_per_rank]
        a = torch.matmul(hidden_states, self.a_proj_weight)
        b = torch.matmul(hidden_states, self.b_proj_weight)

        # ── Causal Conv1d on QKV ──
        if is_prefill:
            k_minus_1 = self.conv_kernel - 1

            # Left context of the causal conv. Chunk 0 of a request sees zeros.
            # A continuation chunk (positions[0] > 0) sees the last K-1 real
            # tokens that the previous chunk saved in conv_state[slot]. The
            # mask is a tensor, not a Python branch, so one compiled graph
            # serves both cases.
            #
            # `positions[:1]` is the first position of the one request of this
            # step, so the mask holds for that request. A prefill step carries
            # one request only: the Neuron platform chunks a prefill at batch
            # size 1, and it does not mix a prefill with a decode. `batch_size`
            # above is 1 for this reason.
            prev_ctx = self.conv_state[slot_idx].to(qkv.dtype)
            if deltanet_snapshot_rows is not None:
                # A prefix cache hit starts from the saved conv window of its prefix.
                snap_src = deltanet_snapshot_rows[:1]
                restore = (snap_src != 0).reshape(1, 1, 1)
                prev_ctx = torch.where(
                    restore, self.conv_snapshot[snap_src].to(prev_ctx.dtype), prev_ctx
                )
            if positions is not None:
                cont_mask = (positions[:1].long() != 0).reshape(1, 1, 1)
                prev_ctx = torch.where(cont_mask, prev_ctx, torch.zeros_like(prev_ctx))
            # The K-1 tokens of left context, then the chunk. Each route takes the
            # form that its compiler lowers correctly and fast in the 8192-token graph
            # (see _causal_depthwise_conv): token-major shifted multiply-adds on the
            # native route, F.conv1d over channels-first on the XLA route.
            token_major = envs.is_native_backend()
            if token_major:
                # [K-1 + tokens, qkv_per_rank]
                conv_in = torch.cat([prev_ctx[0].transpose(0, 1), qkv], dim=0)
                conv_out = _causal_depthwise_conv(conv_in, self.conv_weight)
            else:
                # [1, qkv_per_rank, K-1 + tokens]
                conv_in = torch.cat(
                    [prev_ctx, qkv.transpose(0, 1).unsqueeze(0)], dim=-1
                )
                conv_out = F.conv1d(
                    conv_in,
                    self.conv_weight,
                    bias=None,
                    padding=0,
                    groups=conv_in.shape[1],
                )

            # Update conv state buffer: save the last (kernel-1) REAL token positions.
            #
            # The chunk spans the PADDED bucket length, not the prompt length, so a state
            # taken from its last K-1 positions holds the *padding* -- e.g. indices
            # 125..127 for a 16-token prompt in a 128 bucket. Decode then convolved
            # [pad, pad, pad, x_new] instead of [x[n-3], x[n-2], x[n-1], x_new], so its q/k/v
            # were wrong from the first decode step while prefill's own outputs stayed correct
            # (at a real position t, the window t-3..t is real tokens or the left context).
            #
            # Measured, port against port at the same position: with padding, decode's q/k/v
            # differed from prefill's by 2.1% / 3.0% / 1.4% and the recurrence output by 7.5%;
            # at zero padding the same probes agree to 4.2e-4 / 1.2e-4 / 1.4e-5 and 1.4e-3,
            # i.e. bf16 rounding. g and beta -- which bypass the conv -- matched EXACTLY in
            # both cases, which is what proves the input hidden state was identical and
            # isolates the conv.
            #
            # The real length comes from the padding mask the layer already receives. Indexing
            # by a computed tensor rather than a Python int keeps it traceable: the length is
            # data-dependent, so a literal slice cannot express it.
            #
            # Index into conv_in (left context ++ chunk), not the bare chunk: the
            # last K-1 real positions are conv_in[real_len : real_len + K-1], which
            # is always in range and spans the chunk boundary when real_len < K-1.
            real_len = deltanet_padding_mask.float().sum().to(torch.int32)
            conv_idx = real_len + torch.arange(
                k_minus_1, device=qkv.device, dtype=torch.int32
            )
            if token_major:
                # [K-1, qkv_per_rank] -> the state layout [1, qkv_per_rank, K-1]
                new_conv_state = (
                    conv_in.index_select(0, conv_idx.long())
                    .transpose(0, 1)
                    .unsqueeze(0)
                    .contiguous()
                )
            else:
                new_conv_state = conv_in.index_select(-1, conv_idx.long()).contiguous()
            self._write_prefill_conv_state(
                slot_idx, new_conv_state, deltanet_snapshot_rows
            )
            qkv_post_conv = F.silu(conv_out)
            if not token_major:
                # [1, qkv_per_rank, tokens] -> [tokens, qkv_per_rank]
                qkv_post_conv = qkv_post_conv.squeeze(0).transpose(0, 1)
        else:
            # Decode: [B, qkv_per_rank, 1] — each request has 1 new token
            qkv_3d = qkv.unsqueeze(-1)  # [B, qkv_per_rank, 1]
            conv_state = self.conv_state[slot_idx]  # [B, qkv_per_rank, kernel-1]
            conv_input = torch.cat(
                [conv_state, qkv_3d], dim=-1
            )  # [B, qkv_per_rank, kernel]

            # Manual depthwise conv: sum over kernel_size dimension
            w = self.conv_weight.squeeze(1)  # [qkv_per_rank, kernel_size]
            conv_out = (conv_input * w.unsqueeze(0)).sum(
                dim=-1, keepdim=True
            )  # [B, qkv_per_rank, 1]

            # Update conv state: shift in new token
            new_conv_state = conv_input[:, :, -(self.conv_kernel - 1) :].contiguous()
            self._write_decode_conv_state(slot_idx, new_conv_state)
            # [B, qkv_per_rank, 1] → squeeze → [B, qkv_per_rank] = [tokens, qkv_per_rank]
            qkv_post_conv = F.silu(conv_out.squeeze(-1))  # [B, qkv_per_rank]

        # ── Split Q, K, V from post-conv output ──
        q_size = self.num_key_heads_per_rank * self.key_dim
        k_size = self.num_key_heads_per_rank * self.key_dim
        v_size = self.num_value_heads_per_rank * self.value_dim

        query = qkv_post_conv[:, :q_size]
        key = qkv_post_conv[:, q_size : q_size + k_size]
        value = qkv_post_conv[:, q_size + k_size :]

        # Reshape to heads: [tokens, dim] → [B, S, num_heads, head_dim]
        query = query.reshape(
            batch_size, seq_len, self.num_key_heads_per_rank, self.key_dim
        )
        key = key.reshape(
            batch_size, seq_len, self.num_key_heads_per_rank, self.key_dim
        )
        value = value.reshape(
            batch_size, seq_len, self.num_value_heads_per_rank, self.value_dim
        )

        # ── Gating: beta and g ──
        beta = b.reshape(batch_size, seq_len, self.num_value_heads_per_rank).sigmoid()
        g = -self.a_log.float().exp() * F.softplus(
            a.reshape(batch_size, seq_len, self.num_value_heads_per_rank).float()
            + self.dt_bias.float()
        )

        # ── Neutralize padding positions (prevents recurrent state corruption) ──
        # deltanet_padding_mask: [T] bf16 tensor, 1.0 for real tokens, 0.0 for padding.
        # Passed as a direct argument (not self.) so Dynamo traces it as a graph input.
        #
        # torch.where, not `x * mask`. A padded row carries whatever the attention
        # layers computed from unwritten cache slots, and that can overflow bf16 to
        # Inf. Then g = -Inf, and -Inf * 0 is NaN, exp(NaN) is NaN, and the state
        # is lost for every real token that follows. Zeroing q, k and v as well
        # keeps the update k @ ((v - kv_mem) * beta) at 0 * 0 instead of 0 * Inf.
        if is_prefill:
            is_real = deltanet_padding_mask.reshape(batch_size, seq_len, 1) != 0
            g = torch.where(is_real, g, torch.zeros_like(g))
            beta = torch.where(is_real, beta, torch.zeros_like(beta))
            is_real_h = is_real.unsqueeze(-1)
            query = torch.where(is_real_h, query, torch.zeros_like(query))
            key = torch.where(is_real_h, key, torch.zeros_like(key))
            value = torch.where(is_real_h, value, torch.zeros_like(value))

        # The v2 prefill kernel reads q and k for each k-head and v in the model
        # dtype, so keep them before the GQA repeat and the float32 cast.
        query_kh, key_kh, value_md = query, key, value

        # ── Expand K heads to match V heads (factor = num_v_heads / num_k_heads) ──
        head_repeat = self.num_value_heads_per_rank // self.num_key_heads_per_rank
        if head_repeat > 1:
            query = (
                query.unsqueeze(3)
                .expand(-1, -1, -1, head_repeat, -1)
                .reshape(
                    batch_size, seq_len, self.num_value_heads_per_rank, self.key_dim
                )
            )
            key = (
                key.unsqueeze(3)
                .expand(-1, -1, -1, head_repeat, -1)
                .reshape(
                    batch_size, seq_len, self.num_value_heads_per_rank, self.key_dim
                )
            )

        # Convert to (B, H, S, dim) and convert to float32 for recurrence.
        # A decode step takes the reshape path. See _swap_adjacent_dims.
        query = _swap_adjacent_dims(query).float()
        key = _swap_adjacent_dims(key).float()
        value = _swap_adjacent_dims(value).float()
        g = _swap_adjacent_dims(g).float()
        beta = _swap_adjacent_dims(beta).float()

        # ── Delta Rule Recurrence ──
        recurrent_state = self.recurrent_state[slot_idx]

        _use_nki_deltanet = (
            can_run_kernel(query)
            and not os.environ.get("VLLM_NEURON_DISABLE_NKI_DELTANET", "") == "1"
        )
        _nki_prefill = (
            _use_nki_deltanet and os.environ.get("VLLM_NEURON_NKI_PREFILL", "1") == "1"
        )
        _nki_decode = (
            _use_nki_deltanet and os.environ.get("VLLM_NEURON_NKI_DECODE", "1") == "1"
        )

        # Do not log the resolved path here. This function is traced by Dynamo, and a
        # Python side effect (even adding to a set to log only once) is a graph break
        # that fails compilation. __init__ logs the toggles instead.
        # True when the decode kernel also applies the gated norm (gdn_tkg).
        gated_output = False
        if is_prefill:
            # Zero initial state for fresh sequences (position 0 = new sequence).
            # Warmup forward passes pollute the buffer; this in-graph reset ensures
            # the first real prefill starts clean. For chunked prefill continuations
            # (position > 0) the buffer carries valid state from the prior chunk.
            #
            # The reset applies to row `slot_idx` only, which is the row of the
            # one request of this step. A prefill step carries one request, so
            # the position of that request decides its own reset. This reset also
            # cleans the row after a decode warmup, which writes every row.
            # A decode step needs no reset, because a decode is a continuation.
            if deltanet_snapshot_rows is not None:
                # A prefix cache hit starts from the saved state of its prefix. Its
                # first position is not 0, so the reset below keeps this state.
                snap_src = deltanet_snapshot_rows[:1]
                restore = (snap_src != 0).reshape(1, 1, 1, 1)
                recurrent_state = torch.where(
                    restore, self.recurrent_snapshot[snap_src], recurrent_state
                )
            if positions is not None:
                reset_mask = (positions[:1].long() == 0).reshape(1, 1, 1, 1)
                # torch.where, not `state * (1 - mask)`: a warmup pass can leave
                # NaN or Inf in the buffer and NaN * 0 is still NaN, which then
                # poisons every real request that follows.
                recurrent_state = torch.where(
                    reset_mask, torch.zeros_like(recurrent_state), recurrent_state
                )

            if _nki_prefill and getattr(self, "_dn_prefill_kernel", "v1") == "v2":
                output, new_recurrent_state = self._nki_chunked_forward_v2(
                    query_kh, key_kh, value_md, g, beta, initial_state=recurrent_state
                )
            elif _nki_prefill:
                output, new_recurrent_state = self._nki_chunked_forward(
                    query, key, value, g, beta, initial_state=recurrent_state
                )
            else:
                output, new_recurrent_state = self._chunked_forward(
                    query, key, value, g, beta, initial_state=recurrent_state
                )
        else:
            decode_kernel = getattr(self, "_dn_decode_kernel", "v1")
            # "inplace" uses the "batched" path for a batch that it cannot take.
            batched_grid, batched_sizes = (
                self._batched_decode_groups(query_kh)
                if decode_kernel in ("batched", "inplace")
                else (0, ())
            )
            if len(batched_sizes) > 1 and deltanet_snapshot_rows is not None:
                # The grouped calls write the state rows in place, so they do
                # not give the new state of the full batch for a snapshot.
                batched_grid = 0
            inplace_grid = (
                self._inplace_decode_grid(query_kh)
                if decode_kernel == "inplace" and deltanet_snapshot_rows is None
                else 0
            )
            if _nki_decode and inplace_grid:
                # The kernel writes the new state rows in place, so the write
                # below does not occur.
                output = self._nki_inplace_decode_step(
                    query_kh, key_kh, value_md, a, b, z, slot_idx, inplace_grid
                )
                new_recurrent_state = None
                gated_output = True
            elif _nki_decode and decode_kernel == "tkg":
                output, new_recurrent_state = self._nki_tkg_decode_step(
                    query_kh, key_kh, value_md, a, b, z, recurrent_state
                )
                gated_output = True
            elif _nki_decode and batched_grid and len(batched_sizes) == 1:
                # The model writes the new state back with an in-place index_put_
                # (below). Keep that write. A functional index_put whose result
                # leaves the graph gave a wrong state at batch 1 in a device test.
                output, new_recurrent_state = self._nki_batched_decode_step(
                    query_kh, key_kh, value_md, a, b, z, recurrent_state, batched_grid
                )
                gated_output = True
            elif _nki_decode and batched_grid:
                # The batch is too large for one call. Each call writes its state
                # rows in place, so the write below does not occur.
                output = self._nki_batched_decode_groups(
                    query_kh,
                    key_kh,
                    value_md,
                    a,
                    b,
                    z,
                    slot_idx,
                    batched_grid,
                    batched_sizes,
                )
                new_recurrent_state = None
                gated_output = True
            elif _nki_decode:
                output, new_recurrent_state = self._nki_recurrent_step(
                    query, key, value, g, beta, recurrent_state.float()
                )
            else:
                output, new_recurrent_state = self._recurrent_step(
                    query, key, value, g, beta, recurrent_state.float()
                )

        if new_recurrent_state is not None:
            # Write back updated state via index_put_ so the aliasing pass
            # can trace the mutation back to the parameter buffer
            self.recurrent_state.index_put_(
                (slot_idx,), new_recurrent_state.to(self.recurrent_state.dtype)
            )
        if deltanet_snapshot_rows is not None:
            # Write the same bf16 state that the request row holds. A hit that
            # restores it then starts from the bits that a miss continues from.
            self.recurrent_snapshot.index_put_(
                (deltanet_snapshot_rows[1:2],),
                new_recurrent_state.to(self.recurrent_snapshot.dtype),
            )

        # ── Output: norm → gate → project ──
        if gated_output:
            # The decode kernel applied the gated norm. The output is [B, v_per_rank].
            output = output.to(self.dtype).reshape(
                batch_size, seq_len, self.num_value_heads_per_rank, self.value_dim
            )
        else:
            output = output.to(self.dtype)
            # The recurrent step gives [B, H, S, v_dim]. The next step needs
            # [B, S, H, v_dim]. A decode step takes the reshape path.
            # See _swap_adjacent_dims.
            output = _swap_adjacent_dims(output)
            output = output.reshape(
                batch_size, seq_len, self.num_value_heads_per_rank, self.value_dim
            )
            output = self.group_norm(output)

            # Output gate: silu(z) element-wise
            z_gate = z.reshape(
                batch_size, seq_len, self.num_value_heads_per_rank, self.value_dim
            )
            output = output * F.silu(z_gate)

        # Flatten heads and project: [tokens, v_per_rank] @ [v_per_rank, hidden]
        output = output.reshape(tokens, -1)
        output = torch.matmul(output, self.o_proj_weight)

        # >>> PARALLELISM <<<
        if is_prefill:
            if self.world_size > 1:
                output = self.tp_group.reduce_scatter(output, dim=0)
        else:
            if self.world_size > 1:
                self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 5: Dense MLP
# Mixed: PARALLELISM (TP intermediate sharding, SP collectives) +
#        MODEL-SPECIFIC (SiLU gate/up/down, no bias, intermediate_size=17408)
# =============================================================================


class Qwen3_5MLP(nn.Module):
    """Dense SiLU MLP with TP intermediate sharding."""

    def __init__(self, config: Qwen3_5Config):
        super().__init__()

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
# Section 6: Decoder Layer (Hybrid dispatch)
# <-- MODEL-SPECIFIC: Dispatches to either FullAttention or DeltaNet based on layer_type
# =============================================================================


class Qwen3_5DecoderLayer(nn.Module):
    """Single decoder layer — dispatches to full attention or DeltaNet."""

    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_full_attention = config.is_full_attention(layer_idx)

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )

        if self.is_full_attention:
            self.self_attn = Qwen3_5FullAttention(config, layer_idx=layer_idx)
        else:
            self.self_attn = Qwen3_5DeltaNet(config, layer_idx=layer_idx)

        self.mlp = Qwen3_5MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        is_prefill: bool = True,
        batch_slot_indices: torch.Tensor | None = None,
        deltanet_padding_mask: torch.Tensor | None = None,
        deltanet_snapshot_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self Attention (or DeltaNet)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.is_full_attention:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
                is_prefill=is_prefill,
                batch_slot_indices=batch_slot_indices,
                deltanet_padding_mask=deltanet_padding_mask,
                deltanet_snapshot_rows=deltanet_snapshot_rows,
            )

        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=is_prefill)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Section 7: Model Backbone
# =============================================================================


class Qwen3_5Model(nn.Module):
    """Qwen3.5 transformer backbone (hybrid layers)."""

    def __init__(self, config: Qwen3_5Config):
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
            ),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        rotary_position_ids: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        batch_slot_indices: torch.Tensor | None = None,
        deltanet_padding_mask: torch.Tensor | None = None,
        deltanet_snapshot_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Use the first full attention layer's metadata (layer 3 in [DDD,A] pattern)
        first_attn_layer_name = (
            f"layers.{self.config.full_attention_interval - 1}.self_attn"
        )
        max_query_len = attn_metadata[first_attn_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_attn_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
        )

        if (
            is_prefill
            and self.world_size > 1
            and inputs_embeds is not None
            and is_token_ids is not None
        ):
            local_len = hidden_states.shape[0]
            start = self.rank * local_len
            inputs_embeds = inputs_embeds[start : start + local_len]
            is_token_ids = is_token_ids[start : start + local_len]

        hidden_states = NF.merge_prompt_embeds(
            hidden_states, inputs_embeds, is_token_ids
        )

        # M-RoPE: use rotary_position_ids [3, T] if provided, else fall back to positions
        rope_positions = (
            rotary_position_ids if rotary_position_ids is not None else positions
        )
        position_embeddings = self.rotary_emb(
            rope_positions, device=hidden_states.device, dtype=hidden_states.dtype
        )

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
                is_prefill=is_prefill,
                batch_slot_indices=batch_slot_indices,
                deltanet_padding_mask=deltanet_padding_mask,
                deltanet_snapshot_rows=deltanet_snapshot_rows,
            )

        hidden_states = self.norm(hidden_states)

        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states


# =============================================================================
# Section 8: LM Head + Weight Loading
# =============================================================================


class Qwen3_5ForCausalLM(nn.Module, SupportsMRoPE):
    """Qwen3.5 model with LM head."""

    # The runner skips the host read of the previous output on a prefill-only
    # step, and merges the decode input_ids on the device after a batch change
    # (NeuronModelRunner._resolve_async_readback_skip).
    supports_async_readback_skip = True
    # The KV share of the HBM budget that leaves room for the DeltaNet state, the
    # snapshot rows and the NEFFs at TP=4 and TP=8 with decode batch [32]
    # (NeuronWorker._get_kv_cap_fraction). 0.30 runs out of device memory at TP=4.
    default_kv_cap_fraction = 0.23

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
        rotary_position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # In eager XLA mode, .to(dtype) can fail; positions used for indexing work as int64.
        try:
            positions = positions.to(torch.int32)
        except (RuntimeError, ValueError):
            pass

        first_attn_layer_name = (
            f"layers.{self.config.full_attention_interval - 1}.self_attn"
        )
        max_query_len = attn_metadata[first_attn_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_attn_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        T = input_ids.shape[0]
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt Length ({T}) must be > world_size ({self.world_size}) for SP."
            )

        batch_slot_indices = kwargs.get("batch_slot_indices", None)
        deltanet_padding_mask = kwargs.get("deltanet_padding_mask", None)
        deltanet_snapshot_rows = kwargs.get("deltanet_snapshot_rows", None)
        hidden_states = self.model(
            input_ids,
            positions,
            rotary_position_ids=rotary_position_ids,
            attn_metadata=attn_metadata,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
            batch_slot_indices=batch_slot_indices,
            deltanet_padding_mask=deltanet_padding_mask,
            deltanet_snapshot_rows=deltanet_snapshot_rows,
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

    def compiler_args_for(self, graph_type, example_inputs, base):
        """Compile every graph at ``-O2``.

        The XLA route defaults to ``-O1``, and ``-O1`` fails the decode graphs of
        batch 3 and more for this model. The native route already uses ``-O2``. An
        explicit ``--optimization-level`` wins, because the runner then does not
        call this policy.

        Example:
            >>> model.compiler_args_for("decode", None, ["-O1"])  # doctest: +SKIP
            ['-O2']
        """
        from vllm_neuron.utils.compiler_args_ctx import with_optimization_level

        return with_optimization_level(list(base or []), 2)

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = Qwen3_5Config.from_configs(hf_config, neuron_config)
        return cls(config)

    # ── SupportsMRoPE ─────────────────────────────────────────────────────

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list,
    ) -> tuple[torch.Tensor, int]:
        seq_len = len(input_tokens)
        if seq_len == 0:
            return torch.zeros((3, 0), dtype=torch.long), 0
        positions = (
            torch.arange(seq_len, dtype=torch.long)
            .unsqueeze(0)
            .expand(3, -1)
            .contiguous()
        )
        return positions, 0

    # ── KV Cache ──────────────────────────────────────────────────────────
    # Only full attention layers have KV caches.
    # DeltaNet layers use recurrent state (managed separately).

    def get_kv_spec(self):
        layers = []
        for i, layer in enumerate(self.model.layers):
            if layer.is_full_attention:
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
        # Name the DeltaNet layers too, so the framework allocates their state
        # and the KV cache manager knows the state exists. Both layer types live
        # at `layers.{i}.self_attn` in the module tree, and one layer is never
        # both types, so one name pattern serves both.
        stateful_layer_names = [
            f"layers.{i}.self_attn"
            for i, layer in enumerate(self.model.layers)
            if not layer.is_full_attention
        ]
        return HybridKVSpec(layers=layers, stateful_layer_names=stateful_layer_names)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config=None, **kwargs):
        """Return the shape of one DeltaNet state row, for each state tensor.

        The runner puts the page count in front of each shape, so these are the
        per-request shapes of one rank.

        Returns the layout that the NKI DeltaNet kernels already read:

            conv_state:      (qkv_dim_of_one_rank, conv_kernel - 1)
            recurrent_state: (value_heads_of_one_rank, key_head_dim, value_head_dim)

        The conv shape equals the upstream generic calculator under the DS conv
        layout, which puts the sharded dimension first. The recurrent shape does
        **not**: the upstream calculator returns (heads, value_dim, key_dim),
        which is the transpose of the last two axes. This port keeps its own
        order, because the kernels and the device tests use it. The two agree in
        size while `linear_key_head_dim == linear_value_head_dim`, and a test
        asserts that, so a config that breaks the equality fails loudly.

        Args:
            vllm_config: The vLLM config. Read the model config and the TP size
                from it.
            **kwargs: Accept `hf_config` and `tp_size` instead, for a test that
                holds no vLLM config.

        Returns:
            A tuple of two shapes: the conv state, then the recurrent state.

        Example:
            >>> import types
            >>> hf = types.SimpleNamespace(
            ...     linear_num_key_heads=16, linear_num_value_heads=48,
            ...     linear_key_head_dim=128, linear_value_head_dim=128,
            ...     linear_conv_kernel_dim=4)
            >>> Qwen3_5ForCausalLM.get_mamba_state_shape_from_config(
            ...     hf_config=hf, tp_size=8)
            ((1280, 3), (6, 128, 128))
        """
        if vllm_config is not None:
            hf = vllm_config.model_config.hf_text_config
            tp_size = int(vllm_config.parallel_config.tensor_parallel_size)
        else:
            hf = kwargs["hf_config"]
            tp_size = int(kwargs.get("tp_size", 1))

        key_dim = int(hf.linear_key_head_dim)
        value_dim = int(hf.linear_value_head_dim)
        num_key_heads_per_rank = int(hf.linear_num_key_heads) // tp_size
        num_value_heads_per_rank = int(hf.linear_num_value_heads) // tp_size
        conv_kernel = int(hf.linear_conv_kernel_dim)

        qkv_dim_per_rank = (
            num_key_heads_per_rank * key_dim * 2 + num_value_heads_per_rank * value_dim
        )
        conv_state_shape = (qkv_dim_per_rank, conv_kernel - 1)
        recurrent_state_shape = (num_value_heads_per_rank, key_dim, value_dim)
        return (conv_state_shape, recurrent_state_shape)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config) -> tuple[torch.dtype, ...]:
        """Return the dtype of the conv state and of the recurrent state.

        Both states use the dtype of the model. The port holds the recurrent
        state in the model dtype and casts to float32 inside the recurrence, so
        the stored dtype stays the model dtype.

        Args:
            vllm_config: The vLLM config.

        Returns:
            The dtype of the conv state, then the dtype of the recurrent state.

        Example:
            >>> import types
            >>> cfg = types.SimpleNamespace(
            ...     model_config=types.SimpleNamespace(dtype=torch.bfloat16))
            >>> Qwen3_5ForCausalLM.get_mamba_state_dtype_from_config(cfg)
            (torch.bfloat16, torch.bfloat16)
        """
        dtype = vllm_config.model_config.dtype
        return (dtype, dtype)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
        for i, layer in enumerate(self.model.layers):
            if layer.is_full_attention:
                layer_name = f"layers.{i}.self_attn"
                if layer_name not in kv_caches:
                    raise KeyError(f"KV cache for layer {layer_name} not initialized")
                layer.self_attn.k_cache = kv_caches[layer_name][0]
                layer.self_attn.v_cache = kv_caches[layer_name][1]

    # ── Weight Loading ────────────────────────────────────────────────────

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load weights from a Qwen3.5-family (Qwen3.6/3.8-27B) HF checkpoint.

        The HF checkpoint uses model.language_model.* prefix for text weights.
        Full attention q_proj is doubled (query+gate interleaved per head) and
        must be split at load time.

        DeltaNet layers use 'linear_attn' submodule (not 'self_attn').
        """
        tp_rank = self.rank
        tp_size = self.world_size

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        # Detect prefix from the safetensors index when there is one (a sharded checkpoint,
        # which is what the 27B release ships).
        import glob as _glob
        import json as _json

        _index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
        _names: list[str] = []
        if os.path.exists(_index_path):
            with open(_index_path) as _f:
                _names = list(_json.load(_f).get("weight_map", {}))
        else:
            # No index: an unsharded checkpoint. The previous fallback read
            # `checkpoint._tensor_name_to_file`, which SafetensorsCheckpoint fills in LAZILY
            # during pipelined loading and is therefore still empty here -- so the prefix
            # silently resolved to "model" and loading died on the first mapped tensor with
            # `KeyError: model.layers.<N>.self_attn.q_proj.weight`. Read the file headers
            # instead; safe_open parses only the header, so this costs no tensor I/O.
            from safetensors import safe_open as _safe_open

            for _f in sorted(
                _glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
            ):
                try:
                    with _safe_open(_f, framework="pt") as _h:
                        _names.extend(_h.keys())
                except Exception as _e:  # noqa: BLE001 - a bad shard must not mask the real error
                    logger.warning(
                        "could not read %s while detecting the weight prefix: %s",
                        _f,
                        _e,
                    )
        has_language_model = any("language_model" in k for k in _names)
        if not _names:
            # Nothing to detect from. Say so rather than defaulting silently: picking the wrong
            # prefix fails much later, as a missing-key error on an unrelated-looking tensor.
            logger.warning(
                "no weight names found in %s (no index and no readable *.safetensors); "
                "assuming the flat 'model.*' layout, which will fail on a "
                "'model.language_model.*' checkpoint",
                checkpoint_path,
            )
        if has_language_model:
            hf_prefix = "model.language_model"
        else:
            hf_prefix = "model"
        logger.info(
            "qwen3_5 weight prefix resolved to %r from %d weight name(s)",
            hf_prefix,
            len(_names),
        )

        mappings = {}

        # Embedding and final norm
        mappings["model.embed_tokens.weight"] = f"{hf_prefix}.embed_tokens.weight"
        mappings["model.norm.weight"] = f"{hf_prefix}.norm.weight"

        # LM head
        if self.config.tie_word_embeddings:
            mappings["lm_head.weight"] = f"{hf_prefix}.embed_tokens.weight"
        else:
            mappings["lm_head.weight"] = "lm_head.weight"

        for layer_id in range(len(self.model.layers)):
            prefix = f"model.layers.{layer_id}"
            hf_layer = f"{hf_prefix}.layers.{layer_id}"
            layer = self.model.layers[layer_id]

            # Layer norms (same for both layer types)
            mappings[f"{prefix}.input_layernorm.weight"] = (
                f"{hf_layer}.input_layernorm.weight"
            )
            mappings[f"{prefix}.post_attention_layernorm.weight"] = (
                f"{hf_layer}.post_attention_layernorm.weight"
            )

            # MLP (same for both layer types)
            mappings[f"{prefix}.mlp.gate_proj_weight"] = (
                f"{hf_layer}.mlp.gate_proj.weight"
            )
            mappings[f"{prefix}.mlp.up_proj_weight"] = f"{hf_layer}.mlp.up_proj.weight"
            mappings[f"{prefix}.mlp.down_proj_weight"] = (
                f"{hf_layer}.mlp.down_proj.weight"
            )

            if layer.is_full_attention:
                # Full attention: q_proj, k_proj, v_proj, gate are handled in
                # post-processing. Only map o_proj and norms here.
                mappings[f"{prefix}.self_attn.o_proj_weight"] = (
                    f"{hf_layer}.self_attn.o_proj.weight"
                )
                mappings[f"{prefix}.self_attn.q_norm.weight"] = (
                    f"{hf_layer}.self_attn.q_norm.weight"
                )
                mappings[f"{prefix}.self_attn.k_norm.weight"] = (
                    f"{hf_layer}.self_attn.k_norm.weight"
                )
            else:
                # DeltaNet: uses 'linear_attn' submodule in HF checkpoint
                hf_la = f"{hf_layer}.linear_attn"
                mappings[f"{prefix}.self_attn.qkv_proj_weight"] = (
                    f"{hf_la}.in_proj_qkv.weight"
                )
                mappings[f"{prefix}.self_attn.z_proj_weight"] = (
                    f"{hf_la}.in_proj_z.weight"
                )
                mappings[f"{prefix}.self_attn.a_proj_weight"] = (
                    f"{hf_la}.in_proj_a.weight"
                )
                mappings[f"{prefix}.self_attn.b_proj_weight"] = (
                    f"{hf_la}.in_proj_b.weight"
                )
                mappings[f"{prefix}.self_attn.a_log"] = f"{hf_la}.A_log"
                mappings[f"{prefix}.self_attn.dt_bias"] = f"{hf_la}.dt_bias"
                mappings[f"{prefix}.self_attn.conv_weight"] = f"{hf_la}.conv1d.weight"
                mappings[f"{prefix}.self_attn.group_norm.weight"] = (
                    f"{hf_la}.norm.weight"
                )
                mappings[f"{prefix}.self_attn.o_proj_weight"] = (
                    f"{hf_la}.out_proj.weight"
                )

        load_device = torch.device("cpu")
        rank_sharded = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, mappings, load_device, strict=False
        ).state_dict

        # Post-processing: Split full attention q_proj into query + gate,
        # load K/V directly, fuse into qkv_proj_weight

        num_heads = self.config.num_attention_heads
        head_dim = self.config.head_dim
        num_kv_heads = self.config.num_key_value_heads
        heads_per_rank = num_heads // tp_size

        def _load_checkpoint_tensor(key: str) -> torch.Tensor:
            file_path = checkpoint._tensor_name_to_file[key]
            sf = checkpoint._open_safetensor_files[file_path]
            return sf.get_tensor(key)

        for layer_id in range(len(self.model.layers)):
            layer = self.model.layers[layer_id]
            if not layer.is_full_attention:
                continue

            prefix = f"model.layers.{layer_id}"
            hf_layer = f"{hf_prefix}.layers.{layer_id}"

            # Load q_proj: [12288, 5120] = [num_heads * head_dim * 2, hidden]
            q_proj_w = _load_checkpoint_tensor(f"{hf_layer}.self_attn.q_proj.weight")
            q_proj_w = q_proj_w.reshape(num_heads, head_dim * 2, -1)
            query_w = q_proj_w[:, :head_dim, :]  # [num_heads, head_dim, hidden]
            gate_w = q_proj_w[:, head_dim:, :]  # [num_heads, head_dim, hidden]

            # TP shard query and gate by heads
            start_head = tp_rank * heads_per_rank
            end_head = start_head + heads_per_rank
            query_w_sharded = query_w[start_head:end_head].reshape(
                heads_per_rank * head_dim, -1
            )  # [q_size_per_rank, hidden]
            gate_w_sharded = gate_w[start_head:end_head].reshape(
                heads_per_rank * head_dim, -1
            )  # [gate_size_per_rank, hidden]

            # Load K and V: [kv_heads * head_dim, hidden]
            k_w = _load_checkpoint_tensor(f"{hf_layer}.self_attn.k_proj.weight")
            v_w = _load_checkpoint_tensor(f"{hf_layer}.self_attn.v_proj.weight")

            # TP shard K/V (replicate if world_size > num_kv_heads)
            attn = layer.self_attn
            if tp_size >= num_kv_heads:
                kv_rank = tp_rank // attn.num_kv_replicas
            else:
                kv_rank = tp_rank
            kv_heads_per_rank = attn.num_key_value_heads_per_rank
            kv_start = kv_rank * kv_heads_per_rank * head_dim
            kv_end = kv_start + kv_heads_per_rank * head_dim
            k_w_sharded = k_w[kv_start:kv_end]  # [kv_size_per_rank, hidden]
            v_w_sharded = v_w[kv_start:kv_end]

            # Fuse Q+K+V: our param is [hidden, qkv_size] (transposed from HF)
            qkv_w = torch.cat(
                [query_w_sharded, k_w_sharded, v_w_sharded], dim=0
            ).T  # [hidden, q+k+v per rank]
            rank_sharded[f"{prefix}.self_attn.qkv_proj_weight"] = qkv_w

            # Gate: our param is [hidden, gate_size] (transposed from HF)
            rank_sharded[f"{prefix}.self_attn.gate_proj_weight"] = gate_w_sharded.T

        # Post-processing: Reorder DeltaNet QKV and conv weights for TP sharding.
        # HF stores in_proj_qkv as [Q_all | K_all | V_all, hidden]. The simple
        # sharding_weight_loader slices contiguously, giving rank 0 only Q rows.
        # We need each rank to have [local_Q | local_K | local_V].
        if tp_size > 1:
            num_k_heads = self.config.linear_num_key_heads  # 16
            num_v_heads = self.config.linear_num_value_heads  # 48
            head_k_dim = self.config.linear_key_head_dim  # 128
            head_v_dim = self.config.linear_value_head_dim  # 128
            key_dim_total = num_k_heads * head_k_dim  # 2048
            value_dim_total = num_v_heads * head_v_dim  # 6144
            local_k_heads = num_k_heads // tp_size  # 2
            local_v_heads = num_v_heads // tp_size  # 6

            for layer_id in range(len(self.model.layers)):
                layer = self.model.layers[layer_id]
                if layer.is_full_attention:
                    continue

                prefix = f"model.layers.{layer_id}"
                qkv_key = f"{prefix}.self_attn.qkv_proj_weight"
                conv_key = f"{prefix}.self_attn.conv_weight"

                # Reload full QKV weight and reorder
                hf_layer = f"{hf_prefix}.layers.{layer_id}"
                hf_la = f"{hf_layer}.linear_attn"
                qkv_full = _load_checkpoint_tensor(f"{hf_la}.in_proj_qkv.weight")
                # qkv_full shape: [10240, 5120] = [Q(2048)+K(2048)+V(6144), hidden]
                q_w = qkv_full[:key_dim_total].reshape(num_k_heads, head_k_dim, -1)
                k_w = qkv_full[key_dim_total : 2 * key_dim_total].reshape(
                    num_k_heads, head_k_dim, -1
                )
                v_w = qkv_full[
                    2 * key_dim_total : 2 * key_dim_total + value_dim_total
                ].reshape(num_v_heads, head_v_dim, -1)

                # Extract this rank's heads
                q_local = q_w[
                    tp_rank * local_k_heads : (tp_rank + 1) * local_k_heads
                ].reshape(-1, qkv_full.shape[1])
                k_local = k_w[
                    tp_rank * local_k_heads : (tp_rank + 1) * local_k_heads
                ].reshape(-1, qkv_full.shape[1])
                v_local = v_w[
                    tp_rank * local_v_heads : (tp_rank + 1) * local_v_heads
                ].reshape(-1, qkv_full.shape[1])

                # Fuse as [local_Q | local_K | local_V] then transpose for our param layout [hidden, qkv_per_rank]
                qkv_reordered = torch.cat([q_local, k_local, v_local], dim=0).T
                rank_sharded[qkv_key] = qkv_reordered

                # Reorder conv1d weight: [10240, 1, 4] → per-rank [qkv_per_rank, 1, 4]
                conv_full = _load_checkpoint_tensor(f"{hf_la}.conv1d.weight")
                # conv_full shape: [10240, 1, 4] — channels in same Q|K|V order
                q_conv = conv_full[:key_dim_total].reshape(
                    num_k_heads, head_k_dim, *conv_full.shape[1:]
                )
                k_conv = conv_full[key_dim_total : 2 * key_dim_total].reshape(
                    num_k_heads, head_k_dim, *conv_full.shape[1:]
                )
                v_conv = conv_full[
                    2 * key_dim_total : 2 * key_dim_total + value_dim_total
                ].reshape(num_v_heads, head_v_dim, *conv_full.shape[1:])

                q_conv_local = q_conv[
                    tp_rank * local_k_heads : (tp_rank + 1) * local_k_heads
                ].reshape(-1, *conv_full.shape[1:])
                k_conv_local = k_conv[
                    tp_rank * local_k_heads : (tp_rank + 1) * local_k_heads
                ].reshape(-1, *conv_full.shape[1:])
                v_conv_local = v_conv[
                    tp_rank * local_v_heads : (tp_rank + 1) * local_v_heads
                ].reshape(-1, *conv_full.shape[1:])

                conv_reordered = torch.cat(
                    [q_conv_local, k_conv_local, v_conv_local], dim=0
                )
                rank_sharded[conv_key] = conv_reordered

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self.load_state_dict(rank_sharded, strict=False, assign=True)

        # Materialize any remaining meta tensors (e.g. recurrent_state,
        # conv_state) that are runtime buffers not loaded from checkpoint.
        for name, param in self.named_parameters():
            if param.is_meta:
                self._parameters_to_materialize = True
                materialized = torch.zeros(param.shape, dtype=param.dtype, device="cpu")
                # Walk the module tree to assign
                parts = name.split(".")
                mod = self
                for part in parts[:-1]:
                    mod = getattr(mod, part)
                setattr(
                    mod,
                    parts[-1],
                    torch.nn.Parameter(materialized, requires_grad=False),
                )
        for name, buf in self.named_buffers():
            if buf.is_meta:
                parts = name.split(".")
                mod = self
                for part in parts[:-1]:
                    mod = getattr(mod, part)
                mod._buffers[parts[-1]] = torch.zeros(
                    buf.shape, dtype=buf.dtype, device="cpu"
                )
