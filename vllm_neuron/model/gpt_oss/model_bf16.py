# SPDX-License-Identifier: Apache-2.0
"""
GPT-OSS BF16 Implementation
======================================

Annotated implementation of GPT-OSS for the Neuron backend.
Designed as the reference model for AI-assisted bring-up of new architectures.

See DESIGN.md for full documentation of parallelism strategies and porting guide.

Supported parallelism: TP, SP, DP, EP, and DP+EP combinations.
Starting model: GPT-OSS-20B (32 experts, top-4, GQA 64Q/8KV), scalable to 120B.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    GPT-OSS specific. Change when porting.

PARALLELISM SHARDING:
  TP-only:  Attention heads sharded, MoE intermediate sharded, all experts on all ranks
  TP+DP:    Independent replicas per DP rank, same sharding as TP-only
  TP+EP:    Attention heads sharded, experts partitioned across ranks (linear placement)
  TP+DP+EP: Cross-DP token exchange for MoE (dispatch/combine), attention is DP-local
"""

import logging
import math

import nki.language as nl
import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.models.interfaces import SupportsEagle3

import vllm_neuron.functional as NF
from vllm_neuron.functional.attention.attention_decode import (
    _swizzle_packed_k,
    _unswizzle_packed_k,
)
from vllm_neuron.functional.attention.attention_decode_mask import _resize_block_len

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.dtype_utils import (
    FP8_CLAMP_MAX,
    validate_fp8_segmented_supported,
)
from vllm_neuron.utils.weight_loader import set_weight_loader, with_rank_override

from transformers import PretrainedConfig
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.sampler import Sampler
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    RouterActFnType,
)

from nkilib.core.moe.moe_cte.moe_cte import (
    MoECTEImplementation,
)

import vllm_neuron.nn as neuron_nn
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.vllm.spec_decode.decorator import async_speculative_decoding

from .config import GptOssConfig

# ── Weight Loaders ────────────────────────────────────────────────────────────
# Reuse existing weight loaders. These handle checkpoint format transformations
# (MXFP4 dequantization, TP sharding, hidden dim padding).
from vllm_neuron.model.gpt_oss.weight_loaders_bf16 import (
    fused_qkv_weight_loader,
    fused_qkv_bias_loader,
    o_proj_weight_loader,
    expert_gate_up_weight_sharding_loader,
    expert_gate_up_bias_sharding_loader,
    expert_down_weight_sharding_loader,
)
from vllm_neuron.utils.weight_loader import (
    expert_parallel_tensor_dim_loader,
    last_dim_padding_weight_loader,
    scaled_bias_loader,
)

# Threshold for decode MoE kernel selection
DEFAULT_SELECTIVE_LOADING_THRESHOLD = 1.0

logger = logging.getLogger(__name__)


def _packed_fp8_viable_for_bucket(
    block_len: int, bs: int, q_head: int, s_active: int, s_prior: int
) -> bool:
    """Whether the packed FP8 decode kernel is usable for this bucket geometry.

    The decode kernel resizes block_len so blocks_per_batch is a multiple of
    (lnc * p_max). The packed layout pairs two consecutive tokens per BF16 slot,
    so it needs the resized block_len to stay >= 2; some bucket geometries
    (small SWA windows, or batch=1 which forces s_prior sharding) resize it down
    to 1. Mirror the kernel's resize math (shared replica) to decide statically,
    per compiled decode NEFF, whether to read the packed cache directly or
    un-swizzle it to the standard layout first.
    """
    if block_len <= 0 or s_prior <= 0:
        return False
    return _resize_block_len(block_len, bs, q_head, s_active, s_prior) >= 2


# =============================================================================
# Section 1: RMS Normalization
# <-- MODEL-SPECIFIC: GPT-OSS uses RMSNorm with variance on unpadded portion
# =============================================================================
class GptOssRMSNorm(nn.Module):
    """RMS Normalization with support for padded hidden dimensions.

    <-- MODEL-SPECIFIC: GPT-OSS pads hidden_size to 3072 for hardware alignment.
    The variance is computed only on the unpadded portion, and the padded positions
    are zeroed after normalization.
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=config.torch_dtype)
        )
        self.unpadded_hidden_size = config.unpadded_hidden_size
        self.hidden_size = config.hidden_size
        self.variance_epsilon = config.rms_norm_eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        hidden_states_clone = hidden_states.clone()
        # <-- MODEL-SPECIFIC: Variance only on unpadded portion
        variance = (
            hidden_states_clone[..., : self.unpadded_hidden_size]
            .pow(2)
            .mean(-1, keepdim=True)
        )
        hidden_states_clone[..., : self.unpadded_hidden_size] = hidden_states_clone[
            ..., : self.unpadded_hidden_size
        ] * torch.rsqrt(variance + self.variance_epsilon)
        output = self.weight * hidden_states_clone
        # <-- MODEL-SPECIFIC: Zero padded positions
        output[..., self.unpadded_hidden_size :] = 0.0

        return output.to(input_dtype)


# =============================================================================
# Section 2: Rotary Position Embedding (YaRN)
# <-- MODEL-SPECIFIC: GPT-OSS uses YaRN scaling for RoPE
# =============================================================================
class GptOssRotaryEmbedding(nn.Module):
    """Rotary Position Embedding with YaRN scaling.

    <-- MODEL-SPECIFIC: The YaRN interpolation/extrapolation logic, concentration
    factor, and beta parameters are specific to GPT-OSS. Other models may use
    standard RoPE, NTK-aware RoPE, or different scaling methods.
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_parameters["rope_theta"]
        rope_parameters = config.rope_parameters or {}
        self.scaling_factor = rope_parameters.get("factor", 1.0)
        self.beta_slow = rope_parameters.get("beta_slow", 1.0)
        self.beta_fast = rope_parameters.get("beta_fast", 32.0)
        self.initial_context_length = rope_parameters.get(
            "original_max_position_embeddings", 8192
        )

        inv_freq, concentration = self._compute_inv_freq_and_concentration("cpu")
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("concentration", concentration, persistent=False)

    def _compute_inv_freq_and_concentration(
        self, device: torch.device
    ) -> tuple[torch.Tensor, float]:
        """<-- MODEL-SPECIFIC: YaRN frequency computation with interpolation mask."""
        freq = self.rope_theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
            / self.head_dim
        )

        concentration = 0.1 * math.log(self.scaling_factor) + 1.0

        d_half = self.head_dim / 2
        low = (
            d_half
            * math.log(self.initial_context_length / (self.beta_fast * 2 * math.pi))
            / math.log(self.rope_theta)
        )
        high = (
            d_half
            * math.log(self.initial_context_length / (self.beta_slow * 2 * math.pi))
            / math.log(self.rope_theta)
        )

        interpolation = 1.0 / (self.scaling_factor * freq)
        extrapolation = 1.0 / freq

        ramp = (torch.arange(d_half, dtype=torch.float32, device=device) - low) / (
            high - low
        )
        mask = 1 - ramp.clamp(0, 1)

        inv_freq = interpolation * (1 - mask) + extrapolation * mask
        concentration = torch.tensor(concentration, device=device)
        return inv_freq, concentration

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin embeddings for given positions.

        Args:
            position_ids: [T] tensor of position indices
            device: target device
            dtype: target dtype

        Returns:
            cos, sin: both of shape [T, head_dim/2]
        """
        inv_freq_expanded = self.inv_freq[None, :].float()  # [1, head_dim/2]
        position_ids_expanded = position_ids[:, None].float()  # [T, 1]

        freqs = position_ids_expanded @ inv_freq_expanded  # [T, head_dim/2]
        cos = freqs.cos() * self.concentration
        sin = freqs.sin() * self.concentration

        return cos.to(dtype=dtype), sin.to(dtype=dtype)


# =============================================================================
# Section 3: Attention
# Mixed: PARALLELISM (TP head sharding, SP, collectives) +
#        MODEL-SPECIFIC (sinks, sliding window, GQA, RoPE application)
# =============================================================================
# NOTE: RoPE is fused into NF.qkv_proj for prefill; decode handles RoPE in
# its own dedicated kernel call. The standalone `_apply_rotary_emb` /
# `apply_rotary_pos_emb` helpers are no longer used by either path.


def _replicated_sinks_loader(
    shard_size: int,
    num_shards: int,
    attention_dp_size: int,
    tp_rank: int,
):
    """Load sinks replicated across attention DP ranks.

    Each DP rank within the same TP position has a different shard of sinks.
    This loader concatenates all DP shards for the given TP position so the
    decode path doesn't need a runtime collective.

    Returns a tensor of size [shard_size * attention_dp_size].
    """
    from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

    ddp = attention_dp_size

    def transform(slices, rank):
        assert len(slices) == 1
        slice_obj = slices[0]
        shards = []
        for dp_rank in range(ddp):
            effective_rank = dp_rank + tp_rank * ddp
            start = (effective_rank % num_shards) * shard_size
            end = start + shard_size
            shards.append(slice_obj[start:end])
        return torch.cat(shards, dim=0).to(torch.float32)

    return SafetensorsWeightLoader(transform=transform)


class GptOssAttention(nn.Module):
    """Multi-head attention with TP head sharding.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads are sharded across TP ranks
    - KV heads are replicated when fewer than TP size (GQA)
    - Prefill: all-gather input → QKV proj → attention → O proj → reduce-scatter
    - Decode: fused megakernel with TP all-reduce

    <-- MODEL-SPECIFIC:
    - GQA with separate Q and KV head counts
    - Learnable attention sinks (per-head logit bias)
    - Sliding window on even-indexed layers
    - YaRN RoPE (externally computed)
    """

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.rms_norm_eps = config.rms_norm_eps
        self.hidden_size = config.hidden_size
        self.unpadded_hidden_size = config.unpadded_hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5
        self.max_seq_len = config.max_position_embeddings

        # <-- MODEL-SPECIFIC: Sliding window on alternating (even) layers
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else None

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: Attention DP setup <<<
        self.attention_dp_size = (
            config.neuron_config.attention_dp_size if config.neuron_config else 1
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_attention_dp_group,
            get_neuron_attention_dp_rank,
        )

        self.attention_dp_group = get_neuron_attention_dp_group()
        self.attention_dp_rank = get_neuron_attention_dp_rank()

        # >>> PARALLELISM: Attention TP group (TP * attn_dp) <<<
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_attention_tp_group,
        )

        self.attn_tp_group = get_neuron_attention_tp_group()

        # Effective sharding degree for Q/O (TP for standard, TP*DDP for attention DP)
        effective_q_shards = self.world_size * self.attention_dp_size

        # >>> PARALLELISM: Head sharding calculation <<<
        # Q heads divided evenly across TP * attention DP ranks
        self.num_attention_heads_per_rank = (
            self.num_attention_heads // effective_q_shards
        )

        # KV heads: cache sizing (always full TP amount)
        self.kv_needs_a2a = (
            self.attention_dp_size > 1
            and self.num_key_value_heads > self.world_size
            and self.num_key_value_heads % effective_q_shards == 0
        )

        if self.world_size >= self.num_key_value_heads:
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            self.num_key_value_heads_per_rank = (
                self.num_key_value_heads // self.world_size
            )
            self.num_kv_replicas = 1

        # KV heads-per-rank used in the QKV projection output. When
        # kv_needs_a2a, attention DP shards KV further than world_size, so
        # the projection emits a smaller K/V block than `num_key_value_heads_per_rank`.
        # Stored on self so the prefill path can pass the correct value to
        # NF.qkv_proj's fused-RoPE (which uses num_kv_heads to delimit the
        # K block within the projection output).
        self.num_kv_heads_for_weight = (
            self.num_key_value_heads // effective_q_shards
            if self.kv_needs_a2a
            else self.num_key_value_heads_per_rank
        )
        num_kv_heads_for_weight = self.num_kv_heads_for_weight  # local alias

        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // num_kv_heads_for_weight
        )

        # Q/KV heads after all-to-all
        self.num_q_heads_after_a2a = (
            self.num_attention_heads_per_rank * self.attention_dp_size
        )
        self.num_kv_heads_after_a2a = (
            num_kv_heads_for_weight * self.attention_dp_size
            if self.kv_needs_a2a
            else self.num_key_value_heads_per_rank
        )

        # >>> PARALLELISM: QKV weight shapes for TP * attention DP <<<
        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = num_kv_heads_for_weight * self.head_dim
        qkv_size = q_size + 2 * kv_size
        o_proj_in_features = (
            self.num_attention_heads * self.head_dim
        ) // effective_q_shards

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size, dtype=self.dtype)
        )
        # Conditionally create bias parameters based on config
        if config.attention_bias:
            self.qkv_proj_bias = nn.Parameter(torch.zeros(qkv_size, dtype=self.dtype))
            self.o_proj_bias = nn.Parameter(torch.zeros(self.hidden_size, dtype=self.dtype))
        else:
            # Register as None so state_dict doesn't expect them
            self.register_parameter('qkv_proj_bias', None)
            self.register_parameter('o_proj_bias', None)
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_proj_in_features, self.hidden_size, dtype=self.dtype)
        )

        # <-- MODEL-SPECIFIC: Learnable attention sinks (per-head)
        # Pre-gathered across attention DP so no runtime all_gather is needed.
        self.sinks = nn.Parameter(
            torch.zeros(self.num_q_heads_after_a2a, dtype=torch.float32)
        )

        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        # KV caches bound externally via bind_kv_cache()
        self.k_cache = None
        self.v_cache = None

        # When the K cache uses the swizzled packed FP8 layout
        # ([num_blocks, (kv_heads,) block_len // 2, d_head, 2]) the attention
        # kernel reads it via bf16-reinterpret + DMA-transpose, and writes are
        # done in the packed layout. Detected from cache rank (K has one extra
        # trailing dim vs V, which is never packed).
        self.fp8_packed = False

        # KV cache quantization scales are set during weight loading.
        # We also need floats since tensor.item() causes a graph break,
        # and kernels currently use floats for perf reasons.
        self.register_buffer("k_scale", None, persistent=False)
        self.register_buffer("v_scale", None, persistent=False)
        self.k_scale_float = 1.0
        self.v_scale_float = 1.0

        # Set up weight loaders for checkpoint loading
        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach weight loaders for checkpoint → parameter transformation.

        >>> PARALLELISM: Weight loaders handle TP (and attention DP) sharding.
        <-- MODEL-SPECIFIC: The fused QKV layout, hidden dim padding, and
        checkpoint tensor naming are model-specific.

        With attention DP: Q/O are sharded across TP*attention DP using an interleaved
        effective rank. KV may also be sharded when kv_needs_a2a.
        """
        ddp = self.attention_dp_size
        effective_q_shards = self.world_size * ddp
        effective_q_rank = self.attention_dp_rank + self.tp_group.rank_in_group * ddp

        qkv_loader = fused_qkv_weight_loader(
            q_size=self.q_size,
            kv_size=self.kv_size,
            shard_dim=1,
            num_shards=effective_q_shards,
            num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            hidden_size=self.hidden_size,
            num_kv_replicas=self.num_kv_replicas,
            kv_num_shards=self.world_size if not self.kv_needs_a2a else None,
        )
        qkv_loader = with_rank_override(qkv_loader, rank=effective_q_rank)
        set_weight_loader(self.qkv_proj_weight, qkv_loader)

        # Only register bias loader if bias parameters exist
        if self.qkv_proj_bias is not None:
            qkv_bias_loader = fused_qkv_bias_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
                num_shards=effective_q_shards,
                num_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                num_kv_replicas=self.num_kv_replicas,
                kv_num_shards=self.world_size if not self.kv_needs_a2a else None,
            )
            qkv_bias_loader = with_rank_override(qkv_bias_loader, rank=effective_q_rank)
            set_weight_loader(self.qkv_proj_bias, qkv_bias_loader)

        o_loader = o_proj_weight_loader(
            shard_size=(self.num_attention_heads * self.head_dim) // effective_q_shards,
            num_shards=effective_q_shards,
            hidden_size=self.hidden_size,
        )
        o_loader = with_rank_override(o_loader, rank=effective_q_rank)
        set_weight_loader(self.o_proj_weight, o_loader)

        o_bias_loader = scaled_bias_loader(
            scale=effective_q_shards, padded_size=self.hidden_size
        )
        o_bias_loader = with_rank_override(o_bias_loader, rank=effective_q_rank)
        set_weight_loader(self.o_proj_bias, o_bias_loader)

        sinks_loader = _replicated_sinks_loader(
            shard_size=self.num_attention_heads_per_rank,
            num_shards=effective_q_shards,
            attention_dp_size=self.attention_dp_size,
            tp_rank=self.tp_group.rank_in_group,
        )
        set_weight_loader(self.sinks, sinks_loader)

    # ── Forward dispatch ─────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        attn_mask=None,
    ):
        """Dispatch to prefill or decode path based on metadata.

        >>> PARALLELISM: Dispatch logic <<<
        - Prefill: all-gather for SP before attention, reduce-scatter after
        - Decode: fused megakernel handles TP internally
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states,
                positions,
                position_embeddings,
                attn_mask,
                attn_metadata,
            )
        else:
            # >>> PARALLELISM: All-gather from SP before attention <<<
            if self.world_size > 1:
                hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

            return self.forward_prefill(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
            )

    def _write_paged_kv_cache(self, k, v, slot_mapping, block_size):
        """Scatter post-RoPE K/V into the paged cache at slot_mapping positions.

        Used on the prefill paths where the cache write is NOT folded into the
        qkv kernel (full prefill needs raw K/V for flash attention; packed-FP8
        segmented prefill cannot use the in-kernel write). FP8 caches store
        ``fp8(clamp(tensor * scale))``; bf16 caches store directly. V is never
        packed; packed K is un-swizzled, scattered, then re-swizzled in place.
        """
        if self.k_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            k_flat = (
                (k.reshape(-1, self.head_dim) * self.k_scale)
                .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                .to(self.k_cache.dtype)
            )
            v_flat = (
                (v.reshape(-1, self.head_dim) * self.v_scale)
                .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                .to(self.k_cache.dtype)
            )
        else:
            k_flat = k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
            v_flat = v.reshape(-1, self.head_dim).to(self.k_cache.dtype)

        nkh = self.num_key_value_heads_per_rank
        block_indices = (slot_mapping // block_size).repeat(nkh)
        position_indices = (slot_mapping % block_size).repeat(nkh)
        head_indices = torch.arange(
            nkh, dtype=torch.long, device=k.device
        ).repeat_interleave(slot_mapping.shape[0])
        index = (block_indices, head_indices, position_indices)

        # V is never packed → scatters directly.
        self.v_cache.index_put_(index, v_flat)
        if self.fp8_packed:
            # Packed K cache [blocks, Nkh, block_size // 2, Dh, 2]: the swizzle
            # interleaves adjacent token positions into the trailing size-2 dim,
            # so a per-token scatter isn't expressible directly. Un-swizzle to
            # the standard layout, scatter, then re-swizzle back in place
            # (matching the decode kernel's write).
            k_unpacked = _unswizzle_packed_k(self.k_cache)
            k_unpacked.index_put_(index, k_flat)
            self.k_cache.copy_(_swizzle_packed_k(k_unpacked))
        else:
            self.k_cache.index_put_(index, k_flat)

    # ── Prefill path ─────────────────────────────────────────────────────

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """Prefill: full-sequence attention with flash attention.

        Pipeline:
        1. QKV projection          >>> PARALLELISM: TP sharded heads <<<
        2. RoPE                    <-- MODEL-SPECIFIC: YaRN RoPE
        3. KV cache update         >>> PARALLELISM: per-rank cache <<<
        4. Flash attention          <-- MODEL-SPECIFIC: sinks + sliding window
        5. Output projection       >>> PARALLELISM: reduce-scatter after O proj <<<
        """
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # ── Step 1: QKV Projection (with fused RoPE) ─────────────────────
        # <-- MODEL-SPECIFIC: YaRN/NTK RoPE — fused into the qkv_proj kernel.
        # GptOssRotaryEmbedding emits cos/sin of shape [T, d_head/2]; the
        # kernel expects [B, T, d_head] for split-in-half RoPE. cat-double
        # makes the second half match the first so the kernel's per-half
        # math is equivalent to gpt-oss's non-interleaved RoPE.
        cos, sin = position_embeddings
        cos_cache = torch.cat([cos, cos], dim=-1).unsqueeze(0)
        sin_cache = torch.cat([sin, sin], dim=-1).unsqueeze(0)

        # ── KV cache metadata (pulled up so segmented branch can fold the
        # post-RoPE K/V cache write into the qkv kernel). ──
        # >>> PARALLELISM: Cache is per-rank (TP sharded KV heads) <<<
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        kv_is_fp8 = self.k_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]

        # ── Step 3+4: KV cache write + Attention ──────────────────────────
        # <-- MODEL-SPECIFIC: sinks, sliding window
        if kv_segment_size:
            # attention_segmented_cte cannot read a non-packed FP8 K cache;
            # fail fast on the unsupported combo (see helper for rationale).
            validate_fp8_segmented_supported(kv_is_fp8, self.fp8_packed)

            # bf16 (non-packed): fold the post-RoPE K/V cache write into the qkv
            # kernel (in-kernel write: must_alias output → FX aliasing pass
            # threads back to self.k_cache / self.v_cache for the downstream
            # segmented_attention). The qkv kernel cannot emit the swizzled
            # packed-FP8 K layout, so packed FP8 falls back to a plain projection
            # + explicit (un)swizzled scatter.
            if not self.fp8_packed:
                # Sanitize slot_mapping for the in-kernel scatter: the qkv NKI
                # kernel uses slot_mapping values as direct DMA offsets into the
                # cache, with no oob_mode.skip on this path — out-of-range slots
                # cause a hardware OOB. Remap sentinels (< 0) and stale values
                # (>= num_blocks * block_size) to slot 0; the K/V they would
                # have written are unused (padding tokens never read), so the
                # harmless write is acceptable. Mirrors PR #2306's decode path.
                num_blocks_total = self.k_cache.shape[0]
                max_slot = num_blocks_total * block_size
                slot_mapping_clamped = torch.where(
                    (slot_mapping < 0) | (slot_mapping >= max_slot),
                    torch.zeros_like(slot_mapping),
                    slot_mapping,
                ).to(torch.int32)

                # In-kernel write is bf16-only (guarded above): FP8 segmented
                # must be packed, which takes the explicit-scatter branch below.
                q_hbm, _, _ = NF.qkv_proj(
                    hidden=hidden_states.unsqueeze(0),
                    qkv_weights=self.qkv_proj_weight,
                    bias=self.qkv_proj_bias.unsqueeze(0) if self.qkv_proj_bias is not None else None,
                    d_head=self.head_dim,
                    cos_cache=cos_cache,
                    sin_cache=sin_cache,
                    num_q_heads=self.num_attention_heads_per_rank,
                    num_kv_heads=self.num_kv_heads_for_weight,
                    k_cache=self.k_cache,
                    v_cache=self.v_cache,
                    use_block_kv=True,
                    block_size=block_size,
                    slot_mapping=slot_mapping_clamped,
                )
                q = (
                    q_hbm.squeeze(0)
                    .view(tokens, self.num_attention_heads_per_rank, self.head_dim)
                    .transpose(0, 1)
                )
            else:
                # Packed FP8 K cache: the qkv kernel cannot write the swizzled
                # layout, so project + RoPE without the in-kernel write, then
                # scatter K/V explicitly (un-swizzle K, scatter, re-swizzle).
                qkv = NF.qkv_proj(
                    hidden=hidden_states.unsqueeze(0),
                    qkv_weights=self.qkv_proj_weight,
                    bias=self.qkv_proj_bias.unsqueeze(0) if self.qkv_proj_bias is not None else None,
                    d_head=self.head_dim,
                    cos_cache=cos_cache,
                    sin_cache=sin_cache,
                    num_q_heads=self.num_attention_heads_per_rank,
                    num_kv_heads=self.num_kv_heads_for_weight,
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
                self._write_paged_kv_cache(k, v, slot_mapping, block_size)

            attn_output = NF.segmented_attention(
                q,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=block_table,
                prior_tokens=cached_seq_len,
                block_size=block_size,
                kv_segment_size=kv_segment_size,
                scale=self.scaling,
                sliding_window=self.sliding_window,
                sink=self.sinks.unsqueeze(1),
                tp_q=True,
                tp_out=True,
                fp8_packed=self.fp8_packed,
            )  # [Nh, Dh, T]
        else:
            # Full prefill: kernel returns concatenated QKV; cache is written
            # via index_put_ since flash_attention needs raw K/V tensors.
            qkv = NF.qkv_proj(
                hidden=hidden_states.unsqueeze(0),
                qkv_weights=self.qkv_proj_weight,
                bias=self.qkv_proj_bias.unsqueeze(0) if self.qkv_proj_bias is not None else None,
                d_head=self.head_dim,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                num_q_heads=self.num_attention_heads_per_rank,
                num_kv_heads=self.num_kv_heads_for_weight,
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

            # KV cache update via index_put_ (flash_attention needs raw K/V).
            self._write_paged_kv_cache(k, v, slot_mapping, block_size)

            # Full prefill: standard flash attention
            k = k.repeat_interleave(self.num_key_value_groups, dim=0)
            v = v.repeat_interleave(self.num_key_value_groups, dim=0)

            q_flash = q.transpose(1, 2)  # [Nh, Dh, T]
            k_flash = k.transpose(1, 2)  # [Nh, Dh, T]
            v_flash = v  # [Nh, T, Dh]

            attn_output = NF.flash_attention(
                q_flash,
                k_flash,
                v_flash,
                scale=self.scaling,
                sliding_window=self.sliding_window,
                sink=self.sinks.unsqueeze(1),
                tp_q=False,
                tp_out=True,
            )  # [Nh, Dh, T]

        # ── Step 5: Output Projection ────────────────────────────────────
        attn_output = attn_output.unsqueeze(0)  # [1, Nh, Dh, T]
        attn_output = NF.o_proj(
            attn_output, self.o_proj_weight, self.o_proj_bias.unsqueeze(0)
        )  # [1, T, H]
        attn_output = attn_output.squeeze(0)  # [T, H]

        # >>> PARALLELISM: Reduce-scatter to return to SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    # ── Decode path ──────────────────────────────────────────────────────

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_mask,
        attn_metadata: object,
    ):
        """Decode: fused megakernel for single-token generation.

        >>> PARALLELISM: The megakernel handles TP internally. <<<
        The kernel performs QKV proj, RoPE, attention, and O proj in one fused call.
        TP all-reduce is done after the kernel.

        <-- MODEL-SPECIFIC: GPT-OSS specific kernel arguments:
        - sink tokens for attention stability
        - sliding window mask generation
        - Non-interleaved RoPE layout
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        # B_local is from metadata (per-DP-rank batch size).
        # Caller ensures input is gathered to B_local * attn_dp via _dp_transition.
        B_local = block_table.shape[0]
        B = B_local * self.attention_dp_size
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        assert tokens == B * S_decode

        hidden_states = hidden_states.to(self.dtype)
        S_ctx = max_blocks_per_seq * block_size
        nkh = self.num_key_value_heads_per_rank

        # Reshape: [B*S, H] → [B, S, H]
        X = hidden_states.view(B, S_decode, hidden)

        # Prepare RoPE for megakernel format: [T, Dh/2] → [Dh/2, B_local, S]
        cos, sin = position_embeddings
        half_d = self.head_dim // 2
        cos_kernel = (
            cos[:, :half_d]
            .view(B_local, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )
        sin_kernel = (
            sin[:, :half_d]
            .view(B_local, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )

        pos_ids = positions.view(1, B_local * S_decode)
        start_pos = None

        pos_ids_kernel = None
        swa_start_pos_ids_kernel = None

        if attn_mask is None:
            if self.sliding_window is not None:
                # The model runner has already trimmed block_table to the
                # window-relevant blocks for SWA layers and provided per-seq
                # swa_kv_pos_offset = start_block * block_size. Shift pos_ids into
                # the trimmed window frame before computing the causal mask.
                swa_kv_pos_offset = attn_metadata[layer_name].get("swa_kv_pos_offset")
                if swa_kv_pos_offset is not None:
                    pos_ids = pos_ids - swa_kv_pos_offset.view(B, 1).expand(
                        B, S_decode
                    ).reshape(1, B * S_decode)
                    # Clamp the trimmed-frame position to >= 0 (see
                    # _precompute_decode_attn_masks for the full rationale).
                    # No-op for a real row; for a no-token padding/freed row
                    # (positions zero-padded to 0, while swa_kv_pos_offset is
                    # positive) it prevents pos_ids = 0 - offset < 0 from
                    # driving the kernel's on-chip mask wrap branch to mark the
                    # whole window valid over an all-(-1) block_table ->
                    # stale-K NaN -> 1006.
                    pos_ids = torch.clamp(pos_ids, min=0)
                start_pos = torch.clamp(pos_ids - self.sliding_window + 1, min=0)

            pos_ids_kernel = pos_ids.view(B_local, S_decode).to(torch.float32)
            swa_start_pos_ids_kernel = (
                start_pos.view(B_local, S_decode).to(torch.float32)
                if start_pos is not None
                else None
            )

        active_blocks_table = block_table

        # Packed FP8 is decided per decode bucket. The cache is *stored* packed
        # (self.fp8_packed), but the packed decode kernel requires the resized
        # block_len to stay >= 2, which some bucket geometries (small SWA
        # windows, batch=1) violate. For a non-viable bucket, un-swizzle the
        # packed cache to the standard [blocks, block_len, d_head] layout, run
        # the unpacked kernel (which updates that standard-layout buffer in
        # place), then re-swizzle the result back into the packed self.k_cache.
        # The viability flag is a static (trace-time) bool, so each compiled
        # decode NEFF takes exactly one branch.
        use_packed_kernel = self.fp8_packed and _packed_fp8_viable_for_bucket(
            block_len=block_size,
            bs=B_local,
            q_head=self.num_q_heads_after_a2a,
            s_active=S_decode,
            s_prior=S_ctx,
        )
        k_cache_arg = (
            self.k_cache
            if (use_packed_kernel or not self.fp8_packed)
            else _unswizzle_packed_k(self.k_cache)
        )

        # >>> PARALLELISM: Fused megakernel with TP-sharded weights <<<
        # In-kernel KV cache update (update_cache=True): the kernel writes K/V
        # in place and the FX aliasing pass threads the write back to the
        # K_cache/V_cache tensors passed in. When the cache is read packed, that
        # target is self.k_cache directly; on the un-swizzled fallback the write
        # lands in the temporary standard-layout buffer and is re-packed below.
        # With update_cache=True the API returns only the attention output.
        output = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.unpadded_hidden_size,
            rmsnorm_X_enabled=False,  # RMSNorm applied by decoder layer
            W_qkv=self.qkv_proj_weight,
            bias_qkv=self.qkv_proj_bias.unsqueeze(0),
            rmsnorm_QK_pre_rope_enabled=False,
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=active_blocks_table,
            K_cache=k_cache_arg,
            V_cache=self.v_cache,
            attention_mask=attn_mask,
            pos_ids=pos_ids_kernel,
            swa_start_pos_ids=swa_start_pos_ids_kernel,
            # Kernel currently requires fusing the K scale dequantization into softmax scale for KV quantization
            softmax_scale=self.scaling / self.k_scale_float,
            # <-- MODEL-SPECIFIC: Sink tokens
            sink=self.sinks.unsqueeze(1),
            update_cache=True,
            kv_cache_update_idx=slot_mapping.view(B_local, S_decode).to(torch.uint32),
            fp8_packed=use_packed_kernel,
            # Kernel currently requires fusing the V scale dequantization into W_out for KV quantization
            W_out=self.o_proj_weight / self.v_scale_float,
            bias_out=self.o_proj_bias.unsqueeze(0),
            transposed_out=False,
            out_in_sb=False,
            k_scale=self.k_scale
            if self.k_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
            else None,
            v_scale=self.v_scale
            if self.v_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
            else None,
            attention_dp=self.attention_dp_size,
            attention_dp_group=self.attention_dp_group.device_group
            if self.attention_dp_group
            else None,
            attention_dp_rank=self.attention_dp_rank,
            kv_needs_a2a=self.kv_needs_a2a,
        )

        # Non-viable packed bucket: the unpacked kernel updated the temporary
        # standard-layout K buffer; re-swizzle it back into the packed cache.
        if self.fp8_packed and not use_packed_kernel:
            self.k_cache.copy_(_swizzle_packed_k(k_cache_arg))

        # >>> PARALLELISM: Sum O-proj partials across TP * attn_dp <<<
        self.attn_tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 4: MoE Experts
# Mixed: PARALLELISM (TP sharding of expert weights, SP collectives) +
#        MODEL-SPECIFIC (SwiGLU activation, clamping, routing params)
# =============================================================================
class GptOssExperts(nn.Module):
    """Expert feed-forward layers with TP, optional EP, and cross-DP EP support.

    >>> PARALLELISM: TP + EP <<<
    - EP disabled (ep_degree=1): all experts on all ranks, intermediate sharded by TP
    - EP within TP (ep_degree=TP): experts partitioned across TP ranks (linear placement),
      full intermediate per rank (tp_degree=1)
    - EP across DP (ep_degree=TP*DP): experts spread across all ranks. Cross-DP token
      reduce-scatter (prefill) or all-reduce (decode) across moe_group.

    <-- MODEL-SPECIFIC:
    - 32 experts, top-4 per token
    - SwiGLU activation with per-gate and per-up clamping
    - Softmax routing with post-scale expert affinities
    - Pre-MLP RMSNorm
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()

        # >>> PARALLELISM: TP + EP configuration <<<
        # - Without EP: experts are replicated, intermediate dim is TP-sharded.
        # - With EP (variable degree): experts are partitioned across ep_degree
        #   ranks, intermediate dim is sharded across tp_degree = world_size / ep_degree.
        #   Pure EP (ep_degree = world_size) gives tp_degree = 1 (no intermediate sharding).
        # moe_group = tp_group for outer collectives (all-reduce, reduce-scatter).
        # ep_tp_group = TP sub-group within EP partition for blockwise mapping.
        self.tp_group = get_tp_group()
        self.rank = self.tp_group.rank_in_group

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        self.ep_enabled = vllm_config.parallel_config.enable_expert_parallel
        self.dp_size = vllm_config.parallel_config.data_parallel_size

        # >>> PARALLELISM: MLP DP setup <<<
        self.mlp_dp_size = (
            config.neuron_config.mlp_dp_size if config.neuron_config else 1
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_mlp_tp_group,
        )

        self.mlp_tp_group = get_neuron_mlp_tp_group()
        self.mlp_tp_rank = self.mlp_tp_group.rank_in_group

        if self.ep_enabled:
            # EP enabled: read degree from neuron parallel state (always
            # initialized as GroupCoordinator when EP is active).
            from vllm_neuron.parallel.neuron_parallel_state import (
                get_neuron_ep_degree,
                get_neuron_ep_rank,
                get_neuron_ep_tp_group,
            )

            self.ep_degree = get_neuron_ep_degree()
            self.ep_rank = get_neuron_ep_rank()
            # EP-TP sub-group: used for blockwise mapping TP coordination.
            # For pure EP (tp_degree=1) this is a single-rank group (no-op collectives).
            # For variable EP+TP this is the TP sub-group within the EP partition.
            self.ep_tp_group = get_neuron_ep_tp_group()
            self.tp_degree = self.ep_tp_group.world_size
        else:
            # No EP: all experts on all ranks, intermediate dim sharded across
            # TP * mlp_dp_size via the MLP TP group.
            self.ep_degree = 1
            self.ep_rank = 0
            self.tp_degree = self.mlp_tp_group.world_size
            self.ep_tp_group = self.tp_group
        self.moe_group = self.mlp_tp_group if not self.ep_enabled else self.tp_group

        # >>> PARALLELISM: Cross-DP EP <<<
        # When EP degree exceeds the TP group size, experts span across DP
        # replicas. MoE needs cross-DP collectives (all-gather/reduce-scatter)
        # to exchange tokens between DP replicas before/after expert computation.
        self.cross_dp_ep = self.dp_size > 1 and self.ep_enabled
        if self.cross_dp_ep:
            from vllm.distributed.parallel_state import (
                get_dp_group,
                get_wide_ep_group,
            )

            self.dp_group = get_dp_group()
            # World-spanning group with device communicator for cross-DP all-reduce
            self.wide_ep_group = get_wide_ep_group()

        self.total_num_experts = config.num_local_experts

        # Linear placement: EP rank k owns experts [k*L .. (k+1)*L)
        self.num_local_experts = config.num_local_experts // self.ep_degree
        self.num_experts_per_token = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.unpadded_hidden_size = config.unpadded_hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        # >>> PARALLELISM: Intermediate dim sharded across TP degree <<<
        self.intermediate_size_per_rank = self.intermediate_size // self.tp_degree
        self.block_size = 256

        # <-- MODEL-SPECIFIC: SwiGLU activation parameters
        self.alpha = config.swiglu_alpha
        self.limit = config.swiglu_limit

        # <-- MODEL-SPECIFIC: Pre-MLP RMSNorm
        self.post_attention_layernorm = GptOssRMSNorm(config)

        # >>> PARALLELISM: Router weights replicated on all ranks (NOT EP-sharded) <<<
        self.router_weight = nn.Parameter(
            torch.empty(self.total_num_experts, self.hidden_size, dtype=torch.bfloat16)
        )
        self.router_bias = nn.Parameter(
            torch.zeros(self.total_num_experts, dtype=torch.bfloat16)
        )

        # >>> PARALLELISM: Expert weights sharded on intermediate dim <<<
        # gate_up_proj: [E, H, I_per_rank*2] (gate and up interleaved)
        self.gate_up_proj_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_size,
                self.intermediate_size_per_rank * 2,
                dtype=config.torch_dtype,
            )
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.zeros(
                self.num_local_experts,
                self.intermediate_size_per_rank * 2,
                dtype=config.torch_dtype,
            )
        )
        # down_proj: [E, I_per_rank, H]
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.intermediate_size_per_rank,
                self.hidden_size,
                dtype=config.torch_dtype,
            )
        )
        self.down_proj_bias = nn.Parameter(
            torch.zeros(
                self.num_local_experts, self.hidden_size, dtype=config.torch_dtype
            )
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Set up weight loaders for TP + EP sharding.

        >>> PARALLELISM: TP sharding of intermediate dim, EP filtering of experts <<<
        <-- MODEL-SPECIFIC: Checkpoint format is MXFP4 packed, needs dequantization
        """
        # Router: pad hidden dim (always full, not EP-sharded)
        set_weight_loader(
            self.router_weight, last_dim_padding_weight_loader(self.hidden_size)
        )
        set_weight_loader(
            self.post_attention_layernorm.weight,
            last_dim_padding_weight_loader(self.hidden_size),
        )

        # Linear EP placement: rank k owns experts [k*L, (k+1)*L)
        local_expert_indices = list(
            range(
                self.ep_rank * self.num_local_experts,
                (self.ep_rank + 1) * self.num_local_experts,
            )
        )

        def _maybe_ep_wrap(loader):
            """Wrap loader with EP expert filtering if ep_degree > 1,
            and with rank override for mlp_dp_size > 1."""
            if self.ep_degree > 1:
                loader = expert_parallel_tensor_dim_loader(local_expert_indices, loader)
            if self.mlp_dp_size > 1:
                loader = with_rank_override(loader, rank=self.mlp_tp_rank)
            return loader

        # <-- MODEL-SPECIFIC: MXFP4 dequantization + TP sharding (+ EP filtering)
        set_weight_loader(
            self.gate_up_proj_weight,
            _maybe_ep_wrap(
                expert_gate_up_weight_sharding_loader(
                    shard_size=self.intermediate_size_per_rank * 2,
                    num_shards=self.tp_degree,
                    hidden_size=self.hidden_size,
                )
            ),
        )
        set_weight_loader(
            self.gate_up_proj_bias,
            _maybe_ep_wrap(
                expert_gate_up_bias_sharding_loader(
                    shard_size=self.intermediate_size_per_rank * 2,
                    num_shards=self.tp_degree,
                )
            ),
        )
        set_weight_loader(
            self.down_proj_weight,
            _maybe_ep_wrap(
                expert_down_weight_sharding_loader(
                    shard_size=self.intermediate_size_per_rank,
                    num_shards=self.tp_degree,
                    hidden_size=self.hidden_size,
                )
            ),
        )
        set_weight_loader(
            self.down_proj_bias,
            _maybe_ep_wrap(
                scaled_bias_loader(scale=self.tp_degree, padded_size=self.hidden_size)
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        is_decode: bool,
        rank: torch.Tensor,
    ) -> torch.Tensor:
        if is_decode:
            return self.forward_decode(hidden_states, rank)
        else:
            return self.forward_prefill(hidden_states, positions, rank)

    def _run_moe_block_tkg(self, hidden_states: torch.Tensor, rank: torch.Tensor):
        """Run the fused MoE decode kernel on given hidden states.

        >>> PARALLELISM: use_all_experts and rank_id for EP <<<
        <-- MODEL-SPECIFIC: Activation, routing, clamping params
        """
        total_tokens = hidden_states.shape[0]
        perc_experts_loaded = (
            total_tokens * self.num_experts_per_token / self.num_local_experts
        )
        use_all_experts = (
            perc_experts_loaded >= DEFAULT_SELECTIVE_LOADING_THRESHOLD
            or self.ep_degree > 1
        )
        rank_id = None
        if use_all_experts:
            rank_id = torch.tensor(
                [[self.ep_rank]], dtype=torch.int32, device=hidden_states.device
            )

        return NF.moe_block_tkg(
            inp=hidden_states.unsqueeze(0),
            gamma=self.post_attention_layernorm.weight.unsqueeze(0).to(torch.float32),
            router_weights=self.router_weight.T,
            expert_gate_up_weights=self.gate_up_proj_weight.reshape(
                self.num_local_experts,
                self.hidden_size,
                2,
                self.intermediate_size_per_rank,
            ),
            expert_down_weights=self.down_proj_weight,
            rank_id=rank_id,
            top_k=self.num_experts_per_token,
            router_bias=self.router_bias.unsqueeze(0),
            expert_gate_up_bias=self.gate_up_proj_bias.reshape(
                self.num_local_experts,
                2,
                self.intermediate_size_per_rank,
            ),
            expert_down_bias=self.down_proj_bias,
            eps=self.rms_norm_eps,
            router_act_fn=RouterActFnType.SOFTMAX,
            router_pre_norm=False,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            hidden_act_fn=ActFnType.Swish,
            gate_clamp_upper_limit=self.limit,
            gate_clamp_lower_limit=None,
            up_clamp_upper_limit=self.limit + 1,
            up_clamp_lower_limit=-self.limit + 1,
            router_mm_dtype=nl.bfloat16,
            hidden_actual=self.unpadded_hidden_size,
            is_all_expert=use_all_experts,
            skip_router_logits=True,
        )

    def forward_decode(self, hidden_states: torch.Tensor, rank: torch.Tensor):
        """Decode: fused MoE kernel.

        >>> PARALLELISM: TKG kernel with EP all-reduce <<<
        >>> PARALLELISM: Cross-DP EP: all-gather tokens across DP before MoE,
        >>>   slice back to own DP replica after all-reduce.
        """
        # >>> PARALLELISM: Cross-DP EP gather <<<
        if self.cross_dp_ep:
            hidden_states = self.dp_group.all_gather(hidden_states, dim=0)

        output = self._run_moe_block_tkg(hidden_states, rank)

        # >>> PARALLELISM: All-reduce across world for cross-DP EP <<<
        if self.cross_dp_ep:
            # All-reduce across entire world (all TP and DP ranks)
            output = self.wide_ep_group.all_reduce(output)
            # Slice to keep only this DP replica's tokens
            dp_rank = self.dp_group.rank_in_group
            tokens_per_dp = output.shape[0] // self.dp_size
            start_idx = dp_rank * tokens_per_dp
            end_idx = start_idx + tokens_per_dp
            output = output[start_idx:end_idx]
        elif self.moe_group.world_size > 1:
            output = self.moe_group.all_reduce(output)

        return output

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rank: torch.Tensor,
    ):
        """Prefill: blockwise MoE with CTE kernel.

        >>> PARALLELISM: All-gather from SP → MoE → reduce-scatter back to SP <<<
        <-- MODEL-SPECIFIC: Softmax routing, SwiGLU activation, clamping
        """
        # <-- MODEL-SPECIFIC: Pre-MLP RMSNorm
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Router: [T/world_size, H] → [T/world_size, E_total]
        expert_affinities = NF.router(
            hidden_states=hidden_states,
            router_weights=self.router_weight.T,
            top_k=self.num_experts_per_token,
            router_bias=self.router_bias,
            # <-- MODEL-SPECIFIC: Softmax routing
            activation="softmax",
            computation_dtype=torch.float32,
        )

        # >>> PARALLELISM: All-gather from SP for full sequence (within TP group) <<<
        if self.tp_group.world_size > 1:
            expert_affinities = self.tp_group.all_gather(expert_affinities, dim=0)
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        # Compute padding mask from positions (True = real token, False = padding).
        # Must be computed before cross-DP gather since positions from different
        # DP replicas are not monotonically increasing when concatenated.
        padding_mask = None
        if positions is not None:
            last_real_idx = torch.argmax(positions)
            token_indices = torch.arange(positions.shape[0], device=positions.device)
            padding_mask = token_indices <= last_real_idx

        # >>> PARALLELISM: Cross-DP EP gather — collect tokens from all DP replicas <<<
        if self.cross_dp_ep:
            expert_affinities = self.dp_group.all_gather(expert_affinities, dim=0)
            hidden_states = self.dp_group.all_gather(hidden_states, dim=0)
            if padding_mask is not None:
                padding_mask = self.dp_group.all_gather(padding_mask, dim=0)

        # >>> PARALLELISM: With EP, map global affinities to local experts <<<
        if self.ep_degree > 1:
            local_expert_indices = torch.arange(
                self.ep_rank * self.num_local_experts,
                (self.ep_rank + 1) * self.num_local_experts,
                device=hidden_states.device,
                dtype=torch.int32,
            )
            expert_affinities = NF.get_local_expert_affinities(
                expert_affinities, local_expert_indices
            )

        # Build blockwise mapping for efficient MoE dispatch
        # >>> PARALLELISM: ep_tp_group provides TP rank coordination within
        # the EP partition. For pure TP this is the full tp_group; for pure EP
        # it is a single-rank group (no sharding); for variable EP+TP it is
        # the TP sub-group within the EP partition.
        (
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
            conditions,
        ) = NF.build_blockwise_mapping(
            expert_affinities=expert_affinities,
            num_local_experts=self.num_local_experts,
            num_experts_per_token=self.num_experts_per_token,
            block_size=self.block_size,
            moe_group=self.ep_tp_group,
            tp_degree=self.tp_degree,
            padding_mask=padding_mask,
        )

        output = NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_block,
            conditions=conditions,
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=self.gate_up_proj_weight.reshape(
                self.num_local_experts,
                self.hidden_size,
                2,
                self.intermediate_size_per_rank,
            ),
            down_proj_weight=self.down_proj_weight,
            gate_and_up_proj_bias=self.gate_up_proj_bias.reshape(
                self.num_local_experts,
                2,
                self.intermediate_size_per_rank,
            ),
            down_proj_bias=self.down_proj_bias.unsqueeze(1),
            # <-- MODEL-SPECIFIC: SwiGLU with clamping
            activation_function=ActFnType.Swish,
            block_size=self.block_size,
            token_position_to_id=token_position_to_id.to(dtype=torch.int32),
            block_to_expert=block_to_expert.to(dtype=torch.int32),
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            gate_clamp_upper_limit=self.limit,
            gate_clamp_lower_limit=None,
            up_clamp_upper_limit=self.limit + 1,
            up_clamp_lower_limit=-self.limit + 1,
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
        )

        # >>> PARALLELISM: Cross-DP EP all-reduce and slice <<<
        if self.cross_dp_ep:
            # All-reduce across entire world (all TP and DP ranks)
            output = self.wide_ep_group.all_reduce(output)
            # Slice to keep only this DP replica's tokens
            dp_rank = self.dp_group.rank_in_group
            tokens_per_dp = output.shape[0] // self.dp_size
            start_idx = dp_rank * tokens_per_dp
            end_idx = start_idx + tokens_per_dp
            output = output[start_idx:end_idx]
            # Slice to keep only this TP rank's SP chunk
            tp_rank = self.moe_group.rank_in_group
            tokens_per_tp = output.shape[0] // self.moe_group.world_size
            start_idx = tp_rank * tokens_per_tp
            end_idx = start_idx + tokens_per_tp
            output = output[start_idx:end_idx]
        # >>> PARALLELISM: Combine expert results and return to SP layout <<<
        elif self.moe_group.world_size > 1:
            output = self.moe_group.reduce_scatter(output, dim=0)

        return output


# =============================================================================
# Section 5: MLP Wrapper
# <-- MODEL-SPECIFIC: Thin wrapper around experts
# =============================================================================
class GptOssMLP(nn.Module):
    """MLP layer with Mixture of Experts.

    <-- MODEL-SPECIFIC: This wrapper exists because GPT-OSS uses MoE.
    A dense model would have a simple MLP here instead.
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.experts = GptOssExperts(config)
        self.dtype = config.torch_dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        is_decode: bool,
        rank: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(self.dtype)
        return self.experts(
            hidden_states, positions=positions, is_decode=is_decode, rank=rank
        )


# =============================================================================
# Section 6: Decoder Layer
# Mixed: PARALLELISM (SP residual handling) +
#        MODEL-SPECIFIC (pre-attention norm, residual connections)
# =============================================================================
def _dp_transition(
    x: torch.Tensor,
    current_group,
    target_group,
    dim: int = 0,
) -> torch.Tensor:
    """Transition tensor between DP gathered states.

    Args:
        x: Input tensor, gathered over current_group.world_size DP ranks along dim.
        current_group: GroupCoordinator for the current DP column group.
        target_group: GroupCoordinator for the target DP column group.
        dim: Batch dimension to gather/slice along.

    Returns:
        Tensor at target gathered state.
    """
    current_dp = current_group.world_size
    target_dp = target_group.world_size
    if current_dp == target_dp:
        return x
    if current_dp > target_dp:
        per_dp = x.shape[dim] // current_dp
        start = (current_group.rank_in_group // target_dp) * target_dp * per_dp
        return x.narrow(dim, start, target_dp * per_dp)
    # Up: go through local first to avoid duplicates from overlapping gathered state,
    # then all-gather to target.
    per_dp = x.shape[dim] // current_dp
    x = x.narrow(dim, current_group.rank_in_group * per_dp, per_dp)
    return target_group.all_gather(x, dim=dim)


def _precompute_decode_attn_masks(
    sliding_window: int | None,
    attn_metadata: dict,
    positions: torch.Tensor,
    num_q_heads_after_a2a: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute the shared decode attention masks for one decode step.

    Returns ``(swa_mask, full_mask)`` consumed by even-idx (SWA) and odd-idx
    (full attention) layers respectively. Computing them once per step lets
    the attention kernel skip in-layer mask generation.

    The model runner trims block_table to the window-relevant blocks for SWA
    layers and provides per-seq ``swa_kv_pos_offset`` (= start_block *
    block_size). The mask is generated over the trimmed cache, so BOTH the
    mask end (``pos_ids``) and the window start (``start_pos``) must live in
    the trimmed frame. Mixing frames (absolute end, trimmed start) degenerates
    the window test once the trim activates (context > S_ctx_swa): every slot
    from the window start through the end of the trimmed cache — including
    stale, not-yet-overwritten slots — gets attended.

    Args:
        sliding_window: SWA window size from model config (None disables the
            window start; even layers then get a causal mask over the trimmed
            cache).
        attn_metadata: Per-layer metadata dict from the model runner.
        positions: [B*S_decode] absolute position ids for the active tokens.
        num_q_heads_after_a2a: Query heads per rank after attention-DP A2A.

    Returns:
        Tuple of (swa_mask, full_mask), each
        [S_ctx, B, num_q_heads_after_a2a, S_decode].

    Example:
        >>> swa_mask, full_mask = _precompute_decode_attn_masks(
        ...     128, attn_metadata, positions, num_q_heads_after_a2a=8
        ... )
    """
    first_layer_name = "layers.0.self_attn"
    block_size = attn_metadata[first_layer_name]["block_size"]
    block_table = attn_metadata[first_layer_name]["block_table_tensor"]
    B_local = block_table.shape[0]
    S_decode = positions.shape[0] // B_local
    pos_ids_flat = positions.reshape(1, B_local * S_decode)

    # --- SWA mask (even layers) ---
    swa_max_blocks = attn_metadata[first_layer_name]["max_blocks_per_seq"]
    S_ctx_swa = swa_max_blocks * block_size

    swa_pos_ids = pos_ids_flat
    swa_start_pos_ids = None
    if sliding_window is not None:
        swa_kv_pos_offset = attn_metadata[first_layer_name].get("swa_kv_pos_offset")
        if swa_kv_pos_offset is not None:
            swa_pos_ids = pos_ids_flat - swa_kv_pos_offset.reshape(B_local, 1).expand(
                B_local, S_decode
            ).reshape(1, B_local * S_decode)
            # Clamp the trimmed-frame position to >= 0. For a real decode row this is
            # a no-op (the window-start offset is <= the row's position, so swa_pos
            # >= 0). It only fires for a no-token row: a padding/freed slot has its
            # position zero-padded to 0, while swa_kv_pos_offset is positive, so
            # without the clamp swa_pos = 0 - offset < 0. That negative position
            # drives the mask's wrap branch (max_start > min_pos) to mark the ENTIRE
            # window valid over that row's all-(-1) block_table -> the kernel's
            # skipped-block stale-K SBUF is read at a mask-valid slot -> NaN ->
            # distributed topk gather(-1) -> nrta 1006. Clamping collapses the window
            # to empty (start=0, end=0), so only the active-token self-slot stays
            # valid; that slot's K is the separately supplied k_active, never a -1
            # block gather, so the row is fully and safely masked. Pure-positional,
            # no block_table, traces cleanly.
            swa_pos_ids = torch.clamp(swa_pos_ids, min=0)
        swa_start_pos_ids = torch.clamp(swa_pos_ids - sliding_window + 1, min=0)

    swa_mask = NF.gen_attention_decode_mask(
        pos_ids=swa_pos_ids.to(torch.float32),
        bs=B_local,
        q_head=num_q_heads_after_a2a,
        s_active=S_decode,
        s_prior=S_ctx_swa,
        start_pos=swa_start_pos_ids.to(torch.float32)
        if swa_start_pos_ids is not None
        else None,
        block_len=block_size,
    )

    # --- Full causal mask (odd layers) ---
    # GPT-OSS alternates SWA (even idx) and full-attention (odd idx) layers,
    # so layer 1 is always the first full-attention layer and its untrimmed
    # block table gives the full-mask context length. GPT-OSS always has >= 2
    # layers (20B: 24, 120B: 36), so this index is safe.
    non_swa_layer_name = "layers.1.self_attn"
    full_max_blocks = attn_metadata[non_swa_layer_name]["max_blocks_per_seq"]
    S_ctx_full = full_max_blocks * block_size

    full_mask = NF.gen_attention_decode_mask(
        pos_ids=pos_ids_flat.to(torch.float32),
        bs=B_local,
        q_head=num_q_heads_after_a2a,
        s_active=S_decode,
        s_prior=S_ctx_full,
        start_pos=None,
        block_len=block_size,
    )

    return swa_mask, full_mask


class GptOssDecoderLayer(nn.Module):
    """Single transformer decoder layer.

    Architecture (MODEL-SPECIFIC):
        hidden_states → RMSNorm → Attention → residual → MoE → residual

    Parallelism (TP + SP):
        - Input arrives in SP layout (T/world_size tokens per rank) during prefill
        - Decode: _dp_transition handles batch state between modules
        - Residual connection operates at whatever gathered state the module outputs
    """

    def __init__(self, config: GptOssConfig, batch_size: int, layer_idx: int):
        super().__init__()
        # <-- MODEL-SPECIFIC: Pre-attention RMSNorm
        self.input_layernorm = GptOssRMSNorm(config)
        self.self_attn = GptOssAttention(config, layer_idx=layer_idx)
        self.mlp = GptOssMLP(config)
        self.layer_idx = layer_idx

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        # >>> PARALLELISM: DP sizes for batch state transitions <<<
        nc = config.neuron_config
        self.attn_dp = nc.attention_dp_size if nc else 1
        self.mlp_dp = nc.mlp_dp_size if nc else 1

        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_attention_dp_group,
            get_neuron_mlp_dp_group,
        )

        self.attn_dp_group = get_neuron_attention_dp_group()
        self.mlp_dp_group = get_neuron_mlp_dp_group()

        # Padding weight loader for input layer norm
        set_weight_loader(
            self.input_layernorm.weight,
            last_dim_padding_weight_loader(config.hidden_size),
        )

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
        attn_mask=None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_decode = self._is_decode(attn_metadata)

        if not is_decode:
            return self._forward_prefill(
                hidden_states, positions, position_embeddings, attn_metadata, rank
            )

        # ── Decode: batch state transitions between modules ──
        # Input arrives at mlp_dp (from previous layer's MLP or embedding transition).

        # Transition mlp_dp → attn_dp
        hidden_states = _dp_transition(
            hidden_states, self.mlp_dp_group, self.attn_dp_group
        )

        # ── Self Attention ───────────────────────────────────────────────
        residual = hidden_states
        # <-- MODEL-SPECIFIC: Pre-attention RMSNorm
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
            attn_mask=attn_mask,
        )
        # Attention outputs at attn_dp (supergroup all-reduce, stays gathered)

        # Residual add at attn_dp, then transition once to mlp_dp
        hidden_states = residual + hidden_states
        hidden_states = _dp_transition(
            hidden_states, self.attn_dp_group, self.mlp_dp_group
        )

        # ── MoE Feed-Forward ─────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.mlp(
            hidden_states,
            positions=positions,
            is_decode=True,
            rank=rank,
        )
        # MLP outputs at mlp_dp (supergroup all-reduce, stays gathered)
        hidden_states = residual + hidden_states

        return hidden_states

    def _forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ── Self Attention ───────────────────────────────────────────────
        residual = hidden_states
        # <-- MODEL-SPECIFIC: Pre-attention RMSNorm
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        # ── MoE Feed-Forward ─────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.mlp(
            hidden_states,
            positions=positions,
            is_decode=False,
            rank=rank,
        )
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Section 7: Model Backbone
# Mixed: PARALLELISM (SP chunking / gathering) +
#        MODEL-SPECIFIC (embedding, layer stack, final norm)
# =============================================================================
class GptOssModel(nn.Module):
    """GPT-OSS transformer backbone.

    >>> PARALLELISM: SP (Sequence Parallelism) <<<
    During prefill:
    - After embedding: chunk sequence across TP ranks (each gets T/world_size)
    - After all layers: all-gather to reconstruct full sequence
    During decode:
    - No SP; all ranks process all tokens
    """

    def __init__(self, config: GptOssConfig, batch_size: int):
        super().__init__()
        self.config = config

        # >>> PARALLELISM: TP group for SP <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: Embedding TP group (TP * embedding_dp_size) + DP column <<<
        self.embedding_dp_size = (
            config.neuron_config.embedding_dp_size if config.neuron_config else 1
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_embedding_tp_group,
            get_neuron_embedding_dp_group,
            get_neuron_mlp_dp_group,
        )

        emb_tp_group = get_neuron_embedding_tp_group()
        emb_device_group = emb_tp_group.device_group
        self.embedding_dp_group = get_neuron_embedding_dp_group()
        self.embedding_tp_rank = emb_tp_group.rank_in_group
        self.mlp_dp_group = get_neuron_mlp_dp_group()

        # >>> PARALLELISM: Vocab-sharded embedding <<<
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=emb_device_group,
        )

        # <-- MODEL-SPECIFIC: Stack of decoder layers
        self.layers = nn.ModuleList(
            [
                GptOssDecoderLayer(config, batch_size, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # <-- MODEL-SPECIFIC: Final RMSNorm
        self.norm = GptOssRMSNorm(config)
        self.rotary_emb = GptOssRotaryEmbedding(config)

        # Weight loaders: shard embedding on vocab dim, pad hidden dim
        from vllm_neuron.utils.weight_loader import (
            sharding_weight_loader_with_padding,
        )

        emb_loader = sharding_weight_loader_with_padding(
            shard_dim=0,
            shard_size=self.embed_tokens.vocab_size_per_rank,
            num_shards=self.embed_tokens.tp_size,
            pad_dim=1,
            padded_size=config.hidden_size,
            unpadded_size=config.unpadded_hidden_size,
        )
        emb_loader = with_rank_override(emb_loader, rank=self.embedding_tp_rank)
        set_weight_loader(self.embed_tokens.weight, emb_loader)
        set_weight_loader(
            self.norm.weight, last_dim_padding_weight_loader(config.hidden_size)
        )

        # Eagle3 speculative decoding: layer indices whose hidden states
        # are collected for the draft model.  Empty until the drafter sets them.
        self.aux_hidden_state_layers = []

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_ids: [T] token ids
            positions: [T] position indices
            attn_metadata: dict with cache management info
            rank: [1] local rank tensor
            inputs_embeds: [T, H] user-provided embeddings (or None)
            is_token_ids: [T] bool mask — True for token-ID positions, False for embed positions

        Returns:
            hidden_states: [T, H]
            aux_hidden_states: list of hidden states at Eagle3 draft layers (empty if not using Eagle3)
        """
        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        # >>> PARALLELISM: Embedding DP — gather input_ids (tiny, just token IDs) <<<
        if not is_prefill and self.embedding_dp_size > 1:
            input_ids = self.embedding_dp_group.all_gather(input_ids, dim=0)

        # >>> PARALLELISM: VocabDimShardedEmbedding handles SP internally <<<
        # scatter_tokens=True (prefill): reduce_scatter → [T/world_size, H]
        # scatter_tokens=False (decode): all_reduce → [T, H]
        emb_rank = rank
        if rank is not None and self.embedding_dp_size > 1:
            emb_rank = rank + (self.embedding_tp_rank - self.rank)
        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=emb_rank
        )
        # Decode: embedding output at emb_dp gathered state

        # >>> PARALLELISM: Transition emb_dp → mlp_dp for decoder layers <<<
        if not is_prefill:
            hidden_states = _dp_transition(
                hidden_states, self.embedding_dp_group, self.mlp_dp_group
            )

        # <-- MODEL-SPECIFIC: GPT-OSS accepts unpadded prompt embeds and pads
        # them to the runtime hidden dim used by this model variant.
        if inputs_embeds is not None:
            target_dim = hidden_states.shape[-1]
            current_dim = inputs_embeds.shape[-1]
            if current_dim > target_dim:
                raise ValueError(
                    f"inputs_embeds dim ({current_dim}) exceeds hidden dim "
                    f"({target_dim}) for GPT-OSS prompt embeddings."
                )
            inputs_embeds = torch.nn.functional.pad(
                inputs_embeds, (0, target_dim - current_dim)
            )

        # >>> PARALLELISM: SP prompt-embed path <<<
        # Shard inputs_embeds/is_token_ids to match SP layout before merging.
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

        # Compute RoPE embeddings
        position_embeddings = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )

        # <-- MODEL-SPECIFIC: Pre-compute decode attention masks ──────────────
        # SWA layers (even idx) get a sliding-window mask; non-SWA layers
        # (odd idx) get a full causal mask. Both are pre-computed so the
        # kernel skips in-layer mask generation.
        swa_mask = None
        full_mask = None
        if not is_prefill:
            swa_mask, full_mask = _precompute_decode_attn_masks(
                sliding_window=self.config.sliding_window,
                attn_metadata=attn_metadata,
                positions=positions,
                num_q_heads_after_a2a=self.layers[0].self_attn.num_q_heads_after_a2a,
            )

        # Run through decoder layers, collecting Eagle3 auxiliary hidden states
        aux_hidden_states = []
        for idx, decoder_layer in enumerate(self.layers):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states)
            # <-- MODEL-SPECIFIC: Sliding window on alternating (even) layers
            attn_mask = swa_mask if idx % 2 == 0 else full_mask
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_mask=attn_mask,
                attn_metadata=attn_metadata,
                rank=rank,
            )

        hidden_states = self.norm(hidden_states)

        # >>> PARALLELISM: SP - all-gather to reconstruct full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
            # TODO: make Eagle3 drafter accept SP-partitioned aux states to avoid this all-gather
            aux_hidden_states = [
                self.tp_group.all_gather(aux, dim=0) for aux in aux_hidden_states
            ]

        return hidden_states, aux_hidden_states


# =============================================================================
# Section 8: Language Model Head
# Mixed: PARALLELISM (column-parallel LM head) +
#        MODEL-SPECIFIC (vocabulary projection, sampling)
# =============================================================================


@async_speculative_decoding
class GptOssForCausalLM(nn.Module, SupportsEagle3):
    """GPT-OSS model with language modeling head.

    >>> PARALLELISM: Column-parallel linear for LM head <<<
    The vocabulary projection is sharded across TP ranks. Each rank computes
    a portion of the logits, then either:
    - Gathered for full logits (when not using on-device sampling)
    - Kept sharded for on-device sampling (sampler handles TP internally)
    """

    def __init__(self, config: GptOssConfig, batch_size: int):
        super().__init__()
        self.config = config
        self.model = GptOssModel(config, batch_size)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: LM Head TP group + DP column <<<
        self.lm_head_dp_size = (
            config.neuron_config.lm_head_dp_size if config.neuron_config else 1
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_lm_head_tp_group,
            get_neuron_lm_head_dp_group,
            get_neuron_mlp_dp_group,
        )

        lm_head_tp_group = get_neuron_lm_head_tp_group()
        self.lm_head_tp_group = lm_head_tp_group
        lm_head_device_group = lm_head_tp_group.device_group
        self.lm_head_dp_group = get_neuron_lm_head_dp_group()
        self.mlp_dp_group = get_neuron_mlp_dp_group()
        lm_head_tp_rank = lm_head_tp_group.rank_in_group

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

        # >>> PARALLELISM: Column-parallel LM head <<<
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=lm_head_device_group,
        )

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=lm_head_device_group,
            )

        # >>> PARALLELISM: Shard lm_head on vocab dim (dim 0), pad hidden dim (dim 1) <<<
        from vllm_neuron.utils.weight_loader import (
            sharding_weight_loader_with_padding,
        )

        lm_head_loader = sharding_weight_loader_with_padding(
            shard_dim=0,
            shard_size=self.lm_head.out_features_per_rank,
            num_shards=self.lm_head.tp_size,
            pad_dim=1,
            padded_size=config.hidden_size,
            unpadded_size=config.unpadded_hidden_size,
        )
        lm_head_loader = with_rank_override(lm_head_loader, rank=lm_head_tp_rank)
        set_weight_loader(self.lm_head.weight, lm_head_loader)

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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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

        # >>> PARALLELISM: Slice to local before sampling position selection <<<
        mlp_dp = self.mlp_dp_group.world_size
        if mlp_dp > 1:
            local_size = hidden_states.shape[0] // mlp_dp
            dp_rank = self.mlp_dp_group.rank_in_group
            hidden_states = hidden_states[
                dp_rank * local_size : (dp_rank + 1) * local_size
            ]

        hidden_states_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )

        # >>> PARALLELISM: Transition local → lm_head_dp <<<
        if self.lm_head_dp_size > 1:
            hidden_states_for_logits = self.lm_head_dp_group.all_gather(
                hidden_states_for_logits, dim=0
            )

        # Ensure dtype matches lm_head weight (MoE CPU fallback may return float32)
        hidden_states_for_logits = hidden_states_for_logits.to(self.config.torch_dtype)

        logits = self.lm_head(hidden_states_for_logits)

        # >>> PARALLELISM: Gather sharded logits for logprobs computation <<<
        gathered_logits = None
        if self._gather_logits:
            gathered_logits = self.lm_head_tp_group.all_gather(logits, dim=1)

        # >>> PARALLELISM: DP-batch slice back to this rank's local rows <<<
        # When lm_head_dp_size > 1 the on-device sampler performs its
        # argmax/top-k as a distributed reduction across the full
        # lm_head_device_group, which requires every rank to hold the same set
        # of rows sharded along the vocab dimension. Slicing `logits` to this
        # DP rank's rows before that reduction breaks the invariant: the ranks
        # end up holding different row blocks, so the reduction combines vocab
        # shards from different sequences and can drop the true argmax token
        # (the winning token's vocab shard may live on another DP rank).
        #
        # To preserve row alignment, the on-device-sampling path samples on the
        # full pre-slice logits and slices the sampled tokens afterward (below).
        # The non-sampling path returns per-DP-rank logits, so it still slices
        # the logits here; likewise the gathered logprobs tensor is per-DP-rank.
        B_local = sampling_positions.shape[0]
        dp_rank = self.lm_head_dp_group.rank_in_group if self.lm_head_dp_size > 1 else 0
        if self.lm_head_dp_size > 1 and gathered_logits is not None:
            gathered_logits = gathered_logits[
                dp_rank * B_local : (dp_rank + 1) * B_local
            ]

        # ── No on-device sampling: return per-DP-rank logits directly ─────
        if self.on_device_sampling_config is None:
            if self.lm_head_dp_size > 1:
                logits = logits[dp_rank * B_local : (dp_rank + 1) * B_local]
            if len(aux_hidden_states) > 0:
                # Eagle3: concatenate aux hidden states on-device so the
                # target NEFF emits a single tensor instead of a list,
                # keeping the cat inside the compile boundary.
                aux_hidden_states_concat = torch.cat(aux_hidden_states, dim=-1)
                return logits, aux_hidden_states_concat
            return logits

        # ── On-device sampling ───────────────────────────────────────────
        # Sample on the ROW-ALIGNED pre-slice logits (see ordering note above),
        # then slice the resulting sampled tokens to this DP rank's rows.
        #
        # The sampler's per-row params (sampling_params / logit_mask) are built
        # per-DP-rank (B_local rows), while the pre-slice logits carry all
        # lm_head_dp*B_local rows. For all_greedy (the on-device sampling mode
        # used by GPT-OSS DI), the argmax path ignores those per-row params
        # entirely, so the row-count mismatch is a no-op and the distributed
        # argmax runs correctly over the row-aligned logits. For the non-greedy
        # path the per-row params would be row-misaligned, so we only take the
        # sample-before-slice route when it is safe (greedy, or no per-row
        # params); otherwise we fall back to the original slice-then-sample
        # (unchanged behavior for those paths).
        sample_pre_slice = self.lm_head_dp_size > 1 and (
            self.sampler.all_greedy or sampling_params is None
        )
        if not sample_pre_slice and self.lm_head_dp_size > 1:
            # Non-greedy DP path: preserve original ordering (slice then sample).
            logits = logits[dp_rank * B_local : (dp_rank + 1) * B_local]

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )
        if sample_pre_slice:
            sampled_tokens = sampled_tokens[dp_rank * B_local : (dp_rank + 1) * B_local]

        # ── Speculative decoding: rejection sampling ─────────────────────
        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            rejection_sampled_tokens = rejection_sampler(
                spec_decode_metadata,
                sampled_tokens,
            )
            if len(aux_hidden_states) > 0:
                aux_hidden_states_concat = torch.cat(aux_hidden_states, dim=-1)
                return (
                    rejection_sampled_tokens,
                    aux_hidden_states_concat,
                    gathered_logits,
                )
            return rejection_sampled_tokens

        # ── Standard return (with optional Eagle3 aux states) ────────────
        if len(aux_hidden_states) > 0:
            aux_hidden_states_concat = torch.cat(aux_hidden_states, dim=-1)
            return sampled_tokens, aux_hidden_states_concat, gathered_logits
        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = GptOssConfig.from_configs(hf_config, neuron_config)
        return cls(config, batch_size=1)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        if layers is not None:
            self.model.aux_hidden_state_layers = list(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        if self.model.aux_hidden_state_layers:
            return tuple(self.model.aux_hidden_state_layers)
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    # ── KV Cache Management ──────────────────────────────────────────────
    # >>> PARALLELISM: KV spec uses per-rank head counts (TP-sharded) <<<
    # <-- MODEL-SPECIFIC: sliding_window_size varies by layer

    def get_kv_spec(self):
        """Returns KV cache specification for vLLM integration."""
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

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor, torch.Tensor]]):
        """Binds pre-allocated KV cache tensors to attention layers."""
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise Exception(f"KV cache for layer {layer_name} not initialized")
            k_cache = kv_caches[layer_name][0]
            v_cache = kv_caches[layer_name][1]
            layer.self_attn.k_cache = k_cache
            layer.self_attn.v_cache = v_cache
            # Detect the swizzled packed FP8 K layout from the bound K cache:
            # packed K is [num_blocks, kv_heads, block_size // 2, head_size, 2]
            # (one rank higher than V, which is never packed).
            layer.self_attn.fp8_packed = k_cache.dim() == v_cache.dim() + 1

    # ── Weight Loading ───────────────────────────────────────────────────

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load weights from a checkpoint with pipelined data movement.

        The weight name mappings below define how HuggingFace checkpoint tensor names
        map to this model's parameter names. This is the primary place to update when
        the checkpoint format changes.

        >>> PARALLELISM: Weight loaders (attached to each parameter) handle TP sharding <<<
        <-- MODEL-SPECIFIC: The mapping between HF names and our parameter names
        """
        tp_rank = self.rank
        tp_size = self.world_size
        logger.info(
            f"load_weights: tp_rank={tp_rank}, tp_size={tp_size}, "
            f"tp_group.rank={self.tp_group.rank_in_group}, "
            f"tp_group.world_size={self.tp_group.world_size}"
        )

        mappings = dict()

        for layer_id in range(len(self.model.layers)):
            layer_prefix = f"model.layers.{layer_id}"

            # <-- MODEL-SPECIFIC: Attention weight mappings
            # HF stores separate q_proj, k_proj, v_proj → we fuse into qkv_proj
            mappings[f"{layer_prefix}.self_attn.qkv_proj_weight"] = [
                f"{layer_prefix}.self_attn.q_proj.weight",
                f"{layer_prefix}.self_attn.k_proj.weight",
                f"{layer_prefix}.self_attn.v_proj.weight",
            ]
            # Only map bias keys if model has attention bias
            if self.model.layers[0].self_attn.qkv_proj_bias is not None:
                mappings[f"{layer_prefix}.self_attn.qkv_proj_bias"] = [
                    f"{layer_prefix}.self_attn.q_proj.bias",
                    f"{layer_prefix}.self_attn.k_proj.bias",
                    f"{layer_prefix}.self_attn.v_proj.bias",
                ]
            mappings[f"{layer_prefix}.input_layernorm.weight"] = (
                f"{layer_prefix}.input_layernorm.weight"
            )
            mappings[f"{layer_prefix}.self_attn.o_proj_weight"] = (
                f"{layer_prefix}.self_attn.o_proj.weight"
            )
            if self.model.layers[0].self_attn.o_proj_bias is not None:
                mappings[f"{layer_prefix}.self_attn.o_proj_bias"] = (
                    f"{layer_prefix}.self_attn.o_proj.bias"
                )
            mappings[f"{layer_prefix}.self_attn.sinks"] = (
                f"{layer_prefix}.self_attn.sinks"
            )

            # <-- MODEL-SPECIFIC: MoE weight mappings
            # Post-attention layernorm lives inside experts module
            mappings[f"{layer_prefix}.mlp.experts.post_attention_layernorm.weight"] = (
                f"{layer_prefix}.post_attention_layernorm.weight"
            )
            mappings[f"{layer_prefix}.mlp.experts.router_weight"] = (
                f"{layer_prefix}.mlp.router.weight"
            )
            mappings[f"{layer_prefix}.mlp.experts.router_bias"] = (
                f"{layer_prefix}.mlp.router.bias"
            )
            mappings[f"{layer_prefix}.mlp.experts.gate_up_proj_bias"] = (
                f"{layer_prefix}.mlp.experts.gate_up_proj_bias"
            )
            mappings[f"{layer_prefix}.mlp.experts.down_proj_bias"] = (
                f"{layer_prefix}.mlp.experts.down_proj_bias"
            )

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        has_mxfp4 = any(
            "gate_up_proj_blocks" in k for k in checkpoint.get_tensor_names()
        )

        for layer_id in range(len(self.model.layers)):
            layer_prefix = f"model.layers.{layer_id}"
            if has_mxfp4:
                mappings[f"{layer_prefix}.mlp.experts.gate_up_proj_weight"] = [
                    f"{layer_prefix}.mlp.experts.gate_up_proj_blocks",
                    f"{layer_prefix}.mlp.experts.gate_up_proj_scales",
                ]
                mappings[f"{layer_prefix}.mlp.experts.down_proj_weight"] = [
                    f"{layer_prefix}.mlp.experts.down_proj_blocks",
                    f"{layer_prefix}.mlp.experts.down_proj_scales",
                ]
            else:
                mappings[f"{layer_prefix}.mlp.experts.gate_up_proj_weight"] = (
                    f"{layer_prefix}.mlp.experts.gate_up_proj"
                )
                mappings[f"{layer_prefix}.mlp.experts.down_proj_weight"] = (
                    f"{layer_prefix}.mlp.experts.down_proj"
                )
        rank_sharded_checkpoint = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, mappings, device
        ).state_dict

        self._load_kv_cache_scales(checkpoint, device)

        self.load_state_dict(rank_sharded_checkpoint, strict=False, assign=True)

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Lightweight weight loading used during CPU compile."""
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        checkpoint._ensure_indexed()
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
                key = f"model.layers.{layer_id}.self_attn.{scale_name}"
                if key in checkpoint._tensor_name_to_file:
                    # Invert scales: checkpoint stores scales for (tensor / scale),
                    # but the kernel quantizes via (tensor * scale), so invert once here.
                    val = 1.0 / checkpoint._get_slice(key)[:].to(
                        dtype=torch.bfloat16, device=device
                    )
                else:
                    val = torch.ones(1, dtype=torch.bfloat16, device=device)
                setattr(attn, scale_name, val.reshape(1, 1))

            attn.k_scale_float = attn.k_scale.item()
            attn.v_scale_float = attn.v_scale.item()
