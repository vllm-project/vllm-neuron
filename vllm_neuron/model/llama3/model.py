# SPDX-License-Identifier: Apache-2.0
"""
Llama BF16 Implementation
====================================

Annotated implementation of Llama for the Neuron backend.
Adapted from GPT-OSS BF16 model for dense (non-MoE) architecture.

Supported parallelism: TP, SP, DP.
Starting model: Llama-3.2-1B (dense, GQA 32Q/8KV, SiLU, tied embeddings).

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    Llama-specific. Change when porting.

PARALLELISM SHARDING:
  TP-only:  Attention heads sharded, MLP intermediate sharded
  TP+DP:    Independent replicas per DP rank, same sharding as TP-only
"""

import logging
import math

import torch
from torch import nn
from nkilib.core.utils.common_types import NormType
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.models.interfaces import SupportsEagle3

import vllm_neuron.functional as NF
from vllm_neuron.functional.attention.attention_decode import (
    _swizzle_packed_k,
    _unswizzle_packed_k,
)
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.parallel.neuron_parallel_state import shard_select
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.dtype_utils import (
    FP8_CLAMP_MAX,
    validate_fp8_segmented_supported,
)
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
    with_rank_override,
)

from transformers import PretrainedConfig
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.sampler import Sampler

import vllm_neuron.nn as neuron_nn
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.vllm.spec_decode.decorator import async_speculative_decoding

from .config import LlamaConfig
from .quantization import QuantScheme

logger = logging.getLogger(__name__)


# Reserved top-level attn_metadata key: forward_decode caches the layer-invariant
# decode mask here on layer 0 and reuses it, so the graph holds one gen_mask_tkg
# kernel. Safe because the sole full attn_metadata.items() walk (spec-decode slot
# correction) is a prologue that runs before any layer writes this key. Assumes
# one mask geometry across layers (true for llama3's single full-attention group;
# a hybrid/SWA model with per-group block_size/window would need a per-group key).
_DECODE_MASK_CACHE_KEY = "_cached_decode_mask"


def _validate_dcp_prefill_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    fp8_packed: bool = False,
) -> None:
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    k_is_fp8 = k_cache.dtype in fp8_dtypes
    v_is_fp8 = v_cache.dtype in fp8_dtypes
    # Absolute ranks, not a relative k.dim() == v.dim() + 1 test, so a future
    # layout shifting both ranks cannot silently pass.
    packed_layout = (
        fp8_packed
        and k_cache.dim() == 5
        and v_cache.dim() == 4
        and k_is_fp8
        and v_is_fp8
    )
    if (k_is_fp8 or v_is_fp8) and not packed_layout:
        raise ValueError(
            "Llama DCP prefill requires a non-FP8 KV cache unless the "
            "static-FP8 path uses a packed FP8 K cache; use "
            "--kv-cache-dtype auto or enable fp8_packed_kv"
        )


def _validate_dcp_decode_sequence_length(dcp_size: int, sequence_length: int) -> None:
    if dcp_size > 1 and sequence_length != 1:
        raise ValueError(
            "Llama DCP decode supports exactly one token per request; "
            f"got {sequence_length}. Speculative decoding with DCP is not supported."
        )


def _segmented_attention_cp_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cached_seq_len: torch.Tensor | None,
    block_size: int,
    dcp_size: int,
    dcp_rank: torch.Tensor,
    dcp_group: object,
    scale: float,
    fp8_packed: bool = False,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """DCP prefill attention: AllGather head-sharded Q, attend against this
    rank's local KV shard (read from the paged cache — the current tokens were
    already written there before this call), LSE-correct across the group,
    ReduceScatter heads back. prior_tokens is the cached prefix per rank
    (cached_seq_len // dcp_size). With fp8_packed, K is read from the packed
    layout and k_scale/v_scale dequantize the cached KV."""
    local_prior = cached_seq_len // dcp_size if cached_seq_len is not None else None
    return NF.segmented_attention_cp(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_table,
        prior_tokens=local_prior,
        block_size=block_size,
        cp_rank=dcp_rank.reshape(1, 1).to(dtype=torch.int32, device=q.device),
        cp_world_size=dcp_size,
        cp_kv_cache_interleave_size=block_size,
        cp_group=dcp_group,
        scale=scale,
        tp_q=True,
        tp_out=True,
        fp8_packed=fp8_packed,
        k_scale=k_scale,
        v_scale=v_scale,
    )


# =============================================================================
# Section 1: RMS Normalization
# <-- MODEL-SPECIFIC: Llama uses standard RMSNorm (no hidden dim padding)
# =============================================================================
class LlamaRMSNorm(nn.Module):
    """RMS Normalization.

    <-- MODEL-SPECIFIC: Standard RMSNorm without padding. Other models may use
    LayerNorm or RMSNorm with padded hidden dim.
    """

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
# Section 2: Rotary Position Embedding (Llama3 RoPE)
# <-- MODEL-SPECIFIC: Llama uses Llama3-style RoPE scaling
# =============================================================================


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """<-- MODEL-SPECIFIC: Interleaved RoPE (rotate_half style)."""
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

    <-- MODEL-SPECIFIC: Llama uses rotate_half style (interleaved). GPT-OSS uses
    non-interleaved (split in half). The RoPE variant must match the checkpoint.

    Args:
        q: [Nh, T, Dh], k: [Nkv, T, Dh]
        cos: [T, Dh], sin: [T, Dh]

    Returns:
        Rotated (q, k) tensors
    """
    cos = cos.unsqueeze(0)  # [1, T, Dh]
    sin = sin.unsqueeze(0)  # [1, T, Dh]
    return _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)


class LlamaRotaryEmbedding(nn.Module):
    """Rotary Position Embedding with Llama3 scaling.

    <-- MODEL-SPECIFIC: The Llama3 frequency scaling (piecewise linear blending
    between original and scaled frequencies) is specific to Llama 3.1+. Other
    models may use standard RoPE, YaRN, or NTK-aware scaling.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.rope_theta = config.rope_parameters["rope_theta"]
        self.head_dim = config.head_dim
        self.rope_parameters = config.rope_parameters

        rope_type = self.rope_parameters.get("rope_type")
        if rope_type == "llama3":
            self.scaling_factor = self.rope_parameters.get("factor", 1.0)
            self.low_freq_factor = self.rope_parameters.get("low_freq_factor", 1.0)
            self.high_freq_factor = self.rope_parameters.get("high_freq_factor", 1.0)
            self.orig_max_position = self.rope_parameters.get(
                "original_max_position_embeddings", 8192
            )
        elif rope_type not in (None, "default"):
            raise NotImplementedError(
                f"rope_type '{rope_type}' is not supported, only 'llama3' is supported."
            )

    def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
                / self.head_dim
            )
        )
        if self.rope_parameters.get("rope_type") == "llama3":
            inv_freq = self._apply_llama3_scaling(inv_freq)
        return inv_freq

    def _apply_llama3_scaling(self, inv_freq: torch.Tensor) -> torch.Tensor:
        """<-- MODEL-SPECIFIC: Llama3 piecewise frequency scaling."""
        low_freq_wavelen = self.orig_max_position / self.low_freq_factor
        high_freq_wavelen = self.orig_max_position / self.high_freq_factor
        wave_len = 2 * math.pi / inv_freq

        if self.low_freq_factor != self.high_freq_factor:
            smooth = (self.orig_max_position / wave_len - self.low_freq_factor) / (
                self.high_freq_factor - self.low_freq_factor
            )
        else:
            smooth = torch.zeros_like(wave_len)

        return torch.where(
            wave_len < high_freq_wavelen,
            inv_freq,
            torch.where(
                wave_len > low_freq_wavelen,
                inv_freq / self.scaling_factor,
                (1 - smooth) * inv_freq / self.scaling_factor + smooth * inv_freq,
            ),
        )

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin embeddings for given positions.

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
#        MODEL-SPECIFIC (GQA config, no bias, no sinks, no sliding window)
# =============================================================================


class LlamaAttention(nn.Module):
    """Multi-head attention with TP head sharding.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads are sharded across TP ranks
    - KV heads are replicated when fewer than TP size (GQA)
    - Prefill: all-gather input → QKV proj → attention → O proj → reduce-scatter
    - Decode: fused megakernel with TP all-reduce

    <-- MODEL-SPECIFIC:
    - GQA with separate Q and KV head counts
    - No attention bias, no sinks, no sliding window
    - Llama3 RoPE (rotate_half style)
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5

        # >>> PARALLELISM: TP group setup <<<
        from vllm.distributed.parallel_state import get_dcp_group

        dcp_group = get_dcp_group()
        self.dcp_size = dcp_group.world_size
        self.dcp_group = dcp_group if self.dcp_size > 1 else None

        # >>> PARALLELISM: DCP — prefill uses the SAME scheme as decode <<<
        # Both prefill and decode:
        #   - Shard Q/K/V/O weights across the FULL TP group (no duplicated
        #     Q heads). Each rank owns num_q_heads/TP heads + a replicated KV
        #     head shard.
        #   - Token-shard the KV cache across the DCP group (vllm `_DCP`),
        #     interleaved by cp_rank = tp_rank % dcp_size.
        #   - Compute attention by AllGathering Q across heads, attending the
        #     gathered Q against the local KV shard, LSE-correcting across the
        #     DCP group, and ReduceScattering heads back to their owner.
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # DCP rank: position within the DCP group = which interleaved token
        # slice this rank owns (matches decode: tp_rank % dcp_size).
        self.cp_chunk_idx = self.dcp_group.rank_in_group if self.dcp_size > 1 else 0

        # >>> PARALLELISM: Dependent DP setup (decode-only Q/O sharding across DP) <<<
        self.attention_dp_size = (
            config.neuron_config.attention_dp_size if config.neuron_config else 1
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_attention_dp_group,
            get_neuron_attention_dp_rank,
        )

        self.attention_dp_group = get_neuron_attention_dp_group()
        self.attention_dp_rank = get_neuron_attention_dp_rank()

        # >>> PARALLELISM: Attention TP supergroup (TP * attn_dp) <<<
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_attention_tp_group,
        )

        self.attn_tp_group = get_neuron_attention_tp_group()

        # Effective sharding degree for Q/O (TP for standard, TP*DDP for attention DP)
        effective_q_shards = self.world_size * self.attention_dp_size

        # >>> PARALLELISM: Head sharding calculation <<<
        # Q heads per rank: sharded across TP*DDP for decode
        self.num_attention_heads_per_rank = (
            self.num_attention_heads // effective_q_shards
        )

        # KV heads per rank: determines cache size (always full TP amount).
        # When num_kv_heads > TP and attention DP is enabled, KV weights are also sharded
        # across attention DP (fewer projected heads). The cache stores the gathered result
        # (full TP heads) so attention reads locally without a2a'ing prior context.
        self.kv_needs_a2a = (
            self.attention_dp_size > 1
            and self.num_key_value_heads > self.world_size
            and self.num_key_value_heads % effective_q_shards == 0
        )

        # Cache sizing: full TP heads (unchanged by attention DP)
        if self.world_size >= self.num_key_value_heads:
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            self.num_key_value_heads_per_rank = (
                self.num_key_value_heads // self.world_size
            )
            self.num_kv_replicas = 1

        # Weight sizing: may be smaller when KV is dependent-DP-sharded
        self.num_kv_heads_for_weight = (
            self.num_key_value_heads // effective_q_shards
            if self.kv_needs_a2a
            else self.num_key_value_heads_per_rank
        )

        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // self.num_kv_heads_for_weight
        )

        # Q/KV heads after all-to-all: each rank receives heads from all attention DP peers
        self.num_q_heads_after_a2a = (
            self.num_attention_heads_per_rank * self.attention_dp_size
        )
        self.num_kv_heads_after_a2a = (
            self.num_kv_heads_for_weight * self.attention_dp_size
            if self.kv_needs_a2a
            else self.num_key_value_heads_per_rank
        )

        # >>> PARALLELISM: QKV weight shapes <<<
        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_kv_heads_for_weight * self.head_dim
        qkv_size = q_size + 2 * kv_size
        o_proj_in_features = (
            self.num_attention_heads * self.head_dim
        ) // effective_q_shards

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

        # Packed FP8 K cache layout. Auto-detected in bind_kv_cache() from the
        # cache rank (packed K is one dim larger than V); see _write_kv_cache /
        # forward_decode for the swizzle handling.
        self.fp8_packed = False

        # KV cache quantization scales are set during weight loading.
        # We also need floats since tensor.item() causes a graph break,
        # and kernels currently use floats for perf reasons.
        self.register_buffer("k_scale", None, persistent=False)
        self.register_buffer("v_scale", None, persistent=False)

        self.k_scale_float = 1.0
        self.v_scale_float = 1.0

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach weight loaders for checkpoint → parameter transformation.

        >>> PARALLELISM: Weight loaders handle TP sharding of checkpoint tensors.
        <-- MODEL-SPECIFIC: Llama stores separate Q, K, V weights (no bias).

        With attention DP: Q/O are sharded across TP*attention DP (not just TP). KV may also
        be sharded across attention DP when num_kv_heads > TP (auto-detected via
        kv_needs_a2a). The effective rank for sharded weights is
        attention_dp_rank + tp_rank * attention_dp_size.
        """

        # Effective rank for weight sharding: when attention DP is enabled,
        # Q/O are sharded across TP * attention_dp using an interleaved rank.
        # When disabled (ddp_size=1), effective_rank == tp_rank (no-op).
        ddp = self.attention_dp_size
        effective_q_rank = self.attention_dp_rank + self.rank * ddp
        o_shard_size = (self.num_attention_heads * self.head_dim) // (
            self.world_size * ddp
        )

        # QKV: Q and KV need different shard ranks with attention DP
        # (Q uses effective_rank, KV uses tp_rank). The loader handles this internally.
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
                shard_dim=1,
                num_shards=self.world_size,
                is_storage_transposed=True,
                num_kv_replicas=self.num_kv_replicas,
                attention_dp_rank=self.attention_dp_rank,
                attention_dp_size=ddp,
                kv_sharded_across_attention_dp=self.kv_needs_a2a,
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            with_rank_override(
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=o_shard_size,
                    num_shards=self.world_size * ddp,
                    is_storage_transposed=True,
                ),
                rank=effective_q_rank,
            ),
        )

    # ── Forward dispatch ─────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
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
                attn_metadata,
            )
        else:
            # >>> PARALLELISM: All-gather from SP before attention <<<
            if self.world_size > 1:
                # >>> PARALLELISM: SP gather (tp_group): S/TP → S <<<
                # DCP uses the same full-S gather as standard prefill; the
                # per-rank KV-shard split happens inside forward_prefill so the
                # query layout matches decode (full Q, sharded KV).
                hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

            return self.forward_prefill(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
            )

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
        2. RoPE                    <-- MODEL-SPECIFIC: Llama3 RoPE
        3. KV cache update         >>> PARALLELISM: per-rank cache <<<
        4. Flash attention          <-- MODEL-SPECIFIC: no sinks, no sliding window
        5. Output projection       >>> PARALLELISM: reduce-scatter after O proj <<<
        """
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)
        if self.dcp_size > 1:
            _validate_dcp_prefill_kv_cache(self.k_cache, self.v_cache)

        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # ── Step 1: QKV Projection (with fused RoPE) ─────────────────────
        # <-- MODEL-SPECIFIC: Llama3 RoPE — fused into the qkv_proj kernel.
        cos, sin = position_embeddings
        cos_cache = cos.unsqueeze(0)
        sin_cache = sin.unsqueeze(0)

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

        def _write_kv_cache(k_to_cache, v_to_cache, slot_map):
            """Write K/V to paged cache using slot mapping."""
            blk_idx = slot_map // block_size
            pos_idx = slot_map % block_size
            num_slots = slot_map.shape[0]
            nkh = self.num_key_value_heads_per_rank

            if self.k_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                k_f = (
                    (k_to_cache.reshape(-1, self.head_dim) * self.k_scale)
                    .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                    .to(self.k_cache.dtype)
                )
                v_f = (
                    (v_to_cache.reshape(-1, self.head_dim) * self.v_scale)
                    .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                    .to(self.k_cache.dtype)
                )
            else:
                k_f = k_to_cache.reshape(-1, self.head_dim).to(self.k_cache.dtype)
                v_f = v_to_cache.reshape(-1, self.head_dim).to(self.v_cache.dtype)

            h_idx = torch.arange(
                nkh, dtype=torch.long, device=k_to_cache.device
            ).repeat_interleave(num_slots)
            index = (blk_idx.repeat(nkh), h_idx, pos_idx.repeat(nkh))
            # V is never packed → scatters directly.
            self.v_cache.index_put_(index, v_f)
            if self.fp8_packed:
                # Packed K cache [blocks, Nkh, block_size // 2, Dh, 2]: the
                # swizzle interleaves adjacent token positions into the trailing
                # size-2 dim, so a per-token scatter isn't expressible directly.
                # Un-swizzle to the standard layout, scatter, then re-swizzle
                # back in place (matching the decode kernel's write).
                k_unpacked = _unswizzle_packed_k(self.k_cache)
                k_unpacked.index_put_(index, k_f)
                self.k_cache.copy_(_swizzle_packed_k(k_unpacked))
            else:
                self.k_cache.index_put_(index, k_f)

        # >>> PARALLELISM: Each rank computes only its sharded Q/K/V heads <<<
        # Segmented prefill (no DCP) folds the post-RoPE K/V cache write into
        # the qkv kernel. Kernel returns (q, k_cache, v_cache); the FX
        # aliasing pass threads the in-place cache write back to self.k_cache
        # / self.v_cache for the downstream segmented_attention. K/V locals
        # are left as None on this branch since segmented_attention reads
        # from the cache, not from local K/V tensors.
        if kv_segment_size and self.dcp_size <= 1:
            # attention_segmented_cte cannot read a non-packed FP8 K cache;
            # fail fast on the unsupported combo (see helper for rationale).
            validate_fp8_segmented_supported(kv_is_fp8, self.fp8_packed)

            # In-kernel post-RoPE K/V write for the bf16 and the packed-FP8 cache
            # alike; segmented_attention below reads the cache.
            # qkv_cte divides by the scale arg and self.k_scale is the pre-inverted
            # 1/s_ckpt, so pass its reciprocal to store K*self.k_scale as decode does.
            k_scale_write = (1.0 / self.k_scale) if kv_is_fp8 else None
            v_scale_write = (1.0 / self.v_scale) if kv_is_fp8 else None

            num_blocks_total = self.k_cache.shape[0]
            max_slot = num_blocks_total * block_size
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
                num_kv_heads=self.num_kv_heads_for_weight,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                k_scale=k_scale_write,
                v_scale=v_scale_write,
                fp8_max=FP8_CLAMP_MAX if kv_is_fp8 else None,
                fp8_min=-FP8_CLAMP_MAX if kv_is_fp8 else None,
                kv_dtype=self.k_cache.dtype if kv_is_fp8 else None,
                use_block_kv=True,
                fp8_packed=self.fp8_packed,
                block_size=block_size,
                slot_mapping=slot_mapping_clamped.unsqueeze(0),
            )
            q = (
                q_hbm.squeeze(0)
                .view(tokens, self.num_attention_heads_per_rank, self.head_dim)
                .transpose(0, 1)
            )
            k = v = None
        else:
            qkv = NF.qkv_proj(
                hidden=hidden_states.unsqueeze(0),
                qkv_weights=self.qkv_proj_weight,
                bias=None,
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

        # RoPE is fused into qkv_proj above — no external application needed.

        # ── Step 4: Write KV cache + Attention ────────────────────────────
        # <-- MODEL-SPECIFIC: No sinks, no sliding window (standard causal attention)
        if self.dcp_size > 1:
            # >>> PARALLELISM: DCP prefill — same scheme as decode <<<
            # Q/K/V were projected for the full S tokens with full-TP head
            # sharding (q: [Nh_q, S, Dh], k/v: [Nkv, S, Dh]). Slice K/V to this
            # rank's interleaved token shard, write that shard to the per-rank
            # cache, then run segmented attention which AllGathers Q across
            # heads, attends the gathered Q against the local KV shard,
            # LSE-corrects across the DCP group, and ReduceScatters heads back.

            # segmented_attention_cp has no packed-FP8 read path; reject the
            # combination rather than silently misreading the swizzled cache.
            assert not self.fp8_packed, (
                "packed FP8 KV cache is not supported with DCP prefill "
                "(segmented_attention_cp cannot read the swizzled K layout)"
            )
            # >>> PARALLELISM: write the FULL sequence to the per-rank cache.
            # slot_mapping is full-length (owned interleaved tokens → real slots,
            # non-owned → reserved null block 0), so the scatter drops non-owned
            # tokens harmlessly. Attention reads this rank's owned KV from cache
            # (segmented_attention_cp is cache-only). No K/V slicing. <<<
            _write_kv_cache(k, v, slot_mapping)

            # >>> PARALLELISM: Segmented attention with head AllGather + LSE <<<
            # <-- MODEL-SPECIFIC: no sinks, no sliding window
            attn_output = _segmented_attention_cp_prefill(  # [Nh_q, Dh, S]
                q=q,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_table=block_table,
                cached_seq_len=cached_seq_len,
                block_size=block_size,
                dcp_size=self.dcp_size,
                dcp_rank=self.dcp_group.rank_in_group_tensor,
                dcp_group=self.dcp_group,
                scale=self.scaling,
            )
        elif kv_segment_size:
            # K/V cache write was folded into the qkv kernel above; just
            # attend over the cache segments here.
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
                # Cache is in the decode-packed layout; segmented_attention
                # converts internally. K-scale dequant stays fused into scale.
                fp8_packed=self.fp8_packed,
                allow_vector_exp=False,
            )  # [Nh, Dh, T]
        else:
            _write_kv_cache(k, v, slot_mapping)

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
                tp_q=False,
                tp_out=True,
                allow_vector_exp=False,
            )  # [Nh, Dh, T]

        # ── Step 5: Output Projection ────────────────────────────────────
        attn_output = attn_output.unsqueeze(0)  # [1, Nh, Dh, T]
        attn_output = NF.o_proj(attn_output, self.o_proj_weight, None)  # [1, T, H]
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
        attn_metadata: object,
    ):
        """Decode: fused megakernel for single-token generation.

        >>> PARALLELISM: The megakernel handles TP internally. <<<
        >>> PARALLELISM: With attention DP, Q/O are sharded across TP*attention DP. <<<

        <-- MODEL-SPECIFIC: No sink tokens, no sliding window, no attention bias.
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
        _validate_dcp_decode_sequence_length(self.dcp_size, S_decode)

        hidden_states = hidden_states.to(self.dtype)
        S_ctx = max_blocks_per_seq * block_size
        nkh = self.num_key_value_heads_per_rank

        X = hidden_states.view(B, S_decode, hidden)

        # Prepare RoPE for megakernel format: [T, Hd] → [Hd/2, B, S]
        # With attention DP, RoPE is applied after the a2a on local batch only
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

        # <-- MODEL-SPECIFIC: Standard causal mask, no sliding window
        # DCP decode engages whenever dcp_size > 1 — prefill now writes the
        # same token-sharded (interleaved) cache, so decode always reads it the
        # DCP way (gather Q heads + LSE correction).
        dcp_decode_active = self.dcp_size > 1
        # With DCP decode, Q is gathered across DCP group → more heads in attention
        mask_q_heads = self.num_q_heads_after_a2a * (
            self.dcp_size if dcp_decode_active else 1
        )
        # With DCP decode, cache is interleaved → use flat slot-based mask.
        # Compute local filled count. The active token (current decode step)
        # is placed at slot s_prior-1 by stage 7b on ALL ranks. To avoid
        # double-counting in LSE correction, only the owning rank includes it
        # by setting local_filled = S_ctx (covering the last slot).
        local_filled = None
        dcp_active_mask = None
        if dcp_decode_active:
            # Per-request prior length: each line's own decode position, not the
            # batch-max cached_seq_len (which over-admits stale KV for shorter
            # lines at bs>1). Mirrors model_static_fp8.forward_decode.
            n = positions.view(B_local, S_decode)[:, 0:1].to(torch.float32)
            W = self.dcp_size
            R = self.dcp_group.rank_in_group_tensor.to(torch.float32)
            vblock = block_size * W

            full_vblocks = torch.div(n, vblock, rounding_mode="trunc")
            remaining = n - full_vblocks * vblock
            local_filled = full_vblocks * block_size + torch.clamp(
                remaining - R * block_size,
                min=0,
                max=block_size,
            )

            owner = torch.div(remaining, block_size, rounding_mode="trunc") % W
            dcp_active_mask = (owner == R).float()

        pos_ids = positions.view(1, B_local * S_decode)
        # Layer-invariant here too: mask_q_heads/local_filled/dcp_active_mask are
        # __init__ constants + per-request positions. Cache/reuse (see key def).
        attention_mask = attn_metadata.get(_DECODE_MASK_CACHE_KEY)
        if attention_mask is None:
            attention_mask = NF.gen_attention_decode_mask(
                pos_ids=pos_ids.to(torch.float32),
                bs=B_local,
                q_head=mask_q_heads,
                s_active=S_decode,
                s_prior=S_ctx,
                start_pos=None,
                block_len=block_size,
                local_filled_slots=local_filled,
                dcp_active_mask=dcp_active_mask if dcp_decode_active else None,
            )
            attn_metadata[_DECODE_MASK_CACHE_KEY] = attention_mask

        active_blocks_table = block_table

        # Packed FP8 is selected by server config (neuron_config.fp8_packed_kv):
        # when enabled, the runner allocates the swizzled K layout and
        # bind_kv_cache sets self.fp8_packed, so the decode kernel always reads
        # the packed cache directly. There is no per-bucket auto-adaptation —
        # the layout is fixed by config, and a packed config must use a bucket
        # geometry the kernel can pack (even decode batch >= 2 so the resized
        # block_len stays >= 2).

        # >>> PARALLELISM: Fused megakernel with TP-sharded weights <<<
        # In-kernel KV cache update (update_cache=True): the kernel writes K/V
        # in place and the FX aliasing pass threads the write back to
        # self.k_cache/self.v_cache (including across the recurrent passes of
        # Eagle3 fused propose). With update_cache=True the API returns only the
        # attention output.
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
            attention_mask=attention_mask,
            # Kernel currently requires fusing the K scale dequantization into softmax scale for KV quantization
            softmax_scale=self.scaling / self.k_scale_float,
            sink=None,
            update_cache=True,
            kv_cache_update_idx=slot_mapping.view(B_local, S_decode).to(torch.uint32),
            fp8_packed=self.fp8_packed,
            # Kernel currently requires fusing the V scale dequantization into W_out for KV quantization
            W_out=self.o_proj_weight / self.v_scale_float,
            bias_out=None,
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
            attention_dp_rank_tensor=(
                self.attention_dp_group.rank_in_group_tensor
                if self.attention_dp_size > 1
                else None
            ),
            kv_needs_a2a=self.kv_needs_a2a,
            dcp_size=self.dcp_size if dcp_decode_active else 1,
            dcp_group=self.dcp_group if dcp_decode_active else None,
        )

        # >>> PARALLELISM: Sum O-proj partials across TP * attn_dp supergroup <<<
        self.attn_tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 4: MLP (Dense)
# Mixed: PARALLELISM (TP intermediate sharding, SP collectives) +
#        MODEL-SPECIFIC (SiLU activation, gate/up/down structure)
# =============================================================================


class LlamaMLP(nn.Module):
    """Dense MLP with TP intermediate sharding.

    >>> PARALLELISM: TP <<<
    - gate_proj, up_proj: hidden → intermediate/TP per rank
    - down_proj: intermediate/TP → hidden per rank
    - Prefill: all-gather before → compute → reduce-scatter after (SP)
    - Decode: compute → all-reduce

    With mlp_dp_size > 1, intermediate is sharded across TP * mlp_dp_size.

    <-- MODEL-SPECIFIC:
    - SiLU activation (gate * silu(up))
    - No bias on any projection
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: MLP TP group (TP * mlp_dp_size) + DP column <<<
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_draft_mlp_dp_group,
            get_neuron_draft_mlp_tp_group,
            get_neuron_mlp_tp_group,
            get_neuron_mlp_dp_group,
        )

        if config.use_draft_mlp_groups:
            mlp_tp_group = get_neuron_draft_mlp_tp_group()
            mlp_dp_group = get_neuron_draft_mlp_dp_group()
        else:
            mlp_tp_group = get_neuron_mlp_tp_group()
            mlp_dp_group = get_neuron_mlp_dp_group()
        self.mlp_tp_group = mlp_tp_group
        self.mlp_tp_size = mlp_tp_group.world_size
        self.mlp_tp_rank = mlp_tp_group.rank_in_group
        self.mlp_dp_group = mlp_dp_group
        self.mlp_dp_size = mlp_dp_group.world_size

        self.hidden_size = config.hidden_size
        # >>> PARALLELISM: Intermediate dim sharded across TP * mlp_dp_size <<<
        self.intermediate_size_per_rank = config.intermediate_size // self.mlp_tp_size

        # <-- MODEL-SPECIFIC: Separate gate and up projections (SwiGLU pattern)
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
        """>>> PARALLELISM: TP (and MLP DP) sharding of MLP weights. <<<"""
        gate_up_loader = sharding_weight_loader(
            shard_dim=1,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.mlp_tp_size,
            is_storage_transposed=True,
        )
        down_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.mlp_tp_size,
            is_storage_transposed=True,
        )

        gate_up_loader = with_rank_override(gate_up_loader, rank=self.mlp_tp_rank)
        down_loader = with_rank_override(down_loader, rank=self.mlp_tp_rank)

        set_weight_loader(self.gate_proj_weight, gate_up_loader)
        set_weight_loader(self.up_proj_weight, gate_up_loader)
        set_weight_loader(self.down_proj_weight, down_loader)

    def forward(
        self,
        hidden_states: torch.Tensor,
        is_prefill: bool,
        *,
        ln_w: torch.Tensor | None = None,
        eps: float = 1e-6,
        fused_add_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        >>> PARALLELISM: SP all-gather/reduce-scatter for prefill, TP supergroup
        all-reduce for decode. Both paths use the MLP TP group (TP *
        mlp_dp_size), because the intermediate dimension shards across that
        group. Caller handles DP batch transitions. <<<
        """
        if is_prefill and fused_add_tensor is not None:
            raise ValueError("Fused MLP residual add is only supported for decode")

        # >>> PARALLELISM: All-gather from SP for full sequence. Use the MLP TP
        # group (TP * mlp_dp_size), because that is the group the intermediate
        # dimension shards across. A gather over the plain TP group gives each
        # rank only the rows of its own DP column, so the reduce-scatter below
        # then sums a subset of the shards. A mlp_dp_size of 1 makes this group
        # equal the plain TP group, which keeps the plain SP path. <<<
        if is_prefill and self.mlp_tp_size > 1:
            hidden_states = self.mlp_tp_group.all_gather(hidden_states, dim=0)

        result = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
            eps=eps,
            ln_w=ln_w.view(1, -1) if ln_w is not None else None,
            norm_type=NormType.RMS_NORM if ln_w is not None else NormType.NO_NORM,
            fused_add_tensor=fused_add_tensor,
            store_fused_add_result=fused_add_tensor is not None,
        )
        if fused_add_tensor is not None:
            output, fused_add_result = result
        else:
            output = result

        # >>> PARALLELISM: Combine TP (+ MLP DP) shards. Both branches reduce
        # over the MLP TP group, because every rank in that group holds one
        # shard of the intermediate dimension. The reduce-scatter returns the
        # same row count that the all-gather above consumed, so the caller keeps
        # its SP layout. <<<
        if is_prefill:
            if self.mlp_tp_size > 1:
                output = self.mlp_tp_group.reduce_scatter(output, dim=0)
        else:
            self.mlp_tp_group.all_reduce(output)

        if fused_add_tensor is not None:
            return output, fused_add_result
        return output


# =============================================================================
# Section 5: Decoder Layer
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
    rank_tensor = current_group.rank_in_group_tensor.to(torch.int64)
    if current_dp > target_dp:
        per_dp = x.shape[dim] // current_dp
        length = target_dp * per_dp
        start = (rank_tensor // target_dp) * target_dp * per_dp
        idx = torch.arange(length, device=x.device, dtype=torch.int64) + start
        return x.index_select(dim, idx)
    # Up: go through local first to avoid duplicates from overlapping gathered state,
    # then all-gather to target.
    per_dp = x.shape[dim] // current_dp
    start = rank_tensor * per_dp
    idx = torch.arange(per_dp, device=x.device, dtype=torch.int64) + start
    x = x.index_select(dim, idx)
    return target_group.all_gather(x, dim=dim)


class LlamaDecoderLayer(nn.Module):
    """Single transformer decoder layer.

    Architecture (MODEL-SPECIFIC):
        hidden_states → RMSNorm → Attention → residual → RMSNorm → MLP → residual

    Parallelism (TP + SP):
        - Input arrives in SP layout (T/world_size tokens per rank) during prefill
        - Decode: _dp_transition handles batch state between modules
        - Residual connection operates at whatever gathered state the module outputs
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        # <-- MODEL-SPECIFIC: Pre-attention and pre-MLP RMSNorm
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        # <-- QUANTIZATION: pick attention/MLP implementations via the
        # quant_spec dispatcher. ``quant_spec=None`` returns the bf16
        # classes so this call site is byte-identical to the pre-dispatch
        # behaviour.
        from .quantization import resolve_attention_mlp_classes

        attn_cls, mlp_cls = resolve_attention_mlp_classes(config.quant_spec, layer_idx)
        self.self_attn = attn_cls(config, layer_idx=layer_idx)
        self.mlp = mlp_cls(config)

        # Record the MLP scheme + RMSNorm eps so the prefill path can
        # branch on the decoder-layer contract (quant MLPs fuse the
        # post-attention RMSNorm internally) without re-resolving.
        self._mlp_scheme = (
            config.quant_spec.get_scheme(layer_idx, prefix="mlp")
            if config.quant_spec is not None
            else QuantScheme.NONE
        )
        self.rms_norm_eps = config.rms_norm_eps

        self.layer_idx = layer_idx

        # >>> PARALLELISM: DP sizes for batch state transitions <<<
        nc = config.neuron_config
        self.attn_dp = nc.attention_dp_size if nc else 1
        self.mlp_dp = nc.mlp_dp_size if nc else 1

        # DP column groups for transitions
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_attention_dp_group,
            get_neuron_draft_mlp_dp_group,
            get_neuron_mlp_dp_group,
        )

        self.attn_dp_group = get_neuron_attention_dp_group()
        if config.use_draft_mlp_groups:
            self.mlp_dp_group = get_neuron_draft_mlp_dp_group()
        else:
            self.mlp_dp_group = get_neuron_mlp_dp_group()

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

        if not is_decode:
            return self._forward_prefill(
                hidden_states, positions, position_embeddings, attn_metadata
            )

        # ── Decode: batch state transitions between modules ──
        # Input arrives at mlp_dp (from previous layer's MLP or embedding transition).

        # Transition mlp_dp → attn_dp
        hidden_states = _dp_transition(
            hidden_states, self.mlp_dp_group, self.attn_dp_group
        )

        # ── Self Attention ───────────────────────────────────────────────
        residual = hidden_states
        # <-- QUANTIZATION: FP8 attention fuses pre-attention RMSNorm
        # with static activation quant on decode via rmsnorm_X_enabled=True.
        if self._mlp_scheme is QuantScheme.FP8_STATIC_PER_TENSOR:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
                ln_w=self.input_layernorm.weight,
                eps=self.rms_norm_eps,
            )
        else:
            # <-- MODEL-SPECIFIC: Pre-attention RMSNorm
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
            )
        # Attention outputs at attn_dp (supergroup all-reduce, stays gathered)

        # Residual add at attn_dp, then transition once to mlp_dp
        hidden_states = residual + hidden_states
        hidden_states = _dp_transition(
            hidden_states, self.attn_dp_group, self.mlp_dp_group
        )

        # ── MLP Feed-Forward ─────────────────────────────────────────────
        residual = hidden_states
        # <-- QUANTIZATION: quant MLPs fuse the pre-MLP RMSNorm with the
        # static activation quant on decode via ``norm_type=RMS_NORM``
        # inside :func:`NF.mlp`. The bf16 path keeps the explicit norm
        # call; the FP8 path must skip it here and forward ``ln_w``/``eps``.
        if self._mlp_scheme is QuantScheme.FP8_STATIC_PER_TENSOR:
            hidden_states = self.mlp(
                hidden_states,
                is_prefill=False,
                ln_w=self.post_attention_layernorm.weight,
                eps=self.rms_norm_eps,
            )
        else:
            # <-- MODEL-SPECIFIC: Pre-MLP RMSNorm
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states, is_prefill=False)
        # MLP outputs at mlp_dp (supergroup all-reduce, stays gathered)
        hidden_states = residual + hidden_states

        return hidden_states

    def _forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        # ── Self Attention ───────────────────────────────────────────────
        residual = hidden_states
        if self._mlp_scheme is QuantScheme.FP8_STATIC_PER_TENSOR:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
                ln_w=self.input_layernorm.weight,
                eps=self.rms_norm_eps,
            )
        else:
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
            )
        hidden_states = residual + hidden_states

        # ── MLP Feed-Forward ─────────────────────────────────────────────
        residual = hidden_states
        # <-- QUANTIZATION: quant MLPs own the post-attention RMSNorm on
        # prefill (they fuse it with the static activation quant via
        # ``NF.rmsnorm_quant``). The bf16 path keeps the explicit norm call.
        if self._mlp_scheme is QuantScheme.FP8_STATIC_PER_TENSOR:
            hidden_states = self.mlp(
                hidden_states,
                is_prefill=True,
                ln_w=self.post_attention_layernorm.weight,
                eps=self.rms_norm_eps,
            )
        else:
            # <-- MODEL-SPECIFIC: Pre-MLP RMSNorm
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states, is_prefill=True)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Section 6: Model Backbone
# Mixed: PARALLELISM (SP chunking / gathering) +
#        MODEL-SPECIFIC (embedding, layer stack, final norm)
# =============================================================================


class LlamaModel(nn.Module):
    """Llama transformer backbone.

    >>> PARALLELISM: SP (Sequence Parallelism) <<<
    During prefill:
    - After embedding: chunk sequence across TP ranks (each gets T/world_size)
    - After all layers: all-gather to reconstruct full sequence
    During decode:
    - No SP; all ranks process all tokens
    """

    def __init__(self, config: LlamaConfig):
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
        self.embedding_tp_group = emb_tp_group
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
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # <-- MODEL-SPECIFIC: Final RMSNorm
        self.norm = LlamaRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = LlamaRotaryEmbedding(config)

        emb_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.embed_tokens.vocab_size_per_rank,
            num_shards=self.embed_tokens.tp_size,
            is_storage_transposed=False,
        )
        emb_loader = with_rank_override(emb_loader, rank=self.embedding_tp_rank)
        set_weight_loader(self.embed_tokens.weight, emb_loader)

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
        encoder_embedding_buffer: torch.Tensor | None = None,
        encoder_block_ids: torch.Tensor | None = None,
        encoder_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
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
        # Pass `rank` so vocab-shard offsets stay as runtime tensor ops; the
        # Python-int fallback would bake per-rank literals into each NEFF and
        # defeat cache reuse across ranks. In production, `rank` is always
        # supplied by the runner (see `neuron_model_runner.py`'s `rank=
        # self.rank_tensor` callsites); the `None` fallback exists only for
        # CPU-mode unit tests that drive this forward directly.
        emb_rank = rank
        if rank is not None and self.embedding_dp_size > 1:
            emb_rank = self.embedding_tp_group.rank_in_group_tensor.to(rank.dtype)
        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=emb_rank
        )
        # Decode: embedding output at emb_dp gathered state

        # >>> PARALLELISM: Transition emb_dp → mlp_dp for decoder layers <<<
        if not is_prefill:
            hidden_states = _dp_transition(
                hidden_states, self.embedding_dp_group, self.mlp_dp_group
            )

        # >>> PARALLELISM: SP prompt-embed path <<<
        # Shard inputs_embeds/is_token_ids to match SP layout before merging.
        if (
            is_prefill
            and self.world_size > 1
            and inputs_embeds is not None
            and is_token_ids is not None
        ):
            inputs_embeds = shard_select(inputs_embeds, 0, self.tp_group)
            is_token_ids = shard_select(is_token_ids, 0, self.tp_group)

        # <-- MODEL-SPECIFIC: Llama uses hidden dim as-is, so no prompt-embed
        # dim padding step is needed before merge.
        hidden_states = NF.merge_prompt_embeds(
            hidden_states, inputs_embeds, is_token_ids
        )

        # Multimodal prefill: scatter the encoder cache rows into the placeholder
        # positions. Decode has no encoder inputs. The merge does the SP remap.
        if (
            is_prefill
            and encoder_embedding_buffer is not None
            and encoder_block_ids is not None
            and encoder_positions is not None
        ):
            hidden_states, _ = NF.merge_multimodal_embeddings(
                hidden_states,
                encoder_embedding_buffer,
                encoder_block_ids,
                encoder_positions,
                self.tp_group.rank_in_group_tensor,
            )

        position_embeddings = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )

        aux_hidden_states = []
        for idx, decoder_layer in enumerate(self.layers):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states)
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
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
# Section 7: Language Model Head
# Mixed: PARALLELISM (column-parallel LM head) +
#        MODEL-SPECIFIC (tied embeddings, sampling)
# =============================================================================


def _prefix_mappings(model: nn.Module, mappings: dict, prefix: str) -> dict:
    """Add ``prefix`` to every checkpoint name that ``mappings`` gives."""
    out = {}
    for name, _ in model.named_parameters():
        src = mappings.get(name, name)
        out[name] = [prefix + k for k in src] if isinstance(src, list) else prefix + src
    return out


class _TextOnlyView(nn.Module):
    """Show only the text parameters of a multimodal wrapper to the loader."""

    def __init__(self, model: nn.Module, skip_prefixes: tuple[str, ...]):
        super().__init__()
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_skip_prefixes", skip_prefixes)

    def named_parameters(self, *args, **kwargs):
        return (
            (n, p)
            for n, p in self._model.named_parameters(*args, **kwargs)
            if not n.startswith(self._skip_prefixes)
        )

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_model"), name)


@async_speculative_decoding
class LlamaForCausalLM(nn.Module, SupportsEagle3):
    """Llama model with language modeling head.

    >>> PARALLELISM: Column-parallel linear for LM head <<<

    <-- MODEL-SPECIFIC: Llama-3.2-1B uses tied embeddings (lm_head shares
    weights with the embedding layer).
    """

    # A multimodal subclass sets these. checkpoint_prefix is the checkpoint
    # prefix of the text weights. load_weights skips parameters that start with
    # a name in own_weight_prefixes, because the subclass loads them itself.
    checkpoint_prefix: str = ""
    own_weight_prefixes: tuple[str, ...] = ()

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)

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

        lm_head_tp_rank = lm_head_tp_group.rank_in_group
        self.lm_head_tp_rank = lm_head_tp_rank

        # <-- MODEL-SPECIFIC: Tied embeddings — lm_head shares weight with embedding
        if config.tie_word_embeddings:
            # >>> PARALLELISM: Column-parallel LM head, weight tied to embedding <<<
            self.lm_head = neuron_nn.ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                dtype=config.torch_dtype,
                gather_output=not self.on_device_sampling_config,
                tp_group=lm_head_device_group,
            )
            # Weight will be tied after loading
        else:
            # >>> PARALLELISM: Column-parallel LM head <<<
            self.lm_head = neuron_nn.ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                dtype=config.torch_dtype,
                gather_output=not self.on_device_sampling_config,
                tp_group=lm_head_device_group,
            )

        lm_head_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.lm_head.out_features_per_rank,
            num_shards=self.lm_head.tp_size,
            is_storage_transposed=False,
        )
        lm_head_loader = with_rank_override(lm_head_loader, rank=lm_head_tp_rank)
        set_weight_loader(self.lm_head.weight, lm_head_loader)

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=lm_head_device_group,
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
        encoder_embedding_buffer: torch.Tensor | None = None,
        encoder_block_ids: torch.Tensor | None = None,
        encoder_positions: torch.Tensor | None = None,
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
            encoder_embedding_buffer=encoder_embedding_buffer,
            encoder_block_ids=encoder_block_ids,
            encoder_positions=encoder_positions,
        )

        # >>> PARALLELISM: Slice to local before sampling position selection <<<
        # sampling_positions are local indices; hidden_states may be gathered at mlp_dp
        if self.mlp_dp_group.world_size > 1:
            hidden_states = shard_select(hidden_states, 0, self.mlp_dp_group)

        hidden_states_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )

        # >>> PARALLELISM: Transition local → lm_head_dp <<<
        if self.lm_head_dp_size > 1:
            hidden_states_for_logits = self.lm_head_dp_group.all_gather(
                hidden_states_for_logits, dim=0
            )
            if self.on_device_sampling_config is not None:
                # ODS samples on the gathered lm_head_dp row layout. Keep
                # per-row inputs aligned with logits until sampling finishes.
                if logit_mask is not None:
                    if logit_mask.shape[0] == sampling_positions.shape[0]:
                        logit_mask = self.lm_head_dp_group.all_gather(logit_mask, dim=0)
                    elif logit_mask.shape[0] != hidden_states_for_logits.shape[0]:
                        raise ValueError(
                            "logit_mask rows must match either local or "
                            "lm_head_dp-gathered sampling rows. "
                            f"Got logit_mask={logit_mask.shape}, "
                            f"local_rows={sampling_positions.shape[0]}, "
                            f"gathered_rows={hidden_states_for_logits.shape[0]}."
                        )
                if sampling_params is not None:
                    sampling_params = self.lm_head_dp_group.all_gather(
                        sampling_params, dim=0
                    )

        logits = self.lm_head(hidden_states_for_logits)

        # >>> PARALLELISM: Gather sharded logits for logprobs computation <<<
        # Only gather when lm_head returns sharded output (ODS mode).
        # When gather_output=True (non-ODS), logits are already full vocab.
        gathered_logits = None
        if self._gather_logits:
            if self.lm_head.gather_output:
                gathered_logits = logits
            else:
                gathered_logits = self.lm_head_tp_group.all_gather(logits, dim=1)

        # >>> PARALLELISM: Slice back to local batch <<<
        if self.lm_head_dp_size > 1:
            B_local = sampling_positions.shape[0]
            dp_offset = (
                self.lm_head_dp_group.rank_in_group_tensor.to(torch.int64) * B_local
            )
            if self.on_device_sampling_config is None:
                idx = (
                    torch.arange(B_local, device=logits.device, dtype=torch.int64)
                    + dp_offset
                )
                logits = logits.index_select(0, idx)
            if gathered_logits is not None:
                idx = (
                    torch.arange(
                        B_local, device=gathered_logits.device, dtype=torch.int64
                    )
                    + dp_offset
                )
                gathered_logits = gathered_logits.index_select(0, idx)

        # ── No on-device sampling: return logits directly ────────────────
        if self.on_device_sampling_config is None:
            if len(aux_hidden_states) > 0:
                # Eagle3: concatenate aux hidden states on-device so the
                # target NEFF emits a single tensor instead of a list. This
                # keeps the cat inside the compile boundary and avoids a
                # host-side barrier between target NEFF output and draft
                # NEFF dispatch.
                aux_hidden_states_concat = torch.cat(aux_hidden_states, dim=-1)
                return logits, aux_hidden_states_concat
            return logits

        # ── On-device sampling ───────────────────────────────────────────
        lm_head_rank = rank
        if rank is not None and self.lm_head_dp_size > 1:
            lm_head_rank = self.lm_head_tp_group.rank_in_group_tensor.to(rank.dtype)
        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=lm_head_rank
        )
        if self.lm_head_dp_size > 1:
            # Slice sampled tokens after sampler; slicing logits earlier would
            # desynchronize SO masks from lm_head_dp-gathered logits.
            # spec_decode_metadata is local to this worker's input batch, so
            # rejection_sampler below also expects the local lm_head_dp row slice.
            B_local = sampling_positions.shape[0]
            dp_offset = (
                self.lm_head_dp_group.rank_in_group_tensor.to(torch.int64) * B_local
            )
            idx = (
                torch.arange(B_local, device=sampled_tokens.device, dtype=torch.int64)
                + dp_offset
            )
            sampled_tokens = sampled_tokens.index_select(0, idx)

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

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        if layers is not None:
            self.model.aux_hidden_state_layers = list(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        if self.model.aux_hidden_state_layers:
            return tuple(self.model.aux_hidden_state_layers)
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = LlamaConfig.from_configs(hf_config, neuron_config)
        return cls(config)

    # ── KV Cache Management ──────────────────────────────────────────────
    # >>> PARALLELISM: KV spec uses per-rank head counts (TP-sharded) <<<

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
                raise Exception(f"KV cache for layer {layer_name} not initialized")
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]
            # Detect the swizzled packed FP8 K layout from the bound cache rank:
            # packed K is [num_blocks, kv_heads, block_size // 2, head_dim, 2],
            # one dim higher than V (which is never packed). The bf16, static-FP8,
            # and Eagle3 attention all declare `fp8_packed` and consume the 5-D
            # cache in their prefill/decode paths; a bf16 K cache stays 4-D, so
            # the value is False there.
            k_is_packed = (
                layer.self_attn.k_cache.dim() == layer.self_attn.v_cache.dim() + 1
            )
            if hasattr(layer.self_attn, "fp8_packed"):
                layer.self_attn.fp8_packed = k_is_packed
            elif k_is_packed:
                # The runner allocated a packed (5-D) K cache (fp8_packed_kv +
                # fp8 KV dtype) but this attention path is not packed-aware.
                # Fail loudly rather than silently reading the swizzled 5-D
                # cache as if it were a standard 4-D one.
                raise NotImplementedError(
                    f"{layer_name}: a packed FP8 K cache was allocated "
                    "(fp8_packed_kv=True) but this attention path does not "
                    "support packed FP8 KV. Unset fp8_packed_kv."
                )

    # ── Weight Loading ───────────────────────────────────────────────────

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load weights from checkpoint.

        >>> PARALLELISM: Weight loaders handle TP sharding <<<
        <-- MODEL-SPECIFIC: HF checkpoint key → model parameter mappings

        Quantized variants (``config.quant_spec is not None``) register
        extra scale buffers on their attention / MLP modules. Those
        buffers are:

          * Non-persistent, so :func:`torch.nn.Module.load_state_dict`
            would drop them as unexpected keys. We therefore pop the
            scale entries out of the pipelined load result and assign
            them via :func:`setattr` on the owning sub-module (same
            pattern as :meth:`_load_kv_cache_scales`).
          * ``float32``; their loaders never emit the compute dtype, so
            the blanket bf16 cast below must skip them. We gate the cast
            on the destination parameter's dtype rather than the source
            tensor's dtype so fp8 weights and fp32 scales both stay put.
        """
        tp_rank = self.rank
        tp_size = self.world_size
        quant_spec = self.config.quant_spec

        mappings: dict[str, str | list[str]] = {}
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

            # <-- MX FP8 dual-buffer (trn3): CTE (MX prefill) + TKG
            # (STATIC decode) attention buffers. The pipelined loader
            # silently skips any that don't exist on this model variant.
            for suffix in ("_cte", "_tkg"):
                mappings[f"{prefix}.self_attn.qkv_proj_weight{suffix}"] = [
                    f"{prefix}.self_attn.q_proj.weight",
                    f"{prefix}.self_attn.k_proj.weight",
                    f"{prefix}.self_attn.v_proj.weight",
                ]
                mappings[f"{prefix}.self_attn.o_proj_weight{suffix}"] = (
                    f"{prefix}.self_attn.o_proj.weight"
                )

            # <-- MODEL-SPECIFIC: Llama norm names
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

            # <-- MX FP8 dual-buffer (trn3): CTE (MX swizzled) + TKG
            # (plain fp8 STATIC decode) MLP buffers.
            for suffix in ("_cte", "_plain", "_tkg"):
                mappings[f"{prefix}.mlp.gate_proj_weight{suffix}"] = (
                    f"{prefix}.mlp.gate_proj.weight"
                )
                mappings[f"{prefix}.mlp.up_proj_weight{suffix}"] = (
                    f"{prefix}.mlp.up_proj.weight"
                )
                mappings[f"{prefix}.mlp.down_proj_weight{suffix}"] = (
                    f"{prefix}.mlp.down_proj.weight"
                )

            # <-- QUANTIZATION: static-FP8 scale buffers are NOT added to
            # ``mappings`` on purpose. Their attached loaders live on
            # non-persistent buffers, which :meth:`load_sharded_pipelined`
            # skips (it iterates ``named_parameters`` only). We load
            # them ourselves after the pipelined load finishes, via
            # :meth:`_load_static_fp8_scales` below, which calls each
            # buffer's own loader and :func:`setattr`s the result.
            # Tracking these source keys here (rather than hard-coding
            # them in the scale-load helper) keeps the checkpoint-key
            # convention co-located with the bf16 weight mappings.

        # <-- MODEL-SPECIFIC: Tied embeddings — load lm_head from embed_tokens checkpoint key
        if self.config.tie_word_embeddings:
            mappings["lm_head.weight"] = "model.embed_tokens.weight"

        # A multimodal subclass keeps the text weights under a prefix. It loads
        # its own encoder weights separately.
        target = self
        if self.own_weight_prefixes:
            target = _TextOnlyView(self, self.own_weight_prefixes)
        if self.checkpoint_prefix:
            mappings = _prefix_mappings(target, mappings, self.checkpoint_prefix)

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, target, mappings, device
        ).state_dict

        # <-- QUANTIZATION: load static-FP8 scale buffers from the
        # checkpoint directly (``load_sharded_pipelined`` only iterates
        # ``named_parameters``; our scale buffers are declared
        # ``persistent=False`` and sit in ``named_buffers``). Uses each
        # buffer's attached :class:`SafetensorsWeightLoader` so the
        # ``(128, 1)`` / ``(128, 3)`` broadcasting and the ``448/240``
        # weight-scale compensation stay co-located with the FP8
        # modeling code.
        if quant_spec is not None:
            self._load_static_fp8_scales(checkpoint, tp_rank, device)

        # Convert to model dtype if needed. Guard against over-casting:
        # quantized weights (fp8) and scale buffers (fp32) must be kept
        # at their loader dtype. Gate on the destination parameter /
        # buffer's dtype instead of the source so the bf16 safety-net
        # still fires on legacy checkpoints that load as fp16 or fp32.
        self._cast_to_model_dtype(rank_sharded)

        self._load_kv_cache_scales(checkpoint, device)

        self.load_state_dict(rank_sharded, strict=False, assign=True)

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """
        Lightweight weight loading used during CPU compile.

        Only loads KV cache scales to ensure scale-derived constants
        k_scale_float/v_scale_float are correctly embedded in graphs.
        """
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        # Open all safetensor files to populate the tensor index
        # without loading any weight data
        checkpoint._ensure_indexed()
        self._load_kv_cache_scales(checkpoint, device)

    # ------------------------------------------------------------------
    # Helpers for load_weights
    # ------------------------------------------------------------------
    # Per-layer static-FP8 scale buffers and their checkpoint sources.
    # The value is a list of checkpoint keys (relative to
    # ``model.layers.{i}``) passed to the buffer's attached
    # :class:`SafetensorsWeightLoader`; the loader knows whether it
    # wants one slice (scalar → ``[128, 1]``) or three (fused QKV →
    # ``[128, 3]`` / QKV input-scale consensus check).
    #
    # Keeping this mapping local to the bf16 model code rather than in
    # :mod:`model_static_fp8` / :mod:`model_mx_fp8` avoids an import dependency from the
    # quantized module back into the generic orchestrator.
    _FP8_ATTN_SCALE_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "qkv_weight_scale",
            (
                "self_attn.q_proj.weight_scale",
                "self_attn.k_proj.weight_scale",
                "self_attn.v_proj.weight_scale",
            ),
        ),
        (
            "qkv_input_scale",
            (
                "self_attn.q_proj.input_scale",
                "self_attn.k_proj.input_scale",
                "self_attn.v_proj.input_scale",
            ),
        ),
        ("o_weight_scale", ("self_attn.o_proj.weight_scale",)),
        ("o_input_scale", ("self_attn.o_proj.input_scale",)),
        # Dual-buffer *_tkg scales (MX prefill + STATIC decode).
        # Only present when _use_mx_prefill is on; skipped otherwise.
        (
            "qkv_weight_scale_tkg",
            (
                "self_attn.q_proj.weight_scale",
                "self_attn.k_proj.weight_scale",
                "self_attn.v_proj.weight_scale",
            ),
        ),
        (
            "qkv_input_scale_tkg",
            (
                "self_attn.q_proj.input_scale",
                "self_attn.k_proj.input_scale",
                "self_attn.v_proj.input_scale",
            ),
        ),
        ("o_weight_scale_tkg", ("self_attn.o_proj.weight_scale",)),
        ("o_input_scale_tkg", ("self_attn.o_proj.input_scale",)),
    )
    _FP8_MLP_SCALE_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("gate_weight_scale", ("mlp.gate_proj.weight_scale",)),
        ("up_weight_scale", ("mlp.up_proj.weight_scale",)),
        ("down_weight_scale", ("mlp.down_proj.weight_scale",)),
        ("gate_up_input_scale", ("mlp.gate_proj.input_scale",)),
        ("down_input_scale", ("mlp.down_proj.input_scale",)),
        # Dual-buffer ``*_tkg`` MLP scales (only present when MX is on).
        ("gate_weight_scale_tkg", ("mlp.gate_proj.weight_scale",)),
        ("up_weight_scale_tkg", ("mlp.up_proj.weight_scale",)),
        ("down_weight_scale_tkg", ("mlp.down_proj.weight_scale",)),
        ("gate_up_input_scale_tkg", ("mlp.gate_proj.input_scale",)),
        ("down_input_scale_tkg", ("mlp.down_proj.input_scale",)),
    )

    def _load_static_fp8_scales(
        self,
        checkpoint: SafetensorsCheckpoint,
        tp_rank: int,
        device: torch.device,
    ) -> None:
        """Load and assign each static-FP8 scale buffer on every layer.

        Calls the :class:`SafetensorsWeightLoader` attached to each
        scale buffer at FP8-module construction time, then :func:`setattr`
        the resulting tensor onto the owning sub-module. This replaces
        the meta-device allocation created under the model-runner's
        ``with torch.device("meta")`` context — without this call the
        buffers stay as meta tensors and later ``self.model.to(device)``
        fails with "Cannot copy out of meta tensor".
        """
        from vllm_neuron.utils.weight_loader import get_weight_loader

        for layer_id, layer in enumerate(self.model.layers):
            prefix = f"model.layers.{layer_id}"
            self._load_scale_group(
                owner=layer.self_attn,
                sources=self._FP8_ATTN_SCALE_SOURCES,
                checkpoint=checkpoint,
                layer_prefix=prefix,
                tp_rank=tp_rank,
                device=device,
                get_loader=get_weight_loader,
            )
            self._load_scale_group(
                owner=layer.mlp,
                sources=self._FP8_MLP_SCALE_SOURCES,
                checkpoint=checkpoint,
                layer_prefix=prefix,
                tp_rank=tp_rank,
                device=device,
                get_loader=get_weight_loader,
            )

    @staticmethod
    def _load_scale_group(
        *,
        owner: nn.Module,
        sources: tuple[tuple[str, tuple[str, ...]], ...],
        checkpoint: "SafetensorsCheckpoint",
        layer_prefix: str,
        tp_rank: int,
        device: torch.device,
        get_loader,
    ) -> None:
        """Load one scale-buffer group (attention or MLP) on ``owner``.

        Raises a clear :class:`KeyError` when a required checkpoint key
        is absent so unexpected checkpoint formats fail at load time instead
        of producing silently-wrong matmuls.
        """
        for buffer_name, key_suffixes in sources:
            # Skip optional buffers that this model variant didn't
            # register (e.g. ``*_tkg`` scales when ``_use_mx`` is off).
            if not hasattr(owner, buffer_name):
                continue
            buf = getattr(owner, buffer_name)
            loader = get_loader(buf)
            slices = []
            for suffix in key_suffixes:
                full_key = f"{layer_prefix}.{suffix}"
                if full_key not in checkpoint._tensor_name_to_file:
                    raise KeyError(
                        f"Missing static-FP8 scale in checkpoint: {full_key!r}. "
                        "ModelOpt FP8 checkpoints must include per-projection "
                        "weight_scale and input_scale entries."
                    )
                slices.append(checkpoint._get_slice(full_key))
            tensor = loader.load(slices, tp_rank).to(device)
            setattr(owner, buffer_name, tensor)

    def _cast_to_model_dtype(self, rank_sharded: dict[str, torch.Tensor]) -> None:
        """Safety-net cast of tensors whose destination parameter is the
        model's compute dtype but whose loader returned a different dtype.

        Tensors with no matching destination parameter (typed tied weights,
        rare edge cases) fall back to ``self.config.torch_dtype`` comparison
        so pre-quantization behaviour is preserved for them.
        """
        target_dtype = self.config.torch_dtype
        for name, tensor in list(rank_sharded.items()):
            dest = self._lookup_parameter(name)
            if dest is None:
                # Legacy fallback (no destination param found): keep the
                # pre-existing behaviour of casting to compute dtype.
                if tensor.dtype != target_dtype:
                    rank_sharded[name] = tensor.to(target_dtype)
                continue
            if dest.dtype == target_dtype and tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

    def _lookup_parameter(self, dotted_name: str) -> torch.Tensor | None:
        """Return the Parameter/Buffer at ``dotted_name`` on ``self``, or
        ``None`` if any component of the path is missing."""
        obj: object = self
        for part in dotted_name.split("."):
            if not hasattr(obj, part):
                return None
            obj = getattr(obj, part)
        if isinstance(obj, torch.Tensor):
            return obj
        return None

    def _load_kv_cache_scales(
        self, checkpoint: SafetensorsCheckpoint, device: torch.device
    ):
        """Load KV cache quantization scales from checkpoint if provided."""

        from vllm_neuron.utils.dtype_utils import (
            QUANTIZED_KV_CACHE_DTYPES,
        )
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()

        for layer_id in range(len(self.model.layers)):
            attn = self.model.layers[layer_id].self_attn

            # Skip setting quantization scales if not using a quantized KV cache
            if vllm_config.cache_config.cache_dtype not in QUANTIZED_KV_CACHE_DTYPES:
                continue

            # Store quantization scale tensors. Default to 1.0 if scales are not in checkpoint
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

            # Store scale floats
            attn.k_scale_float = attn.k_scale.item()
            attn.v_scale_float = attn.v_scale.item()
