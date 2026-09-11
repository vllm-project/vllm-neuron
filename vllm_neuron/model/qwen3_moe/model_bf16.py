# SPDX-License-Identifier: Apache-2.0
"""
Qwen3 MoE BF16 Implementation
======================================

Qwen3 MoE model for the Neuron backend, ported from Qwen3 MoE reference.

Supported parallelism: TP, SP, DP, EP, and DP+EP combinations.
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
    NormType,
    RouterActFnType,
)

from nkilib.core.moe.moe_cte.moe_cte import (
    MoECTEImplementation,
)

import vllm_neuron.nn as neuron_nn
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.vllm.spec_decode.decorator import async_speculative_decoding

from .config import Qwen3MoeConfig

from .weight_loaders_bf16 import (
    fused_qkv_weight_loader,
    o_proj_weight_loader,
    expert_gate_up_weight_sharding_loader,
    expert_down_weight_sharding_loader,
)
from vllm_neuron.utils.weight_loader import (
    expert_parallel_grouped_loader,
)

# Threshold for decode MoE kernel selection
DEFAULT_SELECTIVE_LOADING_THRESHOLD = 1.0

logger = logging.getLogger(__name__)


def _packed_fp8_viable_for_bucket(
    block_len: int, bs: int, q_head: int, s_active: int, s_prior: int
) -> bool:
    """Whether the packed FP8 decode kernel is usable for this bucket geometry."""
    if block_len <= 0 or s_prior <= 0:
        return False
    return _resize_block_len(block_len, bs, q_head, s_active, s_prior) >= 2


# =============================================================================
# Section 1: RMS Normalization
# =============================================================================
class Qwen3MoeRMSNorm(nn.Module):
    """Standard RMSNorm — no padding logic needed."""

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=config.torch_dtype)
        )
        self.variance_epsilon = config.rms_norm_eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


# =============================================================================
# Section 2: Rotary Position Embedding (standard RoPE, optional YaRN scaling)
# =============================================================================
class Qwen3MoeRotaryEmbedding(nn.Module):
    """Rotary Position Embedding with rope_theta=1,000,000.

    Plain RoPE by default. When ``config.rope_scaling`` requests YaRN, the
    inverse frequencies are rebuilt with NTK-by-parts interpolation and an
    attention-temperature ("concentration") factor is applied to cos/sin.

    Qwen3-30B-A3B is trained for 32,768 tokens and extends to 131,072 with:
        {"rope_type": "yarn", "factor": 4.0,
         "original_max_position_embeddings": 32768}

    The YaRN maths here is ported verbatim from GptOssRotaryEmbedding in this
    same package, which is where the rest of this model was ported from, so the
    two share one implementation and one set of numerics.

    Note that this is STATIC YaRN: the scaling factor is fixed at load time and
    applied to every request regardless of its length, which is how every
    open-source framework implements it. Qwen therefore advises against enabling
    it unless long contexts are actually needed, and suggests matching ``factor``
    to the real workload (2.0 for 65,536 rather than 4.0 for 131,072).
    """

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta

        rope_scaling = getattr(config, "rope_scaling", None) or {}
        rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type")
        self.use_yarn = rope_type == "yarn"

        if self.use_yarn:
            self.scaling_factor = float(rope_scaling.get("factor", 1.0))
            # Absent from Qwen's published config; YaRN paper defaults.
            self.beta_slow = float(rope_scaling.get("beta_slow", 1.0))
            self.beta_fast = float(rope_scaling.get("beta_fast", 32.0))
            self.initial_context_length = int(
                rope_scaling.get("original_max_position_embeddings", 32768)
            )
            inv_freq, concentration = self._compute_yarn_inv_freq_and_concentration(
                "cpu"
            )
            self.register_buffer("concentration", concentration, persistent=False)
            logger.info(
                "Qwen3MoE YaRN RoPE enabled: factor=%s, original_max_position_"
                "embeddings=%s, beta_fast=%s, beta_slow=%s -> effective context %s",
                self.scaling_factor,
                self.initial_context_length,
                self.beta_fast,
                self.beta_slow,
                int(self.initial_context_length * self.scaling_factor),
            )
        else:
            inv_freq = 1.0 / (
                self.rope_theta
                ** (
                    torch.arange(0, self.head_dim, 2, dtype=torch.float, device="cpu")
                    / self.head_dim
                )
            )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_yarn_inv_freq_and_concentration(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """YaRN (NTK-by-parts) frequency computation.

        Low-frequency dimensions, which encode long-range position, are
        interpolated by ``1 / factor``. High-frequency dimensions, which encode
        local ordering, are left as-is (extrapolated). ``ramp``/``mask`` blend
        the two across the dimensions that fall between the beta_fast and
        beta_slow wavelength cutoffs.
        """
        freq = self.rope_theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
            / self.head_dim
        )

        # Attention temperature: compensates for the entropy change caused by
        # compressing positions. Applied to cos/sin rather than to the logits.
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
        return inv_freq, torch.tensor(concentration, device=device)

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :].float()  # [1, head_dim/2]
        position_ids_expanded = position_ids[:, None].float()  # [T, 1]

        freqs = position_ids_expanded @ inv_freq_expanded  # [T, head_dim/2]
        cos = freqs.cos()
        sin = freqs.sin()
        if self.use_yarn:
            cos = cos * self.concentration
            sin = sin * self.concentration

        return cos.to(dtype=dtype), sin.to(dtype=dtype)


# =============================================================================
# Section 3: Attention
# Mixed: PARALLELISM (TP head sharding, SP, collectives) +
#        MODEL-SPECIFIC (sinks, sliding window, GQA, RoPE application)
# =============================================================================
# NOTE: RoPE is fused into NF.qkv_proj for prefill; decode handles RoPE in
# its own dedicated kernel call. The standalone `_apply_rotary_emb` /
# `apply_rotary_pos_emb` helpers are no longer used by either path.


class Qwen3MoeAttention(nn.Module):
    """Multi-head attention with TP head sharding.

    No sinks, no sliding window, no attention bias.
    """

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.rms_norm_eps = config.rms_norm_eps
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5
        self.max_seq_len = config.max_position_embeddings

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
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_proj_in_features, self.hidden_size, dtype=self.dtype)
        )

        # QK-norm: RMSNorm on head_dim applied pre-RoPE
        self.q_norm_weight = nn.Parameter(
            torch.empty(self.head_dim, dtype=torch.float32)
        )
        self.k_norm_weight = nn.Parameter(
            torch.empty(self.head_dim, dtype=torch.float32)
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
        self.k_scale = torch.ones(1, 1, dtype=torch.bfloat16)
        self.v_scale = torch.ones(1, 1, dtype=torch.bfloat16)
        self.k_scale_float = 1.0
        self.v_scale_float = 1.0

        # Set up weight loaders for checkpoint loading
        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach weight loaders for checkpoint → parameter transformation."""
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

        o_loader = o_proj_weight_loader(
            shard_size=(self.num_attention_heads * self.head_dim) // effective_q_shards,
            num_shards=effective_q_shards,
            hidden_size=self.hidden_size,
        )
        o_loader = with_rank_override(o_loader, rank=effective_q_rank)
        set_weight_loader(self.o_proj_weight, o_loader)

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
        # Qwen3MoeRotaryEmbedding emits cos/sin of shape [T, d_head/2]; the
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
                    qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
                    qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
                    qk_norm_pre_rope_eps=self.rms_norm_eps,
                    qk_norm_pre_rope_q_gamma=self.q_norm_weight.unsqueeze(0),
                    qk_norm_pre_rope_k_gamma=self.k_norm_weight.unsqueeze(0),
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
                    d_head=self.head_dim,
                    cos_cache=cos_cache,
                    sin_cache=sin_cache,
                    num_q_heads=self.num_attention_heads_per_rank,
                    num_kv_heads=self.num_kv_heads_for_weight,
                    qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
                    qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
                    qk_norm_pre_rope_eps=self.rms_norm_eps,
                    qk_norm_pre_rope_q_gamma=self.q_norm_weight.unsqueeze(0),
                    qk_norm_pre_rope_k_gamma=self.k_norm_weight.unsqueeze(0),
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
                d_head=self.head_dim,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                num_q_heads=self.num_attention_heads_per_rank,
                num_kv_heads=self.num_kv_heads_for_weight,
                qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
                qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
                qk_norm_pre_rope_eps=self.rms_norm_eps,
                qk_norm_pre_rope_q_gamma=self.q_norm_weight,
                qk_norm_pre_rope_k_gamma=self.k_norm_weight,
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
                tp_q=False,
                tp_out=True,
            )  # [Nh, Dh, T]

        # ── Step 5: Output Projection ────────────────────────────────────
        attn_output = attn_output.unsqueeze(0)  # [1, Nh, Dh, T]
        attn_output = NF.o_proj(attn_output, self.o_proj_weight)  # [1, T, H]
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

        <-- MODEL-SPECIFIC: Qwen3 MoE specific kernel arguments:
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

        pos_ids_kernel = None
        swa_start_pos_ids_kernel = None

        if attn_mask is None:
            pos_ids_kernel = pos_ids.view(B_local, S_decode).to(torch.float32)

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
            rmsnorm_X_enabled=False,  # RMSNorm applied by decoder layer
            W_qkv=self.qkv_proj_weight,
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=self.rms_norm_eps,
            rmsnorm_QK_pre_rope_W_Q=self.q_norm_weight,
            rmsnorm_QK_pre_rope_W_K=self.k_norm_weight,
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
            update_cache=True,
            kv_cache_update_idx=slot_mapping.view(B_local, S_decode).to(torch.uint32),
            fp8_packed=use_packed_kernel,
            # Kernel currently requires fusing the V scale dequantization into W_out for KV quantization
            W_out=self.o_proj_weight / self.v_scale_float,
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
class Qwen3MoeExperts(nn.Module):
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

    def __init__(self, config: Qwen3MoeConfig):
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

        self.total_num_experts = config.num_experts

        # Linear placement: EP rank k owns experts [k*L .. (k+1)*L)
        self.num_local_experts = config.num_experts // self.ep_degree
        self.num_experts_per_token = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.rms_norm_eps = config.rms_norm_eps

        # >>> PARALLELISM: Intermediate dim sharded across TP degree <<<
        self.intermediate_size_per_rank = self.intermediate_size // self.tp_degree
        self.block_size = 256

        # Pre-MLP RMSNorm
        self.post_attention_layernorm = Qwen3MoeRMSNorm(config)

        # >>> PARALLELISM: Router weights replicated on all ranks (NOT EP-sharded) <<<
        self.router_weight = nn.Parameter(
            torch.empty(self.total_num_experts, self.hidden_size, dtype=torch.bfloat16)
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
        # down_proj: [E, I_per_rank, H]
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.intermediate_size_per_rank,
                self.hidden_size,
                dtype=config.torch_dtype,
            )
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Set up weight loaders for TP + EP sharding."""
        # Linear EP placement: rank k owns experts [k*L, (k+1)*L)
        local_expert_indices = list(
            range(
                self.ep_rank * self.num_local_experts,
                (self.ep_rank + 1) * self.num_local_experts,
            )
        )

        def _maybe_ep_wrap(loader):
            if self.ep_degree > 1:
                # Qwen3-MoE checkpoints store experts as SEPARATE per-expert
                # tensors, so the loader receives a flat list grouped by item:
                # [gate_0..gate_{E-1}, up_0..up_{E-1}] (2 groups) for gate_up and
                # [down_0..down_{E-1}] (1 group) for down_proj. The correct EP
                # wrapper selects the local expert range *within each group*.
                #
                # Do NOT use expert_parallel_tensor_dim_loader here: that one is
                # for GPT-OSS's FUSED [E, ...] layout and applies a SliceView on
                # expert_dim=0 inside each tensor. Against Qwen3's per-expert
                # tensors (each [I, H]) it slices the intermediate dimension
                # instead of experts, leaves the list length at 2*total_experts,
                # and fails as "Expected 64 slices (gate+up), got 256".
                loader = expert_parallel_grouped_loader(
                    local_expert_indices, loader, self.total_num_experts
                )
            if self.mlp_dp_size > 1:
                loader = with_rank_override(loader, rank=self.mlp_tp_rank)
            return loader

        set_weight_loader(
            self.gate_up_proj_weight,
            _maybe_ep_wrap(
                expert_gate_up_weight_sharding_loader(
                    shard_size=self.intermediate_size_per_rank * 2,
                    num_shards=self.tp_degree,
                    hidden_size=self.hidden_size,
                    num_experts=self.num_local_experts,
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
                    num_experts=self.num_local_experts,
                )
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
            eps=self.rms_norm_eps,
            router_act_fn=RouterActFnType.SOFTMAX,
            router_pre_norm=False,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            hidden_act_fn=ActFnType.SiLU,
            router_mm_dtype=nl.bfloat16,
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
            activation_function=ActFnType.SiLU,
            block_size=self.block_size,
            token_position_to_id=token_position_to_id.to(dtype=torch.int32),
            block_to_expert=block_to_expert.to(dtype=torch.int32),
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
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
class Qwen3MoeMLP(nn.Module):
    """MLP layer with Mixture of Experts.

    <-- MODEL-SPECIFIC: This wrapper exists because Qwen3 MoE uses MoE.
    A dense model would have a simple MLP here instead.
    """

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.experts = Qwen3MoeExperts(config)
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


def _precompute_decode_attn_mask(
    attn_metadata: dict,
    positions: torch.Tensor,
    num_q_heads_after_a2a: int,
) -> torch.Tensor:
    """Pre-compute the shared decode causal attention mask (no sliding window)."""
    first_layer_name = "layers.0.self_attn"
    block_size = attn_metadata[first_layer_name]["block_size"]
    block_table = attn_metadata[first_layer_name]["block_table_tensor"]
    B_local = block_table.shape[0]
    S_decode = positions.shape[0] // B_local
    pos_ids_flat = positions.reshape(1, B_local * S_decode)

    max_blocks = attn_metadata[first_layer_name]["max_blocks_per_seq"]
    S_ctx = max_blocks * block_size

    mask = NF.gen_attention_decode_mask(
        pos_ids=pos_ids_flat.to(torch.float32),
        bs=B_local,
        q_head=num_q_heads_after_a2a,
        s_active=S_decode,
        s_prior=S_ctx,
        start_pos=None,
        block_len=block_size,
    )

    return mask


class Qwen3MoeDecoderLayer(nn.Module):
    """Single transformer decoder layer.

    Architecture (MODEL-SPECIFIC):
        hidden_states → RMSNorm → Attention → residual → MoE → residual

    Parallelism (TP + SP):
        - Input arrives in SP layout (T/world_size tokens per rank) during prefill
        - Decode: _dp_transition handles batch state between modules
        - Residual connection operates at whatever gathered state the module outputs
    """

    def __init__(self, config: Qwen3MoeConfig, batch_size: int, layer_idx: int):
        super().__init__()
        # <-- MODEL-SPECIFIC: Pre-attention RMSNorm
        self.input_layernorm = Qwen3MoeRMSNorm(config)
        self.self_attn = Qwen3MoeAttention(config, layer_idx=layer_idx)
        self.mlp = Qwen3MoeMLP(config)
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
class Qwen3MoeModel(nn.Module):
    """Qwen3 MoE transformer backbone.

    >>> PARALLELISM: SP (Sequence Parallelism) <<<
    During prefill:
    - After embedding: chunk sequence across TP ranks (each gets T/world_size)
    - After all layers: all-gather to reconstruct full sequence
    During decode:
    - No SP; all ranks process all tokens
    """

    def __init__(self, config: Qwen3MoeConfig, batch_size: int):
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
                Qwen3MoeDecoderLayer(config, batch_size, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # <-- MODEL-SPECIFIC: Final RMSNorm
        self.norm = Qwen3MoeRMSNorm(config)
        self.rotary_emb = Qwen3MoeRotaryEmbedding(config)

        from vllm_neuron.utils.weight_loader import (
            sharding_weight_loader_with_padding,
        )

        emb_loader = sharding_weight_loader_with_padding(
            shard_dim=0,
            shard_size=self.embed_tokens.vocab_size_per_rank,
            num_shards=self.embed_tokens.tp_size,
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

        # <-- MODEL-SPECIFIC: Qwen3 MoE accepts unpadded prompt embeds and pads
        # them to the runtime hidden dim used by this model variant.
        if inputs_embeds is not None:
            target_dim = hidden_states.shape[-1]
            current_dim = inputs_embeds.shape[-1]
            if current_dim > target_dim:
                raise ValueError(
                    f"inputs_embeds dim ({current_dim}) exceeds hidden dim "
                    f"({target_dim}) for Qwen3 MoE prompt embeddings."
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

        # Pre-compute decode causal attention mask (same for all layers)
        attn_mask = None
        if not is_prefill:
            attn_mask = _precompute_decode_attn_mask(
                attn_metadata=attn_metadata,
                positions=positions,
                num_q_heads_after_a2a=self.layers[0].self_attn.num_q_heads_after_a2a,
            )

        # Run through decoder layers, collecting Eagle3 auxiliary hidden states
        aux_hidden_states = []
        for idx, decoder_layer in enumerate(self.layers):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states)
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
class Qwen3MoeForCausalLM(nn.Module, SupportsEagle3):
    """Qwen3 MoE model with language modeling head.

    >>> PARALLELISM: Column-parallel linear for LM head <<<
    The vocabulary projection is sharded across TP ranks. Each rank computes
    a portion of the logits, then either:
    - Gathered for full logits (when not using on-device sampling)
    - Kept sharded for on-device sampling (sampler handles TP internally)
    """

    def __init__(self, config: Qwen3MoeConfig, batch_size: int):
        super().__init__()
        self.config = config
        self.model = Qwen3MoeModel(config, batch_size)

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
        # used by Qwen3 MoE DI), the argmax path ignores those per-row params
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
        config = Qwen3MoeConfig.from_configs(hf_config, neuron_config)
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
                    sliding_window_size=None,
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

            # Attention weight mappings (no bias)
            mappings[f"{layer_prefix}.self_attn.qkv_proj_weight"] = [
                f"{layer_prefix}.self_attn.q_proj.weight",
                f"{layer_prefix}.self_attn.k_proj.weight",
                f"{layer_prefix}.self_attn.v_proj.weight",
            ]
            mappings[f"{layer_prefix}.input_layernorm.weight"] = (
                f"{layer_prefix}.input_layernorm.weight"
            )
            mappings[f"{layer_prefix}.self_attn.o_proj_weight"] = (
                f"{layer_prefix}.self_attn.o_proj.weight"
            )
            mappings[f"{layer_prefix}.self_attn.q_norm_weight"] = (
                f"{layer_prefix}.self_attn.q_norm.weight"
            )
            mappings[f"{layer_prefix}.self_attn.k_norm_weight"] = (
                f"{layer_prefix}.self_attn.k_norm.weight"
            )

            # MoE weight mappings
            mappings[f"{layer_prefix}.mlp.experts.post_attention_layernorm.weight"] = (
                f"{layer_prefix}.post_attention_layernorm.weight"
            )
            mappings[f"{layer_prefix}.mlp.experts.router_weight"] = (
                f"{layer_prefix}.mlp.gate.weight"
            )
            # Expert weights: provide per-expert keys as list for the weight loader
            num_experts = self.config.num_experts
            mappings[f"{layer_prefix}.mlp.experts.gate_up_proj_weight"] = [
                f"{layer_prefix}.mlp.experts.{e}.gate_proj.weight"
                for e in range(num_experts)
            ] + [
                f"{layer_prefix}.mlp.experts.{e}.up_proj.weight"
                for e in range(num_experts)
            ]
            mappings[f"{layer_prefix}.mlp.experts.down_proj_weight"] = [
                f"{layer_prefix}.mlp.experts.{e}.down_proj.weight"
                for e in range(num_experts)
            ]

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
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
