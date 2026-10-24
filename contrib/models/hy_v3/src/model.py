# SPDX-License-Identifier: Apache-2.0
"""
Hy3 (HYV3) Static-FP8 Implementation
====================================

Implementation of tencent/Hy3-FP8 for the Neuron backend.

Checkpoint format: every linear layer (attention, dense MLP, shared expert and
each routed expert) stores ``float8_e4m3fn`` weights with a scalar BF16
``weight_scale`` and a scalar static ``input_scale``. ``lm_head`` and
``embed_tokens`` are BF16. The MTP (next-n) layer stored after the decoder
stack is not loaded.

Supported parallelism: TP (attention heads / dense MLP), SP (prefill), EP for
the routed experts (``--enable-expert-parallel``; full EP by default), and DCP
(``--decode-context-parallel-size``) to shard the KV cache across the ranks
that hold a replicated KV head.

Kernel mapping:
  - Attention prefill: ``NF.qkv_proj`` (STATIC) -> torch QK-norm + RoPE ->
    flash / segmented attention -> ``NF.o_proj`` (STATIC).
  - Attention decode: ``NF.attention_decode`` megakernel (STATIC QKV/O, fused
    input RMSNorm, pre-RoPE QK-norm, in-kernel KV cache update).
  - Dense MLP / shared expert: ``NF.rmsnorm_quant`` + ``NF.mlp`` (STATIC).
  - Routed experts: torch router (sigmoid + selection-only bias), then
    ``NF.moe_cte`` (``shard_on_i_hybrid``, per-channel FP8 weight dequant) for
    prefill and the ``moe_tkg`` kernel (STATIC FP8 weights) for decode. Both run
    BF16 activations x FP8 weights.

FP8 range handling: every FP8 weight is halved at load time to fit TRN2's
240-capped e4m3, with the scales compensated (see ``weight_loaders.py``).

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code.
  # <-- MODEL-SPECIFIC: ...    Hy3 specific.
  # <-- STATIC-FP8: ...        FP8 weight / scale handling.
"""

import logging

import nki.language as nl
import os

import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.mlp.mlp_parameters import TKG_BS_SEQLEN_THRESHOLD
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    NormType,
    QuantizationType,
)
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_dcp_group, get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.functional.attention.attention_decode import (
    _swizzle_packed_k,
    _unswizzle_packed_k,
    packed_fp8_viable_for_bucket,
)
from vllm_neuron.functional.moe.moe_tkg_wrapper import moe_tkg_wrapper
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec, paged_slot_indices
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.parallel.neuron_parallel_state import shard_select
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.dtype_utils import (
    FP8_CLAMP_MAX,
    QUANTIZED_KV_CACHE_DTYPES,
    validate_fp8_segmented_supported,
)
from vllm_neuron.utils.weight_loader import (
    expert_parallel_grouped_loader,
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from . import kv_scales
from .config import HYV3Config
from .kernels.moe_tkg_hits import moe_tkg_hits
from .kernels.moe_tkg_hits_batch import moe_tkg_hits_batch
from .weight_loaders import (
    FP8_DTYPE as _FP8_DTYPE,
)
from .weight_loaders import (
    PMAX as _PMAX,
)
from .weight_loaders import (
    expert_down_loader,
    expert_gate_up_loader,
    expert_scale_loader,
    padded_mlp_loader,
    scalar_scale_loader,
    with_fp8_halving,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Section 0: Constants and helpers
# =============================================================================

# CTE MLP kernel produces wrong results when the per-rank intermediate size is
# not a multiple of 128, so the dense MLP intermediate is zero-padded per rank.
_MLP_INTERMEDIATE_ALIGN = 128

# Blockwise MoE prefill block size (shard_on_i requires a multiple of 256).
_MOE_BLOCK_SIZE = 256

# Ranks per group sharing one copy of the shared expert during decode.
_SHARED_DECODE_SHARDS = 8

# The decode attention mask is identical for every layer; it is built once per
# step and cached in attn_metadata under this key.
_DECODE_MASK_CACHE_KEY = "_hy_v3_decode_mask"


def _validate_dcp_decode_sequence_length(dcp_size: int, sequence_length: int) -> None:
    if dcp_size > 1 and sequence_length != 1:
        raise ValueError(
            "Hy3 DCP decode supports exactly one token per request; "
            f"got {sequence_length}."
        )


def _fp32_param(*shape) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape, dtype=torch.float32), requires_grad=False)


# FP8 KV cache. Scales are per-layer Python floats from the KV scale profile
# (src/kv_scales.py), so they are graph constants: a different profile traces
# to different HLO and therefore different NEFFs (compile-cache isolation).
_FP8_KV_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

# Test hook: decode reads the packed K cache through the unpacked kernel
# (un-swizzle -> kernel -> re-swizzle the returned K) even when the packed
# kernel is viable, so the fallback write-back can be exercised on every bucket.
_FORCE_UNPACKED_DECODE = os.environ.get("HY3_FP8_KV_FORCE_UNPACKED_DECODE") == "1"


def _is_fp8_kv(cache: torch.Tensor | None) -> bool:
    return cache is not None and cache.dtype in _FP8_KV_DTYPES


def _quantize_kv(x: torch.Tensor, q, dtype: torch.dtype) -> torch.Tensor:
    """Cache write: ``fp8(clamp(x * q, +-FP8_CLAMP_MAX))`` (q = 1 / dequant)."""
    return (x.to(torch.float32) * q).clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(dtype)


def _write_packed_k(
    k_cache: torch.Tensor,
    k_rows: torch.Tensor,
    block_idx: torch.Tensor,
    head_idx: torch.Tensor,
    pos_idx: torch.Tensor,
) -> None:
    """Write pair-aligned K rows into the packed cache [blocks, kvh, block // 2, D, 2].

    Same pair coordinates and preconditions as vllm_neuron's ``scatter_packed_k``
    (rows (2j, 2j + 1) are lanes 0 / 1 of one packed slot; null-block rows are
    padding), but without its gather of every target slot to preserve the null
    lane: that O(tokens) gather made the full-model prefill NEFFs 2-4x larger and
    their compile outlast the startup TP barrier. In Hy3 prefill the only
    half-valid slot is the last one of an odd-length chunk, whose null lane is
    the request's next position in its own (not yet full, so never shared)
    block; it is written before it is ever read. All-null slots go to the null
    block like the upstream helper.
    """
    num_rows, d_head = k_rows.shape
    k_pairs = k_rows.view(-1, 2, d_head).transpose(1, 2).contiguous()
    block_lanes = block_idx.view(-1, 2)
    head_lanes = head_idx.view(-1, 2)
    pos_lanes = pos_idx.view(-1, 2)
    lane0_valid = block_lanes[:, 0] > 0
    block_pair = torch.where(lane0_valid, block_lanes[:, 0], block_lanes[:, 1])
    head_pair = torch.where(lane0_valid, head_lanes[:, 0], head_lanes[:, 1])
    outer_pair = torch.where(lane0_valid, pos_lanes[:, 0], pos_lanes[:, 1]) // 2
    pair_valid = (block_lanes > 0).any(dim=1)
    outer_pair = torch.where(pair_valid, outer_pair, torch.zeros_like(outer_pair))
    k_cache.index_put_((block_pair, head_pair, outer_pair), k_pairs)


# Segmented prefill over an FP8 cache: "bf16" (default) reads a BF16 copy of
# the request's blocks (see _request_blocks_bf16) with the BF16 segmented
# kernel; "packed" reads the packed FP8 K cache in the NKI kernel. On the full
# model (80 layers, 8139 blocks, prefill buckets up to 8192) the packed read
# makes the prefill NEFFs 10-16x larger (48-116+ MB vs 3-7 MB) and their cold
# compile take hours, past the startup TP barrier. DCP prefill always uses the
# BF16 copy (see forward_prefill).
_PREFILL_READ = os.environ.get("HY3_FP8_KV_PREFILL_READ", "bf16")


def _request_blocks_bf16(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    packed: bool,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BF16 copy of one prefill request's KV blocks and the matching block table.

    Prefill runs one request per step, so its block table row lists every block
    the segmented kernel may read. Gathering those (at most max_model_len /
    block_size) blocks, un-swizzling a packed K and casting to BF16 is exact
    (FP8 codes are representable in BF16) and small next to the prefill; the
    kernel then reads a standard BF16 cache with a 0..n-1 block table.
    """
    blocks = block_table.reshape(-1).to(torch.long).clamp(min=0)
    k_sel = torch.index_select(k_cache, 0, blocks)
    if packed:
        k_sel = _unswizzle_packed_k(k_sel)
    v_sel = torch.index_select(v_cache, 0, blocks)
    local_table = torch.arange(
        blocks.shape[0], dtype=block_table.dtype, device=block_table.device
    ).view(block_table.shape)
    return k_sel.to(dtype), v_sel.to(dtype), local_table


def _scale_column(value: float, device: torch.device) -> torch.Tensor:
    """A [PMAX, 1] fp32 replicated scalar, the layout the attention kernels take."""
    return torch.full((_PMAX, 1), value, dtype=torch.float32, device=device)


# =============================================================================
# Section 1: RMSNorm + RoPE
# <-- MODEL-SPECIFIC: standard RMSNorm, default RoPE (rotate_half)
# =============================================================================


class HYV3RMSNorm(nn.Module):
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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class HYV3RotaryEmbedding(nn.Module):
    def __init__(self, config: HYV3Config):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
                / self.head_dim
            )
        )
        freqs = position_ids[:, None].float() * inv_freq[None, :]  # [T, Hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, Hd]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


# =============================================================================
# Section 2: Attention (static FP8, QK-norm before RoPE)
# =============================================================================


class HYV3Attention(nn.Module):
    """GQA attention with TP head sharding and static-FP8 projections.

    >>> PARALLELISM: TP <<<
    - Q heads sharded across TP; KV heads replicated when TP > num KV heads.
    - Prefill: RMSNorm (SP) -> all-gather -> attention -> reduce-scatter.
    - Decode: fused megakernel (input RMSNorm fused) -> all-reduce.

    >>> PARALLELISM: DCP (decode context parallelism) <<<
    With TP > num_kv_heads every KV head is replicated on TP / num_kv_heads
    ranks. DCP turns those replicas into a sequence-sharded group: each rank
    stores only the KV blocks it owns (interleaved by block_size), and
    attention combines the per-rank partial results with an LSE correction
    (``segmented_attention_cp`` in prefill, the DCP path of
    ``attention_decode`` in decode). KV capacity grows by ``dcp_size``.

    <-- MODEL-SPECIFIC: per-head q_norm / k_norm applied before RoPE.
    """

    def __init__(self, config: HYV3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_name = f"layers.{layer_idx}.self_attn"
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5
        self.rms_norm_eps = config.rms_norm_eps

        # >>> PARALLELISM: TP group / head sharding <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
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

        # >>> PARALLELISM: DCP group = ranks sharing one replicated KV head <<<
        dcp_group = get_dcp_group()
        self.dcp_size = dcp_group.world_size
        self.dcp_group = dcp_group if self.dcp_size > 1 else None
        if self.dcp_size > 1 and self.num_kv_replicas % self.dcp_size != 0:
            raise ValueError(
                f"decode_context_parallel_size={self.dcp_size} must divide the "
                f"KV replication factor {self.num_kv_replicas} "
                f"(tensor_parallel_size / num_key_value_heads)."
            )

        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim
        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        # <-- STATIC-FP8: fp8 weights + [128, 3] / [128, 1] fp32 scales.
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, q_size + 2 * kv_size, dtype=_FP8_DTYPE),
            requires_grad=False,
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(q_size, self.hidden_size, dtype=_FP8_DTYPE),
            requires_grad=False,
        )
        self.qkv_weight_scale = _fp32_param(_PMAX, 3)
        self.qkv_input_scale = _fp32_param(_PMAX, 1)
        self.o_weight_scale = _fp32_param(_PMAX, 1)
        self.o_input_scale = _fp32_param(_PMAX, 1)

        # <-- MODEL-SPECIFIC: per-head QK-norm (head_dim RMSNorm, replicated)
        self.q_norm = HYV3RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)
        self.k_norm = HYV3RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)

        # KV caches bound externally via bind_kv_cache(): BF16, or FP8 with K in
        # either the standard [blocks, kvh, block, D] or the packed
        # [blocks, kvh, block // 2, D, 2] layout (fp8_packed, set at bind time).
        self.k_cache = None
        self.v_cache = None
        self.fp8_packed = False
        # FP8 KV quantizing scales q = 1 / dequant (cache = fp8(x * q)), set from
        # the KV scale profile by load_weights / load_weights_lite. 1.0 is only
        # the BF16-KV value; FP8 KV refuses to bind without a loaded profile.
        self.k_scale_float = 1.0
        self.v_scale_float = 1.0
        self.kv_scales_loaded = False

        self._setup_weight_loaders()

    def set_kv_scales(self, q_k: float, q_v: float, device) -> None:
        """<-- FP8 KV: this layer's quantizing scales (q = 1 / dequant).

        Decode passes them to the megakernel as compile-time values. Prefill
        reads them from ``kv_scale_t`` ([q_k, q_v, 1/q_k, 1/q_v], a graph input)
        so that all 80 prefill layers stay structurally identical: with
        per-layer constants the compiler cannot reuse one layer module for the
        others, which made the full-model prefill NEFFs 10-16x larger and their
        cold compile take hours.
        """
        self.k_scale_float, self.v_scale_float = q_k, q_v
        self.kv_scale_t = nn.Parameter(
            torch.tensor(
                [q_k, q_v, 1.0 / q_k, 1.0 / q_v], dtype=torch.float32, device=device
            ),
            requires_grad=False,
        )
        self.kv_scales_loaded = True

    def _setup_weight_loaders(self):
        set_weight_loader(
            self.qkv_proj_weight,
            with_fp8_halving(
                fused_qkv_weight_loader(
                    q_size=self.q_size,
                    kv_size=self.kv_size,
                    shard_dim=1,
                    num_shards=self.world_size,
                    is_storage_transposed=True,
                    num_kv_replicas=self.num_kv_replicas,
                )
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            with_fp8_halving(
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=self.q_size,
                    num_shards=self.world_size,
                    is_storage_transposed=True,
                )
            ),
        )
        set_weight_loader(self.qkv_weight_scale, scalar_scale_loader(3, cols=3))
        set_weight_loader(self.qkv_input_scale, scalar_scale_loader(3, cols=1))
        set_weight_loader(self.o_weight_scale, scalar_scale_loader())
        set_weight_loader(self.o_input_scale, scalar_scale_loader())

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
        ln_w: torch.Tensor,
        is_decode: bool,
        rank: torch.Tensor,
    ) -> torch.Tensor:
        # >>> PARALLELISM: DCP rank as a runtime tensor (SPMD-clean) <<<
        dcp_rank = self.dcp_group.rank_in_group_tensor if self.dcp_size > 1 else None
        if is_decode:
            return self.forward_decode(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
                ln_w,
                dcp_rank,
            )

        # Pre-attention RMSNorm in the SP region; the STATIC qkv kernel needs bf16 input.
        x = hidden_states.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps)
        hidden_states = (x * ln_w.view(1, -1)).to(self.dtype)
        # >>> PARALLELISM: All-gather from SP <<<
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(
            hidden_states, position_embeddings, attn_metadata, dcp_rank
        )

    def _qk_norm(self, t: torch.Tensor, norm: HYV3RMSNorm) -> torch.Tensor:
        tf = t.to(torch.float32)
        tf = tf * torch.rsqrt(tf.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
        return tf * norm.weight.to(torch.float32)

    def _fp8_prefill_q(self, q: torch.Tensor) -> torch.Tensor:
        """Q * (1 / q_k): (Q / q_k) . (K * q_k) = Q . K for the cached FP8 K."""
        return (q.to(torch.float32) * self.kv_scale_t[2]).to(q.dtype)

    def _fp8_prefill_out(self, out: torch.Tensor) -> torch.Tensor:
        """Attention over V * q_v, times 1 / q_v."""
        return (out.to(torch.float32) * self.kv_scale_t[3]).to(out.dtype)

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
        dcp_rank: torch.Tensor | None,
    ) -> torch.Tensor:
        tokens = hidden_states.shape[0]
        meta = attn_metadata[self.layer_name]

        # Step 1: QKV projection (STATIC fp8)
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
            d_head=self.head_dim,
            quantization_type=QuantizationType.STATIC,
            qkv_w_scale=self.qkv_weight_scale,
            qkv_in_scale=self.qkv_input_scale,
            num_q_heads=self.num_attention_heads_per_rank,
            num_kv_heads=self.num_key_value_heads_per_rank,
        ).squeeze(0)
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

        # Step 2: <-- MODEL-SPECIFIC: QK-norm then rotate_half RoPE (fp32)
        cos, sin = position_embeddings
        cos = cos.unsqueeze(0).to(torch.float32)
        sin = sin.unsqueeze(0).to(torch.float32)
        q = self._qk_norm(q, self.q_norm)
        k = self._qk_norm(k, self.k_norm)
        q = (q * cos + _rotate_half(q) * sin).to(self.dtype)
        k = (k * cos + _rotate_half(k) * sin).to(self.dtype)

        # Step 3: KV cache write. Padding slots (-1) are redirected to the null block.
        # >>> PARALLELISM: with DCP the slot_mapping is full length; tokens this
        # rank does not own map to the null block, so the full K/V is written.
        slot_mapping = meta["slot_mapping"]
        block_size = meta["block_size"]
        block_idx, pos_idx = paged_slot_indices(
            slot_mapping, block_size, self.k_cache.shape[0]
        )
        nkh = self.num_key_value_heads_per_rank
        index = (
            block_idx.repeat(nkh),
            torch.arange(nkh, dtype=torch.long, device=q.device).repeat_interleave(
                slot_mapping.shape[0]
            ),
            pos_idx.repeat(nkh),
        )
        # <-- FP8 KV: K is quantized after QK-norm + RoPE, V after v_proj, with
        # the profile's static per-layer scale.
        kv_fp8 = _is_fp8_kv(self.k_cache)
        if kv_fp8:
            k_rows = _quantize_kv(
                k.reshape(-1, self.head_dim), self.kv_scale_t[0], self.k_cache.dtype
            )
            v_rows = _quantize_kv(
                v.reshape(-1, self.head_dim), self.kv_scale_t[1], self.v_cache.dtype
            )
        else:
            k_rows = k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
            v_rows = v.reshape(-1, self.head_dim).to(self.v_cache.dtype)
        self.v_cache.index_put_(index, v_rows)
        if self.fp8_packed:
            # Packed K [blocks, kvh, block // 2, D, 2] stores token pairs
            # (2j, 2j + 1). One prefill per step starting at a block-aligned
            # (prefix hit) or chunk-aligned position keeps rows pair-aligned.
            _write_packed_k(self.k_cache, k_rows, *index)
        else:
            self.k_cache.index_put_(index, k_rows)

        # Step 4: Attention
        kv_segment_size = meta.get("kv_segment_size")
        if kv_fp8 and (self.dcp_size > 1 or kv_segment_size):
            # The segmented kernels read a non-packed FP8 K cache wrongly.
            validate_fp8_segmented_supported(kv_fp8, self.fp8_packed)
        if self.dcp_size > 1:
            # Full Q against this rank's local KV shard (read from the cache,
            # which now includes the current tokens), LSE-combined across the
            # DCP group.
            cached_seq_len = meta.get("cached_seq_len")
            k_attn, v_attn = self.k_cache, self.v_cache
            block_table = meta["block_table_tensor"]
            if kv_fp8:
                # <-- FP8 KV under DCP: on the installed nkilib (nki 0.7.0) the
                # KV-parallel segmented kernel reads the PRIOR segments of a
                # packed FP8 K cache wrongly (multi-chunk prefill / prefix hits
                # diverge from BF16 KV, with or without its k_scale / v_scale;
                # the first chunk is correct). Read a BF16 copy of this rank's
                # blocks of the request instead (exact for FP8 codes).
                k_attn, v_attn, block_table = _request_blocks_bf16(
                    self.k_cache, self.v_cache, block_table, self.fp8_packed, self.dtype
                )
            attn_output = NF.segmented_attention_cp(
                q=self._fp8_prefill_q(q) if kv_fp8 else q,
                k_cache=k_attn,
                v_cache=v_attn,
                block_tables=block_table,
                prior_tokens=(
                    cached_seq_len // self.dcp_size
                    if cached_seq_len is not None
                    else None
                ),
                block_size=block_size,
                cp_rank=dcp_rank.reshape(1, 1).to(torch.int32),
                cp_world_size=self.dcp_size,
                cp_kv_cache_interleave_size=block_size,
                cp_group=self.dcp_group,
                # <-- FP8 KV: the BF16 copy holds k * q_k / v * q_v. Q carries the
                # K dequant (see _fp8_prefill_q) and the V dequant is applied
                # after the LSE combine (linear in V).
                scale=self.scaling,
                tp_q=True,
                tp_out=True,
            )
            if kv_fp8:
                attn_output = self._fp8_prefill_out(attn_output)
        elif kv_segment_size:
            # <-- FP8 KV: the cache holds k * q_k / v * q_v. Q carries the K
            # dequant and the output the V dequant (runtime scales, so the
            # layers stay identical; see set_kv_scales).
            k_attn, v_attn = self.k_cache, self.v_cache
            block_table, packed_read = meta["block_table_tensor"], self.fp8_packed
            if kv_fp8 and _PREFILL_READ == "bf16":
                k_attn, v_attn, block_table = _request_blocks_bf16(
                    self.k_cache, self.v_cache, block_table, self.fp8_packed, self.dtype
                )
                packed_read = False
            attn_output = NF.segmented_attention(
                self._fp8_prefill_q(q) if kv_fp8 else q,
                k_cache=k_attn,
                v_cache=v_attn,
                block_tables=block_table,
                prior_tokens=meta.get("cached_seq_len"),
                block_size=block_size,
                kv_segment_size=kv_segment_size,
                scale=self.scaling,
                tp_q=True,
                tp_out=True,
                fp8_packed=packed_read,
            )
            if kv_fp8:
                attn_output = self._fp8_prefill_out(attn_output)
        else:
            # Full prefill attends over this step's BF16 K/V, not the cache.
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

        # Step 5: Output projection (STATIC fp8)
        attn_output = NF.o_proj(
            attn_output.unsqueeze(0),
            self.o_proj_weight,
            None,
            quantization_type=QuantizationType.STATIC,
            weight_scales=self.o_weight_scale,
            input_scales=self.o_input_scale,
        ).squeeze(0)

        # >>> PARALLELISM: Reduce-scatter back to SP <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)
        return attn_output.contiguous()

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
        ln_w: torch.Tensor,
        dcp_rank: torch.Tensor | None,
    ) -> torch.Tensor:
        meta = attn_metadata[self.layer_name]
        slot_mapping = meta["slot_mapping"]
        block_size = meta["block_size"]
        block_table = meta["block_table_tensor"]
        S_ctx = meta["max_blocks_per_seq"] * block_size

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        _validate_dcp_decode_sequence_length(self.dcp_size, S_decode)
        X = hidden_states.to(self.dtype).view(B, S_decode, hidden)

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

        # >>> PARALLELISM: DCP — the block table is already this rank's shard
        # (S_ctx is local). Mask the prior tokens this rank owns and let only
        # the owning rank count the active token, so it is combined once.
        local_filled = None
        dcp_active_mask = None
        if self.dcp_size > 1:
            W = self.dcp_size
            R = dcp_rank.to(torch.float32)
            n = positions.view(B, S_decode)[:, 0:1].to(torch.float32)
            vblock = block_size * W
            full_vblocks = torch.div(n, vblock, rounding_mode="trunc")
            remaining = n - full_vblocks * vblock
            local_filled = full_vblocks * block_size + torch.clamp(
                remaining - R * block_size, min=0, max=block_size
            )
            owner = torch.div(remaining, block_size, rounding_mode="trunc") % W
            dcp_active_mask = (owner == R).float()

        # The decode mask is identical for every layer: build it once per step.
        attention_mask = attn_metadata.get(_DECODE_MASK_CACHE_KEY)
        if attention_mask is None:
            attention_mask = NF.gen_attention_decode_mask(
                pos_ids=positions.view(1, B * S_decode).to(torch.float32),
                bs=B,
                q_head=self.num_attention_heads_per_rank * self.dcp_size,
                s_active=S_decode,
                s_prior=S_ctx,
                start_pos=None,
                block_len=block_size,
                local_filled_slots=local_filled,
                dcp_active_mask=dcp_active_mask,
            )
            attn_metadata[_DECODE_MASK_CACHE_KEY] = attention_mask

        # <-- FP8 KV (installed nkilib attention_block_tkg contract):
        #  * k_scale / v_scale are QUANT multipliers: the kernel writes the
        #    active token as fp8(clip(x * q)); the history is read undequantized.
        #  * K: the explicit softmax_scale must carry 1 / q_k.
        #  * V: the attention output reaches the STATIC O-projection scaled by
        #    q_v. The O-proj quantizes attn / input_scale and rescales by
        #    weight_scale * input_scale, so scale input_scale by q_v (the FP8
        #    input keeps its normal range) and weight_scale by 1 / q_v (their
        #    product, and so the output, is unchanged). Scaling only the weight
        #    scale would clip the O-proj input at 240 / q_v of its range.
        kv_fp8 = _is_fp8_kv(self.k_cache)
        o_weight_scale, o_input_scale = self.o_weight_scale, self.o_input_scale
        decode_kv_kwargs = {}
        packed_fallback = False
        k_cache_arg = self.k_cache
        if kv_fp8:
            if self.v_scale_float != 1.0:
                o_input_scale = o_input_scale * self.v_scale_float
                o_weight_scale = o_weight_scale / self.v_scale_float
            use_packed = (
                self.fp8_packed
                and not _FORCE_UNPACKED_DECODE
                and packed_fp8_viable_for_bucket(
                    block_len=block_size,
                    bs=B,
                    # DCP gathers the group's Q heads (as in the decode mask).
                    q_head=self.num_attention_heads_per_rank * self.dcp_size,
                    s_active=S_decode,
                    s_prior=S_ctx,
                )
            )
            # A packed cache on a bucket the packed kernel cannot take is read
            # (and updated) through a standard-layout copy; the updated copy is
            # swizzled back below so the next step sees the new token.
            packed_fallback = self.fp8_packed and not use_packed
            if packed_fallback:
                k_cache_arg = _unswizzle_packed_k(self.k_cache)
            decode_kv_kwargs = dict(
                k_scale=_scale_column(self.k_scale_float, X.device),
                v_scale=_scale_column(self.v_scale_float, X.device),
                fp8_packed=use_packed,
                return_updated_kv=packed_fallback,
            )

        output = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            rmsnorm_X_enabled=True,
            rmsnorm_X_eps=self.rms_norm_eps,
            rmsnorm_X_gamma=ln_w.view(1, -1),
            W_qkv=self.qkv_proj_weight,
            bias_qkv=None,
            quantization_type_qkv=QuantizationType.STATIC,
            weight_dequant_scale_qkv=self.qkv_weight_scale,
            input_dequant_scale_qkv=self.qkv_input_scale,
            # <-- MODEL-SPECIFIC: pre-RoPE QK-norm with learned gammas
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=self.q_norm.variance_epsilon,
            rmsnorm_QK_pre_rope_W_Q=self.q_norm.weight.view(1, -1),
            rmsnorm_QK_pre_rope_W_K=self.k_norm.weight.view(1, -1),
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=block_table,
            K_cache=k_cache_arg,
            V_cache=self.v_cache,
            attention_mask=attention_mask,
            softmax_scale=self.scaling / self.k_scale_float,
            sink=None,
            update_cache=True,
            kv_cache_update_idx=slot_mapping.view(B, S_decode).to(torch.uint32),
            W_out=self.o_proj_weight,
            bias_out=None,
            quantization_type_out=QuantizationType.STATIC,
            weight_dequant_scale_out=o_weight_scale,
            input_dequant_scale_out=o_input_scale,
            transposed_out=False,
            out_in_sb=False,
            dcp_size=self.dcp_size,
            dcp_group=self.dcp_group,
            dcp_active_owner=dcp_active_mask,
            **decode_kv_kwargs,
        )
        if packed_fallback:
            # Use the kernel's RETURNED K (k_cache_arg is the pre-kernel copy).
            output, k_updated, _ = output
            self.k_cache.copy_(_swizzle_packed_k(k_updated))

        # >>> PARALLELISM: TP all-reduce of the O projection (after the DCP combine) <<<
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output)
        return output.view(tokens, hidden)


# =============================================================================
# Section 3: Static-FP8 gated MLP (dense layer + shared expert)
# =============================================================================


class HYV3StaticFP8MLP(nn.Module):
    """SiLU-gated MLP with fused RMSNorm + static FP8 activation quant.

    The caller owns the collectives: ``forward`` returns this rank's
    (possibly partial) result for the tokens it was given.

    ``tp_sharded=True``: intermediate dim sharded across TP (dense layer).
    ``tp_sharded=False``: full weights on every rank (shared expert); used on
    SP-local tokens in prefill and on all tokens in decode, so no reduction
    is needed. ``num_shards`` / ``shard_rank`` instead shard the intermediate
    dim over a sub-group whose partials the caller combines (decode copy of
    the shared expert).
    """

    def __init__(
        self,
        config: HYV3Config,
        intermediate_size: int,
        tp_sharded: bool,
        num_shards: int | None = None,
        shard_rank: int | None = None,
    ):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.tp_sharded = tp_sharded and self.world_size > 1
        self.hidden_size = config.hidden_size
        self.act_dtype = config.torch_dtype

        # ``num_shards`` / ``shard_rank`` shard the intermediate dim over a
        # sub-group without TP collectives (the caller combines the partials).
        if num_shards is None:
            num_shards = self.world_size if self.tp_sharded else 1
            shard_rank = self.tp_group.rank_in_group if self.tp_sharded else 0
        shard = -(-intermediate_size // num_shards)
        self.intermediate_shard = shard
        self.intermediate_size_per_rank = (
            -(-shard // _MLP_INTERMEDIATE_ALIGN) * _MLP_INTERMEDIATE_ALIGN
        )
        I = self.intermediate_size_per_rank

        self.gate_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, I, dtype=_FP8_DTYPE), requires_grad=False
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, I, dtype=_FP8_DTYPE), requires_grad=False
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(I, self.hidden_size, dtype=_FP8_DTYPE), requires_grad=False
        )
        self.gate_weight_scale = _fp32_param(_PMAX, 1)
        self.up_weight_scale = _fp32_param(_PMAX, 1)
        self.down_weight_scale = _fp32_param(_PMAX, 1)
        self.gate_up_input_scale = _fp32_param(_PMAX, 1)
        self.down_input_scale = _fp32_param(_PMAX, 1)

        tp_rank = shard_rank
        set_weight_loader(
            self.gate_proj_weight, padded_mlp_loader(False, shard, I, tp_rank)
        )
        set_weight_loader(
            self.up_proj_weight, padded_mlp_loader(False, shard, I, tp_rank)
        )
        set_weight_loader(
            self.down_proj_weight, padded_mlp_loader(True, shard, I, tp_rank)
        )
        for name in ("gate_weight_scale", "up_weight_scale", "down_weight_scale"):
            set_weight_loader(getattr(self, name), scalar_scale_loader())
        set_weight_loader(self.gate_up_input_scale, scalar_scale_loader(2, cols=1))
        set_weight_loader(self.down_input_scale, scalar_scale_loader())

    def forward(
        self,
        hidden_states: torch.Tensor,
        ln_w: torch.Tensor,
        eps: float,
        is_prefill: bool,
    ) -> torch.Tensor:
        # >>> PARALLELISM: SP all-gather (TP-sharded MLP only) <<<
        gather = is_prefill and self.tp_sharded
        num_tokens = hidden_states.shape[0] * (self.world_size if gather else 1)

        # The kernel picks CTE vs TKG from its token count. CTE needs fp8 input
        # (explicit fused RMSNorm + static quant); TKG needs bf16 input and
        # fuses the RMSNorm itself (it fails to compile on fp8 input).
        if num_tokens > TKG_BS_SEQLEN_THRESHOLD:
            hidden_states = NF.rmsnorm_quant(
                hidden_states,
                ln_w=ln_w,
                input_dequant_scale=self.gate_up_input_scale,
                eps=eps,
                quantization_type=QuantizationType.STATIC,
            )
            norm_type, mlp_ln_w = NormType.NO_NORM, None
        else:
            norm_type, mlp_ln_w = NormType.RMS_NORM, ln_w.view(1, -1)
        if gather:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
            eps=eps,
            ln_w=mlp_ln_w,
            norm_type=norm_type,
            quantization_type=QuantizationType.STATIC,
            gate_w_scale=self.gate_weight_scale,
            up_w_scale=self.up_weight_scale,
            down_w_scale=self.down_weight_scale,
            gate_up_in_scale=self.gate_up_input_scale,
            down_in_scale=self.down_input_scale,
            output_dtype="bfloat16",
        )


class HYV3DenseMLP(nn.Module):
    """Dense MLP layer (first_k_dense_replace layers), TP-sharded."""

    def __init__(self, config: HYV3Config):
        super().__init__()
        self.mlp = HYV3StaticFP8MLP(config, config.intermediate_size, tp_sharded=True)
        self.tp_group = self.mlp.tp_group
        self.world_size = self.mlp.world_size

    def forward(self, hidden_states, ln_w, eps, is_prefill, rank):
        output = self.mlp(hidden_states, ln_w, eps, is_prefill)
        # >>> PARALLELISM: combine TP partial sums <<<
        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                output = self.tp_group.all_reduce(output)
        return output


# =============================================================================
# Section 4: MoE (sigmoid router + shared expert + FP8 routed experts)
# =============================================================================


def hy3_route(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    expert_bias: torch.Tensor,
    top_k: int,
    route_norm: bool,
    scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """<-- MODEL-SPECIFIC: Hy3 router.

    Returns dense ``[T, E]`` fp32 affinities (zero for unselected experts) and
    the ``[T, K]`` selected expert indices.
    """
    logits = hidden_states.to(torch.float32) @ router_weight.T
    scores = torch.sigmoid(logits)
    _, top_k_index = torch.topk(scores + expert_bias, top_k, dim=-1)
    top_k_weights = scores.gather(1, top_k_index)
    if route_norm:
        top_k_weights = top_k_weights / (
            top_k_weights.sum(dim=-1, keepdim=True) + 1e-20
        )
    top_k_weights = top_k_weights * scaling_factor
    affinities = torch.zeros_like(scores).scatter(1, top_k_index, top_k_weights)
    return affinities, top_k_index


def routed_experts_cte(
    normed: torch.Tensor,
    local_affinities: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_scale: torch.Tensor,
    top_k: int,
    moe_group,
    tp_degree: int,
) -> torch.Tensor:
    """Routed experts for prefill (``moe_cte``): this rank's partial output.

    ``local_affinities`` is ``[T, L]`` for the ``L`` local experts. Weights are
    FP8 (``[L, H, 2, I]`` / ``[L, I, H]``) with per-channel dequant scales.
    """
    T = normed.shape[0]
    num_local_experts = gate_up_weight.shape[0]
    masked, token_position_to_id, block_to_expert, conditions = (
        NF.build_blockwise_mapping(
            expert_affinities=local_affinities,
            num_local_experts=num_local_experts,
            num_experts_per_token=top_k,
            block_size=_MOE_BLOCK_SIZE,
            moe_group=moe_group,
            tp_degree=tp_degree,
        )
    )
    # shard_on_i_hybrid always runs `num_static_block` blocks and skips the
    # remaining (worst-case sized) block slots at runtime when they are
    # empty. With few local experts most of the static worst case is
    # padding, so this is ~3x faster than shard_on_i at T=2048.
    # (neuronx-cc 2.27 failed with NCC_ICFG001 when this kernel followed
    # segmented attention in one graph; fixed in neuronx-cc 2.28.)
    # HY3_MOE_CTE_IMPL=shard_on_i: neuronx-cc 2.28 fails shard_on_i_hybrid at
    # ep_degree 8 / 4 (I/tp 192 / 96) with NCC_IGCA108 in bwmm_shard_on_I.py.
    impl = (
        MoECTEImplementation.shard_on_i
        if os.environ.get("HY3_MOE_CTE_IMPL") == "shard_on_i"
        else MoECTEImplementation.shard_on_i_hybrid
    )
    return NF.moe_cte(
        implementation=impl,
        num_static_block=num_local_experts,
        conditions=conditions,
        hidden_states=normed,
        expert_affinities_masked=masked,
        gate_up_proj_weight=gate_up_weight,
        down_proj_weight=down_weight,
        gate_up_proj_scale=gate_up_scale,
        down_proj_scale=down_scale,
        activation_function=ActFnType.SiLU,
        block_size=_MOE_BLOCK_SIZE,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        skip_token=True,
        is_tensor_update_accumulating=True,
        compute_dtype=nl.bfloat16,
    )[:T]


def routed_experts_tkg(
    normed: torch.Tensor,
    affinities: torch.Tensor,
    top_k_index: torch.Tensor,
    rank_id: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_scale: torch.Tensor,
) -> torch.Tensor:
    """Routed experts for decode (``moe_tkg``, all local experts): partial output.

    ``affinities`` is ``[T, E]`` over all experts; ``rank_id`` (``[1, 1]``
    int32) selects this rank's ``L`` experts. Scales are per-expert scalars.
    """
    num_local_experts = gate_up_weight.shape[0]
    # moe_tkg selects STATIC from the presence of input scales; on TRN2 the
    # activations stay BF16 so the values are unused.
    dummy_in_scale = torch.ones(
        (num_local_experts, 1), dtype=torch.float32, device=normed.device
    )
    # NF.moe_tkg only admits MX / BF16 weights; call its kernel wrapper
    # directly for plain float8_e4m3fn weights with scalar scales.
    return wrap_nki(moe_tkg_wrapper)[2](
        hidden_input=normed,
        expert_gate_up_weights=gate_up_weight,
        expert_down_weights=down_weight,
        expert_affinities=affinities,
        expert_index=top_k_index.to(torch.int32),
        is_all_expert=True,
        rank_id=rank_id,
        expert_gate_up_weights_scale=gate_up_scale,
        expert_down_weights_scale=down_scale,
        expert_gate_up_input_scale=dummy_in_scale,
        expert_down_input_scale=dummy_in_scale,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        activation_fn=ActFnType.SiLU,
    )


def routed_experts_tkg_hits(
    normed: torch.Tensor,
    affinities: torch.Tensor,
    top_k_index: torch.Tensor,
    rank_id: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_scale: torch.Tensor,
) -> torch.Tensor:
    """Routed experts for a one-token decode (``moe_tkg_hits``): partial output.

    Same contract as ``routed_experts_tkg``, but only the top-k slots whose
    expert is one of this rank's ``L`` experts are computed: they are packed
    first and the kernel loops over their count at runtime.
    """
    e_l = gate_up_weight.shape[0]
    k = top_k_index.shape[1]
    base = rank_id.long().reshape(1, 1) * e_l
    local_ids = base.reshape(1) + torch.arange(e_l, device=normed.device, dtype=torch.long)
    local_aff = NF.get_local_expert_affinities(affinities, local_ids)  # [1, L]
    local_idx = top_k_index.long() - base
    valid = (local_idx >= 0) & (local_idx < e_l)
    # Local hits first, in slot order: the kernel runs the first num_hits.
    slot = torch.arange(k, device=normed.device, dtype=torch.float32).reshape(1, k)
    order = torch.topk(torch.where(valid, slot, slot + k), k, dim=1, largest=False).indices
    packed = torch.gather(torch.where(valid, local_idx, 0), 1, order)
    # int32 explicitly: sum() of an int32 tensor is int64.
    num_hits = valid.sum(dim=1, keepdim=True, dtype=torch.int32)
    dummy_in_scale = torch.ones((e_l, 1), dtype=torch.float32, device=normed.device)
    return wrap_nki(moe_tkg_hits)[2](
        hidden_input=normed,
        expert_gate_up_weights=gate_up_weight,
        expert_down_weights=down_weight,
        expert_affinities=local_aff.to(torch.float32),
        expert_index=packed.to(torch.int32),
        num_hits=num_hits,
        expert_gate_up_weights_scale=gate_up_scale,
        expert_down_weights_scale=down_scale,
        expert_gate_up_input_scale=dummy_in_scale,
        expert_down_input_scale=dummy_in_scale,
        activation_fn=ActFnType.SiLU,
    )



def routed_experts_tkg_hits_batch(
    normed: torch.Tensor,
    affinities: torch.Tensor,
    top_k_index: torch.Tensor,
    rank_id: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_scale: torch.Tensor,
) -> torch.Tensor:
    """Routed experts for a few-token decode (``moe_tkg_hits_batch``): partial output.

    Same contract as ``routed_experts_tkg``, but only this rank's experts that
    at least one token routed to are computed, each once for the whole batch:
    they are packed first (with their affinity columns) and the kernel loops
    over their count at runtime.
    """
    e_l = gate_up_weight.shape[0]
    base = rank_id.long().reshape(1, 1) * e_l
    expert = torch.arange(e_l, device=normed.device, dtype=torch.long)
    local_aff = NF.get_local_expert_affinities(affinities, base.reshape(1) + expert)  # [T, L]
    local_idx = top_k_index.long() - base  # [T, K], local if in [0, L)
    hit = (local_idx.unsqueeze(-1) == expert.view(1, 1, e_l)).any(dim=1).any(dim=0)  # [L]
    # Hit experts first, in expert order: the kernel runs the first num_hits.
    key = torch.where(hit, expert, expert + e_l).to(torch.float32).view(1, e_l)
    order = torch.topk(key, e_l, dim=1, largest=False).indices.view(e_l)
    packed_aff = torch.index_select(local_aff.to(torch.float32), 1, order)
    # int32 explicitly: sum() of a bool tensor is int64.
    num_hits = hit.sum(dtype=torch.int32).view(1, 1)
    dummy_in_scale = torch.ones((e_l, 1), dtype=torch.float32, device=normed.device)
    return wrap_nki(moe_tkg_hits_batch)[2](
        hidden_input=normed,
        expert_gate_up_weights=gate_up_weight,
        expert_down_weights=down_weight,
        expert_affinities=packed_aff,
        expert_index=order.view(1, e_l).to(torch.int32),
        num_hits=num_hits,
        expert_gate_up_weights_scale=gate_up_scale,
        expert_down_weights_scale=down_scale,
        expert_gate_up_input_scale=dummy_in_scale,
        expert_down_input_scale=dummy_in_scale,
        activation_fn=ActFnType.SiLU,
    )

class HYV3MoE(nn.Module):
    """Hy3 sparse MoE block.

    >>> PARALLELISM: EP (+ optional TP within an EP group) for routed experts <<<
    - Experts are placed linearly: EP rank k owns ``[k*L, (k+1)*L)``.
    - Router and shared expert are replicated on every rank.
    - Prefill: router + shared expert on SP-local tokens; tokens and
      affinities are all-gathered for the routed experts, whose partial
      outputs are reduce-scattered back to SP.
    - Decode: tokens are replicated; routed partial outputs are all-reduced.

    <-- MODEL-SPECIFIC routing:
        scores = sigmoid(x @ W_r^T)                      (fp32)
        top-k selection on scores + expert_bias          (bias selects only)
        weights = scores[topk] / sum(scores[topk]) * router_scaling_factor
    """

    def __init__(self, config: HYV3Config):
        super().__init__()
        from vllm.config import get_current_vllm_config

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.hidden_size = config.hidden_size
        self.total_num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_tok
        self.router_scaling_factor = config.router_scaling_factor
        self.route_norm = config.route_norm

        # >>> PARALLELISM: EP configuration <<<
        vllm_config = get_current_vllm_config()
        if vllm_config.parallel_config.enable_expert_parallel:
            from vllm_neuron.parallel.neuron_parallel_state import (
                get_neuron_ep_degree,
                get_neuron_ep_rank,
                get_neuron_ep_tp_group,
            )

            self.ep_degree = get_neuron_ep_degree()
            self.ep_rank = get_neuron_ep_rank()
            self.ep_tp_group = get_neuron_ep_tp_group()
            self.tp_degree = self.ep_tp_group.world_size
            self.ep_tp_rank = self.ep_tp_group.rank_in_group
        else:
            self.ep_degree = 1
            self.ep_rank = 0
            self.ep_tp_group = self.tp_group
            self.tp_degree = self.world_size
            self.ep_tp_rank = self.tp_group.rank_in_group

        self.num_local_experts = self.total_num_experts // self.ep_degree
        self.intermediate_size_per_rank = config.moe_intermediate_size // self.tp_degree
        L, I_tp, H = (
            self.num_local_experts,
            self.intermediate_size_per_rank,
            self.hidden_size,
        )

        # <-- MODEL-SPECIFIC: router (fp32, replicated) + selection-only bias
        self.router_weight = _fp32_param(self.total_num_experts, H)
        self.expert_bias = _fp32_param(self.total_num_experts)

        # <-- STATIC-FP8: routed expert weights (local experts, I sharded by tp_degree)
        self.gate_up_proj_weight = nn.Parameter(
            torch.empty(L, H, 2, I_tp, dtype=_FP8_DTYPE), requires_grad=False
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(L, I_tp, H, dtype=_FP8_DTYPE), requires_grad=False
        )
        # Prefill (moe_cte shard_on_i): per-channel dequant scales.
        self.gate_up_proj_scale = _fp32_param(L, 1, 2 * I_tp)
        self.down_proj_scale = _fp32_param(L, 1, H)
        # Decode (moe_tkg STATIC): per-expert scalar dequant scales.
        self.gate_up_proj_scale_tkg = _fp32_param(L, 2, 1)
        self.down_proj_scale_tkg = _fp32_param(L, 1)

        # <-- MODEL-SPECIFIC: shared expert(s), full weights on every rank
        shared_size = config.moe_intermediate_size * config.num_shared_experts
        self.shared_mlp = HYV3StaticFP8MLP(config, shared_size, tp_sharded=False)

        # >>> PARALLELISM: decode copy of the shared expert sharded over groups
        # of `_SHARED_DECODE_SHARDS` ranks. Its partial output (scaled by
        # shards / world_size, since every group computes a full copy) rides
        # on the routed experts' all-reduce, so decode reads ~1/8 of the
        # shared weights per rank with no extra collective.
        self.shared_decode_shards = _SHARED_DECODE_SHARDS
        if (
            self.world_size % self.shared_decode_shards == 0
            and self.world_size > self.shared_decode_shards
        ):
            self.shared_mlp_decode = HYV3StaticFP8MLP(
                config,
                shared_size,
                tp_sharded=False,
                num_shards=self.shared_decode_shards,
                shard_rank=self.tp_group.rank_in_group % self.shared_decode_shards,
            )
            self.shared_decode_weight = self.shared_decode_shards / self.world_size
        else:
            self.shared_mlp_decode = None

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        L, I_tp, H = (
            self.num_local_experts,
            self.intermediate_size_per_rank,
            self.hidden_size,
        )
        local_expert_indices = list(range(self.ep_rank * L, (self.ep_rank + 1) * L))

        def ep(loader):
            return expert_parallel_grouped_loader(
                local_expert_indices, loader, total_num_experts=self.total_num_experts
            )

        set_weight_loader(
            self.gate_up_proj_weight,
            ep(expert_gate_up_loader(L, I_tp, self.ep_tp_rank)),
        )
        set_weight_loader(
            self.down_proj_weight, ep(expert_down_loader(L, I_tp, self.ep_tp_rank))
        )
        set_weight_loader(
            self.gate_up_proj_scale,
            ep(
                expert_scale_loader(
                    L,
                    2,
                    lambda s: (
                        s.view(L, 2, 1).expand(L, 2, I_tp).reshape(L, 1, 2 * I_tp)
                    ),
                )
            ),
        )
        set_weight_loader(
            self.down_proj_scale,
            ep(expert_scale_loader(L, 1, lambda s: s.view(L, 1, 1).expand(L, 1, H))),
        )
        set_weight_loader(
            self.gate_up_proj_scale_tkg,
            ep(expert_scale_loader(L, 2, lambda s: s.view(L, 2, 1))),
        )
        set_weight_loader(
            self.down_proj_scale_tkg,
            ep(expert_scale_loader(L, 1, lambda s: s.view(L, 1))),
        )

    def _route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return hy3_route(
            hidden_states,
            self.router_weight,
            self.expert_bias,
            self.num_experts_per_token,
            self.route_norm,
            self.router_scaling_factor,
        )

    def _ep_row(self, device: torch.device) -> torch.Tensor:
        """``[1]`` int32 EP partition index of this worker, as a runtime tensor.

        Read from the EP group's stamped ``rank_in_group_tensor`` (not the
        Python int) so the traced graph is identical on every rank (SPMD).
        """
        if self.ep_degree <= 1:
            return torch.zeros(1, dtype=torch.int32, device=device)
        from vllm_neuron.parallel.neuron_parallel_state import get_neuron_ep_group

        return get_neuron_ep_group().rank_in_group_tensor.reshape(1).to(torch.int32)

    def forward(self, hidden_states, ln_w, eps, is_prefill, rank):
        # <-- MODEL-SPECIFIC: post-attention RMSNorm feeds router + routed experts
        x = hidden_states.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        normed = (x * ln_w.to(torch.float32)).to(hidden_states.dtype)
        affinities, top_k_index = self._route(normed)

        if is_prefill:
            # Shared expert: full weights on SP-local tokens, output complete.
            shared_output = self.shared_mlp(hidden_states, ln_w, eps, is_prefill)
            return self._forward_prefill(normed, affinities, rank) + shared_output

        if self.shared_mlp_decode is not None:
            shared_partial = (
                self.shared_mlp_decode(hidden_states, ln_w, eps, is_prefill)
                * self.shared_decode_weight
            )
            return self._forward_decode(
                normed, affinities, top_k_index, rank, shared_partial
            )
        shared_output = self.shared_mlp(hidden_states, ln_w, eps, is_prefill)
        return (
            self._forward_decode(normed, affinities, top_k_index, rank) + shared_output
        )

    def _forward_prefill(self, normed, affinities, rank):
        # >>> PARALLELISM: All-gather tokens + affinities from SP <<<
        if self.world_size > 1:
            affinities = self.tp_group.all_gather(affinities, dim=0)
            normed = self.tp_group.all_gather(normed, dim=0)

        # >>> PARALLELISM: keep only this rank's experts <<<
        if self.ep_degree > 1:
            local_expert_indices = self._ep_row(
                normed.device
            ).long() * self.num_local_experts + torch.arange(
                self.num_local_experts, device=normed.device, dtype=torch.long
            )
            affinities = NF.get_local_expert_affinities(
                affinities, local_expert_indices
            )

        output = routed_experts_cte(
            normed,
            affinities,
            self.gate_up_proj_weight,
            self.down_proj_weight,
            self.gate_up_proj_scale,
            self.down_proj_scale,
            top_k=self.num_experts_per_token,
            moe_group=self.ep_tp_group,
            tp_degree=self.tp_degree,
        )

        # >>> PARALLELISM: combine expert partials and return to SP <<<
        if self.world_size > 1:
            output = self.tp_group.reduce_scatter(output, dim=0)
        return output

    def _use_selective_decode(self, num_tokens: int) -> bool:
        """moe_tkg selective (only the routed experts' I/tp slices) vs all-expert.

        All-expert reads every local expert each step (all experts / world per
        rank whatever the EP x TP split); selective reads top_k slices of I/tp,
        so it wins while tokens x top_k < local experts (the gpt_oss rule).
        HY3_DECODE_SELECTIVE=1/0 forces it.
        """
        forced = os.environ.get("HY3_DECODE_SELECTIVE")
        if forced in ("0", "1"):
            return forced == "1"
        return num_tokens * self.num_experts_per_token < self.num_local_experts

    def _forward_decode_selective(self, normed, affinities, top_k_index):
        e_l = self.num_local_experts
        base = self._ep_row(normed.device).long().reshape(1, 1) * e_l
        local_ids = base.reshape(1) + torch.arange(e_l, device=normed.device, dtype=torch.long)
        local_aff = NF.get_local_expert_affinities(affinities, local_ids)  # [T, E_L]
        local_idx = top_k_index.long() - base
        valid = (local_idx >= 0) & (local_idx < e_l)
        # A slot whose expert lives in another EP group points at a local expert
        # this token did not select (affinity 0): it adds exactly nothing.
        spare = torch.argmin((local_aff > 0).to(torch.int32), dim=1, keepdim=True)
        local_idx = torch.where(valid, local_idx, spare)
        dummy_in_scale = torch.ones((e_l, 1), dtype=torch.float32, device=normed.device)
        return wrap_nki(moe_tkg_wrapper)[2](
            hidden_input=normed,
            expert_gate_up_weights=self.gate_up_proj_weight,
            expert_down_weights=self.down_proj_weight,
            # fp32: the selective kernel rejects bf16 affinities.
            expert_affinities=local_aff.to(torch.float32),
            expert_index=local_idx.to(torch.int32),
            is_all_expert=False,
            expert_gate_up_weights_scale=self.gate_up_proj_scale_tkg,
            expert_down_weights_scale=self.down_proj_scale_tkg,
            expert_gate_up_input_scale=dummy_in_scale,
            expert_down_input_scale=dummy_in_scale,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            activation_fn=ActFnType.SiLU,
        )

    def _use_hit_decode(self, num_tokens: int) -> bool:
        """moe_tkg_hits: one token, only the experts routed to this rank.

        Both moe_tkg modes run a fixed amount of expert work per token: all
        local experts, or all top_k slots, the non-local ones on a dummy
        expert. moe_tkg_hits loops over the local hits at runtime, so the
        layer costs the busiest rank's hits (usually 1 at EP64) instead.
        HY3_DECODE_HITS=0 turns it off.
        """
        return num_tokens == 1 and os.environ.get("HY3_DECODE_HITS", "1") != "0"

    def _forward_decode_hits(self, normed, affinities, top_k_index):
        return routed_experts_tkg_hits(
            normed,
            affinities,
            top_k_index,
            self._ep_row(normed.device).view(1, 1),
            self.gate_up_proj_weight,
            self.down_proj_weight,
            self.gate_up_proj_scale_tkg,
            self.down_proj_scale_tkg,
        )

    def _use_hit_batch_decode(self, num_tokens: int) -> bool:
        """moe_tkg_hits_batch: a few tokens, only the local experts any of them routed to.

        Each hit expert is read once for the batch. The busiest rank's hits
        grow with the batch, and once it hits all its experts the runtime loop
        is slower than moe_tkg's unrolled all-expert path, so the batch limit
        is 4 tokens per 3 local experts (EP64: 4, EP32: 8).
        HY3_DECODE_HITS_MAX_T overrides it; HY3_DECODE_HITS=0 turns it off.
        """
        if os.environ.get("HY3_DECODE_HITS", "1") == "0":
            return False
        max_t = os.environ.get("HY3_DECODE_HITS_MAX_T")
        limit = int(max_t) if max_t else (4 * self.num_local_experts) // 3
        return 1 < num_tokens <= limit

    def _forward_decode(self, normed, affinities, top_k_index, rank, extra=None):
        if self._use_hit_batch_decode(normed.shape[0]):
            output = routed_experts_tkg_hits_batch(
                normed,
                affinities,
                top_k_index,
                self._ep_row(normed.device).view(1, 1),
                self.gate_up_proj_weight,
                self.down_proj_weight,
                self.gate_up_proj_scale_tkg,
                self.down_proj_scale_tkg,
            )
            if extra is not None:
                output = output + extra
            if self.world_size > 1:
                output = self.tp_group.all_reduce(output)
            return output
        if self._use_hit_decode(normed.shape[0]):
            output = self._forward_decode_hits(normed, affinities, top_k_index)
            if extra is not None:
                output = output + extra
            if self.world_size > 1:
                output = self.tp_group.all_reduce(output)
            return output
        if self._use_selective_decode(normed.shape[0]):
            output = self._forward_decode_selective(normed, affinities, top_k_index)
            if extra is not None:
                output = output + extra
            if self.world_size > 1:
                output = self.tp_group.all_reduce(output)
            return output
        output = routed_experts_tkg(
            normed,
            affinities,
            top_k_index,
            self._ep_row(normed.device).view(1, 1),
            self.gate_up_proj_weight,
            self.down_proj_weight,
            self.gate_up_proj_scale_tkg,
            self.down_proj_scale_tkg,
        )
        if extra is not None:
            output = output + extra
        # >>> PARALLELISM: combine expert (and sharded shared-expert) partials <<<
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output)
        return output


# =============================================================================
# Section 5: Decoder layer / backbone
# =============================================================================


class HYV3DecoderLayer(nn.Module):
    def __init__(self, config: HYV3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.rms_norm_eps = config.rms_norm_eps
        self.input_layernorm = HYV3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = HYV3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.self_attn = HYV3Attention(config, layer_idx)
        self.mlp = (
            HYV3MoE(config) if config.is_moe_layer(layer_idx) else HYV3DenseMLP(config)
        )

    def forward(
        self,
        hidden_states,
        positions,
        position_embeddings,
        attn_metadata,
        is_decode,
        rank,
    ):
        # The norms are fused into the attention / MLP modules.
        attn_out = self.self_attn(
            hidden_states,
            positions,
            position_embeddings,
            attn_metadata,
            ln_w=self.input_layernorm.weight,
            is_decode=is_decode,
            rank=rank,
        )
        hidden_states = hidden_states + attn_out
        mlp_out = self.mlp(
            hidden_states,
            ln_w=self.post_attention_layernorm.weight,
            eps=self.rms_norm_eps,
            is_prefill=not is_decode,
            rank=rank,
        )
        return hidden_states + mlp_out


class HYV3Model(nn.Module):
    """>>> PARALLELISM: SP during prefill — embed -> chunk -> layers -> all-gather. <<<"""

    def __init__(self, config: HYV3Config):
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
        self.layers = nn.ModuleList(
            [HYV3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = HYV3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = HYV3RotaryEmbedding(config)

    def forward(
        self,
        input_ids,
        positions,
        attn_metadata,
        is_prefill: bool,
        rank=None,
        inputs_embeds=None,
        is_token_ids=None,
    ):
        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
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
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
                is_decode=not is_prefill,
                rank=rank,
            )
        hidden_states = self.norm(hidden_states)

        # >>> PARALLELISM: SP — all-gather the full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return hidden_states


# =============================================================================
# Section 6: Causal LM + weight loading
# =============================================================================


class HYV3ForCausalLM(nn.Module):
    """Hy3 model with column-parallel LM head."""

    def __init__(self, config: HYV3Config):
        super().__init__()
        self.config = config
        self.model = HYV3Model(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        nc = config.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        self._gather_logits = nc is not None and (
            nc.max_logprobs != 0 or nc.debug_logits_dir is not None
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
                self.on_device_sampling_config, process_group=self.tp_group.device_group
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: dict | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ):
        positions = positions.to(torch.int32)
        first = attn_metadata["layers.0.self_attn"]
        is_prefill = first["max_query_len"] > first["decode_token_threshold"]

        T = input_ids.shape[0]
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt Length ({T}) must be > world_size ({self.world_size}) for SP."
            )

        hidden_states = self.model(
            input_ids,
            positions,
            attn_metadata,
            is_prefill=is_prefill,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
        )
        hidden_states = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )
        logits = self.lm_head(hidden_states)

        if self.on_device_sampling_config is None:
            return logits

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )
        gathered_logits = (
            self.tp_group.all_gather(logits, dim=1) if self._gather_logits else None
        )
        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        return cls(HYV3Config.from_configs(hf_config, neuron_config))

    # ── KV cache ──────────────────────────────────────────────────────────

    def get_kv_spec(self) -> KVSpec:
        return KVSpec(
            layers=[
                LayerSpec(
                    name=f"layers.{i}.self_attn",
                    num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                    head_size=layer.self_attn.head_dim,
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
                for i, layer in enumerate(self.model.layers)
            ]
        )

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
        for i, layer in enumerate(self.model.layers):
            name = f"layers.{i}.self_attn"
            if name not in kv_caches:
                raise KeyError(f"KV cache for layer {name} not initialized")
            attn = layer.self_attn
            k_cache, v_cache = kv_caches[name]
            k_fp8, v_fp8 = _is_fp8_kv(k_cache), _is_fp8_kv(v_cache)
            if k_fp8 != v_fp8:
                raise ValueError(
                    f"{name}: K cache {k_cache.dtype} and V cache {v_cache.dtype} "
                    "must both be FP8 or both BF16"
                )
            # Packed K (neuron_config.fp8_packed_kv) is one dim higher than V,
            # which is never packed.
            packed = k_cache.dim() == v_cache.dim() + 1
            if packed and not (k_fp8 and k_cache.dim() == 5 and v_cache.dim() == 4):
                raise ValueError(
                    f"{name}: unexpected KV layout K {tuple(k_cache.shape)} "
                    f"{k_cache.dtype}, V {tuple(v_cache.shape)} {v_cache.dtype}"
                )
            if k_fp8 and not attn.kv_scales_loaded:
                raise RuntimeError(
                    f"{name}: FP8 KV cache bound but no KV scale profile was loaded "
                    f"(set ${kv_scales.PROFILE_ENV})"
                )
            attn.k_cache = k_cache
            attn.v_cache = v_cache
            attn.fp8_packed = packed

    # ── Weight loading ────────────────────────────────────────────────────

    def _weight_mappings(self) -> dict[str, str | list[str]]:
        """<-- MODEL-SPECIFIC: model parameter name -> checkpoint key(s)."""
        cfg = self.config
        E = cfg.num_experts
        m: dict[str, str | list[str]] = {
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            "lm_head.weight": (
                "model.embed_tokens.weight"
                if cfg.tie_word_embeddings
                else "lm_head.weight"
            ),
            "model.norm.weight": "model.norm.weight",
        }

        def fp8_linear(dst: str, src: str):
            m[f"{dst}_weight"] = f"{src}.weight"

        for i, layer in enumerate(self.model.layers):
            p = f"model.layers.{i}"
            a = f"{p}.self_attn"
            m[f"{a}.qkv_proj_weight"] = [f"{a}.{n}_proj.weight" for n in "qkv"]
            m[f"{a}.qkv_weight_scale"] = [f"{a}.{n}_proj.weight_scale" for n in "qkv"]
            m[f"{a}.qkv_input_scale"] = [f"{a}.{n}_proj.input_scale" for n in "qkv"]
            m[f"{a}.o_proj_weight"] = f"{a}.o_proj.weight"
            m[f"{a}.o_weight_scale"] = f"{a}.o_proj.weight_scale"
            m[f"{a}.o_input_scale"] = f"{a}.o_proj.input_scale"
            m[f"{a}.q_norm.weight"] = f"{a}.q_norm.weight"
            m[f"{a}.k_norm.weight"] = f"{a}.k_norm.weight"
            m[f"{p}.input_layernorm.weight"] = f"{p}.input_layernorm.weight"
            m[f"{p}.post_attention_layernorm.weight"] = (
                f"{p}.post_attention_layernorm.weight"
            )

            def gated_mlp(dst: str, src: str):
                for n in ("gate", "up", "down"):
                    m[f"{dst}.{n}_proj_weight"] = f"{src}.{n}_proj.weight"
                    m[f"{dst}.{n}_weight_scale"] = f"{src}.{n}_proj.weight_scale"
                m[f"{dst}.gate_up_input_scale"] = [
                    f"{src}.gate_proj.input_scale",
                    f"{src}.up_proj.input_scale",
                ]
                m[f"{dst}.down_input_scale"] = f"{src}.down_proj.input_scale"

            if isinstance(layer.mlp, HYV3DenseMLP):
                gated_mlp(f"{p}.mlp.mlp", f"{p}.mlp")
                continue

            mp = f"{p}.mlp"
            m[f"{mp}.router_weight"] = f"{mp}.router.gate.weight"
            m[f"{mp}.expert_bias"] = f"{mp}.expert_bias"

            def experts(proj: str, suffix: str, mp: str = mp) -> list[str]:
                return [f"{mp}.experts.{e}.{proj}.{suffix}" for e in range(E)]

            gate_up_w = experts("gate_proj", "weight") + experts("up_proj", "weight")
            gate_up_s = experts("gate_proj", "weight_scale") + experts(
                "up_proj", "weight_scale"
            )
            m[f"{mp}.gate_up_proj_weight"] = gate_up_w
            m[f"{mp}.down_proj_weight"] = experts("down_proj", "weight")
            m[f"{mp}.gate_up_proj_scale"] = gate_up_s
            m[f"{mp}.gate_up_proj_scale_tkg"] = gate_up_s
            m[f"{mp}.down_proj_scale"] = experts("down_proj", "weight_scale")
            m[f"{mp}.down_proj_scale_tkg"] = experts("down_proj", "weight_scale")
            gated_mlp(f"{mp}.shared_mlp", f"{mp}.shared_mlp")
            if layer.mlp.shared_mlp_decode is not None:
                gated_mlp(f"{mp}.shared_mlp_decode", f"{mp}.shared_mlp")
        return m

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, self._weight_mappings(), device
        ).state_dict
        self.load_state_dict(rank_sharded, strict=False, assign=True)
        for name, param in self.named_parameters():
            if param.is_meta:
                raise RuntimeError(
                    f"Parameter {name} was not loaded from the checkpoint"
                )
        self._load_kv_scale_profile(checkpoint_path, device)

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """CPU compile: only the KV scales (decode uses them as graph constants)."""
        self._load_kv_scale_profile(checkpoint_path, device)

    def _load_kv_scale_profile(self, checkpoint_path: str, device) -> None:
        """<-- FP8 KV: per-layer K/V scales from the calibrated profile.

        No-op for a BF16 KV cache. For FP8 KV a profile is required (a missing
        one is an error, never a silent scale of 1.0); it is validated against
        this platform's FP8 range and the checkpoint's attention weights.
        """
        from vllm.config import get_current_vllm_config

        cache_dtype = get_current_vllm_config().cache_config.cache_dtype
        if cache_dtype not in QUANTIZED_KV_CACHE_DTYPES:
            return
        if not os.path.isdir(checkpoint_path):
            raise ValueError(
                "Hy3 FP8 KV needs a local checkpoint directory to validate the KV "
                f"scale profile against; got {checkpoint_path!r}"
            )
        path = kv_scales.resolve_profile_path(checkpoint_path)
        if path is None:
            raise ValueError(
                f"kv_cache_dtype={cache_dtype!r} needs a KV scale profile: set "
                f"${kv_scales.PROFILE_ENV} or put kv_scales.safetensors in the "
                "checkpoint directory (see scripts/calibrate_kv_scales.py)"
            )
        layers = self.model.layers
        q_k, q_v, meta = kv_scales.load_profile(
            path,
            checkpoint_path,
            num_layers=len(layers),
            target_fp8_max=FP8_CLAMP_MAX,
            check_fingerprint=os.environ.get("HY3_KV_SCALE_SKIP_FINGERPRINT") != "1",
        )
        for layer, qk, qv in zip(layers, q_k, q_v):
            layer.self_attn.set_kv_scales(qk, qv, device)
        logger.info(
            "Hy3 FP8 KV scale profile %s (sha256 %s, %d layers): "
            "q_k in [%.4g, %.4g], q_v in [%.4g, %.4g]",
            path,
            meta["sha256"][:16],
            len(layers),
            min(q_k),
            max(q_k),
            min(q_v),
            max(q_v),
        )
