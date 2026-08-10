# SPDX-License-Identifier: Apache-2.0
"""Complete native Qwen3 implementation for vLLM-Neuron 0.21.0.

Supports:
- Dense: Qwen3-32B (32 layers, GQA 32/8, SwiGLU MLP)
- MoE: Tongyi-DeepResearch-30B-A3B (48 layers, 128 experts, top-8)

Architecture: RMSNorm, standard RoPE, no bias, no sliding window, no sinks.
"""

import logging
import os
from dataclasses import dataclass, field

import nki.language as nl
import torch
import torch.nn as nn
import torch.nn.functional as F
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    NormType,
    RouterActFnType,
)
from transformers import PretrainedConfig

from vllm.distributed.parallel_state import (
    get_tp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.dtype_utils import (
    FP8_CLAMP_MAX,
    validate_fp8_segmented_supported,
)
import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    fused_qkv_weight_loader as standard_fused_qkv_weight_loader,
    set_weight_loader,
    with_rank_override,
    sharding_weight_loader,
)
from vllm_neuron.functional.attention.attention_decode import (
    _swizzle_packed_k,
    _unswizzle_packed_k,
)
from vllm_neuron.functional.attention.attention_decode_mask import _resize_block_len
from vllm_neuron.functional.moe.router import RouterComputationOrder

logger = logging.getLogger(__name__)


def _packed_fp8_viable_for_bucket(
    block_len: int, bs: int, q_head: int, s_active: int, s_prior: int
) -> bool:
    """Whether the packed FP8 decode kernel is usable for this bucket geometry."""
    if block_len <= 0 or s_prior <= 0:
        return False
    return _resize_block_len(block_len, bs, q_head, s_active, s_prior) >= 2


# ============================================================================
# Config
# ============================================================================

@dataclass
class Qwen3Config:
    """Qwen3 configuration matching HF transformers."""

    # Architecture
    vocab_size: int = 151936
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 11008
    moe_intermediate_size: int = 0
    rms_norm_eps: float = 1e-6

    # MoE (0 = dense)
    num_local_experts: int = 0
    num_experts_per_tok: int = 0
    use_qk_norm: bool = True
    norm_topk_prob: bool = False
    decoder_sparse_step: int = 1
    mlp_only_layers: list[int] = field(default_factory=list)

    # Feature flags
    attention_bias: bool = False
    mlp_bias: bool = False

    # RoPE
    rope_theta: float = 10000.0
    max_position_embeddings: int = 32768

    # Runtime
    torch_dtype: torch.dtype = torch.bfloat16
    neuron_config: NeuronConfig | None = None

    @classmethod
    def from_configs(cls, hf_config, neuron_config):
        """Build from HF + Neuron configs."""
        if isinstance(hf_config, dict):
            config_dict = hf_config
        else:
            config_dict = hf_config.to_dict()

        # Extract known fields
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in field_names}

        # Parse torch_dtype
        if "torch_dtype" in filtered and isinstance(filtered["torch_dtype"], str):
            filtered["torch_dtype"] = getattr(torch, filtered["torch_dtype"])

        filtered["neuron_config"] = neuron_config
        filtered["num_local_experts"] = config_dict.get(
            "num_experts", config_dict.get("num_local_experts", 0)
        )
        filtered["num_experts_per_tok"] = config_dict.get(
            "num_experts_per_tok", 0
        )
        filtered["moe_intermediate_size"] = config_dict.get(
            "moe_intermediate_size", 0
        )
        filtered["use_qk_norm"] = config_dict.get("use_qk_norm", True)

        # Newer Transformers versions store Qwen3's RoPE base in
        # ``rope_parameters`` rather than the legacy top-level
        # ``rope_theta`` field. Falling back to this dataclass's 10k default
        # silently corrupts long-context positions for official Qwen3 models,
        # whose base is 1M.
        rope_theta = config_dict.get("rope_theta")
        if rope_theta is None:
            rope_parameters = config_dict.get("rope_parameters") or {}
            rope_theta = rope_parameters.get("rope_theta")
        if rope_theta is not None:
            filtered["rope_theta"] = float(rope_theta)

        config = cls(**filtered)
        if config.attention_bias:
            raise ValueError("Qwen3 attention_bias=True is not supported")
        if not config.use_qk_norm:
            raise ValueError("Qwen3 use_qk_norm=False is not supported")
        if config.torch_dtype != torch.bfloat16:
            raise ValueError(
                "Qwen3 currently supports only torch.bfloat16, "
                f"got {config.torch_dtype}"
            )
        return config


def _sanitize_slot_mapping(
    slot_mapping: torch.Tensor, max_slot: int
) -> torch.Tensor:
    """Map padding/out-of-range cache slots to vLLM's reserved null block."""
    return torch.where(
        (slot_mapping < 0) | (slot_mapping >= max_slot),
        torch.zeros_like(slot_mapping),
        slot_mapping,
    ).to(torch.long)


# ============================================================================
# Basic Layers
# ============================================================================

class Qwen3RMSNorm(nn.Module):
    """RMS Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class Qwen3RotaryEmbedding(nn.Module):
    """Standard rotary position embedding for Qwen3."""

    def __init__(self, head_dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # No buffers - compute on-the-fly to avoid meta tensor issues

    def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
        """Compute inverse frequencies on target device."""
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
        )
        return inv_freq

    def forward(
        self,
        positions: torch.Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cos, sin) for given positions."""
        if device is None:
            device = positions.device
        if dtype is None:
            dtype = torch.float32

        inv_freq = self._compute_inv_freq(device)
        # Keep the phase calculation in FP32. BF16 cannot represent every
        # integer position above 256, so computing frequencies in the model
        # dtype silently aliases long-context token positions.
        positions = positions.to(device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        return cos, sin


# ============================================================================
# Attention
# ============================================================================

class Qwen3Attention(nn.Module):
    """Multi-head attention with GQA and per-head QK RMSNorm.

    Architecture:
    - Fused QKV projection (no bias)
    - Per-head RMSNorm on Q and K before RoPE
    - M-RoPE applied to Q and K
    - Flash attention (prefill) or fused megakernel (decode)
    - Output projection (no bias)

    Parallelism (keep as-is when porting):
    - Q/K/V heads sharded across TP ranks
    - KV heads replicated when fewer than TP size
    - Prefill: SP all-gather → compute → reduce-scatter
    - Decode: fused megakernel with TP all-reduce
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.rms_norm_eps = config.rms_norm_eps
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        # >>> PARALLELISM: Head sharding calculation <<<
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

        # >>> PARALLELISM: QKV weight shapes for TP <<<
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

        # Per-head QK RMSNorm before RoPE
        self.q_layernorm = Qwen3RMSNorm(
            self.head_dim, config.rms_norm_eps, self.dtype
        )
        self.k_layernorm = Qwen3RMSNorm(
            self.head_dim, config.rms_norm_eps, self.dtype
        )

        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        self.k_cache = None
        self.v_cache = None

        # FP8 KV cache support
        self.fp8_packed = False
        self.register_buffer("k_scale", None, persistent=False)
        self.register_buffer("v_scale", None, persistent=False)
        self.k_scale_float = 1.0
        self.v_scale_float = 1.0

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach weight loaders for checkpoint → parameter transformation.

        >>> PARALLELISM: Weight loaders handle TP sharding of checkpoint tensors.
        No bias. is_storage_transposed=True for HF checkpoint format.
        """
        set_weight_loader(
            self.qkv_proj_weight,
            standard_fused_qkv_weight_loader(
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

    # ── KV cache write helper ───────────────────────────────────────────

    def _write_paged_kv_cache(self, k, v, slot_mapping, block_size):
        """Scatter post-RoPE K/V into the paged cache.

        FP8 caches store fp8(clamp(tensor * scale)); BF16 caches store directly.
        Packed FP8 K is un-swizzled, scattered, then re-swizzled in place.
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
        max_slot = self.k_cache.shape[0] * block_size
        slot_mapping = _sanitize_slot_mapping(slot_mapping, max_slot)
        block_indices = (slot_mapping // block_size).repeat(nkh)
        position_indices = (slot_mapping % block_size).repeat(nkh)
        head_indices = torch.arange(
            nkh, dtype=torch.long, device=k.device
        ).repeat_interleave(slot_mapping.shape[0])
        index = (block_indices, head_indices, position_indices)

        self.v_cache.index_put_(index, v_flat)
        if self.fp8_packed:
            k_unpacked = _unswizzle_packed_k(self.k_cache)
            k_unpacked.index_put_(index, k_flat)
            self.k_cache.copy_(_swizzle_packed_k(k_unpacked))
        else:
            self.k_cache.index_put_(index, k_flat)

    # ── Forward dispatch ─────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ):
        """Dispatch to prefill or decode path based on metadata."""
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
        """Prefill: QKV proj → QK norm → RoPE → flash attention → O proj."""
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # Step 1: Fused QKV Projection + QK RMSNorm + M-RoPE
        cos, sin = position_embeddings
        cos_cache = cos.unsqueeze(0)
        sin_cache = sin.unsqueeze(0)

        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            num_q_heads=self.num_attention_heads_per_rank,
            num_kv_heads=self.num_key_value_heads_per_rank,
            d_head=self.head_dim,
            qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_eps=self.rms_norm_eps,
            qk_norm_pre_rope_q_gamma=self.q_layernorm.weight.unsqueeze(0),
            qk_norm_pre_rope_k_gamma=self.k_layernorm.weight.unsqueeze(0),
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

        # Write K/V into paged cache (handles FP8 quantization + packed layout)
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        kv_is_fp8 = self.k_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
        if kv_segment_size:
            validate_fp8_segmented_supported(kv_is_fp8, self.fp8_packed, self.k_cache)

        self._write_paged_kv_cache(k, v, slot_mapping, block_size)

        if kv_segment_size:
            if cached_seq_len is None:
                raise ValueError(
                    "cached_seq_len is required when segmented prefill is enabled"
                )
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

        # Step 5: Output Projection
        attn_output = attn_output.unsqueeze(0)
        attn_output = NF.o_proj(attn_output, self.o_proj_weight)
        attn_output = attn_output.squeeze(0)

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
        """Decode: fused megakernel with QK-norm and RoPE."""
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        assert tokens == B * S_decode

        hidden_states = hidden_states.to(self.dtype)
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

        pos_ids_kernel = positions.view(B, S_decode).to(torch.float32)

        kv_is_fp8 = self.k_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]

        # FP8 packed: check if the packed kernel is viable for this bucket
        S_ctx = block_table.shape[-1] * block_size
        use_packed_kernel = self.fp8_packed and _packed_fp8_viable_for_bucket(
            block_len=block_size,
            bs=B,
            q_head=self.num_attention_heads_per_rank,
            s_active=S_decode,
            s_prior=S_ctx,
        )
        k_cache = (
            self.k_cache
            if (use_packed_kernel or not self.fp8_packed)
            else _unswizzle_packed_k(self.k_cache)
        )
        if not self.fp8_packed:
            k_cache = (
                k_cache.squeeze(1) if k_cache.dim() == 4 and nkh else k_cache
            )
        v_cache = (
            self.v_cache.squeeze(1) if self.v_cache.dim() == 4 and nkh else self.v_cache
        )

        active_blocks_table = block_table

        W_q_norm = self.q_layernorm.weight.view(self.head_dim, 1)
        W_k_norm = self.k_layernorm.weight.view(self.head_dim, 1)

        output = NF.attention_decode(
            X=X,
            W_qkv=self.qkv_proj_weight,
            rmsnorm_X_enabled=False,
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_W_Q=W_q_norm,
            rmsnorm_QK_pre_rope_W_K=W_k_norm,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=active_blocks_table,
            K_cache=k_cache,
            V_cache=v_cache,
            pos_ids=pos_ids_kernel,
            swa_start_pos_ids=None,
            softmax_scale=self.scaling / self.k_scale_float,
            update_cache=True,
            kv_cache_update_idx=slot_mapping.view(B, S_decode).to(torch.uint32),
            fp8_packed=use_packed_kernel,
            W_out=self.o_proj_weight / self.v_scale_float,
            transposed_out=False,
            out_in_sb=False,
            k_scale=self.k_scale if kv_is_fp8 else None,
            v_scale=self.v_scale if kv_is_fp8 else None,
        )

        # Re-swizzle if we un-swizzled for a non-viable bucket
        if self.fp8_packed and not use_packed_kernel:
            self.k_cache.copy_(_swizzle_packed_k(k_cache))

        # >>> PARALLELISM: TP all-reduce after megakernel <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        return output
class Qwen3DenseMLP(nn.Module):
    """Dense SwiGLU MLP with TP intermediate sharding.

    Architecture: down_proj(silu(gate_proj(x)) * up_proj(x))
    No bias on any projection.

    Parallelism (keep as-is when porting):
    - gate/up/down projections sharded on intermediate dim across TP
    - Prefill: SP all-gather → compute → reduce-scatter
    - Decode: compute → all-reduce
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
        """SwiGLU forward with SP/TP collectives."""
        # >>> PARALLELISM: All-gather from SP for full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        )

        # >>> PARALLELISM: Combine TP shards <<<
        if is_prefill:
            if self.world_size > 1:
                output = self.tp_group.reduce_scatter(output, dim=0)
        else:
            if self.world_size > 1:
                self.tp_group.all_reduce(output)

        return output
# ============================================================================
# MLP - MoE
# ============================================================================

def _qwen_expert_gate_up_loader(num_experts, shard_size, num_shards):
    """Fuse HF per-expert gate/up matrices as [E, H, 2, I/TP]."""
    def transform(slices, rank):
        assert len(slices) == num_experts * 2
        start = (rank % num_shards) * shard_size
        end = start + shard_size
        result = []
        for expert_idx in range(num_experts):
            gate = slices[expert_idx][start:end, :].T
            up = slices[num_experts + expert_idx][start:end, :].T
            result.append(torch.stack((gate, up), dim=1))
        return torch.stack(result, dim=0)
    return SafetensorsWeightLoader(transform=transform)


def _qwen_expert_down_loader(num_experts, shard_size, num_shards):
    """Stack HF per-expert down matrices as [E, I/TP, H]."""
    def transform(slices, rank):
        assert len(slices) == num_experts
        start = (rank % num_shards) * shard_size
        end = start + shard_size
        return torch.stack([weight[:, start:end].T for weight in slices], dim=0)
    return SafetensorsWeightLoader(transform=transform)


class Qwen3MoeExperts(nn.Module):
    """Qwen3 BF16 sparse MoE using blockwise CTE and fused TKG kernels."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.total_num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.dtype = config.torch_dtype
        self.rms_norm_eps = config.rms_norm_eps
        self.norm_topk_prob = config.norm_topk_prob
        self.block_size = 256

        from vllm.config import get_current_vllm_config

        self.ep_enabled = (
            get_current_vllm_config().parallel_config.enable_expert_parallel
        )
        if self.ep_enabled:
            from vllm_neuron.parallel.neuron_parallel_state import (
                get_neuron_ep_degree,
                get_neuron_ep_rank,
                get_neuron_ep_tp_group,
            )
            self.ep_degree = get_neuron_ep_degree()
            self.ep_rank = get_neuron_ep_rank()
            self.ep_tp_group = get_neuron_ep_tp_group()
            self.tp_degree = self.ep_tp_group.world_size
            self.weight_rank = self.ep_tp_group.rank_in_group
        else:
            self.ep_degree = 1
            self.ep_rank = 0
            self.ep_tp_group = self.tp_group
            self.tp_degree = self.world_size
            self.weight_rank = self.tp_group.rank_in_group

        assert self.intermediate_size > 0
        assert self.total_num_experts % self.ep_degree == 0
        assert self.intermediate_size % self.tp_degree == 0
        self.num_experts = self.total_num_experts // self.ep_degree
        self.intermediate_size_per_rank = self.intermediate_size // self.tp_degree
        first_expert = self.ep_rank * self.num_experts
        self.local_expert_indices = list(
            range(first_expert, first_expert + self.num_experts)
        )

        # Decode fuses Qwen's post-attention RMSNorm into the TKG kernel.
        self.post_attention_layernorm = Qwen3RMSNorm(
            self.hidden_size, self.rms_norm_eps, self.dtype
        )
        self.router_weight = nn.Parameter(
            torch.empty(self.total_num_experts, self.hidden_size, dtype=self.dtype)
        )
        self.gate_up_proj_weight = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.hidden_size,
                2,
                self.intermediate_size_per_rank,
                dtype=self.dtype,
            )
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.intermediate_size_per_rank,
                self.hidden_size,
                dtype=self.dtype,
            )
        )
        gate_up_loader = _qwen_expert_gate_up_loader(
            self.num_experts, self.intermediate_size_per_rank, self.tp_degree
        )
        down_loader = _qwen_expert_down_loader(
            self.num_experts, self.intermediate_size_per_rank, self.tp_degree
        )
        if self.weight_rank != self.tp_group.rank_in_group:
            gate_up_loader = with_rank_override(gate_up_loader, self.weight_rank)
            down_loader = with_rank_override(down_loader, self.weight_rank)
        set_weight_loader(self.gate_up_proj_weight, gate_up_loader)
        set_weight_loader(self.down_proj_weight, down_loader)

        self.fp8_weights_enabled = False
        self.fp8_only = False
        self.gate_up_weights_scale = None
        self.down_weights_scale = None
        self.gate_up_input_scale = None
        self.down_input_scale = None

    def quantize_weights_to_fp8(self, fp8_only=False):
        """Create FP8 weight copies for decode (STATIC quant path on Trn2).

        The NKI moe_block_tkg kernel's STATIC quantization path performs
        BF16 × FP8 matmul natively on Trn2 hardware. Weights are stored as
        torch.float8_e4m3fn (non-packed).

        If fp8_only=True, the original BF16 weights are deleted after
        quantization to save HBM. Prefill will dequantize FP8→BF16 on the fly.
        """
        _FP8_MAX = 240.0
        E = self.num_experts
        device = self.gate_up_proj_weight.device

        gate_up_w = self.gate_up_proj_weight.data.cpu().float()  # [E, H, 2, I]
        down_w = self.down_proj_weight.data.cpu().float()  # [E, I, H]

        gate_up_scales = torch.zeros(E, 2, 1, dtype=torch.float32)
        down_scales = torch.zeros(E, 1, dtype=torch.float32)

        for e in range(E):
            gate_amax = gate_up_w[e, :, 0, :].abs().max().item()
            up_amax = gate_up_w[e, :, 1, :].abs().max().item()
            down_amax = down_w[e].abs().max().item()

            gate_scale = gate_amax / _FP8_MAX if gate_amax > 0 else 1.0
            up_scale = up_amax / _FP8_MAX if up_amax > 0 else 1.0
            down_scale = down_amax / _FP8_MAX if down_amax > 0 else 1.0

            gate_up_w[e, :, 0, :] = (gate_up_w[e, :, 0, :] / gate_scale).clamp(
                -_FP8_MAX, _FP8_MAX
            )
            gate_up_w[e, :, 1, :] = (gate_up_w[e, :, 1, :] / up_scale).clamp(
                -_FP8_MAX, _FP8_MAX
            )
            down_w[e] = (down_w[e] / down_scale).clamp(-_FP8_MAX, _FP8_MAX)

            gate_up_scales[e, 0, 0] = gate_scale
            gate_up_scales[e, 1, 0] = up_scale
            down_scales[e, 0] = down_scale

        gate_up_fp8 = gate_up_w.to(torch.float8_e4m3fn)  # [E, H, 2, I]
        down_fp8 = down_w.to(torch.float8_e4m3fn)  # [E, I, H]

        self.register_buffer('gate_up_proj_weight_fp8', gate_up_fp8.to(device))
        self.register_buffer('down_proj_weight_fp8', down_fp8.to(device))

        self.gate_up_weights_scale = gate_up_scales.to(device)
        self.down_weights_scale = down_scales.to(device)
        self.gate_up_input_scale = torch.ones(E, 1, dtype=torch.float32, device=device)
        self.down_input_scale = torch.ones(E, 1, dtype=torch.float32, device=device)
        self.fp8_weights_enabled = True
        self.fp8_only = fp8_only

        if fp8_only:
            del self.gate_up_proj_weight
            del self.down_proj_weight
            self.gate_up_proj_weight = None
            self.down_proj_weight = None

        logger.info(
            f"Quantized {E} expert weights to FP8 "
            f"(gate_up: {list(gate_up_fp8.shape)}, down: {list(down_fp8.shape)}, "
            f"amax range: {gate_up_scales[:, 0, 0].min():.4f}-{gate_up_scales[:, 0, 0].max():.4f})"
            f"{' [FP8-only, BF16 originals deleted]' if fp8_only else ''}"
        )

    def _dequant_gate_up_for_prefill(self):
        """Dequantize FP8 gate_up weights back to BF16 for prefill CTE kernel.
        gate_up shape: [E, H, 2, I], scale shape: [E, 2, 1]"""
        w = self.gate_up_proj_weight_fp8.float()  # [E, H, 2, I]
        scale = self.gate_up_weights_scale  # [E, 2, 1]
        # Reshape scale to [E, 1, 2, 1] for broadcast across H and I dims
        w = w * scale.unsqueeze(1)  # [E, 2, 1] → unsqueeze(1) → [E, 1, 2, 1]
        return w.to(torch.bfloat16)

    def _dequant_down_for_prefill(self):
        """Dequantize FP8 down weights back to BF16 for prefill CTE kernel.
        down shape: [E, I, H], scale shape: [E, 1]"""
        w = self.down_proj_weight_fp8.float()  # [E, I, H]
        scale = self.down_weights_scale  # [E, 1]
        w = w * scale.unsqueeze(-1)  # [E, 1] → [E, 1, 1] broadcast to [E, I, H]
        return w.to(torch.bfloat16)

    def forward(self, hidden_states, positions, is_decode, rank=None):
        if is_decode:
            return self.forward_decode(hidden_states)
        return self.forward_prefill(hidden_states, positions)

    def forward_decode(self, hidden_states):
        use_all_experts = (self.ep_enabled or hidden_states.shape[0] * self.top_k >= self.num_experts)
        rank_id = None
        if use_all_experts:
            rank_id = torch.tensor(
                [[self.ep_rank]], dtype=torch.int32, device=hidden_states.device
            )
        fp8_kwargs = {}
        if self.fp8_weights_enabled:
            gate_up_w = self.gate_up_proj_weight_fp8
            down_w = self.down_proj_weight_fp8
            fp8_kwargs = dict(
                expert_gate_up_weights_scale=self.gate_up_weights_scale,
                expert_down_weights_scale=self.down_weights_scale,
                expert_gate_up_input_scale=self.gate_up_input_scale,
                expert_down_input_scale=self.down_input_scale,
            )
        else:
            gate_up_w = self.gate_up_proj_weight
            down_w = self.down_proj_weight
        output = NF.moe_block_tkg(
            inp=hidden_states.unsqueeze(0),
            gamma=self.post_attention_layernorm.weight.unsqueeze(0).to(torch.float32),
            router_weights=self.router_weight.T,
            expert_gate_up_weights=gate_up_w,
            expert_down_weights=down_w,
            rank_id=rank_id,
            top_k=self.top_k,
            eps=self.rms_norm_eps,
            router_act_fn=RouterActFnType.SOFTMAX,
            router_pre_norm=True,
            norm_topk_prob=self.norm_topk_prob,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            hidden_act_fn=ActFnType.SiLU,
            router_mm_dtype=nl.bfloat16,
            hidden_actual=self.hidden_size,
            is_all_expert=use_all_experts,
            skip_router_logits=True,
            **fp8_kwargs,
        )
        if self.fp8_weights_enabled:
            output = output.to(torch.bfloat16)
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output)
        return output


    def forward_prefill(self, hidden_states, positions):
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.norm_topk_prob:
            expert_affinities = NF.router(
                hidden_states=hidden_states,
                router_weights=self.router_weight.T,
                top_k=self.top_k,
                activation="softmax",
                computation_dtype=torch.float32,
                router_computation_order=(
                    RouterComputationOrder.PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER
                ),
            )
        else:
            # Qwen3 applies softmax over every expert before selecting top-k.
            # Unlike the normalized variant, the selected probabilities retain
            # their mass relative to all experts instead of summing to one.
            router_logits = F.linear(
                hidden_states.to(torch.float32),
                self.router_weight.to(torch.float32),
            )
            router_probs = F.softmax(router_logits, dim=-1)
            router_top_probs, router_indices = torch.topk(
                router_probs, self.top_k, dim=-1
            )
            expert_affinities = torch.zeros_like(router_probs).scatter(
                1, router_indices, router_top_probs
            )
        if self.world_size > 1:
            expert_affinities = self.tp_group.all_gather(expert_affinities, dim=0)
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        last_real_idx = torch.argmax(positions)
        token_indices = torch.arange(positions.shape[0], device=positions.device)
        padding_mask = token_indices <= last_real_idx
        if self.ep_enabled:
            local_indices = torch.arange(
                self.ep_rank * self.num_experts,
                (self.ep_rank + 1) * self.num_experts,
                device=hidden_states.device,
                dtype=torch.int32,
            )
            expert_affinities = NF.get_local_expert_affinities(
                expert_affinities, local_indices
            )

        (
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
            conditions,
        ) = NF.build_blockwise_mapping(
            expert_affinities=expert_affinities,
            num_local_experts=self.num_experts,
            num_experts_per_token=self.top_k,
            block_size=self.block_size,
            moe_group=self.ep_tp_group,
            tp_degree=self.tp_degree,
            padding_mask=padding_mask,
        )
        if self.fp8_weights_enabled and self.fp8_only:
            gate_up_w = self._dequant_gate_up_for_prefill()
            down_w = self._dequant_down_for_prefill()
        else:
            gate_up_w = self.gate_up_proj_weight
            down_w = self.down_proj_weight
        output = NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_block,
            conditions=conditions,
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=gate_up_w,
            down_proj_weight=down_w,
            activation_function=ActFnType.SiLU,
            block_size=self.block_size,
            token_position_to_id=token_position_to_id.to(torch.int32),
            block_to_expert=block_to_expert.to(torch.int32),
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
        )
        if self.world_size > 1:
            output = self.tp_group.reduce_scatter(output, dim=0)
        return output


# ============================================================================
# MLP Wrapper
# ============================================================================

class Qwen3MLP(nn.Module):
    """Conditional MLP: dense or MoE."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.is_moe = (
            config.num_local_experts > 0
            and layer_idx not in config.mlp_only_layers
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        )

        if self.is_moe:
            self.experts = Qwen3MoeExperts(config)
        else:
            self.dense_mlp = Qwen3DenseMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        is_decode: bool,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.is_moe:
            return self.experts(hidden_states, positions, is_decode, rank)
        else:
            return self.dense_mlp(hidden_states, is_prefill=not is_decode)


# ============================================================================
# Decoder Layer
# ============================================================================

class Qwen3DecoderLayer(nn.Module):
    """Transformer decoder layer: norm -> attn -> residual -> norm -> mlp -> residual."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps, config.torch_dtype)
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config, layer_idx)
        self.post_attention_layernorm = (
            None
            if self.mlp.is_moe
            else Qwen3RMSNorm(
                config.hidden_size, config.rms_norm_eps, config.torch_dtype
            )
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
        is_decode: bool,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Layer forward."""
        # Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Inject layer name into attn_metadata
        layer_name = f"layers.{self.layer_idx}.self_attn"
        attn_metadata_with_name = {**attn_metadata, "layer_name": layer_name}

        hidden_states = self.self_attn(
            hidden_states,
            positions,
            position_embeddings,
            attn_metadata_with_name,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        if not self.mlp.is_moe:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, positions, is_decode, rank)
        hidden_states = residual + hidden_states

        return hidden_states


# ============================================================================
# Model
# ============================================================================

class Qwen3Model(nn.Module):
    """Qwen3 transformer backbone."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config

        # TP groups
        self.tp_group = get_tp_group()
        self.rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()

        # Embedding
        self.embed_tokens = VocabDimShardedEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
        )

        # RoPE
        self.rotary_emb = Qwen3RotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

        # Decoder layers
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])

        # Final norm
        self.norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps, config.torch_dtype)


    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: dict,
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Model forward: embedding -> layers -> norm."""
        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name]["decode_token_threshold"]
        is_decode = max_query_len <= decode_token_threshold
        is_prefill = not is_decode

        # SP validation
        T = input_ids.shape[0]
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt length ({T}) must be > world_size ({self.world_size}) "
                f"and divisible by world_size for SP."
            )

        # Embedding
        hidden_states = self.embed_tokens(input_ids, scatter_tokens=is_prefill, rank=rank)

        # Merge prompt embeds if provided
        if inputs_embeds is not None and is_token_ids is not None:
            # SP shard inputs_embeds
            if is_prefill and self.world_size > 1:
                local_len = hidden_states.shape[0]
                start = self.rank * local_len
                inputs_embeds = inputs_embeds[start : start + local_len]
                is_token_ids = is_token_ids[start : start + local_len]

            hidden_states = NF.merge_prompt_embeds(hidden_states, inputs_embeds, is_token_ids)

        # RoPE
        position_embeddings = self.rotary_emb(positions, device=hidden_states.device, dtype=hidden_states.dtype)

        # Decoder layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
                is_decode,
                rank,
            )

        # Final norm
        hidden_states = self.norm(hidden_states)
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states


# ============================================================================
# CausalLM
# ============================================================================

class Qwen3ForCausalLM(nn.Module):
    """Qwen3 language model head."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)

        # TP info
        self.tp_group = get_tp_group()
        self.rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()

        # On-device sampler
        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config
            if config.neuron_config
            else None
        )
        self._gather_logits = (
            config.neuron_config is not None
            and config.neuron_config.max_logprobs != 0
        ) or (
            config.neuron_config is not None
            and config.neuron_config.debug_logits_dir is not None
        )

        # LM head
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=self.on_device_sampling_config is None,
            tp_group=self.tp_group.device_group,
        )

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(self.on_device_sampling_config, process_group=self.tp_group.device_group)
        else:
            self.sampler = None

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        """Build from configs."""
        config = Qwen3Config.from_configs(hf_config, neuron_config)
        return cls(config)

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
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Forward: input -> hidden -> logits (-> sample)."""
        positions = positions.to(torch.int32)

        # Model forward
        hidden_states = self.model(
            input_ids,
            positions,
            attn_metadata,
            rank,
            inputs_embeds,
            is_token_ids,
        )

        hidden_states_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )
        logits = self.lm_head(hidden_states_for_logits)

        gathered_logits = None
        if self._gather_logits:
            if self.lm_head.gather_output:
                gathered_logits = logits
            else:
                gathered_logits = self.tp_group.all_gather(logits, dim=1)

        # CPU sampling path (no on-device sampler)
        if self.sampler is None:
            return logits

        # Standard on-device sampling
        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )

        return sampled_tokens, gathered_logits

    def get_kv_spec(self):
        """Return KV cache specification."""
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
        """Bind KV cache tensors."""
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise Exception(f"KV cache for layer {layer_name} not initialized")
            k_cache, v_cache = kv_caches[layer_name]
            layer.self_attn.k_cache = k_cache
            layer.self_attn.v_cache = v_cache
            # Detect packed FP8 K layout (one rank higher than V)
            layer.self_attn.fp8_packed = k_cache.dim() == v_cache.dim() + 1

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load weights from HF checkpoint."""
        tp_rank = self.rank
        tp_size = self.world_size

        logger.info(
            f"Qwen3 load_weights: tp_rank={tp_rank}, tp_size={tp_size}"
        )

        mappings = {}

        # Embeddings
        mappings["model.embed_tokens.weight"] = "model.embed_tokens.weight"
        mappings["lm_head.weight"] = "lm_head.weight"
        mappings["model.norm.weight"] = "model.norm.weight"

        for layer_id in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_id}"

            # Attention (weight loaders handle the fusion)
            mappings[f"{prefix}.self_attn.qkv_proj_weight"] = [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ]
            mappings[f"{prefix}.self_attn.o_proj_weight"] = f"{prefix}.self_attn.o_proj.weight"
            mappings[f"{prefix}.self_attn.q_layernorm.weight"] = f"{prefix}.self_attn.q_norm.weight"
            mappings[f"{prefix}.self_attn.k_layernorm.weight"] = f"{prefix}.self_attn.k_norm.weight"
            mappings[f"{prefix}.input_layernorm.weight"] = f"{prefix}.input_layernorm.weight"

            # MLP
            layer = self.model.layers[layer_id]
            if layer.mlp.is_moe:
                mappings[
                    f"{prefix}.mlp.experts.post_attention_layernorm.weight"
                ] = f"{prefix}.post_attention_layernorm.weight"
                mappings[f"{prefix}.mlp.experts.router_weight"] = f"{prefix}.mlp.gate.weight"
                expert_ids = layer.mlp.experts.local_expert_indices if hasattr(
                    layer.mlp.experts, "local_expert_indices"
                ) else range(self.config.num_local_experts)
                expert_ids = list(expert_ids)
                mappings[f"{prefix}.mlp.experts.gate_up_proj_weight"] = [
                    f"{prefix}.mlp.experts.{e}.gate_proj.weight" for e in expert_ids
                ] + [
                    f"{prefix}.mlp.experts.{e}.up_proj.weight" for e in expert_ids
                ]
                mappings[f"{prefix}.mlp.experts.down_proj_weight"] = [
                    f"{prefix}.mlp.experts.{e}.down_proj.weight" for e in expert_ids
                ]
            else:
                mappings[f"{prefix}.post_attention_layernorm.weight"] = f"{prefix}.post_attention_layernorm.weight"
                mappings[f"{prefix}.mlp.dense_mlp.gate_proj_weight"] = f"{prefix}.mlp.gate_proj.weight"
                mappings[f"{prefix}.mlp.dense_mlp.up_proj_weight"] = f"{prefix}.mlp.up_proj.weight"
                mappings[f"{prefix}.mlp.dense_mlp.down_proj_weight"] = f"{prefix}.mlp.down_proj.weight"

        # Load checkpoint
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded_checkpoint = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, mappings, device
        ).state_dict

        self._load_kv_cache_scales(checkpoint, device)

        # Apply weights
        self.load_state_dict(rank_sharded_checkpoint, strict=False, assign=True)

        if os.environ.get("VLLM_NEURON_FP8_EXPERT_WEIGHTS") == "1":
            self._quantize_expert_weights_to_fp8()

        logger.info(f"Successfully loaded Qwen3 weights from {checkpoint_path}")

    def _load_kv_cache_scales(
        self, checkpoint: SafetensorsCheckpoint, device: torch.device
    ):
        """Load KV cache quantization scales from checkpoint if provided."""
        from vllm_neuron.utils.dtype_utils import QUANTIZED_KV_CACHE_DTYPES

        try:
            from vllm.config import get_current_vllm_config
            vllm_config = get_current_vllm_config()
            cache_dtype = vllm_config.cache_config.cache_dtype
        except Exception:
            return

        if cache_dtype not in QUANTIZED_KV_CACHE_DTYPES:
            return

        for layer_id in range(self.config.num_hidden_layers):
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

    def _quantize_expert_weights_to_fp8(self):
        """Quantize all MoE expert weights from BF16 to FP8 for decode speedup."""
        fp8_only = os.environ.get("VLLM_NEURON_FP8_ONLY") == "1"
        for layer_id in range(self.config.num_hidden_layers):
            layer = self.model.layers[layer_id]
            if hasattr(layer.mlp, 'is_moe') and layer.mlp.is_moe:
                layer.mlp.experts.quantize_weights_to_fp8(fp8_only=fp8_only)
        mode = "FP8-only (BF16 deleted)" if fp8_only else "FP8 decode + BF16 prefill"
        logger.info(f"FP8 expert weight quantization complete for all MoE layers [{mode}]")
