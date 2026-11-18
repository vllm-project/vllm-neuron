# SPDX-License-Identifier: Apache-2.0
"""
BailingMoeV2 BF16 Implementation
================================

Implementation of BailingMoeV2 (inclusionAI/Ling-mini-2.0) for the Neuron backend.
Adapted from the Qwen3 dense template (attention, dense MLP, backbone) and the
GPT-OSS BF16 template (expert parallelism of the routed experts).

Supported parallelism: TP and SP for attention and the dense MLP. TP or EP for
the routed experts. The shared expert is replicated on each rank.

Architecture of the reference model:
  - 20 decoder layers. Layer 0 has a dense SiLU MLP (intermediate 5120).
    Layers 1 to 19 have a MoE block: 256 routed experts, top-8, intermediate
    512, and one shared expert (intermediate 512).
  - GQA attention: 16 Q heads, 4 KV heads, head_dim 128, no bias.
  - Per-head QK RMSNorm before RoPE.
  - Partial RoPE: the first 64 of the 128 head dims rotate (rotate_half pairs
    i and i + 32). The other 64 dims pass through.
  - Sigmoid router in FP32 with an expert bias that changes only the selection.
    Group-limited top-k: 8 groups, keep 4 groups, then top-8 experts.
    The weights are the sigmoid scores of the selected experts, normalized to a
    sum of 1 and multiplied by routed_scaling_factor (2.5).

Partial RoPE on the kernels:
  The prefill QKV kernel and the decode attention kernel apply RoPE with the
  rotate_half rule over the full head_dim. The loader permutes the dims of
  each Q head and each K head (and the QK-norm gammas) so that the rotary dims
  are at [0:32] and [64:96] and the pass-through dims are at [32:64] and
  [96:128]. The rotary tables put cos=1 and sin=0 on the pass-through dims.
  RMSNorm and the dot product q.k do not change under the same permutation of
  q and k, and V is not permuted. Thus the attention output is equal to the
  HF output. The KV cache keeps K in the permuted layout for prefill and decode.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    BailingMoeV2-specific. Change when porting.
"""

import logging
from dataclasses import dataclass

import nki.language as nl
import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group
from transformers import PretrainedConfig

import vllm_neuron.functional as NF
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    NormType,
)
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX
from vllm_neuron.utils.neuron_utils import can_run_kernel
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    expert_parallel_grouped_loader,
    set_weight_loader,
    sharding_weight_loader,
)

import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import (
    build_per_head_block_table,
    paged_slot_indices,
    scatter_new_kv,
)
from .config import BailingMoeV2Config

logger = logging.getLogger(__name__)

# Tokens in each block of the blockwise prefill MoE kernel.
MOE_PREFILL_BLOCK_SIZE = 256

# Parameters that stay in FP32 after load. The HF router computes in FP32.
FP32_PARAM_SUFFIXES = ("mlp.gate.weight", "mlp.gate.expert_bias")


# =============================================================================
# Section 1: RMS Normalization
# <-- MODEL-SPECIFIC: HF BailingMoeV2RMSNorm casts the normalized value to the
#     input dtype before it multiplies by the weight.
# =============================================================================


class BailingMoeV2RMSNorm(nn.Module):
    """RMS normalization with the HF BailingMoeV2 cast order."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# =============================================================================
# Section 2: Partial Rotary Position Embedding
# <-- MODEL-SPECIFIC: rotary_dim = partial_rotary_factor * head_dim (64 of 128).
# =============================================================================


def rope_head_permutation(head_dim: int, rotary_dim: int) -> torch.Tensor:
    """Return the per-head dim permutation that the loader applies to Q and K.

    The permuted head is ``new = old[perm]``. The first half of the new head
    holds the first half of the rotary dims and the first half of the
    pass-through dims. The second half holds the other halves. Then the
    rotate_half rule over the full head pairs rotary dim ``i`` with rotary dim
    ``i + rotary_dim // 2``, which is the HF partial RoPE rule. When
    ``rotary_dim == head_dim`` the permutation is the identity.

    Args:
        head_dim: The size of one attention head.
        rotary_dim: The number of head dims that RoPE rotates. It must be even
            and not larger than ``head_dim``.

    Returns:
        A ``[head_dim]`` int64 tensor on CPU. The result is on CPU also in a
        ``torch.device("meta")`` context.

    Example:
        >>> rope_head_permutation(8, 4).tolist()
        [0, 1, 4, 5, 2, 3, 6, 7]
    """
    if rotary_dim % 2 != 0 or rotary_dim > head_dim or head_dim % 2 != 0:
        raise ValueError(
            f"rotary_dim ({rotary_dim}) must be even and not larger than "
            f"head_dim ({head_dim}). head_dim must be even."
        )
    r = rotary_dim // 2
    half = head_dim // 2
    # The runner builds the model in a torch.device("meta") context. Thus give
    # the CPU device explicitly. A CPU weight that a meta index selects is
    # memory with no values, and torch gives no error.
    order = (
        list(range(0, r))
        + list(range(rotary_dim, rotary_dim + half - r))
        + list(range(r, rotary_dim))
        + list(range(rotary_dim + half - r, head_dim))
    )
    return torch.tensor(order, dtype=torch.int64, device="cpu")


def _cpu_permutation(perm: torch.Tensor) -> torch.Tensor:
    """Return ``perm`` as a CPU int64 tensor. Refuse a meta tensor.

    Example:
        >>> _cpu_permutation(rope_head_permutation(8, 4)).device.type
        'cpu'
    """
    if perm.is_meta:
        raise ValueError(
            "The RoPE head permutation is a meta tensor, so the loaded Q and K "
            "weights have no values. Build it with rope_head_permutation."
        )
    return perm.to(device="cpu", dtype=torch.int64)


class BailingMoeV2RotaryEmbedding(nn.Module):
    """Partial RoPE tables in the permuted head layout.

    <-- MODEL-SPECIFIC: inv_freq uses rotary_dim, not head_dim, as in HF.
    The pass-through dims get cos=1 and sin=0.
    """

    def __init__(self, config: BailingMoeV2Config):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim

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
        inv_freq = self._compute_inv_freq(device)  # [R/2]
        inv_freq_expanded = inv_freq[:, None].float()  # [R/2, 1]
        positions_expanded = position_ids[None, :].float()  # [1, T]
        freqs = (inv_freq_expanded @ positions_expanded).transpose(0, 1)  # [T, R/2]
        num_pass = self.head_dim // 2 - self.rotary_dim // 2
        cos_half = freqs.cos()
        sin_half = freqs.sin()
        if num_pass > 0:
            ones = torch.ones(
                freqs.shape[0], num_pass, dtype=freqs.dtype, device=freqs.device
            )
            cos_half = torch.cat([cos_half, ones], dim=-1)
            sin_half = torch.cat([sin_half, torch.zeros_like(ones)], dim=-1)
        cos = torch.cat([cos_half, cos_half], dim=-1)  # [T, Hd]
        sin = torch.cat([sin_half, sin_half], dim=-1)  # [T, Hd]
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


# =============================================================================
# Section 3: Weight loaders
# <-- MODEL-SPECIFIC: fused query_key_value tensor, per-head RoPE permutation,
#     per-expert checkpoint tensors.
# =============================================================================


def bailing_qkv_weight_loader(
    num_q_heads: int,
    num_kv_heads: int,
    num_q_heads_per_rank: int,
    num_kv_heads_per_rank: int,
    num_kv_replicas: int,
    head_dim: int,
    perm: torch.Tensor,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Load the fused HF ``query_key_value`` weight into the ``[H, qkv]`` layout.

    The HF tensor is ``[(Nq + 2 * Nkv) * Dh, H]`` with the Q heads first, then
    the K heads, then the V heads. The loader takes the heads of this rank,
    applies ``perm`` to each Q head and each K head, and transposes the result.

    Args:
        num_q_heads: The total number of Q heads.
        num_kv_heads: The total number of KV heads.
        num_q_heads_per_rank: The number of Q heads on one rank.
        num_kv_heads_per_rank: The number of KV heads on one rank.
        num_kv_replicas: The number of ranks that share one KV head.
        head_dim: The size of one head.
        perm: The ``[head_dim]`` permutation from :func:`rope_head_permutation`.
        num_shards: The tensor-parallel degree.

    Returns:
        A SafetensorsWeightLoader for the ``qkv_proj_weight`` parameter.

    Example:
        >>> perm = rope_head_permutation(128, 64)
        >>> loader = bailing_qkv_weight_loader(16, 4, 2, 1, 2, 128, perm, 8)
    """
    perm = _cpu_permutation(perm)

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1, "bailing_qkv_weight_loader expects one fused tensor"
        slice_obj = slices[0]
        local_rank = rank % num_shards
        kv_rank = local_rank // num_kv_replicas

        q_start = local_rank * num_q_heads_per_rank * head_dim
        q_end = q_start + num_q_heads_per_rank * head_dim
        k_base = num_q_heads * head_dim
        k_start = k_base + kv_rank * num_kv_heads_per_rank * head_dim
        k_end = k_start + num_kv_heads_per_rank * head_dim
        v_base = (num_q_heads + num_kv_heads) * head_dim
        v_start = v_base + kv_rank * num_kv_heads_per_rank * head_dim
        v_end = v_start + num_kv_heads_per_rank * head_dim

        q = slice_obj[q_start:q_end, :]
        k = slice_obj[k_start:k_end, :]
        v = slice_obj[v_start:v_end, :]
        hidden = q.shape[-1]
        q = q.reshape(num_q_heads_per_rank, head_dim, hidden)[:, perm, :]
        k = k.reshape(num_kv_heads_per_rank, head_dim, hidden)[:, perm, :]
        q = q.reshape(-1, hidden)
        k = k.reshape(-1, hidden)
        return torch.cat([q, k, v], dim=0).T

    return SafetensorsWeightLoader(transform=transform)


def rope_permuted_vector_loader(perm: torch.Tensor) -> SafetensorsWeightLoader:
    """Load a ``[head_dim]`` QK-norm gamma and apply the RoPE head permutation.

    Example:
        >>> loader = rope_permuted_vector_loader(rope_head_permutation(128, 64))
    """
    perm = _cpu_permutation(perm)

    def transform(slices: list, rank: int) -> torch.Tensor:
        return slices[0][:][perm]

    return SafetensorsWeightLoader(transform=transform)


def expert_gate_up_loader(
    shard_size: int, num_shards: int, num_local_experts: int
) -> SafetensorsWeightLoader:
    """Fuse per-expert HF gate and up weights into ``[E_L, H, 2 * I_shard]``.

    ``slices`` is ``[gate_0 .. gate_{E_L-1}, up_0 .. up_{E_L-1}]``. Each HF tensor
    is ``[I, H]``. The loader takes the intermediate shard of this rank, transposes
    each tensor, and puts gate before up on the last dim. The model then reads the
    parameter as ``[E_L, H, 2, I_shard]`` with gate at index 0.

    Example:
        >>> loader = expert_gate_up_loader(shard_size=512, num_shards=1, num_local_experts=32)
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 2 * num_local_experts, (
            f"expert_gate_up_loader expects {2 * num_local_experts} slices, "
            f"got {len(slices)}"
        )
        start = (rank % num_shards) * shard_size
        end = start + shard_size
        experts = []
        for e in range(num_local_experts):
            gate = slices[e][start:end, :].T
            up = slices[num_local_experts + e][start:end, :].T
            experts.append(torch.cat([gate, up], dim=-1))
        return torch.stack(experts, dim=0)

    return SafetensorsWeightLoader(transform=transform)


def expert_down_loader(
    shard_size: int, num_shards: int, num_local_experts: int
) -> SafetensorsWeightLoader:
    """Stack per-expert HF down weights into ``[E_L, I_shard, H]``.

    Each HF tensor is ``[H, I]``. The loader takes the intermediate shard of this
    rank and transposes it.

    Example:
        >>> loader = expert_down_loader(shard_size=512, num_shards=1, num_local_experts=32)
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == num_local_experts, (
            f"expert_down_loader expects {num_local_experts} slices, got {len(slices)}"
        )
        start = (rank % num_shards) * shard_size
        end = start + shard_size
        return torch.stack([s[:, start:end].T for s in slices], dim=0)

    return SafetensorsWeightLoader(transform=transform)


# =============================================================================
# Section 4: Attention
# Mixed: PARALLELISM (TP head sharding, SP, collectives) +
#        MODEL-SPECIFIC (GQA, QK-norm before partial RoPE, no bias)
# =============================================================================


class BailingMoeV2Attention(nn.Module):
    """Multi-head attention with TP head sharding, QK-norm and partial RoPE.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads sharded across TP ranks
    - KV heads replicated when fewer than TP size (GQA)
    - Prefill: all-gather -> QKV kernel -> attention kernel -> O proj -> reduce-scatter
    - Decode: fused attention decode kernel -> all-reduce

    <-- MODEL-SPECIFIC:
    - 16 Q heads / 4 KV heads, head_dim 128
    - Per-head QK-norm before RoPE
    - Partial RoPE through the load-time head permutation
    """

    def __init__(self, config: BailingMoeV2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
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

        # <-- MODEL-SPECIFIC: Per-head QK-norm (HF names query_layernorm / key_layernorm)
        self.query_layernorm = BailingMoeV2RMSNorm(
            self.head_dim, config.rms_norm_eps, self.dtype
        )
        self.key_layernorm = BailingMoeV2RMSNorm(
            self.head_dim, config.rms_norm_eps, self.dtype
        )

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
        <-- MODEL-SPECIFIC: Fused HF query_key_value, RoPE head permutation.
        """
        perm = rope_head_permutation(self.head_dim, self.rotary_dim)
        set_weight_loader(
            self.qkv_proj_weight,
            bailing_qkv_weight_loader(
                num_q_heads=self.num_attention_heads,
                num_kv_heads=self.num_key_value_heads,
                num_q_heads_per_rank=self.num_attention_heads_per_rank,
                num_kv_heads_per_rank=self.num_key_value_heads_per_rank,
                num_kv_replicas=self.num_kv_replicas,
                head_dim=self.head_dim,
                perm=perm,
                num_shards=self.world_size,
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
        set_weight_loader(
            self.query_layernorm.weight, rope_permuted_vector_loader(perm)
        )
        set_weight_loader(self.key_layernorm.weight, rope_permuted_vector_loader(perm))

    # -- Forward dispatch --------------------------------------------------

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

    # -- Prefill path ------------------------------------------------------

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
            qk_norm_pre_rope_eps=self.query_layernorm.variance_epsilon,
            qk_norm_pre_rope_q_gamma=self.query_layernorm.weight.view(1, -1),
            qk_norm_pre_rope_k_gamma=self.key_layernorm.weight.view(1, -1),
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

        # Step 3: Attention. No sinks, no sliding window.
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

        # Step 4: Output projection
        attn_output = attn_output.unsqueeze(0)
        attn_output = NF.o_proj(attn_output, self.o_proj_weight, None)
        attn_output = attn_output.squeeze(0)

        # >>> PARALLELISM: Reduce-scatter to SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    # -- Decode path -------------------------------------------------------

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
    ):
        """>>> PARALLELISM: Fused attention decode kernel with TP. <<<
        <-- MODEL-SPECIFIC: QK-norm through rmsnorm_QK_pre_rope_enabled=True.
        The cos/sin tables already hold the partial RoPE values.
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

        # Positions are necessary to make the causal attention mask.
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

        common_kwargs = dict(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            rmsnorm_X_enabled=False,
            W_qkv=self.qkv_proj_weight,
            bias_qkv=None,
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=self.query_layernorm.variance_epsilon,
            rmsnorm_QK_pre_rope_W_Q=self.query_layernorm.weight.view(1, -1),
            rmsnorm_QK_pre_rope_W_K=self.key_layernorm.weight.view(1, -1),
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
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

        # GQA (nkh > 1) reaches the fused decode kernel only with a per-head 3D
        # block table and the 4D cache.
        if nkh > 1:
            active_blocks_table = build_per_head_block_table(block_table, nkh).to(
                torch.int32
            )
            output, K_new, V_new = NF.attention_decode(
                active_blocks_table=active_blocks_table,
                K_cache=self.k_cache,
                V_cache=self.v_cache,
                **common_kwargs,
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
        # 3D [num_blocks, block_len, d_head] cache and 2D [B, num_blocks] table.
        k_cache = self.k_cache.squeeze(1) if self.k_cache.dim() == 4 else self.k_cache
        v_cache = self.v_cache.squeeze(1) if self.v_cache.dim() == 4 else self.v_cache
        active_blocks_table = block_table.to(torch.int32)

        output, K_new, V_new = NF.attention_decode(
            active_blocks_table=active_blocks_table,
            K_cache=k_cache,
            V_cache=v_cache,
            **common_kwargs,
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
# Section 5: Dense MLP (layer 0) and replicated shared expert
# =============================================================================


class BailingMoeV2MLP(nn.Module):
    """Dense SiLU MLP with TP intermediate sharding (layer 0).

    >>> PARALLELISM: TP <<<
    - Prefill: all-gather -> NF.mlp -> reduce-scatter (SP)
    - Decode: NF.mlp -> all-reduce
    """

    def __init__(self, config: BailingMoeV2Config, intermediate_size: int):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.hidden_size = config.hidden_size
        self.intermediate_size_per_rank = intermediate_size // self.world_size

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
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """>>> PARALLELISM: SP all-gather/reduce-scatter for prefill, all-reduce for decode. <<<"""
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        )

        # Add the TP partial outputs in FP32, then round one time to BF16. A BF16
        # collective rounds after each partial sum. With 8 partial outputs, the
        # error of the CPU collective is 2.6 times the BF16 reference error.
        if self.world_size > 1:
            dtype = output.dtype
            output = output.to(torch.float32)
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                # Use the return value. The collective does not always write
                # the sum in place.
                output = self.tp_group.all_reduce(output)
            output = output.to(dtype)

        return output


class BailingMoeV2SharedExpert(nn.Module):
    """Shared expert, replicated on each rank with the full intermediate size.

    >>> PARALLELISM: No collective. <<<
    The intermediate size is 512. A TP shard of 64 is too small for the MLP
    kernel tiles, so each rank keeps the full weights. In prefill the expert
    runs on the SP token shard of this rank. In decode it runs on the
    replicated tokens.
    """

    def __init__(self, config: BailingMoeV2Config, intermediate_size: int):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.gate_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, intermediate_size, dtype=config.torch_dtype)
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, intermediate_size, dtype=config.torch_dtype)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(intermediate_size, config.hidden_size, dtype=config.torch_dtype)
        )

        gate_up_loader = sharding_weight_loader(
            shard_dim=1,
            shard_size=intermediate_size,
            num_shards=1,
            is_storage_transposed=True,
        )
        down_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=intermediate_size,
            num_shards=1,
            is_storage_transposed=True,
        )
        set_weight_loader(self.gate_proj_weight, gate_up_loader)
        set_weight_loader(self.up_proj_weight, gate_up_loader)
        set_weight_loader(self.down_proj_weight, down_loader)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        )


# =============================================================================
# Section 6: Router and MoE block
# Mixed: PARALLELISM (EP expert placement, TP/EP collectives) +
#        MODEL-SPECIFIC (sigmoid group-limited routing, shared expert)
# =============================================================================


def bailing_route(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    expert_bias: torch.Tensor,
    n_group: int,
    topk_group: int,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the BailingMoeV2 sigmoid group-limited top-k routing in FP32.

    The expert bias changes only the selection. The weights come from the sigmoid
    scores without the bias.

    Args:
        hidden_states: ``[T, H]`` input of the MoE block (after the norm).
        gate_weight: ``[E, H]`` router weight.
        expert_bias: ``[E]`` selection bias.
        n_group: The number of expert groups.
        topk_group: The number of groups that each token keeps.
        top_k: The number of experts for each token.
        routed_scaling_factor: The multiplier of the routing weights.
        norm_topk_prob: If True, normalize the selected scores to a sum of 1.

    Returns:
        A tuple ``(topk_idx, topk_weight, dense_affinities)``. ``topk_idx`` is
        ``[T, top_k]`` int64. ``topk_weight`` is ``[T, top_k]`` FP32.
        ``dense_affinities`` is ``[T, E]`` FP32 with zeros for the experts that
        the token does not select.

    Example:
        >>> h = torch.randn(4, 16)
        >>> w = torch.randn(8, 16)
        >>> idx, wt, dense = bailing_route(h, w, torch.zeros(8), 2, 1, 2, 2.5)
        >>> dense.shape
        torch.Size([4, 8])
    """
    num_tokens = hidden_states.shape[0]
    num_experts = gate_weight.shape[0]
    logits = torch.nn.functional.linear(
        hidden_states.to(torch.float32), gate_weight.to(torch.float32)
    )
    scores = torch.sigmoid(logits)
    scores_for_routing = scores + expert_bias.to(torch.float32)

    group_scores = (
        scores_for_routing.view(num_tokens, n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, n_group, num_experts // n_group)
        .reshape(num_tokens, -1)
    )
    masked_scores = scores_for_routing.masked_fill(~score_mask.bool(), float("-inf"))
    _, topk_idx = torch.topk(masked_scores, k=top_k, dim=-1)

    topk_weight = torch.gather(scores, dim=1, index=topk_idx)
    if top_k > 1 and norm_topk_prob:
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weight = topk_weight * routed_scaling_factor

    dense_affinities = torch.zeros_like(scores).scatter_(1, topk_idx, topk_weight)
    return topk_idx, topk_weight, dense_affinities


@dataclass
class MoEParallelism:
    """Expert placement of one rank.

    ``ep_degree`` ranks hold disjoint expert ranges. ``tp_degree`` ranks share
    one expert range and shard its intermediate dim. ``ep_tp_group`` is the
    group of these ``tp_degree`` ranks.
    """

    ep_enabled: bool
    ep_degree: int
    ep_rank: int
    tp_degree: int
    ep_tp_group: object


def resolve_moe_parallelism() -> MoEParallelism:
    """Read the expert placement of this rank from the vLLM and Neuron state.

    Without ``--enable-expert-parallel`` each rank holds all experts and the
    TP group shards the intermediate dim. With it, the Neuron EP state gives
    the expert range of this rank.

    Returns:
        The MoEParallelism of this rank.

    Example:
        >>> placement = resolve_moe_parallelism()  # in a worker after init
        >>> placement.ep_degree
        8
    """
    from vllm.config import get_current_vllm_config

    parallel_config = get_current_vllm_config().parallel_config
    tp_group = get_tp_group()
    if not parallel_config.enable_expert_parallel:
        return MoEParallelism(
            ep_enabled=False,
            ep_degree=1,
            ep_rank=0,
            tp_degree=tp_group.world_size,
            ep_tp_group=tp_group,
        )

    if parallel_config.data_parallel_size > 1:
        raise ValueError(
            "BailingMoeV2 expert parallelism across data-parallel replicas is not "
            f"supported (data_parallel_size={parallel_config.data_parallel_size}). "
            "Set data_parallel_size=1, or disable --enable-expert-parallel."
        )

    from vllm_neuron.parallel.neuron_parallel_state import (
        get_neuron_ep_degree,
        get_neuron_ep_rank,
        get_neuron_ep_tp_group,
    )

    ep_tp_group = get_neuron_ep_tp_group()
    return MoEParallelism(
        ep_enabled=True,
        ep_degree=get_neuron_ep_degree(),
        ep_rank=get_neuron_ep_rank(),
        tp_degree=ep_tp_group.world_size,
        ep_tp_group=ep_tp_group,
    )


def _can_use_nki_moe_tkg(hidden_states: torch.Tensor) -> bool:
    """Return True when the decode MoE NKI kernel (``NF.moe_tkg``) can run.

    Example:
        >>> _can_use_nki_moe_tkg(torch.zeros(4, 2048, dtype=torch.bfloat16))
        False
    """
    if not can_run_kernel(hidden_states):
        return False
    return hidden_states.dtype == torch.bfloat16


class BailingMoeV2Gate(nn.Module):
    """FP32 router weight and expert bias. HF names: ``mlp.gate.*``."""

    def __init__(self, config: BailingMoeV2Config):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_size, dtype=torch.float32)
        )
        self.expert_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32)
        )


class BailingMoeV2SparseMoeBlock(nn.Module):
    """Routed experts plus one shared expert.

    >>> PARALLELISM: TP + EP <<<
    - EP (``--enable-expert-parallel``, ep_degree = TP): rank k holds experts
      [k * E_L, (k + 1) * E_L) with the full intermediate size.
    - No EP: each rank holds all experts with an intermediate shard.
    - Prefill: router on the SP shard -> all-gather -> blockwise MoE kernel ->
      reduce-scatter. The shared expert runs on the SP shard.
    - Decode: router on the replicated tokens -> MoE decode kernel ->
      all-reduce. Then add the shared expert, which runs on each rank.

    <-- MODEL-SPECIFIC: sigmoid group-limited routing, SiLU experts, POST_SCALE.
    """

    def __init__(self, config: BailingMoeV2Config):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        placement = resolve_moe_parallelism()
        self.ep_enabled = placement.ep_enabled
        self.ep_degree = placement.ep_degree
        self.ep_rank = placement.ep_rank
        self.tp_degree = placement.tp_degree
        self.ep_tp_group = placement.ep_tp_group
        # moe_group combines the expert outputs of all ranks.
        self.moe_group = self.tp_group

        self.hidden_size = config.hidden_size
        self.total_num_experts = config.num_experts
        if self.total_num_experts % self.ep_degree != 0:
            raise ValueError(
                f"num_experts ({self.total_num_experts}) must divide by "
                f"ep_degree ({self.ep_degree})."
            )
        self.num_local_experts = self.total_num_experts // self.ep_degree
        self.num_experts_per_token = config.num_experts_per_tok
        self.intermediate_size = config.moe_intermediate_size
        if self.intermediate_size % self.tp_degree != 0:
            raise ValueError(
                f"moe_intermediate_size ({self.intermediate_size}) must divide by "
                f"the expert tensor-parallel degree ({self.tp_degree})."
            )
        self.intermediate_size_per_rank = self.intermediate_size // self.tp_degree
        self.block_size = MOE_PREFILL_BLOCK_SIZE

        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob

        if self.intermediate_size_per_rank < 128:
            logger.warning(
                "BailingMoeV2 expert intermediate shard is %d. The prefill MoE "
                "kernel needs 128 or more, so prefill uses the torch fallback. "
                "Set --enable-expert-parallel to keep the full intermediate size.",
                self.intermediate_size_per_rank,
            )

        self.gate = BailingMoeV2Gate(config)

        self.gate_up_proj_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_size,
                2 * self.intermediate_size_per_rank,
                dtype=config.torch_dtype,
            )
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.intermediate_size_per_rank,
                self.hidden_size,
                dtype=config.torch_dtype,
            )
        )

        self.shared_experts = None
        if config.num_shared_experts:
            self.shared_experts = BailingMoeV2SharedExpert(
                config, config.moe_intermediate_size * config.num_shared_experts
            )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """>>> PARALLELISM: EP filter of the experts, TP shard of the intermediate dim. <<<"""
        local_expert_indices = list(
            range(
                self.ep_rank * self.num_local_experts,
                (self.ep_rank + 1) * self.num_local_experts,
            )
        )
        set_weight_loader(
            self.gate_up_proj_weight,
            expert_parallel_grouped_loader(
                local_expert_indices,
                expert_gate_up_loader(
                    shard_size=self.intermediate_size_per_rank,
                    num_shards=self.tp_degree,
                    num_local_experts=self.num_local_experts,
                ),
                total_num_experts=self.total_num_experts,
            ),
        )
        set_weight_loader(
            self.down_proj_weight,
            expert_parallel_grouped_loader(
                local_expert_indices,
                expert_down_loader(
                    shard_size=self.intermediate_size_per_rank,
                    num_shards=self.tp_degree,
                    num_local_experts=self.num_local_experts,
                ),
                total_num_experts=self.total_num_experts,
            ),
        )

    def _gate_up_4d(self) -> torch.Tensor:
        return self.gate_up_proj_weight.reshape(
            self.num_local_experts,
            self.hidden_size,
            2,
            self.intermediate_size_per_rank,
        )

    def _ep_rank_tensor(self, device: torch.device) -> torch.Tensor:
        """Return the EP rank as a ``[1]`` int32 graph input (SPMD-safe)."""
        if self.ep_degree <= 1:
            return torch.zeros((1,), dtype=torch.int32, device=device)
        from vllm_neuron.parallel.neuron_parallel_state import get_neuron_ep_group

        return get_neuron_ep_group().rank_in_group_tensor.reshape(1).to(torch.int32)

    def _local_affinities(self, dense_affinities: torch.Tensor) -> torch.Tensor:
        """Select the ``[T, E_L]`` affinities of the experts on this rank."""
        if self.ep_degree <= 1:
            return dense_affinities
        local_expert_indices = (
            torch.arange(
                self.num_local_experts,
                device=dense_affinities.device,
                dtype=torch.int32,
            )
            + self._ep_rank_tensor(dense_affinities.device) * self.num_local_experts
        )
        return NF.get_local_expert_affinities(dense_affinities, local_expert_indices)

    def _route(self, hidden_states: torch.Tensor):
        return bailing_route(
            hidden_states,
            self.gate.weight,
            self.gate.expert_bias,
            n_group=self.n_group,
            topk_group=self.topk_group,
            top_k=self.num_experts_per_token,
            routed_scaling_factor=self.routed_scaling_factor,
            norm_topk_prob=self.norm_topk_prob,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        is_prefill: bool,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if is_prefill:
            return self.forward_prefill(hidden_states, positions)
        return self.forward_decode(hidden_states)

    def forward_prefill(
        self, hidden_states: torch.Tensor, positions: torch.Tensor | None
    ) -> torch.Tensor:
        """Prefill: blockwise MoE kernel (``NF.moe_cte``) on the full sequence."""
        shared_output = (
            self.shared_experts(hidden_states)
            if self.shared_experts is not None
            else None
        )

        # Router on the SP shard: [T / world_size, E] in FP32
        _, _, expert_affinities = self._route(hidden_states)

        # >>> PARALLELISM: All-gather from SP for the full sequence <<<
        if self.world_size > 1:
            expert_affinities = self.tp_group.all_gather(expert_affinities, dim=0)
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        # Padding mask from positions (True = real token, False = padding)
        padding_mask = None
        if positions is not None:
            last_real_idx = torch.argmax(positions)
            token_indices = torch.arange(positions.shape[0], device=positions.device)
            padding_mask = token_indices <= last_real_idx

        # >>> PARALLELISM: With EP, keep the affinities of the local experts <<<
        expert_affinities = self._local_affinities(expert_affinities)

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
            gate_up_proj_weight=self._gate_up_4d(),
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
        output = output.to(hidden_states.dtype)

        # Set the pad rows to zero. The pad tokens keep their experts in the
        # blockwise mapping, and the kernel multiplies each expert output by
        # the masked affinity 0. A non-finite expert output then gives NaN,
        # because 0 x inf is NaN. A select does not multiply, so a pad row
        # cannot carry NaN into the next layer. The real rows do not change.
        if padding_mask is not None:
            output = torch.where(
                padding_mask.unsqueeze(1), output, torch.zeros_like(output)
            )

        # >>> PARALLELISM: Combine expert results and return to SP layout <<<
        if self.moe_group.world_size > 1:
            output = self.moe_group.reduce_scatter(output, dim=0)

        if shared_output is not None:
            output = output + shared_output
        return output

    def _torch_decode_experts(
        self, hidden_states: torch.Tensor, local_affinities: torch.Tensor
    ) -> torch.Tensor:
        """Torch decode experts for CPU mode. The serving geometry uses NF.moe_tkg."""
        gate_up = self._gate_up_4d()
        gate = torch.einsum("th,ehi->eti", hidden_states, gate_up[:, :, 0, :])
        up = torch.einsum("th,ehi->eti", hidden_states, gate_up[:, :, 1, :])
        intermediate = torch.nn.functional.silu(gate) * up
        expert_out = torch.einsum("eti,eih->eth", intermediate, self.down_proj_weight)
        weights = local_affinities.to(torch.float32).T.unsqueeze(-1)  # [E_L, T, 1]
        output = (expert_out.to(torch.float32) * weights).sum(dim=0)
        return output.to(hidden_states.dtype)

    def forward_decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Decode: MoE decode kernel (``NF.moe_tkg``) on the replicated tokens."""
        shared_output = (
            self.shared_experts(hidden_states)
            if self.shared_experts is not None
            else None
        )

        topk_idx, _, expert_affinities = self._route(hidden_states)

        if _can_use_nki_moe_tkg(hidden_states):
            # All-expert mode: each rank runs its E_L local experts on all tokens.
            # The kernel slices the [T, E] affinities to this rank with rank_id.
            output = NF.moe_tkg(
                hidden_input=hidden_states,
                expert_gate_up_weights=self._gate_up_4d(),
                expert_down_weights=self.down_proj_weight,
                expert_affinities=expert_affinities.to(hidden_states.dtype),
                expert_index=topk_idx.to(torch.int32),
                is_all_expert=True,
                rank_id=self._ep_rank_tensor(hidden_states.device).reshape(1, 1),
                expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
                activation_fn=ActFnType.SiLU,
            )
        else:
            output = self._torch_decode_experts(
                hidden_states, self._local_affinities(expert_affinities)
            )

        # >>> PARALLELISM: Sum the expert outputs of all ranks <<<
        if self.moe_group.world_size > 1:
            output = self.moe_group.all_reduce(output)

        if shared_output is not None:
            output = output + shared_output
        return output


# =============================================================================
# Section 7: Decoder Layer
# =============================================================================


class BailingMoeV2DecoderLayer(nn.Module):
    """Single BailingMoeV2 decoder layer.

    Architecture: hidden -> RMSNorm -> Attention -> residual -> RMSNorm -> MLP or
    MoE -> residual
    """

    def __init__(self, config: BailingMoeV2Config, layer_idx: int):
        super().__init__()
        self.input_layernorm = BailingMoeV2RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = BailingMoeV2RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.attention = BailingMoeV2Attention(config, layer_idx=layer_idx)
        self.is_moe = config.is_moe_layer(layer_idx)
        if self.is_moe:
            self.mlp = BailingMoeV2SparseMoeBlock(config)
        else:
            self.mlp = BailingMoeV2MLP(config, config.intermediate_size)
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
        hidden_states = self.attention(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(
            hidden_states, is_prefill=not is_decode, positions=positions
        )
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Section 8: Model Backbone
# =============================================================================


class BailingMoeV2Model(nn.Module):
    """BailingMoeV2 transformer backbone.

    >>> PARALLELISM: SP during prefill: embed -> shard -> layers -> all-gather. <<<
    """

    def __init__(self, config: BailingMoeV2Config):
        super().__init__()
        self.config = config

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: Vocab-sharded embedding (HF name word_embeddings) <<<
        self.word_embeddings = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )

        self.layers = nn.ModuleList(
            [
                BailingMoeV2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = BailingMoeV2RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = BailingMoeV2RotaryEmbedding(config)

        set_weight_loader(
            self.word_embeddings.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.word_embeddings.vocab_size_per_rank,
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
    ) -> torch.Tensor:
        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        # >>> PARALLELISM: VocabDimShardedEmbedding handles SP internally <<<
        hidden_states = self.word_embeddings(
            input_ids, scatter_tokens=is_prefill, rank=rank
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

        # >>> PARALLELISM: SP all-gather to the full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states


# =============================================================================
# Section 9: LM Head + Weight Loading
# =============================================================================


class BailingMoeV2ForCausalLM(nn.Module):
    """BailingMoeV2 model with LM head.

    >>> PARALLELISM: Column-parallel LM head. <<<
    <-- MODEL-SPECIFIC: Untied embeddings (separate lm_head.weight). Text only.
    """

    def __init__(self, config: BailingMoeV2Config):
        super().__init__()
        self.config = config
        self.model = BailingMoeV2Model(config)

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

        if config.vocab_size % self.world_size != 0:
            raise ValueError(
                f"vocab_size ({config.vocab_size}) must divide by the "
                f"tensor-parallel degree ({self.world_size})."
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
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if inputs_embeds is not None:
            raise ValueError(
                "BailingMoeV2 is text only. Input embeddings are not supported. "
                "Send token IDs."
            )

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

        hidden_states = self.model(
            input_ids, positions, attn_metadata=attn_metadata, rank=rank
        )

        # One-token decode is row-aligned by construction. Skip the gather.
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
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = BailingMoeV2Config.from_configs(hf_config, neuron_config)
        return cls(config)

    # -- KV Cache ----------------------------------------------------------

    def get_kv_spec(self):
        layers = []
        for i, layer in enumerate(self.model.layers):
            layers.append(
                LayerSpec(
                    name=f"layers.{i}.self_attn",
                    num_kv_heads=layer.attention.num_key_value_heads_per_rank,
                    head_size=layer.attention.head_dim,
                    dtype=layer.attention.dtype,
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
            layer.attention.k_cache = kv_caches[layer_name][0]
            layer.attention.v_cache = kv_caches[layer_name][1]

    # -- Weight Loading ----------------------------------------------------

    def get_weight_mappings(self) -> dict[str, str | list[str]]:
        """Return the map from model parameter names to HF checkpoint keys.

        Example:
            >>> mappings = model.get_weight_mappings()
            >>> mappings["model.layers.0.attention.qkv_proj_weight"]
            'model.layers.0.attention.query_key_value.weight'
        """
        mappings = {
            "model.word_embeddings.weight": "model.word_embeddings.weight",
            "lm_head.weight": "lm_head.weight",
            "model.norm.weight": "model.norm.weight",
        }

        for layer_id, layer in enumerate(self.model.layers):
            prefix = f"model.layers.{layer_id}"

            mappings[f"{prefix}.attention.qkv_proj_weight"] = (
                f"{prefix}.attention.query_key_value.weight"
            )
            mappings[f"{prefix}.attention.o_proj_weight"] = (
                f"{prefix}.attention.dense.weight"
            )
            mappings[f"{prefix}.attention.query_layernorm.weight"] = (
                f"{prefix}.attention.query_layernorm.weight"
            )
            mappings[f"{prefix}.attention.key_layernorm.weight"] = (
                f"{prefix}.attention.key_layernorm.weight"
            )
            mappings[f"{prefix}.input_layernorm.weight"] = (
                f"{prefix}.input_layernorm.weight"
            )
            mappings[f"{prefix}.post_attention_layernorm.weight"] = (
                f"{prefix}.post_attention_layernorm.weight"
            )

            if not layer.is_moe:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    mappings[f"{prefix}.mlp.{proj}_weight"] = (
                        f"{prefix}.mlp.{proj}.weight"
                    )
                continue

            num_experts = layer.mlp.total_num_experts
            mappings[f"{prefix}.mlp.gate.weight"] = f"{prefix}.mlp.gate.weight"
            mappings[f"{prefix}.mlp.gate.expert_bias"] = (
                f"{prefix}.mlp.gate.expert_bias"
            )
            mappings[f"{prefix}.mlp.gate_up_proj_weight"] = [
                f"{prefix}.mlp.experts.{e}.gate_proj.weight" for e in range(num_experts)
            ] + [f"{prefix}.mlp.experts.{e}.up_proj.weight" for e in range(num_experts)]
            mappings[f"{prefix}.mlp.down_proj_weight"] = [
                f"{prefix}.mlp.experts.{e}.down_proj.weight" for e in range(num_experts)
            ]
            if layer.mlp.shared_experts is not None:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    mappings[f"{prefix}.mlp.shared_experts.{proj}_weight"] = (
                        f"{prefix}.mlp.shared_experts.{proj}.weight"
                    )

        return mappings

    @staticmethod
    def cast_loaded_state(
        state_dict: dict[str, torch.Tensor], target_dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        """Cast loaded tensors to ``target_dtype``. Keep the router tensors in FP32.

        Example:
            >>> sd = {"model.layers.1.mlp.gate.weight": torch.zeros(2, dtype=torch.bfloat16)}
            >>> BailingMoeV2ForCausalLM.cast_loaded_state(sd, torch.bfloat16)[
            ...     "model.layers.1.mlp.gate.weight"].dtype
            torch.float32
        """
        for name, tensor in state_dict.items():
            dtype = (
                torch.float32 if name.endswith(FP32_PARAM_SUFFIXES) else target_dtype
            )
            if tensor.dtype != dtype:
                state_dict[name] = tensor.to(dtype)
        return state_dict

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load weights from the BailingMoeV2 HF checkpoint.

        >>> PARALLELISM: Weight loaders handle the TP and EP sharding. <<<
        """
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, self.get_weight_mappings(), device
        ).state_dict

        rank_sharded = self.cast_loaded_state(rank_sharded, self.config.torch_dtype)

        self._load_kv_cache_scales(checkpoint, device)
        self.load_state_dict(rank_sharded, strict=False, assign=True)

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Lightweight weight loading used during CPU compile.

        Loads only the KV cache scales, so that the compile-time constant
        ``softmax_scale = self.scaling / self.k_scale_float`` is correct.
        """
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        # The lite path does not index through load_sharded_pipelined. Index here,
        # or the scale lookups miss and use 1.0.
        checkpoint._ensure_indexed()
        self._load_kv_cache_scales(checkpoint, device)

    def _load_kv_cache_scales(
        self, checkpoint: SafetensorsCheckpoint, device: torch.device
    ):
        from vllm_neuron.utils.dtype_utils import QUANTIZED_KV_CACHE_DTYPES
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.cache_config.cache_dtype not in QUANTIZED_KV_CACHE_DTYPES:
            return

        for layer_id in range(len(self.model.layers)):
            attn = self.model.layers[layer_id].attention
            for scale_name in ("k_scale", "v_scale"):
                key = f"model.layers.{layer_id}.attention.{scale_name}"
                if key in checkpoint._tensor_name_to_file:
                    val = 1.0 / checkpoint._get_slice(key)[:].to(
                        dtype=torch.bfloat16, device=device
                    )
                else:
                    val = torch.ones(1, dtype=torch.bfloat16, device=device)
                setattr(attn, scale_name, val.reshape(1, 1))

            attn.k_scale_float = attn.k_scale.item()
            attn.v_scale_float = attn.v_scale.item()
