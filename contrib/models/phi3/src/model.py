# SPDX-License-Identifier: Apache-2.0
"""
Phi3 BF16 Implementation
========================

Implementation of Phi3 (for example Phi-4-mini-instruct) for the Neuron backend.
The structure follows the SmolLM3 and Llama3 BF16 models. The decode attention
follows Qwen3-VL, because it routes GQA with more than one KV head on each rank
to the NKI decode kernel.

Phi3 differs from Llama in these items:

- The checkpoint stores one fused ``qkv_proj`` weight and one fused
  ``gate_up_proj`` weight. The weight loaders split them for each TP rank.
- RoPE rotates only the first ``rotary_dim`` head dimensions (partial RoPE).
- RoPE uses LongRoPE: a short factor and a long factor for each frequency, and
  an attention factor that scales cos and sin.

PARTIAL ROPE LAYOUT:
  HF Phi3 rotates the pairs (i, i + rotary_dim / 2) for i < rotary_dim / 2 and
  does not change the dimensions from rotary_dim to head_dim. The NKI decode
  kernel rotates the pairs (j, j + head_dim / 2) of the full head. The weight
  loader permutes the Q and K output rows of each head to make these pairs the
  same (see ``rope_head_permutation``). Cos is 1 and sin is 0 on the
  dimensions that do not rotate. Q and K use the same permutation, so the
  Q.K scores do not change. V and O do not change.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    Phi3-specific. Change when porting.

PARALLELISM SHARDING:
  TP-only:  Attention heads sharded, MLP intermediate sharded, SP in prefill.
"""

import logging
import math

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
from vllm_neuron.model.kv_cache import (
    KVSpec,
    LayerSpec,
    build_per_head_block_table,
    build_per_head_slot_mapping,
    paged_slot_indices,
)
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.neuron_utils import can_run_kernel
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from transformers import PretrainedConfig
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.sampler import Sampler

import vllm_neuron.nn as neuron_nn
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding

from .config import Phi3Config

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1: RMS Normalization
# <-- MODEL-SPECIFIC: HF Phi3RMSNorm normalizes in FP32, casts to the input
# dtype, and then multiplies by the weight.
# =============================================================================


class Phi3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


# =============================================================================
# Section 2: Partial LongRoPE
# <-- MODEL-SPECIFIC: Phi3 partial rotary embedding with LongRoPE scaling.
# =============================================================================


def rope_head_permutation(head_dim: int, rotary_dim: int) -> list[int]:
    """Return the head-dimension permutation for the kernel RoPE layout.

    Entry ``i`` is the HF dimension that goes to position ``i``. The result puts
    the HF rotated pair (i, i + rotary_dim / 2) at (i, i + head_dim / 2). The
    dimensions that do not rotate fill the remaining positions in each half.

    Example::

        >>> rope_head_permutation(8, 4)
        [0, 1, 4, 5, 2, 3, 6, 7]
    """
    r = rotary_dim // 2
    p = (head_dim - rotary_dim) // 2
    return (
        list(range(0, r))
        + list(range(rotary_dim, rotary_dim + p))
        + list(range(r, rotary_dim))
        + list(range(rotary_dim + p, head_dim))
    )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply full-head rotate_half RoPE in the permuted head layout.

    Args:
        q: [Nh, T, Dh], k: [Nkv, T, Dh]
        cos: [T, Dh], sin: [T, Dh] in the permuted layout.
    """
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class Phi3RotaryEmbedding(nn.Module):
    """Partial RoPE with LongRoPE scaling.

    The output cos/sin have shape [T, head_dim] in the permuted layout. Each half
    holds ``rotary_dim / 2`` scaled cos (or sin) values and then
    ``(head_dim - rotary_dim) / 2`` entries of 1 (or 0).

    LongRoPE selection follows HF ``longrope_frequency_update``: the long factor
    applies when the largest position in the forward pass is not smaller than
    ``original_max_position_embeddings``. Otherwise the short factor applies.
    The attention factor scales the rotated part in both cases.
    """

    def __init__(self, config: Phi3Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.base = config.rope_theta
        self.rope_type = config.rope_type
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.short_factor = config.rope_short_factor
        self.long_factor = config.rope_long_factor

        if self.rope_type == "longrope":
            attention_factor = config.rope_attention_factor
            if attention_factor is None:
                factor = (
                    config.max_position_embeddings
                    / config.original_max_position_embeddings
                )
                attention_factor = (
                    1.0
                    if factor <= 1.0
                    else math.sqrt(
                        1
                        + math.log(factor)
                        / math.log(config.original_max_position_embeddings)
                    )
                )
            self.attention_factor = float(attention_factor)
        else:
            self.attention_factor = 1.0

        # Make the frequencies one time on the CPU and keep them as buffers, as
        # GPT-OSS does. The runner builds the model on the meta device, and the
        # explicit CPU device gives real values. The runner then moves the
        # buffers to the device. Thus the compiled graph has no tensor constant
        # that comes from a Python list.
        cpu = torch.device("cpu")
        if self.rope_type == "longrope":
            self.register_buffer(
                "short_inv_freq",
                self._inv_freq(self.short_factor, cpu),
                persistent=False,
            )
            self.register_buffer(
                "long_inv_freq",
                self._inv_freq(self.long_factor, cpu),
                persistent=False,
            )
        else:
            self.register_buffer(
                "inv_freq", self._inv_freq(None, cpu), persistent=False
            )

    def _inv_freq(self, factors: list[float] | None, device) -> torch.Tensor:
        exponent = (
            torch.arange(
                0, self.rotary_dim, 2, dtype=torch.int64, device=device
            ).float()
            / self.rotary_dim
        )
        inv_freq = 1.0 / (self.base**exponent)
        if factors is not None:
            inv_freq = inv_freq / torch.tensor(
                factors, dtype=torch.float32, device=device
            )
        return inv_freq

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rope_type == "longrope":
            use_long = (position_ids >= self.original_max_position_embeddings).any()
            inv_freq = torch.where(use_long, self.long_inv_freq, self.short_inv_freq)
        else:
            inv_freq = self.inv_freq

        # [T, rotary_dim / 2]
        freqs = position_ids[:, None].float() * inv_freq[None, :].float()
        cos = (freqs.cos() * self.attention_factor).to(dtype)
        sin = (freqs.sin() * self.attention_factor).to(dtype)

        pass_half = (self.head_dim - self.rotary_dim) // 2
        if pass_half > 0:
            tokens = position_ids.shape[0]
            ones = torch.ones(tokens, pass_half, dtype=dtype, device=device)
            zeros = torch.zeros(tokens, pass_half, dtype=dtype, device=device)
            cos = torch.cat((cos, ones), dim=-1)
            sin = torch.cat((sin, zeros), dim=-1)

        # [T, head_dim]
        return torch.cat((cos, cos), dim=-1), torch.cat((sin, sin), dim=-1)


# =============================================================================
# Section 3: Weight loaders for fused checkpoint tensors
# <-- MODEL-SPECIFIC: Phi3 stores fused qkv_proj and gate_up_proj weights.
# =============================================================================


def phi3_qkv_weight_loader(
    num_q_heads_per_rank: int,
    num_kv_heads_per_rank: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_kv_replicas: int,
    head_permutation: list[int],
) -> SafetensorsWeightLoader:
    """Shard the fused HF ``qkv_proj.weight`` for one TP rank.

    The checkpoint tensor has shape [(Nq + 2 * Nkv) * Dh, H] in the order Q, K,
    V. The loader takes the Q heads and the KV heads of the rank, permutes the
    Q and K head dimensions for the kernel RoPE layout, and returns
    [H, (Nq_rank + 2 * Nkv_rank) * Dh].
    """
    q_size = num_q_heads_per_rank * head_dim
    kv_size = num_kv_heads_per_rank * head_dim
    q_total = num_q_heads * head_dim
    kv_total = num_kv_heads * head_dim
    perm = list(head_permutation)

    def _permute(w: torch.Tensor, heads: int) -> torch.Tensor:
        hidden = w.shape[-1]
        index = torch.tensor(perm, dtype=torch.long)
        return w.view(heads, head_dim, hidden)[:, index, :].reshape(-1, hidden)

    def transform(slices, rank: int) -> torch.Tensor:
        assert len(slices) == 1, "phi3_qkv_weight_loader takes one fused tensor"
        s = slices[0]
        kv_rank = rank // num_kv_replicas
        q = s[rank * q_size : (rank + 1) * q_size, :]
        k = s[q_total + kv_rank * kv_size : q_total + (kv_rank + 1) * kv_size, :]
        v_start = q_total + kv_total + kv_rank * kv_size
        v = s[v_start : v_start + kv_size, :]
        q = _permute(q, num_q_heads_per_rank)
        k = _permute(k, num_kv_heads_per_rank)
        return torch.cat((q.t(), k.t(), v.t()), dim=1)

    return SafetensorsWeightLoader(transform=transform)


def phi3_gate_up_weight_loader(
    intermediate_size: int, intermediate_size_per_rank: int, part: int
) -> SafetensorsWeightLoader:
    """Shard one half of the fused HF ``gate_up_proj.weight`` for one TP rank.

    The checkpoint tensor has shape [2 * I, H]. Rows [0, I) are the gate
    projection and rows [I, 2 * I) are the up projection. Set ``part`` to 0 for
    the gate and to 1 for the up projection. The loader returns [H, I_rank].
    """
    offset = part * intermediate_size

    def transform(slices, rank: int) -> torch.Tensor:
        assert len(slices) == 1, "phi3_gate_up_weight_loader takes one fused tensor"
        start = offset + rank * intermediate_size_per_rank
        return slices[0][start : start + intermediate_size_per_rank, :].t()

    return SafetensorsWeightLoader(transform=transform)


# =============================================================================
# Section 4: Attention
# =============================================================================


class Phi3Attention(nn.Module):
    """GQA attention with TP head sharding.

    >>> PARALLELISM: TP <<<
    - Prefill: all-gather input -> QKV proj -> RoPE -> attention -> O proj ->
      reduce-scatter
    - Decode: fused NKI megakernel -> TP all-reduce

    <-- MODEL-SPECIFIC: fused QKV checkpoint, partial LongRoPE, no bias, no
    sliding window.
    """

    def __init__(self, config: Phi3Config, layer_idx: int):
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
        self.rank = self.tp_group.rank_in_group

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

        # KV caches bound externally via bind_kv_cache()
        self.k_cache = None
        self.v_cache = None

        set_weight_loader(
            self.qkv_proj_weight,
            phi3_qkv_weight_loader(
                num_q_heads_per_rank=self.num_attention_heads_per_rank,
                num_kv_heads_per_rank=self.num_key_value_heads_per_rank,
                num_q_heads=self.num_attention_heads,
                num_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                num_kv_replicas=self.num_kv_replicas,
                head_permutation=rope_head_permutation(
                    self.head_dim, config.rotary_dim
                ),
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=o_proj_in_features,
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

        # >>> PARALLELISM: All-gather from SP before attention <<<
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(
            hidden_states, positions, position_embeddings, attn_metadata
        )

    # -- Prefill path ---------------------------------------------------------

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
        tokens, _ = hidden_states.shape

        # Step 1: QKV projection (NKI qkv kernel when its gate accepts the shape)
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
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

        # Step 2: Partial LongRoPE in the permuted head layout
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Step 3: Update the KV cache. The K cache holds the permuted layout.
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

        # Step 4: Attention
        if kv_segment_size:
            # Segmented prefill: the queries attend over the paged KV cache.
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
            k = k.repeat_interleave(self.num_key_value_groups, dim=0)
            v = v.repeat_interleave(self.num_key_value_groups, dim=0)
            attn_output = NF.flash_attention(
                q.transpose(1, 2),  # [Nh, Dh, T]
                k.transpose(1, 2),  # [Nh, Dh, T]
                v,  # [Nh, T, Dh]
                scale=self.scaling,
                tp_q=False,
                tp_out=True,
            )  # [Nh, Dh, T]

        # Step 5: Output projection
        attn_output = attn_output.unsqueeze(0)
        attn_output = NF.o_proj(attn_output, self.o_proj_weight, None)
        attn_output = attn_output.squeeze(0)

        # >>> PARALLELISM: Reduce-scatter to return to SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    # -- Decode path ----------------------------------------------------------

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
    ):
        """Decode with the fused NKI megakernel (QKV, RoPE, attention, O proj)."""
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

        # [T, Dh] -> [Dh / 2, B, S]. The first half holds the scaled rotary
        # values, then 1 (cos) or 0 (sin) for the dimensions that do not rotate.
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

        # The kernel accepts GQA with nkh > 1 only with a per-head 3D block
        # table [B, nkh, num_blocks] and per-head write slots. The torch fallback
        # (CPU mode, or NKI kernels disabled) accepts only the shared 2D table and
        # the flat slots. This is the same branch as _can_use_attention_block_kernel.
        if can_run_kernel(X):
            active_blocks_table = build_per_head_block_table(block_table, nkh).to(
                torch.int32
            )
            kv_cache_update_idx = build_per_head_slot_mapping(
                slot_mapping, block_size, nkh, B
            )
        else:
            active_blocks_table = block_table
            kv_cache_update_idx = slot_mapping.view(B, S_decode)

        # >>> PARALLELISM: Fused megakernel with TP-sharded weights <<<
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
            pos_ids=pos_ids_kernel,
            swa_start_pos_ids=None,
            softmax_scale=self.scaling,
            sink=None,
            update_cache=True,
            kv_cache_update_idx=kv_cache_update_idx,
            W_out=self.o_proj_weight,
            bias_out=None,
            transposed_out=False,
            out_in_sb=False,
        )

        # >>> PARALLELISM: TP all-reduce after megakernel <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 5: MLP (Dense SwiGLU)
# =============================================================================


class Phi3MLP(nn.Module):
    """Dense SwiGLU MLP: down(silu(gate(x)) * up(x)). No bias.

    >>> PARALLELISM: TP <<<
    - gate/up: hidden -> intermediate/TP per rank; down: intermediate/TP -> hidden
    - Prefill: all-gather -> compute -> reduce-scatter (SP)
    - Decode: compute -> all-reduce

    <-- MODEL-SPECIFIC: The fused checkpoint gate_up_proj splits into separate
    gate and up parameters, because the NKI MLP kernel takes them separately.
    """

    def __init__(self, config: Phi3Config):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

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

        set_weight_loader(
            self.gate_proj_weight,
            phi3_gate_up_weight_loader(
                config.intermediate_size, self.intermediate_size_per_rank, part=0
            ),
        )
        set_weight_loader(
            self.up_proj_weight,
            phi3_gate_up_weight_loader(
                config.intermediate_size, self.intermediate_size_per_rank, part=1
            ),
        )
        set_weight_loader(
            self.down_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.intermediate_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
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
        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 6: Decoder Layer
# =============================================================================


class Phi3DecoderLayer(nn.Module):
    """hidden -> RMSNorm -> Attention -> residual -> RMSNorm -> MLP -> residual.

    HF Phi3 has resid_attn_dropout and resid_mlp_dropout. Dropout is the
    identity at inference.
    """

    def __init__(self, config: Phi3Config, layer_idx: int):
        super().__init__()
        self.input_layernorm = Phi3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = Phi3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.self_attn = Phi3Attention(config, layer_idx=layer_idx)
        self.mlp = Phi3MLP(config)
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
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=not is_decode)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Section 7: Model Backbone
# =============================================================================


class Phi3Model(nn.Module):
    """Phi3 transformer backbone.

    >>> PARALLELISM: SP in prefill <<<
    - After embedding: each TP rank holds T / world_size tokens.
    - After the final norm: all-gather the full sequence.
    """

    def __init__(self, config: Phi3Config):
        super().__init__()
        self.config = config

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

        self.layers = nn.ModuleList(
            [
                Phi3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Phi3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Phi3RotaryEmbedding(config)

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
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        hidden_states = self.embed_tokens(
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

        # >>> PARALLELISM: SP - all-gather to reconstruct full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states


# =============================================================================
# Section 8: Language Model Head
# =============================================================================


class Phi3ForCausalLM(nn.Module):
    """Phi3 model with a column-parallel LM head and on-device sampling.

    <-- MODEL-SPECIFIC: Phi-4-mini uses tied embeddings. lm_head.weight loads
    from the embed_tokens checkpoint key.
    """

    def __init__(self, config: Phi3Config):
        super().__init__()
        self.config = config
        self.model = Phi3Model(config)

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
        inputs_embeds: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is not None:
            raise ValueError(
                "Phi3 does not support inputs_embeds. Give input_ids instead."
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
                f"Prompt length {T} is not valid for sequence parallelism. It must be "
                f"larger than world_size ({self.world_size}) and a multiple of it."
            )

        hidden_states = self.model(
            input_ids, positions, attn_metadata=attn_metadata, rank=rank
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

        # >>> PARALLELISM: Gather sharded logits for logprobs computation <<<
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
        config = Phi3Config.from_configs(hf_config, neuron_config)
        return cls(config)

    # -- KV Cache Management --------------------------------------------------

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
                raise RuntimeError(
                    f"KV cache for layer {layer_name} is not initialized. "
                    f"Allocate the KV cache before bind_kv_cache()."
                )
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # -- Weight Loading -------------------------------------------------------

    def checkpoint_mappings(self) -> dict[str, str]:
        """Return the parameter name -> HF checkpoint key mappings."""
        mappings = dict()
        for layer_id in range(len(self.model.layers)):
            prefix = f"model.layers.{layer_id}"
            mappings[f"{prefix}.self_attn.qkv_proj_weight"] = (
                f"{prefix}.self_attn.qkv_proj.weight"
            )
            mappings[f"{prefix}.self_attn.o_proj_weight"] = (
                f"{prefix}.self_attn.o_proj.weight"
            )
            mappings[f"{prefix}.input_layernorm.weight"] = (
                f"{prefix}.input_layernorm.weight"
            )
            mappings[f"{prefix}.post_attention_layernorm.weight"] = (
                f"{prefix}.post_attention_layernorm.weight"
            )
            mappings[f"{prefix}.mlp.gate_proj_weight"] = (
                f"{prefix}.mlp.gate_up_proj.weight"
            )
            mappings[f"{prefix}.mlp.up_proj_weight"] = (
                f"{prefix}.mlp.gate_up_proj.weight"
            )
            mappings[f"{prefix}.mlp.down_proj_weight"] = (
                f"{prefix}.mlp.down_proj.weight"
            )
        if self.config.tie_word_embeddings:
            mappings["lm_head.weight"] = "model.embed_tokens.weight"
        return mappings

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, self.checkpoint_mappings(), device
        ).state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self.load_state_dict(rank_sharded, strict=False, assign=True)
