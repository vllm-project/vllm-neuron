# SPDX-License-Identifier: Apache-2.0
"""
GLM-4 BF16 Implementation
=========================

Implementation of GLM-4 (``Glm4ForCausalLM``, for example GLM-4-32B-0414) for
the Neuron native backend. The structure follows the Llama3 dense model in
``vllm_neuron/model/llama3/``.

GLM-4 differs from Llama in two items:

1. Sandwich norms. Each decoder layer has four RMSNorms:
   ``x = x + post_self_attn_ln(attn(input_ln(x)))`` and
   ``x = x + post_mlp_ln(mlp(post_attention_ln(x)))``.
2. Partial interleaved RoPE. RoPE rotates only the first
   ``rotary_dim = head_dim * partial_rotary_factor`` dimensions of each head.
   It uses (even, odd) pairs, and the remaining dimensions pass through.

The NKI prefill QKV kernel (fused RoPE) and the decode attention kernel use
full-head "rotate half" RoPE: they rotate dimension ``j`` with dimension
``j + head_dim / 2``. To use these kernels, the weight loader permutes the
output columns of each Q head and each K head (see ``glm4_rope_permutation``).
After the permutation, each GLM pair ``(2i, 2i + 1)`` is at ``(i, i + head_dim/2)``.
The pass-through dimensions get a frequency of zero (cos=1, sin=0). The
permutation is the same for Q and K, so Q.K is not changed. V is not
permuted, so the attention output and o_proj keep the checkpoint layout.
The KV cache holds K in the permuted layout. Prefill and decode use the same
layout, so a KV cache transfer between prefill and decode nodes is consistent.

Supported parallelism: TP, SP.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code.
  # <-- MODEL-SPECIFIC: ...    GLM-4-specific code.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from transformers import PretrainedConfig
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.sampler import Sampler

import vllm_neuron.nn as neuron_nn
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding

from .config import Glm4Config

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1: RMS Normalization
# <-- MODEL-SPECIFIC: GLM-4 uses plain RMSNorm (no offset, no bias)
# =============================================================================


class Glm4RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            hidden_states, self.weight.shape, self.weight, self.variance_epsilon
        )


# =============================================================================
# Section 2: Rotary Position Embedding (partial, interleaved)
# <-- MODEL-SPECIFIC: GLM-4 partial interleaved RoPE in the rotate-half layout
# =============================================================================


def glm4_rope_permutation(head_dim: int, rotary_dim: int) -> torch.Tensor:
    """Return the per-head column order that maps GLM RoPE to rotate-half RoPE.

    The result ``perm`` has ``head_dim`` entries. Column ``n`` of a permuted
    head is column ``perm[n]`` of the checkpoint head. With ``half = head_dim // 2``
    and ``r = rotary_dim // 2``:

    - ``n < r``: ``perm[n] = 2n`` (the even element of GLM pair ``n``).
    - ``half <= n < half + r``: ``perm[n] = 2(n - half) + 1`` (the odd element).
    - Other ``n``: the pass-through dimensions ``rotary_dim .. head_dim - 1``, in order.

    Example::

        >>> glm4_rope_permutation(8, 4).tolist()
        [0, 2, 4, 5, 1, 3, 6, 7]
    """
    if head_dim % 2 != 0 or rotary_dim % 2 != 0 or not 0 < rotary_dim <= head_dim:
        raise ValueError(
            f"Rotary dimension {rotary_dim} is not valid for head_dim {head_dim}. "
            f"Both values must be even, and rotary_dim must be from 2 to head_dim."
        )
    half = head_dim // 2
    r = rotary_dim // 2
    num_pass = (head_dim - rotary_dim) // 2
    # Set the device explicitly. The model can be built under a meta device
    # context, and a meta tensor has no index values.
    perm = torch.empty(head_dim, dtype=torch.long, device="cpu")
    idx = torch.arange(r, device="cpu")
    perm[:r] = 2 * idx
    perm[half : half + r] = 2 * idx + 1
    pass_idx = torch.arange(num_pass, device="cpu")
    perm[r:half] = rotary_dim + pass_idx
    perm[half + r :] = rotary_dim + num_pass + pass_idx
    return perm


def glm4_permute_qk_columns(
    qkv: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    perm: torch.Tensor,
) -> torch.Tensor:
    """Permute the Q and K head columns of a fused ``[H, q + 2 * kv]`` weight.

    V columns do not change.

    Example::

        >>> w = torch.arange(2 * 3 * 4).reshape(2, 12).float()
        >>> out = glm4_permute_qk_columns(w, 1, 1, 4, torch.tensor([0, 2, 1, 3]))
        >>> out[0].tolist()
        [0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    """
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    rows = qkv.shape[0]
    q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=1)
    q = q.reshape(rows, num_q_heads, head_dim)[:, :, perm].reshape(rows, q_size)
    k = k.reshape(rows, num_kv_heads, head_dim)[:, :, perm].reshape(rows, kv_size)
    return torch.cat([q, k, v], dim=1)


class Glm4RotaryEmbedding(nn.Module):
    """cos/sin tables in the rotate-half layout for the permuted Q/K heads.

    Entry ``j < rotary_dim / 2`` of the half table holds GLM frequency ``j``.
    The other entries hold frequency zero, which gives cos=1 and sin=0 for the
    pass-through dimensions.
    """

    def __init__(self, config: Glm4Config):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim

    def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
        # <-- MODEL-SPECIFIC: GLM frequencies use rotary_dim, not head_dim.
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device=device)
                / self.rotary_dim
            )
        )
        num_pass = (self.head_dim - self.rotary_dim) // 2
        if num_pass > 0:
            inv_freq = torch.cat(
                [inv_freq, torch.zeros(num_pass, dtype=torch.float, device=device)]
            )
        return inv_freq  # [head_dim / 2]

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos, sin, both with shape [T, head_dim]."""
        inv_freq = self._compute_inv_freq(device)
        inv_freq_expanded = inv_freq[:, None].float()  # [Hd/2, 1]
        positions_expanded = position_ids[None, :].float()  # [1, T]
        freqs = (inv_freq_expanded @ positions_expanded).transpose(0, 1)  # [T, Hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, Hd]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


# =============================================================================
# Section 3: Attention
# =============================================================================


class Glm4Attention(nn.Module):
    """GQA attention with TP head sharding.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads are sharded across TP ranks.
    - KV heads are replicated when there are fewer KV heads than TP ranks.
    - Prefill: all-gather -> QKV proj (fused RoPE) -> attention -> O proj -> reduce-scatter
    - Decode: fused attention block kernel, then TP all-reduce

    <-- MODEL-SPECIFIC: no attention bias, no sinks, no sliding window. The Q/K
    columns are permuted at load time (see the module docstring).
    """

    def __init__(self, config: Glm4Config, layer_idx: int):
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

        if self.num_attention_heads % self.world_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) is not divisible "
                f"by the TP size ({self.world_size}). Use a TP size that divides it."
            )

        # >>> PARALLELISM: Head sharding calculation <<<
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
        if self.world_size >= self.num_key_value_heads:
            if self.world_size % self.num_key_value_heads != 0:
                raise ValueError(
                    f"TP size ({self.world_size}) is not a multiple of "
                    f"num_key_value_heads ({self.num_key_value_heads}). "
                    f"Use a TP size that is a multiple of it."
                )
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

        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        # KV caches bound externally via bind_kv_cache()
        self.k_cache = None
        self.v_cache = None

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach the checkpoint loaders.

        <-- MODEL-SPECIFIC: Load separate Q, K, V weights into a fused QKV
        weight, then permute the Q/K head columns for the rotate-half RoPE layout.
        """
        base_qkv_loader = fused_qkv_weight_loader(
            q_size=self.q_size,
            kv_size=self.kv_size,
            shard_dim=1,
            num_shards=self.world_size,
            is_storage_transposed=True,
            num_kv_replicas=self.num_kv_replicas,
        )
        perm = glm4_rope_permutation(self.head_dim, self.rotary_dim)
        num_q = self.num_attention_heads_per_rank
        num_kv = self.num_key_value_heads_per_rank
        head_dim = self.head_dim

        def qkv_transform(slices, rank):
            fused = base_qkv_loader.load(slices, rank)
            return glm4_permute_qk_columns(
                fused, num_q, num_kv, head_dim, perm.to(fused.device)
            )

        set_weight_loader(
            self.qkv_proj_weight, SafetensorsWeightLoader(transform=qkv_transform)
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

    # -- Forward dispatch -----------------------------------------------------

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

    # -- Prefill path ---------------------------------------------------------

    def _write_kv_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ) -> None:
        """Write K/V [Nkv, T, Dh] to the paged cache with the slot mapping."""
        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size
        num_slots = slot_mapping.shape[0]
        nkh = self.num_key_value_heads_per_rank

        k_flat = k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
        v_flat = v.reshape(-1, self.head_dim).to(self.v_cache.dtype)
        head_indices = torch.arange(
            nkh, dtype=torch.long, device=k.device
        ).repeat_interleave(num_slots)
        index = (block_indices.repeat(nkh), head_indices, position_indices.repeat(nkh))
        self.k_cache.index_put_(index, k_flat)
        self.v_cache.index_put_(index, v_flat)

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """Prefill: QKV projection with fused RoPE, then attention and O proj."""
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # <-- MODEL-SPECIFIC: GLM RoPE in the rotate-half layout (permuted Q/K
        # columns). The qkv_proj kernel applies it.
        cos, sin = position_embeddings
        cos_cache = cos.unsqueeze(0)
        sin_cache = sin.unsqueeze(0)

        # >>> PARALLELISM: Cache is per-rank (TP sharded KV heads) <<<
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        if kv_segment_size:
            # Segmented prefill. The qkv kernel writes the post-RoPE K/V into
            # the cache, and segmented_attention reads the cache.
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
                num_kv_heads=self.num_key_value_heads_per_rank,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                use_block_kv=True,
                block_size=block_size,
                slot_mapping=slot_mapping_clamped.unsqueeze(0),
            )
            q = (
                q_hbm.squeeze(0)
                .view(tokens, self.num_attention_heads_per_rank, self.head_dim)
                .transpose(0, 1)
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
                allow_vector_exp=False,
            )  # [Nh, Dh, T]
        else:
            qkv = NF.qkv_proj(
                hidden=hidden_states.unsqueeze(0),
                qkv_weights=self.qkv_proj_weight,
                bias=None,
                d_head=self.head_dim,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                num_q_heads=self.num_attention_heads_per_rank,
                num_kv_heads=self.num_key_value_heads_per_rank,
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

            self._write_kv_cache(k, v, slot_mapping, block_size)

            k = k.repeat_interleave(self.num_key_value_groups, dim=0)
            v = v.repeat_interleave(self.num_key_value_groups, dim=0)

            attn_output = NF.flash_attention(
                q.transpose(1, 2),  # [Nh, Dh, T]
                k.transpose(1, 2),  # [Nh, Dh, T]
                v,  # [Nh, T, Dh]
                scale=self.scaling,
                tp_q=False,
                tp_out=True,
                allow_vector_exp=False,
            )  # [Nh, Dh, T]

        attn_output = attn_output.unsqueeze(0)  # [1, Nh, Dh, T]
        attn_output = NF.o_proj(attn_output, self.o_proj_weight, None)  # [1, T, H]
        attn_output = attn_output.squeeze(0)  # [T, H]

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
        """Decode: fused attention block kernel (QKV, RoPE, cache, attention, O proj)."""
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        assert tokens == B * S_decode

        hidden_states = hidden_states.to(self.dtype)
        S_ctx = max_blocks_per_seq * block_size

        X = hidden_states.view(B, S_decode, hidden)

        # Kernel RoPE format: [T, Hd] -> [Hd/2, B, S]. The rotate-half table
        # repeats the half table, so the first half is sufficient.
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

        # >>> PARALLELISM: Fused kernel with TP-sharded weights <<<
        # The kernel writes the new K/V tokens into the cache in place
        # (update_cache=True), as in the Llama3 decode path.
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
            active_blocks_table=block_table,
            K_cache=self.k_cache,
            V_cache=self.v_cache,
            attention_mask=attention_mask,
            softmax_scale=self.scaling,
            sink=None,
            update_cache=True,
            kv_cache_update_idx=slot_mapping.view(B, S_decode).to(torch.uint32),
            W_out=self.o_proj_weight,
            bias_out=None,
            transposed_out=False,
            out_in_sb=False,
        )

        # >>> PARALLELISM: TP all-reduce after the kernel <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 4: MLP (Dense, fused gate_up checkpoint tensor)
# =============================================================================


def glm4_gate_up_weight_loader(
    intermediate_size: int,
    intermediate_size_per_rank: int,
    num_shards: int,
    is_up: bool,
) -> SafetensorsWeightLoader:
    """Load the gate half or the up half of a GLM ``gate_up_proj.weight``.

    The checkpoint tensor has shape ``[2 * I, H]``. Rows ``[0, I)`` are gate and
    rows ``[I, 2 * I)`` are up. The loader returns the TP shard of the selected
    half as ``[H, I / num_shards]``.

    Example::

        set_weight_loader(
            mlp.up_proj_weight,
            glm4_gate_up_weight_loader(23040, 2880, 8, is_up=True),
        )
    """
    offset = intermediate_size if is_up else 0

    def transform(slices, rank):
        assert len(slices) == 1, "gate_up loader expects one checkpoint slice"
        local_rank = rank % num_shards
        start = offset + local_rank * intermediate_size_per_rank
        end = start + intermediate_size_per_rank
        return slices[0][start:end, :].T

    return SafetensorsWeightLoader(transform=transform)


class Glm4MLP(nn.Module):
    """Dense SwiGLU MLP with TP intermediate sharding.

    >>> PARALLELISM: TP <<<
    - Prefill: all-gather before, reduce-scatter after (SP)
    - Decode: all-reduce after

    <-- MODEL-SPECIFIC: ``down(up * silu(gate))``. The checkpoint stores gate and
    up as one ``gate_up_proj.weight`` tensor, and the loaders split it.
    """

    def __init__(self, config: Glm4Config):
        super().__init__()

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        if config.intermediate_size % self.world_size != 0:
            raise ValueError(
                f"intermediate_size ({config.intermediate_size}) is not divisible "
                f"by the TP size ({self.world_size}). Use a TP size that divides it."
            )

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
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
        set_weight_loader(
            self.gate_proj_weight,
            glm4_gate_up_weight_loader(
                self.intermediate_size,
                self.intermediate_size_per_rank,
                self.world_size,
                is_up=False,
            ),
        )
        set_weight_loader(
            self.up_proj_weight,
            glm4_gate_up_weight_loader(
                self.intermediate_size,
                self.intermediate_size_per_rank,
                self.world_size,
                is_up=True,
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
# Section 5: Decoder Layer (sandwich norms)
# =============================================================================


class Glm4DecoderLayer(nn.Module):
    """GLM-4 decoder layer.

    <-- MODEL-SPECIFIC:
        r = x; x = post_self_attn_ln(attn(input_ln(x))); x = r + x
        r = x; x = post_mlp_ln(mlp(post_attention_ln(x))); x = r + x
    """

    def __init__(self, config: Glm4Config, layer_idx: int):
        super().__init__()
        eps = config.rms_norm_eps
        dtype = config.torch_dtype
        self.input_layernorm = Glm4RMSNorm(config.hidden_size, eps, dtype)
        self.post_self_attn_layernorm = Glm4RMSNorm(config.hidden_size, eps, dtype)
        self.post_attention_layernorm = Glm4RMSNorm(config.hidden_size, eps, dtype)
        self.post_mlp_layernorm = Glm4RMSNorm(config.hidden_size, eps, dtype)
        self.self_attn = Glm4Attention(config, layer_idx=layer_idx)
        self.mlp = Glm4MLP(config)
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
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=not is_decode)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Section 6: Model Backbone
# =============================================================================


class Glm4Model(nn.Module):
    """GLM-4 transformer backbone.

    >>> PARALLELISM: SP <<<
    Prefill: the embedding scatters tokens across TP ranks, and the output is
    all-gathered after the final norm. Decode: no SP.
    """

    def __init__(self, config: Glm4Config):
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
            [Glm4DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )

        self.norm = Glm4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Glm4RotaryEmbedding(config)

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
# Section 7: Language Model Head
# =============================================================================


class Glm4ForCausalLM(nn.Module):
    """GLM-4 model with a column-parallel LM head (untied embeddings)."""

    def __init__(self, config: Glm4Config):
        super().__init__()
        self.config = config
        self.model = Glm4Model(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        if config.vocab_size % self.world_size != 0:
            raise ValueError(
                f"vocab_size ({config.vocab_size}) is not divisible by the TP size "
                f"({self.world_size}). Use a TP size that divides it."
            )

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
                "GLM-4 does not support inputs_embeds. Give input_ids instead."
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
                f"Prompt length ({T}) is not valid for SP with world_size "
                f"({self.world_size}). Use a prefill bucket that is a multiple "
                f"of world_size and larger than it."
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
        config = Glm4Config.from_configs(hf_config, neuron_config)
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
                    f"Allocate the KV cache before the model binds it."
                )
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # -- Weight Loading -------------------------------------------------------

    @staticmethod
    def checkpoint_mappings(num_layers: int) -> dict[str, str | list[str]]:
        """Return the parameter-name to checkpoint-key mappings.

        A parameter that is not in the result uses its own name as the key.
        """
        mappings: dict[str, str | list[str]] = {}
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
            # <-- MODEL-SPECIFIC: gate and up come from one fused tensor.
            mappings[f"{prefix}.mlp.gate_proj_weight"] = (
                f"{prefix}.mlp.gate_up_proj.weight"
            )
            mappings[f"{prefix}.mlp.up_proj_weight"] = (
                f"{prefix}.mlp.gate_up_proj.weight"
            )
            mappings[f"{prefix}.mlp.down_proj_weight"] = (
                f"{prefix}.mlp.down_proj.weight"
            )
        return mappings

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        mappings = self.checkpoint_mappings(len(self.model.layers))
        if self.config.tie_word_embeddings:
            mappings["lm_head.weight"] = "model.embed_tokens.weight"

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, mappings, device
        ).state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self.load_state_dict(rank_sharded, strict=False, assign=True)
