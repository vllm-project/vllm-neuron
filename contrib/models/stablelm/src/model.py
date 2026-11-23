# SPDX-License-Identifier: Apache-2.0
"""
StableLM BF16 Implementation
============================

Implementation of StableLM (``StableLmForCausalLM``) for the Neuron backend.
The structure follows the plugin's Llama3 BF16 model (``vllm_neuron/model/llama3``).

Differences from Llama:

- Each norm is a full LayerNorm with a bias.
- RoPE is partial. It rotates the first ``rotary_dim`` dims of each head with
  the rotate_half pairing (dim i with dim i + rotary_dim / 2). The other dims
  pass through unchanged.
- The model uses multi-head attention (32 Q heads and 32 KV heads).
- The lm_head is not tied to the embedding.

Partial RoPE in the permuted head layout:
  The weight loader permutes the Q and K output columns of each head (see
  ``rotary_head_dim_permutation``). After the permutation, the rotary pairs
  are at (i, i + head_dim / 2) for i < rotary_dim / 2, and the cos/sin tables
  hold cos=1 and sin=0 for all other dims. A full-head rotate_half RoPE then
  gives the partial RoPE of HF. The Q.K dot product does not change, because
  Q and K use the same permutation. V and O are not permuted. Prefill and
  decode use the same layout, so the KV cache layout is the same in both.

Decode attention:
  The decode megakernel (``NF.attention_decode``) is not used. Its RoPE stage
  accepts only d_head 64 or 128, and StableLM Zephyr 3B has d_head 80. Decode
  uses the QKV projection kernel. RoPE, the KV cache write, paged attention
  and the O projection are in torch. The O projection and its all-reduce are
  in FP32.

Supported parallelism: TP, SP.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    StableLM-specific. Change when porting.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
from vllm_neuron.model.kv_cache import (
    KVSpec,
    LayerSpec,
    paged_slot_indices,
)
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

from .config import StableLmConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1: LayerNorm with bias
# <-- MODEL-SPECIFIC: StableLM uses nn.LayerNorm (mean subtraction and bias)
# =============================================================================


class StableLmLayerNorm(nn.Module):
    """Full LayerNorm with a learned weight and bias."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            hidden_states,
            self.weight.shape,
            self.weight,
            self.bias,
            self.variance_epsilon,
        )


# =============================================================================
# Section 2: Partial RoPE in the permuted head layout
# <-- MODEL-SPECIFIC: partial_rotary_factor < 1
# =============================================================================


def rotary_head_dim_permutation(head_dim: int, rotary_dim: int) -> torch.Tensor:
    """Return the head-dim permutation for partial RoPE.

    ``new[n] = orig[perm[n]]``. The rotary pair (j, j + rotary_dim / 2) of the
    original layout moves to (j, j + head_dim / 2). The pass-through dims fill
    the other positions in their original order.

    Example:
        >>> rotary_head_dim_permutation(8, 4).tolist()
        [0, 1, 4, 5, 2, 3, 6, 7]
    """
    half_r = rotary_dim // 2
    half_d = head_dim // 2
    n_pass_half = half_d - half_r
    # Set the device explicitly. The runner builds the model inside a
    # torch.device("meta") context, and a tensor without a device then has no data.
    cpu = torch.device("cpu")
    return torch.cat(
        [
            torch.arange(0, half_r, device=cpu),
            torch.arange(rotary_dim, rotary_dim + n_pass_half, device=cpu),
            torch.arange(half_r, rotary_dim, device=cpu),
            torch.arange(rotary_dim + n_pass_half, head_dim, device=cpu),
        ]
    )


def partial_rotary_qkv_weight_loader(
    base_loader: SafetensorsWeightLoader,
    q_size: int,
    kv_size: int,
    head_dim: int,
    rotary_dim: int,
) -> SafetensorsWeightLoader:
    """Wrap a fused QKV loader. Permute the Q and K columns of each head.

    The base loader gives ``[hidden, q_size + 2 * kv_size]``. Each head has
    ``head_dim`` adjacent columns. V columns do not change.

    The permutation is a CPU tensor. The model runner calls this function
    inside a ``torch.device("meta")`` context, and the loader then still has a
    valid permutation at load time.

    Example:
        >>> loader = partial_rotary_qkv_weight_loader(base, 320, 320, 80, 20)  # doctest: +SKIP
    """
    perm = rotary_head_dim_permutation(head_dim, rotary_dim)

    def _permute_heads(w: torch.Tensor) -> torch.Tensor:
        hidden, cols = w.shape
        w = w.reshape(hidden, cols // head_dim, head_dim)
        return w[:, :, perm.to(w.device)].reshape(hidden, cols)

    def transform(slices, rank: int) -> torch.Tensor:
        fused = base_loader.load(slices, rank)
        q = fused[:, :q_size]
        k = fused[:, q_size : q_size + kv_size]
        v = fused[:, q_size + kv_size :]
        return torch.cat([_permute_heads(q), _permute_heads(k), v], dim=1)

    return SafetensorsWeightLoader(transform=transform)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply full-head rotate_half RoPE in the permuted layout.

    Args:
        q: [Nh, T, Dh], k: [Nkv, T, Dh]
        cos: [T, Dh], sin: [T, Dh]
    """
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)

    def _rot(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    return _rot(q), _rot(k)


class StableLmRotaryEmbedding(nn.Module):
    """Partial RoPE tables for the permuted head layout.

    The first ``rotary_dim / 2`` entries of each half hold the real cos/sin.
    The other entries hold cos=1 and sin=0, so these dims do not rotate.
    """

    def __init__(self, config: StableLmConfig):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin, both with shape [T, head_dim]."""
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device=device)
                / self.rotary_dim
            )
        )
        freqs = position_ids[:, None].float() * inv_freq[None, :]  # [T, R/2]
        n_pass_half = self.head_dim // 2 - self.rotary_dim // 2
        ones = torch.ones(freqs.shape[0], n_pass_half, dtype=torch.float, device=device)
        zeros = torch.zeros_like(ones)
        cos_half = torch.cat((freqs.cos(), ones), dim=-1)  # [T, Dh/2]
        sin_half = torch.cat((freqs.sin(), zeros), dim=-1)
        cos = torch.cat((cos_half, cos_half), dim=-1)
        sin = torch.cat((sin_half, sin_half), dim=-1)
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


# =============================================================================
# Section 3: Attention
# =============================================================================


class StableLmAttention(nn.Module):
    """Multi-head attention with TP head sharding and partial RoPE.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads are sharded across TP ranks
    - Prefill: all-gather input -> QKV proj -> attention -> O proj -> reduce-scatter
    - Decode: QKV proj -> paged attention -> O proj -> all-reduce

    <-- MODEL-SPECIFIC:
    - No QKV bias, no O bias, no QK norm
    - Partial RoPE in the permuted head layout
    """

    def __init__(self, config: StableLmConfig, layer_idx: int):
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

        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        # KV caches bound externally via bind_kv_cache()
        self.k_cache = None
        self.v_cache = None

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach the weight loaders.

        <-- MODEL-SPECIFIC: The checkpoint stores separate Q, K, V weights.
        The QKV loader permutes the Q and K head dims for partial RoPE.
        """
        base_qkv_loader = fused_qkv_weight_loader(
            q_size=self.q_size,
            kv_size=self.kv_size,
            shard_dim=1,
            num_shards=self.world_size,
            is_storage_transposed=True,
            num_kv_replicas=self.num_kv_replicas,
        )
        set_weight_loader(
            self.qkv_proj_weight,
            partial_rotary_qkv_weight_loader(
                base_qkv_loader,
                q_size=self.q_size,
                kv_size=self.kv_size,
                head_dim=self.head_dim,
                rotary_dim=self.rotary_dim,
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

    # -- Forward dispatch -----------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ):
        """Send the call to the prefill path or to the decode path."""
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

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """Prefill: QKV proj, partial RoPE, KV cache write, attention, O proj."""
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # -- Step 1: QKV Projection -------------------------------------------
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

        # -- Step 2: Partial RoPE (permuted layout) ---------------------------
        # head_dim 80 does not divide 512, so the qkv kernel cannot fuse RoPE.
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # -- Step 3: Update KV Cache -----------------------------------------
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        self._write_kv_cache(k, v, slot_mapping, block_size)

        # -- Step 4: Attention ------------------------------------------------
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
            )  # [Nh, Dh, T]
        else:
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

        # -- Step 5: Output Projection ----------------------------------------
        attn_output = attn_output.unsqueeze(0)  # [1, Nh, Dh, T]
        attn_output = NF.o_proj(attn_output, self.o_proj_weight, None)  # [1, T, H]
        attn_output = attn_output.squeeze(0)  # [T, H]

        # >>> PARALLELISM: Reduce-scatter to return to SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    # -- KV cache write ------------------------------------------------------

    def _write_kv_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ) -> None:
        """Write K and V of each token into the paged cache.

        Args:
            k, v: [Nkv, T, Dh] in the permuted head layout.
            slot_mapping: [T] physical slot of each token. -1 marks a pad slot.
        """
        # >>> PARALLELISM: Cache is per-rank (TP sharded KV heads) <<<
        # The cache is [num_blocks, nkh, block_size, Dh]. Write each token at
        # (block, head, position). A pad slot (-1) goes to the null block.
        nkh = self.num_key_value_heads_per_rank
        block_indices, position_indices = paged_slot_indices(
            slot_mapping, block_size, self.k_cache.shape[0]
        )

        k_flat = k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
        v_flat = v.reshape(-1, self.head_dim).to(self.v_cache.dtype)

        head_indices_for_put = torch.arange(
            nkh, dtype=torch.long, device=k.device
        ).repeat_interleave(slot_mapping.shape[0])
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

    # -- Decode path ----------------------------------------------------------

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
    ):
        """Decode: QKV proj, partial RoPE, KV cache write, paged attention, O proj.

        <-- MODEL-SPECIFIC: The decode megakernel (``NF.attention_decode``) is
        not used. Its RoPE stage accepts only d_head 64 or 128, and StableLM
        Zephyr 3B has d_head 80. The torch fallback of the megakernel does not
        accept a per-head block table when a rank holds more than one KV head.
        This path uses the QKV projection kernel. It computes RoPE, attention
        and the O projection in torch in FP32, and it casts to BF16 once after
        the all-reduce. The KV cache stays in BF16.
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        assert tokens == B * S_decode

        hidden_states = hidden_states.to(self.dtype)
        nh = self.num_attention_heads_per_rank
        nkh = self.num_key_value_heads_per_rank
        head_dim = self.head_dim

        # -- Step 1: QKV Projection -------------------------------------------
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
        ).squeeze(0)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
        q = q.view(tokens, nh, head_dim).transpose(0, 1)  # [Nh, T, Dh]
        k = k.view(tokens, nkh, head_dim).transpose(0, 1)  # [Nkv, T, Dh]
        v = v.view(tokens, nkh, head_dim).transpose(0, 1)  # [Nkv, T, Dh]

        # -- Step 2: Partial RoPE (permuted layout) ---------------------------
        # Rotate in FP32. The KV cache write rounds K to the cache dtype once.
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(
            q.to(torch.float32),
            k.to(torch.float32),
            cos.to(torch.float32),
            sin.to(torch.float32),
        )

        # -- Step 3: Update KV Cache -----------------------------------------
        # Write before the gather, so that each token also attends to itself.
        self._write_kv_cache(k, v, slot_mapping, block_size)

        # -- Step 4: Paged attention over the block table ---------------------
        # Logical position j of a sequence is at block table entry
        # j // block_size, offset j % block_size. A -1 entry reads the null
        # block. The position mask removes it.
        max_blocks = block_table.shape[-1]
        s_ctx = max_blocks * block_size
        flat_blocks = torch.clamp(block_table, min=0).to(torch.long).reshape(-1)
        k_ctx = (
            self.k_cache[flat_blocks]
            .view(B, max_blocks, nkh, block_size, head_dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(B, nkh, s_ctx, head_dim)
        )
        v_ctx = (
            self.v_cache[flat_blocks]
            .view(B, max_blocks, nkh, block_size, head_dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(B, nkh, s_ctx, head_dim)
        )
        if self.num_key_value_groups > 1:
            k_ctx = k_ctx.repeat_interleave(self.num_key_value_groups, dim=1)
            v_ctx = v_ctx.repeat_interleave(self.num_key_value_groups, dim=1)

        # The tokens of one sequence are adjacent: [Nh, B * S, Dh] -> [B, Nh, S, Dh]
        q_b = q.reshape(nh, B, S_decode, head_dim).transpose(0, 1)

        # Keep the attention in FP32 up to the output cast after the all-reduce.
        k_ctx_t = k_ctx.to(torch.float32).transpose(-1, -2)
        scores = torch.matmul(q_b, k_ctx_t) * self.scaling  # [B, Nh, S, S_ctx]

        # A pad token has position -1. Clamp it to 0, so that its row keeps one
        # valid entry and the softmax does not give NaN.
        query_pos = positions.view(B, S_decode).to(torch.long).clamp(min=0)
        ctx_pos = torch.arange(s_ctx, dtype=torch.long, device=hidden_states.device)
        allowed = ctx_pos.view(1, 1, s_ctx) <= query_pos.unsqueeze(-1)  # [B, S, S_ctx]
        scores = scores.masked_fill(~allowed.unsqueeze(1), float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(probs, v_ctx.to(torch.float32))  # [B, Nh, S, Dh]

        # -- Step 5: Output Projection ----------------------------------------
        # Compute the partial sum of each rank in FP32, and all-reduce in FP32.
        # A BF16 partial sum on each rank adds one rounding per rank, and the
        # decode error then becomes larger than the error of the HF BF16 module.
        # [B, Nh, S, Dh] -> [B * S, Nh * Dh], the token order of the input.
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(tokens, nh * head_dim)
        output = torch.matmul(attn_output, self.o_proj_weight.to(torch.float32))

        # >>> PARALLELISM: TP all-reduce after the O projection <<<
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output)

        return output.to(self.dtype).contiguous()


# =============================================================================
# Section 4: MLP (Dense SwiGLU)
# =============================================================================


class StableLmMLP(nn.Module):
    """Dense SwiGLU MLP with TP intermediate sharding.

    >>> PARALLELISM: TP <<<
    - Prefill: all-gather before -> compute -> reduce-scatter after (SP)
    - Decode: compute -> all-reduce
    """

    def __init__(self, config: StableLmConfig):
        super().__init__()

        # >>> PARALLELISM: TP group setup <<<
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
        # >>> PARALLELISM: All-gather from SP for full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        # <-- MODEL-SPECIFIC: down(silu(gate(x)) * up(x))
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
# Section 5: Decoder Layer
# =============================================================================


class StableLmDecoderLayer(nn.Module):
    """Decoder layer with sequential residuals.

    hidden -> LayerNorm -> Attention -> residual -> LayerNorm -> MLP -> residual
    """

    def __init__(self, config: StableLmConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = StableLmLayerNorm(
            config.hidden_size, config.layer_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = StableLmLayerNorm(
            config.hidden_size, config.layer_norm_eps, config.torch_dtype
        )
        self.self_attn = StableLmAttention(config, layer_idx=layer_idx)
        self.mlp = StableLmMLP(config)
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
# Section 6: Model Backbone
# =============================================================================


class StableLmModel(nn.Module):
    """StableLM transformer backbone.

    >>> PARALLELISM: SP (Sequence Parallelism) <<<
    Prefill: the embedding scatters tokens across TP ranks. The model
    all-gathers the sequence after the final norm. Decode does not use SP.
    """

    def __init__(self, config: StableLmConfig):
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
                StableLmDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = StableLmLayerNorm(
            config.hidden_size, config.layer_norm_eps, config.torch_dtype
        )
        self.rotary_emb = StableLmRotaryEmbedding(config)

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


class StableLmForCausalLM(nn.Module):
    """StableLM model with an untied, column-parallel language model head."""

    def __init__(self, config: StableLmConfig):
        super().__init__()
        self.config = config
        self.model = StableLmModel(config)

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
                "StableLM on Neuron does not support inputs_embeds. "
                "Give input_ids instead."
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
                f"({self.world_size}). Use a prefill bucket that is larger than "
                f"world_size and divisible by world_size."
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
        config = StableLmConfig.from_configs(hf_config, neuron_config)
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
                    "Allocate the KV cache before you bind it."
                )
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # -- Weight Loading -------------------------------------------------------

    @staticmethod
    def checkpoint_mappings(num_layers: int) -> dict[str, str | list[str]]:
        """Return the parameter-name to checkpoint-key mappings.

        A parameter that is not in the mappings uses its own name as the key
        (``model.embed_tokens.weight``, ``model.norm.weight``,
        ``model.norm.bias`` and ``lm_head.weight``).
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
            for norm in ("input_layernorm", "post_attention_layernorm"):
                for p in ("weight", "bias"):
                    mappings[f"{prefix}.{norm}.{p}"] = f"{prefix}.{norm}.{p}"
            mappings[f"{prefix}.mlp.gate_proj_weight"] = (
                f"{prefix}.mlp.gate_proj.weight"
            )
            mappings[f"{prefix}.mlp.up_proj_weight"] = f"{prefix}.mlp.up_proj.weight"
            mappings[f"{prefix}.mlp.down_proj_weight"] = (
                f"{prefix}.mlp.down_proj.weight"
            )
        return mappings

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load the checkpoint weights for this TP rank."""
        mappings = self.checkpoint_mappings(len(self.model.layers))

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, mappings, device
        ).state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self.load_state_dict(rank_sharded, strict=False, assign=True)
