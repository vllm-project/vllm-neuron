# SPDX-License-Identifier: Apache-2.0
"""
ERNIE 4.5 (dense) BF16 Implementation
=====================================

Implementation of ERNIE 4.5 dense (for example ``baidu/ERNIE-4.5-0.3B-PT``)
for the Neuron native backend. The structure comes from the SmolLM3 and Llama3
BF16 models.

ERNIE 4.5 dense is a Llama-type decoder with ONE important difference: RoPE is
interleaved (GLM style). The rotation pairs the dimensions (2i, 2i + 1) of each
head. The Neuron kernels pair the dimensions (i, i + Dh/2) ("rotate half").

The weight loader reorders the output rows of q_proj and k_proj in each head
to [0, 2, 4, ..., Dh-2, 1, 3, ..., Dh-1]. After this reorder, the rotate-half
RoPE gives the same rotation as the interleaved RoPE, in the reordered layout.
Attention uses only the dot product q . k, and the same reorder applies to Q and
K. Thus the attention scores do not change. V and o_proj do not change. The
prefill path, the decode path and the KV cache all use the reordered layout.

Supported parallelism: TP, SP, DP.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    ERNIE 4.5 specific. Change when porting.
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
    build_per_head_block_table,
    build_per_head_slot_mapping,
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

from .config import Ernie4_5Config

logger = logging.getLogger(__name__)


# =============================================================================
# Section 0: Interleaved RoPE -> rotate-half layout conversion
# <-- MODEL-SPECIFIC: ERNIE 4.5 uses interleaved RoPE
# =============================================================================


def interleaved_to_half_rotary_permutation(head_dim: int) -> torch.Tensor:
    """Return the head-dimension order that converts interleaved RoPE to rotate-half RoPE.

    Index ``i`` of the result is the source dimension for the new dimension ``i``.
    The even source dimensions go to the first half. The odd source dimensions
    go to the second half.

    Args:
        head_dim: The head dimension. It must be even.

    Returns:
        A ``torch.long`` tensor with shape ``[head_dim]``.

    Example::

        >>> interleaved_to_half_rotary_permutation(4)
        tensor([0, 2, 1, 3])
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim ({head_dim}) is odd. RoPE needs an even head_dim.")
    return torch.cat(
        [
            torch.arange(0, head_dim, 2, dtype=torch.long),
            torch.arange(1, head_dim, 2, dtype=torch.long),
        ]
    )


def permute_rotary_heads(
    weight: torch.Tensor, head_dim: int, dim: int = -1
) -> torch.Tensor:
    """Reorder the head dimension of each head in a Q or K projection weight.

    Apply this to the Q and K output features so that a rotate-half RoPE kernel
    gives the result of the interleaved RoPE.

    Args:
        weight: A 2D weight. The ``dim`` axis holds ``num_heads * head_dim``
            output features, head after head.
        head_dim: The head dimension.
        dim: The axis that holds the output features.

    Returns:
        The reordered weight, with the same shape as ``weight``.

    Example::

        q_weight = permute_rotary_heads(q_proj_weight, head_dim=128, dim=0)
    """
    dim = dim % weight.dim()
    size = weight.shape[dim]
    if size % head_dim != 0:
        raise ValueError(
            f"Feature size {size} is not a multiple of head_dim ({head_dim}). "
            "Check the head count and the head_dim of the config."
        )
    perm = interleaved_to_half_rotary_permutation(head_dim).to(weight.device)
    moved = weight.movedim(dim, -1)
    lead = moved.shape[:-1]
    out = moved.reshape(*lead, size // head_dim, head_dim)[..., perm]
    return out.reshape(*lead, size).movedim(-1, dim)


def ernie_fused_qkv_weight_loader(
    q_size: int,
    kv_size: int,
    head_dim: int,
    num_shards: int,
    num_kv_replicas: int = 1,
) -> SafetensorsWeightLoader:
    """Make a fused QKV loader that also converts the interleaved RoPE layout.

    The loader shards and fuses Q, K and V into a ``[hidden, q + 2 * kv]``
    parameter (storage transposed). Then it reorders the head dimension of the
    Q and K columns for each head. The V columns do not change.

    Args:
        q_size: The Q features per rank.
        kv_size: The K (and V) features per rank.
        head_dim: The head dimension.
        num_shards: The TP world size.
        num_kv_replicas: The number of ranks that hold the same KV head.

    Returns:
        A ``SafetensorsWeightLoader``.

    Example::

        set_weight_loader(
            attn.qkv_proj_weight,
            ernie_fused_qkv_weight_loader(256, 128, 128, num_shards=8, num_kv_replicas=4),
        )
    """
    base = fused_qkv_weight_loader(
        q_size=q_size,
        kv_size=kv_size,
        shard_dim=1,
        num_shards=num_shards,
        is_storage_transposed=True,
        num_kv_replicas=num_kv_replicas,
    )

    def transform(slices, rank: int) -> torch.Tensor:
        fused = base.load(slices, rank)  # [H, q + 2 * kv]
        q, k, v = torch.split(fused, [q_size, kv_size, kv_size], dim=1)
        return torch.cat(
            [
                permute_rotary_heads(q, head_dim, dim=1),
                permute_rotary_heads(k, head_dim, dim=1),
                v,
            ],
            dim=1,
        )

    return SafetensorsWeightLoader(transform=transform)


def get_checkpoint_mappings(
    num_hidden_layers: int,
    tie_word_embeddings: bool = True,
    prefix: str = "",
) -> dict[str, str | list[str]]:
    """Return the map from module parameter names to HF checkpoint keys.

    Parameters that do not have an entry use the same name as the checkpoint key.

    Args:
        num_hidden_layers: The number of decoder layers.
        tie_word_embeddings: If True, load ``lm_head.weight`` from the embedding.
        prefix: The prefix of the module parameter names, for example ``""``
            for ``Ernie4_5ForCausalLM``.

    Returns:
        A dict for ``SafetensorsCheckpoint.load_sharded``.

    Example::

        mappings = get_checkpoint_mappings(18, tie_word_embeddings=True)
    """
    mappings: dict[str, str | list[str]] = {}
    for layer_id in range(num_hidden_layers):
        ck = f"model.layers.{layer_id}"
        p = f"{prefix}model.layers.{layer_id}"
        # <-- MODEL-SPECIFIC: Separate Q, K, V -> fused QKV
        mappings[f"{p}.self_attn.qkv_proj_weight"] = [
            f"{ck}.self_attn.q_proj.weight",
            f"{ck}.self_attn.k_proj.weight",
            f"{ck}.self_attn.v_proj.weight",
        ]
        mappings[f"{p}.self_attn.o_proj_weight"] = f"{ck}.self_attn.o_proj.weight"
        mappings[f"{p}.input_layernorm.weight"] = f"{ck}.input_layernorm.weight"
        mappings[f"{p}.post_attention_layernorm.weight"] = (
            f"{ck}.post_attention_layernorm.weight"
        )
        mappings[f"{p}.mlp.gate_proj_weight"] = f"{ck}.mlp.gate_proj.weight"
        mappings[f"{p}.mlp.up_proj_weight"] = f"{ck}.mlp.up_proj.weight"
        mappings[f"{p}.mlp.down_proj_weight"] = f"{ck}.mlp.down_proj.weight"

    mappings[f"{prefix}model.embed_tokens.weight"] = "model.embed_tokens.weight"
    mappings[f"{prefix}model.norm.weight"] = "model.norm.weight"
    # <-- MODEL-SPECIFIC: Tied embeddings. The checkpoint has no lm_head key.
    mappings[f"{prefix}lm_head.weight"] = (
        "model.embed_tokens.weight" if tie_word_embeddings else "lm_head.weight"
    )
    return mappings


# =============================================================================
# Section 1: RMS Normalization
# =============================================================================


class Ernie4_5RMSNorm(nn.Module):
    """RMS normalization (no mean subtraction, no bias)."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            hidden_states, self.weight.shape, self.weight, self.variance_epsilon
        )


# =============================================================================
# Section 2: Rotary Position Embedding
# <-- MODEL-SPECIFIC: The cos/sin tables use the rotate-half layout. The Q/K
# weight reorder (Section 0) makes this equal to the HF interleaved RoPE.
# =============================================================================


class Ernie4_5RotaryEmbedding(nn.Module):
    """Rotary position embedding without scaling, in the rotate-half layout."""

    def __init__(self, config: Ernie4_5Config):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim

    def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
        return 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
                / self.head_dim
            )
        )

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin for the positions.

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
# =============================================================================


class Ernie4_5Attention(nn.Module):
    """Multi-head attention with TP head sharding.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads are sharded across TP ranks
    - KV heads are replicated when there are fewer KV heads than TP ranks (GQA)
    - Prefill: all-gather input -> QKV proj with fused RoPE -> attention ->
      O proj -> reduce-scatter
    - Decode: fused megakernel with TP all-reduce

    <-- MODEL-SPECIFIC:
    - GQA, no attention bias, no sinks, no sliding window
    - Interleaved RoPE. The loader reorders the Q/K weights (Section 0)
    """

    def __init__(self, config: Ernie4_5Config, layer_idx: int):
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
        if self.num_attention_heads % self.world_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) is not divisible "
                f"by the TP size ({self.world_size}). Use a TP size that divides it."
            )
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
        if self.world_size >= self.num_key_value_heads:
            if self.world_size % self.num_key_value_heads != 0:
                raise ValueError(
                    f"TP size ({self.world_size}) is not a multiple of "
                    f"num_key_value_heads ({self.num_key_value_heads}). Use a TP "
                    "size that is a multiple of the KV head count."
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
        """Attach the weight loaders.

        <-- MODEL-SPECIFIC: The QKV loader reorders the Q/K head dimensions for
        the interleaved RoPE.
        """
        set_weight_loader(
            self.qkv_proj_weight,
            ernie_fused_qkv_weight_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
                head_dim=self.head_dim,
                num_shards=self.world_size,
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

    # -- Forward dispatch -----------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ):
        """Select the prefill path or the decode path from the metadata."""
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
        """Prefill: full-sequence attention.

        Pipeline:
        1. QKV projection with fused RoPE   >>> PARALLELISM: TP sharded heads <<<
        2. KV cache update                  >>> PARALLELISM: per-rank cache <<<
        3. Flash attention or segmented attention
        4. Output projection                >>> PARALLELISM: reduce-scatter <<<
        """
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # -- Step 1: QKV projection with fused RoPE ---------------------------
        cos, sin = position_embeddings
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
            d_head=self.head_dim,
            cos_cache=cos.unsqueeze(0),
            sin_cache=sin.unsqueeze(0),
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

        # -- Step 2: Update KV cache ------------------------------------------
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size

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

        # -- Step 3: Attention ------------------------------------------------
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
            attn_output = NF.flash_attention(
                q.transpose(1, 2),  # [Nh, Dh, T]
                k.transpose(1, 2),  # [Nh, Dh, T]
                v,  # [Nh, T, Dh]
                scale=self.scaling,
                tp_q=False,
                tp_out=True,
            )  # [Nh, Dh, T]

        # -- Step 4: Output projection ----------------------------------------
        attn_output = attn_output.unsqueeze(0)  # [1, Nh, Dh, T]
        attn_output = NF.o_proj(attn_output, self.o_proj_weight, None)  # [1, T, H]
        attn_output = attn_output.squeeze(0)  # [T, H]

        # >>> PARALLELISM: Reduce-scatter to return to the SP layout <<<
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
        """Decode: fused megakernel for token generation.

        >>> PARALLELISM: The megakernel uses the TP-sharded weights. <<<
        The kernel writes the new K/V tokens into the cache (update_cache=True).
        With more than 1 KV head on each rank (for example TP 1), the kernel
        needs a per-head block table and per-head write slots.
        """
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

        # RoPE in the megakernel format: [T, Hd] -> [Hd/2, B, S]
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

        # >>> PARALLELISM: GQA cache addressing for the megakernel <<<
        # The kernel folds the 4D cache [num_blocks, nkh, block, Hd] into a
        # pool of num_blocks * nkh blocks. With nkh > 1, a 2D table sends the
        # call to the torch fallback. Thus give the per-head table and the
        # per-head slots of that pool. With nkh == 1, both stay 2D.
        nkh = self.num_key_value_heads_per_rank
        if nkh > 1:
            active_blocks_table = build_per_head_block_table(block_table, nkh).to(
                torch.int32
            )
        else:
            active_blocks_table = block_table
        kv_cache_update_idx = build_per_head_slot_mapping(
            slot_mapping, block_size, nkh, B
        )

        # <-- MODEL-SPECIFIC: rope_contiguous_layout=True (rotate half). The
        # reordered Q/K weights make this equal to the HF interleaved RoPE.
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
            softmax_scale=self.scaling,
            sink=None,
            update_cache=True,
            kv_cache_update_idx=kv_cache_update_idx,
            W_out=self.o_proj_weight,
            bias_out=None,
            transposed_out=False,
            out_in_sb=False,
        )

        # >>> PARALLELISM: TP all-reduce after the megakernel <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 4: MLP (Dense, SwiGLU)
# =============================================================================


class Ernie4_5MLP(nn.Module):
    """Dense MLP with TP intermediate sharding: down(silu(gate(x)) * up(x)).

    >>> PARALLELISM: TP <<<
    - gate_proj, up_proj: hidden -> intermediate/TP per rank
    - down_proj: intermediate/TP -> hidden per rank
    - Prefill: all-gather before, reduce-scatter after (SP)
    - Decode: all-reduce after
    """

    def __init__(self, config: Ernie4_5Config):
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
        # >>> PARALLELISM: All-gather from SP for the full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        )

        # >>> PARALLELISM: Combine the TP shards <<<
        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 5: Decoder Layer
# =============================================================================


class Ernie4_5DecoderLayer(nn.Module):
    """One decoder layer.

    hidden_states -> RMSNorm -> Attention -> residual -> RMSNorm -> MLP -> residual
    """

    def __init__(self, config: Ernie4_5Config, layer_idx: int):
        super().__init__()
        self.input_layernorm = Ernie4_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = Ernie4_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.self_attn = Ernie4_5Attention(config, layer_idx=layer_idx)
        self.mlp = Ernie4_5MLP(config)
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


class Ernie4_5Model(nn.Module):
    """ERNIE 4.5 dense backbone.

    >>> PARALLELISM: SP (sequence parallelism) <<<
    Prefill: the embedding scatters the tokens across the TP ranks. The model
    all-gathers the full sequence after the final norm.
    Decode: each rank processes all tokens.
    """

    def __init__(self, config: Ernie4_5Config):
        super().__init__()
        self.config = config

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        if config.vocab_size % self.world_size != 0:
            raise ValueError(
                f"vocab_size ({config.vocab_size}) is not divisible by the TP size "
                f"({self.world_size}). Use a TP size that divides it."
            )

        # >>> PARALLELISM: Vocab-sharded embedding <<<
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )

        self.layers = nn.ModuleList(
            [
                Ernie4_5DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Ernie4_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Ernie4_5RotaryEmbedding(config)

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

        # >>> PARALLELISM: SP - all-gather the full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states


# =============================================================================
# Section 7: Language Model Head
# =============================================================================


class Ernie4_5ForCausalLM(nn.Module):
    """ERNIE 4.5 dense model with the language modeling head.

    >>> PARALLELISM: Column-parallel linear for the LM head <<<
    <-- MODEL-SPECIFIC: Tied embeddings. lm_head.weight loads from
    model.embed_tokens.weight.
    """

    def __init__(self, config: Ernie4_5Config):
        super().__init__()
        self.config = config
        self.model = Ernie4_5Model(config)

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
                "ERNIE 4.5 does not support inputs_embeds. Give input_ids."
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
                f"Prompt length ({T}) is not compatible with SP. It must be larger "
                f"than world_size ({self.world_size}) and a multiple of it."
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

        # >>> PARALLELISM: Gather the sharded logits for logprobs <<<
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
        config = Ernie4_5Config.from_configs(hf_config, neuron_config)
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
                    f"KV cache for layer {layer_name} is not initialized. Call "
                    "bind_kv_cache with a cache for each layer."
                )
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # -- Weight Loading -------------------------------------------------------

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load the TP-sharded weights from the HF checkpoint."""
        mappings = get_checkpoint_mappings(
            len(self.model.layers), self.config.tie_word_embeddings
        )

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        result = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, mappings, device
        )
        if result.unexpected_keys:
            logger.warning(
                "ERNIE 4.5 checkpoint has %d keys that no parameter uses: %s",
                len(result.unexpected_keys),
                result.unexpected_keys[:8],
            )
        rank_sharded = result.state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self.load_state_dict(rank_sharded, strict=False, assign=True)
