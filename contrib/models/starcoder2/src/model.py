# SPDX-License-Identifier: Apache-2.0
"""
StarCoder2 BF16 Implementation
==============================

Implementation of StarCoder2 for the native Neuron route. The structure comes
from the SmolLM3 / Llama3 dense models. The bias pattern comes from GPT-OSS BF16.

StarCoder2 is different from Llama in these items:

- LayerNorm with a bias, in place of RMSNorm.
- A non-gated MLP: ``c_proj(gelu_tanh(c_fc(x)))``. The MLP kernel runs it
  with ``skip_gate=True``.
- A bias on Q, K, V, O, ``c_fc`` and ``c_proj``.

Supported parallelism: TP, SP, DP.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Parallelism code from the reference models.
  # <-- MODEL-SPECIFIC: ...    StarCoder2-specific code.

PARALLELISM SHARDING:
  TP-only:  Attention heads sharded, MLP intermediate sharded
  Row-parallel biases (O, c_proj) are divided by the TP size. Each rank adds
  its part, and the reduce-scatter or the all-reduce gives the full bias.
"""

import logging

import torch
import torch.nn.functional as F
from nkilib.core.utils.common_types import ActFnType
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
from vllm_neuron.functional.mlp import _can_use_kernel as _mlp_kernel_accepts
import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    fused_qkv_weight_loader,
    scaled_bias_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import Starcoder2Config

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1: Weight loaders
# <-- MODEL-SPECIFIC: StarCoder2 has a bias on Q, K and V.
# =============================================================================


def fused_qkv_bias_loader(
    q_size: int,
    kv_size: int,
    num_kv_replicas: int = 1,
) -> SafetensorsWeightLoader:
    """Make a loader that fuses the Q, K and V biases of one TP rank.

    The rank-to-KV-head map is the same as in ``fused_qkv_weight_loader``.
    Rank ``r`` gets Q heads ``[r * q_size, (r + 1) * q_size)`` and KV head
    group ``r // num_kv_replicas``.

    Args:
        q_size: Number of Q bias elements on one rank (Q heads per rank x head_dim).
        kv_size: Number of K (or V) bias elements on one rank.
        num_kv_replicas: Number of ranks that hold a copy of the same KV heads.

    Returns:
        SafetensorsWeightLoader that returns a ``[q_size + 2 * kv_size]`` tensor.

    Example:
        >>> loader = fused_qkv_bias_loader(q_size=768, kv_size=128, num_kv_replicas=2)
        >>> set_weight_loader(attn.qkv_proj_bias, loader)
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 3
        q_slice, k_slice, v_slice = slices
        kv_rank = rank // num_kv_replicas
        q_tensor = q_slice[rank * q_size : (rank + 1) * q_size]
        k_tensor = k_slice[kv_rank * kv_size : (kv_rank + 1) * kv_size]
        v_tensor = v_slice[kv_rank * kv_size : (kv_rank + 1) * kv_size]
        return torch.cat([q_tensor, k_tensor, v_tensor], dim=0)

    return SafetensorsWeightLoader(transform=transform)


# =============================================================================
# Section 2: Layer Normalization
# <-- MODEL-SPECIFIC: StarCoder2 uses nn.LayerNorm (mean subtraction + bias)
# =============================================================================


class Starcoder2LayerNorm(nn.Module):
    """LayerNorm with weight and bias. This is the same math as ``nn.LayerNorm``."""

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
# Section 3: Rotary Position Embedding (default RoPE, rotate_half)
# =============================================================================


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
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
    """Apply rotary position embeddings to the query and key tensors.

    Args:
        q: [Nh, T, Dh], k: [Nkv, T, Dh]
        cos: [T, Dh], sin: [T, Dh]

    Returns:
        Rotated (q, k) tensors
    """
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    return _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)


class Starcoder2RotaryEmbedding(nn.Module):
    """Default RoPE (no scaling) with the ``rope_theta`` of the config."""

    def __init__(self, config: Starcoder2Config):
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
        """Compute the cos/sin embeddings for the positions.

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
# Section 4: Attention
# Mixed: PARALLELISM (TP head sharding, SP, collectives) +
#        MODEL-SPECIFIC (GQA, QKV and O bias)
# =============================================================================


class Starcoder2Attention(nn.Module):
    """GQA attention with TP head sharding and a bias on Q, K, V and O.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads are sharded across TP ranks
    - KV heads are replicated when there are fewer KV heads than TP ranks
    - Prefill: all-gather input -> QKV proj -> attention -> O proj -> reduce-scatter
    - Decode: fused decode attention kernel, then TP all-reduce

    <-- MODEL-SPECIFIC:
    - QKV bias: fused per rank, ``[q_size + 2 * kv_size]``
    - O bias: divided by the TP size (row-parallel)
    """

    def __init__(self, config: Starcoder2Config, layer_idx: int):
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

        if self.num_attention_heads % self.world_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) is not a "
                f"multiple of the TP size ({self.world_size}). Use a TP size "
                "that divides the number of attention heads."
            )

        # >>> PARALLELISM: Head sharding calculation <<<
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
        if self.world_size >= self.num_key_value_heads:
            if self.world_size % self.num_key_value_heads != 0:
                raise ValueError(
                    f"TP size ({self.world_size}) is not a multiple of "
                    f"num_key_value_heads ({self.num_key_value_heads})."
                )
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            if self.num_key_value_heads % self.world_size != 0:
                raise ValueError(
                    f"num_key_value_heads ({self.num_key_value_heads}) is not a "
                    f"multiple of the TP size ({self.world_size})."
                )
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
        # <-- MODEL-SPECIFIC: QKV and O biases
        self.qkv_proj_bias = nn.Parameter(torch.zeros(qkv_size, dtype=self.dtype))
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_proj_in_features, self.hidden_size, dtype=self.dtype)
        )
        self.o_proj_bias = nn.Parameter(torch.zeros(self.hidden_size, dtype=self.dtype))

        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        # bind_kv_cache() sets the KV caches.
        self.k_cache = None
        self.v_cache = None

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach the weight loaders that shard the checkpoint tensors."""
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
                shard_dim=1,
                num_shards=self.world_size,
                is_storage_transposed=True,
                num_kv_replicas=self.num_kv_replicas,
            ),
        )
        set_weight_loader(
            self.qkv_proj_bias,
            fused_qkv_bias_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
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
        # >>> PARALLELISM: Row-parallel bias, divided by the TP size <<<
        set_weight_loader(self.o_proj_bias, scaled_bias_loader(scale=self.world_size))

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
                hidden_states, positions, position_embeddings, attn_metadata
            )

        # >>> PARALLELISM: All-gather from SP before attention <<<
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(
            hidden_states, positions, position_embeddings, attn_metadata
        )

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """Prefill: full-sequence attention with the flash attention kernel.

        Pipeline:
        1. QKV projection with bias   >>> PARALLELISM: TP sharded heads <<<
        2. RoPE
        3. KV cache update            >>> PARALLELISM: per-rank cache <<<
        4. Flash attention (causal)
        5. O projection with bias     >>> PARALLELISM: reduce-scatter <<<
        """
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, _ = hidden_states.shape

        # -- Step 1: QKV projection -------------------------------------------
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=self.qkv_proj_bias.unsqueeze(0),
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

        # -- Step 2: RoPE -----------------------------------------------------
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # -- Step 3: KV cache update ------------------------------------------
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

        # -- Step 4: Attention ------------------------------------------------
        # <-- MODEL-SPECIFIC: Causal attention. The factory makes sure that
        # max_model_len is not larger than the sliding window.
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

        # -- Step 5: O projection with the rank part of the bias -------------
        attn_output = NF.o_proj(
            attn_output.unsqueeze(0),
            self.o_proj_weight,
            self.o_proj_bias.unsqueeze(0),
        ).squeeze(0)  # [T, H]

        # >>> PARALLELISM: Reduce-scatter to return to SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
    ):
        """Decode: fused decode attention kernel (QKV, RoPE, cache, attention, O).

        >>> PARALLELISM: The kernel uses the TP-sharded weights. <<<
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

        # RoPE for the kernel format: [T, Hd] -> [Hd/2, B, S]
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

        # The kernel writes the new K/V tokens into the cache (update_cache=True).
        output = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            rmsnorm_X_enabled=False,
            W_qkv=self.qkv_proj_weight,
            bias_qkv=self.qkv_proj_bias.unsqueeze(0),
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
            bias_out=self.o_proj_bias.unsqueeze(0),
            transposed_out=False,
            out_in_sb=False,
        )

        # >>> PARALLELISM: TP all-reduce after the kernel <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 5: MLP (non-gated, GELU tanh, with bias)
# =============================================================================

_SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2 / pi)


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """GELU with the tanh approximation, written with elementary ops.

    This is the formula of ``F.gelu(x, approximate="tanh")`` and of HF
    ``gelu_pytorch_tanh``. The torch fallback below calls it in place of
    ``F.gelu``: on the Neuron route that is not ``neuron_native``,
    ``libtorch_neuronx_lite`` replaces ``torch._C._nn.gelu`` with a custom
    function that ``torch.compile`` cannot trace. ``tanh``, ``mul`` and
    ``add`` trace on every route.
    """
    return 0.5 * x * (1.0 + torch.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)))


class Starcoder2MLP(nn.Module):
    """Non-gated MLP with TP intermediate sharding.

    >>> PARALLELISM: TP <<<
    - c_fc: hidden -> intermediate/TP per rank (bias sharded the same way)
    - c_proj: intermediate/TP -> hidden per rank (bias divided by the TP size)
    - Prefill: all-gather before -> compute -> reduce-scatter after (SP)
    - Decode: compute -> all-reduce
    - The reduction adds the partial sums in FP32.

    <-- MODEL-SPECIFIC: ``c_proj(gelu_tanh(c_fc(x) + b_fc)) + b_proj``. The MLP
    kernel runs this with ``skip_gate=True``. The kernel does not read the gate
    weight then, but the API needs a tensor, so the up weight is given again.
    When the NKI kernel cannot run this shape (or cannot run at all, as in CPU
    mode), ``_torch_mlp`` does the same arithmetic as the kernel, without
    ``F.gelu``.
    """

    def __init__(self, config: Starcoder2Config):
        super().__init__()

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        if config.intermediate_size % self.world_size != 0:
            raise ValueError(
                f"intermediate_size ({config.intermediate_size}) is not a "
                f"multiple of the TP size ({self.world_size})."
            )

        self.hidden_size = config.hidden_size
        self.intermediate_size_per_rank = config.intermediate_size // self.world_size

        self.c_fc_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_size_per_rank,
                dtype=config.torch_dtype,
            )
        )
        self.c_fc_bias = nn.Parameter(
            torch.zeros(self.intermediate_size_per_rank, dtype=config.torch_dtype)
        )
        self.c_proj_weight = nn.Parameter(
            torch.empty(
                self.intermediate_size_per_rank,
                config.hidden_size,
                dtype=config.torch_dtype,
            )
        )
        self.c_proj_bias = nn.Parameter(
            torch.zeros(config.hidden_size, dtype=config.torch_dtype)
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """>>> PARALLELISM: TP sharding of the MLP weights and biases. <<<"""
        set_weight_loader(
            self.c_fc_weight,
            sharding_weight_loader(
                shard_dim=1,
                shard_size=self.intermediate_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        set_weight_loader(
            self.c_fc_bias,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.intermediate_size_per_rank,
                num_shards=self.world_size,
            ),
        )
        set_weight_loader(
            self.c_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.intermediate_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        set_weight_loader(self.c_proj_bias, scaled_bias_loader(scale=self.world_size))

    def _torch_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute the MLP of this rank with the arithmetic of the kernel.

        The kernel adds the c_fc bias and applies GELU in FP32. It rounds the
        intermediate to the input dtype, and it accumulates c_proj in FP32.
        ``NF.mlp`` falls back to BF16 for each step, which adds a rounding error
        at each step. This path keeps the CPU result as accurate as the kernel.

        Returns:
            The FP32 partial sum of this rank, with the scaled c_proj bias.
        """
        up = torch.addmm(
            self.c_fc_bias.float(), hidden_states.float(), self.c_fc_weight.float()
        )
        act = _gelu_tanh(up).to(hidden_states.dtype)
        return torch.addmm(
            self.c_proj_bias.float(), act.float(), self.c_proj_weight.float()
        )

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        # >>> PARALLELISM: All-gather from SP for the full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        out_dtype = hidden_states.dtype
        # <-- MODEL-SPECIFIC: The TKG kernel reads the dtype of the gate bias
        # when a projection bias is present, also with skip_gate. Give the up
        # bias as the gate bias. The kernel does not add it.
        fc_bias = self.c_fc_bias.unsqueeze(0)
        proj_bias = self.c_proj_bias.unsqueeze(0)
        # Call NF.mlp only when it runs the NKI kernel. When the kernel refuses
        # the shape (for example the prefill kernel at an intermediate size per
        # rank above 4096, so TP 1 and TP 2), NF.mlp would take its own torch
        # fallback, which calls F.gelu inside the traced graph. _torch_mlp uses
        # _gelu_tanh instead, and keeps the kernel's FP32 accumulation.
        if _mlp_kernel_accepts(
            hidden_states, self.c_fc_weight, fc_bias, fc_bias, proj_bias
        ):
            output = NF.mlp(
                hidden_states,
                self.c_fc_weight,
                self.c_fc_weight,
                self.c_proj_weight,
                skip_gate=True,
                act_fn=ActFnType.GELU_Tanh_Approx,
                gate_b=fc_bias,
                up_b=fc_bias,
                down_b=proj_bias,
            )
        else:
            output = self._torch_mlp(hidden_states)

        # >>> PARALLELISM: Combine TP shards <<<
        # Add the TP partial sums in FP32. A BF16 sum rounds each partial sum
        # and each step of the reduction.
        if self.world_size > 1:
            output = output.float()
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                self.tp_group.all_reduce(output)

        return output.to(out_dtype)


# =============================================================================
# Section 6: Decoder Layer
# =============================================================================


class Starcoder2DecoderLayer(nn.Module):
    """One decoder layer.

    hidden_states -> LayerNorm -> Attention -> residual -> LayerNorm -> MLP -> residual
    """

    def __init__(self, config: Starcoder2Config, layer_idx: int):
        super().__init__()
        self.input_layernorm = Starcoder2LayerNorm(
            config.hidden_size, config.norm_epsilon, config.torch_dtype
        )
        self.post_attention_layernorm = Starcoder2LayerNorm(
            config.hidden_size, config.norm_epsilon, config.torch_dtype
        )
        self.self_attn = Starcoder2Attention(config, layer_idx=layer_idx)
        self.mlp = Starcoder2MLP(config)
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


class Starcoder2Model(nn.Module):
    """StarCoder2 transformer backbone.

    >>> PARALLELISM: SP (Sequence Parallelism) <<<
    Prefill: the embedding scatters the tokens across TP ranks. After the
    final norm, an all-gather makes the full sequence again.
    Decode: no SP. All ranks process all tokens.
    """

    def __init__(self, config: Starcoder2Config):
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
                Starcoder2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = Starcoder2LayerNorm(
            config.hidden_size, config.norm_epsilon, config.torch_dtype
        )
        self.rotary_emb = Starcoder2RotaryEmbedding(config)

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

        # >>> PARALLELISM: SP - all-gather to make the full sequence again <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states


# =============================================================================
# Section 8: Language Model Head
# =============================================================================


class Starcoder2ForCausalLM(nn.Module):
    """StarCoder2 model with a language modeling head.

    >>> PARALLELISM: Column-parallel linear for the LM head <<<

    <-- MODEL-SPECIFIC: StarCoder2-3B ties the LM head to the embedding. The
    loader reads ``lm_head.weight`` from the embedding key when the weights are tied.
    """

    def __init__(self, config: Starcoder2Config):
        super().__init__()
        self.config = config
        self.model = Starcoder2Model(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        if config.vocab_size % self.world_size != 0:
            raise ValueError(
                f"vocab_size ({config.vocab_size}) is not a multiple of the TP "
                f"size ({self.world_size})."
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
                "StarCoder2 does not support inputs_embeds. Give input_ids."
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
                f"Prompt length ({T}) must be a multiple of the TP size "
                f"({self.world_size}) and larger than it for SP. Use a "
                "prefill bucket that is a multiple of the TP size."
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

        # >>> PARALLELISM: Gather sharded logits for logprobs <<<
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
        config = Starcoder2Config.from_configs(hf_config, neuron_config)
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
                    # The factory makes sure that max_model_len is not larger
                    # than the sliding window, so the cache is full attention.
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
                    f"KV cache for layer {layer_name} is not set. "
                    "Initialize the KV cache before the forward pass."
                )
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # -- Weight Loading -------------------------------------------------------

    @staticmethod
    def checkpoint_mappings(
        num_hidden_layers: int, tie_word_embeddings: bool
    ) -> dict[str, str | list[str]]:
        """Return the map from model parameter names to HF checkpoint keys.

        A parameter that is not in the map uses its own name as the key.
        """
        mappings: dict[str, str | list[str]] = {}
        for layer_id in range(num_hidden_layers):
            prefix = f"model.layers.{layer_id}"
            attn = f"{prefix}.self_attn"
            mappings[f"{attn}.qkv_proj_weight"] = [
                f"{attn}.q_proj.weight",
                f"{attn}.k_proj.weight",
                f"{attn}.v_proj.weight",
            ]
            mappings[f"{attn}.qkv_proj_bias"] = [
                f"{attn}.q_proj.bias",
                f"{attn}.k_proj.bias",
                f"{attn}.v_proj.bias",
            ]
            mappings[f"{attn}.o_proj_weight"] = f"{attn}.o_proj.weight"
            mappings[f"{attn}.o_proj_bias"] = f"{attn}.o_proj.bias"

            for norm in ("input_layernorm", "post_attention_layernorm"):
                mappings[f"{prefix}.{norm}.weight"] = f"{prefix}.{norm}.weight"
                mappings[f"{prefix}.{norm}.bias"] = f"{prefix}.{norm}.bias"

            for proj in ("c_fc", "c_proj"):
                mappings[f"{prefix}.mlp.{proj}_weight"] = f"{prefix}.mlp.{proj}.weight"
                mappings[f"{prefix}.mlp.{proj}_bias"] = f"{prefix}.mlp.{proj}.bias"

        if tie_word_embeddings:
            mappings["lm_head.weight"] = "model.embed_tokens.weight"
        return mappings

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load the weights from the checkpoint and cast them to BF16."""
        mappings = self.checkpoint_mappings(
            len(self.model.layers), self.config.tie_word_embeddings
        )

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, mappings, device
        ).state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        expected = {name for name, _ in self.named_parameters()}
        missing = sorted(expected - set(rank_sharded))
        if missing:
            raise RuntimeError(
                f"StarCoder2 checkpoint has no tensor for {len(missing)} "
                f"parameters, for example {missing[:5]}. Make sure that the "
                "checkpoint is a complete StarCoder2 checkpoint."
            )

        self.load_state_dict(rank_sharded, strict=False, assign=True)
