# SPDX-License-Identifier: Apache-2.0
"""
Seed-OSS BF16 Implementation
============================

Implementation of Seed-OSS (``SeedOssForCausalLM``) for the Neuron backend.
Adapted from the SmolLM3 BF16 model, which is a dense Llama-style model.

Seed-OSS is a Llama-style dense decoder with one difference: the Q, K, and V
projections have a bias (``attention_bias=True``). The output projection and
the MLP have no bias. All layers apply RoPE.

Supported parallelism: TP, SP.
Reference geometry: Seed-OSS-36B (GQA 80Q/8KV, head_dim 128, SiLU, untied
embeddings).

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    Seed-OSS-specific. Change when porting.

KERNEL DISPATCH LOG:
  Each traced graph writes one ``[seed_oss_dispatch]`` line for each main-path
  kernel of the first and the last decoder layer. The line records the phase,
  the operation, the TP rank, and if the NKI kernel or the torch fallback runs.
  The lines come from the same eligibility functions that the NF wrappers use.
"""

import logging
import os
import socket

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
from vllm_neuron.functional.attention.attention_cte import (
    _can_use_flash_attention_kernel,
)
from vllm_neuron.functional.attention.attention_decode import (
    _can_use_attention_block_kernel,
)
from vllm_neuron.functional.attention.attention_segmented_cte import (
    _can_use_segmented_kernel,
)
from vllm_neuron.functional.attention.o_proj import (
    _can_use_kernel as _can_use_o_proj_kernel,
)
from vllm_neuron.functional.attention.qkv import (
    NormType,
    QKVOutputLayout,
    _can_use_qkv_kernel,
)
from vllm_neuron.functional.mlp import _can_use_kernel as _can_use_mlp_kernel
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

from .config import SeedOssConfig

logger = logging.getLogger(__name__)

DISPATCH_LOG_TAG = "[seed_oss_dispatch]"


# This must stay behind assume_constant_result. A bare logger call in a traced
# forward is a hard Dynamo failure under fullgraph. The decorator runs the call
# off-graph at trace time. The TP rank is read here, off-graph, so that the
# traced graph has no rank-specific guard.
@torch._dynamo.assume_constant_result
def _log_dispatch(phase: str, op: str, kernel: bool, layer: int) -> None:
    """Write one greppable line that names the kernel decision for one op.

    Grep for ``[seed_oss_dispatch]`` to see the decision of each traced graph.

    Example:
        >>> _log_dispatch("prefill", "qkv_proj", True, 0)
    """
    try:
        tp_group = get_tp_group()
        rank = tp_group.rank_in_group
        world = tp_group.world_size
    except Exception:
        rank, world = -1, -1
    logger.info(
        "%s phase=%s op=%s kernel=%s layer=%d rank=%d world=%d host=%s pid=%d",
        DISPATCH_LOG_TAG,
        phase,
        op,
        kernel,
        layer,
        rank,
        world,
        socket.gethostname(),
        os.getpid(),
    )


# =============================================================================
# Section 1: RMS Normalization
# <-- MODEL-SPECIFIC: Seed-OSS uses standard RMSNorm (weight * x_hat)
# =============================================================================


class SeedOssRMSNorm(nn.Module):
    """RMS Normalization (no mean subtraction, no bias)."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            hidden_states, self.weight.shape, self.weight, self.variance_epsilon
        )


# =============================================================================
# Section 2: Rotary Position Embedding (default RoPE, rotate_half)
# =============================================================================


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """<-- MODEL-SPECIFIC: rotate_half RoPE."""
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
    cos = cos.unsqueeze(0)  # [1, T, Dh]
    sin = sin.unsqueeze(0)  # [1, T, Dh]
    return _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)


class SeedOssRotaryEmbedding(nn.Module):
    """Rotary Position Embedding with the default RoPE type (no scaling).

    <-- MODEL-SPECIFIC: Seed-OSS uses rope_type "default" with rope_theta 1e7.
    The default type has an attention scaling of 1.0.
    """

    def __init__(self, config: SeedOssConfig):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim

    def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
                / self.head_dim
            )
        )
        return inv_freq

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin embeddings for the given positions.

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
#        MODEL-SPECIFIC (GQA config, QKV bias, no sinks, no sliding window)
# =============================================================================


def fused_qkv_bias_loader(
    q_size: int,
    kv_size: int,
    num_kv_replicas: int = 1,
) -> SafetensorsWeightLoader:
    """Make a loader that shards and fuses the Q, K, and V biases.

    The layout agrees with ``fused_qkv_weight_loader``: rank ``r`` gets Q shard
    ``r`` and KV shard ``r // num_kv_replicas``, in the order [Q, K, V].

    Args:
        q_size: Size of the Q bias for each rank.
        kv_size: Size of the K bias (and of the V bias) for each rank.
        num_kv_replicas: Number of ranks that share one KV shard.

    Returns:
        SafetensorsWeightLoader for the fused 1D QKV bias.

    Example:
        >>> loader = fused_qkv_bias_loader(q_size=1280, kv_size=128)
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 3, "fused_qkv_bias_loader expects [Q, K, V] slices"
        q_slice, k_slice, v_slice = slices
        kv_rank = rank // num_kv_replicas
        q = q_slice[rank * q_size : (rank + 1) * q_size]
        k = k_slice[kv_rank * kv_size : (kv_rank + 1) * kv_size]
        v = v_slice[kv_rank * kv_size : (kv_rank + 1) * kv_size]
        return torch.cat([q, k, v], dim=0)

    return SafetensorsWeightLoader(transform=transform)


class SeedOssAttention(nn.Module):
    """Multi-head attention with TP head sharding and a fused QKV bias.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads are sharded across TP ranks
    - KV heads are replicated when there are fewer KV heads than TP ranks
    - Prefill: all-gather input -> QKV proj -> attention -> O proj -> reduce-scatter
    - Decode: fused attention block kernel, then TP all-reduce

    <-- MODEL-SPECIFIC:
    - GQA with separate Q and KV head counts
    - Bias on Q, K, and V (attention_bias). No bias on O.
    - No sinks, no sliding window
    """

    def __init__(
        self, config: SeedOssConfig, layer_idx: int, log_dispatch: bool = False
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5
        self.has_qkv_bias = bool(config.attention_bias)
        self.log_dispatch = log_dispatch

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        if self.num_attention_heads % self.world_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) is not divisible "
                f"by the TP size ({self.world_size}). Use a TP size of 2, 4, or 8."
            )

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
        # <-- MODEL-SPECIFIC: fused [Q, K, V] bias for this rank
        if self.has_qkv_bias:
            self.qkv_proj_bias = nn.Parameter(torch.empty(qkv_size, dtype=self.dtype))
        else:
            self.qkv_proj_bias = None
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
        """Attach weight loaders that change checkpoint tensors to parameters.

        >>> PARALLELISM: Weight loaders do the TP sharding of checkpoint tensors.
        <-- MODEL-SPECIFIC: Seed-OSS keeps separate Q, K, V weights and biases.
        """
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
        if self.qkv_proj_bias is not None:
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

    def _qkv_bias_2d(self) -> torch.Tensor | None:
        """Return the fused QKV bias as [1, qkv_size], or None."""
        if self.qkv_proj_bias is None:
            return None
        return self.qkv_proj_bias.unsqueeze(0)

    # -- Forward dispatch -----------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ):
        """Send the call to the prefill path or to the decode path.

        >>> PARALLELISM: Dispatch logic <<<
        - Prefill: all-gather for SP before attention, reduce-scatter after
        - Decode: the fused kernel does the attention on the full batch
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
        1. QKV projection with bias  >>> PARALLELISM: TP sharded heads <<<
        2. RoPE
        3. KV cache update           >>> PARALLELISM: per-rank cache <<<
        4. Flash or segmented attention
        5. Output projection         >>> PARALLELISM: reduce-scatter after O proj <<<
        """
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # -- Step 1: QKV Projection (with bias) -------------------------------
        hidden_3d = hidden_states.unsqueeze(0)
        if self.log_dispatch:
            _log_dispatch(
                "prefill",
                "qkv_proj",
                _can_use_qkv_kernel(
                    hidden_3d,
                    self.qkv_proj_weight,
                    None,
                    None,
                    None,
                    QKVOutputLayout.BSD,
                    NormType.NO_NORM,
                    None,
                ),
                self.layer_idx,
            )
        qkv = NF.qkv_proj(
            hidden=hidden_3d,
            qkv_weights=self.qkv_proj_weight,
            bias=self._qkv_bias_2d(),
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

        # -- Step 2: Apply RoPE -----------------------------------------------
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # -- Step 3: Update KV Cache -----------------------------------------
        # >>> PARALLELISM: Cache is per-rank (TP sharded KV heads) <<<
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
        # <-- MODEL-SPECIFIC: No sinks, no sliding window (standard causal attention)
        if kv_segment_size:
            # Segmented prefill: attend over KV cache segments
            if self.log_dispatch:
                _log_dispatch(
                    "prefill",
                    "segmented_attention",
                    _can_use_segmented_kernel(
                        q,
                        kv_segment_size=kv_segment_size,
                        block_size=block_size,
                        tp_q=True,
                        sliding_window=None,
                    ),
                    self.layer_idx,
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
            )  # [Nh, Dh, T]
        else:
            # Full prefill: standard flash attention
            k = k.repeat_interleave(self.num_key_value_groups, dim=0)
            v = v.repeat_interleave(self.num_key_value_groups, dim=0)

            q_flash = q.transpose(1, 2)  # [Nh, Dh, T]
            k_flash = k.transpose(1, 2)  # [Nh, Dh, T]
            v_flash = v  # [Nh, T, Dh]

            if self.log_dispatch:
                _log_dispatch(
                    "prefill",
                    "flash_attention",
                    _can_use_flash_attention_kernel(v=v_flash),
                    self.layer_idx,
                )
            attn_output = NF.flash_attention(
                q_flash,
                k_flash,
                v_flash,
                scale=self.scaling,
                tp_q=False,
                tp_out=True,
            )  # [Nh, Dh, T]

        # -- Step 5: Output Projection (no bias) ------------------------------
        attn_output = attn_output.unsqueeze(0)  # [1, Nh, Dh, T]
        if self.log_dispatch:
            _log_dispatch(
                "prefill",
                "o_proj",
                _can_use_o_proj_kernel(attn_output, self.o_proj_weight),
                self.layer_idx,
            )
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
        """Decode: fused attention block kernel for token generation.

        >>> PARALLELISM: The kernel uses TP-sharded weights. All-reduce after. <<<

        <-- MODEL-SPECIFIC: The fused QKV bias goes to the kernel as bias_qkv.
        No sink tokens, no sliding window, no output bias.
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

        # Prepare RoPE for the kernel format: [T, Hd] -> [Hd/2, B, S]
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

        # <-- MODEL-SPECIFIC: Standard causal mask, no sliding window
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

        active_blocks_table = block_table

        if self.log_dispatch:
            _log_dispatch(
                "decode",
                "attention_decode",
                _can_use_attention_block_kernel(
                    X=X,
                    V_cache=self.v_cache,
                    active_blocks_table=active_blocks_table,
                    attention_dp=1,
                ),
                self.layer_idx,
            )

        # >>> PARALLELISM: Fused kernel with TP-sharded weights <<<
        # The kernel writes the new K/V tokens into the cache (update_cache=True).
        output = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            rmsnorm_X_enabled=False,
            W_qkv=self.qkv_proj_weight,
            bias_qkv=self._qkv_bias_2d(),
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
# Section 4: MLP (Dense)
# Mixed: PARALLELISM (TP intermediate sharding, SP collectives) +
#        MODEL-SPECIFIC (SiLU activation, gate/up/down structure, no bias)
# =============================================================================


class SeedOssMLP(nn.Module):
    """Dense SwiGLU MLP with TP intermediate sharding.

    >>> PARALLELISM: TP <<<
    - gate_proj, up_proj: hidden -> intermediate/TP per rank
    - down_proj: intermediate/TP -> hidden per rank
    - Prefill: all-gather before -> compute -> reduce-scatter after (SP)
    - Decode: compute -> all-reduce

    <-- MODEL-SPECIFIC: down(silu(gate(x)) * up(x)), no bias (mlp_bias=False)
    """

    def __init__(
        self, config: SeedOssConfig, layer_idx: int = 0, log_dispatch: bool = False
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.log_dispatch = log_dispatch

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        if config.intermediate_size % self.world_size != 0:
            raise ValueError(
                f"intermediate_size ({config.intermediate_size}) is not divisible "
                f"by the TP size ({self.world_size}). Use a different TP size."
            )

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
        """
        >>> PARALLELISM: SP all-gather/reduce-scatter for prefill, all-reduce for decode <<<
        """
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        if self.log_dispatch:
            _log_dispatch(
                "prefill" if is_prefill else "decode",
                "mlp",
                _can_use_mlp_kernel(hidden_states, self.gate_proj_weight),
                self.layer_idx,
            )
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


class SeedOssDecoderLayer(nn.Module):
    """Single transformer decoder layer.

    Architecture (MODEL-SPECIFIC):
        hidden_states -> RMSNorm -> Attention -> residual -> RMSNorm -> MLP -> residual

    Parallelism (TP + SP):
        - Input arrives in SP layout (T/world_size tokens per rank) during prefill
        - Attention and MLP all-gather before and reduce-scatter after
        - The residual add operates on the SP-local slice
    """

    def __init__(
        self, config: SeedOssConfig, layer_idx: int, log_dispatch: bool = False
    ):
        super().__init__()
        self.input_layernorm = SeedOssRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = SeedOssRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.self_attn = SeedOssAttention(
            config, layer_idx=layer_idx, log_dispatch=log_dispatch
        )
        self.mlp = SeedOssMLP(config, layer_idx=layer_idx, log_dispatch=log_dispatch)
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


class SeedOssModel(nn.Module):
    """Seed-OSS transformer backbone.

    >>> PARALLELISM: SP (Sequence Parallelism) <<<
    During prefill:
    - After embedding: chunk the sequence across TP ranks (T/world_size each)
    - After all layers: all-gather to make the full sequence again
    During decode:
    - No SP. All ranks process all tokens.
    """

    def __init__(self, config: SeedOssConfig):
        super().__init__()
        self.config = config

        # >>> PARALLELISM: TP group for SP <<<
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

        # Write the dispatch log for the first and the last layer. All layers
        # have the same geometry, so these two layers show the decision of all.
        last_layer = config.num_hidden_layers - 1
        self.layers = nn.ModuleList(
            [
                SeedOssDecoderLayer(
                    config,
                    layer_idx,
                    log_dispatch=layer_idx in (0, last_layer),
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = SeedOssRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = SeedOssRotaryEmbedding(config)

        # Weight loaders: shard the embedding on the vocab dim
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

        # >>> PARALLELISM: VocabDimShardedEmbedding handles SP internally <<<
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
# Section 7: Language Model Head
# =============================================================================


class SeedOssForCausalLM(nn.Module):
    """Seed-OSS model with a language modeling head.

    >>> PARALLELISM: Column-parallel linear for the LM head <<<

    <-- MODEL-SPECIFIC: Seed-OSS has untied embeddings. The LM head loads
    ``lm_head.weight``. A tied variant loads ``model.embed_tokens.weight``.
    """

    def __init__(self, config: SeedOssConfig):
        super().__init__()
        self.config = config
        self.model = SeedOssModel(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        if config.vocab_size % self.world_size != 0:
            raise ValueError(
                f"vocab_size ({config.vocab_size}) is not divisible by the TP size "
                f"({self.world_size}). Use a different TP size."
            )

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
                "Seed-OSS on Neuron does not support inputs_embeds. Use input_ids."
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
                f"Prompt length ({T}) must be larger than world_size "
                f"({self.world_size}) and divisible by it for SP."
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

            rejection_sampled_tokens = rejection_sampler(
                spec_decode_metadata, sampled_tokens
            )
            return rejection_sampled_tokens

        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = SeedOssConfig.from_configs(hf_config, neuron_config)
        return cls(config)

    # -- KV Cache Management --------------------------------------------------

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
                raise RuntimeError(
                    f"KV cache for layer {layer_name} is not initialized. "
                    "Call bind_kv_cache with a cache for each layer."
                )
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # -- Weight Loading -------------------------------------------------------

    def checkpoint_mappings(self) -> dict[str, str | list[str]]:
        """Return the map from model parameter name to checkpoint key(s)."""
        mappings: dict[str, str | list[str]] = dict()
        mappings["model.embed_tokens.weight"] = "model.embed_tokens.weight"
        mappings["model.norm.weight"] = "model.norm.weight"
        for layer_id in range(len(self.model.layers)):
            prefix = f"model.layers.{layer_id}"

            # <-- MODEL-SPECIFIC: Separate Q, K, V -> fused QKV weight and bias
            mappings[f"{prefix}.self_attn.qkv_proj_weight"] = [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ]
            if self.config.attention_bias:
                mappings[f"{prefix}.self_attn.qkv_proj_bias"] = [
                    f"{prefix}.self_attn.q_proj.bias",
                    f"{prefix}.self_attn.k_proj.bias",
                    f"{prefix}.self_attn.v_proj.bias",
                ]
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
                f"{prefix}.mlp.gate_proj.weight"
            )
            mappings[f"{prefix}.mlp.up_proj_weight"] = f"{prefix}.mlp.up_proj.weight"
            mappings[f"{prefix}.mlp.down_proj_weight"] = (
                f"{prefix}.mlp.down_proj.weight"
            )

        if self.config.tie_word_embeddings:
            mappings["lm_head.weight"] = "model.embed_tokens.weight"
        else:
            mappings["lm_head.weight"] = "lm_head.weight"
        return mappings

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load the TP-sharded weights from the checkpoint.

        >>> PARALLELISM: Weight loaders do the TP sharding <<<
        """
        mappings = self.checkpoint_mappings()

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, mappings, device
        ).state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        # Every parameter must come from the checkpoint. A parameter that stays
        # on the meta device gives wrong output with no error.
        missing = sorted(
            name for name, _ in self.named_parameters() if name not in rank_sharded
        )
        if missing:
            raise RuntimeError(
                f"The checkpoint does not supply {len(missing)} Seed-OSS "
                f"parameters, for example {missing[:4]}. Make sure that the "
                "checkpoint is a complete SeedOssForCausalLM checkpoint."
            )

        self.load_state_dict(rank_sharded, strict=False, assign=True)
