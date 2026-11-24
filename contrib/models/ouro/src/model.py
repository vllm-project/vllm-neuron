# SPDX-License-Identifier: Apache-2.0
"""
Ouro BF16 Implementation
========================

Implementation of ByteDance Ouro (a looped language model) for the Neuron
backend.

Ouro differs from a Llama-style dense model in three ways:

1. Universal Transformer loop. One stack of ``num_hidden_layers`` decoder
   layers runs ``total_ut_steps`` times with shared weights. The final norm
   applies after each step, and the normed output is the input of the next
   step.
2. Sandwich norms. Each decoder layer applies ``input_layernorm_2`` to the
   attention output and ``post_attention_layernorm_2`` to the MLP output,
   before each residual add.
3. Early exit. A ``Linear(hidden, 1)`` gate after each step gives an exit
   probability. The language-model head uses the hidden state of the first
   step where the cumulative exit probability reaches
   ``early_exit_threshold``.

Each UT step writes its own KV cache. The KV spec therefore has
``num_hidden_layers * total_ut_steps`` layers, with the names
``layers.{ut_step * num_hidden_layers + layer_idx}.self_attn``. This is the
same cache index that the HF model uses.

Supported parallelism: TP, SP, DP.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    Ouro-specific.
"""

import logging

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

from transformers import PretrainedConfig

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec, paged_slot_indices
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from .compat import build_per_head_block_table
from .config import OuroConfig

logger = logging.getLogger(__name__)


def kv_layer_name(ut_step: int, layer_idx: int, num_hidden_layers: int) -> str:
    """Return the KV cache layer name for one UT step of one decoder layer."""
    return f"layers.{ut_step * num_hidden_layers + layer_idx}.self_attn"


# =============================================================================
# Section 1: RMS Normalization
# <-- MODEL-SPECIFIC: HF OuroRMSNorm. Normalize in fp32, cast to the input
# dtype, then multiply by the weight in the input dtype.
# =============================================================================


class OuroRMSNorm(nn.Module):
    """RMS normalization with the HF Ouro rounding order."""

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
# Section 2: Rotary Position Embedding (default RoPE, rotate_half layout)
# =============================================================================


class OuroRotaryEmbedding(nn.Module):
    """Default RoPE. Compute the angles in fp32, then cast cos/sin."""

    def __init__(self, config: OuroConfig):
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
        """Return cos and sin, both with the shape [T, head_dim]."""
        inv_freq = self._compute_inv_freq(device)
        inv_freq_expanded = inv_freq[:, None].float()
        positions_expanded = position_ids[None, :].float()
        freqs = (inv_freq_expanded @ positions_expanded).transpose(0, 1)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


# =============================================================================
# Section 3: Attention
# =============================================================================


class OuroKVSlot(nn.Module):
    """Hold the KV cache of one UT step of one attention layer.

    bind_kv_cache() sets k_cache and v_cache. The forward adapter finds the
    cache owners from these two attributes.
    """

    def __init__(self):
        super().__init__()
        self.k_cache = None
        self.v_cache = None


class OuroAttention(nn.Module):
    """Multi-head attention with TP head sharding and one KV cache per UT step.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads are sharded across TP ranks.
    - Prefill: all-gather input -> QKV proj + RoPE -> attention -> O proj
      -> reduce-scatter.
    - Decode: fused decode megakernel, then TP all-reduce.

    <-- MODEL-SPECIFIC:
    - MHA (16 Q heads, 16 KV heads), head_dim 128, no bias.
    - The weights are shared across UT steps. The KV cache is not shared:
      kv_slots[u] holds the cache of UT step u.
    """

    def __init__(self, config: OuroConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_hidden_layers = config.num_hidden_layers
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
                f"num_attention_heads ({self.num_attention_heads}) does not divide "
                f"by the TP degree ({self.world_size}). Use a TP degree that "
                "divides the number of attention heads."
            )
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

        # <-- MODEL-SPECIFIC: One KV cache owner for each UT step.
        self.kv_slots = nn.ModuleList(
            [OuroKVSlot() for _ in range(config.total_ut_steps)]
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach the checkpoint -> parameter loaders (TP sharding)."""
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
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=(self.num_attention_heads * self.head_dim)
                // self.world_size,
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
        ut_step: int = 0,
    ):
        """Select the prefill path or the decode path from the metadata."""
        layer_name = kv_layer_name(ut_step, self.layer_idx, self.num_hidden_layers)
        slot = self.kv_slots[ut_step]
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata[layer_name],
                slot,
            )

        # >>> PARALLELISM: All-gather from SP before attention <<<
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return self.forward_prefill(
            hidden_states,
            positions,
            position_embeddings,
            attn_metadata[layer_name],
            slot,
        )

    # -- Prefill path ---------------------------------------------------------

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        layer_metadata: dict,
        slot: OuroKVSlot,
    ) -> torch.Tensor:
        """Prefill: fused QKV + RoPE, cache write, attention, O proj."""
        hidden_states = hidden_states.to(self.dtype)
        tokens, _ = hidden_states.shape

        # -- Step 1: QKV projection with fused RoPE ---------------------------
        # cos/sin are [T, head_dim] in the rotate_half layout.
        cos, sin = position_embeddings
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            cos_cache=cos.unsqueeze(0),
            sin_cache=sin.unsqueeze(0),
            num_q_heads=self.num_attention_heads_per_rank,
            num_kv_heads=self.num_key_value_heads_per_rank,
            d_head=self.head_dim,
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

        # -- Step 2: Write the KV cache of this UT step -----------------------
        block_size = layer_metadata["block_size"]
        block_table = layer_metadata["block_table_tensor"]
        cached_seq_len = layer_metadata.get("cached_seq_len")
        kv_segment_size = layer_metadata.get("kv_segment_size")

        k_cache = slot.k_cache
        v_cache = slot.v_cache
        self._write_kv_cache(k, v, layer_metadata["slot_mapping"], block_size, slot)

        # -- Step 3: Attention ------------------------------------------------
        # Use the segmented kernel when the scheduler splits the prompt into
        # KV segments. The queries then attend to the prior cached KV.
        if kv_segment_size:
            attn_output = NF.segmented_attention(
                q,
                k_cache=k_cache,
                v_cache=v_cache,
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
        attn_output = NF.o_proj(attn_output.unsqueeze(0), self.o_proj_weight, None)
        attn_output = attn_output.squeeze(0)  # [T, H]

        # >>> PARALLELISM: Reduce-scatter to return to SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    def _write_kv_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
        slot: OuroKVSlot,
    ) -> None:
        """Write K and V [nkh, T, Dh] into the paged cache of one UT step."""
        k_cache = slot.k_cache
        v_cache = slot.v_cache

        block_indices, position_indices = paged_slot_indices(
            slot_mapping, block_size, k_cache.shape[0]
        )

        k_flat = k.reshape(-1, self.head_dim).to(k_cache.dtype)
        v_flat = v.reshape(-1, self.head_dim).to(v_cache.dtype)

        head_indices_for_put = torch.arange(
            self.num_key_value_heads_per_rank,
            dtype=torch.long,
            device=k.device,
        ).repeat_interleave(slot_mapping.shape[0])
        block_indices_for_put = block_indices.repeat(self.num_key_value_heads_per_rank)
        position_indices_for_put = position_indices.repeat(
            self.num_key_value_heads_per_rank
        )

        k_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            k_flat,
        )
        v_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            v_flat,
        )

    # -- Decode path ----------------------------------------------------------

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        layer_metadata: dict,
        slot: OuroKVSlot,
    ) -> torch.Tensor:
        """Decode: fused megakernel (QKV + RoPE + attention + O proj).

        The megakernel attends to the new token from its own K and V. This
        method writes the new K and V into the cache.
        """
        slot_mapping = layer_metadata["slot_mapping"]
        block_size = layer_metadata["block_size"]
        block_table = layer_metadata["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        assert tokens == B * S_decode

        hidden_states = hidden_states.to(self.dtype)
        nkh = self.num_key_value_heads_per_rank

        X = hidden_states.view(B, S_decode, hidden)

        # RoPE in the kernel layout: [T, Dh] -> [Dh/2, B, S].
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

        # The kernel folds the 4D cache [num_blocks, nkh, block_size, Dh] into
        # a [num_blocks * nkh] pool. The per-head table uses the numbering
        # base_block * nkh + h.
        active_blocks_table = build_per_head_block_table(block_table, nkh)

        output, _, _ = NF.attention_decode(
            X=X,
            W_qkv=self.qkv_proj_weight,
            rmsnorm_QK_pre_rope_enabled=False,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=active_blocks_table,
            K_cache=slot.k_cache,
            V_cache=slot.v_cache,
            pos_ids=pos_ids_kernel,
            swa_start_pos_ids=None,
            softmax_scale=self.scaling,
            update_cache=False,
            W_out=self.o_proj_weight,
            transposed_out=False,
            out_in_sb=False,
        )

        # Compute the cache K and V outside the megakernel. At B * S <= 64 the
        # in-kernel QKV projection adds its fp32 partial sums in bf16, and at
        # LNC 2 it also adds the result of the other core in bf16. This gives
        # a cache K with an error 1.4 times the bf16 reference error. One
        # matmul with fp32 accumulation rounds each value one time.
        q_size = self.qkv_split_indices[0]
        kv = torch.matmul(hidden_states, self.qkv_proj_weight[:, q_size:])
        k, v = kv.split(nkh * self.head_dim, dim=-1)
        k = k.view(tokens, nkh, self.head_dim).transpose(0, 1)
        v = v.view(tokens, nkh, self.head_dim).transpose(0, 1)
        cos_k = cos.to(self.dtype).unsqueeze(0)
        sin_k = sin.to(self.dtype).unsqueeze(0)
        k_rot = torch.cat((-k[..., half_d:], k[..., :half_d]), dim=-1)
        k = k * cos_k + k_rot * sin_k
        self._write_kv_cache(k, v, slot_mapping.reshape(-1), block_size, slot)

        # >>> PARALLELISM: TP all-reduce after megakernel <<<
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output)

        # The residual add needs the token layout [T, H].
        return output.reshape(tokens, hidden)


# =============================================================================
# Section 4: MLP (dense SwiGLU)
# =============================================================================


class OuroMLP(nn.Module):
    """Dense SwiGLU MLP with TP intermediate sharding.

    Computes down_proj(silu(gate_proj(x)) * up_proj(x)). No bias.
    """

    def __init__(self, config: OuroConfig):
        super().__init__()

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        if config.intermediate_size % self.world_size != 0:
            raise ValueError(
                f"intermediate_size ({config.intermediate_size}) does not divide "
                f"by the TP degree ({self.world_size}). Use a TP degree that "
                "divides the intermediate size."
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
                output = self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 5: Decoder Layer (sandwich norms)
# =============================================================================


class OuroDecoderLayer(nn.Module):
    """One Ouro decoder layer.

    Architecture (MODEL-SPECIFIC):
        h   = x + input_layernorm_2(attn(input_layernorm(x)))
        out = h + post_attention_layernorm_2(mlp(post_attention_layernorm(h)))
    """

    def __init__(self, config: OuroConfig, layer_idx: int):
        super().__init__()
        H, eps, dtype = config.hidden_size, config.rms_norm_eps, config.torch_dtype
        self.input_layernorm = OuroRMSNorm(H, eps, dtype)
        self.input_layernorm_2 = OuroRMSNorm(H, eps, dtype)
        self.post_attention_layernorm = OuroRMSNorm(H, eps, dtype)
        self.post_attention_layernorm_2 = OuroRMSNorm(H, eps, dtype)
        self.self_attn = OuroAttention(config, layer_idx=layer_idx)
        self.mlp = OuroMLP(config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        ut_step: int = 0,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
            ut_step=ut_step,
        )
        hidden_states = self.input_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=is_prefill)
        hidden_states = self.post_attention_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# =============================================================================
# Section 6: Early exit
# =============================================================================


def select_exit_hidden(
    step_hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    threshold: float | None,
) -> torch.Tensor:
    """Select the hidden state of the exit step for each token.

    Args:
        step_hidden: [N, U, H] normed hidden state after each UT step.
        gate_weight: [H, 1] early-exit gate weight.
        gate_bias: [1] early-exit gate bias.
        threshold: Cumulative exit probability threshold. None selects the
            last step.

    Returns:
        [N, H] hidden state of the exit step.

    The function follows the HF rounding order in the model dtype:
    lambda_u = sigmoid(gate(h_u)), p_u = lambda_u * remaining,
    remaining = remaining * (1 - lambda_u), and p_last = remaining. The
    cumulative sum accumulates in fp32 and rounds each output to the model
    dtype, as torch.cumsum does for bf16. The exit step is the first step
    where the cumulative sum is at or above the threshold. If no step
    reaches the threshold, the exit step is the last step.

    A cumulative sum near the threshold can change the exit step after a
    small numerical difference. This changes the selected hidden state, so
    the logits of that token can show a large difference.
    """
    N, U, H = step_hidden.shape
    last = step_hidden[:, U - 1]
    if threshold is None or U == 1:
        return last

    dtype = step_hidden.dtype
    remaining = torch.ones(N, dtype=dtype, device=step_hidden.device)
    acc = torch.zeros(N, dtype=torch.float32, device=step_hidden.device)
    reached = []
    for u in range(U):
        # The router computes sigmoid(h @ W + b) for top_k=1 of one expert.
        # This is the HF gate: torch.sigmoid(Linear(H, 1)(h)). Call it once
        # for each step, as the HF model does. One [N * U, H] matmul can round
        # a gate logit differently, and change the exit step of a token.
        lam_u = NF.router(
            step_hidden[:, u],
            gate_weight,
            top_k=1,
            router_bias=gate_bias,
            activation="sigmoid",
            computation_dtype=dtype,
        ).view(N)
        if u < U - 1:
            p_u = lam_u * remaining
            remaining = remaining * (1.0 - lam_u)
        else:
            p_u = remaining
        acc = acc + p_u.to(torch.float32)
        reached.append(acc.to(dtype) >= threshold)

    selected = last
    for u in range(U - 2, -1, -1):
        selected = torch.where(reached[u].unsqueeze(-1), step_hidden[:, u], selected)
    return selected


# =============================================================================
# Section 7: Model Backbone (UT loop)
# =============================================================================


class OuroModel(nn.Module):
    """Ouro backbone with the Universal Transformer loop.

    >>> PARALLELISM: SP <<<
    During prefill, each rank holds T / world_size tokens between layers. The
    model all-gathers the normed output of each UT step at the end.
    """

    def __init__(self, config: OuroConfig):
        super().__init__()
        self.config = config
        self.total_ut_steps = config.total_ut_steps

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
        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=False,
            ),
        )

        self.layers = nn.ModuleList(
            [OuroDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = OuroRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = OuroRotaryEmbedding(config)

        # <-- MODEL-SPECIFIC: Early-exit gate, Linear(H, 1) with bias. The
        # weight is stored as [H, 1] (the router layout).
        self.early_exit_gate_weight = nn.Parameter(
            torch.empty(config.hidden_size, 1, dtype=config.torch_dtype)
        )
        self.early_exit_gate_bias = nn.Parameter(
            torch.empty(1, dtype=config.torch_dtype)
        )
        set_weight_loader(
            self.early_exit_gate_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=config.hidden_size,
                num_shards=1,
                is_storage_transposed=True,
            ),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the normed hidden state of each UT step, [T, U, H]."""
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

        step_outputs = []
        for ut_step in range(self.total_ut_steps):
            for decoder_layer in self.layers:
                hidden_states = decoder_layer(
                    hidden_states,
                    positions=positions,
                    position_embeddings=position_embeddings,
                    attn_metadata=attn_metadata,
                    ut_step=ut_step,
                    is_prefill=is_prefill,
                )
            # <-- MODEL-SPECIFIC: The final norm applies after each UT step,
            # and the normed output feeds the next step.
            hidden_states = self.norm(hidden_states)
            step_outputs.append(hidden_states)

        tokens_local, hidden = hidden_states.shape
        stacked = torch.stack(step_outputs, dim=1)  # [T_local, U, H]

        # >>> PARALLELISM: SP - all-gather to reconstruct full sequence <<<
        if is_prefill and self.world_size > 1:
            stacked = self.tp_group.all_gather(
                stacked.reshape(tokens_local, self.total_ut_steps * hidden), dim=0
            ).view(-1, self.total_ut_steps, hidden)

        return stacked


# =============================================================================
# Section 8: Language Model Head
# =============================================================================


class OuroForCausalLM(nn.Module):
    """Ouro model with the early-exit selection and the language-model head.

    >>> PARALLELISM: Column-parallel linear for LM head <<<
    <-- MODEL-SPECIFIC: lm_head is not tied to the embedding.
    """

    def __init__(self, config: OuroConfig):
        super().__init__()
        self.config = config
        self.model = OuroModel(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

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

        if config.vocab_size % self.world_size != 0:
            raise ValueError(
                f"vocab_size ({config.vocab_size}) does not divide by the TP "
                f"degree ({self.world_size}). Use a TP degree that divides the "
                "vocabulary size."
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
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is not None:
            raise ValueError(
                "Ouro on Neuron does not support inputs_embeds. Pass input_ids."
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
                f"({self.world_size}) and divide by it for SP. Use a prefill "
                "bucket that is a multiple of world_size."
            )

        step_hidden = self.model(
            input_ids, positions, attn_metadata=attn_metadata, rank=rank
        )  # [T, U, H]

        # Apply the exit gate only at the sampling positions.
        if sampling_positions is not None:
            step_hidden = torch.index_select(step_hidden, 0, sampling_positions)
        hidden_states_for_logits = select_exit_hidden(
            step_hidden,
            self.model.early_exit_gate_weight,
            self.model.early_exit_gate_bias,
            self.config.early_exit_threshold,
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
        config = OuroConfig.from_configs(hf_config, neuron_config)
        return cls(config)

    # -- KV Cache Management --------------------------------------------------
    # >>> PARALLELISM: KV spec uses per-rank head counts (TP-sharded) <<<

    def kv_layer_names(self) -> list[str]:
        """Return the KV layer names in UT-step-major order."""
        L = self.config.num_hidden_layers
        return [
            kv_layer_name(u, i, L)
            for u in range(self.config.total_ut_steps)
            for i in range(L)
        ]

    def get_kv_spec(self):
        attn = self.model.layers[0].self_attn
        layers = [
            LayerSpec(
                name=name,
                num_kv_heads=attn.num_key_value_heads_per_rank,
                head_size=attn.head_dim,
                dtype=attn.dtype,
                sliding_window_size=None,
                chunk_size=None,
            )
            for name in self.kv_layer_names()
        ]
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor, torch.Tensor]]):
        L = self.config.num_hidden_layers
        for u in range(self.config.total_ut_steps):
            for i, layer in enumerate(self.model.layers):
                layer_name = kv_layer_name(u, i, L)
                if layer_name not in kv_caches:
                    raise RuntimeError(
                        f"KV cache for layer {layer_name} is not initialized. "
                        "Allocate the caches from get_kv_spec()."
                    )
                slot = layer.self_attn.kv_slots[u]
                slot.k_cache = kv_caches[layer_name][0]
                slot.v_cache = kv_caches[layer_name][1]

    # -- Weight Loading -------------------------------------------------------

    def build_checkpoint_mappings(self) -> dict[str, str | list[str]]:
        """Return the parameter name -> checkpoint key mappings."""
        mappings: dict[str, str | list[str]] = {}
        for layer_id in range(len(self.model.layers)):
            prefix = f"model.layers.{layer_id}"
            mappings[f"{prefix}.self_attn.qkv_proj_weight"] = [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ]
            mappings[f"{prefix}.self_attn.o_proj_weight"] = (
                f"{prefix}.self_attn.o_proj.weight"
            )
            for norm in (
                "input_layernorm",
                "input_layernorm_2",
                "post_attention_layernorm",
                "post_attention_layernorm_2",
            ):
                mappings[f"{prefix}.{norm}.weight"] = f"{prefix}.{norm}.weight"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                mappings[f"{prefix}.mlp.{proj}_weight"] = f"{prefix}.mlp.{proj}.weight"

        mappings["model.embed_tokens.weight"] = "model.embed_tokens.weight"
        mappings["model.norm.weight"] = "model.norm.weight"
        mappings["model.early_exit_gate_weight"] = "model.early_exit_gate.weight"
        mappings["model.early_exit_gate_bias"] = "model.early_exit_gate.bias"
        if self.config.tie_word_embeddings:
            mappings["lm_head.weight"] = "model.embed_tokens.weight"
        else:
            mappings["lm_head.weight"] = "lm_head.weight"
        return mappings

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load the TP-sharded weights from the checkpoint."""
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        result = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, self.build_checkpoint_mappings(), device
        )
        if result.missing_keys:
            raise RuntimeError(
                f"The checkpoint does not have the keys of {len(result.missing_keys)} "
                f"parameters, for example {sorted(result.missing_keys)[:5]}. "
                "Make sure that the checkpoint is a complete Ouro checkpoint."
            )
        if result.unexpected_keys:
            raise RuntimeError(
                f"The checkpoint has {len(result.unexpected_keys)} keys that the "
                f"model does not use, for example {sorted(result.unexpected_keys)[:5]}. "
                "Make sure that the checkpoint is an Ouro checkpoint."
            )
        rank_sharded = result.state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self.load_state_dict(rank_sharded, strict=False, assign=True)
