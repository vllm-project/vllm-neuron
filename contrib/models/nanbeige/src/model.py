# SPDX-License-Identifier: Apache-2.0
"""
Nanbeige BF16 Implementation
============================

Implementation of Nanbeige4.2 for the Neuron backend.
Adapted from the SmolLM3 BF16 model (dense, Llama-style GQA).

Nanbeige4.2 differs from Llama in one way: the model runs the stack of
physical decoder layers ``num_loops`` times with the same weights. Each
(loop, layer) pair has its own KV cache slot. The slot index is
``loop_idx * num_hidden_layers + layer_idx``, the same as the HF remote code.
The final RMSNorm runs after each loop when ``skip_loop_final_norm`` is False.

The residual stream stays in FP32 through all loops, and the logits are
FP32. The kernels receive BF16 inputs. The 2 loops double the depth of the
model, and a BF16 residual rounds at each of the 88 residual additions.

Supported parallelism: TP, SP, DP.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    Nanbeige-specific. Change when porting.
"""

import logging

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from transformers import PretrainedConfig
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.sampler import Sampler

import vllm_neuron.nn as neuron_nn
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding

from .config import NanbeigeConfig

logger = logging.getLogger(__name__)


def logical_layer_name(config: NanbeigeConfig, loop_idx: int, layer_idx: int) -> str:
    """Return the KV cache layer name of one (loop, physical layer) pair."""
    return f"layers.{loop_idx * config.num_hidden_layers + layer_idx}.self_attn"


# =============================================================================
# Section 1: RMS Normalization
# =============================================================================


class NanbeigeRMSNorm(nn.Module):
    """RMS Normalization (no mean subtraction, no bias).

    <-- MODEL-SPECIFIC: The model applies this norm about 90 times for each
    token, because the layer stack runs two loops. Calculate in FP32 and
    round one time. ``F.rms_norm`` on a BF16 input can lower to a BF16
    reduction on Neuron, and the loops increase that error.

    The input is the FP32 residual stream. The output has the weight dtype,
    because the kernels take BF16 inputs. Set ``out_dtype`` to FP32 for the
    norm between two loops, because its output is the next residual stream.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(
        self, hidden_states: torch.Tensor, out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(out_dtype or self.weight.dtype)


def _all_reduce_fp32(tp_group, partial: torch.Tensor) -> torch.Tensor:
    """Sum the TP partial outputs in FP32. Return the FP32 sum.

    <-- MODEL-SPECIFIC: The two loops give 44 decoder layers and 88 TP sums
    for each token. A BF16 ring reduction rounds at each step, and the loops
    increase that error. The FP32 sum goes directly into the FP32 residual
    stream, so it does not round to BF16.
    """
    summed = partial.to(torch.float32)
    tp_group.all_reduce(summed)
    return summed


def _reduce_scatter_fp32(tp_group, partial: torch.Tensor, dim: int) -> torch.Tensor:
    """Reduce-scatter the TP partial outputs in FP32 (see ``_all_reduce_fp32``)."""
    return tp_group.reduce_scatter(partial.to(torch.float32), dim=dim)


# =============================================================================
# Section 2: Rotary Position Embedding (Standard RoPE)
# =============================================================================


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """<-- MODEL-SPECIFIC: rotate_half style RoPE."""
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
    """
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    return _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)


class NanbeigeRotaryEmbedding(nn.Module):
    """Standard RoPE without scaling. The frequencies are in FP32, as in HF."""

    def __init__(self, config: NanbeigeConfig):
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
        """Return cos and sin, both of shape [T, head_dim]."""
        inv_freq = self._compute_inv_freq(device)
        inv_freq_expanded = inv_freq[:, None].float()
        positions_expanded = position_ids[None, :].float()
        freqs = (inv_freq_expanded @ positions_expanded).transpose(0, 1)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


# =============================================================================
# Section 3: Attention
# =============================================================================


class NanbeigeAttention(nn.Module):
    """GQA attention with TP head sharding and one KV cache for each loop.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads are sharded across TP ranks
    - KV heads are replicated when there are fewer KV heads than TP ranks
    - Prefill: all-gather input -> QKV proj -> attention -> O proj -> reduce-scatter
    - Decode: fused megakernel with TP all-reduce

    <-- MODEL-SPECIFIC:
    - The weights are shared across loops. The KV cache is not shared.
      ``k_cache_{loop}`` and ``v_cache_{loop}`` hold the cache of each loop.
    """

    def __init__(self, config: NanbeigeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_loops = config.num_loops
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

        # <-- MODEL-SPECIFIC: One KV cache pair for each loop. bind_kv_cache()
        # sets these attributes.
        for loop_idx in range(self.num_loops):
            setattr(self, f"k_cache_{loop_idx}", None)
            setattr(self, f"v_cache_{loop_idx}", None)

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach the weight loaders that shard the checkpoint tensors for TP."""
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

    def layer_name(self, loop_idx: int) -> str:
        return logical_layer_name(self.config, loop_idx, self.layer_idx)

    def caches(self, loop_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            getattr(self, f"k_cache_{loop_idx}"),
            getattr(self, f"v_cache_{loop_idx}"),
        )

    # -- Forward dispatch -----------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        loop_idx: int = 0,
    ):
        """Dispatch to the prefill path or to the decode path."""
        layer_name = self.layer_name(loop_idx)
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states, positions, position_embeddings, attn_metadata, loop_idx
            )

        # >>> PARALLELISM: All-gather from SP before attention <<<
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return self.forward_prefill(
            hidden_states, positions, position_embeddings, attn_metadata, loop_idx
        )

    # -- Prefill path ---------------------------------------------------------

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        loop_idx: int = 0,
    ) -> torch.Tensor:
        """Prefill: full-sequence attention with flash attention."""
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        k_cache, v_cache = self.caches(loop_idx)

        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # >>> PARALLELISM: Each rank computes only its sharded Q/K/V heads <<<
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

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # >>> PARALLELISM: Cache is per-rank (TP sharded KV heads) <<<
        layer_name = self.layer_name(loop_idx)
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size

        k_flat = k.reshape(-1, self.head_dim).to(k_cache.dtype)
        v_flat = v.reshape(-1, self.head_dim).to(v_cache.dtype)

        head_indices_for_put = torch.arange(
            self.num_key_value_heads_per_rank,
            dtype=torch.long,
            device=hidden_states.device,
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
                q.transpose(1, 2),
                k.transpose(1, 2),
                v,
                scale=self.scaling,
                tp_q=False,
                tp_out=True,
            )  # [Nh, Dh, T]

        attn_output = attn_output.unsqueeze(0)
        attn_output = NF.o_proj(attn_output, self.o_proj_weight, None)
        attn_output = attn_output.squeeze(0)

        # >>> PARALLELISM: Reduce-scatter to return to SP layout <<<
        if self.world_size > 1:
            attn_output = _reduce_scatter_fp32(self.tp_group, attn_output, dim=0)

        return attn_output.contiguous()

    # -- Decode path ----------------------------------------------------------

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
        loop_idx: int = 0,
    ):
        """Decode: fused megakernel. The kernel updates the KV cache in place."""
        k_cache, v_cache = self.caches(loop_idx)

        layer_name = self.layer_name(loop_idx)
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

        # Megakernel RoPE layout: [T, Hd] -> [Hd/2, B, S]
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
            K_cache=k_cache,
            V_cache=v_cache,
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

        # >>> PARALLELISM: TP all-reduce after megakernel <<<
        if self.world_size > 1:
            output = _all_reduce_fp32(self.tp_group, output)

        return output


# =============================================================================
# Section 4: MLP (Dense SwiGLU)
# =============================================================================


class NanbeigeMLP(nn.Module):
    """Dense SwiGLU MLP ``down(silu(gate(x)) * up(x))`` with TP sharding."""

    def __init__(self, config: NanbeigeConfig):
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

        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        )

        # >>> PARALLELISM: Combine TP shards <<<
        if self.world_size > 1:
            if is_prefill:
                output = _reduce_scatter_fp32(self.tp_group, output, dim=0)
            else:
                output = _all_reduce_fp32(self.tp_group, output)

        return output


# =============================================================================
# Section 5: Decoder Layer
# =============================================================================


class NanbeigeDecoderLayer(nn.Module):
    """One physical decoder layer. Each loop calls it with its loop index.

    hidden_states -> RMSNorm -> Attention -> residual -> RMSNorm -> MLP -> residual

    <-- MODEL-SPECIFIC: The residual is FP32, and the layer returns FP32. A
    BF16 input (the embedding output) becomes FP32 at the first residual.
    """

    def __init__(self, config: NanbeigeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.input_layernorm = NanbeigeRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = NanbeigeRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.self_attn = NanbeigeAttention(config, layer_idx=layer_idx)
        self.mlp = NanbeigeMLP(config)
        self.layer_idx = layer_idx

    def _is_decode(self, attn_metadata, loop_idx: int) -> bool:
        layer_name = logical_layer_name(self.config, loop_idx, self.layer_idx)
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
        return max_query_len <= decode_token_threshold

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        loop_idx: int = 0,
    ) -> torch.Tensor:
        is_decode = self._is_decode(attn_metadata, loop_idx)

        residual = hidden_states.to(torch.float32)
        hidden_states = self.input_layernorm(residual)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
            loop_idx=loop_idx,
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


class NanbeigeModel(nn.Module):
    """Nanbeige backbone: embedding -> num_loops x (all layers -> norm).

    >>> PARALLELISM: SP (Sequence Parallelism) <<<
    During prefill, each rank holds T/world_size tokens between the layers.
    The model all-gathers the full sequence after the last loop only.
    """

    def __init__(self, config: NanbeigeConfig):
        super().__init__()
        self.config = config

        # >>> PARALLELISM: TP group for SP <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )

        self.layers = nn.ModuleList(
            [
                NanbeigeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = NanbeigeRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = NanbeigeRotaryEmbedding(config)

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

        # <-- MODEL-SPECIFIC: Run the full layer stack num_loops times. The
        # norm between two loops keeps the FP32 residual stream. The last norm
        # also gives FP32, because the language model head computes in FP32.
        num_loops = self.config.num_loops
        for loop_idx in range(num_loops):
            for decoder_layer in self.layers:
                hidden_states = decoder_layer(
                    hidden_states,
                    positions=positions,
                    position_embeddings=position_embeddings,
                    attn_metadata=attn_metadata,
                    loop_idx=loop_idx,
                )
            if not self.config.skip_loop_final_norm:
                hidden_states = self.norm(hidden_states, out_dtype=torch.float32)

        if self.config.skip_loop_final_norm:
            hidden_states = self.norm(hidden_states, out_dtype=torch.float32)

        # >>> PARALLELISM: SP - all-gather to reconstruct full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states


# =============================================================================
# Section 7: Language Model Head
# =============================================================================


class NanbeigeForCausalLM(nn.Module):
    """Nanbeige model with an untied, column-parallel language model head."""

    def __init__(self, config: NanbeigeConfig):
        super().__init__()
        self.config = config
        self.model = NanbeigeModel(config)

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
            raise ValueError("Nanbeige does not support inputs_embeds. Send input_ids.")

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
                f"Prompt length ({T}) is not valid for sequence parallelism. "
                f"Use a length that is larger than world_size ({self.world_size}) "
                f"and a multiple of it."
            )

        hidden_states = self.model(
            input_ids, positions, attn_metadata=attn_metadata, rank=rank
        )

        hidden_states_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )

        # <-- MODEL-SPECIFIC: Compute the logits in FP32. A BF16 logit has a
        # step of 0.125 at a value of 16, and this rounding adds error to each
        # token of the 44-layer stack. The cast of the BF16 weight is exact.
        logits = torch.nn.functional.linear(
            hidden_states_for_logits.to(torch.float32),
            self.lm_head.weight.to(torch.float32),
        )
        # >>> PARALLELISM: Gather the vocabulary shards, as the lm_head does <<<
        if self.lm_head.gather_output and self.world_size > 1:
            logits = self.tp_group.all_gather(logits, dim=1)

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
        config = NanbeigeConfig.from_configs(hf_config, neuron_config)
        return cls(config)

    # -- KV Cache Management --------------------------------------------------
    # <-- MODEL-SPECIFIC: One KV cache slot for each (loop, physical layer).

    def get_kv_spec(self):
        layers = []
        for loop_idx in range(self.config.num_loops):
            for layer in self.model.layers:
                attn = layer.self_attn
                layers.append(
                    LayerSpec(
                        name=attn.layer_name(loop_idx),
                        num_kv_heads=attn.num_key_value_heads_per_rank,
                        head_size=attn.head_dim,
                        dtype=attn.dtype,
                        sliding_window_size=None,
                        chunk_size=None,
                    )
                )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor, torch.Tensor]]):
        for loop_idx in range(self.config.num_loops):
            for layer in self.model.layers:
                attn = layer.self_attn
                layer_name = attn.layer_name(loop_idx)
                if layer_name not in kv_caches:
                    raise RuntimeError(
                        f"KV cache for layer {layer_name} is not initialized. "
                        f"Allocate a KV cache for each name from get_kv_spec()."
                    )
                setattr(attn, f"k_cache_{loop_idx}", kv_caches[layer_name][0])
                setattr(attn, f"v_cache_{loop_idx}", kv_caches[layer_name][1])

    # -- Weight Loading -------------------------------------------------------

    def get_weight_mappings(self) -> dict[str, str | list[str]]:
        """Return the map from model parameter names to checkpoint keys."""
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

        mappings["model.embed_tokens.weight"] = "model.embed_tokens.weight"
        mappings["model.norm.weight"] = "model.norm.weight"
        if self.config.tie_word_embeddings:
            mappings["lm_head.weight"] = "model.embed_tokens.weight"
        else:
            mappings["lm_head.weight"] = "lm_head.weight"
        return mappings

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load the TP-sharded weights from the checkpoint."""
        mappings = self.get_weight_mappings()

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, mappings, device
        ).state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self.load_state_dict(rank_sharded, strict=False, assign=True)
