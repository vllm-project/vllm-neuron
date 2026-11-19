# SPDX-License-Identifier: Apache-2.0
"""
Jina Embeddings v5 text (EuroBERT encoder) for the pooling runner.

The backbone is a EuroBERT bidirectional encoder:

    embed -> N x [RMSNorm -> non-causal attention -> residual
                  -> RMSNorm -> SiLU gated MLP -> residual] -> RMSNorm

The model has no LM head. ``forward`` returns the flat ``[T, H]`` hidden
states. The runner ``_pool`` path then applies the upstream pooler (LAST
token, then L2 normalize).

>>> PARALLELISM: TP + SP (sequence parallel), the same layout as the plugin's
Llama prefill. <<<
<-- MODEL-SPECIFIC: non-causal attention. A key mask removes the bucket pad
tokens, because the attention is bidirectional.
"""

import logging

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

from nkilib.core.utils.common_types import NormType

import vllm_neuron.functional as NF
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.parallel.neuron_parallel_state import shard_select
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
    with_rank_override,
)

from .config import JinaEmbeddingsV5Config, jina_v5_config_from_hf

logger = logging.getLogger(__name__)


def _is_prefill(attn_metadata: dict, layer_idx: int = 0) -> bool:
    layer = attn_metadata[f"layers.{layer_idx}.self_attn"]
    return layer["max_query_len"] > layer["decode_token_threshold"]


def build_key_bounds(
    positions: torch.Tensor, num_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``attention_cte`` key bounds for one padded prefill sequence.

    The runner schedules a maximum of one prefill in each batch, and it gives
    each pad token the last real position. Thus the real length is
    ``max(positions) + 1``. Each query attends to the keys in ``[0, real_len)``.

    Args:
        positions: ``[T]`` positions of the padded sequence.
        num_heads: The number of query heads on this rank.

    Returns:
        ``(bound_min, bound_max)``, each ``[num_heads, T, 1]`` int32.

    Example:
        >>> lo, hi = build_key_bounds(torch.tensor([0, 1, 2, 2]), 2)
        >>> hi[0, :, 0].tolist()
        [3, 3, 3, 3]
    """
    tokens = positions.shape[0]
    real_len = (positions.max() + 1).to(torch.int32)
    bound_min = torch.zeros(
        num_heads, tokens, 1, dtype=torch.int32, device=positions.device
    )
    bound_max = bound_min + real_len
    return bound_min, bound_max


# =============================================================================
# RMS normalization and rotary position embedding
# These two modules are the plugin's Llama RMSNorm and default RoPE, copied
# here so that this folder does not import another model's package.
# =============================================================================


class JinaEmbeddingsV5RMSNorm(nn.Module):
    """RMS normalization with a float32 variance.

    <-- MODEL-SPECIFIC: EuroBERT uses standard RMSNorm without bias.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class JinaEmbeddingsV5RotaryEmbedding(nn.Module):
    """Default rotate-half RoPE.

    <-- MODEL-SPECIFIC: EuroBERT uses the default RoPE frequencies. The config
    rejects any scaling that changes them.
    """

    def __init__(self, config: JinaEmbeddingsV5Config):
        super().__init__()
        self.rope_theta = config.rope_parameters["rope_theta"]
        self.head_dim = config.head_dim
        rope_type = config.rope_parameters.get("rope_type")
        if rope_type not in (None, "default"):
            raise NotImplementedError(
                f"rope_type {rope_type!r} is not supported for Jina Embeddings "
                "v5. Use a checkpoint with default RoPE."
            )

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the cos and sin embeddings, each ``[T, head_dim]``."""
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
                / self.head_dim
            )
        )
        inv_freq_expanded = inv_freq[:, None].float()  # [Hd/2, 1]
        positions_expanded = position_ids[None, :].float()  # [1, T]
        freqs = (inv_freq_expanded @ positions_expanded).transpose(0, 1)  # [T, Hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, Hd]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


# =============================================================================
# Attention
# =============================================================================


class JinaEmbeddingsV5Attention(nn.Module):
    """EuroBERT non-causal self-attention.

    >>> PARALLELISM: TP head sharding. The constructor and the weight loaders
    are the TP-only part of the plugin's Llama attention: Q/K/V/O heads shard
    across TP, and KV heads replicate when TP is larger than the KV head
    count. Prefill gathers the SP slices before attention and reduce-scatters
    after it. <<<

    <-- MODEL-SPECIFIC:
    - Prefill: fused QKV + RoPE (``NF.qkv_proj``), then the NKI
      ``attention_cte`` kernel (``NF.flash_attention``) with
      ``causal_mask=False`` and key bounds. The model writes no KV cache,
      because an encoder never reads it back.
    - Decode-classified step (each sequence has 1 token): the attention of a
      1-token sequence equals its value vector. This path computes
      ``o_proj(v_proj(x))`` with float32 accumulation and a float32 TP sum.
      It does not read the KV cache.
    """

    def __init__(self, config: JinaEmbeddingsV5Config, layer_idx: int):
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
                f"num_attention_heads={self.num_attention_heads} is not "
                f"divisible by the tensor parallel size {self.world_size}. "
                "Use a tensor parallel size that divides the head count."
            )
        if (
            self.world_size >= self.num_key_value_heads
            and self.world_size % self.num_key_value_heads != 0
        ) or (
            self.world_size < self.num_key_value_heads
            and self.num_key_value_heads % self.world_size != 0
        ):
            raise ValueError(
                f"num_key_value_heads={self.num_key_value_heads} and the tensor "
                f"parallel size {self.world_size} do not divide each other. "
                "Use a tensor parallel size that divides the KV head count, or "
                "a multiple of it."
            )

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
        self.num_kv_heads_for_weight = self.num_key_value_heads_per_rank
        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // self.num_kv_heads_for_weight
        )

        # >>> PARALLELISM: QKV weight shapes <<<
        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_kv_heads_for_weight * self.head_dim
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

        # The runner binds the KV caches through ``bind_kv_cache``. The
        # encoder never reads or writes them.
        self.k_cache = None
        self.v_cache = None

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach the checkpoint-to-parameter loaders.

        >>> PARALLELISM: The loaders give the TP shard of this rank. <<<
        <-- MODEL-SPECIFIC: The checkpoint stores separate Q, K and V weights
        without bias. The QKV loader fuses them into one parameter.
        """
        o_shard_size = (self.num_attention_heads * self.head_dim) // self.world_size
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
            with_rank_override(
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=o_shard_size,
                    num_shards=self.world_size,
                    is_storage_transposed=True,
                ),
                rank=self.rank,
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
        key_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if not _is_prefill(attn_metadata, self.layer_idx):
            return self.forward_single_token(hidden_states)

        # >>> PARALLELISM: SP gather: T/TP -> T <<<
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_encoder_prefill(
            hidden_states, position_embeddings, key_bounds
        )

    def forward_encoder_prefill(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        key_bounds: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(self.dtype)
        tokens = hidden_states.shape[0]
        cos, sin = position_embeddings

        # <-- MODEL-SPECIFIC: rotate-half RoPE, fused into the QKV kernel.
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
            d_head=self.head_dim,
            cos_cache=cos.unsqueeze(0),
            sin_cache=sin.unsqueeze(0),
            num_q_heads=self.num_attention_heads_per_rank,
            num_kv_heads=self.num_kv_heads_for_weight,
        ).squeeze(0)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim)
        k = k.view(tokens, self.num_kv_heads_for_weight, self.head_dim)
        v = v.view(tokens, self.num_kv_heads_for_weight, self.head_dim)
        q = q.transpose(0, 1)  # [Nh, T, Dh]
        k = k.transpose(0, 1).repeat_interleave(self.num_key_value_groups, dim=0)
        v = v.transpose(0, 1).repeat_interleave(self.num_key_value_groups, dim=0)

        bound_min, bound_max = key_bounds
        # <-- MODEL-SPECIFIC: bidirectional attention. The bounds remove the
        # bucket pad keys. This is the same call form as the Qwen3-ASR and
        # Qwen3-VL encoders.
        attn_output = NF.flash_attention(
            q,
            k,
            v,
            scale=self.scaling,
            causal_mask=False,
            tp_q=True,
            tp_k=True,
            tp_out=True,
            bound_min=bound_min,
            bound_max=bound_max,
            allow_vector_exp=False,
        )  # [Nh, Dh, T]

        attn_output = NF.o_proj(attn_output.unsqueeze(0), self.o_proj_weight, None)
        attn_output = attn_output.squeeze(0)  # [T, H]

        # >>> PARALLELISM: reduce-scatter back to the SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)
        return attn_output.contiguous()

    def forward_single_token(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # A softmax over one key gives weight 1.0, so the output is v.
        # <-- MODEL-SPECIFIC: keep v, the O-proj partials and the TP sum in
        # float32, then round to the model dtype one time. A bf16 partial on
        # each rank adds one rounding step for each layer. Across all layers,
        # this makes the error larger than the error of the bf16 reference.
        hidden_states = hidden_states.to(self.dtype)
        tokens = hidden_states.shape[0]
        v_weight = self.qkv_proj_weight[:, self.q_size + self.kv_size :]
        v = torch.matmul(hidden_states.float(), v_weight.float())
        v = v.view(tokens, self.num_kv_heads_for_weight, self.head_dim)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        output = torch.matmul(v.reshape(tokens, -1), self.o_proj_weight.float())
        # >>> PARALLELISM: sum the O-proj partials across TP <<<
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output)
        return output.to(self.dtype)


# =============================================================================
# MLP
# =============================================================================


class JinaEmbeddingsV5MLP(nn.Module):
    """SiLU gated MLP with TP intermediate sharding.

    This is the plugin's Llama MLP without the MLP data parallel and fused
    norm options, which this model does not use.

    >>> PARALLELISM: TP <<<
    - gate_proj, up_proj: hidden -> intermediate/TP on each rank
    - down_proj: intermediate/TP -> hidden on each rank
    - Prefill: all-gather before, reduce-scatter after (SP)
    - Decode-classified step: all-reduce after

    <-- MODEL-SPECIFIC:
    - SiLU activation (silu(gate) * up)
    - No bias on any projection
    """

    def __init__(self, config: JinaEmbeddingsV5Config):
        super().__init__()

        # >>> PARALLELISM: The MLP TP group. The factory requires
        # mlp_dp_size == 1, so this group has the same ranks as the TP group. <<<
        from vllm_neuron.parallel.neuron_parallel_state import get_neuron_mlp_tp_group

        self.mlp_tp_group = get_neuron_mlp_tp_group()
        self.mlp_tp_size = self.mlp_tp_group.world_size
        self.mlp_tp_rank = self.mlp_tp_group.rank_in_group

        self.hidden_size = config.hidden_size
        if config.intermediate_size % self.mlp_tp_size != 0:
            raise ValueError(
                f"intermediate_size={config.intermediate_size} is not divisible "
                f"by the tensor parallel size {self.mlp_tp_size}. Use a tensor "
                "parallel size that divides the intermediate size."
            )
        self.intermediate_size_per_rank = config.intermediate_size // self.mlp_tp_size

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
        """>>> PARALLELISM: TP sharding of the MLP weights. <<<"""
        gate_up_loader = sharding_weight_loader(
            shard_dim=1,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.mlp_tp_size,
            is_storage_transposed=True,
        )
        down_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.mlp_tp_size,
            is_storage_transposed=True,
        )

        gate_up_loader = with_rank_override(gate_up_loader, rank=self.mlp_tp_rank)
        down_loader = with_rank_override(down_loader, rank=self.mlp_tp_rank)

        set_weight_loader(self.gate_proj_weight, gate_up_loader)
        set_weight_loader(self.up_proj_weight, gate_up_loader)
        set_weight_loader(self.down_proj_weight, down_loader)

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        # >>> PARALLELISM: All-gather from SP for the full sequence <<<
        if is_prefill and self.mlp_tp_size > 1:
            hidden_states = self.mlp_tp_group.all_gather(hidden_states, dim=0)

        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
            eps=1e-6,
            ln_w=None,
            norm_type=NormType.NO_NORM,
            fused_add_tensor=None,
            store_fused_add_result=False,
        )

        # >>> PARALLELISM: Combine the TP shards. Prefill reduce-scatters back
        # to the SP layout. The decode-classified step all-reduces. <<<
        if is_prefill:
            if self.mlp_tp_size > 1:
                output = self.mlp_tp_group.reduce_scatter(output, dim=0)
        elif self.mlp_tp_size > 1:
            output = self.mlp_tp_group.all_reduce(output)
        return output


# =============================================================================
# Encoder layer and backbone
# =============================================================================


class JinaEmbeddingsV5Layer(nn.Module):
    """EuroBERT pre-norm encoder layer."""

    def __init__(self, config: JinaEmbeddingsV5Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = JinaEmbeddingsV5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = JinaEmbeddingsV5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.self_attn = JinaEmbeddingsV5Attention(config, layer_idx=layer_idx)
        self.mlp = JinaEmbeddingsV5MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
        key_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        is_prefill = _is_prefill(attn_metadata, self.layer_idx)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            positions,
            position_embeddings,
            attn_metadata,
            key_bounds=key_bounds,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=is_prefill)
        return residual + hidden_states


class JinaEmbeddingsV5Encoder(nn.Module):
    """EuroBERT backbone with the sequence parallel (SP) and TP layout.

    >>> PARALLELISM: In prefill, the embedding scatters the tokens across TP
    (T/TP for each rank). The final all-gather rebuilds the full sequence. <<<
    """

    def __init__(self, config: JinaEmbeddingsV5Config):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_embedding_tp_group,
        )

        emb_tp_group = get_neuron_embedding_tp_group()
        self.embedding_tp_rank = emb_tp_group.rank_in_group
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=emb_tp_group.device_group,
        )
        emb_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.embed_tokens.vocab_size_per_rank,
            num_shards=self.embed_tokens.tp_size,
            is_storage_transposed=False,
        )
        set_weight_loader(
            self.embed_tokens.weight,
            with_rank_override(emb_loader, rank=self.embedding_tp_rank),
        )

        self.layers = nn.ModuleList(
            [
                JinaEmbeddingsV5Layer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = JinaEmbeddingsV5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = JinaEmbeddingsV5RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: dict,
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_prefill = _is_prefill(attn_metadata)

        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
        )
        if (
            is_prefill
            and self.world_size > 1
            and inputs_embeds is not None
            and is_token_ids is not None
        ):
            inputs_embeds = shard_select(inputs_embeds, 0, self.tp_group)
            is_token_ids = shard_select(is_token_ids, 0, self.tp_group)
        hidden_states = NF.merge_prompt_embeds(
            hidden_states, inputs_embeds, is_token_ids
        )

        # RoPE uses the full positions. Attention gathers the full sequence.
        position_embeddings = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )
        key_bounds = None
        if is_prefill:
            key_bounds = build_key_bounds(
                positions,
                self.layers[0].self_attn.num_attention_heads_per_rank,
            )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
                key_bounds=key_bounds,
            )

        hidden_states = self.norm(hidden_states)

        # >>> PARALLELISM: SP all-gather to rebuild the full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return hidden_states


# =============================================================================
# Pooling model
# =============================================================================


class JinaEmbeddingsV5ForEmbedding(nn.Module):
    """Jina Embeddings v5 encoder with the upstream embedding pooler."""

    is_pooling_model = True
    default_seq_pooling_type = "LAST"

    def __init__(self, config: JinaEmbeddingsV5Config):
        super().__init__()
        self.config = config
        self.model = JinaEmbeddingsV5Encoder(config)
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        from vllm.config import get_current_vllm_config
        from vllm.model_executor.layers.pooler import DispatchPooler

        vllm_config = get_current_vllm_config()
        cache_dtype = getattr(vllm_config.cache_config, "cache_dtype", None)
        if (cache_dtype or "").startswith("fp8"):
            raise ValueError(
                f"kv_cache_dtype={cache_dtype!r} is not supported for Jina "
                "Embeddings v5. The encoder does not use a KV cache. Use the "
                "default kv_cache_dtype (auto)."
            )
        if getattr(vllm_config.cache_config, "enable_prefix_caching", False):
            raise ValueError(
                "Prefix caching is not supported for Jina Embeddings v5. The "
                "attention is bidirectional. Set --no-enable-prefix-caching."
            )
        scheduler_config = vllm_config.scheduler_config
        model_config = vllm_config.model_config
        if (
            scheduler_config is not None
            and model_config is not None
            and getattr(scheduler_config, "enable_chunked_prefill", False)
            and scheduler_config.max_num_batched_tokens < model_config.max_model_len
        ):
            raise ValueError(
                "Chunked prefill is not supported for Jina Embeddings v5. The "
                "attention is bidirectional. Set --max-num-batched-tokens to "
                "a value that is not smaller than --max-model-len."
            )

        pooler_config = model_config.pooler_config if model_config else None
        if pooler_config is None:
            raise ValueError(
                "pooler_config is not set. Start Jina Embeddings v5 with "
                "--runner pooling."
            )
        self.pooler = DispatchPooler.for_embedding(pooler_config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: dict | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        positions = positions.to(torch.int32)
        tokens = input_ids.shape[0]
        # >>> PARALLELISM: SP length check <<<
        if _is_prefill(attn_metadata) and (
            tokens <= self.world_size or tokens % self.world_size != 0
        ):
            raise ValueError(
                f"Prompt length {tokens} is not valid for SP with world size "
                f"{self.world_size}. Use a bucket that is larger than and "
                "divisible by the world size."
            )
        return self.model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
        )

    @classmethod
    def from_configs(cls, hf_config, neuron_config: NeuronConfig | None):
        return cls(jina_v5_config_from_hf(hf_config, neuron_config))

    # ── KV cache ──────────────────────────────────────────────────────────
    # The runner calls these hooks for each model. The encoder never reads
    # the KV cache, but the runner still allocates and binds it.

    def get_kv_spec(self) -> KVSpec:
        return KVSpec(
            layers=[
                LayerSpec(
                    name=f"layers.{i}.self_attn",
                    num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                    head_size=layer.self_attn.head_dim,
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
                for i, layer in enumerate(self.model.layers)
            ]
        )

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise KeyError(f"KV cache for layer {layer_name} is not initialized")
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # ── Weight loading ────────────────────────────────────────────────────

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        load_encoder_weights(
            self, checkpoint_path, device, cache_dir, prefix="model.", rank=self.rank
        )


def checkpoint_mappings(num_layers: int, prefix: str = "model.") -> dict:
    """Map the parameter names to the EuroBERT checkpoint keys.

    The checkpoint keys have no ``model.`` prefix. The checkpoint has no LM
    head. Q, K and V go into one fused parameter.

    Args:
        num_layers: The number of encoder layers.
        prefix: The parameter name prefix of the encoder module.

    Returns:
        A dict from the parameter name to a checkpoint key, or to a list of
        checkpoint keys for a fused parameter.

    Example:
        >>> checkpoint_mappings(1)["model.norm.weight"]
        'norm.weight'
    """
    mappings: dict[str, str | list[str]] = {
        f"{prefix}embed_tokens.weight": "embed_tokens.weight",
        f"{prefix}norm.weight": "norm.weight",
    }
    for i in range(num_layers):
        p = f"{prefix}layers.{i}"
        c = f"layers.{i}"
        mappings[f"{p}.self_attn.qkv_proj_weight"] = [
            f"{c}.self_attn.q_proj.weight",
            f"{c}.self_attn.k_proj.weight",
            f"{c}.self_attn.v_proj.weight",
        ]
        mappings[f"{p}.self_attn.o_proj_weight"] = f"{c}.self_attn.o_proj.weight"
        mappings[f"{p}.input_layernorm.weight"] = f"{c}.input_layernorm.weight"
        mappings[f"{p}.post_attention_layernorm.weight"] = (
            f"{c}.post_attention_layernorm.weight"
        )
        mappings[f"{p}.mlp.gate_proj_weight"] = f"{c}.mlp.gate_proj.weight"
        mappings[f"{p}.mlp.up_proj_weight"] = f"{c}.mlp.up_proj.weight"
        mappings[f"{p}.mlp.down_proj_weight"] = f"{c}.mlp.down_proj.weight"
    return mappings


def load_encoder_weights(
    module: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    cache_dir: str | None,
    prefix: str,
    rank: int,
    pipelined: bool = True,
):
    """Load the TP shard of this rank from the checkpoint into ``module``.

    Args:
        module: The pooling model (``prefix="model."``) or the encoder
            (``prefix=""``).
        checkpoint_path: The checkpoint directory.
        device: The load device.
        cache_dir: The download directory, or ``None``.
        prefix: The parameter name prefix of the encoder.
        rank: The TP rank.
        pipelined: Use the pipelined distributed loader. Set ``False`` for a
            CPU test, which loads without distributed coordination.

    Returns:
        The ``CheckpointLoadResult``. Its ``unexpected_keys`` lists each
        checkpoint tensor that no parameter uses.

    Raises:
        NotImplementedError: The checkpoint has task LoRA adapters. The port
            loads only the base encoder weights.

    Example:
        >>> load_encoder_weights(model, "/path/to/model", "cpu", None, "model.", 0)
    """
    import os

    adapters_dir = os.path.join(checkpoint_path, "adapters")
    if os.path.isdir(adapters_dir):
        raise NotImplementedError(
            f"The checkpoint has task LoRA adapters in {adapters_dir}. The "
            "Neuron port loads only the base encoder weights. Use a "
            "checkpoint without the adapters directory."
        )

    config = module.config
    tp_size = get_tp_group().world_size
    mappings = checkpoint_mappings(config.num_hidden_layers, prefix)
    checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
    if pipelined:
        result = checkpoint.load_sharded_pipelined(
            rank, tp_size, module, mappings, device
        )
    else:
        result = checkpoint.load_sharded(rank, tp_size, module, mappings, device)
    state_dict = result.state_dict

    expected = set(mappings)
    missing = expected - set(state_dict)
    if missing:
        raise RuntimeError(
            f"Weight loading did not give {len(missing)} parameters, for "
            f"example {sorted(missing)[:3]}. Make sure that the checkpoint is "
            "a complete Jina Embeddings v5 checkpoint."
        )
    for name, tensor in state_dict.items():
        if tensor.dtype != config.torch_dtype:
            state_dict[name] = tensor.to(config.torch_dtype)
    module.load_state_dict(state_dict, strict=False, assign=True)
    return result
