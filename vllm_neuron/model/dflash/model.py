# SPDX-License-Identifier: Apache-2.0
"""Qwen3-shaped DFlash draft model for Neuron.

The DFlash checkpoint owns only draft weights. The target model provides the
first verified token and selected hidden states; embedding and LM-head weights
omitted by the draft checkpoint are loaded from the target checkpoint.
"""

from __future__ import annotations

import os

import torch
from nkilib.core.utils.common_types import NormType
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.gpt_oss.model_bf16 import GptOssRotaryEmbedding
from vllm_neuron.model.gpt_oss.weight_loaders_bf16 import (
    fused_qkv_bias_loader,
    fused_qkv_weight_loader,
)
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.llama3.eagle3_model import (
    _make_rmsnorm,
    embedding_sharding_padding_weight_loader,
)
from vllm_neuron.model.llama3.model import LlamaMLP, LlamaRotaryEmbedding
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    last_dim_padding_weight_loader,
    scaled_bias_loader,
    set_weight_loader,
    sharding_weight_loader_with_padding,
    with_rank_override,
)

from .config import DFlashConfig

# Diagnostic only: skip writing context K/V into the draft cache. Produces
# wrong proposals (the drafter sees no context) but isolates how much of the
# draft NEFF's time that write path costs. Never set this in a real run.
_SKIP_CONTEXT_KV = os.environ.get("VLLM_NEURON_DFLASH_SKIP_CONTEXT_KV") == "1"


def _make_rotary(config: DFlashConfig) -> nn.Module:
    """Pick the rotary implementation the drafter was trained with.

    DFlash drafters inherit their target's positional encoding: the GPT-OSS
    drafter uses YaRN, the Llama 3.1 one plain RoPE. Both implementations here
    expose the same ``(position_ids, device, dtype) -> (cos, sin)`` contract, so
    only the frequency construction differs.
    """
    rope_type = config.rope_type
    if rope_type == "yarn":
        return GptOssRotaryEmbedding(config)
    if rope_type in ("default", "llama3"):
        return LlamaRotaryEmbedding(config)
    raise ValueError(
        f"Unsupported DFlash draft rope_type {rope_type!r}; "
        "expected one of 'yarn', 'llama3', 'default'"
    )


def resolve_verified_block(
    raw_sampled_token_ids: torch.Tensor,
    last_token_indices: torch.Tensor,
    *,
    vocab_size: int,
    query_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve the bonus token and the rewound context boundary.

    Verification leaves each row of ``raw_sampled_token_ids`` as a run of
    accepted samples followed by a rejection sentinel, so one count of valid
    entries yields both answers: the last accepted sample is the bonus token
    that seeds the next proposal block, and every rejected slot moves that
    request's context boundary back by one.

    Args:
        raw_sampled_token_ids: ``[bs, 1]`` after a prefill, ``[bs, query_len]``
            in verified decode.
        last_token_indices: Index of each request's last target query token.
        vocab_size: Draft vocabulary size; ids outside it are rejections.
        query_len: Proposal block size (``num_speculative_tokens + 1``).

    Returns:
        ``(bonus, last_token_indices)``, the int32 bonus token per request and
        the boundary rewound over rejected tokens.
    """
    valid = (raw_sampled_token_ids >= 0) & (raw_sampled_token_ids < vocab_size)
    valid_count = valid.sum(dim=1)

    # gather() requires a long index; an int32 one fails in the XLA lowering.
    last_valid = valid_count.sub(1).clamp_min(0).unsqueeze(1).to(torch.long)
    bonus = raw_sampled_token_ids.gather(1, last_valid).squeeze(1).to(torch.int32)
    # A row with no valid sample would otherwise seed the block with whatever
    # sentinel sits in column zero.
    bonus = torch.where(valid_count > 0, bonus, torch.zeros_like(bonus))

    # After a prefill there is a single sample and nothing to rewind.
    if raw_sampled_token_ids.shape[1] > 1:
        num_rejected = (query_len - valid_count).clamp_min(0)
        last_token_indices = last_token_indices - num_rejected.to(
            last_token_indices.dtype
        )
    return bonus, last_token_indices


def _feature_projection_loader(
    *,
    target_hidden_size: int,
    draft_hidden_size: int,
    padded_hidden_size: int,
    num_features: int,
) -> SafetensorsWeightLoader:
    """Pad each concatenated target feature independently before projection."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        del rank
        assert len(slices) == 1
        weight = slices[0][:]
        expected = (draft_hidden_size, target_hidden_size * num_features)
        assert tuple(weight.shape) == expected, (
            f"DFlash fc.weight must be {expected}, got {tuple(weight.shape)}"
        )
        features = torch.split(weight, target_hidden_size, dim=1)
        padded_features = [
            torch.nn.functional.pad(f, (0, padded_hidden_size - target_hidden_size))
            for f in features
        ]
        weight = torch.cat(padded_features, dim=1)
        return torch.nn.functional.pad(
            weight, (0, 0, 0, padded_hidden_size - draft_hidden_size)
        )

    return SafetensorsWeightLoader(transform=transform)


class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.dtype = config.torch_dtype
        self.scaling = config.head_dim**-0.5
        self.rms_norm_eps = config.rms_norm_eps
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
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

        self.q_size = self.num_attention_heads_per_rank * self.head_dim
        self.kv_size = self.num_key_value_heads_per_rank * self.head_dim
        qkv_size = self.q_size + 2 * self.kv_size
        self.qkv_split_indices = [self.q_size, self.q_size + self.kv_size]

        self.attention_bias = config.attention_bias
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(self.q_size, self.hidden_size, dtype=self.dtype)
        )
        # GPT-OSS's drafter has attention biases, Llama 3.1's does not. Register
        # them only when the checkpoint carries them, so load_weights stays
        # strict instead of silently tolerating missing tensors.
        if self.attention_bias:
            self.qkv_proj_bias = nn.Parameter(torch.zeros(qkv_size, dtype=self.dtype))
            self.o_proj_bias = nn.Parameter(
                torch.zeros(self.hidden_size, dtype=self.dtype)
            )
        else:
            self.qkv_proj_bias = None
            self.o_proj_bias = None
        self.q_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=self.dtype))
        self.k_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=self.dtype))
        self.rotary_emb = _make_rotary(config)

        self.k_cache = None
        self.v_cache = None
        self._setup_weight_loaders(config)

    def _setup_weight_loaders(self, config: DFlashConfig) -> None:
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
                shard_dim=1,
                num_shards=self.world_size,
                num_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                hidden_size=self.hidden_size,
                num_kv_replicas=self.num_kv_replicas,
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader_with_padding(
                shard_dim=0,
                shard_size=self.q_size,
                num_shards=self.world_size,
                pad_dim=1,
                padded_size=config.hidden_size,
                unpadded_size=config.unpadded_hidden_size,
                is_storage_transposed=True,
            ),
        )
        if self.qkv_proj_bias is not None:
            set_weight_loader(
                self.qkv_proj_bias,
                fused_qkv_bias_loader(
                    q_size=self.q_size,
                    kv_size=self.kv_size,
                    num_shards=self.world_size,
                    num_kv_heads=self.num_key_value_heads,
                    head_dim=self.head_dim,
                    num_kv_replicas=self.num_kv_replicas,
                ),
            )
        if self.o_proj_bias is not None:
            set_weight_loader(
                self.o_proj_bias,
                with_rank_override(
                    scaled_bias_loader(
                        scale=self.world_size, padded_size=config.hidden_size
                    ),
                    rank=self.rank,
                ),
            )
        set_weight_loader(
            self.q_norm_weight, last_dim_padding_weight_loader(self.head_dim)
        )
        set_weight_loader(
            self.k_norm_weight, last_dim_padding_weight_loader(self.head_dim)
        )

    def _rope_half(
        self, positions: torch.Tensor, device, dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin as [T, head_dim // 2], whichever rotary is in use.

        The two implementations disagree on width: GptOssRotaryEmbedding returns
        the raw [T, head_dim // 2] frequencies, while LlamaRotaryEmbedding
        already returns cat(freqs, freqs) at [T, head_dim]. Both kernels here
        want the half-width form (the decode path slices it, the projection path
        doubles it), so normalise once rather than at each call site.
        """
        cos, sin = self.rotary_emb(positions, device=device, dtype=dtype)
        half = self.head_dim // 2
        return cos[:, :half], sin[:, :half]

    def project_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ) -> None:
        """Project verified context states into this layer's draft K/V cache.

        Uses qkv_proj's in-kernel cache write. That path reshapes a 4-D paged
        cache for the kernel and copies it back afterwards, neither of which is
        a view, so its cost scales with the *allocated* cache rather than with
        the handful of slots in ``slot_mapping``. Sizing the cache to what the
        server can actually use (see --num-gpu-blocks-override in the example
        scripts) is what keeps that affordable on the decode path.

        Projecting plainly and scattering the slots ourselves would be cheaper
        still, but the K written by this path does not match a plain projection
        plus a hand-applied RoPE -- V matches exactly, K does not -- so the
        rotation the kernel fuses into the write is not reproducible from the
        public API today. Correctness wins.
        """
        cos, sin = self._rope_half(
            context_positions, context_states.device, context_states.dtype
        )
        cos_cache = torch.cat([cos, cos], dim=-1).unsqueeze(0)
        sin_cache = torch.cat([sin, sin], dim=-1).unsqueeze(0)
        _, _, _ = NF.qkv_proj(
            hidden=context_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None
            if self.qkv_proj_bias is None
            else self.qkv_proj_bias.unsqueeze(0),
            d_head=self.head_dim,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            num_q_heads=self.num_attention_heads_per_rank,
            num_kv_heads=self.num_key_value_heads_per_rank,
            qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_eps=self.rms_norm_eps,
            qk_norm_pre_rope_q_gamma=self.q_norm_weight.unsqueeze(0),
            qk_norm_pre_rope_k_gamma=self.k_norm_weight.unsqueeze(0),
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            use_block_kv=True,
            block_size=block_size,
            slot_mapping=slot_mapping.to(torch.int32),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict,
        active_mask: torch.Tensor,
        pos_ids: torch.Tensor,
    ) -> torch.Tensor:
        layer_name = f"layers.{self.layer_idx}.self_attn"
        meta = attn_metadata[layer_name]
        block_table = meta["block_table_tensor"]
        block_size = meta["block_size"]
        batch_size = block_table.shape[0]
        query_len = hidden_states.shape[0] // batch_size
        x = hidden_states.view(batch_size, query_len, self.hidden_size).to(self.dtype)

        cos, sin = self._rope_half(positions, x.device, x.dtype)
        half = self.head_dim // 2
        # Contiguous [half, bs, query_len] is the layout the TKG kernel's DMA
        # expects; permute alone leaves non-standard strides.
        cos_kernel = cos.view(batch_size, query_len, half).permute(2, 0, 1).contiguous()
        sin_kernel = sin.view(batch_size, query_len, half).permute(2, 0, 1).contiguous()

        output = NF.attention_decode(
            X=x,
            W_qkv=self.qkv_proj_weight,
            bias_qkv=(
                None if self.qkv_proj_bias is None else self.qkv_proj_bias.unsqueeze(0)
            ),
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=self.rms_norm_eps,
            rmsnorm_QK_pre_rope_W_Q=self.q_norm_weight.unsqueeze(0),
            rmsnorm_QK_pre_rope_W_K=self.k_norm_weight.unsqueeze(0),
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            active_blocks_table=block_table,
            K_cache=self.k_cache,
            V_cache=self.v_cache,
            # Active-only overlay; the prior mask is generated on-chip from
            # pos_ids, so no context-spanning mask is ever materialised.
            attention_mask=active_mask,
            pos_ids=pos_ids,
            softmax_scale=self.scaling,
            update_cache=False,
            W_out=self.o_proj_weight,
            bias_out=(
                None if self.o_proj_bias is None else self.o_proj_bias.unsqueeze(0)
            ),
        )
        # update_cache=False returns query K/V as well; they intentionally do
        # not enter the persistent context cache because proposal tokens are
        # unverified and DFlash attention is bidirectional within the block.
        output = output[0] if isinstance(output, tuple) else output
        self.tp_group.all_reduce(output)
        return output.reshape(-1, self.hidden_size)


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = _make_rmsnorm(config)
        self.self_attn = DFlashAttention(config, layer_idx)
        self.post_attention_layernorm = _make_rmsnorm(config)
        self.mlp = LlamaMLP(config)
        mlp_shards = self.mlp.mlp_tp_size
        mlp_rank = self.mlp.mlp_tp_rank
        per_rank = self.mlp.intermediate_size_per_rank
        for weight in (self.mlp.gate_proj_weight, self.mlp.up_proj_weight):
            set_weight_loader(
                weight,
                with_rank_override(
                    sharding_weight_loader_with_padding(
                        shard_dim=1,
                        shard_size=per_rank,
                        num_shards=mlp_shards,
                        is_storage_transposed=True,
                        pad_dim=0,
                        padded_size=config.hidden_size,
                        unpadded_size=config.unpadded_hidden_size,
                    ),
                    rank=mlp_rank,
                ),
            )
        set_weight_loader(
            self.mlp.down_proj_weight,
            with_rank_override(
                sharding_weight_loader_with_padding(
                    shard_dim=0,
                    shard_size=per_rank,
                    num_shards=mlp_shards,
                    is_storage_transposed=True,
                    pad_dim=1,
                    padded_size=config.hidden_size,
                    unpadded_size=config.unpadded_hidden_size,
                ),
                rank=mlp_rank,
            ),
        )

    def forward(self, hidden_states, positions, attn_metadata, active_mask, pos_ids):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, positions, attn_metadata, active_mask, pos_ids
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=False)
        return residual + hidden_states


class DFlashDraftModel(nn.Module):
    def __init__(self, config: DFlashConfig, start_layer_idx: int):
        super().__init__()
        self.config = config
        self.start_layer_idx = start_layer_idx
        self.target_layer_ids = config.target_layer_ids
        if not self.target_layer_ids:
            raise ValueError("DFlash checkpoint must provide target_layer_ids")
        self.num_speculative_tokens = config.block_size - 1
        self.mask_token_id = config.mask_token_id

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=get_tp_group().device_group,
        )
        set_weight_loader(
            self.embed_tokens.weight,
            embedding_sharding_padding_weight_loader(
                vocab_size_per_rank=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.embed_tokens.tp_size,
                padded_hidden_size=config.hidden_size,
            ),
        )
        self.fc = nn.Linear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            dtype=config.torch_dtype,
        )
        self.hidden_norm = _make_rmsnorm(config)
        self.layers = nn.ModuleList(
            [
                DFlashDecoderLayer(config, start_layer_idx + i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = _make_rmsnorm(config)
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=False,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader_with_padding(
                shard_dim=0,
                shard_size=self.lm_head.out_features_per_rank,
                num_shards=self.lm_head.tp_size,
                pad_dim=1,
                padded_size=config.hidden_size,
                unpadded_size=config.unpadded_hidden_size,
            ),
        )
        sampling_config = config.neuron_config.on_device_sampling_config
        if sampling_config is None:
            raise ValueError("DFlash on Neuron requires on-device greedy sampling")
        self.sampler = Sampler(
            sampling_config, process_group=get_tp_group().device_group
        )

        set_weight_loader(
            self.fc.weight,
            _feature_projection_loader(
                target_hidden_size=config.target_hidden_size,
                draft_hidden_size=config.unpadded_hidden_size,
                padded_hidden_size=config.hidden_size,
                num_features=len(self.target_layer_ids),
            ),
        )
        set_weight_loader(
            self.hidden_norm.weight, last_dim_padding_weight_loader(config.hidden_size)
        )
        set_weight_loader(
            self.norm.weight, last_dim_padding_weight_loader(config.hidden_size)
        )
        # Every RMSNorm gamma is hidden-wide, so each needs the same padding as
        # the model's own norms when the target pads hidden (GPT-OSS 2880->3072).
        # The projections pad inside their own loaders; these do not.
        for layer in self.layers:
            set_weight_loader(
                layer.input_layernorm.weight,
                last_dim_padding_weight_loader(config.hidden_size),
            )
            set_weight_loader(
                layer.post_attention_layernorm.weight,
                last_dim_padding_weight_loader(config.hidden_size),
            )

    @classmethod
    def from_configs(cls, config, start_layer_idx: int, neuron_config=None):
        return cls(DFlashConfig.from_configs(config, neuron_config), start_layer_idx)

    def _build_proposal_block(
        self,
        target_positions: torch.Tensor,
        last_token_indices: torch.Tensor,
        raw_sampled_token_ids: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Derive the proposal block's token ids, positions and attention mask.

        Kept inside the model so it is traced into the draft NEFF. Doing it in
        the proposer instead puts eager XLA ops between the target NEFF and
        this one, which both breaks graph capture and serialises the two
        executables. Eagle3 builds its bonus token inside its own NEFF for the
        same reason.

        Verification leaves ``raw_sampled_token_ids`` as a valid prefix per
        request followed by a rejection sentinel, so the count of valid entries
        gives both the bonus token (the last accepted sample) and how far the
        context boundary has to rewind over rejected tokens.
        """
        batch_size = block_table.shape[0]
        query_len = self.num_speculative_tokens + 1
        device = raw_sampled_token_ids.device

        bonus, last_token_indices = resolve_verified_block(
            raw_sampled_token_ids,
            last_token_indices,
            vocab_size=self.config.vocab_size,
            query_len=query_len,
        )

        # The block is the bonus token followed by mask tokens; the drafter
        # fills the masked slots in one non-causal pass.
        input_ids = torch.full(
            (batch_size, query_len),
            self.mask_token_id,
            dtype=torch.int32,
            device=device,
        )
        input_ids[:, 0] = bonus

        base_positions = target_positions[last_token_indices] + 1
        offsets = torch.arange(query_len, device=device, dtype=base_positions.dtype)
        query_positions = base_positions[:, None] + offsets[None, :]

        # All-ones active mask: every query in the block attends the whole
        # block. This is the block-diffusion property of DFlash — making it
        # causal changes the algorithm, not just the numerics.
        #
        # Only the active-token overlay is materialised here. Passing pos_ids
        # to attention_decode makes the kernel generate the prior causal mask
        # on-chip (iota < pos_id), so this tensor stays
        # [S_tkg, B, q_heads, S_tkg] instead of spanning the whole KV context.
        # At an 8k context that is a ~1000x smaller mask, built without a
        # separate mask kernel per step, and the active overlay is exactly
        # where DFlash's bidirectionality lives.
        q_heads = self.layers[0].self_attn.num_attention_heads_per_rank
        active_mask = torch.ones(
            query_len,
            batch_size,
            q_heads,
            query_len,
            dtype=torch.float32,
            device=device,
        )
        # Every query in the block gets the SAME prior bound: the block's start
        # position. The kernel masks prior slots with `iota < pos_id`, so
        # passing each token's own position would let slot j see cache slots
        # p..p+j-1 -- which for DFlash still hold the previous step's rejected
        # context, because the proposal block runs with update_cache=False and
        # is never written to the cache. The pre-generated mask this replaced
        # took the same uniform min-position bound.
        pos_ids = (
            base_positions.to(torch.float32).unsqueeze(1).expand(batch_size, query_len)
        )
        return (
            input_ids.reshape(-1),
            query_positions.reshape(-1),
            active_mask,
            pos_ids.contiguous(),
        )

    def forward(
        self,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        last_token_indices: torch.Tensor,
        raw_sampled_token_ids: torch.Tensor,
        context_slot_mapping: torch.Tensor,
        attn_metadata: dict,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        first_meta = next(iter(attn_metadata.values()))
        block_table = first_meta["block_table_tensor"]
        block_size = first_meta["block_size"]

        # Padded / invalid slots go to the null block, matching the runner's
        # NULL_BLOCK_ID convention for write-side padding.
        max_slot = self.layers[0].self_attn.k_cache.shape[0] * block_size
        context_slot_mapping = torch.where(
            (context_slot_mapping < 0) | (context_slot_mapping >= max_slot),
            torch.zeros_like(context_slot_mapping),
            context_slot_mapping,
        )

        # Verified context: project the target's captured hidden states into
        # this drafter's own K/V cache at the context positions.
        context_states = self.hidden_norm(self.fc(target_hidden_states))

        input_ids, positions, active_mask, pos_ids = self._build_proposal_block(
            target_positions,
            last_token_indices,
            raw_sampled_token_ids,
            block_table,
            block_size,
        )

        hidden_states = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        # Project each layer's context K/V immediately before that layer
        # attends, rather than projecting all layers up front.
        #
        # qkv_proj(use_block_kv=True) must-alias returns the whole k_cache and
        # v_cache as graph outputs -- 111MB per tensor here. Doing all eight
        # projections in one loop and all eight attentions in another leaves
        # sixteen of those buffers live across the boundary (~1.8GB), which the
        # compiler cannot keep aliased and ends up materialising and spilling.
        # Interleaving keeps one layer's pair live at a time. Correct because
        # layer i's attention only ever reads layer i's context.
        for layer in self.layers:
            if not _SKIP_CONTEXT_KV:
                layer.self_attn.project_context_kv(
                    context_states,
                    target_positions,
                    context_slot_mapping,
                    block_size,
                )
            hidden_states = layer(
                hidden_states, positions, attn_metadata, active_mask, pos_ids
            )
        hidden_states = self.norm(hidden_states)
        batch_size = block_table.shape[0]
        query_len = hidden_states.shape[0] // batch_size
        # Position i of the block predicts token i+1, so slot 0 (the bonus
        # token, already known) is dropped and the remaining num_spec slots
        # are the proposals.
        proposal_states = hidden_states.view(batch_size, query_len, -1)[:, 1:, :]
        logits = self.lm_head(proposal_states.reshape(-1, self.config.hidden_size))
        return self.sampler(logits).to(torch.int32).view(batch_size, query_len - 1)

    def get_kv_spec(self) -> KVSpec:
        return KVSpec(
            layers=[
                LayerSpec(
                    name=f"layers.{layer.self_attn.layer_idx}.self_attn",
                    num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                    head_size=layer.self_attn.head_dim,
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
                for layer in self.layers
            ]
        )

    def bind_kv_cache(self, kv_caches) -> None:
        for layer in self.layers:
            name = f"layers.{layer.self_attn.layer_idx}.self_attn"
            layer.self_attn.k_cache, layer.self_attn.v_cache = kv_caches[name]

    def load_weights(self, checkpoint_path: str, device, cache_dir=None) -> None:
        if not os.path.isdir(checkpoint_path):
            from huggingface_hub import snapshot_download

            checkpoint_path = snapshot_download(checkpoint_path, cache_dir=cache_dir)

        mappings = {
            "fc.weight": "fc.weight",
            "hidden_norm.weight": "hidden_norm.weight",
            "norm.weight": "norm.weight",
        }
        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}"
            model_prefix = f"layers.{i}"
            mappings[f"{model_prefix}.self_attn.qkv_proj_weight"] = [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ]
            mappings[f"{model_prefix}.self_attn.o_proj_weight"] = (
                f"{prefix}.self_attn.o_proj.weight"
            )
            # Only GPT-OSS's drafter carries attention biases; Llama 3.1's sets
            # attention_bias=false and ships no such tensors.
            if layer.self_attn.qkv_proj_bias is not None:
                mappings[f"{model_prefix}.self_attn.qkv_proj_bias"] = [
                    f"{prefix}.self_attn.q_proj.bias",
                    f"{prefix}.self_attn.k_proj.bias",
                    f"{prefix}.self_attn.v_proj.bias",
                ]
            if layer.self_attn.o_proj_bias is not None:
                mappings[f"{model_prefix}.self_attn.o_proj_bias"] = (
                    f"{prefix}.self_attn.o_proj.bias"
                )
            mappings[f"{model_prefix}.self_attn.q_norm_weight"] = (
                f"{prefix}.self_attn.q_norm.weight"
            )
            mappings[f"{model_prefix}.self_attn.k_norm_weight"] = (
                f"{prefix}.self_attn.k_norm.weight"
            )
            mappings[f"{model_prefix}.input_layernorm.weight"] = (
                f"{prefix}.input_layernorm.weight"
            )
            mappings[f"{model_prefix}.post_attention_layernorm.weight"] = (
                f"{prefix}.post_attention_layernorm.weight"
            )
            mappings[f"{model_prefix}.mlp.gate_proj_weight"] = (
                f"{prefix}.mlp.gate_proj.weight"
            )
            mappings[f"{model_prefix}.mlp.up_proj_weight"] = (
                f"{prefix}.mlp.up_proj.weight"
            )
            mappings[f"{model_prefix}.mlp.down_proj_weight"] = (
                f"{prefix}.mlp.down_proj.weight"
            )

        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        result = checkpoint.load_sharded_pipelined(
            get_tp_group().rank_in_group,
            get_tp_group().world_size,
            self,
            mappings,
            device,
            strict=False,
        )
        missing, unexpected = self.load_state_dict(
            result.state_dict, strict=False, assign=True
        )
        allowed_missing = {"embed_tokens.weight", "lm_head.weight"}
        missing_set = set(missing)
        if missing_set != allowed_missing or unexpected or result.unexpected_keys:
            raise RuntimeError(
                "DFlash draft checkpoint weight mismatch: "
                f"missing={sorted(missing_set)}, unexpected="
                f"{sorted(set(unexpected) | set(result.unexpected_keys))}"
            )

    def load_target_weights(self, checkpoint_path: str, device, cache_dir=None) -> None:
        """Load embedding and LM-head tensors omitted by DFlash checkpoints."""
        if not os.path.isdir(checkpoint_path):
            from huggingface_hub import snapshot_download

            checkpoint_path = snapshot_download(checkpoint_path, cache_dir=cache_dir)
        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        shared_weights = nn.Module()
        shared_weights.add_module("embed_tokens", self.embed_tokens)
        shared_weights.add_module("lm_head", self.lm_head)
        # Targets that tie their LM head to the embedding (Qwen3 4B and the
        # other small Qwen3s) ship no lm_head.weight, so both of the drafter's
        # borrowed tensors come from the embedding.
        lm_head_source = (
            "model.embed_tokens.weight"
            if "lm_head.weight" not in checkpoint.get_tensor_names()
            else "lm_head.weight"
        )
        result = checkpoint.load_sharded_pipelined(
            get_tp_group().rank_in_group,
            get_tp_group().world_size,
            shared_weights,
            {
                "embed_tokens.weight": "model.embed_tokens.weight",
                "lm_head.weight": lm_head_source,
            },
            device,
            strict=False,
        )
        if result.missing_keys:
            raise RuntimeError(
                "GPT-OSS target is missing DFlash shared weights: "
                f"{result.missing_keys}"
            )
        _, unexpected = self.load_state_dict(
            result.state_dict, strict=False, assign=True
        )
        if unexpected:
            raise RuntimeError(f"Unexpected target weights for DFlash: {unexpected}")
