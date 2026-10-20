# SPDX-License-Identifier: Apache-2.0
"""Qwen3-based DFlash drafter for block-parallel speculative decoding.

The drafter projects target auxiliary hidden states into its context KV cache,
then predicts a masked token block with non-causal attention. Its embedding and
LM head are aliased from the target model.

Two optional checkpoint features use the upstream vLLM DFlash format:

- A reduced draft vocabulary. The config sets ``draft_vocab_size``. The
  checkpoint ships its own ``lm_head`` and a ``d2t`` tensor, where
  ``target_id = draft_id + d2t[draft_id]``.
- A separate mask embedding. The checkpoint directory holds
  ``mask_embedding.pt``, which replaces the embedding of ``mask_token_id``.
"""

import os

import torch
import torch.nn as nn
from vllm.distributed.parallel_state import get_tp_group

from vllm_neuron.functional.attention.attention_decode_mask import (
    gen_attention_decode_mask,
)
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.model.qwen3.config import DFlashConfig
from vllm_neuron.model.qwen3.model import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
import vllm_neuron.nn as neuron_nn
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _correct_target_positions_for_async(
    target_positions: torch.Tensor,
    prev_sampled_token_ids: torch.Tensor | None,
    prev_num_draft_tokens: torch.Tensor | None,
    req_indices_per_token: torch.Tensor | None,
    vocab_size: int,
) -> torch.Tensor:
    """Correct optimistic target positions using previous rejection results."""
    if (
        prev_sampled_token_ids is None
        or prev_num_draft_tokens is None
        or req_indices_per_token is None
    ):
        return target_positions

    prev_valid_mask = (prev_sampled_token_ids != -1) & (
        prev_sampled_token_ids < vocab_size
    )
    prev_valid_count = prev_valid_mask.sum(dim=1)
    prev_num_rejected = (1 + prev_num_draft_tokens) - prev_valid_count
    return target_positions - prev_num_rejected[req_indices_per_token]


def _current_seq_lens(
    bonus_pos: torch.Tensor,
    raw_sampled_token_ids: torch.Tensor,
    prev_sampled_token_ids: torch.Tensor | None,
    prev_num_draft_tokens: torch.Tensor | None,
    cur_num_draft_tokens: torch.Tensor | None,
    vocab_size: int,
    num_speculative_tokens: int,
    valid_per_req: torch.Tensor | None = None,
) -> torch.Tensor:
    """Derive sequence lengths after removing rejected draft positions."""
    seq_lens = bonus_pos + 1
    if raw_sampled_token_ids.shape[1] <= 1:
        return torch.clamp(seq_lens, min=0)

    if valid_per_req is None:
        valid_per_req = (
            (raw_sampled_token_ids != -1) & (raw_sampled_token_ids < vocab_size)
        ).sum(dim=1)
    num_accepted = torch.clamp(valid_per_req - 1, min=0)
    if cur_num_draft_tokens is not None:
        proposed = cur_num_draft_tokens
    elif prev_sampled_token_ids is None and prev_num_draft_tokens is not None:
        proposed = prev_num_draft_tokens
    else:
        proposed = torch.full_like(valid_per_req, num_speculative_tokens)
    num_rejected = torch.clamp(proposed - num_accepted, min=0)
    return torch.clamp(seq_lens - num_rejected, min=0)


def _paged_slots(
    positions: torch.Tensor,
    req_ids: torch.Tensor,
    kv_block_table: torch.Tensor,
    kv_block_size: int,
    kv_max_blocks_per_seq: int,
) -> torch.Tensor:
    """Map absolute positions to paged KV-cache slots for each request."""
    kv_block_table_flat = kv_block_table.contiguous().reshape(-1)
    kv_block_numbers = positions // kv_block_size
    kv_block_numbers = torch.clamp(kv_block_numbers, max=kv_max_blocks_per_seq - 1)
    flat_idx = req_ids * kv_max_blocks_per_seq + kv_block_numbers
    kv_block_ids = torch.index_select(kv_block_table_flat, 0, flat_idx)
    slot = kv_block_ids * kv_block_size + (positions % kv_block_size)
    slot = torch.where(kv_block_ids < 0, torch.full_like(slot, -1), slot)
    return slot.clamp(min=0)


# =============================================================================
# Decoder Layer
# =============================================================================


class DFlashQwen3DecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn.forward_decode(
            hidden_states=hidden_states,
            positions=None,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=False)
        hidden_states = residual + hidden_states
        return hidden_states


def _find_checkpoint_file(checkpoint_path: str, file_name: str) -> str | None:
    """Return the path of an optional file in a local or HuggingFace checkpoint."""
    if os.path.isdir(checkpoint_path):
        path = os.path.join(checkpoint_path, file_name)
        return path if os.path.isfile(path) else None
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        return hf_hub_download(checkpoint_path, file_name)
    except EntryNotFoundError:
        return None


# =============================================================================
# Model Backbone
# =============================================================================


class DFlashQwen3Model(nn.Module):
    """DFlash drafter backbone with a target-owned token embedding."""

    def __init__(self, config: DFlashConfig, start_layer_idx: int = 0):
        super().__init__()
        self.config = config
        dtype = config.torch_dtype

        # Offset cache names past target layers while keeping checkpoint indices local.
        self.layers = nn.ModuleList(
            [
                DFlashQwen3DecoderLayer(config, start_layer_idx + i)
                for i in range(config.num_hidden_layers)
            ]
        )

        self.fc = neuron_nn.ColumnParallelLinear(
            len(config.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
            gather_output=True,
            dtype=dtype,
            tp_group=get_tp_group().device_group,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
        self.norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
        self.rotary_emb = Qwen3RotaryEmbedding(config)

        # The checkpoint omits this embedding; alias_target_head supplies its weight.
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=dtype,
            tp_group=get_tp_group().device_group,
        )
        # Set by DFlashQwen3ForCausalLM.load_weights when the checkpoint ships
        # mask_embedding.pt. None means mask slots use embed_tokens[mask_token_id].
        self.register_buffer("mask_embedding", None, persistent=False)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed the flattened draft block with the target's embedding table."""
        embeds = self.embed_tokens(input_ids, scatter_tokens=False)
        if getattr(self, "mask_embedding", None) is not None:
            is_mask = (input_ids == self.config.mask_token_id).unsqueeze(-1)
            embeds = torch.where(is_mask, self.mask_embedding.to(embeds.dtype), embeds)
        return embeds

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,  # [B*S_q] query-block token ids ([bonus, mask...])
        positions: torch.Tensor,  # [B*S_q] query-token positions
        attn_metadata: dict,
    ) -> torch.Tensor:
        """Run a draft block over context KV populated by the wrapper."""
        hidden_states = self.embed_input_ids(input_ids)
        cos, sin = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )

        # Build one non-causal mask on device and share it across draft layers.
        # Tests may provide a precomputed mask through attention metadata.
        attn0 = self.layers[0].self_attn
        layer0_meta = attn_metadata[f"layers.{attn0.layer_idx}.self_attn"]
        attention_mask = layer0_meta.get("attention_mask", None)
        if attention_mask is None:
            bs = layer0_meta["block_table_tensor"].shape[0]
            block_size = layer0_meta["block_size"]
            s_prior = layer0_meta["max_blocks_per_seq"] * block_size
            s_active = positions.shape[0] // bs
            attention_mask = gen_attention_decode_mask(
                pos_ids=positions.view(1, bs * s_active).to(torch.float32),
                bs=bs,
                q_head=attn0.num_attention_heads_per_rank,
                s_active=s_active,
                s_prior=s_prior,
                block_len=block_size,
                is_causal=False,
            )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, (cos, sin), attn_metadata, attention_mask
            )
        return self.norm(hidden_states)

    def _build_fused_kv_buffers(self) -> None:
        """Stack layer K/V weights for one context projection matmul."""

        first_qkv_weight = self.layers[0].self_attn.qkv_proj_weight
        if first_qkv_weight.is_meta:
            fused_kv_size = sum(2 * layer.self_attn.kv_size for layer in self.layers)
            self._fused_kv_weight = first_qkv_weight.new_empty(
                (fused_kv_size, first_qkv_weight.shape[0])
            )
            return

        kv_weights = []
        for layer in self.layers:
            attn = layer.self_attn
            q_size, kv_size = attn.q_size, attn.kv_size

            qkv_cpu = attn.qkv_proj_weight.detach().cpu()
            k_w = qkv_cpu[:, q_size : q_size + kv_size]
            v_w = qkv_cpu[:, q_size + kv_size :]
            kv_weights.append(torch.cat([k_w.t(), v_w.t()], dim=0))

        self._fused_kv_weight = torch.cat(kv_weights, dim=0).to(first_qkv_weight.device)

    @torch.no_grad()
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor,
    ) -> None:
        """Project target auxiliary states into every draft layer's KV cache.

        Args:
            context_states: Concatenated target states ``[num_ctx, n_target * H]``.
            context_positions: Position per context token ``[num_ctx]``.
            context_slot_mapping: Paged-cache slot per token ``[num_ctx]``.
        """

        assert hasattr(self, "_fused_kv_weight"), (
            "call bind_kv_cache (which builds the fused KV weight) before precompute"
        )

        L = len(self.layers)
        attn0 = self.layers[0].self_attn
        # The fused weight contains only this rank's KV heads.
        nkv, hd = attn0.num_key_value_heads_per_rank, attn0.head_dim
        num_ctx = context_states.shape[0]

        ctx = self.hidden_norm(self.fc(context_states))
        all_kv_flat = nn.functional.linear(ctx, self._fused_kv_weight)

        all_kv = all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4)
        all_k = all_kv[0]
        all_v = all_kv[1]

        cos, sin = self.rotary_emb(
            context_positions, device=ctx.device, dtype=ctx.dtype
        )
        cos = cos[:, None, :]
        sin = sin[:, None, :]

        block_size = attn0.k_cache.shape[2]

        block_idx = context_slot_mapping // block_size
        pos_idx = context_slot_mapping % block_size
        head_idx = torch.arange(nkv, device=ctx.device).repeat_interleave(num_ctx)

        block_idx_put = block_idx.repeat(nkv)
        pos_idx_put = pos_idx.repeat(nkv)

        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            k = attn.k_norm(all_k[i])
            k = k * cos + _rotate_half(k) * sin
            v = all_v[i]

            k_flat = k.transpose(0, 1).reshape(-1, hd)
            v_flat = v.transpose(0, 1).reshape(-1, hd)

            attn.k_cache.index_put_(
                (block_idx_put, head_idx, pos_idx_put), k_flat.to(attn.k_cache.dtype)
            )
            attn.v_cache.index_put_(
                (block_idx_put, head_idx, pos_idx_put), v_flat.to(attn.v_cache.dtype)
            )


# =============================================================================
# CausalLM Wrapper
# =============================================================================


class DFlashQwen3ForCausalLM(nn.Module):
    """DFlash drafter with block forward and token selection."""

    # These checkpoint weights are supplied by the target after loading.
    has_own_embed_tokens = False
    has_own_lm_head = False
    # Set in __init__ when the config has a reduced draft vocabulary.
    has_vocab_mapping = False

    def __init__(self, config: DFlashConfig, start_layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.model = DFlashQwen3Model(config, start_layer_idx=start_layer_idx)
        self.num_speculative_tokens = config.block_size - 1

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config
            if config.neuron_config is not None
            else None
        )
        self._return_logits = (
            config.neuron_config is not None
            and config.neuron_config.debug_logits_dir is not None
        )

        # A reduced draft vocabulary needs its own lm_head and the d2t id map.
        self.has_vocab_mapping = config.draft_vocab_size not in (
            None,
            config.vocab_size,
        )
        if self.has_vocab_mapping:
            self.has_own_lm_head = True
            self.register_buffer(
                "draft_to_target",
                torch.zeros(config.draft_vocab_size, dtype=torch.int32),
                persistent=False,
            )

        # Keep logits sharded when the distributed sampler can consume them.
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.draft_vocab_size if self.has_vocab_mapping else config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=self.on_device_sampling_config is None,
            tp_group=self.tp_group.device_group,
        )

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=self.tp_group.device_group,
            )

    @torch.no_grad()
    def forward(
        self,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        last_token_indices: torch.Tensor,
        attn_metadata: dict,
        raw_sampled_token_ids: torch.Tensor,
        prev_sampled_token_ids: torch.Tensor,
        prev_num_draft_tokens: torch.Tensor,
        req_indices_per_token: torch.Tensor,
        cur_num_draft_tokens: torch.Tensor,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Prepare and run one complete block-parallel proposal."""
        (
            input_ids,
            positions,
            draft_attn_metadata,
            keep_idx,
            context_positions,
            context_slot_mapping,
            bonus_token_ids,
        ) = self._prepare_proposal_inputs(
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            last_token_indices=last_token_indices,
            attn_metadata=attn_metadata,
            raw_sampled_token_ids=raw_sampled_token_ids,
            prev_sampled_token_ids=prev_sampled_token_ids,
            prev_num_draft_tokens=prev_num_draft_tokens,
            req_indices_per_token=req_indices_per_token,
            cur_num_draft_tokens=cur_num_draft_tokens,
        )

        self.model.precompute_and_store_context_kv(
            target_hidden_states,
            context_positions,
            context_slot_mapping,
        )

        hidden_states = self.model(
            input_ids,
            positions,
            draft_attn_metadata,
        )

        drafts_hidden = torch.index_select(hidden_states, 0, keep_idx)
        logits = self.lm_head(drafts_hidden)
        draft_token_ids = self._sample_draft_tokens(logits)

        batch_size = last_token_indices.shape[0]
        block_len = self.config.block_size
        draft_token_ids = draft_token_ids.view(batch_size, block_len - 1).to(
            torch.int32
        )
        draft_and_bonus_token_ids = torch.cat(
            [bonus_token_ids.view(batch_size, 1), draft_token_ids], dim=1
        )
        if self._return_logits:
            return draft_and_bonus_token_ids, draft_token_ids, logits
        return draft_and_bonus_token_ids, draft_token_ids

    def _prepare_proposal_inputs(
        self,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        last_token_indices: torch.Tensor,
        attn_metadata: dict,
        raw_sampled_token_ids: torch.Tensor,
        prev_sampled_token_ids: torch.Tensor,
        prev_num_draft_tokens: torch.Tensor,
        req_indices_per_token: torch.Tensor,
        cur_num_draft_tokens: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Build proposal positions, slots, metadata, and query tokens on device."""
        batch_size = last_token_indices.shape[0]
        device = target_hidden_states.device
        block_len = self.config.block_size
        vocab_size = self.config.vocab_size

        valid_mask = (raw_sampled_token_ids != -1) & (
            raw_sampled_token_ids < vocab_size
        )
        valid_count = valid_mask.sum(dim=1)
        target_positions = _correct_target_positions_for_async(
            target_positions,
            prev_sampled_token_ids,
            prev_num_draft_tokens,
            req_indices_per_token,
            vocab_size,
        )

        bonus_pos = target_positions[last_token_indices]
        seq_lens = _current_seq_lens(
            bonus_pos,
            raw_sampled_token_ids,
            prev_sampled_token_ids,
            prev_num_draft_tokens,
            cur_num_draft_tokens,
            vocab_size,
            self.num_speculative_tokens,
            valid_per_req=valid_count,
        )
        block_offsets = torch.arange(block_len, device=device)
        block_positions = (seq_lens[:, None] + block_offsets[None, :]).reshape(-1)

        attn0 = self.model.layers[0].self_attn
        first_layer_name = f"layers.{attn0.layer_idx}.self_attn"
        base_meta = attn_metadata[first_layer_name]
        kv_block_size = attn0.k_cache.shape[2]
        kv_block_table = base_meta["block_table_tensor"][0:batch_size].to(torch.int32)
        kv_max_blocks_per_seq = int(base_meta["max_blocks_per_seq"])

        # Recover each flattened target row's request id from the per-request
        # last-row boundaries, then map each token/row to its paged KV slot.
        num_ctx = target_hidden_states.shape[0]
        ctx_rows = torch.arange(num_ctx, device=device)
        ctx_req = (ctx_rows[:, None] > last_token_indices[None, :]).sum(dim=1)
        ctx_is_pad = ctx_req >= batch_size
        ctx_req = torch.clamp(ctx_req, max=batch_size - 1)
        context_slot_mapping = _paged_slots(
            target_positions,
            ctx_req,
            kv_block_table,
            kv_block_size,
            kv_max_blocks_per_seq,
        )
        context_slot_mapping = torch.where(
            ctx_is_pad, torch.zeros_like(context_slot_mapping), context_slot_mapping
        )

        block_req = torch.arange(batch_size * block_len, device=device) // block_len
        block_slot_mapping = _paged_slots(
            block_positions,
            block_req,
            kv_block_table,
            kv_block_size,
            kv_max_blocks_per_seq,
        )
        per_layer = {
            "slot_mapping": block_slot_mapping,
            "block_size": kv_block_size,
            "block_table_tensor": kv_block_table,
            "max_blocks_per_seq": kv_max_blocks_per_seq,
        }
        draft_attn_metadata = {
            f"layers.{layer.self_attn.layer_idx}.self_attn": per_layer
            for layer in self.model.layers
        }

        # Extract each request's last valid sampled token as the bonus token.
        last_valid_idx = torch.clamp(valid_count - 1, min=0)
        bonus_token_ids = (
            raw_sampled_token_ids.gather(1, last_valid_idx.unsqueeze(1).to(torch.long))
            .squeeze(1)
            .to(torch.int32)
        )
        bonus_token_ids = torch.where(
            valid_count > 0,
            bonus_token_ids,
            torch.zeros_like(bonus_token_ids),
        )

        # Build [bonus, mask, mask, ..., mask] block for each request, flattened.
        mask_token_ids = torch.full(
            (batch_size, block_len - 1),
            self.config.mask_token_id,
            dtype=bonus_token_ids.dtype,
            device=device,
        )
        input_ids = torch.cat(
            [bonus_token_ids.view(batch_size, 1), mask_token_ids], dim=1
        ).reshape(-1)

        # Used during sampling to select only draft positions from the block output.
        keep_offsets = torch.arange(1, block_len, device=device)
        keep_idx = (
            torch.arange(batch_size, device=device)[:, None] * block_len
            + keep_offsets[None, :]
        ).reshape(-1)

        return (
            input_ids,
            block_positions,
            draft_attn_metadata,
            keep_idx,
            target_positions,
            context_slot_mapping,
            bonus_token_ids,
        )

    def _sample_draft_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        if self.on_device_sampling_config is not None:
            token_ids = self.sampler(
                logits,
                tp_rank=self.tp_group.rank_in_group_tensor,
            ).to(torch.int32)
        else:
            token_ids = torch.argmax(logits, dim=-1).to(torch.int32)
        if self.has_vocab_mapping:
            token_ids = self.draft_to_target[token_ids.long()]
        return token_ids

    @classmethod
    def from_configs(
        cls,
        config,
        start_layer_idx: int = 0,
        neuron_config: NeuronConfig | None = None,
    ):
        return cls(
            DFlashConfig.from_configs(config, neuron_config),
            start_layer_idx=start_layer_idx,
        )

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None = None
    ) -> None:
        """
        Load the z-lab checkpoint with rank-sharded, fused-QKV mapping.
        """

        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        mappings = {}
        mappings["model.fc.weight"] = "fc.weight"
        mappings["model.hidden_norm.weight"] = "hidden_norm.weight"
        mappings["model.norm.weight"] = "norm.weight"

        for layer_id in range(len(self.model.layers)):
            ckpt_prefix = f"layers.{layer_id}"
            mappings[f"model.layers.{layer_id}.self_attn.qkv_proj_weight"] = [
                f"{ckpt_prefix}.self_attn.q_proj.weight",
                f"{ckpt_prefix}.self_attn.k_proj.weight",
                f"{ckpt_prefix}.self_attn.v_proj.weight",
            ]
            mappings[f"model.layers.{layer_id}.self_attn.o_proj_weight"] = (
                f"{ckpt_prefix}.self_attn.o_proj.weight"
            )

            mappings[f"model.layers.{layer_id}.self_attn.q_norm.weight"] = (
                f"{ckpt_prefix}.self_attn.q_norm.weight"
            )
            mappings[f"model.layers.{layer_id}.self_attn.k_norm.weight"] = (
                f"{ckpt_prefix}.self_attn.k_norm.weight"
            )

            mappings[f"model.layers.{layer_id}.mlp.gate_proj_weight"] = (
                f"{ckpt_prefix}.mlp.gate_proj.weight"
            )
            mappings[f"model.layers.{layer_id}.mlp.up_proj_weight"] = (
                f"{ckpt_prefix}.mlp.up_proj.weight"
            )
            mappings[f"model.layers.{layer_id}.mlp.down_proj_weight"] = (
                f"{ckpt_prefix}.mlp.down_proj.weight"
            )

            mappings[f"model.layers.{layer_id}.input_layernorm.weight"] = (
                f"{ckpt_prefix}.input_layernorm.weight"
            )
            mappings[f"model.layers.{layer_id}.post_attention_layernorm.weight"] = (
                f"{ckpt_prefix}.post_attention_layernorm.weight"
            )

        # Embedding and LM heads are shared with target.
        expected_absent = set()
        if not self.has_own_embed_tokens:
            expected_absent.add("model.embed_tokens.weight")
        if not self.has_own_lm_head:
            expected_absent.add("lm_head.weight")

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir=cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            tp_rank,
            tp_size,
            self,
            mappings,
            device,
            strict=False,
        ).state_dict
        missing, unexpected = self.load_state_dict(
            rank_sharded, strict=False, assign=True
        )
        real_missing = set(missing) - expected_absent
        assert not real_missing, f"unexpected missing weights: {sorted(real_missing)}"
        assert not unexpected, (
            f"unexpected weights in checkpoint: {sorted(set(unexpected))}"
        )

        if self.has_vocab_mapping:
            self._load_vocab_map(checkpoint, device)
        self.model.mask_embedding = self._read_mask_embedding(checkpoint_path, device)

        with torch.no_grad():
            if not self.has_own_embed_tokens and self.model.embed_tokens.weight.is_meta:
                self.model.embed_tokens.weight = nn.Parameter(
                    torch.empty_like(self.model.embed_tokens.weight, device=device),
                    requires_grad=False,
                )
            if not self.has_own_lm_head and self.lm_head.weight.is_meta:
                self.lm_head.weight = nn.Parameter(
                    torch.empty_like(self.lm_head.weight, device=device),
                    requires_grad=False,
                )

    def _load_vocab_map(self, checkpoint: SafetensorsCheckpoint, device) -> None:
        """Build ``draft_to_target`` from the checkpoint ``d2t`` offsets."""
        d2t = checkpoint.get_tensor("d2t")
        if d2t is None:
            raise ValueError(
                "The DFlash config sets draft_vocab_size, but the checkpoint has no "
                "d2t tensor. Add d2t, or remove draft_vocab_size from the config."
            )
        if d2t.numel() != self.config.draft_vocab_size:
            raise ValueError(
                f"The d2t tensor has {d2t.numel()} entries, but draft_vocab_size is "
                f"{self.config.draft_vocab_size}. Make the two values the same."
            )
        draft_ids = torch.arange(d2t.numel(), dtype=torch.long)
        # Convert the dtype on the CPU, then move. The device copy does not
        # convert the dtype.
        draft_to_target = (draft_ids + d2t.long()).to(torch.int32)
        self.draft_to_target = draft_to_target.to(device)

    def _read_mask_embedding(self, checkpoint_path: str, device) -> torch.Tensor | None:
        """Read the optional ``mask_embedding.pt`` file of the checkpoint."""
        path = _find_checkpoint_file(checkpoint_path, "mask_embedding.pt")
        if path is None:
            return None
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            file_token_id = state.get("mask_token_id", self.config.mask_token_id)
            if file_token_id != self.config.mask_token_id:
                raise ValueError(
                    f"mask_embedding.pt has mask_token_id {file_token_id}, but the "
                    f"config has {self.config.mask_token_id}. Make the two values "
                    "the same."
                )
            state = state["embedding"]
        embedding = state.reshape(-1)
        if embedding.numel() != self.config.hidden_size:
            raise ValueError(
                f"mask_embedding.pt has {embedding.numel()} values, but hidden_size "
                f"is {self.config.hidden_size}. Use a mask embedding for this draft."
            )
        return embedding.to(self.config.torch_dtype).to(device)

    def alias_target_head(self, target_model: nn.Module) -> None:
        target_embed = target_model.model.embed_tokens
        target_lm_head = target_model.lm_head

        if not self.has_own_embed_tokens:
            dst = self.model.embed_tokens.weight
            src = target_embed.weight
            assert dst.shape == src.shape, (
                f"embed alias shape mismatch: draft {tuple(dst.shape)} != "
                f"target {tuple(src.shape)}"
            )
            self.model.embed_tokens.weight = target_embed.weight

        if not self.has_own_lm_head:
            # The draft borrows the target's lm_head weight, so it must use the
            # target's weight layout: "out_in" -> F.linear, "in_out" -> matmul.
            # The dense MXFP8 target stores lm_head as "in_out" (see
            # model_mxfp8.py); BF16 targets keep the default "out_in". Mirror the
            # target's layout before aliasing so the shapes match and the draft's
            # forward runs the matching matmul path.
            target_layout = getattr(target_lm_head, "weight_layout", "out_in")
            assert (
                self.lm_head.in_features == target_lm_head.in_features
                and self.lm_head.out_features == target_lm_head.out_features
            ), (
                "lm_head alias dim mismatch: draft "
                f"(in={self.lm_head.in_features}, out={self.lm_head.out_features}) "
                f"!= target (in={target_lm_head.in_features}, "
                f"out={target_lm_head.out_features})"
            )
            self.lm_head.weight_layout = target_layout
            self.lm_head.weight = target_lm_head.weight

    def get_kv_spec(self) -> KVSpec:
        layers = []
        for layer in self.model.layers:
            attn = layer.self_attn
            layer_name = f"layers.{attn.layer_idx}.self_attn"
            layers.append(
                LayerSpec(
                    name=layer_name,
                    num_kv_heads=attn.num_key_value_heads_per_rank,
                    head_size=attn.head_dim,
                    dtype=attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
        for layer in self.model.layers:
            attn = layer.self_attn
            layer_name = f"layers.{attn.layer_idx}.self_attn"
            if layer_name not in kv_caches:
                raise RuntimeError(f"KV cache for layer {layer_name} not initialized")
            attn.k_cache = kv_caches[layer_name][0]
            attn.v_cache = kv_caches[layer_name][1]

        self.model._build_fused_kv_buffers()
