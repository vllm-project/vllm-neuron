# SPDX-License-Identifier: Apache-2.0
"""
Hy3 MTP (next-n prediction) draft model for speculative decoding
================================================================

tencent/Hy3-FP8 ships ``num_nextn_predict_layers = 1`` MTP layer after the
decoder stack (``model.layers.80.*``). One draft step for token position i:

    h = eh_proj(cat(enorm(embed(t_{i+1})), hnorm(h_i)))      # h_i: target final hidden
    h = decoder_layer_80(h)                                  # attention + MoE, FP8
    h = final_layernorm(h);  t_{i+2} = argmax(lm_head(h))

The embedding, lm_head and final norm are the target's (aliased, no copy). The
target hands over its last layer's output before the final norm (h_i above is
that output after the target's final norm). Further draft steps feed the MTP
output ``h`` back in place of the target hidden state.

The class follows the fused-draft contract of the Neuron spec-decode proposers
(``vllm_neuron/vllm/spec_decode/mtp.py``, modeled on
``Eagle3LlamaForCausalLM``): one forward runs the KV-update pass over the
target's tokens and then ``num_speculative_tokens - 1`` recurrent decode
passes, unrolled into one graph, and returns ``(tokens [bs, 1+k],
drafts [bs, k], logits)``.

>>> PARALLELISM: same as the target (TP attention, EP experts, SP prefill). <<<
"""

import dataclasses
import logging
import os

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.parallel.neuron_parallel_state import shard_select
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX, QUANTIZED_KV_CACHE_DTYPES
from vllm_neuron.utils.weight_loader import set_weight_loader, sharding_weight_loader

from . import kv_scales
from .config import HYV3Config
from .model import (
    HYV3DecoderLayer,
    HYV3RMSNorm,
    HYV3RotaryEmbedding,
    bind_attention_kv,
    decoder_layer_weight_mappings,
    prepare_sparse_decode,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Spec-decode helpers (same semantics as llama3/eagle3_model.py; copied to keep
# the contrib model self-contained)
# =============================================================================


def extract_accepted_tokens(
    input_ids: torch.Tensor,
    sampling_positions: torch.Tensor,
    raw_sampled_token_ids: torch.Tensor,
    vocab_size: int,
    num_speculative_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Patch the bonus token into ``input_ids`` and move ``sampling_positions``
    back over rejected drafts. Returns ``(input_ids, sampling_positions, bonus)``.

    ``raw_sampled_token_ids`` is ``[bs, k+1]`` (-1 = rejected) after a verify
    step, or ``[bs, 1]`` after a prefill.
    """
    valid_mask = (raw_sampled_token_ids != -1) & (raw_sampled_token_ids < vocab_size)
    valid_count = valid_mask.sum(dim=1)
    last_valid_idx = torch.clamp(valid_count - 1, min=0)
    bonus = (
        raw_sampled_token_ids.gather(1, last_valid_idx.unsqueeze(1).to(torch.long))
        .squeeze(1)
        .to(torch.int32)
    )
    bonus = torch.where(valid_count > 0, bonus, torch.zeros_like(bonus))
    if raw_sampled_token_ids.shape[1] > 1:
        num_rejected = torch.clamp(num_speculative_tokens + 1 - valid_count, min=0)
        sampling_positions = torch.clamp(
            sampling_positions - num_rejected.to(sampling_positions.dtype), min=0
        )
    input_ids = input_ids.scatter(0, sampling_positions.to(torch.long), bonus)
    return input_ids, sampling_positions, bonus


def compute_slot_mapping(
    positions: torch.Tensor, block_table_tensor: torch.Tensor, block_size: int
) -> torch.Tensor:
    """KV slot per position from the block table (-1 for inactive blocks)."""
    block_ids = block_table_tensor.gather(
        dim=1, index=(positions // block_size).view(-1, 1).long()
    ).view(-1)
    slot = block_ids * block_size + (positions % block_size)
    return torch.where(block_ids < 0, torch.full_like(slot, -1), slot)


# =============================================================================
# MTP draft model
# =============================================================================


class HYV3MTPForCausalLM(nn.Module):
    """Hy3 MTP draft (registered as ``HYV3MTPModel``)."""

    def __init__(self, config: HYV3Config, start_layer_idx: int):
        super().__init__()
        self.config = config
        self.start_layer_idx = start_layer_idx
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        H = config.hidden_size

        # <-- MODEL-SPECIFIC: MTP input projection (BF16 in the checkpoint).
        self.enorm = HYV3RMSNorm(H, config.rms_norm_eps, config.torch_dtype)
        self.hnorm = HYV3RMSNorm(H, config.rms_norm_eps, config.torch_dtype)
        # >>> PARALLELISM: eh_proj [H, 2H] column-sharded over TP, output gathered.
        self.eh_proj = neuron_nn.ColumnParallelLinear(
            2 * H,
            H,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=True,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.eh_proj.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.eh_proj.out_features_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=False,
            ),
        )
        # The MTP decoder layer keeps its checkpoint index so its attention
        # layer name ("layers.80.self_attn") matches the draft KV cache.
        self.layers = nn.ModuleList([HYV3DecoderLayer(config, start_layer_idx)])
        self.final_layernorm = HYV3RMSNorm(H, config.rms_norm_eps, config.torch_dtype)
        self.rotary_emb = HYV3RotaryEmbedding(config)

        # Aliased to the target's modules by alias_target_weights(). The target
        # hands over its pre-final-norm hidden state, so the first pass applies
        # the target's final norm (target_norm) before hnorm.
        self.embed_tokens = None
        self.lm_head = None
        self.target_norm = None

        nc = config.neuron_config
        sampling = nc.on_device_sampling_config if nc else None
        self.sampler = (
            Sampler(sampling, process_group=self.tp_group.device_group)
            if sampling is not None
            else None
        )

        # Set by the proposer.
        self.num_speculative_tokens = 1
        self.kv_fill_only = False

    @property
    def layer_name(self) -> str:
        return f"layers.{self.start_layer_idx}.self_attn"

    # ── construction / weights ────────────────────────────────────────────

    @classmethod
    def from_configs(
        cls,
        config: PretrainedConfig,
        start_layer_idx: int,
        neuron_config: NeuronConfig | None = None,
    ) -> "HYV3MTPForCausalLM":
        hy_config = HYV3Config.from_configs(config, neuron_config)
        # The MTP layers follow the decoder stack and are MoE layers.
        n_mtp = getattr(config, "n_predict", None) or getattr(
            config, "num_nextn_predict_layers", 1
        )
        assert n_mtp == 1, f"Hy3 MTP supports one MTP layer, got {n_mtp}"
        hy_config = dataclasses.replace(
            hy_config,
            mlp_layer_types=list(hy_config.mlp_layer_types)
            + ["sparse"] * (start_layer_idx + n_mtp - len(hy_config.mlp_layer_types)),
        )
        return cls(hy_config, start_layer_idx)

    def alias_target_weights(self, target_model: nn.Module) -> None:
        """Share the target's embedding and lm_head (the checkpoint has no MTP copy)."""
        target = getattr(target_model, "_orig_mod", target_model)
        self.embed_tokens = target.model.embed_tokens
        self.lm_head = target.lm_head
        self.target_norm = target.model.norm

    def _weight_mappings(self) -> dict[str, str | list[str]]:
        src = f"model.layers.{self.start_layer_idx}"
        m: dict[str, str | list[str]] = {
            "enorm.weight": f"{src}.enorm.weight",
            "hnorm.weight": f"{src}.hnorm.weight",
            "eh_proj.weight": f"{src}.eh_proj.weight",
            "final_layernorm.weight": f"{src}.final_layernorm.weight",
        }
        m.update(
            decoder_layer_weight_mappings(
                self.layers[0], "layers.0", src, self.config.num_experts
            )
        )
        return m

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None = None
    ) -> None:
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.tp_group.rank_in_group,
            self.world_size,
            self,
            self._weight_mappings(),
            device,
        ).state_dict
        self.load_state_dict(rank_sharded, strict=False, assign=True)
        for name, param in self.named_parameters():
            if param.is_meta:
                raise RuntimeError(f"MTP parameter {name} was not loaded")
        self._load_kv_cache_scales(checkpoint_path, device)
        prepare_sparse_decode(self.layers[0], device)

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None = None
    ) -> None:
        """CPU compile: only the KV scales (decode uses them as graph constants)."""
        self._load_kv_cache_scales(checkpoint_path, device)
        prepare_sparse_decode(self.layers[0], device, from_weights=False)

    def _load_kv_cache_scales(self, checkpoint_path: str, device) -> None:
        """<-- FP8 KV: the MTP layer's K/V scales, entry ``start_layer_idx`` (80)
        of the same profile / checkpoint scales as the target's 80 layers.

        A profile calibrated without the MTP layer has 80 entries and is
        rejected here: calibrate one with ``calibrate_kv_scales_neuron.py
        calibrate --mtp``. (The draft's scales only affect acceptance; the
        output is verified by the target.)
        """
        from vllm.config import get_current_vllm_config

        cache_dtype = get_current_vllm_config().cache_config.cache_dtype
        if cache_dtype not in QUANTIZED_KV_CACHE_DTYPES:
            return
        if not os.path.isdir(checkpoint_path):
            raise ValueError(
                "Hy3 FP8 KV reads its KV scales from a local checkpoint directory; "
                f"got {checkpoint_path!r}"
            )
        i = self.start_layer_idx
        try:
            s = kv_scales.resolve_kv_scales(checkpoint_path, i + 1, FP8_CLAMP_MAX)
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"Hy3 MTP with an FP8 KV cache needs KV scales for the MTP layer "
                f"(model.layers.{i}): {e}. Calibrate a profile that includes it with "
                "scripts/calibrate_kv_scales_neuron.py calibrate --mtp."
            ) from e
        self.layers[0].self_attn.set_kv_scales(s.q_k[i], s.q_v[i], device)
        (logger.warning if s.source == "default" else logger.info)(
            "Hy3 MTP FP8 KV scales: source=%s layer=%d %s q_k=%.4g q_v=%.4g",
            s.source, i, s.detail, s.q_k[i], s.q_v[i],
        )

    # ── KV cache ──────────────────────────────────────────────────────────

    def get_kv_spec(self) -> KVSpec:
        attn = self.layers[0].self_attn
        return KVSpec(
            layers=[
                LayerSpec(
                    name=self.layer_name,
                    num_kv_heads=attn.num_key_value_heads_per_rank,
                    head_size=attn.head_dim,
                    dtype=attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
            ]
        )

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        if self.layer_name not in kv_caches:
            raise KeyError(f"KV cache for layer {self.layer_name} not initialized")
        k_cache, v_cache = kv_caches[self.layer_name]
        bind_attention_kv(
            self.layers[0].self_attn, self.layer_name, k_cache, v_cache,
            sparse=os.environ.get("HY3_SPARSE_DRAFT", "1") != "0",
        )

    # ── forward ───────────────────────────────────────────────────────────

    def _mtp_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        prev_hidden: torch.Tensor,
        attn_metadata: dict,
        is_prefill: bool,
        rank: torch.Tensor | None,
    ) -> torch.Tensor:
        """One MTP layer pass; returns the final-normed hidden states (full rows)."""
        # >>> PARALLELISM: prefill runs the layer on SP token shards <<<
        sp = is_prefill and self.world_size > 1
        embeds = self.embed_tokens(input_ids, scatter_tokens=sp, rank=rank)
        local_positions = shard_select(positions, 0, self.tp_group) if sp else positions
        if sp:
            prev_hidden = shard_select(prev_hidden, 0, self.tp_group)
        # <-- MODEL-SPECIFIC: position 0 has no previous token to predict from.
        embeds = embeds * (local_positions != 0).unsqueeze(-1).to(embeds.dtype)
        h = self.eh_proj(
            torch.cat(
                [self.enorm(embeds), self.hnorm(prev_hidden.to(embeds.dtype))], dim=-1
            )
        )
        position_embeddings = self.rotary_emb(positions, device=h.device, dtype=h.dtype)
        h = self.layers[0](
            h,
            positions,
            position_embeddings,
            attn_metadata,
            is_decode=not is_prefill,
            rank=rank,
        )
        h = self.final_layernorm(h)
        if sp:
            h = self.tp_group.all_gather(h, dim=0)
        return h

    def _lm_head_and_sample(
        self, hidden_states: torch.Tensor, rank: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.lm_head(hidden_states)
        if self.sampler is not None:
            tokens = self.sampler(logits, tp_rank=rank)
        else:
            tokens = torch.argmax(logits, dim=-1)
        return tokens.to(torch.int32), logits

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        initial_target_hidden_states: torch.Tensor,
        attn_metadata: dict | None = None,
        sampling_positions: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        raw_sampled_token_ids: torch.Tensor | None = None,
        prev_sampled_token_ids: torch.Tensor | None = None,
        prev_num_draft_tokens: torch.Tensor | None = None,
        req_indices_per_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """KV-update pass over the target's tokens, then k-1 recurrent passes.

        ``input_ids`` are the target tokens shifted by one (the proposer does
        the shift); ``initial_target_hidden_states`` is the target's final
        hidden state for every token ``[T, H]``.
        """
        positions = positions.to(torch.int32)
        if (
            prev_sampled_token_ids is not None
            and prev_num_draft_tokens is not None
            and req_indices_per_token is not None
        ):
            positions, attn_metadata = (
                NF.correct_spec_decode_positions_and_slot_mapping(
                    positions,
                    attn_metadata,
                    prev_sampled_token_ids,
                    prev_num_draft_tokens,
                    req_indices_per_token,
                    self.config.vocab_size,
                )
            )

        bonus = None
        if raw_sampled_token_ids is not None:
            input_ids, sampling_positions, bonus = extract_accepted_tokens(
                input_ids,
                sampling_positions,
                raw_sampled_token_ids,
                self.config.vocab_size,
                self.num_speculative_tokens,
            )

        meta = attn_metadata[self.layer_name]
        is_prefill = meta["max_query_len"] > meta["decode_token_threshold"]

        # Initial pass: writes the MTP layer's KV for every target token.
        # <-- MODEL-SPECIFIC: the MTP layer consumes the target's normed output.
        hidden = self._mtp_step(
            input_ids,
            positions,
            self.target_norm(initial_target_hidden_states),
            attn_metadata,
            is_prefill,
            rank,
        )
        if self.kv_fill_only:
            placeholder = torch.zeros(
                (sampling_positions.shape[0], self.num_speculative_tokens),
                dtype=torch.int32,
                device=input_ids.device,
            )
            return placeholder, placeholder, None

        hidden = hidden[sampling_positions]
        cur_positions = positions[sampling_positions]
        draft, _ = self._lm_head_and_sample(hidden, rank)
        tokens = [bonus, draft] if bonus is not None else [draft]

        block_table = meta["block_table_tensor"]
        block_size = meta["block_size"]
        max_blocks_per_seq = meta["max_blocks_per_seq"]
        # Recurrent passes (unrolled by torch.compile; k is a trace-time constant).
        for _ in range(self.num_speculative_tokens - 1):
            cur_positions = cur_positions + 1
            step_meta = {
                self.layer_name: {
                    "block_table_tensor": block_table,
                    "slot_mapping": compute_slot_mapping(
                        cur_positions, block_table, block_size
                    ),
                    "max_query_len": 1,
                    "block_size": block_size,
                    "max_blocks_per_seq": max_blocks_per_seq,
                    "decode_token_threshold": 1,
                }
            }
            hidden = self._mtp_step(
                draft, cur_positions, hidden, step_meta, False, rank
            )
            draft, _ = self._lm_head_and_sample(hidden, rank)
            tokens.append(draft)

        stacked = torch.stack(tokens, dim=1)
        drafts_only = torch.stack(tokens[1:] if bonus is not None else tokens, dim=1)
        return stacked, drafts_only, None
