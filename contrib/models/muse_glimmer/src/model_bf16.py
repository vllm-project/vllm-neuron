# SPDX-License-Identifier: Apache-2.0
"""Muse-Glimmer-30B text decoder for vLLM Neuron (BF16, TP + SP).

Mimics the validated path (HF transformers MuseGlimmer under torch.compile fullgraph=True on
trn3): every layer is the same plain-PyTorch math as HF — no NKI kernels in the model. The one
thing vLLM forces to differ is the KV cache: a paged cache read through block tables and
written through slot mappings, instead of HF's StaticCache.

Plugin structure (from vllm_neuron/model/qwen3/model.py) kept for the runner contract: config,
factory, get_kv_spec / bind_kv_cache, load_weights, SP prefill (all-gather / reduce-scatter),
vocab-sharded embedding and column-parallel LM head (plain PyTorch + collectives).

Muse-Glimmer specifics (as in HF MuseGlimmerTextModel):
  - gated attention: o_proj(attn * sigmoid(gate_proj(x)))
  - weight-less Q/K RMSNorm; Q scaled by qk_scale_factor (3.87) — folded with 1/sqrt(Dh)
  - [local x3, global] layers: local = sliding window 2048 + RoPE theta 500000, global = NoPE
  - four centered norms per layer (x * (1 + w)); post-* norms use post_norm_eps
  - weight-less norm after the embedding; standard final RMSNorm
  - logits: cap * tanh(logits * output_multiplier / cap), cap = final_logit_softcapping
"""

import logging
import math

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.functional.merge_multimodal_embeddings import (
    _merge_multimodal_embeddings_torch,
)
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

from .config import MuseGlimmerTextConfig

logger = logging.getLogger(__name__)


# ── norms ────────────────────────────────────────────────────────────────────


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """HF MuseGlimmer RMS core, in fp32: x * (mean(x^2) + eps)^-0.5."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class CenteredRMSNorm(
    nn.Module
):  # <-- MUSE: HF MuseGlimmerTextCenteredRMSNorm, x * (1 + w)
    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (_rms(x, self.eps) * (1.0 + self.weight.float())).to(x.dtype)


class RMSNorm(nn.Module):  # <-- MUSE: HF MuseGlimmerRMSNorm (final norm), x * w
    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (_rms(x, self.eps) * self.weight.float()).to(x.dtype)


# ── RoPE ─────────────────────────────────────────────────────────────────────


class RotaryEmbedding(nn.Module):
    """Default RoPE, rotate_half (contiguous halves) layout, like HF."""

    def __init__(self, config: MuseGlimmerTextConfig):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim

    def forward(self, positions: torch.Tensor, device, dtype):
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
                / self.head_dim
            )
        )
        freqs = (inv_freq[:, None] @ positions[None, :].float()).transpose(
            0, 1
        )  # [T, Dh/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, Dh]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [T, N, Dh]; cos/sin [T, Dh]. rotate_half layout, like HF apply_rotary_pos_emb."""
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (x * cos[:, None, :] + rot * sin[:, None, :]).to(x.dtype)


# ── attention ────────────────────────────────────────────────────────────────


class MuseGlimmerAttention(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_name = f"layers.{layer_idx}.self_attn"
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.eps = config.rms_norm_eps
        self.use_rope = config.has_rope(layer_idx)  # <-- MUSE
        self.sliding_window = (
            config.sliding_window if config.is_sliding(layer_idx) else None
        )
        # <-- MUSE: softmax scale folded into the Q-norm gamma; kernels run at scale 1.0
        self.q_gamma_value = config.qk_scale_factor * config.head_dim**-0.5

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
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
        self.qkv_split_indices = [self.q_size, self.q_size + self.kv_size]
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(
                self.hidden_size, self.q_size + 2 * self.kv_size, dtype=self.dtype
            )
        )
        self.gate_proj_weight = (
            nn.Parameter(  # <-- MUSE: attention output gate, sharded like Q
                torch.empty(self.hidden_size, self.q_size, dtype=self.dtype)
            )
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(self.q_size, self.hidden_size, dtype=self.dtype)
        )

        self.k_cache = None
        self.v_cache = None

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
            self.gate_proj_weight,
            sharding_weight_loader(
                shard_dim=1,
                shard_size=self.q_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.q_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    def _qk_gammas(self, device):
        # <-- MUSE: weight-less Q/K norm. K gamma 1; Q gamma = full softmax scale.
        q = torch.full(
            (1, self.head_dim), self.q_gamma_value, dtype=self.dtype, device=device
        )
        k = torch.ones((1, self.head_dim), dtype=self.dtype, device=device)
        return q, k

    def forward(self, hidden_states, positions, position_embeddings, attn_metadata):
        md = attn_metadata[self.layer_name]
        is_decode = md["max_query_len"] <= md["decode_token_threshold"]
        if not is_decode and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(
                hidden_states, dim=0
            )  # SP shard -> full sequence
        out = self._attention(
            hidden_states, positions, position_embeddings, md
        )  # [T, H], partial over heads
        if self.world_size > 1:
            if is_decode:
                self.tp_group.all_reduce(out)
            else:
                out = self.tp_group.reduce_scatter(out, dim=0)  # back to SP layout
        return out

    def _attention(self, hidden_states, positions, position_embeddings, md):
        """Same math as the validated HF path (MuseGlimmerTextAttention, eager/SDPA), in plain
        PyTorch, for prefill and decode alike. The only difference is where cached K/V live:
        vLLM's paged cache, read through the block table, instead of HF's StaticCache.

          QKV -> weight-less Q/K RMSNorm (Q gamma = 3.87/sqrt(Dh), the whole softmax scale)
          -> RoPE (rotate_half; NoPE layers skip it) -> softmax(QK^T + mask) V over
          [cached ; new] -> * sigmoid(gate_proj(x)) -> o_proj.  New K/V written last.
        Mask: causal by absolute position; local layers keep the last `sliding_window`
        positions (HF: p - window < j <= p).
        """
        slot_mapping = md["slot_mapping"]
        block_table = md["block_table_tensor"]  # [B, nblk], -1 = unused
        bs = md["block_size"]
        B, nblk = block_table.shape
        tokens = hidden_states.shape[0]
        S = tokens // B
        assert tokens == B * S, (
            f"{tokens} tokens do not split evenly over {B} sequences"
        )
        hidden_states = hidden_states.to(self.dtype)
        nq, nkh, D = (
            self.num_attention_heads_per_rank,
            self.num_key_value_heads_per_rank,
            self.head_dim,
        )
        q_gamma, k_gamma = self._qk_gammas(hidden_states.device)

        qkv = hidden_states @ self.qkv_proj_weight
        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
        q = (_rms(q.view(tokens, nq, D), self.eps) * q_gamma.float().view(1, 1, -1)).to(
            self.dtype
        )
        k = (
            _rms(k.view(tokens, nkh, D), self.eps) * k_gamma.float().view(1, 1, -1)
        ).to(self.dtype)
        v = v.view(tokens, nkh, D)
        if self.use_rope:
            cos, sin = position_embeddings  # [T, Dh]
            q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)

        # cached K/V per sequence: [B, nkh, nblk*bs, D]
        safe_table = torch.clamp(block_table, min=0).to(torch.long)
        kc = (
            self.k_cache[safe_table]
            .permute(0, 2, 1, 3, 4)
            .reshape(B, nkh, nblk * bs, D)
        )
        vc = (
            self.v_cache[safe_table]
            .permute(0, 2, 1, 3, 4)
            .reshape(B, nkh, nblk * bs, D)
        )

        # absolute position of each cached slot; SWA tables are trimmed to the window and
        # start at swa_kv_pos_offset (= first kept block * block_size)
        slot_pos = torch.arange(nblk * bs, device=hidden_states.device).view(
            1, 1, -1
        )  # [1, 1, C]
        if self.sliding_window is not None and md.get("swa_kv_pos_offset") is not None:
            slot_pos = slot_pos + md["swa_kv_pos_offset"].view(B, 1, 1).to(
                slot_pos.dtype
            )
        pos = positions.view(B, S, 1).to(slot_pos.dtype)  # [B, S, 1]
        valid = (block_table >= 0).repeat_interleave(bs, dim=1).view(B, 1, -1)
        # Only tokens computed in EARLIER steps are in the cache: positions before this chunk's
        # first token. The chunk's own K/V are written after this read, so its slots still hold
        # stale data; they come from the new k/v below. (Decode: chunk = 1 token, so this is
        # every earlier position.)
        chunk_start = positions.view(B, S)[:, :1].to(slot_pos.dtype).view(B, 1, 1)
        prior_ok = valid & (slot_pos < chunk_start) & (slot_pos < pos)
        new_pos = positions.view(B, 1, S).to(
            slot_pos.dtype
        )  # positions of the new tokens
        # Causal among new tokens by index, not by position: the runner pads a chunk at the
        # end with copies of the last real position, so a position test would let the last
        # real token attend to every pad token.
        idx = torch.arange(S, device=hidden_states.device)
        new_ok = (idx.view(1, 1, S) <= idx.view(1, S, 1)).expand(B, S, S)
        if self.sliding_window is not None:
            prior_ok = prior_ok & (slot_pos > pos - self.sliding_window)
            new_ok = new_ok & (new_pos > pos - self.sliding_window)
        allowed = torch.cat([prior_ok, new_ok], dim=-1)  # [B, S, C+S]

        qb = q.view(B, S, nq, D).permute(0, 2, 1, 3)  # [B, nq, S, D]
        kb = torch.cat(
            [kc, k.view(B, S, nkh, D).permute(0, 2, 1, 3).to(kc.dtype)], dim=2
        )
        vb = torch.cat(
            [vc, v.view(B, S, nkh, D).permute(0, 2, 1, 3).to(vc.dtype)], dim=2
        )
        kb = kb.repeat_interleave(nq // nkh, dim=1)  # HF repeat_kv
        vb = vb.repeat_interleave(nq // nkh, dim=1)
        scores = torch.matmul(qb.float(), kb.float().transpose(-1, -2))
        scores = scores.masked_fill(~allowed.view(B, 1, S, -1), float("-inf"))
        attn = torch.matmul(torch.softmax(scores, dim=-1), vb.float()).to(
            self.dtype
        )  # [B, nq, S, D]

        gate = (
            torch.sigmoid(hidden_states @ self.gate_proj_weight)
            .view(B, S, nq, D)
            .permute(0, 2, 1, 3)
        )
        attn = (attn * gate.to(attn.dtype)).permute(0, 2, 1, 3).reshape(tokens, nq * D)
        out = attn @ self.o_proj_weight

        self._write_kv(
            k.transpose(0, 1), v.transpose(0, 1), slot_mapping, bs
        )  # [nkh, T, D]
        return out

    def _write_kv(self, k, v, slot_mapping, block_size):
        nkh = self.num_key_value_heads_per_rank
        block_idx, pos_idx = paged_slot_indices(
            slot_mapping, block_size, self.k_cache.shape[0]
        )
        heads = torch.arange(nkh, dtype=torch.long, device=k.device).repeat_interleave(
            slot_mapping.shape[0]
        )
        index = (block_idx.repeat(nkh), heads, pos_idx.repeat(nkh))
        self.k_cache.index_put_(
            index, k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
        )
        self.v_cache.index_put_(
            index, v.reshape(-1, self.head_dim).to(self.v_cache.dtype)
        )


# ── MLP ──────────────────────────────────────────────────────────────────────


class MuseGlimmerMLP(nn.Module):
    """SwiGLU (silu), no bias; intermediate sharded across TP."""

    def __init__(self, config: MuseGlimmerTextConfig):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        inter = config.intermediate_size // self.world_size
        dt = config.torch_dtype
        self.gate_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, inter, dtype=dt)
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, inter, dtype=dt)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(inter, config.hidden_size, dtype=dt)
        )
        gate_up = sharding_weight_loader(
            shard_dim=1,
            shard_size=inter,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        set_weight_loader(self.gate_proj_weight, gate_up)
        set_weight_loader(self.up_proj_weight, gate_up)
        set_weight_loader(
            self.down_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=inter,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    def forward(self, x, is_prefill: bool):
        if is_prefill and self.world_size > 1:
            x = self.tp_group.all_gather(x, dim=0)
        # HF MuseGlimmerTextMLP: down(silu(gate(x)) * up(x))
        out = (
            torch.nn.functional.silu(x @ self.gate_proj_weight)
            * (x @ self.up_proj_weight)
        ) @ self.down_proj_weight
        if self.world_size > 1:
            if is_prefill:
                out = self.tp_group.reduce_scatter(out, dim=0)
            else:
                self.tp_group.all_reduce(out)
        return out


# ── decoder layer + backbone ─────────────────────────────────────────────────


class MuseGlimmerDecoderLayer(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig, layer_idx: int):
        super().__init__()
        h, dt = config.hidden_size, config.torch_dtype
        self.input_layernorm = CenteredRMSNorm(h, config.rms_norm_eps, dt)
        self.post_attention_layernorm = CenteredRMSNorm(h, config.post_norm_eps, dt)
        self.pre_feedforward_layernorm = CenteredRMSNorm(h, config.rms_norm_eps, dt)
        self.post_feedforward_layernorm = CenteredRMSNorm(h, config.post_norm_eps, dt)
        self.self_attn = MuseGlimmerAttention(config, layer_idx)
        self.mlp = MuseGlimmerMLP(config)

    def forward(
        self, hidden_states, positions, position_embeddings, attn_metadata, is_prefill
    ):
        residual = hidden_states
        x = self.self_attn(
            self.input_layernorm(hidden_states),
            positions,
            position_embeddings,
            attn_metadata,
        )
        hidden_states = residual + self.post_attention_layernorm(x)
        residual = hidden_states
        x = self.mlp(
            self.pre_feedforward_layernorm(hidden_states), is_prefill=is_prefill
        )
        return residual + self.post_feedforward_layernorm(x)


class MuseGlimmerTextModel(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
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
                pad_shard=True,
            ),
        )
        self.layers = nn.ModuleList(
            [
                MuseGlimmerDecoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, config.torch_dtype)
        self.rotary_emb = RotaryEmbedding(config)

    def forward(
        self,
        input_ids,
        positions,
        attn_metadata,
        rank=None,
        inputs_embeds=None,
        is_token_ids=None,
        encoder_embedding_buffer=None,
        encoder_block_ids=None,
        encoder_positions=None,
    ):
        md = attn_metadata["layers.0.self_attn"]
        is_prefill = md["max_query_len"] > md["decode_token_threshold"]
        sp = is_prefill  # sequence parallel in prefill, like the Qwen3 template
        h = self.embed_tokens(input_ids, scatter_tokens=sp, rank=rank)
        h = NF.merge_prompt_embeds(h, inputs_embeds, is_token_ids)
        h = _rms(h, self.config.rms_norm_eps).to(
            h.dtype
        )  # <-- MUSE: weight-less embedding norm
        # Image embeddings replace the placeholder positions AFTER the embedding norm, as in
        # HF MuseGlimmerModel.forward (normed embed, then masked_scatter of image features).
        # Plain-PyTorch reference of the plugin's merge (gather cache blocks, scatter by position).
        if (
            is_prefill
            and encoder_embedding_buffer is not None
            and encoder_block_ids is not None
            and encoder_positions is not None
        ):
            h, _ = _merge_multimodal_embeddings_torch(
                h,
                encoder_embedding_buffer,
                encoder_block_ids,
                encoder_positions,
                rank if rank is not None else 0,
            )
        pos_emb = self.rotary_emb(positions, device=h.device, dtype=h.dtype)
        for layer in self.layers:
            h = layer(h, positions, pos_emb, attn_metadata, is_prefill)
        h = self.norm(h)
        if sp and self.world_size > 1:
            h = self.tp_group.all_gather(h, dim=0)
        return h


class MuseGlimmerForCausalLM(nn.Module):
    """Muse-Glimmer for vLLM Neuron: text decoder + (optional) vision encoder for images."""

    def __init__(
        self, config: MuseGlimmerTextConfig, hf_config=None, encoder_neuron_config=None
    ):
        super().__init__()
        self.config = config
        self.model = MuseGlimmerTextModel(config)
        self.encoder_neuron_config = encoder_neuron_config
        self.visual = None
        if (
            hf_config is not None
            and getattr(hf_config, "vision_config", None) is not None
        ):
            from .vision_encoder_bf16 import (
                GRIDS,
                MuseGlimmerVisionEncoder,
                precompute_vision_kwargs,
            )

            # Built on CPU (the runner builds the model under the meta device): its
            # non-persistent buffers (vision RoPE inv_freq) must hold real values.
            # Replicated on every rank, like the validated one-core vision run.
            with torch.device("cpu"):
                self.visual = MuseGlimmerVisionEncoder(hf_config).to(config.torch_dtype)
                # host-side grid metadata per fixed shape (patch count -> metadata)
                self._vision_meta = {
                    n: precompute_vision_kwargs(self.visual.tower, torch.tensor([g]))
                    for n, g in GRIDS.items()
                }
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        nc = config.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        self._gather_logits = nc is not None and (
            nc.max_logprobs != 0 or nc.debug_logits_dir is not None
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
                self.on_device_sampling_config, process_group=self.tp_group.device_group
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        inputs_embeds=None,
        is_token_ids=None,
        attn_metadata=None,
        sampling_positions=None,
        sampling_params=None,
        spec_decode_metadata=None,
        logit_mask=None,
        rank=None,
        encoder_embedding_buffer=None,
        encoder_block_ids=None,
        encoder_positions=None,
        **kwargs,
    ):
        if spec_decode_metadata is not None:
            raise NotImplementedError(
                "speculative decoding is not supported for Muse-Glimmer yet"
            )
        positions = positions.to(torch.int32)
        md = attn_metadata["layers.0.self_attn"]
        is_prefill = md["max_query_len"] > md["decode_token_threshold"]
        T = input_ids.shape[0]
        if is_prefill and (T <= self.world_size or T % self.world_size != 0):
            raise ValueError(
                f"Prompt length ({T}) must be > and divisible by world_size ({self.world_size}) for SP."
            )
        h = self.model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
            encoder_embedding_buffer=encoder_embedding_buffer,
            encoder_block_ids=encoder_block_ids,
            encoder_positions=encoder_positions,
        )
        if (
            not is_prefill
            and md["max_query_len"] == 1
            and h.shape[0] == sampling_positions.shape[0]
        ):
            h_logits = h
        else:
            h_logits = torch.index_select(h, dim=0, index=sampling_positions)
        logits = self.lm_head(h_logits)
        # <-- MUSE: cap * tanh(logits * multiplier / cap); elementwise, so vocab shards are fine
        cap = self.config.final_logit_softcapping
        logits = (
            torch.tanh(logits.float() * (self.config.output_multiplier / cap)) * cap
        ).to(logits.dtype)

        if self.on_device_sampling_config is None:
            return logits
        sampled = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )
        gathered = (
            self.tp_group.all_gather(logits, dim=1) if self._gather_logits else None
        )
        return sampled, gathered

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig,
        encoder_neuron_config=None,
    ):
        return cls(
            MuseGlimmerTextConfig.from_configs(hf_config, neuron_config),
            hf_config,
            encoder_neuron_config,
        )

    # ── vision: runner hooks (vllm-neuron VLM contract) ──

    def get_encoder(self):
        return self.visual

    def _vision_num_blocks(self, bucket, enc):
        # the runner's write_block_ids length for a bucket (neuron_worker encoder warmup)
        return (
            math.ceil(
                math.ceil(bucket / enc.encoder_attention_block_size) / enc.dp_size
            )
            * enc.dp_size
        )

    def build_encoder_synthetic_inputs(self, bucket, encoder_neuron_config, device):
        """Warmup inputs for one vision graph: bucket = patch count of a fixed shape
        (1024 image, 2080 video); that grid's real metadata + zero pixels."""
        if bucket not in self._vision_meta:
            raise ValueError(
                f"encoder bucket {bucket} is not a fixed vision shape {sorted(self._vision_meta)}"
            )
        patch_dim = self.visual.tower.patch_embedder.patch_embedding.in_features
        kw = {k: v.to(device) for k, v in self._vision_meta[bucket].items()}
        kw["pixel_values"] = torch.zeros(
            bucket, patch_dim, dtype=torch.float32, device=device
        )
        return kw

    def embed_multimodal(
        self,
        encoder_cache=None,
        mm_hashes=None,
        image_pixel_values=None,
        image_feature_sizes=None,
        video_pixel_values=None,
        video_feature_sizes=None,
        video_timestamps=None,
        **kwargs,
    ):
        """Encode images or videos into the on-device encoder cache, one item per vision-graph call.
        The runner groups by modality, so one of image_/video_pixel_values is set per call."""
        pix = (
            image_pixel_values if image_pixel_values is not None else video_pixel_values
        )
        if pix is None:
            raise ValueError(
                "embed_multimodal: no image_pixel_values / video_pixel_values"
            )
        items = (
            list(pix) if not isinstance(pix, torch.Tensor) or pix.dim() == 3 else [pix]
        )
        assert len(items) == len(mm_hashes), (
            f"{len(items)} items vs {len(mm_hashes)} hashes"
        )
        enc = self.encoder_neuron_config
        device = next(self.visual.parameters()).device
        for mm_hash, pv in zip(mm_hashes, items):
            pv = pv.reshape(-1, pv.shape[-1])
            n_patches = pv.shape[0]
            if n_patches not in self._vision_meta:
                raise ValueError(
                    f"{n_patches} patches is not a fixed vision shape {sorted(self._vision_meta)}"
                )
            n_embed = n_patches // self.visual.merge_factor
            num_blocks = self._vision_num_blocks(n_patches, enc)
            meta = {k: v.to(device) for k, v in self._vision_meta[n_patches].items()}
            blocks = encoder_cache.allocate(
                mm_hash,
                encoder_cache.dense_tokens_per_block(n_embed, encoder_cache.block_size),
            )
            write = torch.tensor(
                blocks + [encoder_cache.scratch_block_id] * (num_blocks - len(blocks)),
                dtype=torch.int64,
            )
            self.visual(
                pixel_values=pv.to(device),
                **meta,
                encoder_cache_buffer=encoder_cache.buffer,
                write_block_ids=write.to(device),
            )

    # ── KV cache ──

    def get_kv_spec(self):
        return KVSpec(
            layers=[
                LayerSpec(
                    name=f"layers.{i}.self_attn",
                    num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                    head_size=layer.self_attn.head_dim,
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=layer.self_attn.sliding_window,
                    chunk_size=None,
                )
                for i, layer in enumerate(self.model.layers)
            ]
        )

    def bind_kv_cache(self, kv_caches):
        for i, layer in enumerate(self.model.layers):
            name = f"layers.{i}.self_attn"
            if name not in kv_caches:
                raise KeyError(f"KV cache for layer {name} not initialized")
            layer.self_attn.k_cache, layer.self_attn.v_cache = (
                kv_caches[name][0],
                kv_caches[name][1],
            )

    # ── weights ──

    def load_weights(self, checkpoint_path, device, cache_dir):
        src = (
            "model.language_model"  # HF MuseGlimmerForConditionalGeneration text prefix
        )
        m = {
            "model.embed_tokens.weight": f"{src}.embed_tokens.weight",
            "model.norm.weight": f"{src}.norm.weight",
            "lm_head.weight": "lm_head.weight",
        }
        for i in range(len(self.model.layers)):
            d, s = f"model.layers.{i}", f"{src}.layers.{i}"
            m[f"{d}.self_attn.qkv_proj_weight"] = [
                f"{s}.self_attn.q_proj.weight",
                f"{s}.self_attn.k_proj.weight",
                f"{s}.self_attn.v_proj.weight",
            ]
            m[f"{d}.self_attn.gate_proj_weight"] = f"{s}.self_attn.gate_proj.weight"
            m[f"{d}.self_attn.o_proj_weight"] = f"{s}.self_attn.o_proj.weight"
            for n in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ):
                m[f"{d}.{n}.weight"] = f"{s}.{n}.weight"
            for n in ("gate_proj", "up_proj", "down_proj"):
                m[f"{d}.mlp.{n}_weight"] = f"{s}.mlp.{n}.weight"
        if (
            self.visual is not None
        ):  # replicated on every rank (no weight loader -> loaded as is)
            for name, _ in self.visual.named_parameters():
                if name.startswith("tower."):
                    m[f"visual.{name}"] = "model.vision_tower." + name[len("tower.") :]
                elif name.startswith("adapter."):
                    m[f"visual.{name}"] = (
                        "model.vision_adapter." + name[len("adapter.") :]
                    )
                elif name == "projection.weight":
                    m["visual.projection.weight"] = "model.vision_projection.weight"
        state = (
            SafetensorsCheckpoint(checkpoint_path, cache_dir)
            .load_sharded_pipelined(self.rank, self.world_size, self, m, device)
            .state_dict
        )
        for k, t in state.items():
            if t.dtype != self.config.torch_dtype:
                state[k] = t.to(self.config.torch_dtype)
        missing = set(dict(self.named_parameters())) - set(state)
        if missing:
            raise RuntimeError(
                f"checkpoint did not provide {len(missing)} parameters, e.g. {sorted(missing)[:4]}"
            )
        self.load_state_dict(state, strict=False, assign=True)
        if self.visual is not None:
            self.visual.to(
                device
            )  # non-persistent buffers (vision RoPE) follow the weights
