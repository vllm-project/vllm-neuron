# SPDX-License-Identifier: Apache-2.0
"""LiquidAI d1-3B language model (LFM2 hybrid backbone) for vLLM on Neuron.

d1-3B is a decision model: it reads a state and named questions in one forward pass,
and the answer is a softmax over each question's option tokens at the answer slot.
On vLLM that is one prefill with ``max_tokens=1`` and raw logits; the prompt
rendering and the option readout run on the host with the model's own code (see
the README).

Architecture (``text_config`` of the ``lfm2_vl`` checkpoint):
  - 30 layers, ``layer_types`` is ``conv`` (gated short conv, kernel 3) or
    ``full_attention`` (GQA 32 q / 8 kv heads, head_dim 64, per-head q/k RMSNorm
    before RoPE, rope_theta 1e6).
  - Every layer: ``x += op(operator_norm(x)); x += mlp(ffn_norm(x))`` with a dense
    SwiGLU MLP (w1 gate, w3 up, w2 down). Final norm ``embedding_norm``; the LM head
    is tied to ``embed_tokens``.

Parallelism: TP over all ranks. Prefill is sequence-parallel between blocks
(all-gather in, reduce-scatter out), like the in-tree models.

Conv state: the conv input ``B*x`` of every token is stored in a paged cache that
is registered as an ordinary attention layer (1 kv head, head_size
``(hidden/TP)/2``: the first half of the channels in K, the second half in V). The
conv at position p reads positions p-1 and p-2 through the block table. vLLM
therefore manages conv state with the same blocks as attention KV, which keeps
chunked prefill and prefix caching correct for conv layers. A multi-question
request becomes several prompts that share the state as a prefix.

Derived from the out-of-tree LFM2 port for vllm-neuron 0.29 (LFM2.5-24B-A2B run,
``lfm2_neuron/model.py``): the MoE is replaced by d1-3B's dense MLP, the checkpoint
keys carry the ``model.language_model.`` prefix, and the decode helpers are local.
"""

from __future__ import annotations

import logging
import math
import os

import torch
import torch.nn.functional as F
from nkilib.core.utils.common_types import NormType
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.functional.merge_multimodal_embeddings import (
    merge_multimodal_embeddings,
)
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec, paged_slot_indices
from vllm_neuron.model.neuron_config import EncoderNeuronConfig, NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import D1TextConfig
from .decode_kv import build_per_head_block_table, scatter_new_kv
from .vision import (
    PATCHES_PER_BLOCK,
    D1VisionConfig,
    D1VisionTower,
    merged_tokens,
    pack_tiles,
    vision_layer_mappings,
    vision_mappings,
)

logger = logging.getLogger(__name__)

# Checkpoint key prefix of the language model inside the lfm2_vl checkpoint.
CKPT_LM = "model.language_model"


def _is_decode(attn_metadata) -> bool:
    meta = next(iter(attn_metadata.values()))
    return meta["max_query_len"] <= meta["decode_token_threshold"]


# =============================================================================
# Norm / RoPE
# =============================================================================
class D1RMSNorm(nn.Module):
    """HF Lfm2RMSNorm: fp32 variance, cast to the input dtype, then * weight."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hs = hidden_states.to(torch.float32)
        variance = hs.pow(2).mean(-1, keepdim=True)
        hs = hs * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hs.to(input_dtype)


class D1RotaryEmbedding(nn.Module):
    """Default RoPE (rotate_half). inv_freq is built per call (meta-device init)."""

    def __init__(self, config: D1TextConfig):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim

    def forward(self, position_ids, device, dtype):
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
                / self.head_dim
            )
        )
        freqs = (inv_freq[:, None].float() @ position_ids[None, :].float()).transpose(
            0, 1
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


# =============================================================================
# Attention (QK-norm before RoPE, GQA, no bias)
# =============================================================================
class D1Attention(nn.Module):
    def __init__(self, config: D1TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_name = f"layers.{layer_idx}.self_attn"
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

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

        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, q_size + 2 * kv_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(
                self.num_attention_heads * self.head_dim // self.world_size,
                self.hidden_size,
                dtype=self.dtype,
            )
        )
        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        self.q_layernorm = D1RMSNorm(self.head_dim, config.norm_eps, self.dtype)
        self.k_layernorm = D1RMSNorm(self.head_dim, config.norm_eps, self.dtype)

        self.k_cache = None
        self.v_cache = None
        # D1_ATTN_PAD128=1: store K/V and run decode attention at head_dim 128 so
        # decode can use the NKI attention_block_tkg kernel (it rejects head_dim
        # 64). Each 64-dim head is laid out as [x[0:32], 0*32, x[32:64], 0*32], so
        # rotate_half over 128 dims pairs the same dims as over 64; the zero pads
        # leave q.k and the output unchanged (W_out pad rows are zero). QK-norm over
        # 128 dims stays exact with gamma/sqrt(2) and eps/2. d1-3B never decodes in
        # normal use (max_tokens=1), so this is off by default.
        self.pad128 = (
            os.environ.get("D1_ATTN_PAD128", "0") == "1" and self.head_dim == 64
        )
        self.cache_head_dim = 128 if self.pad128 else self.head_dim
        self.use_decode_kernel = self.cache_head_dim % 128 == 0
        for name in ("qkv_pad_weight", "o_pad_weight", "q_gamma_pad", "k_gamma_pad"):
            self.register_buffer(name, None, persistent=False)

        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=q_size,
                kv_size=kv_size,
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
                shard_size=self.num_attention_heads * self.head_dim // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    @staticmethod
    def _pad_heads(x):
        """[..., 64] -> [..., 128] in the rotate-half-compatible padded layout."""
        z = torch.zeros_like(x[..., :32])
        return torch.cat([x[..., :32], z, x[..., 32:], z], dim=-1)

    @torch.no_grad()
    def build_padded_weights(self):
        """Derive the pad-128 decode weights from the loaded 64-dim weights."""
        if not self.pad128:
            return
        nh, nkh, hd, H = (
            self.num_attention_heads_per_rank,
            self.num_key_value_heads_per_rank,
            self.head_dim,
            self.hidden_size,
        )
        dev = self.qkv_proj_weight.device
        # Build on host (eager Neuron tensors reject these non-contiguous views).
        host = torch.device("meta") if dev.type == "meta" else torch.device("cpu")
        W = self.qkv_proj_weight.data.to(host).contiguous().view(H, nh + 2 * nkh, hd)
        qkv = self._pad_heads(W).reshape(H, -1).contiguous()
        Wo = (
            self.o_proj_weight.data.to(host)
            .contiguous()
            .view(nh, hd, H)
            .transpose(1, 2)
            .contiguous()
        )
        o = self._pad_heads(Wo).transpose(1, 2).contiguous().reshape(nh * 128, H)
        s = 1.0 / math.sqrt(2.0)
        gq = self._pad_heads(self.q_layernorm.weight.data.to(host) * s).view(1, 128)
        gk = self._pad_heads(self.k_layernorm.weight.data.to(host) * s).view(1, 128)
        self.qkv_pad_weight = qkv.to(dev)
        self.o_pad_weight = o.contiguous().to(dev)
        self.q_gamma_pad = gq.contiguous().to(dev)
        self.k_gamma_pad = gk.contiguous().to(dev)

    def forward(self, hidden_states, positions, position_embeddings, attn_metadata):
        meta = attn_metadata[self.layer_name]
        if meta["max_query_len"] <= meta["decode_token_threshold"]:
            return self.forward_decode(
                hidden_states, positions, position_embeddings, meta
            )
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(hidden_states, positions, position_embeddings, meta)

    def forward_prefill(self, hidden_states, positions, position_embeddings, meta):
        hidden_states = hidden_states.to(self.dtype)
        tokens, _ = hidden_states.shape
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
            qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_eps=self.q_layernorm.variance_epsilon,
            qk_norm_pre_rope_q_gamma=self.q_layernorm.weight.view(1, -1),
            qk_norm_pre_rope_k_gamma=self.k_layernorm.weight.view(1, -1),
        ).squeeze(0)
        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
        nh, nkh, hd = (
            self.num_attention_heads_per_rank,
            self.num_key_value_heads_per_rank,
            self.head_dim,
        )
        q = q.view(tokens, nh, hd).transpose(0, 1)
        k = k.view(tokens, nkh, hd).transpose(0, 1)
        v = v.view(tokens, nkh, hd).transpose(0, 1)
        if self.pad128:
            q, k, v = self._pad_heads(q), self._pad_heads(k), self._pad_heads(v)
            hd = 128

        slot_mapping = meta["slot_mapping"]
        block_size = meta["block_size"]
        block_indices, position_indices = paged_slot_indices(
            slot_mapping, block_size, self.k_cache.shape[0]
        )
        k_flat = k.reshape(-1, hd).to(self.k_cache.dtype)
        v_flat = v.reshape(-1, hd).to(self.v_cache.dtype)
        head_idx = torch.arange(
            nkh, dtype=torch.long, device=hidden_states.device
        ).repeat_interleave(slot_mapping.shape[0])
        idx = (block_indices.repeat(nkh), head_idx, position_indices.repeat(nkh))
        self.k_cache.index_put_(idx, k_flat)
        self.v_cache.index_put_(idx, v_flat)

        kv_segment_size = meta.get("kv_segment_size")
        if kv_segment_size:
            # Chunked prefill / prefix-cache hit: attend over the cached prior KV.
            attn_output = NF.segmented_attention(
                q,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=meta["block_table_tensor"],
                prior_tokens=meta.get("cached_seq_len"),
                block_size=block_size,
                kv_segment_size=kv_segment_size,
                scale=self.scaling,
                tp_q=True,
                tp_out=True,
            )
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
            )

        attn_output = NF.o_proj(
            attn_output.unsqueeze(0),
            self.o_pad_weight if self.pad128 else self.o_proj_weight,
            None,
        ).squeeze(0)
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)
        return attn_output.contiguous()

    def forward_decode(self, hidden_states, positions, position_embeddings, meta):
        slot_mapping = meta["slot_mapping"]
        block_size = meta["block_size"]
        block_table = meta["block_table_tensor"]
        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S = tokens // B
        assert tokens == B * S
        S_ctx = meta["max_blocks_per_seq"] * block_size
        nkh = self.num_key_value_heads_per_rank
        hidden_states = hidden_states.to(self.dtype)
        X = hidden_states.view(B, S, hidden)

        cos, sin = position_embeddings
        half_d = self.head_dim // 2
        cos_k = cos[:, :half_d].view(B, S, half_d).permute(2, 0, 1).contiguous()
        sin_k = sin[:, :half_d].view(B, S, half_d).permute(2, 0, 1).contiguous()
        cos_k, sin_k = cos_k.to(self.dtype), sin_k.to(self.dtype)
        if self.pad128:
            cos_k = torch.cat([cos_k, cos_k], dim=0)
            sin_k = torch.cat([sin_k, sin_k], dim=0)
            W_qkv, W_out = self.qkv_pad_weight, self.o_pad_weight
            g_q, g_k = self.q_gamma_pad, self.k_gamma_pad
            qk_eps = self.q_layernorm.variance_epsilon / 2.0
        else:
            W_qkv, W_out = self.qkv_proj_weight, self.o_proj_weight
            g_q = self.q_layernorm.weight.view(1, -1)
            g_k = self.k_layernorm.weight.view(1, -1)
            qk_eps = self.q_layernorm.variance_epsilon

        attention_mask = NF.gen_attention_decode_mask(
            pos_ids=positions.view(1, B * S).to(torch.float32),
            bs=B,
            q_head=self.num_attention_heads_per_rank,
            s_active=S,
            s_prior=S_ctx,
            start_pos=None,
            block_len=block_size,
        )

        if nkh > 1:
            # head_dim 64 fails the attention_block_tkg kernel contract; a 2D table
            # with a 4D cache routes to the torch fallback inside NF.attention_decode.
            if self.use_decode_kernel:
                active_blocks_table = build_per_head_block_table(block_table, nkh)
            else:
                active_blocks_table = block_table
            active_blocks_table = active_blocks_table.to(torch.int32)
            k_cache, v_cache = self.k_cache, self.v_cache
        else:
            active_blocks_table = block_table.to(torch.int32)
            k_cache = self.k_cache.squeeze(1)
            v_cache = self.v_cache.squeeze(1)

        output, K_new, V_new = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            rmsnorm_X_enabled=False,
            W_qkv=W_qkv,
            bias_qkv=None,
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=qk_eps,
            rmsnorm_QK_pre_rope_W_Q=g_q,
            rmsnorm_QK_pre_rope_W_K=g_k,
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_k,
            sin=sin_k,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=active_blocks_table,
            K_cache=k_cache,
            V_cache=v_cache,
            attention_mask=attention_mask,
            softmax_scale=self.scaling,
            sink=None,
            update_cache=False,
            W_out=W_out,
            bias_out=None,
            transposed_out=False,
            out_in_sb=False,
            k_scale=None,
            v_scale=None,
            attention_dp=1,
            attention_dp_group=None,
            attention_dp_rank=0,
            kv_needs_a2a=False,
        )
        if K_new.dim() == 3:
            # torch fallback / kv_heads == 1 kernel: [d_head, B*kv_heads, S]
            K_new = K_new.reshape(self.cache_head_dim, B, nkh, S)
        scatter_new_kv(
            self.k_cache,
            self.v_cache,
            K_new,
            V_new,
            slot_mapping,
            block_size,
            nkh,
            self.cache_head_dim,
        )
        if self.world_size > 1:
            self.tp_group.all_reduce(output)
        return output


# =============================================================================
# Gated short convolution (LFM2 "conv" layer)
# =============================================================================
class D1ShortConv(nn.Module):
    """y = out_proj(C * causal_depthwise_conv(B * x)), [B, C, x] = in_proj(h).

    TP: channels sharded (column-parallel in_proj, per-channel conv, row-parallel
    out_proj). Conv state = per-token ``B*x`` in this layer's paged cache.
    """

    def __init__(self, config: D1TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_name = f"layers.{layer_idx}.conv"
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.L = config.conv_L_cache
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        if config.hidden_size % (2 * self.world_size):
            raise ValueError(
                f"hidden_size {config.hidden_size} must split into 2 * TP={self.world_size}"
            )
        self.ch = config.hidden_size // self.world_size
        self.half = self.ch // 2

        self.in_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, 3 * self.ch, dtype=self.dtype)
        )
        self.conv_weight = nn.Parameter(torch.empty(self.ch, self.L, dtype=self.dtype))
        self.out_proj_weight = nn.Parameter(
            torch.empty(self.ch, self.hidden_size, dtype=self.dtype)
        )
        self.k_cache = None
        self.v_cache = None

        ch, H, ws = self.ch, self.hidden_size, self.world_size

        def in_proj_transform(slices, rank):
            # Checkpoint rows are [B; C; x], each H wide; take this rank's channels.
            r = rank % ws
            s = slices[0]
            parts = [s[j * H + r * ch : j * H + (r + 1) * ch, :] for j in range(3)]
            return torch.cat(parts, dim=0).T  # [H, 3*ch]

        def conv_transform(slices, rank):
            r = rank % ws
            return slices[0][r * ch : (r + 1) * ch, :, :].reshape(ch, self.L)

        set_weight_loader(
            self.in_proj_weight, SafetensorsWeightLoader(in_proj_transform)
        )
        set_weight_loader(self.conv_weight, SafetensorsWeightLoader(conv_transform))
        set_weight_loader(
            self.out_proj_weight,
            sharding_weight_loader(
                shard_dim=0, shard_size=ch, num_shards=ws, is_storage_transposed=True
            ),
        )

    def _gather_prev(self, pos, block_table, block_size, k):
        """``B*x`` of position ``pos - k`` for each row -> [N, ch] (zero before 0)."""
        idx = pos.to(torch.int64) - k
        valid = idx >= 0
        idx = torch.clamp(idx, min=0)
        nblk = block_table.shape[1]
        col = torch.clamp(
            torch.div(idx, block_size, rounding_mode="floor"), max=nblk - 1
        )
        blk = torch.gather(block_table.to(torch.int64), 1, col.unsqueeze(1)).squeeze(1)
        nb = self.k_cache.shape[0]
        blk = torch.where((blk < 0) | (blk >= nb), torch.zeros_like(blk), blk)
        off = idx - torch.div(idx, block_size, rounding_mode="floor") * block_size
        head = torch.zeros_like(blk)
        prev = torch.cat(
            [self.k_cache[blk, head, off], self.v_cache[blk, head, off]], dim=-1
        )
        return prev * valid.unsqueeze(1).to(prev.dtype)

    def _write(self, bx, slot_mapping, block_size):
        bi, pi = paged_slot_indices(slot_mapping, block_size, self.k_cache.shape[0])
        hi = torch.zeros_like(bi)
        bx = bx.to(self.k_cache.dtype)
        self.k_cache.index_put_((bi, hi, pi), bx[:, : self.half])
        self.v_cache.index_put_((bi, hi, pi), bx[:, self.half :])

    def forward(self, hidden_states, positions, attn_metadata):
        meta = attn_metadata[self.layer_name]
        is_decode = meta["max_query_len"] <= meta["decode_token_threshold"]
        block_size = meta["block_size"]
        block_table = meta["block_table_tensor"]
        if not is_decode and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        hidden_states = hidden_states.to(self.dtype)
        T = hidden_states.shape[0]

        bcx = torch.matmul(hidden_states, self.in_proj_weight)  # [T, 3ch]
        b, c, x = torch.split(bcx, self.ch, dim=-1)
        bx = b * x  # [T, ch]
        w = self.conv_weight.to(torch.float32)  # [ch, 3]: taps for t-2, t-1, t

        if is_decode:
            # One token per sequence (B == T).
            p1 = self._gather_prev(positions, block_table, block_size, 1)
            p2 = self._gather_prev(positions, block_table, block_size, 2)
            conv = p2.float() * w[:, 0] + p1.float() * w[:, 1] + bx.float() * w[:, 2]
        else:
            # Prefill: one sequence per step (NeuronScheduler max_prefills_per_batch=1).
            # The 2 tokens before this chunk come from the cache (chunked prefill or a
            # prefix-cache hit); they are zero at position 0.
            start = positions[:1]
            bt0 = block_table[:1]
            p1 = self._gather_prev(start, bt0, block_size, 1)
            p2 = self._gather_prev(start, bt0, block_size, 2)
            ext = torch.cat([p2, p1, bx], dim=0).float()  # [T+2, ch]
            conv = (
                ext[0:T] * w[:, 0] + ext[1 : T + 1] * w[:, 1] + ext[2 : T + 2] * w[:, 2]
            )
        self._write(bx, meta["slot_mapping"], block_size)

        y = c * conv.to(self.dtype)
        out = torch.matmul(y, self.out_proj_weight)  # [T, H] partial
        if self.world_size > 1:
            if is_decode:
                self.tp_group.all_reduce(out)
            else:
                out = self.tp_group.reduce_scatter(out, dim=0)
        return out


# =============================================================================
# Dense SwiGLU MLP
# =============================================================================
class D1MLP(nn.Module):
    def __init__(self, config: D1TextConfig):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        if config.intermediate_size % self.world_size:
            raise ValueError(
                f"intermediate_size {config.intermediate_size} not divisible by "
                f"TP={self.world_size}"
            )
        I_r = config.intermediate_size // self.world_size
        dt = config.torch_dtype
        H = config.hidden_size
        self.gate_proj_weight = nn.Parameter(torch.empty(H, I_r, dtype=dt))
        self.up_proj_weight = nn.Parameter(torch.empty(H, I_r, dtype=dt))
        self.down_proj_weight = nn.Parameter(torch.empty(I_r, H, dtype=dt))
        gu = sharding_weight_loader(
            shard_dim=1,
            shard_size=I_r,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        dn = sharding_weight_loader(
            shard_dim=0,
            shard_size=I_r,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        set_weight_loader(self.gate_proj_weight, gu)
        set_weight_loader(self.up_proj_weight, gu)
        set_weight_loader(self.down_proj_weight, dn)

    def forward(self, hidden_states, is_decode):
        if not is_decode and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        out = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        )
        if self.world_size > 1:
            if is_decode:
                self.tp_group.all_reduce(out)
            else:
                out = self.tp_group.reduce_scatter(out, dim=0)
        return out


# =============================================================================
# Decoder / backbone / LM
# =============================================================================
class D1DecoderLayer(nn.Module):
    def __init__(self, config: D1TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"
        if self.is_attention_layer:
            self.self_attn = D1Attention(config, layer_idx)
        else:
            self.conv = D1ShortConv(config, layer_idx)
        self.feed_forward = D1MLP(config)
        dt = config.torch_dtype
        self.operator_norm = D1RMSNorm(config.hidden_size, config.norm_eps, dt)
        self.ffn_norm = D1RMSNorm(config.hidden_size, config.norm_eps, dt)

    @property
    def cache_module(self) -> nn.Module:
        return self.self_attn if self.is_attention_layer else self.conv

    def forward(
        self, hidden_states, positions, position_embeddings, attn_metadata, is_decode
    ):
        h = self.operator_norm(hidden_states)
        if self.is_attention_layer:
            h = self.self_attn(h, positions, position_embeddings, attn_metadata)
        else:
            h = self.conv(h, positions, attn_metadata)
        hidden_states = hidden_states + h
        h = self.feed_forward(self.ffn_norm(hidden_states), is_decode)
        return hidden_states + h


class D1Model(nn.Module):
    def __init__(self, config: D1TextConfig):
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
            [D1DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.embedding_norm = D1RMSNorm(
            config.hidden_size, config.norm_eps, config.torch_dtype
        )
        self.rotary_emb = D1RotaryEmbedding(config)

    def forward(
        self,
        input_ids,
        positions,
        attn_metadata,
        rank=None,
        encoder_embedding_buffer=None,
        encoder_block_ids=None,
        encoder_positions=None,
    ):
        is_decode = _is_decode(attn_metadata)
        sp = not is_decode
        hidden_states = self.embed_tokens(input_ids, scatter_tokens=sp, rank=rank)
        if sp and encoder_embedding_buffer is not None:
            # Scatter the cached image embeddings into the image-token positions
            # (SP-sharded hidden states; the helper remaps global positions).
            hidden_states, _ = merge_multimodal_embeddings(
                hidden_states,
                encoder_embedding_buffer,
                encoder_block_ids,
                encoder_positions,
                rank=self.tp_group.rank_in_group_tensor,
            )
        position_embeddings = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, positions, position_embeddings, attn_metadata, is_decode
            )
        hidden_states = self.embedding_norm(hidden_states)
        if sp and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return hidden_states


class D1NeuronForCausalLM(nn.Module):
    """The model the Neuron runner builds, compiles and serves."""

    def __init__(
        self,
        config: D1TextConfig,
        vision_config: D1VisionConfig | None = None,
        encoder_neuron_config: EncoderNeuronConfig | None = None,
    ):
        super().__init__()
        self.config = config
        self.model = D1Model(config)
        # Vision tower only when the runner configures the encoder (images).
        self.vision_config = (
            vision_config if encoder_neuron_config is not None else None
        )
        self.encoder_neuron_config = encoder_neuron_config
        self.visual = D1VisionTower(vision_config) if self.vision_config else None
        self._pos_table_cpu = None
        if self.visual is not None:
            # The runner always builds an encoder config for this (multimodal)
            # class, with defaults when none is given; those do not fit d1-3B.
            hint = (
                'pass additional_config["encoder_neuron_config"] = '
                '{"num_encoder_input_buckets": [4096, 12288], '
                f'"encoder_attention_block_size": {PATCHES_PER_BLOCK}}} '
                "(needed for text-only serving too; see the d1-3B README)"
            )
            bs = encoder_neuron_config.encoder_attention_block_size
            if bs != PATCHES_PER_BLOCK:
                raise ValueError(
                    f"d1-3B: encoder_attention_block_size must be {PATCHES_PER_BLOCK} "
                    f"(one tile per block), got {bs}: {hint}"
                )
            if not encoder_neuron_config.num_encoder_input_buckets:
                raise ValueError(f"d1-3B: no num_encoder_input_buckets: {hint}")
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
                shard_size=self.lm_head.out_features_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=False,
            ),
        )
        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config, process_group=self.tp_group.device_group
            )
        # D1_FP32_LOGITS=1 (default): the LM head runs in fp32 from an fp32 copy of its
        # weight. d1-3B's answer is a softmax over a few option logits, and bf16 logits
        # (step 0.125 at magnitude 16-32) can tie options that fp32 separates (measured:
        # an exact tie that flipped a top-1 on batch8_3q). The work-d1-3b runtime also
        # computes option logits in fp32. Costs ~262 MB HBM per rank at TP=4.
        self.fp32_logits = os.environ.get("D1_FP32_LOGITS", "1") == "1"
        self.register_buffer("lm_head_weight_fp32", None, persistent=False)

    @torch.no_grad()
    def _build_fp32_head(self):
        if not self.fp32_logits:
            return
        w = self.lm_head.weight.detach()
        if w.device.type == "meta":
            self.lm_head_weight_fp32 = torch.empty(
                w.shape, dtype=torch.float32, device="meta"
            )
            return
        # Cast on the host: eager dtype casts on the Neuron device are not supported.
        self.lm_head_weight_fp32 = w.cpu().to(torch.float32).to(w.device)

    def _logits(self, h):
        if self.lm_head_weight_fp32 is None:
            return self.lm_head(h)
        logits = F.linear(h.to(torch.float32), self.lm_head_weight_fp32)  # vocab shard
        if not self.on_device_sampling_config and self.world_size > 1:
            logits = self.tp_group.all_gather(logits, dim=1)
        return logits

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
        if inputs_embeds is not None:
            raise NotImplementedError(
                "d1-3B contrib: prompt embeddings are not supported"
            )
        positions = positions.to(torch.int32)
        is_decode = _is_decode(attn_metadata)
        T = input_ids.shape[0]
        if not is_decode and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prefill length ({T}) must be a multiple of TP={self.world_size} and "
                "larger than it (sequence-parallel prefill)."
            )

        hidden_states = self.model(
            input_ids,
            positions,
            attn_metadata,
            rank=rank,
            encoder_embedding_buffer=encoder_embedding_buffer,
            encoder_block_ids=encoder_block_ids,
            encoder_positions=encoder_positions,
        )

        if is_decode and hidden_states.shape[0] == sampling_positions.shape[0]:
            h = hidden_states
        else:
            h = torch.index_select(hidden_states, dim=0, index=sampling_positions)
        logits = self._logits(h)

        if self.on_device_sampling_config is None:
            return logits
        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )
        gathered_logits = None
        if self._gather_logits:
            gathered_logits = self.tp_group.all_gather(logits, dim=1)
        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
        encoder_neuron_config: EncoderNeuronConfig | None = None,
    ) -> D1NeuronForCausalLM:
        vision = D1VisionConfig.from_hf(hf_config) if encoder_neuron_config else None
        return cls(
            D1TextConfig.from_configs(hf_config, neuron_config),
            vision_config=vision,
            encoder_neuron_config=encoder_neuron_config,
        )

    # ── Vision encoder (runner interface) ────────────────────────────────
    def _blocks_for_bucket(self, bucket: int) -> int:
        """Encoder blocks of a bucket, padded like the runner's warmup (DP)."""
        enc = self.encoder_neuron_config
        n = math.ceil(bucket / enc.encoder_attention_block_size)
        dp = max(1, enc.dp_size)
        return math.ceil(n / dp) * dp

    def get_encoder(self) -> nn.Module | None:
        return self.visual

    def build_encoder_synthetic_inputs(
        self, bucket: int, encoder_neuron_config: EncoderNeuronConfig, device
    ) -> dict[str, torch.Tensor]:
        """Shape-only inputs of one bucket (the runner adds the cache buffer and
        ``write_block_ids``)."""
        vc = self.vision_config
        B, T = self._blocks_for_bucket(bucket), PATCHES_PER_BLOCK
        bias = torch.zeros(B, 1, 1, T, dtype=torch.float32, device=device)
        return {
            "pixel_values": torch.zeros(
                B, T, vc.patch_dim, dtype=vc.dtype, device=device
            ),
            "pos_embeds": torch.zeros(
                B, T, vc.hidden_size, dtype=vc.dtype, device=device
            ),
            "key_bias": bias,
            "unshuffle_idx": torch.zeros(
                B,
                vc.tokens_per_block * vc.merge_factor,
                dtype=torch.int32,
                device=device,
            ),
        }

    def embed_multimodal(
        self,
        pixel_values: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        num_patches: torch.Tensor | None = None,
        encoder_cache=None,
        mm_hashes: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Encode images into the on-device encoder cache, one tile per block.

        Args:
            pixel_values: [num_tiles, <=1024, 768] patches of every tile of every
                image, in order (an image's tiles, its thumbnail last).
            spatial_shapes: [num_tiles, 2] (h, w) patch grid per tile (CPU).
            num_patches: [num_images] tiles per image (CPU).
            encoder_cache: The runner's ``EncoderCacheBlocks``.
            mm_hashes: One id per image, in order.
        """
        if self.visual is None:
            raise RuntimeError("d1-3B: images need encoder_neuron_config")
        vc = self.vision_config
        # Batched kwargs can arrive as per-item lists; flatten to tile order.
        if isinstance(pixel_values, (list, tuple)):
            pixel_values = torch.cat(
                [p.reshape(-1, *p.shape[-2:]) for p in pixel_values]
            )
        if isinstance(spatial_shapes, (list, tuple)):
            spatial_shapes = torch.cat([s.reshape(-1, 2) for s in spatial_shapes])
        if isinstance(num_patches, (list, tuple)):
            num_patches = torch.tensor([int(n) for n in num_patches])
        shapes = spatial_shapes.cpu()
        tiles_per_image = [int(n) for n in num_patches.cpu().reshape(-1).tolist()]
        if (
            len(tiles_per_image) != len(mm_hashes)
            or sum(tiles_per_image) != shapes.shape[0]
        ):
            raise ValueError(
                f"d1-3B embed_multimodal: {len(mm_hashes)} hashes, tiles {tiles_per_image}, "
                f"{shapes.shape[0]} tile shapes"
            )
        if self._pos_table_cpu is None:
            self._pos_table_cpu = (
                self.visual.position_embedding_weight.detach().cpu().float()
            )

        # 1. One cache block per tile; each holds the tile's merged tokens.
        write_ids: list[int] = []
        t = 0
        for mm_hash, n in zip(mm_hashes, tiles_per_image):
            per_tile = [
                merged_tokens(vc, int(h), int(w)) for h, w in shapes[t : t + n].tolist()
            ]
            write_ids += encoder_cache.allocate(mm_hash, per_tile)
            t += n

        # 2. Smallest compiled bucket with a block per tile.
        num_tiles = shapes.shape[0]
        buckets = sorted(self.encoder_neuron_config.num_encoder_input_buckets)
        fitting = [b for b in buckets if self._blocks_for_bucket(b) >= num_tiles]
        if not fitting:
            raise ValueError(
                f"d1-3B: {num_tiles} tiles exceed the largest encoder bucket "
                f"({buckets[-1]} patches = {self._blocks_for_bucket(buckets[-1])} tiles)"
            )
        num_blocks = self._blocks_for_bucket(fitting[0])

        # 3. Host-side packing, then one device call that writes the cache.
        host = pack_tiles(
            vc, pixel_values.cpu(), shapes, self._pos_table_cpu, num_blocks
        )
        write_ids += [encoder_cache.scratch_block_id] * (num_blocks - len(write_ids))
        device = next(self.visual.parameters()).device
        self.visual(
            host["pixel_values"].to(device),
            host["pos_embeds"].to(device),
            host["key_bias"].to(device),
            host["unshuffle_idx"].to(device),
            encoder_cache.buffer,
            torch.tensor(write_ids, dtype=torch.int64).to(device),
        )

    # ── KV / conv-state cache ────────────────────────────────────────────
    def get_kv_spec(self) -> KVSpec:
        layers = []
        for layer in self.model.layers:
            if layer.is_attention_layer:
                a = layer.self_attn
                layers.append(
                    LayerSpec(
                        name=a.layer_name,
                        num_kv_heads=a.num_key_value_heads_per_rank,
                        head_size=a.cache_head_dim,
                        dtype=a.dtype,
                    )
                )
            else:
                c = layer.conv
                layers.append(
                    LayerSpec(
                        name=c.layer_name,
                        num_kv_heads=1,
                        head_size=c.half,
                        dtype=c.dtype,
                    )
                )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        for layer in self.model.layers:
            mod = layer.cache_module
            if mod.layer_name not in kv_caches:
                raise KeyError(f"KV cache for layer {mod.layer_name} not initialized")
            mod.k_cache = kv_caches[mod.layer_name][0]
            mod.v_cache = kv_caches[mod.layer_name][1]

    # ── Weights ──────────────────────────────────────────────────────────
    def _mappings(self) -> dict[str, str | list[str]]:
        """Model parameter name -> checkpoint key(s). The vision tower and projector
        are not loaded (text-only)."""
        m = {
            "model.embed_tokens.weight": f"{CKPT_LM}.embed_tokens.weight",
            "lm_head.weight": f"{CKPT_LM}.embed_tokens.weight",  # tied
            "model.embedding_norm.weight": f"{CKPT_LM}.embedding_norm.weight",
        }
        for i, layer in enumerate(self.model.layers):
            p, c = f"model.layers.{i}", f"{CKPT_LM}.layers.{i}"
            m[f"{p}.operator_norm.weight"] = f"{c}.operator_norm.weight"
            m[f"{p}.ffn_norm.weight"] = f"{c}.ffn_norm.weight"
            if layer.is_attention_layer:
                m[f"{p}.self_attn.qkv_proj_weight"] = [
                    f"{c}.self_attn.q_proj.weight",
                    f"{c}.self_attn.k_proj.weight",
                    f"{c}.self_attn.v_proj.weight",
                ]
                m[f"{p}.self_attn.o_proj_weight"] = f"{c}.self_attn.out_proj.weight"
                m[f"{p}.self_attn.q_layernorm.weight"] = (
                    f"{c}.self_attn.q_layernorm.weight"
                )
                m[f"{p}.self_attn.k_layernorm.weight"] = (
                    f"{c}.self_attn.k_layernorm.weight"
                )
            else:
                m[f"{p}.conv.in_proj_weight"] = f"{c}.conv.in_proj.weight"
                m[f"{p}.conv.conv_weight"] = f"{c}.conv.conv.weight"
                m[f"{p}.conv.out_proj_weight"] = f"{c}.conv.out_proj.weight"
            f = f"{p}.feed_forward"
            m[f"{f}.gate_proj_weight"] = f"{c}.feed_forward.w1.weight"
            m[f"{f}.up_proj_weight"] = f"{c}.feed_forward.w3.weight"
            m[f"{f}.down_proj_weight"] = f"{c}.feed_forward.w2.weight"
        if self.visual is not None:
            m.update(vision_mappings("visual"))
            m.update(
                vision_layer_mappings("visual", self.vision_config.num_hidden_layers)
            )
        return m

    def load_weights_lite(self, checkpoint_path, device, cache_dir):
        """CPU compile (meta device): give the derived pad-128 buffers their shapes."""
        for layer in self.model.layers:
            if layer.is_attention_layer:
                layer.self_attn.build_padded_weights()
        self._build_fp32_head()

    def load_weights(self, checkpoint_path, device, cache_dir):
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        sd = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, self._mappings(), device
        ).state_dict
        for name, t in sd.items():
            if t.dtype != self.config.torch_dtype:
                sd[name] = t.to(self.config.torch_dtype)
        missing, _ = self.load_state_dict(sd, strict=False, assign=True)
        for layer in self.model.layers:
            if layer.is_attention_layer:
                layer.self_attn.build_padded_weights()
        self._build_fp32_head()
        missing = [n for n in missing if "_pad" not in n]
        if missing:
            raise RuntimeError(f"d1-3B load_weights: missing parameters {missing[:10]}")
