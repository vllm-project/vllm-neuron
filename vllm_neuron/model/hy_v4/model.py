# SPDX-License-Identifier: Apache-2.0
"""Tencent Hy4 (``HYV4ForCausalLM``) text decoder for Neuron.

78 layers of ``iHC -> gated MLA -> iHC -> MoE``, one hidden state carried as
``hc_mult`` = 4 streams. See ``config.py`` for the shape summary and ``moe.py``
for the experts. This file is everything else, and four things in it give wrong
answers without failing:

* **The residual is four streams, not one.** Each sub-block reads a sigmoid-
  weighted sum of the streams (``HYV4HCPre``) and writes its output back to all
  four with a second set of gates (``HYV4HCPost``); there is no plain
  ``x + f(x)`` anywhere. The gate projections, their RMS factor and the
  combination are fp32, as in every reference implementation.
* **MLA is computed in latent space.** The cache holds, per token, the normed
  512-dim KV latent and the shared 64-dim rope key -- 576 values, stored as the
  K and V halves of a 288-wide "head" so that one page is exactly the bytes the
  planner reserved. ``kv_b_proj`` is absorbed: the no-rope query is multiplied
  into latent space, attention runs against the latent directly, and the value
  up-projection is applied to the attention output. Identical in exact
  arithmetic to expanding K/V per head, and 13x less cache.
* **Every head has a learned sink logit**, a column with no value that only
  takes probability mass, and the attention output is gated elementwise by
  ``sigmoid(gate_proj(x))`` before ``o_proj``.
* **The rope pairing is a choice** -- see ``ROPE_STYLE`` in ``config.py``.

**DSA is not implemented yet.** Hy4 attends to the top ``index_topk`` = 2048
keys its lightning indexer picks (shared across groups of four layers). Up to
2048 tokens of context that selection is *every* key, so dense attention is
exact there; ``HYV4ForCausalLM`` refuses a longer ``max_model_len`` rather
than silently diverging past it. The indexer weights are not loaded.

Parallelism is plain TP over the 64 query heads (one per rank at world 64) and
over every MLP's intermediate dim. The MLA down-projections (``q_a_proj``,
``kv_a_proj_with_mqa``) are replicated: their outputs feed per-token norms over
the full width.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import ROPE_STYLE, HYV4Config
from .moe import HYV4MLP, HYV4MoE

HF_PREFIX = "model"


def _replicated_transposed() -> SafetensorsWeightLoader:
    """Whole ``[out, in]`` checkpoint matrix, stored ``[in, out]`` for ``x @ W``."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][:].T)


class HYV4RMSNorm(nn.Module):
    """``weight * normalise(x)``, normalised in fp32 and cast back *before* the scale.

    That order is ``HYV4RMSNorm``'s (``self.weight * hidden.to(input_dtype)``),
    and it is not the zero-centred norm Qwen3.5 uses.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * h.to(dtype)


# ── iHC ──────────────────────────────────────────────────────────────────


class HYV4HCPre(nn.Module):
    """Reduce ``[T, hc, H]`` streams to one ``[T, H]`` input, and emit the post gates.

    ``fn`` projects the RMS-normalised flattened streams to ``2 * hc`` logits:
    the first ``hc`` weigh the streams into the sub-block input, the second ``hc``
    are how strongly the sub-block's output is written back to each stream.
    """

    def __init__(self, config: HYV4Config):
        super().__init__()
        hc, h = config.hc_mult, config.hidden_size
        self.hc_mult = hc
        self.eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.magnitude = config.hc_magnitude
        self.fn = nn.Parameter(torch.empty(2 * hc, hc * h, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty(2 * hc, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(2, dtype=torch.float32))

    def forward(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = streams.shape[0]
        hc = self.hc_mult
        x = streams.float()
        flat = x.reshape(tokens, -1)
        rsqrt = torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + self.eps)
        mixes = (flat @ self.fn.t()) * rsqrt  # [T, 2 hc]
        pre = (
            torch.sigmoid(mixes[:, :hc] * self.scale[0] + self.base[:hc]) + self.hc_eps
        )
        post = (
            self.magnitude
            * torch.sigmoid(mixes[:, hc:] * self.scale[1] + self.base[hc:])
            + self.hc_eps
        )
        out = (pre.unsqueeze(-1) * x).sum(dim=1)
        return out.to(streams.dtype), post


def hc_post(
    out: torch.Tensor, residual: torch.Tensor, post: torch.Tensor
) -> torch.Tensor:
    """``residual[t, i] + post[t, i] * out[t]``, in fp32, back in the streams' dtype."""
    y = post.unsqueeze(-1) * out.float().unsqueeze(1) + residual.float()
    return y.to(residual.dtype)


class HYV4HCHead(nn.Module):
    """Final collapse of the streams before ``model.norm`` (no post gates)."""

    def __init__(self, config: HYV4Config):
        super().__init__()
        hc, h = config.hc_mult, config.hidden_size
        self.eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.fn = nn.Parameter(torch.empty(hc, hc * h, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty(hc, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        tokens = streams.shape[0]
        x = streams.float()
        flat = x.reshape(tokens, -1)
        rsqrt = torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + self.eps)
        mixes = (flat @ self.fn.t()) * rsqrt
        pre = torch.sigmoid(mixes * self.scale + self.base) + self.hc_eps
        return (pre.unsqueeze(-1) * x).sum(dim=1).to(streams.dtype)


# ── RoPE ─────────────────────────────────────────────────────────────────


class HYV4RotaryEmbedding(nn.Module):
    """``(cos, sin)`` of ``[T, rope_dim / 2]`` for the 64 rope dims of MLA."""

    inv_freq: torch.Tensor

    def __init__(self, config: HYV4Config):
        super().__init__()
        dim = config.qk_rope_head_dim
        # device="cpu": the runner builds the model under the meta device, and a
        # buffer that no checkpoint tensor overwrites would stay meta.
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.float, device="cpu") / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x [T, ..., 64]`` by ``cos/sin [T, 32]``, broadcast over middle dims.

    Interleaved pairs ``(x[2i], x[2i+1])`` by default, ``(x[i], x[i+32])`` under
    ``HY4_ROPE_STYLE=neox``. Rotation is done in fp32.
    """
    dtype = x.dtype
    while cos.dim() < x.dim():
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    x = x.float()
    if ROPE_STYLE == "interleaved":
        pairs = x.view(*x.shape[:-1], -1, 2)
        x1, x2 = pairs[..., 0], pairs[..., 1]
        out = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        return out.flatten(-2).to(dtype)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(dtype)


# ── Attention ────────────────────────────────────────────────────────────


class HYV4Attention(nn.Module):
    """Gated MLA with learnable sinks, attending to all keys (see module docstring)."""

    def __init__(self, config: HYV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_name = f"layers.{layer_idx}.self_attn"
        self.dtype = config.torch_dtype
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        heads = config.num_attention_heads
        if heads % self.world_size:
            raise ValueError(
                f"num_attention_heads={heads} must be divisible by "
                f"tp_size={self.world_size}"
            )
        self.num_heads = heads // self.world_size
        self.nope_dim = config.qk_nope_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.qk_dim = config.qk_head_dim
        self.v_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.latent_dim = config.kv_latent_dim
        if self.latent_dim % 2:
            raise ValueError(f"kv latent width {self.latent_dim} must be even")
        # The K and V halves of the cache page, each this wide.
        self.cache_head_dim = self.latent_dim // 2
        self.scaling = self.qk_dim**-0.5
        hidden = config.hidden_size
        nh = self.num_heads

        self.q_a_proj_weight = nn.Parameter(
            torch.empty(hidden, config.q_lora_rank, dtype=self.dtype)
        )
        self.q_a_layernorm = HYV4RMSNorm(
            config.q_lora_rank, config.rms_norm_eps, self.dtype
        )
        self.q_b_proj_weight = nn.Parameter(
            torch.empty(config.q_lora_rank, nh * self.qk_dim, dtype=self.dtype)
        )
        self.kv_a_proj_weight = nn.Parameter(
            torch.empty(hidden, self.latent_dim, dtype=self.dtype)
        )
        self.kv_a_layernorm = HYV4RMSNorm(
            self.kv_lora_rank, config.rms_norm_eps, self.dtype
        )
        # kv_b_proj split per head into its key and value halves, kept in the
        # checkpoint's [out, latent] orientation: both are used against the latent.
        self.w_uk = nn.Parameter(
            torch.empty(nh, self.nope_dim, self.kv_lora_rank, dtype=self.dtype)
        )
        self.w_uv = nn.Parameter(
            torch.empty(nh, self.v_dim, self.kv_lora_rank, dtype=self.dtype)
        )
        self.gate_proj_weight = nn.Parameter(
            torch.empty(hidden, nh * self.v_dim, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(nh * self.v_dim, hidden, dtype=self.dtype)
        )
        self.sinks = nn.Parameter(torch.zeros(nh, dtype=torch.float32))

        self.kv_pages: torch.Tensor | None = None

        set_weight_loader(self.q_a_proj_weight, _replicated_transposed())
        set_weight_loader(self.kv_a_proj_weight, _replicated_transposed())
        set_weight_loader(
            self.q_b_proj_weight,
            sharding_weight_loader(
                shard_dim=1,
                shard_size=nh * self.qk_dim,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        set_weight_loader(
            self.gate_proj_weight,
            sharding_weight_loader(
                shard_dim=1,
                shard_size=nh * self.v_dim,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=nh * self.v_dim,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        set_weight_loader(
            self.sinks,
            sharding_weight_loader(
                shard_dim=0, shard_size=nh, num_shards=self.world_size
            ),
        )
        per_head = self.nope_dim + self.v_dim

        def kv_b(part: str) -> SafetensorsWeightLoader:
            def transform(slices: list, rank: int) -> torch.Tensor:
                (source,) = slices  # [heads * (nope + v), kv_lora_rank]
                rows = source[rank * nh * per_head : (rank + 1) * nh * per_head, :]
                rows = rows.reshape(nh, per_head, self.kv_lora_rank)
                return (
                    rows[:, : self.nope_dim]
                    if part == "k"
                    else rows[:, self.nope_dim :]
                )

            return SafetensorsWeightLoader(transform=transform)

        set_weight_loader(self.w_uk, kv_b("k"))
        set_weight_loader(self.w_uv, kv_b("v"))

    # ── shared pieces ────────────────────────────────────────────────────

    def _project(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x [T, H]`` -> (query ``[T, nh, 576]``, latent ``[T, 576]``, gate).

        The query's no-rope part is absorbed through ``w_uk`` so that it scores
        directly against the cached latent; its rope part scores against the
        cached rope key. Concatenating both gives one 576-wide dot product.
        """
        tokens = x.shape[0]
        q_resid = self.q_a_layernorm(x @ self.q_a_proj_weight)
        q = (q_resid @ self.q_b_proj_weight).view(tokens, self.num_heads, self.qk_dim)
        q_nope, q_rope = q[..., : self.nope_dim], q[..., self.nope_dim :]
        q_rope = apply_rope(q_rope, cos, sin)
        q_lat = torch.einsum("tnd,ndc->tnc", q_nope, self.w_uk)
        query = torch.cat((q_lat, q_rope), dim=-1)

        kv = x @ self.kv_a_proj_weight
        c = self.kv_a_layernorm(kv[:, : self.kv_lora_rank])
        k_rope = apply_rope(kv[:, self.kv_lora_rank :], cos, sin)
        latent = torch.cat((c, k_rope), dim=-1)

        gate = x @ self.gate_proj_weight
        return query, latent, gate

    def _finish(self, o_lat: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """``o_lat [T, nh, 512]`` -> value up-projection, gate, ``o_proj`` partial."""
        tokens = o_lat.shape[0]
        out = torch.einsum("tnc,nvc->tnv", o_lat.to(self.dtype), self.w_uv)
        out = out.reshape(tokens, self.num_heads * self.v_dim)
        out = out * torch.sigmoid(gate.float()).to(out.dtype)
        out = out @ self.o_proj_weight
        if self.world_size > 1:
            out = self.tp_group.all_reduce(out)
        return out

    def _write_kv(self, latent: torch.Tensor, metadata: dict) -> None:
        """Scatter ``[T, 576]`` into the paged cache as two 288-wide halves.

        Same slot handling as ``Qwen3_5Attention._write_kv``: padded tokens
        (slot 0) go to the private scratch page, never to vLLM's null block.

        Written as one 2-index scatter into the ``[pages * 2, block, 288]`` view
        rather than a 4-index one into ``[pages, 2, 1, block, 288]``: with one
        KV head the head index is all zeros, and neuronx-cc folds such an index
        to a rank-0 constant and then fails the decode graph with
        ``NCC_IRRW901 RewriteWeights assertion error: expected tensor.rank=0 to
        be equal len(addrs)``.
        """
        slot_mapping = metadata["slot_mapping"]
        block_size = metadata["block_size"]
        num_pages = self.kv_pages.shape[0]
        half = self.cache_head_dim
        scratch = num_pages - 1
        blocks = torch.where(
            slot_mapping > 0,
            slot_mapping // block_size,
            torch.full_like(slot_mapping, scratch),
        )
        offsets = slot_mapping % block_size
        rows = torch.cat((blocks * 2, blocks * 2 + 1))  # K half, then V half
        values = torch.cat((latent[:, :half], latent[:, half:])).to(self.kv_pages.dtype)
        flat = self.kv_pages.view(num_pages * 2, self.kv_pages.shape[3], half)
        flat.index_put_((rows, torch.cat((offsets, offsets))), values)

    def _with_sinks(self, scores: torch.Tensor) -> torch.Tensor:
        """Softmax over ``[..., keys]`` plus one sink column per head, sink dropped.

        ``scores`` is ``[nh, rows, keys]``-shaped with heads first.
        """
        sink = self.sinks.view(-1, 1, 1).expand(scores.shape[0], scores.shape[1], 1)
        probs = torch.softmax(torch.cat((scores, sink.float()), dim=-1), dim=-1)
        return probs[..., :-1]

    # ── phases ───────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
    ) -> torch.Tensor:
        metadata = attn_metadata[self.layer_name]
        if metadata["max_query_len"] <= metadata["decode_token_threshold"]:
            return self.forward_decode(x, positions, position_embeddings, metadata)
        return self.forward_prefill(x, position_embeddings, metadata)

    def forward_prefill(
        self, x: torch.Tensor, position_embeddings: tuple, metadata: dict
    ) -> torch.Tensor:
        """One sequence, causal over itself (the assumption Qwen3.5's prefill makes)."""
        cos, sin = position_embeddings
        tokens = x.shape[0]
        query, latent, gate = self._project(x.to(self.dtype), cos, sin)
        self._write_kv(latent, metadata)

        q = query.transpose(0, 1).float()  # [nh, T, 576]
        scores = (q @ latent.float().t().unsqueeze(0)) * self.scaling  # [nh, T, T]
        causal = torch.ones(tokens, tokens, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(causal, float("-inf"))
        probs = self._with_sinks(scores)
        o_lat = probs.to(self.dtype) @ latent[:, : self.kv_lora_rank].unsqueeze(0)
        return self._finish(o_lat.transpose(0, 1), gate)

    def forward_decode(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple,
        metadata: dict,
    ) -> torch.Tensor:
        """One new token per request against its paged latent history.

        The gather follows ``Qwen3_5Attention.forward_decode``: pages flattened to
        rank 3 and indexed per request, dead rows and table padding redirected to
        the private zero page, the current token's own latent appended as an
        extra key column (the ``index_put_`` above is not visible to the gather).
        """
        cos, sin = position_embeddings
        block_table = metadata["block_table_tensor"]
        num_reqs = block_table.shape[0]
        query, latent, gate = self._project(x.to(self.dtype), cos, sin)
        self._write_kv(latent, metadata)

        zero_page = self.kv_pages.shape[0] - 2
        bt = block_table.to(torch.long)
        live_rows = (metadata["slot_mapping"].view(num_reqs, -1)[:, 0] > 0).view(-1, 1)
        blocks = torch.where(
            live_rows & (bt > 0) & (bt < zero_page), bt, torch.full_like(bt, zero_page)
        )
        num_pages, block_size = self.kv_pages.shape[0], self.kv_pages.shape[3]
        ctx_len = blocks.shape[1] * block_size
        half = self.cache_head_dim

        # [pages, 2, 1, block, 288] -> [pages * 2, block, 288]: K half of page p
        # is row 2p, V half is row 2p + 1.
        flat = self.kv_pages.reshape(num_pages * 2, block_size, half)
        k_half = flat[(blocks * 2).reshape(-1)].reshape(num_reqs, ctx_len, half)
        v_half = flat[(blocks * 2 + 1).reshape(-1)].reshape(num_reqs, ctx_len, half)
        ctx = torch.cat((k_half, v_half), dim=-1)  # [reqs, ctx, 576]

        key_positions = torch.arange(ctx_len, device=x.device).view(1, ctx_len)
        visible = key_positions < positions.view(num_reqs, 1)  # [reqs, ctx]
        # A select, not a multiply: recycled pages can hold non-finite bytes.
        ctx = torch.where(visible.unsqueeze(-1), ctx, torch.zeros_like(ctx))

        q = query.float()  # [reqs, nh, 576]
        scores = (q @ ctx.float().transpose(1, 2)) * self.scaling  # [reqs, nh, ctx]
        scores = scores.masked_fill(~visible.unsqueeze(1), float("-inf"))
        scores_cur = (q * latent.float().unsqueeze(1)).sum(
            -1, keepdim=True
        ) * self.scaling

        # Heads first for the sink broadcast, then back.
        logits = torch.cat((scores, scores_cur), dim=-1).transpose(
            0, 1
        )  # [nh, reqs, ctx+1]
        probs = self._with_sinks(logits).transpose(0, 1)  # [reqs, nh, ctx+1]
        c_all = torch.cat(
            (
                ctx[..., : self.kv_lora_rank],
                latent[:, None, : self.kv_lora_rank].to(ctx.dtype),
            ),
            dim=1,
        )
        o_lat = probs.to(self.dtype) @ c_all.to(self.dtype)  # [reqs, nh, 512]
        return self._finish(o_lat, gate)


# ── Decoder ──────────────────────────────────────────────────────────────


class HYV4DecoderLayer(nn.Module):
    def __init__(self, config: HYV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        dtype = config.torch_dtype
        self.input_layernorm = HYV4RMSNorm(
            config.hidden_size, config.rms_norm_eps, dtype
        )
        self.post_attention_layernorm = HYV4RMSNorm(
            config.hidden_size, config.rms_norm_eps, dtype
        )
        self.attn_hc = HYV4HCPre(config)
        self.ffn_hc = HYV4HCPre(config)
        self.self_attn = HYV4Attention(config, layer_idx)
        self.mixer_name = self.self_attn.layer_name
        self.is_moe = config.is_moe_layer(layer_idx)
        self.mlp = (
            HYV4MoE(config)
            if self.is_moe
            else HYV4MLP(
                config.hidden_size, config.intermediate_size, self.world_size, dtype
            )
        )

    def forward(
        self,
        streams: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
    ) -> torch.Tensor:
        residual = streams
        x, post = self.attn_hc(streams)
        x = self.self_attn(
            self.input_layernorm(x), positions, position_embeddings, attn_metadata
        )
        streams = hc_post(x, residual, post)

        residual = streams
        x, post = self.ffn_hc(streams)
        x = self.mlp(self.post_attention_layernorm(x))
        if self.world_size > 1:
            x = self.tp_group.all_reduce(x)
        return hc_post(x, residual, post)


class HYV4TextModel(nn.Module):
    def __init__(self, config: HYV4Config):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
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
                num_shards=self.embed_tokens.tp_size,
            ),
        )
        self.layers = nn.ModuleList(
            HYV4DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.hc_head = HYV4HCHead(config)
        self.norm = HYV4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = HYV4RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: dict,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        streams = x.unsqueeze(1).expand(-1, self.config.hc_mult, -1).contiguous()
        position_embeddings = self.rotary_emb(positions)
        for layer in self.layers:
            streams = layer(streams, positions, position_embeddings, attn_metadata)
        return self.norm(self.hc_head(streams))


def checkpoint_mappings(config: HYV4Config) -> dict[str, str]:
    """Port parameter name -> converted-checkpoint key, for every loaded parameter."""
    m: dict[str, str] = {}
    for i in range(config.num_hidden_layers):
        hf, ours = f"{HF_PREFIX}.layers.{i}", f"model.layers.{i}"
        m[f"{ours}.input_layernorm.weight"] = f"{hf}.input_layernorm.weight"
        m[f"{ours}.post_attention_layernorm.weight"] = (
            f"{hf}.post_attention_layernorm.weight"
        )
        for mine, theirs in (("attn_hc", "hc_attn_layer"), ("ffn_hc", "hc_mlp_layer")):
            for p in ("fn", "base", "scale"):
                m[f"{ours}.{mine}.{p}"] = f"{hf}.{theirs}.hc_pre.hc_{p}"

        a, ha = f"{ours}.self_attn", f"{hf}.self_attn"
        m[f"{a}.q_a_proj_weight"] = f"{ha}.q_a_proj.weight"
        m[f"{a}.q_a_layernorm.weight"] = f"{ha}.q_a_layernorm.weight"
        m[f"{a}.q_b_proj_weight"] = f"{ha}.q_b_proj.weight"
        m[f"{a}.kv_a_proj_weight"] = f"{ha}.kv_a_proj_with_mqa.weight"
        m[f"{a}.kv_a_layernorm.weight"] = f"{ha}.kv_a_layernorm.weight"
        m[f"{a}.w_uk"] = f"{ha}.kv_b_proj.weight"
        m[f"{a}.w_uv"] = f"{ha}.kv_b_proj.weight"
        m[f"{a}.gate_proj_weight"] = f"{ha}.linear_gate.weight"
        m[f"{a}.o_proj_weight"] = f"{ha}.o_proj.weight"
        m[f"{a}.sinks"] = f"{ha}.learnable_sink_param"

        mlp, hm = f"{ours}.mlp", f"{hf}.mlp"
        if config.is_moe_layer(i):
            m[f"{mlp}.router_weight"] = f"{hm}.gate.weight"
            m[f"{mlp}.e_score_correction_bias"] = f"{hm}.gate.e_score_correction_bias"
            m[f"{mlp}.experts_gate_up_weight"] = f"{hm}.experts.gate_up_proj"
            m[f"{mlp}.experts_gate_up_scale"] = f"{hm}.experts.gate_up_proj_scale"
            m[f"{mlp}.experts_down_weight"] = f"{hm}.experts.down_proj_t"
            m[f"{mlp}.experts_down_scale"] = f"{hm}.experts.down_proj_scale"
            shared, hs = f"{mlp}.shared_experts", f"{hm}.shared_experts"
        else:
            shared, hs = mlp, hm
        for p in ("gate_proj", "up_proj", "down_proj"):
            m[f"{shared}.{p}_weight"] = f"{hs}.{p}.weight"

    m["model.embed_tokens.weight"] = f"{HF_PREFIX}.embed_tokens.weight"
    m["model.norm.weight"] = f"{HF_PREFIX}.norm.weight"
    for p in ("fn", "base", "scale"):
        m[f"model.hc_head.{p}"] = f"{HF_PREFIX}.hc_head.hc_head_{p}"
    m["lm_head.weight"] = (
        f"{HF_PREFIX}.embed_tokens.weight"
        if config.tie_word_embeddings
        else "lm_head.weight"
    )
    return m


class HYV4ForCausalLM(nn.Module):
    """Hy4-preview text model, on the ``convert_mxfp8_to_neuron_fp8.py`` checkpoint.

    The MTP layer (``model.mtp_layers.0``) and the DSA indexers are not loaded.
    """

    # Two private pages per buffer (zero page + write sink), as Qwen3.5 uses.
    kv_cache_page_major = True

    def __init__(
        self, config: HYV4Config | None = None, *, vllm_config=None, prefix: str = ""
    ):
        # ``vllm_config``/``prefix`` exist only so vLLM's engine process, which
        # has no class of its own for this architecture and so inspects this
        # one, recognises it as a text-generation model (``is_vllm_model``
        # checks the keyword). The runner builds it through ``from_configs``.
        if config is None:
            raise TypeError("build HYV4ForCausalLM with from_configs()")
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        nc = config.neuron_config
        from vllm.config import get_current_vllm_config

        try:
            max_len = get_current_vllm_config().model_config.max_model_len
        except (AssertionError, AttributeError, RuntimeError):
            max_len = None  # no engine context: CPU checks
        if max_len is not None and max_len > config.index_topk:
            raise NotImplementedError(
                f"max_model_len={max_len} > index_topk={config.index_topk}: past "
                f"{config.index_topk} tokens Hy4 attends only to the keys its DSA "
                f"indexer selects, which is not implemented yet, and dense "
                f"attention would silently diverge from the model."
            )

        self.model = HYV4TextModel(config)

        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        self._gather_logits = nc is not None and (
            nc.max_logprobs != 0 or nc.debug_logits_dir is not None
        )
        self.lm_dtype = torch.float32 if config.lm_head_fp32 else config.torch_dtype
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=self.lm_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )
        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config, process_group=self.tp_group.device_group
            )

    # ── KV cache ─────────────────────────────────────────────────────────

    def get_kv_spec(self) -> KVSpec:
        return KVSpec(
            layers=[
                LayerSpec(
                    name=layer.self_attn.layer_name,
                    num_kv_heads=1,
                    head_size=layer.self_attn.cache_head_dim,
                    dtype=self.config.torch_dtype,
                )
                for layer in self.model.layers
            ]
        )

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        for layer in self.model.layers:
            attn = layer.self_attn
            tensors = kv_caches.get(attn.layer_name)
            if tensors is None:
                raise Exception(f"cache for layer {attn.layer_name} not initialized")
            if len(tensors) != 1:
                raise Exception(
                    f"layer {attn.layer_name} expected one page-major tensor, "
                    f"got {len(tensors)}"
                )
            attn.kv_pages = tensors[0]

    # ── Forward ──────────────────────────────────────────────────────────

    # The other half of vLLM's text-generation protocol (see ``__init__``).
    # The runner calls ``forward`` only; these are not on its path.
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids, scatter_tokens=False)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states.to(self.lm_dtype))

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        rotary_position_ids: torch.Tensor | None = None,
        attn_metadata: dict | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
    ):
        positions = positions.to(torch.int32)
        hidden = self.model(input_ids, positions, attn_metadata, rank=rank)
        hidden = torch.index_select(hidden, 0, sampling_positions)
        logits = self.lm_head(hidden.to(self.lm_dtype))

        gathered_logits = None
        if self._gather_logits:
            gathered_logits = self.tp_group.all_gather(logits, dim=1)
        if self.on_device_sampling_config is None:
            return logits
        sampled = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )
        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            return rejection_sampler(spec_decode_metadata, sampled)
        return sampled, gathered_logits

    # ── Construction and weights ─────────────────────────────────────────

    @classmethod
    def from_configs(
        cls, hf_config, text_neuron_config=None, vision_neuron_config=None
    ):
        if vision_neuron_config is not None:
            raise NotImplementedError("Hy4 is a text-only model")
        config = HYV4Config.from_configs(hf_config, neuron_config=text_neuron_config)
        quant = config.neuron_quantization or {}
        if quant.get("format") != "fp8_row_experts":
            raise ValueError(
                "HYV4ForCausalLM on Neuron loads the Trn2 fp8 checkpoint written by "
                "examples/vllm_neuron/models/hy_v4/convert_mxfp8_to_neuron_fp8.py "
                "(config.json has no neuron_quantization.format=fp8_row_experts). "
                "The bf16 checkpoint does not fit one trn2.48xlarge, and the "
                "published MXFP8 one has no Trn2 kernel."
            )
        return cls(config)

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank,
            self.world_size,
            self,
            checkpoint_mappings(self.config),
            device,
            strict=False,
        ).state_dict

        # strict=False (buffers are not in the checkpoint), so a wrong mapping key
        # would leave a parameter at torch.empty and generate fluent garbage.
        params = dict(self.named_parameters())
        unfilled = sorted(set(params) - set(rank_sharded))
        if unfilled:
            raise RuntimeError(
                f"{len(unfilled)} parameter(s) got no checkpoint tensor: {unfilled[:8]}"
                + (" ..." if len(unfilled) > 8 else "")
            )
        for name, tensor in rank_sharded.items():
            want = params[name].dtype
            if tensor.dtype != want:
                rank_sharded[name] = tensor.to(want)
        self.load_state_dict(rank_sharded, strict=False, assign=True)


__all__ = ["HYV4ForCausalLM", "HYV4TextModel"]
