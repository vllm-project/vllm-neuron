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
  512-dim KV latent, the shared 64-dim rope key and the 128-dim DSA indexer key
  (stored first) -- 704 values, stored as the K and V halves of a 352-wide "head" so that one
  page is exactly the bytes the planner reserved. ``kv_b_proj`` is absorbed:
  the no-rope query is multiplied into latent space, attention runs against
  the latent directly, and the value up-projection is applied to the attention
  output. Identical in exact arithmetic to expanding K/V per head. Note what
  it does and does not save: the latent is 704 values against 64 x 512 for
  all heads expanded, but every rank holds a full copy, and at one head per
  rank an expanded cache would be 512 + 128 -- so at world 64 the latent
  cache is ~10% *larger* per rank. Sharding the latent's tokens across ranks
  (context parallelism) is what would turn it into a saving.
* **Every head has a learned sink logit**, a column with no value that only
  takes probability mass, and the attention output is gated elementwise by
  ``sigmoid(gate_proj(x))`` before ``o_proj``.
* **The rope pairing is interleaved** -- see ``ROPE_STYLE`` in ``config.py``.

**DSA.** Each query attends to the top ``index_topk`` = 2048 keys a lightning
indexer picks (``HYV4Indexer``); the 20 ``"full"`` layers run one, the other 58
reuse the selection of the last full layer before them. Below 2048 visible keys
the selection is every key. The sparse attention is computed as dense attention
masked to the selection: exact, but its cost still grows with the context.

**Prefill is chunked** (the plugin's segmented prefill): each step is one chunk
of one request, attending to the ``cached_seq_len`` tokens already in the cache
plus causally to itself, so a prompt longer than the prefill bucket is handled
across steps and per-step cost does not grow with the bucket.

Parallelism is plain TP over the 64 query heads (one per rank at world 64) and
over every MLP's intermediate dim. The MLA down-projections (``q_a_proj``,
``kv_a_proj_with_mqa``) are replicated: their outputs feed per-token norms over
the full width.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
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

# HY4_DEBUG_CAPTURE=1 traces ``capture_tensor`` calls for the layers listed in
# HY4_DEBUG_LAYERS (default "0") into the graph (used with
# neuron_config.tensor_capture). Off, nothing is traced.
DEBUG_CAPTURE = os.environ.get("HY4_DEBUG_CAPTURE") == "1"
DEBUG_LAYERS = {int(x) for x in os.environ.get("HY4_DEBUG_LAYERS", "0").split(",")}
# HY4_PERF_SKIP=collectives,attn,moe -- timing experiments only, outputs are
# wrong: "collectives" makes the per-layer all-reduces the identity and the
# DSA mask all-gather a local repeat; "attn"/"moe" replace that sub-block's
# output with zeros (attention then writes no KV and selects nothing).
PERF_SKIP = {x for x in os.environ.get("HY4_PERF_SKIP", "").split(",") if x}


def _all_reduce(group, t: torch.Tensor) -> torch.Tensor:
    return t if "collectives" in PERF_SKIP else group.all_reduce(t)


def _all_gather(group, t: torch.Tensor, dim: int, world: int) -> torch.Tensor:
    if "collectives" in PERF_SKIP:
        return torch.cat([t] * world, dim=dim)
    return group.all_gather(t, dim=dim)


def _capture(layer_idx: int, name: str, tensor: torch.Tensor) -> None:
    if DEBUG_CAPTURE and layer_idx in DEBUG_LAYERS:
        from vllm_neuron.accuracy import capture_tensor

        capture_tensor(f"hy4.l{layer_idx}.{name}", tensor)


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

# Bisection steps for the top-k threshold; see ``select_topk``. 24 halvings of
# the per-row score range leave ties within ~6e-8 of it unresolved, which only
# ever widens the selection by keys tied with the k-th.
TOPK_BISECTION_STEPS = 24
# Keys per block of the prefill attention's online softmax (see
# ``HYV4Attention._attend_blocked``).
ATTN_KEY_BLOCK = 2048
# Stand-in for -inf/+inf in the selection, where real infinities would turn the
# bisection midpoint into NaN.
_BIG = 3.0e38


def select_topk(scores: torch.Tensor, visible: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean mask of each row's top-``k`` visible keys (all of them if fewer).

    ``scores``/``visible`` are ``[rows, keys]``; every row must have at least one
    visible key (each query always sees itself).

    By bisection on a per-row threshold rather than ``torch.topk``: it is a few
    elementwise passes over ``[rows, keys]``, with no k=2048 sort and no scatter
    to turn indices back into a mask. The invariant is ``count(s >= lo) >= k``
    (or ``lo`` is the row minimum, which selects everything visible), so the
    result is exactly the top-k up to keys tied with the k-th score.
    """
    s = scores.float()
    hi = s.masked_fill(~visible, -_BIG).amax(dim=-1, keepdim=True)
    lo = s.masked_fill(~visible, _BIG).amin(dim=-1, keepdim=True)
    for _ in range(TOPK_BISECTION_STEPS):
        mid = (lo + hi) * 0.5
        count = ((s >= mid) & visible).float().sum(dim=-1, keepdim=True)
        up = count >= k
        lo = torch.where(up, mid, lo)
        hi = torch.where(up, hi, mid)
    return visible & (s >= lo)


def _zero_beyond(ctx: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
    """Zero the gathered cache entries at positions ``>= limit`` (per row).

    Not needed for correctness -- those keys are masked out of the scores, and
    this model's single KV group only ever holds finite entries -- but in the
    one build without it the 32K prefill was ~2.7x slower (150 s vs 55 s per
    chunk) and its graph twice the size (596K vs 284K instructions per
    subgraph). Why is not understood; the masked build is the measured one.

    The predicate here is computed *separately* from ``in_prefix`` (float
    subtraction, not the integer compare) on purpose. Applying ``in_prefix``
    itself to the ``[rows, 32K, 704]`` context made neuronx-cc 2.27 compute
    ``in_prefix`` wrongly wherever it was reused -- 128-key blocks set past the
    prefix, even on a dead row at position 0 -- so every 32K decode attended
    to the wrong keys while the CPU, and the same ops compiled separately,
    were right. A second, independent predicate compiles correctly (checked
    on device in a standalone graph of the decode attention chain).
    """
    rows = ctx.shape[0] if ctx.dim() == 3 else 1
    ctx_len = ctx.shape[-2]
    positions = torch.arange(ctx_len, device=ctx.device, dtype=torch.float32)
    # A tensor threshold: comparing against a Python float lowers to an f64
    # constant, which neuronx-cc rejects (NCC_ESPP004).
    half = torch.full((1, 1), 0.5, dtype=torch.float32, device=ctx.device)
    keep = (limit.float().view(rows, 1) - positions.view(1, ctx_len)) > half
    keep = keep.to(ctx.dtype)
    return ctx * (keep.unsqueeze(-1) if ctx.dim() == 3 else keep.view(ctx_len, 1))


class HYV4Indexer(nn.Module):
    """DSA lightning indexer: scores every visible key, the top ``index_topk`` attend.

    ``score[t, s] = sum_h w[t, h] * relu(q[t, h] . k[s])`` with 32 query heads of
    128 against one 128-wide key per token (LayerNorm'd, with the rope applied to
    its *last* 64 dims, as is the query's). Replicated on every rank: it is small
    (~9.4M parameters) and its output, a selection over keys, is needed by each
    rank's attention head in full. The reference's Hadamard rotation and fp8
    scoring are precision/speed devices that leave the selection's definition
    unchanged, and are skipped, as transformers does.
    """

    def __init__(self, config: HYV4Config):
        super().__init__()
        self.dtype = config.torch_dtype
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        hidden = config.hidden_size
        self.wq_b_weight = nn.Parameter(
            torch.empty(
                config.q_lora_rank, self.n_heads * self.head_dim, dtype=self.dtype
            )
        )
        self.wk_weight = nn.Parameter(
            torch.empty(hidden, self.head_dim, dtype=self.dtype)
        )
        self.k_norm_weight = nn.Parameter(
            torch.ones(self.head_dim, dtype=torch.float32)
        )
        self.k_norm_bias = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32))
        self.weights_proj_weight = nn.Parameter(
            torch.empty(hidden, self.n_heads, dtype=torch.float32)
        )
        # vLLM's HYV4 indexer uses 1e-6 here, not rms_norm_eps.
        self.k_norm_eps = 1e-6
        self.weight_scale = self.head_dim**-0.5 * self.n_heads**-0.5
        self.rope_first = config.indexer_rope_first
        for param in (self.wq_b_weight, self.wk_weight, self.weights_proj_weight):
            set_weight_loader(param, _replicated_transposed())

    def _rope_tail(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        if self.rope_first:  # GLM-5.x: the rotated dims lead
            r = self.rope_dim
            return torch.cat((apply_rope(x[..., :r], cos, sin), x[..., r:]), dim=-1)
        keep = self.head_dim - self.rope_dim
        return torch.cat((x[..., :keep], apply_rope(x[..., keep:], cos, sin)), dim=-1)

    def keys(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """``x [T, H]`` -> ``[T, 128]``, what the cache stores for this layer."""
        k = F.layer_norm(
            (x @ self.wk_weight).float(),
            (self.head_dim,),
            self.k_norm_weight,
            self.k_norm_bias,
            self.k_norm_eps,
        )
        return self._rope_tail(k, cos, sin).to(self.dtype)

    def queries(
        self,
        q_resid: torch.Tensor,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (``[T, heads, 128]`` queries, ``[T, heads]`` fp32 head weights)."""
        q = (q_resid @ self.wq_b_weight).view(-1, self.n_heads, self.head_dim)
        weights = (x.float() @ self.weights_proj_weight) * self.weight_scale
        return self._rope_tail(q, cos, sin), weights


class HYV4Attention(nn.Module):
    """Gated MLA with learnable sinks over the keys DSA selects.

    The cache entry per token is 704 wide: the 128-wide indexer key, then the
    576-wide MLA latent (normed kv latent + rope key), stored as the K and V
    halves of one 352-wide head. Layers whose indexer is ``"shared"``
    leave the indexer slot zero and reuse the selection of the last ``"full"``
    layer before them, which the decoder threads through.
    """

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
        self.index_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.entry_dim = self.latent_dim + self.index_dim
        # Entry layout [index key 128 | kv latent 512 | rope key 64]: the
        # indexer key first, so both slices start on a 128-column boundary.
        # With it last (column 576), the prefill indexer's matmul against
        # ``keys[:, 576:]`` failed neuronx-cc ISA validation at -O1 ("Copy;
        # inst failed assertion check: 'start_addr_active_channels'").
        self.index_slice = slice(0, self.index_dim)
        self.latent_slice = slice(self.index_dim, self.entry_dim)
        if self.entry_dim % 2:
            raise ValueError(f"cache entry width {self.entry_dim} must be even")
        # The K and V halves of the cache page, each this wide.
        self.cache_head_dim = self.entry_dim // 2
        self.scaling = self.qk_dim**-0.5
        hidden = config.hidden_size
        nh = self.num_heads

        self.indexer = (
            HYV4Indexer(config) if config.indexer_types[layer_idx] == "full" else None
        )

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
        # kv_b_proj split per head into its key and value halves. The key half keeps the
        # checkpoint's [out, latent] orientation (it contracts over out); the value half is
        # stored [latent, out], the way it contracts. In [out, latent] the value up-projection
        # transposed the weight every step, and in GLM-5.3's decode graph that
        # transpose's spill/reload got an out-of-range address (NCC_INLA001 / ISA validation).
        self.w_uk = nn.Parameter(
            torch.empty(nh, self.nope_dim, self.kv_lora_rank, dtype=self.dtype)
        )
        self.w_uv = nn.Parameter(
            torch.empty(nh, self.kv_lora_rank, self.v_dim, dtype=self.dtype)
        )
        self.gated = config.gated_mla
        if self.gated:
            self.gate_proj_weight = nn.Parameter(
                torch.empty(hidden, nh * self.v_dim, dtype=self.dtype)
            )
        self.o_proj_weight = nn.Parameter(
            torch.empty(nh * self.v_dim, hidden, dtype=self.dtype)
        )
        self.learnable_sink = config.learnable_sink
        if self.learnable_sink:
            self.sinks = nn.Parameter(torch.zeros(nh, dtype=torch.float32))
        else:
            # No sinks (GLM-5.x): a -1e30 logit takes exactly zero weight in every softmax
            # here, dense and key-blocked, so no path changes. device="cpu": the model is
            # built under the meta device, and a buffer made there has no data to move (as
            # for HYV4RotaryEmbedding.inv_freq).
            self.register_buffer(
                "sinks", torch.full((nh,), -1.0e30, device="cpu"), persistent=False
            )

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
        if self.gated:
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
        if self.learnable_sink:
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
                if part == "k":
                    return rows[:, : self.nope_dim]
                return rows[:, self.nope_dim :].transpose(1, 2).contiguous()

            return SafetensorsWeightLoader(transform=transform)

        set_weight_loader(self.w_uk, kv_b("k"))
        set_weight_loader(self.w_uv, kv_b("v"))

    # ── shared pieces ────────────────────────────────────────────────────

    def _project(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x [T, H]`` -> (query ``[T, nh, 576]``, entry ``[T, 704]``, gate, q_resid).

        The query's no-rope part is absorbed through ``w_uk`` so that it scores
        directly against the cached latent; its rope part scores against the
        cached rope key. Concatenating both gives one 576-wide dot product.
        ``entry`` is what this token writes to the cache.
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
        if self.indexer is not None:
            index_key = self.indexer.keys(x, cos, sin)
        else:
            index_key = torch.zeros(
                tokens, self.index_dim, dtype=self.dtype, device=x.device
            )
        entry = torch.cat((index_key.to(c.dtype), c, k_rope.to(c.dtype)), dim=-1)

        gate = x @ self.gate_proj_weight if self.gated else None
        return query, entry, gate, q_resid

    def _finish(self, o_lat: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """``o_lat [T, nh, 512]`` -> value up-projection, gate, ``o_proj`` partial."""
        tokens = o_lat.shape[0]
        out = torch.einsum("tnc,ncv->tnv", o_lat.to(self.dtype), self.w_uv)
        out = out.reshape(tokens, self.num_heads * self.v_dim)
        if gate is not None:
            out = out * torch.sigmoid(gate.float()).to(out.dtype)
        out = out @ self.o_proj_weight
        if self.world_size > 1:
            out = _all_reduce(self.tp_group, out)
        return out

    def _write_kv(self, entry: torch.Tensor, metadata: dict) -> None:
        """Scatter ``[T, 704]`` into the paged cache as two 352-wide halves.

        Same slot handling as ``Qwen3_5Attention._write_kv``: padded tokens
        (slot 0) go to the private scratch page, never to vLLM's null block.

        Written as one 2-index scatter into the ``[pages * 2, block, 352]`` view
        rather than a 4-index one into ``[pages, 2, 1, block, 352]``: with one
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
        values = torch.cat((entry[:, :half], entry[:, half:])).to(self.kv_pages.dtype)
        flat = self.kv_pages.view(num_pages * 2, self.kv_pages.shape[3], half)
        flat.index_put_((rows, torch.cat((offsets, offsets))), values)

    def _gather(
        self, block_table: torch.Tensor, live_rows: torch.Tensor
    ) -> torch.Tensor:
        """``[R, W]`` block ids -> ``[R, W * block_size, 704]`` cache entries.

        Pages flattened to rank 3 and indexed per row, as in
        ``Qwen3_5Attention.forward_decode``. Dead rows, table padding (-1) and
        vLLM's null block are redirected to the private zero page.
        """
        num_rows = block_table.shape[0]
        zero_page = self.kv_pages.shape[0] - 2
        bt = block_table.to(torch.long)
        blocks = torch.where(
            live_rows & (bt > 0) & (bt < zero_page), bt, torch.full_like(bt, zero_page)
        )
        num_pages, block_size = self.kv_pages.shape[0], self.kv_pages.shape[3]
        ctx_len = blocks.shape[1] * block_size
        half = self.cache_head_dim
        # [pages, 2, 1, block, 352] -> [pages * 2, block, 352]: K half of page p
        # is row 2p, V half is row 2p + 1.
        flat = self.kv_pages.reshape(num_pages * 2, block_size, half)
        k_half = flat[(blocks * 2).reshape(-1)].reshape(num_rows, ctx_len, half)
        v_half = flat[(blocks * 2 + 1).reshape(-1)].reshape(num_rows, ctx_len, half)
        return torch.cat((k_half, v_half), dim=-1)

    def _with_sinks(self, scores: torch.Tensor) -> torch.Tensor:
        """Softmax over ``[..., keys]`` plus one sink column per head, sink dropped.

        ``scores`` is ``[nh, rows, keys]``-shaped with heads first.
        """
        sink = self.sinks.view(-1, 1, 1).expand(scores.shape[0], scores.shape[1], 1)
        probs = torch.softmax(torch.cat((scores, sink.float()), dim=-1), dim=-1)
        return probs[..., :-1]

    def _attend_blocked(
        self, query: torch.Tensor, keys: torch.Tensor, selected: torch.Tensor
    ) -> torch.Tensor:
        """Prefill attention with sinks, ``[nh, C, 512]``, in key blocks.

        ``query`` ``[nh, C, 576]``, ``keys`` ``[K, 704]``, ``selected`` ``[C, K]``.

        At 32K the ``[C, K]`` score matrix is ~285 MB of fp32 per head per
        layer against ~24 MB of SBUF per core, and a softmax over its full rows
        made neuronx-cc spill it: profiled on one layer, 2.6 GB spilled out and
        7.3 GB reloaded per step, as tens of millions of ~150-byte DMA packets,
        with the Tensor engine 6% busy. So keys are processed ``ATTN_KEY_BLOCK``
        at a time with an online softmax (running max ``m``, running sum ``l``,
        and the output rescaled by ``exp(m_old - m_new)``), and the sink -- a
        logit with no value -- joins at the end, in the max and the sum.
        """
        nh, chunk, _ = query.shape
        q = query.float() * self.scaling
        # A finite floor, not -inf: with -inf a row that has nothing selected
        # in the blocks so far computes exp(-inf - -inf) = NaN. With the floor,
        # its masked keys briefly count with weight 1, and the first selected
        # key (every row has one) rescales them by exp(floor - m) = 0.
        floor = -1.0e30
        m = torch.full((nh, chunk, 1), floor, device=q.device)
        l = torch.zeros((nh, chunk, 1), device=q.device)
        acc = torch.zeros((nh, chunk, self.kv_lora_rank), device=q.device)
        total = keys.shape[0]
        for start in range(0, total, ATTN_KEY_BLOCK):
            stop = min(start + ATTN_KEY_BLOCK, total)
            lat = keys[start:stop, self.latent_slice]
            s = q @ lat.float().t().unsqueeze(0)  # [nh, C, B]
            s = s.masked_fill(~selected[:, start:stop].unsqueeze(0), floor)
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            corr = torch.exp(m - m_new)
            e = torch.exp(s - m_new)
            l = l * corr + e.sum(dim=-1, keepdim=True)
            acc = (
                acc * corr
                + (e.to(self.dtype) @ lat[:, : self.kv_lora_rank].unsqueeze(0)).float()
            )
            m = m_new
        sink = self.sinks.view(-1, 1, 1).float()
        m_f = torch.maximum(m, sink)
        scale = torch.exp(m - m_f)
        return acc * scale / (l * scale + torch.exp(sink - m_f))

    # ── phases ───────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
        selected: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (output, this layer's key selection, for the next shared layers)."""
        metadata = attn_metadata[self.layer_name]
        if self.indexer is None and selected is None:
            raise ValueError(
                f"{self.layer_name} shares its DSA selection but no full-indexer "
                f"layer ran before it"
            )
        if metadata["max_query_len"] <= metadata["decode_token_threshold"]:
            rows = metadata["block_table_tensor"].shape[0]
            # See forward_decode_split for why the batch picks the form.
            split = rows == 1 or rows >= 16
            decode = self.forward_decode_split if split else self.forward_decode
            return decode(x, positions, position_embeddings, metadata, selected)
        return self.forward_prefill(x, position_embeddings, metadata, selected, rank)

    def _select_prefill(
        self,
        iq: torch.Tensor,
        iw: torch.Tensor,
        ik: torch.Tensor,
        visible: torch.Tensor,
        rank: torch.Tensor | None,
    ) -> torch.Tensor:
        """DSA selection ``[C, K]`` for a prefill chunk, split by query across ranks.

        Every rank needs the whole selection (its head attends to every query's
        keys), but each query's selection depends only on that query. So rank
        ``r`` scores and selects for its ``C / world`` queries against all keys,
        and one all-gather assembles the mask. Done replicated instead, the
        indexer and the top-k bisection were ~90% of the prefill graph at 32K
        context -- 15.7M instructions at -O3, over neuronx-cc's 10M limit.

        ``rank`` is the runner's tensor, not ``rank_in_group``: a Python rank
        would bake into the trace and make 64 different graphs.
        """
        chunk, world = iq.shape[0], self.world_size
        if world == 1 or chunk % world:
            rows = slice(None)
            local_q, local_w, local_vis = iq, iw, visible
        else:
            per_rank = chunk // world
            if rank is None:
                rank = torch.tensor(self.tp_group.rank_in_group, device=iq.device)
            rows = rank.reshape(1).to(torch.long) * per_rank + torch.arange(
                per_rank, device=iq.device
            )
            local_q = torch.index_select(iq, 0, rows)
            local_w = torch.index_select(iw, 0, rows)
            local_vis = torch.index_select(visible, 0, rows)
        # A flat matmul, not ``einsum("qhd,kd->qhk")``: with the key-blocked
        # attention in the same graph, neuronx-cc 2.27 failed ISA validation on
        # the einsum's lowering ("Copy; inst failed assertion check:
        # 'start_addr_active_channels'") at -O1.
        rows_q, heads, dim = local_q.shape
        s = (local_q.reshape(rows_q * heads, dim) @ ik.t()).view(rows_q, heads, -1)
        s = torch.relu(s.float())
        scores = (s * local_w.unsqueeze(-1)).sum(dim=1)  # [C / world, K]
        local = select_topk(scores, local_vis, self.index_topk)
        _capture(self.layer_idx, "index_scores_local", scores)
        _capture(self.layer_idx, "selected_local", local.to(torch.bfloat16))
        if isinstance(rows, slice):
            return local
        # The all-gather's block order is the collective's, which on Trn2 is
        # *not* tp-rank order (the rank -> device mapping is a mesh; see
        # ``_TRN2_MESH``). Assuming it was gave every query some other query's
        # selection -- coherent-looking garbage from the first token. So the
        # ranks are gathered too, and a 0/1 permutation matmul puts block ``j``
        # (from rank ``j``) at rows ``[j * per_rank, (j + 1) * per_rank)``.
        # Exact in bf16: every output element is one 0/1 product.
        #
        # bf16 for the collective, which bool may not be on every backend; back
        # by a cast, not ``> 0.5``: a bf16-vs-Python-float compare lowers to an
        # f64 constant, which neuronx-cc rejects (NCC_ESPP004).
        gathered = _all_gather(self.tp_group, local.to(torch.bfloat16), 0, world)
        if "collectives" in PERF_SKIP:
            sources = torch.arange(world, dtype=torch.float32, device=iq.device)
        else:
            sources = self.tp_group.all_gather(rank.reshape(1).to(torch.float32), dim=0)
        targets = torch.arange(world, dtype=torch.float32, device=iq.device)
        perm = (targets.view(world, 1) == sources.view(1, world)).to(torch.bfloat16)
        ordered = perm @ gathered.reshape(world, -1)
        _capture(self.layer_idx, "gather_sources", sources)
        return ordered.reshape(chunk, -1).to(torch.bool)

    def forward_prefill(
        self,
        x: torch.Tensor,
        position_embeddings: tuple,
        metadata: dict,
        selected: torch.Tensor | None,
        rank: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One chunk of one sequence: the cached prefix, then causal over the chunk.

        The scheduler runs one request per prefill step, and ``cached_seq_len``
        tokens of it are already in the cache -- an earlier chunk, or a prefix
        hit. The prefix is read from the cache; the chunk's own keys come from
        this step's projections, because the ``index_put_`` writing them is not
        visible to a gather in the same graph. Keys are laid out
        ``[cache span (W * block_size) | chunk (C)]``; a cache key is visible iff
        its position is below ``cached_seq_len``.
        """
        cos, sin = position_embeddings
        chunk = x.shape[0]
        x = x.to(self.dtype)
        query, entry, gate, q_resid = self._project(x, cos, sin)
        self._write_kv(entry, metadata)

        prior = metadata["cached_seq_len"].reshape(1, 1).to(torch.long)
        block_table = metadata["block_table_tensor"][:1]
        live = torch.ones(1, 1, dtype=torch.bool, device=x.device)
        ctx = self._gather(block_table, live)[0]  # [ctx_len, 704]
        ctx_len = ctx.shape[0]
        in_prefix = torch.arange(ctx_len, device=x.device).view(1, ctx_len) < prior
        ctx = _zero_beyond(ctx, prior.reshape(1, 1))

        keys = torch.cat((ctx, entry.to(ctx.dtype)), dim=0)  # [K, 704]
        causal = torch.ones(chunk, chunk, dtype=torch.bool, device=x.device).tril()
        visible = torch.cat((in_prefix.expand(chunk, ctx_len), causal), dim=1)

        if self.indexer is not None:
            iq, iw = self.indexer.queries(q_resid, x, cos, sin)
            selected = self._select_prefill(
                iq, iw, keys[:, self.index_slice], visible, rank
            )

        o_lat = self._attend_blocked(query.transpose(0, 1), keys, selected)
        out = self._finish(o_lat.transpose(0, 1), gate)
        _capture(self.layer_idx, "selected", selected.to(torch.bfloat16))
        _capture(self.layer_idx, "attn_out", out)
        return out, selected

    def forward_decode(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple,
        metadata: dict,
        selected: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One new token per request against its paged history.

        Keys are ``[cache span | the token itself]``: the current token's entry is
        appended, since the ``index_put_`` above is not visible to the gather.
        """
        cos, sin = position_embeddings
        block_table = metadata["block_table_tensor"]
        num_reqs = block_table.shape[0]
        x = x.to(self.dtype)
        query, entry, gate, q_resid = self._project(x, cos, sin)
        self._write_kv(entry, metadata)

        live_rows = (metadata["slot_mapping"].view(num_reqs, -1)[:, 0] > 0).view(-1, 1)
        ctx = self._gather(block_table, live_rows)  # [reqs, ctx_len, 704]
        ctx_len = ctx.shape[1]
        key_positions = torch.arange(ctx_len, device=x.device).view(1, ctx_len)
        in_prefix = key_positions < positions.view(num_reqs, 1)  # [reqs, ctx_len]
        ctx = _zero_beyond(ctx, positions.view(num_reqs, 1))

        keys = torch.cat(
            (ctx, entry.to(ctx.dtype).unsqueeze(1)), dim=1
        )  # [reqs, K, 704]
        visible = torch.cat(
            (in_prefix, torch.ones(num_reqs, 1, dtype=torch.bool, device=x.device)),
            dim=1,
        )

        if self.indexer is not None:
            iq, iw = self.indexer.queries(q_resid, x, cos, sin)  # [reqs, 32, 128]
            ik = keys[..., self.index_slice]  # [reqs, K, 128]
            s = torch.relu((iq @ ik.transpose(1, 2)).float())  # [reqs, 32, K]
            index_scores = (s * iw.unsqueeze(-1)).sum(dim=1)
            selected = select_topk(index_scores, visible, self.index_topk)

        lat = keys[..., self.latent_slice]  # [reqs, K, 576]
        q = query.float()  # [reqs, nh, 576]
        scores = (q @ lat.float().transpose(1, 2)) * self.scaling  # [reqs, nh, K]
        scores = scores.masked_fill(~selected.unsqueeze(1), float("-inf"))
        # Heads first for the sink broadcast, then back.
        probs = self._with_sinks(scores.transpose(0, 1)).transpose(0, 1)
        o_lat = probs.to(self.dtype) @ lat[..., : self.kv_lora_rank]  # [reqs, nh, 512]
        out = self._finish(o_lat, gate)
        _capture(self.layer_idx, "attn_in", x)
        _capture(self.layer_idx, "selected", selected.to(torch.bfloat16))
        _capture(self.layer_idx, "attn_out", out)
        return out, selected

    def forward_decode_split(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple,
        metadata: dict,
        selected: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``forward_decode`` with the token's own scores computed apart.

        Same result; it never concatenates the ``[reqs, span, 704]`` keys and does
        the indexer's head-weighted sum as a matmul. Which form neuronx-cc 2.27
        lowers better depends on the batch, measured per full-indexer layer at a
        4K span (attention only, one core): 1 row 1.7 vs 22.5 ms, 16 rows 9.3 vs
        16.8 ms -- but 2/4/8 rows 3.9/6.1/10.5 vs 3.0/2.7/3.8 ms. At 1 row the
        original made the batch-1 graph 3x slower per token than the 4-row one.
        """
        cos, sin = position_embeddings
        block_table = metadata["block_table_tensor"]
        num_reqs = block_table.shape[0]
        x = x.to(self.dtype)
        query, entry, gate, q_resid = self._project(x, cos, sin)
        self._write_kv(entry, metadata)

        live_rows = (metadata["slot_mapping"].view(num_reqs, -1)[:, 0] > 0).view(-1, 1)
        ctx = self._gather(block_table, live_rows)  # [reqs, ctx_len, 704]
        ctx_len = ctx.shape[1]
        key_positions = torch.arange(ctx_len, device=x.device).view(1, ctx_len)
        in_prefix = key_positions < positions.view(num_reqs, 1)  # [reqs, ctx_len]
        ctx = _zero_beyond(ctx, positions.view(num_reqs, 1))
        own = entry.to(ctx.dtype).unsqueeze(1)  # [reqs, 1, 704]
        visible = torch.cat(
            (in_prefix, torch.ones(num_reqs, 1, dtype=torch.bool, device=x.device)),
            dim=1,
        )
        # Scores against the cache and against the token itself are computed
        # separately and concatenated, rather than concatenating the
        # [reqs, 32K, 704] keys (a full copy of the context per layer).

        if self.indexer is not None:
            iq, iw = self.indexer.queries(q_resid, x, cos, sin)  # [reqs, 32, 128]
            s = torch.cat(
                (
                    iq @ ctx[..., self.index_slice].transpose(1, 2),
                    iq @ own[..., self.index_slice].transpose(1, 2),
                ),
                dim=-1,
            )
            s = torch.relu(s.float())  # [reqs, 32, K]
            index_scores = (iw.unsqueeze(1) @ s).squeeze(1)  # [reqs, K]
            selected = select_topk(index_scores, visible, self.index_topk)

        lat = ctx[..., self.latent_slice]  # [reqs, ctx_len, 576]
        own_lat = own[..., self.latent_slice]  # [reqs, 1, 576]
        q = query.float()  # [reqs, nh, 576]
        scores = torch.cat(
            (q @ lat.float().transpose(1, 2), q @ own_lat.float().transpose(1, 2)),
            dim=-1,
        ) * self.scaling  # [reqs, nh, K]
        scores = scores.masked_fill(~selected.unsqueeze(1), float("-inf"))
        # Heads first for the sink broadcast, then back.
        probs = self._with_sinks(scores.transpose(0, 1)).transpose(0, 1).to(self.dtype)
        rank_v = self.kv_lora_rank
        o_lat = (
            probs[..., :ctx_len] @ lat[..., :rank_v]
            + probs[..., ctx_len:] @ own_lat[..., :rank_v]
        )  # [reqs, nh, 512]
        out = self._finish(o_lat, gate)
        _capture(self.layer_idx, "attn_in", x)
        _capture(self.layer_idx, "selected", selected.to(torch.bfloat16))
        _capture(self.layer_idx, "attn_out", out)
        return out, selected


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
        selected: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (streams, DSA key selection for the layers that share it)."""
        residual = streams
        x, post = self.attn_hc(streams)
        if "attn" in PERF_SKIP:
            x = torch.zeros_like(x)
        else:
            x, selected = self.self_attn(
                self.input_layernorm(x),
                positions,
                position_embeddings,
                attn_metadata,
                selected,
                rank,
            )
        streams = hc_post(x, residual, post)

        residual = streams
        x, post = self.ffn_hc(streams)
        x = self.post_attention_layernorm(x)
        _capture(self.layer_idx, "mlp_in", x)
        x = torch.zeros_like(x) if "moe" in PERF_SKIP else self.mlp(x)
        if self.world_size > 1:
            x = _all_reduce(self.tp_group, x)
        _capture(self.layer_idx, "mlp_out", x)
        return hc_post(x, residual, post), selected


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
        selected = None
        for layer in self.layers:
            streams, selected = layer(
                streams, positions, position_embeddings, attn_metadata, selected, rank
            )
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
        if config.indexer_types[i] == "full":
            ia, hi = f"{a}.indexer", f"{ha}.indexer"
            m[f"{ia}.wq_b_weight"] = f"{hi}.wq_b.weight"
            m[f"{ia}.wk_weight"] = f"{hi}.wk.weight"
            m[f"{ia}.k_norm_weight"] = f"{hi}.k_norm.weight"
            m[f"{ia}.k_norm_bias"] = f"{hi}.k_norm.bias"
            m[f"{ia}.weights_proj_weight"] = f"{hi}.weights_proj.weight"

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

    The MTP layer (``model.mtp_layers.0``) is not loaded.
    """

    # Two private pages per buffer (zero page + write sink), as Qwen3.5 uses.
    kv_cache_page_major = True
    # Overridden by models built on this port (glm_dsa).
    text_model_cls = HYV4TextModel
    checkpoint_mappings = staticmethod(checkpoint_mappings)

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

        self.model = self.text_model_cls(config)

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
            self.checkpoint_mappings(self.config),
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
