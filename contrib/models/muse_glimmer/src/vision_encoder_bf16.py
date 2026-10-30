# SPDX-License-Identifier: Apache-2.0
"""MuseGlimmer perception encoder (BF16) for vLLM-Neuron.

HF reference: ``MuseGlimmerVisionModel`` + ``MuseGlimmerVisionAdapter`` +
``vision_projection`` + ``perception_emb_norm`` (``modeling_muse_glimmer.py``).

Pipeline (all tensors block-packed ``[num_blocks, block_size, ...]``)::

    pixel patches (1176 = temporal 2 x RGB x 14 x 14, flattened (t, c, y, x))
      -> patch_embedding (Linear, no bias) + bilinear pos-table (4 taps, CPU indices)
      -> ln_pre
      -> N x [ norm1 -> attn (q/k/v bias, 2D axial RoPE, window|full bounds) -> + ;
               norm2 -> MLP (erf GELU) -> + ]
      -> ln_post
      -> 2x2 pixel shuffle, channel-interleaved (out[c*4 + j] = tok_j[c])
      -> merged_order gather (pack order -> HF merged order)
      -> adapter fc1 -> GELU -> fc2 -> GELU (no biases)
      -> vision_projection (no bias) -> weightless RMSNorm
      -> encoder-cache index_put_

The window reorder of HF is replaced by the CPU pack order (see
``utils/vision_preprocessing.py``): window layers and full layers differ only
in their ``[bound_min, bound_max)`` tensors, selected per layer by the static
``layer_types`` list (no graph branch).

Parallelism (``encoder_neuron_config``):
  * DP (default, tp=1, dp=world): every rank holds all weights, processes
    ``num_blocks / dp`` blocks, then all-gathers the final features.
  * TP: attention heads, MLP intermediate and adapter (fc1 column / fc2 row)
    are sharded over the vision TP group; patch embed, pos table, norms and
    the projection are replicated.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm_neuron.functional as NF
from nkilib.core.utils.common_types import ActFnType, NormType
from vllm_neuron.parallel.neuron_parallel_state import (
    get_neuron_encoder_dp_group,
    get_neuron_encoder_tp_group,
)
from vllm_neuron.utils.dynamo_graph import allow_in_graph_cacheable
from vllm_neuron.utils.weight_loader import set_weight_loader, sharding_weight_loader

from .weight_loaders import transposed_loader, vis_qkv3_bias_loader, vis_qkv3_weight_loader

# attention_cte reads bounds in 128-row tiles.
_ATTN_SEQ_ALIGN = 128


@allow_in_graph_cacheable
def gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact (erf) GELU, traceable by torch.compile on Neuron."""
    return 0.5 * x * (1.0 + torch.erf(x / 1.4142135623730951))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """rotate_half RoPE: concat(-x2, x1)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

WINDOW = "window_attention"
FULL = "full_attention"

HF_VISION_PREFIX = "model.vision_tower"


def vision_compute_dtype() -> torch.dtype:
    """BF16 in production; ``MUSE_GLIMMER_COMPUTE_DTYPE=float32`` is a CPU test knob."""
    name = os.environ.get("MUSE_GLIMMER_COMPUTE_DTYPE", "bfloat16")
    return getattr(torch, name)


# vision encoder knobs (read once at model build; all default to the original encoder graph).
# MUSE_GLIMMER_VIS_OPT       neuronx-cc -O for the encoder graph (per-graph policy), off = engine level.
# MUSE_GLIMMER_VIS_WIN_SPLIT 1: window layers attend in 1024-row sub-blocks ([nb*bs/1024, 1024]); the
#                            CPU packer then keeps every 32x32 window inside one 1024-row chunk.
# MUSE_GLIMMER_VIS_MLP       torch (erf GELU, default) | nki (NF.mlp skip_gate, nl.gelu) | fp32gelu.
# MUSE_GLIMMER_VIS_LN        torch (default) | fused (norm1 into NF.qkv_proj, norm2 into NF.mlp LAYER_NORM).
# MUSE_GLIMMER_VIS_ROPE      fp32 (default, HF) | bf16 (rotation in the compute dtype).
VIS_OPT_ENV = "MUSE_GLIMMER_VIS_OPT"
VIS_WIN_SPLIT_ENV = "MUSE_GLIMMER_VIS_WIN_SPLIT"
VIS_MLP_ENV = "MUSE_GLIMMER_VIS_MLP"
VIS_LN_ENV = "MUSE_GLIMMER_VIS_LN"
VIS_ROPE_ENV = "MUSE_GLIMMER_VIS_ROPE"
WIN_CHUNK = 1024  # 32 x 32 patches: the largest window


def _env_choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    v = os.environ.get(name, default).strip().lower() or default
    if v not in allowed:
        raise ValueError(f"{name}={v!r} not in {allowed}")
    return v


def vision_opt_level() -> int | None:
    v = _env_choice(VIS_OPT_ENV, "off", ("off", "0", "1", "2", "3"))
    return None if v in ("off", "0") else int(v)


def vision_win_split() -> bool:
    return _env_choice(VIS_WIN_SPLIT_ENV, "0", ("0", "1")) == "1"


def vision_mlp_mode() -> str:
    return _env_choice(VIS_MLP_ENV, "torch", ("torch", "nki", "fp32gelu"))


def vision_ln_mode() -> str:
    return _env_choice(VIS_LN_ENV, "torch", ("torch", "fused"))


def vision_rope_mode() -> str:
    return _env_choice(VIS_ROPE_ENV, "fp32", ("fp32", "bf16"))


class MuseGlimmerVisionAttention(nn.Module):
    """Non-causal MHA (16 heads x 96) with q/k/v bias, axial RoPE and bounds."""

    def __init__(self, hidden_size: int, num_heads: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.dtype = dtype
        self.rope_mode = vision_rope_mode()

        # >>> PARALLELISM: heads sharded over the vision TP group <<<
        self.tp_group = get_neuron_encoder_tp_group()
        self.world_size = self.tp_group.world_size
        if num_heads % self.world_size:
            raise ValueError(f"vision heads {num_heads} not divisible by vision TP {self.world_size}")
        self.num_heads_per_rank = num_heads // self.world_size
        hd = self.num_heads_per_rank * self.head_dim

        self.qkv_weight = nn.Parameter(torch.empty(hidden_size, 3 * hd, dtype=dtype))
        self.qkv_bias = nn.Parameter(torch.empty(3 * hd, dtype=dtype))
        self.proj_weight = nn.Parameter(torch.empty(hd, hidden_size, dtype=dtype))
        self.proj_bias = nn.Parameter(torch.empty(hidden_size, dtype=dtype))

        set_weight_loader(self.qkv_weight, vis_qkv3_weight_loader(self.num_heads_per_rank, self.head_dim))
        set_weight_loader(self.qkv_bias, vis_qkv3_bias_loader(self.num_heads_per_rank, self.head_dim))
        set_weight_loader(
            self.proj_weight,
            sharding_weight_loader(
                shard_dim=0, shard_size=hd, num_shards=self.world_size, is_storage_transposed=True
            ),
        )

    def forward(self, x, cos, sin, bound_min, bound_max, ln=None, win=None):
        """``ln``: optional ``(weight, bias, eps)`` LayerNorm fused into the QKV kernel.

        ``win``: optional ``(gather [nb, K*C], scatter [nb, bs], K, C)``: attend in
        ``nb*K`` sub-blocks of ``C`` rows instead of ``nb`` blocks of ``bs`` rows; the
        bounds are then the chunk-local ``[nb*K, C, 1]`` tensors.
        """
        nb, bs, _ = x.shape
        n, d = self.num_heads_per_rank, self.head_dim
        x = x.to(self.dtype)
        if ln is None:
            qkv = NF.qkv_proj(hidden=x, qkv_weights=self.qkv_weight, bias=self.qkv_bias.unsqueeze(0))
        else:
            lw, lb, eps = ln
            qkv = NF.qkv_proj(
                hidden=x, qkv_weights=self.qkv_weight, bias=self.qkv_bias.unsqueeze(0),
                norm_weights=lw.unsqueeze(0), norm_b=lb.unsqueeze(0), norm_type=NormType.LAYER_NORM, eps=eps,
            )
        hd = n * d
        q = qkv[:, :, :hd].reshape(nb, bs, n, d)
        k = qkv[:, :, hd : 2 * hd].reshape(nb, bs, n, d)
        v = qkv[:, :, 2 * hd :].reshape(nb, bs, n, d)

        # <-- MODEL-SPECIFIC: axial 2D RoPE applied in fp32 (HF upcasts q/k).
        # MUSE_GLIMMER_VIS_ROPE=bf16: rotate in the compute dtype instead.
        rd = self.dtype if self.rope_mode == "bf16" else torch.float32
        c = cos.to(rd).unsqueeze(2)
        s = sin.to(rd).unsqueeze(2)
        qf, kf = q.to(rd), k.to(rd)
        q = (qf * c + _rotate_half(qf) * s).to(self.dtype)
        k = (kf * c + _rotate_half(kf) * s).to(self.dtype)

        if win is not None:
            gather, scatter, K, C = win  # flat row indices (block offsets folded in)
            # [nb*bs, n*d] rows -> [nb*K*C] chunk slots (pad slots read a pad row; bounds (0, 0))
            q, k, v = (torch.index_select(t.reshape(nb * bs, n * d), 0, gather) for t in (q, k, v))
            nb_a, bs_a = nb * K, C
        else:
            nb_a, bs_a = nb, bs
        q = q.reshape(nb_a, bs_a, n, d).permute(0, 2, 1, 3).reshape(nb_a * n, bs_a, d)
        k = k.reshape(nb_a, bs_a, n, d).permute(0, 2, 1, 3).reshape(nb_a * n, bs_a, d)
        v = v.reshape(nb_a, bs_a, n, d).permute(0, 2, 1, 3).reshape(nb_a * n, bs_a, d)
        bmin = bound_min.repeat_interleave(n, dim=0)
        bmax = bound_max.repeat_interleave(n, dim=0)
        o = NF.flash_attention(
            q, k, v, scale=self.scale, causal_mask=False, tp_q=True, tp_k=True,
            bound_min=bmin, bound_max=bmax,
        )
        if win is not None:
            # [nb*K*n, C, d] -> chunk-slot rows [nb*K*C, n*d] -> block rows -> [nb, N, D, S]
            o = o.view(nb * K, n, C, d).permute(0, 2, 1, 3).reshape(nb * K * C, n * d)
            o = torch.index_select(o, 0, scatter).reshape(nb, bs, n, d).permute(0, 2, 3, 1)
        else:
            o = o.view(nb, n, bs, d).permute(0, 1, 3, 2)  # [nb, N, D, S]
        out = NF.o_proj(o, self.proj_weight)
        if self.world_size > 1:
            self.tp_group.all_reduce(out)
        return out + self.proj_bias


class MuseGlimmerVisionMLP(nn.Module):
    """fc1 (+bias) -> GELU -> fc2 -> all-reduce -> + bias, with optional variants.

    fc1 is column-sharded and fc2 row-sharded over the vision TP group.

    ``MUSE_GLIMMER_VIS_MLP``:
      * ``torch`` (default): torch matmuls + erf GELU.
      * ``fp32gelu``: the erf GELU in fp32 (observed on another encoder: the bf16 erf form does
        not fuse into the ScalarE GELU activation on neuronx-cc 2.28; the fp32 form does).
      * ``nki``: nkilib MLP kernel, ``skip_gate=True``, ``ActFnType.GELU`` (``nl.gelu``,
        exact erf) with the fc1 bias; fc2 bias stays outside (after the all-reduce).
    ``ln`` (``MUSE_GLIMMER_VIS_LN=fused`` with ``nki``) folds norm2 into the kernel.
    """

    def __init__(self, hidden_size, intermediate_size, dtype, mode: str):
        super().__init__()
        self.tp_group = get_neuron_encoder_tp_group()
        self.world_size = self.tp_group.world_size
        per = intermediate_size // self.world_size
        self.intermediate_size_per_rank = per
        self.fc1_weight = nn.Parameter(torch.empty(hidden_size, per, dtype=dtype))
        self.fc1_bias = nn.Parameter(torch.empty(per, dtype=dtype))
        self.fc2_weight = nn.Parameter(torch.empty(per, hidden_size, dtype=dtype))
        self.fc2_bias = nn.Parameter(torch.empty(hidden_size, dtype=dtype))
        set_weight_loader(self.fc1_weight, sharding_weight_loader(
            shard_dim=1, shard_size=per, num_shards=self.world_size, is_storage_transposed=True))
        set_weight_loader(self.fc1_bias, sharding_weight_loader(
            shard_dim=0, shard_size=per, num_shards=self.world_size))
        set_weight_loader(self.fc2_weight, sharding_weight_loader(
            shard_dim=0, shard_size=per, num_shards=self.world_size, is_storage_transposed=True))
        # nkilib mlp_cte_sharding forces I-sharding at intermediate_size >= 4096, and I-sharding
        # drops projection biases: wrong output with the fc1 bias (simulator, H=1024 I=4096: cos
        # 0.90). Vision TP=4 has I/rank = 2240; TP<=2 falls back to torch.
        if mode == "nki" and self.intermediate_size_per_rank >= 4096:
            mode = "torch"
        self.mode = mode

    def forward(self, hidden_states, ln=None):
        if self.mode == "nki":
            shp = hidden_states.shape
            h2 = hidden_states.reshape(-1, shp[-1])
            kw = {}
            if ln is not None:
                lw, lb, eps = ln
                kw = dict(ln_w=lw.unsqueeze(0), norm_b=lb.unsqueeze(0), norm_type=NormType.LAYER_NORM, eps=eps)
            out = NF.mlp(h2, self.fc1_weight, self.fc1_weight, self.fc2_weight, skip_gate=True,
                         act_fn=ActFnType.GELU, up_b=self.fc1_bias.unsqueeze(0), **kw)
            out = out.reshape(shp)
        else:
            if ln is not None:
                lw, lb, eps = ln
                hidden_states = F.layer_norm(hidden_states, (hidden_states.shape[-1],), lw, lb, eps)
            out = hidden_states @ self.fc1_weight + self.fc1_bias
            if self.mode == "fp32gelu":
                out = gelu(out.to(torch.float32)).to(hidden_states.dtype)
            else:
                out = gelu(out)
            out = out @ self.fc2_weight
        if self.world_size > 1:
            self.tp_group.all_reduce(out)
        return out + self.fc2_bias


class MuseGlimmerVisionBlock(nn.Module):
    def __init__(self, vc, layer_type: str, dtype: torch.dtype) -> None:
        super().__init__()
        if layer_type not in (WINDOW, FULL):
            raise ValueError(f"unknown vision layer_type {layer_type!r}")
        self.layer_type = layer_type
        # HF hard-codes eps 1e-5 for norm1/norm2 (not read from config).
        self.norm1 = nn.LayerNorm(vc.hidden_size, eps=1e-5, dtype=dtype)
        self.norm2 = nn.LayerNorm(vc.hidden_size, eps=1e-5, dtype=dtype)
        self.attn = MuseGlimmerVisionAttention(vc.hidden_size, vc.num_attention_heads, dtype)
        self.mlp = MuseGlimmerVisionMLP(vc.hidden_size, vc.intermediate_size, dtype, vision_mlp_mode())
        # LN fusion: norm1 -> NF.qkv_proj LAYER_NORM; norm2 -> NF.mlp LAYER_NORM (nki MLP only).
        self.fuse_ln = vision_ln_mode() == "fused"

    def _ln(self, norm):
        return (norm.weight, norm.bias, norm.eps)

    def forward(self, x, cos, sin, full_bounds, window_bounds, win=None):
        bmin, bmax = window_bounds if self.layer_type == WINDOW else full_bounds
        w = win if self.layer_type == WINDOW else None
        if self.fuse_ln:
            x = x + self.attn(x, cos, sin, bmin, bmax, ln=self._ln(self.norm1), win=w)
            if self.mlp.mode == "nki":
                return x + self.mlp(x, ln=self._ln(self.norm2))
            return x + self.mlp(self.norm2(x))
        x = x + self.attn(self.norm1(x), cos, sin, bmin, bmax, win=w)
        return x + self.mlp(self.norm2(x))


class MuseGlimmerVisionAdapter(nn.Module):
    """fc1 (no bias) -> GELU -> fc2 (no bias) -> GELU; fc1 column / fc2 row TP."""

    def __init__(self, in_dim: int, hidden: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.tp_group = get_neuron_encoder_tp_group()
        self.world_size = self.tp_group.world_size
        if hidden % self.world_size:
            raise ValueError(f"projector_hidden_size {hidden} not divisible by vision TP")
        per = hidden // self.world_size
        self.fc1_weight = nn.Parameter(torch.empty(in_dim, per, dtype=dtype))
        self.fc2_weight = nn.Parameter(torch.empty(per, hidden, dtype=dtype))
        set_weight_loader(
            self.fc1_weight,
            sharding_weight_loader(shard_dim=1, shard_size=per, num_shards=self.world_size,
                                   is_storage_transposed=True),
        )
        set_weight_loader(
            self.fc2_weight,
            sharding_weight_loader(shard_dim=0, shard_size=per, num_shards=self.world_size,
                                   is_storage_transposed=True),
        )

    def forward(self, x):
        x = gelu(x @ self.fc1_weight)
        x = x @ self.fc2_weight
        if self.world_size > 1:
            self.tp_group.all_reduce(x)
        return gelu(x)


class MuseGlimmerVisionPosEmbed(nn.Module):
    """4-tap bilinear blend of the learned 32x32 table (indices / weights from CPU)."""

    def __init__(self, num_positions: int, hidden: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_positions, hidden, dtype=dtype))

    def forward(self, idx, wt):
        emb = F.embedding(idx.long(), self.weight)  # [4, nb, bs, H]
        # HF multiplies the table rows by fp32 weights and sums in fp32.
        return (emb.to(torch.float32) * wt.to(torch.float32).unsqueeze(-1)).sum(0)


class MuseGlimmerVisionModel(nn.Module):
    """Perception encoder + adapter + projection + norm, writing into the encoder cache."""

    def __init__(self, config, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        vc = config.vision_config
        tc = config.text_config
        self.config = config
        self.vc = vc
        self.dtype = dtype or vision_compute_dtype()
        dt = self.dtype
        if not vc.layer_types or len(vc.layer_types) != vc.num_hidden_layers:
            raise ValueError("vision layer_types must list every layer")
        if vc.hidden_size % vc.num_attention_heads:
            raise ValueError("vision hidden_size must be divisible by heads")
        self.merge = vc.merge_size
        self.merge_unit = vc.merge_size**2
        self.hidden = vc.hidden_size
        self.patch_dim = vc.patch_temporal * 3 * vc.patch_size**2
        if config.out_hidden_size != self.hidden * self.merge_unit:
            raise ValueError(
                f"out_hidden_size {config.out_hidden_size} != vision hidden "
                f"{self.hidden} x merge {self.merge_unit}"
            )
        self.text_hidden = tc.hidden_size
        self.rms_eps = float(tc.rms_norm_eps)

        # >>> PARALLELISM: vision DP group (block scatter / gather) <<<
        self.dp_group = get_neuron_encoder_dp_group()
        self.dp_size = self.dp_group.world_size if self.dp_group else 1

        self.patch_embedding_weight = nn.Parameter(torch.empty(self.patch_dim, self.hidden, dtype=dt))
        set_weight_loader(self.patch_embedding_weight, transposed_loader())
        self.pos_embed = MuseGlimmerVisionPosEmbed(vc.pos_emb_height * vc.pos_emb_width, self.hidden, dt)
        self.ln_pre = nn.LayerNorm(self.hidden, eps=vc.layer_norm_eps, dtype=dt)
        self.layers = nn.ModuleList(
            [MuseGlimmerVisionBlock(vc, lt, dt) for lt in vc.layer_types]
        )
        self.ln_post = nn.LayerNorm(self.hidden, eps=vc.layer_norm_eps, dtype=dt)
        self.adapter = MuseGlimmerVisionAdapter(config.out_hidden_size, config.projector_hidden_size, dt)
        self.projection_weight = nn.Parameter(
            torch.empty(config.projector_hidden_size, self.text_hidden, dtype=dt)
        )
        set_weight_loader(self.projection_weight, transposed_loader())

    # ------------------------------------------------------------------
    def encode(
        self,
        pixel_values: torch.Tensor,
        pos_emb_idx: torch.Tensor,
        pos_emb_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
        window_bound_min: torch.Tensor,
        window_bound_max: torch.Tensor,
        merged_order: torch.Tensor,
        win_gather: torch.Tensor | None = None,
        win_scatter: torch.Tensor | None = None,
        win_bound_min: torch.Tensor | None = None,
        win_bound_max: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Local blocks -> features ``[nb, bs / merge_unit, text_hidden]`` (HF merged order).

        ``win_*`` ( window split): window layers attend in chunks; see
        ``pack_vision_inputs(win_chunks=...)``. Requires ``bs % 128 == 0`` (no in-graph pad).
        """
        nb, real_bs, _ = pixel_values.shape
        mu = self.merge_unit
        real_merged = real_bs // mu
        padded = -(-real_bs // _ATTN_SEQ_ALIGN) * _ATTN_SEQ_ALIGN
        pad = padded - real_bs
        if pad:
            # Pad rows: bounds (0, 0) -> attend to nothing; nothing attends them.
            pixel_values = F.pad(pixel_values, (0, 0, 0, pad))
            cos = F.pad(cos, (0, 0, 0, pad))
            sin = F.pad(sin, (0, 0, 0, pad))
            bound_min = F.pad(bound_min, (0, 0, 0, pad))
            bound_max = F.pad(bound_max, (0, 0, 0, pad))
            window_bound_min = F.pad(window_bound_min, (0, 0, 0, pad))
            window_bound_max = F.pad(window_bound_max, (0, 0, 0, pad))
            pos_emb_idx = F.pad(pos_emb_idx, (0, pad))
            pos_emb_weight = F.pad(pos_emb_weight, (0, pad))

        h = pixel_values.to(self.dtype) @ self.patch_embedding_weight
        h = h + self.pos_embed(pos_emb_idx, pos_emb_weight).to(self.dtype)
        h = self.ln_pre(h)
        full_b = (bound_min, bound_max)
        win_b = (window_bound_min, window_bound_max)
        win = None
        if win_gather is not None:
            if pad:
                raise ValueError(f"window split needs block rows % {_ATTN_SEQ_ALIGN} == 0, got {real_bs}")
            K = win_gather.shape[1] // WIN_CHUNK
            ob = torch.arange(nb, dtype=win_gather.dtype, device=win_gather.device).unsqueeze(1)
            g = (win_gather + ob * real_bs).reshape(-1)
            sc = (win_scatter + ob * (K * WIN_CHUNK)).reshape(-1)
            win = (g, sc, K, WIN_CHUNK)
            win_b = (win_bound_min, win_bound_max)
        for layer in self.layers:
            h = layer(h, cos, sin, full_b, win_b, win=win)
        h = self.ln_post(h)

        # <-- MODEL-SPECIFIC: channel-interleaved 2x2 shuffle (pack order makes it a reshape).
        h = h[:, :real_bs]
        h = h.reshape(nb, real_merged, mu, self.hidden).transpose(-1, -2).reshape(
            nb, real_merged, mu * self.hidden
        )
        # pack (window-major) merged order -> HF merged order, per block.
        offs = torch.arange(nb, dtype=merged_order.dtype, device=merged_order.device).unsqueeze(1)
        flat_idx = (merged_order + offs * real_merged).reshape(-1)
        h = torch.index_select(h.reshape(nb * real_merged, -1), 0, flat_idx).reshape(nb, real_merged, -1)

        h = self.adapter(h)
        h = h @ self.projection_weight
        hf = h.to(torch.float32)
        hf = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + self.rms_eps)
        return hf.to(self.dtype)

    def forward(
        self,
        pixel_values: torch.Tensor,
        pos_emb_idx: torch.Tensor,
        pos_emb_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
        window_bound_min: torch.Tensor,
        window_bound_max: torch.Tensor,
        merged_order: torch.Tensor,
        encoder_cache_buffer: torch.Tensor,
        write_block_ids: torch.Tensor,
        win_gather: torch.Tensor | None = None,
        win_scatter: torch.Tensor | None = None,
        win_bound_min: torch.Tensor | None = None,
        win_bound_max: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # >>> PARALLELISM: DP scatter (each rank takes its slice of blocks) <<<
        if self.dp_size > 1:
            if win_gather is not None:
                nb0 = pixel_values.shape[0]
                per0 = nb0 // self.dp_size
                K = win_gather.shape[1] // WIN_CHUNK
                s0 = self.dp_group.rank_in_group * per0
                win_gather, win_scatter = win_gather[s0 : s0 + per0], win_scatter[s0 : s0 + per0]
                win_bound_min = win_bound_min[s0 * K : (s0 + per0) * K]
                win_bound_max = win_bound_max[s0 * K : (s0 + per0) * K]
            nb = pixel_values.shape[0]
            assert nb % self.dp_size == 0, f"num_blocks {nb} not divisible by vision dp {self.dp_size}"
            per = nb // self.dp_size
            s = self.dp_group.rank_in_group * per
            e = s + per
            pixel_values = pixel_values[s:e]
            pos_emb_idx = pos_emb_idx[:, s:e]
            pos_emb_weight = pos_emb_weight[:, s:e]
            cos, sin = cos[s:e], sin[s:e]
            bound_min, bound_max = bound_min[s:e], bound_max[s:e]
            window_bound_min, window_bound_max = window_bound_min[s:e], window_bound_max[s:e]
            merged_order = merged_order[s:e]

        feats = self.encode(
            pixel_values, pos_emb_idx, pos_emb_weight, cos, sin, bound_min, bound_max,
            window_bound_min, window_bound_max, merged_order,
            win_gather, win_scatter, win_bound_min, win_bound_max,
        )
        # >>> PARALLELISM: DP gather <<<
        if self.dp_size > 1:
            feats = self.dp_group.all_gather(feats, dim=0)
        # a block of fewer rows than the cache block fills its head.
        cache_rows = encoder_cache_buffer.shape[1]
        if feats.shape[1] < cache_rows:
            feats = F.pad(feats, (0, 0, 0, cache_rows - feats.shape[1]))
        encoder_cache_buffer.index_put_((write_block_ids,), feats.to(encoder_cache_buffer.dtype))
        return encoder_cache_buffer

    # ------------------------------------------------------------------
    def build_weight_mappings(self) -> dict[str, str | list[str]]:
        p = HF_VISION_PREFIX
        m: dict[str, str | list[str]] = {
            "patch_embedding_weight": f"{p}.patch_embedder.patch_embedding.weight",
            "pos_embed.weight": f"{p}.patch_embedder.position_embedding_table.weight",
            "ln_pre.weight": f"{p}.ln_pre.weight",
            "ln_pre.bias": f"{p}.ln_pre.bias",
            "ln_post.weight": f"{p}.ln_post.weight",
            "ln_post.bias": f"{p}.ln_post.bias",
            "adapter.fc1_weight": "model.vision_adapter.fc1.weight",
            "adapter.fc2_weight": "model.vision_adapter.fc2.weight",
            "projection_weight": "model.vision_projection.weight",
        }
        for i in range(len(self.layers)):
            o, c = f"layers.{i}", f"{p}.layers.{i}"
            m[f"{o}.attn.qkv_weight"] = [f"{c}.attn.{x}_proj.weight" for x in "qkv"]
            m[f"{o}.attn.qkv_bias"] = [f"{c}.attn.{x}_proj.bias" for x in "qkv"]
            m[f"{o}.attn.proj_weight"] = f"{c}.attn.proj.weight"
            m[f"{o}.attn.proj_bias"] = f"{c}.attn.proj.bias"
            m[f"{o}.mlp.fc1_weight"] = f"{c}.mlp.fc1.weight"
            m[f"{o}.mlp.fc1_bias"] = f"{c}.mlp.fc1.bias"
            m[f"{o}.mlp.fc2_weight"] = f"{c}.mlp.fc2.weight"
            m[f"{o}.mlp.fc2_bias"] = f"{c}.mlp.fc2.bias"
            for nrm in ("norm1", "norm2"):
                m[f"{o}.{nrm}.weight"] = f"{c}.{nrm}.weight"
                m[f"{o}.{nrm}.bias"] = f"{c}.{nrm}.bias"
        return m

    def load_weights(self, checkpoint_path: str, device="cpu", cache_dir: str | None = None) -> None:
        from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

        tp = get_neuron_encoder_tp_group()
        mappings = self.build_weight_mappings()
        params = dict(self.named_parameters())
        unmapped = sorted(set(params) - set(mappings))
        if unmapped:
            raise RuntimeError(f"MuseGlimmer vision: parameters without mapping: {unmapped}")
        ckpt = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        sd = ckpt.load_sharded(
            rank=tp.rank_in_group, world_size=tp.world_size, model=self,
            mappings=mappings, device=device, strict=True,
        ).state_dict
        for name, t in sd.items():
            if t.dtype != params[name].dtype:
                sd[name] = t.to(params[name].dtype)
            if tuple(sd[name].shape) != tuple(params[name].shape):
                raise RuntimeError(
                    f"MuseGlimmer vision: {name} loaded {tuple(sd[name].shape)}, "
                    f"expected {tuple(params[name].shape)}"
                )
        self.load_state_dict(sd, strict=False, assign=True)
