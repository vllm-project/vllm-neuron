# SPDX-License-Identifier: Apache-2.0
"""
MuseGlimmer Text Decoder (BF16)
===============================

Text decoder of ``MuseGlimmerForConditionalGeneration`` for vLLM-Neuron.

Architecture (per layer, see HF ``transformers`` ``models/muse_glimmer/modeling_muse_glimmer.py``):

    residual = x
    x = input_layernorm(x)                        # centered RMSNorm, (1 + w), eps 1e-5
    x = attention(x)                              # gated, QK-normed, SWA+RoPE or full+NoPE
    x = post_attention_layernorm(x)               # centered RMSNorm, eps 1e-8
    x = residual + x
    residual = x
    x = pre_feedforward_layernorm(x)              # centered RMSNorm, eps 1e-5
    x = mlp(x)                                    # SwiGLU
    x = post_feedforward_layernorm(x)             # centered RMSNorm, eps 1e-8
    x = residual + x

Attention:
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    q, k = rmsnorm(q), rmsnorm(k)                 # weightless, per head, eps 1e-5
    q = q * qk_scale_factor                       # folded into the softmax scale here
    q, k = rope(q, k)  (sliding layers only; full layers are NoPE)
    o = softmax(q k^T / sqrt(d)) v                # sliding window on sliding layers
    o = o * sigmoid(x @ Wgate)                    # output gate, before o_proj
    out = o @ Wo

Backbone: embed -> weightless RMSNorm (embed_norm) -> [vision scatter] ->
layers -> weighted RMSNorm. Logits: ``cap * tanh(lm_head(x) * mult / cap)``.

The perception encoder lives in ``vision_encoder_bf16.py`` (``self.visual``);
``embed_multimodal`` packs images on CPU and writes the projected features into
the on-device encoder cache. ``mm_language_model_only`` skips the tower.
"""

from __future__ import annotations

import logging
import math
import os
import time as _time

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.models.interfaces import SupportsEagle3

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from nkilib.core.utils.common_types import NormType
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig, EncoderNeuronConfig
from vllm_neuron.functional.merge_multimodal_embeddings import (
    merge_multimodal_embeddings,
)
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel
from vllm_neuron.vllm.spec_decode.decorator import async_speculative_decoding
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
    sharding_weight_loader_with_padding,
)

from . import decode_fusions as _df
from .config import MuseGlimmerConfig, MuseGlimmerTextConfig
from .utils.decode_kv import build_per_head_block_table
from .utils.vision_block_packing import ffd_pack_images, select_vision_bucket
from .utils.vision_preprocessing import pack_vision_inputs, plan_video_items
from .vision_encoder_bf16 import (
    WIN_CHUNK,
    MuseGlimmerVisionModel,
    vision_opt_level,
    vision_win_split,
)
from .utils.vision_preprocessing import max_window_chunks

logger = logging.getLogger(__name__)

# Escape hatch: route decode attention through the plain-torch reference
# (``_torch_attention_decode_impl``) instead of the fused NKI megakernel.
DECODE_TORCH_ENV = "MUSE_GLIMMER_DECODE_TORCH"


def _decode_torch_forced() -> bool:
    return os.environ.get(DECODE_TORCH_ENV, "0").strip().lower() in ("1", "true", "yes")


# decode KV-write mode.
#   0 = per-layer index_put_ inside the layer loop (old behaviour);
#   1 = defer: the same per-layer index_put_ calls, issued after the layer loop;
#   2 = defer + coalesce: one index_put_ per physical (k, v) buffer (default).
# Why: layers share hybrid KV buffers (``kv_cache_tensors[*].shared_by``). The
# FX pass rewrites in-place index_put_ to functional form, and with N layers on
# one buffer only the last result of the chain is aliasable, so the N-1
# intermediates become full-buffer HBM copies when other ops sit between the
# writes (1.31 GB per step measured at batch 13).
KV_DEFER_ENV = "MUSE_GLIMMER_KV_DEFER"


# neuronx-cc -O level for the plain decode graph only (runner per-graph policy,
# ``compiler_args_for``). Default 3: -O3 disables neuronx-cc modular flow (as Qwen3-VL's
# decode policy) and on trn2 / SDK 2.33 cuts the decode step 10-13 % at BS >= 32, is
# neutral at BS <= 16, and gives the same greedy tokens and logits as -O2. "off" (or "0") keeps the engine level for every graph (0.29 native
# route: -O2). Prefill, the encoder and spec-verify graphs keep the engine level. An
# explicit --optimization-level disables per-graph policies in the runner.
DECODE_OPT_ENV = "MUSE_GLIMMER_DECODE_OPT"


def _decode_opt_level() -> int | None:
    v = os.environ.get(DECODE_OPT_ENV, "3").strip().lower() or "3"
    if v in ("off", "0"):
        return None
    if v not in ("1", "2", "3"):
        raise ValueError(f"{DECODE_OPT_ENV}={v!r} not in (off, 1, 2, 3)")
    return int(v)


# vision encoder graph shape knobs (see vision_encoder_bf16.py for the module knobs).
VIS_SMALL_BLOCK_ENV = "MUSE_GLIMMER_VIS_SMALL_BLOCK"
_VIS_TIMING_FILE = os.environ.get("MUSE_GLIMMER_VIS_TIMING_FILE", "")


def _vis_timing() -> bool:
    """host/device split of each VE call (adds two device syncs; off by default).

    On when ``MUSE_GLIMMER_VIS_TIMING=1``, or when ``$MUSE_GLIMMER_VIS_TIMING_FILE.on``
    exists (toggle at run time without restarting a server).
    """
    if os.environ.get("MUSE_GLIMMER_VIS_TIMING", "0") == "1":
        return True
    return bool(_VIS_TIMING_FILE) and os.path.exists(_VIS_TIMING_FILE + ".on")


def _vis_small_block(vnc) -> bool:
    """Buckets below the block size compile ONE block of ``bucket`` rows (block = image size)."""
    on = os.environ.get(VIS_SMALL_BLOCK_ENV, "0").strip() == "1"
    return on and vnc.dp_size == 1


def _graph_block_rows(bucket: int, vnc) -> tuple[int, int]:
    """(rows per block, num_blocks) of the encoder graph for ``bucket``."""
    bs, dp = vnc.encoder_attention_block_size, vnc.dp_size
    if bucket < bs and _vis_small_block(vnc):
        if bucket % 128:
            raise ValueError(f"{VIS_SMALL_BLOCK_ENV}: bucket {bucket} must be a multiple of 128")
        return bucket, 1
    return bs, math.ceil(math.ceil(bucket / bs) / dp) * dp


def _vis_win_chunks(rows: int) -> int:
    """Window-split chunk count for a block of ``rows`` (0 = no split)."""
    if not vision_win_split() or rows <= WIN_CHUNK:
        return 0
    return max_window_chunks(rows, WIN_CHUNK)


def _kv_defer_mode() -> int:
    v = os.environ.get(KV_DEFER_ENV, "2").strip()
    if v not in ("0", "1", "2"):
        raise ValueError(f"{KV_DEFER_ENV}={v!r} not in (0, 1, 2)")
    return int(v)


def _resolve_kv_ptr_groups(layers) -> list[int] | None:
    """``layer_idx -> group id`` keyed on the (k, v) ``data_ptr()`` pair.

    NOT traced: called from ``bind_kv_cache`` only. ``.data_ptr()`` inside the
    trace aborts Dynamo, and the per-layer cache attributes are distinct view
    objects over shared storage, so ``is`` cannot group them. Returns None
    (-> per-layer deferred writes) if a pointer cannot be read.
    """
    try:
        ptr_to_gid: dict[tuple, int] = {}
        group_of: list[int] = []
        for layer in layers:
            attn = layer.self_attn
            if attn.k_cache is None or attn.v_cache is None:
                return None
            # Shapes are in the key too: the coalesced write indexes the first
            # member's view, so all members must view the buffer identically.
            key = (int(attn.k_cache.data_ptr()), int(attn.v_cache.data_ptr()),
                   tuple(attn.k_cache.shape), tuple(attn.v_cache.shape))
            group_of.append(ptr_to_gid.setdefault(key, len(ptr_to_gid)))
        return group_of
    except Exception:  # pragma: no cover - defensive
        logger.warning(
            "MuseGlimmer: could not resolve KV data_ptr groups; using per-layer "
            "deferred KV writes.", exc_info=True,
        )
        return None


def _issue_decode_kv_writes(writeback: list, group_of: list[int] | None) -> None:
    """Issue the deferred decode KV writes after the layer loop.

    ``writeback`` entries: ``(layer_idx, attn, k_new, v_new, slot_mapping,
    block_size)`` with ``k_new``/``v_new`` ``[nkh, B*S, d]``.

    Byte-identity of the deferral: decode attention runs ``update_cache=False``.
    The kernel (and the torch reference) attends over the PRIOR cache, with the
    prior mask ``iota < pos`` excluding the token's own slot, plus the active
    K/V it holds on chip (the torch reference overlays it into the last S_tkg
    columns). No layer reads the slot written this step, and another layer on the
    same buffer reads only its own KV group's blocks, so moving the writes after
    the loop leaves every attention input unchanged.

    ``group_of is None`` -> the same per-layer index_put_ calls (mode 1).
    Otherwise -> one index_put_ per physical buffer (mode 2). Concatenation is
    valid only for disjoint slots. The members of one buffer come from different
    KV groups (``shared_by`` takes one layer per group), each with its own
    ``slot_mapping`` and its own blocks, so real slots never collide. The
    ``slot_mapping`` object-identity check below (free at trace time) keeps two
    layers of one KV group out of the same concatenation. Decode padding rows
    use NULL_BLOCK_ID (0) -> slot 0 in the reserved null block, which is never
    attended unmasked; duplicate pad writes there are harmless, as they already
    are within one layer in mode 0.
    """
    if group_of is None:
        for _li, attn, k, v, slot, bsz in writeback:
            attn._write_paged_kv(k, v, slot, bsz)
        return

    groups: dict = {}
    order: list = []
    for item in writeback:
        li, attn, _k, _v, slot, _bsz = item
        gid = group_of[li] if li < len(group_of) else ("solo", li)
        g = groups.get(gid)
        if g is not None and any(slot is s for s in g[1]):
            gid = ("solo", li)  # same slot source: do not concatenate
            g = None
        if g is None:
            g = groups[gid] = [[], []]
            order.append(gid)
        g[0].append(item)
        g[1].append(slot)

    for gid in order:
        items = groups[gid][0]
        if len(items) == 1:
            _li, attn, k, v, slot, bsz = items[0]
            attn._write_paged_kv(k, v, slot, bsz)
            continue
        attn0 = items[0][1]
        idx = [attn._paged_kv_indices(slot, bsz, k.device) for _li, attn, k, _v, slot, bsz in items]
        b = torch.cat([i[0] for i in idx])
        h = torch.cat([i[1] for i in idx])
        p = torch.cat([i[2] for i in idx])
        d = attn0.head_dim
        kc, vc = attn0.k_cache, attn0.v_cache
        kc.index_put_((b, h, p), torch.cat([it[2].reshape(-1, d) for it in items]))
        vc.index_put_((b, h, p), torch.cat([it[3].reshape(-1, d) for it in items]))


# Decode attention tiles s_prior into P_MAX partitions per program; at LNC=2 it shards
# s_prior over two programs, so s_prior must be a multiple of LNC * P_MAX. The runner's
# trimmed sliding-window table is only P_MAX-aligned (window 2048, block 32 -> 68 blocks
# = 2176 tokens), which the kernel rejects at LNC=2 ("Sharded s_prior must be divisible
# by p_max") and the torch reference cannot mask either. The kernel also re-tiles the
# cache into gcd(block_size, s_prior / (programs * P_MAX))-token DMA chunks, so a
# power-of-two s_prior keeps the chunks large (4096 -> 16 tokens; 2304 -> 1 token).
_P_MAX = 128


def pad_swa_decode_table(table: torch.Tensor, block_size: int, lnc: int) -> torch.Tensor:
    """Pad a trimmed sliding-window decode block table to a power-of-two token count.

    The target is the next power of two at or above the table's token count, and at
    least ``lnc * 128``. The pad columns are ``-1`` (inactive): the kernel skips them
    (``oob_mode.skip``) and, for single-token decode steps, the mask never reaches them,
    because the trimmed-frame positions stay below the original width. The shapes are
    static, so warmup and runtime pad identically.
    """
    nb = table.shape[-1]
    tokens = nb * block_size
    target = max(lnc * _P_MAX, 1 << max(tokens - 1, 0).bit_length())
    if target == tokens or target % block_size:
        return table
    pad = torch.full((*table.shape[:-1], target // block_size - nb), -1, dtype=table.dtype,
                     device=table.device)
    return torch.cat([table, pad], dim=-1)


# ---------------------------------------------------------------------------
# Section 1: Normalization
# ---------------------------------------------------------------------------


class MuseGlimmerRMSNorm(nn.Module):
    """Standard RMSNorm (optionally weightless), computed in fp32.

    HF: ``x * pow(mean(x^2) + eps, -0.5) [* w]`` then cast back. rsqrt and
    pow(-0.5) agree to fp32 rounding.
    """

    def __init__(self, dim: int | None, eps: float, dtype: torch.dtype, with_scale: bool):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        xf = x.to(torch.float32)
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.with_scale:
            xf = xf * self.weight.to(torch.float32)
        return xf.to(in_dtype)


class MuseGlimmerCenteredRMSNorm(nn.Module):
    """Centered RMSNorm: ``x * rsqrt(mean(x^2) + eps) * (1 + w)``.

    ``w`` is stored as an **fp32** parameter and ``1 + w`` is formed in fp32,
    exactly like HF. Folding ``+1`` into a bf16 weight would round ``1 + w`` to
    8 mantissa bits.
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        xf = x.to(torch.float32)
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        xf = xf * (1.0 + self.weight)
        return xf.to(in_dtype)


# ---------------------------------------------------------------------------
# Section 2: RoPE (plain 1D, rotate_half, full head_dim; sliding layers only)
# ---------------------------------------------------------------------------


class MuseGlimmerRotaryEmbedding(nn.Module):
    """Default RoPE. inv_freq is computed lazily (meta-tensor rule)."""

    def __init__(self, config: MuseGlimmerTextConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(cos, sin)`` each ``[T, head_dim // 2]`` in fp32."""
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(
                    0, self.head_dim, 2, dtype=torch.float32, device=positions.device
                )
                / self.head_dim
            )
        )
        freqs = positions.reshape(-1).to(torch.float32)[:, None] * inv_freq[None, :]
        return freqs.cos(), freqs.sin()


# ---------------------------------------------------------------------------
# Section 3: Attention
# ---------------------------------------------------------------------------


class MuseGlimmerAttention(nn.Module):
    """Gated GQA attention with weightless QK-norm.

    Parallelism (kept from qwen3_vl / gpt_oss):
    - Q heads and the gate are column-sharded; o_proj is row-sharded.
    - KV heads < TP -> replicated (``num_kv_replicas``); rank r owns KV head
      ``r // num_kv_replicas``, which matches HF's repeat_kv grouping.
    - Prefill: SP all-gather -> attention -> reduce-scatter.
    - Decode: fused megakernel (W_out=None) -> gate -> o_proj -> all-reduce.
    """

    def __init__(self, config: MuseGlimmerTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_name = f"layers.{layer_idx}.self_attn"
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.qk_norm_eps = config.rms_norm_eps
        # <-- MODEL-SPECIFIC: Q multiplier folded into the softmax scale.
        self.scaling = config.attention_scale
        # <-- MODEL-SPECIFIC: sliding (RoPE) vs full (NoPE) per layer.
        self.is_sliding = config.is_sliding(layer_idx)
        self.apply_rope = config.uses_rope(layer_idx)
        self.sliding_window = config.sliding_window if self.is_sliding else None

        # >>> PARALLELISM: TP group + head sharding <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        if self.num_attention_heads % self.world_size != 0:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} not divisible by "
                f"TP={self.world_size}"
            )
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
        if self.world_size >= self.num_key_value_heads:
            if self.world_size % self.num_key_value_heads != 0:
                raise ValueError(
                    f"TP={self.world_size} not a multiple of num_kv_heads="
                    f"{self.num_key_value_heads}"
                )
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            if self.num_key_value_heads % self.world_size != 0:
                raise ValueError(
                    f"num_kv_heads={self.num_key_value_heads} not divisible by "
                    f"TP={self.world_size}"
                )
            self.num_key_value_heads_per_rank = self.num_key_value_heads // self.world_size
            self.num_kv_replicas = 1
        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
        )

        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim
        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, q_size + 2 * kv_size, dtype=self.dtype)
        )
        # <-- MODEL-SPECIFIC: output gate, sharded exactly like Q.
        self.gate_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, q_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(q_size, self.hidden_size, dtype=self.dtype)
        )

        # decode fusion flags (read once; MUSE_GLIMMER_DECODE_FUSE).
        self.fuse = _df.decode_fuse_flags()
        self.k_cache = None
        self.v_cache = None
        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        loaders = {
            "qkv_proj": fused_qkv_weight_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
                shard_dim=1,
                num_shards=self.world_size,
                is_storage_transposed=True,
                num_kv_replicas=self.num_kv_replicas,
            ),
            "gate_proj": sharding_weight_loader(
                shard_dim=1,
                shard_size=self.q_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
            "o_proj": sharding_weight_loader(
                shard_dim=0,
                shard_size=self.q_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        }
        for name, ld in loaders.items():
            set_weight_loader(getattr(self, f"{name}_weight"), ld)

    def _gate(self, x: torch.Tensor) -> torch.Tensor:
        """sigmoid(x @ Wgate), x [T, H] -> [T, q_size]."""
        return torch.sigmoid(x @ self.gate_proj_weight)

    # ── dispatch ────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attn_metadata: dict,
        kv_writeback: list | None = None,
        in_norm: nn.Module | None = None,
    ) -> torch.Tensor:
        """``in_norm``: the input norm was NOT applied by the caller; fold it into the kernel."""
        meta = attn_metadata[self.layer_name]
        if meta["max_query_len"] <= meta["decode_token_threshold"]:
            return self.forward_decode(
                hidden_states, positions, position_embeddings, attn_metadata, kv_writeback, in_norm
            )
        if in_norm is not None:
            hidden_states = in_norm(hidden_states)
        # >>> PARALLELISM: SP all-gather before attention <<<
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(hidden_states, position_embeddings, attn_metadata)

    def _paged_kv_indices(self, slot_mapping, block_size, device):
        """(block_idx, head_idx, pos_idx) for ``[nkh, T, d]`` values flattened to ``[nkh*T, d]``."""
        nkh = self.num_key_value_heads_per_rank
        n = slot_mapping.shape[0]
        block_idx = (slot_mapping // block_size).repeat(nkh)
        pos_idx = (slot_mapping % block_size).repeat(nkh)
        head_idx = torch.arange(nkh, dtype=torch.long, device=device).repeat_interleave(n)
        return block_idx, head_idx, pos_idx

    def _write_paged_kv(self, k, v, slot_mapping, block_size):
        """k, v: [nkh, T, d] -> 4D paged cache [blocks, nkh, block, d]."""
        idx = self._paged_kv_indices(slot_mapping, block_size, k.device)
        self.k_cache.index_put_(idx, k.reshape(-1, self.head_dim))
        self.v_cache.index_put_(idx, v.reshape(-1, self.head_dim))

    # ── prefill ─────────────────────────────────────────────────────────

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attn_metadata: dict,
    ) -> torch.Tensor:
        meta = attn_metadata[self.layer_name]
        hidden_states = hidden_states.to(self.dtype)
        tokens = hidden_states.shape[0]

        # <-- MODEL-SPECIFIC: RoPE only on sliding layers (full layers are NoPE).
        cos_cache = sin_cache = None
        if self.apply_rope:
            cos, sin = position_embeddings
            cos_cache = torch.cat((cos, cos), dim=-1).unsqueeze(0).to(self.dtype)
            sin_cache = torch.cat((sin, sin), dim=-1).unsqueeze(0).to(self.dtype)

        # <-- MODEL-SPECIFIC: weightless QK RMSNorm (gamma=None) before RoPE.
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            d_head=self.head_dim,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            num_q_heads=self.num_attention_heads_per_rank,
            num_kv_heads=self.num_key_value_heads_per_rank,
            qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_eps=self.qk_norm_eps,
            qk_norm_pre_rope_q_gamma=None,
            qk_norm_pre_rope_k_gamma=None,
        ).squeeze(0)
        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
        q = q.reshape(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(0, 1)
        k = k.reshape(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)
        v = v.reshape(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)

        self._write_paged_kv(k, v, meta["slot_mapping"], meta["block_size"])

        kv_segment_size = meta.get("kv_segment_size")
        if kv_segment_size:
            # Chunked (segmented) prefill: attend over the paged cache, which
            # now holds prior chunks plus this chunk.
            attn_output = NF.segmented_attention(
                q,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=meta["block_table_tensor"],
                prior_tokens=meta.get("cached_seq_len"),
                block_size=meta["block_size"],
                kv_segment_size=kv_segment_size,
                scale=self.scaling,
                sliding_window=self.sliding_window,
                tp_q=True,
                tp_out=True,
            )  # [N, D, T]
        else:
            # attention_cte handles GQA natively (q batch N, kv batch N/groups): no K/V replication.
            attn_output = NF.flash_attention(
                q.transpose(1, 2),  # [N, D, T]
                k.transpose(1, 2),  # [N, D, T]
                v,  # [N, T, D]
                scale=self.scaling,
                sliding_window=self.sliding_window,
                tp_q=False,
                tp_out=True,
            )  # [N, D, T]

        # <-- MODEL-SPECIFIC: sigmoid output gate before o_proj.
        gate = self._gate(hidden_states)  # [T, N*D]
        gate = gate.view(tokens, self.num_attention_heads_per_rank, self.head_dim).permute(1, 2, 0)
        attn_output = (attn_output * gate).to(self.dtype)
        out = NF.o_proj(attn_output.unsqueeze(0), self.o_proj_weight).squeeze(0)  # [T, H]

        # >>> PARALLELISM: reduce-scatter back to SP layout <<<
        if self.world_size > 1:
            out = self.tp_group.reduce_scatter(out, dim=0)
        return out.contiguous()

    # ── decode ──────────────────────────────────────────────────────────

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attn_metadata: dict,
        kv_writeback: list | None = None,
        in_norm: nn.Module | None = None,
    ) -> torch.Tensor:
        meta = attn_metadata[self.layer_name]
        slot_mapping = meta["slot_mapping"]
        block_size = meta["block_size"]
        block_table = meta["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S = tokens // B
        assert tokens == B * S
        if self.sliding_window is not None:
            padded = pad_swa_decode_table(block_table, block_size, get_logical_nc_config())
            if padded is not block_table and S != 1:
                # Multi-token steps can wrap the window mask past the trimmed width.
                raise NotImplementedError("MuseGlimmer: padded SWA decode table needs one token per step")
            block_table = padded
        nkh = self.num_key_value_heads_per_rank
        d = self.head_dim

        hidden_states = hidden_states.to(self.dtype)
        X = hidden_states.view(B, S, hidden)

        cos_k = sin_k = None
        if self.apply_rope:
            cos, sin = position_embeddings  # [T, d/2]
            cos_k = cos.view(B, S, d // 2).permute(2, 0, 1).contiguous().to(self.dtype)
            sin_k = sin.view(B, S, d // 2).permute(2, 0, 1).contiguous().to(self.dtype)

        # In-kernel mask generation: pos_ids (+ SWA window start).
        pos = positions.view(B, S).to(torch.float32)
        swa_start = None
        if self.sliding_window is not None:
            # Runner trims the SWA block table to the window and reports the
            # trimmed-frame offset (= start_block * block_size).
            off = meta.get("swa_kv_pos_offset")
            if off is not None:
                pos = torch.clamp(pos - off.view(B, 1).to(torch.float32), min=0.0)
            # HF window: kv > q - W, i.e. start = pos - W + 1 (inclusive).
            swa_start = torch.clamp(pos - (self.sliding_window - 1), min=0.0)

        k_cache, v_cache = self.k_cache, self.v_cache
        table = block_table
        if nkh == 1:
            # 3D [blocks, block, d] cache + 2D table: kernel-eligible.
            k_cache = k_cache.squeeze(1) if k_cache.dim() == 4 else k_cache
            v_cache = v_cache.squeeze(1) if v_cache.dim() == 4 else v_cache
        elif can_run_kernel(X) and not _decode_torch_forced():
            # GQA reaches the kernel only with a per-head 3D table.
            table = build_per_head_block_table(block_table, nkh)

        on_kernel = can_run_kernel(X) and not _decode_torch_forced()
        # gate + o_proj inside the megakernel (bf16 attention weights, single B*S tile).
        # Only for decode batch buckets >= MUSE_GLIMMER_DECODE_FUSE_GATE_MIN_BS (slower at B=1 on trn2).
        use_gate = ("gate" in self.fuse and tokens <= 128 and not _decode_torch_forced()
                    and B >= _df.gate_min_bs() and (nkh == 1 or on_kernel))
        # fold the input norm into the kernel (gamma = 1 + w in fp32). Without the in-kernel
        # gate the torch gate still needs the normed input, so the norm is computed in torch as well.
        norm_kw = {}
        if in_norm is not None:
            norm_kw = dict(rmsnorm_X_enabled=True, rmsnorm_X_eps=in_norm.eps,
                           rmsnorm_X_gamma=_df.centered_gamma(in_norm))
            if not use_gate:
                hidden_states = in_norm(hidden_states)
                if not on_kernel:  # torch fallback: plain normed input, no fold
                    X = hidden_states.view(B, S, hidden)
                    norm_kw = {}
        early_gate = None
        if "early_gate" in self.fuse and not use_gate:
            # Cheap variant: issue the gate projection before the megakernel (weight DMA overlap).
            early_gate = self._gate(hidden_states)
        decode_kwargs = dict(
            X=X,
            W_qkv=self.qkv_proj_weight,
            rmsnorm_X_enabled=False,  # input norm applied by the decoder layer (unless folded, norm_kw)
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=self.qk_norm_eps,
            rmsnorm_QK_pre_rope_W_Q=None,
            rmsnorm_QK_pre_rope_W_K=None,
            cos=cos_k,
            sin=sin_k,
            rope_contiguous_layout=True,
            active_blocks_table=table,
            K_cache=k_cache,
            V_cache=v_cache,
            softmax_scale=self.scaling,
            update_cache=False,
            W_out=None,  # gate must be applied before o_proj
        )
        decode_kwargs.update(norm_kw)
        if use_gate:
            return self._forward_decode_gated(decode_kwargs, pos, swa_start, B, S, table,
                                              slot_mapping, block_size, kv_writeback)
        if _decode_torch_forced():
            from vllm_neuron.functional.attention.attention_decode import (
                _torch_attention_decode_impl,
            )

            s_prior = table.shape[-1] * block_size
            mask = NF.gen_attention_decode_mask(
                pos_ids=pos.reshape(1, B * S),
                bs=B,
                q_head=self.num_attention_heads_per_rank,
                s_active=S,
                s_prior=s_prior,
                start_pos=swa_start.reshape(1, B * S) if swa_start is not None else None,
                block_len=block_size,
            )
            output, K_new, V_new = _torch_attention_decode_impl(
                attention_mask=mask, **decode_kwargs
            )
        else:
            output, K_new, V_new = NF.attention_decode(
                K_cache_transposed=False,
                transposed_out=False,
                out_in_sb=False,
                pos_ids=pos,
                swa_start_pos_ids=swa_start,
                **decode_kwargs,
            )

        # output: [B, Nq, d, S] -> [B*S, Nq*d]
        attn = output.permute(0, 3, 1, 2).reshape(B * S, self.q_size)
        gate = early_gate if early_gate is not None else self._gate(hidden_states)
        attn = (attn * gate).to(self.dtype)
        out = attn @ self.o_proj_weight  # [B*S, H]

        # Manual KV scatter. K_new: [d, B*nkh, S] (torch) / [d, B, nkh, S] or
        # [d, B, S] (kernel); V_new: [B, nkh, S, d].
        k_new = K_new.reshape(d, B, nkh, S).permute(2, 1, 3, 0).reshape(nkh, B * S, d)
        v_new = V_new.reshape(B, nkh, S, d).permute(1, 0, 2, 3).reshape(nkh, B * S, d)
        if kv_writeback is not None:
            # defer the write past the layer loop (see _issue_decode_kv_writes).
            kv_writeback.append((self.layer_idx, self, k_new, v_new, slot_mapping, block_size))
        else:
            self._write_paged_kv(k_new, v_new, slot_mapping, block_size)

        # >>> PARALLELISM: TP all-reduce of o_proj partials <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(out)
        return out


    def _forward_decode_gated(self, kw, pos, swa_start, B, S, table, slot_mapping, block_size,
                              kv_writeback):
        """megakernel with the sigmoid gate and o_proj in SBUF (``gate_decode``)."""
        from .gate_decode import gated_attention_decode

        out, K_new, V_new = gated_attention_decode(
            X=kw["X"], W_qkv=kw["W_qkv"], W_gate=self.gate_proj_weight, W_out=self.o_proj_weight,
            K_cache=kw["K_cache"], V_cache=kw["V_cache"], active_blocks_table=table,
            softmax_scale=kw["softmax_scale"], qk_norm_eps=self.qk_norm_eps, cos=kw["cos"], sin=kw["sin"],
            rmsnorm_X_enabled=kw["rmsnorm_X_enabled"], rmsnorm_X_eps=kw.get("rmsnorm_X_eps"),
            rmsnorm_X_gamma=kw.get("rmsnorm_X_gamma"),
            force_torch=_decode_torch_forced(), pos_ids=pos, swa_start_pos_ids=swa_start)
        nkh, d = self.num_key_value_heads_per_rank, self.head_dim
        k_new = K_new.reshape(d, B, nkh, S).permute(2, 1, 3, 0).reshape(nkh, B * S, d)
        v_new = V_new.reshape(B, nkh, S, d).permute(1, 0, 2, 3).reshape(nkh, B * S, d)
        if kv_writeback is not None:
            kv_writeback.append((self.layer_idx, self, k_new, v_new, slot_mapping, block_size))
        else:
            self._write_paged_kv(k_new, v_new, slot_mapping, block_size)
        out = out.reshape(B * S, self.hidden_size)
        if self.world_size > 1:
            self.tp_group.all_reduce(out)
        return out


# ---------------------------------------------------------------------------
# Section 4: MLP (SwiGLU)
# ---------------------------------------------------------------------------

# nkilib mlp picks TKG for B*S <= 96 (TKG_BS_SEQLEN_THRESHOLD).
_MLP_TKG_MAX_T = 96


class MuseGlimmerMLP(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        if config.intermediate_size % self.world_size != 0:
            raise ValueError("intermediate_size must be divisible by TP")
        inter = config.intermediate_size // self.world_size
        H = config.hidden_size
        dt = config.torch_dtype
        self.gate_proj_weight = nn.Parameter(torch.empty(H, inter, dtype=dt))
        self.up_proj_weight = nn.Parameter(torch.empty(H, inter, dtype=dt))
        self.down_proj_weight = nn.Parameter(torch.empty(inter, H, dtype=dt))
        col = sharding_weight_loader(
            shard_dim=1, shard_size=inter, num_shards=self.world_size, is_storage_transposed=True
        )
        row = sharding_weight_loader(
            shard_dim=0, shard_size=inter, num_shards=self.world_size, is_storage_transposed=True
        )
        set_weight_loader(self.gate_proj_weight, col)
        set_weight_loader(self.up_proj_weight, col)
        set_weight_loader(self.down_proj_weight, row)

    def forward_fused_decode(self, residual: torch.Tensor, attn_out: torch.Tensor, norm) -> tuple:
        """(residual + attn_out) -> centered RMSNorm (gamma = 1 + w, fp32) -> MLP, in one NF.mlp
        call (TKG fused add + norm). Returns ``(mlp_out, residual + attn_out)``; mlp_out is all-reduced."""
        kw = dict(eps=norm.eps, ln_w=_df.centered_gamma(norm), norm_type=NormType.RMS_NORM,
                  fused_add_tensor=attn_out, store_fused_add_result=True)
        out, hsum = NF.mlp(residual, self.gate_proj_weight, self.up_proj_weight, self.down_proj_weight, **kw)
        if self.world_size > 1:
            self.tp_group.all_reduce(out)
        return out, hsum.to(residual.dtype)

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        out = NF.mlp(
            hidden_states, self.gate_proj_weight, self.up_proj_weight, self.down_proj_weight
        )
        if self.world_size > 1:
            if is_prefill:
                out = self.tp_group.reduce_scatter(out, dim=0)
            else:
                self.tp_group.all_reduce(out)
        return out


# ---------------------------------------------------------------------------
# Section 5: Decoder layer (sandwich norms)
# ---------------------------------------------------------------------------


class MuseGlimmerDecoderLayer(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        H = config.hidden_size
        self.input_layernorm = MuseGlimmerCenteredRMSNorm(H, config.rms_norm_eps)
        self.post_attention_layernorm = MuseGlimmerCenteredRMSNorm(H, config.post_norm_eps)
        self.pre_feedforward_layernorm = MuseGlimmerCenteredRMSNorm(H, config.rms_norm_eps)
        self.post_feedforward_layernorm = MuseGlimmerCenteredRMSNorm(H, config.post_norm_eps)
        self.self_attn = MuseGlimmerAttention(config, layer_idx)
        self.mlp = MuseGlimmerMLP(config)

    def forward(self, hidden_states, positions, position_embeddings, attn_metadata, is_prefill,
                kv_writeback=None):
        fuse = self.self_attn.fuse if not is_prefill else ()
        T = hidden_states.shape[0]
        residual = hidden_states
        if "norm" in fuse:  # input norm folded into the decode megakernel
            h = self.self_attn(hidden_states, positions, position_embeddings, attn_metadata, kv_writeback,
                               in_norm=self.input_layernorm)
        else:
            h = self.self_attn(self.input_layernorm(hidden_states), positions, position_embeddings,
                               attn_metadata, kv_writeback)
        h = self.post_attention_layernorm(h)

        if "ffn" in fuse and T <= _MLP_TKG_MAX_T:  # residual add + pre-FFN norm inside NF.mlp
            h, residual = self.mlp.forward_fused_decode(residual, h, self.pre_feedforward_layernorm)
        else:
            hidden_states = residual + h
            residual = hidden_states
            h = self.pre_feedforward_layernorm(hidden_states)
            h = self.mlp(h, is_prefill=is_prefill)
        h = self.post_feedforward_layernorm(h)
        return residual + h


# ---------------------------------------------------------------------------
# Section 6: Backbone
# ---------------------------------------------------------------------------


class MuseGlimmerTextModel(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

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
                is_storage_transposed=False,
                pad_shard=True,
            ),
        )
        # <-- MODEL-SPECIFIC: weightless norm right after the embedding lookup.
        self.embed_norm = MuseGlimmerRMSNorm(
            None, config.rms_norm_eps, config.torch_dtype, with_scale=False
        )
        self.layers = nn.ModuleList(
            [MuseGlimmerDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = MuseGlimmerRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype, with_scale=True
        )
        self.rotary_emb = MuseGlimmerRotaryEmbedding(config)
        # read once (static Python int; no env read inside the trace).
        self.kv_defer_mode = _kv_defer_mode()
        # layer_idx -> physical (k_ptr, v_ptr) group id, resolved in bind_kv_cache
        # (outside the trace). None -> per-layer deferred writes.
        self.kv_ptr_group_of: list[int] | None = None
        # Spec decode: pre-layer capture indices, set by the runner (outside the trace).
        self.aux_hidden_state_layers: list[int] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict,
        rank: torch.Tensor | None = None,
        encoder_embedding_buffer: torch.Tensor | None = None,
        encoder_block_ids: torch.Tensor | None = None,
        encoder_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        meta0 = attn_metadata["layers.0.self_attn"]
        is_prefill = meta0["max_query_len"] > meta0["decode_token_threshold"]

        # >>> PARALLELISM: embedding reduce-scatters (SP) in prefill <<<
        h = self.embed_tokens(input_ids, scatter_tokens=is_prefill, rank=rank)
        h = self.embed_norm(h)

        # Vision features replace placeholder rows AFTER embed_norm (HF order).
        # 0.29 contract: the whole encoder-cache buffer + per-step block ids; the SP
        # rank comes from the runtime tensor so every rank traces one graph (SPMD).
        if (
            is_prefill
            and encoder_embedding_buffer is not None
            and encoder_block_ids is not None
            and encoder_positions is not None
        ):
            h, _ = merge_multimodal_embeddings(
                h,
                encoder_embedding_buffer,
                encoder_block_ids,
                encoder_positions,
                rank=self.tp_group.rank_in_group_tensor,
            )

        position_embeddings = self.rotary_emb(positions)

        # decode KV writes are deferred past the layer loop (mode >= 1).
        kv_writeback = [] if (not is_prefill and self.kv_defer_mode >= 1) else None
        # Spec decode (EAGLE3 / DFlash): capture the residual stream entering each
        # requested layer (= HF output_hidden_states of layer idx - 1).
        aux = []
        for idx, layer in enumerate(self.layers):
            if idx in self.aux_hidden_state_layers:
                aux.append(h)
            h = layer(h, positions, position_embeddings, attn_metadata, is_prefill, kv_writeback)
        if kv_writeback:
            _issue_decode_kv_writes(
                kv_writeback, self.kv_ptr_group_of if self.kv_defer_mode >= 2 else None
            )

        h = self.norm(h)
        if is_prefill and self.world_size > 1:
            h = self.tp_group.all_gather(h, dim=0)
            aux = [self.tp_group.all_gather(a, dim=0) for a in aux]
        return h, aux


# ---------------------------------------------------------------------------
# Section 7: Top-level model (LM head, softcap, KV spec, weight loading)
# ---------------------------------------------------------------------------

HF_TEXT_PREFIX = "model.language_model"


@async_speculative_decoding
class MuseGlimmerForConditionalGeneration(nn.Module, SupportsEagle3):
    """MuseGlimmer multimodal model: text decoder + perception encoder."""

    def __init__(self, config: MuseGlimmerConfig):
        super().__init__()
        vnc = config.vision_config.neuron_config if config.vision_config else None
        self.mm_encoder_only = bool(vnc and vnc.mm_encoder_only)
        self.mm_language_model_only = bool(vnc and vnc.mm_language_model_only)
        if self.mm_encoder_only:
            raise NotImplementedError(
                "MuseGlimmer: mm_encoder_only (vision-only EPD pool) is not "
                "supported yet; run the monolithic model."
            )
        self.config = config
        self.text_config = config.text_config
        tc = config.text_config

        # Vision tower (skipped on a language-only pool).
        self.visual = None if self.mm_language_model_only else MuseGlimmerVisionModel(config)

        self.language_model = MuseGlimmerTextModel(tc)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        nc = tc.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        debug_logits = nc is not None and nc.debug_logits_dir is not None
        self._gather_logits = (nc is not None and nc.max_logprobs != 0) or debug_logits

        if tc.vocab_size % self.world_size != 0:
            raise ValueError(f"vocab_size={tc.vocab_size} not divisible by TP={self.world_size}")
        self.lm_head = neuron_nn.ColumnParallelLinear(
            tc.hidden_size,
            tc.vocab_size,
            bias=False,
            dtype=tc.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader_with_padding(
                shard_dim=0,
                shard_size=tc.vocab_size // self.world_size,
                num_shards=self.world_size,
                pad_dim=1,
                padded_size=tc.hidden_size,
                unpadded_size=tc.hidden_size,
            ),
        )
        self.sampler = (
            Sampler(self.on_device_sampling_config, process_group=self.tp_group.device_group)
            if self.on_device_sampling_config is not None
            else None
        )
        # <-- MODEL-SPECIFIC: logit multiplier + tanh softcap (Python floats).
        self.logit_softcap = float(tc.final_logit_softcapping)
        self.logit_pre_scale = float(tc.output_multiplier) / self.logit_softcap

    def _softcap(self, logits: torch.Tensor) -> torch.Tensor:
        """``cap * tanh(logits * mult / cap)`` in fp32 (elementwise, so it
        commutes with vocab sharding and is applied before sampling and
        before the logprobs gather)."""
        return torch.tanh(logits.to(torch.float32) * self.logit_pre_scale) * self.logit_softcap

    @property
    def compiler_args_for(self):
        """per-graph neuronx-cc policy (decode graph -O, ``MUSE_GLIMMER_DECODE_OPT``).

        The runner enables per-graph args when this attribute is not None. With
        ``MUSE_GLIMMER_DECODE_OPT=off`` it is None and every graph keeps the engine level.
        """
        level = _decode_opt_level()
        vis = vision_opt_level()  # encoder graph -O (MUSE_GLIMMER_VIS_OPT)
        if level is None and vis is None:
            return None
        from vllm_neuron.utils.compiler_args_ctx import with_optimization_level

        def policy(graph_type, example_inputs, base):
            base = list(base or [])
            if graph_type == "decode" and level is not None:
                return with_optimization_level(base, level)
            if graph_type == "encoder" and vis is not None:
                return with_optimization_level(base, vis)
            return base

        return policy

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        encoder_embedding_buffer: torch.Tensor | None = None,
        encoder_block_ids: torch.Tensor | None = None,
        encoder_positions: torch.Tensor | None = None,
        **kwargs,
    ):
        positions = positions.to(torch.int32)
        meta0 = attn_metadata["layers.0.self_attn"]
        is_prefill = meta0["max_query_len"] > meta0["decode_token_threshold"]
        T = input_ids.shape[0]
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(f"Prompt Length ({T}) must be > world_size ({self.world_size}) for SP.")

        h, aux = self.language_model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            encoder_embedding_buffer=encoder_embedding_buffer,
            encoder_block_ids=encoder_block_ids,
            encoder_positions=encoder_positions,
        )
        h = torch.index_select(h, dim=0, index=sampling_positions)
        logits = self._softcap(self.lm_head(h))
        # Spec decode (DFlash / EAGLE3): one [T, n_aux * H] tensor, cat inside the NEFF.
        aux_cat = torch.cat(aux, dim=-1) if aux else None

        gathered_logits = None
        if self._gather_logits:
            gathered_logits = self.tp_group.all_gather(logits, dim=1)

        if self.on_device_sampling_config is None:
            return (logits, aux_cat) if aux_cat is not None else logits

        sampled = self.sampler(logits, sampling_params, logit_mask=logit_mask, tp_rank=rank)
        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            rejected = rejection_sampler(spec_decode_metadata, sampled)
            if aux_cat is not None:
                # @async_speculative_decoding appends last_accepted_token.
                return rejected, aux_cat, gathered_logits
            return rejected
        if aux_cat is not None:
            return sampled, aux_cat, gathered_logits
        return sampled, gathered_logits

    # ── speculative decoding (SupportsEagle3 target interface) ──────────

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        if layers is not None:
            n = len(self.language_model.layers)
            assert all(0 <= i < n for i in layers), (layers, n)
            self.language_model.aux_hidden_state_layers = list(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        if self.language_model.aux_hidden_state_layers:
            return tuple(self.language_model.aux_hidden_state_layers)
        n = len(self.language_model.layers)
        return (2, n // 2, n - 3)


    # ── multimodal ──────────────────────────────────────────────────────

    # no_grad: the runner calls this outside a grad-free context; without it the
    # encoder-cache index_put_ keeps every activation's autograd graph alive (CPU mode
    # leaked ~9 GB per 4096-patch image).
    @torch.no_grad()
    def embed_multimodal(
        self,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        encoder_cache=None,
        mm_hashes: list[str] | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """Encode images or videos and write their features into the on-device encoder cache.

        Flow (as qwen3_vl): allocate cache blocks per item -> pick a vision
        bucket -> one-item-per-block packing -> CPU preprocessing in the
        MuseGlimmer pack order -> VE graph scatter-writes into the cache.

        Video (``pixel_values_videos`` / ``video_grid_thw``) uses the image path
        unchanged: HF ``get_video_features`` == ``get_image_features``, and the
        HF ViT attends per temporal group (``cu_seqlens`` repeats ``h*w`` ``t``
        times; windows never cross a group). Each ``[T, H, W]`` row therefore
        becomes ``T`` items ``[1, H, W]`` that share the video's block run
        (``group_ids``): a block holds whole temporal groups, each with its own
        full/window bounds, RoPE and position taps, so the features equal HF's.
        A group must fit in one block (``H*W <= encoder_attention_block_size``);
        a video spans ``ceil(T / (block // (H*W)))`` blocks, non-last blocks
        may carry a pad tail recorded in ``tokens_per_block``.
        """
        if self.visual is None:
            raise RuntimeError("MuseGlimmer: vision tower not built (mm_language_model_only)")
        if pixel_values is not None and pixel_values_videos is not None:
            raise ValueError("embed_multimodal: images and videos must come in separate calls")
        is_video = pixel_values is None and pixel_values_videos is not None
        if is_video:
            pixel_values, image_grid_thw = pixel_values_videos, video_grid_thw
        if pixel_values is None or image_grid_thw is None:
            raise ValueError(
                "embed_multimodal requires (pixel_values, image_grid_thw) or "
                "(pixel_values_videos, video_grid_thw)"
            )

        cache = encoder_cache
        vc = self.config.vision_config
        vnc = vc.neuron_config
        t0 = _time.perf_counter()
        mu = vc.merge_size**2
        block_size = vnc.encoder_attention_block_size
        grid_thw = image_grid_thw.to(torch.int64).reshape(-1, 3)
        rows = grid_thw.tolist()
        if len(mm_hashes) != len(rows):
            raise ValueError(f"{len(mm_hashes)} mm_hashes for {len(rows)} items")

        if is_video:
            grid_ve, group_ids, tpb = plan_video_items(grid_thw, block_size, vc.merge_size)
            block_map = [cache.allocate(mm_hashes[i], tpb[i]) for i in range(len(rows))]
        else:
            for t, h, w in rows:
                if t * h * w > block_size:
                    raise ValueError(
                        f"encoder_attention_block_size={block_size} < image patches {t * h * w}; "
                        "an image must fit in one block. Increase the block size / bucket "
                        "or lower mm_processor_kwargs max_image_tokens."
                    )
            grid_ve = grid_thw
            group_ids = None
            block_map = [
                cache.allocate(mm_hashes[i], cache.dense_tokens_per_block(t * h * w // mu, block_size // mu))
                for i, (t, h, w) in enumerate(rows)
            ]
        tokens = [int(t * h * w) for t, h, w in grid_ve.tolist()]
        required_blocks = sum(len(b) for b in block_map)
        _bucket, num_blocks = select_vision_bucket(
            max(sum(tokens), required_blocks * block_size),
            vnc.num_encoder_input_buckets, block_size, dp_size=vnc.dp_size,
        )
        # (MUSE_GLIMMER_VIS_SMALL_BLOCK): one-block requests run the smallest
        # bucket that holds them; a bucket below the block size is ONE block of `bucket` rows.
        blk_rows = block_size
        if required_blocks == 1 and _vis_small_block(vnc):
            _bucket = min(b for b in vnc.num_encoder_input_buckets if b >= sum(tokens))
            blk_rows, num_blocks = _graph_block_rows(_bucket, vnc)
        assignment = ffd_pack_images(
            tokens, blk_rows, num_blocks, one_item_per_block=True, group_ids=group_ids
        )
        packed = pack_vision_inputs(
            pixel_values.reshape(-1, vc.patch_dim),
            grid_ve,
            assignment,
            head_dim=vc.head_dim,
            rope_theta=vc.rope_theta,
            num_grid_per_side=vc.pos_emb_height,
            merge_size=vc.merge_size,
            rope_dtype=self.visual.dtype,
            win_chunks=_vis_win_chunks(blk_rows),
            win_chunk=WIN_CHUNK,
        )
        flat = [b for blocks in block_map for b in blocks]
        write_ids = torch.tensor(flat + [cache.scratch_block_id] * (num_blocks - len(flat)), dtype=torch.int64)
        t1 = _time.perf_counter()

        device = next(self.visual.parameters()).device
        kw = packed.as_kwargs()
        kw["pixel_values"] = kw["pixel_values"].to(self.visual.dtype)
        kw = {k: v.to(device) for k, v in kw.items()}
        write_dev = write_ids.to(device)
        timing = _vis_timing()
        sync = (lambda: torch.neuron.synchronize()) if device.type == "neuron" else (lambda: None)
        if timing:  # host / H2D / device split (synchronize so each part is real)
            sync()
            t_h2d = _time.perf_counter()
        with self._scoped_encoder_args():
            self.visual(**kw, encoder_cache_buffer=cache.buffer, write_block_ids=write_dev)
        if timing:
            sync()
            t2 = _time.perf_counter()
            # synchronize() can return before the native-backend graph finishes (observed on trn2:
            # 3 ms for a 50-layer encode); a host read of the written cache row waits for it.
            try:
                cache.buffer[int(write_ids[0])][:1].cpu()
            except Exception as e:  # noqa: BLE001
                logger.warning("VIS_TIMING readback failed: %r", e)
            t3 = _time.perf_counter()
            rec = dict(pid=os.getpid(), items=len(grid_thw), video=int(is_video), patches=sum(tokens),
                       bucket=_bucket, rows=blk_rows, blocks=num_blocks, cpu_ms=(t1 - t0) * 1e3,
                       h2d_ms=(t_h2d - t1) * 1e3, device_ms=(t2 - t_h2d) * 1e3,
                       readback_ms=(t3 - t_h2d) * 1e3, t=_time.time())
            logger.warning("VIS_TIMING %s", rec)
            if _VIS_TIMING_FILE:
                import json as _json

                with open(_VIS_TIMING_FILE, "a") as fh:
                    fh.write(_json.dumps(rec) + "\n")
        logger.debug(
            "[embed_multimodal] %d %s, %d patches, %d blocks: cpu=%.1fms device=%.1fms",
            len(grid_thw), "videos" if is_video else "images", sum(tokens), num_blocks,
            (t1 - t0) * 1e3, (_time.perf_counter() - t1) * 1e3,
        )

    def _scoped_encoder_args(self):
        """scope the "encoder" per-graph compiler args (MUSE_GLIMMER_VIS_OPT) for the VE call."""
        from contextlib import nullcontext

        base = getattr(self, "_neuron_base_compiler_args", None)
        policy = self.compiler_args_for
        # Unset knob: no scope, the plain call (same graph hash as warmup).
        if base is None or policy is None or vision_opt_level() is None:
            return nullcontext()
        from libtorch_neuronx_lite.compile.compiler_args import scoped_compiler_args

        return scoped_compiler_args(policy("encoder", None, base))

    def get_encoder(self) -> torch.nn.Module | None:
        """The vision tower (None on a language-only pool)."""
        return self.visual

    def build_encoder_synthetic_inputs(
        self, bucket: int, encoder_neuron_config: EncoderNeuronConfig, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Shape-only inputs for ``self.visual`` warmup / tracing (per bucket)."""
        vc = self.config.vision_config
        bs, nb = _graph_block_rows(bucket, encoder_neuron_config)
        mu = vc.merge_size**2
        z = lambda *shape, dtype: torch.zeros(*shape, dtype=dtype, device=device)  # noqa: E731
        win = {}
        K = _vis_win_chunks(bs)
        if K:  # chunk gather/scatter + chunk-local bounds
            win = {
                "win_gather": z(nb, K * WIN_CHUNK, dtype=torch.int64),
                "win_scatter": z(nb, bs, dtype=torch.int64),
                "win_bound_min": z(nb * K, WIN_CHUNK, 1, dtype=torch.int32),
                "win_bound_max": z(nb * K, WIN_CHUNK, 1, dtype=torch.int32),
            }
        return {
            "pixel_values": z(nb, bs, vc.patch_dim, dtype=self.visual.dtype),
            "pos_emb_idx": z(4, nb, bs, dtype=torch.int32),
            "pos_emb_weight": z(4, nb, bs, dtype=torch.float32),
            "cos": z(nb, bs, vc.head_dim, dtype=torch.float32),
            "sin": z(nb, bs, vc.head_dim, dtype=torch.float32),
            "bound_min": z(nb, bs, 1, dtype=torch.int32),
            "bound_max": z(nb, bs, 1, dtype=torch.int32),
            "window_bound_min": z(nb, bs, 1, dtype=torch.int32),
            "window_bound_max": z(nb, bs, 1, dtype=torch.int32),
            # Built on CPU then moved: aten::repeat (unfold) has no neuron impl.
            "merged_order": torch.arange(bs // mu, dtype=torch.int64).repeat(nb, 1).to(device)
            if device != torch.device("meta")
            else z(nb, bs // mu, dtype=torch.int64),
        } | win  # same kwarg order as embed_multimodal (PackedVisionInputs field order)

    # ── construction ────────────────────────────────────────────────────

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: EncoderNeuronConfig | None = None,
    ):
        config = MuseGlimmerConfig.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            encoder_neuron_config=encoder_neuron_config,
        )
        return cls(config)

    # ── KV cache ────────────────────────────────────────────────────────

    def get_kv_spec(self) -> KVSpec:
        layers = []
        for i, layer in enumerate(self.language_model.layers):
            attn = layer.self_attn
            layers.append(
                LayerSpec(
                    name=f"layers.{i}.self_attn",
                    num_kv_heads=attn.num_key_value_heads_per_rank,
                    head_size=attn.head_dim,
                    dtype=attn.dtype,
                    # <-- MODEL-SPECIFIC: per-layer SWA -> SlidingWindowSpec
                    sliding_window_size=attn.sliding_window,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
        for i, layer in enumerate(self.language_model.layers):
            name = f"layers.{i}.self_attn"
            if name not in kv_caches:
                raise Exception(f"KV cache for layer {name} not initialized")
            layer.self_attn.k_cache = kv_caches[name][0]
            layer.self_attn.v_cache = kv_caches[name][1]
            if layer.self_attn.k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                raise NotImplementedError(
                    "MuseGlimmer (contrib): an FP8 KV cache is not supported; use the default "
                    "--kv-cache-dtype auto (BF16)."
                )
        # resolve physical-buffer groups here, never inside the trace.
        lm = self.language_model
        lm.kv_ptr_group_of = _resolve_kv_ptr_groups(lm.layers)
        if lm.kv_ptr_group_of is not None:
            groups: dict[int, list[int]] = {}
            for i, g in enumerate(lm.kv_ptr_group_of):
                groups.setdefault(g, []).append(i)
            logger.info(
                "MuseGlimmer KV: %s=%d, %d layers -> %d physical (k,v) buffers: %s",
                KV_DEFER_ENV, lm.kv_defer_mode, len(lm.layers), len(groups),
                list(groups.values()),
            )

    # ── weights ─────────────────────────────────────────────────────────

    def build_text_weight_mappings(self) -> dict[str, str | list[str]]:
        m: dict[str, str | list[str]] = {}
        for i in range(len(self.language_model.layers)):
            hf = f"{HF_TEXT_PREFIX}.layers.{i}"
            ours = f"language_model.layers.{i}"
            m[f"{ours}.self_attn.qkv_proj_weight"] = [
                f"{hf}.self_attn.q_proj.weight",
                f"{hf}.self_attn.k_proj.weight",
                f"{hf}.self_attn.v_proj.weight",
            ]
            m[f"{ours}.self_attn.gate_proj_weight"] = f"{hf}.self_attn.gate_proj.weight"
            m[f"{ours}.self_attn.o_proj_weight"] = f"{hf}.self_attn.o_proj.weight"
            for n in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ):
                m[f"{ours}.{n}.weight"] = f"{hf}.{n}.weight"
            for p in ("gate_proj", "up_proj", "down_proj"):
                m[f"{ours}.mlp.{p}_weight"] = f"{hf}.mlp.{p}.weight"
        m["language_model.embed_tokens.weight"] = f"{HF_TEXT_PREFIX}.embed_tokens.weight"
        m["language_model.norm.weight"] = f"{HF_TEXT_PREFIX}.norm.weight"
        m["lm_head.weight"] = "lm_head.weight"
        return m

    def load_weights(self, checkpoint_path: str, device: torch.device, cache_dir: str | None) -> None:
        mappings = self.build_text_weight_mappings()
        params = {n: p for n, p in self.named_parameters() if not n.startswith("visual.")}
        unmapped = sorted(set(params) - set(mappings))
        if unmapped:
            raise RuntimeError(f"MuseGlimmer: parameters without a checkpoint mapping: {unmapped}")

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        visual, self.visual = self.visual, None  # text-TP load must not see vision params
        try:
            sd = checkpoint.load_sharded_pipelined(
                self.rank, self.world_size, self, mappings, device, strict=True
            ).state_dict
        finally:
            self.visual = visual
        # Keep each parameter's own dtype: bf16 weights, fp32 centered-norm w.
        for name, tensor in sd.items():
            want = params[name].dtype
            if tensor.dtype != want:
                sd[name] = tensor.to(want)
        missing = sorted(set(params) - set(sd))
        if missing:
            raise RuntimeError(f"MuseGlimmer: checkpoint did not provide: {missing}")
        self.load_state_dict(sd, strict=False, assign=True)
        # Vision tower (model.vision_tower.*, vision_adapter, vision_projection) on the
        # vision TP group (DP: full copy per rank).
        if self.visual is not None:
            self.visual.load_weights(checkpoint_path, device="cpu", cache_dir=cache_dir)


def per_tower_param_bytes(model) -> dict[str, int]:
    def _bytes(module) -> int:
        return 0 if module is None else sum(p.nbytes for p in module.parameters())

    return {
        "visual_param_bytes": _bytes(getattr(model, "visual", None)),
        "language_model_param_bytes": _bytes(getattr(model, "language_model", None)),
    }
