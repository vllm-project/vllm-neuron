# SPDX-License-Identifier: Apache-2.0
"""
MiMo-V2.5 BF16 Implementation
=============================

Text decoder for ``MiMoV2ForCausalLM`` (``model_type="mimo_v2"``) on the
PyTorch-native Neuron backend. Ported from the reference GPT-OSS
(``gpt_oss/model_bf16.py``, parallelism patterns) and Qwen3.5-MoE
(``qwen3_5_moe/model.py``, eager paged attention + static-block MoE) models.

Architecture (48 layers, hidden 4096, 256 experts, top-8):

  * **Hybrid attention.** ``hybrid_layer_pattern[i] == 0`` -> FULL attention,
    ``== 1`` -> SWA (window 128). Note the polarity; the 9 full layers are
    0, 5, 11, 17, 23, 29, 35, 41, 47.
  * **Asymmetric head dims.** Q/K are 192 wide, V is 128. Both the paged K and
    V caches are allocated at 192 (the framework derives one ``head_size`` per
    layer) and V is zero-padded on write / sliced back on read.
  * **Asymmetric KV heads.** Full layers have 4 KV heads, SWA layers 8.
  * **Partial RoPE.** Only the first ``int(192 * 0.334) = 64`` dims are rotated.
  * **Dual RoPE base.** ``rope_theta=1e7`` on full layers, ``swa_rope_theta=1e4``
    on SWA layers, so the backbone maintains TWO cos/sin caches (mirroring HF's
    ``rotary_emb`` / ``swa_rotary_emb``).
  * **Attention sink.** One learned logit bias per Q head, on SWA layers only
    (``add_swa_attention_sink_bias=True``, ``add_full_attention_sink_bias=False``).
  * **``attention_value_scale=0.707``** multiplies V before it enters the cache.
  * **Sigmoid ``noaux_tc`` router** with an ``e_score_correction_bias`` that
    biases SELECTION but not the returned weights.
  * **Layer 0 is dense** (``moe_layer_freq[0] == 0``); layers 1..47 are MoE.

Why attention is hand-written eager instead of using the fused kernels:

  * ``NF.flash_attention`` / ``NF.attention_decode`` / ``NF.segmented_attention``
    all cap ``head_dim`` at 128. Worse, the flash gate inspects only V's shape
    (128 here) so it would silently ACCEPT this model and then run with a
    truncated Q/K. Prefill and decode therefore both use explicit fp32 eager
    attention, which also lets us reproduce HF's sink handling exactly
    (concat sink column -> subtract row max -> fp32 softmax -> drop the column).
  * ``NF.qkv_proj`` and ``NF.o_proj`` ARE used: at TP=64 the fused QKV width is
    512 (<= 4096 and <= H), ``H % 128 == 0``, and o_proj sees N=1 head of D=128.

Supported parallelism: TP + SP + EP. Attention/MLP/embedding/LM-head DP are
NOT supported (this model's KV geometry and torch router were not validated
against the DP transitions); ``from_configs`` rejects those configs.
"""

import logging
import math
import os

import nki.language as nl
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.functional.topk import topk as neuron_topk
from vllm_neuron.nki.nki_hop import can_run_kernel
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    expert_parallel_interleaved_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import MiMoV2Config
from .weight_loaders_bf16 import (
    dense_down_fp8_loader,
    dense_gate_up_fp8_loader,
    expert_down_fp8_loader,
    expert_gate_up_fp8_loader,
    fused_qkv_fp8_loader,
    infer_qkv_disk_tp,
)

logger = logging.getLogger(__name__)

# Additive mask value for disallowed attention slots. Applied to fp32 scores
# before the softmax; -1e9 (rather than finfo.min) keeps the subsequent
# subtract-max finite even for an all-masked row.
_MASK_NEG = -1e9


def _topk_last_dim(tensor: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k over the last dim, avoiding a bare ``torch.topk`` on device.

    ``torch.topk`` lowers to an HLO ``sort``, which neuronx-cc rejects outright on
    trn2 -- "[NCC_EVRF029] Operation sort is not supported on trn2. Use supported
    equivalent operation like TopK or replace it with an alternate implementation
    via Neuron Kernel Interface (NKI)." So the router's expert selection cannot
    use it: it breaks COMPILATION (47 sorts, one per MoE layer), not just speed.

    ``vllm_neuron.functional.topk.topk`` is that NKI alternative: with
    ``process_group=None`` it takes the single-device path, using the rotational
    NKI kernel where its feasibility gate admits the shape and falling back to
    ``torch.topk`` otherwise -- which keeps CPU parity tests working unchanged.
    The router's ``[T, E]`` score tensor is replicated, not sharded, so no
    process group / rank is involved and ``gather_dim`` is inert.

    Returns sorted-descending values with int64 indices, the same contract as
    ``torch.topk``; the router's callers do not depend on the order either way.
    """
    return neuron_topk(tensor, k=k, dim=-1, gather_dim=-1)


# =============================================================================
# Section 0: Neuron-safe tensor helpers
# =============================================================================


def _repeat_interleave_heads(x: torch.Tensor, repeats: int, dim: int) -> torch.Tensor:
    """Index-free equivalent of ``x.repeat_interleave(repeats, dim=dim)``.

    ``torch.repeat_interleave`` lowers to an indirect (vector-DGE) gather on
    Neuron: the per-output index ``arange(out) // repeats`` is a runtime access
    pattern, so neuronx-cc cannot prove it in bounds and emits the DGE in
    ``OOBMode.ERROR``, which faults at execute (nrta-1006). The broadcast form
    below uses only ``unsqueeze``/``expand``/``reshape`` — no indirect
    addressing — and is bit-identical: each slice along ``dim`` is repeated
    ``repeats`` times CONSECUTIVELY, exactly ``repeat_interleave``'s order.
    """
    if repeats == 1:
        return x
    shape = list(x.shape)
    exp = shape[: dim + 1] + [repeats] + shape[dim + 1 :]
    out = shape[:dim] + [shape[dim] * repeats] + shape[dim + 1 :]
    return x.unsqueeze(dim + 1).expand(*exp).reshape(*out)


# =============================================================================
# Section 1: RMS Normalization
# =============================================================================


class MiMoV2RMSNorm(nn.Module):
    """Plain RMSNorm: ``weight * (x * rsqrt(mean(x^2) + eps))``.

    Matches ``MiMoV2RMSNorm`` in the HF modeling file exactly: fp32 variance,
    then multiply by the raw weight — there is NO ``1 + weight`` fold (unlike
    Qwen3.5), so the checkpoint gamma is used as stored.
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
        return self.weight * hidden_states.to(input_dtype)


# =============================================================================
# Section 2: Rotary Position Embedding (partial RoPE, dual base)
# =============================================================================


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partial RoPE: rotate the first ``rope_dim`` dims, pass the rest through.

    ``cos``/``sin`` arrive HALF-width (``[T, rope_dim // 2]``) and are doubled
    here. HF emits them full-width and does not double inside its helper; the
    two conventions are numerically identical as long as one is used
    consistently, and this one is the device-proven form.

    Args:
        q: ``[num_q_heads, T, head_dim]``.
        k: ``[num_kv_heads, T, head_dim]``.
        cos, sin: ``[T, rope_dim // 2]``.
        rope_dim: rotated width (64 for the released config).
    """
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(0)  # [1, T, rope_dim]
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(0)

    q_rot, q_pass = q[..., :rope_dim], q[..., rope_dim:]
    k_rot, k_pass = k[..., :rope_dim], k[..., rope_dim:]

    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)

    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


class MiMoV2RotaryEmbedding(nn.Module):
    """Partial RoPE frequency table. Two instances exist per model.

    HF's ``MiMoV2RotaryEmbedding(config, is_swa)`` swaps in ``swa_rope_theta``
    and ``swa_head_dim`` when ``is_swa``, then computes
    ``dim = int(head_dim * partial_rotary_factor)`` and
    ``inv_freq = 1 / base ** (arange(0, dim, 2) / dim)``. The released config
    has ``head_dim == swa_head_dim == 192`` so only the base differs
    (1e7 full vs 1e4 SWA) — but the geometry is read per-variant anyway so a
    future config with differing SWA head dims still works.
    """

    inv_freq: torch.Tensor

    def __init__(self, config: MiMoV2Config, is_swa: bool):
        super().__init__()
        base = config.swa_rope_theta if is_swa else config.rope_theta
        head_dim = config.swa_head_dim if is_swa else config.head_dim
        dim = int(head_dim * config.partial_rotary_factor)
        if dim % 2 != 0:
            raise ValueError(
                f"rope_dim must be even, got {dim} "
                f"(head_dim={head_dim}, partial_rotary_factor="
                f"{config.partial_rotary_factor})"
            )
        self.rope_dim = dim
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float, device="cpu") / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        position_ids: torch.Tensor,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(cos, sin)``, each ``[T, rope_dim // 2]``."""
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        freqs = torch.outer(position_ids.float(), inv_freq)  # [T, rope_dim/2]
        return freqs.cos().to(dtype), freqs.sin().to(dtype)


# =============================================================================
# Section 3: Attention (hybrid full / sliding-window)
# =============================================================================


class MiMoV2Attention(nn.Module):
    """Hybrid attention with TP head sharding and an eager paged KV path.

    Per-layer geometry is selected from ``hybrid_layer_pattern``: full layers
    use ``num_key_value_heads`` and no window, SWA layers use
    ``swa_num_key_value_heads`` plus a ``sliding_window``-wide window and the
    learned per-head sink bias.

    KV cache layout: ``[num_blocks, kv_heads_per_rank, block_size, 192]`` for
    BOTH K and V. V is only 128 wide, so the trailing 64 columns are written as
    zeros and sliced off after every gather. The framework derives a single
    ``head_size`` per layer from :class:`LayerSpec`, so a 128-wide V cache would
    require a second spec field that does not exist.
    """

    def __init__(self, config: MiMoV2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.is_swa = config.is_swa_layer(layer_idx)

        # <-- MODEL-SPECIFIC: per-layer full/SWA geometry
        if self.is_swa:
            self.num_attention_heads = config.swa_num_attention_heads
            self.num_key_value_heads = config.swa_num_key_value_heads
            self.head_dim = config.swa_head_dim
            self.v_head_dim = config.swa_v_head_dim
            self.sliding_window = config.sliding_window
            has_sink = config.add_swa_attention_sink_bias
        else:
            self.num_attention_heads = config.num_attention_heads
            self.num_key_value_heads = config.num_key_value_heads
            self.head_dim = config.head_dim
            self.v_head_dim = config.v_head_dim
            self.sliding_window = None
            has_sink = config.add_full_attention_sink_bias

        # HF: ``self.scaling = self.head_dim ** -0.5`` — the Q/K width (192),
        # NOT the V width. Using v_head_dim here would silently mis-scale every
        # logit by sqrt(192/128).
        self.scaling = self.head_dim**-0.5
        self.rope_dim = int(self.head_dim * config.partial_rotary_factor)
        self.v_scale = config.attention_value_scale

        # Single cache head width for both K and V (see class docstring).
        self.kv_cache_head_dim = max(self.head_dim, self.v_head_dim)

        # >>> PARALLELISM: TP head sharding <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        if self.num_attention_heads % self.world_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be "
                f"divisible by tp_degree ({self.world_size})"
            )
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size

        # KV replication: when TP >= kv_heads every rank keeps ONE replicated KV
        # head (consecutive groups of num_kv_replicas ranks share it) instead of
        # a zero-width slice from an integer division.
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

        # >>> PARALLELISM: fused QKV / O shapes for this rank <<<
        q_size = self.num_attention_heads_per_rank * self.head_dim
        k_size = self.num_key_value_heads_per_rank * self.head_dim
        v_size = self.num_key_value_heads_per_rank * self.v_head_dim
        self.q_size, self.k_size, self.v_size = q_size, k_size, v_size
        self.qkv_split_indices = [q_size, q_size + k_size]

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, q_size + k_size + v_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(
                self.num_attention_heads_per_rank * self.v_head_dim,
                self.hidden_size,
                dtype=self.dtype,
            )
        )

        # <-- MODEL-SPECIFIC: learned attention sink, one scalar per Q head.
        # nn.Parameter (not a buffer) because ``load_sharded_pipelined``
        # iterates ``named_parameters`` only — a buffer would never be loaded.
        self.attention_sink_bias = (
            nn.Parameter(
                torch.zeros(self.num_attention_heads_per_rank, dtype=torch.float32)
            )
            if has_sink
            else None
        )

        # Bound externally via bind_kv_cache().
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

        self.qkv_disk_tp = config.qkv_disk_tp
        self._setup_weight_loaders()

    def set_qkv_disk_tp(self, disk_tp: int) -> None:
        """Pin the fused-QKV on-disk shard count and reinstall its loader.

        Called from ``load_weights`` once the checkpoint has been probed; see
        :meth:`MiMoV2ForCausalLM._resolve_qkv_disk_tp` for why it cannot be
        known at construction time.
        """
        self.qkv_disk_tp = disk_tp
        self._install_qkv_loader()

    def _install_qkv_loader(self):
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_fp8_loader(
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                v_head_dim=self.v_head_dim,
                num_shards=self.world_size,
                num_kv_replicas=self.num_kv_replicas,
                dtype=self.dtype,
                disk_tp=self.qkv_disk_tp,
            ),
        )

    def _setup_weight_loaders(self):
        """Attach TP-sharding + FP8-dequantizing loaders to each parameter."""
        self._install_qkv_loader()
        # o_proj is in ``quantization_config.ignored_layers`` — stored as plain
        # bf16 ``[hidden, num_heads * v_head_dim]``, so the generic sharding
        # loader serves it (transposed storage, shard the input dim).
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=(self.num_attention_heads * self.v_head_dim)
                // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        if self.attention_sink_bias is not None:
            set_weight_loader(
                self.attention_sink_bias,
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=self.num_attention_heads_per_rank,
                    num_shards=self.world_size,
                ),
            )

    # ── Forward dispatch ─────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
    ) -> torch.Tensor:
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states, positions, position_embeddings, attn_metadata
            )

        # >>> PARALLELISM: all-gather out of SP before attention <<<
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(
            hidden_states, positions, position_embeddings, attn_metadata
        )

    # ── Shared QKV projection ────────────────────────────────────────────

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project, split, scale V, and apply partial RoPE.

        Returns ``(q, k, v)`` as ``[heads, T, dim]`` head-major tensors, with
        RoPE already applied to Q/K and ``attention_value_scale`` already
        folded into V (HF applies it FIRST in ``_forward_attention``, before
        the rope split, so the cached V is the scaled one).
        """
        tokens = hidden_states.shape[0]
        nqh = self.num_attention_heads_per_rank
        nkh = self.num_key_value_heads_per_rank

        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
        ).squeeze(0)
        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        q = q.view(tokens, nqh, self.head_dim).transpose(0, 1)
        k = k.view(tokens, nkh, self.head_dim).transpose(0, 1)
        v = v.view(tokens, nkh, self.v_head_dim).transpose(0, 1)

        # <-- MODEL-SPECIFIC: attention_value_scale, applied before caching.
        if self.v_scale is not None:
            v = v * self.v_scale

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, self.rope_dim)
        return q, k, v

    # ── Eager attention core ─────────────────────────────────────────────

    def _eager_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        additive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """fp32 eager attention with HF's exact sink handling.

        Mirrors ``eager_attention_forward`` in the HF modeling file: scaled
        scores, additive mask, concat the per-head sink as one extra logit
        column, subtract the row max, fp32 softmax, then DROP the sink column
        before the value matmul. The sink therefore removes probability mass
        from the real slots without contributing any value vector.

        Args:
            q: ``[B, nqh, S_q, head_dim]``.
            k: ``[B, nqh, S_kv, head_dim]`` (already GQA-expanded).
            v: ``[B, nqh, S_kv, v_head_dim]`` (already GQA-expanded).
            additive_mask: broadcastable to ``[B, nqh, S_q, S_kv]``; 0 where
                allowed, ``_MASK_NEG`` where masked.

        Returns:
            ``[B, nqh, S_q, v_head_dim]`` in the module dtype.
        """
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scaling
        scores = scores + additive_mask

        if self.attention_sink_bias is not None:
            B, nqh, S_q, _ = scores.shape
            sink = self.attention_sink_bias.float().reshape(1, nqh, 1, 1).expand(
                B, nqh, S_q, 1
            )
            scores = torch.cat([scores, sink], dim=-1)

        scores = scores - scores.max(dim=-1, keepdim=True).values
        probs = torch.softmax(scores, dim=-1)
        if self.attention_sink_bias is not None:
            probs = probs[..., :-1]

        return torch.matmul(probs, v.float()).to(self.dtype)

    def _out_proj(self, attn_out: torch.Tensor, tokens: int) -> torch.Tensor:
        """``[B, nqh, S, v_head_dim]`` -> ``[T, hidden]`` via ``NF.o_proj``.

        The 4-D ``[B, N, D, S]`` layout is used (rather than the 3-D one) so the
        NKI output-projection kernel is eligible: it needs ``D <= 128`` (V is
        exactly 128), ``N <= 17`` (1 head per rank at TP=64), and ``H % 2 == 0``.
        """
        B, nqh, S, vhd = attn_out.shape
        active = attn_out.permute(0, 1, 3, 2).contiguous()  # [B, N, D, S]
        out = NF.o_proj(active, self.o_proj_weight, None)  # [B, S, H]
        return out.reshape(tokens, self.hidden_size)

    # ── Paged KV cache write ─────────────────────────────────────────────

    def _write_paged_kv_cache(
        self, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor, block_size: int
    ) -> None:
        """Scatter post-RoPE K and (padded) V into the paged cache.

        ``k``/``v`` are head-major ``[nkh, T, dim]``. V is zero-padded from
        ``v_head_dim`` to ``kv_cache_head_dim`` because both caches share one
        ``head_size``; the padding is sliced off again after every gather.

        Delegates to ``NF.write_paged_kv_cache``, whose NKI kernel scatters the
        rows straight into the cache buffer. A plain ``index_put_`` here cost
        ~171 ms of the 199 ms decode step: XLA has no in-place scatter, so each
        of the 96 writes per step materialized a copy of the whole 322 MB pool,
        and because vLLM shares one raw cache tensor across ~6 layers the FX
        aliasing pass could only make the last write per buffer in-place.

        The kernel's aliased outputs are deliberately discarded (as llama3 does
        with ``NF.qkv_proj``'s): rebinding ``self.k_cache`` mid-forward would
        mutate module state under ``torch.compile``, and it isn't needed --
        ``AliasingOutputRewritePass`` finds the kernel's getitem outputs in the
        FX graph and rewires every downstream reader of the shared placeholder
        to the latest write itself.
        """
        pad = self.kv_cache_head_dim - self.v_head_dim

        k_flat = k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
        v_flat = v.reshape(-1, self.v_head_dim)
        if pad:
            v_flat = F.pad(v_flat, (0, pad))
        v_flat = v_flat.to(self.v_cache.dtype)

        NF.write_paged_kv_cache(
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            k=k_flat,
            v=v_flat,
            slot_mapping=slot_mapping,
            block_size=block_size,
            num_kv_heads=self.num_key_value_heads_per_rank,
        )

    def _gather_paged_kv(
        self, block_table: torch.Tensor, block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather the paged window into ``[B, nkh, S_ctx, dim]`` K and V.

        FP8 caches are dequantized to the compute dtype BEFORE the fancy index:
        indexing into a float8 tensor is not supported and gathers garbage on
        device. V is sliced back to ``v_head_dim`` here.
        """
        B, num_blocks_per_seq = block_table.shape
        S_ctx = num_blocks_per_seq * block_size
        nkh = self.num_key_value_heads_per_rank

        if self.k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            k_src = self.k_cache.to(self.dtype)
            v_src = self.v_cache.to(self.dtype)
        else:
            k_src, v_src = self.k_cache, self.v_cache

        flat_idx = block_table.reshape(-1)
        k_blocks = k_src[flat_idx].view(
            B, num_blocks_per_seq, nkh, block_size, self.kv_cache_head_dim
        )
        v_blocks = v_src[flat_idx].view(
            B, num_blocks_per_seq, nkh, block_size, self.kv_cache_head_dim
        )
        # .contiguous() guards the device's non-contiguous-reshape hazard.
        k_gathered = k_blocks.permute(0, 2, 1, 3, 4).reshape(
            B, nkh, S_ctx, self.kv_cache_head_dim
        )
        v_gathered = v_blocks.permute(0, 2, 1, 3, 4).reshape(
            B, nkh, S_ctx, self.kv_cache_head_dim
        )
        return k_gathered[..., : self.head_dim], v_gathered[..., : self.v_head_dim]

    # ── Prefill ──────────────────────────────────────────────────────────

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(self.dtype)
        tokens = hidden_states.shape[0]
        layer_name = f"layers.{self.layer_idx}.self_attn"
        meta = attn_metadata[layer_name]

        if meta.get("kv_segment_size"):
            # Chunked prefill / prefix caching would need the prior paged
            # window spliced with the live K/V (as the decode path does).
            # Not implemented: single-shot prefill is what the default
            # NeuronConfig produces (kv_segment_size_buckets is None -> 0).
            raise NotImplementedError(
                "MiMo-V2.5 does not support chunked prefill / prefix caching "
                "yet (kv_segment_size must be 0)."
            )

        q, k, v = self._project_qkv(hidden_states, position_embeddings)

        self._write_paged_kv_cache(k, v, meta["slot_mapping"], meta["block_size"])

        # Prefill runs one sequence at a time on Neuron.
        nqh = self.num_attention_heads_per_rank
        q = q.unsqueeze(0)  # [1, nqh, T, head_dim]
        k = _repeat_interleave_heads(
            k.unsqueeze(0), self.num_key_value_groups, dim=1
        )
        v = _repeat_interleave_heads(
            v.unsqueeze(0), self.num_key_value_groups, dim=1
        )

        mask = self._prefill_mask(tokens, hidden_states.device)
        attn_out = self._eager_attention(q, k, v, mask)

        attn_out = self._out_proj(attn_out, tokens)

        # >>> PARALLELISM: reduce-scatter back to SP <<<
        if self.world_size > 1:
            attn_out = self.tp_group.reduce_scatter(attn_out, dim=0)
        return attn_out.contiguous()

    def _prefill_mask(self, tokens: int, device: torch.device) -> torch.Tensor:
        """Additive causal (+ sliding-window) mask, ``[1, 1, T, T]``.

        Built from sequence-order indices rather than ``positions``: in a
        single-shot prefill the two agree for every real token, and the runner's
        right-padding repeats the last real position (which would make padded
        rows alias the last real row under a position-based mask). Padded rows'
        outputs are discarded downstream either way.
        """
        idx = torch.arange(tokens, device=device)
        allowed = idx.view(1, tokens) <= idx.view(tokens, 1)
        if self.sliding_window is not None:
            allowed = allowed & (
                idx.view(1, tokens) > idx.view(tokens, 1) - self.sliding_window
            )
        return torch.where(
            allowed,
            torch.zeros((), dtype=torch.float32, device=device),
            torch.full((), _MASK_NEG, dtype=torch.float32, device=device),
        ).view(1, 1, tokens, tokens)

    # ── Decode ───────────────────────────────────────────────────────────

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
    ) -> torch.Tensor:
        """Eager paged decode.

        Ordering is load-bearing: the prior window is gathered, the live K/V is
        spliced into that LOCAL copy, attention runs, and only THEN is the cache
        written. An ``index_put_`` followed by an indexed gather of the same
        tensor is not ordered under neuronx-cc/XLA, so a write-then-read would
        read the stale slot at the current token's own position — correct on CPU
        eager, corrupt on device.
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        meta = attn_metadata[layer_name]
        block_table = meta["block_table_tensor"]
        block_size = meta["block_size"]
        slot_mapping = meta["slot_mapping"]

        hidden_states = hidden_states.to(self.dtype)
        tokens = hidden_states.shape[0]
        B = block_table.shape[0]
        S_decode = tokens // B
        S_ctx = block_table.shape[1] * block_size
        nqh = self.num_attention_heads_per_rank
        nkh = self.num_key_value_heads_per_rank

        q, k, v = self._project_qkv(hidden_states, position_embeddings)

        k_gathered, v_gathered = self._gather_paged_kv(block_table, block_size)

        # Position of each active token inside the GATHERED window. For SWA the
        # runner trims the block table to the window-relevant blocks and reports
        # the trim origin as ``swa_kv_pos_offset`` (= start_block * block_size),
        # so both the splice index and the mask must live in that trimmed frame.
        # Mixing frames degenerates the window test once the trim activates.
        pos = positions.reshape(B, S_decode).long()
        if self.sliding_window is not None:
            offset = meta.get("swa_kv_pos_offset")
            if offset is not None:
                # clamp(min=0): a padding/freed row has its position zero-padded
                # to 0 while the offset is positive, so the shift would go
                # negative and mark the whole (all -1 block_table) window valid,
                # reading stale K -> NaN. Clamping collapses that row's window
                # to the self slot only.
                pos = torch.clamp(pos - offset.reshape(B, 1).long(), min=0)
        pos = torch.clamp(pos, max=S_ctx - 1)

        # Splice the live post-RoPE K/V into the LOCAL gathered window.
        k_live = k.reshape(nkh, B, S_decode, self.head_dim).permute(1, 0, 2, 3)
        v_live = v.reshape(nkh, B, S_decode, self.v_head_dim).permute(1, 0, 2, 3)
        k_idx = pos.view(B, 1, S_decode, 1).expand(B, nkh, S_decode, self.head_dim)
        v_idx = pos.view(B, 1, S_decode, 1).expand(B, nkh, S_decode, self.v_head_dim)
        k_gathered = k_gathered.scatter(2, k_idx, k_live.to(k_gathered.dtype))
        v_gathered = v_gathered.scatter(2, v_idx, v_live.to(v_gathered.dtype))

        k_gathered = _repeat_interleave_heads(
            k_gathered, self.num_key_value_groups, dim=1
        )
        v_gathered = _repeat_interleave_heads(
            v_gathered, self.num_key_value_groups, dim=1
        )

        q = q.reshape(nqh, B, S_decode, self.head_dim).permute(1, 0, 2, 3)
        mask = self._decode_mask(pos, S_ctx, hidden_states.device)
        attn_out = self._eager_attention(q, k_gathered, v_gathered, mask)

        attn_out = self._out_proj(attn_out, tokens)

        # Cache write AFTER attention (see method docstring). slot_mapping is in
        # the FULL cache frame, unaffected by any SWA block-table trim.
        self._write_paged_kv_cache(k, v, slot_mapping, block_size)

        # >>> PARALLELISM: TP all-reduce (no SP during decode) <<<
        if self.world_size > 1:
            attn_out = self.tp_group.all_reduce(attn_out)
        return attn_out

    def _decode_mask(
        self, pos: torch.Tensor, S_ctx: int, device: torch.device
    ) -> torch.Tensor:
        """Additive mask over the gathered window, ``[B, 1, S_decode, S_ctx]``.

        ``pos`` is each active token's slot in the gathered (possibly trimmed)
        frame; a query attends slots ``0..pos`` inclusive, further restricted to
        the trailing ``sliding_window`` slots on SWA layers.
        """
        B, S_decode = pos.shape
        slot = torch.arange(S_ctx, device=device).view(1, 1, 1, S_ctx)
        end = pos.view(B, 1, S_decode, 1)
        allowed = slot <= end
        if self.sliding_window is not None:
            start = torch.clamp(end - self.sliding_window + 1, min=0)
            allowed = allowed & (slot >= start)
        return torch.where(
            allowed,
            torch.zeros((), dtype=torch.float32, device=device),
            torch.full((), _MASK_NEG, dtype=torch.float32, device=device),
        )


# =============================================================================
# Section 4: Dense MLP (layer 0 only)
# =============================================================================


class MiMoV2DenseMLP(nn.Module):
    """SwiGLU MLP for the layers where ``moe_layer_freq[i]`` is 0.

    Only layer 0 in the released config. ``NF.mlp`` is used because it has a
    PyTorch fallback (CPU compile mode works) and fuses gate/up/down.
    """

    def __init__(self, config: MiMoV2Config):
        super().__init__()
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        if config.intermediate_size % self.world_size != 0:
            raise ValueError(
                f"intermediate_size ({config.intermediate_size}) must be "
                f"divisible by tp_degree ({self.world_size})"
            )
        self.intermediate_size_per_rank = config.intermediate_size // self.world_size

        i_pr = self.intermediate_size_per_rank
        self.gate_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, i_pr, dtype=self.dtype)
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, i_pr, dtype=self.dtype)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(i_pr, self.hidden_size, dtype=self.dtype)
        )

        self.quantized = config.is_fp8_checkpoint
        self._install_loaders()

    def set_quantized(self, quantized: bool) -> None:
        """Switch between the fp8-dequantizing and plain-bf16 loaders."""
        if quantized == self.quantized:
            return
        self.quantized = quantized
        self._install_loaders()

    def _install_loaders(self):
        i_pr = self.intermediate_size_per_rank
        quantized = self.quantized
        set_weight_loader(
            self.gate_proj_weight,
            dense_gate_up_fp8_loader(i_pr, self.world_size, self.dtype)
            if quantized
            else _plain_transposed_shard_loader(i_pr, self.world_size, shard_rows=True),
        )
        set_weight_loader(
            self.up_proj_weight,
            dense_gate_up_fp8_loader(i_pr, self.world_size, self.dtype)
            if quantized
            else _plain_transposed_shard_loader(i_pr, self.world_size, shard_rows=True),
        )
        set_weight_loader(
            self.down_proj_weight,
            dense_down_fp8_loader(i_pr, self.world_size, self.dtype)
            if quantized
            else _plain_transposed_shard_loader(
                i_pr, self.world_size, shard_rows=False
            ),
        )

    def forward(
        self, hidden_states: torch.Tensor, is_decode: bool
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(self.dtype)

        # >>> PARALLELISM: all-gather out of SP before the column-parallel matmul
        # <<< In prefill the residual stream is sequence-parallel (T/tp rows per
        # rank), but a column-then-row-parallel MLP needs the FULL sequence: each
        # rank computes a partial sum over its intermediate slice, and the
        # reduce_scatter below both completes that sum and returns to the SP
        # layout. Without the gather, reduce_scatter gets T/tp rows and asserts on
        # ``shape[0] % world_size`` (T/tp**2 is not integral). Same shape contract
        # as MiMoV2Attention.forward and MiMoV2SparseMoeBlock.forward_prefill.
        if not is_decode and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        out = NF.mlp(
            hidden=hidden_states,
            gate_w=self.gate_proj_weight,
            up_w=self.up_proj_weight,
            down_w=self.down_proj_weight,
            act_fn=ActFnType.SiLU,
        )
        if self.world_size > 1:
            # Column-then-row parallel: every rank holds a partial sum over the
            # intermediate dim. Decode all-reduces; prefill reduce-scatters
            # straight back into the SP layout.
            if is_decode:
                out = self.tp_group.all_reduce(out)
            else:
                out = self.tp_group.reduce_scatter(out, dim=0)
        return out.to(self.dtype)


def _plain_transposed_shard_loader(
    shard_size: int, num_shards: int, shard_rows: bool
) -> SafetensorsWeightLoader:
    """BF16 fallback loader for a dense MLP projection.

    Used only for an unquantized re-export of the checkpoint (the released one
    is fp8, handled by the dequantizing loaders). HF stores ``[out, in]``; our
    params are ``[in, out]``. ``shard_rows`` selects gate/up (shard HF dim 0)
    vs down (shard HF dim 1).
    """
    return sharding_weight_loader(
        shard_dim=1 if shard_rows else 0,
        shard_size=shard_size,
        num_shards=num_shards,
        is_storage_transposed=True,
    )


# =============================================================================
# Section 5: Sparse MoE (sigmoid noaux_tc router + static-block NF kernels)
# =============================================================================


class MiMoV2SparseMoeBlock(nn.Module):
    """256-expert top-8 MoE with EP, and a hand-written ``noaux_tc`` router.

    The router CANNOT use ``NF.router`` or the fused ``moe_block_tkg`` router:
    neither can express MiMo's group-limited selection
    (``group_score = top2(scores_for_choice).sum()`` per group, keep
    ``topk_group`` groups) nor the bias split — ``e_score_correction_bias`` is
    added for SELECTION only, while the returned weights are gathered from the
    UNBIASED sigmoid scores. So routing is computed in fp32 torch and only the
    expert MLPs go through the NKI kernels
    (``moe_cte`` for prefill, ``moe_tkg`` for decode).

    With the released config (``n_group=1``, ``topk_group=1``) the group mask is
    all ones and selection degenerates to a plain top-8 over the biased scores,
    but the general form is implemented so a grouped config still works.
    """

    def __init__(self, config: MiMoV2Config):
        super().__init__()
        from vllm.config import get_current_vllm_config
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_ep_degree,
            get_neuron_ep_rank,
            get_neuron_ep_tp_group,
        )

        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.tp_group = get_tp_group()

        self.ep_enabled = (
            get_current_vllm_config().parallel_config.enable_expert_parallel
        )
        self.ep_degree = get_neuron_ep_degree() if self.ep_enabled else 1
        self.ep_rank = get_neuron_ep_rank() if self.ep_enabled else 0
        self.ep_tp_group = (
            get_neuron_ep_tp_group() if self.ep_enabled else self.tp_group
        )
        # Intermediate-dim sharding degree within an EP partition (1 for pure EP).
        self.tp_degree = self.tp_group.world_size // self.ep_degree
        # Outer collective group: the full TP world in both TP-only and EP modes.
        self.moe_group = self.tp_group

        self.total_num_experts = config.n_routed_experts
        if self.total_num_experts % self.ep_degree != 0:
            raise ValueError(
                f"n_routed_experts ({self.total_num_experts}) must be divisible "
                f"by ep_degree ({self.ep_degree})"
            )
        self.num_local_experts = self.total_num_experts // self.ep_degree
        self.top_k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = (
            config.routed_scaling_factor
            if config.routed_scaling_factor is not None
            else 1.0
        )
        if config.moe_intermediate_size % self.tp_degree != 0:
            raise ValueError(
                f"moe_intermediate_size ({config.moe_intermediate_size}) must "
                f"be divisible by moe tp_degree ({self.tp_degree})"
            )
        self.intermediate_size_per_rank = (
            config.moe_intermediate_size // self.tp_degree
        )
        # Baseline MoE dispatch block size for decode, where a single token
        # produces the minimum block count regardless. Prefill derives its own
        # value from the token count -- see ``_prefill_block_size``.
        self.block_size = int(os.environ.get("VLLM_NEURON_MOE_BLOCK_SIZE", 256))

        # Router weight + correction bias are fp32 and REPLICATED (never
        # EP-sharded): every rank scores all experts, then keeps its own slice.
        self.router_weight = nn.Parameter(
            torch.empty(self.total_num_experts, self.hidden_size, dtype=torch.float32)
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.total_num_experts, dtype=torch.float32)
        )

        self.gate_up_proj_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_size,
                self.intermediate_size_per_rank * 2,
                dtype=self.dtype,
            )
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.intermediate_size_per_rank,
                self.hidden_size,
                dtype=self.dtype,
            )
        )

        self.quantized = config.is_fp8_checkpoint
        self._setup_weight_loaders()

    def set_quantized(self, quantized: bool) -> None:
        """Switch between the fp8-dequantizing and plain-bf16 expert loaders."""
        if quantized == self.quantized:
            return
        self.quantized = quantized
        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        quantized = self.quantized
        local_expert_indices = list(
            range(
                self.ep_rank * self.num_local_experts,
                (self.ep_rank + 1) * self.num_local_experts,
            )
        )

        def _maybe_ep_wrap(loader):
            # MiMo ships one key per expert, so the checkpoint mapping is a flat
            # INTERLEAVED list ([e0_gate_w, e0_gate_s, e0_up_w, e0_up_s, ...])
            # and the interleaved EP wrapper slices it to this rank's experts
            # before the transform ever runs.
            if self.ep_degree > 1:
                return expert_parallel_interleaved_loader(
                    local_expert_indices, loader, self.total_num_experts
                )
            return loader

        set_weight_loader(
            self.gate_up_proj_weight,
            _maybe_ep_wrap(
                expert_gate_up_fp8_loader(
                    shard_size=self.intermediate_size_per_rank * 2,
                    num_shards=self.tp_degree,
                    quantized=quantized,
                    dtype=self.dtype,
                )
            ),
        )
        set_weight_loader(
            self.down_proj_weight,
            _maybe_ep_wrap(
                expert_down_fp8_loader(
                    shard_size=self.intermediate_size_per_rank,
                    num_shards=self.tp_degree,
                    quantized=quantized,
                    dtype=self.dtype,
                )
            ),
        )

    # ── Router (noaux_tc, fp32 torch) ────────────────────────────────────

    def _route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """MiMo's ``noaux_tc`` routing, verbatim from ``MiMoV2MoEGate``.

        Returns ``(expert_affinities, expert_index)`` where affinities is a
        dense ``[T, E]`` fp32 tensor holding the top-k weights scattered into
        their expert slots (zeros elsewhere), and ``expert_index`` is
        ``[T, top_k]`` int32 global expert ids.
        """
        x = hidden_states.float()
        logits = F.linear(x, self.router_weight)  # [T, E]
        scores = logits.sigmoid()

        # The bias steers SELECTION only; the returned weight is the UNBIASED
        # score. Folding the bias into the weight is a silent accuracy bug.
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)

        if self.n_group > 1:
            T = scores.shape[0]
            group_scores = (
                _topk_last_dim(scores_for_choice.view(T, self.n_group, -1), 2)[0]
                .sum(dim=-1)
            )  # [T, n_group]
            group_idx = _topk_last_dim(group_scores, self.topk_group)[1]
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1.0)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(T, self.n_group, scores.shape[-1] // self.n_group)
                .reshape(T, -1)
            )
            scores_for_choice = scores_for_choice.masked_fill(
                ~score_mask.bool(), float("-inf")
            )

        _, topk_idx = _topk_last_dim(scores_for_choice, self.top_k)
        topk_weight = scores.gather(1, topk_idx)
        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (
                topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            )
        topk_weight = topk_weight * self.routed_scaling_factor

        affinities = torch.zeros_like(scores).scatter(1, topk_idx, topk_weight)
        return affinities, topk_idx.to(torch.int32)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        is_decode: bool,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if is_decode:
            return self.forward_decode(hidden_states, rank)
        return self.forward_prefill(hidden_states, positions, rank)

    def _ep_rank_tensor(self, rank: torch.Tensor | None, device) -> torch.Tensor:
        """This rank's EP partition index, as a TRACED int32 scalar tensor.

        ``self.ep_rank`` (a Python int captured at construction) must NOT reach
        the graph: ``vllm_neuron.compile.backend`` sets ``per_rank=False``, so a
        single NEFF is compiled once and executed on every TP rank. Baking in the
        constructing rank's value would make all 64 ranks read rank 0's expert
        window. The runner threads the TP-local rank in as a tensor input
        (``neuron_model_runner.py`` ``self.rank_tensor``) precisely so this stays
        dynamic; ``self.ep_rank`` is only for host-side weight loading, which
        genuinely does run once per process.
        """
        if rank is None:
            # CPU/component-test path: no rank input, single process.
            return torch.tensor(self.ep_rank, dtype=torch.int32, device=device)
        if self.ep_degree == 1:
            return torch.zeros_like(rank).to(torch.int32)
        return (
            (rank % (self.ep_degree * self.tp_degree)) // self.tp_degree
        ).to(torch.int32)

    def _local_expert_slice(
        self, affinities: torch.Tensor, rank: torch.Tensor | None
    ) -> torch.Tensor:
        """``[T, E]`` -> ``[T, E_local]`` for this EP rank."""
        if self.ep_degree == 1:
            return affinities
        ep_rank = self._ep_rank_tensor(rank, affinities.device)
        local_expert_indices = (
            torch.arange(
                self.num_local_experts,
                device=affinities.device,
                dtype=torch.int32,
            )
            + ep_rank * self.num_local_experts
        )
        return NF.get_local_expert_affinities(affinities, local_expert_indices)

    def _gate_up_kernel_view(self) -> torch.Tensor:
        return self.gate_up_proj_weight.reshape(
            self.num_local_experts,
            self.hidden_size,
            2,
            self.intermediate_size_per_rank,
        )

    def _prefill_block_size(self, num_tokens: int) -> int:
        """MoE dispatch block size for a ``num_tokens``-long prefill.

        The ``shard_on_block`` kernel faults on device with a "scatter/gather
        (indirect memory copy via vector DGE) out-of-bound access" once the block
        count grows past a threshold. Measured at TP=64/EP=64 (E_local=4,
        top_k=8, so ``min(top_k, E_local) = 4`` local slots per token) with

            N = ceil((T*4 - (E_local-1)) / block_size) + E_local - 1

        | T    | block_size | N  | result |
        |------|-----------:|---:|--------|
        | 512  |        256 | 11 | ok     |
        | 1024 |        512 | 11 | ok     |
        | 1024 |        256 | 19 | FAULT  |
        | 1024 |        128 | 35 | FAULT  |

        N=11 is clean at both sequence lengths while N>=19 faults, and N=35
        (whose per-shard count 18 IS divisible by the kernel's
        BLOCK_PARALLEL_FACTOR of 3) faults too -- so the trigger is the block
        count itself, not a divisibility condition on it. Solving N <= 11 for the
        4-local-expert case gives ``block_size >= T/2``, hence the T//2 floor.
        Forcing the torch mapping path (which is what builds the block table)
        still faults, and forcing the torch MoE compute makes the fault vanish,
        which is what localizes this to the dispatch kernel rather than to the
        mapping or to attention.

        The cost is real: a bigger block means more padding waste inside each
        block, since a block holds one expert's tokens only. It is bounded
        though, because ``skip_token=True`` turns padded rows' gathers into
        no-op DMAs rather than compute.
        """
        # ceil(T/2), rounded up to a power of two so the block size stays a
        # clean DMA granularity (T is a power-of-two bucket in practice, making
        # this exactly T//2).
        half = max(1, (num_tokens + 1) // 2)
        return max(self.block_size, 1 << (half - 1).bit_length())

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(self.dtype)

        # Routing runs on this rank's SP shard, then affinities and hidden are
        # gathered to the full sequence for the blockwise dispatch.
        affinities = self._route(hidden_states)[0]
        if self.tp_group.world_size > 1:
            affinities = self.tp_group.all_gather(affinities, dim=0)
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        # Padding mask (True = real token). The runner right-pads the prompt to
        # the bucket length with tokens whose positions REPEAT the last real
        # position, so argmax finds the last real index. Without this mask the
        # padding tokens are routed to experts, inflating the per-expert block
        # count until the dispatch gather overruns the statically sized buffers
        # (nrta-1006 at execute, invisible during warmup where every row is real).
        padding_mask = None
        if positions is not None:
            last_real_idx = torch.argmax(positions)
            token_indices = torch.arange(positions.shape[0], device=positions.device)
            padding_mask = token_indices <= last_real_idx

        affinities = self._local_expert_slice(affinities, rank)

        # Sized from the (static, post-all-gather) token count so the dispatch
        # kernel's block count stays inside the range it can address; see
        # _prefill_block_size.
        num_tokens = hidden_states.shape[0]
        block_size = self._prefill_block_size(num_tokens)

        (
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
            conditions,
        ) = NF.build_blockwise_mapping(
            expert_affinities=affinities,
            num_local_experts=self.num_local_experts,
            num_experts_per_token=self.top_k,
            block_size=block_size,
            moe_group=self.ep_tp_group,
            tp_degree=self.tp_degree,
            padding_mask=padding_mask,
        )

        num_static_block = math.ceil(
            num_tokens * self.top_k / self.ep_degree / block_size
        )
        # Bound the dispatch gather index to the valid hidden_states rows while
        # PRESERVING the -1 padding sentinel that skip_token consumes.
        token_position_to_id = token_position_to_id.clamp(min=-1, max=num_tokens - 1)

        output = NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_block,
            conditions=conditions,
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=self._gate_up_kernel_view(),
            down_proj_weight=self.down_proj_weight,
            activation_function=ActFnType.SiLU,
            # Must match the mapping's block_size exactly: the kernel recovers
            # N as len(token_position_to_id) // block_size, so a mismatch
            # silently reinterprets the block table.
            block_size=block_size,
            token_position_to_id=token_position_to_id.to(dtype=torch.int32),
            block_to_expert=block_to_expert.to(dtype=torch.int32),
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            # skip_weight=True is REQUIRED, not an optimization: shard_on_block
            # over-allocates blocks (padding by +E-1 for per-expert rounding) and
            # internally memsets the trailing padding blocks' expert ids to E —
            # one past the valid [0, E-1] weight rows. With skip_weight=False
            # that padding-block weight DMA runs in oob_mode.error and faults
            # (nrta-1006); skip_weight flips it to oob_mode.skip.
            skip_weight=True,
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
            num_static_block=num_static_block,
        )

        if self.moe_group.world_size > 1:
            output = self.moe_group.reduce_scatter(output, dim=0)
        # The fp32 router affinities promote the POST_SCALE multiply in the
        # torch/meta path to fp32; cast back so the residual stream stays bf16.
        return output.to(self.dtype)

    def forward_decode(
        self, hidden_states: torch.Tensor, rank: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(self.dtype)
        affinities, expert_index = self._route(hidden_states)

        if can_run_kernel(hidden_states):
            rank_id = self._ep_rank_tensor(rank, hidden_states.device).reshape(1, 1)
            # Static all-expert dispatch, deliberately. The two alternatives were
            # both measured on device and both lose:
            #
            #   - ``is_all_expert=False`` (routed/selective) indexes the LOCAL
            #     [E_L, H, 2, I] weights with the GLOBAL expert id and takes no
            #     rank_id, so it is EP-incompatible by construction (which is why
            #     gpt_oss/model_bf16.py force-enables all-expert when ep_degree>1).
            #     It also does K=8 experts of work per token against all-expert's
            #     E_local=4, and does not compile at these dims.
            #   - ``is_all_expert_dynamic=True`` skips token blocks with no routed
            #     tokens, and in an isolated single-kernel microbenchmark that
            #     looked like a 1.2-1.55x win at T>=16 (bit-exact). END TO END IT
            #     IS SLOWER: median ITL 160.6/152.9/148.1 ms at concurrency
            #     1/16/32 versus 147.2/147.2/147.1 static, same host, same
            #     recompile. The microbenchmark measured one kernel per host round
            #     trip; in the real graph these are 47 back-to-back invocations
            #     and the data-dependent control flow (nl.dynamic_range +
            #     _find_routed_tokens) costs more than the skipping saves.
            output = NF.moe_tkg(
                hidden_input=hidden_states,
                expert_gate_up_weights=self._gate_up_kernel_view(),
                expert_down_weights=self.down_proj_weight,
                # Full [T, E] affinities; the kernel slices to this rank's
                # [T, E_local] window using rank_id.
                expert_affinities=affinities.to(self.dtype),
                expert_index=expert_index,
                is_all_expert=True,
                rank_id=rank_id,
                mask_unselected_experts=True,
                expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
                activation_fn=ActFnType.SiLU,
                output_dtype=self.dtype,
            )
        else:
            # ``NF.moe_tkg`` has no PyTorch fallback (it raises off-device), but
            # CPU compile / component tests must still run the block.
            output = self._experts_torch(hidden_states, affinities)

        if self.moe_group.world_size > 1:
            output = self.moe_group.all_reduce(output)
        return output.to(self.dtype)

    def _experts_torch(
        self, hidden_states: torch.Tensor, affinities: torch.Tensor
    ) -> torch.Tensor:
        """Dense all-local-expert reference for CPU mode.

        Equivalent to ``moe_tkg(is_all_expert=True, POST_SCALE)``: run every
        local expert over every token and accumulate weighted by that token's
        affinity for the expert (zero for unselected experts, so they drop out).
        """
        lo = self.ep_rank * self.num_local_experts
        local_aff = affinities[:, lo : lo + self.num_local_experts].to(torch.float32)
        gate_up = self._gate_up_kernel_view()  # [E_L, H, 2, I]
        out = torch.zeros(
            hidden_states.shape[0],
            self.hidden_size,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        for e in range(self.num_local_experts):
            gate = hidden_states @ gate_up[e, :, 0, :]
            up = hidden_states @ gate_up[e, :, 1, :]
            expert_out = (F.silu(gate.float()) * up.float()) @ self.down_proj_weight[
                e
            ].float()
            out = out + expert_out * local_aff[:, e : e + 1]
        return out


# =============================================================================
# Section 6: Decoder Layer
# =============================================================================


class MiMoV2DecoderLayer(nn.Module):
    """``norm -> attn -> +residual -> norm -> mlp -> +residual``.

    Both layernorms are EXPLICIT members of the layer (they are not folded into
    the MoE block, unlike the Qwen3.5 port) and both use the plain
    :class:`MiMoV2RMSNorm`, matching ``MiMoV2DecoderLayer`` in HF.
    """

    def __init__(self, config: MiMoV2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_moe = config.is_moe_layer(layer_idx)

        self.input_layernorm = MiMoV2RMSNorm(
            config.hidden_size, config.layernorm_epsilon, config.torch_dtype
        )
        self.self_attn = MiMoV2Attention(config, layer_idx)
        self.post_attention_layernorm = MiMoV2RMSNorm(
            config.hidden_size, config.layernorm_epsilon, config.torch_dtype
        )
        self.mlp = (
            MiMoV2SparseMoeBlock(config) if self.is_moe else MiMoV2DenseMLP(config)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
        is_prefill: bool,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            hidden_states = self.mlp(hidden_states, positions, not is_prefill, rank)
        else:
            hidden_states = self.mlp(hidden_states, not is_prefill)
        return residual + hidden_states


# =============================================================================
# Section 7: Backbone
# =============================================================================


class MiMoV2Model(nn.Module):
    """Embedding + hybrid decoder stack + final norm.

    >>> PARALLELISM: SP <<<
    Prefill scatters the sequence across TP ranks right out of the embedding
    (``scatter_tokens=True``) and all-gathers after the final norm; decode runs
    every token on every rank.
    """

    def __init__(self, config: MiMoV2Config):
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
        self.layers = nn.ModuleList(
            [
                MiMoV2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = MiMoV2RMSNorm(
            config.hidden_size, config.layernorm_epsilon, config.torch_dtype
        )

        # <-- MODEL-SPECIFIC: two rotary caches, one per RoPE base.
        self.rotary_emb = MiMoV2RotaryEmbedding(config, is_swa=False)
        self.swa_rotary_emb = MiMoV2RotaryEmbedding(config, is_swa=True)

        # No weight loader is set here: VocabDimShardedEmbedding installs its own
        # (shard_dim=0, shard_size=vocab_size_per_rank, pad_shard=True), which is
        # what we want -- vocab_size_per_rank is a ceil, so pad_shard matters for
        # any TP degree that does not divide 152576 evenly.

        # Index of the first full-attention layer; its metadata carries the
        # untrimmed block table. Layer 0 is full in the released config, but
        # read it from the pattern rather than assuming.
        self.first_full_layer_idx = next(
            (i for i in range(config.num_hidden_layers) if not config.is_swa_layer(i)),
            0,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: dict,
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layer_name = f"layers.{self.first_full_layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
        is_prefill = max_query_len > decode_token_threshold

        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
        )

        # >>> PARALLELISM: shard prompt embeds to match the SP layout <<<
        if (
            is_prefill
            and self.world_size > 1
            and inputs_embeds is not None
            and is_token_ids is not None
        ):
            local_len = hidden_states.shape[0]
            start = self.rank * local_len
            inputs_embeds = inputs_embeds[start : start + local_len]
            is_token_ids = is_token_ids[start : start + local_len]

        hidden_states = NF.merge_prompt_embeds(
            hidden_states, inputs_embeds, is_token_ids
        )

        full_pe = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )
        swa_pe = self.swa_rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                positions=positions,
                position_embeddings=swa_pe if layer.self_attn.is_swa else full_pe,
                attn_metadata=attn_metadata,
                is_prefill=is_prefill,
                rank=rank,
            )

        hidden_states = self.norm(hidden_states)

        # >>> PARALLELISM: SP -> full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return hidden_states


# =============================================================================
# Section 8: Language Model Head
# =============================================================================


class MiMoV2ForCausalLM(nn.Module):
    """MiMo-V2.5 with a column-parallel LM head and on-device sampling."""

    def __init__(self, config: MiMoV2Config):
        super().__init__()
        self.config = config
        self.model = MiMoV2Model(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        nc = config.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        debug_logits_enabled = nc is not None and nc.debug_logits_dir is not None
        self._gather_logits = (
            nc is not None and nc.max_logprobs != 0
        ) or debug_logits_enabled

        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )
        # ColumnParallelLinear installs its own row-sharding loader; no override.

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=self.tp_group.device_group,
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: dict | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        positions = positions.to(torch.int32)

        layer_name = f"layers.{self.model.first_full_layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
        is_prefill = max_query_len > decode_token_threshold

        T = input_ids.shape[0]
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt length ({T}) must be > world_size ({self.world_size}) "
                f"and divisible by it for sequence parallelism."
            )

        hidden_states = self.model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
        )

        hidden_states_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )
        hidden_states_for_logits = hidden_states_for_logits.to(self.config.torch_dtype)
        logits = self.lm_head(hidden_states_for_logits)

        if self.on_device_sampling_config is None:
            return logits

        gathered_logits = None
        if self._gather_logits:
            gathered_logits = self.tp_group.all_gather(logits, dim=1)

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )

        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            return rejection_sampler(spec_decode_metadata, sampled_tokens)

        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = MiMoV2Config.from_configs(hf_config, neuron_config)
        _validate_parallelism(neuron_config)
        return cls(config)

    # ── KV cache ─────────────────────────────────────────────────────────

    def get_kv_spec(self) -> KVSpec:
        """One spec per layer; SWA layers declare their window.

        ``head_size`` is the Q/K width (192) for BOTH caches — see
        :class:`MiMoV2Attention` on why the 128-wide V is padded rather than
        given its own size.
        """
        layers = []
        for i, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            layers.append(
                LayerSpec(
                    name=f"layers.{i}.self_attn",
                    num_kv_heads=attn.num_key_value_heads_per_rank,
                    head_size=attn.kv_cache_head_dim,
                    dtype=attn.dtype,
                    sliding_window_size=attn.sliding_window,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise KeyError(f"KV cache for layer {layer_name} not initialized")
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # ── Weight loading ───────────────────────────────────────────────────

    def _build_mappings(self, quantized: bool | None = None) -> dict:
        """Map our parameter names to checkpoint keys.

        The released checkpoint pairs every quantized weight with a
        ``.weight_scale_inv`` companion, so those entries are 2-element lists
        that the dequantizing loaders consume; an unquantized re-export drops
        to 1-element lists and the same loaders take their bf16 branch.
        MoE experts are separate keys per expert, listed INTERLEAVED
        (``e0`` items, then ``e1`` items, ...) to match
        ``expert_parallel_interleaved_loader``.

        Args:
            quantized: whether the checkpoint carries ``*_scale_inv`` companions.
                ``None`` falls back to the config's ``quantization_config``,
                which is absent when the serving path strips it (see
                :meth:`_detect_quantized`).
        """
        cfg = self.config
        if quantized is None:
            quantized = cfg.is_fp8_checkpoint

        def q(key: str) -> str | list[str]:
            """Checkpoint entry for a possibly-quantized weight."""
            return [key, f"{key}_scale_inv"] if quantized else key

        mappings: dict = {
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            "model.norm.weight": "model.norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

        for i in range(cfg.num_hidden_layers):
            hf = f"model.layers.{i}"
            ours = f"model.layers.{i}"

            mappings[f"{ours}.input_layernorm.weight"] = f"{hf}.input_layernorm.weight"
            mappings[f"{ours}.post_attention_layernorm.weight"] = (
                f"{hf}.post_attention_layernorm.weight"
            )

            # Fused QKV: ONE pre-sharded tensor on disk (+ its scale grid).
            mappings[f"{ours}.self_attn.qkv_proj_weight"] = q(
                f"{hf}.self_attn.qkv_proj.weight"
            )
            # o_proj is in ignored_layers -> plain bf16, no scale companion.
            mappings[f"{ours}.self_attn.o_proj_weight"] = f"{hf}.self_attn.o_proj.weight"
            if self.model.layers[i].self_attn.attention_sink_bias is not None:
                mappings[f"{ours}.self_attn.attention_sink_bias"] = (
                    f"{hf}.self_attn.attention_sink_bias"
                )

            if cfg.is_moe_layer(i):
                mappings[f"{ours}.mlp.router_weight"] = f"{hf}.mlp.gate.weight"
                mappings[f"{ours}.mlp.e_score_correction_bias"] = (
                    f"{hf}.mlp.gate.e_score_correction_bias"
                )
                gate_up: list[str] = []
                down: list[str] = []
                for e in range(cfg.n_routed_experts):
                    ep = f"{hf}.mlp.experts.{e}"
                    for proj in ("gate_proj", "up_proj"):
                        key = f"{ep}.{proj}.weight"
                        gate_up.append(key)
                        if quantized:
                            gate_up.append(f"{key}_scale_inv")
                    key = f"{ep}.down_proj.weight"
                    down.append(key)
                    if quantized:
                        down.append(f"{key}_scale_inv")
                mappings[f"{ours}.mlp.gate_up_proj_weight"] = gate_up
                mappings[f"{ours}.mlp.down_proj_weight"] = down
            else:
                mappings[f"{ours}.mlp.gate_proj_weight"] = q(f"{hf}.mlp.gate_proj.weight")
                mappings[f"{ours}.mlp.up_proj_weight"] = q(f"{hf}.mlp.up_proj.weight")
                mappings[f"{ours}.mlp.down_proj_weight"] = q(f"{hf}.mlp.down_proj.weight")

        return mappings

    def _detect_quantized(self, checkpoint: SafetensorsCheckpoint) -> bool:
        """Decide fp8-vs-bf16 from the CHECKPOINT, not from ``config.json``.

        vLLM auto-derives ``quantization`` from ``hf_config.quantization_config``
        and then rejects "fp8" against the platform's allowlist
        (``neuron_quant`` / ``compressed-tensors`` / ``modelopt``), so the serving
        path has to strip that key via ``hf_overrides``. Deriving the loader
        branch from the same stripped config would then silently pick the bf16
        branch and read raw fp8 bytes as bf16 -- garbage, with no error. The
        presence of a ``*_scale_inv`` companion is the ground truth.

        Nothing else in the load path needs the stripped key: the 128x128 tile
        shape is the loaders' own constant, and ``o_proj``'s unquantized-ness is
        expressed structurally (its mapping never carries a scale companion)
        rather than by consulting ``ignored_layers``.
        """
        checkpoint._ensure_indexed()
        names = checkpoint.get_tensor_names()
        probe = (
            f"model.layers.{self.model.first_full_layer_idx}"
            ".self_attn.qkv_proj.weight_scale_inv"
        )
        quantized = probe in names
        if quantized != self.config.is_fp8_checkpoint:
            logger.info(
                "MiMo-V2.5: checkpoint fp8=%s (probed %s), config said %s; "
                "trusting the checkpoint.",
                quantized,
                probe,
                self.config.is_fp8_checkpoint,
            )
        # No-op where the branch already matches, so this is unconditional.
        for layer in self.model.layers:
            layer.mlp.set_quantized(quantized)
        return quantized

    def _resolve_qkv_disk_tp(
        self, checkpoint: SafetensorsCheckpoint, quantized: bool
    ) -> None:
        """Pin the fused-QKV on-disk shard count, then reinstall the qkv loaders.

        ``from_configs`` never receives the checkpoint path, so this cannot be
        settled at construction; it has to happen here, before the first slice is
        read. A FULL-attention layer's grid is unambiguous (nkv=4 -> the K and V
        sections ceil-pad, so 4 shards give 108 scale rows where 1 gives 106),
        whereas an SWA layer's is not (nkv=8 -> 116 rows for 1, 2 and 4 alike).
        Probe a full layer, then hand the answer to every layer's loader.

        An explicit ``config.qkv_disk_tp`` wins, and a bf16 re-export has no
        scale grid at all to probe, so both short-circuit.
        """
        if not quantized:
            return

        disk_tp = self.config.qkv_disk_tp
        if disk_tp is None:
            probe_layer = self.model.first_full_layer_idx
            attn = self.model.layers[probe_layer].self_attn
            if attn.is_swa:
                # Every layer is SWA: nothing to probe. Leave the loaders on
                # per-tensor inference, which raises rather than guessing.
                logger.warning(
                    "MiMo-V2.5: no full-attention layer to probe for the "
                    "fused-QKV disk shard count; set config.qkv_disk_tp if "
                    "loading raises on an ambiguous scale grid."
                )
                return
            checkpoint._ensure_indexed()
            scale_name = (
                f"model.layers.{probe_layer}.self_attn.qkv_proj.weight_scale_inv"
            )
            scale_rows = int(checkpoint._get_slice(scale_name).get_shape()[0])
            disk_tp = infer_qkv_disk_tp(
                scale_rows,
                attn.num_attention_heads,
                attn.num_key_value_heads,
                attn.head_dim,
                attn.v_head_dim,
            )
            logger.info(
                "MiMo-V2.5: fused-QKV disk shard count = %d "
                "(probed layer %d, %d scale rows)",
                disk_tp,
                probe_layer,
                scale_rows,
            )

        for layer in self.model.layers:
            layer.self_attn.set_qkv_disk_tp(disk_tp)

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        quantized = self._detect_quantized(checkpoint)
        logger.info(
            "MiMo-V2.5 load_weights: rank=%d world_size=%d fp8_checkpoint=%s",
            self.rank,
            self.world_size,
            quantized,
        )
        self._resolve_qkv_disk_tp(checkpoint, quantized)
        state_dict = checkpoint.load_sharded_pipelined(
            self.rank,
            self.world_size,
            self,
            self._build_mappings(quantized=quantized),
            device,
        ).state_dict
        self.load_state_dict(state_dict, strict=False, assign=True)

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Lightweight path used during CPU compile (index only, no tensors)."""
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        checkpoint._ensure_indexed()


def _validate_parallelism(neuron_config: NeuronConfig | None) -> None:
    """Reject the DP modes this port has not been wired for.

    Attention/MLP/embedding/LM-head data parallelism needs per-module DP group
    transitions and a DP-aware effective rank in every weight loader (see
    ``gpt_oss/model_bf16.py``). None of that is implemented here, and silently
    running with DP > 1 would shard weights against the wrong rank.
    """
    if neuron_config is None:
        return
    for field in (
        "attention_dp_size",
        "mlp_dp_size",
        "embedding_dp_size",
        "lm_head_dp_size",
    ):
        size = getattr(neuron_config, field, 1) or 1
        if size > 1:
            raise ValueError(
                f"MiMo-V2.5 does not support {field}={size}; only TP + SP + EP "
                f"are implemented. Set {field}=1."
            )
