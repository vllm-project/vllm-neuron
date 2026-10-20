# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL MXFP8 text decoder: CPU-dequant baseline + on-device MX attention.

The LLM-Compressor MXFP8 checkpoint stores the seven text-decoder Linears
(q/k/v/o/gate/up/down) as F8_E4M3 ``.weight`` + U8 (E8M0) ``.weight_scale``.
``neuron_config.modules_to_not_convert`` (the skip list) selects, per projection,
whether it is CPU-dequanted to BF16 or kept as native on-device MX weights:

  - ``["self_attn", "mlp"]``: every projection is CPU-dequanted F8_E4M3 -> BF16 on
    the loader thread; the decoder layer builds ``Qwen3VLTextAttentionBF16`` +
    ``Qwen3VLTextMLPBF16`` (imported from ``model_bf16``), so the device sees only
    BF16 and runs the bf16 forward graph. Dequant is exact (the scale is a power of
    two), so logits match the BF16/HF reference within tolerance.
  - ``["mlp"]``: the decoder layer builds ``Qwen3VLTextAttentionMX`` (native
    on-device MX) + ``Qwen3VLTextMLPBF16`` (bf16 CPU-dequant).
  - ``[]``: full-transformer MXFP8 — both attention AND the MLP (``Qwen3VLTextMLPMX``,
    prefill CTE + decode TKG) run on-device MX. The K/V cache stays bf16.

MX MLP requires MX attention (the decoder layer only builds the MX MLP when attention
is also MX), so ``["self_attn"]`` alone (bf16 attn + MX mlp) is rejected at load.
``Qwen3VLTextMLPMX`` is validated in isolation by the MLP CTE + TKG three-way module
tests. See the ``Qwen3VLTextMLPMX`` section comment.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF

# nkilib's MLP CTE-vs-TKG kernel split (mlp_parameters.is_mlp_tkg, AUTO mode): B*S
# <= threshold -> TKG kernel, else CTE. Aliased on import (not hard-coded) so the
# plugin's layout/kernel agreement guard tracks nkilib if the threshold ever changes.
from nkilib.core.mlp.mlp_parameters import (
    TKG_BS_SEQLEN_THRESHOLD as _MLP_TKG_BS_SEQLEN_THRESHOLD,
)
from nkilib.core.utils.common_types import (
    MLPGateUpWeightLayout,
    NormType,
    QKVWeightLayout,
    QuantizationType,
)
from vllm_neuron.model.interfaces import (
    SupportsMRoPE,
    SupportsEncoderWarmup,
)
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    set_weight_loader,
)

from .config import Qwen3VLConfig, Qwen3VLTextConfig

# Modules reused from the BF16 model by composition, not inheritance. RMSNorm is a
# quantization-independent building block; the attention/MLP aliases are exactly the
# BF16 fallback that ``modules_to_not_convert`` selects and instantiates. The MX
# classes here instantiate these — they do not subclass them — so reusing them does
# not reintroduce the bf16 inheritance coupling this model removed.
from .model_bf16 import (
    HF_TEXT_PREFIX,
    Qwen3VLTextRMSNorm,
)
from .model_bf16 import Qwen3VLTextAttention as Qwen3VLTextAttentionBF16
from .model_bf16 import Qwen3VLTextMLP as Qwen3VLTextMLPBF16

# Quantization-agnostic orchestration shared with the BF16 model: the top-level
# forward, vision encoding, KV spec/bind, M-RoPE, from_configs, the text backbone
# forward, and the shared __init__ wiring (via _init_*_common helpers this model's
# __init__ calls with its own MX classes). The mixins carry no per-quant compute
# and define no __init__ of their own, so they do not reintroduce inheritance
# coupling.
from .model_common import (
    Qwen3VLGenerationMixin,
    Qwen3VLTextBackboneForwardMixin,
)
from .utils.decode_kv import build_per_head_block_table, scatter_new_kv
from ..kv_cache import paged_slot_indices
from .weight_loaders_mxfp8 import (
    _FP8_DTYPE,
    _Q_WIDTH,
    _SCALES_PER_TILE,
    _TILE_SIZE,
    _n_i512_tiles,
    _n_i512_tiles_tkg,
    MX_GROUP_SIZE,
    build_mx_attention_mappings,
    build_mx_mlp_mappings,
    fused_qkv_scale_loader_3d,
    fused_qkv_weight_loader_3d,
    mlp_down_scale_loader,
    mlp_down_scale_loader_tkg,
    mlp_down_weight_loader,
    mlp_down_weight_loader_tkg,
    mlp_gate_up_scale_loader,
    mlp_gate_up_scale_loader_tkg,
    mlp_gate_up_weight_loader,
    mlp_gate_up_weight_loader_tkg,
    mxfp8_fused_qkv_loader,
    mxfp8_sharded_transposed_loader,
    o_proj_scale_loader,
    o_proj_weight_loader,
    pre_shuffle_h,
)


# ---------------------------------------------------------------------------
# Section 3: Attention
# ---------------------------------------------------------------------------
# Native-MXFP8 attention (prefill + decode) — on-device QuantizationType.MX
# ---------------------------------------------------------------------------
# Standalone on-device FP8 attention. TP / head-sharding / QK-norm / RoPE setup
# is self-contained. QKV is one native 3D-fp8 layout
# shared by prefill and decode; O-Proj needs TWO buffers (see the TODO). S0 (input
# RMSNorm + ROW quant) is not done here — it lives on Qwen3VLTextDecoderLayerMX.
#
# TODO(NMI-191): the two-buffer O-Proj is a TEMPORARY solution — the prefill
# NF.o_proj kernel wants 2D-uint32 ``[N*D//4, H]`` x4-packed while the decode
# mega-kernel wants 3D-unpacked-fp8 ``[N*D//4, H, 4]`` (is_h_transposed_by_4), so
# both are materialized at load time from the same checkpoint key (no runtime
# repack, ~one extra o-proj weight/layer/rank): ``o_proj_weight_decode`` (3D-fp8)
# and ``o_proj_weight_prefill`` (2D-uint32). ``_run_decode_megakernel``
# reads the decode buffer via the ``_decode_o_proj_weight`` hook. Unify once the two
# kernels accept a single canonical layout, then drop the second buffer + the hook.


class Qwen3VLTextAttentionMX(nn.Module):
    """Qwen3-VL text attention with native-MXFP8 QKV + O-Proj (prefill + decode)."""

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()

        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.rms_norm_eps = config.rms_norm_eps
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

        # >>> PARALLELISM: QKV weight shapes for TP <<<
        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim
        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        self.q_layernorm = Qwen3VLTextRMSNorm(
            self.head_dim, config.rms_norm_eps, self.dtype
        )
        self.k_layernorm = Qwen3VLTextRMSNorm(
            self.head_dim, config.rms_norm_eps, self.dtype
        )

        self.k_cache = None
        self.v_cache = None

        fused_i = self.q_size + 2 * self.kv_size  # per-rank fused QKV out-dim
        # O-proj per-rank in-dim = Q out-dim. Assumes O-proj N*D total ==
        # num_attention_heads*head_dim and num_attention_heads % world_size == 0
        # (holds for Qwen3-VL-32B; NOT for v_head_dim != qk_head_dim variants).
        # State the assumption loudly here — otherwise floor-division silently
        # undersizes o_proj_weight and the loader's nd_total==shard_size*num_shards
        # assert fails cryptically at load. Stored on self so __init__ and
        # _setup_weight_loaders share one definition of the sharding formula.
        assert self.num_attention_heads % self.world_size == 0, (
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
            f"world_size ({self.world_size}) for the O-proj nd_per_rank assumption."
        )
        self.nd_per_rank = self.num_attention_heads * self.head_dim // self.world_size
        nd_per_rank = self.nd_per_rank
        H = self.hidden_size

        # Initialize native MX projection buffers (layouts + rationale in the class
        # TODO). Shapes below encode each buffer's layout.
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(H // _Q_WIDTH, fused_i, _Q_WIDTH, dtype=_FP8_DTYPE),
            requires_grad=False,
        )
        self.qkv_weight_scale = nn.Parameter(
            torch.empty(H // MX_GROUP_SIZE, fused_i, dtype=torch.uint8),
            requires_grad=False,
        )
        self.o_proj_weight_decode = nn.Parameter(  # decode (3D-fp8)
            torch.empty(nd_per_rank // _Q_WIDTH, H, _Q_WIDTH, dtype=_FP8_DTYPE),
            requires_grad=False,
        )
        self.o_proj_weight_prefill = nn.Parameter(  # prefill (2D-uint32 x4-packed)
            torch.empty(nd_per_rank // _Q_WIDTH, H, dtype=torch.uint32),
            requires_grad=False,
        )
        self.o_proj_weight_scale = nn.Parameter(  # shared by both kernels
            torch.empty(nd_per_rank // MX_GROUP_SIZE, H, dtype=torch.uint8),
            requires_grad=False,
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach the native MX loaders (3D-fp8 QKV weight, two O-Proj weight
        buffers — 3D-fp8 for decode + 2D-uint32 for prefill — and uint8 scales)."""
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader_3d(self.world_size, self.num_kv_replicas),
        )
        set_weight_loader(
            self.qkv_weight_scale,
            fused_qkv_scale_loader_3d(self.world_size, self.num_kv_replicas),
        )
        nd_per_rank = self.nd_per_rank
        # Decode: 3D-fp8 pack.
        set_weight_loader(
            self.o_proj_weight_decode,
            o_proj_weight_loader(nd_per_rank, self.world_size, pack_3d=True),
        )
        # Prefill: 2D-uint32 x4-pack — same checkpoint key, different layout.
        set_weight_loader(
            self.o_proj_weight_prefill,
            o_proj_weight_loader(nd_per_rank, self.world_size, pack_3d=False),
        )
        set_weight_loader(
            self.o_proj_weight_scale,
            o_proj_scale_loader(nd_per_rank, self.world_size),
        )

    # ── Forward dispatch ─────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ):
        """Dispatch to the native-MX prefill (CTE) or decode (TKG) path.

        For prefill, ``hidden_states`` is the ROW-packed FP8 ``[T, H+4]`` activation
        produced by ``Qwen3VLTextDecoderLayerMX`` (S0 = input RMSNorm + ROW quant
        lives on the layer, like the bf16 ``input_layernorm``). This module takes the
        SAME signature as the bf16 base ``forward`` (no ``ln_w``/``eps``).
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states, positions, position_embeddings, attn_metadata
            )
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(
            hidden_states, positions, position_embeddings, attn_metadata
        )

    # ── Prefill path ─────────────────────────────────────────────────────

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """Native-MXFP8 prefill chain.

        qkv_proj(MX, MX_CONTIGUOUS, with fused QK-norm + M-RoPE) -> flash_attention
        (bf16) -> o_proj(MX). QK-norm + RoPE are fused into ``NF.qkv_proj`` (applied
        in the kernel's post-matmul per-head path after dequant); only the input
        RMSNorm/residual/swizzle are forbidden on the native-row-MX path.

        ``hidden_states`` is the ROW-packed FP8 ``[T, H+4]`` produced upstream by
        ``Qwen3VLTextDecoderLayerMX`` via ``NF.rmsnorm_quant(ROW)`` (S0); the per-row
        scale rides in the +4 tail, so no ``qkv_in_scale`` is passed.
        """
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        # Reject raw bf16 [T, H]: the +4 tail carries the per-row ROW scale.
        assert hidden_states.shape[-1] == self.hidden_size + _Q_WIDTH, (
            f"Qwen3VLTextAttentionMX.forward_prefill expects the ROW-packed FP8 "
            f"input [T, H+{_Q_WIDTH}] from rmsnorm_quant(ROW); got last dim "
            f"{hidden_states.shape[-1]} (H={self.hidden_size}). Drive this module "
            f"via Qwen3VLTextDecoderLayerMX, which packs first."
        )

        tokens = hidden_states.shape[0]

        cos, sin = position_embeddings
        cos_cache = torch.cat((cos, cos), dim=-1).unsqueeze(0)
        sin_cache = torch.cat((sin, sin), dim=-1).unsqueeze(0)

        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            quantization_type=QuantizationType.MX,
            qkv_w_scale=self.qkv_weight_scale,
            qkv_in_scale=None,  # ROW-packed input carries its own per-row scale
            weight_layout=QKVWeightLayout.MX_CONTIGUOUS,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            num_q_heads=self.num_attention_heads_per_rank,
            num_kv_heads=self.num_key_value_heads_per_rank,
            d_head=self.head_dim,
            qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_eps=self.rms_norm_eps,
            qk_norm_pre_rope_q_gamma=self.q_layernorm.weight.unsqueeze(0),
            qk_norm_pre_rope_k_gamma=self.k_layernorm.weight.unsqueeze(0),
        ).squeeze(0)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )

        # KV cache update.
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        block_indices, position_indices = paged_slot_indices(
            slot_mapping, block_size, self.k_cache.shape[0]
        )

        k_flat = k.reshape(-1, self.head_dim)
        v_flat = v.reshape(-1, self.head_dim)

        head_indices_for_put = torch.arange(
            self.num_key_value_heads_per_rank,
            dtype=torch.long,
            device=hidden_states.device,
        ).repeat_interleave(slot_mapping.shape[0])
        block_indices_for_put = block_indices.repeat(self.num_key_value_heads_per_rank)
        position_indices_for_put = position_indices.repeat(
            self.num_key_value_heads_per_rank
        )

        self.k_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            k_flat,
        )
        self.v_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            v_flat,
        )

        # Attention (bf16 q/k/v — MXFP8 applies to the projection weights, not
        # the KV cache). Under segmented / chunked prefill the current segment's
        # queries must attend over all prior cached KV, so route to the
        # segmented kernel which reads K/V from the paged cache (block_table +
        # prior_tokens). Otherwise plain flash attention over the current tokens
        # is sufficient.
        if kv_segment_size:
            attn_output = NF.segmented_attention(
                q,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=block_table,
                prior_tokens=cached_seq_len,
                block_size=block_size,
                kv_segment_size=kv_segment_size,
                scale=self.scaling,
                tp_q=True,
                tp_out=True,
            )
        else:
            k = k.repeat_interleave(self.num_key_value_groups, dim=0)
            v = v.repeat_interleave(self.num_key_value_groups, dim=0)

            q_flash = q.transpose(1, 2)
            k_flash = k.transpose(1, 2)
            v_flash = v

            attn_output = NF.flash_attention(
                q_flash,
                k_flash,
                v_flash,
                scale=self.scaling,
                tp_q=False,
                tp_out=True,
            )

        # Output projection (native MX) — uses the 2D-uint32 prefill buffer.
        attn_output = attn_output.unsqueeze(0)
        attn_output = NF.o_proj(
            attn_output,
            self.o_proj_weight_prefill,
            None,
            quantization_type=QuantizationType.MX,
            input_scales=None,  # hardware MX-quantizes the bf16 activation
            weight_scales=self.o_proj_weight_scale,
        )
        attn_output = attn_output.squeeze(0)

        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    # ── Decode path ──────────────────────────────────────────────────────

    def _decode_cos_sin(
        self,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        B: int,
        S_decode: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reshape HF ``[T, head_dim]`` cos/sin into the kernel's ``[half_d, B, S]``
        layout."""
        cos, sin = position_embeddings
        half_d = self.head_dim // 2
        cos_kernel = (
            cos[:, :half_d]
            .view(B, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )
        sin_kernel = (
            sin[:, :half_d]
            .view(B, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )
        return cos_kernel, sin_kernel

    @property
    def _decode_o_proj_weight(self) -> torch.Tensor:
        """The 3D-fp8 decode-layout O-Proj buffer ``_run_decode_megakernel`` reads
        as ``W_out``."""
        return self.o_proj_weight_decode

    def _run_decode_megakernel(
        self,
        X: torch.Tensor,
        cos_kernel: torch.Tensor,
        sin_kernel: torch.Tensor,
        active_blocks_table: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        pos_ids_kernel: torch.Tensor,
        **quant_kwargs,
    ):
        """Invoke the fused decode mega-kernel (QKV+QK-norm+RoPE+attn+O-proj).

        Returns ``(output, K_new, V_new)``; the caller does the KV-cache scatter.
        """
        W_q_norm = self.q_layernorm.weight.view(self.head_dim, 1)
        W_k_norm = self.k_layernorm.weight.view(self.head_dim, 1)

        return NF.attention_decode(
            X=X,
            W_qkv=self.qkv_proj_weight,
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_W_Q=W_q_norm,
            rmsnorm_QK_pre_rope_W_K=W_k_norm,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=active_blocks_table,
            K_cache=K_cache,
            V_cache=V_cache,
            pos_ids=pos_ids_kernel,
            swa_start_pos_ids=None,
            softmax_scale=self.scaling,
            update_cache=False,
            W_out=self._decode_o_proj_weight,
            transposed_out=False,
            out_in_sb=False,
            **quant_kwargs,
        )

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
    ):
        """Native-MXFP8 decode via the fused mega-kernel (QK-norm, RoPE, GQA).

        The o_proj weight is the 3D-fp8 ``self.o_proj_weight_decode`` (consumed by
        ``_run_decode_megakernel`` as ``W_out`` via the ``_decode_o_proj_weight``
        hook), not the prefill buffer.
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        assert tokens == B * S_decode

        hidden_states = hidden_states.to(self.dtype)
        nkh = self.num_key_value_heads_per_rank

        assert hidden % 512 == 0, (
            f"pre_shuffle_h (is_h_transposed_by_4) needs hidden_size % 512 == 0 "
            f"(128*4 H-tile); got {hidden}."
        )
        X = pre_shuffle_h(hidden_states.view(B, S_decode, hidden))

        # NOTE: S0 (input RMSNorm) is NOT fused into the kernel (rmsnorm_X_enabled=False);
        # it stays on the decoder layer. In-kernel S0 was tried but the three-way CPU
        # reference under-models the kernel's in-kernel RMSNorm+quantize_mx noise, so it
        # is deferred until the reference models that noise (a reference gap, not a bug).

        cos_kernel, sin_kernel = self._decode_cos_sin(position_embeddings, B, S_decode)

        pos_ids_kernel = positions.view(B, S_decode).to(torch.float32)

        # Per-head 3D block table (shared builder — same GQA global-index convention
        # as the CPU reference, so the two cannot drift). The 4D KV cache is kept as-is
        # (the GQA kernel consumes the per-head layout).
        active_blocks_table = build_per_head_block_table(block_table, nkh)

        output, K_new, V_new = self._run_decode_megakernel(
            X,
            cos_kernel,
            sin_kernel,
            active_blocks_table,
            self.k_cache,
            self.v_cache,
            pos_ids_kernel,
            quantization_type_qkv=QuantizationType.MX,
            weight_dequant_scale_qkv=self.qkv_weight_scale,
            input_dequant_scale_qkv=None,
            quantization_type_out=QuantizationType.MX,
            weight_dequant_scale_out=self.o_proj_weight_scale,
            input_dequant_scale_out=None,
            is_h_transposed_by_4=True,
        )

        # Normalize the kernel's new-K/V tokens to 4D per-head layout. The fused
        # kernel drops the kv_heads axis on K when kv_heads == 1 (TP >= num_kv_heads,
        # e.g. TP8): K_new is 3D [d_head, B, S] instead of 4D [d_head, B, kv, S] (V is
        # always 4D [B, kv, S, d_head]). Re-insert the singleton head axis so
        # scatter_new_kv sees the uniform 4D contract for all kv_heads.
        if nkh == 1 and K_new.dim() == 3:
            K_new = K_new.unsqueeze(2)  # [d_head, B, S] -> [d_head, B, 1, S]

        # KV cache update (per-head permute + paged scatter; see scatter_new_kv).
        scatter_new_kv(
            self.k_cache,
            self.v_cache,
            K_new,
            V_new,
            slot_mapping,
            block_size,
            nkh,
            self.head_dim,
        )

        # >>> PARALLELISM: TP all-reduce after megakernel <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        return output


# ---------------------------------------------------------------------------
# Section 5: MLP
# ---------------------------------------------------------------------------
# Native-MXFP8 MLP (prefill CTE + decode TKG) — on-device QuantizationType.MX
# ---------------------------------------------------------------------------
# NMI-189: the MLP counterpart of the attention MX chain above. Two MX kernels:
#   * CTE (prefill, nkilib CR-283286137): activation is ROW-packed FP8 [T, H+4]
#     from rmsnorm_quant(ROW) (per-row scale in the +4 tail); weights are 6D/4D
#     H_X4_INNERMOST scalar-fp8 + 5D/3D u8 scales. Mixed row+mx dequant.
#   * TKG (decode, mlp_tkg_quad_fp8_mx): activation is PLAIN bf16 [T, H] quantized
#     online by the kernel (per-32 MX); weights are 3D uint32 x4-packed ([128,H/512,I]
#     gate/up H-x4, [128,I/512,H] down I-x4) + 3D u8 scales, passed with
#     gate_up_w_layout=CONTIGUOUS (nkilib's "legacy MX" TKG path; the 6D H_X4_* layouts
#     are the CTE/STATIC_MX kernels, which the MX TKG kernel rejects).
# Both use per-32 E8M0 weight scales fed to nc_matmul_mx. The two layouts are
# INCOMPATIBLE, so gate/up/down are split into ``_prefill`` + ``_decode`` buffers
# loaded from the SAME checkpoint keys (like the o_proj two-buffer split above and
# llama3's ``gate_proj_weight`` vs ``..._tkg``). See the class docstring for the
# forward-time buffer selection.


class Qwen3VLTextMLPMX(nn.Module):
    """Qwen3-VL SwiGLU MLP with native-MXFP8 gate/up/down, prefill (CTE) + decode (TKG).

    Two weight buffer sets (like the attention module's o_proj two-buffer split): the
    CTE (prefill) and TKG (decode) MX kernels want INCOMPATIBLE weight layouts, so
    ``forward(hidden, is_prefill)`` picks the matching set. Prefill consumes a
    ROW-packed FP8 ``[T, H+4]`` activation (per-row scale in the +4 tail); decode
    consumes plain bf16 ``[T, H]`` and the kernel quantizes it online (per-32 MX).
    """

    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__()

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        self.hidden_size = config.hidden_size
        # >>> PARALLELISM: Intermediate dim sharded across TP <<<
        self.intermediate_size_per_rank = config.intermediate_size // self.world_size
        self.dtype = config.torch_dtype

        H = self.hidden_size
        i_pr = self.intermediate_size_per_rank
        # Tile constants come from the loader module so these nn.Parameter shapes
        # cannot drift from the swizzle the loaders produce (the MX kernel does not
        # validate scale shapes, so a mismatch would miscompute, not error).
        n_h512 = H // _TILE_SIZE
        n_i512 = _n_i512_tiles(i_pr)  # CTE even 512-tile count — see _n_i512_tiles
        n_i512_tkg = _n_i512_tiles_tkg(i_pr)  # TKG plain-ceil count (shards H, not I)
        n_h_scale = _SCALES_PER_TILE

        # Native MX weight layouts. The CTE (prefill)
        # buffers use the 6D/4D H_X4_INNERMOST swizzle; the TKG (decode) buffers use
        # the 3D uint32 layout the decode kernel reads (H-x4 for gate/up, I-x4 for
        # down). The torch.empty(...) calls below are the source of truth for shapes.
        # --- CTE (prefill): 6D/4D scalar-fp8 + 5D/3D u8 scales (H_X4_INNERMOST) ---
        self.gate_proj_weight_prefill = nn.Parameter(
            torch.empty(128, n_h512, n_i512, _Q_WIDTH, 128, _Q_WIDTH, dtype=_FP8_DTYPE),
            requires_grad=False,
        )
        self.up_proj_weight_prefill = nn.Parameter(
            torch.empty(128, n_h512, n_i512, _Q_WIDTH, 128, _Q_WIDTH, dtype=_FP8_DTYPE),
            requires_grad=False,
        )
        self.down_proj_weight_prefill = nn.Parameter(
            torch.empty(128, n_i512, H, _Q_WIDTH, dtype=_FP8_DTYPE),
            requires_grad=False,
        )
        self.gate_proj_weight_scale_prefill = nn.Parameter(
            torch.empty(n_h_scale, n_h512, n_i512, _Q_WIDTH, 128, dtype=torch.uint8),
            requires_grad=False,
        )
        self.up_proj_weight_scale_prefill = nn.Parameter(
            torch.empty(n_h_scale, n_h512, n_i512, _Q_WIDTH, 128, dtype=torch.uint8),
            requires_grad=False,
        )
        self.down_proj_weight_scale_prefill = nn.Parameter(
            torch.empty(n_h_scale, n_i512, H, dtype=torch.uint8),
            requires_grad=False,
        )
        # --- TKG (decode): 3D uint32 weights + 3D u8 scales (H_X4_MIDDLE) ---
        # gate/up: [128, H/512, I_pad] uint32 (x4 along H); scale [16, H/512, I_pad] u8.
        # down:    [128, ceil(I/512), H] uint32 (x4 along I, STRIDE-128); scale
        #   [16, ceil(I/512), H] u8. The down decode buffers are RE-QUANTIZED from the
        #   checkpoint (dequant + re-quant in the kernel's stride-128 MX blocks) because
        #   the TKG down kernel contracts stride-128 and its per-32 hardware scale block
        #   can't take the checkpoint's per-32-contiguous scale (device-confirmed). See
        #   _down_requant_stride128_tkg in weight_loaders_mxfp8.py.
        # The gate/up FREE-I is padded to a whole 512-tile (i_pad = n_i512_tkg*512):
        # the MX TKG kernel derives intermediate_size from gate_proj.shape[-1] and tiles
        # by ceil(I/512), reading/writing i_pad columns — an unpadded 6400 buffer would
        # be read out-of-bounds (garbage output). The padded columns are inert (fp8 0
        # weight -> 0 gate/up -> SiLU*up = 0 -> 0 down contribution). Matches the down
        # path (already padded) and the nkilib test's ceil-tiled gate/up buffer.
        i_pad = n_i512_tkg * _TILE_SIZE
        self.gate_proj_weight_decode = nn.Parameter(
            torch.empty(128, n_h512, i_pad, dtype=torch.uint32), requires_grad=False
        )
        self.up_proj_weight_decode = nn.Parameter(
            torch.empty(128, n_h512, i_pad, dtype=torch.uint32), requires_grad=False
        )
        self.down_proj_weight_decode = nn.Parameter(
            torch.empty(128, n_i512_tkg, H, dtype=torch.uint32), requires_grad=False
        )
        self.gate_proj_weight_scale_decode = nn.Parameter(
            torch.empty(n_h_scale, n_h512, i_pad, dtype=torch.uint8),
            requires_grad=False,
        )
        self.up_proj_weight_scale_decode = nn.Parameter(
            torch.empty(n_h_scale, n_h512, i_pad, dtype=torch.uint8),
            requires_grad=False,
        )
        self.down_proj_weight_scale_decode = nn.Parameter(
            torch.empty(n_h_scale, n_i512_tkg, H, dtype=torch.uint8),
            requires_grad=False,
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach the native-MX MLP loaders — CTE (6D/4D) + TKG (3D uint32/u8)."""
        H = self.hidden_size
        i_pr = self.intermediate_size_per_rank
        ws = self.world_size
        # CTE (prefill) loaders.
        set_weight_loader(
            self.gate_proj_weight_prefill, mlp_gate_up_weight_loader(i_pr, H, ws)
        )
        set_weight_loader(
            self.up_proj_weight_prefill, mlp_gate_up_weight_loader(i_pr, H, ws)
        )
        set_weight_loader(
            self.down_proj_weight_prefill, mlp_down_weight_loader(i_pr, H, ws)
        )
        set_weight_loader(
            self.gate_proj_weight_scale_prefill, mlp_gate_up_scale_loader(i_pr, H, ws)
        )
        set_weight_loader(
            self.up_proj_weight_scale_prefill, mlp_gate_up_scale_loader(i_pr, H, ws)
        )
        set_weight_loader(
            self.down_proj_weight_scale_prefill, mlp_down_scale_loader(i_pr, H, ws)
        )
        # TKG (decode) loaders — same checkpoint keys, 3D uint32/u8 pack.
        set_weight_loader(
            self.gate_proj_weight_decode, mlp_gate_up_weight_loader_tkg(i_pr, H, ws)
        )
        set_weight_loader(
            self.up_proj_weight_decode, mlp_gate_up_weight_loader_tkg(i_pr, H, ws)
        )
        set_weight_loader(
            self.down_proj_weight_decode, mlp_down_weight_loader_tkg(i_pr, H, ws)
        )
        set_weight_loader(
            self.gate_proj_weight_scale_decode,
            mlp_gate_up_scale_loader_tkg(i_pr, H, ws),
        )
        set_weight_loader(
            self.up_proj_weight_scale_decode, mlp_gate_up_scale_loader_tkg(i_pr, H, ws)
        )
        set_weight_loader(
            self.down_proj_weight_scale_decode, mlp_down_scale_loader_tkg(i_pr, H, ws)
        )

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        """Native-MX SwiGLU, prefill (CTE) or decode (TKG) by ``is_prefill``.

        Prefill: ``hidden_states`` is the ROW-packed FP8 ``[T_local, H+4]`` produced
        upstream by the wrapping layer's ``NF.rmsnorm_quant(ROW)`` (per-row scale in
        the +4 tail). Decode: plain bf16 ``[T_local, H]`` — the TKG kernel quantizes it
        online (per-32 MX), so there is NO +4 tail. Each path drives ``NF.mlp`` with
        its own weight buffer set + gate/up layout; the wrapper/nkilib pick the actual
        CTE vs TKG kernel by token count, so the caller must hand it the matching
        buffers (mirrors the attention o_proj two-buffer split).
        """
        if is_prefill:
            gate_w = self.gate_proj_weight_prefill
            up_w = self.up_proj_weight_prefill
            down_w = self.down_proj_weight_prefill
            gate_w_scale = self.gate_proj_weight_scale_prefill
            up_w_scale = self.up_proj_weight_scale_prefill
            down_w_scale = self.down_proj_weight_scale_prefill
            gate_up_w_layout = MLPGateUpWeightLayout.H_X4_INNERMOST
            # Reject a raw bf16 [T, H]: the +4 tail carries the per-row ROW scale.
            # Assert the contract EXPLICITLY (like the attention forward_prefill) rather
            # than rely on NF.mlp's mod-128 inference, which is ambiguous when H+4
            # happens to align to 128.
            assert hidden_states.shape[-1] == self.hidden_size + _Q_WIDTH, (
                f"Qwen3VLTextMLPMX prefill expects the ROW-packed FP8 input "
                f"[T, H+{_Q_WIDTH}] from rmsnorm_quant(ROW); got last dim "
                f"{hidden_states.shape[-1]} (H={self.hidden_size}). Drive this module "
                f"via a layer that packs first (post_attention_layernorm ROW)."
            )
        else:
            gate_w = self.gate_proj_weight_decode
            up_w = self.up_proj_weight_decode
            down_w = self.down_proj_weight_decode
            gate_w_scale = self.gate_proj_weight_scale_decode
            up_w_scale = self.up_proj_weight_scale_decode
            down_w_scale = self.down_proj_weight_scale_decode
            # CONTIGUOUS: for QuantizationType.MX this selects nkilib's "legacy MX"
            # TKG path — 3D x4-packed weights [128, n_H512, I] (mlp_parameters.py:606).
            # The 6D H_X4_* layouts are the CTE / STATIC_MX kernels, NOT the MX TKG one
            # (which rejects rank-6 weights). Verified on-device.
            gate_up_w_layout = MLPGateUpWeightLayout.CONTIGUOUS
            # Decode feeds bf16 [T, H] (no +4 tail) — the kernel MX-quantizes it online.
            assert hidden_states.shape[-1] == self.hidden_size, (
                f"Qwen3VLTextMLPMX decode expects bf16 [T, H] (H="
                f"{self.hidden_size}); got last dim {hidden_states.shape[-1]}."
            )
            # PRE-SHUFFLE H so the kernel's internal _layout_adapter_hbm presents
            # CONTIGUOUS-4 logical-H (512*h512 + 4*p + q) at each contraction slot,
            # matching the contiguous-4 gate/up weight pack. See the full derivation in
            # mlp_gate_up_weight_loader_tkg's docstring (weight_loaders_mxfp8.py).
            hidden_states = pre_shuffle_h(hidden_states)

        # >>> PARALLELISM (mirror the bf16 Qwen3VLTextMLP exactly): prefill is
        # sequence-parallel — all-gather the SP-sharded activation before the MLP;
        # decode is NOT SP (one token/rank replicated) — feed the activation as-is.
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        # ROUTING-SEAM GUARD (backstop for the config-time check in load_weights): the
        # weight LAYOUT was picked by ``is_prefill`` but nkilib picks the KERNEL by the
        # token count here (is_mlp_tkg: B*S <= threshold -> TKG). Assert they agree so a
        # mismatched (layout, kernel) pair can't silently reach the kernel. ``shape[0]``
        # is concrete per compiled bucket, so this fires at warmup-compile.
        n_tokens = hidden_states.shape[0]
        if is_prefill:
            assert n_tokens > _MLP_TKG_BS_SEQLEN_THRESHOLD, (
                f"Qwen3VLTextMLPMX prefill selected the CTE weight layout but the "
                f"all-gathered token count ({n_tokens}) is <= the nkilib TKG "
                f"threshold ({_MLP_TKG_BS_SEQLEN_THRESHOLD}), so NF.mlp would route to "
                f"the TKG kernel and reject the 6D CTE weights. Prefill buckets must "
                f"exceed {_MLP_TKG_BS_SEQLEN_THRESHOLD} tokens."
            )
        else:
            assert n_tokens <= _MLP_TKG_BS_SEQLEN_THRESHOLD, (
                f"Qwen3VLTextMLPMX decode selected the TKG weight layout but the token "
                f"count ({n_tokens}) exceeds the nkilib TKG threshold "
                f"({_MLP_TKG_BS_SEQLEN_THRESHOLD}), so NF.mlp would route to the CTE "
                f"kernel and reject the 3D TKG weights. Keep decode B*S <= "
                f"{_MLP_TKG_BS_SEQLEN_THRESHOLD} (max_num_seqs x spec width), or add a "
                f"decode path with CTE-layout buffers."
            )

        output = NF.mlp(
            hidden_states,
            gate_w,
            up_w,
            down_w,
            quantization_type=QuantizationType.MX,
            gate_w_scale=gate_w_scale,
            up_w_scale=up_w_scale,
            down_w_scale=down_w_scale,
            gate_up_in_scale=None,  # prefill: ROW +4 tail; decode: kernel online-quant
            down_in_scale=None,  # kernel MX-quantizes the intermediate online
            gate_up_w_layout=gate_up_w_layout,
            # Pass the dtype as a STRING (matches llama3's MX call). Without an explicit
            # output_dtype the kernel defaults the output to the (fp8) hidden dtype, and
            # the bf16 residual add then fails ("Promotion for Float8 not supported").
            output_dtype="bfloat16",
        )

        # >>> PARALLELISM: down is contraction(I)-sharded, so its output is a partial
        # sum. Prefill reduce-scatters back to SP; decode all-reduces (replicated).
        # Matches the bf16 Qwen3VLTextMLP collective pattern exactly.
        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                self.tp_group.all_reduce(output)

        return output.contiguous()


# ---------------------------------------------------------------------------
# Section 6: Decoder Layer
# ---------------------------------------------------------------------------
# Single decoder layer that routes between BF16 and MX modules based on
# ``modules_to_not_convert``. No separate BF16/MX decoder layer variants.

# Quantized projection param suffixes per text decoder layer. Each pairs with
# its HF ``.weight`` + ``.weight_scale`` companion in the mappings built below.
_QUANT_SUFFIXES = (
    "self_attn.qkv_proj_weight",
    "self_attn.o_proj_weight",
    "mlp.gate_proj_weight",
    "mlp.up_proj_weight",
    "mlp.down_proj_weight",
)


def _keep_bf16(name: str, skip: list[str] | None) -> bool:
    """True if projection ``name`` is CPU-dequanted to BF16 (kept out of the
    on-device FP8 path). A projection whose FQN contains a skip-list token is
    dequanted; otherwise it takes the native MX path."""
    return bool(skip) and any(token in name for token in skip)


def _attn_on_device_mx(skip: list[str] | None) -> bool:
    """True if attention runs the on-device native-MXFP8 path.

    Attention is on-device MX iff ``"self_attn"`` is NOT in ``modules_to_not_convert``:
    ``["self_attn", "mlp"]`` -> both bf16; ``["mlp"]`` -> MX attention + bf16 MLP;
    ``[]`` -> both on-device MX (full-transformer MXFP8).
    """
    return not _keep_bf16("self_attn", skip)


def _mlp_on_device_mx(skip: list[str] | None) -> bool:
    """True if the MLP runs the on-device native-MXFP8 path.

    MLP is on-device MX iff ``"mlp"`` is NOT in ``modules_to_not_convert``. Only
    reachable when attention is ALSO MX (the served MX decoder layer is built only
    then), so ``[]`` -> full-transformer MX; ``["mlp"]`` -> MX attention + bf16 MLP.
    """
    return not _keep_bf16("mlp", skip)


def _validate_mlp_mx_bucket_bounds(neuron_config: object) -> None:
    """Reject a config whose MX-MLP buckets would straddle nkilib's CTE/TKG split.

    ``Qwen3VLTextMLPMX`` carries incompatible per-path weight layouts and picks one by
    ``is_prefill``, but ``NF.mlp`` (mode=AUTO) lets nkilib pick the KERNEL by token
    count (``is_mlp_tkg``: ``B*S <= TKG_BS_SEQLEN_THRESHOLD`` -> TKG). So the served MX
    MLP only works when every DECODE bucket stays ``<= threshold`` (TKG layout -> TKG
    kernel) and every PREFILL bucket stays ``> threshold`` (CTE layout -> CTE kernel).
    A decode ``num_seqs`` bucket > threshold (e.g. ``max_num_seqs=128``, or spec-decode
    that multiplies B*S) or a prefill ``num_batched_tokens`` bucket <= threshold would
    hand the wrong-rank weights to the kernel — caught here at config-load with an
    actionable message instead of an opaque shape error at warmup-compile. Buckets that
    are ``None`` here are defaulted downstream (by the scheduler / model runner) and are
    re-checked by the per-forward routing-seam guard in ``Qwen3VLTextMLPMX.forward``,
    which is the ultimate backstop.

    Note: this cannot see the spec-decode width (unknown to the model at load), so the
    runtime guard remains authoritative; this is the early, explicit config check.
    """
    threshold = _MLP_TKG_BS_SEQLEN_THRESHOLD
    decode_buckets = getattr(neuron_config, "num_seqs_buckets", None)
    if decode_buckets:
        too_big = [b for b in decode_buckets if b > threshold]
        assert not too_big, (
            f"Full-transformer MX MLP (modules_to_not_convert excludes 'mlp') needs "
            f"every decode num_seqs bucket <= the nkilib TKG threshold ({threshold}); "
            f"got buckets {decode_buckets} with {too_big} over it. A decode step with "
            f"B*S > {threshold} routes to the CTE MLP kernel, which rejects the 3D TKG "
            f"weight layout. Keep max_num_seqs (and any spec-decode width) so decode "
            f"B*S <= {threshold}, or keep 'mlp' in modules_to_not_convert (bf16 MLP)."
        )
    prefill_buckets = getattr(neuron_config, "num_batched_tokens_buckets", None)
    if prefill_buckets:
        too_small = [b for b in prefill_buckets if b <= threshold]
        assert not too_small, (
            f"Full-transformer MX MLP needs every prefill num_batched_tokens bucket > "
            f"the nkilib TKG threshold ({threshold}); got buckets {prefill_buckets} "
            f"with {too_small} at/under it. A prefill of <= {threshold} tokens routes "
            f"to the TKG MLP kernel, which rejects the 6D CTE weight layout. Raise the "
            f"small prefill bucket(s) above {threshold}, or keep 'mlp' in "
            f"modules_to_not_convert."
        )


def _rmsnorm_row_pack(
    hidden_states: torch.Tensor, ln_w: torch.Tensor, eps: float
) -> torch.Tensor:
    """S0: fused input RMSNorm + ROW FP8 quant -> packed ``[T, H+4]`` activation.

    The single definition of the native-MX prefill S0, shared by the served MX
    decoder layer (``Qwen3VLTextDecoderLayerMX``) and the CTE three-way test's
    attention-only wrapper so the two cannot drift. ``NF.qkv_proj(MX)`` reads the
    per-row scale from the +4 tail (no ``qkv_in_scale``).
    """
    return NF.rmsnorm_quant(
        hidden_states,
        ln_w,
        None,
        eps=eps,
        quantization_type=QuantizationType.ROW,
    )


# ---------------------------------------------------------------------------
# Native-MXFP8 decoder layer
# ---------------------------------------------------------------------------
# The production decoder layer. A standalone nn.Module that selects BF16 or MX
# attention/MLP per projection from modules_to_not_convert (see the class
# docstring for the three modes), owning its own MLP, post_attention_layernorm
# and prefill/decode dispatch. The CTE attention chain (S0->S7) is validated in
# isolation by an attention-only wrapper in the CTE three-way module test.


class Qwen3VLTextDecoderLayer(nn.Module):
    """Unified decoder layer that selects BF16 or MX attention/MLP based on
    ``modules_to_not_convert``.

    Modes:
      - ["self_attn", "mlp"]: Qwen3VLTextAttentionBF16 + Qwen3VLTextMLPBF16
      - ["mlp"]: Qwen3VLTextAttentionMX + Qwen3VLTextMLPBF16
      - []: Qwen3VLTextAttentionMX + Qwen3VLTextMLPMX
    """

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        nc = config.neuron_config
        skip = nc.modules_to_not_convert if nc else None

        self.attn_on_device_mx = _attn_on_device_mx(skip)
        if self.attn_on_device_mx:
            self.self_attn = Qwen3VLTextAttentionMX(config, layer_idx=layer_idx)
        else:
            self.self_attn = Qwen3VLTextAttentionBF16(config, layer_idx=layer_idx)

        self.mlp_on_device_mx = _mlp_on_device_mx(skip)
        if self.mlp_on_device_mx:
            self.mlp = Qwen3VLTextMLPMX(config)
        else:
            self.mlp = Qwen3VLTextMLPBF16(config)

        self.input_layernorm = Qwen3VLTextRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rms_norm_eps = config.rms_norm_eps
        self.dtype = config.torch_dtype

    def _is_decode(self, attn_metadata: object) -> bool:
        layer_name = f"layers.{self.self_attn.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
        return max_query_len <= decode_token_threshold

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        is_decode = self._is_decode(attn_metadata)

        # Attention: MX prefill uses rmsnorm_row_pack, everything else uses plain norm
        residual = hidden_states
        if self.attn_on_device_mx and not is_decode:
            attn_in = _rmsnorm_row_pack(
                hidden_states.to(self.dtype),
                self.input_layernorm.weight,
                self.rms_norm_eps,
            )
        else:
            attn_in = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(
            attn_in, positions, position_embeddings, attn_metadata
        )

        # MLP: MX MLP prefill uses rmsnorm_row_pack, everything else uses plain norm
        residual = hidden_states
        if self.mlp_on_device_mx and not is_decode:
            mlp_in = _rmsnorm_row_pack(
                hidden_states.to(self.dtype),
                self.post_attention_layernorm.weight,
                self.rms_norm_eps,
            )
        else:
            mlp_in = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(mlp_in, is_prefill=not is_decode)

        return hidden_states


# ---------------------------------------------------------------------------
# Section 7: Model Backbone
# ---------------------------------------------------------------------------
# Vocab-sharded embedding → decoder layers → final RMSNorm.
# SP: embedding reduce-scatters during prefill, all-reduces during decode.
# After all layers, all-gather reconstructs the full sequence for prefill.


class Qwen3VLTextModel(Qwen3VLTextBackboneForwardMixin, nn.Module):
    """Qwen3-VL text transformer backbone.

    Supports optional vision inputs for multimodal inference:
    - vision_embeddings: scattered into token embeddings at vision_mask positions
    - deepstack_vision_embeds: added to hidden states after specific decoder layers
      (DeepStack: visual features from intermediate vision encoder layers are
      injected into early text decoder layers)

    """

    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__()
        # Shared backbone wiring lives in the mixin; pass the MX decoder layer.
        self._init_backbone_common(config, Qwen3VLTextDecoderLayer)

    # forward() is provided by Qwen3VLTextBackboneForwardMixin (model_common).


# ---------------------------------------------------------------------------
# Section 8: LM Head + Weight Loading
# ---------------------------------------------------------------------------
# Top-level multimodal model. Contains the text backbone (language_model)
# and the vision encoder (visual).


class Qwen3VLForConditionalGeneration(
    Qwen3VLGenerationMixin, nn.Module, SupportsEncoderWarmup, SupportsMRoPE
):
    """Qwen3-VL multimodal model with language modeling head.

    Supports both text-only and vision+text flows. The forward signature
    accepts vision_embeddings, vision_mask, and deepstack_vision_embeds
    which are empty for text-only inputs.
    """

    def __init__(
        self,
        config: Qwen3VLConfig,
    ):
        super().__init__()
        # Shared top-level wiring lives in the mixin; pass the MX text backbone.
        self._init_generation_common(config, Qwen3VLTextModel)

    # ── Weight Loading ───────────────────────────────────────────────────

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load the MXFP8 checkpoint.

        Attention and MLP are each either CPU-dequanted to BF16 (in the skip list)
        or kept as native on-device MX weights (excluded). For the native-MX paths
        the modules' own loaders (attached in ``_setup_weight_loaders``) do the
        pack; this method only supplies the param->checkpoint-key mappings.
        Configs: ``["self_attn","mlp"]`` full bf16; ``["mlp"]`` MX attention + bf16
        MLP; ``[]`` full-transformer MX (MX attention + MX MLP).
        """
        # Vision-only EPD (VE) pool: no text model was built (``language_model is
        # None``), so skip the whole text load and only load the vision tower —
        # mirrors the bf16 parent's ``if not self.mm_encoder_only`` gate. Without this
        # the ``len(self.language_model.layers)`` loop below derefs None. factory.py
        # selects this MX model on ``quantization == "mxfp8"`` alone, so a VE pool with
        # an mxfp8 checkpoint reaches here.
        if self.mm_encoder_only:
            if self.visual is not None and not getattr(
                self, "mm_language_model_only", False
            ):
                self.visual.load_weights(checkpoint_path, device="cpu", cpu_mode=True)
            return

        tp_rank = self.rank
        tp_size = self.world_size

        nc = self.text_config.neuron_config
        skip = nc.modules_to_not_convert if nc else None
        attn_mx = _attn_on_device_mx(skip)
        mlp_mx = _mlp_on_device_mx(skip)

        # MLP MX requires MX attention (the served MX decoder layer — which owns the
        # MX MLP — is only built when attention is MX). Reject a bf16-attn + MX-mlp
        # combo loudly rather than silently loading the wrong weights.
        assert not (mlp_mx and not attn_mx), (
            "modules_to_not_convert excludes 'mlp' but includes 'self_attn': MX MLP "
            "without MX attention is unsupported (the MX decoder layer requires MX "
            'attention). Use [] for full MX, ["mlp"] for MX attn + bf16 MLP.'
        )

        # The MX MLP picks its weight layout per path but nkilib picks the kernel by
        # token count; reject buckets that would straddle the CTE/TKG split at config
        # load (clear message) instead of failing opaquely at warmup-compile.
        if mlp_mx:
            _validate_mlp_mx_bucket_bounds(nc)

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        available = checkpoint.get_tensor_names()

        # Guard: a non-skip-listed projection is OK only when its on-device MX path is
        # wired — self_attn (attn_mx) or mlp (mlp_mx). Anything else is unimplemented.
        for suffix in _QUANT_SUFFIXES:
            if _keep_bf16(suffix, skip):
                continue  # bf16 CPU-dequant — always supported
            if "self_attn" in suffix and attn_mx:
                continue  # on-device MX attention — supported
            if "mlp" in suffix and mlp_mx:
                continue  # on-device MX MLP — supported
            raise NotImplementedError(
                f"{suffix!r} is not in modules_to_not_convert but its on-device FP8 "
                "path is unimplemented. Use [] for full MX, ['mlp'] for MX attention "
                '+ bf16 MLP, or ["self_attn", "mlp"] for the full bf16 baseline.'
            )

        mappings: dict = {}
        for layer_id in range(len(self.language_model.layers)):
            hf = f"{HF_TEXT_PREFIX}.layers.{layer_id}"
            ours = f"language_model.layers.{layer_id}"
            layer = self.language_model.layers[layer_id]
            attn, mlp = layer.self_attn, layer.mlp

            if attn_mx:
                # Native on-device MX attention: the module attached its 3D-fp8
                # loaders in __init__; supply only the QKV + O-Proj weight/scale
                # mappings via the shared builder. The q/k/input_layernorm norms are
                # set once in the shared norm block below (both branches).
                mappings.update(
                    build_mx_attention_mappings(f"{hf}.self_attn", layer_prefix=ours)
                )
            else:
                # BF16 baseline: CPU-dequant qkv + o_proj F8 -> BF16.
                # qkv (fused): interleaved [qW, qS, kW, kS, vW, vS] -> dequant loader.
                mappings[f"{ours}.self_attn.qkv_proj_weight"] = [
                    f"{hf}.self_attn.q_proj.weight",
                    f"{hf}.self_attn.q_proj.weight_scale",
                    f"{hf}.self_attn.k_proj.weight",
                    f"{hf}.self_attn.k_proj.weight_scale",
                    f"{hf}.self_attn.v_proj.weight",
                    f"{hf}.self_attn.v_proj.weight_scale",
                ]
                set_weight_loader(
                    attn.qkv_proj_weight,
                    mxfp8_fused_qkv_loader(
                        q_size=attn.q_size,
                        kv_size=attn.kv_size,
                        num_shards=attn.world_size,
                        num_kv_replicas=attn.num_kv_replicas,
                    ),
                )

                # o_proj: input-sharded (block axis), [weight, weight_scale].
                mappings[f"{ours}.self_attn.o_proj_weight"] = [
                    f"{hf}.self_attn.o_proj.weight",
                    f"{hf}.self_attn.o_proj.weight_scale",
                ]
                set_weight_loader(
                    attn.o_proj_weight,
                    mxfp8_sharded_transposed_loader(
                        shard_dim=0,
                        shard_size=(attn.num_attention_heads * attn.head_dim)
                        // attn.world_size,
                        num_shards=attn.world_size,
                    ),
                )

            if mlp_mx:
                # Native on-device MX MLP: the module attached its CTE+TKG loaders in
                # __init__; supply the gate/up/down weight+scale mappings (both the
                # _prefill and _decode buffer sets, same checkpoint keys) via the
                # shared builder. post_attention_layernorm is set in the norm block.
                mappings.update(
                    build_mx_mlp_mappings(f"{hf}.mlp", layer_prefix=ours, decode=True)
                )
            else:
                # BF16 baseline: gate/up output-sharded, down input-sharded, CPU-dequant.
                for proj, shard_dim in (
                    ("gate_proj", 1),
                    ("up_proj", 1),
                    ("down_proj", 0),
                ):
                    mappings[f"{ours}.mlp.{proj}_weight"] = [
                        f"{hf}.mlp.{proj}.weight",
                        f"{hf}.mlp.{proj}.weight_scale",
                    ]
                    set_weight_loader(
                        getattr(mlp, f"{proj}_weight"),
                        mxfp8_sharded_transposed_loader(
                            shard_dim=shard_dim,
                            shard_size=mlp.intermediate_size_per_rank,
                            num_shards=mlp.world_size,
                        ),
                    )

            # Non-quantized text params (norms): bf16 single-key mappings.
            mappings[f"{ours}.self_attn.q_layernorm.weight"] = (
                f"{hf}.self_attn.q_norm.weight"
            )
            mappings[f"{ours}.self_attn.k_layernorm.weight"] = (
                f"{hf}.self_attn.k_norm.weight"
            )
            mappings[f"{ours}.input_layernorm.weight"] = f"{hf}.input_layernorm.weight"
            mappings[f"{ours}.post_attention_layernorm.weight"] = (
                f"{hf}.post_attention_layernorm.weight"
            )

        # Embedding, final norm, LM head: bf16, no scale companion.
        mappings["language_model.embed_tokens.weight"] = (
            f"{HF_TEXT_PREFIX}.embed_tokens.weight"
        )
        mappings["language_model.norm.weight"] = f"{HF_TEXT_PREFIX}.norm.weight"
        if self.text_config.tie_word_embeddings:
            mappings["lm_head.weight"] = f"{HF_TEXT_PREFIX}.embed_tokens.weight"
        else:
            mappings["lm_head.weight"] = "lm_head.weight"

        # Per-layer quant params that MUST populate post-load. The MX attention adds
        # the qkv/o-proj scales + the second o-proj buffer to the base .weight set.
        quant_suffixes = list(_QUANT_SUFFIXES)
        if attn_mx:
            # MX attention has no plain ``o_proj_weight``; it has two layout-specific
            # buffers (decode 3D-fp8 + prefill 2D-uint32) plus the qkv/o-proj scales.
            quant_suffixes.remove("self_attn.o_proj_weight")
            quant_suffixes += [
                "self_attn.qkv_weight_scale",
                "self_attn.o_proj_weight_decode",
                "self_attn.o_proj_weight_prefill",
                "self_attn.o_proj_weight_scale",
            ]
        if mlp_mx:
            # MX MLP has no plain ``mlp.*_proj_weight``; each projection has two
            # layout-specific buffers (prefill 6D/4D + decode 3D) + their scales.
            for proj in ("gate_proj", "up_proj", "down_proj"):
                quant_suffixes.remove(f"mlp.{proj}_weight")
                quant_suffixes += [
                    f"mlp.{proj}_weight_prefill",
                    f"mlp.{proj}_weight_scale_prefill",
                    f"mlp.{proj}_weight_decode",
                    f"mlp.{proj}_weight_scale_decode",
                ]
        quant_fqns = [
            f"language_model.layers.{i}.{s}"
            for i in range(len(self.language_model.layers))
            for s in quant_suffixes
        ]

        # Fail loud if any mapped checkpoint key is absent. load is strict=False,
        # so a typo'd / renamed key would otherwise silently leave the param's
        # empty buffer in place (a non-crashing garbage model). Covers the
        # quantized weight+scale keys AND the bf16 text params (norms, q/k norms,
        # embed_tokens, norm, lm_head); vision params are not in `mappings` (they
        # load via self.visual.load_weights) so they are correctly not checked.
        for fqn, keys in mappings.items():
            for key in [keys] if isinstance(keys, str) else keys:
                assert key in available, f"missing MXFP8 key {key!r} for {fqn}"

        result = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, mappings, device, strict=False
        )

        # Normalize dequanted params to bf16, but NEVER cast the native-MX attention
        # params (fp8 weights, uint8 scales, uint32 x4-packed o-proj) — casting those
        # to bf16 would destroy the on-device layout.
        target_dtype = self.text_config.torch_dtype
        _mx_native_dtypes = (torch.uint8, torch.uint32, _FP8_DTYPE)
        for name, tensor in result.state_dict.items():
            if tensor.dtype in _mx_native_dtypes:
                continue
            if tensor.dtype != target_dtype:
                result.state_dict[name] = tensor.to(target_dtype)

        # Post-load: every quantized projection must be populated (strict=False
        # does not raise on a missing key, only records it).
        for fqn in quant_fqns:
            assert fqn in result.state_dict, (
                f"MXFP8 weight {fqn!r} not loaded (check checkpoint keys)"
            )

        self.load_state_dict(result.state_dict, strict=False, assign=True)

        # Vision encoder weights load on the vision TP group (bf16, no dequant).
        # Skipped on a language-only PD pool (no vision tower built), mirroring the
        # bf16 parent's guard — otherwise self.visual is None -> AttributeError.
        # ``getattr`` default keeps this correct on parents that predate the EPD
        # ``mm_language_model_only`` flag (always build/load the vision tower there).
        if self.visual is not None and not getattr(
            self, "mm_language_model_only", False
        ):
            self.visual.load_weights(checkpoint_path, device="cpu", cpu_mode=True)
