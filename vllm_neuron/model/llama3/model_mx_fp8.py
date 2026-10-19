# SPDX-License-Identifier: Apache-2.0
"""
Llama Static MX FP8 Implementation
===========================

Static MX FP8 (STATIC_MX kernel) variant of
:mod:`vllm_neuron.model.llama3.model` for Trn3. Every projection
(QKV / O proj / MLP) uses the STATIC_MX kernel path on both prefill
(CTE) and decode (TKG).

Weights are stored in ``float8_e4m3fn`` with a single per-tensor dequant
scale. Activations are static-FP8 quantized for QKV/O proj and MLP. Both
weight and input scales are pre-broadcast to ``(128, 1)`` fp32 at load
time (or ``(128, 3)`` for fused QKV) to avoid runtime broadcast for
better performance.

Only :class:`LlamaAttention` and :class:`LlamaMLP` are provided here;
the decoder layer, model backbone, and causal-LM head still come from
:mod:`.model`.

This implementation has been tested against checkpoints quantized with
``quant_method='model_opt'`` and ``quant_algo='FP8'`` e.g.
nvidia/Llama-3.1-8B-Instruct-FP8.

Module responsibilities (differ from bf16 sibling)
--------------------------------------------------
:class:`LlamaAttention` is **bf16-in / bf16-out** on both prefill and
decode. The STATIC ``NF.qkv_proj`` and ``NF.o_proj`` kernels take a
bf16 activation and handle fp8 quant/dequant internally using
``qkv_in_scale`` / ``o_in_scale``. The decoder layer runs its plain
``input_layernorm`` in bf16 and passes the result straight to
``self_attn`` — there is no pre-attention ``rmsnorm_quant``.

:class:`LlamaMLP` owns **both the post-attention RMSNorm and the MLP**
on the prefill path: its ``forward`` internally calls
:func:`NF.rmsnorm_quant` (fused RMSNorm + static fp8 quant) to produce
fp8 activations for :func:`NF.mlp`, using a ``ln_w`` supplied by the
decoder layer. The decoder layer must therefore **skip** its plain
``post_attention_layernorm`` on prefill when dispatching to this MLP.
On decode this module is MLP-only — the attention megakernel already
emits bf16 and the MLP kernel handles the static re-quant internally.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Parallelism code; matches the bf16 sibling.
  # <-- STATIC-FP8: ...        Places that differ from bf16 (dtype, scales,
                                kernel kwargs).
"""

import logging

import torch
from torch import nn
from vllm.distributed.parallel_state import get_dcp_group, get_tp_group
from nkilib.core.utils.common_types import (
    NormType,
    QKVWeightLayout,
    QuantizationType,
)

try:
    from nkilib.core.utils.common_types import MLPGateUpWeightLayout
except ImportError:
    from enum import IntEnum

    class MLPGateUpWeightLayout(IntEnum):
        CONTIGUOUS = 0
        H_X4_INNERMOST = 1
        H_X4_MIDDLE = 2


import vllm_neuron.functional as NF
from vllm_neuron.functional.attention.attention_decode import (
    _swizzle_packed_k,
    _unswizzle_packed_k,
    scatter_packed_k,
)
from vllm_neuron.model.gpt_oss.model_bf16 import _packed_fp8_viable_for_bucket
from vllm_neuron.utils.dtype_utils import (
    FP8_CLAMP_MAX,
    validate_fp8_segmented_supported,
)

from .config import LlamaConfig
from .model import (
    _DECODE_MASK_CACHE_KEY,
    _segmented_attention_cp_prefill,
    _validate_dcp_decode_sequence_length,
    _validate_dcp_prefill_kv_cache,
)
from . import weight_loaders_mx_fp8 as _mx_fp8_loaders

logger = logging.getLogger(__name__)


# =============================================================================
# Static-FP8 constants
# =============================================================================

# Static-FP8 weight dtype.
_FP8_DTYPE = torch.float8_e4m3fn

# Partition dimension size the kernels expect for pre-broadcast scales.
_PMAX = 128

# Number of per-projection scales fused into the QKV weight_scale tensor
# (one each for Q, K, V). Matches the NF.qkv_proj STATIC contract.
_QKV_FUSED = 3

# x4 packing factor for MX FP8: 4 ``float8_e4m3fn`` bytes per ``nl.float8_e4m3fn_x4``.
_Q_WIDTH = 4


# =============================================================================
# Loader selection: static FP8 vs MX FP8
# =============================================================================
#
# Both schemes consume the same ModelOpt static-FP8 checkpoint (plain
# fp8 weights, scalar fp32 weight + activation scales). They differ in
# the *layout* of the weights as they sit in HBM:
#
#   * static FP8: plain ``[H, I]`` / ``[H, fused]`` fp8.
#   * MX FP8 (Trn3 only): MLP gate/up/down weights are pre-swizzled
#     into the layout the trn3 nkilib MLP CTE STATIC_MX kernel expects
#     (other projections stay plain).
#
# The choice is made at module-construction time. Forward picks the
# matching ``QuantizationType`` (STATIC vs STATIC_MX) when calling NF
# kernel wrappers; the wrappers fall back from STATIC_MX to STATIC at
# runtime when per-call constraints fail (e.g. BxS%4 on the QKV TKG MX
# path).


def _pick_loader_module():
    """Return the loader module (``weight_loaders_*``) to use."""
    return _mx_fp8_loaders


# Names that the dual-buffer path duplicates. Both prefill and decode
# are MX, but the prefill (CTE) and decode (TKG) kernels need different
# packed weight layouts, so the canonical names carry the prefill (CTE)
# layout and ``<name>_tkg`` carry the decode (TKG) layout. Used by
# ``_TkgAlias`` to redirect the TKG loader-attach functions to write to
# the ``_tkg`` parameters without modifying the existing loader code.
#
# Scales are aliased too because the prefill (CTE) and decode (TKG) MX
# scale loaders can emit different shapes; keeping separate buffers lets
# each attach write to its own buffer without clobbering the other.
_TKG_ALIASED_ATTRS = (
    "qkv_proj_weight",
    "qkv_weight_scale",
    "qkv_input_scale",
    "o_proj_weight",
    "o_weight_scale",
    "o_input_scale",
    "gate_proj_weight",
    "up_proj_weight",
    "down_proj_weight",
    "gate_weight_scale",
    "up_weight_scale",
    "down_weight_scale",
    "gate_up_input_scale",
    "down_input_scale",
)


class _TkgAlias:
    """Module proxy that aliases ``<name> -> module.<name>_tkg`` for the
    parameters the dual-buffer decode path duplicates. Other attribute
    reads pass through to the underlying module.

    The decode (TKG) loader-attach functions in ``weight_loaders_mx_fp8``
    write to ``module.qkv_proj_weight`` etc. via ``set_weight_loader``.
    Wrapping the LlamaAttention / LlamaMLP module in this proxy and
    handing it to those attach functions causes the loaders to land on
    the ``_tkg`` parameters instead — no edits to the loader files.
    """

    __slots__ = ("_module",)

    def __init__(self, module):
        object.__setattr__(self, "_module", module)

    def __getattr__(self, name: str):
        # __getattr__ only fires on misses; __slots__ + object.__setattr__
        # keep this loop-free.
        if name in _TKG_ALIASED_ATTRS:
            tkg_name = f"{name}_tkg"
            if hasattr(self._module, tkg_name):
                return getattr(self._module, tkg_name)
        return getattr(self._module, name)


# Per-call helpers picking the kernel ``QuantizationType`` / weight
# layout. STATIC_MX is always on: every projection uses the STATIC_MX
# kernel path on both prefill and decode.


def _quant_type_mlp_cte() -> QuantizationType:
    """MLP CTE (prefill): STATIC_MX."""
    return QuantizationType.STATIC_MX


def _quant_type_mlp_tkg() -> QuantizationType:
    """MLP TKG (decode): STATIC_MX."""
    return QuantizationType.STATIC_MX


def _quant_type_o_proj_cte() -> QuantizationType:
    """O proj CTE (prefill): STATIC_MX."""
    return QuantizationType.STATIC_MX


def _quant_type_o_proj_tkg() -> QuantizationType:
    """O proj TKG (decode): STATIC_MX."""
    return QuantizationType.STATIC_MX


def _quant_type_qkv_cte() -> QuantizationType:
    """QKV CTE (prefill): STATIC_MX."""
    return QuantizationType.STATIC_MX


def _quant_type_qkv_tkg() -> QuantizationType:
    """QKV TKG (decode): STATIC_MX."""
    return QuantizationType.STATIC_MX


def _quant_type_rmsnorm() -> QuantizationType:
    """``rmsnorm_quant`` only understands STATIC; the OCP clamp range is
    chosen via the kernel's ``auto_resolve_fp8_dtype`` flag, not the
    ``QuantizationType`` enum."""
    return QuantizationType.STATIC


def _qkv_weight_layout() -> QKVWeightLayout:
    """``MX_INTERLEAVED`` for the STATIC_MX path (3D [H//4, I, 4] fp8 with
    H-reorder)."""
    return QKVWeightLayout.MX_INTERLEAVED


def _mlp_gate_up_w_layout():
    """``H_X4_INNERMOST`` for STATIC_MX MLP CTE.

    INNERMOST is the pre-quantized-input layout: since prefill feeds the MLP
    fp8 hidden (from ``rmsnorm_quant``), this lets the kernel DMA-transpose the
    hidden on load and skip the PE ``nc_transpose`` + Vector quant (the
    ``mlpp_has_dma_xpose`` fast path). H_X4_MIDDLE is the bf16-input layout.
    """
    return MLPGateUpWeightLayout.H_X4_INNERMOST


# =============================================================================
# Attention (static FP8)
# =============================================================================


class LlamaAttention(nn.Module):
    """GQA attention with TP head sharding, static FP8 weights and scales.

    Structurally identical to :class:`vllm_neuron.model.llama3.model.LlamaAttention`
    except:
      * QKV / O weights are allocated in ``float8_e4m3fn``.
      * Four scale buffers are registered (qkv weight/input, o weight/input).
      * Prefill QKV projection passes STATIC kwargs to :func:`NF.qkv_proj`.
      * Prefill output projection passes STATIC kwargs to :func:`NF.o_proj`.
      * Decode passes STATIC kwargs to :func:`NF.attention_decode`.

    Parallelism, KV cache handling, RoPE, and flash/segmented attention are
    unchanged from the bf16 sibling.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        # Compute dtype (bf16/fp32) used for activations outside the quantized
        # matmul. Weights are FP8 regardless of this.
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        dcp_group = get_dcp_group()
        self.dcp_size = dcp_group.world_size
        self.dcp_group = dcp_group if self.dcp_size > 1 else None

        # >>> PARALLELISM: Dependent DP setup (decode-only Q/O sharding across DP) <<<
        self.attention_dp_size = (
            config.neuron_config.attention_dp_size if config.neuron_config else 1
        )
        self.attention_dp_group = None
        self.attention_dp_rank = 0
        if self.attention_dp_size > 1:
            from vllm_neuron.parallel.neuron_parallel_state import (
                get_neuron_attention_dp_group,
                get_neuron_attention_dp_rank,
            )

            self.attention_dp_group = get_neuron_attention_dp_group()
            self.attention_dp_rank = get_neuron_attention_dp_rank()

        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_attention_tp_group,
        )

        self.attn_tp_group = get_neuron_attention_tp_group()

        # Effective sharding degree for Q/O (TP for standard, TP*DDP for attention DP)
        effective_q_shards = self.world_size * self.attention_dp_size

        # >>> PARALLELISM: Head sharding calculation <<<
        self.num_attention_heads_per_rank = (
            self.num_attention_heads // effective_q_shards
        )

        self.kv_needs_a2a = (
            self.attention_dp_size > 1
            and self.num_key_value_heads > self.world_size
            and self.num_key_value_heads % effective_q_shards == 0
        )

        if self.world_size >= self.num_key_value_heads:
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            self.num_key_value_heads_per_rank = (
                self.num_key_value_heads // self.world_size
            )
            self.num_kv_replicas = 1

        self.num_kv_heads_for_weight = (
            self.num_key_value_heads // effective_q_shards
            if self.kv_needs_a2a
            else self.num_key_value_heads_per_rank
        )
        num_kv_heads_for_weight = self.num_kv_heads_for_weight

        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // num_kv_heads_for_weight
        )

        self.num_q_heads_after_a2a = (
            self.num_attention_heads_per_rank * self.attention_dp_size
        )
        self.num_kv_heads_after_a2a = (
            num_kv_heads_for_weight * self.attention_dp_size
            if self.kv_needs_a2a
            else self.num_key_value_heads_per_rank
        )

        # >>> PARALLELISM: QKV weight shapes <<<
        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = num_kv_heads_for_weight * self.head_dim
        qkv_size = q_size + 2 * kv_size
        o_proj_in_features = (
            self.num_attention_heads * self.head_dim
        ) // effective_q_shards

        # QKV (prefill MX, CTE layout): [H//4, qkv_size, 4] fp8.
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(
                self.hidden_size // _Q_WIDTH,
                qkv_size,
                _Q_WIDTH,
                dtype=_FP8_DTYPE,
            ),
            requires_grad=False,
        )

        # O proj: identical shape + dtype on both paths. STATIC_MX only
        # rearranges the byte content host-side (see
        # ``weight_pack_mx_fp8.mx_shuffle_o_proj``).
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_proj_in_features, self.hidden_size, dtype=_FP8_DTYPE)
        )

        # ── Dual-buffer (prefill CTE ≠ decode TKG layout) ─────────────
        # The prefill (CTE) and decode (TKG) MX kernels need different
        # packed QKV / O proj weight layouts. The canonical buffers carry
        # the prefill (CTE) layout and ``*_tkg`` mirror buffers carry the
        # decode (TKG) layout. Same source weights from the checkpoint,
        # two physical HBM copies. Memory cost: ~2× QKV + O proj weight
        # size; trn3 has the headroom.
        # MX TKG: [dim//4, other, 4] fp8
        self.qkv_proj_weight_tkg = nn.Parameter(
            torch.empty(
                self.hidden_size // _Q_WIDTH,
                qkv_size,
                _Q_WIDTH,
                dtype=_FP8_DTYPE,
            ),
            requires_grad=False,
        )
        self.o_proj_weight_tkg = nn.Parameter(
            torch.empty(
                o_proj_in_features // _Q_WIDTH,
                self.hidden_size,
                _Q_WIDTH,
                dtype=_FP8_DTYPE,
            ),
            requires_grad=False,
        )

        # Dequant / input scale shapes (STATIC_MX QKV CTE-via-MX):
        # weight [1, 3] / input [1, 1] (compact scalar); O proj [_PMAX, 1].
        qkv_w_scale_shape = (1, _QKV_FUSED)
        qkv_in_scale_shape = (1, 1)
        self.register_buffer(
            "qkv_weight_scale",
            torch.empty(*qkv_w_scale_shape, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "qkv_input_scale",
            torch.empty(*qkv_in_scale_shape, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "o_weight_scale",
            torch.empty(_PMAX, 1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "o_input_scale",
            torch.empty(_PMAX, 1, dtype=torch.float32),
            persistent=False,
        )

        # Scales for the decode (TKG) buffers. Kept separate from the
        # prefill (CTE) scales so each loader writes its own buffer.
        tkg_qkv_w_scale_shape = (1, _QKV_FUSED)
        tkg_qkv_in_scale_shape = (1, 1)
        self.register_buffer(
            "qkv_weight_scale_tkg",
            torch.empty(*tkg_qkv_w_scale_shape, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "qkv_input_scale_tkg",
            torch.empty(*tkg_qkv_in_scale_shape, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "o_weight_scale_tkg",
            torch.empty(_PMAX, 1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "o_input_scale_tkg",
            torch.empty(_PMAX, 1, dtype=torch.float32),
            persistent=False,
        )

        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        # KV caches bound externally via bind_kv_cache()
        self.k_cache = None
        self.v_cache = None
        self.fp8_packed = False

        # KV cache quantization scales (populated at weight-load time).
        self.k_scale = None
        self.v_scale = None
        self.k_scale_float = 1.0
        self.v_scale_float = 1.0

        self._setup_weight_loaders(config)

    # ------------------------------------------------------------------
    # Weight loaders
    # ------------------------------------------------------------------

    def _setup_weight_loaders(self, config: LlamaConfig):
        """Attach weight loaders.

        Attaches the MX prefill (CTE) loaders to the canonical
        ``qkv_proj_weight`` / ``o_proj_weight`` parameters and, in
        addition, attaches the MX decode (TKG) loaders to
        ``qkv_proj_weight_tkg`` / ``o_proj_weight_tkg`` (and the
        corresponding ``_tkg`` scales) via :class:`_TkgAlias` so
        ``forward_decode`` has decode-layout weights to read.
        """
        common_kwargs = dict(
            q_size=self.q_size,
            kv_size=self.kv_size,
            world_size=self.world_size,
            num_kv_replicas=self.num_kv_replicas,
            attention_dp_size=self.attention_dp_size,
            attention_dp_rank=self.attention_dp_rank,
            kv_needs_a2a=self.kv_needs_a2a,
            num_attention_heads=self.num_attention_heads,
            head_dim=self.head_dim,
        )
        loaders = _pick_loader_module()
        loaders.attach_attention_loaders(self, **common_kwargs)

        _mx_fp8_loaders.attach_attention_loaders_tkg(_TkgAlias(self), **common_kwargs)

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        ln_w: torch.Tensor = None,
        eps: float = 1e-5,
    ):
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
                ln_w=ln_w,
                eps=eps,
            )

        # <-- STATIC_MX (prefill): fused RMSNorm + fp8 quant in SP,
        # then all-gather fp8 result. QKV CTE kernel receives fp8 input
        # and skips internal quantization (_is_fp8_input=True path).
        if ln_w is not None:
            in_scale = self.qkv_input_scale
            if in_scale.shape[0] == 1:
                in_scale = in_scale.expand(128, 1)
            hidden_states = NF.rmsnorm_quant(
                hidden_states,
                ln_w=ln_w,
                input_dequant_scale=in_scale,
                eps=eps,
                quantization_type=_quant_type_rmsnorm(),
            )
        # >>> PARALLELISM: All-gather from SP after rmsnorm_quant <<<
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(
            hidden_states, positions, position_embeddings, attn_metadata
        )

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """Prefill path.

        Step 1 (QKV projection) and Step 5 (O projection) pass STATIC
        kernel kwargs. Steps 2–4 (RoPE, KV cache write, attention core)
        are unchanged from bf16.

        The activation dtype is bf16 on entry and stays bf16 throughout;
        the STATIC qkv_proj / o_proj kernels handle the internal fp8
        quant/dequant against the provided scales. We cast once at the
        top to match the reference FP8 Llama3 model's prefill contract.
        """
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)
        if self.dcp_size > 1:
            _validate_dcp_prefill_kv_cache(self.k_cache, self.v_cache)

        # Input is fp8 from rmsnorm_quant — do NOT cast to bf16.
        tokens, hidden = hidden_states.shape

        # ── Step 1: QKV Projection (STATIC, fused RoPE) ─────────────────
        cos, sin = position_embeddings
        cos_cache = cos.unsqueeze(0)
        sin_cache = sin.unsqueeze(0)
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
            d_head=self.head_dim,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            quantization_type=_quant_type_qkv_cte(),
            qkv_w_scale=self.qkv_weight_scale,
            qkv_in_scale=self.qkv_input_scale,
            weight_layout=_qkv_weight_layout(),
            num_q_heads=self.num_attention_heads_per_rank,
            num_kv_heads=self.num_kv_heads_for_weight,
        ).squeeze(0)

        qkv = qkv.to(self.dtype)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        k = k.view(tokens, self.num_kv_heads_for_weight, self.head_dim).transpose(0, 1)
        v = v.view(tokens, self.num_kv_heads_for_weight, self.head_dim).transpose(0, 1)

        # ── Step 3: KV cache update (unchanged) ────────────────────────
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        # DCP prefill writes the FULL sequence to the per-rank cache: slot_mapping
        # is full-length (owned interleaved tokens → real slots, non-owned →
        # reserved null block 0), so the scatter drops non-owned tokens harmlessly.
        # No K/V slicing; attention reads owned KV back from the cache.
        k_write, v_write = k, v

        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size

        if self.k_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            k_flat = (
                (k_write.reshape(-1, self.head_dim) * self.k_scale)
                .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                .to(self.k_cache.dtype)
            )
            v_flat = (
                (v_write.reshape(-1, self.head_dim) * self.v_scale)
                .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                .to(self.k_cache.dtype)
            )
        else:
            k_flat = k_write.reshape(-1, self.head_dim).to(self.k_cache.dtype)
            v_flat = v_write.reshape(-1, self.head_dim).to(self.k_cache.dtype)

        head_indices_for_put = torch.arange(
            self.num_key_value_heads_per_rank,
            dtype=torch.long,
            device=hidden_states.device,
        ).repeat_interleave(slot_mapping.shape[0])
        block_indices_for_put = block_indices.repeat(self.num_key_value_heads_per_rank)
        position_indices_for_put = position_indices.repeat(
            self.num_key_value_heads_per_rank
        )

        self.v_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            v_flat,
        )
        if self.fp8_packed:
            scatter_packed_k(
                self.k_cache,
                k_flat,
                block_indices_for_put,
                head_indices_for_put,
                position_indices_for_put,
            )
        else:
            self.k_cache.index_put_(
                (block_indices_for_put, head_indices_for_put, position_indices_for_put),
                k_flat,
            )

        # ── Step 4: Attention (unchanged) ───────────────────────────────
        if self.dcp_size > 1:
            attn_output = _segmented_attention_cp_prefill(
                q=q,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_table=block_table,
                cached_seq_len=cached_seq_len,
                block_size=block_size,
                dcp_size=self.dcp_size,
                dcp_rank=self.dcp_group.rank_in_group_tensor,
                dcp_group=self.dcp_group,
                scale=self.scaling,
            )
        elif kv_segment_size:
            kv_is_fp8 = self.k_cache.dtype in [
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            ]
            validate_fp8_segmented_supported(kv_is_fp8, self.fp8_packed)
            attn_output = NF.segmented_attention(
                q,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=block_table,
                prior_tokens=cached_seq_len,
                block_size=block_size,
                kv_segment_size=kv_segment_size,
                scale=self.scaling / self.k_scale_float,
                tp_q=True,
                tp_out=True,
                fp8_packed=self.fp8_packed,
                allow_vector_exp=False,
            )  # [Nh, Dh, T]
            attn_output = attn_output / self.v_scale_float
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
                allow_vector_exp=False,
            )

        # ── Step 5: Output Projection (STATIC) ──────────────────────────
        # <-- STATIC-FP8: attn_output is bf16 from the attention core; it
        # needs to be quantized inside the kernel using o_input_scale.
        attn_output = attn_output.unsqueeze(0)  # [1, Nh, Dh, T]
        attn_output = NF.o_proj(
            attn_output,
            self.o_proj_weight,
            None,
            quantization_type=_quant_type_o_proj_cte(),
            weight_scales=self.o_weight_scale,
            input_scales=self.o_input_scale,
        )
        attn_output = attn_output.squeeze(0)

        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
        ln_w: torch.Tensor = None,
        eps: float = 1e-5,
    ):
        """Decode path (fused megakernel) with STATIC_MX QKV and output.

        Decode reads the ``*_tkg`` decode-layout (TKG) weights + scales;
        the canonical buffers carry the prefill (CTE) layout which the
        decode kernel can't consume. On a real DI prefill-only instance
        vLLM never routes decode requests here, so the decode graph is
        only compiled — never executed; tracing with fake tensors is
        therefore safe.
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B_local = block_table.shape[0]
        B = B_local * self.attention_dp_size
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        assert tokens == B * S_decode
        _validate_dcp_decode_sequence_length(self.dcp_size, S_decode)

        hidden_states = hidden_states.to(self.dtype)
        S_ctx = max_blocks_per_seq * block_size
        nkh = self.num_key_value_heads_per_rank

        X = hidden_states.view(B, S_decode, hidden)

        # Pre-shuffle H for STATIC_MX QKV TKG kernel
        if ln_w is not None:
            x_f32 = X.to(torch.float32)
            var = x_f32.pow(2).mean(-1, keepdim=True)
            X = (x_f32 * torch.rsqrt(var + eps)).to(X.dtype) * ln_w.view(1, 1, -1)
            ln_w = None
        X = (
            X.view(B, S_decode, hidden // 512, 128, 4)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
            .view(B, S_decode, hidden)
        )

        cos, sin = position_embeddings
        half_d = self.head_dim // 2
        cos_kernel = (
            cos[:, :half_d]
            .view(B_local, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )
        sin_kernel = (
            sin[:, :half_d]
            .view(B_local, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )

        dcp_active = self.dcp_size > 1
        mask_q_heads = self.num_q_heads_after_a2a * (self.dcp_size if dcp_active else 1)

        local_filled = None
        dcp_active_mask = None
        if dcp_active:
            n = positions.view(B_local, S_decode)[:, :1].to(torch.float32)
            virtual_block_size = block_size * self.dcp_size
            full_virtual_blocks = torch.div(
                n, virtual_block_size, rounding_mode="trunc"
            )
            remaining = n - full_virtual_blocks * virtual_block_size
            dcp_rank = self.dcp_group.rank_in_group_tensor.to(torch.float32)
            local_filled = full_virtual_blocks * block_size + torch.clamp(
                remaining - dcp_rank * block_size,
                min=0,
                max=block_size,
            )
            owner = (
                torch.div(remaining, block_size, rounding_mode="trunc") % self.dcp_size
            )
            dcp_active_mask = (owner == dcp_rank).float()

        pos_ids = positions.view(1, B_local * S_decode)
        # Layer-invariant mask: cache on layer 0, reuse after (see key def).
        attention_mask = attn_metadata.get(_DECODE_MASK_CACHE_KEY)
        if attention_mask is None:
            attention_mask = NF.gen_attention_decode_mask(
                pos_ids=pos_ids.to(torch.float32),
                bs=B_local,
                q_head=mask_q_heads,
                s_active=S_decode,
                s_prior=S_ctx,
                start_pos=None,
                block_len=block_size,
                local_filled_slots=local_filled,
                dcp_active_mask=dcp_active_mask,
            )
            attn_metadata[_DECODE_MASK_CACHE_KEY] = attention_mask

        use_packed_kernel = self.fp8_packed and _packed_fp8_viable_for_bucket(
            block_len=block_size,
            bs=B_local,
            q_head=self.num_q_heads_after_a2a,
            s_active=S_decode,
            s_prior=S_ctx,
        )
        packed_fallback = self.fp8_packed and not use_packed_kernel
        k_cache = _unswizzle_packed_k(self.k_cache) if packed_fallback else self.k_cache
        v_cache = self.v_cache

        active_blocks_table = block_table

        # <-- STATIC_MX: pass STATIC_MX quant enum + four scale tensors
        # (qkv + output). KV-scale fusion into softmax/W_out is unchanged
        # from bf16. Dual-buffer (prefill CTE ≠ decode TKG layout): read
        # the ``_tkg`` decode-layout weights + scales; the canonical
        # buffers carry the prefill layout the decode kernel can't consume.
        qkv_w = self.qkv_proj_weight_tkg
        qkv_w_scale = self.qkv_weight_scale_tkg
        qkv_in_scale = self.qkv_input_scale_tkg
        o_w = self.o_proj_weight_tkg
        o_w_scale = self.o_weight_scale_tkg
        o_in_scale = self.o_input_scale_tkg
        # The packed-fallback bucket hands the kernel a COMPUTED TEMPORARY, whose
        # aliased in-kernel write is dropped; consume the alias OUTPUT instead.
        _needs_k_back = packed_fallback
        _decode_result = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            rmsnorm_X_enabled=ln_w is not None,
            rmsnorm_X_eps=eps if ln_w is not None else None,
            rmsnorm_X_gamma=ln_w.view(1, -1) if ln_w is not None else None,
            W_qkv=qkv_w,
            bias_qkv=None,
            quantization_type_qkv=_quant_type_qkv_tkg(),
            is_h_transposed_by_4=True,
            weight_dequant_scale_qkv=qkv_w_scale,
            input_dequant_scale_qkv=qkv_in_scale,
            rmsnorm_QK_pre_rope_enabled=False,
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=active_blocks_table,
            K_cache=k_cache,
            V_cache=v_cache,
            attention_mask=attention_mask,
            softmax_scale=self.scaling / self.k_scale_float,
            sink=None,
            update_cache=True,
            kv_cache_update_idx=slot_mapping.view(-1, S_decode).to(torch.uint32),
            W_out=o_w,
            bias_out=None,
            quantization_type_out=_quant_type_o_proj_tkg(),
            weight_dequant_scale_out=o_w_scale,
            input_dequant_scale_out=o_in_scale,
            transposed_out=False,
            out_in_sb=False,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
            fp8_packed=use_packed_kernel,
            attention_dp=self.attention_dp_size,
            attention_dp_group=self.attention_dp_group.device_group
            if self.attention_dp_group
            else None,
            attention_dp_rank=self.attention_dp_rank,
            attention_dp_rank_tensor=(
                self.attention_dp_group.rank_in_group_tensor
                if self.attention_dp_group is not None and self.attention_dp_size > 1
                else None
            ),
            kv_needs_a2a=self.kv_needs_a2a,
            dcp_size=self.dcp_size if dcp_active else 1,
            dcp_group=self.dcp_group if dcp_active else None,
            return_updated_kv=_needs_k_back,
        )
        if _needs_k_back:
            output, _k_updated, _v_updated = _decode_result
        else:
            output = _decode_result

        # Use the kernel's RETURNED K: k_cache is the PRE-kernel temp.
        if packed_fallback:
            self.k_cache.copy_(_swizzle_packed_k(_k_updated))

        self.attn_tp_group.all_reduce(output)
        return output


# =============================================================================
# MLP (static FP8)
# =============================================================================


class LlamaMLP(nn.Module):
    """Fused pre-MLP RMSNorm + SwiGLU MLP with TP intermediate sharding
    (static FP8).

    Unlike the bf16 sibling (which is MLP-only), this module owns **both
    the post-attention RMSNorm and the MLP** on the prefill path: the
    RMSNorm is fused with static fp8 quant via :func:`NF.rmsnorm_quant`
    inside ``forward`` to produce fp8 activations for :func:`NF.mlp`.
    The decoder layer must skip its plain ``post_attention_layernorm``
    on prefill and instead pass that norm's ``weight`` and
    ``config.rms_norm_eps`` into this module's ``forward``.

    Differs from :class:`vllm_neuron.model.llama3.model.LlamaMLP`:
      * gate/up/down weights are fp8e4m3fn.
      * Five scale buffers are registered (gate/up/down weight, gate_up/down input).
      * :func:`NF.mlp` is called with ``quantization_type=STATIC`` + scales.
      * :func:`NF.rmsnorm_quant` is fused into ``forward`` on prefill to
        produce fp8 activations; see the ``forward`` docstring.

    Pre-MLP RMSNorm fusion on prefill
    ---------------------------------
    The MLP kernel's STATIC path consumes fp8 activations (with
    ``gate_up_input_scale`` used internally to dequantize them back to
    the compute dtype for the gate/up matmul). To produce those fp8
    activations we fuse the pre-MLP RMSNorm with the static quant step
    via :func:`NF.rmsnorm_quant`. The norm weight (``ln_w``) and ``eps``
    are passed in from the decoder layer so the MLP does not have to
    own a duplicate RMSNorm parameter.

    Decoder-layer contract (prefill):
      * The decoder layer must **not** run the plain
        ``post_attention_layernorm`` before calling this MLP on prefill;
        instead it passes that norm's ``weight`` as ``ln_w`` and
        ``config.rms_norm_eps`` as ``eps``. Running the plain norm as
        well would double-apply RMSNorm.

    Decoder-layer contract (decode):
      * Unchanged. The preceding :class:`LlamaAttention.forward_decode`
        fused megakernel already emits bf16; the MLP kernel handles
        the static re-quantization of that bf16 input via
        ``gate_up_input_scale``. ``ln_w`` / ``eps`` are ignored on
        decode.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_draft_mlp_dp_group,
            get_neuron_draft_mlp_tp_group,
            get_neuron_mlp_tp_group,
            get_neuron_mlp_dp_group,
        )

        if config.use_draft_mlp_groups:
            mlp_tp_group = get_neuron_draft_mlp_tp_group()
            mlp_dp_group = get_neuron_draft_mlp_dp_group()
        else:
            mlp_tp_group = get_neuron_mlp_tp_group()
            mlp_dp_group = get_neuron_mlp_dp_group()
        self.mlp_tp_group = mlp_tp_group
        self.mlp_tp_size = mlp_tp_group.world_size
        self.mlp_tp_rank = mlp_tp_group.rank_in_group
        self.mlp_dp_group = mlp_dp_group
        self.mlp_dp_size = mlp_dp_group.world_size

        self.hidden_size = config.hidden_size
        self.intermediate_size_per_rank = config.intermediate_size // self.mlp_tp_size
        # <-- STATIC-FP8: record the compute dtype the MLP must emit so the
        # fp8 kernel output is cast back to bf16 (otherwise it would default
        # to ``hidden.dtype``, which on prefill is the fp8 output of
        # ``NF.rmsnorm_quant`` — the downstream residual add would crash).
        self.act_dtype = config.torch_dtype

        # MX (prefill / CTE) weight layouts: gate/up are 6-D ``H_X4_MIDDLE``
        # ``[_PMAX, H/512, I_padded/512, 4, _PMAX, 4]``, down is 4-D
        # ``[_PMAX, I_padded/512, H, 4]``.
        # The MX loaders pad the intermediate dim up to a multiple of 512
        # via ``ceil(I/_TILE_SIZE) * _TILE_SIZE``; the placeholder shape
        # must match what the loader emits or ``load_state_dict`` size-
        # checks fail (``strict=False`` only relaxes key presence).
        _TILE = _PMAX * _Q_WIDTH  # 512
        assert config.hidden_size % _TILE == 0
        n_h_tiles = config.hidden_size // _TILE
        n_i_tiles = (self.intermediate_size_per_rank + _TILE - 1) // _TILE
        self.gate_proj_weight = nn.Parameter(
            torch.zeros(
                _PMAX,
                n_h_tiles,
                n_i_tiles,
                _Q_WIDTH,
                _PMAX,
                _Q_WIDTH,
                dtype=_FP8_DTYPE,
            )
        )
        self.up_proj_weight = nn.Parameter(
            torch.zeros(
                _PMAX,
                n_h_tiles,
                n_i_tiles,
                _Q_WIDTH,
                _PMAX,
                _Q_WIDTH,
                dtype=_FP8_DTYPE,
            )
        )
        self.down_proj_weight = nn.Parameter(
            torch.zeros(
                _PMAX,
                n_i_tiles,
                config.hidden_size,
                _Q_WIDTH,
                dtype=_FP8_DTYPE,
            )
        )

        # ── Dual-buffer (prefill CTE ≠ decode TKG layout) ─────────────
        # Allocate the STATIC_MX TKG-layout mirrors so ``forward_decode``
        # can read the decode-layout weights. The MLP scales are the same
        # shape on both CTE/TKG — reuse them.
        _TILE = _PMAX * _Q_WIDTH
        n_h_tiles = config.hidden_size // _TILE
        n_i_tiles = (self.intermediate_size_per_rank + _TILE - 1) // _TILE
        # 6D H_X4_MIDDLE: [128_H, H/512, I/512, 4_I, 128_I, 4_H]
        self.gate_proj_weight_tkg = nn.Parameter(
            torch.zeros(
                _PMAX,
                n_h_tiles,
                n_i_tiles,
                _Q_WIDTH,
                _PMAX,
                _Q_WIDTH,
                dtype=_FP8_DTYPE,
            ),
            requires_grad=False,
        )
        self.up_proj_weight_tkg = nn.Parameter(
            torch.zeros(
                _PMAX,
                n_h_tiles,
                n_i_tiles,
                _Q_WIDTH,
                _PMAX,
                _Q_WIDTH,
                dtype=_FP8_DTYPE,
            ),
            requires_grad=False,
        )
        # Down: [128_I, I/512, H, 4_I]
        self.down_proj_weight_tkg = nn.Parameter(
            torch.zeros(
                _PMAX,
                n_i_tiles,
                config.hidden_size,
                _Q_WIDTH,
                dtype=_FP8_DTYPE,
            ),
            requires_grad=False,
        )

        # <-- STATIC-FP8: per-projection dequant scales + per-stage input
        # scales (gate/up share one input scale; down has its own because
        # it takes the activation function's output).
        for name in (
            "gate_weight_scale",
            "up_weight_scale",
            "down_weight_scale",
            "gate_up_input_scale",
            "down_input_scale",
        ):
            self.register_buffer(
                name,
                torch.empty(_PMAX, 1, dtype=torch.float32),
                persistent=False,
            )

        # Dual-buffer (prefill CTE ≠ decode TKG): separate scale buffers
        # for the decode path. Buffer shapes match the prefill path (the
        # kernels accept the same ``[_PMAX, 1]`` broadcast).
        for name in (
            "gate_weight_scale_tkg",
            "up_weight_scale_tkg",
            "down_weight_scale_tkg",
            "gate_up_input_scale_tkg",
            "down_input_scale_tkg",
        ):
            self.register_buffer(
                name,
                torch.empty(_PMAX, 1, dtype=torch.float32),
                persistent=False,
            )

        self._setup_weight_loaders(config)

    def _setup_weight_loaders(self, config):
        """Attach MLP loaders.

        The canonical gate/up/down weights live in the STATIC_MX prefill
        (CTE) 6-D / 4-D layouts (loaded via the MX loader) and
        ``forward_decode`` reads the ``*_tkg`` decode (TKG) layout buffers
        loaded via the MX TKG loader through :class:`_TkgAlias`. Both
        copies share the MLP scales (same [_PMAX, 1] shape on both).
        """
        common_kwargs = dict(
            intermediate_size_per_rank=self.intermediate_size_per_rank,
            mlp_tp_size=self.mlp_tp_size,
            mlp_tp_rank=self.mlp_tp_rank,
            hidden_size=config.hidden_size,
        )
        loaders = _pick_loader_module()
        loaders.attach_mlp_loaders(self, **common_kwargs)

        _mx_fp8_loaders.attach_mlp_loaders_tkg(_TkgAlias(self), **common_kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        is_prefill: bool,
        ln_w: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Run the pre-MLP RMSNorm + static-FP8 MLP.

        The norm is fused with the static fp8 quant. The fusion happens in
        different places on prefill vs decode (to match the reference):

        * **Prefill:** explicit :func:`NF.rmsnorm_quant` before
          :func:`NF.mlp`; kernel is called with ``norm_type=NO_NORM``
          since the norm already ran. Required because CTE mode
          (large-seqlen path) does not support fused norm+quant inside
          :func:`NF.mlp`.
        * **Decode:** :func:`NF.mlp` with ``norm_type=RMS_NORM`` and
          ``ln_w=ln_w.view(1, -1)``. TKG mode (small-B×S path) fuses
          the norm and the static quant in one kernel.

        Either way, the decoder layer must **skip** its plain
        ``post_attention_layernorm`` and forward the norm weight and eps
        via ``ln_w`` / ``eps`` here.

        Args:
            hidden_states: Residual-stream activation in the compute
                dtype (bf16). Shape ``[T_local, H]`` on prefill (SP
                slice) or ``[T, H]`` on decode.
            is_prefill: Prefill vs decode path selector (matches the
                bf16 sibling).
            ln_w: ``post_attention_layernorm.weight`` from the decoder
                layer. Required on both paths now.
            eps: RMSNorm epsilon. Required on both paths.
        """
        if ln_w is None:
            raise ValueError(
                "LlamaMLP (static FP8) requires ln_w on both prefill and "
                "decode; the decoder layer must pass "
                "post_attention_layernorm.weight."
            )

        if is_prefill:
            # <-- STATIC-FP8 (prefill): rmsnorm_quant in SP region, then gather.
            hidden_states = NF.rmsnorm_quant(
                hidden_states,
                ln_w=ln_w,
                input_dequant_scale=self.gate_up_input_scale,
                eps=eps,
                quantization_type=_quant_type_rmsnorm(),
            )
            # >>> PARALLELISM: All-gather from SP after rmsnorm_quant. Use the
            # MLP TP group (TP * mlp_dp_size), because that is the group the
            # intermediate dimension shards across. See LlamaMLP.forward. <<<
            if self.mlp_tp_size > 1:
                hidden_states = self.mlp_tp_group.all_gather(hidden_states, dim=0)
            mlp_norm_type = NormType.NO_NORM
            mlp_ln_w = None
        else:
            # <-- STATIC-FP8 (decode): TKG mode fuses RMSNorm + static
            # quant inside NF.mlp; hand it the (1, H)-shaped gamma and
            # let the kernel do both.
            mlp_norm_type = NormType.RMS_NORM
            mlp_ln_w = ln_w.view(1, -1)

        # Dual-buffer (prefill CTE ≠ decode TKG layout): decode reads the
        # ``*_tkg`` mirrors with the decode-layout weights + scales; the
        # else branch (prefill) reads the canonical CTE buffers.
        if not is_prefill:
            gate_w = self.gate_proj_weight_tkg
            up_w = self.up_proj_weight_tkg
            down_w = self.down_proj_weight_tkg
            gate_w_scale = self.gate_weight_scale_tkg
            up_w_scale = self.up_weight_scale_tkg
            down_w_scale = self.down_weight_scale_tkg
            gate_up_in_scale = self.gate_up_input_scale_tkg
            down_in_scale = self.down_input_scale_tkg
            mlp_quant_type = _quant_type_mlp_tkg()
            # STATIC_MX TKG gate/up layout.
            mlp_layout = MLPGateUpWeightLayout.H_X4_MIDDLE
        else:
            gate_w = self.gate_proj_weight
            up_w = self.up_proj_weight
            down_w = self.down_proj_weight
            gate_w_scale = self.gate_weight_scale
            up_w_scale = self.up_weight_scale
            down_w_scale = self.down_weight_scale
            gate_up_in_scale = self.gate_up_input_scale
            down_in_scale = self.down_input_scale
            mlp_quant_type = _quant_type_mlp_cte()
            mlp_layout = _mlp_gate_up_w_layout()

        output = NF.mlp(
            hidden_states,
            gate_w,
            up_w,
            down_w,
            eps=eps,
            ln_w=mlp_ln_w,
            norm_type=mlp_norm_type,
            quantization_type=mlp_quant_type,
            gate_w_scale=gate_w_scale,
            up_w_scale=up_w_scale,
            down_w_scale=down_w_scale,
            gate_up_in_scale=gate_up_in_scale,
            down_in_scale=down_in_scale,
            gate_up_w_layout=mlp_layout,
            output_dtype="bfloat16",
        )

        # >>> PARALLELISM: Combine TP (+ MLP DP) shards. Both branches reduce
        # over the MLP TP group, because every rank in that group holds one
        # shard of the intermediate dimension. See LlamaMLP.forward. <<<
        if is_prefill:
            if self.mlp_tp_size > 1:
                output = self.mlp_tp_group.reduce_scatter(output, dim=0)
        else:
            self.mlp_tp_group.all_reduce(output)

        return output
