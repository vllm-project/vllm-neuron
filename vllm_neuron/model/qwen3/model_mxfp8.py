# SPDX-License-Identifier: Apache-2.0
"""Qwen3 dense MXFP8 implementation.

MXFP8 variant of :mod:`vllm_neuron.model.qwen3.model` for compressed-tensors
checkpoints with groupwise FP8 weights/scales. This is a kernel-enabled POC
that reuses the native-MX attention and MLP layouts already proven in the
Qwen3-VL text path, while keeping the dense Qwen3 modeling contracts:
standard rotate-half RoPE, segmented-prefill support, and the dense top-level
model surface.
"""

from __future__ import annotations

import logging
import math

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron import envs
from vllm_neuron.utils.neuron_utils import can_run_kernel
from nkilib.core.mlp.mlp_parameters import (
    TKG_BS_SEQLEN_THRESHOLD as _MLP_TKG_BS_SEQLEN_THRESHOLD,
)
from nkilib.core.utils.common_types import (
    MLPGateUpWeightLayout,
    NormType,
    QKVWeightLayout,
    QuantizationType,
)
from vllm_neuron.functional.attention.attention_decode import (
    _swizzle_packed_k,
    _unswizzle_packed_k,
    packed_fp8_viable_for_bucket,
    scatter_packed_k,
)
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.dtype_utils import (
    FP8_CLAMP_MAX,
    validate_fp8_segmented_supported,
)
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import Qwen3Config
from .model import (
    _decode_sp_viable,
    Qwen3Attention as Qwen3AttentionBf16,
    Qwen3DecoderLayer as Qwen3DecoderLayerBf16,
    Qwen3ForCausalLM as Qwen3ForCausalLMBf16,
    Qwen3MLP as Qwen3MLPBf16,
    Qwen3Model as Qwen3ModelBf16,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from .quantization import QuantScheme, QuantizationSpec

from ..kv_cache import paged_slot_indices

# TODO: extract these decode-time KV helpers to a model-neutral shared module so
# dense Qwen3 does not import through the Qwen3-VL package tree.
from ..qwen3_vl.utils.decode_kv import build_per_head_block_table, scatter_new_kv
from .weight_loaders_mxfp8 import (
    _FP8_DTYPE,
    _PMAX,
    _Q_WIDTH,
    _SCALE_FOLD_ROWS,
    _SCALE_FOLD_SLOTS,
    _SCALES_PER_TILE,
    _TILE_SIZE,
    _n_i512_tiles,
    _n_i512_tiles_tkg,
    MX_GROUP_SIZE,
    gate_up_i_pad_tkg,
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
    o_proj_scale_loader,
    o_proj_weight_loader,
    pre_shuffle_h,
)

# Model-specific kernel swap (on by default): run the decode MLP with the in-tree
# _kernels/mlp kernel instead of the shared nkilib kernel. This choice is
# Qwen3-32B-specific, so it lives here in the model, not in the shared functional
# dispatcher. Resolve the switch once at import (outside the traced forward), so
# the graph sees a plain Python constant and no import runs inside forward.
# The kernel is on by default. Disable it with VLLM_NEURON_QWEN3_INTREE_MLP=0.
# TODO: this dedicated flag is a bring-up escape hatch to A/B the in-tree kernel
# against the shared nkilib MLP while it proves out. Remove it once the kernel is
# established and rely on the general VLLM_NEURON_DISABLE_NKI_KERNELS switch.
_QWEN3_INTREE_MLP = envs.VLLM_NEURON_QWEN3_INTREE_MLP
# Which configs the in-tree decode MLP path can run. Three rules apply:
#
#   1. Geometry. H and I must match Qwen3-32B dense. The kernel constants and the
#      MX weight/scale packing assume this model.
#   2. MX groups. I/rank must be a multiple of 32, so the per-rank E8M0 scale row
#      count of the down projection is exact.
#   3. PSUM. The kernel gives one PSUM bank to each padded I512 tile, so
#      _n_i512_tiles_tkg(I/rank) must not exceed _PSUM_BMAX_TILES.
#
# For Qwen3-32B (H=5120, I=25600) the rules admit TP 8, 16 and 32:
#
#   tp   I/rank   I/rank % 32   TKG tiles   eligible
#    4     6400             0          13   no (rule 3: needs 13 banks, 8 exist)
#    8     3200             0           7   yes
#   16     1600             0           4   yes
#   32      800             0           2   yes
#   64      400            16           1   no (rule 2: 400 % 32 = 16)
#
# TP4 first motivated this gate. A TP4 run of the decode MLP module test failed
# its numeric guard at 42.5x L-inf against a 5x limit. Rule 3 keeps TP4 out, and
# the kernel repeats the same bank assert, so a TP4 trace stops with a clear
# message. The cause of the 42.5x result is not identified, so keep rule 3 even if
# the bank bound grows later.
#
# See qwen3_intree_mlp.py for the layout contract.
_QWEN3_32B_HIDDEN = 5120
_QWEN3_32B_INTERMEDIATE = 25600
_PSUM_BMAX_TILES = 8  # PSUM banks; mirrors _PSUM_BMAX in the kernel


def _intree_mlp_reject_reason(
    hidden_size: int, intermediate_size_per_rank: int, world_size: int
) -> str | None:
    """Return why the in-tree decode MLP path refuses this config, else None.

    The gate and the fallback log message read the same rules from this one
    function, so the log always names the rule that applies.
    """

    intermediate_size = intermediate_size_per_rank * world_size
    if hidden_size != _QWEN3_32B_HIDDEN or intermediate_size != _QWEN3_32B_INTERMEDIATE:
        return (
            f"geometry H={hidden_size}, I={intermediate_size} is not Qwen3-32B "
            f"dense (H={_QWEN3_32B_HIDDEN}, I={_QWEN3_32B_INTERMEDIATE})"
        )
    if intermediate_size_per_rank % MX_GROUP_SIZE:
        return (
            f"I/rank={intermediate_size_per_rank} is not a multiple of "
            f"{MX_GROUP_SIZE}, so the per-rank E8M0 scale row count is inexact"
        )
    tiles = _n_i512_tiles_tkg(intermediate_size_per_rank)
    if tiles > _PSUM_BMAX_TILES:
        return (
            f"I/rank={intermediate_size_per_rank} needs {tiles} PSUM banks and "
            f"only {_PSUM_BMAX_TILES} exist"
        )
    return None


def _intree_mlp_supports(
    hidden_size: int, intermediate_size_per_rank: int, world_size: int
) -> bool:
    """True if the in-tree decode MLP path can load and run this config."""

    return (
        _intree_mlp_reject_reason(hidden_size, intermediate_size_per_rank, world_size)
        is None
    )


# One log line per distinct MLP kernel decision. Every decoder layer builds the
# same config, so a per-layer log gives 64 identical lines for each rank.
_intree_mlp_logged: set[tuple[int, int, bool]] = set()


# Token counts where neuronx-cc cannot compile the FUSED decode MLP. The fused
# kernel reads the raw pre-norm activation, so the residual add that follows
# consumes the same tensor that feeds the kernel's layout transpose. The compiler
# couples the two layouts, and at these counts the access pattern it derives for
# the transpose trips an assertion:
#
#   transpose.1_pftranspose_<N> [INTERNAL_ERROR] [NCC_ILSA901]
#   LegalizeSundaAccess assertion error: unexpected AP of matmult dst
#
# Measured on trn3 (neuronxcc 2.0.404505) at TP8, on both LNC degrees: 2 and 4
# fail, and 1, 8, 16, 32, and 64 compile. The count that matters is the residual
# add's, not the transpose's -- a padded activation still fails. Every layout
# respelling fails too, and so does every value-preserving copy of either operand,
# because the compiler folds it away. Only a reduction between the two tensors
# breaks the coupling, so the fallback here is the un-fused path, whose separate
# RMSNorm supplies exactly that.
#
# The measurement covers TP8. The list applies at every eligible degree, because
# the un-fused fallback is correct everywhere, and a degree that does compile the
# fusion loses it only at these two counts. Measure TP16 and TP32 before you
# shorten the list.
#
# These counts are served: decode buckets are powers of 2 (see
# get_default_num_seqs_buckets), and the runtime pads the live sequence count up,
# so batch 2, 3, and 4 all land here. Short prefill reaches them too.
#
# TODO(<ticket>): a frontend graph must not be able to make the compiler assert.
# Delete this list and the branch it drives when the compiler fix ships.
_ILSA901_TOKEN_COUNTS = (2, 4)

# Platform-target family that carries the MX scale partition-offset addressing mode
# (MxMatmulScalePidx). The folded decode scale layout needs it.
_SCALE_PIDX_PLATFORM_PREFIX = "trn3"


def _target_supports_scale_pidx() -> bool:
    """Return True when the platform target carries the MX scale start offset.

    The folded decode scale layout reads a scale at a partition start offset of
    4, 8 or 12. That addressing mode landed for trn3/gen4 only, and the compiler
    rejects it on trn1 and trn2, so the caller must keep the pre-scattered layout
    there.

    Returns:
        True on a trn3 platform target, else False.

    Example:
        >>> isinstance(_target_supports_scale_pidx(), bool)
        True
    """
    try:
        from libtorch_neuronx_lite.compile.platform import get_platform_target

        return str(get_platform_target()).startswith(_SCALE_PIDX_PLATFORM_PREFIX)
    except Exception:
        # No target is readable outside a compile context (CPU mode, unit tests).
        # Choose the layout every target accepts.
        return False


_qwen3_intree_decode_fn = None
if _QWEN3_INTREE_MLP:
    from vllm_neuron.functional.qwen3_intree_mlp import (
        qwen3_32b_mxfp8_mlp_tkg_decode as _qwen3_intree_decode_fn,
    )
else:
    logging.getLogger(__name__).warning(
        "Qwen3-32B MXFP8 decode: the in-tree _kernels/mlp kernel is DISABLED "
        "(VLLM_NEURON_QWEN3_INTREE_MLP=0). The shared nkilib kernel runs instead."
    )

# One kernel fuses post-attention RMSNorm + MX quantize + SBUF token all-gather + MLP, extending the in-tree decode MLP.
_qwen3_sp_a2a_mlp_decode_fn = None
_qwen3_sp_a2a_supports = None
if _QWEN3_INTREE_MLP:
    try:
        from vllm_neuron._kernels.qwen3_32b_sp.qwen3_32b_sp_norm_mlp import (
            qwen3_32b_mxfp8_sp_a2a_mlp_decode as _qwen3_sp_a2a_mlp_decode_fn,
        )
        from vllm_neuron._kernels.qwen3_32b_sp.qwen3_32b_sp_norm_mlp import (
            qwen3_sp_a2a_supports as _qwen3_sp_a2a_supports,
        )
    except ImportError:
        logging.getLogger(__name__).warning(
            "Qwen3-32B MXFP8 sequence-parallel decode: the fused norm+MLP kernel "
            "is not available. The separate norm and the HBM all-gather run "
            "instead. Install a build that ships "
            "vllm_neuron._kernels.qwen3_32b_sp.qwen3_32b_sp_norm_mlp to get the fused path."
        )

# Largest decode token count the in-tree MLP kernel takes in one entry. The down
# projection makes the intermediate stationary, so the token axis is that operand's
# free axis and nc_matmul_mx caps it at 128. A larger batch needs token tiling, which
# this kernel does not do.
_INTREE_MAX_DECODE_TOKENS = 128

_SUPPORTED_QWEN3_MXFP8_SCHEMES = {
    QuantScheme.MXFP8_DYNAMIC_PER_TOKEN_CHANNELWISE,
}


def _decode_only_weights(config: Qwen3Config) -> bool:
    """Return True when this pool serves decode only.

    platform.py derives the flag from ``kv_role``. The tests and the CPU paths
    build a config with no ``neuron_config``, so default to False there: an
    unnecessary prefill weight wastes HBM, a missing one breaks prefill.
    """

    return bool(
        config.neuron_config is not None and config.neuron_config.decode_only_weights
    )


# Log the LNC fallback once; __init__ calls the gate for every layer.
_decode_sp_lnc_logged = False


def _decode_sp_enabled(config: Qwen3Config) -> bool:
    """Return True when this pool asks for sequence-parallel decode. Prefill is
    always sequence-parallel; this extends the layout to decode. Defaults to
    False when ``neuron_config`` is absent (tests, CPU paths). Read once in
    ``__init__``: a lazy read inside ``forward`` changes the torch.compile guards.

    Sequence-parallel decode is LNC1-only. At another LNC degree the pool keeps
    the all-reduce decode path, and one INFO line gives the reason.
    """

    requested = bool(
        config.neuron_config is not None
        and config.neuron_config.enable_decode_sequence_parallel
    )
    if not requested:
        return False
    lnc = get_logical_nc_config()
    if lnc != 1:
        global _decode_sp_lnc_logged
        if not _decode_sp_lnc_logged:
            _decode_sp_lnc_logged = True
            logging.getLogger(__name__).info(
                "Sequence-parallel decode is LNC1-only. This pool runs at "
                "NEURON_LOGICAL_NC_CONFIG=%d, so decode keeps the all-reduce path.",
                lnc,
            )
        return False
    return True


def _use_inkernel_decode_kv_write(fp8_packed: bool, num_kv_heads_per_rank: int) -> bool:
    """Return whether decode may update the KV cache inside the megakernel.

    Keep BF16 KV writes outside the attention launch. Each decode query reads
    through its own position, so the cache write overlaps the cache read range.
    The shared kernel does not synchronize that overlap across NeuronCores,
    which makes the attention result nondeterministic.

    Preserve the existing packed-FP8 single-head path until packed KV updates
    have a generally supported external scatter.
    """

    return fp8_packed and num_kv_heads_per_rank == 1


def _normalize_decode_k_for_external_scatter(
    k_new: torch.Tensor, num_kv_heads_per_rank: int
) -> torch.Tensor:
    """Restore the singleton KV-head axis that the kernel drops at one head."""

    if num_kv_heads_per_rank == 1 and k_new.dim() == 3:
        return k_new.unsqueeze(2)
    return k_new


def _rmsnorm_row_pack(
    hidden_states: torch.Tensor, ln_w: torch.Tensor, eps: float
) -> torch.Tensor:
    # Prefill MX kernels consume ROW-quantized activations directly, so the
    # layernorm step also performs the pack from bf16 [T, H] to [T, H + scale].
    return NF.rmsnorm_quant(
        hidden_states,
        ln_w,
        None,
        eps=eps,
        quantization_type=QuantizationType.ROW,
    )


class Qwen3AttentionMxfp8(Qwen3AttentionBf16):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)

        fused_i = self.q_size + 2 * self.kv_size
        assert self.num_attention_heads % self.world_size == 0, (
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
            f"world_size ({self.world_size}) for the O-proj sharding."
        )
        self.nd_per_rank = self.num_attention_heads * self.head_dim // self.world_size
        nd_per_rank = self.nd_per_rank
        hidden_size = self.hidden_size

        # A decode-only pool traces no prefill graph, so its prefill-layout
        # weights are unreachable HBM. Skip them and give the KV cache the bytes.
        self.decode_only_weights = _decode_only_weights(config)

        del self.qkv_proj_weight
        del self.o_proj_weight

        # Decode and prefill use different native-MX layouts for O-proj. A
        # decode-only pool keeps the decode form only.
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(
                hidden_size // _Q_WIDTH,
                fused_i,
                _Q_WIDTH,
                dtype=_FP8_DTYPE,
            ),
            requires_grad=False,
        )
        self.qkv_weight_scale = nn.Parameter(
            torch.empty(hidden_size // MX_GROUP_SIZE, fused_i, dtype=torch.uint8),
            requires_grad=False,
        )
        self.o_proj_weight_decode = nn.Parameter(
            torch.empty(
                nd_per_rank // _Q_WIDTH, hidden_size, _Q_WIDTH, dtype=_FP8_DTYPE
            ),
            requires_grad=False,
        )
        if not self.decode_only_weights:
            self.o_proj_weight_prefill = nn.Parameter(
                torch.empty(nd_per_rank // _Q_WIDTH, hidden_size, dtype=torch.uint32),
                requires_grad=False,
            )
        self.o_proj_weight_scale = nn.Parameter(
            torch.empty(nd_per_rank // MX_GROUP_SIZE, hidden_size, dtype=torch.uint8),
            requires_grad=False,
        )

        # Whether the bound FP8 K cache uses the swizzled packed layout. Only
        # knowable at bind_kv_cache time (from the cache rank), not here -- see
        # bind_kv_cache's `k_cache.dim() == v_cache.dim() + 1` probe. Default
        # False so a bf16 or non-packed FP8 cache takes the standard kernel path.
        self.fp8_packed = False

        self._setup_weight_loaders()

    @property
    def _decode_o_proj_weight(self) -> torch.Tensor:
        return self.o_proj_weight_decode

    def _setup_weight_loaders(self) -> None:
        if not hasattr(self, "qkv_weight_scale"):
            return super()._setup_weight_loaders()

        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader_3d(self.world_size, self.num_kv_replicas),
        )
        set_weight_loader(
            self.qkv_weight_scale,
            fused_qkv_scale_loader_3d(self.world_size, self.num_kv_replicas),
        )
        set_weight_loader(
            self.o_proj_weight_decode,
            o_proj_weight_loader(self.nd_per_rank, self.world_size, pack_3d=True),
        )
        if not self.decode_only_weights:
            set_weight_loader(
                self.o_proj_weight_prefill,
                o_proj_weight_loader(self.nd_per_rank, self.world_size, pack_3d=False),
            )
        set_weight_loader(
            self.o_proj_weight_scale,
            o_proj_scale_loader(self.nd_per_rank, self.world_size),
        )

    def _decode_cos_sin(
        self,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        batch_size: int,
        s_decode: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = position_embeddings
        half_d = self.head_dim // 2
        # The decode megakernel consumes the rotate-half RoPE tables in
        # [half_d, batch, seq] form rather than the dense model's default view.
        cos_kernel = (
            cos[:, :half_d]
            .view(batch_size, s_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )
        sin_kernel = (
            sin[:, :half_d]
            .view(batch_size, s_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )
        return cos_kernel, sin_kernel

    def _run_decode_megakernel(
        self,
        hidden_states: torch.Tensor,
        cos_kernel: torch.Tensor,
        sin_kernel: torch.Tensor,
        active_blocks_table: torch.Tensor,
        positions: torch.Tensor,
        k_cache_arg: torch.Tensor,
        use_packed_kernel: bool,
        kv_cache_update_idx: torch.Tensor | None,
        gamma_shuffled: torch.Tensor | None = None,
        eps: float | None = None,
        return_updated_kv: bool = False,
        sp_a2a_partials: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_norm_w = self.q_norm.weight.view(self.head_dim, 1)
        k_norm_w = self.k_norm.weight.view(self.head_dim, 1)

        # ``gamma_shuffled`` is the input RMSNorm gamma, pre-shuffled at weight
        # load to the same flat-H layout as the hidden state
        # (is_h_transposed_by_4). Pass it straight to the kernel, so the reshape
        # and the permute stay off the decode critical path. The kernel skips
        # layout validation, so a natural-order gamma silently corrupts output.
        #
        # The megakernel arguments both entries below share. The SP a2a entry
        # lives in the SP kernel package, NOT in the shared attention_decode
        # (which stays staging-identical and serves every other model); it
        # replaces X and the rmsnorm_X_* fusion with the in-kernel prologue.
        megakernel_kwargs = dict(
            W_qkv=self.qkv_proj_weight,
            quantization_type_qkv=QuantizationType.MX,
            weight_dequant_scale_qkv=self.qkv_weight_scale,
            input_dequant_scale_qkv=None,
            is_h_transposed_by_4=True,
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=self.q_norm.variance_epsilon,
            rmsnorm_QK_pre_rope_W_Q=q_norm_w,
            rmsnorm_QK_pre_rope_W_K=k_norm_w,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=active_blocks_table,
            K_cache=k_cache_arg,
            V_cache=self.v_cache,
            pos_ids=positions,
            swa_start_pos_ids=None,
            # FP8 KV: the K-scale is folded into softmax_scale and the V-scale is
            # applied by the kernel (is_fp8_kv = k_scale is not None). Unlike the
            # bf16 path we must NOT divide W_out by v_scale_float -- W_out is a
            # packed fp8/uint32 MX buffer, so an elementwise divide is illegal;
            # llama3's MX decode does the same (passes the scales, never scales
            # W_out). With a bf16 KV cache k_scale/v_scale are None and
            # k_scale_float == 1.0, so this is bit-identical to the prior path.
            softmax_scale=self.scaling / self.k_scale_float,
            # KV write strategy (see forward_decode). kv_cache_update_idx is set
            # only for the packed-FP8 single-head path: then the megakernel writes
            # K/V into K_cache/V_cache in place and returns the attention output.
            # Otherwise update_cache=False and the caller scatters K/V externally
            # from the returned (output, k_new, v_new).
            # The in-kernel write is NOT TP-general: kv_cache_update_idx encodes
            # (batch, position) with no KV-head dim, so at 2+ KV heads/rank the
            # NKI kernel fails to compile (revert 8d631c979 of cb928bd38).
            update_cache=kv_cache_update_idx is not None,
            kv_cache_update_idx=kv_cache_update_idx,
            W_out=self._decode_o_proj_weight,
            quantization_type_out=QuantizationType.MX,
            weight_dequant_scale_out=self.o_proj_weight_scale,
            input_dequant_scale_out=None,
            transposed_out=False,
            out_in_sb=False,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
            fp8_packed=use_packed_kernel,
            # Packed fallback passes a computed temporary; its aliased in-kernel
            # write is dropped, so ask for the alias outputs.
            return_updated_kv=return_updated_kv,
        )

        # sp_a2a_partials is None on every non-SP forward and on layer 0 of an
        # SP step (the carry starts after its MLP), so the shared functional
        # entry below runs exactly as on staging.
        if sp_a2a_partials is not None:
            from vllm_neuron._kernels.qwen3_32b_sp.qwen3_32b_sp_norm_attn import (
                attention_decode_sp_a2a,
            )
            from vllm_neuron._kernels.qwen3_32b_sp.qwen3_32b_sp_norm_mlp import (
                _resolve_replica_mesh,
            )

            assert gamma_shuffled is not None and eps is not None, (
                "the a2a attention prologue needs gamma_shuffled and eps for "
                "the input RMSNorm."
            )
            return attention_decode_sp_a2a(
                sp_a2a_partials,
                hidden_states,
                gamma_shuffled,
                eps,
                self.world_size,
                _resolve_replica_mesh(self.tp_group),
                **megakernel_kwargs,
            )

        return NF.attention_decode(
            X=hidden_states,
            rmsnorm_X_enabled=gamma_shuffled is not None,
            rmsnorm_X_eps=eps,
            rmsnorm_X_gamma=gamma_shuffled,
            **megakernel_kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        gamma_shuffled: torch.Tensor | None = None,
        eps: float | None = None,
        decode_sp_active: bool = False,
        sp_a2a_partials: torch.Tensor | None = None,
    ):
        """Dispatch to prefill or decode. ``gamma_shuffled`` is the H-4 shuffled
        ``input_layernorm`` gamma and ``eps`` its epsilon; the decoder layer sets
        them on decode only, to fuse the input RMSNorm into the megakernel, and
        prefill ignores them. ``decode_sp_active`` says the caller gives a token shard
        and wants the o_proj partials back UN-REDUCED for the a2a MLP composer;
        only the decoder layer knows the residual stream is also sharded."""
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        # Short queries stay on the decode megakernel path; longer prompts use
        # the prefill kernels and, when TP > 1, first rebuild the full hidden
        # state because the ROW-packed prefill projection is not rank-local.
        if max_query_len <= decode_token_threshold:
            # KV heads are sharded, not tokens, so every rank needs every token. This bf16 gather runs only at a2a layer 0 (no carry yet) and when the attention composer is unavailable; the a2a attention prologue absorbs the whole boundary otherwise. pre_shuffle_h commutes with the gather (test/unit/test_qwen3_rmsnorm_shuffle.py).
            if decode_sp_active and sp_a2a_partials is None:
                hidden_states = pre_shuffle_h(hidden_states.to(self.dtype))
                hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
            return self.forward_decode(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
                gamma_shuffled=gamma_shuffled,
                eps=eps,
                decode_sp_active=decode_sp_active,
                h_preshuffled=decode_sp_active,
                sp_a2a_partials=sp_a2a_partials,
            )

        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(
            hidden_states, positions, position_embeddings, attn_metadata
        )

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)
        # Keep this check after the warmup early return above. A decode-only LNC-1
        # process traces this path but never runs it, so an earlier check would stop
        # the trace.
        if self.decode_only_weights:
            raise RuntimeError(
                "Qwen3 MXFP8 got a prefill request on a decode-only pool, which "
                "holds no prefill weights. Run prefill on a kv_producer or "
                "kv_both server."
            )
        if get_logical_nc_config() != 2:
            raise NotImplementedError(
                "Qwen3 MXFP8 prefill runs at LNC=2 only. Set "
                "NEURON_LOGICAL_NC_CONFIG=2. The decode path also runs at LNC=1."
            )

        assert hidden_states.shape[-1] == self.hidden_size + _Q_WIDTH, (
            f"Qwen3AttentionMxfp8.forward_prefill expects ROW-packed [T, H+{_Q_WIDTH}] "
            f"input; got last dim {hidden_states.shape[-1]} (H={self.hidden_size})."
        )

        tokens = hidden_states.shape[0]
        cos, sin = position_embeddings

        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        kv_is_fp8 = self.k_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]

        # Segmented prefill: fuse the KV cache write into the QKV kernel
        # (use_block_kv=True) to eliminate the HBM round-trip for K/V
        # materialization + explicit scatter. The MX kernel returns Q in fp8;
        # cast it to bf16 for the downstream attention. K/V dequant is folded
        # into the softmax scale and output (same pattern as gpt_oss/model_mxfp4).
        #
        # Full prefill (no kv_segment_size): flash_attention needs raw K/V
        # tensors, so the fused path is not applicable — use explicit scatter.
        if kv_segment_size:
            validate_fp8_segmented_supported(
                kv_is_fp8=kv_is_fp8,
                fp8_packed=self.fp8_packed,
            )
            # Sanitize slot_mapping: the qkv NKI kernel uses slot_mapping as
            # direct DMA offsets — out-of-range slots cause a hardware OOB.
            # Remap sentinels (< 0) and stale values to slot 0 (harmless write
            # to the scratch block).
            num_blocks_total = self.k_cache.shape[0]
            max_slot = num_blocks_total * block_size
            slot_mapping_clamped = torch.where(
                (slot_mapping < 0) | (slot_mapping >= max_slot),
                torch.zeros_like(slot_mapping),
                slot_mapping,
            ).to(torch.int32)

            # qkv_cte divides K/V by the scale arg, but self.k_scale is stored
            # inverted. Pass 1/self.k_scale so the write matches the old scatter.
            k_scale_write = (1.0 / self.k_scale) if kv_is_fp8 else None
            v_scale_write = (1.0 / self.v_scale) if kv_is_fp8 else None

            q, _, _ = NF.qkv_proj(
                hidden=hidden_states.unsqueeze(0),
                qkv_weights=self.qkv_proj_weight,
                bias=None,
                quantization_type=QuantizationType.MX,
                qkv_w_scale=self.qkv_weight_scale,
                qkv_in_scale=None,
                weight_layout=QKVWeightLayout.MX_CONTIGUOUS,
                cos_cache=cos.unsqueeze(0),
                sin_cache=sin.unsqueeze(0),
                num_q_heads=self.num_attention_heads_per_rank,
                num_kv_heads=self.num_key_value_heads_per_rank,
                d_head=self.head_dim,
                qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
                qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
                qk_norm_pre_rope_eps=self.q_norm.variance_epsilon,
                qk_norm_pre_rope_q_gamma=self.q_norm.weight.view(1, self.head_dim),
                qk_norm_pre_rope_k_gamma=self.k_norm.weight.view(1, self.head_dim),
                # In-kernel KV cache write
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                k_scale=k_scale_write,
                v_scale=v_scale_write,
                fp8_max=FP8_CLAMP_MAX if kv_is_fp8 else None,
                fp8_min=-FP8_CLAMP_MAX if kv_is_fp8 else None,
                kv_dtype=self.k_cache.dtype if kv_is_fp8 else None,
                use_block_kv=True,
                fp8_packed=self.fp8_packed,
                block_size=block_size,
                slot_mapping=slot_mapping_clamped.reshape(1, tokens),
            )
            # MX kernel returns Q in fp8; cast to bf16 for attention.
            q = (
                q.squeeze(0)
                .to(self.dtype)
                .view(tokens, self.num_attention_heads_per_rank, self.head_dim)
                .transpose(0, 1)
            )
            # FP8 KV dequant fold: the in-kernel write stored K/k_scale and
            # V/v_scale (same bytes as the old explicit scatter). Fold K-dequant
            # into softmax scale and V-dequant out of the output. Both are
            # exact scalar no-ops at k_scale=v_scale=1.0.
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
            )
            if self.v_scale_float != 1.0:
                attn_output = attn_output / self.v_scale_float
        else:
            # Full prefill: flash_attention needs raw K/V tensors, so project
            # without the in-kernel write and scatter explicitly.
            qkv = NF.qkv_proj(
                hidden=hidden_states.unsqueeze(0),
                qkv_weights=self.qkv_proj_weight,
                bias=None,
                quantization_type=QuantizationType.MX,
                qkv_w_scale=self.qkv_weight_scale,
                qkv_in_scale=None,
                weight_layout=QKVWeightLayout.MX_CONTIGUOUS,
                cos_cache=cos.unsqueeze(0),
                sin_cache=sin.unsqueeze(0),
                num_q_heads=self.num_attention_heads_per_rank,
                num_kv_heads=self.num_key_value_heads_per_rank,
                d_head=self.head_dim,
                qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
                qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
                qk_norm_pre_rope_eps=self.q_norm.variance_epsilon,
                qk_norm_pre_rope_q_gamma=self.q_norm.weight.view(1, self.head_dim),
                qk_norm_pre_rope_k_gamma=self.k_norm.weight.view(1, self.head_dim),
            ).squeeze(0)

            q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
            q = q.view(
                tokens, self.num_attention_heads_per_rank, self.head_dim
            ).transpose(0, 1)
            k = k.view(
                tokens, self.num_key_value_heads_per_rank, self.head_dim
            ).transpose(0, 1)
            v = v.view(
                tokens, self.num_key_value_heads_per_rank, self.head_dim
            ).transpose(0, 1)

            # Write K/V into the paged cache for future segments.
            block_indices, position_indices = paged_slot_indices(
                slot_mapping, block_size, self.k_cache.shape[0]
            )
            if kv_is_fp8:
                k_flat = (
                    (k.reshape(-1, self.head_dim) * self.k_scale)
                    .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                    .to(self.k_cache.dtype)
                )
                v_flat = (
                    (v.reshape(-1, self.head_dim) * self.v_scale)
                    .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                    .to(self.v_cache.dtype)
                )
            else:
                k_flat = k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
                v_flat = v.reshape(-1, self.head_dim).to(self.v_cache.dtype)

            head_indices_for_put = torch.arange(
                self.num_key_value_heads_per_rank,
                dtype=torch.long,
                device=hidden_states.device,
            ).repeat_interleave(slot_mapping.shape[0])
            block_indices_for_put = block_indices.repeat(
                self.num_key_value_heads_per_rank
            )
            position_indices_for_put = position_indices.repeat(
                self.num_key_value_heads_per_rank
            )
            index = (
                block_indices_for_put,
                head_indices_for_put,
                position_indices_for_put,
            )
            self.v_cache.index_put_(index, v_flat)
            if self.fp8_packed:
                scatter_packed_k(
                    self.k_cache,
                    k_flat,
                    block_indices_for_put,
                    head_indices_for_put,
                    position_indices_for_put,
                )
            else:
                self.k_cache.index_put_(index, k_flat)

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
            self.o_proj_weight_prefill,
            None,
            quantization_type=QuantizationType.MX,
            input_scales=None,
            weight_scales=self.o_proj_weight_scale,
        ).squeeze(0)

        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
        gamma_shuffled: torch.Tensor | None = None,
        eps: float | None = None,
        decode_sp_active: bool = False,
        h_preshuffled: bool = False,
        sp_a2a_partials: torch.Tensor | None = None,
    ):
        """Decode via the fused megakernel. ``gamma_shuffled``/``eps``, when
        given, also fuse the input RMSNorm into the megakernel (``gamma_shuffled``
        carries the H-4 shuffled gamma that the ``is_h_transposed_by_4`` kernel
        reads). ``decode_sp_active`` returns the o_proj output UN-REDUCED: the next
        a2a composer reduces it in-kernel (``hidden_states`` must already hold
        every token: ``forward`` gathers it, except when the a2a attention
        prologue builds it); ``h_preshuffled`` says the input already holds the H-4 tile order
        and the kernel dtype, so do not shuffle or cast again."""
        # gamma_shuffled and eps must be set together to fuse the input RMSNorm.
        assert (gamma_shuffled is None) == (eps is None), (
            "gamma_shuffled and eps must be passed together. Set both to fuse "
            "the input RMSNorm, or leave both unset."
        )
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]

        batch_size = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        sp_a2a = sp_a2a_partials is not None
        if sp_a2a:
            # hidden_states is the residual shard, the partials hold every token; one token per sequence here.
            assert sp_a2a_partials.shape[0] == batch_size, (
                f"the a2a prologue expects [T, H] partials; got "
                f"{sp_a2a_partials.shape[0]} tokens for batch {batch_size}."
            )
            assert tokens * self.world_size == batch_size, (
                f"the a2a prologue expects a [T / world_size, H] residual "
                f"shard; got {tokens} tokens for batch {batch_size} at world "
                f"size {self.world_size}."
            )
            s_decode = 1
        else:
            s_decode = tokens // batch_size
            assert tokens == batch_size * s_decode

        assert hidden % 512 == 0, (
            f"Qwen3AttentionMxfp8 decode requires hidden_size % 512 == 0, got {hidden}."
        )

        # Decode kernels want H preshuffled to the MX QKV tile order; the shard path did it.
        if sp_a2a:
            pass
        elif not h_preshuffled:
            hidden_states = pre_shuffle_h(
                hidden_states.to(self.dtype).view(batch_size, s_decode, hidden)
            )
        else:
            hidden_states = hidden_states.view(batch_size, s_decode, hidden)
        cos_kernel, sin_kernel = self._decode_cos_sin(
            position_embeddings, batch_size, s_decode
        )
        pos_ids_kernel = positions.view(batch_size, s_decode).to(torch.float32)
        # The decode path addresses cache blocks per KV head, not per sequence,
        # so expand the block table into the megakernel's per-head contract.
        active_blocks_table = build_per_head_block_table(
            block_table, self.num_key_value_heads_per_rank
        )

        # Packed FP8 is decided per decode bucket. The cache is *stored* packed
        # (self.fp8_packed), but the packed decode kernel requires the resized
        # block_len to stay >= 2, which some bucket geometries (batch=1) violate.
        # For a non-viable bucket, un-swizzle the packed cache to the standard
        # layout and run the unpacked kernel. The viability flag is a static
        # (trace-time) bool, so each compiled decode NEFF takes one branch.
        s_ctx = max_blocks_per_seq * block_size
        use_packed_kernel = self.fp8_packed and packed_fp8_viable_for_bucket(
            block_len=block_size,
            bs=batch_size,
            q_head=self.num_attention_heads_per_rank,
            s_active=s_decode,
            s_prior=s_ctx,
        )
        # BF16 cache writes stay outside the megakernel. Decode reads through the
        # current query positions, so updating those same slots in the attention
        # launch creates an unsynchronized cross-core read/write overlap.
        #
        # Keep the existing packed-FP8 optimization only when each rank owns one
        # KV head. kv_cache_update_idx has no KV-head dimension, so the packed
        # multi-head path still falls back to its existing unsupported guard.
        nkh = self.num_key_value_heads_per_rank
        use_inkernel_write = _use_inkernel_decode_kv_write(self.fp8_packed, nkh)
        if use_inkernel_write:
            kv_cache_update_idx = slot_mapping.view(batch_size, s_decode).to(
                torch.uint32
            )
        else:
            kv_cache_update_idx = None

        # Read target -- and, for the in-kernel write, the write target too.
        # Viable-packed or non-packed reads/writes self.k_cache directly; a
        # non-viable packed bucket un-swizzles to a standard-layout buffer for
        # the unpacked kernel, which the in-kernel write re-swizzles back once
        # (below). For the external scatter path this buffer is read-only -- the
        # scatter re-derives its write target from self.k_cache.
        k_cache_arg = (
            self.k_cache
            if (use_packed_kernel or not self.fp8_packed)
            else _unswizzle_packed_k(self.k_cache)
        )

        megakernel_out = self._run_decode_megakernel(
            hidden_states,
            cos_kernel,
            sin_kernel,
            active_blocks_table,
            pos_ids_kernel,
            k_cache_arg,
            use_packed_kernel,
            kv_cache_update_idx,
            gamma_shuffled=gamma_shuffled,
            eps=eps,
            return_updated_kv=(self.fp8_packed and not use_packed_kernel),
            sp_a2a_partials=sp_a2a_partials,
        )

        sp_residual_out = None
        if sp_a2a:
            # The a2a composer appends the updated residual stream.
            megakernel_out, sp_residual_out = megakernel_out

        if use_inkernel_write:
            # Kernel wrote K/V in place and returned the attention output alone.
            output = megakernel_out
            if self.fp8_packed and not use_packed_kernel:
                # Non-viable packed bucket: the unpacked kernel wrote the
                # un-swizzled temp buffer; re-swizzle it back into the packed
                # cache once (the viable path already wrote swizzled in place).
                # Use the kernel's RETURNED K: k_cache_arg is the PRE-kernel temp.
                output, _k_updated, _v_updated = megakernel_out
                self.k_cache.copy_(_swizzle_packed_k(_k_updated))
        else:
            # BF16 uses the external scatter for every TP degree. Packed FP8 only
            # reaches this branch at >1 KV head/rank, which remains unsupported.
            output, k_new, v_new = megakernel_out
            if self.fp8_packed:
                raise NotImplementedError(
                    "packed FP8 KV decode at >1 KV head/rank is not supported: "
                    "the in-kernel write has no KV-head dim and the external-"
                    "scatter fallback round-trips the whole packed cache. Use "
                    "TP >= KV-head count for decode, or use this config for "
                    "prefill-only."
                )
            k_new = _normalize_decode_k_for_external_scatter(k_new, nkh)
            scatter_new_kv(
                self.k_cache,
                self.v_cache,
                k_new,
                v_new,
                slot_mapping,
                block_size,
                self.num_key_value_heads_per_rank,
                self.head_dim,
            )

        if decode_sp_active:
            # The a2a MLP block reduces these partials on the PE in-kernel, so return them un-reduced.
            pass
        elif self.world_size > 1:
            self.tp_group.all_reduce(output)

        if sp_a2a:
            return output, sp_residual_out
        return output


class Qwen3MLPMxfp8(Qwen3MLPBf16):
    def __init__(self, config: Qwen3Config):
        self._base_config = config
        super().__init__(config)

        # Decide once whether the decode MLP uses the in-tree kernel: the config
        # must pass the eligibility rules (see _intree_mlp_reject_reason), and it
        # must honor the global NKI kernel switch
        # (VLLM_NEURON_DISABLE_NKI_KERNELS / VLLM_NEURON_CPU_MODE via
        # can_run_kernel). Every other path uses the shared nkilib kernel. This
        # one choice drives the decode scale layout, the weight loaders, and the
        # forward, so they stay consistent.
        reject_reason = _intree_mlp_reject_reason(
            self.hidden_size, self.intermediate_size_per_rank, self.world_size
        )
        self._use_intree_mlp = (
            _QWEN3_INTREE_MLP and reject_reason is None and can_run_kernel()
        )
        # The folded (pack64) decode scale needs a start offset on the MX scale
        # partition axis, and that addressing mode is trn3/gen4 only -- the compiler
        # rejects it on trn1/trn2, which CI can schedule for TP16/TP32. Same values
        # either way, so this changes bytes moved and not numbers.
        self._use_folded_scales = self._use_intree_mlp and _target_supports_scale_pidx()
        # Report the SELECTED kernel either way, at info. Which MLP kernel runs
        # drives the decode scale layout, the weight loaders and the forward, and
        # it was previously only reported on the fallback path at debug -- so a
        # silent fallback was indistinguishable from the kernel running. Report it
        # one time for each distinct decision, because every decoder layer builds
        # the same config.
        decision = (
            self.world_size,
            self.intermediate_size_per_rank,
            self._use_intree_mlp,
        )
        if decision not in _intree_mlp_logged:
            _intree_mlp_logged.add(decision)
            logger = logging.getLogger(__name__)
            if self._use_intree_mlp:
                logger.info(
                    "Qwen3-32B MXFP8 decode MLP: the in-tree streaming kernel runs "
                    "(TP%d, I/rank=%d, %d TKG 512-tiles, PSUM holds %d).",
                    self.world_size,
                    self.intermediate_size_per_rank,
                    _n_i512_tiles_tkg(self.intermediate_size_per_rank),
                    _PSUM_BMAX_TILES,
                )
            else:
                # Name the actual cause. The config can be eligible and the kernel
                # still skipped because NKI kernels are off globally
                # (VLLM_NEURON_DISABLE_NKI_KERNELS / CPU mode). A shape reason in
                # that case sends the reader after the wrong thing.
                cause = reject_reason
                if cause is None:
                    cause = "can_run_kernel() is False (NKI kernels are disabled)"
                logger.info(
                    "Qwen3-32B MXFP8 decode MLP: the nkilib fallback runs at TP%d, "
                    "because %s.",
                    self.world_size,
                    cause,
                )

        hidden_size = self.hidden_size
        intermediate_size_per_rank = self.intermediate_size_per_rank
        n_h512 = hidden_size // _TILE_SIZE
        n_i512 = _n_i512_tiles(intermediate_size_per_rank)
        n_i512_tkg = _n_i512_tiles_tkg(intermediate_size_per_rank)
        n_h_scale = _SCALES_PER_TILE

        # Size the decode scale parameters to the layout that runs: 64 rows for the
        # folded in-tree layout, 128 for the pre-scattered in-tree layout, and the
        # 16-row layout for nkilib.
        if self._use_folded_scales:
            decode_scale_rows = _SCALE_FOLD_ROWS
        elif self._use_intree_mlp:
            decode_scale_rows = _PMAX
        else:
            decode_scale_rows = n_h_scale

        # A decode-only pool traces no prefill graph, so its prefill-layout
        # weights are unreachable HBM. Skip them and give the KV cache the bytes.
        self.decode_only_weights = _decode_only_weights(config)

        del self.gate_proj_weight
        del self.up_proj_weight
        del self.down_proj_weight

        # A long prefill uses the larger prefill layout, while decode and a short
        # prefill use the decode layout. A pool that serves both stores both, to
        # keep runtime routing cheap. A decode-only pool stores the decode set
        # only, so the block below does not run.
        if not self.decode_only_weights:
            self.gate_proj_weight_prefill = nn.Parameter(
                torch.empty(
                    128, n_h512, n_i512, _Q_WIDTH, 128, _Q_WIDTH, dtype=_FP8_DTYPE
                ),
                requires_grad=False,
            )
            self.up_proj_weight_prefill = nn.Parameter(
                torch.empty(
                    128, n_h512, n_i512, _Q_WIDTH, 128, _Q_WIDTH, dtype=_FP8_DTYPE
                ),
                requires_grad=False,
            )
            self.down_proj_weight_prefill = nn.Parameter(
                torch.empty(128, n_i512, hidden_size, _Q_WIDTH, dtype=_FP8_DTYPE),
                requires_grad=False,
            )
            self.gate_proj_weight_scale_prefill = nn.Parameter(
                torch.empty(
                    n_h_scale, n_h512, n_i512, _Q_WIDTH, 128, dtype=torch.uint8
                ),
                requires_grad=False,
            )
            self.up_proj_weight_scale_prefill = nn.Parameter(
                torch.empty(
                    n_h_scale, n_h512, n_i512, _Q_WIDTH, 128, dtype=torch.uint8
                ),
                requires_grad=False,
            )
            self.down_proj_weight_scale_prefill = nn.Parameter(
                torch.empty(n_h_scale, n_i512, hidden_size, dtype=torch.uint8),
                requires_grad=False,
            )

        # The in-tree kernel consumes gate/up I one 128-wide lane at a time, so it
        # needs only a multiple of 128. nkilib needs a whole 512-tile. The down
        # weight keeps whole 512-tiles for both, because its I axis is the
        # x4-packed contraction. Qwen3-32B at TP8 saves 384 I columns of gate and
        # of up, which is 10.7% of the bytes of each.
        i_pad = gate_up_i_pad_tkg(intermediate_size_per_rank, self._use_intree_mlp)
        self.gate_proj_weight_decode = nn.Parameter(
            torch.empty(128, n_h512, i_pad, dtype=torch.uint32), requires_grad=False
        )
        self.up_proj_weight_decode = nn.Parameter(
            torch.empty(128, n_h512, i_pad, dtype=torch.uint32), requires_grad=False
        )
        self.down_proj_weight_decode = nn.Parameter(
            torch.empty(128, n_i512_tkg, hidden_size, dtype=torch.uint32),
            requires_grad=False,
        )
        # Decode scales. The folded layout moves a factor of 4 off the free axis:
        # gate/up hold n_i512*128 columns and down ceil(n_i512/4) buffers. The
        # pre-scattered layout keeps the full free axis on 128 partitions, 112 of
        # them zero. nkilib uses 16 rows, and prefill keeps 16 either way.
        gu_scale_cols = i_pad
        down_scale_tiles = n_i512_tkg
        if self._use_folded_scales:
            gu_scale_cols = n_i512_tkg * _PMAX
            down_scale_tiles = math.ceil(n_i512_tkg / _SCALE_FOLD_SLOTS)
        self.gate_proj_weight_scale_decode = nn.Parameter(
            torch.empty(decode_scale_rows, n_h512, gu_scale_cols, dtype=torch.uint8),
            requires_grad=False,
        )
        self.up_proj_weight_scale_decode = nn.Parameter(
            torch.empty(decode_scale_rows, n_h512, gu_scale_cols, dtype=torch.uint8),
            requires_grad=False,
        )
        self.down_proj_weight_scale_decode = nn.Parameter(
            torch.empty(
                decode_scale_rows, down_scale_tiles, hidden_size, dtype=torch.uint8
            ),
            requires_grad=False,
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self, config: Qwen3Config | None = None) -> None:
        # The sentinel must be a DECODE parameter. A decode-only pool creates no
        # prefill parameter, and a prefill sentinel would then send this call to
        # the bf16 super(), which reads the gate_proj_weight this class deleted.
        if not hasattr(self, "gate_proj_weight_scale_decode"):
            return super()._setup_weight_loaders(config or self._base_config)

        hidden_size = self.hidden_size
        intermediate_size_per_rank = self.intermediate_size_per_rank
        world_size = self.world_size
        # Match the decode scale scatter to the path that runs (see the param
        # shapes above): scatter for the in-tree kernel, 16-row for nkilib.
        use_intree = self._use_intree_mlp
        use_folded = self._use_folded_scales
        # Read the same padded gate/up I extent the parameters use, so the layout
        # the loader writes cannot disagree with the buffer it writes into.
        i_pad_decode = self.gate_proj_weight_decode.shape[2]

        if not self.decode_only_weights:
            set_weight_loader(
                self.gate_proj_weight_prefill,
                mlp_gate_up_weight_loader(
                    intermediate_size_per_rank, hidden_size, world_size
                ),
            )
            set_weight_loader(
                self.up_proj_weight_prefill,
                mlp_gate_up_weight_loader(
                    intermediate_size_per_rank, hidden_size, world_size
                ),
            )
            set_weight_loader(
                self.down_proj_weight_prefill,
                mlp_down_weight_loader(
                    intermediate_size_per_rank, hidden_size, world_size
                ),
            )
            set_weight_loader(
                self.gate_proj_weight_scale_prefill,
                mlp_gate_up_scale_loader(
                    intermediate_size_per_rank, hidden_size, world_size
                ),
            )
            set_weight_loader(
                self.up_proj_weight_scale_prefill,
                mlp_gate_up_scale_loader(
                    intermediate_size_per_rank, hidden_size, world_size
                ),
            )
            set_weight_loader(
                self.down_proj_weight_scale_prefill,
                mlp_down_scale_loader(
                    intermediate_size_per_rank, hidden_size, world_size
                ),
            )

        set_weight_loader(
            self.gate_proj_weight_decode,
            mlp_gate_up_weight_loader_tkg(
                intermediate_size_per_rank, hidden_size, world_size, i_pad=i_pad_decode
            ),
        )
        set_weight_loader(
            self.up_proj_weight_decode,
            mlp_gate_up_weight_loader_tkg(
                intermediate_size_per_rank, hidden_size, world_size, i_pad=i_pad_decode
            ),
        )
        # The TKG decode down-proj weight and scale are two separate parameters
        # fed the same [weight, weight_scale] checkpoint pair; share one cache so
        # the expensive dequant->re-quant runs once per shard, not twice.
        down_tkg_requant_cache: dict = {}
        set_weight_loader(
            self.down_proj_weight_decode,
            mlp_down_weight_loader_tkg(
                intermediate_size_per_rank,
                hidden_size,
                world_size,
                cache=down_tkg_requant_cache,
            ),
        )
        set_weight_loader(
            self.gate_proj_weight_scale_decode,
            mlp_gate_up_scale_loader_tkg(
                intermediate_size_per_rank,
                hidden_size,
                world_size,
                scatter=use_intree,
                fold=use_folded,
                i_pad=i_pad_decode,
            ),
        )
        set_weight_loader(
            self.up_proj_weight_scale_decode,
            mlp_gate_up_scale_loader_tkg(
                intermediate_size_per_rank,
                hidden_size,
                world_size,
                scatter=use_intree,
                fold=use_folded,
                i_pad=i_pad_decode,
            ),
        )
        set_weight_loader(
            self.down_proj_weight_scale_decode,
            mlp_down_scale_loader_tkg(
                intermediate_size_per_rank,
                hidden_size,
                world_size,
                cache=down_tkg_requant_cache,
                scatter=use_intree,
                fold=use_folded,
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        is_prefill: bool,
        sp_shard: bool | None = None,
        ln_w: torch.Tensor | None = None,
        eps: float | None = None,
    ) -> torch.Tensor:
        """Run the MLP. ``is_prefill`` selects the weight layout; ``sp_shard``
        the activation layout (token shard in, token shard out), separate
        because a short prompt keeps prefill's sequence-parallel activations on
        the decode weights; ``sp_shard=None`` defaults to ``is_prefill``.
        ``ln_w`` fuses the RMSNorm into the decode kernel and is unavailable on
        the sequence-parallel path, which normalizes in its own fused kernel.
        """
        if sp_shard is None:
            sp_shard = is_prefill
        # The kernel wrapper checks the ln_w/eps pairing and the gamma shape,
        # because those are its input contract and every caller must meet them.
        # The two checks below are routing decisions that only this model makes.
        #
        # Only the in-tree decode kernel has a norm stage. The CTE prefill kernel
        # and the nkilib fallback drop ln_w without a message, so refuse it there.
        # The token count goes in too, because the fusion is also unavailable at
        # the counts the compiler cannot build (see _ILSA901_TOKEN_COUNTS).
        # Callers use can_fuse_decode_norm() to select the path.
        assert ln_w is None or self.can_fuse_decode_norm(hidden_states.shape[0]), (
            "Qwen3MLPMxfp8 can fuse the RMSNorm only on the in-tree decode "
            f"kernel, and not at T={hidden_states.shape[0]}. Normalize the "
            "activation before the call, or check can_fuse_decode_norm(T) first."
        )
        assert not (is_prefill and ln_w is not None), (
            "Qwen3MLPMxfp8 prefill cannot fuse the RMSNorm: the CTE kernel has "
            "no norm stage. Normalize the activation before the call."
        )
        assert not (sp_shard and ln_w is not None), (
            "Qwen3MLPMxfp8 cannot fuse the RMSNorm on a token shard: the "
            "sequence-parallel path normalizes in its own kernel."
        )

        if is_prefill:
            if self.decode_only_weights:
                raise RuntimeError(
                    "Qwen3 MXFP8 got a prefill request on a decode-only pool, which "
                    "holds no prefill weights. Run prefill on a kv_producer or "
                    "kv_both server."
                )
            gate_w = self.gate_proj_weight_prefill
            up_w = self.up_proj_weight_prefill
            down_w = self.down_proj_weight_prefill
            gate_w_scale = self.gate_proj_weight_scale_prefill
            up_w_scale = self.up_proj_weight_scale_prefill
            down_w_scale = self.down_proj_weight_scale_prefill
            gate_up_w_layout = MLPGateUpWeightLayout.H_X4_INNERMOST
            assert hidden_states.shape[-1] == self.hidden_size + _Q_WIDTH, (
                f"Qwen3MLPMxfp8 prefill expects ROW-packed [T, H+{_Q_WIDTH}] input; "
                f"got last dim {hidden_states.shape[-1]} (H={self.hidden_size})."
            )
        else:
            gate_w = self.gate_proj_weight_decode
            up_w = self.up_proj_weight_decode
            down_w = self.down_proj_weight_decode
            gate_w_scale = self.gate_proj_weight_scale_decode
            up_w_scale = self.up_proj_weight_scale_decode
            down_w_scale = self.down_proj_weight_scale_decode
            gate_up_w_layout = MLPGateUpWeightLayout.CONTIGUOUS
            assert hidden_states.shape[-1] == self.hidden_size, (
                f"Qwen3MLPMxfp8 decode expects bf16 [T, H] input; got last dim "
                f"{hidden_states.shape[-1]} (H={self.hidden_size})."
            )
            # The nkilib TKG decode kernel expects the hidden dimension in the
            # same tiled order as the attention decode megakernel. The in-tree
            # kernel builds that order with one permute of the natural [T, H]
            # tensor, so it does not need this shuffle. The shuffle ends with
            # .contiguous(), so it costs one [T, H] HBM write and read for each
            # layer. See qwen3_intree_mlp for the layout it derives instead.
            #
            # Dropping the shuffle does NOT fix the NCC_ILSA901 failure of the
            # fused path. That failure needs the residual add, and it survives
            # every spelling of this layout. See _ILSA901_TOKEN_COUNTS.
            if not self._use_intree_mlp:
                hidden_states = pre_shuffle_h(hidden_states)
            # ln_w arrives in the kernel layout from the weight loader, so no
            # shuffle is needed here. The kernel pairs gamma with the activation on
            # the flat H index, so both must use the same H-4 order. The wrapper
            # checks the gamma shape against the activation.

        if sp_shard and self.world_size > 1:
            # Both kernels contract the full hidden dimension, so each rank needs every token; this closes the SP region.
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        # Read the count after the gather: the nkilib threshold is global.
        n_tokens = hidden_states.shape[0]
        use_intree = self._use_intree_mlp and not is_prefill
        if is_prefill:
            assert n_tokens > _MLP_TKG_BS_SEQLEN_THRESHOLD, (
                f"Qwen3MLPMxfp8 prefill selected CTE weights but token count "
                f"{n_tokens} is <= the nkilib TKG threshold "
                f"{_MLP_TKG_BS_SEQLEN_THRESHOLD}."
            )
        elif use_intree:
            # The in-tree kernel does not consult the nkilib TKG threshold, so a
            # decode batch above it is legal here. Its own bound is the down
            # projection's stationary free size, which nc_matmul_mx caps at 128.
            assert n_tokens <= _INTREE_MAX_DECODE_TOKENS, (
                f"Qwen3MLPMxfp8 decode token count {n_tokens} exceeds "
                f"{_INTREE_MAX_DECODE_TOKENS}, the largest the in-tree kernel "
                f"takes in one entry (the down projection caps its stationary "
                f"free size at 128). Lower max_num_seqs, or tile the token axis."
            )
        else:
            # nkilib fallback. NF.mlp derives AUTO, and nkilib's is_mlp_tkg then
            # splits TKG from CTE by B*S: a token count above the threshold would
            # reach CTE and read the TKG weight layout this branch selected, which
            # is wrong and silent. Keep the threshold on this path.
            assert n_tokens <= _MLP_TKG_BS_SEQLEN_THRESHOLD, (
                f"Qwen3MLPMxfp8 decode selected TKG weights but token count "
                f"{n_tokens} exceeds the nkilib TKG threshold "
                f"{_MLP_TKG_BS_SEQLEN_THRESHOLD}. The in-tree kernel takes a "
                f"wider batch; this is the nkilib fallback path."
            )

        if use_intree:
            # Model-specific swap: run the decode MLP with the in-tree kernel. It
            # consumes the natural [T, H] activation and returns the per-rank
            # [T, H] output; the all-reduce below reduces it across ranks. The
            # in-tree path cannot run every config, so an ineligible one uses
            # nkilib below (see _intree_mlp_reject_reason). ln_w fuses the RMSNorm
            # into the kernel, and hidden_states is then the raw pre-norm
            # activation. Without ln_w it is already normalized.
            output = _qwen3_intree_decode_fn(
                hidden_states,
                gate_w,
                up_w,
                down_w,
                gate_w_scale,
                up_w_scale,
                down_w_scale,
                ln_w=ln_w,
                eps=eps,
            )
        else:
            output = NF.mlp(
                hidden_states,
                gate_w,
                up_w,
                down_w,
                quantization_type=QuantizationType.MX,
                gate_w_scale=gate_w_scale,
                up_w_scale=up_w_scale,
                down_w_scale=down_w_scale,
                gate_up_in_scale=None,
                down_in_scale=None,
                gate_up_w_layout=gate_up_w_layout,
                output_dtype="bfloat16",
            )

        if self.world_size > 1:
            if sp_shard:
                # Branch on the activation layout, not is_prefill: an all_reduce on a shard sums different tokens, silently.
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                self.tp_group.all_reduce(output)

        return output.contiguous()

    def can_fuse_decode_norm(self, n_tokens: int | None = None) -> bool:
        """Report whether the decode MLP kernel applies the RMSNorm itself.

        Only the in-tree kernel has a norm stage. The nkilib fallback, which runs
        at every tensor-parallel degree except 8, needs a normalized activation.

        The token count declines the fusion at the counts in
        ``_ILSA901_TOKEN_COUNTS``. Omit it to ask only whether this kernel has a
        norm stage at all.

        Args:
            n_tokens: The token count of the activation. Omit it for a
                capability check that ignores the shape.

        Returns:
            True if the kernel applies the RMSNorm for this token count.
        """
        if not self._use_intree_mlp:
            return False
        return n_tokens is None or n_tokens not in _ILSA901_TOKEN_COUNTS

    def use_prefill_cte(self, local_tokens: int) -> bool:
        # The nkilib threshold is defined on the gathered token count, so scale
        # the local rank's view back to the global prompt width before routing.
        return local_tokens * self.world_size > _MLP_TKG_BS_SEQLEN_THRESHOLD

    def use_sp_a2a(self, tokens: int) -> bool:
        # LNC1 all-to-all form of the SP MLP block; ``tokens`` is the global count, and the kernel gate owns the shape rules.
        if not (
            self._use_intree_mlp
            and _qwen3_sp_a2a_mlp_decode_fn is not None
            and _qwen3_sp_a2a_supports is not None
            and self.world_size > 1
            and tokens % self.world_size == 0
        ):
            return False
        return _qwen3_sp_a2a_supports(
            self.hidden_size,
            tokens // self.world_size,
            self.world_size,
            get_logical_nc_config(),
        )

    def forward_sp_a2a(
        self,
        attn_partials: torch.Tensor,
        residual: torch.Tensor,
        gamma_shuffled: torch.Tensor,
        eps: float,
        residual_in_shuffled: bool = False,
        residual_out_shuffled: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ``attn_partials``: bf16 [T, H] un-reduced; ``residual``: bf16 [T_s, H] carrier. Returns the UN-REDUCED [T, H] MLP partials for the next layer's a2a prologue plus the updated [T_s, H] residual; the model loop closes the last carry with one reduce-scatter.
        output, residual_out = _qwen3_sp_a2a_mlp_decode_fn(
            attn_partials,
            residual,
            self.gate_proj_weight_decode,
            self.up_proj_weight_decode,
            self.down_proj_weight_decode,
            self.gate_proj_weight_scale_decode,
            self.up_proj_weight_scale_decode,
            self.down_proj_weight_scale_decode,
            gamma_shuffled=gamma_shuffled,
            eps=eps,
            tp_group=self.tp_group,
            residual_in_shuffled=residual_in_shuffled,
            residual_out_shuffled=residual_out_shuffled,
        )
        return output, residual_out


class Qwen3DecoderLayerMxfp8(Qwen3DecoderLayerBf16):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        nn.Module.__init__(self)
        # The last layer returns the residual in NATURAL order for the carry close; earlier layers pass the FLAT pre-shuffled carrier (see qwen3_32b_sp_prologue.py).
        self.is_last_layer = layer_idx == config.num_hidden_layers - 1
        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        # The decode kernels fuse both RMSNorms and read gamma in the
        # is_h_transposed_by_4 layout. Keep a pre-shuffled copy of each gamma, so
        # the reshape and the permute stay off the decode critical path. Prefill
        # keeps the natural gamma above for its own RMSNorm. Initialize to ones so
        # an unfilled buffer acts as an identity multiply.
        self.input_layernorm_gamma_shuffled = self._make_gamma_shuffled(config)
        # The MLP gamma goes to the in-tree decode kernel, which reads
        # [128, H // 512, 4]. Store that layout, so the traced graph holds no gamma
        # transpose. See _make_gamma_kernel_layout.
        self.post_attention_layernorm_gamma_shuffled = self._make_gamma_kernel_layout(
            config
        )
        # The SP norm+MLP kernel reads a [1, H] gamma in H-4 order, not the [128, H // 512, 4] kernel layout above.
        # Skip this parameter for SP-off pools. The forward reads it only when SP is active.
        if _decode_sp_enabled(config):
            self.post_attention_layernorm_gamma_sp = self._make_gamma_shuffled(config)
        self.self_attn = Qwen3AttentionMxfp8(config, layer_idx=layer_idx)
        self.mlp = Qwen3MLPMxfp8(config)
        self.layer_idx = layer_idx
        self.rms_norm_eps = config.rms_norm_eps
        self.dtype = config.torch_dtype
        # This __init__ bypasses the BF16 base, so set every attribute the forward reads; both gate the SP decode layout.
        self.decode_sp_enabled = _decode_sp_enabled(config)
        self.world_size = self.self_attn.world_size

    def _make_gamma_shuffled(self, config: Qwen3Config) -> nn.Parameter:
        """Build a parameter that holds an RMSNorm gamma in the H-4 layout.

        Same checkpoint source as the natural gamma, but transformed into the
        layout the decode kernels' fused RMSNorm requires. The gamma is replicated
        across TP, so the loader does not shard. The weight-name mapping in
        ``load_weights`` selects the checkpoint tensor for each parameter.

        Args:
            config: The model config. It gives the hidden size and the dtype.

        Returns:
            The parameter, with its pre-shuffling weight loader attached.
        """

        def _h4_shuffle(slices: list, rank: int) -> torch.Tensor:
            assert len(slices) == 1
            gamma = slices[0][:]
            return pre_shuffle_h(gamma.view(1, 1, -1)).view(1, -1)

        param = nn.Parameter(
            torch.ones(1, config.hidden_size, dtype=config.torch_dtype),
            requires_grad=False,
        )
        set_weight_loader(param, SafetensorsWeightLoader(transform=_h4_shuffle))
        return param

    def _make_gamma_kernel_layout(self, config: Qwen3Config) -> nn.Parameter:
        """Build an RMSNorm gamma parameter in the decode MLP kernel's layout.

        The in-tree decode MLP kernel reads gamma as ``[128, H // 512, 4]``, with
        ``gamma[p, t512, q] = gamma_natural[t512 * 512 + p * 4 + q]``. This loader
        writes that layout, so the traced graph holds no gamma transpose.

        A ``[1, H]`` gamma plus a reshape and a permute in the graph gives the same
        values, and it made neuronx-cc fail. At T=2 the gamma transpose lowers to a
        pf-transpose with a 1024-byte tile, and LegalizeSundaAccess rejects its
        matmul destination access pattern (NCC_ILSA901). gamma is a parameter, so
        the layout costs nothing at load time and this removes the operator.

        The sibling ``_make_gamma_shuffled`` stays for the attention gamma. The
        attention decode megakernel reads that one as ``[1, H]``.

        Args:
            config: The model config. It gives the hidden size and the dtype.

        Returns:
            The parameter, with its layout weight loader attached.
        """
        n_h512 = config.hidden_size // _TILE_SIZE

        def _kernel_layout(slices: list, rank: int) -> torch.Tensor:
            assert len(slices) == 1
            gamma = slices[0][:].reshape(n_h512, _PMAX, _Q_WIDTH)
            return gamma.permute(1, 0, 2).contiguous()

        param = nn.Parameter(
            torch.ones(_PMAX, n_h512, _Q_WIDTH, dtype=config.torch_dtype),
            requires_grad=False,
        )
        set_weight_loader(param, SafetensorsWeightLoader(transform=_kernel_layout))
        return param

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        sp_partials: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        is_decode = self._is_decode(attn_metadata)
        # Sequence-parallel decode is a2a-only: active means the whole a2a path
        # runs, and a shape it refuses keeps the all-reduce path (there is no
        # torch reduce-scatter/all-gather boundary tier). ``positions`` holds
        # every token on every rank; a shard lost that count.
        # ``max_query_len == 1`` keeps speculative decode (Eagle3 widens the query
        # to 1 + draft length) on the all-reduce path; the whole gate must match
        # ``Qwen3ModelMxfp8._decode_sp_layout`` or the backbone shards what the
        # layer treats as full.
        decode_sp_active = (
            is_decode
            and self.decode_sp_enabled
            and attn_metadata[f"layers.{self.layer_idx}.self_attn"]["max_query_len"]
            == 1
            and _decode_sp_viable(positions.numel(), self.world_size)
            and envs.VLLM_NEURON_QWEN3_SP_A2A
            and self.mlp.use_sp_a2a(positions.numel())
        )

        residual = hidden_states
        # Both-boundary a2a: the a2a attention prologue reduces the previous layer's un-reduced MLP partials, adds the residual and norms in-kernel; when this layer cannot consume the carry, close it here in torch — equal to that layer's own reduce-scatter + residual add.
        sp_a2a_attn = False
        sp_a2a_avail = False
        if decode_sp_active:
            from vllm_neuron._kernels.qwen3_32b_sp.qwen3_32b_sp_norm_attn import (
                sp_a2a_attn_available,
            )

            sp_a2a_avail = sp_a2a_attn_available()
        if sp_partials is not None:
            sp_a2a_attn = decode_sp_active and sp_a2a_avail
            if not sp_a2a_attn:
                hidden_states = hidden_states + self.self_attn.tp_group.reduce_scatter(
                    sp_partials, dim=0
                )
                sp_partials = None
                residual = hidden_states
        if sp_a2a_attn:
            attn_in = hidden_states
            gamma_shuffled = self.input_layernorm_gamma_shuffled
            eps = self.rms_norm_eps
        elif decode_sp_active:
            # a2a layer 0 (no carry yet) and the attention-composer-unavailable case: norm the shard here; the megakernel can only fuse after the gather, so pass no gamma.
            attn_in = self.input_layernorm(hidden_states)
            gamma_shuffled = None
            eps = None
        elif is_decode:
            # Fuse the input RMSNorm into the decode megakernel. Pass the raw
            # pre-norm hidden state and the gamma/eps. The kernel normalizes the
            # hidden state inside the QKV matmul. The residual stays the pre-norm
            # tensor, so this is residual-safe.
            attn_in = hidden_states
            # Pass the pre-shuffled gamma; the kernel consumes it directly.
            gamma_shuffled = self.input_layernorm_gamma_shuffled
            eps = self.rms_norm_eps
        else:
            # Attention prefill consumes ROW-packed activations, unlike decode.
            attn_in = _rmsnorm_row_pack(
                hidden_states.to(self.dtype),
                self.input_layernorm.weight,
                self.rms_norm_eps,
            )
            gamma_shuffled = None
            eps = None
        attn_out = self.self_attn(
            hidden_states=attn_in,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
            gamma_shuffled=gamma_shuffled,
            eps=eps,
            decode_sp_active=decode_sp_active,
            sp_a2a_partials=sp_partials if sp_a2a_attn else None,
        )
        if sp_a2a_attn:
            # The composer returns the updated residual stream with the attention partials.
            attn_out, residual = attn_out
        if decode_sp_active:
            # attn_out is the UN-REDUCED [T, H] partials; the a2a MLP block reduces, adds the residual, norms, quantizes, gathers and runs the MLP in one launch; the FLAT pre-shuffled residual carrier is touched by torch only at layer 0 (natural in) and after the last layer (natural out).
            mlp_partials, new_residual = self.mlp.forward_sp_a2a(
                attn_out.to(self.dtype),
                residual.to(self.dtype),
                gamma_shuffled=self.post_attention_layernorm_gamma_sp,
                eps=self.rms_norm_eps,
                residual_in_shuffled=sp_a2a_attn,
                residual_out_shuffled=sp_a2a_avail and not self.is_last_layer,
            )
            # Carry both to the next layer; the model loop closes the carry after the last layer with one reduce-scatter and one add.
            return mlp_partials, new_residual
        # Guard the residual layout: a one-token shard meeting a replicated residual broadcasts ([8, H] + [1, H]) with no error.
        assert attn_out.shape == residual.shape, (
            f"attention output {tuple(attn_out.shape)} does not match the "
            f"residual {tuple(residual.shape)}. The sequence-parallel layout of "
            f"the two streams disagrees (decode_sp_active={decode_sp_active})."
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        mlp_uses_prefill_cte = (not is_decode) and self.mlp.use_prefill_cte(
            hidden_states.shape[0]
        )
        if (
            is_decode
            and not mlp_uses_prefill_cte
            and self.mlp.can_fuse_decode_norm(hidden_states.shape[0])
        ):
            # Fuse the post-attention RMSNorm — DECODE ONLY (residual-safe: it stays pre-norm); the token count gates the branch (see _ILSA901_TOKEN_COUNTS). The activation here is never a token shard: a sequence-parallel decode step returned above, and a short prefill (<= 96 global tokens, decode weights) is sharded, so it takes the outside-norm branch below, whose sp_shard gathers and reduce-scatters correctly.
            mlp_out = self.mlp(
                hidden_states.to(self.dtype),
                is_prefill=False,
                ln_w=self.post_attention_layernorm_gamma_shuffled,
                eps=self.rms_norm_eps,
            )
        elif not mlp_uses_prefill_cte:
            # Norm OUTSIDE the kernel: the nkilib fallback (no norm stage) and the in-tree kernel at a count the fusion cannot compile.
            mlp_out = self.mlp(
                self.post_attention_layernorm(hidden_states),
                is_prefill=False,
                sp_shard=(not is_decode),
            )
        else:
            # Long (CTE) prefill: the CTE MX kernel has no norm stage, so normalize first with the row-pack helper.
            mlp_in = _rmsnorm_row_pack(
                hidden_states.to(self.dtype),
                self.post_attention_layernorm.weight,
                self.rms_norm_eps,
            )
            mlp_out = self.mlp(mlp_in, is_prefill=True, sp_shard=(not is_decode))
        assert mlp_out.shape == residual.shape, (
            f"MLP output {tuple(mlp_out.shape)} does not match the residual "
            f"{tuple(residual.shape)}. The sequence-parallel layout of the two "
            f"streams disagrees (decode_sp_active={decode_sp_active})."
        )
        hidden_states = residual + mlp_out
        return hidden_states


class Qwen3ModelMxfp8(Qwen3ModelBf16):
    def __init__(self, config: Qwen3Config) -> None:
        nn.Module.__init__(self)
        self.config = config

        # Set by the runner via set_aux_hidden_state_layers(); the inherited
        # BF16 forward references it (Eagle3 aux-hidden-state path). MXFP8 has
        # no Eagle3 but shares that forward, so the attribute must exist. This
        # __init__ bypasses Qwen3ModelBf16.__init__ (which sets it), so set it
        # here to match the bf16/llama3/gpt_oss inner models. Empty = no aux
        # capture.
        self.aux_hidden_state_layers = []

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        # The inherited BF16 forward reads this, and this __init__ bypasses the base.
        self.decode_sp_enabled = _decode_sp_enabled(config)

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayerMxfp8(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Qwen3RotaryEmbedding(config)

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

    def _decode_sp_layout(self, num_tokens: int, max_query_len: int) -> bool:
        """Sequence-parallel decode is a2a-only: the backbone shards the token
        axis exactly when the layers run the a2a composers, so this mirrors the
        ``decode_sp_active`` gate of ``Qwen3DecoderLayerMxfp8.forward``. A
        mismatch shards what the layers treat as full (or the reverse), and
        every rank then reduces a different token slice with no shape error."""
        return (
            super()._decode_sp_layout(num_tokens, max_query_len)
            and envs.VLLM_NEURON_QWEN3_SP_A2A
            and self.layers[0].mlp.use_sp_a2a(num_tokens)
        )


class Qwen3ForCausalLMMxfp8(Qwen3ForCausalLMBf16):
    def __init__(self, config: Qwen3Config) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.model = Qwen3ModelMxfp8(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config
            if config.neuron_config
            else None
        )
        debug_logits_enabled = (
            config.neuron_config is not None
            and config.neuron_config.debug_logits_dir is not None
        )
        self._gather_logits = (
            config.neuron_config is not None and config.neuron_config.max_logprobs != 0
        ) or debug_logits_enabled

        # Store lm_head weights transposed ([in, out]) so the decode (TKG) graph uses
        # torch.matmul instead of F.linear, dropping the ~1.5ms transpose at the end of
        # the decode profile (measured on TRN3 FP8-KV bs16: 1.5ms -> 0.33ms between the
        # AllReduce and AllGather; see PR #3716, applied from Qwen3-VL to Qwen3 dense MXFP8).
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
            weight_layout="in_out",
        )
        # Shard the vocab (output) dim; in the "in_out" layout that is dim 1, and the
        # checkpoint stores [out, in] so the loader transposes on load.
        transposed = self.lm_head.weight_layout == "in_out"
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(
                shard_dim=1 if transposed else 0,
                shard_size=config.vocab_size // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=transposed,
            ),
        )

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=self.tp_group.device_group,
            )

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor, torch.Tensor]]):
        # Override the BF16 base to detect the swizzled packed FP8 K layout and
        # set each attention's fp8_packed flag. Packed K is
        # [num_blocks, kv_heads, block_size // 2, head_dim, 2] -- one dim higher
        # than V (which is never packed); a bf16 or standard 4-D FP8 K cache
        # stays the same rank as V, so fp8_packed stays False.
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise KeyError(f"KV cache for layer {layer_name} not initialized")
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]
            # Packed K is one dim higher than V (V is never packed); a bf16 or
            # standard 4-D FP8 K cache stays the same rank as V. Every dense
            # attention layer is a Qwen3AttentionMxfp8 (which always defines
            # fp8_packed), so set it directly from the K/V rank delta.
            layer.self_attn.fp8_packed = (
                layer.self_attn.k_cache.dim() == layer.self_attn.v_cache.dim() + 1
            )

    @staticmethod
    def _get_quantization_spec(
        hf_config: PretrainedConfig | dict | None,
    ) -> QuantizationSpec:
        # Reuse the factory's parser (single source for the getattr/to_dict
        # extraction) and layer on the raise-if-unsupported guard that the
        # factory copy intentionally omits, so this class is also safe to
        # construct directly (not only via the factory).
        from .factory import Qwen3ForCausalLM as Qwen3Factory

        spec = Qwen3Factory._get_quantization_spec(hf_config)
        if spec is None or spec.linear_scheme not in _SUPPORTED_QWEN3_MXFP8_SCHEMES:
            raise ValueError(
                "Qwen3 dense MXFP8 model surface requires a supported "
                "compressed-tensors MXFP8 checkpoint contract."
            )
        return spec

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig
    ) -> "Qwen3ForCausalLMMxfp8":
        cls._get_quantization_spec(hf_config)
        config = Qwen3Config.from_configs(hf_config, neuron_config)
        return cls(config)

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        tp_rank = self.rank
        tp_size = self.world_size

        mappings: dict[str, str | list[str]] = {
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            "lm_head.weight": "lm_head.weight",
            "model.norm.weight": "model.norm.weight",
        }

        # A decode-only pool creates no prefill parameter, so drop the prefill
        # keys from the mappings and from the post-load recheck below. This names
        # the weight set to load, not the step type: `is_prefill` already means
        # "this step is a prefill" elsewhere in this file.
        load_prefill_weights = not _decode_only_weights(self.config)

        quant_suffixes = [
            "self_attn.qkv_proj_weight",
            "self_attn.qkv_weight_scale",
            "self_attn.o_proj_weight_decode",
            "self_attn.o_proj_weight_scale",
            "mlp.gate_proj_weight_decode",
            "mlp.gate_proj_weight_scale_decode",
            "mlp.up_proj_weight_decode",
            "mlp.up_proj_weight_scale_decode",
            "mlp.down_proj_weight_decode",
            "mlp.down_proj_weight_scale_decode",
        ]
        if load_prefill_weights:
            quant_suffixes += [
                "self_attn.o_proj_weight_prefill",
                "mlp.gate_proj_weight_prefill",
                "mlp.gate_proj_weight_scale_prefill",
                "mlp.up_proj_weight_prefill",
                "mlp.up_proj_weight_scale_prefill",
                "mlp.down_proj_weight_prefill",
                "mlp.down_proj_weight_scale_prefill",
            ]

        for layer_id in range(len(self.model.layers)):
            hf = f"model.layers.{layer_id}"
            ours = f"model.layers.{layer_id}"
            # Dense Qwen3 keeps its own native-MX mapping helpers so this path
            # can evolve without reaching through the Qwen3-VL package.
            mappings.update(
                build_mx_attention_mappings(
                    f"{hf}.self_attn", layer_prefix=ours, prefill=load_prefill_weights
                )
            )
            mappings.update(
                build_mx_mlp_mappings(
                    f"{hf}.mlp",
                    layer_prefix=ours,
                    decode=True,
                    prefill=load_prefill_weights,
                )
            )
            mappings[f"{ours}.self_attn.q_norm.weight"] = (
                f"{hf}.self_attn.q_norm.weight"
            )
            mappings[f"{ours}.self_attn.k_norm.weight"] = (
                f"{hf}.self_attn.k_norm.weight"
            )
            mappings[f"{ours}.input_layernorm.weight"] = f"{hf}.input_layernorm.weight"
            # Same checkpoint source, pre-shuffled for decode's fused RMSNorm.
            mappings[f"{ours}.input_layernorm_gamma_shuffled"] = (
                f"{hf}.input_layernorm.weight"
            )
            mappings[f"{ours}.post_attention_layernorm.weight"] = (
                f"{hf}.post_attention_layernorm.weight"
            )
            # Same checkpoint source, pre-shuffled for the decode MLP's fused
            # RMSNorm.
            mappings[f"{ours}.post_attention_layernorm_gamma_shuffled"] = (
                f"{hf}.post_attention_layernorm.weight"
            )
            # Same source in the [1, H] SP order; without it the buffer keeps its ones init and the norm is a silent identity.
            # The parameter exists only when the config asks for SP decode.
            if _decode_sp_enabled(self.config):
                mappings[f"{ours}.post_attention_layernorm_gamma_sp"] = (
                    f"{hf}.post_attention_layernorm.weight"
                )

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        available = checkpoint.get_tensor_names()

        # Fail before sharded load if the checkpoint contract is incomplete;
        # otherwise missing MX tensors surface later as harder-to-debug runtime
        # layout errors. Raise (not assert) so the check survives `python -O`.
        for fqn, keys in mappings.items():
            for key in [keys] if isinstance(keys, str) else keys:
                if key not in available:
                    raise KeyError(f"missing MXFP8 key {key!r} for {fqn}")

        result = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, mappings, device, strict=False
        )

        target_dtype = self.config.torch_dtype
        native_mx_dtypes = (torch.uint8, torch.uint32, _FP8_DTYPE)
        for name, tensor in result.state_dict.items():
            # Keep prepacked MX tensors in their native storage dtype; only cast
            # ordinary bf16 weights and norms to the model's execution dtype.
            if tensor.dtype in native_mx_dtypes:
                continue
            if tensor.dtype != target_dtype:
                result.state_dict[name] = tensor.to(target_dtype)

        for layer_id in range(len(self.model.layers)):
            for suffix in quant_suffixes:
                fqn = f"model.layers.{layer_id}.{suffix}"
                # The sharded loader runs with strict=False because the mapping
                # helpers expand into multiple packed tensors. Recheck the MX
                # surfaces we require before materializing the state dict. Raise
                # (not assert) so the check survives `python -O`.
                if fqn not in result.state_dict:
                    raise KeyError(
                        f"MXFP8 weight {fqn!r} not loaded (check checkpoint keys)"
                    )

        self._load_kv_cache_scales(checkpoint, device)
        self.load_state_dict(result.state_dict, strict=False, assign=True)
