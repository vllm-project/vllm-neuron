# SPDX-License-Identifier: Apache-2.0
"""a2a sequence-parallel attention composer for Qwen3-32B dense MXFP8 decode (LNC1).

Twin of ``qwen3_32b_sp_norm_mlp.py`` for the pre-attention block boundary. One
launch takes the previous layer's UN-REDUCED MLP partials and the FLAT
pre-shuffled residual carrier and chains::

    carrier load -> QKV weight prefetch (DMAs stream under the collectives)
    -> all-to-all + fp32 PE reduce + residual add -> carrier store
    -> input RMSNorm -> MX quantize -> packed SBUF all-gather
    -> nkilib attention megakernel on the pre-quantized activation

The quantized pair already matches nkilib's internal ``[128, H // 512, T]``
``float8_e4m3fn_x4`` + uint8 E8M0 layout, so the megakernel's QKV projection
consumes it with no relayout.

Layouts
-------
``H = 5120``, ``T_s`` this rank's token shard, ``T = T_s * W`` at
tensor-parallel degree ``W`` (release shape: ``T_s = 8``, ``W = 8``, bs64).

- ``sp_partials``: bf16 HBM ``[T, H]`` un-reduced MLP partials.
- ``sp_residual``: bf16 HBM ``[T_s, H]`` carrier, FLAT pre-shuffled (this
  composer is always mid-chain).
- ``sp_gamma``: bf16 ``[1, H]`` input RMSNorm weight, ``pre_shuffle_h`` order.
- QKV weights/scales: nkilib's fp8 ``[H // 4, I, 4]`` + uint8 ``[H // 32, I]``,
  preloaded exact-shaped for the single-block decode geometry
  (``H_packed_shard = 640``, ``I = 1280``).
- returns: the megakernel outputs plus the updated bf16 ``[T_s, H]`` carrier.

Port notes (vs nkilib ``_qkv_tkg_mx_impl`` and ``attention_block_tkg``)
-----------------------------------------------------------------------
The projection-side compute mirrors nkilib ``_qkv_tkg_mx_impl``. The deliberate
changes:

1. Steps 1 (load + norm) and 2 (quantize) of ``_qkv_tkg_mx_impl`` are removed:
   the prologue produced the quantized pair in SBUF, so ``_qkv_mx_prequant``
   builds the same validated config and enters at the projection stage.
2. The QKV weight and scale DMAs are issued BEFORE the collectives
   (``_preload_qkv_weights``) and consumed by the vendored projection in
   ``qwen3_32b_qkv_tkg_projection_mxfp_preloaded.py``, which skips its own
   loads; multi-block geometry falls back to nkilib's stock projection.
3. nkilib has no parameter for a pre-quantized activation yet, so this module
   REBINDS the ``qkv`` symbol inside ``attention_block_tkg`` with a trace-time
   interceptor: the composer arms a module-level holder for exactly one
   megakernel call (try/finally disarms it), and with the holder empty the
   original ``qkv`` runs unchanged, so every other model is unaffected. The
   megakernel's ``X`` argument becomes a never-read shape/dtype placeholder.
4. The updated residual carrier is appended to the megakernel outputs, so the
   caller carries it to the layer's MLP composer.

This seam shrinks to the composer alone once nkilib grows
``hidden_qtz_sb``/``hidden_scales_sb`` and preloaded-weights parameters.
"""

from typing import Optional

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
import torch
from nkilib.core.utils.common_types import QuantizationType
from torch import Tensor

from vllm_neuron._kernels.qwen3_32b_sp.qwen3_32b_sp_prologue import (
    sp_gather_replicated,
)

# These nkilib internals are unstable API; missing them only disables this path.
try:
    from nkilib.core.qkv.qkv_tkg_mx_impl import _qkv_tkg_projection_mxfp

    from vllm_neuron._kernels.qwen3_32b_sp.qwen3_32b_qkv_tkg_projection_mxfp_preloaded import (  # noqa: E501
        _qkv_tkg_projection_mxfp_preloaded,
    )
    from nkilib.core.qkv.qkv_tkg_mx_utils import (
        QKV_TKG_MXFP_UserInput,
        _build_config,
        _validate_user_inputs,
    )
    from nkilib.experimental.transformer import attention_block_tkg as _abt_module
    from nkilib.experimental.transformer.attention_block_tkg import (
        attention_block_tkg as _nki_attention_block_tkg,
    )

    _SP_ATTN_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the installed nkilib
    _abt_module = None
    _nki_attention_block_tkg = None
    _SP_ATTN_AVAILABLE = False

__all__ = ["sp_a2a_attn_available"]

_P = 128
_TILE = 512

# Per-trace holder the composer arms; tracing is single-threaded, so a list is safe.
_SP_PREQUANT_HOLDER: list = []


def _preload_qkv_weights(weights_qtz_hbm, weight_scales_hbm):
    """Issue this core's QKV weight and scale DMAs, exact-shaped, right now.

    The composer calls this BETWEEN the prologue's norm and gather stages, so
    the QKV weights stream on the DGE queues while the exchange and the
    all-gather run on the collective queue. The buffers and the access
    patterns mirror ``_qkv_tkg_projection_mxfp`` for its single-block case
    (one I block, one H weight block, one buffer) — the Qwen3-32B decode
    geometry: H_packed_shard = 640 <= 2048 and I = 1280 <= 4096. The buffer
    keeps only the H-128 tiles this core reads (5), not the padded 16 of
    nkilib's own allocation.

    Args:
        weights_qtz_hbm: fp8 HBM ``[H // 4, I, 4]`` MXFP8 QKV weight.
        weight_scales_hbm: uint8 HBM ``[H // 32, I]`` E8M0 QKV weight scale.

    Returns:
        ``(weights_qtz_sb, weight_scales_sb)``, or ``None`` for multi-block geometry.

    Example:
        >>> # traced inside the sequence-parallel attention composer
    """
    h_packed, i_size, _q = weights_qtz_hbm.shape
    num_shards = nl.num_programs(axes=0)
    shard_id = nl.program_id(axis=0)
    h_packed_shard = h_packed // num_shards

    # Single-block geometry only; anything else keeps nkilib's own loads.
    if i_size > 4096 or h_packed_shard > 2048 or h_packed_shard % 128:
        return None
    n_tiles = h_packed_shard // 128

    weights_qtz_sb = nl.ndarray(
        (128, 1, n_tiles, i_size * 4),
        dtype=weights_qtz_hbm.dtype,
        buffer=nl.sbuf,
        name="qkv_w_pre",
    )
    weight_scales_sb = nl.ndarray(
        (128, 1, n_tiles, i_size),
        dtype=nl.uint8,
        buffer=nl.sbuf,
        name="qkv_s_pre",
    )

    # One DMA for the whole shard; the patterns mirror the projection's.
    nisa.dma_copy(
        dst=weights_qtz_sb.ap(
            pattern=[
                [n_tiles * i_size * 4, 128],
                [i_size * 4, n_tiles],
                [1, i_size * 4],
            ],
            offset=0,
        ),
        src=weights_qtz_hbm.ap(
            pattern=[
                [i_size * 4, 128],
                [128 * i_size * 4, n_tiles],
                [1, i_size * 4],
            ],
            offset=shard_id * h_packed_shard * i_size * 4,
        ),
    )

    # 16 scale rows per H-512 tile scatter over 4 SBUF quadrants, 4 rows each (MX matmul layout).
    scale_rows_shard = h_packed_shard // 8
    for h_tile in range(n_tiles):
        for quad in range(4):
            nisa.dma_copy(
                dst=weight_scales_sb[quad * 32 : quad * 32 + 4, 0, h_tile, :i_size],
                src=weight_scales_hbm.ap(
                    pattern=[[i_size, 4], [1, i_size]],
                    offset=shard_id * scale_rows_shard * i_size
                    + (h_tile * 16 + quad * 4) * i_size,
                    dtype=nl.uint8,
                ),
            )
    return weights_qtz_sb, weight_scales_sb


def _qkv_mx_prequant(q_act, s_act, qkv_preloaded=None, **qkv_kwargs):
    """Run nkilib's MX QKV projection on a pre-quantized SBUF activation.

    Mirrors ``_qkv_tkg_mx_impl`` with steps 1 (load + norm) and 2 (quantize)
    removed: build the same validated config, then call the projection stage
    with the prologue's tensors.

    Args:
        q_act: float8_e4m3fn_x4 SBUF activation ``[128, H // 512, T]``; half
            the H tiles when pre-sharded.
        s_act: uint8 SBUF E8M0 scales, same shape as ``q_act``.
        **qkv_kwargs: The keyword arguments of the intercepted ``qkv()`` call.

    Returns:
        The QKV output, reshaped to ``[B, S, I]`` like ``_qkv_tkg_mx_impl``.

    Example:
        >>> # called by the qkv interceptor inside the megakernel trace
    """
    hidden = qkv_kwargs["input"]
    user_inputs = QKV_TKG_MXFP_UserInput(
        hidden=hidden,
        weights_qtz_hbm=qkv_kwargs["fused_qkv_weights"],
        norm_w=None,
        d_head=qkv_kwargs.get("d_head"),
        num_kv_heads=qkv_kwargs.get("num_kv_heads"),
        num_q_heads=qkv_kwargs.get("num_q_heads"),
        output_layout=qkv_kwargs["output_layout"],
        eps=qkv_kwargs.get("norm_eps", 1e-6),
        norm_type=qkv_kwargs["fused_norm_type"],
        quantization_type=qkv_kwargs["quantization_type"],
        is_h_dim_4h_transposed=qkv_kwargs.get("is_h_dim_4h_transposed", False),
        weight_scales_hbm=qkv_kwargs.get("qkv_w_scale"),
        input_scale_hbm=qkv_kwargs.get("qkv_in_scale"),
        output_in_sbuf=qkv_kwargs.get("store_output_in_sbuf", False),
        qkv_bias=qkv_kwargs.get("bias"),
        norm_bias=None,
        hidden_actual=qkv_kwargs.get("hidden_actual"),
        sbm=qkv_kwargs.get("sbm"),
    )
    _validate_user_inputs(user_inputs)
    cfg = _build_config(user_inputs)

    n_h512 = hidden.shape[2] // _TILE
    n_h512_shard = n_h512 // cfg.num_shards
    assert q_act.shape[1] == n_h512, (
        f"replicated activation has {q_act.shape[1]} H tiles, expected {n_h512}."
    )
    h512_off = cfg.shard_id * n_h512_shard
    q_shard = q_act.slice(1, h512_off, h512_off + n_h512_shard)
    s_shard = s_act.slice(1, h512_off, h512_off + n_h512_shard)

    # With preloaded weights the vendored projection skips its own loads.
    if qkv_preloaded is not None:
        output_hbm = _qkv_tkg_projection_mxfp_preloaded(
            hidden_qtz_sb=q_shard,
            hidden_scales_sb=s_shard,
            weights_qtz_hbm=user_inputs.weights_qtz_hbm,
            weight_scales_hbm=user_inputs.weight_scales_hbm,
            cfg=cfg,
            bias_hbm=user_inputs.qkv_bias,
            input_scale=None,
            preloaded=qkv_preloaded,
        )
    else:
        output_hbm = _qkv_tkg_projection_mxfp(
            hidden_qtz_sb=q_shard,
            hidden_scales_sb=s_shard,
            weights_qtz_hbm=user_inputs.weights_qtz_hbm,
            weight_scales_hbm=user_inputs.weight_scales_hbm,
            cfg=cfg,
            bias_hbm=user_inputs.qkv_bias,
            input_scale=None,
        )
    return output_hbm.reshape((cfg.B, cfg.S, cfg.I))


# Idempotent capture: the tag stops a reload from re-wrapping the interceptor.
if _SP_ATTN_AVAILABLE and not getattr(_abt_module.qkv, "_sp_prequant_intercept", False):
    _ORIG_QKV = _abt_module.qkv
elif _SP_ATTN_AVAILABLE:
    _ORIG_QKV = _abt_module.qkv._sp_prequant_orig
else:
    _ORIG_QKV = None


def _qkv_intercept(**qkv_kwargs):
    """Route the megakernel's QKV call to the pre-quantized projection.

    Falls through to nkilib's original ``qkv`` unless the composer set the
    holder, so every non-sequence-parallel trace is unchanged.

    Example:
        >>> # installed over attention_block_tkg.qkv at import time
    """
    if not _SP_PREQUANT_HOLDER:
        return _ORIG_QKV(**qkv_kwargs)
    # Consume the entry so a trace failure past here cannot leak it into a later trace.
    q_act, s_act, qkv_preloaded = _SP_PREQUANT_HOLDER.pop()
    return _qkv_mx_prequant(q_act, s_act, qkv_preloaded=qkv_preloaded, **qkv_kwargs)


# Rebind the name the megakernel resolves at call time (see module docstring).
if _SP_ATTN_AVAILABLE:
    _qkv_intercept._sp_prequant_intercept = True
    _qkv_intercept._sp_prequant_orig = _ORIG_QKV
    _abt_module.qkv = _qkv_intercept


def sp_a2a_attn_available() -> bool:
    """Return True when the a2a attention composer can run.

    The composer needs the nkilib QKV seam of this module. The caller checks
    the shape rules with ``qwen3_sp_a2a_supports`` and the one-token rule.

    Example:
        >>> # ok = sp_a2a_attn_available()
    """
    from vllm_neuron.utils.neuron_utils import can_run_kernel

    return _SP_ATTN_AVAILABLE and can_run_kernel()


def sp_a2a_norm_attn_block_body(
    sp_partials: nl.ndarray,
    sp_residual: nl.ndarray,
    sp_gamma: nl.ndarray,
    sp_eps: float,
    sp_tp_degree: int,
    sp_replica_mesh: tuple,
    **attention_kwargs,
):
    """Trace body: reduce + residual + norm + quantize + gather + attention.

    The LNC1 all-to-all twin of ``sp_norm_attn_block_body``. The caller gives
    the UN-REDUCED MLP partials of the previous layer and the natural-order
    residual shard. One launch runs the collective (all-to-all + fp32 PE
    reduction), the residual add, the input RMSNorm,
    the MX quantize, the packed SBUF all-gather, and the attention
    megakernel on the pre-quantized activation. This absorbs the torch
    reduce-scatter + residual add + norm + bf16 all-gather of the
    pre-attention boundary, the same way ``_qwen3_sp_a2a_mlp_body`` absorbs
    the pre-MLP boundary.

    Args:
        sp_partials: bf16 HBM ``[T, H]`` un-reduced MLP partial sums.
        sp_residual: bf16 HBM ``[T_s, H]`` residual carrier, FLAT pre-shuffled layout.
        sp_gamma: bf16 HBM ``[1, H]`` input RMSNorm gamma, ``pre_shuffle_h`` order.
        sp_eps: Input RMSNorm epsilon.
        sp_tp_degree: Tensor-parallel degree.
        sp_replica_mesh: Every cell of the tensor-parallel replica mesh.

    Returns:
        ``[output, k, v, residual_out]``; ``residual_out`` stays FLAT
            pre-shuffled, bf16 HBM ``[T_s, H]``.

    Example:
        >>> # traced by wrap_nki via attention_decode; not called directly
    """
    from vllm_neuron._kernels.mlp.qwen3_32b_mxfp8_mlp_tkg import _quantize_mx_3d
    from vllm_neuron._kernels.norm.qwen3_32b_mxfp8_rmsnorm_quant_tkg import (
        _compute_rmsnorm,
        _load_gamma,
        _swizzle_to_mlp_layout,
    )
    from vllm_neuron._kernels.qwen3_32b_sp.qwen3_32b_sp_prologue import (
        sp_a2a_reduce_residual,
        sp_load_shuffled_flat,
        sp_store_shuffled_flat,
    )

    assert _SP_ATTN_AVAILABLE, (
        "the installed nkilib misses the QKV MX internals this composer "
        "needs; sp_a2a_attn_available must gate this path."
    )
    t_full, h = sp_partials.shape
    t_shard = t_full // sp_tp_degree
    n512 = h // _TILE
    h1 = h // _P

    # Use an explicit loop: the NKI tracer rejects a list comprehension.
    cells = []
    for cell in sp_replica_mesh:
        cells.append(list(cell))
    rg = ncc.ReplicaGroup(cells)

    # Small DMAs first (they gate the critical path), then the collective.
    gamma_sb = _load_gamma(sp_gamma, h1)
    # The residual carrier is FLAT pre-shuffled: one contiguous DMA, not 2-byte-element strided.
    residual_sb = sp_load_shuffled_flat(sp_residual, t_shard, h, "a2a_attn_resid")

    # Prefetch the QKV weight DMAs before the collective so they stream on the DGE queues.
    qkv_preloaded = _preload_qkv_weights(
        attention_kwargs["W_qkv"],
        attention_kwargs["weight_dequant_scale_qkv"],
    )

    state = sp_a2a_reduce_residual(sp_partials, residual_sb, sp_tp_degree, rg)

    # Return the updated residual stream to torch in natural order.
    residual_out = nl.ndarray((t_shard, h), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    sp_store_shuffled_flat(state, residual_out, t_shard, h)

    # Norm + MX quantize + packed gather — the a2a MLP composer's stages, feeding QKV.
    normed = _compute_rmsnorm(state, gamma_sb, sp_eps, h, t_shard, h1)
    swizzled = _swizzle_to_mlp_layout(normed, n512, t_shard, h1)
    q_n, s_n = _quantize_mx_3d(swizzled, n512, t_shard, "a2a_attn_norm")
    q_act, s_act = sp_gather_replicated(q_n, s_n, t_shard, h, sp_tp_degree, rg)

    # Shape/dtype placeholder for the megakernel's X; no instruction ever reads it.
    x_placeholder = nl.ndarray(
        (t_full, 1, h), dtype=sp_residual.dtype, buffer=nl.shared_hbm
    )

    # Arm the interceptor; the try/finally below disarms it on every exit.
    del _SP_PREQUANT_HOLDER[:]
    _SP_PREQUANT_HOLDER.append((q_act, s_act, qkv_preloaded))
    try:
        result = _nki_attention_block_tkg(
            X=x_placeholder,
            rmsnorm_X_enabled=False,
            rmsnorm_X_eps=None,
            rmsnorm_X_gamma=None,
            **attention_kwargs,
        )
    finally:
        armed = bool(_SP_PREQUANT_HOLDER)
        del _SP_PREQUANT_HOLDER[:]
    assert not armed, (
        "the qkv interceptor did not consume the pre-quantized activation: "
        "the megakernel never reached its QKV stage."
    )
    out, k_upd, v_upd = result
    return [out, k_upd, v_upd, residual_out]


@nki.jit
def _torch_compatible_sp_a2a_attn_kernel(
    # -- a2a prologue input (replaces X + rmsnorm_X_*): MLP partials + residual shard
    sp_partials: Tensor,
    sp_residual: Tensor = None,
    sp_gamma: Tensor = None,
    sp_eps: float = 1e-6,
    sp_tp_degree: int = 1,
    sp_replica_mesh: tuple = (),
    # -- input meta
    X_hidden_dim_actual: int = None,
    # -- qkv projections
    W_qkv: Tensor = None,
    bias_qkv: Tensor = None,
    quantization_type_qkv: QuantizationType = QuantizationType.NONE,
    weight_dequant_scale_qkv: torch.Tensor = None,
    input_dequant_scale_qkv: torch.Tensor = None,
    # -- Q/K processing: pre-RoPE RMSNorm
    rmsnorm_QK_pre_rope_enabled: bool = False,
    rmsnorm_QK_pre_rope_eps: float = 1e-6,
    rmsnorm_QK_pre_rope_W_Q: Tensor = None,
    rmsnorm_QK_pre_rope_W_K: Tensor = None,
    # -- Q/K processing: RoPE
    cos: Optional[Tensor] = None,
    sin: Optional[Tensor] = None,
    rope_contiguous_layout: bool = True,
    # -- Q/K processing: post-RoPE RMSNorm
    rmsnorm_QK_post_rope_enabled: bool = False,
    rmsnorm_QK_post_rope_eps: float = 1e-6,
    rmsnorm_QK_post_rope_W_Q: Tensor = None,
    rmsnorm_QK_post_rope_W_K: Tensor = None,
    # -- attention
    K_cache_transposed: bool = False,
    active_blocks_table: Tensor = None,
    K_cache: Tensor = None,
    V_cache: Tensor = None,
    attention_mask: Tensor = None,
    sink: Tensor = None,
    softmax_scale: float = None,
    # -- in-kernel mask generation
    pos_ids: Tensor = None,
    swa_start_pos_ids: Tensor = None,
    # -- KV cache update
    update_cache: bool = False,
    kv_cache_update_idx: Tensor = None,
    k_scale: Tensor = None,
    v_scale: Tensor = None,
    # -- packed FP8 K cache layout
    fp8_packed: bool = False,
    # -- output projection
    W_out: Tensor = None,
    bias_out: Tensor = None,
    quantization_type_out: QuantizationType = QuantizationType.NONE,
    weight_dequant_scale_out: Tensor = None,
    input_dequant_scale_out: Tensor = None,
    transposed_out: bool = False,
    # -- output control
    out_in_sb: bool = False,
    skip_attention: bool = False,
    # -- STATIC_MX layout flag
    is_h_transposed_by_4: bool = False,
    # -- sbm
    sbm_lower_bound: int = None,
    sbm_upper_bound: int = None,
    sbm_use_auto_alloc: bool = True,
    sbm_default_stack_alloc: bool = True,
):
    """
    All-to-all sequence-parallel decode entry (LNC1): the kernel prologue
    (collective + fp32 PE reduction + residual add + norm + MX quantize +
    packed SBUF all-gather) replaces the ``X`` input, the ``rmsnorm_X_*``
    fusion and the torch pre-attention boundary. Appends the updated
    residual stream to the megakernel outputs. See the module docstring
    for the design.
    """
    from vllm_neuron.functional.attention.attention_decode import _maybe_build_sbm

    sbm = _maybe_build_sbm(
        sbm_lower_bound, sbm_upper_bound, sbm_use_auto_alloc, sbm_default_stack_alloc
    )

    return sp_a2a_norm_attn_block_body(
        sp_partials,
        sp_residual,
        sp_gamma,
        sp_eps,
        sp_tp_degree,
        sp_replica_mesh,
        X_hidden_dim_actual=X_hidden_dim_actual,
        W_qkv=W_qkv,
        bias_qkv=bias_qkv,
        quantization_type_qkv=quantization_type_qkv,
        weight_dequant_scale_qkv=weight_dequant_scale_qkv,
        input_dequant_scale_qkv=input_dequant_scale_qkv,
        rmsnorm_QK_pre_rope_enabled=rmsnorm_QK_pre_rope_enabled,
        rmsnorm_QK_pre_rope_eps=rmsnorm_QK_pre_rope_eps,
        rmsnorm_QK_pre_rope_W_Q=rmsnorm_QK_pre_rope_W_Q,
        rmsnorm_QK_pre_rope_W_K=rmsnorm_QK_pre_rope_W_K,
        cos=cos,
        sin=sin,
        rope_contiguous_layout=rope_contiguous_layout,
        rmsnorm_QK_post_rope_enabled=rmsnorm_QK_post_rope_enabled,
        rmsnorm_QK_post_rope_eps=rmsnorm_QK_post_rope_eps,
        rmsnorm_QK_post_rope_W_Q=rmsnorm_QK_post_rope_W_Q,
        rmsnorm_QK_post_rope_W_K=rmsnorm_QK_post_rope_W_K,
        K_cache_transposed=K_cache_transposed,
        active_blocks_table=active_blocks_table,
        K_cache=K_cache,
        V_cache=V_cache,
        attention_mask=attention_mask,
        sink=sink,
        softmax_scale=softmax_scale,
        update_cache=update_cache,
        kv_cache_update_idx=kv_cache_update_idx,
        k_scale=k_scale,
        v_scale=v_scale,
        fp8_packed=fp8_packed,
        W_out=W_out,
        bias_out=bias_out,
        quantization_type_out=quantization_type_out,
        weight_dequant_scale_out=weight_dequant_scale_out,
        input_dequant_scale_out=input_dequant_scale_out,
        transposed_out=transposed_out,
        out_in_sb=out_in_sb,
        sbm=sbm,
        skip_attention=skip_attention,
        is_h_transposed_by_4=is_h_transposed_by_4,
        KVDP=1,
        KVDP_replica_group=None,
        pos_ids=pos_ids,
        swa_start_pos_ids=swa_start_pos_ids,
        S_ctx=None,
    )


def attention_decode_sp_a2a(
    sp_partials: torch.Tensor,
    sp_residual: torch.Tensor,
    sp_gamma: torch.Tensor,
    sp_eps: float,
    sp_tp_degree: int,
    sp_replica_mesh: tuple,
    *,
    return_updated_kv: bool = False,
    **megakernel_kwargs,
):
    """Sequence-parallel a2a attention entry — the SP twin of
    ``NF.attention_decode``, kept OUT of the shared functional layer so the
    path every other model takes stays untouched.

    Args:
        sp_partials: bf16 ``[T, H]`` un-reduced MLP partials of the previous layer.
        sp_residual: bf16 ``[T_s, H]`` residual carrier, FLAT pre-shuffled.
        sp_gamma: bf16 ``[1, H]`` input RMSNorm gamma, ``pre_shuffle_h`` order.
        sp_eps: Input RMSNorm epsilon.
        sp_tp_degree: Tensor-parallel degree.
        sp_replica_mesh: Every cell of the tensor-parallel replica mesh.
        return_updated_kv: With ``update_cache`` set, also return the aliased
            K/V outputs (the packed-fallback contract of ``attention_decode``).
        **megakernel_kwargs: The megakernel arguments of
            ``_torch_compatible_sp_a2a_attn_kernel`` (weights, RoPE, cache,
            output projection, sbm bounds).

    Returns:
        ``(result, residual_out)``. ``result`` follows ``attention_decode``'s
        contract (the attention output alone under ``update_cache`` unless
        ``return_updated_kv``); ``residual_out`` is the updated bf16
        ``[T_s, H]`` carrier.

    Example:
        >>> # out, carrier = attention_decode_sp_a2a(partials, residual, gamma, 1e-6, 8, mesh, W_qkv=..., ...)
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from vllm_neuron.utils.hardware_config import get_logical_nc_config

    if sp_residual is None:
        raise ValueError(
            "sp_partials needs sp_residual: the composer adds the residual "
            "stream in the kernel."
        )
    if None in (sp_gamma, sp_eps, sp_tp_degree, sp_replica_mesh):
        raise ValueError(
            "sp_partials needs sp_gamma, sp_eps, sp_tp_degree and sp_replica_mesh."
        )
    active_blocks_table = megakernel_kwargs.get("active_blocks_table")
    if active_blocks_table is not None:
        assert active_blocks_table.dtype == torch.int32, (
            f"active_blocks_table must be int32 with -1 padding, got "
            f"{active_blocks_table.dtype}"
        )

    # Mask preparation, mirroring the shared ``attention_decode``: with
    # ``pos_ids`` set, fp8 KV (``k_scale``) takes the full external mask (the
    # fused on-chip mask changes the softmax/PV accumulation order), otherwise
    # the kernel generates the prior mask on-chip and ``attention_mask``
    # carries only the active-token overlay. The full pre-generated mask is
    # then transposed when nkilib's QK-swap path expects it.
    from vllm_neuron.functional.attention.attention_decode import (
        _build_default_active_mask,
        _infer_kv_heads,
        _maybe_transpose_mask_for_qk_swap,
    )
    from vllm_neuron.functional.attention.attention_decode_mask import (
        gen_attention_decode_mask,
    )

    pos_ids = megakernel_kwargs.get("pos_ids")
    attention_mask = megakernel_kwargs.get("attention_mask")
    swa_start_pos_ids = megakernel_kwargs.get("swa_start_pos_ids")
    V_cache = megakernel_kwargs.get("V_cache")
    W_qkv = megakernel_kwargs.get("W_qkv")
    k_scale = megakernel_kwargs.get("k_scale")
    fp8_packed = megakernel_kwargs.get("fp8_packed", False)
    if pos_ids is not None:
        if active_blocks_table is None:
            raise ValueError(
                "pos_ids (fused mask-gen) requires block KV cache "
                "(active_blocks_table must be provided)"
            )
        # The composer runs one token per sequence (the a2a gate enforces it).
        b_x = sp_partials.shape[0]
        s_tkg_x = 1
        kv_heads = _infer_kv_heads(V_cache)
        d_head_local = V_cache.shape[-1]
        q_heads_local = W_qkv.shape[1] // d_head_local - 2 * kv_heads
        block_len_local = V_cache.shape[1] if V_cache.dim() == 3 else V_cache.shape[2]
        s_ctx_local = active_blocks_table.shape[-1] * block_len_local
        if k_scale is not None:
            attention_mask = gen_attention_decode_mask(
                pos_ids=pos_ids.reshape(1, b_x * s_tkg_x).to(torch.float32),
                bs=b_x,
                q_head=q_heads_local,
                s_active=s_tkg_x,
                s_prior=s_ctx_local,
                start_pos=swa_start_pos_ids.reshape(1, b_x * s_tkg_x).to(torch.float32)
                if swa_start_pos_ids is not None
                else None,
                block_len=block_len_local,
            )
            pos_ids = None
            swa_start_pos_ids = None
        elif attention_mask is None:
            attention_mask = _build_default_active_mask(
                B=b_x,
                S_tkg=s_tkg_x,
                q_heads=q_heads_local,
                device=sp_partials.device,
                dcp_active_owner=None,
            )
    if pos_ids is None:
        attention_mask = _maybe_transpose_mask_for_qk_swap(
            attention_mask=attention_mask,
            V_cache=V_cache,
            active_blocks_table=active_blocks_table,
            k_scale=k_scale,
            fp8_packed=fp8_packed,
        )
    megakernel_kwargs["pos_ids"] = pos_ids
    megakernel_kwargs["swa_start_pos_ids"] = swa_start_pos_ids
    megakernel_kwargs["attention_mask"] = attention_mask

    update_cache = bool(megakernel_kwargs.get("update_cache", False))
    lnc_degree = get_logical_nc_config()
    wrapped = wrap_nki(_torch_compatible_sp_a2a_attn_kernel)
    kernel_out = wrapped[lnc_degree](
        sp_partials,
        sp_residual=sp_residual,
        sp_gamma=sp_gamma,
        sp_eps=sp_eps,
        sp_tp_degree=sp_tp_degree,
        sp_replica_mesh=sp_replica_mesh,
        **megakernel_kwargs,
    )
    # The composer appends the updated residual stream; the K/V outputs are the
    # in-place-written caches under update_cache (redundant to return unless the
    # packed fallback asks for the alias outputs).
    sp_residual_out = kernel_out[-1]
    kernel_out = kernel_out[:-1]
    if update_cache:
        result = kernel_out if return_updated_kv else kernel_out[0]
    else:
        result = kernel_out
    return result, sp_residual_out
