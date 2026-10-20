# SPDX-License-Identifier: Apache-2.0
import math

import torch
from torch import Tensor

from nkilib.core.mlp.mlp import mlp as nkilib_mlp
from nkilib.core.mlp.mlp_parameters import BS_TILE_SIZE, TKG_BS_SEQLEN_THRESHOLD
from nkilib.core.utils.common_types import (
    ActFnType,
    ComputationMode,
    DtypeMode,
    NormType,
    QuantizationType,
)
from vllm_neuron import envs
from libtorch_neuronx_lite.compile.platform import get_platform_target
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.functional.rmsnorm_quant import rmsnorm_quant
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel

wrapped_mlp = wrap_nki(nkilib_mlp)

# nkilib kernel tiling constraints; some checks depend on the active LNC degree.
# from nkilib source code
# TODO: feature request - constants below needs to be exported by nki API
_NUM_HW_PSUM_BANKS = 8
_SRC_PROJ_INT_DIM_TILE_SIZE = 512
# This module wraps the kernel at ``wrapped_mlp[get_logical_nc_config()]``, which is also
# the core count nkilib shards over (get_verified_program_sharding_info returns it as
# num_shards whenever H-sharding is live: "H1 must be evenly divisible by num_shards").
# The shape checks below read it rather than assume 2, since at lnc=1 the sharding
# constraints are always satisfiable.
# The T bound below which _can_use_kernel applies a TKG constraint under AUTO --
# NOT nkilib's TKG/CTE split (TKG_BS_SEQLEN_THRESHOLD = 96); 96 would push T in
# (96, 128] onto the I-tiling check and reject large-I shapes that work today.
# DECODE takes TKG at any T, so it takes a TKG constraint regardless of this bound.
_TKG_H_SHARDING_T_LIMIT = 128
_MIN_H_FOR_I_SHARDING = 7168
_MIN_I_FOR_I_SHARDING = 1024
_MAX_T_FOR_I_SHARDING = 256

# A ROW-packed MX activation carries a per-row fp32 dequant scale in a trailing
# 4 bytes (one fp8 byte each), so the input is [T, H+4]. Not every MX caller packs
# (e.g. a bf16-in TKG path passes [T, H]); _can_use_kernel strips this tail only
# when present (H % 128 == 4).
_MX_PACKED_SCALE_BYTES = 4

# MX weight-packing tile geometry (shared MX kernel layout, NOT model-specific).
# The H_X4_* / MX_CONTIGUOUS swizzles tile the contraction dim into 512-element
# tiles of 128 partitions x 4 x4-packed fp8 bytes. The per-model MX weight loaders
# (e.g. qwen3_vl) import these so the packed nn.Parameter shapes and the loader
# swizzle share one definition.
_PMAX = 128  # partition dim (nl.tile_size.pmax); also nkilib's H0, so H1 = H // _PMAX
_Q_WIDTH = 4  # 4 float8_e4m3fn bytes per nl.float8_e4m3fn_x4 / uint32 element
_TILE_SIZE = _PMAX * _Q_WIDTH  # 512 -- the 512-element H/I tile of the MX layout

# Targets on which the fused static-FP8 decode MLP (in-kernel RMSNorm + static
# quant) has been validated on hardware.
# TODO: add trn3pre once it has a module run of its own. trn2 is listed on the
# strength of the decode work this change serves, not of a module run: the module
# test's three-way comparison is Trn3-only by construction, because Trn2's 240/448
# fp8 weight downscale is not mirrored in its references.
_FUSED_DECODE_NORM_TARGETS = ("trn2", "trn3")


def _target_supports_fused_decode_norm(target: str) -> bool:
    """Whether a platform-target string names a validated fused-decode target.

    Prefix-matched, not exact: ``get_platform_target()`` may return a revision-suffixed
    target (``trn3-rev2``). ``normalize_target_family()`` is deliberately NOT used --
    it folds ``trn3pre`` into ``trn3``, and ``trn3pre`` is excluded here.

    Example:
        >>> _target_supports_fused_decode_norm("trn3-rev2")
        True
        >>> _target_supports_fused_decode_norm("trn3pre")
        False
    """
    return any(
        target == validated or target.startswith(f"{validated}-")
        for validated in _FUSED_DECODE_NORM_TARGETS
    )


def _compute_fused_decode_norm_supported() -> bool:
    """Resolve the platform once, at import, outside any traced region.

    ``get_platform_target()`` instantiates ``torch.classes.neuron.Runtime()`` to
    query the runtime when no ``NEURON_PLATFORM_TARGET_OVERRIDE`` is set. That is
    a torchbind construction, so it must not sit in a ``torch.compile`` graph --
    hence the module-level constant below, mirroring ``_USE_ALLGATHER_FALLBACK``
    in ``functional/collectives/all_to_all.py``.

    Returns:
        True if this host's target has a validated fused decode kernel. False if
        the platform cannot be detected, which keeps the pre-quantizing path.
    """
    if envs.VLLM_NEURON_CPU_MODE:
        return False
    try:
        return _target_supports_fused_decode_norm(get_platform_target())
    except Exception:
        # No detectable Neuron platform; stay on the pre-quantizing path. Broad on
        # purpose: this runs at import of a module every model path imports, so
        # anything narrower turns a probe failure into an unimportable package.
        # Unlike _USE_ALLGATHER_FALLBACK the import stays at module scope: the module
        # attribute is the seam two tests patch to assert the probe is never reached
        # per call, and a function-local import removes that seam.
        return False


# Resolved at import, never per forward: see _compute_fused_decode_norm_supported.
_FUSED_DECODE_NORM_SUPPORTED = _compute_fused_decode_norm_supported()

# Quantization types whose CTE path folds a static input scale into the dequant.
# nkilib gates that on ``is_logical_quant_static()``, i.e. QuantizationType
# .is_logical_static() == (STATIC, STATIC_MX) -- both the basic and MX CTE
# kernels. Spelled as a tuple rather than calling ``is_logical_static()`` because
# a bound-method call is not Dynamo-traceable (see the note on _column_tiling).
_STATIC_INPUT_SCALE_QUANT_TYPES = (
    QuantizationType.STATIC,
    QuantizationType.STATIC_MX,
)


def _tkg_runs_shape(T: int, H: int) -> bool:
    """Whether nkilib's TKG kernel can run a ``[T, H]`` activation at the active grid.

    TKG picks its sharding *axis* by T. It shards on T when
    ``T >= BS_TILE_SIZE * lnc`` and ``T % lnc == 0``, which gives each core all of H,
    so no H constraint applies. Otherwise it shards on H, and ``H1 = H // 128`` must
    divide across the ``lnc`` cores -- ``H % 256 == 0`` at lnc=2, and at lnc=1 both
    arms hold for any 128-divisible H.

    One definition, used by both the decode-fusion decision and the eligibility gate:
    if they disagreed, a shape could be routed to TKG and then rejected by the gate,
    which is how a static-FP8 call ends up on the no-dequant torch fallback.

    Example:
        >>> _tkg_runs_shape(256, 8320), _tkg_runs_shape(129, 8320)
        (True, False)
    """
    if H % _PMAX != 0:
        # H1 = H // 128 would truncate, so no sharding axis can run it. Checked here and
        # not only in the gate: this predicate also decides whether to claim DECODE, and
        # the two must not disagree.
        return False
    lnc = get_logical_nc_config()
    if T >= BS_TILE_SIZE * lnc and T % lnc == 0:
        return True
    return (H // _PMAX) % lnc == 0


def _can_fuse_decode_norm(hidden: Tensor, quantization_type: QuantizationType) -> bool:
    """Whether ``mlp`` can fold the pre-MLP RMSNorm into the static-FP8 kernel.

    True only for static-FP8, on a target where the fused decode path is
    hardware-validated, **and** for a shape TKG can actually run: fusing the norm
    commits the call to TKG (it is what selects ``DECODE``), so claiming it for a
    shape TKG must reject would strand the call on the torch fallback, which applies
    no dequant. Declining instead sends it back to the pre-quantize route, which reaches
    CTE -- and runs any ``H % 128 == 0`` -- for ``T > _TKG_H_SHARDING_T_LIMIT``. For T in
    ``(TKG_BS_SEQLEN_THRESHOLD, _TKG_H_SHARDING_T_LIMIT]`` the pre-quantize fires but the
    gate still applies the TKG H-sharding test, so an odd ``H1`` there still reaches the
    fallback, now with an fp8 activation. That band is unchanged from before ``mlp`` owned
    the decision and is the false-reject gap :func:`_can_use_kernel` documents.

    ``mlp`` consults this to choose between folding the norm into the kernel and
    pre-quantizing the activation itself (see
    :func:`_needs_prequantized_static_input`); callers pass ``RMS_NORM`` and let
    ``mlp`` decide.

    Args:
        hidden: The activation ``mlp`` will dispatch, for its ``[T, H]`` shape.
        quantization_type: The quantization type the caller will pass to ``mlp``.

    Returns:
        True if ``mlp`` fuses the norm for this configuration, False if it must
        pre-normalize and pre-quantize instead.

    Example:
        >>> from nkilib.core.utils.common_types import QuantizationType
        >>> _can_fuse_decode_norm(torch.zeros(32, 256), QuantizationType.NONE)
        False
    """
    # Pass the activation, exactly as the eligibility gate does. Without it the device
    # defaults to "" and reads as non-CPU, so a CPU tensor outside CPU mode would claim
    # DECODE here and be rejected by the gate -- the disagreement this shares
    # _tkg_runs_shape to avoid.
    if not can_run_kernel(hidden):
        return False
    if quantization_type != QuantizationType.STATIC:
        return False
    T, H = hidden.shape
    if not _tkg_runs_shape(T, H):
        return False
    # Reads the import-time constant: this runs inside the traced graph, so it
    # must not probe the runtime here.
    return _FUSED_DECODE_NORM_SUPPORTED


def _needs_prequantized_static_input(
    hidden: Tensor,
    norm_type: NormType,
    quantization_type: QuantizationType,
) -> bool:
    """Whether ``mlp`` must normalize and quantize the activation before dispatch.

    True only for ``STATIC``, the one type :func:`rmsnorm_quant` can produce;
    ``STATIC_MX`` hits the same CTE hole but has no fused-quant path here, so it stays
    the caller's responsibility and :func:`_check_static_input_is_quantized` rejects it.

    This is the route a shape TKG cannot run falls back to, and it is the one the
    model layer used for large-batch decode before ``mlp`` owned the decision: quantize
    here, pass ``NO_NORM``, and let CTE run it -- CTE does not shard H, so it has no
    ``H % 256`` requirement. That holds above ``_TKG_H_SHARDING_T_LIMIT``; between
    nkilib's threshold and that bound the gate still takes a TKG constraint, so the
    quantized activation can still be refused there (the pre-existing false-reject gap).
    """
    if quantization_type != QuantizationType.STATIC:
        return False
    if norm_type != NormType.RMS_NORM:
        return False
    if hidden.shape[0] <= TKG_BS_SEQLEN_THRESHOLD:
        return False
    return not _can_fuse_decode_norm(hidden, quantization_type)


def mlp(
    hidden: Tensor,
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
    eps: float = 1e-6,
    ln_w: Tensor | None = None,
    skip_gate: bool = False,
    act_fn: ActFnType = ActFnType.SiLU,
    gate_b: Tensor | None = None,
    up_b: Tensor | None = None,
    down_b: Tensor | None = None,
    norm_b: Tensor | None = None,
    norm_type: NormType = NormType.NO_NORM,
    fused_add_tensor: Tensor | None = None,
    store_fused_add_result: bool = False,
    quantization_type: QuantizationType = QuantizationType.NONE,
    gate_w_scale: Tensor | None = None,
    up_w_scale: Tensor | None = None,
    down_w_scale: Tensor | None = None,
    gate_up_in_scale: Tensor | None = None,
    down_in_scale: Tensor | None = None,
    quant_clipping_bound: float = 0.0,
    output_dtype: torch.dtype | None = None,
    gate_clamp_upper_limit: float | None = None,
    gate_clamp_lower_limit: float | None = None,
    up_clamp_upper_limit: float | None = None,
    up_clamp_lower_limit: float | None = None,
    gate_up_w_layout=None,
    column_tiling: bool | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """MLP API that automatically selects between nkilib NKI kernel and PyTorch fallback.

    This function checks kernel constraints and dispatches to:
    - nkilib MLP kernel: Default option when running on Neuron
    - PyTorch implementation: When in CPU mode (VLLM_NEURON_CPU_MODE=1) or constraints are violated

    Args:
        hidden: Input hidden state tensor with shape [T, H] where T is total tokens.
        gate_w: Gate projection weights with shape [H, I].
        up_w: Up projection weights with shape [H, I].
        down_w: Down projection weights with shape [I, H].
        eps: Epsilon for numerical stability in normalization (default: 1e-6).
        ln_w: Normalization weights (gamma) with shape [1, H].
        skip_gate: If True, skip gate projection and apply activation to up projection.
        act_fn: Activation function type (SiLU, GELU, GELU_Tanh_Approx, Swish).
        gate_b: Optional gate projection bias with shape [1, I].
        up_b: Optional up projection bias with shape [1, I].
        down_b: Optional down projection bias with shape [1, H].
        norm_b: Optional normalization bias with shape [1, H]. Only for LayerNorm.
        norm_type: Type of normalization (NO_NORM, RMS_NORM, RMS_NORM_SKIP_GAMMA, LAYER_NORM).
        fused_add_tensor: Optional tensor for fused residual add before normalization.
        store_fused_add_result: If True and fused_add_tensor is provided, return fused add
            result as second output.
        quantization_type: Quantization type (NONE, STATIC, ROW, MX).
        gate_w_scale: FP8 dequantization scales for gate weights.
        up_w_scale: FP8 dequantization scales for up weights.
        down_w_scale: FP8 dequantization scales for down weights.
        gate_up_in_scale: FP8 dequantization scales for gate/up input (static quant).
        down_in_scale: FP8 dequantization scales for down input (static quant).
        quant_clipping_bound: Clipping bound for FP8 row quantization (default: 0.0).
        output_dtype: Output tensor data type. If None, uses hidden tensor's dtype.
        gate_clamp_upper_limit: Upper clamp bound for gate projection results.
        gate_clamp_lower_limit: Lower clamp bound for gate projection results.
        up_clamp_upper_limit: Upper clamp bound for up projection results.
        up_clamp_lower_limit: Lower clamp bound for up projection results.
        column_tiling: Column tiling override for the gate-up and down projections
            on the decode path. ``None`` keeps the default: on for non-MX dtypes,
            off for all MX dtypes. ``False`` turns tiling off regardless of dtype.
            ``True`` is a no-op (the default already enables tiling for non-MX and
            the kernel requires tiling off for MX, so ``True`` cannot force it on).

    Returns:
        Tensor with shape [T, H] normally, or tuple[Tensor, Tensor] when
        fused_add_tensor is provided and store_fused_add_result is True.

    Example:
        >>> import torch
        >>> from vllm_neuron.functional.mlp import mlp
        >>>
        >>> T, H, I = 32, 256, 512
        >>> hidden = torch.randn(T, H)
        >>> gate_w = torch.randn(H, I)
        >>> up_w = torch.randn(H, I)
        >>> down_w = torch.randn(I, H)
        >>> output = mlp(hidden, gate_w, up_w, down_w)
        >>> output.shape
        torch.Size([32, 256])
    """
    # Placing the pre-MLP norm is this function's decision, not the caller's:
    # where the kernel cannot fuse it, normalize and quantize here so both the
    # kernel and the torch fallback below see the same activation.
    if _needs_prequantized_static_input(hidden, norm_type, quantization_type):
        hidden = rmsnorm_quant(
            hidden,
            ln_w=ln_w,
            input_dequant_scale=gate_up_in_scale,
            eps=eps,
            quantization_type=QuantizationType.STATIC,
        )
        norm_type = NormType.NO_NORM
        ln_w = None

    T, H = hidden.shape

    # Fused static-FP8 RMSNorm is the decode contract; prefill is prequantized.
    # DECODE keeps the fused path at any T -- only claim it where the kernel is
    # validated and can run the shape, so an unvalidated target or a shape TKG must
    # reject takes nkilib's own T-based choice instead. Derived before the eligibility
    # gate, which validates whichever of the CTE/TKG constraint sets this mode selects.
    computation_mode = (
        ComputationMode.DECODE
        if norm_type == NormType.RMS_NORM
        and _can_fuse_decode_norm(hidden, quantization_type)
        else ComputationMode.AUTO
    )

    if _can_use_kernel(
        hidden, gate_w, gate_b, up_b, down_b, quantization_type, computation_mode
    ):
        # Reshape [T, H] -> [1, T, H] for kernel (kernel expects [B, S, H])
        hidden_3d = hidden.unsqueeze(0)

        # Provide default normalization weights if norm is requested but ln_w is None
        if norm_type != NormType.NO_NORM and ln_w is None:
            ln_w = torch.ones((1, H), dtype=hidden.dtype, device=hidden.device)

        # Reshape fused_add_tensor to 3D if provided
        fused_add_3d = None
        if fused_add_tensor is not None:
            fused_add_3d = fused_add_tensor.unsqueeze(0)

        # The MXFP hardware path (MX / STATIC_MX / ROW_MX) does not use column
        # tiling -- the kernel asserts ``use_tkg_*_column_tiling=False`` whenever
        # ``is_dtype_mx()`` (mlp_parameters: "MX quantization (mxfp) does not use
        # column tiling"). Compare against the enum members directly rather than
        # calling ``quantization_type.is_mx()``: a bound-method call is not
        # traceable by Dynamo ("Unexpected type in sourceless builder: method"),
        # whereas the membership check is. Covers MX and ROW_MX, not only STATIC_MX.
        _column_tiling = quantization_type not in (
            QuantizationType.MX,
            QuantizationType.STATIC_MX,
            QuantizationType.ROW_MX,
        )

        # The caller turns the tiling off for both projections. Keep the MX rule
        # above, because the kernel asserts on tiling for an MX dtype, so a
        # caller cannot turn the tiling on for an MX dtype.
        if column_tiling is not None:
            _column_tiling = _column_tiling and column_tiling

        _check_static_input_is_quantized(
            hidden, norm_type, quantization_type, computation_mode
        )

        # The MX prefill kernel has no norm stage, so it ignores norm_type silently.
        if (
            quantization_type == QuantizationType.MX
            and norm_type != NormType.NO_NORM
            and computation_mode != ComputationMode.DECODE
            and T > TKG_BS_SEQLEN_THRESHOLD
        ):
            raise ValueError(
                f"norm_type={norm_type.name} with MX and T={T} is larger than "
                f"{TKG_BS_SEQLEN_THRESHOLD}. This routes to the prefill kernel, "
                "which ignores norm_type. Normalize the activation first, then "
                "pass norm_type=NO_NORM."
            )

        result = wrapped_mlp[get_logical_nc_config()](
            hidden_tensor=hidden_3d,
            gate_proj_weights_tensor=gate_w,
            up_proj_weights_tensor=up_w,
            down_proj_weights_tensor=down_w,
            normalization_weights_tensor=ln_w,
            gate_proj_bias_tensor=gate_b,
            up_proj_bias_tensor=up_b,
            down_proj_bias_tensor=down_b,
            normalization_bias_tensor=norm_b,
            fused_add_tensor=fused_add_3d,
            store_fused_add_result=store_fused_add_result,
            activation_fn=act_fn,
            normalization_type=norm_type,
            quantization_type=quantization_type,
            gate_w_scale=gate_w_scale,
            up_w_scale=up_w_scale,
            down_w_scale=down_w_scale,
            gate_up_in_scale=gate_up_in_scale,
            down_in_scale=down_in_scale,
            quant_clipping_bound=quant_clipping_bound,
            output_dtype=output_dtype,
            store_output_in_sbuf=False,
            eps=eps,
            skip_gate_proj=skip_gate,
            use_tkg_gate_up_proj_column_tiling=_column_tiling,
            use_tkg_down_proj_column_tiling=_column_tiling,
            use_tkg_down_proj_optimized_layout=False,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            mode=computation_mode,
            force_cte_mode=False,
            dtype_mode=DtypeMode.AUTO,
            **(
                {"gate_up_w_layout": gate_up_w_layout}
                if gate_up_w_layout is not None
                else {}
            ),
        )

        # Handle return value: nkilib returns list/tuple when store_fused_add_result
        if isinstance(result, (tuple, list)):
            if store_fused_add_result and len(result) == 2:
                return result[0].squeeze(0), result[1].squeeze(0)
            return result[0].squeeze(0)
        return result.squeeze(0)
    else:
        return _torch_mlp_impl(
            hidden=hidden,
            ln_w=ln_w,
            gate_w=gate_w,
            up_w=up_w,
            down_w=down_w,
            eps=eps,
            skip_gate=skip_gate,
            act_fn=act_fn,
            gate_b=gate_b,
            up_b=up_b,
            down_b=down_b,
            norm_b=norm_b,
            norm_type=norm_type,
            fused_add_tensor=fused_add_tensor,
            store_fused_add_result=store_fused_add_result,
        )


def _check_static_input_is_quantized(
    hidden: Tensor,
    norm_type: NormType,
    quantization_type: QuantizationType,
    computation_mode: ComputationMode,
) -> None:
    """Reject a static-quant call that would silently corrupt its NKI-kernel output.

    The CTE path selects its quantized projection on the *weights*
    (``quant_params.is_quant()``), never on the activation's dtype, and folds
    ``gate_up_in_scale`` into the dequant unconditionally. A statically quantized call
    that reaches CTE without a quantized activation therefore gets a scale applied to a
    value that was never quantized: valid shapes, valid dtypes, wrong numbers, no assert.
    nkilib gates the fold on ``is_logical_quant_static()``, so the MX CTE kernel has the
    same hole.

    In practice only ``STATIC_MX`` reaches the raise: ``mlp`` runs
    :func:`_needs_prequantized_static_input` first and rewrites ``norm_type`` to
    ``NO_NORM`` for ``STATIC``, which is the condition below. The check still covers both,
    because what makes it safe for STATIC is a caller ordering rather than anything the
    type system holds.

    Keyed on ``norm_type``, never on ``hidden.dtype``: ``NO_NORM`` is the caller declaring
    it already normalized and quantized (the prefill contract), while a dtype sniff would
    make the shipping prefill path depend on how the fp8 dtype surfaces through
    ``rmsnorm_quant`` and the all-gather, which the caller does not control.

    Guards the NKI-kernel path only, since ``mlp`` calls it inside the
    ``_can_use_kernel`` branch. The same combination reaching the torch fallback is wrong
    by a different mechanism -- ``_torch_mlp_impl`` applies no dequant at all -- and is
    left unguarded rather than hoisted, because hoisting changes the failure mode of a
    path no current measurement covers. That no-dequant fallback is also why
    :func:`_can_fuse_decode_norm` declines a shape TKG cannot run instead of claiming
    ``DECODE`` for it: declining routes the call to pre-quantize, which reaches CTE -- where
    the activation *is* quantized -- above ``_TKG_H_SHARDING_T_LIMIT``. In
    ``(TKG_BS_SEQLEN_THRESHOLD, _TKG_H_SHARDING_T_LIMIT]`` an odd ``H1`` still reaches this
    fallback, now with an fp8 activation, byte-identically to before ``mlp`` owned the
    decision; that band is the false-reject gap :func:`_can_use_kernel` documents.

    Raises:
        ValueError: If the call would reach CTE with an unquantized activation.
    """
    if quantization_type not in _STATIC_INPUT_SCALE_QUANT_TYPES:
        return
    if norm_type != NormType.RMS_NORM:
        return
    if computation_mode == ComputationMode.DECODE:
        return
    if hidden.shape[0] <= TKG_BS_SEQLEN_THRESHOLD:
        return
    raise ValueError(
        f"{quantization_type.name} quantization with norm_type={norm_type.name} "
        f"and T={hidden.shape[0]} > {TKG_BS_SEQLEN_THRESHOLD} routes to the CTE "
        "kernel, which fuses neither the norm nor the quant, so it would apply "
        "gate_up_in_scale to an activation that was never quantized and silently "
        "produce wrong results. Quantize the activation first (see rmsnorm_quant) "
        "and pass norm_type=NO_NORM."
    )


def _can_use_kernel(
    hidden: Tensor,
    gate_w: Tensor,
    gate_b: Tensor | None = None,
    up_b: Tensor | None = None,
    down_b: Tensor | None = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    computation_mode: ComputationMode = ComputationMode.AUTO,
) -> bool:
    """Check if the nkilib NKI kernel can be used for the given dimensions.

    Returns False when ``VLLM_NEURON_CPU_MODE`` is enabled, the hidden tensor is on
    CPU, hidden dimension is not aligned to 128, or the model dimensions
    violate nkilib tiling constraints for the active LNC grid.

    Which constraint applies follows the path nkilib will take, so it must
    track ``is_mlp_tkg``: TKG whenever ``computation_mode`` is ``DECODE`` (at *any*
    T), or under ``AUTO`` for small T; CTE otherwise.
    - TKG: whichever constraint its **sharding axis** carries. TKG shards on T when
      ``T >= BS_TILE_SIZE * lnc`` and ``T % lnc == 0``, giving each core all of H, so
      no H constraint applies. Otherwise it shards on H, and ``H1 = H // 128`` must
      divide across the ``lnc`` cores, i.e. ``H % 256 == 0`` at lnc=2. Keying the
      H constraint on ``DECODE`` alone would reject a large-batch decode that TKG
      runs correctly on the T axis.
    - CTE: the I-tiling constraint. Intermediate dim tiles must fit in PSUM banks,
      ``ceil(effective_I / 512) <= 8`` where ``effective_I`` is ``I // lnc`` when
      I-sharding is active, otherwise ``I``.

    ``lnc`` above is :func:`get_logical_nc_config`. At lnc=1 the sharding constraints
    are always satisfiable, leaving only ``H % 128`` and the unsharded
    ``ceil(I / 512) <= 8`` bound.

    ``DECODE`` never reaches the H-sharding rejection: :func:`_can_fuse_decode_norm`
    applies the same :func:`_tkg_runs_shape` test before claiming ``DECODE``, so a shape
    TKG must reject resolves to ``AUTO`` and :func:`_needs_prequantized_static_input`
    routes it to pre-quantize. That matters because a ``False`` here selects
    ``_torch_mlp_impl``, which applies no dequant. Above
    ``_TKG_H_SHARDING_T_LIMIT`` the re-routed call lands on CTE and runs; at or below it
    the gate takes a TKG constraint again, so the pre-quantized call can still be
    refused -- the first residual gap below, unchanged by this routing.

    Two residual gaps, both latent because every shipping hidden size is a multiple
    of 256:
    - False-reject: ``AUTO`` splits at ``TKG_BS_SEQLEN_THRESHOLD`` (96) while this
      uses ``_TKG_H_SHARDING_T_LIMIT`` (128), so for T in (96, 128] the H-sharding
      constraint is applied to a T that routes to CTE, and a shape with
      ``H % 128 == 0`` but not ``% 256`` (e.g. H=8320) is rejected where CTE would
      have run it. Lowering the bound to 96 trades that for a worse gap: T in
      (96, 128] would take the I-tiling check, which rejects large-I shapes that work
      today (e.g. H=8192, I=28672), because I-sharding only halves I when H >= 7168,
      I >= 1024 and T <= 256.
    - Silent-fallback, **pre-existing and unchanged by the mode plumbing**: a static
      call that lands on ``False`` here reaches the no-dequant fallback. That is true
      for the rejections above this one (CPU mode, ``H % 128 != 0``) and for a static
      call at ``T <= 128`` with an odd ``H1``, whether its activation is raw bf16 or
      already pre-quantized. It is the gap :func:`_check_static_input_is_quantized`
      records; closing it needs a config-time rejection, not a change here.

    Native-MX (``QuantizationType.MX``) is a special case, and it is path-neutral
    here (CTE *and* TKG share this gate; nkilib's ``is_mlp_tkg`` does the CTE/TKG
    split internally by ``B*S`` — this function never picks the path). ``gate_w`` is
    a pre-tiled swizzled weight whose ``shape[1]`` is NOT ``I``, and the activation
    may be EITHER ROW-packed FP8 ``[T, H+4]`` (CTE prefill: per-row dequant scale in
    a trailing 4 bytes) OR plain ``[T, H]`` (e.g. a TKG path that quantizes a bf16
    input on-device). Both break the plain 2D checks above, so for MX we:
      * strip a trailing +4 from H *only when present* (``H % 128 == 4``) before the
        128-alignment check, so a packed ``H+4`` and an unpacked ``H`` both validate;
      * bypass the ``gate_w.shape[1]``-derived PSUM-bank check entirely — the MX
        kernel pre-tiles the intermediate dim and validates its own layout/PSUM
        budget at trace time, failing loud if a config doesn't fit.

    Note: The nkilib kernel supports bfloat16 and float8 (float8_e4m3) dtypes.
    Other dtypes are not supported and will cause a validation error from nkilib.
    Callers are responsible for providing tensors with a supported dtype.

    Args:
        hidden: Input hidden state tensor with shape [T, H] (or [T, H+4] for MX).
        gate_w: Gate projection weights with shape [H, I] (or 6D for MX).
        gate_b: Optional gate projection bias with shape [1, I].
        up_b: Optional up projection bias with shape [1, I].
        down_b: Optional down projection bias with shape [1, H].
        quantization_type: Quantization type; ``MX`` takes the packed-input path.
        computation_mode: The ``mode`` ``mlp`` will pass the kernel. ``DECODE`` routes
            to TKG at any T, so it selects a TKG constraint above.

    Returns:
        True if the NKI kernel can be used, False otherwise.

    Example:
        >>> hidden = torch.randn(32, 256, dtype=torch.bfloat16)
        >>> gate_w = torch.randn(256, 512, dtype=torch.bfloat16)
        >>> _can_use_kernel(hidden, gate_w)
        True
    """
    if not can_run_kernel(hidden):
        return False
    T, H = hidden.shape
    lnc_degree = get_logical_nc_config()

    # Native-MX: strip a row-packed +4 tail when present (see _MX_PACKED_SCALE_BYTES),
    # then require 128-alignment. A real MX weight always has H % 128 == 0 (the
    # H_X4_INNERMOST swizzle requires it), so the H % 128 == 4 arm only ever fires on
    # a genuine ROW-packed [T, H+4] activation -- it can't silently eat 4 real
    # elements of an unpacked input. The kernel does its own I-tiling/PSUM-bank
    # validation, so skip the 2D-weight-derived bank check below.
    if quantization_type == QuantizationType.MX:
        H_eff = H - _MX_PACKED_SCALE_BYTES if H % 128 == _MX_PACKED_SCALE_BYTES else H
        return H_eff % 128 == 0

    if H % 128 != 0:
        return False

    # DECODE cannot be keyed on T: nkilib's is_mlp_tkg returns True for it at any T,
    # so a TKG constraint applies even above the AUTO bound -- but which one depends on
    # the axis TKG shards, which it picks by T. Same predicate the DECODE claim used, so
    # this arm agrees with it by construction; the checks ABOVE (CPU, H % 128) are the
    # ones that can still reject a DECODE call.
    # Returning here also skips the CTE I-tiling check below, deliberately: TKG tiles the
    # intermediate dim itself and fails loud if a config does not fit, so I and the biases
    # are not constraints on this arm.
    if computation_mode == ComputationMode.DECODE or T <= _TKG_H_SHARDING_T_LIMIT:
        return _tkg_runs_shape(T, H)

    # CTE: PSUM bank constraint on intermediate dimension
    I = gate_w.shape[1]
    has_bias = gate_b is not None or up_b is not None or down_b is not None
    can_shard_on_i = (
        H >= _MIN_H_FOR_I_SHARDING
        and I >= _MIN_I_FOR_I_SHARDING
        and T <= _MAX_T_FOR_I_SHARDING
        and not has_bias
    )
    effective_i = I // lnc_degree if can_shard_on_i else I
    return math.ceil(effective_i / _SRC_PROJ_INT_DIM_TILE_SIZE) <= _NUM_HW_PSUM_BANKS


def _get_activation_fn(act_fn: ActFnType):
    """Get the corresponding torch activation function.

    Args:
        act_fn: Activation function type from nkilib's ActFnType enum.

    Returns:
        A callable torch activation function.

    Raises:
        ValueError: If the activation function type is unsupported.

    Example:
        >>> fn = _get_activation_fn(ActFnType.SiLU)
        >>> fn(torch.tensor([1.0]))
        tensor([0.7311])
    """
    if act_fn == ActFnType.SiLU or act_fn == ActFnType.Swish:
        return torch.nn.functional.silu
    elif act_fn == ActFnType.GELU:
        return torch.nn.functional.gelu
    elif act_fn == ActFnType.GELU_Tanh_Approx:
        return lambda x: torch.nn.functional.gelu(x, approximate="tanh")
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


def _torch_rms_norm(x: Tensor, weight: Tensor, eps: float, skip_gamma: bool) -> Tensor:
    """RMSNorm implementation in PyTorch."""
    # x: [T, H], weight: [1, H]
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    if not skip_gamma:
        x_normed = x_normed * weight
    return x_normed


def _torch_layer_norm(
    x: Tensor, weight: Tensor | None, bias: Tensor | None, eps: float
) -> Tensor:
    """LayerNorm implementation in PyTorch."""
    # x: [T, H], weight: [1, H], bias: [1, H] or None
    mean = x.mean(dim=-1, keepdim=True)
    variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    x_normed = (x - mean) * torch.rsqrt(variance + eps)
    if weight is not None:
        x_normed = x_normed * weight
    if bias is not None:
        x_normed = x_normed + bias
    return x_normed


def _torch_mlp_impl(
    hidden: Tensor,
    ln_w: Tensor | None,
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
    eps: float,
    skip_gate: bool,
    act_fn: ActFnType,
    gate_b: Tensor | None,
    up_b: Tensor | None,
    down_b: Tensor | None,
    norm_b: Tensor | None,
    norm_type: NormType | None,
    fused_add_tensor: Tensor | None = None,
    store_fused_add_result: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """PyTorch implementation of MLP with optional normalization and fused residual add.

    Computation flow:
    0. Optional fused residual add
    1. Optional normalization (RMSNorm or LayerNorm)
    2. Gate projection: x @ gate_w + gate_b (if not skip_gate)
    3. Up projection: x @ up_w + up_b
    4. Activation: act_fn(gate) * up  OR  act_fn(up) if skip_gate
    5. Down projection: result @ down_w + down_b

    Args:
        hidden: Input hidden state tensor with shape [T, H].
        ln_w: Normalization weights (gamma) with shape [1, H].
        gate_w: Gate projection weights with shape [H, I].
        up_w: Up projection weights with shape [H, I].
        down_w: Down projection weights with shape [I, H].
        eps: Epsilon for numerical stability.
        skip_gate: If True, skip gate projection.
        act_fn: Activation function type.
        gate_b: Optional gate projection bias.
        up_b: Optional up projection bias.
        down_b: Optional down projection bias.
        norm_b: Optional normalization bias.
        norm_type: Type of normalization.
        fused_add_tensor: Optional tensor for fused residual add.
        store_fused_add_result: If True, return fused add result as second output.

    Returns:
        Tensor or tuple[Tensor, Tensor]: MLP output, optionally with fused add result.

    Example:
        >>> hidden = torch.randn(32, 256)
        >>> gate_w = torch.randn(256, 512)
        >>> up_w = torch.randn(256, 512)
        >>> down_w = torch.randn(512, 256)
        >>> output = _torch_mlp_impl(hidden, None, gate_w, up_w, down_w, 1e-6,
        ...     False, ActFnType.SiLU, None, None, None, None, NormType.NO_NORM)
        >>> output.shape
        torch.Size([32, 256])
    """

    x = hidden

    # Step 0: Fused residual add
    fused_add_output = None
    if fused_add_tensor is not None:
        x = x + fused_add_tensor
        if store_fused_add_result:
            fused_add_output = x.clone()

    # Step 1: Apply normalization
    if norm_type == NormType.RMS_NORM:
        x = _torch_rms_norm(x, ln_w, eps, skip_gamma=False)
    elif norm_type == NormType.RMS_NORM_SKIP_GAMMA:
        x = _torch_rms_norm(x, ln_w, eps, skip_gamma=True)
    elif norm_type == NormType.LAYER_NORM:
        x = _torch_layer_norm(x, ln_w, norm_b, eps)
    # NormType.NO_NORM: skip normalization

    # Step 2 & 3: Gate and Up projections
    if not skip_gate:
        # Gate projection
        gate = x @ gate_w
        if gate_b is not None:
            gate = gate + gate_b

    # Up projection
    up = x @ up_w
    if up_b is not None:
        up = up + up_b

    # Step 4: Apply activation
    activation_fn = _get_activation_fn(act_fn)

    if skip_gate:
        # When skip_gate is True, apply activation directly to up projection
        activated = activation_fn(up)
    else:
        # Standard GLU-style: act(gate) * up
        activated = activation_fn(gate) * up

    # Step 5: Down projection
    output = activated @ down_w
    if down_b is not None:
        output = output + down_b

    if fused_add_output is not None:
        return output, fused_add_output
    return output
