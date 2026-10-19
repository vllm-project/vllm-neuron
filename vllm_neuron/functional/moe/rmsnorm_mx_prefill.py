# SPDX-License-Identifier: Apache-2.0
"""Fused RMSNorm + MX-quantize + Router top-K for MoE prefill (token-major [T, H]).

Wraps the nkilib ``rmsnorm_mx_prefill`` kernel, which fuses, in one pass over the
prefill tokens:

  1. RMSNorm of the hidden states (mean over ``unpadded_hidden_size``),
  2. MX (microscaling fp8) quantization of the normed activations, and
  3. the MoE router (linear -> top-K -> activation -> scatter).

It replaces the standalone torch ``GptOssRMSNorm`` + torch ``NF.router`` at the head
of the gpt-oss mxfp4 prefill MoE. The fused kernel emits a PACKED fp8 tensor
``[T, H + scale_region]`` (``[hidden_quant | MX scales]``) that feeds the fp8-hidden
path of ``bwmm_shard_on_block_mx`` directly (the two are a co-designed producer/consumer
pair), plus the dense ``[T, E]`` expert affinities the blockwise mapping consumes.

Layout contract (verified against the kernel source — see
``doc/.../fuse-rmsnorm-router-prefill-plan.md``):
  - ``hidden_states``: natural H order by default, or the model's 4-way-interleaved H
    order with ``hidden_interleaved=True`` (the kernel de-interleaves during its load, so
    gpt-oss no longer needs a per-forward ``unshuffle_hidden_dim_last``). Either way the
    caller must NOT pre-swizzle -- the kernel applies its internal swizzle in the transpose.
  - ``gamma``: natural H order (indexes hidden directly), independent of
    ``hidden_interleaved``.
  - ``router_weights``: PRE-PERMUTED into the kernel's swizzle-H order
    (``swizzle_h_index``). The router matmul runs against the swizzle-transposed
    activations, so row h of the natural weight must sit at swizzle slot h_swz.

When the kernel cannot run (CPU / non-Trn3 / H not a multiple of 512 / E or top_k out
of range) this falls back to a torch RMSNorm + router. The fallback serves non-kernel
callers and this module's functional tests; it is NOT model-accurate for the gpt-oss
mxfp4 path (see ``_torch_impl``).
"""

from typing import Optional, Tuple

import torch
import nki.language as nl

from nkilib.core.rmsnorm.rmsnorm_mx_prefill import (
    rmsnorm_mx_prefill as _rmsnorm_mx_prefill_kernel,
)
from nkilib.core.utils.common_types import RouterActFnType

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel


# MX quant requires H to be a multiple of the 512-wide MX block.
_MX_BLOCK = 512
# Router gemm moving free-dim cap (E <= 512) and top-K cap from the kernel.
_E_MAX = 512
_TOP_K_MAX = 8
# pack_affinities row layout (see nkilib rmsnorm_mx_prefill kernel): each fp8 row is
# ``[hidden(H) | MX scale(scale_region) | affinities(E bf16 == 2E fp8 cols) | pad]``.
# The affinity region is the DENSE [T, E] affinities in plain expert order, stored as
# bf16 bit-reinterpreted into fp8 column pairs. _BF16_AS_FP8 = 2 fp8 cols per bf16 value.
_BF16_AS_FP8 = 2

# MX-quant constants (mirror the golden decode_packed_output / reference_mx_dequant layout):
# one MX scale covers 32 contiguous hidden values; the fp8 e4m3 max exponent is 7; scales are
# packed in the HW-quadrant folded layout (4 h512 tiles share a 128-wide scale block).
_MX_QUANT_BLOCK = 32  # hidden values per MX scale
_FP8_E4M3_MAX = 448.0
_FP8_E4M3_MAX_EXP = 7
_SCALE_BLOCK = 128  # scale columns per packed 128-block
_Q_WIDTH_MX = 4
_QUADRANT = 32
_SCALES_PER_BLOCK = 4  # h512 tiles folded per 128-wide scale block (pack_scales=True)


def _mx_scale_columns(hidden_size: int, device: torch.device) -> torch.Tensor:
    """Map each 32-wide MX block index -> its column in the packed HW-quadrant scale region.

    Exact inverse of the golden ``decode_packed_output`` scale addressing: block ``b`` spans
    hidden ``[32*b, 32*b+32)``; with ``h512 = b // 16`` and ``blk = b % 16`` the folded column is
    ``(h512 // 4) * 128 + (blk // 4) * 32 + (h512 % 4) * 4 + (blk % 4)``. Returned as a 1-D int64
    index of length ``H // 32`` so the packer can scatter one uint8 exponent per block per token.

    Build the ``arange`` directly on ``device``, not on CPU + ``.to(device)``. A host-constant
    lifted via ``.to()`` keeps its ``neuron`` device tag through the Neuron device_rewriter, while
    the in-graph ``torch.zeros`` target is rewritten to ``xla``; the scatter then fails with a
    self/index device mismatch. An in-graph index stays on the scatter target's rewritten device.
    """
    num_blocks = hidden_size // _MX_QUANT_BLOCK
    blocks_per_h512 = _MX_BLOCK // _MX_QUANT_BLOCK  # 16
    b = torch.arange(num_blocks, dtype=torch.int64, device=device)
    h512 = b // blocks_per_h512
    blk = b % blocks_per_h512
    return (
        (h512 // _SCALES_PER_BLOCK) * _SCALE_BLOCK
        + (blk // _Q_WIDTH_MX) * _QUADRANT
        + (h512 % _SCALES_PER_BLOCK) * _Q_WIDTH_MX
        + (blk % _Q_WIDTH_MX)
    )


def _torch_pack_mx_prefill(
    norm: torch.Tensor,
    expert_affinities: torch.Tensor,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Simulate the kernel's ``pack_affinities=True`` packed row in pure torch.

    Builds the ``[T, row_region]`` fp8 row VIEWED AS bf16 ``[T, row_region // 2]`` that the kernel
    emits, so ``slice_affinities_from_packed`` and the downstream ``moe_cte`` packed-tail read
    consume the fallback output identically to the kernel output. Layout (see module docstring):
    ``[hidden(H) fp8 | MX scale(scale_region) uint8-as-fp8 | affinities(E) bf16-as-fp8 | pad]``.

    Faithful, NOT performant -- a torch reference for when the NKI kernel path is disabled. It
    mirrors ``decode_packed_output`` (hidden natural-H fp8 + folded HW-quadrant uint8 scales) and
    the affinity tail (dense ``[T, E]`` bf16 reinterpreted into fp8 column pairs).

    Args:
        norm: ``[T, H]`` fp32 normalized activations (post-RMSNorm, pre-quant).
        expert_affinities: ``[T, E]`` dense router affinities (zero outside top-k).
        compute_dtype: the kernel's quant-feed dtype (router_weights.dtype); ``norm`` is rounded
            through it before quantization, matching ``reference_mx_dequant``.

    Returns:
        ``[T, row_region // 2]`` bf16-viewed packed row.
    """
    # Reinterprets go through _bitcast (NKI DMA reinterpret), NOT torch.view: a plain fp8/uint8
    # <-> bf16 .view has no FX->HLO lowering on Neuron, while _bitcast does. Each region gets one
    # _bitcast to bf16, then the regions concatenate in bf16 -- never as fp8.
    from vllm_neuron.functional.moe.permute_routed_tokens import _bitcast

    T, H = norm.shape
    E = expert_affinities.shape[1]

    # Round through the kernel's compute dtype, then MX-quantize per 32-wide block.
    x = norm.to(compute_dtype).to(torch.float32)
    blocks = x.reshape(T, H // _MX_QUANT_BLOCK, _MX_QUANT_BLOCK)
    # Biased fp8-block exponent = max IEEE exponent over the 32 values, minus e4m3 max_exp.
    # The IEEE biased exponent field of a value v is floor(log2(|v|)) + 127. floor(log2) is
    # monotonic, so the block field is floor(log2(amax|block|)) + 127. Use log2, not torch.frexp:
    # frexp has no FX->HLO lowering and segfaults neuronx-cc (device-confirmed). For the normal
    # values the MX norm produces, log2 is bit-identical to the frexp/bit-extract golden. Clamp
    # amax to a tiny floor so an all-zero block maps to field 0.
    block_amax = blocks.abs().amax(dim=2, keepdim=True)  # [T, n_blocks, 1]
    block_max_exp = torch.floor(torch.log2(torch.clamp(block_amax, min=1e-30))) + 127.0
    scale_uint8 = torch.clamp(block_max_exp - _FP8_E4M3_MAX_EXP, 0, 255).to(torch.int64)
    factor = torch.pow(2.0, (scale_uint8.to(torch.float32) - 127.0))
    factor = torch.where(factor == 0, torch.ones_like(factor), factor)
    hidden_fp8 = (
        torch.clamp(blocks / factor, -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .reshape(T, H)
    )  # [T, H] fp8, natural-H order

    # Scale region: one exponent BYTE per 32-block, at its folded HW-quadrant column. Scatter the
    # exponents into a [T, scale_region] int32 (one value in [0,255] per column). The bf16-pack
    # note below assembles these bytes into int32 words and reinterprets to bf16, so no fp8 op runs.
    num_h512 = H // _MX_BLOCK
    n_packed = (num_h512 + _SCALES_PER_BLOCK - 1) // _SCALES_PER_BLOCK
    scale_region = (
        n_packed * _SCALE_BLOCK
    )  # fp8 columns of scale (multiple of 128 -> mult of 4)
    scale_bytes = torch.zeros(T, scale_region, dtype=torch.int32, device=norm.device)
    cols = _mx_scale_columns(
        H, norm.device
    )  # [n_blocks], built on norm.device in-graph
    scale_bytes = scale_bytes.index_copy(
        1, cols, scale_uint8.squeeze(-1).to(torch.int32)
    )  # out-of-place: in-place index_copy_ on a traced tensor is unsafe on XLA/Neuron

    # Pack the scale region into bf16 WITHOUT any fp8 op. Reason: _bitcast emits one nki e4m3
    # flavor, and a later fp8 op (slice/gather) on that result makes neuronx-cc mix its two
    # mutually-exclusive e4m3 types -> NCC_EOCP001 (device-confirmed). So combine 4 exponent bytes
    # into one int32 word (byte b of word w = fp8 column 4w+b), then bitcast int32 -> bf16 (2 bf16
    # per word), giving scale_region//2 bf16 columns whose byte layout matches the fp8 row.
    sb = scale_bytes.reshape(T, scale_region // 4, 4)
    scale_words = (
        sb[..., 0] | (sb[..., 1] << 8) | (sb[..., 2] << 16) | (sb[..., 3] << 24)
    )  # [T, scale_region//4] int32
    scale_bf16 = _bitcast(scale_words, torch.bfloat16)  # [T, scale_region // 2]

    # hidden: OCP fp8 -> bf16 (single-type reinterpret). affinities: already bf16.
    hidden_bf16 = _bitcast(hidden_fp8, torch.bfloat16)  # [T, H // 2]
    aff_bf16 = expert_affinities.to(torch.bfloat16)  # [T, E] (== 2E fp8 cols)

    # Pad the fp8 row to a multiple of 4 columns (kernel's fp32-reinterpret transpose requires it);
    # in bf16 that pad is (pad_cols // 2) columns.
    row_region = H + scale_region + _BF16_AS_FP8 * E
    pad_cols = (-row_region) % 4
    parts = [hidden_bf16, scale_bf16, aff_bf16]
    if pad_cols:
        parts.append(
            torch.zeros(T, pad_cols // 2, dtype=torch.bfloat16, device=norm.device)
        )
    return torch.cat(parts, dim=1)  # [T, row_region // 2] bf16-viewed packed row


def slice_affinities_from_packed(
    packed_bf16: torch.Tensor, hidden_size: int, num_experts: int
) -> torch.Tensor:
    """Slice the dense ``[T, E]`` bf16 affinities from a bf16-viewed pack_affinities row.

    On ``pack_affinities=True`` the kernel returns the packed row already VIEWED AS bf16, so
    the affinity tail is plain bf16 and recovering it is a pure column slice -- NO bitcast, NO
    extra collective. The affinity region rides inside ``moe_hidden``, so an ``all_gather``
    of ``moe_hidden`` alone carries it (per-row); this replaces the separate affinity
    scatter + ``[T, E]`` all-gather. Column order is plain expert order, so ``(result != 0)``
    is the correct top-k routing mask for ``build_blockwise_mapping``.

    The packed row is ``[hidden(H) | scale(scale_region) | affinities(E) | pad]``. In the
    fp8 view the affinity region starts at fp8 column ``H + scale_region``; the bf16 view
    halves every fp8 column count, so it starts at bf16 column ``(H + scale_region) // 2``.

    Args:
        packed_bf16: ``[T, row_region // 2]`` bf16-viewed packed rows (the wrapper's first
            return value on ``pack_affinities=True``; post all-gather is fine).
        hidden_size: ``H`` (padded), to locate the scale/affinity regions.
        num_experts: ``E`` (dense/global expert count), the affinity region width.

    Returns:
        ``[T, E]`` bf16 dense affinities (zero outside each token's top-k).

    Example:
        >>> moe_hidden, _expert_index = rmsnorm_mx_prefill(..., pack_affinities=True)
        >>> gathered = tp_group.all_gather(moe_hidden, dim=0)   # bf16, affinities ride along
        >>> aff = slice_affinities_from_packed(gathered, hidden_size=3072, num_experts=128)
    """
    num_h512 = hidden_size // _MX_BLOCK
    n_packed = (num_h512 + 3) // 4  # pack_scales=True: 4 H512 tiles per 128-block
    scale_region = n_packed * 128
    # fp8-column affin offset -> bf16-column offset (bf16 view halves the fp8 column count).
    affin_off_bf16 = (hidden_size + scale_region) // _BF16_AS_FP8
    return packed_bf16[:, affin_off_bf16 : affin_off_bf16 + num_experts]


def rmsnorm_mx_prefill(
    hidden_states: torch.Tensor,
    gamma: torch.Tensor,
    router_weights: torch.Tensor,
    router_bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    top_k: int = 1,
    router_act_fn: RouterActFnType = RouterActFnType.SOFTMAX,
    unpadded_hidden_size: Optional[int] = None,
    pack_affinities: bool = False,
    hidden_interleaved: bool = False,
    residual: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    """Fused RMSNorm + MX-quantize + Router top-K for MoE prefill.

    Args:
        hidden_states: ``[B, S, H]`` or ``[T, H]`` bf16 input in PLAIN natural H order
            (the kernel swizzles internally — do not pre-permute). H must be a multiple
            of 512 for the kernel path.
        gamma: ``[1, H]`` or ``[H]`` RMSNorm weights in natural H order.
        router_weights: ``[H, E]`` router weights PRE-PERMUTED into the kernel's
            swizzle-H order. Pass ``.T`` of the stored ``[E, H]`` only after applying the
            swizzle permute in the weight loader.
        router_bias: ``[1, E]`` or ``[E]`` optional router bias.
        eps: RMSNorm epsilon.
        top_k: experts selected per token (<= 8).
        router_act_fn: SOFTMAX or SIGMOID (applied to the top-K logits).
        unpadded_hidden_size: actual (unpadded) hidden size for the RMS mean denominator.
            The sum-of-squares is over the full (padded) H; only the divisor uses this.
            Defaults to H.
        pack_affinities: when True (Option B), the dense ``[T, E]`` affinities are folded
            into each packed row (``[hidden | scale | affinities | pad]``) so the downstream
            ``bwmm_shard_on_block_mx`` reads them in-kernel — no separate affinity tensor
            and no ``expert_affinities_masked`` gather. On this path the kernel returns
            ``out_packed`` VIEWED AS bf16 (the affinity dtype), so the caller reads affinities
            directly with no bitcast; ``moe_cte`` re-views it to fp8 for hidden+scale. The
            second return value is the ``[T, top_k]`` ``expert_index``. When False (Option A,
            default) the second return value is the dense ``[T, E]`` affinities.
        hidden_interleaved: when True, ``hidden_states`` (and ``residual``) arrive in the
            model's 4-way interleaved H order (``stored[q*(H/4) + p] == natural[4*p + q]``) and
            the kernel de-interleaves during its load. Lets gpt-oss drop the per-forward
            ``unshuffle_hidden_dim_last``. ``gamma`` stays natural, and the packed output stays
            natural H, so nothing downstream changes.
        residual: optional ``[B, S, H]`` or ``[T, H]`` residual in the same H order as
            ``hidden_states``. When set, the kernel adds it before the norm
            (``hidden = hidden_states + residual``) on the DMA engine and also returns the
            pre-norm sum, so the caller can use it as the next residual instead of doing the
            add itself. Fuses the post-attention residual add into this launch.

    Returns:
        Tuple ``(norm_hidden, routing)``:
            norm_hidden: kernel path, pack_affinities=False -> ``[T, row_region]`` fp8 packed
                (``row_region = H + scale_region``). pack_affinities=True -> the SAME packed
                row but VIEWED AS bf16 ``[T, row_region // 2]`` (affinity tail readable as bf16;
                pass straight to ``moe_cte`` which re-views to fp8). torch fallback ->
                pack_affinities=True: the SAME packed bf16 row (built by ``_torch_pack_mx_prefill``,
                layout-faithful to the kernel); pack_affinities=False: ``[T, H]`` bf16 unquantized
                norm.
            routing: ``expert_index [T, top_k]`` int32 when ``pack_affinities=True``, else
                dense ``expert_affinities [T, E]`` bf16 (zero outside top-K).
        When ``residual`` is set a third element follows: ``output_residual`` ``[T, H]``, the
        pre-norm sum in the same H order as the inputs.
    """
    _validate_inputs(
        hidden_states, gamma, router_weights, router_bias, top_k, router_act_fn
    )

    # Legalize hidden to [B, S, H] for the kernel; gamma to [1, H].
    hidden_3d = hidden_states
    if hidden_3d.ndim == 2:
        hidden_3d = hidden_3d.unsqueeze(0)
    if gamma.ndim == 1:
        gamma = gamma.unsqueeze(0)
    residual_3d = residual
    if residual_3d is not None:
        if residual_3d.ndim == 2:
            residual_3d = residual_3d.unsqueeze(0)
        assert residual_3d.shape == hidden_3d.shape, (
            f"residual must match hidden_states {tuple(hidden_3d.shape)}, "
            f"got {tuple(residual_3d.shape)}"
        )

    if _can_use_kernel(hidden_3d, router_weights, top_k):
        # wrap_nki(...)[N] selects the LNC degree (grid size); the kernel only
        # supports LNC 1 or 2 and shards tokens across cores itself, so use [2]
        # (matches rmsnorm_router_topk_tkg / moe_cte). NOT the return-value count.
        wrapped = wrap_nki(_rmsnorm_mx_prefill_kernel)
        if pack_affinities:
            # Affinities folded into out_packed -> kernel returns [out_packed, expert_index].
            outs = wrapped[2](
                hidden_states=hidden_3d,
                gamma=gamma,
                router_weights=router_weights,
                router_bias=router_bias,
                eps=eps,
                top_k=top_k,
                router_act_fn=router_act_fn,
                qmx_output_dtype=nl.float8_e4m3fn_x4,
                pack_scales=True,
                pack_affinities=True,
                unpadded_hidden_size=unpadded_hidden_size,
                hidden_interleaved=hidden_interleaved,
                residual=residual_3d,
            )
            if residual_3d is not None:
                out_packed, expert_index, output_residual = outs
                return out_packed, expert_index, output_residual
            out_packed, expert_index = outs
            return out_packed, expert_index
        # Option A: kernel returns [out_packed, expert_index, expert_affinities]; drop
        # expert_index (build_blockwise_mapping derives blocks from dense affinities).
        outs = wrapped[2](
            hidden_states=hidden_3d,
            gamma=gamma,
            router_weights=router_weights,
            router_bias=router_bias,
            eps=eps,
            top_k=top_k,
            router_act_fn=router_act_fn,
            qmx_output_dtype=nl.float8_e4m3fn_x4,
            pack_scales=True,
            pack_affinities=False,
            unpadded_hidden_size=unpadded_hidden_size,
            hidden_interleaved=hidden_interleaved,
            residual=residual_3d,
        )
        if residual_3d is not None:
            out_packed, _expert_index, expert_affinities, output_residual = outs
            return out_packed, expert_affinities, output_residual
        out_packed, _expert_index, expert_affinities = outs
        return out_packed, expert_affinities

    # Residual add happens before the norm, exactly as the kernel does it.
    output_residual = None
    if residual_3d is not None:
        hidden_3d = hidden_3d + residual_3d
        output_residual = hidden_3d.reshape(-1, hidden_3d.shape[-1])

    if hidden_interleaved:
        # The torch path works in natural H (gamma/router_weights index natural h), so undo the
        # caller's interleave here to match what the kernel does internally.
        *pre, H = hidden_3d.shape
        hidden_3d = (
            hidden_3d.reshape(*pre, 4, H // 4)
            .transpose(-2, -1)
            .reshape(*pre, H)
            .contiguous()
        )

    row, routing = _torch_impl(
        hidden_3d,
        gamma,
        router_weights,
        router_bias,
        eps,
        top_k,
        router_act_fn,
        unpadded_hidden_size,
        pack_affinities,
    )
    if output_residual is not None:
        return row, routing, output_residual
    return row, routing


def _validate_inputs(
    hidden_states: torch.Tensor,
    gamma: torch.Tensor,
    router_weights: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    top_k: int,
    router_act_fn: RouterActFnType,
) -> None:
    """Validate inputs for rmsnorm_mx_prefill."""
    assert router_act_fn in (RouterActFnType.SOFTMAX, RouterActFnType.SIGMOID), (
        f"router_act_fn must be SOFTMAX or SIGMOID, got {router_act_fn}"
    )
    assert hidden_states.ndim in (2, 3), (
        f"hidden_states must be [B, S, H] or [T, H], got shape {hidden_states.shape}"
    )
    H = hidden_states.shape[-1]
    assert gamma.shape in ((1, H), (H,)), (
        f"gamma must be [1, {H}] or [{H}], got shape {gamma.shape}"
    )
    assert router_weights.ndim == 2 and router_weights.shape[0] == H, (
        f"router_weights must be [H={H}, E], got shape {router_weights.shape}"
    )
    E = router_weights.shape[1]
    if router_bias is not None:
        assert router_bias.shape in ((1, E), (E,)), (
            f"router_bias must be [1, {E}] or [{E}], got shape {router_bias.shape}"
        )
    assert 1 <= top_k <= E, f"top_k must be in [1, E={E}], got {top_k}"


def _can_use_kernel(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    top_k: int,
) -> bool:
    """Return True if the NKI kernel can run for these inputs, else False.

    Constraints (from the kernel): device supports NKI (Trn3+), H % 512 == 0,
    E <= 512, top_k <= 8. On any miss we fall back to the torch reference.
    """
    if not can_run_kernel(hidden_states):
        return False
    H = hidden_states.shape[-1]
    E = router_weights.shape[1]
    if H % _MX_BLOCK != 0:
        return False
    if E > _E_MAX or top_k > _TOP_K_MAX:
        return False
    return True


def _torch_impl(
    hidden_states: torch.Tensor,
    gamma: torch.Tensor,
    router_weights: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    eps: float,
    top_k: int,
    router_act_fn: RouterActFnType,
    unpadded_hidden_size: Optional[int],
    pack_affinities: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch fallback: RMSNorm + router, simulating the kernel's packing on ``pack_affinities``.

    Delegates to ``_torch_rms_norm`` and ``_torch_router_impl`` (shared with ``NF.router``)
    so the norm + routing math has a single source of truth. Runs when the kernel cannot
    (CPU / non-Trn3 / H not a multiple of 512 / E or top_k out of range, or the kernel is
    force-disabled). Faithful, NOT performant.

    Returns ``(row, routing)``:
      - ``pack_affinities=True``: ``row`` is the packed ``[T, row_region // 2]`` bf16-viewed
        ``[hidden | scale | affinities | pad]`` row (built by ``_torch_pack_mx_prefill``, mirroring
        the kernel layout so ``slice_affinities_from_packed`` + the downstream packed-tail read
        consume it identically); ``routing`` is ``expert_index [T, top_k]`` int32.
      - ``pack_affinities=False``: ``row`` is the plain bf16 ``[T, H]`` norm; ``routing`` is the
        dense ``expert_affinities [T, E]``.

    NOTE: the packed hidden round-trips fp8 e4m3, so it is lower precision than the kernel only up
    to e4m3 rounding; the layout is bit-faithful. The router math assumes natural-order weights, so
    SWIZZLE-H weights (as the model passes) route as if natural. This is fine for the wrapper's own
    functional tests (natural weights); the on-device model normally takes the kernel path.
    """
    # Lazy import to avoid a circular import at module load.
    from vllm_neuron.functional.moe.router import (
        _torch_rms_norm,
        _torch_router_impl,
        RouterComputationOrder,
    )

    B, S, H = hidden_states.shape
    T = B * S

    # RMSNorm via the shared primitive. eps here is the mean-over-H form; the fallback uses
    # the standard mean (it does not apply the kernel's padded-H mean over
    # unpadded_hidden_size), which is one more reason it is not model-accurate for gpt-oss.
    norm = _torch_rms_norm(hidden_states.reshape(T, H), gamma.reshape(H), eps).to(
        hidden_states.dtype
    )

    activation = "softmax" if router_act_fn == RouterActFnType.SOFTMAX else "sigmoid"
    expert_affinities, _logits = _torch_router_impl(
        hidden_states=norm,
        router_weights=router_weights,
        top_k=top_k,
        router_bias=router_bias.reshape(-1) if router_bias is not None else None,
        activation=activation,
        gamma=None,  # norm already applied above
        eps=eps,
        computation_dtype=torch.float16,
        router_computation_order=RouterComputationOrder.PRENORM_LINEAR_TOPK_ACT_SCATTER,
    )
    if pack_affinities:
        # Simulate the kernel's packed row so downstream consumers (slice_affinities_from_packed,
        # moe_cte packed-tail read) see the same layout. compute_dtype = router_weights.dtype.
        packed = _torch_pack_mx_prefill(
            norm.to(torch.float32), expert_affinities, router_weights.dtype
        )
        # Match the kernel's pack_affinities return: expert_index [T, top_k] int32.
        expert_index = torch.topk(expert_affinities, top_k, dim=-1).indices.to(
            torch.int32
        )
        return packed, expert_index
    return norm, expert_affinities.to(torch.bfloat16)
