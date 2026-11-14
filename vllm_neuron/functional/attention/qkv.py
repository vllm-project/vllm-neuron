# SPDX-License-Identifier: Apache-2.0
from typing import Optional
from torch import Tensor
import torch

import nki
import nki.language as nl

from nkilib.core.qkv.qkv import qkv as nki_qkv_kernel
from nkilib.core.utils.common_types import (
    DtypeMode,
    NormType,
    QKNormConfig,
    QKVOutputLayout,
    QKVWeightLayout,
    QuantizationType,
)
from nkilib.core.utils.allocator import SbufManager

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel
from vllm_neuron.utils.hardware_config import get_logical_nc_config

# qkv_cte allocates the packed-FP8 store buffer with ``dtype=cfg.kv_dtype`` and
# only treats an NKI dtype (or the ``"float8e4"`` sentinel) as FP8; a ``torch``
# dtype falls through to bf16, breaking the packed ``.view(bf16)`` free-dim math.
# The public API takes torch dtypes (CPU fallback uses ``.to(kv_dtype)``), so map
# to the NKI dtype only for the on-device kernel dispatch. Mirrors
# ``permute_routed_tokens._TORCH_NKI_DTYPE_MAP``.
_TORCH_NKI_KV_DTYPE_MAP = {
    torch.float8_e4m3fn: nl.float8_e4m3fn,
    torch.float8_e5m2: nl.float8_e5m2,
}


def _validate_in_kernel_kv_cache_write_args(
    k_cache: Optional[Tensor],
    v_cache: Optional[Tensor],
    use_block_kv: bool,
    slot_mapping: Optional[Tensor],
    block_size: Optional[int],
) -> None:
    """Guardrail for the in-kernel KV cache write path.

    The underlying NKI kernel triggers the in-kernel write whenever
    ``k_cache``/``v_cache`` are non-None, and lets ``use_block_kv`` independently
    select between the paged and flat cache layouts. vLLM-Neuron's caches are
    always paged (``[num_blocks, num_kv_heads, block_size, d_head]``), so the
    flat-layout in-kernel write path would silently corrupt cache state. This
    helper rejects that combination (and partial cache args) at the wrapper
    boundary so misuse fails fast instead of producing wrong logits.
    """
    has_kv_cache = k_cache is not None or v_cache is not None
    if not has_kv_cache:
        return
    assert k_cache is not None and v_cache is not None, (
        "k_cache and v_cache must be passed together for in-kernel KV cache write"
    )
    assert use_block_kv, (
        "in-kernel KV cache write requires use_block_kv=True (paged layout); "
        "flat-layout cache write is not supported"
    )
    assert slot_mapping is not None and block_size is not None, (
        "slot_mapping and block_size are required when use_block_kv=True"
    )


def _maybe_build_sbm(
    sbm_lower_bound: Optional[int],
    sbm_upper_bound: Optional[int],
    sbm_use_auto_alloc: bool,
    sbm_default_stack_alloc: bool,
) -> Optional[SbufManager]:
    """Reconstruct SbufManager only when bounds are explicitly provided."""

    if sbm_lower_bound is None and sbm_upper_bound is None:
        return None

    return SbufManager(
        sb_lower_bound=sbm_lower_bound,
        sb_upper_bound=sbm_upper_bound,
        use_auto_alloc=sbm_use_auto_alloc,
        default_stack_alloc=sbm_default_stack_alloc,
    )


@nki.jit
def _torch_compatible_qkv_kernel(
    hidden: Tensor,
    qkv_weights: Tensor,
    norm_weights: Optional[Tensor] = None,
    mlp_prev: Optional[Tensor] = None,
    attn_prev: Optional[Tensor] = None,
    d_head: Optional[int] = None,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    eps: float = 1e-6,
    norm_type: NormType = NormType.NO_NORM,
    use_dma_transpose: bool = True,
    store_output_in_sbuf: bool = False,
    bias: Optional[Tensor] = None,
    norm_b: Optional[Tensor] = None,
    hidden_actual: Optional[int] = None,
    cos_cache: Optional[Tensor] = None,
    sin_cache: Optional[Tensor] = None,
    num_q_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    qkv_w_scale: Optional[Tensor] = None,
    qkv_in_scale: Optional[Tensor] = None,
    weight_layout: QKVWeightLayout = QKVWeightLayout.CONTIGUOUS,
    fused_residual_add: Optional[bool] = False,
    fused_rope: Optional[bool] = False,
    qk_norm_pre_rope_q_norm: Optional[NormType] = None,
    qk_norm_pre_rope_k_norm: Optional[NormType] = None,
    qk_norm_pre_rope_eps: float = 1e-6,
    qk_norm_pre_rope_gamma_fused_in_rope_caches: bool = False,
    qk_norm_pre_rope_q_gamma: Optional[Tensor] = None,
    qk_norm_pre_rope_k_gamma: Optional[Tensor] = None,
    qk_norm_pre_rope_q_beta: Optional[Tensor] = None,
    qk_norm_pre_rope_k_beta: Optional[Tensor] = None,
    qk_norm_post_rope_q_norm: Optional[NormType] = None,
    qk_norm_post_rope_k_norm: Optional[NormType] = None,
    qk_norm_post_rope_eps: float = 1e-6,
    qk_norm_post_rope_gamma_fused_in_rope_caches: bool = False,
    qk_norm_post_rope_q_gamma: Optional[Tensor] = None,
    qk_norm_post_rope_k_gamma: Optional[Tensor] = None,
    qk_norm_post_rope_q_beta: Optional[Tensor] = None,
    qk_norm_post_rope_k_beta: Optional[Tensor] = None,
    sbm_lower_bound: int = None,
    sbm_upper_bound: int = None,
    sbm_use_auto_alloc: bool = False,
    sbm_default_stack_alloc: bool = True,
    use_auto_allocation: bool = False,
    # --- In-kernel block KV cache write ---
    k_cache: Optional[Tensor] = None,
    v_cache: Optional[Tensor] = None,
    k_scale: Optional[Tensor] = None,
    v_scale: Optional[Tensor] = None,
    fp8_max: Optional[float] = None,
    fp8_min: Optional[float] = None,
    kv_dtype: Optional[torch.dtype] = None,
    use_block_kv: bool = False,
    transpose_k_cache: bool = False,
    fp8_packed: bool = False,
    block_size: Optional[int] = None,
    slot_mapping: Optional[Tensor] = None,
):
    sbm = _maybe_build_sbm(
        sbm_lower_bound=sbm_lower_bound,
        sbm_upper_bound=sbm_upper_bound,
        sbm_use_auto_alloc=sbm_use_auto_alloc,
        sbm_default_stack_alloc=sbm_default_stack_alloc,
    )

    has_pre_rope = (
        qk_norm_pre_rope_q_norm is not None or qk_norm_pre_rope_k_norm is not None
    )
    has_post_rope = (
        qk_norm_post_rope_q_norm is not None or qk_norm_post_rope_k_norm is not None
    )

    assert not has_pre_rope or (num_q_heads is not None and num_kv_heads is not None), (
        "num_q_heads and num_kv_heads are required when pre-rope qk_norm is enabled"
    )
    assert not has_post_rope or (
        num_q_heads is not None and num_kv_heads is not None
    ), "num_q_heads and num_kv_heads are required when post-rope qk_norm is enabled"

    _validate_in_kernel_kv_cache_write_args(
        k_cache=k_cache,
        v_cache=v_cache,
        use_block_kv=use_block_kv,
        slot_mapping=slot_mapping,
        block_size=block_size,
    )

    qk_norm_pre_rope_config = (
        QKNormConfig(
            q_norm=qk_norm_pre_rope_q_norm,
            k_norm=qk_norm_pre_rope_k_norm,
            eps=qk_norm_pre_rope_eps,
            gamma_fused_in_rope_caches=qk_norm_pre_rope_gamma_fused_in_rope_caches,
        )
        if has_pre_rope
        else None
    )

    qk_norm_post_rope_config = (
        QKNormConfig(
            q_norm=qk_norm_post_rope_q_norm,
            k_norm=qk_norm_post_rope_k_norm,
            eps=qk_norm_post_rope_eps,
            gamma_fused_in_rope_caches=qk_norm_post_rope_gamma_fused_in_rope_caches,
        )
        if has_post_rope
        else None
    )

    return nki_qkv_kernel(
        # Core inputs
        input=hidden,
        fused_qkv_weights=qkv_weights,
        output_layout=output_layout,
        # Bias
        bias=bias,
        # Quantization
        quantization_type=quantization_type,
        qkv_w_scale=qkv_w_scale,
        qkv_in_scale=qkv_in_scale,
        weight_layout=weight_layout,
        # Fused Residual Add
        fused_residual_add=fused_residual_add,
        mlp_prev=mlp_prev,
        attention_prev=attn_prev,
        # Fused Norm
        fused_norm_type=norm_type,
        gamma_norm_weights=norm_weights,
        layer_norm_bias=norm_b,
        norm_eps=eps,
        hidden_actual=hidden_actual,
        # Fused RoPE
        fused_rope=fused_rope,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        d_head=d_head,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        # In-kernel block KV cache write (post-RoPE K/V)
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
        fp8_max=fp8_max,
        fp8_min=fp8_min,
        kv_dtype=kv_dtype,
        use_block_kv=use_block_kv,
        transpose_k_cache=transpose_k_cache,
        fp8_packed=fp8_packed,
        block_size=block_size,
        slot_mapping=slot_mapping,
        # Output storage
        store_output_in_sbuf=store_output_in_sbuf,
        # QK-Norm: passed directly as pre/post rope
        qk_norm_pre_rope=qk_norm_pre_rope_config,
        qk_norm_pre_rope_q_gamma=qk_norm_pre_rope_q_gamma,
        qk_norm_pre_rope_k_gamma=qk_norm_pre_rope_k_gamma,
        qk_norm_pre_rope_q_beta=qk_norm_pre_rope_q_beta,
        qk_norm_pre_rope_k_beta=qk_norm_pre_rope_k_beta,
        qk_norm_post_rope=qk_norm_post_rope_config,
        qk_norm_post_rope_q_gamma=qk_norm_post_rope_q_gamma,
        qk_norm_post_rope_k_gamma=qk_norm_post_rope_k_gamma,
        qk_norm_post_rope_q_beta=qk_norm_post_rope_q_beta,
        qk_norm_post_rope_k_beta=qk_norm_post_rope_k_beta,
        # SBUF management
        sbm=sbm,
        use_auto_allocation=use_auto_allocation,
        # DMA
        load_input_with_DMA_transpose=use_dma_transpose,
        # FP8 dtype: AUTO resolves to OCP (e4m3fn) on TRN3, NON_OCP on TRN2
        dtype_mode=DtypeMode.AUTO,
    )


def _apply_qk_norm(
    output: Tensor,
    fused_qkv_dim: int,
    d_head: Optional[int],
    num_q_heads: int,
    num_kv_heads: int,
    qk_norm_q_norm: Optional[NormType],
    qk_norm_k_norm: Optional[NormType],
    qk_norm_eps: float,
    qk_norm_gamma_fused_in_rope_caches: bool,
    qk_norm_q_gamma: Optional[Tensor],
    qk_norm_k_gamma: Optional[Tensor],
    qk_norm_q_beta: Optional[Tensor],
    qk_norm_k_beta: Optional[Tensor],
) -> Tensor:
    """Apply per-head normalization to Q and K heads in the fused QKV output."""
    assert not qk_norm_gamma_fused_in_rope_caches, (
        "gamma_fused_in_rope_caches is not supported in the PyTorch reference for now"
    )
    assert qk_norm_q_norm == NormType.RMS_NORM, (
        f"Only RMS_NORM is supported for QK-norm for now, got q_norm={qk_norm_q_norm}"
    )
    assert qk_norm_k_norm == NormType.RMS_NORM, (
        f"Only RMS_NORM is supported for QK-norm for now, got k_norm={qk_norm_k_norm}"
    )

    qk_d_head = (
        d_head
        if d_head is not None
        else fused_qkv_dim // (num_q_heads + 2 * num_kv_heads)
    )
    num_norm_heads = num_q_heads + num_kv_heads
    for head_idx in range(num_norm_heads):
        start = head_idx * qk_d_head
        end = start + qk_d_head
        head = output[:, :, start:end].float()
        variance = head.pow(2).mean(-1, keepdim=True)
        head = head * torch.rsqrt(variance + qk_norm_eps)
        gamma = qk_norm_q_gamma if head_idx < num_q_heads else qk_norm_k_gamma
        if gamma is not None:
            head = head * gamma
        beta = qk_norm_q_beta if head_idx < num_q_heads else qk_norm_k_beta
        if beta is not None:
            head = head + beta
        output[:, :, start:end] = head.to(output.dtype)
    return output


def _apply_qkv_output_layout(
    x: Tensor,
    d_head: Optional[int],
    output_layout: QKVOutputLayout,
) -> Tensor:
    """Permute a ``[B, S, num_heads * d_head]`` tensor into ``output_layout``.

    ``num_heads`` is ``x.shape[-1] // d_head``, so this serves the fused Q+K+V
    tensor of the plain path and the Q-only tensor of the KV-cache-write path.

    Args:
        x: A projection output, shape ``[B, S, num_heads * d_head]``.
        d_head: The head dimension. Each layout except ``BSD`` needs it.
        output_layout: The requested layout.

    Returns:
        ``x`` when ``output_layout`` is ``BSD``. A permuted view for each other
        layout.

    Examples:
        >>> import torch
        >>> from nkilib.core.utils.common_types import QKVOutputLayout
        >>> q = torch.zeros(1, 8, 256)  # 4 heads x d_head 64
        >>> _apply_qkv_output_layout(q, 64, QKVOutputLayout.BNSd).shape
        torch.Size([1, 4, 8, 64])
    """
    if output_layout == QKVOutputLayout.BSD:
        return x

    if d_head is None:
        raise ValueError(f"d_head must be specified for {output_layout} output layout")

    # Use reshape: on the KV-cache-write path x is a non-contiguous slice.
    B, S, D = x.shape
    x = x.reshape(B, S, D // d_head, d_head)

    if output_layout == QKVOutputLayout.BNSd:
        return x.permute(0, 2, 1, 3)
    if output_layout == QKVOutputLayout.NBSd:
        return x.permute(2, 0, 1, 3)
    if output_layout == QKVOutputLayout.NBdS:
        return x.permute(2, 0, 3, 1)
    raise ValueError(f"Unsupported output layout: {output_layout}")


def _torch_qkv_in_kernel_cache_write_impl(
    hidden: Tensor,
    qkv_weights: Tensor,
    bias: Optional[Tensor],
    d_head: Optional[int],
    cos_cache: Optional[Tensor],
    sin_cache: Optional[Tensor],
    num_q_heads: Optional[int],
    num_kv_heads: Optional[int],
    k_cache: Tensor,
    v_cache: Tensor,
    k_scale: Optional[Tensor],
    v_scale: Optional[Tensor],
    fp8_max: Optional[float],
    fp8_min: Optional[float],
    kv_dtype: Optional[torch.dtype],
    transpose_k_cache: bool,
    fp8_packed: bool,
    block_size: Optional[int],
    slot_mapping: Optional[Tensor],
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    norm_weights: Optional[Tensor] = None,
    norm_type: NormType = NormType.NO_NORM,
    eps: float = 1e-6,
    qk_norm_pre_rope_q_norm: Optional[NormType] = None,
    qk_norm_pre_rope_k_norm: Optional[NormType] = None,
    qk_norm_pre_rope_eps: float = 1e-6,
    qk_norm_pre_rope_gamma_fused_in_rope_caches: bool = False,
    qk_norm_pre_rope_q_gamma: Optional[Tensor] = None,
    qk_norm_pre_rope_k_gamma: Optional[Tensor] = None,
    qk_norm_pre_rope_q_beta: Optional[Tensor] = None,
    qk_norm_pre_rope_k_beta: Optional[Tensor] = None,
    qk_norm_post_rope_q_norm: Optional[NormType] = None,
    qk_norm_post_rope_k_norm: Optional[NormType] = None,
    qk_norm_post_rope_eps: float = 1e-6,
    qk_norm_post_rope_gamma_fused_in_rope_caches: bool = False,
    qk_norm_post_rope_q_gamma: Optional[Tensor] = None,
    qk_norm_post_rope_k_gamma: Optional[Tensor] = None,
    qk_norm_post_rope_q_beta: Optional[Tensor] = None,
    qk_norm_post_rope_k_beta: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """PyTorch fallback for ``qkv_proj(use_block_kv=True, ...)``.

    Returns ``(q, k_cache, v_cache)``: runs projection + fused norm + QK-norm +
    RoPE via the shared ``_torch_qkv_impl``, then scatters the post-RoPE K/V into
    the paged cache **in place** at the ``slot_mapping`` positions (mirroring the
    kernel's must-alias semantics so downstream cache readers see the write).

    The fused-norm and QK-norm parameters MUST be forwarded to ``_torch_qkv_impl``
    so the fallback matches the kernel exactly: models like Qwen3 apply a per-head
    RMS QK-norm before RoPE, and dropping it here makes the projected Q and the
    cached K diverge from the kernel (and from the plain non-cache-write fallback),
    silently corrupting attention.

    Intentionally does not use ``nkilib...qkv_cte.qkv_cte_torch_ref``: it is
    numpy + python per-slot iteration (not dynamo/XLA traceable) and
    zero-fills the whole cache on every call (wiping prior-chunk K/V in CPU
    mode). The in-place scatter below avoids both. ``slot_mapping`` /
    ``block_size`` non-None and ``use_block_kv`` are guaranteed by the
    ``_validate_in_kernel_kv_cache_write_args`` guardrail at the wrapper.
    """
    if transpose_k_cache:
        raise NotImplementedError(
            "torch fallback does not support transpose_k_cache=True; the "
            "transposed-K layout is implemented only on the kernel path."
        )
    if fp8_packed:
        raise NotImplementedError(
            "torch fallback does not support fp8_packed=True; the packed-FP8 "
            "5D per-head K layout is implemented only on the kernel path. "
            "Run packed segmented prefill on-device (NKI), not in CPU mode."
        )
    if k_cache.dim() != 4:
        raise NotImplementedError(
            "torch fallback expects a 4D paged cache "
            "[num_blocks, num_kv_heads, block_size, d_head], got "
            f"{tuple(k_cache.shape)}."
        )
    if d_head is None or num_q_heads is None or num_kv_heads is None:
        raise ValueError(
            "d_head, num_q_heads, and num_kv_heads are required for the "
            "in-kernel KV cache write fallback."
        )

    qkv_out = _torch_qkv_impl(
        hidden=hidden,
        qkv_weights=qkv_weights,
        norm_weights=norm_weights,
        d_head=d_head,
        output_layout=QKVOutputLayout.BSD,
        eps=eps,
        norm_type=norm_type,
        bias=bias,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        qk_norm_pre_rope_q_norm=qk_norm_pre_rope_q_norm,
        qk_norm_pre_rope_k_norm=qk_norm_pre_rope_k_norm,
        qk_norm_pre_rope_eps=qk_norm_pre_rope_eps,
        qk_norm_pre_rope_gamma_fused_in_rope_caches=qk_norm_pre_rope_gamma_fused_in_rope_caches,
        qk_norm_pre_rope_q_gamma=qk_norm_pre_rope_q_gamma,
        qk_norm_pre_rope_k_gamma=qk_norm_pre_rope_k_gamma,
        qk_norm_pre_rope_q_beta=qk_norm_pre_rope_q_beta,
        qk_norm_pre_rope_k_beta=qk_norm_pre_rope_k_beta,
        qk_norm_post_rope_q_norm=qk_norm_post_rope_q_norm,
        qk_norm_post_rope_k_norm=qk_norm_post_rope_k_norm,
        qk_norm_post_rope_eps=qk_norm_post_rope_eps,
        qk_norm_post_rope_gamma_fused_in_rope_caches=qk_norm_post_rope_gamma_fused_in_rope_caches,
        qk_norm_post_rope_q_gamma=qk_norm_post_rope_q_gamma,
        qk_norm_post_rope_k_gamma=qk_norm_post_rope_k_gamma,
        qk_norm_post_rope_q_beta=qk_norm_post_rope_q_beta,
        qk_norm_post_rope_k_beta=qk_norm_post_rope_k_beta,
    )  # [B, S, (num_q_heads + 2 * num_kv_heads) * d_head]

    q_dim = num_q_heads * d_head
    kv_dim = num_kv_heads * d_head
    q = qkv_out[:, :, :q_dim]
    k = qkv_out[:, :, q_dim : q_dim + kv_dim]
    v = qkv_out[:, :, q_dim + kv_dim :]

    # FP8 KV cache stores quantized values: divide by the dequant scale then
    # clamp (matches the qkv_cte kernel / qkv_cte_torch_ref convention). BF16
    # caches store post-RoPE values directly.
    if k_scale is not None and v_scale is not None:
        k = (k.to(torch.float32) / k_scale.float().flatten()[0]).clamp(fp8_min, fp8_max)
        v = (v.to(torch.float32) / v_scale.float().flatten()[0]).clamp(fp8_min, fp8_max)
    store_dtype = kv_dtype if kv_dtype is not None else k_cache.dtype

    # Scatter into the 4D cache [num_blocks, num_kv_heads, block_size, d_head].
    # k/v reshape to [n_tokens * num_kv_heads, d_head] in token-major,
    # head-minor order; the (block, head, pos) index triple below is built in
    # the same order so each row lands in its slot. (This differs from
    # model.py's head-major scatter, whose source K is [num_kv_heads, S, d_head].)
    n_tokens = k.shape[0] * k.shape[1]
    slot = slot_mapping.reshape(n_tokens).to(torch.long)
    block_of_token = slot // block_size
    pos_of_token = slot % block_size

    head_index = torch.arange(num_kv_heads, device=k_cache.device).repeat(n_tokens)
    block_index = block_of_token.repeat_interleave(num_kv_heads)
    pos_index = pos_of_token.repeat_interleave(num_kv_heads)

    k_vals = k.reshape(n_tokens * num_kv_heads, d_head).to(store_dtype)
    v_vals = v.reshape(n_tokens * num_kv_heads, d_head).to(store_dtype)
    # Q only: the cache keeps its own layout. Relayout Q before the scatter,
    # because the helper can raise and a raise after the scatter leaves the
    # caches half written.
    q = _apply_qkv_output_layout(q, d_head, output_layout)

    k_cache.index_put_((block_index, head_index, pos_index), k_vals)
    v_cache.index_put_((block_index, head_index, pos_index), v_vals)

    return q, k_cache, v_cache


def _torch_qkv_impl(
    hidden: Tensor,
    qkv_weights: Tensor,
    norm_weights: Optional[Tensor] = None,
    mlp_prev: Optional[Tensor] = None,
    attn_prev: Optional[Tensor] = None,
    d_head: Optional[int] = None,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    eps: float = 1e-6,
    norm_type: NormType = NormType.NO_NORM,
    bias: Optional[Tensor] = None,
    norm_b: Optional[Tensor] = None,
    hidden_actual: Optional[int] = None,
    cos_cache: Optional[Tensor] = None,
    sin_cache: Optional[Tensor] = None,
    num_q_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    qk_norm_pre_rope_q_norm: Optional[NormType] = None,
    qk_norm_pre_rope_k_norm: Optional[NormType] = None,
    qk_norm_pre_rope_eps: float = 1e-6,
    qk_norm_pre_rope_gamma_fused_in_rope_caches: bool = False,
    qk_norm_pre_rope_q_gamma: Optional[Tensor] = None,
    qk_norm_pre_rope_k_gamma: Optional[Tensor] = None,
    qk_norm_pre_rope_q_beta: Optional[Tensor] = None,
    qk_norm_pre_rope_k_beta: Optional[Tensor] = None,
    qk_norm_post_rope_q_norm: Optional[NormType] = None,
    qk_norm_post_rope_k_norm: Optional[NormType] = None,
    qk_norm_post_rope_eps: float = 1e-6,
    qk_norm_post_rope_gamma_fused_in_rope_caches: bool = False,
    qk_norm_post_rope_q_gamma: Optional[Tensor] = None,
    qk_norm_post_rope_k_gamma: Optional[Tensor] = None,
    qk_norm_post_rope_q_beta: Optional[Tensor] = None,
    qk_norm_post_rope_k_beta: Optional[Tensor] = None,
) -> Tensor:
    """
    PyTorch reference implementation of QKV projection.
    """
    has_pre_rope = (
        qk_norm_pre_rope_q_norm is not None or qk_norm_pre_rope_k_norm is not None
    )
    has_post_rope = (
        qk_norm_post_rope_q_norm is not None or qk_norm_post_rope_k_norm is not None
    )

    if has_pre_rope or has_post_rope:
        assert num_q_heads is not None and num_kv_heads is not None, (
            "num_q_heads and num_kv_heads must be specified when qk_norm is enabled"
        )

    B, S, H = hidden.shape
    fused_qkv_dim = qkv_weights.shape[1]

    # Step 1: Fused add
    if mlp_prev is not None and attn_prev is not None:
        hidden = hidden + mlp_prev + attn_prev

    # Step 2: Apply normalization
    if hidden_actual is None:
        hidden_actual = H

    if norm_type == NormType.RMS_NORM:
        variance = hidden.pow(2).sum(-1, keepdim=True) / hidden_actual
        hidden = hidden * torch.rsqrt(variance + eps)
        if norm_weights is not None:
            hidden = hidden * norm_weights

    elif norm_type == NormType.LAYER_NORM:
        mean = hidden.mean(-1, keepdim=True)
        variance = hidden.var(-1, keepdim=True, unbiased=False)
        hidden = (hidden - mean) * torch.rsqrt(variance + eps)
        if norm_weights is not None:
            hidden = hidden * norm_weights
        if norm_b is not None:
            hidden = hidden + norm_b

    elif norm_type == NormType.RMS_NORM_SKIP_GAMMA:
        variance = hidden.pow(2).sum(-1, keepdim=True) / hidden_actual
        hidden = hidden * torch.rsqrt(variance + eps)

    # Step 3: Matmul
    output = hidden @ qkv_weights

    # Step 4: Add bias
    if bias is not None:
        output = output + bias

    # Step 4.5: Apply per-head QK-norm (pre-RoPE)
    if has_pre_rope:
        output = _apply_qk_norm(
            output,
            fused_qkv_dim,
            d_head,
            num_q_heads,
            num_kv_heads,
            qk_norm_pre_rope_q_norm,
            qk_norm_pre_rope_k_norm,
            qk_norm_pre_rope_eps,
            qk_norm_pre_rope_gamma_fused_in_rope_caches,
            qk_norm_pre_rope_q_gamma,
            qk_norm_pre_rope_k_gamma,
            qk_norm_pre_rope_q_beta,
            qk_norm_pre_rope_k_beta,
        )

    # Step 5: Apply RoPE if cos/sin caches provided
    fused_rope = (
        cos_cache is not None
        and sin_cache is not None
        and num_q_heads is not None
        and num_kv_heads is not None
    )

    if fused_rope:
        rope_d_head = (
            d_head
            if d_head is not None
            else fused_qkv_dim // (num_q_heads + 2 * num_kv_heads)
        )
        num_rope_heads = num_q_heads + num_kv_heads

        for head_idx in range(num_rope_heads):
            start_idx = head_idx * rope_d_head
            end_idx = start_idx + rope_d_head
            half_d = rope_d_head // 2

            head_output = output[:, :, start_idx:end_idx]
            x1 = head_output[:, :, :half_d]
            x2 = head_output[:, :, half_d:]

            cos = cos_cache
            sin = sin_cache[:, :, :half_d]

            rotated_x1 = x1 * cos[:, :, :half_d] - x2 * sin
            rotated_x2 = x2 * cos[:, :, half_d:] + x1 * sin

            output[:, :, start_idx : start_idx + half_d] = rotated_x1
            output[:, :, start_idx + half_d : end_idx] = rotated_x2

    # Step 5.5: Apply per-head QK-norm (post-RoPE)
    if has_post_rope:
        output = _apply_qk_norm(
            output,
            fused_qkv_dim,
            d_head,
            num_q_heads,
            num_kv_heads,
            qk_norm_post_rope_q_norm,
            qk_norm_post_rope_k_norm,
            qk_norm_post_rope_eps,
            qk_norm_post_rope_gamma_fused_in_rope_caches,
            qk_norm_post_rope_q_gamma,
            qk_norm_post_rope_k_gamma,
            qk_norm_post_rope_q_beta,
            qk_norm_post_rope_k_beta,
        )

    # Step 6: Handle output layout. Here the head count spans Q, K and V.
    return _apply_qkv_output_layout(output, d_head, output_layout)


def _can_use_qkv_kernel(
    hidden: Tensor,
    qkv_weights: Tensor,
    mlp_prev: Optional[Tensor],
    attn_prev: Optional[Tensor],
    d_head: Optional[int],
    output_layout: QKVOutputLayout,
    norm_type: NormType,
    norm_b: Optional[Tensor],
    quantization_type: QuantizationType = QuantizationType.NONE,
    qk_norm_pre_rope_q_beta: Optional[Tensor] = None,
    qk_norm_pre_rope_k_beta: Optional[Tensor] = None,
    qk_norm_post_rope_q_beta: Optional[Tensor] = None,
    qk_norm_post_rope_k_beta: Optional[Tensor] = None,
    fused_rope: bool = False,
) -> bool:
    """
    Check if the NKI QKV kernel can be used based on constraints.

    Returns True if all constraints are satisfied, False otherwise.
    """
    if not can_run_kernel(hidden):
        return False

    # The fused RoPE path needs a d_head that divides the 512-element PSUM bank.
    # qkv_cte evicts PSUM per head and computes
    # ``NUM_HEADS_PER_PSUM_BANK = 512 // d_head``.
    # The division truncates, so a d_head that does not divide 512 leaves a remainder,
    # and each head that crosses a bank boundary reads the wrong PSUM offset and
    # produces incorrect output.
    if fused_rope and (d_head is None or 512 % d_head != 0):
        return False

    # nkilib nki_qkv_kernel does not support qk_norm beta weights yet
    if (
        qk_norm_pre_rope_q_beta is not None
        or qk_norm_pre_rope_k_beta is not None
        or qk_norm_post_rope_q_beta is not None
        or qk_norm_post_rope_k_beta is not None
    ):
        return False

    _, _, H = hidden.shape
    fused_qkv_dim = qkv_weights.shape[1]

    if H > 24576:
        return False

    if H % 128 != 0 and quantization_type not in (
        QuantizationType.MX,
        QuantizationType.ROW_MX,
    ):
        return False

    if fused_qkv_dim > 4096:
        return False

    # Residual add: both must be provided or neither
    if (mlp_prev is None) != (attn_prev is None):
        return False

    # norm_b only supported for LAYER_NORM
    if norm_b is not None and norm_type != NormType.LAYER_NORM:
        return False

    # NBdS layout is not supported in the kernel
    if output_layout == QKVOutputLayout.NBdS:
        return False

    # For NBSd layout, d_head must be 128
    if output_layout == QKVOutputLayout.NBSd:
        if d_head is None or d_head != 128:
            return False

    return True


def qkv_proj(
    hidden: Tensor,
    qkv_weights: Tensor,
    norm_weights: Optional[Tensor] = None,
    mlp_prev: Optional[Tensor] = None,
    attn_prev: Optional[Tensor] = None,
    d_head: Optional[int] = None,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    eps: float = 1e-6,
    norm_type: NormType = NormType.NO_NORM,
    use_dma_transpose: bool = True,
    store_output_in_sbuf: bool = False,
    bias: Optional[Tensor] = None,
    norm_b: Optional[Tensor] = None,
    hidden_actual: Optional[int] = None,
    cos_cache: Optional[Tensor] = None,
    sin_cache: Optional[Tensor] = None,
    num_q_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    qkv_w_scale: Optional[Tensor] = None,
    qkv_in_scale: Optional[Tensor] = None,
    weight_layout: QKVWeightLayout = QKVWeightLayout.CONTIGUOUS,
    sbm_lower_bound: int = None,
    sbm_upper_bound: int = None,
    sbm_use_auto_alloc: bool = False,
    sbm_default_stack_alloc: bool = True,
    use_auto_allocation: bool = False,
    qk_norm_pre_rope_q_norm: Optional[NormType] = None,
    qk_norm_pre_rope_k_norm: Optional[NormType] = None,
    qk_norm_pre_rope_eps: float = 1e-6,
    qk_norm_pre_rope_gamma_fused_in_rope_caches: bool = False,
    qk_norm_pre_rope_q_gamma: Optional[Tensor] = None,
    qk_norm_pre_rope_k_gamma: Optional[Tensor] = None,
    qk_norm_pre_rope_q_beta: Optional[Tensor] = None,
    qk_norm_pre_rope_k_beta: Optional[Tensor] = None,
    qk_norm_post_rope_q_norm: Optional[NormType] = None,
    qk_norm_post_rope_k_norm: Optional[NormType] = None,
    qk_norm_post_rope_eps: float = 1e-6,
    qk_norm_post_rope_gamma_fused_in_rope_caches: bool = False,
    qk_norm_post_rope_q_gamma: Optional[Tensor] = None,
    qk_norm_post_rope_k_gamma: Optional[Tensor] = None,
    qk_norm_post_rope_q_beta: Optional[Tensor] = None,
    qk_norm_post_rope_k_beta: Optional[Tensor] = None,
    # --- In-kernel block KV cache write ---
    k_cache: Optional[Tensor] = None,
    v_cache: Optional[Tensor] = None,
    k_scale: Optional[Tensor] = None,
    v_scale: Optional[Tensor] = None,
    fp8_max: Optional[float] = None,
    fp8_min: Optional[float] = None,
    kv_dtype: Optional[torch.dtype] = None,
    use_block_kv: bool = False,
    transpose_k_cache: bool = False,
    fp8_packed: bool = False,
    block_size: Optional[int] = None,
    slot_mapping: Optional[Tensor] = None,
) -> "Tensor | tuple[Tensor, Tensor, Tensor]":
    """
    QKV Projection API that automatically selects between NKI kernel and PyTorch fallback.

    When ``use_block_kv=True`` (in-kernel KV cache write path), returns the
    3-tuple ``(q, k_cache, v_cache)`` mirroring the kernel's must-alias
    output. Otherwise returns the standard concatenated-QKV ``Tensor``.

    This function checks kernel constraints and dispatches to:
    - nki_qkv_kernel: When all constraints are satisfied and running on Neuron
    - PyTorch implementation: When constraints are violated or not on Neuron

    This kernel implements QKV projection with optional normalization fusion used in
    transformer models: output = norm(hidden + mlp_prev + attn_prev) @ qkv_weights + bias

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension
        fused_qkv_dim: (num_q_heads + 2 * num_kv_heads) * d_head

    Args:
        hidden: Input tensor with shape [B, S, H]
        qkv_weights: Weight tensor with shape [H, fused_qkv_dim]
        norm_weights: Optional normalization weights (gamma) with shape [1, H]
        mlp_prev: Optional previous MLP output for residual with shape [B, S, H]
        attn_prev: Optional previous attention output for residual with shape [B, S, H]
        d_head: Head dimension. Required for each output layout except BSD, and
            for RoPE
        output_layout: Output tensor layout:
            - BSD: [B, S, fused_qkv_dim]
            - NBSd: [num_heads, B, S, d_head]
            - BNSd: [B, num_heads, S, d_head]
            - NBdS: [num_heads, B, d_head, S]
        eps: Epsilon for numerical stability in normalization. Default: 1e-6
        norm_type: Normalization type:
            - NO_NORM: No normalization
            - RMS_NORM: Root Mean Square normalization
            - LAYER_NORM: Layer normalization
            - RMS_NORM_SKIP_GAMMA: RMSNorm with gamma pre-multiplied in weights
        use_dma_transpose: Use DMA transpose for weight loading. Default: True
        store_output_in_sbuf: Store output in SBUF instead of HBM. Default: False
        bias: Optional bias tensor with shape [1, fused_qkv_dim]
        norm_b: Optional LayerNorm bias with shape [1, H]
        hidden_actual: Actual hidden dimension for padded tensors
        cos_cache: Cosine cache for RoPE with shape [B, S, d_head]
        sin_cache: Sine cache for RoPE with shape [B, S, d_head]
        num_q_heads: Number of query heads for RoPE
        num_kv_heads: Number of key/value heads for RoPE
        quantization_type: Type of quantization (NONE, ROW, STATIC). Default: NONE
        qkv_w_scale: QKV weight scale tensor for quantization
        qkv_in_scale: QKV input scale tensor for static quantization
        sbm: Optional SBUF manager for memory allocation control
        use_auto_allocation: Whether to use automatic SBUF allocation. Default: False
        qk_norm_pre_rope_q_norm: Normalization type for Q heads before RoPE
        qk_norm_pre_rope_k_norm: Normalization type for K heads before RoPE
        qk_norm_pre_rope_eps: Epsilon for pre-RoPE QK-norm. Default: 1e-6
        qk_norm_pre_rope_gamma_fused_in_rope_caches: Whether gamma is pre-multiplied
            into cos/sin caches for pre-RoPE norm. Default: False
        qk_norm_pre_rope_q_gamma: Per-head gamma for Q heads applied before RoPE [1, d_head]
        qk_norm_pre_rope_k_gamma: Per-head gamma for K heads applied before RoPE [1, d_head]
        qk_norm_pre_rope_q_beta: Per-head beta for Q heads applied before RoPE [1, d_head]
        qk_norm_pre_rope_k_beta: Per-head beta for K heads applied before RoPE [1, d_head]
        qk_norm_post_rope_q_norm: Normalization type for Q heads after RoPE
        qk_norm_post_rope_k_norm: Normalization type for K heads after RoPE
        qk_norm_post_rope_eps: Epsilon for post-RoPE QK-norm. Default: 1e-6
        qk_norm_post_rope_gamma_fused_in_rope_caches: Whether gamma is pre-multiplied
            into cos/sin caches for post-RoPE norm. Default: False
        qk_norm_post_rope_q_gamma: Per-head gamma for Q heads applied after RoPE [1, d_head]
        qk_norm_post_rope_k_gamma: Per-head gamma for K heads applied after RoPE [1, d_head]
        qk_norm_post_rope_q_beta: Per-head beta for Q heads applied after RoPE [1, d_head]
        qk_norm_post_rope_k_beta: Per-head beta for K heads applied after RoPE [1, d_head]

    Returns:
        Output tensor with shape determined by output_layout. With a KV cache
        the call returns ``(q, k_cache, v_cache)``, and output_layout applies to
        Q only. K and V keep the paged cache layout.

    Kernel Constraints (falls back to PyTorch if violated):
        - hidden must be 3D tensor with shape [B, S, H]
        - qkv_weights must be 2D tensor with shape [H, fused_qkv_dim]
        - Hidden dimension H must match weight's first dimension
        - Hidden dimension H must be divisible by 128
        - Hidden dimension H must be <= 24576
        - fused_qkv_dim must be <= 4096
        - mlp_prev and attn_prev must both be provided or neither
        - norm_b is only supported for LAYER_NORM
        - For NBSd layout, d_head must be 128
        - NBdS layout is not kernel-supported (falls back to PyTorch)

    Notes:
        - The kernel automatically selects between decode (TKG) and prefill (CTE)
          implementations based on sequence length threshold
        - RoPE fusion requires cos_cache, sin_cache, num_q_heads, and num_kv_heads
    """

    has_pre_rope = (
        qk_norm_pre_rope_q_norm is not None or qk_norm_pre_rope_k_norm is not None
    )
    has_post_rope = (
        qk_norm_post_rope_q_norm is not None or qk_norm_post_rope_k_norm is not None
    )

    if has_pre_rope or has_post_rope:
        assert num_q_heads is not None and num_kv_heads is not None, (
            "num_q_heads and num_kv_heads must be specified when qk_norm is enabled"
        )

    # Check if kernel can be used
    can_use_kernel = _can_use_qkv_kernel(
        hidden=hidden,
        qkv_weights=qkv_weights,
        mlp_prev=mlp_prev,
        attn_prev=attn_prev,
        d_head=d_head,
        output_layout=output_layout,
        norm_type=norm_type,
        norm_b=norm_b,
        quantization_type=quantization_type,
        qk_norm_pre_rope_q_beta=qk_norm_pre_rope_q_beta,
        qk_norm_pre_rope_k_beta=qk_norm_pre_rope_k_beta,
        qk_norm_post_rope_q_beta=qk_norm_post_rope_q_beta,
        qk_norm_post_rope_k_beta=qk_norm_post_rope_k_beta,
        fused_rope=cos_cache is not None or sin_cache is not None,
    )

    if can_use_kernel:
        # Compute derived boolean flags
        fused_residual_add = mlp_prev is not None and attn_prev is not None
        fused_rope = (
            cos_cache is not None
            and sin_cache is not None
            and num_q_heads is not None
            and num_kv_heads is not None
        )

        # QK-norm and RoPE require d_head; infer from weight shape if not provided
        if d_head is None and num_q_heads is not None and num_kv_heads is not None:
            fused_qkv_dim = qkv_weights.shape[1]
            d_head = fused_qkv_dim // (num_q_heads + 2 * num_kv_heads)

        wrapped_kernel = wrap_nki(_torch_compatible_qkv_kernel)

        # Reshape vLLM-Neuron's 4D paged K/V cache to the kernel's documented
        # 3D contract. The kernel internally does
        # ``cache_hbm.reshape((num_blocks*block_size, kv_dim))`` (see
        # ``nkilib/core/qkv/qkv_cte.py:762``) which assumes the input is
        # row-major-3D ``[num_blocks, block_size, num_kv_heads*d_head]``.
        # vLLM-Neuron's canonical buffer is 4D
        # ``[num_blocks, num_kv_heads, block_size, d_head]``; the strides
        # only coincide when ``num_kv_heads == 1``. Without this reshape,
        # ``num_kv_heads > 1`` configs (e.g. Llama-3.2-1B at TP=2) get K/V
        # written at wrong slot offsets because the kernel walks the buffer
        # with the wrong strides.
        # contiguous() forces the row-major-3D copy so the kernel's DMA
        # strides match; the writeback below mirrors the post-kernel state
        # back into the original 4D buffers so downstream readers
        # (segmented_attention, FX aliasing pass) see consistent updates.
        # Packed-FP8 K cache is 5D per-head
        # ``[num_blocks, num_kv_heads, block_size//2, d_head, 2]`` — it already
        # matches the kernel's packed-write contract, so it must be passed
        # through as-is (NOT reshaped). Only the bf16 unpacked 4D paged cache
        # needs the 4D->3D bridge below.
        _reshape_kv_4d_to_3d = (
            use_block_kv
            and k_cache is not None
            and k_cache.dim() == 4
            and not transpose_k_cache
            and not fp8_packed
        )
        if _reshape_kv_4d_to_3d:
            nb_, nkh_, bs_, dh_ = k_cache.shape
            k_cache_arg = k_cache.permute(0, 2, 1, 3).reshape(nb_, bs_, nkh_ * dh_)
            v_cache_arg = v_cache.permute(0, 2, 1, 3).reshape(nb_, bs_, nkh_ * dh_)
        else:
            k_cache_arg = k_cache
            v_cache_arg = v_cache

        # Packed-FP8 qkv_cte loads the KV quant scales with a fixed
        # ``dma_copy(src=k_scale_hbm[0:pmax, 0:1])`` (qkv_cte.py), so the scale
        # must be HBM shape ``[pmax, 1]`` (per-partition, broadcast) — a scalar
        # ``[1, 1]`` triggers an out-of-bounds on dim 0. (The decode kernel
        # accepts ``[1,1]`` or ``[pmax,1]``; qkv_cte requires ``[pmax,1]``.)
        # Broadcast the per-tensor scalar the caller passes to ``[pmax, 1]``.
        k_scale_arg, v_scale_arg = k_scale, v_scale
        if fp8_packed:
            _PMAX = 128
            if k_scale_arg is not None and k_scale_arg.shape[0] != _PMAX:
                k_scale_arg = k_scale_arg.reshape(1, 1).expand(_PMAX, 1).contiguous()
            if v_scale_arg is not None and v_scale_arg.shape[0] != _PMAX:
                v_scale_arg = v_scale_arg.reshape(1, 1).expand(_PMAX, 1).contiguous()

        # Translate the torch FP8 ``kv_dtype`` to the NKI dtype for the kernel
        # dispatch (see ``_TORCH_NKI_KV_DTYPE_MAP`` above for why).
        kv_dtype_arg = _TORCH_NKI_KV_DTYPE_MAP.get(kv_dtype, kv_dtype)

        result = wrapped_kernel[get_logical_nc_config()](
            hidden=hidden,
            qkv_weights=qkv_weights,
            norm_weights=norm_weights,
            mlp_prev=mlp_prev,
            attn_prev=attn_prev,
            d_head=d_head,
            output_layout=output_layout,
            eps=eps,
            norm_type=norm_type,
            use_dma_transpose=use_dma_transpose,
            store_output_in_sbuf=store_output_in_sbuf,
            bias=bias,
            norm_b=norm_b,
            hidden_actual=hidden_actual,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            quantization_type=quantization_type,
            qkv_w_scale=qkv_w_scale,
            qkv_in_scale=qkv_in_scale,
            weight_layout=weight_layout,
            fused_residual_add=fused_residual_add,
            fused_rope=fused_rope,
            qk_norm_pre_rope_q_norm=qk_norm_pre_rope_q_norm,
            qk_norm_pre_rope_k_norm=qk_norm_pre_rope_k_norm,
            qk_norm_pre_rope_eps=qk_norm_pre_rope_eps,
            qk_norm_pre_rope_gamma_fused_in_rope_caches=qk_norm_pre_rope_gamma_fused_in_rope_caches,
            qk_norm_pre_rope_q_gamma=qk_norm_pre_rope_q_gamma,
            qk_norm_pre_rope_k_gamma=qk_norm_pre_rope_k_gamma,
            qk_norm_pre_rope_q_beta=qk_norm_pre_rope_q_beta,
            qk_norm_pre_rope_k_beta=qk_norm_pre_rope_k_beta,
            qk_norm_post_rope_q_norm=qk_norm_post_rope_q_norm,
            qk_norm_post_rope_k_norm=qk_norm_post_rope_k_norm,
            qk_norm_post_rope_eps=qk_norm_post_rope_eps,
            qk_norm_post_rope_gamma_fused_in_rope_caches=qk_norm_post_rope_gamma_fused_in_rope_caches,
            qk_norm_post_rope_q_gamma=qk_norm_post_rope_q_gamma,
            qk_norm_post_rope_k_gamma=qk_norm_post_rope_k_gamma,
            qk_norm_post_rope_q_beta=qk_norm_post_rope_q_beta,
            qk_norm_post_rope_k_beta=qk_norm_post_rope_k_beta,
            sbm_lower_bound=sbm_lower_bound,
            sbm_upper_bound=sbm_upper_bound,
            sbm_use_auto_alloc=sbm_use_auto_alloc,
            sbm_default_stack_alloc=sbm_default_stack_alloc,
            use_auto_allocation=use_auto_allocation,
            # In-kernel block KV cache write
            k_cache=k_cache_arg,
            v_cache=v_cache_arg,
            k_scale=k_scale_arg,
            v_scale=v_scale_arg,
            fp8_max=fp8_max,
            fp8_min=fp8_min,
            kv_dtype=kv_dtype_arg,
            use_block_kv=use_block_kv,
            transpose_k_cache=transpose_k_cache,
            fp8_packed=fp8_packed,
            block_size=block_size,
            slot_mapping=slot_mapping,
        )

        # When we reshaped K/V into the kernel's 3D contract, the kernel
        # wrote into the temporary 3D copy, NOT into the original 4D
        # buffers. Mirror the writes back so downstream readers see the
        # post-write state. Return the original 4D buffers so the FX
        # aliasing pass rewires to the buffers segmented_attention reads.
        if _reshape_kv_4d_to_3d:
            q_out, k_3d_post, v_3d_post = result
            k_cache.copy_(k_3d_post.reshape(nb_, bs_, nkh_, dh_).permute(0, 2, 1, 3))
            v_cache.copy_(v_3d_post.reshape(nb_, bs_, nkh_, dh_).permute(0, 2, 1, 3))
            return q_out, k_cache, v_cache
        return result
    else:
        if quantization_type != QuantizationType.NONE:
            raise NotImplementedError(
                "QKV Projection torch fallback currently does not support quantization"
            )

        # In-kernel block KV cache write path: defer to upstream
        # nkilib torch reference, which scatters K/V into the paged
        # cache at slot_mapping positions and returns the 3-tuple
        # (q, k_cache, v_cache) the caller expects. The standard
        # torch fallback below returns a single concatenated QKV
        # tensor — incompatible with the in-kernel-write call site.
        if use_block_kv:
            return _torch_qkv_in_kernel_cache_write_impl(
                hidden=hidden,
                qkv_weights=qkv_weights,
                bias=bias,
                d_head=d_head,
                output_layout=output_layout,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                k_cache=k_cache,
                v_cache=v_cache,
                k_scale=k_scale,
                v_scale=v_scale,
                fp8_max=fp8_max,
                fp8_min=fp8_min,
                kv_dtype=kv_dtype,
                transpose_k_cache=transpose_k_cache,
                fp8_packed=fp8_packed,
                block_size=block_size,
                slot_mapping=slot_mapping,
                # Forward fused-norm + QK-norm so the fallback matches the kernel
                # (e.g. Qwen3's per-head RMS QK-norm before RoPE). Omitting these
                # silently corrupts Q and the cached K.
                norm_weights=norm_weights,
                norm_type=norm_type,
                eps=eps,
                qk_norm_pre_rope_q_norm=qk_norm_pre_rope_q_norm,
                qk_norm_pre_rope_k_norm=qk_norm_pre_rope_k_norm,
                qk_norm_pre_rope_eps=qk_norm_pre_rope_eps,
                qk_norm_pre_rope_gamma_fused_in_rope_caches=qk_norm_pre_rope_gamma_fused_in_rope_caches,
                qk_norm_pre_rope_q_gamma=qk_norm_pre_rope_q_gamma,
                qk_norm_pre_rope_k_gamma=qk_norm_pre_rope_k_gamma,
                qk_norm_pre_rope_q_beta=qk_norm_pre_rope_q_beta,
                qk_norm_pre_rope_k_beta=qk_norm_pre_rope_k_beta,
                qk_norm_post_rope_q_norm=qk_norm_post_rope_q_norm,
                qk_norm_post_rope_k_norm=qk_norm_post_rope_k_norm,
                qk_norm_post_rope_eps=qk_norm_post_rope_eps,
                qk_norm_post_rope_gamma_fused_in_rope_caches=qk_norm_post_rope_gamma_fused_in_rope_caches,
                qk_norm_post_rope_q_gamma=qk_norm_post_rope_q_gamma,
                qk_norm_post_rope_k_gamma=qk_norm_post_rope_k_gamma,
                qk_norm_post_rope_q_beta=qk_norm_post_rope_q_beta,
                qk_norm_post_rope_k_beta=qk_norm_post_rope_k_beta,
            )

        return _torch_qkv_impl(
            hidden=hidden,
            qkv_weights=qkv_weights,
            norm_weights=norm_weights,
            mlp_prev=mlp_prev,
            attn_prev=attn_prev,
            d_head=d_head,
            output_layout=output_layout,
            eps=eps,
            norm_type=norm_type,
            bias=bias,
            norm_b=norm_b,
            hidden_actual=hidden_actual,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            qk_norm_pre_rope_q_norm=qk_norm_pre_rope_q_norm,
            qk_norm_pre_rope_k_norm=qk_norm_pre_rope_k_norm,
            qk_norm_pre_rope_eps=qk_norm_pre_rope_eps,
            qk_norm_pre_rope_gamma_fused_in_rope_caches=qk_norm_pre_rope_gamma_fused_in_rope_caches,
            qk_norm_pre_rope_q_gamma=qk_norm_pre_rope_q_gamma,
            qk_norm_pre_rope_k_gamma=qk_norm_pre_rope_k_gamma,
            qk_norm_pre_rope_q_beta=qk_norm_pre_rope_q_beta,
            qk_norm_pre_rope_k_beta=qk_norm_pre_rope_k_beta,
            qk_norm_post_rope_q_norm=qk_norm_post_rope_q_norm,
            qk_norm_post_rope_k_norm=qk_norm_post_rope_k_norm,
            qk_norm_post_rope_eps=qk_norm_post_rope_eps,
            qk_norm_post_rope_gamma_fused_in_rope_caches=qk_norm_post_rope_gamma_fused_in_rope_caches,
            qk_norm_post_rope_q_gamma=qk_norm_post_rope_q_gamma,
            qk_norm_post_rope_k_gamma=qk_norm_post_rope_k_gamma,
            qk_norm_post_rope_q_beta=qk_norm_post_rope_q_beta,
            qk_norm_post_rope_k_beta=qk_norm_post_rope_k_beta,
        )
