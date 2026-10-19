# SPDX-License-Identifier: Apache-2.0
"""Torch entry point for the in-tree Qwen3-32B MXFP8 decode MLP kernel.

Wraps the ``@nki.jit`` kernel in ``vllm_neuron._kernels.mlp`` so the Qwen3-32B
MXFP8 decode path can call it from the traced model graph, the same way
``mlp_deepseek_mx`` wraps its NKL kernel. It launches at the process LNC degree:
grid 2 at LNC2 shards the MLP across the two cores (H contraction for gate/up with
a cross-core reduce, H output for down), matching the nkilib decode MLP; grid 1 at
LNC1 runs the single-core path. Each tensor-parallel rank runs it on its own weight
shard, and vLLM does the cross-rank all-reduce.

This module runs the in-tree kernel in the real decode path. It is on by default;
disable it with ``VLLM_NEURON_QWEN3_INTREE_MLP=0`` (see ``model_mxfp8.py``).
"""

from typing import Optional

import nki
import nki.language as nl
from torch import Tensor

from vllm_neuron._kernels.mlp.qwen3_32b_mxfp8_mlp_tkg import (
    _RING_DEPTH_DEFAULT,
    _mlp_tkg_impl,
)

_P = 128
_TILE = 512

__all__ = ["qwen3_32b_mxfp8_mlp_tkg_decode"]


def _to_kernel_h_layout(hidden: Tensor) -> Tensor:
    """Permute a natural ``[T, H]`` activation into the kernel's x4 H layout.

    The kernel reads ``hidden[p, t512, t, q] = x[t, t512 * 512 + p * 4 + q]``, so
    one permute of the natural tensor gives the layout. The caller must NOT run
    ``pre_shuffle_h`` first: that shuffle targets the nkilib decode kernel, and
    this permute then undid its H order, so the pair cost one ``[T, H]`` HBM write
    and read for each layer.

    Args:
        hidden: bf16 ``[T, H]`` activation in natural H order.

    Returns:
        A ``[128, H // 512, T, 4]`` view of the same values.

    Example:
        >>> import torch
        >>> _to_kernel_h_layout(torch.zeros(2, 1024)).shape
        torch.Size([128, 2, 2, 4])
    """
    t, h = hidden.shape
    n_h512 = h // _TILE
    q_width = _TILE // _P
    return hidden.reshape(t, n_h512, _P, q_width).permute(2, 1, 0, 3)


@nki.jit(mode="trace")
def _qwen3_mlp_trace(
    hidden: nl.ndarray,
    gate_w: nl.ndarray,
    up_w: nl.ndarray,
    down_w: nl.ndarray,
    gate_s: nl.ndarray,
    up_s: nl.ndarray,
    down_s: nl.ndarray,
    gamma: Optional[nl.ndarray] = None,
    eps: float = 1e-6,
) -> nl.ndarray:
    """Trace-mode entry that runs the in-tree MLP kernel body.

    Args:
        hidden: bf16 ``[128, H // 512, T, 4]`` activation. It is the raw pre-norm
            activation when ``gamma`` is given, else already normalized.
        gate_w: uint32 ``[128, H // 512, I]`` MXFP8 gate weight.
        up_w: uint32 ``[128, H // 512, I]`` MXFP8 up weight.
        down_w: uint32 ``[128, I // 512, H]`` MXFP8 down weight.
        gate_s: uint8 ``[128, H // 512, I]`` pre-scattered E8M0 gate scale.
        up_s: uint8 ``[128, H // 512, I]`` pre-scattered E8M0 up scale.
        down_s: uint8 ``[128, I // 512, H]`` pre-scattered E8M0 down scale.
        gamma: Optional bf16 ``[128, H // 512, 4]`` RMSNorm weight in the x4 H
            layout. It drives the fused RMSNorm.
        eps: RMSNorm epsilon. It applies only when ``gamma`` is given.

    Returns:
        bf16 ``[T, H]`` MLP output.

    Example:
        >>> # traced by wrap_nki; not called directly
    """
    return _mlp_tkg_impl(
        hidden,
        gate_w,
        up_w,
        down_w,
        gate_s,
        up_s,
        down_s,
        _RING_DEPTH_DEFAULT,
        gamma=gamma,
        eps=eps,
    )


def qwen3_32b_mxfp8_mlp_tkg_decode(
    hidden: Tensor,
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
    gate_s: Tensor,
    up_s: Tensor,
    down_s: Tensor,
    ln_w: Tensor | None = None,
    eps: float | None = None,
) -> Tensor:
    """Run the in-tree Qwen3-32B MXFP8 decode MLP for one tensor-parallel rank.

    Reshapes the natural ``[T, H]`` activation into the kernel's H-partitioned x4
    layout with one permute, then invokes the kernel through ``wrap_nki`` at the
    process LNC degree (grid 2 at LNC2 shards across the two cores; grid 1 at LNC1
    runs the single-core path). The caller passes the activation in natural H
    order: this function derives the x4 layout itself, so the caller must NOT run
    ``pre_shuffle_h`` first (that shuffle is for the nkilib decode kernel). The
    decode TKG weights and scales the Qwen3-32B MXFP8 model already produces match
    the kernel's input contract, so they pass through unchanged.

    Args:
        hidden: bf16 ``[T, H]`` decode activation in natural H order.
        gate_w: uint32 ``[128, H // 512, I]`` MXFP8 gate weight.
        up_w: uint32 ``[128, H // 512, I]`` MXFP8 up weight.
        down_w: uint32 ``[128, I // 512, H]`` MXFP8 down weight.
        gate_s: uint8 ``[128, H // 512, I]`` pre-scattered E8M0 gate scale.
        up_s: uint8 ``[128, H // 512, I]`` pre-scattered E8M0 up scale.
        down_s: uint8 ``[128, I // 512, H]`` pre-scattered E8M0 down scale.
        ln_w: bf16 ``[128, H // 512, 4]`` RMSNorm weight, in the kernel layout
            already. The model builds it in
            ``Qwen3DecoderLayerMxfp8._make_gamma_kernel_layout`` and holds it in
            ``post_attention_layernorm_gamma_shuffled``. It pairs with ``hidden``
            on the flat H index. When given, the kernel fuses the RMSNorm on the
            raw ``hidden``; when ``None``, ``hidden`` is already normalized. Pass
            ``eps`` with it.
        eps: RMSNorm epsilon. Required when ``ln_w`` is given, ignored otherwise.

    Returns:
        bf16 ``[T, H]`` per-rank MLP output (the caller does the all-reduce).

    Example:
        >>> # out = qwen3_32b_mxfp8_mlp_tkg_decode(hidden, gw, uw, dw, gs, us, ds)
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from vllm_neuron.utils.hardware_config import get_logical_nc_config

    assert (ln_w is None) == (eps is None), (
        "ln_w and eps must be passed together to fuse the RMSNorm, or both unset."
    )

    _, h = hidden.shape
    n_h512 = h // _TILE
    q_width = _TILE // _P
    # Check the gamma shape against the activation. The kernel pairs gamma with the
    # activation on the flat H index, so a wrong shape gives a silently wrong norm.
    # Refuse it here rather than in one caller.
    assert ln_w is None or tuple(ln_w.shape) == (_P, n_h512, q_width), (
        f"ln_w must be the kernel-layout gamma with shape "
        f"({_P}, {n_h512}, {q_width}) to match the activation H={h}; got "
        f"{tuple(ln_w.shape)}. The model builds this layout in "
        "_make_gamma_kernel_layout."
    )
    # One permute, not two. The caller no longer applies pre_shuffle_h first, so
    # this is the only transpose on the decode path.
    #
    # This does NOT fix the NCC_ILSA901 failure of the fused path at T=2 and T=4.
    # That failure needs the residual add in the caller's graph, and it survives
    # every spelling of this permute. The model declines the fusion at those
    # counts instead -- see _ILSA901_TOKEN_COUNTS in model_mxfp8.py.
    hidden_4d = _to_kernel_h_layout(hidden)

    # Launch at the process LNC degree: grid 2 at LNC2 (shard the MLP across the two
    # cores, matching the nkilib decode MLP path) and grid 1 at LNC1 (single-core
    # path). The kernel reads program_id/num_programs and reduces to a single shard
    # at grid 1, so both grids are supported. The traced grid must match the LNC the
    # graph compiles for, else neuronx-cc rejects the kernel (NCC_ENKI003).
    lnc = get_logical_nc_config()
    kernel_kwargs = {
        "hidden": hidden_4d,
        "gate_w": gate_w,
        "up_w": up_w,
        "down_w": down_w,
        "gate_s": gate_s,
        "up_s": up_s,
        "down_s": down_s,
    }
    if ln_w is not None:
        # Pass gamma straight through. The model stores it in the kernel layout
        # already, so the graph holds no gamma transpose, which keeps a reshape and
        # a permute off every decode step.
        #
        # Gamma is not the tensor that fails to compile. The penguin IR names the
        # ACTIVATION permute above as the failing transpose, and removing the gamma
        # transpose did not clear NCC_ILSA901.
        kernel_kwargs["gamma"] = ln_w
        kernel_kwargs["eps"] = eps
    result = wrap_nki(_qwen3_mlp_trace)[lnc](**kernel_kwargs)
    return result[0] if isinstance(result, (tuple, list)) else result
