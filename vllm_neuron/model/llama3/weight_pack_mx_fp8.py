# SPDX-License-Identifier: Apache-2.0
"""Weight packing helpers for the Llama3 STATIC_MX (Trn3) path.

Implements the STATIC_MX x4-pack on-disk layout conventions for Llama3.

Forward direction of the layouts documented in
:class:`nkilib.core.utils.common_types.QKVWeightLayout`:

- ``MX_INTERLEAVED``: a row reorder (matching the interleaved layout
  produced by DMA transpose on the input) followed by a reshape to the
  3-D ``[H//4, I, 4]`` fp8 layout the kernel consumes.

The o-proj path does **not** pack — its kernel reshapes a contiguous
``[N*D, H]`` operand to ``[N*D//4, H, 4]`` via ``TensorView`` at access
time, so the host only applies a byte-shuffle (no dtype change).
"""

from __future__ import annotations

import torch


_FP8_DTYPE = torch.float8_e4m3fn


def _mx_interleaved_h_perm(H: int) -> torch.Tensor:
    """Row permutation for ``MX_INTERLEAVED``.

    Per ``QKVWeightLayout.MX_INTERLEAVED``::

        h_idx = arange(H).reshape(2, H//4, 2).transpose(1, 0, 2).reshape(H)
    """
    assert H % 4 == 0, f"H={H} must be divisible by 4"
    return torch.arange(H).reshape(2, H // 4, 2).transpose(0, 1).reshape(H).contiguous()


def qkv_weight_pack_mx_interleaved(w_HI: torch.Tensor) -> torch.Tensor:
    """Forward direction of ``MX_INTERLEAVED`` (DMA-transpose MX path).

    Reorder rows by ``h_idx``, then reshape to 3-D ``[H//4, I, 4]`` fp8.

    Args:
        w_HI: ``torch.float8_e4m3fn``, shape ``[H, I]`` with ``H % 4 == 0``.

    Returns:
        ``torch.float8_e4m3fn``, shape ``[H//4, I, 4]``.
    """
    H, I = w_HI.shape
    h_idx = _mx_interleaved_h_perm(H)
    permuted_bytes = w_HI.view(torch.uint8)[h_idx, :].contiguous()
    permuted = permuted_bytes.view(_FP8_DTYPE)
    return permuted.reshape(H // 4, 4, I).permute(0, 2, 1).contiguous()


def mx_shuffle_o_proj(w_NDH: torch.Tensor) -> torch.Tensor:
    """Apply the STATIC_MX byte shuffle to a 2D ``[N*D, H]`` fp8 tensor.

    The o-proj CTE STATIC_MX kernel views the weight as
    ``[N*D//4, H, 4]`` (an x4-strided 3D operand) and slices on H. To
    produce that layout from a contiguous ``[N*D, H]`` source, the host
    applies::

        reshape(N*D//4, 4, H).transpose(0, 2, 1)  -> [N*D//4, H, 4]

    then reshapes back to ``[N*D, H]``. Shape and dtype are unchanged;
    only byte order shifts. The transpose is routed through a ``uint8``
    byte view because some torch CPU builds lack ``transpose`` for
    ``float8_e4m3fn``; at the byte level it is bit-equivalent.
    """
    assert w_NDH.dim() == 2, f"expected 2D [N*D, H], got {tuple(w_NDH.shape)}"
    nd, h = w_NDH.shape
    assert nd % 4 == 0, f"N*D={nd} must be divisible by 4 for STATIC_MX o-proj"
    bytes_view = w_NDH.contiguous().view(torch.uint8)
    shuffled = (
        bytes_view.reshape(nd // 4, 4, h).transpose(1, 2).reshape(nd, h).contiguous()
    )
    return shuffled.view(_FP8_DTYPE)
