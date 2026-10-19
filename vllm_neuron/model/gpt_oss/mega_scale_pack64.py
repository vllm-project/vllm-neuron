# SPDX-License-Identifier: Apache-2.0
"""The PACK64 gate/up MX scale layout: which layout to write, and the remap that writes it.

The kernel chooses the layout, so the decision function takes the kernel's own flag reader
instead of reading the environment. A scale in the layout the kernel does not expect is
silent garbage into the matmul rather than an error, so the decision is taken once, here.
"""

from typing import Callable

import torch

SCALE_P16 = 16  # scale partitions in the unpacked gate/up layout
SCALE_GATE_UP = 2
SCALE_ROWS_PER_QUADRANT = 4  # SCALE_ROWS_PER_QUADRANT in the kernel package
SCALE_FREE_PER_ROW = 128


def pack64_gate_up_scale(scale: torch.Tensor) -> torch.Tensor:
    """Remap a gate/up MX scale tensor to the kernel's PACK64 HBM layout.

    ``[E, 16, 2, H/512, I]`` becomes ``[E, 2, 64, H/512, I/4]``.

    The packed load reads 16 scale rows for each SBUF quadrant in place of 4.
    This moves the 4-wide I factor out of the free dimension and into the
    partitions: for data free index ``i = tile_i*512 + k*128 + j``, the scale
    goes to partition ``16*q + 4*k + r`` and free index ``(h, tile_i*128 + j)``.
    The remap is exact and invertible. It mirrors ``pack64_gate_up_scale`` in the
    kernel package (``test/utils/mx_utils.py``), which is the reference.

    Args:
        scale: The scale tensor, with trailing dimensions ``(16, 2, H/512, I)``.

    Returns:
        The contiguous tensor, with those dimensions replaced by
        ``(2, 64, H/512, I/4)``.

    Example:
        >>> import torch
        >>> pack64_gate_up_scale(torch.zeros(1, 16, 2, 6, 3072)).shape
        torch.Size([1, 2, 64, 6, 768])
    """
    if scale.ndim < 5:
        raise ValueError(
            "The unpacked gate/up MX scale needs at least 5 dimensions "
            f"[..., E, 16, 2, H/512, I]; got {tuple(scale.shape)}."
        )

    *lead, p16, g, n_h512, i_dim = scale.shape
    if p16 != SCALE_P16:
        raise ValueError(
            f"The gate/up MX scale has {p16} partitions; the PACK64 remap needs "
            f"{SCALE_P16} scale partitions in [..., E, 16, 2, H/512, I]."
        )
    if g != SCALE_GATE_UP:
        raise ValueError(
            f"The gate/up MX scale has a gate/up axis of {g}; the unpacked "
            f"layout needs a gate/up axis of {SCALE_GATE_UP}."
        )

    q_n = p16 // SCALE_ROWS_PER_QUADRANT
    r_n = k_n = SCALE_ROWS_PER_QUADRANT
    j_n = SCALE_FREE_PER_ROW
    tile_i = k_n * j_n
    if i_dim % tile_i != 0:
        raise ValueError(
            f"I={i_dim} is not a multiple of {tile_i}. The PACK64 scale layout "
            f"needs complete {tile_i}-wide I tiles."
        )
    n_ti = i_dim // tile_i

    n_lead = len(lead)
    x = scale.reshape(*lead, q_n, r_n, g, n_h512, n_ti, k_n, j_n)
    # Relative axes are 0=q 1=r 2=g 3=n_h512 4=n_ti 5=k 6=j.
    # The target order is g, q, k, r, n_h512, n_ti, j.
    perm = tuple(range(n_lead)) + tuple(n_lead + a for a in (2, 0, 5, 1, 3, 4, 6))
    x = x.permute(*perm).contiguous()
    return x.reshape(*lead, g, q_n * k_n * r_n, n_h512, n_ti * j_n)


def pack64_gate_up_scale_transform(scale: torch.Tensor, layer=None) -> torch.Tensor:
    """Remap the scale for a ``_post_transforms`` entry, which passes the owning layer.

    The layer is not used. This exists only to give the remap the ``fn(tensor, layer)`` signature
    that the weight loader calls.

    Example:
        >>> import torch
        >>> pack64_gate_up_scale_transform(torch.zeros(1, 16, 2, 6, 3072), None).ndim
        5
    """
    return pack64_gate_up_scale(scale)


def kernel_packs_gate_up_scale(
    flag_reader: Callable[[str], bool] | None,
) -> bool:
    """Return True when the kernel reads the gate/up MX scale in the PACK64 layout."""
    # Fail closed. An nkilib too old to give a flag reader still has a kernel, and that
    # kernel reads the unpacked layout. A packed write into it is silent garbage, because
    # the kernel tests the layout in one direction only.
    if flag_reader is None:
        return False
    return flag_reader("VLLM_NEURON_MEGA_SCALE_PACK64")
