# SPDX-License-Identifier: Apache-2.0
"""SwiGLU (SiLU-and-multiply) NKI kernel.

Computes the gated activation used by the MLP of most decoder-only models::

    out = silu(x[:, :H]) * x[:, H:]

where the input holds the ``gate`` and ``up`` projections concatenated along
the last dimension.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import torch

# Partition-dim tile height. nl.tile_size.pmax is not a plain Python int, and
# the compiler requires Python ints for SBUF tile shapes.
P_MAX = 128

# Free-dim tile width. Keeps one gate tile + one up tile + the result resident
# in SBUF at once for the dtypes this kernel accepts.
F_TILE_SIZE = 512


@nki.jit
def silu_and_mul(x: nl.ndarray) -> nl.ndarray:
    """Compute ``silu(gate) * up`` for a gate/up-concatenated input.

    Dimensions:
        B: outer (token) dimension, any size — tiled by ``P_MAX``.
        H: half of the last dimension, any size — tiled by ``F_TILE_SIZE``.

    Args:
        x: Input HBM tensor of shape ``[B, 2 * H]``. The first ``H`` columns are
            the gate projection, the last ``H`` columns the up projection.

    Returns:
        Output HBM tensor of shape ``[B, H]`` and the same dtype as ``x``.

    Pseudocode:
        gate, up = x[:, :H], x[:, H:]
        for each (p_tile, f_tile):
            g = load(gate[p_tile, f_tile])
            u = load(up[p_tile, f_tile])
            store(out[p_tile, f_tile], silu(g) * u)
    """
    batch, two_hidden = x.shape
    hidden = two_hidden // 2

    out = nl.ndarray((batch, hidden), dtype=x.dtype, buffer=nl.shared_hbm)

    num_p_tiles = (batch + P_MAX - 1) // P_MAX
    num_f_tiles = (hidden + F_TILE_SIZE - 1) // F_TILE_SIZE

    for p_idx in nl.affine_range(num_p_tiles):
        p_start = p_idx * P_MAX
        p_size = min(P_MAX, batch - p_start)

        for f_idx in nl.affine_range(num_f_tiles):
            f_start = f_idx * F_TILE_SIZE
            f_size = min(F_TILE_SIZE, hidden - f_start)

            gate_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)
            up_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)

            nisa.dma_copy(
                dst=gate_sb[0:p_size, 0:f_size],
                src=x[p_start : p_start + p_size, f_start : f_start + f_size],
            )
            nisa.dma_copy(
                dst=up_sb[0:p_size, 0:f_size],
                src=x[
                    p_start : p_start + p_size,
                    hidden + f_start : hidden + f_start + f_size,
                ],
            )

            # silu(g) = g * sigmoid(g), then multiply by the up projection.
            # Both steps run on the activation/vector engines in float32 and the
            # result is cast back to the input dtype on store.
            act_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=act_sb[0:p_size, 0:f_size],
                data=gate_sb[0:p_size, 0:f_size],
                op=nl.silu,
            )

            result_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=result_sb[0:p_size, 0:f_size],
                data1=act_sb[0:p_size, 0:f_size],
                data2=up_sb[0:p_size, 0:f_size],
                op=nl.multiply,
            )

            nisa.dma_copy(
                dst=out[p_start : p_start + p_size, f_start : f_start + f_size],
                src=result_sb[0:p_size, 0:f_size],
            )

    return out


def silu_and_mul_torch_ref(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Torch reference for :func:`silu_and_mul`.

    The signature must match the kernel's (``nkilib_testing`` asserts this), and
    the return must be a dict keyed by output name.

    Args:
        x: Input tensor of shape ``[B, 2 * H]`` (gate concatenated with up).

    Returns:
        Dict with key ``"out"`` holding a ``[B, H]`` tensor.

    Example:
        >>> import torch
        >>> x = torch.ones(1, 4)
        >>> silu_and_mul_torch_ref(x=x)["out"].shape
        torch.Size([1, 2])
    """
    hidden = x.shape[-1] // 2
    gate, up = x[..., :hidden], x[..., hidden:]
    # Compute in float32 for a stable oracle, then cast back to the kernel's
    # output dtype so tolerances reflect only genuine kernel error.
    out = (torch.nn.functional.silu(gate.float()) * up.float()).to(x.dtype)
    return {"out": out}
