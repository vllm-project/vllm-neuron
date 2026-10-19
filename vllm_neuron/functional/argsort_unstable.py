# SPDX-License-Identifier: Apache-2.0
"""argsort_unstable functional API."""

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert

import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel

_ELEMS_PER_PASS = 8

_NEG_INF = float("-inf")
_POS_INF = float("inf")


def argsort_unstable(
    data: torch.Tensor,
    descending: bool = False,
) -> torch.Tensor:
    """Perform unstable argsort on a 1D or [1, N] input tensor.

    Elements with equal values may appear in any order relative to their
    original positions. Vectorization along a batch dimension is not supported.

    Args:
        data (torch.Tensor): 1D [N] or 2D [1, N] int32/float32 tensor to sort.
        descending (bool): When True, return indices for descending order.
            Defaults to False (ascending).

    Returns:
        torch.Tensor: Same shape as input, int32 tensor of argsort indices.

    Example:
        >>> data = torch.tensor([[5, 2, 5, 3]], dtype=torch.int32)
        >>> indices = argsort_unstable(data)
        >>> # indices = [[1, 3, 0, 2]] or [[1, 3, 2, 0]] (unstable for ties)
    """
    _validate_inputs(data)

    if _can_use_kernel(data):
        wrapped = wrap_nki(_argsort_unstable_nki)
        return wrapped[2](data=data, descending=descending, output_in_sbuf=False)
    else:
        return _torch_argsort_unstable(data, descending)


def _validate_inputs(data: torch.Tensor) -> None:
    """Validate inputs for argsort_unstable."""
    assert data.ndim in (1, 2), (
        f"argsort_unstable only supports 1D [N] or 2D [1, N] input, got shape {data.shape}"
    )
    N = data.shape[-1]
    assert N <= 2**31 - 1, (
        f"argsort_unstable requires N <= int32 max ({2**31 - 1}), got {N=}"
    )


def _can_use_kernel(data: torch.Tensor) -> bool:
    """Check if the NKI kernel can be used.

    Constraints:
    - Data must be fp32 or int32
    """
    if not can_run_kernel(data):
        return False

    if data.dtype not in (torch.float32, torch.int32):
        return False

    return True


def _torch_argsort_unstable(
    data: torch.Tensor, descending: bool = False
) -> torch.Tensor:
    """Neuron-compilable torch argsort matching the NKI kernel behavior.

    Uses argmax on masked comparisons, with N/8 passes.

    Args:
        data: [N] or [1, N] tensor.
        descending: sort direction.

    Returns:
        Same shape as input, int32 tensor of argsort indices.
    """
    is_1d = data.ndim == 1
    x = data.reshape(-1).to(torch.float32)
    N = x.shape[0]
    device = x.device

    # Pad to multiple of 8
    N_padded = ((N + _ELEMS_PER_PASS - 1) // _ELEMS_PER_PASS) * _ELEMS_PER_PASS
    if N_padded > N:
        sentinel = _POS_INF if descending else _NEG_INF
        x = torch.cat([x, torch.full((N_padded - N,), sentinel, device=device)])

    num_passes = N_padded // _ELEMS_PER_PASS
    indices = torch.zeros(N_padded, dtype=torch.int32, device=device)

    # In each pass, find the 8 largest values and then the top 8's indices
    for pass_idx in range(num_passes):
        top_vals, _ = torch.topk(x, _ELEMS_PER_PASS, largest=True, sorted=True)

        for val_idx in range(_ELEMS_PER_PASS - 1, -1, -1):
            val = top_vals[val_idx]
            # Find first position matching val (argmax on equality mask)
            match_mask = (x == val).to(torch.int32)
            pos = torch.argmax(match_mask).to(torch.int32)

            # Store index
            if descending:
                out_pos = _ELEMS_PER_PASS * pass_idx + val_idx
            else:
                out_pos = _ELEMS_PER_PASS * (num_passes - pass_idx) - 1 - val_idx
            indices[out_pos] = pos

            # Replace matched position with -inf
            x = x.scatter(
                0,
                pos.unsqueeze(0).to(torch.int32),
                torch.tensor([_NEG_INF], device=device),
            )

    # Extract valid indices
    if descending:
        result = indices[:N]
    else:
        result = indices[N_padded - N :]

    return result if is_1d else result.unsqueeze(0)


# TODO: upstream into nkilib when kernel is finalized/stable
@nki.jit
def _argsort_unstable_nki(data, descending=False, output_in_sbuf=False):
    """
    Perform unstable argsort on 1D input buffer. Elements with equal values
    may appear in any order relative to their original positions.

    For example:

        data = [5, 2, 5, 3]

        Pass 0 (ascending mode):
          max8 vals = [5, 5, 3, 2, ...]
          nc_match_replace8 matches: pos 0, pos 2, pos 3, pos 1
          reversed output indices: [1, 3, 2, 0]

        Result: indices = [1, 3, 2, 0]  (values in order: 2, 3, 5, 5)
        Indices [2, 0] corresponding to values [5, 5] are not in original order.

    Dimensions:
        N: Number of elements to sort. Must be >= 1.

    Args:
        data (nl.NkiTensor): [1, N] or [N] int32/float32 tensor in HBM or SBUF.
            SBUF input is sorted in place and requires N to be a multiple of 8.
        descending (bool): When True, return indices for descending order. Defaults to ascending.
        output_in_sbuf (bool): When True, return SBUF output. Defaults to HBM output.

    Returns:
        indices (nl.NkiTensor): [1, N] or [N] int32 tensor (matches input layout) containing the argsort indices.
    """

    # Extract shapes, validate
    is_1d = len(data.shape) == 1
    if is_1d:
        N = data.shape[0]
        data = data.reshape((1, N))
    else:
        N = data.shape[1]
        kernel_assert(
            data.shape[0] == 1, f"Expected data.shape=[1, N] or [N], got {data.shape}"
        )
    kernel_assert(N >= 1, f"Expected N >= 1, got {N}")
    kernel_assert(N <= 2147483647, f"Expected N <= INT32_MAX, got {N}")

    # Pad N to next multiple of 8 with sentinel values that sort to the end
    N_padded = ((N + _ELEMS_PER_PASS - 1) // _ELEMS_PER_PASS) * _ELEMS_PER_PASS
    num_passes = N_padded // _ELEMS_PER_PASS
    needs_padding = N_padded != N

    # SBUF input is sorted in place, so it cannot be padded.
    if data.buffer == nl.sbuf:
        kernel_assert(
            not needs_padding,
            f"SBUF input requires N to be a multiple of {_ELEMS_PER_PASS}, got N={N}",
        )
        data_sb = data
    else:
        data_sb = nl.ndarray((1, N_padded), dtype=nl.float32, buffer=nl.sbuf)
        # Only the padding region needs sentinels
        if needs_padding:
            sentinel = _POS_INF if descending else _NEG_INF
            nisa.memset(data_sb[0, N:N_padded], sentinel)
        data_load_sb = nl.ndarray((1, N), dtype=data.dtype, buffer=nl.sbuf)
        nisa.dma_copy(data_load_sb, data)
        nisa.tensor_copy(data_sb[0, :N], data_load_sb[0, :N])

    # Argsort using N_padded/8 max8 + nc_match_replace8 passes
    argsort_indices_sb = nl.ndarray((1, N_padded), dtype=nl.uint32, buffer=nl.sbuf)
    for pass_idx in nl.sequential_range(num_passes):
        val_buf = nl.ndarray((1, _ELEMS_PER_PASS), dtype=nl.float32)
        nisa.max8(dst=val_buf, src=data_sb)

        idx_pattern = [[N_padded, 1], [1 if descending else -1, _ELEMS_PER_PASS]]
        idx_offset = (
            _ELEMS_PER_PASS * pass_idx
            if descending
            else _ELEMS_PER_PASS * (num_passes - pass_idx) - 1
        )
        nisa.nc_match_replace8(
            dst=data_sb,
            data=data_sb,
            vals=val_buf,
            imm=_NEG_INF,
            dst_idx=argsort_indices_sb.ap(pattern=idx_pattern, offset=idx_offset),
        )

    # Extract valid indices (sentinels sort to front for ascending, to back for descending)
    if needs_padding:
        result_sb = nl.ndarray((1, N), dtype=nl.uint32, buffer=nl.sbuf)
        valid_start = 0 if descending else N_padded - N
        nisa.tensor_copy(
            result_sb[0, :N], argsort_indices_sb[0, valid_start : valid_start + N]
        )
    else:
        result_sb = argsort_indices_sb

    # Output as int32 (safe since N <= INT32_MAX)
    if output_in_sbuf:
        result = result_sb.view(nl.int32)
    else:
        result = nl.ndarray(
            (1, N), dtype=nl.int32, buffer=nl.shared_hbm, name="argsort_indices_hbm"
        )
        nisa.dma_copy(result, result_sb.view(nl.int32))

    return result.reshape((N,)) if is_1d else result
