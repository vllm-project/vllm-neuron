# SPDX-License-Identifier: Apache-2.0
"""build_all2all_combine_metadata functional API."""

import torch

from ..cumsum import cumsum

from vllm_neuron.utils.neuron_utils import can_run_kernel

_SUPPORTED_SEND_COUNTS_DTYPES = [torch.int32, torch.uint32]


def build_all2all_combine_metadata(
    send_counts: torch.Tensor,
    recv_displs: torch.Tensor = None,
) -> torch.Tensor:
    """Build metadata for all2all combine. Computes send_displs from send_counts, and places
    recv_displs in row3 of the metadata buffer if provided. Send_counts is expected to already be known
    from the recv_counts of all2all dispatch.

    All metadata rows represent the number of elements being sent/recieved, not the number of tokens.

    Args:
        send_counts (torch.Tensor): [replica_group_size] int32/uint32 tensor of per-rank element counts to send.
        recv_displs (torch.Tensor, optional): Pre-computed recv displacements.

    Returns:
        torch.Tensor: [4, replica_group_size] uint32 tensor.
            Row 0: send counts (number of elements per rank), Row 1: send displacements,
            Row 2: recv counts (zeros), Row 3: recv displacements (zeros or provided).

    Example:
        >>> send_counts = torch.tensor([4, 4, 4, 4, 4, 4, 4, 4], dtype=torch.uint32)
        >>> meta = build_all2all_combine_metadata(send_counts)
        >>> # meta[0] = [4,4,4,4,4,4,4,4], meta[1] = [0,4,8,12,16,20,24,28]
    """
    _validate_inputs(send_counts)

    if _can_use_kernel(send_counts):
        # TODO: NKI kernel dispatch
        pass

    return _torch_impl(send_counts, recv_displs)


def _validate_inputs(
    send_counts: torch.Tensor,
) -> None:
    """Validate inputs for build_all2all_combine_metadata."""
    assert send_counts.dtype in _SUPPORTED_SEND_COUNTS_DTYPES, (
        f"Expected send_counts.dtype in {_SUPPORTED_SEND_COUNTS_DTYPES}, got {send_counts.dtype=}"
    )


def _can_use_kernel(
    send_counts: torch.Tensor,
) -> bool:
    """Check if the NKI kernel can be used."""
    if not can_run_kernel(send_counts):
        return False

    # TODO: add kernel integration
    return False


def _torch_impl(
    send_counts: torch.Tensor,
    recv_displs: torch.Tensor = None,
) -> torch.Tensor:
    """PyTorch implementation of build_all2all_combine_metadata."""

    # Utilize int32 ops internally, which XLA natively supports
    replica_group_size = send_counts.shape[-1]
    zeros = torch.zeros(
        replica_group_size, dtype=torch.int32, device=send_counts.device
    )
    send_counts = send_counts.to(torch.int32)
    recv_counts = zeros
    recv_displs = recv_displs.to(torch.int32) if recv_displs is not None else zeros

    # Compute send_displs as cumsum(send_counts) with offset of 1
    send_displs = torch.zeros(
        replica_group_size, dtype=torch.int32, device=send_counts.device
    )
    send_displs[1:] = cumsum(send_counts[:-1].unsqueeze(0)).squeeze(0)

    # Stack metadata, with cast back to uint32
    return torch.stack([send_counts, send_displs, recv_counts, recv_displs], dim=0).to(
        torch.uint32
    )
