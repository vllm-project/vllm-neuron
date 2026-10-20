# SPDX-License-Identifier: Apache-2.0
"""build_all_gatherv_metadata functional API."""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import dge_mode

from nkilib.core.utils.kernel_assert import kernel_assert

import torch

from vllm.distributed.parallel_state import GroupCoordinator

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel

_RANKS_PER_CHIP = 4
_SUPPORTED_RECV_COUNTS_DTYPES = [torch.uint32]


def build_all_gatherv_metadata(
    all2all_recv_counts: torch.Tensor, group: GroupCoordinator
) -> torch.Tensor:
    """Build all-gather-v metadata from an all2all collective's recv counts.

    The all-gather-v send count for this rank is the total number of elements it
    received from the preceding all2all (the sum over all N recv-count entries),
    broadcast across all ranks in the group. Send displs and recv counts are zeroed.

    Args:
        all2all_recv_counts (torch.Tensor): [1, N] uint32 per-source recv counts
            (in elements). N need not equal replica_group_size.
        group (GroupCoordinator): Determines replica_group_size = dst-rank count.

    Returns:
        torch.Tensor: [3, replica_group_size] uint32 metadata. Row 0: send counts
            (this rank's total recv elements, broadcast), Row 1: send displs (zeros),
            Row 2: recv counts (zeros).

    Example:
        >>> # all2all_recv_counts = [[3, 5, 0, 4, 1, 2, 0, 6]] -> sum 21 (N=8 != RG=4)
        >>> meta = build_all_gatherv_metadata(all2all_recv_counts, group)  # group.world_size=4
        >>> # meta[0] = [21, 21, 21, 21], meta[1] = [0, 0, 0, 0], meta[2] = [0, 0, 0, 0]
    """
    replica_group_size = group.world_size

    _validate_inputs(all2all_recv_counts)

    if _can_use_kernel(all2all_recv_counts, replica_group_size):
        wrapped = wrap_nki(_build_all_gatherv_metadata_nki)
        return wrapped[2](all2all_recv_counts, replica_group_size)

    return _torch_impl(all2all_recv_counts, replica_group_size)


def _validate_inputs(all2all_recv_counts: torch.Tensor) -> None:
    """Validate inputs for build_all_gatherv_metadata."""
    assert all2all_recv_counts.dtype in _SUPPORTED_RECV_COUNTS_DTYPES, (
        f"Expected all2all_recv_counts.dtype in {_SUPPORTED_RECV_COUNTS_DTYPES}, "
        f"got {all2all_recv_counts.dtype=}"
    )


def _can_use_kernel(all2all_recv_counts: torch.Tensor, replica_group_size: int) -> bool:
    """Check if the NKI kernel can be used for build_all_gatherv_metadata.

    Kernel constraints:
        - Device must support NKI kernels (can_run_kernel)
        - replica_group_size == 4 (kernel is hardcoded for 4 dst ranks)
    """
    return can_run_kernel(all2all_recv_counts) and replica_group_size == _RANKS_PER_CHIP


def _torch_impl(
    all2all_recv_counts: torch.Tensor, replica_group_size: int
) -> torch.Tensor:
    """PyTorch reference for build_all_gatherv_metadata.

    Sums all2all_recv_counts to a single total; agv output is (3, R) uint32 with
    row 0 = total recv count broadcast across R columns, rows 1-2 = zeros
    (recv_displs / send_counts in the agv collective).

    Args:
        all2all_recv_counts (torch.Tensor): (1, N) uint32 per-source recv counts.
        replica_group_size (int): number of destination ranks (R).

    Returns:
        torch.Tensor: (3, R) uint32 contiguous metadata.

    Example:
        >>> _torch_impl(torch.tensor([[3, 5, 0, 4]], dtype=torch.uint32), 4)
        tensor([[12, 12, 12, 12], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.uint32)
    """
    # Compute in int32 and cast to uint32 as the final op so the dtype survives XLA lowering.
    total = all2all_recv_counts.to(torch.int32).sum()
    send_counts = total.repeat(replica_group_size)
    zeros = torch.zeros(
        replica_group_size, dtype=torch.int32, device=all2all_recv_counts.device
    )
    return torch.stack([send_counts, zeros, zeros], dim=0).to(torch.uint32)


@nki.jit
def _build_all_gatherv_metadata_nki(all2all_recv_counts, replica_group_size):
    """Build all-gather-v metadata from the all-to-all recv counts.

    Sums all2all_recv_counts into a single total and broadcasts it across the R
    columns of the (3, R) output; rows 1 and 2 (recv_displs, send_counts) are zero.

    Args:
        all2all_recv_counts: (1, N) uint32 per-source recv counts.
        replica_group_size: number of destination ranks (R), must be 4.

    Returns:
        (3, R) uint32 agv metadata in HBM.
    """

    # FIXME: XLA traces uint32 -> int32; patch with bitcast to ensure correct type
    if all2all_recv_counts.dtype != nl.uint32:
        all2all_recv_counts = all2all_recv_counts.view(nl.uint32)

    kernel_assert(
        replica_group_size == _RANKS_PER_CHIP,
        f"replica_group_size must be 4, got {replica_group_size}",
    )
    kernel_assert(
        all2all_recv_counts.dtype == nl.uint32,
        f"all2all_recv_counts must be uint32, got {all2all_recv_counts.dtype}",
    )

    recv_counts_sb = nl.ndarray(
        (1, all2all_recv_counts.shape[-1]),
        dtype=all2all_recv_counts.dtype,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(recv_counts_sb, all2all_recv_counts, dge_mode=dge_mode.none)

    send_counts_sb = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    agv_metadata = nl.ndarray(
        (3, replica_group_size), dtype=nl.uint32, buffer=nl.shared_hbm
    )

    # compute sum of recv counts
    nisa.tensor_reduce(
        data=recv_counts_sb,
        op=nl.add,
        axis=(1,),
        dst=send_counts_sb,
    )

    # copy w/ broadcast to RG size
    nisa.dma_copy(
        src=send_counts_sb.ap(
            pattern=[[1, 1], [0, replica_group_size], [1, 1]],
            offset=0,
        ),
        dst=agv_metadata[0, :],
        dge_mode=dge_mode.none,
    )

    # zero out sdispls, recv counts
    zero_sb = nl.ndarray((2, replica_group_size), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.memset(zero_sb, 0)
    nisa.dma_copy(agv_metadata[1:, :], zero_sb, dge_mode=dge_mode.none)

    return agv_metadata
