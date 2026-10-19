# SPDX-License-Identifier: Apache-2.0
"""
Process group utilities for creating and querying subgroups.
"""

import logging
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from libtorch_neuronx_lite.compile.platform import get_platform_target

logger = logging.getLogger(__name__)

# TRN2 8x8 mesh topology.
TRN2_8x8_MESH = [
    [0, 1, 2, 3, 12, 13, 14, 15],
    [4, 5, 6, 7, 8, 9, 10, 11],
    [16, 17, 18, 19, 28, 29, 30, 31],
    [20, 21, 22, 23, 24, 25, 26, 27],
    [32, 33, 34, 35, 44, 45, 46, 47],
    [36, 37, 38, 39, 40, 41, 42, 43],
    [48, 49, 50, 51, 60, 61, 62, 63],
    [52, 53, 54, 55, 56, 57, 58, 59],
]


def group_ranks_tensor(
    process_group: ProcessGroup,
    device: torch.device,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """Build a tensor of the group's global ranks using only graph-native ops.

    ``torch.tensor(list)`` inside a compiled region embeds a real (non-fake)
    constant tensor in the traced graph; the native-route StableHLO lowering
    runs under FakeTensorMode and rejects such constants. Composing the tensor
    from ``torch.full`` + ``torch.stack`` keeps every node fake-traceable. The
    rank list is a trace-time Python constant, so the loop unrolls at trace
    time; group sizes are tiny, so the eager-mode cost is negligible.
    """
    return torch.stack(
        [
            torch.full((), r, dtype=dtype, device=device)
            for r in dist.get_process_group_ranks(process_group)
        ]
    )


def get_group_slice_indices(
    num_items: int,
    world_size: int,
    process_group: ProcessGroup,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute indices for slicing items based on process group membership.

    Supports non-contiguous process groups by using the actual rank ordering.

    Args:
        num_items: Total number of items to slice
        world_size: Number of ranks in the world group
        process_group: The process group to compute indices for
        device: Device to create the index tensor on

    Returns:
        Tensor of indices for selecting items belonging to this group

    Example:
        # num_items = 16
        # world_size = 16
        # process_group = [0, 1, 2, 3, 4, 5, 6, 7]
        # Returns indices [0, 1, 2, 3, 4, 5, 6, 7]
        indices = get_group_slice_indices(16, 8, process_group, device)

        # num_items = 16
        # world_size = 8
        # process_group = [0, 1, 2, 3]
        # Returns indices [0, 1, 2, 3, 4, 5, 6, 7]
        indices = get_group_slice_indices(16, 8, process_group, device)

        # num_items=16
        # world_size=8
        # process_group=[0, 2, 4, 6]
        # Returns indices [0, 1, 4, 5, 8, 9, 12, 13]
        indices = get_group_slice_indices(16, 8, process_group, device)
    """
    items_per_rank = num_items // world_size
    rank_indices = group_ranks_tensor(process_group, device)
    return (
        rank_indices.unsqueeze(1) * items_per_rank
        + torch.arange(items_per_rank, device=device)
    ).flatten()


def create_row_col_groups(
    row_size: int,
    col_size: int,
    rank: int,
) -> tuple[ProcessGroup, ProcessGroup]:
    """
    Create row and column process groups from a 2D mesh of ranks.

    Arranges ranks into a (col_size x row_size) mesh and returns the row/column
    groups that the given rank belongs to.

    Example:
        # 8 ranks arranged as 2x4 mesh:
        #   [[0, 1, 2, 3],
        #    [4, 5, 6, 7]]
        # Row groups: [0,1,2,3], [4,5,6,7]
        # Col groups: [0,4], [1,5], [2,6], [3,7]
        # rank=5 -> row_group=[4,5,6,7], col_group=[1,5]
        row_group, col_group = create_row_col_groups(row_size=4, col_size=2, rank=5)

    Args:
        row_size: Size of row subgroups
        col_size: Size of column subgroups
        rank: The rank of the current process

    Returns:
        Tuple of (row_group, col_group) for the current rank
    """
    logger.debug("Creating subgroups...")

    group_size = row_size * col_size

    # TRN2 requires a special mesh topology when running an 8x8 setup
    use_8x8_mesh = (
        group_size == 64 and row_size == 8 and get_platform_target() == "trn2"
    )

    if use_8x8_mesh:
        mesh = TRN2_8x8_MESH
    else:
        # Contiguous groups: split ranks into col_size rows of row_size each
        mesh = [list(range(i * row_size, (i + 1) * row_size)) for i in range(col_size)]

    row_group = None
    col_group = None

    # Create row subgroups
    for row in mesh:
        ranks = list(row)
        new_group = dist.new_group(ranks)
        if rank in ranks:
            row_group = new_group

    # Create column subgroups
    for col_idx in range(len(mesh[0])):
        ranks = [row[col_idx] for row in mesh]
        new_group = dist.new_group(ranks)
        if rank in ranks:
            col_group = new_group

    logger.debug(
        f"Created subgroups: row_size={row_size}, col_size={col_size}, rank={rank}, "
        f"row_group={dist.get_process_group_ranks(row_group)}, "
        f"col_group={dist.get_process_group_ranks(col_group)}"
    )

    return row_group, col_group
