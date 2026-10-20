# SPDX-License-Identifier: Apache-2.0

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import dge_mode

from nkilib.core.utils.kernel_assert import kernel_assert

import torch

from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_node_count,
    get_world_group,
)

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel
from ..cumsum import cumsum


def build_all2all_dispatch_metadata(
    expert_index: torch.Tensor,
    num_experts: int,
    num_elements_per_token: int,
    group: GroupCoordinator,
    recv_displs: torch.Tensor = None,
    num_experts_per_node: int = None,
) -> torch.Tensor:
    """Build all2all-v dispatch metadata (send_counts/send_displs) from expert_index.

    Each token contributes 1 to send_counts[dst] for every UNIQUE rank its K experts
    route to (experts on the same rank are de-duped). Counts are then scaled by
    num_elements_per_token. All rows are in element units, not token units.

    Optional hierarchical mode (``num_experts_per_node`` set): builds the
    intra-node-stage metadata. Only experts on the current server count; off-server
    experts and -1 padding are dropped. The current server is inferred from the
    global rank.

    Args:
        expert_index (torch.Tensor): [T, K] int32 expert indices per token (-1 = padding).
        num_experts (int): Total number of experts.
        num_elements_per_token (int): Elements per token (e.g. hidden_size); scales token counts.
        group (GroupCoordinator): Determines replica_group_size = dst-rank count.
        recv_displs (torch.Tensor, optional): Pre-computed recv displacements (row 3).
        num_experts_per_node (int, optional): Experts per server; enables hierarchical mode.

    Returns:
        torch.Tensor: [4, replica_group_size] uint32. Rows: send_counts, send_displs,
            recv_counts (zeros), recv_displs (zeros or provided).

    Example (plain dispatch, world_size=8, one expert per rank):
        >>> expert_index = torch.arange(8, dtype=torch.int32).unsqueeze(1)
        >>> meta = build_all2all_dispatch_metadata(expert_index, num_experts=8,
        ...     num_elements_per_token=1, group=group)
        >>> # meta[0] = [1,1,1,1,1,1,1,1], meta[1] = [0,1,2,3,4,5,6,7]

    Example (hierarchical, num_experts=16 over 2 servers, this rank on server 0 so
    on-server experts are [0..7] -> ranks [0..7]; off-server experts [8..15] and -1
    are dropped):
        >>> expert_index = torch.tensor([[3, -1], [10, -1], [0, 1]], dtype=torch.int32)
        >>> meta = build_all2all_dispatch_metadata(expert_index, num_experts=16,
        ...     num_elements_per_token=1, group=group, num_experts_per_node=8)
        >>> # token 0 -> rank 3; token 1 -> off-server (dropped); token 2 -> ranks 0 and 1
        >>> # meta[0] = [1,1,0,1,0,0,0,0]
    """

    # Convert from GroupCoordinator -> size
    replica_group_size = group.world_size

    _validate_inputs(
        expert_index,
        replica_group_size,
        num_experts,
        num_elements_per_token,
    )

    # Hierarchical intra-node mode: current_node = global_rank / ranks_per_node.
    current_node = None
    if num_experts_per_node is not None:
        ranks_per_node = get_world_group().world_size // get_node_count()
        current_node = get_world_group().rank_in_group_tensor // ranks_per_node

    if _can_use_kernel(expert_index, recv_displs, num_experts_per_node):
        wrapped = wrap_nki(_build_all2all_dispatch_metadata_one_token_nki)
        return wrapped[2](
            expert_index=expert_index,
            num_experts=num_experts,
            num_elements_per_token=num_elements_per_token,
            replica_group_size=replica_group_size,
        )

    return _torch_impl(
        expert_index,
        num_experts,
        num_elements_per_token,
        replica_group_size,
        recv_displs,
        num_experts_per_node=num_experts_per_node,
        current_node=current_node,
    )


def _validate_inputs(
    expert_index: torch.Tensor,
    replica_group_size: int,
    num_experts: int,
    num_elements_per_token: int,
) -> None:
    """Validate inputs for build_all2all_dispatch_metadata."""
    assert expert_index.dtype == torch.int32, (
        f"Expected expert_index.dtype = torch.int32, got {expert_index.dtype=}"
    )
    assert num_experts % replica_group_size == 0, (
        f"num_experts must be divisible by replica_group_size, got {num_experts=}, {replica_group_size=}"
    )
    assert replica_group_size > 1, (
        f"expected replica_group_size > 1, got {replica_group_size=}"
    )
    assert num_elements_per_token > 0, (
        f"num_elements_per_token must be greater than 0, got {num_elements_per_token=}"
    )


def _can_use_kernel(
    expert_index: torch.Tensor,
    recv_displs: torch.Tensor,
    num_experts_per_node: int,
) -> bool:
    """Check if the single-token NKI kernel applies.

    Kernel constraints:
        - Device must support NKI kernels (can_run_kernel)
        - T (expert_index.shape[0]) must be 1
        - recv_displs must be None
        - num_experts_per_node must be None (no hierarchical mode)
    """
    if not can_run_kernel(expert_index):
        return False
    return (
        expert_index.shape[0] == 1  # T = 1
        and recv_displs is None
        and num_experts_per_node is None
    )


def _torch_impl(
    expert_index: torch.Tensor,
    num_experts: int,
    num_elements_per_token: int,
    replica_group_size: int,
    recv_displs: torch.Tensor = None,
    num_experts_per_node: int = None,
    current_node: torch.Tensor = None,
) -> torch.Tensor:
    """PyTorch implementation of build_all2all_dispatch_metadata."""

    # Utilize int32 ops internally, which XLA natively supports
    zeros = torch.zeros(
        replica_group_size, dtype=torch.int32, device=expert_index.device
    )
    recv_counts = zeros
    recv_displs = recv_displs.to(torch.int32) if recv_displs is not None else zeros

    # Step 1: Map each (token, k) to its destination rank.
    # When a token is routed to multiple experts that are on the same destination rank,
    # the (token, rank) pair is de-duped: it counts as 1, not K.
    T, K = expert_index.shape

    if num_experts_per_node is not None:
        # Intra-node stage: mark off-server experts with -1 so they're excluded from send counts.
        num_experts_per_rank = num_experts_per_node // replica_group_size
        expert_node = expert_index // num_experts_per_node
        local_rank = (
            expert_index - current_node * num_experts_per_node
        ) // num_experts_per_rank
        dst_ranks = torch.where(
            expert_node == current_node,
            local_rank,
            torch.full_like(expert_index, -1),
        ).to(torch.int32)  # [T, K], -1 for off-server experts
    else:
        n_local_experts = num_experts // replica_group_size
        dst_ranks = (expert_index // n_local_experts).to(torch.int32)  # [T, K]

    # Step 2: scatter (t, dst_rank) into a [T, mask_width] mask, then slice to replica_group_size.
    # mask_width >= K for XLA scatter lowering; +1 garbage col in hierarchical mode to absorb dst=-1.
    needs_garbage_col = num_experts_per_node is not None
    mask_width = max(K, replica_group_size) + (1 if needs_garbage_col else 0)
    if needs_garbage_col:
        rank_mask = torch.zeros(
            T, mask_width, dtype=torch.int32, device=expert_index.device
        )
        # -1 -> garbage column (last position); on-server ranks keep their index.
        garbage_col = mask_width - 1
        scatter_cols = torch.where(dst_ranks < 0, garbage_col, dst_ranks)
        rank_mask.scatter_(1, scatter_cols, 1)
        rank_mask = rank_mask[:, :replica_group_size]
    else:
        rank_mask = torch.zeros(
            T, mask_width, dtype=torch.int32, device=expert_index.device
        )
        rank_mask.scatter_(1, dst_ranks, 1)
        rank_mask = rank_mask[:, :replica_group_size]

    # Step 3: Sum across tokens to get per-rank de-duped token counts, then scale by
    # num_elements_per_token to get per-rank send_counts in elements (matches the unit
    # of the all2all_v collective metadata).
    send_counts = rank_mask.sum(dim=0).to(torch.int32) * num_elements_per_token

    # Compute send_displs as cumsum(send_counts) with offset of 1
    send_displs = torch.zeros(
        replica_group_size, dtype=torch.int32, device=expert_index.device
    )
    send_displs[1:] = cumsum(send_counts[:-1].unsqueeze(0)).squeeze(0)

    output_metadata = torch.stack(
        [send_counts, send_displs, recv_counts, recv_displs], dim=0
    )

    return output_metadata.to(torch.uint32)


@nki.jit
def _build_all2all_dispatch_metadata_one_token_nki(
    expert_index,
    num_experts,
    num_elements_per_token,
    replica_group_size,
):
    T, _ = expert_index.shape

    kernel_assert(T == 1, f"T must be 1, got {T=}")

    output_metadata = nl.ndarray(
        (4, replica_group_size), dtype=nl.uint32, buffer=nl.shared_hbm
    )

    # zero recv counts / displs
    zeros = nl.ndarray((2, replica_group_size), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.memset(zeros, 0)
    nisa.dma_copy(output_metadata[2:, :], zeros)

    expert_index_sb = nl.ndarray(expert_index.shape, expert_index.dtype, buffer=nl.sbuf)
    send_counts_sb = nl.ndarray(
        (1, replica_group_size), dtype=nl.uint32, buffer=nl.sbuf
    )
    send_displs_sb = nl.ndarray(
        (1, replica_group_size), dtype=nl.uint32, buffer=nl.sbuf
    )
    init_sb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    ones_sb = nl.ndarray((1, replica_group_size - 1), dtype=nl.float32, buffer=nl.sbuf)

    nisa.dma_copy(expert_index_sb, expert_index, dge_mode=dge_mode.none)

    n_local_experts = num_experts // replica_group_size
    for dst_idx in range(replica_group_size):
        lo = dst_idx * n_local_experts
        hi = (dst_idx + 1) * n_local_experts

        # Range check (expert in [lo, hi)): tensor_scalar chains, not fused-AND, so AND the two predicates via multiply.
        ge_mask = nl.ndarray(
            expert_index.shape, dtype=expert_index.dtype, buffer=nl.sbuf
        )
        lt_mask = nl.ndarray(
            expert_index.shape, dtype=expert_index.dtype, buffer=nl.sbuf
        )
        rank_mask = nl.ndarray(
            expert_index.shape, dtype=expert_index.dtype, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            data=expert_index_sb,
            op0=nl.greater_equal,
            operand0=lo,
            dst=ge_mask,
        )
        nisa.tensor_scalar(
            data=expert_index_sb,
            op0=nl.less,
            operand0=hi,
            dst=lt_mask,
        )
        nisa.tensor_tensor(data1=ge_mask, data2=lt_mask, op=nl.multiply, dst=rank_mask)

        # max-reduce across K (free dim) to dedup the per-token contribution, then scale by num_elements_per_token.
        reduced = nl.ndarray((1, 1), dtype=expert_index.dtype, buffer=nl.sbuf)
        nisa.tensor_reduce(data=rank_mask, op=nl.max, axis=(1,), dst=reduced)
        nisa.tensor_scalar(
            data=reduced,
            op0=nl.multiply,
            operand0=num_elements_per_token,
            dst=send_counts_sb[0, dst_idx : dst_idx + 1],
        )

    nisa.memset(send_displs_sb[0, 0], 0)  # first val is always 0
    nisa.memset(init_sb, 0.0)
    nisa.memset(ones_sb, 1.0)
    nisa.tensor_tensor_scan(
        initial=init_sb,
        data0=ones_sb,
        op1=nl.add,
        data1=send_counts_sb[0, : replica_group_size - 1],
        op0=nl.multiply,
        dst=send_displs_sb[0, 1:],
    )

    nisa.dma_copy(output_metadata[0, :], send_counts_sb)
    nisa.dma_copy(output_metadata[1, :], send_displs_sb)

    return output_metadata
