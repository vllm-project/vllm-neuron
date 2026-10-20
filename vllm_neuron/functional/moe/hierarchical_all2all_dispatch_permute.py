# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_node_count,
    get_world_group,
)

from ..argsort_unstable import argsort_unstable
from .permute_routed_tokens import _bitcast


def hierarchical_all2all_dispatch_permute(
    internode_dispatch_output: torch.Tensor,
    expert_index: torch.Tensor,
    num_experts_per_node: int,
    local_ep_group: GroupCoordinator,
):
    """Re-permute the inter-node all-to-all-v output for the intra-node dispatch.

    After the inter-node stage, this rank holds the tokens whose experts live on
    its own node. This NF permutes those tokens by their intra-node destination
    rank so they can be shipped with a second (intra-node) all-to-all-v.

    Which experts belong to this node is derived from the rank/topology: with
    ``world_size`` total EP ranks across ``get_node_count()`` nodes, there are
    ``ranks_per_node = world_size // nnodes`` ranks per node and this rank sits on
    ``node_id = rank // ranks_per_node``. Node ``n`` owns experts
    ``[n * num_experts_per_node, (n + 1) * num_experts_per_node)``. Tokens routed
    to experts on a *different* node are dropped here (they were already routed
    elsewhere by the inter-node stage).

    Args:
        internode_dispatch_output (torch.Tensor): [T, C] inter-node dispatch
            output, each row an opaque payload [hidden | affinities | token idx].
        expert_index (torch.Tensor): [T, K] int32 top-K expert ids (-1 when padded).
        num_experts_per_node (int): Number of experts owned by each node.
        ep_group (GroupCoordinator): The EP group. Its ``rank_in_group`` and
            ``world_size`` give this rank's id and the total number of EP ranks.

    Returns:
        torch.Tensor: [T*K, C] payload rows grouped by intra-node destination
            rank. Rows for off-node / de-duped / padded entries are zero-filled
            with a -2 int32 token-index sentinel.
    """

    ranks_per_node = get_world_group().world_size // get_node_count()
    current_node = get_world_group().rank_in_group_tensor // ranks_per_node

    return _torch_impl(
        internode_dispatch_output=internode_dispatch_output,
        expert_index=expert_index,
        num_experts_per_node=num_experts_per_node,
        current_node=current_node,
        local_ep_group_size=local_ep_group.world_size,
    )


def _torch_impl(
    internode_dispatch_output,
    expert_index,
    num_experts_per_node,
    current_node,
    local_ep_group_size,
):
    # Step 1: Dims + node/rank topology.
    T, _ = internode_dispatch_output.shape
    _, K = expert_index.shape
    num_experts_per_rank = num_experts_per_node // local_ep_group_size

    # Step 2: Build inverse argsort array, with adjustment for de-duped token/rank pairs
    # Step 2.1: Map each routed expert to its intra-node destination rank.
    # Node n owns experts [n*npn, (n+1)*npn); a local rank within this node owns
    # num_experts_per_rank of them. Experts on another node (or padded -1 entries)
    # are marked -1 so they're excluded — only tokens for THIS node's experts are
    # permuted. The -1 sentinel rejoins the de-dup path below (scattered to the
    # garbage row and discarded).
    expert_node = expert_index // num_experts_per_node
    local_rank = (
        expert_index - current_node * num_experts_per_node
    ) // num_experts_per_rank
    expert_ranks = torch.where(
        expert_node == current_node,
        local_rank,
        torch.full_like(expert_index, -1),
    ).to(torch.int32)
    expert_ranks_deduped = expert_ranks.clone()
    for k in range(1, K):
        # Compare column k against all prior columns [0..k-1] at once
        matches = (
            expert_ranks_deduped[:, :k] == expert_ranks_deduped[:, k : k + 1]
        ).any(dim=1)
        expert_ranks_deduped[:, k] = torch.where(
            matches, -1, expert_ranks_deduped[:, k]
        )
    dedupe_count = (expert_ranks_deduped == -1).sum()

    # Step 2.2: Compute inverse argsort
    token_argsort = argsort_unstable(expert_ranks_deduped.flatten().to(torch.int32))
    token_inv_argsort = torch.zeros_like(token_argsort)
    token_inv_argsort[token_argsort] = torch.arange(
        T * K, dtype=torch.int32, device=expert_index.device
    )

    # Step 2.3: Adjust inverse argsort so that all de-duped tokens have idx 0
    token_inv_argsort_adjusted = (
        (token_inv_argsort - dedupe_count + 1).clamp(min=0).to(torch.int32)
    )

    # Step 3:: Broadcast [T, H] -> [T*K, H]
    internode_dispatch_output_bc_K = (
        internode_dispatch_output.unsqueeze(1).expand(-1, K, -1).reshape(T * K, -1)
    )

    # Step 4: Group tokens by destination rank, with de-dupe
    # Bitcast to a same-width integer dtype, which is supported on CPU and doesn't canonicalize NaNs.
    if str(expert_index.device) == "cpu":
        int_scatter_dtype = (
            torch.int8
            if internode_dispatch_output_bc_K.element_size() == 1
            else torch.int16
        )
        internode_dispatch_output_bc_K = _bitcast(
            internode_dispatch_output_bc_K, int_scatter_dtype
        )

    # Scatter tokens into output buffer using adjusted inv argsort array. De-dupes are scattered into row 0
    # NOTE: padded rows have token index -2 for better debuggability. -2 index post dispatch = metadata was incorrect.
    n_idx_cols = 4 // internode_dispatch_output_bc_K.element_size()
    n_data_cols = internode_dispatch_output_bc_K.shape[-1] - n_idx_cols
    n_rows = T * K + 1
    zeros_part = torch.zeros(
        (n_rows, n_data_cols),
        dtype=internode_dispatch_output_bc_K.dtype,
        device=expert_index.device,
    )
    neg_two_int32 = torch.full(
        (n_rows, 1), -2, dtype=torch.int32, device=expert_index.device
    )
    neg_two_native = _bitcast(neg_two_int32, internode_dispatch_output_bc_K.dtype)
    output_permuted = torch.concat([zeros_part, neg_two_native], dim=1)
    output_permuted.scatter_(
        0,
        token_inv_argsort_adjusted.unsqueeze(1).expand_as(
            internode_dispatch_output_bc_K
        ),
        internode_dispatch_output_bc_K,
    )

    # CPU mode does not support fp8 scatter_; convert back to hidden.dtype
    if str(expert_index.device) == "cpu":
        output_permuted = _bitcast(output_permuted, internode_dispatch_output.dtype)

    # Discard row 0, which contains garbage/de-duped tokens
    return output_permuted[1:, :]
