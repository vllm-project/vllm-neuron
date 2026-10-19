# SPDX-License-Identifier: Apache-2.0
import math
import torch
import torch.distributed as dist
from torch import Tensor
from typing import TYPE_CHECKING, Optional
import logging

import nki
from nkilib.core.subkernels.find_nonzero_indices import find_nonzero_indices
from nkilib.core.subkernels.indexed_flatten import indexed_flatten

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel

find_nonzero_indices_jit = nki.jit()(find_nonzero_indices)
indexed_flatten_jit = nki.jit()(indexed_flatten)


logger = logging.getLogger(__name__)


def build_blockwise_mapping(
    expert_affinities: Tensor,
    num_local_experts: int,
    num_experts_per_token: int,
    block_size: int,
    moe_group: "GroupCoordinator",
    tp_degree: int = 1,
    padding_mask: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Build blockwise token-to-expert mapping for the kernel.

    Args:
        expert_affinities: [T, E_local] router scores (zeros for non-selected experts)
        num_local_experts: total number of local experts (E_local)
        num_experts_per_token: top-k experts per token
        block_size: tokens per block
        moe_group: GroupCoordinator for the MoE process group. This is the same
            physical group of ranks regardless of whether MoE uses TP or EP, but
            the semantics differ:
            - TP (no EP): each rank holds all experts with sharded intermediate
              dimensions. Ranks must collectively gather/reduce partial results.
            - EP: each rank holds a disjoint subset of experts with full
              intermediate dimensions. Each rank computes its local experts
              independently — no sharding collectives needed.
            Because the same group serves both roles, tp_degree must be passed
            separately to indicate which mode is active.
        tp_degree: the degree of intermediate-dimension sharding across ranks
            for MoE expert computation. When tp_degree > 1 (TP without EP),
            expert work is sharded across ranks and collectives (all_gather,
            all_reduce) combine partial results. When tp_degree = 1 (EP enabled
            or single rank), each rank processes all its local experts
            independently with no collectives. Defaults to 1.
        padding_mask: [T,] boolean mask where True = real token, False = padding.
            If not specified, all tokens are treated as real.

    Returns:
        expert_affinities_masked: [T*E_local, 1] flattened affinities for kernel
        token_position_to_id: [N*block_size] mapping from positions to token IDs
        block_to_expert: [N] expert assignment per block
        conditions: [N] valid block indicators

    Example:
        >>> # After router computation (topk + softmax + scatter)
        >>> # expert_affinities has shape [T, E_local] with zeros for non-selected experts
        >>> T, E_local, top_k, block_size = 512, 8, 2, 128
        >>> expert_affinities = torch.zeros(T, E_local, device="xla")
        >>> # Simulate router output: each token selects top_k experts
        >>> for t in range(T):
        ...     selected = torch.randperm(E_local)[:top_k]
        ...     expert_affinities[t, selected] = torch.softmax(torch.randn(top_k), dim=0)
        >>>
        >>> # Build blockwise mapping
        >>> affinities_masked, token_pos_to_id, block_to_expert, conditions = build_blockwise_mapping(
        ...     expert_affinities=expert_affinities,
        ...     num_local_experts=E_local,
        ...     num_experts_per_token=top_k,
        ...     block_size=block_size,
        ... )
        >>>
        >>> # Outputs can be passed to bwmm_shard_on_block_kernel
        >>> # affinities_masked: [T*E_local, 1] - flattened affinities
        >>> # token_pos_to_id: [N*block_size] - maps block positions to token IDs (-1 for padding)
        >>> # block_to_expert: [N] - expert index for each block
        >>> # conditions: [N] - 1 if block has valid tokens, 0 otherwise
    """
    total_tokens = expert_affinities.shape[0]
    device = expert_affinities.device

    expert_mask = (expert_affinities != 0).to(torch.float32)
    expert_affinities_masked = _apply_padding_mask(
        expert_affinities, padding_mask
    ).view(-1, 1)  # [T*E_local, 1]

    # Kernel configs
    max_chunk_size = 16384
    chunk_size = min(total_tokens, max_chunk_size)
    f_len = min(128, total_tokens // 16)

    # When TP-sharded, the kernel processes E_kernel = E_local // tp_degree experts
    # per rank. The kernel constraints must be checked against E_kernel, not E_local.
    sharded = tp_degree > 1 and num_local_experts % tp_degree == 0
    E_for_kernel = num_local_experts // tp_degree if sharded else num_local_experts
    can_use_find_nonzero_kernel = _can_use_find_nonzero_indices_kernel(
        T=total_tokens,
        E=E_for_kernel,
        chunk_size=chunk_size,
        tensor=expert_mask,
    )
    can_use_indexed_flatten_kernel = _can_use_indexed_flatten_kernel(
        T=total_tokens,
        tensor=expert_mask,
        f_len=f_len,
    )
    use_kernel_flow = can_use_find_nonzero_kernel and can_use_indexed_flatten_kernel

    if use_kernel_flow:
        token_position_to_id, block_to_expert, num_blocks = (
            _build_blockwise_mapping_kernel(
                expert_mask=expert_mask,
                num_local_experts=num_local_experts,
                num_experts_per_token=num_experts_per_token,
                block_size=block_size,
                tp_degree=tp_degree,
                total_tokens=total_tokens,
                chunk_size=chunk_size,
                f_len=f_len,
                moe_group=moe_group,
            )
        )
    else:
        token_position_to_id, block_to_expert, num_blocks = (
            _build_blockwise_mapping_torch(
                expert_mask=expert_mask,
                num_local_experts=num_local_experts,
                num_experts_per_token=num_experts_per_token,
                block_size=block_size,
                total_tokens=total_tokens,
                tp_degree=tp_degree,
                moe_group=moe_group,
            )
        )

    # Build conditions (valid block indicators)
    blocks = token_position_to_id.view(num_blocks, block_size)
    conditions = torch.any(blocks != -1, dim=1).to(torch.int32)

    return (
        expert_affinities_masked,
        token_position_to_id.to(torch.int32),
        block_to_expert.to(torch.int32),
        conditions,
    )


def _build_blockwise_mapping_kernel(
    expert_mask: Tensor,
    num_local_experts: int,
    num_experts_per_token: int,
    block_size: int,
    tp_degree: int,
    total_tokens: int,
    chunk_size: int,
    f_len: int,
    moe_group: "GroupCoordinator",
) -> tuple[Tensor, Tensor, int]:
    """
    Build blockwise mapping using find_nonzero_indices and indexed_flatten NKI kernels.

    This is the optimized kernel path for building token-to-expert mappings on Neuron hardware.

    Args:
        expert_mask: [T, E_local] binary mask indicating selected experts
        num_local_experts: total number of local experts (E_local)
        num_experts_per_token: top-k experts per token
        block_size: tokens per block
        tp_degree: tensor parallelism degree
        total_tokens: total number of tokens (T)
        chunk_size: chunk size for find_nonzero_indices kernel
        f_len: flatten length for indexed_flatten kernel
        moe_group: GroupCoordinator for MoE collectives

    Returns:
        token_position_to_id: [N*block_size] mapping from positions to token IDs
        block_to_expert: [N] expert assignment per block
        num_blocks: total number of blocks (N)
    """
    device = expert_mask.device
    find_nonzero_indices_nki = wrap_nki(find_nonzero_indices_jit)
    indexed_flatten_nki = wrap_nki(indexed_flatten_jit)

    E_local = num_local_experts
    # SPMD: read this rank's group position as an on-device tensor (stamped onto the
    # group) so col_start_id below is a graph INPUT, not a per-rank baked constant — all
    # tp_degree ranks then share ONE compiled NEFF. Callers stamp via
    # stamp_spmd_rank_tensors (model runner / executor; tests do the same).
    rank_tensor = moe_group.rank_in_group_tensor

    sharded = tp_degree > 1 and E_local % tp_degree == 0

    if sharded:
        E_kernel = E_local // tp_degree
        # This rank's expert-column offset. col_start_id feeds the kernels as a runtime
        # offset value (not a shape), so a dynamic tensor keeps the graph rank-invariant.
        #
        # SPMD / chunked-prefill trace hazard: under chunked prefill
        # (max_num_batched_tokens < max_model_len), torch.compile constant-folds
        # ``(rank_tensor % tp_degree) * E_kernel`` into a Python-int-derived
        # ``torch.tensor([<rank * E_kernel>])`` per rank, splintering the compile
        # cache into one NEFF per rank -- the exact failure SPMD exists to prevent
        # (see gpt-oss commit fe1252e42 for the sibling fix).
        #
        # Two hazards to sidestep:
        #   1. ``% tp_degree`` is redundant here: ``moe_group`` is the ep_tp
        #      coordinator (world_size == tp_degree), so ``rank_in_group_tensor``
        #      is already in ``[0, tp_degree)``. Dropping it removes an operator
        #      dynamo is prone to specialize on.
        #   2. ``.reshape(1)`` is a no-op (rank_tensor is already shape ``(1,)``)
        #      but its presence appears to be one of the fold-boundary nodes.
        #      Drop it and let the 1-D tensor broadcast through the multiply.
        col_start_id = (rank_tensor.to(torch.int32) * E_kernel).to(torch.int32)

        # Step 1: Get token indices per expert using kernel (local shard)
        indices, tokens_per_expert_local = find_nonzero_indices_nki[2](
            input_tensor=expert_mask.to(torch.float32),
            col_start_id=col_start_id,
            n_cols=E_kernel,
            chunk_size=chunk_size,
        )

        # Gather tokens_per_expert across ranks: [E_kernel,] -> [E_local,]
        tokens_per_expert = moe_group.all_gather(tokens_per_expert_local, dim=0)
    else:
        E_kernel = E_local
        # Use torch.zeros rather than torch.tensor([0], ...): the two are
        # semantically identical (length-1 int32 zero on `device`), but
        # RankConstantCheckPass under NEURON_LIBTORCH_SPMD_STRICT flags any
        # torch.tensor / torch.full small-int singleton as a suspected baked
        # rank constant. The [0] here is genuinely rank-invariant (E_local
        # is the same across ranks in this branch), so torch.zeros keeps the
        # value while sidestepping the heuristic.
        col_start_id = torch.zeros(1, dtype=torch.int32, device=device)

        # Step 1: Get token indices per expert using kernel (all local experts)
        indices, tokens_per_expert = find_nonzero_indices_nki[2](
            input_tensor=expert_mask.to(torch.float32),
            col_start_id=col_start_id,
            n_cols=E_kernel,
            chunk_size=chunk_size,
        )

    # Compute block mapping
    blocks_per_expert = ((tokens_per_expert + block_size - 1) // block_size).to(
        dtype=torch.long
    )
    total_local_tokens = total_tokens * min(num_experts_per_token, num_local_experts)
    num_blocks = (
        math.ceil((total_local_tokens - (num_local_experts - 1)) / block_size)
        + num_local_experts
        - 1
    )
    num_blocks = min(num_blocks, total_local_tokens)

    # Compute inclusive cumsum
    inclusive_cumsum = _cumsum_matmul(blocks_per_expert.unsqueeze(1))  # [E_local, 1]

    # Create exclusive cumsum by prepending 0 and removing last element
    zero = torch.zeros(1, 1, dtype=inclusive_cumsum.dtype, device=device)
    cumulative_blocks_per_expert = torch.cat(
        [zero, inclusive_cumsum[:-1]], dim=0
    )  # [E_local, 1]

    # Step 2: Flatten indices to token_position_to_id using kernel
    row_offsets = cumulative_blocks_per_expert * (block_size // f_len)

    if sharded:
        # [E_local/TP, T] --> [N * block_size,]
        token_position_to_id_padded = indexed_flatten_nki[2](
            input_tensor=indices,
            f_len=f_len,
            output_len=num_blocks * block_size + total_tokens,
            row_offsets=row_offsets.reshape(-1).to(torch.int32),
            row_offsets_start=col_start_id,
            padding_val=-1,
        )

        # Aggregate across ranks using max reduction (no SUM equivalent
        # on GroupCoordinator, so use device_group directly for MAX op)
        token_position_to_id = token_position_to_id_padded[
            : num_blocks * block_size
        ].clone()
        dist.all_reduce(
            token_position_to_id,
            op=dist.ReduceOp.MAX,
            group=moe_group.device_group,
        )
    else:
        # [E_local, T] --> [N * block_size,]
        token_position_to_id_padded = indexed_flatten_nki[2](
            input_tensor=indices,
            f_len=f_len,
            output_len=num_blocks * block_size + total_tokens,
            row_offsets=row_offsets.reshape(-1).to(torch.int32),
            # torch.zeros over torch.tensor([0], ...): see the col_start_id
            # comment above (RankConstantCheckPass heuristic).
            row_offsets_start=torch.zeros(1, dtype=torch.int32, device=device),
            padding_val=-1,
        )
        token_position_to_id = token_position_to_id_padded[
            : num_blocks * block_size
        ].clone()

    block_ids = torch.arange(num_blocks, device=device, dtype=torch.int32)  # [N, ]
    block_to_expert = torch.sum(
        block_ids.unsqueeze(0) >= cumulative_blocks_per_expert[1:], dim=0
    ).to(torch.int32)  # [N, ]

    return token_position_to_id, block_to_expert, num_blocks


def _build_blockwise_mapping_torch(
    expert_mask: Tensor,
    num_local_experts: int,
    num_experts_per_token: int,
    block_size: int,
    total_tokens: int,
    tp_degree: int,
    moe_group: "GroupCoordinator",
) -> tuple[Tensor, Tensor, int]:
    """
    Build blockwise mapping using pure PyTorch operations.

    This is the fallback path when NKI kernels cannot be used (e.g., on CPU or when
    constraints are not met). Each rank processes its shard of experts and collectives
    combine the results (same pattern as kernel path).

    Args:
        expert_mask: [T, E_local] binary mask indicating selected experts
        num_local_experts: total number of local experts (E_local)
        num_experts_per_token: top-k experts per token
        block_size: tokens per block
        total_tokens: total number of tokens (T)
        tp_degree: tensor parallelism degree
        moe_group: GroupCoordinator for MoE collectives

    Returns:
        token_position_to_id: [N*block_size] mapping from positions to token IDs
        block_to_expert: [N] expert assignment per block
        num_blocks: total number of blocks (N)
    """
    device = expert_mask.device
    E_local = num_local_experts
    # SPMD: this rank's on-device rank tensor (stamped onto the group) so the
    # expert-column shard is selected by a graph INPUT, not a baked per-rank constant.
    # Callers stamp via stamp_spmd_rank_tensors (model runner / executor; tests too).
    rank_tensor = moe_group.rank_in_group_tensor

    def _shard_cols(t, e_shard):
        """Select this rank's [.., e_shard] expert-column block. index_select on a
        rank-tensor offset keeps a static width with dynamic values (rank-invariant
        graph)."""
        idx = rank_tensor.to(torch.int64) * e_shard + torch.arange(
            e_shard, device=t.device, dtype=torch.int64
        )
        return t.index_select(1, idx)

    # Step 1: Get tokens_per_expert — sharded across ranks when possible
    if E_local % tp_degree == 0 and tp_degree > 1:
        E_shard = E_local // tp_degree
        tokens_per_expert_local = torch.sum(_shard_cols(expert_mask, E_shard), dim=0)
        tokens_per_expert = moe_group.all_gather(tokens_per_expert_local, dim=0)
    else:
        tokens_per_expert = torch.sum(expert_mask, dim=0)  # [E_local]

    # Compute block mapping
    blocks_per_expert = ((tokens_per_expert + block_size - 1) // block_size).to(
        dtype=torch.long
    )
    total_local_tokens = total_tokens * min(num_experts_per_token, num_local_experts)
    num_blocks = (
        math.ceil((total_local_tokens - (num_local_experts - 1)) / block_size)
        + num_local_experts
        - 1
    )
    num_blocks = min(num_blocks, total_local_tokens)

    block_ids = torch.arange(num_blocks, device=device, dtype=torch.long)
    cumulative_blocks_per_expert = _cumsum_matmul(
        blocks_per_expert.unsqueeze(1)
    ).squeeze(1)
    block_to_expert = torch.sum(
        block_ids.unsqueeze(1) >= cumulative_blocks_per_expert[:-1], dim=1
    ).to(torch.long)

    # Step 2: Flatten indices to token_position_to_id — sharded across ranks
    token_position_by_id_and_expert = _cumsum_matmul(expert_mask)  # [T, E_local]
    expert_block_offsets = cumulative_blocks_per_expert * block_size
    token_position_by_id_and_expert[:, 1:] += expert_block_offsets[:-1]
    token_position_by_id_and_expert = token_position_by_id_and_expert.masked_fill(
        expert_mask == 0, 0
    ).to(dtype=torch.long)

    tokens_idx = torch.arange(total_tokens, device=device, dtype=torch.long)

    if E_local % tp_degree == 0 and tp_degree > 1:
        # Each rank scatters only its shard of experts into the output
        E_shard = E_local // tp_degree
        local_positions = _shard_cols(token_position_by_id_and_expert, E_shard)
        local_mask = _shard_cols(expert_mask, E_shard)
        local_positions = local_positions.masked_fill(local_mask == 0, 0).to(
            dtype=torch.long
        )

        token_position_to_id = -1 * torch.ones(
            num_blocks * block_size + 1, device=device, dtype=torch.long
        )
        token_position_to_id[local_positions] = tokens_idx.unsqueeze(1)
        token_position_to_id = token_position_to_id[1:]

        # Aggregate across ranks using max reduction
        dist.all_reduce(
            token_position_to_id,
            op=dist.ReduceOp.MAX,
            group=moe_group.device_group,
        )
    else:
        token_position_to_id = -1 * torch.ones(
            num_blocks * block_size + 1, device=device, dtype=torch.long
        )
        token_position_to_id[token_position_by_id_and_expert] = tokens_idx.unsqueeze(1)
        token_position_to_id = token_position_to_id[1:]

    return token_position_to_id, block_to_expert, num_blocks


def _cumsum_matmul(tensor: Tensor, dim: int = 0, tril_size: int = 2048) -> Tensor:
    """Cumulative sum using matrix multiplication for better hardware utilization."""
    if len(tensor.shape) != 2:
        raise ValueError(f"Expected 2D input tensor, got shape: {tensor.shape}")
    if dim != 0:
        raise NotImplementedError(f"Only dim=0 supported, got dim={dim}")

    dtype = tensor.dtype
    num_tokens = tensor.shape[0]

    if num_tokens % tril_size == 0:
        num_iters = num_tokens // tril_size
        last_iter_tokens = tril_size
    else:
        num_iters = num_tokens // tril_size + 1
        last_iter_tokens = num_tokens % tril_size

    results = []
    rolling_sum = torch.zeros(
        1, tensor.shape[1], dtype=torch.float32, device=tensor.device
    )

    for i in range(num_iters):
        iter_tokens = tril_size if i < num_iters - 1 else last_iter_tokens
        tril = torch.tril(
            torch.ones(
                iter_tokens, iter_tokens, device=tensor.device, dtype=torch.float32
            )
        )
        input_slice = tensor.narrow(0, i * tril_size, iter_tokens).to(
            dtype=torch.float32
        )
        output_slice = rolling_sum + torch.matmul(tril, input_slice)
        results.append(output_slice)
        if i < num_iters - 1:
            rolling_sum = output_slice.narrow(0, -1, 1)

    return torch.cat(results, dim=0).to(dtype)


def _apply_padding_mask(
    expert_affinities: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Zeros out expert affinities for padded tokens.

    Args:
        expert_affinities: Tensor of shape (T, E) with expert affinities.
        padding_mask: Tensor of shape (T,) where True = real token, False = padding.
            If None, all tokens are treated as real (no masking applied).

    Returns:
        expert_affinities with padding tokens zeroed out.
    """
    if padding_mask is None:
        return expert_affinities

    return expert_affinities * padding_mask.float().unsqueeze(1)  # [T, 1]


def _can_use_find_nonzero_indices_kernel(
    T: int,
    E: int,
    chunk_size: int,
    tensor: Tensor,
) -> bool:
    """Check if find_nonzero_indices kernel can be used."""
    if not can_run_kernel(tensor):
        return False

    # E must be divisible by logical_nc_config (2)
    if E % 2 != 0:
        return False

    # T must be divisible by chunk_size
    if T % chunk_size != 0:
        return False

    # chunk size must be divisible by 128 (partition size)
    if chunk_size % 128 != 0:
        return False

    return True


def _can_use_indexed_flatten_kernel(
    T: int,
    tensor: Tensor,
    f_len: int,
) -> bool:
    """Check if indexed_flatten kernel can be used."""
    if not can_run_kernel(tensor):
        return False

    # T must be divisible by f_len
    if T % f_len != 0:
        return False

    # (T // f_len) must be divisible by 16
    if (T // f_len) % 16 != 0:
        return False

    return True
