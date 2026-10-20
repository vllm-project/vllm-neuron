# SPDX-License-Identifier: Apache-2.0
import math
import os
import tempfile

import numpy as np

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import dge_mode

from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast

import torch

from vllm.distributed.parallel_state import GroupCoordinator, get_world_group
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel
from ..argsort_unstable import argsort_unstable, _argsort_unstable_nki


def permute_routed_tokens(
    hidden_input: torch.Tensor,
    expert_index: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    group: GroupCoordinator,
    is_sequence_parallel: bool = False,
    pack_expert_index: bool = False,
    affinity_slice_size: int | None = None,
) -> torch.Tensor:
    """
    Prepare tokens for all2all dispatch by permuting tokens by destination rank and
    concatenating hidden_input, per-rank affinities, and token indices. This enables hidden
    states and metadata to be dispatched in a single all2all call.

    When a token is routed to multiple experts that are on the same destination rank,
    the token appears only once in the contiguous group of rows that will be dispatched to that rank.
    This means the output buffer may have trailing padding rows whose contents are undefined.

    Args:
        hidden_input (torch.Tensor): [T, n_input_cols] bf16 or fp8 tensor of hidden states.
        expert_index (torch.Tensor): [T, K] int32 tensor of top-K expert indices per token.
        expert_affinities_masked (torch.Tensor): [T, E] bf16 tensor of expert affinities,
            with zeros for non-routed token/expert pairs.
        group (GroupCoordinator): The distributed group coordinator. Its world_size determines
            the number of destination ranks (n_dst_ranks). Routing is always based on
            ``E // group.world_size`` experts per dst.
        is_sequence_parallel (bool): If True, token indices are global. The rank offset
            (rank_id * T) is added to local token indices so that each rank's tokens have
            globally unique IDs. In SP mode the per-rank offset is derived from the
            on-device ``get_world_group().rank_in_group_tensor`` (stamped by
            ``stamp_spmd_rank_tensors``) so the traced graph is identical across
            ranks (SPMD). Defaults to False.
        pack_expert_index (bool): If True, append [T, K] expert_index (as int32 in 2*K
            hidden-dtype cols) to each row; used by hierarchical dispatch's inter-server stage.
        affinity_slice_size (int | None): Affinity columns per row. Defaults to
            ``E // group.world_size``. Override to size affinities for a finer
            dst-rank block than routing uses (e.g. inter-server stage in
            hierarchical dispatch). Must divide E and be <= ``E // group.world_size``.

    Returns:
        torch.Tensor: [T*K, n_output_cols] tensor where each row is
            [hidden_state, local_affinities, token_index] sorted by destination rank.
            token_index is 1-indexed (1..T) because MoE will determine routed tokens by checking for nonzero indices.
            The contents of the padding region (rows beyond the actual token count) are undefined.

    Example:
        >>> # T=4 tokens, K=2, E=8 experts, 4 ranks (group.world_size=4), fp8 hidden
        >>> hidden_input = torch.randn(4, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        >>> expert_index = torch.tensor([[0, 1], [2, 5], [7, 4], [1, 6]], dtype=torch.int32)
        >>> affinities = torch.zeros(4, 8, dtype=torch.bfloat16)
        >>> out = permute_routed_tokens(hidden_input, expert_index, affinities, group=group)
        >>> # n_local_experts = E // world_size = 2
        >>> # Token 0 → experts 0,1 → both rank 0 (DEDUP: counts as 1 row)
        >>> # Token 1 → experts 2,5 → ranks 1,2 (2 rows)
        >>> # Token 2 → experts 7,4 → ranks 3,2 (2 rows)
        >>> # Token 3 → experts 1,6 → ranks 0,3 (2 rows)
        >>> # Total valid rows = 7, padding rows = 1 (T*K=8 total)
        >>> #
        >>> # Output rows grouped by dst rank:
        >>> #   Row 0: token 0 → rank 0
        >>> #   Row 1: token 3 → rank 0
        >>> #   Row 2: token 1 → rank 1
        >>> #   Row 3: token 1 → rank 2
        >>> #   Row 4: token 2 → rank 2
        >>> #   Row 5: token 2 → rank 3
        >>> #   Row 6: token 3 → rank 3
        >>> #   Row 7: [padding — contents undefined]
        >>> # out.shape = (T*K=8, H*2 + n_local_experts + 2) = (8, 260)
    """

    # Convert from GroupCoordinator -> size
    replica_group_size = group.world_size

    _validate_inputs(
        hidden_input,
        expert_index,
        expert_affinities_masked,
        replica_group_size,
        affinity_slice_size,
    )

    # 1-indexed token ids, offset per rank for sequence parallel.
    # SPMD: in SP mode the per-rank base id (rank * T + 1) is a runtime tensor read
    # from the stamped ``get_world_group().rank_in_group_tensor`` — not a baked
    # ``dist.get_rank()`` constant — so every rank traces one identical graph. The
    # one-token kernel and torch fallback consume this tensor directly; the a2av
    # kernel consumes the equivalent per-rank ``rank_id`` tensor (rank * T added
    # inside the kernel).
    T, _ = expert_index.shape
    # Non-SP branch: rank-invariant constant 1. Use ones() rather than
    # torch.tensor([1], ...) to sidestep RankConstantCheckPass's small-int
    # singleton heuristic under SPMD strict.
    token_base_index = (
        get_world_group().rank_in_group_tensor * T + 1
        if is_sequence_parallel
        else torch.ones(1, dtype=torch.int32, device=hidden_input.device)
    )

    # Single-token packed path (hierarchical inter-server stage) routes to the dedicated kernel.
    if _can_use_one_token_kernel(
        hidden_input,
        expert_index,
        expert_affinities_masked,
        pack_expert_index,
        affinity_slice_size,
    ):
        wrapped = wrap_nki(_permute_one_token_nki)
        return wrapped[2](
            hidden_states=hidden_input,
            expert_affinities=expert_affinities_masked,
            expert_index=expert_index,
            token_base_index=token_base_index,
        )

    elif _can_use_kernel(
        hidden_input,
        expert_index,
        expert_affinities_masked,
        replica_group_size,
        pack_expert_index,
        affinity_slice_size,
    ):
        # The a2av kernel takes the per-rank offset as a [1, 1] ``rank_id`` tensor
        # (it adds rank * T to rank-invariant iota ids inside the kernel). Source
        # it from the stamped world-group rank tensor so the graph is SPMD.
        rank_id = (
            get_world_group().rank_in_group_tensor.reshape(1, 1).to(torch.int32)
            if is_sequence_parallel
            else None
        )
        wrapped = wrap_nki(_permute_routed_tokens_a2av_nki)
        return wrapped[2](
            hidden_input=hidden_input,
            expert_index=expert_index,
            expert_affinities_masked=expert_affinities_masked,
            replica_group_size=replica_group_size,
            rank_id=rank_id,
        )

    return _torch_impl(
        hidden_input,
        expert_index,
        expert_affinities_masked,
        replica_group_size,
        token_base_index,
        pack_expert_index,
        affinity_slice_size,
    )


def _validate_inputs(
    hidden_input: torch.Tensor,
    expert_index: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    replica_group_size: int,
    affinity_slice_size: int | None = None,
) -> None:
    """Validate inputs for permute_routed_tokens."""
    T, K = expert_index.shape
    _, E = expert_affinities_masked.shape

    assert hidden_input.shape[0] == T, (
        f"hidden_input rows ({hidden_input.shape[0]}) must match expert_index rows ({T})"
    )
    assert expert_affinities_masked.shape[0] == T, (
        f"expert_affinities rows ({expert_affinities_masked.shape[0]}) must match T ({T})"
    )
    assert E % replica_group_size == 0, (
        f"E must be divisible by replica_group_size, got {E=}, {replica_group_size=}"
    )
    assert not (hidden_input.element_size() < 2 and hidden_input.shape[-1] % 2 != 0), (
        f"Expected dim1 of hidden_input divisible by 2 when hidden_input is fp8, got {hidden_input.shape=} {hidden_input.dtype=}"
    )
    if affinity_slice_size is not None:
        num_experts_per_rank = E // replica_group_size
        assert E % affinity_slice_size == 0, (
            f"E must be divisible by affinity_slice_size, got {E=}, {affinity_slice_size=}"
        )
        if affinity_slice_size < E:
            assert affinity_slice_size <= num_experts_per_rank, (
                f"affinity_slice_size ({affinity_slice_size}) must be <= "
                f"num_experts_per_rank ({num_experts_per_rank}); the slice must "
                f"fit within a single dst-rank's expert block."
            )
            assert num_experts_per_rank % affinity_slice_size == 0, (
                f"num_experts_per_rank ({num_experts_per_rank}) must be divisible "
                f"by affinity_slice_size ({affinity_slice_size}) for a clean "
                f"nesting of affinity blocks within dst-rank blocks."
            )


def _can_use_kernel(
    hidden_input: torch.Tensor,
    expert_index: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    replica_group_size: int,
    pack_expert_index: bool = False,
    affinity_slice_size: int | None = None,
) -> bool:
    """Check if the NKI kernel can be used for permute_routed_tokens."""
    if not can_run_kernel(hidden_input):
        return False

    # pack_expert_index / affinity_slice_size are hierarchical-only; kernel
    # doesn't support them yet, so fall back to the torch impl.
    if pack_expert_index or affinity_slice_size is not None:
        return False

    T, K = expert_index.shape
    _, E = expert_affinities_masked.shape

    if K != 4:
        return False
    if T * K > 128:
        return False
    if (T * K) % 8 != 0:
        return False
    if E % replica_group_size != 0:
        return False

    n_local = E // replica_group_size
    if n_local != 2:
        return False

    if hidden_input.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        return False

    return True


def _can_use_one_token_kernel(
    hidden_input: torch.Tensor,
    expert_index: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    pack_expert_index: bool = False,
    affinity_slice_size: int | None = None,
) -> bool:
    """Check if the single-token packed NKI kernel (_permute_one_token_nki) applies.

    Handles the hierarchical inter-server stage: a single token (T=1) with packed
    expert indices and unsliced affinities.

    Kernel constraints:
        - Device must support NKI kernels (can_run_kernel)
        - T == 1 (single token)
        - H divisible by 256 (H sharded across LNC, each shard a pmax multiple)
        - dtypes: hidden bf16, expert_index int32, affinities bf16
        - pack_expert_index is True
        - affinities are full-width (affinity_slice_size == E)
    """
    if not can_run_kernel(hidden_input):
        return False
    return (
        hidden_input.shape[0] == 1  # T = 1
        and hidden_input.shape[1] % 256 == 0  # H sharded across LNC, pmax-aligned
        and hidden_input.dtype == torch.bfloat16
        and expert_index.dtype == torch.int32
        and expert_affinities_masked.dtype == torch.bfloat16
        and pack_expert_index
        and affinity_slice_size == expert_affinities_masked.shape[1]
    )


def _torch_impl(
    hidden_input: torch.Tensor,
    expert_index: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    replica_group_size: int,
    token_base_index: torch.Tensor = 1,
    pack_expert_index: bool = False,
    affinity_slice_size: int | None = None,
):
    """Torch implementation of permute_routed_tokens.

    ``token_base_index`` is a [1] int32 runtime tensor (or int 1) added to the
    rank-invariant 0-based token indices, so the traced graph is identical across
    ranks (SPMD).
    """

    # Step 1: Extract constants
    T, K = expert_index.shape
    _, E = expert_affinities_masked.shape
    num_experts_per_rank = E // replica_group_size
    # Routing keeps using num_experts_per_rank (= dst-rank block size). Affinities
    # may be sliced finer (e.g. for hierarchical inter-server stage that routes by
    # node but lays out the recv buffer for the eventual intra-server dst rank).
    num_affinity_per_rank = (
        affinity_slice_size if affinity_slice_size is not None else num_experts_per_rank
    )
    slice_expert_affinities = num_affinity_per_rank < E

    # Step 2: Build inverse argsort array, with adjustment for de-duped token/rank pairs
    # Step 2.1: Build de-duped expert rank mapping
    expert_ranks = (expert_index // num_experts_per_rank).to(torch.int32)
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

    # Step 3: Concatenate [hidden | affinities | token idx | expert idx (optional)], with bitcast to hidden.dtype
    # Step 3.1: Broadcast [T, H] -> [T*K, H]
    hidden_input_bc_K = hidden_input.unsqueeze(1).expand(-1, K, -1).reshape(T * K, -1)

    # Step 3.2: Gather affinities per T, K pair. Slice size defaults to the
    # dst-rank block (num_experts_per_rank) but can be overridden via
    # num_affinity_per_rank — in that case we gather the FINER block
    # corresponding to the expert's owning rank under the finer slice.
    affinity_block_id = (expert_index // num_affinity_per_rank).to(torch.int32)
    offsets = affinity_block_id * num_affinity_per_rank  # [T, K]
    local_idx = torch.arange(
        num_affinity_per_rank, dtype=torch.int32, device=expert_ranks.device
    )
    gather_idx = offsets.unsqueeze(-1) + local_idx  # [T, K, n_affinity]

    # Expand affinities to [T * K, num_affinity_per_rank]
    if slice_expert_affinities:
        # When num_affinity_per_rank < E, slice affinities based on dst rank
        expert_affinities_masked_prepared = (
            expert_affinities_masked.unsqueeze(1)
            .expand(-1, K, -1)
            .gather(2, gather_idx.to(torch.int32))
            .reshape(T * K, num_affinity_per_rank)
        )
    else:
        # When num_affinity_per_rank == E, skip slicing expert affinities and only broadcast
        expert_affinities_masked_prepared = (
            expert_affinities_masked.unsqueeze(1).expand(-1, K, -1).reshape(T * K, -1)
        )

    # Step 3.3: Build token indices [T*K, 1], starting from token_base_index.
    # The 0-based arange is rank-invariant and ``token_base_index`` (a [1] int32
    # runtime tensor in SP mode) is added as a tensor op, so the traced graph is
    # identical across ranks (one shared NEFF). The stamped rank tensor is
    # swapped to meta for the duration of the parallel-trace fork pool (see
    # ``spmd_rank_tensors_on_meta``), so the add is device-consistent both under
    # the meta trace and at real-device runtime.
    token_indices = (
        (
            torch.arange(T, dtype=torch.int32, device=expert_index.device)
            + token_base_index
        )
        .repeat_interleave(K)
        .reshape(T * K, 1)
    )

    # Step 3.4: Broadcast expert index to [T * K, K]
    if pack_expert_index:
        expert_index_bc_K = (
            expert_index.unsqueeze(1).expand(-1, K, -1).reshape(T * K, -1)
        )

    # Step 3.5: Concatenate with bitcast to hidden_input.dtype
    concat_list = [
        hidden_input_bc_K,
        _bitcast(expert_affinities_masked_prepared, hidden_input_bc_K.dtype),
        _bitcast(token_indices, hidden_input_bc_K.dtype),
    ]
    if pack_expert_index:
        concat_list.append(_bitcast(expert_index_bc_K, hidden_input_bc_K.dtype))

    data_concat = torch.concat(concat_list, dim=1)

    # Step 4: Group tokens by destination rank, with de-dupe
    # Bitcast to a same-width integer dtype, which is supported on CPU and doesn't canonicalize NaNs.
    if str(expert_index.device) == "cpu":
        int_scatter_dtype = (
            torch.int8 if data_concat.element_size() == 1 else torch.int16
        )
        data_concat = _bitcast(data_concat, int_scatter_dtype)

    # Scatter tokens into output buffer using adjusted inv argsort array. De-dupes are scattered into row 0
    # NOTE: padded rows have token index -2 for better debuggability. -2 index post dispatch = metadata was incorrect.
    n_idx_cols = 4 // data_concat.element_size()
    n_expert_cols = K * n_idx_cols if pack_expert_index else 0
    n_data_cols = data_concat.shape[-1] - n_idx_cols - n_expert_cols
    n_rows = T * K + 1
    zeros_part = torch.zeros(
        (n_rows, n_data_cols),
        dtype=data_concat.dtype,
        device=expert_index.device,
    )
    neg_two_int32 = torch.full(
        (n_rows, 1), -2, dtype=torch.int32, device=expert_index.device
    )
    neg_two_native = _bitcast(neg_two_int32, data_concat.dtype)
    concat_list = [zeros_part, neg_two_native]
    if pack_expert_index:
        # NOTE: padded rows have expert index -1, which is expected by hierarchical a2av inter-server/intra-server transition logic.
        neg_one_int32 = torch.full(
            (n_rows, K), -1, dtype=torch.int32, device=expert_index.device
        )
        concat_list.append(_bitcast(neg_one_int32, data_concat.dtype))
    output_permuted = torch.concat(concat_list, dim=1)
    output_permuted.scatter_(
        0, token_inv_argsort_adjusted.unsqueeze(1).expand_as(data_concat), data_concat
    )

    # CPU mode does not support fp8 scatter_; convert back to hidden.dtype
    if str(expert_index.device) == "cpu":
        output_permuted = _bitcast(output_permuted, hidden_input_bc_K.dtype)

    # Discard row 0, which contains garbage/de-duped tokens
    return output_permuted[1:, :]


# FIXME: everything below this line is a hack, remove when FX->HLO natively supports bitcasting
def _bitcast(data, dtype):
    # Conversion map
    _TORCH_NKI_DTYPE_MAP = {
        torch.float8_e4m3fn: nl.float8_e4m3fn,
        torch.bfloat16: nl.bfloat16,
        torch.int32: nl.int32,
    }
    # Same-dtype bitcast is a no-op; skip it
    if data.dtype == dtype:
        return data
    elif str(data.device) != "cpu":
        wrapped = wrap_nki(_bitcast_nki)
        nki_dtype = _TORCH_NKI_DTYPE_MAP[dtype]
        return wrapped[2](data, nki_dtype)
    else:
        return data.view(dtype)


@nki.jit
def _bitcast_nki(data, nki_dtype):
    data = data.view(nki_dtype)
    data_new_dtype = nl.ndarray(data.shape, nki_dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(data_new_dtype, data)

    return data_new_dtype


# ── NKI kernel implementations ────────────────────────────────────


def _build_token_major_select_npy(T, K):
    TK = T * K
    sel = np.zeros((T, K, TK), dtype=np.float32)
    for k in range(K):
        for t in range(T):
            sel[t, k, t * K + k] = 1.0
    fd, path = tempfile.mkstemp(prefix=f"prt_a2av_sel_{T}x{K}_", suffix=".npy")
    os.close(fd)
    np.save(path, sel)
    return path


_TOKEN_MAJOR_SELECT_NPY = {
    f"{_T}_4": _build_token_major_select_npy(_T, 4) for _T in range(2, 33, 2)
}

_SUPPORTED_K = [1, 2, 4, 8]
_SUPPORTED_HIDDEN_DTYPES = [
    nl.bfloat16,
    nl.float8_e4m3,
    nl.float8_e4m3fn,
    nl.float8_e5m2,
]
_EXPERT_AFFINITY_COLS = {
    nl.bfloat16: 1,
    nl.float8_e4m3: 2,
    nl.float8_e4m3fn: 2,
    nl.float8_e5m2: 2,
}
_TOKEN_INDEX_COLS = {
    nl.bfloat16: 2,
    nl.float8_e4m3: 4,
    nl.float8_e4m3fn: 4,
    nl.float8_e5m2: 4,
}


def _broadcast_scalar_to_partitions(scalar_11_sb, P, dtype=nl.float32):
    out_sb = nl.ndarray((P, 1), dtype=dtype, buffer=nl.sbuf)
    stream_shuffle_broadcast(src=scalar_11_sb, dst=out_sb)
    return out_sb


def _fill_padding_row(hbm_tensor, row_idx, n_data_cols, tok_cols, dtype):
    total_cols = n_data_cols + tok_cols
    template_sb = nl.ndarray((1, total_cols), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(dst=template_sb[:, :n_data_cols], value=0)
    neg_two_i32 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=neg_two_i32, value=-2)
    nisa.tensor_copy(
        src=neg_two_i32.view(dtype),
        dst=template_sb[:, n_data_cols:],
    )
    nisa.dma_copy(dst=hbm_tensor[row_idx : row_idx + 1, :], src=template_sb)


def _validate_and_extract_shapes_nki(
    hidden_input, expert_index, expert_affinities_masked, replica_group_size
):
    T, n_cols = hidden_input.shape
    T_expert, K = expert_index.shape
    T_affinity, E = expert_affinities_masked.shape
    pmax = nl.tile_size.pmax
    kernel_assert(
        hidden_input.dtype in _SUPPORTED_HIDDEN_DTYPES,
        f"Expected hidden_input.dtype in {_SUPPORTED_HIDDEN_DTYPES} but got {hidden_input.dtype=}",
    )
    kernel_assert(K in _SUPPORTED_K, f"Expected K in {_SUPPORTED_K}, got {K=}")
    kernel_assert(T * K <= pmax, f"Expected T * K <= {pmax}, got {T=}, {K=}")
    kernel_assert(
        T * K % 8 == 0,
        f"Expected T * K divisible by 8 (argsort constraint), got {T=}, {K=}",
    )
    kernel_assert(
        E % replica_group_size == 0,
        f"Expected E divisible by replica_group_size, got {E=}, {replica_group_size=}",
    )
    n_local = E // replica_group_size
    kernel_assert(
        n_local > 0 and (n_local & (n_local - 1)) == 0,
        f"Expected num_experts_per_rank to be a power of two, got {n_local=}",
    )
    kernel_assert(
        T == T_expert and T == T_affinity,
        f"Expected same dim0, but got {T}, {T_expert}, {T_affinity}",
    )
    return T, n_cols, K, E


@nki.jit
def _permute_routed_tokens_a2av_nki(
    hidden_input: nl.NkiTensor,
    expert_index: nl.NkiTensor,
    expert_affinities_masked: nl.NkiTensor,
    replica_group_size: int,
    rank_id: nl.NkiTensor = None,
):
    T, n_cols, K, E = _validate_and_extract_shapes_nki(
        hidden_input, expert_index, expert_affinities_masked, replica_group_size
    )
    dtype = hidden_input.dtype
    TK = T * K

    n_local = E // replica_group_size
    kernel_assert(
        K == 4 and n_local == 2,
        "Permute routed token is only validated for K=4 and E_local=2",
    )
    log2_K = int(math.log2(K))
    log2_n_local = int(math.log2(n_local))
    aff_unit = _EXPERT_AFFINITY_COLS[dtype]
    tok_cols = _TOKEN_INDEX_COLS[dtype]
    aff_cols = n_local * aff_unit
    total_cols = n_cols + aff_cols + tok_cols

    if nl.program_id(0) == 1:
        output_hbm = nl.ndarray(
            (TK, total_cols),
            dtype=dtype,
            buffer=nl.shared_hbm,
            name="prt_a2av_output_hbm",
        )
        return output_hbm

    # Compute de-duped destination rank per (t, k)
    idx_free_i32 = nl.ndarray((T, K), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_free_i32, src=expert_index)
    ranks_free_i32 = nl.ndarray((T, K), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        data=idx_free_i32, op0=nl.right_shift, operand0=log2_n_local, dst=ranks_free_i32
    )
    ranks_free = nl.ndarray((T, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ranks_free, src=ranks_free_i32)

    # De-dup across K
    sentinel = float(replica_group_size)
    deduped_free = nl.ndarray((T, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=deduped_free[:, 0:1], src=ranks_free[:, 0:1])
    for k in range(1, K):
        matched = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=matched, value=0.0)
        for j in range(k):
            eq = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=eq,
                data1=ranks_free[:, k : k + 1],
                data2=deduped_free[:, j : j + 1],
                op=nl.equal,
            )
            nisa.tensor_tensor(dst=matched, data1=matched, data2=eq, op=nl.maximum)
        gap = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            data=ranks_free[:, k : k + 1],
            op0=nl.subtract,
            operand0=sentinel,
            reverse0=True,
            dst=gap,
        )
        nisa.tensor_tensor(dst=gap, data1=gap, data2=matched, op=nl.multiply)
        nisa.tensor_tensor(
            dst=deduped_free[:, k : k + 1],
            data1=ranks_free[:, k : k + 1],
            data2=gap,
            op=nl.add,
        )

    # Flatten de-duped ranks [T, K] -> [1, T*K] via matmul
    sel_npy_path = _TOKEN_MAJOR_SELECT_NPY[f"{T}_{K}"]
    sel_hbm = nl.shared_constant(sel_npy_path)
    sel_sb = nl.ndarray((T, K, TK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sel_sb, src=sel_hbm)
    deduped_flat_psum = nl.ndarray((1, TK), dtype=nl.float32, buffer=nl.psum)
    for k in range(K):
        nisa.nc_matmul(
            dst=deduped_flat_psum,
            stationary=deduped_free[:, k : k + 1],
            moving=sel_sb[:, k, :],
            is_moving_onezero=True,
            accumulate=(k != 0),
        )
    deduped_flat_sb = nl.ndarray((1, TK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=deduped_flat_sb, src=deduped_flat_psum)

    # valid_count = T*K - #sentinel
    dedupe_mask = nl.ndarray((1, TK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        data=deduped_flat_sb, op0=nl.equal, operand0=sentinel, dst=dedupe_mask
    )
    dedupe_count_sb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=dedupe_count_sb, op=nl.add, data=dedupe_mask, axis=(1,))
    valid_count_sb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        data=dedupe_count_sb,
        op0=nl.subtract,
        operand0=float(TK),
        reverse0=True,
        dst=valid_count_sb,
    )

    # Argsort de-duped ranks
    argsort_idx_F_sb = _argsort_unstable_nki(
        data=deduped_flat_sb, descending=False, output_in_sbuf=True
    )

    # Transpose argsort indices to [T*K, 1]
    argsort_psum = nl.ndarray((TK, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(argsort_psum, argsort_idx_F_sb.view(nl.float32))
    argsort_T_sb = nl.ndarray((TK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=argsort_T_sb, src=argsort_psum)
    sorted_slot_f32 = nl.ndarray((TK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sorted_slot_f32, src=argsort_T_sb.view(nl.uint32))

    # Build packed rows [T*K + 1, total_cols]
    packed_hbm = nl.ndarray(
        (TK + 1, total_cols),
        dtype=dtype,
        buffer=nl.shared_hbm,
        name="prt_a2av_packed_hbm",
    )

    iota_p_i32 = nl.ndarray((TK, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=iota_p_i32, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    slot_token_i32 = nl.ndarray((TK, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        data=iota_p_i32, op0=nl.right_shift, operand0=log2_K, dst=slot_token_i32
    )

    # Hidden broadcast
    for k in range(K):
        nisa.dma_copy(
            dst=packed_hbm.ap(
                pattern=[[K * total_cols, T], [1, n_cols]], offset=k * total_cols
            ),
            src=hidden_input,
        )

    # Affinity gather + token index
    meta_cols = aff_cols + tok_cols
    meta_sb = nl.ndarray((TK, meta_cols), dtype=dtype, buffer=nl.sbuf)

    idx_part_i32 = nl.ndarray((TK, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_part_i32, src=expert_index.reshape((TK, 1)))
    rank_nlocal_i32 = nl.ndarray((TK, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        data=idx_part_i32,
        op0=nl.right_shift,
        operand0=log2_n_local,
        op1=nl.left_shift,
        operand1=log2_n_local,
        dst=rank_nlocal_i32,
    )
    aff_base_i32 = nl.ndarray((TK, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        data=slot_token_i32, op0=nl.multiply, operand0=E, dst=aff_base_i32
    )
    nisa.tensor_tensor(
        dst=aff_base_i32, data1=aff_base_i32, data2=rank_nlocal_i32, op=nl.add
    )
    aff_base_u32 = aff_base_i32.view(nl.uint32)
    expert_affinities_flat = expert_affinities_masked.reshape((T * E, 1))
    nisa.dma_copy(
        src=expert_affinities_flat.ap(
            pattern=[[aff_unit, TK], [1, aff_cols]],
            offset=0,
            vector_offset=aff_base_u32,
            indirect_dim=0,
            dtype=dtype,
        ),
        dst=meta_sb[:, :aff_cols],
    )

    # Token index
    token_idx_i32 = nl.ndarray((TK, 1), dtype=nl.int32, buffer=nl.sbuf)
    if rank_id != None:  # noqa: E711
        rank_id_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=rank_id_sb, src=rank_id)
        offset_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(data=rank_id_sb, op0=nl.multiply, operand0=T, dst=offset_sb)
        offset_bc = _broadcast_scalar_to_partitions(offset_sb, TK, dtype=nl.int32)
        nisa.tensor_tensor(
            dst=token_idx_i32, data1=slot_token_i32, data2=offset_bc, op=nl.add
        )
        nisa.tensor_scalar(
            data=token_idx_i32, op0=nl.add, operand0=1, dst=token_idx_i32
        )
    else:
        nisa.tensor_scalar(
            data=slot_token_i32, op0=nl.add, operand0=1, dst=token_idx_i32
        )
    nisa.tensor_copy(src=token_idx_i32.view(dtype), dst=meta_sb[:, aff_cols:])

    # Write metadata to packed_hbm
    nisa.dma_copy(
        dst=packed_hbm.ap(pattern=[[total_cols, TK], [1, meta_cols]], offset=n_cols),
        src=meta_sb,
    )
    _fill_padding_row(packed_hbm, TK, n_cols + aff_cols, tok_cols, dtype)

    # Gather-permute
    iota_p_f32 = nl.ndarray((TK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=iota_p_f32, src=iota_p_i32)
    valid_bc = _broadcast_scalar_to_partitions(valid_count_sb, TK)
    is_real = nl.ndarray((TK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=is_real, data1=iota_p_f32, data2=valid_bc, op=nl.less)
    gather_src_f32 = nl.ndarray((TK, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        data=sorted_slot_f32, op0=nl.subtract, operand0=float(TK), dst=gather_src_f32
    )
    nisa.tensor_tensor(
        dst=gather_src_f32, data1=gather_src_f32, data2=is_real, op=nl.multiply
    )
    nisa.tensor_scalar(
        data=gather_src_f32, op0=nl.add, operand0=float(TK), dst=gather_src_f32
    )
    gather_src_u32 = nl.ndarray((TK, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gather_src_u32, src=gather_src_f32)

    output_sb = nl.ndarray((TK, total_cols), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        src=packed_hbm.ap(
            pattern=[[total_cols, TK], [1, total_cols]],
            offset=0,
            vector_offset=gather_src_u32,
            indirect_dim=0,
        ),
        dst=output_sb,
    )
    output_hbm = nl.ndarray(
        (TK, total_cols), dtype=dtype, buffer=nl.shared_hbm, name="prt_a2av_output_hbm"
    )
    nisa.dma_copy(dst=output_hbm, src=output_sb)
    return output_hbm


@nki.jit
def _permute_one_token_nki(
    hidden_states: nl.NkiTensor,
    expert_affinities: nl.NkiTensor,
    expert_index: nl.NkiTensor,
    token_base_index: nl.NkiTensor,
):
    """Optimized single-token kernel for token permute; skips sorting and simply broadcasts T->K.

    Sharding: H is sharded on LNC; affinities on PNC0; token/expert index on PNC1.

    ``token_base_index`` is a [1] int32 runtime tensor = this rank's first token
    id (rank * T + 1; T == 1 here). It is broadcast across the K output rows and
    written directly, so the traced graph is identical across ranks (SPMD) — no
    ``dist.get_rank()`` constant baked into the kernel.
    """

    # Validation, constants
    T, H = hidden_states.shape
    _, E = expert_affinities.shape
    _, K = expert_index.shape
    pmax = nl.tile_size.pmax
    _, n_prgs, prg_id = get_verified_program_sharding_info("permute_one_token", (0, 1))
    H_local = H // n_prgs
    H_offset = H_local * prg_id
    H_inner = H_local // pmax
    # Spread the (1, E) affinity row across partitions via the largest divisor of E that is <= pmax.
    E_par = min(E, pmax)
    while E % E_par != 0:
        E_par -= 1
    E_inner = E // E_par

    kernel_assert(T == 1, f"T must be 1, got {T}")
    kernel_assert(
        hidden_states.dtype == nl.bfloat16,
        f"hidden_states must be bfloat16, got {hidden_states.dtype}",
    )
    kernel_assert(
        expert_index.dtype == nl.int32,
        f"expert_index must be int32, got {expert_index.dtype}",
    )
    kernel_assert(
        expert_affinities.dtype == nl.bfloat16,
        f"expert_affinities must be bfloat16, got {expert_affinities.dtype}",
    )
    kernel_assert(
        H_local % pmax == 0, f"H_local ({H_local}) must be divisible by pmax ({pmax})"
    )

    output_hbm = nl.ndarray(
        (K, H + E + 2 + K * 2), dtype=hidden_states.dtype, buffer=nl.shared_hbm
    )

    # HBM→HBM dma_copy with T=1 can't tile across partitions; stage through SBUF,
    # reshaping the single row to (pmax, H_inner) so the DMA engine can align.
    hidden_sb = nl.ndarray((pmax, H_inner), dtype=hidden_states.dtype, buffer=nl.sbuf)
    hidden_2d_view = hidden_states[0, nl.ds(H_offset, H_local)].reshape((pmax, H_inner))
    nisa.dma_copy(src=hidden_2d_view, dst=hidden_sb, dge_mode=dge_mode.none)

    for k_idx in range(K):
        dst_view = output_hbm[k_idx, nl.ds(H_offset, H_local)].reshape((pmax, H_inner))
        nisa.dma_copy(src=hidden_sb, dst=dst_view, dge_mode=dge_mode.none)

    if prg_id == 0:
        # Affinities are (1, E); stage through sbuf reshaped to (E_par, E_inner).
        aff_sb = nl.ndarray(
            (E_par, E_inner), dtype=expert_affinities.dtype, buffer=nl.sbuf
        )
        aff_2d_view = expert_affinities[0, :].reshape((E_par, E_inner))
        nisa.dma_copy(src=aff_2d_view, dst=aff_sb, dge_mode=dge_mode.none)
        for k_idx in range(K):
            dst_view = output_hbm[k_idx, H : H + E].reshape((E_par, E_inner))
            nisa.dma_copy(src=aff_sb, dst=dst_view, dge_mode=dge_mode.none)
    else:
        # Pack token_base_index (2 bf16 cols) + expert_index (K*2 bf16 cols);
        # both tiny (T=1, K=4), broadcast across K rows from SBUF.
        nisa.dma_copy(
            src=token_base_index.reshape((1, 1))
            .view(hidden_states.dtype)
            .broadcast(dim=0, size=K),
            dst=output_hbm[:, H + E : H + E + 2],
            dge_mode=dge_mode.none,
        )

        expert_idx_sb = nl.ndarray((1, K), dtype=expert_index.dtype, buffer=nl.sbuf)
        nisa.dma_copy(src=expert_index, dst=expert_idx_sb, dge_mode=dge_mode.none)
        for k_idx in range(K):
            nisa.dma_copy(
                src=expert_idx_sb.view(hidden_states.dtype),
                dst=output_hbm[k_idx, H + E + 2 :],
                dge_mode=dge_mode.none,
            )

    return output_hbm
