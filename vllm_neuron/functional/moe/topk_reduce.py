# SPDX-License-Identifier: Apache-2.0
"""topk_reduce functional API."""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import dge_mode

from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import (
    div_ceil,
    get_verified_program_sharding_info,
)

import torch

from vllm.distributed.parallel_state import get_world_group

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel

_TILE_H = 512
_P_TILE_MAX = 128
_T_TILE_MAX = 128
_N_16BIT_ELEM_PER_INT32 = 2
_SUPPORTED_INPUT_DTYPES = [nl.bfloat16, nl.float16]


def topk_reduce(
    input: torch.Tensor,
    T: int,
    K: int,  # TODO: deprecate K arg (unused by the matmul kernel; kept for call-site compat).
    is_sequence_parallel: bool = False,
) -> torch.Tensor:
    """Compute sparse MoE Top-K reduction across all2all collective output buffer.

    Gathers scattered rows by packed global token index and reduces over the rows
    sharing that index. Each token has a dynamic number of rows scattered at
    arbitrary positions in the input, with remaining rows being padding.

    Token indices are 1-indexed. When is_sequence_parallel=False, indices are 1..T.
    When is_sequence_parallel=True, indices are global: rank r owns (r*T+1)..(r*T+T),
    and the offset rank_id*T is subtracted before indexing into the output.

    Padded rows must have index -1.

    Shapes: input [TK_padded, H+2] → output [T, H].
        Kernel constraints: TK_padded <= 128 or a multiple of 128, and H
        divisible by 1024. Otherwise falls back to the CPU reference.

    Args:
        input (torch.Tensor): [TK_padded, H + 2] bf16/fp16. Sparse buffer with
            the routed rows. Final 2 bf16 columns encode a packed int32
            global token index (1-indexed, -1 for padding).
        T (int): Number of output tokens per rank (1 to 128). When
            is_sequence_parallel=True, T represents the global number of tokens / world_size.
        is_sequence_parallel (bool): If True, token indices are expected to represent
            global token indices. Using local token indices with is_sequence_parallel=True,
            or using global token indices with is_sequence_parallel=False, will result in
            out of bounds at runtime. In SP mode the per-rank token-id offset
            (rank * T) is derived from the on-device
            ``get_world_group().rank_in_group_tensor`` (stamped by
            ``stamp_spmd_rank_tensors``) so the traced graph is identical across
            ranks (SPMD). Defaults to False.

    Returns:
        torch.Tensor: [T, H] bf16/fp16. out[t] = sum of all rows with index
            (rank_id*T + t + 1) in SP mode, or (t + 1) in non-SP mode.

    Example:
        >>> # Non-SP, dense: T=2, H=4, input shape [4, 6]
        >>> # indices [1, 1, 2, 2] → out[0] = row0 + row1, out[1] = row2 + row3
        >>> out = topk_reduce(input, T=2)  # shape [2, 4]

        >>> # Non-SP, sparse: T=2, H=4, input shape [8, 6]
        >>> # indices [1, -1, 1, 2, -1, -1, 2, -1] → out[0] = row0 + row2, out[1] = row3 + row6
        >>> out = topk_reduce(input, T=2)  # shape [2, 4]

        >>> # SP: rank 1, T=8, H=4, input shape [16, 6], world_size=4
        >>> # global indices [9, 9, 10, 10, ..., 16, 16] (rank 1 owns 9..16)
        >>> # offset = rank * T = 1 * 8 = 8, local indices after subtract: [1, 1, 2, 2, ..., 8, 8]
        >>> out = topk_reduce(input, T=8, is_sequence_parallel=True)  # shape [8, 4]
    """
    _validate_inputs(input, T, K)
    # SPMD: in SP mode the per-rank base id (rank * T + 1) is a runtime tensor
    # read from the stamped ``get_world_group().rank_in_group_tensor``, so the
    # traced graph is identical across ranks (one shared NEFF) — the kernel loads
    # it into SBUF and broadcast-adds it rather than baking ``dist.get_rank()``
    # into ``nisa.iota``'s offset. Non-SP is the rank-invariant constant 1.
    # In the non-SP branch the value is the rank-invariant constant 1.
    # Use ones() rather than torch.tensor([1], ...) so RankConstantCheckPass
    # (which flags any small-int singleton built via torch.tensor / torch.full)
    # doesn't fire on this genuinely-shared constant under SPMD strict.
    token_base_index = (
        get_world_group().rank_in_group_tensor * T + 1
        if is_sequence_parallel
        else torch.ones(1, dtype=torch.int32, device=input.device)
    )

    if _can_use_kernel(input):
        wrapped = wrap_nki(_topk_reduce_nki)
        return wrapped[2](input, T, K, token_base_index)
    else:
        return _cpu_topk_reduce(input, T, token_base_index)


def _validate_inputs(input: torch.Tensor, T: int, K: int) -> None:
    """Validate topk_reduce arguments."""
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T=}")

    # TODO: take away this check when torch impl works on HW
    TK_padded = input.shape[0]
    H = input.shape[1] - 2
    if not _can_use_kernel(input) and str(input.device) != "cpu":
        raise ValueError(
            f"Expected input.shape[0] in [1, {_P_TILE_MAX}] or a multiple of {_P_TILE_MAX}, "
            f"and H divisible by 1024 for topk_reduce execution on hardware, got {TK_padded=} {H=}"
        )


def _can_use_kernel(input: torch.Tensor) -> bool:
    """Check if the NKI topk_reduce kernel can be used.

    Kernel constraints:
        - Device must support NKI kernels (can_run_kernel)
        - input.shape[0] (TK_padded) in [1, 128] or a multiple of 128
        - H (input.shape[1] - 2) must be divisible by 1024
    """
    if not can_run_kernel(input):
        return False

    H = input.shape[1] - 2
    TK_padded = input.shape[0]

    if not (1 <= TK_padded <= _P_TILE_MAX or TK_padded % _P_TILE_MAX == 0):
        return False
    if H % 1024 != 0:
        return False

    return True


def _cpu_topk_reduce(
    input: torch.Tensor, T: int, token_base_index: torch.Tensor
) -> torch.Tensor:
    """CPU-only reference implementation using scatter_add."""

    H = input.shape[1] - 2
    indices = input.view(torch.int32)[:, -1]  # (N,) packed token indices

    # Map global 1-indexed token indices to local 1-indexed
    indices = indices - (token_base_index - 1)
    # Shift indices: valid tokens are 1..T → map to rows 1..T; <=0 → maps to 0
    # We accumulate into row 0 as a garbage bin, then discard it.
    out = torch.zeros(T + 1, H, dtype=input.dtype)
    bucket = indices.clamp(min=0).long()
    out.scatter_add_(0, bucket.unsqueeze(1).expand(-1, H), input[:, :H])
    return out[1:]  # discard row 0 (garbage from -1 and 0 indices)


# TODO: upstream into nkilib when kernel is finalized/stable
@nki.jit
def _topk_reduce_nki(
    input: nl.NkiTensor,
    T: int,
    K: int,  # TODO: deprecate K arg (retained for the public topk_reduce() signature).
    token_base_index: nl.NkiTensor,
    is_hierarchical: bool = False,
    token_group_stride: int = 1,
    token_group_size: int = 1,
):
    """Sparse reduce as a one-hot matmul, tiled over input rows and output T.

    Builds a (TK_padded, T) one-hot reduce matrix from the packed token-id
    column and computes `ones.T @ input[:, :H]`. Output row t searches for token
    id ``base + (t // G) * (group_stride * G) + (t % G)`` (G = token_group_size):
    contiguous groups of G ids, with successive groups spaced group_stride apart.
    G=1 reduces to the plain arithmetic progression ``base + t * group_stride``.

    Args:
        input: (TK_padded, H + 2) bf16/fp16 combine buffer; last 2 cols hold the packed int32 token id.
        T: number of output token slots.
        token_base_index: token id searched for by output row 0.
        is_hierarchical: when True, pack routed token ids into the trailing 2 bf16 cols (see above).
        token_group_stride: per-group stride; group-start distance is group_stride * G.
        token_group_size: number of contiguous ids per group (G).

    Example:
        base=1, G=2, group_stride=64: distance 64*2=128, ids 1, 2, 129, 130,
        257, 258, ... (2-token groups, one per rank in a 64-rank server exchange).
    """
    kernel_assert(
        token_group_size <= _T_TILE_MAX
        and _T_TILE_MAX % token_group_size == 0
        and T % token_group_size == 0,
        f"token_group_size ({token_group_size}) must divide {_T_TILE_MAX} and T ({T})",
    )

    TK_padded, H_concat = input.shape
    H = H_concat - _N_16BIT_ELEM_PER_INT32
    kernel_assert(
        input.dtype in _SUPPORTED_INPUT_DTYPES,
        f"input must be one of {_SUPPORTED_INPUT_DTYPES}, got {input.dtype}",
    )
    # Input rows are P-tiled by 128, so callers must pad them to a 128-row multiple.
    kernel_assert(
        TK_padded <= _P_TILE_MAX or TK_padded % _P_TILE_MAX == 0,
        f"TK_padded must be <= {_P_TILE_MAX} or a multiple of {_P_TILE_MAX} "
        f"(pad the input rows to a {_P_TILE_MAX} multiple), got {TK_padded=}",
    )
    P_TILE = min(_P_TILE_MAX, TK_padded)

    H_shard = H // 2
    kernel_assert(
        H_shard % _TILE_H == 0, f"H_shard ({H_shard}) must be a multiple of {_TILE_H}"
    )

    _, n_prgs, prg_id = get_verified_program_sharding_info("topk_reduce_v2", (0, 1))
    is_last_core = prg_id == n_prgs - 1
    pack_indices = is_hierarchical and is_last_core
    psum_dtype = (
        input.dtype if nisa.get_nc_version() >= nisa.nc_version.gen4 else nl.float32
    )

    num_p_tiles = div_ceil(TK_padded, P_TILE)
    num_h_tiles = div_ceil(H_shard, _TILE_H)
    num_t_tiles = div_ceil(T, _T_TILE_MAX)

    output_shape = (T, H_concat) if is_hierarchical else (T, H)
    output_hbm = nl.ndarray(output_shape, input.dtype, buffer=nl.shared_hbm)

    # Load the packed token id per P-tile once, hoisted above the T-loop.
    index_sb_list = []
    for p in range(num_p_tiles):
        p_slice = nl.ds(p * P_TILE, P_TILE)
        index_sb = nl.ndarray((P_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            index_sb,
            input[p_slice, -2:].view(nl.int32),
            dge_mode=dge_mode.none,
            name=f"topk_reduce_v2_load_idx_p{p}",
        )
        index_sb_list.append(index_sb)

    # Constant all-zeros tile for the non-finite scrub below; write-once, read-only.
    zeros_sb = nl.ndarray((P_TILE, _TILE_H), input.dtype, buffer=nl.sbuf)
    nisa.memset(zeros_sb, 0.0)

    # Group-start distance: successive id groups are group_stride*G apart.
    group_dist = token_group_stride * token_group_size

    # Load the base token id, broadcast across partitions for the per-tile offset add.
    token_base_index_sb = nl.ndarray((P_TILE, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        token_base_index_sb,
        token_base_index.reshape((1, 1)).broadcast(dim=0, size=P_TILE),
    )

    for t_tile_idx in range(num_t_tiles):
        t_offset = t_tile_idx * _T_TILE_MAX
        tile_T = min(_T_TILE_MAX, T - t_offset)
        t_slice = nl.ds(t_offset, tile_T)

        # Search ids for this T-tile, replicated across partitions; built once (independent of p) and reused below.
        search_iota = nl.ndarray((P_TILE, tile_T), dtype=nl.float32, buffer=nl.sbuf)
        if token_group_size > 1:
            # base=1,G=2,group_dist=128 -> [1, 2, 129, 130, 257, 258, ...]
            nisa.iota(
                search_iota,
                [[group_dist, tile_T // token_group_size], [1, token_group_size]],
                offset=0,
            )
        else:
            # base=1,stride=4 -> [1, 5, 9, 13, ...]
            nisa.iota(
                search_iota,
                [[token_group_stride, tile_T], [1, 1]],
                offset=0,
            )

        # Shift the zero-based iota by this tile's base id (base + tile's group offset).
        tile_token_base_index = nl.ndarray(
            (P_TILE, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            data=token_base_index_sb,
            op0=nl.add,
            operand0=(t_offset // token_group_size) * group_dist,
            dst=tile_token_base_index,
        )
        nisa.tensor_scalar(
            data=search_iota,
            op0=nl.add,
            operand0=tile_token_base_index,
            dst=search_iota,
        )

        # Build this T-tile's one-hot reduce matrix per P-tile.
        ones_sb_list = []
        for p in range(num_p_tiles):
            ones_sb = nl.ndarray((P_TILE, tile_T), input.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(
                data=search_iota, op0=nl.equal, operand0=index_sb_list[p], dst=ones_sb
            )
            ones_sb_list.append(ones_sb)

        # H-tile pipeline: load(N+1) // matmul(N) // spill(N-1) overlap via per-tile sbuf alloc.
        for tile_idx in range(num_h_tiles):
            tile_h_offset = tile_idx * _TILE_H
            hbm_tile_slice = nl.ds(prg_id * H_shard + tile_h_offset, _TILE_H)

            output_tile_sb = nl.ndarray((tile_T, _TILE_H), input.dtype, buffer=nl.sbuf)
            reduced_psum = nl.ndarray(
                (tile_T, _TILE_H), dtype=psum_dtype, buffer=nl.psum
            )

            # Accumulate (P_TILE, tile_T)^T @ (P_TILE, _TILE_H) into the same psum across P-tiles.
            for p in range(num_p_tiles):
                p_slice = nl.ds(p * P_TILE, P_TILE)
                input_partial_sb = nl.ndarray(
                    (P_TILE, _TILE_H), input.dtype, buffer=nl.sbuf
                )
                nisa.dma_copy(
                    input_partial_sb,
                    input[p_slice, hbm_tile_slice],
                    dge_mode=dge_mode.none,
                    name=f"topk_reduce_v2_load_t{t_tile_idx}_tile{tile_idx}_p{p}",
                )
                # Zero non-finite values before the matmul: it multiplies every row
                # (incl. unrouted rows, not guaranteed finite) by a 0/1 weight, and
                # 0 * inf = 0 * nan = nan poisons the whole column-sum. Detect inf and
                # nan dtype-agnostically: (x - x) != (x - x) is true only for non-finite
                # x (clamp won't work: fp32 max isn't representable in bf16/fp16).
                diff_sb = nl.ndarray((P_TILE, _TILE_H), input.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    data1=input_partial_sb,
                    data2=input_partial_sb,
                    op=nl.subtract,
                    dst=diff_sb,
                )
                is_nonfinite_sb = nl.ndarray(
                    (P_TILE, _TILE_H), dtype=nl.uint8, buffer=nl.sbuf
                )
                nisa.tensor_tensor(
                    data1=diff_sb,
                    data2=diff_sb,
                    op=nl.not_equal,
                    dst=is_nonfinite_sb,
                )
                nisa.tensor_copy_predicated(
                    src=zeros_sb,
                    predicate=is_nonfinite_sb,
                    dst=input_partial_sb,
                )
                nisa.nc_matmul(
                    stationary=ones_sb_list[p],
                    moving=input_partial_sb,
                    is_stationary_onezero=True,
                    dst=reduced_psum,
                )

            nisa.tensor_copy(src=reduced_psum, dst=output_tile_sb)
            nisa.dma_copy(
                output_hbm[t_slice, hbm_tile_slice],
                output_tile_sb,
                dge_mode=dge_mode.none,
                name=f"topk_reduce_v2_spill_t{t_tile_idx}_tile{tile_idx}",
            )

        if pack_indices:
            # Reuse the search ids: transpose search_iota row 0 to the partition axis, casting fp32 -> int32 on copy-out.
            arange_T_psum = nl.ndarray((tile_T, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(data=search_iota[0:1, :], dst=arange_T_psum)
            arange_T = nl.ndarray((tile_T, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(src=arange_T_psum, dst=arange_T)

            # Routed-slot detection: column-sum ones_sb across P-tiles via a ones-moving matmul; sum > 0 means routed.
            ones_vec = nl.ndarray((P_TILE, 1), input.dtype, buffer=nl.sbuf)
            nisa.memset(ones_vec, 1.0)
            routed_count_psum = nl.ndarray(
                (tile_T, 1), dtype=psum_dtype, buffer=nl.psum
            )
            for p in range(num_p_tiles):
                nisa.nc_matmul(
                    stationary=ones_sb_list[p],
                    moving=ones_vec,
                    is_stationary_onezero=True,
                    dst=routed_count_psum,
                )

            routed = nl.ndarray((tile_T, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                data=routed_count_psum,
                op0=nl.greater,
                operand0=0,
                dst=routed,
            )

            routed_indices_sb = nl.ndarray((tile_T, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.memset(routed_indices_sb, 0)
            nisa.tensor_copy_predicated(
                src=arange_T,
                predicate=routed,
                dst=routed_indices_sb,
            )

            nisa.dma_copy(
                output_hbm[t_slice, H:].view(nl.int32),
                routed_indices_sb,
                dge_mode=dge_mode.none,
                name=f"topk_reduce_v2_pack_indices_t{t_tile_idx}",
            )

    return output_hbm
