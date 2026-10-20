# SPDX-License-Identifier: Apache-2.0

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import dge_mode, oob_mode

from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import (
    div_ceil,
    get_verified_program_sharding_info,
)

import torch

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel
from .topk_reduce import _topk_reduce_nki as topk_reduce_kernel

_SUPPORTED_INPUT_DTYPES = [nl.bfloat16, nl.float16]


def hierarchical_all2all_combine_reduce(
    input: torch.Tensor,
    T_max: int,
    token_base_index: torch.Tensor,
    token_group_stride: int = 1,
    token_group_size: int = 1,
):
    """Reduce the intra-server combine output to one row per token, between the
    intra-server and inter-server all-to-all-v of hierarchical all2all.

    After the intra-server all-to-all-v, a rank holds up to T_max distinct tokens,
    each scattered across one or more rows (same token id repeated). This gathers
    rows by token id, sums them, and compacts the present tokens into the front of
    the output in ascending token-id order (padding/absent rows zero-filled). The
    result feeds the inter-server all-to-all-v.

    Only the token ids this rank owns are reduced. Output row t targets
    ``id = token_base_index + (t // G) * (token_group_stride * G) + (t % G)`` for
    t in [0, T_max), G = token_group_size: contiguous groups of G ids spaced
    token_group_stride apart. G=1 is the plain progression base + t * group_stride.

    Args:
        input: [N, H+2] bf16. Each row is [hidden | packed int32 token id]; padding
            rows carry token id -1.
        T_max: Max number of distinct token ids this rank can own (> 1).
        token_base_index: [1] int32 runtime tensor — the first (smallest) token
            id this rank owns, 1-indexed. Passed as a tensor (read from the
            stamped rank_in_group_tensor) so the traced graph is identical across
            ranks (SPMD).
        token_group_stride: Per-group stride; group-start distance is group_stride * G.
        token_group_size: Number of contiguous ids per group (G).

    Returns:
        [T_max, H+2] bf16: present tokens summed and packed to the front in
        ascending token-id order; trailing 2 cols hold the packed id (0 for padding).
    """

    _validate_inputs(input, T_max)

    if _can_use_kernel():
        wrapped = wrap_nki(_hierarchical_all2all_combine_reduce_nki)
        return wrapped[2](
            input, T_max, token_base_index, token_group_stride, token_group_size
        )
    else:
        return _torch_impl(
            input, T_max, token_base_index, token_group_stride, token_group_size
        )


def _validate_inputs(input, T_max):
    if not _can_use_kernel() and str(input.device) != "cpu":
        raise ValueError(
            "hierarchical_all2all_combine_reduce does not support execution on hardware without NKI kernel"
        )

    H = input.shape[1] - 2
    assert H % 1024 == 0 and T_max > 1 and input.dtype == torch.bfloat16, (
        f"Expected T_max>1, H divisible by 1024, and input.dtype = torch.bfloat16, got {T_max=} {H=} {input.dtype=}"
    )


def _can_use_kernel():
    if not can_run_kernel():
        return False

    return True


def _torch_impl(
    input,
    T_max,
    token_base_index: torch.Tensor,
    token_group_stride=1,
    token_group_size=1,
):
    """Torch impl of hierarchical_all2all_combine_reduce (CPU reference).

    Output row t targets id base + (t // G) * (group_stride * G) + (t % G), where
    G = token_group_size: contiguous groups of G ids spaced group_stride apart.
    G=1 is the plain progression base + t * group_stride.
    """
    _, H_concat = input.shape
    H = H_concat - 2
    indices = input.view(torch.int32)[:, -1]  # (N,) packed token indices

    G = token_group_size
    group_dist = token_group_stride * G
    # Owned ids in output-row (t) order, matching the kernel's search per slot.
    owned = [token_base_index + (t // G) * group_dist + (t % G) for t in range(T_max)]
    out = torch.zeros(T_max, H + 2, dtype=input.dtype, device=input.device)
    row = 0
    for tok_id in owned:
        match = indices == tok_id
        if bool(match.any()):
            out[row, :H] = input[match, :H].sum(dim=0)
            out[row, -2:] = torch.tensor(
                [tok_id], dtype=torch.int32, device=input.device
            ).view(input.dtype)
            row += 1
    return out


@nki.jit
def _hierarchical_all2all_combine_reduce_nki(
    input: nl.NkiTensor,
    T_max: int,
    token_base_index: nl.NkiTensor,
    token_group_stride: int = 1,
    token_group_size: int = 1,
):
    """NKI implementation of hierarchical all2all topk reduction. Supports dynamic T with grouped/strided/offset ids for EP+TP."""

    kernel_assert(
        input.dtype in _SUPPORTED_INPUT_DTYPES,
        f"input must be one of {_SUPPORTED_INPUT_DTYPES}, got {input.dtype}",
    )

    pmax = nl.tile_size.pmax

    # Reduced [T, H | index]
    reduced = topk_reduce_kernel(
        input=input,
        T=T_max,
        K=None,  # FIXME: remove placeholder; K is unused by the matmul topk-reduce kernel
        token_base_index=token_base_index,
        is_hierarchical=True,
        token_group_stride=token_group_stride,
        token_group_size=token_group_size,
    )

    # Sync point for SB->HBM
    # FIXME: persist topk_reduce output in SBUF, skip reloading before packing.
    nisa.core_barrier(reduced, cores=(0, 1))

    # Allocations, shapes
    pmax = nl.tile_size.pmax
    T, H_concat = reduced.shape
    tile_T = min(T, pmax)
    num_T_tiles = div_ceil(T, tile_T)

    _, n_prgs, prg_id = get_verified_program_sharding_info(
        "hierarchical_all2all_combine_reduce", (0, 1)
    )
    if num_T_tiles == 1:
        # Nothing to shard — both NCs run the full kernel on the single tile.
        n_prgs, prg_id = 1, 0
    local_num_T_tiles = (
        div_ceil(num_T_tiles - prg_id, n_prgs) if num_T_tiles > prg_id else 0
    )

    # View reduced's final 2 bf16 columns as a contiguous [1, T] int32 strip,
    # loaded straight into partition 0 of indices_sb without any HBM copy.
    indices_view = (
        reduced.slice(dim=1, start=H_concat - 2, end=H_concat)
        .view(nl.int32)
        .permute((1, 0))
    )
    indices_sb = nl.ndarray((pmax, T), dtype=nl.int32, buffer=nl.sbuf)

    # Find nonzero (replicated on each NC)
    nisa.dma_copy(
        indices_sb[0, :],
        indices_view,
        dge_mode=dge_mode.none,
        name="pack_tokens_load_indices",
    )
    nonzero_with_count_out = nl.ndarray((pmax, T + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nonzero_with_count(src=indices_sb, dst=nonzero_with_count_out, padding_val=-1)

    # Transpose indices for the local (ping-pong) tiles only
    nonzero_indices_T = nl.ndarray(
        (tile_T, local_num_T_tiles), dtype=nl.int32, buffer=nl.sbuf
    )
    nonzero_indices_T_psum = nl.ndarray(
        (tile_T, local_num_T_tiles), dtype=nl.float32, buffer=nl.psum
    )
    for local_idx in range(local_num_T_tiles):
        global_tile_idx = prg_id + local_idx * n_prgs
        nisa.nc_transpose(
            data=nonzero_with_count_out[
                0, nl.ds(global_tile_idx * tile_T, tile_T)
            ].view(nl.float32),
            dst=nonzero_indices_T_psum[:, local_idx : local_idx + 1],
        )
    nisa.tensor_copy(src=nonzero_indices_T_psum, dst=nonzero_indices_T.view(nl.float32))

    # Gather tokens, place back into output buffer (each NC handles its own tiles)
    output_permuted = nl.ndarray(reduced.shape, reduced.dtype, buffer=nl.shared_hbm)

    for local_idx in range(local_num_T_tiles):
        global_tile_idx = prg_id + local_idx * n_prgs
        output_packed_sb = nl.ndarray((tile_T, H_concat), reduced.dtype, buffer=nl.sbuf)
        # FIXME: probably not needed?
        nisa.memset(output_packed_sb, 0)
        nisa.dma_copy(
            src=reduced.ap(
                pattern=[[H_concat, tile_T], [1, H_concat]],
                offset=0,
                vector_offset=nonzero_indices_T[:, local_idx : local_idx + 1],
                indirect_dim=0,
            ),
            dst=output_packed_sb,
            oob_mode=oob_mode.skip,
            name=f"pack_tokens_gather_tile{global_tile_idx}",
        )

        nisa.dma_copy(
            output_permuted[nl.ds(global_tile_idx * tile_T, tile_T), :],
            output_packed_sb,
            dge_mode=dge_mode.none,
            name=f"pack_tokens_store_tile{global_tile_idx}",
        )

    return output_permuted
