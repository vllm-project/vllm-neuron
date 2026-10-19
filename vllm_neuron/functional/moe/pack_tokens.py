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


def pack_tokens(input: torch.Tensor):
    """Compact rows with a nonzero token index to the front of the buffer.

    The final 2 cols of each [:, H+2] row are an int32 token index (bitcast to bf16):
    a nonzero index marks a valid row, a 0 index marks padding.

    Input [T, H+2]: valid rows scattered anywhere, padding rows interspersed.
    Output [T, H+2]: valid rows packed to the front in original order, followed by
        padding rows (0 index); the H hidden values in the padding tail are undefined.
    """
    _validate_inputs(input)

    if _can_use_kernel(input):
        wrapped = wrap_nki(_pack_tokens_nki)
        return wrapped[2](input)

    return _torch_impl(input)


def _validate_inputs(input):
    # The torch fallback traces to a data-dependent shape, so it's CPU-only; on
    # device the NKI kernel must be used.
    if not _can_use_kernel(input) and input.device.type != "cpu":
        raise ValueError(
            f"pack_tokens requires the NKI kernel or CPU input, got {input.device=}"
        )

    if input.dtype != torch.bfloat16:
        raise ValueError(f"input must be bfloat16, got {input.dtype}")


def _can_use_kernel(input):
    if not can_run_kernel(input):
        return False

    return True


def _torch_impl(input):
    """Reference: mask out zero-index rows and pack the rest to the front."""

    index = input[:, -2:].contiguous().view(torch.int32).flatten()
    nonzero_mask = index != 0
    count_nonzero = sum(nonzero_mask)
    output = torch.zeros_like(input)
    output[:count_nonzero, :] = input[nonzero_mask, :]

    return output


@nki.jit
def _pack_tokens_nki(input):
    """Pack rows of [T, H_concat] (final 2 cols = int32 index) with a nonzero index
    to the top of the output buffer.

    Tiles are split ping-pong between PNCs in LNC2.
    """

    # Allocations, shapes
    pmax = nl.tile_size.pmax
    T, H_concat = input.shape
    tile_T = min(T, pmax)
    num_T_tiles = div_ceil(T, tile_T)

    _, n_prgs, prg_id = get_verified_program_sharding_info("pack_tokens", (0, 1))
    if num_T_tiles == 1:
        # Nothing to shard — both NCs run the full kernel on the single tile.
        n_prgs, prg_id = 1, 0
    local_num_T_tiles = (
        div_ceil(num_T_tiles - prg_id, n_prgs) if num_T_tiles > prg_id else 0
    )

    # View input's final 2 bf16 columns as a contiguous [1, T] int32 strip,
    # loaded straight into partition 0 of indices_sb without any HBM copy.
    indices_view = (
        input.slice(dim=1, start=H_concat - 2, end=H_concat)
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
    output_permuted = nl.ndarray(input.shape, input.dtype, buffer=nl.shared_hbm)

    # Initialize the output index columns as 0.
    # FIXME: we can remove this memset after we turn on AG-v
    zero_idx = nl.ndarray((tile_T, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(zero_idx, 0)
    for tile_idx in range(num_T_tiles):
        nisa.dma_copy(
            output_permuted[nl.ds(tile_idx * tile_T, tile_T), nl.ds(H_concat - 2, 2)],
            zero_idx.view(input.dtype),
            dge_mode=dge_mode.none,
            name=f"pack_tokens_zero_idx_tile{tile_idx}",
        )

    dynamic = T > 256

    if not dynamic:
        for local_idx in range(local_num_T_tiles):
            global_tile_idx = prg_id + local_idx * n_prgs
            output_packed_sb = nl.ndarray(
                (tile_T, H_concat), input.dtype, buffer=nl.sbuf
            )
            # Zero so OOB-skipped lanes / reused SBUF slots start deterministic.
            nisa.memset(output_packed_sb, 0)
            nisa.dma_copy(
                src=input.ap(
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
    else:
        # Per-NC trip count from the global nonzero count keeps both NCs in LNC2
        # lockstep: num_local_tiles = ceil(num_nonzero / tile_T / n_prgs).
        # tile_T == 128 == 2^7; fold the per-NC halving and tile_T division into
        # one (count + bias) >> shift.
        per_nc_shift = 7
        if n_prgs > 1:
            kernel_assert(
                n_prgs == 2,
                f"dynamic branch only supports LNC in {{1,2}}, got {n_prgs=}",
            )
            per_nc_shift += 1
        bias = (1 << per_nc_shift) - 1

        num_tiles = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            data=nonzero_with_count_out[0, -1],
            op0=nl.add,
            operand0=bias,
            dst=num_tiles,
        )
        nisa.tensor_scalar(
            data=num_tiles,
            op0=nl.right_shift,
            operand0=per_nc_shift,
            dst=num_tiles,
        )
        reg = nisa.register_alloc()
        nisa.register_load(src=num_tiles, dst=reg)

        # Reshape (T, H_concat) -> (num_T_tiles // n_prgs, n_prgs, tile_T, H_concat)
        # so output_4d[loop_idx, prg_id] is the right global tile (scalar_offset=loop_idx
        # plus a compile-time prg_id offset).
        kernel_assert(
            num_T_tiles % n_prgs == 0,
            f"dynamic branch requires num_T_tiles divisible by n_prgs, got {num_T_tiles=} {n_prgs=}",
        )
        output_4d = output_permuted.reshape(
            (num_T_tiles // n_prgs, n_prgs, tile_T, H_concat)
        )
        store_compile_offset = prg_id * tile_T * H_concat

        def _dynamic_gather_body(loop_idx):
            output_packed_sb = nl.ndarray(
                (tile_T, H_concat), input.dtype, buffer=nl.sbuf
            )
            nisa.memset(output_packed_sb, 0)

            # Gather indices for this NC's loop_idx-th local tile: scalar_offset
            # picks column loop_idx of nonzero_indices_T (tile_T, local_num_T_tiles).
            indices = nl.ndarray((tile_T, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(
                src=nonzero_indices_T.ap(
                    pattern=[[local_num_T_tiles, tile_T], [1, 1]],
                    offset=0,
                    scalar_offset=loop_idx,
                    indirect_dim=1,
                ),
                dst=indices,
            )

            nisa.dma_copy(
                src=input.ap(
                    pattern=[[H_concat, tile_T], [1, H_concat]],
                    offset=0,
                    vector_offset=indices,
                    indirect_dim=0,
                ),
                dst=output_packed_sb,
                oob_mode=oob_mode.skip,
                name="pack_tokens_dyn_gather",
            )

            # Store at output_4d[loop_idx, prg_id, :, :].
            nisa.dma_copy(
                output_4d.ap(
                    pattern=[[H_concat, tile_T], [1, H_concat]],
                    offset=store_compile_offset,
                    scalar_offset=loop_idx,
                    indirect_dim=0,
                ),
                output_packed_sb,
                dge_mode=dge_mode.hwdge,
                name="pack_tokens_dyn_store",
            )

        nl.fori_loop(0, reg, _dynamic_gather_body)

    return output_permuted
