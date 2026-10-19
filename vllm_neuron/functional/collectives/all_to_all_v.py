# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Tuple

import torch

import nki
import nki.isa as nisa
from nki.isa import dge_mode
import nki.collectives as ncc
import nki.language as nl

from vllm.distributed.parallel_state import GroupCoordinator, get_world_group
from vllm_neuron import envs
from vllm_neuron.parallel.neuron_parallel_state import (
    is_internode_group,
    get_node_group,
    try_get_rank_in_group_tensor,
)
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki


def all_to_all_v(
    input: torch.Tensor,
    output: torch.Tensor,
    group: GroupCoordinator,
    metadata: torch.Tensor,
    recv_counts_known: bool = False,
    has_rdispls: bool = False,
    priority: Optional[int] = None,
    cc_use_intermediate_io: bool = False,
) -> torch.Tensor:
    """Perform a variable-length all-to-all on the given replica group and input/output tensors.

    Unlike all_to_all which splits and concatenates along a collective_dim, all_to_all_v treats tensors as flat buffers of elements.
    Counts and displacements in the metadata tensor are in elements (row-major order), not slices along a particular dimension.

    This API is a thin wrapper on top of the nki.collectives.all_to_all_v instruction, which executes the collective.

    Args:
        input: Input tensor to redistribute.
        output: Output tensor to store result.
        group: GroupCoordinator for ranks that the collective will be executed across.
        metadata: Metadata tensor of shape (4, world_size), dtype uint32.
            Row 0: send counts, Row 1: send displacements,
            Row 2: recv counts (can be empty), Row 3: recv displacements (can be empty).
        recv_counts_known: Whether the collective should populate row 2 of metadata with recv_counts.
        has_rdispls: Not yet supported. Whether row 3 of metadata contains real recv_displs.
        priority: Not yet supported. DMA quality-of-service priority level 0-3 where lower is higher (Trn3+ only).
        cc_use_intermediate_io: Whether all_to_all_v should utilize intermediate buffers for collective I/O,
            which comes with a small performance penalty. Necessary in some cases, since NEFF I/O cannot also be collective I/O.

    Returns:
        output: Output tensor populated with data from the collective operation.
        metadata: Original metadata tensor, with recv_counts (row 2) optionally populated by the collective.

    Example:
        >>> # world_size=8, each rank sends 4 elements to every other rank
        >>> # input shape: (32,), output shape: (32,), metadata shape: (4, 8)
        >>> metadata = torch.zeros(4, 8, dtype=torch.uint32)
        >>> metadata[0] = torch.tensor([4, 4, 4, 4, 4, 4, 4, 4])  # send counts
        >>> metadata[1] = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28])  # send displs
        >>> output, metadata = all_to_all_v(input, output, group, metadata)
        >>> # output[0:4] = data from rank 0, output[4:8] = data from rank 1, ...
    """

    # Resolve the FULL replica-group mesh EAGERLY (outside NKI/compile scope) as a
    # plain 2D int list — the SPMD-consistent partition the collective's
    # ReplicaGroup needs. ``group.ranks`` alone is only this rank's cell, which
    # would yield a per-rank single-cell ReplicaGroup (NOT SPMD).
    from libtorch_neuronx_lite.parallel.replica_mesh import resolve_full_mesh

    mesh = resolve_full_mesh(group)

    # Convert from GroupCoordinator -> tuple[int] (this rank's cell). Drives the
    # metadata-width / group-size validation and the internode-group check, which
    # are all per-cell; the ReplicaGroup uses the full ``mesh`` above.
    group = tuple(group.ranks)

    _validate_all_to_all_v(group, metadata, priority)

    wrapped = wrap_nki(_all_to_all_v_nki)

    do_metadata_a2a = not envs.VLLM_NEURON_SWITCH_CC and not is_internode_group(group)

    # Full SPMD node-group mesh for the recv-counts A2A (resolved eagerly, like
    # ``mesh`` above). On our (pre-#2869) tree the metadata A2A previously built
    # its ReplicaGroup from this rank's single node-group cell
    # (``tuple(get_node_group().ranks)``), which is per-rank (NOT SPMD). Resolve
    # the FULL node-group mesh instead so the emitted ReplicaGroup is identical
    # across ranks (one shared NEFF).
    metadata_mesh = resolve_full_mesh(get_node_group()) if do_metadata_a2a else None

    # Stride between this rank's metadata slots in the unstrided server-group
    # buffer. Ranks-per-node under the 1rpd A2A-v layout. Matches #3129, which
    # hard-codes 4 (see NOTE below).
    # NOTE: #3129 hard-codes this stride as the literal ``4`` (and the modulo
    # ``rank % 4``). ``get_node_group().world_size`` is the semantically-correct
    # ranks-per-node value; we keep the literal 4 to port #3129 verbatim.
    metadata_a2a_stride = 4 if do_metadata_a2a else None

    # SPMD: derive this rank's per-node slot index from the runtime
    # ``rank_in_group_tensor`` (stamped by stamp_spmd_rank_tensors) so the traced
    # graph is rank-invariant. When the world group has NOT been stamped (direct
    # non-SPMD callers that skip stamp_spmd_rank_tensors), fall back to a tensor
    # built from the eager, trace-safe ``rank_in_group`` int (numerically
    # identical to the legacy ``dist.get_rank() % 4``, but never non-concrete
    # under fullgraph tracing — unlike ``dist.get_rank()`` itself).
    if do_metadata_a2a:
        world_group = get_world_group()
        rank_in_group_tensor = try_get_rank_in_group_tensor(world_group)
        if rank_in_group_tensor is not None:
            rank_in_group_id = rank_in_group_tensor % metadata_a2a_stride
        else:
            rank_in_group_id = torch.tensor(
                [world_group.rank_in_group % metadata_a2a_stride],
                dtype=torch.int32,
                device=metadata.device,
            )
    else:
        rank_in_group_id = None

    return wrapped[2](
        input=input,
        output=output,
        group=group,
        mesh=tuple(tuple(cell) for cell in mesh),
        metadata=metadata,
        recv_counts_known=recv_counts_known,
        has_rdispls=has_rdispls,
        priority=priority,
        cc_use_intermediate_io=cc_use_intermediate_io,
        # FIXME: remove the below when 1rpd recv counts propagation is fixed
        rank_in_group_id=rank_in_group_id,
        metadata_a2a_stride=metadata_a2a_stride,
        metadata_mesh=tuple(tuple(cell) for cell in metadata_mesh)
        if metadata_mesh is not None
        else None,
    )


def _validate_all_to_all_v(group, metadata, priority):
    """
    Check if the collective inputs are valid.

    Constraints:
        - Not using CPU mode (NKI simulation does not support collectives yet)
        - Priority must be None (not yet supported)
        - Metadata must have uint32 dtype
        - Metadata must have 4 rows
        - Metadata must have the same number of columns as group.size
    """

    assert not envs.VLLM_NEURON_CPU_MODE, (
        f"all_to_all_v collective is not supported on CPU mode, got {envs.VLLM_NEURON_CPU_MODE=}"
    )
    assert priority is None, (
        f"all_to_all_v collective does not yet support priority != None, but got {priority=}"
    )
    assert metadata.dtype in (torch.uint32, "uint32"), (
        f"Expected metadata.dtype == torch.uint32, but got {metadata.dtype=}"
    )
    assert len(metadata.shape) == 2, f"Expected 2D metadata, but got {metadata.shape=}"
    assert metadata.shape[0] == 4, (
        f"Expected dim0 of metadata of size 4, but got {metadata.shape=}"
    )
    assert metadata.shape[1] == len(group), (
        f"Expected dim1 of metadata equal to process group size, but got {metadata.shape=}, {len(group)=}"
    )


@nki.jit
def _all_to_all_v_nki(
    input: nl.NkiTensor,
    output: nl.NkiTensor,
    group: Tuple[int],
    mesh: Tuple[Tuple[int]],
    metadata: nl.NkiTensor,
    recv_counts_known: bool = False,
    has_rdispls: bool = False,
    priority: Optional[int] = None,
    cc_use_intermediate_io: bool = False,
    rank_in_group_id: Optional[nl.NkiTensor] = None,
    metadata_a2a_stride: int | None = None,
    metadata_mesh: Tuple[Tuple[int]] | None = None,
) -> nl.NkiTensor:
    """Thin wrapper of nki.collectives.all_to_all_v, which executes a variable-length all-to-all collective."""

    # FIXME: XLA traces uint32 -> int32; patch with bitcast to ensure correct type
    if metadata.dtype != nl.uint32:
        metadata = metadata.view(nl.uint32)

    # Build the ReplicaGroup from the FULL mesh (all cells) so the emitted
    # replica group is identical across ranks (SPMD consistent). Use an explicit
    # loop — NKI's tracer rejects list comprehensions ("unsupported expression").
    cells = []
    for cell in mesh:
        cells.append(list(cell))
    replica_group = ncc.ReplicaGroup(cells)

    # Legalize 1D input tensors to 2D
    original_input_shape = input.shape
    original_output_shape = output.shape
    new_input_shape = input.shape if len(input.shape) > 1 else (input.shape[0], 1)
    new_output_shape = output.shape if len(output.shape) > 1 else (output.shape[0], 1)
    input = input.reshape(new_input_shape)
    output = output.reshape(new_output_shape)

    # NEFF I/O cannot be collective I/O; when kernel I/O is NEFF I/O, copy to intermediate buffers before calling collective
    if cc_use_intermediate_io:
        cc_input = nl.ndarray(input.shape, input.dtype, buffer=nl.shared_hbm)
        cc_output = nl.ndarray(output.shape, output.dtype, buffer=nl.shared_hbm)
        cc_metadata = nl.ndarray(metadata.shape, metadata.dtype, buffer=nl.shared_hbm)

        nisa.dma_copy(cc_input, input)
        nisa.dma_copy(cc_output, output)
        nisa.dma_copy(cc_metadata, metadata)
    else:
        cc_input = input
        cc_output = output
        cc_metadata = metadata

    # Zero the destination before the collective: with has_rdispls=False,
    # all_to_all_v writes only the received rows and leaves the tail untouched,
    # so stale rows from a reused buffer can leak into the downstream reduce.
    # memset an SBUF tile then DMA it to the HBM dst. SBUF partition dim <= 128,
    # so tile the row (partition) dim in pmax-sized chunks: cc_output rows scale
    # as num_tokens * ep_group_size and routinely exceed 128 for real batches.
    if nisa.get_nc_version() >= nisa.nc_version.gen4:
        zrows, zcols = cc_output.shape
        P = nl.tile_size.pmax
        zero_sb = nl.ndarray(
            (min(P, zrows), zcols), dtype=cc_output.dtype, buffer=nl.sbuf
        )
        nisa.memset(zero_sb, 0)
        for row0 in range(0, zrows, P):
            rows = min(P, zrows - row0)
            nisa.dma_copy(cc_output[row0 : row0 + rows, :], zero_sb[:rows, :])

    # Call collective
    ncc.all_to_all_v(
        srcs=[cc_input],
        dsts=[cc_output],
        replica_group=replica_group,
        metadata_tensor=cc_metadata,
        recv_counts_known=recv_counts_known,
        has_rdispls=has_rdispls,
        priority=priority,
    )

    # FIXME: recv_counts propagation is broken on Trn2 intra-server; use a static all2all to propagate recv counts
    if (
        rank_in_group_id is not None
        and metadata_a2a_stride is not None
        and metadata_mesh is not None
        and not recv_counts_known
    ):
        # A2A is not supported on strided RG, copy into unstrided RG and do an A2A on the entire server-group
        strided_group_len = cc_metadata.shape[1] * metadata_a2a_stride
        tmp_src_md = nl.ndarray(
            (1, strided_group_len), dtype=nl.uint32, buffer=nl.shared_hbm
        )
        zero_sb = nl.ndarray((1, strided_group_len), dtype=nl.uint32, buffer=nl.sbuf)
        rank_in_group_id_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(zero_sb, 0, engine=nisa.gpsimd_engine)
        nisa.dma_copy(tmp_src_md, zero_sb)
        nisa.dma_copy(rank_in_group_id_sb, rank_in_group_id)
        nisa.dma_copy(
            src=cc_metadata[0, :],
            dst=tmp_src_md.ap(
                pattern=[
                    [strided_group_len, 1],
                    [metadata_a2a_stride, strided_group_len // metadata_a2a_stride],
                    [1, 1],
                ],
                offset=0,
                scalar_offset=rank_in_group_id_sb,
                indirect_dim=1,
            ),
            dge_mode=dge_mode.hwdge,
        )
        # Build the node-group ReplicaGroup from its full mesh; explicit loop (NKI tracer rejects comprehensions).
        md_cells = []
        for cell in metadata_mesh:
            md_cells.append(list(cell))
        a2a_rg = ncc.ReplicaGroup(md_cells)

        # Propagate recv counts with server-group A2A
        tmp_dst_md = nl.ndarray(
            (1, strided_group_len), dtype=nl.uint32, buffer=nl.shared_hbm
        )
        ncc.all_to_all(
            srcs=[tmp_src_md],
            dsts=[tmp_dst_md],
            replica_group=a2a_rg,
            collective_dim=0,
            priority=priority,
        )

        # Copy with unstride back into metadata
        nisa.dma_copy(
            src=tmp_dst_md.ap(
                pattern=[
                    [strided_group_len, 1],
                    [metadata_a2a_stride, strided_group_len // metadata_a2a_stride],
                    [1, 1],
                ],
                offset=0,
                scalar_offset=rank_in_group_id_sb,
                indirect_dim=1,
            ),
            dst=cc_metadata[
                2:3, :
            ],  # NOTE: needs to remain 2D access to avoid compilation error when lowering DmaCopy
            dge_mode=dge_mode.hwdge,
        )

    # When kernel I/O is NEFF I/O, copy back from intermediate buffers after calling collective
    if cc_use_intermediate_io:
        nisa.dma_copy(output, cc_output)
        nisa.dma_copy(metadata, cc_metadata)

    # Reset I/O to original shapes
    input = input.reshape(original_input_shape)
    output = output.reshape(original_output_shape)

    return output, metadata
