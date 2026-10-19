# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Tuple

import torch

import nki
import nki.isa as nisa
import nki.collectives as ncc
import nki.language as nl

from vllm.distributed.parallel_state import GroupCoordinator
from vllm_neuron import envs
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki


# TODO: Need to test and extend to max allowed RG size
_TP4_GROUP_SIZE = 4


def all_gather_v(
    input: torch.Tensor,
    output: torch.Tensor,
    group: GroupCoordinator,
    metadata: torch.Tensor,
    recv_counts_known: bool = False,
    has_rdispls: bool = False,
    priority: Optional[int] = None,
    cc_use_intermediate_io: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform a variable-length all-gather on the given replica group and input/output tensors.

    Unlike ``all_gather`` which concatenates along a collective dim, ``all_gather_v`` treats
    tensors as flat element buffers. Counts and displacements in the metadata tensor are in
    elements (row-major order), not slices along a particular dimension.

    The send side is fully per-destination: from this rank's ``input``, the chunk
    ``input[send_displs[r] : send_displs[r] + send_counts[r]]`` is sent to rank ``r``. With
    ``has_rdispls=False`` each sender's chunk lands in an equal-sized slot of ``output``:
    sender ``s``'s data is written at ``output[s * slot_elems : ...]`` where
    ``slot_elems = output.numel() / group_size``.

    Broadcast-style all-gather (the common case): set ``send_counts[r]`` and ``send_displs[r]``
    identically across all ``r`` so every destination receives the same chunk from this rank;
    ``output`` then holds the rank-ordered concatenation of all senders' chunks. Set
    ``send_counts[r] = 0`` to send nothing to rank ``r`` (its slot is left untouched).

    This API is a thin wrapper on top of the nki.collectives.all_gather_v instruction, which
    executes the collective.

    Args:
        input: Input tensor to gather from (this rank's contribution).
        output: Output tensor to store the gathered result. Sized for the worst case
            (up to ``group_size`` full slots) even when this rank sends fewer elements.
        group: GroupCoordinator for the ranks the collective is executed across. Must be TP4.
        metadata: Metadata tensor of shape (3, group_size), dtype uint32.
            Row 0: send counts (per destination rank), Row 1: send displacements,
            Row 2: recv counts (output when recv_counts_known=False; see flag).
            Row 3 (recv displacements) is omitted since has_rdispls=False is the only
            supported mode.
        recv_counts_known: Whether row 2 (recv_counts) is already known. When False (default)
            the collective writes per-rank received counts into row 2 during execution, which
            can be read back afterwards; when True row 2 is left untouched.
        has_rdispls: Not supported. Must be False. (Row 3 recv-displacements is not yet wired.)
        priority: DMA quality-of-service priority level 0-3 where lower is higher priority
            (NeuronCore-v4+ / Trn3+ only; leave None on Trn2). Passed through to the
            underlying instruction, which validates it against the target hardware.
        cc_use_intermediate_io: Whether all_gather_v should utilize intermediate buffers for
            collective I/O, which comes with a small performance penalty. Necessary in some
            cases, since NEFF I/O cannot also be collective I/O.

    Returns:
        output: Output tensor populated with data from the collective operation.
        metadata: Original metadata tensor, with recv_counts (row 2) optionally populated by
            the collective.

    Example:
        >>> # TP4: each rank broadcasts its own 4 elements to all 4 ranks.
        >>> # input shape: (4,), output shape: (16,), metadata shape: (3, 4)
        >>> metadata = torch.zeros(3, 4, dtype=torch.uint32)
        >>> metadata[0] = torch.tensor([4, 4, 4, 4])  # send counts (same chunk to every dst)
        >>> metadata[1] = torch.tensor([0, 0, 0, 0])  # send displs (broadcast → all 0)
        >>> output, metadata = all_gather_v(input, output, group, metadata)
        >>> # output[0:4] = rank 0's data, output[4:8] = rank 1's data, ...
    """

    # Resolve the FULL replica-group mesh EAGERLY (outside NKI/compile scope) as a
    # plain 2D int list — the SPMD-consistent partition the collective's
    # ReplicaGroup needs. ``group.ranks`` alone is only this rank's cell, which
    # would yield a per-rank single-cell ReplicaGroup (NOT SPMD).
    from libtorch_neuronx_lite.parallel.replica_mesh import resolve_full_mesh

    mesh = resolve_full_mesh(group)

    # Convert from GroupCoordinator -> tuple[int] (this rank's cell). Drives the
    # metadata-width / group-size validation; the ReplicaGroup uses ``mesh`` above.
    group = tuple(group.ranks)

    _validate_all_gather_v(group, metadata, has_rdispls)

    wrapped = wrap_nki(_all_gather_v_nki)

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
    )


def _validate_all_gather_v(group, metadata, has_rdispls):
    """
    Check if the collective inputs are valid.

    Constraints:
        - Not using CPU mode (NKI simulation does not support collectives yet)
        - has_rdispls must be False (not yet supported)
        - Metadata must have uint32 dtype
        - Metadata must have 3 rows (send_counts, send_displs, recv_counts)
        - Metadata must have the same number of columns as group size
        - Group size must be exactly 4 (TP4, intra-chip, LNC=2)

    ``priority`` is passed straight through to the underlying instruction, which
    validates it against the target hardware (NeuronCore-v4+ only).
    """

    assert not envs.VLLM_NEURON_CPU_MODE, (
        f"all_gather_v collective is not supported on CPU mode, got {envs.VLLM_NEURON_CPU_MODE=}"
    )
    assert not has_rdispls, (
        f"all_gather_v collective does not yet support has_rdispls=True, but got {has_rdispls=}"
    )
    assert metadata.dtype in (torch.uint32, "uint32"), (
        f"Expected metadata.dtype == torch.uint32, but got {metadata.dtype=}"
    )
    assert len(metadata.shape) == 2, f"Expected 2D metadata, but got {metadata.shape=}"
    assert metadata.shape[0] == 3, (
        f"Expected dim0 of metadata of size 3 (send_counts, send_displs, recv_counts) "
        f"since has_rdispls=False, but got {metadata.shape=}"
    )
    assert metadata.shape[1] == len(group), (
        f"Expected dim1 of metadata equal to process group size, but got {metadata.shape=}, {len(group)=}"
    )
    assert len(group) == _TP4_GROUP_SIZE, (
        f"all_gather_v collective is scoped to require replica group size: {_TP4_GROUP_SIZE} ranks "
        f"in the collective group, but got {len(group)=}"
    )


@nki.jit
def _all_gather_v_nki(
    input: nl.NkiTensor,
    output: nl.NkiTensor,
    group: Tuple[int],
    mesh: Tuple[Tuple[int]],
    metadata: nl.NkiTensor,
    recv_counts_known: bool = False,
    has_rdispls: bool = False,
    priority: Optional[int] = None,
    cc_use_intermediate_io: bool = False,
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """Thin wrapper of nki.collectives.all_gather_v, which executes a variable-length all-gather collective."""

    # FIXME: XLA traces uint32 -> int32; patch with bitcast to ensure correct type
    if metadata.dtype != nl.uint32:
        metadata = metadata.view(nl.uint32)

    # Build the ReplicaGroup from the FULL mesh (all cells) so the emitted replica
    # group is identical across ranks (SPMD consistent). Use an explicit loop —
    # NKI's tracer rejects list comprehensions ("unsupported expression").
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

    # Call collective
    ncc.all_gather_v(
        srcs=[cc_input],
        dsts=[cc_output],
        replica_group=replica_group,
        metadata_tensor=cc_metadata,
        recv_counts_known=recv_counts_known,
        has_rdispls=has_rdispls,
        priority=priority,
    )

    # When kernel I/O is NEFF I/O, copy back from intermediate buffers after calling collective
    if cc_use_intermediate_io:
        nisa.dma_copy(output, cc_output)
        nisa.dma_copy(metadata, cc_metadata)

    # Reset I/O to original shapes
    input = input.reshape(original_input_shape)
    output = output.reshape(original_output_shape)

    return output, metadata
