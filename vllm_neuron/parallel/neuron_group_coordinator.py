# SPDX-License-Identifier: Apache-2.0
"""GroupCoordinator subclass that declares an explicit replica-group mesh.

vLLM's :class:`~vllm.distributed.parallel_state.GroupCoordinator` creates one
``torch.distributed.new_group`` per cell of the partition it is given, but does
not attach the full partition to those process groups. XLA collectives lowered
with ``use_global_device_ids=True`` need that full partition (the "mesh"), and
it must be identical across ranks to be SPMD compliant.

``GroupCoordinator`` lives in vLLM (an external dependency we cannot modify), so
we subclass it and declare the mesh via the :func:`active_mesh` context around
``super().__init__()``. The mesh-registry hook on ``dist.new_group`` then records
the explicit mesh for every cell's process group — no need to thread
``pg_options`` through the (unmodifiable) base constructor.

Meshes are always explicit: the mesh defaults to the ``group_ranks`` partition
passed in (which *is* the full mesh at every Neuron call site) and can be
overridden with the ``mesh`` argument.

The mesh registry and :func:`active_mesh` context manager live in
``libtorch_neuronx_lite.parallel.replica_mesh`` (the compile/lowering side), so
both the FX pass / XLA lowering and this coordinator agree on the same registry.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import List, Optional, Union

from torch.distributed import Backend
from vllm.distributed.parallel_state import GroupCoordinator


class NeuronGroupCoordinator(GroupCoordinator):
    """``GroupCoordinator`` that registers its full replica-group mesh.

    Behaves exactly like the base class, except the full partition is declared
    (via :func:`active_mesh`) while the underlying process groups are created, so
    each cell's PG is registered in the mesh registry.

    Args:
        group_ranks: The complete partition of the world into replica groups.
        local_rank: Local device rank (passed through to the base class).
        torch_distributed_backend: Distributed backend (passed through).
        use_device_communicator: Whether to build a device communicator.
        use_message_queue_broadcaster: Passed through to the base class.
        group_name: Optional group name (passed through).
        use_all2all: Whether to build the all2all device communicator
            variant (passed through to the base class).
        mesh: The full replica-group mesh to declare. Defaults to
            ``group_ranks`` — the partition is itself the mesh at every Neuron
            call site. Override only when the mesh differs from ``group_ranks``.

    Example:
        >>> coord = NeuronGroupCoordinator(
        ...     group_ranks=[[0, 1], [2, 3]], local_rank=0,
        ...     torch_distributed_backend="gloo", use_device_communicator=True,
        ... )  # declares mesh [[0, 1], [2, 3]] for both cells' PGs
    """

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        use_device_communicator: bool,
        use_message_queue_broadcaster: bool = False,
        group_name: Optional[str] = None,
        use_all2all: bool = False,
        mesh: Optional[List[List[int]]] = None,
    ):
        # Imported lazily, not at module scope: libtorch_neuronx_lite.parallel
        # ships only in the Neuron-variant wheel, and the CPU-variant wheel used
        # by the build-time test image has no such submodule.
        #
        # Since this PR routes vLLM's own group creation (_WORLD/_TP/_DP/... via
        # init_world_group / init_model_parallel_group) through this class,
        # *every* group is a NeuronGroupCoordinator -- including in MPExecutor
        # tests on the CPU image, which construct groups without libtorch. So a
        # missing module must degrade to plain GroupCoordinator behaviour rather
        # than raise: the mesh registry only feeds SPMD XLA collective lowering,
        # which needs libtorch anyway, so on a CPU host there is nothing to
        # declare a mesh for. When the module is present (every Neuron path) the
        # mesh is declared exactly as before.
        try:
            from libtorch_neuronx_lite.parallel.replica_mesh import (
                active_mesh,
                register_mesh,
            )
        except ImportError:
            active_mesh = None
            register_mesh = None

        mesh = group_ranks if mesh is None else mesh
        mesh_ctx = active_mesh(mesh) if active_mesh is not None else nullcontext()
        with mesh_ctx:
            super().__init__(
                group_ranks=group_ranks,
                local_rank=local_rank,
                torch_distributed_backend=torch_distributed_backend,
                use_device_communicator=use_device_communicator,
                use_message_queue_broadcaster=use_message_queue_broadcaster,
                group_name=group_name,
                use_all2all=use_all2all,
            )

        # Defense-in-depth: register the mesh explicitly after construction.
        #
        # The ``active_mesh`` context above only populates the registry if the
        # ``dist.new_group`` hook installed by ``apply_mesh_registry_patch`` is
        # still active when the base constructor runs. ``torch_xla`` unconditionally
        # rebinds ``dist.new_group`` to its own ``new_xla_process_group`` at import
        # (``xla_backend.py``: ``dist.new_group = new_xla_process_group``), which
        # drops our wrapper. If that import lands *after* the last
        # ``apply_mesh_registry_patch`` call — e.g. lazily, via the first
        # collective — every group built afterwards is created through the
        # un-wrapped ``new_group`` and registers nothing, leaving
        # ``get_all_replica_groups`` to fall back to this rank's single cell
        # (non-SPMD, per-rank cache hash).
        #
        # Registering here, keyed by the process group PyTorch actually assigned
        # (``self.device_group.group_name`` — the same key the ``_c10d_functional``
        # collectives resolve to at lowering time), makes registration independent
        # of new_group-hook liveness and idempotent with it (same key/value). The
        # non-SPMD / MPMD fallback path is unaffected: it never constructs a
        # ``NeuronGroupCoordinator``, so no mesh is registered and callers resolve
        # to the legacy single-cell mesh exactly as before.
        device_group = getattr(self, "device_group", None)
        if register_mesh is not None and device_group is not None:
            register_mesh(device_group.group_name, mesh)

    @property
    def rank_in_group_tensor(self):
        """This group's rank as an on-device tensor, for SPMD-safe reads.

        ``stamp_spmd_rank_tensors`` sets the backing attribute on every process
        group at distributed init. Reading the rank as a tensor rather than a
        Python int keeps it a runtime graph input, so all ranks trace one
        identical graph and the compile collapses to a single shared NEFF; an int
        would bake a per-rank constant and splinter the compile cache.

        Returns:
            A 1-D int32 tensor of shape ``(1,)`` holding ``rank_in_group``.
            Shape ``(1,)`` rather than scalar because NKI rejects 0-D shapes;
            :func:`shard_select` handles the reshape for slice call sites.

        Raises:
            AttributeError: If the group was never stamped. Call
                ``stamp_spmd_rank_tensors(device)`` after distributed init (both
                the serving and MPExecutor paths already do). Failing here is
                deliberate: silently deriving a CPU tensor would reintroduce the
                per-rank constant this property exists to avoid.

        Example:
            >>> # Keep it 1-D; do NOT .reshape(()) — a 0-D scalar gets
            >>> # constant-folded under chunked prefill (see fe1252e42).
            >>> offset = group.rank_in_group_tensor.to(torch.int64) * chunk
        """
        tensor = self.__dict__.get("_spmd_rank_in_group_tensor")
        if tensor is None:
            raise AttributeError(
                f"group '{getattr(self, 'unique_name', 'unknown')}' has no "
                "stamped rank tensor. Call stamp_spmd_rank_tensors(device) after "
                "distributed init before reading rank_in_group_tensor."
            )
        return tensor

    @rank_in_group_tensor.setter
    def rank_in_group_tensor(self, tensor) -> None:
        """Let ``stamp_spmd_rank_tensors`` (and the meta-swap) set the tensor."""
        self.__dict__["_spmd_rank_in_group_tensor"] = tensor
