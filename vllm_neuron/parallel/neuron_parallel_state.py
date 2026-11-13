# SPDX-License-Identifier: Apache-2.0
"""
Neuron parallel state — single owner of all process group initialization.

Why this exists:
    vLLM's parallel_state module provides TP, PP, DP, and EP GroupCoordinator
    instances for GPU-centric parallelism. Neuron requires additional parallelism
    dimensions that vLLM doesn't natively support:

    - Expert Parallelism (EP): Distributes MoE experts across ranks. With WS=8
      and EP=2, consecutive ranks form TP sub-groups within each EP partition:
        EP group 0 TP: [0,1,2,3]
        EP group 1 TP: [4,5,6,7]

    This module is the single entry point for ALL process group initialization on
    Neuron — both vLLM's groups and Neuron-specific groups.

    Two initialization paths:

    1. **Serving path** — ``init_neuron_distributed_environment()`` wraps vLLM's
       ``init_distributed_environment`` + ``ensure_model_parallel_initialized``
       and then creates Neuron groups. Called from NeuronWorker.

    2. **Test path** (MPExecutor) — ``initialize_neuron_parallel_state()`` is
       called after ``dist.init_process_group()`` with explicit
       ``tp_global_ranks`` / ``local_rank``. Bootstraps vLLM's ``_WORLD`` and
       ``_TP`` and creates Neuron groups in one shot.
"""

import contextlib
import logging
from datetime import timedelta
from typing import TYPE_CHECKING

import vllm.distributed.parallel_state as vllm_parallel_state

if TYPE_CHECKING:
    import torch
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_world_group,
)

from vllm_neuron.parallel.neuron_group_coordinator import NeuronGroupCoordinator

logger = logging.getLogger(__name__)

_TRN2_PHYSICAL_NCS_PER_DEVICE = 8

_SWEPT_GROUP_NAMES = frozenset(
    {
        "neuron_encoder_dp",
        "neuron_attention_dp",
    }
)

# Rank strides that Trn2 refuses, per LNC degree.
# Measured: 98 shapes on trn2.48xlarge, both degrees.
_REFUSED_STRIDES_BY_LNC: dict[int, frozenset[int]] = {
    2: frozenset({2, 8, 32}),  # 4 cores per device; strides 1, 4 and 16 work
    1: frozenset({2, 4, 16, 64}),  # 8 cores per device; strides 1, 8 and 32 work
}


def _group_stride(ranks: list[int]) -> int | None:
    """Return the constant rank stride of a group, or None when it varies."""
    if len(ranks) < 2:
        return 1
    stride = ranks[1] - ranks[0]
    for a, b in zip(ranks, ranks[1:]):
        if b - a != stride:
            return None
    return stride


def _trn2_group_shape_is_supported(ranks: list[int], lnc: int) -> bool:
    """Tell if a sweep measured this rank group as refused on Trn2.

    Trn2 only. The strides come from a Trn2 sweep, so a False here is not a
    verdict for Inf2, Trn1 or Trn3. The caller checks the family first.

    Returns True for every shape the sweep did not measure as refused, so this
    reports a known-bad configuration and does not predict a new one.

    Args:
        ranks: The global ranks of one group, in ascending order.
        lnc: The LNC degree the process runs at, 1 or 2.

    Returns:
        bool: False when the sweep measured this shape as refused.

    Examples:
        >>> _trn2_group_shape_is_supported([0, 4, 8, 12], lnc=2)
        True
        >>> _trn2_group_shape_is_supported([0, 4, 8, 12], lnc=1)
        False
    """
    refused = _REFUSED_STRIDES_BY_LNC.get(lnc)
    if refused is None or len(ranks) < 2:
        return True  # unknown degree, or nothing to check

    stride = _group_stride(ranks)
    if stride is None:
        return True  # no constant stride, so outside the measured set

    cores_per_device = _TRN2_PHYSICAL_NCS_PER_DEVICE // lnc
    if ranks[0] // cores_per_device == ranks[-1] // cores_per_device:
        return True  # one device, and every such group passed

    return stride not in refused


def _check_group_shapes(group_ranks: list[list[int]], group_name: str | None) -> None:
    """Stop, or warn, when a rank group matches a shape that Trn2 refuses."""
    from vllm_neuron import envs

    if envs.VLLM_NEURON_CPU_MODE:
        return

    from vllm_neuron.utils.hardware_config import (
        get_instance_family,
        get_logical_nc_config,
    )

    try:
        if get_instance_family() != "trn2":
            return
    except RuntimeError as e:
        logger.warning(
            "Cannot read the instance family (%s). Skipping the Trn2 "
            "group-shape check for the process group %r.",
            e,
            group_name,
        )
        return

    lnc = get_logical_nc_config()
    cores_per_device = _TRN2_PHYSICAL_NCS_PER_DEVICE // lnc

    for ranks in group_ranks:
        ordered = sorted(ranks)
        if _trn2_group_shape_is_supported(ordered, lnc):
            continue
        detail = (
            f"Process group {group_name!r} has the ranks {ordered} and a stride "
            f"of {_group_stride(ordered)}. A sweep measured this shape as "
            f"unsupported on Trn2 at LNC {lnc}, where one device holds "
            f"{cores_per_device} logical cores. The runtime cannot build a "
            f"collective for it, and it aborts at NEFF load. Use a stride of 1, "
            f"or of {'4 or 16' if lnc == 2 else '8 or 32'}."
        )
        if group_name in _SWEPT_GROUP_NAMES:
            raise ValueError(detail)
        logger.warning("%s", detail)
        return


def init_model_parallel_group(
    group_ranks: list[list[int]],
    local_rank: int,
    backend: str,
    use_message_queue_broadcaster: bool = False,
    group_name: str | None = None,
    use_device_communicator: bool = True,
    use_all2all: bool = False,
    mesh: list[list[int]] | None = None,
) -> GroupCoordinator:
    """Create a Neuron GroupCoordinator that declares its full replica mesh.

    Drop-in replacement for vLLM's ``init_model_parallel_group`` that builds a
    :class:`NeuronGroupCoordinator` so every group registers its full partition
    (the mesh) for SPMD-consistent XLA collective lowering. The mesh defaults to
    ``group_ranks`` and may be overridden via ``mesh`` for the rare case where
    the partition passed to the coordinator is not itself the intended mesh.

    Args:
        group_ranks: The complete partition of the world into replica groups.
        local_rank: Local device rank.
        backend: Distributed backend (e.g. "gloo").
        use_message_queue_broadcaster: Passed through to the coordinator.
        group_name: Optional group name.
        use_device_communicator: Whether to build a device communicator.
        mesh: Explicit full mesh; defaults to ``group_ranks``.

    Returns:
        The constructed :class:`NeuronGroupCoordinator`.

    Example:
        >>> tp = init_model_parallel_group([[0, 1], [2, 3]], 0, "gloo")

    """
    _check_group_shapes(group_ranks, group_name)
    return NeuronGroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=use_device_communicator,
        use_message_queue_broadcaster=use_message_queue_broadcaster,
        group_name=group_name,
        use_all2all=use_all2all,
        mesh=mesh,
    )


_vllm_factory_patched = False


def _patch_vllm_init_model_parallel_group() -> None:
    """Route vLLM's own group creation through the mesh-declaring factory.

    vLLM's ``initialize_model_parallel`` builds its internal groups (_TP, _DP,
    _EP, _DCP, _PP, ...) by calling the module-global
    ``init_model_parallel_group``. Those groups can be non-world-spanning (e.g.
    the cross-DP _DP group ``[[0,4],[1,5],[2,6],[3,7]]``) and are resolved by
    mesh-needing collectives, so they must declare their full partition too.

    We cannot edit vLLM, so we replace that module attribute with the Neuron
    factory (which builds :class:`NeuronGroupCoordinator`, defaulting the mesh to
    each group's ``group_ranks``). Idempotent; safe to call before every
    ``initialize_model_parallel``.

    Example:
        >>> _patch_vllm_init_model_parallel_group()
    """
    global _vllm_factory_patched
    if _vllm_factory_patched:
        return
    vllm_parallel_state.init_model_parallel_group = init_model_parallel_group
    _vllm_factory_patched = True
    logger.debug("Patched vllm.distributed.parallel_state.init_model_parallel_group")


_vllm_world_group_patched = False


def init_world_group(
    ranks: list[int], local_rank: int, backend: str
) -> GroupCoordinator:
    """Build the world group as a mesh-declaring Neuron coordinator.

    Drop-in replacement for vLLM's ``init_world_group``, matching its arguments
    exactly (including ``use_device_communicator=False`` and
    ``group_name="world"``) so the only difference is the class: a
    :class:`NeuronGroupCoordinator`, which declares its mesh and therefore
    exposes the ``rank_in_group_tensor`` property.

    That property matters because the a2av token-permute and combine-reduce read
    ``get_world_group().rank_in_group_tensor`` to derive per-rank token offsets
    at runtime. Leaving ``_WORLD`` as a bare ``GroupCoordinator`` would make
    those reads hit an un-typed instance attribute instead.

    Args:
        ranks: All world ranks.
        local_rank: Local device rank.
        backend: Distributed backend.

    Returns:
        The world :class:`NeuronGroupCoordinator`.

    Example:
        >>> world = init_world_group([0, 1, 2, 3], 0, "gloo")
    """
    return NeuronGroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=False,
        group_name="world",
    )


def _patch_vllm_init_world_group() -> None:
    """Route vLLM's world-group creation through the Neuron coordinator.

    ``init_distributed_environment`` builds ``_WORLD`` via the module-global
    ``init_world_group``, which is separate from ``init_model_parallel_group``
    and so needs its own patch. Idempotent; must run before
    ``init_distributed_environment``.

    Example:
        >>> _patch_vllm_init_world_group()
    """
    global _vllm_world_group_patched
    if _vllm_world_group_patched:
        return
    vllm_parallel_state.init_world_group = init_world_group
    _vllm_world_group_patched = True
    logger.debug("Patched vllm.distributed.parallel_state.init_world_group")


def _install_mesh_patches() -> None:
    """Re-install the replica-mesh patches at a distributed-init entry point.

    Delegates to ``vllm_neuron.vllm.patches.replica_mesh_patch``, which owns both
    hooks (see that module for why they are needed and why they run at import
    time). Re-applied here because torch_xla rebinds ``dist.new_group`` at its
    own import time, which can land after vllm_neuron was imported and drop our
    wrapper. Both hooks are idempotent.
    """
    from vllm_neuron.vllm.patches.replica_mesh_patch import (
        apply_replica_mesh_patches,
    )

    apply_replica_mesh_patches()


# ---------------------------------------------------------------------------
# Module-level GroupCoordinator singletons (mirrors vLLM's _TP, _PP, etc.)
# ---------------------------------------------------------------------------

_NEURON_EP: GroupCoordinator | None = None
_NEURON_EP_TP: GroupCoordinator | None = None

# attention DP group (cross-DP column: same TP position across attention_dp_size DP groups)
_NEURON_ATTENTION_DP: GroupCoordinator | None = None
# Attention TP group (TP * attention_dp_size ranks)
_NEURON_ATTENTION_TP: GroupCoordinator | None = None

# Embedding TP group (TP * embedding_dp_size ranks, passed as tp_group to embedding)
_NEURON_EMBEDDING_TP: GroupCoordinator | None = None
# Embedding DP column group (batch all-gather / reduce-scatter)
_NEURON_EMBEDDING_DP: GroupCoordinator | None = None

# LM Head TP group (TP * lm_head_dp_size ranks, passed as tp_group to lm_head)
_NEURON_LM_HEAD_TP: GroupCoordinator | None = None
# LM Head DP column group (batch all-gather / slice)
_NEURON_LM_HEAD_DP: GroupCoordinator | None = None

# MLP TP group (TP * mlp_dp_size ranks, passed as tp_group to MLP)
_NEURON_MLP_TP: GroupCoordinator | None = None
# MLP DP group (cross-DP column for mlp_dp_size)
_NEURON_MLP_DP: GroupCoordinator | None = None

# Draft MLP TP group (TP * draft_mlp_dp_size ranks, passed as tp_group to the
# Eagle3 draft MLP)
_NEURON_DRAFT_MLP_TP: GroupCoordinator | None = None
# Draft MLP DP group (cross-DP column for draft_mlp_dp_size)
_NEURON_DRAFT_MLP_DP: GroupCoordinator | None = None

# Encoder TP group (TP * encoder_dp_size ranks, or same as global TP when encoder_tp == text_tp)
_NEURON_ENCODER_TP: GroupCoordinator | None = None
# Encoder DP group (column group across encoder DP ranks at same TP position)
_NEURON_ENCODER_DP: GroupCoordinator | None = None
# DCP KV group (same heads, different tokens — AllGather Q + LSE correction here)
_NEURON_DCP_KV_GROUP: GroupCoordinator | None = None
# DCP TP group (different heads, same tokens — SP and weight sharding here)
_NEURON_DCP_TP_GROUP: GroupCoordinator | None = None

# Sampling DP groups
_SAMPLING_ALL2ALL: GroupCoordinator | None = None
_SAMPLING_GATHER: GroupCoordinator | None = None

# Wide EP group (all ranks with device communicator for cross-DP all-reduce)
_WIDE_EP: GroupCoordinator | None = None

# Inter-node positional peer group (cross-node: same local position across all nodes)
_NEURON_INTER_NODE_POSITIONAL_PEER: GroupCoordinator | None = None

# Intra-node EP group (portion of the EP group residing on the same server)
_NEURON_INTRANODE_EP: GroupCoordinator | None = None

# Inter-server EP group (cross-server peers derived from my EP column)
_NEURON_INTERNODE_EP: GroupCoordinator | None = None

# Node group (all ranks on the same server), independent of any parallelism dim
_NEURON_NODE_GROUP: GroupCoordinator | None = None

# Stashed original so our destroy wrapper can call through
_ORIGINAL_DESTROY_MODEL_PARALLEL = None


# ---------------------------------------------------------------------------
# Rank computation helpers
# ---------------------------------------------------------------------------


def _uses_trn2_torus_mesh() -> bool:
    """Whether the hand-laid Trn2 8x8 mesh applies to this host.

    True on Trn2, and when the instance family cannot be read (CPU mode, or a
    host without the sysfs product name), which keeps the long-standing
    default. False on every other family, Trn3 included.
    """
    from vllm_neuron import envs

    if envs.VLLM_NEURON_CPU_MODE:
        return True
    from vllm_neuron.utils.hardware_config import get_instance_family

    try:
        family = get_instance_family()
    except RuntimeError:
        return True
    return family == "trn2"


def _build_2d_mesh(
    ranks: list[int],
    row_size: int,
    num_rows: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Build a 2D mesh of ranks and return (row_groups, col_groups).

    On TRN2 with a 64-rank 8x8 setup, uses a special non-contiguous mesh
    topology unless ``VLLM_NEURON_SWITCH_CC`` is set. All parallelism dimensions
    (EP, Sampling DP) share this mesh logic.

    TODO: Remove VLLM_NEURON_SWITCH_CC and derive topology from instance type.

    Args:
        ranks: Global rank IDs to arrange in the mesh.
        row_size: Number of ranks per row.
        num_rows: Number of rows.

    Returns:
        Tuple of (row_groups, col_groups) where each is a list of rank lists.

    Example: ranks=[0..7], row_size=4, num_rows=2
      row_groups: [[0,1,2,3], [4,5,6,7]]
      col_groups: [[0,4], [1,5], [2,6], [3,7]]
    """
    from vllm_neuron import envs

    total = row_size * num_rows

    # Special case: 8x8 topology for TRN2. The pairing below (devices 0+3, 1+2,
    # ...) follows Trn2's 4x4 torus; on a Trn3 box the device graph is a switch
    # and those pairs are not linked, so the runtime refuses the group:
    #   ENC:enc_post_operation Collective operation ALLREDUCE ... over group id 0
    #   (rank_n:8 node_n:1) is currently not supported.
    #   Replica-group 0: [0,1,2,3,12,13,14,15]
    # So the mesh is applied to Trn2 only; every other family keeps contiguous
    # rows, whose device pairs (0+1, 2+3, ...) are linked on Trn3.
    if (
        not envs.VLLM_NEURON_SWITCH_CC
        and total == 64
        and row_size == 8
        and _uses_trn2_torus_mesh()
    ):
        mesh = [
            [0, 1, 2, 3, 12, 13, 14, 15],
            [4, 5, 6, 7, 8, 9, 10, 11],
            [16, 17, 18, 19, 28, 29, 30, 31],
            [20, 21, 22, 23, 24, 25, 26, 27],
            [32, 33, 34, 35, 44, 45, 46, 47],
            [36, 37, 38, 39, 40, 41, 42, 43],
            [48, 49, 50, 51, 60, 61, 62, 63],
            [52, 53, 54, 55, 56, 57, 58, 59],
        ]
    else:
        mesh = [ranks[i * row_size : (i + 1) * row_size] for i in range(num_rows)]

    row_groups = [list(row) for row in mesh]
    col_groups = [[mesh[r][c] for r in range(num_rows)] for c in range(row_size)]

    return row_groups, col_groups


def _build_ep_group_ranks(
    world_size: int,
    ep_degree: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Compute EP-TP (row) and EP (column) group rank lists.

    Example: world_size=8, ep_degree=2
      EP-TP groups: [[0,1,2,3], [4,5,6,7]]
      EP groups:    [[0,4], [1,5], [2,6], [3,7]]
    """
    tp_per_ep = world_size // ep_degree
    return _build_2d_mesh(
        list(range(world_size)), row_size=tp_per_ep, num_rows=ep_degree
    )


def _build_sampling_dp_meshes(
    tp_degree: int,
    dp_degree: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Compute all2all and gather meshes for data-parallel sampling.

    all2all mesh: groups of dp_degree ranks for batch/vocab redistribution.
    gather mesh: transpose of all2all mesh, for gathering topk results.

    Example: tp_degree=8, dp_degree=2
      all2all: [[0,1], [2,3], [4,5], [6,7]]
      gather:  [[0,2,4,6], [1,3,5,7]]
    """
    gather_grp_size = tp_degree // dp_degree
    return _build_2d_mesh(
        list(range(tp_degree)), row_size=dp_degree, num_rows=gather_grp_size
    )


def _build_component_tp_group_ranks(
    world_size: int,
    tp_size: int,
    dp_component_size: int,
) -> list[list[int]]:
    """
    Build rectangular supergroup rank lists of size ``tp_size * dp_component_size``.

    Each supergroup contains ``dp_component_size`` consecutive DP groups' worth of
    ranks. Ranks within a supergroup are ordered so that TP all-gather followed by
    DP-component all-gather produces contiguous output.

    Example: world_size=32, tp_size=8, dp_component_size=4
      One supergroup: [0,1,2,...,31]

    Example: world_size=64, tp_size=8, dp_component_size=4  (partial: 2 supergroups)
      Supergroup 0: [0,1,...,31]
      Supergroup 1: [32,33,...,63]

    Returns:
        List of supergroups, each a list of ``tp_size * dp_component_size`` global ranks.
    """
    dp_size = world_size // tp_size
    if dp_size % dp_component_size != 0:
        raise ValueError(
            f"dp_size ({dp_size}) must be divisible by dp_component_size ({dp_component_size})"
        )

    num_supergroups = dp_size // dp_component_size
    groups = []

    for sg in range(num_supergroups):
        dp_start = sg * dp_component_size
        # Order: all TP ranks for DP0, then all TP ranks for DP1, ...
        # This gives contiguous vocab ordering after all-gather within the group.
        ranks = []
        for d in range(dp_component_size):
            for tp_pos in range(tp_size):
                ranks.append((dp_start + d) * tp_size + tp_pos)
        groups.append(ranks)

    return groups


def _build_dp_column_group_ranks(
    world_size: int,
    tp_size: int,
    attention_dp_size: int,
) -> list[list[int]]:
    """
    Compute attention DP column group rank lists.

    Each attention DP column group connects ranks at the same TP position across
    ``attention_dp_size`` consecutive DP groups. With partial attention DP (attention_dp_size < dp_size),
    multiple independent attention DP supergroups are formed.

    Example: world_size=32, tp_size=8, attention_dp_size=4  (4 DP groups, full attention DP)
      DP groups: [0..7], [8..15], [16..23], [24..31]
      attention DP columns (one per TP position):
        [0, 8, 16, 24], [1, 9, 17, 25], ..., [7, 15, 23, 31]

    Example: world_size=64, tp_size=8, attention_dp_size=4  (8 DP groups, partial attention DP=4)
      Supergroup 0 (DPs 0-3): [0..31]
        columns: [0,8,16,24], [1,9,17,25], ...
      Supergroup 1 (DPs 4-7): [32..63]
        columns: [32,40,48,56], [33,41,49,57], ...

    Returns:
        List of column groups, each a list of ``attention_dp_size`` global ranks.
    """
    dp_size = world_size // tp_size
    if dp_size % attention_dp_size != 0:
        raise ValueError(
            f"dp_size ({dp_size}) must be divisible by attention_dp_size ({attention_dp_size})"
        )

    num_supergroups = dp_size // attention_dp_size
    col_groups = []

    for sg in range(num_supergroups):
        # DP groups in this supergroup
        dp_start = sg * attention_dp_size
        # For each TP position, build the column group
        for tp_pos in range(tp_size):
            col = [(dp_start + d) * tp_size + tp_pos for d in range(attention_dp_size)]
            col_groups.append(col)

    return col_groups


def _build_inter_node_positional_peer_group_ranks(
    world_size: int,
    nnodes: int,
) -> list[list[int]]:
    """
    Compute inter-node positional peer group rank lists for cross-node communication.

    Each group connects ranks at the same local position across all nodes.
    Used for all-to-all-v in MoE where experts at the same position on
    different nodes need to exchange tokens.

    Example: world_size=16, nnodes=2 (8 ranks per node)
      Group 0: [0, 8]   (local pos 0 on each node)
      Group 1: [1, 9]   (local pos 1 on each node)
      ...
      Group 7: [7, 15]  (local pos 7 on each node)

    Example: world_size=64, nnodes=8 (8 ranks per node)
      Group 0: [0, 8, 16, 24, 32, 40, 48, 56]
      Group 1: [1, 9, 17, 25, 33, 41, 49, 57]
      ...
      Group 7: [7, 15, 23, 31, 39, 47, 55, 63]

    Example: world_size=128, nnodes=2 (64 ranks per node)
      Group 0: [0, 64]
      Group 1: [1, 65]
      ...
      Group 63: [63, 127]

    Returns:
        List of groups, each a list of ``nnodes`` global ranks.
    """
    if world_size % nnodes != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by nnodes ({nnodes})"
        )
    ranks_per_node = world_size // nnodes
    return [
        [node * ranks_per_node + local_pos for node in range(nnodes)]
        for local_pos in range(ranks_per_node)
    ]


def _build_internode_ep_group_ranks(
    ep_group_ranks: list[list[int]],
    ranks_per_node: int,
) -> list[list[int]]:
    """Cross-server peers for each on-server position within an EP column.

    For each EP column, partition by node into per-server slices, then for each
    on-server position p emit one group: [slice_node0[p], slice_node1[p], ...].

    Example: EP col [0,4,8,12], ranks_per_node=8 -> per-server slices [0,4] and [8,12]
        position-0 group: [0, 8]; position-1 group: [4, 12].
    """
    groups: list[list[int]] = []
    for ep_group in ep_group_ranks:
        by_node: dict[int, list[int]] = {}
        for r in ep_group:
            by_node.setdefault(r // ranks_per_node, []).append(r)
        nodes_sorted = sorted(by_node)
        if not nodes_sorted:
            continue
        slice_len = len(by_node[nodes_sorted[0]])
        if any(len(by_node[n]) != slice_len for n in nodes_sorted):
            raise ValueError(
                f"EP column {ep_group} does not split evenly across nodes "
                f"(ranks_per_node={ranks_per_node})"
            )
        for pos in range(slice_len):
            groups.append([by_node[n][pos] for n in nodes_sorted])
    return groups


def _build_intranode_ep_group_ranks(
    ep_group_ranks: list[list[int]],
    ranks_per_node: int,
) -> list[list[int]]:
    """
    Split each EP (column) group into per-node subgroups.

    The intra-node EP group is the portion of a rank's EP group that lives on the
    same server. Used for the second (intra-node) stage of hierarchical all-to-all-v,
    after the inter-node positional-peer stage has shipped tokens to the right node.

    Each input EP group is partitioned by node id (``rank // ranks_per_node``),
    preserving order, and every non-empty per-node slice becomes its own subgroup.

    Example: ep_group_ranks=[[0,1,...,15]], ranks_per_node=8 (world_size=16, nnodes=2)
      Intra-node EP groups: [[0,1,2,3,4,5,6,7], [8,9,10,11,12,13,14,15]]

    Example: ep_group_ranks=[[0,8],[1,9],...,[7,15]], ranks_per_node=8 (ep_degree=2)
      Each EP group already straddles 2 nodes, so each splits into two size-1
      subgroups: [[0],[8],[1],[9],...] (intra-node EP is trivial here).

    Returns:
        List of subgroups, each a list of global ranks sharing one node.
    """
    subgroups: list[list[int]] = []
    for ep_group in ep_group_ranks:
        by_node: dict[int, list[int]] = {}
        for rank in ep_group:
            by_node.setdefault(rank // ranks_per_node, []).append(rank)
        for node in sorted(by_node):
            subgroups.append(by_node[node])
    return subgroups


def _build_node_group_ranks(world_size: int, nnodes: int) -> list[list[int]]:
    """Partition the world into contiguous per-server blocks.

    Ranks are laid out server-major (server s owns [s*rpn, (s+1)*rpn)), so each
    block holds every rank on one server. Independent of any parallelism dim.

    Example: world_size=16, nnodes=2 -> [[0..7], [8..15]].
    """
    if world_size % nnodes != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by nnodes ({nnodes})"
        )
    ranks_per_node = world_size // nnodes
    return [
        list(range(s * ranks_per_node, (s + 1) * ranks_per_node)) for s in range(nnodes)
    ]


# ---------------------------------------------------------------------------
# Core: create Neuron-specific groups (assumes vLLM state exists)
# ---------------------------------------------------------------------------


def _create_neuron_groups(
    ep_degree: int,
    sampling_dp_degree: int = 1,
    attention_dp_size: int = 1,
    embedding_dp_size: int = 1,
    lm_head_dp_size: int = 1,
    mlp_dp_size: int = 1,
    draft_mlp_dp_size: int | None = None,
    encoder_tp_size: int = 1,
    encoder_dp_size: int = 1,
    nnodes: int = 1,
) -> None:
    """
    Create Neuron-specific GroupCoordinators (EP, Sampling DP, attention DP, DCP KV).

    Assumes vLLM's ``_WORLD`` and ``_TP`` are already initialized.
    """
    global _NEURON_EP, _NEURON_EP_TP
    global _SAMPLING_ALL2ALL, _SAMPLING_GATHER
    global _NEURON_ATTENTION_DP, _NEURON_ATTENTION_TP
    global _NEURON_EMBEDDING_TP, _NEURON_EMBEDDING_DP
    global _NEURON_LM_HEAD_TP, _NEURON_LM_HEAD_DP
    global _NEURON_MLP_TP, _NEURON_MLP_DP
    global _NEURON_DRAFT_MLP_TP, _NEURON_DRAFT_MLP_DP
    global _NEURON_ENCODER_TP, _NEURON_ENCODER_DP
    global _WIDE_EP
    global _NEURON_DCP_KV_GROUP
    global _NEURON_INTER_NODE_POSITIONAL_PEER
    global _NEURON_INTRANODE_EP
    global _NEURON_INTERNODE_EP
    global _NEURON_NODE_GROUP

    tp_coordinator = vllm_parallel_state.get_tp_group()
    tp_global_ranks = tp_coordinator.ranks
    tp_size = tp_coordinator.world_size
    local_rank = get_world_group().local_rank
    backend = "gloo"

    # --- DCP KV group ---
    # Connects ranks that have the same KV heads but different tokens (used for
    # AllGather Q + LSE correction). Matches vllm's _DCP layout: consecutive
    # chunks of dcp_size within each TP group.
    # Example: TP=8, DCP=2 → KV groups [[0,1],[2,3],[4,5],[6,7]]
    # Example: TP=8, DCP=4 → KV groups [[0,1,2,3],[4,5,6,7]]
    # When DCP=1, creates size-1 groups (no-op gathers).
    from vllm.distributed.parallel_state import _DCP

    dcp_size = _DCP.world_size if _DCP is not None else 1

    world_size = get_world_group().world_size
    num_tp_groups = world_size // tp_size
    tp_pair_size = tp_size // dcp_size
    all_kv_groups = []
    for g in range(num_tp_groups):
        base = g * tp_size
        for chunk in range(tp_pair_size):
            kv_group = [base + chunk * dcp_size + r for r in range(dcp_size)]
            all_kv_groups.append(kv_group)

    _NEURON_DCP_KV_GROUP = init_model_parallel_group(
        all_kv_groups,
        local_rank,
        backend,
        group_name="neuron_dcp_kv",
    )
    logger.debug(
        "Initialized DCP KV gather GroupCoordinator: ranks=%s",
        _NEURON_DCP_KV_GROUP.ranks,
    )

    # DCP TP group: ranks that share the same token chunk but have different
    # head shards. Used for SP and weight sharding. Same position across all
    # DCP groups within the TP group (strided by dcp_size).
    # Example: TP=8, DCP=2 → TP pairs [[0,2,4,6],[1,3,5,7]] (size TP/DCP=4)
    # Example: TP=8, DCP=4 → TP pairs [[0,4],[1,5],[2,6],[3,7]] (size TP/DCP=2)
    global _NEURON_DCP_TP_GROUP
    all_tp_pairs = []
    for g in range(num_tp_groups):
        base = g * tp_size
        for pos in range(dcp_size):
            tp_pair = [base + pos + i * dcp_size for i in range(tp_pair_size)]
            all_tp_pairs.append(tp_pair)

    _NEURON_DCP_TP_GROUP = init_model_parallel_group(
        all_tp_pairs,
        local_rank,
        backend,
        group_name="neuron_dcp_tp",
    )
    logger.debug(
        "Initialized DCP TP GroupCoordinator: ranks=%s",
        _NEURON_DCP_TP_GROUP.ranks,
    )

    # --- Node group ---
    # All ranks on the same server, independent of any parallelism dimension.
    node_group_ranks = _build_node_group_ranks(world_size, nnodes)
    _NEURON_NODE_GROUP = init_model_parallel_group(
        node_group_ranks,
        local_rank,
        backend,
        group_name="neuron_node",
    )
    logger.debug(
        "Initialized Node GroupCoordinator: ranks=%s",
        _NEURON_NODE_GROUP.ranks,
    )

    # --- Expert Parallelism ---
    if ep_degree > 1:
        world_size = get_world_group().world_size

        if world_size % ep_degree != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by "
                f"ep_degree ({ep_degree})"
            )

        ep_tp_group_ranks, ep_group_ranks = _build_ep_group_ranks(world_size, ep_degree)

        _NEURON_EP_TP = init_model_parallel_group(
            ep_tp_group_ranks,
            local_rank,
            backend,
            group_name="neuron_ep_tp",
        )
        _NEURON_EP = init_model_parallel_group(
            ep_group_ranks,
            local_rank,
            backend,
            group_name="neuron_ep",
        )

        logger.debug(
            "Initialized EP GroupCoordinators: ep_tp_ranks=%s, ep_ranks=%s",
            _NEURON_EP_TP.ranks,
            _NEURON_EP.ranks,
        )

        # Intra-node EP: the portion of each EP group on the same server. This is
        # the destination group for the second (intra-node) stage of hierarchical
        # all-to-all-v. When nnodes == 1 it is identical to the EP group.
        ranks_per_node = world_size // nnodes
        intranode_ep_group_ranks = _build_intranode_ep_group_ranks(
            ep_group_ranks, ranks_per_node
        )
        _NEURON_INTRANODE_EP = init_model_parallel_group(
            intranode_ep_group_ranks,
            local_rank,
            backend,
            group_name="neuron_intranode_ep",
        )
        logger.debug(
            "Initialized Intra-Node EP GroupCoordinator: ranks=%s",
            _NEURON_INTRANODE_EP.ranks,
        )

        # Inter-server EP: cross-server peers within each EP column. Only
        # meaningful when nnodes > 1 AND the EP column straddles servers.
        if nnodes > 1:
            internode_ep_group_ranks = _build_internode_ep_group_ranks(
                ep_group_ranks, ranks_per_node
            )
            _NEURON_INTERNODE_EP = init_model_parallel_group(
                internode_ep_group_ranks,
                local_rank,
                backend,
                group_name="neuron_internode_ep",
            )
            logger.debug(
                "Initialized Inter-Node EP GroupCoordinator: ranks=%s",
                _NEURON_INTERNODE_EP.ranks,
            )

        # Wide EP: Create world-spanning group with device communicator only when
        # EP spans across DP replicas (dp_size > 1). vLLM's world_group has
        # use_device_communicator=False, so we create our own for fast tensor collectives.
        dp_size = world_size // tp_size
        if dp_size > 1:
            _WIDE_EP = init_model_parallel_group(
                group_ranks=[list(range(world_size))],
                local_rank=local_rank,
                backend=backend,
                use_device_communicator=True,
                group_name="wide_ep",
            )
            logger.debug(
                "Initialized Wide EP GroupCoordinator: ranks=%s",
                _WIDE_EP.ranks,
            )

    # --- Sampling Data Parallelism ---
    if sampling_dp_degree > 1:
        if tp_size % sampling_dp_degree != 0:
            raise ValueError(
                f"tp_size ({tp_size}) must be divisible by "
                f"sampling_dp_degree ({sampling_dp_degree})"
            )

        all2all_mesh, gather_mesh = _build_sampling_dp_meshes(
            tp_size, sampling_dp_degree
        )

        _SAMPLING_ALL2ALL = init_model_parallel_group(
            all2all_mesh,
            local_rank,
            backend,
            group_name="neuron_sampling_all2all",
        )
        _SAMPLING_GATHER = init_model_parallel_group(
            gather_mesh,
            local_rank,
            backend,
            group_name="neuron_sampling_gather",
        )
        logger.debug(
            "Initialized Sampling DP GroupCoordinators: all2all_ranks=%s, gather_ranks=%s",
            _SAMPLING_ALL2ALL.ranks,
            _SAMPLING_GATHER.ranks,
        )

    # --- Inter-Node Positional Peer Group (cross-node) ---
    if nnodes > 1:
        world_size = get_world_group().world_size
        peer_group_ranks = _build_inter_node_positional_peer_group_ranks(
            world_size, nnodes
        )
        _NEURON_INTER_NODE_POSITIONAL_PEER = init_model_parallel_group(
            peer_group_ranks,
            local_rank,
            backend,
            group_name="neuron_inter_node_positional_peer",
        )
        logger.debug(
            "Initialized Inter-Node Positional Peer GroupCoordinator: peer_ranks=%s",
            _NEURON_INTER_NODE_POSITIONAL_PEER.ranks,
        )

    # --- Attention Dependent DP ---
    if attention_dp_size > 1:
        world_size = get_world_group().world_size
        dp_size = world_size // tp_size

        if dp_size % attention_dp_size != 0:
            raise ValueError(
                f"dp_size ({dp_size}) must be divisible by attention_dp_size ({attention_dp_size})"
            )

        attention_dp_col_groups = _build_dp_column_group_ranks(
            world_size, tp_size, attention_dp_size
        )

        _NEURON_ATTENTION_DP = init_model_parallel_group(
            attention_dp_col_groups,
            local_rank,
            backend,
            group_name="neuron_attention_dp",
        )

        logger.debug(
            "Initialized attention DP GroupCoordinator: attention_dp_ranks=%s",
            _NEURON_ATTENTION_DP.ranks,
        )

    # --- Embedding / LM Head / MLP TP supergroups ---
    # Always created; when dp_size=1 they equal the regular TP group.
    world_size = get_world_group().world_size

    # Build TP supergroups, reusing when sizes match
    _component_tp_cache: dict[int, GroupCoordinator] = {}

    def _get_component_tp_group(dp_component_size: int, name: str) -> GroupCoordinator:
        if dp_component_size in _component_tp_cache:
            return _component_tp_cache[dp_component_size]
        groups = _build_component_tp_group_ranks(world_size, tp_size, dp_component_size)
        coord = init_model_parallel_group(groups, local_rank, backend, group_name=name)
        _component_tp_cache[dp_component_size] = coord
        return coord

    _NEURON_ATTENTION_TP = _get_component_tp_group(
        attention_dp_size, "neuron_attention_tp"
    )
    _NEURON_EMBEDDING_TP = _get_component_tp_group(
        embedding_dp_size, "neuron_embedding_tp"
    )
    _NEURON_LM_HEAD_TP = _get_component_tp_group(lm_head_dp_size, "neuron_lm_head_tp")
    _NEURON_MLP_TP = _get_component_tp_group(mlp_dp_size, "neuron_mlp_tp")
    # The caches above return the target group when the degrees match, so equal
    # degrees create nothing new.
    if draft_mlp_dp_size is None:
        draft_mlp_dp_size = mlp_dp_size
    _NEURON_DRAFT_MLP_TP = _get_component_tp_group(
        draft_mlp_dp_size, "neuron_draft_mlp_tp"
    )

    # Build DP column groups, reusing when sizes match
    _dp_col_cache: dict[int, GroupCoordinator] = {}

    def _get_dp_col_group(dp_component_size: int, name: str) -> GroupCoordinator:
        if dp_component_size in _dp_col_cache:
            return _dp_col_cache[dp_component_size]
        if dp_component_size == attention_dp_size and _NEURON_ATTENTION_DP is not None:
            _dp_col_cache[dp_component_size] = _NEURON_ATTENTION_DP
            return _NEURON_ATTENTION_DP
        dp_size = world_size // tp_size
        if dp_size % dp_component_size != 0:
            raise ValueError(
                f"dp_size ({dp_size}) must be divisible by {name} dp_component_size ({dp_component_size})"
            )
        groups = _build_dp_column_group_ranks(world_size, tp_size, dp_component_size)
        coord = init_model_parallel_group(groups, local_rank, backend, group_name=name)
        _dp_col_cache[dp_component_size] = coord
        return coord

    _NEURON_ATTENTION_DP = _get_dp_col_group(attention_dp_size, "neuron_attention_dp")
    _NEURON_EMBEDDING_DP = _get_dp_col_group(embedding_dp_size, "neuron_embedding_dp")
    _NEURON_LM_HEAD_DP = _get_dp_col_group(lm_head_dp_size, "neuron_lm_head_dp")
    _NEURON_MLP_DP = _get_dp_col_group(mlp_dp_size, "neuron_mlp_dp")
    _NEURON_DRAFT_MLP_DP = _get_dp_col_group(draft_mlp_dp_size, "neuron_draft_mlp_dp")

    # --- Encoder TP group ---
    # When encoder_tp_size equals tp_size, reuse the global TP group (no new group needed).
    # When encoder_tp_size differs, create a separate group.
    if encoder_tp_size > 0 and encoder_tp_size != tp_size:
        # Encoder uses a different TP degree — create its own group
        # Each encoder TP group has encoder_tp_size ranks
        # TODO: This assumes contiguous rank layout (valid for single-node
        # trn2.48xlarge). For multi-node, derive encoder TP groups from the
        # actual TP group topology instead of hardcoded contiguous ranges.
        num_encoder_tp_groups = world_size // encoder_tp_size
        encoder_tp_groups = [
            list(range(i * encoder_tp_size, (i + 1) * encoder_tp_size))
            for i in range(num_encoder_tp_groups)
        ]
        _NEURON_ENCODER_TP = init_model_parallel_group(
            encoder_tp_groups, local_rank, backend, group_name="neuron_encoder_tp"
        )
    else:
        _NEURON_ENCODER_TP = None  # will fall back to get_tp_group()

    # --- Encoder DP group ---
    # When encoder_dp_size > 1, create a column group of ranks at the same TP
    # position across encoder DP groups. Used for scatter/gather of encoder blocks.
    # E.g., world_size=16, encoder_tp=4, encoder_dp=4:
    #   TP groups: [0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]
    #   DP columns: [0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]
    if encoder_dp_size > 1:
        encoder_dp_col_groups = _build_dp_column_group_ranks(
            world_size, encoder_tp_size, encoder_dp_size
        )
        _NEURON_ENCODER_DP = init_model_parallel_group(
            encoder_dp_col_groups, local_rank, backend, group_name="neuron_encoder_dp"
        )
    else:
        _NEURON_ENCODER_DP = None


def _patch_destroy() -> None:
    """Wrap vLLM's ``destroy_model_parallel`` to also destroy Neuron groups."""
    global _ORIGINAL_DESTROY_MODEL_PARALLEL

    if _ORIGINAL_DESTROY_MODEL_PARALLEL is None:
        _ORIGINAL_DESTROY_MODEL_PARALLEL = vllm_parallel_state.destroy_model_parallel
    vllm_parallel_state.destroy_model_parallel = _neuron_destroy_model_parallel


def _patch_getters() -> None:
    """Add Neuron-specific getter functions to vLLM's parallel_state module."""
    vllm_parallel_state.get_neuron_ep_group = get_neuron_ep_group
    vllm_parallel_state.get_neuron_ep_tp_group = get_neuron_ep_tp_group
    vllm_parallel_state.get_neuron_ep_degree = get_neuron_ep_degree
    vllm_parallel_state.get_neuron_ep_rank = get_neuron_ep_rank
    vllm_parallel_state.get_neuron_sampling_all2all_group = (
        get_neuron_sampling_all2all_group
    )
    vllm_parallel_state.get_neuron_sampling_gather_group = (
        get_neuron_sampling_gather_group
    )
    vllm_parallel_state.get_neuron_sampling_dp_degree = get_neuron_sampling_dp_degree
    vllm_parallel_state.get_neuron_attention_dp_group = get_neuron_attention_dp_group
    vllm_parallel_state.get_neuron_attention_dp_size = get_neuron_attention_dp_size
    vllm_parallel_state.get_neuron_attention_dp_rank = get_neuron_attention_dp_rank
    vllm_parallel_state.get_neuron_attention_tp_group = get_neuron_attention_tp_group
    vllm_parallel_state.get_neuron_embedding_tp_group = get_neuron_embedding_tp_group
    vllm_parallel_state.get_neuron_embedding_dp_group = get_neuron_embedding_dp_group
    vllm_parallel_state.get_neuron_lm_head_tp_group = get_neuron_lm_head_tp_group
    vllm_parallel_state.get_neuron_lm_head_dp_group = get_neuron_lm_head_dp_group
    vllm_parallel_state.get_neuron_mlp_tp_group = get_neuron_mlp_tp_group
    vllm_parallel_state.get_neuron_mlp_dp_group = get_neuron_mlp_dp_group
    vllm_parallel_state.get_neuron_mlp_dp_size = get_neuron_mlp_dp_size
    vllm_parallel_state.get_neuron_mlp_dp_rank = get_neuron_mlp_dp_rank
    vllm_parallel_state.get_neuron_draft_mlp_tp_group = get_neuron_draft_mlp_tp_group
    vllm_parallel_state.get_neuron_draft_mlp_dp_group = get_neuron_draft_mlp_dp_group
    vllm_parallel_state.get_neuron_draft_mlp_dp_size = get_neuron_draft_mlp_dp_size
    vllm_parallel_state.get_neuron_encoder_tp_group = get_neuron_encoder_tp_group
    vllm_parallel_state.get_wide_ep_group = get_wide_ep_group
    vllm_parallel_state.get_neuron_dcp_kv_group = get_neuron_dcp_kv_group
    vllm_parallel_state.get_neuron_dcp_tp_group = get_neuron_dcp_tp_group
    vllm_parallel_state.get_neuron_inter_node_positional_peer_group = (
        get_neuron_inter_node_positional_peer_group
    )
    vllm_parallel_state.get_neuron_intranode_ep_group = get_neuron_intranode_ep_group
    vllm_parallel_state.get_neuron_internode_ep_group = get_neuron_internode_ep_group
    vllm_parallel_state.get_node_group = get_node_group
    vllm_parallel_state.is_internode_group = is_internode_group


def _neuron_destroy_model_parallel() -> None:
    """Wrapper: destroy Neuron groups, then call vLLM's original destroy."""
    destroy_neuron_parallel_state()

    if _ORIGINAL_DESTROY_MODEL_PARALLEL is not None:
        _ORIGINAL_DESTROY_MODEL_PARALLEL()


# ---------------------------------------------------------------------------
# Serving path: single entry point for NeuronWorker
# ---------------------------------------------------------------------------


def init_neuron_distributed_environment(
    world_size: int,
    rank: int,
    local_rank: int,
    distributed_init_method: str | None,
    backend: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int = 1,
    decode_context_parallel_size: int = 1,
    ep_degree: int = 1,
    sampling_dp_degree: int = 1,
    attention_dp_size: int = 1,
    embedding_dp_size: int = 1,
    lm_head_dp_size: int = 1,
    mlp_dp_size: int = 1,
    draft_mlp_dp_size: int | None = None,
    encoder_tp_size: int = 1,
    encoder_dp_size: int = 1,
    nnodes: int = 1,
    timeout: timedelta | None = None,
) -> None:
    """
    Initialize all distributed state for Neuron in one call.

    Wraps vLLM's ``init_distributed_environment`` and
    ``ensure_model_parallel_initialized``, then creates Neuron-specific
    groups (EP, attention DP). Called from ``NeuronWorker._init_neuron_distributed_environment_and_runtime``.

    Args:
        world_size: Total number of ranks.
        rank: Global rank of this process.
        local_rank: Local rank on this node.
        distributed_init_method: PyTorch distributed init method (e.g. "env://").
        backend: Distributed backend (e.g. "gloo", "xla", "neuron").
        tensor_parallel_size: TP degree.
        pipeline_parallel_size: PP degree.
        decode_context_parallel_size: DCP degree.
        ep_degree: Expert parallelism degree.
        attention_dp_size: KV data-parallelism degree (decode-only Q/O sharding).
        sampling_dp_degree: Sampling data-parallelism degree.
        embedding_dp_size: Embedding data-parallelism degree.
        lm_head_dp_size: LM head data-parallelism degree.
        mlp_dp_size: MLP data-parallelism degree.
        draft_mlp_dp_size: Eagle3 draft MLP data-parallelism degree. None
            inherits mlp_dp_size.
        encoder_tp_size: Encoder TP degree (default 1 means no encoder TP).
        encoder_dp_size: Encoder DP degree (default 1 means no encoder DP).
        nnodes: Number of nodes in the deployment.
    """
    # 0. Install the SPMD mesh patches before any process group is created, so
    # every group (Neuron + vLLM-internal) declares its full replica mesh.
    _install_mesh_patches()

    # 1. vLLM distributed init (dist.init_process_group + _WORLD)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method=distributed_init_method,
        backend=backend,
        timeout=timeout,
    )

    # 2. vLLM model-parallel init (_TP, _PP, _DP, _EP, _DCP, _PCP)
    ensure_model_parallel_initialized(
        tensor_parallel_size,
        pipeline_parallel_size,
        1,  # pcp=1 (not used on Neuron)
        decode_context_parallel_size,
        backend=backend,
    )

    # 3. Neuron-specific groups (EP, Sampling DP, attention DP, DCP KV, etc.)
    _create_neuron_groups(
        ep_degree,
        sampling_dp_degree,
        attention_dp_size,
        embedding_dp_size,
        lm_head_dp_size,
        mlp_dp_size,
        draft_mlp_dp_size,
        encoder_tp_size,
        encoder_dp_size,
        nnodes,
    )

    # 4. Patch vLLM's module with getters and destroy wrapper
    _patch_getters()
    _patch_destroy()


# ---------------------------------------------------------------------------
# Test path: direct initialization (MPExecutor)
# ---------------------------------------------------------------------------


def _ensure_vllm_parallel_state(
    tp_global_ranks: list[int],
    local_rank: int,
    tp_size: int,
    dp_size: int = 1,
    dcp_size: int = 1,
    backend: str = "gloo",
    nnodes: int = 1,
) -> None:
    """
    One-time bootstrap of vLLM's parallel state (test/MPExecutor path only).

    Sets up the persistent foundation that survives across REINIT_PARALLEL cycles:
      - _WORLD group (all ranks)
      - _NODE_COUNT
      - in_the_same_node_as patch (avoids barrier crash on Neuron)

    Then calls _reinit_vllm_model_parallel to create the sub-groups (_TP, _DP, _PP).

    Only runs once per worker process (_WORLD guard). The sub-groups created here
    may later be destroyed and recreated by _reinit_vllm_model_parallel when
    dp_size changes, but the _WORLD group and patches persist forever.
    """
    if vllm_parallel_state._WORLD is not None:
        return

    # WORLD group must include ALL ranks (across all DP replicas).
    # vLLM's initialize_model_parallel uses world_size to compute DP groups.
    import torch.distributed as dist

    world_size = dist.get_world_size()
    all_ranks = list(range(world_size))

    vllm_parallel_state._WORLD = vllm_parallel_state.init_world_group(
        all_ranks,
        local_rank,
        backend,
    )
    vllm_parallel_state._NODE_COUNT = nnodes

    # Patch in_the_same_node_as to avoid barrier() crash on Neuron.
    # Mirrors NeuronWorker._patch_in_same_node_as_function: torch.distributed
    # .barrier() calls torch._C._get_accelerator() before backend dispatch,
    # which fails on Neuron (PrivateUse1 with no PrivateUse1HooksInterface).
    # Map each group-local rank back to its global rank before assigning a
    # node, so subgroups (TP/DP/EP) report same-node membership correctly
    # regardless of how their ranks are laid out across nodes.
    ranks_per_node = world_size // nnodes

    def patched_in_the_same_node_as(pg, source_rank=0):
        global_ranks = [
            dist.get_global_rank(pg, r) for r in range(dist.get_world_size(group=pg))
        ]
        source_node = global_ranks[source_rank] // ranks_per_node
        return [g // ranks_per_node == source_node for g in global_ranks]

    vllm_parallel_state.in_the_same_node_as = patched_in_the_same_node_as

    _reinit_vllm_model_parallel(tp_size, dp_size, backend, dcp_size)


def _reinit_vllm_model_parallel(
    tp_size: int,
    dp_size: int = 1,
    backend: str = "gloo",
    dcp_size: int = 1,
) -> None:
    """
    (Re)initialize vLLM's model-parallel sub-groups (_TP, _DP, _PP, etc.).

    This is the repeatable counterpart to _ensure_vllm_parallel_state.
    It only creates the sub-groups — it does NOT touch _WORLD or patches.

    Called in two scenarios:
      1. Initial setup: from _ensure_vllm_parallel_state after _WORLD is created.
      2. REINIT_PARALLEL: from the executor worker loop after destroy_model_parallel()
         when dp_size changes and sub-groups need rebuilding with new sizes.

    Creates a minimal VllmConfig with the given tp_size/dp_size and passes it
    to vLLM's initialize_model_parallel via set_current_vllm_config.
    """
    from vllm.config import VllmConfig, ParallelConfig, set_current_vllm_config

    parallel_config = ParallelConfig(
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        decode_context_parallel_size=dcp_size,
    )
    vllm_config = VllmConfig(parallel_config=parallel_config)
    # Ensure vLLM's internal groups (_TP, _DP, _EP, _DCP, ...) also declare their
    # full replica mesh — some (e.g. cross-DP _DP) are non-world-spanning and are
    # resolved by mesh-needing XLA collectives.
    _install_mesh_patches()
    with set_current_vllm_config(vllm_config):
        vllm_parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            decode_context_model_parallel_size=dcp_size,
            backend=backend,
        )


def initialize_neuron_parallel_state(
    ep_degree: int = 1,
    sampling_dp_degree: int = 1,
    dp_size: int = 1,
    attention_dp_size: int = 1,
    embedding_dp_size: int = 1,
    lm_head_dp_size: int = 1,
    mlp_dp_size: int = 1,
    draft_mlp_dp_size: int | None = None,
    encoder_tp_size: int = 1,
    encoder_dp_size: int = 1,
    dcp_size: int = 1,
    nnodes: int = 1,
    *,
    tp_global_ranks: list[int] | None = None,
    local_rank: int | None = None,
) -> None:
    """
    Initialize all parallel state (vLLM + Neuron) in one shot.

    For standalone tests (MPExecutor) where only ``dist.init_process_group``
    has been called. Pass ``tp_global_ranks`` and ``local_rank`` to bootstrap
    vLLM's ``_WORLD`` and ``_TP``, then create Neuron groups on top.

    For the serving path, use ``init_neuron_distributed_environment()`` instead.

    Args:
        ep_degree: Expert parallelism degree.
        dp_size: vLLM data parallelism degree. When >1, vLLM creates DP groups
            and EP groups can span across DP boundaries.
        dcp_size: Decode context parallelism degree (also used for prefill CP).
        sampling_dp_degree: Sampling data-parallelism degree.
        attention_dp_size: Attention data-parallelism degree.
        embedding_dp_size: Embedding data-parallelism degree.
        lm_head_dp_size: LM head data-parallelism degree.
        mlp_dp_size: MLP data-parallelism degree.
        draft_mlp_dp_size: Eagle3 draft MLP data-parallelism degree. None
            inherits mlp_dp_size.
        encoder_tp_size: Encoder TP degree (default 1, full DP).
        encoder_dp_size: Encoder DP degree (1 means no encoder DP).
        nnodes: Number of nodes in the deployment.
        tp_global_ranks: Global ranks forming the TP group for this rank's
            DP replica. When provided, bootstraps vLLM's parallel state.
        local_rank: Local rank on this node.
    """
    if is_initialized():
        logger.warning("Neuron parallel state already initialized, skipping.")
        return

    # Install the SPMD mesh patches before any process group is created.
    _install_mesh_patches()

    # Bootstrap vLLM's parallel state if needed (test path)
    if tp_global_ranks is not None:
        if local_rank is None:
            import torch.distributed as dist

            local_rank = dist.get_rank()
        tp_size = len(tp_global_ranks)
        _ensure_vllm_parallel_state(
            tp_global_ranks,
            local_rank,
            tp_size=tp_size,
            dp_size=dp_size,
            dcp_size=dcp_size,
            nnodes=nnodes,
        )
    _create_neuron_groups(
        ep_degree,
        sampling_dp_degree,
        attention_dp_size,
        embedding_dp_size,
        lm_head_dp_size,
        mlp_dp_size,
        draft_mlp_dp_size,
        encoder_tp_size,
        encoder_dp_size,
        nnodes,
    )
    _patch_getters()
    _patch_destroy()


def shard_select(
    x: "torch.Tensor",
    dim: int,
    group: GroupCoordinator,
    n_shards: int | None = None,
    rank: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """SPMD-safe equivalent of ``x[rank * chunk : (rank + 1) * chunk]`` on ``dim``.

    Slice/narrow bounds must be trace-time ints, so a Python-int rank bakes a
    per-rank constant into the graph and splinters the compile cache into one
    NEFF per rank. Deriving the offset from the group's stamped rank tensor keeps
    it a runtime graph input, so every rank traces an identical graph.

    Owns the dtype/shape handling (the stamp is ``int32[1]`` because NKI rejects
    0-D shapes) so call sites don't repeat ``.to(torch.int64).reshape(())``.

    Args:
        x: Tensor to shard.
        dim: Dimension to shard along.
        group: Process group whose ``rank_in_group`` selects the shard.
        n_shards: Number of shards. Defaults to ``group.world_size``; pass
            explicitly when the shard count differs from the group size.
        rank: Optional rank tensor within ``group``. The function uses
            ``group.rank_in_group_tensor`` when this value is ``None``.
            Pass this value to derive all group ranks from one graph input.

    Returns:
        ``x``'s shard for this rank, of length ``x.shape[dim] // n_shards``.

    Example:
        >>> local = shard_select(hidden_states, 0, get_neuron_mlp_dp_group())
    """
    import torch

    n = n_shards if n_shards is not None else group.world_size
    total = x.shape[dim]
    if total % n != 0:
        raise ValueError(
            f"shard_select: dim {dim} has size {total}, not divisible by "
            f"{n} shards (group={getattr(group, 'unique_name', group)})."
        )
    chunk = total // n
    # Keep the rank tensor 1-D (shape (1,)) — do NOT ``.reshape(())`` to a 0-D
    # scalar. Under chunked prefill (max_num_batched_tokens < max_model_len)
    # torch.compile constant-folds a 0-D reshape into a Python scalar and bakes
    # ``slice(rank*chunk, (rank+1)*chunk)`` into the FX graph per rank,
    # splintering the compile cache. The 1-D form broadcasts through arange +
    # add without collapsing, so every rank traces one identical graph.
    rank_tensor = group.rank_in_group_tensor if rank is None else rank
    offset = rank_tensor.to(torch.int64) * chunk
    idx = torch.arange(chunk, device=x.device, dtype=torch.int64) + offset
    return x.index_select(dim, idx)


def stamp_spmd_rank_tensors(device) -> list:
    """Stamp every process group's rank_in_group as an on-device int32 tensor.

    Enables SPMD compatibility: torch.compile treats rank_in_group_tensor as a
    graph input, so all ranks share one graph. An int ``rank_in_group`` would bake
    in a per-rank constant instead. Models read it via
    ``group.rank_in_group_tensor`` (or ``shard_select``, which owns the dtype
    and shape handling).

    Returns the stamped tensors so the caller can keep them alive.

    Args:
        device: Target device for the stamped tensors (e.g. ``"neuron:0"``).

    Returns:
        The list of stamped ``rank_in_group_tensor`` tensors (one per group).

    Example:
        >>> ranks = stamp_spmd_rank_tensors("neuron:0")
    """
    import torch
    from vllm.distributed import parallel_state as ps

    # ps._groups is a vLLM-internal dict (name -> weakref[GroupCoordinator]),
    # confirmed present with this shape in the pinned vLLM 0.24.0. It is not
    # public API: if a future vLLM pin renames or restructures it, this loop
    # raises AttributeError at call time (fail-loud), which surfaces at
    # distributed-init rather than as a later confusing meta/device rank-tensor
    # collision. Re-verify on every vLLM version bump.
    stamped: list[torch.Tensor] = []
    names: list[str] = []
    for name, group_ref in list(ps._groups.items()):
        group = group_ref()
        if group is None:
            continue
        if getattr(group, "rank_in_group_tensor", None) is None:
            # Stamp as a 1-D tensor of shape (1,); this is more friendly for NKI, which rejects 0-D shape ()
            group.rank_in_group_tensor = torch.tensor(
                [group.rank_in_group], dtype=torch.int32, device=device
            )
        stamped.append(group.rank_in_group_tensor)
        names.append(f"{name}(r={group.rank_in_group},ws={group.world_size})")
    logger.info("SPMD rank tensors stamped on %d groups: %s", len(names), names)
    return stamped


def try_get_rank_in_group_tensor(group: GroupCoordinator):
    """Return ``group``'s stamped rank tensor, or ``None`` if it has none.

    Named accessor for the "stamped-or-None" contract, so callers that must
    tolerate an unstamped group (direct non-SPMD entry points that skip
    ``stamp_spmd_rank_tensors``) don't reach into the attribute with a bare
    ``getattr(group, "rank_in_group_tensor", None)`` and silently swallow the
    property's deliberate ``AttributeError``.

    Unlike reading ``group.rank_in_group_tensor`` directly -- which raises
    ``AttributeError`` on an unstamped :class:`NeuronGroupCoordinator` to stop a
    per-rank CPU tensor leaking into a traced graph -- this returns ``None`` so
    the caller can pick a trace-safe fallback (e.g. a tensor built from the
    eager ``rank_in_group`` int). Prefer reading the property directly on the
    SPMD path where a missing stamp is a bug worth failing on.

    Args:
        group: Process group to read the stamped rank tensor from.

    Returns:
        The 1-D int32 ``rank_in_group_tensor`` if stamped, else ``None``.

    Example:
        >>> t = try_get_rank_in_group_tensor(get_world_group())
        >>> rank_id = t % stride if t is not None else _eager_fallback()
    """
    try:
        return group.rank_in_group_tensor
    except AttributeError:
        # NeuronGroupCoordinator raises when unstamped; a base vLLM
        # GroupCoordinator has no such attribute at all. Both mean "no stamp".
        return None


@contextlib.contextmanager
def spmd_rank_tensors_on_meta():
    """Temporarily swap every group's ``rank_in_group_tensor`` to the meta device.

    Wrap a meta-device compile/trace with this so the stamped SPMD rank tensors
    flow into the graph as *meta* graph inputs (matching the meta-swapped model
    params) instead of the real-device buffers ``stamp_spmd_rank_tensors``
    installs for runtime. On exit the originals are restored, so the real
    per-rank value is bound at execution time.

    Why this is needed: ``stamp_spmd_rank_tensors(device)`` stamps each process
    group's rank tensor on the *real* device (e.g. ``neuron:6``) so it becomes a
    runtime graph input. But that tensor lives on the vLLM group object, not on
    the model's params/buffers, so it is NOT touched when the parallel-trace
    fork-child meta-swaps the model (``_swap_to_meta_no_free`` only walks
    ``nn.Module`` params/buffers/attrs). During a meta trace the model +
    synthetic inputs are on meta while the rank tensor stays on neuron, so any
    ``arange(..., device=meta) + rank_tensor`` (SP token_base_index, EP
    ``_ep_rank_id``, ``local_expert_indices``, moe_blockwise ``col_start_id``,
    hierarchical ``current_node`` …) raises "two different devices meta, neuron"
    (FakeTensor propagation), and even a read-site ``.to(meta)`` fails because
    the rank tensor is *lifted* as a real-device graph input that the CPU-compile
    input validator (all inputs must be meta) then rejects.

    Only groups currently on a non-meta device are swapped and restored; groups
    already on meta (e.g. nested calls, CPU-compile where the stamp device was
    already meta) are left untouched.

    Example:
        >>> with spmd_rank_tensors_on_meta():
        ...     capture_backend_model(**meta_kwargs)  # traces on meta
    """
    import torch
    from vllm.distributed import parallel_state as ps

    saved: list = []
    for _name, group_ref in list(ps._groups.items()):
        group = group_ref()
        if group is None:
            continue
        tensor = getattr(group, "rank_in_group_tensor", None)
        if tensor is None or tensor.device.type == "meta":
            continue
        saved.append((group, tensor))
        group.rank_in_group_tensor = torch.empty(
            tensor.shape, dtype=tensor.dtype, device="meta"
        )
    try:
        yield
    finally:
        for group, tensor in saved:
            group.rank_in_group_tensor = tensor


# ---------------------------------------------------------------------------
# Public getters
# ---------------------------------------------------------------------------


def is_initialized() -> bool:
    """Return True if any Neuron parallel groups have been created."""
    return any(
        g is not None
        for g in [
            _NEURON_EP,
            _NEURON_EP_TP,
            _SAMPLING_ALL2ALL,
            _SAMPLING_GATHER,
            _NEURON_ATTENTION_DP,
            _NEURON_ATTENTION_TP,
            _NEURON_EMBEDDING_TP,
            _NEURON_EMBEDDING_DP,
            _NEURON_LM_HEAD_TP,
            _NEURON_LM_HEAD_DP,
            _NEURON_MLP_TP,
            _NEURON_MLP_DP,
            _NEURON_DRAFT_MLP_TP,
            _NEURON_DRAFT_MLP_DP,
            _NEURON_ENCODER_TP,
            _WIDE_EP,
            _NEURON_DCP_KV_GROUP,
            _NEURON_DCP_TP_GROUP,
            _NEURON_INTER_NODE_POSITIONAL_PEER,
            _NEURON_INTRANODE_EP,
            _NEURON_INTERNODE_EP,
        ]
    )


def get_neuron_ep_group() -> GroupCoordinator:
    """Return the expert-parallelism GroupCoordinator for this rank."""
    assert _NEURON_EP is not None, (
        "Neuron EP group is not initialized. "
        "Call initialize_neuron_parallel_state() with ep_degree > 1."
    )
    return _NEURON_EP


def get_neuron_ep_tp_group() -> GroupCoordinator:
    """Return the TP sub-group within the EP partition for this rank."""
    assert _NEURON_EP_TP is not None, (
        "Neuron EP-TP group is not initialized. "
        "Call initialize_neuron_parallel_state() with ep_degree > 1."
    )
    return _NEURON_EP_TP


def get_neuron_ep_degree() -> int:
    """Return the expert-parallelism degree (1 if not initialized)."""
    if _NEURON_EP is None:
        return 1
    return get_world_group().world_size // _NEURON_EP_TP.world_size


def get_neuron_ep_rank() -> int:
    """Return this rank's EP partition index (0 if not initialized)."""
    if _NEURON_EP is None:
        return 0
    return _NEURON_EP.rank_in_group


def get_neuron_sampling_all2all_group() -> GroupCoordinator:
    """Return the all2all GroupCoordinator for sampling DP."""
    assert _SAMPLING_ALL2ALL is not None, (
        "Sampling all2all group is not initialized. "
        "Call init with sampling_dp_degree > 1."
    )
    return _SAMPLING_ALL2ALL


def get_neuron_sampling_gather_group() -> GroupCoordinator:
    """Return the gather GroupCoordinator for sampling DP."""
    assert _SAMPLING_GATHER is not None, (
        "Sampling gather group is not initialized. "
        "Call init with sampling_dp_degree > 1."
    )
    return _SAMPLING_GATHER


def get_neuron_sampling_dp_degree() -> int:
    """Return the sampling data-parallelism degree (1 if not initialized)."""
    if _SAMPLING_ALL2ALL is None:
        return 1
    return _SAMPLING_ALL2ALL.world_size


def get_neuron_attention_dp_group() -> GroupCoordinator:
    """Return the attention DP column GroupCoordinator for this rank."""
    assert _NEURON_ATTENTION_DP is not None, (
        "Neuron attention DP group is not initialized. "
        "Call initialize_neuron_parallel_state() with attention_dp_size > 1."
    )
    return _NEURON_ATTENTION_DP


def get_neuron_attention_dp_size() -> int:
    """Return the attention DP degree (1 if not initialized)."""
    if _NEURON_ATTENTION_DP is None:
        return 1
    return _NEURON_ATTENTION_DP.world_size


def get_neuron_attention_dp_rank() -> int:
    """Return this rank's position within its attention DP column group (0 if not initialized)."""
    if _NEURON_ATTENTION_DP is None:
        return 0
    return _NEURON_ATTENTION_DP.rank_in_group


def get_neuron_attention_tp_group() -> GroupCoordinator:
    """Return the Attention TP GroupCoordinator (TP * attention_dp_size ranks)."""
    assert _NEURON_ATTENTION_TP is not None, (
        "Neuron Attention TP group is not initialized."
    )
    return _NEURON_ATTENTION_TP


def get_neuron_dcp_kv_group() -> GroupCoordinator:
    """Return the DCP KV gather group for this rank.

    Ranks with same heads but different tokens (size DCP).
    Used for AllGather Q + LSE correction during prefill with context parallelism.
    """
    assert _NEURON_DCP_KV_GROUP is not None, "DCP KV group is not initialized."
    return _NEURON_DCP_KV_GROUP


def get_neuron_dcp_tp_group() -> GroupCoordinator:
    """Return the DCP TP group for this rank.

    Ranks with different heads but same tokens (size TP/DCP).
    Used for SP and weight sharding in DCP prefill.
    """
    assert _NEURON_DCP_TP_GROUP is not None, "DCP TP group is not initialized."
    return _NEURON_DCP_TP_GROUP


def get_neuron_embedding_tp_group() -> GroupCoordinator:
    """Return the Embedding TP GroupCoordinator (TP * embedding_dp_size ranks)."""
    assert _NEURON_EMBEDDING_TP is not None, (
        "Neuron Embedding TP group is not initialized. "
        "Call initialize_neuron_parallel_state() with embedding_dp_size > 1."
    )
    return _NEURON_EMBEDDING_TP


def get_neuron_embedding_dp_group() -> GroupCoordinator:
    """Return the Embedding DP column GroupCoordinator for batch all-gather/reduce-scatter."""
    assert _NEURON_EMBEDDING_DP is not None, (
        "Neuron Embedding DP group is not initialized. "
        "Call initialize_neuron_parallel_state() with embedding_dp_size > 1."
    )
    return _NEURON_EMBEDDING_DP


def get_neuron_lm_head_tp_group() -> GroupCoordinator:
    """Return the LM Head TP GroupCoordinator (TP * lm_head_dp_size ranks)."""
    assert _NEURON_LM_HEAD_TP is not None, (
        "Neuron LM Head TP group is not initialized. "
        "Call initialize_neuron_parallel_state() with lm_head_dp_size > 1."
    )
    return _NEURON_LM_HEAD_TP


def get_neuron_lm_head_dp_group() -> GroupCoordinator:
    """Return the LM Head DP column GroupCoordinator for batch all-gather/slice."""
    assert _NEURON_LM_HEAD_DP is not None, (
        "Neuron LM Head DP group is not initialized. "
        "Call initialize_neuron_parallel_state() with lm_head_dp_size > 1."
    )
    return _NEURON_LM_HEAD_DP


def get_neuron_mlp_tp_group() -> GroupCoordinator:
    """Return the MLP TP GroupCoordinator (TP * mlp_dp_size ranks)."""
    assert _NEURON_MLP_TP is not None, (
        "Neuron MLP TP group is not initialized. "
        "Call initialize_neuron_parallel_state() with mlp_dp_size > 1."
    )
    return _NEURON_MLP_TP


def get_neuron_mlp_dp_group() -> GroupCoordinator:
    """Return the MLP DP column GroupCoordinator for this rank."""
    assert _NEURON_MLP_DP is not None, (
        "Neuron MLP DP group is not initialized. "
        "Call initialize_neuron_parallel_state() with mlp_dp_size > 1."
    )
    return _NEURON_MLP_DP


def get_neuron_mlp_dp_size() -> int:
    """Return the MLP DP degree (1 if not initialized)."""
    if _NEURON_MLP_DP is None:
        return 1
    return _NEURON_MLP_DP.world_size


def get_neuron_mlp_dp_rank() -> int:
    """Return this rank's position within its MLP DP column group (0 if not initialized)."""
    if _NEURON_MLP_DP is None:
        return 0
    return _NEURON_MLP_DP.rank_in_group


def get_neuron_draft_mlp_tp_group() -> GroupCoordinator:
    """Return the draft MLP TP GroupCoordinator (TP * draft_mlp_dp_size ranks)."""
    assert _NEURON_DRAFT_MLP_TP is not None, (
        "Neuron draft MLP TP group is not initialized. "
        "Call initialize_neuron_parallel_state() first. The group exists at "
        "every draft_mlp_dp_size, including 1."
    )
    return _NEURON_DRAFT_MLP_TP


def get_neuron_draft_mlp_dp_group() -> GroupCoordinator:
    """Return the draft MLP DP column GroupCoordinator for this rank."""
    assert _NEURON_DRAFT_MLP_DP is not None, (
        "Neuron draft MLP DP group is not initialized. "
        "Call initialize_neuron_parallel_state() first. The group exists at "
        "every draft_mlp_dp_size, including 1."
    )
    return _NEURON_DRAFT_MLP_DP


def get_neuron_draft_mlp_dp_size() -> int:
    """Return the draft MLP DP degree (1 if not initialized)."""
    if _NEURON_DRAFT_MLP_DP is None:
        return 1
    return _NEURON_DRAFT_MLP_DP.world_size


def get_neuron_encoder_tp_group() -> GroupCoordinator:
    """Return the Encoder TP GroupCoordinator.

    Falls back to the global TP group when no separate encoder TP group
    was created (i.e., encoder uses the same TP as text).
    """
    if _NEURON_ENCODER_TP is None:
        return vllm_parallel_state.get_tp_group()
    return _NEURON_ENCODER_TP


def get_neuron_encoder_dp_group() -> GroupCoordinator | None:
    """Return the Encoder DP GroupCoordinator, or None if encoder DP is disabled.

    When encoder dp_size > 1, this group contains ranks at the same TP position
    across encoder DP groups. Used for scatter/gather of encoder blocks.
    Returns None when dp_size == 1 (no encoder DP).
    """
    return _NEURON_ENCODER_DP


def get_wide_ep_group() -> GroupCoordinator:
    """Return the Wide EP GroupCoordinator (all ranks with device communicator).

    This group is used for all-reduce operations that span across all TP and DP ranks
    when expert parallelism spans across data parallel replicas (ep_degree > 1 and dp_size > 1).
    Unlike vLLM's world_group which has use_device_communicator=False, this group has a
    device communicator for fast tensor collectives.
    """
    assert _WIDE_EP is not None, (
        "Wide EP group is not initialized. "
        "This group is only created when ep_degree > 1 and dp_size > 1."
    )
    return _WIDE_EP


def get_neuron_inter_node_positional_peer_group() -> GroupCoordinator:
    """Return the inter-node positional peer GroupCoordinator for cross-node collectives."""
    assert _NEURON_INTER_NODE_POSITIONAL_PEER is not None, (
        "Neuron inter-node positional peer group is not initialized. "
        "Call initialize with nnodes > 1."
    )
    return _NEURON_INTER_NODE_POSITIONAL_PEER


def get_neuron_intranode_ep_group() -> GroupCoordinator:
    """Return the intra-node EP GroupCoordinator (portion of the EP group on this server).

    This is the destination group for the second (intra-node) stage of hierarchical
    all-to-all-v. When nnodes == 1 it spans the same ranks as the full EP group.
    """
    assert _NEURON_INTRANODE_EP is not None, (
        "Neuron intra-node EP group is not initialized. "
        "This group is only created when ep_degree > 1."
    )
    return _NEURON_INTRANODE_EP


def get_neuron_internode_ep_group() -> GroupCoordinator:
    """Return the inter-server EP GroupCoordinator (cross-server peers in my EP column)."""
    assert _NEURON_INTERNODE_EP is not None, (
        "Neuron inter-server EP group is not initialized. "
        "This group is only created when ep_degree > 1 and nnodes > 1."
    )
    return _NEURON_INTERNODE_EP


def is_internode_group(group) -> bool:
    """Whether ``group`` spans multiple nodes (i.e. crosses servers).

    Accepts either a ``GroupCoordinator`` or an iterable of global ranks. Returns
    False for single-node deployments; otherwise True iff the group's ranks
    straddle more than one node under the current ranks-per-node layout.
    """
    node_count = vllm_parallel_state.get_node_count()
    if node_count == 1:
        return False

    world_size = vllm_parallel_state.get_world_group().world_size
    ranks_per_node = world_size // node_count

    ranks = group.ranks if hasattr(group, "ranks") else group
    group_nodes = {r // ranks_per_node for r in ranks}
    return len(group_nodes) > 1


def get_node_group() -> GroupCoordinator:
    """Return the GroupCoordinator for all ranks on this rank's server.

    The group's ranks are the contiguous server-major block owning this rank
    (server s owns [s*rpn, (s+1)*rpn)), so every rank on the same server shares
    one coordinator — the group needed for on-server (intra-node) collectives.

    Example:
        >>> get_node_group().ranks  # world_size=16, nnodes=2, this rank on server 0
        [0, 1, 2, 3, 4, 5, 6, 7]
    """
    assert _NEURON_NODE_GROUP is not None, (
        "Neuron node group is not initialized. "
        "Call initialize_neuron_parallel_state() first."
    )
    return _NEURON_NODE_GROUP


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


def destroy_neuron_parallel_state() -> None:
    """Destroy Neuron GroupCoordinators and remove patched attributes."""
    global _NEURON_EP, _NEURON_EP_TP
    global _SAMPLING_ALL2ALL, _SAMPLING_GATHER
    global _NEURON_ATTENTION_DP, _NEURON_ATTENTION_TP
    global _NEURON_EMBEDDING_TP, _NEURON_EMBEDDING_DP
    global _NEURON_LM_HEAD_TP, _NEURON_LM_HEAD_DP
    global _NEURON_MLP_TP, _NEURON_MLP_DP
    global _NEURON_DRAFT_MLP_TP, _NEURON_DRAFT_MLP_DP
    global _NEURON_ENCODER_TP, _NEURON_ENCODER_DP
    global _WIDE_EP
    global _NEURON_DCP_KV_GROUP, _NEURON_DCP_TP_GROUP
    global _NEURON_INTER_NODE_POSITIONAL_PEER
    global _NEURON_INTRANODE_EP
    global _NEURON_INTERNODE_EP
    global _NEURON_NODE_GROUP
    global _ORIGINAL_DESTROY_MODEL_PARALLEL

    # Collect unique groups to destroy (some may be aliased)
    seen_ids = set()
    for group in [
        _NEURON_EP,
        _NEURON_EP_TP,
        _SAMPLING_ALL2ALL,
        _SAMPLING_GATHER,
        _NEURON_ATTENTION_DP,
        _NEURON_ATTENTION_TP,
        _NEURON_EMBEDDING_TP,
        _NEURON_EMBEDDING_DP,
        _NEURON_LM_HEAD_TP,
        _NEURON_LM_HEAD_DP,
        _NEURON_MLP_TP,
        _NEURON_MLP_DP,
        _NEURON_DRAFT_MLP_TP,
        _NEURON_DRAFT_MLP_DP,
        _NEURON_ENCODER_TP,
        _NEURON_ENCODER_DP,
        _WIDE_EP,
        _NEURON_DCP_KV_GROUP,
        _NEURON_DCP_TP_GROUP,
        _NEURON_INTER_NODE_POSITIONAL_PEER,
        _NEURON_INTRANODE_EP,
        _NEURON_INTERNODE_EP,
        _NEURON_NODE_GROUP,
    ]:
        if group is not None and id(group) not in seen_ids:
            seen_ids.add(id(group))
            group.destroy()

    _NEURON_EP = None
    _NEURON_EP_TP = None
    _SAMPLING_ALL2ALL = None
    _SAMPLING_GATHER = None
    _NEURON_ATTENTION_DP = None
    _NEURON_ATTENTION_TP = None
    _NEURON_EMBEDDING_TP = None
    _NEURON_EMBEDDING_DP = None
    _NEURON_LM_HEAD_TP = None
    _NEURON_LM_HEAD_DP = None
    _NEURON_MLP_TP = None
    _NEURON_MLP_DP = None
    _NEURON_DRAFT_MLP_TP = None
    _NEURON_DRAFT_MLP_DP = None
    _NEURON_ENCODER_TP = None
    _WIDE_EP = None
    _NEURON_DCP_KV_GROUP = None
    _NEURON_DCP_TP_GROUP = None
    _NEURON_INTER_NODE_POSITIONAL_PEER = None
    _NEURON_INTRANODE_EP = None
    _NEURON_INTERNODE_EP = None
    _NEURON_NODE_GROUP = None
    # Restore original destroy_model_parallel
    if _ORIGINAL_DESTROY_MODEL_PARALLEL is not None:
        vllm_parallel_state.destroy_model_parallel = _ORIGINAL_DESTROY_MODEL_PARALLEL
        _ORIGINAL_DESTROY_MODEL_PARALLEL = None

    # Remove patched getter attributes
    for attr in [
        "get_neuron_ep_group",
        "get_neuron_ep_tp_group",
        "get_neuron_ep_degree",
        "get_neuron_ep_rank",
        "get_neuron_sampling_all2all_group",
        "get_neuron_sampling_gather_group",
        "get_neuron_sampling_dp_degree",
        "get_neuron_attention_dp_group",
        "get_neuron_attention_dp_size",
        "get_neuron_attention_dp_rank",
        "get_neuron_attention_tp_group",
        "get_neuron_embedding_tp_group",
        "get_neuron_embedding_dp_group",
        "get_neuron_lm_head_tp_group",
        "get_neuron_lm_head_dp_group",
        "get_neuron_mlp_tp_group",
        "get_neuron_mlp_dp_group",
        "get_neuron_mlp_dp_size",
        "get_neuron_mlp_dp_rank",
        "get_neuron_draft_mlp_tp_group",
        "get_neuron_draft_mlp_dp_group",
        "get_neuron_draft_mlp_dp_size",
        "get_neuron_encoder_tp_group",
        "get_wide_ep_group",
        "get_neuron_dcp_kv_group",
        "get_neuron_dcp_tp_group",
        "get_neuron_inter_node_positional_peer_group",
        "get_neuron_intranode_ep_group",
        "get_neuron_internode_ep_group",
        "get_node_group",
        "is_internode_group",
    ]:
        if hasattr(vllm_parallel_state, attr):
            delattr(vllm_parallel_state, attr)


def world_barrier(timeout: timedelta | None = None):
    """Barrier that bypasses torch.distributed.barrier() to avoid PrivateUse1 crash.
    PyTorch's barrier() unconditionally calls torch._C._get_accelerator()
    before dispatching to the backend, which crashes on Neuron because
    PrivateUse1HooksInterface is not registered. This helper goes directly
    to the gloo cpu_group on the world coordinator, skipping that codepath.
    """
    from vllm.distributed.parallel_state import get_world_group

    timeout = timeout or _default_barrier_timeout()
    group = get_world_group()
    if group.world_size > 1:
        group.cpu_group.set_timeout(timeout)
        group.cpu_group.barrier().wait(timeout=timeout)


def tp_barrier(timeout: timedelta | None = None):
    """TP-group-scoped barrier using the same PrivateUse1 workaround.

    Synchronizes only the ranks within the local TP group. This is the
    correct scope for graph capture and compilation coordination, since
    each TP group (one model replica) captures and compiles independently.
    """
    from vllm.distributed.parallel_state import get_tp_group

    timeout = timeout or _default_barrier_timeout()
    group = get_tp_group()
    if group.world_size > 1:
        group.cpu_group.set_timeout(timeout)
        group.cpu_group.barrier().wait(timeout=timeout)


def _default_barrier_timeout() -> timedelta:
    """Barrier timeout from env, read lazily (reflects call-time env, not import)."""
    from vllm_neuron import envs

    return timedelta(seconds=envs.VLLM_NEURON_BARRIER_TIMEOUT)
