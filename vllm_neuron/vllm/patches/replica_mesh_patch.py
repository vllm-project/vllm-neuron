# SPDX-License-Identifier: Apache-2.0
"""Route vLLM's group creation through the mesh-declaring Neuron factory.

XLA lowers a collective from the replica-group mesh, and every rank must lower
byte-identical HLO for the compile to collapse to a single SPMD NEFF. That
requires each process group to declare its *full* partition of the world, not
just the ranks local to this coordinator.

Two hooks are needed, and both must be installed before any process group is
created:

  1. ``apply_mesh_registry_patch`` (from libtorch) re-wraps ``dist.new_group``
     to capture meshes declared via ``xla_pg_options`` / ``active_mesh``.
     torch_xla replaces ``dist.new_group`` with its own wrapper at import time,
     which can clobber a hook installed earlier, so this is idempotent and
     safe -- and necessary -- to re-apply.

  2. ``_patch_vllm_init_model_parallel_group`` swaps vLLM's module-global
     ``init_model_parallel_group`` for the Neuron factory, so vLLM's own
     internal groups (_TP/_DP/_EP/_DCP/_PP) are built as
     ``NeuronGroupCoordinator`` and declare their meshes too. Those groups can
     be non-world-spanning (e.g. the cross-DP _DP group
     ``[[0,4],[1,5],[2,6],[3,7]]``) and are resolved by mesh-needing
     collectives, so they cannot be skipped.

  3. ``_patch_vllm_init_world_group`` does the same for ``_WORLD``, which
     ``init_distributed_environment`` builds through a *separate* module-global
     (``init_world_group``). The a2av token-permute and combine-reduce read
     ``get_world_group().rank_in_group_tensor``, so the world group needs to be
     a ``NeuronGroupCoordinator`` too. The replacement matches vLLM's arguments
     exactly -- only the class differs.

Applied from ``NeuronWorker.init_device()`` right before
``_init_neuron_distributed_environment_and_runtime()``: that location runs
after vLLM has finished importing (no circular ``vllm.utils.torch_utils``
import, which plugin-load time hit) and still before any process group is
created. ``apply_patches()`` (from ``NeuronPlatform.check_and_update_config()``)
would be too late -- groups may already exist. Spawn-mode workers re-enter
``init_device()``, so this survives re-imports without module-level execution.
"""

import logging

logger = logging.getLogger(__name__)


def apply_replica_mesh_patches() -> None:
    """Install both replica-mesh hooks. Idempotent; safe to call repeatedly.

    No-ops when ``libtorch_neuronx_lite`` is unavailable (CPU-only images do
    not ship it, since it links the neuron runtime), so importing vllm_neuron
    on a CPU host stays working.

    Example:
        >>> apply_replica_mesh_patches()
    """
    try:
        from libtorch_neuronx_lite.parallel.replica_mesh import (
            apply_mesh_registry_patch,
        )
    except ImportError:
        # CPU-only image: no libtorch, so no XLA collectives to lower and
        # nothing to declare meshes for.
        return

    apply_mesh_registry_patch()

    # Imported here rather than at module scope: neuron_parallel_state pulls in
    # vllm.distributed, and doing that at patch-module import time would run
    # before the platform plugin is fully registered.
    from vllm_neuron.parallel.neuron_parallel_state import (
        _patch_vllm_init_model_parallel_group,
        _patch_vllm_init_world_group,
    )

    _patch_vllm_init_model_parallel_group()
    _patch_vllm_init_world_group()
