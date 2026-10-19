# Neuron Parallel State Design

______________________________________________________________________

## 1. Problem Statement

| Challenge | Impact |
|---|---|
| vLLM's parallel state only supports TP, PP, DP, EP for GPU topologies | Neuron needs additional parallelism dimensions (EP with custom TP sub-groups) |
| Models created their own process groups in `__init__` | Duplicated group creation across models, no centralized lifecycle management |
| Process groups leaked on teardown | Groups not destroyed when `destroy_model_parallel` was called |
| Tests bypassed vLLM's parallel state entirely | `dist.group.WORLD` used directly, diverging from vLLM conventions |

**Goal**: Single module that owns all process group initialization (both vLLM's and Neuron's), follows vLLM's `GroupCoordinator` conventions, and provides a clean path toward contributing these parallelism dimensions upstream.

______________________________________________________________________

## 2. Architecture

```text
neuron_parallel_state.py
    |
    |-- init_neuron_distributed_environment()    # Serving path (NeuronWorker)
    |       |-- vllm: init_distributed_environment()    -> _WORLD
    |       |-- vllm: ensure_model_parallel_initialized() -> _TP, _PP, _DP, _EP, ...
    |       |-- neuron: _create_neuron_groups()          -> _NEURON_EP_TP, _NEURON_EP
    |       |-- _patch_getters()                         -> adds get_neuron_*() to vllm PS
    |       |-- _patch_destroy()                         -> wraps destroy_model_parallel
    |
    |-- initialize_neuron_parallel_state()       # Test path (MPExecutor)
    |       |-- _ensure_vllm_parallel_state()           -> _WORLD, _TP (lightweight)
    |       |-- _create_neuron_groups()                  -> same as above
    |       |-- _patch_getters() / _patch_destroy()
    |
    |-- destroy_neuron_parallel_state()          # Teardown (both paths)
            |-- destroys _NEURON_* GroupCoordinators
            |-- restores original destroy_model_parallel
```

______________________________________________________________________

## 3. Current Neuron-Specific Parallelisms

### Expert Parallelism (EP)

Distributes MoE experts across contiguous rank partitions. Each partition forms its own TP sub-group for intra-expert collectives.

**Example**: WS=8, EP=2

```text
EP group 0 TP: [0,1,2,3]
EP group 1 TP: [4,5,6,7]
EP groups (columns): [0,4], [1,5], [2,6], [3,7]
```

**GroupCoordinators created**: `_NEURON_EP_TP`, `_NEURON_EP`

______________________________________________________________________

## 4. How to Add a New Parallelism Dimension

Adding a new parallelism (e.g., a "Foo Parallelism" with degree `foo_degree`) involves these steps:

### 4.1. Define the group topology

Add a `_build_foo_group_ranks(...)` function in `neuron_parallel_state.py` that computes the list of rank lists for each group. Follow the existing `_build_ep_group_ranks` pattern:

```python
def _build_foo_group_ranks(
    world_size: int,
    foo_degree: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Compute Foo group rank lists. Returns (foo_tp_groups, foo_groups)."""
    ...
```

### 4.2. Create GroupCoordinators

Add the creation logic to `_create_neuron_groups()`:

```python
_NEURON_FOO: GroupCoordinator | None = None
_NEURON_FOO_TP: GroupCoordinator | None = None

# Inside _create_neuron_groups():
if foo_degree > 1:
    foo_tp_group_ranks, foo_group_ranks = _build_foo_group_ranks(...)
    _NEURON_FOO_TP = init_model_parallel_group(
        foo_tp_group_ranks, local_rank, backend, group_name="neuron_foo_tp",
    )
    _NEURON_FOO = init_model_parallel_group(
        foo_group_ranks, local_rank, backend, group_name="neuron_foo",
    )
```

### 4.3. Add getters

Add getter functions following the existing pattern:

```python
def get_neuron_foo_group() -> GroupCoordinator: ...
def get_neuron_foo_tp_group() -> GroupCoordinator: ...
def get_neuron_foo_degree() -> int: ...
def get_neuron_foo_rank() -> int: ...
```

Register them in `_patch_getters()`.

### 4.4. Wire into destroy

Add the new coordinators to `destroy_neuron_parallel_state()` and `is_initialized()`.

### 4.5. Accept config

Add `foo_degree` parameter to `init_neuron_distributed_environment()` and `initialize_neuron_parallel_state()`. Read it from `neuron_config` in the worker and pass through.

### 4.6. Use in models

Models access groups from parallel state — never create their own:

```python
from vllm.distributed.parallel_state import get_tp_group
from vllm_neuron.parallel.neuron_parallel_state import (
    get_neuron_foo_group,
    get_neuron_foo_tp_group,
)

class MyModule(nn.Module):
    def __init__(self, ...):
        self.tp_group = get_tp_group()  # GroupCoordinator
        self.foo_group = get_neuron_foo_tp_group()  # GroupCoordinator

    def forward(self, x):
        # Use GroupCoordinator methods for collectives
        x = self.tp_group.all_gather(x, dim=0)
        x = self.foo_group.reduce_scatter(x, dim=0)
        # Use .device_group only when a raw ProcessGroup is needed
        indices = get_group_slice_indices(..., process_group=self.foo_group.device_group)
```

### 4.7. Update MPExecutor

Add the new degree parameter to `MPExecutor.__init__` and `worker_process` so tests can exercise the new parallelism.

______________________________________________________________________

## 5. Conventions

- **GroupCoordinator, not raw ProcessGroup**: All groups are `GroupCoordinator` instances. Models store the coordinator and use its methods (`all_gather`, `reduce_scatter`, `all_reduce`). Extract `.device_group` only when an API requires a raw `ProcessGroup`.
- **Parallel state is the single owner**: Models never call `dist.new_group()`. All groups are created in `neuron_parallel_state.py`.
- **vLLM's groups accessed via vLLM's API**: Use `get_tp_group()`, `get_world_group()`, etc. from `vllm.distributed.parallel_state`. Don't store `dist.group.WORLD`.
- **Config stores degrees, not groups**: `NeuronConfig` holds `ep_degree`, etc. (integers). Process groups live in parallel state, not config.

______________________________________________________________________

## 6. Path to Upstream Contribution

This module exists because vLLM's `parallel_state` doesn't currently support registering custom parallelism dimensions from OOT (out-of-tree) platform plugins. Our approach is a temporary bridge:

**What we do today:**

- Monkey-patch `destroy_model_parallel` to include Neuron group teardown
- Add `get_neuron_*()` getters to vLLM's `parallel_state` module at runtime
- Bootstrap vLLM's `_WORLD` / `_TP` directly for the test path

**What we want upstream:**

- A `register_parallel_group(name, group_ranks, ...)` API in vLLM that lets OOT platforms define custom parallelism dimensions without monkey-patching
- Lifecycle hooks in `initialize_model_parallel` / `destroy_model_parallel` for platform-specific group creation/teardown
- A `get_parallel_group(name)` generic getter so platforms don't need to patch the module namespace

**Migration path:**

Once vLLM provides these APIs, we would:

1. Replace `_patch_getters()` / `_patch_destroy()` with `register_parallel_group()`
2. Replace `_ensure_vllm_parallel_state()` with proper OOT platform init hooks
3. Our `init_neuron_distributed_environment()` becomes a thin wrapper that calls vLLM's init + registers Neuron groups via the public API
4. `get_neuron_ep_group()` becomes `get_parallel_group("neuron_ep")`
