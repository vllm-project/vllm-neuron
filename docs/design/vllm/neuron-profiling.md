# Runtime NRT Profiling via vLLM's Profiler Interface

<!-- meta: description: Neuron profiling integration -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

## Overview

This design integrates Neuron Runtime (NRT) profiling with vLLM's existing
`/start_profile` and `/stop_profile` HTTP endpoints, allowing users to toggle
hardware-level profiling on a live serving instance without restarting.

## NRT API Choice

We use `nrt_inspect_begin_with_options` / `nrt_inspect_stop`. This API provides:

- Unified configuration for both device and system profiles via
  `nrt_inspect_config_set_activity`
- Per-NeuronCore control via `nrt_inspect_config_set_capture_enabled_for_nc`
- No model handle required — works at the runtime level
- Compatible with async execution (`nrta_execute_schedule`)

Device profile mode is set to SESSION (`NRT_INSPECT_DEVICE_PROFILE_MODE_SESSION`),
which captures all activity on each NeuronCore into a single NTFF per core.
MODEL mode is not compatible with async execution.

## Configuration

### vLLM profiler config

We reuse `"cuda"` as the profiler kind to mount the `/start_profile` and
`/stop_profile` endpoints (vLLM's `ProfilerKind` is `Literal["torch", "cuda"]`
and cannot be extended by a plugin):

```bash
--profiler-config '{"profiler": "cuda"}'
```

### Neuron-specific config via `--additional-config`

```bash
vllm serve <model> \
  --profiler-config '{"profiler": "cuda"}' \
  --additional-config '{
    "neuron_profiler": {
      "activities": ["device_profile", "system_profile"],
      "neuron_cores": [0, 1, 2, 3],
      "output_dir": "/tmp/nrt_profile"
    }
  }'
```

| Field | Type | Default | Description |
|---|---|---|---|
| `activities` | list[str] | `["device_profile", "system_profile"]` | Activity types to capture. Valid values: `"system_profile"`, `"device_profile"`, `"host_memory"`, `"cpu_util"`, `"all"` |
| `neuron_cores` | list[int] \| null | `null` (rank 0 only) | Which worker ranks to profile. When null, only rank 0 profiles. Each rank has one NeuronCore. |
| `output_dir` | str | `./neuron_profiles` | Output directory for profile data |
| `sys_trace_max_events_per_nc` | int \| null | `null` (NRT default) | Max number of system trace events stored across all ring buffers per NeuronCore. Increase for long profiling sessions to avoid event loss |

## vLLM Profiler Architecture

```text
HTTP POST /start_profile
    → AsyncLLM.start_profile()
    → EngineCore.profile(is_start=True)
    → ModelExecutor.profile(is_start=True)
    → collective_rpc("profile", args=(True, prefix))
    → NeuronWorker.profile(is_start=True)
    → NeuronProfiler.start()
    → torch.classes.neuron.Runtime().start_profiling(...)
```

The `/start_profile` and `/stop_profile` routes are only mounted when
`--profiler-config.profiler` is set at server startup.

## Design

### Components

#### `NeuronProfiler` (`vllm_neuron/vllm/worker/neuron_profiler.py`)

Subclasses `vllm.profiler.wrapper.WorkerProfiler` to implement `_start()` and
`_stop()` methods that call NRT profiling bindings exposed by
libtorch-neuron-lite via `torch.classes.neuron.Runtime`.

#### `NeuronWorker` (`vllm_neuron/vllm/worker/neuron_worker.py`)

- `_init_profiler(vllm_config)`: Creates `NeuronProfiler` if profiling is
  enabled and this rank should profile (based on `neuron_cores` config).
- `profile(is_start, ...)`: Delegates to `NeuronProfiler.start()`/`stop()`.
  No-op on ranks that don't have a profiler.
- `execute_model(...)`: Calls `self._profiler.step()` each iteration to drive
  the `delay_iterations` / `max_iterations` state machine.

#### libtorch-neuron-lite C++ bindings (`csrc/neuron_op/runtime.cpp`)

Exposes `start_profiling` and `stop_profiling` on `torch.classes.neuron.Runtime`:

```python
runtime = torch.classes.neuron.Runtime()
runtime.start_profiling(
    output_dir,           # str
    activities,           # list[str]
    neuron_cores,         # Optional[list[int]]
    sys_trace_max_events, # Optional[int]
    neff_cache_dir,       # Optional[str]
)
runtime.stop_profiling()
```

The `neff_cache_dir` parameter tells NRT where to find compiled NEFFs so they
can be copied alongside the NTFF traces. This enables Neuron Explorer to
correlate device profiles with NEFF instruction data.

### Per-rank profiling

Each `NeuronWorker` corresponds to one rank with one NeuronCore. The profiler
is only created on ranks that should participate in profiling. By default,
only rank 0 profiles to minimize overhead. Users can specify
`"neuron_cores": [0, 1, 2, 3]` to profile multiple ranks.

### Iteration control

The `WorkerProfiler` base class handles `delay_iterations` and
`max_iterations` via its `step()` state machine:

| Config field | Behavior | Supported |
|---|---|---|
| `delay_iterations` | Skip N engine steps before calling `nrt_inspect_begin_with_options` | Yes |
| `max_iterations` | Auto-call `nrt_inspect_stop` after N engine steps | Yes |
| `warmup_iterations` | PyTorch profiler schedule | N/A |
| `wait_iterations` | PyTorch profiler schedule | N/A |
| `active_iterations` | PyTorch profiler schedule | N/A |

One iteration = one `execute_model` call = one batched forward pass.

## Usage

### Basic usage

```bash
vllm serve <model> \
  --profiler-config '{"profiler": "cuda"}' \
  --additional-config '{"neuron_profiler": {"output_dir": "/tmp/nrt_profile"}}'

curl -X POST http://localhost:8000/start_profile
# ... send requests ...
curl -X POST http://localhost:8000/stop_profile
```

### Device profile only, specific ranks

```bash
vllm serve <model> \
  --profiler-config '{"profiler": "cuda"}' \
  --additional-config '{
    "neuron_profiler": {
      "activities": ["device_profile"],
      "neuron_cores": [0, 1],
      "output_dir": "/tmp/nrt_device"
    }
  }'
```

### Profiling steady-state (skip warmup)

```bash
vllm serve <model> \
  --profiler-config '{"profiler": "cuda", "delay_iterations": 50, "max_iterations": 20}' \
  --additional-config '{"neuron_profiler": {"output_dir": "/tmp/nrt_profile"}}'

curl -X POST http://localhost:8000/start_profile
# Auto-starts after 50 iterations, auto-stops after 20 more
```

### Using vllm bench

```bash
vllm bench serve \
    --backend vllm \
    --model <model> \
    --dataset-name sharegpt \
    --dataset-path sharegpt.json \
    --profile \
    --num-prompts 5
```

## Disaggregated Inference (DI) Support

In a disaggregated inference setup, prefill and decode run as separate vLLM
server instances (potentially on different hosts). Each instance has its own
`/start_profile` and `/stop_profile` endpoints — there is no built-in mechanism
in vLLM to atomically profile across both.

### Proxy server pass-through

The DI proxy server sits in front of the prefill and decode servers,
routing requests to the appropriate backend. We extend the toy proxy server to
forward `/start_profile` and `/stop_profile` to all backend servers:

```text
Client                    Proxy                  Prefill Server    Decode Server
  │                         │                         │                 │
  ├─ POST /start_profile ──→│                         │                 │
  │                         ├─ POST /start_profile ──→│                 │
  │                         ├─ POST /start_profile ──────────────────→│
  │                         │                         │                 │
  │              (200 OK after all backends respond)  │                 │
  │←────────────────────────┤                         │                 │
  │                         │                         │                 │
  │         ... requests profiled on both servers ... │                 │
  │                         │                         │                 │
  ├─ POST /stop_profile ───→│                         │                 │
  │                         ├─ POST /stop_profile ───→│                 │
  │                         ├─ POST /stop_profile ────────────────────→│
  │                         │                         │                 │
  │←────────────────────────┤                         │                 │
```

The proxy fans out the profile request to all backends in parallel and returns
success only when all have responded. Each server stores its profiles locally.

Other production deployment libraries may offer similar proxy server features
to pass-through forward `/start_profile` and `/stop_profile` to all backend
servers. Otherwise, users can manually hit these endpoints on each DI server
that they want to profile.

## Profile Output

After `/stop_profile`, the output directory contains:

```text
output_dir/
  i-<instance_id>_pid_<pid>/<timestamp>/
    profile_nc_0_session_0.ntff    # Device profile (NTFF)
    ntrace.pb                      # System trace
    trace_info.pb                  # Trace metadata
    cpu_util.pb                    # CPU utilization (if enabled)
    host_mem.pb                    # Host memory (if enabled)
  neffs/
    graph_<hash1>.neff             # Compiled NEFFs for Neuron Explorer
    graph_<hash2>.neff
```

NTFF files can be viewed with Neuron Explorer. NEFFs are required for Neuron
Explorer to render device profiles — they are automatically copied from the
compile cache.
