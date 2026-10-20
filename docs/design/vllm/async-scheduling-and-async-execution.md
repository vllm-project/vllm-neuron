# Async Scheduling and Async Execution

<!-- meta: description: Async scheduling and execution design -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

## Table of Contents

- [Overview](#overview)
- [When Does Async Scheduling Help Most?](#when-does-async-scheduling-help-most)
- [Custom Async Execution Integration with Async Scheduling](#custom-async-execution-integration-with-async-scheduling)
- [Normal Flow (Synchronous Baseline)](#normal-flow-synchronous-baseline)
- [Async Flow (Steady-State)](#async-flow-steady-state)
- [What Does "Breaking the Async Flow" Mean?](#what-does-breaking-the-async-flow-mean)
- [How Do We Break the Async Flow Safely?](#how-do-we-break-the-async-flow-safely)
- [Development and Debugging](#development-and-debugging)

## Overview

Async scheduling overlaps CPU work (scheduling, input preparation, output
processing) with Neuron device execution (NEFF forward pass) to reduce
per-step latency. In synchronous mode, the CPU idles while the device runs
the forward pass, and the device idles while the CPU prepares the next step.
Async scheduling eliminates this idle time by:

1. Using a batch queue (depth 2) in the engine core to schedule one step
   ahead, so the worker always has work ready.
2. Feeding the previous step's sampled token future directly as `input_ids`
   into the next forward pass (device-to-device, no CPU roundtrip), using
   the Neuron async runtime to enqueue NEFF executions ahead of time.
3. Materializing tokens to CPU on a separate async output thread, overlapped
   with the engine core scheduling the next step.

When the batch composition changes (requests finish, new requests join), or
when the previous sampled-token future cannot be reused for the next decode
input shape, the async flow must be "broken" — the pending device future is
materialized on the critical path, batch state is updated, and CPU-built
`input_ids` is used for that step. If the request set is unchanged but the
batch order changed, the future can be remapped into the new order; because
the remap clones/forces the future on the worker main thread, that step is
accounted as a sync fallback. The async flow automatically reinstates on the
next steady-state decode step.

Async scheduling is on by default and requires on-device sampling. To
disable, use `--no-async-scheduling`.

## When Does Async Scheduling Help Most?

All workloads benefit from async scheduling, but these scenarios see the
largest gains:

- **Smaller models (faster NEFF execution)**: The CPU overhead (scheduling,
  input preparation, output processing) is roughly constant regardless of
  model size. For smaller models where the NEFF forward pass is fast, the
  CPU overhead becomes a larger fraction of the total step time. Async
  scheduling hides this overhead by overlapping it with device execution,
  so the relative speedup is greater.

- **Longer average response length**: Async scheduling only engages during
  steady-state decode (consecutive decode steps with no batch composition
  change). Longer responses mean more consecutive decode steps, so the
  async path is active for a larger fraction of the total generation.
  Short responses that finish quickly cause frequent batch composition
  changes, forcing sync fallbacks that reduce the benefit.

## Custom Async Execution Integration with Async Scheduling

The upstream vLLM GPU implementation uses CUDA streams for async execution —
the GPU runtime provides stream semantics where operations on the same stream
are ordered and the CPU can enqueue work without blocking. The CPU simply
enqueues kernels and continues.

The Neuron runtime does not currently provide `torch.cuda.Stream`-style
semantics. Once native PyTorch support for Neuron is available, it will
provide stream semantics and this custom implementation can be simplified.
Until then, the runtime offers an async execution mode where:

- A NEFF (compiled model) can be enqueued for execution with device tensor
  futures as inputs. The runtime guarantees the NEFF will wait for the
  input futures to resolve before executing.
- Output tensors are also futures — accessing them (e.g., `.to("cpu")`)
  blocks until the NEFF completes.

This means we must explicitly manage the future lifecycle:

1. **Hold futures across steps**: Store the sampled token future in
   `async_execution_buffer` and pass it as `input_ids` to the next NEFF
   execution, avoiding the device→host→device roundtrip.
2. **Intentionally break the async flow**: When the batch composition
   changes, or when a shape/dtype mismatch forces CPU-built inputs, we must
   materialize the future on the critical path (blocking `.to("cpu")`) to
   update CPU-side state before proceeding. The GPU implementation handles
   this implicitly via stream synchronization; we must do it explicitly.
   When the request set is unchanged but the request order changes, we can
   remap the future on-device, but the remap still forces the future and is
   counted as a sync fallback step.
3. **Thread-safe materialization**: Both the async output thread and the
   worker main thread may call `get_output()` on the same future. A
   per-object lock prevents double-materialization.

## Normal Flow (Synchronous Baseline)

In synchronous mode (`--no-async-scheduling`), each inference step runs as a
strictly sequential pipeline. The engine core orchestrates the flow:

1. **Scheduler** selects which requests to run and produces a
   `SchedulerOutput`.
2. **Engine core** calls `worker.execute_model(scheduler_output)`.
3. Inside the worker, `NeuronModelRunner.execute_model()` does the heavy
   lifting:

   a. `_update_states()` — applies the scheduler's decisions (add/remove
      requests, update token counts, manage sequence IDs).

   b. `_prepare_model_input()` — builds `input_ids`, `positions`,
      `logits_indices`, and `attn_metadata` from the current
      `InputBatch` state.

   c. `_execute_model_forward()` — moves tensors to the Neuron device,
      runs `model(**kwargs)`, and **blocks** on
      `model_output_tensor.to("cpu")` to copy the result back to host
      memory.

   d. Stores the result in `execute_model_state` and returns `None`.

4. Engine core sees `None` and calls `worker.sample_tokens(grammar_output)`.
5. `NeuronModelRunner.sample_tokens()` unpacks the stored state, runs the
   sampler (`_sample()`), updates batch state with the sampled tokens
   (`_update_batch_state_with_samples()`), and returns a
   `ModelRunnerOutput`.
6. Engine core passes the output to `scheduler.update_from_output()` which
   processes finished requests, frees blocks, and prepares for the next step.

The critical property of this flow is that **every tensor transfer is
blocking**: the CPU waits for the device-to-host copy of logits/sampled tokens
to complete before proceeding. This means the CPU is idle during the entire
model forward pass, and the device is idle during scheduling and input
preparation.

```text
Step N                                          Step N+1
─────────────────────────────────────────────    ──────────────────────

┌─────────────────────────────────────────────────────────────────────┐
│ Engine Core                                                        │
│                                                                    │
│  schedule()  ──►  execute_model()  ──►  sample_tokens()  ──►  ...  │
│      │                  │                     │                     │
└──────┼──────────────────┼─────────────────────┼─────────────────────┘
       │                  │                     │
       ▼                  ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ NeuronModelRunner                                                  │
│                                                                    │
│  ┌──────────────┐  ┌──────────────────────────────┐  ┌──────────┐  │
│  │ _update_     │  │ _execute_model_forward()     │  │ _sample()│  │
│  │  states()    │  │                              │  │          │  │
│  │              │  │  ┌────────────────────────┐  │  │          │  │
│  │ _prepare_    │  │  │ model(**kwargs)        │  │  │          │  │
│  │  model_      │  │  │   (Neuron device)      │  │  │          │  │
│  │  input()     │  │  └────────────────────────┘  │  │          │  │
│  │              │  │  output.to("cpu")  ◄─ BLOCK  │  │          │  │
│  └──────────────┘  └──────────────────────────────┘  └──────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

Timeline (single step):

CPU:   ┃ schedule │ prepare_input │░░░░░ IDLE ░░░░░│ sample │ update ┃
Device:┃░░░░░░░░░░░░░░░░░░░░░░░░░│ model forward  │░░░░░░░░░░░░░░░░┃
                                   ▲               ▲
                                   │               │
                               .to(device)    .to("cpu") ← blocking
```

The `░░░░░ IDLE` regions are the latency cost of synchronous execution:
the CPU stalls while the device runs the forward pass, and the device sits
idle while the CPU schedules and prepares the next step. Async scheduling
exists to overlap these two phases.

## Async Flow (Steady-State)

Steady-state means consecutive **decode** steps where the batch composition
does not change — no requests finish, no new requests join. When the batch
changes, we leave steady-state and enter the "breaking" path described in
later sections.

### Key Idea

In synchronous mode the CPU blocks on `.to("cpu")` to copy sampled tokens
back from the device before the next step can begin. Async scheduling
eliminates that blocking transfer from the critical path by:

1. Keeping the sampled token tensor on the device and feeding it directly
   as `input_ids` into the next forward pass (device-to-device, no CPU
   roundtrip). This is made possible by the Neuron async runtime: we can
   pass the future of a device tensor to a NEFF and enqueue the NEFF
   execution ahead of time. The runtime guarantees that the NEFF will wait
   for the tensor to be ready before executing.
2. Materializing the tokens to CPU on a **separate thread** — the multiproc
   executor's `async_output_busy_loop` thread calls
   `AsyncNeuronModelRunnerOutput.get_output()`, which performs the
   `.to("cpu")` transfer and updates batch state. This runs concurrently
   with the engine core's main thread scheduling the next step.

So the CPU does eventually see every token — but the materialization happens
off the critical path, overlapped with scheduling and input preparation for
the next step. This also guarantees that output tokens are streamed back to
the client: the async output thread materializes and enqueues each step's
tokens as soon as they're ready, so the client receives tokens incrementally
rather than waiting for the full sequence to complete.

### Step-by-Step Flow

Consider two consecutive decode steps, Step N and Step N+1, with the same
batch of requests.

**Step N (produces futures):**

1. `execute_model()` calls `_update_states()` and `_prepare_model_input()`
   as usual. On the very first async step, `input_ids` comes from CPU
   (there are no futures yet).
2. `_execute_model_forward()` runs `model(**kwargs)` on the Neuron device.
   The model returns `sampled_token_ids` as a device tensor future.
3. Unlike sync mode, the `.to("cpu")` call is **skipped** — the output
   stays on device as a future.
4. `sample_tokens()` takes a snapshot of `req_ids` via `_bookkeeping_sync()`
   (so the next step's `_update_states()` can freely mutate `InputBatch`
   without corrupting the current step's output mapping), runs `_sample()`,
   and calls `_generate_model_runner_output()`.
   Note: there is no race condition here — `execute_model()` and
   `sample_tokens()` are dispatched sequentially on the same worker
   process, so the next step's `_update_states()` is guaranteed not to
   run before the current step's `_bookkeeping_sync()` completes.
5. `_generate_model_runner_output()` stores the device-side sampled token
   future into `async_execution_buffer["futures_sampled_token_ids"]` and
   wraps the output in an `AsyncNeuronModelRunnerOutput`.
6. The `AsyncNeuronModelRunnerOutput` is returned to the engine core. In
   the multiproc executor, `handle_output()` puts it on the
   `async_output_queue`. A separate `async_output_busy_loop` thread picks
   it up and calls `get_output()`, which blocks on `.to("cpu")` and writes
   the sampled tokens into `token_ids_cpu` and `req_state.output_token_ids`
   via `_update_batch_state_with_samples()`. This materialization runs
   **concurrently** with the engine core scheduling Step N+1.
7. With async scheduling, the engine core uses `step_with_batch_queue()`
   with a batch queue of size 2 (`max_concurrent_batches`). The engine
   core is always one step ahead: by the time Step N's `get_output()`
   materializes, Step N+1 has already been scheduled and enqueued. Once
   Step N's output is processed, the engine core schedules Step N+2 and
   then blocks waiting for Step N+1's output to materialize.

**Step N+1 (consumes futures):**

1. `execute_model()` runs `_update_states()` and `_prepare_model_input()`.
   `_prepare_model_input()` builds `input_ids` from `token_ids_cpu` as
   usual — but these are **stale** (they don't contain Step N's sampled
   tokens, which are still on device).
2. `execute_model()` checks whether the async swap is eligible: the buffer
   is populated, this is a decode step, and the batch composition did not
   change.
3. It reads `async_execution_buffer["futures_sampled_token_ids"]` — the
   device tensor future from Step N — and checks that its shape and dtype
   match the current padded decode input.
4. Since the batch hasn't changed, the shape and dtype match. If the request
   order is unchanged, the stale CPU-built `input_ids` is **replaced** with
   the device-side future.
5. If the request set is the same but `condense()` changed slot order, the
   future is remapped into the current request order before it is used. This
   produces correct `input_ids`, but the clone/remap forces the future on the
   worker main thread, so this step is counted as sync fallback rather than
   steady-state async.
6. The future (still on device) is passed directly to
   `model(**kwargs)` — no device→host→device roundtrip.
7. The cycle repeats: new sampled token futures stay on device, get stored
   in the buffer, and feed into Step N+2.

### Where the Overlap Happens

While the device is executing Step N's forward pass, the CPU is free to run
the scheduler for Step N+1, call `_update_states()`, and prepare inputs.
The CPU-built `input_ids` will be thrown away in favor of the device future,
but the scheduling and state bookkeeping work is done in parallel with the
device computation.

```text
Steady-state async (batch unchanged):

              Step N                              Step N+1
              ───────────────────────────         ───────────────────────────

Worker        ┃ execute_model(N) │ sample(N) │    │ execute_model(N+1) │ sample(N+1) │
main          ┃ (input_ids =     │           │    │ (input_ids =        │             │
thread:       ┃  future(N-1))    │           │    │  future(N))         │             │
              ┃       │          │           │    │       │              │             │
Device:       ┃ ══ forward(N) ═══════════════╪════╪══ forward(N+1) ══════════════════ ...
              ┃    waits for ◄───────┐       │    │    waits for ◄───────┐
              ┃    future(N-1)       │       │    │    future(N)         │
              ┃                      │       │    │                      │
              ┃                      │       │    │                      │
Worker        ┃    get_output(N-1)   │       │    │    get_output(N)     │
async         ┃    waits for ◄───────┘       │    │    waits for ◄───────┘
thread:       ┃    future(N-1)               │    │    future(N)
              ┃    .to("cpu") + update_batch │    │    .to("cpu") + update_batch
              ┃                              │    │
Engine        ┃    ◄── block ──► update(N-1) │    │    ◄── block ──► update(N)
core:         ┃                  schedule(N+1)    │                   schedule(N+2)

Both the device (NEFF execution) and the async thread (.to("cpu"))
wait on the same future — the sampled token future from the previous step.
```

### The `async_execution_buffer`

This dict is the bridge between consecutive async steps. In steady-state,
only `futures_sampled_token_ids` is consumed:

| Key | Value | Purpose |
|-----|-------|---------|
| `futures_sampled_token_ids` | Device tensor future `[num_reqs]` | Sampled token future from the previous step, fed as `input_ids` into the next forward pass |
| `prev_req_ids_ordered` | Ordered `list[str]` of request IDs | Request order that produced the future. Its set is compared against the next step's scheduled requests to detect composition changes; its ordering is used to remap same-set reordered futures. |

The buffer also stores `async_output` (`AsyncNeuronModelRunnerOutput`), but
this is only accessed when breaking the async flow — see
[How Do We Break the Async Flow Safely?](#how-do-we-break-the-async-flow-safely).

## What Does "Breaking the Async Flow" Mean?

"Breaking the async flow" means forcing the materialization of a previous
step's device tensor future on the **worker main thread** (the critical path),
rather than letting it be consumed only by the async output thread and the
next NEFF execution.

### Why Does It Happen?

Consider the steady-state async flow: step N is a decode step, and step N+1
has already been scheduled ahead of time (batch queue depth = 2). During step
N, one or more requests complete (EOS or max output tokens). We only discover
this after step N's output is materialized — when the engine core's
`scheduler.update_from_output(N)` checks stop conditions and marks the
request as finished. At that point, step N+1 is already in flight on the
worker.

After processing step N's output, the engine core schedules step N+2. This
time, the `SchedulerOutput` reflects the updated state: the finished request
is gone, and the scheduler may have added a new prefill, filled the decode
batch to max batch size, or simply continued decode with fewer requests. The
batch composition has changed.

### Why Can't We Stay Async?

In steady-state, the worker discards the CPU-built `input_ids` and replaces
them with the device future from the previous step. For step N+2, we cannot
always do that:

- If the request set changed, the scheduler's `input_ids` must be respected.
  Step N+2's `input_ids` come from the scheduler's updated view of the world
  (with the finished request removed, possibly a new prefill added, etc.).
  We must use them as-is, not replace them with a stale device future.
- If the request set is unchanged but the padded input shape or dtype differs,
  the future cannot be passed to the next NEFF input. We must materialize the
  pending output, rebuild CPU inputs from refreshed state, and use those
  rebuilt inputs.
- If the request set is unchanged but the request order changed, the future
  has the right shape but wrong slot ownership. We remap it before use; this
  keeps inputs correct but forces the future, so it is not counted as a
  steady-state async step.

For composition changes and shape/dtype mismatches, "breaking" means:
materialize step N+1's future on the main thread (`get_output()`), update all
batch state, rebuild `input_ids` from the now-correct CPU state, and proceed
synchronously for this step. For same-set order changes, the worker remaps the
future into the current order; this does not rebuild CPU inputs, but it still
forces the future on the main thread and is accounted as sync fallback. The
async flow can resume once the batch stabilizes again.

## How Do We Break the Async Flow Safely?

Breaking the async flow happens on the worker main thread, on the critical
path of step N+2's execution. After breaking, the async flow is automatically
reinstated at the end of the same step.

Note: the scheduler always schedules one step ahead — it is unaware of the
async/sync distinction. It is the worker (specifically `NeuronModelRunner`)
that detects when the async flow must be broken and is responsible for
materializing futures, adjusting batch state, and reinstating the async
pipeline.

### Pre-`_update_states` Guard in `execute_model()`

This is the first check, before any state mutation happens for step N+2.

```python
# In execute_model(), before _update_states():
if self.use_async_scheduling:
    if not self.async_execution_buffer:
        self._batch_composition_changed = False
    else:
        curr_req_ids = frozenset(scheduler_output.num_scheduled_tokens.keys())
        prev_req_ids_ordered = self.async_execution_buffer.get(
            "prev_req_ids_ordered"
        )
        prev_req_ids = (
            frozenset(prev_req_ids_ordered)
            if prev_req_ids_ordered is not None
            else None
        )
        composition_changed = (
            prev_req_ids is None or curr_req_ids != prev_req_ids
        )
        if composition_changed:
            self._materialize_pending_async_output("batch composition changed")
            self._batch_composition_changed = True
        else:
            self._batch_composition_changed = False
```

What it does:

- Detects batch composition changes by comparing the current step's
  scheduled request IDs (`curr_req_ids`) against the previous step's
  (`prev_req_ids`, derived from `prev_req_ids_ordered`, which is stored in
  the buffer when the future was produced).
  If the sets differ for ANY reason — requests finished, new requests
  joined, requests became unscheduled — it's a composition change.
  This single set comparison is simpler and more robust than checking
  individual signals (`finished_req_ids`, `scheduled_new_reqs`, etc.)
  because it catches all cases including the max_tokens pipelining gap
  where a request disappears from `num_scheduled_tokens` one step before
  appearing in `finished_req_ids`.
- If composition changed, it calls `get_output()` on the previous step's
  `AsyncNeuronModelRunnerOutput`, which:
  - Blocks until the device tensor future is ready
  - Copies sampled tokens to CPU via `.to("cpu")`
  - Writes them into `token_ids_cpu` and `req_state.output_token_ids` via
    `_update_batch_state_with_samples()`
- Sets `_batch_composition_changed = True` so that the swap logic later
  knows not to replace `input_ids` with the stale device future.
- This ensures `_update_states()` (called immediately after) sees correct
  token data when it calls `input_batch.add_request()` for resumed/new
  requests — `add_request` copies from `req_state.output_token_ids` into
  `token_ids_cpu`, so the data must be up to date.
- After materialization, `_update_states()` and `_prepare_model_input()` run
  against correct state, so `input_ids` is built correctly from CPU and
  does not need rebuilding.

The `prev_req_ids_ordered` list is stored in `_generate_model_runner_output()`
as `list(req_ids_output_copy)` — the exact order of request IDs that
produced the future. The same list supports both composition detection (via
set comparison) and same-set reorder remapping (via index lookup).

### Async/Sync Swap Decision in `execute_model()`

After `_update_states()` and `_prepare_model_input()` have run, the swap
logic decides whether to use the device future or the CPU-built `input_ids`:

```python
# In execute_model(), after _prepare_model_input():
if self.use_async_scheduling:
    if (
        self.async_execution_buffer
        and self._is_decode()
        and not self._batch_composition_changed
    ):
        future = self.async_execution_buffer["futures_sampled_token_ids"]
        if future.shape[0] == input_ids.shape[0] and future.dtype == input_ids.dtype:
            input_ids, broke_async = self._maybe_remap_async_future(
                future, input_ids
            )
            if broke_async:
                self._sync_fallback_steps += 1
            else:
                self._async_steps += 1
        else:
            # Shape/dtype mismatch: materialize pending output, rebuild inputs,
            # and use refreshed CPU-built input_ids.
            if self._materialize_pending_async_output(
                "async future shape/dtype mismatch"
            ):
                input_ids, positions, logits_indices, attn_metadata, spec_decode_metadata = (
                    self._prepare_model_input(scheduler_output)
                )
            self._sync_fallback_steps += 1
    else:
        # Broken: use CPU-built input_ids as-is.
        self._sync_fallback_steps += 1
```

The async path is used only when all four conditions are met:

- `async_execution_buffer` is non-empty (not the first iteration)
- `_is_decode()` returns `True` (all requests are in decode phase — note:
  this must be checked after `_update_states()` so it reflects the current
  step's batch, not the previous one)
- `_batch_composition_changed` is `False` (no requests finished/added/resumed)
- The future shape and dtype match `input_ids` — `_prepare_model_input()`
  applies decode batch padding (e.g., 1 request padded to bucket size 4),
  and the future must match this padded shape. On the first decode step
  after a prefill, the future has shape `[num_reqs]` (from the prefill
  output) while `input_ids` has shape `[padded_bucket_size]`, causing a
  mismatch and a one-step sync fallback. The fallback first materializes the
  pending async output, then rebuilds model inputs from refreshed CPU state.
  This prevents stale CPU `input_ids` from being used after the latest
  sampled token still lived only in a device future. From the second decode
  step onward, the model output (and thus the future) already has the padded
  shape, so the shapes match and async resumes.
- If the request set is unchanged but the current batch order differs from
  `prev_req_ids_ordered`, `_maybe_remap_async_future()` remaps real request
  slots into the current order and leaves padding slots unchanged. This keeps
  token ownership correct after `condense()` reorder. Because remap uses
  `clone()`/gather and forces the future on the main thread, the step is
  counted as `_sync_fallback_steps`, not `_async_steps`.

When any condition fails, `input_ids` from `_prepare_model_input()` is used
as-is — it was already built from correct CPU state (either because the
pre-`_update_states` guard materialized, because the shape/dtype mismatch
path materialized and rebuilt it, or because this is a prefill/first iteration
where no stale futures exist).

### Reinstatement

The async flow is automatically reinstated at the end of every step,
regardless of whether it was broken:

1. `_execute_model_forward()` always skips `.to("cpu")` when
   `use_async_scheduling` is True — the model output stays on device as a
   future.
2. `_generate_model_runner_output()` always stores
   `sampler_output.sampled_token_ids` (the new device future) into
   `async_execution_buffer["futures_sampled_token_ids"]`.
3. `sample_tokens()` always wraps the output in
   `AsyncNeuronModelRunnerOutput` and stores it in
   `async_execution_buffer["async_output"]`.

So even after a sync fallback step, the buffer is repopulated with fresh
futures. The next step will use the async path if it's a decode step with
no composition change and matching shapes, and the async flow resumes.

## Development and Debugging

### `get_output()` Call Patterns Across TP Ranks

Only the output rank (TP0) has its `AsyncNeuronModelRunnerOutput` processed
by the async output thread. This is because the multiproc executor's
`worker_busy_loop` only calls `handle_output(output)` when
`self.rank == output_rank` — non-output ranks discard their output.

This means:

- **TP0**: `get_output()` is called on every step by the async output thread
  (via `enqueue_output`). Additionally, the worker main thread calls it
  when breaking the async flow. With the threading lock, only one thread
  materializes; the other hits the "already actualized" else branch.

- **TP1–TP7**: `get_output()` is only called by the worker main thread when
  breaking the async flow (the async output thread never sees their
  objects). In steady-state, their `AsyncNeuronModelRunnerOutput` objects
  are created and stored in the buffer but never materialized via
  `get_output()` — they are simply overwritten by the next step's output.
