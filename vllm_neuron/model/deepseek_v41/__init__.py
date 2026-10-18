# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4.1-Flash on 16 Trainium 2 devices — package setup
============================================================

This port serves the checkpoint from **HBM**: 64 ranks, one per logical
NeuronCore of a trn2.48xlarge, with the routed experts partitioned by id and the
engram tables partitioned by row, so a forward pass reads no file. Its ancestor
(`vllm_neuron/model/deepseek_v41_timeshare`, still in-tree and still passing its
own verifier) ran 4 ranks on one device and streamed both families off disk every
forward; that is where most of the measurements quoted in these comments come
from, and each says so.

What carries over unchanged is the host-side accounting: a rank still pays host
memory per loaded NEFF, there are still ~120 of them, and there are now 16x as
many ranks paying it. What does not carry over is anything whose premise was the
disk.

Two jobs, both done at import time:

1. Export `DeepseekV41ForCausalLM` so `vllm_neuron.model.registry` can map the
   checkpoint's `architectures[0]` onto this port.
2. Teach `transformers` what `model_type: "deepseek_v41"` is.

Why (2) lives here, and why it has to:

vLLM builds `ModelConfig` *before* it ever consults
`vllm_neuron.model.registry`, and `ModelConfig` resolves the checkpoint through
`transformers.AutoConfig`. `deepseek_v41` is a brand-new architecture that
upstream `transformers` (5.19) has never seen, so that resolution raised

    ValueError: The checkpoint you are trying to load has model type
    `deepseek_v41` but Transformers does not recognize this architecture.

and the engine died in `vllm.engine.arg_utils.create_model_config`, before the
platform plugin, the registry, the model class, or any device existed. Every
downstream failure was this one.

The import ordering that makes registering here sufficient: the verifier runs as
`python -m vllm_neuron.model.deepseek_v41.verifier.run_verifier`, and
`runpy` imports each parent package before the target module — so this file
executes before `run_verifier.main()` reaches `from vllm import LLM`. In worker
subprocesses the same registration arrives via
`neuron_worker -> vllm_neuron.model.registry -> this package`, and the config
object itself unpickles by importing this module, which re-runs the
registration. Nothing has to be registered twice to be safe: `AutoConfig`'s
`exist_ok=True` makes it idempotent.

The three config classes below are *transformers* config objects — plumbing, so
vLLM can read a shape off the checkpoint. The port's own architectural config,
which is the one `check_config` holds against the official release, stays in
`config.py`; `DeepseekV41Config.from_configs` consumes either one.
"""

from __future__ import annotations

import logging
import os
import sys

from transformers import AutoConfig, PretrainedConfig

from . import config as _config
from .factory import DeepseekV41ForCausalLM

logger = logging.getLogger(__name__)

# This instance has no EFA adapter attached (`/sys/class/infiniband` is empty
# and `fi_info -p efa` finds none; the dev trn2.3xlarge had none either, for the
# different reason that the instance type has none), so
# `NeuronWorker._set_efa_affinity` cannot resolve
# /sys/bus/pci/devices/<bdf>/infiniband and every worker dies with
#
#     RuntimeError: No EFA device found at ... This is expected on instances
#     without EFA ... set NEURON_SKIP_EFA_AFFINITY=1 to skip it
#
# which is the remedy its own message names: EFA affinity pins worker threads
# to the NUMA node nearest the network adapter, a CPU placement optimization
# with no bearing on correctness or on which device the math runs. `setdefault`
# so an operator who exports it (either way) still wins, and set here rather
# than later because workers inherit this process's environment when they are
# spawned.
#
# It does not limit the 64-rank collectives either: every rank is on this one
# instance, so the all-reduces go over NeuronLink between chips, and EFA is the
# inter-node path this deployment never takes.
os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")

# A timeshared model's `forward` streams weights off disk between compute
# steps, so it is `@torch._dynamo.disable`d and the per-step compute is
# compiled underneath it (see `model.DeepseekV41ForCausalLM.forward`). Dynamo
# refuses to inline a disabled function under `fullgraph=True` —
#
#     Unsupported: Skip inlining torch.compiler.disable()d function
#
# and `NeuronModelRunner.load_model` sets `fullgraph = not
# VLLM_NEURON_DEBUG_MODE`, so this variable is how a model asks for the
# `fullgraph=False` outer compile it needs.
#
# **The name is misleading and worth stating plainly: it does not move anything
# to the host and it relaxes no numerics.** Its only two effects in this repo
# are `fullgraph` here and whether the graph-capture backend is constructed
# (`envs.py`, `neuron_model_runner.py:1431-1442`, `spec_decode/eagle.py:376`).
# Every compute step still compiles to a NEFF and runs on the device; what
# changes is that dynamo is allowed to end a graph at the host boundaries
# instead of raising.
os.environ.setdefault("VLLM_NEURON_DEBUG_MODE", "1")

# Warmup is a cheap pre-compile for an ordinary model. On the single-device
# ancestor it was also a 251 GiB disk read — `NeuronModelRunner` warms up by
# running a prefill at the batched-token bucket, which routes to most of the 384
# experts in all 37 MoE layers, and that port streamed every one of them off
# disk before serving anything.
#
# **That reason is gone on this deployment**: the experts are resident, so a
# warmup prefill reads nothing. It is still skipped, for the weaker reason it
# always also had — skipping loses nothing but the pre-compile, since the graphs
# warmup would build are built by the first real request instead out of the same
# `~/.cache/neuron_libtorch` cache — and because the warmup's shapes are not
# necessarily the ones a request keys, so it can cost compiles that nothing
# reuses. Worth re-testing now that it is merely a time trade rather than an
# hour of disk.
os.environ.setdefault("VLLM_NEURON_SKIP_PREFILL_DECODE_WARMUP", "1")

#: The largest prefill this port can run in one forward on one 24 GiB bank, and
#: a size `SUPPORTED_KV_SEGMENT_SIZES` accepts. See
#: `require_segmented_prefill`; 8192 rather than a smaller segment because a
#: segment is a full pass over the streamed expert population, so halving it
#: doubles the disk traffic for the same prompt — and because 1024 and 2048
#: both run into a `neuronx-cc` `Tensorizer/MaskPropagation` pass that takes
#: minutes per graph on this model's shapes (`.autohelix/notes/iter-5.md` §5e).
PREFILL_SEGMENT_TOKENS = 8192

# vLLM's multi-process executor gives each `execute_model` RPC
# `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, **300 by default**, and raises
#
#     TimeoutError: RPC call to sample_tokens timed out.
#
# when a worker has not answered by then. For an ordinary model 300 s is
# generous. For this one a single forward streams the routed experts of 37 MoE
# layers off disk — ~230 s at the measured 1.1 GB/s before any compilation —
# and the first forward at a new shape also builds its graphs. Measured at
# `--tensor-parallel-size 4`: all four ranks reached layer 39 of the first
# prefill and the executor killed the engine anyway
# (`.autohelix/notes/iter-5.md` §10h).
#
# It only matters above TP=1, because TP=1 uses the in-process `uni` executor
# and never sends an RPC — which is why four iterations at TP=1 never saw it.
# A timeout is not a correctness knob: nothing about what is computed changes,
# only how long the parent waits before declaring the worker dead.
os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "5400")

# ── The three timeouts that wedged the instance at TP=4, re-ordered ──────────
#
# `config.TENSOR_PARALLEL_SIZE`'s note records the failure: graphs compile
# mid-forward here, and a rank still compiling outlasts the peers waiting for it
# inside a collective. They gave up, retried against a rank still compiling, and
# left the collective's device-side state in a condition no process could clear
# — two reboots. The ordering was
#
#     collective RX gives up     960 s   (`CCOM WARN Timeout waiting for RX`)
#     a measured compile        1045, 1161, 1167 s
#     NEFF cache lock gives up  1200 s
#
# A *waiting* rank is the one that has to be patient, so every wait here is
# raised above any compile this model has produced. What this buys is not
# speed — it converts an unrecoverable wedge into plain slowness, which a
# `kill` recovers from, because a collective that is still waiting has not yet
# entered the retry path that corrupts the device state.
#
#   `NEURON_CCOM_NO_INFINITE_TIMEOUT=0`
#       `libnccom.so` registers a parameter `CCOM_NO_INFINITE_TIMEOUT`
#       (`ccom_param_lock_NO_INFINITE_TIMEOUT` in the symbol table) beside the
#       explicit env string `NEURON_CCOM_SOCK_TIMEOUT`. Turning it off asks the
#       collective to wait rather than to retry. The `NEURON_`-prefixed spelling
#       is the library's convention for its parameters; the bare one is set too,
#       since the prefix is applied at runtime and only one of the two can be
#       read — an ignored variable costs nothing.
#   `NEURON_CCOM_SOCK_TIMEOUT`
#       the socket wait the 960 s RX figure derives from.
#   `NEURON_RT_EXEC_TIMEOUT`
#       libtorch_neuronx_lite sets this to 300 s itself
#       (`_compiler/__init__.py:88`). A graph whose only work is a collective
#       executes for as long as its slowest peer takes to arrive, so at TP>1
#       300 s is a limit on *other ranks' compile time*. 300 is overridden
#       rather than `setdefault`ed because libtorch usually gets there first;
#       any other value is treated as the operator's and left alone.
#   `TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT`
#       the four ranks share one compile cache and serialize on this lock
#       (`cache_utils.get_lock_timeout`, default 1200 s). It was 33 s from
#       failing and did fail once.
#   `NEURON_LIBTORCH_COMPILATION_TIMEOUT`
#       **the fourth timeout, and the one that actually killed iteration 9's
#       best 4-rank run.** It is a different mechanism from the NEFF cache lock
#       above: the four ranks compile the *same* graph, so one takes the entry
#       and the other three sit in `libtorch_neuronx_lite/compile/cache.py`'s
#       `_wait_for_completion`, polling for the artifact. That wait is bounded
#       by `envs.NEURON_LIBTORCH_COMPILATION_TIMEOUT`, **default 600 s**, and a
#       waiter that reaches it does not degrade gracefully — `cache.py:363`
#       prints `Compilation failed - terminating process for cleanup` and
#       *terminates the worker*, which the parent then reports as
#       `Worker proc VllmWorker-1 died unexpectedly` with no other diagnostic.
#       That is exactly iteration 9's probe 11, from its own log: rank 3 took a
#       cache-miss key at 01:21:25, ranks 0-2 printed `Waiting for shared
#       compilation by process ...`, and at 01:31:26 — 600.3 s later, to the
#       tenth of a second, and *before* rank 3 finished — all three waiters
#       terminated themselves. Rank 3's compile then completed anyway: the
#       cache entry's 132 MB NEFF is timestamped 01:32, a minute after the run
#       was already dead. So the run was killed by three peers' patience
#       running out, not by a crash, and iteration 9's note ("a worker died with
#       no diagnostic ... I have no evidence to choose") can be closed.
#       Raised to 7200 for the same reason as every other wait in this block: a
#       waiting process is the one that has to be patient, and a graph of this
#       model has been measured at 1167 s.
#
# None of this is a correctness knob and none of it has any effect at TP=1,
# where there is no second rank to wait for: what is computed does not change,
# only how long a process waits before declaring a peer dead. The compilation
# timeout in particular is read only on the waiting path, which a single-process
# engine never enters.
os.environ.setdefault("NEURON_CCOM_NO_INFINITE_TIMEOUT", "0")
os.environ.setdefault("CCOM_NO_INFINITE_TIMEOUT", "0")
os.environ.setdefault("NEURON_CCOM_SOCK_TIMEOUT", "7200")
os.environ.setdefault("TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT", "7200")
os.environ.setdefault("NEURON_LIBTORCH_COMPILATION_TIMEOUT", "7200")
if os.environ.get("NEURON_RT_EXEC_TIMEOUT", "300") == "300":
    os.environ["NEURON_RT_EXEC_TIMEOUT"] = "7200"

# ── The multi-rank host environment, applied only above one rank ─────────────
#
# Four ranks ask ~4x of *host* memory what one rank asks, and the 4-rank runs of
# iterations 8 and 9 died of that rather than of anything on the device: the
# ranks sat at 6-7 GiB of their 24 GiB HBM banks while committed host memory ran
# to 112.9 GiB of a 114.8 GiB `CommitLimit` and the kernel refused with
# `std::bad_alloc`. The allocation that overflows appears as the graphs *after*
# layer 39 load — the LM head and the three MTP stages — taking every rank from
# ~9.7 to ~26 GiB within two seconds of each other. It reports itself as
# `NDBG:ndebug_info_graph_build`, the runtime's per-NEFF debug-info graph, but
# that is the straw: the same load costs ~2 GiB a rank at TP=2 and ~16 at TP=4,
# so the thing that scales is the peer count, not the NEFF.
#
# Six settings, each with a measurement behind it, bring a 4-rank run through
# that load. They are applied as a group and only when this port's own
# `config.TENSOR_PARALLEL_SIZE` is above 1, because every one of them is dead
# weight at a single rank and the single-rank path is the one that passes the
# gate today. An operator who exports any of them keeps their value.
#
#   `NEURON_RT_DBG_FORCE_2DEV_PROXY=1`  the one that moves the number: peak
#       commitment 82.9 GiB instead of 112.9, no `bad_alloc`, a 4-rank run
#       through the post-layer-39 load for the first time. It is an undocumented
#       `NEURON_RT_DBG_*`, so its *semantics* were checked rather than assumed,
#       at TP=2 where a known-good run exists to diff against: with and without
#       it, all three completions are byte-identical over 32 tokens each and the
#       rejection sampler's acceptance counts match step for step (iter-9 §6b).
#       It changes how collective data moves, not what a collective computes.
#   `NEURON_RT_INSPECT_ENABLE=0`, `NEURON_RT_INSPECT_PRECACHE_ENABLE=0`  a rank's
#       base commitment falls 13 -> 9.7 GiB. These disable runtime *inspection*
#       (the profiling/debug side-channel), not any part of execution.
#   `MALLOC_ARENA_MAX=2`, `MALLOC_TRIM_THRESHOLD_`/`MALLOC_TOP_PAD_=128 KiB`  the
#       largest single win: a rank at layer 39 holds 15.8 GiB instead of 25.8.
#       glibc gives each thread its own 64 MB arena by default and a rank here
#       runs ~97 threads, so the arenas alone are most of that. The environment
#       variables are only read at *process* startup, and vLLM's workers are
#       `fork`ed from a process that has long since initialised its allocator —
#       so `apply_multi_rank_host_tuning` also calls `mallopt` directly, which is
#       honoured at any time and is inherited across `fork`. Both, because the
#       env vars are what a `spawn`ed worker would get.
#
# Four things that look like levers and are not, so nobody spends a run on them
# again (iter-9 §2b, §2c, measured over eleven 4-rank runs):
#
#   `NEURON_RT_CC_ALG_TYPES=INTRA_SINGLE_STEP_MESH_ALG`  no help and a net
#       negative. The overflowing mapping is byte-identical under the default
#       `INTRA_RING_ALG` and under the mesh, and the mesh *adds* an
#       `allr_sz <= ch_buf_split_sz` assert that forces small all-reduce chunks.
#       (The ~30 GiB figure once reported for the mesh was a run that asserted
#       before the expensive graphs had loaded.)
#   `NEURON_RT_CC_CHANNEL_BUFFER_N`, `NEURON_RT_CC_CHUNK_SIZE`  no change.
#   chunking the all-reduce (`_ALLREDUCE_CHUNK_ELEMS`)  exact, harmless, and
#       worth no host memory.
#   a compiler flag to strip the NEFF's debug-info section  does not exist;
#       `neuronx-cc compile --help-hidden-list`'s 84 options have nothing for it.
#
# `NEURON_HOST_MEM_PER_RANK_GIB` is set here too, and that needs justifying,
# because the preflight in `vllm_neuron/vllm/host_memory.py` is a guard against
# the failure that twice cost a reboot and lowering a guard to admit a run is
# exactly the move that would be cheating. What makes it calibration instead:
# `DEFAULT_PER_RANK_GIB = 24.5` was measured *without* the six settings above,
# on a run that then died. Under the configuration this block actually
# establishes, a rank's touched commitment peaks at ~18 GiB, so 24.5 describes a
# run this port no longer makes. The figure set below is the measured peak plus
# headroom, and the guard still refuses anything past it — `vm.overcommit_memory=2`
# remains the real backstop underneath. If a run dies of host memory anyway, the
# number here is wrong and should go *up*, not the limit down.
#
#   iteration 10 measurement, four ranks, this exact environment, from
#   `/proc/<worker>/status` every 5 s through the post-layer-39 load:
#   see `.autohelix/notes/iter-10.md` §2 for the table.
#
# `NEURON_RT_IO_RING_CACHE_SIZE` is iteration 11's addition and the only one
# here that is not about `malloc`. `neuron_worker.py:713` raises the runtime's
# own default of 1 to **32** — "so the runtime reuses the per-step DMA ring
# across decode steps instead of rebuilding it every step (kbl_exec_pre), which
# otherwise dominates async-decode model-submit time" — and offers the override
# in the same comment. It is a sound default for a model whose decode step is
# milliseconds. This port's decode step is ~10 s (it streams the routed experts
# of 37 MoE layers), so the submit-time win is noise, while 32 cached DMA rings
# *per loaded NEFF* is charged against exactly the pool that ran out at four
# ranks:
#
#     TDRV:vring_allocate_next_desc   Failed to expand vring page pool
#
# Restoring the runtime's default is therefore a cost this port pays nothing
# for. Above one rank only, because at one rank the pool has never run out.
#
# Two neighbouring knobs were tried with it and **must not be touched**:
# `NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS` (the worker raises it 8 -> 32 "to
# avoid intermittent Execution Queue Full issues") and
# `NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS`. Setting either to 1 makes the
# very first forward die with
#
#     RuntimeError: Failed to schedule neff execution. status=7
#                   message=Execution Queue Full
#
# — the queue is not per in-flight *request* here, it is the submission queue
# for the many NEFFs one forward executes. `/tmp/iter11/probe3.log`.
_MULTI_RANK_ENV = {
    "NEURON_RT_INSPECT_ENABLE": "0",
    "NEURON_RT_INSPECT_PRECACHE_ENABLE": "0",
    "MALLOC_ARENA_MAX": "2",
    "MALLOC_TRIM_THRESHOLD_": "131072",
    "MALLOC_TOP_PAD_": "131072",
    "NEURON_HOST_MEM_PER_RANK_GIB": "20.0",
}

#: `NEURON_RT_IO_RING_CACHE_SIZE`, and why this port no longer forces it to 1.
#:
#: `neuron_worker.py:713` raises the runtime's default of 1 to **32** so that
#: "the runtime reuses the per-step DMA ring across decode steps instead of
#: rebuilding it every step (kbl_exec_pre), **which otherwise dominates
#: async-decode model-submit time**" — their emphasis's worth, not ours; a
#: cache-size sweep put the knee at 32.
#:
#: The single-device port overrode that back to 1, and its reason was sound
#: *there*: "this port's decode step is ~10 s, so it buys nothing here", while
#: 32 cached rings per loaded NEFF were charged against the same host pool that
#: was failing with `TDRV:vring_allocate_next_desc Failed to expand vring page
#: pool` on a 114.8 GiB `CommitLimit`.
#:
#: Both premises are gone. A resident decode step is not ~10 s — it reads ~1 GiB
#: of weights from HBM and should be milliseconds — so per-execution submit cost
#: is no longer noise but the thing being measured. And the host limit is 1700
#: GiB. Meanwhile this model's forward executes **hundreds** of graphs
#: (`fullgraph=False`, see `VLLM_NEURON_DEBUG_MODE`), so a per-execution ring
#: rebuild is paid hundreds of times a forward rather than once.
#:
#: Left to the worker's own default, which is what the sweep chose. Set
#: `NEURON_RT_IO_RING_CACHE_SIZE=1` to restore the old behaviour for an A/B.

#: `NEURON_RT_DBG_FORCE_2DEV_PROXY`, and why it is **not** in the dict above any
#: more.
#:
#: It is the setting that made a 4-rank run possible on the trn2.3xlarge — peak
#: commitment 82.9 GiB instead of 112.9 — and its semantics were checked rather
#: than assumed, at TP=2, by diffing completions and acceptance counts against a
#: known-good run. But both the measurement and the check were made with **all
#: ranks on one device**, and it is an undocumented `NEURON_RT_DBG_*` whose name
#: says it forces collective data to move as though there were two devices. On
#: this deployment there are sixteen, in a 4x4 torus, and a knob that overrides
#: how a collective routes is exactly the kind that can be silently wrong when
#: the topology it was validated against is not the topology it is applied to.
#:
#: So it is applied only in the regime it was measured in. The host-memory
#: problem it solved is also not this deployment's: `CommitLimit` here is 1700
#: GiB against the dev host's 114.8, which is what
#: `/etc/sysctl.d/99-neuron-host-memory.conf` sets.
#:
#: `DEEPSEEK_V41_FORCE_2DEV_PROXY=1` applies it anyway, for the A/B.
_2DEV_PROXY_MAX_RANKS = 4

#: `mallopt` parameter numbers from glibc's `malloc.h`, paired with the value the
#: matching `MALLOC_*` environment variable above asks for. Negative by
#: convention: `M_TRIM_THRESHOLD -1`, `M_TOP_PAD -2`, `M_ARENA_MAX -8`.
_MALLOPT_PARAMS = ((-8, 2), (-1, 131072), (-2, 131072))


def apply_multi_rank_host_tuning(world_size: int) -> None:
    """Establish the host-memory environment a >1-rank run of this port needs.

    A no-op at one rank, and a no-op for any variable the operator has already
    exported. See the block above for what each setting is worth; the short
    version is that without them four ranks run the host's commitment to 112.9
    GiB of a 114.8 GiB limit while loading the graphs after layer 39, and with
    them the same load peaks at 82.9.

    Called from `check_and_update_config` as well as at import, because a worker
    that unpickles the config re-enters this module while the driver's
    environment is already set — `setdefault` makes both paths idempotent.
    """
    if world_size <= 1:
        return
    settings = dict(_MULTI_RANK_ENV)
    if (
        world_size <= _2DEV_PROXY_MAX_RANKS
        or os.environ.get("DEEPSEEK_V41_FORCE_2DEV_PROXY") == "1"
    ):
        settings["NEURON_RT_DBG_FORCE_2DEV_PROXY"] = "1"
    applied = {}
    for key, value in settings.items():
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    # The glibc tunables again, through `mallopt`, which takes effect in *this*
    # process rather than only in one started afresh. Children inherit it across
    # `fork`, which is how vLLM makes its workers by default.
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        for param, value in _MALLOPT_PARAMS:
            libc.mallopt(ctypes.c_int(param), ctypes.c_int(value))
    except OSError as exc:  # pragma: no cover - glibc is always present on Linux
        logger.info("deepseek_v41: mallopt unavailable (%s)", exc)
    if applied:
        logger.info(
            "deepseek_v41: %d rank(s) — applied the measured multi-rank "
            "host environment %s; see __init__.py for what each is worth",
            world_size, applied,
        )


# Applied at import, on the rank count this port *declares*, because
# `NEURON_HOST_MEM_PER_RANK_GIB` has to be in the environment before
# `vllm_neuron/vllm/platform.py:366` runs its host-memory preflight — and that
# call lives inside the platform's own `check_and_update_config`, which the
# port's hook wraps *around*, so a value supplied from inside the hook's body
# would arrive after the preflight had already read the default.
# `declare_prefill_requirements` applies it a second time, on the world size
# vLLM actually resolved, which covers `--tensor-parallel-size` on the command
# line disagreeing with `config.py`.
apply_multi_rank_host_tuning(_config.TENSOR_PARALLEL_SIZE)

__all__ = [
    "DeepseekV41ForCausalLM",
    "DeepseekV41HFConfig",
    "DeepseekV41TextConfig",
    "DeepseekV41VisionConfig",
    "ISLPY_VALIDATED",
    "PREFILL_SEGMENT_TOKENS",
    "quiet_per_call_logging",
    "register_hf_config",
]


#: Loggers outside this package that emit **once per call** and so scale with
#: the forward, the layer count and the rank count all at once. Quieted to
#: WARNING, which leaves every failure and fallback visible.
#:
#: `vllm_neuron.functional.topk` logs `[topk] kernel=... rows=... vocab=... k=...`
#: on every top-k, which this model performs per MoE layer and per sampling
#: step: **4,352 lines of run 2's log**. The line is genuinely useful once — it
#: says whether the NKI rotational kernel or the torch fallback ran — and it
#: says the same thing every time after that.
_PER_CALL_LOGGERS = ("vllm_neuron.functional.topk",)


def quiet_per_call_logging() -> None:
    """Cut the log down to what a reader can actually use.

    Two things, both scoped to this port and to loggers it is the heavy user of
    — nothing here changes logging policy for the other architectures this
    backend serves.

    **1. Per-call loggers go to WARNING.** See `_PER_CALL_LOGGERS`.

    **2. This package's records are emitted once instead of twice.**
    `vllm_neuron/logging_config.py` deliberately puts a handler on the
    `vllm_neuron` logger *and* leaves `propagate=True`, so that logs survive
    vLLM reconfiguring the root logger and still pick up its worker prefix. The
    cost is that every record is written twice — **73% of run 2's 177,144 lines
    were exact duplicates.** That trade is the repo's to make for its own
    models; this package opts out by carrying its own handler and not
    propagating. The `(Worker_TP0 pid=...)` prefix is unaffected: it is added by
    vLLM's wrapper around the worker's stdout, not by a formatter.

    Idempotent, and skipped entirely if an operator has set
    `DEEPSEEK_V41_VERBOSE_LOGGING=1`.
    """
    if os.environ.get("DEEPSEEK_V41_VERBOSE_LOGGING") == "1":
        return
    for name in _PER_CALL_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    package = logging.getLogger(__name__)
    if not package.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - "
                "%(message)s"
            )
        )
        package.addHandler(handler)
        package.propagate = False


def put_bundled_compiler_on_path() -> None:
    """Make `shutil.which("neuronx-cc")` find the compiler in this venv.

    `libtorch_neuronx_lite.compile.backend` locates the compiler with
    `shutil.which("neuronx-cc")` and raises

        RuntimeError: neuronx-cc compiler binary does not exist

    when it is absent, so **every graph this port has not already cached fails
    to build** — the engine comes up, the weights load, and the first forward
    dies in the backend.

    `neuronx-cc` is a console script installed by the `neuronx-cc` wheel, so it
    lands in the environment's own `bin/` — the directory `sys.executable` is
    in — and is on `PATH` only for a shell that activated the venv. The
    documented invocations in `docs/` all run inside an activated environment;
    an invocation that names the interpreter by path instead, which is how this
    port's own verifier is launched, inherits a `PATH` without it. That is a
    launcher difference with nothing to do with the model, and it costs a whole
    device run to discover, so it is repaired here rather than relied on.

    Appended rather than prepended, and only when `which` cannot already find
    it: an operator with a deliberately chosen compiler on `PATH` keeps it.
    """
    import shutil

    if shutil.which("neuronx-cc") is not None:
        return
    bindir = os.path.dirname(os.path.abspath(sys.executable))
    if not os.path.isfile(os.path.join(bindir, "neuronx-cc")):
        return
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + bindir
    logger.info(
        "deepseek_v41: added %s to PATH so neuronx-cc is found "
        "(the compiler ships in this interpreter's own environment)", bindir,
    )


#: Fields the top-level config mirrors off `text_config`. vLLM and a handful of
#: generic HF helpers read them without descending into the sub-config.
_MIRRORED_FROM_TEXT = (
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
    "rms_norm_eps",
    "hidden_act",
    "intermediate_size",
)


class DeepseekV41TextConfig(PretrainedConfig):
    """The `text_config` block: the language model this port actually serves.

    Written in the transformers-5 style — annotated class attributes, no
    `__init__` — because `PreTrainedConfig.__init_subclass__` turns every
    subclass into a `kw_only` dataclass. Two consequences worth knowing:
    annotating a class attribute makes it a dataclass *field* (so a mutable
    default raises), and a subclass without its own `__init__` gets wrapped to
    accept arbitrary keywords, which is what lets the rest of the checkpoint's
    keys (`compress_ratios`, the `engram_*` family, the `dspark_*` family, ...)
    land as plain attributes instead of a `TypeError`.

    Defaults are the official release's, so `DeepseekV41TextConfig()` describes
    the real model rather than a toy.
    """

    model_type = "deepseek_v41_text"

    vocab_size: int = 129280
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    moe_intermediate_size: int = 2304
    max_position_embeddings: int = 1_048_576
    rms_norm_eps: float = 1e-20
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    # Absent from this checkpoint: the FFN is MoE, so the per-expert width is
    # `moe_intermediate_size`. vLLM's memory profiling reads
    # `intermediate_size` unconditionally, so `__post_init__` mirrors it rather
    # than leaving an AttributeError lying around.
    intermediate_size: int | None = None

    def __post_init__(self, **kwargs):
        if self.intermediate_size is None:
            self.intermediate_size = self.moe_intermediate_size
        super().__post_init__(**kwargs)


class DeepseekV41VisionConfig(PretrainedConfig):
    """The `vision_config` block: recorded, not modelled.

    The official config always carries one, so it has to parse; this port serves
    the text path only (see `factory._validate_config`), and the vision tower is
    off the `tokens -> logits` path.
    """

    model_type = "deepseek_v41_vision"

    num_hidden_layers: int = 32
    hidden_size: int = 1024
    num_attention_heads: int = 16
    intermediate_size: int = 2816
    patch_size: int = 14


class DeepseekV41HFConfig(PretrainedConfig):
    """Top level of the checkpoint's nested config: `model_type: deepseek_v41`.

    `sub_configs` is what lets `transformers` build the two blocks above out of
    the nested JSON, and what makes `get_text_config()` return the text block —
    which is where vLLM looks for `hidden_size`, `num_hidden_layers` and the
    rest.
    """

    model_type = "deepseek_v41"
    sub_configs = {"text_config": DeepseekV41TextConfig}

    text_config: dict | PretrainedConfig | None = None
    #: The checkpoint's `vision_config` block, renamed. It is recorded, not
    #: modelled — this port serves the text path only, and the vision tower is
    #: off the `tokens -> logits` path (`factory._validate_config`,
    #: `config.py`'s module docstring, and `floorplan/`, which treats placing
    #: the tower as an error).
    #:
    #: The rename is load-bearing, not cosmetic. `NeuronPlatform` keys its whole
    #: vision path off `hasattr(hf_config, "vision_config")`
    #: (`_resolve_vision_auto_config`: "Only called for models with
    #: hf_config.vision_config"), which populates
    #: `additional_config["vision_neuron_config"]`, which makes
    #: `NeuronModelRunner.load_model` take its multimodal branch and call
    #: `from_configs(hf_config=..., text_neuron_config=...,
    #: vision_neuron_config=...)`. Serving this checkpoint as the causal LM it
    #: is means not advertising a vision tower that this port does not build.
    vision_tower_config: dict | PretrainedConfig | None = None
    #: How the weights are *stored*: `quant_method: fp8` with ue8m0 scales over
    #: 32x32 blocks for attention, `expert_dtype: fp4` for the experts. Carried
    #: through as a plain dict — `weight_loader.py` reads it, and
    #: `transformers` leaves a dict alone instead of resolving it against its
    #: own quantizer registry, which has no entry for this layout.
    quantization_config: dict | None = None
    image_token_id: int = 129_264
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config = DeepseekV41TextConfig(**self.text_config)
        elif self.text_config is None:
            self.text_config = DeepseekV41TextConfig()

        # `vision_config` as the checkpoint spells it, moved to the name above.
        vision = kwargs.pop("vision_config", None)
        if vision is not None and self.vision_tower_config is None:
            self.vision_tower_config = vision
        if isinstance(self.vision_tower_config, dict):
            self.vision_tower_config = DeepseekV41VisionConfig(
                **self.vision_tower_config
            )

        super().__post_init__(**kwargs)

        for field in _MIRRORED_FROM_TEXT:
            if getattr(self, field, None) is None:
                setattr(self, field, getattr(self.text_config, field, None))

    def get_text_config(self, *args, **kwargs) -> PretrainedConfig:
        """The language-model half, which is the half this port serves."""
        return self.text_config


def register_hf_config() -> None:
    """Make `transformers.AutoConfig` recognize this architecture.

    Idempotent: `exist_ok=True` means a second call (worker subprocess, config
    unpickle, a re-import under a different parent) is a no-op rather than a
    `ValueError`.
    """
    for config_cls in (
        DeepseekV41TextConfig,
        DeepseekV41VisionConfig,
        DeepseekV41HFConfig,
    ):
        AutoConfig.register(config_cls.model_type, config_cls, exist_ok=True)


def register_architecture() -> None:
    """Make vLLM's own `ModelRegistry` accept `DeepseekV41ForCausalLM`.

    `ModelConfig` validates `architectures[0]` against `vllm.ModelRegistry`
    before any Neuron code runs, and raises

        Model architectures ['DeepseekV41ForCausalLM'] are not supported for now

    if it is absent. `vllm_neuron.model.registry` is *not* enough on its own:
    `NeuronWorker` copies it into vLLM's registry, but that happens in the
    worker, long after the main process validated `ModelConfig`. Every other
    architecture this backend serves (Llama, Qwen3, GptOss) happens to also be
    in vLLM's in-tree list, so the gap only shows up for a model vLLM has never
    heard of.

    `ARCH` is the checkpoint's `architectures[0]`; the target is
    `factory.DeepseekV41ForCausalLM`, matching `vllm_neuron.model.registry`, so
    both registries resolve to the same class.
    """
    from vllm.model_executor.models.registry import ModelRegistry

    if ARCH in ModelRegistry.get_supported_archs():
        return
    ModelRegistry.register_model(ARCH, f"{__name__}.factory:{ARCH}")


def declare_storage_quantization() -> None:
    """Tell the Neuron platform it can serve this checkpoint's fp8 storage.

    `ModelConfig._verify_quantization` reads the checkpoint's
    `quantization_config.quant_method` — `"fp8"` here — and hands it to
    `NeuronPlatform.verify_quantization`, whose `supported_quantization` list is
    `neuron_quant`, `compressed-tensors`, `modelopt`. So engine construction
    died on

        fp8 quantization is currently not supported in neuron

    Note what that list means: it names the *vLLM quantization backends* the
    Neuron platform knows how to drive. There is no vLLM backend for this
    checkpoint's layout — fp8 weights with ue8m0 scales over 32x32 blocks,
    experts in fp4 — and there should not be one, because this port does the
    dequantization itself: `weight_loader.py` on the staging path, and the
    fp8/fp4 NKI kernels under `kernels/`. Adding `"fp8"` here says "this port
    handles fp8-stored weights", which is what is true. It does not make vLLM
    apply any quantization machinery to these weights: `neuron_config`'s own
    `quantization` field stays `None`, so nothing in the Neuron weight path is
    rerouted, and `_validate_quantization_config` only constrains
    `compressed-tensors` checkpoints.

    Setting the checkpoint's own `quantization_config` aside instead does *not*
    work, and that dead end is worth recording:
    `vllm.transformers_utils.config.get_config` re-reads `config.json` off disk
    and force-assigns `config.quantization_config = <raw dict>` after
    `AutoConfig` has returned, so the value cannot be influenced from the
    config class at all.

    The list belongs to `vllm_neuron/vllm/platform.py`, which this port may not
    edit, so the declaration is made from here — the port's own setup module —
    rather than by widening a file outside its scope.
    """
    from vllm.platforms import current_platform

    platform_cls = type(current_platform)
    supported = getattr(platform_cls, "supported_quantization", None)
    if supported is not None and "fp8" not in supported:
        platform_cls.supported_quantization = [*supported, "fp8"]


#: The newest `islpy` neuronx-cc 2.27 can actually call. See
#: `warn_on_unsupported_islpy`.
ISLPY_VALIDATED = (2026, 1)


def warn_on_unsupported_islpy() -> None:
    """Say so at import if the compiler's integer-set binding is too new.

    `neuronx-cc`'s `IntegerSetAnalysis` is a wrapper over **islpy**, and its
    `Simplifier.optimizeMemSet` pass calls `is_subset()` on every graph whose
    simplification reaches a zero-fill — which, in this model, is most of them.
    With `islpy 2026.2.2` installed, that call does not match any overload the
    binding exposes and the compiler aborts:

        [INTERNAL_ERROR] [NCC_ISMP902] Simplifier error: is_subset():
        incompatible function arguments. The following argument types are
        supported: -

    The empty list of supported types is the tell: it is a pybind11 overload
    error, not anything about the graph. The same HLO, the same compiler build
    and the same NumPy compile to a NEFF against `islpy 2026.1`.

    **This is worth a line of log because of how it fails.** The error names a
    compiler pass and arrives one graph at a time, so it reads as a
    shape-specific compiler bug in the model — two iterations of this port
    chased it as one. The remedy is a pin:

        VIRTUAL_ENV=.../vllm-neuron/.venv uv pip install islpy==2026.1

    (that venv is managed by `uv`; it has no `bin/pip` and `python -m pip`
    fails.) A warning rather than a raise: the version comparison is against
    what this port was validated with, not a known-bad list, so a newer islpy
    that fixes the binding should not be fatal.
    """
    try:
        from importlib.metadata import version

        raw = version("islpy")
        parts = tuple(int(p) for p in raw.split(".")[:2])
    except Exception:  # noqa: BLE001 - diagnostics only, never fatal
        return
    if parts <= ISLPY_VALIDATED:
        return
    logger.warning(
        "deepseek_v41: islpy %s is newer than the %s neuronx-cc "
        "2.27 was validated against. If compiles fail with "
        "'[NCC_ISMP902] Simplifier error: is_subset()', that is why, and it is "
        "not the graph — pin islpy==%s. See "
        "deepseek_v41.warn_on_unsupported_islpy",
        raw, ".".join(str(p) for p in ISLPY_VALIDATED),
        ".".join(str(p) for p in ISLPY_VALIDATED),
    )


def _serves_this_port(vllm_config) -> bool:
    """Is `vllm_config` about to serve this checkpoint?

    Every hook below has to be inert for the other architectures this backend
    serves, which share the same platform object.
    """
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return False
    arches = getattr(model_config, "architectures", None)
    if not arches:
        arches = getattr(
            getattr(model_config, "hf_config", None), "architectures", None
        )
    return bool(arches) and ARCH in arches


def require_segmented_prefill(vllm_config) -> None:
    """Cap the prefill at what one Trainium bank can hold in one forward.

    **What this is.** `PREFILL_SEGMENT_TOKENS` is a property of the model: a
    prefill forward keeps a `[tokens, hc_mult, hidden]` stream, the attention
    scores over the sparse window and the MoE's staged experts live at the same
    time, and at 9424 tokens in one shot that does not fit in the 24 GiB a
    NeuronCore has. A longer prompt is served by *segmenting* the prefill, which
    is the whole timeshare idea applied along the token axis instead of the
    weight axis, and it is what this port is built for — `PagedSpan` addresses
    every cache by absolute position precisely so a prompt can arrive in pieces.

    So this declares the largest prefill the model can run, by lowering
    `scheduler_config.max_num_batched_tokens` to it. Lowering that below
    `max_model_len` is what `resolve_segmented_prefill_config` reads as "chunked
    prefill", and 8192 is one of the sizes it accepts, so the engine
    auto-enables segmented prefill at `kv_segment_size_buckets=[8192]` through
    its own supported path. Nothing is bypassed and no check is relaxed: the
    engine ends up in the configuration `docs/guides/features-guide.md`
    recommends for long contexts (`--max-num-batched-tokens 8192`).

    **Why it has to be here.** `max_num_batched_tokens` is an
    engine-construction argument, settled before any model class exists — the
    same reason the verifier has to read `config.TENSOR_PARALLEL_SIZE` rather
    than letting the model pick its own world size. A caller who does not pass
    it gets `max_num_batched_tokens == max_model_len`, which is a single-shot
    prefill, and on this build a single-shot prefill with vLLM's default
    Automatic Prefix Caching is rejected outright in
    `NeuronModelRunner.__init__`:

        ValueError: Automatic Prefix Caching (APC) requires segmented prefill
        to be enabled. Either disable APC with '--no-enable-prefix-caching' or
        set 'max_num_batched_tokens' to a supported segmented prefill size
        [512, 1024, 2048, 4096, 8192] to auto-enable segmented prefill.

    That raise is in the model *runner*'s constructor, which runs before
    `load_model`, so the hooks this port already uses to reach engine
    configuration (`model._cap_kv_cache_blocks`, from `load_weights`) are all
    downstream of it. The only place a model package gets to speak before it is
    the platform's `check_and_update_config`, which is where vLLM platforms
    normally clamp exactly this field — and which lives in
    `vllm_neuron/vllm/platform.py`, a file this port may not edit. So the
    declaration is made from the port's own setup module, the same way
    `declare_storage_quantization` widens the platform's quantization list.

    **It only ever lowers, and never contradicts the operator.** If a caller
    passed `--max-num-batched-tokens`, or set `kv_segment_size_buckets`
    themselves, or asked for a prefill this port can already run in one shot,
    this does nothing.
    """
    if not _serves_this_port(vllm_config):
        return
    scheduler = getattr(vllm_config, "scheduler_config", None)
    model_config = vllm_config.model_config
    if scheduler is None:
        return

    neuron_config = (getattr(vllm_config, "additional_config", None) or {}).get(
        "neuron_config", {}
    )
    if "kv_segment_size_buckets" in neuron_config:
        return
    if "num_batched_tokens_buckets" in neuron_config:
        return

    batched = int(getattr(scheduler, "max_num_batched_tokens", 0) or 0)
    max_model_len = int(getattr(model_config, "max_model_len", 0) or 0)
    # The prefill the engine would actually run in one forward. Below the
    # segment size there is nothing to segment, and a caller who already named
    # a chunk size is already on the segmented path.
    single_shot = min(batched, max_model_len) if batched else max_model_len
    if single_shot <= PREFILL_SEGMENT_TOKENS:
        return

    scheduler.max_num_batched_tokens = PREFILL_SEGMENT_TOKENS
    if getattr(scheduler, "max_num_partial_prefills", 1) and hasattr(
        scheduler, "long_prefill_token_threshold"
    ):
        # Keep vLLM's own long-prefill accounting consistent with the new
        # budget; it is derived from `max_num_batched_tokens` when unset and a
        # stale larger value would let one request claim the whole budget it no
        # longer has.
        threshold = int(getattr(scheduler, "long_prefill_token_threshold", 0) or 0)
        if threshold > PREFILL_SEGMENT_TOKENS:
            scheduler.long_prefill_token_threshold = PREFILL_SEGMENT_TOKENS
    logger.info(
        "deepseek_v41: max_num_batched_tokens %d -> %d for a "
        "max_model_len of %d. A prefill longer than %d tokens does not fit one "
        "NeuronCore's bank in a single forward, so it is segmented; see "
        "deepseek_v41.require_segmented_prefill",
        batched, PREFILL_SEGMENT_TOKENS, max_model_len, PREFILL_SEGMENT_TOKENS,
    )


#: The prefill segment a *multi-rank* run uses. `PREFILL_SEGMENT_TOKENS` above
#: is the ceiling for a single rank; this is a clamp that only applies above one
#: rank, and it is the balance point between two costs that pull in opposite
#: directions. Both were measured at four ranks on 2026-10-13 and the numbers
#: are in `.autohelix/notes/iter-11.md` §2:
#:
#:   * **compile time grows with the segment**, worse than linearly, because a
#:     segment is a prefill graph's row count. 512 rows is 35 s a graph; 2048
#:     rows is 1.5M backend instructions and had not finished after 25 minutes.
#:   * **host memory grows with the segment *count*.** Every chunk of a chunked
#:     prefill traces its own graphs — `read_len` and the span length are shapes
#:     — and every loaded NEFF keeps host memory for its DMA rings in each of
#:     the four worker processes. Measured: ~12 GiB of `Committed_AS` per
#:     chunk's graph set at 512 rows (~25 at 1024), on a 114.8 GiB
#:     `CommitLimit`. Before the read was pinned, a 2008-token prompt in
#:     512-token segments was four sets and the third one ran the limit out:
#:
#:         Committed_AS 111.7 GiB of 114.8   (07:09:57, forward 2 of 4)
#:         TDRV:vring_allocate_next_desc   Failed to expand vring page pool
#:         ENC:__compose_channel           failed to compose cc op with mesh
#:         NMGR:dlr_build_and_load_cc_resources  Failed to build and load
#:                                               collectives resources
#:         RuntimeError: Failed to schedule neff execution. status=1
#:
#:     That is a *host* allocation failing — the vring pool grows in 1 MiB
#:     pages (`VRING_DESC_PAGE_SIZE`) and `vm.overcommit_memory=2` refused the
#:     next one — and it is the same allocation iteration 10 saw as
#:     `dma_util_expand_tiling_dims`/`create_data_refill_rings`. It is not
#:     device HBM, which sat at 7.5 GiB of 24 a rank.
#:
#: **512, and the reason it is the small end rather than the large one is
#: `model._COMPRESS_READ_CEILING`.** Without that, chunking was a straight
#: trade: four 512-token chunks are four graph sets where two 1024-token chunks
#: are two, at a fixed product, and neither fits. With the read pinned to the
#: context, chunks *of equal length share one set* — the 2008-token case's
#: chunks 2 and 3 now add no graphs and no compiles at all — so the number of
#: chunks stops being what costs, and the cheap-to-compile row count wins.
#: 1024 would work as well and compile a little slower per graph; 2048 would be
#: one set and cannot be compiled at all at four ranks (below).
#:
#: What is still not shared is the **last** chunk, which is short (472 of 512
#: tokens for this prompt) and so keys its own set. Bucketing the span length
#: the way the read is bucketed is the remaining ~12 GiB and the one change
#: between this port and a four-rank verifier pass — `.autohelix/notes/iter-11.md`
#: §1.5, including the hazard (the padding rows must not write KV).
#:
#: Phase 2 of the verifier asks for 512 already and keeps it: this only lowers.
#:
#: The evidence for the compile half, which is the one that is easy to doubt:
#: the graph iteration 10's compiler was working on when it gave up is still on
#: disk, with no `.neff` beside it —
#:
#:     ~/.cache/neuron_libtorch/neuron/compile_cache/f4aaaa37...  (no .neff)
#:     example_inputs    (2048, 4, 5120)   <- 2048 rows, hc_mult 4, hidden
#:     log-neuron-cc.txt 25 minutes in, still in the backend scheduler:
#:                       "allocs=124716 instructions=1515322"
#:
#: — and that is one of the ~6 structurally distinct graphs a chunk needs. It is
#: *not* the `Tensorizer/MaskPropagation` stall iteration 5 named: this one is
#: past that pass and in `PostSched`, so it is the instruction count rather than
#: a pathological pass, and a smaller row count is the only lever on it.
#:
#: And what the segment costs, so the trade is on the record: a segment is one
#: full pass over the streamed expert population, so a 2008-token prompt in
#: 512-token chunks streams the routed experts four times instead of once —
#: about 15 minutes against 4 on this host's measured 1.1 GB/s. **Nothing about what
#: is computed changes**: segmented prefill is this port's own design
#: (`PagedSpan` addresses every cache by absolute position precisely so a prompt
#: can arrive in pieces), the whole prompt is prefilled either way, and
#: attention still reads the whole context — the later chunks read the earlier
#: chunks' KV out of the cache.
MULTI_RANK_PREFILL_SEGMENT_TOKENS = 512

#: The value, and the whole argument for it, live in `config.py` beside
#: `TENSOR_PARALLEL_SIZE` and `ATTENTION_TENSOR_PARALLEL_SIZE`: those three
#: are the port's tunable parallelism declarations, so they belong in one
#: editable place rather than scattered between the config and the hooks
#: that apply them. Imported under the old name so the hooks below are
#: unchanged.
from .config import MAX_NUM_SEQS as MULTI_RANK_MAX_NUM_SEQS  # noqa: E402


def require_multi_rank_budget(vllm_config) -> None:
    """Declare the prefill segment and batch size this port serves above one rank.

    Two clamps, both inert at one rank, both lowering only, and neither touching
    what the model computes:

      * `max_num_batched_tokens` down to `MULTI_RANK_PREFILL_SEGMENT_TOKENS`,
        which is a compile-budget declaration;
      * `max_num_seqs` down to `MULTI_RANK_MAX_NUM_SEQS`, which is a
        host-memory one.

    The constants' comments carry the measurements. This runs before
    `require_segmented_prefill` (which then has nothing left to do) and before
    `require_kv_cache_ceiling` (which sizes the cache from `max_num_seqs`, so it
    has to see the clamped value).

    Like the other hooks it defers to an operator who has configured the
    segmented-prefill path themselves, and it says what it did in the log rather
    than changing the engine silently.
    """
    if not _serves_this_port(vllm_config):
        return
    parallel = getattr(vllm_config, "parallel_config", None)
    scheduler = getattr(vllm_config, "scheduler_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    if parallel is None or scheduler is None or model_config is None:
        return
    world_size = int(getattr(parallel, "world_size", 1) or 1)
    if world_size <= 1:
        return

    neuron_config = (getattr(vllm_config, "additional_config", None) or {}).get(
        "neuron_config", {}
    )
    operator_chose_buckets = (
        "kv_segment_size_buckets" in neuron_config
        or "num_batched_tokens_buckets" in neuron_config
    )

    max_model_len = int(getattr(model_config, "max_model_len", 0) or 0)
    batched = int(getattr(scheduler, "max_num_batched_tokens", 0) or 0)
    segment = MULTI_RANK_PREFILL_SEGMENT_TOKENS
    # Only below `max_model_len` is a segment a segment; at or above it the
    # engine reads it as a single-shot prefill, which is the configuration
    # `require_segmented_prefill` exists to avoid.
    if (
        not operator_chose_buckets
        and batched > segment
        and max_model_len > segment
    ):
        scheduler.max_num_batched_tokens = segment
        threshold = int(getattr(scheduler, "long_prefill_token_threshold", 0) or 0)
        if threshold > segment:
            scheduler.long_prefill_token_threshold = segment
        logger.info(
            "deepseek_v41: %d rank(s) — max_num_batched_tokens %d -> %d. "
            "A segment's row count is a prefill graph's size, and at %d ranks every "
            "sharded shape is a fresh compile; see "
            "deepseek_v41.MULTI_RANK_PREFILL_SEGMENT_TOKENS",
            world_size, batched, segment, world_size,
        )

    seqs = int(getattr(scheduler, "max_num_seqs", 0) or 0)
    if seqs > MULTI_RANK_MAX_NUM_SEQS:
        scheduler.max_num_seqs = MULTI_RANK_MAX_NUM_SEQS
        logger.info(
            "deepseek_v41: %d rank(s) — max_num_seqs %d -> %d. More "
            "concurrent sequences key more graph shapes and each loaded graph holds "
            "host memory for its code, DMA rings and scratchpad in every one of the "
            "%d worker processes; iteration 10 measured a graph load failing at "
            "Committed_AS 90.1 GiB of 114.8 with %d sequences at %d ranks. Every "
            "request is still served, one at a time; see "
            "deepseek_v41.MULTI_RANK_MAX_NUM_SEQS",
            world_size, seqs, MULTI_RANK_MAX_NUM_SEQS, world_size, seqs, world_size,
        )


#: Above this many KV slots the ceiling below declines to act, for the reason
#: `model._cap_kv_cache_blocks` gives: `num_gpu_blocks_override` is applied in
#: either direction, so a configuration whose working set is genuinely large
#: must be left to vLLM's memory-based sizing rather than handed a number that
#: does not fit. Kept equal to `model._KV_CAP_MAX_SLOTS`.
KV_CEILING_MAX_SLOTS = 131072


def require_attention_group(vllm_config) -> None:
    """Ask the backend for the 8-rank group the attention reduction needs.

    **The problem.** `config.ATTENTION_TENSOR_PARALLEL_SIZE` splits the
    attention 8 ways and replicates it over the remaining ranks, and the
    reduction at the end of `Attention.forward` was a collective over the
    *whole world*: 64 ranks summing 8 distinct partials eight times over, then
    dividing by 8. Correct — and 43 world-wide collectives a forward, plus 8
    more from the indexers, where 8-rank ones would do. At this rank count the
    collective count times the participant count is the forward's budget.

    **The group already exists.** `vllm_neuron.parallel.neuron_parallel_state`
    builds a 2D mesh when expert parallelism is configured: with `ep_degree = E`
    over a world of `W`, the *EP-TP* groups are `W / E` consecutive ranks each.
    At `W=64, E=8` that is eight groups of eight consecutive ranks —
    `[0..7], [8..15], ...` — and because `attn_rank` is `rank % 8`, **each such
    group holds exactly one of each attention shard.** So an EP-TP all-reduce is
    precisely the sum the attention wants, with no replica correction at all:
    one of each partial, summed once.

    Nothing else about expert parallelism is being asked for. This port shards
    its own weights (`model._build_shard_plan`) and partitions the experts by id
    across the full world itself, so the generic expert-parallel loaders and MoE
    modules are not on its path; what is wanted here is the mesh, and
    `ep_degree` is how the backend is told to build it.

    `enable_expert_parallel` has to be set because `resolve_ep_degree` refuses
    an explicit `ep_degree > 1` without it.

    Inert if the operator configured either themselves, and inert at a world
    size that does not exceed the attention split — there is no replication to
    undo then, and the world group is already the right one.
    """
    if not _serves_this_port(vllm_config):
        return
    parallel = getattr(vllm_config, "parallel_config", None)
    if parallel is None:
        return
    world_size = int(getattr(parallel, "world_size", 1) or 1)
    from .config import ATTENTION_TENSOR_PARALLEL_SIZE

    if world_size <= ATTENTION_TENSOR_PARALLEL_SIZE:
        return
    if getattr(parallel, "enable_expert_parallel", False):
        logger.info(
            "deepseek_v41: expert parallelism is already enabled; leaving the "
            "parallel mesh alone"
        )
        return

    additional = getattr(vllm_config, "additional_config", None)
    if additional is None:
        return
    neuron_config = dict(additional.get("neuron_config", {}) or {})
    if "ep_degree" in neuron_config:
        return

    # `ep_degree = world / attn_tp` is what makes an EP-TP group `attn_tp`
    # ranks wide, which is the group being asked for.
    ep_degree = world_size // ATTENTION_TENSOR_PARALLEL_SIZE
    parallel.enable_expert_parallel = True
    neuron_config["ep_degree"] = ep_degree
    additional["neuron_config"] = neuron_config
    logger.info(
        "deepseek_v41: ep_degree=%d so the EP-TP groups are %d consecutive "
        "ranks — the attention's %d-way split replicated %d times, which makes "
        "an EP-TP all-reduce exactly one of each partial instead of a %d-rank "
        "collective summing each eight times; see "
        "deepseek_v41.require_attention_group",
        ep_degree, ATTENTION_TENSOR_PARALLEL_SIZE,
        ATTENTION_TENSOR_PARALLEL_SIZE, ep_degree, world_size,
    )


def require_kv_cache_ceiling(vllm_config) -> None:
    """Cap the KV cache where the engine actually sizes it.

    `model._cap_kv_cache_blocks` has the arithmetic and the argument for it:
    vLLM sizes the cache from the memory free when it profiles, which is after
    the weights load and *before* any of this port's ~120 NEFFs are loaded, so
    it hands out a cache many times larger than the scheduler can fill and the
    graphs then have nowhere to go. The ceiling is one full `max_model_len` per
    sequence times the running batch — a ceiling, not a reduction below what
    correctness needs, since one sequence cannot reference more slots than its
    own context.

    **What is new here is the process it runs in, and it was a silent bug.**
    `_cap_kv_cache_blocks` runs from `load_weights`, i.e. in a *worker*, and
    sets `cache_config.num_gpu_blocks_override` on the worker's own
    `VllmConfig`. The code that reads that field — `get_kv_cache_config`,
    `vllm/v1/core/kv_cache_utils.py` — runs in the **EngineCore** process. At
    one rank those are the same process, because `world_size == 1` selects the
    in-process `uni` executor, so the mutation is visible and the cap works. At
    four ranks the workers are separate processes with unpickled copies, so the
    mutation goes nowhere and **the cap silently stops applying**. Measured on
    this host, a four-rank run of the verifier's own arguments:

        capping the KV cache at 1184 blocks of 32 (37888 slots)   <- the worker
        GPU KV cache size: 91,098 tokens                          <- EngineCore
        Maximum concurrency for 9,424 tokens per request: 9.67x

    — no `Overriding num_gpu_blocks=...` line at all, and a device-tensor
    census at layer 4 finding 43 cache tensors of `(2, 2523, 1, 32, 512)` where
    the cap had named 1184 blocks. That is 6.6 GiB of KV plus 1.1 GiB of
    index/window buffers against the 3.1 + 0.5 GiB the cap intends: **4.1 GiB
    of a 24 GiB bank**, which is most of what sharding the weights across four
    ranks had just freed.

    So the ceiling is declared here instead, from
    `NeuronPlatform.check_and_update_config`, which runs once in the process
    that builds the `VllmConfig` — before EngineCore, before the workers, and
    before anything reads the field. Workers get it by unpickling.
    `_cap_kv_cache_blocks` stays where it is and becomes a no-op when this has
    acted (it returns early if the override is already set), so a path that
    reaches `load_weights` without this hook still gets capped.

    Only ever lowers, only for this port, and inert if the operator set
    `num_gpu_blocks_override` themselves.
    """
    if not _serves_this_port(vllm_config):
        return
    cache = getattr(vllm_config, "cache_config", None)
    scheduler = getattr(vllm_config, "scheduler_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    if cache is None or scheduler is None or model_config is None:
        return
    if getattr(cache, "num_gpu_blocks_override", None) is not None:
        return

    # The block size the cache is **laid out** with, which is not necessarily
    # the one `cache_config` carries now: `update_block_size_for_backend` runs
    # later and settles on 32 unless the operator named one. Dividing by a
    # stale 16 is what made the worker-side cap realize half the reduction it
    # named, before iteration 5 fixed it; the same rule is applied here.
    block = int(getattr(cache, "block_size", None) or 32)
    if not getattr(cache, "user_specified_block_size", False):
        block = 32
    max_model_len = int(getattr(model_config, "max_model_len", 0) or 0)
    max_num_seqs = int(getattr(scheduler, "max_num_seqs", 0) or 0)
    if max_model_len <= 0 or max_num_seqs <= 0:
        return
    per_sequence = -(-max_model_len // block) + 1
    blocks = per_sequence * max_num_seqs
    if blocks * block > KV_CEILING_MAX_SLOTS:
        logger.info(
            "deepseek_v41: leaving the KV cache to vLLM — this "
            "configuration wants %d slots (%d per sequence x %d sequences), "
            "which is not obviously oversized",
            blocks * block, per_sequence, max_num_seqs,
        )
        return
    cache.num_gpu_blocks_override = blocks
    logger.info(
        "deepseek_v41: KV cache ceiling %d blocks of %d (%d slots) — "
        "%d per sequence x %d sequences, one full context each. Declared here "
        "rather than from load_weights because the engine sizes the cache in "
        "the EngineCore process, which a worker's mutation does not reach; see "
        "deepseek_v41.require_kv_cache_ceiling",
        blocks, block, blocks * block, per_sequence, max_num_seqs,
    )


#: Set to "1" to serve this checkpoint without speculative decoding. Only for
#: the A/B that proves speculation changes nothing: the same prompts with it on
#: and off have to produce identical tokens, and the only way to show that is to
#: run both. Speculation is **on by default** — the `mtp.*` stages are the
#: model's own drafter and they run on every forward either way, so declining to
#: propose their draft buys nothing but the verification step.
NO_SPEC_DECODE_ENV = "DEEPSEEK_V41_NO_SPEC_DECODE"


def require_dspark_speculation(vllm_config) -> None:
    """Give the engine the speculative config that turns the MTP draft into a
    proposal.

    **Why a hook and not an engine argument.** `speculative_config` is settled
    in `VllmConfig.__post_init__`, before any model class exists, so a caller
    who does not pass `--speculative-config` gets none — and the verifier does
    not pass one, deliberately: what is being checked is this port's own
    inference path, not an operator's ability to configure it. The platform's
    `check_and_update_config` is the one point a model package can speak before
    EngineCore and the workers, which is where `require_segmented_prefill` and
    `require_kv_cache_ceiling` already act from.

    `attach_speculative_config` builds it, checks that a proposer is registered
    for the method (so a missing `register_proposer` raises here instead of
    several minutes into the first forward), and redoes
    `_set_max_num_scheduled_tokens`, which `__post_init__` ran when there was no
    speculative config to account for.

    **What it costs, stated plainly**: `decode_token_threshold` becomes
    `1 + num_speculative_tokens`, so a decode step is padded to 6 tokens per
    request and keys decode graphs a non-speculative run never built. That is
    real compile time on the first run of a new shape set. What it buys is up to
    5 tokens per target forward, and on this port a forward streams 7.3 GiB of
    routed experts per layer — so a verified draft is close to free.

    **What it does to what the model emits, measured.** The draft is verified
    against the target's own logits by `vllm_neuron.nn.rejection_sampler` under
    greedy acceptance (`model.py`'s `_rejection_sample`), which keeps exactly
    the prefix the target would have produced unaided — so the *rule* is
    exact. The arithmetic is not yet: a verify step computes `1 + k` rows per
    sequence, and the torch sum-reductions still on the live path tile
    differently at a different row count. On the verifier's four prompts, same
    tree and same cache, 2 of 4 are byte-identical with speculation on and off
    and 2 flip one near-tied token each (`.autohelix/notes/iter-8.md` §3c).
    That is a bug with a known fix (iter-8 §3d), not a trade.

    Inert if the operator supplied their own `speculative_config`, and skipped
    entirely when `DEEPSEEK_V41_NO_SPEC_DECODE=1`.
    """
    if not _serves_this_port(vllm_config):
        return
    if getattr(vllm_config, "speculative_config", None) is not None:
        logger.info(
            "deepseek_v41: a speculative config is already set "
            "(method=%s); leaving it alone",
            getattr(vllm_config.speculative_config, "method", "?"),
        )
        return
    if os.environ.get(NO_SPEC_DECODE_ENV) == "1":
        logger.warning(
            "deepseek_v41: %s=1, so the mtp.* stages' draft is "
            "computed and not proposed. This is the control arm of the "
            "speculation A/B, not a configuration to serve in.",
            NO_SPEC_DECODE_ENV,
        )
        return

    from .config import DeepseekV41Config
    from .drafter import METHOD

    num_speculative_tokens = int(DeepseekV41Config.dspark_block_size)
    try:
        from vllm_neuron.vllm.spec_decode import attach_speculative_config
    except ImportError as exc:  # noqa: BLE001 - reported at error level, never silent
        logger.error(
            "deepseek_v41: this build of vllm_neuron.vllm.spec_decode "
            "has no attach_speculative_config (%s), so the DSpark draft cannot "
            "be proposed to the scheduler and inference will decode one token "
            "at a time. Everything else is unaffected.",
            exc,
        )
        return
    # Speculation here is synchronous, and that is a correctness requirement
    # rather than a preference.
    #
    # With `async_scheduling` on, `NeuronModelRunner.take_draft_token_ids` hands
    # the *scheduler* `[-1] * num_spec` placeholders instead of the drafts and
    # relies on the runner to fill the real tokens in from device futures before
    # anything reads them. Every path that does that filling in this build is
    # Eagle-specific: `_uses_async_eagle3_transition_cache` is literally
    # `use_async_scheduling and self.is_eagle3_spec`, and the
    # `num_computed_tokens` correction prologue needs a model decorated with
    # `@async_speculative_decoding`. This port is neither, so the drafts would
    # reach the device as placeholders and verify against nothing.
    #
    # There is also nothing here for async scheduling to overlap. Its point is
    # to run the host's bookkeeping under the *next* NEFF, and on this port a
    # forward streams gigabytes of routed experts off disk — the host work is
    # noise beside it — while the draft comes out of the target's own forward
    # rather than a second NEFF to pipeline against.
    #
    # The practical tell that the async path is not meant for an in-model
    # drafter: `propose` is handed the rejection output as a *device* tensor, and
    # every host-side read of it (including the runner's own bonus-token
    # extraction) then goes through the backend's eager fallback, which raised
    # `Expected self.dtype() == dst.dtype()` on the first verify step.
    scheduler = getattr(vllm_config, "scheduler_config", None)
    if scheduler is not None and getattr(scheduler, "async_scheduling", False):
        scheduler.async_scheduling = False
        logger.info(
            "deepseek_v41: async scheduling off — the DSpark draft is "
            "proposed synchronously; see "
            "deepseek_v41.require_dspark_speculation"
        )

    spec = attach_speculative_config(
        vllm_config, METHOD, num_speculative_tokens
    )
    logger.info(
        "deepseek_v41: speculative decoding through the mtp.* stages "
        "— method=%s, %d draft tokens per step, drafted by DSparkProposer from "
        "the stages this forward already runs; see "
        "deepseek_v41.require_dspark_speculation",
        getattr(spec, "method", METHOD), num_speculative_tokens,
    )


def declare_prefill_requirements() -> None:
    """Install `require_segmented_prefill` on the platform's config hook.

    `NeuronPlatform.check_and_update_config` is the one point where a vLLM
    platform adjusts a resolved `VllmConfig`, and it runs in the process that
    builds the config — before the engine core, the workers, or the model
    class. This wraps it rather than replacing it: the platform's own work runs
    first, unchanged, and the port's requirement is applied to the result.

    Idempotent, because this module is imported in the parent process (through
    `runpy`, ahead of `from vllm import LLM`), again when a config unpickles,
    and again in every worker through `vllm_neuron.model.registry`.
    """
    from vllm.platforms import current_platform

    platform_cls = type(current_platform)
    if getattr(platform_cls, "_deepseek_v41_prefill_hook", False):
        return
    original = platform_cls.check_and_update_config

    def check_and_update_config(cls, vllm_config) -> None:
        # Before `original`, not after: the platform's own body runs the
        # host-memory preflight, and the preflight reads
        # `NEURON_HOST_MEM_PER_RANK_GIB`. This is also the first point where the
        # *resolved* world size is known, so it catches a `--tensor-parallel-size`
        # that disagrees with `config.TENSOR_PARALLEL_SIZE`.
        apply_multi_rank_host_tuning(vllm_config.parallel_config.world_size)
        # Before `original`, like the host tuning: the platform's own body
        # resolves the parallel configuration, and the mesh this asks for has to
        # be part of what it resolves.
        require_attention_group(vllm_config)
        original(vllm_config)
        require_multi_rank_budget(vllm_config)
        require_segmented_prefill(vllm_config)
        require_kv_cache_ceiling(vllm_config)
        require_dspark_speculation(vllm_config)

    platform_cls.check_and_update_config = classmethod(check_and_update_config)
    platform_cls._deepseek_v41_prefill_hook = True


def register_draft_proposer() -> None:
    """Put `DSparkProposer` in `vllm_neuron.vllm.spec_decode.PROPOSERS`.

    So that `speculative_config.method == "deepseek_mtp"` builds this model's own MTP
    stages as the drafter instead of looking for a second checkpoint. Registering is
    cheap and side-effect free — `NeuronModelRunner` only reaches for the registry when
    a speculative config names the method — so it is done unconditionally here, where
    the rest of the port's hooks are installed.

    Registering is also what arms the `SpeculativeConfig` patch for a drafter with no
    draft *model* (`vllm_neuron/vllm/patches/in_model_proposer_patch.py`), so it has to
    happen before `require_dspark_speculation` builds the config — which it does, since
    that runs from the platform hook and this runs at import.

    The other half is `require_dspark_speculation`, which supplies the
    `speculative_config` that makes `NeuronModelRunner` build this class and call
    `propose`. Registering alone would leave the drafts computed and dropped.

    The import is local and guarded: `drafter` imports `config`, and this module is
    imported from `vllm_neuron.model.registry` during platform start-up, so an import
    error here would take registration down with it — which costs the port everything,
    for a feature nothing can call yet.
    """
    try:
        from vllm_neuron.vllm.spec_decode import register_proposer

        from .drafter import METHOD, DSparkProposer
    except ImportError as exc:  # noqa: BLE001 - reported, never silent
        logger.warning(
            "deepseek_v41: DSpark proposer not registered (%s). Speculative "
            "decoding through the mtp.* stages is unavailable; everything else is "
            "unaffected.",
            exc,
        )
        return
    register_proposer(METHOD, DSparkProposer)


#: The checkpoint's `architectures[0]`, and the key both registries use.
ARCH = "DeepseekV41ForCausalLM"

quiet_per_call_logging()
put_bundled_compiler_on_path()
warn_on_unsupported_islpy()
register_hf_config()
register_architecture()
declare_storage_quantization()
declare_prefill_requirements()
register_draft_proposer()
