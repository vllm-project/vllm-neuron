# SPDX-License-Identifier: Apache-2.0
"""The bridge from `model.py`'s modules to the two megakernels.

    run_layer(model, block, stream, pre_mix, step)   one `Block` = one `layer_forward`
    run_draft(model, seed, tap, main_hidden, span, paged)   `forward_spec` = one `draft_forward`

Weights are gathered once per block into the kernel's argument order
(`LayerWeights`), with the few re-layouts the kernels want (`fn.T` and `gate.T`
as fp32, the routed experts stacked on a leading axis) done on the host at first
use. Per forward, `StepIndices` turns the scheduler's block tables and slot
mapping into the kernels' index columns (`mk_host.build_attn_indices`) and carries
the state the layers hand down the stack: the current compressed/index caches,
the latest top-k, the DSpark taps.

The kernels do their own collectives, so nothing here touches a process group;
the replica groups are passed as constants (consecutive ranks).
"""

from __future__ import annotations

import functools
import glob
import hashlib
import logging
import os
import statistics
import time
from dataclasses import dataclass, field

import torch

from . import mk_host
from .kernels import mk_kernels

_HERE = os.path.dirname(os.path.abspath(__file__))
_mk_log = logging.getLogger(__name__)
#: `DEEPSEEK_V41_HOST_PROF=1`: time the host side of every forward (`HostProf`) and log a
#: summary on rank 0 every `DEEPSEEK_V41_HOST_PROF_EVERY` forwards (default 32).
_HOST_PROF = os.environ.get("DEEPSEEK_V41_HOST_PROF", "") not in ("", "0")
_HOST_PROF_SYNC = os.environ.get("DEEPSEEK_V41_HOST_PROF_SYNC", "") not in ("", "0")
_CPROFILE = os.environ.get("DEEPSEEK_V41_CPROFILE", "")


class HostProf:
    """Host time of each forward, by part (perf_counter, ms): `gap` since the previous
    forward returned (the engine's scheduling / output processing / input preparation),
    `pre` until the blocks start (span / slot metadata, index math, hashes, embedding),
    `blocks` the block loop -- `args` (each launch's arguments), `entry` (the compiled
    entry's call: dynamo's guards and the runtime's submission) and `post` (after it) --
    and `tail` after it (collapse, head, sampling, the drafter)."""

    def __init__(self, every: int):
        self.every, self.rows, self.cur, self.last_end = every, [], None, None

    def begin(self) -> None:
        t = time.perf_counter()
        gap = (t - self.last_end) * 1e3 if self.last_end is not None else float("nan")
        self.cur = {"t0": t, "gap": gap, "args": 0.0, "entry": 0.0, "post": 0.0, "launches": 0}

    def add(self, key: str, seconds: float) -> None:
        if self.cur is not None:
            self.cur[key] = self.cur.get(key, 0.0) + seconds * 1e3

    @torch._dynamo.disable
    def mark(self, key: str, sync=None) -> None:
        if self.cur is not None:
            if _HOST_PROF_SYNC and sync is not None:
                sync.cpu()
            self.cur[key] = (time.perf_counter() - self.cur["t0"]) * 1e3

    def end(self, step, rank: int) -> None:
        if self.cur is None:
            return
        t = time.perf_counter()
        c, self.cur, self.last_end = self.cur, None, t
        c["total"] = (t - c["t0"]) * 1e3
        c["phase"] = getattr(step, "phase", "?")
        c["T"] = getattr(step, "T", 0)
        if "blocks0" in c and "blocks1" in c:
            c["pre"], c["blocks"], c["tail"] = c["blocks0"], c["blocks1"] - c["blocks0"], c["total"] - c["blocks1"]
            if "head" in c and "sampled" in c:
                c["t_head"], c["t_sample"] = c["head"] - c["blocks1"], c["sampled"] - c["head"]
                c["t_draft"] = c["total"] - c["sampled"]
                if "collapsed" in c and "lshard" in c:
                    c["t_collapse"], c["t_lshard"] = c["collapsed"] - c["blocks1"], c["lshard"] - c["collapsed"]
                    c["t_gather"] = c["head"] - c["lshard"]
        self.rows.append(c)
        if len(self.rows) < self.every:
            return
        rows, self.rows = self.rows, []
        if rank != 0:
            return
        for phase in sorted({r["phase"] for r in rows}):
            rs = [r for r in rows if r["phase"] == phase]

            def med(k):
                v = [r[k] for r in rs if k in r and r[k] == r[k]]
                return statistics.median(v) if v else float("nan")

            _mk_log.warning(
                "host prof %s x%d (T=%s): gap %.2f | forward %.2f = pre %.2f + blocks %.2f (args %.2f, "
                "entry %.2f (first launch %.2f), post %.2f over %d launches) + tail %.2f (head %.2f, "
                "sampling %.2f, draft %.2f) ms (medians)",
                phase, len(rs), sorted({r["T"] for r in rs}), med("gap"), med("total"), med("pre"),
                med("blocks"), med("args"), med("entry"), med("entry_first"), med("post"),
                int(med("launches")), med("tail"), med("t_head"), med("t_sample"), med("t_draft"))
            if any("t_collapse" in r for r in rs):    # (the separate tail ran)
                _mk_log.warning("host prof %s head = collapse %.2f + head matmul %.2f + logits gather %.2f ms%s",
                                phase, med("t_collapse"), med("t_lshard"), med("t_gather"),
                                " (marks synced to the device)" if _HOST_PROF_SYNC else "")


PROF = HostProf(int(os.environ.get("DEEPSEEK_V41_HOST_PROF_EVERY", "32"))) if _HOST_PROF else None


class RpcProf:
    """`DEEPSEEK_V41_HOST_PROF`: the model runner's per-step calls (ms): how long each runs
    and how long the worker waits between them (the engine's scheduling and the RPC to the
    64 workers), summarized with `HostProf` on rank 0."""

    CALLS = ("execute_model", "sample_tokens", "take_draft_token_ids")

    def __init__(self, every: int):
        self.every, self.rows, self.cur, self.last_end = every, [], {}, None

    def wrap(self, cls, rank: int) -> None:
        self.rank = rank
        for name in self.CALLS:
            fn = getattr(cls, name, None)
            if fn is None or getattr(fn, "_rpc_prof", False):
                continue

            def timed(runner, *args, _fn=fn, _name=name, **kwargs):
                if _name == "execute_model" and _CPROFILE and self.rank == 0:
                    self.cprofile_tick()
                t0 = time.perf_counter()
                if _name == "execute_model" and self.cur:
                    # a new step: close the previous one
                    self.flush(self.cur)
                    self.cur = {}
                if self.last_end is not None:
                    key = f"idle>{_name}"
                    self.cur[key] = self.cur.get(key, 0.0) + (t0 - self.last_end) * 1e3
                try:
                    return _fn(runner, *args, **kwargs)
                finally:
                    t1 = time.perf_counter()
                    self.cur[_name] = self.cur.get(_name, 0.0) + (t1 - t0) * 1e3
                    self.last_end = t1

            timed._rpc_prof = True
            setattr(cls, name, timed)

    calls = 0
    cprof = None

    def cprofile_tick(self) -> None:
        """`DEEPSEEK_V41_CPROFILE=<start>:<count>`: profile execute_model calls
        start .. start + count - 1 and everything between them, then report."""
        import cProfile
        import io
        import pstats

        start, count = (int(v) for v in _CPROFILE.split(":"))
        self.calls += 1
        if self.calls == start:
            self.cprof = cProfile.Profile()
            self.cprof.enable()
        elif self.calls == start + count and self.cprof is not None:
            self.cprof.disable()
            self.cprof.dump_stats("/mnt/nvme/ds41_cprof_rank0.prof")
            for key in ("tottime", "cumulative"):
                buf = io.StringIO()
                pstats.Stats(self.cprof, stream=buf).sort_stats(key).print_stats(45)
                _mk_log.warning("cprofile over %d steps by %s:\n%s", count, key, buf.getvalue())
            self.cprof = None

    def flush(self, row) -> None:
        self.rows.append(row)
        if len(self.rows) < self.every:
            return
        rows, self.rows = self.rows, []
        if getattr(self, "rank", 0) != 0:
            return
        keys = sorted({k for r in rows for k in r})

        def med(k):
            v = [r[k] for r in rows if k in r]
            return statistics.median(v) if v else float("nan")

        _mk_log.warning("rpc prof x%d: %s (medians, ms)", len(rows),
                        " | ".join(f"{k} {med(k):.2f}" for k in keys))


RPC_PROF = RpcProf(int(os.environ.get("DEEPSEEK_V41_HOST_PROF_EVERY", "32"))) if _HOST_PROF else None


#: Zero tensors the launches read but no kernel writes (the top-k placeholder of a step's
#: first compressed layers, a ratio-0 block's 1-column top-k), one per shape: a fresh
#: `torch.zeros` on the device per launch was a device op of its own.
_ZEROS: dict = {}


def _zeros(shape: tuple, device) -> torch.Tensor:
    key = (shape, str(device))
    t = _ZEROS.get(key)
    if t is None:
        t = _ZEROS[key] = torch.zeros(*shape, dtype=torch.int32, device=device)
    return t


_IN_FLIGHT = int(os.environ.get("DEEPSEEK_V41_MK_IN_FLIGHT", "12"))
#: Small outputs of the megakernel launches submitted by this process and not yet
#: known complete, oldest first, **across forwards**: the runtime's submission queue
#: holds `NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS` (32) executions, and under
#: pipeline parallelism a stage's host launches batch after batch without ever
#: waiting on the device (the first stage with eight prefills in flight queued 64
#: executions and died with "Failed to schedule neff execution ... Execution Queue
#: Full", 2026-10-24), so the window has to span the steps, not sit inside one.
#: The compiled graphs around the kernels (the forward's tail, the drafter's
#: prologue / epilogue) sit on the same queue untracked: 12 tracked launches plus
#: a tail's worth of graphs stays under 32 now that a step drafts in one launch
#: (ten sequences drafting one by one, 3 untracked graphs each, overflowed at 12
#: on 2026-10-24; the window went to 8 for a while and cost the flat scheme ~2 ms
#: a token).
_pending: list = []


def _track(small: torch.Tensor) -> None:
    """Keep at most `_IN_FLIGHT` launches outstanding: read the oldest one's tiny
    output back, which blocks until that execution -- and every earlier one on the
    queue -- has completed. The device never drains: the host stays a dozen
    launches ahead of it."""
    _pending.append(small)
    if len(_pending) > _IN_FLIGHT:
        _pending.pop(0).cpu()
#: `DEEPSEEK_V41_MK_TRACE=1`: log every megakernel launch on each stage's first rank (which entry,
#: the step's phase and shape), to pin a device hang on one kernel.
_MK_TRACE = os.environ.get("DEEPSEEK_V41_MK_TRACE", "") not in ("", "0")
#: The scheme is read back from vLLM's parallel groups (`PipePlan.from_groups`):
#: "tp64" when the engine runs one tensor-parallel group, "tp8pp8" when it runs
#: `pipeline_parallel_size` stages of `tensor_parallel_size` ranks (`config.SCHEME`
#: sizes the engine arguments). Under "tp8pp8" each stage owns 5 blocks, builds and
#: loads only those, reduces over the stage, and hands its activations to the next
#: stage through `pp_kernels.boundary_body` (a 2-rank collective per tensor-parallel
#: offset); vLLM's scheduler keeps `pipeline_parallel_size` batches in flight.


@dataclass
class PipePlan:
    """Where this rank sits in the deployment scheme and which blocks it owns."""

    scheme: str
    world: int
    rank: int
    tp: int                     # ranks a stage (the world under tp64)
    pp: int                     # stages
    stage: int
    stage_rank: int
    block_stage: list           # stage of each backbone block

    @staticmethod
    def build(scheme: str, world: int, rank: int, n_layers: int, attn_tp: int) -> "PipePlan":
        if scheme == "tp64":
            return PipePlan(scheme, world, rank, world, 1, 0, rank, [0] * n_layers)
        pp = 8
        if world % pp:
            raise ValueError(f"tp8pp8 wants a world divisible by 8, got {world}")
        tp = world // pp
        if tp != attn_tp:
            raise ValueError(f"tp8pp8: a stage is {tp} ranks but the attention shards {attn_tp} ways")
        block_stage = [i * pp // n_layers for i in range(n_layers)]
        return PipePlan(scheme, world, rank, tp, pp, rank // tp, rank % tp, block_stage)

    @staticmethod
    def from_groups(n_layers: int) -> "PipePlan":
        """The plan vLLM's parallel groups imply: a stage is one tensor-parallel group
        (consecutive global ranks), the pipeline rank is the stage. Global rank ids are
        what the in-kernel collectives' replica groups name, so `rank` is global."""
        import torch.distributed as dist
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group

        tp_group, pp_group = get_tp_group(), get_pp_group()
        tp, pp = int(tp_group.world_size), int(pp_group.world_size)
        stage, stage_rank = int(pp_group.rank_in_group), int(tp_group.rank_in_group)
        world = tp * pp
        rank = stage * tp + stage_rank
        if dist.is_initialized() and dist.get_rank() != rank:
            raise RuntimeError(
                f"rank {dist.get_rank()}: vLLM placed it at tensor-parallel rank {stage_rank} of "
                f"pipeline stage {stage}, which is not the consecutive layout "
                f"(stage * tp + tp_rank = {rank}) the kernels' replica groups assume"
            )
        if pp == 1:
            return PipePlan("tp64", world, rank, world, 1, 0, rank, [0] * n_layers)
        scheme = "tp8pp8"
        if (tp, pp) != (8, 8):
            raise ValueError(f"the pipeline scheme is written for 8 stages x 8 ranks, got pp={pp} tp={tp}")
        block_stage = [i * pp // n_layers for i in range(n_layers)]
        return PipePlan(scheme, world, rank, tp, pp, stage, stage_rank, block_stage)

    def owns(self, layer_id: int) -> bool:
        return self.block_stage[layer_id] == self.stage

    def owned(self) -> list:
        """The blocks of this stage, in order."""
        return [i for i, st in enumerate(self.block_stage) if st == self.stage]

    @property
    def is_first(self) -> bool:
        return self.stage == 0

    @property
    def is_last(self) -> bool:
        return self.stage == self.pp - 1

    @property
    def first_block(self) -> int:
        return self.owned()[0]

    @property
    def last_block(self) -> int:
        return self.owned()[-1]

    def ends_stage(self, layer_id: int) -> bool:
        n = len(self.block_stage)
        return layer_id + 1 < n and self.block_stage[layer_id + 1] != self.block_stage[layer_id]

    @property
    def lead_last(self) -> int:
        """The rank that broadcasts the final stream: the first rank of the last stage."""
        return (self.pp - 1) * self.tp

    def groups_stage(self):
        return mk_host.replica_groups(self.world, self.tp)

    def groups_full(self):
        return mk_host.replica_groups(self.world, self.world)

    def boundary_groups(self, parity: int):
        """32 pairs `(8s + t, 8(s+1) + t)` for the sending stages `s` of this parity
        (the odd layout pairs stage 7 with stage 0: a pair where nobody sends)."""
        pairs = []
        for st in range(parity, self.pp, 2):
            nxt = (st + 1) % self.pp
            for t in range(self.tp):
                pairs.append((st * self.tp + t, nxt * self.tp + t))
        return tuple(sorted(pairs))

    def boundary_role(self, sending_stage: int) -> int:
        if self.stage == sending_stage:
            return 1
        if self.stage == sending_stage + 1:
            return 2
        return 0
#: `DEEPSEEK_V41_MK_DEBUG=<layer>:<stage>` runs that layer's kernel at the debug
#: `stage` (see `layer_kernel`'s docstring), saves its `dbg` output from rank 0 to
#: `DEEPSEEK_V41_DUMP_DIR/mk_layer<layer>_stage<stage>.pt` and stops the forward.
_MK_DEBUG = os.environ.get("DEEPSEEK_V41_MK_DEBUG", "")
#: `DEEPSEEK_V41_KCAPTURE=<dir>`: record one block kernel launch -- every argument
#: before the launch and every output after it -- on every rank, to
#: `<dir>/<tag>/L<layer>/rank<rank>.pt`, for standalone kernel work. Armed by
#: `<dir>/ARM.json` (`{"tag", "layers", "phase", "nth"}` plus any of `"start"`,
#: `"min_start"`, `"nspans"`, written by the driver while the server is idle): the
#: `nth` forward of `phase` after the arm whose first span starts at position `start`,
#: whose every span starts at or after `min_start` and which has `nspans` spans (each
#: filter only when given) is captured, at each of `layers`.
_KCAPTURE = os.environ.get("DEEPSEEK_V41_KCAPTURE", "")
_kcap = {"arm": None, "mtime": None, "step": None, "count": 0, "done": set()}


def _kcapture_target(model, layer_id: int, step) -> dict | None:
    """The arm spec when this launch is the one to record, else None."""
    path = os.path.join(_KCAPTURE, "ARM.json")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    if mtime != _kcap["mtime"]:
        import json

        with open(path) as fh:
            _kcap.update(arm=json.load(fh), mtime=mtime, step=None, count=0, done=set())
    arm = _kcap["arm"]
    starts = [int(s[2]) for s in step.spans]
    if (step.phase != arm["phase"]
            or (arm.get("start") is not None and starts[0] != int(arm["start"]))
            or (arm.get("min_start") is not None and min(starts) < int(arm["min_start"]))
            or (arm.get("nspans") is not None and len(starts) != int(arm["nspans"]))):
        return None
    if _kcap["step"] is not step:
        _kcap["step"] = step
        _kcap["count"] += 1
    if _kcap["count"] != int(arm["nth"]) or layer_id not in arm["layers"] or layer_id in _kcap["done"]:
        return None
    _kcap["done"].add(layer_id)
    return arm


def _kcapture_save(model, arm, kernel, layer_id: int, step, args, out) -> None:
    from .kernels import mk_kernels as _mkk

    names = _mkk._positional_names(_mkk.pkg(model.mk_plan.scheme).layer_kernel.layer_body)
    rec = {
        "kernel": kernel.func.__name__, "names": names, "args": args,
        "outs": [o.detach().cpu() for o in out],
        "meta": {"layer": layer_id, "rank": int(model.rank), "world": int(model.world_size),
                 "phase": step.phase, "spans": [tuple(int(v) for v in s) for s in step.spans],
                 "T": int(step.T), "arm": arm},
    }
    d = os.path.join(_KCAPTURE, arm["tag"], f"L{layer_id:02d}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"rank{int(model.rank):02d}.pt")
    torch.save(rec, path + ".tmp")
    os.replace(path + ".tmp", path)
    _mk_log.warning("kcapture: %s layer %d rank %d -> %s", kernel.func.__name__, layer_id, model.rank, d)


#: `DEEPSEEK_V41_FCAPTURE=<dir>`: record fused decode launches (`run_fused`) on every rank,
#: for standalone work on a fused megakernel -- the launch's parameters before it, its outputs,
#: and the rows it wrote into the caches it updates in place -- to
#: `<dir>/<tag>/<launch>/rank<rank>.pt` (with the launch's generated source and its members).
#: Armed by `<dir>/ARM.json` (`{"tag", "launches", "phase", "nth"}` plus any of `"min_start"`,
#: `"nspans"`, as for `_KCAPTURE`): the `nth` matching forward after the arm is captured at
#: each launch named in `launches`. Unlike `_KCAPTURE`, the decode step stays fused. Unset,
#: a launch only tests the flag.
_FCAPTURE = os.environ.get("DEEPSEEK_V41_FCAPTURE", "")
_fcap = {"arm": None, "mtime": None, "step": None, "count": 0, "done": set()}
#: The kernel arguments a block updates in place (its caches): their written rows are recorded.
_FCAPTURE_INPLACE = ("window_kv_cache", "compress_kv_cache", "index_k_cache", "pool_kv", "pool_score")


def _fcapture_target(fp, step) -> dict | None:
    """The arm spec when this fused launch is one to record, else None."""
    path = os.path.join(_FCAPTURE, "ARM.json")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    if mtime != _fcap["mtime"]:
        import json

        with open(path) as fh:
            _fcap.update(arm=json.load(fh), mtime=mtime, step=None, count=0, done=set())
    arm = _fcap["arm"]
    starts = [int(s[2]) for s in step.spans]
    if (step.phase != arm["phase"]
            or (arm.get("min_start") is not None and min(starts) < int(arm["min_start"]))
            or (arm.get("nspans") is not None and len(starts) != int(arm["nspans"]))):
        return None
    if _fcap["step"] is not step:
        _fcap["step"] = step
        _fcap["count"] += 1
    if _fcap["count"] != int(arm["nth"]) or fp.name not in arm["launches"] or fp.name in _fcap["done"]:
        return None
    _fcap["done"].add(fp.name)
    return arm


def _fcapture_members(fp) -> list:
    """`[(module, kernel name, positional names, [param index or ("o", m, k)] ...)]` of each
    member of the fused launch, from its generated source (`mk_kernels.fused`)."""
    import linecache
    import re

    g = fp.fk.func.__globals__
    lines = linecache.getlines(fp.fk.func.__code__.co_filename)
    members = []
    for line in lines:
        m = re.match(r"\s+o(\d+) = _k(\d+)\.func\((.*)\)\s*$", line)
        if not m:
            continue
        k = g[f"_k{m.group(2)}"]
        srcs = []
        for a in (x.strip() for x in m.group(3).split(",") if x.strip()):
            o = re.match(r"o(\d+)\[(\d+)\]", a)
            srcs.append(("o", int(o.group(1)), int(o.group(2))) if o else int(a[1:]))
        members.append((k.func.__module__, k.func.__name__, mk_kernels._positional_names(k.func), srcs))
    return members


def _fcapture_save(model, arm, fp, step, params, before, out) -> None:
    """Write this rank's record of the launch (see `_FCAPTURE`)."""
    import linecache

    members = _fcapture_members(fp)
    inplace = sorted({srcs[i] for _, _, names, srcs in members for i, n in enumerate(names)
                      if n in _FCAPTURE_INPLACE and isinstance(srcs[i], int)})
    # (the outputs first: their read-back waits for the launch, which updates the caches in place)
    outs = [o.detach().cpu() for o in out]
    written = {}
    for j in inplace:
        after = params[j].detach().cpu()
        rows = (after != before[j]).reshape(after.shape[0], -1).any(dim=1).nonzero().reshape(-1)
        written[j] = {"rows": rows, "values": after[rows].clone()}
    rec = {
        "launch": fp.name, "source": "".join(linecache.getlines(fp.fk.func.__code__.co_filename)),
        "members": members, "params": before, "outs": outs,
        "written": written,
        "meta": {"rank": int(model.rank), "world": int(model.world_size), "phase": step.phase,
                 "spans": [tuple(int(v) for v in s) for s in step.spans], "T": int(step.T),
                 "ntaps": fp.ntaps, "idx_member": fp.idx_member, "head": fp.head, "arm": arm},
    }
    d = os.path.join(_FCAPTURE, arm["tag"], fp.name)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"rank{int(model.rank):02d}.pt")
    torch.save(rec, path + ".tmp")
    os.replace(path + ".tmp", path)
    _mk_log.warning("fcapture: %s rank %d -> %s (%d params, %d in-place)", fp.name, model.rank, d,
                    len(params), len(inplace))


@functools.lru_cache(maxsize=None)
def kernel_version(kind: str = "block", scheme: str = "tp64") -> int:
    """A hash of the kernel source files a kernel is built from, folded into its
    constant `flags`: the NKI compile caches key a kernel on its *own* function
    source, so an edit to a helper module would otherwise be served a stale NEFF.
    The files are the scheme's own package (`kernels/<scheme>`: each scheme has
    its own copies of every megakernel module, so an edit for one scheme never
    touches the other's NEFFs). `kind` "block": the block kernel and the helper
    modules; "draft": the drafter and the helpers; "pp": the stage boundary and
    the helpers it imports. Each kind leaves out the other kinds' own modules, so
    an edit to the drafter or the boundary does not invalidate the 40 block
    kernels' NEFFs (and vice versa). An optimized block kernel's kind is its package
    ("r2_prefill", "r2_decode", "r1_prefill", "r1_decode": `mk_kernels.OPTIMIZED_KINDS`),
    which has its own helper copies in `kernels/<scheme>/<kind>/`; the hash covers that
    package only (`mk_kernels.version_kind`)."""
    h = hashlib.sha256()
    kdir = os.path.join(_HERE, "kernels", scheme)
    own = {"block": ("layer_kernel.py",), "draft": ("draft_kernel.py",), "pp": ("pp_kernels.py",),
           "head": ("head_kernel.py",)}
    if kind not in own and kind not in mk_kernels.OPTIMIZED_KINDS:
        raise ValueError(f"kernel_version: unknown kind {kind!r}")
    if not os.path.isdir(kdir):
        raise ValueError(f"kernel_version: no kernel package for scheme {scheme!r}")
    if kind not in own:
        files = sorted(glob.glob(os.path.join(kdir, kind, "*.py")))
        if not files:
            raise ValueError(f"kernel_version: no {kind} package for scheme {scheme!r}")
    else:
        files = sorted(glob.glob(os.path.join(kdir, "*.py")))
        others = {os.path.join(kdir, name) for k, names in own.items() if k != kind for name in names}
        files = [f for f in files if f not in others]
    files += [os.path.join(_HERE, "mk_host.py")]
    for path in files:
        with open(path, "rb") as fh:
            h.update(fh.read())
    return int.from_bytes(h.digest()[:4], "little") & 0x7FFFFFFF


@functools.lru_cache(maxsize=None)
def entry(kernel):
    """The compiled entry point of one specialized megakernel (`mk_kernels`), launched
    on both cores."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from . import model as M

    M._raise_dynamo_limits()
    wrapped = wrap_nki(kernel)[2]

    def run(*args):
        return wrapped(*args)

    run.__name__ = f"nki_{kernel.func.__name__}"
    return torch.compile(run, backend=M._compile_backend(), fullgraph=True, options=M._COMPILE_OPTIONS)


#: `DEEPSEEK_V41_DIRECT_LAUNCH` (default 1): a fused decode launch of the flat scheme
#: (`run_fused`) calls its NEFF's executor itself, with the same output tensors every step,
#: instead of going through the compiled entry (Dynamo's guards, AOTAutograd's runtime
#: wrapper and `_lite_execute`, which allocates fresh outputs). With the step's own inputs
#: in `_step_buf`s too, every IO tensor of the launch is where it was the step before, and
#: the runtime reuses the launch's DMA ring (`NEURON_RT_IO_RING_CACHE_SIZE=1` keeps one per
#: NEFF) instead of rebuilding it in `kbl_exec_pre`. Measured 2026-10-27 on 8 ranks, the
#: first fused launch (embedding + blocks 0-7, 492 IO tensors) with the same inputs every
#: call: 3.0 ms of host time a launch through the entry, 2.15 ms on the executor with fresh
#: outputs, 1.85 ms with the same outputs. 0: through the compiled entry.
_DIRECT = os.environ.get("DEEPSEEK_V41_DIRECT_LAUNCH", "1") not in ("", "0")


@dataclass
class _DirectLaunch:
    """A launch's NEFF executor, the parameter each NEFF input is, and the outputs every
    launch of it writes."""

    ex: object
    pmap: list
    outs: list


_UNSET = object()


def _calibrate(kernel, params):
    """Launch `kernel` through its compiled entry while watching the one executor call it
    makes (`native_backend._lite_execute`, swapped for the call). Returns `(out, direct)`:
    `direct` is None unless every NEFF input is one of `params` and the entry returns the
    executor's own outputs in order (no FX-pass IO mutation, no aliased output)."""
    from libtorch_neuronx_lite.compile import native_backend as NB

    seen = []
    lite_execute = NB._lite_execute

    def watch(artifact, inputs):
        out = lite_execute(artifact, inputs)
        seen.append((artifact, list(inputs), out))
        return out

    NB._lite_execute = watch
    try:
        out = entry(kernel)(*params)
    finally:
        NB._lite_execute = lite_execute
    if len(seen) != 1:
        return out, None
    art, ins, louts = seen[0]
    at = {id(x): j for j, x in enumerate(params) if isinstance(x, torch.Tensor)}
    pmap = [at.get(id(t)) for t in ins]
    if (not ins or None in pmap or getattr(art, "pass_mutations", None) or art.io_spec.aliases
            or len(out) != len(louts) or any(a is not b for a, b in zip(out, louts))):
        return out, None
    outs = [torch.empty(tuple(o.shape), dtype=o.dtype, device=ins[0].device) for o in art.io_spec.outputs]
    return out, _DirectLaunch(NB._executor_for(art), pmap, outs)


def _launch(kernel, launches: dict, slots, params):
    """`entry(kernel)(*params)`, or the same launch straight on its executor (`_DIRECT`):
    `launches` holds the plan's `_DirectLaunch`es (None: stays on the entry) by the shapes
    of the `params` the step fills (`slots`, `(index, ...)`), which pick the NEFF. The
    first launch of each goes through the entry (`_calibrate`). Returns `(outputs, direct)`."""
    dkey = tuple(tuple(params[j].shape) for j, *_ in slots)
    d = launches.get(dkey, _UNSET)
    if d is _UNSET:
        out, launches[dkey] = _calibrate(kernel, params)
        return out, False
    if d is None:
        return entry(kernel)(*params), False
    d.ex.execute([params[j] for j in d.pmap], d.outs)
    return d.outs, True


def _host(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu()


def _execute_context():
    """The execution hooks installed in libtorch_neuronx_lite (None normally): a direct
    launch would skip them, so `run_fused` keeps the entry while there are any."""
    from libtorch_neuronx_lite.compile.execute_context import get_execute_context

    return get_execute_context()


def _host_table(p) -> torch.Tensor:
    """A span's block table on the host: the copy `_paged_spans` kept, else read back."""
    return p.host_table if getattr(p, "host_table", None) is not None else _host(p.block_table)


def _host_slots(p) -> torch.Tensor:
    """A span's slots on the host: the copy `_paged_spans` kept, else read back."""
    return p.host_slots if getattr(p, "host_slots", None) is not None else _host(p.slot_mapping)


#: (name, shape, dtype, device) -> the device tensor a decode step's own host-built input
#: (an index column, an engram layer's hash ids, the token ids) is written into: the same
#: tensor every step, so a launch's IO tensors are where they were the step before and the
#: runtime can reuse the launch's DMA ring (see `_DIRECT`). Rewriting one in place is safe
#: under the flat scheme: a step starts after the previous one's launches completed (rank 0
#: read its tokens back, and every rank's launches finished before the drafter's collectives
#: on rank 0 could), and within a step each is written once, before the first launch that
#: reads it is submitted (`for_ratio` / `hash_cols` build a column on first use). Not under
#: the pipeline scheme, which keeps several batches in flight.
_STEP_BUFS: dict = {}


def _step_buf(name, t: torch.Tensor, device) -> torch.Tensor:
    """`t` (host) in its `_STEP_BUFS` tensor on `device`."""
    t = t.contiguous()
    key = (name, tuple(t.shape), t.dtype, str(device))
    buf = _STEP_BUFS.get(key)
    if buf is None:
        buf = _STEP_BUFS[key] = t.to(device)
    else:
        buf.copy_(t)
    return buf


def _dev(t: torch.Tensor, device) -> torch.Tensor:
    return t.contiguous().to(device)


def _row(t: torch.Tensor, device) -> torch.Tensor:
    """A 1-D buffer as the `[1, n]` row the kernels broadcast from."""
    return _dev(_host(t).reshape(1, -1), device)


def _t_f32(t: torch.Tensor, device) -> torch.Tensor:
    """`w.T` as fp32, for the helpers that stage a weight transposed."""
    return _dev(_host(t).t().float(), device)


def _freqs_table(freqs_cis, device) -> torch.Tensor:
    """`(cos, sin)` -> `[max_pos, 64]` fp32 (cos | sin)."""
    cos, sin = freqs_cis
    return _dev(torch.cat([_host(cos).float(), _host(sin).float()], -1), device)


@dataclass
class Dummies:
    """Placeholders for the arguments a layer archetype does not read."""

    i32: torch.Tensor
    u8: torch.Tensor
    bf16: torch.Tensor
    f32: torch.Tensor

    @staticmethod
    def on(device) -> "Dummies":
        return Dummies(
            i32=torch.zeros(1, 1, dtype=torch.int32, device=device),
            u8=torch.zeros(1, 1, dtype=torch.uint8, device=device),
            bf16=torch.zeros(1, 1, dtype=torch.bfloat16, device=device),
            f32=torch.zeros(1, 1, dtype=torch.float32, device=device),
        )


@dataclass
class LayerWeights:
    """One block's weights in `layer_forward`'s argument order, in three runs:
    `hc_attn` (before the index columns), `compress` (between the caches and the
    engram), `moe` (after the engram)."""

    hc_attn: list
    compress: list
    engram: list
    moe: list
    ratio: int
    is_kv_src: bool
    is_idx_src: bool
    has_engram: bool
    tap: bool
    window: int


def mx_fp8(weight: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """An fp8 `[n, k]` e4m3 weight with `[n/32, k/32]` e8m0 scales, re-laid for the
    tensor engine's MX matmul (`kernels/<scheme>/mk_mx.py`): `[k/4, n]` int32 words packing
    `k = 4p + j` in byte `j`, and `[k/32, n]` scale bytes (one per k-group and column).
    CPU in, CPU out."""
    n, k = weight.shape
    w = weight.reshape(n, k // 4, 4).permute(1, 0, 2).contiguous().view(torch.int32).reshape(k // 4, n)
    s = scale[: n // 32, : k // 32].t().repeat_interleave(32, 1).contiguous()
    return w, s


def mx_fp4(packed: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """An fp4 `[n, k/2]` e2m1-pair weight with `[n, k/32]` e8m0 scales: `[k/4, n]` int16
    words (nibble `j` = `k = 4p + j`) and `[k/32, n]` scale bytes."""
    n, k2 = packed.shape
    w = packed.reshape(n, k2 // 2, 2).permute(1, 0, 2).contiguous().view(torch.int16).reshape(k2 // 2, n)
    s = scale[:, : k2 // 16].t().contiguous()
    return w, s


def _fp8(linear) -> tuple[torch.Tensor, torch.Tensor]:
    """The MX layout of an fp8 linear's weight, converted once on the host and kept
    on the device beside the original (`linear._mx`)."""
    assert linear.storage == "fp8", linear.storage
    mx = getattr(linear, "_mx", None)
    if mx is None:
        w, s = mx_fp8(_host(linear.weight), _host(linear.scale))
        mx = (w.to(linear.weight.device), s.to(linear.weight.device))
        linear._mx = mx
    return mx


def _experts_mx(ffn) -> dict:
    """The resident expert pool in the MX layout: `w1`/`w3` `[E, dim/4, inter]` int16
    with `[E, dim/32, inter]` scales, `w2` `[E, inter/4, dim]` with `[E, inter/32, dim]`.
    Converted once; the original stacked buffers are released afterwards (the
    megakernel is the only reader of a resident pool), which keeps the device
    footprint where it was."""
    mx = getattr(ffn, "stacked_mx", None)
    if mx is not None:
        return mx
    stacked = ffn.stacked
    device = stacked["w1.weight"].device
    mx = {}
    for name in ("w1", "w2", "w3"):
        ws, ss = [], []
        wh, sh = _host(stacked[f"{name}.weight"]), _host(stacked[f"{name}.scale"])
        for e in range(wh.shape[0]):
            w, s = mx_fp4(wh[e], sh[e])
            ws.append(w)
            ss.append(s)
        mx[f"{name}.weight"] = torch.stack(ws).to(device)
        mx[f"{name}.scale"] = torch.stack(ss).to(device)
    ffn.stacked_mx = mx
    # release the per-op layout: the slot views first, then the stacks
    for slot in ffn.cache_slots:
        for name in ("w1", "w2", "w3"):
            linear = getattr(slot, name)
            linear.weight = None
            linear.scale = None
    ffn.stacked = {k: None for k in stacked}
    return mx


def layer_weights(model, block) -> LayerWeights:
    """Gather one block's weights (see `LayerWeights`)."""
    device = model.device
    attn, ffn = block.attn, block.ffn
    cfg = model.config
    layer_id = attn.layer_id
    d = model._mk_dummies
    hc_attn = [
        _t_f32(block.hc_attn.fn, device), _row(block.hc_attn.base, device), _row(block.hc_attn.scale, device),
        _row(block.attn_norm.weight, device),
        _t_f32(block.hc_ffn.fn, device), _row(block.hc_ffn.base, device), _row(block.hc_ffn.scale, device),
        _row(block.ffn_norm.weight, device),
        *_fp8(attn.wq_a), _row(attn.q_norm.weight, device), *_fp8(attn.wq_b), *_fp8(attn.wkv),
        _row(attn.kv_norm.weight, device), *_fp8(attn.wo_a), *_fp8(attn.wo_b),
        _dev(_host(attn.attn_sink).float().reshape(-1, 1), device),
        model._mk_freqs(attn),
    ]
    # Compressor / indexer weights. Layers that are not a source still need
    # tensors of the right rank; they get the nearest source's (unread).
    src = attn if attn.is_kv_source else model._mk_nearest_source(layer_id, "kv")
    isrc = attn if attn.is_index_source else model._mk_nearest_source(layer_id, "index")
    comp = src.compressor
    compress = [
        comp.wkv.weight, _t_f32(comp.wkv.weight, device),
        _t_f32(comp.wgate.weight, device) if comp.compress_ratio > 1 else d.f32,
        _row(comp.norm.weight, device),
    ]
    ix = isrc.indexer
    ixk = (src.indexer if src.indexer is not None and src.indexer.owns_k else model._mk_nearest_source(layer_id, "kv").indexer)
    index = [
        *_fp8(ix.wq_b), _t_f32(ix.weights_proj.weight, device), ixk.wk.weight, _row(ixk.k_norm.weight, device),
    ]
    engram = [d.bf16, d.i32, d.u8, d.u8, d.f32]
    if block.engram is not None:
        eg = block.engram
        if not eg.resident:
            raise RuntimeError(
                f"layer {layer_id}: the engram table is not resident on the device, and the "
                "megakernel reads it there. Raise DEEPSEEK_V41_ENGRAM_TABLE_BYTES."
            )
        engram = [
            # row_start 0: the ids reach the kernel already local to the shard (see
            # `StepIndices.hash_cols`).
            eg.table, torch.zeros(1, 1, dtype=torch.int32, device=device),
            *_fp8(eg.wkv),
            _dev((_host(eg.q_weight).float() * _host(eg.k_weight).float()).reshape(1, -1), device),
        ]
    stacked = getattr(ffn, "stacked", None)
    if (stacked is None and getattr(ffn, "stacked_mx", None) is None) or not ffn.resident:
        raise RuntimeError(
            f"layer {layer_id}: the routed experts are not resident as one stacked buffer "
            "(the expert pool streams). The megakernel needs them on the device; raise "
            "DEEPSEEK_V41_EXPERT_CACHE_BYTES."
        )
    stacked = _experts_mx(ffn)
    moe = [
        _t_f32(ffn.gate.weight, device), _row(ffn.gate.bias, device),
        torch.tensor([[float(ffn.experts_start_idx)]], dtype=torch.float32, device=device),
        stacked["w1.weight"], stacked["w1.scale"], stacked["w3.weight"], stacked["w3.scale"],
        stacked["w2.weight"], stacked["w2.scale"],
        *_fp8(ffn.shared_experts.w1), *_fp8(ffn.shared_experts.w3), *_fp8(ffn.shared_experts.w2),
    ]
    return LayerWeights(
        hc_attn=hc_attn, compress=compress + index, engram=engram, moe=moe,
        ratio=int(attn.compress_ratio), is_kv_src=bool(attn.is_kv_source),
        is_idx_src=bool(attn.is_index_source), has_engram=block.engram is not None,
        tap=layer_id in set(cfg.dspark_target_layer_ids), window=int(attn.window_size),
    )


@dataclass
class StepIndices:
    """Per-forward index columns and the state the layers hand down the stack."""

    spans: list
    paged: list
    device: torch.device
    hashes: torch.Tensor | None = None
    #: rows a span of a decode step has (1 committed + the draft block)
    decode_rows: int = 6
    #: a decode step writes its own device inputs into `_step_buf`s (the flat scheme's)
    stable: bool = False
    T: int = 0
    phase: str = "prefill"
    _by_ratio: dict = field(default_factory=dict)
    _hash_cols: dict = field(default_factory=dict)
    topk: torch.Tensor | None = None
    compress_cache: torch.Tensor | None = None
    index_cache: torch.Tensor | None = None
    taps: list = field(default_factory=list)
    #: `[T, 1]` int32: the step's greedy tokens, when its last fused launch ran the head
    greedy_ids: torch.Tensor | None = None
    #: `[T, 1]` int32: the step's input tokens, when its first fused launch embeds them
    embed_ids: torch.Tensor | None = None

    def __post_init__(self):
        self.T = sum(length for _, length, _ in self.spans)
        self.block_size = int(self.paged[0].block_size)
        self.tables = [_host_table(p).to(torch.int64) for p in self.paged]
        self.slot_mapping = torch.cat([_host_slots(p).reshape(-1) for p in self.paged]).to(torch.int64)
        lengths = {length for _, length, _ in self.spans}
        if len(lengths) != 1:
            raise NotImplementedError(
                f"the layer megakernel wants every span of a step to have the same length, got "
                f"{sorted(lengths)} (a mixed prefill/decode step). Serve with max_num_seqs=1."
            )
        self.phase = "decode" if lengths.pop() <= self.decode_rows else "prefill"
        self.stable = self.stable and self.phase == "decode"

    def read_len(self, ratio: int) -> int:
        if not ratio:
            return 0
        lens = {int(p.read_lens[ratio]) for p in self.paged}
        if len(lens) != 1:
            raise NotImplementedError(f"spans with different compressed read lengths {sorted(lens)}")
        return lens.pop()

    def for_ratio(self, ratio: int, window: int) -> mk_host.AttnIndices:
        key = (ratio, window)
        idx = self._by_ratio.get(key)
        if idx is None:
            idx = mk_host.build_attn_indices(
                self.spans, self.tables, self.slot_mapping, self.block_size, window,
                self.read_len(ratio), ratio, self.T,
            )
            for name in ("row_pos", "write_slots", "win_slots", "cmp_slots", "twin", "wv", "cbase",
                         "clen", "grp_pos", "grp_mslots", "grp_wslots"):
                t = getattr(idx, name)
                if t is not None:
                    setattr(idx, name, _step_buf(("idx", key, name), t, self.device) if self.stable
                            else t.to(self.device))
            self._by_ratio[key] = idx
        return idx

    def hash_cols(self, index: int, row_start: int) -> torch.Tensor:
        """This engram layer's `[T, 24]` int32 row ids, **local to this rank's shard**
        (`global - row_start`). The kernel's gather turns ids into fp32 for the range
        mask, and a global id of this 384M-row table is not exactly representable
        above 2^24; a local id is below the shard width (6M), so it is exact, and an
        out-of-range one only has to keep its sign / stay out of range."""
        key = (index, row_start)
        t = self._hash_cols.get(key)
        if t is None:
            if self.hashes is None:
                raise RuntimeError("an engram layer ran but the step has no n-gram hashes")
            t = (_host(self.hashes)[0, :, index, :].to(torch.int64) - int(row_start)).to(torch.int32)
            t = _step_buf(("hash", key), t, self.device) if self.stable else t.contiguous().to(self.device)
            self._hash_cols[key] = t
        return t


@torch._dynamo.disable
def make_step(spans, paged, device, hashes, decode_rows=6, stable=False) -> StepIndices:
    """`StepIndices` for this forward, built outside the frame dynamo traces
    (the index math is host work on CPU copies of the block tables). `stable`: the
    deployment runs one batch at a time (the flat scheme), so a decode step may write
    its own inputs into the `_step_buf`s."""
    return StepIndices(spans, paged, device, hashes, decode_rows=decode_rows, stable=stable)


#: layer id -> (window cache, its `[slots, head_dim]` view): the view, made once per cache
_WIN_VIEWS: dict = {}


def _window_view(attn) -> torch.Tensor:
    cache = attn.window_kv_cache
    got = _WIN_VIEWS.get(attn.layer_id)
    if got is None or got[0] is not cache:
        got = _WIN_VIEWS[attn.layer_id] = (cache, cache.view(-1, attn.head_dim))
    return got[1]


def layer_args(model, block, stream, pre_mix, step: StepIndices, stage: int = 9):
    """`(kernel, args, lw, arch)` of one block's launch: its kernel (`mk_kernels.layer`) and
    positional arguments. `stream` / `pre_mix` / `step.topk` are put in as given (a fused
    launch passes placeholders for the ones an earlier member makes). Records the kv
    source's caches in `step` for the blocks after it."""
    lw = model._mk_layer(block)
    d = model._mk_dummies
    T = int(step.T)
    B = len(step.spans)
    idx = step.for_ratio(lw.ratio, lw.window)
    attn = block.attn
    if lw.ratio:
        if step.topk is None:
            step.topk = _zeros((T, int(model.config.index_topk)), step.device)
        topk_in = step.topk
    else:
        topk_in = _zeros((T, 1), step.device)
    if lw.is_kv_src:
        step.compress_cache = attn.compress_kv_cache
        step.index_cache = attn.index_k_cache
        if lw.ratio == 2:
            pool_kv, pool_score = attn.compressor.pool_kv, attn.compressor.pool_score
        else:
            pool_kv, pool_score = d.f32, d.f32
        grp = [idx.grp_pos, idx.grp_mslots, idx.grp_wslots]
    else:
        pool_kv, pool_score = d.f32, d.f32
        grp = [d.i32, d.i32, d.i32]
    if lw.ratio and (step.compress_cache is None or step.index_cache is None):
        raise RuntimeError(f"layer {attn.layer_id} reads compressed KV before any kv-source layer ran")
    compress_cache = step.compress_cache if lw.ratio else d.bf16
    index_cache = step.index_cache if lw.ratio else d.bf16
    if lw.is_idx_src and attn.indexer.uses_candidates:
        # The candidate-block mask of layer 20 keeps the best `candidate_topk_blocks`
        # blocks of `candidate_block_size` groups; it only removes groups when the
        # context holds more than that, which the kernel does not model.
        limit = int(model.config.candidate_topk_blocks) * int(model.config.candidate_block_size)
        if step.read_len(lw.ratio) > limit:
            raise NotImplementedError(
                f"the context reads {step.read_len(lw.ratio)} compressed groups, more than the "
                f"{limit} the candidate-block mask keeps; the megakernel assumes the mask is a no-op"
            )
    plan = model.mk_plan
    hash_ids = d.i32
    engram = lw.engram
    # The engram read-out is in-kernel under both schemes: the table is sharded over
    # the ranks of `_mk_groups_world` (the world under tp64, the stage under tp8pp8).
    in_kernel_engram = lw.has_engram
    if in_kernel_engram:
        hash_ids = step.hash_cols(list(model.config.engram_layer_ids).index(attn.layer_id),
                                  int(block.engram.row_start))
    else:
        engram = [d.bf16, d.i32, d.u8, d.u8, d.f32]
    arch = (lw.ratio, int(lw.is_kv_src), int(lw.is_idx_src), int(in_kernel_engram), int(lw.tap))
    kver = kernel_version(mk_kernels.version_kind(step.phase, arch, plan.scheme), plan.scheme)
    flags = (lw.ratio, B, lw.window, stage, arch[1], arch[2], arch[3], arch[4], kver)
    args = [
        stream, pre_mix, *lw.hc_attn,
        idx.row_pos, idx.write_slots, idx.win_slots, idx.cmp_slots, idx.twin, idx.wv, idx.cbase, topk_in,
        _window_view(attn), compress_cache, pool_kv, pool_score, index_cache,
        *lw.compress[:4], *grp, *lw.compress[4:], idx.clen if lw.ratio else d.i32,
        hash_ids, *engram, *lw.moe,
        model._mk_groups_attn, model._mk_groups_world, flags,
    ]
    kernel = mk_kernels.layer(step.phase, arch, plan.scheme)
    return kernel, args, lw, arch


@torch._dynamo.disable
def run_layer(model, block, stream: torch.Tensor, pre_mix: torch.Tensor, step: StepIndices):
    """One backbone block through `layer_forward`. Returns `(stream, ffn_pre)`.

    Outside dynamo (`torch._dynamo.disable`, as `MoE.forward` is): the index
    columns are host work, and the kernel entry is its own `torch.compile`d
    function, which compiles from here as the per-op steps do."""
    t0 = time.perf_counter() if PROF is not None else 0.0
    attn = block.attn
    stage = 9
    debug = None
    if _MK_DEBUG:
        dbg_layer, dbg_stage = (int(v) for v in _MK_DEBUG.split(":"))
        if dbg_layer == attn.layer_id:
            stage, debug = dbg_stage, (dbg_layer, dbg_stage)
    kernel, args, lw, arch = layer_args(model, block, stream, pre_mix, step, stage)
    plan = model.mk_plan
    T, B = int(step.T), len(step.spans)
    if _MK_TRACE and plan.stage_rank == 0:
        # (the kind tells an optimized kernel from the generic body's entry of the same name)
        _mk_log.warning("mk launch %s [%s] layer=%d T=%d B=%d spans=%s", kernel.func.__name__,
                        mk_kernels.version_kind(step.phase, arch, plan.scheme), attn.layer_id, T, B,
                        step.spans)
    kcap = _kcapture_target(model, attn.layer_id, step) if _KCAPTURE else None
    if kcap is not None:
        # Copies taken before the launch: the kernel writes the caches in place.
        cap_args = [a.detach().cpu().clone() if isinstance(a, torch.Tensor) else a for a in args]
    if PROF is not None:
        t1 = time.perf_counter()
        PROF.add("args", t1 - t0)
    out = entry(kernel)(*args)
    if PROF is not None:
        t2 = time.perf_counter()
        PROF.add("entry", t2 - t1)
        PROF.add("launches", 1e-3)
    if kcap is not None:
        _kcapture_save(model, kcap, kernel, attn.layer_id, step, cap_args, out)
    if debug is not None:
        dump_dir = os.environ.get("DEEPSEEK_V41_DUMP_DIR", ".")
        if model.rank == 0:
            os.makedirs(dump_dir, exist_ok=True)
            path = os.path.join(dump_dir, f"mk_layer{debug[0]}_stage{debug[1]}.pt")
            torch.save({"dbg": out[0].cpu(), "stream_in": stream.cpu(), "pre_mix_in": pre_mix.cpu()}, path)
        raise RuntimeError(f"DEEPSEEK_V41_MK_DEBUG: layer {debug[0]} stage {debug[1]} dumped; stopping")
    out_stream, ffn_pre, topk_out, tap_out = out[0], out[1], out[2], out[3]
    # A forward submits one execution per layer with nothing on the host waiting on
    # any of them (the per-op path got a sync a layer from the router's `.cpu()`):
    # `_track` bounds the outstanding launches with the tiny `ffn_pre`.
    _track(ffn_pre)
    if lw.is_idx_src:
        step.topk = topk_out
    if lw.tap:
        step.taps.append(tap_out)
    if PROF is not None:
        PROF.add("post", time.perf_counter() - t2)
    return out_stream, ffn_pre


#: `DEEPSEEK_V41_FUSE_DECODE=<n>` (default 8): a decode step of the flat scheme launches each
#: run of n consecutive blocks as ONE fused kernel (`mk_kernels.fused`: the blocks'
#: optimized kernels in sequence) instead of one launch per block. Every launch of a NEFF
#: with collectives costs ~1.5 ms of host time in the runtime's submission, whatever its
#: size (measured: 1.5-1.8 ms with 1-16 all-reduces and 74-300 arguments, 0.25 ms without a
#: collective), and a block's decode kernel runs in ~0.6-0.9 ms: one launch per block made
#: the decode step host-bound (40 launches ~62 ms of a ~95 ms bs1 step). 0 or 1: one launch
#: a block. Prefill keeps one launch a block (device-bound; fusing would multiply compiles).
_FUSE_DECODE = int(os.environ.get("DEEPSEEK_V41_FUSE_DECODE", "8"))
#: `DEEPSEEK_V41_FUSE_GROUPS=<n1>,<n2>,...` (default empty): the first fused launches of a
#: decode step take n1, n2, ... blocks, the rest `_FUSE_DECODE` each. Only the first launch's
#: submission delays the device (the later ones are submitted while it runs), so a smaller
#: first launch starts the device sooner, as long as the second is submitted before the
#: first finishes. Each grouping is its own set of fused NEFFs.
_FUSE_GROUPS = tuple(int(v) for v in os.environ.get("DEEPSEEK_V41_FUSE_GROUPS", "").split(",") if v.strip())


def _fuse_groups(n: int) -> list:
    """`[(start, end))` of each fused launch of a decode step's `n` blocks."""
    out, g = [], 0
    for size in _FUSE_GROUPS:
        if g >= n:
            break
        out.append((g, min(n, g + size)))
        g += size
    while g < n:
        out.append((g, min(n, g + _FUSE_DECODE)))
        g += _FUSE_DECODE
    return out


#: `DEEPSEEK_V41_FUSE_HEAD` (default 1): an all-greedy speculative decode step runs the
#: head -- collapse, final norm, vocab-parallel head and the argmax -- as the last member
#: of its last fused launch (`mk_kernels.head_greedy`) instead of five separate launches,
#: two `[T, vocab]` all-gathers and the plugin sampler's own collectives. 0: separately.
_FUSE_HEAD = os.environ.get("DEEPSEEK_V41_FUSE_HEAD", "1") not in ("", "0")


def head_args(model) -> list:
    """The fused head's arguments after the stream / pre_mix: the final norm's row, this
    rank's head rows, its first vocabulary column, the world's groups and the flags."""
    got = model.__dict__.get("_mk_head_args")
    if got is None:
        device = model.head_weight.device
        got = model.__dict__["_mk_head_args"] = [
            _row(model.norm.weight, device), model.head_weight,
            torch.tensor([[int(model.rank * model.part_vocab_size)]], dtype=torch.int32, device=device),
            model._mk_groups_full, (int(model.config.vocab_size), kernel_version("head", "tp64")),
        ]
    return got


#: `DEEPSEEK_V41_FUSE_EMBED` (default 1): a decode step's first fused launch embeds the
#: step's tokens and expands them to the hyper-connection copies itself
#: (`mk_kernels.embed_expand`) instead of the embedding's all-reduce graph and the
#: `hc_expand` launch, both between the step's start and its first block. 0: separately.
_FUSE_EMBED = os.environ.get("DEEPSEEK_V41_FUSE_EMBED", "1") not in ("", "0")


def fuses_embed(model, spans) -> bool:
    """Whether this step's first fused launch embeds its tokens (`run_blocks`): a decode
    step of the flat scheme (`StepIndices.phase`'s rule, from the spans)."""
    lengths = {length for _, length, _ in spans}
    if len(lengths) != 1:
        return False
    length = lengths.pop()
    return (_FUSE_EMBED and _FUSE_DECODE > 1 and length <= 1 + int(model.config.dspark_block_size)
            and length * len(spans) <= 128 and model.mk_plan.scheme == "tp64"
            and not _MK_DEBUG and not _KCAPTURE)


@torch._dynamo.disable
def host_ids(input_ids: torch.Tensor) -> torch.Tensor:
    """The step's `[T]` token ids read back to the host (outside dynamo: traced, the copy
    would be part of a graph and stay on the device)."""
    return input_ids.detach().reshape(-1).cpu()


@torch._dynamo.disable
def embed_ids(input_ids: torch.Tensor, host: torch.Tensor | None = None) -> torch.Tensor:
    """The step's tokens as the `[T, 1]` int32 column the fused embedding reads (through
    the host: the device is idle at a step's start, and a dtype cast on it would be a
    graph of its own), in its `_step_buf` (a fused decode step of the flat scheme is the
    only caller). `host`: the tokens already read back."""
    if host is None:
        host = host_ids(input_ids)
    return _step_buf(("ids",), host.reshape(-1).to(torch.int32).reshape(-1, 1), input_ids.device)


def embed_args(model) -> list:
    """The fused embedding's arguments after the ids: this rank's embedding rows, its
    first vocabulary row, the world's groups and the flags."""
    got = model.__dict__.get("_mk_embed_args")
    if got is None:
        got = model.__dict__["_mk_embed_args"] = [
            model.embed_weight, head_args(model)[2], model._mk_groups_full,
            (int(model.config.hc_mult), kernel_version("head", "tp64")),
        ]
    return got


def fuses_head(model, step) -> bool:
    """Whether this step's blocks can end with the fused greedy head (`run_blocks`)."""
    return (_FUSE_HEAD and _FUSE_DECODE > 1 and step.phase == "decode" and step.T <= 128
            and model.mk_plan.scheme == "tp64" and not _MK_DEBUG and not _KCAPTURE)


class _Out:
    """Output `k` of member `m` of a fused launch being planned (`("o", m, k)`)."""

    __slots__ = ("m", "k")

    def __init__(self, m: int, k: int):
        self.m, self.k = m, k


#: the `StepIndices.for_ratio` columns a launch can read (`mk_host.AttnIndices`)
_IDX_FIELDS = ("row_pos", "write_slots", "win_slots", "cmp_slots", "twin", "wv", "cbase",
               "clen", "grp_pos", "grp_mslots", "grp_wslots")
_PLAN_CHECK = os.environ.get("DEEPSEEK_V41_FUSED_PLAN_CHECK", "") not in ("", "0")


@dataclass
class FusedPlan:
    """A fused launch as `run_fused` planned it once: the kernel, its parameters with the
    step's own tensors as `slots` (`(index, kind, key)`: "in" the incoming stream (0) /
    pre_mix (1), "topk" the step's top-k so far, "idx" a `for_ratio(*key[0])` column,
    "hash" a `hash_cols(*key)` table), what the launch returns, and the step state the
    blocks leave for later launches (the kv source's caches, a placeholder top-k)."""

    fk: object
    name: str
    params: list
    slots: list
    ntaps: int
    idx_member: int | None
    head: bool
    after: tuple
    trace: str
    #: the shapes of the step's own tensors -> `_DirectLaunch` / None (`_launch`)
    direct: dict = field(default_factory=dict)


#: (layer ids, phase, T, B, head) -> FusedPlan
_FUSED_PLANS: dict = {}


def _plan_fused(model, blocks, stream, pre_mix, step: StepIndices, head: bool,
                embed: bool = False) -> FusedPlan:
    """`run_fused`'s plan, from `layer_args` (see `FusedPlan`). With `embed` member 0 is
    the embedding (the blocks are members 1..), which makes the first block's stream."""
    plan = model.mk_plan
    zeros = {id(z) for z in _ZEROS.values()}
    topk0 = step.topk if isinstance(step.topk, torch.Tensor) and id(step.topk) not in zeros else None
    kernels, plans, params, pidx = [], [], [], {}
    taps_at, idx_member = [], None
    s, pm = stream, pre_mix
    off = 0
    if embed:
        srcs = []
        for x in [step.embed_ids, *embed_args(model)]:
            key = ("t", id(x)) if isinstance(x, torch.Tensor) else ("c", x)
            if key not in pidx:
                pidx[key] = len(params)
                params.append(x)
            srcs.append(("p", pidx[key]))
        kernels.append(mk_kernels.embed_expand())
        plans.append(tuple(srcs))
        s, pm, off = _Out(0, 0), _Out(0, 1), 1
    for m0, block in enumerate(blocks):
        m = m0 + off
        kernel, args, lw, arch = layer_args(model, block, s, pm, step)
        srcs = []
        for x in args:
            if isinstance(x, _Out):
                srcs.append(("o", x.m, x.k))
                continue
            key = ("t", id(x)) if isinstance(x, torch.Tensor) else ("c", x)
            if key not in pidx:
                pidx[key] = len(params)
                params.append(x)
            srcs.append(("p", pidx[key]))
        kernels.append(kernel)
        plans.append(tuple(srcs))
        if lw.is_idx_src:
            step.topk = _Out(m, 2)
            idx_member = m
        if lw.tap:
            taps_at.append(m)
        s, pm = _Out(m, 0), _Out(m, 1)
    n = len(blocks) + off          # (the members so far; the last block is member n - 1)
    if head:
        srcs = [("o", n - 1, 0), ("o", n - 1, 1)]
        for x in head_args(model):
            key = ("t", id(x)) if isinstance(x, torch.Tensor) else ("c", x)
            if key not in pidx:
                pidx[key] = len(params)
                params.append(x)
            srcs.append(("p", pidx[key]))
        kernels.append(mk_kernels.head_greedy())
        plans.append(tuple(srcs))
    returns = [("o", n - 1, 0), ("o", n - 1, 1)] + [("o", m, 3) for m in taps_at]
    if idx_member is not None:
        returns.append(("o", idx_member, 2))
    if head:
        returns.append(("o", n, 0))
    ids = [b.attn.layer_id for b in blocks]
    name = (f"layers_{step.phase}{'_embed' if embed else ''}_L{ids[0]:02d}_L{ids[-1]:02d}"
            f"{'_head' if head else ''}_{plan.scheme}")
    fk = mk_kernels.fused(name, tuple(kernels), tuple(plans), len(params), tuple(returns))
    # the step's own tensors: refilled every step
    dyn = {}
    if embed:
        dyn[id(step.embed_ids)] = ("ids", None)
    else:
        dyn.update({id(stream): ("in", 0), id(pre_mix): ("in", 1)})
    if topk0 is not None:
        dyn[id(topk0)] = ("topk", None)
    for rkey, idx in step._by_ratio.items():
        for f in _IDX_FIELDS:
            t = getattr(idx, f)
            if t is not None:
                dyn.setdefault(id(t), ("idx", (rkey, f)))
    for hkey, t in step._hash_cols.items():
        dyn.setdefault(id(t), ("hash", hkey))
    slots = [(j, *dyn[id(x)]) for j, x in enumerate(params) if isinstance(x, torch.Tensor) and id(x) in dyn]
    static = [None if isinstance(x, torch.Tensor) and id(x) in dyn else x for x in params]
    zeros = {id(z) for z in _ZEROS.values()}       # (the blocks may have made one)
    after = (step.compress_cache, step.index_cache,
             step.topk if isinstance(step.topk, torch.Tensor) and id(step.topk) in zeros else None)
    trace = ",".join(k.func.__module__.rsplit(".", 2)[-2] for k in kernels)
    return FusedPlan(fk, name, static, slots, len(taps_at), idx_member, head, after, trace)


def _fill_fused(fp: FusedPlan, stream, pre_mix, step: StepIndices):
    """`fp`'s parameters for this step, or None when a slot has nothing this step."""
    params = list(fp.params)
    for j, kind, key in fp.slots:
        if kind == "in":
            x = stream if key == 0 else pre_mix
        elif kind == "ids":
            x = step.embed_ids
        elif kind == "topk":
            x = step.topk
        elif kind == "idx":
            x = getattr(step.for_ratio(*key[0]), key[1])
        else:
            x = step.hash_cols(*key)
        if not isinstance(x, torch.Tensor):
            return None
        params[j] = x
    return params


@torch._dynamo.disable
def run_fused(model, blocks, stream: torch.Tensor, pre_mix: torch.Tensor, step: StepIndices,
              head: bool = False, embed: bool = False):
    """`blocks` (consecutive) as one fused launch. Returns `(stream, ffn_pre)` of the last;
    with `head`, the greedy head runs after them in the same launch (`step.greedy_ids`).

    Each block's arguments are `layer_args`' with the stream / pre_mix / top-k an earlier
    member makes as that member's output; every distinct tensor is one parameter (a kv
    source's caches, the step's index columns and the dummies are shared), so a write in
    one member is ordered before the reads in the next. The launch returns the last
    stream / pre_mix, the taps and the last index source's top-k. The plan is made once
    per (blocks, phase, T, B, head) (`FusedPlan`); a step refills the step's own tensors."""
    t0 = time.perf_counter() if PROF is not None else 0.0
    plan = model.mk_plan
    key = (tuple(b.attn.layer_id for b in blocks), step.phase, int(step.T), len(step.spans), bool(head),
           bool(embed))
    fp = _FUSED_PLANS.get(key)
    params = None if fp is None else _fill_fused(fp, stream, pre_mix, step)
    if params is None or _PLAN_CHECK:
        topk_in = step.topk
        built = _plan_fused(model, blocks, stream, pre_mix, step, head, embed)
        step.topk = topk_in              # (the plan left its members' placeholders there)
        fresh = _fill_fused(built, stream, pre_mix, step)
        if params is not None:
            same = (len(params) == len(fresh) and built.fk is fp.fk
                    and all(a is b or (not isinstance(a, torch.Tensor) and not isinstance(b, torch.Tensor)
                                       and a == b) for a, b in zip(params, fresh)))
            if not same:
                raise RuntimeError(f"DEEPSEEK_V41_FUSED_PLAN_CHECK: the cached plan of {fp.name} "
                                   f"differs from this step's ({len(params)} vs {len(fresh)} parameters)")
        fp, params = built, fresh
        _FUSED_PLANS[key] = fp
    cc, ic, zt = fp.after
    step.compress_cache, step.index_cache = cc, ic
    if zt is not None and step.topk is None:
        step.topk = zt
    if _MK_TRACE and plan.stage_rank == 0:
        _mk_log.warning("mk launch %s [%s] T=%d B=%d spans=%s", fp.name, fp.trace, step.T,
                        len(step.spans), step.spans)
    if PROF is not None:
        t1 = time.perf_counter()
        PROF.add("args", t1 - t0)
    fcap = _fcapture_target(fp, step) if _FCAPTURE else None
    if fcap is not None:
        before = [x.detach().cpu().clone() if isinstance(x, torch.Tensor) else x for x in params]
    if _DIRECT and step.stable and _execute_context() is None:
        out, direct = _launch(fp.fk, fp.direct, fp.slots, params)
    else:
        out, direct = entry(fp.fk)(*params), False
    if PROF is not None:
        t2 = time.perf_counter()
        PROF.add("entry", t2 - t1)
        PROF.add("launches", 1e-3)
        if embed:
            PROF.add("entry_first", t2 - t1)
    stream, pre_mix = out[0], out[1]
    step.taps.extend(out[2 : 2 + fp.ntaps])
    if fp.idx_member is not None:
        step.topk = out[2 + fp.ntaps]
    if fp.head:
        step.greedy_ids = out[-1]
    if fcap is not None:
        _fcapture_save(model, fcap, fp, step, params, before, out)
    if not direct:
        # (a direct launch writes the same outputs every step: reading one back later
        # would wait for whichever launch wrote it last; the step's own read-back of its
        # tokens bounds those launches)
        _track(pre_mix)
    if PROF is not None:
        PROF.add("post", time.perf_counter() - t2)
    return stream, pre_mix


def _draft_meta(seeds, taps, pos):
    """`[B, 3]` int32 (seed, tap, pos) per sequence, as a compiled step: the seeds
    and taps come out of the rejection sampler on the device, so this stays
    there (no read-back before the draft launch)."""
    return torch.stack([seeds.reshape(-1), taps.reshape(-1), pos.reshape(-1)], dim=-1).to(torch.int32)


def draft_consts(B: int, K: int, ntab_p: int) -> torch.Tensor:
    """`[B*K, 4]` int32 per draft row: (i, b, b * ntab_p, i * B + b) -- the row's
    token index, its sequence, its sequence's base row in the kernel's slot table
    and its markov embedding row (CPU)."""
    r = torch.arange(B * K)
    i, b = r % K, r // K
    return torch.stack([i, b, b * ntab_p, i * B + b], dim=-1).to(torch.int32).contiguous()


def draft_weights(model) -> list:
    """The drafter's weights in `draft_forward`'s argument order (after the index
    columns, before the caches), plus the head's, as two lists."""
    device = model.norm.weight.device
    stages = list(model.mtp)
    s0, last = stages[0], stages[-1]

    def stack(get):
        # Through the host: a one-time copy, and no eager op on the device.
        return _dev(torch.stack([_host(get(s)) for s in stages]), device)

    stack_host = stack

    def stack8(get):
        """The stages' fp8 linears in the MX layout, stacked: `[3, K/4, N]`, `[3, K/32, N]`."""
        pairs = [_fp8(get(s)) for s in stages]
        return (_dev(torch.stack([_host(w) for w, _ in pairs]), device),
                _dev(torch.stack([_host(sc) for _, sc in pairs]), device))

    head = [
        *_fp8(s0.main_proj), _row(s0.main_norm.weight, device), model.embed_weight,
        torch.tensor([[int(model.rank * model.part_vocab_size)]], dtype=torch.int32, device=device),
    ]
    hc = [
        *[_t_f32(s.hc_attn.fn, device) for s in stages],
        stack_host(lambda s: s.hc_attn.base), stack_host(lambda s: s.hc_attn.scale),
        stack(lambda s: s.attn_norm.weight),
        *[_t_f32(s.hc_ffn.fn, device) for s in stages],
        stack_host(lambda s: s.hc_ffn.base), stack_host(lambda s: s.hc_ffn.scale),
        stack(lambda s: s.ffn_norm.weight),
        *stack8(lambda s: s.attn.wq_a), stack(lambda s: s.attn.q_norm.weight),
        *stack8(lambda s: s.attn.wq_b),
        *stack8(lambda s: s.attn.wkv), stack(lambda s: s.attn.kv_norm.weight),
        *stack8(lambda s: s.attn.wo_a),
        *stack8(lambda s: s.attn.wo_b),
        stack_host(lambda s: s.attn.attn_sink.float().reshape(-1, 1)),
        model._mk_freqs(s0.attn),
    ]
    for s in stages:
        if (getattr(s.ffn, "stacked", None) is None and getattr(s.ffn, "stacked_mx", None) is None) or not s.ffn.resident:
            raise RuntimeError("the DSpark stages' routed experts are not resident as stacked buffers")
    mxs = [_experts_mx(s.ffn) for s in stages]
    moe = [
        *[_t_f32(s.ffn.gate.weight, device) for s in stages],
        stack_host(lambda s: s.ffn.gate.bias),
        torch.tensor([[float(s0.ffn.experts_start_idx)]], dtype=torch.float32, device=device),
        *[_dev(torch.stack([_host(m[k]) for m in mxs]), device)
          for k in ("w1.weight", "w1.scale", "w3.weight", "w3.scale", "w2.weight", "w2.scale")],
        *stack8(lambda s: s.ffn.shared_experts.w1), *stack8(lambda s: s.ffn.shared_experts.w3),
        *stack8(lambda s: s.ffn.shared_experts.w2),
    ]
    tail = [
        _row(last.norm.weight, device), model.head_weight,
        last.markov_head.embed_weight, last.markov_head.head_weight, last.confidence_head.proj_weight,
    ]
    return head + hc + moe, tail


def _draft_meta_seq(seed, tap, pos):
    """`[1, 3]` int32 (seed, tap, pos) of one sequence, as a compiled step (the flat
    scheme's drafter prologue, unchanged)."""
    return torch.stack([seed.reshape(-1)[:1], tap.reshape(-1)[:1], pos.reshape(-1)[:1]]).to(torch.int32).reshape(1, 3)


def _run_draft_seq(model, seed_ids, tap, main_hidden, span, paged):
    """The flat scheme's drafter launch for one sequence (`draft_kernel_seq`): the
    path as it was before the pipeline work, so `tp64` keeps its behaviour."""
    from . import model as M

    cfg = model.config
    device = main_hidden.device
    begin, length, start_pos = span
    main = main_hidden[begin : begin + length]
    meta = M._compiled(_draft_meta_seq)(seed_ids, tap, paged.pos)
    _track(meta)
    table = _host_table(paged).to(torch.int32).reshape(-1, 1).contiguous().to(device)
    row_pos = (start_pos + torch.arange(length)).to(torch.int32).reshape(-1, 1).to(device)
    write_slots = _host_slots(paged).reshape(-1)[:length].to(torch.int32).reshape(-1, 1).contiguous().to(device)
    weights, tail = model._mk_draft()
    K = int(cfg.dspark_block_size)
    stages = list(model.mtp)
    flags = (int(paged.at_zero), K, int(stages[0].attn.window_size), 9, kernel_version("draft", model.mk_plan.scheme),
             int(cfg.dspark_noise_token_id), int(cfg.vocab_size), int(cfg.dspark_num_experts_per_tok),
             int(paged.block_size))
    caches = [s.attn.window_kv_cache.view(-1, s.attn.head_dim) for s in stages]
    args = [meta, main, row_pos, write_slots, table, *weights, *caches, *tail,
            model._mk_groups_attn, model._mk_groups_full, flags]
    kernel = mk_kernels.draft("prefill" if paged.at_zero else "decode", model.mk_plan.scheme)
    if _MK_TRACE and model.mk_plan.stage_rank == 0:
        _mk_log.warning("mk launch %s span=%s", kernel.func.__name__, span)
    out = entry(kernel)(*args)
    output_ids, confidence = out[0], out[1]
    _track(confidence)
    return output_ids, None, confidence.reshape(1, K)


#: `DEEPSEEK_V41_DRAFT_BATCHED` (default 1): the flat scheme drafts every sequence of a step
#: of several in one launch (`mk_kernels.draft_batched`) instead of one launch per sequence
#: -- ten drafter launches a step at bs10, each its own NEFF execution (~1.5 ms of host
#: submission with collectives) and its own collective-bound device time. A step of one
#: sequence keeps the per-sequence kernel (measured 2026-10-27: bs10 TPOT 49.5 -> 41.1 ms
#: batched, bs1 17.7 -> 18.1 ms). 0: one launch per sequence always.
_DRAFT_BATCHED = os.environ.get("DEEPSEEK_V41_DRAFT_BATCHED", "1") not in ("", "0")


@torch._dynamo.disable
def run_draft(model, seeds, taps, main_hidden, spans, paged):
    """`_draft` for every sequence of the step: `[(output_ids [1, K+1] int32, None,
    confidence [1, K] fp32)]`, one per span; a batched launch also leaves its
    `[B, K+1]` ids in `model.last_draft_ids` (what `DSparkProposer.propose` reads back
    in one copy), the per-sequence launches None there. The flat scheme launches its drafter
    once per sequence (`_run_draft_seq`, the path as before); the pipeline scheme
    drafts all of them in one launch (below).

    launch: `[(output_ids [1, K+1] int64, None, confidence [1, K] fp32)]`, one per
    span. The kernel drafts the B sequences' blocks side by side (B * K rows of one
    tile), so a step's drafting costs one kernel's collectives however many
    sequences it has -- a 10-sequence batch on the pipeline's last stage drafted
    one sequence after another at ~15 ms each before (2026-10-24)."""
    batched_tp64 = model.mk_plan.scheme == "tp64" and _DRAFT_BATCHED and len(spans) > 1
    model.last_draft_ids = None
    if model.mk_plan.scheme == "tp64" and not batched_tp64:
        return [
            _run_draft_seq(model, seeds.reshape(-1)[b : b + 1], taps.reshape(-1)[b : b + 1],
                           main_hidden, span, pg)
            for b, (span, pg) in enumerate(zip(spans, paged))
        ]
    from . import model as M

    cfg = model.config
    device = main_hidden.device
    B = len(spans)
    lengths = {length for _, length, _ in spans}
    if len(lengths) != 1:
        raise NotImplementedError(f"the drafter wants spans of one length, got {sorted(lengths)}")
    K = int(cfg.dspark_block_size)
    stages = list(model.mtp)
    win = int(stages[0].attn.window_size)
    ntab_p = -(-(win + K) // 128) * 128          # the kernel's slot-table rows per sequence (128-row tiles)
    # The spans tile the step's rows in order (`_sequence_spans` / `_bucket_spans`), so the
    # rows the drafter reads are one slice of `main_hidden`; a `torch.cat` of the spans'
    # slices ran through the host (~12 ms at bs10, measured 2026-10-27).
    first = spans[0][0]
    if all(begin == first + b * length for b, (begin, length, _) in enumerate(spans)):
        rows = B * spans[0][1]
        main = main_hidden if first == 0 and rows == int(main_hidden.shape[0]) else main_hidden[first : first + rows]
    else:
        main = torch.cat([main_hidden[begin : begin + length] for begin, length, _ in spans])
    row_pos = torch.cat([start_pos + torch.arange(length) for _, length, start_pos in spans])
    row_pos = row_pos.to(torch.int32).reshape(-1, 1).contiguous().to(device)
    write_slots = torch.cat([
        _host_slots(p).reshape(-1)[:length] for p, (_, length, _) in zip(paged, spans)
    ]).to(torch.int32).reshape(-1, 1).contiguous().to(device)
    tables = [_host_table(p).to(torch.int32).reshape(-1) for p in paged]
    nblk = max(int(t.numel()) for t in tables)
    table = torch.stack([
        torch.nn.functional.pad(t, (0, nblk - int(t.numel()))) for t in tables
    ]).reshape(-1, 1).contiguous().to(device)
    # (from the spans' start positions on the host, not a `torch.cat` of their device copies)
    pos = torch.tensor([int(p.start_pos) for p in paged], dtype=paged[0].pos.dtype).to(device)
    meta = M._compiled(_draft_meta)(seeds.reshape(-1)[:B], taps.reshape(-1)[:B], pos)
    _track(meta)
    consts = model.__dict__.setdefault("_mk_draft_consts", {})
    rows_c = consts.get((B, K, ntab_p))
    if rows_c is None:
        rows_c = consts[(B, K, ntab_p)] = draft_consts(B, K, ntab_p).to(device)
    weights, tail = model._mk_draft()
    at_zero = int(paged[0].at_zero)
    kscheme = "tp8pp8" if batched_tp64 else model.mk_plan.scheme   # the drafter body's package
    flags = (at_zero, K, win, 9, kernel_version("draft", kscheme),
             int(cfg.dspark_noise_token_id), int(cfg.vocab_size), int(cfg.dspark_num_experts_per_tok),
             int(paged[0].block_size))
    caches = [s.attn.window_kv_cache.view(-1, s.attn.head_dim) for s in stages]
    args = [meta, rows_c, main, row_pos, write_slots, table, *weights, *caches, *tail,
            model._mk_groups_attn, model._mk_groups_full, flags]
    # The drafter runs on the ranks that hold the head and the sampler: the flat
    # world under tp64, the last stage under tp8pp8 (`_mk_groups_full` is that
    # stage's groups); it is a distinct entry per scheme, `draft_<phase>_<scheme>`,
    # as every other kernel is.
    dphase = "prefill" if at_zero else "decode"
    kernel = mk_kernels.draft_batched(dphase) if batched_tp64 else mk_kernels.draft(dphase, model.mk_plan.scheme)
    if _MK_TRACE and model.mk_plan.stage_rank == 0:
        _mk_log.warning("mk launch %s B=%d spans=%s", kernel.func.__name__, B, spans)
    out = entry(kernel)(*args)
    output_ids, confidence = out[0], out[1]
    _track(confidence)
    model.last_draft_ids = output_ids
    conf = confidence.reshape(B, K)
    return [(output_ids[b : b + 1], None, conf[b : b + 1]) for b in range(B)]


@torch._dynamo.disable
@dataclass
class SourceReplica:
    """This stage's replica of a kv-source block another stage owns: the rows the
    source wrote this step arrive through the stage boundary and are scattered into
    these caches, which the stage's consumers of that source then read."""

    layer_id: int
    compress_ratio: int
    window_size: int
    compress_kv_cache: torch.Tensor
    index_k_cache: torch.Tensor
    is_kv_source = True


def shipped_source_id(config, stage_first_block: int):
    """The kv-source block whose cache rows cross the boundary *into* the stage whose
    first block is `stage_first_block`: the latest source before it, or None."""
    srcs = [sid for sid in config.kv_source_layer_ids if sid < stage_first_block]
    return max(srcs) if srcs else None


def topk_width(model, step, upto_layer: int) -> int:
    """Columns of the index top-k the latest index source at or before `upto_layer`
    produced this step: the kernel keeps `min(index_topk, read_len)` of its ratio's
    compressed read (`layer_kernel`: `topk = min(topk_in.shape[1], read_len)`), so the
    width follows the context length (384 at a 608-token context for ratio 2, 512 at
    2104). Both sides of a stage boundary size the top-k payload with this, or their
    kernels would be traced for different shapes and the runtime would refuse the
    pair's collective ("replica group signature mismatch")."""
    # The width is a running minimum along the blocks: an index source keeps
    # `min(width of the top-k it was handed, its own read length)`, so a later
    # source with a longer read (ratio 1 after ratio 2) inherits the narrower width.
    cfg = model.config
    width = int(cfg.index_topk)
    for sid in sorted(cfg.index_source_layer_ids):
        if sid > upto_layer:
            break
        ratio = int(cfg.compress_ratios[sid])
        if ratio:
            width = min(width, int(step.read_len(ratio)))
    return width


def _boundary_args(model, step, src, T, upto_layer):
    """`(topk, cmp_cache, idx_cache, slots)` for a boundary carrying `src`'s rows and
    the top-k of the latest index source at or before `upto_layer`."""
    d = model._mk_dummies
    width = topk_width(model, step, upto_layer)
    topk = step.topk if step.topk is not None else torch.zeros(
        T, width, dtype=torch.int32, device=step.device)
    if int(topk.shape[1]) != width:
        raise RuntimeError(
            f"stage {model.mk_plan.stage}: the index top-k is {int(topk.shape[1])} wide, the "
            f"boundary expects {width} (the latest index source up to block {upto_layer})"
        )
    if src is None:
        return topk, d.bf16, d.bf16, d.i32
    idx = step.for_ratio(int(src.compress_ratio), int(src.window_size))
    return topk, src.compress_kv_cache, src.index_k_cache, idx.grp_wslots


@torch._dynamo.disable
def run_recv(model, step, T):
    """A pipeline stage's entry (every stage but the first): the previous stage's
    stream, pre_mix and index top-k, and the rows the shipped kv source wrote this
    step, scattered into this stage's replica of its caches."""
    plan = model.mk_plan
    cfg = model.config
    src_id = shipped_source_id(cfg, plan.first_block)
    src = model._mk_nearest_source(src_id, "kv") if src_id is not None else None
    stream = torch.zeros(T, cfg.hc_mult, cfg.hidden_size, dtype=torch.bfloat16, device=step.device)
    pre_mix = torch.zeros(T, cfg.hc_mult, dtype=torch.float32, device=step.device)
    topk, cmp_cache, idx_cache, slots = _boundary_args(model, step, src, T, plan.first_block - 1)
    parity = (plan.stage - 1) % 2
    kernel = mk_kernels.boundary(parity)
    if _MK_TRACE and plan.stage_rank == 0:
        _mk_log.warning("mk launch %s (recv) stage=%d T=%d src=%s args=%s", kernel.func.__name__, plan.stage, T, src_id,
                        _arg_sig([stream, pre_mix, topk, cmp_cache, idx_cache, slots]))
    out = entry(kernel)(stream, pre_mix, topk, cmp_cache, idx_cache, slots, _role_tensor(model, 0.0),
                        plan.boundary_groups(parity), (kernel_version("pp", "tp8pp8"),))
    _track(out[1])
    step.topk = out[2]
    if src is not None:
        step.compress_cache, step.index_cache = src.compress_kv_cache, src.index_k_cache
    return out[0], out[1]


@torch._dynamo.disable
def run_send(model, stream, pre_mix, step):
    """A pipeline stage's exit (every stage but the last): hand the stream, pre_mix,
    the current index top-k and the current kv source's rows of this step to the
    next stage. Returns a small device tensor that completes with the hand-off."""
    plan = model.mk_plan
    T = int(stream.shape[0])
    src = model._mk_nearest_source(plan.last_block, "kv")
    topk, cmp_cache, idx_cache, slots = _boundary_args(model, step, src, T, plan.last_block)
    parity = plan.stage % 2
    kernel = mk_kernels.boundary(parity)
    if _MK_TRACE and plan.stage_rank == 0:
        _mk_log.warning("mk launch %s (send) stage=%d T=%d src=%d args=%s", kernel.func.__name__, plan.stage, T, src.layer_id,
                        _arg_sig([stream, pre_mix, topk, cmp_cache, idx_cache, slots]))
    out = entry(kernel)(stream, pre_mix, topk, cmp_cache, idx_cache, slots, _role_tensor(model, 1.0),
                        plan.boundary_groups(parity), (kernel_version("pp", "tp8pp8"),))
    _track(out[1])
    return out[1]


def _arg_sig(tensors) -> list:
    """`(shape, dtype, stride, contiguous)` of each tensor, for the launch trace."""
    return [(tuple(t.shape), str(t.dtype).replace("torch.", ""), tuple(t.stride()), bool(t.is_contiguous()))
            for t in tensors]


def _role_tensor(model, value: float) -> torch.Tensor:
    """`[1, 1]` fp32 on the device: 1.0 on the sending side of a boundary, 0.0 on
    the receiving one (the kernel's runtime role; see `pp_kernels`)."""
    cache = model.__dict__.setdefault("_mk_role_tensors", {})
    t = cache.get(value)
    if t is None:
        t = torch.full((1, 1), float(value), dtype=torch.float32).to(model.device)
        cache[value] = t
    return t


def run_blocks(model, stream, pre_mix, step, dump=None, head=False, embed=False):
    """This rank's backbone blocks through the megakernels (every block under tp64,
    the stage's five under tp8pp8). Returns `(stream, pre_mix)`; `step.taps` holds
    the DSpark taps on the stage that owns the tapped blocks. `head` (an all-greedy
    step, `fuses_head`): the last fused launch also runs the greedy head and
    `step.greedy_ids` holds its tokens. `embed` (`fuses_embed`): the first fused launch
    embeds `step.embed_ids` itself (`stream` / `pre_mix` are None)."""
    if PROF is not None:
        PROF.mark("blocks0", pre_mix)
    fuse = (_FUSE_DECODE > 1 and step.phase == "decode" and model.mk_plan.scheme == "tp64"
            and dump is None and not _MK_DEBUG and not _KCAPTURE)
    blocks = list(model.layers.values())
    if embed and not fuse:
        raise RuntimeError("run_blocks: the step's embedding was left to the fused launch, "
                           "but this step does not fuse (`fuses_embed` and `run_blocks` disagree)")
    for g, end in _fuse_groups(len(blocks)) if fuse else ():
        stream, pre_mix = run_fused(model, blocks[g:end], stream, pre_mix, step,
                                    head=head and end >= len(blocks), embed=embed and g == 0)
    for layer_id, block in () if fuse else zip(model.block_ids, model.layers.values()):
        stream, pre_mix = run_layer(model, block, stream, pre_mix, step)
        if dump is not None:
            dump(f"layer{layer_id}", stream)
            dump(f"premix{layer_id}", pre_mix)
    if PROF is not None:
        PROF.mark("blocks1", pre_mix)
    return stream, pre_mix
