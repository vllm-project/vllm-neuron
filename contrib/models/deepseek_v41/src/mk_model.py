# SPDX-License-Identifier: Apache-2.0
"""The bridge from `model.py`'s modules to the megakernels (Trainium2).

    prepare(model)                                   once, after the weights load
    make_step(spans, paged, ...)                     once a forward
    run_layer(model, block, stream, pre_mix, step)   one `Block` = one layer kernel
    run_draft(model, seed, tap, main_hidden, span, paged)   `forward_spec` = one drafter kernel

**Weights.** `prepare` re-lays every fp8 linear the kernels read as its
dequantized bf16 transpose (`kmajor_bf16`: `[in, out]`, the contraction on the
partition axis, the tensor engine's moving-operand layout) and releases the fp8
bytes, so each kernel's weight tile is one plain DMA (see `kernels/mk_gemm`). The
values are the ones the per-op GEMMs compute in-kernel on every call
(`blockquant._decode_e4m3` times the row scale), so nothing about the arithmetic
moves; the per-op path cannot run afterwards, which is why the choice is made once,
at load (`DEEPSEEK_V41_MEGAKERNEL`). The routed experts stay fp4 and k-major, as
the resident pool holds them, restacked once onto a leading expert axis with the
per-op slots left as views of the stack. Everything else a kernel reads -- `fn.T`
and the router weight as fp32, norms as `[1, n]` rows, the RoPE table as one
`[max_pos, 64]` (cos | sin) table -- is built once per block on first use.

**Per forward.** `StepIndices` turns the scheduler's block tables and slot mapping
into the kernels' index columns (`mk_host.build_attn_indices`) from host copies the
forward already made, and carries what the layers hand down the stack: the current
compressed / index-key caches, the latest top-k, the DSpark taps.

**Which kernel.** `layer_kernel.archetype` names a block from its config; the
phase is the step's shape -- at most 128 rows is a decode-shaped step (the
column-split kernel), more is a prefill chunk (the row-split kernel). The kernels
do their own collectives; the replica groups are constants (consecutive ranks).
"""

from __future__ import annotations

import functools
import glob
import hashlib
import inspect
import logging
import os
import time
from dataclasses import dataclass, field

import torch

from . import mk_host
from .kernels import draft_kernel, layer_kernel

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
#: Kernels submitted with nothing on the host waiting on them; see `_throttle`. Each
#: drafter kernel brings two small helper graphs the limiter does not track, so 8
#: keeps the worst case (8 x 3) inside the runtime's 32-deep queue.
_IN_FLIGHT = 8
#: `DEEPSEEK_V41_MK_SYNC=1` waits for every kernel before submitting the next and
#: logs its wall time on the first and last rank -- the way to name a kernel that
#: does not finish.
_MK_SYNC = os.environ.get("DEEPSEEK_V41_MK_SYNC", "") not in ("", "0")
PT = 128


class HostProf:
    """`DEEPSEEK_V41_HOST_PROF=1`: the host time of every forward, by part (ms; every rank
    times, so every rank takes the same dynamo-disabled calls and traces the same graphs;
    rank 0 logs),
    logged as medians every `DEEPSEEK_V41_HOST_PROF_EVERY` forwards (default 32) per phase:
    `gap` since the previous forward returned (the engine's scheduling, output processing
    and input preparation), `pre` until the block loop (span metadata, embedding, hashes),
    `layers` the loop -- `args` (each launch's arguments), `launch` (the call that submits
    it) and `post` -- `head` (collapse, norm, logits, sampling) and `draft` (the drafter).
    `DEEPSEEK_V41_HOST_PROF_WAIT=1` adds `wait`: how long the next forward, on entry, blocks
    reading back the previous one's last output (`tail`) -- 0 when the device had already
    drained, i.e. when the step boundary, not the device, is the critical path."""

    KEYS = ("gap", "wait", "pre", "layers", "args", "launch", "post", "head", "draft", "total")
    WAIT = os.environ.get("DEEPSEEK_V41_HOST_PROF_WAIT", "") not in ("", "0")

    def __init__(self, every: int):
        self.every, self.rows, self.cur, self.last_end, self.rank = every, {}, None, None, 0
        self.tail = None

    #: `DEEPSEEK_V41_CPROFILE=<start>:<count>`: cProfile rank 0 from forward `start` to forward
    #: `start + count` -- the forwards and everything between them (the engine's side) -- and log
    #: the top functions by cumulative and own time
    _cprof = os.environ.get("DEEPSEEK_V41_CPROFILE", "")
    calls, cprofile = 0, None

    def _cprofile_tick(self) -> None:
        import cProfile
        import io
        import pstats

        start, count = (int(v) for v in self._cprof.split(":"))
        self.calls += 1
        if self.calls == start:
            self.cprofile = cProfile.Profile()
            self.cprofile.enable()
        elif self.calls == start + count and self.cprofile is not None:
            self.cprofile.disable()
            for key in ("cumulative", "tottime"):
                buf = io.StringIO()
                pstats.Stats(self.cprofile, stream=buf).sort_stats(key).print_stats(40)
                logger.warning("deepseek_v41 cprofile over %d forwards by %s:\n%s", count, key, buf.getvalue())
            self.cprofile = None

    @torch._dynamo.disable
    def begin(self) -> None:
        if self._cprof and self.rank == 0:
            self._cprofile_tick()
        t = time.perf_counter()
        gap = (t - self.last_end) * 1e3 if self.last_end else float("nan")
        wait = float("nan")
        if self.WAIT and self.tail is not None:
            self.tail.cpu()
            wait = (time.perf_counter() - t) * 1e3
            t = time.perf_counter()
        self.tail = None
        self.cur = {"t0": t, "gap": gap, "wait": wait, "args": 0.0, "launch": 0.0, "post": 0.0,
                    "launches": 0}

    def add(self, key: str, seconds: float) -> None:
        if self.cur is not None:
            self.cur[key] = self.cur.get(key, 0.0) + seconds * 1e3

    @torch._dynamo.disable
    def mark(self, key: str) -> None:
        if self.cur is not None:
            self.cur[key] = (time.perf_counter() - self.cur["t0"]) * 1e3

    @torch._dynamo.disable
    def end(self) -> None:
        if self.cur is None:
            return
        phase = self.cur.get("phase", "?")
        t = time.perf_counter()
        c, self.cur, self.last_end = self.cur, None, t
        if self.tail is None:
            # the step's last launch: the drafter's (`run_draft_batched` sets it), else the
            # newest throttled output
            pending = getattr(self, "pending", None)
            self.tail = pending[-1] if pending else None
        c["total"] = (t - c["t0"]) * 1e3
        m = [c.get(k, c["total"]) for k in ("m_layers0", "m_layers1", "m_head")]
        c["pre"], c["layers"], c["head"], c["draft"] = m[0], m[1] - m[0], m[2] - m[1], c["total"] - m[2]
        rows = self.rows.setdefault(phase, [])
        rows.append(c)
        if len(rows) >= self.every and self.rank != 0:
            rows.clear()
        elif len(rows) >= self.every:
            import statistics

            def med(k):
                v = [r[k] for r in rows if r[k] == r[k]]
                return statistics.median(v) if v else float("nan")

            logger.warning("deepseek_v41 host prof %s x%d: %s ms (medians; %d launches)", phase, len(rows),
                           " | ".join(f"{k} {med(k):.2f}" for k in self.KEYS), c["launches"])
            rows.clear()


class RpcProf:
    """`DEEPSEEK_V41_HOST_PROF=1`: the model runner's per-step calls (ms) -- how long each
    runs and how long the worker sits between them (`idle>call`: the engine's output
    processing and scheduling, and the RPC to the 64 workers) -- logged as medians on
    rank 0 (a port of the Trainium3 `RpcProf`, `origin/deepseek_v41_trn3`)."""

    CALLS = ("execute_model", "sample_tokens", "take_draft_token_ids")

    def __init__(self, every: int):
        self.every, self.rows, self.cur, self.last_end, self.rank = every, [], {}, None, 0

    def wrap(self, cls, rank: int) -> None:
        self.rank = rank
        for name in self.CALLS:
            fn = getattr(cls, name, None)
            if fn is None or getattr(fn, "_rpc_prof", False):
                continue

            def timed(runner, *args, _fn=fn, _name=name, **kwargs):
                t0 = time.perf_counter()
                if _name == "execute_model" and self.cur:
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

    def flush(self, row) -> None:
        import statistics

        self.rows.append(row)
        if len(self.rows) < self.every:
            return
        rows, self.rows = self.rows, []
        if self.rank != 0:
            return
        keys = sorted({k for r in rows for k in r})

        def med(k):
            v = [r[k] for r in rows if k in r]
            return statistics.median(v) if v else float("nan")

        logger.warning("deepseek_v41 rpc prof x%d: %s (medians, ms)", len(rows),
                       " | ".join(f"{k} {med(k):.2f}" for k in keys))


HOST_PROF = (HostProf(int(os.environ.get("DEEPSEEK_V41_HOST_PROF_EVERY", "32")))
             if os.environ.get("DEEPSEEK_V41_HOST_PROF", "") not in ("", "0") else None)
RPC_PROF = RpcProf(HOST_PROF.every) if HOST_PROF is not None else None


@functools.lru_cache(maxsize=None)
def kernel_version() -> int:
    """A hash of every kernel source file, folded into the kernels' constant
    `flags`: the NKI compile caches key a kernel on its *own* function source,
    so an edit to a helper module would otherwise be served a stale NEFF."""
    h = hashlib.sha256()
    files = sorted(glob.glob(os.path.join(_HERE, "kernels", "**", "*.py"), recursive=True))
    files += [os.path.join(_HERE, "mk_host.py")]
    for path in files:
        with open(path, "rb") as fh:
            h.update(fh.read())
    return int.from_bytes(h.digest()[:4], "little") & 0x7FFFFFFF


@functools.lru_cache(maxsize=None)
def kernel_key(kernel) -> int:
    """`kernel_version()` folded with the kernel's module: lite's NKI cache keys a kernel on
    its file's basename, its name and its own function source, which a `tuned/` entry and
    the shared-module one of the same (arch, phase) have in common -- without this a cache
    built by one would serve the other."""
    import zlib

    module = (kernel.func if hasattr(kernel, "func") else kernel).__module__
    return (kernel_version() ^ zlib.crc32(module.encode())) & 0x7FFFFFFF


@functools.lru_cache(maxsize=None)
def entry(kernel):
    """The compiled entry point of one kernel, launched on both physical cores."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from . import model as M

    M._raise_dynamo_limits()
    wrapped = wrap_nki(kernel)[2]

    def run(*args):
        return wrapped(*args)

    name = getattr(kernel, "__name__", None) or getattr(getattr(kernel, "func", None), "__name__", "k")
    run.__name__ = f"nki_mk_{name}"
    # A code object of its own per kernel: dynamo keeps its compiled entries on the
    # code object, and one shared `run` would make every kernel a recompile of the
    # previous one (`len(args) == 41` guards) and share one cache-size budget.
    run.__code__ = run.__code__.replace(co_name=run.__name__)
    return torch.compile(run, backend=M._compile_backend_name(), fullgraph=True,
                         options=M._COMPILE_OPTIONS)


@functools.lru_cache(maxsize=None)
@functools.lru_cache(maxsize=None)
def params(kernel) -> tuple[str, ...]:
    func = kernel.func if hasattr(kernel, "func") else kernel
    return tuple(inspect.signature(func).parameters)


#: `DEEPSEEK_V41_DIRECT_LAUNCH` (default 1): a decode step's layer kernels are launched
#: on their NEFF's executor directly -- the arguments in the NEFF's input order, the KV caches
#: passed back as their own aliased outputs, every other output into a tensor of that
#: launch's own, the same every step -- instead of through the compiled entry (dynamo's
#: guards, AOTAutograd's wrapper and `_lite_execute`, which allocates fresh outputs). With the
#: step's index columns in `_step_buf`s too, a launch's IO tensors are where they were the
#: step before. The first launch of each (layer, shapes) goes through the entry and records
#: the mapping (`_calibrate`); one that cannot be mapped stays on the entry. 0: the entry.
_DIRECT = os.environ.get("DEEPSEEK_V41_DIRECT_LAUNCH", "1") not in ("", "0")


@dataclass
class _DirectLaunch:
    """A launch's executor; `pmap[i]` the argument that is NEFF input i; `outs[i]` NEFF
    output i's tensor (None: aliased to input `alias[i]`, a cache written in place); `ret`
    the entry's return values, each `("o", i)` (NEFF output i) or `("a", i)` (NEFF input i)."""

    ex: object
    pmap: list
    outs: list
    alias: dict
    ret: list


#: (layer id, shapes of the step's own arguments) -> `_DirectLaunch` / None (stays on the entry)
_DIRECTS: dict = {}
_UNSET = object()


def _calibrate(kernel, args):
    """`entry(kernel)(*args)`, watching the one executor call it makes. Returns
    `(out, direct)`; `direct` is None unless every NEFF input is one of `args`, the artifact's
    recorded FX-pass IO mutations are ALIAS-only (metadata), and every value the entry
    returns is a NEFF output or an aliased input."""
    from libtorch_neuronx_lite.compile import native_backend as NB

    seen = []
    lite_execute = NB._lite_execute

    def watch(artifact, inputs):
        res = lite_execute(artifact, inputs)
        seen.append((artifact, list(inputs), res))
        return res

    NB._lite_execute = watch
    try:
        out = entry(kernel)(*args)
    finally:
        NB._lite_execute = lite_execute

    def no(why):
        if os.environ.get("DEEPSEEK_V41_DIRECT_DEBUG"):
            logger.warning("deepseek_v41 direct launch: no mapping (%s)", why)
        return out, None

    if len(seen) != 1:
        return no(f"{len(seen)} executor calls")
    pm = getattr(seen[0][0], "pass_mutations", None) or []
    kinds = {m.kind.value for p in pm for m in (*p.input_mutations, *p.output_mutations)}
    if kinds - {"alias"}:
        # (ALIAS entries are metadata -- `_lite_execute` skips them -- anything else changes
        # the IO the executor sees)
        return no(f"FX-pass IO mutations {sorted(kinds)}")
    art, ins, louts = seen[0]
    at = {id(x): j for j, x in enumerate(args) if isinstance(x, torch.Tensor)}
    pmap = [at.get(id(t)) for t in ins]
    if not ins or None in pmap:
        return no(f"{sum(p is None for p in pmap)} of {len(ins)} NEFF inputs are not arguments")
    alias = {a.output_index: a.input_index for a in art.io_spec.aliases}
    # `_lite_execute` returns every output when the artifact recorded mutations (ALIAS-only
    # here), else only the unaliased ones
    free = (list(range(len(art.io_spec.outputs))) if pm
            else [i for i in range(len(art.io_spec.outputs)) if i not in alias])
    if len(louts) != len(free):
        return no(f"{len(louts)} executor outputs vs {len(free)} unaliased")
    ret = []
    for e in (out if isinstance(out, (tuple, list)) else (out,)):
        k = next((m for m, t in enumerate(louts) if t is e), None)
        if k is not None:
            ret.append(("o", free[k]))
            continue
        j = next((m for m, t in enumerate(ins) if t is e), None)
        if j is None:
            return no(f"return value {len(ret)} {tuple(getattr(e, 'shape', ()))} is neither an output "
                      f"nor an input (aliases {sorted(alias.items())})")
        ret.append(("a", j))
    outs = [None if i in alias else torch.empty(tuple(o.shape), dtype=o.dtype, device=ins[0].device)
            for i, o in enumerate(art.io_spec.outputs)]
    return out, _DirectLaunch(NB._executor_for(art), pmap, outs, alias, ret)


def _launch(kernel, key, args):
    """`entry(kernel)(*args)`, on the executor directly once calibrated (`_DIRECT`)."""
    d = _DIRECTS.get(key, _UNSET)
    if d is _UNSET:
        out, _DIRECTS[key] = _calibrate(kernel, args)
        if _DIRECTS[key] is None:
            logger.warning("deepseek_v41: %s stays on the compiled entry (no direct mapping)",
                           getattr(getattr(kernel, "func", kernel), "__name__", kernel))
        # (a one-output entry returns the bare tensor; the direct launch, a tuple)
        return out if isinstance(out, (tuple, list)) else (out,)
    if d is None:
        out = entry(kernel)(*args)
        return out if isinstance(out, (tuple, list)) else (out,)
    ins = [args[j] for j in d.pmap]
    outs = [ins[d.alias[i]] if t is None else t for i, t in enumerate(d.outs)]
    d.ex.execute(ins, outs)
    return tuple(outs[i] if kind == "o" else ins[i] for kind, i in d.ret)


#: (name, shape, dtype, device) -> the device tensor a decode step's own host-built input (an
#: index column, the live-row count) is written into: the same tensor every step, so a direct
#: launch's IO is where it was the step before. Rewriting one in place is safe: a step starts
#: after the previous step's launches completed (rank 0 read its tokens back), and within a
#: step each is written once, before the first launch that reads it is submitted.
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


@torch._dynamo.disable
def step_tensor(name, values, dtype, device) -> torch.Tensor:
    """Host `values` (a list) in a `_STEP_BUFS` tensor on `device`, for the traced
    forward: one copy into a persistent buffer instead of a fresh device tensor a step."""
    return _step_buf(name, torch.tensor(values, dtype=dtype), device)


#: shape -> a zero int32 top-k the layers before the step's first index source read (no
#: kernel writes it): one per shape instead of a `torch.zeros` on the device every step
_ZEROS: dict = {}


def _zeros_i32(shape, device) -> torch.Tensor:
    key = (tuple(shape), str(device))
    z = _ZEROS.get(key)
    if z is None:
        z = _ZEROS[key] = torch.zeros(shape, dtype=torch.int32, device=device)
    return z


#: the arguments a decode step builds itself (their shapes pick the NEFF a launch runs)
_STEP_ARGS = ("stream", "pre_mix", "row_pos", "write_slots", "win_slots", "twin", "wv", "cmp_slots",
              "cbase", "topk_in", "grp_pos", "grp_mslots", "grp_wslots", "clen", "hash_ids", "live")


def _host(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu()


def _dev(t: torch.Tensor, device) -> torch.Tensor:
    return t.contiguous().to(device)


def _row(t: torch.Tensor, device) -> torch.Tensor:
    """A 1-D buffer as the `[1, n]` row the kernels broadcast from."""
    return _dev(_host(t).reshape(1, -1), device)


def _t_f32(t: torch.Tensor, device) -> torch.Tensor:
    """`w.T` as fp32, for the helpers that stage a weight transposed in fp32."""
    return _dev(_host(t).t().float(), device)


def _t_bf16(t: torch.Tensor, device) -> torch.Tensor:
    """`w.T` of a bf16 weight (no quantization), k-major."""
    return _dev(_host(t).t().to(torch.bfloat16), device)


# --------------------------------------------------------------------------- weights


def kmajor_bf16(linear, release: bool = True) -> torch.Tensor:
    """An fp8 `BlockQuantLinear` as the kernels read it: `dequant(w).T`, bf16,
    `[in, out]`, on the weight's device. Converted once on the host and cached on
    the linear (`linear._kb16`); the fp8 buffers are released afterwards."""
    kb = getattr(linear, "_kb16", None)
    if kb is not None:
        return kb
    from .weight_loader import dequant_block_fp8

    assert linear.storage == "fp8", linear.storage
    device = linear.weight.device
    w = dequant_block_fp8(_host(linear.weight), _host(linear.scale), linear.block, torch.float32)
    kb = w.t().contiguous().to(torch.bfloat16).to(device)
    linear._kb16 = kb
    if release:
        linear.weight = None
        linear.scale = None
        if getattr(linear, "row_scale", None) is not None:
            linear.row_scale = None
    return kb


def stacked_experts(ffn) -> dict:
    """The resident expert pool as `[E, k', n]` k-major fp4 / e8m0 stacks (`w1`,
    `s1`, ...), the per-op slots rewired as views of them. Built once."""
    st = getattr(ffn, "_mk_stacked", None)
    if st is not None:
        return st
    if not ffn.resident or len(ffn.cache_slots) != ffn.n_local_experts:
        raise RuntimeError(
            f"layer {ffn.layer_id}: the routed experts are not resident (the pool streams); "
            "the megakernels read every local expert from HBM. Raise "
            "DEEPSEEK_V41_EXPERT_CACHE_BYTES."
        )
    device = ffn.cache_slots[0].w1.weight.device
    st = {}
    for name in ("w1", "w2", "w3"):
        for buf, key in (("weight", "w"), ("scale", "s")):
            parts = [_host(getattr(getattr(slot, name), buf)) for slot in ffn.cache_slots]
            stack = torch.stack(parts).contiguous().to(device)
            st[f"{key}{name[1]}"] = stack
            for i, slot in enumerate(ffn.cache_slots):
                setattr(getattr(slot, name), buf, stack[i])
    ffn._mk_stacked = st
    return st


def _freqs_table(attn, device) -> torch.Tensor:
    """`attn.freqs_cis` as `[max_pos, 64]` fp32 (cos | sin)."""
    cos, sin = attn.freqs_cis
    return _dev(torch.cat([_host(cos).float(), _host(sin).float()], -1), device)


@dataclass
class LayerWeights:
    """One block's static kernel arguments by name, and its archetype."""

    arch: str
    ratio: int
    kv_src: bool
    idx_src: bool
    engram: bool
    tap: bool
    window: int
    named: dict


def prepare(model) -> None:
    """Re-lay every weight the kernels read (see the module docstring). Called once,
    from `load_weights`, so the KV cache is sized against the converted footprint."""
    if HOST_PROF is not None:
        HOST_PROF.rank = int(model.rank)
        HOST_PROF.pending = model.__dict__.setdefault("_mk_pending", [])
        try:
            from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

            RPC_PROF.wrap(NeuronModelRunner, int(model.rank))
        except ImportError:  # outside a server (the unit tests)
            pass
    for block in model.layers:
        _block_weights(model, block)
    draft_weights(model, with_freqs=False)
    _free_staging_slots(model)
    # before the TP8 build sizes itself against the free HBM
    _build_shared_split(model)
    from . import mk_tp8

    mk_tp8.build(model, stacked_experts)
    logger.info("deepseek_v41: megakernel weights prepared (fp8 linears as bf16 k-major, "
                "expert pools stacked)")


#: `DEEPSEEK_V41_SHARED_SPLIT` (default 1): the decode kernels read this rank's slice of
#: each block's shared expert (`kernels/shared_split`) instead of the whole of it.
SHARED_SPLIT = os.environ.get("DEEPSEEK_V41_SHARED_SPLIT", "1") not in ("", "0")


def shared_slices(w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor, members: int,
                  j: int) -> dict:
    """Member `j` of `members`' slice of a shared expert (`kernels/shared_split`) from the
    whole k-major `w1` / `w3` `[dim, inter]` and `w2` `[inter, dim]` (host tensors)."""
    from .kernels import shared_split as SS

    dim, inter = w1.shape
    cols = []
    for t in SS.member_tiles(inter // SS.TILE, members, j):
        cols.extend(range(t * SS.TILE, (t + 1) * SS.TILE))
    idx = torch.tensor(cols, dtype=torch.long)
    s1 = torch.zeros((dim, SS.SLICED_COLS), dtype=w1.dtype)
    s3 = torch.zeros((dim, SS.SLICED_COLS), dtype=w3.dtype)
    s2 = torch.zeros((SS.SLICED_COLS, w2.shape[1]), dtype=w2.dtype)
    s1[:, : len(cols)] = w1.index_select(1, idx)
    s3[:, : len(cols)] = w3.index_select(1, idx)
    s2[: len(cols), :] = w2.index_select(0, idx)
    return {"sh_w1": s1, "sh_w3": s3, "sh_w2": s2}


def _build_shared_split(model) -> None:
    """Each block's shared-expert slice for this rank (`kernels/shared_split`): it replaces
    the whole `sh_w1` / `sh_w3` / `sh_w2` in the block's kernel arguments (prefill and
    decode), and the whole ones are freed -- before the TP8 build sizes itself against
    the free HBM."""
    if not SHARED_SPLIT or model.world_size == 1:
        return
    from .kernels import shared_split as SS

    t0 = time.perf_counter()
    rank = int(model.rank)
    members = list(next(g for g in model._mk_groups_attn if rank in g))
    j = members.index(rank)
    built = 0
    for block in model.layers:
        n = _block_weights(model, block)
        w1, w3, w2 = n["sh_w1"], n["sh_w3"], n["sh_w2"]
        dim, inter = w1.shape
        if inter % SS.TILE or SS.width(inter // SS.TILE, len(members)) != SS.SLICED_COLS:
            logger.info("deepseek_v41: shared expert not split (%d intermediate columns over "
                        "%d members)", inter, len(members))
            return
        sl = shared_slices(_host(w1), _host(w3), _host(w2), len(members), j)
        n.update({k: _dev(v, w1.device) for k, v in sl.items()})
        # the whole weights' last references: the k-major cache on each linear
        se = block.ffn.shared_experts
        for lin in (se.w1, se.w3, se.w2):
            lin._kb16 = None
        del w1, w3, w2
        built += 1
    # the drafter's stages the same way (both its phases: its attention all-reduce is the
    # attention group's too)
    dw = draft_weights(model, with_freqs=False)
    sl = [shared_slices(_host(dw["sh_w1"][i]), _host(dw["sh_w3"][i]), _host(dw["sh_w2"][i]),
                        len(members), j) for i in range(dw["sh_w1"].shape[0])]
    model._mk_draft_sh = {k: _dev(torch.stack([x[k] for x in sl]), dw["sh_w1"].device)
                          for k in ("sh_w1", "sh_w3", "sh_w2")}
    logger.info("deepseek_v41: decode shared expert split over the attention group on %d "
                "layers (member %d of %d, %d of %d intermediate columns, %.1f s) for rank %d",
                built, j, len(members), SS.SLICED_COLS, inter, time.perf_counter() - t0, rank)


def _free_staging_slots(model) -> None:
    """Release the per-op path's per-layer expert staging slot (one expert's buffers a
    block, ~0.75 GiB a rank): with the pool resident the megakernels never stage."""
    freed = 0
    for block in list(model.layers) + list(model.mtp):
        ffn = block.ffn
        if not getattr(ffn, "resident", False):
            continue
        for name in ("w1", "w2", "w3"):
            linear = getattr(ffn.expert_slot, name)
            for buf in ("weight", "scale"):
                t = getattr(linear, buf, None)
                if isinstance(t, torch.Tensor) and t.numel() > 1:
                    freed += t.numel() * t.element_size()
                    setattr(linear, buf, t.new_empty((0,)))
    if freed:
        logger.info("deepseek_v41: released %.2f GiB of per-op expert staging slots",
                    freed / 2**30)


#: `DEEPSEEK_V41_MOE_TP8_MAX_SEQS` (default 4): the decode steps of at most this many
#: sequences run the TP8EP8 MoE (`kernels/moe_tp8`), the larger ones EP64. TP8EP8 wins
#: where the routing is sparse and lopsided -- bs1: 36 picks over 384 experts, the
#: busiest of 64 ranks runs ~3 experts and most run none (in1024_bs1 37.9 -> 39.5
#: tok/s); by bs10 every rank has 3-6 active experts, EP64's imbalance is ~1.6x and a
#: TP8EP8 chunk's fixed costs give the gain back (in512_bs10 160.2 -> 158.2).
TP8_MAX_SEQS = int(os.environ.get("DEEPSEEK_V41_MOE_TP8_MAX_SEQS", "4"))


def _tp8_args(block, kernel, nseqs):
    """The decode kernel's TP8EP8 MoE arguments for `block` (`mk_tp8`), if it has them,
    `kernel` (a tuned decode package's) runs them and the step has at most
    `TP8_MAX_SEQS` sequences."""
    t8 = getattr(block, "_mk_tp8", None)
    if t8 is None or nseqs > TP8_MAX_SEQS:
        return None
    import sys

    mod = sys.modules.get((kernel.func if hasattr(kernel, "func") else kernel).__module__)
    return t8 if getattr(mod, "moe_tp8", None) is not None else None


def _block_weights(model, block) -> dict:
    """The weights of `block` that need no runtime state (no caches, no RoPE table)."""
    cached = getattr(block, "_mk_static", None)
    if cached is not None:
        return cached
    device = model.norm.weight.device
    attn, ffn = block.attn, block.ffn
    n = {
        "hca_fn_t": _t_f32(block.hc_attn.fn, device), "hca_base": _row(block.hc_attn.base, device),
        "hca_scale": _row(block.hc_attn.scale, device), "attn_norm_w": _row(block.attn_norm.weight, device),
        "hcf_fn_t": _t_f32(block.hc_ffn.fn, device), "hcf_base": _row(block.hc_ffn.base, device),
        "hcf_scale": _row(block.hc_ffn.scale, device), "ffn_norm_w": _row(block.ffn_norm.weight, device),
        "wq_a": kmajor_bf16(attn.wq_a), "q_norm_w": _row(attn.q_norm.weight, device),
        "wq_b": kmajor_bf16(attn.wq_b), "wkv": kmajor_bf16(attn.wkv),
        "kv_norm_w": _row(attn.kv_norm.weight, device),
        "wo_a": kmajor_bf16(attn.wo_a), "wo_b": kmajor_bf16(attn.wo_b),
        "attn_sink": _dev(_host(attn.attn_sink).float().reshape(-1, 1), device),
        "gate_w_t": _t_f32(ffn.gate.weight, device), "gate_b": _row(ffn.gate.bias, device),
        "expert_lo": torch.tensor([[float(ffn.experts_start_idx)]], dtype=torch.float32, device=device),
        "sh_w1": kmajor_bf16(ffn.shared_experts.w1), "sh_w3": kmajor_bf16(ffn.shared_experts.w3),
        "sh_w2": kmajor_bf16(ffn.shared_experts.w2),
    }
    n.update(stacked_experts(ffn))
    if attn.is_kv_source:
        comp = attn.compressor
        if comp.compress_ratio > 1:
            n["cm_wkv_t"] = _t_f32(comp.wkv.weight, device)
            n["cm_wgate_t"] = _t_f32(comp.wgate.weight, device)
        else:
            n["cm_wkv"] = _t_bf16(comp.wkv.weight, device)
        n["cm_norm_w"] = _row(comp.norm.weight, device)
        n["ix_wk"] = _t_bf16(attn.indexer.wk.weight, device)
        n["ix_knorm_w"] = _row(attn.indexer.k_norm.weight, device)
    if attn.is_index_source:
        n["ix_wq_b"] = kmajor_bf16(attn.indexer.wq_b)
        n["ix_wproj_t"] = _t_f32(attn.indexer.weights_proj.weight, device)
    if block.engram is not None:
        eg = block.engram
        if not eg.resident:
            raise RuntimeError(
                f"layer {block.layer_id}: the engram table is not resident on the device, and "
                "the megakernel reads it there. Raise DEEPSEEK_V41_ENGRAM_TABLE_BYTES."
            )
        n["etable"] = eg.table
        # The ids reach the kernel already local to the shard (`StepIndices.hash_cols`).
        n["erow_start"] = torch.zeros(1, 1, dtype=torch.int32, device=device)
        n["e_wkv"] = kmajor_bf16(eg.wkv)
        n["e_qk"] = _dev((_host(eg.q_weight).float() * _host(eg.k_weight).float()).reshape(1, -1), device)
    block._mk_static = n
    return n


def layer_weights(model, block) -> LayerWeights:
    lw = getattr(block, "_mk_weights", None)
    if lw is not None:
        return lw
    attn = block.attn
    cfg = model.config
    n = dict(_block_weights(model, block))
    n["freqs"] = model._mk_freqs(attn)
    ratio = int(attn.compress_ratio)
    tap = attn.layer_id in set(cfg.dspark_target_layer_ids)
    arch = layer_kernel.archetype(ratio, attn.is_kv_source, attn.is_index_source,
                                  block.engram is not None, tap)
    lw = LayerWeights(arch=arch, ratio=ratio, kv_src=bool(attn.is_kv_source),
                      idx_src=bool(attn.is_index_source), engram=block.engram is not None,
                      tap=tap, window=int(attn.window_size), named=n)
    block._mk_weights = lw
    return lw


# --------------------------------------------------------------------------- per step


@dataclass
class StepIndices:
    """Per-forward index columns and the state the layers hand down the stack."""

    spans: list
    paged: list
    device: torch.device
    hashes: torch.Tensor | None
    tables: list
    slot_mapping: torch.Tensor
    live_rows: int
    T: int = 0
    _by_ratio: dict = field(default_factory=dict)
    _hash_cols: dict = field(default_factory=dict)
    topk: torch.Tensor | None = None
    compress_cache: torch.Tensor | None = None
    index_cache: torch.Tensor | None = None
    taps: list = field(default_factory=list)
    #: Outputs of the layers submitted and not yet known complete, oldest first.
    pending: list = field(default_factory=list)

    def __post_init__(self):
        self.T = sum(length for _, length, _ in self.spans)
        self.block_size = int(self.paged[0].block_size)
        lengths = {length for _, length, _ in self.spans}
        if len(lengths) != 1:
            raise NotImplementedError(
                f"the layer kernels want every span of a step to have the same length, got "
                f"{sorted(lengths)}; the Neuron scheduler does not mix prefill and decode"
            )
        #: a decode step writes its own device inputs into `_step_buf`s (see `_DIRECT`)
        self.stable = _DIRECT and self.T <= PT
        live = torch.tensor([[float(self.live_rows)]], dtype=torch.float32)
        self.live = _step_buf(("live",), live, self.device) if self.stable else live.to(self.device)

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

    def hash_cols(self, index: int, row_start: torch.Tensor) -> torch.Tensor:
        """This engram layer's `[T, 24]` int32 row ids, **local to this rank's shard**
        (`global - row_start`): the kernel's gather turns ids into fp32 for its range
        mask, and a global id of a 384M-row table is not exact above 2^24, while a
        local one is below the shard width; an off-shard one only has to stay out of
        range, which a rounded value does."""
        t = self._hash_cols.get(index)
        if t is None:
            from . import model as M

            if self.hashes is None:
                raise RuntimeError("an engram layer ran but the step has no n-gram hashes")
            t = M._compiled(_local_ids)(self.hashes, index, row_start)
            self._hash_cols[index] = t
        return t


def _local_ids(hashes, index, row_start):
    """`hashes[0, :, index, :] - row_start` as int32, as a compiled step."""
    return (hashes[0, :, index, :] - row_start).to(torch.int32)


_HBM_LOG_AT = (64, 512)
_FORWARDS = [0]


def _log_hbm_once(model) -> None:
    """The HBM the runtime reports at a couple of forwards into serving (rank 0): what the
    weights, the KV cache and every loaded NEFF's own allocations leave free -- the
    headroom `mk_tp8`'s budget reserves."""
    _FORWARDS[0] += 1
    if _FORWARDS[0] not in _HBM_LOG_AT or int(model.rank) != 0:
        return
    try:
        used, free = torch.classes.neuron.Runtime().get_vnc_memory_stats()
    except Exception:  # noqa: BLE001
        return
    logger.warning("deepseek_v41: HBM at forward %d: %.2f GiB used, %.2f GiB free", _FORWARDS[0],
                   used / 2**30, free / 2**30)


@torch._dynamo.disable
def run_layers(model, stream, pre_mix, spans, paged, hashes, live_rows):
    """Every backbone block of one forward through its layer kernel. Returns
    `(stream, pre_mix, taps)`. Outside dynamo, with the per-step host data (the
    span list, the host block tables and slot lists `_paged_spans` kept) built and
    read only here."""
    from . import model as M

    if HOST_PROF is not None and HOST_PROF.cur is not None:
        HOST_PROF.cur["phase"] = f"{'decode' if int(stream.shape[0]) <= PT else 'prefill'}_B{len(spans)}"
    _log_hbm_once(model)
    step = StepIndices(
        spans, paged, stream.device, hashes,
        [span.host_table for span in paged],
        torch.as_tensor([slot for span in paged for slot in span.host_slots], dtype=torch.int64),
        int(live_rows),
    )
    for layer_id, block in enumerate(model.layers):
        if layer_id % M._HEAP_TRIM_EVERY == 0:
            M.host_heap_trim(f"layer {layer_id}", model.world_size)
        stream, pre_mix = run_layer(model, block, stream, pre_mix, step)
    return stream, pre_mix, step.taps


@torch._dynamo.disable
def run_layer(model, block, stream: torch.Tensor, pre_mix: torch.Tensor, step: StepIndices):
    """One backbone block through its archetype's kernel. Returns `(stream, ffn_pre)`."""
    t_args = time.perf_counter()
    lw = layer_weights(model, block)
    attn = block.attn
    T = int(stream.shape[0])
    B = len(step.spans)
    phase = "decode" if T <= PT else "prefill"
    if phase == "prefill" and B != 1:
        raise NotImplementedError(f"a {T}-row step over {B} sequences: the prefill kernels serve one")
    idx = step.for_ratio(lw.ratio, lw.window)
    hd = attn.head_dim
    n = dict(lw.named)
    n.update(
        stream=stream, pre_mix=pre_mix, row_pos=idx.row_pos, write_slots=idx.write_slots,
        win_slots=idx.win_slots, twin=idx.twin, wv=idx.wv,
        window_kv_cache=attn.window_kv_cache.view(-1, hd), live=step.live,
        groups_attn=model._mk_groups_attn, groups_world=model._mk_groups_world,
        flags=(B, lw.window, 9, kernel_key(layer_kernel.KERNELS[(lw.arch, phase)])),
    )
    if lw.kv_src:
        step.compress_cache = attn.compress_kv_cache.view(-1, hd)
        step.index_cache = attn.index_k_cache.view(-1, attn.index_k_cache.shape[-1])
        n.update(grp_pos=idx.grp_pos, grp_mslots=idx.grp_mslots, grp_wslots=idx.grp_wslots)
        if lw.ratio == 2:
            n.update(pool_kv=attn.compressor.pool_kv, pool_score=attn.compressor.pool_score)
    if lw.ratio:
        if step.compress_cache is None or step.index_cache is None:
            raise RuntimeError(f"layer {attn.layer_id} reads compressed KV before any kv-source layer ran")
        if attn.is_index_source and attn.indexer.uses_candidates:
            # The candidate-block mask keeps the best `candidate_topk_blocks` blocks of
            # `candidate_block_size` groups; it only removes groups when the context
            # holds more than that, which the kernel does not model.
            #
            # TODO(candidate mask): implement it in the kernels to serve contexts past
            # `candidate_topk_blocks * candidate_block_size` compressed groups (16384,
            # i.e. ~16k tokens at the ratio-1 layers). This is a correctness feature,
            # not a performance one: past the limit it changes which compressed
            # positions layers 24-36 (and the consumers behind them) attend to. What
            # the per-op path does (`model.select_candidate_blocks`, `Indexer.forward`):
            #   1. in the `c1_source` kernel (layer 20), after the reduced, clen-masked
            #      index scores: the max over each block of `candidate_block_size`
            #      groups, the block holding `(clen - 1) // candidate_block_size`
            #      forced in, a top-`candidate_topk_blocks` over blocks, and a keep mask
            #      of the blocks whose value beats NEG_BOUND, returned to the bridge and
            #      handed down the stack like `step.topk`;
            #   2. in the `c1_index` kernels, the scores outside a kept block set to
            #      NEG_BOUND before `index_topk_rows`.
            # Until then the guard below raises rather than compute a different top-k.
            limit = int(model.config.candidate_topk_blocks) * int(model.config.candidate_block_size)
            if step.read_len(lw.ratio) > limit:
                raise NotImplementedError(
                    f"the context reads {step.read_len(lw.ratio)} compressed groups, more than the "
                    f"{limit} the candidate-block mask keeps; the kernels assume the mask is a no-op"
                )
        if step.topk is None:
            step.topk = _zeros_i32((T, int(model.config.index_topk)), step.device)
        n.update(cmp_slots=idx.cmp_slots, cbase=idx.cbase, topk_in=step.topk,
                 compress_kv_cache=step.compress_cache)
    if lw.idx_src:
        n.update(index_k_cache=step.index_cache, clen=idx.clen)
    if lw.engram:
        index = list(model.config.engram_layer_ids).index(attn.layer_id)
        n["hash_ids"] = step.hash_cols(index, block.engram.row_start)
    kernel = layer_kernel.KERNELS[(lw.arch, phase)]
    if phase == "decode":
        t8 = _tp8_args(block, kernel, B)
        if t8 is not None:
            n.update(t8)
    prof = HOST_PROF if HOST_PROF is not None and HOST_PROF.cur is not None else None
    if prof is not None:
        t_launch = time.perf_counter()
        prof.add("args", t_launch - t_args)
    if _MK_SYNC:
        t0 = time.perf_counter()
        if model.rank in (0, model.world_size - 1):
            logger.warning("deepseek_v41 MK_SYNC rank %d layer %d %s/%s T=%d B=%d submit",
                           model.rank, attn.layer_id, lw.arch, phase, T, B)
    args = [n[name] for name in params(kernel)]
    if step.stable and not _MK_SYNC:
        key = (attn.layer_id, tuple(tuple(n[a].shape) for a in _STEP_ARGS if a in n))
        out = _launch(kernel, key, args)
    else:
        out = entry(kernel)(*args)
    if prof is not None:
        t_post = time.perf_counter()
        prof.add("launch", t_post - t_launch)
        prof.cur["launches"] += 1
    if _MK_SYNC:
        out[1].cpu()
        if model.rank in (0, model.world_size - 1):
            logger.warning("deepseek_v41 MK_SYNC rank %d layer %d done in %.1f ms", model.rank,
                           attn.layer_id, 1e3 * (time.perf_counter() - t0))
    out_stream, ffn_pre = out[0], out[1]
    k = 3 + ((4 if lw.ratio == 2 else 2) if lw.kv_src else 0)
    if lw.idx_src:
        step.topk = out[k]
        k += 1
    if lw.tap:
        step.taps.append(out[k])
    _throttle(model, ffn_pre)
    if prof is not None:
        prof.add("post", time.perf_counter() - t_post)
    return out_stream, ffn_pre


def _throttle(model, small: torch.Tensor) -> None:
    """Keep at most `_IN_FLIGHT` of this bridge's kernels outstanding.

    The runtime's submission queue holds `NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS`
    (32) executions and a forward submits a kernel a layer, then a drafter kernel (and
    its two small helper graphs) a sequence, with nothing on the host waiting on any of
    them -- at ten sequences that overflowed the queue (`Execution Queue Full`,
    in512_bs10). Reading the oldest outstanding kernel's tiny output back blocks until
    that execution, and every earlier one, has completed; the device never drains.
    The list lives on the model so it spans the layers and the drafts of a step."""
    pending = model.__dict__.setdefault("_mk_pending", [])
    pending.append(small)
    while len(pending) > _IN_FLIGHT:
        pending.pop(0).cpu()


# --------------------------------------------------------------------------- drafter


def _draft_meta(seed, tap, pos):
    return torch.stack([seed.reshape(-1)[:1], tap.reshape(-1)[:1], pos.reshape(-1)[:1]]).to(torch.int32).reshape(1, 3)


def _draft_meta_b(seeds, taps, pos):
    """`meta [B, 3]` int32 (seed, tap, pos) per sequence, as a compiled step."""
    return torch.stack([seeds.reshape(-1), taps.reshape(-1), pos.reshape(-1)], 1).to(torch.int32)


#: `DEEPSEEK_V41_DRAFT_BATCHED` (default 1): a decode step of B > 1 sequences drafts them all
#: in one launch of `draft_batched.draft_decode_batched` (bit-identical to B launches of
#: `draft_decode`, which re-stream the three stages' weights once per sequence). 0: one
#: launch a sequence.
DRAFT_BATCHED = os.environ.get("DEEPSEEK_V41_DRAFT_BATCHED", "1") not in ("", "0")


@torch._dynamo.disable
def run_draft_batched(model, seeds, taps, main_hidden, spans, paged):
    """`forward_spec` for every sequence of a decode step in one drafter launch:
    `[(output_ids [1, K+1] int64, None, confidence [1, K] fp32)]` a sequence."""
    from . import model as M
    from .kernels import draft_batched

    cfg = model.config
    device = main_hidden.device
    B = len(spans)
    K = int(cfg.dspark_block_size)
    stages = list(model.mtp)
    win = int(stages[0].attn.window_size)
    ntab_p = -(-(win + K) // PT) * PT
    lengths = {length for _, length, _ in spans}
    assert len(lengths) == 1, f"the batched drafter wants equal spans, got {sorted(lengths)}"
    L = lengths.pop()
    # the spans tile the step's rows in order, so the main rows are one slice
    first = spans[0][0]
    if all(begin == first + b * L for b, (begin, _, _) in enumerate(spans)):
        rows = B * L
        main = main_hidden if first == 0 and rows == int(main_hidden.shape[0]) else main_hidden[first : first + rows]
    else:
        main = torch.cat([main_hidden[begin : begin + length] for begin, length, _ in spans])
    pos = _step_buf(("dpos", B), torch.tensor([int(sp) for _, _, sp in spans], dtype=torch.int64), device)
    meta = M._compiled(_draft_meta_b)(seeds.reshape(-1)[:B], taps.reshape(-1)[:B], pos)
    consts = model.__dict__.setdefault("_mk_draft_b_consts", {})
    ck = (B, K, ntab_p)
    if ck not in consts:
        rows_c = torch.tensor([[b, i, b * ntab_p, 0] for b in range(B) for i in range(K)], dtype=torch.int32)
        consts[ck] = (rows_c.to(device), torch.tensor([[float(B * K)]], dtype=torch.float32).to(device))
    rows_c, live = consts[ck]
    row_pos = torch.cat([sp + torch.arange(length) for _, length, sp in spans]).to(torch.int32).reshape(-1, 1)
    write_slots = torch.cat([torch.as_tensor(p.host_slots[:length], dtype=torch.int32)
                             for p, (_, length, _) in zip(paged, spans)]).reshape(-1, 1)
    tables = [p.host_table.to(torch.int32).reshape(-1) for p in paged]
    nblk = max(int(t.numel()) for t in tables)
    table = torch.stack([torch.nn.functional.pad(t, (0, nblk - int(t.numel()))) for t in tables]).reshape(-1, 1)
    n = dict(draft_weights(model))
    n.update(getattr(model, "_mk_draft_sh", None) or {})
    n.update(
        meta=meta, rows_c=rows_c, main_hidden=main, live=live,
        row_pos=_step_buf(("drow_pos", B), row_pos, device),
        write_slots=_step_buf(("dwrite_slots", B), write_slots, device),
        block_table=_step_buf(("dblock_table", B), table, device),
        cache0=stages[0].attn.window_kv_cache.view(-1, stages[0].attn.head_dim),
        cache1=stages[1].attn.window_kv_cache.view(-1, stages[1].attn.head_dim),
        cache2=stages[2].attn.window_kv_cache.view(-1, stages[2].attn.head_dim),
        groups_attn=model._mk_groups_attn, groups_world=model._mk_groups_world,
        flags=(K, win, 9, kernel_key(draft_batched.draft_decode_batched),
               int(cfg.dspark_noise_token_id), int(cfg.vocab_size), int(cfg.dspark_num_experts_per_tok),
               int(paged[0].block_size)),
    )
    kernel = draft_batched.draft_decode_batched
    args = [n[name] for name in params(kernel)]
    if _DIRECT and not _MK_SYNC:
        out = _launch(kernel, ("draft_b", tuple(tuple(t.shape) for t in (main, table, rows_c))), args)
    else:
        out = entry(kernel)(*args)
    output_ids, confidence = out[0], out[1]
    _throttle(model, confidence)
    ids = M._compiled(_ids_i64)(output_ids)
    conf = confidence.reshape(B, K)
    return [(ids[b : b + 1], None, conf[b : b + 1]) for b in range(B)]


#: `DEEPSEEK_V41_HEAD_GREEDY` (default 1): an all-greedy speculative decode step runs
#: the collapse, final norm, vocabulary-parallel head and the argmax as one launch
#: (`kernels/head_kernel.head_greedy`) in place of the collapse / norm / head graphs and
#: the sampler; `model.forward` decides from the runner's `sampling_hint`.
HEAD_GREEDY = os.environ.get("DEEPSEEK_V41_HEAD_GREEDY", "1") not in ("", "0")

#: `DEEPSEEK_V41_EMBED_KERNEL` (default 1): a decode step embeds its tokens and expands them
#: to the hyper-connection copies in one launch (`kernels/head_kernel.embed_expand`) in
#: place of the embedding graph (its XLA all-reduce) and the `hc_expand` kernel.
EMBED_KERNEL = os.environ.get("DEEPSEEK_V41_EMBED_KERNEL", "1") not in ("", "0")


#: `DEEPSEEK_V41_CHECK_ENDS=1` (debugging): every decode step also runs the separate
#: embedding / head path and logs (rank 0) where `run_embed` / `run_head_greedy` differ.
CHECK_ENDS = os.environ.get("DEEPSEEK_V41_CHECK_ENDS", "") not in ("", "0")


_CHECK_NOTES: set = set()


@torch._dynamo.disable
def check_note(msg: str) -> None:
    """Log `msg` once (rank 0)."""
    if msg not in _CHECK_NOTES:
        _CHECK_NOTES.add(msg)
        logger.warning("deepseek_v41 CHECK %s", msg)


@torch._dynamo.disable
def check_ends(name, got, ref, input_ids) -> None:
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    g, r = got.cpu(), ref.cpu()
    if name == "head ids":
        logits = r.float()
        want = logits.argmax(-1)
        ids = g.reshape(-1).to(torch.int64)
        bad = (ids != want).nonzero().reshape(-1).tolist()
        if bad and rank == 0:
            rows = [(t, int(ids[t]), int(want[t]), float(logits[t, ids[t].clamp(0, logits.shape[1] - 1)]),
                     float(logits[t, want[t]])) for t in bad[:6]]
            logger.warning("deepseek_v41 CHECK %s: %d/%d rows differ (row, got, want, logit[got], logit[want]) %s",
                           name, len(bad), ids.numel(), rows)
        elif rank == 0:
            logger.warning("deepseek_v41 CHECK %s: %d rows equal", name, ids.numel())
        return
    same = g.shape == r.shape and torch.equal(g.contiguous().view(torch.uint8), r.contiguous().view(torch.uint8))
    if rank == 0:
        if same:
            logger.warning("deepseek_v41 CHECK %s: equal %s", name, tuple(g.shape))
        else:
            diff = (g.float() - r.float()).abs().reshape(g.shape[0], -1).amax(1) if g.shape == r.shape else None
            logger.warning("deepseek_v41 CHECK %s: DIFFER shapes %s %s dtypes %s %s, row max |diff| %s, ids %s",
                           name, tuple(g.shape), tuple(r.shape), g.dtype, r.dtype,
                           None if diff is None else diff.tolist(),
                           None if input_ids is None else input_ids.cpu().reshape(-1).tolist())


@torch._dynamo.disable
def run_embed(model, input_ids):
    """A decode step's `(stream [T, hc, dim], pre_mix [T, hc])` -- the token embedding and
    the hyper-connection expand -- in one launch (`kernels/head_kernel.embed_expand`)."""
    from .kernels import head_kernel

    device = input_ids.device
    T = int(input_ids.shape[0])
    n = model.__dict__.get("_mk_embed_")
    if n is None:
        part = int(model.part_vocab_size)
        n = model.__dict__["_mk_embed_"] = {
            "embed_w": model.embed_weight,
            "vocab_start": torch.tensor([[int(model.rank * part)]], dtype=torch.int32).to(device),
            "groups_world": model._mk_groups_world,
            "flags": (int(model.config.hc_mult), kernel_key(head_kernel.embed_expand)),
        }
    kernel = head_kernel.embed_expand
    # int32 ids as they are; int64 ones as int32 (low, high) pairs -- a view, no device op
    ids = (input_ids.reshape(T, 1) if input_ids.dtype == torch.int32
           else input_ids.reshape(T).view(torch.int32).reshape(T, 2))
    args = [ids] + [n[name] for name in params(kernel)[1:]]
    if _DIRECT and not _MK_SYNC:
        out = _launch(kernel, ("embed", T, input_ids.dtype), args)
    else:
        out = entry(kernel)(*args)
    return out[0], out[1]


@torch._dynamo.disable
def run_head_greedy(model, stream, pre_mix):
    """The greedy token of every row of a decode step, `[T, 1]` int32, in one launch."""
    from .kernels import head_kernel

    device = stream.device
    n = model.__dict__.get("_mk_head_")
    if n is None:
        part = int(model.part_vocab_size)
        bias = model.__dict__.get("_vocab_pad_bias_t")
        n = model.__dict__["_mk_head_"] = {
            "norm_w": _row(model.norm.weight, device),
            "head_w": model.head_weight,
            "pad_bias": (bias.reshape(1, part) if bias is not None
                         else torch.zeros(1, part, dtype=torch.float32).to(device)),
            "vocab_start": torch.tensor([[int(model.rank * part)]], dtype=torch.int32).to(device),
            "groups_world": model._mk_groups_world,
            "flags": (kernel_key(head_kernel.head_greedy),),
        }
    kernel = head_kernel.head_greedy
    args = [stream, pre_mix] + [n[name] for name in params(kernel)[2:]]
    if _DIRECT and not _MK_SYNC:
        return _launch(kernel, ("head", tuple(stream.shape)), args)[0]
    out = entry(kernel)(*args)
    return out[0] if isinstance(out, (tuple, list)) else out


def _ids_i64(ids):
    """The drafted ids as int64, as a compiled step (a dtype cast is what the
    backend's eager fallback does not implement)."""
    return ids.to(torch.int64)


def draft_weights(model, with_freqs: bool = True) -> dict:
    """The drafter's static arguments by name (built once; the RoPE table, which
    `bind_kv_cache` builds after the weights load, joins on first use)."""
    dw = getattr(model, "_mk_draft_", None)
    if dw is not None:
        if with_freqs and "freqs" not in dw:
            dw["freqs"] = _freqs_table(model.mtp[0].attn, model.norm.weight.device)
        return dw
    device = model.norm.weight.device
    stages = list(model.mtp)
    s0, last = stages[0], stages[-1]

    def stack(get, dtype=None):
        # Through the host: a one-time copy, and no eager op on the device.
        t = torch.stack([_host(get(s)) for s in stages])
        return _dev(t if dtype is None else t.to(dtype), device)

    n = {
        "main_proj": kmajor_bf16(s0.main_proj), "main_norm_w": _row(s0.main_norm.weight, device),
        "embed_w": model.embed_weight,
        "vocab_start": torch.tensor([[int(model.rank * model.part_vocab_size)]], dtype=torch.int32, device=device),
        "hca_base": stack(lambda s: s.hc_attn.base), "hca_scale": stack(lambda s: s.hc_attn.scale),
        "attn_norm_w": stack(lambda s: s.attn_norm.weight),
        "hcf_base": stack(lambda s: s.hc_ffn.base), "hcf_scale": stack(lambda s: s.hc_ffn.scale),
        "ffn_norm_w": stack(lambda s: s.ffn_norm.weight),
        "wq_a": stack(lambda s: kmajor_bf16(s.attn.wq_a)), "q_norm_w": stack(lambda s: s.attn.q_norm.weight),
        "wq_b": stack(lambda s: kmajor_bf16(s.attn.wq_b)),
        "wkv": stack(lambda s: kmajor_bf16(s.attn.wkv)), "kv_norm_w": stack(lambda s: s.attn.kv_norm.weight),
        "wo_a": stack(lambda s: kmajor_bf16(s.attn.wo_a)), "wo_b": stack(lambda s: kmajor_bf16(s.attn.wo_b)),
        "attn_sink": stack(lambda s: s.attn.attn_sink.float().reshape(-1, 1)),
        "gate_b": stack(lambda s: s.ffn.gate.bias),
        "expert_lo": torch.tensor([[float(s0.ffn.experts_start_idx)]], dtype=torch.float32, device=device),
        "live": torch.tensor([[float(model.config.dspark_block_size)]], dtype=torch.float32, device=device),
        "sh_w1": stack(lambda s: kmajor_bf16(s.ffn.shared_experts.w1)),
        "sh_w3": stack(lambda s: kmajor_bf16(s.ffn.shared_experts.w3)),
        "sh_w2": stack(lambda s: kmajor_bf16(s.ffn.shared_experts.w2)),
        "fnorm_w": _row(last.norm.weight, device),
        "head_w": _t_bf16(model.head_weight, device),
        "mk_embed_w": last.markov_head.embed_weight, "mk_head_w": last.markov_head.head_weight,
        "conf_w": last.confidence_head.proj_weight,
    }
    for i, s in enumerate(stages):
        n[f"hca_fn_t{i}"] = _t_f32(s.hc_attn.fn, device)
        n[f"hcf_fn_t{i}"] = _t_f32(s.hc_ffn.fn, device)
        n[f"gate_w_t{i}"] = _t_f32(s.ffn.gate.weight, device)
    exps = [stacked_experts(s.ffn) for s in stages]
    for key in ("w1", "s1", "w3", "s3", "w2", "s2"):
        n[key] = _dev(torch.stack([_host(e[key]) for e in exps]), device)
    # The per-stage copies are views of the stacks from here on, so the drafter's
    # weights are held once.
    for i, s in enumerate(stages):
        for name, lin in (("wq_a", s.attn.wq_a), ("wq_b", s.attn.wq_b), ("wkv", s.attn.wkv),
                          ("wo_a", s.attn.wo_a), ("wo_b", s.attn.wo_b), ("sh_w1", s.ffn.shared_experts.w1),
                          ("sh_w3", s.ffn.shared_experts.w3), ("sh_w2", s.ffn.shared_experts.w2)):
            lin._kb16 = n[name][i]
        st = s.ffn._mk_stacked
        for key in st:
            st[key] = n[key][i]
        for e, slot in enumerate(s.ffn.cache_slots):
            for lname in ("w1", "w2", "w3"):
                lin = getattr(slot, lname)
                lin.weight = n["w" + lname[1]][i, e]
                lin.scale = n["s" + lname[1]][i, e]
    model._mk_draft_ = n
    if with_freqs:
        n["freqs"] = _freqs_table(s0.attn, device)
    return n


@torch._dynamo.disable
def run_draft(model, seed_ids, tap, main_hidden, span, paged):
    """`forward_spec` for one sequence through its phase's drafter kernel:
    `(output_ids [1, K+1] int64, None, confidence [1, K] fp32)`."""
    from . import model as M

    cfg = model.config
    device = main_hidden.device
    host_table, host_slots = paged.host_table, paged.host_slots
    begin, length, start_pos = span
    main = main_hidden[begin : begin + length]
    meta = M._compiled(_draft_meta)(seed_ids, tap, paged.pos)
    n = dict(draft_weights(model))
    n.update(getattr(model, "_mk_draft_sh", None) or {})
    K = int(cfg.dspark_block_size)
    stages = list(model.mtp)
    n.update(
        meta=meta, main_hidden=main,
        row_pos=(start_pos + torch.arange(length)).to(torch.int32).reshape(-1, 1).to(device),
        write_slots=torch.as_tensor(host_slots[:length], dtype=torch.int32).reshape(-1, 1).to(device),
        block_table=host_table.to(torch.int32).reshape(-1, 1).contiguous().to(device),
        cache0=stages[0].attn.window_kv_cache.view(-1, stages[0].attn.head_dim),
        cache1=stages[1].attn.window_kv_cache.view(-1, stages[1].attn.head_dim),
        cache2=stages[2].attn.window_kv_cache.view(-1, stages[2].attn.head_dim),
        groups_attn=model._mk_groups_attn, groups_world=model._mk_groups_world,
        flags=(K, int(stages[0].attn.window_size), 9, kernel_version(),
               int(cfg.dspark_noise_token_id), int(cfg.vocab_size), int(cfg.dspark_num_experts_per_tok),
               int(paged.block_size)),
    )
    kernel = draft_kernel.draft_prefill if paged.at_zero else draft_kernel.draft_decode
    if _MK_SYNC and model.rank in (0, model.world_size - 1):
        logger.warning("deepseek_v41 MK_SYNC rank %d draft %s L=%d submit", model.rank,
                       "prefill" if paged.at_zero else "decode", length)
    out = entry(kernel)(*[n[name] for name in params(kernel)])
    if _MK_SYNC:
        out[0].cpu()
        if model.rank in (0, model.world_size - 1):
            logger.warning("deepseek_v41 MK_SYNC rank %d draft done", model.rank)
    output_ids, confidence = out[0], out[1]
    _throttle(model, confidence)
    return M._compiled(_ids_i64)(output_ids), None, confidence.reshape(1, K)
