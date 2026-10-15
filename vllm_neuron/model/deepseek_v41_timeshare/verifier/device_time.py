# SPDX-License-Identifier: Apache-2.0
"""On-device time for one window of engine steps, via NRT device profiles.

What this measures, and why it is not wall clock
------------------------------------------------
A `generate` on this port spends time in three places: on the device executing
NEFFs, moving data between host and device, and in the host runtime (scheduling,
sampling, the Python layer, and this port's per-layer weight streaming). Only
the first scales with the model; the rest is an artifact of *this* machine. The
target is a 16-device `trn2.48xlarge` that keeps the weights resident, so a
wall-clock TTFT measured here would mostly describe the development host. It is
not a small correction either — measured below, a three-prompt prefill window
is 62 s of device time inside a 372 s span.

`Summary.total_time` from a Neuron profile is documented as the number that
excludes the host:

    Total duration of on device execution time for the run, as reported by the
    runtime from explicit TPB start/stop notifications stored in the NTFF.
    [...] It does not include host-side overhead such as host<->device data
    movement or host runtime/framework overhead.

But for a window of many executions it still spans from the first execution's
start to the last one's end, so the host gaps *between* executions are inside
it (491 s against 62 s of real device time, on the window above). `ExecutionInfo`
has one row per execution with `execution_start_ts`/`execution_end_ts`, so
summing per-execution spans removes those gaps too. That sum —
`Window.device_busy_s` — is the on-device time this module reports, with the
window span and `Summary.total_time` kept beside it for comparison.

How a window is captured
------------------------
`profiler_engine_kwargs()` arms the engine's own NRT profiler — the one behind
vLLM's `/start_profile`, see `docs/guides/how-to-profile-workloads.md` — and
`measure()` wraps one call in `start_profile()`/`stop_profile()`. Device profile
only and rank 0 only, both to keep the capture cheap: the port disables runtime
inspection above one rank because it costs host commitment (`__init__.py`'s
`_MULTI_RANK_ENV`), and a four-rank run peaks at ~83 GiB of a 114.8 GiB
`CommitLimit`. One rank's device profile costs ~16 GiB of commitment while the
session is open, which fits in that headroom; four ranks' worth of device *and*
system profiles is not worth finding out about.

Two measured facts shape the rest of the design:

  * **A window must stay small.** One forward of this model is ~1400 NEFF
    executions, a session NTFF runs ~1 GB per forward, and the runtime's
    notification buffer drops events past ~6700 executions — at which point
    `neuron-explorer` stops processing ("Dropped notifications detected -
    stopping device profile processing after 6699 executions out of 8574") and
    the sum is a lower bound. So windows here are 1-4 forwards, `max_iterations`
    caps the capture, and a dropped window is reported as `dropped=True`
    rather than quietly short.
  * **The ingest needs the NEFFs laid out the way the runtime wrote them.**
    `neuron-explorer view -d <dir>` resolves a session's NEFFs by UID from
    `<dir>/neffs/<uid>/graph_<uid>.neff`; `-s <ntff>` alone fails with "no NEFF
    provided and no .neff file found", and the filenames are cache hashes
    rather than the UIDs the NTFF asks for, so they cannot be matched by name.
    `measure()` therefore builds one directory per window in that shape
    (NTFF symlinked, NEFFs hardlinked) and `summarize()` points `-d` at it.

Nothing here raises. A window whose capture or ingest fails comes back with
`device_busy_s=None` and the reason in `notes`, because a measurement failing
must not fail the verifier that carries it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

#: Device profile only. The system profile answers a different question (where
#: host time goes) at a much larger size, and `host_memory`/`cpu_util` are
#: host-side series this does not use.
ACTIVITIES = ("device_profile",)

#: Forwards to skip before recording starts, and how many to record. Together
#: they place a 2-forward window at a known offset inside a `generate`, which is
#: what lets each window hold forwards of a single *kind*: skipping one forward
#: turns a warm single-prompt generate (re-prefill, then decode steps) into a
#: recording of decode steps alone, and a `generate` of a throwaway prompt
#: before the real one does the same for a prefill.
#:
#: This replaced measuring whole windows and subtracting them, which this port's
#: variance does not survive: two windows doing identical prefill work measured
#: 23.8 s and 29.5 s on device, because an execution's span includes waiting for
#: weights streamed from the host and that wait moves with what else is in
#: flight. Recording whole forwards of one kind needs no subtraction.
#:
#: `NeuronWorker.execute_model` calls `WorkerProfiler.step()` **before** the
#: forward it is about to run, not after it, so the counters are offset by one
#: from the obvious reading: with `delay_iterations=D` the session opens just
#: before forward D and closes just before forward D+M, which records forwards
#: D through D+M-1. `D=1` and `D=0` both record forward 1.
#:
#: Hence D=2: forward 1 is the one that gets skipped. Verified the hard way —
#: `D=1, M=1` recorded forward 1 of each window, which made a window meant to
#: hold the needle's prefill hold a throwaway prompt's instead, and the window
#: meant to hold decode steps hold the prefill.
#:
#: M=2 records two forwards, which is also about the most this port can afford:
#: a forward is ~1400 NEFF executions and the runtime's notification buffer
#: starts dropping past ~6700.
#: The default pair: record forwards 1-2 of whatever a window wraps. A window
#: that must start later — a decode window, which has to clear every prefill
#: segment first — passes its own `delay_iterations`; see
#: `run_verifier._decode_delay`.
DELAY_ITERATIONS = 1
#: Three, not two: `_resolve_measurement` drops the first prefill pass of a
#: window as a warm-up (first-touch costs the resident-weight target pays once
#: at startup, measured at 3.8x the steady pass), so a two-forward window would
#: leave TTFT resting on a single sample. The decode window stays at two, where
#: no warm-up is dropped — see `_measure_decode`.
MAX_ITERATIONS = 3

#: The forwards a window records, as 1-based offsets into the `generate` it
#: wraps: `delay .. delay + max - 1`. With the defaults above, forwards 1-2.
RECORDED_FORWARDS = tuple(
    range(DELAY_ITERATIONS, DELAY_ITERATIONS + MAX_ITERATIONS)
)


def recorded_forwards(delay: int = DELAY_ITERATIONS, count: int = MAX_ITERATIONS) -> tuple[int, ...]:
    """Which forwards a window with this `delay_iterations` records."""
    return tuple(range(delay, delay + count))

#: What `neuron-explorer view` is told to skip.
#:
#: The **instruction and DMA traces are both kept**, and together they are the
#: measurement. Without them the only timing available is the execution *span*,
#: and on this port a span is almost entirely host-caused waiting: measured on
#: one window, 29.19 s of spans holding 1.46 s of device activity, because the
#: weights arrive from NVMe at ~155 MB/s a rank inside each execution, and
#: because a per-layer all-reduce sits armed waiting for its three peers — 20 s
#: of span for 30 ms of transfer.
#:
#: The DMA trace is kept because `DmaPacket` is, in the schema's own words, a
#: transfer "within a Neuron Device": it is the HBM->SBUF staging that every
#: kernel needs and that the 16-device target still pays. Measured on that same
#: window it is 1.03 s of the 1.46 s — two thirds of the device's real work — so
#: an earlier cut of this file that dropped it under-counted by a third. What is
#: *not* in this trace, and so needs no excluding, is the host->device copy: the
#: weight staging is an `nrt` memcpy outside the NEFF, which is why its wait
#: shows up as idle rather than as work.
#:
#: Costs ~6 min and ~2.8 GB of parquet a window, against 0.3 s and 1 MB for the
#: span-only ingest. Paid once per window after the engine is gone, which is the
#: only time there is memory for it: a lighter version of this ingest OOM'd at
#: 22 GiB RSS while four ranks held 85 GiB of the 115 GiB `CommitLimit`. The
#: heavy tables are deleted once read; see `_prune`.
_INGEST_FLAGS = (
    "--ignore-event-trace",
    "--ignore-nc-buf-usage",
    "--ignore-instruction-hierarchy",
)

#: Enough for a cold start of the tool on a multi-GB NTFF (the 4 GB reference
#: window ingests in ~40 s), short enough that a hung ingest is not forever.
_INGEST_TIMEOUT_S = 1800

#: After `stop_profile()` the runtime still has to write the NTFF out -- 4 GB of
#: it. Poll for the file to stop growing rather than guessing a sleep.
_FLUSH_TIMEOUT_S = 600
_FLUSH_QUIET_S = 5.0

#: `neuron-explorer`'s own words when the capture lost events. Both numbers are
#: worth keeping: the second is how many executions the window really had.
_DROP_RE = re.compile(
    r"Dropped notifications detected.*?after (\d+) executions out of (\d+)"
)


@dataclass
class Capture:
    """The artifacts of one profiled window, before they are read."""

    label: str
    window_dir: Path | None = None
    ntffs: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class Window:
    """One profiled window of engine steps, read.

    `device_busy_s` is the on-device time: the sum of per-execution spans, with
    the host gaps between executions excluded. `None` means the window was not
    measured -- see `notes`. `dropped` means the capture lost events and the
    sum is a lower bound.
    """

    label: str
    device_busy_s: float | None = None
    span_s: float | None = None
    summary_total_time_s: float | None = None
    executions: int = 0
    executions_total: int | None = None
    dropped: bool = False
    per_neff: dict[str, dict[str, float]] = field(default_factory=dict)
    ntffs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: The window split into passes, each priced as compute + real collective
    #: time. This is what the metrics are built from; everything above it is
    #: context, including `device_busy_s`, which is the span sum the metrics
    #: used to be built from and which is mostly host waiting.
    passes: list["Pass"] = field(default_factory=list)

    @property
    def device_ms(self) -> float:
        """Device time for the whole window: the passes' sum."""
        return sum(p.device_ns for p in self.passes) / 1e6

    @property
    def ok(self) -> bool:
        return bool(self.passes)

    def as_dict(self, per_neff_top: int = 12) -> dict[str, Any]:
        ranked = sorted(self.per_neff.items(), key=lambda kv: -kv[1]["busy_s"])
        return {
            "label": self.label,
            # The measurement. `span_sum_ms` below is the same window counted the
            # old way, kept so the gap between them stays visible.
            "device_ms": self.device_ms,
            "passes": [p.as_dict() for p in self.passes],
            "span_sum_ms": _ms(self.device_busy_s),
            "device_busy_ms": _ms(self.device_busy_s),
            "span_ms": _ms(self.span_s),
            "summary_total_time_ms": _ms(self.summary_total_time_s),
            "executions": self.executions,
            "executions_total": self.executions_total,
            "dropped": self.dropped,
            "per_neff": {
                name: {"count": v["count"], "device_busy_ms": v["busy_s"] * 1e3}
                for name, v in ranked[:per_neff_top]
            },
            "ntffs": self.ntffs,
            "notes": self.notes,
        }


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else seconds * 1e3


def recorded_count(forwards: int | None) -> int:
    """How many forwards a window actually recorded.

    The window records the forwards at `RECORDED_FORWARDS`, so a `generate`
    shorter than that records fewer — and one longer records no more, because
    `max_iterations` closes the session.
    """
    if not forwards:
        return 0
    return sum(1 for j in RECORDED_FORWARDS if j <= forwards)


def profiler_engine_kwargs(
    output_dir: str | os.PathLike[str],
    activities: tuple[str, ...] = ACTIVITIES,
    neuron_cores: list[int] | None = None,
    delay_iterations: int = DELAY_ITERATIONS,
    max_iterations: int = MAX_ITERATIONS,
) -> dict[str, Any]:
    """Engine arguments that arm the NRT profiler without starting it.

    `neuron_cores=None` leaves the worker's own default in place, which is rank
    0 only (`NeuronWorker._should_profile`). Arming costs nothing until
    `start_profile()`: `NeuronProfiler` is created lazily on the first start.

    `disable_log_stats=False` is here because the measurement needs the engine's
    own step counter — `vllm:iteration_tokens_total`'s count is how many forwards
    a window ran, which is what turns a window's device time into a per-step
    figure. The offline `LLM` disables stats by default.
    """
    from vllm.config import ProfilerConfig

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        # `"cuda"` is how the Neuron plugin's profiler is selected: vLLM's
        # `ProfilerKind` is `Literal["torch", "cuda"]` and a plugin cannot
        # extend it, so the platform reinterprets "cuda" as NRT inspect.
        "profiler_config": ProfilerConfig(
            profiler="cuda",
            delay_iterations=delay_iterations,
            max_iterations=max_iterations,
        ),
        "additional_config": {
            "neuron_profiler": {
                "activities": list(activities),
                "output_dir": str(out),
                "neuron_cores": neuron_cores,
            }
        },
        "disable_log_stats": False,
    }


def engine_counters(llm: Any) -> dict[str, float] | None:
    """Engine-wide step and token counters, for differencing around a window.

    `steps` is the number of forwards the engine has run (one observation of
    `vllm:iteration_tokens_total` per scheduler step), which is the one thing a
    device profile cannot tell us and the measurement needs.
    """
    try:
        metrics = llm.get_metrics()
    except Exception:  # noqa: BLE001 - absent counters are a note, not a failure
        return None
    out = {"steps": 0.0, "iteration_tokens": 0.0, "generation_tokens": 0.0,
           "prompt_tokens": 0.0}
    for metric in metrics:
        name = getattr(metric, "name", "")
        if name == "vllm:iteration_tokens_total":
            out["steps"] = float(getattr(metric, "count", 0) or 0)
            out["iteration_tokens"] = float(getattr(metric, "sum", 0) or 0)
        elif name == "vllm:generation_tokens":
            out["generation_tokens"] = float(getattr(metric, "value", 0) or 0)
        elif name == "vllm:prompt_tokens":
            out["prompt_tokens"] = float(getattr(metric, "value", 0) or 0)
    return out


def engine_max_num_seqs(llm: Any) -> int | None:
    """The batch size the engine settled on, which is not the one asked for.

    This port halves it to 1 above one rank (`__init__.py`'s
    `check_and_update_config`: "4 rank(s) — max_num_seqs 4 -> 1"), and that
    decides whether a decode step advances one sequence or all of them — which
    is the difference between a per-sequence tokens-per-step figure and one
    several times too small.
    """
    try:
        return int(llm.llm_engine.vllm_config.scheduler_config.max_num_seqs)
    except Exception:  # noqa: BLE001 - absent is a note, not a failure
        return None


def measure(
    llm: Any,
    output_dir: str | os.PathLike[str],
    label: str,
    fn: Callable[[], Any],
) -> tuple[Any, Capture]:
    """Run `fn()` with a profile session open and collect its artifacts.

    Returns `fn`'s own result and the capture. Reading the capture is
    `summarize()`, deliberately separate: the ingest peaks at ~17 GiB of host
    memory, which is not something to ask for while an engine is resident.

    `fn` runs whether or not profiling can be started, and its exceptions
    propagate: the caller's work is the point, the measurement is not.
    """
    out = Path(output_dir)
    capture = Capture(label=label)
    before = _ntffs(out)

    started = False
    try:
        llm.start_profile()
        started = True
    except Exception as exc:  # noqa: BLE001 - a profiler that will not start is a note
        capture.notes.append(f"start_profile failed: {exc!r}")

    try:
        result = fn()
    finally:
        if started:
            try:
                llm.stop_profile()
            except Exception as exc:  # noqa: BLE001
                capture.notes.append(f"stop_profile failed: {exc!r}")

    if not started:
        return result, capture

    new = _settled_new_ntffs(out, before, capture.notes)
    if not new:
        capture.notes.append(
            f"no new .ntff under {out} after the window — profiling produced nothing"
        )
        return result, capture

    capture.window_dir, note = _stage_window(out, label, new)
    if note:
        capture.notes.append(note)
    capture.ntffs = new
    return result, capture


def summarize(capture: Capture, num_layers: int | None = None) -> Window:
    """Ingest a staged window and sum its per-execution device time.

    `num_layers` is how many per-layer executions make one pass; pass
    `layer_count(model)` so it comes from the checkpoint rather than from a
    constant in this file.
    """
    window = Window(label=capture.label, notes=list(capture.notes),
                    ntffs=[str(p) for p in capture.ntffs])
    if capture.window_dir is None:
        if not window.notes:
            window.notes.append("nothing was captured for this window")
        return window

    dest = capture.window_dir / "ingest"
    rows, summary, drops, note = _ingest(capture.window_dir, dest)
    if note:
        window.notes.append(note)
    window.summary_total_time_s = summary
    if drops:
        window.dropped = True
        window.executions_total = drops[1]
        window.notes.append(
            f"the capture dropped notifications: {drops[0]} of {drops[1]} "
            "executions were processed, so the device time below is a lower bound"
        )
    if not rows:
        return window

    window.executions = len(rows)
    if window.executions_total is None:
        window.executions_total = len(rows)
    busy = 0.0
    for row in rows:
        seconds = max(0.0, (row["end"] - row["start"]) / 1e9)
        busy += seconds
        entry = window.per_neff.setdefault(row["neff"], {"count": 0, "busy_s": 0.0})
        entry["count"] += 1
        entry["busy_s"] += seconds
    window.device_busy_s = busy
    window.span_s = (
        max(r["end"] for r in rows) - min(r["start"] for r in rows)
    ) / 1e9
    # The measurement proper: passes, each priced as engine-busy plus real
    # collective time. `device_busy_s` above is kept for contrast — it is the
    # span sum, which on this port is 15-60x larger because the host waits are
    # inside the executions.
    window.passes, pass_notes = passes_from_ingest(
        dest, num_layers if num_layers else _LAYERS_FALLBACK
    )
    window.notes.extend(pass_notes)
    _prune(capture, window)
    return window


#: Ingested tables above this stay only until the window is priced. `DmaPacket`
#: is 47 million rows and ~3 GB of the ~3.2 GB a prefill window ingests to; the
#: small tables (`Summary`, `ExecutionInfo`, `TensorInfo`, `NeffHeader`, and the
#: merged-interval sources once read) are worth keeping for a post-mortem.
_INGEST_KEEP_BYTES = 50 * 1024 * 1024


def _prune(capture: Capture, window: Window) -> None:
    """Drop the raw NTFFs and the bulky parquet once they have been read.

    A window's trace is ~1 GB a forward and a run captures six windows, and the
    parquet it ingests to is another ~16 GB a run. Neither is needed once the
    passes are priced, and a loop calling this every iteration fills the disk:
    measured at 16 GB for one run against 245 GB free, i.e. the run would have
    died of `ENOSPC` around iteration 10 of 13. The small tables stay.
    """
    for path in capture.ntffs:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError as exc:  # noqa: PERF203 - one note per failure is fine
            window.notes.append(f"could not remove {path}: {exc!r}")
    if capture.window_dir is None:
        return
    for link in capture.window_dir.glob("session-*/*.ntff"):
        try:
            link.unlink(missing_ok=True)
        except OSError:
            pass
    reclaimed = 0
    for table in capture.window_dir.glob("ingest/*.parquet"):
        try:
            size = table.stat().st_size
            if size > _INGEST_KEEP_BYTES:
                table.unlink()
                reclaimed += size
        except OSError:
            pass
    if reclaimed:
        window.notes.append(
            f"pruned {reclaimed / 2**30:.1f} GiB of ingested parquet; the tables "
            "under 50 MiB stay for a post-mortem"
        )


# ── profile artifacts ────────────────────────────────────────────────────────


def _ntffs(out: Path) -> set[Path]:
    return set(out.glob("**/*.ntff")) if out.exists() else set()


def _settled_new_ntffs(out: Path, before: set[Path], notes: list[str]) -> list[Path]:
    """The NTFFs this window added, once the runtime has finished writing them."""
    deadline = time.monotonic() + _FLUSH_TIMEOUT_S
    sizes: dict[Path, int] = {}
    quiet_since: float | None = None
    while time.monotonic() < deadline:
        new = sorted(_ntffs(out) - before)
        now = {p: p.stat().st_size for p in new if p.exists()}
        if new and now == sizes and all(v > 0 for v in now.values()):
            if quiet_since is None:
                quiet_since = time.monotonic()
            elif time.monotonic() - quiet_since >= _FLUSH_QUIET_S:
                return new
        else:
            quiet_since = None
            sizes = now
        time.sleep(1.0)
    new = sorted(_ntffs(out) - before)
    if new:
        notes.append(
            f"NTFF size was still changing after {_FLUSH_TIMEOUT_S}s; read it anyway"
        )
    return new


def _stage_window(out: Path, label: str, ntffs: list[Path]) -> tuple[Path | None, str | None]:
    """Lay one window's artifacts out the way `view -d` expects them.

        <out>/windows/<label>/<session>/<name>.ntff   symlink to the capture
        <out>/windows/<label>/neffs/<uid>/*.neff      hardlinks to the copies

    Symlink for the NTFF because it is gigabytes; hardlinks for the NEFFs
    because the tool's recursive search for them does not follow a symlinked
    directory. Per window, because `-d` ingests every profile under the
    directory it is given and the sessions all land under one `output_dir`.
    """
    window_dir = out / "windows" / label
    try:
        for i, ntff in enumerate(ntffs):
            session = window_dir / f"session-{i}"
            session.mkdir(parents=True, exist_ok=True)
            link = session / ntff.name
            if not link.exists():
                link.symlink_to(ntff)
        staged = 0
        for neff in out.glob("neffs/*/*.neff"):
            dest = window_dir / "neffs" / neff.parent.name / neff.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                os.link(neff, dest)
            staged += 1
        if not staged:
            return window_dir, (
                f"no NEFFs under {out / 'neffs'} — the runtime copies the ones a "
                "session executed beside it, and the ingest cannot resolve a "
                "profile without them"
            )
    except OSError as exc:
        return None, f"could not stage the window's artifacts: {exc!r}"
    return window_dir, None


def _ingest(
    window_dir: Path, dest: Path
) -> tuple[list[dict[str, Any]], float | None, tuple[int, int] | None, str | None]:
    """`neuron-explorer view -d` → (execution rows, total_time, drops, note)."""
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        "neuron-explorer",
        "view",
        "--output-format",
        "parquet",
        "--output-file",
        str(dest),
        # Without this the tool leaves a UI/API server running and never
        # returns; with it, it ingests and exits.
        "--ingest-only",
        *_INGEST_FLAGS,
        "-d",
        str(window_dir),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_INGEST_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        return [], None, None, f"ingest timed out after {_INGEST_TIMEOUT_S}s"
    except FileNotFoundError:
        return [], None, None, "neuron-explorer is not on PATH — cannot read the profile"

    output = f"{proc.stdout}\n{proc.stderr}"
    match = _DROP_RE.search(output)
    drops = (int(match.group(1)), int(match.group(2))) if match else None

    rows, note = _execution_rows(dest)
    total = _summary_total_time(dest)
    if not rows and proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        note = f"ingest failed: {' / '.join(t[:200] for t in tail)}"
    return rows, total, drops, note


#: Fallback layer count, used only if the checkpoint's own config cannot be
#: read. The per-layer graph runs once per layer, so this is how many of its
#: executions make one pass — and it must not be a constant in practice:
#: hardcoding this model's 40 would silently misgroup passes for any checkpoint
#: with a different depth, which is exactly the kind of input-specific
#: assumption this file should not contain. `layer_count()` reads it from the
#: checkpoint.
_LAYERS_FALLBACK = 40


def layer_count(model: str | os.PathLike[str] | None) -> int:
    """`num_hidden_layers` from the checkpoint's own config.

    Read rather than assumed, so a checkpoint of a different depth splits into
    passes correctly instead of quietly mis-splitting.
    """
    if model is None:
        return _LAYERS_FALLBACK
    try:
        import json

        config = json.loads((Path(model) / "config.json").read_text())
    except Exception:  # noqa: BLE001 - the fallback is the point
        return _LAYERS_FALLBACK
    for key in ("num_hidden_layers", "n_layers", "num_layers"):
        for where in (config, config.get("text_config") or {}):
            value = where.get(key)
            if isinstance(value, int) and value > 0:
                return value
    return _LAYERS_FALLBACK


def _merge(starts, ends):
    """Disjoint intervals covering the same time as `starts`/`ends`.

    numpy rather than Python: `DmaPacket` is 68 million rows on one window, and
    building that many Python ints costs more memory than the engine the
    measurement is about.
    """
    import numpy as np

    if len(starts) == 0:
        return np.empty(0, dtype="int64"), np.empty(0, dtype="int64")
    order = np.argsort(starts, kind="stable")
    s, e = starts[order], ends[order]
    reach = np.maximum.accumulate(e)
    head = np.empty(len(s), dtype=bool)
    head[0] = True
    head[1:] = s[1:] > reach[:-1]
    idx = np.flatnonzero(head)
    return s[idx], np.maximum.reduceat(e, idx)


def _traced_fraction(dest: Path, lo: int, hi: int, executions: int) -> float | None:
    """What share of a pass's executions the instruction trace actually covered.

    **Reported, never charged.** The NRT instruction trace is bounded: measured
    on one real prefill window, the first 554 executions of the per-layer graph
    carry ~5 100 instructions each and every one after carries a 362-424
    instruction skeleton, while their wall spans stay identical (1.23 ms median
    in all three passes). `ActiveTime` is built by merging instruction intervals,
    so it thins out with the trace — 1 023 activity rows an execution before the
    cutoff, 359-368 after — and a pass priced from what was traced therefore
    reads low once the window outruns the budget: 1 171 ms for the first pass of
    a window against 300 ms for the second, for the same 512 rows of work.

    The metric deliberately stays the *measured* union of compute, DMA and
    collective time, so this number is here to make that effect visible rather
    than to correct it. Two things it is good for: a pass at a few percent is
    mostly untraced and its absolute value means little (the relative one is
    still reproducible to ~1%), and a change in coverage *between* iterations is
    a warning that two numbers are not comparable — fusing kernels cuts the
    execution count, which moves coverage, which moves the figure for a reason
    that is not speed.

    `None` means there is no instruction table to judge coverage from.
    """
    import numpy as np

    try:
        import pyarrow.parquet as pq

        t = pq.read_table(dest / "Instruction.parquet", columns=["start_ts"])
    except Exception:  # noqa: BLE001 - no table, no coverage signal
        return None
    if not executions:
        return None
    i_s = t.column("start_ts").to_numpy(zero_copy_only=False).astype("int64")
    inpass = int(((i_s >= lo) & (i_s < hi)).sum())
    # A fully traced execution of the heavy graph runs ~5 100 instructions; the
    # skeleton is ~363. Coverage is where this pass's average sits between them.
    return min(1.0, (inpass / executions) / _FULL_TRACE_INSTRUCTIONS)


#: Instructions in a fully traced execution of the per-layer graph, measured.
#: Only the scale matters here: it turns instructions-per-execution into a
#: fraction that reads 1.0 when everything was traced.
_FULL_TRACE_INSTRUCTIONS = 5100.0


def _covered(starts, ends, lo: int, hi: int) -> int:
    """How much of `[lo, hi)` the disjoint intervals cover."""
    import numpy as np

    if len(starts) == 0 or hi <= lo:
        return 0
    first = int(np.searchsorted(ends, lo, side="right"))
    last = int(np.searchsorted(starts, hi, side="left"))
    if last <= first:
        return 0
    a = np.maximum(starts[first:last], lo)
    b = np.minimum(ends[first:last], hi)
    return int(np.clip(b - a, 0, None).sum())


@dataclass
class Activity:
    """Every interval in which the device was doing something, by source.

    Three sources, and all three are device work:

    * **engines** — `ActiveTime`, the compute.
    * **dma** — `DmaPacket`, transfers *within* the device. Mostly HBM->SBUF
      weight staging, which a resident-weight machine still pays, so it counts.
      On one window it was 1.03 s of 1.46 s of total activity.
    * **collectives** — `CcOp`, what an all-reduce actually transferred, as
      opposed to how long its execution sat armed waiting for peers (30 ms
      against 20 s).

    The union is what "the device was busy" means, because the engines and the
    DMA queues run in parallel; summing them double-counts (per-source sum
    2.5 s against a 1.46 s union on the same window).
    """

    engine: tuple
    dma: tuple
    collective: tuple
    union: tuple

    @classmethod
    def read(cls, dest: Path) -> "Activity":
        import numpy as np

        def intervals(table: str) -> tuple:
            try:
                import pyarrow.parquet as pq

                t = pq.read_table(dest / f"{table}.parquet",
                                  columns=["start_ts", "end_ts"])
            except Exception:  # noqa: BLE001 - a missing table is no activity
                return (np.empty(0, dtype="int64"), np.empty(0, dtype="int64"))
            s = t.column("start_ts").to_numpy(zero_copy_only=False).astype("int64")
            e = t.column("end_ts").to_numpy(zero_copy_only=False).astype("int64")
            keep = e > s
            return _merge(s[keep], e[keep])

        engine = intervals("ActiveTime")
        dma = intervals("DmaPacket")
        collective = intervals("CcOp")
        both = (
            np.concatenate([engine[0], dma[0], collective[0]]),
            np.concatenate([engine[1], dma[1], collective[1]]),
        )
        return cls(engine=engine, dma=dma, collective=collective,
                   union=_merge(*both))


@dataclass
class Pass:
    """One pass through the model's layers, and what the device really spent on it.

    `device_ns` is the **union of engine, device-DMA and collective activity**
    clipped to this pass: every microsecond in which the device was doing
    something, counted once. Deliberately not the execution span, which here is
    15-30x larger and almost all host waiting — weights arriving from NVMe, and
    an all-reduce sitting armed for its peers. Neither happens on the 16-device
    target, where the weights are resident.

    It is a **lower bound** on what that target would spend, one-sided in a known
    way: idle *inside* a kernel is excluded along with the host waits, because
    the host->device copy the device waits for is absent from the device profile,
    so a bubble and a weight wait are the same observation. `utilisation` keeps a
    bubbly kernel visible — engine time over device time, which reads low when a
    kernel mostly waits on its own DMA.
    """

    index: int
    kind: str                 # "prefill" | "decode" | "unknown"
    rows: int | None          # the pass's token width, off the NEFF
    executions: int
    span_ns: int              # first execution's start to last one's end
    device_ns: int            # union of all device activity: the measurement
    engine_ns: int            # engines only
    dma_ns: int               # within-device transfer only
    collective_ns: int        # what the collectives actually transferred
    busy_ns: int              # sum of execution spans — the old, contaminated number
    #: Share of this pass's executions the instruction trace covered. Reported,
    #: never charged — see `_traced_fraction` for what it is good for.
    traced_fraction: float | None = None

    @property
    def utilisation(self) -> float:
        """Engine time over device time: low means the device waited on itself."""
        return self.engine_ns / self.device_ns if self.device_ns else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "rows": self.rows,
            "executions": self.executions,
            "device_ms": self.device_ns / 1e6,
            "engine_ms": self.engine_ns / 1e6,
            "dma_ms": self.dma_ns / 1e6,
            "collective_ms": self.collective_ns / 1e6,
            "utilisation": self.utilisation,
            "traced_fraction": self.traced_fraction,
            "span_sum_ms": self.busy_ns / 1e6,
            "span_ms": self.span_ns / 1e6,
        }


def passes_from_ingest(dest: Path, num_layers: int = _LAYERS_FALLBACK) -> tuple[list[Pass], list[str]]:
    """Split an ingested window into passes and price each one.

    Five tables, each answering one question:

    * `ExecutionInfo` — when each NEFF execution ran, which is how the window is
      split into passes: at every `num_layers`-th execution of the per-layer
      graph (the NEFF running a multiple of `num_layers` times that costs most).
    * `ActiveTime`, `DmaPacket`, `CcOp` — when the device was busy, and at what.
      Their union inside a pass is the pass's device time; see `Activity`.
    * `TensorInfo` — the `hidden_states` width the NEFF declares, which says
      prefill (a KV segment, e.g. 512) from decode (`1 + k`, e.g. 6). Read off
      the compiled graph rather than inferred from scheduling order: a
      2008-token prompt prefills as four 512-row passes, so "the second forward"
      is still prefill, which is what an earlier cut of this got wrong.
    """
    notes: list[str] = []
    execs = _table(dest, "ExecutionInfo")
    if not execs:
        return [], [f"no ExecutionInfo in {dest}"]

    rows = sorted(
        (s, e, n)
        for s, e, n in zip(
            execs.get("execution_start_ts", []),
            execs.get("execution_end_ts", []),
            execs.get("neff_name", []),
        )
        if s is not None and e is not None
    )
    if not rows:
        return [], [f"no executions with timestamps in {dest}"]

    marker, marker_note = _pass_marker(rows, num_layers)
    if marker_note:
        notes.append(marker_note)
    # Every `num_layers`-th marker execution, not every one: the marker runs once
    # per *layer*, so splitting at each would price the window per layer and
    # report 80 "passes" for a window holding two.
    hits = [s for s, _, n in rows if n == marker] if marker else []
    bounds = hits[::num_layers] if hits else [rows[0][0]]

    try:
        activity = Activity.read(dest)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return [], [f"could not read the activity tables: {exc!r}"]

    width, width_note = _declared_width(dest)
    if width_note:
        notes.append(width_note)

    out: list[Pass] = []
    for i, start in enumerate(bounds):
        stop = bounds[i + 1] if i + 1 < len(bounds) else rows[-1][1] + 1
        segment = [r for r in rows if start <= r[0] < stop]
        if not segment:
            continue
        end = max(r[1] for r in segment)
        out.append(
            Pass(
                index=i,
                kind=_kind_for(width),
                rows=width,
                executions=len(segment),
                span_ns=end - start,
                device_ns=_covered(*activity.union, start, end),
                engine_ns=_covered(*activity.engine, start, end),
                dma_ns=_covered(*activity.dma, start, end),
                collective_ns=_covered(*activity.collective, start, end),
                busy_ns=sum(b - a for a, b, _ in segment),
                traced_fraction=_traced_fraction(dest, start, end, len(segment)),
            )
        )
    return out, notes


def _pass_marker(rows: list[tuple[int, int, str]], num_layers: int) -> tuple[str | None, str | None]:
    """The NEFF that runs once per layer, which is what delimits a pass."""
    import collections

    count: dict[str, int] = collections.Counter(n for _, _, n in rows)
    cost: dict[str, int] = collections.Counter()
    for a, b, n in rows:
        cost[n] += b - a
    candidates = [n for n, c in count.items() if c and c % num_layers == 0 and c >= num_layers]
    if not candidates:
        return None, (
            f"no NEFF runs a multiple of {num_layers} times, so the window could not be "
            "split into passes; it is priced as one pass"
        )
    return max(candidates, key=lambda n: cost[n]), None


def _intervals(table: dict[str, list[Any]] | None, start: str, end: str) -> list[tuple[int, int]]:
    if not table:
        return []
    return [
        (a, b)
        for a, b in zip(table.get(start, []), table.get(end, []))
        if a is not None and b is not None and b > a
    ]


def _declared_width(dest: Path) -> tuple[int | None, str | None]:
    """The token width this window's passes computed, off the NEFF's own tensors.

    `hidden_states` is the activation the per-layer graph takes, so its middle
    dimension is the number of rows the pass computed: a KV segment size for a
    prefill, `1 + num_speculative_tokens` for a decode step. Reading it from the
    NEFF is what makes "this is a prefill pass" a fact rather than an inference
    from scheduling order.
    """
    table = _table(dest, "TensorInfo")
    if not table:
        return None, "no TensorInfo, so the pass width could not be read off the NEFF"
    widths: set[int] = set()
    for name, kind, shape in zip(
        table.get("kernel_param_name", []),
        table.get("type", []),
        table.get("shape", []),
    ):
        if name != "hidden_states" or kind != "IN" or not shape:
            continue
        parts = str(shape).strip("[]").split()
        if len(parts) >= 2 and parts[-2].isdigit():
            widths.add(int(parts[-2]))
    if not widths:
        return None, "no `hidden_states` input found, so the pass width is unknown"
    if len(widths) > 1:
        return max(widths), (
            f"the window holds passes of more than one width {sorted(widths)}; they are "
            "priced together under the largest, which mixes prefill and decode"
        )
    return widths.pop(), None


def _kind_for(width: int | None) -> str:
    """prefill or decode, from the pass width.

    A decode step is narrow — one token plus the drafter's `k` — and a prefill
    segment is a KV segment size. The boundary is set well above any plausible
    `1 + k` and well below the smallest segment this build supports (512).
    """
    if width is None:
        return "unknown"
    return "decode" if width <= 64 else "prefill"


def _table(dest: Path, name: str) -> dict[str, list[Any]] | None:
    path = dest / f"{name}.parquet"
    if not path.is_file():
        return None
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    return pq.read_table(path).to_pydict()


def _execution_rows(dest: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not (dest / "ExecutionInfo.parquet").is_file():
        return [], None
    table = _table(dest, "ExecutionInfo")
    if table is None:
        return [], (
            "pyarrow is not installed, so the per-execution table cannot be read. "
            "`Summary.total_time` is reported instead, but it is a different "
            "quantity — it includes the host gaps between executions — so no "
            "on-device figure is derived from it."
        )
    rows = []
    for name, start, end in zip(
        table.get("neff_name", []),
        table.get("execution_start_ts", []),
        table.get("execution_end_ts", []),
    ):
        if start is None or end is None:
            continue
        rows.append({"neff": name or "<no neff>", "start": start, "end": end})
    return rows, None


def _summary_total_time(dest: Path) -> float | None:
    table = _table(dest, "Summary")
    if table:
        values = [v for v in table.get("total_time", []) if v is not None]
        if values:
            return float(sum(values))
    return None


def read_summary_json(window_dir: Path) -> float | None:
    """`Summary.total_time` without pyarrow, for when the parquet cannot be read.

    Kept separate from the on-device figure on purpose: this number includes
    the host gaps between executions (measured: 491 s against 62 s of device
    time), so it is recorded as context and never as TTFT.
    """
    cmd = [
        "neuron-explorer",
        "view",
        "--output-format",
        "summary-json",
        *_INGEST_FLAGS,
        "-d",
        str(window_dir),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_INGEST_TIMEOUT_S
        )
        data = json.loads(proc.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return None
    totals = [
        v["total_time"]
        for v in data.values()
        if isinstance(v, dict) and v.get("total_time") is not None
    ]
    return float(sum(totals)) if totals else None
