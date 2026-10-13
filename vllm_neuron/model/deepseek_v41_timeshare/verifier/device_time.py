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
DELAY_ITERATIONS = 2
MAX_ITERATIONS = 2

#: The forwards a window records, as 1-based offsets into the `generate` it
#: wraps. With the settings above: the second and third.
RECORDED_FORWARDS = tuple(
    range(DELAY_ITERATIONS, DELAY_ITERATIONS + MAX_ITERATIONS)
)

#: `neuron-explorer view` reads the whole trace by default -- every
#: instruction, DMA packet and semaphore update -- which on a window this size
#: is minutes of processing and GiBs of parquet. `ExecutionInfo` and `Summary`
#: come from the execution timeline and the runtime's TPB notifications, which
#: survive all of these: on a reference profile this ingest returns
#: `total_time` and the execution rows identical to a full ingest, in 0.3 s
#: instead of 90 s.
_LIGHT_INGEST = (
    "--ignore-dma-trace",
    "--ignore-instruction-trace",
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

    @property
    def ok(self) -> bool:
        return self.device_busy_s is not None

    def as_dict(self, per_neff_top: int = 12) -> dict[str, Any]:
        ranked = sorted(self.per_neff.items(), key=lambda kv: -kv[1]["busy_s"])
        return {
            "label": self.label,
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


def summarize(capture: Capture) -> Window:
    """Ingest a staged window and sum its per-execution device time."""
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
    _prune(capture, window)
    return window


def _prune(capture: Capture, window: Window) -> None:
    """Drop the raw NTFFs once they have been read.

    A window's trace is ~1 GB a forward and a run captures six of them, so
    keeping them would leave tens of GB per run in the temp directory. The
    parquet the numbers came from stays, under the window's `ingest/`.
    """
    for path in capture.ntffs:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError as exc:  # noqa: PERF203 - one note per failure is fine
            window.notes.append(f"could not remove {path}: {exc!r}")
    if capture.window_dir is not None:
        for link in capture.window_dir.glob("session-*/*.ntff"):
            try:
                link.unlink(missing_ok=True)
            except OSError:
                pass


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
        *_LIGHT_INGEST,
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
        *_LIGHT_INGEST,
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
