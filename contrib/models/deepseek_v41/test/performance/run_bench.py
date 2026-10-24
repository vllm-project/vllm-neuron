#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the DeepSeek-V4.1-Flash port with `vllm bench serve`.

The method the onboarding guide prescribes — "Benchmark and tune performance":
serve the model, then drive it with `vllm bench serve --dataset-name random`.
Random input is the point: it fixes the token count exactly, so a cell of the
matrix measures the shape it names rather than whatever a corpus happened to
tokenize to, and no two requests share a prefix that the prefix cache could
serve instead of computing.

    python contrib/models/deepseek_v41/test/performance/run_bench.py \\
        --model <checkpoint-dir>

**The matrix.** Input 512 / 1k / 2k / 8k, output 128, batch 1 and 10 — eight
cells. Per cell it reports

    ttft_p50_ms      time to first token, median
    tpot_p50_ms      time per output token, median (decode only; the first
                     token is TTFT's, not TPOT's)
    decode_tok_s     output tokens per second, aggregate over the batch

as `##autohelix[...]` lines for the loop to read back, and as a table at the
end for a person to read. Medians rather than means because one slow request —
a cold bucket, a compile — otherwise moves the figure more than the change
being measured.

**One server per batch size, not per cell.** `max_num_seqs` is an
engine-construction argument and it decides the compiled decode shapes, so a
batch-1 measurement taken on an engine built for 10 is not a batch-1
measurement. Within a batch size the four input lengths share an engine: input
length changes the prefill shape, which this port segments anyway, and a cell's
first request pays for its own bucket. That is two engine builds instead of
eight.

**`--model` is required and has no default.** The same checkpoint the verifier
serves, and it is checked: `architectures[0]` must be `DeepseekV41ForCausalLM`,
so a benchmark cannot quietly measure a different or smaller model. A speed
number against the wrong weights is worse than no number.

NOTE: this runs the model, so it needs the Trainium devices to itself.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

#: The test folder, which holds `model_package` and is the import root for the
#: `performance` package this module belongs to.
_TEST_ROOT = Path(__file__).resolve().parent.parent
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

import model_package  # noqa: E402

# Import the port here, in the orchestrator, not only in the server child. This
# is not new behaviour: the module used to be run as
# `-m vllm_neuron.model.deepseek_v41.bench.run_bench`, and `runpy` imports every
# parent package first, so the port was always imported in this process too. Its
# import-time `apply_multi_rank_host_tuning` sets the `_MULTI_RANK_ENV` variables,
# and because `_serve` passes no `env`, the server inherits them from here.
#
# `load()` also exports the import root into `PYTHONPATH`, which is how the
# `-m performance._serve_main` child resolves both `performance` and the model
# package — `_serve` deliberately passes no `env` of its own.
model_package.load()

#: The matrix. Overridable, but these are the sizes the deployment is specified
#: against.
DEFAULT_INPUT_LENS = (512, 1024, 2048, 8192)
DEFAULT_BATCH_SIZES = (1, 10)
DEFAULT_OUTPUT_LEN = 128

#: Tokens per prompt in the prose cell. Acceptance is sampled over decode
#: steps, so this is the sample size: 96 tokens over the verifier's 4 cases
#: at ~1.5 accepted a step is ~250 draft decisions.
DEFAULT_PROSE_OUTPUT_LEN = 96

#: Seconds of total silence from the engine before it is killed. A healthy
#: 64-rank forward writes a line every ~380 ms, and the quietest legitimate
#: stretch is a cold graph compile, measured at up to ~20 min on this port --
#: hence 25 min rather than something tighter.
DEFAULT_STALL_TIMEOUT = 1500

#: Lines that mean the run is over, whatever the clock says. Matched against the
#: tail of the server log while a cell runs, because waiting out a timeout after
#: one of these appears buys nothing and costs the devices.
#:
#: Taken from an actual wedge, 2026-10-18 03:42:28. The cell was healthy --
#: `max_num_seqs 32 -> 1` in force, one sequence at a time, 37 requests already
#: answered -- and then rank 51 alone failed:
#:
#:   TDRV:ddrs_build_ring_from_template  Failed to allocate memory for dma ring
#:                                       for queue qSPIO0_0
#:   TDRV:dmem_buf_copyin                Copy from buffer to memory failed:
#:                                       timeout (ret=-110)
#:   RuntimeError: Failed to schedule neff execution. status=1 message=Unknown
#:
#: The other 63 ranks then sat in a collective waiting for a rank that was never
#: coming. Nothing about it was a batching deadlock: execution was sequential
#: throughout. It was one rank running out of device memory for DMA descriptor
#: rings, three seconds after a normal forward.
_FATAL_PATTERNS = (
    "Failed to schedule neff execution",
    "Failed to allocate memory for dma ring",
    "Failed to build dma rings for io",
    "WorkerProc hit an exception",
    "EngineCore encountered a fatal error",
)

#: Bytes of the server log tail scanned each poll. The signatures above appear
#: within a few KB of the failure, and reading the whole file every 5 s would
#: cost more than the check is worth once a log passes a few hundred MB.
_FATAL_SCAN_BYTES = 256 * 1024


def _fatal_in_log(path: Path) -> str | None:
    """The first fatal signature in the log's tail, or None."""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > _FATAL_SCAN_BYTES:
                fh.seek(size - _FATAL_SCAN_BYTES)
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    for pattern in _FATAL_PATTERNS:
        if pattern in tail:
            return pattern
    return None

#: Shakespeare's sonnets, 2155 verse lines, built from Project Gutenberg #1041.
#: `sonnet.txt` does not ship with vLLM, so it has to be supplied.
DEFAULT_SONNET_PATH = "/mnt/nvme/datasets/sonnet.txt"

#: Slack over the time the performance targets imply a cell should take. 3x,
#: because a cell that is merely 3x slower than target is a failing measurement
#: worth having, and one that is 10x slower is a wedge worth killing.
CELL_TIMEOUT_SLACK = 3.0

#: Floor for a derived cell timeout, which sizes the *measured* pass. By then
#: the warm-up has compiled the cell's shapes, so this is steady-state work.
CELL_TIMEOUT_FLOOR = 1800

#: Added to the warm-up's timeout only, because the warm-up is where compile is
#: paid and the derivation above sizes steady-state work with no allowance for
#: it. An optimised kernel costs more to compile, not less, so the budget has to
#: leave room for one.
#:
#: Sized from measurement. A 64-rank compile of a single new shape has been
#: measured at 1045-1167 s. A cell re-keys more than one: iteration 5's
#: `BQ_2CORE=fp4,fp8` split re-keyed 125 distinct graph keys for one in512 cell,
#: because fp8 appears in every graph holding an attention projection. And
#: iteration 6 lost its whole gate to this: the in8192 warm-up was still
#: amortising compile -- 253.98 s then 116.22 s a request, ETA falling 29:37 to
#: 11:37 -- when the flat 1800 s fired, so the measured pass ran cold, paid the
#: compile again, timed out too, and took the acceptance probe with it. Four
#: judges and the verifier had passed; the iteration was rejected for a timeout
#: during compilation.
#:
#: 2400 s is ~2x the worst measured single-shape compile, which covers a cell
#: that re-keys several shapes without letting a genuine wedge sit for hours.
#: The measured pass keeps the tight budget, so a wedge there is still caught in
#: CELL_TIMEOUT_FLOOR.
CELL_WARMUP_COMPILE_ALLOWANCE = 2400


def _cell_timeout(args, batch: int, output_len: int) -> int:
    """How long one cell gets, derived from the performance targets.

    A flat timeout is wrong in both directions: too tight for an 8k prefill at
    batch 10, too loose for 512 at batch 1 -- where it means a wedge sits for
    the full budget. So the budget is what the targets say the work should cost.

    `--bench-timeout` overrides it, for when a cell needs a number rather than
    a derivation.
    """
    if args.bench_timeout:
        return args.bench_timeout
    # Requests run `batch` at a time, so `ceil(n / batch)` rounds deep. Each
    # round costs one TTFT plus `output_len` tokens at the target TPOT.
    rounds = -(-(args.num_prompts or _num_prompts(batch)) // batch)
    per_round = TARGET_TTFT_S + output_len * TARGET_TPOT_S
    return int(max(CELL_TIMEOUT_FLOOR, CELL_TIMEOUT_SLACK * rounds * per_round))


#: The deployment's bars, here only to size the timeouts.
#: TTFT is the 1.5 s-per-1k bar at the longest input in the matrix;
#: TPOT is 1/50 s, the aggregate decode bar taken per stream as an upper bound.
TARGET_TTFT_S = 12.0
TARGET_TPOT_S = 0.25

#: `--served-model-name`, and what every client here asks for.
SERVED_NAME = "deepseek-v41-flash"

#: Requests per cell. Enough that a median means something, few enough that the
#: 8k x 10 cell does not dominate the run: at batch `b` this is `max(8, 2 * b)`,
#: so batch 1 sends 8 sequentially and batch 10 sends 20 as two full rounds.
def _num_prompts(batch: int) -> int:
    """How many requests a cell sends.

    **One wave, not two.** This was `2 * batch`, which the warm-up wants (a
    prefill running alongside `batch - 1` decodes is a shape a simultaneous
    batch never produces) and which the *measurement* cannot afford: with
    `--max-concurrency b` and `2b` requests, round two cannot start until round
    one finishes decoding, so half the TTFT sample is queue wait and
    `output_throughput` -- tokens over the run's whole wall clock -- is divided
    by a serial structure rather than a concurrent one.

    Measured 2026-10-21 at in512_bs10, same engine, same session: 20 prompts
    gave 5.01 tok/s with mean TTFT 4207 ms and P90 7070 ms, while 10 prompts
    gave 10.74 tok/s. The second round also **keyed 104 extra compiled graphs**
    inside the measured pass, because the mixed prefill/decode states only a
    second round reaches had never been warmed.

    The mixed shapes are still traversed, by `--num-warmups` in `_bench_serve`,
    which is where warming belongs.
    """
    return max(8, batch)


#: Context the engine is built with: the largest input plus the output plus
#: slack, so one engine covers every input length in its batch-size group.
CONTEXT_HEADROOM = 128

#: The architecture this benchmark is for. See the module docstring.
ARCH = "DeepseekV41ForCausalLM"

#: Anchored to this file, like the verifier's report: the benchmark owns no path
#: outside this package except the checkpoint it is told about.
DEFAULT_REPORT = Path(__file__).resolve().parent / "bench_report.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        required=True,
        help="checkpoint directory (required; no default, so no path outside "
        "this repository is baked in). The same checkpoint the verifier "
        "serves; see the model README's \"Example Checkpoints\".",
    )
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="defaults to the port's own `config.TENSOR_PARALLEL_SIZE`, so the "
        "benchmark measures the parallelism the implementation chose",
    )
    p.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=None,
        help="defaults to the port's `config.PIPELINE_PARALLEL_SIZE` (1, or 8 under "
        "DEEPSEEK_V41_SCHEME=tp8pp8)",
    )
    p.add_argument("--input-lens", default=",".join(map(str, DEFAULT_INPUT_LENS)))
    p.add_argument("--batch-sizes", default=",".join(map(str, DEFAULT_BATCH_SIZES)))
    p.add_argument("--output-len", type=int, default=DEFAULT_OUTPUT_LEN)
    p.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="requests per cell; default max(8, 2 * batch)",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="engine context. Default: the largest input + output + "
        f"{CONTEXT_HEADROOM} of slack. Pass it to run a subset of the matrix on "
        "the *same* engine shape the full matrix uses — otherwise a smoke run "
        "over one input length builds a smaller context and its compiled "
        "graphs are not the ones the full run needs.",
    )
    p.add_argument("--port", type=int, default=8321)
    p.add_argument(
        "--server-timeout",
        type=int,
        default=5400,
        help="seconds to wait for the server to become healthy. Generous "
        "because a cold compile cache builds this model's graphs on the way "
        "up, and 64 ranks share one cache.",
    )
    p.add_argument(
        "--bench-timeout",
        type=int,
        default=0,
        help="seconds for one cell. 0 (default) derives it from the performance targets "
             "per cell — see `_cell_timeout`. A flat number is wrong in both directions: "
             "too tight for an 8k prefill at batch 10, and too loose at 512 x 1, where it "
             "means a wedged engine sits for the whole budget.",
    )
    p.add_argument(
        "--prose-dataset", default="sharegpt", choices=("sharegpt", "verifier"),
        help="text for the acceptance probe. sharegpt is real dialogue; verifier falls "
             "back to the correctness gate's own four cases.",
    )
    p.add_argument(
        "--dataset-path", default=DEFAULT_SONNET_PATH,
        help="the sonnet text the matrix is built from. Plain lines of English; "
             "`SonnetDataset` does `readlines()` and samples them.",
    )
    p.add_argument(
        "--stall-timeout", type=int, default=DEFAULT_STALL_TIMEOUT,
        help="kill the engine when it has written nothing for this many seconds. The "
             "safeguard against a wedged collective: catching it in minutes beats waiting "
             "out the cell timeout, because the longer 64 ranks sit wedged the likelier a "
             "host reboot is needed to clear them. 0 disables.",
    )
    p.add_argument(
        "--skip-warmup-pass", action="store_true",
        help="run only the measured pass, relying on `--num-warmups` to warm "
             "the cell. **Measured as a regression, so not the default.** "
             "`--num-warmups` sends N copies of the *first* dataset request, "
             "so it warms one prompt length; the discarded pass sends N "
             "distinct sonnet prompts, whose lengths vary around the target "
             "and key different prefill shapes. At in512_bs1 on 2026-10-21, "
             "skipping it moved TTFT 826 -> 1825 ms (+121%) and decode "
             "9.59 -> 8.01 tok/s, with **zero** NEFF misses either way -- so "
             "the cost it leaves in the measured window is dynamo re-tracing, "
             "not compilation. TPOT moved only +8%, which is the tell: prefill "
             "shapes are length-dependent and decode shapes are not.",
    )
    p.add_argument(
        "--temperature", default="0",
        help="sampling temperature sent with every request; '0' is greedy and "
             "is the default because acceptance is a first-order term in every "
             "throughput figure here. Pass 'none' to send no temperature and "
             "let the server choose, which is what upstream does by default.",
    )
    p.add_argument(
        "--prose-output-len", type=int, default=DEFAULT_PROSE_OUTPUT_LEN,
        help="tokens generated per prompt in the prose acceptance probe. Acceptance "
             "is sampled over decode steps, so this is the sample size.",
    )
    p.add_argument(
        "--no-prose-probe", action="store_true",
        help="skip the prose acceptance probe. The report then carries no "
             "acceptance figure, which is intended: an unmeasured bar is not a met one.",
    )
    p.add_argument("--report", default=str(DEFAULT_REPORT))
    p.add_argument(
        "--keep-server",
        action="store_true",
        help="leave the last server running, for poking at it by hand",
    )
    return p.parse_args(argv)


def _check_arch(model: str) -> None:
    """Refuse to benchmark anything but this architecture.

    Cheap, and it closes the one way a benchmark can be badly wrong without
    looking wrong: measuring a different model. `check_config.py` is the
    verifier's equivalent and goes much further; this only establishes that the
    thing being timed is the thing being onboarded.
    """
    config_path = Path(model) / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"bench: no config.json at {config_path}")
    arches = json.loads(config_path.read_text()).get("architectures") or []
    if ARCH not in arches:
        raise SystemExit(
            f"bench: {model} declares architectures={arches!r}, which does not "
            f"contain {ARCH}. This benchmark is for the DeepSeek-V4.1-Flash "
            "port; pass the checkpoint the verifier serves."
        )
    print(f"bench: serving {model} ({ARCH})")


def _free_port(port: int) -> int:
    """`port`, or the next free one, so a stale server does not silently get
    benchmarked instead of the one this run started."""
    for candidate in range(port, port + 64):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", candidate)) != 0:
                return candidate
    raise SystemExit(f"bench: no free port in [{port}, {port + 64})")


#: The pipeline degree `main` resolved, read by `_serve` (one process, one engine).
_PARALLEL: dict = {"pp": 1}


def _serve(
    model: str, tp: int, max_model_len: int, max_num_seqs: int, port: int,
    log: Path,
) -> subprocess.Popen:
    """Start `vllm serve` and return the process. Caller waits for health."""
    cmd = [
        # `_serve_main`, not `vllm.entrypoints.openai.api_server` directly:
        # the server resolves the checkpoint through `transformers.AutoConfig`
        # before it loads anything from `vllm_neuron`, and this architecture is
        # only registered when the port is imported. See `_serve_main`.
        sys.executable, "-m",
        "performance._serve_main",
        "--model", model,
        "--served-model-name", SERVED_NAME,
        "--tensor-parallel-size", str(tp),
        "--pipeline-parallel-size", str(_PARALLEL["pp"]),
        # Synchronous scheduling for pipeline stages (see run_verifier._run_phase).
        *(["--no-async-scheduling"] if _PARALLEL["pp"] > 1 else []),
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--port", str(port),
        # Off for the benchmark, deliberately: the random dataset shares no
        # prefix, so the cache can only cost bookkeeping — and leaving it on
        # would make a repeated cell measure the cache instead of the model.
        "--no-enable-prefix-caching",
    ]
    print(f"bench: starting server on port {port} "
          f"(max_model_len={max_model_len}, max_num_seqs={max_num_seqs})")
    print(f"bench:   {' '.join(cmd)}")
    handle = log.open("w")
    def _child_setup() -> None:
        # Two things, and they pull in opposite directions.
        #
        # `setsid` gives the server its own process group so `_stop` can take
        # the 64 workers down together instead of orphaning them. But it also
        # puts the server OUTSIDE the bench's group -- so when the harness
        # times a constraint out and kills the bench's process group, the
        # server survives holding all 64 cores. That is the orphan that has
        # forced a host reboot here more than once.
        #
        # `PR_SET_PDEATHSIG` closes it: the kernel SIGKILLs this process the
        # moment its parent dies, however the parent died. So the server is
        # independently killable AND cannot outlive the bench.
        import ctypes

        os.setsid()
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0
        )

    return subprocess.Popen(
        cmd, stdout=handle, stderr=subprocess.STDOUT,
        preexec_fn=_child_setup,  # noqa: PLW1509 - deliberate, see above
    )


def _wait_healthy(proc: subprocess.Popen, port: int, timeout: int) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    last = 0.0
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(
                f"bench: server exited with {proc.returncode} before becoming "
                "healthy; see its log"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print(f"bench: server healthy after "
                          f"{timeout - (deadline - time.time()):.0f} s")
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        if time.time() - last > 120:
            last = time.time()
            print(f"bench:   still waiting for the server "
                  f"({deadline - time.time():.0f} s left)")
        time.sleep(5)
    raise SystemExit(f"bench: server was not healthy within {timeout} s")


#: Processes that may be holding a Neuron device on this host, by cmdline. Only
#: these are force-killed when a teardown leaves devices busy, so a concurrent
#: verifier run is never caught by it.
_ENGINE_MARKERS = ("_serve_main", "EngineCore", "Worker_TP", "multiproc_executor")

#: How long to wait for 64 ranks to drop their device handles after the engine
#: is signalled. Measured: they take 30-40 s, so this is generous but finite.
_DEVICE_RELEASE_WAIT = 120


def _device_holders() -> list[int]:
    """PIDs holding a `/dev/neuron*` handle that look like an engine.

    Read from `/proc`, not from `lsof`: the bench runs unprivileged and these
    are its own descendants, so their fds are readable without sudo.
    """
    held: list[int] = []
    me = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == me:
            continue
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(
                "utf-8", "replace"
            ).replace("\x00", " ")
            comm = Path(f"/proc/{pid}/comm").read_text(errors="replace")
        except OSError:
            continue
        if not any(mark in cmd or mark in comm for mark in _ENGINE_MARKERS):
            continue
        fd_dir = Path(f"/proc/{pid}/fd")
        try:
            for fd in fd_dir.iterdir():
                try:
                    if "/dev/neuron" in os.readlink(fd):
                        held.append(pid)
                        break
                except OSError:
                    continue
        except OSError:
            continue
    return held


def _stop(proc: subprocess.Popen) -> None:
    """Stop an engine **and wait for its devices to come back.**

    Killing the parent is not enough and its absence caused a cascade: vLLM's
    `multiproc_executor` workers are not reliably in the parent's process
    group, so `EngineCore` and all 64 `Worker_TP*` survived a `_stop` and kept
    `/dev/neuron*` open. The next engine then failed `nrt_init` on all 64 ranks
    (`NRT_FAILURE`, "The PyTorch Neuron Runtime could not be initialized") and
    the cell reported `server exited with 1 before becoming healthy` --
    measured 2026-10-21, and the same orphaning has been cleaned up by hand
    repeatedly through this project. One leaked engine poisons every cell after
    it, so teardown waits for the devices rather than for the parent.
    """
    if proc.poll() is None:
        print("bench: stopping server")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=180)
        except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=60)
            except Exception:  # noqa: BLE001 - best effort teardown
                pass

    deadline = time.monotonic() + _DEVICE_RELEASE_WAIT
    while time.monotonic() < deadline:
        holders = _device_holders()
        if not holders:
            return
        time.sleep(5)
    holders = _device_holders()
    if not holders:
        return
    print(f"bench:   {len(holders)} process(es) still hold a Neuron device "
          f"{_DEVICE_RELEASE_WAIT}s after teardown; killing them so the next "
          f"cell can start", file=sys.stderr)
    for pid in holders:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    for _ in range(12):
        time.sleep(5)
        if not _device_holders():
            return
    print("bench:   devices are still held; the next cell will probably fail "
          "to initialise the runtime", file=sys.stderr)


#: **Retired 2026-10-21.** The warm-up now generates the measurement's own
#: output length; this constant is kept only so the reasoning that replaced it
#: is on the record.
#:
#: It used to be 8, on the argument that the warm-up exists to compile and that
#: "the decode shape this model keys is `(1 + k) * running_seqs` rows — a
#: function of the batch, not of how many tokens have been emitted", so a
#: longer generation reaches no shape a short one does not.
#:
#: **That argument is false, and two independent measurements say so.**
#: Iteration 10 found a cell with a fully warm NEFF cache still keying 48 new
#: graph signatures, appearing *late in the drain* at `Running: 4 reqs` — states
#: an 8-token run never reaches. And on 2026-10-21 an 18-minute full-length
#: warm-up at 10 prompts left 104 further graphs to be keyed inside a 20-prompt
#: measured pass. Emitted tokens do reach shapes, through the drain.
WARMUP_OUTPUT_LEN = 8


def _sonnet_flags(input_len: int, dataset_path: str) -> callable:
    """`vllm bench serve` flags for a cell of real English at a target length.

    Returns a callable taking the output length, because the warm-up and the
    measurement of one cell differ only in that.

    **Why sonnet rather than random.** `--dataset-name random` feeds random
    token *ids*, and that is not a neutral choice for this model: a drafter
    predicts the next token, there is nothing to predict in noise, and DSpark's
    n-gram/Markov head leans on real language statistics hardest of all.
    Measured on this port, acceptance is 11-15% on random against 41% on prose
    -- and since a decode step emits `1 + accepted` tokens, every throughput
    figure measured on random is a lower bound rather than the real number.

    `sonnet` is the only dataset in `vllm bench` that gives real text *and* a
    target input length; every other prose option (sharegpt, spec_bench, hf)
    takes whatever length the source happens to be.

    Two things to know about it:

      - **the length is approximate.** It picks whole verse lines until it
        reaches the target (`num_input_lines = round((input_len - base_offset) /
        avg_len)`), so a "1024" cell lands near 1024, not on it. That is why the
        report records `mean_input_tokens` from the run's own
        `total_input_tokens`: TTFT has to be normalised by the measured
        length rather than the nominal one.
      - **`--sonnet-prefix-len 0`, deliberately.** The default is 200, which
        gives every request in a cell the same 200-token opening -- and a
        drafter seeing the same text repeatedly would post an acceptance rate
        that says more about the prefix than the model.
    """
    def build(output_len: int) -> list[str]:
        return [
            "--dataset-name", "sonnet",
            "--dataset-path", dataset_path,
            "--sonnet-input-len", str(input_len),
            "--sonnet-output-len", str(output_len),
            "--sonnet-prefix-len", "0",
        ]
    return build


#: vLLM's own spec-decode counters, as the Prometheus endpoint spells them.
_SPEC_METRICS = {
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    "num_drafts": "vllm:spec_decode_num_drafts_total",
}


def _scrape_spec_counters(port: int) -> dict[str, float]:
    """vLLM's spec-decode counters from a live server's `/metrics`.

    Summed over label sets: one engine has one, but a counter that grew under
    two must not be read as whichever was exposed last.
    """
    out = {k: 0.0 for k in _SPEC_METRICS}
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/metrics", timeout=30
        ) as resp:
            text = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"bench:   /metrics unreachable: {exc}", file=sys.stderr)
        return out
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        bare = name.split("{", 1)[0].strip()
        for key, metric in _SPEC_METRICS.items():
            if bare == metric:
                try:
                    out[key] += float(value)
                except ValueError:
                    pass
    return out


#: ShareGPT conversations, for the acceptance probe. Real dialogue rather than
#: Shakespeare: the matrix is sonnet because only sonnet can hit a target input
#: length, but acceptance has no length constraint, so it can use the most
#: representative text available.
DEFAULT_SHAREGPT_PATH = "/mnt/nvme/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"

#: Conversations the probe sends. Enough decode steps for the rate to mean
#: something, few enough that the probe stays a tail on an existing cell.
PROSE_PROMPTS = 6


def _prose_prompts(model: str, dataset: str) -> list[tuple[str, str, int]]:
    """Prompts for the acceptance probe, as `(id, text, n_tokens)`.

    `sharegpt` takes the first human turn of each conversation -- real dialogue,
    which is what the drafter will actually see in service. Falls back to the
    verifier's own cases when the ShareGPT file is absent, so the probe degrades
    to something rather than nothing; which was used is recorded in the report.

    The prompts are sent raw, with no chat template applied, because they go to
    `/v1/completions`. Templating here and again at the server would change the
    tokens the drafter sees.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    if dataset == "sharegpt" and Path(DEFAULT_SHAREGPT_PATH).is_file():
        raw = json.loads(Path(DEFAULT_SHAREGPT_PATH).read_text())
        out: list[tuple[str, str, int]] = []
        for conv in raw:
            turns = conv.get("conversations") or []
            if not turns or turns[0].get("from") != "human":
                continue
            text = (turns[0].get("value") or "").strip()
            n = len(tok(text).input_ids)
            # Long enough that the drafter has context, short enough to fit any
            # cell's engine.
            if not (64 <= n <= 1024):
                continue
            out.append((f"sharegpt-{conv.get('id', len(out))}", text, n))
            if len(out) >= PROSE_PROMPTS:
                return out
        if out:
            return out
        print("bench:   no usable ShareGPT conversation found; falling back to the "
              "verifier's cases", file=sys.stderr)

    from deepseek_v41.verifier import prompts as vp
    from deepseek_v41.verifier.run_verifier import CHAT_TEMPLATE

    case_set = vp.build_cases()
    rendered, notes = vp.render(case_set, model, CHAT_TEMPLATE)
    for note in notes:
        print(f"bench:   {note}")
    return [(c.id, text, n) for c, (text, n) in zip(case_set.cases, rendered)]


def _prose_probe(model: str, port: int, output_len: int, timeout: int,
                 dataset: str = "sharegpt") -> dict:
    """Measure speculative acceptance on real prose, on a server already up.

    **Why this is a probe and not a cell.** Acceptance is the one figure the
    random cells cannot produce: it is a statement about how predictable the
    text is, and `--dataset-name random` feeds random token *ids*, which an
    n-gram/Markov drafter cannot predict even in principle. This port measures
    49% on prose against 16-18% on random.

    So the prompts have to be real text -- and they are the verifier's own
    cases, imported rather than copied, because "the verifier and the bench
    serve the same model" is a requirement of this port and the same prompts is
    that requirement one level down. `prompts.render` has already applied the
    chat template, which is why these go to `/v1/completions` rather than the
    chat endpoint: templating them twice would change the tokens.

    It runs on an engine that is **already serving**, so it costs no extra
    engine build -- which is the whole point. A cell of its own would have been
    a fifth context to compile. `vllm bench serve --dataset-name custom` would
    have been less code and is not available: `CustomDataset` reads its JSONL
    through pandas, and pandas is not installed in this environment, where it
    resolves to a `PlaceholderModule` that raises on use.

    The counters are read **before and after**, so the figure is this probe's
    and not contaminated by the random traffic this server has already served.

    Only prompts that fit the server's context are sent, and which ones is
    recorded: a 2008-token needle has nothing to do with a 768-token engine.
    """
    pairs = _prose_prompts(model, dataset)
    before = _scrape_spec_counters(port)
    sent: list[str] = []
    generated = 0
    for case_id, text, n_tok in pairs:
        payload = json.dumps({
            "model": SERVED_NAME,
            "prompt": text,
            "max_tokens": output_len,
            "temperature": 0.0,
            # Every request emits exactly `output_len` tokens, so the number of
            # decode steps the rate is sampled over is known.
            "ignore_eos": True,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions", data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode())
        except (urllib.error.HTTPError, urllib.error.URLError, OSError,
                TimeoutError) as exc:
            # A prompt longer than the engine's context is the expected
            # rejection here, not a bug: say which and carry on.
            print(f"bench:   prose prompt {case_id} ({n_tok} tok) rejected: {exc}",
                  file=sys.stderr)
            continue
        sent.append(case_id)
        generated += (body.get("usage") or {}).get("completion_tokens") or 0

    after = _scrape_spec_counters(port)
    drafted = int(after["draft_tokens"] - before["draft_tokens"])
    accepted = int(after["accepted_tokens"] - before["accepted_tokens"])
    drafts = int(after["num_drafts"] - before["num_drafts"])
    return {
        "cases": sent,
        "dataset": dataset,
        "output_len": output_len,
        "tokens_generated": generated,
        "source": "vllm:spec_decode_* /metrics delta around the prose requests",
        "num_drafts": drafts,
        "draft_tokens": drafted,
        "accepted_tokens": accepted,
        "acceptance_pct": (100.0 * accepted / drafted) if drafted else None,
    }


def _bench_serve(
    model: str, port: int, dataset_flags: list[str], batch: int,
    num_prompts: int, result_dir: Path, result_file: str, timeout: int,
    server: subprocess.Popen | None = None,
    server_log: Path | None = None, stall_timeout: int = 0,
    num_warmups: int = 0, temperature: float | None = 0.0,
) -> dict | None:
    """One `vllm bench serve` invocation. Returns its parsed result, or None.

    `server` is the engine this is driving, watched while the client runs.
    **`vllm bench serve` has no per-request timeout**, so if the engine goes
    away mid-measurement the client waits for replies that will never come
    until `timeout` expires. Measured 2026-10-18: an engine was sent SIGTERM
    while healthy -- 1 running, 9 waiting, generation throughput rising
    2.7 -> 4.3 tokens/s, a forward every ~375 ms -- and the client then sat for
    the full 7200 s. Two hours spent learning nothing, and the figure it looked
    like from outside was a model stall, which it was not.

    So the server is polled alongside the client. If it exits first, the client
    is killed and this returns None with the engine's exit code named, which is
    both faster and a different diagnosis from a timeout.
    """
    # The `vllm` CLI is what the onboarding guide names. There is deliberately
    # **no module fallback**: `python -m vllm.benchmarks.serve` has no
    # `__main__` block, so it imports its dependencies, exits 0, and writes no
    # result file. Measured 2026-10-21: six client arms "completed" in 6 s each
    # with rc=0 and no JSON. A fallback that silently measures nothing is worse
    # than no fallback, so this raises instead.
    vllm_bin = shutil.which("vllm")
    if not vllm_bin:
        raise RuntimeError(
            "`vllm` is not on PATH. `python -m vllm.benchmarks.serve` is not a "
            "usable substitute: the module has no __main__ block and exits 0 "
            "without benchmarking anything."
        )
    cmd = [vllm_bin, "bench", "serve"]
    cmd += [
        # `openai`, not `openai-chat`: the completions endpoint takes the random
        # dataset's token ids as given, where the chat endpoint would wrap them
        # in a template and change the input length the cell is named for.
        "--backend", "openai",
        "--host", "127.0.0.1", "--port", str(port),
        "--model", SERVED_NAME,
        "--tokenizer", model,
        *dataset_flags,
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(batch),
        # Every request emits exactly `output_len` tokens, so TPOT and the
        # token count are not at the mercy of an early stop.
        "--ignore-eos",
        # Greedy. Without this the client sends no temperature and the server
        # samples at its own default, which upstream warns about on every run
        # ("no longer sets temperature==0 (greedy) in requests by default").
        # It matters here because acceptance is a first-order term in every
        # throughput figure: measured 2026-10-21 at in512_bs10, greedy raised
        # acceptance 37.6% -> 43.7% and acceptance length 2.88 -> 3.19, worth
        # ~10% of aggregate decode. Declared as this deployment's sampling
        # policy rather than discovered per run.
        *(("--temperature", str(temperature)) if temperature is not None else ()),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,90,99",
        "--save-result",
        # Per-request ttfts/tpots/itls land in the result JSON. The aggregate
        # alone cannot distinguish "the engine is slow" from "one request
        # stalled for 60 s", and that distinction has cost this project
        # several wrong conclusions.
        "--save-detailed",
        "--result-dir", str(result_dir),
        "--result-filename", result_file,
    ]
    # Upstream's own warm-up: `num_warmups` copies of the first dataset request
    # at its real output length, under the same `--max-concurrency` semaphore,
    # awaited before the timer starts (`vllm/benchmarks/serve.py:870`). This is
    # what the two-pass structure below could not do, because the passes are
    # separate client processes and the cost being absorbed is **dynamo
    # re-tracing**, not NEFF compilation. Measured 2026-10-21 at in512_bs10,
    # both arms with zero NEFF cache misses: without it 10.74 tok/s, with it
    # 24.82 tok/s.
    if num_warmups > 0:
        cmd += ["--num-warmups", str(num_warmups)]
    started = time.monotonic()
    client = subprocess.Popen(cmd)
    log_path = server_log if server_log is not None else None
    last_size, last_grew = (log_path.stat().st_size if log_path and log_path.exists()
                            else 0), time.monotonic()
    while True:
        try:
            client.wait(timeout=5.0)
            break
        except subprocess.TimeoutExpired:
            pass

        # A rank that dies says so. Catching the signature is seconds rather
        # than the 25 minutes the silence watchdog below would take, and the
        # devices come back that much sooner.
        if log_path is not None:
            fatal = _fatal_in_log(log_path)
            if fatal:
                print(f"bench:   FATAL in the engine log: {fatal!r} — killing the cell "
                      "now rather than waiting out the timeout.", file=sys.stderr)
                if "dma ring" in fatal or "dma rings" in fatal:
                    print("bench:   That is device memory for DMA descriptor rings, not "
                          "HBM for weights. Each loaded NEFF caches "
                          "`NEURON_RT_IO_RING_CACHE_SIZE` rings (the worker default is "
                          "32) and this model keys ~120 graphs, so the ring pool scales "
                          "with graph count. Lowering that variable is the first thing "
                          "to try.", file=sys.stderr)
                print("bench:   The surviving ranks are wedged in a collective. "
                      "`~/reset_neuron.sh` reloads the driver, which clears it without "
                      "a host reboot.", file=sys.stderr)
                client.kill(); client.wait()
                if server is not None:
                    _stop(server)
                return None

        # A wedged engine that says nothing at all. The backstop for failures
        # with no log signature.
        if log_path is not None and stall_timeout > 0:
            size = log_path.stat().st_size if log_path.exists() else 0
            if size != last_size:
                last_size, last_grew = size, time.monotonic()
            elif time.monotonic() - last_grew > stall_timeout:
                print(f"bench:   the engine has written nothing for "
                      f"{stall_timeout} s — killing it rather than waiting out the "
                      f"{timeout} s cell timeout. A wedged 64-rank collective gets "
                      "harder to clear the longer it sits.", file=sys.stderr)
                client.kill(); client.wait()
                if server is not None:
                    _stop(server)
                return None

        if server is not None and server.poll() is not None:
            print(f"bench:   the engine exited {server.returncode} while the client was "
                  f"still running — killing the client rather than waiting out the "
                  f"{timeout} s timeout. This is a lost server, not a slow model; see its "
                  "log for why it went away.", file=sys.stderr)
            client.kill()
            client.wait()
            return None
        if time.monotonic() - started > timeout:
            print(f"bench:   timed out after {timeout} s", file=sys.stderr)
            client.kill()
            client.wait()
            return None
    if client.returncode:
        print(f"bench:   exit {client.returncode}", file=sys.stderr)
        return None

    path = result_dir / result_file
    if not path.is_file():
        print(f"bench:   no result written at {path}", file=sys.stderr)
        return None
    return json.loads(path.read_text())


def _parse_temperature(raw) -> float | None:
    """`--temperature` as a float, or None for "send none and let the server
    choose" (upstream's default, and what every run before 2026-10-21 did)."""
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in ("none", "", "server"):
        return None
    return float(raw)


def _bench_cell(
    model: str, port: int, name: str, flags: callable, output_len: int, batch: int,
    num_prompts: int, result_dir: Path, timeout: int,
    server: subprocess.Popen | None = None,
    server_log: Path | None = None, stall_timeout: int = 0,
    temperature: float | None = 0.0, skip_warmup_pass: bool = False,
) -> dict | None:
    """One cell: warm it up, then measure it.

    **The warm-up is not optional and its absence is not a small error.** A
    cell's first requests pay for that cell's own compiled buckets — the prefill
    shape for its input length, the decode shape for its batch — and on this
    model a graph can take minutes to build. At batch 10 the whole first round
    of ten requests pays it concurrently, which is half the sample at the
    default prompt count, so an unwarmed cell does not report a slow median: it
    reports a compile.

    So each cell runs twice, and the discarded pass generates the **same output
    length** as the measurement. It used to generate 8 tokens to stay cheap;
    see `WARMUP_OUTPUT_LEN` for the measurements that retired that.

    **It sends the same number of requests as the measurement, not fewer.** That
    looks like waste and is not: with `--max-concurrency b` and `2b` requests,
    the scheduler admits a new request each time one finishes, so a *prefill
    runs concurrently with `b - 1` decodes* — a shape that never occurs if the
    warm-up submits exactly `b` requests which all start together and drain
    together. Matching the request count makes the warm-up traverse the same
    scheduling trajectory, so every shape the measurement will key already
    exists. An earlier cut sent `max(batch, 2)`, which left that mixed
    prefill/decode shape to compile inside the batch-10 measurements.

    A failed warm-up is a note, not a failure: the measurement that follows is
    then simply a cold one, and it is better to report a number with that
    caveat than to drop the cell.
    """
    warm = None
    if skip_warmup_pass:
        print(f"\nbench: cell {name} — warm-up pass skipped; "
              f"`--num-warmups {num_prompts}` carries the warming")
    else:
        print(f"\nbench: cell {name} — warm-up: {num_prompts} prompts, "
              f"{output_len} out (discarded)")
        warm = _bench_serve(
            model, port, flags(output_len), batch, num_prompts,
            result_dir, f"{name}.warmup.json",
            # Compile is paid here, so this pass gets the allowance; see
            # `CELL_WARMUP_COMPILE_ALLOWANCE` for what it is sized from.
            timeout + CELL_WARMUP_COMPILE_ALLOWANCE, server=server,
            server_log=server_log, stall_timeout=stall_timeout,
            temperature=temperature,
        )
    if warm is None and not skip_warmup_pass:
        print(f"bench:   warm-up for {name} did not complete; the measurement "
              "below is a cold one", file=sys.stderr)

    print(f"bench: cell {name} — measuring: {num_prompts} prompts, "
          f"concurrency {batch}, {output_len} out")
    # Where this cell's measured window starts in the server log. With one
    # engine serving several cells the acceptance counters are no longer "this
    # cell's and nothing else's", so they are read from here rather than from
    # the top of the file. It also excludes the warm-up, which is correct: the
    # reported rate should be the measured pass's.
    log_offset = 0
    if server_log is not None:
        try:
            log_offset = server_log.stat().st_size
        except OSError:
            log_offset = 0
    measured = _bench_serve(
        model, port, flags(output_len), batch, num_prompts,
        result_dir, f"{name}.json", timeout, server=server,
        server_log=server_log, stall_timeout=stall_timeout,
        # A full wave re-traced in-process immediately before the timer starts,
        # on top of the discarded pass above. Belt and braces: the pass above
        # warms the engine's graphs, this one warms whatever the client's own
        # process re-traces per invocation.
        num_warmups=num_prompts,
        temperature=temperature,
    )
    if measured is not None:
        # Recorded on the result, so a reader of the report can tell a warmed
        # measurement from a cold one without going back to the console.
        # With the discarded pass skipped there is no separate warm-up to have
        # completed, and `--num-warmups` is awaited by the client before the
        # timer starts -- so a measurement that exists at all was warmed.
        measured["_warmed"] = True if skip_warmup_pass else warm is not None
        measured["_log_offset"] = log_offset
    return measured


#: The port's own clamp line, e.g. `max_num_seqs 10 -> 1`.
_CLAMP_RE = re.compile(r"max_num_seqs (\d+) -> (\d+)")


def _engine_batch(server_log: Path, requested: int) -> int:
    """The batch the engine actually serves, which may be below what was asked.

    **This guard exists because its absence produced a meaningless number.**
    `__init__.MULTI_RANK_MAX_NUM_SEQS` clamps concurrent sequences above one
    rank, and it is currently 1 because multi-sequence decode at 64 ranks hangs.
    A server asked for 10 therefore serves **one at a time** — while
    `vllm bench serve --max-concurrency 10` keeps ten requests in flight. The
    nine that cannot run sit in the engine's queue, and their queue wait lands
    in their TTFT:

        in512_bs10: ttft p50 234,981 ms, p99 260,168 ms (ratio 1.1x),
                    output_throughput 4.71 tok/s against batch 1's 4.26

    A p99/p50 ratio of 1.1 is the tell — every request waited, so it is not a
    slow outlier but queueing. And the throughput is batch 1's, because the
    engine *is* batch 1. Reported without this check, that row reads as "batch
    10 has terrible latency" when what it measures is a queue.

    Returns the clamped value, or `requested` when no clamp was logged.
    """
    try:
        text = server_log.read_text(errors="replace")
    except OSError:
        return requested
    matches = _CLAMP_RE.findall(text)
    return int(matches[-1][1]) if matches else requested


#: `DSparkProposer`'s running acceptance line, as it appears in a server log.
_ACCEPT_RE = re.compile(
    r"DSpark verify step \d+ — .*?(\d+)/(\d+) \((\d+)%\) over the run"
)


def _acceptance(server_log: Path, offset: int = 0) -> float | None:
    """This cell's speculative acceptance rate, from its own server's log.

    **Why the benchmark reports this at all.** Speculation decides how many
    tokens a decode step emits — `1 + accepted` — so it is a first-order term in
    every throughput figure here, and on `--dataset-name random` it very nearly
    switches itself off. Random input is random *token ids*, and a drafter
    predicts the next token; there is nothing to predict in noise, and DSpark's
    Markov/n-gram head leans on real language statistics hardest of all.
    Measured on this port: **49%** acceptance on the verifier's prose against
    **16-18%** on the random dataset.

    That is not a fault in the model, and it is not a reason to stop using
    random input — exact lengths are worth more to a benchmark than realistic
    token statistics. It does mean the throughput in this table is a **lower
    bound** on what real traffic would see, and a reader who does not know the
    acceptance rate cannot tell that. So it is a column.

    The counters are cumulative over a process, and with one engine per cell
    this log belongs to exactly this cell, so its last line is this cell's rate.
    Returns None rather than 0.0 when nothing matched: "speculation was not
    measured" and "speculation accepted nothing" are different claims.
    """
    try:
        with server_log.open(errors="replace") as fh:
            fh.seek(offset)
            text = fh.read()
    except OSError:
        return None
    matches = _ACCEPT_RE.findall(text)
    if not matches:
        return None
    kept, offered, _pct = matches[-1]
    offered_n = int(offered)
    return 100.0 * int(kept) / offered_n if offered_n else None


#: vLLM's own per-interval spec-decode line, logged by `SpecDecodingLogging.log`
#: whenever stat logging is on — which it is for a served engine. The counts are
#: per logging interval, not cumulative, so every line is summed.
_VLLM_SPEC_RE = re.compile(
    r"SpecDecoding metrics:.*?Accepted: (\d+) tokens, Drafted: (\d+) tokens"
)


def _vllm_acceptance(server_log: Path, offset: int = 0) -> dict:
    """This cell's acceptance rate as *vLLM* counted it, not as the port did.

    `_acceptance` above reads `drafter.py`'s own running line, which is the
    useful one to watch and the wrong one to trust: `drafter.py` is inside
    `scope.editable`, so the loop can change the code that computes the number
    it is judged on. vLLM's scheduler accumulates the same quantity in
    `SpecDecodingStats.observe_draft` from the model runner's output, in code
    this port cannot edit, and logs it as `Accepted: N tokens, Drafted: M`.

    Recorded here beside the port's own figure so the two can be compared.
    Measured on prose, because random token ids are not predictable by a
    drafter at any quality.

    Returns zero counts rather than None when the pattern never matched, with
    `found: False`, so "not measured" stays distinguishable from "accepted
    nothing".
    """
    try:
        with server_log.open(errors="replace") as fh:
            fh.seek(offset)
            text = fh.read()
    except OSError:
        return {"found": False, "accepted": 0, "drafted": 0, "pct": None}
    pairs = _VLLM_SPEC_RE.findall(text)
    accepted = sum(int(a) for a, _ in pairs)
    drafted = sum(int(d) for _, d in pairs)
    return {
        "found": bool(pairs),
        "accepted": accepted,
        "drafted": drafted,
        "pct": (100.0 * accepted / drafted) if drafted else None,
    }


#: Figures emitted per cell, in `_emit_metrics`.
_EMITTED = ("ttft_p50_ms", "tpot_p50_ms", "decode_tok_s", "vllm_acceptance_pct")


def _emit_metrics(cell: dict) -> None:
    """One `##autohelix[...]` line per figure, named `<metric>_<cell>`.

    Called as soon as a cell has figures, not at the end of the matrix. The
    difference matters: `bench_report.json` is rewritten after every cell, so a
    run that dies on cell 7 keeps six cells' measurements -- but the emission
    used to happen only after the last cell, so the same run reported *nothing*
    to a loop reading these lines. Emitting per cell makes the two records agree.

    Only figures that exist are emitted: a missing metric must not read as a
    zero.
    """
    m = cell.get("metrics")
    if not m:
        return
    for key in _EMITTED:
        value = m.get(key)
        if isinstance(value, (int, float)):
            print(f"##autohelix[{key}_{cell['name']}={value:.6g}]", flush=True)


def _metrics(raw: dict) -> dict:
    """The three figures this benchmark reports, out of `vllm bench serve`.

    `vllm bench serve` writes many keys; these are the ones the matrix is
    specified against. `output_throughput` is already aggregate output tokens
    per second over the run's wall clock, which is the "decode speed" a serving
    deployment is measured by; the per-stream rate is derived from TPOT so both
    views are on the record.
    """
    tpot = raw.get("median_tpot_ms")
    return {
        "ttft_p50_ms": raw.get("median_ttft_ms"),
        "ttft_p99_ms": raw.get("p99_ttft_ms"),
        "tpot_p50_ms": tpot,
        "tpot_p99_ms": raw.get("p99_tpot_ms"),
        "decode_tok_s": raw.get("output_throughput"),
        "per_stream_tok_s": (1000.0 / tpot) if tpot else None,
        "total_tok_s": raw.get("total_token_throughput"),
        "completed": raw.get("completed"),
        "duration_s": raw.get("duration"),
        # Sonnet rounds to whole verse lines, so the nominal input length and
        # the real one differ. Everything that normalises by input length has to
        # use this, not the cell's name.
        "mean_input_tokens": (
            (raw.get("total_input_tokens") or 0) / raw["completed"]
            if raw.get("completed") else None
        ),
        # False means the cell's warm-up did not complete, so these figures
        # include that cell's compile cost. See `_bench_cell`.
        "warmed": raw.get("_warmed", False),
    }


def _print_table(cells: list[dict]) -> None:
    print("\n" + "=" * 86)
    print("DeepSeek-V4.1-Flash — forward inference, `vllm bench serve`, random input")
    print("=" * 94)
    header = (f"{'cell':>13} {'out':>5} {'ctx':>7} "
              f"{'TTFT p50':>10} {'TPOT p50':>10} {'decode':>10} "
              f"{'per-str':>9} {'accept':>7} {'vllm':>6} {'done':>5}")
    print(header)
    print(f"{'':>13} {'':>5} {'':>7} {'ms':>10} {'ms':>10} "
          f"{'tok/s':>10} {'tok/s':>9} {'%':>7} {'%':>6} {'':>5}")
    print("-" * 94)
    for c in cells:
        m = c["metrics"]
        if m is None:
            print(f"{c['name']:>13} {c['output_len']:>5} "
                  f"{c.get('max_model_len', '-'):>7} {'FAILED':>10}")
            continue

        def fmt(v, spec=".1f"):
            return format(v, spec) if isinstance(v, (int, float)) else "-"

        print(f"{c['name']:>13} {c['output_len']:>5} "
              f"{c.get('max_model_len', '-'):>7} "
              f"{fmt(m['ttft_p50_ms']):>10} {fmt(m['tpot_p50_ms']):>10} "
              f"{fmt(m['decode_tok_s'], '.2f'):>10} "
              f"{fmt(m['per_stream_tok_s'], '.2f'):>9} "
              f"{fmt(m.get('draft_acceptance_pct'), '.0f'):>7} "
              f"{fmt(m.get('vllm_acceptance_pct'), '.0f'):>6} "
              f"{m['completed'] if m['completed'] is not None else '-':>5}"
              + ("" if m.get("warmed") else "  COLD")
              + ("" if not c.get("concurrency_capped")
                 else f"  CAPPED->{c.get('engine_max_num_seqs')}"))
    print("=" * 94)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _check_arch(args.model)

    tp = args.tensor_parallel_size
    source = "--tensor-parallel-size"
    if tp is None:
        from deepseek_v41.config import TENSOR_PARALLEL_SIZE

        tp, source = TENSOR_PARALLEL_SIZE, "config.TENSOR_PARALLEL_SIZE"
    print(f"bench: tensor_parallel_size={tp} (from {source})")
    pp = args.pipeline_parallel_size
    if pp is None:
        from deepseek_v41.config import PIPELINE_PARALLEL_SIZE

        pp = PIPELINE_PARALLEL_SIZE
    print(f"bench: pipeline_parallel_size={pp}")
    _PARALLEL["pp"] = pp

    input_lens = [int(x) for x in args.input_lens.split(",") if x]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x]
    max_model_len = args.max_model_len or (
        max(input_lens) + args.output_len + CONTEXT_HEADROOM
    )
    shortest_needed = max(input_lens) + args.output_len
    if max_model_len < shortest_needed:
        raise SystemExit(
            f"bench: --max-model-len {max_model_len} cannot hold the largest "
            f"cell ({max(input_lens)} in + {args.output_len} out = "
            f"{shortest_needed}). vLLM would reject the prompt rather than "
            "measure it."
        )

    report_path = Path(args.report)
    result_dir = report_path.parent / "cells"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir = result_dir

    report: dict = {
        "model": args.model,
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
        "max_model_len": max_model_len,
        "output_len": args.output_len,
        "input_lens": input_lens,
        "batch_sizes": batch_sizes,
        "cells": [],
    }

    def _write() -> None:
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    # **One engine per cell.** The context is the thing that makes this
    # necessary, not the batch: `model._COMPRESS_READ_CEILING` pins the
    # compressed-KV read to `max_model_len`, so a 512-token request served by an
    # 8448-context engine reads 8448 compressed groups per layer on 38 of 40
    # layers instead of the 672 its own context needs — 13x the work, on every
    # forward, prefill and decode alike. Sharing one engine across input lengths
    # therefore does not measure those lengths; it measures the largest one.
    #
    # Measured: the 512-token cell reported TTFT 2318 ms at an 8448 context
    # against ~1250 ms for a short prompt at 608. So each cell gets a context
    # sized to itself, at the cost of an engine build per cell.
    #
    # `--max-model-len` overrides this and pins one context for every cell,
    # which is the right thing when the question is "what does serving an 8k
    # context cost a short request" rather than "how fast is a 512-token
    # request".
    #: context -> (proc, port, log) for engines left running between cells.
    _servers: dict[int, tuple] = {}

    def run_cell(name: str, flags: callable, context: int, batch: int,
                 num_prompts: int, output_len: int, extra: dict,
                 probe_prose: bool = False) -> None:
        """Serve one engine, warm it, measure it, and record the cell.

        When `probe_prose`, the prose acceptance probe runs on this cell's
        server before teardown -- see `_prose_probe` for why it rides along on
        an engine that is already up instead of building one of its own.
        """
        # One engine per *context*, shared by every batch size at that
        # context. At `MAX_NUM_SEQS = 1` the port clamps every engine to one
        # sequence whatever the server was started with, so the bs1 and bs10
        # cells of one input length are running byte-identical engines today
        # and differ only in client concurrency -- there was never a reason to
        # build two. Sharing also means the second cell inherits the first's
        # warm graphs, which matters now that the warm-up runs at full output
        # length. The cache key is the context because
        # `model._COMPRESS_READ_CEILING` pins the indexer's `read_len` to the
        # engine's context: one engine for *all* cells would make `in512` do
        # 8448-sized indexer reads and stop measuring what it is named for.
        #
        # **Revisit this if `MAX_NUM_SEQS` ever rises above 1.** The engine is
        # started at `max(batch_sizes)`, and above the clamp a shared engine
        # pins `num_seqs_buckets` to that maximum, so `_bucket_spans` would pad
        # a batch-1 decode step out to the full span count -- paying for ten
        # sequences to serve one. At that point the cache key has to become
        # `(context, batch)` again, which is what it effectively was before.
        # The key is `(context, the batch this engine will actually serve)`.
        # Below the clamp every cell maps to the same key -- `MAX_NUM_SEQS = 1`
        # makes a bs1 and a bs10 cell byte-identical engines -- so one engine
        # serves both. Above it they separate again, and they must: a shared
        # `max_num_seqs = 10` engine pins `num_seqs_buckets` to 10, and
        # `_bucket_spans` would then pad a one-sequence decode step out to ten
        # spans, paying for ten sequences to serve one.
        from deepseek_v41.config import MAX_NUM_SEQS as _CLAMP
        serve_batch = min(batch, _CLAMP) if _CLAMP else batch
        key = (context, serve_batch)
        cached = _servers.get(key)
        if cached is not None:
            proc, port, log = cached
            print(f"\nbench: cell {name} — reusing the context-{context} "
                  f"max_num_seqs-{serve_batch} engine on port {port}")
        else:
            # **At most one engine at a time.** Every engine here takes all 64
            # ranks, so a second one cannot start while the first lives: it
            # fails `nrt_init` on all 64 workers with NRT_FAILURE and the cell
            # reports "server exited with 1 before becoming healthy". Measured
            # 2026-10-21, when lifting the clamp first made two keys per
            # context possible. So anything still held is dropped here.
            for stale_key in list(_servers):
                stale = _servers.pop(stale_key)
                print(f"bench:   dropping the context-{stale_key[0]} "
                      f"max_num_seqs-{stale_key[1]} engine first "
                      f"(one engine owns all {tp} ranks)")
                _stop(stale[0])
            port = _free_port(args.port)
            log = log_dir / f"server_ctx{context}_ms{serve_batch}.log"
            proc = _serve(args.model, tp, context, serve_batch, port, log)
        raw = None
        served_batch = batch
        try:
            if cached is None:
                _wait_healthy(proc, port, args.server_timeout)
            served_batch = _engine_batch(log, batch)
            if served_batch < batch:
                print(
                    f"bench: WARNING cell {name} — the engine clamped "
                    f"max_num_seqs to {served_batch}, so it serves "
                    f"{served_batch} sequence(s) at a time while the client "
                    f"keeps {batch} in flight. TTFT will be mostly queue wait "
                    f"and the throughput will be {served_batch}-way, not "
                    f"{batch}-way. See config.MAX_NUM_SEQS.",
                    file=sys.stderr,
                )
            raw = _bench_cell(
                args.model, port, name, flags, output_len, batch, num_prompts,
                result_dir, _cell_timeout(args, batch, output_len), server=proc,
                server_log=log, stall_timeout=args.stall_timeout,
                # Greedy is withheld from multi-sequence cells. It is worth
                # ~10% of aggregate decode (acceptance 37.6% -> 43.7%) and is
                # safe at one sequence and on a clamped engine, but measured
                # 2026-10-21 it **hangs a cold uncapped multi-sequence
                # engine**: identical config, 10 concurrent requests, the
                # warm-up froze at 3 of 10 with every worker blocked in
                # `_engram_rows` (`state=S`, `wchan=do_wait_intr_irq`) and no
                # NEFF activity, while the same run with no temperature
                # completed and measured 25.24 tok/s. A 10% figure is not
                # worth a known hang in the default path.
                temperature=(
                    _parse_temperature(args.temperature)
                    if min(batch, _CLAMP or batch) <= 1 else None
                ),
                skip_warmup_pass=args.skip_warmup_pass,
            )
            if probe_prose:
                print(f"\nbench: prose acceptance probe on {name}'s server "
                      f"(context {context}, {args.prose_output_len} out)")
                report["prose"] = _prose_probe(
                    args.model, port, args.prose_output_len,
                    _cell_timeout(args, 1, args.prose_output_len),
                    dataset=args.prose_dataset,
                )
                report["prose"]["host_cell"] = name
                report["prose"]["max_model_len"] = context
                pr = report["prose"]
                pct = pr["acceptance_pct"]
                if isinstance(pct, (int, float)):
                    print(f"##autohelix[spec_acceptance_pct={pct:.6g}]", flush=True)
                print(f"bench:   {pr['accepted_tokens']}/{pr['draft_tokens']} drafts "
                      f"accepted = {(f'{pct:.1f}%' if pct is not None else 'unmeasured')} "
                      f"over case(s) {pr['cases']}")
                _write()
        finally:
            # Left running for the next cell at this context; torn down by the
            # caller when the context changes or the matrix ends.
            _servers[key] = (proc, port, log)
        metrics = _metrics(raw) if raw else None
        if metrics is not None:
            # Read after teardown: the counters are cumulative and this server
            # served exactly this cell, so they are this cell's and nothing
            # else's.
            off = int(raw.get("_log_offset") or 0)
            metrics["draft_acceptance_pct"] = _acceptance(log, off)
            # The same quantity from vLLM's own counters, which the port cannot
            # edit. See `_vllm_acceptance`.
            vllm_spec = _vllm_acceptance(log, off)
            metrics["vllm_acceptance_pct"] = vllm_spec["pct"]
            metrics["vllm_accepted_tokens"] = vllm_spec["accepted"]
            metrics["vllm_draft_tokens"] = vllm_spec["drafted"]
        report["cells"].append({
            "name": name,
            "output_len": output_len,
            "batch": batch,
            "max_model_len": context,
            "engine_max_num_seqs": served_batch,
            "concurrency_capped": served_batch < batch,
            "num_prompts": num_prompts,
            "metrics": metrics,
            "server_log": str(log),
            **extra,
        })
        _write()  # after every cell, so a crash keeps what ran
        _emit_metrics(report["cells"][-1])

    # The prose probe needs the widest context the matrix offers, and batch 1 so
    # it is not competing with 31 other sequences for the counters it reads.
    probe_cell = None
    if not args.no_prose_probe:
        widest = max(input_lens)
        probe_batch = min(batch_sizes)
        probe_cell = f"in{widest}_bs{probe_batch}"

    # Grouped by context, not by batch: every cell sharing an engine runs
    # consecutively so the engine can be built once and dropped as soon as its
    # last cell is done, instead of being held for the whole matrix.
    for input_len in input_lens:
        context = args.max_model_len or (
            input_len + args.output_len + CONTEXT_HEADROOM
        )
        for batch in batch_sizes:
            name = f"in{input_len}_bs{batch}"
            run_cell(
                name=name,
                flags=_sonnet_flags(input_len, args.dataset_path),
                context=context,
                batch=batch,
                num_prompts=args.num_prompts or _num_prompts(batch),
                output_len=args.output_len,
                extra={"input_len": input_len, "dataset": "sonnet"},
                probe_prose=(name == probe_cell),
            )
        if not args.keep_server:
            for k in [k for k in _servers if k[0] == context]:
                held = _servers.pop(k, None)
                if held is not None:
                    _stop(held[0])
    for held in list(_servers.values()):
        if not args.keep_server:
            _stop(held[0])
    _servers.clear()


    _print_table(report["cells"])
    ran = [c for c in report["cells"] if c["metrics"]]
    report["completed_cells"] = len(ran)
    report["total_cells"] = len(report["cells"])
    _write()
    print(f"\nbench: {len(ran)} of {len(report['cells'])} cell(s) measured — "
          f"report at {report_path}")
    print(f"##bench[cells_ok={len(ran)}]")
    return 0 if len(ran) == len(report["cells"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
