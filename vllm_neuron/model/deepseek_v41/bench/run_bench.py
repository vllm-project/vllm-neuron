#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the DeepSeek-V4.1-Flash port with `vllm bench serve`.

The method the onboarding guide prescribes — "Benchmark and tune performance":
serve the model, then drive it with `vllm bench serve --dataset-name random`.
Random input is the point: it fixes the token count exactly, so a cell of the
matrix measures the shape it names rather than whatever a corpus happened to
tokenize to, and no two requests share a prefix that the prefix cache could
serve instead of computing.

    python -m vllm_neuron.model.deepseek_v41.bench.run_bench \\
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

#: The matrix. Overridable, but these are the sizes the deployment is specified
#: against.
DEFAULT_INPUT_LENS = (512, 1024, 2048, 8192)
DEFAULT_BATCH_SIZES = (1, 10)
DEFAULT_OUTPUT_LEN = 128

#: Requests per cell. Enough that a median means something, few enough that the
#: 8k x 10 cell does not dominate the run: at batch `b` this is `max(8, 2 * b)`,
#: so batch 1 sends 8 sequentially and batch 10 sends 20 as two full rounds.
def _num_prompts(batch: int) -> int:
    return max(8, 2 * batch)


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
        "serves — see the port's references/INDEX.md.",
    )
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="defaults to the port's own `config.TENSOR_PARALLEL_SIZE`, so the "
        "benchmark measures the parallelism the implementation chose",
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
        default=7200,
        help="seconds for one cell",
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
        "vllm_neuron.model.deepseek_v41.bench._serve_main",
        "--model", model,
        "--served-model-name", "deepseek-v41-flash",
        "--tensor-parallel-size", str(tp),
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
    return subprocess.Popen(
        cmd, stdout=handle, stderr=subprocess.STDOUT,
        # Its own process group, so tearing it down takes the 64 workers with
        # it rather than orphaning them on the devices.
        start_new_session=True,
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


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
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


#: Output tokens a warm-up request generates. Short on purpose: the warm-up
#: exists to *compile*, not to measure, and the decode shape this model keys is
#: `(1 + k) * running_seqs` rows — a function of the batch, not of how many
#: tokens have been emitted. The compressed-group read is pinned to the context
#: (`model._COMPRESS_READ_CEILING`) and the window is a fixed 128 slots, so a
#: longer generation reaches no shape a short one does not. 8 tokens compiles
#: the same graph set as 128 and costs a fifteenth of the time.
WARMUP_OUTPUT_LEN = 8


def _bench_serve(
    model: str, port: int, input_len: int, output_len: int, batch: int,
    num_prompts: int, result_dir: Path, result_file: str, timeout: int,
) -> dict | None:
    """One `vllm bench serve` invocation. Returns its parsed result, or None."""
    # The `vllm` CLI is what the onboarding guide names; fall back to the module
    # it dispatches to if this environment has no console script on PATH.
    vllm_bin = shutil.which("vllm")
    cmd = (
        [vllm_bin, "bench", "serve"] if vllm_bin
        else [sys.executable, "-m", "vllm.benchmarks.serve"]
    )
    cmd += [
        # `openai`, not `openai-chat`: the completions endpoint takes the random
        # dataset's token ids as given, where the chat endpoint would wrap them
        # in a template and change the input length the cell is named for.
        "--backend", "openai",
        "--host", "127.0.0.1", "--port", str(port),
        "--model", "deepseek-v41-flash",
        "--tokenizer", model,
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        # Exact lengths, not a range: the cell is named for its shape.
        "--random-range-ratio", "0",
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(batch),
        # Every request emits exactly `output_len` tokens, so TPOT and the
        # token count are not at the mercy of an early stop.
        "--ignore-eos",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,90,99",
        "--save-result",
        "--result-dir", str(result_dir),
        "--result-filename", result_file,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
    except subprocess.CalledProcessError as exc:
        print(f"bench:   exit {exc.returncode}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"bench:   timed out after {timeout} s", file=sys.stderr)
        return None

    path = result_dir / result_file
    if not path.is_file():
        print(f"bench:   no result written at {path}", file=sys.stderr)
        return None
    return json.loads(path.read_text())


def _bench_cell(
    model: str, port: int, input_len: int, output_len: int, batch: int,
    num_prompts: int, result_dir: Path, timeout: int,
) -> dict | None:
    """One cell: warm it up, then measure it.

    **The warm-up is not optional and its absence is not a small error.** A
    cell's first requests pay for that cell's own compiled buckets — the prefill
    shape for its input length, the decode shape for its batch — and on this
    model a graph can take minutes to build. At batch 10 the whole first round
    of ten requests pays it concurrently, which is half the sample at the
    default prompt count, so an unwarmed cell does not report a slow median: it
    reports a compile.

    So each cell runs twice. The first run is discarded and generates
    `WARMUP_OUTPUT_LEN` tokens instead of the full output, which is what makes
    it cheap.

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
    name = f"in{input_len}_bs{batch}"
    print(f"\nbench: cell {name} — warm-up: {num_prompts} prompts, "
          f"{input_len} in / {WARMUP_OUTPUT_LEN} out (discarded)")
    warm = _bench_serve(
        model, port, input_len, WARMUP_OUTPUT_LEN, batch, num_prompts,
        result_dir, f"{name}.warmup.json", timeout,
    )
    if warm is None:
        print(f"bench:   warm-up for {name} did not complete; the measurement "
              "below is a cold one", file=sys.stderr)

    print(f"bench: cell {name} — measuring: {num_prompts} prompts, "
          f"concurrency {batch}, {input_len} in / {output_len} out")
    measured = _bench_serve(
        model, port, input_len, output_len, batch, num_prompts,
        result_dir, f"{name}.json", timeout,
    )
    if measured is not None:
        # Recorded on the result, so a reader of the report can tell a warmed
        # measurement from a cold one without going back to the console.
        measured["_warmed"] = warm is not None
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


def _acceptance(server_log: Path) -> float | None:
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
        text = server_log.read_text(errors="replace")
    except OSError:
        return None
    matches = _ACCEPT_RE.findall(text)
    if not matches:
        return None
    kept, offered, _pct = matches[-1]
    offered_n = int(offered)
    return 100.0 * int(kept) / offered_n if offered_n else None


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
        # False means the cell's warm-up did not complete, so these figures
        # include that cell's compile cost. See `_bench_cell`.
        "warmed": raw.get("_warmed", False),
    }


def _print_table(cells: list[dict]) -> None:
    print("\n" + "=" * 86)
    print("DeepSeek-V4.1-Flash — forward inference, `vllm bench serve`, random input")
    print("=" * 94)
    header = (f"{'input':>7} {'batch':>6} {'out':>5} {'ctx':>7} "
              f"{'TTFT p50':>10} {'TPOT p50':>10} {'decode':>10} "
              f"{'per-str':>9} {'accept':>7} {'done':>5}")
    print(header)
    print(f"{'':>7} {'':>6} {'':>5} {'':>7} {'ms':>10} {'ms':>10} "
          f"{'tok/s':>10} {'tok/s':>9} {'%':>7} {'':>5}")
    print("-" * 94)
    for c in cells:
        m = c["metrics"]
        if m is None:
            print(f"{c['input_len']:>7} {c['batch']:>6} {c['output_len']:>5} "
                  f"{c.get('max_model_len', '-'):>7} {'FAILED':>10}")
            continue

        def fmt(v, spec=".1f"):
            return format(v, spec) if isinstance(v, (int, float)) else "-"

        print(f"{c['input_len']:>7} {c['batch']:>6} {c['output_len']:>5} "
              f"{c.get('max_model_len', '-'):>7} "
              f"{fmt(m['ttft_p50_ms']):>10} {fmt(m['tpot_p50_ms']):>10} "
              f"{fmt(m['decode_tok_s'], '.2f'):>10} "
              f"{fmt(m['per_stream_tok_s'], '.2f'):>9} "
              f"{fmt(m.get('draft_acceptance_pct'), '.0f'):>7} "
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
        from ..config import TENSOR_PARALLEL_SIZE

        tp, source = TENSOR_PARALLEL_SIZE, "config.TENSOR_PARALLEL_SIZE"
    print(f"bench: tensor_parallel_size={tp} (from {source})")

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
    for batch in batch_sizes:
        for input_len in input_lens:
            context = args.max_model_len or (
                input_len + args.output_len + CONTEXT_HEADROOM
            )
            port = _free_port(args.port)
            log = log_dir / f"server_in{input_len}_bs{batch}.log"
            proc = _serve(args.model, tp, context, batch, port, log)
            raw = None
            served_batch = batch
            try:
                _wait_healthy(proc, port, args.server_timeout)
                served_batch = _engine_batch(log, batch)
                if served_batch < batch:
                    print(
                        f"bench: WARNING cell in{input_len}_bs{batch} — the "
                        f"engine clamped max_num_seqs to {served_batch}, so it "
                        f"serves {served_batch} sequence(s) at a time while the "
                        f"client keeps {batch} in flight. TTFT will be mostly "
                        "queue wait and the throughput will be "
                        f"{served_batch}-way, not {batch}-way. See "
                        "deepseek_v41.MULTI_RANK_MAX_NUM_SEQS.",
                        file=sys.stderr,
                    )
                n = args.num_prompts or _num_prompts(batch)
                raw = _bench_cell(
                    args.model, port, input_len, args.output_len, batch, n,
                    result_dir, args.bench_timeout,
                )
            finally:
                if not args.keep_server:
                    _stop(proc)
            metrics = _metrics(raw) if raw else None
            if metrics is not None:
                # Read after teardown: the drafter's counters are cumulative and
                # this server served exactly this cell, so its last line is this
                # cell's acceptance rate and nothing else's.
                metrics["draft_acceptance_pct"] = _acceptance(log)
            cell = {
                "input_len": input_len,
                "output_len": args.output_len,
                "batch": batch,
                "max_model_len": context,
                "engine_max_num_seqs": served_batch,
                "concurrency_capped": served_batch < batch,
                "num_prompts": args.num_prompts or _num_prompts(batch),
                "metrics": metrics,
                "server_log": str(log),
            }
            report["cells"].append(cell)
            _write()  # after every cell, so a crash keeps what ran

    _print_table(report["cells"])

    # ── `##autohelix[...]`, one line per figure per cell ──────────────────
    #
    # Named `<metric>_in<input>_bs<batch>` so a loop can gate on a single cell
    # without parsing a table. Only figures that exist are emitted: a missing
    # metric must not read as a zero.
    for c in report["cells"]:
        m = c["metrics"]
        if not m:
            continue
        suffix = f"in{c['input_len']}_bs{c['batch']}"
        for key in ("ttft_p50_ms", "tpot_p50_ms", "decode_tok_s"):
            value = m.get(key)
            if isinstance(value, (int, float)):
                print(f"##autohelix[{key}_{suffix}={value:.6g}]")

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
