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

#: Tokens per prompt in the prose cell. Acceptance is sampled over decode
#: steps, so this is the sample size: 96 tokens over the verifier's 4 cases
#: at ~1.5 accepted a step is ~250 draft decisions.
DEFAULT_PROSE_OUTPUT_LEN = 96

#: `--served-model-name`, and what every client here asks for.
SERVED_NAME = "deepseek-v41-flash"

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
    p.add_argument(
        "--prose-output-len", type=int, default=DEFAULT_PROSE_OUTPUT_LEN,
        help="tokens generated per prompt in the prose acceptance probe. Acceptance "
             "is sampled over decode steps, so this is the sample size.",
    )
    p.add_argument(
        "--no-prose-probe", action="store_true",
        help="skip the prose acceptance probe. `check_acceptance` then has nothing to "
             "read and fails, which is intended: an unmeasured bar is not a met one.",
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
        "--served-model-name", SERVED_NAME,
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


def _random_flags(input_len: int) -> callable:
    """`vllm bench serve` flags for a random-token cell of a given input length.

    Returns a callable taking the output length, because the warm-up and the
    measurement of one cell differ only in that.
    """
    def build(output_len: int) -> list[str]:
        return [
            "--dataset-name", "random",
            "--random-input-len", str(input_len),
            "--random-output-len", str(output_len),
            # Exact lengths, not a range: the cell is named for its shape.
            "--random-range-ratio", "0",
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


def _prose_probe(model: str, port: int, output_len: int, timeout: int) -> dict:
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
    from ..verifier import prompts
    from ..verifier.run_verifier import CHAT_TEMPLATE

    case_set = prompts.build_cases()
    rendered, notes = prompts.render(case_set, model, CHAT_TEMPLATE)
    for note in notes:
        print(f"bench:   {note}")

    before = _scrape_spec_counters(port)
    sent: list[str] = []
    generated = 0
    for case, (text, n_tok) in zip(case_set.cases, rendered):
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
            print(f"bench:   prose prompt {case.id} ({n_tok} tok) rejected: {exc}",
                  file=sys.stderr)
            continue
        sent.append(case.id)
        generated += (body.get("usage") or {}).get("completion_tokens") or 0

    after = _scrape_spec_counters(port)
    drafted = int(after["draft_tokens"] - before["draft_tokens"])
    accepted = int(after["accepted_tokens"] - before["accepted_tokens"])
    drafts = int(after["num_drafts"] - before["num_drafts"])
    return {
        "cases": sent,
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
        "--model", SERVED_NAME,
        "--tokenizer", model,
        *dataset_flags,
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
    started = time.monotonic()
    client = subprocess.Popen(cmd)
    while True:
        try:
            client.wait(timeout=5.0)
            break
        except subprocess.TimeoutExpired:
            pass
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


def _bench_cell(
    model: str, port: int, name: str, flags: callable, output_len: int, batch: int,
    num_prompts: int, result_dir: Path, timeout: int,
    server: subprocess.Popen | None = None,
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
    print(f"\nbench: cell {name} — warm-up: {num_prompts} prompts, "
          f"{WARMUP_OUTPUT_LEN} out (discarded)")
    warm = _bench_serve(
        model, port, flags(WARMUP_OUTPUT_LEN), batch, num_prompts,
        result_dir, f"{name}.warmup.json", timeout, server=server,
    )
    if warm is None:
        print(f"bench:   warm-up for {name} did not complete; the measurement "
              "below is a cold one", file=sys.stderr)

    print(f"bench: cell {name} — measuring: {num_prompts} prompts, "
          f"concurrency {batch}, {output_len} out")
    measured = _bench_serve(
        model, port, flags(output_len), batch, num_prompts,
        result_dir, f"{name}.json", timeout, server=server,
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


#: vLLM's own per-interval spec-decode line, logged by `SpecDecodingLogging.log`
#: whenever stat logging is on — which it is for a served engine. The counts are
#: per logging interval, not cumulative, so every line is summed.
_VLLM_SPEC_RE = re.compile(
    r"SpecDecoding metrics:.*?Accepted: (\d+) tokens, Drafted: (\d+) tokens"
)


def _vllm_acceptance(server_log: Path) -> dict:
    """This cell's acceptance rate as *vLLM* counted it, not as the port did.

    `_acceptance` above reads `drafter.py`'s own running line, which is the
    useful one to watch and the wrong one to trust: `drafter.py` is inside
    `scope.editable`, so the loop can change the code that computes the number
    it is judged on. vLLM's scheduler accumulates the same quantity in
    `SpecDecodingStats.observe_draft` from the model runner's output, in code
    this port cannot edit, and logs it as `Accepted: N tokens, Drafted: M`.

    Recorded here beside the port's own figure so the two can be compared. The
    gate that actually depends on acceptance is `verifier/check_acceptance.py`,
    which reads these counters directly through `LLM.get_metrics()` rather than
    scraping a log — and measures on prose, because random token ids are not
    predictable by an n-gram drafter at any quality.

    Returns zero counts rather than None when the pattern never matched, with
    `found: False`, so "not measured" stays distinguishable from "accepted
    nothing".
    """
    try:
        text = server_log.read_text(errors="replace")
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
    def run_cell(name: str, flags: callable, context: int, batch: int,
                 num_prompts: int, output_len: int, extra: dict,
                 probe_prose: bool = False) -> None:
        """Serve one engine, warm it, measure it, and record the cell.

        When `probe_prose`, the prose acceptance probe runs on this cell's
        server before teardown -- see `_prose_probe` for why it rides along on
        an engine that is already up instead of building one of its own.
        """
        port = _free_port(args.port)
        log = log_dir / f"server_{name}.log"
        proc = _serve(args.model, tp, context, batch, port, log)
        raw = None
        served_batch = batch
        try:
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
                result_dir, args.bench_timeout, server=proc,
            )
            if probe_prose:
                print(f"\nbench: prose acceptance probe on {name}'s server "
                      f"(context {context}, {args.prose_output_len} out)")
                report["prose"] = _prose_probe(
                    args.model, port, args.prose_output_len, args.bench_timeout,
                )
                report["prose"]["host_cell"] = name
                report["prose"]["max_model_len"] = context
                pr = report["prose"]
                pct = pr["acceptance_pct"]
                print(f"bench:   {pr['accepted_tokens']}/{pr['draft_tokens']} drafts "
                      f"accepted = {(f'{pct:.1f}%' if pct is not None else 'unmeasured')} "
                      f"over case(s) {pr['cases']}")
                _write()
        finally:
            if not args.keep_server:
                _stop(proc)
        metrics = _metrics(raw) if raw else None
        if metrics is not None:
            # Read after teardown: the counters are cumulative and this server
            # served exactly this cell, so they are this cell's and nothing
            # else's.
            metrics["draft_acceptance_pct"] = _acceptance(log)
            # The same quantity from vLLM's own counters, which the port cannot
            # edit. See `_vllm_acceptance`.
            vllm_spec = _vllm_acceptance(log)
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

    # The prose probe needs the widest context the matrix offers, and batch 1 so
    # it is not competing with 31 other sequences for the counters it reads.
    probe_cell = None
    if not args.no_prose_probe:
        widest = max(input_lens)
        probe_batch = min(batch_sizes)
        probe_cell = f"in{widest}_bs{probe_batch}"

    for batch in batch_sizes:
        for input_len in input_lens:
            name = f"in{input_len}_bs{batch}"
            run_cell(
                name=name,
                flags=_random_flags(input_len),
                context=args.max_model_len or (
                    input_len + args.output_len + CONTEXT_HEADROOM
                ),
                batch=batch,
                num_prompts=args.num_prompts or _num_prompts(batch),
                output_len=args.output_len,
                extra={"input_len": input_len, "dataset": "random"},
                probe_prose=(name == probe_cell),
            )


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
        suffix = c["name"]
        keys = ("ttft_p50_ms", "tpot_p50_ms", "decode_tok_s")
        if c.get("dataset") == "prose":
            # Its latencies are not comparable to the random cells', so only the
            # figure it exists for is emitted.
            keys = ("vllm_acceptance_pct",)
        for key in keys:
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
