#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply the deployment's performance and concurrency bars to a bench run.

    python -m vllm_neuron.model.deepseek_v41.verifier.check_targets \\
        --check latency      [--report <bench_report.json>]
    python -m vllm_neuron.model.deepseek_v41.verifier.check_targets \\
        --check concurrency  [--target-batch 32]

Reads `bench/bench_report.json`, so it needs no device and costs seconds. The
`bench` constraint is what produces that report; these two are the rules applied
to it. Splitting them means a run that is fast but serializes, or one that
batches correctly but misses the latency bar, fails the constraint that is
actually wrong rather than one combined verdict.

It lives under `verifier/` rather than `bench/` for the same reason every other
`check_*` does: `verifier/` and `bench/` are both outside `scope.editable`, but
keeping the *bars* beside the other gates makes it obvious that moving one is
not something an iteration can do.

**`--check latency`** -- two bars, and they are read differently on purpose.

  TTFT < 1.5 s per 1k input tokens, on **every** measured cell. Latency is a
  per-sample property, so a cell that misses it is a configuration a user would
  experience as slow, whatever the others do. Normalising by input length is
  what makes one bar meaningful across 512 and 8192 tokens; `ttft_p50` is
  already per-request, so no further per-sample reduction is needed.

  Decode > 50 tokens/s, on the **concurrent** cells only, aggregate. This is the
  interpretation to check if the number looks generous: `decode_tok_s` is
  `vllm bench serve`'s `output_throughput`, which is output tokens per second
  summed over everything in flight -- the figure a serving deployment is rated
  by. Per-stream decode on this port is ~4.9 tokens/s (TPOT ~206 ms), so 50
  tokens/s is not reachable at batch 1 by any amount of tuning short of a 10x
  single-stream win; it is reachable at batch 32 by batching working at all.
  Gating batch 1 on it would be gating something arithmetic forbids, so the
  batch-1 figure is reported and the bar is applied where throughput is won.
  `per_stream_tok_s` is printed alongside so the two views cannot be confused.

**`--check concurrency`** -- that 32 concurrent requests are really served
concurrently, and that the run ends.

  Three separate things, because the failure this is written against passed two
  of them. A batch-10 cell measured here reported plausible numbers while the
  engine had clamped itself to `max_num_seqs 10 -> 1`: the requests completed,
  the latencies were real, and what was measured was a queue. So:

    1. the engine actually admitted the batch (`engine_max_num_seqs >= 32`),
    2. every request finished (`completed == num_prompts`) -- no hang, no stall,
       and the server exited rather than being killed,
    3. aggregate throughput is several times the batch-1 figure at the same
       input length. Serialised requests give a batch-32 cell the *same*
       throughput as batch 1; genuine batching multiplies it. This is the test
       that distinguishes them, and it is why the bench matrix has to contain
       both batches at one input length for this check to run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_VERIFIER_DIR = Path(__file__).resolve().parent
_PORT_DIR = _VERIFIER_DIR.parent
DEFAULT_REPORT = _PORT_DIR / "bench" / "bench_report.json"

#: TTFT bar, milliseconds per 1024 input tokens.
TTFT_MS_PER_1K = 1500.0

#: Aggregate decode bar, output tokens per second, on concurrent cells.
MIN_DECODE_TOK_S = 50.0

#: The concurrency the deployment has to support.
DEFAULT_TARGET_BATCH = 32

#: How much more aggregate throughput a concurrent cell must deliver than
#: batch 1 at the same input length before it counts as really batching.
#: Deliberately far below the ideal `target_batch` x -- speculation, the
#: scheduler and the shared prefill all take their cut, and the question here is
#: "is this concurrent or serialised", not "is it perfectly parallel". A
#: serialised cell scores ~1.0.
DEFAULT_MIN_SPEEDUP = 4.0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", required=True, choices=("latency", "concurrency"))
    p.add_argument("--report", default=str(DEFAULT_REPORT))
    p.add_argument("--ttft-ms-per-1k", type=float, default=TTFT_MS_PER_1K)
    p.add_argument("--min-decode-tok-s", type=float, default=MIN_DECODE_TOK_S)
    p.add_argument("--target-batch", type=int, default=DEFAULT_TARGET_BATCH)
    p.add_argument("--min-speedup", type=float, default=DEFAULT_MIN_SPEEDUP)
    return p.parse_args(argv)


def _load(path: Path) -> dict | None:
    import json

    if not path.is_file():
        print(f"check_targets: no bench report at {path}.\n"
              "  Nothing to check, so this fails rather than passing vacuously. The `bench` "
              "constraint writes it; if that run failed, fix it first.")
        return None
    return json.loads(path.read_text())


def _cells(report: dict) -> list[dict]:
    """The measured cells the latency and concurrency bars apply to.

    Every cell is a random-token cell of a known input length, which is what
    makes a TTFT-per-1k figure mean anything. The prose measurement is not a
    cell at all -- it is `report["prose"]`, a probe that rides along on one of
    these servers -- so it is not here to be excluded. The `dataset` filter is
    kept so that a future prose cell could not silently acquire a latency bar
    normalised against prompt lengths it does not control.
    """
    return [c for c in (report.get("cells") or [])
            if c.get("metrics") and c.get("dataset") != "prose"]


def _label(cell: dict) -> str:
    return cell.get("name") or f"in{cell['input_len']}_bs{cell['batch']}"


def _check_latency(report: dict, args) -> list[str]:
    problems: list[str] = []
    cells = _cells(report)
    skipped = [c for c in (report.get("cells") or [])
               if not c.get("metrics") and c.get("dataset") != "prose"]

    if not cells:
        return ["the bench report has no cell with measurements in it."]
    for cell in skipped:
        problems.append(
            f"cell {_label(cell)} produced no measurements, so its latency is unknown. "
            "An unmeasured configuration is not a passing one."
        )

    print("  cell          input    TTFT p50   per 1k      TPOT p50   decode  per-stream")
    print("  " + "-" * 74)
    for cell in sorted(cells, key=lambda c: (c["input_len"], c["batch"])):
        m = cell["metrics"]
        ttft = m.get("ttft_p50_ms")
        per_1k = (ttft / (cell["input_len"] / 1024.0)) if ttft else None
        decode = m.get("decode_tok_s")
        stream = m.get("per_stream_tok_s")
        tpot = m.get("tpot_p50_ms")
        over = "" if per_1k is None or per_1k <= args.ttft_ms_per_1k else " OVER"
        ttft_s = f"{ttft:.0f} ms" if ttft else "-"
        per_1k_s = f"{per_1k:.0f} ms" if per_1k else "-"
        tpot_s = f"{tpot:.0f} ms" if tpot else "-"
        decode_s = f"{decode:.1f}" if decode else "-"
        stream_s = f"{stream:.1f}" if stream else "-"
        print(f"  {_label(cell):<13} {cell['input_len']:>5}   {ttft_s:>9}   "
              f"{per_1k_s:>7}{over:<5}   {tpot_s:>9}  {decode_s:>7}  {stream_s:>10}")
        if per_1k is None:
            problems.append(f"cell {_label(cell)} reported no TTFT.")
        elif per_1k > args.ttft_ms_per_1k:
            problems.append(
                f"cell {_label(cell)}: TTFT {ttft:.0f} ms over {cell['input_len']} input "
                f"tokens is {per_1k:.0f} ms per 1k, above the "
                f"{args.ttft_ms_per_1k:.0f} ms bar."
            )
        if per_1k is not None:
            print(f"##autohelix[ttft_ms_per_1k_{_label(cell)}={per_1k:.1f}]")

    # Decode: concurrent cells only, and say so when there are none rather than
    # quietly having nothing to check.
    concurrent = [c for c in cells if c["batch"] > 1 and not c.get("concurrency_capped")]
    if not concurrent:
        problems.append(
            "no concurrent cell produced a figure, so aggregate decode throughput was never "
            f"measured. The bar ({args.min_decode_tok_s:g} tok/s aggregate) applies to "
            "concurrent cells — per-stream decode here is ~4.9 tok/s, so batch 1 cannot "
            "reach it. Run the bench with a batch size above 1, and see `--check "
            "concurrency` if the engine clamped it."
        )
    else:
        best = max(concurrent, key=lambda c: c["metrics"].get("decode_tok_s") or 0.0)
        best_tok_s = best["metrics"].get("decode_tok_s") or 0.0
        print(f"\n  best aggregate decode: {best_tok_s:.1f} tok/s at {_label(best)} "
              f"(bar {args.min_decode_tok_s:g})")
        print(f"##autohelix[decode_tok_s_best={best_tok_s:.2f}]")
        if best_tok_s < args.min_decode_tok_s:
            problems.append(
                f"best aggregate decode {best_tok_s:.1f} tok/s (at {_label(best)}) is below "
                f"the {args.min_decode_tok_s:g} tok/s bar."
            )
    return problems


def _check_concurrency(report: dict, args) -> list[str]:
    problems: list[str] = []
    target = args.target_batch
    all_cells = [c for c in (report.get("cells") or [])
                 if c.get("dataset") != "prose"]
    at_target = [c for c in all_cells if c["batch"] == target]

    if not at_target:
        return [
            f"the bench report has no cell at batch {target}, so the concurrency this "
            f"constraint is about was never exercised. Run the bench with "
            f"`--batch-sizes 1,{target}`."
        ]

    baselines = {c["input_len"]: c for c in all_cells
                 if c["batch"] == 1 and c.get("metrics")}

    for cell in sorted(at_target, key=lambda c: c["input_len"]):
        label = _label(cell)
        metrics = cell.get("metrics")
        served = cell.get("engine_max_num_seqs")

        # 1. did the engine admit the batch at all?
        if cell.get("concurrency_capped"):
            problems.append(
                f"cell {label}: the engine clamped itself to max_num_seqs={served} after "
                f"being asked for {target}. Whatever it measured, it was not {target} "
                "concurrent sequences — `config.MAX_NUM_SEQS` is the knob, and the stall "
                "recorded beside it is why it is where it is."
            )
            continue

        # 2. did every request finish?
        if metrics is None:
            problems.append(
                f"cell {label} produced no measurements. A concurrent cell that returns "
                "nothing is the hang this constraint exists to catch — check the cell's "
                f"server log ({cell.get('server_log')}) for where it stopped."
            )
            continue
        completed = metrics.get("completed")
        wanted = cell.get("num_prompts")
        if completed != wanted:
            problems.append(
                f"cell {label}: {completed} of {wanted} requests completed. The rest "
                "hung, stalled or errored; a partial run is not support for this "
                "concurrency."
            )

        # 3. was it concurrent, or a queue?
        base = baselines.get(cell["input_len"])
        agg = metrics.get("decode_tok_s")
        if base is None:
            problems.append(
                f"cell {label}: no batch-1 cell at input {cell['input_len']} to compare "
                "against, so serialisation cannot be ruled out. The bench matrix needs "
                f"both batch 1 and batch {target} at one input length."
            )
        elif agg:
            base_agg = base["metrics"].get("decode_tok_s") or 0.0
            speedup = (agg / base_agg) if base_agg else None
            if speedup is None:
                print(f"  {label}: {agg:.1f} tok/s aggregate (no batch-1 figure to "
                      "compare against)")
            else:
                print(f"  {label}: {agg:.1f} tok/s aggregate against {base_agg:.1f} at "
                      f"batch 1 = {speedup:.1f}x")
            if metrics.get("ttft_p50_ms"):
                ttft_ratio = (metrics.get("ttft_p99_ms") or 0.0) / metrics["ttft_p50_ms"]
                print(f"    TTFT p99/p50 {ttft_ratio:.2f}, engine max_num_seqs {served}, "
                      f"{completed}/{wanted} completed")
            if speedup is not None:
                print(f"##autohelix[concurrency_speedup_in{cell['input_len']}="
                      f"{speedup:.2f}]")
                if speedup < args.min_speedup:
                    problems.append(
                        f"cell {label}: aggregate throughput is only {speedup:.1f}x the "
                        f"batch-1 figure, below the {args.min_speedup:g}x bar. {target} "
                        "requests served one after another score about 1.0x, so this looks "
                        "like a queue rather than a batch. Compare TPOT: real batching "
                        "raises per-step cost a little and multiplies the token count; "
                        "serialisation leaves both alone."
                    )
    return problems


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = _load(Path(args.report))
    if report is None:
        print(f"##check_targets_{args.check}[passed=0]")
        return 1

    print(f"check_targets --check {args.check}: {report.get('model')} at "
          f"tensor_parallel_size={report.get('tensor_parallel_size')}, "
          f"output_len={report.get('output_len')}")
    problems = (_check_latency if args.check == "latency" else _check_concurrency)(
        report, args
    )

    print()
    for problem in problems:
        print(f"  FAIL {problem}")
    if not problems:
        print(f"check_targets: the {args.check} bars hold")
    print(f"##check_targets_{args.check}[passed={0 if problems else 1}]")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
