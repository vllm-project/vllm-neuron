#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assert DSpark speculative decoding accepts enough drafts to be worth having.

    python -m vllm_neuron.model.deepseek_v41.verifier.check_acceptance \\
        [--min-acceptance-pct 27] [--report <bench_report.json>]

Reads `bench/bench_report.json`, so it needs no device and costs seconds. The
`bench` constraint produces the evidence; this is the rule applied to it.

**Where the number comes from, and why not from the port.** The port already
reports an acceptance rate: `drafter.py`'s `_report_acceptance` logs a running
`kept/offered` line. That is the right thing for a human watching a run and the
wrong thing for a gate, because `drafter.py` is inside `scope.editable` -- the
loop can change the code that computes the number it is judged on. Counting
padding as accepted, counting the bonus token (sampled from the target, so
accepted by definition), or dividing by a smaller denominator each raise the
reported rate without improving a single prediction.

So the bar is applied to vLLM's own counters instead. The bench reads
`vllm:spec_decode_num_draft_tokens` and `vllm:spec_decode_num_accepted_tokens`
off a live server's `/metrics`, before and after a set of prose requests, and
records the delta as `report["prose"]`. Those counts are accumulated by vLLM's
scheduler in `update_from_output` (`SpecDecodingStats.observe_draft`) from the
model runner's output, in code this port cannot edit; `NeuronScheduler`
subclasses that scheduler and calls `super()`, which is what makes them
available on this backend at all. Each random cell's rate is recorded the same
way from its server log, with the port's own figure beside it, so a divergence
between the two is visible rather than silent -- they agreed to a few tenths of
a point when this was written.

**The bar applies to the prose probe only.** This is the one judgement in the
constraint and it is worth stating plainly, because it is the difference between
a bar that can be met and one that cannot. Acceptance is a statement about how
predictable the text is. `--dataset-name random` feeds random token *ids*, and a
drafter leaning on n-gram and Markov statistics has nothing to predict in noise:
this port measures 49% on prose against 16-18% on random. A 27% bar against the
random cells would therefore be a bar on the dataset, unreachable by any change
to the model. So the bench sends the verifier's own cases to a server it already
has up -- costing no extra engine build -- and that measurement is what is gated
here. The random cells' rates are reported, because they condition every
throughput figure in the matrix, and are not pass/fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_VERIFIER_DIR = Path(__file__).resolve().parent
_PORT_DIR = _VERIFIER_DIR.parent
DEFAULT_REPORT = _PORT_DIR / "bench" / "bench_report.json"

#: The bar, as a percentage of offered drafts accepted, on prose.
DEFAULT_MIN_ACCEPTANCE_PCT = 27.0

#: Below this many offered drafts the rate is noise rather than a measurement,
#: and a gate that passes on noise is worse than one that says so.
MIN_DRAFT_TOKENS = 100


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", default=str(DEFAULT_REPORT))
    p.add_argument(
        "--min-acceptance-pct", type=float, default=DEFAULT_MIN_ACCEPTANCE_PCT,
        help=f"fail below this (default {DEFAULT_MIN_ACCEPTANCE_PCT:g}%%)",
    )
    p.add_argument("--min-draft-tokens", type=int, default=MIN_DRAFT_TOKENS)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    path = Path(args.report)

    if not path.is_file():
        print(f"check_acceptance: no bench report at {path}.\n"
              "  Nothing to check, so this fails rather than passing vacuously. The `bench` "
              "constraint writes it; if that run failed, fix it first.")
        print("##check_acceptance[passed=0]")
        return 1

    report = json.loads(path.read_text())
    cells = report.get("cells") or []
    prose = report.get("prose")
    random_cells = [c for c in cells if c.get("dataset") == "random" and c.get("metrics")]

    print(f"check_acceptance: {report.get('model')} at "
          f"tensor_parallel_size={report.get('tensor_parallel_size')}")

    if random_cells:
        print("\n  random cells (reported, not gated — random token ids are not "
              "predictable by an n-gram drafter):")
        for c in sorted(random_cells, key=lambda c: (c["input_len"], c["batch"])):
            m = c["metrics"]
            pct = m.get("vllm_acceptance_pct")
            own = m.get("draft_acceptance_pct")
            print(f"    {c['name']:<13} vllm {m.get('vllm_accepted_tokens', 0):>5}/"
                  f"{m.get('vllm_draft_tokens', 0):<6} "
                  f"{(f'{pct:.1f}%' if pct is not None else '-'):>7}"
                  f"   port reported {(f'{own:.1f}%' if own is not None else '-'):>7}")

    problems: list[str] = []
    if not prose:
        problems.append(
            "the bench report has no `prose` section, so acceptance was never measured on "
            "text a drafter can predict. The bar applies to that measurement; run the bench "
            "without `--no-prose-probe`."
        )
    else:
        accepted = prose.get("accepted_tokens") or 0
        drafted = prose.get("draft_tokens") or 0
        pct = prose.get("acceptance_pct")
        drafts = prose.get("num_drafts") or 0
        mean_len = (1.0 + accepted / drafts) if drafts else None

        print(f"\n  prose probe on {prose.get('host_cell')}'s server "
              f"(context {prose.get('max_model_len')}, {prose.get('output_len')} out, "
              f"case(s) {prose.get('cases')}):")
        print(f"    {accepted}/{drafted} drafts accepted = "
              f"{(f'{pct:.1f}%' if pct is not None else 'unmeasured')} "
              f"(bar {args.min_acceptance_pct:g}%)"
              + (f", mean acceptance length {mean_len:.2f}" if mean_len else ""))
        print(f"    source: {prose.get('source')}")
        if pct is not None:
            print(f"##autohelix[spec_acceptance_pct={pct:.2f}]")

        if not prose.get("cases"):
            problems.append(
                "no prose prompt was accepted by the server, so nothing was measured. The "
                "probe skips prompts longer than the engine's context — check the bench's "
                "stderr for which were rejected, and widen the matrix if all of them were."
            )
        elif not drafted:
            problems.append(
                "vLLM counted no offered drafts at all over the prose requests. Either "
                "speculation was not configured (no `speculative_config`, so the drafter "
                "was never built) or nothing reached the verify path — either way the "
                "drafter is not being used, which is the thing this checks."
            )
        elif drafted < args.min_draft_tokens:
            problems.append(
                f"only {drafted} offered drafts, below the {args.min_draft_tokens} needed "
                "for the rate to mean anything. Raise `--prose-output-len` on the bench."
            )
        elif pct is not None and pct < args.min_acceptance_pct:
            problems.append(
                f"prose acceptance {pct:.1f}% is below the {args.min_acceptance_pct:g}% bar "
                f"({accepted} of {drafted} offered drafts kept)."
            )

    print()
    for problem in problems:
        print(f"  FAIL {problem}")
    if not problems:
        print("check_acceptance: speculation is accepting enough drafts to pay for itself")
    print(f"##check_acceptance[passed={0 if problems else 1}]")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
