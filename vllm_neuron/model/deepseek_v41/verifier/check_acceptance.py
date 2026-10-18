#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assert DSpark speculative decoding accepts enough drafts to be worth having.

    python -m vllm_neuron.model.deepseek_v41.verifier.check_acceptance \\
        --model <checkpoint-dir> [--min-acceptance-pct 27]

**Where the number comes from, and why not from the port.** The port already
reports an acceptance rate: `drafter.py`'s `_report_acceptance` logs a running
`kept/offered` line, and `bench/run_bench.py` reads it. That is the right thing
for a human watching a run and the wrong thing for a gate, because `drafter.py`
is inside `scope.editable` -- the loop can change the code that computes the
number it is judged on. Counting padding as accepted, counting the bonus token
(which is sampled from the target and accepted by definition), or dividing by a
smaller denominator each raise the reported rate without improving a single
prediction.

So this reads vLLM's own counters instead, through `LLM.get_metrics()`:

    vllm:spec_decode_num_draft_tokens      drafts offered to the verifier
    vllm:spec_decode_num_accepted_tokens   drafts the verifier kept

They are accumulated by vLLM's scheduler in `update_from_output`
(`SpecDecodingStats.observe_draft`) from the model runner's output, in code this
port cannot edit. `NeuronScheduler` subclasses vLLM's `Scheduler` and calls
`super().update_from_output`, which is what makes them available on this
backend at all. `vllm:spec_decode_num_accepted_tokens_per_pos` is reported too,
because a drafter that only ever lands its first token is a different problem
from one that lands a short run.

**Prose, not random tokens.** The input is the verifier's own cases -- the same
prompts the correctness gate uses. That is deliberate and it is the only honest
choice for a *threshold*: acceptance is a statement about how predictable the
text is, and `vllm bench serve --dataset-name random` feeds random token *ids*,
which an n-gram/Markov drafter cannot predict even in principle. Measured on
this port: 49% on prose against 16-18% on random. Gating 27% against random
data would be gating the dataset, and no change to the model could pass it. The
bench still records its own per-cell rate; it is just not a pass/fail line.

Needs a Trainium device: it runs the model. Builds one engine, generates, reads
the counters, and exits -- no judge, no correctness claim. `run_verifier.py` is
the correctness gate; this one is only about the drafter paying for itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Tokens generated per case. Longer than the correctness gate's 32, because
#: this is a measurement over decode steps and 32 tokens at ~1.5 accepted a step
#: is a sample of about 20. Still short enough that the engine build dominates.
MAX_TOKENS = 96
TEMPERATURE = 0.0

#: Slack above `longest prompt + MAX_TOKENS`, as in `run_verifier.py`.
CONTEXT_HEADROOM = 64

#: The bar, as a percentage of offered drafts accepted. Overridable on the
#: command line; this is the default the constraint runs with.
DEFAULT_MIN_ACCEPTANCE_PCT = 27.0

_VERIFIER_DIR = Path(__file__).resolve().parent
_PORT_DIR = _VERIFIER_DIR.parent

CHAT_TEMPLATE = _PORT_DIR / "chat_template.jinja"
DEFAULT_REPORT = _VERIFIER_DIR / "acceptance_report.json"

_DRAFT = "vllm:spec_decode_num_draft_tokens"
_ACCEPTED = "vllm:spec_decode_num_accepted_tokens"
_PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos"
_NUM_DRAFTS = "vllm:spec_decode_num_drafts"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="checkpoint directory")
    p.add_argument(
        "--min-acceptance-pct", type=float, default=DEFAULT_MIN_ACCEPTANCE_PCT,
        help=f"fail below this (default {DEFAULT_MIN_ACCEPTANCE_PCT:g}%%)",
    )
    p.add_argument("--tensor-parallel-size", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--report", default=str(DEFAULT_REPORT))
    p.add_argument(
        "--chat-template", default=str(CHAT_TEMPLATE),
        help="applied to cases with role=user (default: the port's bundled template)",
    )
    return p.parse_args(argv)


def _counters(metrics: list) -> dict:
    """The spec-decode counters out of a `get_metrics()` snapshot.

    Summed over label sets: with one engine there is one, but a counter that
    grew under two label sets must not be read as whichever came last.
    """
    from vllm.v1.metrics.reader import Counter, Vector

    out: dict = {"draft_tokens": 0, "accepted_tokens": 0, "num_drafts": 0,
                 "accepted_per_pos": None, "found": []}
    for m in metrics:
        if m.name in (_DRAFT, _ACCEPTED, _NUM_DRAFTS) and isinstance(m, Counter):
            key = {_DRAFT: "draft_tokens", _ACCEPTED: "accepted_tokens",
                   _NUM_DRAFTS: "num_drafts"}[m.name]
            out[key] += int(m.value)
            out["found"].append(m.name)
        elif m.name == _PER_POS and isinstance(m, Vector):
            prev = out["accepted_per_pos"]
            vals = [int(v) for v in m.values]
            out["accepted_per_pos"] = (
                vals if prev is None else [a + b for a, b in zip(prev, vals)]
            )
            out["found"].append(m.name)
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from . import prompts
    from ..config import TENSOR_PARALLEL_SIZE, MAX_NUM_SEQS

    tp = args.tensor_parallel_size or TENSOR_PARALLEL_SIZE
    case_set = prompts.build_cases()
    selected = case_set.cases

    from vllm import LLM, SamplingParams
    from vllm_neuron.utils.bucket_utils import SUPPORTED_KV_SEGMENT_SIZES

    # `render` takes the model path, not a tokenizer: it loads the checkpoint's
    # own tokenizer so the token counts are the ones the engine will see.
    rendered, notes = prompts.render(case_set, args.model, Path(args.chat_template))
    for note in [*case_set.notes, *notes]:
        print(f"check_acceptance: {note}")

    lengths = [n for _, n in rendered]
    max_model_len = max(lengths) + args.max_tokens + CONTEXT_HEADROOM
    segments = sorted(s for s in SUPPORTED_KV_SEGMENT_SIZES if s < max_model_len)
    mnbt = segments[-1] if segments else max_model_len
    apc = mnbt < max_model_len

    print(f"check_acceptance: {len(selected)} prose case(s), {args.max_tokens} tokens each, "
          f"greedy; tensor_parallel_size={tp}, max_model_len={max_model_len}, "
          f"max_num_seqs={MAX_NUM_SEQS}")

    # `disable_log_stats=False` is what makes the counters exist: with stat
    # logging off, vLLM never builds the Prometheus metrics this reads. The
    # offline `LLM` defaults it to on for throughput, so it has to be asked for.
    llm = LLM(
        model=args.model,
        max_num_seqs=MAX_NUM_SEQS,
        max_model_len=max_model_len,
        max_num_batched_tokens=mnbt,
        tensor_parallel_size=tp,
        enable_prefix_caching=apc,
        disable_log_stats=False,
    )

    texts = [text for text, _ in rendered]
    outputs = llm.generate(
        texts, SamplingParams(max_tokens=args.max_tokens, temperature=TEMPERATURE)
    )
    generated = sum(len(o.outputs[0].token_ids) for o in outputs)

    counters = _counters(llm.get_metrics())
    draft = counters["draft_tokens"]
    accepted = counters["accepted_tokens"]
    pct = (100.0 * accepted / draft) if draft else None

    report = {
        "model": args.model,
        "tensor_parallel_size": tp,
        "max_num_seqs": MAX_NUM_SEQS,
        "cases": [c.id for c in selected],
        "max_tokens": args.max_tokens,
        "tokens_generated": generated,
        "source": "vllm:spec_decode_* counters via LLM.get_metrics()",
        "draft_tokens": draft,
        "accepted_tokens": accepted,
        "num_drafts": counters["num_drafts"],
        "accepted_per_pos": counters["accepted_per_pos"],
        "acceptance_pct": pct,
        "min_acceptance_pct": args.min_acceptance_pct,
    }

    problems: list[str] = []
    if not counters["found"]:
        problems.append(
            "vLLM reported no `vllm:spec_decode_*` counters at all. Either speculation "
            "was not configured for this run (no `speculative_config`, so the drafter was "
            "never built) or nothing reached the verify path. Either way the drafter is "
            "not being used, which is the thing this checks."
        )
    elif not draft:
        problems.append(
            f"{_DRAFT} is 0: no draft was ever offered to the verifier over "
            f"{generated} generated token(s). The MTP stages may be computing drafts that "
            "nothing proposes to the scheduler."
        )
    elif pct is not None and pct < args.min_acceptance_pct:
        problems.append(
            f"acceptance {pct:.1f}% is below the {args.min_acceptance_pct:g}% bar "
            f"({accepted} of {draft} offered drafts kept)."
        )

    report["passed"] = not problems
    report["problems"] = problems
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")

    print()
    if pct is not None:
        mean_len = (1.0 + accepted / counters["num_drafts"]) if counters["num_drafts"] else None
        print(f"check_acceptance: {accepted}/{draft} drafts accepted = {pct:.1f}% "
              f"(bar {args.min_acceptance_pct:g}%)"
              + (f", mean acceptance length {mean_len:.2f}" if mean_len else ""))
        if counters["accepted_per_pos"]:
            print(f"check_acceptance: accepted per draft position "
                  f"{counters['accepted_per_pos']}")
        print(f"##autohelix[spec_acceptance_pct={pct:.2f}]")
    for problem in problems:
        print(f"  FAIL {problem}")
    if not problems:
        print("check_acceptance: speculation is accepting enough drafts to pay for itself")
    print(f"##check_acceptance[passed={0 if problems else 1}]")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
