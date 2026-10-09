#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verifier for the DeepSeek-V4.1-Flash (timeshare) port.

The onboarding guide's smoke test with more inputs and a judge. Two pass
conditions, both required:

  1. **Nothing raised.** Engine construction and all 11 generations complete
     without an exception.
  2. **Nothing gibberish.** A Sonnet 5 judge at medium effort reads every
     completion and finds each one reasonable rather than gibberish.

Exits 0 only when both hold. Prints `##verifier[passed=1]` on success and
`##verifier[passed=0]` otherwise, so a queue can grep for it.

Inputs: all 9 of TranSonic's sample prompts (6 short, 3 long at 2k/8k/16k),
the guide's `"Hello, my name is"`, and `"What is the capital of France?"`.
32 tokens each, greedy. The TranSonic sets are copied into `inputs/` beside
this file and read via a path anchored to this package, so the verifier needs
nothing outside its own directory.

    python -m vllm_neuron.model.deepseek_v41_timeshare.verifier.run_verifier \\
        --model <checkpoint-dir>

`--model` is required and has no default: the checkpoint lives outside this
repository, so naming a location here would bake an outside path into the
verifier. Every path the verifier *owns* resolves inside vllm-neuron.

NOTE: this runs the model, so it needs a Trainium device. It is the one piece
of this port that must not be run while another job holds the device.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

#: Per the spec: 32 tokens, greedy so a rerun is comparable.
MAX_TOKENS = 32
TEMPERATURE = 0.0

#: Anchored to this file's own location: no hardcoded absolute path to break if
#: the repo moves, no cwd-relative path to break depending on where the
#: verifier is launched from, and nothing resolving outside this package.
_VERIFIER_DIR = Path(__file__).resolve().parent
_PORT_DIR = _VERIFIER_DIR.parent

CHAT_TEMPLATE = _PORT_DIR / "chat_template.jinja"
DEFAULT_REPORT = _VERIFIER_DIR / "verifier_report.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        required=True,
        help="checkpoint directory (required; no default, so no path outside "
        "this repository is baked in). Use the local copy rather than "
        "re-downloading — see the port's references/INDEX.md.",
    )
    p.add_argument("--tensor-parallel-size", type=int, default=8)
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument(
        "--max-model-len",
        type=int,
        default=16384 + MAX_TOKENS,
        help="must fit the 16k long prompt plus the generation (default: 16416)",
    )
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument(
        "--chat-template",
        default=CHAT_TEMPLATE,
        help="applied to cases with role=user (default: the port's bundled template)",
    )
    p.add_argument(
        "--report",
        default=None,
        help=f"write the full result JSON here (default: {DEFAULT_REPORT})",
    )
    p.add_argument(
        "--skip-judge",
        action="store_true",
        help="run generations and report them, but do not call the judge. The "
        "run then cannot pass — use it to inspect raw output.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from .prompts import build_cases

    # Long prompts are dropped (and recorded) rather than overflowing the context.
    case_set = build_cases(max_prompt_tokens=max(args.max_model_len - MAX_TOKENS, 0))
    print(f"verifier: {len(case_set)} case(s)")
    for note in case_set.notes:
        print(f"  note: {note}")

    report: dict = {
        "model": args.model,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "notes": list(case_set.notes),
        "cases": [],
        "execution_error": None,
        "judge": None,
        "passed": False,
    }
    report_path = Path(args.report) if args.report else DEFAULT_REPORT

    def _write() -> None:
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    # ── Condition 1: nothing raised ───────────────────────────────────────
    results: list[dict] = []
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        sampling = SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE)

        tokenizer = llm.get_tokenizer()
        chat_template = None
        tpl = Path(args.chat_template)
        if tpl.is_file():
            chat_template = tpl.read_text()

        prompts: list[str] = []
        for case in case_set.cases:
            text = case.prompt
            if case.role == "user" and chat_template is not None:
                try:
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": case.prompt}],
                        chat_template=chat_template,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception as exc:  # noqa: BLE001 - recorded, not fatal
                    report["notes"].append(
                        f"{case.id}: chat template failed ({exc}); sent raw"
                    )
            prompts.append(text)

        outputs = llm.generate(prompts, sampling)

        for case, out in zip(case_set.cases, outputs):
            completion = out.outputs[0]
            results.append(
                {
                    "id": case.id,
                    "source": case.source,
                    "role": case.role,
                    "expected_substring": case.expected_substring,
                    "prompt_excerpt": case.excerpt(),
                    "prompt_chars": len(case.prompt),
                    "text": completion.text,
                    "num_tokens": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                }
            )
    except Exception:  # noqa: BLE001 - any exception is a verifier failure
        report["execution_error"] = traceback.format_exc()
        report["cases"] = results
        _write()
        print(report["execution_error"], file=sys.stderr)
        print(f"\nverifier: execution FAILED — report at {report_path}")
        print("##verifier[passed=0]")
        return 1

    report["cases"] = results

    print("\n" + "=" * 72)
    for r in results:
        print(f"[{r['id']}]  ({r['num_tokens']} tok, {r['finish_reason']})")
        print(f"  {r['text']!r}")
    print("=" * 72)

    if args.skip_judge:
        report["judge"] = {"skipped": True}
        _write()
        print(f"\nverifier: generations OK, judge skipped — report at {report_path}")
        print("##verifier[passed=0]")
        return 1

    # ── Condition 2: nothing gibberish ────────────────────────────────────
    from .judge import JUDGE_EFFORT, JUDGE_MODEL, judge_outputs

    print(f"\njudging {len(results)} completion(s) with {JUDGE_MODEL} "
          f"(effort {JUDGE_EFFORT})...")
    verdict = judge_outputs(results)
    report["judge"] = {
        "model": JUDGE_MODEL,
        "effort": JUDGE_EFFORT,
        "passed": verdict.passed,
        "error": verdict.error,
        "verdicts": [
            {"id": v.id, "verdict": v.verdict, "reason": v.reason}
            for v in verdict.verdicts
        ],
    }

    for v in verdict.verdicts:
        print(f"  {'ok  ' if v.ok else 'GIB '} {v.id}: {v.reason}")
    if verdict.error:
        print(f"  judge error: {verdict.error}", file=sys.stderr)

    report["passed"] = verdict.passed
    _write()

    print(f"\nverifier: {'PASSED' if verdict.passed else 'FAILED'} — "
          f"report at {report_path}")
    print(f"##verifier[passed={1 if verdict.passed else 0}]")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
