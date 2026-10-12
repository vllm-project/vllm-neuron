#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verifier for the DeepSeek-V4.1-Flash (timeshare) port.

The onboarding guide's smoke test with more inputs and a judge. Two pass
conditions, both required:

  1. **Nothing raised.** Engine construction and all 4 generations complete
     without an exception.
  2. **Nothing gibberish.** A Sonnet 5 judge at medium effort reads every
     completion and finds each one reasonable rather than gibberish.

Exits 0 only when both hold. Prints `##verifier[passed=1]` on success and
`##verifier[passed=0]` otherwise, so a queue can grep for it.

Inputs: 4 prompts — TranSonic's `short-prose` and its 8192-token
`long-needle-8192-0`, `"Hello, what is your name?"` and `"What is the capital
of France?"`. 32 tokens each, greedy. The TranSonic sets are copied
into `inputs/` beside this file and read via a path anchored to this package,
so the verifier needs nothing outside its own directory; `prompts.SELECTED_IDS`
picks which rows of them to use.

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

#: Slack above `longest prompt + MAX_TOKENS`, so a context sized to the exact sum does not
#: leave the scheduler with zero room. Small, because context costs KV cache, and on this
#: port KV cache competes with the graph set for the same HBM.
CONTEXT_HEADROOM = 64
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
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="engine tensor-parallel degree. Defaults to the port's own "
        "`config.TENSOR_PARALLEL_SIZE`, so the parallelism is the "
        "implementation's choice rather than the verifier's; pass this only to "
        "override it for one run.",
    )
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="engine context length. Default: measured — the longest selected prompt as "
        f"the real tokenizer encodes it, plus {MAX_TOKENS} generated tokens plus "
        f"{CONTEXT_HEADROOM} headroom. Pass this only to override, and note that a value "
        "too small drops cases rather than failing, which is recorded in the report's notes.",
    )
    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="default: the largest supported KV segment size below the resolved max_model_len, "
        "which is what auto-enables segmented prefill. A value >= max_model_len is single-shot "
        "prefill, which this build only accepts with prefix caching off — the verifier turns it "
        "off for you in that case.",
    )
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


def _phases(cases, kept, args) -> list[dict]:
    """Split the cases into one-engine-each groups: the longest alone, the rest together.

    The longest case is the one whose context and graph set dominate, and the one whose
    failures have historically taken the others down with it. Everything else shares a phase so
    the multi-sequence paths stay covered -- iteration 4's three state bugs (`SharedAttentionState`
    per batch, the n-gram hash cache shared between sequences, the indexer's mask under chunked
    prefill) were all invisible at `max_num_seqs=1`.

    An explicit `--max-model-len` is an override of the whole run, so it collapses to a single
    phase: the caller asked for a specific configuration and splitting it would silently not
    deliver it. `--max-num-seqs` is read the same way when it is distinguishable from the
    default.
    """
    pairs = list(zip(cases, kept))
    overridden = args.max_model_len is not None or getattr(
        args, "max_num_seqs_explicit", False
    )
    if overridden:
        return [{
            "name": "all",
            "cases": pairs,
            "max_model_len": None,          # resolved by the caller's own arguments
            "max_num_seqs": args.max_num_seqs,
        }]
    if len(pairs) < 2:
        return [{"name": "all", "cases": pairs, "max_model_len": None,
                 "max_num_seqs": args.max_num_seqs}]

    longest = max(pairs, key=lambda p: p[1][1])
    rest = [p for p in pairs if p is not longest]

    def context_for(group) -> int:
        """The phase's context: what its own prompts need, floored so a segment fits under it.

        Without the floor a phase of 37-to-158-token prompts asks for 254, no
        `SUPPORTED_KV_SEGMENT_SIZES` entry is below that, and it silently becomes a single-shot
        prefill with prefix caching off -- a path the gate's own default never takes. The floor
        keeps every phase on segmented prefill, which is the one the port is built for and the
        one its graphs are cached for.
        """
        from vllm_neuron.utils.bucket_utils import SUPPORTED_KV_SEGMENT_SIZES

        need = max(n for _, (_, n) in group) + MAX_TOKENS + CONTEXT_HEADROOM
        floor = min(SUPPORTED_KV_SEGMENT_SIZES) + MAX_TOKENS + CONTEXT_HEADROOM
        return max(need, floor)

    return [
        {"name": longest[0].id, "cases": [longest],
         "max_model_len": context_for([longest]), "max_num_seqs": 1},
        {"name": "short", "cases": rest,
         "max_model_len": context_for(rest), "max_num_seqs": args.max_num_seqs},
    ]


def _run_phase(phase: dict, args, tensor_parallel_size: int) -> list[dict]:
    """One engine, one `generate`, then tear it down so its HBM comes back."""
    import gc

    from vllm import LLM, SamplingParams
    from vllm_neuron.utils.bucket_utils import SUPPORTED_KV_SEGMENT_SIZES

    max_model_len = phase["max_model_len"] or args.max_model_len
    segments = sorted(s for s in SUPPORTED_KV_SEGMENT_SIZES if s < max_model_len)
    if args.max_num_batched_tokens is not None:
        mnbt = args.max_num_batched_tokens
    else:
        mnbt = segments[-1] if segments else max_model_len
    apc = mnbt < max_model_len

    print(f"\nverifier: phase {phase['name']} — max_model_len={max_model_len}, "
          f"max_num_batched_tokens={mnbt}, max_num_seqs={phase['max_num_seqs']}, "
          f"prefix caching {'on' if apc else 'off'}")
    phase.update(max_model_len=max_model_len, max_num_batched_tokens=mnbt,
                 enable_prefix_caching=apc)

    llm = LLM(
        model=args.model,
        max_num_seqs=phase["max_num_seqs"],
        max_model_len=max_model_len,
        max_num_batched_tokens=mnbt,
        tensor_parallel_size=tensor_parallel_size,
        enable_prefix_caching=apc,
    )
    try:
        outputs = llm.generate(
            [text for _, (text, _) in phase["cases"]],
            SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE),
        )
        out = []
        for (case, (_, n_prompt)), o in zip(phase["cases"], outputs):
            completion = o.outputs[0]
            out.append({
                "id": case.id,
                "phase": phase["name"],
                "source": case.source,
                "role": case.role,
                "expected_substring": case.expected_substring,
                "prompt_excerpt": case.excerpt(),
                "prompt_chars": len(case.prompt),
                "prompt_tokens": n_prompt,
                "text": completion.text,
                "num_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
            })
        return out
    finally:
        # `del` then collect: the engine holds the device until its process exits, and vLLM v1
        # runs EngineCore out of process, so dropping the last reference is what releases HBM.
        # Without this the next phase builds its engine alongside this one's resident graphs,
        # which is the accumulation the split exists to avoid.
        del llm
        gc.collect()
        print(f"verifier: phase {phase['name']} torn down")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # The port owns its parallelism (see `config.TENSOR_PARALLEL_SIZE`). Read at
    # run time, not import time, so it tracks whatever the implementation
    # currently says. A nonsensical value is rejected here rather than becoming
    # an obscure engine failure several minutes in.
    tensor_parallel_size = args.tensor_parallel_size
    source = "--tensor-parallel-size"
    if tensor_parallel_size is None:
        from ..config import TENSOR_PARALLEL_SIZE

        tensor_parallel_size = TENSOR_PARALLEL_SIZE
        source = "config.TENSOR_PARALLEL_SIZE"
    if not isinstance(tensor_parallel_size, int) or tensor_parallel_size < 1:
        print(
            f"verifier: {source} = {tensor_parallel_size!r}, which is not a positive "
            "integer",
            file=sys.stderr,
        )
        print("##verifier[passed=0]")
        return 1
    print(f"verifier: tensor_parallel_size={tensor_parallel_size} (from {source})")

    from .prompts import build_cases, render

    case_set = build_cases()

    # Render and measure with the real tokenizer *before* the engine exists, because
    # `max_model_len` is an engine-construction argument and the only thing that knows how long
    # a prompt is, is the tokenizer that will encode it.
    #
    # The earlier version sized the context from `target_tokens` and had vLLM reject its own
    # prompt: `long-needle-8192-0` is named for the 8192 tokens it was *generated to target* and
    # encodes to 9328 here, so a `8192 + MAX_TOKENS` context was 1136 short, the gate failed
    # with `VLLMValidationError` before a single token, and the `verifier` constraint was
    # unsatisfiable no matter what the port did. Nominal lengths are not lengths.
    prompts, render_notes = render(case_set, args.model, Path(args.chat_template))
    case_set.notes.extend(render_notes)
    lengths = [n for _, n in prompts]

    if args.max_model_len is not None:
        max_model_len = args.max_model_len
    else:
        max_model_len = (max(lengths) if lengths else 0) + MAX_TOKENS + CONTEXT_HEADROOM
        print(
            f"verifier: max_model_len={max_model_len} "
            f"(longest prompt {max(lengths) if lengths else 0} tok + {MAX_TOKENS} generated "
            f"+ {CONTEXT_HEADROOM} headroom)"
        )

    # Only now can a case be dropped honestly, against the budget it actually has to fit.
    budget = max_model_len - MAX_TOKENS
    kept: list[tuple[str, int]] = []
    kept_cases = []
    for case, (text, n) in zip(list(case_set.cases), prompts):
        if n > budget:
            case_set.notes.append(
                f"skipped {case.id}: {n} prompt tokens exceeds "
                f"max_model_len {max_model_len} - {MAX_TOKENS} generated = {budget}"
            )
            continue
        kept.append((text, n))
        kept_cases.append(case)
    case_set.cases = kept_cases

    # Chunked prefill, at the largest segment this context can use.
    #
    # The default used to be `max_model_len` — single-shot prefill — which this build rejects
    # before any model code runs, so the constraint was unsatisfiable regardless of the port for
    # a second reason after the nominal-length one above. Two rules collide:
    # `resolve_segmented_prefill_config` reads `max_num_batched_tokens >= max_model_len` as
    # single-shot and returns no `kv_segment_size_buckets`, and `neuron_model_runner` requires
    # those buckets whenever prefix caching is on — which vLLM 0.24 defaults to. The raise is in
    # `NeuronModelRunner.__init__`, upstream of `load_model`, so no model code could have
    # intervened. Every invocation in the repo's own `docs/` passes one or the other.
    #
    # `SUPPORTED_KV_SEGMENT_SIZES` rather than a literal 8192: the segment has to be a size the
    # attention kernel has buckets for, and strictly below `max_model_len` or it is single-shot
    # again — which is why this is not simply `or 8192`. A `--max-model-len 544` run would then
    # ask for a segment larger than its own context and fail the same way.
    from vllm_neuron.utils.bucket_utils import SUPPORTED_KV_SEGMENT_SIZES

    segments = sorted(s for s in SUPPORTED_KV_SEGMENT_SIZES if s < max_model_len)
    if args.max_num_batched_tokens is not None:
        max_num_batched_tokens = args.max_num_batched_tokens
    else:
        max_num_batched_tokens = segments[-1] if segments else max_model_len

    # Prefix caching needs segmented prefill. When the configuration is single-shot anyway —
    # the caller asked for it, or the context is below the smallest segment — turn APC off
    # instead of letting engine construction raise. This is the remedy the error names first.
    enable_prefix_caching = max_num_batched_tokens < max_model_len
    print(
        f"verifier: max_num_batched_tokens={max_num_batched_tokens} "
        f"({'segmented' if enable_prefix_caching else 'single-shot'} prefill, "
        f"prefix caching {'on' if enable_prefix_caching else 'off'})"
    )

    print(f"verifier: {len(case_set)} case(s)")
    for note in case_set.notes:
        print(f"  note: {note}")

    report: dict = {
        "model": args.model,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": max_model_len,
        # In the report because `runs-on-trainium`'s judge reads it, and because which prefill
        # mode ran is the difference between a 9k-token segment and a short one.
        "max_num_batched_tokens": max_num_batched_tokens,
        "enable_prefix_caching": enable_prefix_caching,
        "prompt_tokens": [n for _, n in kept],
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
    #
    # One engine per phase, torn down between them, because HBM does not come
    # back any other way. A phase's cost is not its own prompts' alone: every
    # prefill arrives with a different `read_len` as the cache fills, so each
    # keys its own `attn_half` graphs, and a graph stays resident with its code,
    # DMA rings and scratchpad. Measured across four prefills in one engine:
    # 13.3 -> 19.0 GiB of a 24 GiB bank, with the long prompt itself accounting
    # for 13.3 of it. The runtime scrubs HBM when a process attaches (the
    # driver logs `ncdev_hbm_scrub_start` on every `npid_attach`), so a second
    # engine starts from a clean bank rather than inheriting the first one's.
    #
    # It also sizes each phase for its own prompts. `max_model_len` is set by
    # the longest case in the phase and the KV ceiling is one of those per
    # sequence times the batch, so grouping a 2008-token needle with three
    # 37-to-158-token prompts made the short ones reserve the needle's context
    # four times over.
    #
    # And it isolates failure. Previously all four went through one `generate`,
    # so an engine that died on the long prompt returned `cases: []` and the
    # three that work produced no evidence at all.
    phases = _phases(case_set.cases, kept, args)
    report["phases"] = [
        {k: v for k, v in p.items() if k != "cases"} for p in phases
    ]
    print(f"verifier: {len(phases)} phase(s), one engine each")
    for p in phases:
        print(f"  {p['name']}: {len(p['cases'])} case(s), "
              f"max_model_len={p['max_model_len']}, max_num_seqs={p['max_num_seqs']}")

    results: list[dict] = []
    errors: list[str] = []
    for phase in phases:
        try:
            results.extend(_run_phase(phase, args, tensor_parallel_size))
        except Exception:  # noqa: BLE001 - a phase failing is a verifier failure
            errors.append(f"phase {phase['name']}:\n{traceback.format_exc()}")
            print(f"\nverifier: phase {phase['name']} FAILED", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

    report["cases"] = results
    if errors:
        report["execution_error"] = "\n\n".join(errors)
        _write()
        print(f"\nverifier: execution FAILED in {len(errors)} of {len(phases)} "
              f"phase(s) — report at {report_path}")
        print(f"  {len(results)} completion(s) from the phases that did run")
        print("##verifier[passed=0]")
        return 1

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
