# Verifier — DeepSeek-V4.1-Flash (timeshare)

The onboarding guide's smoke test, widened to 11 inputs and given a judge.

```bash
# from the vllm-neuron repo root
.venv/bin/python -m vllm_neuron.model.deepseek_v41_timeshare.verifier.run_verifier \
    --model <checkpoint-dir>
```

`--model` is **required and has no default**. The checkpoint lives outside this
repository, and the verifier deliberately holds no path to anything outside
`vllm-neuron` — so the location is yours to pass, not ours to assume. Use the
local copy rather than re-downloading; the port's `references/INDEX.md` says
where it is.

> **This one runs the model, so it needs a free Trainium device.** Every other
> part of this port is CPU-only. Check `neuron-ls` before starting — the hc
> optimization queue holds the device for hours at a time.

## Pass conditions

Both are required; `##verifier[passed=1]` and exit 0 only when both hold.

| | condition | how |
|---|---|---|
| 1 | nothing raised | engine construction and all 11 generations complete without an exception |
| 2 | nothing gibberish | Sonnet 5 at medium effort reads every completion and finds each reasonable |

The judge **fails closed**. A reply that will not parse, a missing per-case
verdict, a nonzero exit, a timeout, or a missing `claude` CLI are all failures —
never a pass by default. A judge that cannot answer has not cleared the model.

## Inputs — 11 cases, 32 tokens each, greedy

| source | cases |
|---|---|
| TranSonic `short.jsonl` | all 6: `short-prose`, `short-code`, `short-math`, `short-reasoning`, `short-summary`, `short-dialogue` |
| TranSonic `long.jsonl` | all 3 needle-in-a-haystack: 2k, 8k, 16k tokens |
| onboarding guide | `smoke-test` — `"Hello, my name is"`, verbatim |
| trivia | `trivia-capital-france` — `"What is the capital of France?"` |

The TranSonic sets are **copied into `inputs/`**, byte-for-byte from their
source, and read via a path anchored to this package. There is no
`--inputs-dir` and no environment
override: every run uses the same 11 prompts, and the verifier needs nothing
outside `vllm-neuron` to assemble them. `inputs/long.provenance.json` travels
with the copies, recording how `long.jsonl` was generated (seed 20260920).

`max_model_len` defaults to 16416 — the 16k prompt plus 32 generated tokens. Set
it lower and the long cases are **skipped and recorded in `notes`**, so a short
run cannot silently look like a full one.

Cases marked `role: user` get the port's `chat_template.jinja` applied; `raw`
cases are plain continuations. A chat-template failure is recorded as a note and
the prompt is sent raw rather than aborting the run.

## What the judge is and is not asked

It grades **fluency and relevance only**. Factual errors, refusals and style are
explicitly *not* gibberish — looping tokens, mojibake, unexpected scripts,
punctuation-only output, and text unrelated to the prompt are.

A 32-token cap truncates mid-sentence constantly, so incomplete answers count as
reasonable. Where a case carries an `expected_substring` (the needle for the
long prompts, `Paris` for trivia), containing it is treated as strong evidence
of a good answer but is never required — 32 tokens may legitimately stop before
restating it, and the rubric forbids marking a case gibberish for omitting it.

## Output

`verifier_report.json` lands beside this file (override with `--report`) holding
every prompt excerpt, completion, token count, finish reason, the judge's
per-case verdicts and any execution traceback.

`--skip-judge` generates and prints without grading, for inspecting raw output.
Such a run reports `passed=0` by construction, since condition 2 was never
tested.
