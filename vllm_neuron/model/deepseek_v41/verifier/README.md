# Verifier — DeepSeek-V4.1-Flash (timeshare)

The onboarding guide's smoke test, widened to 4 inputs and given a judge.

**Correctness only — it generates and judges, and times nothing.** An earlier cut
also measured on-device TTFT and time-per-output-token, captured two extra NRT
profile windows a phase and printed `##autohelix[...]` metrics; all of it, and
`device_time.py` with it, has been removed. So a run is one engine per phase and
one `generate` each, and the only question it answers is whether the tokens are
right. A performance number comes from a harness built for that, not from here.

```bash
# from the vllm-neuron repo root
.venv/bin/python -m vllm_neuron.model.deepseek_v41.verifier.run_verifier \
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
| 1 | nothing raised | engine construction and all 4 generations complete without an exception |
| 2 | nothing gibberish | Sonnet 5 at medium effort reads every completion and finds each reasonable |

The judge **fails closed**. A reply that will not parse, a missing per-case
verdict, a nonzero exit, a timeout, or a missing `claude` CLI are all failures —
never a pass by default. A judge that cannot answer has not cleared the model.

## Inputs — 4 cases, 32 tokens each, greedy

| source | cases |
|---|---|
| TranSonic `short.jsonl` | `short-prose` (chat-templated) |
| TranSonic `long.jsonl` | `long-needle-2048-0` — needle-in-a-haystack, 2008 tokens as the checkpoint's tokenizer encodes it |
| onboarding guide | `smoke-test` — `"Hello, what is your name?"`, chat-templated (the guide's raw `"Hello, my name is"` made a reasoning-tuned model degenerate into a repeat loop) |
| trivia | `trivia-capital-france` — `"What is the capital of France?"` |

The TranSonic sets are **copied into `inputs/`**, byte-for-byte from their
source, and read via a path anchored to this package. There is no
`--inputs-dir` and no environment
override: every run uses the same 4 prompts, and the verifier needs nothing
outside `vllm-neuron` to assemble them. Which rows of the copied sets are used is
`prompts.SELECTED_IDS` — one short and one long, because every prompt costs a
full pass over 475 GiB of timeshared weights. The unselected rows stay in
`inputs/`, so widening the set is a one-line change and needs no re-copying. `inputs/long.provenance.json` travels
with the copies, recording how `long.jsonl` was generated (seed 20260920).

`max_model_len` is **measured, not assumed**: `prompts.render` encodes every selected
prompt with the checkpoint's own tokenizer — chat template applied, since that changes the
count — and the default is `longest + 32 + 64` headroom. For the current set that is **2104**
(2008 + 32 + 64) for the long phase and 608 for the short one, the latter floored so a KV
segment fits under it.

Never take the length off the row's name. A needle row is named for the token count it was
generated to *target*, and `target_tokens` is that nominal figure: `long-needle-2048-0`
encodes to 2008, and the 8192 row encodes to **9328**. An earlier cut sized the context as
`target_tokens + 32`, so the gate rejected its own prompt with `VLLMValidationError` before a
single token and the `verifier` constraint was unsatisfiable regardless of the port.

Pass `--max-model-len` to override. A value too small **skips** the cases that do not fit and
records each in `notes`, against measured lengths, so a short run cannot look like a full one.

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

