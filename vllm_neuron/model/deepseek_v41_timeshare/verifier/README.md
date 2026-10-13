# Verifier — DeepSeek-V4.1-Flash (timeshare)

The onboarding guide's smoke test, widened to 4 inputs and given a judge.

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
| 1 | nothing raised | engine construction and all 4 generations complete without an exception |
| 2 | nothing gibberish | Sonnet 5 at medium effort reads every completion and finds each reasonable |

The judge **fails closed**. A reply that will not parse, a missing per-case
verdict, a nonzero exit, a timeout, or a missing `claude` CLI are all failures —
never a pass by default. A judge that cannot answer has not cleared the model.

## Inputs — 4 cases, 32 tokens each, greedy

| source | cases |
|---|---|
| TranSonic `short.jsonl` | `short-prose` (chat-templated) |
| TranSonic `long.jsonl` | `long-needle-8192-0` — needle-in-a-haystack at 8192 tokens |
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
count — and the default is `longest + 32 + 64` headroom. For the current set that is 9424.

This used to be `8192 + 32 = 8224`, taken from `long-needle-8192-0`'s `target_tokens`. That
field is the length the sample was generated to *target*; it encodes to **9328**, so the gate
rejected its own prompt with `VLLMValidationError` before a single token and the `verifier`
constraint was unsatisfiable regardless of the port. Nominal lengths are not lengths.

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

## On-device time — what it measures, and what it is not

Each phase also reports **on-device** TTFT and time-per-output-token, as
`##autohelix[...]` metrics and in `verifier_report.json` under `device_time`:

| metric | measured? | what it is |
|---|---|---|
| `device_ttft_ms_<phase>` | measured | one prefill forward, cold prefix cache |
| `device_step_ms_<phase>` | measured | one decode forward |
| `device_tpot_ms_<phase>` | derived | the step divided by the tokens a step emits |
| `device_total_ms_<phase>` | derived | what this phase's 32-token generate would cost on device at those rates |

`<phase>` is `long` (the needle) or `short` (the other three cases).

**Device time, not wall clock.** The target is a 16-device `trn2.48xlarge` that
keeps the weights resident and builds its graphs once; a wall-clock figure from
this 1-device host would mostly describe the host. The numbers come from the
`ExecutionInfo` table of an NRT device profile — one row per NEFF execution,
summed — so the host gaps *between* executions are excluded. On this port that
is most of the clock: a prefill window measured 62 s on device inside a 372 s
span.

Two honest caveats. An execution's span still contains whatever happens *inside*
it, and on this port that includes waiting for weights streamed from the host —
which is exactly what the 16-device target would not do, so these figures are an
upper bound for it. And the capture itself is not free; it is on for two forwards
of each window and costs wall-clock time, not output.

**The correctness run is untouched.** All 4 cases, 32 tokens, greedy, same
engine. The measurement adds a profile session *around* that generate — the
runtime records two forwards and stops itself — plus two short extra generates
*after* the judged completions have been collected. Nothing it does changes what
is scheduled or sampled, and nothing it does can fail the run: every step of it
is wrapped, and a failure lands in `notes` with the metrics left out.

**How a forward of one kind gets measured.** A window records the 2nd and 3rd
forwards of whatever it wraps (`delay_iterations=2`, `max_iterations=2`), which
is what keeps prefill and decode from landing in the same trace:

| window | wraps | records |
|---|---|---|
| `<phase>-generate` | the real 32-token generate | its forwards 2-3 — decode steps, since forward 1 is the first prompt's prefill |
| `<phase>-cold-prefill` | `"Hello."`, then the phase's longest prompt twice, cache reset before each | two cold prefills of that prompt |
| `<phase>-cold-decode` | the cache reset, then the same prompt for 3 tokens | two decode steps — that prompt's prefill is the skipped forward |

Mind the offset: `NeuronWorker.execute_model` calls the profiler's `step()`
*before* the forward, so `delay_iterations=D` records forwards D..D+M-1 rather
than the obvious reading. `D=1` records forward **1**.

Every window prefills cold, resetting the prefix cache first, because that is
what the run's own generate does. An earlier version let the decode window
reuse the warm cache the previous window left behind — cheaper, and a path the
gate never takes: on the 2008-token needle it killed the engine with
`RuntimeError: nrta status=1006`, an out-of-bounds execution. Reaching a
full-prefix-cache hit at that length looks like a live bug in the port, worth a
look on its own; it is not one the measurement needs to walk into.

The two built windows hold a known kind by construction. The third — around the
real generate — does not: this port forces `max_num_seqs` to 1 above one rank,
so a phase's prompts run one after another and "the first N forwards are the
prefills" is false for it. It is identified instead by fingerprint: the graph a
window spends most of its device time in, matched against the two known
windows', since a prefill and a decode forward run different graphs for the same
layer. A window matching neither is reported and not pooled.

Forwards are then pooled by kind, so TTFT is the mean of two cold prefills and
the step figure the mean of four decode forwards, two of them from the 32-token
run itself. Means rather than single samples because one forward is a noisy
sample here: two windows of identical prefill work measured 23.8 s and 29.5 s,
an execution's span moving with whatever else is streaming weights at the time.

Tokens-per-step is measured on the 32-token generate (`vllm:iteration_tokens_total`
counts forwards, `vllm:generation_tokens` counts tokens), because with
speculation a step emits `1 + accepted` tokens and one step's device time is not
one token's.

`--no-device-time` skips all of it. `--profile-dir` chooses where the profiles
go (default: a per-pid directory under the system temp dir, *not* the
repository — an NTFF is ~1 GB a forward). The raw NTFFs are deleted once read;
the parquet they were read from stays under `<profile-dir>/windows/<window>/ingest/`,
including a per-NEFF device-time breakdown that is the obvious place to start an
optimization pass. Reading it needs `pyarrow`; without it the metrics are
skipped rather than guessed, because the one figure available without it —
`Summary.total_time` — is the window's span, not its device time (491 s against
62 s on the window above).
