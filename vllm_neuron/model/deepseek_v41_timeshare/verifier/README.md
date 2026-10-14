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

Each phase reports on-device TTFT and time-per-output-token, as `##autohelix[...]`
metrics and under `device_time` in the report:

| metric | measured? | what it is |
|---|---|---|
| `device_ttft_ms_<phase>` | measured × segments | the mean steady prefill pass, times the segments the prompt needs |
| `device_step_ms_<phase>` | measured | the mean decode pass |
| `device_tpot_ms_<phase>` | derived | the step divided by the tokens a step emits |
| `device_total_ms_<phase>` | derived | a 32-token generate at those rates |

`<phase>` is `long` (the needle) or `short` (the other three cases).

### Device time is device-busy, not execution span

A NEFF's *execution span* is almost all host waiting on this port, and the two
reasons were both measured here:

- **A per-layer all-reduce sits armed waiting for its three peer ranks.** Its
  span is ~20 s; `CcOp` says the transfer is **30 ms**. The ranks arrive apart
  because each is reading expert weights off NVMe at ~155 MB/s.
- **The weights arrive *inside* the execution.** This port streams them per
  layer per forward, so a span contains the host→device copy that stages them —
  an `nrt` memcpy outside the NEFF, which is why the device profile shows it as
  idle rather than as work.

Neither is device work, and neither exists on the 16-device target where the
weights are resident. So the metric is the union of the three traces that *are*
device work:

```
device time = union( engine ActiveTime , device DmaPacket , CcOp transfer )
```

The union, not the sum: the engines and the DMA queues run in parallel, and on
one window the per-source sum was 2.5 s where the union was 1.46 s. **DMA is
included** — `DmaPacket` is, in the schema's own words, a transfer *within a
Neuron Device*, i.e. the HBM→SBUF staging every kernel needs and the target
still pays. It is 1.03 s of that 1.46 s, so an earlier cut of this file which
dropped it under-counted by a third. Measured on one real window: **1 460 ms of
device time inside 29 190 ms of execution spans**, a factor of 20. The window's
`span_sum_ms` is printed and recorded beside every figure so that gap never
hides again.

The exclusion is one-sided in a known way: idle *inside* a pass with no engine,
no DMA and no collective active is dropped along with the host waits, because
the copy the device is waiting for is absent from the device profile — a kernel
bubble and a weight wait are the same observation here. So a badly written kernel
is forgiven its gaps. `utilisation` (engine ÷ device) is printed next to every
pass to keep that visible: 76-85% on decode, 64-74% on prefill as measured.

An earlier cut of this file gated on the spans, which meant it gated on NVMe
bandwidth: a loop optimising it won 30-55% by removing disk reads while
expert-GEMM time stayed identical to the microsecond, and that win does not
exist on the target machine.

### The first prefill pass of a window is a warm-up

It is excluded from the pool, and reported separately as `prefill_warmup_ms`.
Measured in one window, same 512 rows, same work: **1 097.9 ms then 291.2 ms**.
The 3.8x is *engine* time (809.6 against 189.6 ms), not DMA, so it is the first
forward of a fresh process rather than a weight wait — a cost the target pays at
startup, not per request. Averaging the two reported a TTFT 2.4x the steady one.

Decode passes get no such treatment. First-then-second they measured 407/414 ms
and 483/392 ms: noise with no warm-up direction, and dropping one of two samples
would leave a gated metric resting on a single pass.

### Passes, and which is prefill

Everything is priced per **pass** — one sweep of the model's layers. A window is
split at every `num_layers`-th execution of the per-layer graph (the NEFF that
runs a multiple of 40 times and costs the most).

A pass's kind comes **off the NEFF**: the `hidden_states` width it declares is
512 for a KV segment and `1 + k` for a verify step. That is a fact from the
compiled graph, not an inference from where the forward sat in the run — and the
inference is what the previous cut got wrong. A 2008-token prompt prefills as
**four** 512-row passes, so its second and third forwards are still prefill, and
every long-phase "decode" figure it reported was a prefill segment.

`num_layers` is `text_config.num_hidden_layers` read from the checkpoint, not a
constant — 40 here, and **the trunk only**: the three DSpark/MTP stages
(`num_nextn_predict_layers`, tapping layers 37/38/39) are not counted. That is
correct for the split twice over: a graph running 3 times a forward can never be
the marker, which must run a multiple of `num_layers` times, and the MTP
executions land *inside* the pass, so the draft's cost is counted in the step
that uses it.

One fragility to know about: if kernel fusion ever leaves no NEFF running a
multiple of `num_layers` times, the marker search fails and the whole window is
priced as one pass — a note, but a silently mis-scaled metric. The fix is to
stride on the window's known forward count instead; see `_pass_marker`.

### The windows

A window records forwards `delay .. delay + max - 1` of whatever it wraps. Mind
the offset: `NeuronWorker.execute_model` calls the profiler's `step()` *before*
the forward, so `delay=1` records forward **1**.

| window | engine | records |
|---|---|---|
| `<phase>-cold-prefill` | the phase's own | `delay=1, max=3` — the prompt three times with the cache reset before each, so three cold prefill segments: one warm-up and two samples. A long prompt fills the window from its first generate alone, since 2008 tokens prefill as four forwards |
| `<phase>-cold-decode` | a second, built after the phase's is gone | `delay = segments + 1, max=2` — the first forward that cannot be a prefill segment, so two decode passes, both kept |

The decode window needs its own engine because `delay_iterations` is fixed when
an engine is built. The cheaper alternative — reaching a decode forward early by
leaving the prefix cache warm — is the path that killed the engine with
`RuntimeError: nrta status=1006` on the needle, and one the run itself never
takes.

TTFT is one prefill pass × `ceil(prompt / rows)` rather than a captured whole
prefill, because four passes of a long prompt exceed the ~6 700 executions at
which the runtime starts dropping notifications.

Tokens-per-step comes from the 32-token generate's own counters
(`vllm:iteration_tokens_total` for forwards, completion lengths for tokens),
because with speculation a step emits `1 + accepted` and one step's device time
is not one token's.

### The correctness run is untouched

All 4 cases, 32 tokens, greedy, same engine arguments. The measurement adds a
profile session around that generate plus short extra generates *after* the
judged completions are collected, and nothing it does can fail the run: every
step is wrapped and a failure lands in `notes` with the metrics left out.

`--no-device-time` skips all of it. `--profile-dir` chooses where profiles go
(default: a per-pid directory under the system temp dir — an NTFF is ~1 GB a
forward, and the instruction trace this now keeps makes it ~2 GB). The ingest
costs ~92 s and ~434 MB of parquet a window, and needs ~20 GiB of host memory —
which is why it runs only after an engine is torn down: the same ingest OOM'd at
22 GiB while four ranks held 85 GiB of the 115 GiB `CommitLimit`. Reading it
needs `pyarrow`; without it the metrics are skipped rather than guessed.
