# Verifier — DeepSeek-V4.1-Flash

The onboarding guide's smoke test, widened to 4 inputs.

**Correctness only — it generates and checks, and times nothing.** An earlier cut
also measured on-device TTFT and time-per-output-token, captured two extra NRT
profile windows a phase and printed `##autohelix[...]` metrics; all of it, and
`device_time.py` with it, has been removed. So a run is one engine per phase and
one `generate` each, and the only question it answers is whether the tokens are
right. A performance number comes from a harness built for that, not from here.

The integration test at `test/integration/test_model.py` drives this harness and
adds deterministic assertions on top, so `pytest` is the normal entry point —
see the [model README](../../README.md#accuracy-integration). To run it
directly, the model package has to be imported first (that is what registers the
architecture), which `test/model_package.py` does:

```bash
# from the vllm-neuron repo root
PYTHONPATH=contrib/models/deepseek_v41/test \
.venv/bin/python -c "import deepseek_v41, sys; \
from deepseek_v41.verifier.run_verifier import main; \
sys.exit(main(sys.argv[1:]))" \
    --model <checkpoint-dir>
```

`--model` is **required and has no default**. The checkpoint lives outside this
repository, and the verifier deliberately holds no path to anything outside
`vllm-neuron` — so the location is yours to pass, not ours to assume. Use the
local copy rather than re-downloading; the [model
README](../../README.md#example-checkpoints) says
where it is.

> **This one runs the model, so it needs a free Trainium device.** Every other
> part of this port is CPU-only. Check `neuron-ls` before starting: the port
> serves at TP=64, so it needs every logical core on the instance.

## Pass conditions

Both are required; `##verifier[passed=1]` and exit 0 only when both hold.

| | condition | how |
|---|---|---|
| 1 | nothing raised | engine construction and all 4 generations complete without an exception |
| 2 | the planted answers come back | every case that declares an `expected_substring` contains it |

Condition 2 **fails closed** in both directions. A missing substring fails, and
so does a prompt set that declares no expectations at all — a verifier that
checks nothing must not report success.

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
full pass over 475 GiB of weights. The unselected rows stay in
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

## What is checked, and what is only printed

The verdict rests on the cases that can be checked **exactly**: those carrying an
`expected_substring` — the access code for the needle prompt, `Paris` for trivia.

`long-needle-2048-0` is the case that carries the weight. It plants an access
code 2008 tokens deep and the completion has to recall it, which broken
long-context attention, a mis-sharded weight or a wrong RoPE cannot produce by
luck.

`short-prose` has no single right answer, so nothing is asserted about it — its
completion is **printed** with the others and left for a reader. That is a
deliberate trade. An earlier cut asked an LLM judge whether each completion was
reasonable or gibberish, which bought a weaker signal for an external
dependency: the judgement was not reproducible between runs, a judge that could
not answer blocked the run, and it needed a `claude` CLI on `PATH` that nobody
running these tests has a reason to have.

Note that a 32-token cap truncates mid-sentence constantly, and this is a
reasoning model in thinking mode — so a completion is often terse chain-of-thought
that quotes the prompt back. That is normal output, not a fault.

## Output

`verifier_report.json` lands beside this file (override with `--report`) holding
every prompt excerpt, completion, token count, finish reason, the per-case
substring results under `checks`, and any execution traceback.

Every completion is printed to stdout before the checks run, so a failing run
shows what the model actually produced rather than only that it was wrong.

