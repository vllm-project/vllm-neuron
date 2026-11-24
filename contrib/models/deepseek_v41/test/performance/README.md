# Bench — DeepSeek-V4.1-Flash

Forward-inference speed, by the method the onboarding guide's "Benchmark and
tune performance" section prescribes: serve the model, then drive it with
`vllm bench serve --dataset-name random`.

```bash
# from the vllm-neuron repo root
export DEEPSEEK_V41_MAX_NUM_SEQS=10
.venv/bin/python contrib/models/deepseek_v41/test/performance/run_bench.py \
    --model <checkpoint-dir>
```

**`DEEPSEEK_V41_MAX_NUM_SEQS` is not optional for the batch cells.** The model
clamps `max_num_seqs` to `config.MAX_NUM_SEQS` (1 by default), and reads it
from the environment when the package is imported. Left unset, every batch-10
cell is served by a one-sequence engine while the client holds ten requests in
flight: TTFT becomes queue wait, and aggregate throughput cannot exceed
single-stream. Measured at in512_bs10, 3.5 tok/s clamped against 30.2 uncapped,
from the same code. The bench warns when it sees a clamp, and the table marks
those cells `CAPPED->1`.

The script puts its own folder's parent (`contrib/models/deepseek_v41/test/`) on
`sys.path` and in `PYTHONPATH`, then imports the model package from the sibling
`src/`. That is also how the server subprocess it starts
(`-m performance._serve_main`) resolves both itself and the model.

This is the **speed** gate. `verifier/` is the **correctness** gate and reports
no timing at all; this reports no correctness. They serve the same checkpoint,
and that is checked rather than assumed — `architectures[0]` must be
`DeepseekV41ForCausalLM`, so a benchmark cannot quietly measure a smaller or
different model. A speed number against the wrong weights is worse than none.

> **Needs the Trainium devices to itself**, like the verifier. It starts real
> servers; check `neuron-ls` first.

## The matrix — 6 cells

| | |
|---|---|
| input | 512, 1k, 8k tokens |
| output | 128 tokens |
| batch | 1 and 10; **1 and 4 at 8k** |

8k runs at batch 4 because the KV cache is **replicated on every one of the 64
ranks**. MLA has one 512-dim latent KV head and each rank needs all of it, so a
rank pays for every sequence's whole context. Ten 8448-token contexts are
7.35 GiB a rank of a 24 GiB bank that already holds 13.92 GiB of weights, and
that cell failed to allocate both ways round. Four fit. `BATCH_OVERRIDES` holds
this, and an explicit `--batch-sizes` overrides it.

Random input, with `--random-range-ratio 0` so the lengths are exact: a cell
measures the shape it is named for, rather than whatever a corpus tokenized to.
No two requests share a prefix, so nothing can be served out of the prefix
cache instead of computed — and prefix caching is off on the server for the same
reason.

`--ignore-eos`, so every request emits exactly 128 tokens. Without it a request
that stops early shortens its own TPOT sample and under-counts the token total.

Requests per cell is `max(8, batch)` — 8 sequentially at batch 1, one full wave
at batch 10. **One wave, not two:** with `--max-concurrency b` and `2b`
requests, round two cannot start until round one finishes decoding, so half the
TTFT sample is queue wait and the throughput is divided by a serial structure.
Measured at in512_bs10 on the same engine: 20 prompts gave 5.01 tok/s, 10 gave
10.74.

## Every cell is warmed, then measured twice

**A warm disk cache is not a warm process.** A cell's first requests pay for
its own compiled buckets, and a graph of this model can take minutes to build;
even with every graph already on disk, dynamo re-tracing is per process. At
in1024_bs10 with a fully warm compile cache, the first pass measured 6.75 tok/s
and the passes after it 25.0. An unwarmed cell does not report a slow median,
it reports a compile.

So each cell runs three passes on one server:

1. **A discarded warm-up** at the measurement's own output length, with
   `--num-warmups` equal to the prompt count. The extra requests are what
   admit a prefill alongside running decodes — a state a single wave never
   reaches — so the measurement is never the first to meet it. It used to
   generate 8 tokens to stay cheap, and that was wrong: shapes are reached
   late in the drain (`Running: 4 reqs`), which an 8-token run never gets to.
2. **Measured pass 1**, kept as evidence that the warm-up finished.
3. **Measured pass 2**, the one reported. If the two disagree by more than 5%
   the cell did not converge and the bench says so; the spread lands in the
   report as `_pass_spread_pct`.

**Every measured pass is scored against its own slice of the server log.** Any
`Local cache miss` (a NEFF compile) or `Recompiling function` (a dynamo
re-trace) inside the window marks the pass `DIRTY`, which is not comparable
with a clean one. The bench exports `TORCH_LOGS=recompiles` for this, since
re-traces are otherwise invisible. Judge pollution by throughput and by the
window, not by median TPOT: the median decode step barely moves, while the
stalls are what the aggregate sees.

A failed warm-up is a note, not a dropped cell: the measurement is then a cold
one, it is marked `COLD` in the table, and `warmed: false` lands in the report.
Better a number with that caveat attached than no number.

## One server per cell

**The context is why, not the batch.** `model._COMPRESS_READ_CEILING` pins the
compressed-KV read to `max_model_len`, so a 512-token request served by an
8448-context engine reads 8448 compressed groups per layer on **38 of 40
layers** instead of the 672 its own context needs. That is 13x the work on every
forward, prefill and decode alike. Sharing one engine across input lengths does
not measure those lengths; it measures the largest one.

Measured here: the 512-token cell reported **TTFT 2318 ms at an 8448 context**
against ~1250 ms for a short prompt at 608. So each cell gets
`max_model_len = input + output + 128`, at the cost of one engine build per cell.

`--max-model-len` pins a single context for every cell instead. That is the
right flag when the question is "what does serving an 8k context cost a short
request" rather than "how fast is a 512-token request" — a real tuning question,
just a different one.

`max_num_seqs` is the batch, and it also has to be per cell: it decides the
compiled decode shapes, so a batch-1 figure measured on an engine built for 10
is not a batch-1 figure.

## What it reports

Per cell, as `##autohelix[<metric>_in<input>_bs<batch>=<value>]` for a loop and
as a table for a person:

| metric | what it is |
|---|---|
| `ttft_p50_ms` | time to first token, median |
| `tpot_p50_ms` | time per output token, median — decode only; the first token belongs to TTFT |
| `decode_tok_s` | output tokens per second, **aggregate** over the batch (`vllm bench serve`'s `output_throughput`) |

and one column that is not a speed figure but conditions every speed figure:

| column | what it is |
|---|---|
| `accept %` | DSpark speculative acceptance over the cell, read from that cell's own server log |

## The prose acceptance probe

The matrix is random-token input, and that very nearly switches speculation off:
a drafter predicts the next token, and there is nothing to predict in noise. So
acceptance gets its own measurement, on real text.

It is a **probe, not a cell**. After the widest-context cell's measurement, and
before that server is torn down, the bench sends the verifier's own cases to
`/v1/completions` and takes the delta in vLLM's `vllm:spec_decode_*` counters
around them. No extra engine build, and the counters are vLLM's own — the
scheduler accumulates them in `update_from_output`, in code the port cannot
edit, which is what makes them usable as a gate when `drafter.py` is editable.
It lands in `bench_report.json` as `report["prose"]`.

Prompts longer than the server's context are skipped, and which ones were sent
is recorded — a 2008-token needle has nothing to do with a 768-token engine.

`vllm bench serve --dataset-name custom` would have been less code and is not
available: `CustomDataset` reads its JSONL through pandas, which is absent from
this environment and resolves to a `PlaceholderModule` that raises on use.

**Read the acceptance column before comparing throughput to anything.** A decode
step emits `1 + accepted` tokens, so acceptance is a first-order term here — and
on `--dataset-name random` it very nearly switches itself off. Random input is
random *token ids*; a drafter predicts the next token, there is nothing to
predict in noise, and DSpark's Markov/n-gram head leans on real language
statistics hardest of all. Measured on this port: **49% on the verifier's prose
against 16-18% on the random dataset**, where 49% is in line with the
single-device ancestor's 52% / 40%.

That is not a fault and not a reason to abandon random input — exact lengths are
worth more to a benchmark than realistic token statistics. It does mean the
throughput here is a **lower bound** on what real traffic would see, and a
reader who cannot see the acceptance rate cannot know that.

Medians, not means: one slow request — a cold bucket, a compile — otherwise
moves the figure more than the change being measured. p99 for TTFT and TPOT,
the per-stream rate (`1000 / TPOT`), the total token throughput and the
completed count all land in `bench_report.json` beside this file, with the raw
`vllm bench serve` output per cell under `cells/`.

Only figures that exist are emitted. A cell that failed prints `FAILED` in the
table and contributes no metric line, because a missing measurement must not
read as a zero.

## Reading the numbers

Check for `COLD`, `CAPPED` and a `DIRTY` window before trusting a row: a
warmed, clean cell's figures are the model's; a cold or dirty one's include a
compile, and a capped one measured queueing. `##bench[cells_ok=N]` says how many
cells produced a figure at all.

`ttft_p50_ms` at `in8192_bs4` is bimodal and should not be compared on its own.
It is a p50 over 8 prompts at concurrency 4, so the TTFTs cluster at T, 2T, 3T
and 4T, and the interpolated median at index 3.5 sits on the T/2T boundary. It
lands near either 1.04 T or 1.52 T depending on how the second wave's arrivals
stagger, independent of the code. Read `in8192_bs1` for 8k prefill latency.

A run against a cold on-disk compile cache is slow whatever the warm-up does —
the cache is built on the way up and on each cell's first request — but it is
not *wrong*, because the warm-up absorbs it. The numbers from a cold-cache run
and a warm-cache run should agree.

`bench_report.json` is rewritten after every cell, so a run that dies keeps
what it measured.
