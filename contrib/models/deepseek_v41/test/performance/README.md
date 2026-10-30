# Bench — DeepSeek-V4.1-Flash

Forward-inference speed, by the method the onboarding guide's "Benchmark and
tune performance" section prescribes: serve the model, then drive it with
`vllm bench serve` (`--dataset-name sonnet`: real English at a target length, see
"The matrix" below).

```bash
# from the vllm-neuron repo root
python contrib/models/deepseek_v41/test/performance/run_bench.py \
    --model <checkpoint-dir> \
    --dataset-path contrib/models/deepseek_v41/test/performance/sonnet.txt
```

`--dataset-path` is required unless the sonnet text sits at the default
`/mnt/nvme/datasets/sonnet.txt`; a copy ships in this folder. The batch-10 cells need
an engine that admits ten sequences (`DEEPSEEK_V41_MAX_NUM_SEQS=10`), so the
reported numbers come from two runs -- see "Reproducing the reported numbers".

## Reproducing the reported numbers

The current reference reports are `bench_report.opt9.json` (batch 1) and
`bench_report.opt9_ms10.json` (batch 10), measured 2026-10-30 on a trn3.48xlarge
with the DLAMI venv `aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0` and the
checkpoint at `/mnt/nvme/hf`. Run them one after the other, each with the devices to
itself:

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
cd <vllm-neuron repo root>

# batch 1: the four input lengths (the default matrix), one-sequence engines
DEEPSEEK_V41_SCHEME=tp64 \
python contrib/models/deepseek_v41/test/performance/run_bench.py --model /mnt/nvme/hf \
    --dataset-path contrib/models/deepseek_v41/test/performance/sonnet.txt \
    --report contrib/models/deepseek_v41/test/performance/bench_report.opt9.json \
    --bench-timeout 7200 --batch-sizes 1

# batch 10: the same input lengths on ten-sequence engines
DEEPSEEK_V41_SCHEME=tp64 DEEPSEEK_V41_MAX_NUM_SEQS=10 \
python contrib/models/deepseek_v41/test/performance/run_bench.py --model /mnt/nvme/hf \
    --dataset-path contrib/models/deepseek_v41/test/performance/sonnet.txt \
    --report contrib/models/deepseek_v41/test/performance/bench_report.opt9_ms10.json \
    --bench-timeout 7200 --input-lens 512,1024,2048,8192 --batch-sizes 10
```

Nothing else is set: every optimization in the reports is the default
(`DEEPSEEK_V41_SCHEME=tp64` is too, spelled out for clarity). The knobs, each `1` by
default unless noted, are `DEEPSEEK_V41_OPT_KERNELS` (`all`), `DEEPSEEK_V41_FUSED_OPT`
(`all`), `DEEPSEEK_V41_FUSE_DECODE` (`8` blocks a decode launch),
`DEEPSEEK_V41_FUSE_EMBED`, `DEEPSEEK_V41_FUSE_HEAD`, `DEEPSEEK_V41_DRAFT_BATCHED`,
`DEEPSEEK_V41_ONE_RPC_STEP` (synchronous scheduling only), `DEEPSEEK_V41_STEP_PROLOGUE`,
`DEEPSEEK_V41_ASYNC` and `DEEPSEEK_V41_FUSE_TAIL`; `src/README.md` ("A decode step is a
handful of launches", "A decode step's inputs are made on the device") says what each
does. The async engine needs this branch's plugin (the runner's `RUNNER_CAPABILITIES`,
the scheduler's holdbacks); on another the port serves synchronously. `--bench-timeout` only bounds a wedged run. Without
`DEEPSEEK_V41_MAX_NUM_SEQS=10` the port clamps the engine to one sequence and a
batch-10 cell queues its ten requests one at a time. A first run against a cold
compile cache spends tens of minutes compiling on the way up and in each cell's
warm-up; the measured figures are the same.
`DEEPSEEK_V41_HOST_PROF=1` adds a per-step host-time breakdown to the server logs
(`cells/server_*.log`) without changing the figures materially.

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

## The matrix — 8 cells

| | |
|---|---|
| input | 512, 1k, 2k, 8k tokens |
| output | 128 tokens |
| batch | 1, 10 |

Sonnet verse (`--dataset-name sonnet`, `--sonnet-prefix-len 0`): real English,
sampled whole lines up to the cell's input length, so the length is approximate and
the report records each cell's `mean_input_tokens`. Until 2026-10-25 the matrix was
random token ids (exact lengths), on which the drafter has nothing to predict --
11-15% acceptance against ~41% on prose -- so every throughput figure was a lower
bound (`run_bench._sonnet_flags` has the measurement). No two requests share a
prefix, so nothing can be served out of the prefix cache instead of computed — and
prefix caching is off on the server for the same reason.

`--ignore-eos`, so every request emits exactly 128 tokens. Without it a request
that stops early shortens its own TPOT sample and under-counts the token total.

Requests per cell is `max(8, 2 * batch)` — 8 at batch 1, 20 at batch 10, so
batch 10 runs two full rounds. Enough for a median to mean something, few
enough that the 8k x 10 cell does not dominate the run.

## Every cell is warmed first

**Not optional, and its absence is not a small error.** A cell's first requests
pay for that cell's own compiled buckets — the prefill shape for its input
length, the decode shape for its batch — and a graph of this model can take
minutes to build. At batch 10 the whole first round of ten requests pays it
concurrently, which is half the sample: an unwarmed cell does not report a slow
median, it reports a compile.

So each cell runs twice. The first run is discarded and generates only
`WARMUP_OUTPUT_LEN = 8` tokens, which is what makes it cheap — the decode shape
this model keys is
`(1 + k) * running_seqs` rows, a function of the batch and not of how many
tokens have been emitted, and with the compressed read pinned to the context
and the window fixed at 128 slots a longer generation reaches no shape a short
one does not. 8 tokens compiles the same graph set as 128.

It sends the **same number of requests** as the measurement, which looks like
waste and is not. With `--max-concurrency b` and `2b` requests the scheduler
admits a new request each time one finishes, so a *prefill runs concurrently
with `b - 1` decodes* — a shape that never occurs if the warm-up submits exactly
`b` requests, which all start together and drain together. Matching the count
makes the warm-up traverse the same scheduling trajectory, so every shape the
measurement will key already exists.

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

Acceptance also gets a measurement of its own, on text that is not the matrix's:
the matrix used to be random-token input, which very nearly switches speculation
off (a drafter predicts the next token, and there is nothing to predict in noise),
and the probe keeps a fixed prose reference across matrix changes.

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
step emits `1 + accepted` tokens, so acceptance is a first-order term here. On
`--dataset-name random` it very nearly switched itself off: random *token ids* give a
drafter nothing to predict, and DSpark's Markov/n-gram head leans on real language
statistics hardest of all. Measured on this port: **49% on the verifier's prose
against 16-18% on the random dataset**, where 49% is in line with the
single-device ancestor's 52% / 40%. That is why the matrix is sonnet verse now
(42-51% acceptance per cell), and why reports from before 2026-10-25 (random input)
are lower bounds that do not compare with later ones.

Medians, not means: one slow request — a cold bucket, a compile — otherwise
moves the figure more than the change being measured. p99 for TTFT and TPOT,
the per-stream rate (`1000 / TPOT`), the total token throughput and the
completed count all land in `bench_report.json` beside this file, with the raw
`vllm bench serve` output per cell under `cells/`.

Only figures that exist are emitted. A cell that failed prints `FAILED` in the
table and contributes no metric line, because a missing measurement must not
read as a zero.

## Reading the numbers

Check the `COLD` column before trusting a row: a warmed cell's figures are the
model's, a cold one's include a compile. `##bench[cells_ok=N]` says how many
cells produced a figure at all.

A run against a cold on-disk compile cache is slow whatever the warm-up does —
the cache is built on the way up and on each cell's first request — but it is
not *wrong*, because the warm-up absorbs it. The numbers from a cold-cache run
and a warm-cache run should agree.

`bench_report.json` is rewritten after every cell, so a run that dies keeps
what it measured.
