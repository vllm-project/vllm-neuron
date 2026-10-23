# Contrib serving benchmark

One benchmark for every contrib model, so that results are comparable across
models, configurations and releases. `bench_matrix.sh` runs `vllm bench serve`
against a server that is already up; `make_prompts.py` builds its real-text
prompts; `summarize.py` turns the result JSONs into tables.

The scripts are generalised from the Hy3 contrib port's `bench_matrix.sh` and
`summarize_bench.py` by Qingzhong Wang
([`qingzwang/vllm-neuron`](https://github.com/qingzwang/vllm-neuron), branch
`hy3-fp8-contrib`, `contrib/models/hy_v3/scripts/`).

## The standard matrix

| Setting | Default | Why |
|---|---|---|
| Prompts | Real conversations (ShareGPT), cut to exact lengths (`DATASET=text`) | See "Prompts" below |
| Input length (ISL) | 1024, 8192 | Short chat, long context |
| Output length (OSL) | 256, 1024 | Short answer, long generation |
| Concurrency | 1, 4, 8, 16, 32 | Latency through batched serving |
| Input jitter | +/-3% (`INPUT_RATIO=0.03`) | Identical lengths in a burst are not real traffic |
| Output jitter | none (`OUTPUT_RATIO=0`) | With `--ignore-eos`, every request decodes exactly OSL tokens, so TPOT is comparable |
| Requests per cell | max(8, 4 x concurrency), plus one warmup wave | Enough for stable medians |
| Concurrency cap | the server's real limit (`MAX_CONC`), added as a cell | See "Concurrency" below |
| Seeds | fixed per (ISL, concurrency) | Every server sees the same prompts |

Long-context models add `ISLS="1024 8192 32768"`; models that batch further
add concurrency 64.

## Prompts

`vllm bench serve --dataset-name random` builds each prompt from consecutive
vocabulary ids, `(offset + i) % vocab_size`. That is not text, and it changes what
is measured:

- a MoE router spreads it unlike real text, so expert load is off;
- a draft model accepts almost none of it, so speculative decoding looks useless.

`make_prompts.py` builds prompts of the same exact lengths from a ShareGPT-format
JSON (`ShareGPT_V3_unfiltered_cleaned_split.json`, as vLLM's and SGLang's
benchmarks use):

- **Concatenated, not repeated.** Each prompt joins whole conversations, starting
  at a random one, until it reaches its length, then cuts there. Different
  conversations are joined rather than one short prompt repeated, because
  repetition is easy for a draft model and would inflate acceptance.
- **Exact lengths.** Lengths are counted as the benchmark counts them,
  `len(tokenizer(prompt).input_ids)` with special tokens.
- **Chat template.** Prompts are sent as completions without one
  (`--skip-chat-template`).
- **Cached and shared.** `bench_matrix.sh` writes the prompts once per
  (ISL, concurrency) under `$OUT_DIR/prompts/`. Every OSL and every model sees the
  same conversations, each model cutting them with its own tokenizer.

`DATASET=random` keeps the old behaviour, for comparison with earlier numbers.

## Concurrency

The scheduler admits a new request only while every running one could still grow
to `max_model_len`. Its limit is `int(X)` from the server's startup log line:

```
Maximum concurrency for <max_model_len> tokens per request: X
```

`X` is usually a little below what the raw KV size suggests (e.g. 15.98 for a
budget sized for 16). A cell above `int(X)` queues the extra requests and
measures a lower concurrency than its label says.

- **Pass the limit.** Set `MAX_CONC` to min(`max_num_seqs`, `int(X)`). Cells above
  it are skipped, and `MAX_CONC` itself is added as a cell when it is not in
  `CONCS`; that is the server's maximum-throughput point. Record `X` in
  `EXTRA_METADATA`.
- **The limit depends on `max_model_len`, not on the ISL.** A server sized for
  8K inputs admits as few 1K requests as 8K ones. To measure short inputs at
  higher concurrency, run them on a second server sized for them. Each server is
  one more compile.
  - Short inputs: `ISLS=1024`, `max_model_len` 2304, a larger `max_num_seqs`.
  - Long inputs: `ISLS=8192`, `max_model_len` 9472.
- **When it is worth it.** One server sized for the longest cell is the default.
  Split the ISLs only where the KV cap at the long `max_model_len` is well below
  the concurrencies of interest.

## Sizing the server for the matrix

The Neuron plugin compiles one graph per bucket, so the server's limits decide
which cells can run:

- **`max_model_len`**: the longest cell, ISL x 1.03 + OSL. 8192 / 1024 needs
  9462; round up to a block multiple, e.g. 9472.
  - Do not over-size it. The scheduler admits a new request only if every
    running one could grow to `max_model_len`, so a larger value lowers the
    concurrency the server will actually reach.
- **`max_num_seqs`** and **`num_seqs_buckets`**: the decode batch sizes. To run
  the concurrencies above, use `[1, 4, 8, 16, 32]` or a subset.
  - Every bucket is another set of decode graphs to compile.
  - On TP=64 models, cap the compiler with `VLLM_NEURON_PARALLEL_COMPILE_WORKERS=2`.
    The native route compiles per rank, and the default of 8 jobs per rank can
    run a 2 TB host out of memory.
- **`num_batched_tokens_buckets`** (prefill): cover the ISLs. Chunked prefill
  splits longer prompts into chunks of the largest bucket.
- **`decode_context_length_buckets`**: at 8K+ inputs, give decode context
  buckets, e.g. `[2048, 4096, 8192, 16384]`. Otherwise every decode step
  attends over `max_model_len`.
- **Large models:** set `NEURON_RT_EXEC_TIMEOUT=1200`. The first executions of
  a 64-rank graph can exceed the runtime's 30 s default, and that shows up as a
  collective hang.

Pass `MAX_CONC` (see "Concurrency") and `MAX_MODEL_LEN` to `bench_matrix.sh`
and it skips the cells the server cannot run, instead of queueing them.

## Running

```bash
# 1. Start the model's server (see its README), e.g. on port 8000.
# 2. Run the matrix against it (MAX_CONC: see "Concurrency"):
SERVED_NAME=hy3 TOKENIZER=tencent/Hy3-FP8 LABEL=hy3-tp64-ep64 \
  CORPUS=ShareGPT_V3_unfiltered_cleaned_split.json \
  MAX_CONC=32 MAX_MODEL_LEN=9472 \
  EXTRA_METADATA="tp=64 ep=64 sdk=2.33 backend=neuron_native" \
  bash contrib/benchmark/bench_matrix.sh
```

Results go to `results/bench/$LABEL/in<ISL>_out<OSL>_c<C>.json`. Every JSON
carries the label, the cell and `EXTRA_METADATA`.

## Comparing

```bash
# one run
python contrib/benchmark/summarize.py results/bench/hy3-tp64-ep64

# before -> after on one model
python contrib/benchmark/summarize.py results/bench/new --baseline results/bench/old

# several models side by side, one metric
python contrib/benchmark/summarize.py results/bench/hy3-tp64-ep64 results/bench/mimo-v2.6-flash \
  --pivot median_tpot_ms

# everything as CSV
python contrib/benchmark/summarize.py results/bench/* --csv all.csv
```

## Reading the numbers

- **TTFT at concurrency > 1 includes queueing.** The plugin runs one prefill
  per step and pauses running decodes while a prefill is waiting. A burst of C
  requests therefore prefills one after another: the i-th request's TTFT is
  about i prefills, and the requests that already have their first token wait
  for the rest before decoding. Compare the median and P99 TTFT rather than the
  mean.
- **TPOT** is over the output tokens after the first; **ITL** is per token.
  With the default exact OSL, they differ only by how decode steps are spread.
