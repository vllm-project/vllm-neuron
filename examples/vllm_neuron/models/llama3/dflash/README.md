# Llama 3.1 8B with DFlash on Trn2

Serves `meta-llama/Llama-3.1-8B-Instruct` with the
[`z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat`](https://huggingface.co/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat)
block-diffusion drafter. DFlash proposes all nine speculative tokens in one
non-causal pass, so the draft cost per step is a single forward, not nine.

## Measured on one Trn2 chip (trn2.3xlarge, TP4, BF16)

Matched runs, 192-token generations, identical flags apart from
`--speculative-config`:

| | baseline | DFlash | |
|---|---|---|---|
| Output tok/s, concurrency 1 | 39.0 | **121.5** | **3.11x** |
| Output tok/s, concurrency 4 | 151.2 | **184.2** | **1.22x** |
| Time per output token (conc 1) | 25.6 ms | **8.1 ms** | |
| TTFT (conc 1) | 35.3 ms | 42.4 ms | +7 ms |
| Mean accepted length | 1.0 | **3.2** of 10 | |

**Why this model does well.** Llama 3.1 is dense, so verifying a block costs
almost the same as verifying one token: measured target step cost is 25.6 ms
for 1 token and 26.2 ms for 4, i.e. ~0.2 ms per additional token. Break-even
therefore sits near an accepted length of 1.2, and the drafter clears it
comfortably.

Contrast gpt-oss-20b, where top-4-of-32 expert routing makes each extra
verified token pull in its own experts (19.0 / 37.5 / 69.8 ms for 1 / 4 / 8
tokens, ~7 ms per token). There break-even needs an accepted length above ~3.5,
and DFlash does not currently reach it. See
[`../../gpt_oss/20b/mxfp4/dflash/`](../../gpt_oss/20b/mxfp4/dflash/).

The gain shrinks with concurrency because batching already amortises the
target's weight reads, which is the same thing speculation is buying.

## Supported configuration

- Target: `meta-llama/Llama-3.1-8B-Instruct` (ungated mirrors work too)
- Drafter: `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat`
- Hardware: Trn2. TP4 on a single chip (`trn2.3xlarge`, LNC=2 -> 4 logical cores)
- Precision: BF16 target, draft, and KV cache
- Speculative tokens: **9** — the checkpoint's proposal block is 10 and that is
  fixed by training, so this is not a tuning knob
- Scheduling: synchronous; prefix caching, chunked prefill and disaggregated
  inference disabled

## Start the server

```bash
./run_tp4.sh
```

Cold compilation takes several minutes. Wait for application startup to
complete before sending requests.

## Validate

Compare greedy output against the same command without `--speculative-config`:

```bash
curl http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model": "meta-llama/Llama-3.1-8B-Instruct",
         "prompt": "The capital of Australia is",
         "temperature": 0, "max_tokens": 24, "return_token_ids": true}'
```

Use `return_token_ids` rather than `logprobs`: with Neuron on-device sampling
the sampled token is not guaranteed to appear in the returned top-k, and vLLM's
logprobs formatter raises `KeyError` on that.

The measurement harness used for the table above lives in
[`../../../spec_decode/dflash_bench/`](../../../spec_decode/dflash_bench/)
and is model-agnostic — pass `--model llama-3.1-8b`.
