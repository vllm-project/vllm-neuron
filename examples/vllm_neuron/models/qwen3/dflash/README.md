# Qwen3 4B with DFlash on Trn2

Serves `Qwen/Qwen3-4B` with the
[`z-lab/Qwen3-4B-DFlash-b16`](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16)
block-diffusion drafter, whose proposal block is 16 (15 speculative tokens).

## Measured on one Trn2 chip (trn2.3xlarge, TP4, BF16)

Matched runs, 192-token generations, identical flags apart from
`--speculative-config`:

| | baseline | DFlash | |
|---|---|---|---|
| Output tok/s, concurrency 1 | 40.5 | **71.0** | **1.75x** |
| Output tok/s, concurrency 4 | 148.6 | 51.3 | **0.35x** |
| Time per output token (conc 1) | 24.7 ms | **14.0 ms** | |
| Mean accepted length | 1.0 | 2.1 of 16 | |
| Per-token acceptance | — | 7.3% | |

**Use this at low concurrency only.** The block of 16 is far larger than the
drafter sustains here: at 7.3% per-token acceptance the mean accepted length is
only 2.1, so most of the block is verified and thrown away. That is affordable
at concurrency 1, where the target is latency-bound and verifying 16 tokens
costs little more than verifying one. At concurrency 4 the target verifies
4 x 16 = 64 tokens per step, and the run is nearly 3x *slower* than the
baseline.

For comparison, the Llama 3.1 8B drafter (block 10) reaches 25% per-token
acceptance and stays ahead at both concurrencies — see
[`../../llama3/dflash/`](../../llama3/dflash/).

`num_speculative_tokens` is fixed at `block_size - 1` by training, so the block
cannot currently be shortened to trade acceptance for verification cost.

## Supported configuration

- Target: `Qwen/Qwen3-4B` — note this checkpoint **ties** its LM head to the
  embedding, which the Neuron Qwen3 implementation now handles
- Drafter: `z-lab/Qwen3-4B-DFlash-b16`
- Hardware: Trn2, TP4 on a single chip (`trn2.3xlarge`)
- Precision: BF16 target, draft, and KV cache
- Speculative tokens: **15**
- Scheduling: synchronous; prefix caching, chunked prefill and disaggregated
  inference disabled

## Start the server

```bash
./run_tp4.sh
```

The measurement harness lives in
[`../../../spec_decode/dflash_bench/`](../../../spec_decode/dflash_bench/)
and is model-agnostic — pass `--model qwen3-4b`.
