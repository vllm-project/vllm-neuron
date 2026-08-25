# DFlash measurement harness

Model-agnostic scripts used to produce the numbers in the DFlash model recipes.
Point them at a running server with `--model <served-model-name>`.

| script | what it does |
|---|---|
| `loadgen.py` | Fixed-concurrency load generator. Counts tokens from the usage record rather than stream chunks, because a speculative chunk carries a whole accepted block — counting chunks undercounts tokens and reports inter-chunk latency as if it were inter-token latency. Reports both. |
| `validate_greedy.py` | Captures greedy completions with exact token IDs. |
| `compare_greedy.py` | Diffs two captures token by token. |
| `scrape_metrics.py` | Pulls the engine's `spec_decode` acceptance counters. |
| `rope_check.py` | Checks a drafter's rotary embedding against the transformers implementation its weights were trained with. |

```bash
python loadgen.py --base-url http://localhost:8000 --model llama-3.1-8b \
    --concurrency 1 --requests 4 --max-tokens 192 --out /tmp/run.json
```

Use `validate_greedy.py` (which requests `return_token_ids`) rather than
`logprobs` for correctness checks: with Neuron on-device sampling the sampled
token is not guaranteed to appear in the returned top-k, and vLLM's logprobs
formatter raises `KeyError` on that.

Note that greedy token equality has a numerical noise floor across
configurations. Two runs of the same non-speculative baseline that differ only
in KV-cache settings already diverge on most 128-token prompts, while the same
server twice is bit-identical.
