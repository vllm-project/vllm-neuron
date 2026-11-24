# Contrib Model: Ouro 1.4B (Looped Language Model, Text)

[Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B) is a looped
(Universal Transformer) decoder-only language model. One stack of 24 decoder
layers (hidden size 2048, 16 attention heads of size 128 with 16 KV heads,
SwiGLU MLP of size 5632, default RoPE) runs 4 times with shared weights. Each
decoder layer has two extra "sandwich" RMSNorms, and an early-exit gate after
each loop step selects the hidden state that the language-model head uses.
Each loop step keeps its own KV cache, so the model has 96 KV cache layers.

This folder serves it with vLLM on AWS Trainium2 through the `vllm-neuron`
plugin, from the BF16 checkpoint with tensor parallelism. See
[src/README.md](src/README.md) for the architecture, the KV cache layout and
the kernels.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix and the
> validation report below record exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`), then install this folder:

```bash
pip install -e contrib/models/ouro
```

The install adds a `vllm.general_plugins` entry point (`src/plugin.py`). vLLM
calls it in every process it starts (the API server, the engine core and each
tensor-parallel worker), and it registers two things:

- **The `"ouro"` config type.** transformers does not include Ouro, and the
  checkpoint's own `configuration_ouro.py` loads only with
  `--trust-remote-code`. `src/hf_config.py` gives a local configuration class
  and registers it with `AutoConfig`, so vLLM parses the checkpoint config
  without remote code. Do not pass `--trust-remote-code`.
- **The `OuroForCausalLM` architecture.** vLLM 0.29 no longer ships an Ouro
  implementation and rejects the architecture without this registration.

`pip uninstall vllm-neuron-contrib-ouro` removes both registrations.

Three settings are specific to this model:

- **BF16 only.** The factory rejects other checkpoint dtypes.
- **Use a TP degree that divides 16.** The 16 heads, the vocabulary of 49152
  and the MLP size of 5632 must divide by the TP degree. The device tests
  use TP 8.
- **KV memory is 4x a 24-layer model.** Each of the 4 loop steps writes its
  own KV cache. At TP 8 one token takes 96 KiB of KV cache on each rank, so
  4 sequences of 9216 tokens take 3.4 GiB on each rank.

### Serve with the OpenAI-compatible server

These are the flags of the tested configuration: a maximum model length of
9216 tokens (a prompt of up to 8192 tokens plus 1024 generated tokens) and up
to 4 sequences. `--host 127.0.0.1` keeps the server local to the host; it
listens on port 8000.

```bash
vllm serve ByteDance/Ouro-1.4B \
    --host 127.0.0.1 \
    --revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 9216 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{"neuron_config": {"on_device_sampling_config": {"all_greedy": true}, "num_batched_tokens_buckets": [1024], "kv_segment_size_buckets": [1024], "num_seqs_buckets": [4]}}'
```

This matches the batch-4 case of the integration test, so with a shared
Neuron compile cache the server reuses those graphs. The server prefills a
long prompt in chunks of 1024 tokens (`--max-num-batched-tokens`), one
request at a time. With `kv_segment_size_buckets` set, each chunk runs the
segmented attention kernel, and each chunk after the first attends to the KV
cache that the earlier chunks wrote. The three bucket lists compile one
prefill graph and one decode graph instead of a power-of-two ladder of each.
`all_greedy` makes every request greedy; remove it to serve sampled requests,
a path that was not validated for this model. `--no-async-scheduling` matches
the integration test, which needs it because the async path returns before
it computes logprobs.

Then send a request:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "ByteDance/Ouro-1.4B",
        "prompt": "The color of the sky is",
        "max_tokens": 32,
        "temperature": 0
    }'
```

### Offline inference with the Python API

```python
from vllm import LLM, SamplingParams

# With `pip install -e contrib/models/ouro` the config type and the
# architecture are registered in every vLLM process; no registration call is
# needed here. These are the engine settings of the batch-4 case of the
# integration test.
llm = LLM(
    model="ByteDance/Ouro-1.4B",
    revision="574fa66cb8bf5abdc979642d01cf2b79b16bfab1",
    tensor_parallel_size=8,
    dtype="bfloat16",
    max_model_len=9216,
    max_num_batched_tokens=1024,
    max_num_seqs=4,
    enable_prefix_caching=False,
    async_scheduling=False,
    additional_config={
        "neuron_config": {
            "on_device_sampling_config": {"all_greedy": True},
            "num_batched_tokens_buckets": [1024],
            "kv_segment_size_buckets": [1024],
            "num_seqs_buckets": [4],
        }
    },
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The color of the sky is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

Without the install, copy `src/` into your application and import it before
you build `LLM(...)` (option 1 in [contrib/README.md](../../README.md)). The
import registers the `"ouro"` config type; then register the factory:
`ModelRegistry.register_model("OuroForCausalLM", OuroForCausalLM)`. The engine
core and the TP workers inherit both registrations only under the default
`fork` start method, so leave `VLLM_WORKER_MULTIPROC_METHOD` unset.

To iterate without Neuron hardware, set `VLLM_NEURON_CPU_MODE=1` and pass
`enforce_eager=True`; `test/unit/test_tiny_e2e.py` shows a complete CPU-mode
configuration.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0, vLLM 0.29.0) | 2.32 and earlier |
| ---------------------- | -------------------------------------------- | ---------------- |
| Trn2 (`trn2.48xlarge`) | TP 8, maximum model length 9216: see the [validation report](#validation-report) | Not tested |
| Trn3                   | Not tested | Not tested |

| Feature | Status |
|---|---|
| BF16 checkpoint | Supported (other dtypes are rejected by the factory) |
| Tensor parallelism | TP 8 in the device integration test; TP 2 and 4 in the CPU tests; other TP degrees not validated on device |
| Context length | Maximum model length 9216: the device integration test (prompts of 8000-8192 tokens, 16 generated tokens) and the long-prompt serve request of the validation report (8190 prompt tokens, 1024 generated tokens). Longer contexts (the checkpoint allows 65536) are not validated |
| Batch size | 1 and 4 in the device integration test; larger batches not validated |
| Chunked (segmented) prefill | 1024-token chunks with 1024-token KV segments in the device integration test; segmented prefill over a cached prefix also in the CPU component tests. Other chunk and segment sizes not validated |
| Early exit | Threshold exit (`early_exit_threshold`, the checkpoint uses 1.0). `early_exit_step` is rejected by the factory |
| Prefix caching | Not validated (every test disables it) |
| Sampled (non-greedy) requests | Not validated (every test decodes greedily) |
| Data parallelism | Not validated |
| RoPE scaling, sliding-window attention | Not supported (rejected by the factory) |
| Speculative decoding, quantized checkpoints | Not tested |
| Performance | Not measured |

## Example Checkpoints

- <https://huggingface.co/ByteDance/Ouro-1.4B> — the tests pin revision
  `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` and check its identity
  (`config.json` fields and the `model.safetensors` size). Set
  `OURO_CHECKPOINT_PATH` to a local copy to skip the download.

Other Ouro checkpoints with the same layout (BF16, `model_type: "ouro"`,
full attention, default RoPE, threshold exit) should load, but none was run.

## Testing

All test directories reach the implementation through `test/_local_src.py`,
which loads `src/` by file location under the unique module name
`contrib_ouro_src`, so the tests do not need the `pip install`.
`test/conftest.py` puts `test/` on `sys.path` (for `_local_src.py`, the
shared helpers `ouro_test_utils.py` and `ouro_component_checks.py`, and the
independent reference `ouro_reference.py`) and shuts down the plugin's
`MPExecutor` worker pool at the end of the session.

The reference (`test/ouro_reference.py`) is an independent torch
implementation of the checkpoint's `modeling_ouro.py` rounding order that
reads the safetensors weights directly. transformers does not include Ouro,
and the checkpoint's remote code targets an older transformers, so every
test compares against this reference.

### Unit tests (CPU mode)

No Neuron hardware is needed, and only `config.json` (1.5 kB) is downloaded.
The component and end-to-end tests build a tiny random-weight checkpoint with
the Ouro layout (2 layers, 4 loop steps, sandwich norms, early-exit gate) in
a temporary directory:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/ouro/test/unit/ -v -s --timeout=3600
```

| File | Covers |
|---|---|
| `test_registration_config.py` | `register()` registers the `"ouro"` config type and the architecture; the installed package layout resolves and passes vLLM's model inspection; the pinned `config.json` parses without remote code (transformers and vLLM); factory guards; vLLM model protocol; KV layer names |
| `test_components_cpu.py` | Tiny checkpoint at TP 2: decoder layer prefill, flash and segmented attention prefill, MLP prefill and decode, three-way against the reference with the KV cache writes and untouched caches of the other loop steps; decode attention at TP 4; early-exit selection bitwise equal to the reference; KV spec, binding and full weight load |
| `test_tiny_e2e.py` | The full vLLM pipeline at TP 4: config parse, load, prefill, 8 decode steps, on-device greedy sampling, full-vocabulary raw logits against the reference with teacher forcing |
| `test_checkpoint_cpu.py` | Tests with the real weights (skipped unless `OURO_CHECKPOINT_PATH` is set): decoder layer prefill at TP 2, early-exit selection with the real gate, KV spec at TP 8, full weight load at TP 2 |

The decode cases run at TP 4 in CPU mode because the torch fallback of the
decode attention kernel supports one KV head per rank; the device kernel has
no such limit. `test_tiny_e2e.py` needs at least 240 MiB free in `/dev/shm`
(vLLM's TP worker queues) and skips otherwise; run the container with a
larger `--shm-size` or with `--ipc=host`.

### Module tests (Neuron)

`test/module/test_components_device.py` compiles one real module at a time at
TP 8 and compares it with the reference: flash prefill (1024 tokens),
segmented prefill (2 x 512 tokens), decode attention (batch 4, context 96),
and the MLP in prefill and decode. The validation report does not cover
these tests. They need 8 NeuronCores:

```bash
OURO_CHECKPOINT_PATH=/path/to/Ouro-1.4B \
pytest contrib/models/ouro/test/module/test_components_device.py -v -s --timeout=3600
```

### Integration test (Neuron)

Serves the model through vLLM at TP 8 on one `trn2.48xlarge` and validates
its logits against the reference:

- batch 1 and batch 4, at a maximum model length of 9216 tokens;
- two prompts built in the test from fixed text and cut to 8100 tokens with
  the checkpoint's tokenizer (the test checks that each one re-tokenizes to
  8000-8192 tokens). The first one ends inside a repeat of a story that
  starts about 7800 tokens earlier; the second one is ledger lines with a
  short repeating pattern;
- chunked prefill in 1024-token chunks with 1024-token KV segments, so each
  prompt prefills in 8 chunks through the segmented attention kernel;
- 16 generated tokens per prompt, the template's tolerance map, and the
  per-prompt checks `pp_max_linf_1.5x` and `pp_max_l2_1.5x` against the CPU
  BF16 reference.

The reference runs on the host CPU in FP32 and in BF16 at the full prompt
length, which takes tens of minutes in the first case; the second case reads
the cached reference logits. The test needs 8 NeuronCores:

```bash
OURO_CHECKPOINT_PATH=/path/to/Ouro-1.4B \
pytest contrib/models/ouro/test/integration/test_model.py -v -s --timeout=14400
```

Without `OURO_CHECKPOINT_PATH` the test downloads the pinned revision. The
test registers the architecture itself, from the test process; the engine
core and the workers inherit the registration through the default `fork`
start method, so do not set `VLLM_WORKER_MULTIPROC_METHOD=spawn`.

## Validation report

### vLLM Neuron 0.29.0.1.2.0 (2026-11-24)

#### Environment

| Image | Instance | TP | vLLM | vllm-neuron | torch | transformers | neuronx-cc |
|---|---|---|---|---|---|---|---|
| `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` | `trn2.48xlarge` (LNC 2), 1 node | 8 | 0.29.0 | 0.29.0.1.2.0 | 2.13.0 | 5.17.0 | 2.28.7631.0+8f7b0cf5 |

Checkpoint of the unit, integration and serve runs: `ByteDance/Ouro-1.4B` at
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.

#### Results

| Test | Configuration | Result |
|---|---|---|
| CPU unit tests, `test/unit/` | `VLLM_NEURON_CPU_MODE=1`, `OURO_CHECKPOINT_PATH` set (the 6 real-weight tests of `test_checkpoint_cpu.py` ran) | 24 passed, 0 failed |
| `test_model_accuracy[1-9216]` | TP 8, `neuron_native`, BF16, batch 1, max model length 9,216, prefill chunk 1,024 with KV segment 1,024, 2 prompts cut to 8,100 tokens (asserted 8,000-8,192) × 16 greedy tokens, on-device greedy sampling | PASSED (1009.709 s) on aggregate σ-ratio 0.6133 ≤ 1.0; static per-prompt bounds 0/2 (below) |
| `test_model_accuracy[4-9216]` | TP 8, `neuron_native`, BF16, batch 4 (each prompt in 4 rows), max model length 9,216, prefill chunk 1,024 with KV segment 1,024, 2 prompts cut to 8,100 tokens (asserted 8,000-8,192) × 16 greedy tokens, on-device greedy sampling | PASSED (411.665 s) on aggregate σ-ratio 0.5410 ≤ 1.0; static per-prompt bounds 0/2 (below) |
| `vllm serve` smoke | The Usage command, installed with `pip install --no-deps -e contrib/models/ouro`; `temperature` 0, `max_tokens` 32 | Passed: 3/3 prompts, 32 completion tokens each; a batched request of all 3 at `max_tokens` 16 returned a prefix of each |
| `vllm serve` long prompt | The same server; a deterministic synthetic ledger document, measured with the server's `/tokenize` endpoint; `max_tokens` 1024, `temperature` 0, `ignore_eos` | 8,190 prompt tokens, 1,024 completion tokens, finish reason `length`, 31.9 s |

The integration test asserts `pp_max_linf_1.5x` and `pp_max_l2_1.5x` on every prompt; both were 2/2 in both cases of this run.

Not run on 0.29: module tests (`test/module/`), not part of the 0.29 run.

#### Logit validation

| Case | Prompt | Overall Status | Max K5 (0.03) | Max K50 (0.05) | Max K1000 (0.08) | Max All (0.09) | Argmax divergence (1 ULP) |
|---|---|---|---|---|---|---|---|
| `[1-9216]` | 1 of 2, `Chapter 1 …` | FAILED | 0.0427 (> 0.03) at token 15 | 0.0427 at token 15 | 0.0427 at token 15 | 0.0359 at token 15 | 0.0000 |
| `[1-9216]` | 2 of 2, `Storeroom ledger …` | FAILED | 0.1005 (> 0.03) at token 14 | 0.1901 (> 0.05) at token 15 | 0.2247 (> 0.08) at token 15 | 0.2695 (> 0.09) at token 15 | 0.0000 |
| `[4-9216]` | 1 of 2, `Chapter 1 …` | FAILED | 0.0397 (> 0.03) at token 15, all 4 rows | 0.0397 at token 15, all 4 rows | 0.0397 at token 15, all 4 rows | 0.0334 at token 15, all 4 rows | 0.0000 |
| `[4-9216]` | 2 of 2, `Storeroom ledger …` | FAILED | 0.0814 (> 0.03) at token 14, all 4 rows | 0.1617 (> 0.05) at token 5, all 4 rows | 0.1939 (> 0.08) at token 5, all 4 rows | 0.1805 (> 0.09) at token 5, all 4 rows | 0.0000 |

Neuron vs the CPU BF16 reference (`test/ouro_reference.py`); token = generated token, row = batch row. Tokens over a bound: `[1-9216]` prompt 1 token 15 (K5), prompt 2 token 5 (K50 0.1649, K1000 0.1968, All 0.1849), tokens 14 and 15 (K5, K50, K1000 and All); `[4-9216]` prompt 1 token 15 (K5), prompt 2 token 5 (K50 0.1617, K1000 0.1939, All 0.1805), tokens 14 and 15 (K5, K50, K1000 and All), all 4 rows. No argmax divergence. At prompt 2 tokens 5, 14 and 15 the CPU BF16 L-inf error against FP32 is 0.1793, 0.0954 and 0.2153.
Both cases passed on aggregate σ-ratio 0.6133 (`[1-9216]`) and 0.5410 (`[4-9216]`) ≤ 1.0 (`multi_prompt_logit_validation`) and on the asserted per-prompt checks; `agg_bc`, `agg_l2_1.2x`, `agg_l2_1.5x`, `agg_linf_1.2x` and `agg_linf_1.5x` FAIL in both cases; printed per-prompt checks: `[1-9216]`: `pp_max_linf_1.5x` 2/2, `pp_max_linf_2.0x` 2/2, `pp_max_l2_1.5x` 2/2, `pp_max_l2_2.0x` 2/2, `pp_tok_linf_1.5x` 0/2, `pp_tok_linf_2.0x` 2/2, `pp_tok_l2_1.5x` 1/2, `pp_tok_l2_2.0x` 2/2, `pp_static_0.03` 1/2, `pp_static_0.05` 1/2; `[4-9216]`: the same, except `pp_tok_linf_1.5x` 1/2.

Error against the FP32 reference, per prompt (max over the 16 generated tokens):

| Case | Prompt | CPU BF16 L-inf | Neuron L-inf | Neuron / CPU BF16 | CPU BF16 L2 | Neuron L2 | Neuron / CPU BF16 |
|---|---|---|---|---|---|---|---|
| `[1-9216]` | 1 of 2 | 0.0317 | 0.0187 | 0.59x | 0.0312 | 0.0276 | 0.88x |
| `[1-9216]` | 2 of 2 | 0.2153 | 0.1088 | 0.51x | 0.3265 | 0.2056 | 0.63x |
| `[4-9216]` | 1 of 2 | 0.0317 | 0.0218 | 0.69x | 0.0312 | 0.0290 | 0.93x |
| `[4-9216]` | 2 of 2 | 0.2153 | 0.1228 | 0.57x | 0.3265 | 0.1905 | 0.58x |

Prompt 1: CPU BF16 maxima at generated token 15, Neuron maxima at token 11. Prompt 2: every maximum at generated token 15.

#### Serve smoke

`/v1/completions`, on-device greedy sampling (`all_greedy`), `temperature` 0; `max_tokens` 32 for the three short prompts, `max_tokens` 1024 with `ignore_eos` for the long prompt; first 60 characters of each output as a Python string literal.

| Prompt | Output |
|---|---|
| `The capital of France is` | `" Paris.\n\n## Step 2: Evaluate the Options\nLet's evaluate each"` |
| `1, 2, 3, 4, 5,` | `' 6, 7, 8, 9, 10, 11, 12, 13, 14,'` (the whole output) |
| `def fibonacci(n):` | `'\n    if n <= 0:\n        return "Input should be a positive i'` |
| Synthetic ledger document, 8,190 tokens | `' The ledger provides a detailed record of trade activities a'` |

### Exceptions

- **All four prompt-runs are outside the static per-prompt bounds.** Both
  integration cases passed on the aggregate σ-ratio and on the asserted
  per-prompt checks `pp_max_linf_1.5x` and `pp_max_l2_1.5x` (see Logit
  validation).
- **Module tests not run on 0.29.**
- Prefix caching, TP sizes other than 8 on device, batch sizes other than 1
  and 4, context beyond 9,216 tokens, prefill chunk and KV segment sizes
  other than 1,024, sampled (non-greedy) requests, `early_exit_step`,
  speculative decoding and Trn3 are not tested.
