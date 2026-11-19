# Nanbeige4.2 (Text)

This folder serves the Nanbeige4.2 text models with vLLM on AWS Trainium
through the `vllm-neuron` plugin. The implementation under `src/` is a
Llama-style dense decoder with one addition: the stack of physical decoder
layers runs twice with the same weights, and each `(loop, layer)` pair holds
its own KV cache. See `src/README.md` for the architecture and the
feature-status card.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md` for installation), then install this folder:

```bash
pip install -e contrib/models/nanbeige
```

The install adds a `vllm.general_plugins` entry point (`src/plugin.py`) that
registers the architecture `NanbeigeForCausalLM` in every vLLM process: the
front end, the engine core and each tensor-parallel worker. So `vllm serve`
and `LLM(...)` find the model with no other setup. The package is installed
from `src/` under the name `nanbeige_neuron`, and
`pip uninstall vllm-neuron-contrib-nanbeige` removes the registration. The
checkpoint ships its own model code, so every entry point needs
`trust_remote_code`.

### Serve with the OpenAI-compatible server

These are the flags of the tested serve run (see the validation report):

```bash
vllm serve Nanbeige/Nanbeige4.2-3B \
    --host 127.0.0.1 \
    --port 8000 \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 2048 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching
```

Then send a request:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Nanbeige/Nanbeige4.2-3B",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

With the folder installed, build the engine directly:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Nanbeige/Nanbeige4.2-3B",
    trust_remote_code=True,
    dtype="bfloat16",
    tensor_parallel_size=8,
    max_model_len=2048,
    max_num_batched_tokens=2048,
    max_num_seqs=1,
    enable_prefix_caching=False,
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The color of the sky is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

Prefill runs sequence-parallel over the padded prefill length, so each
prefill token bucket must be greater than the tensor-parallel size and a
multiple of it. The prompt itself can be any length: the serve smoke test
below sent prompts of 6, 7 and 16 tokens at TP8.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1` in the environment. The fused decode megakernel does
not run on CPU, so CPU mode covers the prefill path only. See the top-level
`README.md` and `AGENTS.md` for the full list of `VLLM_NEURON_*` environment
variables.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn3 | Not tested | Not tested |
| Trn2 | Working (TP 8) | Not tested |

Tested in the Neuron Deep Learning Container
`public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04`
with vLLM 0.29.0, transformers 5.17.0 and neuronx-cc 2.28.7631.0. The earlier
runs in the validation report used `vllm-neuron` 0.31.0.1.2.0 and name no
Neuron SDK release, so the matrix does not count them.

## Example Checkpoints

- <https://huggingface.co/Nanbeige/Nanbeige4.2-3B> (tested at the pinned revision
  `b82e54bd609793562a75cbf9337970a93369eab5`)

## Testing

The integration test runs the model through vLLM's offline `LLM` API on
Neuron at TP8 and validates its per-token logits against a HuggingFace
reference, for one sequence and for four sequences at a 2,048-token context:

```bash
pytest contrib/models/nanbeige/test/integration/test_model.py \
    -v -s --timeout=3600
```

Set `NANBEIGE4_CHECKPOINT_PATH` to a local checkpoint directory to avoid a
Hub download. The test registers the factory from this folder's `src/` itself,
so it does not need the `pip install` above. Its goldens come from the
independent upstream-Llama loop reference in `test/nanbeige_utils.py`
(transformers' `LlamaForCausalLM`, with the physical layers run once per loop
and a KV cache per loop), not from the checkpoint's own model code; see
Exceptions.

Every test here loads this folder's `src/` by file location through
`test/_local_src.py`, so the tests work from any working directory and after
the folder is copied out of the repository. They need the `vllm-neuron` plugin
installed, but nothing else from the repository. `test/conftest.py` shuts down
the plugin's worker pool at the end of the session, so pytest exits after it
prints its summary.

Run one model's tests per pytest invocation, or pass
`--import-mode=importlib`. The reason is pytest's default import mode, not
this folder: contrib/README.md gives every model the same
`test/integration/test_model.py` filename, and prepend mode cannot import two
test modules with the same basename in one run.

### Unit tests (CPU mode)

```bash
VLLM_NEURON_CPU_MODE=1 VLLM_NEURON_BACKEND=neuron_native \
    NANBEIGE4_CHECKPOINT_PATH=/path/to/Nanbeige4.2-3B \
    pytest contrib/models/nanbeige/test/unit/ -v
```

Four cases: the registration and factory wiring with the config parse, the
factory's refusal of each unsupported option, a TP2 full-model load and
prefill-logit comparison, and a cross-check of the independent reference
against the checkpoint's own model code.

These tests read the real checkpoint and refuse to run without it. They also
check that the checkpoint directory names the pinned revision
(`b82e54bd609793562a75cbf9337970a93369eab5`), so point the variable at a
directory whose path contains that revision. The module tests below have the
same requirement.

### Module tests

`test/module/test_decoder_layer.py` is the per-component test: it runs one
physical decoder layer in isolation at TP2, once for loop 0 and once for
loop 1, and gates on `assert_close_three_way` against an upstream
`LlamaDecoderLayer` reference — for the layer output and for the KV cache that
each loop owns. It is parameterized over the first and the last physical
layer, which is two cases.

```bash
VLLM_NEURON_CPU_MODE=1 VLLM_NEURON_BACKEND=neuron_native \
    NANBEIGE4_CHECKPOINT_PATH=/path/to/Nanbeige4.2-3B \
    pytest contrib/models/nanbeige/test/module/ -v -s --timeout=600
```

The unit tests and the module tests share `test/nanbeige_utils.py`, which sits
at the root of the test tree for that reason. The fused decode megakernel does
not run on CPU, so the module test covers prefill only.

## Validation report

### vLLM Neuron 0.29.0.1.2.0 (2026-11-19)

#### Environment

| Image | Instance | TP | vLLM | vllm-neuron | torch | transformers | neuronx-cc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` | `trn2.48xlarge` (LNC=2) | 8 | 0.29.0 | 0.29.0.1.2.0 | 2.13.0 | 5.17.0 | 2.28.7631.0+8f7b0cf5 |

#### Results

| Test | Configuration | Result |
| --- | --- | --- |
| CPU unit tests, `test/unit/` | CPU mode, `neuron_native` backend, TP2 full-model load | 4 passed (127.97 s) |
| `test_model_accuracy[1-2048]` | TP8, 1 sequence, `max_model_len` 2048, 2 prompts x 32 tokens | PASSED |
| `test_model_accuracy[4-2048]` | TP8, 4 sequences, `max_model_len` 2048, 2 prompts x 32 tokens | PASSED |
| `vllm serve` smoke | The flags under Usage, 3 prompts | passed (3/3 prompts) |

Not run on 0.29: module tests (`test/module/`, 2 cases), not in the 0.29 run script; last pass on 0.31.0.1.2.0 (CPU).

#### Logit validation

| Case | Prompt | Overall Status | K5 (<= 0.011) | K50 (<= 0.02) | K1000 (<= 0.03) | All (<= 0.05) |
| --- | --- | --- | --- | --- | --- | --- |
| `[1-2048]` | 1 | PASSED | 0.0100 | 0.0161 | 0.0161 | 0.0200 |
| `[1-2048]` | 2 | FAILED | 0.0168 (> 0.011) | 0.0168 | 0.0168 | 0.0186 |
| `[4-2048]` | 1 | PASSED | 0.0100 | 0.0161 | 0.0161 | 0.0200 |
| `[4-2048]` | 2 | FAILED | 0.0168 (> 0.011) | 0.0168 | 0.0168 | 0.0186 |

Two-way check, Neuron vs CPU BF16 reference; max over all validated tokens, the same in each of the 4 sequences of `[4-2048]`. No argmax divergence (0.0000, threshold 1 ULP).
Prompt 2 over K5 at generated token 24 (L-inf vs FP32: Neuron 0.0175, CPU BF16 0.0111). Both cases passed on aggregate σ-ratio 0.9350 ≤ 1.0 (`multi_prompt_logit_validation`).

#### Serve smoke

`/v1/completions`, `max_tokens` 32, `temperature` 0; a 4th request with the 3 prompts batched (`max_tokens` 16) returned a prefix of each completion.

| Prompt | Prompt tokens | Completion tokens | Completion (first 60 characters, JSON-quoted) |
| --- | --- | --- | --- |
| `The capital of France is` | 6 | 32 | `" Paris.\n</think>\n\nHere are a few ways to respond to your sta"` |
| `1, 2, 3, 4, 5,` | 16 | 32 | `" 6, 7, 8, 9, 10, 11, 12, 13, 14,"` (whole completion) |
| `def fibonacci(n):` | 7 | 32 | `"\n 不仅如此，斐波那契数列还有多种变体，例如卢卡斯数列、斐波那契编码、斐波那契堆等数据结构。\n\n卢卡斯"` (whole completion) |

#### Changes for 0.29

- `pyproject.toml`, `src/plugin.py` (new): install the folder as `nanbeige_neuron` with a `vllm.general_plugins` entry point, replacing the manual registration steps.
- `src/compat.py` (new): vendored `hoist_rope_parameters` (not in 0.29.0.1.2.0), copied unchanged from a later plugin release; `src/config.py` imports it. No other `src/` file changed.
- `test/conftest.py` (new): shuts down the plugin's `MPExecutor` worker pool at session end, so pytest exits.
- `test/integration/test_model.py`: goldens from `load_loop_reference` in `test/nanbeige_utils.py` (variant `transformers_llama_loop_v1`), not the remote code, which calls `DynamicCache.from_legacy_cache` (removed in transformers 5.17).
- `test/integration/test_model.py`: maps the padding ids past the tokenizer vocabulary to `""` in the engine's tokenizer (vLLM 0.29.0's detokenizer crashed on their `None` token strings).

### Earlier runs (vLLM Neuron 0.31.0.1.2.0)

Before the changes above, the port was validated with `vllm-neuron`
0.31.0.1.2.0 and vLLM 0.31.0, with the native runtime backend
(`neuron_native`) on `trn2.48xlarge`. Those runs imported `src/` through the
repository-rooted `contrib.models.nanbeige.src` path, not through
`test/_local_src.py`.

| Test | Result |
| --- | --- |
| CPU-mode loading, configuration and decoder-layer cases (then one file; now the 4 cases of `test/unit/` and the 2 of `test/module/`, with no case added or dropped) | 6 of 6 passed |
| Two-node device test at TP8, 4 ranks on each of two nodes: a decoder-layer prefill-and-decode comparison for the first and the last physical layer, a server logit validation and a serving run | 8 of 8 passed (the 4 cases on each node) |

The run results give the verdict and the case counts of the device test, not
the result of each case. Its server logit case compared the full-vocabulary
logits of each of 32 generated tokens against an FP32 baseline plus a
same-dtype teacher-forced HuggingFace reference, with the default tolerance
map of `vllm_neuron.accuracy`. Its serving case sent 4 concurrent OpenAI-API
completion requests of exactly 1,024 prompt tokens each, forced 512 output
tokens with `min_tokens` and `ignore_eos`, and asserted the token counts and
a non-empty completion. There was no run at 8,192 in / 1,024 out and no
offline (`LLM.generate`) run at either shape. The device test depends on
shared server and multi-node test helpers that are not in this repository, so
it is not in this folder and cannot be run again from it.

### Exceptions

- **No dataset accuracy result.** There is no GSM8K or other dataset
  accuracy run, so this README states no accuracy score. The accuracy
  evidence is the logit validation above.
- **One two-way bound is exceeded.** On vLLM Neuron 0.29.0.1.2.0, prompt 2
  exceeds the top-5 two-way bound (0.0168 against 0.011) at one token. The
  integration cases pass on the aggregate σ-ratio, as described above.
- **The integration goldens do not come from the checkpoint's model code.**
  The remote code cannot generate on transformers 5.17 (see the changes
  above), so the goldens come from the folder's independent upstream-Llama
  loop reference. `test_nanbeige_reference_matches_remote_code` compares that
  reference with the remote code in FP32, without a KV cache, on the
  last-position logits of one prompt (`atol=2e-3`, `rtol=1e-3`, same argmax),
  and passed in the 0.29.0.1.2.0 run.
- **Logprob strings for padding ids.** The mapping of the padding ids to `""`
  is in the integration test only. A request outside the test that returns
  logprobs for those ids is not covered by it.
- **Offline path with the installed package.** The `vllm serve` smoke test
  ran with this folder installed. The offline `LLM(...)` path ran in the
  integration test, which registers the factory itself and ran before the
  install. The offline example under Usage, which relies on the installed
  entry point alone, has no separate run.
- **Module tests not run since the move to `test/_local_src.py`.** The 2
  cases of `test/module/` were not part of the 0.29.0.1.2.0 run. Their last
  recorded pass is the 0.31.0.1.2.0 CPU run, which used the old import path.
- **`PROMPTS_2` is vendored.** The two validation prompts are the
  `PROMPTS_2` set that the port's original tests used, copied verbatim into
  `test/nanbeige_utils.py` and into the integration test, as
  contrib/README.md permits.
- **The registry assertion has a different meaning.** The original CPU test
  asserted that the plugin's own `get_models()` maps the architecture to the
  factory. A contrib model is not in `get_models()`. So the test in this
  folder registers the local factory with vLLM's `ModelRegistry` and asserts
  that the architecture resolves to the factory. The case count is the same.
- **Untested configurations.** Only TP8 (device: one node on 0.29.0.1.2.0,
  two nodes on 0.31.0.1.2.0) and TP2 (CPU) have results; no other
  tensor-parallel degree, no Trn3, no quantized weights, no speculative
  decoding (the code has a rejection-sampling branch with no evidence behind
  it), no data-parallel or disaggregated (prefill/decode) deployment, no
  prefix caching (the runs disabled it), and no context length beyond 2,048
  tokens, even though the checkpoint config allows far more.
- **No performance data.** No benchmark or latency test ran for this model,
  so this README states no throughput, latency or time-to-first-token figure.
