# Contrib Model: Gemma3 text (rnj-1-instruct)

[rnj-1-instruct](https://huggingface.co/EssentialAI/rnj-1-instruct) is an
8B-parameter dense decoder with the Gemma 3 text architecture
(`Gemma3ForCausalLM`, `model_type` `gemma3_text`): 32 layers with full
attention in every layer, hidden size 4096, grouped-query attention with
32 query heads and 8 KV heads of size 128, a per-head QK-norm, a GELU-tanh
MLP of size 16384, YaRN RoPE, a final logit soft cap of 30 and a 128256-token
vocabulary with tied embeddings. The checkpoint stores FP32 weights.

This folder serves it with vLLM on AWS Trainium2 through the `vllm-neuron`
plugin, in BF16 with tensor parallelism. See [src/README.md](src/README.md)
for the architecture, the differences from the plugin's Qwen3 model and the
per-feature status.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix and the
> validation report below record exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`), then install this folder:

```bash
pip install -e contrib/models/gemma3
```

The install makes `src/` importable as the `gemma3_neuron` package and adds a
`vllm.general_plugins` entry point (`src/plugin.py`) that registers
`Gemma3ForCausalLM` in every vLLM process (the API server, the engine core and
each tensor-parallel worker), so `vllm serve` and `LLM(...)` find it with no
other setup. vLLM ships its own `Gemma3ForCausalLM`; this registration
replaces it with the Neuron implementation while the package is installed.
`pip uninstall vllm-neuron-contrib-gemma3` removes it.

Two settings are specific to this model:

- **Serve in BF16.** Pass `--dtype bfloat16`. The checkpoint stores FP32
  weights; the implementation supports only the BF16 served dtype and casts
  each weight when it loads it.
- **Use a TP degree that divides the 32 query heads and the 128256-token
  vocabulary.** The recorded runs use TP 8 (4 query heads and one KV head on
  each rank).

### Serve with the OpenAI-compatible server

```bash
vllm serve EssentialAI/rnj-1-instruct \
    --revision 5ef96d7a82b2bff8009192979432c9bc64e870cd \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --block-size 32 \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{"neuron_config": {"on_device_sampling_config": {"all_greedy": true}, "num_batched_tokens_buckets": [2048], "num_seqs_buckets": [4]}}'
```

`--host 127.0.0.1` keeps the server local to the host; it listens on port
8000. This configuration matches the batch-4 case of the integration test, so
with a shared Neuron compile cache the server reuses those graphs. The two
bucket lists compile one prefill graph and one decode graph instead of a
ladder of each; drop them to compile the default ladder. `all_greedy` makes
every request greedy; remove it (and `--no-async-scheduling`) to serve
sampled requests, a path that was not run for this model.

Then send a request:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "EssentialAI/rnj-1-instruct",
        "prompt": "The color of the sky is",
        "max_tokens": 32,
        "temperature": 0
    }'
```

### Offline inference with the Python API

```python
from vllm import LLM, SamplingParams

# With `pip install -e contrib/models/gemma3` the architecture is registered
# in every vLLM process; no registration call is needed here.
llm = LLM(
    model="EssentialAI/rnj-1-instruct",
    revision="5ef96d7a82b2bff8009192979432c9bc64e870cd",
    tensor_parallel_size=8,
    dtype="bfloat16",
    max_model_len=2048,
    max_num_batched_tokens=2048,
    max_num_seqs=4,
    block_size=32,
    enable_prefix_caching=False,
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The color of the sky is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

Without the install, register the factory from your script before you build
`LLM(...)` (option 1 in [contrib/README.md](../../README.md)), with `src/`
copied into your application:
`ModelRegistry.register_model("Gemma3ForCausalLM", Gemma3ForCausalLM)`. The
engine core and the TP workers inherit that registration only under the
default `fork` start method, so leave `VLLM_WORKER_MULTIPROC_METHOD` unset.

To iterate without Neuron hardware, set `VLLM_NEURON_CPU_MODE=1` and pass
`enforce_eager=True`; `test/unit/test_gemma3_tiny_e2e.py` shows a complete
CPU-mode configuration.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0, vLLM 0.29.0) | 2.32 and earlier |
| ---------------------- | -------------------------------------------- | ---------------- |
| Trn2 (`trn2.48xlarge`) | TP 8: see the [validation report](#validation-report) | Not tested |
| Trn3                   | Not tested | Not tested |

| Feature | Status |
|---|---|
| FP32 checkpoint, BF16 serving | Supported (other served dtypes are rejected by the factory) |
| Tensor parallelism | TP 8 recorded. The one-KV-head-per-rank decode branch of TP 8 and TP 16 is CPU-tested with a tiny model at TP 2. TP 1, 2, 4 and 16 are legal but were not run from this folder |
| Context length | 2048 recorded; the model supports up to 32768 (YaRN) |
| Prefix caching / segmented prefill | Not tested (every recorded run disables prefix caching) |
| Sliding-window Gemma 3 checkpoints | Not supported (the factory rejects `sliding_attention` layers) |
| Speculative decoding, quantized checkpoints, data parallelism | Not tested |

## Example Checkpoints

- <https://huggingface.co/EssentialAI/rnj-1-instruct> — the tests pin revision
  `5ef96d7a82b2bff8009192979432c9bc64e870cd`.

Other `Gemma3ForCausalLM` text checkpoints load only when every layer is a
full-attention layer and the checkpoint has no attention logit soft cap; the
Google Gemma 3 text checkpoints use sliding-window layers and are rejected at
construction.

## Testing

### Unit tests (CPU mode)

No Neuron hardware and no checkpoint download are needed. The tests read the
pinned `config.json` from `test/gemma3_test_utils.py` and build a tiny
random-weight checkpoint with the rnj-1 layout in a temporary directory:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/gemma3/test/unit/ -v -s --timeout=1800
```

| File | Covers |
|---|---|
| `test_gemma3_config.py` | Plugin entry point and vLLM model protocol, config parse (transformers 5 `rope_parameters`, raw `config.json`, plain dict), factory guards, YaRN RoPE against HF (also when built on the meta device), weight-map coverage against the HF model and a saved checkpoint, TP 8 / TP 16 head split, weight-loader shards and the tied LM head |
| `test_gemma3_components.py` | Gemma RMSNorm (`1 + w`) and the kernel gamma, the scaled embedding, the GELU-tanh MLP through `NF.mlp` three-way against HF (one rank and the sum of 8 TP shards), final soft cap, decoder-layer order, the KV helpers in `src/compat.py` |
| `test_gemma3_tiny_e2e.py` | The full vLLM pipeline at TP 2 (one KV head per rank, the decode branch of TP 8): registration, FP32-to-BF16 load, prefill, 8 decode steps across a KV block boundary, on-device greedy sampling, full-vocabulary raw logits against HF teacher forcing |

The tiny end-to-end test starts vLLM worker processes, whose message queues
reserve a few hundred MiB of `/dev/shm`. It skips itself when less than
512 MiB is free; run it on a host or in a container with a larger `/dev/shm`
(for example `docker run --shm-size=1g`).

### Integration test (Neuron)

Serves the model through vLLM at TP 8 on one `trn2.48xlarge` and validates
its logits against a HuggingFace reference (batch 1 and batch 4, 2048-token
model length, 16 tokens per prompt, the template's tolerance map). It needs
8 NeuronCores, and the HF reference runs the 8B model on the CPU in FP32 and
BF16, so allow a generous timeout:

```bash
GEMMA3_CHECKPOINT_PATH=/path/to/rnj-1-instruct \
pytest contrib/models/gemma3/test/integration/test_model.py -v -s --timeout=10800
```

Without `GEMMA3_CHECKPOINT_PATH` the test downloads the pinned revision.

The integration test does not need the `pip install`: it registers the factory
itself, from the test process, before it builds the engine. vLLM's engine-core
and worker processes inherit that registration through the default `fork`
start method, so do not set `VLLM_WORKER_MULTIPROC_METHOD=spawn` for this test.

All test directories reach the implementation through `test/_local_src.py`,
which loads `src/` by file location under the unique module name
`contrib_gemma3_src`, so no test needs the repository root on `sys.path`.
`test/conftest.py` puts the `test/` directory on `sys.path`, and shuts down
the plugin's `MPExecutor` worker pool at the end of the session; without that
pytest prints its summary and then does not exit.

## Validation report

### vLLM Neuron 0.29.0.1.2.0 (2026-11-23)

#### Environment

| Image | Instance | TP | vLLM | vllm-neuron | torch | transformers | neuronx-cc |
|---|---|---|---|---|---|---|---|
| `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` | `trn2.48xlarge` (LNC 2), 1 node | 8 | 0.29.0 | 0.29.0.1.2.0 | 2.13.0 | 5.17.0 | 2.28.7631.0+8f7b0cf5 |

Checkpoint of the integration test and the serve smoke: `EssentialAI/rnj-1-instruct` at
`5ef96d7a82b2bff8009192979432c9bc64e870cd`.

#### Results

| Test | Configuration | Result |
|---|---|---|
| CPU unit tests, `test/unit/` | `VLLM_NEURON_CPU_MODE=1` | 17 passed, 0 failed |
| `test_model_accuracy[1-2048]` | TP 8, `neuron_native`, BF16 (FP32 checkpoint), batch 1, max model length 2,048, 2 prompts × 16 greedy tokens, on-device greedy sampling | PASSED (210.86 s); static per-prompt bounds 2/2 (below); aggregate σ-ratio 0.9085 ≤ 1.0 |
| `test_model_accuracy[4-2048]` | TP 8, `neuron_native`, BF16 (FP32 checkpoint), batch 4 (each prompt in 4 rows), max model length 2,048, 2 prompts × 16 greedy tokens, on-device greedy sampling | PASSED (243.35 s); static per-prompt bounds 2/2 (below); aggregate σ-ratio 0.9085 ≤ 1.0 |
| `vllm serve` smoke | The Usage command, installed with `pip install -e contrib/models/gemma3` | Passed: 3/3 prompts, 32 completion tokens each; a batched request of all 3 returned a prefix of each |

#### Logit validation

| Case | Prompt | Overall Status | Max K5 (0.03) | Max K50 (0.05) | Max K1000 (0.08) | Max All (0.09) | Argmax divergence (1 ULP) |
|---|---|---|---|---|---|---|---|
| `[1-2048]` | 1 of 2, `The color of the sky is` | PASSED | 0.0080 at token 2 | 0.0109 at token 11 | 0.0126 at token 2 | 0.0139 at token 2 | 0.0000 |
| `[1-2048]` | 2 of 2, `A short summary of the history of the printing press:` | PASSED | 0.0118 at token 15 | 0.0159 at token 15 | 0.0217 at token 15 | 0.0196 at token 13 | 0.0000 |
| `[4-2048]` | 1 of 2, `The color of the sky is` | PASSED | 0.0080 at token 2, all 4 rows | 0.0109 at token 11, all 4 rows | 0.0126 at token 2, all 4 rows | 0.0139 at token 2, all 4 rows | 0.0000 |
| `[4-2048]` | 2 of 2, `A short summary of the history of the printing press:` | PASSED | 0.0118 at token 15, all 4 rows | 0.0159 at token 15, all 4 rows | 0.0217 at token 15, all 4 rows | 0.0196 at token 13, all 4 rows | 0.0000 |

Neuron vs CPU BF16 reference; token = generated token, row = batch row. No token over a bound. No argmax divergence.
Both cases passed: two-way validation 2/2 prompts, aggregate σ-ratio 0.9085 ≤ 1.0 (`multi_prompt_logit_validation`; the test asserts `result.passed` only); `agg_bc` FAIL; `agg_l2_1.2x`, `agg_l2_1.5x`, `agg_linf_1.2x`, `agg_linf_1.5x` PASS; printed per-prompt checks, the same in both cases: `pp_max_linf_1.5x` 2/2, `pp_max_linf_2.0x` 2/2, `pp_max_l2_1.5x` 2/2, `pp_max_l2_2.0x` 2/2, `pp_tok_linf_1.5x` 2/2, `pp_tok_linf_2.0x` 2/2, `pp_tok_l2_1.5x` 2/2, `pp_tok_l2_2.0x` 2/2, `pp_static_0.03` 2/2, `pp_static_0.05` 2/2.

Error against the FP32 reference, per prompt (max over the 16 generated tokens):

| Case | Prompt | CPU BF16 L-inf | Neuron L-inf | Neuron / CPU BF16 | CPU BF16 L2 | Neuron L2 | Neuron / CPU BF16 |
|---|---|---|---|---|---|---|---|
| `[1-2048]` | 1 of 2 | 0.0155 | 0.0126 | 0.81x | 0.0352 | 0.0315 | 0.89x |
| `[1-2048]` | 2 of 2 | 0.0163 | 0.0161 | 0.99x | 0.0313 | 0.0310 | 0.99x |
| `[4-2048]` | 1 of 2 | 0.0155 | 0.0126 | 0.81x | 0.0352 | 0.0315 | 0.89x |
| `[4-2048]` | 2 of 2 | 0.0163 | 0.0161 | 0.99x | 0.0313 | 0.0310 | 0.99x |

Prompt 1: CPU BF16 L-inf maximum at generated token 0, Neuron L-inf at token 2, both L2 maxima at token 7. Prompt 2: both L-inf maxima at token 15, both L2 maxima at token 9.

#### Serve smoke

`/v1/completions`, `max_tokens` 32, on-device greedy sampling (`all_greedy`); first 60 characters of each output as a Python string literal.

| Prompt | Output |
|---|---|
| `The capital of France is` | `' Paris. The capital of France is Paris. The capital of Franc'` |
| `1, 2, 3, 4, 5,` | `' 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16'` (the whole output) |
| `def fibonacci(n):` | `' #  (1) n = 10;\n    if n <= 0:\n        return 0\n    elif n ='` |

### Exceptions

- **The serve smoke output for `The capital of France is` repeats itself.**
  The full 32-token output is
  `' Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital'`.
  The logit validation of both integration cases passed against the HF
  reference (see Logit validation).
- **The server default temperature comes from the checkpoint.** The serve log
  warns that the checkpoint's `generation_config.json` overrides the default
  sampling parameters with `temperature` 0.2. The smoke run used `all_greedy`
  on-device sampling; sampled serving was not run.
- **Dropped: the port's GSM8K (100 samples, batch 1 and 4) and serving
  benchmark (1024 input / 100 output tokens, batch 1 and 4) tests at TP 8**,
  which passed on `vllm-neuron` 0.31.0.1.2.0 with the port before the 0.29
  changes, and its two-node disaggregated logit test (TP 16, 1 prefill + 1
  decode node). They depend on the plugin's shared serving and multi-node test
  harness, which is outside this folder. No accuracy-benchmark score and no
  performance figures on 0.29.
- **Dropped: the port's kernel-dispatch instrumentation** (a debug FX pass in
  the plugin and the test that read its records). It does not change the
  model.
- **Rewritten: the port's real-checkpoint component tests.** They read the
  33 GB FP32 checkpoint. The shipped unit tests use the pinned `config.json`
  and random weights, and add the tiny end-to-end test.
- **The decode branch with more than one KV head per rank (TP 1, 2, 4) is not
  verified.** In CPU mode with the NKI simulator its decode attention returns
  NaN when the block table has inactive (`-1`) entries; it was not run on
  device. TP 8 and TP 16 use the one-KV-head-per-rank branch.
- Prefix caching (the segmented-prefill path), TP sizes other than 8 on
  device, context beyond 2048 tokens, speculative decoding and Trn3 are not
  tested.
