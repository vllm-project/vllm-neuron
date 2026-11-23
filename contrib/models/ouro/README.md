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
  and the MLP size of 5632 must divide by the TP degree. The recorded runs
  use TP 8.
- **KV memory is 4x a 24-layer model.** Each of the 4 loop steps writes its
  own KV cache.

### Serve with the OpenAI-compatible server

These are the flags of the tested configuration. `--host 127.0.0.1` keeps the
server local to the host; it listens on port 8000.

```bash
vllm serve ByteDance/Ouro-1.4B \
    --host 127.0.0.1 \
    --revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 2048 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{"neuron_config": {"on_device_sampling_config": {"all_greedy": true}, "num_batched_tokens_buckets": [1024], "num_seqs_buckets": [4]}}'
```

This matches the batch-4 case of the integration test, so with a shared
Neuron compile cache the server reuses those graphs. The two bucket lists
compile one prefill graph and one decode graph instead of a power-of-two
ladder of each. `all_greedy` makes every request greedy; remove it (and
`--no-async-scheduling`) to serve sampled requests, a path that was not run
for this model.

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
# needed here.
llm = LLM(
    model="ByteDance/Ouro-1.4B",
    revision="574fa66cb8bf5abdc979642d01cf2b79b16bfab1",
    tensor_parallel_size=8,
    dtype="bfloat16",
    max_model_len=2048,
    max_num_batched_tokens=1024,
    max_num_seqs=4,
    enable_prefix_caching=False,
    additional_config={
        "neuron_config": {
            "num_batched_tokens_buckets": [1024],
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
| Trn2 (`trn2.48xlarge`) | TP 8: see the [validation report](#validation-report) | Not tested |
| Trn3                   | Not tested | Not tested |

| Feature | Status |
|---|---|
| BF16 checkpoint | Supported (other dtypes are rejected by the factory) |
| Tensor parallelism | TP 8 on device; TP 2 and 4 in the CPU tests |
| Context length | 2048 |
| Early exit | Threshold exit (`early_exit_threshold`, the checkpoint uses 1.0). `early_exit_step` is rejected by the factory |
| Prefix caching / segmented prefill | Segmented prefill is covered by the component tests; prefix caching was not run end to end (every recorded run disables it) |
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
| `test_checkpoint_cpu.py` | The port's CPU tests with the real weights (skipped unless `OURO_CHECKPOINT_PATH` is set): decoder layer prefill at TP 2, early-exit selection with the real gate, KV spec at TP 8, full weight load at TP 2 |

The decode cases run at TP 4 in CPU mode because the torch fallback of the
decode attention kernel supports one KV head per rank; the device kernel has
no such limit. `test_tiny_e2e.py` needs at least 240 MiB free in `/dev/shm`
(vLLM's TP worker queues) and skips otherwise; run the container with a
larger `--shm-size` or with `--ipc=host`.

### Module tests (Neuron)

`test/module/test_components_device.py` compiles one real module at a time at
TP 8 and compares it with the reference: flash prefill (1024 tokens),
segmented prefill (2 x 512 tokens), decode attention (batch 4, context 96),
and the MLP in prefill and decode. It needs 8 NeuronCores:

```bash
OURO_CHECKPOINT_PATH=/path/to/Ouro-1.4B \
pytest contrib/models/ouro/test/module/test_components_device.py -v -s --timeout=3600
```

### Integration test (Neuron)

Serves the model through vLLM at TP 8 on one `trn2.48xlarge` and validates
its logits against the reference (batch 1 and batch 4, 2048-token model
length, 1024-token prefill bucket, 16 tokens per prompt, the template's
tolerance map). It needs 8 NeuronCores:

```bash
OURO_CHECKPOINT_PATH=/path/to/Ouro-1.4B \
pytest contrib/models/ouro/test/integration/test_model.py -v -s --timeout=10800
```

Without `OURO_CHECKPOINT_PATH` the test downloads the pinned revision. The
test registers the architecture itself, from the test process; the engine
core and the workers inherit the registration through the default `fork`
start method, so do not set `VLLM_WORKER_MULTIPROC_METHOD=spawn`.

## Validation report

### vLLM Neuron 0.29.0.1.2.0 (2026-11-23)

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
| `test_model_accuracy[1-2048]` | TP 8, `neuron_native`, BF16, batch 1, max model length 2,048, prefill bucket 1,024, 2 prompts × 16 greedy tokens, on-device greedy sampling | PASSED (288.288 s): 2/2 prompts within the static bounds; aggregate σ-ratio 0.9627 ≤ 1.0 |
| `test_model_accuracy[4-2048]` | TP 8, `neuron_native`, BF16, batch 4 (each prompt in 4 rows), max model length 2,048, prefill bucket 1,024, 2 prompts × 16 greedy tokens, on-device greedy sampling | PASSED (351.339 s): 2/2 prompts within the static bounds; aggregate σ-ratio 0.9505 ≤ 1.0 |
| `vllm serve` smoke | The Usage command, installed with `pip install -e contrib/models/ouro` | Passed: 3/3 prompts, 32 completion tokens each; a batched request of all 3 returned a prefix of each |

Not run on 0.29: module tests (`test/module/`), not part of the 0.29 run.

#### Logit validation

| Case | Prompt | Overall Status | Max K5 (0.03) | Max K50 (0.05) | Max K1000 (0.08) | Max All (0.09) | Argmax divergence (1 ULP) |
|---|---|---|---|---|---|---|---|
| `[1-2048]` | 1 of 2, `The color of the sky is` | PASSED | 0.0113 at token 11 | 0.0113 at token 11 | 0.0113 at token 11 | 0.0105 at token 11 | 0.0000 |
| `[1-2048]` | 2 of 2, `A short summary of the history of the printing press:` | PASSED | 0.0104 at token 12 | 0.0104 at token 12 | 0.0104 at token 12 | 0.0109 at token 3 | 0.0000 |
| `[4-2048]` | 1 of 2, `The color of the sky is` | PASSED | 0.0110 at token 11, all 4 rows | 0.0110 at token 11, all 4 rows | 0.0110 at token 11, all 4 rows | 0.0104 at token 13, all 4 rows | 0.0000 |
| `[4-2048]` | 2 of 2, `A short summary of the history of the printing press:` | PASSED | 0.0086 at token 3, all 4 rows | 0.0086 at token 1, all 4 rows | 0.0131 at token 12, all 4 rows | 0.0107 at token 12, all 4 rows | 0.0000 |

Neuron vs the CPU BF16 reference (`test/ouro_reference.py`); token = generated token, row = batch row. No token over a bound. No argmax divergence.
Both cases passed with 2/2 prompts within the static bounds (two-way validation) and aggregate σ-ratio 0.9627 (`[1-2048]`) and 0.9505 (`[4-2048]`) ≤ 1.0 (`multi_prompt_logit_validation`); `agg_bc` PASS (`[1-2048]`) and FAIL (`[4-2048]`); printed per-prompt checks: `[1-2048]`: `pp_max_linf_1.5x` 2/2, `pp_max_linf_2.0x` 2/2, `pp_max_l2_1.5x` 2/2, `pp_max_l2_2.0x` 2/2, `pp_tok_linf_1.5x` 1/2, `pp_tok_linf_2.0x` 2/2, `pp_tok_l2_1.5x` 2/2, `pp_tok_l2_2.0x` 2/2, `pp_static_0.03` 2/2, `pp_static_0.05` 2/2; `[4-2048]`: each of the same ten checks 2/2.

Error against the FP32 reference, per prompt (max over the 16 generated tokens):

| Case | Prompt | CPU BF16 L-inf | Neuron L-inf | Neuron / CPU BF16 | CPU BF16 L2 | Neuron L2 | Neuron / CPU BF16 |
|---|---|---|---|---|---|---|---|
| `[1-2048]` | 1 of 2 | 0.0102 | 0.0085 | 0.83x | 0.0107 | 0.0112 | 1.05x |
| `[1-2048]` | 2 of 2 | 0.0118 | 0.0114 | 0.97x | 0.0161 | 0.0143 | 0.89x |
| `[4-2048]` | 1 of 2 | 0.0102 | 0.0074 | 0.73x | 0.0107 | 0.0099 | 0.93x |
| `[4-2048]` | 2 of 2 | 0.0118 | 0.0114 | 0.97x | 0.0161 | 0.0143 | 0.89x |

Prompt 1: CPU BF16 L-inf maximum at generated token 13, Neuron L-inf maximum at token 2. Prompt 2: every maximum at generated token 0.

#### Serve smoke

`/v1/completions`, on-device greedy sampling (`all_greedy`), 32 completion tokens; first 60 characters of each output as a Python string literal.

| Prompt | Output |
|---|---|
| `The capital of France is` | `' Paris.\n\n## Step 2: Evaluate the options\nA: London - This is'` |
| `1, 2, 3, 4, 5,` | `' 6, 7, 8, 9, 10, 11, 12, 13, 14,'` (the whole output) |
| `def fibonacci(n):` | `'\n    if n <= 0:\n        return "Input should be a positive i'` |

### Exceptions

- **Module tests not run on 0.29.**
- **Dropped: the two-node disaggregated (1P1D) logit and serving tests.**
  They depend on the plugin's shared multi-node test harness, which is
  outside this folder. The shipped integration test is single-node at the
  same TP degree (TP 8), with the same reference (`test/ouro_reference.py`,
  not the checkpoint's remote code). It uses the template's tolerance map;
  the port's disaggregated logit test used a tighter map (K5 0.015, K50
  0.04, K1000 0.06, All 0.065). The maxima of this run (K5 0.0113, K50
  0.0113, K1000 0.0131, All 0.0109) are within that map.
- **CPU-mode decode at TP 4.** The decode cases of `test/unit/` run at TP 4,
  because the torch fallback of the decode attention kernel supports one KV
  head per rank. `test_tiny_e2e.py` skips when `/dev/shm` has less than
  240 MiB free.
- Prefix caching (end to end), TP sizes other than 8 on device, context
  beyond 2048 tokens, sampled (non-greedy) requests, `early_exit_step`,
  speculative decoding and Trn3 are not tested.
