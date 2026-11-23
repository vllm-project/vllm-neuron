# Contrib Model: OLMo-3 7B Instruct (Text)

[Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct) is a
7B-parameter dense decoder-only language model: 32 post-norm layers, hidden
size 4096, multi-head attention with 32 heads of size 128, a full-width
QK-norm, a SwiGLU MLP of size 11008 and a 100278-token vocabulary. Three of
every four layers use sliding-window attention (window 4096) with the default
RoPE; every fourth layer uses full attention with YaRN RoPE.

This folder serves it with vLLM on AWS Trainium2 through the `vllm-neuron`
plugin, from the BF16 checkpoint with tensor parallelism. See
[src/README.md](src/README.md) for the architecture, the numerics and the
sliding-window handling.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix and the
> validation report below record exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`), then install this folder:

```bash
pip install -e contrib/models/olmo3
```

The install adds a `vllm.general_plugins` entry point that registers
`Olmo3ForCausalLM` in every vLLM process (the API server, the engine core and
each tensor-parallel worker), so `vllm serve` and `LLM(...)` find it with no
other setup. vLLM maps `Olmo3ForCausalLM` to its Transformers backend by
default; this registration replaces that with the Neuron implementation while
the package is installed. `pip uninstall vllm-neuron-contrib-olmo3` removes it.

Two settings are specific to this model:

- **Compile at `-O3`.** Pass `--optimization-level 3` (`optimization_level=3`
  offline). The implementation is validated at neuronx-cc `-O3`; at the
  default `-O2` the compiled full model loses accuracy from layer 20 on (see
  [src/README.md](src/README.md#compiler-optimization-level)).
- **Use a TP degree of 2, 4, 8 or 16.** It must divide the 32 heads and leave
  an even number of query heads on each rank. The recorded runs use TP 16.

### Serve with the OpenAI-compatible server

```bash
vllm serve allenai/Olmo-3-7B-Instruct \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --tensor-parallel-size 16 \
    --optimization-level 3 \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --block-size 32 \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{"neuron_config": {"on_device_sampling_config": {"all_greedy": true}, "num_batched_tokens_buckets": [2048], "num_seqs_buckets": [4]}}'
```

This is the configuration of the recorded smoke run. It matches the batch-4
case of the integration test, so with a shared Neuron compile cache the
server reuses those graphs. The two bucket lists compile one prefill graph
and one decode graph instead of a power-of-two ladder of each; the `-O3`
compile of this model is slow, so this shortens the startup. Drop them to
compile the full ladder. `all_greedy` makes every request greedy; remove it
(and `--no-async-scheduling`) to serve sampled requests, a path that was not
run for this model.

Then send a request:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "allenai/Olmo-3-7B-Instruct",
        "prompt": "The color of the sky is",
        "max_tokens": 32,
        "temperature": 0
    }'
```

Do not pass the checkpoint's own `chat_template.jinja` through
`--chat-template`: on vLLM 0.29 the server renders that template once at
startup without a `tools` variable, the template's `tools | tojson` raises a
`TypeError`, and the server exits. Without `--chat-template` the server does
not run that startup render.

### Offline inference with the Python API

```python
from vllm import LLM, SamplingParams

# With `pip install -e contrib/models/olmo3` the architecture is registered
# in every vLLM process; no registration call is needed here.
llm = LLM(
    model="allenai/Olmo-3-7B-Instruct",
    tensor_parallel_size=16,
    dtype="bfloat16",
    optimization_level=3,
    max_model_len=2048,
    max_num_batched_tokens=2048,
    max_num_seqs=4,
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
`ModelRegistry.register_model("Olmo3ForCausalLM", Olmo3ForCausalLM)`. The
engine core and the TP workers inherit that registration only under the
default `fork` start method, so leave `VLLM_WORKER_MULTIPROC_METHOD` unset.

To iterate without Neuron hardware, set `VLLM_NEURON_CPU_MODE=1` and pass
`enforce_eager=True`; `test/unit/test_olmo3_tiny_e2e.py` shows a complete
CPU-mode configuration.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0, vLLM 0.29.0) | 2.32 and earlier |
| ---------------------- | -------------------------------------------- | ---------------- |
| Trn2 (`trn2.48xlarge`) | TP 16: see the [validation report](#validation-report) | Not tested |
| Trn3                   | Not tested | Not tested |

| Feature | Status |
|---|---|
| BF16 checkpoint | Supported (other checkpoint dtypes are rejected by the factory) |
| Tensor parallelism | TP 16 recorded; TP 2, 4 and 8 are legal but were not run on device |
| Context length | 2048 recorded. Up to 4096 the sliding window masks nothing; beyond 4096 the window is applied by the attention mask, CPU-tested only |
| Prefix caching / segmented prefill | Not tested (every recorded run disables prefix caching) |
| Compiler level | `-O3` required (see Usage) |
| Speculative decoding, quantized checkpoints, data parallelism | Not tested |
| Performance | Not measured. Activations are FP32 and the NKI projection and attention kernels are not used, so expect lower throughput than a BF16 kernel-based model of the same size |

## Example Checkpoints

- <https://huggingface.co/allenai/Olmo-3-7B-Instruct> — the tests pin revision
  `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`.

Other OLMo-3 checkpoints with the same `Olmo3ForCausalLM` layout (BF16, no
attention bias, `sliding_attention` / `full_attention` layer types, default
RoPE on the sliding-window layers) should load, but none was run.

## Testing

### Unit tests (CPU mode)

No Neuron hardware and no checkpoint download are needed. The tests build a
tiny random-weight checkpoint with the OLMo-3 layout in a temporary
directory:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/olmo3/test/unit/ -v -s --timeout=1800
```

| File | Covers |
|---|---|
| `test_olmo3_config.py` | Plugin entry point and vLLM model protocol, config parse (transformers 5 `rope_parameters`, raw `config.json`), RoPE tables on the meta device, factory and TP-degree guards, default and YaRN RoPE against HF over 65,536 positions, weight-map coverage against the HF model, TP 16 weight-loader shards and vocabulary padding |
| `test_olmo3_layer_cpu.py` | One sliding-window and one full-attention decoder layer at TP 2, prefill, three-way against HF FP32 / BF16 with an active window (16 tokens); the same on the real checkpoint (window 48) when `OLMO3_CHECKPOINT_PATH` is set |
| `test_olmo3_tiny_e2e.py` | The full vLLM pipeline at TP 2: load, prefill, 8 decode steps past the 16-token window, on-device greedy sampling, full-vocabulary raw logits against HF teacher forcing |

Set `OLMO3_CHECKPOINT_PATH` to a local snapshot of the pinned revision to
also run the real-checkpoint layer cases; they read one layer's weights.

### Integration test (Neuron)

Serves the model through vLLM at TP 16 on one `trn2.48xlarge` and validates
its logits against a HuggingFace reference (batch 1 and batch 4, 2048-token
model length, 16 tokens per prompt, the template's tolerance map). It needs
16 NeuronCores; `-O3` compiles take a long time, so allow a generous timeout:

```bash
OLMO3_CHECKPOINT_PATH=/path/to/Olmo-3-7B-Instruct \
pytest contrib/models/olmo3/test/integration/test_model.py -v -s --timeout=14400
```

Without `OLMO3_CHECKPOINT_PATH` the test downloads the pinned revision.

## Validation report

### vLLM Neuron 0.29.0.1.2.0 (2026-11-19)

#### Environment

| Image | Instance | TP | vLLM | vllm-neuron | torch | transformers | neuronx-cc |
|---|---|---|---|---|---|---|---|
| `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` | `trn2.48xlarge` (LNC 2), 1 node | 16 | 0.29.0 | 0.29.0.1.2.0 | 2.13.0 | 5.17.0 | 2.28.7631.0+8f7b0cf5 |

#### Results

| Test | Configuration | Result |
|---|---|---|
| CPU unit tests, `test/unit/` | `VLLM_NEURON_CPU_MODE=1`, `OLMO3_CHECKPOINT_PATH` set (the two real-checkpoint layer cases ran) | 13 passed |
| Integration logit test, `test_model_accuracy[1-2048]` | batch 1, max model length 2,048, TP 16, `-O3` | PASSED (295.60 s) |
| Integration logit test, `test_model_accuracy[4-2048]` | batch 4, max model length 2,048, TP 16, `-O3` | PASSED (319.93 s) |
| `vllm serve` smoke | the serve command under Usage | Passed, 3 of 3 prompts |

#### Logit validation

| Case | Prompt | Overall Status | Max divergence (1 ULP) | Max K5 (0.03) | Max K50 (0.05) | Max K1000 (0.08) | Max All (0.09) |
|---|---|---|---|---|---|---|---|
| `[1-2048]` | 1/2, `The color of the sky is` | PASSED | 0.0000 | 0.0075 | 0.0079 | 0.0079 | 0.0079 |
| `[1-2048]` | 2/2, `A short summary of the history of the printing press:` | PASSED | 0.0000 | 0.0082 | 0.0090 | 0.0090 | 0.0090 |
| `[4-2048]` | 1/2, `The color of the sky is` | PASSED | 0.0000 | 0.0079 | 0.0091 | 0.0091 | 0.0091 |
| `[4-2048]` | 2/2, `A short summary of the history of the printing press:` | PASSED | 0.0000 | 0.0085 | 0.0090 | 0.0090 | 0.0091 |

Two-way validation 2/2 prompts in both cases; no bound exceeded.
`[1-2048]`: aggregate σ-ratio 0.8218 (≤ 1.0 = pass), `agg_bc` FAIL; `pp_max_l2_1.5x`, `pp_max_l2_2.0x`, `pp_max_linf_1.5x`, `pp_max_linf_2.0x`, `pp_static_0.03`, `pp_static_0.05`, `pp_tok_l2_1.5x`, `pp_tok_l2_2.0x`, `pp_tok_linf_1.5x`, `pp_tok_linf_2.0x` 2/2 each.
`[4-2048]`: aggregate σ-ratio 0.8104 (≤ 1.0 = pass), `agg_bc` FAIL; `pp_max_l2_1.5x`, `pp_max_l2_2.0x`, `pp_max_linf_1.5x`, `pp_max_linf_2.0x`, `pp_static_0.03`, `pp_static_0.05`, `pp_tok_l2_1.5x`, `pp_tok_l2_2.0x`, `pp_tok_linf_1.5x`, `pp_tok_linf_2.0x` 2/2 each.
16 generated tokens per prompt (batch 4: the prompt replicated to 4 rows, 64 tokens) vs the HF BF16 reference (FP32 baseline); no argmax divergence.

Error against the FP32 reference, per prompt (max over the 16 generated tokens; ratio = Neuron max / CPU BF16 max):

| Case | Prompt | CPU BF16 L-inf | Neuron L-inf | Neuron / CPU BF16 | CPU BF16 L2 | Neuron L2 | Neuron / CPU BF16 |
|---|---|---|---|---|---|---|---|
| `[1-2048]` | 1/2 | 0.0087 | 0.0046 | 0.53x | 0.0203 | 0.0184 | 0.91x |
| `[1-2048]` | 2/2 | 0.0090 | 0.0050 | 0.56x | 0.0220 | 0.0200 | 0.91x |
| `[4-2048]` | 1/2 | 0.0087 | 0.0054 | 0.62x | 0.0203 | 0.0183 | 0.90x |
| `[4-2048]` | 2/2 | 0.0090 | 0.0045 | 0.50x | 0.0220 | 0.0203 | 0.92x |

#### Serve smoke

`/v1/completions`, max_tokens 32, temperature 0; 32 tokens returned per prompt. Batched request (all three, max_tokens 16): a prefix of the same text.

| Prompt | Start of the completion (first 60 characters) |
|---|---|
| `The capital of France is` | ` ____. A) Paris B) London C) Rome D) Berlin\nAnswer\nA\nWhich o` |
| `1, 2, 3, 4, 5,` | ` 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16` (the whole output) |
| `def fibonacci(n):` | `  \n    # Initialize first two Fibonacci numbers \n    if n <=  ` |

### Exceptions

- **Dropped: the two-node disaggregated test.** It depends on the plugin's
  shared multi-node test harness, which is outside this folder. The shipped
  integration test is single-node and colocated at the same TP degree.
- **Dropped: the device module and diagnostic tests** (per-component
  three-way tests at TP 16 and the full-model layer-by-layer probes used to
  locate the `-O2` error). They need the shared test utilities and real
  weights on device; the integration test covers the full model.
- **Untested:** contexts beyond 2048 tokens on device (the window is active
  only beyond 4096), prefix caching, TP other than 16 on device, Trn3.
