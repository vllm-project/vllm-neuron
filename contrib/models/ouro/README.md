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
| Context length | Maximum model length 9216 in the device integration test, with prompts of 8000-8192 tokens and 16 generated tokens. Longer contexts (the checkpoint allows 65536) are not validated |
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

The validation report for the configuration in this README (TP 8 on one
`trn2.48xlarge`, maximum model length 9216, batch 1 and 4, prompts of
8000-8192 tokens) is pending.
