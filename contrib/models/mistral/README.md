# Contrib Model: Mistral-7B-v0.3 (Text)

[Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3) is a
7.2B-parameter dense decoder-only language model: 32 layers, hidden size
4096, grouped-query attention with 32 query heads and 8 KV heads, a SwiGLU
MLP of size 14336, RMSNorm, default RoPE with theta 1e6 and a 32768-token
vocabulary. Unlike v0.1 it has no sliding window, so every layer attends over
the full context.

This folder serves it with vLLM on AWS Trainium2 through the `vllm-neuron`
plugin, in BF16 with tensor parallelism. The model uses the plugin's shared
operators and NKI kernels (fused QKV, flash and segmented attention, fused
MLP) and is derived from the plugin's dense reference implementation; see
[src/README.md](src/README.md) for the architecture and files.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`), then install this folder:

```bash
pip install -e contrib/models/mistral
```

The install adds a `vllm.general_plugins` entry point that registers
`MistralForCausalLM` in every vLLM process, so `vllm serve` and `LLM(...)`
find it with no other setup. It replaces vLLM's GPU `MistralForCausalLM`
while installed.

### Checkpoint

Download without `consolidated.safetensors` (a second copy of the weights
under Mistral's own tensor names, which the plugin's loader would otherwise
read too) and use the local directory as the model path:

```bash
hf download mistralai/Mistral-7B-Instruct-v0.3 \
    --exclude consolidated.safetensors \
    --local-dir /data/Mistral-7B-Instruct-v0.3
```

### Serve with the OpenAI-compatible server

```bash
vllm serve /data/Mistral-7B-Instruct-v0.3 \
    --served-model-name Mistral-7B-Instruct-v0.3 \
    --tensor-parallel-size 4 \
    --max-model-len 2048 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 2048 \
    --no-enable-prefix-caching
```

Prefix caching (on by default in vLLM) needs segmented prefill on Neuron,
which the plugin enables only when `--max-num-batched-tokens` is one of 512,
1024, 2048, 4096 or 8192 and smaller than `--max-model-len`. Either turn it
off as above, or use sizes that meet both conditions, for example
`--max-model-len 4096 --max-num-batched-tokens 1024` (tested, see below).

Then send a request:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Mistral-7B-Instruct-v0.3",
        "messages": [{"role": "user", "content": "Name three prime numbers."}],
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/data/Mistral-7B-Instruct-v0.3",
    tensor_parallel_size=4,
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

To iterate without Neuron hardware, set `VLLM_NEURON_CPU_MODE=1` and pass
`enforce_eager=True`.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn2 (`trn2.3xlarge`, LNC=2) | Working: TP 2 and TP 4 | Not tested |
| Trn3                   | Not tested | Not tested |

Tested on the Neuron Deep Learning AMI (Ubuntu 24.04, SDK 2.33.0) with vLLM
0.29.0, neuronx-cc 2.28.7631.0, Python 3.12, `mistralai/Mistral-7B-Instruct-v0.3`.
CPU mode (`VLLM_NEURON_CPU_MODE=1`): unit, module and integration tests pass,
the latter with the real 7B checkpoint at TP 1 and TP 2.

| Feature | Status |
|---|---|
| BF16 | ✅ |
| Tensor parallelism | ✅ TP 2 and TP 4 on Trn2 |
| Prefix caching | ✅ (see Usage for the size rule) |
| Long context | ✅ to 14k input tokens at `--max-model-len 16384` |
| Sampling (greedy, temperature, top-p, seeds, stop sequences) | ✅ |
| TP 1 on Trn2 | ❌ did not work on vLLM Neuron 0.29.0.1.2.0: the fused decode MLP kernel runs out of SBUF at an intermediate size of 14336 per rank, so use TP 2 or more |
| Quantized checkpoints, `sliding_window` checkpoints (Mistral-7B-v0.1) | ❌ rejected by the factory |
| Sequence parallelism, data parallelism inside one server, decode context parallelism, FP8 KV cache, Eagle3 target | Present in the code, not run for this model |

## Example Checkpoints

- <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3>
- <https://huggingface.co/mistralai/Mistral-7B-v0.3>

Not supported: checkpoints that set `sliding_window`, such as
Mistral-7B-v0.1 (`sliding_window: 4096`), and quantized Mistral checkpoints.

## Testing

All tests read the checkpoint from `MISTRAL_MODEL` (a local directory, see
[Checkpoint](#checkpoint)). For CPU runs, build a tiny random-weight
checkpoint with the Mistral-7B-v0.3 layout (2 layers):

```bash
python contrib/models/mistral/test/unit/tiny_checkpoint.py /tmp/tiny-mistral
```

### Results on Trn2

trn2.3xlarge, BF16, `mistralai/Mistral-7B-Instruct-v0.3`:

| Test | TP | Result |
| ---- | -- | ------ |
| Integration (logits vs Hugging Face), batch 1 / 2048, batch 4 / 2048, batch 2 / 8192 | 4 | 3 of 3 passed, zero argmax divergence, worst top-5 error 0.014 (limit 0.03) |
| Integration, batch 1 / 2048 | 2 | Passed, zero divergence |
| Three-way logit validation, 8 prompts (plain text, chat template, code), 16 tokens each | 2, 4 | 8 of 8 passed at each TP, zero divergence |
| MLP module tests, prefill and decode | 1, 2, 4 | 10 of 10 passed; the 2 TP 1 decode cases are expected failures (see the compatibility matrix) |
| Needle-in-a-haystack at 7,390 and 13,943 input tokens | 4 | Passed |
| Serving checks: chat template, greedy determinism, seeded sampling, stop sequence, `max_tokens`, prefix-cache speed-up, 8 concurrent requests | 4 | 9 of 9 passed |

Every prompt passes the static top-k thresholds. The three-way aggregate
sigma ratio against the FP32 reference is about 1.4 to 1.5 (MLP module tests
alone: 0.8 to 1.1), so the extra error is outside the MLP; not traced further.

### Integration test

Serves the model through vLLM and validates its logits against a Hugging Face
reference. `MISTRAL_TP` sets the tensor parallel size (default 1 in CPU mode,
4 on device).

```bash
MISTRAL_MODEL=/data/Mistral-7B-Instruct-v0.3 MISTRAL_TP=4 \
    pytest contrib/models/mistral/test/integration/test_model.py -v -s --timeout=3600
```

CPU mode with the tiny checkpoint:

```bash
VLLM_NEURON_CPU_MODE=1 MISTRAL_MODEL=/tmp/tiny-mistral MISTRAL_TP=2 \
    pytest contrib/models/mistral/test/integration/test_model.py -v -s --timeout=1800
```

In CPU mode the argmax check allows 2 BF16 ULPs instead of 1: the random-weight
checkpoint has near-tied top logits, and per-rank BF16 rounding at TP 2 shifts
one of them by 2 ULPs. Top-k tolerances are the same in both modes and the
device runs keep the 1 ULP rule.

### Unit tests

Config parsing, factory checks, registration, and RoPE / RMSNorm against
Hugging Face transformers. No checkpoint or device needed:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/mistral/test/unit/ -v
```

### Module tests

The MLP, prefill and decode, against Hugging Face with real weights for one
layer at tensor parallel sizes 1, 2 and 4 (three-way comparison, see the
[module test guidelines](../../../docs/design/accuracy/module_test_guidelines.md)):

```bash
VLLM_NEURON_CPU_MODE=1 MISTRAL_MODEL=/tmp/tiny-mistral \
    pytest contrib/models/mistral/test/module/ -v -s --timeout=600
```

Drop `VLLM_NEURON_CPU_MODE=1` to run on a Neuron device, where the decode
cases at world size 1 are reported as expected failures (see the
compatibility matrix). There is no attention module test yet; attention is
covered by the integration test.
