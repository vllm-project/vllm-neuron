# Contrib Model: Gemma 4 31B (text)

[Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it) is a dense decoder that repeats five sliding-window layers and one
global layer (60 layers, hidden size 5376, 32 query heads, vocabulary 262144). Sliding layers use head_dim 256 with 16 KV heads
and a 1024-token window. Global layers use head_dim 512 with 4 KV heads, no V projection (V = K), and partial RoPE. The model also
uses Q/K/V RMS norms, a per-layer scalar, four norms per layer, GeGLU, scaled embeddings, tied embeddings, and final logit
softcapping.

This folder serves the text decoder in BF16 with vLLM on AWS Trainium2 through the `vllm-neuron` plugin, on one `trn2.3xlarge`
(one Trainium2 chip, TP=4). Vision and audio towers are not implemented.

> **Note:** Contrib models are community contributions and may have limited testing or performance optimization compared to the
> officially-supported models under `vllm_neuron/model/`. The compatibility matrix below records exactly what was tested.

## Attribution

The model code is adapted from Gemma 4 work by Armin Agha-Ebrahim (GitHub: `arminagha1234`). Changes in this folder include the
single-chip TP=4 configuration, the vLLM Neuron 0.29 model interface, the paged KV-cache handling described in
[src/README.md](src/README.md), and the tests.

## Usage

Use a Neuron environment with `vllm-neuron` 0.29.0.1.2.0, for example the Neuron Deep Learning Container
`public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04`, then install this model.
The install adds a `vllm.general_plugins` entry point that registers Gemma 4 in every vLLM process:

```bash
pip install -e contrib/models/gemma4
```

The tested configuration runs at logical NeuronCore config 2 on an instance without EFA:

```bash
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_SKIP_EFA_AFFINITY=1
```

### Serve with the OpenAI-compatible server

```bash
vllm serve google/gemma-4-31B-it \
    --served-model-name gemma4 \
    --tensor-parallel-size 4 \
    --max-model-len 12288 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 512 \
    --gpu-memory-utilization 0.70 \
    --enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{
        "neuron_config": {
            "num_batched_tokens_buckets": [512],
            "num_seqs_buckets": [1, 4],
            "enable_structured_outputs": true,
            "on_device_sampling_config": {"all_greedy": true}
        }
    }'
```

Engine start, including compilation, took 323 s with `--max-num-seqs 1` and `"num_seqs_buckets": [1]`, and 460 to 467 s with the
configuration above (engine initialization time in the vLLM log, on a `trn2.3xlarge`). The settings:

- **`--tensor-parallel-size 4`**: the BF16 weights are about 62 GB. At TP=2 that is 31 GB per rank, which is more than the 24 GB
  of HBM of one logical core (96 GB over 4 cores at LNC=2). Other TP degrees were not tried.
- **`--max-num-batched-tokens 512` and `num_batched_tokens_buckets: [512]`**: prompts are prefilled in 512-token segments.
- **`--max-num-seqs 4` and `num_seqs_buckets: [1, 4]`**: up to four sequences per decode step, with one decode graph for a single
  sequence and one for four.
- **`--no-async-scheduling` and `enable_structured_outputs`**: with asynchronous scheduling on, a JSON-schema request returned
  HTTP 500 in our tests.
- **`all_greedy`**: on-device greedy sampling.

Then send a request:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gemma4",
        "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
        "max_tokens": 16
    }'
```

### Offline inference with the Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="google/gemma-4-31B-it",
    tensor_parallel_size=4,
    max_model_len=12288,
    max_num_seqs=4,
    max_num_batched_tokens=512,
    gpu_memory_utilization=0.70,
    enable_prefix_caching=True,
    async_scheduling=False,
    additional_config={
        "neuron_config": {
            "num_batched_tokens_buckets": [512],
            "num_seqs_buckets": [1, 4],
            "on_device_sampling_config": {"all_greedy": True},
        }
    },
)
out = llm.chat(
    [[{"role": "user", "content": "What is the capital of France?"}]],
    SamplingParams(temperature=0.0, max_tokens=32),
)
print(out[0].outputs[0].text)
```

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn2 (`trn2.3xlarge`, TP=4, LNC=2) | Working | Not tested |
| Trn3 | Not tested | Not tested |

Tested on the Neuron DLC `pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` as shipped (vLLM 0.29.0,
vLLM Neuron 0.29.0.1.2.0, transformers 5.17.0), with only this folder installed.

| Feature | Status |
|---|---|
| Text generation, chunked (segmented) prefill | ✅ |
| Strict JSON output (structured outputs) | ✅ with synchronous scheduling (HTTP 500 with asynchronous scheduling) |
| Prefix caching | ✅ |
| BF16 | ✅ |
| Tensor parallelism | ✅ TP=4 (other values not tested) |
| Several sequences per server (`--max-num-seqs` 4) | ✅ outputs identical to single-request decoding; 1.56x aggregate throughput at 4 clients (see Performance) |
| FP8 KV cache | ❌ the plugin rejects `--kv-cache-dtype fp8` with prefix caching because segmented prefill cannot read a non-packed FP8 cache, and the packed layout is not implemented for this model |
| FP8 weights (compressed-tensors checkpoints) | ❌ not implemented; see the TODO below |
| Vision and audio inputs | ❌ |

## Accuracy

Greedy tokens from the full model on Neuron were compared with the Hugging Face reference (transformers 5.17, CPU, BF16), teacher-forced
on the Neuron-generated sequence: the margin is the reference's best logit minus its logit for the Neuron token. Across 72 generated
tokens on three prompts, every Neuron token was the reference's argmax (margin 0.000).

| Prompt tokens | Covers | Argmax agreement |
|---|---|---|
| 5 | basic path | 24/24 |
| 881 | crosses one 512-token prefill segment | 24/24 |
| 2372 | crosses the 1024-token sliding window and several segments | 24/24 |

## Performance

Measured on one `trn2.3xlarge` (TP=4, LNC=2, BF16), SDK 2.33, vLLM Neuron 0.29.0.1.2.0, greedy decoding, synchronous scheduling, with the
server configuration shown under Usage. Prompts are random tokens (no prefix reuse); `c` is the number of concurrent clients. TPOT is the
time per output token after the first one.

**One client (`c=1`)**

| Input / output tokens | TTFT p50 | TPOT p50 | Decode tokens/s |
|---|---|---|---|
| 1024 / 128 | 0.82 s | 47.2 ms | 21.2 |
| 4096 / 128 | 3.30 s | 47.5 ms | 21.1 |
| 8192 / 256 | 6.60 s | 48.0 ms | 20.8 |

TPOT stayed between 47.1 and 48.0 ms for prompts from 85 to 11,395 tokens. Prefill ran at about 1.2k tokens/s. A strict JSON-schema request
had a TPOT of 48.9 ms against 47.7 ms without a constraint (60 integers, valid JSON).

**Several clients.** Greedy outputs of four requests decoded together were identical to the same requests decoded alone. Aggregate
decode throughput with 128 new tokens per client:

| Clients | Aggregate tokens/s | Per client |
|---|---|---|
| 1 | 19.9 | 19.9 |
| 2 | 26.3 | 13.2 |
| 4 | 31.1 | 7.8 |

Throughput grows less than linearly: the decode step took about 50 ms with one sequence and about 129 ms with four. The cause was not
investigated. The plugin prefills one sequence at a time (its log says Neuron only supports chunking prefills with batch size 1), and with
four clients the TTFT p50 was 2.1 s, 8.2 s and 16.4 s for 1k, 4k and 8k input tokens.

**Asynchronous scheduling.** With `--async-scheduling` and no structured output, TPOT for a single client was 38.9 ms (25.7 tokens/s)
against 48.8 ms with synchronous scheduling. With asynchronous scheduling on, a JSON-schema request returned HTTP 500.

**Strict JSON with a long prompt.** With a ~9.4k-token prompt, JSON-schema output and greedy decoding, 8 of 10 requests returned valid JSON.
In the two invalid ones the output continued with newlines and spaces after the first field until `max_tokens` was reached. For one of
them, a Hugging Face forward pass over the Neuron-generated tokens preferred the same token at every position that the grammar did not
force; at the point where the output stopped, its logit for a newline was 28.4 and for a comma 1.1. Mitigations were not tested.

## Example Checkpoints

- <https://huggingface.co/google/gemma-4-31B-it> (Apache-2.0; not gated)

## Testing

Install the package first, so that TP worker processes register the model:

```bash
pip install -e contrib/models/gemma4
```

CPU tests (no Neuron hardware; the NKI simulator covers the KV-cache kernels):

```bash
VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 pytest contrib/models/gemma4/test/unit/ -v
```

The integration test serves the checkpoint on a `trn2.3xlarge` and checks a factual answer, greedy determinism, a long prompt, and strict
JSON output:

```bash
NEURON_LOGICAL_NC_CONFIG=2 NEURON_SKIP_EFA_AFFINITY=1 pytest contrib/models/gemma4/test/integration/test_model.py \
    -v -s --timeout=7200
```

## TODO

- Serve pre-quantized FP8 checkpoints (compressed-tensors format). Today only BF16 weights are loaded.
- FP8 KV cache: implement the packed FP8 layout (write, gather and prefill) so that segmented prefill and prefix caching keep working.
- Find out why the decode step grows from about 50 ms with one sequence to about 129 ms with four.
- Test other TP degrees and Trn3.
