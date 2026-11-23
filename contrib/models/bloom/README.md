# Contrib Model: Bloom (bloomz-560m, Text)

[bigscience/bloomz-560m](https://huggingface.co/bigscience/bloomz-560m) is a
559M-parameter multilingual decoder-only language model, the
instruction-tuned (xP3) version of `bigscience/bloom-560m`: 24 layers, hidden
size 1024, multi-head attention with 16 heads, ALiBi position biases instead of
RoPE, LayerNorm with bias (also directly after the word embedding), a
non-gated GeLU MLP of size 4096 and a 250,880-id vocabulary with tied
embeddings.

This folder serves it with vLLM on AWS Trainium2 through the `vllm-neuron`
plugin, in BF16 with tensor parallelism. The plugin's attention kernels have no
bias input, so the implementation encodes ALiBi in padding lanes of the head
dimension, where the kernels compute it exactly; see
[src/README.md](src/README.md) for the architecture, the ALiBi encoding and the
feature status.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix and the
> validation report below record exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`), then install this folder:

```bash
pip install -e contrib/models/bloom
```

The install adds a `vllm.general_plugins` entry point that registers
`BloomForCausalLM` in every vLLM process, so `vllm serve` and `LLM(...)` find
it with no other setup, at any tensor parallel size. It replaces vLLM's GPU
`BloomForCausalLM` while installed. `pip uninstall vllm-neuron-contrib-bloom`
removes the registration.

### Checkpoint

Download the pinned revision (`*.json` and `model.safetensors`, about 1.1 GB;
the repository also has a duplicate `pytorch_model.bin` and training logs that
are not needed) and use the local directory as the model path:

```bash
hf download bigscience/bloomz-560m \
    --revision a2845d7e13dd12efae154a9f1c63fcc2e0cc4b05 \
    --include "*.json" "*.safetensors" \
    --local-dir /data/bloomz-560m
```

### Serve with the OpenAI-compatible server

```bash
vllm serve /data/bloomz-560m \
    --served-model-name bloomz-560m \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --additional-config '{"neuron_config": {"quantization": "bf16"}}'
```

The checkpoint stores FP16 weights; the implementation computes in BF16, and
`--dtype bfloat16` with `"quantization": "bf16"` is the configuration that was
tested. Prompts are padded to a token bucket, and with sequence-parallel
prefill the bucket must be larger than the tensor parallel size and divisible
by it, which every power-of-2 bucket at TP 2 is.

Then send a request:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "bloomz-560m",
        "prompt": "Translate to English: Je suis heureux.",
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

```python
from vllm import LLM, SamplingParams

# With `pip install -e contrib/models/bloom`, the entry point registers the
# architecture in this process and in the engine and worker processes.
llm = LLM(
    model="/data/bloomz-560m",
    tensor_parallel_size=2,
    dtype="bfloat16",
    max_model_len=2048,
    max_num_batched_tokens=2048,
    max_num_seqs=4,
    enable_prefix_caching=False,
    additional_config={"neuron_config": {"quantization": "bf16"}},
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The capital of France is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

Without the install, register the factory from your own script before you
construct `LLM(...)` (option 1 in [contrib/README.md](../../README.md)), as
`test/integration/test_model.py` does:

```python
from vllm import ModelRegistry

# Your copy of contrib/models/bloom/src/, under whatever name you gave it.
from your_package.factory import BloomForCausalLM

ModelRegistry.register_model("BloomForCausalLM", BloomForCausalLM)
```

That registration reaches the engine-core and worker processes only because
they are `fork`ed from your process (the default). Because vLLM ships its own
`BloomForCausalLM`, a process that misses it does not report an unknown
architecture: it resolves vLLM's GPU class and then fails in the Neuron model
runner with an `AttributeError` on `from_configs`. Under
`VLLM_WORKER_MULTIPROC_METHOD=spawn`, and for `vllm serve`, use the install
above.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1`. See the top-level `README.md` and `AGENTS.md` for the
full list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn2 (`trn2.48xlarge`) | TP 2: see the [validation report](#validation-report) | Not tested |
| Trn3                   | Not tested | Not tested |

Software: vLLM Neuron 0.29.0.1.2.0 with vLLM 0.29.0, transformers 5.17.0,
Python 3.14, the default native route (`neuron_native`). In CPU mode
(`VLLM_NEURON_CPU_MODE=1`) on the same software, the unit and module tests
pass (see [Testing](#testing)).

| Feature | Status |
|---|---|
| BF16 (FP16 checkpoint) | Tested configuration |
| Tensor parallelism | TP 2 tested; TP 1, 4 and 8 are legal but not run on 0.29; TP 16 is rejected (the decode kernel needs an even head count per rank) |
| Max model length | 2048 tested; up to 16384 accepted (the limit of the exact ALiBi position encoding) |
| Prefix caching / segmented prefill | Code path present, not tested: disable prefix caching as above |
| Speculative decoding, quantized checkpoints, FP8 KV cache | Not supported or not tested |

## Example Checkpoints

- <https://huggingface.co/bigscience/bloomz-560m> at revision
  `a2845d7e13dd12efae154a9f1c63fcc2e0cc4b05` — the checkpoint the tests pin.

Other Bloom checkpoints share the architecture but were not run. The factory
rejects configurations the implementation does not handle:
`apply_residual_connection_post_layernorm=True`, `slow_but_exact=True` with
`pretraining_tp > 1`, and a head dimension above 122 (for example
`bigscience/bloom-1b7`, `bloom-7b1` and `bloom`, whose head dimension is 128).

## Testing

All tests read the checkpoint from `BLOOM_CHECKPOINT_PATH` (a local directory
of the pinned revision, see [Checkpoint](#checkpoint)). Without it they
download the pinned revision from the Hugging Face Hub. The tests reach the
implementation through `test/_local_src.py`, which loads `src/` by file
location under the module name `contrib_bloom_src`, so they need neither the
install nor the repository root on `sys.path`.

### Integration test

Serves the model through vLLM on Neuron at TP 2 on one node (2 NeuronCores)
with the offline `LLM(...)` API, and validates the raw logits of 16 greedy
tokens for each of 10 prompts against a Hugging Face reference (FP32 baseline
and a BF16 pass teacher-forced along the FP32 sequence) with
`multi_prompt_logit_validation`, at batch 1 and batch 4 with a 2048-token
model length:

```bash
BLOOM_CHECKPOINT_PATH=/data/bloomz-560m \
    pytest contrib/models/bloom/test/integration/test_model.py \
    -v -s --timeout=7200
```

The tolerance map is the template's BF16 dense map: argmax divergence bound
`1e-5`, top-K logit bounds 0.03 (K=5), 0.05 (K=50), 0.08 (K=1000) and 0.09
(all). Three model-specific details are in the test:

- The Hugging Face Bloom classes do not support SDPA, so the reference loads
  with eager attention.
- The model scores 250,880 ids but the tokenizer has 250,680. The test
  requests the logits of every id with `detokenize=False`, because the
  plugin's generic helper sizes each row from the tokenizer and vLLM cannot
  detokenize the 200 padding ids.
- `ignore_eos=True`: bloomz often ends a sentence with EOS early, and the
  reference always decodes 16 tokens.

### Unit tests

`test/unit/test_loading.py` checks the registration contract (the
architecture string, the exported factory, vLLM's text-generation model
protocol and the entry point in `src/plugin.py`), the config parse and the
validation errors, the ALiBi slopes against the Hugging Face ones, the
traceable GeLU of the CPU MLP path against `torch.nn.functional.gelu`, the
checkpoint mapping coverage, the QKV / output-projection / bias loader
transforms (including the exact ALiBi identity) for both TP 2 ranks, the
loaders built on the meta device, and a full TP 2 model load on CPU:

```bash
VLLM_NEURON_CPU_MODE=1 BLOOM_CHECKPOINT_PATH=/data/bloomz-560m \
    pytest contrib/models/bloom/test/unit/ -v
```

### Module tests

`test/module/test_components.py` compares layer-0 attention (256-token
prefill, then one decode step through the decode kernel's torch fallback) and
the MLP (prefill and decode, through the CPU torch path) at TP 2 with the
Hugging Face Bloom modules, using
the three-way pattern of the
[module test guidelines](../../../docs/design/accuracy/module_test_guidelines.md).
They run in CPU mode only:

```bash
VLLM_NEURON_CPU_MODE=1 BLOOM_CHECKPOINT_PATH=/data/bloomz-560m \
    pytest contrib/models/bloom/test/module/ -v -s --timeout=1800
```

## Validation report

### vLLM Neuron 0.29.0.1.2.0 (2026-11-19)

#### Environment

| Image | Instance | TP | vLLM | vllm-neuron | torch | transformers | neuronx-cc |
|---|---|---|---|---|---|---|---|
| `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` | `trn2.48xlarge` (LNC 2), 1 node | 2 | 0.29.0 | 0.29.0.1.2.0 | 2.13.0 | 5.17.0 | 2.28.7631.0+8f7b0cf5 |

Checkpoint: `bigscience/bloomz-560m` at revision
`a2845d7e13dd12efae154a9f1c63fcc2e0cc4b05`.

#### Results

| Test | Configuration | Result |
|---|---|---|
| CPU unit tests, `test/unit/` | `VLLM_NEURON_CPU_MODE=1`, `BLOOM_CHECKPOINT_PATH` set | 9 passed |
| Integration logit test, `test_model_accuracy[1-2048]` | batch 1, max model length 2,048, TP 2, `VLLM_NEURON_BACKEND=neuron_native` | PASSED (697.78 s) |
| Integration logit test, `test_model_accuracy[4-2048]` | batch 4, max model length 2,048, TP 2, `VLLM_NEURON_BACKEND=neuron_native` | PASSED (1,951.42 s) |
| `vllm serve` smoke | the serve command under Usage with model `bigscience/bloomz-560m` and no `--served-model-name`, `VLLM_NEURON_BACKEND=neuron_native` | Passed, 3 of 3 prompts |

Not part of the 0.29 device run: module tests (`test/module/`).

#### Logit validation

`test_model_accuracy[1-2048]`:

| Prompt | Overall Status | Max divergence (1 ULP) | Tokens with argmax divergence | Max K5 (0.03) | Max K50 (0.05) | Max K1000 (0.08) | Max All (0.09) |
|---|---|---|---|---|---|---|---|
| 1/10, `The capital of France is Paris, …` | PASSED | 2.0000 at token 11 (≤ 1 ULP) | 10 of 16 | 0.0068 | 0.0068 | 0.0068 | 0.0068 |
| 2/10, `In machine learning, …` | PASSED | 2.0000 at token 2 (≤ 1 ULP) | 3 of 16 | 0.0062 | 0.0062 | 0.0062 | 0.0062 |
| 3/10, `The theory of relativity …` | PASSED | 2.0000 at token 11 (≤ 1 ULP) | 9 of 16 | 0.0054 | 0.0054 | 0.0054 | 0.0054 |
| 4/10, `Python is a high-level …` | PASSED | 2.0000 at token 2 (≤ 1 ULP) | 11 of 16 | 0.0060 | 0.0060 | 0.0060 | 0.0060 |
| 5/10, `The Amazon rainforest …` | PASSED | 2.0000 at token 10 (≤ 1 ULP) | 9 of 16 | 0.0067 | 0.0067 | 0.0067 | 0.0067 |
| 6/10, `Quantum computing …` | PASSED | 0.0000 | 5 of 16 | 0.0049 | 0.0049 | 0.0049 | 0.0051 |
| 7/10, `The Great Wall of China …` | PASSED | 0.0000 | 7 of 16 | 0.0141 | 0.0141 | 0.0141 | 0.0141 |
| 8/10, `Climate change …` | PASSED | 2.0000 at token 2 (≤ 1 ULP) | 7 of 16 | 0.0067 | 0.0067 | 0.0067 | 0.0067 |
| 9/10, `The human genome …` | PASSED | 2.0000 at token 1 (≤ 1 ULP) | 8 of 16 | 0.0069 | 0.0069 | 0.0069 | 0.0069 |
| 10/10, `Artificial intelligence …` | PASSED | 0.0000 | 6 of 16 | 0.0070 | 0.0070 | 0.0070 | 0.0070 |

Test passed on two-way validation 10/10 prompts and on aggregate σ-ratio 0.9205 ≤ 1.0 (`multi_prompt_logit_validation`); `agg_bc` FAIL; `pp_max_l2_1.5x` 8/10, `pp_max_l2_2.0x` 10/10, `pp_max_linf_1.5x` 10/10, `pp_max_linf_2.0x` 10/10, `pp_static_0.03` 10/10, `pp_static_0.05` 10/10, `pp_tok_l2_1.5x` 8/10, `pp_tok_l2_2.0x` 10/10, `pp_tok_linf_1.5x` 9/10, `pp_tok_linf_2.0x` 10/10.
`pp_max_l2_1.5x` and `pp_tok_l2_1.5x` miss prompts 2 and 8, both at generated token 1: Neuron L2 0.0041 vs CPU BF16 L2 0.0023 (1.78x) and 0.0043 vs 0.0026 (1.69x). `pp_tok_linf_1.5x` misses prompt 4: Neuron L-inf 0.0045 vs CPU BF16 L-inf 0.0030 (1.51x) at generated token 15.

`test_model_accuracy[4-2048]`:

| Prompt | Overall Status | Max divergence (1 ULP) | Tokens with argmax divergence | Max K5 (0.03) | Max K50 (0.05) | Max K1000 (0.08) | Max All (0.09) |
|---|---|---|---|---|---|---|---|
| 1/10, `The capital of France is Paris, …` | PASSED | 2.0000 at token 11 (≤ 1 ULP) | 32 of 64 | 0.0068 | 0.0068 | 0.0068 | 0.0068 |
| 2/10, `In machine learning, …` | PASSED | 2.0000 at token 2 (≤ 1 ULP) | 12 of 64 | 0.0062 | 0.0062 | 0.0062 | 0.0062 |
| 3/10, `The theory of relativity …` | PASSED | 2.0000 at token 11 (≤ 1 ULP) | 32 of 64 | 0.0054 | 0.0054 | 0.0054 | 0.0054 |
| 4/10, `Python is a high-level …` | PASSED | 2.0000 at token 2 (≤ 1 ULP) | 44 of 64 | 0.0060 | 0.0060 | 0.0060 | 0.0060 |
| 5/10, `The Amazon rainforest …` | PASSED | 2.0000 at token 10 (≤ 1 ULP) | 32 of 64 | 0.0067 | 0.0067 | 0.0067 | 0.0067 |
| 6/10, `Quantum computing …` | PASSED | 0.0000 | 20 of 64 | 0.0049 | 0.0049 | 0.0049 | 0.0051 |
| 7/10, `The Great Wall of China …` | PASSED | 0.0000 | 28 of 64 | 0.0141 | 0.0141 | 0.0141 | 0.0141 |
| 8/10, `Climate change …` | PASSED | 2.0000 at token 2 (≤ 1 ULP) | 28 of 64 | 0.0067 | 0.0067 | 0.0067 | 0.0067 |
| 9/10, `The human genome …` | PASSED | 2.0000 at token 1 (≤ 1 ULP) | 32 of 64 | 0.0069 | 0.0069 | 0.0069 | 0.0069 |
| 10/10, `Artificial intelligence …` | PASSED | 0.0000 | 24 of 64 | 0.0070 | 0.0070 | 0.0070 | 0.0070 |

Test passed on two-way validation 10/10 prompts and on aggregate σ-ratio 0.9198 ≤ 1.0 (`multi_prompt_logit_validation`); `agg_bc` FAIL; `pp_max_l2_1.5x` 8/10, `pp_max_l2_2.0x` 10/10, `pp_max_linf_1.5x` 10/10, `pp_max_linf_2.0x` 10/10, `pp_static_0.03` 10/10, `pp_static_0.05` 10/10, `pp_tok_l2_1.5x` 8/10, `pp_tok_l2_2.0x` 10/10, `pp_tok_linf_1.5x` 9/10, `pp_tok_linf_2.0x` 10/10.
The same prompts miss the same checks, with the same values in batch row 0.

16 generated tokens per prompt (4 batch rows of 16 in `[4-2048]`) vs the HF BF16 reference (FP32 baseline); prompts in the order of `PROMPTS` in `test/integration/test_model.py`; token = generated token, the maxima of `[4-2048]` are in batch row 0. Divergence = the gap between the Neuron logits of the Neuron argmax and the reference argmax, at a token where the two differ; bound 1 BF16 ULP of those logits.

Error against the FP32 reference, per prompt (max over the 16 generated tokens; `[1-2048]` and batch row 0 of `[4-2048]` print the same values):

| Prompt | CPU BF16 L-inf | Neuron L-inf | Neuron / CPU BF16 | CPU BF16 L2 | Neuron L2 | Neuron / CPU BF16 |
|---|---|---|---|---|---|---|
| 1/10 | 0.0081 | 0.0050 | 0.61x | 0.0049 | 0.0024 | 0.50x |
| 2/10 | 0.0048 | 0.0066 | 1.39x | 0.0023 | 0.0041 | 1.78x |
| 3/10 | 0.0058 | 0.0049 | 0.85x | 0.0032 | 0.0026 | 0.81x |
| 4/10 | 0.0056 | 0.0052 | 0.93x | 0.0030 | 0.0028 | 0.92x |
| 5/10 | 0.0070 | 0.0064 | 0.92x | 0.0039 | 0.0035 | 0.90x |
| 6/10 | 0.0055 | 0.0052 | 0.95x | 0.0025 | 0.0027 | 1.11x |
| 7/10 | 0.0124 | 0.0048 | 0.39x | 0.0089 | 0.0027 | 0.30x |
| 8/10 | 0.0057 | 0.0074 | 1.31x | 0.0026 | 0.0043 | 1.69x |
| 9/10 | 0.0063 | 0.0050 | 0.79x | 0.0033 | 0.0028 | 0.86x |
| 10/10 | 0.0117 | 0.0051 | 0.44x | 0.0079 | 0.0025 | 0.32x |

#### Serve smoke

`/v1/completions`, max_tokens 32, temperature 0; completion tokens 2, 32 and 32. Batched request (all three, max_tokens 16): a prefix of the same text for each prompt.

| Prompt | Start of the completion (first 60 characters) |
|---|---|
| `The capital of France is` | ` Paris` (the whole output, 2 completion tokens) |
| `1, 2, 3, 4, 5,` | ` 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,` (the whole output) |
| `def fibonacci(n):` | `\n    """Compute the fibonacci number of a given number.\n    ` |

### Exceptions

- The port's GSM8K accuracy and performance comparison tests are not shipped:
  they depend on the plugin repository's shared server, benchmark and
  result-publishing test helpers.
- The port's device component tests and its full-model serving and logit test
  ran on two nodes with one rank each, through a multi-node launcher in the
  plugin repository's test tree. They are not shipped; the integration test
  runs the same TP 2 logit validation on one node with the offline API.
- No performance figures are stated.
