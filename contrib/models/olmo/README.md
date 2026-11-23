# OLMo (Text)

Serve the original OLMo text model (`allenai/OLMo-1B-hf`, architecture
`OlmoForCausalLM`) with vLLM on AWS Trainium through the `vllm-neuron` plugin.
This is the first OLMo release in the transformers format, not OLMo 2 (see
`contrib/models/olmo2` for that, a separate architecture). The implementation
is in `src/`, and `src/README.md` describes the architecture and the
per-feature status.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix and the
> validation report below record exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`), then install this folder as a vLLM plugin (Option 3 in
[contrib/README.md](../../README.md)):

```bash
pip install -e contrib/models/olmo
```

The install adds a `vllm.general_plugins` entry point (`olmo_neuron`). vLLM
calls it in every process it starts — the API server, the engine core and each
tensor-parallel worker — so `vllm serve` and `LLM(...)` with
`tensor_parallel_size` 8 find the model with no other setup. vLLM ships its own
`OlmoForCausalLM` entry; the plugin replaces it with this Neuron
implementation. `pip uninstall vllm-neuron-contrib-olmo` removes the
registration.

### Serve with the OpenAI-compatible server

```bash
vllm serve allenai/OLMo-1B-hf \
    --revision aee7752d9c08ee4775e9b0091426d8410e8f6a89 \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --block-size 32 \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{"neuron_config": {"quantization": "bf16", "on_device_sampling_config": {"all_greedy": true}}}'
```

Then send a request:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "allenai/OLMo-1B-hf",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

With the package installed, `LLM(...)` resolves the architecture by itself:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="allenai/OLMo-1B-hf",
    revision="aee7752d9c08ee4775e9b0091426d8410e8f6a89",
    tensor_parallel_size=8,
    dtype="bfloat16",
    max_model_len=2048,
    max_num_batched_tokens=2048,
    max_num_seqs=4,
    block_size=32,
    enable_prefix_caching=False,
    additional_config={
        "neuron_config": {
            "quantization": "bf16",
            "on_device_sampling_config": {"all_greedy": True},
        }
    },
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The color of the sky is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

Without the install, register the factory from your script before you build
the engine (Option 1 in `contrib/README.md`): copy `src/` into your
application and call
`ModelRegistry.register_model("OlmoForCausalLM", OlmoForCausalLM)` with the
factory from `src/factory.py`. The engine-core and worker processes inherit
that registration only under the default `fork` start method, so do not set
`VLLM_WORKER_MULTIPROC_METHOD=spawn` on that path. The integration test uses
this option.

`quantization: "bf16"` is not optional for this model: the published
checkpoint stores FP32 weights and the implementation computes in BF16.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1`. See the top-level `README.md` and `AGENTS.md` for the
full list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

| Instance / version | vLLM Neuron 0.29.0.1.2.0 (Neuron SDK 2.33) | Other versions |
| ------------------ | ------------------------------------------ | -------------- |
| Trn2 (trn2.48xlarge) | TP 8 — see the Validation report         | Not tested     |
| Trn3                 | Not tested                               | Not tested     |

## Example Checkpoints

- <https://huggingface.co/allenai/OLMo-1B-hf> — the tests pin revision
  `aee7752d9c08ee4775e9b0091426d8410e8f6a89`.

Other `OlmoForCausalLM` checkpoints have the same architecture, but none was
run. The implementation rejects a checkpoint that sets `clip_qkv`, an
attention bias or `rope_scaling`, and a head count that does not give an even
number of heads per rank at the chosen TP size.

## Testing

The integration test serves the model through vLLM on Neuron and validates its
raw logits against a HuggingFace reference (batch 1 and batch 4, 2,048-token
model length). It needs 8 NeuronCores on one node. Set `OLMO_CHECKPOINT_PATH`
to a local copy of the pinned revision, or let the test download it (about
4.7 GB):

```bash
pytest contrib/models/olmo/test/integration/test_model.py \
    -v -s --timeout=7200
```

### Unit tests

The unit tests run in CPU mode with no Neuron hardware, in well under a
minute:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/olmo/test/unit/ -v
```

- `test_loading.py`: vLLM's model protocol, the plugin entry point, the config
  parse, the factory and TP-size checks, the checkpoint mapping coverage and
  the TP 8 weight-loader shards.
- `test_components.py`: the non-parametric LayerNorm and RoPE against HF.
- `test_tiny_e2e.py`: a random-weight tiny OLMo (FP32 checkpoint, tied
  embeddings, 4 KV heads per rank) served through `LLM(...)` at TP 2, through
  the NKI kernels in the simulator and through the torch fallback, compared
  with HF `OlmoForCausalLM`.

The unit tests never download the weights. They fetch the config and tokenizer
JSON files of the pinned revision and read the checkpoint's safetensors header;
set `OLMO_CHECKPOINT_PATH` to run them offline.

All tests reach the implementation through `test/_local_src.py`, which loads
`src/` by file location under the unique module name `contrib_olmo_src`, so the
folder keeps working when it is copied into an application. `test/conftest.py`
puts `test/` on `sys.path` (for `_local_src` and `olmo_test_utils`) and shuts
down the plugin's worker pool at the end of the session.

## Validation report

### vLLM Neuron 0.29.0.1.2.0 (2026-11-19)

#### Environment

| Image | Instance | TP | vLLM | vllm-neuron | torch | transformers | neuronx-cc |
|---|---|---|---|---|---|---|---|
| `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` | `trn2.48xlarge` (LNC 2), 1 node | 8 | 0.29.0 | 0.29.0.1.2.0 | 2.13.0 | 5.17.0 | 2.28.7631.0+8f7b0cf5 |

#### Results

| Test | Configuration | Result |
|---|---|---|
| CPU unit tests, `test/unit/` | `VLLM_NEURON_CPU_MODE=1`, `OLMO_CHECKPOINT_PATH` set | 12 passed (`test_loading.py` 7, `test_components.py` 3, `test_tiny_e2e.py` 2) |
| Integration logit test, `test_model_accuracy[1-2048]` | batch 1, max model length 2,048, TP 8 | PASSED (129.5 s) |
| Integration logit test, `test_model_accuracy[4-2048]` | batch 4, max model length 2,048, TP 8 | PASSED (162.8 s) |
| `vllm serve` smoke | the serve command under Usage | Passed, 3 of 3 prompts |

#### Logit validation

| Case | Prompt | Overall Status | Max divergence (1 ULP) | Max K5 (0.03) | Max K50 (0.05) | Max K1000 (0.08) | Max All (0.09) |
|---|---|---|---|---|---|---|---|
| `[1-2048]` | 1/2, `The Amazon rainforest is …` | FAILED | 0.0000 (argmax differs at token 15, Δ 0.0000) | 0.0213 | 0.0377 | 0.0868 (> 0.08) at token 0 | 0.1340 (> 0.09) at token 0 |
| `[1-2048]` | 2/2, `The human genome contains …` | PASSED | 0.0000 | 0.0144 | 0.0266 | 0.0693 | 0.0876 |
| `[4-2048]` | 1/2, `The Amazon rainforest is …` | FAILED | 0.0000 (argmax differs at token 15, Δ 0.0000) | 0.0213 | 0.0377 | 0.0868 (> 0.08) at token 0 | 0.1340 (> 0.09) at token 0 |
| `[4-2048]` | 2/2, `The human genome contains …` | PASSED | 0.0000 | 0.0144 | 0.0266 | 0.0693 | 0.0876 |

Neuron vs CPU BF16 reference; token = generated token. Prompt 1: K1000 0.0868 > 0.08 and All 0.1340 > 0.09 at generated token 0 (batch 4: all four rows); at generated token 15 the argmax differs with Δ 0.0000, within the 1-ULP bound, and validation continues teacher-forced.
Test passed on aggregate σ-ratio 0.5610 (`[1-2048]`) and 0.5615 (`[4-2048]`) ≤ 1.0 (`multi_prompt_logit_validation`); `agg_bc` FAIL in both cases; `pp_static_0.03` 0/2 and every other `pp_*` check 2/2 in both cases.
16 generated tokens per prompt vs the HF BF16 reference (FP32 baseline).

Error against the FP32 reference, per prompt (max over the 16 generated tokens):

| Case | Prompt | CPU BF16 L-inf | Neuron L-inf | Neuron / CPU BF16 | CPU BF16 L2 | Neuron L2 | Neuron / CPU BF16 |
|---|---|---|---|---|---|---|---|
| `[1-2048]` | 1/2 | 0.1015 | 0.0334 | 0.33x | 0.1166 | 0.0423 | 0.36x |
| `[1-2048]` | 2/2 | 0.0595 | 0.0407 | 0.68x | 0.0694 | 0.0507 | 0.73x |
| `[4-2048]` | 1/2 | 0.1015 | 0.0334 | 0.33x | 0.1166 | 0.0423 | 0.36x |
| `[4-2048]` | 2/2 | 0.0595 | 0.0407 | 0.68x | 0.0694 | 0.0507 | 0.73x |

Every maximum is at generated token 0, the token of the prompt 1 miss.

#### Serve smoke

`/v1/completions`, max_tokens 32, temperature 0; 32 tokens returned per prompt. Batched request (all three, max_tokens 16): a prefix of the same text.

| Prompt | Start of the completion (first 60 characters) |
|---|---|
| `The capital of France is` | ` the city of Paris. The capital of France is the city of Par` |
| `1, 2, 3, 4, 5,` | ` 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,` (the whole completion) |
| `def fibonacci(n):` | `\n        if n == 0:\n            return 1\n        if n == 1:\n` |

### Exceptions

- **Prompt 1 is over the two-way K1000 and All tolerances.**
- **Tests not carried over from the original port.** The port's GSM8K
  accuracy test, its performance comparison test, its two-node serving test
  and launcher, and its device module tests for attention and MLP depend on
  shared test harnesses outside this folder (server helpers, multi-node
  launch, benchmark helpers). The integration test here carries the logit
  validation of the port's serving test, on a single node.
- **Rewritten registry assertion.** The port's CPU test asserted that the
  plugin's built-in registry maps `OlmoForCausalLM` to the factory. A contrib
  model is deliberately absent from that registry, so `test_loading.py` checks
  the plugin entry point and the vLLM model protocol instead.
- **Untested configurations.** Prefix caching and the segmented-prefill path,
  tensor-parallel sizes other than 2 (CPU) and 8 (device), speculative
  decoding, Trn3, and any plugin version other than 0.29.0.1.2.0.
