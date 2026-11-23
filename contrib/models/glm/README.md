# Contrib Model: GLM-4-9B-Chat (Text)

[GLM-4-9B-Chat](https://huggingface.co/zai-org/glm-4-9b-chat-hf) is a 9.4B
parameter dense decoder-only language model: 40 layers, hidden size 4096,
grouped-query attention with 32 query heads and 2 KV heads (head_dim 128) and
a QKV bias, partial interleaved RoPE over half of each head, a SwiGLU MLP of
size 13696, RMSNorm and a 151552-id vocabulary with an untied LM head. This
folder serves the transformers-format release (`zai-org/glm-4-9b-chat-hf`,
architecture `GlmForCausalLM`) with vLLM on AWS Trainium2 through the
`vllm-neuron` plugin, in BF16 with tensor parallelism. The implementation is
in `src/`, and [src/README.md](src/README.md) describes the architecture, the
load-time RoPE permutation and the per-feature status.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix and the
> validation report below record exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`), then install this folder:

```bash
pip install -e contrib/models/glm
```

The install adds a `vllm.general_plugins` entry point (`glm_neuron`) that
registers `GlmForCausalLM` in every vLLM process, so `vllm serve` and
`LLM(...)` with tensor parallelism find it with no other setup. vLLM ships its
own GPU `GlmForCausalLM`; this registration replaces it while the package is
installed. `pip uninstall vllm-neuron-contrib-glm` removes it.

### Serve with the OpenAI-compatible server

The configuration below is one trn2.48xlarge node with 16 NeuronCores, the
tensor parallel size of every recorded run:

```bash
vllm serve zai-org/glm-4-9b-chat-hf \
    --revision 8599336fc6c125203efb2360bfaf4c80eef1d1bf \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --tensor-parallel-size 16 \
    --max-model-len 2048 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{"neuron_config": {"quantization": "bf16", "num_batched_tokens_buckets": [1024]}}'
```

`--max-num-batched-tokens 1024` below `--max-model-len 2048` makes prefill run
as segmented (chunked) prefill with one 1024-token graph, the prefill
configuration of the recorded runs. Keep the prefill bucket equal to
`--max-num-batched-tokens` (the default KV segment size): the port found that
an extra, smaller prefill bucket failed for short prompts.

Then send a request:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "zai-org/glm-4-9b-chat-hf",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

With the package installed, nothing else is needed to register the model:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="zai-org/glm-4-9b-chat-hf",
    revision="8599336fc6c125203efb2360bfaf4c80eef1d1bf",
    tensor_parallel_size=16,
    dtype="bfloat16",
    max_model_len=2048,
    max_num_batched_tokens=1024,
    max_num_seqs=4,
    enable_prefix_caching=False,
    additional_config={
        "neuron_config": {
            "quantization": "bf16",
            "num_batched_tokens_buckets": [1024],
        }
    },
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The color of the sky is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

Without the install, register the factory from your own script before you
construct `LLM(...)` (option 1 in [contrib/README.md](../../README.md)), as
`test/integration/test_model.py` does:

```python
from vllm import ModelRegistry

# Your copy of contrib/models/glm/src/, under whatever name you gave it.
from your_package.factory import GlmForCausalLM

ModelRegistry.register_model("GlmForCausalLM", GlmForCausalLM)
```

That registration reaches vLLM's engine-core and worker processes only
because the offline path starts them with the `fork` start method, the default
for a Python script. Under `VLLM_WORKER_MULTIPROC_METHOD=spawn` a child
re-imports from scratch and resolves vLLM's own `GlmForCausalLM` instead; use
the package install in that case.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1`. See the top-level `README.md` and `AGENTS.md` for the
full list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn2 (trn2.48xlarge)   | TP 16, single node: see the Validation report | Not tested |
| Trn3                   | Not tested | Not tested |

CPU mode (`VLLM_NEURON_CPU_MODE=1`) on vLLM Neuron 0.29.0.1.2.0 (vLLM 0.29.0,
transformers 5.17.0, Python 3.14): the unit tests pass, see
[Testing](#testing).

## Example Checkpoints

- <https://huggingface.co/zai-org/glm-4-9b-chat-hf> — the only checkpoint with
  recorded runs. The tests pin revision
  `8599336fc6c125203efb2360bfaf4c80eef1d1bf`.

Other `GlmForCausalLM` (`model_type: glm`) checkpoints in BF16 with default
RoPE and SiLU have the same architecture, but none was run. The factory
rejects a non-BF16 checkpoint, `rope_scaling` other than the default and an
activation other than SiLU. The original `THUDM/glm-4-9b-chat` release uses
remote code (`ChatGLMForConditionalGeneration`) and is not this architecture.

## Testing

The tests read the checkpoint from `GLM4_CHECKPOINT_PATH`, a local copy of the
pinned revision. Without it, the integration test downloads the pinned
revision (about 19 GB) from the HuggingFace hub, and the unit tests download
only the metadata files (`config.json`, the weight index and the tokenizer,
about 20 MB).

### Integration test

Serves the model through vLLM on Neuron at TP 16 on one node, with the
offline `LLM(...)` API, and validates the raw logits of 16 decoded tokens for
2 prompts against a HuggingFace reference (FP32 baseline and BF16
teacher-forced pass), at batch 1 and batch 4 with a 2048-token model length.
The tolerance map is the bf16 dense-golden map of the template (argmax bound
1e-5; top-5 0.03, top-50 0.05, top-1000 0.08, all 0.09). It needs 16
NeuronCores:

```bash
GLM4_CHECKPOINT_PATH=/path/to/glm-4-9b-chat-hf \
    pytest contrib/models/glm/test/integration/test_model.py \
    -v -s --timeout=10800
```

A first run computes the HuggingFace goldens on the host CPU (about 40 GB of
host memory for the FP32 model) and compiles the TP 16 graphs. If the engine
start times out, raise `VLLM_ENGINE_READY_TIMEOUT_S` (vLLM's default is 600 s).

### Unit tests

No Neuron hardware is needed:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/glm/test/unit/ -v -s --timeout=3600
```

- `test_glm_cpu.py`: the registration contract (architecture string, vLLM's
  model protocol, the plugin entry point), the config parse and the factory
  checks, the RoPE permutation and frequencies and RMSNorm against HF, a
  compiled rotary graph with no in-place write, the TP 16 QKV and gate/up
  loaders (also for a model built on the meta device, with 16 CPU ranks), the
  weight mapping against the checkpoint index, and the padded-prefill KV write
  and segmented-prefill block table.
- `test_glm_cpu.py::test_glm_full_model_load_and_prefill_logits_cpu` runs
  only when `GLM4_CHECKPOINT_PATH` is set: a 40-layer TP 16 CPU load of the
  real weights, then prefill logits compared three-way with HF. It needs about
  100 GB of host memory.
- `test_tiny_glm_e2e.py`: the whole vLLM pipeline (load, prefill, decode,
  greedy sampling) at TP 4 on a random-weight 2-layer GLM with the same
  attention layout per rank as the real model at TP 16, against HF.

All test directories reach the implementation through `test/_local_src.py`,
which loads `src/` by file location under the unique module name
`contrib_glm_src`, so the folder keeps working when it is copied into an
application. `test/glm_test_utils.py` holds the shared checkpoint helpers, and
`test/conftest.py` puts the `test/` directory on `sys.path` and shuts down the
plugin's `MPExecutor` worker pool at the end of the session.

## Validation report

### vLLM Neuron 0.29.0.1.2.0 (2026-11-19)

#### Environment

| Image | Instance | TP | vLLM | vllm-neuron | torch | transformers | neuronx-cc |
|---|---|---|---|---|---|---|---|
| `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` | `trn2.48xlarge` (LNC 2), 1 node | 16 | 0.29.0 | 0.29.0.1.2.0 | 2.13.0 | 5.17.0 | 2.28.7631.0+8f7b0cf5 |

Checkpoint: `zai-org/glm-4-9b-chat-hf` at revision
`8599336fc6c125203efb2360bfaf4c80eef1d1bf`.

#### Results

| Test | Configuration | Result |
|---|---|---|
| CPU unit tests, `test/unit/` | `VLLM_NEURON_CPU_MODE=1`, `GLM4_CHECKPOINT_PATH` set (includes `test_glm_full_model_load_and_prefill_logits_cpu[16]`) | 37 passed |
| Integration logit test, `test_model_accuracy[1-2048]` | batch 1, max model length 2,048, prefill bucket 1,024, TP 16, `VLLM_ENGINE_READY_TIMEOUT_S=7200` | PASSED (260.96 s) |
| Integration logit test, `test_model_accuracy[4-2048]` | batch 4, max model length 2,048, prefill bucket 1,024, TP 16, `VLLM_ENGINE_READY_TIMEOUT_S=7200` | PASSED (293.72 s) |
| `vllm serve` smoke | the serve command under Usage, `VLLM_ENGINE_READY_TIMEOUT_S=7200` | Passed, 3 of 3 prompts |

#### Logit validation

`test_model_accuracy[1-2048]`:

| Prompt | Overall Status | Max divergence (1 ULP) | Max K5 (0.03) | Max K50 (0.05) | Max K1000 (0.08) | Max All (0.09) |
|---|---|---|---|---|---|---|
| 1/2, `The Amazon rainforest is …` | PASSED | 0.0000 | 0.0086 | 0.0154 | 0.0226 | 0.0226 |
| 2/2, `The human genome contains …` | PASSED | 0.0000 | 0.0106 | 0.0139 | 0.0193 | 0.0167 |

Test passed on two-way validation 2/2 prompts (`multi_prompt_logit_validation`); aggregate σ-ratio 1.1333 (≤ 1.0 = pass); `agg_bc` FAIL; `pp_max_l2_1.5x` 2/2, `pp_max_l2_2.0x` 2/2, `pp_max_linf_1.5x` 2/2, `pp_max_linf_2.0x` 2/2, `pp_static_0.03` 2/2, `pp_static_0.05` 2/2, `pp_tok_l2_1.5x` 2/2, `pp_tok_l2_2.0x` 2/2, `pp_tok_linf_1.5x` 0/2, `pp_tok_linf_2.0x` 2/2.
`pp_tok_linf_1.5x` misses both prompts: prompt 1 at generated token 7, Neuron L-inf 0.0101 vs CPU BF16 L-inf 0.0064 (1.58x); prompt 2 at generated token 12, 0.0140 vs 0.0086 (1.63x).

`test_model_accuracy[4-2048]`:

| Prompt | Overall Status | Max divergence (1 ULP) | Max K5 (0.03) | Max K50 (0.05) | Max K1000 (0.08) | Max All (0.09) |
|---|---|---|---|---|---|---|
| 1/2, `The Amazon rainforest is …` | PASSED | 0.0000 | 0.0086 | 0.0154 | 0.0226 | 0.0226 |
| 2/2, `The human genome contains …` | PASSED | 0.0000 | 0.0106 | 0.0139 | 0.0193 | 0.0167 |

Test passed on two-way validation 2/2 prompts (`multi_prompt_logit_validation`); aggregate σ-ratio 1.1333 (≤ 1.0 = pass); `agg_bc` FAIL; `pp_max_l2_1.5x` 2/2, `pp_max_l2_2.0x` 2/2, `pp_max_linf_1.5x` 2/2, `pp_max_linf_2.0x` 2/2, `pp_static_0.03` 2/2, `pp_static_0.05` 2/2, `pp_tok_l2_1.5x` 2/2, `pp_tok_l2_2.0x` 2/2, `pp_tok_linf_1.5x` 0/2, `pp_tok_linf_2.0x` 2/2.
The same prompts miss `pp_tok_linf_1.5x`, with the same values in batch row 0.

16 generated tokens per prompt (4 batch rows of 16 in `[4-2048]`) vs the HF BF16 reference (FP32 baseline); prompts in the order of `PROMPTS` in `test/integration/test_model.py`; no argmax divergence.

Error against the FP32 reference, per prompt (max over the 16 generated tokens; `[1-2048]` and batch row 0 of `[4-2048]` print the same values):

| Prompt | CPU BF16 L-inf | Neuron L-inf | Neuron / CPU BF16 | CPU BF16 L2 | Neuron L2 | Neuron / CPU BF16 |
|---|---|---|---|---|---|---|
| 1/2 | 0.0142 | 0.0171 | 1.20x | 0.0176 | 0.0173 | 0.98x |
| 2/2 | 0.0116 | 0.0140 | 1.21x | 0.0119 | 0.0124 | 1.04x |

#### Serve smoke

`/v1/completions`, max_tokens 32, temperature 0; 32 tokens returned per prompt. Batched request (all three, max_tokens 16): a prefix of the same text for each prompt.

| Prompt | Start of the completion (first 60 characters) |
|---|---|
| `The capital of France is` | ` Paris. Paris is known for its rich history, stunning archit` |
| `1, 2, 3, 4, 5,` | ` 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16` (the whole output) |
| `def fibonacci(n):` | `Here is a simple implementation of the Fibonacci sequence in` |

### Exceptions

- **Topology changed from the port's runs.** The port's device logit run
  served the model as two TP 16 servers on two nodes (one prefill, one decode,
  KV transferred between them). `test/integration/test_model.py` serves it
  colocated on one node at the same TP 16, with the same model length (2048),
  prefill size and bucket (1024), BF16 and greedy on-device sampling. Batch 4
  is new; the port's runs used batch 1.
- **Full-vocabulary `generate_fn`.** The test packs logits with its own
  `_full_vocab_generate_fn` instead of the plugin's
  `create_offline_vllm_generate_fn`, because the model scores 151552 ids and
  the tokenizer has 151329 base tokens; see the function's docstring. The
  comparison and the tolerance map are the plugin's.
- **Dropped: the port's disaggregated-inference tests** (the 1P1D serving and
  logit test and the 2-node component tests). They need two nodes, a KV
  transfer connector, a proxy and MPI, and import helpers from the plugin's
  test tree (`test.utils.*`, `test.vllm_neuron.utils.*`). Their logit check is
  what `test/integration/test_model.py` carries now, with the prompts copied
  locally.
- **Dropped from the CPU tests:** the checks of the disaggregated-inference
  test harness (proxy readiness, prefill bucket and sampling settings of the
  2-node servers, failure probes, log tails). They test helpers that are not
  part of this folder.
- **Rewritten assertion.** The port's CPU test asserted that
  `vllm_neuron.model.registry.get_models()` maps `GlmForCausalLM` to the
  factory. A contrib model is deliberately absent from that registry, so the
  shipped test checks the registration contract instead.
- **Untested:** prefix caching, context beyond 2048 tokens, tensor parallel
  sizes other than 16 on device, speculative decoding and Trn3.
