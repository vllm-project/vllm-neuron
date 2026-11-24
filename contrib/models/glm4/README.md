# Contrib Model: GLM-4-32B-0414 (Text)

[GLM-4-32B-0414](https://huggingface.co/zai-org/GLM-4-32B-0414) is a 32B
parameter dense decoder-only language model: 61 layers, hidden size 6144,
grouped-query attention with 48 query heads and 2 KV heads (head_dim 128) and
no attention bias, partial interleaved RoPE over half of each head, four
RMSNorms per decoder layer ("sandwich" norms), a SwiGLU MLP of size 23040 and
a 151552-id vocabulary with an untied LM head. This folder serves it
(architecture `Glm4ForCausalLM`, `model_type: glm4`) with vLLM on AWS
Trainium2 through the `vllm-neuron` plugin, in BF16 with tensor parallelism.
The implementation is in `src/`, and [src/README.md](src/README.md) describes
the architecture, the load-time RoPE permutation and the per-feature status.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix and the
> validation report below record exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`), then install this folder:

```bash
pip install -e contrib/models/glm4
```

The install adds a `vllm.general_plugins` entry point (`glm4_neuron`) that
registers `Glm4ForCausalLM` in every vLLM process, so `vllm serve` and
`LLM(...)` with tensor parallelism find it with no other setup. vLLM ships its
own GPU `Glm4ForCausalLM`; this registration replaces it while the package is
installed. `pip uninstall vllm-neuron-contrib-glm4` removes it.

### Serve with the OpenAI-compatible server

The configuration below is the batch-4 configuration of the integration
test: one trn2.48xlarge node, TP 8 on 8 NeuronCores (LNC 2), a 9216-token
model length (room for an 8192-token prompt and 1024 generated tokens),
1024-token prefill chunks and up to 4 sequences. The first start compiles the 61-layer graphs,
which takes longer than the default engine-start (600 s) and per-graph compile
(600 s) timeouts, so raise them:

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=5400
export NEURON_LIBTORCH_COMPILATION_TIMEOUT=2400
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600

vllm serve zai-org/GLM-4-32B-0414 \
    --revision 077b5c2f5c43bd3239fd605a0600229e8facbd4a \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 9216 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [1024], "num_seqs_buckets": [4]}}'
```

`--max-num-batched-tokens 1024` below `--max-model-len 9216` makes prefill run
as segmented (chunked) prefill with one 1024-token graph and a 1024-token KV
segment: an 8192-token prompt takes 8 prefill steps, and each chunk after the
first attends to the K/V of the earlier chunks in the paged KV cache. When
`--max-num-batched-tokens` is below `--max-model-len`, keep it at a supported
KV segment size (512, 1024, 2048, 4096 or 8192): vLLM Neuron 0.29.0.1.2.0
turns segmented prefill on by default only for those sizes. With another
size, segmented prefill stays off, and each chunk of a longer prompt runs
through the flash-attention path of `src/model.py`, which attends only to the
tokens of that chunk.

Then send a request:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "zai-org/GLM-4-32B-0414",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

With the package installed, nothing else is needed to register the model (set
the three timeouts above in the environment for the first, compiling run):

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="zai-org/GLM-4-32B-0414",
    revision="077b5c2f5c43bd3239fd605a0600229e8facbd4a",
    tensor_parallel_size=8,
    dtype="bfloat16",
    max_model_len=9216,
    max_num_batched_tokens=1024,
    max_num_seqs=4,
    enable_prefix_caching=False,
    async_scheduling=False,
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

Without the install, register the factory from your own script before you
construct `LLM(...)` (option 1 in [contrib/README.md](../../README.md)), as
`test/integration/test_model.py` does:

```python
from vllm import ModelRegistry

# Your copy of contrib/models/glm4/src/, under whatever name you gave it.
from your_package.factory import Glm4ForCausalLM

ModelRegistry.register_model("Glm4ForCausalLM", Glm4ForCausalLM)
```

That registration reaches vLLM's engine-core and worker processes only
because the offline path starts them with the `fork` start method, the default
for a Python script. Under `VLLM_WORKER_MULTIPROC_METHOD=spawn` a child
re-imports from scratch and resolves vLLM's own `Glm4ForCausalLM` instead; use
the package install in that case. The `vllm` command line sets
`VLLM_WORKER_MULTIPROC_METHOD=spawn` when it is not already set, so
`vllm serve` needs the package install.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1`. See
[docs/guides/reference-configuration.md](../../../docs/guides/reference-configuration.md)
for the `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn2 (trn2.48xlarge)   | Working: TP 8, one node, 9216-token model length (see the Validation report) | Not tested |
| Trn3                   | Not tested | Not tested |

CPU mode (`VLLM_NEURON_CPU_MODE=1`) on vLLM Neuron 0.29.0.1.2.0 (vLLM 0.29.0,
transformers 5.17.0, Python 3.14): the unit tests pass, see
[Testing](#testing).

## Example Checkpoints

- <https://huggingface.co/zai-org/GLM-4-32B-0414> — the checkpoint the tests
  run. They pin revision `077b5c2f5c43bd3239fd605a0600229e8facbd4a`.

Other `Glm4ForCausalLM` (`model_type: glm4`) checkpoints of the GLM-4-0414
family in BF16 have the same architecture, but the tests do not run them. The
factory rejects a non-BF16 or quantized checkpoint, an attention bias, and a
`partial_rotary_factor` that does not give an even rotary dimension. The
earlier `zai-org/glm-4-9b-chat-hf` release is a different architecture
(`GlmForCausalLM`, no sandwich norms, with a QKV bias).

## Testing

The tests read the checkpoint from `GLM4_0414_CHECKPOINT_PATH`, a local copy
of the pinned revision. Without it, the integration test downloads the pinned
revision (about 65 GB) from the HuggingFace hub, and the unit tests download
only the metadata files (`config.json`, the weight index and the tokenizer,
about 20 MB).

### Integration test

Serves the model through vLLM on Neuron at TP 8 on one node, with the offline
`LLM(...)` API, in the configuration of the serve example above (with
`max_num_seqs` equal to the batch size): a 9216-token model length, segmented
prefill in 1024-token chunks (one 1024-token prefill graph, KV segment 1024),
BF16 and on-device greedy sampling. The test builds
two prompts from fixed text in the test file, a prose document whose sections
repeat in a fixed order and a Python module of small generated functions, and
cuts each one at a sentence or function boundary to 8000 to 8192 tokens of
the checkpoint's tokenizer. Each prompt runs as 8 prefill chunks. The test
validates the raw logits of 16 decoded tokens per prompt against a
HuggingFace reference (FP32 baseline and BF16 teacher-forced pass), at batch 1
and batch 4 (four copies of each prompt), cases `[1-9216]` and `[4-9216]`.
The tolerance map is the bf16 dense-golden map of the template (argmax bound
1e-5; top-5 0.03, top-50 0.05, top-1000 0.08, all 0.09), and the test also
requires `pp_max_linf_1.5x` and `pp_max_l2_1.5x` on every prompt. It needs 8
NeuronCores:

```bash
GLM4_0414_CHECKPOINT_PATH=/path/to/GLM-4-32B-0414 \
    pytest contrib/models/glm4/test/integration/test_model.py \
    -v -s --timeout=14400
```

A first run computes the HuggingFace references on the host CPU: the FP32
model (about 130 GB of host memory) and then the BF16 model, one at a time,
with SDPA attention. Each 8k-token FP32 prefill also holds about 5 GB of
logits. The run then compiles the TP 8 graphs for each case. The test sets
`VLLM_ENGINE_READY_TIMEOUT_S=5400`, `NEURON_LIBTORCH_COMPILATION_TIMEOUT=2400`
and `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600` unless they are already set.

The test packs the vLLM logits with its own `_full_vocab_generate_fn` instead
of the plugin's `create_offline_vllm_generate_fn`, because the model scores
151552 ids and the tokenizer has 151329 base tokens (see the function's
docstring). It also sets `ignore_eos`, because the HF reference always decodes
16 tokens. The comparison and the tolerance map are the plugin's.

### Unit tests

No Neuron hardware is needed:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/glm4/test/unit/ -v -s --timeout=3600
```

- `test_glm4_cpu.py`: the registration contract (architecture string, vLLM's
  model protocol, the plugin entry point), the config parse and the factory
  checks, the RoPE permutation and frequencies against the HF GLM-4 RoPE at
  positions up to 9215 (also under a meta-device context), and a tiny
  random-weight decoder layer at TP 2 (prefill, then 64 decode steps through
  the KV cache) compared three-way with the HF `Glm4DecoderLayer`.
- `test_glm4_weight_loading.py`: the weight mapping of the 61-layer model
  against every tensor in the checkpoint index. With
  `GLM4_0414_CHECKPOINT_PATH` set, also the TP 8 rank-7 load of every
  parameter, a meta-device build of the QKV loader, and the QKV (with the RoPE
  permutation), O, gate, up, down, embedding and LM head shards against
  independent slices of the checkpoint tensors (about 10 GB of host memory).
- `test_tiny_glm4_e2e.py`: the whole vLLM pipeline (load, prefill, decode,
  greedy sampling) at TP 4 on a random-weight 2-layer GLM-4 with one KV head
  per rank (each replicated on 2 ranks), as the real model has at TP 8,
  against HF. It runs flash-attention prefill, and segmented (chunked) prefill
  of a 600-token prompt in two chunks of at most 512 tokens. The TP 4 engine needs at least 256 MiB free in
  `/dev/shm` (vLLM's worker queues); in a container started with the docker
  default of 64 MiB the test skips, so start the container with `--shm-size`
  or `--ipc=host`.

All test directories reach the implementation through `test/_local_src.py`,
which loads `src/` by file location under the unique module name
`contrib_glm4_src`, so the folder keeps working when it is copied into an
application. `test/glm4_test_utils.py` holds the shared checkpoint helpers, and
`test/conftest.py` puts the `test/` directory on `sys.path` and shuts down the
plugin's `MPExecutor` worker pool at the end of the session.

### Not validated

- Flash-attention (single-step) prefill on device: the device configuration
  uses segmented prefill. CPU mode covers it with a tiny model.
- Context beyond 9216 tokens (the checkpoint has 32768 positions).
- Logits beyond the first 16 generated tokens of each prompt.
- Batch sizes other than 1 and 4, and tensor parallel sizes other than 8, on
  device (CPU mode covers TP 2 and TP 4 with tiny models).
- Prefix caching, speculative decoding and Trn3.

## Validation report

### vLLM Neuron 0.29.0.1.2.0 (2026-11-24)

#### Environment

| Image | Instance | TP | vLLM | vllm-neuron | torch | transformers | neuronx-cc |
|---|---|---|---|---|---|---|---|
| `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` | `trn2.48xlarge` (LNC 2), 1 node | 8 | 0.29.0 | 0.29.0.1.2.0 | 2.13.0 | 5.17.0 | 2.28.7631.0+8f7b0cf5 |

Checkpoint of the integration test: `zai-org/GLM-4-32B-0414` at
`077b5c2f5c43bd3239fd605a0600229e8facbd4a`.

#### Results

| Test | Configuration | Result |
|---|---|---|
| CPU unit tests, `test/unit/` | `VLLM_NEURON_CPU_MODE=1`, `GLM4_0414_CHECKPOINT_PATH` set | 15 passed, 0 failed (includes the 3 weight tests and both `test_tiny_glm4_e2e.py` cases) |
| `test_model_accuracy[1-9216]` | TP 8, `neuron_native`, BF16, batch 1, max model length 9,216, segmented prefill (bucket 1,024, KV segment 1,024), 2 prompts of 8,000 to 8,192 tokens × 16 greedy tokens, on-device greedy sampling | PASSED (674.13 s, including the HF reference): 2/2 prompts within `TOL_MAP` on all 16 tokens, no argmax divergence; aggregate σ-ratio 0.9358 ≤ 1.0 |
| `test_model_accuracy[4-9216]` | TP 8, `neuron_native`, BF16, batch 4 (each prompt in 4 rows), max model length 9,216, segmented prefill (bucket 1,024, KV segment 1,024), 2 prompts of 8,000 to 8,192 tokens × 16 greedy tokens, on-device greedy sampling | PASSED (397.12 s): 2/2 prompts within `TOL_MAP` on all 64 tokens, no argmax divergence; aggregate σ-ratio 0.9180 ≤ 1.0 |
| `vllm serve` smoke | The Usage command (with `--revision` and the three timeout variables), installed with `pip install --no-deps -e contrib/models/glm4` | Passed: 3/3 prompts, 32 completion tokens each; a batched request of all 3 at `max_tokens` 16 returned a prefix of each |
| `vllm serve` 8k-token request | Same server; one `/v1/completions` request with a deterministic synthetic ledger document (295 numbered records and a request to summarize them), measured with the server's `/tokenize` endpoint; `max_tokens` 1024, `temperature` 0, `ignore_eos` | Passed: 8,173 prompt tokens, 1,024 completion tokens, finish reason `length`, 28.8 s |

The test asserts `pp_max_linf_1.5x` and `pp_max_l2_1.5x` on every prompt; both were 2/2 in `[1-9216]` and in `[4-9216]` in this run.

Engine start (profile, KV cache, warmup) took 205.40 s (compilation 198.15 s)
in `[1-9216]`, 232.21 s (compilation 225.08 s) in `[4-9216]` and 234.26 s
(compilation 229.25 s) for `vllm serve`. The HF reference (FP32 and BF16) of
`[1-9216]` took 385.38 s; `[4-9216]` read it from the golden cache.

#### Logit validation

| Case | Prompt | Overall Status | Max K5 (0.03) | Max K50 (0.05) | Max K1000 (0.08) | Max All (0.09) | Argmax divergence (1 ULP) |
|---|---|---|---|---|---|---|---|
| `[1-9216]` | 1 of 2, prose, `Section 1. Tropical rainforests …` | PASSED | 0.0131 at token 6 | 0.0132 at token 8 | 0.0327 at token 8 | 0.0463 at token 8 | 0.0000 |
| `[1-9216]` | 2 of 2, code, `"""Helpers for lists of numbers. …` | PASSED | 0.0057 at token 4 | 0.0126 at token 2 | 0.0126 at token 2 | 0.0145 at token 4 | 0.0000 |
| `[4-9216]` | 1 of 2, prose, `Section 1. Tropical rainforests …` | PASSED | 0.0152 at token 6 | 0.0152 at token 6 | 0.0266 at token 6 | 0.0237 at token 6 | 0.0000 |
| `[4-9216]` | 2 of 2, code, `"""Helpers for lists of numbers. …` | PASSED | 0.0053 at token 7 | 0.0093 at token 5 | 0.0106 at token 6 | 0.0151 at token 4 | 0.0000 |

Neuron vs CPU BF16 reference; token = generated token. `[4-9216]`: positions
of batch row 0; each of the 4 rows printed the same per-row maxima. No argmax
divergence.
`[1-9216]`: two-way validation 2/2 prompts (`multi_prompt_logit_validation`);
aggregate σ-ratio 0.9358 ≤ 1.0; `agg_bc` FAIL, `agg_l2_1.2x` FAIL,
`agg_l2_1.5x` FAIL, `agg_linf_1.2x` FAIL, `agg_linf_1.5x` PASS; printed
per-prompt checks: `pp_max_linf_1.5x` 2/2, `pp_max_linf_2.0x` 2/2,
`pp_max_l2_1.5x` 2/2, `pp_max_l2_2.0x` 2/2, `pp_tok_linf_1.5x` 0/2,
`pp_tok_linf_2.0x` 1/2, `pp_tok_l2_1.5x` 1/2, `pp_tok_l2_2.0x` 2/2,
`pp_static_0.03` 1/2, `pp_static_0.05` 2/2.
`[4-9216]`: two-way validation 2/2 prompts; aggregate σ-ratio 0.9180 ≤ 1.0;
`agg_bc` FAIL, `agg_l2_1.2x` FAIL, `agg_l2_1.5x` PASS, `agg_linf_1.2x` FAIL,
`agg_linf_1.5x` PASS; printed per-prompt checks: `pp_max_linf_1.5x` 2/2,
`pp_max_linf_2.0x` 2/2, `pp_max_l2_1.5x` 2/2, `pp_max_l2_2.0x` 2/2,
`pp_tok_linf_1.5x` 1/2, `pp_tok_linf_2.0x` 1/2, `pp_tok_l2_1.5x` 2/2,
`pp_tok_l2_2.0x` 2/2, `pp_static_0.03` 2/2, `pp_static_0.05` 2/2.
Per-token ratios over 1.5x: `[1-9216]` prompt 1 L-inf tokens 13 (1.56x) and
14 (2.10x), L2 token 13 (1.53x), prompt 2 L-inf tokens 5 (1.78x) and 8
(1.69x); `[4-9216]` row 0, prompt 1 L-inf tokens 13 (1.57x) and 14 (2.81x).

Error against the FP32 reference, per prompt (max over the 16 generated
tokens; `[4-9216]` prints batch row 0):

| Case | Prompt | CPU BF16 L-inf | Neuron L-inf | Neuron / CPU BF16 | CPU BF16 L2 | Neuron L2 | Neuron / CPU BF16 |
|---|---|---|---|---|---|---|---|
| `[1-9216]` | 1 of 2 | 0.0228 | 0.0322 | 1.41x | 0.0390 | 0.0329 | 0.84x |
| `[1-9216]` | 2 of 2 | 0.0152 | 0.0128 | 0.84x | 0.0215 | 0.0208 | 0.97x |
| `[4-9216]` | 1 of 2 | 0.0228 | 0.0236 | 1.04x | 0.0390 | 0.0351 | 0.90x |
| `[4-9216]` | 2 of 2 | 0.0152 | 0.0101 | 0.66x | 0.0215 | 0.0180 | 0.84x |

Generated token of each Neuron maximum: `[1-9216]` prompt 1 L-inf and L2
token 8; prompt 2 L-inf token 15, L2 token 1. `[4-9216]` prompt 1 L-inf and
L2 token 3; prompt 2 L-inf token 2, L2 token 8.

#### Serve smoke

`/v1/completions`, `temperature` 0, `max_tokens` 32; first 60 characters of each output as a Python string literal. The 8k-token request: `max_tokens` 1024, `temperature` 0, `ignore_eos`.

| Prompt | Output |
|---|---|
| `The capital of France is` | `' Paris. It is the largest city in France. It is located on t'` |
| `1, 2, 3, 4, 5,` | `' 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16'` |
| `def fibonacci(n):` | `' \n    if n<0: \n        print("Incorrect input") \n    # First'` |
| Synthetic ledger document, 295 records, 8,173 tokens | `'Okay, here is a detailed report summarizing the trade ledger'` |

### Exceptions

- **Informational checks that did not pass.** `pp_tok_linf_1.5x` 0/2 in
  `[1-9216]` and 1/2 in `[4-9216]`, `pp_tok_linf_2.0x` 1/2 in both cases,
  `pp_tok_l2_1.5x` 1/2 and `pp_static_0.03` 1/2 in `[1-9216]`, `agg_bc`,
  `agg_l2_1.2x` and `agg_linf_1.2x` FAIL in both cases, `agg_l2_1.5x` FAIL in
  `[1-9216]` (see Logit validation). The test asserts `result.passed`,
  `pp_max_linf_1.5x` and `pp_max_l2_1.5x`.
- **Logits validated for 16 generated tokens per prompt.** The 1,024
  generated tokens of the 8k-token serve request are not compared with a
  reference; the end of that output repeats `Stored, Sold, Repaired,`.
- **Full-vocabulary `generate_fn`.** The test packs logits with its own
  `_full_vocab_generate_fn` instead of the plugin's
  `create_offline_vllm_generate_fn`, because the model scores 151552 ids and
  the tokenizer has 151329 base tokens, and it sets `ignore_eos`; see the
  function's docstring. The comparison and the tolerance map are the plugin's.
- Flash-attention (single-step) prefill on device, prefix caching, context
  beyond 9,216 tokens, batch sizes other than 1 and 4, tensor parallel sizes
  other than 8 on device, speculative decoding and Trn3 are not tested.
