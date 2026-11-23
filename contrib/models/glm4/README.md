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

The configuration below is one trn2.48xlarge node with 8 NeuronCores (LNC 2),
the tensor parallel size of every recorded run. The first start compiles the
61-layer graphs, which takes longer than the default engine-start (600 s) and
per-graph compile (600 s) timeouts, so raise them:

```bash
export VLLM_ENGINE_READY_TIMEOUT_S=5400
export NEURON_LIBTORCH_COMPILATION_TIMEOUT=2400
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600

vllm serve zai-org/GLM-4-32B-0414 \
    --revision 077b5c2f5c43bd3239fd605a0600229e8facbd4a \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 2048 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [1024], "num_seqs_buckets": [4]}}'
```

`--max-num-batched-tokens 1024` below `--max-model-len 2048` makes prefill run
as segmented (chunked) prefill with one 1024-token graph, the prefill
configuration of the recorded runs. When `--max-num-batched-tokens` is below
`--max-model-len`, keep it at a supported KV segment size (512, 1024, 2048,
4096 or 8192): vLLM Neuron 0.29.0.1.2.0 turns segmented prefill on by default
only for those sizes. With another size, a prompt longer than one chunk runs
each chunk as full prefill, which attends only to the tokens of that chunk
(the CPU test `test_tiny_glm4_e2e.py` showed wrong logits for a 128-token
chunk).

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
the package install in that case.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1`. See the top-level `README.md` and `AGENTS.md` for the
full list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn2 (trn2.48xlarge)   | See the Validation report: TP 8, one node | Not tested |
| Trn3                   | Not tested | Not tested |

CPU mode (`VLLM_NEURON_CPU_MODE=1`) on vLLM Neuron 0.29.0.1.2.0 (vLLM 0.29.0,
transformers 5.17.0, Python 3.14): the unit tests pass, see
[Testing](#testing).

## Example Checkpoints

- <https://huggingface.co/zai-org/GLM-4-32B-0414> — the only checkpoint with
  recorded runs. The tests pin revision
  `077b5c2f5c43bd3239fd605a0600229e8facbd4a`.

Other `Glm4ForCausalLM` (`model_type: glm4`) checkpoints of the GLM-4-0414
family in BF16 have the same architecture, but none was run. The factory
rejects a non-BF16 or quantized checkpoint, an attention bias, and a
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
`LLM(...)` API, and validates the raw logits of 16 decoded tokens for 2
prompts against a HuggingFace reference (FP32 baseline and BF16
teacher-forced pass), at batch 1 and batch 4 with a 2048-token model length
and a 1024-token prefill bucket. The tolerance map is the bf16 dense-golden
map of the template (argmax bound 1e-5; top-5 0.03, top-50 0.05, top-1000
0.08, all 0.09). It needs 8 NeuronCores:

```bash
GLM4_0414_CHECKPOINT_PATH=/path/to/GLM-4-32B-0414 \
    pytest contrib/models/glm4/test/integration/test_model.py \
    -v -s --timeout=14400
```

A first run computes the HuggingFace goldens on the host CPU (about 130 GB of
host memory for the FP32 model) and compiles the TP 8 graphs. The test sets
`VLLM_ENGINE_READY_TIMEOUT_S=5400`, `NEURON_LIBTORCH_COMPILATION_TIMEOUT=2400`
and `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600` unless they are already set.

### Unit tests

No Neuron hardware is needed:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/glm4/test/unit/ -v -s --timeout=3600
```

- `test_glm4_cpu.py`: the registration contract (architecture string, vLLM's
  model protocol, the plugin entry point), the config parse and the factory
  checks, the RoPE permutation and frequencies against the HF GLM-4 RoPE
  (also under a meta-device context), and a tiny random-weight decoder layer at
  TP 2 (prefill, then 64 decode steps through the KV cache) compared three-way
  with the HF `Glm4DecoderLayer`.
- `test_glm4_weight_loading.py`: the weight mapping of the 61-layer model
  against every tensor in the checkpoint index. With
  `GLM4_0414_CHECKPOINT_PATH` set, also the TP 8 rank-7 load of every
  parameter, a meta-device build of the QKV loader, and the QKV (with the RoPE
  permutation), O, gate, up, down, embedding and LM head shards against
  independent slices of the checkpoint tensors (about 10 GB of host memory).
- `test_tiny_glm4_e2e.py`: the whole vLLM pipeline (load, prefill, decode,
  greedy sampling) at TP 4 on a random-weight 2-layer GLM-4 with one KV head
  per rank (each replicated on 2 ranks), as the real model has at TP 8,
  against HF. It runs
  flash-attention prefill and segmented (chunked) prefill. The TP 4 engine
  needs at least 256 MiB free in `/dev/shm` (vLLM's worker queues); in a
  container started with the docker default of 64 MiB the test skips, so start
  the container with `--shm-size` or `--ipc=host`.

All test directories reach the implementation through `test/_local_src.py`,
which loads `src/` by file location under the unique module name
`contrib_glm4_src`, so the folder keeps working when it is copied into an
application. `test/glm4_test_utils.py` holds the shared checkpoint helpers, and
`test/conftest.py` puts the `test/` directory on `sys.path` and shuts down the
plugin's `MPExecutor` worker pool at the end of the session.

## Validation report

### vLLM Neuron 0.29.0.1.2.0 (2026-11-23)

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
| `test_model_accuracy[1-2048]` | TP 8, `neuron_native`, BF16, batch 1, max model length 2,048, segmented prefill (bucket 1,024, KV segment 1,024), 2 prompts × 16 greedy tokens, on-device greedy sampling | PASSED (340.040 s): 2/2 prompts within `TOL_MAP` on all 16 tokens, no argmax divergence; aggregate σ-ratio 0.9207 ≤ 1.0 |
| `test_model_accuracy[4-2048]` | TP 8, `neuron_native`, BF16, batch 4 (each prompt in 4 rows), max model length 2,048, segmented prefill (bucket 1,024, KV segment 1,024), 2 prompts × 16 greedy tokens, on-device greedy sampling | PASSED (387.375 s): 2/2 prompts within `TOL_MAP` on all 64 tokens, no argmax divergence; aggregate σ-ratio 0.9385 ≤ 1.0 |
| `vllm serve` smoke | The Usage command (with `--revision` and the three timeout variables), installed with `pip install --no-deps -e contrib/models/glm4` | Passed: 3/3 prompts, 32 completion tokens each; a batched request of all 3 at `max_tokens` 16 returned a prefix of each |

The test asserts `pp_max_linf_1.5x` and `pp_max_l2_1.5x` on every prompt; both were 2/2 in `[1-2048]` and in `[4-2048]` in this run.

Engine start (profile, KV cache, warmup) took 212.18 s (compilation 204.87 s)
in `[1-2048]`, 232.51 s (compilation 225.24 s) in `[4-2048]` and 225.58 s
(compilation 220.47 s) for `vllm serve`.

#### Logit validation

| Case | Prompt | Overall Status | Max K5 (0.03) | Max K50 (0.05) | Max K1000 (0.08) | Max All (0.09) | Argmax divergence (1 ULP) |
|---|---|---|---|---|---|---|---|
| `[1-2048]` | 1 of 2, `The Amazon rainforest …` | PASSED | 0.0075 at token 6 | 0.0076 at token 2 | 0.0077 at token 5 | 0.0151 at token 11 | 0.0000 |
| `[1-2048]` | 2 of 2, `The human genome …` | PASSED | 0.0084 at token 4 | 0.0092 at token 10 | 0.0094 at token 1 | 0.0174 at token 1 | 0.0000 |
| `[4-2048]` | 1 of 2, `The Amazon rainforest …` | PASSED | 0.0074 at token 4 | 0.0074 at token 4 | 0.0090 at token 13 | 0.0091 at token 0 | 0.0000 |
| `[4-2048]` | 2 of 2, `The human genome …` | PASSED | 0.0074 at token 2 | 0.0123 at token 4 | 0.0123 at token 4 | 0.0135 at token 6 | 0.0000 |

Neuron vs CPU BF16 reference; token = generated token. `[4-2048]`: positions
of batch row 0; each of the 4 rows printed the same per-row maxima. No argmax
divergence.
`[1-2048]`: two-way validation 2/2 prompts (`multi_prompt_logit_validation`);
aggregate σ-ratio 0.9207 ≤ 1.0; `agg_bc` FAIL, `agg_l2_1.2x` PASS,
`agg_l2_1.5x` PASS, `agg_linf_1.2x` FAIL, `agg_linf_1.5x` FAIL; printed
per-prompt checks: `pp_max_linf_1.5x` 2/2, `pp_max_linf_2.0x` 2/2,
`pp_max_l2_1.5x` 2/2, `pp_max_l2_2.0x` 2/2, `pp_tok_linf_1.5x` 0/2,
`pp_tok_linf_2.0x` 1/2, `pp_tok_l2_1.5x` 2/2, `pp_tok_l2_2.0x` 2/2,
`pp_static_0.03` 2/2, `pp_static_0.05` 2/2.
`[4-2048]`: two-way validation 2/2 prompts; aggregate σ-ratio 0.9385 ≤ 1.0;
`agg_bc` FAIL, `agg_l2_1.2x` FAIL, `agg_l2_1.5x` PASS, `agg_linf_1.2x` PASS,
`agg_linf_1.5x` PASS; printed per-prompt checks: `pp_max_linf_1.5x` 2/2,
`pp_max_linf_2.0x` 2/2, `pp_max_l2_1.5x` 2/2, `pp_max_l2_2.0x` 2/2,
`pp_tok_linf_1.5x` 0/2, `pp_tok_linf_2.0x` 2/2, `pp_tok_l2_1.5x` 2/2,
`pp_tok_l2_2.0x` 2/2, `pp_static_0.03` 2/2, `pp_static_0.05` 2/2.
Per-token L-inf ratios over 1.5x: `[1-2048]` prompt 1 tokens 0 (1.72x),
9 (1.79x) and 12 (2.58x), prompt 2 token 11 (1.82x); `[4-2048]` row 0,
prompt 1 token 0 (1.72x), prompt 2 tokens 5 (1.73x) and 11 (1.80x).

Error against the FP32 reference, per prompt (max over the 16 generated
tokens; `[4-2048]` prints batch row 0):

| Case | Prompt | CPU BF16 L-inf | Neuron L-inf | Neuron / CPU BF16 | CPU BF16 L2 | Neuron L2 | Neuron / CPU BF16 |
|---|---|---|---|---|---|---|---|
| `[1-2048]` | 1 of 2 | 0.0097 | 0.0104 | 1.07x | 0.0103 | 0.0099 | 0.96x |
| `[1-2048]` | 2 of 2 | 0.0106 | 0.0104 | 0.98x | 0.0125 | 0.0100 | 0.80x |
| `[4-2048]` | 1 of 2 | 0.0097 | 0.0095 | 0.98x | 0.0103 | 0.0096 | 0.93x |
| `[4-2048]` | 2 of 2 | 0.0106 | 0.0103 | 0.97x | 0.0125 | 0.0125 | 1.00x |

Generated token of each Neuron maximum: `[1-2048]` prompt 1 L-inf tokens 11
and 12, L2 token 12; prompt 2 L-inf token 11, L2 token 7. `[4-2048]` prompt 1
L-inf and L2 token 3; prompt 2 L-inf token 11, L2 token 13.

#### Serve smoke

`/v1/completions`, `temperature` 0, `max_tokens` 32; first 60 characters of each output as a Python string literal.

| Prompt | Output |
|---|---|
| `The capital of France is` | `" Paris. It is the largest city in France and the country's p"` |
| `1, 2, 3, 4, 5,` | `' 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16'` |
| `def fibonacci(n):` | `' \n    if n<0: \n        print("Incorrect input") \n    elif n='` |

### Exceptions

- **Topology changed from the port's runs.** The port's device logit run
  served the model as two TP 8 servers on two nodes (one prefill, one decode,
  KV transferred between them). `test/integration/test_model.py` serves it
  colocated on one node at the same TP 8, with the same model length (2048),
  prefill size and bucket (1024), BF16 and greedy on-device sampling. Batch 4
  is new; the port's runs used batch 1.
- **Tolerance map.** The port's logit test used the plugin's default
  thresholds. This test uses the template's bf16 dense-golden `TOL_MAP`.
- **Full-vocabulary `generate_fn`.** The test packs logits with its own
  `_full_vocab_generate_fn` instead of the plugin's
  `create_offline_vllm_generate_fn`, because the model scores 151552 ids and
  the tokenizer has 151329 base tokens; see the function's docstring. The
  comparison and the tolerance map are the plugin's.
- **Informational checks that did not pass.** `pp_tok_linf_1.5x` 0/2 in both
  cases, `pp_tok_linf_2.0x` 1/2 in `[1-2048]`, `agg_bc` FAIL in both cases,
  `agg_linf_1.2x` and `agg_linf_1.5x` FAIL in `[1-2048]`, `agg_l2_1.2x` FAIL
  in `[4-2048]` (see Logit validation).
- **Single-chunk prompts on device.** Both prompts are shorter than one
  1024-token chunk, so the device runs used the segmented-prefill graph with
  no prior KV. A prompt over several chunks is covered only by the CPU test
  `test_tiny_glm4_e2e.py[4-segmented]` (random weights).
- **Dropped: the port's disaggregated-inference test** (1P1D logits and a
  1024-in / 512-out serving request). It needs two nodes, a KV transfer
  connector, a proxy and MPI, and imports helpers from the plugin's test tree.
  Its logit check is what `test/integration/test_model.py` carries now, with
  the prompts copied locally.
- **Dropped: the port's device module tests** (layer-0 attention prefill,
  segmented prefill and decode, and MLP prefill and decode at TP 8 with the
  real weights, against HF). They were not run on vLLM Neuron 0.29.0.1.2.0.
  The CPU tiny decoder-layer test and the tiny end-to-end test cover the same
  code paths with random weights.
- **Rewritten assertion.** The port's CPU test asserted that
  `vllm_neuron.model.registry.get_models()` maps `Glm4ForCausalLM` to the
  factory. A contrib model is deliberately absent from that registry, so the
  shipped test checks the registration contract instead.
- Flash-attention prefill on device, prefix caching, context beyond 2048
  tokens, tensor parallel sizes other than 8 on device, speculative decoding
  and Trn3 are not tested.
