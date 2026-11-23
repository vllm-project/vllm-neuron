# HyperCLOVAX SEED Think 14B (Text)

Serve a HyperCLOVAX text model (`naver-hyperclovax/HyperCLOVAX-SEED-Think-14B`,
whose `config.json` declares `HyperCLOVAXForCausalLM`) with vLLM on AWS
Trainium through the `vllm-neuron` plugin. It is a 14.7B-parameter dense
decoder: 38 layers, grouped-query attention with 48 query heads and 8 KV heads
(head dimension 128), a SwiGLU MLP, an untied LM head over a 110,592-id
vocabulary, and two HyperCLOVAX-specific features: MuP scale factors
(embedding, attention softmax, residual branch and logit multipliers) and
peri-layer normalization (an extra RMSNorm on the output of each attention and
MLP branch, before the residual add). The checkpoint stores FP32 weights; the
implementation computes in BF16. The implementation is in `src/`, and
[src/README.md](src/README.md) describes the architecture and the per-feature
status.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix and the
> validation report below record exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`), then install this folder:

```bash
pip install -e contrib/models/hyperclovax
```

The install adds a `vllm.general_plugins` entry point (option 3 in
[contrib/README.md](../../README.md)) that registers `HyperCLOVAXForCausalLM`
in every vLLM process, so `vllm serve`, `LLM(...)` and tensor parallelism find
it with no other setup. vLLM maps this architecture name to its own
implementation, and the entry point replaces that mapping while the package is
installed. `pip uninstall vllm-neuron-contrib-hyperclovax` removes it.

The checkpoint is about 59 GB (FP32). The recorded runs used TP 8 on
`trn2.48xlarge`.

### Serve with the OpenAI-compatible server

```bash
vllm serve naver-hyperclovax/HyperCLOVAX-SEED-Think-14B \
    --revision 9b74e35d4c7e4ffec489f4171273caca8948a2b9 \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching
```

`--dtype bfloat16` is not optional: without it vLLM takes the FP32 dtype of the
checkpoint, while this implementation computes in BF16 (the loader casts each
weight). Prefix caching (on by default in vLLM) needs segmented prefill on
Neuron, which the plugin does not enable when `--max-num-batched-tokens` equals
`--max-model-len`, so this command turns it off. No recorded run had it on.

The first start compiles the prefill and decode graphs of the 14B model, which
can take longer than vLLM's default 600-second engine start-up wait. Set
`VLLM_ENGINE_READY_TIMEOUT_S=7200` in the environment if the server stops
with an engine start-up timeout.

Then send a request:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

With the folder installed as above, the offline API needs no registration
call:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="naver-hyperclovax/HyperCLOVAX-SEED-Think-14B",
    revision="9b74e35d4c7e4ffec489f4171273caca8948a2b9",
    dtype="bfloat16",
    tensor_parallel_size=8,
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

If you do not install the folder, copy `src/` into your application as an
importable package and register the factory before you construct
`LLM(...)` (option 1 in [contrib/README.md](../../README.md)):

```python
from vllm import ModelRegistry

# Your copy of contrib/models/hyperclovax/src/, under whatever name you gave it.
from your_package.factory import HyperCLOVAXForCausalLM

ModelRegistry.register_model("HyperCLOVAXForCausalLM", HyperCLOVAXForCausalLM)
```

That registration reaches vLLM's engine-core and worker processes only because
the offline path starts them with the `fork` start method (vLLM's default). Do
not set `VLLM_WORKER_MULTIPROC_METHOD=spawn` with it: a spawned child never
runs the call, resolves vLLM's own `HyperCLOVAXForCausalLM` instead, and fails
in the Neuron model runner on `from_configs`. The installed entry point has no
such limit.

The checkpoint's `config.json` has an `auto_map` that names a
`modeling_hyperclovax.py` file the checkpoint does not ship. Do not pass
`trust_remote_code=True` (or `--trust-remote-code`): vLLM and transformers
5.17 use their built-in HyperCLOVAX configuration without it.

Prefill runs sequence-parallel, so the plugin pads each prefill to a token
bucket that is larger than the TP degree and divisible by it.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1`; the tiny end-to-end unit test does exactly that. See
the top-level `README.md` and `AGENTS.md` for the full list of
`VLLM_NEURON_*` environment variables.

## Compatibility Matrix

| Instance / version | `vllm-neuron` 0.29.0.1.2.0 (Neuron SDK 2.33, vLLM 0.29.0) | Other versions |
| ------------------ | --------------------------------------------------------- | -------------- |
| Trn2 (trn2.48xlarge) | Working (TP 8): unit, integration and `vllm serve` smoke passed (see the Validation report) | Not tested |
| Trn3                 | Not tested                                              | Not tested     |

The native route (`VLLM_NEURON_BACKEND=neuron_native`) is the only route this
implementation targets.

## Example Checkpoints

- <https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-14B>. The
  tests pin revision `9b74e35d4c7e4ffec489f4171273caca8948a2b9`.

No other `HyperCLOVAXForCausalLM` checkpoint was run. The factory rejects a
checkpoint with an attention or MLP bias, a RoPE type other than `default`, an
activation other than SiLU, or a dtype other than FP32 or BF16.

## Testing

All test directories reach the implementation through `test/_local_src.py`,
which loads `src/` by file location under the unique module name
`contrib_hyperclovax_src`, so the tests work without the folder being
installed and after it is copied out of the repository. `test/conftest.py`
puts `test/` on `sys.path` for `_local_src.py` and the shared helper
`hyperclovax_test_utils.py` (checkpoint resolution), and shuts down the
plugin's worker pool at the end of the session.

Tests that need the real checkpoint read it from `HYPERCLOVAX_CHECKPOINT_PATH`,
a local snapshot of the pinned revision (the directory or its `files/`
subdirectory):

```bash
hf download naver-hyperclovax/HyperCLOVAX-SEED-Think-14B \
    --revision 9b74e35d4c7e4ffec489f4171273caca8948a2b9 \
    --local-dir /data/HyperCLOVAX-SEED-Think-14B
export HYPERCLOVAX_CHECKPOINT_PATH=/data/HyperCLOVAX-SEED-Think-14B
```

### Integration test

`test/integration/test_model.py` serves the model with the offline `LLM(...)`
API on one node at TP 8 and validates its raw per-token logits against a
HuggingFace reference (FP32 baseline plus a BF16 teacher-forced pass), at
batch 1 and batch 4 with a 2,048-token model length. It needs 8 NeuronCores.
Without `HYPERCLOVAX_CHECKPOINT_PATH` it downloads the pinned revision. The HF
reference runs on the host CPU in FP32 (the loader reads the weights in BF16
and then casts the model to FP32), so the host needs about 90 GB of free memory
for it.

```bash
pytest contrib/models/hyperclovax/test/integration/test_model.py \
    -v -s --timeout=14400
```

### Unit tests

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/hyperclovax/test/unit/ \
    -v --timeout=7200
```

- `test_factory.py` (no checkpoint): the registration contract (the factory
  that `src/` exports passes vLLM's model-protocol checks, and the
  `pyproject.toml` entry point registers it), the config validation and parse,
  the checkpoint mappings, the vendored 0.29 helper in `src/compat.py`, the
  O projection in `src/o_proj.py` (against the plugin's, and its FP32-input
  fallback against the NKI kernel in the CPU simulator), and the uint32 KV
  cache write slots of the decode path.
- `test_checkpoint.py`: the config, the 421 weight-index keys and the dtype and
  shape of every tensor of the released checkpoint. Without
  `HYPERCLOVAX_CHECKPOINT_PATH` it downloads only `config.json` and
  `model.safetensors.index.json` and reads the safetensors headers from the
  hub; the one case that reads tensor values skips.
- `test_tiny_hyperclovax_e2e.py` (no checkpoint, no hardware): writes a 1-layer
  random-weight checkpoint in FP32 and in BF16, with MuP multipliers other
  than 1 and distinct post-norm weights, and runs the full vLLM pipeline on
  CPU (NKI kernels in the simulator) at TP 2 and TP 1. Each case runs in its
  own spawned process. The TP 2 cases start vLLM's multiprocess executor,
  whose message queues need about 240 MiB of `/dev/shm`; in a container, run
  with `--shm-size=1g` (Docker's default is 64 MiB) or `--ipc=host`.

## Validation report

### vLLM Neuron 0.29.0.1.2.0 (2026-11-23)

#### Environment

| Image | Instance | TP | vLLM | vllm-neuron | torch | transformers | neuronx-cc |
|---|---|---|---|---|---|---|---|
| `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` | `trn2.48xlarge` (LNC 2), 1 node | 8 | 0.29.0 | 0.29.0.1.2.0 | 2.13.0 | 5.17.0 | 2.28.7631.0+8f7b0cf5 |

Checkpoint of the integration test: `naver-hyperclovax/HyperCLOVAX-SEED-Think-14B` at
`9b74e35d4c7e4ffec489f4171273caca8948a2b9`.

#### Results

| Test | Configuration | Result |
|---|---|---|
| CPU unit tests, `test/unit/` | `VLLM_NEURON_CPU_MODE=1`, `HYPERCLOVAX_CHECKPOINT_PATH` set | 48 passed, 0 failed (`test_factory.py` 40, `test_checkpoint.py` 4, `test_tiny_hyperclovax_e2e.py` 4) |
| `test_model_accuracy[1-2048]` | TP 8, `neuron_native`, BF16, batch 1, max model length 2,048, 2 prompts × 16 greedy tokens, on-device greedy sampling | PASSED (216.41 s): static per-prompt bounds 2/2 (below); aggregate σ-ratio 0.8671 ≤ 1.0 |
| `test_model_accuracy[4-2048]` | TP 8, `neuron_native`, BF16, batch 4 (each prompt in 4 rows), max model length 2,048, 2 prompts × 16 greedy tokens, on-device greedy sampling | PASSED (168.21 s): static per-prompt bounds 2/2 (below); aggregate σ-ratio 0.8932 ≤ 1.0 |
| `vllm serve` smoke | The Usage command (`--revision 9b74e35d4c7e4ffec489f4171273caca8948a2b9`), `VLLM_NEURON_BACKEND=neuron_native`, installed with `pip install --no-deps -e contrib/models/hyperclovax` | Passed: 3/3 prompts, 32 completion tokens each; a batched request of all 3 at `max_tokens` 16 returned a prefix of each |

The integration test also asserts the per-prompt FP32 error checks `pp_max_linf_1.5x` and `pp_max_l2_1.5x`: 2/2 in both cases.

Not run on 0.29: the port's two-node component, logit and serving tests (not shipped in this folder).

#### Logit validation

| Case | Prompt | Overall Status | Max K5 (0.011) | Max K50 (0.02) | Max K1000 (0.03) | Max All (0.05) | Argmax divergence (1 ULP) |
|---|---|---|---|---|---|---|---|
| `[1-2048]` | 1 of 2, `The Amazon rainforest …` | PASSED | 0.0079 at token 4 | 0.0084 at token 6 | 0.0084 at token 6 | 0.0084 at token 6 | 0.0000 |
| `[1-2048]` | 2 of 2, `The human genome …` | PASSED | 0.0101 at token 14 | 0.0101 at token 14 | 0.0104 at token 8 | 0.0104 at token 8 | 0.0000 |
| `[4-2048]` | 1 of 2, `The Amazon rainforest …` | PASSED | 0.0070 at token 15 | 0.0070 at token 15 | 0.0078 at token 14 | 0.0087 at token 3 | 0.0000 |
| `[4-2048]` | 2 of 2, `The human genome …` | PASSED | 0.0086 at token 14 | 0.0144 at token 14 | 0.0144 at token 14 | 0.0144 at token 14 | 0.0000 |

Neuron vs CPU BF16 reference; token = generated token, row = batch row. In `[4-2048]` the per-row maxima are equal in all 4 rows; the token is the one printed for row 0. Prompt 1, token 10, in both cases (all 4 rows of `[4-2048]`): the Neuron argmax differs from the reference argmax with Δ = 0.0000 ("Acceptable divergence", within 1 ULP); validation continued teacher-forced (K5 0.0001 / All 0.0051 at that token in `[1-2048]`, K5 0.0011 / All 0.0060 in `[4-2048]`). No other argmax difference.
Both cases passed (`multi_prompt_logit_validation`; the test asserts `result.passed` only): two-way 2/2 prompts, aggregate σ-ratio 0.8671 (`[1-2048]`) and 0.8932 (`[4-2048]`) ≤ 1.0. Printed per-prompt checks, `[1-2048]`: `pp_max_l2_1.5x` 2/2, `pp_max_l2_2.0x` 2/2, `pp_max_linf_1.5x` 2/2, `pp_max_linf_2.0x` 2/2, `pp_static_0.03` 2/2, `pp_static_0.05` 2/2, `pp_tok_l2_1.5x` 1/2, `pp_tok_l2_2.0x` 1/2, `pp_tok_linf_1.5x` 1/2, `pp_tok_linf_2.0x` 2/2; `[4-2048]`: the same except `pp_tok_l2_1.5x` 0/2 and `pp_tok_l2_2.0x` 2/2. Printed informational aggregate checks, `[1-2048]`: `agg_bc` FAIL, `agg_l2_1.2x` FAIL, `agg_l2_1.5x` PASS, `agg_linf_1.2x` PASS, `agg_linf_1.5x` PASS; `[4-2048]`: `agg_bc` FAIL, `agg_l2_1.2x` FAIL, `agg_l2_1.5x` FAIL, `agg_linf_1.2x` FAIL, `agg_linf_1.5x` PASS.

Error against the FP32 reference, per prompt (max over the 16 generated tokens):

| Case | Prompt | CPU BF16 L-inf | Neuron L-inf | Neuron / CPU BF16 | CPU BF16 L2 | Neuron L2 | Neuron / CPU BF16 |
|---|---|---|---|---|---|---|---|
| `[1-2048]` | 1 of 2 | 0.0080 | 0.0072 | 0.90x | 0.0126 | 0.0097 | 0.77x |
| `[1-2048]` | 2 of 2 | 0.0134 | 0.0111 | 0.83x | 0.0314 | 0.0289 | 0.92x |
| `[4-2048]` | 1 of 2 | 0.0080 | 0.0088 | 1.10x | 0.0126 | 0.0102 | 0.81x |
| `[4-2048]` | 2 of 2 | 0.0134 | 0.0112 | 0.84x | 0.0314 | 0.0285 | 0.91x |

CPU BF16 maxima: prompt 1 at generated token 3, prompt 2 at token 13. Neuron maxima (L-inf, L2): `[1-2048]` prompt 1 tokens 13 and 3, prompt 2 tokens 13 and 8; `[4-2048]` prompt 1 token 15 for both, prompt 2 tokens 13 and 8.

#### Serve smoke

`/v1/completions`, `temperature` 0, `max_tokens` 32; first 60 characters of each output as a Python string literal.

| Prompt | Output |
|---|---|
| `The capital of France is` | `' Paris. The capital of Germany is Berlin. The capital of Jap'` |
| `1, 2, 3, 4, 5,` | `' 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16'` |
| `def fibonacci(n):` | `' \n    if n <= 0: \n        return "Invalid input. Please ente'` |

### Exceptions

- **Prompt 1 has one argmax difference within the 1-ULP bound** (generated
  token 10, Δ = 0.0000, both cases). The informational aggregate checks
  `agg_bc` and `agg_l2_1.2x` failed in both cases, and `agg_l2_1.5x` and
  `agg_linf_1.2x` in `[4-2048]` (see Logit validation).
- **Port tests not shipped and not run on 0.29.** The port's two-node TP 8
  device run (layer-0 attention in flash prefill at 128 and 1,024 tokens,
  segmented prefill in two 512-token chunks and decode, the MLP, full-model
  logits with host sampling, one 1,024-token prompt with 512 output tokens
  through `vllm serve`, and per-rank NKI kernel-dispatch checks) passed on a
  development build of the plugin. Those tests depend on a two-node launcher
  and shared test infrastructure outside this folder, and the kernel-dispatch
  trace they read is plugin instrumentation that this folder does not carry.
- **The integration test differs from the port's logit run**: one node instead
  of two, the offline API with on-device greedy sampling instead of
  `vllm serve` with host sampling, an added batch-4 case, and logit rows over
  the full 110,592-id vocabulary with detokenization off and end-of-sequence
  ignored.
- **`src/o_proj.py`** carries the 0.29.0.1.2.0 `o_proj` with the port's
  FP32-input torch fallback. The CPU unit tests exercise the fallback and
  compare it with the NKI kernel in the simulator; which path the device
  graphs used was not recorded.
- **The serve smoke ran with `VLLM_NEURON_BACKEND=neuron_native` and
  `VLLM_ENGINE_READY_TIMEOUT_S=7200` set**; the Usage command sets neither.
- Prefix caching (the segmented-prefill path), TP sizes other than 1 and 2
  (CPU) and 8, context beyond 2,048 tokens, quantization, speculative
  decoding, other HyperCLOVAX checkpoints and Trn3 are not tested.
