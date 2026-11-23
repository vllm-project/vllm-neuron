# Contrib Model: StarCoder2-3B (Text, Code)

[StarCoder2-3B](https://huggingface.co/bigcode/starcoder2-3b) is a 3B-parameter
dense decoder-only code model: 30 layers, hidden size 3072, grouped-query
attention with 24 query heads and 2 KV heads (head size 128), default RoPE
with theta 999999.44, and a 49152-token vocabulary with tied input and output
embeddings. It differs from a Llama-style decoder in three places:

- LayerNorm with a bias (`norm_epsilon` 1e-5) in place of RMSNorm.
- A non-gated MLP, `c_proj(gelu_tanh(c_fc(x)))`, with intermediate size 12288.
- A bias on every linear projection: Q, K, V, O, `c_fc` and `c_proj`.

This folder serves it with vLLM on AWS Trainium2 through the `vllm-neuron`
plugin, in BF16 with tensor parallelism. The implementation under `src/` uses
the plugin's shared operators and NKI kernels: fused QKV projection with bias,
flash attention for prefill, the fused decode attention kernel (QKV, RoPE, KV
cache write and O projection with biases), and the fused MLP kernel with the
GELU tanh activation.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix and the
> validation report below record exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`), then install this folder:

```bash
pip install -e contrib/models/starcoder2
```

The install adds a `vllm.general_plugins` entry point (`src/plugin.py`) that
registers `Starcoder2ForCausalLM` in every vLLM process: the API server, the
engine core and each tensor-parallel worker. So `vllm serve` and `LLM(...)`
find it with no other setup. vLLM maps `Starcoder2ForCausalLM` to its own
Transformers backend; while this package is installed, the entry point
replaces that mapping with the Neuron class in `src/`. `pip uninstall
vllm-neuron-contrib-starcoder2` removes the registration.

The recorded runs set `VLLM_NEURON_BACKEND=neuron_native`, so the commands
below set it too.

### Serve with the OpenAI-compatible server

```bash
VLLM_NEURON_BACKEND=neuron_native vllm serve bigcode/starcoder2-3b \
    --revision 733247c55e3f73af49ce8e9c7949bf14af205928 \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --tensor-parallel-size 4 \
    --max-model-len 2048 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 1 \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{"neuron_config": {"quantization": "bf16", "on_device_sampling_config": {"all_greedy": true}, "num_batched_tokens_buckets": [256, 1024], "num_seqs_buckets": [1]}}'
```

The two bucket lists are the prefill and batch shapes of the recorded run;
each bucket compiles its own graph.

Then send a request:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "bigcode/starcoder2-3b",
        "prompt": "def fibonacci(n):",
        "max_tokens": 32,
        "temperature": 0
    }'
```

StarCoder2-3B is a base code model, not a chat model, so use the completions
endpoint.

### Offline inference with the Python API

```python
from vllm import LLM, SamplingParams

# With this folder pip-installed, the plugin entry point has already
# registered Starcoder2ForCausalLM. Run with VLLM_NEURON_BACKEND=neuron_native.
llm = LLM(
    model="bigcode/starcoder2-3b",
    revision="733247c55e3f73af49ce8e9c7949bf14af205928",
    tensor_parallel_size=4,
    dtype="bfloat16",
    max_model_len=2048,
    max_num_batched_tokens=1024,
    max_num_seqs=1,
    enable_prefix_caching=False,
    additional_config={
        "neuron_config": {
            "quantization": "bf16",
            "on_device_sampling_config": {"all_greedy": True},
            "num_batched_tokens_buckets": [256, 1024],
            "num_seqs_buckets": [1],
        }
    },
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["def fibonacci(n):"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

Without the install, copy `src/` into your application and call
`ModelRegistry.register_model("Starcoder2ForCausalLM", <the factory class>)`
before you construct `LLM(...)` (option 1 in
[contrib/README.md](../../README.md)). That registration reaches the engine
core and the workers only because the offline path forks them; under
`VLLM_WORKER_MULTIPROC_METHOD=spawn` they resolve vLLM's Transformers backend
class instead, which fails in the Neuron model runner with an `AttributeError`
on `from_configs`. `test/integration/test_model.py` uses this option.

Configuration notes:

- `quantization: "bf16"` is required: the checkpoint stores FP32 weights, and
  the implementation computes in BF16. The factory refuses any other
  quantization.
- The implementation does not apply StarCoder2's 4096-token sliding window.
  The factory therefore refuses a `--max-model-len` larger than the window, so
  every sequence fits inside it and full causal attention is exact.
- Prefill is sequence-parallel: a prefill bucket must be a multiple of the
  tensor-parallel size and larger than it.
- 24 query heads and 2 KV heads: the tensor-parallel size must divide 24 and be
  a multiple of 2 (or divide 2). Only TP 4 was run.

To iterate without Neuron hardware, set `VLLM_NEURON_CPU_MODE=1`; the tiny
end-to-end unit test does exactly that. See the top-level `README.md` for the
`VLLM_NEURON_*` environment variables.

### How the MLP runs

The MLP is non-gated with biases. On Neuron, `NF.mlp` runs the fused NKI MLP
kernel with `skip_gate=True` and `act_fn=ActFnType.GELU_Tanh_Approx`, so the
kernel computes the activation. At TP 4 the per-rank shapes (hidden 3072,
intermediate 3072) pass the kernel's eligibility checks in `NF.mlp` for both
decode (1 to 4 tokens) and prefill (256 and 1,024 tokens), so the device path
never calls `torch.nn.functional.gelu`.

The model calls `NF.mlp` only when the plugin's own kernel predicate accepts
the shape. Otherwise `NF.mlp` would take its internal torch fallback, which
calls `F.gelu` inside the traced graph. That happens, for example, for the
prefill kernel at an intermediate size per rank above 4,096 (TP 1 and TP 2).
For those shapes, and whenever NKI kernels cannot run at all (CPU mode, or
`VLLM_NEURON_DISABLE_NKI_KERNELS=1`), `Starcoder2MLP._torch_mlp` computes the
same arithmetic as the kernel, in FP32. It writes the GELU tanh formula with
`tanh`, `mul` and `add` rather than calling `F.gelu`: when
`VLLM_NEURON_BACKEND` is not `neuron_native`, `libtorch_neuronx_lite` replaces
`torch._C._nn.gelu` with a function that `torch.compile` cannot trace.
`test/unit/test_loading.py` checks the formula, that the fallback never calls
torch's GELU (`test_mlp_fallback_without_torch_gelu`), and which shapes go to
the kernel (`test_mlp_kernel_routing`).

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn2 (`trn2.48xlarge`) | TP 4, see [Validation report](#validation-report) | Not tested |
| Trn3                   | Not tested | Not tested |

| Feature | Status |
|---|---|
| BF16 (from the FP32 checkpoint) | Supported |
| Tensor parallelism | TP 4 run; other sizes not run |
| Context length | Up to 2048 tokens run; the factory refuses more than the 4096-token sliding window |
| Prefix caching, segmented prefill | Present in the code, not run (every recorded run disabled prefix caching) |
| Sliding-window attention beyond 4096 tokens | Not implemented; refused by the factory |
| Quantized checkpoints, RoPE scaling, `use_bias=False` | Refused by the factory |
| Speculative decoding | Present in the code, not run |

## Example Checkpoints

- <https://huggingface.co/bigcode/starcoder2-3b> — the tests pin revision
  `733247c55e3f73af49ce8e9c7949bf14af205928`.

Other StarCoder2 checkpoints (7B, 15B) share the architecture but were not
run; their head counts and sliding windows differ, so check the
tensor-parallel and context-length notes under Usage.

## Testing

The tests reach the implementation through `test/_local_src.py`, which loads
`src/` by file location under the unique module name `contrib_starcoder2_src`.
Nothing in the tests needs the repository root on `sys.path` or this package
installed. `test/starcoder2_test_utils.py` holds the shared constants and the
checkpoint helpers, and `test/conftest.py` puts `test/` on `sys.path` and
provides the `dist_cleanup` and worker-pool cleanup fixtures.

The checkpoint is about 12 GB (FP32). Tests that need the weights read them
from `STARCODER2_CHECKPOINT_PATH`, a local copy of the pinned revision:

```bash
hf download bigcode/starcoder2-3b \
    --revision 733247c55e3f73af49ce8e9c7949bf14af205928 \
    --local-dir /data/starcoder2-3b
export STARCODER2_CHECKPOINT_PATH=/data/starcoder2-3b
```

### Integration test

The integration test serves the model through the offline `LLM(...)` API on
one node at TP 4, and validates the raw logits of two prompts (16 decode
tokens each) against a HuggingFace golden reference, with the strict `1e-5`
argmax bound and top-K logit bounds of 0.03 (K=5), 0.05 (K=50), 0.08 (K=1000)
and 0.09 (all). It needs 4 NeuronCores. Without `STARCODER2_CHECKPOINT_PATH`
it downloads the pinned revision.

```bash
VLLM_NEURON_BACKEND=neuron_native \
pytest contrib/models/starcoder2/test/integration/test_model.py \
    -v -s --timeout=7200
```

### Unit tests

`test/unit/` needs no Neuron hardware:

- `test_loading.py`: the registration contract and the plugin entry point,
  the config parse, the factory checks, the TP size checks, the checkpoint
  mapping coverage and the per-rank shapes (from the safetensors header of the
  pinned revision, about 400 KB from the Hub), the weight-loader transforms,
  the MLP fallback without torch's GELU, and a full 4-rank CPU load of the
  checkpoint (skips without `STARCODER2_CHECKPOINT_PATH`).
- `test_tiny_e2e.py`: the full vLLM pipeline at TP 4 on a random-weight tiny
  StarCoder2 (2 layers, 1 KV head per rank as in the real model), compared
  with the HF reference by the three-way rule.

```bash
VLLM_NEURON_CPU_MODE=1 VLLM_NEURON_BACKEND=neuron_native \
pytest contrib/models/starcoder2/test/unit/ -v -s --timeout=3600
```

### Module tests

`test/module/test_components.py` compares layer-0 attention and the layer-15
MLP with the HuggingFace modules (the three-way pattern in the
[module test guidelines](../../../docs/design/accuracy/module_test_guidelines.md)),
at TP 4, in prefill and in decode. Every case needs the checkpoint weights.
`test_attention_prefill` and `test_mlp_prefill` also run in CPU mode; the two
decode cases need 4 NeuronCores.

```bash
pytest contrib/models/starcoder2/test/module/ -v -s --timeout=3600
```

## Validation report

### vLLM Neuron 0.29.0.1.2.0 (2026-11-19)

#### Environment

| Image | Instance | TP | vLLM | vllm-neuron | torch | transformers | neuronx-cc |
|---|---|---|---|---|---|---|---|
| `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` | `trn2.48xlarge` (LNC 2), 1 node | 4 | 0.29.0 | 0.29.0.1.2.0 | 2.13.0 | 5.17.0 | 2.28.7631.0+8f7b0cf5 |

#### Results

| Test | Configuration | Result |
|---|---|---|
| CPU unit tests, `test/unit/` | `VLLM_NEURON_CPU_MODE=1`, `VLLM_NEURON_BACKEND=neuron_native`, `STARCODER2_CHECKPOINT_PATH` set (`test_full_checkpoint_load[4]` ran) | 16 passed |
| Integration logit test, `test_model_accuracy[1-2048]` | batch 1, max model length 2,048, 1,024 batched tokens, TP 4, `VLLM_NEURON_BACKEND=neuron_native` | PASSED (126.27 s) on aggregate σ-ratio 0.7689 ≤ 1.0; 1 of 2 prompts within the per-prompt top-K bounds (prompt 1 FAILED, below) |
| `vllm serve` smoke | the serve command under Usage | Passed, 3 of 3 prompts |

Not run on 0.29: module tests (`test/module/`), not part of the 0.29 run.

#### Logit validation

| Case | Prompt | Overall Status | Max divergence (1 ULP) | Max K5 (0.03) | Max K50 (0.05) | Max K1000 (0.08) | Max All (0.09) |
|---|---|---|---|---|---|---|---|
| `[1-2048]` | 1/2, `The Amazon rainforest is the largest tropical rainforest in the world, …` | FAILED | 0.0000 | 0.0418 (> 0.03) | 0.0837 (> 0.05) | 0.0930 (> 0.08) | 0.0247 |
| `[1-2048]` | 2/2, `The human genome contains approximately three billion base pairs of DNA, …` | PASSED | 0.0000 | 0.0175 | 0.0396 | 0.0568 | 0.0239 |

Prompt 1: K5 0.0418 > 0.03, K50 0.0837 > 0.05 and K1000 0.0930 > 0.08 at generated token 0, the only token of 16 out of bounds; two-way validation 1/2 prompts. Test passed on aggregate σ-ratio 0.7689 ≤ 1.0 (`multi_prompt_logit_validation`).
`[1-2048]`: aggregate σ-ratio 0.7689 (≤ 1.0 = pass), `agg_bc` FAIL; `pp_max_l2_1.5x`, `pp_max_l2_2.0x`, `pp_max_linf_1.5x`, `pp_max_linf_2.0x`, `pp_static_0.03`, `pp_static_0.05`, `pp_tok_l2_1.5x`, `pp_tok_l2_2.0x`, `pp_tok_linf_1.5x`, `pp_tok_linf_2.0x` 2/2 each.
16 generated tokens per prompt vs the HF BF16 reference (FP32 baseline); no argmax divergence.

Error against the FP32 reference, per prompt (max over the 16 generated tokens; ratio = Neuron max / CPU BF16 max):

| Case | Prompt | CPU BF16 L-inf | Neuron L-inf | Neuron / CPU BF16 | CPU BF16 L2 | Neuron L2 | Neuron / CPU BF16 |
|---|---|---|---|---|---|---|---|
| `[1-2048]` | 1/2 | 0.0243 | 0.0129 | 0.53x | 0.0190 | 0.0193 | 1.02x |
| `[1-2048]` | 2/2 | 0.0217 | 0.0085 | 0.39x | 0.0261 | 0.0248 | 0.95x |

#### Serve smoke

`/v1/completions`, max_tokens 32, temperature 0; 32 tokens returned per prompt. Batched request (all three, max_tokens 16): a prefix of the same text.

| Prompt | Start of the completion (first 60 characters) |
|---|---|
| `The capital of France is` | ``  Paris.\n\n```\n\nI want to get the value of the attribute `valu `` |
| `1, 2, 3, 4, 5,` | ` 6, 7, 8, 9, 10, 11, 12, 13, 14,` (the whole output) |
| `def fibonacci(n):` | `\n    if n == 0:\n        return 0\n    elif n == 1:\n        re` |

### Exceptions

- **Topology.** The port's recorded logit run spread TP 4 over 2 nodes (2 ranks
  per node). This folder ships a single-node TP 4 test instead; the multi-node
  launcher and its serving harness were not carried over.
- **Untested configurations.** Prefix caching and segmented prefill, tensor
  parallel sizes other than 4, context beyond 2,048 tokens, batch sizes above
  1, speculative decoding, and Trn3.
