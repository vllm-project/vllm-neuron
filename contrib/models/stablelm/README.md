# Contrib Model: StableLM Zephyr 3B (Text)

[StableLM Zephyr 3B](https://huggingface.co/stabilityai/stablelm-zephyr-3b) is
a 2.8B-parameter dense decoder-only chat model: 32 layers, hidden size 2560,
multi-head attention with 32 query and 32 KV heads of size 80, a SwiGLU MLP of
size 6912, LayerNorm with a bias, partial RoPE over the first 25% of each head
(theta 10000) and a 50304-entry vocabulary with an untied LM head.

This folder serves it with vLLM on AWS Trainium2 through the `vllm-neuron`
plugin, in BF16 with tensor parallelism. The model uses the plugin's shared
operators and NKI kernels (fused QKV projection, flash and segmented attention,
O projection, fused MLP); decode attention runs in torch, because the decode
megakernel's RoPE stage does not accept head size 80. See
[src/README.md](src/README.md) for the architecture, the partial-RoPE layout and
the feature status.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix and the
> validation report below record exactly what was tested.

## Usage

Install `vllm-neuron` 0.29.0.1.2.0 in your Neuron environment (see the
top-level `README.md`), for example the Neuron Deep Learning Container
`public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04`,
then install this folder:

```bash
pip install -e contrib/models/stablelm
```

The install adds a `vllm.general_plugins` entry point that registers
`StableLmForCausalLM` in every vLLM process (front end, engine core and each
Neuron worker), so `vllm serve` and `LLM(...)` find it with no other setup. It
replaces vLLM's GPU `StableLmForCausalLM` while installed;
`pip uninstall vllm-neuron-contrib-stablelm` removes it.

### Serve with the OpenAI-compatible server

```bash
vllm serve stabilityai/stablelm-zephyr-3b \
    --revision fe1fd5e36ccb8cb2cb1dc225ab6d2962692b9837 \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching
```

`--host 127.0.0.1` keeps the server on the local machine. Prefix caching (on
by default in vLLM) runs the segmented-prefill path, which no test of this
model covers, so the command turns it off.

Then send a request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "stabilityai/stablelm-zephyr-3b",
        "messages": [{"role": "user", "content": "Name three prime numbers."}],
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

```python
from vllm import LLM, SamplingParams

# With this folder installed (pip install -e contrib/models/stablelm), the
# plugin entry point registers StableLmForCausalLM in every vLLM process.
llm = LLM(
    model="stabilityai/stablelm-zephyr-3b",
    revision="fe1fd5e36ccb8cb2cb1dc225ab6d2962692b9837",
    tensor_parallel_size=8,
    dtype="bfloat16",
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

Without the install, copy `src/` into your application as a package and
register the factory before you construct `LLM(...)`:
`ModelRegistry.register_model("StableLmForCausalLM", StableLmForCausalLM)`
(option 1 in [contrib/README.md](../../README.md)). That registration reaches
vLLM's engine-core and worker processes only through the default `fork` start
method; with `VLLM_WORKER_MULTIPROC_METHOD=spawn`, install the folder instead.

To iterate without Neuron hardware, set `VLLM_NEURON_CPU_MODE=1` and pass
`enforce_eager=True` (see `test/unit/test_tiny_e2e.py`). See the top-level
`README.md` and `AGENTS.md` for the `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn2 (`trn2.48xlarge`) | See the Validation report: TP 8, one node | Not tested |
| Trn3                   | Not tested | Not tested |

The software is the 0.29.0.1.2.0 Neuron Deep Learning Container named under
Usage, as shipped: vLLM 0.29.0, vLLM Neuron 0.29.0.1.2.0, transformers 5.17.0,
torch 2.13.0, Python 3.14. CPU mode (`VLLM_NEURON_CPU_MODE=1`) in that
container: all unit tests pass (see Testing).

## Example Checkpoints

- <https://huggingface.co/stabilityai/stablelm-zephyr-3b> at revision
  `fe1fd5e36ccb8cb2cb1dc225ab6d2962692b9837`, the revision the tests pin. The
  checkpoint has its own license (see the model card).

`stabilityai/stablelm-3b-4e1t` has the same architecture fields
(`partial_rotary_factor` 0.25, no QKV bias, no QK LayerNorm, sequential
residual, BF16) but was not run. The StableLM 2 checkpoints are not supported:
the 1.6B models set `use_qkv_bias` and store FP16 weights, and the 12B model
sets `qk_layernorm` and `use_parallel_residual`. The factory refuses each of
these options at construction.

## Testing

The tests read the checkpoint from `STABLELM_CHECKPOINT_PATH` (a local
directory). Without it, the device tests download the pinned revision from the
HuggingFace hub (about 5.6 GB). The CPU unit tests need no checkpoint.

### Integration test

Serves the model through vLLM on Neuron at TP 8 on one `trn2.48xlarge` node,
batch 1 and batch 4 at a 2048-token model length, and validates the raw
logits of 16 decoded tokens for two prompts against a HuggingFace reference
(`multi_prompt_logit_validation`, three-way: FP32 HF, BF16 HF, Neuron). The
tolerance map is the plugin's default map, the one the port's own logit test
passed with: top-K bounds 0.011 (K=5), 0.02 (K=50), 0.03 (K=1000) and 0.05
(all). The argmax-divergence bound is the plugin's default, 1 BF16 ULP.

```bash
STABLELM_CHECKPOINT_PATH=/data/stablelm-zephyr-3b \
    pytest contrib/models/stablelm/test/integration/test_model.py \
    -v -s --timeout=7200
```

The model scores 50304 ids, but the tokenizer has 50277 tokens. The test
therefore requests the logits of all 50304 ids itself, and maps the padding
ids, which have no token string, to `""` in the engine's tokenizer, because
vLLM 0.29 decodes the string of every returned logprob id.

### Unit tests (CPU mode)

No Neuron hardware and no download:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/stablelm/test/unit/ -v
```

- `test_loading.py`: the plugin entry point and vLLM's model protocol checks,
  the config parse of the pinned `config.json`, the factory checks, checkpoint
  mapping coverage against the HF tensor names, the partial-RoPE QKV loader,
  and a TP 2 load of a tiny random-weight checkpoint through `load_weights`.
- `test_components.py`: partial RoPE and LayerNorm against HF, then attention
  prefill, attention prefill to decode and the MLP (prefill and decode)
  against HF at TP 2 (three-way), on the tiny checkpoint.
- `test_tiny_e2e.py`: the full vLLM offline pipeline at TP 2 on the tiny
  checkpoint, with the raw logits of every vocabulary id checked against HF.

The tiny checkpoint has 2 layers and hidden size 1280, and keeps head size
80, the 0.25 rotary factor and the LayerNorm bias; at TP 2 it has the 864 MLP
columns per rank of the real model at TP 8.

### Module tests (Neuron)

Attention prefill, attention prefill to decode and the MLP (prefill and
decode) at TP 8 on Neuron with the real checkpoint, against HF layer 0
(three-way, see the
[module test guidelines](../../../docs/design/accuracy/module_test_guidelines.md)):

```bash
STABLELM_CHECKPOINT_PATH=/data/stablelm-zephyr-3b \
    pytest contrib/models/stablelm/test/module/ -v -s --timeout=3600
```

All test directories reach the implementation through `test/_local_src.py`,
which loads `src/` by file location under the module name
`contrib_stablelm_src`, so no test needs the repository root on `sys.path` or
the folder installed. `test/stablelm_test_utils.py` holds the shared helpers
(checkpoint resolution, the tiny checkpoint, the HF references and the
component runners).

## Validation report

### vLLM Neuron 0.29.0.1.2.0 (2026-11-19)

#### Environment

| Image | Instance | TP | vLLM | vllm-neuron | torch | transformers | neuronx-cc |
|---|---|---|---|---|---|---|---|
| `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` | `trn2.48xlarge` (LNC 2), 1 node | 8 | 0.29.0 | 0.29.0.1.2.0 | 2.13.0 | 5.17.0 | 2.28.7631.0+8f7b0cf5 |

Checkpoint of the integration test: `stabilityai/stablelm-zephyr-3b` at
`fe1fd5e36ccb8cb2cb1dc225ab6d2962692b9837`.

#### Results

| Test | Configuration | Result |
|---|---|---|
| CPU unit tests, `test/unit/` | `VLLM_NEURON_CPU_MODE=1` | 13 passed, 0 failed |
| `test_model_accuracy[1-2048]` | TP 8, `neuron_native`, BF16, batch 1, max model length 2,048, 2 prompts × 16 greedy tokens, on-device greedy sampling | PASSED (174.77 s) on aggregate σ-ratio 0.6857 ≤ 1.0; static per-prompt bounds 0/2 (below) |
| `test_model_accuracy[4-2048]` | TP 8, `neuron_native`, BF16, batch 4 (each prompt in 4 rows), max model length 2,048, 2 prompts × 16 greedy tokens, on-device greedy sampling | PASSED (223.25 s) on aggregate σ-ratio 0.7430 ≤ 1.0; static per-prompt bounds 0/2 (below) |
| `vllm serve` smoke | The Usage command without `--revision` (served revision `main`), installed with `pip install --no-deps -e contrib/models/stablelm` | Passed: 3/3 prompts, 32 completion tokens each; a batched request of all 3 at `max_tokens` 16 returned a prefix of each |

Not run on 0.29: module tests (`test/module/`), not part of the 0.29 run.

#### Logit validation

| Case | Prompt | Overall Status | Max K5 (0.011) | Max K50 (0.02) | Max K1000 (0.03) | Max All (0.05) | Argmax divergence (1 ULP) |
|---|---|---|---|---|---|---|---|
| `[1-2048]` | 1 of 2, `The Amazon rainforest …` | FAILED | 0.1055 (> 0.011) at token 0 | 0.1055 (> 0.02) at token 0 | 0.1055 (> 0.03) at token 0 | 0.1055 (> 0.05) at token 0 | 0.0000 |
| `[1-2048]` | 2 of 2, `The human genome …` | FAILED | 0.0475 (> 0.011) at token 6 | 0.0475 (> 0.02) at token 6 | 0.0482 (> 0.03) at token 8 | 0.0559 (> 0.05) at token 1 | 0.0000 |
| `[4-2048]` | 1 of 2, `The Amazon rainforest …` | FAILED | 0.1055 (> 0.011) at token 0, all 4 rows | 0.1055 (> 0.02) at token 0, all 4 rows | 0.1055 (> 0.03) at token 0, all 4 rows | 0.1055 (> 0.05) at token 0, all 4 rows | 0.0000 |
| `[4-2048]` | 2 of 2, `The human genome …` | FAILED | 0.0397 (> 0.011) at token 5, row 2 | 0.0397 (> 0.02) at token 5, row 2 | 0.0513 (> 0.03) at token 8, row 0 | 0.0598 (> 0.05) at token 8, row 0 | 0.0000 |

Neuron vs CPU BF16 reference; token = generated token, row = batch row. Tokens over a bound: `[1-2048]` prompt 1 tokens 0 and 13 (token 13: K5 0.0164), prompt 2 tokens 0 to 11 and 14; `[4-2048]` prompt 1 token 0 and token 8 (rows 0, 2 and 3: K5 0.0115), prompt 2 tokens 0 to 11 and 14. No argmax divergence.
Both cases passed on aggregate σ-ratio 0.6857 (`[1-2048]`) and 0.7430 (`[4-2048]`) ≤ 1.0 (`multi_prompt_logit_validation`; the test asserts `result.passed` only); `agg_bc` FAIL; printed per-prompt checks, the same in both cases: `pp_max_linf_1.5x` 2/2, `pp_max_linf_2.0x` 2/2, `pp_max_l2_1.5x` 2/2, `pp_max_l2_2.0x` 2/2, `pp_tok_linf_1.5x` 2/2, `pp_tok_linf_2.0x` 2/2, `pp_tok_l2_1.5x` 2/2, `pp_tok_l2_2.0x` 2/2, `pp_static_0.03` 0/2, `pp_static_0.05` 1/2.

Error against the FP32 reference, per prompt (max over the 16 generated tokens):

| Case | Prompt | CPU BF16 L-inf | Neuron L-inf | Neuron / CPU BF16 | CPU BF16 L2 | Neuron L2 | Neuron / CPU BF16 |
|---|---|---|---|---|---|---|---|
| `[1-2048]` | 1 of 2 | 0.0808 | 0.0346 | 0.43x | 0.0913 | 0.0437 | 0.48x |
| `[1-2048]` | 2 of 2 | 0.1253 | 0.0837 | 0.67x | 0.1235 | 0.0863 | 0.70x |
| `[4-2048]` | 1 of 2 | 0.0808 | 0.0346 | 0.43x | 0.0913 | 0.0437 | 0.48x |
| `[4-2048]` | 2 of 2 | 0.1253 | 0.0837 | 0.67x | 0.1235 | 0.0863 | 0.70x |

Every maximum is at generated token 0.

#### Serve smoke

`/v1/completions`, `temperature` 0, `max_tokens` 32; first 60 characters of each output as a Python string literal.

| Prompt | Output |
|---|---|
| `The capital of France is` | `' Paris., Paris is known for its rich history, beautiful arch'` |
| `1, 2, 3, 4, 5,` | `' 6, 7, 8, 9, 10, 11, 12\n\nHere is a Python code that reads a '` |
| `def fibonacci(n):` | `'\n    if n <= 1:\n        return n\n    else:\n        return fi'` |

### Exceptions

- **All four prompt-runs are outside the static per-prompt bounds.** Both
  integration cases passed on the aggregate σ-ratio (see Logit validation).
- **Module tests not run on 0.29.**
- **The serve smoke ran at revision `main`**, without the `--revision` flag of
  the Usage command.
- Prefix caching (the segmented-prefill path), TP sizes other than 2 and 8,
  context beyond 2048 tokens, speculative decoding and Trn3 are not tested.
