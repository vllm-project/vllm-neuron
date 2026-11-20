# Phi3 (Text)

Serve a Phi-3-architecture text model (`microsoft/Phi-4-mini-instruct`, whose
`config.json` declares `Phi3ForCausalLM`) with vLLM on AWS Trainium through the
`vllm-neuron` plugin. The implementation is in `src/`, and `src/README.md`
describes the architecture, the partial LongRoPE handling and the per-feature
limits.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`), then install this folder:

```bash
pip install -e contrib/models/phi3
```

The install adds `src/` as the package `phi3_neuron` and a
`vllm.general_plugins` entry point (`src/plugin.py`) that registers
`Phi3ForCausalLM` (Option 3 in [contrib/README.md](../../README.md)). vLLM runs
the entry point in every process it starts, so `vllm serve` and `LLM(...)` need
no other setup.

Install the folder before you start vLLM. Upstream vLLM has its own
`Phi3ForCausalLM`, so without the install vLLM picks that class instead of
reporting an unknown architecture, and the Neuron model runner then fails at
model load because that class has no `from_configs`.

### Serve with the OpenAI-compatible server

This is the command of the recorded serve run (see the validation report):

```bash
vllm serve microsoft/Phi-4-mini-instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 4 \
    --disable-sliding-window \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{"neuron_config": {"on_device_sampling_config": {"all_greedy": true}}}'
```

`--host 127.0.0.1` keeps the server on the loopback interface.

`--disable-sliding-window` is recommended, not required. This implementation
has no sliding-window attention path. The factory accepts a window that is not
smaller than `max_position_embeddings`. Phi-4-mini declares a 262,144-token
window and a 131,072-token position range, so the window has no effect for this
checkpoint.

Then send a request:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "microsoft/Phi-4-mini-instruct",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

With the folder installed, `LLM(...)` finds the architecture without a
`register_model` call:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="microsoft/Phi-4-mini-instruct",
    tensor_parallel_size=8,
    dtype="bfloat16",
    max_model_len=4096,
    max_num_batched_tokens=4096,
    max_num_seqs=4,
    disable_sliding_window=True,
    enable_prefix_caching=False,
    async_scheduling=False,
    additional_config={
        "neuron_config": {"on_device_sampling_config": {"all_greedy": True}}
    },
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The color of the sky is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

The integration test builds `LLM(...)` with these engine arguments
(`max_num_seqs` set to its batch size) plus its logit-capture settings, and it
registers the factory itself instead of through the installed entry point (see
Testing).

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1`. See the top-level `README.md` and `AGENTS.md` for the
full list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn3 | Not tested | Not tested |
| Trn2 | Working (TP 8) | Not tested |

Tested in the Neuron Deep Learning Container
`public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04`
with vLLM 0.29.0.

## Example Checkpoints

- <https://huggingface.co/microsoft/Phi-4-mini-instruct> — the only checkpoint
  with recorded runs. The tests pin revision
  `cfbefacb99257ffa30c83adab238a50856ac3083`.

The implementation accepts a `Phi3ForCausalLM` checkpoint with standard or
LongRoPE scaling, BF16 weights and a sliding window no smaller than
`max_position_embeddings`. No other checkpoint was run.

## Testing

The integration test runs TP 8 on one node, the topology of every single-node
run in the validation report. The unit and module tests keep the port's TP 4.
Both sizes are valid for this implementation (see `src/README.md`,
"Parallelism").

The tests do not need the `pip install -e` step. Each test file loads `src/` by
file location, and the integration test registers the factory with
`ModelRegistry` itself before it builds the engine. The engine core and the
workers inherit that registration because vLLM forks them, so the integration
test skips itself when `VLLM_WORKER_MULTIPROC_METHOD` is not `fork`.

Every test resolves the checkpoint through `test/phi3_utils.py`: it downloads
the pinned revision from the HuggingFace hub, or uses `PHI3_CHECKPOINT_PATH`
when that points at a checkpoint already on disk.

The integration test serves the model through vLLM on Neuron and validates its
logits against a HuggingFace reference. It needs 8 NeuronCores:

```bash
pytest contrib/models/phi3/test/integration/test_model.py \
    -v -s --timeout=7200
```

### Unit tests

`test/unit/test_phi3_cpu.py` runs the registration contract, the config parse,
the fused QKV and gate/up weight transforms, full weight coverage with the tied
LM head, partial LongRoPE against the HuggingFace rotary embedding, and
attention and full-model logits at TP 4. It needs no Neuron hardware:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/phi3/test/unit/ \
    -v -s --timeout=3600
```

### Module tests

`test/module/test_phi3_device_modules.py` compares layer-0 attention (flash
prefill, segmented prefill and decode) and the MLP with a HuggingFace
reference, using the three-way pattern in the
[module test guidelines](../../../docs/design/accuracy/module_test_guidelines.md),
and runs a full-model prefill in the serving pad layout. All four cases need
Neuron hardware and 4 ranks.

```bash
pytest contrib/models/phi3/test/module/ -v -s --timeout=14400
```

The module tests put those 4 ranks on 2 nodes, with a multi-node
launcher that is not part of this contribution. Set
`phi3_utils.DEVICE_NNODES = 1` to run all 4 ranks on one node, or set
`NODE_RANK` and the Neuron multi-node environment yourself on each node.

### Test layout

Beside the three test folders the guide describes, `test/` holds three files:

- `test/_local_src.py` — the loaders. `load()` loads `src/` by file location
  under the module name `contrib_phi3_src`, and `load_test_utils()` loads
  `test/phi3_utils.py` under `contrib_phi3_test_utils`. Each test file loads
  `_local_src.py` itself, also by file location, under
  `contrib_phi3_local_src`. Nothing goes on `sys.path` and no module name can
  collide with another contrib model's, so the tests keep importing this
  model's own code after the folder is copied out of the repository. Build any
  string module path (a patch target, a logger name) from `SRC_PACKAGE` rather
  than writing `contrib_phi3_src` out.
- `test/phi3_utils.py` — the port's own helper, vendored into this folder with
  the model. It holds the checkpoint resolution, the MPExecutor topology, the
  worker-side harnesses and the HuggingFace references. Both kept test files
  use most of it, so it sits at the `test/` root instead of being duplicated
  into `test/unit/` and `test/module/`.
- `test/conftest.py` — a session fixture that shuts down the plugin's
  `MPExecutor` worker pool when the session ends. Without it the pool's workers
  outlive the last test and pytest does not exit.

One pytest invocation can collect only one contrib model's
`test/integration/test_model.py` unless you pass `--import-mode=importlib`,
because every contrib model has a file with that name. Running one model's
tests at a time, as the commands above do, is unaffected.

## Validation report

### vLLM Neuron 0.29.0.1.2.0 (2026-11-19)

#### Environment

| Image | Instance | TP | vLLM | vllm-neuron | torch | transformers | neuronx-cc |
|---|---|---|---|---|---|---|---|
| `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` (Neuron SDK 2.33) | `trn2.48xlarge` (LNC 2) | 8 | 0.29.0 | 0.29.0.1.2.0 (as shipped) | 2.13.0 | 5.17.0 | 2.28.7631.0+8f7b0cf5 |

#### Results

| Test | Configuration | Result |
|---|---|---|
| CPU unit tests, `test/unit/` | `VLLM_NEURON_CPU_MODE=1`, TP 4 | 9 passed |
| `test_model_accuracy[1-4096]` | TP 8, batch 1, max_model_len 4096, 16 generated tokens per prompt | PASSED |
| `test_model_accuracy[4-4096]` | TP 8, batch 4 (each prompt in all 4 rows), max_model_len 4096, 16 generated tokens per prompt | PASSED |
| `vllm serve` smoke | the Usage command, after `pip install -e` | passed (3/3 prompts) |

Not run on 0.29: module tests (they default to 4 ranks on 2 nodes, `phi3_utils.DEVICE_NNODES = 2`; this run had one node).

#### Logit validation

| Parametrization | Prompt | Overall Status | Max divergence (≤ 1 ULP) | Max K5 (≤ 0.03) | Max K50 (≤ 0.05) | Max K1000 (≤ 0.08) | Max All (≤ 0.09) |
|---|---|---|---|---|---|---|---|
| `[1-4096]` | 1 `The color of the sky is` | PASSED | 0.5000 | 0.0103 | 0.0164 | 0.0164 | 0.0164 |
| `[1-4096]` | 2 `A short summary of the history of the printing press:` | PASSED | 0.0000 | 0.0158 | 0.0181 | 0.0182 | 0.0182 |
| `[4-4096]` | 1 `The color of the sky is` | PASSED | 0.5000 | 0.0103 | 0.0164 | 0.0164 | 0.0164 |
| `[4-4096]` | 2 `A short summary of the history of the printing press:` | PASSED | 0.0000 | 0.0158 | 0.0181 | 0.0182 | 0.0182 |

Prompt 1: argmax divergence at generated tokens 11 and 15 in every batch row, accepted as within 1 BF16 ULP (Δ 0.0000 and 0.5000) and teacher-forced. Prompt 2: no divergence.

Informational aggregate checks, both parametrizations: `agg_bc`, `agg_l2_1.2x`, `agg_l2_1.5x` FAIL; `agg_sigma_ratio` 0.8194 ≤ 1.0.

#### Serve smoke

`/v1/completions`, `temperature` 0, `max_tokens` 32, one request per prompt; one 3-prompt request with `max_tokens` 16 returned the same starts.

| Prompt | First 60 characters of the completion (JSON-escaped) |
|---|---|
| `The capital of France is` | `" Paris. What is the capital of Italy? The capital of Italy i"` |
| `1, 2, 3, 4, 5,` | `" 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16"` |
| `def fibonacci(n):` | `" \n    if n <= 1: \n        return n \n    else: \n        retur"` |

#### Changes for 0.29

- `src/compat.py` (new): `build_per_head_block_table` (absent from `vllm_neuron.model.kv_cache` in 0.29.0.1.2.0); `src/model.py` imports it from here.
- `src/model.py`: decode attention casts the KV-cache write slots passed to `NF.attention_decode` (per-head slot mapping and `slot_mapping`) to `torch.uint32`.
- `pyproject.toml`, `src/plugin.py` (new): the `vllm.general_plugins` entry point described under Usage.
- `test/conftest.py` (new): session fixture that shuts down the plugin's `MPExecutor` worker pool.
- `test/integration/test_model.py`: HuggingFace reference loads transformers' built-in `Phi3ForCausalLM` (SDPA) instead of the checkpoint's remote code; golden cache variant `transformers_builtin_phi3`.
- Unchanged: `src/config.py`, `src/factory.py`, the unit and module tests.

### Exceptions

- **Not run on vLLM Neuron 0.29.0.1.2.0:** the module tests; TP 1, 2 and 4 on
  device; `LLM(...)` through the installed entry point (the integration test
  registers the factory itself, and `vllm serve` is the path that ran through
  the entry point); prefix caching and segmented prefill in a served engine;
  context beyond 4,096 tokens, including the LongRoPE long factor in serving
  (it applies from position 4,096, and only the CPU unit tests cover it); an
  8,192-input / 1,024-output workload; speculative decoding; performance
  measurement; Trn3.
- **TP correction.** `src/README.md` first said "the decode kernel needs an
  even number of Q heads on each rank, so TP=8 is not valid". That is wrong.
  The per-rank constraint on the decode path is `q_heads % kv_heads == 0` (the
  GQA head-split check in `vllm_neuron/functional/attention/attention_decode.py`
  and the matching assert in the NKI `attention_block_tkg` kernel). At TP 4 a
  rank holds 2 KV heads, so the check reads as "an even number of Q heads per
  rank"; at TP 8 a rank holds 1 KV head and 3 Q heads, which satisfies it. The
  card keeps the original claim beside the correction.
- **No other checkpoint.** `Phi3ForCausalLM` is the architecture string of
  several Phi-3-family checkpoints, and this folder is named for it, but
  `microsoft/Phi-4-mini-instruct` is the only checkpoint with a recorded run.
