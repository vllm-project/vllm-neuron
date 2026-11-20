# EXAONE 4.0 (Text)

This folder serves the EXAONE 4.0 text models with vLLM on AWS Trainium
through the `vllm-neuron` plugin. The implementation under `src/` is a dense
decoder with hybrid attention: three sliding-window layers with RoPE followed
by one full layer with no position embedding, repeated through the stack. See
`src/README.md` for the architecture and the feature-status card.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md` for installation), then install this folder:

```bash
pip install -e contrib/models/exaone4
```

The install adds a `vllm.general_plugins` entry point (`src/plugin.py`) that
registers `Exaone4ForCausalLM` in every vLLM process, so `vllm serve` and
`LLM(...)` need no other setup. vLLM ships its own `Exaone4ForCausalLM`; while
this folder is installed, the entry point replaces it with the class in
`src/`. `pip uninstall vllm-neuron-contrib-exaone4` removes the registration.

The 32B checkpoint is about 64 GB in BF16; the recorded runs used TP8 on
`trn2.48xlarge`.

### Serve with the OpenAI-compatible server

These are the flags of the recorded `vllm serve` run (see the validation
report):

```bash
vllm serve LGAI-EXAONE/EXAONE-4.0-32B \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching
```

Then send a request:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "LGAI-EXAONE/EXAONE-4.0-32B",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

### Offline inference with the Python API

```python
from vllm import LLM, SamplingParams

# The installed entry point registers Exaone4ForCausalLM; no
# ModelRegistry call is needed.
llm = LLM(
    model="LGAI-EXAONE/EXAONE-4.0-32B",
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

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1` in the environment; the tiny end-to-end unit test
does exactly that. See the top-level `README.md` and `AGENTS.md` for the full
list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| --- | --- | --- |
| Trn3 | Not tested | Not tested |
| Trn2 | Working (TP 8) | Not tested |

Tested in the Neuron Deep Learning Container
`public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04`;
the validation report lists the versions.

## Example Checkpoints

- <https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B/tree/a1d54d1c148c30881ed27e035b650da489b51b92>
  (revision `a1d54d1c148c30881ed27e035b650da489b51b92`; the 0.29 run loaded
  `main`, which has pointed at this commit since 2025-08-04)

## Testing

The integration test serves the model through vLLM on Neuron and validates its
per-token logits against a HuggingFace reference:

```bash
pytest contrib/models/exaone4/test/integration/test_model.py \
    -v -s --timeout=7200
```

Set `EXAONE4_CHECKPOINT_PATH` to a local checkpoint directory to avoid a Hub
download. The test compiles one graph for a 32B model, so allow a long
timeout.

`test/_local_src.py` loads `src/` by file location, under the module name
`contrib_exaone4_src`, and each test module loads that helper by file location
too. Nothing goes on `sys.path`, so the tests do not need the repository root
and keep working after this folder is copied into an application — they only
need `vllm-neuron` itself installed, and several contrib models can be
collected in one pytest run without resolving each other's code.

### Optional: unit tests

```bash
VLLM_NEURON_CPU_MODE=1 EXAONE4_CHECKPOINT_PATH=/path/to/EXAONE-4.0-32B \
    pytest contrib/models/exaone4/test/unit/ -v
```

Two suites live here. `test_exaone4_loading.py` reads real checkpoint tensors
and skips when `EXAONE4_CHECKPOINT_PATH` is not set.
`test_tiny_exaone4_e2e.py` needs no checkpoint and no hardware: it writes a
random-weight tiny EXAONE 4.0 model and runs the full vLLM pipeline on CPU at
TP2 and TP1.

### Optional: module tests

```bash
VLLM_NEURON_CPU_MODE=1 EXAONE4_CHECKPOINT_PATH=/path/to/EXAONE-4.0-32B \
    pytest contrib/models/exaone4/test/module/ -v -s --timeout=600
```

`test_exaone4_rope.py` is the one module test: it validates a single component
(`Exaone4RotaryEmbedding`) in isolation against the HuggingFace
`Exaone4RotaryEmbedding`, using the Llama3 scaling parameters of the real
config. It skips when `EXAONE4_CHECKPOINT_PATH` is not set.

The remaining component-level checks stay in `test/unit/` because they validate
weight-loading transforms rather than a component's forward pass against a
reference: the TP8 fused-QKV and o_proj loaders, and the padded MLP shards
(which the loading suite also compares numerically against the unpadded
checkpoint MLP). See the Exceptions section.

## Validation report

### vLLM Neuron 0.29.0.1.2.0 (2026-11-18)

#### Environment

| Image | Instance | TP | vLLM | vllm-neuron | torch | transformers | neuronx-cc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` (+ `pytest-timeout`) | `trn2.48xlarge` (LNC 2) | 8 | 0.29.0 | 0.29.0.1.2.0 | 2.13.0 | 5.17.0 | 2.28.7631.0+8f7b0cf5 |

#### Results

| Test | Configuration | Result |
| --- | --- | --- |
| CPU unit tests, `test/unit/` | `VLLM_NEURON_CPU_MODE=1` | 3 passed, 6 skipped (checkpoint tests of `test_exaone4_loading.py`; `EXAONE4_CHECKPOINT_PATH` not set) |
| `test_model_accuracy[4-2048]` | TP 8, BF16, batch 4, `max_model_len` 2048, 16 generated tokens per prompt | PASSED (544.90 s) |
| `vllm serve` smoke | Usage flags, after `pip install -e contrib/models/exaone4` | Passed (3/3 prompts) |

Not run on 0.29: module test `test/module/test_exaone4_rope.py`.

#### Logit validation

| Prompt | Overall Status | Max divergence (≤ 1 ULP) | Max K5 (≤ 0.025) | Max K50 (≤ 0.04) | Max K1000 (≤ 0.06) | Max All (≤ 0.065) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 of 2 | PASSED | 0.0000 | 0.0154 | 0.0154 | 0.0154 | 0.0154 |
| 2 of 2 | PASSED | 0.2500 | 0.0120 | 0.0120 | 0.0130 | 0.0130 |

No value over threshold. Accepted argmax divergences: prompt 1 at generated token 11 (Δ 0.0000), prompt 2 at generated token 1 (Δ 0.2500).
Test passed on 2/2 prompts (`multi_prompt_logit_validation`); the aggregate block reports `agg_sigma_ratio` 1.2516 (> 1.0) and FAIL for its other `agg_*` checks.

#### Serve smoke

`/v1/completions`, `max_tokens` 32, `temperature` 0; `\n` marks a newline.

| Prompt | First 60 characters |
| --- | --- |
| `The capital of France is` | ` Paris."\n\nThe answer is: "The capital of France is Paris."` |
| `1, 2, 3, 4, 5,` | ` 6, 7, 8, 9, 10, 11, 12, 13, 14,` |
| `def fibonacci(n):` | `\n    if n <= 0:\n        return 0\n    elif n == 1:\n        re` |

#### Changes for 0.29

- `src/compat.py` (new): vendors `hoist_rope_parameters`, `build_per_head_block_table` and `scatter_new_kv`, copied unchanged from a later `vllm-neuron` (not in 0.29.0.1.2.0).
- `src/config.py`, `src/model.py`: import those helpers from `.compat` (`paged_slot_indices` still comes from the plugin).
- `pyproject.toml`, `src/plugin.py` (new): install the `exaone4_neuron` package and register `exaone4_neuron.factory:Exaone4ForCausalLM` via a `vllm.general_plugins` entry point.
- Nothing else in `src/` or `test/` changed.

### Exceptions

- **Three of the port's test files are dropped.** The disaggregated
  (`test_exaone4_di.py`), GSM8K (`test_native_comparison_accuracy.py`) and
  benchmark (`test_native_comparison_benchmark.py`) suites are not shipped,
  together with their shared support module. Each imports test infrastructure
  that lives outside the model folder — `test/utils/simple_server`,
  `test/utils/network`, `test/utils/process`, `test/utils/bench`,
  `test/utils/metric_checks`, `test/vllm_neuron/utils/di_utils`,
  `test/vllm_neuron/utils/logit_test_utils` and
  `test/vllm_neuron/utils/test_prompts` — and the first also needs `mpirun`
  across two nodes with a NIXL side channel, while the other two need the
  `WORKLOAD_OUTPUT_RW` result-publishing contract. Vendoring that
  infrastructure would have meant copying a server launcher, a proxy, an MPI
  harness and a metric-publishing layer into the model folder, so the suites
  were dropped.
- **No test in this folder reproduces the two-node deployment.** The shipped
  integration test, run on 0.29.0.1.2.0, is single-node and offline.
- **The checkpoint resolver changed.** The port's loading test resolved the
  checkpoint through a shared test helper
  (`test/vllm_neuron/utils/logit_test_utils.get_model_checkpoint`) that does
  not ship here. The test now reads `EXAONE4_CHECKPOINT_PATH` and skips when
  it is unset or does not hold a checkpoint. The case count is unchanged, but
  the suite can now skip instead of running.
- **`PROMPTS_2` is vendored.** The two validation prompts came from
  `test/vllm_neuron/utils/test_prompts.py` and are copied verbatim into the
  integration test, as contrib/README.md sanctions.
- **The registry assertion changed meaning.** The port's test asserted that
  `vllm_neuron.model.registry.get_models()` maps the architecture to the
  factory. A contrib model is deliberately not in `get_models()`, so the
  shipped test registers the local factory with vLLM's `ModelRegistry` and
  asserts that the architecture resolves to it instead of to vLLM's own
  EXAONE 4.0 class. The tiny end-to-end test registers it the same way, which
  the in-tree version did not need to do.
- **The tiny test's conftest was inlined.** The port kept the CPU-mode
  environment overrides (`VLLM_ENABLE_V1_MULTIPROCESSING=0`,
  `VLLM_NEURON_MIN_KV_BUDGET_GIB=0`, `NKI_SIMULATOR=1`,
  `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`) in a `conftest.py` beside the test.
  They now live in an autouse fixture inside the test module, so the loading
  suite in the same directory is unaffected.
- **The tiny test's `forked` marker takes effect when `pytest-forked` is
  installed.** In the plugin's own test tree, `test/vllm_neuron/conftest.py`
  blocks `pytest-forked` unless `--forked` is passed, so the marker was inert
  by default. `contrib/` has no such conftest, so with `pytest-forked`
  installed each tiny end-to-end case runs in a forked child. That is what the
  marker asks for, and the two cases were written to be process-isolated, but
  the module-scoped fixtures are then built inside each fork, so the tiny
  checkpoint is written once per case instead of once per module. The
  0.29.0.1.2.0 run did not have `pytest-forked` installed (pytest warned about
  an unknown `forked` mark), so the marker had no effect there.
- **Only one component test is module-shaped.** `test/module/` holds the RoPE
  case, which compares one component against its HuggingFace counterpart.
  The loader and padded-MLP cases stay in `test/unit/` because they check
  weight-loading transforms against checkpoint slices rather than running a
  component against a reference implementation. There is no module test that
  runs attention or the MLP forward pass against a HuggingFace module in the
  sense of `docs/design/accuracy/module_test_guidelines.md`; the full-model
  evidence (the tiny end-to-end suite and the recorded logit run) is what
  covers those components here.
- **Registration must replace vLLM's own class.** vLLM ships an
  `Exaone4ForCausalLM`, so a process in which this folder's registration has
  not run resolves vLLM's class instead of reporting an unsupported
  architecture. The Neuron model runners then call `model_cls.from_configs`
  (`vllm_neuron/vllm/worker/neuron_model_runner.py`), which vLLM's classes do
  not define, so the run fails with an `AttributeError` rather than producing
  wrong output. Installing the folder registers it in every process; the
  0.29.0.1.2.0 `vllm serve` run used that path. The tests do not need the
  install: they register the class loaded from `src/`, and the engine and
  workers inherit it through `fork`, the default start method. That failure
  mode is read from the code, not observed.
- **The test imports were converted to a local loader.** The port imported the
  modeling code as `contrib.models.exaone4.src...`, which resolves only with
  the repository root on `sys.path`. The tests now load `src/` by file location
  through `test/_local_src.py` under the module name `contrib_exaone4_src`, so
  the folder survives being copied out of the repository. The assertions are
  unchanged; the module name they key on is not.
- **Untested configurations.** Only TP8 has device results (TP1 and TP2 appear
  in the CPU tiny test); no other tensor-parallel degree, no Trn3, no
  quantized weights, no FP8 KV cache (the code has that path, with no
  evidence), no speculative decoding (likewise), no prefix caching (every run
  disabled it), and no context at or beyond the 4,096-token sliding window, so
  the window boundary and the block-table trimming above it are unexercised.
