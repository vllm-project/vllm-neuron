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
| Trn2 (trn2.48xlarge) | TP 8: _pending the device runs (see the Validation report)_ | Not tested |
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

### Results on vLLM Neuron 0.29.0.1.2.0

_To be filled in from the device runs of this folder: the unit tests, the
integration test and a `vllm serve` smoke test on `trn2.48xlarge` with the
0.29.0.1.2.0 / Neuron SDK 2.33 container._

| Test | Instance | Result |
|---|---|---|
| `test/unit` (CPU mode) | trn2.48xlarge | _pending_ |
| `test/integration/test_model.py` (TP 8, batch 1 and 4) | trn2.48xlarge | _pending_ |
| `vllm serve` smoke test (TP 8) | trn2.48xlarge | _pending_ |

Locally, in the 0.29.0.1.2.0 container without Neuron devices and without a
local checkpoint, `test/unit` gives 47 passed and 1 skipped (the
tensor-value case of `test_checkpoint.py`).

### Recorded runs of the port

The implementation in `src/` comes from a port that was validated on a
development build of the plugin, before it was adapted to the public
0.29.0.1.2.0 release. These runs passed, all on `trn2.48xlarge` with the
native route, against the pinned checkpoint revision:

| Run | Nodes | TP | Result |
|---|---|---|---|
| CPU loading tests | 1 | 1 and 2 (CPU mode) | PASS |
| Device tests | 2 (4 ranks each) | 8 | PASS |

- The CPU run covers the tests shipped here in `test_factory.py`,
  `test_checkpoint.py` and `test_tiny_hyperclovax_e2e.py`, before the changes
  listed below. The record holds the aggregate verdict.
- The device run compared layer-0 attention (flash prefill at 128 and 1,024
  tokens, segmented prefill in two 512-token chunks, and decode) and the MLP
  against the HF modules three-way; compared the full-model logits of 16
  greedy tokens for the same two prompts as the integration test against the
  HF FP32 and BF16 references with the default tolerance map of
  `multi_prompt_logit_validation` (batch 1, 2,048-token model length, host
  sampling); sent one 1,024-token prompt with 512 output tokens through
  `vllm serve`; and checked that every TP rank compiled prefill and decode
  graphs in which all 38 layers selected the NKI kernel of each main-path
  component. We hold the verdicts, not the logit metrics.

### Differences between this folder and the recorded runs

- **The integration test is single-node.** The recorded logit run split the
  8 TP ranks across two nodes and read the logits through `vllm serve` with
  host sampling. `test/integration/test_model.py` runs all 8 ranks on one node
  with the offline API and on-device greedy sampling, with the same two
  prompts, 16 output tokens, the 2,048-token model length, and the same
  tolerance map: `(1e-5, 0.011)` at top-5, `(1e-5, 0.02)` at top-50,
  `(1e-5, 0.03)` at top-1000 and `(1e-5, 0.05)` over the full vocabulary, with
  the strict `1e-5` argmax bound. It adds a batch-4 case.
- **The logits are read over the full model vocabulary.** The tokenizer has
  110,524 entries and a base `vocab_size` of 110,491, while the model scores
  110,592 ids. The test therefore builds each logit row at the model's
  vocabulary width, disables detokenization of the logprob ids (the 68 ids
  past the tokenizer have no token string), and ignores end-of-sequence so
  that vLLM decodes the same 16 tokens as the HF reference.
- **The O projection travels with the model.** The port changed the plugin's
  `NF.o_proj` torch fallback so that it accepts the FP32 attention output that
  this model feeds it in prefill (the FP32 input keeps the TP reduce-scatter of
  the partial sums in FP32). A contrib folder may not change the plugin, so
  `src/o_proj.py` carries the 0.29.0.1.2.0 `o_proj` with that one fallback
  change, and `src/model.py` calls it. The NKI kernel dispatch is the
  plugin's, unchanged; on device the kernel takes the FP32 input as before.
- **0.29 compatibility.** `hoist_rope_parameters` is not in vLLM Neuron
  0.29.0.1.2.0, so `src/compat.py` carries a copy. Every other plugin API the
  model uses resolves against the installed 0.29.0.1.2.0 plugin.
- **Registration.** The port registered the model in the plugin's own
  registry. This folder registers it through the `pyproject.toml` entry point,
  or, in the tests, with `ModelRegistry.register_model` from `src/` directly.
  The port's registry test became the registration-contract tests in
  `test_factory.py`.
- **Dropped from the port.** The kernel dispatch trace (a test-only wrapper of
  the compile backend and its environment variable) is plugin
  instrumentation, not part of the model, so it is not carried; the
  kernel-dispatch checks that read it are dropped with it. The two-node
  component and serving tests are not shipped: they depend on a two-node
  launcher and shared test infrastructure outside this folder. Their results
  are the device row above, but they cannot be re-run from this folder.
- **Checkpoint resolution.** The port's tests read a fixed shared path. Here
  the tests read `HYPERCLOVAX_CHECKPOINT_PATH`. The unit tests never download
  the 59 GB checkpoint, and the integration test downloads the pinned revision
  without it.

### Untested configurations

Prefix caching and segmented prefill on this folder (the port's component
test covered segmented prefill for one layer), TP degrees other than 8 on
device, context beyond 2,048 tokens, quantization, speculative decoding,
other HyperCLOVAX checkpoints, Trn3, and any plugin version other than the
development build above and 0.29.0.1.2.0.
