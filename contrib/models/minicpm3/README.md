# MiniCPM3 (Text)

Serve MiniCPM3 (`openbmb/MiniCPM3-4B`) with vLLM on AWS Trainium through the
`vllm-neuron` plugin. The implementation is in `src/`, and `src/README.md`
describes the architecture (multi-head latent attention mapped onto the fused
multi-head kernels) and the per-feature status.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`), then install this folder:

```bash
pip install -e contrib/models/minicpm3
```

The install makes `src/` importable as the `minicpm3_neuron` package and adds a
`vllm.general_plugins` entry point (`src/plugin.py`) that registers
`MiniCPM3ForCausalLM` in every vLLM process. vLLM has its own
`MiniCPM3ForCausalLM`; the registration replaces it with this implementation.
`vllm serve` and `LLM(...)` then find the architecture with no other setup.

The published checkpoint ships only `pytorch_model.bin` (no `.safetensors`).
`src/checkpoints.py` adds the `.bin` loading that the 0.29 plugin's loader
lacks, so no conversion step is needed. Do not pass `--trust-remote-code`:
the checkpoint's remote code targets transformers 4.41, and vLLM reads the
config through the MiniCPM3 support built into transformers.

### Serve with the OpenAI-compatible server

These are the flags of the tested configuration. `--host 127.0.0.1` keeps the
server local to the host; it listens on port 8000.

```bash
vllm serve openbmb/MiniCPM3-4B \
    --host 127.0.0.1 \
    --revision d6b14ddaefdb11c624dd75c3c779549bc90b08cb \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{"neuron_config": {"quantization": "bf16", "on_device_sampling_config": {"all_greedy": true}}}'
```

Then send a request:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "openbmb/MiniCPM3-4B",
        "prompt": "The color of the sky is",
        "max_tokens": 32
    }'
```

The weight loader fetches `pytorch_model.bin` by the Hub id; `--revision`
pins the config and the tokenizer. To pin all files to one revision, download
a snapshot first (`hf download openbmb/MiniCPM3-4B --revision
d6b14ddaefdb11c624dd75c3c779549bc90b08cb --local-dir <dir>`) and serve
`<dir>`.

### Offline inference with the Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="openbmb/MiniCPM3-4B",
    revision="d6b14ddaefdb11c624dd75c3c779549bc90b08cb",
    tensor_parallel_size=8,
    dtype="bfloat16",
    max_model_len=2048,
    max_num_batched_tokens=2048,
    max_num_seqs=4,
    enable_prefix_caching=False,
    async_scheduling=False,
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

`all_greedy` on-device sampling makes every request greedy. The checkpoint's
`generation_config.json` asks for sampling (temperature 0.8, top-p 0.8); drop
`on_device_sampling_config` to serve sampled requests.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1`. See the top-level `README.md` and `AGENTS.md` for the
full list of `VLLM_NEURON_*` environment variables.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn3 | Not tested | Not tested |
| Trn2 | See the validation report (TP 8) | Not tested |

## Example Checkpoints

- <https://huggingface.co/openbmb/MiniCPM3-4B> at revision
  `d6b14ddaefdb11c624dd75c3c779549bc90b08cb` — the only checkpoint with
  recorded runs. The tests pin this revision (set `MINICPM3_CHECKPOINT_PATH`
  to a local snapshot to skip the 8 GB download).

The implementation rejects a checkpoint that is not BF16, has no
`q_lora_rank`, or has an attention or MLP bias.

## Testing

The integration test serves the model through vLLM on Neuron and validates its
logits against a HuggingFace reference (the `MiniCPM3ForCausalLM` built into
transformers, not the checkpoint's remote code). It needs 8 NeuronCores on one
trn2.48xlarge:

```bash
pytest contrib/models/minicpm3/test/integration/test_model.py \
    -v -s --timeout=10800
```

The integration test does not need the `pip install`: it registers the factory
itself, from the test process, before it builds the engine. vLLM's engine-core
and worker processes inherit that registration through the default `fork`
start method, so do not set `VLLM_WORKER_MULTIPROC_METHOD=spawn` for this test
unless the folder is installed.

### Optional: unit tests

The unit tests need no Neuron hardware and download nothing: they build small
random-weight checkpoints with the MiniCPM3-4B geometry, saved as
`pytorch_model.bin` like the published checkpoint.

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/minicpm3/test/unit/ \
    -v -s --timeout=3600
```

- `test/unit/test_loading.py`: registration and the vLLM model protocol,
  factory checks, config and padded head layout, RoPE buffers, checkpoint key
  coverage, the `.bin` loader fallback of `src/checkpoints.py`, and the fused
  MLA weight math in FP32 against the HF attention.
- `test/unit/test_components.py`: CPU ranks under `MPExecutor`. Layer-0
  attention (prefill, then 8 decode steps through the paged KV cache) and MLP
  against HF at TP 4, the port's topology for these checks, and a 2-layer full
  model loaded through `load_weights` with its prefill and decode logits
  against HF at TP 8; all three-way (FP32 HF, BF16 HF, this port). The
  single-layer checks stay at TP 4 because on CPU the collectives add the BF16
  partial outputs of the ranks in BF16, and at 8 ranks that rounding alone
  exceeds the three-way bound for one layer.
- `test/unit/test_tiny_e2e.py`: the full vLLM `LLM(...)` pipeline on a tiny
  random checkpoint at TP 1 and TP 2 (5 heads per rank, as at TP 8 on the real
  model); the first generated token must match the FP32 HF reference. The
  TP 2 case needs about 640 MiB of `/dev/shm` for vLLM's worker queues and
  skips on a smaller `/dev/shm` (Docker's default is 64 MiB; run the container
  with `--shm-size=1g` or `--ipc=host`).

Two tests need the real weights and skip unless `MINICPM3_CHECKPOINT_PATH` is
set: the 746-key coverage check in `test_loading.py`, and the full-model logits
on the real 62-layer checkpoint in `test_components.py` (8 CPU ranks and an
FP32 HF reference of the full model, so several tens of GB of host memory).

All test directories reach the implementation through `test/_local_src.py`,
which loads `src/` by file location under the unique module name
`contrib_minicpm3_src`. `test/minicpm3_test_utils.py` holds the shared helpers
(the pinned checkpoint, the HF reference loader, the `MPExecutor` harnesses).
`test/conftest.py` puts the `test/` directory on `sys.path` so that the test
directories find those two modules, provides the `dist_cleanup` fixture that
the plugin's test tree would otherwise supply, and shuts down the plugin's
`MPExecutor` worker pool when the session ends; without that pytest prints its
summary and then does not exit.

## Validation report

### vLLM Neuron 0.29.0.1.2.0

<!-- Filled in from the device run: environment, results of the unit tests
and of test_model_accuracy[1-2048] / [4-2048], the logit-validation table, and
a serve smoke. -->

_Pending: device results for this folder on vLLM Neuron 0.29.0.1.2.0._

#### Changes from the port for 0.29

- `src/checkpoints.py` (new): the 0.29 `SafetensorsCheckpoint` with the
  `pytorch_model*.bin` fallback of a later release. The port changed
  `vllm_neuron/utils/checkpoints.py` in the plugin; a contrib folder carries
  the loader itself instead. `src/model.py` imports it from there.
- `src/compat.py` (new): vendors `hoist_rope_parameters` (0.29 has no
  `vllm_neuron.model.config_utils`) and `build_per_head_block_table` (not in
  0.29's `vllm_neuron.model.kv_cache`).
- `src/model.py`: the decode write slots (`kv_cache_update_idx`) are passed to
  `NF.attention_decode` as `uint32` in both the kernel and the torch-fallback
  branch.
- `pyproject.toml`, `src/plugin.py` (new): install `src/` as the
  `minicpm3_neuron` package with a `vllm.general_plugins` entry point.
- Tests: the port's tests read the 8 GB checkpoint and ran on 2 nodes at TP 4.
  The unit tests here use random-weight checkpoints with the real geometry and
  run at TP 4 (components) and TP 8 (full model) on one host; the integration
  test is the template's offline logit test at TP 8 on one node.
- Model math, weight layout and test tolerances: otherwise unchanged.

### Earlier runs of the port

The port ran on a pre-0.29 development build of the plugin, with its own
tests (not the tests in this folder). Only verdicts were recorded.

| Run | Topology | Result |
|---|---|---|
| CPU loading and components (real checkpoint) | 1 node, TP 4 CPU ranks | PASS |
| Device functional: kernel gates, TP 4 attention and MLP against HF, full-model logits through `vllm serve`, 1024-in / 512-out serving | 2 trn2.48xlarge nodes, TP 4 | PASS |
| GSM8K (chain of thought, 100 questions), batch 1 and 4 | 1 trn2.48xlarge, TP 8 | PASS (the gate was a completed run; no score threshold) |
| Serving benchmark, 1024 in / 100 out, batch 1 and 4 | 1 trn2.48xlarge, TP 8 | PASS (measurement only) |

### Exceptions

- **No recorded logit validation at TP 8 before this folder.** The port's
  logit test ran at TP 4 over 2 nodes; the TP 8 runs gated only on a completed
  GSM8K run and a completed benchmark. `test/integration/test_model.py` is
  the first logit check at TP 8 (5 heads per rank).
- **Dropped from the port:** its 2-node device tests and launcher, its GSM8K
  and benchmark tests, and its serving test. They depend on shared test
  helpers outside this folder and on a 2-node topology.
- **Untested configurations.** Prefix caching and the segmented-prefill path,
  TP sizes other than 8 on device, context beyond 2,048 tokens, speculative
  decoding, Trn3, and any `vllm-neuron` version other than 0.29.0.1.2.0 for
  the code in this folder.
