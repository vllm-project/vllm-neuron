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
lacks, so no conversion step is needed. `--trust-remote-code` is not needed:
vLLM reads the config through the MiniCPM3 support built into transformers.
The checkpoint's remote modeling code (`modeling_minicpm.py`, written for
transformers 4.41) does not import on the transformers 5.17 of vLLM Neuron
0.29.0.1.2.0, and nothing in this folder uses it.

### Serve with the OpenAI-compatible server

These are the flags of the tested configuration: a model length of 9,216
tokens (an 8,192-token prompt plus 1,024 generated tokens), up to 4 sequences,
and chunked prefill in 2,048-token chunks. `--host 127.0.0.1` keeps the server
local to the host; it listens on port 8000.

```bash
vllm serve openbmb/MiniCPM3-4B \
    --host 127.0.0.1 \
    --revision d6b14ddaefdb11c624dd75c3c779549bc90b08cb \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 9216 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{"neuron_config": {"quantization": "bf16", "on_device_sampling_config": {"all_greedy": true}, "num_batched_tokens_buckets": [2048], "kv_segment_size_buckets": [2048]}}'
```

With `--max-num-batched-tokens 2048`, vLLM prefills a longer prompt in
2,048-token chunks. `"kv_segment_size_buckets": [2048]` turns on the
segmented-prefill attention: in every prefill step the queries of the chunk
also attend to the KV cache that the earlier chunks of the same prompt wrote.
With the single prefill bucket of 2,048 tokens, one prefill graph serves every
prompt length up to the model length.

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
    max_model_len=9216,
    max_num_batched_tokens=2048,
    max_num_seqs=4,
    enable_prefix_caching=False,
    async_scheduling=False,
    additional_config={
        "neuron_config": {
            "quantization": "bf16",
            "on_device_sampling_config": {"all_greedy": True},
            "num_batched_tokens_buckets": [2048],
            "kv_segment_size_buckets": [2048],
        }
    },
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The color of the sky is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

`all_greedy` on-device sampling makes every request greedy. The checkpoint's
`generation_config.json` asks for sampling (temperature 0.8, top-p 0.8). To
serve sampled requests, remove the `on_device_sampling_config` entry: the
default on-device sampler then applies the temperature, top-k and top-p of
each request. Sampled requests are not validated.

To iterate without Neuron hardware, run in CPU mode by setting
`VLLM_NEURON_CPU_MODE=1`; see
[`docs/model-dev/cpu-development.md`](../../../docs/model-dev/cpu-development.md).

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn3 | Not tested | Not tested |
| Trn2 | See the validation report (TP 8) | Not tested |

## Example Checkpoints

- <https://huggingface.co/openbmb/MiniCPM3-4B> at revision
  `d6b14ddaefdb11c624dd75c3c779549bc90b08cb`, the checkpoint that the tests
  and the commands above use. The tests pin this revision (set
  `MINICPM3_CHECKPOINT_PATH` to a local snapshot to skip the 8 GB download).

The implementation rejects a checkpoint that is not BF16, has no
`q_lora_rank`, or has an attention or MLP bias.

## Testing

The integration test serves the model through vLLM on Neuron and validates its
logits against a HuggingFace reference (the `MiniCPM3ForCausalLM` built into
transformers, not the checkpoint's remote code). It needs 8 NeuronCores on one
trn2.48xlarge:

```bash
pytest contrib/models/minicpm3/test/integration/test_model.py \
    -v -s --timeout=14400
```

What the integration test runs:

- Two cases, `[1-9216]` and `[4-9216]`: batch 1 and batch 4 (each prompt in 4
  rows) at a model length of 9,216 tokens, TP 8, BF16, the `neuron_native`
  route, on-device greedy sampling, prefix caching off and async scheduling
  off.
- Two deterministic prompts, built in the test file from fixed text (numbered
  prose chapters and a weather log) and cut to 8,160 text tokens with the
  checkpoint's tokenizer. With the BOS token, each prompt has between 8,000
  and 8,192 tokens; the test asserts this on the token ids that the reference
  and vLLM consume.
- The chunked prefill of the serve command above: 2,048-token chunks through
  the segmented-prefill attention, four chunks per prompt.
- 16 greedy tokens per prompt, compared with the FP32 and BF16 HF references
  by `multi_prompt_logit_validation` with the template's tolerance map. The
  test asserts `result.passed` and the per-prompt checks `pp_max_linf_1.5x`
  and `pp_max_l2_1.5x` on every prompt.

The HF reference runs the full model in FP32 and in BF16 on the host CPU for
each long prompt, which takes longer than the vLLM part of the test; hence the
timeout. `get_cached_reference_goldens` caches the reference on disk, so the
second case reuses it.

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
  attention against HF at TP 4, in two flows: a 128-token prefill followed by
  8 decode steps through the paged KV cache, and a 1,536-token prefill in
  three 512-token chunks through the segmented-prefill attention. The layer-0
  MLP against HF at TP 4, and a 2-layer full model loaded through
  `load_weights` with its prefill and decode logits against HF at TP 8. All
  checks are three-way (FP32 HF, BF16 HF, this implementation). The
  single-layer checks stay at TP 4 because on CPU the collectives add the BF16
  partial outputs of the ranks in BF16, and at 8 ranks that rounding alone
  exceeds the three-way bound for one layer.
- `test/unit/test_tiny_e2e.py`: the full vLLM `LLM(...)` pipeline on a tiny
  random checkpoint at TP 1 and TP 2 (5 heads per rank, as at TP 8 on the real
  model); the first generated token must match the FP32 HF reference. The
  TP 2 case needs about 640 MiB of `/dev/shm` for vLLM's worker queues and
  skips on a smaller `/dev/shm` (Docker's default is 64 MiB; run the
  container with `--shm-size=1g` or `--ipc=host`).

On CPU, `NF.segmented_attention` runs its torch fallback, so the chunked
prefill check of `test_components.py` covers how the model drives the
segmented-prefill attention (layouts, prior tokens, KV cache writes) and not
the NKI kernel; the integration test runs the kernel.

Two tests need the real weights and skip unless `MINICPM3_CHECKPOINT_PATH` is
set: the 746-key coverage check in `test_loading.py`, and the full-model logits
on the real 62-layer checkpoint in `test_components.py` (8 CPU ranks and an
FP32 HF reference of the full model, so several tens of GB of host memory).

All test directories reach the implementation through `test/_local_src.py`,
which loads `src/` by file location under the unique module name
`contrib_minicpm3_src`. `test/minicpm3_test_utils.py` holds the shared helpers
(the pinned checkpoint, the HF reference loader, the `MPExecutor` harnesses).
`test/conftest.py` puts the `test/` directory on `sys.path` so that the test
directories find those two modules, and shuts down the `MPExecutor` worker
pool when the session ends; without that pytest prints its summary and then
does not exit.

### Coverage

- **Validated configuration:** the serve command and the integration test
  above: TP 8 on one trn2.48xlarge, BF16, a model length of 9,216 tokens,
  chunked prefill in 2,048-token chunks through the segmented-prefill
  attention, and batch 1 and batch 4.
- **Logits** are compared with the HF reference for the first 16 generated
  tokens after each prompt of 8,000 to 8,192 tokens. Longer generations, up to
  the 1,024 tokens that the model length leaves after an 8,192-token prompt,
  are not compared with a reference.
- **Not validated:** one-shot prefill through `NF.flash_attention` (used when
  `kv_segment_size_buckets` is not set); prefix caching (it uses the
  segmented-prefill attention, but no test produces a cache hit); contexts
  longer than 9,216 tokens (the checkpoint allows 32,768 positions); batch
  sizes other than 1 and 4; TP sizes other than 8 on Neuron; sampled
  (non-greedy) requests; async scheduling; speculative decoding; Trn3.
- **The weight loader fetches `pytorch_model.bin` by the Hub id** (revision
  `main`); `--revision` pins only the config and the tokenizer. The
  integration test loads a local snapshot of the pinned revision.

## Validation report

The validation report for the 9,216-token configuration above (vLLM Neuron
0.29.0.1.2.0, Trn2, TP 8) is pending. It will list the environment and the
results of the unit tests, the integration test and the serve command.
