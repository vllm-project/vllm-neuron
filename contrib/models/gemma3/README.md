# Contrib Model: Gemma3 text (rnj-1-instruct)

[rnj-1-instruct](https://huggingface.co/EssentialAI/rnj-1-instruct) is an
8B-parameter dense decoder with the Gemma 3 text architecture
(`Gemma3ForCausalLM`, `model_type` `gemma3_text`): 32 layers with full
attention in every layer, hidden size 4096, grouped-query attention with
32 query heads and 8 KV heads of size 128, a per-head QK-norm, a GELU-tanh
MLP of size 16384, YaRN RoPE, a final logit soft cap of 30 and a 128256-token
vocabulary with tied embeddings. The checkpoint stores FP32 weights.

This folder serves it with vLLM on AWS Trainium2 through the `vllm-neuron`
plugin, in BF16 with tensor parallelism. See [src/README.md](src/README.md)
for the architecture, the differences from the plugin's Qwen3 model and the
per-feature status.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix below states
> what the tests cover, and the validation report records their results.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level
`README.md`), then install this folder:

```bash
pip install -e contrib/models/gemma3
```

The install makes `src/` importable as the `gemma3_neuron` package and adds a
`vllm.general_plugins` entry point (`src/plugin.py`) that registers
`Gemma3ForCausalLM` in every vLLM process (the API server, the engine core and
each tensor-parallel worker), so `vllm serve` and `LLM(...)` find it with no
other setup. vLLM ships its own `Gemma3ForCausalLM`; this registration
replaces it with the Neuron implementation while the package is installed.
`pip uninstall vllm-neuron-contrib-gemma3` removes it.

Two settings are specific to this model:

- **Serve in BF16.** Pass `--dtype bfloat16`. The checkpoint stores FP32
  weights; the implementation supports only the BF16 served dtype and casts
  each weight when it loads it.
- **Use a TP degree that divides the 32 query heads and the 128256-token
  vocabulary.** The tests use TP 8 (4 query heads and one KV head on each
  rank).

### Serve with the OpenAI-compatible server

```bash
vllm serve EssentialAI/rnj-1-instruct \
    --revision 5ef96d7a82b2bff8009192979432c9bc64e870cd \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 9216 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    --block-size 32 \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{"neuron_config": {"on_device_sampling_config": {"all_greedy": true}, "num_batched_tokens_buckets": [8192], "kv_segment_size_buckets": [8192], "num_seqs_buckets": [4]}}'
```

`--host 127.0.0.1` keeps the server local to the host; it listens on port
8000. This is the configuration of the batch-4 case of the integration test:
a 9216-token model length (an 8192-token prompt plus 1024 generated tokens),
prefill steps of 8192 tokens, and up to 4 sequences in a decode step.

- **Prefill.** `--max-num-batched-tokens 8192` is below the model length, so
  vLLM runs chunked prefill, and `kv_segment_size_buckets` turns on the
  plugin's segmented prefill with an 8192-token KV segment. A prompt of up to
  8192 tokens prefills in one step. A longer prompt prefills in two steps,
  and the second step reads the first 8192 tokens back from the KV cache as
  one segment; the integration test does not run that case.
- **Buckets.** The three bucket lists compile one prefill graph and one decode
  graph instead of a ladder of each.
- **Sampling.** `all_greedy` makes every request greedy; remove it (and
  `--no-async-scheduling`) to serve sampled requests, a path that was not run
  for this model. The checkpoint's `generation_config.json` sets
  `temperature` 0.2, and vLLM uses it as the default for requests that do not
  set a temperature (it logs that at startup); pass
  `--generation-config vllm` to keep vLLM's own defaults.

Then send a request:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "EssentialAI/rnj-1-instruct",
        "prompt": "The color of the sky is",
        "max_tokens": 32,
        "temperature": 0
    }'
```

### Offline inference with the Python API

```python
from vllm import LLM, SamplingParams

# With `pip install -e contrib/models/gemma3` the architecture is registered
# in every vLLM process; no registration call is needed here.
llm = LLM(
    model="EssentialAI/rnj-1-instruct",
    revision="5ef96d7a82b2bff8009192979432c9bc64e870cd",
    tensor_parallel_size=8,
    dtype="bfloat16",
    max_model_len=9216,
    max_num_batched_tokens=8192,
    max_num_seqs=4,
    block_size=32,
    enable_prefix_caching=False,
    async_scheduling=False,
    additional_config={
        "neuron_config": {
            "on_device_sampling_config": {"all_greedy": True},
            "num_batched_tokens_buckets": [8192],
            "kv_segment_size_buckets": [8192],
            "num_seqs_buckets": [4],
        }
    },
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
outputs = llm.generate(["The color of the sky is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

Without the install, register the factory from your script before you build
`LLM(...)` (option 1 in [contrib/README.md](../../README.md)), with `src/`
copied into your application:
`ModelRegistry.register_model("Gemma3ForCausalLM", Gemma3ForCausalLM)`. The
engine core and the TP workers inherit that registration only under the
default `fork` start method, so leave `VLLM_WORKER_MULTIPROC_METHOD` unset.

To iterate without Neuron hardware, set `VLLM_NEURON_CPU_MODE=1` and pass
`enforce_eager=True`; `test/unit/test_gemma3_tiny_e2e.py` shows a complete
CPU-mode configuration.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0, vLLM 0.29.0) | 2.32 and earlier |
| ---------------------- | -------------------------------------------- | ---------------- |
| Trn2 (`trn2.48xlarge`) | TP 8: see the [validation report](#validation-report) | Not tested |
| Trn3                   | Not tested | Not tested |

| Feature | Status |
|---|---|
| FP32 checkpoint, BF16 serving | Supported (other served dtypes are rejected by the factory) |
| Tensor parallelism | TP 8 (integration test). The one-KV-head-per-rank decode branch of TP 8 and TP 16 is CPU-tested with a tiny model at TP 2. TP 1, 2, 4 and 16 are legal; no test in this folder runs them on device |
| Context length | 9216-token model length with prompts of 8000 to 8192 tokens; the integration test validates the logits of the first 16 generated tokens. Generation further into the 9216-token context is not validated against a reference, and longer contexts (the checkpoint has 32768 positions, YaRN) are not validated |
| Batch size | 1 and 4 (integration test) |
| Prefill | Segmented prefill, one 8192-token step per prompt with no prior KV (integration test). A prompt longer than 8192 tokens (a second step that reads prior KV) and full prefill without segments are CPU-tested only |
| Prefix caching | Not validated (every test disables prefix caching) |
| Sliding-window Gemma 3 checkpoints | Not supported (the factory rejects `sliding_attention` layers) |
| Speculative decoding, quantized checkpoints, data parallelism | Not tested |

## Example Checkpoints

- <https://huggingface.co/EssentialAI/rnj-1-instruct> — the tests pin revision
  `5ef96d7a82b2bff8009192979432c9bc64e870cd`.

Other `Gemma3ForCausalLM` text checkpoints load only when every layer is a
full-attention layer and the checkpoint has no attention logit soft cap; the
Google Gemma 3 text checkpoints use sliding-window layers and are rejected at
construction.

## Testing

### Unit tests (CPU mode)

No Neuron hardware and no checkpoint download are needed. The tests read the
pinned `config.json` from `test/gemma3_test_utils.py` and build a tiny
random-weight checkpoint with the rnj-1 layout in a temporary directory:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/gemma3/test/unit/ -v -s --timeout=1800
```

| File | Covers |
|---|---|
| `test_gemma3_config.py` | Plugin entry point and vLLM model protocol, config parse (transformers 5 `rope_parameters`, raw `config.json`, plain dict), factory guards, YaRN RoPE against HF (also when built on the meta device), weight-map coverage against the HF model and a saved checkpoint, TP 8 / TP 16 head split, weight-loader shards and the tied LM head |
| `test_gemma3_components.py` | Gemma RMSNorm (`1 + w`) and the kernel gamma, the scaled embedding, the GELU-tanh MLP through `NF.mlp` three-way against HF (one rank and the sum of 8 TP shards), final soft cap, decoder-layer order, the KV helpers in `src/compat.py`, and the segmented-prefill attention branch against the full-prefill branch (a 700-token prompt in two 512-token steps, the second with 512 prior tokens, with the torch reference of the attention kernels) |
| `test_gemma3_tiny_e2e.py` | The full vLLM pipeline at TP 2 (one KV head per rank, the decode branch of TP 8): registration, FP32-to-BF16 load, prefill, 8 decode steps across a KV block boundary, on-device greedy sampling, full-vocabulary raw logits against HF teacher forcing |

The tiny end-to-end test starts vLLM worker processes, whose message queues
reserve a few hundred MiB of `/dev/shm`. It skips itself when less than
512 MiB is free; run it on a host or in a container with a larger `/dev/shm`
(for example `docker run --shm-size=1g`).

### Integration test (Neuron)

Serves the model through vLLM at TP 8 on one `trn2.48xlarge` and validates
its logits against a HuggingFace reference: batch 1 and batch 4, a 9216-token
model length, two prompts of 8000 to 8192 tokens, 16 generated tokens per
prompt, the template's tolerance map and the per-prompt checks
`pp_max_linf_1.5x` and `pp_max_l2_1.5x`. The prompts are synthetic documents
that the test builds from fixed text and truncates with the checkpoint's
tokenizer; each prompt prefills in one 8192-token segmented-prefill step. The
test needs 8 NeuronCores, and the HF reference runs the 8B model on the host
CPU in FP32 and BF16 over both long prompts, so allow a generous timeout:

```bash
GEMMA3_CHECKPOINT_PATH=/path/to/rnj-1-instruct \
pytest contrib/models/gemma3/test/integration/test_model.py -v -s --timeout=14400
```

Without `GEMMA3_CHECKPOINT_PATH` the test downloads the pinned revision.

The integration test does not need the `pip install`: it registers the factory
itself, from the test process, before it builds the engine. vLLM's engine-core
and worker processes inherit that registration through the default `fork`
start method, so do not set `VLLM_WORKER_MULTIPROC_METHOD=spawn` for this test.

All test directories reach the implementation through `test/_local_src.py`,
which loads `src/` by file location under the unique module name
`contrib_gemma3_src`, so no test needs the repository root on `sys.path`.
`test/conftest.py` puts the `test/` directory on `sys.path`, and shuts down
the plugin's `MPExecutor` worker pool at the end of the session; without that
pytest prints its summary and then does not exit.

## Validation report

The validation report for the 9216-token configuration (the integration test
above and the `vllm serve` command in [Usage](#usage)) is pending.
