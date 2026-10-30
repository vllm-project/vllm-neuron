# Contrib Model: MuseGlimmer (meta-models/Muse-Glimmer-30B)

Serves [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B)
(`MuseGlimmerForConditionalGeneration`) with vLLM on AWS Trainium through the
`vllm-neuron` plugin. MuseGlimmer is a multimodal model: a dense 52-layer, 30B-parameter
decoder with sliding-window and global attention layers, plus a 1.8B-parameter perception
encoder (ViT) for images and video frames. Input is text, images and video. Output is text.
This folder runs the model in BF16 on one trn2.3xlarge (one Trainium2 device, TP=4).

The architecture, kernel mapping, knobs and feature status are in
[src/README.md](src/README.md).

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

## Usage

### Install / register

vLLM already ships its own GPU implementation under the same architecture name
(`MuseGlimmerForConditionalGeneration`). Its multimodal processor emits different
keyword arguments than the Neuron encoder takes, so the Neuron class has to replace it in
**every** vLLM process: the API server and the engine core build the multimodal processor
and the encoder budget from the registered class, and the Neuron workers build the model.
The engine core and the workers are spawned processes, so a `ModelRegistry.register_model`
call in your own script (Option 1 in [contrib/README.md](../../README.md)) reaches only
the calling process. Under `vllm serve` there is no script at all. Without the override,
`vllm serve` fails at startup with `TypeError: Unexpected keyword argument max_image_tokens`.

This folder is therefore a small Python package that registers the model through a
`vllm.general_plugins` entry point, which vLLM runs in every process at startup:

```bash
pip install -e contrib/models/muse_glimmer     # imports as `muse_glimmer_neuron`
```

After that, both `vllm serve` and the offline `LLM(...)` API pick up the Neuron
implementation with no further code. If you copy `src/` into your own package instead,
call `muse_glimmer_neuron.register()` from your own `vllm.general_plugins` entry point.

This is the "register from your own plugin" route that the contrib guidelines allow. It
needs no change to `vllm_neuron/model/registry.py`. It does need one small plugin fix that
is in the same PR (see "Plugin requirement" below).

### Serve with the OpenAI-compatible server

`scripts/serve.sh` wraps the BF16 configuration used for the results below:

```bash
MODEL=/path/to/Muse-Glimmer-30B bash contrib/models/muse_glimmer/scripts/serve.sh
```

The core of it:

```bash
export NEURON_SKIP_EFA_AFFINITY=1                    # trn2.3xlarge has no EFA
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.15   # leave device memory for the graphs
vllm serve meta-models/Muse-Glimmer-30B --tensor-parallel-size 4 \
  --max-model-len 8192 --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching --max-logprobs 0 \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' --mm-processor-kwargs '{"max_image_tokens": 1024}' \
  --additional-config '{
    "neuron_config": {"num_batched_tokens_buckets": [512, 1024, 2048, 4096, 8192],
                      "num_seqs_buckets": [1, 2, 4, 8, 16],
                      "decode_context_length_buckets": [4096],
                      "on_device_sampling_config": {"all_greedy": true}},
    "vision_neuron_config": {"vision_attention_block_size": 4096, "num_vision_tokens_buckets": [4096],
                             "encoder_cache_num_blocks": 16, "tp_size": 4}}' \
  --port 8000
```

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "meta-models/Muse-Glimmer-30B", "max_tokens": 128,
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"}},
    {"type": "text", "text": "Describe this image."}]}]}'
```

Add 64 GB of swap before the first start. The compiler can use a lot of host memory for
the TP=4 graphs.

### Offline inference with the Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-models/Muse-Glimmer-30B",
    tensor_parallel_size=4,
    max_model_len=8192,
    max_num_seqs=4,
    additional_config={"neuron_config": {"num_seqs_buckets": [1, 2, 4]}},
    # Text only: skip the vision tower.
    # additional_config={"mm_language_model_only": True, ...},
)
outputs = llm.chat([{"role": "user", "content": "The capital of France is"}],
                   SamplingParams(temperature=0.0, max_tokens=32))
print(outputs[0].outputs[0].text)
```

For images, pass `vision_neuron_config` as in the server example and send
`{"type": "image_pil", "image_pil": img}` content. For video, raise the `video` limit in
`limit_mm_per_prompt` and send frames with their metadata, as with any vLLM video model.

### Configuration notes

- **KV cap.** With the default `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` (0.30) the compiled
  graphs do not fit next to the BF16 weights. 0.15 admits 16 sequences at
  `max_model_len` 8192.
- **Validate a new cap or bucket set with a sweep that runs every prefill, decode and
  vision bucket.** Device memory for each graph's DMA rings is allocated the first time the
  graph runs, so a short test can pass while a full sweep fails.
- **Images.** `max_image_tokens` 1024 (4096 patches) matches the 4096-patch vision bucket. An
  image must fit in one `vision_attention_block_size` block. Many concurrent small images
  need about 4 encoder-cache blocks per concurrent image.
- **Sliding window.** The model pads each trimmed sliding-window decode table to a power of
  two in its own code (see [src/README.md](src/README.md#sliding-window-decode-table)); no
  plugin setting is needed.

## Plugin requirement

The PR that adds this folder also changes the plugin's multimodal capability lookups
(`vllm_neuron/utils/mm_utils.py`). Before the change they read the model class only from
`get_models()`, which by design does not list contrib models. The encoder-cache sizing then
returned 0 blocks per request for MuseGlimmer, and every image request failed with
`Encoder block truncation: request has more cached blocks than max_encoder_blocks_per_request=0`.
They now also look in vLLM's `ModelRegistry` (`vllm_neuron.model.registry.get_model_cls`).
Text-only requests work without the change.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vllm-neuron 0.29) | 2.32 and earlier |
| ---------------------- | ----------------------- | ---------------- |
| Trn3                   | Not tested              | Not tested       |
| Trn2 (trn2.3xlarge, TP=4, LNC=2) | Working       | Not tested (this folder) |

Tested in CPU mode (`VLLM_NEURON_CPU_MODE=1`) at TP=1 and TP=4 on this branch. The trn2
results below were measured with the same model code on vllm-neuron 0.29.0.1.2.0 (SDK 2.33).

## Results (one Trainium2 device, TP=4)

**Accuracy, BF16, against Hugging Face transformers BF16:**
- 66-prompt equivalence set on trn2.3xlarge (text, image, video; 16 greedy tokens): 66/66
  prompts with all 16 tokens equal, prefill logit cosine >= 0.99997. Measured on vllm-neuron
  0.24 / SDK 2.32;
  on 0.29 / SDK 2.33 the BF16 long-form text checks are unchanged (32/32 tokens on 3 prompts,
  including a 5,246-token prompt).
- Task-level, served with `vllm serve` at TP=4 on one device (greedy, up to 3072 answer
  tokens): GSM8K 95.0 %, MMLU-Pro 75.4 %, ChartQA 80.0 %, MMMU validation 68.5 %.

**Performance, BF16, trn2.3xlarge** (`vllm bench serve`, random dataset, `--ignore-eos`,
greedy; cap 0.15, 16 sequences, vision bucket 4096):

| Workload (in / out tokens) | Peak output tok/s |
|---|---|
| 128 / 128 | 370 |
| 128 / 512 | 455 |
| 1024 / 256 | 379 |
| 2048 / 128 | 224 |
| 2048 / 512 | 379 |
| 896x672 image + 256 / 128 | 184 |

TPOT at concurrency 1: 27.2 ms.

FP8 weights, an FP8 KV cache, speculative decoding with the
[assistant drafter](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant) and
sliding-window KV admission raise throughput further. They need plugin changes beyond
model support, so they are not in this folder; the branch
[`jimburtoft/vllm-neuron@contrib/muse-glimmer-0.29`](https://github.com/jimburtoft/vllm-neuron/tree/contrib/muse-glimmer-0.29)
has them, with its own README.

## Example Checkpoints

- <https://huggingface.co/meta-models/Muse-Glimmer-30B> (gated)

## Testing

Install the contrib package first:

```bash
pip install -e contrib/models/muse_glimmer
```

### Integration test (accuracy, served through vLLM on Neuron)

The full model is too large for an FP32 Hugging Face reference on the host, so the test
builds a reduced checkpoint with the real weights of the first 4 text and 4 vision layers
(`scripts/make_mini.py`) and compares it with HF transformers on the same weights:
three-way logit validation on text prompts (one longer than the 2048-token window), and
image prompts (prompt token ids equal to HF's; greedy tokens equal to HF FP32's up to a
near-tie).

```bash
MUSE_GLIMMER_CKPT=/path/to/Muse-Glimmer-30B \
  pytest contrib/models/muse_glimmer/test/integration/test_model.py -v -s --timeout=7200
```

`MUSE_GLIMMER_MINI_CKPT` caches the reduced checkpoint (default `/tmp/muse_glimmer_mini4`).
The same test runs in CPU mode as a functional smoke run
(`VLLM_NEURON_CPU_MODE=1 MUSE_GLIMMER_SEQ_LEN=3072`).

### Unit tests (CPU)

```bash
VLLM_NEURON_CPU_MODE=1 MUSE_GLIMMER_PROCESSOR=meta-models/Muse-Glimmer-30B \
  pytest contrib/models/muse_glimmer/test/unit/ -v
```

- `test_components.py`: registration, config guards, the sliding-window table padding
  (including attention over the padded table against the aligned window), the per-head
  decode table.
- `test_tiny_e2e.py`: a tiny random checkpoint (4 text + 4 vision layers at the released head
  geometry) through vLLM in CPU mode at TP=1 and TP=4, in FP32. Text and image greedy tokens
  must equal the HF FP32 greedy tokens exactly. `MUSE_GLIMMER_PROCESSOR` (the tokenizer and
  processor files) is needed for the image cases; without it they are skipped. Each case takes
  a few minutes and about 10 GB of host memory.
