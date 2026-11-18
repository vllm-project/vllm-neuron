# Contrib Model: Muse-Glimmer-30B (text, image, video)

[Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) is a multimodal language model that accepts text, images, and video. The text decoder has 52 layers in a repeating pattern of three sliding-window layers (window 2048, RoPE) and one global layer (no positional encoding), with gated attention and 2 KV heads. The vision encoder is a windowed-attention ViT that merges 2x2 patches into one embedding. The model writes its reasoning in a separate `to=self` channel before the answer.

This folder serves it with vLLM on AWS Trainium3 through the `vllm-neuron` plugin, in BF16 on Model Runner V1. The model is plain PyTorch over the paged KV cache (no NKI kernels); see [src/README.md](src/README.md) for the architecture and feature status.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

## Usage

Use a Neuron environment with `vllm-neuron` 0.29.0.1.2.0 (for example the Neuron Deep Learning Container `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04`), then install this model. The install adds a `vllm.general_plugins` entry point that registers Muse-Glimmer in every vLLM process, so `vllm serve` and `LLM(...)` find it with no other setup:

```bash
pip install -e contrib/models/muse_glimmer
```

The validated configuration runs at logical NeuronCore config 2. vLLM Neuron sets 1 when the variable is unset, which leaves too little HBM per core for the 30B weights:

```bash
export NEURON_LOGICAL_NC_CONFIG=2
```

### Serve with the OpenAI-compatible server

```bash
vllm serve meta-models/Muse-Glimmer-30B \
    --served-model-name Muse-Glimmer-30B \
    --tensor-parallel-size 2 \
    --max-model-len 2048 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 1024 \
    --num-gpu-blocks-override 512 \
    --limit-mm-per-prompt '{"image": 1, "video": 1}' \
    --media-io-kwargs '{"video": {"num_frames": -1}}' \
    --no-enable-prefix-caching \
    --reasoning-parser muse_glimmer \
    --default-chat-template-kwargs '{"reasoning_strength": "low"}' \
    --additional-config '{
        "neuron_config": {
            "num_batched_tokens_buckets": [1024],
            "num_seqs_buckets": [1]
        },
        "encoder_neuron_config": {
            "encoder_attention_block_size": 1024,
            "num_encoder_input_buckets": [1024, 1564, 1632, 2080],
            "encoder_cache_num_blocks": 16
        }
    }'
```

The first start compiles the text and vision graphs (about 15 minutes). The settings:

- **`num_encoder_input_buckets`**: one vision graph per fixed shape, in patches. Each input is resized to the shape with the closest aspect ratio:

  | Input | Shape | Patches | Vision tokens |
  |---|---|---|---|
  | Square image | 448x448 | 1024 | 256 |
  | Landscape image (for example 640x480) | 644x476 | 1564 | 391 |
  | Portrait image (for example 480x640) | 476x672 | 1632 | 408 |
  | Video | 8 frames at 480x368, real timestamps | 2080 | 520 |

- **`encoder_cache_num_blocks: 16`**: encoder-cache blocks hold 256 tokens (`encoder_attention_block_size` / 4), and each item pads its last block (a 520-token video takes 3 blocks, a 391-token image 2). The default size counts tokens only, so a video and an image together can fill it; 16 blocks leave room.
- **`--no-enable-prefix-caching`**: in vLLM Neuron 0.29.0.1.2.0, a prefix-cache hit can leave only an image's end marker to prefill; the runner then requires the image in the encoder cache and stops the engine (fixed in later vLLM Neuron releases).
- **`--media-io-kwargs '{"video": {"num_frames": -1}}'`**: vLLM passes all decoded frames and their timestamps; the processor picks 8 frames and writes the real timestamps into the prompt.
- **`--reasoning-parser muse_glimmer`**: puts the `to=self` reasoning in `reasoning_content`, so `content` holds only the final answer.
- **`--default-chat-template-kwargs '{"reasoning_strength": "low"}'`**: the chat template defaults to `high`. A request can set its own strength with a system message such as `Reasoning strength: high`.
- **`--num-gpu-blocks-override 512`**: vLLM makes 4 KV-cache groups (sliding-window and full layers), so one 2048-token sequence needs 4 x 64 blocks of 32 tokens.

Then send a request (the reasoning is in `reasoning_content`):

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Muse-Glimmer-30B",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 128
    }'
```

Images and videos go in as OpenAI `image_url` / `video_url` content parts (URLs or base64 data URLs).

### Offline inference with the Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-models/Muse-Glimmer-30B",
    tensor_parallel_size=2,
    max_model_len=2048,
    max_num_seqs=1,
    max_num_batched_tokens=1024,
    num_gpu_blocks_override=512,
    limit_mm_per_prompt={"image": 1, "video": 1},
    media_io_kwargs={"video": {"num_frames": -1}},
    enable_prefix_caching=False,
    additional_config={
        "neuron_config": {"num_batched_tokens_buckets": [1024], "num_seqs_buckets": [1]},
        "encoder_neuron_config": {
            "encoder_attention_block_size": 1024,
            "num_encoder_input_buckets": [1024, 1564, 1632, 2080],
            "encoder_cache_num_blocks": 16,
        },
    },
)
outputs = llm.chat(
    [[{"role": "user", "content": "What is the capital of France?"}]],
    SamplingParams(temperature=0.0, max_tokens=128),
    chat_template_kwargs={"reasoning_strength": "low"},
)
# Offline output holds both channels; the answer follows "to=user".
print(outputs[0].outputs[0].text.rsplit("to=user", 1)[-1].strip())
```

Without `pip install`, call `muse_glimmer_neuron.register()` before `LLM(...)` (with `src/` on the path as `muse_glimmer_neuron`); this works for TP=1 only, because TP worker processes register through the installed entry point.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vLLM Neuron 0.29.0.1.2.0) | 2.32 and earlier |
| ---------------------- | ------------------------------- | ---------------- |
| Trn3 (`trn3-dev1.48xlarge`, 2 devices at LNC=2, TP=2) | Working | Not tested |
| Trn2 | Not tested | Not tested |

Tested on the Neuron DLC `pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` as shipped (vLLM 0.29.0, vLLM Neuron 0.29.0.1.2.0, transformers 5.17.0 with the `muse_glimmer` model), with only this folder installed.

| Feature | Status |
|---|---|
| Text; single image + text; video + text; image + video + text in one prompt | ✅ |
| Multi-image + text | ❌ |
| BF16 | ✅ |
| MXFP8 | ❌ |
| Tensor parallelism | ✅ TP=2 (TP above 2 replicates the 2 KV heads; not validated) |
| Encoder parallelism, EPD, segmented prefill, speculative decoding | ❌ |
| On-device greedy sampling | ✅ |
| Prefix caching | ❌ on vLLM Neuron 0.29.0.1.2.0 (see Usage) |
| Model Runner V2 | ❌ (runs on MRv1) |

## Accuracy

**POPE** (Polling-based Object Probing Evaluation): Yes/No questions about object presence in COCO images, in three subsets (popular, adversarial, random), with the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) `@fd3884b` request and Yes/No rule, `--max-tokens 256`, reasoning strength `low`, Trn3 BF16, TP=2.

This folder, on vLLM Neuron 0.29.0.1.2.0 (DLC as shipped), on a fixed 800-question subset: the 400 "yes" questions, which are the same in all three subsets, and 400 adversarial "no" questions (seed 0, `scripts/pope_subset.py` in [yahavb/Muse-Glimmer-30B](https://github.com/yahavb/Muse-Glimmer-30B)):

| Metric | Muse-Glimmer-30B |
|---|---|
| F1 | 85.99 |
| Precision / Recall | 94.33 / 79.00 |
| Accuracy | 87.12 |

All 800 answers were Yes or No (none empty or unparsed); the server logged no errors.

Full POPE (5,000 questions) on an earlier revision of this model (every image resized to 448x448; the last prompt token attended to the prefill padding, both fixed here): overall F1 85.35; adversarial / popular / random F1 84.86 / 85.15 / 86.05; precision 97.22, recall 76.07.

Reproduce the full run, with the server from [Usage](#serve-with-the-openai-compatible-server) running:

```bash
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit && git checkout fd3884b && pip install . && cd ..
python VLMEvalKit/run.py --data POPE --model Muse-Glimmer-30B \
    --base-url http://localhost:8000/v1 --max-tokens 256 --judge exact_matching
```

With `--max-num-seqs 1`, the 5,000 questions take about 9 hours.

## Example Checkpoints

- <https://huggingface.co/meta-models/Muse-Glimmer-30B> (gated; set `HF_TOKEN`)

## Testing

Install the package first, so that TP worker processes register the model:

```bash
pip install -e contrib/models/muse_glimmer
```

The integration test serves the checkpoint on Trn3 and checks text, image, and video answers:

```bash
NEURON_LOGICAL_NC_CONFIG=2 pytest contrib/models/muse_glimmer/test/integration/test_model.py \
    -v -s --timeout=3600
```

### Unit tests

CPU-mode tests (no Neuron hardware): the fixed shapes, the encoder sizing hooks, the registration, the processor lookup, and a tiny random-weight model compared with HF transformers at TP=1 and TP=2 (near-greedy: every vLLM token within 0.05 logit of HF's best):

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/muse_glimmer/test/unit/ -v --forked
```
