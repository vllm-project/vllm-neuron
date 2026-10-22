# Voxtral Model Recipe

<!-- meta: description: Model recipe for deploying Voxtral with vLLM on Neuron,
including supported checkpoints, feature support, accuracy results, and
deployment guidance for Voxtral-Mini-3B on Trn2. -->
<!-- meta: keywords: vLLM, Neuron, Voxtral, Voxtral-Mini-3B, audio, speech
recognition, ASR, transcription, multimodal, DFlash, model recipe, Trn2, Trainium -->
<!-- meta: date_updated: 2026-10-21 -->
<!-- Content type: model-card -->

## Introduction

[Voxtral](https://huggingface.co/collections/mistralai/voxtral) is an audio
language model from Mistral AI. It uses a Whisper-style audio encoder, a
2-layer audio adapter, and a Llama-architecture text decoder. It does speech
transcription, speech translation, and audio understanding.

Voxtral is supported for inference serving with
[vLLM Neuron Plugin](https://github.com/vllm-project/vllm-neuron) using the
Neuron SDK on AWS Trainium2 (`trn2`) hardware. Upstream vLLM supplies the audio
processor and the prompt format. The plugin runs the audio encoder, the
adapter, and the decoder on Neuron.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| Voxtral-Mini-3B-2507 | [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | Trn2 | BF16 |

## Features

| Category | Feature | Status |
|---|---|---|
| **Multimodal Inputs** | Text | ✅ |
| | Audio + text (chat) | ✅ |
| | Audio longer than 30 s (multiple 30 s windows) | ✅ |
| | `/v1/audio/transcriptions` endpoint | ✅ |
| **Quantization** | BF16 | ✅ |
| **Parallelism** | Tensor parallelism (TP) | ✅ |
| | Encoder tensor parallelism | ✅ |
| **Performance** | On-device sampling | ✅ |
| | DFlash speculative decoding | ✅ |
| **Compilation** | torch.compile (native backend) | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for Voxtral

## Configuration

Use the Mistral tokenizer. vLLM reads the model config from `params.json` (the
default for Mistral repositories) or from `config.json` (`--config-format hf`).
The two formats give the same result.

The plugin computes the audio embeddings in 30 s windows. Each window gives 375
audio tokens, and each encoder cache block holds one window. Set these values in
`encoder_neuron_config`:

| Field | Value | Note |
|---|---|---|
| `encoder_attention_block_size` | `375` | One cache block holds one 30 s window. |
| `num_encoder_input_buckets` | for example `[1, 4]` | Windows in one encoder call. The largest bucket limits the audio length of one step. |
| `tp_size` | `4` on `trn2.3xlarge` | The encoder has 20 attention heads, so `tp_size` must divide 20. |
| `encoder_cache_num_blocks` | `max(num_encoder_input_buckets) * max_num_seqs + 2` | Set it when `max_num_seqs > 1`. One clip uses one block for each 30 s window. |

For batched serving, also set `neuron_config.decode_context_length_buckets` (for example
`[512, 1024]` with `--max-model-len 2048`). Short transcripts then decode with a smaller attention
context.

```bash
vllm serve mistralai/Voxtral-Mini-3B-2507 \
    --tokenizer-mode mistral \
    --tensor-parallel-size 4 \
    --max-model-len 2048 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 2048 \
    --limit-mm-per-prompt '{"audio": 1}' \
    --additional-config '{
        "neuron_config": {
            "num_batched_tokens_buckets": [512, 2048],
            "on_device_sampling_config": {"all_greedy": true}
        },
        "encoder_neuron_config": {
            "num_encoder_input_buckets": [1, 4],
            "encoder_attention_block_size": 375,
            "tp_size": 4
        }
    }'
```

Use `temperature=0.0` for transcription, as the model card recommends.

### DFlash speculative decoding

[`jburtoft/Voxtral-Mini-3B-2507-DFlash`](https://huggingface.co/jburtoft/Voxtral-Mini-3B-2507-DFlash)
is a public DFlash draft for Voxtral-Mini-3B-2507 (5 layers, block size 16,
Apache-2.0). It is trained on Trainium2 with openly licensed speech data; the
training code is at [jimburtoft/voxtral-dflash](https://github.com/jimburtoft/voxtral-dflash). Add it
to the serving command:

```bash
--speculative-config '{"method": "dflash", "model": "jburtoft/Voxtral-Mini-3B-2507-DFlash", "num_speculative_tokens": 15}'
```

Set `num_speculative_tokens` to `block_size - 1` of the draft (15 for this draft).
Serve it at its trained block size: acceptance falls at smaller blocks.

A DFlash draft for Voxtral uses the upstream vLLM DFlash checkpoint format. The
plugin also reads the two optional features of that format: a reduced draft
vocabulary (`draft_vocab_size` and a `d2t` tensor) and a separate mask
embedding (`mask_embedding.pt`).

Measured on `trn2.3xlarge` with Neuron SDK 2.33 (the DLAMI vLLM environment; TP=4,
encoder TP=4, LNC=2, greedy, one request at a time, `/v1/audio/transcriptions`,
3 passes):

| Set | Without draft p50 | With draft p50 | Speedup | Acceptance length | WER without / with |
|---|--:|--:|--:|--:|---|
| LibriSpeech test-clean (24 clips) | 202.7 ms | 85.8 ms | 2.36x | 8.69 | 2.09% / 2.09% |
| LibriSpeech test-other (64 clips) | 180.4 ms | 89.1 ms | 2.02x | 6.63 | 4.32% / 4.41% |

The acceptance length counts the bonus token (1 + accepted draft tokens per
step). The test-other WER difference is one clip of 64 whose transcript differs
by bf16 rounding in the verify graph.

With several requests in flight the draft also raises throughput. On LibriSpeech
test-other (`--max-num-seqs 32`, `decode_context_length_buckets` `[512, 1024]`,
128 requests), the real-time factor at 32 concurrent requests is 96x without the
draft and 121x with it.

#### Long audio: skip the draft for long prompts

A draft trained on short clips accepts fewer tokens on long contexts, and for
long audio the verification can cost more than the draft saves. Set
`neuron_config.spec_decode_max_prompt_len` to decode longer prompts without
speculation. One 30 s window is 375 audio tokens, so `2500` turns the draft off
above about two minutes of audio:

```json
"neuron_config": {"spec_decode_max_prompt_len": 2500}
```

While a request above the limit is running, the other requests in the batch also
decode without speculation, because a decode batch must give every request the
same number of draft tokens. Requests that arrive after the long ones finish
speculate as usual. The option is off by default.

## Accuracy Evaluation

**Benchmark:** Earnings22 (ArtificialAnalysis/Earnings22-Cleaned-AA-chunked),
scored with the Open ASR Leaderboard normalizer through the
`/v1/audio/transcriptions` endpoint. The reference is the Open ASR Leaderboard
result for Voxtral-Mini-3B-2507: 10.69%.

Measured on `trn2.3xlarge` with Neuron SDK 2.33 (TP=4, encoder TP=4, greedy):
**7.80%** over 341 chunks.

With the public DFlash draft the WER is **7.77%** on the same 341 chunks, with an
acceptance length of 5.17. Check the acceptance length, not only the WER, after a
change to the draft path: a broken draft path keeps the target's transcripts but
drops the acceptance length to about 1.

## Common Issues

- **The encoder fails with a head-count error.** The encoder has 20 attention
  heads. Set `encoder_neuron_config.tp_size` to 1, 2, 4, 5, 10, or 20.
- **A long clip fails with an encoder bucket error.** One step can encode at
  most the largest `num_encoder_input_buckets` value of 30 s windows. Add a
  larger bucket.
- **The server fails with `encoder_cache_num_blocks=... is too small`.** Set
  `encoder_cache_num_blocks` as the configuration table shows.
- **Long audio needs more device memory headroom.** A clip of several minutes
  needs a larger `--max-model-len` (each 30 s window adds 375 tokens) and a
  larger largest encoder bucket. If warmup fails with `nrt_tensor_allocate
  status=4`, raise `VLLM_NEURON_RESERVED_HEADROOM_GIB` (8 worked for 5-minute
  clips with `--max-model-len 8192`).
