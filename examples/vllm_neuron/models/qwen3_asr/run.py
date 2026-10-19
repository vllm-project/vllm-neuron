# SPDX-License-Identifier: Apache-2.0
"""Offline speech-to-text example for Qwen3-ASR-1.7B on Neuron.

Transcribes an audio clip with the vLLM offline API. The example builds a
chat prompt that has one audio placeholder and one text instruction, then
passes the waveform through ``multi_modal_data``. It also shows a text-only
mode for a quick language-model smoke run.

Audio mode needs an audio backend to decode the clip. Install it with
``pip install vllm[audio]`` (it adds soundfile and PyAV). Text-only mode does
not need this backend.

Usage:
    # Transcribe a bundled sample clip
    python examples/vllm_neuron/models/qwen3_asr/run.py \
        --model-checkpoint <path-to-checkpoint>/Qwen_Qwen3-ASR-1.7B

    # Transcribe your own clip (any format that soundfile reads, e.g. wav/flac/ogg)
    python examples/vllm_neuron/models/qwen3_asr/run.py \
        --model-checkpoint <path-to-checkpoint>/Qwen_Qwen3-ASR-1.7B \
        --audio-file /path/to/speech.wav

    # Text-only (no audio passed)
    python examples/vllm_neuron/models/qwen3_asr/run.py \
        --model-checkpoint <path-to-checkpoint>/Qwen_Qwen3-ASR-1.7B \
        --text-only
"""

import argparse
import os

os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = "1200"
os.environ["NEURON_LIBTORCH_COMPILATION_TIMEOUT"] = "1200"

from transformers import AutoProcessor

from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.multimodal.media.audio import load_audio

# The transcription instruction. Qwen3-ASR emits the transcript of the audio.
ASR_INSTRUCTION = "Transcribe the audio."

# The default sample clip. It is one of the public vLLM audio assets and is a
# short speech recording.
DEFAULT_AUDIO_ASSET = "mary_had_lamb"

# The rate the audio encoder expects, in Hz.
SAMPLING_RATE = 16000

# At block size 128 one block holds one inference window, and
# one window is about 8 s of audio. So 8 blocks give an encoder budget of about
# 64 s. The default clip is about 16 s, which uses 2 blocks.
AUDIO_BLOCK_BUCKETS = [8]
AUDIO_BLOCK_SIZE = 128


def create_audio_input(processor: AutoProcessor, audio: tuple) -> dict:
    """Build one audio + text prompt.

    ``audio`` is a ``(waveform, sampling_rate)`` tuple. vLLM resamples the
    waveform to the rate the model needs, so any input rate is accepted.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "text", "text": ASR_INSTRUCTION},
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt, "multi_modal_data": {"audio": [audio]}}


def create_text_only_inputs() -> list[str]:
    """Plain text prompts (no audio passed at inference time)."""
    return [
        "I am gonna keep counting forever, 1 2 3 4 5 ",
        "The capital of France is ",
        "Once upon a time, there was a ",
        "def fibonacci(n):",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="Qwen/Qwen3-ASR-1.7B",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Path to an audio clip to transcribe (defaults to a sample clip)",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Run text-only prompts (skip audio)",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=2,
        help="Tensor parallel size",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    # Text-only skips the audio encoder budget, so drop the dummy audio item.
    limit_mm_per_prompt = {"audio": 0} if args.text_only else {"audio": 1}

    llm = LLM(
        model=args.model_checkpoint,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        max_num_seqs=4,
        tensor_parallel_size=args.tp_size,
        enable_prefix_caching=False,
        limit_mm_per_prompt=limit_mm_per_prompt,
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "num_batched_tokens_buckets": [4096],
                "num_seqs_buckets": [4],
                "on_device_sampling_config": {"all_greedy": True},
            },
            "encoder_neuron_config": {
                "num_encoder_input_buckets": AUDIO_BLOCK_BUCKETS,
                "encoder_attention_block_size": AUDIO_BLOCK_SIZE,
            },
        },
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens, temperature=0.0, top_p=1.0
    )

    if args.text_only:
        prompts = create_text_only_inputs()
        outputs = llm.generate(prompts, sampling_params)
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt!r}")
            print(f"Generated: {output.outputs[0].text!r}")
            print()
        return

    if args.audio_file:
        # Pass sr=SAMPLING_RATE to force a normalized float waveform. vLLM
        # resamples the rate itself, so 16 kHz is not a requirement here. The
        # reason is normalization. With sr=None, load_audio keeps the native
        # rate. Its PyAV fallback (used when soundfile is absent) then returns
        # un-normalized integer PCM, and the encoder finds no speech in it. A
        # target rate makes load_audio resample, which also normalizes the audio.
        audio = load_audio(args.audio_file, sr=SAMPLING_RATE)
        print(f"[qwen3-asr] transcribing: {args.audio_file}")
    else:
        audio = AudioAsset(DEFAULT_AUDIO_ASSET).audio_and_sample_rate
        print(f"[qwen3-asr] transcribing sample clip: {DEFAULT_AUDIO_ASSET}")

    processor = AutoProcessor.from_pretrained(
        args.model_checkpoint, trust_remote_code=True
    )
    outputs = llm.generate([create_audio_input(processor, audio)], sampling_params)
    print(f"Instruction: {ASR_INSTRUCTION!r}")
    print(f"Transcript: {outputs[0].outputs[0].text!r}")
    print()


if __name__ == "__main__":
    main()
