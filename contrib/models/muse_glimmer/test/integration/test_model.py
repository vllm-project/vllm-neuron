# SPDX-License-Identifier: Apache-2.0
"""Integration test for Muse-Glimmer-30B, served through vLLM on Neuron (Trn3).

Serves the real checkpoint with the configuration from the README (TP=2,
logical NeuronCore config 2) and checks the answers to text, image, and video
questions whose answers are known.

Muse-Glimmer is registered through ``muse_glimmer_neuron.register()``. Install
the package so that the TP worker processes see the model too:

    pip install -e contrib/models/muse_glimmer
    NEURON_LOGICAL_NC_CONFIG=2 pytest contrib/models/muse_glimmer/test/integration/test_model.py \
        -v -s --timeout=3600

The first run compiles the text and vision graphs (about 15 minutes).
"""

import glob
import os

import pytest
from PIL import Image, ImageDraw

MODEL_PATH = os.environ.get("MUSE_GLIMMER_MODEL_PATH", "meta-models/Muse-Glimmer-30B")
CHAT_TEMPLATE_KWARGS = {"reasoning_strength": "low"}

pytestmark = pytest.mark.skipif(
    not glob.glob("/dev/neuron*"), reason="needs Neuron devices (Trn3)"
)


def _answer(text: str) -> str:
    """Drop the model's reasoning channel (to=self) and keep the final answer."""
    return text.rsplit("to=user", 1)[-1].strip().lower()


def _red_circles(n: int) -> Image.Image:
    img = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(img)
    for i in range(n):
        x = 60 + i * 130
        draw.ellipse((x, 170, x + 100, 270), fill="red")
    return img


@pytest.fixture(scope="module")
def llm():
    from vllm import LLM

    return LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        max_model_len=2048,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        num_gpu_blocks_override=512,
        limit_mm_per_prompt={"image": 1, "video": 1},
        media_io_kwargs={"video": {"num_frames": -1}},
        allowed_local_media_path="/",
        # vLLM Neuron 0.29.0.1.2.0: with prefix caching, a cache hit can leave only an
        # image's end marker to prefill, and the runner then stops on the encoder-cache
        # read (fixed in later vLLM Neuron releases). See the README.
        enable_prefix_caching=False,
        additional_config={
            "neuron_config": {
                "num_batched_tokens_buckets": [1024],
                "num_seqs_buckets": [1],
            },
            "encoder_neuron_config": {
                "encoder_attention_block_size": 1024,
                "num_encoder_input_buckets": [1024, 1564, 1632, 2080],
                # Cache blocks hold 256 tokens; items pad their last block (a 520-token
                # video takes 3), so give the cache room beyond the token budget.
                "encoder_cache_num_blocks": 16,
            },
        },
    )


def _ask(llm, content, max_tokens=256):
    from vllm import SamplingParams

    out = llm.chat(
        [[{"role": "user", "content": content}]],
        SamplingParams(temperature=0.0, max_tokens=max_tokens),
        chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
    )
    answer = _answer(out[0].outputs[0].text)
    print(
        f"Q: {content if isinstance(content, str) else content[-1]['text']!r} -> {answer!r}"
    )
    return answer


def test_text(llm):
    assert "paris" in _ask(llm, "What is the capital of France? Answer in one word.")


def test_image_count(llm):
    content = [
        {"type": "image_pil", "image_pil": _red_circles(3)},
        {
            "type": "text",
            "text": "How many red circles are in the image? Answer with a number.",
        },
    ]
    answer = _ask(llm, content)
    assert "3" in answer or "three" in answer


def test_image_color(llm):
    content = [
        {"type": "image_pil", "image_pil": Image.new("RGB", (640, 480), (0, 160, 0))},
        {"type": "text", "text": "What color is this image? Answer in one word."},
    ]
    assert "green" in _ask(llm, content)


def test_video(llm):
    from vllm.assets.video import VideoAsset

    path = os.path.abspath(VideoAsset("baby_reading").video_path)
    content = [
        {"type": "video_url", "video_url": {"url": f"file://{path}"}},
        {"type": "text", "text": "Who is in this video? Answer in a few words."},
    ]
    answer = _ask(llm, content)
    assert any(w in answer for w in ("baby", "child", "toddler", "kid", "infant"))
