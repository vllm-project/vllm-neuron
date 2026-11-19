# SPDX-License-Identifier: Apache-2.0
"""Integration test for Qwen3.8 27B (text), served through vLLM on Neuron (Trn2).

Serves the real checkpoint with the configuration from the README (one Trainium2 chip, TP=8, logical NeuronCore
config 1) and checks a factual answer, a count that the GDN state must carry, greedy determinism, batch invariance,
a prompt that is prefilled in two chunks, and a tool call in the chat template's format.

Install the package so that the TP worker processes see the model too:

    pip install -e contrib/models/qwen38
    NEURON_LOGICAL_NC_CONFIG=1 NEURON_SKIP_EFA_AFFINITY=1 \
        pytest contrib/models/qwen38/test/integration/test_model.py -v -s
"""

import glob
import os

import pytest

MODEL_PATH = os.environ.get("QWEN38_MODEL_PATH", "Qwen/Qwen3.8-27B")

pytestmark = pytest.mark.skipif(not glob.glob("/dev/neuron*"), reason="needs Neuron devices (Trn2)")


@pytest.fixture(scope="module")
def llm():
    from vllm import LLM

    return LLM(
        model=MODEL_PATH,
        tensor_parallel_size=8,
        max_model_len=32768,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        block_size=2048,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=False,
        optimization_level=1,
        additional_config={
            "neuron_config": {
                "num_batched_tokens_buckets": [2048],
                "num_seqs_buckets": [1, 2, 4],
                "decode_context_length_buckets": [4096, 16384],
                "on_device_sampling_config": {},
            }
        },
    )


def _complete(llm, prompts, max_tokens=32):
    from vllm import SamplingParams

    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens))  # greedy
    return [o.outputs[0].text for o in outs]


def test_capital(llm):
    assert "paris" in _complete(llm, ["The capital of France is"])[0].lower()


def test_count_carries_state(llm):
    # a broken GDN state between decode steps shows up as a loop of a few tokens
    out = _complete(llm, ["Count from 1 to 20: 1, 2, 3,"], max_tokens=48)[0]
    assert "10, 11, 12" in out, out


def test_greedy_is_deterministic(llm):
    prompt = ["List three primary colors, comma separated:"]
    assert _complete(llm, prompt) == _complete(llm, prompt)


def test_batch_equals_single(llm):
    prompts = ["The capital of France is", "def fibonacci(n):\n    ", "Count from 1 to 20: 1, 2, 3,",
               "Water boils at"]
    alone = [_complete(llm, [p])[0] for p in prompts]
    assert _complete(llm, prompts) == alone


def test_two_chunk_prefill(llm):
    # about 3k tokens: two 2048-token prefill chunks, the second continuing the first chunk's GDN state
    filler = " ".join(["The quick brown fox jumps over the lazy dog."] * 300)
    prompt = f"{filler}\nThe secret word is BANANA. Repeat the secret word:"
    assert "banana" in _complete(llm, [prompt], max_tokens=16)[0].lower()


def test_tool_call_format(llm):
    from vllm import SamplingParams

    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    out = llm.chat([{"role": "user", "content": "What's the weather in Lisbon right now?"}],
                   SamplingParams(temperature=0.0, max_tokens=1024), tools=tools)
    text = out[0].outputs[0].text
    assert "<function=get_weather>" in text and "lisbon" in text.lower(), text
