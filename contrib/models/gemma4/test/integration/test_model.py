# SPDX-License-Identifier: Apache-2.0
"""Integration test for Gemma 4 31B (text), served through vLLM on Neuron (Trn2).

Serves the real checkpoint with the configuration from the README (one
Trainium2 chip, TP=4, logical NeuronCore config 2) and checks a factual answer,
arithmetic, greedy determinism, a long prompt and strict-JSON structured output.

Install the package so that the TP worker processes see the model too:

    pip install -e contrib/models/gemma4
    NEURON_LOGICAL_NC_CONFIG=2 NEURON_SKIP_EFA_AFFINITY=1 \
        pytest contrib/models/gemma4/test/integration/test_model.py -v -s --timeout=7200

Starting the engine compiles the graphs (5 to 8 minutes in our runs on a trn2.3xlarge, see the README).
"""

import glob
import json
import os

import pytest

MODEL_PATH = os.environ.get("GEMMA4_MODEL_PATH", "google/gemma-4-31B-it")

pytestmark = pytest.mark.skipif(
    not glob.glob("/dev/neuron*"), reason="needs Neuron devices (Trn2)"
)


@pytest.fixture(scope="module")
def llm():
    from vllm import LLM

    return LLM(
        model=MODEL_PATH,
        tensor_parallel_size=4,
        max_model_len=12288,
        max_num_seqs=4,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.70,
        enable_prefix_caching=True,
        async_scheduling=False,
        additional_config={
            "neuron_config": {
                "num_batched_tokens_buckets": [512],
                "num_seqs_buckets": [1, 4],
                "enable_structured_outputs": True,
                "on_device_sampling_config": {"all_greedy": True},
            }
        },
    )


def _chat(llm, prompt, max_tokens=64, **sp_kwargs):
    from vllm import SamplingParams

    out = llm.chat(
        [[{"role": "user", "content": prompt}]],
        SamplingParams(temperature=0.0, max_tokens=max_tokens, **sp_kwargs),
    )
    text = out[0].outputs[0].text.strip()
    print(f"Q: {prompt!r} -> {text!r}")
    return text


def test_capital(llm):
    assert "paris" in _chat(llm, "What is the capital of France? Answer in one word.").lower()


def test_arithmetic(llm):
    assert "56" in _chat(llm, "What is 7 times 8? Answer with the number only.")


def test_greedy_is_deterministic(llm):
    prompt = "List three primary colors, comma separated."
    assert _chat(llm, prompt) == _chat(llm, prompt)


def test_long_prompt_segmented_prefill(llm):
    # ~3k tokens: several 512-token prefill segments, crossing the 1024 sliding window.
    filler = " ".join(f"item{i}" for i in range(1500))
    prompt = f"Here is a list: {filler}\nThe secret word is BANANA. What is the secret word?"
    assert "banana" in _chat(llm, prompt, max_tokens=16).lower()


def test_structured_json(llm):
    from vllm.sampling_params import StructuredOutputsParams

    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
            "population_millions": {"type": "number"},
        },
        "required": ["city", "country", "population_millions"],
        "additionalProperties": False,
    }
    text = _chat(
        llm,
        "Give the city, country and population in millions of the capital of Japan as JSON.",
        max_tokens=128,
        structured_outputs=StructuredOutputsParams(json=schema),
    )
    data = json.loads(text)
    assert set(data) == {"city", "country", "population_millions"}
    assert "tokyo" in data["city"].lower()
