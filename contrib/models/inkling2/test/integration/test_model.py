# SPDX-License-Identifier: Apache-2.0
"""Integration test for Inkling 2 Mini (text), served through vLLM on Neuron (one trn2.3xlarge, TP=8).

Serves the checkpoint view from ``examples/prepare_checkpoint.sh`` with the README's configuration and checks a
factual answer, greedy determinism, batch invariance, and two prompts whose greedy continuations were checked
against Hugging Face (teacher-forced, every step agreeing): one inside the 1024-token sliding window and one past it.

    pip install -e contrib/models/inkling2
    contrib/models/inkling2/examples/prepare_checkpoint.sh /path/to/Inkling-2-Mini /path/to/Inkling-2-Mini-bf16
    INKLING2_MODEL_PATH=/path/to/Inkling-2-Mini-bf16 NEURON_LOGICAL_NC_CONFIG=1 NEURON_SKIP_EFA_AFFINITY=1 \
        pytest contrib/models/inkling2/test/integration/test_model.py -v -s
"""

import glob
import os

import pytest

MODEL_PATH = os.environ.get("INKLING2_MODEL_PATH", "/models/Inkling-2-Mini-bf16")

pytestmark = pytest.mark.skipif(not glob.glob("/dev/neuron*"), reason="needs Neuron devices (Trn2)")

BASE = ("The history of computing is a story of abstraction. Each generation of engineers built tools that hid the "
        "details of the layer below, so that the next generation could think about larger problems. ")


@pytest.fixture(scope="module")
def llm():
    from vllm import LLM

    return LLM(
        model=MODEL_PATH,
        tensor_parallel_size=8,
        max_model_len=16384,
        max_num_seqs=2,
        max_num_batched_tokens=256,
        block_size=256,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=False,
        optimization_level=3,
        additional_config={"neuron_config": {"num_batched_tokens_buckets": [256], "num_seqs_buckets": [2],
                                             "on_device_sampling_config": {"all_greedy": True}}},
    )


def greedy(llm, prompts, n=24):
    from vllm import SamplingParams

    out = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True))
    return [o.outputs[0].text for o in out]


def test_factual(llm):
    assert greedy(llm, ["The capital of France is"], 4)[0].strip().startswith("Paris")


def test_greedy_is_deterministic(llm):
    p = [BASE * 6 + "In summary,"]
    assert greedy(llm, p) == greedy(llm, p)


def test_batch_invariance(llm):
    a, b = BASE * 6 + "In summary,", BASE * 20 + "To conclude,"
    alone = [greedy(llm, [a])[0], greedy(llm, [b])[0]]
    assert greedy(llm, [a, b]) == alone


def test_inside_the_window(llm):
    # 759 tokens; HF's greedy continuation (teacher-forced 24/24 on this server)
    out = greedy(llm, [BASE * 21 + "In summary,"])[0]
    assert out.startswith(" the history of computing is a story of abstraction")


def test_past_the_window(llm):
    # 4863 tokens, past the 1024-token window of 20 of the 24 layers; HF teacher-forced 24/24 on this server
    out = greedy(llm, [BASE * 135 + "To conclude,"])[0]
    assert out.startswith(" the history of computing is a story of abstraction")
