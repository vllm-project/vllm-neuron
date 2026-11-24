# SPDX-License-Identifier: Apache-2.0
"""Integration test for Tencent Hy4-preview, served through vLLM on one trn2.48xlarge.

Logit validation against a transformers reference (the template's method) is out
of reach here: the reference would have to hold the 770B model in fp32 on the host.
This test checks greedy generations instead, against outputs recorded on device
(``README.md``, "Measured"), plus the ~30K-token pass-key retrieval:

* the four demo prompts must start with the recorded continuation (its first
  ~16 tokens; later tokens may drift, since bf16 near-ties between experts make
  the tail compiler-version dependent);
* the pass key hidden at the start of ~30K tokens of filler must come back.

The model is registered by this folder's ``vllm.general_plugins`` entry point, so
install it first (it has to be importable in every worker, not just this process)::

    pip install -e contrib/models/hy4_preview
    HY4_MODEL=<dir>/Hy4-preview-neuron-fp8 VLLM_NEURON_BACKEND=vllm_neuron \\
        pytest contrib/models/hy4_preview/test/integration/test_model.py -v -s --timeout=7200

A cold run compiles the 32K prefill graph and two decode graphs (~30 min).
"""

import os

import pytest

# The converted checkpoint (see README.md); there is no Hub copy of it.
MODEL_PATH = os.environ.get("HY4_MODEL")
if not MODEL_PATH:
    pytest.skip(f"set HY4_MODEL to the converted checkpoint (Hy4-preview-neuron-fp8)", allow_module_level=True)

# Recorded on trn2.48xlarge, Neuron SDK 2.33, vllm-neuron 0.29.0.1.2.0 (and
# identical on SDK 2.27 / vllm-neuron 0.24).
EXPECTED = {
    "The capital of France is": " __________.\nA. Sydney\nB. New York\nC. Paris\nD. Ottawa",
    "I am gonna keep counting forever, 1 2 3 4 5": " 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21",
    "def fibonacci(n):": '\n    """Return the nth value in the Fibonacci series."""\n    if n == 0:\n        return 0',
    "Once upon a time, there was a": " king who was very rich. He had a lot of gold and silver, but he was not happy.",
}

PASS_KEY = "71432"
NEEDLE_TOKENS = 30000
FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. "
    "There and back again. "
)


def needle_prompt(tokenizer, approx_tokens: int) -> str:
    """The pass key first, then filler, then the question (same as scripts/run.py)."""
    head = (
        "There is an important piece of information hidden inside a lot of "
        "irrelevant text. Find it and memorize it.\n"
        f"The pass key is {PASS_KEY}. Remember it. {PASS_KEY} is the pass key.\n"
    )
    tail = "\nWhat is the pass key? The pass key is"
    per_filler = len(tokenizer.encode(FILLER, add_special_tokens=False))
    budget = approx_tokens - len(
        tokenizer.encode(head + tail, add_special_tokens=False)
    )
    return head + FILLER * max(0, budget // per_filler) + tail


@pytest.fixture(scope="module")
def llm():
    from vllm import LLM

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=64,
        max_model_len=32768,
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
        # Rank 0 compiles while the others wait in a gloo barrier (default 1 h).
        distributed_timeout_seconds=4 * 3600,
        cpu_distributed_timeout_seconds=4 * 3600,
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "num_batched_tokens_buckets": [2048],
                "num_seqs_buckets": [1, 4],
                "on_device_sampling_config": {"all_greedy": True},
            }
        },
    )
    yield llm
    llm.llm_engine.engine_core.shutdown()


@pytest.mark.integration
def test_demo_prompts(llm) -> None:
    from vllm import SamplingParams

    prompts = list(EXPECTED)
    outputs = llm.generate(prompts, SamplingParams(max_tokens=32, temperature=0.0))
    for prompt, output in zip(prompts, outputs):
        text = output.outputs[0].text
        print(f"{prompt!r} -> {text!r}")
        assert text.startswith(EXPECTED[prompt]), (
            f"{prompt!r}: expected {EXPECTED[prompt]!r}..."
        )


@pytest.mark.integration
def test_needle_30k(llm) -> None:
    from vllm import SamplingParams

    prompt = needle_prompt(llm.get_tokenizer(), NEEDLE_TOKENS)
    (output,) = llm.generate([prompt], SamplingParams(max_tokens=16, temperature=0.0))
    text = output.outputs[0].text
    print(f"{len(output.prompt_token_ids)} tokens -> {text!r}")
    assert len(output.prompt_token_ids) > 28000
    assert PASS_KEY in text
