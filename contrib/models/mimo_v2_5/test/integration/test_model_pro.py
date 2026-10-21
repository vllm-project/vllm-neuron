# SPDX-License-Identifier: Apache-2.0
"""Greedy smoke test for MiMo-V2.5-Pro, served through vLLM on Neuron.

``test_model.py`` validates V2.5's logits against a HuggingFace reference. That
is not feasible for Pro: an FP32 golden pass over its ~1.02T parameters needs
~4 TB of host RAM, twice what a trn2.48xlarge has. So this test checks what a
broken port gets wrong first instead:

* **coherence**: greedy completions of factual prompts contain the expected
  answer (``"The capital of France is"`` -> ``"Paris"``), and no completion is
  empty or a single token repeated;
* **determinism**: two greedy passes over the same prompts give identical
  token ids.

It runs at TP=64 / EP=64 on one trn2.48xlarge with the routed experts in fp8
(``MIMO_V2_5_EXPERT_DTYPE=auto`` resolves to fp8 for Pro; see the README), and
is skipped unless ``MIMO_V2_5_PRO_CHECKPOINT`` points at the checkpoint:

    MIMO_V2_5_PRO_CHECKPOINT=/path/to/MiMo-V2.5-Pro \\
    pytest contrib/models/mimo_v2_5/test/integration/test_model_pro.py \\
        -v -s --timeout=10800

The model is registered the same way ``test_model.py`` does it (Option 1 in
``contrib/README.md``).
"""

import os

import pytest
from test_model import _text_only_overrides, register_mimo_v2_5

MODEL_PATH = os.environ.get("MIMO_V2_5_PRO_CHECKPOINT")

TENSOR_PARALLEL_SIZE = 64
EP_DEGREE = 64
MAX_TOKENS = 32

# (prompt, substring the greedy completion must contain, case-insensitive).
# Question/answer form so the answer comes first: as plain continuations
# ("The capital of France is") the models describe the city for 24+ tokens
# before naming it (V2.5-Pro on Jupiter, V2.6-Pro-RL on Paris and Tokyo).
PROMPTS = [
    ("Question: What is the capital of France?\nAnswer:", "paris"),
    ("Question: What is the capital of Japan?\nAnswer:", "tokyo"),
    ("Question: Water is made of hydrogen and which other element?\nAnswer:", "oxygen"),
]

pytestmark = pytest.mark.skipif(
    not MODEL_PATH,
    reason="set MIMO_V2_5_PRO_CHECKPOINT to a MiMo-V2.5-Pro checkpoint to run",
)


def _degenerate(token_ids: list[int]) -> bool:
    return len(token_ids) == 0 or len(set(token_ids)) == 1


@pytest.mark.integration
@pytest.mark.parametrize("batch_size, seq_len", [(1, 1024)])
def test_model_pro_greedy_smoke(batch_size: int, seq_len: int) -> None:
    register_mimo_v2_5()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        enable_expert_parallel=True,
        trust_remote_code=True,
        hf_overrides=_text_only_overrides,
        max_model_len=seq_len,
        max_num_batched_tokens=seq_len,
        max_num_seqs=batch_size,
        enable_prefix_caching=False,
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "ep_degree": EP_DEGREE,
                "on_device_sampling_config": {"all_greedy": True},
                "num_batched_tokens_buckets": [seq_len],
                "num_seqs_buckets": [batch_size],
            }
        },
    )
    try:
        params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS)
        prompts = [p for p, _ in PROMPTS]
        first = llm.generate(prompts, params)
        second = llm.generate(prompts, params)

        failures = []
        for (prompt, expected), a, b in zip(PROMPTS, first, second):
            text = a.outputs[0].text
            ids_a = list(a.outputs[0].token_ids)
            ids_b = list(b.outputs[0].token_ids)
            print(f"{prompt!r} -> {text!r}")
            if expected not in text.lower():
                failures.append(f"{prompt!r}: {expected!r} not in {text!r}")
            if _degenerate(ids_a):
                failures.append(f"{prompt!r}: degenerate output {ids_a}")
            if ids_a != ids_b:
                failures.append(
                    f"{prompt!r}: greedy runs differ:\n  {ids_a}\n  {ids_b}"
                )
        assert not failures, "\n".join(failures)
    finally:
        llm.llm_engine.engine_core.shutdown()
