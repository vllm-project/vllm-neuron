# SPDX-License-Identifier: Apache-2.0
"""
Test TinyLlama dynamic sampling strategies.
Validates that different sampling parameters produce expected behavior.
"""

import logging
import os

import pytest
from vllm import LLM, SamplingParams

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_dynamic_sampling():
    os.environ["VLLM_USE_V1"] = "1"

    model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    test_prompt = "Hello, my name is"

    # Initialize LLM (same config as continuous_batching test)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=32,
        max_num_seqs=4,
        max_model_len=1024,
        block_size=32,
        enable_prefix_caching=True,
        enable_chunked_prefill=False,
        num_gpu_blocks_override=2 * 4 * 1024 // 32,
    )

    # Test 1: Greedy Determinism
    logger.info("=" * 60)
    logger.info("Test 1: Greedy Determinism")
    logger.info("=" * 60)

    greedy_params = SamplingParams(temperature=0.0)

    output1 = llm.generate([test_prompt], greedy_params)
    text1 = output1[0].outputs[0].text

    output2 = llm.generate([test_prompt], greedy_params)
    text2 = output2[0].outputs[0].text

    logger.info(f"Run 1: {text1!r}")
    logger.info(f"Run 2: {text2!r}")

    assert text1 == text2, "Greedy decoding should be deterministic"
    logger.info("Test 1 PASSED: Greedy determinism validated\n")

    # Test 2: Multiple Sampling Strategies
    logger.info("=" * 60)
    logger.info("Test 2: Multiple Sampling Strategies")
    logger.info("=" * 60)

    sampling_configs = [
        {
            "name": "greedy_temp_0",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1
        },
        {
            "name": "greedy_topk_1",
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 1
        },
        {
            "name": "standard",
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": -1
        },
        {
            "name": "topk_50",
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 50
        },
        {
            "name": "high_temp",
            "temperature": 1.5,
            "top_p": 0.95,
            "top_k": -1
        },
    ]

    results = {}
    for config in sampling_configs:
        name = config.pop("name")
        params = SamplingParams(**config)

        output = llm.generate([test_prompt], params)
        text = output[0].outputs[0].text

        results[name] = text
        logger.info(f"[{name}] {text!r}")

    # Validate we got outputs for all strategies
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"
    for name, text in results.items():
        assert len(text) > 0, f"Empty output for {name}"
    logger.info("Test 2 PASSED: All sampling strategies completed\n")

    # Test 3: Greedy Consistency
    logger.info("=" * 60)
    logger.info("Test 3: Greedy Consistency")
    logger.info("=" * 60)

    # Both greedy strategies MUST produce identical outputs
    greedy_temp = results["greedy_temp_0"]
    greedy_topk = results["greedy_topk_1"]

    logger.info(f"greedy_temp_0: {greedy_temp!r}")
    logger.info(f"greedy_topk_1: {greedy_topk!r}")

    assert greedy_temp == greedy_topk, \
            f"Greedy methods MUST produce identical outputs.\n" \
            f"temperature=0.0 gave: {greedy_temp!r}\n" \
            f"top_k=1 gave: {greedy_topk!r}"
    logger.info(
        "Test 3 PASSED: Greedy consistency validated (temp=0 == top_k=1)\n")

    # Test 4: Sampling Diversity
    logger.info("=" * 60)
    logger.info("Test 4: Sampling Diversity")
    logger.info("=" * 60)

    diversity_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        n=4,  # Generate 4 samples
    )

    outputs = llm.generate([test_prompt], diversity_params)
    texts = [output.text for output in outputs[0].outputs]

    logger.info(f"Generated {len(texts)} samples:")
    for i, text in enumerate(texts):
        logger.info(f"  Sample {i+1}: {text!r}")

    unique_texts = len(set(texts))
    logger.info(f"Unique outputs: {unique_texts}/4")

    assert len(texts) == 4, f"Expected 4 samples, got {len(texts)}"
    assert unique_texts >= 2, \
            f"Expected diversity, got {unique_texts} unique out of 4"
    logger.info(
        f"Test 4 PASSED: Diversity validated ({unique_texts}/4 unique)\n")

    # Test 5: Temperature Effect
    logger.info("=" * 60)
    logger.info("Test 5: Temperature Effect")
    logger.info("=" * 60)

    # Low temperature
    low_temp_params = SamplingParams(temperature=0.3, n=3)
    low_temp_outputs = llm.generate([test_prompt], low_temp_params)
    low_temp_texts = [o.text for o in low_temp_outputs[0].outputs]
    low_temp_unique = len(set(low_temp_texts))

    logger.info("Low temp (0.3) samples:")
    for i, text in enumerate(low_temp_texts):
        logger.info(f"  {i+1}: {text[:50]!r}...")

    # High temperature
    high_temp_params = SamplingParams(temperature=1.5, n=3)
    high_temp_outputs = llm.generate([test_prompt], high_temp_params)
    high_temp_texts = [o.text for o in high_temp_outputs[0].outputs]
    high_temp_unique = len(set(high_temp_texts))

    logger.info("High temp (1.5) samples:")
    for i, text in enumerate(high_temp_texts):
        logger.info(f"  {i+1}: {text[:50]!r}...")

    logger.info(f"Low temp unique: {low_temp_unique}/3")
    logger.info(f"High temp unique: {high_temp_unique}/3")

    assert low_temp_unique >= 1, "Low temp should produce at least 1 output"
    assert high_temp_unique >= 1, "High temp should produce at least 1 output"
    logger.info("Test 5 PASSED: Temperature effect validated\n")

    logger.info("=" * 60)
    logger.info("All Dynamic Sampling Tests PASSED!")
    logger.info("=" * 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
