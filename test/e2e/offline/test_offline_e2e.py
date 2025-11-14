# SPDX-License-Identifier: Apache-2.0
"""
Consolidated e2e tests for offline inference.
"""

import logging
import os
from test.e2e.offline.offline_integration_test import (vllm_integ_test,
                                                       vllm_integ_test_llama4)
from typing import Any, Dict

import pytest
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

os.environ["VLLM_USE_V1"] = "1"

# Test configurations
BATCH_SIZE_CONFIGS = [
    {
        "title": "open_llama3b_batch1",
        "max_batch_size": 1,
    },
    {
        "title": "open_llama3b_batch4",
        "max_batch_size": 4,
    },
    {
        "title": "open_llama3b_batch8",
        "max_batch_size": 8,
    },
]

# Common parameters that don't change across tests
BASE_CONFIG = {
    "tp_degree": 16,
    "block_size": 32,
    "enable_prefix_caching": False,
    "n_positions": 256,
    "dtype": "bfloat16",
    "model_name_or_path": "openlm-research/open_llama_3b",
    "override_neuron_config": {
        "enable_bucketing": False,
    },
    "top_k": 1,
}

LLAMA4_CONFIG_TRN2 = {
    "batch_size": 4,
    "max_model_len": 4096,
    "tp_degree": 64,
    "block_size": 4096,
    "enable_prefix_caching": False,
    "dtype": "bfloat16",
    "override_neuron_config": {
        "text_neuron_config": {
            "blockwise_matmul_config": {
                "block_size": 256,
                "use_block_parallel": True,
                "block_sharding_strategy": "HI_LO",
                "skip_dma_token": True,
                "skip_dma_weight": True,
                "parallelize_token_to_block_mapping": True,
            },
            "sequence_parallel_enabled": True,
            "skip_warmup": True,
            "save_sharded_checkpoint": True,
            "fused_qkv": True,
            "cast_type": "as-declared",
            "cc_pipeline_tiling_factor": 1,
            "k_cache_transposed": False,
            "enable_bucketing": False,
        },
        "vision_neuron_config": {
            "fused_qkv": True,
            "save_sharded_checkpoint": True,
            "enable_bucketing": False,
        }
    },
}


@pytest.mark.llama_inference
@pytest.mark.vllm_integ_test
@pytest.mark.parametrize("config", BATCH_SIZE_CONFIGS)
def test_openllama3b_batch_variations(config: Dict[str, Any]):
    """
    Offline vLLM inference test against open_llama_3b with different batch sizes
    """
    test_config = {**BASE_CONFIG, **config}
    vllm_integ_test(**test_config)


def test_llama4():
    """
    Offline vLLM inference for Llama4 vision model.
    
    Uses meta-llama/Llama-4-Scout-17B-16E-Instruct from HuggingFace.
    """
    model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    logger.info(f"Testing config {LLAMA4_CONFIG_TRN2}")
    LLAMA4_CONFIG_TRN2.update({
        "title": "llama4-scout",
        "model_name_or_path": model_name
    })
    vllm_integ_test_llama4(**LLAMA4_CONFIG_TRN2)
    logger.info("Llama4 offline integ test passed!")


# Edge case tests


def test_min_tokens_trumps_eos():
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_num_seqs=4,
        max_model_len=2048,
        tensor_parallel_size=32,
        num_gpu_blocks_override=4096,
        block_size=32,
        enable_prefix_caching=True,
        additional_config=dict(override_neuron_config=dict(
            async_mode=False,
            is_prefix_caching=True,
            is_block_kv_layout=True,
            pa_num_blocks=4096,
            pa_block_size=32,
            skip_warmup=True,
        )),
    )

    prompts = [
        "explain the fractional banking thoroughly.",
        "explain all religions thoroughly.",
    ]

    min_tokens = 32
    max_tokens = 64
    sampling_params = SamplingParams(top_k=10,
                                     temperature=0.8,
                                     top_p=0.95,
                                     min_tokens=min_tokens,
                                     max_tokens=max_tokens)
    outputs = llm.generate(prompts, sampling_params)

    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        token_ids = output.outputs[0].token_ids
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print(f"Full request output: {output}")
        print("-" * 50)
        assert len(
            token_ids
        ) >= min_tokens, "Generated response is shorter than min_tokens"
        assert len(
            token_ids
        ) <= max_tokens, "Generated response is longer than max_tokens"
        assert 2 in token_ids, "Generated response for this prompt is known to include the eos_token [2]"


@pytest.mark.timeout(900)
def test_high_token_budget():
    model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    llm = LLM(
        model=model_path,
        tensor_parallel_size=32,
        max_model_len=4096,
        max_num_seqs=1,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )

    # Create a long prompt to test >4k token context support
    base_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models process vast amounts of data.",
        "Natural language understanding requires contextual awareness.",
        "Deep neural networks learn hierarchical representations.",
    ]

    # Build a long prompt by repeating varied sentences (~4k tokens)
    prompt = " ".join(base_sentences * 100)

    sampling_params = SamplingParams(top_k=10,
                                     temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=32,
                                     min_tokens=32)
    outputs = llm.generate([prompt], sampling_params)

    # Verify output
    print("\nGenerated Output:")
    print("=" * 50)
    generated_text = outputs[0].outputs[0].text
    print(f"Generated text: '{generated_text}'")
    print("-" * 50)
    assert len(generated_text) > 1


# test that OnDeviceSamplingConfig does not cause
# issues when reloading a compiled model
def test_double_on_device_sampling_config():
    from neuronx_distributed_inference.models.config import \
        OnDeviceSamplingConfig
    original_visible_cores = os.environ.get("NEURON_RT_VISIBLE_CORES", None)
    original_compiled_path = os.environ.get("NEURON_COMPILED_ARTIFACTS", None)
    try:
        os.environ["NEURON_RT_VISIBLE_CORES"] = "0-15"
        os.environ[
            "NEURON_COMPILED_ARTIFACTS"] = "/home/ubuntu/on_device_edge_case_compiled"
        llm = LLM(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_num_seqs=1,
            max_model_len=256,
            tensor_parallel_size=16,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            additional_config={
                "override_neuron_config":
                dict(on_device_sampling_config=OnDeviceSamplingConfig(), ),
            })

        prompts = [
            "explain the fractional banking thoroughly.",
        ]

        sampling_params = SamplingParams(top_k=10,
                                         temperature=0.8,
                                         top_p=0.95,
                                         max_tokens=32,
                                         min_tokens=32)
        outputs = llm.generate(prompts, sampling_params)

        # Print all generated outputs
        print("\nAll Generated Outputs:")
        print("=" * 50)
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\nPrompt {i+1}: '{prompt}'")
            print(f"Generated text: '{generated_text}'")
            print("-" * 50)
            assert len(generated_text) > 1

        os.environ["NEURON_RT_VISIBLE_CORES"] = "16-31"
        # Now run it again with the same model but this time it
        # will load from compiled artifacts
        llm2 = LLM(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_num_seqs=1,
            max_model_len=256,
            tensor_parallel_size=16,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            additional_config={
                "override_neuron_config":
                dict(on_device_sampling_config=OnDeviceSamplingConfig(), ),
            })
        sampling_params = SamplingParams(top_k=10,
                                         temperature=0.8,
                                         top_p=0.95,
                                         max_tokens=32,
                                         min_tokens=32)
        outputs = llm2.generate(prompts, sampling_params)

        # Print all generated outputs
        print("\nAll Generated Outputs:")
        print("=" * 50)
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\nPrompt {i+1}: '{prompt}'")
            print(f"Generated text: '{generated_text}'")
            print("-" * 50)
            assert len(generated_text) > 1
    except Exception as e:
        raise e
    finally:
        if original_visible_cores is not None:
            os.environ["NEURON_RT_VISIBLE_CORES"] = original_visible_cores
        else:
            del os.environ["NEURON_RT_VISIBLE_CORES"]
        if original_compiled_path is not None:
            os.environ["NEURON_COMPILED_ARTIFACTS"] = original_compiled_path
        else:
            del os.environ["NEURON_COMPILED_ARTIFACTS"]
